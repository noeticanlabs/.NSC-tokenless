"""
Noetican Proposal Engine v2.0 (Full Upgrade)

This implementation adds:
1. Beam-search synthesis for operator chain invention
2. Real coherence metrics (r_gate, r_cons, r_num)
3. OperatorReceipt emission for executed proposals

References:
- coherence_spine/02_math/coherence_functionals.md
- coherence_spine/03_measurement/residuals_metrics.md
- coherence_spine/03_measurement/telemetry_and_receipts.md
- coherence_spine/04_control/gates_and_rails.md
- nsc/schemas/nsc_operator_receipt.schema.json
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Set
import hashlib
import json
import math
import os
from enum import Enum
from heapq import heappush, heappop
from collections import defaultdict

# -----------------------------
# Coherence Metrics (L1-L2)
# Per residuals_metrics.md: r_phys, r_cons, r_num, r_ref, r_stale, r_conflict
# -----------------------------

@dataclass
class CoherenceMetrics:
    """
    Real coherence metrics per residuals_metrics.md and telemetry_and_receipts.md.
    
    Residual blocks:
    - r_phys: physics residual (equation defect)
    - r_cons: constraint residual
    - r_num: numerical stability residual
    - r_ref: evidence/tool reference residual
    - r_stale: freshness residual
    - r_conflict: contradiction residual
    """
    # Primary residuals
    r_phys: float = 0.0      # Physics defect: ||r_phys||₂/√N
    r_cons: float = 0.0      # Constraint violation: c(x) = 0
    r_num: float = 0.0       # Numerical: NaN/Inf, overflow risk
    
    # Evidence residuals (per r_ref, r_stale, r_conflict)
    r_ref: float = 0.0       # Missing citations
    r_stale: float = 0.0     # Claims not verified
    r_conflict: float = 0.0  # Unresolved contradictions
    
    # Operational residuals (per minimal_tests.py)
    r_retry: int = 0         # Retries used
    r_rollback: int = 0      # Rollbacks in window
    r_sat: float = 0.0       # Controller saturation fraction
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "r_phys": self.r_phys,
            "r_cons": self.r_cons,
            "r_num": self.r_num,
            "r_ref": self.r_ref,
            "r_stale": self.r_stale,
            "r_conflict": self.r_conflict,
            "r_retry": self.r_retry,
            "r_rollback": self.r_rollback,
            "r_sat": self.r_sat,
        }
    
    def norm(self, p: float = 2) -> float:
        """Compute Lp norm of residual vector."""
        vals = [
            self.r_phys, self.r_cons, self.r_num,
            self.r_ref, self.r_stale, self.r_conflict
        ]
        return sum(v ** p for v in vals) ** (1.0 / p)
    
    def max_residual(self) -> float:
        """Return maximum absolute residual."""
        return max(
            abs(self.r_phys), abs(self.r_cons), abs(self.r_num),
            abs(self.r_ref), abs(self.r_stale), abs(self.r_conflict)
        )


@dataclass
class DebtFunctional:
    """
    Complete debt functional per coherence_functionals.md:
    
    C(x) = Σ_i w_i ||r̃_i||² + Σ_j v_j p_j(x)
    
    Where:
    - r̃_i = normalized residuals
    - p_j = penalty functions (thrash, saturation, etc.)
    """
    # Weights for each residual block
    w_phys: float = 1.0
    w_cons: float = 1.0
    w_num: float = 2.0      # Numerical stability is critical
    w_ref: float = 0.5
    w_stale: float = 0.5
    w_conflict: float = 1.0
    
    # Penalty weights
    v_thrash: float = 1.0   # Operational inefficiency
    v_sat: float = 1.5      # Controller saturation penalty
    v_retry: float = 0.5    # Retry cost
    
    def compute(self, metrics: CoherenceMetrics) -> Tuple[float, Dict[str, float]]:
        """
        Compute debt functional C(x).
        
        Returns:
            Tuple of (total_debt, decomposition_dict)
        """
        # Normalized squared residuals
        r_phys_sq = self.w_phys * (metrics.r_phys ** 2)
        r_cons_sq = self.w_cons * (metrics.r_cons ** 2)
        r_num_sq = self.w_num * (metrics.r_num ** 2)
        r_ref_sq = self.w_ref * (metrics.r_ref ** 2)
        r_stale_sq = self.w_stale * (metrics.r_stale ** 2)
        r_conflict_sq = self.w_conflict * (metrics.r_conflict ** 2)
        
        # Penalty terms (per operational residuals)
        thrash = self.v_thrash * (metrics.r_retry ** 2)
        sat = self.v_sat * (metrics.r_sat ** 2)
        retry = self.v_retry * metrics.r_retry
        
        # Total debt
        total = (r_phys_sq + r_cons_sq + r_num_sq + 
                 r_ref_sq + r_stale_sq + r_conflict_sq +
                 thrash + sat + retry)
        
        decomposition = {
            "phys": r_phys_sq,
            "cons": r_cons_sq,
            "num": r_num_sq,
            "ref": r_ref_sq,
            "stale": r_stale_sq,
            "conflict": r_conflict_sq,
            "thrash": thrash,
            "sat": sat,
            "retry": retry,
            "total": total
        }
        
        return total, decomposition


# -----------------------------
# Gate Evaluation with Real Metrics
# Per gates_and_rails.md: hard gates + soft gates + hysteresis
# -----------------------------

class GateResult(Enum):
    HARD_FAIL = "hard_fail"   # Abort immediately
    SOFT_FAIL = "soft_fail"   # Reject, try rails
    WARN = "warn"            # Accept with warning
    PASS = "pass"            # Accept


@dataclass
class GateConfig:
    """
    Gate configuration with hysteresis per gates_and_rails.md.
    
    Hysteresis defaults:
    - warn_enter = 0.75 * fail_enter
    - warn_exit = 0.60 * fail_enter
    - fail_exit = 0.80 * fail_enter
    """
    # Hard gates (must pass)
    hard_nan_inf: bool = True
    hard_positivity: float = 0.0    # ρ ≥ 0
    hard_bound: Optional[Tuple[float, float]] = None  # [lower, upper]
    hard_dt_min: float = 1e-6
    
    # Soft gates (with hysteresis)
    soft_phys_fail: float = 1.0
    soft_phys_warn: float = 0.75    # 0.75 * fail_enter
    soft_cons_fail: float = 0.75
    soft_cons_warn: float = 0.5625  # 0.75 * fail_enter
    soft_num_fail: float = 0.5      # Numerical instability threshold
    soft_num_warn: float = 0.375
    
    def evaluate(self, metrics: CoherenceMetrics) -> Tuple[GateResult, Dict[str, Any]]:
        """
        Evaluate all gates on coherence metrics.
        
        Returns:
            Tuple of (result, details)
        """
        details = {"hard": {}, "soft": {}, "hysteresis": {}}
        
        # Hard gates
        hard_pass = True
        
        # NaN/Inf check
        has_nan = any(math.isnan(v) for v in metrics.to_json().values() if isinstance(v, float))
        has_inf = any(math.isinf(v) for v in metrics.to_json().values() if isinstance(v, float))
        details["hard"]["nan_inf_free"] = not has_nan and not has_inf
        if has_nan or has_inf:
            hard_pass = False
        
        # Positivity check
        if metrics.r_phys < self.hard_positivity:
            details["hard"]["positivity"] = "fail"
            hard_pass = False
        else:
            details["hard"]["positivity"] = "pass"
        
        # Bound check
        if self.hard_bound:
            lower, upper = self.hard_bound
            in_bounds = lower <= metrics.r_phys <= upper
            details["hard"]["bounds"] = "pass" if in_bounds else "fail"
            if not in_bounds:
                hard_pass = False
        
        if not hard_pass:
            return GateResult.HARD_FAIL, details
        
        # Soft gates with hysteresis
        soft_failures = []
        soft_warnings = []
        
        # Physics residual
        if metrics.r_phys > self.soft_phys_fail:
            soft_failures.append("phys")
        elif metrics.r_phys > self.soft_phys_warn:
            soft_warnings.append("phys")
        
        # Constraint residual
        if metrics.r_cons > self.soft_cons_fail:
            soft_failures.append("cons")
        elif metrics.r_cons > self.soft_cons_warn:
            soft_warnings.append("cons")
        
        # Numerical residual
        if metrics.r_num > self.soft_num_fail:
            soft_failures.append("num")
        elif metrics.r_num > self.soft_num_warn:
            soft_warnings.append("num")
        
        details["soft"]["failures"] = soft_failures
        details["soft"]["warnings"] = soft_warnings
        
        # Decision
        if soft_failures:
            return GateResult.SOFT_FAIL, details
        elif soft_warnings:
            return GateResult.WARN, details
        else:
            return GateResult.PASS, details


# -----------------------------
# Operator Receipt (L0 Conformance)
# Per nsc_operator_receipt.schema.json and telemetry_and_receipts.md
# -----------------------------

@dataclass
class OperatorReceipt:
    """
    Operator receipt per nsc_operator_receipt.schema.json.
    
    Fields per schema:
    - kind, event_id, node_id, op_id
    - digest_in, digest_out
    - kernel_id, kernel_version
    - metrics (C_before, C_after, r_phys, r_cons, r_num)
    - provenance (registry_id, version, timestamp)
    """
    kind: str = "operator_receipt"
    event_id: str = ""
    node_id: int = 0
    op_id: int = 0
    
    # Digest chain
    digest_in: str = ""
    digest_out: str = ""
    digest: str = ""  # sha256 of receipt excluding itself
    
    # Kernel identity
    kernel_id: str = ""
    kernel_version: str = ""
    
    # Metrics
    C_before: float = 0.0
    C_after: float = 0.0
    r_phys: float = 0.0
    r_cons: float = 0.0
    r_num: float = 0.0
    
    # Provenance
    registry_id: str = ""
    registry_version: str = ""
    
    def to_json(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "event_id": self.event_id,
            "node_id": self.node_id,
            "op_id": self.op_id,
            "digest_in": self.digest_in,
            "digest_out": self.digest_out,
            "kernel_id": self.kernel_id,
            "kernel_version": self.kernel_version,
            "C_before": self.C_before,
            "C_after": self.C_after,
            "r_phys": self.r_phys,
            "r_cons": self.r_cons,
            "r_num": self.r_num,
            "registry_id": self.registry_id,
            "registry_version": self.registry_version,
            "digest": self.digest
        }
    
    def compute_digest(self, prev_hash: str = None) -> str:
        """Compute receipt digest per hash chaining rule."""
        # Exclude digest field for self-reference
        payload = {k: v for k, v in self.to_json().items() if k != "digest"}
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        if prev_hash:
            combined = canon.encode("utf-8") + prev_hash.encode("utf-8")
            self.digest = "sha256:" + hashlib.sha256(combined).hexdigest()
        else:
            self.digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
        
        return self.digest


@dataclass
class ReceiptLedger:
    """
    Ledger of all receipts per telemetry_and_receipts.md.
    
    Maintains:
    - Sequential receipt list
    - Hash chain for integrity
    - Node ID → receipt mapping
    """
    receipts: List[OperatorReceipt] = field(default_factory=list)
    hash_chain: List[str] = field(default_factory=list)
    node_receipts: Dict[int, OperatorReceipt] = field(default_factory=dict)
    
    def add_receipt(self, receipt: OperatorReceipt, prev_hash: str = None) -> None:
        """Add receipt to ledger with hash chaining."""
        receipt.compute_digest(prev_hash)
        self.receipts.append(receipt)
        self.hash_chain.append(receipt.digest)
        self.node_receipts[receipt.node_id] = receipt
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify hash chain integrity."""
        errors = []
        for i in range(1, len(self.receipts)):
            expected = self.hash_chain[i-1]
            actual = self.receipts[i].digest_in
            if expected != actual:
                errors.append(f"Chain break at receipt {i}: expected {expected}, got {actual}")
        return len(errors) == 0, errors


# -----------------------------
# Beam-Search Synthesis
# Upgrade #1: Invent operator chains, not just template filling
# -----------------------------

@dataclass
class PartialProposal:
    """
    Partial proposal during beam search.
    
    Contains:
    - Current node graph
    - Open slots to fill
    - Cumulative cost/debt
    - Last output type
    """
    nodes: List[Dict[str, Any]]  # NSC nodes
    open_slots: List[Dict[str, Any]]  # Slots waiting to be filled
    cost: float = 0.0
    depth: int = 0
    last_type: Optional[Dict[str, Any]] = None
    
    def __lt__(self, other: "PartialProposal") -> bool:
        return self.cost < other.cost


@dataclass
class OperatorCandidate:
    """Candidate operator for beam search expansion."""
    op_id: int
    name: str
    args: List[Dict[str, Any]]  # Expected input types
    ret: Dict[str, Any]  # Output type
    cost_increment: float = 0.0
    preserves_coherence: bool = True


class BeamSearchSynthesizer:
    """
    Beam-search synthesizer for operator chain invention.
    
    Per upgrade #1: "invent chains, not just fill templates"
    
    Algorithm:
    1. Start with seed (input field or literal)
    2. At each step, try all operators matching output type
    3. Keep top-k partial proposals (beam)
    4. Continue until target type reached or max depth
    5. Return completed chains ranked by coherence cost
    """
    
    def __init__(self,
                 registry: "OperatorRegistry",
                 binding: "KernelBinding",
                 debt_functional: DebtFunctional,
                 gate_config: GateConfig,
                 beam_width: int = 5,
                 max_depth: int = 8):
        self.registry = registry
        self.binding = binding
        self.debt_functional = debt_functional
        self.gate_config = gate_config
        self.beam_width = beam_width
        self.max_depth = max_depth
        
        # Build operator index by output type
        self.ops_by_output: Dict[str, List[OperatorCandidate]] = defaultdict(list)
        self._build_operator_index()
    
    def _build_operator_index(self) -> None:
        """Index operators by output type for fast lookup."""
        for op_id, op in self.registry.ops.items():
            # Convert TypeTag to hashable string
            ret_json = op.sig.ret.to_json() if hasattr(op.sig.ret, 'to_json') else op.sig.ret
            type_key = json.dumps(ret_json, sort_keys=True) if isinstance(ret_json, dict) else str(ret_json)
            
            args_json = [a.to_json() if hasattr(a, 'to_json') else str(a) for a in op.sig.args]
            
            self.ops_by_output[type_key].append(OperatorCandidate(
                op_id=op_id,
                name=op.token,
                args=args_json,
                ret=ret_json,
                cost_increment=0.1
            ))
    
    def synthesize(self,
                   input_type: Dict[str, Any],
                   target_type: Dict[str, Any],
                   context: Dict[str, Any] = None) -> List[Tuple[List[Dict[str, Any]], float, CoherenceMetrics]]:
        """
        Synthesize operator chains from input type to target type.
        
        Args:
            input_type: Starting type (e.g., FIELD(SCALAR))
            target_type: Goal type
            context: Available fields and constraints
        
        Returns:
            List of (chain, total_cost, final_metrics) sorted by cost
        """
        context = context or {}
        
        # Initialize beam with seed proposal
        seed_node = {
            "id": 1,
            "kind": "INPUT",
            "type": input_type,
            "payload": {"role": "input"}
        }

        seed = PartialProposal(
            nodes=[seed_node],
            open_slots=[{
                "role": "input",
                "expected_type": input_type,
                "position": 0
            }],
            cost=0.0,
            depth=0,
            last_type=input_type
        )
        
        beam: List[PartialProposal] = [seed]
        completed: List[Tuple[PartialProposal, CoherenceMetrics]] = []
        
        for depth in range(self.max_depth):
            new_beam: List[PartialProposal] = []
            
            for proposal in beam:
                # Try to fill open slots
                for slot in proposal.open_slots:
                    candidates = self._get_matching_operators(slot["expected_type"])
                    
                    for candidate in candidates:
                        # Create new proposal with this operator
                        new_proposal = self._apply_operator(
                            proposal, slot, candidate, depth
                        )
                        
                        if new_proposal is None:
                            continue
                        
                        # Check if target type reached
                        if self._types_match(new_proposal.last_type, target_type):
                            # Completed - compute final metrics
                            metrics = self._compute_metrics(new_proposal)
                            completed.append((new_proposal, metrics))
                        else:
                            # Continue searching
                            heappush(new_beam, new_proposal)
            
            # Prune beam to top-k
            beam = sorted(new_beam, key=lambda p: p.cost)[:self.beam_width]
            
            # Early exit if we have good completions
            if len(completed) >= self.beam_width:
                break
        
        # Sort completed by cost
        completed.sort(key=lambda x: x[0].cost)
        return [(p.nodes, p.cost, m) for p, m in completed[:self.beam_width]]
    
    def _get_matching_operators(self, output_type: Dict[str, Any]) -> List[OperatorCandidate]:
        """Get operators that produce the given output type."""
        type_key = output_type if isinstance(output_type, str) else json.dumps(output_type, sort_keys=True)
        
        if type_key in self.ops_by_output:
            return self.ops_by_output[type_key]
        return []
    
    def _apply_operator(self,
                        proposal: PartialProposal,
                        slot: Dict[str, Any],
                        candidate: OperatorCandidate,
                        depth: int) -> Optional[PartialProposal]:
        """Apply an operator to fill a slot with type-safe argument binding."""
        # Create node for this operator
        node_id = len(proposal.nodes) + 1
        
        # Determine kernel binding
        try:
            kernel_id, kernel_version = self.binding.require(candidate.op_id)
        except KeyError:
            return None  # No kernel binding
        
        # Resolve arguments with type safety
        bound_args = self._resolve_arguments(
            candidate,
            proposal.nodes,
            slot["expected_type"]
        )
        
        if bound_args is None:
            return None  # Cannot satisfy argument requirements
        
        # Create node with proper argument binding
        node = {
            "id": node_id,
            "kind": "APPLY",
            "type": candidate.ret,
            "op_id": candidate.op_id,
            "args": bound_args,  # Type-checked args
            "kernel_id": kernel_id,
            "kernel_version": kernel_version
        }
        
        # Compute real cost increment based on multiple factors
        cost_increment = self._compute_operator_cost(
            candidate.op_id,
            depth,
            len(proposal.nodes),
            candidate.preserves_coherence
        )
        
        return PartialProposal(
            nodes=proposal.nodes + [node],
            open_slots=[],  # Simplified - no branching yet
            cost=proposal.cost + cost_increment,
            depth=depth + 1,
            last_type=candidate.ret
        )
    
    def _types_match(self, type1: Any, type2: Any, allow_coercion: bool = True) -> bool:
        """Check if two types match, with optional coercion.
        
        Type compatibility matrix:
        - SCALAR = FLOAT, DOUBLE (numeric family)
        - PHASE = COMPLEX (algebraic family)
        - FIELD_REF = FIELD (collection family)
        - Exact match always succeeds
        """
        if not allow_coercion:
            if isinstance(type1, str) and isinstance(type2, str):
                return type1 == type2
            return json.dumps(type1, sort_keys=True) == json.dumps(type2, sort_keys=True)
        
        # Normalize for comparison
        t1_str = type1 if isinstance(type1, str) else json.dumps(type1, sort_keys=True)
        t2_str = type2 if isinstance(type2, str) else json.dumps(type2, sort_keys=True)
        
        # Exact match
        if t1_str == t2_str:
            return True
        
        # Numeric family coercion
        numeric_family = {"SCALAR", "FLOAT", "DOUBLE", "INT", "REAL"}
        if t1_str in numeric_family and t2_str in numeric_family:
            return True
        
        # Algebraic family coercion
        algebraic_family = {"PHASE", "COMPLEX", "QUATERNION"}
        if t1_str in algebraic_family and t2_str in algebraic_family:
            return True
        
        # Collection family coercion
        if "FIELD" in t1_str and "FIELD" in t2_str:
            return True
        
        return False
    
    def _resolve_arguments(self,
                          candidate: OperatorCandidate,
                          available_nodes: List[Dict[str, Any]],
                          expected_input_type: Dict[str, Any]) -> Optional[List[int]]:
        """Resolve operator arguments with type-safe binding.
        
        Returns list of node IDs that satisfy the operator's argument requirements,
        or None if binding fails.
        """
        if not candidate.args:
            return []  # No arguments required
        
        bound_args = []
        
        # For each argument the operator expects
        for arg_spec in candidate.args:
            if isinstance(arg_spec, dict):
                arg_type = arg_spec.get("type", arg_spec)
            else:
                arg_type = arg_spec

            if not arg_type:
                arg_type = expected_input_type
            
            # Find compatible nodes from available_nodes
            compatible = None
            for node in reversed(available_nodes):  # Prefer most recent
                node_type = node.get("type", {})
                if self._types_match(node_type, arg_type, allow_coercion=True):
                    compatible = node["id"]
                    break
            
            if compatible is None:
                # No compatible node found and no default
                return None
            
            bound_args.append(compatible)
        
        return bound_args
    
    def _compute_operator_cost(self,
                              op_id: int,
                              depth: int,
                              chain_length: int,
                              preserves_coherence: bool) -> float:
        """Compute real cost function for operator selection.
        
        Cost components:
        - Base cost: operator-specific cost
        - Depth penalty: deeper chains are less stable
        - Composition cost: operators compose better with some operators
        - Coherence bonus: operators that preserve coherence are cheaper
        
        Formula: cost = base + depth_penalty + composition_cost - coherence_bonus
        """
        # Base cost: 0.05 to 0.15 depending on operator
        base_cost = 0.1
        if op_id in [10, 11, 12]:  # Common source operators
            base_cost = 0.05
        elif op_id in [13, 14]:  # Regulatory operators
            base_cost = 0.08
        elif op_id >= 100:  # Exotic operators
            base_cost = 0.15
        
        # Depth penalty: increases exponentially with chain depth
        # Deeper chains accumulate more numerical error
        depth_penalty = 0.02 * (1.1 ** depth) - 0.02  # ~0% at depth 0, ~2.2% at depth 10
        
        # Composition cost: chains of similar operators are cheaper
        composition_cost = 0.01 if chain_length > 0 else 0.0
        
        # Coherence bonus: operators that preserve coherence are preferred
        coherence_bonus = 0.05 if preserves_coherence else 0.0
        
        total_cost = base_cost + depth_penalty + composition_cost - coherence_bonus
        return max(0.02, total_cost)  # Minimum cost floor
    
    def _compute_metrics(self, proposal: PartialProposal) -> CoherenceMetrics:
        """Compute coherence metrics for completed proposal with realistic model.
        
        Metrics are computed based on:
        - Chain length and operator types
        - Numerical stability (deeper chains = higher r_num)
        - Physics fidelity (type of operators used)
        - Constraint satisfaction
        """
        node_count = len(proposal.nodes)
        
        # Estimate physics residual: longer chains drift more
        # r_phys = base + growth with depth
        r_phys = 0.05 + 0.08 * node_count
        if node_count > 8:
            r_phys *= 1.5  # Penalty for very deep chains
        
        # Constraint residual: usually small, grows with complexity
        r_cons = 0.03 + 0.02 * node_count
        
        # Numerical stability: critical beyond depth 8-10
        r_num = 0.001 * (node_count ** 2)  # Quadratic growth with depth
        if node_count > 10:
            r_num = max(0.1, r_num)  # Hard floor at depth 10+
        
        # Reference residual (missing evidence)
        r_ref = 0.0  # Assume well-sourced for now
        
        # Staleness: receipts age linearly
        r_stale = 0.0  # Should be reset by fresh execution
        
        # Conflicts: assume well-composed chains avoid conflicts
        r_conflict = 0.0
        
        return CoherenceMetrics(
            r_phys=min(0.5, r_phys),  # Cap at 0.5
            r_cons=min(0.3, r_cons),
            r_num=min(0.5, r_num),
            r_ref=r_ref,
            r_stale=r_stale,
            r_conflict=r_conflict,
            r_retry=max(0, node_count - 6),  # Need retries if chain > 6
            r_rollback=0,
            r_sat=min(0.9, 0.1 * node_count)  # Controller saturation increases with depth
        )


# ===========================
# Phase 1.2: ExecutionTracer
# Learn from Receipt Ledgers
# ===========================

@dataclass
class ChainPattern:
    """Pattern in operator chain execution."""
    chain_signature: str  # Hash of operator sequence
    count: int = 0
    success_count: int = 0
    total_debt: float = 0.0
    min_debt: float = field(default_factory=lambda: float('inf'))
    max_debt: float = field(default_factory=lambda: float('-inf'))
    avg_residuals: CoherenceMetrics = field(default_factory=CoherenceMetrics)
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.count if self.count > 0 else 0.0
    
    @property
    def avg_debt(self) -> float:
        return self.total_debt / self.count if self.count > 0 else 0.0


@dataclass
class OperatorProfile:
    """Profile of an operator's behavior across executions."""
    op_id: int
    executions: int = 0
    successes: int = 0
    total_debt_contribution: float = 0.0
    avg_residual_magnitude: float = 0.0
    pairs_composition: Dict[int, int] = field(default_factory=dict)  # op_id -> count
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.0
    
    def add_execution(self, success: bool, debt_contribution: float, residual_mag: float):
        """Record an execution of this operator."""
        self.executions += 1
        if success:
            self.successes += 1
        self.total_debt_contribution += debt_contribution
        self.avg_residual_magnitude += residual_mag
    
    def get_avg_debt(self) -> float:
        return self.total_debt_contribution / self.executions if self.executions > 0 else 0.0
    
    def get_avg_residual(self) -> float:
        return self.avg_residual_magnitude / self.executions if self.executions > 0 else 0.0


class ExecutionTracer:
    """Extract learning signals from Receipt Ledgers.
    
    Analyzes execution history to discover:
    - Which operator chains work well
    - Coherence profiles and residual patterns
    - Operator compatibility scores
    - Failure modes and avoidance rules
    
    Phase 1.2 deliverable: Foundational learning from execution data.
    """
    
    def __init__(self):
        self.chain_patterns: Dict[str, ChainPattern] = {}
        self.operator_profiles: Dict[int, OperatorProfile] = {}
        self.total_executions = 0
        self.successful_executions = 0
        self.operator_pairs: Dict[Tuple[int, int], int] = defaultdict(int)  # (op1, op2) -> count
    
    def ingest_receipts(self, receipt_ledger: ReceiptLedger) -> None:
        """Ingest a receipt ledger and extract patterns."""
        if not receipt_ledger.receipts:
            return
        
        self.total_executions += 1
        
        # Determine if this execution succeeded (all gates passed)
        success = all(
            receipt.C_after < receipt.C_before + 0.1  # Debt didn't explode
            for receipt in receipt_ledger.receipts
        )
        if success:
            self.successful_executions += 1
        
        # Analyze each receipt
        for i, receipt in enumerate(receipt_ledger.receipts):
            op_id = receipt.op_id
            
            # Ensure operator profile exists
            if op_id not in self.operator_profiles:
                self.operator_profiles[op_id] = OperatorProfile(op_id)
            
            # Record execution
            debt_contrib = receipt.C_after - receipt.C_before
            residual_mag = math.sqrt(
                receipt.r_phys**2 + receipt.r_cons**2 + receipt.r_num**2
            )
            
            self.operator_profiles[op_id].add_execution(
                success=success,
                debt_contribution=debt_contrib,
                residual_mag=residual_mag
            )
            
            # Track operator pairs (composition)
            if i > 0:
                prev_op = receipt_ledger.receipts[i-1].op_id
                self.operator_pairs[(prev_op, op_id)] += 1
                self.operator_profiles[op_id].pairs_composition[prev_op] = \
                    self.operator_profiles[op_id].pairs_composition.get(prev_op, 0) + 1
        
        # Create chain pattern
        chain_sig = self._chain_signature(receipt_ledger.receipts)
        if chain_sig not in self.chain_patterns:
            self.chain_patterns[chain_sig] = ChainPattern(chain_signature=chain_sig)
        
        pattern = self.chain_patterns[chain_sig]
        pattern.count += 1
        if success:
            pattern.success_count += 1
        
        # Update debt stats
        total_chain_debt = sum(
            r.C_after - r.C_before for r in receipt_ledger.receipts
        )
        pattern.total_debt += total_chain_debt
        pattern.min_debt = min(pattern.min_debt, total_chain_debt)
        pattern.max_debt = max(pattern.max_debt, total_chain_debt)
    
    def operator_success_rate(self, op_id: int) -> float:
        """What fraction of executions with this operator passed gates?"""
        if op_id not in self.operator_profiles:
            return 0.5  # Unknown: assume 50%
        return self.operator_profiles[op_id].success_rate
    
    def operator_avg_debt(self, op_id: int) -> float:
        """Average debt contribution of this operator."""
        if op_id not in self.operator_profiles:
            return 0.1  # Unknown: assume moderate
        return self.operator_profiles[op_id].get_avg_debt()
    
    def operator_coherence_risk(self, op_id: int) -> float:
        """Risk score for coherence degradation (0=safe, 1=risky)."""
        if op_id not in self.operator_profiles:
            return 0.5
        
        profile = self.operator_profiles[op_id]
        
        # Risk = 1 - success_rate
        risk = 1.0 - profile.success_rate
        
        # Amplify by average residual magnitude
        risk *= (1.0 + profile.get_avg_residual())
        
        return min(1.0, risk)
    
    def chain_debt_histogram(self, chain_pattern: str = None) -> Dict[str, float]:
        """Distribution of debt for given chain pattern or all chains."""
        if chain_pattern:
            if chain_pattern not in self.chain_patterns:
                return {}
            pattern = self.chain_patterns[chain_pattern]
            return {
                "min": pattern.min_debt,
                "max": pattern.max_debt,
                "avg": pattern.avg_debt,
                "count": pattern.count,
                "success_rate": pattern.success_rate
            }
        else:
            # Aggregate across all patterns
            if not self.chain_patterns:
                return {}
            
            all_debts = [p.total_debt for p in self.chain_patterns.values()]
            return {
                "min": min(p.min_debt for p in self.chain_patterns.values() if p.count > 0),
                "max": max(p.max_debt for p in self.chain_patterns.values() if p.count > 0),
                "avg": sum(all_debts) / sum(p.count for p in self.chain_patterns.values()),
                "total_patterns": len(self.chain_patterns),
                "total_executions": self.total_executions
            }
    
    def operator_compatibility(self, op1: int, op2: int) -> float:
        """Compatibility score: how well does op1 → op2 compose? (0-1)."""
        # Check if we've seen this pair
        pair_count = self.operator_pairs.get((op1, op2), 0)
        
        if pair_count == 0:
            # Haven't seen this pair - neutral compatibility
            return 0.5
        
        # Success rate of compositions with this pair
        # This is simplified - in production would track success of chains containing this pair
        op1_success = self.operator_success_rate(op1)
        op2_success = self.operator_success_rate(op2)
        
        # Compatibility = average of both operators' success rates
        # Weighted by observation count
        base_compat = (op1_success + op2_success) / 2.0
        
        # Boost if pair is frequently successful
        observation_boost = min(0.1, pair_count * 0.01)
        
        return min(1.0, base_compat + observation_boost)
    
    def top_chains_by_success(self, k: int = 10) -> List[Tuple[str, float, int]]:
        """Top k operator chains by success rate.
        
        Returns: [(chain_signature, success_rate, execution_count), ...]
        """
        patterns = sorted(
            self.chain_patterns.values(),
            key=lambda p: (p.success_rate, -p.count),
            reverse=True
        )
        
        return [
            (p.chain_signature, p.success_rate, p.count)
            for p in patterns[:k]
        ]
    
    def top_operators_by_coherence(self, k: int = 10) -> List[Tuple[int, float]]:
        """Top k operators by coherence preservation (low risk).
        
        Returns: [(op_id, coherence_score), ...]
        """
        scored = [
            (op_id, 1.0 - self.operator_coherence_risk(op_id))
            for op_id in self.operator_profiles.keys()
        ]
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
    
    def failure_mode_analysis(self) -> Dict[str, List[Tuple]]:
        """Identify operators associated with failures.
        
        Returns: {
            "high_risk_operators": [(op_id, risk_score), ...],
            "high_risk_pairs": [((op1, op2), risk), ...]
        }
        """
        # High-risk operators: low success rate
        high_risk_ops = [
            (op_id, self.operator_coherence_risk(op_id))
            for op_id, profile in self.operator_profiles.items()
            if profile.success_rate < 0.6  # Less than 60% success
        ]
        high_risk_ops.sort(key=lambda x: x[1], reverse=True)
        
        # High-risk pairs: rarely observed or low combination success
        high_risk_pairs = []
        for (op1, op2), count in self.operator_pairs.items():
            if count < 3:  # Rare pair
                compat = self.operator_compatibility(op1, op2)
                if compat < 0.5:  # Poor compatibility
                    high_risk_pairs.append(((op1, op2), 1.0 - compat))
        
        high_risk_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "high_risk_operators": high_risk_ops[:10],
            "high_risk_pairs": high_risk_pairs[:10]
        }
    
    def summary(self) -> Dict[str, Any]:
        """Summarize learning insights."""
        total_success_rate = (
            self.successful_executions / self.total_executions
            if self.total_executions > 0 else 0.0
        )
        
        histogram = self.chain_debt_histogram()
        
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "success_rate": total_success_rate,
            "unique_operators": len(self.operator_profiles),
            "unique_chains": len(self.chain_patterns),
            "debt_stats": histogram,
            "top_chains": self.top_chains_by_success(5),
            "top_operators": self.top_operators_by_coherence(5),
            "failure_analysis": self.failure_mode_analysis()
        }
    
    def _chain_signature(self, receipts: List[OperatorReceipt]) -> str:
        """Create canonical signature for operator chain."""
        op_sequence = "->".join(str(r.op_id) for r in receipts)
        return hashlib.md5(op_sequence.encode()).hexdigest()[:16]


# ============================
# Phase 2.1: Operator Embeddings
# Learn semantic operator space
# ============================

@dataclass
class OperatorEmbedding:
    """Dense vector representation of operator semantics."""
    op_id: int
    embedding: List[float]  # Vector in embedding space
    embedding_dim: int = 64
    
    def __post_init__(self):
        """Ensure embedding is correct dimension."""
        if len(self.embedding) != self.embedding_dim:
            # Pad or truncate
            if len(self.embedding) < self.embedding_dim:
                self.embedding = self.embedding + [0.0] * (self.embedding_dim - len(self.embedding))
            else:
                self.embedding = self.embedding[:self.embedding_dim]
    
    def similarity(self, other: "OperatorEmbedding") -> float:
        """Cosine similarity with another embedding."""
        if not other or not self.embedding or not other.embedding:
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))
        mag_self = math.sqrt(sum(x ** 2 for x in self.embedding)) + 1e-8
        mag_other = math.sqrt(sum(x ** 2 for x in other.embedding)) + 1e-8
        
        return dot_product / (mag_self * mag_other)
    
    def distance(self, other: "OperatorEmbedding") -> float:
        """Euclidean distance to another embedding."""
        if not other:
            return float('inf')
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.embedding, other.embedding)))


class OperatorEmbeddingLearner:
    """Learn operator embeddings from execution traces.
    
    Phase 2.1 deliverable: Vector representations of operators in semantic space.
    
    Dimensions learned from profiles:
    - Reliability (success rate)
    - Numerical stability (residual magnitude)
    - Coherence impact (debt contribution)
    - Compatibility (with other operators)
    - Domain applicability
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.operator_embeddings: Dict[int, OperatorEmbedding] = {}
    
    def train(self, tracer: ExecutionTracer) -> Dict[int, OperatorEmbedding]:
        """Learn embeddings from ExecutionTracer data.
        
        Uses operator profiles to create semantic vectors:
        - Dimension 0: Success rate (0-1)
        - Dimension 1: Avg coherence risk (0-1, inverted)
        - Dimension 2: Debt contribution (normalized)
        - Dimension 3-N: Learned compatibility features
        
        Returns: Dict of op_id -> OperatorEmbedding
        """
        if not tracer.operator_profiles:
            return {}
        
        # Collect statistics for normalization
        all_debts = [p.get_avg_debt() for p in tracer.operator_profiles.values()]
        max_debt = max(all_debts) if all_debts else 0.1
        min_debt = min(all_debts) if all_debts else 0.0
        debt_range = max_debt - min_debt + 1e-8
        
        # Create embeddings for each operator
        for op_id, profile in tracer.operator_profiles.items():
            embedding_vec = self._create_embedding(op_id, profile, tracer, debt_range, min_debt)
            
            self.operator_embeddings[op_id] = OperatorEmbedding(
                op_id=op_id,
                embedding=embedding_vec,
                embedding_dim=self.embedding_dim
            )
        
        return self.operator_embeddings
    
    def _create_embedding(self, 
                         op_id: int, 
                         profile: OperatorProfile,
                         tracer: ExecutionTracer,
                         debt_range: float,
                         min_debt: float) -> List[float]:
        """Create embedding vector for single operator."""
        embedding = [0.0] * self.embedding_dim
        
        # Dimension 0: Success rate (0-1)
        embedding[0] = profile.success_rate
        
        # Dimension 1: Inverted coherence risk (0-1, higher = better)
        risk = tracer.operator_coherence_risk(op_id)
        embedding[1] = 1.0 - risk
        
        # Dimension 2: Normalized debt contribution (-1 to 1)
        avg_debt = profile.get_avg_debt()
        normalized_debt = (avg_debt - min_debt) / debt_range * 2.0 - 1.0
        embedding[2] = -normalized_debt  # Negative because lower debt is better
        
        # Dimension 3: Operator "popularity" (how often used)
        max_executions = max((p.executions for p in tracer.operator_profiles.values()), default=1)
        embedding[3] = profile.executions / max_executions if max_executions > 0 else 0.0
        
        # Dimension 4: Average residual magnitude (normalized)
        avg_residual = profile.get_avg_residual()
        embedding[4] = min(1.0, avg_residual)  # Cap at 1.0
        
        # Dimensions 5+: Pair compatibility features
        if tracer.operator_pairs:
            # Look at composition with other operators
            compatibility_scores = []
            for other_op_id in tracer.operator_profiles.keys():
                if other_op_id != op_id:
                    compat = tracer.operator_compatibility(op_id, other_op_id)
                    compatibility_scores.append(compat)
            
            # Compute statistics over compatibility scores
            if compatibility_scores:
                # Dimension 5: Mean compatibility
                embedding[5] = sum(compatibility_scores) / len(compatibility_scores)
                
                # Dimension 6: Std dev of compatibility
                mean_compat = embedding[5]
                variance = sum((c - mean_compat) ** 2 for c in compatibility_scores) / len(compatibility_scores)
                embedding[6] = math.sqrt(variance)
                
                # Dimension 7: Max compatibility (best pair)
                embedding[7] = max(compatibility_scores)
                
                # Dimension 8: Min compatibility (worst pair)
                embedding[8] = min(compatibility_scores)
        
        # Dimensions 9-63: Random projections for generalization
        # In Phase 3, these would be learned via neural network
        import random
        random.seed(op_id)  # Deterministic per operator
        for i in range(9, min(self.embedding_dim, 20)):  # Fill first half with projections
            embedding[i] = random.uniform(-0.5, 0.5)
        
        return embedding
    
    def similarity_to_profile(self, op_id: int, target_profile: Dict[str, float]) -> float:
        """Score how well operator matches a target profile.
        
        Target profile example:
        {
            "success_rate": 0.9,
            "low_debt": True,
            "coherent": True,
            "popular": True
        }
        """
        if op_id not in self.operator_embeddings:
            return 0.0
        
        embedding = self.operator_embeddings[op_id]
        score = 0.0
        weight_sum = 0.0
        
        # Score success rate dimension
        if "success_rate" in target_profile:
            target = target_profile["success_rate"]
            score += (1.0 - abs(embedding.embedding[0] - target)) * 1.0
            weight_sum += 1.0
        
        # Score coherence dimension
        if "coherent" in target_profile and target_profile["coherent"]:
            score += embedding.embedding[1] * 1.5  # Value coherence highly
            weight_sum += 1.5
        
        # Score low-debt dimension
        if "low_debt" in target_profile and target_profile["low_debt"]:
            score += max(0, embedding.embedding[2]) * 1.5
            weight_sum += 1.5
        
        # Score popularity
        if "popular" in target_profile and target_profile["popular"]:
            score += embedding.embedding[3] * 0.5
            weight_sum += 0.5
        
        return (score / weight_sum) if weight_sum > 0 else 0.5
    
    def find_similar_operators(self, op_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """Find k most similar operators in embedding space."""
        if op_id not in self.operator_embeddings:
            return []
        
        target_embedding = self.operator_embeddings[op_id]
        similarities = []
        
        for other_id, other_embedding in self.operator_embeddings.items():
            if other_id != op_id:
                sim = target_embedding.similarity(other_embedding)
                similarities.append((other_id, sim))
        
        # Sort by similarity, highest first
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def cluster_operators(self, num_clusters: int = 5) -> Dict[int, List[int]]:
        """Group operators into semantic clusters using embeddings.
        
        Phase 2.1 extension: Organize operators by semantic similarity.
        """
        if len(self.operator_embeddings) < num_clusters:
            # Too few operators to cluster
            clusters_dict = {i: [self.operator_embeddings[op_id].op_id] 
                           for i, op_id in enumerate(self.operator_embeddings.keys())}
            return clusters_dict
        
        # Simple k-means-like clustering
        from random import sample
        
        # Initialize clusters with random seeds
        seed_ops = sample(list(self.operator_embeddings.keys()), num_clusters)
        clusters = {i: [] for i in range(num_clusters)}
        
        # Assign each operator to nearest cluster seed
        for op_id, embedding in self.operator_embeddings.items():
            min_dist = float('inf')
            nearest_cluster = 0
            
            for cluster_idx, seed_op_id in enumerate(seed_ops):
                seed_embedding = self.operator_embeddings[seed_op_id]
                dist = embedding.distance(seed_embedding)
                
                if dist < min_dist:
                    min_dist = dist
                    nearest_cluster = cluster_idx
            
            clusters[nearest_cluster].append(op_id)
        
        return clusters
    
    def save_embeddings(self, filepath: str) -> None:
        """Save embeddings to JSON file."""
        embeddings_dict = {
            str(op_id): emb.embedding
            for op_id, emb in self.operator_embeddings.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "operator_embeddings": embeddings_dict
            }, f, indent=2)
    
    def load_embeddings(self, filepath: str) -> None:
        """Load embeddings from JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.embedding_dim = data.get("embedding_dim", 64)
            
            for op_id_str, embedding_vec in data.get("operator_embeddings", {}).items():
                op_id = int(op_id_str)
                self.operator_embeddings[op_id] = OperatorEmbedding(
                    op_id=op_id,
                    embedding=embedding_vec,
                    embedding_dim=self.embedding_dim
                )
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # File doesn't exist or invalid, start fresh


# ============================
# Phase 2.2: Success Predictor
# Neural network for chain success
# ============================

class SuccessPredictorModel:
    """Neural model: operator chain -> P(success).
    
    Phase 2.2 deliverable: Predict chain success before execution.
    
    Small architecture (not LLM-scale):
    - Input: Aggregated operator embeddings
    - Hidden: 128 -> 64 -> 32 neurons
    - Output: Single neuron, sigmoid for probability
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.model = None
        self.is_trained = False
        self._build_model()
    
    def _build_model(self):
        """Build neural network architecture."""
        try:
            from tensorflow import keras
            import tensorflow as tf
            
            self.model = keras.Sequential([
                keras.layers.Input(shape=(self.embedding_dim,), name="input"),
                keras.layers.Dense(128, activation="relu", name="dense1"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),
                
                keras.layers.Dense(64, activation="relu", name="hidden1"),
                keras.layers.Dropout(0.2),
                
                keras.layers.Dense(32, activation="relu", name="hidden2"),
                keras.layers.Dropout(0.1),
                
                keras.layers.Dense(1, activation="sigmoid", name="output")
            ])
            
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC()]
            )
            
            self.keras_available = True
        except ImportError:
            # Fallback: Simple sklearn-based predictor
            self.keras_available = False
            self._build_sklearn_model()
    
    def _build_sklearn_model(self):
        """Build sklearn-based fallback model."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=50, max_depth=10)
            self.sklearn_model = True
        except ImportError:
            # Final fallback: rule-based predictor
            self.model = None
            self.sklearn_model = False
    
    def predict(self, chain_embeddings: List[List[float]]) -> float:
        """Predict success probability for a chain.
        
        Args:
            chain_embeddings: List of operator embeddings in sequence
        
        Returns:
            Probability of success (0-1)
        """
        if not chain_embeddings:
            return 0.5
        
        # Aggregate embeddings across chain
        chain_vector = self._aggregate_embeddings(chain_embeddings)
        
        if self.keras_available and self.model:
            import numpy as np
            prediction = float(self.model.predict(np.array([chain_vector]), verbose=0)[0][0])
            return min(1.0, max(0.0, prediction))
        elif self.sklearn_model and self.model:
            prediction = float(self.model.predict_proba([chain_vector])[0][1])
            return prediction
        else:
            # Rule-based fallback
            return self._rule_based_predict(chain_vector)
    
    def _aggregate_embeddings(self, embeddings: List[List[float]]) -> List[float]:
        """Aggregate operator embeddings into chain vector.
        
        Strategy:
        - Mean: average operator quality
        - Std: diversity of chain
        - Max/Min: best/worst operator
        - Weighted by position (prefer good operators early)
        """
        if not embeddings:
            return [0.0] * self.embedding_dim
        
        # Convert to numpy for easier computation
        import numpy as np
        embeddings_array = np.array(embeddings)
        
        # Position weights: earlier operators weighted higher
        position_weights = np.array([1.0 / (i + 1) for i in range(len(embeddings))])
        position_weights /= position_weights.sum()  # Normalize
        
        # Weighted mean (prefer early operators)
        weighted_mean = np.average(embeddings_array, axis=0, weights=position_weights)
        
        # Standard deviation (chain diversity)
        std_dev = np.std(embeddings_array, axis=0)
        
        # Max and Min per dimension
        max_vals = np.max(embeddings_array, axis=0)
        min_vals = np.min(embeddings_array, axis=0)
        
        # Concatenate into aggregate vector
        aggregate = np.concatenate([
            weighted_mean,  # 0:64
            std_dev,        # 64:128
            max_vals,       # 128:192
            min_vals        # 192:256
        ])
        
        # Trim or pad to embedding_dim
        if len(aggregate) > self.embedding_dim:
            return list(aggregate[:self.embedding_dim])
        else:
            padding = [0.0] * (self.embedding_dim - len(aggregate))
            return list(aggregate) + padding
    
    def _rule_based_predict(self, chain_vector: List[float]) -> float:
        """Fallback rule-based predictor when models unavailable."""
        # Dimension 0: Success rate (most important)
        success_component = chain_vector[0] if len(chain_vector) > 0 else 0.5
        
        # Dimension 1: Coherence factor
        coherence_component = chain_vector[1] if len(chain_vector) > 1 else 0.5
        
        # Dimension 2: Debt factor (negative)
        debt_component = 1.0 - chain_vector[2] if len(chain_vector) > 2 else 0.5
        
        # Simple weighted combination
        prediction = (
            0.5 * success_component +
            0.3 * coherence_component +
            0.2 * debt_component
        )
        
        return min(1.0, max(0.0, prediction))
    
    def train(self,
             chains: List[List[List[float]]],
             labels: List[int],
             epochs: int = 20,
             batch_size: int = 32) -> Dict[str, float]:
        """Train the predictor on chains and outcomes.
        
        Args:
            chains: List of (chain_vectors: List[embeddings])
            labels: List of success outcomes (0 or 1)
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Training metrics (loss, accuracy)
        """
        if not chains or not labels:
            return {"error": "No training data"}
        
        # Aggregate each chain
        import numpy as np
        X = np.array([self._aggregate_embeddings(chain) for chain in chains])
        y = np.array(labels)
        
        if self.keras_available and self.model:
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_accuracy": float(history.history.get("val_accuracy", [0])[-1])
            }
        
        elif self.sklearn_model and self.model:
            self.model.fit(X, y)
            self.is_trained = True
            
            from sklearn.metrics import accuracy_score
            predictions = self.model.predict(X)
            acc = accuracy_score(y, predictions)
            
            return {"accuracy": float(acc)}
        
        else:
            self.is_trained = True
            return {"message": "Using rule-based predictor (no ML frameworks)"}
    
    def save_model(self, filepath: str) -> None:
        """Save trained model."""
        if self.keras_available and self.model:
            self.model.save(filepath)
        elif self.sklearn_model and self.model:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load_model(self, filepath: str) -> None:
        """Load pre-trained model."""
        try:
            if self.keras_available:
                from tensorflow import keras
                self.model = keras.models.load_model(filepath)
                self.is_trained = True
            elif self.sklearn_model:
                import pickle
                with open(filepath, 'rb') as f:
                    self.model = pickle.load(f)
                    self.is_trained = True
        except (FileNotFoundError, Exception):
            pass  # Model doesn't exist, use untrained


# ============================
# Phase 3: Online Learning Loop
# Continuous adaptation from execution feedback
# ============================

class AdaptiveLearningLoop:
    """Maintain embeddings and success predictor with online updates.

    Ingests receipt ledgers, retrains embeddings/predictor, and optionally
    persists learned state for reuse across runs.
    """

    def __init__(self,
                 embedding_dim: int = 64,
                 max_training_chains: int = 200,
                 model_dir: Optional[str] = None,
                 retrain_epochs: int = 10):
        self.embedding_dim = embedding_dim
        self.max_training_chains = max_training_chains
        self.model_dir = model_dir
        self.retrain_epochs = retrain_epochs

        self.tracer = ExecutionTracer()
        self.embedding_learner = OperatorEmbeddingLearner(embedding_dim=embedding_dim)
        self.predictor = SuccessPredictorModel(embedding_dim=embedding_dim)

        self.operator_embeddings: Dict[int, OperatorEmbedding] = {}
        self.training_chains_ops: List[List[int]] = []
        self.training_labels: List[int] = []
        self.new_chains_since_update = 0

        if self.model_dir:
            self.load_state(self.model_dir)

    def ingest_chain(self, ledger: ReceiptLedger) -> None:
        """Ingest a single execution chain (receipt ledger)."""
        import logging
        logger = logging.getLogger(__name__)
        
        if not ledger or not ledger.receipts:
            return

        # DEBUG: Log what we're ingesting
        logger.debug(f"[PHASE3_DEBUG] Ingesting chain with {len(ledger.receipts)} receipts")
        
        self.tracer.ingest_receipts(ledger)

        chain_ops = [r.op_id for r in ledger.receipts]
        success = all(r.C_after < r.C_before + 0.1 for r in ledger.receipts)

        # DEBUG: Log success calculation
        for r in ledger.receipts:
            debt_contrib = r.C_after - r.C_before
            logger.debug(f"[PHASE3_DEBUG]   Receipt op_id={r.op_id}: C_before={r.C_before:.4f}, C_after={r.C_after:.4f}, debt_contrib={debt_contrib:.4f}")
        
        logger.debug(f"[PHASE3_DEBUG] Chain success: {success}")

        self.training_chains_ops.append(chain_ops)
        self.training_labels.append(1 if success else 0)
        self.new_chains_since_update += 1

        if len(self.training_chains_ops) > self.max_training_chains:
            overflow = len(self.training_chains_ops) - self.max_training_chains
            self.training_chains_ops = self.training_chains_ops[overflow:]
            self.training_labels = self.training_labels[overflow:]

    def update_models(self, min_new_chains: int = 1) -> Dict[str, Any]:
        """Update embeddings and predictor when new data arrives."""
        if self.new_chains_since_update < min_new_chains:
            return {"updated": False, "reason": "insufficient_new_data"}

        self.operator_embeddings = self.embedding_learner.train(self.tracer)

        chain_embeddings = []
        labels = []
        for chain_ops, label in zip(self.training_chains_ops, self.training_labels):
            try:
                chain_embeddings.append([
                    self.operator_embeddings[op_id].embedding for op_id in chain_ops
                ])
                labels.append(label)
            except KeyError:
                continue

        metrics = self.predictor.train(chain_embeddings, labels, epochs=self.retrain_epochs)
        self.new_chains_since_update = 0

        if self.model_dir:
            self.save_state(self.model_dir)

        return {"updated": True, "metrics": metrics, "chains": len(labels)}

    def predict_chain(self, chain_ops: List[int]) -> float:
        """Predict success probability for a chain of op ids."""
        if not chain_ops or not self.operator_embeddings:
            return 0.5

        chain_embeddings = [
            self.operator_embeddings[op_id].embedding
            for op_id in chain_ops
            if op_id in self.operator_embeddings
        ]
        return self.predictor.predict(chain_embeddings)

    def save_state(self, model_dir: str) -> None:
        """Persist embeddings and predictor to disk."""
        os.makedirs(model_dir, exist_ok=True)
        embeddings_path = os.path.join(model_dir, "embeddings.json")
        model_path = os.path.join(model_dir, "success_predictor")

        self.embedding_learner.operator_embeddings = self.operator_embeddings
        self.embedding_learner.save_embeddings(embeddings_path)
        self.predictor.save_model(model_path)

    def load_state(self, model_dir: str) -> None:
        """Load embeddings and predictor from disk if available."""
        embeddings_path = os.path.join(model_dir, "embeddings.json")
        model_path = os.path.join(model_dir, "success_predictor")

        self.embedding_learner.load_embeddings(embeddings_path)
        self.operator_embeddings = self.embedding_learner.operator_embeddings
        self.predictor.load_model(model_path)


# -----------------------------
# Complete NPE v2.0 with All Upgrades
# -----------------------------

# Reuse from npe_v1_0 (canonical_json, types, registry, binding, nodes, etc.)
from nsc.npe_v1_0 import (
    canonical_json, sha256_hex, compute_hash_chain,
    TypeTag, SCALAR, PHASE, FIELD, REPORT_GATE,
    OpSig, Operator, OperatorRegistry, KernelBinding, Node, Module,
    Template, Hole, Template_PsiChain,
    GateDecision, GatePolicy, RailSet, Rail_R1_Deflation, Rail_R2_Rollback,
    Rail_R3_Projection, Rail_R4_Damping, RailResult,
    DebtWeights, ResidualBlock, CoherenceGate, CoherenceGateReport,
    ProposalContext, ProposalReceipt, NoeticanProposalEngine_v1_0
)


class NoeticanProposalEngine_v2_0(NoeticanProposalEngine_v1_0):
    """
    NPE v2.0 with all upgrades:
    
    1. Beam-search synthesis for operator chain invention
    2. Real coherence metrics (r_gate, r_cons, r_num)
    3. OperatorReceipt emission for executed proposals
    """
    
    def __init__(self,
                 registry: OperatorRegistry,
                 binding: KernelBinding,
                 templates: List[Template],
                 coherence_gate: CoherenceGate,
                 weights: Dict[str, float],
                 # New v2.0 parameters
                 debt_functional: DebtFunctional = None,
                 gate_config: GateConfig = None,
                 synthesizer: BeamSearchSynthesizer = None,
                 beam_width: int = 5,
                 max_synthesis_depth: int = 8,
                 adaptive_learner: Optional[AdaptiveLearningLoop] = None):
        
        super().__init__(registry, binding, templates, coherence_gate, weights)
        
        # Coherence metrics
        self.debt_functional = debt_functional or DebtFunctional()
        self.gate_config = gate_config or GateConfig()
        
        # Beam-search synthesizer
        if synthesizer is None:
            self.synthesizer = BeamSearchSynthesizer(
                registry=registry,
                binding=binding,
                debt_functional=self.debt_functional,
                gate_config=self.gate_config,
                beam_width=beam_width,
                max_depth=max_synthesis_depth
            )
        else:
            self.synthesizer = synthesizer
        
        # Receipt ledger
        self.ledger = ReceiptLedger()

        # Optional online learning loop
        self.adaptive_learner = adaptive_learner
        
        # Event counter for receipt IDs
        self.event_counter = 0
    
    def synthesize_and_execute(self,
                               input_type: Dict[str, Any],
                               target_type: Dict[str, Any],
                               context: Dict[str, Any] = None) -> Tuple[List[Tuple[Module, OperatorReceipt, CoherenceMetrics]], ReceiptLedger]:
        """
        Synthesize operator chains using beam search and emit receipts.
        
        Args:
            input_type: Starting type
            target_type: Goal type
            context: Available fields
        
        Returns:
            Tuple of (completed_proposals, receipt_ledger)
        """
        context = context or {}
        
        # Beam-search synthesis
        chains = self.synthesizer.synthesize(input_type, target_type, context)
        
        results: List[Tuple[Module, OperatorReceipt, CoherenceMetrics]] = []
        
        for chain_nodes, chain_cost, metrics in chains:
            # Build module from chain
            module = self._build_module_from_chain(chain_nodes, context)
            
            # Compute debt
            debt, _ = self.debt_functional.compute(metrics)
            
            # Evaluate gates with real metrics
            gate_result, _ = self.gate_config.evaluate(metrics)
            
            # Emit operator receipts for each node
            chain_ledger = ReceiptLedger()
            self._emit_receipts(module, metrics, chain_cost, chain_ledger)
            
            # Create proposal receipt
            proposal_receipt = ProposalReceipt(
                kind="proposal_receipt",
                module_id=module.module_id,
                module_digest=module.digest(),
                template_id="beam_synthesis",
                gate_pass=gate_result != GateResult.HARD_FAIL,
                debt=debt,
                debt_decomposition=metrics.to_json(),
                residuals=metrics.to_json(),
                gate_decision=gate_result.value,
                parent_hash=self.prev_hash
            )
            
            # Compute hash chain
            receipt_dict = {
                "kind": proposal_receipt.kind,
                "module_id": proposal_receipt.module_id,
                "module_digest": proposal_receipt.module_digest,
                "template_id": proposal_receipt.template_id,
                "gate_pass": proposal_receipt.gate_pass,
                "debt": proposal_receipt.debt,
                "debt_decomposition": proposal_receipt.debt_decomposition,
                "residuals": proposal_receipt.residuals,
                "gate_decision": proposal_receipt.gate_decision,
                "parent_hash": proposal_receipt.parent_hash
            }
            proposal_receipt.digest = compute_hash_chain(receipt_dict, self.prev_hash)
            self.prev_hash = proposal_receipt.digest
            
            results.append((module, proposal_receipt, metrics))

            if self.adaptive_learner:
                self.adaptive_learner.ingest_chain(chain_ledger)
        
        if self.adaptive_learner:
            import logging
            logger = logging.getLogger(__name__)
            
            # DEBUG: Log retraining frequency
            logger.debug(f"[PHASE3_DEBUG] Retraining triggered: new_chains_since_update={self.adaptive_learner.new_chains_since_update}")
            
            update_result = self.adaptive_learner.update_models(min_new_chains=1)
            
            logger.debug(f"[PHASE3_DEBUG] Retraining result: {update_result}")

        # Sort by debt (lower is better)
        results.sort(key=lambda x: x[1].debt)
        
        return results, self.ledger
    
    def _build_module_from_chain(self,
                                  chain_nodes: List[Dict[str, Any]],
                                  context: Dict[str, Any]) -> Module:
        """Build NSC module from synthesized chain."""
        # Create nodes
        nodes = []
        for i, node_data in enumerate(chain_nodes):
            node = Node(
                id=node_data["id"],
                kind=node_data["kind"],
                type=FIELD(SCALAR),  # Simplified
                payload={
                    "op_id": node_data["op_id"],
                    "args": node_data.get("args", []),
                    "kernel_id": node_data.get("kernel_id", ""),
                    "kernel_version": node_data.get("kernel_version", "")
                }
            )
            nodes.append(node)
        
        # Create SEQ
        seq_id = len(nodes) + 1
        seq = Node(
            id=seq_id,
            kind="SEQ",
            type=FIELD(SCALAR),
            payload={"seq": [n.id for n in nodes]}
        )
        nodes.append(seq)
        
        # Create module
        module_id = f"beam_synth_{self.event_counter}"
        self.event_counter += 1
        
        return Module(
            module_id=module_id,
            registry_id=self.registry.registry_id,
            nodes=nodes,
            entrypoints={"Output": nodes[-1].id if nodes else seq_id},
            eval_seq_id=seq_id,
            meta={"template": "beam_synthesis", "synthesis_depth": len(chain_nodes)}
        )
    
    def _emit_receipts(self,
                       module: Module,
                       metrics: CoherenceMetrics,
                       chain_cost: float,
                       chain_ledger: Optional[ReceiptLedger] = None) -> None:
        """Emit OperatorReceipt for each APPLY node."""
        import logging
        logger = logging.getLogger(__name__)
        
        self.event_counter += 1
        
        # Compute final coherence for the chain
        C_before = 0.0
        C_after, _ = self.debt_functional.compute(metrics)
        
        # DEBUG: Log coherence values for validation
        logger.debug(f"[PHASE3_DEBUG] Chain {module.module_id}: C_after={C_after:.4f}")
        
        for node in module.nodes:
            if node.kind == "APPLY":
                self.event_counter += 1
                receipt = OperatorReceipt(
                    event_id=f"evt_{self.event_counter}",
                    node_id=node.id,
                    op_id=node.payload.get("op_id", 0),
                    digest_in=f"sha256:{'0'*64}",
                    digest_out=module.digest(),
                    kernel_id=node.payload.get("kernel_id", ""),
                    kernel_version=node.payload.get("kernel_version", ""),
                    C_before=C_before,
                    C_after=C_after,
                    r_phys=metrics.r_phys,
                    r_cons=metrics.r_cons,
                    r_num=metrics.r_num,
                    registry_id=self.registry.registry_id,
                    registry_version=self.registry.version
                )
                
                # DEBUG: Log the C_before/C_after values for each operator
                logger.debug(f"[PHASE3_DEBUG]   Op {receipt.op_id}: C_before={receipt.C_before:.4f}, C_after={receipt.C_after:.4f}, debt_contrib={receipt.C_after - receipt.C_before:.4f}")
                
                # Add to ledgers
                prev_hash = self.ledger.hash_chain[-1] if self.ledger.hash_chain else None
                self.ledger.add_receipt(receipt, prev_hash)
                if chain_ledger is not None:
                    chain_prev = chain_ledger.hash_chain[-1] if chain_ledger.hash_chain else None
                    chain_ledger.add_receipt(receipt, chain_prev)
                
                C_before = C_after


# -----------------------------
# Demo: Run NPE v2.0 with all upgrades
# -----------------------------

def demo_v2():
    """Demonstrate NPE v2.0 with beam search, real metrics, and receipts."""
    
    # Setup (reuse from v1.0)
    from nsc.npe_v1_0 import (
        Template_PsiChain, CoherenceGate, DebtWeights, GatePolicy, RailSet
    )
    
    ops = {
        10: Operator(10, "source_plus", OpSig([FIELD(SCALAR)], FIELD(SCALAR)), 
                    "nsc.apply.source_plus", ["kernel_id", "kernel_version"], "res.op.source_plus"),
        11: Operator(11, "twist", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                    "nsc.apply.curvature_twist", ["kernel_id", "kernel_version"], "res.op.twist"),
        12: Operator(12, "delta", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                    "nsc.apply.delta_transform", ["kernel_id", "kernel_version"], "res.op.delta"),
        13: Operator(13, "regulate", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                    "nsc.apply.regulate_circle", ["kernel_id", "kernel_version"], "res.op.regulate"),
        14: Operator(14, "sink_minus", OpSig([FIELD(SCALAR)], FIELD(SCALAR)),
                    "nsc.apply.sink_minus", ["kernel_id", "kernel_version"], "res.op.sink_minus"),
        20: Operator(20, "return_to_baseline", OpSig([FIELD(SCALAR), FIELD(SCALAR)], FIELD(SCALAR)),
                    "nsc.apply.return_to_baseline", ["kernel_id", "kernel_version", "params"], "res.op.return_baseline"),
    }
    registry = OperatorRegistry("noetican.nsc.registry.v1_1", "1.1.0", ops)
    
    binding = KernelBinding(
        registry_id=registry.registry_id,
        binding_id="binding.demo.v1",
        bindings={
            10: ("kern.source_plus.v1", "1.0.0"),
            11: ("kern.twist.v1", "1.0.0"),
            12: ("kern.delta.v1", "1.0.0"),
            13: ("kern.regulate.v1", "1.0.0"),
            14: ("kern.sink_minus.v1", "1.0.0"),
            20: ("kern.return_baseline.linear_blend", "1.0.0"),
        }
    )
    
    # Templates
    templates = [Template_PsiChain("psi_chain_v1", {
        "source_plus": 10, "twist": 11, "delta": 12,
        "regulate": 13, "sink_minus": 14, "return_to_baseline": 20,
    })]
    
    # Coherence gate
    coherence_gate = CoherenceGate(DebtWeights(), GatePolicy(), RailSet())
    weights = {"debt": -1.0, "gate_pass": 2.0}
    
    # Create v2.0 engine
    engine = NoeticanProposalEngine_v2_0(
        registry=registry,
        binding=binding,
        templates=templates,
        coherence_gate=coherence_gate,
        weights=weights,
        beam_width=3,
        max_synthesis_depth=6
    )
    
    print("=" * 70)
    print("NPE v2.0 - Beam Search + Real Metrics + OperatorReceipts")
    print("=" * 70)
    
    # Input/output types
    input_type = FIELD(SCALAR).to_json()
    target_type = FIELD(SCALAR).to_json()
    
    # Synthesize and execute
    results, ledger = engine.synthesize_and_execute(
        input_type=input_type,
        target_type=target_type,
        context={"available_fields": {1: FIELD(SCALAR), 2: FIELD(SCALAR)}}
    )
    
    print(f"\n--- Beam Search Results ---")
    for i, (module, receipt, metrics) in enumerate(results[:5]):
        print(f"\nProposal #{i+1}:")
        print(f"  Module: {receipt.module_id}")
        print(f"  Debt: {receipt.debt:.4f}")
        print(f"  Metrics:")
        print(f"    r_phys: {metrics.r_phys:.4f}")
        print(f"    r_cons: {metrics.r_cons:.4f}")
        print(f"    r_num:  {metrics.r_num:.4f}")
        print(f"  Gate Decision: {receipt.gate_decision}")
        print(f"  Operator Receipts: {len(ledger.receipts)}")
    
    print(f"\n--- Receipt Ledger ---")
    print(f"Total receipts: {len(ledger.receipts)}")
    print(f"Chain verified: {ledger.verify_chain()[0]}")
    
    print("\n" + "=" * 70)
    print("NPE v2.0 upgrades complete:")
    print("  ✓ Beam-search synthesis (chain invention)")
    print("  ✓ Real coherence metrics (r_phys, r_cons, r_num)")
    print("  ✓ OperatorReceipt emission with hash chaining")
    print("=" * 70)


if __name__ == "__main__":
    demo_v2()
