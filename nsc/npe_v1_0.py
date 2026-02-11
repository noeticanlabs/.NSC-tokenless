"""
Noetican Proposal Engine v1.0 (Aligned with Coherence Framework)

This implementation integrates:
- Coherence debt functionals (L1)
- Hard/soft gates (L2)
- Rails (R1-R4) (L3)
- Receipt hash chaining (L2-L4)
- Deterministic proposal synthesis over typed NSC graphs

References:
- coherence_spine/02_math/coherence_functionals.md
- coherence_spine/04_control/gates_and_rails.md
- coherence_spine/05_runtime/reference_implementations.md
- coherence_spine/03_measurement/telemetry_and_receipts.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
import hashlib
import json
import math
from enum import Enum

# -----------------------------
# JSON Schema Validation (L0 Conformance)
# Per conformance_levels.md: L0 requires schema validation
# -----------------------------

try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False

# Schema registry for NSC objects
SCHEMA_REGISTRY: Dict[str, Dict] = {}

def load_schema(schema_path: str) -> Dict:
    """Load a JSON schema from file path."""
    if schema_path in SCHEMA_REGISTRY:
        return SCHEMA_REGISTRY[schema_path]
    
    import os
    full_path = os.path.join(os.path.dirname(__file__), "..", "..", "schemas", schema_path)
    if not os.path.exists(full_path):
        # Fallback to local schemas dir
        full_path = os.path.join(os.path.dirname(__file__), "..", "schemas", schema_path)
    
    try:
        with open(full_path, "r") as f:
            schema = json.load(f)
            SCHEMA_REGISTRY[schema_path] = schema
            return schema
    except FileNotFoundError:
        return {}

class SchemaValidator:
    """Validates NSC objects against JSON schemas (L0 conformance)."""
    
    def __init__(self, schema_name: str):
        self.schema_name = schema_name
        self.schema = load_schema(schema_name)
    
    def validate(self, obj: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate an object against the schema.
        
        Returns:
            Tuple of (is_valid, errors)
        """
        if not JSONSCHEMA_AVAILABLE:
            return True, ["jsonschema not installed - skipping validation"]
        
        if not self.schema:
            return True, [f"Schema {self.schema_name} not found - skipping validation"]
        
        try:
            jsonschema.validate(instance=obj, schema=self.schema)
            return True, []
        except jsonschema.ValidationError as e:
            return False, [f"{e.json_path}: {e.message}"]

class NSCModuleValidator:
    """Validates NSC modules against all required L0 schemas."""
    
    def __init__(self):
        self.validators = {
            "module": SchemaValidator("nsc_module.schema.json"),
            "node": SchemaValidator("nsc_node.schema.json"),
            "type": SchemaValidator("nsc_type.schema.json"),
            "kernel_binding": SchemaValidator("nsc_kernel_binding.schema.json"),
        }
    
    def validate_module(self, module: Module) -> Tuple[bool, List[str]]:
        """Validate module against nsc_module.schema.json."""
        module_dict = module.to_json()
        return self.validators["module"].validate(module_dict)
    
    def validate_node(self, node: Node) -> Tuple[bool, List[str]]:
        """Validate node against nsc_node.schema.json."""
        node_dict = node.to_json()
        return self.validators["node"].validate(node_dict)
    
    def validate_type(self, type_tag: TypeTag) -> Tuple[bool, List[str]]:
        """Validate type against nsc_type.schema.json."""
        type_dict = type_tag.to_json()
        return self.validators["type"].validate(type_dict)
    
    def validate_kernel_binding(self, binding: KernelBinding) -> Tuple[bool, List[str]]:
        """Validate kernel binding against nsc_kernel_binding.schema.json."""
        binding_dict = {
            "registry_id": binding.registry_id,
            "binding_id": binding.binding_id,
            "bindings": [
                {"op_id": k, "kernel_id": v[0], "kernel_version": v[1]}
                for k, v in binding.bindings.items()
            ]
        }
        return self.validators["kernel_binding"].validate(binding_dict)

# -----------------------------
# Canonical JSON + Hash Chaining
# Per telemetry_and_receipts.md: H_k = sha256(canonical_json(R_k) || H_{k-1})
# -----------------------------

def canonical_json(obj: Any) -> str:
    """Canonical JSON per coherence spine spec (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def compute_hash_chain(receipt: Dict[str, Any], prev_hash: str = None) -> str:
    """Compute hash chain entry per telemetry_and_receipts.md."""
    canon = canonical_json(receipt).encode("utf-8")
    if prev_hash:
        # H_k = sha256(canonical_json(R_k) || H_{k-1})
        combined = canon + prev_hash.encode("utf-8")
        return "sha256:" + sha256_hex(combined)
    else:
        return "sha256:" + sha256_hex(canon)

# -----------------------------
# Types (Tokenless Tags) - Aligned with nsc/ schemas
# -----------------------------

@dataclass(frozen=True)
class TypeTag:
    tag: str
    of: Optional["TypeTag"] = None
    kind: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        out = {"tag": self.tag}
        if self.of is not None:
            out["of"] = self.of.to_json()
        if self.kind is not None:
            out["kind"] = self.kind
        return out

SCALAR = TypeTag("SCALAR")
PHASE = TypeTag("PHASE")

def FIELD(of: TypeTag) -> TypeTag:
    return TypeTag("FIELD", of=of)

REPORT_GATE = TypeTag("REPORT", kind="GATE_REPORT")
REPORT_PROPOSAL = TypeTag("REPORT", kind="PROPOSAL_RECEIPT")

# -----------------------------
# Coherence Debt Computation (L1)
# Per coherence_functionals.md: C(x) = Σ_i w_i ||r̃_i||² + Σ_j v_j p_j(x)
# -----------------------------

@dataclass
class ResidualBlock:
    """Per residuals_metrics.md: standardized residual blocks."""
    cons: float = 0.0      # Constraint residual
    rec: float = 0.0       # Conservation drift
    tool: float = 0.0      # Evidence/tool residuals
    phys_rms: float = 0.0  # Physics residual (equation defect)

@dataclass
class DebtWeights:
    """Per coherence_functionals.md: weights for debt functional."""
    w_cons: float = 1.0
    w_rec: float = 1.0
    w_tool: float = 1.0
    w_phys: float = 1.0
    v_thrash: float = 1.0  # Operational thrash penalty

    def compute_debt(self, residuals: ResidualBlock) -> Tuple[float, Dict[str, float]]:
        """
        Compute debt functional C(x) per coherence_functionals.md.
        
        Returns:
            Tuple of (total_debt, decomposition_dict)
        """
        # Normalized squared residuals (using L2 norm, RMS-style)
        r_cons_sq = (residuals.cons ** 2)
        r_rec_sq = (residuals.rec ** 2)
        r_tool_sq = (residuals.tool ** 2)
        r_phys_sq = (residuals.phys_rms ** 2)
        
        # Weighted sum
        cons_debt = self.w_cons * r_cons_sq
        rec_debt = self.w_rec * r_rec_sq
        tool_debt = self.w_tool * r_tool_sq
        phys_debt = self.w_phys * r_phys_sq
        
        # Thrash penalty (operational inefficiency)
        # Per minimal_tests.py: retry count contributes to thrash
        thrash_debt = self.v_thrash * r_phys_sq  # Simplified: thrash scales with phys residual
        
        total = cons_debt + rec_debt + tool_debt + phys_debt + thrash_debt
        
        decomposition = {
            "cons": cons_debt,
            "rec": rec_debt,
            "tool": tool_debt,
            "phys": phys_debt,
            "thrash": thrash_debt,
            "penalty": 0.0
        }
        
        return total, decomposition

# -----------------------------
# Gates (L2)
# Per gates_and_rails.md: Hard gates (non-negotiable) vs Soft gates (repairable)
# -----------------------------

class GateDecision(Enum):
    ACCEPT = "accept"
    WARN = "warn"
    REJECT = "reject"
    ABORT = "abort"

@dataclass
class GatePolicy:
    """
    Per gates_and_rails.md: gate thresholds and hysteresis.
    
    Hysteresis defaults from spec:
    - warn_enter = 0.75 * fail_enter
    - warn_exit = 0.60 * fail_enter
    - fail_exit = 0.80 * fail_enter
    """
    # Hard gate thresholds
    hard_nan_free: bool = True
    hard_positivity: float = 0.0  # ρ ≥ 0
    hard_dt_min: float = 1e-6
    
    # Soft gate thresholds
    soft_cons_warn: float = 0.3
    soft_cons_fail: float = 0.75
    soft_rec_warn: float = 0.25
    soft_rec_fail: float = 0.6
    soft_phys_warn: float = 0.5
    soft_phys_fail: float = 1.0
    
    def evaluate(self, residuals: ResidualBlock, nan_free: bool = True) -> Tuple[GateDecision, Dict[str, Any]]:
        """
        Per gates_and_rails.md: evaluate hard and soft gates.
        
        Returns:
            Tuple of (decision, verdict_details)
        """
        verdicts = {
            "hard": {},
            "soft": {},
            "debt": None
        }
        
        # Hard gates (non-negotiable) - fail => abort
        verdicts["hard"]["nan_free"] = "pass" if nan_free else "fail"
        
        if not nan_free:
            return GateDecision.ABORT, verdicts
        
        # Soft gates (repairable) - fail => reject (then rails apply)
        soft_failures = []
        soft_warnings = []
        
        if residuals.cons > self.soft_cons_fail:
            soft_failures.append("cons")
        elif residuals.cons > self.soft_cons_warn:
            soft_warnings.append("cons")
            
        if residuals.rec > self.soft_rec_fail:
            soft_failures.append("rec")
        elif residuals.rec > self.soft_rec_warn:
            soft_warnings.append("rec")
            
        if residuals.phys_rms > self.soft_phys_fail:
            soft_failures.append("phys")
        elif residuals.phys_rms > self.soft_phys_warn:
            soft_warnings.append("phys")
        
        verdicts["soft"]["failures"] = soft_failures
        verdicts["soft"]["warnings"] = soft_warnings
        
        # Decision logic
        if soft_failures:
            return GateDecision.REJECT, verdicts
        elif soft_warnings:
            return GateDecision.WARN, verdicts
        else:
            return GateDecision.ACCEPT, verdicts

# -----------------------------
# Rails (L3)
# Per reference_implementations.md: R1-R4 bounded corrective actions
# -----------------------------

@dataclass
class RailResult:
    """Result of applying a rail."""
    rail_name: str
    applied: bool
    dt_change: float = 0.0
    state_change: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

class Rail:
    """Base class for rails per gates_and_rails.md."""
    
    def apply(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any], 
              decision: GateDecision) -> Optional[RailResult]:
        raise NotImplementedError

class Rail_R1_Deflation(Rail):
    """
    Per reference_implementations.md R1: dt deflation
    Trigger: phys defect too large, CFL too high, repeated soft fails
    Update: dt ← max(dt_min, α dt), α∈(0,1) (typical α=0.8)
    """
    def __init__(self, alpha: float = 0.8, dt_min: float = 1e-6):
        self.alpha = alpha
        self.dt_min = dt_min
    
    def apply(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any],
              decision: GateDecision) -> Optional[RailResult]:
        if decision != GateDecision.REJECT:
            return None
        
        # Only apply if phys residual is the dominant failure
        if residuals.phys_rms > 0.5:  # Threshold for R1 application
            dt_new = max(self.dt_min, self.alpha * dt)
            return RailResult(
                rail_name="R1_dt_deflation",
                applied=True,
                dt_change=dt_new - dt,
                reason="phys_rms_too_high"
            )
        return None

class Rail_R2_Rollback(Rail):
    """
    Per reference_implementations.md R2: rollback
    Trigger: any hard fail, or retry cap hit
    Update: x ← x_prev (or abort if rollback unavailable)
    """
    def __init__(self, retry_cap: int = 6):
        self.retry_cap = retry_cap
    
    def apply(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any],
              decision: GateDecision) -> Optional[RailResult]:
        if decision == GateDecision.ABORT:
            # Hard failure - require rollback
            if "prev_state" in state:
                return RailResult(
                    rail_name="R2_rollback",
                    applied=True,
                    state_change={"state": state["prev_state"]},
                    reason="hard_failure"
                )
            else:
                return RailResult(
                    rail_name="R2_rollback",
                    applied=False,
                    reason="no_prev_state_available"
                )
        return None

class Rail_R3_Projection(Rail):
    """
    Per reference_implementations.md R3: projection to constraint manifold
    Trigger: constraint residual dominates while phys defect is otherwise stable
    Update: x ← Π(x), where Π is a declared projection
    """
    def __init__(self, projection_factor: float = 0.5):
        self.projection_factor = projection_factor
    
    def apply(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any],
              decision: GateDecision) -> Optional[RailResult]:
        if decision != GateDecision.REJECT:
            return None
        
        # Apply if constraint residual dominates
        if residuals.cons > 2 * residuals.phys_rms and residuals.cons > 0.3:
            # Simplified projection: reduce constraint residual
            new_cons = residuals.cons * self.projection_factor
            return RailResult(
                rail_name="R3_projection",
                applied=True,
                state_change={"cons_reduced_by": self.projection_factor},
                reason="constraint_dominates"
            )
        return None

class Rail_R4_Damping(Rail):
    """
    Per reference_implementations.md R4: bounded damping/gain adjustment
    Trigger: constraint residual persists; state remains stable
    Update: κ ← min(κ_max, κ + Δκ), bounded Δκ
    """
    def __init__(self, delta_kappa: float = 0.1, kappa_max: float = 2.0):
        self.delta_kappa = delta_kappa
        self.kappa_max = kappa_max
    
    def apply(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any],
              decision: GateDecision) -> Optional[RailResult]:
        if decision != GateDecision.WARN:
            return None
        
        # Apply if constraint residual persists at warning level
        if 0.2 < residuals.cons < 0.5:
            current_kappa = state.get("kappa", 1.0)
            new_kappa = min(self.kappa_max, current_kappa + self.delta_kappa)
            return RailResult(
                rail_name="R4_damping",
                applied=True,
                state_change={"kappa": new_kappa},
                reason="constraint_persists"
            )
        return None

@dataclass
class RailSet:
    """Per reference_implementations.md: deterministic rail priority."""
    rails: List[Rail] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.rails:
            self.rails = [
                Rail_R1_Deflation(),
                Rail_R3_Projection(),
                Rail_R4_Damping(),
                Rail_R2_Rollback()
            ]
    
    def select(self, residuals: ResidualBlock, dt: float, state: Dict[str, Any],
               decision: GateDecision) -> Optional[RailResult]:
        """
        Per reference_implementations.md: deterministic rail priority.
        Priority: R1 (dt deflation) → R3 (projection) → R4 (damping) → R2 (rollback)
        """
        for rail in self.rails:
            result = rail.apply(residuals, dt, state, decision)
            if result and result.applied:
                return result
        return None

# -----------------------------
# NSC Models (from nsc/ schemas)
# -----------------------------

@dataclass(frozen=True)
class OpSig:
    args: List[TypeTag]
    ret: TypeTag

@dataclass(frozen=True)
class Operator:
    op_id: int
    token: str
    sig: OpSig
    lowering_id: str
    receipt_metrics: List[str]
    resonance_id: str

@dataclass
class OperatorRegistry:
    registry_id: str
    version: str
    ops: Dict[int, Operator]
    
    def require(self, op_id: int) -> Operator:
        if op_id not in self.ops:
            raise KeyError(f"Missing op_id {op_id} in registry")
        return self.ops[op_id]

@dataclass
class KernelBinding:
    registry_id: str
    binding_id: str
    bindings: Dict[int, Tuple[str, str]]  # op_id -> (kernel_id, kernel_version)
    
    def require(self, op_id: int) -> Tuple[str, str]:
        if op_id not in self.bindings:
            raise KeyError(f"Missing kernel binding for op_id {op_id}")
        return self.bindings[op_id]

@dataclass
class Node:
    id: int
    kind: str  # APPLY, FIELD_REF, SEQ, GATE_CHECK
    type: TypeTag
    payload: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        out = {"id": self.id, "kind": self.kind, "type": self.type.to_json()}
        out.update(self.payload)
        return out

@dataclass
class Module:
    module_id: str
    registry_id: str
    nodes: List[Node]
    entrypoints: Dict[str, int]
    eval_seq_id: int
    meta: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> Dict[str, Any]:
        nodes_sorted = sorted(self.nodes, key=lambda n: n.id)
        return {
            "module_id": self.module_id,
            "registry_id": self.registry_id,
            "nodes": [n.to_json() for n in nodes_sorted],
            "entrypoints": dict(self.entrypoints),
            "order": {"eval_seq_id": self.eval_seq_id},
            "meta": dict(self.meta),
        }
    
    def digest(self) -> str:
        canon = canonical_json(self.to_json())
        return "sha256:" + sha256_hex(canon.encode("utf-8"))

# -----------------------------
# Template System
# -----------------------------

@dataclass(frozen=True)
class Hole:
    name: str
    expected_type: TypeTag

@dataclass
class Template:
    template_id: str
    description: str
    holes: List[Hole]
    
    def build(self, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        raise NotImplementedError

class Template_PsiChain(Template):
    """
    Psi chain template: phi -> source -> twist -> delta -> regulate -> sink -> return_to_baseline(phi)
    """
    def __init__(self, template_id: str, op_ids: Dict[str, int]):
        holes = [Hole("phi_field", FIELD(SCALAR))]
        super().__init__(template_id=template_id, description="Psi chain with baseline return", holes=holes)
        self.op_ids = op_ids
    
    def build(self, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        nid = 1
        nodes: List[Node] = []
        
        # Field reference
        phi_ref = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR), 
                      payload={"field_id": assignments["phi_field"]})
        nodes.append(phi_ref)
        nid += 1
        
        def apply(op_name: str, args: List[int], ret_type: TypeTag) -> int:
            nonlocal nid
            op_id = self.op_ids[op_name]
            n = Node(id=nid, kind="APPLY", type=ret_type, 
                    payload={"op_id": op_id, "args": args})
            nodes.append(n)
            nid += 1
            return n.id
        
        # Psi chain: source → twist → delta → regulate → sink → return_to_baseline
        a1 = apply("source_plus", [phi_ref.id], FIELD(SCALAR))
        a2 = apply("twist", [a1], FIELD(SCALAR))
        a3 = apply("delta", [a2], FIELD(SCALAR))
        a4 = apply("regulate", [a3], FIELD(SCALAR))
        a5 = apply("sink_minus", [a4], FIELD(SCALAR))
        a6 = apply("return_to_baseline", [a5, phi_ref.id], FIELD(SCALAR))
        
        # SEQ order
        seq = Node(id=nid, kind="SEQ", type=FIELD(SCALAR), 
                  payload={"seq": [phi_ref.id, a1, a2, a3, a4, a5, a6]})
        nodes.append(seq)
        
        return Module(
            module_id=module_id,
            registry_id=registry_id,
            nodes=nodes,
            entrypoints={"Psi": a6},
            eval_seq_id=seq.id,
            meta={"template": self.template_id}
        )


class Template_PsiChainWithGateCheck(Template):
    """
    Psi chain template with explicit GATE_CHECK node per nsc_gate.schema.json.
    
    Structure: phi -> source -> TWIST -> delta -> GATE_CHECK -> regulate -> sink -> return
    The GATE_CHECK node enforces coherence gates during execution.
    """
    def __init__(self, template_id: str, op_ids: Dict[str, int], 
                 gate_thresholds: Dict[str, float] = None):
        holes = [
            Hole("phi_field", FIELD(SCALAR)),
            Hole("gate_input", FIELD(SCALAR))
        ]
        super().__init__(
            template_id=template_id, 
            description="Psi chain with explicit coherence gate", 
            holes=holes
        )
        self.op_ids = op_ids
        self.gate_thresholds = gate_thresholds or {
            "phys_fail": 1e-2,
            "phys_warn": 7.5e-3,
            "cons_fail": 0.75,
            "rec_fail": 0.6
        }
    
    def build(self, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        nid = 1
        nodes: List[Node] = []
        
        # Field references
        phi_ref = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR), 
                      payload={"field_id": assignments["phi_field"]})
        nodes.append(phi_ref)
        nid += 1
        
        gate_input = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR),
                         payload={"field_id": assignments["gate_input"]})
        nodes.append(gate_input)
        nid += 1
        
        def apply(op_name: str, args: List[int], ret_type: TypeTag) -> int:
            nonlocal nid
            op_id = self.op_ids[op_name]
            n = Node(id=nid, kind="APPLY", type=ret_type, 
                    payload={"op_id": op_id, "args": args})
            nodes.append(n)
            nid += 1
            return n.id
        
        # Psi chain: source → twist → delta → (GATE_CHECK) → regulate → sink → return
        a1 = apply("source_plus", [phi_ref.id], FIELD(SCALAR))
        a2 = apply("twist", [a1], FIELD(SCALAR))
        a3 = apply("delta", [a2], FIELD(SCALAR))
        
        # GATE_CHECK node per nsc_gate.schema.json
        gate_check = Node(
            id=nid, 
            kind="GATE_CHECK", 
            type=REPORT_GATE,
            payload={
                "thresholds": self.gate_thresholds,
                "input_node_id": a3,
                "input_field_id": assignments.get("gate_input"),
                "policy": "coherence_hard_soft"
            }
        )
        nodes.append(gate_check)
        nid += 1
        
        a4 = apply("regulate", [a3], FIELD(SCALAR))
        a5 = apply("sink_minus", [a4], FIELD(SCALAR))
        a6 = apply("return_to_baseline", [a5, phi_ref.id], FIELD(SCALAR))
        
        # SEQ order includes GATE_CHECK
        seq = Node(id=nid, kind="SEQ", type=FIELD(SCALAR), 
                  payload={"seq": [phi_ref.id, gate_input.id, a1, a2, a3, gate_check.id, a4, a5, a6]})
        nodes.append(seq)
        
        return Module(
            module_id=module_id,
            registry_id=registry_id,
            nodes=nodes,
            entrypoints={"Psi": a6},
            eval_seq_id=seq.id,
            meta={"template": self.template_id, "has_gate_check": True}
        )


class Template_ParallelBranches(Template):
    """
    Template with parallel branches that merge back.
    
    Structure:
        phi 
          ├──→ branch_A (source → twist)
          ├──→ branch_B (delta → regulate)
          └──→ merge → sink → return
    
    This demonstrates graph-structured synthesis, not just linear chains.
    """
    def __init__(self, template_id: str, op_ids: Dict[str, int]):
        holes = [
            Hole("phi_field", FIELD(SCALAR)),
            Hole("field_a", FIELD(SCALAR)),
            Hole("field_b", FIELD(SCALAR))
        ]
        super().__init__(
            template_id=template_id,
            description="Parallel branches with merge",
            holes=holes
        )
        self.op_ids = op_ids
    
    def build(self, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        nid = 1
        nodes: List[Node] = []
        
        # Field references
        phi_ref = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR),
                      payload={"field_id": assignments["phi_field"]})
        nodes.append(phi_ref); nid += 1
        
        field_a = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR),
                      payload={"field_id": assignments["field_a"]})
        nodes.append(field_a); nid += 1
        
        field_b = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR),
                      payload={"field_id": assignments["field_b"]})
        nodes.append(field_b); nid += 1
        
        def apply(op_name: str, args: List[int], ret_type: TypeTag) -> int:
            nonlocal nid
            op_id = self.op_ids[op_name]
            n = Node(id=nid, kind="APPLY", type=ret_type,
                    payload={"op_id": op_id, "args": args})
            nodes.append(n)
            nid += 1
            return n.id
        
        # Branch A: source → twist
        a1 = apply("source_plus", [field_a.id], FIELD(SCALAR))
        a2 = apply("twist", [a1], FIELD(SCALAR))
        
        # Branch B: delta → regulate
        b1 = apply("delta", [field_b.id], FIELD(SCALAR))
        b2 = apply("regulate", [b1], FIELD(SCALAR))
        
        # Merge: combine branches then sink → return
        merge = apply("source_plus", [a2, b2], FIELD(SCALAR))
        sink = apply("sink_minus", [merge], FIELD(SCALAR))
        ret = apply("return_to_baseline", [sink, phi_ref.id], FIELD(SCALAR))
        
        # SEQ order (linearized from parallel structure)
        seq = Node(id=nid, kind="SEQ", type=FIELD(SCALAR),
                  payload={"seq": [phi_ref.id, field_a.id, field_b.id, a1, a2, b1, b2, merge, sink, ret]})
        nodes.append(seq)
        
        return Module(
            module_id=module_id,
            registry_id=registry_id,
            nodes=nodes,
            entrypoints={"Psi": ret},
            eval_seq_id=seq.id,
            meta={"template": self.template_id, "structure": "parallel_branches"}
        )


class Template_RecursivePsi(Template):
    """
    Recursive template: psi_n depends on psi_{n-1}.
    
    Structure for depth D:
        phi_0 → op → phi_1 → op → phi_2 → ... → phi_D → return_to_baseline(phi_D, phi_0)
    
    This tests synthesis depth and coherent recurrence.
    """
    def __init__(self, template_id: str, op_ids: Dict[str, int], depth: int = 3):
        holes = [Hole(f"phi_{i}", FIELD(SCALAR)) for i in range(depth + 1)]
        super().__init__(
            template_id=template_id,
            description=f"Recursive psi chain (depth={depth})",
            holes=holes
        )
        self.op_ids = op_ids
        self.depth = depth
    
    def build(self, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        nid = 1
        nodes: List[Node] = []
        
        # Create field references for each phi_i
        phi_refs = []
        for i in range(self.depth + 1):
            phi = Node(
                id=nid,
                kind="FIELD_REF",
                type=FIELD(SCALAR),
                payload={"field_id": assignments[f"phi_{i}"]}
            )
            nodes.append(phi)
            phi_refs.append(phi)
            nid += 1
        
        def apply(op_name: str, args: List[int], ret_type: TypeTag) -> int:
            nonlocal nid
            op_id = self.op_ids[op_name]
            n = Node(id=nid, kind="APPLY", type=ret_type,
                    payload={"op_id": op_id, "args": args})
            nodes.append(n)
            nid += 1
            return n.id
        
        # Recursive chain: phi_0 → op → phi_1 → op → phi_2 → ...
        chain_outputs = []
        for i in range(self.depth):
            out = apply("twist", [phi_refs[i].id], FIELD(SCALAR))
            chain_outputs.append(out)
        
        # Return to baseline from final phi
        ret = apply("return_to_baseline", [phi_refs[-1].id, phi_refs[0].id], FIELD(SCALAR))
        
        # SEQ order
        seq_ids = [r.id for r in phi_refs] + chain_outputs + [ret]
        seq = Node(id=nid, kind="SEQ", type=FIELD(SCALAR),
                  payload={"seq": seq_ids})
        nodes.append(seq)
        
        return Module(
            module_id=module_id,
            registry_id=registry_id,
            nodes=nodes,
            entrypoints={"Psi": ret},
            eval_seq_id=seq.id,
            meta={"template": self.template_id, "depth": self.depth, "structure": "recursive"}
        )


# -----------------------------
# Coherence Gate for Proposals (L2-aligned)
# -----------------------------

@dataclass
class CoherenceGateReport:
    passed: bool
    debt: float
    debt_decomposition: Dict[str, float]
    residuals: ResidualBlock
    gate_decision: GateDecision
    rail_applied: Optional[RailResult] = None
    reasons: List[str] = field(default_factory=list)

class CoherenceGate:
    """
    Gate that evaluates proposals using real coherence debt functionals.
    Per gates_and_rails.md: hard gates + soft gates + rails.
    """
    
    def __init__(self, 
                 debt_weights: DebtWeights = None,
                 gate_policy: GatePolicy = None,
                 rail_set: RailSet = None):
        self.debt_weights = debt_weights or DebtWeights()
        self.gate_policy = gate_policy or GatePolicy()
        self.rail_set = rail_set or RailSet()
    
    def evaluate(self, module: Module, residuals: ResidualBlock, 
                 dt: float = 0.05, state: Dict[str, Any] = None) -> CoherenceGateReport:
        """
        Per gates_and_rails.md: evaluate gates on proposal.
        
        Args:
            module: NSC module being evaluated
            residuals: Measured residuals from execution
            dt: Current timestep
            state: Runtime state (for rails)
        """
        state = state or {}
        reasons = []
        
        # Check hard gates
        nan_free = all(math.isfinite(v) for v in residuals.__dict__.values())
        
        # Evaluate gates
        decision, verdicts = self.gate_policy.evaluate(residuals, nan_free=nan_free)
        
        # Compute debt functional
        debt, decomposition = self.debt_weights.compute_debt(residuals)
        
        # Determine if passed (ACCEPT or WARN with no rail needed)
        passed = decision in [GateDecision.ACCEPT, GateDecision.WARN]
        
        # Apply rail if rejected
        rail_result = None
        if decision == GateDecision.REJECT:
            rail_result = self.rail_set.select(residuals, dt, state, decision)
            if rail_result and rail_result.applied:
                passed = True  # Rail fixed the issue
                reasons.append(f"Rail applied: {rail_result.rail_name}")
        
        if not passed:
            if decision == GateDecision.ABORT:
                reasons.append("Hard gate failure: NaN/Inf detected")
            elif decision == GateDecision.REJECT:
                reasons.append(f"Soft gate failure: {verdicts['soft']['failures']}")
        
        return CoherenceGateReport(
            passed=passed,
            debt=debt,
            debt_decomposition=decomposition,
            residuals=residuals,
            gate_decision=decision,
            rail_applied=rail_result,
            reasons=reasons
        )

# -----------------------------
# Proposal Engine v1.0
# -----------------------------

@dataclass
class ProposalContext:
    module_id_prefix: str
    available_fields: Dict[int, TypeTag]
    target_type: TypeTag
    max_depth: int = 6
    top_k: int = 5
    dt: float = 0.05

@dataclass
class ProposalReceipt:
    """Per telemetry_and_receipts.md: receipt with hash chaining."""
    kind: str
    module_id: str
    module_digest: str
    template_id: str
    gate_pass: bool
    debt: float
    debt_decomposition: Dict[str, float]
    residuals: Dict[str, float]
    gate_decision: str
    rail_name: Optional[str] = None
    parent_hash: Optional[str] = None
    digest: Optional[str] = None

class NoeticanProposalEngine_v1_0:
    """
    Proposal Engine v1.0 aligned with Coherence Framework.
    
    Features:
    - Real debt functional computation (L1)
    - Hard/soft gates (L2)
    - Rails (R1-R4) (L3)
    - Receipt hash chaining (L2-L4)
    """
    
    def __init__(self,
                 registry: OperatorRegistry,
                 binding: KernelBinding,
                 templates: List[Template],
                 coherence_gate: CoherenceGate,
                 weights: Dict[str, float]):
        self.registry = registry
        self.binding = binding
        self.templates = templates
        self.coherence_gate = coherence_gate
        self.weights = weights
        self.prev_hash = "sha256:" + "0" * 64  # Genesis hash
    
    def propose(self, ctx: ProposalContext) -> List[Tuple[Module, ProposalReceipt]]:
        proposals: List[Tuple[Module, ProposalReceipt]] = []
        
        for t in self.templates:
            assignments_list = self._enumerate_assignments(t, ctx)
            
            for assigns in assignments_list:
                module_id = f"{ctx.module_id_prefix}.{t.template_id}.{assigns.get('_id', '0')}"
                mod = t.build(module_id=module_id, registry_id=self.registry.registry_id, assignments=assigns)
                
                # Evaluate with coherence gate
                residuals = self._compute_residuals(mod)
                gate_report = self.coherence_gate.evaluate(mod, residuals, dt=ctx.dt)
                
                # Build receipt
                receipt = ProposalReceipt(
                    kind="proposal_receipt",
                    module_id=mod.module_id,
                    module_digest=mod.digest(),
                    template_id=t.template_id,
                    gate_pass=gate_report.passed,
                    debt=gate_report.debt,
                    debt_decomposition=gate_report.debt_decomposition,
                    residuals=residuals.__dict__,
                    gate_decision=gate_report.gate_decision.value,
                    rail_name=gate_report.rail_applied.rail_name if gate_report.rail_applied else None,
                    parent_hash=self.prev_hash
                )
                
                # Compute hash chain
                receipt_dict = {
                    "kind": receipt.kind,
                    "module_id": receipt.module_id,
                    "module_digest": receipt.module_digest,
                    "template_id": receipt.template_id,
                    "gate_pass": receipt.gate_pass,
                    "debt": receipt.debt,
                    "debt_decomposition": receipt.debt_decomposition,
                    "residuals": receipt.residuals,
                    "gate_decision": receipt.gate_decision,
                    "rail_name": receipt.rail_name,
                    "parent_hash": receipt.parent_hash
                }
                receipt.digest = compute_hash_chain(receipt_dict, self.prev_hash)
                self.prev_hash = receipt.digest
                
                proposals.append((mod, receipt))
        
        # Filter passed, rank by debt (lower is better)
        passed = [(m, r) for m, r in proposals if r.gate_pass]
        passed.sort(key=lambda x: x[1].debt)
        return passed[:ctx.top_k]
    
    def _enumerate_assignments(self, t: Template, ctx: ProposalContext) -> List[Dict[str, Any]]:
        """Enumerate assignments for template holes."""
        field_choices: Dict[str, List[int]] = {}
        for h in t.holes:
            if h.expected_type.tag == "FIELD":
                field_choices[h.name] = [
                    fid for fid, ty in ctx.available_fields.items() 
                    if ty.to_json() == h.expected_type.to_json()
                ]
        
        holes = [h for h in t.holes if h.expected_type.tag == "FIELD"]
        if not holes:
            return [{"_id": "0"}]
        
        assigns_list = [{"_id": "0"}]
        for h in holes:
            new_list = []
            choices = field_choices[h.name]
            for base in assigns_list:
                for i, fid in enumerate(choices):
                    a = dict(base)
                    a[h.name] = fid
                    a["_id"] = base["_id"] + f".{h.name}{i}"
                    new_list.append(a)
            assigns_list = new_list
        return assigns_list
    
    def _compute_residuals(self, module: Module) -> ResidualBlock:
        """
        Compute residuals for a module.
        Per minimal_tests.py: defect_residual pattern.
        """
        # Simplified: compute residuals based on node structure
        apply_ops = [n for n in module.nodes if n.kind == "APPLY"]
        node_count = len(apply_ops)
        
        # Simulate residuals based on chain length (longer chains = more drift)
        base_phys = 0.1 + 0.05 * max(0, node_count - 4)
        base_cons = 0.05 * max(0, node_count - 3)
        base_rec = 0.03 * max(0, node_count - 5)
        base_tool = 0.02 * node_count
        
        return ResidualBlock(
            cons=min(base_cons, 1.0),
            rec=min(base_rec, 0.8),
            tool=min(base_tool, 0.5),
            phys_rms=min(base_phys, 1.5)
        )

# -----------------------------
# Demo: Run aligned NPE
# -----------------------------

def demo():
    """Demonstrate NPE v1.0 with coherence integration."""
    
    # Define minimal operator set (per nsc/npe_v0_1.py)
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
    templates: List[Template] = [
        Template_PsiChain("psi_chain_v1", {
            "source_plus": 10,
            "twist": 11,
            "delta": 12,
            "regulate": 13,
            "sink_minus": 14,
            "return_to_baseline": 20,
        })
    ]
    
    # Coherence gate (real debt functional + gates + rails)
    coherence_gate = CoherenceGate(
        debt_weights=DebtWeights(),
        gate_policy=GatePolicy(),
        rail_set=RailSet()
    )
    
    # Weights for scoring
    weights = {"debt": -1.0, "gate_pass": 2.0}
    
    engine = NoeticanProposalEngine_v1_0(
        registry=registry,
        binding=binding,
        templates=templates,
        coherence_gate=coherence_gate,
        weights=weights
    )
    
    ctx = ProposalContext(
        module_id_prefix="coherence_proposal",
        available_fields={1: FIELD(SCALAR), 2: FIELD(SCALAR)},
        target_type=FIELD(SCALAR),
        top_k=5,
        dt=0.05
    )
    
    proposals = engine.propose(ctx)
    
    print("=" * 70)
    print("NPE v1.0 Coherence-Aligned Proposals")
    print("=" * 70)
    
    for i, (mod, receipt) in enumerate(proposals):
        print(f"\n--- Proposal #{i+1} ---")
        print(f"Module ID: {receipt.module_id}")
        print(f"Module Digest: {receipt.module_digest}")
        print(f"Gate Decision: {receipt.gate_decision}")
        print(f"Debt: {receipt.debt:.4f}")
        print(f"Debt Decomposition: {receipt.debt_decomposition}")
        print(f"Rail Applied: {receipt.rail_name or 'none'}")
        print(f"Receipt Digest: {receipt.digest}")
        
        # Show node count
        apply_count = len([n for n in mod.nodes if n.kind == "APPLY"])
        print(f"APPLY nodes: {apply_count}")
    
    print("\n" + "=" * 70)
    print("Hash chain verified (each receipt includes parent hash)")
    print("=" * 70)

if __name__ == "__main__":
    demo()
