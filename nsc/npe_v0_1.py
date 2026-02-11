from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import hashlib
import json
import math
import random

# -----------------------------
# Canonical JSON + hashing
# -----------------------------

def canonical_json(obj: Any) -> str:
    # canonical-json-v1: sorted keys, no whitespace
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha256_digest(obj: Any) -> str:
    s = canonical_json(obj).encode("utf-8")
    return "sha256:" + hashlib.sha256(s).hexdigest()

# -----------------------------
# Types (tokenless tags)
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
PHASE  = TypeTag("PHASE")
def FIELD(of: TypeTag) -> TypeTag:
    return TypeTag("FIELD", of=of)

REPORT_GATE   = TypeTag("REPORT", kind="GATE_REPORT")
REPORT_PROPOSAL = TypeTag("REPORT", kind="PROPOSAL_RECEIPT")

# -----------------------------
# Operator registry
# -----------------------------

@dataclass(frozen=True)
class OpSig:
    args: List[TypeTag]
    ret: TypeTag

@dataclass(frozen=True)
class Operator:
    op_id: int
    token: str                 # human label only
    sig: OpSig
    lowering_id: str
    receipt_metrics: List[str] # required fields for op receipts (for execution)
    resonance_id: str          # placeholder id (still tokenless)

@dataclass
class OperatorRegistry:
    registry_id: str
    version: str
    ops: Dict[int, Operator]

    def require(self, op_id: int) -> Operator:
        if op_id not in self.ops:
            raise KeyError(f"Missing op_id {op_id} in registry")
        return self.ops[op_id]

# -----------------------------
# Kernel binding (op_id -> kernel)
# -----------------------------

@dataclass
class KernelBinding:
    registry_id: str
    binding_id: str
    bindings: Dict[int, Tuple[str, str]]  # op_id -> (kernel_id, kernel_version)

    def require(self, op_id: int) -> Tuple[str, str]:
        if op_id not in self.bindings:
            raise KeyError(f"Missing kernel binding for op_id {op_id}")
        return self.bindings[op_id]

# -----------------------------
# NSC Node + Module model
# -----------------------------

@dataclass
class Node:
    id: int
    kind: str
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
        return sha256_digest(self.to_json())

# -----------------------------
# Template system (tokenless skeletons)
# -----------------------------

@dataclass(frozen=True)
class Hole:
    name: str
    expected_type: TypeTag

@dataclass
class Template:
    """
    A template is a parametric graph with typed holes.
    Filling produces a valid NSC module candidate.
    """
    template_id: str
    description: str
    holes: List[Hole]

    # function that builds nodes when holes are assigned
    def build(self, *, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        raise NotImplementedError

# -----------------------------
# Gates + scoring
# -----------------------------

@dataclass
class GateReport:
    passed: bool
    reasons: List[str]
    metrics: Dict[str, float]

@dataclass
class Proposal:
    module: Module
    score: float
    gate_report: GateReport
    receipt: Dict[str, Any]

class Gate:
    def check(self, module: Module, registry: OperatorRegistry, binding: KernelBinding) -> GateReport:
        raise NotImplementedError

class Gate_RegistryAndBinding(Gate):
    def check(self, module: Module, registry: OperatorRegistry, binding: KernelBinding) -> GateReport:
        reasons = []
        passed = True
        # Ensure APPLY nodes use known op_id and have kernel bindings
        for n in module.nodes:
            if n.kind == "APPLY":
                op_id = n.payload.get("op_id")
                try:
                    registry.require(op_id)
                except KeyError as e:
                    passed = False
                    reasons.append(str(e))
                try:
                    binding.require(op_id)
                except KeyError as e:
                    passed = False
                    reasons.append(str(e))
        return GateReport(passed=passed, reasons=reasons, metrics={})

class Gate_Typecheck(Gate):
    def check(self, module: Module, registry: OperatorRegistry, binding: KernelBinding) -> GateReport:
        # minimal tokenless typecheck: APPLY arg types match signature
        node_by_id = {n.id: n for n in module.nodes}
        reasons = []
        passed = True

        def same_type(a: TypeTag, b: TypeTag) -> bool:
            return a.to_json() == b.to_json()

        for n in module.nodes:
            if n.kind != "APPLY":
                continue
            op = registry.require(n.payload["op_id"])
            args = n.payload.get("args", [])
            if len(args) != len(op.sig.args):
                passed = False
                reasons.append(f"Node {n.id}: arg count mismatch for op {op.op_id}")
                continue
            for i, (arg_id, exp_t) in enumerate(zip(args, op.sig.args)):
                if arg_id not in node_by_id:
                    passed = False
                    reasons.append(f"Node {n.id}: missing arg node {arg_id}")
                    continue
                got_t = node_by_id[arg_id].type
                if not same_type(got_t, exp_t):
                    passed = False
                    reasons.append(
                        f"Node {n.id}: arg[{i}] type mismatch: got {got_t.to_json()} expected {exp_t.to_json()}"
                    )
            # output type check
            if not same_type(n.type, op.sig.ret):
                passed = False
                reasons.append(
                    f"Node {n.id}: return type mismatch: got {n.type.to_json()} expected {op.sig.ret.to_json()}"
                )
        return GateReport(passed=passed, reasons=reasons, metrics={})

class Gate_DeterminismShape(Gate):
    def check(self, module: Module, registry: OperatorRegistry, binding: KernelBinding) -> GateReport:
        # ensures eval_seq_id exists and references a SEQ, and SEQ is non-empty
        reasons = []
        passed = True
        node_by_id = {n.id: n for n in module.nodes}
        if module.eval_seq_id not in node_by_id:
            passed = False
            reasons.append("eval_seq_id missing")
            return GateReport(passed, reasons, {})
        seq = node_by_id[module.eval_seq_id]
        if seq.kind != "SEQ":
            passed = False
            reasons.append("eval_seq_id must reference SEQ node")
        else:
            s = seq.payload.get("seq", [])
            if not isinstance(s, list) or len(s) == 0:
                passed = False
                reasons.append("SEQ.seq must be non-empty")
        return GateReport(passed=passed, reasons=reasons, metrics={})

class Gate_CoherenceHeuristic(Gate):
    """
    Placeholder gate: in real integration, this calls your coherence metrics:
      - residual norms, CFL, budget debt, etc.
    Here we use a deterministic heuristic:
      penalize long sequences and prefer "regulated + return_to_baseline" patterns.
    """
    def __init__(self, prefer_ops: List[int], length_penalty: float = 0.02):
        self.prefer_ops = set(prefer_ops)
        self.length_penalty = length_penalty

    def check(self, module: Module, registry: OperatorRegistry, binding: KernelBinding) -> GateReport:
        apply_ops = [n.payload["op_id"] for n in module.nodes if n.kind == "APPLY"]
        length = len(apply_ops)
        prefer_hits = sum(1 for op_id in apply_ops if op_id in self.prefer_ops)

        # deterministic "coherence score" in [0,1]
        C = 0.75 + 0.05 * prefer_hits - self.length_penalty * max(0, length - 4)
        C = max(0.0, min(1.0, C))

        passed = C >= 0.70
        reasons = [] if passed else [f"coherence score too low: {C:.3f}"]
        return GateReport(passed=passed, reasons=reasons, metrics={"C": C, "len": float(length), "prefer_hits": float(prefer_hits)})

# -----------------------------
# Proposal Engine
# -----------------------------

@dataclass
class ProposalContext:
    module_id_prefix: str
    available_fields: Dict[int, TypeTag]  # field_id -> type
    target_type: TypeTag
    max_depth: int = 6
    top_k: int = 5

class NoeticanProposalEngine:
    def __init__(
        self,
        registry: OperatorRegistry,
        binding: KernelBinding,
        templates: List[Template],
        gates: List[Gate],
        weights: Dict[str, float],
    ):
        self.registry = registry
        self.binding = binding
        self.templates = templates
        self.gates = gates
        self.weights = weights

    def propose(self, ctx: ProposalContext) -> List[Proposal]:
        proposals: List[Proposal] = []
        # Enumerate template fillings (small v0.1)
        for t in self.templates:
            # build assignment space for holes:
            # - if hole expects FIELD(SCALAR), assign a matching field_id
            # - if expects op_id, assign a matching operator id (by signature constraints in template)
            assignments_list = self._enumerate_assignments(t, ctx)
            for assigns in assignments_list:
                module_id = f"{ctx.module_id_prefix}.{t.template_id}.{assigns.get('_id','0')}"
                mod = t.build(module_id=module_id, registry_id=self.registry.registry_id, assignments=assigns)

                gate_report = self._run_gates(mod)
                score = self._score(gate_report)

                receipt = {
                    "kind": "proposal_receipt",
                    "module_id": mod.module_id,
                    "module_digest": mod.digest(),
                    "template_id": t.template_id,
                    "assignments": {k: v for k, v in assigns.items() if not k.startswith("_")},
                    "gate_pass": gate_report.passed,
                    "gate_reasons": gate_report.reasons,
                    "metrics": gate_report.metrics,
                }
                receipt["digest"] = self._receipt_digest(receipt)

                proposals.append(Proposal(module=mod, score=score, gate_report=gate_report, receipt=receipt))

        # keep only passed, ranked
        passed = [p for p in proposals if p.gate_report.passed]
        passed.sort(key=lambda p: p.score, reverse=True)
        return passed[: ctx.top_k]

    def _receipt_digest(self, receipt: Dict[str, Any]) -> str:
        r = dict(receipt)
        r.pop("digest", None)
        return sha256_digest(r)

    def _run_gates(self, mod: Module) -> GateReport:
        reasons = []
        metrics: Dict[str, float] = {}
        passed = True
        for g in self.gates:
            gr = g.check(mod, self.registry, self.binding)
            if not gr.passed:
                passed = False
                reasons.extend(gr.reasons)
            for k, v in gr.metrics.items():
                metrics[k] = v
        return GateReport(passed=passed, reasons=reasons, metrics=metrics)

    def _score(self, gr: GateReport) -> float:
        # linear scorer over known metrics
        s = 0.0
        for k, w in self.weights.items():
            s += w * float(gr.metrics.get(k, 0.0))
        # harsh penalty if failed (but we filter failures anyway)
        if not gr.passed:
            s -= 1.0
        return s

    def _enumerate_assignments(self, t: Template, ctx: ProposalContext) -> List[Dict[str, Any]]:
        # Small deterministic enumeration: assign each FIELD hole with any matching field_id
        field_choices: Dict[str, List[int]] = {}
        for h in t.holes:
            if h.expected_type.tag == "FIELD":
                field_choices[h.name] = [fid for fid, ty in ctx.available_fields.items() if ty.to_json() == h.expected_type.to_json()]
            else:
                field_choices[h.name] = []

        # cartesian product over field holes (v0.1)
        holes = [h for h in t.holes if h.expected_type.tag == "FIELD"]
        if not holes:
            return [{"_id": "0"}]

        assigns_list: List[Dict[str, Any]] = [{"_id": "0"}]
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

# -----------------------------
# Concrete Templates
# -----------------------------

class Template_PsiChain(Template):
    """
    Build: phi -> source -> twist -> delta -> regulate -> sink -> return_to_baseline(phi)
    This is tokenless: op_ids and field_ids are numeric; human labels only in meta.
    """
    def __init__(self, template_id: str, op_ids: Dict[str, int]):
        holes = [Hole("phi_field", FIELD(SCALAR))]
        super().__init__(template_id=template_id, description="Psi chain with baseline return", holes=holes)
        self.op_ids = op_ids

    def build(self, *, module_id: str, registry_id: str, assignments: Dict[str, Any]) -> Module:
        nid = 1
        nodes: List[Node] = []

        # field ref
        phi_ref = Node(id=nid, kind="FIELD_REF", type=FIELD(SCALAR), payload={"field_id": assignments["phi_field"]})
        nodes.append(phi_ref); nid += 1

        def apply(op_name: str, args: List[int], ret_type: TypeTag) -> int:
            nonlocal nid
            op_id = self.op_ids[op_name]
            n = Node(id=nid, kind="APPLY", type=ret_type, payload={"op_id": op_id, "args": args})
            nodes.append(n)
            nid += 1
            return n.id

        a1 = apply("source_plus", [phi_ref.id], FIELD(SCALAR))
        a2 = apply("twist", [a1], FIELD(SCALAR))
        a3 = apply("delta", [a2], FIELD(SCALAR))
        a4 = apply("regulate", [a3], FIELD(SCALAR))
        a5 = apply("sink_minus", [a4], FIELD(SCALAR))
        a6 = apply("return_to_baseline", [a5, phi_ref.id], FIELD(SCALAR))

        # SEQ order
        seq = Node(id=nid, kind="SEQ", type=FIELD(SCALAR), payload={"seq": [phi_ref.id, a1, a2, a3, a4, a5, a6]})
        nodes.append(seq)

        return Module(
            module_id=module_id,
            registry_id=registry_id,
            nodes=nodes,
            entrypoints={"Psi": a6},
            eval_seq_id=seq.id,
            meta={"template": self.template_id}
        )

# -----------------------------
# Demo: build a minimal registry and run proposals
# -----------------------------

def demo() -> None:
    # Define minimal operator set with fixed op_ids
    ops = {
        10: Operator(10, "source_plus", OpSig([FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.source_plus", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after"], "res.op.source_plus"),
        11: Operator(11, "twist",       OpSig([FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.curvature_twist", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after"], "res.op.twist"),
        12: Operator(12, "delta",       OpSig([FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.delta_transform", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after"], "res.op.delta"),
        13: Operator(13, "regulate",    OpSig([FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.regulate_circle", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after"], "res.op.regulate"),
        14: Operator(14, "sink_minus",  OpSig([FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.sink_minus", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after"], "res.op.sink_minus"),
        20: Operator(20, "return_to_baseline", OpSig([FIELD(SCALAR), FIELD(SCALAR)], FIELD(SCALAR)), "nsc.apply.return_to_baseline", ["kernel_id","kernel_version","digest_in","digest_out","C_before","C_after","params"], "res.op.return_baseline"),
    }
    registry = OperatorRegistry("noetican.nsc.registry.v1_1", "1.1.0", ops)

    binding = KernelBinding(
        registry_id=registry.registry_id,
        binding_id="binding.demo.v1",
        bindings={
            10: ("kern.source_plus.v1","1.0.0"),
            11: ("kern.twist.v1","1.0.0"),
            12: ("kern.delta.v1","1.0.0"),
            13: ("kern.regulate.v1","1.0.0"),
            14: ("kern.sink_minus.v1","1.0.0"),
            20: ("kern.return_baseline.linear_blend","1.0.0"),
        }
    )

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

    gates: List[Gate] = [
        Gate_RegistryAndBinding(),
        Gate_Typecheck(),
        Gate_DeterminismShape(),
        Gate_CoherenceHeuristic(prefer_ops=[13, 20], length_penalty=0.015),
    ]

    weights = {"C": 1.0, "prefer_hits": 0.1, "len": -0.02}

    engine = NoeticanProposalEngine(registry, binding, templates, gates, weights)

    ctx = ProposalContext(
        module_id_prefix="proposal",
        available_fields={
            1: FIELD(SCALAR),
            2: FIELD(SCALAR),
        },
        target_type=FIELD(SCALAR),
        top_k=5
    )

    props = engine.propose(ctx)
    for i, p in enumerate(props):
        print(f"\n=== Proposal #{i} score={p.score:.3f} module_digest={p.module.digest()} ===")
        print("metrics:", p.gate_report.metrics)
        print("receipt_digest:", p.receipt["digest"])
        print(canonical_json(p.module.to_json()))

if __name__ == "__main__":
    demo()
