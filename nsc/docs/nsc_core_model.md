# NSC Core Model

## 1. Module

An NSC Module contains:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `module_id` | string | Yes | Unique identifier for this module |
| `registry_id` | string | Yes | Operator registry version ID |
| `types` | object | Yes | Type table (type name → type def) |
| `nodes` | array | Yes | Node table (array of node objects) |
| `entrypoints` | object | Yes | Named node IDs for external access |
| `order` | object | Yes | Evaluation order declaration |
| `meta` | object | No | Metadata (field env, comments, etc.) |

### Module Example Structure

```json
{
  "module_id": "psi_minimal",
  "registry_id": "noetican.nsc.registry.v1",
  "types": { "t_field_scalar": { "tag": "FIELD", "of": { "tag": "SCALAR" } } },
  "nodes": [...],
  "entrypoints": { "Psi": 7 },
  "order": { "eval_seq_id": 99 }
}
```

## 2. Node Kinds

NSC defines a minimal complete set of node kinds:

| Kind | Purpose | Output Type |
|------|---------|-------------|
| `CONST` | Typed literal value | Declared type |
| `FIELD_REF` | References field from environment | `FIELD(T)` |
| `APPLY` | Apply operator to arguments | Operator's ret type |
| `SEQ` | Explicit evaluation sequence | Type of last node |
| `GATE_CHECK` | Evaluate gate predicate | `REPORT(GATE_REPORT)` |
| `GOVERN` | Produce action plan from reports | `ACTION_PLAN` |
| `EMIT` | Emit braid events/receipts | `REPORT(EVENT)` or `BYTES` |
| `STEP` | Time-step primitive (optional) | New state |

### Node Fields by Kind

**CONST:**
```json
{ "id": 1, "kind": "CONST", "type": { "tag": "I64" }, "value": 42 }
```

**FIELD_REF:**
```json
{ "id": 1, "kind": "FIELD_REF", "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } }, "field_id": 1 }
```

**APPLY:**
```json
{ "id": 5, "kind": "APPLY", "type": { "tag": "SCALAR" }, "op_id": 10, "args": [1, 3] }
```

**SEQ:**
```json
{ "id": 99, "kind": "SEQ", "type": { "tag": "SCALAR" }, "seq": [1, 2, 3, 4] }
```

**GATE_CHECK:**
```json
{ "id": 20, "kind": "GATE_CHECK", "type": { "tag": "REPORT", "kind": "GATE_REPORT" }, "gate_id": 1, "snapshot": 7 }
```

**GOVERN:**
```json
{ "id": 30, "kind": "GOVERN", "type": { "tag": "ACTION_PLAN" }, "history": 9001, "reports": [20] }
```

**EMIT:**
```json
{ "id": 40, "kind": "EMIT", "type": { "tag": "REPORT", "kind": "EVENT" }, "emit_kind": "gate_report", "payload_ref": 20 }
```

**STEP:**
```json
{ "id": 100, "kind": "STEP", "dt_ref": 50, "rhs_ref": 60, "method_id": 1 }
```

## 3. Evaluation Model

### Sequential Semantics
- Nodes are evaluated in SEQ order at runtime
- SEQ node with `seq: [1,2,3]` means evaluate node 1, then 2, then 3
- The result of SEQ is the result of the last node in the sequence

### Purity Requirements
- APPLY nodes are pure (must not mutate history or state)
- Side effects occur only via EMIT nodes
- STEP produces a new state object, not hidden mutation

### Entrypoints
Named entrypoints allow external systems to invoke specific computations:

```json
{
  "entrypoints": {
    "Psi": 7,
    "GateReport": 20,
    "ActionPlan": 30
  }
}
```

## 4. Runtime Environment

The runtime provides:

| Component | Purpose |
|-----------|---------|
| Initial environment | Field bindings, constants |
| History buffer | Prior receipts/events for GOVERN |
| Kernel library | Deterministic kernels keyed by `(op_id, kernel_id, version)` |
| Gate definitions | Gate predicates and thresholds |

## 5. Module Graph Structure

```
Entry (field_ref) ──▶ APPLY(source_plus) ──▶ APPLY(twist) ──▶ APPLY(delta)
        │                                                                 │
        ▼                                                                 ▼
APPLY(return_to_baseline) ◀─────────────────────────────────────────────┘
        │
        ▼
    Result
```

The SEQ node defines evaluation order: `field_ref → source_plus → twist → delta → return_to_baseline`
