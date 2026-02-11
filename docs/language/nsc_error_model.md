# NSC Error Model (Normative) — NSC Tokenless v1.1

NSC treats errors as structured, auditable outcomes.

## 1) Error categories
- VALIDATION_ERROR: schema or typecheck failure
- REGISTRY_ERROR: missing operator entry for an op_id used
- KERNEL_BINDING_ERROR: missing kernel binding for required op_id
- NUMERIC_INVALID: NaN/Inf encountered in canonical payload domain or kernel output
- DECODE_REJECT: identity decoding produced REJECT control outcome
- GATE_FAIL: GateReport pass=false
- GOVERNANCE_ABORT: ActionPlan action=ABORT

## 2) Handling rules
### VALIDATION_ERROR
- Execution MUST NOT proceed.
- If an event sink exists, runtime SHOULD emit an audit BraidEvent recording:
  - kind: "validation_error"
  - payload: {reason, schema, path}

### REGISTRY_ERROR / KERNEL_BINDING_ERROR
- Execution MUST NOT proceed for the affected APPLY.
- Runtime MUST emit an audit event if event sink exists:
  - kind: "binding_error" or "registry_error"
  - payload includes op_id and registry_id and missing fields.

### NUMERIC_INVALID
- Runtime MUST trap NaN/Inf (no propagation).
- If governed, runtime MUST produce a GateReport recommending ABORT or a direct ABORT audit event.
- A conforming run MUST record this event (cannot silently clamp without reporting).

### DECODE_REJECT
- Lowering MUST stop for affected segment.
- A DecodeReport event MUST be emitted.

### GATE_FAIL / GOVERNANCE_ABORT
- Must be represented in braid events (GateReport and ActionPlan).
- If action is ABORT, the run must terminate after emitting the relevant events.

## 3) Nonconformance triggers
- Treating REJECT as an operator
- Executing APPLY without OperatorReceipt
- Emitting receipts without correct canonical digests
