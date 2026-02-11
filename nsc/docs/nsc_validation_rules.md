# NSC Conformance Validation Rules (MUST-level)

A module is conforming iff it satisfies all MUST-level rules below.

## 1. Module Validation (MUST)

The module object MUST validate against `schemas/nsc_module.schema.json`.

### Required Fields
```json
{
  "module_id": "...",      // MUST be non-empty string
  "registry_id": "...",    // MUST be non-empty string
  "types": { ... },        // MUST be object
  "nodes": [ ... ],        // MUST be non-empty array
  "entrypoints": { ... },  // MUST be object
  "order": { ... }         // MUST be object with eval_seq_id
}
```

## 2. Node Validation (MUST)

Every node MUST validate against `schemas/nsc_node.schema.json`.

### Required Fields Per Kind

| Kind | Required Fields |
|------|-----------------|
| CONST | `id`, `kind`, `type`, `value` |
| FIELD_REF | `id`, `kind`, `type`, `field_id` |
| APPLY | `id`, `kind`, `type`, `op_id`, `args` |
| SEQ | `id`, `kind`, `type`, `seq` |
| GATE_CHECK | `id`, `kind`, `type`, `gate_id`, `snapshot` |
| GOVERN | `id`, `kind`, `type`, `history`, `reports` |
| EMIT | `id`, `kind`, `type`, `emit_kind`, `payload_ref` |
| STEP | `id`, `kind`, `dt_ref`, `rhs_ref`, `method_id` |

## 3. Node ID Resolution (MUST)

All referenced node IDs MUST exist:

```json
// args MUST reference existing nodes
{ "op_id": 10, "args": [1, 3, 7] }  // 1, 3, 7 MUST exist

// seq MUST reference existing nodes
{ "kind": "SEQ", "seq": [1, 2, 3, 4] }  // All MUST exist

// payload_ref MUST reference existing node
{ "kind": "EMIT", "payload_ref": 20 }  // 20 MUST exist
```

## 4. Evaluation Order (MUST)

The `order.eval_seq_id` MUST reference a SEQ node:

```json
{ "order": { "eval_seq_id": 99 } }  // 99 MUST be a SEQ node
```

The SEQ node MUST define at least one evaluation path to an entrypoint.

## 5. Operator Registry (MUST)

Every APPLY node MUST reference an `op_id` that exists in the declared registry:

```json
{
  "registry_id": "noetican.nsc.registry.v1",
  "nodes": [
    { "id": 5, "kind": "APPLY", "op_id": 10, "args": [1] }  // op_id 10 MUST exist in registry
  ]
}
```

## 6. Type Checking (MUST)

Type checking MUST succeed:

### CONST Type
```json
{ "kind": "CONST", "type": { "tag": "I64" }, "value": 42 }  // value type MUST match type
```

### APPLY Type
```json
// op_id 10 has signature: FIELD(SCALAR) -> FIELD(SCALAR)
{ "kind": "APPLY", "op_id": 10, "args": [1] }  // arg 1 MUST be FIELD(SCALAR)
```

### SEQ Type
```json
{ "kind": "SEQ", "seq": [1, 2, 3], "type": { "tag": "SCALAR" } }  // type MUST match last node
```

## 7. Emission Validation (MUST)

If a node kind implies emission, emitted objects MUST validate:

| Node Kind | Emitted Schema |
|-----------|---------------|
| GATE_CHECK | `nsc_gate.schema.json` |
| GOVERN | `nsc_action_plan.schema.json` |
| EMIT | `nsc_braid_event.schema.json` |

## 8. Glyph Decode (MUST)

If glyph decoding is present:

- DecodeReport MUST be emitted
- REJECT MUST only be used as control outcome, never as operator

```json
{
  "chosen": { "kind": "REJECT", "control_row_id": 0 }  // Valid: REJECT as control
}

// INVALID:
{ "chosen": { "kind": "GLYPH_ID", "glyph_id": "REJECT" } }  // REJECT as operator
```

## 9. Receipt Requirements (MUST)

Every governed operator application MUST emit an OperatorReceipt with:

- Valid `run_id` and `event_id`
- Correct `thread_id` from allowed values
- Valid `op_id` matching the applied operator
- Valid `digests` (SHA-256 format)
- Valid `delta` with coherence values in [0, 1]

## 10. Braid Event Requirements (MUST)

Every braid event MUST have:

- Unique `event_id` within the run
- Valid `thread_id` from allowed values
- Non-negative `step_index`
- Valid `parents` array (referencing prior events)
- Valid `digest` (SHA-256 of canonical payload)

## Validation Checklist

```
□ Module validates against nsc_module.schema.json
□ All nodes validate against nsc_node.schema.json
□ All referenced node IDs exist
□ order.eval_seq_id references a valid SEQ node
□ All APPLY op_ids exist in registry
□ Type checking succeeds
□ GATE_CHECK emissions validate against nsc_gate.schema.json
□ GOVERN emissions validate against nsc_action_plan.schema.json
□ EMIT emissions validate against nsc_braid_event.schema.json
□ DecodeReport emitted if glyph decoding used
□ REJECT only used as control outcome
□ All OperatorReceipts have required fields
□ All BraidEvents have required fields
```

## Failure Consequences

Failure of any MUST rule means the run is **nonconforming**:
- The runtime MUST reject nonconforming modules
- The runtime MUST NOT execute nonconforming modules
- Nonconforming runs MUST NOT produce receipts
