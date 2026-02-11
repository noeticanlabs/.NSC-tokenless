# NSC Type System

Types are represented as small tagged objects (tokenless).

## 1. Base Types

| Tag | Description |
|-----|-------------|
| `I64` | 64-bit signed integer |
| `F64` | 64-bit floating point |
| `BOOL` | Boolean value |
| `BYTES` | Arbitrary byte sequence |
| `SCALAR` | Semantic scalar value |
| `PHASE` | Semantic phase value |

## 2. Composite Types

| Tag | Parameters | Description |
|-----|------------|-------------|
| `FIELD` | `of: Type` | Typed field reference |
| `REPORT` | `kind: string` | Report object (gate, decode, receipt, event) |
| `ACTION_PLAN` | — | Governance action plan |

### REPORT Kinds

| Kind | Description |
|------|-------------|
| `GATE_REPORT` | Gate check result |
| `DECODE_REPORT` | Glyph decode result |
| `OP_RECEIPT` | Operator application receipt |
| `EVENT` | Braid event |

## 3. Type JSON Representation

### Scalar Types
```json
{ "tag": "I64" }
{ "tag": "F64" }
{ "tag": "BOOL" }
{ "tag": "BYTES" }
{ "tag": "SCALAR" }
{ "tag": "PHASE" }
```

### Field Type
```json
{ "tag": "FIELD", "of": { "tag": "SCALAR" } }
{ "tag": "FIELD", "of": { "tag": "FIELD", "of": { "tag": "SCALAR" } } }
```

### Report Type
```json
{ "tag": "REPORT", "kind": "GATE_REPORT" }
{ "tag": "REPORT", "kind": "OP_RECEIPT" }
```

### Action Plan Type
```json
{ "tag": "ACTION_PLAN" }
```

## 4. Typing Rules

### CONST
```json
{ "id": 1, "kind": "CONST", "type": { "tag": "I64" }, "value": 42 }
```
CONST has declared type matching `type` field.

### FIELD_REF
```json
{ "id": 2, "kind": "FIELD_REF", "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } }, "field_id": 1 }
```
FIELD_REF returns `FIELD(T)` and is typed via environment declaration.

### APPLY
```json
{ "id": 5, "kind": "APPLY", "type": { "tag": "SCALAR" }, "op_id": 10, "args": [1, 3] }
```
APPLY is well-typed if registry defines signature matching arg types.

### SEQ
```json
{ "id": 99, "kind": "SEQ", "type": { "tag": "SCALAR" }, "seq": [1, 2, 3, 4] }
```
SEQ returns the type of its last node.

### GATE_CHECK
```json
{ "id": 20, "kind": "GATE_CHECK", "type": { "tag": "REPORT", "kind": "GATE_REPORT" }, "gate_id": 1, "snapshot": 7 }
```
GATE_CHECK returns `REPORT(GATE_REPORT)`.

### GOVERN
```json
{ "id": 30, "kind": "GOVERN", "type": { "tag": "ACTION_PLAN" }, "history": 9001, "reports": [20] }
```
GOVERN returns `ACTION_PLAN`.

### EMIT
```json
{ "id": 40, "kind": "EMIT", "type": { "tag": "REPORT", "kind": "EVENT" }, "emit_kind": "gate_report", "payload_ref": 20 }
```
EMIT returns `REPORT(EVENT)` or `BYTES` digest (implementation option).

## 5. No Implicit Coercions

NSC forbids implicit coercions. If conversion is needed, it is an operator
with its own `op_id`. For example:

```json
{ "id": 100, "kind": "APPLY", "op_id": 99, "args": [1] }  // i64_to_f64
```

## 6. Type Equality

Two types are equal iff their normalized representation matches exactly.

FIELD(T) compares by its parameter type T:
```json
// These are equal:
{ "tag": "FIELD", "of": { "tag": "SCALAR" } }
{ "tag": "FIELD", "of": { "tag": "SCALAR" } }

// These are NOT equal:
{ "tag": "FIELD", "of": { "tag": "SCALAR" } }
{ "tag": "FIELD", "of": { "tag": "PHASE" } }
```
