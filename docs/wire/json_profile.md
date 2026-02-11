# JSON Wire Profile (Normative) — NSC Tokenless v1.1

NSC objects MAY be transported in JSON. This is a transport, not semantics.

## Constraints
- Node IDs, op_id, field_id, method_id MUST be integers.
- Registry identifiers and kernel identifiers are strings.
- BYTES values must be base64 strings if transported in JSON.

## Validation
Before execution, a runtime MUST validate:
- module against module schema
- registry against registry schema
- kernel binding against kernel binding schema (if used)
- receipts/events against their schemas

## Determinism note
JSON parsing MUST NOT reorder arrays. SEQ order depends on array order.
