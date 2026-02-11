# Safe Extensions (Normative) — NSC Tokenless v1.1

Extensions are allowed, but drift is forbidden.

## Allowed extensions
- Add new operators by allocating new op_id values in a new registry version.
- Add new kernels with new (kernel_id, kernel_version).
- Add new node kinds only by publishing a new language version and schemas.

## Forbidden extensions
- Changing an operator's type signature under the same op_id.
- Reusing a kernel_id with different semantics without bumping kernel_version.
- Executing without emitting mandatory receipts.
- Treating control outcomes (REJECT) as operators.

## Extension discipline
Any extension MUST:
- declare new schema version if structure changes
- provide conformance test vectors
- provide replay implications (what must be recorded)
