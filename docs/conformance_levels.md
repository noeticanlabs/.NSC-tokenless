# Conformance Levels (Normative) — NSC Tokenless v1.1

A runtime MAY claim conformance at one of the following levels. Claims must be true.

## L0 — Core Tokenless (Required baseline)
A runtime claiming L0 MUST:
- Validate modules against schemas:
  - `nsc_module.schema.json`
  - `nsc_node.schema.json`
  - `nsc_type.schema.json`
- Resolve operators by `(registry_id, version, op_id)` from an Operator Registry.
- Use a Kernel Binding (or a declared default binding) and record kernel identity in receipts.
- Enforce explicit evaluation order via SEQ nodes.
- Emit OperatorReceipt for every executed APPLY node.
- Canonicalize and hash receipts and module digests per wire docs.

## L1 — Governed
Includes L0 plus MUST:
- Support GateCheck evaluation and GateReport schema.
- Support Governance evaluation and ActionPlan schema.
- Emit GateReport and ActionPlan braid events when those nodes are present.
- Support Replay Verification reporting (ReplayReport schema) for runs it produces.

## L2 — Identity-Aware (Decode-capable)
Includes L1 plus MUST (if decode is used):
- Emit DecodeReport for every decode attempt.
- Treat REJECT as control-only outcome.
- If contextual ambiguity recovery is performed, emit audit events describing it.

A runtime MUST NOT claim L2 if it decodes glyphs but fails to emit DecodeReports.
