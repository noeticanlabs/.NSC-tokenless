# Canonical Hashing (Normative) — NSC Tokenless v1.1

## 1) Hash function
SHA-256 is used for all digests.

Digest string format:
`sha256:<64 lowercase hex>`

## 2) Module digest
Module digest MUST be computed over the canonical JSON serialization of the module object,
with the following normalization:
- `nodes` MUST be sorted by ascending `id` before hashing (regardless of source order)
- `entrypoints` keys sorted (canonical JSON already does this)
- `meta` included if present (so meta changes affect digest); if you want meta ignored,
  omit it or split it out of the module object.

See `schemas/nsc_module_digest.schema.json`.

## 3) Receipt digests
Every OperatorReceipt MUST contain a `digest` field equal to SHA-256 of canonical JSON
serialization of the receipt object *excluding* the `digest` field itself.
(Implementation rule: compute digest on a copy with `digest` removed or set to null.)

Same rule applies to BraidEvents: compute digest excluding digest field.

## 4) Digest chaining
BraidEvents SHOULD reference parent events in `parents` so the audit trail forms a DAG.
Digest chaining is not strictly required for L0 but is recommended for L1+.

## 5) Field digests
Field digests are runtime-defined but MUST be deterministic and MUST include:
- domain_id or equivalent identity
- shape or lattice metadata
- stable summary statistics (min/max/mean/norm or declared alternative)
- kernel identifiers relevant to producing that field
Field digests MUST NOT include:
- memory addresses
- wall-clock time
- nondeterministic iteration artifacts
