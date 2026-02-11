# CBOR Wire Profile (Optional; Normative if used) — NSC Tokenless v1.1

CBOR may be used for compact transport. Tokenless systems often prefer CBOR.

## Minimum CBOR requirements if used
- RFC 8949 canonical CBOR (definite lengths, canonical map key ordering)
- No NaN/Inf floats
- Integers must be encoded minimally

## Hashing rule
Unless an alternative canonical CBOR hashing profile is published, all digests MUST still be
computed over canonical JSON (canonical-json-v1). CBOR is transport; JSON canonical form is hash basis.

If a runtime hashes CBOR directly, it MUST:
- publish the canonical CBOR hashing profile
- ensure replay verifier reproduces the same bytes
