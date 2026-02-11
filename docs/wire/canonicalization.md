# Canonicalization (Normative) — NSC Tokenless v1.1

Canonicalization defines how structured objects become bytes for hashing and replay.

## 1) Canonical JSON profile (canonical-json-v1)
Canonical JSON MUST follow:
- UTF-8 encoding
- Object keys sorted lexicographically by Unicode codepoint
- Arrays preserve order
- No insignificant whitespace
- Numbers MUST be finite (no NaN/Inf)
- Numbers MUST be normalized:
  - No trailing `.0` if integer
  - No leading `+`
  - No exponent if avoidable; if exponent used, must be minimal form

If a runtime cannot guarantee numeric normalization, it MUST either:
- declare a different canonical profile and publish it, OR
- be considered nonconforming for digest claims.

## 2) Canonical payload sets
Canonicalization applies to:
- ModuleDigest payloads
- OperatorReceipt payloads
- GateReport payloads (when used)
- ActionPlan payloads (when used)
- DecodeReport payloads (when used)
- BraidEvent envelopes (payload + envelope fields)
- ReplayReport outputs (verifier artifacts)

## 3) Unknown fields
For objects validated against schemas, unknown fields are allowed only if schema permits
(additionalProperties true). Unknown fields, if present, MUST be included in canonicalization,
otherwise digest mismatch will occur.

## 4) Prohibited values
Canonical payloads MUST NOT contain:
- NaN, +Inf, -Inf
- platform-dependent binary blobs without declared encoding
- timestamps unless explicitly part of semantics (rare; discouraged)
