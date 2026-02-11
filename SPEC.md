# NSC Tokenless v1.1 Specification (Consolidated)

**Version:** 1.1.0  
**Date:** 2026-02-11  
**Normative References:** JSON Schema Draft 2020-12, SHA-256 (FIPS 180-4), RFC 2119

---

## Table of Contents

1. [Terminology](docs/terminology.md)
2. [Design Goals](docs/design_goals.md)
3. [Conformance Levels](docs/conformance_levels.md)
4. [Wire Layer](docs/wire/)
5. [Language](docs/language/)
6. [Governance](docs/governance/)
7. [Glyphs](docs/glyphs/)
8. [Security](docs/security/)
9. [Schemas](#schemas)
10. [Test Vectors](#test-vectors)

---

## 1 Terminology (Normative)

See [`docs/terminology.md`](docs/terminology.md)

**Core Language Terms:**
- **Tokenless**: No textual token parsing; execution depends on typed structures + numeric IDs
- **Module**: Typed node table + explicit evaluation order + entrypoints + registry reference
- **Node**: Typed computation element identified by integer `id`
- **Operator**: Abstract operation identified by integer `op_id` within an Operator Registry
- **Kernel**: Concrete implementation identified by `(kernel_id, kernel_version)`
- **Kernel Binding**: Mapping `op_id → kernel_id/kernel_version`
- **Environment**: Runtime map from `field_id` to handles and types
- **Gate**: Deterministic predicate producing GateReport
- **Governance**: Policy consuming history + reports → ActionPlan
- **Receipt**: Structured execution record (OperatorReceipt, GateReport, ActionPlan)
- **Braid Event**: Time/thread envelope wrapping receipts
- **Decode**: GLLL vector → glyph identity or REJECT
- **REJECT**: Control-row outcome (not an operator)

**Integrity Terms:**
- **Canonicalization**: Deterministic serialization procedure
- **Digest**: SHA-256 hash expressed as `sha256:<64 hex>`
- **Replay Verification**: Deterministic check for reproducibility

---

## 2 Design Goals (Normative)

See [`docs/design_goals.md`](docs/design_goals.md)

1. **Tokenless Semantics** - No token parsing for semantics
2. **Determinism** - Identical inputs → identical outputs + digests
3. **Auditability** - Receipts for every APPLY
4. **Replay** - Runs should be reproducible
5. **Controlled Failure** - Structured error representation
6. **Extensibility Without Drift** - New meaning requires new op_id/version

---

## 3 Conformance Levels (Normative)

See [`docs/conformance_levels.md`](docs/conformance_levels.md)

### L0 — Core Tokenless (Required baseline)
- Validate modules against `nsc_module.schema.json`, `nsc_node.schema.json`, `nsc_type.schema.json`
- Resolve operators by `(registry_id, version, op_id)` from Operator Registry
- Use Kernel Binding (or default) and record kernel identity in receipts
- Enforce explicit evaluation order via SEQ nodes
- Emit OperatorReceipt for every executed APPLY node
- Canonicalize and hash receipts and module digests

### L1 — Governed
Includes L0 plus:
- Support GateCheck evaluation and GateReport schema
- Support Governance evaluation and ActionPlan schema
- Emit GateReport and ActionPlan braid events
- Support Replay Verification reporting

### L2 — Identity-Aware (Decode-capable)
Includes L1 plus (if decode is used):
- Emit DecodeReport for every decode attempt
- Treat REJECT as control-only outcome
- Emit audit events for ambiguity recovery

---

## 4 Wire Layer

See [`docs/wire/`](docs/wire/)

### 4.1 Canonicalization
See [`docs/wire/canonicalization.md`](docs/wire/canonicalization.md)

**Canonical JSON profile (canonical-json-v1):**
- UTF-8 encoding
- Object keys sorted lexicographically
- Arrays preserve order
- No insignificant whitespace
- No NaN/Inf
- Numbers normalized (no trailing `.0`, no leading `+`)

### 4.2 Canonical Hashing
See [`docs/wire/canonical_hashing.md`](docs/wire/canonical_hashing.md)

- SHA-256 for all digests
- Format: `sha256:<64 lowercase hex>`
- Module digest: nodes sorted by `id` before hashing
- Receipt digests: computed excluding the `digest` field
- Hash chaining: `H_k = sha256(canonical_json(R_k) || H_{k-1})`

### 4.3 JSON Profile
See [`docs/wire/json_profile.md`](docs/wire/json_profile.md)

- Node IDs, op_id, field_id, method_id must be integers
- BYTES values → base64 in JSON
- Validate against schemas before execution

### 4.4 CBOR Profile (Optional)
See [`docs/wire/cbor_profile.md`](docs/wire/cbor_profile.md)

- RFC 8949 canonical CBOR
- Digests still computed over canonical JSON

---

## 5 Language

See [`docs/language/`](docs/language/)

### 5.1 Module Linking
See [`docs/language/nsc_module_linking.md`](docs/language/nsc_module_linking.md)

**Link-compatible modules:**
- Same `registry_id`
- Identical registry version
- Compatible type systems

**Output:**
- Linked module with renumbered nodes (max_id(A) + 1000 offset)
- Namespaced entrypoints
- Deterministic order composition

### 5.2 Error Model
See [`docs/language/nsc_error_model.md`](docs/language/nsc_error_model.md)

**Error categories:**
- VALIDATION_ERROR, REGISTRY_ERROR, KERNEL_BINDING_ERROR
- NUMERIC_INVALID, DECODE_REJECT
- GATE_FAIL, GOVERNANCE_ABORT

**Nonconformance triggers:**
- Treating REJECT as an operator
- Executing APPLY without OperatorReceipt
- Emitting receipts without correct canonical digests

---

## 6 Governance

See [`docs/governance/`](docs/governance/)

### 6.1 Replay Verification
See [`docs/governance/replay_verification.md`](docs/governance/replay_verification.md)

**Inputs:**
- NSC module M
- ModuleDigest Dm
- Operator registry R
- Kernel binding B
- Initial state E0
- BraidEvents stream

**Verification steps:**
1. Validate M, R, B against schemas
2. Recompute module digest and compare
3. Ensure registry matches
4. Ensure kernel bindings match
5. Re-execute module deterministically
6. Compare OperatorReceipts within tolerances
7. Compare GateReports (if present)
8. Report first mismatch

**Tolerances:**
```
|x - y| <= abs + rel * max(|x|, |y|)
```

---

## 7 Glyphs

See [`docs/glyphs/`](docs/glyphs/)

### 7.1 GLLL Codebook
See [`docs/glyphs/glll_codebook.md`](docs/glyphs/glll_codebook.md)

**Structure:**
- `n`: vector length
- `reserved_control_rows`: REJECT required
- `rows`: array of `{row_id, glyph_id, vector}` entries
- `mapping_to_op_id`: glyph_id → op_id mapping

**Constraints:**
- Vectors: length n, entries in {-1, +1}
- Unique row_id, unique glyph_id
- REJECT control row must exist

### 7.2 GLLL Decode
See [`docs/glyphs/glll_decode.md`](docs/glyphs/glll_decode.md)

**Scoring:**
```
s(row) = (w · v_row) / n
```

**Decision rule:**
```
If (s1 < tau_abs) OR (margin < tau_margin):
  output REJECT
Else:
  output GLYPH_ID of best row
```

### 7.3 REJECT Control Row
See [`docs/glyphs/reject_control_row.md`](docs/glyphs/reject_control_row.md)

- Control outcome, NOT an operator
- Cannot be lowered to APPLY
- Must emit DecodeReport
- Optional recovery with audit events

### 7.4 Ambiguity Recovery
See [`docs/glyphs/ambiguity_recovery.md`](docs/glyphs/ambiguity_recovery.md)

**Permitted when:**
- Decode produced REJECT
- Deterministic context rule exists

**Protocol:**
1. Emit DecodeReport with top-k candidates
2. Test candidates in context (typecheck, operator set)
3. Choose first valid deterministically
4. Emit audit event with recovery details

---

## 8 Security

See [`docs/security/`](docs/security/)

### 8.1 Threat Model
See [`docs/security/threat_model.md`](docs/security/threat_model.md)

**Threats in scope:**
- Semantic drift, Kernel drift
- Nondeterminism, Receipt omission
- Ambiguity laundering, Tampering

**Mitigations:**
- Versioned registries, Kernel binding
- Canonical hashing, Mandatory receipts
- REJECT semantics, Replay verification

### 8.2 Integrity
See [`docs/security/integrity.md`](docs/security/integrity.md)

- Digest integrity: all receipts/events contain digests
- Parent linkage for audit trail DAG
- Optional signatures wrap digests
- Registry/binding identity in receipts

### 8.3 Safe Extensions
See [`docs/security/safe_extensions.md`](docs/security/safe_extensions.md)

**Allowed extensions:**
- New operators (new op_id, new registry version)
- New kernels (new kernel_id/kernel_version)
- New node kinds (new language version)

**Forbidden:**
- Changing operator semantics under same op_id
- Executing without receipts
- Treating REJECT as operator

---

## 9 Schemas

### 9.1 NSC Kernel Binding Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NSC Kernel Binding",
  "type": "object",
  "required": ["registry_id", "binding_id", "bindings"],
  "properties": {
    "registry_id": { "type": "string", "minLength": 1 },
    "binding_id": { "type": "string", "minLength": 1 },
    "bindings": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["op_id", "kernel_id", "kernel_version"],
        "properties": {
          "op_id": { "type": "integer", "minimum": 1 },
          "kernel_id": { "type": "string", "minLength": 1 },
          "kernel_version": { "type": "string", "minLength": 1 }
        }
      }
    }
  }
}
```

### 9.2 NSC Module Digest Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NSC Module Digest",
  "type": "object",
  "required": ["module_id", "registry_id", "digest_alg", "canonical_profile", "digest"],
  "properties": {
    "module_id": { "type": "string", "minLength": 1 },
    "registry_id": { "type": "string", "minLength": 1 },
    "digest_alg": { "type": "string", "enum": ["sha256"] },
    "canonical_profile": { "type": "string", "enum": ["canonical-json-v1"] },
    "digest": { "type": "string", "pattern": "^sha256:[0-9a-f]{64}$" }
  }
}
```

### 9.3 NSC Replay Report Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "NSC ReplayReport",
  "type": "object",
  "required": ["replay_pass", "module_digest_ok", "registry_ok", "kernel_binding_ok", "tolerances"],
  "properties": {
    "replay_pass": { "type": "boolean" },
    "module_digest_ok": { "type": "boolean" },
    "registry_ok": { "type": "boolean" },
    "kernel_binding_ok": { "type": "boolean" },
    "tolerances": {
      "type": "object",
      "required": ["abs", "rel"],
      "properties": {
        "abs": { "type": "number", "minimum": 0 },
        "rel": { "type": "number", "minimum": 0 }
      }
    },
    "first_mismatch": {
      "type": "object",
      "required": ["event_id", "field", "expected", "observed"]
    }
  }
}
```

---

## 10 Test Vectors

### 10.1 Canonical JSON Hash Vectors

See [`test-vectors/canonical_json_hash_vectors.md`](test-vectors/canonical_json_hash_vectors.md)

**Vector 1 (minimal):**
```
{"a":1,"b":[2,3],"c":{"d":4}}
```

**Vector 2 (key ordering):**
```
{"a":0,"m":1,"z":2}
```

**Vector 3 (prohibited):**
Payload containing NaN/Inf MUST be rejected.

### 10.2 Hadamard Decode Vectors (n=16)

See [`test-vectors/hadamard_decode_vectors_n16.md`](test-vectors/hadamard_decode_vectors_n16.md)

**Thresholds:**
- `tau_abs = 0.60`
- `tau_margin = 0.10`

**Cases:**
- Clean decode (w = v_row1) → GLYPH_ID
- Low absolute (w ≈ 0) → REJECT
- Tight margin (margin < 0.10) → REJECT

### 10.3 Conformance Cases

See [`test-vectors/conformance_cases.md`](test-vectors/conformance_cases.md)

1. APPLY without OperatorReceipt → FAIL
2. REJECT lowered to APPLY → FAIL
3. kernel_version missing → FAIL
4. receipt digest incorrect → FAIL
5. module digest mismatch → FAIL

---

## Appendix A: Examples

### A.1 Kernel Binding Example

See [`examples/registries/kernels.binding.example.json`](examples/registries/kernels.binding.example.json)

### A.2 Replay Report Example

See [`examples/receipts/replay_report.example.json`](examples/receipts/replay_report.example.json)

### A.3 Decode Reject Braid Event

See [`examples/events/braid_event.decode_reject.json`](examples/events/braid_event.decode_reject.json)

---

## Appendix B: Coherence Framework Integration

The NPE (Noetican Proposal Engine) can be aligned with the Coherence Framework:

### B.1 Debt Functionals (L1)
```
C(x) = Σ_i w_i ||r̃_i||² + Σ_j v_j p_j(x)
```

### B.2 Gates (L2)
- **Hard gates**: NaN/Inf free, positivity (ρ ≥ 0), domain bounds
- **Soft gates**: phys residual RMS, constraint residual RMS

### B.3 Rails (L3)
- **R1**: dt deflation (α ∈ (0,1), typical 0.8)
- **R2**: rollback to x_prev
- **R3**: projection to constraint manifold
- **R4**: bounded damping/gain adjustment

### B4. Hash Chaining (L2-L4)
```
H_k = sha256(canonical_json(R_k) || H_{k-1})
```

See `nsc/npe_v1_0.py` for aligned implementation.

---

*End of NSC Tokenless v1.1 Specification*
