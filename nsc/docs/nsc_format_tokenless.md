# NSC Tokenless Format (Canonical)

This document defines how NSC achieves "tokenless" execution.

## 1. Canonical Representation

A conforming NSC runtime MUST accept an NSC Module object with:
- Numeric node IDs (positive integers starting at 1)
- Numeric operator IDs (`op_id`) referencing the operator registry
- Explicit argument edges by node ID (not by name)
- Explicit evaluation order via SEQ node references

Strings may exist as metadata only. Strings MUST NOT be required to execute.

### Example: Tokenless vs Tokenful

**Tokenful (not valid NSC):**
```json
{
  "operators": ["source_plus", "twist", "delta"],
  "apply": [
    { "op": "source_plus", "args": ["phi"] },
    { "op": "twist", "args": ["result"] }
  ]
}
```

**Tokenless (valid NSC):**
```json
{
  "nodes": [
    { "id": 1, "kind": "FIELD_REF", "field_id": 1 },
    { "id": 2, "kind": "APPLY", "op_id": 10, "args": [1] },
    { "id": 3, "kind": "APPLY", "op_id": 11, "args": [2] }
  ]
}
```

## 2. Operator Identity

Operators are identified by integer `op_id`. The meaning of `op_id` is defined by
the Operator Registry (versioned). An NSC module is only valid when it declares:
- `registry_id` (string, version tag)
- The runtime has that registry loaded

The registry maps `op_id` → operator metadata including:
- Type signature
- Kernel requirements
- Receipt metrics
- Resonance projection

## 3. GLLL Glyph Support (Optional Input Channel)

If a system receives glyphs from a noisy channel, tokenless mapping occurs before NSC:
- A GLLL vector decodes to a `glyph_id` or REJECT control
- That `glyph_id` maps to an operator registry entry via the codebook
- The NSC module uses `op_id` only for execution

REJECT is a control event (not an operator). It must be logged as a DecodeReport.

### GLLL to NSC Pipeline

```
GLLL Vector ──▶ Decoder ──▶ glyph_id OR REJECT
                                   │
                                   ▼
                          Codebook Lookup
                                   │
                                   ▼
                              op_id (in NSC)
```

## 4. Canonical Hash Rules

NSC uses canonical hashing for receipts and digests:
- Canonical JSON serialization (sorted keys, UTF-8, no whitespace)
- SHA-256 digests written as `sha256:<hex>` (lowercase hex, 64 chars)

A conforming implementation MUST compute digests in the same canonical form.

### Canonical JSON Rules
1. Objects have keys sorted alphabetically
2. Strings are UTF-8 encoded
3. No whitespace outside string values
4. Arrays have elements in declared order

## 5. Allowed Encodings

The normative model is independent of wire encoding. Permitted practical encodings:

| Encoding | Notes |
|----------|-------|
| JSON | Default. Human-readable, widest tooling support |
| CBOR | Binary, compact, good for large modules |
| Protobuf | Schema-defined binary format |
| Flatbuffers | Zero-copy binary, memory-mapped |

Conformance is about semantic equivalence, not byte-for-byte equality.
A JSON module and its CBOR equivalent are conformant if they produce identical
execution behavior.

## 6. Module Digest (Normative)

Each module SHOULD include a Module Digest for integrity verification:

```json
{
  "module_digest": {
    "algorithm": "sha256",
    "value": "sha256:...",
    "canonical_form": "..."
  }
}
```

The digest is computed over the canonical module representation excluding
the `module_digest` field itself. See [Module Digest](nsc_module_digest.md) for
full specification.
