# NSC Module Digest Specification

The Module Digest provides integrity verification for NSC modules.

## 1. Purpose

The Module Digest enables:
- Detection of module tampering
- Verification of module identity
- Caching with cache invalidation on change
- Audit trail linking

## 2. Digest Computation

The digest is computed over the **canonical module representation** excluding
the `module_digest` field itself.

### Canonical Form
1. Remove the `module_digest` field
2. Sort object keys alphabetically
3. Encode as canonical JSON (no whitespace)
4. Compute SHA-256 hash
5. Encode as lowercase hex: `sha256:<64-char-hex>`

### Canonical JSON Example

**Original:**
```json
{
  "module_id": "psi_minimal",
  "registry_id": "noetican.nsc.registry.v1",
  "types": { "t_field_scalar": { "tag": "FIELD", "of": { "tag": "SCALAR" } } },
  "nodes": [
    { "id": 1, "kind": "FIELD_REF", "field_id": 1, "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } } }
  ],
  "entrypoints": { "Psi": 7 },
  "order": { "eval_seq_id": 99 },
  "module_digest": { "algorithm": "sha256", "value": "sha256:..." }
}
```

**Canonical (for hashing):**
```json
{"entrypoints":{"Psi":7},"module_id":"psi_minimal","nodes":[{"args":[1],"field_id":1,"id":1,"kind":"FIELD_REF","type":{"of":{"tag":"SCALAR"},"tag":"FIELD"}}],"order":{"eval_seq_id":99},"registry_id":"noetican.nsc.registry.v1","types":{"t_field_scalar":{"of":{"tag":"SCALAR"},"tag":"FIELD"}}}
```

## 3. Module Digest Structure

```json
{
  "module_digest": {
    "algorithm": "sha256",
    "value": "sha256:2bb0f2c0b2c4c0d8b2a6b3c4d1e2f3a4b5c6d7e8f90123456789abcdef0123",
    "canonical_form": "sha256:a1b2c3d4e5f6..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `algorithm` | string | Hash algorithm (currently "sha256") |
| `value` | string | The digest value `sha256:<hex>` |
| `canonical_form` | string | Optional: digest of canonical form (for verification) |

## 4. Module with Digest

```json
{
  "module_id": "psi_minimal",
  "registry_id": "noetican.nsc.registry.v1",
  "types": {
    "t_field_scalar": { "tag": "FIELD", "of": { "tag": "SCALAR" } }
  },
  "nodes": [
    { "id": 1, "kind": "FIELD_REF", "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } }, "field_id": 1 },
    { "id": 2, "kind": "APPLY", "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } }, "op_id": 10, "args": [1] },
    { "id": 99, "kind": "SEQ", "type": { "tag": "FIELD", "of": { "tag": "SCALAR" } }, "seq": [1, 2] }
  ],
  "entrypoints": { "Psi": 2 },
  "order": { "eval_seq_id": 99 },
  "module_digest": {
    "algorithm": "sha256",
    "value": "sha256:2bb0f2c0b2c4c0d8b2a6b3c4d1e2f3a4b5c6d7e8f90123456789abcdef0123"
  }
}
```

## 5. Verification Process

### Runtime Verification
1. Extract `module_digest.value`
2. Compute digest of canonical module (without module_digest)
3. Compare computed digest with declared digest
4. Reject if mismatched

### External Verification
1. Load module
2. Compute canonical JSON (sorted keys)
3. Compute SHA-256
4. Verify matches `module_digest.value`

## 6. Compatibility

### Version 1.0
- Algorithm: SHA-256
- Format: `sha256:<64-char-hex>`
- Canonical JSON: RFC 8785 (JSON Canonicalization Scheme)

### Future Versions May Support
- SHA-3 family
- BLAKE3
- Cryptographic signatures

## 7. Implementation Notes

### JSON Canonicalization
Use RFC 8785-compliant canonicalization:
- No whitespace except as required by JSON
- Object keys sorted lexicographically
- Arrays preserved as-is (no sorting)
- Strings escaped per JSON spec

### Example (Python)
```python
import json
import hashlib

def compute_module_digest(module):
    # Clone and remove module_digest
    canonical = {k: v for k, v in module.items() if k != 'module_digest'}
    # Canonical JSON
    canonical_json = json.dumps(canonical, separators=(',', ':'), sort_keys=True)
    # SHA-256
    digest = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
    return f"sha256:{digest}"
```

## 8. Schema Integration

The module digest is optional in v1.0 but recommended for production use.

```json
{
  "module_id": "psi_minimal",
  // ... other required fields ...
  "module_digest": {
    "algorithm": { "type": "string", "const": "sha256" },
    "value": { "type": "string", "pattern": "^sha256:[0-9a-f]{64}$" }
  }
}
```
