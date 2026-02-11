# NSC Operator Registry

NSC is tokenless because operator meaning is referenced by `op_id` (integer).

## 1. Registry Object

A registry contains:

| Field | Type | Description |
|-------|------|-------------|
| `registry_id` | string | Unique identifier for this registry |
| `version` | string | Semantic version (e.g., "1.0.0") |
| `ops` | array | Array of operator entries |

### Registry Structure
```json
{
  "registry_id": "noetican.nsc.registry.v1",
  "version": "1.0.0",
  "ops": [...]
}
```

## 2. Operator Entry

Each operator entry contains:

| Field | Type | Description |
|-------|------|-------------|
| `op_id` | integer | Unique positive ID for this operator |
| `token` | string | Human-readable name (metadata only) |
| `type_signature` | object | Input and output types |
| `lowering_id` | string | Kernel lowering identifier |
| `resonance_projection_id` | string | Resonance projection identifier |
| `receipt_metrics` | array | Metrics to include in receipts |
| `kernel_policy` | object | Kernel requirements |

### Operator Entry Example
```json
{
  "op_id": 10,
  "token": "source_plus",
  "type_signature": {
    "args": [{ "tag": "FIELD", "of": { "tag": "SCALAR" } }],
    "ret": { "tag": "FIELD", "of": { "tag": "SCALAR" } }
  },
  "lowering_id": "nsc.apply.source_plus",
  "resonance_projection_id": "res.op.source_plus",
  "receipt_metrics": ["kernel_id", "kernel_version", "C_before", "C_after"],
  "kernel_policy": { "requires_kernel_id": true, "requires_kernel_version": true }
}
```

## 3. Type Signature

The `type_signature` defines operator arity and types:

```json
{
  "args": [
    { "tag": "FIELD", "of": { "tag": "SCALAR" } },
    { "tag": "FIELD", "of": { "tag": "SCALAR" } }
  ],
  "ret": { "tag": "FIELD", "of": { "tag": "SCALAR" } }
}
```

## 4. Kernel Policy

The `kernel_policy` declares kernel requirements:

```json
{
  "requires_kernel_id": true,
  "requires_kernel_version": true
}
```

If `requires_kernel_id` is true, the runtime MUST provide a kernel ID
when applying this operator. If `requires_kernel_version` is true, the
runtime MUST provide a kernel version.

## 5. Extension Rules

New operators MUST:
1. Choose a new `op_id` not previously used in that `registry_id`
2. Declare type signature and required receipt metrics
3. Declare determinism policy (pure or step-affecting)
4. Declare whether it is allowed in governed mode

### Operator ID Allocation
- `1-9`: Reserved for system operators
- `10-99`: Core computation operators
- `100-999`: Application-specific operators
- `1000+`: Extension operators

## 6. Forbidden Drift

Once published:
- An `op_id`'s type signature MUST NOT change
- An `op_id`'s semantic category MUST NOT change

If you need a different signature, allocate a new `op_id`.

## 7. Minimal Canonical Operator Set (v1)

| op_id | token | Purpose |
|-------|-------|---------|
| 10 | `source_plus` | Add coherence from source |
| 11 | `curvature_twist` | Apply curvature twist |
| 12 | `delta_transform` | Delta transformation |
| 13 | `regulate_circle` | Regulation control |
| 14 | `sink_minus` | Dissipation sink |
| 20 | `return_to_baseline` | Return to baseline |

## 8. Receipt Metrics

Operators declare which metrics MUST be included in their receipts:

| Metric | Description |
|--------|-------------|
| `kernel_id` | Kernel identifier used |
| `kernel_version` | Kernel version |
| `digest_in` | Input digest |
| `digest_out` | Output digest |
| `C_before` | Coherence before application |
| `C_after` | Coherence after application |
| `dC` | Change in coherence |
| `r_before` | Residual bundle before |
| `r_after` | Residual bundle after |
| `injection` | Injection metric |
| `geometry` | Geometry metric |
| `transform` | Transform metric |
| `control` | Control metric |
| `fixups` | Fixups metric |
| `dissipation` | Dissipation metric |
| `params` | Operator parameters |
