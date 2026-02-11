# Determinism Contract

A conforming NSC runtime MUST be deterministic under identical:
- Module object
- Initial environment/state
- Operator registry (exact version)
- Kernel IDs + versions (exact)
- Numeric settings (precision, rounding mode if applicable)
- RNG seed if randomness is permitted (recommended: forbid by default)

## 1. Ordering Guarantees

### Explicit SEQ Order
Every module MUST define a SEQ node that declares evaluation order:

```json
{
  "id": 99,
  "kind": "SEQ",
  "seq": [1, 2, 3, 4, 5, 6, 7]
}
```

The runtime MUST evaluate nodes in exactly this order. If a runtime parallelizes
internally, results must match the sequential semantics.

### Parallelization Safety
If a runtime implements parallel execution:
- It MUST produce identical results to sequential execution
- This is a conformance requirement, not an optimization hint

## 2. Floating Point Discipline

NSC does not mandate a specific floating point implementation, but requires:

### Required Declarations
The runtime MUST record and expose:
- Float type (F32 or F64)
- Math library/kernel version identifiers
- Any non-deterministic hardware features disabled or recorded

### Conforming Implementations
An implementation MUST either:
1. Use IEEE 754 with default rounding, documented
2. Use fixed-point arithmetic with documented precision
3. Use software floating point with deterministic behavior

## 3. Forbidden Nondeterminism

The following are explicitly prohibited:

| Category | Forbidden Behaviors |
|----------|---------------------|
| Randomness | Hidden RNG, unseeded randomness, time-based seeding |
| Time | Time-dependent behavior, deadlines affecting output |
| Iteration | Unordered map/set iteration, non-deterministic reduction order |
| Memory | Address-based behavior, uninitialized memory reads |
| Concurrency | Race conditions, lock ordering variations |

### Exception: Permitted RNG
If a module requires randomness:
1. It MUST declare `rng_seed` in the module metadata
2. The seed MUST be provided by the calling context
3. The RNG algorithm MUST be deterministic for a given seed

Example:
```json
{
  "meta": {
    "rng_seed": "fixed_seed_for_testing"
  }
}
```

## 4. Kernel Determinism

Each operator application uses a kernel. The kernel MUST be deterministic:

```json
{
  "kernel": {
    "kernel_id": "kern.return_baseline.linear_blend",
    "kernel_version": "1.0.0"
  }
}
```

The runtime MUST:
1. Use the declared kernel for each operator
2. Reject modules that reference unknown kernels
3. Reject modules that use kernels with mismatched versions

## 5. Module Identity

Two modules are identical if:
1. They have identical `module_id`
2. They reference identical `registry_id`
3. They have structurally identical `nodes` arrays
4. They have identical `order.eval_seq_id`

### Structural Equality
Nodes are structurally equal if:
- Same `kind`
- Same `op_id` (if APPLY)
- Same `args` (if APPLY)
- Same `field_id` (if FIELD_REF)
- Same `seq` (if SEQ)

## 6. Verification

A conforming implementation SHOULD provide:
- Determinism test suite
- Seeded test mode
- Audit of nondeterministic operations

### Recommended Test Pattern
```json
{
  "meta": {
    "test_mode": true,
    "determinism_check": true,
    "expected_digest": "sha256:..."
  }
}
```
