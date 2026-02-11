# Replay Verification (Normative) — NSC Tokenless v1.1

Replay verification is the mechanism that turns "audit logs" into "checkable truth."

## 1) Inputs
A replay verifier consumes:
- NSC module M
- ModuleDigest Dm
- Operator registry R (matching M.registry_id and version)
- Kernel binding B (mapping op_id -> kernel_id/version)
- Initial state/environment E0 (runtime-specific; must be consistent)
- BraidEvents stream (or receipts extracted from them)

## 2) Verification steps (minimum)
1) Validate M, R, B against schemas.
2) Recompute module digest and compare to Dm.
3) Ensure registry matches (registry_id + version).
4) Ensure kernel bindings match B for every op_id executed.
5) Re-execute module deterministically from E0 under B.
6) For each executed APPLY:
   - recompute digests in/out
   - recompute delta metrics (C and residual bundle r) if defined deterministically by runtime
   - compare against emitted OperatorReceipt within tolerances
7) For each GATE_CHECK (if present):
   - recompute GateReport, compare within tolerances
8) Report first mismatch if any.

## 3) Tolerances
Replay verifier MUST declare tolerances used:
- absolute tolerance abs
- relative tolerance rel

A float value x matches y if:
|x - y| <= abs + rel * max(|x|, |y|)

## 4) Output
Verifier emits ReplayReport object validating against `schemas/nsc_replay_report.schema.json`.

## 5) Minimal acceptable replay in L0
L0 runtimes MUST at least verify:
- module digest
- registry identity
- kernel identity recorded in receipts
- receipt digests correct

L1+ runtimes SHOULD verify operator deltas and gate/governance reports.
