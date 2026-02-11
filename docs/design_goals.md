# Design Goals (Normative) — NSC Tokenless v1.1

1. **Tokenless Semantics**
   Execution MUST not depend on parsing textual tokens. All semantics are driven by typed
   node graphs, integer IDs, and versioned registries/bindings.

2. **Determinism**
   Under identical inputs (module, registry, kernel binding, initial state/environment,
   numeric config, and any declared RNG seed), a conforming runtime MUST produce identical
   outputs and identical digests for module and receipts, modulo declared numeric tolerances.

3. **Auditability**
   Execution MUST emit receipts that make it impossible to "do work without leaving
   fingerprints." OperatorReceipts are mandatory for executed APPLY nodes. Governed
   artifacts are mandatory when relevant node kinds exist.

4. **Replay**
   A run SHOULD be replayable. At minimum, receipts MUST provide sufficient information
   to validate operator identity (op_id), kernel identity (kernel_id/version), digests, and
   key metric deltas.

5. **Controlled Failure**
   Failures (decode ambiguity, gate failures, numeric invalid states) MUST be representable
   as structured events. The system MUST NOT silently "fix" or suppress these conditions.

6. **Extensibility Without Drift**
   Extensions MUST not mutate existing operator meaning. New meaning requires new op_id
   or new registry version, and kernel changes must be versioned and recorded.
