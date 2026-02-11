# Threat Model (Normative) — NSC Tokenless v1.1

NSC focuses on preventing undetected drift and unaudited execution.

## Threats in scope
1) **Semantic drift**: operator meaning changes without versioning.
2) **Kernel drift**: kernel implementation changes without recording version.
3) **Nondeterminism**: parallel reduction order, RNG, time-based behavior.
4) **Receipt omission**: operators run without emitting required receipts.
5) **Ambiguity laundering**: decode ambiguity is "resolved" without record.
6) **Tampering**: receipts/events altered after emission.

## Out of scope
- Key management and authentication systems (signing is optional)
- Side-channel attacks
- Hardware fault tolerance

## Mitigations specified
- versioned registries
- kernel binding with explicit kernel_id/version
- canonical hashing of module and receipts/events
- mandatory receipt emission
- explicit REJECT semantics
- replay verification reports
