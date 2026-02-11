# Ambiguity Recovery (Optional Policy; Normative if used) — NSC Tokenless v1.1

Ambiguity recovery is allowed but must be auditable.

## 1) When recovery is permitted
Recovery MAY be attempted when:
- Decode produced REJECT due to margin/abs thresholds, AND
- A deterministic context rule exists (type constraints, grammar constraints, manifold wiring constraints).

## 2) Recovery protocol (MUST if used)
1) Emit DecodeReport with top-k candidates and scores.
2) Evaluate candidates deterministically:
   - map glyph_id -> op_id using codebook mapping_to_op_id
   - test each candidate in context (typecheck, allowed operator set, chord well-formedness)
3) Choose one candidate deterministically (first valid in ordered list).
4) Emit audit event:
   - kind: "decode_recovery"
   - payload includes: decode_report_event_id, candidates_tested, chosen_candidate, rule_id
5) Proceed only if recovery succeeds; otherwise treat as final failure.

## 3) Prohibited behavior
- No heuristic ML guessing unless it is deterministic and recorded as such.
- No silent substitution.
- No changing thresholds without recording them.
