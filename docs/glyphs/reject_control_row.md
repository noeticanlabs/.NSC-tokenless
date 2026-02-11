# REJECT Control Row (Normative) — NSC Tokenless v1.1

REJECT is the formal "fail closed" identity result.

## 1) What REJECT is
- A control outcome produced by decode only.
- Not an operator.
- Not permitted to lower into APPLY.

## 2) Required behavior
When REJECT occurs:
- Emit DecodeReport braid event (audit thread recommended).
- Stop lowering/executing the affected segment.
- Do not substitute a "best guess" silently.

## 3) Optional recovery
A system MAY attempt contextual recovery (see ambiguity recovery doc), but MUST:
- keep the original DecodeReport
- emit an audit event describing the recovery attempt
- never hide that REJECT occurred originally
