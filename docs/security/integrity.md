# Integrity (Normative) — NSC Tokenless v1.1

Integrity means "what ran is what is recorded."

## 1) Digest integrity
All receipts and braid events MUST contain a digest computed from canonical payload bytes.
This makes tampering detectable.

## 2) Parent linkage
Braid events SHOULD link to parents so the record is a DAG; missing parents weaken audit strength
but do not necessarily break L0 conformance unless required by profile.

## 3) Optional signatures
A runtime MAY sign digests. If signatures are used:
- Signature fields MUST NOT be included in canonical digest computation.
- Signatures wrap the digest; they do not define it.

## 4) Registry and binding integrity
Receipts MUST record:
- op_id
- kernel_id and kernel_version
- registry_id and registry version at run level (via module or braid metadata)

Without these, replay verification cannot be meaningful.
