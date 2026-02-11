# Terminology (Normative) — NSC Tokenless v1.1

This document defines terms used across the NSC specification. "MUST", "SHOULD", and "MAY"
are interpreted as in RFC 2119.

## Core language terms

**Tokenless**
A language is tokenless if a conforming implementation can execute modules without
parsing any textual tokens that affect semantics. Text fields may exist for metadata
and human readability, but execution MUST depend only on typed structures and numeric IDs.

**Module**
A self-contained unit of execution: typed node table + explicit evaluation order +
entrypoints + registry reference.

**Node**
A typed computation element identified by integer `id`. Nodes form a directed graph
via integer references (edges) and are evaluated under an explicit SEQ ordering.

**Operator**
An abstract operation identified by integer `op_id` within an Operator Registry.
Operator meaning is immutable once published under an `(registry_id, version, op_id)` triple.

**Kernel**
A concrete implementation of an operator, identified by `(kernel_id, kernel_version)`.
Kernel selection for a run is specified by a Kernel Binding.

**Kernel Binding**
A mapping that assigns a kernel implementation to each `op_id` used in a run.
This prevents silent drift (same `op_id` executed by different kernels without audit).

**Environment**
The runtime-provided map from `field_id` to actual field handles/data structures and
their types. Environment contents are outside the language but are referenced deterministically.

**Gate**
A deterministic predicate/check on a snapshot that produces a GateReport, used to detect
constraint violation, numerical instability, or coherence debt.

**Governance**
A deterministic policy that consumes history + gate reports and produces an ActionPlan.

**Receipt**
A structured record of execution. NSC requires OperatorReceipts for APPLY evaluations,
and GateReports/ActionPlans when governed nodes are present.

**Braid Event**
A time/thread envelope that wraps receipts and other events in an append-only record.
Every receipt SHOULD be emitted as a braid event.

**Decode**
Identity decoding from GLLL vectors into glyph identities (or REJECT control outcome).
Decode is optional; if used, it is normatively specified by the glyph docs and schemas.

**REJECT**
A reserved control-row outcome indicating insufficient decoding confidence. REJECT is NOT
an operator and MUST NOT be lowered to APPLY.

## Integrity terms

**Canonicalization**
A deterministic serialization procedure that yields identical bytes across platforms given
equivalent structured objects.

**Digest**
A SHA-256 hash of canonical bytes, expressed as `sha256:<64 hex>`.

**Replay Verification**
A deterministic check that a run can be reproduced from module+registry+bindings+initial state
and that emitted receipts match expectations within declared tolerances.
