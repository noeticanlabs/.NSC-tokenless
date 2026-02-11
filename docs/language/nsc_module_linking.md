# NSC Module Linking (Normative) — NSC Tokenless v1.1

Linking combines modules without introducing token parsing or semantic drift.

## 1) Inputs and compatibility
Two modules A and B are link-compatible if:
- A.registry_id == B.registry_id
- Their operator registry version is identical (or both specify same registry_id and the runtime
  binds a single registry version explicitly)
- Their type systems are compatible (same type tags; no tag collisions)

## 2) Linking outputs
A linker produces a new module L with:
- module_id = "link(" + A.module_id + "," + B.module_id + ")" (or declared policy)
- registry_id preserved
- nodes: union of nodes from A and B with renumbering to avoid id collisions
- entrypoints: merged with namespacing:
  - "A.<name>" for A entrypoints
  - "B.<name>" for B entrypoints
- order.eval_seq_id referencing a SEQ node in L that defines deterministic evaluation order.

## 3) Deterministic renumbering policy (recommended)
Let max_id(A) be maximum node id in A.
Renumber B nodes as:
- id' = id + max_id(A) + 1000
This avoids accidental overlap and preserves relative ordering.

Renumber all node references accordingly (args, seq, snapshot, payload_ref, etc.).

## 4) Order composition
The simplest deterministic linking order is concatenation:
- L.SEQ = A.SEQ followed by B.SEQ
If A and B have independent entrypoints, the linker MAY create separate SEQ nodes and
set multiple entrypoints. However, `order.eval_seq_id` MUST still be a single SEQ id.

## 5) No hidden insertion
The linker MUST NOT insert implicit coercions or operators that change behavior without
explicit nodes. Any inserted nodes must appear in L.nodes and affect module digest.

## 6) Digest rule
The linked module digest MUST reflect the linked module object. Linking is not transparent.
