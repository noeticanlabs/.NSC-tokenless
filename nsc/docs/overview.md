# NSC Tokenless — Overview

NSC is the executable form of Noetican computation.

A **tokenless** language means:
- Programs can be executed without textual parsing.
- The "surface form" can be JSON/CBOR/Protobuf/etc., but the semantics are defined
  over typed node graphs and numeric IDs, not strings.

NSC sits between:
- high-level meaning grammars (GHLL / Noetica)
and
- runtime execution (GM-OS style solver/executor)
and
- time/audit logging (GML braid).

## Core Design Principles

### 1. Tokenless Execution
All operator identity is by integer `op_id`. Strings exist only as metadata.
A conforming runtime never needs to parse glyph tokens for execution.

### 2. Deterministic Evaluation
Every NSC module declares explicit evaluation order via SEQ nodes.
The runtime must follow this order exactly. No hidden scheduling.

### 3. Audit-First
Every operator application that affects governed computation MUST emit a receipt.
Every gate check MUST emit a report.
Every governance decision MUST emit an action plan.
All of these are also braid events in the time-structured ledger.

### 4. Typed Operator Application
No implicit coercions. Operators declare their input and output types.
Type checking happens at module load time, not at runtime.

### 5. Optional Governance
Governance is built-in but optional. A module may:
- Execute without gates (pure computation)
- Include gates without governance (reports only)
- Include full governance (gate → govern → emit)

## What Makes a Conforming Run

A run is conforming only if it produces:
1. Deterministic node evaluation in declared order
2. Declared gate checks (if present)
3. Explicit governance actions (if present)
4. Receipts for each governed operator application
5. Braid events for all state changes

## Relationship to Other Noetica Components

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  GHLL / Noetica │────▶│   NSC Tokenless │────▶│  GM-OS Runtime  │
│  (high-level)   │     │  (executable)   │     │  (solver/exec)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   GLLL Codec    │     │   GML Braid     │
                        │  (glyph→op_id)  │     │  (audit ledger) │
                        └─────────────────┘     └─────────────────┘
```

## Version Compatibility

NSC v1.0 is designed to be stable. The following invariants hold:
- `op_id` meaning never changes within a `registry_id` version
- Schema validation is sufficient for basic conformance
- Receipt format is backward-compatible within major versions
