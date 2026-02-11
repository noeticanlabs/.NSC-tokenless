# Receipts and Braid Events

NSC is audit-first. A conforming run MUST emit receipts and braid events.

## 1. OperatorReceipt (Mandatory)

Every governed operator application MUST emit an OperatorReceipt.

### Receipt Structure
```json
{
  "run_id": "demo_run_0001",
  "event_id": "op_000042",
  "thread_id": "micro",
  "step_index": 12,
  "kind": "op_apply",
  "op_id": 20,
  "kernel": {
    "kernel_id": "kern.return_baseline.linear_blend",
    "kernel_version": "1.0.0"
  },
  "digests": {
    "digest_in": {
      "x": "sha256:2bb0f2c0b2c4c0d8b2a6b3c4d1e2f3a4b5c6d7e8f90123456789abcdef012345",
      "ref": "sha256:0aa1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f809123456789abcdef012345678"
    },
    "digest_out": "sha256:9f8e7d6c5b4a392817161514131211100f0e0d0c0b0a09080706050403020100"
  },
  "delta": {
    "C_before": 0.9928,
    "C_after": 0.9951,
    "dC": 0.0023,
    "r_before": { "r_gate": 0.0112, "r_cons": 0.0047, "r_num": 0.0189 },
    "r_after": { "r_gate": 0.0086, "r_cons": 0.0039, "r_num": 0.0151 }
  },
  "params": { "mode": "linear_blend", "alpha": 0.15 },
  "parents": ["gate_000041"],
  "digest": "sha256:1111111111111111111111111111111111111111111111111111111111111111"
}
```

### Receipt Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Unique run identifier |
| `event_id` | string | Unique event identifier |
| `thread_id` | enum | Execution thread (micro/step/macro/io/audit) |
| `step_index` | integer | Step number |
| `kind` | enum | Event kind (op_apply/op_ref/coupling_apply/frame_event) |
| `op_id` | integer | Operator that was applied |
| `kernel` | object | Kernel ID and version |
| `digests` | object | Input/output digests |
| `delta` | object | Coherence and residual changes |
| `params` | object | Operator parameters |
| `resources` | object | Resource usage metrics |
| `parents` | array | Parent event IDs |
| `digest` | string | SHA-256 of this receipt |

### Thread IDs

| Thread | Description |
|--------|-------------|
| `micro` | Micro-step operations |
| `step` | Full simulation steps |
| `macro` | Macro-level coordination |
| `io` | Input/output operations |
| `audit` | Audit and governance events |

## 2. GateReport (Mandatory if GATE_CHECK exists)

Every GATE_CHECK node MUST emit a GateReport.

### Gate Report Fields
| Field | Type | Description |
|-------|------|-------------|
| `pass` | boolean | Gate pass/fail |
| `C` | number | Current coherence |
| `r` | object | Residual bundle |
| `recommended_action` | enum | Suggested action |
| `notes` | string | Human-readable notes |

## 3. DecodeReport (Mandatory if GLLL decoding used)

If glyph decoding is performed, a DecodeReport MUST be emitted.

### Decode Report Structure
```json
{
  "n": 16,
  "chosen": { "kind": "REJECT", "control_row_id": 0 },
  "topk": [
    { "glyph_id": "GLYPH.SOURCE_PLUS", "score": 0.31 },
    { "glyph_id": "GLYPH.TWIST", "score": 0.29 },
    { "glyph_id": "GLYPH.DELTA", "score": 0.28 }
  ],
  "thresholds": { "tau_abs": 0.60, "tau_margin": 0.10 },
  "margin": 0.02
}
```

### Decode Report Fields

| Field | Type | Description |
|-------|------|-------------|
| `n` | integer | Vector dimension |
| `chosen` | object | Chosen glyph or REJECT |
| `topk` | array | Top-k candidates with scores |
| `thresholds` | object | Thresholds used (tau_abs, tau_margin) |
| `margin` | number | Margin between top candidates |

## 4. Braid Event

All receipts and reports are braid events in the time-structured ledger.

### Braid Event Structure
```json
{
  "run_id": "demo_run_0001",
  "event_id": "e1",
  "thread_id": "micro",
  "step_index": 0,
  "kind": "op_apply",
  "payload": { "op_id": 10, "note": "source_plus applied" },
  "parents": [],
  "digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000"
}
```

### Braid Event Fields

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Run identifier |
| `event_id` | string | Unique event ID |
| `thread_id` | enum | Execution thread |
| `step_index` | integer | Step number |
| `kind` | string | Event kind |
| `payload` | object | Event payload |
| `parents` | array | Parent event IDs (causal chain) |
| `digest` | string | SHA-256 of payload |

### Causal Chain (Parents)
The `parents` array establishes causal relationships between events:
- Each event references its causal predecessors
- This creates a directed acyclic graph (DAG)
- Enables replay and verification of execution

## 5. Receipt Emission Timing

| Node Kind | When Receipt Emitted |
|-----------|---------------------|
| APPLY | Immediately after operator execution |
| GATE_CHECK | Immediately after gate evaluation |
| GOVERN | Immediately after action plan generation |
| EMIT | When braid event is written |

## 6. Digest Computation

All digests are computed over canonical JSON:

```json
// Input digest (single value)
"digest_in": "sha256:2bb0f2c0b2c4c0d8b2a6b3c4d1e2f3a4b5c6d7e8f90123456789abcdef012345"

// Input digest (multiple values)
"digest_in": {
  "x": "sha256:...",
  "ref": "sha256:..."
}
```
