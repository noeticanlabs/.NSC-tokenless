# Gates and Governance

NSC supports governed computation through a built-in gate and governance system.

## 1. GateCheck ABI

A gate check is instantaneous evaluation of a predicate on a snapshot.

### GATE_CHECK Node
```json
{
  "id": 20,
  "kind": "GATE_CHECK",
  "type": { "tag": "REPORT", "kind": "GATE_REPORT" },
  "gate_id": 1,
  "snapshot": 7
}
```

| Field | Type | Description |
|-------|------|-------------|
| `gate_id` | integer | Gate definition ID |
| `snapshot` | integer | Node ID of the state snapshot to evaluate |

### Gate Report Structure
```json
{
  "pass": true,
  "C": 0.9951,
  "r": {
    "r_gate": 0.0086,
    "r_cons": 0.0039,
    "r_num": 0.0151
  },
  "recommended_action": "NONE",
  "notes": "All metrics within thresholds."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `pass` | boolean | Whether the gate passed |
| `C` | number | Coherence value [0, 1] |
| `r` | object | Residual bundle |
| `r.r_gate` | number | Gate residual |
| `r.r_cons` | number | Consistency residual |
| `r.r_num` | number | Numerical residual |
| `recommended_action` | enum | Suggested corrective action |
| `notes` | string | Human-readable notes |

### Recommended Actions

| Action | Description |
|--------|-------------|
| `NONE` | No action needed |
| `DT_REDUCE` | Reduce time step |
| `ROLLBACK` | Roll back to previous state |
| `PROJECT` | Project to next step |
| `DAMP` | Apply damping |
| `ABORT` | Abort execution |

## 2. Governance ABI

Governance produces an action plan based on history and gate reports.

### GOVERN Node
```json
{
  "id": 30,
  "kind": "GOVERN",
  "type": { "tag": "ACTION_PLAN" },
  "history": 9001,
  "reports": [20]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `history` | integer | History buffer node ID |
| `reports` | array | Gate report node IDs |

### Action Plan Structure
```json
{
  "action": "DAMP",
  "params": {
    "damping_coefficient": 0.85
  },
  "reasons": [
    {
      "code": "COHERENCE_DEGRADATION",
      "detail": "C dropped below 0.99 threshold"
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `action` | enum | Chosen action |
| `params` | object | Action parameters |
| `reasons` | array | Structured reasons for the decision |

### Action Parameters by Type

**DT_REDUCE:**
```json
{ "dt_multiplier": 0.5 }
```

**ROLLBACK:**
```json
{ "target_step": 42 }
```

**PROJECT:**
```json
{ "projection_depth": 5 }

**DAMP:**
```json
{ "damping_coefficient": 0.85 }
```

## 3. Tokenless Enforcement

GateReport and ActionPlan are structured objects validated against schemas.
No textual parsing is required.

### Validation Flow
```
GATE_CHECK Node
    │
    ▼
Evaluate Gate Predicate
    │
    ▼
Generate GateReport (validated against nsc_gate.schema.json)
    │
    ├─── Pass? ──▶ GOVERN Node
    │                  │
    ▼                  ▼
               Generate ActionPlan (validated against nsc_action_plan.schema.json)
```

## 4. Gate Definition

Gates are defined in the runtime, not in the module. A gate definition includes:

| Field | Type | Description |
|-------|------|-------------|
| `gate_id` | integer | Unique identifier |
| `name` | string | Human-readable name |
| `predicate` | function | Coherence threshold check |
| `thresholds` | object | Threshold values |

### Example Gate Definition (Runtime)
```json
{
  "gate_id": 1,
  "name": "coherence_gate",
  "predicate": "coherence_above_threshold",
  "thresholds": {
    "C_min": 0.95,
    "r_gate_max": 0.05,
    "r_cons_max": 0.02,
    "r_num_max": 0.10
  }
}
```

## 5. Governance History

The history buffer contains:
- Prior operator receipts
- Prior gate reports
- Prior action plans
- Braid events

This enables governance to make informed decisions based on run history.
