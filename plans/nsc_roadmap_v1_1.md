# NSC Tokenless v1.1 - Complete Roadmap

**Version:** 1.1.0  
**Date:** 2026-02-11  
**Status:** Planning

---

## Executive Summary

This roadmap outlines the complete implementation plan for NSC Tokenless v1.1, covering specification finalization, runtime implementation, testing, packaging, and web interface development.

**Phase Structure:**
- **Phase 0**: Spec Review & Consistency (Foundation)
- **Phase 1**: Runtime Execution & L0-L2 Conformance
- **Phase 2**: Testing & Packaging
- **Phase 3**: Web Interface
- **Phase 4**: Publication & Release

---

## Phase 0: Specification Review & Consistency

**Duration:** 1 week  
**Objective:** Ensure NSC and Coherence specs are consistent and complete

### 0.1 Spec Audit

| Task | Deliverable | Owner |
|------|-------------|-------|
| Cross-reference NSC spec with parent Coherence Framework | Gap report | TBD |
| Verify all schema references are correct | Schema validation report | TBD |
| Check terminology alignment | Terminology matrix | TBD |
| Verify test vectors match spec | Vector validation | TBD |

### 0.2 Deliverables

```
docs/
  ✓ terminology.md
  ✓ design_goals.md
  ✓ conformance_levels.md
  ✓ wire/canonicalization.md
  ✓ wire/canonical_hashing.md
  ✓ wire/json_profile.md
  ✓ wire/cbor_profile.md
  ✓ language/nsc_module_linking.md
  ✓ language/nsc_error_model.md
  ✓ governance/replay_verification.md
  ✓ glyphs/glll_codebook.md
  ✓ glyphs/glll_decode.md
  ✓ glyphs/reject_control_row.md
  ✓ glyphs/ambiguity_recovery.md
  ✓ security/threat_model.md
  ✓ security/integrity.md
  ✓ security/safe_extensions.md

schemas/
  ✓ nsc_kernel_binding.schema.json
  ✓ nsc_module_digest.schema.json
  ✓ nsc_replay_report.schema.json
  (existing: nsc_module.schema.json, nsc_node.schema.json, etc.)

test-vectors/
  ✓ canonical_json_hash_vectors.md
  ✓ hadamard_decode_vectors_n16.md
  ✓ conformance_cases.md

examples/
  ✓ registries/kernels.binding.example.json
  ✓ receipts/replay_report.example.json
  ✓ events/braid_event.decode_reject.json
```

### 0.3 Exit Criteria
- [ ] All spec files reviewed and approved
- [ ] Schema validation passes
- [ ] Terminology aligned between NSC and Coherence
- [ ] Test vectors validated

---

## Phase 1: Runtime Execution & L0-L2 Conformance

**Duration:** 3 weeks  
**Objective:** Implement full execution runtime with L0, L1, L2 conformance

### 1.1 NSC Runtime Engine

```
nsc/runtime/
  __init__.py
  executor.py        # Module execution engine
  evaluator.py       # Node evaluation (APPLY, FIELD_REF, SEQ, GATE_CHECK)
  interpreter.py     # Main interpreter loop
  state.py          # Runtime state management
  environment.py    # Field environment
```

**Key Classes:**
- `Executor`: Executes NSC modules with deterministic ordering
- `NodeEvaluator`: Evaluates individual nodes
- `RuntimeState`: Tracks execution state, field values
- `Environment`: Maps field_id to actual data

### 1.2 L0 Conformance Implementation

| Requirement | Implementation |
|-------------|----------------|
| Module validation | `SchemaValidator` against `nsc_module.schema.json` |
| Registry resolution | `OperatorRegistry.require(op_id)` |
| Kernel binding | `KernelBinding.require(op_id)` |
| SEQ ordering | `Interpreter.eval_sequence(seq_id)` |
| OperatorReceipt emission | `ReceiptEmitter.emit_operator_receipt()` |
| Canonical hashing | `canonical_json()` + `sha256_hex()` |

### 1.3 L1 Governed Implementation

| Requirement | Implementation |
|-------------|----------------|
| GateCheck evaluation | `GateChecker.evaluate(node, state)` |
| GateReport schema | `nsc_gate_report.schema.json` |
| ActionPlan generation | `GovernanceEngine.evaluate(reports)` |
| ReplayReport generation | `ReplayVerifier.verify(module, receipts)` |

### 1.4 L2 Identity-Aware Implementation

| Requirement | Implementation |
|-------------|----------------|
| GLLL decode | `GLLLDecoder.decode(vector, codebook)` |
| REJECT handling | `RejectHandler.emit_report(decode_result)` |
| DecodeReport schema | `nsc_decode_report.schema.json` |
| Ambiguity recovery | `RecoveryEngine.apply(decode_report)` |

### 1.5 Phase 1 Deliverables

```
nsc/runtime/
  executor.py        # Core execution engine
  evaluator.py       # Node evaluation
  interpreter.py     # Main loop
  state.py          # State management
  environment.py    # Field environment

nsc/governance/
  gate_checker.py   # GateCheck evaluation
  gate_report.py    # GateReport generation
  governance.py     # ActionPlan generation
  replay.py         # Replay verification

nsc/glyphs/
  glll.py           # GLLL decode
  codebook.py       # Codebook handling
  reject.py         # REJECT control row

schemas/
  nsc_gate_report.schema.json
  nsc_decode_report.schema.json
  nsc_action_plan.schema.json
  nsc_replay_report.schema.json  (existing)

tests/
  test_runtime_executor.py
  test_governance.py
  test_glyphs.py
```

### 1.6 Exit Criteria
- [ ] L0: All modules execute with receipts
- [ ] L1: Gates, governance, replay work
- [ ] L2: Decode, REJECT, recovery work
- [ ] All conformance tests pass

---

## Phase 2: Testing & Packaging

**Duration:** 2 weeks  
**Objective:** Comprehensive testing and pip-installable package

### 2.1 Test Suite Structure

```
tests/
  __init__.py
  conftest.py           # Pytest configuration
  
  unit/
    test_canonicalization.py
    test_hashing.py
    test_types.py
    test_registry.py
    test_binding.py
    
  integration/
    test_module_execution.py
    test_proposal_engine.py
    test_governance.py
    test_replay.py
    
  conformance/
    test_l0_conformance.py
    test_l1_conformance.py
    test_l2_conformance.py
    
  test_vectors/
    test_canonical_vectors.py
    test_decode_vectors.py
    test_conformance_cases.py
```

### 2.2 Test Coverage Targets

| Category | Coverage Target |
|----------|-----------------|
| Core types | 100% |
| Canonicalization | 100% |
| Hash chaining | 100% |
| Registry/Binding | 100% |
| Proposal engine | 90% |
| Runtime execution | 90% |
| Governance | 85% |
| Glyphs | 80% |

### 2.3 Packaging

**`pyproject.toml`:**
```toml
[project]
name = "nsc-tokenless"
version = "1.1.0"
description = "NSC Tokenless v1.1 - Noetican Structured Code"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "jsonschema>=4.17.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]

[tool.coverage.run]
source = ["nsc"]

[tool.black]
line-length = 100
target-version = ["py310"]
```

### 2.4 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest --cov=nsc --cov-report=xml
      - run: coverage-badge -o coverage.svg

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install black mypy
      - run: black --check nsc/
      - run: mypy nsc/
```

### 2.5 Phase 2 Deliverables

```
tests/
  (all test files above)

pyproject.toml
.github/workflows/ci.yml
CONTRIBUTING.md
```

### 2.6 Exit Criteria
- [ ] Unit tests: >95% coverage
- [ ] Integration tests: All scenarios pass
- [ ] Conformance tests: L0-L2 pass
- [ ] Package installs with `pip install -e .`
- [ ] CI pipeline passes

---

## Phase 3: Web Interface

**Duration:** 3 weeks  
**Objective:** React-based web interface for NPE

### 3.1 Architecture

```
src/
  App.tsx              # Main app
  main.tsx             # Entry point
  
  components/
    proposal/
      ProposalList.tsx
      ProposalCard.tsx
      ProposalDetail.tsx
      
    coherence/
      MetricsPanel.tsx
      DebtChart.tsx
      GateStatus.tsx
      
    module/
      ModuleViewer.tsx
      NodeGraph.tsx
      
    editor/
      TemplateEditor.tsx
      CodeEditor.tsx
      
  hooks/
    useNPE.ts          # NPE API hook
    useProposals.ts    # Proposal state
    useMetrics.ts      # Coherence metrics
    
  lib/
    api.ts             # NPE API client
    nsc.ts             # NSC type definitions
    coherence.ts       # Coherence utilities
    
  pages/
    Home.tsx
    Proposals.tsx
    Modules.tsx
    Governance.tsx
    Settings.tsx
```

### 3.2 Features

| Feature | Priority |
|---------|----------|
| Proposal list with filtering | P0 |
| Proposal detail view | P0 |
| Coherence metrics dashboard | P0 |
| Module visualizer | P1 |
| Template editor | P1 |
| Governance panel | P1 |
| Real-time metrics | P2 |
| Export proposals | P2 |

### 3.3 API Endpoints

```
POST /api/propose
  Body: { context, templates, weights }
  Response: { proposals: [...], ledger_hash }

POST /api/execute
  Body: { module_id, module_digest }
  Response: { receipts: [...], final_metrics }

POST /api/replay
  Body: { module, receipts, tolerances }
  Response: { replay_pass, mismatches }

GET /api/registries/{id}
GET /api/bindings/{id}
GET /api/templates
```

### 3.4 Phase 3 Deliverables

```
src/
  (all frontend files)

api/
  server.py           # FastAPI server
  routes/
    proposals.py
    execute.py
    replay.py
    registries.py

package.json  (update)
vite.config.js  (update)
```

### 3.5 Exit Criteria
- [ ] Proposal list and detail views work
- [ ] Coherence metrics display correctly
- [ ] API endpoints functional
- [ ] Responsive design
- [ ] Dark mode support

---

## Phase 4: Publication & Release

**Duration:** 1 week  
**Objective:** Finalize and publish v1.1

### 4.1 Release Checklist

- [ ] Final code review
- [ ] Update CHANGELOG.md
- [ ] Tag release: `v1.1.0`
- [ ] Publish to PyPI
- [ ] Update GitHub release
- [ ] Update documentation

### 4.2 CHANGELOG.md Structure

```markdown
# Changelog

## [1.1.0] - 2026-02-?? 

### Added
- Complete NSC Tokenless v1.1 specification
- NPE v0.1, v1.0, v2.0 implementations
- Beam-search synthesis
- Real coherence metrics (r_phys, r_cons, r_num)
- OperatorReceipt emission
- L0, L1, L2 conformance

### Changed
- (List breaking changes)

### Fixed
- (List bug fixes)

### Removed
- (List deprecated features)
```

### 4.3 Versioning Strategy

**Semantic Versioning:**
- `MAJOR`: Breaking changes to spec
- `MINOR`: New features, backward compatible
- `PATCH`: Bug fixes, spec clarifications

**Specification Versioning:**
- NSC Tokenless: `1.1.0` (this release)
- Coherence Framework: `1.0.x` (from parent)

### 4.4 Phase 4 Deliverables

```
CHANGELOG.md (updated)
RELEASE_NOTES.md
.github/ISSUE_TEMPLATE/
.github/PULL_REQUEST_TEMPLATE.md
```

### 4.5 Exit Criteria
- [ ] Release tagged and published
- [ ] PyPI package available
- [ ] Documentation updated
- [ ] GitHub release created

---

## Resource Estimates

| Phase | Developer Days | Dependencies |
|-------|----------------|--------------|
| Phase 0 | 5 days | - |
| Phase 1 | 15 days | Phase 0 |
| Phase 2 | 10 days | Phase 1 |
| Phase 3 | 15 days | Phase 1 |
| Phase 4 | 5 days | All phases |

**Total: 50 developer days (~10 weeks at 5 days/week)**

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Spec inconsistencies | High | Medium | Phase 0 review |
| Runtime complexity | Medium | Medium | Incremental development |
| Test coverage gaps | Medium | Low | CI enforcement |
| Frontend delays | Low | Low | Phase 3 depends on Phase 1 API |
| Packaging issues | Low | Low | Standard tooling |

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Spec compliance | 100% (L0-L2) |
| Test coverage | >95% |
| API uptime | 99.9% |
| Documentation completeness | 100% |
| PyPI downloads (month 1) | 1000 |

---

## Appendix A: File Inventory

### NSC Specification Files

```
docs/
├── terminology.md
├── design_goals.md
├── conformance_levels.md
├── wire/
│   ├── canonicalization.md
│   ├── canonical_hashing.md
│   ├── json_profile.md
│   └── cbor_profile.md
├── language/
│   ├── nsc_module_linking.md
│   └── nsc_error_model.md
├── governance/
│   └── replay_verification.md
├── glyphs/
│   ├── glll_codebook.md
│   ├── glll_decode.md
│   ├── reject_control_row.md
│   └── ambiguity_recovery.md
└── security/
    ├── threat_model.md
    ├── integrity.md
    └── safe_extensions.md

schemas/
├── nsc_kernel_binding.schema.json
├── nsc_module_digest.schema.json
├── nsc_replay_report.schema.json
├── nsc_gate_report.schema.json
├── nsc_decode_report.schema.json
├── nsc_action_plan.schema.json
├── nsc_module.schema.json
├── nsc_node.schema.json
├── nsc_type.schema.json
├── nsc_operator_receipt.schema.json
├── nsc_operator_registry.schema.json
├── nsc_braid_event.schema.json
└── nsc_gate.schema.json

test-vectors/
├── canonical_json_hash_vectors.md
├── hadamard_decode_vectors_n16.md
└── conformance_cases.md

examples/
├── registries/kernels.binding.example.json
├── receipts/replay_report.example.json
└── events/braid_event.decode_reject.json
```

### NSC Implementation Files

```
nsc/
├── __init__.py
├── npe_v0_1.py      # Reference implementation
├── npe_v1_0.py      # Coherence-aligned
├── npe_v2_0.py      # Full upgrades
├── LICENSE.md
└── README.md

nsc/runtime/
├── __init__.py
├── executor.py
├── evaluator.py
├── interpreter.py
├── state.py
└── environment.py

nsc/governance/
├── __init__.py
├── gate_checker.py
├── gate_report.py
├── governance.py
└── replay.py

nsc/glyphs/
├── __init__.py
├── glll.py
├── codebook.py
└── reject.py

tests/
├── conftest.py
├── unit/
├── integration/
├── conformance/
└── test_vectors/
```

---

*Plan created: 2026-02-11*  
*Version: 1.0*
