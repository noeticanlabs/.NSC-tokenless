# Index — NSC Tokenless v1.0

## Core Documentation

1. [Overview](overview.md) — what NSC is and why tokenless matters
2. [NSC Tokenless Format](nsc_format_tokenless.md) — canonical wire rules for tokenless encoding
3. [Core Model](nsc_core_model.md) — module, graph, nodes, evaluation model
4. [Type System](nsc_type_system.md) — types, checking, and well-formedness
5. [Operator Registry](nsc_operator_registry.md) — registry, IDs, invariants, extension rules
6. [Execution Determinism](nsc_execution_determinism.md) — determinism contract and numeric discipline
7. [Gates and Governance](nsc_gates_and_governance.md) — gate ABI and governance ABI
8. [Receipts and Braid](nsc_receipts_and_braid.md) — receipts + braid events (append-only audit)
9. [Validation Rules](nsc_validation_rules.md) — MUST-level validation rules for conformance
10. [Module Digest](nsc_module_digest.md) — module integrity verification

## Normative Artifacts

### Schemas (JSON Schema Draft 2020-12)
- [Module Schema](../schemas/nsc_module.schema.json)
- [Node Schema](../schemas/nsc_node.schema.json)
- [Type Schema](../schemas/nsc_type.schema.json)
- [Operator Registry Schema](../schemas/nsc_operator_registry.schema.json)
- [Gate Schema](../schemas/nsc_gate.schema.json)
- [Action Plan Schema](../schemas/nsc_action_plan.schema.json)
- [Operator Receipt Schema](../schemas/nsc_operator_receipt.schema.json)
- [Decode Report Schema](../schemas/nsc_decode_report.schema.json)
- [Braid Event Schema](../schemas/nsc_braid_event.schema.json)
- [Module Digest Schema](../schemas/nsc_module_digest.schema.json)

### Examples (Executable Artifacts)
- [Operator Registry v1](../examples/registries/operator_registry.v1.json)
- [GLLL Codebook Minimal](../examples/registries/glll_codebook_minimal.n16.json)
- [PSI Minimal Module](../examples/modules/psi_minimal.tokenless.json)
- [PSI Governed Module](../examples/modules/psi_governed.tokenless.json)
- [Braid Minimal](../examples/modules/braid_minimal.json)
- [Operator Receipt Sample](../examples/receipts/sample_operator_receipt.json)
- [Gate Report Sample](../examples/receipts/sample_gate_report.json)
- [Decode Report Reject Sample](../examples/receipts/sample_decode_report_reject.json)

## Quick Start

To validate an NSC module against this specification:

```bash
# Validate a module against the schema
ajv validate -s schemas/nsc_module.schema.json -d examples/modules/psi_minimal.tokenless.json

# Validate operator registry
ajv validate -s schemas/nsc_operator_registry.schema.json -d examples/registries/operator_registry.v1.json
```

## Related Documentation
- [Noetica Whitepaper](../../paper/whitepaper.tex) — theoretical foundations
- [Foundations Paper](../../paper/01_Foundations.md) — core concepts
