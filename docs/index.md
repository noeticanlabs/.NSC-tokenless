# NSC Tokenless v1.1 Documentation Index

## Getting Started
- [Overview](overview.md) - What NSC is and why tokenless matters
- [Core Model](nsc_core_model.md) - Module, graph, nodes, evaluation model
- [Execution Determinism](nsc_execution_determinism.md) - How deterministic execution works

## Language Reference
- [Module Linking](language/nsc_module_linking.md) - Combining multiple modules
- [Error Model](language/nsc_error_model.md) - Error handling and reporting

## Wire Format
- [Canonicalization](wire/canonicalization.md) - Deterministic JSON serialization
- [Canonical Hashing](wire/canonical_hashing.md) - SHA-256 digest format
- [JSON Profile](wire/json_profile.md) - JSON encoding rules
- [CBOR Profile](wire/cbor_profile.md) - Optional CBOR encoding

## Governance
- [Replay Verification](governance/replay_verification.md) - Reproducibility verification

## Security
- [Threat Model](security/threat_model.md) - Security considerations
- [Integrity](security/integrity.md) - Integrity guarantees
- [Safe Extensions](security/safe_extensions.md) - Extension security

## Glyphs (GLLL)
- [Codebook](glyphs/glll_codebook.md) - GLLL codebook specification
- [Decode](glyphs/glll_decode.md) - Signal decoding process
- [Reject Control Row](glyphs/reject_control_row.md) - REJECT handling
- [Ambiguity Recovery](glyphs/ambiguity_recovery.md) - Ambiguity resolution

## API Reference
- [CLI Commands](nsc_cli_reference.md) - Command-line interface reference
- [Python API](nsc_python_api.md) - Python module API
- [Web API](nsc_web_api.md) - REST API reference (FastAPI)

## Schemas
See `nsc/schemas/` directory for JSON Schema definitions:
- `nsc_module.schema.json` - Module structure
- `nsc_node.schema.json` - Node structure
- `nsc_operator_receipt.schema.json` - Receipt format
- `nsc_braid_event.schema.json` - Braid event format

## Examples
See `nsc/examples/` directory for executable examples.

## Version Information
- [Version Strategy](versions.md) - NPE version management (v0.1, v1.0, v2.0)
