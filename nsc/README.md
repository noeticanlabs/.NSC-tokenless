# Noetican .NSC Tokenless Language (v1.0)

NSC (.NSC) is the Noetican executable layer: a deterministic, auditable, tokenless
intermediate language for governed computation.

**Tokenless** means: a conforming runtime does not require any textual tokens for parsing.
Programs are executed from structured objects (IDs, vectors, typed fields, nodes, and receipts).

## What NSC guarantees
- Deterministic evaluation order (explicit sequencing)
- Typed operator application (no implicit coercions)
- Mandatory gate hooks + governance decisions (optional, but standardized)
- Mandatory receipts (audit trail) and braid events (time-structured ledger)

## Directory Structure
- `docs/` - Specification documents
- `schemas/` - JSON Schema definitions (normative)
- `examples/` - Executable examples demonstrating conformance

## Entry Points
- [Index](docs/index.md) - Navigation and document overview
- [Overview](docs/overview.md) - What NSC is and why tokenless matters
- [Core Model](docs/nsc_core_model.md) - Module, graph, nodes, evaluation model

## Integration with Noetica Workspace
This specification complements the existing [paper/](../paper/whitepaper.tex) directory,
providing the executable layer for the theoretical foundations described there.

## Version
- **Version**: 1.0.0
- **Status**: Release Candidate
- **License**: See [LICENSE.md](LICENSE.md)

## Contributing
This specification uses JSON Schema Draft 2020-12 for normative definitions.
All examples are designed to be immediately validatable against the schemas.
