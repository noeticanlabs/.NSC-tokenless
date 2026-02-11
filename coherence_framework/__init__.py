"""
Coherence Framework: Universal Field Equation and Coherence Governance

A unified theory of governed evolution providing:
1. A canonical form (Universal Field Equation) for expressing evolution laws
2. A governance layer (gates and rails) for enforcing affordability constraints
3. An accountability infrastructure (Ω-ledger) for auditable compliance

See https://github.com/noeticanlabs/coherence-framework for full documentation.
"""

__version__ = "2.0.0"
__author__ = "Noeticean Labs"
__license__ = "MIT AND UFE-CL-1.0"

from coherence_framework.schemas import get_schema_path, load_schema, validate_receipt
from coherence_framework.conformance import (
    SchemaValidator,
    HashChainValidator,
    ConformanceChecker,
)

__all__ = [
    "get_schema_path",
    "load_schema",
    "validate_receipt",
    "SchemaValidator",
    "HashChainValidator",
    "ConformanceChecker",
    "__version__",
]
