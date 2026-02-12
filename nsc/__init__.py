"""
NSC Tokenless v1.1 - Noetican Executable Layer

A deterministic, auditable, tokenless intermediate language for governed computation.

Exports:
- NPE: Noetican Proposal Engine v2.0 (default, with beam-search synthesis)
- NPE_v1_0: Stable v1.0 implementation
- NPE_v0_1: Legacy v0.1 (deprecated)
"""

from nsc.npe_v2_0 import (
    NPE,
    CoherenceMetrics,
    GatePolicy,
    ReceiptValidator,
    OperatorReceipt,
    GateReport,
    GovernanceAction,
    Proposal,
    BeamSearchConfig,
)

# Import v1.0 for alternative use
from nsc.npe_v1_0 import (
    NPE as NPE_v1_0,
    SchemaValidator,
    ReceiptChainValidator,
)

# Import v0.1 with deprecation warning
import warnings as _warnings


def __getattr__(name):
    """Lazy import for deprecated v0_1 to avoid import overhead."""
    if name == "NPE_v0_1":
        _warnings.warn(
            "nsc.NPE_v0_1 is deprecated and will be removed in a future version. "
            "Use nsc.NPE (v2.0 default) or nsc.NPE_v1_0 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from nsc.npe_v0_1 import NPE as NPE_v0_1

        return NPE_v0_1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "1.1.0"

__all__ = [
    "NPE",  # Default: v2.0
    "NPE_v1_0",  # Stable alternative
    "NPE_v0_1",  # Deprecated, use NPE or NPE_v1_0
    "CoherenceMetrics",
    "GatePolicy",
    "ReceiptValidator",
    "OperatorReceipt",
    "GateReport",
    "GovernanceAction",
    "Proposal",
    "BeamSearchConfig",
    "SchemaValidator",
    "ReceiptChainValidator",
]
