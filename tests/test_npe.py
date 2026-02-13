"""Test NPE modules (npe_v0_1, npe_v1_0, npe_v2_0)."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.npe_v1_0 import (
    NoeticanProposalEngine_v1_0,
    Module,
    OperatorRegistry,
    KernelBinding,
)


class TestNPEv1_0:
    """Tests for NPE v1.0."""

    def test_npe_creation(self):
        """Test creating NPE v1.0."""
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        npe = NoeticanProposalEngine_v1_0(
            registry=registry,
            binding=binding,
            templates=[],
            coherence_gate=MagicMock(),
            weights=MagicMock()
        )
        assert npe is not None

    def test_npe_has_required_methods(self):
        """Test NPE has required methods."""
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        npe = NoeticanProposalEngine_v1_0(
            registry=registry,
            binding=binding,
            templates=[],
            coherence_gate=MagicMock(),
            weights=MagicMock()
        )
        
        assert hasattr(npe, 'propose')
        assert hasattr(npe.coherence_gate, 'evaluate')


class TestNPEv2_0:
    """Tests for NPE v2.0."""

    def test_coherence_metrics_creation(self):
        """Test CoherenceMetrics creation."""
        from nsc.npe_v2_0 import CoherenceMetrics
        
        metrics = CoherenceMetrics(
            r_phys=0.001,
            r_cons=0.002,
            r_num=0.0005
        )
        assert metrics.r_phys == 0.001

    def test_operator_receipt_creation(self):
        """Test OperatorReceipt creation."""
        from nsc.npe_v2_0 import OperatorReceipt
        
        receipt = OperatorReceipt(
            event_id="test",
            node_id=1,
            op_id=1001,
            kernel_id="k1",
            kernel_version="1.0",
            digest_in="",
            digest_out="",
            C_before=0.0,
            C_after=1.0,
            r_phys=0.001,
            r_cons=0.002,
            r_num=0.0005
        )
        assert receipt.event_id == "test"


class TestNPEIntegration:
    """Integration tests for NPE."""

    def test_full_proposal_pipeline(self):
        """Test full proposal pipeline."""
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        npe = NoeticanProposalEngine_v1_0(
            registry=registry,
            binding=binding,
            templates=[],
            coherence_gate=MagicMock(),
            weights=MagicMock()
        )
        assert npe is not None
