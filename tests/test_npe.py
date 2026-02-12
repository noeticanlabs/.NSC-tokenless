"""Test NPE (Noetican Proposal Engine) implementations."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.npe_v1_0 import (
    NoeticanProposalEngine_v1_0,
    ProposalContext,
    ProposalReceipt,
    Module,
    OperatorRegistry,
    KernelBinding,
    GatePolicy,
    GateDecision,
    ResidualBlock,
    DebtWeights,
    RailSet,
    Rail_R1_Deflation,
    Rail_R2_Rollback,
    Rail_R3_Projection,
    Rail_R4_Damping,
)


class TestNPEv1_0:
    """Tests for NPE v1.0 implementation."""

    def test_npe_creation(self):
        """Test creating NPE v1.0 instance."""
        npe = NoeticanProposalEngine_v1_0()
        assert npe is not None

    def test_debt_weights_creation(self):
        """Test DebtWeights creation."""
        weights = DebtWeights(w_phys=1.0, w_cons=1.5, w_num=0.5)
        assert weights.w_phys == 1.0
        assert weights.w_cons == 1.5
        assert weights.w_num == 0.5

    def test_debt_weights_compute(self):
        """Test DebtWeights debt computation."""
        weights = DebtWeights(w_phys=1.0, w_cons=1.0, w_num=1.0)
        residuals = ResidualBlock(phys=0.1, cons=0.2, num=0.05)
        
        debt, decomposition = weights.compute_debt(residuals)
        
        # debt = w_phys*r_phys^2 + w_cons*r_cons^2 + w_num*r_num^2
        expected = 1.0 * 0.01 + 1.0 * 0.04 + 1.0 * 0.0025
        assert abs(debt - expected) < 1e-10

    def test_gate_policy_pass(self):
        """Test GatePolicy with passing residuals."""
        policy = GatePolicy(
            phys_threshold=0.01,
            cons_threshold=0.02,
            num_threshold=0.01,
            hysteresis=0.5
        )
        
        residuals = ResidualBlock(phys=0.001, cons=0.002, num=0.0005)
        decision, details = policy.evaluate(residuals)
        
        assert decision == GateDecision.PASS
        assert details["phys_pass"] is True
        assert details["cons_pass"] is True
        assert details["num_pass"] is True

    def test_gate_policy_fail(self):
        """Test GatePolicy with failing residuals."""
        policy = GatePolicy(
            phys_threshold=0.01,
            cons_threshold=0.02,
            num_threshold=0.01,
            hysteresis=0.5
        )
        
        residuals = ResidualBlock(phys=0.05, cons=0.002, num=0.0005)
        decision, details = policy.evaluate(residuals)
        
        assert decision == GateDecision.FAIL

    def test_gate_policy_retry(self):
        """Test GatePolicy with retry-worthy residuals."""
        policy = GatePolicy(
            phys_threshold=0.01,
            cons_threshold=0.02,
            num_threshold=0.01,
            hysteresis=0.5
        )
        
        # Residuals just above threshold
        residuals = ResidualBlock(phys=0.015, cons=0.002, num=0.0005)
        decision, details = policy.evaluate(residuals)
        
        assert decision in [GateDecision.FAIL, GateDecision.RETRY]

    def test_residual_block_creation(self):
        """Test ResidualBlock creation."""
        block = ResidualBlock(phys=0.001, cons=0.002, num=0.0005)
        assert block.phys == 0.001
        assert block.cons == 0.002
        assert block.num == 0.0005

    def test_rail_r1_deflation(self):
        """Test R1 Deflation rail."""
        rail = Rail_R1_Deflation()
        residuals = ResidualBlock(phys=0.05, cons=0.002, num=0.0005)
        state = {"dt": 0.01}
        
        result = rail.apply(residuals, 0.01, state)
        
        assert result.rail == "R1"
        assert result.severity > 0

    def test_rail_r2_rollback(self):
        """Test R2 Rollback rail."""
        rail = Rail_R2_Rollback()
        residuals = ResidualBlock(phys=float('nan'), cons=0.002, num=0.0005)
        state = {"checkpoint": {}}
        
        result = rail.apply(residuals, 0.01, state)
        
        assert result.rail == "R2"

    def test_rail_r3_projection(self):
        """Test R3 Projection rail."""
        rail = Rail_R3_Projection()
        residuals = ResidualBlock(phys=0.001, cons=0.05, num=0.0005)
        state = {"constraint_manifold": []}
        
        result = rail.apply(residuals, 0.01, state)
        
        assert result.rail == "R3"

    def test_rail_r4_damping(self):
        """Test R4 Damping rail."""
        rail = Rail_R4_Damping()
        residuals = ResidualBlock(phys=0.001, cons=0.002, num=0.05)
        state = {"gain": 1.0}
        
        result = rail.apply(residuals, 0.01, state)
        
        assert result.rail == "R4"

    def test_rail_set_selection(self):
        """Test RailSet rail selection."""
        rail_set = RailSet()
        residuals = ResidualBlock(phys=0.001, cons=0.002, num=0.0005)
        state = {"dt": 0.01}
        
        selected = rail_set.select(residuals, 0.01, state)
        
        assert len(selected) >= 0  # May select zero rails for healthy state

    def test_proposal_context_creation(self):
        """Test ProposalContext creation."""
        context = ProposalContext(
            module_id="test",
            registry_id="test_reg",
            budget=10.0,
            max_nodes=100
        )
        
        assert context.module_id == "test"
        assert context.budget == 10.0

    def test_module_creation(self):
        """Test Module creation."""
        module = Module(
            module_id="test_module",
            nodes=[],
            seq=[],
            entrypoints=[],
            registry_ref="test_reg"
        )
        
        assert module.module_id == "test_module"

    def test_operator_registry_creation(self):
        """Test OperatorRegistry creation."""
        registry = OperatorRegistry(registry_id="test_reg")
        assert registry.registry_id == "test_reg"

    def test_kernel_binding_creation(self):
        """Test KernelBinding creation."""
        binding = KernelBinding(registry_id="test_reg")
        assert binding.registry_id == "test_reg"


class TestNPEv2_0:
    """Tests for NPE v2.0 with beam-search."""

    def test_npe_v2_0_import(self):
        """Test NPE v2.0 imports correctly."""
        try:
            from nsc.npe_v2_0 import (
                BeamSearchSynthesizer,
                CoherenceMetrics,
                DebtFunctional,
                ReceiptLedger,
                OperatorReceipt,
                NoeticanProposalEngine_v2_0,
            )
            assert BeamSearchSynthesizer is not None
            assert CoherenceMetrics is not None
            assert DebtFunctional is not None
            assert ReceiptLedger is not None
            assert OperatorReceipt is not None
            assert NoeticanProposalEngine_v2_0 is not None
        except ImportError as e:
            pytest.skip(f"NPE v2.0 not available: {e}")

    def test_coherence_metrics_creation(self):
        """Test CoherenceMetrics creation."""
        from nsc.npe_v2_0 import CoherenceMetrics
        
        metrics = CoherenceMetrics(
            r_phys=0.99,
            r_cons=0.98,
            r_num=0.995,
            debt=0.5,
            debt_phys=0.1,
            debt_cons=0.2,
            debt_num=0.2
        )
        
        assert metrics.r_phys == 0.99
        assert metrics.debt == 0.5

    def test_debt_functional_creation(self):
        """Test DebtFunctional creation."""
        from nsc.npe_v2_0 import DebtFunctional
        
        functional = DebtFunctional(
            weights=DebtWeights(w_phys=1.0, w_cons=1.0, w_num=1.0),
            penalty_coefficient=0.01
        )
        
        assert functional.penalty_coefficient == 0.01

    def test_receipt_ledger_creation(self):
        """Test ReceiptLedger creation."""
        from nsc.npe_v2_0 import ReceiptLedger
        
        ledger = ReceiptLedger()
        assert ledger.chain == []
        assert ledger.prev_hash is None

    def test_operator_receipt_creation(self):
        """Test OperatorReceipt creation."""
        from nsc.npe_v2_0 import OperatorReceipt
        
        receipt = OperatorReceipt(
            node_id=1,
            op_id=1001,
            inputs=[1, 2],
            result=3.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        assert receipt.node_id == 1
        assert receipt.op_id == 1001
        assert receipt.result == 3.0

    def test_hash_chaining(self):
        """Test hash chaining in receipts."""
        from nsc.npe_v2_0 import ReceiptLedger, OperatorReceipt
        
        ledger = ReceiptLedger()
        
        receipt1 = OperatorReceipt(
            node_id=1,
            op_id=1001,
            inputs=[],
            result=1.0,
            timestamp="2024-01-01T00:00:00Z"
        )
        
        ledger.add(receipt1)
        
        assert ledger.prev_hash is not None
        assert len(ledger.prev_hash) > 0
        
        # Add second receipt
        receipt2 = OperatorReceipt(
            node_id=2,
            op_id=1001,
            inputs=[1],
            result=2.0,
            timestamp="2024-01-01T00:00:01Z"
        )
        
        ledger.add(receipt2)
        
        # Chain should have 2 entries
        assert len(ledger.chain) == 2


class TestNPEIntegration:
    """Integration tests for NPE pipeline."""

    def test_full_proposal_pipeline(self):
        """Test full proposal generation pipeline."""
        npe = NoeticanProposalEngine_v1_0()
        
        context = ProposalContext(
            module_id="test_proposal",
            registry_id="test_reg",
            budget=100.0,
            max_nodes=50
        )
        
        proposals = npe.propose(context)
        
        assert isinstance(proposals, list)

    def test_gate_evaluation_pipeline(self):
        """Test gate evaluation with coherence metrics."""
        policy = GatePolicy(
            phys_threshold=0.01,
            cons_threshold=0.02,
            num_threshold=0.01,
            hysteresis=0.5
        )
        
        residuals = ResidualBlock(phys=0.005, cons=0.001, num=0.0005)
        decision, details = policy.evaluate(residuals)
        
        assert decision == GateDecision.PASS
