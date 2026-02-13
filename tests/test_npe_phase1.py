"""
Test Phase 1 NPE v2.0 Enhancements

Tests for:
- Enhanced type matching with coercion
- Real cost functions for beam search
- Argument resolution with type safety
- ExecutionTracer pattern learning
"""

import pytest
from nsc.npe_v2_0 import (
    BeamSearchSynthesizer, ExecutionTracer, OperatorReceipt, ReceiptLedger,
    CoherenceMetrics, ChainPattern, OperatorProfile, OperatorCandidate
)
from nsc.npe_v1_0 import (
    TypeTag, SCALAR, PHASE, FIELD,
    OpSig, Operator, OperatorRegistry, KernelBinding
)
import json


class TestTypeMatching:
    """Test enhanced type matching with coercion."""
    
    def test_exact_type_match(self):
        """Exact types should match."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        assert synth._types_match("SCALAR", "SCALAR") is True
        assert synth._types_match("FLOAT", "FLOAT") is True
        assert synth._types_match({"kind": "FIELD"}, {"kind": "FIELD"}) is True
    
    def test_numeric_family_coercion(self):
        """Numeric types should coerce to each other."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        # All numeric types should match
        assert synth._types_match("SCALAR", "FLOAT") is True
        assert synth._types_match("FLOAT", "DOUBLE") is True
        assert synth._types_match("INT", "REAL") is True
        assert synth._types_match("SCALAR", "INT") is True
    
    def test_algebraic_family_coercion(self):
        """Algebraic types should coerce to each other."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        assert synth._types_match("PHASE", "COMPLEX") is True
        assert synth._types_match("COMPLEX", "QUATERNION") is True
    
    def test_collection_family_coercion(self):
        """Collection types should coerce."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        assert synth._types_match("FIELD_REF", "FIELD") is True
        assert synth._types_match("FIELD(SCALAR)", "FIELD(INT)") is True
    
    def test_no_cross_family_coercion(self):
        """Types from different families should not match."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        assert synth._types_match("SCALAR", "PHASE") is False
        assert synth._types_match("FLOAT", "COMPLEX") is False
    
    def test_no_coercion_mode(self):
        """Strict mode should only allow exact matches."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        assert synth._types_match("SCALAR", "FLOAT", allow_coercion=False) is False
        assert synth._types_match("SCALAR", "SCALAR", allow_coercion=False) is True


class TestCostFunctions:
    """Test realistic cost computation."""
    
    def test_base_cost_varies_by_operator(self):
        """Different operators should have different base costs."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        # Source operators (10, 11, 12) should be cheaper
        cost_10 = synth._compute_operator_cost(10, depth=0, chain_length=0, preserves_coherence=False)
        cost_20 = synth._compute_operator_cost(20, depth=0, chain_length=0, preserves_coherence=False)
        
        assert cost_10 < cost_20, "Source operators should be cheaper"
    
    def test_depth_penalty_increases(self):
        """Cost should increase with depth."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        cost_0 = synth._compute_operator_cost(10, depth=0, chain_length=0, preserves_coherence=False)
        cost_5 = synth._compute_operator_cost(10, depth=5, chain_length=0, preserves_coherence=False)
        cost_10 = synth._compute_operator_cost(10, depth=10, chain_length=0, preserves_coherence=False)
        
        assert cost_0 < cost_5 < cost_10, "Deeper chains should cost more"
    
    def test_coherence_bonus(self):
        """Operators that preserve coherence should be cheaper."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        cost_good = synth._compute_operator_cost(10, depth=2, chain_length=0, preserves_coherence=True)
        cost_bad = synth._compute_operator_cost(10, depth=2, chain_length=0, preserves_coherence=False)
        
        assert cost_good < cost_bad, "Coherence-preserving ops should be cheaper"
    
    def test_minimum_cost_floor(self):
        """Cost should never go below minimum floor."""
        synth = BeamSearchSynthesizer(registry=None, binding=None, beam_width=5)
        
        cost = synth._compute_operator_cost(10, depth=0, chain_length=0, preserves_coherence=True)
        assert cost >= 0.02, "Cost should be above floor of 0.02"


class TestExecutionTracer:
    """Test receipt ingestion and pattern learning."""
    
    def test_ingest_single_receipt(self):
        """Should ingest a single receipt ledger."""
        tracer = ExecutionTracer()
        
        receipt = OperatorReceipt(
            kind="operator_receipt",
            event_id="evt_001",
            node_id=1,
            op_id=10,
            C_before=0.0,
            C_after=0.1,
            r_phys=0.05,
            r_cons=0.02,
            r_num=0.01
        )
        
        ledger = ReceiptLedger()
        ledger.add_receipt(receipt)
        
        tracer.ingest_receipts(ledger)
        
        assert tracer.total_executions == 1
        assert tracer.successful_executions == 1  # Debt increase < 0.1
        assert 10 in tracer.operator_profiles
    
    def test_operator_success_rate(self):
        """Should track operator success rates."""
        tracer = ExecutionTracer()
        
        # Successful execution
        receipt1 = OperatorReceipt(
            op_id=10, C_before=0.0, C_after=0.05,
            r_phys=0.01, r_cons=0.01, r_num=0.01
        )
        ledger1 = ReceiptLedger()
        ledger1.add_receipt(receipt1)
        tracer.ingest_receipts(ledger1)
        
        # Failed execution
        receipt2 = OperatorReceipt(
            op_id=10, C_before=0.0, C_after=0.2,  # Debt exploded
            r_phys=0.1, r_cons=0.1, r_num=0.01
        )
        ledger2 = ReceiptLedger()
        ledger2.add_receipt(receipt2)
        tracer.ingest_receipts(ledger2)
        
        # Should be 50% success rate
        success_rate = tracer.operator_success_rate(10)
        assert 0.4 < success_rate < 0.6, f"Expected ~0.5, got {success_rate}"
    
    def test_chain_pattern_recognition(self):
        """Should recognize repeated chain patterns."""
        tracer = ExecutionTracer()
        
        # Same chain twice
        for _ in range(2):
            receipt1 = OperatorReceipt(op_id=10, C_before=0.0, C_after=0.05, r_phys=0.01, r_cons=0.01, r_num=0.01)
            receipt2 = OperatorReceipt(op_id=11, C_before=0.05, C_after=0.08, r_phys=0.02, r_cons=0.01, r_num=0.01)
            
            ledger = ReceiptLedger()
            ledger.add_receipt(receipt1)
            ledger.add_receipt(receipt2)
            tracer.ingest_receipts(ledger)
        
        # Should have exactly 1 unique chain pattern
        assert len(tracer.chain_patterns) == 1
        
        # That pattern should have count=2
        pattern_data = list(tracer.chain_patterns.values())[0]
        assert pattern_data.count == 2
    
    def test_operator_compatibility(self):
        """Should compute operator pair compatibility."""
        tracer = ExecutionTracer()
        
        # Execute chain: 10 -> 11
        for _ in range(3):
            receipt1 = OperatorReceipt(op_id=10, C_before=0.0, C_after=0.05, r_phys=0.01, r_cons=0.01, r_num=0.01)
            receipt2 = OperatorReceipt(op_id=11, C_before=0.05, C_after=0.08, r_phys=0.02, r_cons=0.01, r_num=0.01)
            
            ledger = ReceiptLedger()
            ledger.add_receipt(receipt1)
            ledger.add_receipt(receipt2)
            tracer.ingest_receipts(ledger)
        
        # Pair (10, 11) should be observed
        compat = tracer.operator_compatibility(10, 11)
        assert compat > 0.5, "Observed pair should have good compatibility"
        
        # Pair (99, 100) should not be observed  
        compat = tracer.operator_compatibility(99, 100)
        assert compat == 0.5, "Unobserved pairs should be neutral (0.5)"
    
    def test_chain_debt_histogram(self):
        """Should compute debt statistics."""
        tracer = ExecutionTracer()
        
        # Execute a few chains
        for i in range(5):
            receipt = OperatorReceipt(
                op_id=10,
                C_before=0.0,
                C_after=0.05 + i * 0.01,  # Varying debt
                r_phys=0.01, r_cons=0.01, r_num=0.01
            )
            ledger = ReceiptLedger()
            ledger.add_receipt(receipt)
            tracer.ingest_receipts(ledger)
        
        histogram = tracer.chain_debt_histogram()
        
        assert "avg" in histogram
        assert "min" in histogram
        assert "max" in histogram
        assert histogram["total_executions"] == 5
    
    def test_failure_mode_analysis(self):
        """Should identify high-risk operators."""
        tracer = ExecutionTracer()
        
        # Create operator with high failure rate
        for i in range(10):
            receipt = OperatorReceipt(
                op_id=99,
                C_before=0.0,
                C_after=0.2 if i < 8 else 0.05,  # Fails 80% of the time
                r_phys=0.1 if i < 8 else 0.01,
                r_cons=0.1 if i < 8 else 0.01,
                r_num=0.01
            )
            ledger = ReceiptLedger()
            ledger.add_receipt(receipt)
            tracer.ingest_receipts(ledger)
        
        analysis = tracer.failure_mode_analysis()
        
        # Op 99 should be in high-risk list
        high_risk_ops = analysis["high_risk_operators"]
        high_risk_op_ids = [op_id for op_id, _ in high_risk_ops]
        assert 99 in high_risk_op_ids, "Low-success-rate op should be flagged as high-risk"
    
    def test_summary_report(self):
        """Should generate comprehensive summary."""
        tracer = ExecutionTracer()
        
        # Add some mixed executions
        for i in range(20):
            success = i % 3 != 0  # 2/3 success rate
            
            receipt = OperatorReceipt(
                op_id=10 + (i % 3),
                C_before=0.0,
                C_after=0.05 if success else 0.2,
                r_phys=0.01, r_cons=0.01, r_num=0.01
            )
            ledger = ReceiptLedger()
            ledger.add_receipt(receipt)
            tracer.ingest_receipts(ledger)
        
        summary = tracer.summary()
        
        assert summary["total_executions"] == 20
        assert "success_rate" in summary
        assert "unique_operators" in summary
        assert "debt_stats" in summary
        assert "top_chains" in summary
        assert "top_operators" in summary
        assert "failure_analysis" in summary


class TestChainPattern:
    """Test ChainPattern data structure."""
    
    def test_success_rate_computation(self):
        """Should compute success rate correctly."""
        pattern = ChainPattern(chain_signature="test_chain")
        pattern.count = 10
        pattern.success_count = 7
        
        assert pattern.success_rate == 0.7
    
    def test_avg_debt_computation(self):
        """Should compute average debt."""
        pattern = ChainPattern(chain_signature="test_chain")
        pattern.count = 5
        pattern.total_debt = 0.5
        
        assert pattern.avg_debt == 0.1


class TestOperatorProfile:
    """Test OperatorProfile data structure."""
    
    def test_add_execution(self):
        """Should record executions."""
        profile = OperatorProfile(op_id=10)
        profile.add_execution(success=True, debt_contribution=0.05, residual_mag=0.02)
        
        assert profile.executions == 1
        assert profile.successes == 1
        assert profile.total_debt_contribution == 0.05
    
    def test_success_rate_tracking(self):
        """Should track success rate."""
        profile = OperatorProfile(op_id=10)
        profile.add_execution(success=True, debt_contribution=0.05, residual_mag=0.02)
        profile.add_execution(success=False, debt_contribution=0.2, residual_mag=0.1)
        profile.add_execution(success=True, debt_contribution=0.03, residual_mag=0.01)
        
        assert profile.executions == 3
        assert profile.successes == 2
        assert profile.success_rate == 2/3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
