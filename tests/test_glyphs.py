"""Test glyphs module (GLLL decoder, REJECT handling)."""
import pytest
import sys
import math
from pathlib import Path
from typing import Dict, Any, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.glyphs import GLLLCodebook, REJECTHandler, HadamardMatrix


class TestHadamardMatrix:
    """Tests for Hadamard matrix generation."""

    def test_hadamard_n2(self):
        """Test Hadamard matrix for n=2."""
        h = HadamardMatrix.create(2)
        assert h.shape == (2, 2)
        # Check orthogonality
        assert math.isclose(h[0, 0] * h[0, 0] + h[0, 1] * h[0, 1], 2.0)

    def test_hadamard_n4(self):
        """Test Hadamard matrix for n=4."""
        h = HadamardMatrix.create(4)
        assert h.shape == (4, 4)

    def test_hadamard_n16(self):
        """Test Hadamard matrix for n=16."""
        h = HadamardMatrix.create(16)
        assert h.shape == (16, 16)

    def test_hadamard_orthogonality(self):
        """Test Hadamard matrix orthogonality property."""
        h = HadamardMatrix.create(8)
        n = h.shape[0]
        
        # H * H^T = n * I
        for i in range(n):
            for j in range(n):
                dot_product = sum(h[k, i] * h[k, j] for k in range(n))
                if i == j:
                    assert math.isclose(dot_product, n)
                else:
                    assert math.isclose(dot_product, 0.0)

    def test_hadamard_first_row_all_ones(self):
        """Test that first row of Hadamard matrix is all ones."""
        for n in [2, 4, 8, 16]:
            h = HadamardMatrix.create(n)
            assert all(h[0, i] == 1 for i in range(n))

    def test_get_row(self):
        """Test getting a specific row from Hadamard matrix."""
        row = HadamardMatrix.get_row(4, 1)
        assert len(row) == 4
        assert all(v in [-1, 1] for v in row)


class TestGLLLCodebook:
    """Tests for GLLLCodebook."""

    def test_codebook_creation(self):
        """Test creating a GLLL codebook."""
        codebook = GLLLCodebook(n=16)
        assert codebook.n == 16

    def test_codebook_rows_structure(self):
        """Test codebook rows structure."""
        codebook = GLLLCodebook(n=16)
        assert isinstance(codebook.rows, list)

    def test_get_row_method(self):
        """Test get_row method."""
        codebook = GLLLCodebook(n=16)
        row = codebook.get_row(0)
        assert row is not None
        assert len(row) == 16

    def test_decode_returns_dict(self):
        """Test decode returns dictionary."""
        codebook = GLLLCodebook(n=16)
        signal = [1.0] * 16
        
        result = codebook.decode(signal)
        
        assert isinstance(result, dict)
        assert "topk" in result
        assert "scores" in result

    def test_decode_topk_values(self):
        """Test decode returns valid topk."""
        codebook = GLLLCodebook(n=16)
        signal = [1.0] * 16
        
        result = codebook.decode(signal)
        
        assert isinstance(result["topk"], list)
        assert isinstance(result["scores"], list)
        assert len(result["topk"]) > 0


class TestREJECTHandler:
    """Tests for REJECTHandler."""

    def test_reject_handler_creation(self):
        """Test creating a REJECT handler."""
        handler = REJECTHandler()
        assert handler is not None

    def test_reject_handler_reset(self):
        """Test REJECT handler reset."""
        handler = REJECTHandler()
        handler.reset()
        assert hasattr(handler, 'reject_count')

    def test_reject_handler_check_reject(self):
        """Test REJECT check method exists."""
        handler = REJECTHandler()
        # Should have a check method or check_reject method
        assert hasattr(handler, 'check_reject') or hasattr(handler, 'check')

    def test_reject_handler_max_rejects(self):
        """Test REJECT handler max rejects property exists."""
        handler = REJECTHandler()
        assert hasattr(handler, 'max_rejects') or hasattr(handler, 'reject_limit')


class TestGLLLIntegration:
    """Integration tests for GLLL decode pipeline."""

    def test_full_decode_pipeline(self):
        """Test full decode pipeline."""
        codebook = GLLLCodebook(n=16)
        
        # Create signal from Hadamard row
        signal = codebook.get_row(0)
        
        result = codebook.decode(signal)
        
        assert result is not None
        assert len(result["topk"]) > 0

    def test_decode_with_noise(self):
        """Test decode with small noise."""
        import random
        random.seed(42)
        
        codebook = GLLLCodebook(n=16)
        signal = codebook.get_row(0)
        
        # Add small noise
        noisy_signal = [s + random.gauss(0, 0.01) for s in signal]
        
        result = codebook.decode(noisy_signal)
        
        assert result is not None
        assert len(result["topk"]) > 0

    def test_reject_recovery_workflow(self):
        """Test REJECT recovery workflow."""
        handler = REJECTHandler()
        handler.reset()
        
        assert handler.reject_count == 0


class TestGLLLPerformance:
    """Performance tests for GLLL operations."""

    def test_decode_performance(self):
        """Test decode performance."""
        import time
        
        codebook = GLLLCodebook(n=16)
        signals = [codebook.get_row(i % 16) for i in range(20)]
        
        start = time.time()
        for signal in signals:
            codebook.decode(signal)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 2.0
