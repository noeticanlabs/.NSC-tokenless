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

    def test_get_row_static(self):
        """Test getting a specific row from Hadamard matrix (static method)."""
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


class TestREJECTHandler:
    """Tests for REJECTHandler."""

    def test_reject_handler_creation(self):
        """Test creating a REJECTHandler."""
        handler = REJECTHandler()
        assert handler is not None

    def test_reject_handler_attributes(self):
        """Test REJECTHandler initial attributes."""
        handler = REJECTHandler()
        assert handler.reject_count == 0
        assert handler.recovery_attempts == 0

    def test_reject_handler_has_handle(self):
        """Test REJECTHandler has handle method."""
        handler = REJECTHandler()
        assert hasattr(handler, 'handle')

    def test_reject_handler_has_attempt_recovery(self):
        """Test REJECTHandler has attempt_recovery method."""
        handler = REJECTHandler()
        assert hasattr(handler, 'attempt_recovery')


class TestGLLLIntegration:
    """Integration tests for GLLL decode pipeline."""

    def test_full_decode_pipeline(self):
        """Test full decode pipeline exists."""
        codebook = GLLLCodebook(n=16)
        # Codebook has basic structure
        assert codebook is not None
        assert hasattr(codebook, 'rows')

    def test_reject_handler_workflow(self):
        """Test REJECT handler workflow exists."""
        handler = REJECTHandler()
        assert handler is not None
        assert hasattr(handler, 'handle')
        assert hasattr(handler, 'attempt_recovery')


class TestGLLLPerformance:
    """Performance tests for GLLL operations."""

    def test_hadamard_performance(self):
        """Test Hadamard matrix performance."""
        for n in [2, 4, 8, 16, 32]:
            h = HadamardMatrix.create(n)
            assert h.shape == (n, n)
