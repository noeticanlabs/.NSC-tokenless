"""Smoke tests for NSC Tokenless v1.1 modules."""
import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestModuleImports:
    """Basic import tests for all modules."""

    def test_import_runtime_state(self):
        """Test runtime.state module imports."""
        from nsc.runtime.state import RuntimeState, FieldHandle
        assert RuntimeState is not None
        assert FieldHandle is not None

    def test_import_runtime_environment(self):
        """Test runtime.environment module imports."""
        from nsc.runtime.environment import Environment
        assert Environment is not None

    def test_import_runtime_evaluator(self):
        """Test runtime.evaluator module imports."""
        from nsc.runtime.evaluator import NodeEvaluator
        assert NodeEvaluator is not None

    def test_import_runtime_executor(self):
        """Test runtime.executor module imports."""
        from nsc.runtime.executor import Executor
        assert Executor is not None

    def test_import_runtime_interpreter(self):
        """Test runtime.interpreter module imports."""
        from nsc.runtime.interpreter import Interpreter
        assert Interpreter is not None

    def test_import_governance_gate_checker(self):
        """Test governance.gate_checker module imports."""
        from nsc.governance.gate_checker import GateChecker
        assert GateChecker is not None

    def test_import_governance_governance(self):
        """Test governance.governance module imports."""
        from nsc.governance.governance import GovernanceEngine
        assert GovernanceEngine is not None

    def test_import_governance_replay(self):
        """Test governance.replay module imports."""
        from nsc.governance.replay import ReplayVerifier
        assert ReplayVerifier is not None

    def test_import_glyphs(self):
        """Test glyphs module imports."""
        from nsc.glyphs import GLLLCodebook, REJECTHandler, HadamardMatrix
        assert GLLLCodebook is not None
        assert REJECTHandler is not None
        assert HadamardMatrix is not None

    def test_import_npe_v0_1(self):
        """Test NPE v0.1 imports."""
        from nsc.npe_v0_1 import NoeticanProposalEngine
        assert NoeticanProposalEngine is not None

    def test_import_npe_v1_0(self):
        """Test NPE v1.0 imports."""
        from nsc.npe_v1_0 import NoeticanProposalEngine_v1_0
        assert NoeticanProposalEngine_v1_0 is not None

    def test_import_npe_v2_0(self):
        """Test NPE v2.0 imports."""
        try:
            from nsc.npe_v2_0 import NoeticanProposalEngine_v2_0
            assert NoeticanProposalEngine_v2_0 is not None
        except ImportError:
            pytest.skip("NPE v2.0 not available")


class TestBasicInstantiation:
    """Test basic module instantiation."""

    def test_instantiate_environment(self, sample_registry):
        """Test Environment instantiation."""
        from nsc.runtime.environment import Environment
        env = Environment(registry=sample_registry)
        assert env is not None

    def test_instantiate_runtime_state(self):
        """Test RuntimeState instantiation."""
        from nsc.runtime.state import RuntimeState
        state = RuntimeState()
        assert state is not None

    def test_instantiate_hadamard_matrix(self):
        """Test HadamardMatrix creation."""
        from nsc.glyphs import HadamardMatrix
        h = HadamardMatrix.create(4)
        assert h is not None
        assert h.shape == (4, 4)

    def test_instantiate_glll_codebook(self):
        """Test GLLLCodebook instantiation."""
        from nsc.glyphs import GLLLCodebook
        codebook = GLLLCodebook(n=16)
        assert codebook is not None
        assert codebook.n == 16

    def test_instantiate_reject_handler(self):
        """Test REJECTHandler instantiation."""
        from nsc.glyphs import REJECTHandler
        handler = REJECTHandler()
        assert handler is not None


class TestModuleStructure:
    """Test module structure and attributes."""

    def test_hadamard_orthogonality(self):
        """Test Hadamard matrix orthogonality."""
        from nsc.glyphs import HadamardMatrix
        import math
        import numpy as np
        
        h = HadamardMatrix.create(4)
        n = h.shape[0]
        
        # Check orthogonality
        for i in range(n):
            for j in range(n):
                dot = sum(h[k, i] * h[k, j] for k in range(n))
                if i == j:
                    assert math.isclose(dot, n)
                else:
                    assert math.isclose(dot, 0.0)

    def test_environment_has_registry(self, sample_registry):
        """Test Environment has registry attribute."""
        from nsc.runtime.environment import Environment
        env = Environment(registry=sample_registry)
        assert hasattr(env, 'registry')

    def test_runtime_state_has_fields(self):
        """Test RuntimeState has fields attribute."""
        from nsc.runtime.state import RuntimeState
        state = RuntimeState()
        assert hasattr(state, 'fields')

    def test_runtime_state_has_node_results(self):
        """Test RuntimeState has node_results attribute."""
        from nsc.runtime.state import RuntimeState
        state = RuntimeState()
        assert hasattr(state, 'node_results')
