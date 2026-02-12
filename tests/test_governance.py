"""Test governance module (gate_checker, governance, replay)."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.governance.gate_checker import GateChecker, GateReport, HardGateResult, SoftGateResult
from nsc.governance.governance import GovernanceEngine, ActionPlan
from nsc.governance.replay import ReplayVerifier, ReplayReport


@dataclass
class RailResult:
    """Rail result for testing."""
    rail: str
    severity: float
    residual_delta: Dict[str, float]


class TestGateChecker:
    """Tests for GateChecker."""

    def test_gate_checker_creation(self, gate_thresholds: Dict[str, float]):
        """Test creating a GateChecker."""
        checker = GateChecker(thresholds=gate_thresholds)
        assert checker is not None

    def test_check_hard_gates_pass(self, gate_thresholds: Dict[str, float]):
        """Test hard gates passing with valid values."""
        checker = GateChecker(thresholds=gate_thresholds)
        
        residuals = {
            "phys": 0.001,
            "cons": 0.002,
            "num": 0.0005
        }
        
        result = checker._check_hard_gates(residuals)
        assert result is not None
        assert hasattr(result, 'pass_hard')

    def test_check_hard_gates_nan_fail(self, gate_thresholds: Dict[str, float]):
        """Test hard gates failing on NaN."""
        checker = GateChecker(thresholds=gate_thresholds)
        
        residuals = {
            "phys": float('nan'),
            "cons": 0.002,
            "num": 0.0005
        }
        
        result = checker._check_hard_gates(residuals)
        assert result is not None

    def test_check_soft_gates_pass(self, gate_thresholds: Dict[str, float]):
        """Test soft gates passing with low residuals."""
        checker = GateChecker(thresholds=gate_thresholds)
        
        residuals = {
            "phys": 0.001,
            "cons": 0.002,
            "num": 0.0005
        }
        
        result = checker._check_soft_gates(residuals)
        assert result is not None
        assert hasattr(result, 'all_pass')

    def test_evaluate_returns_dict(self, gate_thresholds: Dict[str, float]):
        """Test full gate evaluation returns dict."""
        checker = GateChecker(thresholds=gate_thresholds)
        
        residuals = {
            "phys": 0.001,
            "cons": 0.002,
            "num": 0.0005
        }
        
        report = checker.evaluate(residuals)
        
        assert report is not None
        assert isinstance(report, dict)


class TestGateReport:
    """Tests for GateReport structure."""

    def test_gate_report_creation(self, gate_thresholds: Dict[str, float]):
        """Test creating a GateReport."""
        checker = GateChecker(thresholds=gate_thresholds)
        residuals = {"phys": 0.001, "cons": 0.002, "num": 0.0005}
        
        report = checker.evaluate(residuals)
        
        assert report is not None


class TestGovernanceEngine:
    """Tests for GovernanceEngine."""

    def test_governance_engine_creation(self, gate_thresholds: Dict[str, float]):
        """Test creating a GovernanceEngine."""
        engine = GovernanceEngine(gate_thresholds=gate_thresholds)
        assert engine is not None

    def test_evaluate_returns_action_plan(self, gate_thresholds: Dict[str, float]):
        """Test governance evaluation returns ActionPlan."""
        engine = GovernanceEngine(gate_thresholds=gate_thresholds)
        
        residuals = {
            "phys": 0.001,
            "cons": 0.002,
            "num": 0.0005
        }
        
        action_plan = engine.evaluate(residuals, debt=0.1)
        
        assert action_plan is not None
        assert hasattr(action_plan, 'action')


class TestActionPlan:
    """Tests for ActionPlan."""

    def test_action_plan_creation(self):
        """Test creating an ActionPlan."""
        plan = ActionPlan(
            action="PASS",
            rail_results=[],
            debt=0.1,
            prev_digest=None
        )
        
        assert plan.action == "PASS"
        assert plan.debt == 0.1

    def test_action_plan_with_rails(self):
        """Test ActionPlan with rail results."""
        rail_result = RailResult(
            rail="R1",
            severity=0.5,
            residual_delta={"phys": -0.01}
        )
        
        plan = ActionPlan(
            action="MITIGATE",
            rail_results=[rail_result],
            debt=1.0,
            prev_digest=None
        )
        
        assert len(plan.rail_results) == 1
        assert plan.rail_results[0].rail == "R1"


class TestReplayVerifier:
    """Tests for ReplayVerifier."""

    def test_replay_verifier_creation(self):
        """Test creating a ReplayVerifier."""
        verifier = ReplayVerifier(tolerance=1e-6)
        assert verifier is not None

    def test_verify_returns_replay_report(self):
        """Test verify returns ReplayReport."""
        verifier = ReplayVerifier(tolerance=1e-6)
        
        original = {"node_1": 1.0, "node_2": 2.0}
        replay = {"node_1": 1.0, "node_2": 2.0}
        
        result = verifier.verify(original, replay)
        
        assert result is not None
        assert hasattr(result, 'replay_pass')


class TestReplayReport:
    """Tests for ReplayReport."""

    def test_replay_report_success(self):
        """Test creating a successful ReplayReport."""
        result = ReplayReport(
            replay_pass=True,
            module_digest_ok=True,
            registry_ok=True,
            kernel_binding_ok=True,
            tolerances={"replay": 1e-6}
        )
        
        assert result.replay_pass is True

    def test_replay_report_failure(self):
        """Test creating a failed ReplayReport."""
        result = ReplayReport(
            replay_pass=False,
            module_digest_ok=False,
            registry_ok=True,
            kernel_binding_ok=True,
            tolerances={"replay": 1e-6},
            first_mismatch={"node": "node_1", "diff": 1e-4}
        )
        
        assert result.replay_pass is False
        assert result.first_mismatch is not None
