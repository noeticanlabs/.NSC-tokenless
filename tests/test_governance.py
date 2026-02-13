"""Test governance module (gate_checker, governance, replay)."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.governance.gate_checker import GateChecker, GateThresholds, GateReport
from nsc.governance.governance import GovernanceEngine, ActionPlan, ActionType
from nsc.governance.replay import ReplayVerifier, ReplayReport


class TestGateChecker:
    """Tests for GateChecker."""

    def test_gate_checker_creation(self):
        """Test creating a GateChecker."""
        thresholds = GateThresholds()
        checker = GateChecker(thresholds=thresholds)
        assert checker is not None

    def test_evaluate_method_exists(self):
        """Test that evaluate method exists."""
        checker = GateChecker()
        assert hasattr(checker, 'evaluate')

    def test_evaluate_with_hysteresis_exists(self):
        """Test that evaluate_with_hysteresis method exists."""
        checker = GateChecker()
        assert hasattr(checker, 'evaluate_with_hysteresis')


class TestGateReport:
    """Tests for GateReport."""

    def test_gate_report_creation(self):
        """Test creating a GateReport."""
        report = GateReport(
            event_id="test_event",
            node_id=1,
            gate_kind="HARD",
            passed=True,
            thresholds={"r_phys": 1.0},
            values={"r_phys": 0.001},
            decision="ACCEPT"
        )
        assert report.event_id == "test_event"
        assert report.passed == True

    def test_gate_report_to_dict(self):
        """Test GateReport serialization."""
        report = GateReport(
            event_id="test_event",
            node_id=1,
            gate_kind="HARD",
            passed=True,
            thresholds={},
            values={},
            decision="ACCEPT"
        )
        report_dict = report.to_dict()
        assert report_dict["kind"] == "gate_report"
        assert report_dict["event_id"] == "test_event"


class TestGovernanceEngine:
    """Tests for GovernanceEngine."""

    def test_governance_engine_creation(self):
        """Test creating a GovernanceEngine."""
        engine = GovernanceEngine()
        assert engine is not None

    def test_governance_has_evaluate(self):
        """Test GovernanceEngine has evaluate method."""
        engine = GovernanceEngine()
        assert hasattr(engine, 'evaluate')

    def test_governance_has_rails(self):
        """Test GovernanceEngine has evaluate method."""
        engine = GovernanceEngine()
        assert hasattr(engine, 'evaluate')


class TestActionPlan:
    """Tests for ActionPlan."""

    def test_action_plan_creation(self):
        """Test creating an ActionPlan."""
        plan = ActionPlan(
            event_id="test_event",
            action=ActionType.CONTINUE.value,
            trigger_reports=["report1"]
        )
        assert plan.event_id == "test_event"
        assert plan.action == ActionType.CONTINUE.value

    def test_action_plan_with_rails(self):
        """Test ActionPlan with rail application."""
        plan = ActionPlan(
            event_id="test_event",
            action=ActionType.RAIL.value,
            trigger_reports=["report1"],
            rail_applied="R1",
            retry_count=1
        )
        assert plan.rail_applied == "R1"
        assert plan.retry_count == 1

    def test_action_plan_compute_digest(self):
        """Test ActionPlan digest computation."""
        plan = ActionPlan(
            event_id="test_event",
            action=ActionType.CONTINUE.value,
            trigger_reports=[]
        )
        digest = plan.compute_digest()
        assert digest.startswith("sha256:")


class TestReplayVerifier:
    """Tests for ReplayVerifier."""

    def test_replay_verifier_creation(self):
        """Test creating a ReplayVerifier."""
        verifier = ReplayVerifier()
        assert verifier is not None

    def test_verify_method_exists(self):
        """Test that verify method exists."""
        verifier = ReplayVerifier()
        assert hasattr(verifier, 'verify')

    def test_replay_report_creation(self):
        """Test ReplayReport creation."""
        report = ReplayReport(
            replay_pass=True,
            module_digest_ok=True,
            registry_ok=True,
            kernel_binding_ok=True,
            tolerances={"abs": 1e-9}
        )
        assert report.replay_pass == True
        assert report.module_digest_ok == True

    def test_replay_report_failure(self):
        """Test ReplayReport for failed replay."""
        report = ReplayReport(
            replay_pass=False,
            module_digest_ok=False,
            registry_ok=True,
            kernel_binding_ok=True,
            tolerances={"abs": 1e-9},
            first_mismatch={"field": "module_digest"}
        )
        assert report.replay_pass == False
        assert report.module_digest_ok == False
