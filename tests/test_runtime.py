"""Test runtime engine (executor, evaluator, interpreter, state, environment)."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.runtime.state import RuntimeState, FieldHandle, CoherenceMetrics, ValueType
from nsc.runtime.environment import Environment, FieldRegistry
from nsc.runtime.evaluator import NodeEvaluator, EvaluatorResult
from nsc.runtime.executor import Executor, ExecutionConfig, ExecutionResult, OperatorReceipt
from nsc.runtime.interpreter import Interpreter, ExecutionContext


class TestRuntimeState:
    """Tests for RuntimeState."""

    def test_empty_state_creation(self):
        """Test creating an empty runtime state."""
        state = RuntimeState()
        assert state.fields == {}
        assert state.node_results == {}
        assert state.execution_history == []

    def test_state_with_fields(self):
        """Test runtime state with field handles."""
        state = RuntimeState()
        state.register_field(1, ValueType.SCALAR, 1.0)
        state.register_field(2, ValueType.SCALAR, 2.0)
        state.register_field(3, ValueType.SCALAR, 3.0)
        assert 1 in state.fields
        assert 2 in state.fields
        assert 3 in state.fields

    def test_state_set_node_result(self):
        """Test setting node evaluation results."""
        state = RuntimeState()
        state.node_results[1] = 1.0
        state.node_results[2] = 2.0
        assert state.node_results[1] == 1.0
        assert state.node_results[2] == 2.0

    def test_state_add_execution_history(self):
        """Test adding execution history entries."""
        state = RuntimeState()
        entry = {"node_id": 1, "duration_ms": 0.5}
        state.execution_history.append(entry)
        assert len(state.execution_history) == 1
        assert state.execution_history[0]["node_id"] == 1

    def test_coherence_metrics_creation(self):
        """Test creating CoherenceMetrics."""
        metrics = CoherenceMetrics(
            r_phys=0.001,
            r_cons=0.002,
            r_num=0.0005
        )
        assert metrics.r_phys == 0.001
        assert metrics.r_cons == 0.002
        assert metrics.r_num == 0.0005

    def test_coherence_metrics_reset(self):
        """Test resetting coherence metrics."""
        metrics = CoherenceMetrics(r_phys=0.5, r_cons=0.6, r_num=0.7)
        metrics.reset()
        assert metrics.r_phys == 0.0
        assert metrics.r_cons == 0.0
        assert metrics.r_num == 0.0


class TestFieldHandle:
    """Tests for FieldHandle."""

    def test_field_handle_creation(self):
        """Test creating a FieldHandle."""
        handle = FieldHandle(field_id=1, value_type=ValueType.SCALAR, value=1.0)
        assert handle.field_id == 1
        assert handle.value_type == ValueType.SCALAR
        assert handle.value == 1.0

    def test_field_handle_is_scalar(self):
        """Test is_scalar method."""
        scalar_handle = FieldHandle(1, ValueType.SCALAR, 1.0)
        phase_handle = FieldHandle(2, ValueType.PHASE, 1.0+2j)
        assert scalar_handle.is_scalar() == True
        assert phase_handle.is_scalar() == False


class TestEnvironment:
    """Tests for Environment."""

    def test_environment_creation(self):
        """Test creating an Environment."""
        env = Environment()
        assert env.registry is not None
        assert env.state is not None

    def test_environment_create_scalar(self):
        """Test creating a scalar field."""
        env = Environment()
        handle = env.create_scalar(1, 1.0, "test_scalar")
        assert handle.field_id == 1
        assert handle.value_type == ValueType.SCALAR

    def test_environment_get_field(self):
        """Test getting a field by ID."""
        env = Environment()
        env.create_scalar(1, 1.0, "test")
        handle = env.get(1)
        assert handle is not None
        assert handle.field_id == 1

    def test_environment_set_field(self):
        """Test setting field value."""
        env = Environment()
        env.create_scalar(1, 1.0, "test")
        env.set(1, 2.0)
        handle = env.get(1)
        assert handle.value == 2.0


class TestNodeEvaluator:
    """Tests for NodeEvaluator."""

    def test_evaluator_creation(self):
        """Test creating a NodeEvaluator."""
        env = Environment()
        state = RuntimeState()
        evaluator = NodeEvaluator(
            registry=MagicMock(),
            binding=MagicMock(),
            environment=env,
            state=state
        )
        assert evaluator is not None

    def test_evaluate_field_ref_node(self):
        """Test evaluating a FIELD_REF node."""
        env = Environment()
        env.create_scalar(1, 42.0, "test")
        state = RuntimeState()
        
        evaluator = NodeEvaluator(
            registry=MagicMock(),
            binding=MagicMock(),
            environment=env,
            state=state
        )
        
        node = {
            "id": 10,
            "kind": "FIELD_REF",
            "payload": {"field_id": 1}
        }
        result = evaluator.evaluate(node)
        assert result.success == True
        assert result.value == 42.0

    def test_evaluate_unknown_node_kind(self):
        """Test evaluating an unknown node kind."""
        env = Environment()
        state = RuntimeState()
        
        evaluator = NodeEvaluator(
            registry=MagicMock(),
            binding=MagicMock(),
            environment=env,
            state=state
        )
        
        node = {
            "id": 10,
            "kind": "UNKNOWN",
            "payload": {}
        }
        result = evaluator.evaluate(node)
        assert result.success == False
        assert "Unknown node kind" in result.error


class TestExecutor:
    """Tests for Executor."""

    def test_executor_creation(self):
        """Test creating an Executor."""
        executor = Executor(
            registry=MagicMock(),
            binding=MagicMock()
        )
        assert executor is not None

    def test_execution_config_defaults(self):
        """Test ExecutionConfig default values."""
        config = ExecutionConfig()
        assert config.validate_modules == True
        assert config.emit_receipts == True
        assert config.strict_mode == False
        assert config.max_steps == 1000

    def test_operator_receipt_to_dict(self):
        """Test OperatorReceipt serialization."""
        receipt = OperatorReceipt(
            event_id="test_event",
            node_id=1,
            op_id=1001,
            kernel_id="kernel_add",
            kernel_version="1.0",
            digest_in="sha256:abc123",
            digest_out="sha256:def456",
            C_before=0.0,
            C_after=1.0,
            r_phys=0.001,
            r_cons=0.002,
            r_num=0.0005
        )
        receipt_dict = receipt.to_dict()
        assert receipt_dict["kind"] == "operator_receipt"
        assert receipt_dict["node_id"] == 1


class TestInterpreter:
    """Tests for Interpreter."""

    def test_interpreter_creation(self):
        """Test creating an Interpreter."""
        interpreter = Interpreter(
            registry=MagicMock(),
            binding=MagicMock()
        )
        assert interpreter is not None

    def test_execution_context_creation(self):
        """Test creating an ExecutionContext."""
        context = ExecutionContext(
            module={"module_id": "test"},
            module_digest="sha256:abc123",
            registry_id="test_registry",
            registry_version="1.0",
            binding_id="test_binding"
        )
        assert context.module["module_id"] == "test"
        assert context.registry_id == "test_registry"


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_execution_result_creation(self):
        """Test creating an ExecutionResult."""
        result = ExecutionResult(
            success=True,
            module_id="test_module"
        )
        assert result.success == True
        assert result.module_id == "test_module"

    def test_execution_result_with_receipts(self):
        """Test ExecutionResult with receipts."""
        receipt = OperatorReceipt(
            event_id="evt1",
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
        result = ExecutionResult(
            success=True,
            module_id="test",
            receipts=[receipt]
        )
        assert len(result.receipts) == 1
