"""Test runtime engine (executor, evaluator, interpreter, state, environment)."""
import pytest
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.runtime.state import RuntimeState, FieldHandle, CoherenceMetrics
from nsc.runtime.environment import Environment, KernelContext
from nsc.runtime.evaluator import NodeEvaluator
from nsc.runtime.executor import Executor
from nsc.runtime.interpreter import Interpreter, ExecutionResult


class TestRuntimeState:
    """Tests for RuntimeState."""

    def test_empty_state_creation(self, empty_runtime_state: RuntimeState):
        """Test creating an empty runtime state."""
        assert empty_runtime_state.fields == {}
        assert empty_runtime_state.node_results == {}
        assert empty_runtime_state.metrics is None
        assert empty_runtime_state.execution_history == []

    def test_state_with_fields(self, initialized_runtime_state: RuntimeState):
        """Test runtime state with field handles."""
        assert len(initialized_runtime_state.fields) == 3
        assert 1 in initialized_runtime_state.fields
        assert 2 in initialized_runtime_state.fields
        assert 3 in initialized_runtime_state.fields

    def test_state_set_node_result(self, empty_runtime_state: RuntimeState):
        """Test setting node evaluation results."""
        empty_runtime_state.node_results[1] = 1.0
        empty_runtime_state.node_results[2] = 2.0
        assert empty_runtime_state.node_results[1] == 1.0
        assert empty_runtime_state.node_results[2] == 2.0

    def test_state_add_execution_history(self, empty_runtime_state: RuntimeState):
        """Test adding execution history entries."""
        entry = {"node_id": 1, "duration_ms": 0.5}
        empty_runtime_state.execution_history.append(entry)
        assert len(empty_runtime_state.execution_history) == 1
        assert empty_runtime_state.execution_history[0]["node_id"] == 1

    def test_coherence_metrics_creation(self):
        """Test creating CoherenceMetrics."""
        metrics = CoherenceMetrics(
            phys=0.001,
            cons=0.002,
            num=0.0005,
            debt=1.5,
            r_phys=0.99,
            r_cons=0.98,
            r_num=0.995
        )
        assert metrics.phys == 0.001
        assert metrics.cons == 0.002
        assert metrics.num == 0.0005
        assert metrics.debt == 1.5


class TestFieldHandle:
    """Tests for FieldHandle."""

    def test_field_handle_creation(self):
        """Test creating a FieldHandle."""
        handle = FieldHandle(field_id=1, dtype="float32")
        assert handle.field_id == 1
        assert handle.dtype == "float32"


class TestEnvironment:
    """Tests for Environment."""

    def test_environment_creation(self, sample_registry: Dict[str, Any]):
        """Test creating an Environment."""
        env = Environment(registry=sample_registry)
        assert env.registry["registry_id"] == "test_registry_v1"
        assert len(env.registry["operators"]) == 4

    def test_environment_get_operator(self, sample_registry: Dict[str, Any]):
        """Test getting an operator from environment."""
        env = Environment(registry=sample_registry)
        op = env.get_operator(1001)
        assert op is not None
        assert op["name"] == "ADD"

    def test_environment_get_nonexistent_operator(self, sample_registry: Dict[str, Any]):
        """Test getting a nonexistent operator returns None."""
        env = Environment(registry=sample_registry)
        op = env.get_operator(9999)
        assert op is None


class TestNodeEvaluator:
    """Tests for NodeEvaluator."""

    def test_evaluator_creation(self, sample_registry: Dict[str, Any]):
        """Test creating a NodeEvaluator."""
        env = Environment(registry=sample_registry)
        evaluator = NodeEvaluator(env=env)
        assert evaluator.env is env

    def test_eval_constant(self, sample_registry: Dict[str, Any], empty_runtime_state: RuntimeState):
        """Test evaluating a constant node."""
        env = Environment(registry=sample_registry)
        evaluator = NodeEvaluator(env=env)
        
        node = {
            "id": 1,
            "type_tag": {"tag": "float"},
            "payload": {"kind": "CONST", "value": 42.0}
        }
        
        result = evaluator._eval_const(node, empty_runtime_state)
        assert result == 42.0


class TestExecutor:
    """Tests for Executor."""

    def test_executor_creation(self, sample_registry: Dict[str, Any]):
        """Test creating an Executor."""
        env = Environment(registry=sample_registry)
        executor = Executor(env=env)
        assert executor.env is env

    def test_execute_simple_module(self, sample_module: Dict[str, Any], sample_registry: Dict[str, Any]):
        """Test executing a simple module."""
        env = Environment(registry=sample_registry)
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
        }
        
        executor = Executor(env=env)
        result = executor.execute(sample_module)
        
        assert result is not None


class TestInterpreter:
    """Tests for Interpreter."""

    def test_interpreter_creation(self, sample_registry: Dict[str, Any]):
        """Test creating an Interpreter."""
        env = Environment(registry=sample_registry)
        interpreter = Interpreter(env=env)
        assert interpreter.env is env

    def test_interpret_simple_module(self, sample_module: Dict[str, Any], sample_registry: Dict[str, Any]):
        """Test interpreting a simple module."""
        env = Environment(registry=sample_registry)
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
        }
        
        interpreter = Interpreter(env=env)
        result = interpreter.interpret(sample_module)
        
        assert result is not None

    def test_interpret_empty_module(self, sample_registry: Dict[str, Any]):
        """Test interpreting an empty module."""
        env = Environment(registry=sample_registry)
        interpreter = Interpreter(env=env)
        
        empty_module = {
            "module_id": "empty",
            "version": "1.1.0",
            "nodes": [],
            "seq": [],
            "entrypoints": [],
            "registry_ref": "test_registry_v1"
        }
        
        result = interpreter.interpret(empty_module)
        assert result is not None


class TestExecutionResult:
    """Tests for ExecutionResult."""

    def test_execution_result_creation(self):
        """Test creating an ExecutionResult."""
        result = ExecutionResult(
            success=True,
            node_count=5,
            outputs={1: 1.0, 2: 2.0},
            receipts=[],
            braid_events=[],
            duration_ms=10.5
        )
        assert result.success is True
        assert result.node_count == 5
        assert result.duration_ms == 10.5

    def test_execution_result_with_receipts(self):
        """Test ExecutionResult with receipts."""
        receipts = [
            {"node_id": 1, "result": 1.0},
            {"node_id": 2, "result": 2.0},
        ]
        result = ExecutionResult(
            success=True,
            node_count=2,
            outputs={},
            receipts=receipts,
            braid_events=[],
            duration_ms=5.0
        )
        assert len(result.receipts) == 2
