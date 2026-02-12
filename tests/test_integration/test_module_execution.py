"""Integration tests for module execution."""

import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from nsc.runtime.interpreter import Interpreter
from nsc.runtime.environment import Environment


class TestModuleExecution:
    """Integration tests for module execution."""
    
    @pytest.fixture
    def sample_module(self):
        """Sample minimal NSC module for testing."""
        return {
            "module_id": "test_module_001",
            "version": "1.1.0",
            "nodes": [
                {
                    "id": 1,
                    "kind": "CONST",
                    "type_tag": {"tag": "float"},
                    "payload": {"value": 1.0}
                },
                {
                    "id": 2,
                    "kind": "CONST",
                    "type_tag": {"tag": "float"},
                    "payload": {"value": 2.0}
                },
                {
                    "id": 3,
                    "kind": "APPLY",
                    "type_tag": {"tag": "float"},
                    "payload": {"op_id": 1001, "inputs": [1, 2]}
                }
            ],
            "seq": [1, 2, 3],
            "entrypoints": [3],
            "registry_ref": "test_registry"
        }
    
    @pytest.fixture
    def default_registry(self):
        """Default operator registry."""
        return {
            "registry_id": "test_registry",
            "version": "1.0",
            "operators": [
                {"op_id": 1001, "name": "ADD", "arg_types": ["float", "float"], "ret_type": "float"},
                {"op_id": 1002, "name": "SUB", "arg_types": ["float", "float"], "ret_type": "float"},
                {"op_id": 1003, "name": "MUL", "arg_types": ["float", "float"], "ret_type": "float"},
                {"op_id": 1004, "name": "DIV", "arg_types": ["float", "float"], "ret_type": "float"},
            ]
        }
    
    def test_execute_simple_module(self, sample_module, default_registry):
        """Test executing a minimal module."""
        env = Environment(registry=default_registry)
        
        # Setup kernels
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
            "kernel_sub": lambda a, b: a - b,
            "kernel_mul": lambda a, b: a * b,
            "kernel_div": lambda a, b: a / b if b != 0 else float('inf'),
        }
        
        interpreter = Interpreter(env=env)
        result = interpreter.interpret(sample_module)
        
        # Verify execution succeeded
        assert result.success is True
        assert result.error is None
    
    def test_execute_from_file(self, default_registry):
        """Test executing a module from file."""
        module_path = Path(__file__).parent.parent / "fixtures" / "modules" / "minimal.json"
        
        if not module_path.exists():
            pytest.skip("Test fixture not found")
        
        with open(module_path) as f:
            module = json.load(f)
        
        env = Environment(registry=default_registry)
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
        }
        
        interpreter = Interpreter(env=env)
        result = interpreter.interpret(module)
        
        assert result.success is True
    
    def test_execution_with_missing_node_reference(self, default_registry):
        """Test execution fails gracefully with invalid node reference."""
        module = {
            "module_id": "test_bad_ref",
            "version": "1.1.0",
            "nodes": [
                {
                    "id": 1,
                    "kind": "CONST",
                    "type_tag": {"tag": "float"},
                    "payload": {"value": 1.0}
                },
                {
                    "id": 2,
                    "kind": "APPLY",
                    "type_tag": {"tag": "float"},
                    "payload": {"op_id": 1001, "inputs": [1, 999]}  # Invalid reference
                }
            ],
            "seq": [1, 2],
            "entrypoints": [2],
            "registry_ref": "test_registry"
        }
        
        env = Environment(registry=default_registry)
        env.kernels = {"kernel_add": lambda a, b: a + b}
        
        interpreter = Interpreter(env=env)
        result = interpreter.interpret(module)
        
        # Should fail due to invalid reference
        assert result.success is False
        assert result.error is not None
