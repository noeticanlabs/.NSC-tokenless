"""Test fixtures for NSC Tokenless v1.1 test suite."""
import pytest
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.runtime.state import RuntimeState, FieldHandle
from nsc.runtime.environment import Environment
from nsc.glyphs import GLLLCodebook, REJECTHandler


@pytest.fixture
def sample_module() -> Dict[str, Any]:
    """Sample minimal NSC module for testing."""
    return {
        "module_id": "test_module_001",
        "version": "1.1.0",
        "nodes": [
            {
                "id": 1,
                "type_tag": {"tag": "float"},
                "payload": {"kind": "CONST", "value": 1.0}
            },
            {
                "id": 2,
                "type_tag": {"tag": "float"},
                "payload": {"kind": "CONST", "value": 2.0}
            },
            {
                "id": 3,
                "type_tag": {"tag": "float"},
                "payload": {"kind": "APPLY", "op_id": 1001, "inputs": [1, 2]}
            }
        ],
        "seq": [1, 2, 3],
        "entrypoints": [3],
        "registry_ref": "test_registry_v1"
    }


@pytest.fixture
def sample_registry() -> Dict[str, Any]:
    """Sample operator registry for testing."""
    return {
        "registry_id": "test_registry_v1",
        "operators": [
            {"op_id": 1001, "name": "ADD", "arg_types": ["float", "float"], "ret_type": "float"},
            {"op_id": 1002, "name": "SUB", "arg_types": ["float", "float"], "ret_type": "float"},
            {"op_id": 1003, "name": "MUL", "arg_types": ["float", "float"], "ret_type": "float"},
            {"op_id": 1004, "name": "DIV", "arg_types": ["float", "float"], "ret_type": "float"},
        ]
    }


@pytest.fixture
def sample_kernel_binding() -> Dict[str, Any]:
    """Sample kernel binding for testing."""
    return {
        "registry_id": "test_registry_v1",
        "bindings": [
            {"op_id": 1001, "kernel": "kernel_add", "version": "1.0"},
            {"op_id": 1002, "kernel": "kernel_sub", "version": "1.0"},
            {"op_id": 1003, "kernel": "kernel_mul", "version": "1.0"},
            {"op_id": 1004, "kernel": "kernel_div", "version": "1.0"},
        ]
    }


@pytest.fixture
def empty_runtime_state() -> RuntimeState:
    """Empty runtime state for testing."""
    return RuntimeState(
        fields={},
        node_results={},
        metrics=None,
        execution_history=[]
    )


@pytest.fixture
def initialized_runtime_state(sample_module: Dict[str, Any]) -> RuntimeState:
    """Initialized runtime state for testing."""
    state = RuntimeState(
        fields={
            1: FieldHandle(field_id=1, dtype="float32"),
            2: FieldHandle(field_id=2, dtype="float32"),
            3: FieldHandle(field_id=3, dtype="float32"),
        },
        node_results={},
        metrics=None,
        execution_history=[]
    )
    return state


@pytest.fixture
def glll_codebook_n16() -> GLLLCodebook:
    """GLLL codebook with n=16 for testing."""
    return GLLLCodebook(n=16)


@pytest.fixture
def reject_handler() -> REJECTHandler:
    """REJECT handler for testing."""
    return REJECTHandler()


@pytest.fixture
def coherence_residuals() -> Dict[str, Any]:
    """Sample coherence residuals for gate testing."""
    return {
        "phys": 0.001,
        "cons": 0.002,
        "num": 0.0005
    }


@pytest.fixture
def gate_thresholds() -> Dict[str, float]:
    """Sample gate thresholds."""
    return {
        "phys": 0.01,
        "cons": 0.02,
        "num": 0.01,
        "nan_check": True,
        "inf_check": True,
        "positivity_check": True
    }
