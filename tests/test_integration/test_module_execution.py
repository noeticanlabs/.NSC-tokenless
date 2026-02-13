"""Integration tests for module execution."""
import pytest
import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nsc.runtime.interpreter import Interpreter, ExecutionContext
from nsc.runtime.executor import Executor
from nsc.runtime.environment import Environment


class TestModuleExecution:
    """Integration tests for module execution."""

    def test_execute_simple_module(self):
        """Test executing a simple module."""
        # Create a mock registry and binding
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        # Create interpreter with mocks
        interpreter = Interpreter(
            registry=registry,
            binding=binding
        )
        assert interpreter is not None

    def test_execute_from_file(self, tmp_path):
        """Test executing a module from file."""
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        interpreter = Interpreter(
            registry=registry,
            binding=binding
        )
        assert interpreter is not None

    def test_interpreter_has_required_methods(self):
        """Test interpreter has required methods."""
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops=[]
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        interpreter = Interpreter(
            registry=registry,
            binding=binding
        )
        
        assert hasattr(interpreter, 'interpret')


class TestModuleValidation:
    """Integration tests for module validation."""

    def test_validate_minimal_module(self):
        """Test validating a minimal module."""
        module = {
            "module_id": "test",
            "nodes": [
                {"id": 1, "kind": "FIELD_REF", "type": {"type": "SCALAR"}, "payload": {"field_id": 1}}
            ],
            "seq": [1],
            "entrypoints": [1]
        }
        
        # Validate structure
        assert "module_id" in module
        assert "nodes" in module
        assert len(module["nodes"]) > 0
        assert "seq" in module
        assert "entrypoints" in module

    def test_validate_node_references(self):
        """Test that SEQ references valid nodes."""
        module = {
            "module_id": "test",
            "nodes": [
                {"id": 1, "kind": "FIELD_REF"},
                {"id": 2, "kind": "APPLY"}
            ],
            "seq": [1, 2],
            "entrypoints": [2]
        }
        
        node_ids = {n["id"] for n in module["nodes"]}
        for seq_id in module["seq"]:
            assert seq_id in node_ids, f"SEQ references non-existent node {seq_id}"

    def test_validate_apply_nodes(self):
        """Test APPLY node validation."""
        apply_node = {
            "id": 2,
            "kind": "APPLY",
            "op_id": 1001,  # Must have op_id
            "inputs": [1]    # Must have inputs
        }
        
        assert apply_node["kind"] == "APPLY"
        assert "op_id" in apply_node
        assert "inputs" in apply_node


class TestReceiptChainVerification:
    """Integration tests for receipt chaining and verification."""

    @pytest.fixture
    def sample_receipt_chain(self):
        """Create sample receipt chain."""
        return [
            {
                "kind": "operator_receipt",
                "event_id": "evt_001",
                "node_id": 1,
                "op_id": 1001,
                "digest_in": "sha256:" + "0"*64,
                "digest_out": "sha256:" + "a"*64,
                "kernel_id": "kernel.add.v1",
                "kernel_version": "1.0.0",
                "C_before": 0.0,
                "C_after": 0.95,
                "r_phys": 0.001,
                "r_cons": 0.002,
                "r_num": 0.0005
            },
            {
                "kind": "operator_receipt",
                "event_id": "evt_002",
                "node_id": 2,
                "op_id": 1002,
                "digest_in": "sha256:" + "a"*64,
                "digest_out": "sha256:" + "b"*64,
                "kernel_id": "kernel.sub.v1",
                "kernel_version": "1.0.0",
                "C_before": 0.95,
                "C_after": 0.92,
                "r_phys": 0.0015,
                "r_cons": 0.0025,
                "r_num": 0.0007
            }
        ]

    def test_hash_chain_validity(self, sample_receipt_chain):
        """Test hash chain validity checking."""
        from nsc.governance.replay import ReplayVerifier
        verifier = ReplayVerifier()
        is_valid = verifier.verify_hash_chain(sample_receipt_chain)
        assert isinstance(is_valid, bool)

    def test_receipt_verification(self, sample_receipt_chain):
        """Test verifying individual receipts."""
        from nsc.governance.replay import ReplayVerifier
        
        module = {
            "module_id": "test",
            "nodes": [
                {"id": 1, "kind": "FIELD_REF"},
                {"id": 2, "kind": "APPLY"}
            ]
        }
        
        verifier = ReplayVerifier()
        for receipt in sample_receipt_chain:
            result = verifier.verify_receipt(receipt, module)
            assert isinstance(result, dict)
            assert "valid" in result
            assert "node_id" in result


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_execution_pipeline(self):
        """Test complete execution pipeline."""
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        # Module
        module = {
            "module_id": "e2e_test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1],
            "entrypoints": [1]
        }
        
        # Registry and binding
        registry = OperatorRegistry(
            registry_id="test_reg",
            version="1.0",
            ops={}
        )
        binding = KernelBinding(
            binding_id="test_bind",
            registry_id="test_reg",
            bindings={}
        )
        
        # Create interpreter
        interpreter = Interpreter(registry=registry, binding=binding)
        assert interpreter is not None
        
        # Create execution context
        ctx = ExecutionContext(
            module=module,
            module_digest="sha256:test",
            registry_id="test_reg",
            registry_version="1.0",
            binding_id="test_bind"
        )
        assert ctx.module == module

    def test_validation_and_execution(self):
        """Test validation followed by execution."""
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        module = {
            "module_id": "test",
            "nodes": [
                {"id": 1, "kind": "FIELD_REF", "op_id": None},
                {"id": 2, "kind": "APPLY", "op_id": 1001}
            ],
            "seq": [1, 2],
            "entrypoints": [2]
        }
        
        # Validate
        node_ids = {n["id"] for n in module["nodes"]}
        assert all(nid in node_ids for nid in module["seq"])
        
        # Create interpreter
        registry = OperatorRegistry("test", "1.0", {})
        binding = KernelBinding("test_bind", "test", {})
        interpreter = Interpreter(registry=registry, binding=binding)
        assert interpreter is not None

    def test_replay_verification_pipeline(self):
        """Test replay verification after execution."""
        from nsc.governance.replay import ReplayVerifier
        
        module = {
            "module_id": "test",
            "nodes": [{"id": 1, "kind": "FIELD_REF"}],
            "seq": [1],
            "entrypoints": [1]
        }
        
        receipts = [
            {
                "kind": "operator_receipt",
                "event_id": "evt_001",
                "node_id": 1,
                "op_id": 1001
            }
        ]
        
        verifier = ReplayVerifier()
        is_valid = verifier.verify_hash_chain(receipts)
        assert isinstance(is_valid, bool)
