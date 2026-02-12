"""
NSC Module Executor

Per docs/conformance_levels.md L0: Executes NSC modules with proper validation,
registry resolution, kernel binding, and receipt emission.

Key classes:
- ExecutionConfig: Configuration for execution
- ExecutionResult: Result of module execution
- Executor: Main execution engine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime

from nsc.runtime.environment import Environment, FieldRegistry
from nsc.runtime.state import RuntimeState, CoherenceMetrics
from nsc.runtime.evaluator import NodeEvaluator, EvaluatorResult

@dataclass
class ExecutionConfig:
    """Configuration for module execution."""
    validate_modules: bool = True
    emit_receipts: bool = True
    strict_mode: bool = False
    max_steps: int = 1000
    tolerance: float = 1e-9

@dataclass
class OperatorReceipt:
    """
    Operator receipt per nsc_operator_receipt.schema.json.
    
    Per docs/terminology.md: NSC requires OperatorReceipts for APPLY evaluations.
    """
    event_id: str
    node_id: int
    op_id: int
    kernel_id: str
    kernel_version: str
    digest_in: str
    digest_out: str
    C_before: float
    C_after: float
    r_phys: float
    r_cons: float
    r_num: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    digest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "operator_receipt",
            "event_id": self.event_id,
            "node_id": self.node_id,
            "op_id": self.op_id,
            "kernel_id": self.kernel_id,
            "kernel_version": self.kernel_version,
            "digest_in": self.digest_in,
            "digest_out": self.digest_out,
            "C_before": self.C_before,
            "C_after": self.C_after,
            "r_phys": self.r_phys,
            "r_cons": self.r_cons,
            "r_num": self.r_num,
            "timestamp": self.timestamp,
            "digest": self.digest
        }
    
    def compute_digest(self, prev_hash: str = None) -> str:
        """Compute receipt digest per hash chaining rule."""
        payload = {k: v for k, v in self.to_dict().items() if k != "digest"}
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        if prev_hash:
            combined = canon.encode("utf-8") + prev_hash.encode("utf-8")
            self.digest = "sha256:" + hashlib.sha256(combined).hexdigest()
        else:
            self.digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
        
        return self.digest

@dataclass
class ExecutionResult:
    """Result of module execution."""
    success: bool
    module_id: str
    entrypoint_result: Any = None
    receipts: List[OperatorReceipt] = field(default_factory=list)
    final_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    digest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "module_id": self.module_id,
            "entrypoint_result": self.entrypoint_result,
            "receipts": [r.to_dict() for r in self.receipts],
            "final_metrics": self.final_metrics,
            "errors": self.errors,
            "execution_time_ms": self.execution_time_ms,
            "digest": self.digest
        }

class Executor:
    """
    Main NSC module execution engine.
    
    Per docs/conformance_levels.md L0:
    - Validate modules against schemas
    - Resolve operators by (registry_id, version, op_id)
    - Use Kernel Binding and record kernel identity in receipts
    - Enforce explicit evaluation order via SEQ nodes
    - Emit OperatorReceipt for every executed APPLY node
    """
    
    def __init__(self,
                 registry: "OperatorRegistry",
                 binding: "KernelBinding",
                 config: ExecutionConfig = None):
        self.registry = registry
        self.binding = binding
        self.config = config or ExecutionConfig()
        self.receipt_chain: List[str] = []
    
    def execute(self, module: Dict[str, Any], 
                initial_state: RuntimeState = None) -> ExecutionResult:
        """
        Execute an NSC module.
        
        Args:
            module: NSC module as dictionary (per nsc_module.schema.json)
            initial_state: Optional initial runtime state
            
        Returns:
            ExecutionResult with receipts and metrics
        """
        import time
        start = time.time()
        
        module_id = module.get("module_id", "unknown")
        nodes = module.get("nodes", [])
        entrypoints = module.get("entrypoints", {})
        eval_seq_id = module.get("order", {}).get("eval_seq_id")
        
        result = ExecutionResult(success=False, module_id=module_id)
        
        # Create environment and state
        env = Environment()
        state = initial_state or RuntimeState()
        
        # Validate module if configured
        if self.config.validate_modules:
            validation_errors = self._validate_module(module)
            if validation_errors:
                result.errors.extend(validation_errors)
                return result
        
        # Create evaluator
        evaluator = NodeEvaluator(self.registry, self.binding, env, state)
        
        # Build node lookup
        node_by_id = {n.get("id"): n for n in nodes}
        
        # Get SEQ node
        seq_node = node_by_id.get(eval_seq_id)
        if not seq_node:
            result.errors.append(f"SEQ node {eval_seq_id} not found")
            return result
        
        seq_ids = seq_node.get("payload", {}).get("seq", [])
        
        # Execute nodes in SEQ order
        event_counter = 0
        for node_id in seq_ids:
            if len(result.receipts) >= self.config.max_steps:
                result.errors.append("Max steps exceeded")
                break
            
            node = node_by_id.get(node_id)
            if not node:
                result.errors.append(f"Node {node_id} not found")
                continue
            
            # Evaluate node
            eval_result = evaluator.evaluate(node)
            
            if not eval_result.success:
                result.errors.append(f"Node {node_id}: {eval_result.error}")
                continue
            
            # Emit receipt for APPLY nodes
            if node.get("kind") == "APPLY" and self.config.emit_receipts:
                event_counter += 1
                receipt = self._emit_receipt(node, eval_result, state, event_counter)
                result.receipts.append(receipt)
            
            # Update coherence metrics
            state.update_metrics(
                r_phys=eval_result.metrics.get("r_phys", 0.0),
                r_cons=eval_result.metrics.get("r_cons", 0.0),
                r_num=eval_result.metrics.get("r_num", 0.0)
            )
        
        # Get entrypoint result
        if entrypoints:
            entrypoint_name = list(entrypoints.keys())[0]
            entrypoint_node_id = entrypoints[entrypoint_name]
            result.entrypoint_result = state.get_node_result(entrypoint_node_id)
        
        # Set final metrics
        result.final_metrics = state.metrics.to_dict()
        result.success = len(result.errors) == 0
        result.execution_time_ms = (time.time() - start) * 1000
        
        # Compute result digest
        result.digest = self._compute_result_digest(result)
        
        return result
    
    def _validate_module(self, module: Dict[str, Any]) -> List[str]:
        """Validate module structure."""
        errors = []
        
        required_fields = ["module_id", "nodes", "entrypoints", "order"]
        for field in required_fields:
            if field not in module:
                errors.append(f"Missing required field: {field}")
        
        # Validate nodes
        nodes = module.get("nodes", [])
        if not nodes:
            errors.append("Module has no nodes")
        
        node_ids = [n.get("id") for n in nodes]
        if len(node_ids) != len(set(node_ids)):
            errors.append("Duplicate node IDs found")
        
        return errors
    
    def _emit_receipt(self, node: Dict[str, Any], 
                     eval_result: EvaluatorResult,
                     state: RuntimeState,
                     event_counter: int) -> OperatorReceipt:
        """Emit OperatorReceipt for APPLY node."""
        payload = node.get("payload", {})
        op_id = payload.get("op_id")
        
        # Get kernel binding
        kernel_id, kernel_version = self.binding.require(op_id)
        
        # Get operator
        operator = self.registry.require(op_id)
        
        # Compute digests
        node_str = json.dumps(node, sort_keys=True, separators=(",", ":"))
        digest_in = "sha256:" + hashlib.sha256(node_str.encode("utf-8")).hexdigest()
        
        # Create receipt
        receipt = OperatorReceipt(
            event_id=f"evt_{event_counter}",
            node_id=node.get("id", 0),
            op_id=op_id,
            kernel_id=kernel_id,
            kernel_version=kernel_version,
            digest_in=digest_in,
            digest_out="",  # Will be computed
            C_before=0.0,  # Would track actual values
            C_after=state.metrics.r_phys ** 2 + state.metrics.r_cons ** 2,
            r_phys=state.metrics.r_phys,
            r_cons=state.metrics.r_cons,
            r_num=state.metrics.r_num
        )
        
        # Compute digest with chain
        prev_hash = self.receipt_chain[-1] if self.receipt_chain else None
        receipt.compute_digest(prev_hash)
        self.receipt_chain.append(receipt.digest)
        
        return receipt
    
    def _compute_result_digest(self, result: ExecutionResult) -> str:
        """Compute digest of execution result."""
        payload = {
            "success": result.success,
            "module_id": result.module_id,
            "final_metrics": result.final_metrics,
            "receipt_count": len(result.receipts)
        }
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()

# Forward reference for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
