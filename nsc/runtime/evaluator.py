"""
NSC Node Evaluator

Per docs/language/nsc_module_linking.md: Nodes form a directed graph via integer references.

Evaluates node kinds:
- FIELD_REF: Get field value
- APPLY: Execute operator
- SEQ: Sequential evaluation order
- GATE_CHECK: Evaluate coherence gates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import math
import json

from nsc.runtime.state import RuntimeState, CoherenceMetrics
from nsc.runtime.environment import Environment
from nsc.npe_v1_0 import OperatorRegistry, KernelBinding

@dataclass
class EvaluatorResult:
    """Result of node evaluation."""
    success: bool
    value: Any = None
    error: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    node_id: int = 0

class NodeEvaluator:
    """
    Evaluates individual NSC nodes.
    
    Per docs/conformance_levels.md L0: Must enforce explicit evaluation order via SEQ nodes.
    """
    
    def __init__(self, 
                 registry: "OperatorRegistry",
                 binding: "KernelBinding",
                 environment: Environment,
                 state: RuntimeState):
        self.registry = registry
        self.binding = binding
        self.environment = environment
        self.state = state
    
    def evaluate(self, node: Dict[str, Any]) -> EvaluatorResult:
        """
        Evaluate a node and return result.
        
        Node structure per nsc_node.schema.json:
        {
            "id": int,
            "kind": "APPLY" | "FIELD_REF" | "SEQ" | "GATE_CHECK",
            "type": TypeTag,
            "payload": {...}
        }
        """
        node_id = node.get("id", 0)
        kind = node.get("kind", "")
        
        try:
            if kind == "FIELD_REF":
                return self._eval_field_ref(node)
            elif kind == "APPLY":
                return self._eval_apply(node)
            elif kind == "SEQ":
                return self._eval_seq(node)
            elif kind == "GATE_CHECK":
                return self._eval_gate_check(node)
            else:
                return EvaluatorResult(
                    success=False,
                    error=f"Unknown node kind: {kind}",
                    node_id=node_id
                )
        except Exception as e:
            return EvaluatorResult(
                success=False,
                error=str(e),
                node_id=node_id
            )
    
    def _eval_field_ref(self, node: Dict[str, Any]) -> EvaluatorResult:
        """Evaluate FIELD_REF: get field value from environment."""
        node_id = node.get("id", 0)
        payload = node.get("payload", {})
        field_id = payload.get("field_id")
        
        if field_id is None:
            return EvaluatorResult(success=False, error="FIELD_REF missing field_id", node_id=node_id)
        
        handle = self.environment.get(field_id)
        if handle is None:
            return EvaluatorResult(success=False, error=f"Field {field_id} not found", node_id=node_id)
        
        return EvaluatorResult(
            success=True,
            value=handle.value,
            metrics={"r_num": 1.0 if handle.is_nan() or handle.is_inf() else 0.0},
            node_id=node_id
        )
    
    def _eval_apply(self, node: Dict[str, Any]) -> EvaluatorResult:
        """
        Evaluate APPLY: execute operator on arguments.
        
        Per docs/conformance_levels.md L0: Must emit OperatorReceipt for every executed APPLY node.
        """
        node_id = node.get("id", 0)
        payload = node.get("payload", {})
        op_id = payload.get("op_id")
        args = payload.get("args", [])
        
        # Validate operator exists
        try:
            operator = self.registry.require(op_id)
        except KeyError:
            return EvaluatorResult(
                success=False,
                error=f"Operator {op_id} not found in registry",
                node_id=node_id
            )
        
        # Validate kernel binding
        try:
            kernel_id, kernel_version = self.binding.require(op_id)
        except KeyError:
            return EvaluatorResult(
                success=False,
                error=f"Kernel binding not found for op_id {op_id}",
                node_id=node_id
            )
        
        # Evaluate arguments
        arg_values = []
        for arg_id in args:
            result = self.state.get_node_result(arg_id)
            if result is None:
                return EvaluatorResult(
                    success=False,
                    error=f"Argument node {arg_id} not evaluated",
                    node_id=node_id
                )
            arg_values.append(result)
        
        # Execute operator (simplified - actual kernel would be called)
        result = self._execute_kernel(operator, kernel_id, kernel_version, arg_values)
        
        # Check for numerical errors
        r_num = 1.0 if (isinstance(result, float) and (math.isnan(result) or math.isinf(result))) else 0.0
        
        # Store result
        self.state.set_node_result(node_id, result)
        
        return EvaluatorResult(
            success=True,
            value=result,
            metrics={"r_num": r_num},
            node_id=node_id
        )
    
    def _execute_kernel(self, operator, kernel_id: str, kernel_version: str, 
                       args: List[Any]) -> Any:
        """
        Execute kernel for operator.
        
        Per docs/terminology.md: Kernel is a concrete implementation identified by 
        (kernel_id, kernel_version).
        """
        # Simplified kernel execution
        # In real implementation, would call actual kernel based on kernel_id
        
        op_name = operator.token
        
        if op_name == "source_plus":
            return args[0] + 0.1 if len(args) > 0 else 0.0
        elif op_name == "twist":
            return args[0] * 0.95 if len(args) > 0 else 0.0
        elif op_name == "delta":
            return args[0] - 0.05 if len(args) > 0 else 0.0
        elif op_name == "regulate":
            return max(0.0, min(1.0, args[0])) if len(args) > 0 else 0.0
        elif op_name == "sink_minus":
            return args[0] - 0.1 if len(args) > 0 else 0.0
        elif op_name == "return_to_baseline":
            return 0.5 * args[0] + 0.5 * args[1] if len(args) > 1 else args[0]
        else:
            # Generic fallback
            return args[0] if args else 0.0
    
    def _eval_seq(self, node: Dict[str, Any]) -> EvaluatorResult:
        """
        Evaluate SEQ: evaluate nodes in order.
        
        Per docs/conformance_levels.md L0: Must enforce explicit evaluation order via SEQ nodes.
        """
        node_id = node.get("id", 0)
        payload = node.get("payload", {})
        seq = payload.get("seq", [])
        
        results = []
        for child_id in seq:
            # Find child node (would be in module.nodes in real implementation)
            # For now, check state
            result = self.state.get_node_result(child_id)
            if result is not None:
                results.append(result)
        
        return EvaluatorResult(
            success=True,
            value=results,
            node_id=node_id
        )
    
    def _eval_gate_check(self, node: Dict[str, Any]) -> EvaluatorResult:
        """
        Evaluate GATE_CHECK: evaluate coherence gates.
        
        Per coherence_spine/04_control/gates_and_rails.md:
        - Hard gates: NaN/Inf, positivity, domain bounds
        - Soft gates: phys residual, constraint residual, conservation drift
        """
        node_id = node.get("id", 0)
        payload = node.get("payload", {})
        thresholds = payload.get("thresholds", {})
        input_node_id = payload.get("input_node_id")
        
        # Get input value
        input_value = self.state.get_node_result(input_node_id) if input_node_id else None
        
        # Evaluate gates
        hard_pass = True
        soft_failures = []
        
        # Hard gate: NaN/Inf check
        if isinstance(input_value, float):
            if math.isnan(input_value) or math.isinf(input_value):
                hard_pass = False
        
        # Hard gate: positivity
        if input_value is not None and input_value < 0:
            if thresholds.get("hard_positivity", 0) > 0:
                hard_pass = False
        
        # Soft gate: threshold check
        if input_value is not None:
            phys_fail = thresholds.get("soft_phys_fail", 1.0)
            if abs(input_value) > phys_fail:
                soft_failures.append("phys")
        
        # Determine decision
        if not hard_pass:
            decision = "ABORT"
        elif soft_failures:
            decision = "REJECT"
        else:
            decision = "ACCEPT"
        
        return EvaluatorResult(
            success=True,
            value={"decision": decision, "hard_pass": hard_pass, "soft_failures": soft_failures},
            node_id=node_id
        )
