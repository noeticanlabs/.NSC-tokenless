"""
NSC Tokenless Runtime Engine

This module provides the core runtime for executing NSC modules:
- Executor: Main module execution engine
- Evaluator: Node evaluation (APPLY, FIELD_REF, SEQ, GATE_CHECK)
- Interpreter: Main interpreter loop with SEQ ordering
- State: Runtime state management
- Environment: Field environment mapping

L0-L2 Conformance:
- L0: Module validation, registry resolution, kernel binding, OperatorReceipt emission
- L1: GateCheck evaluation, Governance, ReplayVerification
- L2: GLLL decode, REJECT handling
"""

from nsc.runtime.executor import Executor, ExecutionResult, ExecutionConfig
from nsc.runtime.evaluator import NodeEvaluator, EvaluatorResult
from nsc.runtime.interpreter import Interpreter, ExecutionContext
from nsc.runtime.state import RuntimeState, FieldHandle
from nsc.runtime.environment import Environment, FieldRegistry

__all__ = [
    "Executor",
    "ExecutionResult", 
    "ExecutionConfig",
    "NodeEvaluator",
    "EvaluatorResult",
    "Interpreter",
    "ExecutionContext",
    "RuntimeState",
    "FieldHandle",
    "Environment",
    "FieldRegistry",
]
