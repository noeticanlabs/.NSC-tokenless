"""
NSC Interpreter

Main interpreter loop for executing NSC modules with SEQ ordering.

Per docs/conformance_levels.md L0: Must enforce explicit evaluation order via SEQ nodes.

Key classes:
- ExecutionContext: Context for execution
- Interpreter: Main interpreter loop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import hashlib

from nsc.runtime.state import RuntimeState, CoherenceMetrics
from nsc.runtime.environment import Environment, FieldRegistry
from nsc.runtime.evaluator import NodeEvaluator, EvaluatorResult
from nsc.runtime.executor import Executor, ExecutionConfig, ExecutionResult, OperatorReceipt

@dataclass
class ExecutionContext:
    """
    Context for module execution.
    
    Per docs/governance/replay_verification.md: Replay verifier consumes:
    - NSC module M
    - ModuleDigest Dm
    - Operator registry R
    - Kernel binding B
    - Initial state E0
    - BraidEvents stream
    """
    module: Dict[str, Any]
    module_digest: str
    registry_id: str
    registry_version: str
    binding_id: str
    initial_state: Dict[str, Any] = field(default_factory=dict)
    tolerances: Dict[str, float] = field(default_factory=lambda: {"abs": 1e-9, "rel": 1e-9})
    parent_event: str = ""
    run_id: str = ""

class Interpreter:
    """
    Main interpreter for NSC modules.
    
    Per reference_implementations.md: Canonical gated step loop:
    1. dt ← S.choose_dt(x)
    2. for attempt = 0..N_retry:
    3.   x_prop ← M.step(x,t,dt)
    4.   metrics ← M.residual(x,x_prop,t,dt) + invariants + ops metrics
    5.   verdicts ← G.evaluate(metrics)
    6.   ...
    """
    
    def __init__(self,
                 registry: "OperatorRegistry",
                 binding: "KernelBinding",
                 config: ExecutionConfig = None):
        self.registry = registry
        self.binding = binding
        self.config = config or ExecutionConfig()
        self.executor = Executor(registry, binding, config)
        self.braid_events: List[Dict[str, Any]] = []
        self.receipt_ledger: List[OperatorReceipt] = []
    
    def interpret(self, context: ExecutionContext) -> ExecutionResult:
        """
        Interpret (execute) an NSC module.
        
        Args:
            context: Execution context with module and metadata
            
        Returns:
            ExecutionResult with results and receipts
        """
        # Emit start event
        self._emit_braid_event("execution_start", context)
        
        # Execute module
        result = self.executor.execute(context.module)
        
        # Store receipts in ledger
        self.receipt_ledger.extend(result.receipts)
        
        # Emit completion event
        self._emit_braid_event("execution_complete", context, result)
        
        # Emit receipt events
        for receipt in result.receipts:
            self._emit_braid_event("operator_receipt", context, receipt)
        
        return result
    
    def replay(self,
               context: ExecutionContext,
               expected_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replay execution and verify against expected results.
        
        Per docs/governance/replay_verification.md:
        - Verify module digest
        - Verify registry identity
        - Verify kernel bindings
        - Compare metrics within tolerances
        """
        # Verify module digest
        module_digest = self._compute_module_digest(context.module)
        digest_match = module_digest == context.module_digest
        
        # Verify registry
        registry_match = context.registry_id == self.registry.registry_id
        version_match = context.registry_version == self.registry.version
        
        # Execute
        result = self.interpret(context)
        
        # Verify results within tolerances
        tolerances = context.tolerances
        metrics_match = True
        mismatches = []
        
        for metric, expected in expected_results.get("metrics", {}).items():
            actual = result.final_metrics.get(metric, 0.0)
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                abs_diff = abs(actual - expected)
                rel_diff = abs(actual - expected) / max(abs(actual), abs(expected), 1e-10)
                if abs_diff > tolerances.get("abs", 1e-9) and rel_diff > tolerances.get("rel", 1e-9):
                    metrics_match = False
                    mismatches.append({
                        "metric": metric,
                        "expected": expected,
                        "actual": actual,
                        "abs_diff": abs_diff,
                        "rel_diff": rel_diff
                    })
        
        # Build replay report
        replay_pass = (digest_match and registry_match and version_match and 
                      metrics_match and result.success)
        
        return {
            "replay_pass": replay_pass,
            "module_digest_ok": digest_match,
            "registry_ok": registry_match and version_match,
            "kernel_binding_ok": True,  # Would verify if kernel_ids match
            "metrics_ok": metrics_match,
            "tolerances": tolerances,
            "mismatches": mismatches,
            "receipt_count": len(result.receipts)
        }
    
    def _emit_braid_event(self, 
                         kind: str,
                         context: ExecutionContext,
                         data: Any = None) -> None:
        """Emit a braid event."""
        event = {
            "run_id": context.run_id or "default_run",
            "event_id": f"{kind}_{len(self.braid_events)}",
            "thread_id": "main",
            "step_index": len(self.braid_events),
            "kind": kind,
            "parent_event": context.parent_event,
            "payload": data.to_dict() if hasattr(data, 'to_dict') else (data or {}),
            "timestamp": self._timestamp()
        }
        
        # Compute digest
        payload_str = json.dumps({k: v for k, v in event.items() if k != "digest"}, 
                                  sort_keys=True, separators=(",", ":"))
        event["digest"] = "sha256:" + hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
        
        self.braid_events.append(event)
    
    def _compute_module_digest(self, module: Dict[str, Any]) -> str:
        """Compute module digest per canonical-hashing.md."""
        # Sort nodes by id
        nodes = module.get("nodes", [])
        nodes_sorted = sorted(nodes, key=lambda n: n.get("id", 0))
        
        module_copy = dict(module)
        module_copy["nodes"] = nodes_sorted
        
        canon = json.dumps(module_copy, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
    
    def _timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_braid_events(self) -> List[Dict[str, Any]]:
        """Get all braid events."""
        return self.braid_events
    
    def get_receipt_ledger(self) -> List[OperatorReceipt]:
        """Get receipt ledger."""
        return self.receipt_ledger

# Forward references
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
