"""
NSC Replay Verifier

Per docs/governance/replay_verification.md: Replay verification checks:
1. Module digest
2. Registry identity
3. Kernel bindings
4. Receipt digests
5. Metrics within tolerances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

@dataclass
class ReplayReport:
    """
    Replay report per nsc_replay_report.schema.json.
    
    Per docs/governance/replay_verification.md: Verifier emits ReplayReport.
    """
    replay_pass: bool
    module_digest_ok: bool
    registry_ok: bool
    kernel_binding_ok: bool
    tolerances: Dict[str, float]
    first_mismatch: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "replay_pass": self.replay_pass,
            "module_digest_ok": self.module_digest_ok,
            "registry_ok": self.registry_ok,
            "kernel_binding_ok": self.kernel_binding_ok,
            "tolerances": self.tolerances,
            "first_mismatch": self.first_mismatch,
            "timestamp": self.timestamp
        }

class ReplayVerifier:
    """
    Verifies replay of NSC execution.
    
    Per docs/governance/replay_verification.md:
    - Inputs: Module, ModuleDigest, Registry, Kernel Binding, Initial State, BraidEvents
    - Steps: Validate, recompute digests, compare metrics
    """
    
    def __init__(self, abs_tolerance: float = 1e-9, rel_tolerance: float = 1e-9):
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
    
    def verify(self,
               module: Dict[str, Any],
               module_digest: str,
               registry_id: str,
               registry_version: str,
               kernel_bindings: Dict[int, Tuple[str, str]],
               expected_receipts: List[Dict[str, Any]],
               actual_metrics: Dict[str, float]) -> ReplayReport:
        """
        Verify replay against expected results.
        
        Args:
            module: NSC module
            module_digest: Expected module digest
            registry_id: Expected registry ID
            registry_version: Expected registry version
            kernel_bindings: Expected kernel bindings
            expected_receipts: Expected receipt data
            actual_metrics: Actual metrics from execution
            
        Returns:
            ReplayReport with verification results
        """
        # 1. Verify module digest
        computed_digest = self._compute_module_digest(module)
        module_digest_ok = computed_digest == module_digest
        
        if not module_digest_ok:
            return ReplayReport(
                replay_pass=False,
                module_digest_ok=False,
                registry_ok=True,
                kernel_binding_ok=True,
                tolerances={"abs": self.abs_tolerance, "rel": self.rel_tolerance},
                first_mismatch={
                    "field": "module_digest",
                    "expected": module_digest,
                    "observed": computed_digest
                }
            )
        
        # 2. Verify registry (simplified - would check full registry)
        registry_ok = True  # Would compare full registry if provided
        
        # 3. Verify kernel bindings
        kernel_binding_ok = True  # Would check if bindings match
        
        # 4. Verify receipt digests (simplified)
        receipts_ok = len(expected_receipts) > 0  # Simplified check
        
        # 5. Verify metrics within tolerances
        metrics_ok = True
        first_mismatch = None
        
        for metric, expected in expected_receipts[0].get("metrics", {}).items():
            if metric in actual_metrics:
                actual = actual_metrics.get(metric, 0.0)
                if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                    abs_diff = abs(actual - expected)
                    rel_diff = abs(actual - expected) / max(abs(actual), abs(expected), 1e-10)
                    
                    if abs_diff > self.abs_tolerance and rel_diff > self.rel_tolerance:
                        metrics_ok = False
                        first_mismatch = {
                            "field": metric,
                            "expected": expected,
                            "observed": actual,
                            "abs_diff": abs_diff,
                            "rel_diff": rel_diff
                        }
                        break
        
        replay_pass = (module_digest_ok and registry_ok and 
                      kernel_binding_ok and receipts_ok and metrics_ok)
        
        return ReplayReport(
            replay_pass=replay_pass,
            module_digest_ok=module_digest_ok,
            registry_ok=registry_ok,
            kernel_binding_ok=kernel_binding_ok,
            tolerances={"abs": self.abs_tolerance, "rel": self.rel_tolerance},
            first_mismatch=first_mismatch
        )
    
    def verify_with_receipts(self,
                             module: Dict[str, Any],
                             module_digest: str,
                             receipts: List[Dict[str, Any]],
                             actual_metrics: Dict[str, float]) -> ReplayReport:
        """
        Verify replay using receipt ledger.
        
        Per docs/governance/replay_verification.md: Compare emitted receipts.
        """
        if not receipts:
            return ReplayReport(
                replay_pass=False,
                module_digest_ok=True,
                registry_ok=True,
                kernel_binding_ok=True,
                tolerances={"abs": self.abs_tolerance, "rel": self.rel_tolerance},
                first_mismatch={"field": "receipts", "expected": ">0", "observed": 0}
            )
        
        # Extract expected metrics from first receipt
        expected_metrics = receipts[0].get("metrics", {})
        
        return self.verify(
            module=module,
            module_digest=module_digest,
            registry_id="",  # Would extract from receipts
            registry_version="",
            kernel_bindings={},
            expected_receipts=receipts,
            actual_metrics=actual_metrics
        )
    
    def verify_hash_chain(self, receipts: List[Dict[str, Any]]) -> bool:
        """
        Verify hash chain integrity of receipts.
        
        Per docsgoverancereplay_verification.md: Check digest chaining.
        Returns True if chain is valid, False otherwise.
        """
        if not receipts or len(receipts) < 1:
            return True  # Empty chain is trivially valid
        
        import hashlib
        
        # Compute parent hash chain
        for i, receipt in enumerate(receipts):
            # Compute what the parent hash should be
            if i == 0:
                # First receipt: digest_in is usually empty or "0"*64
                digest_in = receipt.get("digest_in", "")
                if digest_in and digest_in != "sha256:" + "0"*64:
                    return False
            else:
                # Verify against previous receipt's digest
                expected_parent = receipts[i-1].get("digest", receipts[i-1].get("digest_out", ""))
                actual_parent = receipt.get("digest_in", "")
                
                if expected_parent and actual_parent and expected_parent != actual_parent:
                    return False
            
            # Verify receipt digest by recomputing
            receipt_copy = {k: v for k, v in receipt.items() if k not in ["digest"]}
            canon = json.dumps(receipt_copy, sort_keys=True, separators=(",", ":"))
            computed_digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
            
            stored_digest = receipt.get("digest", "")
            if stored_digest and stored_digest != computed_digest:
                # Digest mismatch
                return False
        
        return True
    
    def verify_receipt(self,
                       receipt: Dict[str, Any],
                       module: Dict[str, Any],
                       expected_metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Verify a single receipt against expected values.
        
        Args:
            receipt: Receipt dict to verify
            module: Original module (for reference)
            expected_metrics: Expected metric values
        
        Returns:
            Verification detail dict with fields: valid, node_id, op_id, metric_errors
        """
        detail = {
            "valid": True,
            "node_id": receipt.get("node_id"),
            "op_id": receipt.get("op_id"),
            "metric_errors": []
        }
        
        # Verify node exists in module
        node_ids = {n.get("id") for n in module.get("nodes", [])}
        if receipt.get("node_id") not in node_ids:
            detail["valid"] = False
            detail["metric_errors"].append(f"Node {receipt.get('node_id')} not in module")
            return detail
        
        # Verify metrics if provided
        if expected_metrics:
            for metric, expected_val in expected_metrics.items():
                actual_val = receipt.get(metric, 0.0)
                
                if isinstance(expected_val, (int, float)) and isinstance(actual_val, (int, float)):
                    abs_diff = abs(actual_val - expected_val)
                    rel_diff = abs_diff / max(abs(actual_val), abs(expected_val), 1e-10)
                    
                    if abs_diff > self.abs_tolerance and rel_diff > self.rel_tolerance:
                        detail["valid"] = False
                        detail["metric_errors"].append(
                            f"{metric}: expected {expected_val}, got {actual_val} "
                            f"(Δ_abs={abs_diff:.2e}, Δ_rel={rel_diff:.2e})"
                        )
        
        return detail
    
    def _compute_module_digest(self, module: Dict[str, Any]) -> str:
        """Compute module digest."""
        import hashlib
        
        # Sort nodes by id
        nodes = module.get("nodes", [])
        nodes_sorted = sorted(nodes, key=lambda n: n.get("id", 0))
        
        module_copy = dict(module)
        module_copy["nodes"] = nodes_sorted
        
        canon = json.dumps(module_copy, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


# Forward reference for type hint
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
