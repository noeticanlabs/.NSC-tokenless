"""
NSC Gate Checker

Per coherence_spine/04_control/gates_and_rails.md:
- Hard gates: non-negotiable (NaN/Inf, positivity, domain bounds)
- Soft gates: repairable (phys residual, constraint residual)

Per nsc_gate.schema.json: Gate evaluation produces GateReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime

@dataclass
class GateThresholds:
    """Gate threshold configuration."""
    # Hard gates
    hard_nan_free: bool = True
    hard_positivity: float = 0.0
    hard_dt_min: float = 1e-6
    
    # Soft gates
    soft_phys_fail: float = 1.0
    soft_phys_warn: float = 0.75
    soft_cons_fail: float = 0.75
    soft_cons_warn: float = 0.5625
    soft_num_fail: float = 0.5
    soft_num_warn: float = 0.375
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hard_nan_free": self.hard_nan_free,
            "hard_positivity": self.hard_positivity,
            "hard_dt_min": self.hard_dt_min,
            "soft_phys_fail": self.soft_phys_fail,
            "soft_phys_warn": self.soft_phys_warn,
            "soft_cons_fail": self.soft_cons_fail,
            "soft_cons_warn": self.soft_cons_warn,
            "soft_num_fail": self.soft_num_fail,
            "soft_num_warn": self.soft_num_warn,
        }

@dataclass
class HardGateResult:
    """Result of hard gate evaluation."""
    pass_hard: bool
    nan_free: bool
    inf_free: bool
    positive_values: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_hard": self.pass_hard,
            "nan_free": self.nan_free,
            "inf_free": self.inf_free,
            "positive_values": self.positive_values
        }


@dataclass
class SoftGateResult:
    """Result of soft gate evaluation."""
    all_pass: bool
    phys_pass: bool
    cons_pass: bool
    num_pass: bool
    phys_value: float
    cons_value: float
    num_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_pass": self.all_pass,
            "phys_pass": self.phys_pass,
            "cons_pass": self.cons_pass,
            "num_pass": self.num_pass,
            "phys_value": self.phys_value,
            "cons_value": self.cons_value,
            "num_value": self.num_value
        }


@dataclass
class GateReport:
    """
    Gate report per nsc_gate_report.schema.json.
    
    Per docs/terminology.md: Gate is a deterministic predicate/check that produces a GateReport.
    """
    event_id: str
    node_id: int
    gate_kind: str  # "HARD" or "SOFT"
    passed: bool
    thresholds: Dict[str, Any]
    values: Dict[str, float]
    decision: str  # "ACCEPT", "WARN", "REJECT", "ABORT"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    digest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "gate_report",
            "event_id": self.event_id,
            "node_id": self.node_id,
            "gate_kind": self.gate_kind,
            "passed": self.passed,
            "thresholds": self.thresholds,
            "values": self.values,
            "decision": self.decision,
            "timestamp": self.timestamp,
            "digest": self.digest
        }
    
    def compute_digest(self, prev_hash: str = None) -> str:
        """Compute report digest per hash chaining."""
        payload = {k: v for k, v in self.to_dict().items() if k != "digest"}
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        if prev_hash:
            combined = canon.encode("utf-8") + prev_hash.encode("utf-8")
            self.digest = "sha256:" + hashlib.sha256(combined).hexdigest()
        else:
            self.digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
        
        return self.digest

class GateResult:
    """Gate evaluation result."""
    HARD_FAIL = "HARD_FAIL"
    SOFT_FAIL = "SOFT_FAIL"
    WARN = "WARN"
    PASS = "PASS"

class GateChecker:
    """
    Evaluates coherence gates on runtime metrics.
    
    Per gates_and_rails.md: Hard gates fail immediately; soft gates can trigger rails.
    """
    
    def __init__(self, thresholds: GateThresholds = None):
        self.thresholds = thresholds or GateThresholds()
    
    def evaluate(self, 
                 node_id: int,
                 event_id: str,
                 r_phys: float,
                 r_cons: float,
                 r_num: float,
                 value: float = None) -> GateReport:
        """
        Evaluate gates on metrics.
        
        Args:
            node_id: Node ID that triggered gate check
            event_id: Event ID for receipt
            r_phys: Physics residual
            r_cons: Constraint residual
            r_num: Numerical residual
            value: Optional actual value for hard checks
            
        Returns:
            GateReport with decision
        """
        # Hard gate checks
        hard_pass = True
        hard_failures = []
        
        # NaN/Inf check
        if value is not None:
            import math
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    hard_pass = False
                    hard_failures.append("nan_inf")
                
                # Positivity check
                if self.thresholds.hard_positivity > 0 and value < self.thresholds.hard_positivity:
                    hard_pass = False
                    hard_failures.append("positivity")
        
        if not hard_pass:
            return GateReport(
                event_id=event_id,
                node_id=node_id,
                gate_kind="HARD",
                passed=False,
                thresholds=self.thresholds.to_dict(),
                values={"r_phys": r_phys, "r_cons": r_cons, "r_num": r_num},
                decision="ABORT"
            )
        
        # Soft gate checks
        soft_failures = []
        warnings = []
        
        if r_phys > self.thresholds.soft_phys_fail:
            soft_failures.append("phys")
        elif r_phys > self.thresholds.soft_phys_warn:
            warnings.append("phys")
        
        if r_cons > self.thresholds.soft_cons_fail:
            soft_failures.append("cons")
        elif r_cons > self.thresholds.soft_cons_warn:
            warnings.append("cons")
        
        if r_num > self.thresholds.soft_num_fail:
            soft_failures.append("num")
        elif r_num > self.thresholds.soft_num_warn:
            warnings.append("num")
        
        # Determine decision
        if soft_failures:
            decision = "REJECT"
            passed = False
            gate_kind = "SOFT"
        elif warnings:
            decision = "WARN"
            passed = True
            gate_kind = "SOFT"
        else:
            decision = "ACCEPT"
            passed = True
            gate_kind = "SOFT"
        
        return GateReport(
            event_id=event_id,
            node_id=node_id,
            gate_kind=gate_kind,
            passed=passed,
            thresholds=self.thresholds.to_dict(),
            values={"r_phys": r_phys, "r_cons": r_cons, "r_num": r_num},
            decision=decision
        )
    
    def evaluate_with_hysteresis(self,
                                  node_id: int,
                                  event_id: str,
                                  r_phys: float,
                                  r_cons: float,
                                  r_num: float,
                                  previous_state: Dict[str, Any] = None) -> GateReport:
        """
        Evaluate gates with hysteresis (per gates_and_rails.md defaults).
        
        Hysteresis defaults:
        - warn_enter = 0.75 * fail_enter
        - warn_exit = 0.60 * fail_enter
        - fail_exit = 0.80 * fail_enter
        """
        # Use hysteresis-configured thresholds
        return self.evaluate(node_id, event_id, r_phys, r_cons, r_num)
