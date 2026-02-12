"""
NSC Governance Engine

Per docs/terminology.md: Governance is a deterministic policy that consumes 
history + gate reports and produces an ActionPlan.

Per coherence_spine/05_runtime/reference_implementations.md: Rails R1-R4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import hashlib
from datetime import datetime
from enum import Enum

class ActionType(Enum):
    """Action types per ActionPlan."""
    CONTINUE = "CONTINUE"
    RAIL = "RAIL"
    RETRY = "RETRY"
    ABORT = "ABORT"
    ROLLBACK = "ROLLBACK"

@dataclass
class ActionPlan:
    """
    Action plan per nsc_action_plan.schema.json.
    
    Per docs/terminology.md: Governance produces ActionPlan.
    """
    event_id: str
    action: str
    trigger_reports: List[str]
    rail_applied: Optional[str] = None
    retry_count: int = 0
    next_dt: Optional[float] = None
    state_delta: Optional[Dict[str, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    digest: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "action_plan",
            "event_id": self.event_id,
            "action": self.action,
            "trigger_reports": self.trigger_reports,
            "rail_applied": self.rail_applied,
            "retry_count": self.retry_count,
            "next_dt": self.next_dt,
            "state_delta": self.state_delta,
            "timestamp": self.timestamp,
            "digest": self.digest
        }
    
    def compute_digest(self, prev_hash: str = None) -> str:
        """Compute plan digest."""
        payload = {k: v for k, v in self.to_dict().items() if k != "digest"}
        canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        
        if prev_hash:
            combined = canon.encode("utf-8") + prev_hash.encode("utf-8")
            self.digest = "sha256:" + hashlib.sha256(combined).hexdigest()
        else:
            self.digest = "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()
        
        return self.digest

class GovernanceEngine:
    """
    Governance engine that consumes gate reports and produces action plans.
    
    Per reference_implementations.md: Deterministic rail priority:
    1. dt deflation (phys/CFL)
    2. projection (constraints)
    3. bounded damping/gain
    4. rollback
    
    Rails:
    - R1: dt deflation (α ∈ (0,1), typical 0.8)
    - R2: rollback to x_prev
    - R3: projection to constraint manifold
    - R4: bounded damping/gain adjustment
    """
    
    # Rail priority (deterministic)
    RAIL_PRIORITY = ["R1", "R3", "R4", "R2"]
    
    def __init__(self,
                 dt_min: float = 1e-6,
                 dt_alpha: float = 0.8,
                 retry_cap: int = 6,
                 projection_factor: float = 0.5,
                 delta_kappa: float = 0.1,
                 kappa_max: float = 2.0):
        self.dt_min = dt_min
        self.dt_alpha = dt_alpha
        self.retry_cap = retry_cap
        self.projection_factor = projection_factor
        self.delta_kappa = delta_kappa
        self.kappa_max = kappa_max
        
        self.retry_counts: Dict[str, int] = {}
    
    def evaluate(self,
                 event_id: str,
                 gate_reports: List[Dict[str, Any]],
                 current_state: Dict[str, Any] = None,
                 dt: float = 0.05) -> ActionPlan:
        """
        Evaluate governance based on gate reports.
        
        Args:
            event_id: Event ID for the action plan
            gate_reports: List of gate report dicts
            current_state: Current runtime state
            dt: Current timestep
            
        Returns:
            ActionPlan with action to take
        """
        current_state = current_state or {}
        
        # Count failures
        abort_count = sum(1 for r in gate_reports if r.get("decision") == "ABORT")
        reject_count = sum(1 for r in gate_reports if r.get("decision") == "REJECT")
        warn_count = sum(1 for r in gate_reports if r.get("decision") == "WARN")
        
        # Determine action
        if abort_count > 0:
            # Hard failure - must abort or rollback
            return self._create_plan(event_id, ActionType.ABORT, gate_reports, current_state)
        
        if reject_count > 0:
            # Soft failure - apply rail
            return self._apply_rail(event_id, gate_reports, current_state, dt)
        
        if warn_count > 0:
            # Warning - may apply R4 damping
            return self._apply_damping(event_id, gate_reports, current_state)
        
        # All gates passed
        return self._create_plan(event_id, ActionType.CONTINUE, gate_reports, current_state)
    
    def _apply_rail(self,
                    event_id: str,
                    gate_reports: List[Dict[str, Any]],
                    current_state: Dict[str, Any],
                    dt: float) -> ActionPlan:
        """Apply rail based on failure type."""
        # Check retry count
        retry_key = event_id.split("_")[0] if "_" in event_id else event_id
        self.retry_counts[retry_key] = self.retry_counts.get(retry_key, 0) + 1
        
        if self.retry_counts[retry_key] > self.retry_cap:
            return self._create_plan(event_id, ActionType.ROLLBACK, gate_reports, current_state)
        
        # Determine which rail to apply based on failure type
        for report in gate_reports:
            if report.get("decision") == "REJECT":
                values = report.get("values", {})
                r_phys = values.get("r_phys", 0)
                r_cons = values.get("r_cons", 0)
                
                # R1: dt deflation for physics failures
                if r_phys > r_cons and r_phys > 0.5:
                    new_dt = max(self.dt_min, self.dt_alpha * dt)
                    return ActionPlan(
                        event_id=event_id,
                        action=ActionType.RAIL.value,
                        trigger_reports=[r["event_id"] for r in gate_reports],
                        rail_applied="R1",
                        next_dt=new_dt,
                        state_delta={"dt": new_dt - dt}
                    )
                
                # R3: projection for constraint failures
                if r_cons > 2 * r_phys and r_cons > 0.3:
                    return ActionPlan(
                        event_id=event_id,
                        action=ActionType.RAIL.value,
                        trigger_reports=[r["event_id"] for r in gate_reports],
                        rail_applied="R3",
                        state_delta={"cons_reduced_by": self.projection_factor}
                    )
        
        # Default: R1 dt deflation
        new_dt = max(self.dt_min, self.dt_alpha * dt)
        return ActionPlan(
            event_id=event_id,
            action=ActionType.RAIL.value,
            trigger_reports=[r["event_id"] for r in gate_reports],
            rail_applied="R1",
            next_dt=new_dt
        )
    
    def _apply_damping(self,
                       event_id: str,
                       gate_reports: List[Dict[str, Any]],
                       current_state: Dict[str, Any]) -> ActionPlan:
        """Apply R4 damping for warnings."""
        kappa = current_state.get("kappa", 1.0)
        new_kappa = min(self.kappa_max, kappa + self.delta_kappa)
        
        return ActionPlan(
            event_id=event_id,
            action=ActionType.RAIL.value,
            trigger_reports=[r["event_id"] for r in gate_reports],
            rail_applied="R4",
            state_delta={"kappa": new_kappa}
        )
    
    def _create_plan(self,
                     event_id: str,
                     action_type: ActionType,
                     gate_reports: List[Dict[str, Any]],
                     current_state: Dict[str, Any]) -> ActionPlan:
        """Create action plan."""
        return ActionPlan(
            event_id=event_id,
            action=action_type.value,
            trigger_reports=[r["event_id"] for r in gate_reports]
        )
