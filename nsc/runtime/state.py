"""
NSC Runtime State Management

Per nsc_error_model.md: State tracks field values, node results, and coherence metrics.

Key classes:
- RuntimeState: Tracks execution state, field values, coherence metrics
- FieldHandle: Handle to a field with type and value
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import math

class ValueType(Enum):
    SCALAR = "SCALAR"
    PHASE = "PHASE"
    FIELD = "FIELD"
    TENSOR = "TENSOR"

@dataclass
class FieldHandle:
    """
    Handle to a field with type and value.
    
    Per docs/terminology.md: Environment maps field_id to actual field handles/data structures.
    """
    field_id: int
    value_type: ValueType
    value: Any
    shape: Optional[List[int]] = None
    
    def is_scalar(self) -> bool:
        return self.value_type == ValueType.SCALAR
    
    def is_nan(self) -> bool:
        if isinstance(self.value, (int, float)):
            return math.isnan(self.value) if isinstance(self.value, float) else False
        return False
    
    def is_inf(self) -> bool:
        if isinstance(self.value, float):
            return math.isinf(self.value)
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "field_id": self.field_id,
            "value_type": self.value_type.value,
            "value": self.value,
            "shape": self.shape,
            "is_nan": self.is_nan(),
            "is_inf": self.is_inf()
        }

@dataclass
class CoherenceMetrics:
    """
    Coherence metrics for runtime monitoring.
    
    Per coherence_spine/03_measurement/residuals_metrics.md:
    - r_phys: physics residual
    - r_cons: constraint residual  
    - r_num: numerical residual
    """
    r_phys: float = 0.0
    r_cons: float = 0.0
    r_num: float = 0.0
    
    def reset(self) -> None:
        self.r_phys = 0.0
        self.r_cons = 0.0
        self.r_num = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "r_phys": self.r_phys,
            "r_cons": self.r_cons,
            "r_num": self.r_num,
        }
    
    def max_residual(self) -> float:
        return max(abs(self.r_phys), abs(self.r_cons), abs(self.r_num))

@dataclass
class RuntimeState:
    """
    Runtime state during module execution.
    
    Tracks:
    - Field values (field_id -> FieldHandle)
    - Node results (node_id -> value)
    - Coherence metrics
    - Execution history
    """
    fields: Dict[int, FieldHandle] = field(default_factory=dict)
    node_results: Dict[int, Any] = field(default_factory=dict)
    metrics: CoherenceMetrics = field(default_factory=CoherenceMetrics)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def register_field(self, field_id: int, value_type: ValueType, 
                       value: Any, shape: Optional[List[int]] = None) -> FieldHandle:
        """Register a new field in the environment."""
        handle = FieldHandle(field_id, value_type, value, shape)
        self.fields[field_id] = handle
        return handle
    
    def get_field(self, field_id: int) -> Optional[FieldHandle]:
        """Get a field by ID."""
        return self.fields.get(field_id)
    
    def set_node_result(self, node_id: int, value: Any) -> None:
        """Store result of node evaluation."""
        self.node_results[node_id] = value
        self.execution_history.append({
            "node_id": node_id,
            "value": value,
            "metrics": self.metrics.to_dict()
        })
    
    def get_node_result(self, node_id: int) -> Optional[Any]:
        """Get result of node evaluation."""
        return self.node_results.get(node_id)
    
    def update_metrics(self, r_phys: float = None, r_cons: float = None, 
                       r_num: float = None) -> None:
        """Update coherence metrics."""
        if r_phys is not None:
            self.metrics.r_phys = r_phys
        if r_cons is not None:
            self.metrics.r_cons = r_cons
        if r_num is not None:
            self.metrics.r_num = r_num
    
    def reset(self) -> None:
        """Reset state for new execution."""
        self.node_results.clear()
        self.metrics.reset()
        self.execution_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": {k: v.to_dict() for k, v in self.fields.items()},
            "node_results": {str(k): v for k, v in self.node_results.items()},
            "metrics": self.metrics.to_dict(),
            "execution_history": self.execution_history
        }
