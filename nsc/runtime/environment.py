"""
NSC Runtime Environment

Per docs/terminology.md: Environment is the runtime-provided map from field_id 
to actual field handles/data structures and their types.

Key classes:
- FieldRegistry: Registry of all field types
- Environment: Maps field_id to FieldHandle with type information
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type, TypeVar
from nsc.runtime.state import FieldHandle, ValueType, RuntimeState

T = TypeVar('T')

@dataclass
class FieldRegistry:
    """
    Registry of available field types.
    
    Per nsc_type.schema.json: Type system with SCALAR, PHASE, FIELD types.
    """
    scalar_fields: Dict[int, str] = field(default_factory=dict)
    phase_fields: Dict[int, str] = field(default_factory=dict)
    field_fields: Dict[int, Type] = field(default_factory=dict)
    
    def register_scalar(self, field_id: int, name: str) -> None:
        """Register a scalar field."""
        self.scalar_fields[field_id] = name
    
    def register_phase(self, field_id: int, name: str) -> None:
        """Register a phase field."""
        self.phase_fields[field_id] = name
    
    def register_field(self, field_id: int, name: str, field_type: Type) -> None:
        """Register a compound field."""
        self.field_fields[field_id] = (name, field_type)
    
    def get_field_info(self, field_id: int) -> Optional[Dict[str, Any]]:
        """Get information about a field."""
        if field_id in self.scalar_fields:
            return {"type": "SCALAR", "name": self.scalar_fields[field_id]}
        if field_id in self.phase_fields:
            return {"type": "PHASE", "name": self.phase_fields[field_id]}
        if field_id in self.field_fields:
            name, ftype = self.field_fields[field_id]
            return {"type": "FIELD", "name": name, "field_type": str(ftype)}
        return None

@dataclass
class KernelContext:
    """
    Kernel execution context.
    
    Per nsc_module_linking.md: Kernel bindings map op_id to kernel implementations.
    """
    kernel_name: str
    version: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class Environment:
    """
    Runtime environment mapping field_id to FieldHandle.
    
    Per docs/terminology.md: Environment contents are outside the language 
    but are referenced deterministically.
    """
    
    def __init__(self, registry: FieldRegistry = None):
        self.registry = registry or FieldRegistry()
        self.state = RuntimeState()
    
    def create_scalar(self, field_id: int, value: float, name: str = None) -> FieldHandle:
        """Create and register a scalar field."""
        self.registry.register_scalar(field_id, name or f"scalar_{field_id}")
        return self.state.register_field(field_id, ValueType.SCALAR, value)
    
    def create_phase(self, field_id: int, value: complex, name: str = None) -> FieldHandle:
        """Create and register a phase field."""
        self.registry.register_phase(field_id, name or f"phase_{field_id}")
        return self.state.register_field(field_id, ValueType.PHASE, value)
    
    def get(self, field_id: int) -> Optional[FieldHandle]:
        """Get a field by ID."""
        return self.state.get_field(field_id)
    
    def set(self, field_id: int, value: Any) -> None:
        """Set field value."""
        handle = self.state.get_field(field_id)
        if handle:
            handle.value = value
    
    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the environment."""
        return self.state.to_dict()
    
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore environment from snapshot."""
        self.state = RuntimeState()
        for field_id, field_data in snapshot.get("fields", {}).items():
            self.state.register_field(
                int(field_id),
                ValueType(field_data["value_type"]),
                field_data["value"],
                field_data.get("shape")
            )
