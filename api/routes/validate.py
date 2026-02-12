"""Validate endpoint for module validation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter()


class ValidateRequest(BaseModel):
    """Request body for module validation."""
    module: Dict[str, Any]


class ValidateResponse(BaseModel):
    """Response body for module validation."""
    valid: bool
    module_id: Optional[str] = None
    node_count: int = 0
    seq_length: int = 0
    entrypoints_count: int = 0
    errors: List[str] = []
    warnings: List[str] = []


def validate_module(module: Dict[str, Any]) -> ValidateResponse:
    """Validate an NSC module."""
    errors: List[str] = []
    warnings: List[str] = []
    
    # Validate module schema structure
    if "nodes" not in module:
        errors.append("Missing required field: 'nodes'")
    if "seq" not in module:
        errors.append("Missing required field: 'seq'")
    
    if errors:
        return ValidateResponse(
            valid=False,
            module_id=module.get("module_id"),
            errors=errors,
            warnings=warnings
        )
    
    # Validate node references
    node_ids = {node.get("id") for node in module.get("nodes", [])}
    seq_ids = module.get("seq", [])
    
    for node_id in seq_ids:
        if node_id not in node_ids:
            errors.append(f"SEQ references non-existent node: {node_id}")
    
    # Check node structure
    for node in module.get("nodes", []):
        if "id" not in node:
            errors.append("Node missing 'id' field")
        if "kind" not in node:
            errors.append(f"Node {node.get('id', '?')} missing 'kind' field")
        if "type" not in node and "type_tag" not in node:
            errors.append(f"Node {node.get('id', '?')} missing 'type' or 'type_tag' field")
    
    # Check APPLY nodes have valid op_ids
    for node in module.get("nodes", []):
        if node.get("kind") == "APPLY":
            if "op_id" not in node:
                errors.append(f"APPLY node {node.get('id', '?')} missing 'op_id'")
            if "inputs" not in node:
                errors.append(f"APPLY node {node.get('id', '?')} missing 'inputs'")
    
    # Check entrypoints exist
    for ep in module.get("entrypoints", []):
        if ep not in node_ids:
            errors.append(f"Entrypoint {ep} references non-existent node")
    
    return ValidateResponse(
        valid=len(errors) == 0,
        module_id=module.get("module_id"),
        node_count=len(module.get("nodes", [])),
        seq_length=len(seq_ids),
        entrypoints_count=len(module.get("entrypoints", [])),
        errors=errors,
        warnings=warnings
    )


@router.post("/validate", response_model=ValidateResponse)
async def validate_module_endpoint(request: ValidateRequest):
    """Validate an NSC module."""
    try:
        return validate_module(request.module)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
