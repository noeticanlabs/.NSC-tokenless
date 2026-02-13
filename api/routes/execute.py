"""Execute endpoint for module execution."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import time

router = APIRouter()


class ExecuteRequest(BaseModel):
    """Request body for module execution."""
    module: Dict[str, Any]
    registry: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None


class ExecuteResponse(BaseModel):
    """Response body for module execution."""
    success: bool
    module_id: Optional[str] = None
    execution_time_ms: float
    results: Optional[Dict[str, Any]] = None
    receipts_count: int = 0
    braid_events_count: int = 0
    error: Optional[str] = None


@router.post("/execute", response_model=ExecuteResponse)
async def execute_module(request: ExecuteRequest):
    """Execute an NSC module."""
    start_time = time.time()
    
    try:
        # Import runtime components
        from nsc.runtime.interpreter import Interpreter
        from nsc.runtime.environment import Environment
        
        # Setup registry
        registry = request.registry or {
            "registry_id": "default",
            "version": "1.0",
            "operators": [
                {"op_id": 1001, "name": "ADD", "arg_types": ["float"], "ret_type": "float"},
                {"op_id": 1002, "name": "SUB", "arg_types": ["float"], "ret_type": "float"},
                {"op_id": 1003, "name": "MUL", "arg_types": ["float"], "ret_type": "float"},
                {"op_id": 1004, "name": "DIV", "arg_types": ["float"], "ret_type": "float"},
            ]
        }
        
        # Create environment
        env = Environment(registry=registry)
        
        # Setup default kernels
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
            "kernel_sub": lambda a, b: a - b,
            "kernel_mul": lambda a, b: a * b,
            "kernel_div": lambda a, b: a / b if b != 0 else float('inf'),
        }
        
        # Create module context
        from nsc.runtime.interpreter import ExecutionContext
        ctx = ExecutionContext(
            module=request.module,
            module_digest="",
            registry_id="default",
            registry_version="1.0",
            binding_id="default"
        )
        
        # Create interpreter and execute
        from nsc.npe_v1_0 import OperatorRegistry, KernelBinding
        
        # Create a registry from provided one
        ops = registry.get("operators", [])
        op_dict = {op["op_id"]: {"name": op["name"]} for op in ops}
        nsc_registry = OperatorRegistry(
            registry_id=registry.get("registry_id", "default"),
            version=registry.get("version", "1.0"),
            ops=op_dict
        )
        nsc_binding = KernelBinding(
            binding_id="default",
            registry_id=registry.get("registry_id", "default"),
            bindings={}
        )
        
        interpreter = Interpreter(registry=nsc_registry, binding=nsc_binding)
        result = interpreter.interpret(ctx)
        
        execution_time = (time.time() - start_time) * 1000
        
        if result.success:
            return ExecuteResponse(
                success=True,
                module_id=request.module.get("module_id"),
                execution_time_ms=execution_time,
                results=result.entrypoint_result,
                receipts_count=len(result.receipts) if result.receipts else 0
            )
        else:
            return ExecuteResponse(
                success=False,
                module_id=request.module.get("module_id"),
                execution_time_ms=execution_time,
                error=", ".join(result.errors) if result.errors else "Unknown error"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
