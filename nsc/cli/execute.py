"""Execute command for NSC CLI."""

import json
import sys
from pathlib import Path
from typing import Any, Dict


def execute_command(args) -> int:
    """Execute an NSC module with full receipt emission."""
    try:
        module_path = Path(args.module)
        registry_path = getattr(args, "registry", None)
        
        # Load module
        with open(module_path) as f:
            module = json.load(f)
        
        # Load registry
        if registry_path:
            with open(registry_path) as f:
                registry = json.load(f)
        else:
            # Use default registry
            registry = {
                "registry_id": "default",
                "version": "1.0",
                "operators": [
                    {"op_id": 1001, "name": "ADD", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1002, "name": "SUB", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1003, "name": "MUL", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1004, "name": "DIV", "arg_types": ["float"], "ret_type": "float"},
                ]
            }
        
        # Import runtime components
        from nsc.runtime.interpreter import Interpreter
        from nsc.runtime.environment import Environment
        
        # Create environment
        env = Environment(registry=registry)
        
        # Setup default kernels
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
            "kernel_sub": lambda a, b: a - b,
            "kernel_mul": lambda a, b: a * b,
            "kernel_div": lambda a, b: a / b if b != 0 else float('inf'),
        }
        
        # Create interpreter and execute
        interpreter = Interpreter(env=env)
        result = interpreter.interpret(module)
        
        # Output results
        output = {
            "success": result.success,
            "module_id": module.get("module_id"),
            "execution_time_ms": result.execution_time_ms,
        }
        
        if result.success:
            output["results"] = result.results
            output["receipts_count"] = len(result.receipts) if result.receipts else 0
            output["braid_events_count"] = len(result.braid_events) if result.braid_events else 0
            
            if getattr(args, "json", False):
                print(json.dumps(output, indent=2))
            else:
                print(f"✓ Execution successful")
                print(f"  Module: {module.get('module_id')}")
                print(f"  Execution time: {result.execution_time_ms:.2f}ms")
                print(f"  Receipts: {output['receipts_count']}")
                print(f"  Braid events: {output['braid_events_count']}")
        else:
            output["error"] = result.error
            print(json.dumps(output, indent=2), file=sys.stderr)
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
