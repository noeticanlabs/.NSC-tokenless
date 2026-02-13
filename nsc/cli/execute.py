"""Execute command for NSC CLI."""

import json
import sys
from pathlib import Path
from typing import Any, Dict

import click

from nsc.runtime.interpreter import Interpreter, ExecutionContext
from nsc.runtime.environment import Environment
from nsc.npe_v1_0 import OperatorRegistry, KernelBinding


@click.command()
@click.argument('module', type=click.Path(exists=True))
@click.option('--registry', '-r', type=click.Path(exists=True), help='Path to operator registry')
@click.option('--json-output', '-j', 'json_output', is_flag=True, help='Output as JSON')
def execute_command(module, registry, json_output):
    """Execute an NSC module with full receipt emission."""
    try:
        module_path = Path(module)
        registry_path = registry
        
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
        
        # Setup kernels in the binding
        # (Kernels are handled by the Interpreter through the Executor)
        
        # Convert registry to OperatorRegistry and KernelBinding
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
        
        # Create execution context
        ctx = ExecutionContext(
            module=module,
            module_digest="",
            registry_id=registry.get("registry_id", "default"),
            registry_version=registry.get("version", "1.0"),
            binding_id="default"
        )
        
        # Create interpreter and execute
        interpreter = Interpreter(registry=nsc_registry, binding=nsc_binding)
        result = interpreter.interpret(ctx)
        
        # Output results
        output = {
            "success": result.success,
            "module_id": result.module_id,
            "execution_time_ms": result.execution_time_ms,
        }
        
        if result.success:
            output["results"] = result.entrypoint_result
            output["receipts_count"] = len(result.receipts) if result.receipts else 0
            output["errors"] = result.errors
            
            if json_output:
                print(json.dumps(output, indent=2))
            else:
                print(f"✓ Execution successful")
                print(f"  Module: {module.get('module_id')}")
                print(f"  Execution time: {result.execution_time_ms:.2f}ms")
                print(f"  Receipts: {output['receipts_count']}")
        else:
            output["errors"] = result.errors
            print(json.dumps(output, indent=2), file=sys.stderr)
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
