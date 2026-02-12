"""NSC Tokenless v1.1 Command Line Interface."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

from nsc.runtime.interpreter import Interpreter
from nsc.runtime.environment import Environment
from nsc.governance.gate_checker import GateChecker
from nsc.glyphs import GLLLCodebook


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NSC Tokenless v1.1 - Tokenless execution engine"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Execute command
    execute_parser = subparsers.add_parser("execute", help="Execute an NSC module")
    execute_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    execute_parser.add_argument("--registry", type=Path, help="Path to operator registry")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate an NSC module")
    validate_parser.add_argument("module", type=Path, help="Path to NSC module JSON")
    
    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode GLLL signal")
    decode_parser.add_argument("signal", type=str, help="JSON array of signal values")
    decode_parser.add_argument("--n", type=int, default=16, help="GLLL n parameter")
    
    # Gate command
    gate_parser = subparsers.add_parser("gate", help="Evaluate gate policy")
    gate_parser.add_argument("residuals", type=str, help="JSON object with residuals")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version info")
    
    args = parser.parse_args()
    
    if args.command == "execute":
        execute_module(args)
    elif args.command == "validate":
        validate_module(args)
    elif args.command == "decode":
        decode_signal(args)
    elif args.command == "gate":
        evaluate_gates(args)
    elif args.command == "version":
        print("NSC Tokenless v1.1.0")
    else:
        parser.print_help()


def execute_module(args):
    """Execute an NSC module."""
    try:
        module_path = args.module
        registry_path = args.registry
        
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
                "operators": [
                    {"op_id": 1001, "name": "ADD", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1002, "name": "SUB", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1003, "name": "MUL", "arg_types": ["float"], "ret_type": "float"},
                    {"op_id": 1004, "name": "DIV", "arg_types": ["float"], "ret_type": "float"},
                ]
            }
        
        # Create environment and interpreter
        env = Environment(registry=registry)
        
        # Setup default kernels
        env.kernels = {
            "kernel_add": lambda a, b: a + b,
            "kernel_sub": lambda a, b: a - b,
            "kernel_mul": lambda a, b: a * b,
            "kernel_div": lambda a, b: a / b if b != 0 else float('inf'),
        }
        
        interpreter = Interpreter(env=env)
        
        # Execute
        result = interpreter.interpret(module)
        
        # Output result
        output = {
            "success": result.success,
            "node_count": result.node_count,
            "outputs": result.outputs,
            "duration_ms": result.duration_ms,
        }
        
        print(json.dumps(output, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def validate_module(args):
    """Validate an NSC module."""
    try:
        module_path = args.module
        
        with open(module_path) as f:
            module = json.load(f)
        
        # Basic validation
        required_fields = ["module_id", "nodes", "seq", "entrypoints", "registry_ref"]
        missing = [f for f in required_fields if f not in module]
        
        if missing:
            print(f"Validation FAILED: Missing fields: {missing}")
            sys.exit(1)
        
        # Check node IDs
        node_ids = {n["id"] for n in module["nodes"]}
        for node_id in module["seq"]:
            if node_id not in node_ids:
                print(f"Validation FAILED: SEQ contains unknown node ID: {node_id}")
                sys.exit(1)
        
        print("Validation PASSED")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def decode_signal(args):
    """Decode a GLLL signal."""
    try:
        signal = json.loads(args.signal)
        n = args.n
        
        codebook = GLLLCodebook(n=n)
        result = codebook.decode(signal)
        
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def evaluate_gates(args):
    """Evaluate gate policy on residuals."""
    try:
        residuals = json.loads(args.residuals)
        
        checker = GateChecker(thresholds={
            "phys": 0.01,
            "cons": 0.02,
            "num": 0.01,
            "nan_check": True,
            "inf_check": True,
            "positivity_check": True,
        })
        
        report = checker.evaluate(residuals)
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
