"""Registry command for NSC CLI - Registry management."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def registry_command(args) -> int:
    """Registry management commands (info, validate, list)."""
    try:
        subcommand = getattr(args, "subcommand", "info")
        
        if subcommand == "list":
            return _registry_list(args)
        elif subcommand == "info":
            return _registry_info(args)
        elif subcommand == "validate":
            return _registry_validate(args)
        else:
            print(f"Error: Unknown subcommand: {subcommand}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _registry_list(args) -> int:
    """List operators in a registry."""
    registry_path = getattr(args, "registry", None)
    
    if not registry_path:
        print("Error: --registry argument required", file=sys.stderr)
        return 1
    
    with open(Path(registry_path)) as f:
        registry = json.load(f)
    
    operators = registry.get("operators", [])
    
    output = {
        "registry_id": registry.get("registry_id", "unknown"),
        "version": registry.get("version", "unknown"),
        "operator_count": len(operators)
    }
    
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
    else:
        print(f"Registry: {output['registry_id']} v{output['version']}")
        print(f"  Operators: {output['operator_count']}")
        for op in operators:
            print(f"    [{op.get('op_id')}] {op.get('name')}")
    
    return 0


def _registry_info(args) -> int:
    """Show detailed registry information."""
    registry_path = getattr(args, "registry", None)
    
    if not registry_path:
        print("Error: --registry argument required", file=sys.stderr)
        return 1
    
    with open(Path(registry_path)) as f:
        registry = json.load(f)
    
    operators = registry.get("operators", [])
    
    # Group by category
    categories = {}
    for op in operators:
        cat = op.get("category", "uncategorized")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(op)
    
    output = {
        "registry_id": registry.get("registry_id"),
        "version": registry.get("version"),
        "description": registry.get("description", ""),
        "operator_count": len(operators),
        "categories": list(categories.keys())
    }
    
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
    else:
        print(f"Registry: {output['registry_id']} v{output['version']}")
        if output['description']:
            print(f"  {output['description']}")
        print(f"  Operators: {output['operator_count']}")
        print(f"  Categories: {', '.join(categories.keys())}")
    
    return 0


def _registry_validate(args) -> int:
    """Validate a registry structure."""
    registry_path = getattr(args, "registry", None)
    
    if not registry_path:
        print("Error: --registry argument required", file=sys.stderr)
        return 1
    
    with open(Path(registry_path)) as f:
        registry = json.load(f)
    
    errors: List[str] = []
    operators = registry.get("operators", [])
    
    # Check required fields
    if "registry_id" not in registry:
        errors.append("Missing 'registry_id'")
    if "operators" not in registry:
        errors.append("Missing 'operators' array")
    
    # Check each operator
    op_ids = set()
    for i, op in enumerate(operators):
        if "op_id" not in op:
            errors.append(f"Operator {i}: missing 'op_id'")
        else:
            if op["op_id"] in op_ids:
                errors.append(f"Duplicate op_id: {op['op_id']}")
            op_ids.add(op["op_id"])
        
        if "name" not in op:
            errors.append(f"Operator {i}: missing 'name'")
    
    output = {
        "valid": len(errors) == 0,
        "registry_id": registry.get("registry_id"),
        "operator_count": len(operators),
        "errors": errors
    }
    
    if output["valid"]:
        if getattr(args, "json", False):
            print(json.dumps(output, indent=2))
        else:
            print(f"✓ Registry valid: {output['registry_id']}")
            print(f"  Operators: {output['operator_count']}")
    else:
        print(json.dumps(output, indent=2), file=sys.stderr)
        return 1
    
    return 0
