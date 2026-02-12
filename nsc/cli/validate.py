"""Validate command for NSC CLI."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def validate_command(args) -> int:
    """Validate an NSC module against schemas."""
    try:
        module_path = Path(args.module)
        
        # Load module
        with open(module_path) as f:
            module = json.load(f)
        
        errors: List[str] = []
        warnings: List[str] = []
        
        # Validate module schema structure
        if "nodes" not in module:
            errors.append("Missing required field: 'nodes'")
        if "seq" not in module:
            errors.append("Missing required field: 'seq'")
        
        if errors:
            output = {
                "valid": False,
                "errors": errors,
                "warnings": warnings
            }
            print(json.dumps(output, indent=2), file=sys.stderr)
            return 1
        
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
        
        # Generate output
        is_valid = len(errors) == 0
        output = {
            "valid": is_valid,
            "module_id": module.get("module_id"),
            "node_count": len(module.get("nodes", [])),
            "seq_length": len(seq_ids),
            "entrypoints_count": len(module.get("entrypoints", [])),
            "errors": errors,
            "warnings": warnings
        }
        
        if is_valid:
            if getattr(args, "json", False):
                print(json.dumps(output, indent=2))
            else:
                print(f"✓ Module valid")
                print(f"  Module: {module.get('module_id', 'unknown')}")
                print(f"  Nodes: {output['node_count']}")
                print(f"  SEQ length: {output['seq_length']}")
        else:
            print(json.dumps(output, indent=2), file=sys.stderr)
            return 1
        
        return 0
        
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
