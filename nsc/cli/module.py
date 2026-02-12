"""Module command for NSC CLI - Module management."""

import json
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List


def module_command(args) -> int:
    """Module management commands (digest, link, info)."""
    try:
        subcommand = getattr(args, "subcommand", "info")
        module_path = Path(args.module)
        
        # Load module
        with open(module_path) as f:
            module = json.load(f)
        
        if subcommand == "info":
            return _module_info(module, args)
        elif subcommand == "digest":
            return _module_digest(module, args)
        elif subcommand == "link":
            return _module_link(module, args)
        else:
            print(f"Error: Unknown subcommand: {subcommand}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _module_info(module: Dict, args) -> int:
    """Show module information."""
    nodes = module.get("nodes", [])
    seq = module.get("seq", [])
    entrypoints = module.get("entrypoints", [])
    
    # Count node types
    kind_counts = {}
    for node in nodes:
        kind = node.get("kind", "UNKNOWN")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    
    output = {
        "module_id": module.get("module_id", "unknown"),
        "version": module.get("version", "unknown"),
        "node_count": len(nodes),
        "seq_length": len(seq),
        "entrypoints": entrypoints,
        "node_types": kind_counts,
        "registry_ref": module.get("registry_ref", "default")
    }
    
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
    else:
        print(f"Module: {output['module_id']}")
        print(f"  Version: {output['version']}")
        print(f"  Nodes: {output['node_count']}")
        print(f"  SEQ length: {output['seq_length']}")
        print(f"  Entrypoints: {len(entrypoints)}")
        print(f"  Node types: {kind_counts}")
    
    return 0


def _module_digest(module: Dict, args) -> int:
    """Compute and show module digest."""
    # Sort nodes by ID for deterministic digest
    nodes = sorted(module.get("nodes", []), key=lambda n: n.get("id", 0))
    
    # Create canonical representation
    canonical = {
        "module_id": module.get("module_id"),
        "version": module.get("version"),
        "nodes": nodes,
        "seq": module.get("seq", []),
        "entrypoints": module.get("entrypoints", []),
        "registry_ref": module.get("registry_ref")
    }
    
    # Compute SHA256 digest
    canonical_json = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = "sha256:" + hashlib.sha256(canonical_json.encode()).hexdigest()
    
    output = {
        "module_id": module.get("module_id"),
        "digest": digest,
        "canonical_size": len(canonical_json)
    }
    
    if getattr(args, "json", False):
        print(json.dumps(output, indent=2))
    else:
        print(f"Module: {output['module_id']}")
        print(f"  Digest: {digest}")
    
    return 0


def _module_link(module: Dict, args) -> int:
    """Link modules together (stub for future implementation)."""
    print("Error: Module linking not yet implemented", file=sys.stderr)
    return 1
