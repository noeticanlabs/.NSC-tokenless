"""Replay command for NSC CLI - Receipt replay verification."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def replay_command(args) -> int:
    """Verify receipt chain through replay."""
    try:
        receipts_path = getattr(args, "receipts", None)
        module_path = getattr(args, "module", None)
        
        if not receipts_path:
            print("Error: --receipts argument required", file=sys.stderr)
            return 1
        
        receipts_file = Path(receipts_path)
        
        # Load receipts
        with open(receipts_file) as f:
            receipts = json.load(f)
        
        # Import replay verifier
        from nsc.governance.replay import ReplayVerifier
        
        verifier = ReplayVerifier()
        
        # Verify hash chaining
        chain_valid = verifier.verify_hash_chain(receipts)
        
        # Verify each receipt if module provided
        verification_details = []
        if module_path:
            with open(Path(module_path)) as f:
                module = json.load(f)
            
            for receipt in receipts:
                detail = verifier.verify_receipt(receipt, module)
                verification_details.append(detail)
        
        output = {
            "receipts_count": len(receipts),
            "hash_chain_valid": chain_valid,
            "verification_details": verification_details,
            "overall_valid": chain_valid and all(
                d.get("valid", False) for d in verification_details
            ) if verification_details else chain_valid
        }
        
        if getattr(args, "json", False):
            print(json.dumps(output, indent=2))
        else:
            print(f"✓ Replay verification complete")
            print(f"  Receipts: {output['receipts_count']}")
            print(f"  Hash chain: {'VALID' if chain_valid else 'INVALID'}")
            if verification_details:
                valid_count = sum(1 for d in verification_details if d.get("valid"))
                print(f"  Receipt verification: {valid_count}/{len(verification_details)}")
            print(f"  Overall: {'VALID' if output['overall_valid'] else 'INVALID'}")
        
        return 0 if output['overall_valid'] else 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
