"""Decode command for NSC CLI - GLLL signal decoding."""

import json
import sys
from typing import Any, List


def decode_command(args) -> int:
    """Decode a GLLL signal to glyph identity."""
    try:
        signal_json = args.signal
        n = getattr(args, "n", 16)
        
        # Parse signal
        try:
            signal = json.loads(signal_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON signal", file=sys.stderr)
            return 1
        
        # Import GLLL components
        from nsc.glyphs import GLLLCodebook, HadamardMatrix
        
        # Create codebook and decode
        codebook = GLLLCodebook(n=n)
        hadamard = HadamardMatrix(n)
        
        # Decode each signal vector
        results = []
        if isinstance(signal, list):
            for i, vec in enumerate(signal):
                if isinstance(vec, list) and len(vec) == n:
                    glyph, confidence, decoded = codebook.decode(vec)
                    results.append({
                        "index": i,
                        "glyph": glyph if glyph != "REJECT" else None,
                        "confidence": confidence,
                        "is_reject": glyph == "REJECT"
                    })
                else:
                    results.append({
                        "index": i,
                        "error": f"Invalid vector length, expected {n}"
                    })
        
        output = {
            "n": n,
            "signal_count": len(signal) if isinstance(signal, list) else 1,
            "results": results
        }
        
        # Count rejects
        reject_count = sum(1 for r in results if r.get("is_reject"))
        output["reject_count"] = reject_count
        output["decoded_count"] = len(results) - reject_count
        
        if getattr(args, "json", False):
            print(json.dumps(output, indent=2))
        else:
            print(f"✓ Decoded {output['decoded_count']} / {output['signal_count']} signals")
            if reject_count > 0:
                print(f"  Rejected: {reject_count}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
