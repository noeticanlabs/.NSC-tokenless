"""Gate command for NSC CLI - Gate policy evaluation."""

import json
import sys
from typing import Any, Dict, List


def gate_command(args) -> int:
    """Evaluate gate policy on residuals."""
    try:
        residuals_json = args.residuals
        
        # Parse residuals
        try:
            residuals = json.loads(residuals_json)
        except json.JSONDecodeError:
            print("Error: Invalid JSON residuals", file=sys.stderr)
            return 1
        
        # Import gate checker
        from nsc.governance.gate_checker import GateChecker, GateThresholds
        
        # Create checker with defaults
        checker = GateChecker()
        
        # Evaluate gates
        result = checker.evaluate(
            residuals.get("r_phys", 0.0),
            residuals.get("r_cons", 0.0),
            residuals.get("r_num", 0.0),
            residuals.get("nan_free", True),
            residuals.get("positive_values", True)
        )
        
        output = {
            "input": residuals,
            "gate_result": {
                "pass_hard": result.pass_hard,
                "pass_soft": result.pass_soft,
                "decision": result.decision.value if hasattr(result.decision, 'value') else result.decision,
                "hard_results": result.hard_results,
                "soft_results": result.soft_results
            }
        }
        
        if getattr(args, "json", False):
            print(json.dumps(output, indent=2))
        else:
            print(f"✓ Gate evaluation complete")
            print(f"  Decision: {output['gate_result']['decision']}")
            print(f"  Hard gates: {'PASS' if result.pass_hard else 'FAIL'}")
            print(f"  Soft gates: {'PASS' if result.pass_soft else 'FAIL'}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
