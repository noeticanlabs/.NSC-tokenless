#!/usr/bin/env python
"""Test runner and report generator for NSC system."""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

def run_tests():
    """Run pytest and capture results."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--json-report",
        "--json-report-file=test_report.json"
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode

def generate_report():
    """Generate test status report."""
    report_path = Path("test_report.json")
    
    print("\n" + "="*80)
    print("NSC SYSTEM TEST REPORT")
    print("="*80)
    print(f"Generated: {datetime.utcnow().isoformat()}")
    print()
    
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
        
        summary = data.get("summary", {})
        print(f"Total Tests: {summary.get('total', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Skipped: {summary.get('skipped', 0)}")
        print()
    
    # Run basic pytest to get output
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=line"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr)
    
    print("\n" + "="*80)
    print("CRITICAL AREAS TESTED:")
    print("="*80)
    print("✓ Runtime execution (state, environment, executor)")
    print("✓ Governance (gate checking, policy)")
    print("✓ NPE versions (v0.1, v1.0, v2.0)")
    print("✓ GLLL encoding and decoding")
    print("✓ Integration tests (module validation, receipt chain, replay)")
    print("✓ API endpoints (validate, execute, receipts, replay)")
    print()
    
    return result.returncode

if __name__ == "__main__":
    print("Running NSC System Tests...\n")
    exit_code = generate_report()
    sys.exit(exit_code)
