"""
Coherence Framework CLI Tools

Command-line interface for validating Coherence receipts and ledgers.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from coherence_framework import __version__
from coherence_framework.conformance import ConformanceChecker


def main():
    """Main entry point for coherence-validate command."""
    parser = argparse.ArgumentParser(
        prog="coherence-validate",
        description="Validate Coherence receipts and ledgers",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Receipt validation
    receipt_parser = subparsers.add_parser(
        "receipt",
        help="Validate a single receipt",
    )
    receipt_parser.add_argument(
        "receipt",
        help="Path to receipt JSON file",
    )
    receipt_parser.add_argument(
        "--schema",
        default="omega_ledger.schema.json",
        help="Schema to validate against (default: omega_ledger.schema.json)",
    )

    # Ledger validation
    ledger_parser = subparsers.add_parser(
        "ledger",
        help="Validate a ledger file",
    )
    ledger_parser.add_argument(
        "ledger",
        help="Path to ledger.jsonl file",
    )
    ledger_parser.add_argument(
        "--schema",
        default="omega_ledger.schema.json",
        help="Schema to validate against (default: omega_ledger.schema.json)",
    )
    ledger_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "receipt":
        return validate_receipt(args.receipt, args.schema)
    elif args.command == "ledger":
        return validate_ledger(args.ledger, args.schema, args.verbose)

    return 1


def validate_receipt(receipt_path: str, schema: str) -> int:
    """
    Validate a single receipt file.

    Args:
        receipt_path: Path to receipt JSON file
        schema: Schema name to validate against

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        receipt_file = Path(receipt_path)
        if not receipt_file.exists():
            print(f"✗ Receipt file not found: {receipt_path}", file=sys.stderr)
            return 1

        with open(receipt_file, 'r') as f:
            receipt = json.load(f)

        checker = ConformanceChecker(schema)
        conformant, report = checker.check_receipt(receipt)

        if conformant:
            print(f"✓ Receipt is valid and conformant")
            return 0
        else:
            print(f"✗ Receipt validation failed:")
            if report.get("schema_errors"):
                print("  Schema errors:")
                for error in report["schema_errors"]:
                    print(f"    - {error}")
            if report.get("hash_errors"):
                print("  Hash errors:")
                for error in report["hash_errors"]:
                    print(f"    - {error}")
            return 1

    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in receipt file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ Error validating receipt: {e}", file=sys.stderr)
        return 1


def validate_ledger(ledger_path: str, schema: str, verbose: bool = False) -> int:
    """
    Validate a ledger file.

    Args:
        ledger_path: Path to ledger.jsonl file
        schema: Schema name to validate against
        verbose: Print detailed output

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        checker = ConformanceChecker(schema)
        conformant, report = checker.check_ledger_file(ledger_path)

        # Print summary
        print(f"Ledger: {ledger_path}")
        print(f"  Total receipts: {report['total_receipts']}")
        print(f"  Valid schema: {report['valid_schema']}/{report['total_receipts']}")
        print(f"  Hash chain valid: {report['hash_chain_valid']}")
        print(f"  Conformant: {'✓ Yes' if conformant else '✗ No'}")

        if verbose or not conformant:
            if report.get("schema_errors"):
                print("\nSchema errors:")
                for error in report["schema_errors"][:10]:  # Limit to first 10
                    print(f"  - {error}")
                if len(report["schema_errors"]) > 10:
                    print(f"  ... and {len(report['schema_errors']) - 10} more")

            if report.get("hash_chain_errors"):
                print("\nHash chain errors:")
                for error in report["hash_chain_errors"][:10]:
                    print(f"  - {error}")
                if len(report["hash_chain_errors"]) > 10:
                    print(f"  ... and {len(report['hash_chain_errors']) - 10} more")

        return 0 if conformant else 1

    except Exception as e:
        print(f"✗ Error validating ledger: {e}", file=sys.stderr)
        return 1


def check_ledger(args=None):
    """Entry point for coherence-check command (legacy)."""
    if args is None:
        parser = argparse.ArgumentParser(
            prog="coherence-check",
            description="Check Coherence ledger conformance",
        )
        parser.add_argument("ledger", help="Path to ledger.jsonl file")
        parser.add_argument("-v", "--verbose", action="store_true")
        args = parser.parse_args()

    return validate_ledger(args.ledger, "omega_ledger.schema.json", args.verbose)


if __name__ == "__main__":
    sys.exit(main())
