"""
Coherence Framework Conformance Module

Provides validators for coherence receipts and ledgers.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from jsonschema import ValidationError, validate
except ImportError:
    ValidationError = Exception
    validate = None

from coherence_framework.schemas import load_schema


class SchemaValidator:
    """Validates receipts against JSON schemas."""

    def __init__(self, schema_name: str = "omega_ledger.schema.json"):
        """
        Initialize validator with a schema.

        Args:
            schema_name: Name of the schema to use
        """
        self.schema_name = schema_name
        self.schema = load_schema(schema_name)

    def validate(self, receipt: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a receipt against the schema.

        Args:
            receipt: Receipt object to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        if validate is None:
            return False, ["jsonschema not installed"]

        try:
            validate(instance=receipt, schema=self.schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]


class HashChainValidator:
    """Validates hash chain integrity of ledgers."""

    @staticmethod
    def validate_receipt_hash(receipt: Dict[str, Any], expected_hash: str = None) -> Tuple[bool, str]:
        """
        Validate the hash of a single receipt.

        Args:
            receipt: Receipt object
            expected_hash: Expected hash value (optional)

        Returns:
            Tuple of (is_valid, message)
        """
        if "receipt_hash" not in receipt:
            return False, "Missing receipt_hash field"

        actual_hash = receipt.get("receipt_hash", "")

        if expected_hash and actual_hash != expected_hash:
            return False, f"Hash mismatch: expected {expected_hash}, got {actual_hash}"

        if not all(c in "0123456789abcdef" for c in actual_hash):
            return False, f"Invalid hash format (not hex): {actual_hash}"

        if len(actual_hash) != 64:
            return False, f"Invalid hash length (expected 64, got {len(actual_hash)})"

        return True, "Valid"

    @staticmethod
    def validate_chain(receipts: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate hash chain across multiple receipts.

        Args:
            receipts: List of receipt objects in order

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        prev_hash = None

        for i, receipt in enumerate(receipts):
            # Validate current receipt hash
            valid, msg = HashChainValidator.validate_receipt_hash(receipt)
            if not valid:
                errors.append(f"Receipt {i}: {msg}")
                continue

            # Check chain link
            parent_hash = receipt.get("parent_hash")
            if i == 0:
                # Genesis block should have null parent
                if parent_hash is not None:
                    errors.append(f"Receipt {i}: Genesis block should have null parent_hash")
            else:
                # Non-genesis should link to previous
                if parent_hash != prev_hash:
                    errors.append(
                        f"Receipt {i}: Chain break - parent_hash {parent_hash} "
                        f"doesn't match previous receipt hash {prev_hash}"
                    )

            prev_hash = receipt.get("receipt_hash")

        return len(errors) == 0, errors


class ConformanceChecker:
    """Checks full conformance with Coherence specification."""

    def __init__(self, schema_name: str = "omega_ledger.schema.json"):
        """
        Initialize conformance checker.

        Args:
            schema_name: Name of schema to validate against
        """
        self.schema_validator = SchemaValidator(schema_name)
        self.hash_validator = HashChainValidator()

    def check_ledger_file(self, ledger_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check a ledger.jsonl file for full conformance.

        Args:
            ledger_path: Path to ledger file

        Returns:
            Tuple of (is_conformant, report)
        """
        report = {
            "conformant": False,
            "total_receipts": 0,
            "valid_schema": 0,
            "schema_errors": [],
            "hash_chain_valid": False,
            "hash_chain_errors": [],
            "warnings": [],
        }

        try:
            ledger_file = Path(ledger_path)
            if not ledger_file.exists():
                report["schema_errors"].append(f"Ledger file not found: {ledger_path}")
                return False, report

            receipts = []
            with open(ledger_file, 'r') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue

                    try:
                        receipt = json.loads(line)
                        receipts.append(receipt)
                        report["total_receipts"] += 1

                        # Validate schema
                        valid, errors = self.schema_validator.validate(receipt)
                        if valid:
                            report["valid_schema"] += 1
                        else:
                            report["schema_errors"].extend(
                                [f"Receipt {i}: {e}" for e in errors]
                            )
                    except json.JSONDecodeError as e:
                        report["schema_errors"].append(f"Line {i}: Invalid JSON - {e}")

            # Validate hash chain
            if receipts:
                valid, errors = self.hash_validator.validate_chain(receipts)
                report["hash_chain_valid"] = valid
                report["hash_chain_errors"] = errors

            # Overall conformance
            report["conformant"] = (
                len(report["schema_errors"]) == 0 and
                report["hash_chain_valid"] and
                report["valid_schema"] == report["total_receipts"]
            )

            return report["conformant"], report

        except Exception as e:
            report["schema_errors"].append(f"Error checking ledger: {str(e)}")
            return False, report

    def check_receipt(self, receipt: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check a single receipt for conformance.

        Args:
            receipt: Receipt object

        Returns:
            Tuple of (is_conformant, report)
        """
        report = {
            "conformant": False,
            "schema_valid": False,
            "schema_errors": [],
            "hash_valid": False,
            "hash_errors": [],
        }

        # Schema validation
        valid, errors = self.schema_validator.validate(receipt)
        report["schema_valid"] = valid
        report["schema_errors"] = errors

        # Hash validation
        valid, msg = self.hash_validator.validate_receipt_hash(receipt)
        report["hash_valid"] = valid
        if not valid:
            report["hash_errors"] = [msg]

        report["conformant"] = report["schema_valid"] and report["hash_valid"]

        return report["conformant"], report
