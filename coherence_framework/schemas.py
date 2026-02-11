"""
Coherence Framework Schemas Module

Provides utilities for accessing, loading, and validating against Coherence JSON schemas.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from jsonschema import Draft7Validator, ValidationError, validate
except ImportError:
    Draft7Validator = None
    ValidationError = None
    validate = None


def get_schema_dir() -> Path:
    """Get the path to the schemas directory."""
    return Path(__file__).parent.parent / "schemas"


def get_coherence_spec_schema_dir() -> Path:
    """Get the path to Coherence_Spec_v1_0 schemas directory."""
    return Path(__file__).parent.parent / "Coherence_Spec_v1_0" / "schemas"


def get_schema_path(schema_name: str) -> Path:
    """
    Get the full path to a schema file.

    Args:
        schema_name: Name of the schema (e.g., "omega_ledger.schema.json")

    Returns:
        Path to the schema file

    Raises:
        FileNotFoundError: If schema is not found
    """
    # Try main schemas directory first
    schema_path = get_schema_dir() / schema_name
    if schema_path.exists():
        return schema_path

    # Try Coherence_Spec_v1_0 schemas directory
    spec_schema_path = get_coherence_spec_schema_dir() / schema_name
    if spec_schema_path.exists():
        return spec_schema_path

    raise FileNotFoundError(
        f"Schema '{schema_name}' not found in {get_schema_dir()} "
        f"or {get_coherence_spec_schema_dir()}"
    )


def load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Load a schema from disk.

    Args:
        schema_name: Name of the schema file

    Returns:
        Parsed JSON schema

    Raises:
        FileNotFoundError: If schema not found
        json.JSONDecodeError: If schema is invalid JSON
    """
    schema_path = get_schema_path(schema_name)
    with open(schema_path, "r") as f:
        return json.load(f)


def validate_receipt(
    receipt: Dict[str, Any],
    schema_name: str = "omega_ledger.schema.json"
) -> bool:
    """
    Validate a receipt against a Coherence schema.

    Args:
        receipt: The receipt object to validate
        schema_name: Name of the schema to validate against

    Returns:
        True if valid

    Raises:
        jsonschema.ValidationError: If validation fails
        ImportError: If jsonschema is not installed
    """
    if validate is None:
        raise ImportError(
            "jsonschema is required for validation. "
            "Install with: pip install jsonschema"
        )

    schema = load_schema(schema_name)
    validate(instance=receipt, schema=schema)
    return True


def get_available_schemas() -> Dict[str, str]:
    """
    List all available schemas.

    Returns:
        Dict mapping schema names to their full paths
    """
    schemas = {}

    # Check main schemas directory
    for schema_file in get_schema_dir().glob("*.schema.json"):
        schemas[schema_file.name] = str(schema_file)

    # Check Coherence_Spec_v1_0 schemas directory
    for schema_file in get_coherence_spec_schema_dir().glob("*.schema.json"):
        if schema_file.name not in schemas:
            schemas[schema_file.name] = str(schema_file)

    return schemas
