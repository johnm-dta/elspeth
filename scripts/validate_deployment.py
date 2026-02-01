#!/usr/bin/env python3
"""Pre-deployment validation for field normalization feature.

This script enforces atomic deployment of the field normalization feature.
Partial deployment creates AUDIT TRAIL CORRUPTION risk:
- Tasks 1-8 implement normalization
- If templates reference normalized names but normalization isn't deployed,
  templates fail silently with KeyError or empty string

Run this as part of CI/CD pipeline pre-deployment checks.

Usage:
    python scripts/validate_deployment.py

Exit codes:
    0: All components deployed together (or none deployed)
    1: Partial deployment detected - DEPLOYMENT VIOLATION
"""

from __future__ import annotations

import sys
from pathlib import Path


def _validate_field_normalization_at_path(base: Path) -> None:
    """Validate field normalization deployment at a given base path.

    This is the core validation logic, separated to enable testing
    with arbitrary paths.

    Args:
        base: Path to the elspeth package directory (e.g., src/elspeth)

    Raises:
        RuntimeError: If partial deployment detected
    """
    # Check what's deployed
    field_norm_exists = (base / "plugins/sources/field_normalization.py").exists()

    config_base_path = base / "plugins/config_base.py"
    tabular_config_exists = config_base_path.exists() and "TabularSourceDataConfig" in config_base_path.read_text()

    identifiers_exists = (base / "core/identifiers.py").exists()

    # If ANY normalization component exists, ALL must exist
    components = {
        "field_normalization.py": field_norm_exists,
        "TabularSourceDataConfig": tabular_config_exists,
        "core/identifiers.py": identifiers_exists,
    }

    deployed = {k for k, v in components.items() if v}
    missing = {k for k, v in components.items() if not v}

    if deployed and missing:
        raise RuntimeError(
            f"DEPLOYMENT VIOLATION: Partial field normalization deployment detected!\n"
            f"Deployed: {sorted(deployed)}\n"
            f"Missing: {sorted(missing)}\n\n"
            f"This creates AUDIT TRAIL CORRUPTION risk.\n"
            f"Deploy ALL components together or NONE."
        )


def validate_field_normalization_deployment() -> None:
    """Ensure atomic deployment of field normalization feature.

    Checks that if any normalization component is present,
    ALL required components are present.

    Raises:
        RuntimeError: If partial deployment detected
    """
    base = Path(__file__).parent.parent / "src" / "elspeth"
    _validate_field_normalization_at_path(base)


if __name__ == "__main__":
    try:
        validate_field_normalization_deployment()
        print("Field normalization deployment validation passed")
    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
