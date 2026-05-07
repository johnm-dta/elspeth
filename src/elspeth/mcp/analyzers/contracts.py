"""Schema contract query functions for the Landscape audit database.

Functions: get_run_contract, explain_field, list_contract_violations.

All functions accept (db, factory) as their first two parameters.
"""

from __future__ import annotations

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.mcp.types import (
    ContractViolationsReport,
    ErrorResult,
    FieldExplanation,
    FieldNotFoundError,
    RunContractReport,
)


def get_run_contract(db: LandscapeDB, factory: RecorderFactory, run_id: str) -> RunContractReport | ErrorResult:
    """Get schema contract for a run.

    Shows the source schema contract with field resolution:
    - Mode (FIXED/FLEXIBLE/OBSERVED)
    - Field mappings (original -> normalized)
    - Inferred types

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query

    Returns:
        Contract details or {"error": "..."} if not found
    """
    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    contract = factory.run_lifecycle.get_run_contract(run_id)
    if contract is None:
        return {"error": f"Run '{run_id}' has no contract stored"}

    # Convert contract to JSON-serializable format
    fields = [
        {
            "normalized_name": f.normalized_name,
            "original_name": f.original_name,
            "python_type": f.python_type.__name__,
            "required": f.required,
            "nullable": f.nullable,
            "source": f.source,
        }
        for f in contract.fields
    ]

    return {
        "run_id": run_id,
        "mode": contract.mode,
        "locked": contract.locked,
        "fields": fields,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "field_count": len(fields),
        "version_hash": contract.version_hash(),
    }


def explain_field(
    db: LandscapeDB, factory: RecorderFactory, run_id: str, field_name: str
) -> FieldExplanation | ErrorResult | FieldNotFoundError:
    """Trace a field's provenance through the pipeline.

    Shows how a field was:
    - Named at source (original)
    - Normalized (to Python identifier)
    - Typed (inferred or declared)

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        field_name: Either normalized or original name

    Returns:
        Field provenance details or {"error": "..."} if not found
    """
    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    contract = factory.run_lifecycle.get_run_contract(run_id)
    if contract is None:
        return {"error": f"Run '{run_id}' has no contract stored"}

    # Resolve field using canonical name resolution (normalized_name takes precedence)
    normalized = contract.find_name(field_name)
    if normalized is not None:
        field_contract = contract.get_field(normalized)
    else:
        field_contract = None

    if field_contract is None:
        available_fields = [f.normalized_name for f in contract.fields]
        return {
            "error": f"Field '{field_name}' not found in contract",
            "available_fields": available_fields,
        }

    return {
        "run_id": run_id,
        "normalized_name": field_contract.normalized_name,
        "original_name": field_contract.original_name,
        "python_type": field_contract.python_type.__name__,
        "required": field_contract.required,
        "nullable": field_contract.nullable,
        "source": field_contract.source,
        "contract_mode": contract.mode,
    }


def list_contract_violations(
    db: LandscapeDB, factory: RecorderFactory, run_id: str, limit: int = 100
) -> ContractViolationsReport | ErrorResult:
    """List contract violations for a run.

    Shows validation errors with contract details:
    - Violation type (type_mismatch, missing_field, extra_field)
    - Field names (original and normalized)
    - Type information (expected vs actual)

    Args:
        db: Database connection
        factory: Recorder factory
        run_id: Run ID to query
        limit: Maximum violations to return (default 100)

    Returns:
        List of violations or {"error": "..."} if run not found
    """
    from sqlalchemy import func, select

    from elspeth.core.landscape.schema import validation_errors_table

    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    with db.connection() as conn:
        # Count total violations (those with violation_type set)
        total_count = (
            conn.execute(
                select(func.count())
                .select_from(validation_errors_table)
                .where((validation_errors_table.c.run_id == run_id) & (validation_errors_table.c.violation_type.isnot(None)))
            ).scalar()
            or 0
        )

        # Get violations with details
        query = (
            select(
                validation_errors_table.c.error_id,
                validation_errors_table.c.violation_type,
                validation_errors_table.c.normalized_field_name,
                validation_errors_table.c.original_field_name,
                validation_errors_table.c.expected_type,
                validation_errors_table.c.actual_type,
                validation_errors_table.c.error,
                validation_errors_table.c.schema_mode,
                validation_errors_table.c.destination,
                validation_errors_table.c.created_at,
            )
            .where((validation_errors_table.c.run_id == run_id) & (validation_errors_table.c.violation_type.isnot(None)))
            .order_by(validation_errors_table.c.created_at.desc())
            .limit(limit)
        )
        rows = conn.execute(query).fetchall()

    violations = [
        {
            "error_id": row.error_id,
            "violation_type": row.violation_type,
            "normalized_field_name": row.normalized_field_name,
            "original_field_name": row.original_field_name,
            "expected_type": row.expected_type,
            "actual_type": row.actual_type,
            "error": row.error,
            "schema_mode": row.schema_mode,
            "destination": row.destination,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]

    return {
        "run_id": run_id,
        "total_violations": total_count,
        "violations": violations,  # type: ignore[typeddict-item]  # structurally correct dict literals
        "limit": limit,
    }
