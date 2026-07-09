"""Schema contract query functions for the Landscape audit database.

Functions: get_run_contract, explain_field, list_contract_violations.

All functions accept (db, factory) as their first two parameters.
"""

from __future__ import annotations

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import LandscapeReadRepositories, RecorderFactory
from elspeth.mcp.types import (
    ContractViolationsReport,
    ErrorResult,
    FieldExplanation,
    FieldNotFoundError,
    RunContractReport,
)

AnalyzerRepositories = LandscapeReadRepositories | RecorderFactory


class _ContractLookupError(Exception):
    """Internal signal: the per-source contract is unavailable for ``run_id``.

    Carries an MCP-shaped ``ErrorResult`` payload so callers can return it
    directly to the MCP surface. Defined for the in-module signalling
    contract between ``_load_first_source_contract`` and the public
    ``get_run_contract`` / ``explain_field`` entry points; not raised
    outside this module.
    """

    def __init__(self, payload: ErrorResult) -> None:
        super().__init__(payload["error"])
        self.payload = payload


def _load_first_source_contract(factory: AnalyzerRepositories, run_id: str) -> SchemaContract:
    """Return the contract of the lowest-ordered source for ``run_id``.

    Per ADR-025 §3 Decision 5 (G6) schema contracts live exclusively in
    ``run_sources``; the run-level singleton was deleted because writes and
    integrity reads had drifted across surfaces. This MCP layer originally
    surfaced "the run contract" — the post-G6 equivalent is the first source's
    contract sorted by ``source_node_id`` (matching the deterministic pick
    ``RecoveryManager.verify_contract_integrity`` returns for the same
    legacy single-source view).

    Raises ``_ContractLookupError`` carrying an ``ErrorResult`` payload
    when the run has no contract stored or the per-source metadata is
    corrupt; the public entry points translate the payload back into
    the MCP-surface ``ErrorResult`` shape.
    """
    try:
        source_records = factory.run_lifecycle.get_run_source_resume_records(run_id)
    except AuditIntegrityError as exc:
        raise _ContractLookupError({"error": f"Run '{run_id}' has corrupt source-contract metadata: {exc}"}) from exc
    if not source_records:
        raise _ContractLookupError({"error": f"Run '{run_id}' has no contract stored"})
    first_source_node_id = sorted(source_records)[0]
    return source_records[first_source_node_id].schema_contract


def get_run_contract(db: LandscapeDB, factory: AnalyzerRepositories, run_id: str) -> RunContractReport | ErrorResult:
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

    Notes:
        Multi-source runs return the lowest-ordered source's contract
        (deterministic by ``source_node_id``) for back-compatibility with the
        single-contract MCP surface. Use the per-source surfaces in
        ``RunLifecycleRepository.get_run_source_resume_records`` for
        complete plural-by-source visibility.
    """
    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        contract = _load_first_source_contract(factory, run_id)
    except _ContractLookupError as exc:
        return exc.payload

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
    db: LandscapeDB, factory: AnalyzerRepositories, run_id: str, field_name: str
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

    Notes:
        Multi-source runs resolve against the lowest-ordered source's contract
        (deterministic by ``source_node_id``). For per-source field
        provenance, consult ``RunLifecycleRepository.get_run_source_resume_records``.
    """
    run = factory.run_lifecycle.get_run(run_id)
    if run is None:
        return {"error": f"Run '{run_id}' not found"}

    try:
        contract = _load_first_source_contract(factory, run_id)
    except _ContractLookupError as exc:
        return exc.payload

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
    db: LandscapeDB, factory: AnalyzerRepositories, run_id: str, limit: int = 100
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
