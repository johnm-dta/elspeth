"""Run lifecycle repository for Landscape audit records.

Owns all run lifecycle operations: begin, complete, finalize, status updates,
schema contracts, secret resolutions, export status, and reproducibility grading.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts import (
    ContractAuditRecord,
    ExportStatus,
    NodeType,
    ReproducibilityGrade,
    Run,
    RunStatus,
    SecretResolution,
    SecretResolutionInput,
)
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.dependency_config import PreflightResult
from elspeth.core.landscape._database_ops import DatabaseOps
from elspeth.core.landscape._helpers import generate_id, now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.model_loaders import RunLoader
from elspeth.core.landscape.reproducibility import compute_grade
from elspeth.core.landscape.schema import (
    RunSourceLifecycleState,
    nodes_table,
    preflight_results_table,
    run_attributions_table,
    run_sources_table,
    runs_table,
    secret_resolutions_table,
)

if TYPE_CHECKING:
    from elspeth.contracts.schema_contract import SchemaContract


@dataclass(frozen=True, slots=True)
class RunSourceResumeRecord:
    """Per-source audit metadata required to replay rows during resume."""

    source_node_id: str
    source_name: str
    lifecycle_state: str
    source_schema_json: str
    schema_contract: SchemaContract


@dataclass(frozen=True, slots=True)
class RunSourceLifecycleRecord:
    """Per-source lifecycle metadata used before resume replay reconstruction."""

    source_node_id: str
    source_name: str
    lifecycle_state: str


@dataclass(frozen=True, slots=True)
class RunSourceFieldResolutionRecord:
    """Per-source source header resolution metadata."""

    source_node_id: str
    source_name: str
    resolution_mapping: Mapping[str, str] | None

    def __post_init__(self) -> None:
        # Frozen-dataclass deep-freeze contract (CLAUDE.md): resolution_mapping
        # is a container field, so frozen=True alone leaves its contents mutable
        # through the attribute reference. Gate on `is not None` — the field is
        # nullable (sources that resolved no headers record None).
        if self.resolution_mapping is not None:
            freeze_fields(self, "resolution_mapping")


# Phase 2.2 (elspeth-0de989c56d): COMPLETED_WITH_FAILURES and EMPTY join the
# terminal set so the engine can finalize runs into the four-value taxonomy
# without a separate "intermediate" lifecycle hop.  The same terminal-write
# guarantees apply: completed_at is set, the row becomes immutable, and
# update_run_status() can no longer overwrite it.
_TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_FAILURES,
        RunStatus.FAILED,
        RunStatus.EMPTY,
        RunStatus.INTERRUPTED,
    }
)
_IMMUTABLE_SUCCESS_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.COMPLETED_WITH_FAILURES,
        RunStatus.EMPTY,
    }
)
_IMMUTABLE_SUCCESS_RUN_STATUS_VALUES = tuple(status.value for status in _IMMUTABLE_SUCCESS_RUN_STATUSES)

_AUTH_PROVIDER_TYPES = frozenset({"local", "oidc", "entra"})
_OPENROUTER_CATALOG_SOURCES = frozenset({"live", "bundled"})

# 64 lowercase hex chars — matches the canonical sha256 hex digest format
# produced by ``hashlib.sha256(...).hexdigest()``. Used by the Tier-1
# write-side guards in this module, ``write_repository.py``, and
# ``web/execution/service.py`` to reject malformed snapshot ids that
# would otherwise corrupt the audit trail without a downstream signal.
_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


def is_valid_sha256_hex(value: str) -> bool:
    """Return True if ``value`` is exactly 64 lowercase hex chars.

    Canonical home for the sha256-hex shape check; imported by
    ``write_repository.py`` and ``web/execution/service.py`` so all three
    write-side guards reject the same out-of-domain values (empty,
    whitespace-only, non-hex strings, upper-case, wrong length).
    """
    return _SHA256_HEX_RE.fullmatch(value) is not None


def validate_run_attribution(*, initiated_by_user_id: str | None, auth_provider_type: str | None) -> None:
    if initiated_by_user_id is None and auth_provider_type is None:
        return
    if type(initiated_by_user_id) is not str or not initiated_by_user_id.strip():
        raise AuditIntegrityError("run attribution requires a non-blank initiated_by_user_id")
    if auth_provider_type not in _AUTH_PROVIDER_TYPES:
        raise AuditIntegrityError(f"run attribution requires auth_provider_type to be one of {sorted(_AUTH_PROVIDER_TYPES)!r}")


def _validate_openrouter_catalog_snapshot(*, sha256: str, source: str) -> None:
    """Tier-1 write-side guard for the OpenRouter catalog snapshot fields.

    Both fields are NOT NULL on the ``runs`` table and define the
    audit-trail anchor for "which catalog blessed this run's model
    decisions". A defective caller passing an empty string or an
    out-of-domain source value would silently corrupt the audit record;
    catch it here so the failure points at the wiring bug.
    """
    if type(sha256) is not str or not is_valid_sha256_hex(sha256):
        raise AuditIntegrityError(f"openrouter_catalog_sha256 must be 64 lowercase hex chars, got {sha256!r}")
    if source not in _OPENROUTER_CATALOG_SOURCES:
        raise AuditIntegrityError(f"openrouter_catalog_source must be one of {sorted(_OPENROUTER_CATALOG_SOURCES)!r}, got {source!r}")


class RunLifecycleRepository:
    """Run lifecycle operations for the Landscape audit trail.

    Handles: begin, complete, finalize, status updates, schema contracts,
    secret resolutions, export status, and reproducibility grading.
    """

    def __init__(self, db: LandscapeDB, ops: DatabaseOps, run_loader: RunLoader) -> None:
        self._db = db
        self._ops = ops
        self._run_loader = run_loader

    def begin_run(
        self,
        config: Mapping[str, Any],
        canonical_version: str,
        *,
        run_id: str | None = None,
        reproducibility_grade: ReproducibilityGrade | None = None,
        status: RunStatus = RunStatus.RUNNING,
        source_schema_json: str | None = None,
        initiated_by_user_id: str | None = None,
        auth_provider_type: str | None = None,
        openrouter_catalog_sha256: str,
        openrouter_catalog_source: str,
    ) -> Run:
        """Begin a new pipeline run.

        Args:
            config: Resolved configuration dictionary
            canonical_version: Version of canonical hash algorithm
            run_id: Optional run ID (generated if not provided)
            reproducibility_grade: Optional grade (FULL_REPRODUCIBLE, etc.)
            status: Initial RunStatus (defaults to RUNNING)
            source_schema_json: Optional serialized source schema for resume type restoration.
                Should be Pydantic model_json_schema() output. Required for proper resume
                type fidelity (datetime/Decimal restoration from payload JSON strings).
            initiated_by_user_id: Optional authenticated user ID that initiated the run.
            auth_provider_type: Optional auth provider namespace for the initiating user.
            openrouter_catalog_sha256: Canonical sha256 of the OpenRouter
                model catalog active at run-create time. Required (Tier-1
                audit completeness): every run records which catalog
                blessed its model decisions. Resolved at the L3 entry
                point via :func:`elspeth.plugins.transforms.llm.model_catalog.read_openrouter_catalog_snapshot_id`.
            openrouter_catalog_source: ``"live"`` if the lifespan probed
                OpenRouter successfully, ``"bundled"`` if the fallback
                served the snapshot. Required (NOT NULL on the column).

        Returns:
            Run model with generated run_id

        Notes:
            Per ADR-025 §3 Decision 5 (G6) schema contracts are stored
            per-source in the ``run_sources`` table — written by
            ``record_run_source`` / ``update_run_source_contract`` as each
            source iterates. The run-level singleton (``runs.schema_contract_json``
            column and the matching ``schema_contract`` parameter) was deleted
            because writers and readers had drifted: writers populated the
            singleton, integrity verification consulted ``run_sources``, and
            the audit trail recorded contracts in one surface while resume
            validated against another.
        """
        if status == RunStatus.COMPLETED:
            raise AuditIntegrityError(
                "begin_run() cannot create a COMPLETED run. Use complete_run() so completed_at is recorded in the audit trail."
            )
        validate_run_attribution(initiated_by_user_id=initiated_by_user_id, auth_provider_type=auth_provider_type)
        _validate_openrouter_catalog_snapshot(
            sha256=openrouter_catalog_sha256,
            source=openrouter_catalog_source,
        )

        run_id = run_id or generate_id()
        settings_json = canonical_json(config)
        config_hash = stable_hash(config)
        timestamp = now()

        # ADR-010 §Decision 3 M3: record the declaration +
        # Tier-1 registries that were in force at run start. Canonicalised
        # JSON so the value is stable across Python invocations and suitable
        # for hash-based cross-run regression detection. Must run AFTER the
        # orchestrator has frozen both registries; begin_run is called from
        # Orchestrator.run() which sequences prepare_for_run() (which
        # freezes) before this call.
        runtime_val_manifest_json = canonical_json(build_runtime_val_manifest())

        run = Run(
            run_id=run_id,
            started_at=timestamp,
            config_hash=config_hash,
            settings_json=settings_json,
            canonical_version=canonical_version,
            status=status,
            reproducibility_grade=reproducibility_grade,
        )

        self._ops.execute_insert(
            runs_table.insert().values(
                run_id=run.run_id,
                started_at=run.started_at,
                config_hash=run.config_hash,
                settings_json=run.settings_json,
                canonical_version=run.canonical_version,
                status=run.status.value,
                reproducibility_grade=run.reproducibility_grade,
                source_schema_json=source_schema_json,
                runtime_val_manifest_json=runtime_val_manifest_json,
                llm_call_count=None,
                seeded_from_cache=False,
                cache_key=None,
                openrouter_catalog_sha256=openrouter_catalog_sha256,
                openrouter_catalog_source=openrouter_catalog_source,
            )
        )
        if initiated_by_user_id is not None and auth_provider_type is not None:
            self._ops.execute_insert(
                run_attributions_table.insert().values(
                    run_id=run.run_id,
                    recorded_at=timestamp,
                    initiated_by_user_id=initiated_by_user_id,
                    auth_provider_type=auth_provider_type,
                ),
                context="run_attributions",
            )

        return run

    def complete_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        reproducibility_grade: ReproducibilityGrade | None = None,
    ) -> Run:
        """Complete a pipeline run.

        Args:
            run_id: Run to complete
            status: Final RunStatus (COMPLETED, FAILED, or INTERRUPTED)
            reproducibility_grade: Optional final grade. When None, preserves
                any grade already stored on the run (e.g., from begin_run).

        Returns:
            Updated Run model

        Raises:
            AuditIntegrityError: If status is not a terminal run status
            AuditIntegrityError: If run_id not found (via execute_update zero-rows check)
        """
        if status not in _TERMINAL_RUN_STATUSES:
            raise AuditIntegrityError(
                f"complete_run() requires terminal status, got {status.value!r}. "
                f"Valid terminal statuses: {sorted(s.value for s in _TERMINAL_RUN_STATUSES)}"
            )

        timestamp = now()

        # Only include reproducibility_grade in UPDATE when explicitly provided.
        # Passing None would overwrite an existing grade with NULL (Bug 318f74).
        values: dict[str, Any] = {
            "status": status.value,
            "completed_at": timestamp,
        }
        if reproducibility_grade is not None:
            values["reproducibility_grade"] = reproducibility_grade

        # Atomic conditional UPDATE: only succeeds when current status is NOT
        # already terminal.  Once a run reaches COMPLETED/FAILED/INTERRUPTED,
        # its terminal status and completed_at are the legal record and must
        # not be overwritten (Bug 3c77199a70).  The resume path transitions
        # FAILED/INTERRUPTED → RUNNING via update_run_status() first, so by the
        # time complete_run() is called the status is RUNNING.
        _terminal_values = [s.value for s in _TERMINAL_RUN_STATUSES]
        with self._db.write_connection() as conn:
            result = conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .where(runs_table.c.status.notin_(_terminal_values))
                .values(**values)
            )
            if result.rowcount == 0:
                existing = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
                if existing is not None and existing.status in _terminal_values:
                    raise AuditIntegrityError(
                        f"Cannot complete run {run_id}: already terminal "
                        f"(status={existing.status!r}). Terminal runs are immutable — "
                        f"the audit record's status and completed_at timestamp cannot "
                        f"be overwritten. Resume path must transition to RUNNING via "
                        f"update_run_status() before re-completing."
                    )
                raise AuditIntegrityError(f"Cannot complete run {run_id}: run not found")

        run = self.get_run(run_id)
        if run is None:
            raise AuditIntegrityError(f"Run {run_id} not found after UPDATE - database corruption or transaction failure")
        return run

    def get_run(self, run_id: str) -> Run | None:
        """Get a run by ID.

        Args:
            run_id: Run ID to retrieve

        Returns:
            Run model or None if not found
        """
        query = select(runs_table).where(runs_table.c.run_id == run_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        return self._run_loader.load(row)

    def get_run_attribution(self, run_id: str) -> tuple[str, str] | None:
        """Return ``(initiated_by_user_id, auth_provider_type)`` for a run if present."""
        query = select(
            run_attributions_table.c.initiated_by_user_id,
            run_attributions_table.c.auth_provider_type,
        ).where(run_attributions_table.c.run_id == run_id)
        row = self._ops.execute_fetchone(query)
        if row is None:
            return None
        initiated_by_user_id = row.initiated_by_user_id
        auth_provider_type = row.auth_provider_type
        validate_run_attribution(initiated_by_user_id=initiated_by_user_id, auth_provider_type=auth_provider_type)
        return initiated_by_user_id, auth_provider_type

    def get_source_schema(self, run_id: str) -> str:
        """Get source schema JSON for a run (for resume/type restoration).

        Args:
            run_id: Run to query

        Returns:
            Source schema JSON string

        Raises:
            AuditIntegrityError: If run not found or has no source schema

        Note:
            This encapsulates Landscape schema access for Orchestrator resume.
            Schema is required for type fidelity when restoring rows from payloads.
        """
        query = select(runs_table.c.source_schema_json).where(runs_table.c.run_id == run_id)
        run_row = self._ops.execute_fetchone(query)

        if run_row is None:
            raise AuditIntegrityError(f"Run {run_id} not found in database")

        source_schema_json = run_row.source_schema_json
        if source_schema_json is None:
            raise AuditIntegrityError(
                f"Run {run_id} has no source schema stored. "
                f"This run was created before source schema storage was implemented. "
                f"Cannot resume without schema - type fidelity would be violated."
            )

        if type(source_schema_json) is not str:
            raise AuditIntegrityError(
                f"Run {run_id} source_schema_json is {type(source_schema_json).__name__}, "
                f"expected str — audit data corruption (Tier 1 violation)"
            )
        return source_schema_json

    def get_runtime_val_manifest(self, run_id: str) -> str:
        """Get the runtime-VAL manifest captured at run start."""
        query = select(runs_table.c.runtime_val_manifest_json).where(runs_table.c.run_id == run_id)
        run_row = self._ops.execute_fetchone(query)

        if run_row is None:
            raise AuditIntegrityError(f"Run {run_id} not found in database")

        runtime_val_manifest_json = run_row.runtime_val_manifest_json
        if runtime_val_manifest_json is None:
            raise AuditIntegrityError(
                f"Run {run_id} has no runtime VAL manifest stored. "
                "Resume cannot verify declaration-contract or Tier-1 registry parity "
                "without this evidence — audit trail is incomplete."
            )

        if type(runtime_val_manifest_json) is not str:
            raise AuditIntegrityError(
                f"Run {run_id} runtime_val_manifest_json is {type(runtime_val_manifest_json).__name__}, "
                "expected str — audit data corruption (Tier 1 violation)"
            )
        return runtime_val_manifest_json

    def record_source_field_resolution(
        self,
        run_id: str,
        resolution_mapping: Mapping[str, str],
        normalization_version: str | None,
    ) -> None:
        """Record field resolution mapping computed during source.load().

        This captures the mapping from original header names (as read from the file)
        to final field names (after normalization and/or field_mapping applied).
        Must be called after source.load() completes but before processing begins.

        Args:
            run_id: Run to update
            resolution_mapping: Dict mapping original header name → final field name
            normalization_version: Algorithm version used for normalization, or None if
                                   no normalization was applied (passthrough or explicit columns)

        Note:
            This is necessary because field resolution depends on actual file headers
            which are only known after load() runs, but node config is registered
            before load(). Without this, audit trail cannot recover original headers.
        """
        resolution_data = {
            "resolution_mapping": resolution_mapping,
            "normalization_version": normalization_version,
        }
        resolution_json = canonical_json(resolution_data)

        try:
            self._ops.execute_update(
                runs_table.update().where(runs_table.c.run_id == run_id).values(source_field_resolution_json=resolution_json)
            )
        except AuditIntegrityError:
            raise  # Preserve original error message from execute_update

    def record_run_source(
        self,
        *,
        run_id: str,
        source_node_id: str,
        source_name: str,
        plugin_name: str,
        config_hash: str,
        lifecycle_state: str | RunSourceLifecycleState,
        source_schema_json: str | None = None,
        schema_contract: SchemaContract | None = None,
        field_resolution_mapping: Mapping[str, str] | None = None,
        normalization_version: str | None = None,
    ) -> None:
        """Record per-source run metadata keyed by source node.

        Multi-source runs cannot honestly store schema, field resolution, or
        lifecycle as run-level singleton columns. This table preserves the
        configuration source name and source node identity for audit queries.
        """
        source_node = self._ops.execute_fetchone(
            select(nodes_table.c.node_type).where(nodes_table.c.run_id == run_id).where(nodes_table.c.node_id == source_node_id)
        )
        if source_node is None:
            raise AuditIntegrityError(
                f"run_sources source_node_id={source_node_id!r} does not exist for run_id={run_id!r}; "
                "per-source resume metadata must reference a registered graph node."
            )
        if source_node.node_type != NodeType.SOURCE.value:
            raise AuditIntegrityError(
                f"run_sources source_node_id={source_node_id!r} for run_id={run_id!r} references "
                f"node_type={source_node.node_type!r}; expected {NodeType.SOURCE.value!r}."
            )
        try:
            lifecycle = RunSourceLifecycleState(lifecycle_state)
        except ValueError as exc:
            allowed = ", ".join(state.value for state in RunSourceLifecycleState)
            raise AuditIntegrityError(f"Invalid run source lifecycle_state={lifecycle_state!r}; expected one of: {allowed}.") from exc

        schema_contract_json: str | None = None
        schema_contract_hash: str | None = None
        if schema_contract is not None:
            audit_record = ContractAuditRecord.from_contract(schema_contract)
            schema_contract_json = audit_record.to_json()
            schema_contract_hash = schema_contract.version_hash()

        field_resolution_json: str | None = None
        if field_resolution_mapping is not None:
            field_resolution_json = canonical_json(
                {
                    "resolution_mapping": field_resolution_mapping,
                    "normalization_version": normalization_version,
                }
            )

        values = {
            "source_name": source_name,
            "plugin_name": plugin_name,
            "config_hash": config_hash,
            "schema_json": source_schema_json,
            "schema_contract_json": schema_contract_json,
            "schema_contract_hash": schema_contract_hash,
            "field_resolution_json": field_resolution_json,
            "lifecycle_state": lifecycle.value,
            "recorded_at": now(),
        }
        existing = self._ops.execute_fetchone(
            select(run_sources_table.c.source_node_id)
            .where(run_sources_table.c.run_id == run_id)
            .where(run_sources_table.c.source_node_id == source_node_id)
        )
        if existing is None:
            self._ops.execute_insert(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id=source_node_id,
                    **values,
                ),
                context="run_sources",
            )
            return

        self._ops.execute_update(
            run_sources_table.update()
            .where(run_sources_table.c.run_id == run_id)
            .where(run_sources_table.c.source_node_id == source_node_id)
            .values(**values),
            context="run_sources",
        )

    def update_run_source_contract(
        self,
        *,
        run_id: str,
        source_node_id: str,
        schema_contract: SchemaContract,
    ) -> None:
        """Persist a first-row-inferred schema contract for one source.

        ``record_run_source(..., lifecycle_state="loading")`` runs before a
        generator source has yielded its first valid row. Observed/flexible
        sources can only lock their contract at that first row, so the
        orchestrator must backfill the matching ``run_sources`` row before row
        processing can fail. Missing rows or mismatched existing contracts are
        audit corruption, not resume-time data quality issues.
        """
        audit_record = ContractAuditRecord.from_contract(schema_contract)
        schema_contract_json = audit_record.to_json()
        schema_contract_hash = schema_contract.version_hash()

        with self._db.write_connection() as conn:
            result = conn.execute(
                run_sources_table.update()
                .where(run_sources_table.c.run_id == run_id)
                .where(run_sources_table.c.source_node_id == source_node_id)
                .where(run_sources_table.c.schema_contract_json.is_(None))
                .values(
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    recorded_at=now(),
                )
            )
            if result.rowcount == 1:
                return

            existing = conn.execute(
                select(
                    run_sources_table.c.schema_contract_json,
                    run_sources_table.c.schema_contract_hash,
                )
                .where(run_sources_table.c.run_id == run_id)
                .where(run_sources_table.c.source_node_id == source_node_id)
            ).fetchone()
            if existing is None:
                raise AuditIntegrityError(
                    f"Cannot update source schema contract for run {run_id} source_node_id={source_node_id!r}: "
                    "run_sources row does not exist."
                )
            if existing.schema_contract_json is None:
                raise AuditIntegrityError(
                    f"Cannot update source schema contract for run {run_id} source_node_id={source_node_id!r}: "
                    "contract JSON is NULL but conditional update affected no rows."
                )
            if existing.schema_contract_hash is None:
                raise AuditIntegrityError(
                    f"Cannot update source schema contract for run {run_id} source_node_id={source_node_id!r}: "
                    "existing contract JSON has no hash."
                )
            existing_contract = ContractAuditRecord.from_json(existing.schema_contract_json).to_schema_contract()
            existing_hash = existing_contract.version_hash()
            if existing_hash != existing.schema_contract_hash:
                raise AuditIntegrityError(
                    f"Existing source schema contract hash mismatch for run {run_id} source_node_id={source_node_id!r}: "
                    f"stored={existing.schema_contract_hash}, recomputed={existing_hash}."
                )
            if existing.schema_contract_hash != schema_contract_hash:
                raise AuditIntegrityError(
                    f"Cannot overwrite source schema contract for run {run_id} source_node_id={source_node_id!r}: "
                    f"existing={existing.schema_contract_hash}, new={schema_contract_hash}."
                )

    def get_run_source_resume_records(self, run_id: str) -> dict[str, RunSourceResumeRecord]:
        """Return per-source schema and contract records for resume.

        The returned mapping is keyed by ``source_node_id`` because rows carry
        source-node identity. Missing schema or contract data is audit
        corruption for resume: replaying rows without it would either lose type
        fidelity or apply the wrong schema contract.
        """
        rows = self._ops.execute_fetchall(
            select(
                run_sources_table.c.source_node_id,
                run_sources_table.c.source_name,
                run_sources_table.c.lifecycle_state,
                run_sources_table.c.schema_json,
                run_sources_table.c.schema_contract_json,
                run_sources_table.c.schema_contract_hash,
            ).where(run_sources_table.c.run_id == run_id)
        )
        records: dict[str, RunSourceResumeRecord] = {}
        for row in rows:
            if row.schema_json is None:
                raise AuditIntegrityError(
                    f"Run {run_id} source {row.source_name!r} ({row.source_node_id}) has no source schema stored. "
                    "Resume cannot preserve type fidelity without per-source schema JSON."
                )
            if row.schema_contract_json is None:
                raise AuditIntegrityError(
                    f"Run {run_id} source {row.source_name!r} ({row.source_node_id}) has no schema contract stored. "
                    "Resume cannot safely wrap rows without the source-scoped contract."
                )
            if row.schema_contract_hash is None:
                raise AuditIntegrityError(
                    f"Run {run_id} source {row.source_name!r} ({row.source_node_id}) has contract JSON but no contract hash."
                )
            contract = ContractAuditRecord.from_json(row.schema_contract_json).to_schema_contract()
            if contract.version_hash() != row.schema_contract_hash:
                raise AuditIntegrityError(
                    f"Run {run_id} source {row.source_name!r} ({row.source_node_id}) contract hash mismatch. "
                    f"Expected {row.schema_contract_hash}, got {contract.version_hash()}."
                )
            records[row.source_node_id] = RunSourceResumeRecord(
                source_node_id=row.source_node_id,
                source_name=row.source_name,
                lifecycle_state=row.lifecycle_state,
                source_schema_json=row.schema_json,
                schema_contract=contract,
            )
        return records

    def get_run_source_lifecycle_records(self, run_id: str) -> dict[str, RunSourceLifecycleRecord]:
        """Return per-source lifecycle records without requiring replay schemas.

        Resume must refuse incomplete sources before reconstructing row replay
        schemas. A declared source that never started is intentionally
        lifecycle_state=ready and may not yet have first-row-inferred contract
        data; treating that as schema corruption would hide the clearer
        source-exhaustion refusal.
        """
        rows = self._ops.execute_fetchall(
            select(
                run_sources_table.c.source_node_id,
                run_sources_table.c.source_name,
                run_sources_table.c.lifecycle_state,
            ).where(run_sources_table.c.run_id == run_id)
        )
        return {
            row.source_node_id: RunSourceLifecycleRecord(
                source_node_id=row.source_node_id,
                source_name=row.source_name,
                lifecycle_state=row.lifecycle_state,
            )
            for row in rows
        }

    def get_source_field_resolution(self, run_id: str) -> dict[str, str] | None:
        """Get source field resolution mapping for a run.

        Returns the mapping from original header names to final (normalized) field names.
        Used by sinks with headers: original to restore original headers.

        Args:
            run_id: Run to query

        Returns:
            Dict mapping original header name -> final field name, or None if no
            field resolution was recorded.

        Note:
            For reverse lookup (final -> original), callers should invert this dict:
            `{v: k for k, v in mapping.items()}`
        """
        query = select(runs_table.c.source_field_resolution_json).where(runs_table.c.run_id == run_id)
        result = self._ops.execute_fetchone(query)

        if result is None:
            raise AuditIntegrityError(f"Run {run_id} not found in database")

        resolution_json = result.source_field_resolution_json
        if resolution_json is None:
            return None

        return self._parse_field_resolution_mapping(
            run_id=run_id,
            resolution_json=resolution_json,
            location="run-level source_field_resolution_json",
        )

    def get_source_field_resolutions(self, run_id: str) -> dict[str, RunSourceFieldResolutionRecord]:
        """Get source-scoped field resolution mappings for a run.

        Multi-source runs store field resolution by ``source_node_id``. A
        ``None`` mapping is preserved for sources that did not record one so
        callers can distinguish "single missing mapping" from "no source
        metadata exists".
        """
        run_exists = self._ops.execute_fetchone(select(runs_table.c.run_id).where(runs_table.c.run_id == run_id))
        if run_exists is None:
            raise AuditIntegrityError(f"Run {run_id} not found in database")

        rows = self._ops.execute_fetchall(
            select(
                run_sources_table.c.source_node_id,
                run_sources_table.c.source_name,
                run_sources_table.c.field_resolution_json,
            )
            .where(run_sources_table.c.run_id == run_id)
            .order_by(run_sources_table.c.source_name, run_sources_table.c.source_node_id)
        )
        records: dict[str, RunSourceFieldResolutionRecord] = {}
        for row in rows:
            mapping = None
            if row.field_resolution_json is not None:
                mapping = self._parse_field_resolution_mapping(
                    run_id=run_id,
                    resolution_json=row.field_resolution_json,
                    location=f"run_sources field_resolution_json for source {row.source_name!r} ({row.source_node_id})",
                )
            records[row.source_node_id] = RunSourceFieldResolutionRecord(
                source_node_id=row.source_node_id,
                source_name=row.source_name,
                resolution_mapping=mapping,
            )
        return records

    def get_resume_field_resolution(self, run_id: str) -> dict[str, str] | None:
        """Return the only safe field-resolution mapping for sink resume.

        Current sink resume hooks accept one mapping per sink, not one mapping
        per source token. For multi-source runs, using the legacy run-level
        singleton can silently apply the wrong source's original headers. This
        method fails closed unless every recorded source mapping is present and
        identical.
        """
        source_records = self.get_source_field_resolutions(run_id)
        if not source_records:
            return self.get_source_field_resolution(run_id)

        if len(source_records) == 1:
            record = next(iter(source_records.values()))
            if record.resolution_mapping is not None:
                return dict(record.resolution_mapping)
            return self.get_source_field_resolution(run_id)

        missing_sources = [record.source_name for record in source_records.values() if record.resolution_mapping is None]
        if missing_sources:
            missing = ", ".join(sorted(missing_sources))
            raise AuditIntegrityError(
                f"Cannot resume headers: original for multi-source run {run_id}: "
                f"source field-resolution mapping is missing for source(s): {missing}. "
                "A single sink resume mapping would be ambiguous."
            )

        by_fingerprint: dict[str, dict[str, str]] = {}
        by_fingerprint_sources: dict[str, list[str]] = {}
        for record in source_records.values():
            if record.resolution_mapping is None:
                raise AssertionError("missing_sources gate should have rejected None mappings")
            fingerprint = canonical_json(record.resolution_mapping)
            if fingerprint not in by_fingerprint:
                by_fingerprint[fingerprint] = dict(record.resolution_mapping)
                by_fingerprint_sources[fingerprint] = []
            by_fingerprint_sources[fingerprint].append(record.source_name)

        if len(by_fingerprint) > 1:
            source_groups = [f"{', '.join(sorted(source_names))}" for source_names in by_fingerprint_sources.values()]
            raise AuditIntegrityError(
                f"Cannot resume headers: original for multi-source run {run_id}: "
                "recorded sources have different original-header mappings "
                f"({'; '.join(source_groups)}). The current sink resume contract accepts "
                "one mapping, so resume must use a source-scoped sink path or fail closed."
            )

        return next(iter(by_fingerprint.values()))

    def _parse_field_resolution_mapping(
        self,
        *,
        run_id: str,
        resolution_json: str,
        location: str,
    ) -> dict[str, str]:
        # Parse the stored JSON structure. This is Tier 1 (our data) — crash on any anomaly.
        try:
            resolution_data = json.loads(resolution_json)
        except json.JSONDecodeError as exc:
            raise AuditIntegrityError(
                f"Corrupt field resolution JSON for run {run_id} in {location}: "
                f"failed to parse stored JSON — database corruption (Tier 1 violation). "
                f"Parse error: {exc}"
            ) from exc
        if not isinstance(resolution_data, dict):
            raise AuditIntegrityError(
                f"Corrupt field resolution data for run {run_id} in {location}: expected dict, got {type(resolution_data).__name__}"
            )

        # Tier 1: resolution_mapping MUST exist if JSON is stored
        # record_source_field_resolution() always stores this key, so missing = corruption
        if "resolution_mapping" not in resolution_data:
            raise AuditIntegrityError(
                f"Corrupt field resolution data for run {run_id} in {location}: "
                f"missing required key 'resolution_mapping'. "
                f"This indicates database corruption — field resolution writers always store this key."
            )

        resolution_mapping = resolution_data["resolution_mapping"]
        if not isinstance(resolution_mapping, dict):
            raise AuditIntegrityError(
                f"Corrupt resolution_mapping for run {run_id} in {location}: expected dict, got {type(resolution_mapping).__name__}"
            )

        # Verify all keys and values are strings (Tier 1 — crash on corruption)
        # Key type check is defense-in-depth: JSON keys are always strings after json.loads(),
        # but guards against hypothetical non-JSON deserialization paths.
        validated_mapping: dict[str, str] = {}
        for key, value in resolution_mapping.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise AuditIntegrityError(
                    f"Corrupt resolution_mapping entry for run {run_id} in {location}: "
                    f"expected str->str, got {type(key).__name__}->{type(value).__name__}"
                )
            validated_mapping[key] = value

        return validated_mapping

    def _execute_atomic_inserts(self, *, context: str, statements: list[Any]) -> None:
        """Execute one or more INSERTs in a single transaction with audit error normalization."""
        if not statements:
            return
        try:
            with self._db.write_connection() as conn:
                for stmt in statements:
                    result = conn.execute(stmt)
                    if result.rowcount == 0:
                        raise LandscapeRecordError(f"{context} — zero rows affected (audit write failure)")
        except LandscapeRecordError:
            raise
        except SQLAlchemyError as exc:
            raise LandscapeRecordError(f"{context} — database rejected audit write: {type(exc).__name__}: {exc}") from exc

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        """Update run status without setting completed_at.

        Used for intermediate status changes (e.g., RUNNING during resume).
        For final completion, use complete_run() instead.

        Args:
            run_id: Run to update
            status: New RunStatus

        Raises:
            AuditIntegrityError: If run_id not found or current status is COMPLETED (immutable)

        Note:
            This encapsulates run status updates for Orchestrator recovery.
            Only updates status field — does not set completed_at or reproducibility_grade.

            COMPLETED runs are immutable — a completed run succeeded and its audit
            record is final. FAILED and INTERRUPTED runs CAN be transitioned back
            to RUNNING during resume (orchestrator recovery path).
        """
        if status in _IMMUTABLE_SUCCESS_RUN_STATUSES:
            raise AuditIntegrityError(
                f"update_run_status() cannot set status to {status.value!r}. "
                "Use complete_run() so completed_at is recorded in the audit trail."
            )

        with self._db.write_connection() as conn:
            # When resuming to RUNNING, clear completed_at atomically.
            # A run cannot be simultaneously RUNNING and completed — that's
            # an impossible state that confuses operational tooling and auditors.
            values: dict[str, Any] = {"status": status.value}
            if status == RunStatus.RUNNING:
                values["completed_at"] = None
            result = conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .where(runs_table.c.status.notin_(_IMMUTABLE_SUCCESS_RUN_STATUS_VALUES))
                .values(**values)
            )
            if result.rowcount == 0:
                existing = conn.execute(select(runs_table.c.status).where(runs_table.c.run_id == run_id)).fetchone()
                if existing is not None and existing.status in _IMMUTABLE_SUCCESS_RUN_STATUS_VALUES:
                    existing_status = RunStatus(existing.status)
                    raise AuditIntegrityError(
                        f"Cannot transition run {run_id} from {existing_status.name} ({existing_status.value!r}) to {status.value!r}. "
                        f"Successful terminal runs are immutable. "
                        f"FAILED/INTERRUPTED runs can be resumed via update_run_status."
                    )
                raise AuditIntegrityError(f"Cannot update run status to {status.value!r}: run {run_id} not found")

    def record_secret_resolutions(
        self,
        run_id: str,
        resolutions: list[SecretResolutionInput],
    ) -> None:
        """Record secret resolution events from deferred records.

        Called by orchestrator after run is created. The resolution records
        were captured during load_secrets_from_config() before the run existed.

        All inserts are batched in a single transaction for atomicity —
        either all resolutions are recorded or none are.

        Args:
            run_id: The run ID to associate resolutions with
            resolutions: Typed resolution records from load_secrets_from_config().
        """
        if not resolutions:
            return
        statements = [
            secret_resolutions_table.insert().values(
                resolution_id=generate_id(),
                run_id=run_id,
                timestamp=rec.timestamp,
                env_var_name=rec.env_var_name,
                source=rec.source,
                vault_url=rec.vault_url,
                secret_name=rec.secret_name,
                fingerprint=rec.fingerprint,
                resolution_latency_ms=rec.resolution_latency_ms,
            )
            for rec in resolutions
        ]
        self._execute_atomic_inserts(
            context=f"record_secret_resolutions run_id={run_id}",
            statements=statements,
        )

    def get_secret_resolutions_for_run(self, run_id: str) -> list[SecretResolution]:
        """Get all secret resolution records for a run.

        These records document which secrets were loaded from Key Vault
        for this run, including their HMAC fingerprints (not values).

        Args:
            run_id: Run ID to query

        Returns:
            List of SecretResolution models, ordered by timestamp
        """
        query = (
            select(secret_resolutions_table)
            .where(secret_resolutions_table.c.run_id == run_id)
            .order_by(secret_resolutions_table.c.timestamp)
        )
        db_rows = self._ops.execute_fetchall(query)
        return [
            SecretResolution(
                resolution_id=row.resolution_id,
                run_id=row.run_id,
                timestamp=row.timestamp,
                env_var_name=row.env_var_name,
                source=row.source,
                vault_url=row.vault_url,
                secret_name=row.secret_name,
                fingerprint=row.fingerprint,
                resolution_latency_ms=row.resolution_latency_ms,
            )
            for row in db_rows
        ]

    def record_preflight_results(
        self,
        run_id: str,
        preflight: PreflightResult,
    ) -> None:
        """Record pre-flight dependency and gate results in the audit trail.

        Called by orchestrator after run is created. Pre-flight results were
        captured during bootstrap_and_run() before the run existed.

        All inserts are batched in a single connection (db.connection() is a
        transaction context manager).

        Args:
            run_id: The run ID to associate results with
            preflight: Combined pre-flight results (dependencies + gates)
        """
        rows_to_insert = []

        for dep in preflight.dependency_runs:
            rows_to_insert.append(
                {
                    "result_id": generate_id(),
                    "run_id": run_id,
                    "result_type": "dependency_run",
                    "name": dep.name,
                    "result_json": canonical_json(
                        {
                            "run_id": dep.run_id,
                            "settings_hash": dep.settings_hash,
                            "duration_ms": dep.duration_ms,
                            "indexed_at": dep.indexed_at,
                        }
                    ),
                    "created_at": now(),
                }
            )

        for gate in preflight.gate_results:
            rows_to_insert.append(
                {
                    "result_id": generate_id(),
                    "run_id": run_id,
                    "result_type": "commencement_gate",
                    "name": gate.name,
                    "result_json": canonical_json(
                        {
                            "condition": gate.condition,
                            "result": gate.result,
                            "context_snapshot": deep_thaw(gate.context_snapshot),
                        }
                    ),
                    "created_at": now(),
                }
            )

        if not rows_to_insert:
            return

        self._execute_atomic_inserts(
            context=f"record_preflight_results run_id={run_id}",
            statements=[preflight_results_table.insert().values(**row_data) for row_data in rows_to_insert],
        )

    def record_readiness_check(
        self,
        run_id: str,
        *,
        name: str,
        collection: str,
        reachable: bool,
        count: int | None,
        message: str,
    ) -> None:
        """Record a readiness check result in the audit trail.

        Called by transforms during on_start() after a provider readiness
        check passes. Records the collection state at startup time so
        auditors can answer "what was the collection state when this ran?"
        """
        row_data = {
            "result_id": generate_id(),
            "run_id": run_id,
            "result_type": "readiness_check",
            "name": name,
            "result_json": canonical_json(
                {
                    "collection": collection,
                    "reachable": reachable,
                    "count": count,
                    "message": message,
                }
            ),
            "created_at": now(),
        }

        self._execute_atomic_inserts(
            context=f"record_readiness_check run_id={run_id}",
            statements=[preflight_results_table.insert().values(**row_data)],
        )

    def list_runs(self, *, status: RunStatus | None = None) -> list[Run]:
        """List all runs in the database.

        Args:
            status: Optional RunStatus filter

        Returns:
            List of Run models, ordered by started_at (newest first)
        """
        query = select(runs_table).order_by(runs_table.c.started_at.desc())

        if status is not None:
            query = query.where(runs_table.c.status == status.value)

        rows = self._ops.execute_fetchall(query)
        return [self._run_loader.load(row) for row in rows]

    def set_export_status(
        self,
        run_id: str,
        status: ExportStatus,
        *,
        error: str | None = None,
        export_format: str | None = None,
        export_sink: str | None = None,
    ) -> None:
        """Set export status for a run.

        This is separate from run status so export failures don't mask
        successful pipeline completion.

        Args:
            run_id: Run to update
            status: ExportStatus (PENDING, COMPLETED, or FAILED)
            error: Error message if status is FAILED
            export_format: Format used (csv, json)
            export_sink: Sink name used for export
        """
        # Validate error/status consistency — error is only meaningful with FAILED
        if error is not None and status != ExportStatus.FAILED:
            raise AuditIntegrityError(
                f"Cannot set export_error with status={status.value}. Error messages are only valid with FAILED status."
            )

        updates: dict[str, Any] = {
            "export_status": status,
            "exported_at": None,
        }

        if status == ExportStatus.COMPLETED:
            updates["exported_at"] = now()
            # Clear stale error when transitioning to completed
            updates["export_error"] = None
        elif status == ExportStatus.PENDING:
            # Clear stale error when transitioning to pending
            updates["export_error"] = None

        # Only set error if explicitly provided (for FAILED status)
        if error is not None:
            updates["export_error"] = error

        if export_format is not None:
            updates["export_format"] = export_format
        if export_sink is not None:
            updates["export_sink"] = export_sink

        try:
            self._ops.execute_update(runs_table.update().where(runs_table.c.run_id == run_id).values(**updates))
        except AuditIntegrityError as exc:
            raise AuditIntegrityError(f"Cannot set export status to {status.value!r}: run {run_id} not found") from exc

    def finalize_run(self, run_id: str, status: RunStatus) -> Run:
        """Finalize a run by computing grade and completing it.

        Convenience method that:
        1. Computes the reproducibility grade based on node determinism
        2. Completes the run with the specified status and computed grade

        Args:
            run_id: Run to finalize
            status: Final RunStatus (COMPLETED, FAILED, or INTERRUPTED)

        Returns:
            Updated Run model

        Note:
            Grade computation and run completion execute in separate transactions.
            This is an accepted limitation — the invariant that all nodes are registered
            before finalize_run is called ensures the grade is stable between reads.
            A single-transaction approach would require refactoring compute_grade's
            database access (tracked for future consideration).
        """
        grade = self.compute_reproducibility_grade(run_id)
        return self.complete_run(run_id, status, reproducibility_grade=grade)

    def compute_reproducibility_grade(self, run_id: str) -> ReproducibilityGrade:
        """Compute reproducibility grade for a run based on node determinism.

        Logic:
        - If any node has determinism='nondeterministic', returns REPLAY_REPRODUCIBLE
        - Otherwise returns FULL_REPRODUCIBLE
        - 'seeded' counts as reproducible

        Args:
            run_id: Run ID to compute grade for

        Returns:
            ReproducibilityGrade enum value

        Note:
            Uses self._db directly (bypassing DatabaseOps) because compute_grade()
            needs raw connection access for multi-statement reads within a single
            connection. This dual access pattern (self._ops + self._db) is accepted:
            future repositories (B2/B3) will also need self._db for atomic
            multi-table transactions. Both are injected via __init__.
        """
        return compute_grade(self._db, run_id)
