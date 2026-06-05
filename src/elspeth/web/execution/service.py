"""ExecutionServiceImpl — background pipeline execution with thread safety.

Thread safety fixes implemented here:
- B2: Always pass shutdown_event=threading.Event() to Orchestrator.run()
- B3: Construct LandscapeDB/PayloadStore from WebSettings resolvers
- B7: except BaseException + future.add_done_callback() safety net
- B8/C1: _call_async() bridges sync thread to async event loop for SessionService

The _run_pipeline() method is the ONLY code that runs outside the asyncio
event loop. Everything else runs in the main async context. Because
SessionService methods are async, _run_pipeline() uses _call_async() to
schedule coroutines on the main event loop from the background thread.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Literal, TypeVar, cast
from uuid import UUID

import structlog
from opentelemetry import metrics
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.audit import SecretResolutionInput
from elspeth.contracts.blobs_inline import BlobInlineRef
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.core.blobs_inline import (
    BLOB_INLINE_AGGREGATE_BYTE_CAP,
    BLOB_INLINE_PER_REF_BYTE_CAP,
    _discover_blob_content_refs,
    _enforce_blob_content_ref_metadata,
    _fetch_blob_contents,
    _substitute_blob_content_refs,
)
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.events import EventBus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.run_lifecycle_repository import is_valid_sha256_hex
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.core.secrets import SecretResolutionError
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.protocol import (
    AllowedMimeType,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    BlobRecord,
    BlobServiceProtocol,
    BlobStateError,
)
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.composer.state import CompositionState
from elspeth.web.config import WebSettings
from elspeth.web.execution.accounting import load_run_accounting_from_db
from elspeth.web.execution.errors import (
    BlobSourcePathMismatchError,
    MalformedBlobRefError,
    PathAllowlistViolationError,
    SemanticContractViolationError,
    UnresolvedInterpretationPlaceholderError,
)
from elspeth.web.execution.failure_samples import format_failure_samples, load_top_failure_samples
from elspeth.web.execution.fanout_guard import (
    ExecutionFanoutGuardRequired,
    annotate_pipeline_yaml_with_fanout_guard,
    evaluate_execution_fanout_guard,
)
from elspeth.web.execution.preflight import build_validated_runtime_graph, resolve_runtime_yaml_paths
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.protocol import ExecutionService, StateAccessError, YamlGenerator
from elspeth.web.execution.schemas import (
    CancelledData,
    CompletedData,
    FailedData,
    ProgressData,
    RunAccounting,
    RunEvent,
    RunStatusResponse,
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
    ValidationResult,
)
from elspeth.web.interpretation_state import InterpretationReviewPending, materialize_state_for_execution
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.protocol import (
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    IllegalRunTransitionError,
    RunAlreadyActiveError,
    RunRecord,
    SessionRunStatus,
    SessionServiceProtocol,
)  # B1: canonical definition
from elspeth.web.sessions.telemetry import _SessionsTelemetry

slog = structlog.get_logger()
_meter = metrics.get_meter(__name__)
_BLOB_INLINE_HASH_MISMATCH_TOTAL = _meter.create_counter(
    name="composer.blob_inline.hash_mismatch_total",
    description="composer-pinned hash did not match runtime-fetched blob content; SLO threshold = 0",
)
_BLOB_INLINE_AUDIT_ROW_TIER1_VIOLATION_TOTAL = _meter.create_counter(
    name="composer.blob_inline.audit_row_tier1_violation_total",
    description="resolved inline blob ref produced no audit row; SLO threshold = 0",
)

T = TypeVar("T")


def _sanitize_error_for_client(exc: BaseException) -> str:
    """Return a client-safe error message for a pipeline failure.

    Only typed exceptions with purpose-built safe messages may expose
    details. Broad built-ins such as ValueError, TypeError, and KeyError
    are reduced to a generic class-name message because their str() output
    can carry validation structure, function signatures, and internal keys.
    The full exception is recorded in runs.error by _run_pipeline's
    except-BaseException block.
    """
    if isinstance(exc, SecretResolutionError):
        return "One or more secret references could not be resolved. Check the Secrets panel."
    return f"Pipeline execution failed ({type(exc).__name__})"


def _schema_contract_violation_errors(exc: PydanticValidationError) -> list[dict[str, str]]:
    """Extract field-level schema diagnostics without raw input values."""
    return [
        {
            "loc": ".".join(str(part) for part in error["loc"]) or "<root>",
            "type": str(error["type"]),
        }
        for error in exc.errors(include_url=False, include_input=False)
    ]


# Phase 2.2 (elspeth-0de989c56d): mapping from the engine's L0 RunStatus
# to the L3 SessionRunStatus Literal that the API surfaces and the web
# sessions DB persists.  The two enums share the same value strings for
# the four-value taxonomy (completed / completed_with_failures / failed /
# empty) so the mapping is the lower-case enum value.  RUNNING and
# INTERRUPTED don't appear here — INTERRUPTED is mapped to "cancelled" in
# the GracefulShutdownError branch separately, and RUNNING is non-terminal
# (the engine never returns it from a normal completion path).
_RUN_RESULT_STATUS_TO_SESSION_STATUS: dict[RunStatus, SessionRunStatus] = {
    RunStatus.COMPLETED: "completed",
    RunStatus.COMPLETED_WITH_FAILURES: "completed_with_failures",
    RunStatus.FAILED: "failed",
    RunStatus.EMPTY: "empty",
}


def _session_status_from_run_result_status(status: RunStatus) -> SessionRunStatus:
    """Translate the engine's L0 RunStatus to the API's SessionRunStatus.

    Raises:
        ValueError: when the engine returns a status this site does not
            handle (RUNNING, INTERRUPTED).  RUNNING from a normal
            completion path is a framework-level invariant violation;
            INTERRUPTED is handled by the GracefulShutdownError branch
            with explicit ``status="cancelled"``.
    """
    try:
        return _RUN_RESULT_STATUS_TO_SESSION_STATUS[status]
    except KeyError as exc:
        raise ValueError(
            f"Cannot translate RunStatus {status!r} to a SessionRunStatus on the success path. "
            f"INTERRUPTED is handled by the GracefulShutdownError branch; RUNNING is non-terminal "
            f"and must not appear after orchestrator.run() returns."
        ) from exc


def _structural_failure_message(*, rows_processed: int, failure_samples: str = "") -> str:
    """elspeth-0de989c56d / elspeth-5069612f3c — synthetic structural error
    for FAILED-from-row-shape after the rows_routed split.

    The L3 RunRecord.__post_init__ invariant requires a non-empty error for
    status='failed'. When the engine returns RunStatus.FAILED from a row-shape
    decision (no exception propagated; no success indicator: rows_succeeded == 0
    AND rows_routed_success == 0), this helper produces a structural fact —
    operator-readable, no candidate-secret material, no echoed user-row data.

    After elspeth-5069612f3c, gate-routed pipelines (rows_routed_success > 0)
    no longer reach this code path — they classify as COMPLETED. This message
    fires only when no row reached EITHER the success-counted terminal state
    OR an intentional gate-routed sink, i.e. when every row failed terminally
    or was diverted via on_error.

    ``failure_samples`` is an optional pre-formatted bullet list of the most
    common per-row error messages (see ``failure_samples.format_failure_samples``).
    When supplied, it is appended so the runs view shows the dominant failure
    modes inline — the panel-expand affordance still has the full per-token
    drill-down, but the headline already names the problem.
    """
    base = (
        f"No row reached a success path (rows_processed={rows_processed}, "
        f"rows_succeeded=0, rows_routed_success=0). "
        f"All rows either failed terminally or were routed via on_error to a "
        f"failure sink."
    )
    if failure_samples:
        return f"{base} Top per-row failures:\n{failure_samples}"
    return f"{base} Expand this run for per-row failure details."


def _partial_completion_message(
    *,
    rows_succeeded: int,
    rows_failed: int,
    rows_routed_failure: int,
    rows_quarantined: int,
    failure_samples: str = "",
) -> str:
    """Operator-readable summary for COMPLETED_WITH_FAILURES runs.

    Sibling to ``_structural_failure_message``: a structural fact, no
    candidate-secret material, no echoed user-row data. Populated into
    ``session_error`` so the frontend (and any other audit consumer) has a
    single field to render for failure-like terminal runs without needing to
    re-implement the L0 ``failure_indicator`` predicate.

    The RunRecord invariant at sessions/protocol.py:237-238 requires a
    non-empty ``error`` only for ``status='failed'`` — populating ``error``
    on COMPLETED_WITH_FAILURES is permitted, and the schema validators
    accept it (see tests/unit/web/execution/test_schemas.py:1156).

    ``failure_samples`` mirrors the same parameter on
    ``_structural_failure_message`` — see its docstring.
    """
    base = (
        f"Run completed with failures (rows_succeeded={rows_succeeded}, "
        f"rows_failed={rows_failed}, rows_routed_failure={rows_routed_failure}, "
        f"rows_quarantined={rows_quarantined})."
    )
    if failure_samples:
        return f"{base} Top per-row failures:\n{failure_samples}"
    return f"{base} Expand this run for per-row failure details."


# B1 fix: RunAlreadyActiveError is NOT defined here — imported from
# sessions.protocol where the canonical definition lives. Defining a
# second class with the same name would prevent app.py's global
# exception handler (which catches sessions.protocol.RunAlreadyActiveError)
# from catching exceptions raised here.


class ExecutionServiceImpl:
    """Pipeline execution service with ThreadPoolExecutor backend.

    Construction: Created inside the FastAPI lifespan async context manager
    (after ProgressBroadcaster), NOT in the synchronous create_app() factory.
    Stored as application state and injected into route handlers via FastAPI's
    dependency injection. The event loop reference is obtained from
    asyncio.get_running_loop() in the lifespan (same loop as ProgressBroadcaster).

    Thread model: execute() submits _run_pipeline() to a ThreadPoolExecutor
    with max_workers=1. The pipeline runs in a background thread. All other
    methods run in the asyncio event loop thread.

    B8/C1 fix: SessionService methods are async. _run_pipeline() runs in a
    background thread and uses _call_async() to bridge async calls back to
    the main event loop via asyncio.run_coroutine_threadsafe().
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        broadcaster: ProgressBroadcaster,
        settings: WebSettings,
        session_service: SessionServiceProtocol,
        yaml_generator: YamlGenerator,
        telemetry: _SessionsTelemetry,
        blob_service: BlobServiceProtocol | None = None,
        secret_service: WebSecretResolver | None = None,
    ) -> None:
        self._loop = loop
        self._broadcaster = broadcaster
        self._settings = settings
        self._session_service = session_service
        self._yaml_generator = yaml_generator
        self._telemetry = telemetry
        self._blob_service = blob_service
        self._secret_service = secret_service
        # AC #17: No run_repository — all Run CRUD delegates to SessionService
        # via create_run(), update_run_status(), get_active_run(), get_run().
        # R6 expanded params: landscape_run_id, pipeline_yaml, rows_processed,
        # rows_failed.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._shutdown_events: dict[str, threading.Event] = {}
        self._shutdown_events_lock = threading.Lock()
        # Per-session asyncio lock to prevent TOCTOU on the active-run check.
        # Keyed by session_id string; lazily created, cleaned up on session
        # deletion via cleanup_session_lock().
        self._session_locks: dict[str, asyncio.Lock] = {}
        # OpenRouter catalog snapshot id, populated by the lifespan after
        # the boot probe via ``set_openrouter_catalog_snapshot()``. Both
        # fields are required to be present before any pipeline executes
        # — run-create writes them into the Landscape ``runs`` row.
        # ``None`` here is a programmer bug (lifespan never set them) and
        # crashes loudly at ``_run_pipeline`` rather than silently
        # dropping the audit field.
        self._openrouter_catalog_sha256: str | None = None
        self._openrouter_catalog_source: str | None = None

    def set_openrouter_catalog_snapshot(self, *, sha256: str, source: str) -> None:
        """Record the boot-time OpenRouter catalog snapshot id.

        Called once from the FastAPI lifespan after
        :func:`prime_openrouter_catalog_from_live` completes (success or
        bundled fallback). Both arguments are required and concrete
        (non-empty string) — the lifespan reads them from
        :func:`elspeth.plugins.transforms.llm.model_catalog.read_openrouter_catalog_snapshot_id`
        which never returns ``None``.

        Snapshot is invariant for the process lifetime: re-priming is
        not supported (staging is restarted on deploys, and a restart
        re-runs the lifespan).
        """
        if not is_valid_sha256_hex(sha256):
            raise RuntimeError(f"openrouter_catalog_sha256 must be 64 lowercase hex chars, got {sha256!r}")
        if source not in ("live", "bundled"):
            raise RuntimeError(f"openrouter_catalog_source must be 'live' or 'bundled', got {source!r}")
        self._openrouter_catalog_sha256 = sha256
        self._openrouter_catalog_source = source

    def _call_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """Bridge an async call from the background thread to the main event loop.

        B8/C1 fix: SessionService methods are async, but _run_pipeline() runs
        in a ThreadPoolExecutor worker thread. This helper schedules the
        coroutine on the main event loop and blocks until it completes.

        R6 fix: 30-second timeout prevents indefinite hangs if the event loop
        cannot run the scheduled coroutine during shutdown or another
        infrastructure stall.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30.0)

    def get_live_run_ids(self) -> frozenset[str]:
        """Return run IDs still owned by an executor thread.

        A run ID is present in _shutdown_events from the moment it is
        registered in _execute_locked (before thread pool submission)
        until the _run_pipeline finally block removes it.

        Cancellation only signals the worker thread via Event.set(); it
        does not mean the thread has finished its GracefulShutdownError
        unwinding or finalization work. Periodic orphan cleanup must keep
        excluding signalled runs until the worker removes them here.

        Thread-safe: returns a snapshot under the lock.
        """
        with self._shutdown_events_lock:
            return frozenset(self._shutdown_events)

    def cleanup_session_lock(self, session_id: str) -> None:
        """Remove the per-session asyncio lock for a deleted session.

        Called from the delete_session route after archive_session()
        completes. Matches the ProgressBroadcaster.cleanup_run() pattern.
        """
        self._session_locks.pop(session_id, None)

    async def shutdown(self) -> None:
        """Shut down the thread pool without blocking the event loop.

        Sets all active shutdown events first so running pipelines can
        terminate gracefully, then drains the executor in a helper thread.
        Worker shutdown paths still use _call_async() to persist terminal
        state on the main event loop, so blocking the loop here can strand
        those final updates.
        """
        with self._shutdown_events_lock:
            events = list(self._shutdown_events.values())
        for event in events:
            event.set()
        await run_sync_in_worker(self._executor.shutdown, True)

    async def execute(
        self,
        session_id: UUID,
        state_id: UUID | None = None,
        *,
        user_id: str | None = None,
        auth_provider_type: str | None = None,
        fanout_ack_token: str | None = None,
    ) -> UUID:
        """Start a background pipeline run.

        B6 enforcement: raises RunAlreadyActiveError if a pending or running
        run already exists for this session.

        Returns the run_id immediately.

        Args:
            session_id: Session to execute.
            state_id: Specific state to execute (latest if None).
            user_id: Authenticated user's ID for scoped secret resolution.
            auth_provider_type: Auth provider namespace for Landscape run attribution.
            fanout_ack_token: Optional launch acknowledgement for high-fanout
                LLM/provider-call risk.

        Note: async because SessionService methods are async. The pipeline
        itself runs in a background thread — only setup is async.
        """
        # TOCTOU fix: per-session asyncio lock serialises the
        # get_active_run → create_run window so two concurrent execute()
        # calls cannot both pass the check before either creates a run.
        session_key = str(session_id)
        lock = self._session_locks.setdefault(session_key, asyncio.Lock())
        async with lock:
            return await self._execute_locked(
                session_id,
                state_id,
                user_id=user_id,
                auth_provider_type=auth_provider_type,
                fanout_ack_token=fanout_ack_token,
            )

    async def _execute_locked(
        self,
        session_id: UUID,
        state_id: UUID | None = None,
        *,
        user_id: str | None = None,
        auth_provider_type: str | None = None,
        fanout_ack_token: str | None = None,
    ) -> UUID:
        """Inner execute — runs under the per-session asyncio.Lock."""
        # B6: One active run per session (AC #17: via SessionService)
        active = await self._session_service.get_active_run(session_id)
        if active is not None:
            raise RunAlreadyActiveError(str(session_id))

        # B4 fix: get_composition_state() doesn't exist on SessionService.
        # Use get_state() for explicit state_id, get_current_state() for latest.
        #
        # IDOR contract: the two "state unreachable" branches below
        # (state does not exist anywhere / state exists in another
        # user's session) MUST be indistinguishable from the client's
        # perspective.  They are folded into a single ``StateAccessError``
        # whose route handler returns a fixed "State not found" literal.
        # See ``protocol.StateAccessError`` for the full rationale; the
        # ``send_message`` route in ``sessions/routes.py`` is the
        # canonical precedent for this IDOR contract.  Do NOT
        # re-introduce distinguishable messages here: the whole reason
        # this branch exists as a discriminated check is to prevent
        # the attacker's probe, and a distinguishable message reopens
        # exactly the oracle the check was added to close.
        state_record = None
        if state_id is not None:
            try:
                state_record = await self._session_service.get_state(state_id)
            except ValueError as exc:
                raise StateAccessError(str(state_id)) from exc
            # Verify state belongs to the requested session (IDOR prevention)
            if state_record.session_id != session_id:
                raise StateAccessError(str(state_id))
        else:
            state_record = await self._session_service.get_current_state(session_id)
            if state_record is None:
                raise ValueError(f"No composition state exists for session {session_id}")

        assert state_record is not None

        # Bridge CompositionStateRecord → CompositionState for generate_yaml().
        # The record stores raw dicts; generate_yaml() needs the typed domain object.
        composition_state = state_from_record(state_record)

        semantic_errors, semantic_contracts = validate_semantic_contracts(composition_state)
        if semantic_errors:
            raise SemanticContractViolationError(
                entries=semantic_errors,
                contracts=semantic_contracts,
            )

        # F-17 / F-21 (Phase 5b Task 5 follow-on) — unresolved interpretation
        # placeholder gate. Runs AFTER semantic-contract validation (the
        # placeholder isn't a contract violation; the LLM transform's
        # prompt_template is still a string) and BEFORE the path allowlist /
        # YAML generation (we fail fast so the placeholder never reaches the
        # runtime engine that would substitute the literal string into the
        # LLM call). Operational telemetry counter emitted with non-content
        # component metadata per unresolved site — explicitly NOT the
        # prompt_template or user-authored term value.
        #
        # Operates under the operator-acknowledged assumption that 18a Task 0
        # (empirical LLM gate ≥ 8/10 staging runs emit
        # {{interpretation:<term>}}) passes; this detector is the
        # runtime-safety net catching cases where the LLM under-fires.
        materialized_state = materialize_state_for_execution(composition_state)
        if isinstance(materialized_state, InterpretationReviewPending):
            for site in materialized_state.sites:
                self._telemetry.interpretation_placeholder_unresolved_at_runtime_total.add(
                    1,
                    attributes={
                        "component_id": site.component_id,
                        "component_type": site.component_type,
                        "kind": site.kind.value,
                    },
                )
            raise UnresolvedInterpretationPlaceholderError(
                sites=tuple(materialized_state.sites),
            )
        composition_state = materialized_state

        # Path allowlist check — defense-in-depth. The validate endpoint also
        # checks this, but /execute does not require /validate first. An
        # authenticated user could skip validation and execute a state that
        # reads files outside the allowed directories.
        if composition_state.source is not None:
            from elspeth.web.paths import allowed_source_directories, resolve_data_path

            allowed_dirs = allowed_source_directories(str(self._settings.data_dir))
            for key in ("path", "file"):
                value = composition_state.source.options.get(key)
                if value is not None:
                    resolved = resolve_data_path(value, str(self._settings.data_dir))
                    if not any(resolved.is_relative_to(d) for d in allowed_dirs):
                        raise PathAllowlistViolationError(f"Source {key}='{value}' resolves outside allowed directories")

        # Sink path allowlist — prevents arbitrary file writes via sink options.
        # Without this, a client can set sink options.path to any absolute or
        # ../ path and /execute will write there.
        if composition_state.outputs:
            from elspeth.web.paths import allowed_sink_directories, resolve_data_path

            allowed_sink_dirs = allowed_sink_directories(str(self._settings.data_dir))
            for output in composition_state.outputs:
                for key in ("path", "file"):
                    value = output.options.get(key)
                    if value is not None:
                        resolved = resolve_data_path(value, str(self._settings.data_dir))
                        if not any(resolved.is_relative_to(d) for d in allowed_sink_dirs):
                            raise PathAllowlistViolationError(
                                f"Sink '{output.name}' {key}='{value}' resolves outside allowed output directories"
                            )

        pipeline_yaml = self._yaml_generator.generate_yaml(composition_state)

        # Resolve relative source/sink paths to absolute in the YAML so
        # plugins see the same paths the allowlist approved.  Without this,
        # plugins call PathConfig.resolved_path() with no base_dir, which
        # resolves relative paths against CWD — not data_dir.
        pipeline_yaml = resolve_runtime_yaml_paths(pipeline_yaml, str(self._settings.data_dir))

        # Pre-validate blob_ref UUID before creating the run record.
        # UUID() can raise ValueError on malformed strings; if that happens
        # after create_run(), the pending run blocks the session permanently
        # because the except block below only cleans up _shutdown_events.
        #
        # Defense-in-depth: verify the blob belongs to this session via the
        # DB ownership record. Without this, a crafted composition state
        # could reference another session's blob path (which would pass the
        # shared-root path allowlist above).
        parsed_blob_id: UUID | None = None
        if composition_state.source is not None and self._blob_service is not None:
            blob_ref = composition_state.source.options.get("blob_ref")
            if blob_ref is not None:
                try:
                    parsed_blob_id = UUID(blob_ref)
                except ValueError as exc:
                    raise MalformedBlobRefError("blob_ref must be a UUID") from exc
                # IDOR contract (mirrors the state_id branch above): the
                # nonexistent-blob and cross-session-blob cases MUST be
                # indistinguishable from the client's perspective.  Both
                # surface as ``BlobNotFoundError`` — ``get_blob`` already
                # raises it for missing rows; we raise the same type for
                # cross-session rows so the route handler returns a
                # byte-identical "Blob not found" 404.  Raising
                # ``ValueError`` here (as an earlier iteration did) not
                # only produced a distinguishable body but also a
                # distinguishable HTTP status (404 vs 500, because
                # ``BlobNotFoundError`` was uncaught), a two-channel
                # oracle strictly worse than the state_id surface.
                blob_record = await self._blob_service.get_blob(parsed_blob_id)
                if blob_record.session_id != session_id:
                    raise BlobNotFoundError(blob_ref)

                # Tier 1 read guard: composition_states.source.options.path
                # is our own audit data and must equal the canonical blob
                # storage_path.  A mismatch (or absence on a blob-backed
                # source) indicates a bug in composer persistence — crash
                # informatively rather than letting the source plugin 500
                # with FileNotFoundError on a structurally invalid path.
                # Offensive-programming pattern: membership check +
                # indexing instead of .get() so the absence case raises
                # the structured BlobSourcePathMismatchError rather than
                # an opaque KeyError.  See elspeth-07089fbaa3.
                source_options = composition_state.source.options
                canonical_path = blob_record.storage_path
                stored_path = source_options["path"] if "path" in source_options else None
                if stored_path != canonical_path:
                    raise BlobSourcePathMismatchError(
                        stored_path=stored_path,
                        canonical_path=canonical_path,
                        blob_id=str(parsed_blob_id),
                        session_id=str(session_id),
                    )

        fanout_guard = evaluate_execution_fanout_guard(
            composition_state,
            data_dir=self._settings.data_dir,
        )
        if fanout_guard is not None:
            if fanout_ack_token != fanout_guard.ack_token:
                raise ExecutionFanoutGuardRequired(fanout_guard)
            pipeline_yaml = annotate_pipeline_yaml_with_fanout_guard(pipeline_yaml, fanout_guard)

        # B9 fix: create_run() generates its own UUID internally and returns
        # a RunRecord. Read the run_id back from the returned record so our
        # _shutdown_events key matches the DB record.
        run_record = await self._session_service.create_run(
            session_id=session_id,
            state_id=state_record.id,  # From the record, not the domain object
            pipeline_yaml=pipeline_yaml,
        )
        run_id = run_record.id  # Use the DB-generated UUID as canonical

        # Register shutdown event immediately so cancel() always finds it.
        # Without this, cancel() firing between create_run() and registration
        # bypasses the event and updates DB to "cancelled" directly — causing
        # an illegal cancelled→running transition when _run_pipeline starts.
        shutdown_event = threading.Event()
        with self._shutdown_events_lock:
            self._shutdown_events[str(run_id)] = shutdown_event

        try:
            # Record blob-to-run linkage for input blobs
            if parsed_blob_id is not None and self._blob_service is not None:
                await self._blob_service.link_blob_to_run(
                    blob_id=parsed_blob_id,
                    run_id=run_id,
                    direction="input",
                )

            # Submit to thread pool
            future = self._executor.submit(
                self._run_pipeline,
                str(run_id),
                pipeline_yaml,
                shutdown_event,
                user_id,
                auth_provider_type,
            )
        except BaseException as exc:
            with self._shutdown_events_lock:
                # Idempotent cleanup of an internal bookkeeping key. Access it
                # directly (R9 remediation); the membership guard preserves the
                # silent no-op when cleanup races or runs twice.
                run_key = str(run_id)
                if run_key in self._shutdown_events:
                    del self._shutdown_events[run_key]
            # Transition run out of pending so the one-active-run constraint
            # doesn't permanently block this session.
            #
            # Narrow catch (canonical pattern, commits b8ba2214/127417cb):
            # ``SQLAlchemyError`` covers every DB-layer failure mode
            # (lock timeout, pool disconnect, deadlock, IntegrityError,
            # OperationalError, ProgrammingError); ``OSError`` covers
            # filesystem-adjacent failures routed through SQLAlchemy on
            # SQLite (``database is locked`` is an OperationalError subclass
            # of SQLAlchemyError, but a disk-full midway through a commit
            # can surface as OSError before SQLAlchemy wraps it). Programmer
            # bugs (AttributeError, TypeError, KeyError) from our own
            # service code must propagate — a cleanup path masking a
            # programmer bug is exactly the silent-wrong-result pattern
            # CLAUDE.md forbids.
            #
            # ``exc_class`` only: ``str(cleanup_err)`` on SQLAlchemyError
            # subclasses expands to ``[SQL: ...] [parameters: ...]`` and
            # appends ``__cause__`` text that can carry DB URLs /
            # credentials. ``str(exc)`` (the original) is similarly unsafe
            # because the outer ``BaseException`` catch sweeps up anything
            # including sanitizer bugs.  The client-facing message is
            # already routed through ``_sanitize_error_for_client`` above;
            # the slog must not re-expose the raw form.
            try:
                await self._session_service.update_run_status(
                    run_id, status="failed", error=f"Setup failed: {_sanitize_error_for_client(exc)}"
                )
            except (SQLAlchemyError, OSError) as cleanup_err:
                slog.error(
                    "run_cleanup_status_update_failed",
                    run_id=str(run_id),
                    original_exc_class=type(exc).__name__,
                    cleanup_exc_class=type(cleanup_err).__name__,
                )
            raise
        # B7 Layer 2: safety net callback
        future.add_done_callback(self._on_pipeline_done)

        return run_id

    async def get_status(
        self,
        run_id: UUID,
        *,
        accounting: RunAccounting | None = None,
        run_record: RunRecord | None = None,
    ) -> RunStatusResponse:
        """Return current run status. AC #17: delegates to SessionService."""
        if run_record is not None:
            if run_record.id != run_id:
                raise RuntimeError(f"Status snapshot run_id mismatch: expected {run_id}, got {run_record.id}")
            run = run_record
        else:
            run = await self._session_service.get_run(run_id)
        event_key = str(run_id)
        with self._shutdown_events_lock:
            event = self._shutdown_events[event_key] if event_key in self._shutdown_events else None
        cancel_requested = event is not None and event.is_set() and run.status not in SESSION_TERMINAL_RUN_STATUS_VALUES
        return RunStatusResponse(
            run_id=str(run.id),
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            accounting=accounting,
            error=run.error,
            landscape_run_id=run.landscape_run_id,
            cancel_requested=cancel_requested,
        )

    async def validate(self, session_id: UUID, *, user_id: str | None = None) -> ValidationResult:
        """Dry-run validation using real engine code paths.

        Wraps the sync validate_pipeline() call via run_in_executor
        to avoid blocking the event loop (AC #16).

        Args:
            session_id: Session whose current state to validate.
            user_id: Authenticated user's ID for scoped secret ref validation.
        """
        state_record = await self._session_service.get_current_state(session_id)
        if state_record is None:
            return ValidationResult(
                is_valid=False,
                checks=[
                    ValidationCheck(
                        name="state_exists",
                        passed=False,
                        detail="No composition state exists for this session",
                        affected_nodes=(),
                        outcome_code=None,
                    )
                ],
                errors=[
                    ValidationError(
                        component_id=None,
                        component_type=None,
                        message="No composition state exists for this session",
                        suggestion="Use the composer to build a pipeline first.",
                        error_code=None,
                    )
                ],
                readiness=ValidationReadiness(
                    authoring_valid=False,
                    execution_ready=False,
                    completion_ready=False,
                    blockers=[
                        ValidationReadinessBlocker(
                            code="state_exists",
                            component_id=None,
                            component_type=None,
                            detail="No composition state exists for this session.",
                        )
                    ],
                ),
            )

        composition_state = state_from_record(state_record)
        return await self.validate_state(composition_state, user_id=user_id, session_id=session_id)

    async def validate_state(
        self,
        state: CompositionState,
        *,
        user_id: str | None = None,
        session_id: UUID | None = None,
    ) -> ValidationResult:
        """Dry-run validation for an already-read composition state.

        Snapshot-style callers use this to keep every projected row on the
        same ``CompositionState.version`` instead of re-reading mutable session
        state between adjacent readiness calculations. When supplied,
        ``session_id`` scopes inline-blob metadata lookups to the same session
        boundary enforced by ``link_blob_to_run()`` at execution time.
        """
        from functools import partial

        from elspeth.web.execution.validation import validate_pipeline

        def _blob_get_metadata(blob_id: UUID) -> BlobRecord | None:
            if self._blob_service is None:
                return None
            try:
                record = self._call_async(self._blob_service.get_blob(blob_id))
            except BlobNotFoundError:
                return None
            if session_id is not None and record.session_id != session_id:
                return None
            return record

        return cast(
            ValidationResult,
            await run_sync_in_worker(
                partial(
                    validate_pipeline,
                    state,
                    self._settings,
                    self._yaml_generator,
                    secret_service=self._secret_service,
                    user_id=user_id,
                    blob_get_metadata=_blob_get_metadata,
                ),
            ),
        )

    async def verify_run_ownership(self, user: UserIdentity, run_id: str) -> bool:
        """Verify that a run belongs to the authenticated user's session.

        Used by the WebSocket handler for IDOR protection. Checks both
        user_id and auth_provider_type to prevent cross-provider access
        when user_id namespaces overlap between providers.
        """
        run = await self._session_service.get_run(UUID(run_id))
        session = await self._session_service.get_session(run.session_id)
        return session.user_id == user.user_id and session.auth_provider_type == self._settings.auth_provider

    async def cancel(self, run_id: UUID) -> None:
        """Cancel a run via the shutdown Event.

        Active runs: sets the Event, Orchestrator detects during row processing.
        Pending runs (no Event registered yet): marks the run as cancelled
        directly via SessionService so _run_pipeline terminates immediately.
        Terminal runs: no-op (idempotent).

        Async because the pending-run path awaits SessionService (we're in
        the event loop thread, not the background thread).
        """
        with self._shutdown_events_lock:
            event = self._shutdown_events.get(str(run_id))
        if event is not None:
            event.set()
        else:
            # No event means either pending (not yet started) or already done
            run = await self._session_service.get_run(run_id)
            if run.status not in SESSION_TERMINAL_RUN_STATUS_VALUES:
                await self._session_service.update_run_status(run_id, status="cancelled")

    # ── Background Thread ──────────────────────────────────────────────

    def _run_pipeline(
        self,
        run_id: str,
        pipeline_yaml: str,
        shutdown_event: threading.Event,
        user_id: str | None = None,
        auth_provider_type: str | None = None,
    ) -> None:
        """Execute a pipeline in the background thread.

        B7 fix: Wrapped in try/except BaseException/finally.
        - except BaseException: Updates run to failed, re-raises.
        - finally: Removes shutdown event from _shutdown_events.

        B2 fix: shutdown_event is ALWAYS passed to orchestrator.run().
        B3 fix: LandscapeDB and PayloadStore from WebSettings resolvers.

        Secret resolution: If secret_service and user_id are available,
        resolves {"secret_ref": "NAME"} patterns in the YAML config before
        loading settings. Resolved values exist only in the worker thread's
        local memory — never persisted.
        """
        landscape_db: LandscapeDB | None = None
        rate_limit_registry: Any | None = None
        telemetry_manager: Any | None = None
        run_uuid = UUID(run_id)
        try:
            # Early shutdown check: if cancel()/shutdown() fired before we
            # start setup, skip the expensive LandscapeDB/plugin/graph work.
            if shutdown_event.is_set():
                self._finalize_output_blobs(run_id, success=False)
                self._call_async(self._session_service.update_run_status(run_uuid, status="cancelled"))
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=UTC),
                        event_type="cancelled",
                        data=CancelledData(
                            source_rows_processed=0,
                            tokens_succeeded=0,
                            tokens_failed=0,
                            tokens_quarantined=0,
                            tokens_routed_success=0,
                            tokens_routed_failure=0,
                        ),
                    ),
                )
                return

            # B8/C1: SessionService is async — bridge from background thread.
            # Cancelled-race recovery: catch only the narrow subclass.  See
            # IllegalRunTransitionError docstring for why bare ValueError must
            # propagate (Tier-1 invariant breaches must not be masked).
            try:
                self._call_async(self._session_service.update_run_status(run_uuid, status="running", landscape_run_id=run_id))
            except IllegalRunTransitionError:
                current = self._call_async(self._session_service.get_run(run_uuid))
                if current.status == "cancelled":
                    self._finalize_output_blobs(run_id, success=False)
                    self._broadcaster.broadcast(
                        run_id,
                        RunEvent(
                            run_id=run_id,
                            timestamp=datetime.now(tz=UTC),
                            event_type="cancelled",
                            data=CancelledData(
                                source_rows_processed=0,
                                tokens_succeeded=0,
                                tokens_failed=0,
                                tokens_quarantined=0,
                                tokens_routed_success=0,
                                tokens_routed_failure=0,
                            ),
                        ),
                    )
                    return
                raise

            # B3 fix: construct from WebSettings, not hardcoded paths
            # NOTE: LandscapeDB is constructed per-run, not shared. This is safe
            # with max_workers=1 (no concurrent access) but wasteful — each run
            # creates a new SQLAlchemy engine. Acceptable for MVP; consider
            # sharing a single instance if profiling shows connection overhead.
            landscape_db = LandscapeDB(
                connection_string=self._settings.get_landscape_url(),
                passphrase=self._settings.landscape_passphrase,
            )
            payload_store = FilesystemPayloadStore(base_path=self._settings.get_payload_store_path())

            # Resolve secret refs before writing YAML to temp file.
            # Resolved values exist only in this thread's local memory — the
            # original pipeline_yaml (persisted in the Run record) is untouched.
            resolved_yaml = pipeline_yaml
            resolved_dict: dict[str, Any] | None = None
            secret_resolution_inputs: list[SecretResolutionInput] = []
            inline_refs: list[BlobInlineRef] = []
            inline_blob_candidate = "blob_ref" in pipeline_yaml and "inline_content" in pipeline_yaml
            needs_config_tree = (self._secret_service is not None and user_id is not None) or inline_blob_candidate
            if needs_config_tree:
                import yaml as _yaml

                config_dict = _yaml.safe_load(pipeline_yaml)
                if type(config_dict) is not dict:
                    raise TypeError(
                        f"generate_yaml() produced non-dict YAML (got {type(config_dict).__name__}) — this is a bug in the YAML generator"
                    )
                resolved_dict = cast(dict[str, Any], config_dict)

                if self._secret_service is not None and user_id is not None:
                    from elspeth.core.secrets import resolve_secret_refs

                    env_ref_names = {item.name for item in self._secret_service.list_refs(user_id)}
                    resolved_dict, resolutions = resolve_secret_refs(
                        resolved_dict,
                        self._secret_service,
                        user_id,
                        env_ref_names=env_ref_names,
                    )

                    # Map ResolvedSecret.scope (web domain) to
                    # SecretResolutionInput.source (audit domain).
                    # "server" secrets are env vars on the host → audit source "env".
                    _SCOPE_TO_AUDIT_SOURCE: dict[str, str] = {
                        "user": "user",
                        "server": "env",
                    }
                    for rs in resolutions:
                        # Offensive lookup against our own code-controlled
                        # mapping: rs.scope is a typed SecretScope Literal, so
                        # an unmapped scope (e.g. a valid "org" with no audit
                        # mapping added yet) is a programmer coverage gap, not
                        # external absence. Direct subscript surfaces it as a
                        # KeyError, re-raised as a meaningful ValueError.
                        try:
                            audit_source = _SCOPE_TO_AUDIT_SOURCE[rs.scope]
                        except KeyError as exc:
                            raise ValueError(
                                f"No audit source mapping for secret scope {rs.scope!r} "
                                f"(secret: {rs.name!r}) — add mapping to _SCOPE_TO_AUDIT_SOURCE"
                            ) from exc
                        secret_resolution_inputs.append(
                            SecretResolutionInput(
                                env_var_name=rs.name,
                                source=audit_source,
                                vault_url=None,
                                secret_name=None,
                                timestamp=time.time(),
                                resolution_latency_ms=0.0,
                                fingerprint=rs.fingerprint,
                            )
                        )

                inline_refs = _discover_blob_content_refs(resolved_dict) if inline_blob_candidate else []
                if inline_refs:
                    if self._blob_service is None:
                        raise RuntimeError("Inline-content blob refs require BlobServiceProtocol wiring")
                    blob_service = self._blob_service

                    unique_blob_ids: list[UUID] = []
                    seen_blob_ids: set[UUID] = set()
                    for ref in inline_refs:
                        if ref.blob_id in seen_blob_ids:
                            continue
                        seen_blob_ids.add(ref.blob_id)
                        unique_blob_ids.append(ref.blob_id)

                    async def _link_inline_blobs_to_run() -> None:
                        await asyncio.gather(
                            *(
                                blob_service.link_blob_to_run(
                                    blob_id=blob_id,
                                    run_id=run_uuid,
                                    direction="input",
                                )
                                for blob_id in unique_blob_ids
                            )
                        )

                    try:

                        async def _gather_inline_blob_metadata() -> list[Any]:
                            return await asyncio.gather(*(blob_service.get_blob(blob_id) for blob_id in unique_blob_ids))

                        metadata_records = self._call_async(_gather_inline_blob_metadata())
                        records_by_blob_id: dict[UUID, BlobRecord] = {
                            blob_id: cast(BlobRecord, record) for blob_id, record in zip(unique_blob_ids, metadata_records, strict=True)
                        }
                        _enforce_blob_content_ref_metadata(
                            inline_refs,
                            records_by_blob_id,
                            per_ref_byte_cap=BLOB_INLINE_PER_REF_BYTE_CAP,
                            aggregate_byte_cap=BLOB_INLINE_AGGREGATE_BYTE_CAP,
                        )
                        self._call_async(_link_inline_blobs_to_run())
                        fetched = self._call_async(_fetch_blob_contents(blob_service, inline_refs))
                        blob_metadata: dict[UUID, tuple[AllowedMimeType, int]] = {
                            blob_id: (cast(AllowedMimeType, record.mime_type), record.size_bytes)
                            for blob_id, record in zip(unique_blob_ids, metadata_records, strict=True)
                        }
                        resolved_dict, blob_resolutions = _substitute_blob_content_refs(
                            resolved_dict,
                            fetched,
                            refs=inline_refs,
                            blob_metadata=blob_metadata,
                        )
                    except BlobIntegrityError:
                        _BLOB_INLINE_HASH_MISMATCH_TOTAL.add(1, {"run_id": run_id})
                        raise
                    self._call_async(
                        self._session_service.record_blob_inline_resolutions(
                            run_id=run_uuid,
                            resolutions=blob_resolutions,
                            attempt=1,
                        )
                    )

                if secret_resolution_inputs or inline_refs:
                    resolved_yaml = _yaml.dump(resolved_dict, default_flow_style=False)

            # Load settings from YAML string — never write resolved secrets
            # to disk.  load_settings_from_yaml_string() parses in-process,
            # bypassing Dynaconf file I/O. Disable server env expansion when
            # inline blobs were substituted: blob bytes are user-authored and
            # must not resolve host ``${VAR}`` values after secret controls.
            settings = load_settings_from_yaml_string(resolved_yaml, expand_env_vars=not inline_refs)
            runtime_graph = build_validated_runtime_graph(settings)
            bundle = runtime_graph.plugin_bundle
            graph = runtime_graph.graph

            # Fold aggregations into transforms, assemble PipelineConfig, and
            # run the four orchestrator route-target validators. The
            # orchestrator runs these validators again at run-init
            # (engine/orchestrator/core.py:1746-1777). The validators are pure
            # and idempotent — calling them here surfaces dangling-reference
            # errors before any rows flow, with a cleaner error surface, and
            # closes the composer/runtime parity gap (issue elspeth-127de6865a).
            pipeline_config = assemble_and_validate_pipeline_config(
                source=bundle.source,
                transforms=bundle.transforms,
                sinks=bundle.sinks,
                aggregations=bundle.aggregations,
                settings=settings,
                graph=graph,
            )

            # Set up EventBus to bridge ProgressEvent -> RunEvent -> broadcaster.
            # _to_run_event is a pure mapping (system code) — let it crash.
            # broadcast() uses call_soon_threadsafe → RuntimeError if the
            # event loop is closed during shutdown.  Only catch that specific
            # infrastructure failure; let programmer bugs (TypeError, etc.) crash.
            def _safe_broadcast(evt: ProgressEvent) -> None:
                run_event = self._to_run_event(run_id, evt)
                try:
                    self._broadcaster.broadcast(run_id, run_event)
                except RuntimeError as broadcast_err:
                    # call_soon_threadsafe raises RuntimeError when the
                    # event loop is closed — expected during shutdown.
                    # Log the class name, not the message: the canonical
                    # CPython wording ("Event loop is closed") is not a
                    # stable contract and future interpreter versions may
                    # reword it.  ``exc_class`` is the diagnostic token
                    # every other site in this module uses.
                    slog.error(
                        "progress_broadcast_failed",
                        run_id=run_id,
                        exc_class=type(broadcast_err).__name__,
                    )

            event_bus = EventBus()
            event_bus.subscribe(ProgressEvent, _safe_broadcast)

            # Match the CLI run path's runtime infrastructure. External-call
            # plugins such as web_scrape require a RateLimitRegistry during
            # on_start(), and the orchestrator also consumes runtime
            # concurrency/checkpoint/telemetry configs.
            from elspeth.contracts.config.runtime import (
                RuntimeCheckpointConfig,
                RuntimeConcurrencyConfig,
                RuntimeRateLimitConfig,
                RuntimeTelemetryConfig,
            )
            from elspeth.core.checkpoint import CheckpointManager
            from elspeth.core.rate_limit import RateLimitRegistry
            from elspeth.telemetry import create_telemetry_manager

            rate_limit_config = RuntimeRateLimitConfig.from_settings(settings.rate_limit)
            concurrency_config = RuntimeConcurrencyConfig.from_settings(settings.concurrency)
            checkpoint_config = RuntimeCheckpointConfig.from_settings(settings.checkpoint)
            telemetry_config = RuntimeTelemetryConfig.from_settings(settings.telemetry)

            rate_limit_registry = RateLimitRegistry(rate_limit_config)
            telemetry_manager = create_telemetry_manager(telemetry_config)
            checkpoint_manager = CheckpointManager(landscape_db) if checkpoint_config.enabled else None

            orchestrator = Orchestrator(
                db=landscape_db,
                event_bus=event_bus,
                rate_limit_registry=rate_limit_registry,
                concurrency_config=concurrency_config,
                checkpoint_manager=checkpoint_manager,
                checkpoint_config=checkpoint_config,
                telemetry_manager=telemetry_manager,
            )

            # B2 fix: ALWAYS pass shutdown_event — suppresses signal handler
            # installation from background thread (Python forbids
            # signal.signal() from non-main threads)
            from elspeth.cli_helpers import _make_sink_factory

            # Read the boot-time catalog snapshot. Direct attribute access
            # (offensive programming): if the lifespan never called
            # ``set_openrouter_catalog_snapshot()`` these are ``None`` and
            # the assertions below crash loudly, surfacing the wiring bug
            # rather than silently writing a NULL audit field.
            catalog_sha = self._openrouter_catalog_sha256
            catalog_source = self._openrouter_catalog_source
            if catalog_sha is None or catalog_source is None:
                raise RuntimeError(
                    "ExecutionServiceImpl has no OpenRouter catalog snapshot. "
                    "set_openrouter_catalog_snapshot() must be called from the "
                    "lifespan before any pipeline executes; this is a wiring bug."
                )

            result = orchestrator.run(
                pipeline_config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
                secret_resolutions=secret_resolution_inputs or None,
                shutdown_event=shutdown_event,  # B2: NEVER omit this
                sink_factory=_make_sink_factory(settings),
                run_id=run_id,
                initiated_by_user_id=user_id,
                auth_provider_type=auth_provider_type,
                openrouter_catalog_sha256=catalog_sha,
                openrouter_catalog_source=catalog_source,
            )

            # Orchestrator.run() returns normally ONLY on completion.
            # If shutdown was requested, it raises GracefulShutdownError
            # (caught below). Do NOT check shutdown_event.is_set() here —
            # cancel() can set the event after processing finishes but
            # before we persist status, causing a completed run to be
            # misclassified as cancelled.

            # Persist the terminal run status before success-finalizing
            # output blobs. If the DB transition loses a race to an
            # external cancellation, we must never expose ready outputs
            # for a cancelled run.
            #
            # Phase 2.2 (elspeth-0de989c56d): the orchestrator's RunResult
            # carries the engine-decided four-value terminal status
            # (completed / completed_with_failures / failed / empty); pass
            # it verbatim to the session-runs DB so an operator reading
            # ``/api/runs/{rid}`` sees the same verdict the engine wrote
            # to Landscape.  The legacy hard-coded ``status="completed"``
            # collapsed S1A's reproducer ("0 succeeded, 6 routed via
            # on_error") into a clean-completion label.
            session_status = _session_status_from_run_result_status(result.status)
            # FAILED-from-row-shape (engine returned normally with
            # rows_succeeded==0) has no exception to surface.  Provide a
            # structural error message that satisfies the
            # ``failed-requires-error`` audit invariant while remaining
            # operator-readable and free of secret/row content.
            session_error: str | None = None
            if result.status in (RunStatus.FAILED, RunStatus.COMPLETED_WITH_FAILURES):
                # Enrich the structural message with the top distinct per-row
                # failures so the runs view shows the dominant cause inline.
                # Tier-1 read. Narrowed to the transient infrastructure
                # failure modes (DB-layer errors, disk/filesystem errors)
                # only: run-status persistence is more important than the
                # optional sample enrichment, so a transient audit-system
                # degradation degrades to the bare structural message (still
                # satisfying the failed-requires-error invariant) and is
                # recorded via the slog warning (audit-system failure
                # exemption per CLAUDE.md logging-telemetry-policy).
                # Malformed audit JSON (json.JSONDecodeError, a ValueError
                # subclass raised by load_top_failure_samples) is Tier-1
                # audit-data corruption and is DELIBERATELY not caught — it
                # must crash per the tier model, not be silently degraded.
                samples_text = ""
                if landscape_db is not None:
                    try:
                        samples = load_top_failure_samples(landscape_db, result.run_id)
                        samples_text = format_failure_samples(samples)
                    except (SQLAlchemyError, OSError):
                        slog.warning(
                            "failure_sample_enrichment_failed",
                            run_id=run_id,
                            landscape_run_id=result.run_id,
                            exc_info=True,
                        )
                if result.status == RunStatus.FAILED:
                    session_error = _structural_failure_message(
                        rows_processed=result.rows_processed,
                        failure_samples=samples_text,
                    )
                else:
                    # RunRecord invariant (sessions/protocol.py:237-238) permits
                    # error on COMPLETED_WITH_FAILURES; only FAILED *requires* it.
                    session_error = _partial_completion_message(
                        rows_succeeded=result.rows_succeeded,
                        rows_failed=result.rows_failed,
                        rows_routed_failure=result.rows_routed_failure,
                        rows_quarantined=result.rows_quarantined,
                        failure_samples=samples_text,
                    )
            # Cancelled-race recovery: catch only the narrow subclass.  See
            # IllegalRunTransitionError docstring for why bare ValueError must
            # propagate (Tier-1 invariant breaches must not be masked).
            try:
                self._call_async(
                    self._session_service.update_run_status(
                        run_uuid,
                        status=session_status,
                        error=session_error,
                        rows_processed=result.rows_processed,
                        rows_succeeded=result.rows_succeeded,
                        rows_failed=result.rows_failed,
                        rows_routed_success=result.rows_routed_success,
                        rows_routed_failure=result.rows_routed_failure,
                        rows_quarantined=result.rows_quarantined,
                    )
                )
            except IllegalRunTransitionError:
                current = self._call_async(self._session_service.get_run(run_uuid))
                if current.status == "cancelled":
                    slog.warning(
                        "run_completed_but_externally_cancelled",
                        run_id=run_id,
                        landscape_run_id=result.run_id,
                        rows_processed=result.rows_processed,
                        rows_failed=result.rows_failed,
                    )
                    self._finalize_output_blobs(run_id, success=False)
                    self._broadcaster.broadcast(
                        run_id,
                        RunEvent(
                            run_id=run_id,
                            timestamp=datetime.now(tz=UTC),
                            event_type="cancelled",
                            data=CancelledData(
                                source_rows_processed=result.rows_processed,
                                tokens_succeeded=result.rows_succeeded,
                                tokens_failed=result.rows_failed,
                                tokens_quarantined=result.rows_quarantined,
                                tokens_routed_success=result.rows_routed_success,
                                tokens_routed_failure=result.rows_routed_failure,
                            ),
                        ),
                    )
                    return
                raise

            # Finalize blobs after the authoritative completion transition
            # succeeds, but before broadcasting the completed terminal event.
            # Finalization failures are logged in _finalize_output_blobs()
            # and must not trigger a second terminal event.
            #
            # Phase 2.2 (elspeth-0de989c56d): treat the four operator-
            # completion statuses (the run reached engine completion and
            # produced output) as success for blob finalization.  The
            # FAILED-from-row-shape case is "engine ran, no row succeeded"
            # — the partial output that exists is still legitimate
            # evidence (e.g. quarantine sink contents), so finalize as
            # success=False to keep the failure-track outputs distinct
            # from clean-completion outputs in the blob lifecycle.
            self._finalize_output_blobs(run_id, success=(result.status != RunStatus.FAILED))

            if result.status == RunStatus.FAILED:
                # Engine returned normally but no row reached success.  Emit
                # the operator-visible failure event so the frontend can
                # render the structural failure mode (S1A / S1B-msg2 shape).
                # session_error is unconditionally populated by the FAILED
                # branch above; assert the structural invariant offensively
                # rather than re-deriving silently via
                # ``or _structural_failure_message(...)``.  A regenerated
                # message would mask a structural inconsistency between
                # the audit-side and SSE-side failure detail.
                assert session_error is not None, (
                    "Tier-1 invariant: session_error must be populated when result.status == RunStatus.FAILED (see the FAILED branch above)"
                )
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=UTC),
                        event_type="failed",
                        data=FailedData(
                            detail=session_error,
                            node_id=None,
                        ),
                    ),
                )
            else:
                if landscape_db is None:
                    raise RuntimeError("Tier-1 invariant: completed run has no open LandscapeDB for accounting projection")
                accounting = load_run_accounting_from_db(landscape_db, landscape_run_id=result.run_id)
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=UTC),
                        event_type="completed",
                        data=CompletedData(
                            # session_status is the engine-decided four-value
                            # operator-completion classification (completed /
                            # completed_with_failures / empty); pass it
                            # verbatim so the SSE event carries the same
                            # status the run-status DB row has.  Frontend
                            # MUST NOT re-derive from row counts.
                            #
                            # cast is sound: in the else branch of
                            # ``if result.status == RunStatus.FAILED`` we know
                            # result.status ∈ {COMPLETED, COMPLETED_WITH_FAILURES,
                            # EMPTY} (RUNNING/INTERRUPTED raise in the mapper),
                            # so session_status ∈ {"completed",
                            # "completed_with_failures", "empty"}.  Pydantic
                            # offensively re-validates the narrow Literal.
                            status=cast(
                                Literal["completed", "completed_with_failures", "empty"],
                                session_status,
                            ),
                            accounting=accounting,
                            landscape_run_id=result.run_id,
                        ),
                    ),
                )

        except GracefulShutdownError as gse:
            # Orchestrator detected shutdown during processing and raised
            # after flushing in-progress work. Finalize → status → broadcast.
            self._finalize_output_blobs(run_id, success=False)
            self._call_async(
                self._session_service.update_run_status(
                    run_uuid,
                    status="cancelled",
                    rows_processed=gse.rows_processed,
                    rows_succeeded=gse.rows_succeeded,
                    rows_failed=gse.rows_failed,
                    rows_routed_success=gse.rows_routed_success,
                    rows_routed_failure=gse.rows_routed_failure,
                    rows_quarantined=gse.rows_quarantined,
                )
            )
            self._broadcaster.broadcast(
                run_id,
                RunEvent(
                    run_id=run_id,
                    timestamp=datetime.now(tz=UTC),
                    event_type="cancelled",
                    data=CancelledData(
                        source_rows_processed=gse.rows_processed,
                        tokens_succeeded=gse.rows_succeeded,
                        tokens_failed=gse.rows_failed,
                        tokens_quarantined=gse.rows_quarantined,
                        tokens_routed_success=gse.rows_routed_success,
                        tokens_routed_failure=gse.rows_routed_failure,
                    ),
                ),
            )

        except BaseException as exc:
            # B7 fix: Catch BaseException (not Exception) to handle
            # KeyboardInterrupt, SystemExit, and OOM-triggered exceptions.
            # Without this, the Run record stays in 'running' forever.

            # Finalize blobs first — before any terminal event surfaces.
            self._finalize_output_blobs(run_id, success=False)

            client_msg = _sanitize_error_for_client(exc)
            if type(exc) is PydanticValidationError:
                slog.error(
                    "run_schema_contract_violation",
                    run_id=run_id,
                    exc_class=type(exc).__name__,
                    error_count=exc.error_count(),
                    schema_errors=_schema_contract_violation_errors(exc),
                )

            # elspeth-879f6de6bd: when an exception fires AFTER the success
            # path has already committed a terminal status (post-completion
            # broadcast crash, audit-write failure, telemetry exhaustion,
            # OOM mid-finalize, etc.), update_run_status(status="failed", ...)
            # raises ValueError because LEGAL_RUN_TRANSITIONS makes every
            # terminal status outgoing-empty.  The pre-fix recovery let that
            # ValueError escape, losing the original ``exc`` into __context__.
            #
            # Recovery has three branches keyed on what we can learn about the
            # audit row's current status:
            #
            #   1. Probe succeeds, row is terminal → skip the illegal status
            #      update AND the ``"failed"`` SSE broadcast (which would
            #      otherwise contradict the audit row's true terminal status —
            #      audit primacy).
            #
            #   2. Probe succeeds, row is non-terminal → fall through to the
            #      normal ``update_run_status("failed", ...)`` recovery
            #      (e.g. a crash mid-orchestration with the row still in
            #      ``running``).
            #
            #   3. Probe fails (SQLAlchemyError / OSError) AND the row is
            #      actually terminal → fall-through update_run_status raises
            #      IllegalRunTransitionError, caught narrowly below.  IRTE
            #      carries ``current_status``, which is the validator's read
            #      of ground truth immediately before it rejected the
            #      transition — so IRTE is in-band proof of terminality and
            #      promotes us into branch 1's audit-primacy stance (suppress
            #      the failed-SSE broadcast).  This is the resolution of the
            #      ambiguity that ``post_exception_run_state_probe_failed``
            #      logged above.
            #
            # Probe is non-signal-only: signals (KeyboardInterrupt, SystemExit)
            # mean the event loop is shutting down — async calls including
            # get_run() are unsafe.  Signal-path broadcast preserves the R6/B7
            # shape; if a signal fires mid-broadcast against a completed run
            # the SSE may diverge from audit, but the event loop is closing
            # and consumers likely won't receive the event regardless.  A
            # narrower fix for that asymmetry is tracked separately if
            # observed in the wild.
            run_already_terminal = False
            if not isinstance(exc, (KeyboardInterrupt, SystemExit)):
                try:
                    current_run = self._call_async(self._session_service.get_run(run_uuid))
                    current_status = current_run.status
                    if current_status in SESSION_TERMINAL_RUN_STATUS_VALUES:
                        run_already_terminal = True
                except (SQLAlchemyError, OSError) as probe_err:
                    # Narrow catch — mirrors the sibling pattern at the
                    # ``update_run_status`` recovery below (commits
                    # b8ba2214/127417cb).  Audit-system *degradation*
                    # (SQLAlchemyError, OSError) falls through to the
                    # best-effort recovery so we don't make the probe-failure
                    # scenario worse than today; slog here is policy-correct
                    # per logging-telemetry-policy because this IS an
                    # audit-system-degradation case.
                    #
                    # ValueError is deliberately NOT caught.  The ValueErrors
                    # ``get_run`` can raise — "Run not found" (the row vanished
                    # mid-run), ``UUID(row.id)`` malformed, non-UTC datetimes
                    # via ``_ensure_utc`` — are all Tier 1 audit-data
                    # corruption.  Per CLAUDE.md tier model, Tier 1 invariant
                    # violations MUST crash immediately; absorbing them here
                    # would silently log audit corruption while the recovery
                    # path falls through to ``update_run_status`` which would
                    # encounter the same corruption anyway.  ``RunRecord``'s
                    # explicit invariant breaches surface as
                    # ``AuditIntegrityError(Exception)`` (not ValueError) and
                    # are likewise — correctly — uncaught here.
                    slog.error(
                        "post_exception_run_state_probe_failed",
                        run_id=run_id,
                        original_exc_class=type(exc).__name__,
                        probe_exc_class=type(probe_err).__name__,
                    )

                if run_already_terminal:
                    # Post-audit-terminal exception path.  The run already
                    # transitioned to a terminal status before this exception
                    # fired (e.g. broadcast crashed AFTER update_run_status
                    # succeeded), so the audit DB carries the truthful
                    # operator-visible outcome and there is nothing more to
                    # record on the audit side.
                    #
                    # We deliberately do NOT slog here.  Per
                    # ``logging-telemetry-policy`` the logger is not for
                    # post-audit operational signal — the SRE-discoverable
                    # surface for this scenario is already two existing
                    # channels:
                    #   1. The audit ``runs`` row (queryable by run_id) —
                    #      carries the truthful terminal status the run
                    #      reached before the post-audit exception.
                    #   2. ``_on_pipeline_done``'s safety-net slog
                    #      (``pipeline_done_callback_exception``) — fires
                    #      against the re-raised ``exc`` once the Future
                    #      completes, with the same class-name chain we
                    #      would otherwise have walked here.
                    # Together these give an SRE the post-audit signal
                    # (correlate the two by run_id) without violating
                    # audit primacy at this site.  Adding a third slog
                    # at this site would be operational noise that
                    # duplicates the safety-net log without contributing
                    # signal beyond the audit row.
                    pass
                else:
                    try:
                        self._call_async(self._session_service.update_run_status(run_uuid, status="failed", error=client_msg))
                    except IllegalRunTransitionError as irte:
                        # elspeth-879f6de6bd recovery branch 3 (probe-failed
                        # AND row-actually-terminal — see the comment block
                        # at the top of this BaseException handler).
                        #
                        # IRTE here is the mechanical artefact of attempting
                        # ``failed`` against a row whose true status is
                        # terminal; ``irte.current_status`` is the validator's
                        # authoritative read of that row, taken inside
                        # SessionService.update_run_status immediately before
                        # it rejected the transition.  That gives us the
                        # ground truth the probe couldn't establish, so we
                        # promote into the audit-primacy stance (suppress the
                        # ``failed`` SSE that would otherwise contradict the
                        # audit row).
                        #
                        # This is a distinct semantic from the established
                        # IllegalRunTransitionError catches at the
                        # ``"running"`` and terminal transitions earlier in
                        # this method (see those sites' ``Cancelled-race
                        # recovery`` comments and the IRTE class docstring):
                        # those handle the cancelled-race artefact; this one
                        # handles the post-completion + probe-failure
                        # artefact.  The IRTE docstring's narrow-subclass
                        # sanction ("never bare ValueError — Tier-1 invariant
                        # breaches must propagate") applies the same way to
                        # both: the other four ValueErrors update_run_status
                        # can raise (run-not-found, landscape_run_id
                        # overwrite, completed-without-landscape,
                        # failed-without-error) are not subclasses of IRTE
                        # and remain uncaught here.
                        #
                        # Offensive Tier-1 assert: if IRTE fires for a
                        # non-terminal current_status, the SessionService
                        # validator has a bug (it shouldn't reject a
                        # non-terminal → failed transition).  Reraise so the
                        # validator regression surfaces rather than being
                        # silently absorbed; the original ``exc`` remains on
                        # ``__context__`` via Python's implicit chaining and
                        # the SRE-discoverable surface is identical to the
                        # pre-fix bug — the right tradeoff for a validator
                        # regression.
                        if irte.current_status not in SESSION_TERMINAL_RUN_STATUS_VALUES:
                            raise
                        run_already_terminal = True
                        slog.error(
                            "post_exception_recovery_aborted_run_terminal",
                            run_id=run_id,
                            original_exc_class=type(exc).__name__,
                            irte_current_status=irte.current_status,
                        )
                    except (SQLAlchemyError, OSError) as status_err:
                        # Narrow catch (canonical pattern, commits b8ba2214/127417cb):
                        # SQLAlchemyError family + OSError only. Programmer bugs in
                        # update_run_status must propagate so they don't masquerade
                        # as a transient status-update failure.  exc_class only —
                        # ``str(status_err)`` can surface SQL + bound parameters +
                        # ``__cause__`` credentials, and ``client_msg`` is already
                        # the sanitized form of ``exc`` (see _sanitize_error_for_client
                        # above), so re-logging it as ``original_error`` gives no
                        # extra triage surface beyond the class name.
                        slog.error(
                            "run_status_update_failed_in_except",
                            run_id=run_id,
                            original_exc_class=type(exc).__name__,
                            status_update_exc_class=type(status_err).__name__,
                        )
            else:
                slog.warning(
                    "skipping_status_update_on_signal",
                    run_id=run_id,
                    exc_type=type(exc).__name__,
                )

            # Broadcast a "failed" SSE event ONLY when the audit row isn't
            # already terminal.  Broadcasting "failed" against a "completed"
            # audit row would tell SSE consumers the opposite of audit truth
            # and violate the audit-primacy constraint in CLAUDE.md.
            # Re-emitting the *correct* terminal SSE event for consumer
            # continuity is a separate UX improvement.
            if not run_already_terminal:
                self._broadcaster.broadcast(
                    run_id,
                    RunEvent(
                        run_id=run_id,
                        timestamp=datetime.now(tz=UTC),
                        event_type="failed",
                        data=FailedData(detail=client_msg, node_id=None),
                    ),
                )
            raise
        finally:
            # Always clean up, regardless of success or failure
            with self._shutdown_events_lock:
                # Idempotent cleanup of an internal bookkeeping key. Access it
                # directly (R9 remediation); the membership guard preserves the
                # silent no-op when cleanup races or runs twice, and matches the
                # sibling submit-failure cleanup above. run_id is contractually
                # registered (str key, synchronous, before submit) so a bare
                # pop-with-default would mask a broken invariant.
                if run_id in self._shutdown_events:
                    del self._shutdown_events[run_id]
            if landscape_db is not None:
                landscape_db.close()
            if rate_limit_registry is not None:
                rate_limit_registry.close()
            if telemetry_manager is not None:
                telemetry_manager.close()
            self._broadcaster.cleanup_run(run_id)

    # Exceptions that can escape finalize_run_output_blobs itself
    # (not per-blob errors, which are captured in the result).
    # Covers: initial query failure (SQLAlchemyError), any OS-level
    # failure outside the per-blob loop (OSError), and blob lifecycle
    # errors from the service layer.  BlobStateError is belt-and-
    # suspenders — caught per-blob inside the service, but included
    # here in case of a code path change.
    #
    # RuntimeError deliberately excluded — too broad.  It would
    # suppress Tier 1 anomaly signals, asyncio errors, and recursion
    # failures.  If the blob service needs a "vanished mid-transaction"
    # signal, it should raise BlobNotFoundError or BlobStateError.
    _FINALIZE_SUPPRESSED: tuple[type[BaseException], ...] = (
        OSError,
        SQLAlchemyError,
        BlobNotFoundError,
        BlobQuotaExceededError,
        BlobStateError,
    )

    def _finalize_output_blobs(self, run_id: str, *, success: bool) -> None:
        """Finalize pending output blobs after a run completes/fails/cancels.

        Uses _call_async to bridge from the background thread to the async
        blob service. Failure here must not mask the original run outcome —
        errors are logged, not raised. Programmer bugs (TypeError,
        AttributeError) are deliberately not caught.
        """
        if self._blob_service is None:
            return
        try:
            result = self._call_async(
                self._blob_service.finalize_run_output_blobs(
                    UUID(run_id),
                    success=success,
                )
            )
            if result.errors:
                slog.error(
                    "blob_finalization_partial_failure",
                    run_id=run_id,
                    success=success,
                    finalized_count=len(result.finalized),
                    error_count=len(result.errors),
                    errors=[{"blob_id": str(e.blob_id), "exc_type": e.exc_type} for e in result.errors],
                )
        except self._FINALIZE_SUPPRESSED as blob_err:
            slog.error(
                "blob_finalization_failed",
                run_id=run_id,
                success=success,
                exc_type=type(blob_err).__name__,
            )

    def _on_pipeline_done(self, future: Future[None]) -> None:
        """B7 Layer 2: Safety net callback.

        Fires when the Future completes. Retrieves (and suppresses) any
        exception so the thread pool doesn't log it to stderr.

        Normal case: _run_pipeline() already recorded the error to the
        audit trail (runs.error) — no duplicate logging needed.

        Edge case: if _run_pipeline's own except-BaseException handler
        failed (e.g. update_run_status raised), the audit trail write
        never completed. In that case this callback is the ONLY place
        the failure surfaces, so we log as a last-resort safety net.
        """
        exc = future.exception()
        if exc is not None and not isinstance(exc, (KeyboardInterrupt, SystemExit)):
            # _run_pipeline's except block logs via slog when the status
            # update itself fails.  If we reach here with an exception,
            # it means _run_pipeline re-raised — the slog call may or
            # may not have succeeded.  One extra last-resort log line is
            # acceptable to ensure the failure is never invisible.
            #
            # Class names only (no ``str(exc)``): pipeline exceptions may
            # chain SQLAlchemyError ([SQL: ...] / [parameters: ...]),
            # Tier-3 sanitizer output, or source-rendering fragments via
            # ``__cause__`` / ``__context__``. Censor-by-length (``[:200]``)
            # is not redaction — the prefix still carries Tier-3 material.
            # The chain walk preserves the diagnostic signal (fault
            # topology) without the payload.
            exc_class_chain: list[str] = []
            current: BaseException | None = exc
            seen: set[int] = set()
            while current is not None and len(exc_class_chain) < 5:
                if id(current) in seen:
                    # ``__context__`` cycles are rare but possible;
                    # bound the walk defensively.
                    break
                seen.add(id(current))
                exc_class_chain.append(type(current).__name__)
                current = current.__cause__ or current.__context__
            slog.error(
                "pipeline_done_callback_exception",
                exc_type=type(exc).__name__,
                exc_class_chain=exc_class_chain,
            )

    def _to_run_event(self, run_id: str, progress: ProgressEvent) -> RunEvent:
        """Translate engine ProgressEvent to web RunEvent.

        Only handles progress events — terminal events (completed, failed,
        cancelled) are constructed inline in _run_pipeline where the full
        run result is available.

        elspeth-5069612f3c — ``rows_routed_success`` / ``rows_routed_failure``
        are plumbed verbatim from the engine ProgressEvent so the streaming
        wire payload mirrors the terminal CompletedData/CancelledData shape
        and matches the TS ``RunEventProgress`` interface.

        ``rows_succeeded`` / ``rows_quarantined`` are plumbed verbatim from
        the engine ProgressEvent. ProgressData requires both fields — see
        the schema docstring for the fabrication-test rationale.
        """
        return RunEvent(
            run_id=run_id,
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=ProgressData(
                source_rows_processed=progress.rows_processed,
                tokens_succeeded=progress.rows_succeeded,
                tokens_failed=progress.rows_failed,
                tokens_quarantined=progress.rows_quarantined,
                tokens_routed_success=progress.rows_routed_success,
                tokens_routed_failure=progress.rows_routed_failure,
            ),
        )


# Protocol conformance enforcement — mypy verifies ExecutionServiceImpl
# structurally satisfies ExecutionService at this assignment. Without this,
# drift between protocol and impl is only caught at cast() call sites.
_: type[ExecutionService] = ExecutionServiceImpl
