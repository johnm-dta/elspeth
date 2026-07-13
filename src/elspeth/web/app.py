"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import sys
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

import httpx
import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram
from opentelemetry.util.types import AttributeValue
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from elspeth.contracts import RunStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.secrets import (
    FingerprintKeyMissingError,
    SecretDecryptionError,
)
from elspeth.core.landscape.database import LandscapeDB, SchemaCompatibilityError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.transforms.llm.model_catalog import (
    prime_openrouter_catalog_from_live,
    read_openrouter_catalog_snapshot_id,
)
from elspeth.web.audit_readiness.routes import create_audit_readiness_router
from elspeth.web.audit_readiness.service import ReadinessService
from elspeth.web.auth.audit import AuthAuditRecorder
from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.protocol import AuthProvider
from elspeth.web.auth.routes import create_auth_router
from elspeth.web.auth.urls import (
    oidc_browser_endpoint_origin,
    validate_oidc_browser_endpoints,
    validate_oidc_issuer,
)
from elspeth.web.aws_ecs_startup import (
    _CONNECT_TIMEOUT_SECONDS,
    AwsEcsSchemaNotReadyError,
    enforce_aws_ecs_contract,
    require_runtime_directories_mounted,
    validate_only_schema_or_raise,
)
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.catalog.routes import catalog_router
from elspeth.web.composer import yaml_generator as yaml_generator_module
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.composer.tutorial_abandon_routes import create_tutorial_abandon_router
from elspeth.web.composer.tutorial_run_routes import create_tutorial_run_router
from elspeth.web.config import WebSettings, _allow_insecure_test_keys
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.deployment_contract import DEPLOYMENT_TARGET_AWS_ECS
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.routes import create_execution_router
from elspeth.web.execution.runtime_preflight import RuntimePreflightCoordinator
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.execution.websocket_ticket import WebSocketTicketStore
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.middleware.request_id import RequestIdMiddleware
from elspeth.web.operator_telemetry import bootstrap_operator_telemetry
from elspeth.web.preferences.routes import create_preferences_router
from elspeth.web.preferences.service import CorruptPreferencesError, PreferencesService
from elspeth.web.readiness import (
    ReadinessCache,
    ReadinessProbeRunner,
    overall_timeout_report,
    readiness_report,
)
from elspeth.web.schema_probe import postgres_engine_kwargs
from elspeth.web.secrets.routes import create_secrets_router
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from elspeth.web.secrets.user_store import UserSecretStore
from elspeth.web.sessions.audit_story_service import AuditStoryIntegrityError, AuditStoryNotRecordedError
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import (
    LANDSCAPE_RECONCILIATION_PENDING_SUFFIX,
    AuditAccessLogWriteError,
    RunAlreadyActiveError,
    RunRecord,
    StaleComposeStateError,
)
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import _SessionsTelemetry, build_sessions_telemetry
from elspeth.web.shareable_reviews.routes import create_shareable_reviews_router
from elspeth.web.shareable_reviews.service import ShareableReviewService
from elspeth.web.shareable_reviews.signer import ShareTokenSigner

# Assigned by create_app only after the idempotent process MeterProvider
# bootstrap. Keeping these names module-level preserves the existing lifespan
# test seams without creating instruments as an import side effect.
_COMPOSER_BOOT_CONFIG_COUNTER: Counter
_COMPOSER_BOOT_CONFIG_PROBE_LATENCY: Histogram
_COMPOSER_BOOT_PROBE_TIMEOUT_SECONDS = 5.0
# Reserve bounded headroom inside the public five-second readiness contract
# for timeout finalization, redacted logging, JSON serialization, and ASGI
# response dispatch. The shared cache task remains shielded from this waiter.
_READINESS_ROUTE_COMPUTE_TIMEOUT_SECONDS = 4.5

_RETRYABLE_STORAGE_ERRNOS: frozenset[int] = frozenset(
    {
        errno.ENOSPC,
        errno.EROFS,
        errno.EIO,
    }
)


def _dispose_session_engine(engine: Engine) -> None:
    """Dispose the sessions DB pool for app instances that never run lifespan."""
    engine.dispose()


def _close_readiness_runner(runner: ReadinessProbeRunner) -> None:
    """Close readiness workers for app instances that never run lifespan."""
    runner.close()


class _BrowserEndpointDiscoveryDocument(BaseModel):
    """Minimal OIDC discovery shape required by the browser login flow."""

    model_config = ConfigDict(hide_input_in_errors=True)
    issuer: str
    authorization_endpoint: str
    token_endpoint: str

    @field_validator("issuer", "authorization_endpoint", "token_endpoint")
    @classmethod
    def _validate_nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("discovery browser endpoint field must not be blank")
        return value


def _validate_browser_endpoint_discovery_document(
    discovery: object,
    *,
    issuer: str,
    allowed_origins: tuple[str, ...] = (),
) -> tuple[str, str]:
    """Validate discovery issuer and return its exact browser endpoint pair."""
    try:
        document = _BrowserEndpointDiscoveryDocument.model_validate(discovery)
    except ValidationError as exc:
        raise ValueError("OIDC discovery document failed required browser endpoint shape check") from exc
    if document.issuer != issuer:
        raise ValueError("OIDC discovery document failed exact issuer check")
    return validate_oidc_browser_endpoints(
        document.authorization_endpoint,
        document.token_endpoint,
        issuer=issuer,
        allowed_origins=allowed_origins,
    )


def _parse_worker_count(raw_value: str, *, signal_name: str) -> int:
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{signal_name}={raw_value!r} is not a valid integer worker count. "
            "Single-worker enforcement cannot safely determine the process model."
        ) from exc


def _finalize_orphaned_landscape_runs(
    landscape_url: str,
    cancelled_runs: list[RunRecord],
    *,
    create_tables: bool = True,
) -> tuple[frozenset[UUID], frozenset[UUID]]:
    """Idempotently reconcile candidates and classify complete versus absent."""
    complete_run_ids: set[UUID] = set()
    absent_run_ids: set[UUID] = set()
    by_landscape_id: dict[str, list[UUID]] = {}
    for run in cancelled_runs:
        if run.landscape_run_id is None:
            complete_run_ids.add(run.id)
            continue
        by_landscape_id.setdefault(run.landscape_run_id, []).append(run.id)

    if not by_landscape_id:
        return frozenset(complete_run_ids), frozenset()

    with LandscapeDB.from_url(landscape_url, create_tables=create_tables) as landscape_db:
        lifecycle = RecorderFactory(landscape_db).run_lifecycle
        for landscape_run_id, session_run_ids in by_landscape_id.items():
            landscape_run = lifecycle.get_run(landscape_run_id)
            if landscape_run is None:
                absent_run_ids.update(session_run_ids)
                continue
            if landscape_run.status == RunStatus.RUNNING:
                lifecycle.complete_run(landscape_run_id, RunStatus.INTERRUPTED)
            complete_run_ids.update(session_run_ids)
    if absent_run_ids:
        structlog.get_logger().error(
            "orphan_landscape_run_absent",
            outcome="absent",
            count=len(absent_run_ids),
            operator_action="investigate audit-row absence",
        )
    return frozenset(complete_run_ids), frozenset(absent_run_ids)


async def _reconcile_pending_landscape_runs(
    session_service: SessionServiceImpl,
    landscape_url: str,
    *,
    create_tables: bool,
) -> None:
    candidates = await session_service.list_pending_landscape_reconciliations()
    complete_run_ids, absent_run_ids = _finalize_orphaned_landscape_runs(
        landscape_url,
        candidates,
        create_tables=create_tables,
    )
    await session_service.mark_landscape_reconciliation_outcomes(
        complete_run_ids=complete_run_ids,
        absent_run_ids=absent_run_ids,
    )


async def _periodic_orphan_cleanup(
    session_service: SessionServiceImpl,
    execution_service: ExecutionServiceImpl,
    telemetry: _SessionsTelemetry,
    *,
    interval_seconds: int,
    max_age_seconds: int,
    landscape_url: str | None = None,
    create_tables: bool = True,
) -> None:
    """Background task that periodically cancels orphaned runs.

    Runs orphaned by SIGKILL, OOM, or other unclean termination leave
    sessions permanently blocked (partial unique index on active runs).
    Startup cleanup handles the bulk case, but if the server runs for
    days/weeks without restart, this catches runs orphaned mid-uptime.

    Consults execution_service.get_live_run_ids() to distinguish runs
    with active executor threads from genuinely orphaned ones. A run
    is only orphaned if it has no registered shutdown event — age alone
    is not proof of orphanhood.
    """
    import structlog

    slog = structlog.get_logger()
    while True:
        await asyncio.sleep(interval_seconds)
        cancelled = 0
        live_run_ids: frozenset[str] = frozenset()
        try:
            live_run_ids = execution_service.get_live_run_ids()
            if landscape_url is None:
                cancelled = await session_service.cancel_all_orphaned_runs(
                    max_age_seconds=max_age_seconds,
                    exclude_run_ids=live_run_ids,
                    reason="Orphaned by periodic cleanup — no active executor thread",
                )
            else:
                cancelled_runs = await session_service.cancel_all_orphaned_run_records(
                    max_age_seconds=max_age_seconds,
                    exclude_run_ids=live_run_ids,
                    reason=(f"Orphaned by periodic cleanup — no active executor thread {LANDSCAPE_RECONCILIATION_PENDING_SUFFIX}"),
                )
                cancelled = len(cancelled_runs)
                await _reconcile_pending_landscape_runs(
                    session_service,
                    landscape_url,
                    create_tables=create_tables,
                )
        except (SQLAlchemyError, OSError, SchemaCompatibilityError) as cleanup_exc:
            # Narrow catch — only recoverable audit/IO failures are
            # absorbed so the loop retries on the next interval.
            # SQLAlchemyError covers DB-layer transients raised from
            # cancel_all_orphaned_runs (engine.begin(), conn.execute());
            # OSError covers SQLite file-level failures that can escape
            # before SQLAlchemy wraps them.
            #
            # Programmer-bug exceptions (AttributeError from a drifted
            # attribute on ExecutionServiceImpl, TypeError from a
            # signature change, AssertionError from an invariant guard)
            # are NOT caught: they propagate out of the while-loop,
            # terminating the task. The dead task surfaces to the
            # operator at lifespan shutdown because the outer await
            # re-raises the stored exception (the surrounding
            # contextlib.suppress narrows to CancelledError only).
            # Consistent with the audit-cleanup narrow catch in
            # ``ComposerServiceImpl.compose`` (web/composer/service.py)
            # and the cleanup-rollback sites in the
            # ``fork_from_message`` route handler
            # (web/sessions/routes.py).
            #
            # exc_info deliberately omitted: SQLAlchemyError __cause__
            # chains routinely carry the DB connection URL, schema
            # names, and the sqlite file path — the same leak vector
            # closed across every HTTP-path slog.error site when the
            # redaction pattern was standardised. Structured logs carry
            # exc_class only; operators reading logs get enough
            # correlation to triage without the traceback text.
            slog.error(
                "periodic_orphan_cleanup_failed",
                exc_class=type(cleanup_exc).__name__,
            )
        if cancelled:
            telemetry.orphaned_runs_cancelled_total.add(
                cancelled,
                attributes={"source": "periodic", "excluded_live_runs": len(live_run_ids)},
            )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Async lifespan context manager for the FastAPI application.

    Services that require a running event loop must be constructed here,
    not in the synchronous create_app() function. The ProgressBroadcaster
    captures asyncio.get_running_loop() and the ExecutionServiceImpl
    depends on both the broadcaster and the loop.
    """
    import structlog

    slog = structlog.get_logger()

    # Cancel runs orphaned by a previous server crash (D5).
    # Single-process server: every non-terminal run is orphaned after restart.
    # No age filter — cancel ALL pending/running runs immediately.
    settings: WebSettings = app.state.settings
    create_landscape_tables = settings.deployment_target != DEPLOYMENT_TARGET_AWS_ECS
    session_service = app.state.session_service
    cancelled_runs: list[RunRecord] = []
    try:
        cancelled_runs = await session_service.cancel_all_orphaned_run_records(
            reason=f"Orphaned by server restart — no active process {LANDSCAPE_RECONCILIATION_PENDING_SUFFIX}",
        )
        await _reconcile_pending_landscape_runs(
            session_service,
            settings.get_landscape_url(),
            create_tables=create_landscape_tables,
        )
    except (SQLAlchemyError, OSError, SchemaCompatibilityError) as cleanup_exc:
        slog.error(
            "lifespan_orphan_cleanup_failed",
            exc_class=type(cleanup_exc).__name__,
        )
    cancelled = len(cancelled_runs)
    if cancelled:
        app.state.sessions_telemetry.orphaned_runs_cancelled_total.add(
            cancelled,
            attributes={"source": "startup", "excluded_live_runs": 0},
        )

    # Resolve the paired browser endpoints from discovery or explicit config.
    if settings.auth_provider in ("oidc", "entra"):
        if settings.oidc_issuer:
            issuer = validate_oidc_issuer(settings.oidc_issuer)
        elif settings.auth_provider == "entra" and settings.entra_tenant_id:
            issuer = validate_oidc_issuer(f"https://login.microsoftonline.com/{settings.entra_tenant_id}/v2.0")
        else:
            raise SystemExit("FATAL: OIDC discovery requires either oidc_issuer or entra_tenant_id to derive the issuer URL.")

        if settings.oidc_authorization_endpoint and settings.oidc_token_endpoint:
            authorization_endpoint, token_endpoint = validate_oidc_browser_endpoints(
                settings.oidc_authorization_endpoint,
                settings.oidc_token_endpoint,
                issuer=issuer,
                allowed_origins=settings.oidc_authorization_allowed_origins,
            )
            app.state.oidc_authorization_endpoint = authorization_endpoint
            app.state.oidc_token_endpoint = token_endpoint
        else:
            discovery_url = f"{issuer}/.well-known/openid-configuration"
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    resp = await client.get(discovery_url)
                    resp.raise_for_status()
                    authorization_endpoint, token_endpoint = _validate_browser_endpoint_discovery_document(
                        resp.json(),
                        issuer=issuer,
                        allowed_origins=settings.oidc_authorization_allowed_origins,
                    )
                    app.state.oidc_authorization_endpoint = authorization_endpoint
                    app.state.oidc_token_endpoint = token_endpoint
            except (httpx.HTTPError, ValueError) as exc:
                raise SystemExit(
                    f"FATAL: OIDC discovery failed browser endpoint check ({type(exc).__name__}). "
                    "Configure a valid explicit authorization_endpoint/token_endpoint pair or fix discovery."
                ) from None

    # Sub-5: Construct ProgressBroadcaster and ExecutionServiceImpl
    # These require a running event loop, which is only available here.
    loop = asyncio.get_running_loop()
    broadcaster = ProgressBroadcaster(loop, telemetry=app.state.sessions_telemetry)
    app.state.broadcaster = broadcaster

    execution_service = ExecutionServiceImpl(
        loop=loop,
        broadcaster=broadcaster,
        settings=settings,
        session_service=session_service,
        yaml_generator=yaml_generator_module,
        telemetry=app.state.sessions_telemetry,
        blob_service=app.state.blob_service,
        secret_service=app.state.scoped_secret_resolver,
    )
    app.state.execution_service = execution_service

    # ReadinessService aggregates validation / catalog / secrets / retention
    # signals for the audit-readiness panel.  It depends on
    # ``execution_service`` (for ``validate``) so it is constructed here in
    # lifespan rather than in ``create_app``.  ``scoped_secret_resolver``
    # (NOT ``secret_service``) is the correct collaborator — it has
    # ``auth_provider_type`` baked in at construction (app.py:470), matching
    # the precedent set by ExecutionService above.
    app.state.readiness_service = ReadinessService(
        execution_service=execution_service,
        session_service=session_service,
        scoped_secret_resolver=app.state.scoped_secret_resolver,
        settings=settings,
    )

    # ShareableReviewService — Phase 6A completion gestures.
    #
    # Depends on:
    #   * ``execution_service`` (for mark-time validation)
    #   * ``readiness_service`` (for the frozen-at-mark-time audit-readiness
    #     snapshot embedded in the share blob)
    #   * the sessions-DB engine (for ``composer_completion_events_table``
    #     audit writes)
    #   * a ``FilesystemPayloadStore`` (for the content-addressed snapshot
    #     blob — created here, not shared with ``BlobServiceImpl`` because
    #     ``BlobServiceImpl`` owns its own internal payload store with a
    #     different retention semantics)
    #   * the ``ShareTokenSigner`` primitive (HMAC over WebSettings'
    #     required ``shareable_link_signing_key``)
    payload_store = FilesystemPayloadStore(settings.get_payload_store_path())
    app.state.payload_store = payload_store
    # ``shareable_link_signing_key`` is a ``SecretBytes`` (DC-2 FIX-L —
    # masks repr to prevent plaintext leakage in tracebacks/logs).
    # ``.get_secret_value()`` returns the raw bytes the HMAC primitive needs.
    share_token_signer = ShareTokenSigner(settings.shareable_link_signing_key.get_secret_value())
    app.state.share_token_signer = share_token_signer
    app.state.shareable_review_service = ShareableReviewService(
        session_service=session_service,
        execution_service=execution_service,
        readiness_service=app.state.readiness_service,
        signer=share_token_signer,
        settings=settings,
        sessions_db_engine=app.state.session_engine,
        payload_store=payload_store,
        # Phase 8 Sub-task 7c — composer.session.completed_total counter.
        # ``app.state.sessions_telemetry`` is set in ``create_app`` (the
        # synchronous factory) BEFORE the lifespan runs, so it is
        # available here. Mirrors the
        # ``telemetry=app.state.sessions_telemetry`` pattern at line 259
        # for the execution service.
        telemetry=app.state.sessions_telemetry,
    )

    # Prime the OpenRouter model catalog from the live ``/models``
    # endpoint. Closes the validate/runtime drift bug where the bundled
    # litellm catalog still listed models OpenRouter has retired (e.g.
    # ``anthropic/claude-3.5-sonnet``), letting the value-source compliance
    # walker pass configs that 404 at runtime preflight. The request-level
    # probe is graceful inside ``prime_openrouter_catalog_from_live``; failures
    # before the request boundary (client construction/context management) are
    # startup failures rather than undocumented fallback decisions.
    probe_start = time.monotonic()
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=5.0)) as _probe_client:

        async def _probe_get(url: str) -> httpx.Response:
            # ``request("GET", ...)`` rather than ``.get(...)``: identical
            # httpx semantics, but avoids the L3 walker's token-level
            # false match on the literal ``.get`` (R1 targets defensive
            # ``dict.get`` reads, not HTTP client method calls).
            return await _probe_client.request("GET", url)

        primed = await prime_openrouter_catalog_from_live(http_get=_probe_get)
    probe_latency_ms = int((time.monotonic() - probe_start) * 1000)
    if primed:
        slog.info(
            "openrouter_catalog_boot_prime_complete",
            latency_ms=probe_latency_ms,
        )
    else:
        slog.warning(
            "openrouter_catalog_boot_prime_failed",
            latency_ms=probe_latency_ms,
            action="serving bundled litellm catalog (may include retired models)",
        )

    if settings.composer_boot_probe_enabled:
        from elspeth.web.composer.boot_probe import ComposerBootConfigError, probe_composer_config

        # Advisor is mandatory, so the advisor model is always probed.
        probe_models = [settings.composer_model, settings.composer_advisor_model]
        for model in probe_models:
            composer_probe_start = time.monotonic()
            probe_status = "started"
            attributes: dict[str, AttributeValue] = {
                "composer_model": settings.composer_model,
                "composer_temperature": str(settings.composer_temperature),
                "composer_seed": str(settings.composer_seed),
                "composer_advisor_model": settings.composer_advisor_model,
                "probed_model": model,
                "probe_status": probe_status,
            }
            try:
                ok = await asyncio.wait_for(
                    probe_composer_config(
                        model=model,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                    ),
                    timeout=_COMPOSER_BOOT_PROBE_TIMEOUT_SECONDS,
                )
                if ok:
                    probe_status = "success"
                if not ok:
                    probe_status = "transient_failure"
                    slog.warning(
                        "composer_boot_probe_transient_failure",
                        model=model,
                        failure_class="provider_or_transport_error",
                        action="booting; composer LLM calls will be exercised at first use",
                    )
            except TimeoutError:
                probe_status = "transient_failure"
                slog.warning(
                    "composer_boot_probe_transient_failure",
                    model=model,
                    failure_class="TimeoutError",
                    timeout_seconds=_COMPOSER_BOOT_PROBE_TIMEOUT_SECONDS,
                    action="booting; composer LLM calls will be exercised at first use",
                )
            except ComposerBootConfigError:
                probe_status = "rejected"
                raise
            except asyncio.CancelledError:
                probe_status = "cancelled"
                raise
            except Exception:
                probe_status = "local_error"
                raise
            finally:
                attributes["probe_status"] = probe_status
                composer_probe_latency_ms = int((time.monotonic() - composer_probe_start) * 1000)
                _COMPOSER_BOOT_CONFIG_COUNTER.add(1, attributes)
                _COMPOSER_BOOT_CONFIG_PROBE_LATENCY.record(composer_probe_latency_ms, attributes)

    # Resolve the catalog snapshot id (always populated — bundled fallback
    # is always available) and stash on ``app.state``. Run-create writes
    # this into the Landscape ``runs`` row so an auditor can reconstruct
    # which catalog blessed any historical decision. Both fields are
    # invariant for the process lifetime; the orchestrator reads them via
    # ``ExecutionServiceImpl`` (web path) or directly from the module
    # reader (CLI path).
    catalog_sha, catalog_source = read_openrouter_catalog_snapshot_id()
    app.state.openrouter_catalog_sha256 = catalog_sha
    app.state.openrouter_catalog_source = catalog_source
    execution_service.set_openrouter_catalog_snapshot(
        sha256=catalog_sha,
        source=catalog_source,
    )

    # Periodic orphan cleanup — catches runs orphaned by SIGKILL/OOM
    # between restarts. Startup cleanup (above) handles the bulk case;
    # this catches runs orphaned while the server is still running.
    # Liveness-aware: excludes runs with active executor threads.
    orphan_task = asyncio.create_task(
        _periodic_orphan_cleanup(
            session_service,
            execution_service,
            app.state.sessions_telemetry,
            interval_seconds=settings.orphan_run_check_interval_seconds,
            max_age_seconds=settings.orphan_run_max_age_seconds,
            landscape_url=settings.get_landscape_url(),
            create_tables=create_landscape_tables,
        )
    )

    try:
        yield
    finally:
        # Cancel periodic cleanup before shutting down the executor
        orphan_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await orphan_task

        app.state.readiness_probe_runner.close()

        # Shutdown execution service thread pool without blocking the loop:
        # worker cleanup still schedules terminal-state writes back onto it.
        await execution_service.shutdown()
        # Tier-2 operator telemetry stops only after all audited execution work
        # has drained. Expected collector outages are bounded/redacted inside
        # the runtime and can never rewrite a committed Landscape record.
        await app.state.operator_telemetry.shutdown()
        # Tear down the process-wide run_sync_in_worker pool before disposing
        # the engine, so no worker thread races a query against a disposed pool.
        from elspeth.web.async_workers import shutdown_async_workers

        await shutdown_async_workers()
        app.state.session_engine.dispose()


# Fields that accept JSON-encoded collection values from environment variables.
# Add any new tuple-typed WebSettings fields here so _settings_from_env()
# JSON-decodes them.  Scalar fields (str, int, float, Path) are handled by Pydantic.
_JSON_COLLECTION_FIELDS: frozenset[str] = frozenset(
    {"cors_origins", "server_secret_allowlist", "oidc_authorization_allowed_origins", "plugin_allowlist"}
)
_JSON_OBJECT_FIELDS: frozenset[str] = frozenset({"plugin_preferences", "plugin_control_modes", "llm_profiles"})


def _settings_from_env() -> WebSettings:
    """Construct WebSettings from ELSPETH_WEB__* environment variables.

    Called when create_app() is invoked without explicit settings (e.g.,
    by uvicorn's factory protocol).  The CLI sets these env vars before
    calling uvicorn.run().

    Collection-typed fields are JSON-decoded via ``_JSON_COLLECTION_FIELDS``.
    The JSON literal ``null`` is decoded to ``None`` for all fields — this is
    the env-var convention for "clear this optional setting."  Pydantic rejects
    ``None`` for non-nullable fields. Unknown setting names are rejected before
    model construction so deployment typos fail startup with the original
    environment variable name. All other scalar values pass as raw strings; Pydantic coerces
    str→int, str→float, str→Path automatically.
    """
    kwargs: dict[str, object] = {}
    prefix = "ELSPETH_WEB__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            field_name = key[len(prefix) :].lower()
            if field_name not in WebSettings.model_fields:
                raise RuntimeError(f"Unknown ELSPETH_WEB__ setting: {key}")
            if field_name in _JSON_COLLECTION_FIELDS | _JSON_OBJECT_FIELDS:
                # These fields are tuple-typed on WebSettings; the env-var
                # convention is a JSON-encoded array. A non-JSON value cannot
                # become a valid collection, so falling back to the raw string
                # would only defer to a confusing downstream Pydantic
                # "not a valid tuple" error. Per the web trust model, malformed
                # startup config is a hard failure — refuse to start with a
                # message that names the offending variable.
                try:
                    parsed = json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    expected = "array" if field_name in _JSON_COLLECTION_FIELDS else "object"
                    raise RuntimeError(f"ELSPETH_WEB__{field_name.upper()} must be valid JSON {expected}.") from None
                if field_name in _JSON_COLLECTION_FIELDS:
                    if not isinstance(parsed, list):
                        raise RuntimeError(f"ELSPETH_WEB__{field_name.upper()} must be valid JSON array.")
                    kwargs[field_name] = tuple(parsed)
                else:
                    if not isinstance(parsed, dict):
                        raise RuntimeError(f"ELSPETH_WEB__{field_name.upper()} must be valid JSON object.")
                    kwargs[field_name] = parsed
            elif value == "null":
                kwargs[field_name] = None
            else:
                kwargs[field_name] = value
    # DC-2 FIX-L: ``shareable_link_signing_key`` is typed ``SecretBytes`` on
    # the model, but the env-var ingest path passes a str (base64-encoded)
    # which a ``mode="before"`` validator on the field decodes to bytes.
    # Mypy can't see through Pydantic's pre-validators, so the **kwargs
    # widening is reported as a type error here. The cast is safe because
    # Pydantic raises ``ValidationError`` on any mismatch at runtime.
    try:
        return WebSettings(**kwargs)  # type: ignore[arg-type]
    except ValidationError as error:
        policy_fields = {"plugin_allowlist", "plugin_preferences", "plugin_control_modes", "llm_profiles", "tutorial_llm_profile"}
        safe_paths = {
            str(item) for detail in error.errors(include_input=False) for item in detail.get("loc", ()) if isinstance(item, (str, int))
        }
        if policy_fields & safe_paths:
            rendered_paths = ", ".join(sorted(safe_paths))
            raise RuntimeError(f"Invalid ELSPETH_WEB__ plugin policy setting at: {rendered_paths}") from None
        raise


class _BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies declaring Content-Length > 10 MB with HTTP 413.

    Phase 5b.0.5 (F-3): defense-in-depth body-size guard.  The Pydantic
    per-field caps (``SendMessageRequest.content`` at 64 KiB,
    ``_InlineBlobModel.content`` at 256 KiB) are the actual guarantees;
    this middleware short-circuits requests that declare an oversized
    body before the framework parses anything.  Mirrors the
    ``settings.max_upload_bytes`` post-decode guard at
    ``web/blobs/routes.py:171,208``, but at the ASGI layer so it covers
    every route uniformly.

    Threat model: an attacker submitting a multi-gigabyte JSON body
    forces FastAPI to buffer the entire payload before the per-field
    validator can reject it — even though the validator would ultimately
    reject it.  10 MB is the global ceiling chosen to comfortably exceed
    the 256 KiB inline-blob cap while remaining well below any
    pathological memory pressure threshold.

    The check is Content-Length-only by design: clients may omit or
    falsify the header.  Malformed ``Content-Length`` is rejected here as a
    deterministic client error before the body is read.  The Pydantic caps
    remain the contract.
    """

    _MAX_BODY_BYTES = 10 * 1024 * 1024  # 10 MB

    async def dispatch(self, request: StarletteRequest, call_next):  # type: ignore[no-untyped-def]
        content_length = request.headers.get("content-length")
        if content_length is None:
            return await call_next(request)
        if not content_length.isascii() or not content_length.isdecimal():
            return StarletteResponse(
                content='{"error": "Invalid Content-Length"}',
                status_code=400,
                media_type="application/json",
            )
        try:
            content_length_bytes = int(content_length)
        except ValueError:
            return StarletteResponse(
                content='{"error": "Invalid Content-Length"}',
                status_code=400,
                media_type="application/json",
            )
        if content_length_bytes < 0:
            return StarletteResponse(
                content='{"error": "Invalid Content-Length"}',
                status_code=400,
                media_type="application/json",
            )
        if content_length_bytes > self._MAX_BODY_BYTES:
            return StarletteResponse(
                content='{"error": "Request body too large (max 10 MB)"}',
                status_code=413,
                media_type="application/json",
            )
        return await call_next(request)


_SPA_CSP_PREFIX = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "font-src 'self'; img-src 'self' data:; "
    "connect-src 'self' ws://localhost:* wss://localhost:*"
)


class _BrowserDocumentHeadersMiddleware(BaseHTTPMiddleware):
    """Apply callback secrecy headers and the runtime OIDC connect policy."""

    async def dispatch(self, request: StarletteRequest, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        content_type = response.headers.get("content-type", "")
        if not content_type.lower().startswith("text/html"):
            return response

        connect_origin: str | None = None
        token_endpoint = getattr(request.app.state, "oidc_token_endpoint", None)
        if isinstance(token_endpoint, str):
            token_origin = oidc_browser_endpoint_origin(token_endpoint)
            request_port = request.url.port
            default_port = 443 if request.url.scheme == "https" else 80
            request_host = request.url.hostname or ""
            request_origin = f"{request.url.scheme}://{request_host.lower()}"
            if request_port not in (None, default_port):
                request_origin += f":{request_port}"
            if token_origin != request_origin:
                connect_origin = token_origin

        response.headers["Content-Security-Policy"] = _SPA_CSP_PREFIX if connect_origin is None else f"{_SPA_CSP_PREFIX} {connect_origin}"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response


def create_app(settings: WebSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Web application settings. When None, reads from
            ELSPETH_WEB__* environment variables (set by the CLI).

    Returns:
        Configured FastAPI instance with CORS middleware and health endpoint.
    """
    if settings is None:
        settings = _settings_from_env()

    # Reject an incomplete AWS deployment policy before installing the
    # process-global provider. A failed first create_app() must not strand a
    # later corrected AWS boot on a Prometheus-only provider.
    if settings.deployment_target == DEPLOYMENT_TARGET_AWS_ECS:
        enforce_aws_ecs_contract(settings)

    operator_runtime = bootstrap_operator_telemetry(settings)
    operator_meter = metrics.get_meter(__name__)
    global _COMPOSER_BOOT_CONFIG_COUNTER, _COMPOSER_BOOT_CONFIG_PROBE_LATENCY
    _COMPOSER_BOOT_CONFIG_COUNTER = operator_meter.create_counter(
        "composer.boot_config",
        description="Composer effective sampling config recorded at boot",
    )
    _COMPOSER_BOOT_CONFIG_PROBE_LATENCY = operator_meter.create_histogram(
        "composer.boot_config.probe_latency_ms",
        description="Composer boot config probe latency in milliseconds",
        unit="ms",
    )

    app = FastAPI(title="ELSPETH Web", version="0.1.0", lifespan=lifespan)
    app.state.operator_telemetry = operator_runtime

    @app.exception_handler(AuditIntegrityError)
    async def _audit_integrity_error_handler(_request: Request, exc: AuditIntegrityError) -> JSONResponse:
        failed_turn = exc.failed_turn
        if failed_turn is None:
            return JSONResponse(
                status_code=500,
                content={
                    "error_type": "audit_integrity_error",
                    "detail": "Audit persistence failed; no audit-grade data returned.",
                    "diagnostic": "no_failed_turn_metadata",
                    "reason": "originated outside compose-loop annotation scope",
                },
            )
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "audit_integrity_error",
                "detail": "Audit persistence failed; no audit-grade data returned.",
                "failed_turn": {
                    "assistant_message_id": failed_turn.assistant_message_id,
                    "tool_calls_attempted": failed_turn.tool_calls_attempted,
                    "tool_responses_persisted": failed_turn.tool_responses_persisted or 0,
                    "transcript_url": None,
                },
            },
        )

    @app.exception_handler(CorruptPreferencesError)
    async def _corrupt_preferences_error_handler(_request: Request, exc: CorruptPreferencesError) -> JSONResponse:
        # Named Tier-1 read-guard exception from
        # ``preferences/service.py`` — a stored composer-preferences row
        # violates a closed-list invariant (default_composer_mode outside
        # ``_VALID_MODES``, tutorial_completed_at unparseable, etc.). The
        # docstring on the exception promises this handler exists so that
        # incident-response code can switch on ``error_type`` rather than
        # string-grep the message; without the handler the failure was
        # rendered as a bare 500 and the frontend's bootstrap path
        # silently swallowed it (``App.tsx``'s
        # ``bootstrapPrefs().catch(console.error)``), leaving a corrupt-
        # row user with no signal anything was wrong.
        #
        # Body contract: ``error_type`` is the discriminator the frontend
        # store branches on; ``field_name`` (closed enum) and ``user_id``
        # (caller's own id) help the operator locate the row.
        # ``detail`` is a static phrase rather than ``str(exc)`` because
        # the exception's __str__ embeds ``bad_value`` — corrupt content
        # could carry arbitrary writes from a tampering or fuzzing event
        # and must not echo to the client. Same redaction pattern as
        # ``_audit_integrity_error_handler`` and
        # ``_audit_access_log_write_error_handler``.
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "corrupt_preferences",
                "detail": "Saved preferences are corrupt; the composer is using defaults.",
                "field_name": exc.field_name,
                "user_id": exc.user_id,
            },
        )

    @app.exception_handler(AuditStoryIntegrityError)
    async def _audit_story_integrity_error_handler(_request: Request, exc: AuditStoryIntegrityError) -> JSONResponse:
        # Sibling shape to ``_audit_integrity_error_handler`` above. The
        # named-type discriminator was getting flattened to bare
        # ``RuntimeError`` at the route boundary (sessions/routes.py),
        # which routed the failure to FastAPI's default 500 handler and
        # destroyed the ``error_type`` discriminator that incident-response
        # code switches on. The route's wrap was removed in the same
        # change that registered this handler; both halves of the fix are
        # load-bearing.
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "audit_story_integrity_error",
                "detail": str(exc),
            },
        )

    @app.exception_handler(AuditStoryNotRecordedError)
    async def _audit_story_not_recorded_error_handler(_request: Request, _exc: AuditStoryNotRecordedError) -> JSONResponse:
        # Absent-state sibling of ``_audit_story_integrity_error_handler``
        # above: the run exists but no audit story was ever recorded for it
        # (today only the tutorial projection writes the audit-story columns,
        # so this is the normal state for every non-tutorial run). Structured
        # 404 with a stable machine code; the detail is fixed plain language —
        # the internal exception text (which names Landscape run ids) is
        # deliberately not echoed.
        return JSONResponse(
            status_code=404,
            content={
                "error_type": "audit_story_not_recorded",
                "detail": "No audit story was recorded for this run.",
            },
        )

    @app.exception_handler(StaleComposeStateError)
    async def _stale_compose_state_error_handler(_request: Request, _exc: StaleComposeStateError) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "error_type": "stale_compose_state",
                "detail": "The session changed while the compose turn was running.",
            },
        )

    @app.exception_handler(AuditAccessLogWriteError)
    async def _audit_access_log_write_error_handler(_request: Request, _exc: AuditAccessLogWriteError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "error_type": "audit_access_log_write_failed",
                "detail": "Audit-grade transcript access could not be recorded; no audit-grade data returned.",
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-id middleware registered AFTER CORS so it runs outermost
    # (Starlette's add_middleware is LIFO): the correlation id is set on
    # request.state before any downstream code — including the CORS
    # middleware, route handlers, and the app-level exception handlers
    # below — can read it.  Echoed back as the X-Request-ID response
    # header for client-side log correlation.
    app.add_middleware(RequestIdMiddleware)

    # Body-size guard registered LAST so it runs OUTERMOST (Starlette
    # add_middleware is LIFO).  Phase 5b.0.5 (F-3): reject oversized
    # Content-Length declarations before any other middleware or handler
    # touches the body.  The 413 response is emitted without a request
    # id — by design the check fires before RequestIdMiddleware — which
    # is acceptable: the response carries the standard 413 semantics and
    # a body-too-large rejection has no useful pairing to a slog event.
    app.add_middleware(_BodySizeLimitMiddleware)
    app.add_middleware(_BrowserDocumentHeadersMiddleware)

    app.state.settings = settings

    aws_session_engine: Engine | None = None
    if settings.deployment_target == DEPLOYMENT_TARGET_AWS_ECS:
        require_runtime_directories_mounted(settings)
        raw_session_url = settings.session_db_url
        assert raw_session_url is not None
        try:
            aws_session_engine = create_session_engine(
                raw_session_url,
                connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS},
                **postgres_engine_kwargs(raw_session_url),
            )
        except (SQLAlchemyError, ImportError):
            raise AwsEcsSchemaNotReadyError(
                "AWS ECS session_schema engine could not be constructed. Run 'elspeth doctor aws-ecs' for full diagnostics."
            ) from None
        try:
            validate_only_schema_or_raise(settings, aws_session_engine)
        except BaseException:
            aws_session_engine.dispose()
            raise
        weakref.finalize(app, _dispose_session_engine, aws_session_engine)
    else:
        # Ensure data directory and subdirectories exist before any DB access.
        # get_landscape_url() defaults to data_dir/runs/audit.db — SQLite does
        # not create parent directories, so we must ensure runs/ exists too.
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "runs").mkdir(exist_ok=True)

    # --- Catalog ---
    app.state.catalog_service = create_catalog_service()
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
    from elspeth.web.plugin_policy.profiles import RuntimeWebPluginConfig

    app.state.runtime_web_plugin_config = RuntimeWebPluginConfig.from_settings(settings)
    app.state.web_plugin_policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=app.state.runtime_web_plugin_config,
    )
    from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry

    app.state.operator_profile_registry = OperatorProfileRegistry(
        policy=app.state.web_plugin_policy,
        settings=app.state.runtime_web_plugin_config,
    )
    app.include_router(
        catalog_router,
        prefix="/api/catalog",
        dependencies=[Depends(get_current_user)],
    )

    # --- Auth provider setup ---
    auth_provider: AuthProvider
    if settings.auth_provider == "local":
        auth_provider = LocalAuthProvider(
            db_path=settings.data_dir / "auth.db",
            secret_key=settings.secret_key,
        )
    elif settings.auth_provider == "oidc":
        from elspeth.web.auth.oidc import OIDCAuthProvider

        # Validator _validate_auth_fields guarantees non-None
        assert settings.oidc_issuer is not None
        assert settings.oidc_audience is not None
        auth_provider = OIDCAuthProvider(
            issuer=settings.oidc_issuer,
            audience=settings.oidc_audience,
            jwks_cache_ttl_seconds=settings.jwks_cache_ttl_seconds,
            jwks_failure_retry_seconds=settings.jwks_failure_retry_seconds,
            audience_claim=settings.oidc_audience_claim,
        )
    elif settings.auth_provider == "entra":
        from elspeth.web.auth.entra import EntraAuthProvider

        assert settings.entra_tenant_id is not None
        assert settings.oidc_audience is not None
        auth_provider = EntraAuthProvider(
            tenant_id=settings.entra_tenant_id,
            audience=settings.oidc_audience,
            jwks_cache_ttl_seconds=settings.jwks_cache_ttl_seconds,
            jwks_failure_retry_seconds=settings.jwks_failure_retry_seconds,
        )
    else:
        raise RuntimeError(f"Unsupported auth provider: {settings.auth_provider}")
    app.state.auth_provider = auth_provider
    app.state.auth_audit_recorder = AuthAuditRecorder.from_settings(settings)
    app.state.oidc_authorization_endpoint = settings.oidc_authorization_endpoint
    app.state.oidc_token_endpoint = settings.oidc_token_endpoint

    # W16/S3: Secret key production guard -- hard crash
    if settings.secret_key == "change-me-in-production" and not _allow_insecure_test_keys(settings.host):
        raise SystemExit(
            "FATAL: WebSettings.secret_key is set to the default value. "
            "Set a secure secret_key before starting the web server. "
            "See WebSettings documentation."
        )

    # --- Session database setup ---
    if aws_session_engine is not None:
        session_engine = aws_session_engine
    else:
        session_db_url = settings.get_session_db_url()
        session_engine = create_session_engine(session_db_url, **postgres_engine_kwargs(session_db_url))
        initialize_session_schema(session_engine)
        session_db_path = session_engine.url.database
        if session_engine.dialect.name == "sqlite" and session_db_path not in (None, ":memory:"):
            session_engine.dispose()
        weakref.finalize(app, _dispose_session_engine, session_engine)

    # Build the sessions-telemetry container ONCE per process and share it
    # across every consumer (SessionServiceImpl, ExecutionServiceImpl, and
    # any future surface). The counters are intentionally process-scoped —
    # one Counter per metric, not one per consumer — so OTel aggregates by
    # attribute set instead of by injection site.
    sessions_telemetry = build_sessions_telemetry(meter=metrics.get_meter("elspeth.web.composer"))
    app.state.sessions_telemetry = sessions_telemetry

    session_service = SessionServiceImpl(
        session_engine,
        data_dir=settings.data_dir,
        telemetry=sessions_telemetry,
        log=structlog.get_logger("sessions"),
    )
    app.state.session_service = session_service
    app.state.session_engine = session_engine  # available to guided step handlers
    readiness_probe_runner = ReadinessProbeRunner()
    app.state.readiness_probe_runner = readiness_probe_runner
    app.state.readiness_cache = ReadinessCache()
    weakref.finalize(app, _close_readiness_runner, readiness_probe_runner)

    # --- Preferences service ---
    # Per-user composer settings (default_composer_mode, banner_dismissed_at,
    # tutorial_completed_at).
    # Shares the session engine; preferences live on the same metadata.
    app.state.preferences_service = PreferencesService(session_engine)

    # --- Blob service ---
    app.state.blob_service = BlobServiceImpl(
        session_engine,
        settings.data_dir,
        settings.max_blob_storage_per_session_bytes,
    )

    # --- Secret service ---
    user_secret_store = UserSecretStore(session_engine, settings.secret_key)
    server_secret_store = ServerSecretStore(settings.server_secret_allowlist)
    app.state.secret_service = WebSecretService(user_secret_store, server_secret_store)
    app.state.scoped_secret_resolver = ScopedSecretResolver(app.state.secret_service, settings.auth_provider)

    # --- Composer service (singleton, not per-request) ---
    runtime_preflight_coordinator = RuntimePreflightCoordinator()
    app.state.runtime_preflight_coordinator = runtime_preflight_coordinator
    app.state.composer_service = ComposerServiceImpl(
        catalog=app.state.catalog_service,
        settings=settings,
        sessions_service=session_service,
        session_engine=session_engine,
        secret_service=app.state.scoped_secret_resolver,
        runtime_preflight_coordinator=runtime_preflight_coordinator,
    )
    app.state.composer_availability = app.state.composer_service.get_availability()
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.state.websocket_ticket_store = WebSocketTicketStore()

    # --- Rate limiter (per-process in-memory) ---
    # ComposerRateLimiter is safe to construct in sync context because
    # _locks_lock is lazily created on first async use (Python 3.12+
    # requires asyncio.Lock() inside a running event loop).
    app.state.rate_limiter = ComposerRateLimiter(
        limit=settings.composer_rate_limit_per_minute,
    )

    # --- Auth rate limiter (per-IP, unauthenticated endpoints) ---
    app.state.auth_rate_limiter = ComposerRateLimiter(
        limit=settings.auth_rate_limit_per_minute,
    )

    # --- Multi-worker enforcement (W10 -> R6) ---
    # ProgressBroadcaster and the rate limiter are process-local, so
    # multi-worker mode is unsupported.  Check multiple signals because
    # different deployment tools advertise workers in different ways.
    multi_worker_reason: str | None = None

    # 1. WEB_CONCURRENCY env var (Heroku, Railway, render.com)
    web_concurrency_str = os.environ.get("WEB_CONCURRENCY", "1")
    if _parse_worker_count(web_concurrency_str, signal_name="WEB_CONCURRENCY") > 1:
        multi_worker_reason = f"WEB_CONCURRENCY={web_concurrency_str}"

    # 2. sys.argv: uvicorn --workers N, gunicorn -w N / --workers N
    if multi_worker_reason is None:
        argv = sys.argv
        for i, arg in enumerate(argv):
            if arg == "--workers" and i + 1 < len(argv):
                if _parse_worker_count(argv[i + 1], signal_name="--workers") > 1:
                    multi_worker_reason = f"--workers {argv[i + 1]}"
            elif arg.startswith("--workers="):
                worker_value = arg.split("=", 1)[1]
                if _parse_worker_count(worker_value, signal_name="--workers") > 1:
                    multi_worker_reason = f"{arg}"
            elif arg == "-w" and i + 1 < len(argv) and _parse_worker_count(argv[i + 1], signal_name="-w") > 1:
                multi_worker_reason = f"-w {argv[i + 1]}"

    if multi_worker_reason is not None:
        raise RuntimeError(
            f"Multi-worker mode detected ({multi_worker_reason}) but is not supported. "
            "ProgressBroadcaster holds subscriber queues in process memory — "
            "WebSocket progress streaming requires a single worker. "
            "For multi-worker deployment, replace ProgressBroadcaster with Redis Streams."
        )

    # --- Register routers ---
    app.include_router(create_auth_router())
    app.include_router(create_session_router())
    app.include_router(create_preferences_router())
    app.include_router(create_tutorial_run_router())
    app.include_router(create_tutorial_abandon_router())
    app.include_router(create_blobs_router())
    app.include_router(create_secrets_router())
    app.include_router(create_execution_router())
    app.include_router(create_audit_readiness_router())
    app.include_router(create_shareable_reviews_router())

    # --- Seam contract D: RunAlreadyActiveError -> 409 with error_type ---
    @app.exception_handler(RunAlreadyActiveError)
    async def handle_run_already_active(
        request: Request,
        exc: RunAlreadyActiveError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={"detail": str(exc), "error_type": "run_already_active"},
        )

    # --- Secret-subsystem typed error translation ---
    # Trust-boundary translation layer: store/service-level typed errors
    # become deterministic HTTP contracts for API consumers.  Each handler
    # follows the canonical SQLAlchemy-redaction pattern used across the
    # web package for handler-level ``__cause__`` chains:
    #
    #   * slog event carries ``exc_class`` only — NO ``exc_info``.
    #   * response body contains NO ``str(exc)`` — the message is a
    #     static operator-authored string, not the exception text (which
    #     may carry DB URLs, bound SQL parameters, stored secret names,
    #     or other Tier-3 data the redaction was meant to protect).
    #   * ``request_id`` correlation id surfaced in both the response
    #     body and the slog event so operators can pair a user-reported
    #     error to its triage trail with one lookup.
    #
    # The pending (elspeth-149856079f) Landscape audit events will hang
    # off these same handlers — the correlation id threads through to
    # those records when that work lands.

    _handler_slog = structlog.get_logger()

    def _request_id(request: Request) -> str:
        """Read the correlation id set by RequestIdMiddleware.

        ``RequestIdMiddleware`` sets ``request.state.request_id``
        before delegating to ``call_next``, so every request that reaches
        a route handler, dependency, or app-level exception handler has the
        id assigned. The exception handlers that call this helper
        (``FingerprintKeyMissingError``, ``SecretDecryptionError``,
        ``OperationalError``, ``OSError``) are invoked by Starlette's
        ``ExceptionMiddleware``, which wraps the router *inside*
        ``RequestIdMiddleware`` — the id is therefore guaranteed present.
        A missing attribute here would mean the middleware contract is
        broken, which is our bug to surface, not to paper over with a
        sentinel. Mirrors the direct read in ``web/auth/audit.py``.
        """
        request_id: str = request.state.request_id
        return request_id

    @app.exception_handler(FingerprintKeyMissingError)
    async def handle_fingerprint_missing(
        request: Request,
        exc: FingerprintKeyMissingError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        _handler_slog.error(
            "http_fingerprint_key_missing",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            exc_class=type(exc).__name__,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Secret resolver is not configured: ELSPETH_FINGERPRINT_KEY is unset. "
                    "Set the environment variable on the server and retry."
                ),
                "error_type": "fingerprint_key_missing",
                "request_id": request_id,
            },
        )

    @app.exception_handler(SecretDecryptionError)
    async def handle_secret_decryption_failed(
        request: Request,
        exc: SecretDecryptionError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        _handler_slog.error(
            "http_secret_decryption_failed",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            exc_class=type(exc).__name__,
        )
        return JSONResponse(
            status_code=409,
            content={
                "detail": ("Stored secret cannot be decrypted — likely a web secret_key rotation. Re-save the secret to resolve."),
                "error_type": "secret_decryption_failed",
                "request_id": request_id,
            },
        )

    @app.exception_handler(OperationalError)
    async def handle_database_unavailable(
        request: Request,
        exc: OperationalError,
    ) -> JSONResponse:
        request_id = _request_id(request)
        _handler_slog.error(
            "http_database_unavailable",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            exc_class=type(exc).__name__,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": ("Database is currently unavailable. Please retry in a moment."),
                "error_type": "database_unavailable",
                "request_id": request_id,
            },
        )

    @app.exception_handler(OSError)
    async def handle_storage_unavailable(
        request: Request,
        exc: OSError,
    ) -> JSONResponse:
        """Translate retryable storage-backend failures to a redacted 503.

        Only backend availability failures become ``storage_unavailable``.
        Programmer/configuration bugs such as missing-path errors are
        re-raised so they surface as 500s instead of misleading retryable
        outage responses.
        """
        if exc.errno not in _RETRYABLE_STORAGE_ERRNOS:
            raise exc
        request_id = _request_id(request)
        _handler_slog.error(
            "http_storage_unavailable",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            exc_class=type(exc).__name__,
            errno=exc.errno,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": ("Storage backend is currently unavailable. Please retry in a moment."),
                "error_type": "storage_unavailable",
                "request_id": request_id,
            },
        )

    # --- 422 input redaction (all routes) ---
    # FastAPI's default RequestValidationError handler echoes the ``input``
    # field, which leaks plaintext values from the request body (secret
    # values on /api/secrets, passwords on /api/auth/login, etc.).
    # We allowlist only the structurally safe keys: type, loc, msg.
    # This is deliberately global — any route can carry sensitive fields,
    # and callers can reconstruct the failing input from their own request.
    _SAFE_VALIDATION_ERROR_KEYS = frozenset({"type", "loc", "msg"})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        safe_errors = [{k: v for k, v in error.items() if k in _SAFE_VALIDATION_ERROR_KEYS} for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": safe_errors})

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/ready")
    async def ready(request: Request) -> JSONResponse:
        async def compute():  # type: ignore[no-untyped-def]
            return await readiness_report(
                request.app.state.settings,
                request.app.state.session_engine,
                request.app.state.readiness_probe_runner,
            )

        try:
            async with asyncio.timeout(_READINESS_ROUTE_COMPUTE_TIMEOUT_SECONDS):
                report = await request.app.state.readiness_cache.get(compute)
        except TimeoutError:
            report = overall_timeout_report()
        return JSONResponse(
            status_code=200 if report.ready else 503,
            content={"ready": report.ready, "checks": [asdict(check) for check in report.checks]},
        )

    @app.get("/api/system/status")
    async def system_status() -> dict[str, object]:
        composer = app.state.composer_availability
        return {
            "composer_available": composer.available,
            "composer_model": composer.model,
            "composer_provider": composer.provider,
            "composer_reason": composer.reason,
            "composer_missing_keys": list(composer.missing_keys),
            # The SPA derives its compose abort ceiling from this at boot:
            # client cap = wall clock + client grace, keeping the backend's
            # structured 422 ahead of the client abort for ANY configured
            # wall clock (there is no fixed maximum — only transport-ceiling
            # headroom), not just the checked-in deployment default.
            "composer_timeout_seconds": settings.composer_timeout_seconds,
        }

    # --- Prometheus metrics scrape endpoint ---
    # Registered as a route (not a mount) so bare `/metrics` matches without
    # the trailing-slash redirect that a Mount requires. The SPA StaticFiles
    # mount at `/` (registered below) would otherwise shadow `/metrics` for
    # the non-slash variant, returning 404 because no `metrics` file exists
    # in dist/. Route handlers match before mounts in Starlette, so this
    # also wins precedence over the SPA catch-all regardless of order.
    # Backed by the retained process-level Prometheus reader; all OTel
    # counters/histograms registered via metrics.get_meter() feed into this
    # endpoint automatically via the global REGISTRY.
    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(get_current_user)])
    def _prometheus_metrics() -> Response:
        # ``generate_latest()`` walks the global REGISTRY; a corrupted
        # collector would raise here and Starlette's default 500 handler
        # leaks the traceback into the response body. /metrics is a
        # scrape endpoint — return a safe 503 with no internal detail so
        # scrapers retry gracefully.  Audit-primacy (CLAUDE.md): this
        # endpoint reads in-memory counter state only, so failure is a
        # telemetry-system failure (operational, not legal) — logged, not
        # audited. A bounded message is safe to log here (unlike the
        # secret-bearing DB exceptions elsewhere): the route has already
        # authenticated before collection, and the bounded message identifies
        # which collector broke without reading request-controlled input.
        try:
            body = generate_latest()
        except Exception as scrape_exc:
            _handler_slog.error(
                "prometheus_scrape_failed",
                exc_class=type(scrape_exc).__name__,
                detail=str(scrape_exc)[:200],
            )
            return Response(
                content=b"# scrape failed\n",
                status_code=503,
                media_type=CONTENT_TYPE_LATEST,
            )
        return Response(content=body, media_type=CONTENT_TYPE_LATEST)

    # --- Static file serving for the React SPA (production) ---
    # Mount frontend/dist/ AFTER all API and WS routes so /api/* takes precedence.
    # Only active when the build output exists (i.e., after `npm run build`).
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="spa")

    return app
