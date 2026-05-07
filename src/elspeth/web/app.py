"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, field_validator
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from elspeth.contracts.secrets import (
    FingerprintKeyMissingError,
    SecretDecryptionError,
)
from elspeth.web.auth.local import LocalAuthProvider
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.protocol import AuthProvider
from elspeth.web.auth.routes import create_auth_router
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.catalog.routes import catalog_router
from elspeth.web.composer import yaml_generator as yaml_generator_module
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import _LOCAL_HOSTS, WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.execution.progress import ProgressBroadcaster
from elspeth.web.execution.routes import create_execution_router
from elspeth.web.execution.runtime_preflight import RuntimePreflightCoordinator
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.middleware.request_id import RequestIdMiddleware
from elspeth.web.secrets.routes import create_secrets_router
from elspeth.web.secrets.server_store import ServerSecretStore
from elspeth.web.secrets.service import ScopedSecretResolver, WebSecretService
from elspeth.web.secrets.user_store import UserSecretStore
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.protocol import RunAlreadyActiveError
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl

_RETRYABLE_STORAGE_ERRNOS: frozenset[int] = frozenset(
    {
        errno.ENOSPC,
        errno.EROFS,
        errno.EIO,
    }
)


class _AuthorizationEndpointDiscoveryDocument(BaseModel):
    """Minimal OIDC discovery shape required by the web app."""

    authorization_endpoint: str

    @field_validator("authorization_endpoint")
    @classmethod
    def _validate_authorization_endpoint(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("authorization_endpoint must not be blank")
        return value


def _validate_authorization_endpoint_discovery_document(discovery: object) -> str:
    """Validate the discovery document shape and return authorization_endpoint."""
    try:
        document = _AuthorizationEndpointDiscoveryDocument.model_validate(discovery)
    except ValidationError as exc:
        raise ValueError("OIDC discovery document must provide a non-empty string 'authorization_endpoint'") from exc
    return document.authorization_endpoint


def _parse_worker_count(raw_value: str, *, signal_name: str) -> int:
    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{signal_name}={raw_value!r} is not a valid integer worker count. "
            "Single-worker enforcement cannot safely determine the process model."
        ) from exc


async def _periodic_orphan_cleanup(
    session_service: SessionServiceImpl,
    execution_service: ExecutionServiceImpl,
    *,
    interval_seconds: int,
    max_age_seconds: int,
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
        try:
            live_run_ids = execution_service.get_live_run_ids()
            cancelled = await session_service.cancel_all_orphaned_runs(
                max_age_seconds=max_age_seconds,
                exclude_run_ids=live_run_ids,
                reason="Orphaned by periodic cleanup — no active executor thread",
            )
            if cancelled:
                slog.info("periodic_orphan_cleanup", cancelled=cancelled, excluded=len(live_run_ids))
        except (SQLAlchemyError, OSError) as cleanup_exc:
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
    session_service = app.state.session_service
    cancelled = await session_service.cancel_all_orphaned_runs(
        reason="Orphaned by server restart — no active process",
    )
    if cancelled:
        slog.info("cancelled_orphaned_runs", count=cancelled)

    # Resolve OIDC authorization_endpoint from discovery or explicit config
    if settings.auth_provider in ("oidc", "entra"):
        if settings.oidc_authorization_endpoint:
            app.state.oidc_authorization_endpoint = settings.oidc_authorization_endpoint
        else:
            import httpx

            if settings.oidc_issuer:
                issuer = settings.oidc_issuer.rstrip("/")
            elif settings.auth_provider == "entra" and settings.entra_tenant_id:
                issuer = f"https://login.microsoftonline.com/{settings.entra_tenant_id}/v2.0"
            else:
                raise SystemExit("FATAL: OIDC discovery requires either oidc_issuer or entra_tenant_id to derive the issuer URL.")
            discovery_url = f"{issuer}/.well-known/openid-configuration"
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
                    resp = await client.get(discovery_url)
                    resp.raise_for_status()
                    app.state.oidc_authorization_endpoint = _validate_authorization_endpoint_discovery_document(resp.json())
            except (httpx.HTTPError, ValueError) as exc:
                raise SystemExit(
                    f"FATAL: OIDC discovery failed for issuer {issuer!r}: {exc}. "
                    f"Either fix the issuer URL or set oidc_authorization_endpoint explicitly."
                ) from exc

    # Sub-5: Construct ProgressBroadcaster and ExecutionServiceImpl
    # These require a running event loop, which is only available here.
    loop = asyncio.get_running_loop()
    broadcaster = ProgressBroadcaster(loop)
    app.state.broadcaster = broadcaster

    execution_service = ExecutionServiceImpl(
        loop=loop,
        broadcaster=broadcaster,
        settings=settings,
        session_service=session_service,
        yaml_generator=yaml_generator_module,
        blob_service=app.state.blob_service,
        secret_service=app.state.scoped_secret_resolver,
    )
    app.state.execution_service = execution_service

    # Periodic orphan cleanup — catches runs orphaned by SIGKILL/OOM
    # between restarts. Startup cleanup (above) handles the bulk case;
    # this catches runs orphaned while the server is still running.
    # Liveness-aware: excludes runs with active executor threads.
    orphan_task = asyncio.create_task(
        _periodic_orphan_cleanup(
            session_service,
            execution_service,
            interval_seconds=settings.orphan_run_check_interval_seconds,
            max_age_seconds=settings.orphan_run_max_age_seconds,
        )
    )

    yield

    # Cancel periodic cleanup before shutting down the executor
    orphan_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await orphan_task

    # Shutdown execution service thread pool without blocking the loop:
    # worker cleanup still schedules terminal-state writes back onto it.
    await execution_service.shutdown()


# Fields that accept JSON-encoded collection values from environment variables.
# Add any new tuple-typed WebSettings fields here so _settings_from_env()
# JSON-decodes them.  Scalar fields (str, int, float, Path) are handled by Pydantic.
_JSON_COLLECTION_FIELDS: frozenset[str] = frozenset({"cors_origins", "server_secret_allowlist"})


def _settings_from_env() -> WebSettings:
    """Construct WebSettings from ELSPETH_WEB__* environment variables.

    Called when create_app() is invoked without explicit settings (e.g.,
    by uvicorn's factory protocol).  The CLI sets these env vars before
    calling uvicorn.run().

    Collection-typed fields are JSON-decoded via ``_JSON_COLLECTION_FIELDS``.
    The JSON literal ``null`` is decoded to ``None`` for all fields — this is
    the env-var convention for "clear this optional setting."  Pydantic rejects
    ``None`` for non-nullable fields, so there is no silent mistyping risk.
    All other scalar values pass as raw strings; Pydantic coerces
    str→int, str→float, str→Path automatically.
    """
    kwargs: dict[str, object] = {}
    prefix = "ELSPETH_WEB__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            field_name = key[len(prefix) :].lower()
            if field_name in _JSON_COLLECTION_FIELDS:
                try:
                    parsed = json.loads(value)
                    kwargs[field_name] = tuple(parsed) if isinstance(parsed, list) else parsed
                except (json.JSONDecodeError, ValueError):
                    kwargs[field_name] = value
            elif value == "null":
                kwargs[field_name] = None
            else:
                kwargs[field_name] = value
    return WebSettings(**kwargs)


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

    app = FastAPI(title="ELSPETH Web", version="0.1.0", lifespan=lifespan)

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

    app.state.settings = settings

    # Ensure data directory and subdirectories exist before any DB access.
    # get_landscape_url() defaults to data_dir/runs/audit.db — SQLite does
    # not create parent directories, so we must ensure runs/ exists too.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / "runs").mkdir(exist_ok=True)

    # --- Catalog ---
    app.state.catalog_service = create_catalog_service()
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
    app.state.auth_provider = auth_provider
    app.state.oidc_authorization_endpoint = None  # Set by lifespan for OIDC/Entra

    # W16/S3: Secret key production guard -- hard crash
    if (
        settings.secret_key == "change-me-in-production"
        and settings.host not in _LOCAL_HOSTS
        and "pytest" not in sys.modules
        and os.environ.get("ELSPETH_ENV") != "test"
    ):
        raise SystemExit(
            "FATAL: WebSettings.secret_key is set to the default value. "
            "Set a secure secret_key before starting the web server. "
            "See WebSettings documentation."
        )

    # --- Session database setup ---
    session_db_url = settings.get_session_db_url()
    session_engine = create_session_engine(session_db_url)
    initialize_session_schema(session_engine)

    session_service = SessionServiceImpl(session_engine, data_dir=settings.data_dir)
    app.state.session_service = session_service

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
        session_engine=session_engine,
        secret_service=app.state.scoped_secret_resolver,
        runtime_preflight_coordinator=runtime_preflight_coordinator,
    )
    app.state.composer_availability = app.state.composer_service.get_availability()
    app.state.composer_progress_registry = ComposerProgressRegistry()

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
    app.include_router(create_blobs_router())
    app.include_router(create_secrets_router())
    app.include_router(create_execution_router())

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

        Defaults to the sentinel ``"unset"`` if the middleware did not
        run (e.g., a handler fires during a pre-middleware exception
        path), ensuring the response body is always JSON-serialisable.
        """
        return getattr(request.state, "request_id", "unset")

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

    @app.get("/api/system/status")
    async def system_status() -> dict[str, object]:
        composer = app.state.composer_availability
        return {
            "composer_available": composer.available,
            "composer_model": composer.model,
            "composer_provider": composer.provider,
            "composer_reason": composer.reason,
            "composer_missing_keys": list(composer.missing_keys),
        }

    # --- Static file serving for the React SPA (production) ---
    # Mount frontend/dist/ AFTER all API and WS routes so /api/* takes precedence.
    # Only active when the build output exists (i.e., after `npm run build`).
    frontend_dist = Path(__file__).parent / "frontend" / "dist"
    if frontend_dist.is_dir():
        from starlette.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="spa")

    return app
