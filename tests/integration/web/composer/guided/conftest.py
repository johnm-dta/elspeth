"""Fixtures for HTTP-layer guided endpoint integration tests.

Establishes SyncASGITestClient-based fixtures for testing the
/api/sessions/{id}/guided/* endpoint routes. Replicates the pattern from
tests/unit/web/sessions/test_fork.py: in-memory SQLite, mock auth, minimal
FastAPI app with session router mounted.

Scope: Fixtures live in this file and are available to all tests in
tests/integration/web/composer/guided/*.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from fastapi import FastAPI
from testcontainers.postgres import PostgresContainer

from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import stable_hash
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.routes import create_blobs_router
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.planning import guided_private_reviewed_facts
from elspeth.web.composer.pipeline_planner import PipelinePlanResult
from elspeth.web.composer.pipeline_proposal import PipelineProposal, PlannerSurface
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def composer_test_client(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[TestClient]:
    """Yields a TestClient wrapping a minimal FastAPI app with session router.

    App state includes:
    - session_service (SessionServiceImpl with in-memory SQLite)
    - blob_service (BlobServiceImpl)
    - auth override: all requests authenticated as "alice"
    - rate_limiter (ComposerRateLimiter)
    - settings (WebSettings with test defaults)
    - audit_recorder (BufferingRecorder for test inspection)

    The app has create_session_router() mounted so POST /api/sessions routes
    are available for fixture smoke tests and guided endpoint tests.

    Usage:
        def test_something(composer_test_client):
            resp = composer_test_client.post("/api/sessions", json={"title": "x"})
            assert resp.status_code == 201
    """
    backend = getattr(request, "param", "sqlite")
    postgres: PostgresContainer | None = None
    if backend == "sqlite":
        # Use independent SQLite connections so route-race tests exercise the
        # same transaction/locking boundary as a deployed file-backed database.
        engine = create_session_engine(f"sqlite:///{tmp_path / 'sessions.sqlite3'}")
    elif backend == "postgres":
        postgres = PostgresContainer("postgres:16-alpine")
        postgres.start()
        engine = create_session_engine(postgres.get_connection_url())
    else:  # pragma: no cover - fixture contract
        raise AssertionError(f"unsupported guided integration backend: {backend}")
    initialize_session_schema(engine)
    database_url = str(engine.url)
    engines_to_dispose = [engine]

    # Session and blob services
    session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.conftest"),
    )
    blob_service = BlobServiceImpl(engine, tmp_path)

    # FastAPI app
    app = FastAPI()

    # Mock auth: all requests authenticated as "alice"
    identity = UserIdentity(user_id="alice", username="alice")

    async def mock_user() -> UserIdentity:
        return identity

    app.dependency_overrides[get_current_user] = mock_user

    # App state: minimal set required by session router
    app.state.session_service = session_service
    app.state.session_engine = engine  # for guided step-2.5 recipe application
    app.state.blob_service = blob_service
    app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    app.state.scoped_secret_resolver = None
    app.state.settings = WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=15,
        composer_max_discovery_turns=10,
        composer_timeout_seconds=85.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        # The legacy guided happy-path fixtures intentionally exercise an
        # identity transform.  Make that optional plugin explicitly part of
        # this test operator's web policy; policy-rejection cases override the
        # request snapshot with a restricted one.
        plugin_allowlist=("transform:passthrough",),
        llm_profiles={
            "task-role": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
    )

    class _DeterministicGuidedPlanner:
        """Explicit test double for the shared planner route seam."""

        async def plan_guided_full_pipeline(self, *, base, recorder, policy_catalog, **_kwargs):
            pipeline = {
                "sources": {
                    "source": {
                        "plugin": "csv",
                        "options": {"path": "/data/input.csv"},
                        "on_success": "results",
                        "on_validation_failure": "discard",
                    }
                },
                "nodes": [],
                "edges": [],
                "outputs": [
                    {
                        "sink_name": "results",
                        "plugin": "json",
                        "options": {"path": "/data/results.jsonl"},
                        "on_write_failure": "discard",
                    }
                ],
            }
            proposal = PipelineProposal.create(
                pipeline=pipeline,
                base=base,
                reviewed_facts={},
                surface=PlannerSurface.GUIDED_FULL,
                repair_count=0,
                skill_hash=stable_hash("deterministic-guided-full-test-planner"),
                covered_deferred_intent_ids=(),
                supersedes_draft_hash=None,
            )
            return (
                PipelinePlanResult(
                    proposal=proposal,
                    tool_call_id=f"guided-full-test-{proposal.draft_hash[:16]}",
                    custody_result="not_required",
                    model_identifier="deterministic-guided-full-test-planner",
                    model_version="v1",
                    provider="test",
                ),
                {
                    "source": frozenset(item.name for item in policy_catalog.list_sources()),
                    "transform": frozenset(item.name for item in policy_catalog.list_transforms()),
                    "sink": frozenset(item.name for item in policy_catalog.list_sinks()),
                },
            )

        async def plan_guided_pipeline(self, *, guided, base, supersedes_draft_hash, recorder, **_kwargs):
            output_names = [guided.reviewed_outputs[stable_id].name for stable_id in guided.output_order]
            pipeline = {
                "sources": {
                    guided.reviewed_sources[stable_id].name: {
                        "plugin": guided.reviewed_sources[stable_id].plugin,
                        "options": deep_thaw(guided.reviewed_sources[stable_id].options),
                        "on_success": output_names[index % len(output_names)],
                        "on_validation_failure": guided.reviewed_sources[stable_id].on_validation_failure,
                    }
                    for index, stable_id in enumerate(guided.source_order)
                },
                "nodes": [],
                "edges": [],
                "outputs": [
                    {
                        "sink_name": guided.reviewed_outputs[stable_id].name,
                        "plugin": guided.reviewed_outputs[stable_id].plugin,
                        "options": deep_thaw(guided.reviewed_outputs[stable_id].options),
                        "on_write_failure": guided.reviewed_outputs[stable_id].on_write_failure,
                    }
                    for stable_id in guided.output_order
                ],
            }
            proposal = PipelineProposal.create(
                pipeline=pipeline,
                base=base,
                reviewed_facts=guided_private_reviewed_facts(guided),
                surface=PlannerSurface.GUIDED_STAGED,
                repair_count=0,
                skill_hash=stable_hash("deterministic-guided-test-planner"),
                covered_deferred_intent_ids=tuple(intent.intent_id for intent in guided.deferred_intents),
                supersedes_draft_hash=supersedes_draft_hash,
            )
            return (
                PipelinePlanResult(
                    proposal=proposal,
                    tool_call_id=f"guided-test-{proposal.draft_hash[:16]}",
                    custody_result="not_required",
                    model_identifier="deterministic-guided-test-planner",
                    model_version="v1",
                    provider="test",
                ),
                {
                    "source": frozenset(source.plugin for source in guided.reviewed_sources.values()),
                    "transform": frozenset(),
                    "sink": frozenset(output.plugin for output in guided.reviewed_outputs.values()),
                },
            )

    app.state.composer_service = _DeterministicGuidedPlanner()
    app.state.rate_limiter = ComposerRateLimiter(limit=100)
    app.state.catalog_service = create_catalog_service()
    runtime_policy = RuntimeWebPluginConfig.from_settings(app.state.settings)
    app.state.web_plugin_policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=runtime_policy,
    )
    app.state.operator_profile_registry = OperatorProfileRegistry(
        policy=app.state.web_plugin_policy,
        settings=runtime_policy,
    )

    class _EmptyInventory:
        def has_server_ref(self, name: str) -> bool:
            return False

        def has_user_ref(self, principal: str, name: str) -> bool:
            return False

        def has_ref(self, principal: str, name: str) -> bool:
            return False

        def server_generation(self, name: str) -> str | None:
            return None

        def user_generation(self, principal: str, name: str) -> str | None:
            return None

    app.state.plugin_snapshot_factory = lambda user: build_plugin_snapshot(
        policy=app.state.web_plugin_policy,
        catalog=app.state.catalog_service,
        profiles=app.state.operator_profile_registry,
        principal_scope=f"local:{user.user_id}",
        secret_inventory=_EmptyInventory(),
        generation_key=b"guided-integration-policy-key",
    )

    # Audit recorder for test inspection (Phase 3 Task 3.4 will wire this)
    app.state.composer_recorder = BufferingRecorder()

    # post_guided_chat (elspeth-a8eeebb3aa) publishes composer-progress
    # snapshots the same way the freeform send_message route does — mirrors
    # the create_app() production wiring at app.py:930. Without this the
    # guided/chat route's unconditional _get_composer_progress_registry(request)
    # call AttributeErrors against this hand-rolled minimal app.
    app.state.composer_progress_registry = ComposerProgressRegistry()

    # Mount session router (sessions + guided endpoints)
    router = create_session_router()
    app.include_router(router)

    # Mount blobs router so tests can upload blobs via /api/sessions/{id}/blobs/inline
    blobs_router = create_blobs_router()
    app.include_router(blobs_router)

    def restart_test_client() -> TestClient:
        """Rebuild the HTTP stack over the fixture's persisted stores."""
        engines_to_dispose[-1].dispose()
        restarted_engine = create_session_engine(database_url)
        initialize_session_schema(restarted_engine)
        engines_to_dispose.append(restarted_engine)

        restarted_app = FastAPI()

        async def restarted_mock_user() -> UserIdentity:
            return UserIdentity(user_id="alice", username="alice")

        restarted_app.dependency_overrides[get_current_user] = restarted_mock_user
        restarted_app.state.session_service = SessionServiceImpl(
            restarted_engine,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test.guided.conftest.restarted"),
        )
        restarted_app.state.session_engine = restarted_engine
        restarted_app.state.blob_service = BlobServiceImpl(restarted_engine, tmp_path)
        restarted_app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        restarted_app.state.scoped_secret_resolver = None
        restarted_app.state.settings = app.state.settings
        restarted_app.state.composer_service = type(app.state.composer_service)()
        restarted_app.state.rate_limiter = ComposerRateLimiter(limit=100)
        restarted_app.state.catalog_service = create_catalog_service()
        restarted_runtime_policy = RuntimeWebPluginConfig.from_settings(restarted_app.state.settings)
        restarted_app.state.web_plugin_policy = compile_web_plugin_policy(
            registry=get_shared_plugin_manager(),
            settings=restarted_runtime_policy,
        )
        restarted_app.state.operator_profile_registry = OperatorProfileRegistry(
            policy=restarted_app.state.web_plugin_policy,
            settings=restarted_runtime_policy,
        )
        restarted_app.state.plugin_snapshot_factory = lambda user: build_plugin_snapshot(
            policy=restarted_app.state.web_plugin_policy,
            catalog=restarted_app.state.catalog_service,
            profiles=restarted_app.state.operator_profile_registry,
            principal_scope=f"local:{user.user_id}",
            secret_inventory=_EmptyInventory(),
            generation_key=b"guided-integration-policy-key",
        )
        restarted_app.state.composer_recorder = BufferingRecorder()
        restarted_app.state.composer_progress_registry = ComposerProgressRegistry()
        restarted_app.state.restart_test_client = restart_test_client
        restarted_app.include_router(create_session_router())
        restarted_app.include_router(create_blobs_router())
        return TestClient(restarted_app)

    # Wrap in TestClient and yield. Tests that need to prove durable recovery
    # can replace it with a wholly new client/app/service stack mid-journey.
    app.state.restart_test_client = restart_test_client
    client = TestClient(app)
    try:
        yield client
    finally:
        for fixture_engine in engines_to_dispose:
            fixture_engine.dispose()
        if postgres is not None:
            postgres.stop()


@pytest.fixture
def audit_recorder(composer_test_client: TestClient) -> BufferingRecorder:
    """Yields the BufferingRecorder from the app state.

    Lets test code inspect emitted ComposerToolInvocation records during
    a request. The recorder is accessible via the app's state.

    Usage:
        def test_something(composer_test_client, audit_recorder):
            # Make a request that triggers composer operations
            resp = composer_test_client.post(...)
            # Inspect recorded invocations
            invocations = audit_recorder.invocations
            assert len(invocations) > 0
            assert invocations[0].tool_name == "..."
    """
    # Extract recorder from the test client's app state
    app = composer_test_client.app
    recorder: BufferingRecorder = app.state.composer_recorder
    return recorder
