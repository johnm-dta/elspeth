"""Shared fixtures for the composer capability-parity real-path matrix (Task 3).

This conftest builds ONE production stack that all parity surfaces share:

* a **real** ``ComposerServiceImpl`` built in WEB mode (operator_profile_registry
  + user-id snapshot factory, exactly as ``app.py`` ``create_app`` wires it — NOT
  ``for_trained_operator``, which has no profile registry and would reject the
  ``profile`` LLM authoring form only on the freeform surface) mounted as
  ``app.state.composer_service`` (so both freeform ``compose`` and guided-full's
  ``plan_guided_full_pipeline`` run the real planner against the same web policy),
  with ``_compute_availability`` forced available;
* a permissive-but-real web plugin policy that admits every plugin the nine
  fixtures use (``csv`` / ``json`` sources+sinks and ``llm`` are already in
  ``REQUIRED_WEB_PLUGIN_IDS``; ``passthrough`` / ``type_coerce`` / ``batch_stats``
  / ``batch_replicate`` are added to the allowlist). The guided conftest
  deliberately restricts to ``transform:passthrough`` — this one must not;
* the scripted deterministic completion double lifted from
  ``tests/unit/web/composer/test_pipeline_planner.py`` (its stateful sequential
  form is what the guided-staged repair cases will need), patched onto the
  module global ``elspeth.web.composer.service._litellm_acompletion`` so the
  real planner response parser, custody, candidate validation, and audited
  ``set_pipeline`` commit are all exercised;
* the freeform recipe fast-path bypass (``match_freeform_recipe_intent`` → None)
  so freeform provably traverses ``plan_pipeline`` +
  ``build_planner_capability_manifest`` rather than a recipe-router graph
  (design false-green trap #2).

Each surface adapter starts from the same fixture intent, drives its real
production entrypoint, and returns the committed ``CompositionState`` read back
from the immutable state store — never an injected ``PipelineProposal`` or
``CompositionState``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
import structlog
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.auth.middleware import get_current_user
from elspeth.web.auth.models import UserIdentity
from elspeth.web.blobs.service import BlobServiceImpl
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.progress import ComposerProgressRegistry
from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools._dispatch import execute_tool
from elspeth.web.config import WebSettings
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.middleware.rate_limit import ComposerRateLimiter
from elspeth.web.plugin_policy.availability import build_plugin_snapshot
from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.routes import create_session_router
from elspeth.web.sessions.routes._helpers import _state_from_record
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

# --------------------------------------------------------------------------- #
# Deterministic completion double (lifted from test_pipeline_planner.py)       #
# --------------------------------------------------------------------------- #


@dataclass
class _Function:
    name: str
    arguments: object


@dataclass
class _ToolCall:
    id: str
    function: _Function | None


@dataclass
class _Message:
    content: str | None
    tool_calls: list[_ToolCall] | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list[_Choice]
    usage: Mapping[str, object]
    model: str | None = "provider/planner-v1"
    id: str = "parity-request-1"


class _ScriptedCompletion:
    """Stateful sequential responder: one popped ``_Response`` per provider call.

    The sequential form (multiple queued responses) is what the guided-staged
    one-repair / repair-exhaustion cases in the next stage will rely on; the
    positive matrix queues exactly one terminal ``emit_pipeline_proposal``.
    """

    def __init__(self, *responses: _Response | BaseException) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> _Response:
        self.requests.append(copy.deepcopy(kwargs))
        if not self._responses:
            raise AssertionError("scripted completion exhausted: real path made more provider calls than scripted")
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def emit_proposal_response(pipeline: Mapping[str, Any]) -> _Response:
    """A terminal response calling ``emit_pipeline_proposal`` with ``pipeline``."""
    return _Response(
        choices=[
            _Choice(
                message=_Message(
                    content=None,
                    tool_calls=[
                        _ToolCall(
                            id="parity-terminal",
                            function=_Function(
                                name="emit_pipeline_proposal",
                                arguments=json.dumps({"pipeline": pipeline}),
                            ),
                        )
                    ],
                )
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01},
    )


# --------------------------------------------------------------------------- #
# Corpus loading                                                               #
# --------------------------------------------------------------------------- #

# parity/conftest.py -> composer -> web -> integration -> tests -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURES_DIR = REPO_ROOT / "evals" / "composer-parity" / "fixtures"


def load_parity_fixtures() -> list[dict[str, Any]]:
    """Load the nine canonical class fixtures, sorted by class for stable ids."""
    fixtures = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(FIXTURES_DIR.glob("*.json"))]
    if len(fixtures) != 9:  # pragma: no cover - corpus contract
        raise AssertionError(f"expected 9 parity fixtures, found {len(fixtures)}")
    return fixtures


PARITY_FIXTURES = load_parity_fixtures()


def rewrite_source_paths(args: Mapping[str, Any], data_dir: Path) -> dict[str, Any]:
    """Rebind every source ``path`` under ``{data_dir}/blobs/`` (the S2 allowlist).

    The committed ``set_pipeline`` dispatch enforces that source file paths live
    under ``{data_dir}/blobs/``; the corpus stores abstract relative names. Both
    surfaces receive the same rewrite so their committed graphs stay identical,
    and the isomorphism helper canonicalizes paths to basename regardless.
    """
    rewritten = copy.deepcopy(dict(args))
    blobs = data_dir / "blobs"

    def fix(spec: dict[str, Any]) -> None:
        options = spec.get("options")
        if isinstance(options, dict) and isinstance(options.get("path"), str):
            options["path"] = str(blobs / Path(options["path"]).name)

    if isinstance(rewritten.get("source"), dict):
        fix(rewritten["source"])
    if isinstance(rewritten.get("sources"), dict):
        for named in rewritten["sources"].values():
            if isinstance(named, dict):
                fix(named)
    return rewritten


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


# --------------------------------------------------------------------------- #
# Shared production stack + per-surface adapters                               #
# --------------------------------------------------------------------------- #

_PARITY_ALLOWLIST = (
    "transform:passthrough",
    "transform:type_coerce",
    "transform:batch_stats",
    "transform:batch_replicate",
)


class ParityEnv:
    """Real production stack plus the freeform + guided-full surface adapters."""

    def __init__(
        self,
        *,
        app: FastAPI,
        composer: ComposerServiceImpl,
        sessions: SessionServiceImpl,
        data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.app = app
        self.composer = composer
        self.sessions = sessions
        self.data_dir = data_dir
        self.monkeypatch = monkeypatch

    def _client(self) -> AsyncClient:
        return AsyncClient(transport=ASGITransport(app=self.app), base_url="http://parity")

    def _script(self, fixture: Mapping[str, Any]) -> Mapping[str, Any]:
        """Patch the completion global to emit this fixture's pipeline; return it."""
        pipeline = rewrite_source_paths(fixture["canonical_arguments"], self.data_dir)
        completion = _ScriptedCompletion(emit_proposal_response(pipeline))
        self.monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)
        return pipeline

    def reference_state(self, fixture: Mapping[str, Any]) -> CompositionState:
        """Ground-truth committed graph: fixture args through set_pipeline directly.

        Built by dispatching ``set_pipeline`` with the fixture's canonical
        arguments against the SAME web plugin policy the surfaces commit through
        (so operator-profile lowering — e.g. the ``llm`` node's ``profile``
        alias — resolves identically). Each surface's real-path committed graph
        must be isomorphic to this, which anchors cross-surface parity.
        """
        pipeline = rewrite_source_paths(fixture["canonical_arguments"], self.data_dir)
        user = UserIdentity(user_id="alice", username="alice")
        snapshot = self.app.state.plugin_snapshot_factory(user)
        policy = PolicyCatalogView(
            self.app.state.catalog_service,
            snapshot,
            self.app.state.operator_profile_registry,
        )
        result = execute_tool(
            "set_pipeline",
            pipeline,
            _empty_state(),
            policy,
            plugin_snapshot=snapshot,
            data_dir=str(self.data_dir),
            session_engine=self.app.state.session_engine,
            secret_service=None,
            user_id="alice",
        )
        if not result.success:
            errors = [(e.component, e.message) for e in result.validation.errors] if result.validation else result.data
            raise AssertionError(f"reference set_pipeline failed for {fixture['class']}: {errors}")
        return result.updated_state

    async def _committed_state(self, session_id: UUID) -> CompositionState:
        record = await self.sessions.get_current_state(session_id)
        if record is None:
            raise AssertionError("no committed composition state after acceptance")
        return _state_from_record(record)

    async def drive(self, surface: str, fixture: Mapping[str, Any]) -> CompositionState:
        if surface == "freeform":
            return await self.drive_freeform(fixture)
        if surface == "guided_full":
            return await self.drive_guided_full(fixture)
        raise AssertionError(f"unknown parity surface {surface!r}")  # pragma: no cover

    async def drive_freeform(self, fixture: Mapping[str, Any]) -> CompositionState:
        """Freeform: real ``compose()`` empty-build path → proposal → accept.

        The reworded imperative intent trips ``_user_request_expects_pipeline_mutation``
        and the recipe bypass guarantees a non-match, so ``compose`` enters
        ``_plan_and_stage_empty_pipeline`` → ``plan_pipeline`` (real planner) and
        stages one pending proposal, which is then accepted over the real HTTP
        route.
        """
        self._script(fixture)
        session = await self.sessions.create_session("alice", "Alice", "local")
        await self.sessions.update_composer_preferences(
            session.id,
            trust_mode="explicit_approve",
            density_default="high",
            actor="test",
        )
        user_message = await self.sessions.add_message(
            session.id,
            "user",
            fixture["intent"],
            writer_principal="route_user_message",
        )
        await self.composer.compose(
            fixture["intent"],
            [],
            _empty_state(),
            session_id=str(session.id),
            user_id="alice",
            user_message_id=str(user_message.id),
        )
        proposals = await self.sessions.list_composition_proposals(session.id, status="pending")
        if len(proposals) != 1:
            raise AssertionError(
                f"freeform did not stage exactly one proposal for {fixture['class']} "
                f"(got {len(proposals)}) — it may have fallen through to the compose loop"
            )
        proposal = proposals[0]
        surface_value = getattr(proposal.pipeline_metadata, "surface", None) if proposal.pipeline_metadata is not None else None
        surface_value = getattr(surface_value, "value", surface_value)
        if surface_value != "freeform":
            raise AssertionError(f"freeform proposal recorded surface {surface_value!r}, not 'freeform'")
        async with self._client() as client:
            response = await client.post(
                f"/api/sessions/{session.id}/proposals/{proposal.id}/accept",
                json={"draft_hash": proposal.pipeline_metadata.draft_hash},
            )
        if response.status_code != 200:
            raise AssertionError(f"freeform accept failed ({response.status_code}): {response.text}")
        return await self._committed_state(session.id)

    async def drive_guided_full(self, fixture: Mapping[str, Any]) -> CompositionState:
        """Guided-full: authenticated ``POST /guided/plan`` → proposal → accept."""
        self._script(fixture)
        async with self._client() as client:
            created = await client.post("/api/sessions", json={"title": "parity guided-full"})
            if created.status_code != 201:
                raise AssertionError(f"session create failed ({created.status_code}): {created.text}")
            session_id = created.json()["id"]
            plan = await client.post(
                f"/api/sessions/{session_id}/guided/plan",
                json={"operation_id": str(uuid4()), "intent": fixture["intent"]},
            )
            if plan.status_code != 200:
                raise AssertionError(f"guided-full plan failed ({plan.status_code}): {plan.text}")
            payload = plan.json()
            if payload["pipeline_metadata"]["surface"] != "guided_full":
                raise AssertionError(f"guided-full proposal recorded surface {payload['pipeline_metadata']['surface']!r}")
            accept = await client.post(
                f"/api/sessions/{session_id}/proposals/{payload['id']}/accept",
                json={"draft_hash": payload["pipeline_metadata"]["draft_hash"]},
            )
            if accept.status_code != 200:
                raise AssertionError(f"guided-full accept failed ({accept.status_code}): {accept.text}")
        return await self._committed_state(UUID(session_id))


def _build_settings(data_dir: Path) -> WebSettings:
    return WebSettings(
        data_dir=data_dir,
        composer_model="test/planner",
        composer_boot_probe_enabled=False,
        composer_max_composition_turns=4,
        composer_max_discovery_turns=3,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=100,
        shareable_link_signing_key=b"\x00" * 32,
        plugin_allowlist=_PARITY_ALLOWLIST,
        llm_profiles={
            "task-role": {
                "provider": "bedrock",
                "model": "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            }
        },
    )


@pytest.fixture
def parity_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ParityEnv:
    """Build the shared real production stack with the two false-green bypasses."""
    engine = create_session_engine(f"sqlite:///{tmp_path / 'sessions.sqlite3'}")
    initialize_session_schema(engine)
    sessions = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.parity"),
    )
    settings = _build_settings(tmp_path)
    catalog = create_catalog_service()
    runtime_policy = RuntimeWebPluginConfig.from_settings(settings)
    web_plugin_policy = compile_web_plugin_policy(
        registry=get_shared_plugin_manager(),
        settings=runtime_policy,
    )
    operator_profile_registry = OperatorProfileRegistry(
        policy=web_plugin_policy,
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

    def build_snapshot(user_id: str) -> PluginAvailabilitySnapshot:
        return build_plugin_snapshot(
            policy=web_plugin_policy,
            catalog=catalog,
            profiles=operator_profile_registry,
            principal_scope=f"local:{user_id}",
            secret_inventory=_EmptyInventory(),
            generation_key=b"parity-integration-policy-key",
        )

    # Force provider availability BEFORE constructing the service (its __init__
    # calls _compute_availability).
    monkeypatch.setattr(
        ComposerServiceImpl,
        "_compute_availability",
        lambda _self: ComposerAvailability(available=True, provider="test", model="test/planner", reason=None),
    )
    # Freeform recipe fast-path bypass (false-green trap #2): guarantee non-match
    # so freeform provably traverses plan_pipeline + build_planner_capability_manifest.
    monkeypatch.setattr("elspeth.web.composer.service.match_freeform_recipe_intent", lambda _message: None)

    # Production wires the composer service in WEB mode — operator_profile_registry
    # plus a user-id-keyed snapshot factory (app.py create_app), NOT
    # for_trained_operator. Match that so freeform PLANNING and guided-full
    # PLANNING both validate the operator-profiled ``llm`` node against the same
    # web plugin policy (trained-operator mode has no profile registry and would
    # reject the ``profile`` authoring form only on the freeform surface).
    composer = ComposerServiceImpl(
        catalog=catalog,
        settings=settings,
        sessions_service=sessions,
        session_engine=engine,
        secret_service=None,
        plugin_snapshot_factory=build_snapshot,
        operator_profile_registry=operator_profile_registry,
    )

    app = FastAPI()

    async def mock_user() -> UserIdentity:
        return UserIdentity(user_id="alice", username="alice")

    app.dependency_overrides[get_current_user] = mock_user
    app.state.session_service = sessions
    app.state.session_engine = engine
    app.state.blob_service = BlobServiceImpl(engine, tmp_path)
    app.state.payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    app.state.scoped_secret_resolver = None
    app.state.settings = settings
    app.state.composer_service = composer
    app.state.rate_limiter = ComposerRateLimiter(limit=1000)
    app.state.catalog_service = catalog
    app.state.web_plugin_policy = web_plugin_policy
    app.state.operator_profile_registry = operator_profile_registry
    app.state.plugin_snapshot_factory = lambda user: build_snapshot(user.user_id)
    app.state.composer_recorder = BufferingRecorder()
    app.state.composer_progress_registry = ComposerProgressRegistry()
    app.include_router(create_session_router())

    try:
        yield ParityEnv(
            app=app,
            composer=composer,
            sessions=sessions,
            data_dir=tmp_path,
            monkeypatch=monkeypatch,
        )
    finally:
        engine.dispose()
