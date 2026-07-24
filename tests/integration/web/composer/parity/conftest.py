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

# Re-register the guided suite's restart-capable HTTP fixture for this package so
# the deferred-intent negatives in ``test_repair_and_deferral.py`` can request it
# (guided/'s conftest is a sibling scope and not inherited here). Importing it in
# the test module instead would shadow the fixture parameter and trip ruff F811.
from tests.integration.web.composer.guided.conftest import composer_test_client  # noqa: F401

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
_REFERENCE_SESSION_ID = "00000000-0000-4000-8000-000000000001"


def rewrite_source_paths(args: Mapping[str, Any], data_dir: Path, session_id: str) -> dict[str, Any]:
    """Rebind source paths under the caller's session-owned blob subtree.

    The committed ``set_pipeline`` dispatch enforces that source file paths live
    under ``{data_dir}/blobs/{session_id}/``; the corpus stores abstract relative
    names. Both surfaces receive the same rewrite so their committed graphs stay
    identical, and the isomorphism helper canonicalizes paths to basename regardless.
    """
    rewritten = copy.deepcopy(dict(args))
    blobs = data_dir / "blobs" / session_id

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
# Guided-staged naming + candidate derivation                                 #
# --------------------------------------------------------------------------- #
#
# The structured guided stage protocol reviews sources and outputs one at a
# time and auto-assigns their names positionally: sources become
# ``source`` / ``source_2`` / … and outputs become ``output`` / ``output_2`` /
# … in review (add) order — the operator cannot name them. ``set_pipeline`` on
# the reviewed candidate (``bind_guided_reviewed_components``) then OVERWRITES
# the scripted planner candidate's source/output plugin+options with the
# reviewed authority, keeping only the candidate's *topology* (node ids, edges,
# and each source's ``on_success`` wiring). So for guided-staged to derive the
# same canonical graph the scripted candidate must (a) name its sources/outputs
# with the guided defaults and (b) rewrite every routing target that points at
# a source/output to the guided default — while free node-to-node connection
# names (``rows``, ``stats``, ``gate_in``, …) stay verbatim (the comparator
# canonicalizes them). The committed graph is then isomorphic to the reference;
# only the component *names* differ, which §8.1 canonicalizes away.


def _guided_naming(args: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Map canonical source/output names to their guided positional defaults."""
    if isinstance(args.get("sources"), dict):
        source_names = list(args["sources"].keys())
    else:
        source_names = ["source"]
    output_names = [output["sink_name"] for output in args["outputs"]]
    source_map = {name: ("source" if index == 0 else f"source_{index + 1}") for index, name in enumerate(source_names)}
    output_map = {name: ("output" if index == 0 else f"output_{index + 1}") for index, name in enumerate(output_names)}
    return source_map, output_map


def _derive_guided_candidate(
    args: Mapping[str, Any],
    source_map: Mapping[str, str],
    output_map: Mapping[str, str],
    data_dir: Path,
    session_id: str,
) -> dict[str, Any]:
    """Rename the canonical pipeline into the guided candidate the planner emits.

    Role-based substitution (never a blind string replace): source dict keys and
    output ``sink_name`` are renamed to guided defaults; every *routing target*
    that names an output (``on_success`` / ``on_error`` / gate routes / fork
    targets / branch targets / ``on_write_failure``) is rewritten only when it
    equals an output name; explicit edge endpoints are rewritten when they name a
    source or output. Node ids, node options, and free connection names are
    emitted verbatim so the committed graph stays isomorphic to the reference.
    """

    def route(value: Any) -> Any:
        # Routing targets never name a source; only outputs (or free connection
        # names, which are left untouched).
        return output_map.get(value, value) if isinstance(value, str) else value

    endpoint = {**source_map, **output_map}

    if isinstance(args.get("sources"), dict):
        source_items = list(args["sources"].items())
        plural = True
    else:
        source_items = [("source", args["source"])]
        plural = False

    candidate_sources: dict[str, Any] = {}
    for name, spec in source_items:
        entry = copy.deepcopy(dict(spec))
        options = entry.get("options")
        if isinstance(options, dict) and isinstance(options.get("path"), str):
            entry["options"] = {**options, "path": str(data_dir / "blobs" / session_id / Path(options["path"]).name)}
        if "on_success" in entry:
            entry["on_success"] = route(entry["on_success"])
        if "on_validation_failure" in entry:
            entry["on_validation_failure"] = route(entry["on_validation_failure"])
        candidate_sources[source_map[name]] = entry

    candidate_nodes: list[dict[str, Any]] = []
    for node in args.get("nodes", []):
        entry = copy.deepcopy(dict(node))
        # The guided proposal wire projection reconstructs the committed candidate
        # via the strict ``CompositionState.from_dict`` (unlike the lenient
        # ``set_pipeline`` tool the freeform/guided-full accept path uses), which
        # requires every node to carry ``plugin`` / ``on_success`` / ``on_error``
        # / ``options``. The fixtures omit them on gate/queue/coalesce nodes; the
        # emit schema accepts explicit nulls (``["string","null"]``), so fill them.
        entry.setdefault("plugin", None)
        entry.setdefault("on_success", None)
        entry.setdefault("on_error", None)
        entry.setdefault("options", {})
        if entry.get("on_success") is not None:
            entry["on_success"] = route(entry["on_success"])
        if entry.get("on_error") is not None:
            entry["on_error"] = route(entry["on_error"])
        if isinstance(entry.get("routes"), dict):
            entry["routes"] = {key: route(value) for key, value in entry["routes"].items()}
        if entry.get("fork_to"):
            entry["fork_to"] = [route(value) for value in entry["fork_to"]]
        if isinstance(entry.get("branches"), dict):
            entry["branches"] = {key: route(value) for key, value in entry["branches"].items()}
        candidate_nodes.append(entry)

    candidate_edges: list[dict[str, Any]] = []
    for edge in args.get("edges", []):
        entry = copy.deepcopy(dict(edge))
        entry["from_node"] = endpoint.get(entry["from_node"], entry["from_node"])
        entry["to_node"] = endpoint.get(entry["to_node"], entry["to_node"])
        candidate_edges.append(entry)

    candidate_outputs: list[dict[str, Any]] = []
    for output in args["outputs"]:
        entry = copy.deepcopy(dict(output))
        entry["sink_name"] = output_map[entry["sink_name"]]
        if entry.get("on_write_failure") is not None:
            entry["on_write_failure"] = route(entry["on_write_failure"])
        candidate_outputs.append(entry)

    candidate: dict[str, Any] = {"nodes": candidate_nodes, "edges": candidate_edges, "outputs": candidate_outputs}
    if plural:
        candidate["sources"] = candidate_sources
    else:
        candidate["source"] = candidate_sources["source"]
    if "metadata" in args:
        candidate["metadata"] = copy.deepcopy(dict(args["metadata"]))
    return candidate


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

    def _script(self, fixture: Mapping[str, Any], session_id: str) -> Mapping[str, Any]:
        """Patch the completion global to emit this fixture's pipeline; return it."""
        pipeline = rewrite_source_paths(fixture["canonical_arguments"], self.data_dir, session_id)
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
        pipeline = rewrite_source_paths(fixture["canonical_arguments"], self.data_dir, _REFERENCE_SESSION_ID)
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
            session_id=_REFERENCE_SESSION_ID,
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
        if surface == "guided_staged":
            return await self.drive_guided_staged(fixture)
        raise AssertionError(f"unknown parity surface {surface!r}")  # pragma: no cover

    async def drive_freeform(self, fixture: Mapping[str, Any]) -> CompositionState:
        """Freeform: real ``compose()`` empty-build path → proposal → accept.

        The reworded imperative intent trips ``_user_request_expects_pipeline_mutation``
        and the recipe bypass guarantees a non-match, so ``compose`` enters
        ``_plan_and_stage_empty_pipeline`` → ``plan_pipeline`` (real planner) and
        stages one pending proposal, which is then accepted over the real HTTP
        route.
        """
        session = await self.sessions.create_session("alice", "Alice", "local")
        self._script(fixture, str(session.id))
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
        async with self._client() as client:
            created = await client.post("/api/sessions", json={"title": "parity guided-full"})
            if created.status_code != 201:
                raise AssertionError(f"session create failed ({created.status_code}): {created.text}")
            session_id = created.json()["id"]
            self._script(fixture, session_id)
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

    # --- guided-staged (the persisted multi-request stage protocol) --------- #

    async def _staged_respond(self, client: AsyncClient, session_id: str, **fields: Any) -> dict[str, Any]:
        """GET the current turn's token then POST one fenced ``/guided/respond``."""
        current = await client.get(f"/api/sessions/{session_id}/guided")
        if current.status_code != 200:
            raise AssertionError(f"guided-staged GET failed ({current.status_code}): {current.text}")
        turn = current.json()["next_turn"]
        body: dict[str, Any] = {
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"] if turn is not None else None,
            **fields,
        }
        resp = await client.post(f"/api/sessions/{session_id}/guided/respond", json=body)
        if resp.status_code != 200:
            raise AssertionError(f"guided-staged respond {sorted(fields)} failed ({resp.status_code}): {resp.text}")
        return resp.json()

    def _source_review_options(self, spec: Mapping[str, Any], output_map: Mapping[str, str], session_id: str) -> dict[str, Any]:
        """Build the Step-1 SCHEMA_FORM options that review one source verbatim.

        ``on_validation_failure`` rides inside the submitted options (the
        transition splits it out into the reviewed source's structural field);
        the source ``path`` is rebound under ``{data_dir}/blobs/`` (the S2 commit
        allowlist), and a validation failure routed to a sink is remapped to that
        sink's guided default name.
        """
        options = copy.deepcopy(dict(spec["options"]))
        if isinstance(options.get("path"), str):
            options["path"] = str(self.data_dir / "blobs" / session_id / Path(options["path"]).name)
        failure = spec.get("on_validation_failure", "discard")
        options["on_validation_failure"] = output_map.get(failure, failure)
        return options

    def _output_review_options(self, output: Mapping[str, Any], output_map: Mapping[str, str], session_id: str) -> dict[str, Any]:
        """Build the Step-2 SCHEMA_FORM options that review one output verbatim."""
        options = copy.deepcopy(dict(output["options"]))
        if isinstance(options.get("path"), str):
            options["path"] = str(self.data_dir / "outputs" / session_id / Path(options["path"]).name)
        failure = output.get("on_write_failure", "discard")
        options["on_write_failure"] = output_map.get(failure, failure)
        return options

    async def drive_guided_staged(self, fixture: Mapping[str, Any], *, start_profile: str | None = None) -> CompositionState:
        """Guided-staged: drive the persisted stage protocol to the sole commit.

        ``start_profile`` (when given) explicitly opens the guided session with
        that workflow profile via ``POST /guided/start`` before the first turn is
        fetched — the tutorial-identity negative passes ``"tutorial"`` so the sole
        planner call runs on the ``TUTORIAL_PROFILE`` surface with its frozen
        lesson, while the reviewed components and the committed graph stay
        identical to the ``live`` staged run. When ``None`` the first GET
        implicitly opens a ``live`` session (the positive-matrix behaviour).

        ``/guided/start`` (implicit on first GET) → per-source review (single
        select → schema form → review) → finish sources → per-output review
        (single select → schema form → passthrough field review → review) →
        finish outputs (the ONLY planner call: real ``plan_guided_pipeline`` →
        scripted completion emits the guided-named candidate → real
        ``bind_guided_reviewed_components`` + candidate validation → durable
        proposal) → review wiring → confirm wiring (the sole commit).

        Only ``service._litellm_acompletion`` is scripted; the structured
        ``/guided/respond`` transitions never touch
        ``chat_solver._litellm_acompletion``. The single response is queued from
        the start so any unexpected pre-finish provider call surfaces as a
        scripted-completion-exhausted error at finish.
        """
        args = fixture["canonical_arguments"]
        source_map, output_map = _guided_naming(args)

        source_items = list(args["sources"].items()) if isinstance(args.get("sources"), dict) else [("source", args["source"])]
        outputs = args["outputs"]

        async with self._client() as client:
            created = await client.post("/api/sessions", json={"title": "parity guided-staged"})
            if created.status_code != 201:
                raise AssertionError(f"session create failed ({created.status_code}): {created.text}")
            session_id = created.json()["id"]
            candidate = _derive_guided_candidate(args, source_map, output_map, self.data_dir, session_id)
            (self.data_dir / "blobs" / session_id).mkdir(parents=True, exist_ok=True)
            (self.data_dir / "outputs" / session_id).mkdir(parents=True, exist_ok=True)

            completion = _ScriptedCompletion(emit_proposal_response(candidate))
            self.monkeypatch.setattr("elspeth.web.composer.service._litellm_acompletion", completion)

            if start_profile is not None:
                started = await client.post(
                    f"/api/sessions/{session_id}/guided/start",
                    json={"profile": start_profile, "operation_id": str(uuid4())},
                )
                if started.status_code != 200:
                    raise AssertionError(f"guided-staged {start_profile} start failed ({started.status_code}): {started.text}")
            else:
                # A ROOT INTENT keeps the finish-outputs transition on the
                # provider planner path: a rootless 1x1 step-3 entry now
                # server-synthesizes the discarded starting sketch with zero
                # planner calls, so a rootless walk could never derive a
                # transform-ful fixture graph from its sole planner call.
                started = await client.post(
                    f"/api/sessions/{session_id}/guided/start",
                    json={
                        "operation_id": str(uuid4()),
                        "intent": f"Build the {fixture['class']} pipeline from the reviewed components.",
                    },
                )
                if started.status_code != 200:
                    raise AssertionError(f"guided-staged intent start failed ({started.status_code}): {started.text}")

            # Step 1 — review every source in canonical order.
            for index, (_name, spec) in enumerate(source_items):
                if index > 0:
                    await self._staged_respond(client, session_id, component_action={"action": "add", "component_kind": "source"})
                await self._staged_respond(client, session_id, chosen=[spec["plugin"]])
                reviewed = await self._staged_respond(
                    client,
                    session_id,
                    edited_values={"plugin": spec["plugin"], "options": self._source_review_options(spec, output_map, session_id)},
                )
                if reviewed["next_turn"]["type"] == "inspect_and_confirm":
                    columns = list(
                        reviewed["next_turn"]["payload"].get("columns") or reviewed["next_turn"]["payload"].get("observed_columns") or ()
                    )
                    reviewed = await self._staged_respond(client, session_id, edited_values={"columns": columns})
                if reviewed["next_turn"]["type"] != "review_components":
                    raise AssertionError(
                        f"guided-staged source {index} landed on {reviewed['next_turn']['type']!r}, not review_components "
                        f"for {fixture['class']}"
                    )
            await self._staged_respond(client, session_id, component_action={"action": "finish", "component_kind": "source"})

            # Step 2 — review every output in canonical order. The finish-output
            # transition is the sole planner call.
            for index, output in enumerate(outputs):
                if index > 0:
                    await self._staged_respond(client, session_id, component_action={"action": "add", "component_kind": "output"})
                await self._staged_respond(client, session_id, chosen=[output["plugin"]])
                await self._staged_respond(
                    client,
                    session_id,
                    edited_values={"plugin": output["plugin"], "options": self._output_review_options(output, output_map, session_id)},
                )
                await self._staged_respond(client, session_id, control_signal="passthrough")
            staged = await self._staged_respond(client, session_id, component_action={"action": "finish", "component_kind": "output"})
            if staged["next_turn"]["type"] != "propose_pipeline":
                raise AssertionError(
                    f"guided-staged finish did not stage a proposal for {fixture['class']} (got {staged['next_turn']['type']!r})"
                )
            proposal = staged["next_turn"]["payload"]
            surface_value = staged["composition_state"]["composer_meta"]["guided_session"]["active_proposal"]
            if surface_value is None:
                raise AssertionError(f"guided-staged staged no active proposal for {fixture['class']}")

            if start_profile == "tutorial":
                # The rootless tutorial entry now stages the server-synthesized
                # starting sketch (zero planner calls); the lesson pipeline
                # arrives via the frozen-prompt REVISION — the real tutorial
                # flow. The revision below is therefore the walk's sole
                # planner call, consuming the scripted completion.
                staged = await self._staged_respond(
                    client,
                    session_id,
                    proposal_id=proposal["proposal_id"],
                    draft_hash=proposal["draft_hash"],
                    edited_values={"revision_instruction": f"Apply the tutorial lesson: build the {fixture['class']} pipeline."},
                )
                if staged["next_turn"]["type"] != "propose_pipeline":
                    raise AssertionError(
                        f"tutorial revision did not stage a proposal for {fixture['class']} (got {staged['next_turn']['type']!r})"
                    )
                proposal = staged["next_turn"]["payload"]

            # Step 3 → Step 4 — review wiring, then confirm (the sole commit).
            reviewed = await self._staged_respond(
                client,
                session_id,
                proposal_id=proposal["proposal_id"],
                draft_hash=proposal["draft_hash"],
                chosen=["review_wiring"],
            )
            wire = reviewed["next_turn"]["payload"]
            confirmed = await self._staged_respond(
                client,
                session_id,
                proposal_id=wire["proposal_id"],
                draft_hash=wire["draft_hash"],
                chosen=["confirm_wiring"],
            )
            if confirmed["terminal"]["kind"] != "completed":
                raise AssertionError(f"guided-staged did not complete for {fixture['class']} (terminal={confirmed['terminal']['kind']!r})")
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
