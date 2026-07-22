"""Live-rendered planner authoring aids: exemplars must validate, byte-for-byte.

The aids exist because the 2026-07-22 pack stress test showed cold planners
fabricating ``blob_id`` values and missing the source options contract: the
pack had rules but no worked exemplars, and the ``no_deployment_plugin_facts``
gate (correctly) forbids plugin literals in the static prompts. These tests
enforce the self-verifying-teaching contract: the exact exemplar objects the
planner prompt carries are run through ``build_set_pipeline_candidate``
against the real catalog, so a drifting exemplar fails CI instead of teaching
planners an invalid shape.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import insert
from sqlalchemy.pool import StaticPool

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.planner_authoring_aids import (
    PLACEHOLDER_BLOB_ID,
    build_planner_authoring_aids,
    discovery_digest,
    fork_coalesce_exemplar_args,
    source_custody_exemplar_args,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.composer.tools import ToolContext, build_set_pipeline_candidate
from elspeth.web.composer.tools import execute_tool as _dispatch_tool
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _trained_view() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot), snapshot


def _profile_view(tmp_path: Path) -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    """Live-deployment posture: one OpenRouter LLM operator profile.

    Mirrors ``_operator_profile_view`` in ``test_set_pipeline_candidate.py`` —
    the posture every failing planner surface (tutorial, guided, freeform web)
    actually runs under, where llm nodes are authored via a profile alias.
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.web.config import WebSettings
    from elspeth.web.plugin_policy.availability import build_plugin_snapshot
    from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
    from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry, RuntimeWebPluginConfig

    settings = WebSettings(
        data_dir=tmp_path,
        composer_model="test/planner",
        composer_max_composition_turns=3,
        composer_max_discovery_turns=2,
        composer_timeout_seconds=20.0,
        composer_rate_limit_per_minute=10,
        shareable_link_signing_key=b"\x00" * 32,
        llm_profiles={
            "sonnet": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
                "credential_scope": "server",
                "credential_ref": "OPENROUTER_API_KEY",
            }
        },
    )
    runtime = RuntimeWebPluginConfig.from_settings(settings)
    policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
    profiles = OperatorProfileRegistry(policy=policy, settings=runtime)

    class _ServerKeyInventory:
        def has_server_ref(self, name: str) -> bool:
            return name == "OPENROUTER_API_KEY"

        def has_user_ref(self, principal: str, name: str) -> bool:
            return False

        def has_ref(self, principal: str, name: str) -> bool:
            return name == "OPENROUTER_API_KEY"

        def server_generation(self, name: str) -> str | None:
            return "gen-1" if name == "OPENROUTER_API_KEY" else None

        def user_generation(self, principal: str, name: str) -> str | None:
            return None

    snapshot = build_plugin_snapshot(
        policy=policy,
        catalog=create_catalog_service(),
        profiles=profiles,
        principal_scope="local:authoring-aids-profile",
        secret_inventory=_ServerKeyInventory(),
        generation_key=b"authoring-aids-key",
    )
    return PolicyCatalogView(create_catalog_service(), snapshot, profiles), snapshot


def _session_with_user_message(content: str) -> tuple[Any, str, str, str]:
    """Session + user chat message whose text contains ``content`` verbatim."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    message_id = str(uuid4())
    message_content = f"Use this exact content:\n{content}"
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="authoring-aids-user",
                auth_provider_type="local",
                title="authoring aids exemplar validation",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=message_id,
                session_id=session_id,
                role="user",
                content=message_content,
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return engine, session_id, message_id, message_content


def _custody_context(
    tmp_path: Path,
    content: str,
    *,
    view: PolicyCatalogView | None = None,
    snapshot: PluginAvailabilitySnapshot | None = None,
) -> ToolContext:
    if view is None or snapshot is None:
        view, snapshot = _trained_view()
    engine, session_id, message_id, message_content = _session_with_user_message(content)
    return ToolContext(
        catalog=view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )


def _create_real_blob(tmp_path: Path, *, filename: str, mime_type: str, content: str) -> tuple[str, ToolContext]:
    """Create a session blob through the real tool and return (blob_id, context)."""
    view, snapshot = _trained_view()
    engine, session_id, message_id, message_content = _session_with_user_message(content)
    (tmp_path / "blobs").mkdir(exist_ok=True)
    context = ToolContext(
        catalog=view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )
    result = _dispatch_tool(
        "create_blob",
        {"filename": filename, "mime_type": mime_type, "content": content},
        _empty_state(),
        view,
        plugin_snapshot=snapshot,
        data_dir=str(tmp_path),
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=message_content,
    )
    assert result.success is True, result.to_dict()
    return result.data["blob_id"], context


class TestSourceCustodyExemplar:
    def test_inline_blob_exemplar_validates_through_the_real_candidate_builder(self, tmp_path: Path) -> None:
        """The exact inline-custody exemplar bytes the prompt carries must build."""
        view, _snapshot = _trained_view()
        args = source_custody_exemplar_args(view)
        assert args is not None
        content = args["source"]["inline_blob"]["content"]
        context = _custody_context(tmp_path, content)

        candidate = build_set_pipeline_candidate(args, _empty_state(), context)

        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"inline custody exemplar rejected: {rejection}"

    def test_existing_blob_exemplar_validates_with_a_real_session_blob(self, tmp_path: Path) -> None:
        """The blob_id binding exemplar must build once given a tool-returned id."""
        view, _snapshot = _trained_view()
        placeholder_args = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert placeholder_args is not None
        inline_args = source_custody_exemplar_args(view)
        assert inline_args is not None
        blob_id, context = _create_real_blob(
            tmp_path,
            filename=inline_args["source"]["inline_blob"]["filename"],
            mime_type=inline_args["source"]["inline_blob"]["mime_type"],
            content=inline_args["source"]["inline_blob"]["content"],
        )
        args = source_custody_exemplar_args(view, blob_id=blob_id)
        assert args is not None

        candidate = build_set_pipeline_candidate(args, _empty_state(), context)

        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"existing-blob exemplar rejected: {rejection}"
        # Single-source contract: only the binding differs between the two
        # variants the prompt shows; everything downstream is byte-identical.
        assert {key: value for key, value in args.items() if key != "source"} == {
            key: value for key, value in placeholder_args.items() if key != "source"
        }
        assert placeholder_args["source"]["blob_id"] == PLACEHOLDER_BLOB_ID

    def test_exemplar_variants_differ_only_in_the_custody_binding(self) -> None:
        view, _snapshot = _trained_view()
        inline_args = source_custody_exemplar_args(view)
        blob_args = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert inline_args is not None and blob_args is not None

        assert "inline_blob" in inline_args["source"]
        assert "blob_id" not in inline_args["source"]
        assert "blob_id" in blob_args["source"]
        assert "inline_blob" not in blob_args["source"]
        # Neither variant authors custody-owned fields by hand.
        for variant in (inline_args, blob_args):
            assert "path" not in variant["source"]["options"]
            assert "blob_ref" not in variant["source"]["options"]
            assert variant["source"]["options"]["schema"]["mode"]
            assert variant["source"]["on_validation_failure"]


def _assert_operator_ruled_topology(args: dict[str, Any]) -> None:
    """Structural invariants BOTH exemplar variants must satisfy.

    Gate condition True / routes true,false -> fork / fork_to two branches;
    one branch transform per fork branch; coalesce branches keyed by FORK
    BRANCH NAME with values naming each branch's arriving connection;
    require_all / union; no on_success on the coalesce; downstream consumes
    the coalesce id and routes to the sink.
    """
    nodes = {node["id"]: node for node in args["nodes"]}
    gates = [node for node in nodes.values() if node["node_type"] == "gate"]
    coalesces = [node for node in nodes.values() if node["node_type"] == "coalesce"]
    assert len(gates) == 1 and len(coalesces) == 1
    gate, coalesce = gates[0], coalesces[0]
    assert gate["condition"] == "True"
    assert set(gate["routes"]) == {"true", "false"}
    assert set(gate["routes"].values()) == {"fork"}
    assert len(gate["fork_to"]) == 2
    branch_nodes = [node for node in nodes.values() if node.get("input") in gate["fork_to"]]
    assert len(branch_nodes) == 2
    assert {node["input"] for node in branch_nodes} == set(gate["fork_to"])
    assert all(node["node_type"] == "transform" for node in branch_nodes)
    assert set(coalesce["branches"]) == set(gate["fork_to"])
    assert set(coalesce["branches"].values()) == {node["on_success"] for node in branch_nodes}
    assert coalesce["policy"] == "require_all"
    assert coalesce["merge"] == "union"
    assert "on_success" not in coalesce
    # Downstream cleanup consumes the coalesce's own node id.
    cleanup = next(node for node in nodes.values() if node.get("input") == coalesce["id"])
    assert cleanup["node_type"] == "transform"
    assert cleanup["on_success"] == args["outputs"][0]["sink_name"]


class TestForkCoalesceExemplar:
    def test_fork_coalesce_exemplar_validates_under_the_live_profile_posture(self, tmp_path: Path) -> None:
        """The exact fork -> two-llm -> coalesce exemplar bytes must build."""
        (tmp_path / "outputs").mkdir(exist_ok=True)
        view, snapshot = _profile_view(tmp_path)
        args = fork_coalesce_exemplar_args(view)
        assert args is not None
        content = args["source"]["inline_blob"]["content"]
        context = _custody_context(tmp_path, content, view=view, snapshot=snapshot)

        candidate = build_set_pipeline_candidate(args, _empty_state(), context)

        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"fork/coalesce exemplar rejected: {rejection}"

    def test_fork_coalesce_exemplar_has_the_operator_ruled_topology(self, tmp_path: Path) -> None:
        """Two separate LLM transform nodes + coalesce merge — never a queries map."""
        view, _snapshot = _profile_view(tmp_path)
        args = fork_coalesce_exemplar_args(view)
        assert args is not None

        _assert_operator_ruled_topology(args)
        nodes = {node["id"]: node for node in args["nodes"]}
        llms = [node for node in nodes.values() if node.get("plugin") == "llm"]
        gate = next(node for node in nodes.values() if node["node_type"] == "gate")
        assert len(llms) == 2
        # One llm per fork branch, each with its own prompt and output field.
        assert {llm["input"] for llm in llms} == set(gate["fork_to"])
        assert len({llm["options"]["prompt_template"] for llm in llms}) == 2
        assert len({llm["options"]["response_field"] for llm in llms}) == 2
        assert all("queries" not in llm["options"] for llm in llms)

    def test_topology_exemplar_renders_and_validates_without_a_usable_llm_profile(self, tmp_path: Path) -> None:
        """No usable llm profile -> the SAME topology with non-LLM branches.

        Fork/gate/coalesce WIRING is pure topology, independent of LLMs — only
        the branch contents need a profile. Gating the whole section on an llm
        profile alias (as the module originally did) meant a deployment with no
        visible LLM profile lost the topology teaching entirely, which is
        exactly backwards. Branch transforms are drawn deterministically from
        the policy-visible catalog: first two alphabetically whose only
        required option is ``schema``.
        """
        view, _snapshot = _trained_view()
        args = fork_coalesce_exemplar_args(view)
        assert args is not None

        _assert_operator_ruled_topology(args)
        nodes = args["nodes"]
        assert all(node.get("plugin") != "llm" for node in nodes)
        rendered = json.dumps(args)
        # Nothing llm-flavored may leak into the profile-less variant: no
        # profile alias, no interpretation review rows (llm-node concern).
        assert '"profile"' not in rendered
        assert "interpretation_requirements" not in rendered
        # Deterministic branch pick, recomputed here from the same posture:
        # first two alphabetically that are non-LLM, non-batch-aware, and
        # authorable with only the universal schema option.
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

        batch_aware = {cls.name for cls in get_shared_plugin_manager().get_transforms() if cls.is_batch_aware}
        renderable = sorted(
            plugin.name
            for plugin in view.list_transforms()
            if plugin.name != "llm"
            and plugin.name not in batch_aware
            and {field.name for field in plugin.config_fields if field.required} <= {"schema"}
        )
        gate = next(node for node in nodes if node["node_type"] == "gate")
        branch_plugins = sorted(node["plugin"] for node in nodes if node.get("input") in gate["fork_to"])
        assert branch_plugins == renderable[:2]
        # The exact bytes the prompt carries must build under this posture.
        (tmp_path / "outputs").mkdir(exist_ok=True)
        content = args["source"]["inline_blob"]["content"]
        context = _custody_context(tmp_path, content)
        candidate = build_set_pipeline_candidate(args, _empty_state(), context)
        rejection = None if candidate.acceptable else (candidate.result.data or {}).get("error")
        assert candidate.acceptable is True, f"topology exemplar rejected: {rejection}"

    def test_trained_posture_payload_carries_the_topology_exemplar(self) -> None:
        """The aids payload renders fork_coalesce even with no llm profile."""
        view, _snapshot = _trained_view()

        payload = build_planner_authoring_aids(view)

        section = payload["fork_coalesce"]
        assert section["set_pipeline_exemplar"] == fork_coalesce_exemplar_args(view)


class TestExemplarDomainDisjointness:
    """Exemplars teach STRUCTURE, never a live acceptance test's solution.

    The fork/coalesce exemplar previously mirrored the live colour-A/B
    acceptance test (color_name/hex source fields, tone/usage branches,
    emotional-tone prompts) — effectively the test's answer key. When a live
    test's domain vocabulary appears in an exemplar, the test measures
    pack-lookup, not planner capability, and its pass signal is contaminated.
    These tests pin the neutral domains and the structural elements that must
    survive any future domain change.
    """

    # The live colour A/B acceptance vocabulary. json.dumps + lowercase scan.
    _COLOUR_ACCEPTANCE_TOKENS = ("color_name", "hex", "cerulean", "saffron", "colour", "color")

    def test_fork_coalesce_exemplar_is_disjoint_from_the_colour_acceptance_domain(self, tmp_path: Path) -> None:
        trained_view, _ = _trained_view()
        profile_view, _ = _profile_view(tmp_path)
        for view in (trained_view, profile_view):
            args = fork_coalesce_exemplar_args(view)
            assert args is not None
            rendered = json.dumps(args).lower()
            for token in self._COLOUR_ACCEPTANCE_TOKENS:
                assert token not in rendered, f"acceptance-domain token {token!r} in fork/coalesce exemplar"
            # The neutral domain is present and carried end to end: source
            # schema fields reappear in the cleanup mapping.
            assert "ticket_id" in rendered and "body" in rendered

    def test_source_custody_exemplar_is_domain_neutral(self) -> None:
        view, _snapshot = _trained_view()
        args = source_custody_exemplar_args(view)
        assert args is not None
        rendered = json.dumps(args).lower()
        for token in self._COLOUR_ACCEPTANCE_TOKENS:
            assert token not in rendered, f"acceptance-domain token {token!r} in custody exemplar"
        # The old quarterly-totals domain is retired with the rewrite.
        assert "quarterly" not in rendered
        assert "sku" in rendered

    def test_llm_branch_prompts_interpolate_the_row_namespace_over_guaranteed_fields(self, tmp_path: Path) -> None:
        """Prompts use {{ row.<field> }} and only name source-guaranteed fields.

        Candidate-level CI does not validate prompt interpolation (the run-19
        readiness rejection came from planners copying a {{ field }} prompt),
        so the exemplar's prompts are pinned here: every placeholder uses the
        row namespace and resolves to a field the source guarantees.
        """
        import re

        view, _snapshot = _profile_view(tmp_path)
        args = fork_coalesce_exemplar_args(view)
        assert args is not None
        guaranteed = set(args["source"]["options"]["schema"]["guaranteed_fields"])
        llms = [node for node in args["nodes"] if node.get("plugin") == "llm"]
        assert len(llms) == 2
        for llm in llms:
            prompt = llm["options"]["prompt_template"]
            placeholders = re.findall(r"\{\{\s*([^}]+?)\s*\}\}", prompt)
            assert placeholders, "each branch prompt must interpolate the row"
            for placeholder in placeholders:
                assert placeholder.startswith("row."), f"placeholder {{{{ {placeholder} }}}} must use the row namespace"
                assert placeholder.removeprefix("row.") in guaranteed
            # Ownership doctrine (G3): backend-owned reviews are never
            # hand-authored; this exemplar stages no planner-owned kinds, so
            # the options carry no interpretation_requirements at all.
            assert "interpretation_requirements" not in llm["options"]

    def test_payload_states_the_per_branch_shape_rule(self, tmp_path: Path) -> None:
        view, _snapshot = _profile_view(tmp_path)

        payload = build_planner_authoring_aids(view)

        section = payload["fork_coalesce"]
        assert section["set_pipeline_exemplar"] == fork_coalesce_exemplar_args(view)
        rules = " ".join(section["rules"])
        assert "separate LLM" in rules
        assert "not a queries map" in rules


class TestDiscoveryDigest:
    def test_digest_covers_every_policy_visible_plugin(self) -> None:
        """DISCOVERY_CYCLE churn re-derives the catalog; the digest IS the catalog."""
        view, _snapshot = _trained_view()

        digest = discovery_digest(view)

        assert {entry["name"] for entry in digest["sources"]} == {plugin.name for plugin in view.list_sources()}
        assert {entry["name"] for entry in digest["transforms"]} == {plugin.name for plugin in view.list_transforms()}
        assert {entry["name"] for entry in digest["sinks"]} == {plugin.name for plugin in view.list_sinks()}

    def test_digest_entries_carry_purpose_required_knobs_and_hints(self) -> None:
        view, _snapshot = _trained_view()

        digest = discovery_digest(view)

        csv_source = next(entry for entry in digest["sources"] if entry["name"] == "csv")
        assert csv_source["purpose"] == next(plugin.description for plugin in view.list_sources() if plugin.name == "csv")
        assert {"schema", "path", "on_validation_failure"} <= set(csv_source["required_options"])
        # composer_hints are the designated live channel for web-policy facts
        # (e.g. the json sink's explicit collision_policy contract) — the
        # digest must surface them verbatim, not summarize them away.
        json_sink = next(entry for entry in digest["sinks"] if entry["name"] == "json")
        assert json_sink["composer_hints"] == list(next(plugin.composer_hints for plugin in view.list_sinks() if plugin.name == "json"))
        assert any("collision_policy" in hint for hint in json_sink["composer_hints"])

    def test_capability_core_discovery_order_speaks_with_the_digest_voice(self) -> None:
        """The static core and the live digest guidance must agree, zero daylight.

        The campaign's central finding: in-context-every-turn contradictions
        dominate planner behavior. The core's discovery-order steps and the
        digest's short-circuit guidance ride in the same context every turn,
        so where they overlap they must share exact sentences — a core that
        says "read the inventories with list_*" every turn undercuts the
        digest on the DISCOVERY_CYCLE failure mode it exists to fix.
        """
        from elspeth.web.composer.capability_skill import load_pipeline_capability_core
        from elspeth.web.composer.planner_authoring_aids import _DISCOVERY_DIGEST_GUIDANCE

        # Word-for-word agreement is about words: collapse the markdown's
        # hard line wrapping before comparing.
        core = load_pipeline_capability_core()
        core_flat = " ".join(core.split())

        # The old unconditional re-discovery instruction must be gone.
        assert "Read the policy-visible inventories" not in core_flat
        assert "Read every selected plugin's authoritative options" not in core_flat
        # Digest-first scoping, shared word-for-word between both texts.
        shared_phrases = (
            "rendered from the live policy-visible catalog at prompt build and is current for this deployment",
            "plan directly from it",
            "for structured repair when a proposal is rejected",
        )
        for phrase in shared_phrases:
            assert phrase in core_flat, f"core lacks shared phrase: {phrase!r}"
            assert phrase in _DISCOVERY_DIGEST_GUIDANCE, f"digest guidance lacks shared phrase: {phrase!r}"
        # The narrowing must not disturb the closed provenance rules the core
        # keeps: model ids only from list_models, secret discovery intact.
        assert "Model identifiers come only from `list_models`" in core
        assert "list_secret_refs" in core

    def test_payload_carries_digest_with_discovery_short_circuit_guidance(self) -> None:
        view, _snapshot = _trained_view()

        payload = build_planner_authoring_aids(view)

        section = payload["discovery_digest"]
        assert section["plugins"] == discovery_digest(view)
        guidance = section["guidance"]
        assert "rarely need" in guidance
        assert "get_plugin_schema" in guidance
        assert "current" in guidance
        # The digest short-circuits catalog re-discovery only: model ids still
        # come solely from list_models, and structured-repair tooling
        # (get_plugin_assistance) keeps its role.
        assert "list_models" in guidance
        assert "get_plugin_assistance" in guidance


class TestAuthoringAidsPayload:
    def test_payload_carries_the_custody_exemplar_and_the_closed_provenance_rule(self) -> None:
        view, _snapshot = _trained_view()

        payload = build_planner_authoring_aids(view)

        custody = payload["source_custody"]
        assert custody["set_pipeline_exemplar_inline_blob"] == source_custody_exemplar_args(view)
        blob_variant = source_custody_exemplar_args(view, blob_id=PLACEHOLDER_BLOB_ID)
        assert blob_variant is not None
        assert custody["existing_blob_source_binding"] == blob_variant["source"]
        rules = " ".join(custody["rules"])
        assert "ONLY from blob-tool output" in rules
        assert "inline_blob" in rules
        assert "Never fabricate" in rules

    def test_payload_never_carries_a_fabricated_blob_identifier(self) -> None:
        """The prompt teaches provenance; it must not model a fake UUID itself."""
        view, _snapshot = _trained_view()

        rendered = json.dumps(build_planner_authoring_aids(view))

        assert PLACEHOLDER_BLOB_ID in rendered
        assert "-4000-" not in rendered  # no synthetic UUID4-shaped identifiers
        assert rendered.count("blob_id") >= 1

    def test_payload_is_canonical_json_compatible(self) -> None:
        from elspeth.core.canonical import canonical_json

        view, _snapshot = _trained_view()

        canonical_json(build_planner_authoring_aids(view))


class TestPromptShieldRules:
    """The aids teach shield-staging with the registered constants verbatim.

    Tutorial finalizer battery (dim_c under-flag): the replan planner
    non-deterministically omitted the ADVISORY prompt_injection_shield
    review row on the scrape→summarize llm node. The omission carries no
    rejection code (the shield review is advisory — warnings only, excluded
    from the blocking contract), so the authoring aids are the ONLY teaching
    lever. Following the 52322ebe1 discipline, the rule must quote the
    registered user_term and the deployment-honest draft constant verbatim —
    imported constants, so the aids can never drift from the contract.
    """

    def test_trained_view_quotes_term_and_available_draft_verbatim(self) -> None:
        from elspeth.web.interpretation_state import (
            PROMPT_SHIELD_AVAILABLE_DRAFT,
            PROMPT_SHIELD_USER_TERM,
            PROMPT_SHIELD_WARNING_DRAFT,
        )

        view, snapshot = _trained_view()
        # The trained snapshot SELECTS a prompt shield — the honest draft is
        # the shield-available wording (mirrors the warning→available upgrade
        # the server itself applies when the shield is selected).
        from elspeth.contracts.plugin_capabilities import PluginCapability

        assert dict(snapshot.selected).get(PluginCapability.PROMPT_SHIELD) is not None

        aids = build_planner_authoring_aids(view)
        rendered = "\n".join(aids["prompt_shield"]["rules"])
        assert PROMPT_SHIELD_USER_TERM in rendered
        assert PROMPT_SHIELD_AVAILABLE_DRAFT in rendered
        assert PROMPT_SHIELD_WARNING_DRAFT not in rendered

    def test_shieldless_view_quotes_the_warning_draft_verbatim(self) -> None:
        from unittest.mock import MagicMock

        from elspeth.contracts.plugin_capabilities import PluginCapability
        from elspeth.web.interpretation_state import (
            PROMPT_SHIELD_AVAILABLE_DRAFT,
            PROMPT_SHIELD_USER_TERM,
            PROMPT_SHIELD_WARNING_DRAFT,
        )
        from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry

        catalog = create_catalog_service()
        trained = PluginAvailabilitySnapshot.for_trained_operator(catalog)
        shieldless = PluginAvailabilitySnapshot.create(
            policy_hash="prompt-shield-rules-shieldless",
            principal_scope="local:prompt-shield-rules",
            available=trained.available,
            unavailable=(),
            selected=tuple(
                (capability, plugin)
                for capability, plugin in dict(trained.selected).items()
                if capability is not PluginCapability.PROMPT_SHIELD
            ),
            usable_profile_aliases=(),
            selected_profile_aliases=(),
            binding_generation_fingerprint="prompt-shield-rules-generation",
        )
        view = PolicyCatalogView(catalog, shieldless, MagicMock(spec=OperatorProfileRegistry))

        aids = build_planner_authoring_aids(view)
        rules = aids["prompt_shield"]["rules"]
        rendered = "\n".join(rules)
        assert PROMPT_SHIELD_USER_TERM in rendered
        assert PROMPT_SHIELD_WARNING_DRAFT in rendered
        assert PROMPT_SHIELD_AVAILABLE_DRAFT not in rendered

    def test_rules_name_the_untrusted_producer_and_the_llm_attachment_point(self) -> None:
        view, _snapshot = _trained_view()

        rendered = "\n".join(build_planner_authoring_aids(view)["prompt_shield"]["rules"])
        assert "web_scrape" in rendered
        assert "interpretation_requirements" in rendered
        assert "llm" in rendered

    def test_section_renders_under_the_live_profile_posture(self, tmp_path: Path) -> None:
        # The failing surface is the tutorial/guided walk under the operator-
        # profile posture — pin that the section actually reaches it (web_scrape
        # and llm are policy-visible there), not just the trained fixture.
        from elspeth.web.interpretation_state import PROMPT_SHIELD_USER_TERM

        view, _snapshot = _profile_view(tmp_path)

        rendered = "\n".join(build_planner_authoring_aids(view)["prompt_shield"]["rules"])
        assert PROMPT_SHIELD_USER_TERM in rendered


class TestModelCustody:
    """Suite run 1 G2 (8/8): never-invent had no sanctioned alternative."""

    def test_model_custody_renders_the_live_profile_alias(self, tmp_path: Path) -> None:
        from elspeth.web.composer.planner_authoring_aids import _usable_llm_profile_alias

        view, _snapshot = _profile_view(tmp_path)
        alias = _usable_llm_profile_alias(view)
        assert alias is not None

        rules = " ".join(build_planner_authoring_aids(view)["model_custody"]["rules"])
        assert f"'{alias}'" in rules
        assert "options.profile" in rules
        assert "llm_model_choice" in rules
        assert "list_models" in rules

    def test_model_custody_without_a_profile_teaches_discovery_only(self) -> None:
        # The trained fixture serves no llm operator profile: the section must
        # still render and steer to list_models rather than invented slugs.
        view, _snapshot = _trained_view()

        rules = " ".join(build_planner_authoring_aids(view)["model_custody"]["rules"])
        assert "No llm operator profile" in rules
        assert "list_models" in rules

    def test_llm_output_contract_blesses_one_string_and_multi_query_only(self) -> None:
        view, _snapshot = _trained_view()

        rules = " ".join(build_planner_authoring_aids(view)["llm_output_contract"]["rules"])
        assert "response_field" in rules
        assert "llm_response" in rules
        assert "multi_query" in rules
        assert "does NOT create row fields" in rules


class TestReviewRegistry:
    """Suite run 1 G5: the closed pipeline_decision registry never surfaced at authoring time."""

    def test_registry_quotes_the_registered_terms_verbatim(self) -> None:
        from elspeth.web.interpretation_state import REGISTERED_PIPELINE_DECISION_USER_TERMS

        view, _snapshot = _trained_view()

        section = build_planner_authoring_aids(view)["review_registry"]
        assert section["registered_pipeline_decision_user_terms"] == sorted(REGISTERED_PIPELINE_DECISION_USER_TERMS)

    def test_registry_rules_carry_ownership_economy_override_and_off_vocab_route(self) -> None:
        view, _snapshot = _trained_view()

        rules = " ".join(build_planner_authoring_aids(view)["review_registry"]["rules"])
        # Closed-vocabulary + off-vocab route (G5).
        assert "metadata.description" in rules
        # Economy-override clause: an explicit user instruction does not waive
        # a registered demanded review — the row RECORDS the decision.
        assert "NEVER waived" in rules
        assert "RECORDS" in rules
        # Ownership: backend-owned kinds are never hand-authored (G3).
        assert "auto-stage" in rules
        assert "llm_prompt_template" in rules
        assert "llm_model_choice" in rules


class TestOwnershipAlignedExemplars:
    """G3: exemplars must not hand-author backend-auto-staged review rows."""

    def test_exemplar_authors_no_backend_owned_review_rows(self, tmp_path: Path) -> None:
        view, _snapshot = _profile_view(tmp_path)
        args = fork_coalesce_exemplar_args(view)
        assert args is not None

        for node in args["nodes"]:
            requirements = node.get("options", {}).get("interpretation_requirements", [])
            assert all(row["kind"] not in {"llm_prompt_template", "llm_model_choice"} for row in requirements), (
                f"node {node['id']} hand-authors a backend-owned review row"
            )


class TestCoalesceVocabulary:
    """Suite run 1 G7: policy/merge vocab + the inert-input convention."""

    def test_rules_state_the_closed_policy_and_merge_vocabularies(self, tmp_path: Path) -> None:
        view, _snapshot = _profile_view(tmp_path)

        rules = " ".join(build_planner_authoring_aids(view)["fork_coalesce"]["rules"])
        assert "require_all, quorum, best_effort, first" in rules
        assert "union, nested, select" in rules
        assert "best_effort" in rules

    def test_exemplar_sets_coalesce_input_to_a_branch_connection(self, tmp_path: Path) -> None:
        view, _snapshot = _profile_view(tmp_path)
        args = fork_coalesce_exemplar_args(view)
        assert args is not None

        coalesce = next(node for node in args["nodes"] if node["node_type"] == "coalesce")
        assert coalesce["input"] in set(coalesce["branches"].values())


class TestNamedButMissingFile:
    """Suite run 1 singleton: honesty must not lose to fabrication."""

    def test_custody_rules_teach_the_discover_ask_decline_sequence(self) -> None:
        view, _snapshot = _trained_view()

        rules = " ".join(build_planner_authoring_aids(view)["source_custody"]["rules"])
        assert "never uploaded" in rules
        assert "list_blobs" in rules
        assert "named gap" in rules


class TestRun2PackEdits:
    """Pack pressure-suite run-2 gap closures (G2/G3/G4/G6/G9), pinned.

    Same discipline as the shield rules: product constants imported so the
    taught text can never drift; teaching co-located with the mandate it
    completes (the aids bless multi_query as the only multi-field shape, so
    the aids must also carry that shape's contract).
    """

    def test_llm_output_rules_teach_the_query_definition_contract(self) -> None:
        view, _snapshot = _trained_view()
        rendered = "\n".join(build_planner_authoring_aids(view)["llm_output_contract"]["rules"])
        assert "input_fields" in rendered  # G2: REQUIRED per-query mapping
        assert "'template'" in rendered or '"template"' in rendered  # G2: per-query key is template
        assert "prompt_template" in rendered  # G2: top-level still required
        assert "max_capacity_retry_seconds" in rendered  # G9: web retry cap
        from elspeth.web.provider_config_policy import WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS

        assert str(WEB_LLM_SEQUENTIAL_MULTI_QUERY_MAX_RETRY_SECONDS) in rendered

    def test_custody_rules_name_the_discovered_path_binding_slot(self) -> None:
        view, _snapshot = _trained_view()
        rendered = "\n".join(build_planner_authoring_aids(view)["source_custody"]["rules"])
        assert "options.path" in rendered  # G4: where a discovered blobs/ path goes
        assert "blob_id" in rendered

    def test_raw_html_cleanup_rules_quote_the_constants_verbatim(self) -> None:
        # G6: the cleanup draft ships verbatim next to the shield draft — the
        # aids are the verbatim-constants channel (imported, drift-proof).
        from elspeth.web.interpretation_state import (
            RAW_HTML_CLEANUP_REVIEW_DRAFT,
            RAW_HTML_CLEANUP_USER_TERM,
        )

        view, _snapshot = _trained_view()
        rendered = "\n".join(build_planner_authoring_aids(view)["raw_html_cleanup"]["rules"])
        assert RAW_HTML_CLEANUP_USER_TERM in rendered
        assert RAW_HTML_CLEANUP_REVIEW_DRAFT in rendered
        assert "field_mapper" in rendered  # attachment point

    def test_llm_digest_hint_scopes_the_shield_advisory_to_fetched_content(self) -> None:
        # G3 DECIDED: the advisory covers llms consuming untrusted REMOTE
        # (fetched) producer output — not "every unshielded llm". The old
        # hint contradicted the skill and the aids' own prompt_shield rule.
        view, _snapshot = _trained_view()
        aids = build_planner_authoring_aids(view)
        llm_entry = next(e for e in aids["discovery_digest"]["plugins"]["transforms"] if e["name"] == "llm")
        hints = "\n".join(llm_entry["composer_hints"])
        assert "EVERY LLM node" not in hints
        assert "externally-fetched" in hints or "externally fetched" in hints

    def test_skill_authoring_exemplar_carries_no_server_review_fields(self) -> None:
        # G5: an exemplar demonstrated the heavy server-side review row
        # (event_id/accepted_value/...) where the authored contract is the
        # short form (canonicalize_authored_node_review_requirements) —
        # three run-2 sims tripled the heavy row identically.
        from elspeth.web.composer.skills import load_skill

        skill = load_skill("pipeline_composer")
        assert '"event_id": null' not in skill
        assert '"accepted_artifact_hash": null' not in skill
