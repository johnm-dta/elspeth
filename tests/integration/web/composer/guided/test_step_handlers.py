"""Integration tests for guided-mode step commit handlers.

These tests use real CompositionState and the real tools.py mutation
helpers; only the catalog is constructed via the public test seam
(create_catalog_service()).
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.guided.steps import (
    StepHandlerResult,
    _observed_columns_from_blob,
    _sink_options_with_step_2_schema_contract,
    handle_step_1_source,
    handle_step_2_sink,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot, PluginId
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry


def _empty_state() -> CompositionState:
    """Construct an empty CompositionState per errata C3 (6-arg required ctor)."""
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def _restricted_policy(hidden: PluginId) -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    catalog = create_catalog_service()
    unrestricted = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="guided-policy",
        principal_scope="local:alice",
        available=unrestricted.available - {hidden},
        unavailable=(),
        selected=unrestricted.selected,
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="guided-policy-generation",
    )
    return PolicyCatalogView(catalog, snapshot, MagicMock(spec=OperatorProfileRegistry)), snapshot


def _trained_policy() -> tuple[PolicyCatalogView, PluginAvailabilitySnapshot]:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return PolicyCatalogView.for_trained_operator(catalog, snapshot), snapshot


class TestStep1Handler:
    def test_direct_submission_cannot_commit_disabled_source(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _restricted_policy(PluginId("source", "csv"))

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "data.csv", "schema": {"mode": "observed"}},
                observed_columns=("a",),
                sample_rows=({"a": "1"},),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
        )

        assert result.tool_result.success is False
        assert result.tool_result.data["error_code"] == "plugin_not_enabled"
        assert result.state is state
        assert result.session is session

    def test_commits_source_to_state_on_success(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "data.csv", "schema": {"mode": "observed"}},
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
        )

        assert isinstance(result, StepHandlerResult)
        assert result.tool_result.success is True
        assert result.state.sources.get("source") is not None
        assert result.state.sources["source"].plugin == "csv"
        assert result.session.step_1_result is not None
        assert result.session.step_1_result.plugin == "csv"
        # Session step pointer is NOT advanced here — the dispatcher does that.
        assert result.session.step == session.step

    def test_commits_resolved_on_validation_failure(self) -> None:
        """The handler commits the SOURCE node's routing from ``resolved`` (the
        composer's choice), not a hardcoded 'discard'. A non-default sentinel
        proves it: this assertion fails on the old hardcode and passes on the
        threaded value. An unknown-sink reference is a non-blocking advisory note,
        so the commit still succeeds."""
        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "data.csv", "schema": {"mode": "observed"}},
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
                on_validation_failure="quarantine_sink",
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
        )

        assert result.tool_result.success is True
        assert result.state.sources["source"].on_validation_failure == "quarantine_sink"
        assert result.session.step_1_result is not None
        assert result.session.step_1_result.on_validation_failure == "quarantine_sink"

    def test_returns_failure_unchanged_session_when_plugin_unknown(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="DEFINITELY_NOT_A_REAL_PLUGIN_xyzzy",
                options={},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
        )

        assert result.tool_result.success is False
        assert result.state is state  # unchanged on failure
        assert result.session.step_1_result is None

    def test_path_source_backfills_observed_columns_into_committed_schema(self, tmp_path) -> None:
        """Manual guided JSON paths under data_dir/blobs publish observed fields.

        The web form can submit a path-only source with empty observed_columns.
        The commit seam must derive bounded field facts from the allowed source
        path and put them into CompositionState, not only into step_1_result,
        because the wire validator reads the committed source schema.
        """
        data_dir = tmp_path
        blobs_dir = data_dir / "blobs"
        blobs_dir.mkdir()
        (blobs_dir / "lines.json").write_text(
            json.dumps([{"line": "alpha"}, {"line": "beta"}]),
            encoding="utf-8",
        )
        catalog, snapshot = _trained_policy()

        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": "blobs/lines.json", "schema": {"mode": "observed"}},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            data_dir=str(data_dir),
        )

        assert result.tool_result.success is True
        assert result.session.step_1_result is not None
        assert result.session.step_1_result.observed_columns == ("line",)
        source = result.state.sources["source"]
        assert tuple(dict(source.options["schema"])["guaranteed_fields"]) == ("line",)

    def test_outside_allowlist_path_is_not_inspected_before_rejection(self, tmp_path, monkeypatch) -> None:
        """Path field inference must not read a local file before S2 validation."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        outside_path = tmp_path / "outside.json"
        outside_path.write_text(json.dumps([{"secret": "do-not-read"}]), encoding="utf-8")

        def _fail_if_called(*_args, **_kwargs) -> tuple[str, ...]:
            raise AssertionError("outside allowlist path was inspected")

        monkeypatch.setattr("elspeth.web.composer.guided.steps.observed_columns_from_path", _fail_if_called)
        catalog, snapshot = _trained_policy()

        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": str(outside_path), "schema": {"mode": "observed"}},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            data_dir=str(data_dir),
        )

        assert result.tool_result.success is False
        assert result.session.step_1_result is None


class TestStep2Handler:
    def test_commits_outputs_to_state_on_success(self) -> None:
        # Step 1 sets a CSV source with on_success="main".
        # Then Step 2 attaches a json output named "main".
        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["a"]}},
                observed_columns=("a",),
                sample_rows=({"a": "1"},),
            ),
        )
        assert step_1.tool_result.success is True

        result = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                        required_fields=("a",),
                        schema_mode="observed",
                    ),
                ),
            ),
        )

        assert result.tool_result.success is True
        assert len(result.state.outputs) == 1
        assert result.state.outputs[0].plugin == "json"
        assert result.state.outputs[0].name == "main"
        assert result.session.step_2_result is not None

    def test_returns_failure_unchanged_when_plugin_unknown(self) -> None:
        state = _empty_state()
        catalog, snapshot = _trained_policy()

        # _execute_set_output validates plugin-name first, before any source
        # presence check — so this exercises the failure path on empty state.
        result = handle_step_2_sink(
            state=state,
            session=GuidedSession.initial(),
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="DEFINITELY_NOT_A_REAL_PLUGIN_xyzzy",
                        options={},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )

        assert result.tool_result.success is False
        assert result.state is state  # identity: unchanged on failure
        assert result.session.step_2_result is None

    def test_refuses_empty_outputs(self) -> None:
        catalog, snapshot = _trained_policy()

        with pytest.raises(InvariantError, match="empty list"):
            handle_step_2_sink(
                state=_empty_state(),
                session=GuidedSession.initial(),
                resolved=SinkResolved(outputs=()),
                catalog=catalog,
                plugin_snapshot=snapshot,
            )

    def test_sink_options_rejects_malformed_present_schema(self) -> None:
        output = SinkOutputResolved(
            plugin="json",
            options={"path": "out.jsonl", "schema_config": "observed"},
            required_fields=("a",),
            schema_mode="observed",
        )

        with pytest.raises(InvariantError, match="schema options"):
            _sink_options_with_step_2_schema_contract(output)


class TestStep3Handler:
    """Tests for handle_step_3_chain_accept.

    The success test composes Step 1 (CSV source) + Step 2 (JSON sink) + Step 3
    (transform chain) end-to-end to exercise the real _execute_set_pipeline
    path. Stubs only at the LLM boundary (proposal is constructed in-test).

    Uses `passthrough` as the transform plugin — it is the simplest transform
    in the catalogue (only `schema` is required) and isolates the wiring
    contract from plugin-config bookkeeping. The brief's `type_coerce` example
    had the wrong options shape (`fields` instead of `conversions`) so I
    switched to `passthrough` to avoid coupling the test to that drift.
    """

    @pytest.mark.parametrize(
        ("tutorial", "proposal_profile", "expected_profile"),
        [
            (True, None, "tutorial-default"),
            (True, "alpha", "tutorial-default"),
            (False, "alpha", "alpha"),
        ],
    )
    def test_guided_chain_applies_operator_llm_profile_selection(
        self,
        tutorial: bool,
        proposal_profile: str | None,
        expected_profile: str,
    ) -> None:
        """The guided solver does not own provider bindings.

        A tutorial proposal may correctly ask for the public ``llm`` transform
        without knowing the operator's opaque profile alias.  The commit seam
        must bind that node to the tutorial profile selected in the request's
        policy snapshot before running the canonical set-pipeline validation.
        Live guided sessions may still choose another operator-approved alias.
        """
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept
        from elspeth.web.config import WebSettings
        from elspeth.web.plugin_policy.availability import build_plugin_snapshot
        from elspeth.web.plugin_policy.compiler import compile_web_plugin_policy
        from elspeth.web.plugin_policy.profiles import RuntimeWebPluginConfig

        settings = WebSettings(
            composer_max_composition_turns=4,
            composer_max_discovery_turns=4,
            composer_timeout_seconds=60,
            composer_rate_limit_per_minute=20,
            shareable_link_signing_key=b"0123456789abcdef0123456789abcdef",
            llm_profiles={
                "alpha": {
                    "provider": "bedrock",
                    "model": "bedrock/apac.amazon.nova-micro-v1:0",
                    "region_name": "ap-southeast-1",
                },
                "tutorial-default": {
                    "provider": "bedrock",
                    "model": "bedrock/apac.amazon.nova-lite-v1:0",
                    "region_name": "ap-southeast-1",
                },
            },
            tutorial_llm_profile="tutorial-default",
        )
        runtime = RuntimeWebPluginConfig.from_settings(settings)
        policy = compile_web_plugin_policy(registry=get_shared_plugin_manager(), settings=runtime)
        profiles = OperatorProfileRegistry(policy=policy, settings=runtime)
        full_catalog = create_catalog_service()

        class _NoSecrets:
            def has_server_ref(self, name: str) -> bool:
                return False

            def has_user_ref(self, principal: str, name: str) -> bool:
                return False

        snapshot = build_plugin_snapshot(
            policy=policy,
            catalog=full_catalog,
            profiles=profiles,
            principal_scope="local:tutorial-user",
            secret_inventory=_NoSecrets(),
            generation_key=b"guided-tutorial-profile-test-key",
        )
        catalog = PolicyCatalogView(full_catalog, snapshot, profiles)
        session = GuidedSession.initial(profile=TUTORIAL_PROFILE if tutorial else EMPTY_PROFILE)

        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["text"]}},
                observed_columns=("text",),
                sample_rows=({"text": "hello"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )
        llm_options: dict[str, object] = {
            "prompt_template": "Summarise {{ row.text }}",
            "response_field": "summary",
            "required_input_fields": ["text"],
            "schema": {"mode": "observed"},
        }
        if proposal_profile is not None:
            llm_options["profile"] = proposal_profile
        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "llm",
                    "options": llm_options,
                    "rationale": "summarise each row",
                },
            ),
            why="summarise the tutorial input",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            proposal=proposal,
            catalog=catalog,
            plugin_snapshot=snapshot,
        )

        assert result.tool_result.success is True, result.tool_result.validation.errors
        assert result.state.nodes[0].options["profile"] == expected_profile

    def test_chain_accepted_commits_and_redirects_to_wire(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import (
            ChainProposal,
        )
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
        )
        assert step_1.tool_result.success is True

        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                        required_fields=("price",),
                        schema_mode="observed",
                    ),
                ),
            ),
        )
        assert step_2.tool_result.success is True

        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "passthrough",
                    "options": {"schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                    "rationale": "echo rows; minimal transform for wiring proof",
                },
            ),
            why="single-step chain verifying chain_in→main wiring",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=proposal,
        )

        assert result.tool_result.success is True, f"set_pipeline failed: {getattr(result.tool_result, 'data', result.tool_result)}"
        assert len(result.state.nodes) == 1
        assert result.state.nodes[0].plugin == "passthrough"
        assert result.state.nodes[0].input == "chain_in"
        assert result.state.nodes[0].on_success == "main"
        assert result.state.sources.get("source") is not None
        assert result.state.sources["source"].on_success == "chain_in"  # rewired
        assert result.session.terminal is None
        assert result.session.step is GuidedStep.STEP_4_WIRE
        assert result.session.step_3_proposal is proposal

    def test_chain_accept_does_not_inject_web_scrape_allowed_hosts(self) -> None:
        """The tutorial's synthetic pages are publicly hosted, so commit injects
        NO SSRF allowlist into the web_scrape node: its ``http`` block is exactly
        what the LLM set (abuse_contact/scraping_reason), and ``allowed_hosts`` is
        absent so the plugin default ``public_only`` applies — full parity with a
        normal backend run (the loopback-CIDR seam was removed once the synthetic
        pages moved to public hosting).
        """
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        catalog, snapshot = _trained_policy()
        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="json",
                options={"path": "urls.json", "schema": {"mode": "observed", "guaranteed_fields": ["url"]}},
                observed_columns=("url",),
                sample_rows=({"url": "https://johnm-dta.github.io/elspeth/tutorial-site/project-1.html"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.json", "schema": {"mode": "observed"}},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )
        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "web_scrape",
                    "options": {
                        "schema": {"mode": "observed", "guaranteed_fields": ["page_content", "page_fp"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "page_content",
                        "fingerprint_field": "page_fp",
                        "format": "markdown",
                        # The LLM sets abuse_contact/scraping_reason but NOT allowed_hosts.
                        "http": {"abuse_contact": "noreply@example.com", "scraping_reason": "demo"},
                    },
                    "rationale": "fetch each url",
                },
            ),
            why="scrape the synthetic pages",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=proposal,
        )

        assert result.tool_result.success is True, f"set_pipeline failed: {getattr(result.tool_result, 'data', result.tool_result)}"
        web_scrape_node = result.state.nodes[0]
        assert web_scrape_node.plugin == "web_scrape"
        http = dict(web_scrape_node.options["http"])
        # No SSRF allowlist is injected — the plugin default 'public_only' applies.
        assert "allowed_hosts" not in http
        # The LLM-set http fields are committed unchanged.
        assert http["abuse_contact"] == "noreply@example.com"
        assert http["scraping_reason"] == "demo"

    def test_row_filter_claim_on_value_transform_rejected(self) -> None:
        """The 2026-07-10 web-eval defect (elspeth-c1d78dac70): the solver
        proposed `value_transform` writing a `_keep` boolean and claimed rows
        where the expression is False would error-route out. value_transform is
        assignment-only — every row passes through — so the accept must reject
        the false row-filter claim and coach toward the honest alternatives."""
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        catalog, snapshot = _trained_policy()
        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "inventory.csv", "schema": {"mode": "observed", "guaranteed_fields": ["quantity"]}},
                observed_columns=("quantity",),
                sample_rows=({"quantity": "5"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "filtered.jsonl", "schema": {"mode": "observed"}},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )

        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "value_transform",
                    "options": {
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "_keep", "expression": "row['quantity'] > 3"}],
                    },
                    "rationale": "Adds a _keep flag; rows where the expression is False will error-route out, keeping only rows with quantity > 3.",
                },
            ),
            why="keep only rows where quantity is greater than 3",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=proposal,
        )

        assert result.tool_result.success is False
        # Failure leaves state and session untouched — no partial commit.
        assert result.state is step_2.state
        assert result.session.step is not GuidedStep.STEP_4_WIRE
        # The rejection names the plugin and the honest construct so the
        # repair loop (repair_context) can coach the model.
        messages = " ".join(e.message for e in result.tool_result.validation.errors)
        assert "value_transform" in messages
        assert "gate" in messages

    def test_value_transform_compute_rationale_accepted(self) -> None:
        """A genuine assignment rationale must NOT trip the row-filter-claim
        lint — value_transform used for what it actually does stays accepted."""
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        catalog, snapshot = _trained_policy()
        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "inventory.csv", "schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "priced.jsonl", "schema": {"mode": "observed"}},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )

        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "value_transform",
                    "options": {
                        "schema": {"mode": "observed"},
                        # Benign row-adjacent verb ("Remove … from … each row")
                        # describes a FIELD edit, not row filtering — must pass.
                        "operations": [{"target": "price_doubled", "expression": "row['price'] * 2"}],
                    },
                    "rationale": "Remove the currency ambiguity by computing a numeric price_doubled field for each row.",
                },
            ),
            why="compute the doubled price per row",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=proposal,
        )

        assert result.tool_result.success is True, f"set_pipeline failed: {getattr(result.tool_result, 'data', result.tool_result)}"
        assert result.session.step is GuidedStep.STEP_4_WIRE
        assert result.state.nodes[0].plugin == "value_transform"

    def test_empty_proposal_is_valid_passthrough_to_wire(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        catalog, snapshot = _trained_policy()
        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="json",
                options={"path": "rows.json", "schema": {"mode": "observed", "guaranteed_fields": ["line"]}},
                observed_columns=("line",),
                sample_rows=({"line": "alpha"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.json", "schema": {"mode": "observed"}},
                        required_fields=(),
                        schema_mode="observed",
                    ),
                ),
            ),
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=ChainProposal(steps=(), why="source rows already match the output"),
        )

        assert result.tool_result.success is True
        assert result.state.nodes == ()
        assert result.state.sources["source"].on_success == "main"
        assert result.session.step is GuidedStep.STEP_4_WIRE
        assert result.session.step_3_proposal is not None
        assert result.session.step_3_proposal.steps == ()

    def test_refuses_when_no_source(self) -> None:
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        proposal = ChainProposal(
            steps=({"plugin": "passthrough", "options": {"schema": {"mode": "observed"}}, "rationale": "x"},),
            why="x",
        )
        catalog, snapshot = _trained_policy()
        with pytest.raises(InvariantError, match=r"no.*source|committed source"):
            handle_step_3_chain_accept(
                state=_empty_state(),
                session=GuidedSession.initial(),
                catalog=catalog,
                plugin_snapshot=snapshot,
                proposal=proposal,
            )


class TestTerminalStampInvariant:
    """The accept seams redirect to STEP_4_WIRE; only wire confirm completes."""

    def test_chain_accept_redirects_to_wire_not_completed(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import ChainProposal, TerminalKind, TerminalState
        from elspeth.web.composer.guided.steps import (
            handle_step_3_chain_accept,
            handle_step_4_wire_confirm,
        )

        state = _empty_state()
        session = GuidedSession.initial()
        catalog, snapshot = _trained_policy()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
            plugin_snapshot=snapshot,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                        required_fields=("price",),
                        schema_mode="observed",
                    ),
                ),
            ),
        )
        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "passthrough",
                    "options": {"schema": {"mode": "observed", "guaranteed_fields": ["price"]}},
                    "rationale": "echo rows",
                },
            ),
            why="single-step chain",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=replace(
                step_2.session,
                terminal=TerminalState(
                    kind=TerminalKind.COMPLETED,
                    reason=None,
                    pipeline_yaml="stale yaml",
                ),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            proposal=proposal,
        )

        assert result.tool_result.success is True
        assert result.session.terminal is None
        assert result.session.step is GuidedStep.STEP_4_WIRE
        assert result.session.step_3_proposal is proposal

        wire = handle_step_4_wire_confirm(state=result.state, session=result.session)

        assert wire.tool_result.success is True
        assert wire.session.terminal is not None
        assert wire.session.terminal.kind is TerminalKind.COMPLETED
        assert wire.session.terminal.pipeline_yaml is not None
        assert len(wire.session.terminal.pipeline_yaml) > 0

    def test_wire_confirm_invalid_pipeline_leaves_terminal_unset(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.steps import handle_step_4_wire_confirm

        session = replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)

        result = handle_step_4_wire_confirm(state=_empty_state(), session=session)

        assert result.tool_result.success is False
        assert result.session.terminal is None
        assert result.session.step is GuidedStep.STEP_4_WIRE


class TestStep1ObservedColumnsDerivation:
    """handle_step_1_source backfills observed_columns from the blob content when
    the resolved source left them empty.

    The LLM's resolve_source sometimes returns observed_columns=[]; the
    transform-chain build keys on observed_columns, so empty columns silently
    route the canonical web-scrape tutorial to a degenerate chain.
    handle_step_1_source is
    the commit convergence point (every step-1 commit passes through it, and it
    already reads the blob for blob_ref enrichment), so the data-authoritative
    backfill lives here.
    """

    @pytest.fixture
    def _seeded_json_urls(self, tmp_path):
        from datetime import UTC, datetime

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        session_id = str(uuid4())
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )
        blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        storage_path = storage_dir / f"{blob_id}_urls.json"
        body = b'[{"url": "https://example/a"}, {"url": "https://example/b"}]'
        storage_path.write_bytes(body)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=blob_id,
                    session_id=session_id,
                    filename="urls.json",
                    mime_type="application/json",
                    size_bytes=len(body),
                    content_hash=_content_hash(body),
                    storage_path=str(storage_path),
                    created_at=now,
                    created_by="assistant",
                    source_description=None,
                    status="ready",
                )
            )
        return engine, session_id, str(storage_path)

    def test_derives_observed_columns_from_blob_when_empty(self, _seeded_json_urls) -> None:
        engine, session_id, storage_path = _seeded_json_urls
        catalog, snapshot = _trained_policy()
        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": storage_path, "schema": {"mode": "observed"}},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            data_dir=None,
            session_engine=engine,
            session_id=session_id,
        )
        assert result.tool_result.success is True
        assert result.session.step_1_result is not None
        # backfilled from the blob's json content
        assert result.session.step_1_result.observed_columns == ("url",)
        # blob_ref enrichment still applies alongside
        assert "blob_ref" in result.session.step_1_result.options

    def test_keeps_observed_columns_when_llm_supplied_them(self, _seeded_json_urls) -> None:
        engine, session_id, storage_path = _seeded_json_urls
        catalog, snapshot = _trained_policy()
        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": storage_path, "schema": {"mode": "observed"}},
                observed_columns=("url", "extra"),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            data_dir=None,
            session_engine=engine,
            session_id=session_id,
        )
        assert result.tool_result.success is True
        # non-empty LLM columns are preserved (bounded scan must not overwrite)
        assert result.session.step_1_result.observed_columns == ("url", "extra")

    def test_resolves_blob_ref_path_sentinel_to_real_storage_path(self, _seeded_json_urls) -> None:
        # Fix B round-trip: a re-submitted schema_form carries the masked
        # blob:<ref> path (the absolute storage_path is kept off the wire). The
        # commit must restore the real path so the pipeline can read the blob.
        engine, session_id, storage_path = _seeded_json_urls
        blob_id = storage_path.split("/")[-1].split("_", 1)[0]
        catalog, snapshot = _trained_policy()
        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": f"blob:{blob_id}", "schema": {"mode": "observed"}},
                observed_columns=("url",),
                sample_rows=(),
            ),
            catalog=catalog,
            plugin_snapshot=snapshot,
            data_dir=None,
            session_engine=engine,
            session_id=session_id,
        )
        assert result.tool_result.success is True
        committed_path = result.session.step_1_result.options["path"]
        # the real absolute path is restored, not the blob: sentinel
        assert committed_path == storage_path
        assert not str(committed_path).startswith("blob:")
        # blob_ref enrichment runs on the restored real path
        assert "blob_ref" in result.session.step_1_result.options

    def test_blob_ref_path_to_unknown_blob_raises(self, _seeded_json_urls) -> None:
        # A sentinel that resolves to no blob must fail loudly, never commit a
        # broken blob: path that the run cannot open.
        engine, session_id, _ = _seeded_json_urls
        catalog, snapshot = _trained_policy()
        with pytest.raises(InvariantError):
            handle_step_1_source(
                state=_empty_state(),
                session=GuidedSession.initial(),
                resolved=SourceResolved(
                    plugin="json",
                    options={"path": f"blob:{uuid4()}", "schema": {"mode": "observed"}},
                    observed_columns=("url",),
                    sample_rows=(),
                ),
                catalog=catalog,
                plugin_snapshot=snapshot,
                data_dir=None,
                session_engine=engine,
                session_id=session_id,
            )

    def test_observed_columns_from_blob_rejects_malformed_blob_record(self) -> None:
        with pytest.raises(InvariantError, match="storage_path"):
            _observed_columns_from_blob(
                {
                    "storage_path": 42,
                    "filename": "rows.json",
                    "mime_type": "application/json",
                }
            )
