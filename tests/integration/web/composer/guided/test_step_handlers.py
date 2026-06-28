"""Integration tests for guided-mode step commit handlers.

These tests use real CompositionState and the real tools.py mutation
helpers; only the catalog is constructed via the public test seam
(create_catalog_service()).
"""

from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.guided.steps import (
    StepHandlerResult,
    handle_step_1_source,
    handle_step_2_sink,
)
from elspeth.web.composer.state import CompositionState, PipelineMetadata
from elspeth.web.dependencies import create_catalog_service


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


class TestStep1Handler:
    def test_commits_source_to_state_on_success(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

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
        catalog = create_catalog_service()

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
        )

        assert result.tool_result.success is True
        assert result.state.sources["source"].on_validation_failure == "quarantine_sink"
        assert result.session.step_1_result is not None
        assert result.session.step_1_result.on_validation_failure == "quarantine_sink"

    def test_returns_failure_unchanged_session_when_plugin_unknown(self) -> None:
        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

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
        )

        assert result.tool_result.success is False
        assert result.state is state  # unchanged on failure
        assert result.session.step_1_result is None


class TestStep2Handler:
    def test_commits_outputs_to_state_on_success(self) -> None:
        # Step 1 sets a CSV source with on_success="main".
        # Then Step 2 attaches a json output named "main".
        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed"}},
                observed_columns=("a",),
                sample_rows=({"a": "1"},),
            ),
        )
        assert step_1.tool_result.success is True

        result = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
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
        catalog = create_catalog_service()

        # _execute_set_output validates plugin-name first, before any source
        # presence check — so this exercises the failure path on empty state.
        result = handle_step_2_sink(
            state=state,
            session=GuidedSession.initial(),
            catalog=catalog,
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
        catalog = create_catalog_service()

        with pytest.raises(InvariantError, match="empty list"):
            handle_step_2_sink(
                state=_empty_state(),
                session=GuidedSession.initial(),
                resolved=SinkResolved(outputs=()),
                catalog=catalog,
            )


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

    def test_chain_accepted_commits_and_redirects_to_wire(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep
        from elspeth.web.composer.guided.state_machine import (
            ChainProposal,
        )
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        state = _empty_state()
        session = GuidedSession.initial()
        catalog = create_catalog_service()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed"}},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
        )
        assert step_1.tool_result.success is True

        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
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
                    "options": {"schema": {"mode": "observed"}},
                    "rationale": "echo rows; minimal transform for wiring proof",
                },
            ),
            why="single-step chain verifying chain_in→main wiring",
        )

        result = handle_step_3_chain_accept(
            state=step_2.state,
            session=step_2.session,
            catalog=catalog,
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

        catalog = create_catalog_service()
        step_1 = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            catalog=catalog,
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

    def test_refuses_empty_proposal(self) -> None:
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        with pytest.raises(InvariantError, match="zero steps"):
            handle_step_3_chain_accept(
                state=_empty_state(),
                session=GuidedSession.initial(),
                catalog=create_catalog_service(),
                proposal=ChainProposal(steps=(), why="empty"),
            )

    def test_refuses_when_no_source(self) -> None:
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        proposal = ChainProposal(
            steps=({"plugin": "passthrough", "options": {"schema": {"mode": "observed"}}, "rationale": "x"},),
            why="x",
        )
        with pytest.raises(InvariantError, match=r"no.*source|committed source"):
            handle_step_3_chain_accept(
                state=_empty_state(),
                session=GuidedSession.initial(),
                catalog=create_catalog_service(),
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
        catalog = create_catalog_service()

        step_1 = handle_step_1_source(
            state=state,
            session=session,
            catalog=catalog,
            resolved=SourceResolved(
                plugin="csv",
                options={"path": "x.csv", "schema": {"mode": "observed"}},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
        )
        step_2 = handle_step_2_sink(
            state=step_1.state,
            session=step_1.session,
            catalog=catalog,
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
                    "options": {"schema": {"mode": "observed"}},
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
        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": storage_path, "schema": {"mode": "observed"}},
                observed_columns=(),
                sample_rows=(),
            ),
            catalog=create_catalog_service(),
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
        result = handle_step_1_source(
            state=_empty_state(),
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="json",
                options={"path": storage_path, "schema": {"mode": "observed"}},
                observed_columns=("url", "extra"),
                sample_rows=(),
            ),
            catalog=create_catalog_service(),
            data_dir=None,
            session_engine=engine,
            session_id=session_id,
        )
        assert result.tool_result.success is True
        # non-empty LLM columns are preserved (bounded scan must not overwrite)
        assert result.session.step_1_result.observed_columns == ("url", "extra")
