"""Tests for identity_node_advisory — flags dead-weight passthrough transforms.

The advisory fires for ``transform → passthrough → sink`` chains where the
passthrough has no schema declaration to anchor an unsatisfied edge contract
and is not on a gate-fork branch. See plan:
``/home/john/.claude/plans/dispatch-prompt-floofy-noodle.md``.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.execution.validation import (
    _CHECK_IDENTITY_NODE_ADVISORY,
    _find_identity_node_advisories,
)

# ── Fixture builders ────────────────────────────────────────────────────


def _make_passthrough_node(
    node_id: str = "forward_summaries",
    input_field: str = "summary_in",
    on_success: str = "json_out",
    options: dict[str, Any] | None = None,
) -> NodeSpec:
    """Identity-shaped passthrough node — no fork, single in/out."""
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="passthrough",
        input=input_field,
        on_success=on_success,
        on_error="discard",
        options=options or {},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _make_real_transform_node(
    node_id: str = "summarize_page",
    on_success: str = "summary_in",
) -> NodeSpec:
    """A non-passthrough upstream transform feeding the passthrough."""
    return NodeSpec(
        id=node_id,
        node_type="transform",
        plugin="value_transform",
        input="page_in",
        on_success=on_success,
        on_error="discard",
        options={"operations": [{"target": "x", "expression": "1"}]},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def _make_observed_sink(name: str = "json_out") -> OutputSpec:
    """Sink with ``schema.mode=observed`` — accepts any inbound row."""
    return OutputSpec(
        name=name,
        plugin="json",
        options={"path": "out.json", "schema": {"mode": "observed"}},
        on_write_failure="discard",
    )


def _make_state_with(
    nodes: tuple[NodeSpec, ...],
    outputs: tuple[OutputSpec, ...],
    source: SourceSpec | None = None,
) -> CompositionState:
    """Construct a CompositionState with a default csv source unless overridden."""
    return CompositionState(
        source=source
        or SourceSpec(
            plugin="csv",
            on_success="page_in",
            options={"path": "in.csv"},
            on_validation_failure="discard",
        ),
        nodes=nodes,
        edges=(),
        outputs=outputs,
        metadata=PipelineMetadata(),
        version=1,
    )


# ── Constant + stub tests ───────────────────────────────────────────────


def test_check_constant_value() -> None:
    """The check name string is the public contract — frontend and LLM both read it."""
    assert _CHECK_IDENTITY_NODE_ADVISORY == "identity_node_advisory"


def test_helper_returns_empty_list_for_empty_state() -> None:
    """Stub: helper exists, returns empty list when state has no nodes."""
    state = CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    findings = _find_identity_node_advisories(state)
    assert findings == []


# ── Detection — positive case (canonical repro) ─────────────────────────


def test_identity_passthrough_to_observed_sink_is_flagged() -> None:
    """Canonical repro: transform → passthrough(no schema) → observed sink."""
    state = _make_state_with(
        nodes=(
            _make_real_transform_node(),
            _make_passthrough_node(),
        ),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.node_id == "forward_summaries"
    assert finding.upstream_id == "summarize_page"
    assert finding.sink_name == "json_out"
    assert finding.sink_schema_mode == "observed"


# ── Negative cases — exemptions and out-of-scope plugins ────────────────


def test_value_transform_with_empty_options_is_not_flagged() -> None:
    """Non-passthrough plugins are out of scope per dispatch — only literal
    ``plugin == "passthrough"`` triggers."""
    value_node = NodeSpec(
        id="vt",
        node_type="transform",
        plugin="value_transform",
        input="summary_in",
        on_success="json_out",
        on_error="discard",
        options={},  # empty options — would otherwise look identity-shaped
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    state = _make_state_with(
        nodes=(_make_real_transform_node(on_success="summary_in"), value_node),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert findings == []


def test_passthrough_with_schema_fields_is_not_flagged() -> None:
    """Concept-5 legitimate use: passthrough declares schema to anchor a
    downstream contract (skill lines 758-768)."""
    node = _make_passthrough_node(
        options={
            "schema": {
                "mode": "fixed",
                "fields": ["id: int", "name: str"],
            }
        },
    )
    state = _make_state_with(
        nodes=(_make_real_transform_node(), node),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert findings == [], (
        "Passthrough with explicit schema.fields is the legitimate Concept-5 use case (skill lines 758-768) and must not be flagged."
    )


def test_passthrough_with_empty_schema_fields_is_flagged() -> None:
    """Empty ``schema.fields`` list is not a real anchor — flag it."""
    node = _make_passthrough_node(
        options={"schema": {"mode": "fixed", "fields": []}},
    )
    state = _make_state_with(
        nodes=(_make_real_transform_node(), node),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert len(findings) == 1


def test_passthrough_with_schema_mode_only_is_flagged() -> None:
    """``schema={mode: observed}`` alone is not Concept-5 anchoring — flag it."""
    node = _make_passthrough_node(
        options={"schema": {"mode": "observed"}},
    )
    state = _make_state_with(
        nodes=(_make_real_transform_node(), node),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert len(findings) == 1


def test_passthrough_downstream_of_gate_fork_is_not_flagged() -> None:
    """Per ``pipeline_composer.md:1517-1518``, per-branch passthrough nodes
    after a fork are the documented legitimate pattern.  The detector skips
    them because the upstream is a gate."""
    upstream_to_gate = NodeSpec(
        id="enrich",
        node_type="transform",
        plugin="value_transform",
        input="page_in",
        on_success="splitter_in",
        on_error="discard",
        options={"operations": [{"target": "x", "expression": "1"}]},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )
    gate = NodeSpec(
        id="splitter",
        node_type="gate",
        plugin=None,
        input="splitter_in",
        on_success=None,
        on_error="discard",
        options={},
        condition=None,
        routes=None,
        fork_to=("path_a_in", "path_b_in"),
        branches=None,
        policy=None,
        merge=None,
    )
    branch_a = _make_passthrough_node(
        node_id="path_a",
        input_field="path_a_in",
        on_success="json_out",
    )
    state = _make_state_with(
        nodes=(upstream_to_gate, gate, branch_a),
        outputs=(_make_observed_sink(),),
    )
    findings = _find_identity_node_advisories(state)
    assert findings == [], (
        "Passthrough fed by gate.fork_to is the legitimate per-branch pattern (skill lines 1517-1518) and must not be flagged."
    )


def test_identity_passthrough_to_fixed_sink_is_flagged() -> None:
    """Sink schema mode is irrelevant — passthrough still adds no contract benefit."""
    fixed_sink = OutputSpec(
        name="csv_out",
        plugin="csv",
        options={
            "path": "out.csv",
            "schema": {"mode": "fixed", "fields": ["id: int", "summary: str"]},
        },
        on_write_failure="discard",
    )
    node = _make_passthrough_node(on_success="csv_out")
    state = _make_state_with(
        nodes=(_make_real_transform_node(), node),
        outputs=(fixed_sink,),
    )
    findings = _find_identity_node_advisories(state)
    assert len(findings) == 1
    assert findings[0].sink_schema_mode == "fixed"
