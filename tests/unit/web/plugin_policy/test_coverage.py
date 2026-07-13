"""Queue-aware required-control coverage over authored pipeline streams."""

from __future__ import annotations

from dataclasses import replace

import pytest

from elspeth.contracts.plugin_capabilities import PluginCapability
from elspeth.web.composer.state import CompositionState, NodeSpec, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.plugin_policy.coverage import build_output_stream_graph, control_coverage_findings


def _node(
    node_id: str,
    plugin: str | None,
    input_stream: str,
    on_success: str | None,
    *,
    node_type: str = "transform",
    options: dict[str, object] | None = None,
    routes: dict[str, str] | None = None,
    fork_to: tuple[str, ...] | None = None,
) -> NodeSpec:
    return NodeSpec(
        id=node_id,
        node_type=node_type,  # type: ignore[arg-type]
        plugin=plugin,
        input=input_stream,
        on_success=on_success,
        on_error="discard" if node_type == "transform" else None,
        options=options or {},
        condition=None,
        routes=routes,
        fork_to=fork_to,
        branches=None,
        policy=None,
        merge=None,
    )


def _queue(queue_id: str) -> NodeSpec:
    return _node(queue_id, None, queue_id, None, node_type="queue")


def _state(*nodes: NodeSpec, source_target: str = "llm_in", sinks: tuple[str, ...] = ("main",)) -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success=source_target,
            options={"path": "rows.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=nodes,
        edges=(),
        outputs=tuple(
            OutputSpec(
                name=name,
                plugin="json",
                options={"path": f"{name}.jsonl", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
            for name in sinks
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _llm(input_stream: str = "llm_in", on_success: str = "main") -> NodeSpec:
    return _node("judge", "llm", input_stream, on_success)


def _shield(node_id: str, input_stream: str, on_success: str, *, detect_only: bool = False) -> NodeSpec:
    return _node(node_id, "azure_prompt_shield", input_stream, on_success, options={"detect_only": detect_only})


def _safety(node_id: str, input_stream: str, on_success: str, *, detect_only: bool = False) -> NodeSpec:
    return _node(node_id, "azure_content_safety", input_stream, on_success, options={"detect_only": detect_only})


@pytest.mark.parametrize(
    ("state", "covered"),
    [
        (_state(_llm()), False),
        (_state(_shield("shield", "raw", "llm_in"), _llm(), source_target="raw"), True),
        # Role mismatch: OUTPUT control cannot satisfy INPUT coverage.
        (_state(_safety("wrong_role", "raw", "llm_in"), _llm(), source_target="raw"), False),
        # A shield after the LLM is too late.
        (_state(_llm(on_success="shield_in"), _shield("shield", "shield_in", "main")), False),
        # Detect-only metadata cannot receive blocking credit.
        (_state(_shield("shield", "raw", "llm_in", detect_only=True), _llm(), source_target="raw"), False),
        # An external-call result after a shield reintroduces untrusted content.
        (
            _state(
                _shield("shield", "raw", "fetch_in"),
                _node("fetch", "web_scrape", "fetch_in", "llm_in"),
                _llm(),
                source_target="raw",
            ),
            False,
        ),
    ],
)
def test_prompt_shield_input_coverage(state: CompositionState, covered: bool) -> None:
    assert (control_coverage_findings(state, PluginCapability.PROMPT_SHIELD) == ()) is covered


def test_prompt_shield_queue_fan_in_requires_every_path() -> None:
    state = _state(
        _node("left", "passthrough", "left_in", "inbound"),
        _shield("right_shield", "right_in", "inbound"),
        _queue("inbound"),
        _llm("inbound"),
        source_target="left_in",
    )
    assert control_coverage_findings(state, PluginCapability.PROMPT_SHIELD)


def test_prompt_shield_queue_fan_in_passes_when_every_path_is_shielded() -> None:
    state = CompositionState(
        sources={
            "left": SourceSpec("csv", "left_raw", {}, "discard"),
            "right": SourceSpec("json", "right_raw", {}, "discard"),
        },
        nodes=(
            _shield("left_shield", "left_raw", "inbound"),
            _shield("right_shield", "right_raw", "inbound"),
            _queue("inbound"),
            _llm("inbound"),
        ),
        edges=(),
        outputs=(OutputSpec("main", "json", {}, "discard"),),
        metadata=PipelineMetadata(),
        version=1,
    )
    assert control_coverage_findings(state, PluginCapability.PROMPT_SHIELD) == ()


def test_prompt_shield_cycle_fails_safe() -> None:
    state = _state(
        _node("cycle_a", "passthrough", "cycle_b_out", "cycle_a_out"),
        _node("cycle_b", "passthrough", "cycle_a_out", "cycle_b_out"),
        _llm("cycle_a_out"),
        source_target="unused",
    )
    assert control_coverage_findings(state, PluginCapability.PROMPT_SHIELD)


def test_extracted_graph_preserves_error_stream_producers() -> None:
    producer = replace(_node("producer", "passthrough", "raw", "success"), on_error="failure")
    graph = build_output_stream_graph((producer,))

    assert graph.producers_by_stream["failure"] == (producer,)


@pytest.mark.parametrize(
    ("state", "covered"),
    [
        (_state(_llm()), False),
        (_state(_llm(on_success="safe_in"), _safety("safety", "safe_in", "main")), True),
        # Role mismatch: INPUT control after the LLM cannot satisfy OUTPUT coverage.
        (_state(_llm(on_success="shield_in"), _shield("wrong_role", "shield_in", "main")), False),
        # A safety control before the LLM does not post-dominate its output.
        (_state(_safety("safety", "raw", "llm_in"), _llm(), source_target="raw"), False),
        (_state(_llm(on_success="safe_in"), _safety("safety", "safe_in", "main", detect_only=True)), False),
        # Multiple effective controls remain valid.
        (
            _state(
                _llm(on_success="safe_a_in"),
                _safety("safety_a", "safe_a_in", "safe_b_in"),
                _safety("safety_b", "safe_b_in", "main"),
            ),
            True,
        ),
    ],
)
def test_content_safety_output_coverage(state: CompositionState, covered: bool) -> None:
    assert (control_coverage_findings(state, PluginCapability.CONTENT_SAFETY) == ()) is covered


def test_content_safety_fan_out_requires_every_sink_path() -> None:
    state = _state(
        _node("judge", "llm", "llm_in", None, fork_to=("safe_in", "unsafe_in")),
        _safety("safety", "safe_in", "safe"),
        _node("unsafe", "passthrough", "unsafe_in", "unsafe"),
        sinks=("safe", "unsafe"),
    )
    assert control_coverage_findings(state, PluginCapability.CONTENT_SAFETY)


def test_content_safety_fan_out_passes_when_each_sink_path_is_controlled() -> None:
    state = _state(
        _node("judge", "llm", "llm_in", None, fork_to=("left_in", "right_in")),
        _safety("left_safety", "left_in", "left"),
        _safety("right_safety", "right_in", "right"),
        sinks=("left", "right"),
    )
    assert control_coverage_findings(state, PluginCapability.CONTENT_SAFETY) == ()


def test_content_safety_unknown_downstream_fails_safe() -> None:
    assert control_coverage_findings(_state(_llm(on_success="missing")), PluginCapability.CONTENT_SAFETY)


def test_content_safety_no_op_thresholds_do_not_receive_blocking_coverage_credit() -> None:
    no_op = _safety("safety", "safe_in", "main")
    no_op = replace(
        no_op,
        options={
            "detect_only": False,
            "thresholds": {"hate": 6, "violence": 6, "sexual": 6, "self_harm": 6},
        },
    )

    assert control_coverage_findings(
        _state(_llm(on_success="safe_in"), no_op),
        PluginCapability.CONTENT_SAFETY,
    )
