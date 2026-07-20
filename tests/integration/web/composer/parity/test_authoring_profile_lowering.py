"""Companion to the readiness-preflight probe: the incremental transform
authoring tools must not false-positive on an operator-profiled multi-query LLM
node either.

``set_pipeline`` already guards the sequential-multi-query retry-budget /
managed-identity policy behind ``if "profile" not in node_options`` — an operator
profile injects the web-safe retry budget only at LOWERING, and the profile-
lowered executable is validated by ``_prevalidate_transform_for_context`` (which
lowers first). The incremental tools ``upsert_node`` / ``patch_node_options`` /
``splice_transform`` ran the raw policy UNCONDITIONALLY, so a profiled multi-query
node was uncommittable through those surfaces — the same false-positive as the
execution readiness preflight, at the authoring surface. These regressions drive
the real profile-aware policy stack (as ``app.py`` wires it) and assert the
profiled node clears the retry-budget gate.

The NON-profiled rejection (the safety gate that must survive the guard) is
covered by ``test_tools.TestTransformLlmRetryBudgetPolicy`` — the parity web
policy requires the profile authoring form, so a direct provider/model ``llm``
node is rejected earlier as ``profile_unavailable`` and cannot reach the
retry-budget gate on this surface.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.auth.models import UserIdentity
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.tools._dispatch import execute_tool


def _empty_state() -> CompositionState:
    return CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)


def _profiled_multi_query_options() -> dict[str, Any]:
    return {
        "profile": "task-role",
        "temperature": 0.0,
        "required_input_fields": ["color_name"],
        "prompt_template": "Assess {{ color_name }}.",
        "queries": {
            "blue": {
                "input_fields": {"color_name": "color_name"},
                "response_format": "structured",
                "output_fields": [{"suffix": "amount", "type": "integer"}],
            }
        },
        "schema": {"mode": "observed"},
    }


_RETRY_BUDGET_MARKER = "sequential multi-query llm"


def _profiled_node(options: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "assess_blue",
        "node_type": "transform",
        "plugin": "llm",
        "input": "rows",
        "on_success": "assessed",
        "on_error": "discard",
        "options": options,
    }


def _tool_kwargs(parity_env: Any) -> dict[str, Any]:
    state = parity_env.app.state
    snapshot = state.plugin_snapshot_factory(UserIdentity(user_id="alice", username="alice"))
    policy = PolicyCatalogView(state.catalog_service, snapshot, state.operator_profile_registry)
    return {
        "policy": policy,
        "plugin_snapshot": snapshot,
        "data_dir": str(parity_env.data_dir),
        "session_engine": state.session_engine,
        "secret_service": None,
        "user_id": "alice",
    }


def _run(parity_env: Any, tool: str, args: dict[str, Any], state: CompositionState) -> Any:
    kwargs = _tool_kwargs(parity_env)
    return execute_tool(tool, args, state, kwargs.pop("policy"), **kwargs)


def test_upsert_node_admits_profiled_multi_query_llm(parity_env: Any) -> None:
    result = _run(parity_env, "upsert_node", _profiled_node(_profiled_multi_query_options()), _empty_state())
    assert result.success is True, result.data
    node = next(n for n in result.updated_state.nodes if n.id == "assess_blue")
    assert node.options.get("profile") == "task-role"
    assert "max_capacity_retry_seconds" not in node.options


def test_patch_node_options_admits_profiled_multi_query_llm(parity_env: Any) -> None:
    created = _run(parity_env, "upsert_node", _profiled_node(_profiled_multi_query_options()), _empty_state())
    assert created.success is True, created.data

    patched = _run(
        parity_env,
        "patch_node_options",
        {"node_id": "assess_blue", "patch": {"temperature": 0.2}},
        created.updated_state,
    )
    assert patched.success is True, patched.data
    node = next(n for n in patched.updated_state.nodes if n.id == "assess_blue")
    assert node.options.get("temperature") == 0.2
    assert node.options.get("profile") == "task-role"


def _splice_base_state() -> CompositionState:
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="rows",
            options={"path": "rows.csv", "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="before",
                node_type="transform",
                plugin="passthrough",
                input="rows",
                on_success="middle",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
            NodeSpec(
                id="after",
                node_type="transform",
                plugin="passthrough",
                input="middle",
                on_success="result",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(
            EdgeSpec(id="source-before", from_node="source", to_node="before", edge_type="on_success", label=None),
            EdgeSpec(id="before-after", from_node="before", to_node="after", edge_type="on_success", label=None),
            EdgeSpec(id="after-output", from_node="after", to_node="result", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="result",
                plugin="json",
                options={"path": "out.jsonl", "schema": {"mode": "observed"}, "mode": "write", "collision_policy": "auto_increment"},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=4,
    )


def test_splice_transform_clears_retry_budget_gate_for_profiled_multi_query(parity_env: Any) -> None:
    # The splice retry-budget gate (transforms.py:_execute_splice_transform) runs
    # BEFORE the whole-pipeline splice validation, so before the fix a profiled
    # multi-query node was rejected here with the retry-budget error. After the
    # fix the gate is cleared; the node reaches the orthogonal context-aware
    # splice validation (which may accept or reject the spliced *graph* for
    # unrelated reasons). Assert only that the retry-budget gate no longer fires.
    args = {
        "predecessor_id": "before",
        "successor_id": "after",
        "node": {
            "id": "assess_blue",
            "plugin": "llm",
            "options": _profiled_multi_query_options(),
            "on_error": "discard",
        },
    }
    result = _run(parity_env, "splice_transform", args, _splice_base_state())
    error = "" if result.success else str(result.data.get("error", "")).lower()
    assert _RETRY_BUDGET_MARKER not in error, result.data
