"""Architecture proof for the hard deletion of guided chain authoring."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi import FastAPI

from elspeth.web.sessions.routes import create_session_router

_ROOT = Path(__file__).resolve().parents[5]
_PRODUCTION_ROOT = _ROOT / "src" / "elspeth" / "web"
_RETIRED_CONTRACTS = (
    "ChainProposal",
    "PROPOSE_CHAIN",
    "solve_chain",
    "handle_step_3_chain_accept",
    "step_3_edit_index",
    "chain_solver",
    "_guided_solve_chain",
    "accepted_step_index",
    "edit_step_index",
    "chain_in",
)
_REMOVED_MODULES = (
    "elspeth.web.composer.guided.chain_solver",
    "elspeth.web.sessions._guided_solve_chain",
    "elspeth.web.composer.guided.steps",
)


def _production_paths() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in _PRODUCTION_ROOT.rglob("*")
            if path.suffix in {".py", ".ts", ".tsx"}
            and "tests" not in path.parts
            and ".test." not in path.name
            and "__pycache__" not in path.parts
        )
    )


def _module_tree(relative_path: str) -> ast.Module:
    path = _ROOT / relative_path
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _named_scope(tree: ast.Module, name: str, *, owner: str | None = None) -> ast.FunctionDef | ast.AsyncFunctionDef:
    nodes: list[ast.stmt]
    if owner is None:
        nodes = tree.body
    else:
        class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
        nodes = class_node.body
    return next(node for node in nodes if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name)


def _call_count(scope: ast.AST, name: str) -> int:
    return sum(
        1
        for node in ast.walk(scope)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == name) or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )
    )


def test_retired_chain_modules_are_deleted_and_non_importable() -> None:
    for module_name in _REMOVED_MODULES:
        assert importlib.util.find_spec(module_name) is None
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


def test_active_source_and_openapi_expose_no_retired_chain_contract() -> None:
    source_violations = {
        str(path.relative_to(_ROOT)): token
        for path in _production_paths()
        for token in _RETIRED_CONTRACTS
        if token in path.read_text(encoding="utf-8")
    }

    app = FastAPI()
    app.include_router(create_session_router())
    openapi = json.dumps(app.openapi(), sort_keys=True, separators=(",", ":"))
    schema_violations = [token for token in _RETIRED_CONTRACTS if token in openapi]

    assert source_violations == {}
    assert schema_violations == []


def test_reachable_surfaces_share_one_planner_implementation_boundary() -> None:
    from elspeth.web.composer import pipeline_planner
    from elspeth.web.composer import service as service_module

    service_tree = _module_tree("src/elspeth/web/composer/service.py")
    guided_route_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided.py")
    guided_full_route_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided_plan.py")
    freeform = _named_scope(service_tree, "compose", owner="ComposerServiceImpl")
    freeform_planner = _named_scope(service_tree, "_plan_and_stage_empty_pipeline", owner="ComposerServiceImpl")
    guided = _named_scope(service_tree, "plan_guided_pipeline", owner="ComposerServiceImpl")
    guided_full = _named_scope(service_tree, "plan_guided_full_pipeline", owner="ComposerServiceImpl")
    guided_route = _named_scope(guided_route_tree, "post_guided_respond")
    guided_full_route = _named_scope(guided_full_route_tree, "post_guided_plan")

    # Freeform has one model-planner fallback. Guided-staged and tutorial use
    # the same reachable adapter for initial and revision branches; the adapter
    # itself contains exactly one call to the shared imported implementation.
    assert service_module.plan_pipeline is pipeline_planner.plan_pipeline
    assert _call_count(freeform, "_plan_and_stage_empty_pipeline") == 1
    assert _call_count(freeform_planner, "plan_pipeline") == 1
    assert _call_count(guided_route, "plan_guided_pipeline") == 3
    assert _call_count(guided, "plan_pipeline") == 1
    assert _call_count(guided_full_route, "plan_guided_full_pipeline") == 1
    assert _call_count(guided_full_route, "plan_pipeline") == 0
    assert _call_count(guided_full, "plan_pipeline") == 1
    guided_source = ast.unparse(guided)
    assert "PlannerSurface.GUIDED_STAGED" in guided_source
    assert "PlannerSurface.TUTORIAL_PROFILE" in guided_source
    assert "PlannerSurface.GUIDED_FULL" in ast.unparse(guided_full)


def test_reachable_surfaces_share_one_lock_assuming_commit_boundary() -> None:
    compose_tree = _module_tree("src/elspeth/web/sessions/routes/composer/compose.py")
    guided_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided.py")
    freeform_route = _named_scope(compose_tree, "recompose")
    guided_route = _named_scope(guided_tree, "post_guided_respond")

    assert _call_count(freeform_route, "settle_pipeline_proposal_under_compose_lock") == 1
    assert _call_count(guided_route, "prepare_pipeline_proposal_commit") == 1
    assert _call_count(guided_route, "accept_guided_pipeline_proposal") == 1


def test_guided_full_router_seam_is_late_bound_with_one_production_controller() -> None:
    """Guided-full stays isolated in the late-bound Task-5 route module."""

    guided_source = (_ROOT / "src/elspeth/web/sessions/routes/composer/guided.py").read_text(encoding="utf-8")
    guided_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided.py")
    guided_plan_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided_plan.py")
    app = FastAPI()
    app.include_router(create_session_router())

    assert "PlannerSurface.GUIDED_FULL" not in guided_source
    assert "/api/sessions/{session_id}/guided/plan" in app.openapi()["paths"]
    assert (
        next(node.name for node in reversed(guided_tree.body) if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef))
        == "post_guided_convert"
    )
    late_import, late_include = guided_tree.body[-2:]
    assert isinstance(late_import, ast.ImportFrom)
    assert late_import.level == 1
    assert late_import.module == "guided_plan"
    assert [(alias.name, alias.asname) for alias in late_import.names] == [("router", "guided_plan_router")]
    assert ast.unparse(late_include) == "router.include_router(guided_plan_router)"
    handlers = [
        node.name
        for node in guided_plan_tree.body
        if isinstance(node, ast.AsyncFunctionDef) and any(isinstance(decorator, ast.Call) for decorator in node.decorator_list)
    ]
    assert handlers == ["post_guided_plan"]


def test_guided_route_handler_module_positions_remain_at_the_signed_layout() -> None:
    guided_tree = _module_tree("src/elspeth/web/sessions/routes/composer/guided.py")
    expected_positions = {
        "get_guided": 47,
        "get_guided_tutorial_sample": 48,
        "post_guided_reenter": 49,
        "reconcile_guided_start_operation": 50,
        "post_guided_start": 51,
        "post_guided_respond": 66,
        "post_guided_chat": 68,
        "post_guided_convert": 69,
    }

    actual_positions = {
        node.name: index
        for index, node in enumerate(guided_tree.body)
        if isinstance(node, ast.AsyncFunctionDef) and node.name in expected_positions
    }

    assert actual_positions == expected_positions
