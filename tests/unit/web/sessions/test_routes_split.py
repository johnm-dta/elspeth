from __future__ import annotations

import ast
import importlib
from pathlib import Path


def test_session_routes_are_split_by_resource_module() -> None:
    modules = {
        "elspeth.web.sessions.routes._helpers",
        "elspeth.web.sessions.routes.sessions",
        "elspeth.web.sessions.routes.composer",
        "elspeth.web.sessions.routes.messages",
        "elspeth.web.sessions.routes.runs",
        "elspeth.web.sessions.routes.interpretation",
    }

    for module_name in modules:
        importlib.import_module(module_name)


def test_session_routes_package_exports_router_factory() -> None:
    routes = importlib.import_module("elspeth.web.sessions.routes")

    assert callable(routes.create_session_router)


def test_session_routes_package_does_not_export_removed_guided_adapters() -> None:
    routes = importlib.import_module("elspeth.web.sessions.routes")

    assert not hasattr(routes, "_dispatch_guided_respond")
    assert not hasattr(routes, "step_advance")


def test_removed_guided_architecture_has_no_compatibility_stubs() -> None:
    removed_modules = (
        Path("src/elspeth/web/composer/guided/chain_solver.py"),
        Path("src/elspeth/web/composer/guided/steps.py"),
        Path("src/elspeth/web/sessions/_guided_solve_chain.py"),
    )
    assert all(not path.exists() for path in removed_modules)

    helpers_tree = ast.parse(Path("src/elspeth/web/sessions/routes/_helpers.py").read_text(encoding="utf-8"))
    emitter_tree = ast.parse(Path("src/elspeth/web/composer/guided/emitters.py").read_text(encoding="utf-8"))
    definitions = {
        node.name
        for tree in (helpers_tree, emitter_tree)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert definitions.isdisjoint(
        {
            "_build_policy_aware_wire_turn",
            "_dispatch_guided_respond",
            "_emit_wire_turn",
            "_guided_step_index",
            "_prefilled_recipe_slot_mismatches",
            "_summarize_guided_response",
            "_validate_control_signal",
            "build_step_3_schema_form_turn",
        }
    )


def test_session_route_package_uses_explicit_imports() -> None:
    routes_dir = Path("src/elspeth/web/sessions/routes")

    offenders: list[str] = []
    for source_path in sorted(routes_dir.glob("*.py")):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
                offenders.append(f"{source_path}:{node.lineno}")

    assert offenders == []


def test_workflow_profile_response_none_for_empty_profile() -> None:
    from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
    from elspeth.web.composer.guided.state_machine import GuidedSession
    from elspeth.web.sessions.routes._helpers import _workflow_profile_response

    empty_session = GuidedSession.initial()  # default = EMPTY_PROFILE
    assert empty_session.profile == EMPTY_PROFILE
    assert _workflow_profile_response(empty_session) is None

    tutorial_session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
    resp = _workflow_profile_response(tutorial_session)
    assert resp is not None
    assert resp.coaching is TUTORIAL_PROFILE.coaching
    assert resp.bookends is TUTORIAL_PROFILE.bookends
