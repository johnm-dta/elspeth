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


def test_session_routes_package_preserves_legacy_patch_seams() -> None:
    routes = importlib.import_module("elspeth.web.sessions.routes")

    assert callable(routes._dispatch_guided_respond)
    assert callable(routes._persist_tool_invocations)
    _slog = routes.slog
    assert callable(routes.step_advance)


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
    assert resp.recipe_match is TUTORIAL_PROFILE.recipe_match
    assert resp.advisor_checkpoints is TUTORIAL_PROFILE.advisor_checkpoints
