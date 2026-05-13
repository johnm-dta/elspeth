from __future__ import annotations

import ast
from pathlib import Path

SERVICE_PATH = Path("src/elspeth/web/composer/service.py")


def test_composer_service_impl_not_patched_at_module_tail() -> None:
    tree = ast.parse(SERVICE_PATH.read_text(encoding="utf-8"))
    forbidden_assignments = {
        ("ComposerServiceImpl", "_run_one_turn_for_test"),
        ("ComposerServiceImpl", "_serialize_response_via_walker"),
        ("ComposerServiceImpl", "_state_payload_for_compose_turn_for_test"),
        ("ComposerServiceImpl", "__init__"),
    }
    hits: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and (target.value.id, target.attr) in forbidden_assignments
            ):
                hits.append(f"{target.value.id}.{target.attr}")
            if isinstance(target, ast.Name) and target.id.startswith("_PHASE3_"):
                hits.append(target.id)
    assert hits == []
