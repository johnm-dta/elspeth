from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The composer-service surface spans these files after the 2026-05-23 refactor
# split ComposerServiceImpl helpers into siblings. Module-tail patches of
# ComposerServiceImpl methods or _PHASE3_* sentinels are forbidden in any of
# them — the anti-pattern can hide in a sibling module just as easily as in
# service.py itself.
COMPOSER_SERVICE_SURFACE_PATHS = (
    Path("src/elspeth/web/composer/service.py"),
    Path("src/elspeth/web/composer/llm_response_parsing.py"),
    Path("src/elspeth/web/composer/_required_paths_validator.py"),
    Path("src/elspeth/web/composer/progress.py"),
)


@pytest.mark.parametrize("module_path", COMPOSER_SERVICE_SURFACE_PATHS, ids=lambda p: p.name)
def test_composer_service_impl_not_patched_at_module_tail(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
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
    assert hits == [], f"forbidden module-tail patches in {module_path}: {hits}"
