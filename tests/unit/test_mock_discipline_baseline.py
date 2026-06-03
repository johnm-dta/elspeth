"""Baseline gate for unspecced test mocks.

The suite still has many legacy ``Mock()`` / ``MagicMock()`` calls without
``spec``/``spec_set``/``autospec``. This gate makes that debt visible and
prevents the count from increasing while focused cleanup replaces legacy mocks
with fakes or specced mocks.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
# Bumped 2559→2585 (2026-06-01): the cicd-judge campaign (judge transport +
# scope-fingerprint binding) added external-SDK response-shape MagicMocks
# (OpenAI/OpenRouter completion/choice/usage fakes) where spec= is brittle and
# low-value. Pay-down of the judge-test mocks is tracked separately; this bump
# matches the prior explicit-bump precedent (9daf156e1: 2554→2559).
BASELINE_UNSPECCED_MOCK_TOTAL = 2585
MOCK_NAMES = frozenset({"Mock", "MagicMock"})
SPEC_KEYWORDS = frozenset({"spec", "spec_set", "autospec", "wraps"})


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_specced_mock_call(node: ast.Call) -> bool:
    return any(keyword.arg in SPEC_KEYWORDS for keyword in node.keywords)


def test_unspecced_mock_baseline_does_not_increase() -> None:
    unspecced_mock_calls: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in MOCK_NAMES:
                continue
            if _is_specced_mock_call(node):
                continue
            rel_path = path.relative_to(REPO_ROOT)
            unspecced_mock_calls.append(f"{rel_path}:{node.lineno}:{node.col_offset}")

    assert len(unspecced_mock_calls) <= BASELINE_UNSPECCED_MOCK_TOTAL
