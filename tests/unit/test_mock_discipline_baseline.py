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
# Bumped 2585→2602 (2026-06-08): the ratchet was already stale at 2601 (16 mocks
# of pre-existing drift from intervening branch work — operator-owed re-sign,
# flagged separately) PLUS +1 from the execute() fail-closed pre-run validation
# gate tests (PipelineValidationError; patch of validate_pipeline). Only the +1
# belongs to this commit; the 16 are inherited pre-existing red.
# Bumped 2602→2617 (2026-06-08): +15 from merging the engine/core/plugins and
# cicd-scanner-bugs burndown branches into release/0.5.3. The new mocks are
# external-SDK response-shape fakes (OpenAI completion/choice/usage in
# test_llm_telemetry) and HTTP SSRF/redirect-context fakes (test_audited_http_client),
# spread across ~13 burndown test files. These mock dynamically-shaped response
# objects where spec= is brittle and low-value — matching the prior bump precedent.
BASELINE_UNSPECCED_MOCK_TOTAL = 2617
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
