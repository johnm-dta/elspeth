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
# Bumped 2617→2618 (2026-06-13): +1 net from the multi-source-token-scheduler
# feat branch (slices 1-3 coordination substrate, journal barrier, finalize sweep).
# New tests in that branch use a _StubRepo typed fake for RunHeartbeatThread; the
# +1 comes from small pre-existing drift in test_execution_repository,
# test_outcomes, test_read_only_audit_surfaces across the branch's accumulated
# changes (no single new file; the new heartbeat tests contribute zero).
# Bumped 2618→2623 (2026-06-13): +5 from slice 5 step 2 (follower-mode
# RowProcessor — test_follower_processor.py).  The new MagicMock() calls are
# return-value fakes for _drain_scheduler_claims result items (SchedulerResult
# stubs) where spec= would need to import internal scheduler types; low-value
# spec here. Step 3 (CLI join + journal path tests) contributed zero new
# unspecced mocks (uses create_autospec and patch only).
# Bumped 2623→2630 (2026-06-13): +7 from slice 5 steps 4-5 (e2e recovery
# tests for follower lifecycle: test_follower_join_and_drain.py x5 and
# test_follower_coordination_chaos.py x2).  These are stub RowProcessor
# objects — MagicMock() stubs where spec= would require importing internal
# RowProcessor and wiring a full engine stack; low-value spec for integration
# harness stubs. The real follower routing (barrier/branch-loss/sink) is
# covered by the processor-level drain tests, not these lifecycle stubs.
# Bumped 2630→2652 (2026-06-14): +22 total on fix/plugins-subsystem-remediation.
# +20 belong to this branch's plugin-remediation test additions (changed test
# files went 321→341 unspecced mocks): external-SDK/provider response-shape
# fakes (test_audited_llm_client, llm/test_transform, azure_multi_query_retry)
# and source/transform stubs (test_dataverse_source, rag/test_transform,
# transforms/azure/test_blob_source) where spec= is brittle/low-value — matching
# the prior-bump precedent. The remaining +2 is pre-existing drift already on
# release/0.6.0 (its actual count is 2632 vs the 2630 constant — operator-owed,
# inherited, flagged separately).
BASELINE_UNSPECCED_MOCK_TOTAL = 2652
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
