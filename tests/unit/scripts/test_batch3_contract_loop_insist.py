"""Tests for the batch3 contract-loop insist predicates."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_HARNESS_ROOT = Path(__file__).resolve().parents[3] / "scripts" / "skill_rgr"
sys.path.insert(0, str(_HARNESS_ROOT))
sys.path.insert(0, str(_HARNESS_ROOT / "scenarios"))

from scenarios import batch3_contract_loop_insist as insist  # noqa: E402


def _set_patch_targets(monkeypatch: pytest.MonkeyPatch, targets: list[str]) -> None:
    stub = SimpleNamespace(patches=[{"target": target} for target in targets])
    monkeypatch.setattr(insist, "_ensure_stub", lambda: stub)


def test_patched_source_first_detects_output_first_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_patch_targets(monkeypatch, ["output:main"])

    assert insist.patched_source_first([]) is True


def test_patched_source_first_ignores_source_patch_after_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_patch_targets(
        monkeypatch,
        [f"node:{insist.ContractLoopStub.INITIAL_NODE_ID}", "source"],
    )

    assert insist.patched_source_first([]) is False
