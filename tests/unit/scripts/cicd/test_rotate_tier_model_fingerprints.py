from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from scripts.cicd import rotate_tier_model_fingerprints as rotator


def _write_allowlist(root: Path, name: str, entries: list[dict[str, Any]]) -> Path:
    path = root / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"allow_hits": entries}, sort_keys=False), encoding="utf-8")
    return path


def _read_allowlist(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(data.get("allow_hits", []))


def test_main_writes_bounded_entry_for_new_only_violation(tmp_path: Path, monkeypatch: Any) -> None:
    allowlist_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    core_yaml = _write_allowlist(allowlist_dir, "core", [])
    monkeypatch.setattr(rotator, "ALLOWLIST_DIR", allowlist_dir)
    monkeypatch.setattr(rotator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rotator, "SRC_ROOT", tmp_path / "src" / "elspeth")
    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "rule_id": "R1",
                "severity": "error",
                "file_path": "core/runtime.py",
                "line": 12,
                "fingerprint": "abc123",
            }
        ],
    )
    monkeypatch.setattr(rotator, "find_enclosing_symbol", lambda _path, _line: "RuntimeBoundary:load")

    assert rotator.main() == 0

    entries = _read_allowlist(core_yaml)
    assert entries == [
        {
            "key": "core/runtime.py:R1:RuntimeBoundary:load:fp=abc123",
            "owner": "trust-tier-maintenance",
            "reason": (
                "ALLOWLIST-FRESH — no stale entry matched this finding during fingerprint rotation; review whether "
                "this is new debt that needs a source fix or a more specific justification"
            ),
            "safety": "Exact-fingerprint allowlist entry only; bounded expiry forces follow-up review instead of adding permanent debt",
            "expires": "2026-08-24",
        }
    ]


def test_main_preserves_metadata_for_duplicate_stale_symbol_rotations(tmp_path: Path, monkeypatch: Any) -> None:
    allowlist_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    core_yaml = _write_allowlist(
        allowlist_dir,
        "core",
        [
            {
                "key": "core/runtime.py:R1:RuntimeBoundary:load:fp=aaa111",
                "owner": "alice",
                "reason": "first justification",
                "safety": "first safety",
                "expires": "2026-06-01",
            },
            {
                "key": "core/runtime.py:R1:RuntimeBoundary:load:fp=bbb222",
                "owner": "bob",
                "reason": "second justification",
                "safety": "second safety",
                "expires": "2026-07-01",
            },
        ],
    )
    monkeypatch.setattr(rotator, "ALLOWLIST_DIR", allowlist_dir)
    monkeypatch.setattr(rotator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rotator, "SRC_ROOT", tmp_path / "src" / "elspeth")
    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "rule_id": "trust_tier.tier_model",
                "message": "Stale tier-model allowlist entry: core/runtime.py:R1:RuntimeBoundary:load:fp=aaa111",
            },
            {
                "rule_id": "trust_tier.tier_model",
                "message": "Stale tier-model allowlist entry: core/runtime.py:R1:RuntimeBoundary:load:fp=bbb222",
            },
            {
                "rule_id": "R1",
                "severity": "error",
                "file_path": "core/runtime.py",
                "line": 12,
                "fingerprint": "new111",
            },
            {
                "rule_id": "R1",
                "severity": "error",
                "file_path": "core/runtime.py",
                "line": 13,
                "fingerprint": "new222",
            },
        ],
    )
    monkeypatch.setattr(rotator, "find_enclosing_symbol", lambda _path, _line: "RuntimeBoundary:load")

    assert rotator.main() == 0

    entries = _read_allowlist(core_yaml)
    assert entries == [
        {
            "key": "core/runtime.py:R1:RuntimeBoundary:load:fp=new111",
            "owner": "alice",
            "reason": "first justification",
            "safety": "first safety",
            "expires": "2026-06-01",
        },
        {
            "key": "core/runtime.py:R1:RuntimeBoundary:load:fp=new222",
            "owner": "bob",
            "reason": "second justification",
            "safety": "second safety",
            "expires": "2026-07-01",
        },
    ]
