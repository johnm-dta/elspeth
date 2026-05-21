"""Tests for tier-model allowlist fingerprint rotation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from scripts.cicd import rotate_tier_model_fingerprints as rotator


def _write_source(tmp_path: Path) -> Path:
    source_path = tmp_path / "src" / "elspeth" / "pkg" / "mod.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        "\n".join(
            [
                "class Example:",
                "    def outer(self) -> None:",
                "        def inner() -> None:",
                "            try:",
                "                pass",
                "            except Exception:",
                "                pass",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return source_path


def _write_allowlist(tmp_path: Path, allow_hits: list[dict[str, Any]]) -> Path:
    allowlist_dir = tmp_path / "config" / "cicd" / "enforce_tier_model"
    allowlist_dir.mkdir(parents=True)
    allowlist_path = allowlist_dir / "pkg.yaml"
    allowlist_path.write_text(yaml.safe_dump({"allow_hits": allow_hits}, sort_keys=False), encoding="utf-8")
    return allowlist_path


def _configure_tmp_repo(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setattr(rotator, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rotator, "ALLOWLIST_DIR", tmp_path / "config" / "cicd" / "enforce_tier_model")
    monkeypatch.setattr(rotator, "SRC_ROOT", tmp_path / "src" / "elspeth")


def test_find_enclosing_symbol_preserves_nested_context(tmp_path: Path) -> None:
    source_path = _write_source(tmp_path)

    assert rotator.find_enclosing_symbol(source_path, 6) == "Example:outer:inner"


def test_new_only_findings_are_reported_without_rewriting_allowlist_yaml(monkeypatch: Any, tmp_path: Path) -> None:
    _configure_tmp_repo(monkeypatch, tmp_path)
    _write_source(tmp_path)
    allowlist_path = _write_allowlist(tmp_path, [])

    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "file_path": "pkg/mod.py",
                "rule_id": "R4",
                "line": 6,
                "fingerprint": "abc123",
                "severity": "error",
                "message": "Broad exception caught without re-raise: except Exception:",
            }
        ],
    )

    assert rotator.main() == 1

    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    assert data["allow_hits"] == []


@pytest.mark.parametrize("rule_id", ["R7", "R8", "R9", "TC", "L1"])
def test_unmatched_new_policy_findings_do_not_become_todo_allowlist_entries(
    monkeypatch: Any,
    tmp_path: Path,
    rule_id: str,
) -> None:
    _configure_tmp_repo(monkeypatch, tmp_path)
    _write_source(tmp_path)
    allowlist_path = _write_allowlist(tmp_path, [])

    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "file_path": "pkg/mod.py",
                "rule_id": rule_id,
                "line": 6,
                "fingerprint": f"new-{rule_id}",
                "severity": "error",
                "message": f"{rule_id} violation",
            }
        ],
    )

    assert rotator.main() == 1

    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    assert data["allow_hits"] == []


def test_split_findings_includes_all_rotatable_tier_rule_ids() -> None:
    findings = [
        {"rule_id": "trust_tier.tier_model", "message": "Stale tier-model allowlist entry: pkg/mod.py:TC:_module_:fp=old"},
        *[
            {
                "file_path": "pkg/mod.py",
                "rule_id": rule_id,
                "line": 1,
                "fingerprint": f"fp-{rule_id}",
                "severity": "error",
                "message": f"{rule_id} violation",
            }
            for rule_id in ("R1", "R7", "R8", "R9", "TC", "L1")
        ],
        {
            "file_path": "pkg/mod.py",
            "rule_id": "R8",
            "line": 1,
            "fingerprint": "warning-fp",
            "severity": "warning",
            "message": "warning only",
        },
    ]

    stale, new = rotator.split_findings(findings)

    assert [entry["message"] for entry in stale] == ["Stale tier-model allowlist entry: pkg/mod.py:TC:_module_:fp=old"]
    assert {entry["rule_id"] for entry in new} == {"R1", "R7", "R8", "R9", "TC", "L1"}
    assert "warning-fp" not in {entry["fingerprint"] for entry in new}


def test_stale_entries_preserve_metadata_per_fingerprint(monkeypatch: Any, tmp_path: Path) -> None:
    _configure_tmp_repo(monkeypatch, tmp_path)
    _write_source(tmp_path)
    allowlist_path = _write_allowlist(
        tmp_path,
        [
            {
                "key": "pkg/mod.py:R8:Example:outer:inner:fp=aaa111",
                "owner": "first-owner",
                "reason": "first reason",
                "safety": "first safety",
                "expires": None,
            },
            {
                "key": "pkg/mod.py:R8:Example:outer:inner:fp=bbb222",
                "owner": "second-owner",
                "reason": "second reason",
                "safety": "second safety",
                "expires": "2026-12-31",
            },
        ],
    )

    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "rule_id": "trust_tier.tier_model",
                "message": "Stale tier-model allowlist entry: pkg/mod.py:R8:Example:outer:inner:fp=aaa111",
            },
            {
                "rule_id": "trust_tier.tier_model",
                "message": "Stale tier-model allowlist entry: pkg/mod.py:R8:Example:outer:inner:fp=bbb222",
            },
            {
                "file_path": "pkg/mod.py",
                "rule_id": "R8",
                "line": 6,
                "fingerprint": "new1",
                "severity": "error",
                "message": "Broad exception caught without re-raise: except Exception:",
            },
            {
                "file_path": "pkg/mod.py",
                "rule_id": "R8",
                "line": 6,
                "fingerprint": "new2",
                "severity": "error",
                "message": "Broad exception caught without re-raise: except Exception:",
            },
        ],
    )

    assert rotator.main() == 0

    data = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))
    assert data["allow_hits"] == [
        {
            "key": "pkg/mod.py:R8:Example:outer:inner:fp=new1",
            "owner": "first-owner",
            "reason": "first reason",
            "safety": "first safety",
            "expires": None,
        },
        {
            "key": "pkg/mod.py:R8:Example:outer:inner:fp=new2",
            "owner": "second-owner",
            "reason": "second reason",
            "safety": "second safety",
            "expires": "2026-12-31",
        },
    ]


def test_rotator_refuses_unmatched_new_without_partial_yaml_write(monkeypatch: Any, tmp_path: Path) -> None:
    """Mixed rotations plus genuinely new findings must leave YAML untouched."""
    _configure_tmp_repo(monkeypatch, tmp_path)
    _write_source(tmp_path)
    allowlist_path = _write_allowlist(
        tmp_path,
        [
            {
                "key": "pkg/mod.py:R8:Example:outer:inner:fp=aaa111",
                "owner": "first-owner",
                "reason": "first reason",
                "safety": "first safety",
                "expires": None,
            }
        ],
    )
    original = yaml.safe_load(allowlist_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        rotator,
        "run_tier_model",
        lambda: [
            {
                "rule_id": "trust_tier.tier_model",
                "message": "Stale tier-model allowlist entry: pkg/mod.py:R8:Example:outer:inner:fp=aaa111",
            },
            {
                "file_path": "pkg/mod.py",
                "rule_id": "R8",
                "line": 6,
                "fingerprint": "new1",
                "severity": "error",
                "message": "Broad exception caught without re-raise: except Exception:",
            },
            {
                "file_path": "pkg/mod.py",
                "rule_id": "R9",
                "line": 1,
                "fingerprint": "brandnew",
                "severity": "error",
                "message": "Unpaired violation",
            },
        ],
    )

    assert rotator.main() == 1
    assert yaml.safe_load(allowlist_path.read_text(encoding="utf-8")) == original
