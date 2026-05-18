"""Tests for shared CI scanner allowlist framework."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from scripts.cicd._framework.allowlist import (
    PerFileAllowlist,
    PerFileRule,
    load_per_file_allowlist,
    parse_per_file_rules,
)


def test_per_file_rule_matches_rule_and_glob() -> None:
    rule = PerFileRule(pattern="core/*.py", rules=["FG1"], reason="test", expires=None)

    assert rule.matches("core/config.py", "FG1") is True
    assert rule.matches("core/config.py", "FG2") is False
    assert rule.matches("plugins/config.py", "FG1") is False


def test_allowlist_tracks_unused_expired_and_exceeded_rules() -> None:
    yesterday = datetime.now(UTC).date() - timedelta(days=1)
    active = PerFileRule(pattern="core/*.py", rules=["FG1"], reason="active", expires=None, max_hits=1)
    expired = PerFileRule(pattern="plugins/*.py", rules=["FG1"], reason="expired", expires=yesterday)
    unused = PerFileRule(pattern="engine/*.py", rules=["FG1"], reason="unused", expires=None)
    allowlist = PerFileAllowlist(per_file_rules=[active, expired, unused])

    assert allowlist.match("core/config.py", "FG1") is active
    assert allowlist.match("core/config.py", "FG1") is active

    assert allowlist.get_unused_rules() == [expired, unused]
    assert allowlist.get_expired_rules() == [expired]
    assert allowlist.get_exceeded_rules() == [active]


def test_parse_per_file_rules_rejects_unknown_rule(capsys: pytest.CaptureFixture[str]) -> None:
    data = {"per_file_rules": [{"pattern": "core/*.py", "rules": ["NOPE"], "reason": "test"}]}

    with pytest.raises(SystemExit) as exc_info:
        parse_per_file_rules(data, valid_rule_ids={"FG1"}, source_file="rules.yaml")

    stderr = capsys.readouterr().err
    assert exc_info.value.code == 1
    assert "unknown rule ID" in stderr
    assert "rules.yaml" in stderr


def test_parse_per_file_rules_rejects_non_numeric_max_hits(capsys: pytest.CaptureFixture[str]) -> None:
    data = {"per_file_rules": [{"pattern": "core/*.py", "rules": ["FG1"], "reason": "test", "max_hits": "many"}]}

    with pytest.raises(SystemExit) as exc_info:
        parse_per_file_rules(data, valid_rule_ids={"FG1"}, source_file="rules.yaml")

    assert exc_info.value.code == 1
    assert "non-numeric max_hits" in capsys.readouterr().err


def test_load_directory_merges_defaults_and_yaml_files(tmp_path: Path) -> None:
    (tmp_path / "_defaults.yaml").write_text("defaults:\n  fail_on_stale: false\n")
    (tmp_path / "rules.yaml").write_text(
        yaml.dump({"per_file_rules": [{"pattern": "core/*.py", "rules": ["FG1"], "reason": "test", "expires": None, "max_hits": 2}]})
    )

    allowlist = load_per_file_allowlist(tmp_path, valid_rule_ids={"FG1"})

    assert allowlist.fail_on_stale is False
    assert len(allowlist.per_file_rules) == 1
    assert allowlist.per_file_rules[0].max_hits == 2


def test_load_single_file_and_missing_path(tmp_path: Path) -> None:
    allowlist_file = tmp_path / "allowlist.yaml"
    allowlist_file.write_text(
        dedent("""\
        defaults:
          fail_on_stale: false
        per_file_rules:
          - pattern: core/*.py
            rules: [FG1]
            reason: test
        """)
    )

    loaded = load_per_file_allowlist(allowlist_file, valid_rule_ids={"FG1"})
    missing = load_per_file_allowlist(tmp_path / "missing.yaml", valid_rule_ids={"FG1"})

    assert loaded.fail_on_stale is False
    assert len(loaded.per_file_rules) == 1
    assert missing.fail_on_stale is True
    assert missing.per_file_rules == []
