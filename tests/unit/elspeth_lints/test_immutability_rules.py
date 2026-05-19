"""Tests for immutability elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from elspeth_lints.core.protocols import RuleContext
from elspeth_lints.rules.immutability.freeze_guards import RULE as FREEZE_GUARDS_RULE
from elspeth_lints.rules.immutability.frozen_annotations import RULE as FROZEN_ANNOTATIONS_RULE


def test_freeze_guards_reports_mapping_proxy_wrap() -> None:
    findings = _analyze_freeze_guards(
        """
        class Example:
            def __post_init__(self):
                object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG1"]
    assert findings[0].file_path == "example.py"
    assert findings[0].line == 4
    assert "Bare MappingProxyType" in findings[0].message


def test_freeze_guards_reports_missing_freeze_fields() -> None:
    findings = _analyze_freeze_guards(
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            data: dict[str, object]
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG3"]
    assert "no __post_init__" in findings[0].message


def test_frozen_annotations_reports_mutable_container_annotation() -> None:
    findings = _analyze_frozen_annotations(
        """
        from dataclasses import dataclass

        @dataclass(frozen=True, slots=True)
        class Example:
            items: list[int]
        """
    )

    assert [finding.rule_id for finding in findings] == ["immutability.frozen_annotations"]
    assert findings[0].file_path == "example.py"
    assert findings[0].line == 6
    assert "Use Sequence/Mapping/tuple/frozenset" in findings[0].message


def test_frozen_annotations_ignores_mapping_annotation() -> None:
    findings = _analyze_frozen_annotations(
        """
        from collections.abc import Mapping
        from dataclasses import dataclass

        @dataclass(frozen=True, slots=True)
        class Example:
            data: Mapping[str, object]
        """
    )

    assert findings == []


def _analyze_freeze_guards(source: str) -> list[object]:
    tree = ast.parse(textwrap.dedent(source))
    return list(FREEZE_GUARDS_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))


def _analyze_frozen_annotations(source: str) -> list[object]:
    tree = ast.parse(textwrap.dedent(source))
    return list(FROZEN_ANNOTATIONS_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))
