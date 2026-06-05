"""Tests for immutability elspeth-lints rules."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from elspeth_lints.core.protocols import Finding, RuleContext
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


def test_freeze_guards_reports_mapping_proxy_attribute_form() -> None:
    findings = _analyze_freeze_guards(
        """
        class Example:
            def __post_init__(self):
                object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG1"]


def test_freeze_guards_reports_multiple_mapping_proxy_wraps() -> None:
    findings = _analyze_freeze_guards(
        """
        class Example:
            def __post_init__(self):
                object.__setattr__(self, "a", MappingProxyType(dict(self.a)))
                object.__setattr__(self, "b", MappingProxyType(dict(self.b)))
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG1", "FG1"]


def test_freeze_guards_ignores_mapping_proxy_outside_post_init() -> None:
    findings = _analyze_freeze_guards(
        """
        class Example:
            def other(self):
                return MappingProxyType(dict(self.data))
        """
    )

    assert findings == []


def test_freeze_guards_reports_isinstance_self_guard() -> None:
    findings = _analyze_freeze_guards(
        """
        class Example:
            def __post_init__(self):
                if isinstance(self.data, (dict, tuple)):
                    pass
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG2"]
    assert "dict" in findings[0].message
    assert "tuple" in findings[0].message


@pytest.mark.parametrize(
    "source",
    [
        """
        class Example:
            def __post_init__(self):
                if isinstance(other, dict):
                    pass
        """,
        """
        class Example:
            def __post_init__(self):
                if isinstance(self.name, str):
                    pass
        """,
        """
        class Example:
            def validate(self):
                if isinstance(self.data, dict):
                    pass
        """,
    ],
)
def test_freeze_guards_ignores_non_freeze_guard_isinstance_patterns(source: str) -> None:
    assert _analyze_freeze_guards(source) == []


def test_freeze_guards_handles_post_init_scope_edges() -> None:
    nested_function = _analyze_freeze_guards(
        """
        class Example:
            def __post_init__(self):
                def helper():
                    MappingProxyType(x)
                helper()
        """
    )
    module_function = _analyze_freeze_guards(
        """
        def __post_init__(self):
            MappingProxyType(dict(self.data))
        """
    )
    nested_class = _analyze_freeze_guards(
        """
        class Outer:
            class Inner:
                def __post_init__(self):
                    MappingProxyType(dict(self.data))
        """
    )
    async_post_init = _analyze_freeze_guards(
        """
        class Example:
            async def __post_init__(self):
                MappingProxyType(dict(self.data))
        """
    )

    assert nested_function == []
    assert module_function == []
    assert [finding.rule_id for finding in nested_class] == ["FG1"]
    assert [finding.rule_id for finding in async_post_init] == ["FG1"]


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


def test_freeze_guards_reports_multiple_missing_container_fields() -> None:
    findings = _analyze_freeze_guards(
        """
        @dataclass(frozen=True)
        class Example:
            data: dict
            items: list
            keys: set
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG3"]
    assert "data" in findings[0].message
    assert "items" in findings[0].message
    assert "keys" in findings[0].message


def test_freeze_guards_reports_post_init_without_freeze_call() -> None:
    findings = _analyze_freeze_guards(
        """
        @dataclass(frozen=True)
        class Example:
            data: dict
            def __post_init__(self):
                validate(self.data)
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG3"]
    assert "lacks freeze_fields/deep_freeze" in findings[0].message


@pytest.mark.parametrize(
    "annotation",
    ["dict", "list", "set", "Dict[str, int]", "List[str]", "Mapping[str, object]", "Sequence[str]", "dict | None", '"dict[str, int]"'],
)
def test_freeze_guards_reports_container_annotation_variants(annotation: str) -> None:
    findings = _analyze_freeze_guards(
        f"""
        @dataclass(frozen=True)
        class Example:
            data: {annotation}
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG3"]


@pytest.mark.parametrize(
    "post_init_body",
    [
        'freeze_fields(self, "data")',
        'object.__setattr__(self, "data", deep_freeze(self.data))',
        'freeze.freeze_fields(self, "data")',
    ],
)
def test_freeze_guards_accepts_deep_freeze_patterns(post_init_body: str) -> None:
    findings = _analyze_freeze_guards(
        f"""
        @dataclass(frozen=True)
        class Example:
            data: dict
            def __post_init__(self):
                {post_init_body}
        """
    )

    assert findings == []


@pytest.mark.parametrize(
    "source",
    [
        """
        @dataclass
        class Example:
            data: dict
        """,
        """
        @dataclass(frozen=False)
        class Example:
            data: dict
        """,
        """
        class Example:
            data: dict
        """,
        """
        @dataclass(frozen=True)
        class Example:
            name: str
            count: int
        """,
    ],
)
def test_freeze_guards_ignores_non_frozen_or_scalar_only_classes(source: str) -> None:
    assert _analyze_freeze_guards(source) == []


def test_freeze_guards_detects_decorator_variants_and_nested_classes() -> None:
    dataclasses_module = _analyze_freeze_guards(
        """
        @dataclasses.dataclass(frozen=True)
        class Example:
            data: dict
        """
    )
    multiple_decorators = _analyze_freeze_guards(
        """
        @outer
        @dataclass(frozen=True)
        class Example:
            data: dict
        """
    )
    nested = _analyze_freeze_guards(
        """
        @dataclass(frozen=True)
        class Outer:
            name: str

            @dataclass(frozen=True)
            class Inner:
                data: dict
        """
    )

    assert [finding.rule_id for finding in dataclasses_module] == ["FG3"]
    assert [finding.rule_id for finding in multiple_decorators] == ["FG3"]
    assert [finding.rule_id for finding in nested] == ["FG3"]
    assert "Inner" in nested[0].message


def test_freeze_guards_mapping_proxy_counts_as_freeze_for_fg3_but_still_reports_fg1() -> None:
    findings = _analyze_freeze_guards(
        """
        @dataclass(frozen=True)
        class Example:
            data: dict
            def __post_init__(self):
                object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        """
    )

    assert [finding.rule_id for finding in findings] == ["FG1"]


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


def test_frozen_annotations_reports_frozen_true_without_slots() -> None:
    findings = _analyze_frozen_annotations(
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            items: list[int]
        """
    )

    assert [finding.rule_id for finding in findings] == ["immutability.frozen_annotations"]


@pytest.mark.parametrize("annotation", ["list[int]", "dict[str, int]", "set[str]", "list[int] | None"])
def test_frozen_annotations_reports_mutable_annotation_variants(annotation: str) -> None:
    findings = _analyze_frozen_annotations(
        f"""
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            items: {annotation}
        """
    )

    assert [finding.rule_id for finding in findings] == ["immutability.frozen_annotations"]


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


@pytest.mark.parametrize(
    "annotation",
    ["Sequence[int]", "Mapping[str, object]", "tuple[int, ...]"],
)
def test_frozen_annotations_ignores_immutable_annotation_variants(annotation: str) -> None:
    findings = _analyze_frozen_annotations(
        f"""
        from collections.abc import Mapping, Sequence
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            data: {annotation}
        """
    )

    assert findings == []


@pytest.mark.parametrize(
    "decorator",
    ["@dataclass", "@dataclass(frozen=False)"],
)
def test_frozen_annotations_ignores_non_frozen_dataclasses(decorator: str) -> None:
    findings = _analyze_frozen_annotations(
        f"""
        from dataclasses import dataclass

        {decorator}
        class Example:
            items: list[int]
        """
    )

    assert findings == []


def test_frozen_annotations_reports_multiple_fields() -> None:
    findings = _analyze_frozen_annotations(
        """
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            items: list[int]
            mapping: dict[str, str]
            unique: set[float]
        """
    )

    assert [finding.line for finding in findings] == [6, 7, 8]


def test_frozen_annotations_handles_future_annotations() -> None:
    mutable = _analyze_frozen_annotations(
        """
        from __future__ import annotations
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            items: list[int]
        """
    )
    clean = _analyze_frozen_annotations(
        """
        from __future__ import annotations
        from collections.abc import Sequence
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Example:
            items: Sequence[int]
        """
    )

    assert [finding.rule_id for finding in mutable] == ["immutability.frozen_annotations"]
    assert clean == []


def test_frozen_annotations_uses_core_loader() -> None:
    """After Plan A Task 6 the rule must not define its own loader."""
    from elspeth_lints.rules.immutability.frozen_annotations import rule as r

    assert "_load_allowlist" not in vars(r), "degenerate loader must be removed"
    assert "_list_value" not in vars(r), "private helper must be removed"
    assert "_mapping_value" not in vars(r), "private helper must be removed"


def test_existing_yaml_loads_with_core_loader() -> None:
    """The migrated YAML must parse under the core allow_hits schema."""
    from elspeth_lints.core.allowlist import load_allowlist

    path = Path("config/cicd/enforce_frozen_annotations/existing.yaml")
    result = load_allowlist(path, valid_rule_ids={"immutability.frozen_annotations"})
    assert len(result.entries) == 9, f"expected 9 live entries, got {len(result.entries)}"


def _analyze_freeze_guards(source: str) -> list[Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return list(FREEZE_GUARDS_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))


def _analyze_frozen_annotations(source: str) -> list[Finding]:
    tree = ast.parse(textwrap.dedent(source))
    return list(FROZEN_ANNOTATIONS_RULE.analyze(tree, Path("example.py"), RuleContext(root=Path("."))))
