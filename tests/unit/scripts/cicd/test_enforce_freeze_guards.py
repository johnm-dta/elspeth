"""Tests for enforce_freeze_guards.py CI scanner.

Tests the FreezeGuardVisitor AST analysis, allowlist matching, and scan
integration. Uses inline source strings via textwrap.dedent for hermetic
tests — no filesystem fixtures needed for visitor tests.

Follows the pattern established by test_enforce_frozen_annotations.py.
"""

from __future__ import annotations

import argparse
import ast
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from scripts.cicd.enforce_freeze_guards import (
    Allowlist,
    Finding,
    FreezeGuardVisitor,
    PerFileRule,
    load_allowlist,
    run_check,
    scan_file,
)


def _scan(source: str, file_path: str = "test.py") -> list[Finding]:
    """Run the freeze guard visitor on a source string and return findings."""
    source = textwrap.dedent(source)
    tree = ast.parse(source)
    source_lines = source.splitlines()
    visitor = FreezeGuardVisitor(file_path, source_lines)
    visitor.visit(tree)
    return visitor.findings


# =============================================================================
# FG1: Bare MappingProxyType detection
# =============================================================================


class TestFG1Detection:
    """FG1 detects bare MappingProxyType wraps in __post_init__."""

    def test_name_form_detected(self) -> None:
        """MappingProxyType(dict(self.x)) using Name import form."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    object.__setattr__(self, "x", MappingProxyType(dict(self.x)))
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG1"
        assert findings[0].symbol_context == ("Foo", "__post_init__")

    def test_attribute_form_detected(self) -> None:
        """types.MappingProxyType(...) using Attribute access form."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    object.__setattr__(self, "x", types.MappingProxyType(dict(self.x)))
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG1"

    def test_not_flagged_outside_post_init(self) -> None:
        """MappingProxyType in other methods is not flagged."""
        findings = _scan("""
            class Foo:
                def some_method(self):
                    return MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 0

    def test_multiple_wraps_detected(self) -> None:
        """Multiple MappingProxyType calls each produce a finding."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    object.__setattr__(self, "a", MappingProxyType(dict(self.a)))
                    object.__setattr__(self, "b", MappingProxyType(dict(self.b)))
        """)
        assert len(findings) == 2
        assert all(f.rule_id == "FG1" for f in findings)


# =============================================================================
# FG2: isinstance freeze guard detection
# =============================================================================


class TestFG2Detection:
    """FG2 detects isinstance type guards in __post_init__."""

    def test_single_type_detected(self) -> None:
        """isinstance(self.x, dict) with single freeze-guard type."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(self.data, dict):
                        pass
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG2"
        assert "dict" in findings[0].message

    def test_tuple_of_types_detected(self) -> None:
        """isinstance(self.x, (dict, tuple)) with multiple freeze-guard types."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(self.data, (dict, tuple)):
                        pass
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG2"
        assert "dict" in findings[0].message
        assert "tuple" in findings[0].message

    def test_non_self_attribute_not_flagged(self) -> None:
        """isinstance(other_var, dict) without self.x is not flagged."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(other_var, dict):
                        pass
        """)
        assert len(findings) == 0

    def test_non_freeze_type_not_flagged(self) -> None:
        """isinstance(self.x, str) with non-freeze-guard type is not flagged."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(self.name, str):
                        pass
        """)
        assert len(findings) == 0

    def test_mapping_proxy_type_in_isinstance(self) -> None:
        """isinstance(self.x, MappingProxyType) is a freeze-guard type."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(self.data, MappingProxyType):
                        pass
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG2"
        assert "MappingProxyType" in findings[0].message

    def test_mapping_abstract_type_in_isinstance(self) -> None:
        """isinstance(self.x, Mapping) is a freeze-guard type."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    if isinstance(self.data, Mapping):
                        pass
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG2"

    def test_not_flagged_outside_post_init(self) -> None:
        """isinstance in other methods is not flagged."""
        findings = _scan("""
            class Foo:
                def validate(self):
                    if isinstance(self.data, dict):
                        pass
        """)
        assert len(findings) == 0


# =============================================================================
# Scope handling
# =============================================================================


class TestScopeHandling:
    """Scope tracking for __post_init__ detection."""

    def test_nested_function_not_flagged(self) -> None:
        """Nested def inside __post_init__ exits the detection scope."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    def helper():
                        MappingProxyType(x)
                    helper()
        """)
        assert len(findings) == 0

    def test_module_level_post_init_not_flagged(self) -> None:
        """Module-level function named __post_init__ is not a dataclass method."""
        findings = _scan("""
            def __post_init__(self):
                MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 0

    def test_nested_post_init_inside_function_not_flagged(self) -> None:
        """__post_init__ nested inside a module-level function is not a dataclass method.

        The parent scope is a function, not a class — even though symbol_stack
        has multiple entries, the immediate enclosing scope must be a ClassDef.
        """
        findings = _scan("""
            def outer():
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 0

    def test_nested_class_own_post_init_flagged(self) -> None:
        """Nested class with its own __post_init__ is flagged independently."""
        findings = _scan("""
            class Outer:
                def __post_init__(self):
                    pass

                class Inner:
                    def __post_init__(self):
                        MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 1
        assert findings[0].symbol_context == ("Outer", "Inner", "__post_init__")

    def test_async_post_init_flagged(self) -> None:
        """async def __post_init__ is handled by visit_AsyncFunctionDef alias."""
        findings = _scan("""
            class Foo:
                async def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG1"

    def test_sibling_method_not_flagged(self) -> None:
        """Other methods in the same class are not in __post_init__ scope."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    pass

                def other_method(self):
                    MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 0

    def test_direct_post_init_body_flagged(self) -> None:
        """Code directly in __post_init__ body is flagged (not just nested)."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
                    if isinstance(self.y, dict):
                        pass
        """)
        assert len(findings) == 2
        assert {f.rule_id for f in findings} == {"FG1", "FG2"}

    def test_nested_class_does_not_inherit_outer_scope(self) -> None:
        """Class defined inside __post_init__ does not inherit detection scope."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    class Helper:
                        x = MappingProxyType({})
        """)
        assert len(findings) == 0


# =============================================================================
# FG3: Missing freeze guard detection
# =============================================================================


class TestFG3Detection:
    """FG3 detects frozen dataclasses with container fields lacking freeze guards."""

    # -------------------------------------------------------------------------
    # Basic detection: no __post_init__
    # -------------------------------------------------------------------------

    def test_no_post_init_at_all(self) -> None:
        """Frozen dataclass with container field and no __post_init__ is flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"
        assert "no __post_init__" in findings[0].message
        assert "['data']" in findings[0].message

    def test_multiple_container_fields_no_post_init(self) -> None:
        """All container fields are listed when __post_init__ is missing."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict
                items: list
                keys: set
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"
        assert "data" in findings[0].message
        assert "items" in findings[0].message
        assert "keys" in findings[0].message

    # -------------------------------------------------------------------------
    # Basic detection: __post_init__ without freeze calls
    # -------------------------------------------------------------------------

    def test_post_init_without_freeze_call(self) -> None:
        """__post_init__ that doesn't call freeze_fields/deep_freeze is flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict
                def __post_init__(self):
                    if not self.data:
                        raise ValueError("data required")
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"
        assert "lacks freeze_fields/deep_freeze" in findings[0].message

    def test_post_init_with_unrelated_calls(self) -> None:
        """__post_init__ with other calls but no freeze calls is flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict
                def __post_init__(self):
                    print(self.data)
                    validate(self.data)
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    # -------------------------------------------------------------------------
    # Container type variations
    # -------------------------------------------------------------------------

    def test_list_annotation(self) -> None:
        """list annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                items: list
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_set_annotation(self) -> None:
        """set annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                tags: set
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_generic_dict_annotation(self) -> None:
        """Dict[str, int] generic annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: Dict[str, int]
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_generic_list_annotation(self) -> None:
        """List[str] generic annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                items: List[str]
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_mapping_annotation(self) -> None:
        """Mapping[str, Any] annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: Mapping[str, Any]
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_sequence_annotation(self) -> None:
        """Sequence[str] annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                items: Sequence[str]
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_lowercase_generic_annotation(self) -> None:
        """PEP 585 style dict[str, int] annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict[str, int]
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_union_with_none_annotation(self) -> None:
        """dict | None union annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: dict | None
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_string_annotation(self) -> None:
        """String forward reference annotation triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Bad:
                data: "dict[str, int]"
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    # -------------------------------------------------------------------------
    # Correct patterns that should NOT be flagged
    # -------------------------------------------------------------------------

    def test_freeze_fields_not_flagged(self) -> None:
        """Frozen dataclass with freeze_fields() is not flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                def __post_init__(self):
                    freeze_fields(self, "data")
        """)
        assert len(findings) == 0

    def test_deep_freeze_not_flagged(self) -> None:
        """Frozen dataclass with deep_freeze() is not flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                def __post_init__(self):
                    object.__setattr__(self, "data", deep_freeze(self.data))
        """)
        assert len(findings) == 0

    def test_module_qualified_freeze_fields(self) -> None:
        """freeze.freeze_fields() qualified call is recognized."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                def __post_init__(self):
                    freeze.freeze_fields(self, "data")
        """)
        assert len(findings) == 0

    def test_scalar_only_not_flagged(self) -> None:
        """Frozen dataclass with only scalar fields is not flagged."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                name: str
                count: int
                active: bool
                ratio: float
        """)
        assert len(findings) == 0

    def test_non_frozen_dataclass_not_flagged(self) -> None:
        """Non-frozen dataclass with containers is not flagged."""
        findings = _scan("""
            @dataclass
            class Good:
                data: dict
        """)
        assert len(findings) == 0

    def test_frozen_false_not_flagged(self) -> None:
        """@dataclass(frozen=False) with containers is not flagged."""
        findings = _scan("""
            @dataclass(frozen=False)
            class Good:
                data: dict
        """)
        assert len(findings) == 0

    def test_non_dataclass_class_not_flagged(self) -> None:
        """Regular class (not dataclass) with container fields is not flagged."""
        findings = _scan("""
            class Good:
                data: dict
        """)
        assert len(findings) == 0

    # -------------------------------------------------------------------------
    # Edge cases and decorator variations
    # -------------------------------------------------------------------------

    def test_dataclass_with_slots(self) -> None:
        """@dataclass(frozen=True, slots=True) triggers FG3."""
        findings = _scan("""
            @dataclass(frozen=True, slots=True)
            class Bad:
                data: dict
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_multiple_decorators(self) -> None:
        """Class with multiple decorators including frozen dataclass is flagged."""
        findings = _scan("""
            @some_decorator
            @dataclass(frozen=True)
            @another_decorator
            class Bad:
                data: dict
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_dataclasses_module_form(self) -> None:
        """dataclasses.dataclass(frozen=True) form is detected."""
        findings = _scan("""
            @dataclasses.dataclass(frozen=True)
            class Bad:
                data: dict
        """)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG3"

    def test_nested_class_independent(self) -> None:
        """Nested frozen dataclass is checked independently."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Outer:
                name: str

                @dataclass(frozen=True)
                class Inner:
                    data: dict
        """)
        # Only Inner has container field without freeze guard
        assert len(findings) == 1
        assert "Inner" in findings[0].message

    def test_async_post_init(self) -> None:
        """async def __post_init__ with freeze_fields is recognized."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                async def __post_init__(self):
                    freeze_fields(self, "data")
        """)
        assert len(findings) == 0

    def test_symbol_context_in_finding(self) -> None:
        """Finding includes class name in symbol_context."""
        findings = _scan("""
            @dataclass(frozen=True)
            class MyDataClass:
                data: dict
        """)
        assert len(findings) == 1
        assert findings[0].symbol_context == ("MyDataClass",)

    def test_mapping_proxy_type_recognized_as_freeze(self) -> None:
        """MappingProxyType calls are recognized as a freeze mechanism (FG1 handles misuse)."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                def __post_init__(self):
                    object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        """)
        # FG3 should NOT fire — MappingProxyType IS a freeze mechanism.
        # FG1 will flag the shallow wrap separately if inappropriate.
        fg3_findings = [f for f in findings if f.rule_id == "FG3"]
        assert len(fg3_findings) == 0

    def test_module_qualified_mapping_proxy_recognized(self) -> None:
        """types.MappingProxyType is also recognized as a freeze mechanism."""
        findings = _scan("""
            @dataclass(frozen=True)
            class Good:
                data: dict
                def __post_init__(self):
                    object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))
        """)
        fg3_findings = [f for f in findings if f.rule_id == "FG3"]
        assert len(fg3_findings) == 0


# =============================================================================
# Allowlist
# =============================================================================


class TestAllowlist:
    """Allowlist matching, staleness, and loading."""

    def test_suppresses_finding(self) -> None:
        """Allowlisted finding is matched and can be filtered out."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 1

        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="test.py",
                    rules=["FG1"],
                    reason="test",
                    expires=None,
                ),
            ]
        )
        # End-to-end: match and filter
        violations = [f for f in findings if allowlist.match(f) is None]
        assert violations == []

    def test_non_matching_pattern_not_suppressed(self) -> None:
        """Finding with non-matching pattern is not suppressed."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="other.py",
                    rules=["FG1"],
                    reason="test",
                    expires=None,
                ),
            ]
        )
        match = allowlist.match(findings[0])
        assert match is None

    def test_max_hits_exceeded_reported(self) -> None:
        """Rules with matched_count > max_hits are reported as exceeded."""
        rule = PerFileRule(
            pattern="test.py",
            rules=["FG1"],
            reason="test",
            expires=None,
            max_hits=1,
        )
        rule.matched_count = 2
        allowlist = Allowlist(per_file_rules=[rule])
        exceeded = allowlist.get_exceeded_rules()
        assert len(exceeded) == 1

    def test_max_hits_exact_boundary_not_exceeded(self) -> None:
        """matched_count == max_hits is NOT exceeded (only > is exceeded)."""
        rule = PerFileRule(
            pattern="test.py",
            rules=["FG1"],
            reason="test",
            expires=None,
            max_hits=1,
        )
        rule.matched_count = 1
        allowlist = Allowlist(per_file_rules=[rule])
        exceeded = allowlist.get_exceeded_rules()
        assert len(exceeded) == 0

    def test_max_hits_via_match_flow(self) -> None:
        """max_hits exceeded detected through the match() call flow."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.a))
                    MappingProxyType(dict(self.b))
        """)
        assert len(findings) == 2

        allowlist = Allowlist(
            per_file_rules=[
                PerFileRule(
                    pattern="test.py",
                    rules=["FG1"],
                    reason="test",
                    expires=None,
                    max_hits=1,
                ),
            ]
        )
        for f in findings:
            allowlist.match(f)
        exceeded = allowlist.get_exceeded_rules()
        assert len(exceeded) == 1

    def test_glob_pattern_matching(self) -> None:
        """Glob patterns in allowlist match file paths correctly.

        fnmatch.fnmatch matches * across path separators on POSIX.
        """
        rule = PerFileRule(
            pattern="contracts/*.py",
            rules=["FG1"],
            reason="test",
            expires=None,
        )
        assert rule.matches("contracts/diversion.py", "FG1") is True
        assert rule.matches("plugins/transform.py", "FG1") is False

    def test_glob_wrong_rule_id_not_matched(self) -> None:
        """Matching pattern but wrong rule_id does not match."""
        rule = PerFileRule(
            pattern="contracts/*.py",
            rules=["FG1"],
            reason="test",
            expires=None,
        )
        assert rule.matches("contracts/diversion.py", "FG2") is False

    def test_unused_rule_reported(self) -> None:
        """Rules with zero matches are reported as unused."""
        rule = PerFileRule(
            pattern="nonexistent.py",
            rules=["FG1"],
            reason="test",
            expires=None,
        )
        allowlist = Allowlist(per_file_rules=[rule])
        unused = allowlist.get_unused_rules()
        assert len(unused) == 1

    def test_expired_rule_reported(self) -> None:
        """Rules past their expiry date are reported."""
        yesterday = (datetime.now(UTC) - timedelta(days=1)).date()
        rule = PerFileRule(
            pattern="test.py",
            rules=["FG1"],
            reason="test",
            expires=yesterday,
        )
        allowlist = Allowlist(per_file_rules=[rule])
        expired = allowlist.get_expired_rules()
        assert len(expired) == 1

    def test_non_expired_rule_not_reported(self) -> None:
        """Rules before their expiry date are not reported."""
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
        rule = PerFileRule(
            pattern="test.py",
            rules=["FG1"],
            reason="test",
            expires=tomorrow,
        )
        allowlist = Allowlist(per_file_rules=[rule])
        expired = allowlist.get_expired_rules()
        assert len(expired) == 0

    def test_directory_loading(self, tmp_path: Path) -> None:
        """load_allowlist reads a directory of YAML files with _defaults.yaml."""
        defaults = tmp_path / "_defaults.yaml"
        defaults.write_text("defaults:\n  fail_on_stale: false\n")

        rules_file = tmp_path / "test_rules.yaml"
        rules_file.write_text(
            yaml.dump(
                {
                    "per_file_rules": [
                        {
                            "pattern": "foo.py",
                            "rules": ["FG1"],
                            "reason": "test",
                        }
                    ]
                }
            )
        )

        allowlist = load_allowlist(tmp_path)
        assert allowlist.fail_on_stale is False
        assert len(allowlist.per_file_rules) == 1
        assert allowlist.per_file_rules[0].pattern == "foo.py"

    def test_unknown_rule_id_exits(self, tmp_path: Path) -> None:
        """Unknown rule IDs in allowlist cause sys.exit(1)."""
        rules_file = tmp_path / "bad.yaml"
        rules_file.write_text(
            yaml.dump(
                {
                    "per_file_rules": [
                        {
                            "pattern": "foo.py",
                            "rules": ["NONEXISTENT"],
                            "reason": "bad",
                        }
                    ]
                }
            )
        )

        with pytest.raises(SystemExit) as exc_info:
            load_allowlist(tmp_path)
        assert exc_info.value.code == 1

    def test_single_file_loading(self, tmp_path: Path) -> None:
        """load_allowlist reads a single YAML file (non-directory)."""
        rules_file = tmp_path / "allowlist.yaml"
        rules_file.write_text(
            yaml.dump(
                {
                    "defaults": {"fail_on_stale": False},
                    "per_file_rules": [
                        {
                            "pattern": "bar.py",
                            "rules": ["FG2"],
                            "reason": "test",
                        }
                    ],
                }
            )
        )

        allowlist = load_allowlist(rules_file)
        assert allowlist.fail_on_stale is False
        assert len(allowlist.per_file_rules) == 1
        assert allowlist.per_file_rules[0].rules == ["FG2"]

    def test_directory_loading_without_defaults(self, tmp_path: Path) -> None:
        """load_allowlist from directory works when _defaults.yaml is absent."""
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(yaml.dump({"per_file_rules": [{"pattern": "x.py", "rules": ["FG1"], "reason": "test"}]}))

        allowlist = load_allowlist(tmp_path)
        # Default fail_on_stale is True when _defaults.yaml is absent
        assert allowlist.fail_on_stale is True
        assert len(allowlist.per_file_rules) == 1

    def test_malformed_expires_date_produces_none(self, tmp_path: Path) -> None:
        """Invalid expires date format is warned but does not crash."""
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text(
            yaml.dump(
                {
                    "per_file_rules": [
                        {
                            "pattern": "x.py",
                            "rules": ["FG1"],
                            "reason": "test",
                            "expires": "not-a-date",
                        }
                    ]
                }
            )
        )

        allowlist = load_allowlist(tmp_path)
        assert len(allowlist.per_file_rules) == 1
        assert allowlist.per_file_rules[0].expires is None

    def test_nonexistent_path_returns_empty_allowlist(self) -> None:
        """load_allowlist with nonexistent file path returns empty allowlist."""
        allowlist = load_allowlist(Path("/nonexistent/path.yaml"))
        assert len(allowlist.per_file_rules) == 0
        assert allowlist.fail_on_stale is True


# =============================================================================
# File scanning
# =============================================================================


class TestScanFile:
    """Integration tests for scan_file."""

    def test_syntax_error_returns_empty(self, tmp_path: Path) -> None:
        """Files with syntax errors return empty findings, not crash."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def foo(:\n    pass\n")
        findings = scan_file(bad_file, tmp_path)
        assert findings == []

    def test_encoding_error_returns_empty(self, tmp_path: Path) -> None:
        """Files with encoding errors return empty findings, not crash."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_bytes(b"\xff\xfe invalid utf-8 \x80\x81")
        findings = scan_file(bad_file, tmp_path)
        assert findings == []

    def test_clean_file_returns_empty(self, tmp_path: Path) -> None:
        """Files with no violations return empty findings."""
        good_file = tmp_path / "good.py"
        good_file.write_text(
            textwrap.dedent("""
            from dataclasses import dataclass
            from elspeth.contracts.freeze import freeze_fields

            @dataclass(frozen=True)
            class Record:
                data: dict

                def __post_init__(self):
                    freeze_fields(self, "data")
        """)
        )
        findings = scan_file(good_file, tmp_path)
        assert findings == []

    def test_violation_detected_in_file(self, tmp_path: Path) -> None:
        """scan_file detects violations and returns correct file paths."""
        bad_file = tmp_path / "bad.py"
        bad_file.write_text(
            textwrap.dedent("""
            class Record:
                def __post_init__(self):
                    object.__setattr__(self, "x", MappingProxyType(dict(self.x)))
        """)
        )
        findings = scan_file(bad_file, tmp_path)
        assert len(findings) == 1
        assert findings[0].rule_id == "FG1"
        assert findings[0].file_path == "bad.py"


# =============================================================================
# run_check integration
# =============================================================================


class TestRunCheck:
    """Integration tests for run_check — the CI entry point."""

    def _make_args(
        self,
        root: Path,
        allowlist: Path | None = None,
        files: list[Path] | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            command="check",
            root=root,
            allowlist=allowlist,
            files=files or [],
        )

    def test_clean_codebase_returns_zero(self, tmp_path: Path) -> None:
        """No violations, no stale rules → exit 0."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "clean.py").write_text(
            textwrap.dedent("""
            class Foo:
                def __post_init__(self):
                    pass
        """)
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        args = self._make_args(root=src, allowlist=allowlist_dir)
        assert run_check(args) == 0

    def test_violation_returns_one(self, tmp_path: Path) -> None:
        """Unallowlisted violation → exit 1."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text(
            textwrap.dedent("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        args = self._make_args(root=src, allowlist=allowlist_dir)
        assert run_check(args) == 1

    def test_allowlisted_violation_returns_zero(self, tmp_path: Path) -> None:
        """Violation covered by allowlist → exit 0."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "bad.py").write_text(
            textwrap.dedent("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump(
                {
                    "per_file_rules": [
                        {
                            "pattern": "bad.py",
                            "rules": ["FG1"],
                            "reason": "test",
                            "max_hits": 1,
                        }
                    ]
                }
            )
        )

        args = self._make_args(root=src, allowlist=allowlist_dir)
        assert run_check(args) == 0

    def test_precommit_mode_skips_staleness(self, tmp_path: Path) -> None:
        """Pre-commit mode (files arg) skips unused rule check."""
        src = tmp_path / "src"
        src.mkdir()
        clean_file = src / "clean.py"
        clean_file.write_text("x = 1\n")

        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()
        # This rule will be unused — but pre-commit mode should not fail
        (allowlist_dir / "rules.yaml").write_text(
            yaml.dump(
                {
                    "per_file_rules": [
                        {
                            "pattern": "nonexistent.py",
                            "rules": ["FG1"],
                            "reason": "stale entry",
                        }
                    ]
                }
            )
        )

        args = self._make_args(root=src, allowlist=allowlist_dir, files=[clean_file])
        assert run_check(args) == 0

    def test_file_outside_root_skipped(self, tmp_path: Path) -> None:
        """Pre-commit mode silently skips files not under --root."""
        src = tmp_path / "src"
        src.mkdir()
        outside = tmp_path / "outside.py"
        outside.write_text(
            textwrap.dedent("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        )
        allowlist_dir = tmp_path / "allowlist"
        allowlist_dir.mkdir()

        args = self._make_args(root=src, allowlist=allowlist_dir, files=[outside])
        # outside.py is not under src/ — should be skipped, returning 0
        assert run_check(args) == 0


# =============================================================================
# Finding data structure
# =============================================================================


class TestFinding:
    """Tests for Finding data structure."""

    def test_canonical_key_with_symbol_context(self) -> None:
        """canonical_key includes file path, rule, symbol context, and fingerprint."""
        findings = _scan("""
            class Foo:
                def __post_init__(self):
                    MappingProxyType(dict(self.x))
        """)
        assert len(findings) == 1
        key = findings[0].canonical_key
        assert "test.py" in key
        assert "FG1" in key
        assert "Foo:__post_init__" in key
        assert "fp=" in key

    def test_canonical_key_module_level_uses_sentinel(self) -> None:
        """canonical_key uses _module_ sentinel when no symbol context."""
        finding = Finding(
            rule_id="FG1",
            file_path="test.py",
            line=1,
            col=0,
            symbol_context=(),
            fingerprint="abc123",
            code_snippet="x = 1",
            message="test",
        )
        assert "_module_" in finding.canonical_key
