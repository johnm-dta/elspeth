"""Tests for contracts enforcement script."""

from pathlib import Path
from unittest.mock import patch

from scripts.check_contracts import (
    FieldCoverageViolation,
    FieldMappingViolation,
    FieldMappingVisitor,
    HardcodeLiteralVisitor,
    HardcodeViolation,
    SettingsAccessVisitor,
    SettingsViolation,
    check_field_name_mappings,
    check_from_settings_coverage,
    check_hardcode_documentation,
    check_settings_alignment,
    extract_from_settings_accesses,
    extract_from_settings_field_mappings,
    extract_from_settings_hardcodes,
    find_settings_classes,
    find_type_definitions,
    get_settings_class_fields,
    load_whitelist,
)


def test_finds_dataclass_definitions(tmp_path: Path) -> None:
    """Finds @dataclass decorated classes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from dataclasses import dataclass

@dataclass
class MyType:
    name: str
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyType"
    assert definitions[0][2] == "dataclass"


def test_finds_dataclass_with_args(tmp_path: Path) -> None:
    """Finds @dataclass(frozen=True) decorated classes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenType:
    value: int
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "FrozenType"
    assert definitions[0][2] == "dataclass"


def test_finds_enum_definitions(tmp_path: Path) -> None:
    """Finds Enum subclasses."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from enum import Enum

class MyEnum(Enum):
    A = "a"
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyEnum"
    assert definitions[0][2] == "Enum"


def test_finds_typeddict_definitions(tmp_path: Path) -> None:
    """Finds TypedDict subclasses."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from typing import TypedDict

class MyDict(TypedDict):
    name: str
    value: int
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyDict"
    assert definitions[0][2] == "TypedDict"


def test_finds_namedtuple_definitions(tmp_path: Path) -> None:
    """Finds NamedTuple subclasses."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from typing import NamedTuple

class MyTuple(NamedTuple):
    x: int
    y: int
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 1
    assert definitions[0][0] == "MyTuple"
    assert definitions[0][2] == "NamedTuple"


def test_ignores_pydantic_basemodel(tmp_path: Path) -> None:
    """Does not flag Pydantic BaseModel classes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from pydantic import BaseModel

class MyModel(BaseModel):
    name: str
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 0


def test_ignores_plugin_schema(tmp_path: Path) -> None:
    """Does not flag PluginSchema classes."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from elspeth.contracts import PluginSchema

class MyPluginConfig(PluginSchema):
    setting: str
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 0


def test_finds_multiple_definitions(tmp_path: Path) -> None:
    """Finds multiple type definitions in a single file."""
    test_file = tmp_path / "test.py"
    test_file.write_text("""
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

@dataclass
class DataType:
    value: int

class StatusEnum(Enum):
    ACTIVE = "active"

class ConfigDict(TypedDict):
    name: str
""")

    definitions = find_type_definitions(test_file)
    assert len(definitions) == 3
    names = {d[0] for d in definitions}
    assert names == {"DataType", "StatusEnum", "ConfigDict"}


def test_whitelist_loading(tmp_path: Path) -> None:
    """Loads whitelist from YAML."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text("""
allowed_external_types:
  - "foo/bar:MyType"
""")

    whitelist, entries = load_whitelist(whitelist_file)
    assert "foo/bar:MyType" in whitelist["types"]
    assert len(entries) == 1
    assert entries[0].value == "foo/bar:MyType"
    assert entries[0].category == "type"


def test_whitelist_loading_empty_file(tmp_path: Path) -> None:
    """Handles empty whitelist file."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text("")

    whitelist, entries = load_whitelist(whitelist_file)
    assert whitelist == {"types": set(), "dicts": set()}
    assert entries == []


def test_whitelist_loading_nonexistent_file(tmp_path: Path) -> None:
    """Handles missing whitelist file."""
    whitelist_file = tmp_path / "nonexistent.yaml"

    whitelist, entries = load_whitelist(whitelist_file)
    assert whitelist == {"types": set(), "dicts": set()}
    assert entries == []


def test_whitelist_loading_multiple_entries(tmp_path: Path) -> None:
    """Loads multiple whitelist entries."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text("""
allowed_external_types:
  - "module/a:TypeA"
  - "module/b:TypeB"
  - "module/c:TypeC"
""")

    whitelist, entries = load_whitelist(whitelist_file)
    assert len(whitelist["types"]) == 3
    assert "module/a:TypeA" in whitelist["types"]
    assert "module/b:TypeB" in whitelist["types"]
    assert "module/c:TypeC" in whitelist["types"]
    assert len(entries) == 3
    assert all(e.category == "type" for e in entries)


def test_handles_syntax_errors(tmp_path: Path) -> None:
    """Gracefully handles files with syntax errors."""
    test_file = tmp_path / "broken.py"
    test_file.write_text("def broken(\n")  # Invalid syntax

    definitions = find_type_definitions(test_file)
    assert definitions == []


def test_handles_unicode_errors(tmp_path: Path) -> None:
    """Gracefully handles files with encoding issues."""
    test_file = tmp_path / "binary.py"
    test_file.write_bytes(b"\x80\x81\x82")  # Invalid UTF-8

    definitions = find_type_definitions(test_file)
    assert definitions == []


# =============================================================================
# Settings Alignment Tests
# =============================================================================


def test_find_settings_classes_finds_settings_suffix(tmp_path: Path) -> None:
    """Finds classes ending in 'Settings'."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
from pydantic import BaseModel

class RetrySettings(BaseModel):
    max_attempts: int

class RateLimitSettings(BaseModel):
    enabled: bool

class SomeConfig(BaseModel):  # Should NOT match - doesn't end in 'Settings'
    value: str
""")

    settings = find_settings_classes(test_file)
    names = {s[0] for s in settings}
    assert names == {"RetrySettings", "RateLimitSettings"}
    assert "SomeConfig" not in names


def test_find_settings_classes_returns_line_numbers(tmp_path: Path) -> None:
    """Returns correct line numbers for Settings classes."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
class FirstSettings:
    pass

class SecondSettings:
    pass
""")

    settings = find_settings_classes(test_file)
    # Line 2 and Line 5 (accounting for leading newline)
    assert len(settings) == 2
    assert settings[0] == ("FirstSettings", 2)
    assert settings[1] == ("SecondSettings", 5)


def test_find_settings_classes_handles_empty_file(tmp_path: Path) -> None:
    """Handles empty files gracefully."""
    test_file = tmp_path / "empty.py"
    test_file.write_text("")

    settings = find_settings_classes(test_file)
    assert settings == []


def test_find_settings_classes_handles_syntax_errors(tmp_path: Path) -> None:
    """Handles syntax errors gracefully."""
    test_file = tmp_path / "broken.py"
    test_file.write_text("class BrokenSettings(\n")  # Invalid syntax

    settings = find_settings_classes(test_file)
    assert settings == []


def test_check_settings_alignment_passes_with_mapping(tmp_path: Path) -> None:
    """Settings class with Runtime counterpart passes."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
class RetrySettings:
    pass
""")

    # Mock the alignment module to have RetrySettings in SETTINGS_TO_RUNTIME
    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"RetrySettings": "RuntimeRetryConfig"},
        ),
        patch("elspeth.contracts.config.alignment.EXEMPT_SETTINGS", set()),
    ):
        violations = check_settings_alignment(test_file)
        assert violations == []


def test_check_settings_alignment_passes_with_exempt(tmp_path: Path) -> None:
    """Settings class in EXEMPT_SETTINGS passes."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
class SourceSettings:
    pass
""")

    # Mock the alignment module to have SourceSettings in EXEMPT_SETTINGS
    with (
        patch("elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME", {}),
        patch("elspeth.contracts.config.alignment.EXEMPT_SETTINGS", {"SourceSettings"}),
    ):
        violations = check_settings_alignment(test_file)
        assert violations == []


def test_check_settings_alignment_detects_orphaned(tmp_path: Path) -> None:
    """Orphaned Settings class is detected as violation."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
class OrphanedSettings:
    pass
""")

    # Mock empty mappings - OrphanedSettings is not mapped or exempt
    with (
        patch("elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME", {}),
        patch("elspeth.contracts.config.alignment.EXEMPT_SETTINGS", set()),
    ):
        violations = check_settings_alignment(test_file)
        assert len(violations) == 1
        assert violations[0].class_name == "OrphanedSettings"
        assert violations[0].line == 2


def test_check_settings_alignment_multiple_classes(tmp_path: Path) -> None:
    """Mixed scenario: mapped, exempt, and orphaned."""
    test_file = tmp_path / "config.py"
    test_file.write_text("""
class MappedSettings:
    pass

class ExemptSettings:
    pass

class OrphanedSettings:
    pass
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"MappedSettings": "RuntimeMappedConfig"},
        ),
        patch("elspeth.contracts.config.alignment.EXEMPT_SETTINGS", {"ExemptSettings"}),
    ):
        violations = check_settings_alignment(test_file)
        # Only OrphanedSettings should be flagged
        assert len(violations) == 1
        assert violations[0].class_name == "OrphanedSettings"


def test_check_settings_alignment_uses_real_mappings() -> None:
    """Integration test: actual core/config.py with real mappings.

    This verifies that the current codebase passes the check.
    If this fails, a new Settings class was added without updating
    SETTINGS_TO_RUNTIME or EXEMPT_SETTINGS in alignment.py.
    """
    import pytest

    config_path = Path("src/elspeth/core/config.py")
    if not config_path.exists():
        pytest.skip("Running from different directory - core/config.py not found")

    violations = check_settings_alignment(config_path)
    assert violations == [], (
        f"Orphaned Settings classes found: {[v.class_name for v in violations]}. "
        "Add to SETTINGS_TO_RUNTIME or EXEMPT_SETTINGS in contracts/config/alignment.py"
    )


def test_settings_violation_dataclass() -> None:
    """SettingsViolation dataclass holds expected fields."""
    violation = SettingsViolation(
        class_name="TestSettings",
        file="/path/to/config.py",
        line=42,
    )
    assert violation.class_name == "TestSettings"
    assert violation.file == "/path/to/config.py"
    assert violation.line == 42


# =============================================================================
# Field Coverage Tests (from_settings() method checks)
# =============================================================================


def test_settings_access_visitor_finds_direct_access() -> None:
    """SettingsAccessVisitor finds settings.X accesses."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        field_a=settings.field_a,
        field_b=settings.field_b,
    )
"""
    tree = ast.parse(code)
    visitor = SettingsAccessVisitor("settings")
    visitor.visit(tree)

    assert visitor.accessed_fields == {"field_a", "field_b"}


def test_settings_access_visitor_finds_chained_access() -> None:
    """SettingsAccessVisitor finds nested access like settings.field.method()."""
    import ast

    code = """
def from_settings(cls, settings):
    # Access settings.nested_field and call a method on it
    value = settings.nested_field.some_method()
    return cls(value=value)
"""
    tree = ast.parse(code)
    visitor = SettingsAccessVisitor("settings")
    visitor.visit(tree)

    # Should capture just the direct attribute, not the chained ones
    assert "nested_field" in visitor.accessed_fields


def test_settings_access_visitor_handles_different_param_names() -> None:
    """SettingsAccessVisitor works with different parameter names."""
    import ast

    code = """
def from_settings(cls, config):
    return cls(field=config.field)
"""
    tree = ast.parse(code)
    visitor = SettingsAccessVisitor("config")
    visitor.visit(tree)

    assert visitor.accessed_fields == {"field"}


def test_extract_from_settings_accesses_finds_method(tmp_path: Path) -> None:
    """extract_from_settings_accesses finds from_settings method."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    field_a: int
    field_b: str

    @classmethod
    def from_settings(cls, settings):
        return cls(
            field_a=settings.field_a,
            field_b=settings.field_b,
        )
""")

    result = extract_from_settings_accesses(runtime_file)

    assert "RuntimeTestConfig" in result
    assert result["RuntimeTestConfig"] == {"field_a", "field_b"}


def test_extract_from_settings_accesses_handles_no_method(tmp_path: Path) -> None:
    """extract_from_settings_accesses handles classes without from_settings."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    field_a: int

    @classmethod
    def default(cls):
        return cls(field_a=1)
""")

    result = extract_from_settings_accesses(runtime_file)

    # Class exists but has no from_settings, so not in result
    assert "RuntimeTestConfig" not in result


def test_extract_from_settings_accesses_multiple_classes(tmp_path: Path) -> None:
    """extract_from_settings_accesses handles multiple Runtime classes."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int

    @classmethod
    def from_settings(cls, settings):
        return cls(max_attempts=settings.max_attempts)


@dataclass
class RuntimeCheckpointConfig:
    enabled: bool
    frequency: int

    @classmethod
    def from_settings(cls, settings):
        return cls(
            enabled=settings.enabled,
            frequency=settings.frequency,
        )
""")

    result = extract_from_settings_accesses(runtime_file)

    assert len(result) == 2
    assert result["RuntimeRetryConfig"] == {"max_attempts"}
    assert result["RuntimeCheckpointConfig"] == {"enabled", "frequency"}


def test_get_settings_class_fields_extracts_fields(tmp_path: Path) -> None:
    """get_settings_class_fields extracts field names from Settings class."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
from pydantic import BaseModel, Field

class TestSettings(BaseModel):
    field_a: int = Field(default=1)
    field_b: str = "default"
    field_c: bool
""")

    fields = get_settings_class_fields(config_file, "TestSettings")

    assert fields == {"field_a", "field_b", "field_c"}


def test_get_settings_class_fields_returns_empty_for_missing_class(tmp_path: Path) -> None:
    """get_settings_class_fields returns empty set for non-existent class."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
class OtherClass:
    pass
""")

    fields = get_settings_class_fields(config_file, "TestSettings")

    assert fields == set()


def test_check_from_settings_coverage_passes_full_coverage(tmp_path: Path) -> None:
    """check_from_settings_coverage passes when all fields are accessed."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
class TestSettings:
    field_a: int
    field_b: str
""")

    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
class RuntimeTestConfig:
    @classmethod
    def from_settings(cls, settings):
        return cls(
            field_a=settings.field_a,
            field_b=settings.field_b,
        )
""")

    with patch(
        "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
        {"TestSettings": "RuntimeTestConfig"},
    ):
        violations = check_from_settings_coverage(config_file, runtime_file)

    assert violations == []


def test_check_from_settings_coverage_detects_orphan(tmp_path: Path) -> None:
    """check_from_settings_coverage detects orphaned Settings fields."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
class TestSettings:
    field_a: int
    field_b: str
    orphaned_field: float
""")

    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
class RuntimeTestConfig:
    @classmethod
    def from_settings(cls, settings):
        return cls(
            field_a=settings.field_a,
            field_b=settings.field_b,
            # orphaned_field is NOT accessed!
        )
""")

    with patch(
        "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
        {"TestSettings": "RuntimeTestConfig"},
    ):
        violations = check_from_settings_coverage(config_file, runtime_file)

    assert len(violations) == 1
    assert violations[0].settings_class == "TestSettings"
    assert violations[0].runtime_class == "RuntimeTestConfig"
    assert violations[0].orphaned_field == "orphaned_field"


def test_check_from_settings_coverage_detects_multiple_orphans(tmp_path: Path) -> None:
    """check_from_settings_coverage detects multiple orphaned fields."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
class TestSettings:
    used_field: int
    orphan_a: str
    orphan_b: float
""")

    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
class RuntimeTestConfig:
    @classmethod
    def from_settings(cls, settings):
        return cls(used_field=settings.used_field)
""")

    with patch(
        "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
        {"TestSettings": "RuntimeTestConfig"},
    ):
        violations = check_from_settings_coverage(config_file, runtime_file)

    assert len(violations) == 2
    orphan_names = {v.orphaned_field for v in violations}
    assert orphan_names == {"orphan_a", "orphan_b"}


def test_check_from_settings_coverage_skips_unmapped_classes(tmp_path: Path) -> None:
    """check_from_settings_coverage skips classes not in SETTINGS_TO_RUNTIME."""
    config_file = tmp_path / "config.py"
    config_file.write_text("""
class UnmappedSettings:
    some_field: int
""")

    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
class RuntimeUnmappedConfig:
    @classmethod
    def from_settings(cls, settings):
        # Doesn't access any fields
        return cls()
""")

    # Empty mapping - no Settings classes are mapped
    with patch(
        "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
        {},
    ):
        violations = check_from_settings_coverage(config_file, runtime_file)

    # No violations because the class isn't in the mapping
    assert violations == []


def test_check_from_settings_coverage_real_codebase() -> None:
    """Integration test: actual codebase has full field coverage.

    This verifies that the current codebase passes the check.
    If this fails, a Settings field was added but not accessed
    in the corresponding from_settings() method.
    """
    import pytest

    config_path = Path("src/elspeth/core/config.py")
    runtime_path = Path("src/elspeth/contracts/config/runtime.py")

    if not config_path.exists() or not runtime_path.exists():
        pytest.skip("Running from different directory - required files not found")

    violations = check_from_settings_coverage(config_path, runtime_path)
    assert violations == [], (
        f"Settings field coverage violations found: "
        f"{[(v.settings_class, v.orphaned_field) for v in violations]}. "
        f"Access these fields in from_settings() or remove them from the Settings class."
    )


def test_field_coverage_violation_dataclass() -> None:
    """FieldCoverageViolation dataclass holds expected fields."""
    violation = FieldCoverageViolation(
        settings_class="TestSettings",
        runtime_class="RuntimeTestConfig",
        orphaned_field="orphaned",
        file="/path/to/runtime.py",
        line=42,
    )
    assert violation.settings_class == "TestSettings"
    assert violation.runtime_class == "RuntimeTestConfig"
    assert violation.orphaned_field == "orphaned"
    assert violation.file == "/path/to/runtime.py"
    assert violation.line == 42


# =============================================================================
# Field Mapping Validation Tests (check_field_name_mappings)
# =============================================================================


def test_field_mapping_visitor_finds_direct_mappings() -> None:
    """FieldMappingVisitor finds runtime_field=settings.settings_field patterns."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        field_a=settings.field_a,
        field_b=settings.field_b,
    )
"""
    tree = ast.parse(code)
    visitor = FieldMappingVisitor("settings")
    visitor.visit(tree)

    assert len(visitor.field_mappings) == 2
    assert ("field_a", "field_a") in visitor.field_mappings
    assert ("field_b", "field_b") in visitor.field_mappings


def test_field_mapping_visitor_finds_renamed_mappings() -> None:
    """FieldMappingVisitor captures renamed field mappings."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        base_delay=settings.initial_delay_seconds,
        max_delay=settings.max_delay_seconds,
    )
"""
    tree = ast.parse(code)
    visitor = FieldMappingVisitor("settings")
    visitor.visit(tree)

    assert len(visitor.field_mappings) == 2
    assert ("base_delay", "initial_delay_seconds") in visitor.field_mappings
    assert ("max_delay", "max_delay_seconds") in visitor.field_mappings


def test_field_mapping_visitor_ignores_non_settings_values() -> None:
    """FieldMappingVisitor ignores values that aren't settings.X."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        field_a=settings.field_a,
        field_b=42,  # Not settings.X
        field_c="constant",  # Not settings.X
        field_d=some_func(),  # Not settings.X
    )
"""
    tree = ast.parse(code)
    visitor = FieldMappingVisitor("settings")
    visitor.visit(tree)

    # Only field_a should be captured
    assert len(visitor.field_mappings) == 1
    assert ("field_a", "field_a") in visitor.field_mappings


def test_field_mapping_visitor_handles_different_param_names() -> None:
    """FieldMappingVisitor works with different parameter names."""
    import ast

    code = """
def from_settings(cls, config):
    return cls(field=config.field)
"""
    tree = ast.parse(code)
    visitor = FieldMappingVisitor("config")
    visitor.visit(tree)

    assert visitor.field_mappings == [("field", "field")]


def test_extract_from_settings_field_mappings_finds_method(tmp_path: Path) -> None:
    """extract_from_settings_field_mappings finds mappings in from_settings method."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    field_a: int
    field_b: str

    @classmethod
    def from_settings(cls, settings):
        return cls(
            field_a=settings.field_a,
            field_b=settings.field_b,
        )
""")

    result = extract_from_settings_field_mappings(runtime_file)

    assert "RuntimeTestConfig" in result
    assert ("field_a", "field_a") in result["RuntimeTestConfig"]
    assert ("field_b", "field_b") in result["RuntimeTestConfig"]


def test_extract_from_settings_field_mappings_finds_renames(tmp_path: Path) -> None:
    """extract_from_settings_field_mappings captures renamed fields."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    base_delay: float
    max_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
        )
""")

    result = extract_from_settings_field_mappings(runtime_file)

    assert "RuntimeRetryConfig" in result
    mappings = result["RuntimeRetryConfig"]
    assert ("base_delay", "initial_delay_seconds") in mappings
    assert ("max_delay", "max_delay_seconds") in mappings


def test_check_field_name_mappings_passes_correct_mapping(tmp_path: Path) -> None:
    """check_field_name_mappings passes when mappings match FIELD_MAPPINGS."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    base_delay: float
    max_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
        )
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"TestSettings": "RuntimeTestConfig"},
        ),
        patch(
            "elspeth.contracts.config.alignment.FIELD_MAPPINGS",
            {
                "TestSettings": {
                    "initial_delay_seconds": "base_delay",
                    "max_delay_seconds": "max_delay",
                }
            },
        ),
    ):
        violations = check_field_name_mappings(runtime_file)

    assert violations == []


def test_check_field_name_mappings_detects_misroute(tmp_path: Path) -> None:
    """check_field_name_mappings detects when settings field maps to wrong runtime field.

    This is the key test case: catching "misrouted" fields where code
    maps a settings field to the wrong runtime field.

    Example: FIELD_MAPPINGS says initial_delay_seconds -> base_delay
    But code has: base_delay=settings.max_delay_seconds (WRONG!)
    """
    runtime_file = tmp_path / "runtime.py"
    # INTENTIONAL MISROUTE: base_delay uses max_delay_seconds instead of initial_delay_seconds
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    base_delay: float
    max_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            base_delay=settings.max_delay_seconds,  # WRONG! Should be initial_delay_seconds
            max_delay=settings.max_delay_seconds,   # Correct
        )
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"TestSettings": "RuntimeTestConfig"},
        ),
        patch(
            "elspeth.contracts.config.alignment.FIELD_MAPPINGS",
            {
                "TestSettings": {
                    "initial_delay_seconds": "base_delay",
                    "max_delay_seconds": "max_delay",
                }
            },
        ),
    ):
        violations = check_field_name_mappings(runtime_file)

    assert len(violations) == 1
    v = violations[0]
    assert v.runtime_class == "RuntimeTestConfig"
    assert v.runtime_field == "base_delay"
    assert v.settings_field == "max_delay_seconds"  # What code has (wrong)
    assert v.expected_settings_field == "initial_delay_seconds"  # What it should be


def test_check_field_name_mappings_detects_swapped_fields(tmp_path: Path) -> None:
    """check_field_name_mappings detects when two fields are swapped."""
    runtime_file = tmp_path / "runtime.py"
    # INTENTIONAL BUG: fields are swapped
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    base_delay: float
    max_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            base_delay=settings.max_delay_seconds,      # WRONG! Should be initial_delay_seconds
            max_delay=settings.initial_delay_seconds,   # WRONG! Should be max_delay_seconds
        )
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"TestSettings": "RuntimeTestConfig"},
        ),
        patch(
            "elspeth.contracts.config.alignment.FIELD_MAPPINGS",
            {
                "TestSettings": {
                    "initial_delay_seconds": "base_delay",
                    "max_delay_seconds": "max_delay",
                }
            },
        ),
    ):
        violations = check_field_name_mappings(runtime_file)

    # Both fields are misrouted
    assert len(violations) == 2
    violations_by_field = {v.runtime_field: v for v in violations}

    assert violations_by_field["base_delay"].settings_field == "max_delay_seconds"
    assert violations_by_field["base_delay"].expected_settings_field == "initial_delay_seconds"

    assert violations_by_field["max_delay"].settings_field == "initial_delay_seconds"
    assert violations_by_field["max_delay"].expected_settings_field == "max_delay_seconds"


def test_check_field_name_mappings_ignores_unmapped_classes(tmp_path: Path) -> None:
    """check_field_name_mappings ignores classes not in FIELD_MAPPINGS."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeConcurrencyConfig:
    max_workers: int

    @classmethod
    def from_settings(cls, settings):
        return cls(max_workers=settings.max_workers)
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"ConcurrencySettings": "RuntimeConcurrencyConfig"},
        ),
        patch(
            "elspeth.contracts.config.alignment.FIELD_MAPPINGS",
            {},  # No field mappings for this class
        ),
    ):
        violations = check_field_name_mappings(runtime_file)

    # No violations - class has no renamed fields in FIELD_MAPPINGS
    assert violations == []


def test_check_field_name_mappings_ignores_direct_name_fields(tmp_path: Path) -> None:
    """check_field_name_mappings ignores fields that aren't in FIELD_MAPPINGS.

    Fields with matching names (e.g., max_attempts=settings.max_attempts)
    don't need to be in FIELD_MAPPINGS and should not cause violations.
    """
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int
    base_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_attempts=settings.max_attempts,          # Direct mapping - not in FIELD_MAPPINGS
            base_delay=settings.initial_delay_seconds,   # Renamed - in FIELD_MAPPINGS
        )
""")

    with (
        patch(
            "elspeth.contracts.config.alignment.SETTINGS_TO_RUNTIME",
            {"RetrySettings": "RuntimeRetryConfig"},
        ),
        patch(
            "elspeth.contracts.config.alignment.FIELD_MAPPINGS",
            {
                "RetrySettings": {
                    "initial_delay_seconds": "base_delay",
                    # max_attempts is NOT in FIELD_MAPPINGS (same name)
                }
            },
        ),
    ):
        violations = check_field_name_mappings(runtime_file)

    # No violations - base_delay correctly uses initial_delay_seconds
    # max_attempts is not in FIELD_MAPPINGS so not checked
    assert violations == []


def test_check_field_name_mappings_real_codebase() -> None:
    """Integration test: actual codebase has correct field mappings.

    This verifies that the current codebase passes the check.
    If this fails, a field mapping in from_settings() doesn't match
    the documented mapping in FIELD_MAPPINGS.
    """
    import pytest

    runtime_path = Path("src/elspeth/contracts/config/runtime.py")

    if not runtime_path.exists():
        pytest.skip("Running from different directory - required files not found")

    violations = check_field_name_mappings(runtime_path)
    assert violations == [], (
        f"Field mapping violations found: "
        f"{[(v.runtime_field, v.settings_field, v.expected_settings_field) for v in violations]}. "
        f"Fix the mapping in from_settings() or update FIELD_MAPPINGS."
    )


def test_field_mapping_violation_dataclass() -> None:
    """FieldMappingViolation dataclass holds expected fields."""
    violation = FieldMappingViolation(
        runtime_class="RuntimeTestConfig",
        runtime_field="base_delay",
        settings_field="max_delay_seconds",
        expected_settings_field="initial_delay_seconds",
        file="/path/to/runtime.py",
        line=42,
    )
    assert violation.runtime_class == "RuntimeTestConfig"
    assert violation.runtime_field == "base_delay"
    assert violation.settings_field == "max_delay_seconds"
    assert violation.expected_settings_field == "initial_delay_seconds"
    assert violation.file == "/path/to/runtime.py"
    assert violation.line == 42


# =============================================================================
# Hardcode Documentation Tests (check_hardcode_documentation)
# =============================================================================


def test_hardcode_literal_visitor_finds_plain_literals() -> None:
    """HardcodeLiteralVisitor finds plain literal values in keyword args."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        field_a=settings.field_a,
        jitter=1.0,
        magic_number=42,
        name="default",
        enabled=True,
    )
"""
    tree = ast.parse(code)
    visitor = HardcodeLiteralVisitor("settings")
    visitor.visit(tree)

    # Should find: jitter=1.0, magic_number=42, name="default", enabled=True
    # Should NOT find: field_a=settings.field_a
    assert len(visitor.hardcoded_literals) == 4
    literals_dict = dict(visitor.hardcoded_literals)
    assert literals_dict["jitter"] == 1.0
    assert literals_dict["magic_number"] == 42
    assert literals_dict["name"] == "default"
    assert literals_dict["enabled"] is True


def test_hardcode_literal_visitor_ignores_function_calls() -> None:
    """HardcodeLiteralVisitor ignores values wrapped in function calls."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        jitter=float(INTERNAL_DEFAULTS["retry"]["jitter"]),
        base_delay=float(settings.initial_delay_seconds),
        computed=some_func(42),
    )
"""
    tree = ast.parse(code)
    visitor = HardcodeLiteralVisitor("settings")
    visitor.visit(tree)

    # Should NOT find any - all are function calls
    assert len(visitor.hardcoded_literals) == 0


def test_hardcode_literal_visitor_ignores_subscripts() -> None:
    """HardcodeLiteralVisitor ignores subscript values like dict["key"]."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        jitter=INTERNAL_DEFAULTS["retry"]["jitter"],
        value=some_dict["key"],
    )
"""
    tree = ast.parse(code)
    visitor = HardcodeLiteralVisitor("settings")
    visitor.visit(tree)

    # Should NOT find any - all are subscripts
    assert len(visitor.hardcoded_literals) == 0


def test_hardcode_literal_visitor_finds_negative_numbers() -> None:
    """HardcodeLiteralVisitor finds negative number literals."""
    import ast

    code = """
def from_settings(cls, settings):
    return cls(
        offset=-1.5,
        countdown=-42,
    )
"""
    tree = ast.parse(code)
    visitor = HardcodeLiteralVisitor("settings")
    visitor.visit(tree)

    assert len(visitor.hardcoded_literals) == 2
    literals_dict = dict(visitor.hardcoded_literals)
    assert literals_dict["offset"] == -1.5
    assert literals_dict["countdown"] == -42


def test_extract_from_settings_hardcodes_finds_method(tmp_path: Path) -> None:
    """extract_from_settings_hardcodes finds hardcodes in from_settings method."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeTestConfig:
    field_a: int
    jitter: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            field_a=settings.field_a,
            jitter=1.0,
        )
""")

    result = extract_from_settings_hardcodes(runtime_file)

    assert "RuntimeTestConfig" in result
    assert len(result["RuntimeTestConfig"]) == 1
    assert ("jitter", 1.0) in result["RuntimeTestConfig"]


def test_extract_from_settings_hardcodes_multiple_classes(tmp_path: Path) -> None:
    """extract_from_settings_hardcodes handles multiple Runtime classes."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int
    jitter: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_attempts=settings.max_attempts,
            jitter=1.0,
        )


@dataclass
class RuntimeCheckpointConfig:
    enabled: bool
    buffer_size: int

    @classmethod
    def from_settings(cls, settings):
        return cls(
            enabled=settings.enabled,
            buffer_size=1000,
        )
""")

    result = extract_from_settings_hardcodes(runtime_file)

    assert len(result) == 2
    assert ("jitter", 1.0) in result["RuntimeRetryConfig"]
    assert ("buffer_size", 1000) in result["RuntimeCheckpointConfig"]


def test_check_hardcode_documentation_passes_documented(tmp_path: Path) -> None:
    """check_hardcode_documentation passes when hardcode is documented."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int
    jitter: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_attempts=settings.max_attempts,
            jitter=1.0,  # Documented in INTERNAL_DEFAULTS["retry"]["jitter"]
        )
""")

    with patch(
        "elspeth.contracts.config.defaults.INTERNAL_DEFAULTS",
        {"retry": {"jitter": 1.0}},
    ):
        violations = check_hardcode_documentation(runtime_file)

    assert violations == []


def test_check_hardcode_documentation_detects_undocumented(tmp_path: Path) -> None:
    """check_hardcode_documentation detects undocumented hardcode (the key test).

    This is the primary test case - catching hardcoded literals that are NOT
    documented in INTERNAL_DEFAULTS.
    """
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int
    jitter: float
    magic_number: int

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_attempts=settings.max_attempts,
            jitter=1.0,
            magic_number=42,  # UNDOCUMENTED - should be violation!
        )
""")

    with patch(
        "elspeth.contracts.config.defaults.INTERNAL_DEFAULTS",
        {"retry": {"jitter": 1.0}},  # magic_number is NOT documented
    ):
        violations = check_hardcode_documentation(runtime_file)

    assert len(violations) == 1
    v = violations[0]
    assert v.runtime_class == "RuntimeRetryConfig"
    assert v.runtime_field == "magic_number"
    assert "42" in v.literal_value
    assert v.subsystem == "retry"


def test_check_hardcode_documentation_detects_wrong_value(tmp_path: Path) -> None:
    """check_hardcode_documentation detects when documented value differs from code."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    jitter: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            jitter=2.0,  # Code has 2.0 but documented as 1.0!
        )
""")

    with patch(
        "elspeth.contracts.config.defaults.INTERNAL_DEFAULTS",
        {"retry": {"jitter": 1.0}},  # Documented as 1.0
    ):
        violations = check_hardcode_documentation(runtime_file)

    assert len(violations) == 1
    v = violations[0]
    assert v.runtime_class == "RuntimeRetryConfig"
    assert v.runtime_field == "jitter"
    assert "2.0" in v.literal_value
    assert "1.0" in v.literal_value  # Should mention the documented value


def test_check_hardcode_documentation_no_subsystem_mapping(tmp_path: Path) -> None:
    """check_hardcode_documentation flags hardcodes in classes without subsystem mapping."""
    runtime_file = tmp_path / "runtime.py"
    # RuntimeUnknownConfig is not in RUNTIME_TO_SUBSYSTEM
    runtime_file.write_text("""
@dataclass
class RuntimeUnknownConfig:
    magic: int

    @classmethod
    def from_settings(cls, settings):
        return cls(
            magic=42,  # No subsystem mapping - all hardcodes are violations
        )
""")

    with patch(
        "elspeth.contracts.config.defaults.INTERNAL_DEFAULTS",
        {},
    ):
        violations = check_hardcode_documentation(runtime_file)

    assert len(violations) == 1
    v = violations[0]
    assert v.runtime_class == "RuntimeUnknownConfig"
    assert "(no subsystem mapping)" in v.subsystem


def test_check_hardcode_documentation_ignores_classes_without_hardcodes(tmp_path: Path) -> None:
    """check_hardcode_documentation passes classes with no hardcoded literals."""
    runtime_file = tmp_path / "runtime.py"
    runtime_file.write_text("""
@dataclass
class RuntimeRetryConfig:
    max_attempts: int
    base_delay: float

    @classmethod
    def from_settings(cls, settings):
        return cls(
            max_attempts=settings.max_attempts,
            base_delay=settings.initial_delay_seconds,
        )
""")

    with patch(
        "elspeth.contracts.config.defaults.INTERNAL_DEFAULTS",
        {"retry": {}},
    ):
        violations = check_hardcode_documentation(runtime_file)

    assert violations == []


def test_check_hardcode_documentation_real_codebase() -> None:
    """Integration test: actual codebase has documented hardcodes.

    This verifies that the current codebase passes the check.
    If this fails, a hardcoded literal was added to from_settings()
    without documenting it in INTERNAL_DEFAULTS.
    """
    import pytest

    runtime_path = Path("src/elspeth/contracts/config/runtime.py")

    if not runtime_path.exists():
        pytest.skip("Running from different directory - required files not found")

    violations = check_hardcode_documentation(runtime_path)
    assert violations == [], (
        f"Undocumented hardcode violations found: "
        f"{[(v.runtime_class, v.runtime_field, v.literal_value) for v in violations]}. "
        f"Add these to INTERNAL_DEFAULTS in contracts/config/defaults.py."
    )


def test_hardcode_violation_dataclass() -> None:
    """HardcodeViolation dataclass holds expected fields."""
    violation = HardcodeViolation(
        runtime_class="RuntimeRetryConfig",
        runtime_field="magic_number",
        literal_value="42",
        subsystem="retry",
        file="/path/to/runtime.py",
        line=0,
    )
    assert violation.runtime_class == "RuntimeRetryConfig"
    assert violation.runtime_field == "magic_number"
    assert violation.literal_value == "42"
    assert violation.subsystem == "retry"
    assert violation.file == "/path/to/runtime.py"
    assert violation.line == 0
