"""Tests for contracts enforcement script."""

from pathlib import Path

from scripts.check_contracts import find_type_definitions, load_whitelist


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
from elspeth.plugins.schemas import PluginSchema

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

    whitelist = load_whitelist(whitelist_file)
    assert "foo/bar:MyType" in whitelist["types"]


def test_whitelist_loading_empty_file(tmp_path: Path) -> None:
    """Handles empty whitelist file."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text("")

    whitelist = load_whitelist(whitelist_file)
    assert whitelist == {"types": set(), "dicts": set()}


def test_whitelist_loading_nonexistent_file(tmp_path: Path) -> None:
    """Handles missing whitelist file."""
    whitelist_file = tmp_path / "nonexistent.yaml"

    whitelist = load_whitelist(whitelist_file)
    assert whitelist == {"types": set(), "dicts": set()}


def test_whitelist_loading_multiple_entries(tmp_path: Path) -> None:
    """Loads multiple whitelist entries."""
    whitelist_file = tmp_path / ".contracts-whitelist.yaml"
    whitelist_file.write_text("""
allowed_external_types:
  - "module/a:TypeA"
  - "module/b:TypeB"
  - "module/c:TypeC"
""")

    whitelist = load_whitelist(whitelist_file)
    assert len(whitelist["types"]) == 3
    assert "module/a:TypeA" in whitelist["types"]
    assert "module/b:TypeB" in whitelist["types"]
    assert "module/c:TypeC" in whitelist["types"]


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
