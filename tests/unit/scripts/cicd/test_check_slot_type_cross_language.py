"""Tests for scripts/cicd/check_slot_type_cross_language.py.

Tests the golden (in-sync) case, drift cases (Python has extras, TS has extras),
and error cases (malformed TS, missing file). Uses subprocess.run to invoke the
script in a realistic way so the test exercises the same code path as CI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers for building minimal mock files in tmp_path
# ---------------------------------------------------------------------------

_PYTHON_TEMPLATE = """\
from typing import Literal

SlotType = Literal[{members}]
"""

_TS_TEMPLATE = """\
export interface RecipeSlotInput {{
  name: string;
  slot_type: {ts_union};
  description: string;
}}
"""


def _write_python(directory: Path, members: list[str]) -> Path:
    """Write a minimal recipes.py with a SlotType Literal."""
    py_dir = directory / "src" / "elspeth" / "web" / "composer"
    py_dir.mkdir(parents=True, exist_ok=True)
    # Write __init__.py for the package chain so importlib finds it
    for parent in [
        directory / "src",
        directory / "src" / "elspeth",
        directory / "src" / "elspeth" / "web",
        directory / "src" / "elspeth" / "web" / "composer",
    ]:
        (parent / "__init__.py").write_text("")
    member_str = ", ".join(f'"{m}"' for m in members)
    f = py_dir / "recipes.py"
    f.write_text(_PYTHON_TEMPLATE.format(members=member_str))
    return f


def _write_ts(directory: Path, members: list[str]) -> Path:
    """Write a minimal guided.ts with a slot_type field union."""
    ts_dir = directory / "src" / "elspeth" / "web" / "frontend" / "src" / "types"
    ts_dir.mkdir(parents=True, exist_ok=True)
    ts_union = " | ".join(f'"{m}"' for m in members)
    f = ts_dir / "guided.ts"
    f.write_text(_TS_TEMPLATE.format(ts_union=ts_union))
    return f


def _run_script(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run check_slot_type_cross_language.py from a given cwd."""
    script = Path(__file__).parents[4] / "scripts" / "cicd" / "check_slot_type_cross_language.py"
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


# ---------------------------------------------------------------------------
# Golden case: both sets in sync
# ---------------------------------------------------------------------------


class TestGoldenCase:
    def test_in_sync_exits_0(self, tmp_path: Path) -> None:
        members = ["blob_id", "str", "float", "int", "str_list"]
        _write_python(tmp_path, members)
        _write_ts(tmp_path, members)
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr

    def test_in_sync_prints_ok_and_sorted_set(self, tmp_path: Path) -> None:
        members = ["str", "int", "float"]  # unsorted order
        _write_python(tmp_path, members)
        _write_ts(tmp_path, ["float", "int", "str"])  # different order
        result = _run_script(tmp_path)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
        # Output must be sorted: float < int < str
        assert "['float', 'int', 'str']" in result.stdout

    def test_real_codebase_is_in_sync(self) -> None:
        """Smoke-test: run against the actual repo.  Fails if the mirror has drifted."""
        repo_root = Path(__file__).parents[4]
        result = _run_script(repo_root)
        assert result.returncode == 0, "SlotType / guided.ts mirror has drifted in the real codebase!\n" + result.stderr


# ---------------------------------------------------------------------------
# Drift case: TypeScript is missing a member Python has
# ---------------------------------------------------------------------------


class TestPythonHasExtra:
    def test_exits_1(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str", "int", "float", "new_type"])
        _write_ts(tmp_path, ["str", "int", "float"])
        result = _run_script(tmp_path)
        assert result.returncode == 1

    def test_reports_python_only_member(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str", "int", "float", "new_type"])
        _write_ts(tmp_path, ["str", "int", "float"])
        result = _run_script(tmp_path)
        assert "new_type" in result.stderr
        assert "In Python only" in result.stderr

    def test_prints_both_sets(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str", "new_type"])
        _write_ts(tmp_path, ["str"])
        result = _run_script(tmp_path)
        assert "Python members" in result.stderr
        assert "TypeScript members" in result.stderr


# ---------------------------------------------------------------------------
# Drift case: TypeScript has a member Python doesn't
# ---------------------------------------------------------------------------


class TestTSHasExtra:
    def test_exits_1(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str", "int"])
        _write_ts(tmp_path, ["str", "int", "ts_only"])
        result = _run_script(tmp_path)
        assert result.returncode == 1

    def test_reports_ts_only_member(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str"])
        _write_ts(tmp_path, ["str", "ts_only"])
        result = _run_script(tmp_path)
        assert "ts_only" in result.stderr
        assert "In TypeScript only" in result.stderr


# ---------------------------------------------------------------------------
# Error case: missing source files
# ---------------------------------------------------------------------------


class TestMissingFiles:
    def test_exits_1_when_python_source_missing(self, tmp_path: Path) -> None:
        _write_ts(tmp_path, ["str"])
        # Do NOT write Python source — Python path will not exist
        result = _run_script(tmp_path)
        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_exits_1_when_ts_source_missing(self, tmp_path: Path) -> None:
        _write_python(tmp_path, ["str"])
        # Do NOT write TS source — TS path will not exist
        result = _run_script(tmp_path)
        assert result.returncode == 1
        assert "not found" in result.stderr
