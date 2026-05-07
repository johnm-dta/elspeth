"""CI scanner: every exception class in errors.py whose name ends in Error or
Violation must have @tier_1_error(reason=...) or a '# TIER-2:' justification
comment with non-empty text.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Project root — the cwd for all subprocess invocations.  The scanner is
# resolved relative to this directory, matching how CI runs it.
_RUN_CWD = str(Path(__file__).resolve().parents[3])


def _run(extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the scanner via sys.executable with cwd set to the project root.

    Using sys.executable ensures the same interpreter as the test runner.
    Setting cwd to the project root makes the script importable and resolves
    relative paths the same way CI does.
    """
    return subprocess.run(
        [
            sys.executable,
            "scripts/cicd/enforce_tier_1_decoration.py",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=_RUN_CWD,
    )


def test_scanner_accepts_compliant_file(tmp_path: Path) -> None:
    """@tier_1_error(reason=..., caller_module=__name__) and # TIER-2: <text> both accepted; scanner passes."""
    good = tmp_path / "errors.py"
    good.write_text(
        "from elspeth.contracts.tier_registry import tier_1_error\n\n"
        "@tier_1_error(reason='audit corruption crashes pipeline', caller_module=__name__)\n"
        "class CriticalError(Exception):\n"
        "    pass\n\n"
        "# TIER-2: row-level plugin bug; routable via on_error.\n"
        "class RowValidationError(Exception):\n"
        "    pass\n"
    )
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(good), "--allowlist", str(allowlist_dir)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_flags_missing_decoration(tmp_path: Path) -> None:
    """A class with no decorator and no comment is flagged and exits non-zero."""
    bad = tmp_path / "errors.py"
    bad.write_text("class MysteryError(Exception):\n    pass\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(bad), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "MysteryError" in result.stdout


def test_scanner_flags_violation_suffix(tmp_path: Path) -> None:
    """Classes ending in Violation are checked just like Error-suffix classes."""
    bad = tmp_path / "errors.py"
    bad.write_text("class WeirdViolation(Exception):\n    pass\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(bad), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "WeirdViolation" in result.stdout


def test_scanner_accepts_tier_2_comment_with_justification(tmp_path: Path) -> None:
    """A '# TIER-2: <text>' comment immediately above the class is accepted."""
    good = tmp_path / "errors.py"
    good.write_text("# TIER-2: row-level data error; quarantine rather than crash.\nclass RowDataError(Exception):\n    pass\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(good), "--allowlist", str(allowlist_dir)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_rejects_tier_2_without_justification(tmp_path: Path) -> None:
    """A bare '# TIER-2:' with nothing after the colon is NOT accepted."""
    bad = tmp_path / "errors.py"
    bad.write_text("# TIER-2:\nclass EmptyJustificationError(Exception):\n    pass\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(bad), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "EmptyJustificationError" in result.stdout


def test_scanner_accepts_qualified_decorator(tmp_path: Path) -> None:
    """@module.tier_1_error(reason='...', caller_module=__name__) qualified-attribute form is accepted."""
    good = tmp_path / "errors.py"
    good.write_text(
        "import elspeth.contracts.tier_registry as reg\n\n"
        "@reg.tier_1_error(reason='framework bug crashes immediately', caller_module=__name__)\n"
        "class QualifiedError(Exception):\n"
        "    pass\n"
    )
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--file", str(good), "--allowlist", str(allowlist_dir)])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("decorator", "expected_detail"),
    [
        (
            "@tier_1_error(reason='framework bug')",
            "missing caller_module kwarg",
        ),
        (
            "@tier_1_error(reason='framework bug', caller_module='')",
            "caller_module value must be the __name__ literal",
        ),
    ],
)
def test_scanner_rejects_missing_or_spoofable_caller_module(
    tmp_path: Path,
    decorator: str,
    expected_detail: str,
) -> None:
    """caller_module must be the literal __name__ identifier, not absent or fabricated."""
    bad = tmp_path / "errors.py"
    bad.write_text(
        f"from elspeth.contracts.tier_registry import tier_1_error\n\n{decorator}\nclass BadCallerModuleError(Exception):\n    pass\n"
    )
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()

    result = _run(["check", "--file", str(bad), "--allowlist", str(allowlist_dir)])

    assert result.returncode != 0
    assert "TDE2 VIOLATIONS FOUND: 1" in result.stdout
    assert expected_detail in result.stdout


def test_scanner_read_error_exits_nonzero(tmp_path: Path) -> None:
    """A file that cannot be read produces a non-zero exit with 'Fatal:' in stderr."""
    unreadable = tmp_path / "errors.py"
    unreadable.write_text("class Fine(Exception): pass\n")
    os.chmod(unreadable, 0o000)
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    try:
        result = _run(["check", "--file", str(unreadable), "--allowlist", str(allowlist_dir)])
        assert result.returncode != 0
        assert "Fatal:" in result.stderr
    finally:
        # Restore permissions so tmp_path cleanup succeeds.
        os.chmod(unreadable, 0o644)
