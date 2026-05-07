"""CI scanner: any class defining to_audit_dict must inherit AuditEvidenceBase."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from scripts.cicd.enforce_audit_evidence_nominal import scan_file


def _run(extra_args: list[str]) -> subprocess.CompletedProcess[str]:
    """Invoke the enforcer via sys.executable with cwd set to the project root.

    Mirrors the pattern in test_enforce_composer_exception_channel.py:17-35.
    Using sys.executable ensures the same interpreter as the test runner;
    setting cwd to the project root makes the script importable.
    """
    return subprocess.run(
        [
            sys.executable,
            "scripts/cicd/enforce_audit_evidence_nominal.py",
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),  # project root
    )


def test_scanner_accepts_compliant_tree(tmp_path: Path) -> None:
    # Compliant tree: only AuditEvidenceBase subclasses define to_audit_dict.
    good = tmp_path / "good.py"
    good.write_text(
        "from elspeth.contracts.audit_evidence import AuditEvidenceBase\n"
        "class Ok(AuditEvidenceBase, RuntimeError):\n"
        "    def to_audit_dict(self): return {}\n"
    )
    # Use an empty allowlist dir so stale-entry detection is a no-op in this
    # test (no entries to be stale), which lets fail_on_stale=true remain
    # honoured without false positives from the real allowlist files.
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_scanner_flags_non_compliant_class(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("class Mimic(RuntimeError):\n    def to_audit_dict(self): return {}\n")
    # Use an empty allowlist dir so stale-entry detection is a no-op in this
    # test (no entries to be stale), which lets fail_on_stale=true remain
    # honoured without false positives from the real allowlist files.
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "to_audit_dict" in result.stdout
    assert "Mimic" in result.stdout


def test_scanner_flags_async_function_def(tmp_path: Path) -> None:
    # ``async def to_audit_dict`` without inheriting AuditEvidenceBase must be flagged.
    bad = tmp_path / "async_bad.py"
    bad.write_text("class AsyncMimic(RuntimeError):\n    async def to_audit_dict(self): return {}\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "to_audit_dict" in result.stdout
    assert "AsyncMimic" in result.stdout


def test_scanner_flags_lambda_assignment(tmp_path: Path) -> None:
    # ``to_audit_dict = lambda self: {}`` without inheriting AuditEvidenceBase must be flagged.
    bad = tmp_path / "lambda_bad.py"
    bad.write_text("class LambdaMimic(RuntimeError):\n    to_audit_dict = lambda self: {}\n")
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
    assert result.returncode != 0
    assert "to_audit_dict" in result.stdout
    assert "LambdaMimic" in result.stdout


def test_repo_errors_file_zero_emission_success_contract_violation_is_nominal() -> None:
    """Regression test for the live repo: overriding ``to_audit_dict`` must come
    with an explicit ``AuditEvidenceBase`` base so AEN1 can verify the contract
    from syntax alone rather than relying on transitive inheritance.
    """

    root = Path("src/elspeth")
    path = root / "contracts" / "errors.py"
    findings = scan_file(path, root)
    flagged = {finding.class_name for finding in findings}
    assert "ZeroEmissionSuccessContractViolation" not in flagged


def test_scanner_read_error_exits_nonzero(tmp_path: Path) -> None:
    # A file that cannot be read (chmod 000) must produce a non-zero exit with
    # "Fatal:" in stderr — not an unhandled traceback.
    unreadable = tmp_path / "unreadable.py"
    unreadable.write_text("class Fine: pass\n")
    os.chmod(unreadable, 0o000)
    allowlist_dir = tmp_path / "allowlist"
    allowlist_dir.mkdir()
    try:
        result = _run(["check", "--root", str(tmp_path), "--allowlist", str(allowlist_dir)])
        assert result.returncode != 0
        assert "Fatal:" in result.stderr
    finally:
        # Restore permissions so tmp_path cleanup succeeds.
        os.chmod(unreadable, 0o644)
