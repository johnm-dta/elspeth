"""Regression tests for the telemetry-backfill range checker allowlist."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / ".githooks" / "check-commit-range-telemetry-backfill.sh"


def _run(cmd: list[str], cwd: Path, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init"], repo)
    _run(["git", "config", "user.email", "ci@example.invalid"], repo)
    _run(["git", "config", "user.name", "CI Test"], repo)
    (repo / ".githooks").mkdir()
    (repo / ".githooks" / "check-commit-range-telemetry-backfill.sh").write_text(
        SCRIPT.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / ".githooks" / "check-commit-range-telemetry-backfill.sh").chmod(0o755)
    return repo


def _commit_shareable_review_without_trailer(repo: Path) -> str:
    target = repo / "src" / "elspeth" / "web" / "shareable_reviews" / "service.py"
    target.parent.mkdir(parents=True)
    target.write_text("TOKEN = 'v1'\n", encoding="utf-8")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-m", "touch shareable review"], repo)
    return _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()


def test_range_checker_honors_active_allowlist_entry(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "test-exemption.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: {violating_sha}
                cohort: a
                reason: |
                  Test fixture: this commit intentionally omits the trailer
                  so the range checker proves it honors active allowlists.
                owner: codex
                expires: 2099-01-01
            """
        ),
        encoding="utf-8",
    )

    proc = _run([".githooks/check-commit-range-telemetry-backfill.sh", f"{base}..HEAD"], repo)

    assert proc.returncode == 0
    assert "allowlisted" in proc.stdout


def test_range_checker_treats_expired_allowlist_entry_as_absent(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "expired-exemption.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: {violating_sha}
                cohort: a
                reason: |
                  Test fixture: expired entries must not suppress enforcement.
                owner: codex
                expires: 2000-01-01
            """
        ),
        encoding="utf-8",
    )

    proc = _run([".githooks/check-commit-range-telemetry-backfill.sh", f"{base}..HEAD"], repo, check=False)

    assert proc.returncode == 1
    assert "lacks the required trailer" in proc.stdout


def test_range_checker_rejects_allowlist_entry_without_owner(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "missing-owner.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: {violating_sha}
                cohort: a
                reason: legitimate exception
                expires: 2099-01-01
            """
        ),
        encoding="utf-8",
    )

    proc = _run([".githooks/check-commit-range-telemetry-backfill.sh", f"{base}..HEAD"], repo, check=False)

    assert proc.returncode == 1
    assert "owner is required" in proc.stderr


def test_range_checker_rejects_allowlist_entry_without_reason(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "missing-reason.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: {violating_sha}
                cohort: a
                owner: codex
                expires: 2099-01-01
            """
        ),
        encoding="utf-8",
    )

    proc = _run([".githooks/check-commit-range-telemetry-backfill.sh", f"{base}..HEAD"], repo, check=False)

    assert proc.returncode == 1
    assert "reason is required" in proc.stderr


def test_range_checker_rejects_empty_block_scalar_reason(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "empty-block-reason.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: {violating_sha}
                cohort: a
                reason: |
                owner: codex
                expires: 2099-01-01
            """
        ),
        encoding="utf-8",
    )

    proc = _run([".githooks/check-commit-range-telemetry-backfill.sh", f"{base}..HEAD"], repo, check=False)

    assert proc.returncode == 1
    assert "reason is required" in proc.stderr
