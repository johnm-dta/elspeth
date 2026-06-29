"""Regression tests for the telemetry-backfill range checker allowlist."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / ".githooks" / "check-commit-range-telemetry-backfill.sh"
COMMIT_MSG_HOOK = REPO_ROOT / ".githooks" / "commit-msg-telemetry-backfill"
ALLOWLIST_README = REPO_ROOT / "config" / "cicd" / "enforce_telemetry_backfill_trailer" / "README.md"


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
    (repo / ".githooks" / "commit-msg-telemetry-backfill").write_text(
        COMMIT_MSG_HOOK.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo / ".githooks" / "commit-msg-telemetry-backfill").chmod(0o755)
    return repo


def _stage_shareable_review(repo: Path) -> None:
    target = repo / "src" / "elspeth" / "web" / "shareable_reviews" / "service.py"
    target.parent.mkdir(parents=True)
    target.write_text("TOKEN = 'v1'\n", encoding="utf-8")
    _run(["git", "add", "."], repo)


def _commit_shareable_review_without_trailer(repo: Path) -> str:
    _stage_shareable_review(repo)
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


def test_range_checker_ignores_entries_inside_block_scalar_reason(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _run(["git", "commit", "--allow-empty", "-m", "base"], repo)
    base = _run(["git", "rev-parse", "HEAD"], repo).stdout.strip()
    violating_sha = _commit_shareable_review_without_trailer(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "block-scalar-smuggling.yaml").write_text(
        dedent(
            f"""\
            entries:
              - commit_sha: 1111111111111111111111111111111111111111
                cohort: a
                owner: codex
                expires: 2099-01-01
                reason: |-
                  This human-readable reason embeds text that looks like a
                  second allowlist entry, but it must remain block-scalar text.
                  - commit_sha: {violating_sha}
                    cohort: a
                    reason: forged nested entry
                    owner: attacker
                    expires: 2099-01-01
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


def test_sha_allowlist_contract_is_documented_as_ci_only() -> None:
    readme = ALLOWLIST_README.read_text(encoding="utf-8")
    hook = COMMIT_MSG_HOOK.read_text(encoding="utf-8")

    assert "CI-only SHA allowlist" in readme
    assert "The local commit-msg hook cannot consume this SHA allowlist" in readme
    assert "The hook and the CI backstop both read" not in readme

    assert "The SHA allowlist is CI-only" in hook
    assert "a per-cohort YAML file may name commits" not in hook


def test_local_commit_msg_hook_reports_sha_allowlist_as_ci_only(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _stage_shareable_review(repo)

    allowlist_dir = repo / "config" / "cicd" / "enforce_telemetry_backfill_trailer"
    allowlist_dir.mkdir(parents=True)
    (allowlist_dir / "local-exemption.yaml").write_text(
        dedent(
            """\
            entries:
              - commit_sha: 1111111111111111111111111111111111111111
                cohort: a
                reason: |
                  This entry cannot apply locally because the commit SHA is
                  unavailable while the commit-msg hook runs.
                owner: codex
                expires: 2099-01-01
            """
        ),
        encoding="utf-8",
    )
    commit_msg = repo / "COMMIT_EDITMSG"
    commit_msg.write_text("touch shareable review\n", encoding="utf-8")

    proc = _run([".githooks/commit-msg-telemetry-backfill", str(commit_msg)], repo, check=False)

    assert proc.returncode == 1
    assert "SHA allowlist exemptions are CI-only" in proc.stderr
    assert "telemetry-backfill: shareable-reviews" in proc.stderr
