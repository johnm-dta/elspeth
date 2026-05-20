"""Regression checks for public community-health and disclosure entrypoints."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_required_community_health_files_exist() -> None:
    for relative_path in ("SECURITY.md", "SUPPORT.md", "GOVERNANCE.md", "CODE_OF_CONDUCT.md"):
        assert (REPO_ROOT / relative_path).is_file(), relative_path


def test_security_policy_routes_sensitive_reports_away_from_public_issues() -> None:
    security = _read("SECURITY.md")

    assert "Do not open a public GitHub issue containing exploit details" in security
    assert "GitHub private vulnerability reporting" in security
    assert "Security disclosure path requested" in security


def test_contributing_and_readme_point_to_security_policy() -> None:
    contributing = _read("CONTRIBUTING.md")
    readme = _read("README.md")

    assert "For security-sensitive issues, follow [SECURITY.md](SECURITY.md)" in contributing
    assert "Do not include exploit details, secrets, personal data, or sensitive audit material in a public issue." in contributing
    assert "Report suspected vulnerabilities through [SECURITY.md](SECURITY.md)" in readme
    assert "[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)" in readme
    assert "[SUPPORT.md](SUPPORT.md)" in readme
    assert "[GOVERNANCE.md](GOVERNANCE.md)" in readme
