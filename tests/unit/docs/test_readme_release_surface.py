"""Regression checks for the public README release surface."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
README = REPO_ROOT / "README.md"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_advertises_rc53_release_surface() -> None:
    text = _readme_text()

    assert "Status: RC-5.3" in text
    assert "status-RC--5.3" in text
    assert "- [RC-5.3 Updates](#rc-53-updates)" in text
    assert "### RC-5.3 Updates" in text
    assert "[Progress Report: RC-1 to RC-5](docs/release/elspeth-progress-rc1-to-rc5.md)" in text
    assert "[Velocity Report: RC-1 to RC-5](docs/release/elspeth-velocity-rc1-to-rc5.md)" in text
    assert "[Audit and Lineage Guarantees](docs/release/guarantees.md)" in text
    assert "[Release Documentation Index](docs/release/README.md)" in text
    assert "[docs/release/](docs/release/)" in text


def test_readme_release_links_resolve() -> None:
    text = _readme_text()
    release_links = {
        "docs/release/executive-summary.md",
        "docs/release/elspeth-progress-rc1-to-rc5.md",
        "docs/release/elspeth-velocity-rc1-to-rc5.md",
        "docs/release/guarantees.md",
        "docs/release/README.md",
    }
    linked_paths = set(re.findall(r"\]\((docs/release/[^)#]+)", text))

    assert release_links <= linked_paths
    for relative_path in release_links:
        assert (REPO_ROOT / relative_path).exists(), relative_path
