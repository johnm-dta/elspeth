"""Release-facing contracts for the standalone ELSPETH project website."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
WEBSITE = ROOT / "website"
PAGES = {
    "index.html": "index.html",
    "authoring.html": "authoring.html",
    "assurance.html": "assurance.html",
    "use-cases.html": "use-cases.html",
    "get-started.html": "get-started.html",
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup(_text(WEBSITE / name), "html.parser")


def test_pages_upload_action_is_recursively_sha_pinned() -> None:
    workflow = _text(ROOT / ".github/workflows/pages.yaml")

    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9  # v5.0.0" in workflow
    assert "actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa" not in workflow


def test_changelog_describes_release_boundaries_precisely() -> None:
    changelog = _text(ROOT / "CHANGELOG.md")
    release = changelog.split("## 0.7.0", maxsplit=1)[0]

    assert "before external publication" in release
    assert "before I/O" not in release
    assert "shared proposal and validation contract" in release
    assert "guided-staged" in release
    assert "elspeth export-resume <run-id> --execute" in release
    assert "sink-effect-v1" not in release


def test_every_page_has_description_favicon_and_current_navigation() -> None:
    favicon = WEBSITE / "favicon.svg"
    assert favicon.is_file()

    for name, current_href in PAGES.items():
        soup = _soup(name)
        description = soup.find("meta", attrs={"name": "description"})
        icon = soup.find("link", attrs={"rel": "icon"})
        current_links = soup.find_all("a", attrs={"aria-current": "page"})

        assert description is not None and description.get("content", "").strip(), name
        assert icon is not None and icon.get("href") == "favicon.svg", name
        assert len(current_links) == 2, name
        assert all(link.get("href") == current_href for link in current_links), name


def test_home_surfaces_both_releases_without_invented_counts() -> None:
    html = _text(WEBSITE / "index.html")

    assert "Current version: 0.7.1" in html
    assert "0.7.0" in html and "LLM-primary" in html
    assert "0.7.1" in html and "recoverable publication" in html.lower()
    assert "240 rows" not in html
    assert "pre-1.0" in html


def test_authoring_describes_current_staged_and_structural_capabilities() -> None:
    html = _text(WEBSITE / "authoring.html")

    assert "source → sink → transforms → wiring" in html
    assert "source → transforms → sink → wire" not in html
    assert "LLM-primary" in html
    assert "candidate remains separate" in html.lower()
    assert "cross-sink write-failure fallback" in html
    assert "require-all coalesce" in html
    assert "freeform and guided-full" in html
    assert "In the bundled" not in html
    assert "38ac0f55" not in html and "098ec06d" not in html
    for capability in ("structural queue", "fork", "coalesce", "text source", "text sink"):
        assert capability in html.lower()


def test_assurance_explains_recovery_without_claiming_bundled_evidence() -> None:
    html = _text(WEBSITE / "assurance.html")

    assert "Recoverable publication" in html
    assert "before external publication" in html
    assert "elspeth export-resume &lt;run-id&gt; --execute" in html
    assert "sink-effect-recovery.md" in html
    assert "bundled run" not in html.lower()
    assert "examples/*/runs/audit.db" not in html


def test_use_cases_are_illustrative_and_distinguish_extension_work() -> None:
    html = _text(WEBSITE / "use-cases.html")

    assert "Illustrative workflow patterns" in html
    assert "Built in" in html
    assert "custom plugins" in html
    assert "sub-second streaming engine" in html

    soup = _soup("use-cases.html")
    tables = soup.select(".table-scroll")
    assert tables
    assert all(table.get("tabindex") == "0" for table in tables)
    assert all(table.get("role") == "region" for table in tables)
    assert all(table.get("aria-label") for table in tables)


def test_get_started_has_runnable_cli_and_complete_composer_paths() -> None:
    html = _text(WEBSITE / "get-started.html")

    assert 'uv pip install -e ".[dev]"' in html
    assert 'uv pip install -e ".[webui,dev]"' in html
    assert "examples/threshold_gate/settings.yaml" in html
    assert "data/submissions.csv" not in html
    assert "npm install" in html and "npm run build" in html
    assert "ELSPETH_WEB__SECRET_KEY" in html
    assert "elspeth composer users add" in html and "--password" in html
    assert "SESSION_SCHEMA_EPOCH" in html and "26 → 36" in html
    assert "SQLITE_SCHEMA_EPOCH" in html and "22 → 29" in html
    assert "aws-ecs-deployment.md" in html


def test_theme_toggle_rerenders_replaced_lucide_icon() -> None:
    script = _text(WEBSITE / "site.js")
    set_icon = script.split("function setIcon()", maxsplit=1)[1].split("// --- copy", maxsplit=1)[0]

    assert "window.lucide.createIcons" in set_icon


def test_assurance_proof_uses_semantic_figure_markup() -> None:
    soup = _soup("assurance.html")
    proof = soup.find("figure", class_="term")

    assert proof is not None
    assert proof.find("figcaption") is not None
    assert not soup.find_all("h4")
