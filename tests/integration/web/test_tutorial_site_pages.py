"""Phase p4 — the 3 synthetic tutorial-site pages (Component 1 of the spec).

Reads the SOURCE files under website/tutorial-site/, which is the GitHub Pages
publish tree. The pages must be unmistakably marked test data, noindexed, and
carry three tables whose values DIFFER across the three projects so the derived
facts vary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_WEBSITE = _ROOT / "website/tutorial-site"
_FRONTEND_PUBLIC = _ROOT / "src/elspeth/web/frontend/public/tutorial-site"
_PAGES = ("project-1.html", "project-2.html", "project-3.html")


@pytest.mark.parametrize("name", _PAGES)
def test_synthetic_page_is_marked_test_data(name: str) -> None:
    html = (_WEBSITE / name).read_text(encoding="utf-8")
    assert "SYNTHETIC TEST DATA ONLY — DO NOT USE" in html
    # Match either the self-closing (' />') or plain ('>') form so the
    # assertion agrees with the XML self-closing fixtures below.
    assert 'content="noindex"' in html


@pytest.mark.parametrize("name", _PAGES)
def test_synthetic_page_has_three_tables(name: str) -> None:
    html = (_WEBSITE / name).read_text(encoding="utf-8").lower()
    # Risk register / schedule / cost breakdown headings.
    assert "risk register" in html
    assert "schedule" in html
    assert "cost breakdown" in html
    # The cost table must be summable (>= 3 explicit dollar figures).
    assert html.count("$") >= 3


@pytest.mark.parametrize("name", _PAGES)
def test_frontend_public_tree_does_not_duplicate_tutorial_page(name: str) -> None:
    assert not (_FRONTEND_PUBLIC / name).exists()


def test_synthetic_pages_have_distinct_cost_totals() -> None:
    # The whole point of differing values: the derived total_cost must vary.
    import re

    totals: list[int] = []
    for name in _PAGES:
        html = (_WEBSITE / name).read_text(encoding="utf-8")
        figures = [int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", html)]
        assert figures, f"{name} has no dollar figures"
        totals.append(sum(figures))
    assert len(set(totals)) == 3, f"cost totals must differ across pages, got {totals}"


def test_synthetic_pages_have_distinct_go_live_dates() -> None:
    # The derived key_date must vary: every page's Go-live row carries a
    # different ISO date. Steps 3-4 copy project-1 verbatim and change ONLY the
    # values, so this guards a forgotten date edit at CI (cheap) instead of at
    # the expensive staging judge run.
    import re

    dates: list[str] = []
    for name in _PAGES:
        html = (_WEBSITE / name).read_text(encoding="utf-8")
        m = re.search(r"Go-live</td><td>(\d{4}-\d{2}-\d{2})</td>", html)
        assert m, f"{name} has no Go-live date row"
        dates.append(m.group(1))
    assert len(set(dates)) == 3, f"go-live dates must differ across pages, got {dates}"


def test_synthetic_pages_have_distinct_project_names() -> None:
    # The derived project_name must vary: each hero <h1> names a distinct
    # project. Guards a forgotten title edit (same copy-verbatim risk).
    import re

    names: list[str] = []
    for name in _PAGES:
        html = (_WEBSITE / name).read_text(encoding="utf-8")
        m = re.search(r"<h1>([^<]+)</h1>", html)
        assert m, f"{name} has no hero <h1>"
        names.append(m.group(1).strip())
    assert len(set(names)) == 3, f"project names must differ across pages, got {names}"
