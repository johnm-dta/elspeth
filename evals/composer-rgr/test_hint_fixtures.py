"""Smoke test proving the hint-strip fixtures actually strip the catalog.

A fixture that silently fails to apply its patch would leave the
differential test useless — both scenarios would behave the same and
the scorer would report misleading GREEN/RED states. This file pins
the fixtures' behaviour so any regression (e.g. someone renames
``_discovery_composer_hints`` and the monkeypatch target goes stale —
``raising=True`` will catch it on test run) shows up here.

This test is NOT part of the default ``pytest`` collection
(``pyproject.toml`` sets ``testpaths = ["tests"]``). Run with:

    .venv/bin/python -m pytest evals/composer-rgr/test_hint_fixtures.py
"""

from __future__ import annotations

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
from elspeth.web.catalog.service import CatalogServiceImpl


def test_baseline_catalog_surfaces_composer_hints(composer_hints_populated: None) -> None:
    """Sanity: without the strip fixture, the csv source publishes hints."""
    catalog = CatalogServiceImpl(get_shared_plugin_manager())
    summaries = {entry.name: entry for entry in catalog.list_sources()}
    assert summaries["csv"].composer_hints, (
        "Baseline catalog must publish csv composer_hints — Phase 1's "
        "csv_source.get_agent_assistance(issue_code=None) override is "
        "missing or unreachable."
    )


def test_stripped_catalog_drops_composer_hints(composer_hints_stripped: None) -> None:
    """With the strip fixture applied, the listing surface returns empty hints."""
    catalog = CatalogServiceImpl(get_shared_plugin_manager())
    for entry in catalog.list_sources():
        assert entry.composer_hints == (), (
            f"source/{entry.name} still carries composer_hints under "
            "composer_hints_stripped — the monkeypatch did not take "
            "effect on _discovery_composer_hints."
        )
    for entry in catalog.list_transforms():
        assert entry.composer_hints == ()
    for entry in catalog.list_sinks():
        assert entry.composer_hints == ()


def test_stripped_catalog_drops_post_call_hints(composer_hints_stripped: None) -> None:
    """With the strip fixture applied, the post-call dispatch returns empty hints for an otherwise-triggering config."""
    catalog = CatalogServiceImpl(get_shared_plugin_manager())
    # csv source with schema.mode='fixed' triggers a hint in production;
    # under the strip fixture it must come back empty.
    hints = catalog.post_call_hints(
        plugin_type="source",
        plugin_name="csv",
        tool_name="set_source",
        config_snapshot={"schema": {"mode": "fixed"}},
    )
    assert hints == (), (
        "csv source still emitted post_call_hints under composer_hints_stripped — the monkeypatch did not take effect on post_call_hints."
    )
