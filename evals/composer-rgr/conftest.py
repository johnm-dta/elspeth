"""Pytest fixtures for the composer-rgr differential eval suite.

Done-bar item 5 for composer-jit-hints Phase 1 (see
``.claude/plans/composer-llm-drifting-hollerith.md``, decision D6): the
``composer_hints_stripped`` fixture monkey-patches the local catalog to
strip ``composer_hints`` and ``post_call_hints`` so an eval scenario can
be re-run *without* the hints to prove the differential — GREEN with
hints, RED without.

Scope and contract
------------------

The fixtures only affect the *in-process* ``CatalogServiceImpl``. Any
eval scenario that drives a remote composer (e.g. the staging server
via ``run_scenario.sh``) is not affected — that path's catalog runs in
the staging process. The differential test path therefore drives the
composer in-process (via ``FastAPI`` ``TestClient`` against the local
app, or by calling the catalog/tool seams directly).

This file is collected by pytest only when invoked with a path under
``evals/`` (``pyproject.toml`` sets ``testpaths = ["tests"]``).
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Any

import pytest


@pytest.fixture
def composer_hints_populated() -> None:
    """Baseline: hints flow normally from plugin overrides into catalog responses.

    No-op fixture. Naming the contract on the test side is the point —
    a scenario that requests ``composer_hints_populated`` is signalling
    that the catalog must surface hints, which is the production
    behaviour after Phase 1. If that ever stops being true, the
    counterpart fixture ``composer_hints_stripped`` makes the inversion
    explicit rather than relying on a magic env var.
    """
    return None


@pytest.fixture
def composer_hints_stripped(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Strip composer_hints and post_call_hints from the local catalog dispatch.

    Patches the two CatalogServiceImpl seams that surface hints:

    * ``_discovery_composer_hints`` — used by ``_to_summary`` and
      ``_build_schema_info`` to populate ``PluginSummary.composer_hints``
      and ``PluginSchemaInfo.composer_hints``. Patched to return ``()``.
    * ``post_call_hints`` — used by ``_attach_post_call_hints`` to
      populate ``ToolResult.post_call_hints`` after a successful
      mutation. Patched to return ``()``.

    Both surfaces are patched together because a scenario testing
    "what does an LLM do without hints?" needs *all* hint channels
    suppressed; leaking either surface would let the LLM recover.

    Scope is function-level (per test), and the patch is unwound when
    the test exits — there is no module/session leakage.
    """
    from elspeth.web.catalog.service import CatalogServiceImpl

    def _no_hints(self: CatalogServiceImpl, plugin_cls: Any) -> tuple[str, ...]:
        return ()

    def _no_post_call_hints(
        self: CatalogServiceImpl,
        *,
        plugin_type: str,
        plugin_name: str,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr(
        CatalogServiceImpl,
        "_discovery_composer_hints",
        _no_hints,
        raising=True,
    )
    monkeypatch.setattr(
        CatalogServiceImpl,
        "post_call_hints",
        _no_post_call_hints,
        raising=True,
    )
    yield None
