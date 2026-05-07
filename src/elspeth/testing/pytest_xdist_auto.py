# src/elspeth/testing/pytest_xdist_auto.py
"""Pytest plugin: auto-enable xdist parallel execution for local runs.

Registered via the ``pytest11`` entry point so it participates in hook
dispatch alongside installed plugins (including xdist itself).

Behaviour:
- **Local (no ``CI`` env var):** defaults to ``-n auto`` when no ``-n``
  flag is given explicitly.  Override with ``-n0`` or ``-n <count>``.
- **CI (``CI=true``):** no-op — tests run sequentially for clearer
  failure output and deterministic coverage collection.
"""

from __future__ import annotations

import os

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Set ``numprocesses = "auto"`` before xdist resolves it.

    Uses ``tryfirst`` so this runs before xdist's own
    ``pytest_cmdline_main(tryfirst=True)`` — both are ``tryfirst``,
    but ours is registered via entry point at install time, giving
    pluggy's LIFO-within-priority ordering control.
    """
    # xdist workers set PYTEST_XDIST_WORKER in child processes.
    # Without this guard, the plugin fork-bombs: each worker loads
    # the entry point, sets -n auto, spawns more workers, repeat.
    if os.environ.get("CI") or os.environ.get("PYTEST_XDIST_WORKER"):
        return

    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses is None:
        config.option.numprocesses = "auto"
