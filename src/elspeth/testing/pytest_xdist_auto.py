# src/elspeth/testing/pytest_xdist_auto.py
"""Pytest plugin: auto-enable xdist parallel execution.

Registered via the ``pytest11`` entry point so it participates in hook
dispatch alongside installed plugins (including xdist itself).

Behaviour:
- **Controller process:** defaults to ``-n auto`` when no ``-n``
  flag is given explicitly.  Override with ``-n0`` or ``-n <count>``.
- **Coverage run:** no-op. The current pytest-cov controller does not support
  this auto-xdist path.
- **xdist worker process:** no-op. Workers inherit their controller's plan and
  must not recursively spawn more workers.
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
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    if getattr(config.option, "cov_source", None):
        return

    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses is None:
        config.option.numprocesses = "auto"
