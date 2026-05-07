"""Process-local plugin-instantiation preflight mode."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_PLUGIN_PREFLIGHT_MODE: ContextVar[bool] = ContextVar("elspeth_plugin_preflight_mode", default=False)


def plugin_preflight_mode_enabled() -> bool:
    """Return True while plugins are being instantiated for runtime preflight."""
    return _PLUGIN_PREFLIGHT_MODE.get()


@contextmanager
def plugin_preflight_mode(enabled: bool) -> Iterator[None]:
    token = _PLUGIN_PREFLIGHT_MODE.set(enabled)
    try:
        yield
    finally:
        _PLUGIN_PREFLIGHT_MODE.reset(token)
