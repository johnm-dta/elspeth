"""Helpers for validating plugin declaration flags."""

from __future__ import annotations

from typing import Any


def _require_bool_flag(plugin: Any, *, attr_name: str) -> bool:
    """Return a declaration flag only when it is an exact ``bool``."""

    value = getattr(plugin, attr_name)
    if type(value) is not bool:
        raise TypeError(f"{type(plugin).__name__}.{attr_name} must be bool, got {type(value).__name__!r}.")
    return value
