"""Test helpers for observing validation-only plugin lifecycle ownership."""

from __future__ import annotations

from typing import Any


class TrackedTransform:
    """Transparent transform proxy that records every close call."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.close_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def close(self) -> None:
        self.close_count += 1
        self._delegate.close()


class TrackingPluginManager:
    """Delegate plugin-manager operations while wrapping created transforms."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.instances: list[TrackedTransform] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def create_transform(self, plugin_name: str, options: dict[str, Any]) -> TrackedTransform:
        tracked = TrackedTransform(self._delegate.create_transform(plugin_name, options))
        self.instances.append(tracked)
        return tracked
