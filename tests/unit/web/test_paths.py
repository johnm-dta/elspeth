"""Tests for shared path resolution helper."""

from __future__ import annotations

from pathlib import Path

from elspeth.web.paths import resolve_data_path


class TestResolveDataPath:
    """Unit tests for resolve_data_path — the single resolution function
    used by both validation and execution service."""

    def test_relative_path_resolved_against_data_dir(self) -> None:
        """A relative path is joined to data_dir before resolving."""
        result = resolve_data_path("blobs/data.csv", "/tmp/data")
        assert result == Path("/tmp/data/blobs/data.csv")

    def test_absolute_path_unchanged(self) -> None:
        """An absolute path resolves to itself (no data_dir involvement)."""
        result = resolve_data_path("/etc/passwd", "/tmp/data")
        assert result == Path("/etc/passwd")

    def test_traversal_resolved(self) -> None:
        """Traversal (../) is resolved by the OS — blocking is the allowlist's job."""
        result = resolve_data_path("../etc/passwd", "/tmp/data")
        assert result == Path("/tmp/etc/passwd")
