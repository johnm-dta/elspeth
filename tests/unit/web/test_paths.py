"""Tests for shared path resolution helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.web.paths import allowed_sink_directories, resolve_data_path


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


class TestAllowedSinkDirectories:
    """Session-scoped sink allowlist (elspeth-bdc17cfdb1).

    The blobs entry is confined to the caller's own session subtree —
    blob storage is laid out as ``blobs/<session_id>/<blob_id>_<name>``
    and the sink side must not be able to address another session's
    subtree. ``outputs`` remains the shared flat pool (recipe defaults
    like ``outputs/classified.jsonl`` resolve there).
    """

    def test_blobs_entry_is_session_scoped(self) -> None:
        result = allowed_sink_directories("/tmp/data", session_id="sess-a")
        assert result == (
            Path("/tmp/data/outputs"),
            Path("/tmp/data/blobs/sess-a"),
        )

    def test_none_session_fails_closed_to_outputs_only(self) -> None:
        """No session identity => no blob access at all, not broad access."""
        result = allowed_sink_directories("/tmp/data", session_id=None)
        assert result == (Path("/tmp/data/outputs"),)

    def test_other_sessions_subtree_is_not_contained(self) -> None:
        """The boundary this exists for: session A's allowlist must not
        contain session B's blob subtree."""
        allowed = allowed_sink_directories("/tmp/data", session_id="sess-a")
        foreign = Path("/tmp/data/blobs/sess-b/out.csv")
        assert not any(foreign.is_relative_to(d) for d in allowed)

    def test_blobs_root_is_not_contained(self) -> None:
        """A file directly under blobs/ (no session segment) is outside
        every session's allowlist."""
        allowed = allowed_sink_directories("/tmp/data", session_id="sess-a")
        assert not any(Path("/tmp/data/blobs/loose.csv").is_relative_to(d) for d in allowed)

    @pytest.mark.parametrize(
        "bad_session_id",
        ["", "a/b", "a\\b", "..", ".", "sess/../other"],
    )
    def test_malformed_session_id_rejected(self, bad_session_id: str) -> None:
        """A session id that could alter the path shape is a contract
        violation at this boundary — raise, never silently widen."""
        with pytest.raises(ValueError, match="session_id"):
            allowed_sink_directories("/tmp/data", session_id=bad_session_id)
