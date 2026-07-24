"""Tests for shared path resolution helper."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.web.paths import (
    allowed_sink_directories,
    allowed_source_directories,
    resolve_data_path,
    resolve_sink_data_path,
)


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
    subtree. Logical paths such as ``outputs/classified.jsonl`` resolve into
    the caller's session-scoped outputs subtree.
    """

    def test_blobs_entry_is_session_scoped(self) -> None:
        result = allowed_sink_directories("/tmp/data", session_id="sess-a")
        assert result == (
            Path("/tmp/data/outputs/sess-a"),
            Path("/tmp/data/blobs/sess-a"),
        )

    def test_none_session_fails_closed_to_no_local_paths(self) -> None:
        """No session identity means neither shared-root reads nor writes."""
        result = allowed_sink_directories("/tmp/data", session_id=None)
        assert result == ()

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

    def test_other_sessions_output_subtree_is_not_contained(self) -> None:
        allowed = allowed_sink_directories("/tmp/data", session_id="sess-a")
        foreign = Path("/tmp/data/outputs/sess-b/out.csv")
        assert not any(foreign.is_relative_to(directory) for directory in allowed)

    @pytest.mark.parametrize(
        "bad_session_id",
        ["", "a/b", "a\\b", "..", ".", "sess/../other"],
    )
    def test_malformed_session_id_rejected(self, bad_session_id: str) -> None:
        """A session id that could alter the path shape is a contract
        violation at this boundary — raise, never silently widen."""
        with pytest.raises(ValueError, match="session_id"):
            allowed_sink_directories("/tmp/data", session_id=bad_session_id)


class TestSessionOwnedLocalPaths:
    def test_source_allowlist_is_session_scoped(self) -> None:
        assert allowed_source_directories("/tmp/data", session_id="sess-a") == (Path("/tmp/data/blobs/sess-a"),)

    def test_source_allowlist_without_session_fails_closed(self) -> None:
        assert allowed_source_directories("/tmp/data", session_id=None) == ()

    def test_logical_output_path_resolves_inside_session_subtree(self) -> None:
        resolved = resolve_sink_data_path("outputs/report.csv", "/tmp/data", session_id="sess-a")
        assert resolved == Path("/tmp/data/outputs/sess-a/report.csv")

    def test_already_scoped_output_path_is_not_double_prefixed(self) -> None:
        resolved = resolve_sink_data_path("outputs/sess-a/report.csv", "/tmp/data", session_id="sess-a")
        assert resolved == Path("/tmp/data/outputs/sess-a/report.csv")

    def test_foreign_absolute_output_is_outside_session_allowlist(self) -> None:
        resolved = resolve_sink_data_path("/tmp/data/outputs/sess-b/report.csv", "/tmp/data", session_id="sess-a")
        allowed = allowed_sink_directories("/tmp/data", session_id="sess-a")
        assert not any(resolved.is_relative_to(directory) for directory in allowed)

    def test_foreign_relative_session_output_is_not_adopted(self) -> None:
        own_session = "11111111-1111-4111-8111-111111111111"
        foreign_session = "22222222-2222-4222-8222-222222222222"
        resolved = resolve_sink_data_path(
            f"outputs/{foreign_session}/report.csv",
            "/tmp/data",
            session_id=own_session,
        )
        allowed = allowed_sink_directories("/tmp/data", session_id=own_session)

        assert resolved == Path(f"/tmp/data/outputs/{foreign_session}/report.csv")
        assert not any(resolved.is_relative_to(directory) for directory in allowed)

    def test_nested_logical_output_path_is_scoped_to_session(self) -> None:
        resolved = resolve_sink_data_path("outputs/reports/daily.csv", "/tmp/data", session_id="sess-a")
        assert resolved == Path("/tmp/data/outputs/sess-a/reports/daily.csv")

    def test_symlink_from_owned_output_to_foreign_output_is_outside_allowlist(self, tmp_path: Path) -> None:
        own_root = tmp_path / "outputs" / "sess-a"
        foreign_root = tmp_path / "outputs" / "sess-b"
        own_root.mkdir(parents=True)
        foreign_root.mkdir(parents=True)
        (own_root / "foreign-link").symlink_to(foreign_root, target_is_directory=True)

        resolved = resolve_sink_data_path("outputs/foreign-link/report.csv", str(tmp_path), session_id="sess-a")
        allowed = allowed_sink_directories(str(tmp_path), session_id="sess-a")

        assert not any(resolved.is_relative_to(directory) for directory in allowed)
