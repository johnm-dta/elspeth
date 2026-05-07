"""Tests for composer MCP session persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from elspeth.composer_mcp.session import (
    InvalidSessionIdError,
    SessionManager,
    SessionNotFoundError,
)
from elspeth.web.composer.state import SourceSpec


class TestSessionManager:
    @pytest.fixture()
    def scratch_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "scratch"

    @pytest.fixture()
    def manager(self, scratch_dir: Path) -> SessionManager:
        return SessionManager(scratch_dir)

    def test_new_session_returns_empty_state(self, manager: SessionManager) -> None:
        session_id, state = manager.new_session()
        assert session_id
        assert state.source is None
        assert state.nodes == ()
        assert state.edges == ()
        assert state.outputs == ()
        assert state.version == 1

    def test_new_session_with_name(self, manager: SessionManager) -> None:
        _session_id, state = manager.new_session(name="my-pipeline")
        assert state.metadata.name == "my-pipeline"

    def test_save_and_load_round_trip(self, manager: SessionManager) -> None:
        session_id, state = manager.new_session()
        source = SourceSpec(
            plugin="csv",
            on_success="source_out",
            options={"path": "data/input.csv"},
            on_validation_failure="discard",
        )
        state = state.with_source(source)
        manager.save(session_id, state)
        loaded = manager.load(session_id)
        assert loaded.source is not None
        assert loaded.source.plugin == "csv"
        assert loaded.source.on_success == "source_out"
        assert loaded.version == state.version

    def test_load_nonexistent_raises(self, manager: SessionManager) -> None:
        # 12 lowercase hex — valid shape, just not on disk. The shape check
        # runs at the _session_path chokepoint, so a malformed id would
        # raise InvalidSessionIdError before we can test missing-file
        # behaviour; a valid-shape id exercises the file-existence path.
        missing = "0" * 12
        with pytest.raises(SessionNotFoundError, match=missing):
            manager.load(missing)

    def test_delete_nonexistent_before_scratch_exists_raises_session_not_found(self, manager: SessionManager) -> None:
        missing = "0" * 12
        with pytest.raises(SessionNotFoundError, match=missing):
            manager.delete(missing)

    def test_list_sessions_empty(self, manager: SessionManager) -> None:
        assert manager.list_sessions() == []

    def test_list_sessions_after_save(self, manager: SessionManager) -> None:
        sid1, s1 = manager.new_session(name="first")
        manager.save(sid1, s1)
        sid2, s2 = manager.new_session(name="second")
        manager.save(sid2, s2)
        sessions = manager.list_sessions()
        assert len(sessions) == 2
        names = {s["name"] for s in sessions}
        assert names == {"first", "second"}

    def test_save_creates_scratch_dir(self, tmp_path: Path) -> None:
        scratch = tmp_path / "nonexistent" / "scratch"
        manager = SessionManager(scratch)
        sid, state = manager.new_session()
        manager.save(sid, state)
        assert scratch.exists()

    def test_delete_session(self, manager: SessionManager) -> None:
        sid, state = manager.new_session()
        manager.save(sid, state)
        manager.delete(sid)
        with pytest.raises(SessionNotFoundError):
            manager.load(sid)

    def test_saved_file_is_valid_json(self, manager: SessionManager, scratch_dir: Path) -> None:
        sid, state = manager.new_session(name="json-check")
        manager.save(sid, state)
        files = list(scratch_dir.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["metadata"]["name"] == "json-check"

    def test_save_rejects_stale_version_and_preserves_newer_state(self, manager: SessionManager) -> None:
        sid, original = manager.new_session(name="original")
        manager.save(sid, original)
        newer = original.with_metadata({"name": "newer"})
        manager.save(sid, newer)

        with pytest.raises(ValueError, match="stale"):
            manager.save(sid, original)

        loaded = manager.load(sid)
        assert loaded.metadata.name == "newer"
        assert loaded.version == newer.version

    def test_save_pre_replace_failure_preserves_prior_file(
        self,
        manager: SessionManager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sid, original = manager.new_session(name="before")
        manager.save(sid, original)
        updated = original.with_metadata({"name": "after"})

        def fail_replace(_self: Path, _target: Path) -> None:
            raise OSError("replace failed")

        monkeypatch.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="replace failed"):
            manager.save(sid, updated)

        loaded = manager.load(sid)
        assert loaded.metadata.name == "before"
        assert loaded.version == original.version

    def test_list_sessions_skips_invalid_filename_even_with_valid_payload(
        self,
        manager: SessionManager,
        scratch_dir: Path,
    ) -> None:
        sid, state = manager.new_session(name="canonical")
        manager.save(sid, state)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / "legacy.backup.json").write_text(json.dumps(state.to_dict()))

        sessions = manager.list_sessions()

        assert {session["session_id"] for session in sessions} == {sid}

    def test_list_sessions_skips_valid_shape_filename_with_malformed_json(
        self,
        manager: SessionManager,
        scratch_dir: Path,
    ) -> None:
        sid, state = manager.new_session(name="canonical")
        manager.save(sid, state)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / "000000000000.json").write_text("{not valid json")

        sessions = manager.list_sessions()

        assert {session["session_id"] for session in sessions} == {sid}

    def test_list_sessions_skips_valid_shape_filename_missing_required_fields(
        self,
        manager: SessionManager,
        scratch_dir: Path,
    ) -> None:
        sid, state = manager.new_session(name="canonical")
        manager.save(sid, state)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        (scratch_dir / "111111111111.json").write_text(
            json.dumps(
                {
                    "metadata": {},
                    "source": None,
                    "nodes": [],
                    "edges": [],
                    "outputs": [],
                }
            )
        )

        sessions = manager.list_sessions()

        assert {session["session_id"] for session in sessions} == {sid}


class TestSessionIdValidation:
    """Tier-3 boundary guards: session_id comes from LLM-controlled MCP args.

    The MCP server passes ``arguments["session_id"]`` straight through to
    ``SessionManager``. Without validation at this boundary, an attacker can
    supply absolute paths or traversal sequences to coerce ``_session_path``
    into arbitrary filesystem targets — a CWE-22 path traversal. These tests
    fix the contract: any session_id that did not come from ``new_session``
    (which emits ``uuid.uuid4().hex[:12]``) is rejected outright.
    """

    @pytest.fixture()
    def scratch_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "scratch"

    @pytest.fixture()
    def manager(self, scratch_dir: Path) -> SessionManager:
        return SessionManager(scratch_dir)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "",
            ".",
            "..",
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/passwd",
            "/tmp/elspeth-pwn",
            "C:\\Windows\\System32\\config",
            "abc/def",
            "abc\\def",
            "abc def",
            "abcdef",  # 6 chars — too short
            "abcdef0123456",  # 13 chars — too long
            "ABCDEF012345",  # uppercase — new_session emits lowercase
            "abcdef01234g",  # non-hex char 'g'
            "abcdef01234.",  # trailing dot
            ".abcdef01234",  # leading dot
            "\x00bcdef01234",  # embedded NUL
            "abcdef01234\n",  # trailing newline
            # Fullwidth-hex homoglyphs (U+FF21..U+FF26, U+FF10..U+FF15) —
            # visually resemble "ABCDEF012345" but are not ASCII and must
            # be rejected by the regex.
            "\uff21\uff22\uff23\uff24\uff25\uff26\uff10\uff11\uff12\uff13\uff14\uff15",
        ],
    )
    def test_save_rejects_malformed_session_id(self, manager: SessionManager, bad_id: str) -> None:
        _sid, state = manager.new_session()
        with pytest.raises(InvalidSessionIdError):
            manager.save(bad_id, state)

    @pytest.mark.parametrize("bad_id", ["", "..", "/etc/passwd", "abc/def", "abcdef"])
    def test_load_rejects_malformed_session_id(self, manager: SessionManager, bad_id: str) -> None:
        with pytest.raises(InvalidSessionIdError):
            manager.load(bad_id)

    @pytest.mark.parametrize("bad_id", ["", "..", "/etc/passwd", "abc/def", "abcdef"])
    def test_delete_rejects_malformed_session_id(self, manager: SessionManager, bad_id: str) -> None:
        with pytest.raises(InvalidSessionIdError):
            manager.delete(bad_id)

    def test_invalid_session_id_error_is_value_error(self) -> None:
        """MCP server catches ValueError at server.py:375 and returns a
        clean isError response. Subclassing ValueError preserves that flow
        while giving tests a specific type to assert on.
        """
        assert issubclass(InvalidSessionIdError, ValueError)

    def test_save_with_absolute_path_does_not_write_outside_scratch(
        self,
        manager: SessionManager,
        scratch_dir: Path,
        tmp_path: Path,
    ) -> None:
        """Proof-of-exploit regression: a pre-fix attacker could write to any
        *.json path. Confirm the target stays untouched and the scratch dir
        has no new files (the write must fail before any filesystem action).
        """
        _sid, state = manager.new_session()
        sentinel = tmp_path / "sentinel"
        sentinel.write_text("original")
        # Absolute-path override: Path("scratch") / "/..." discards "scratch"
        malicious = str(sentinel)  # "/.../sentinel" → write target "/.../sentinel.json"
        with pytest.raises(InvalidSessionIdError):
            manager.save(malicious, state)
        assert sentinel.read_text() == "original"
        assert not (tmp_path / "sentinel.json").exists()
        # No scratch dir created from an attack attempt.
        assert not scratch_dir.exists() or list(scratch_dir.glob("*.json")) == []

    def test_delete_with_traversal_does_not_remove_external_file(
        self,
        manager: SessionManager,
        tmp_path: Path,
    ) -> None:
        """Delete primitive must be neutralised — the target file survives."""
        target = tmp_path / "keepme.json"
        target.write_text('{"keep": true}')
        with pytest.raises(InvalidSessionIdError):
            manager.delete(str(target.with_suffix("")))
        assert target.exists()

    def test_load_with_traversal_does_not_read_external_file(
        self,
        manager: SessionManager,
        tmp_path: Path,
    ) -> None:
        """Read primitive must be neutralised — no filesystem access occurs.
        In particular, no JSONDecodeError should surface (that would confirm
        we at least opened the file), and no SessionNotFoundError either
        (that would confirm we checked existence). The validator short-
        circuits before any I/O.
        """
        secret = tmp_path / "secret.json"
        secret.write_text("not-json-content-that-would-leak-via-decode-error")
        with pytest.raises(InvalidSessionIdError):
            manager.load(str(secret.with_suffix("")))

    def test_new_session_id_always_passes_validator(self, manager: SessionManager) -> None:
        """Invariant: the shape enforced by the validator matches what
        new_session emits. If the generator format ever changes, this test
        fails and forces the validator to be updated alongside.
        """
        allowlist = re.compile(r"^[a-f0-9]{12}$")
        for _ in range(64):
            session_id, _state = manager.new_session()
            assert allowlist.match(session_id), f"new_session emitted {session_id!r} — update _SESSION_ID_RE"

    def test_valid_session_id_still_works_after_validation(self, manager: SessionManager) -> None:
        """Regression: legitimate round-trips are unaffected by the guard."""
        sid, state = manager.new_session(name="roundtrip")
        manager.save(sid, state)
        loaded = manager.load(sid)
        assert loaded.metadata.name == "roundtrip"
        manager.delete(sid)
