"""Tests for the strict, line-oriented local text sink."""

from __future__ import annotations

import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode
from elspeth.plugins.sinks.text_sink import TextSink, TextSinkConfig
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


def _config(path: Path, **overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "path": str(path),
        "field": "line_text",
        "encoding": "utf-8",
        "mode": "write",
        "schema": {"mode": "observed"},
    }
    config.update(overrides)
    return config


def _sink_context() -> PluginContext:
    return make_context(landscape=make_factory().plugin_audit_writer())


@pytest.mark.parametrize("encoding", ["utf-8", "ascii", "latin-1", "cp1252"])
def test_text_sink_accepts_only_supported_ascii_compatible_encodings(encoding: str) -> None:
    parsed = TextSinkConfig.from_dict(_config(Path("out.txt"), encoding=encoding), plugin_name="text")
    assert parsed.encoding == encoding


@pytest.mark.parametrize("encoding", ["utf-16", "utf-32", "utf-7", "shift_jis", "unknown-codec"])
def test_text_sink_rejects_encodings_outside_closed_set(encoding: str) -> None:
    with pytest.raises(PluginConfigError, match="encoding"):
        TextSinkConfig.from_dict(_config(Path("out.txt"), encoding=encoding), plugin_name="text")


@pytest.mark.parametrize("field", ["", "two words", "with-hyphen", "9lives", "class"])
def test_text_sink_rejects_non_identifier_or_keyword_field(field: str) -> None:
    with pytest.raises(PluginConfigError, match="identifier"):
        TextSinkConfig.from_dict(_config(Path("out.txt"), field=field), plugin_name="text")


def test_text_sink_config_schema_has_no_headers_property() -> None:
    assert "headers" not in TextSinkConfig.model_json_schema()["properties"]


@pytest.mark.parametrize(
    ("mode", "collision_policy"),
    [
        ("append", "fail_if_exists"),
        ("append", "auto_increment"),
        ("write", "append_or_create"),
    ],
)
def test_text_sink_rejects_incompatible_mode_and_collision_policy(mode: str, collision_policy: str) -> None:
    with pytest.raises(PluginConfigError, match="collision_policy"):
        TextSinkConfig.from_dict(
            _config(Path("out.txt"), mode=mode, collision_policy=collision_policy),
            plugin_name="text",
        )


def test_text_sink_writes_canonical_lf_and_whole_artifact_hash(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([{"line_text": "alpha"}, {"line_text": ""}, {"line_text": "omega"}], _sink_context())
    sink.close()

    expected = b"alpha\n\nomega\n"
    assert path.read_bytes() == expected
    assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()
    assert result.artifact.size_bytes == len(expected)


def test_text_sink_multiple_write_batches_preserve_prior_committed_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    sink.write([{"line_text": "first"}], _sink_context())
    result = sink.write([{"line_text": "second"}], _sink_context())

    expected = b"first\nsecond\n"
    assert path.read_bytes() == expected
    assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize("value", [None, 1, True, b"bytes", ["list"], {"nested": "object"}])
def test_text_sink_diverts_non_string_without_coercion(tmp_path: Path, value: object) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([{"line_text": value}], _sink_context())

    assert not path.exists()
    assert len(result.diversions) == 1


@pytest.mark.parametrize("value", ["line\nnext", "line\rnext", "line\r\nnext"])
def test_text_sink_diverts_embedded_record_separators(tmp_path: Path, value: str) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([{"line_text": "safe"}, {"line_text": value}], _sink_context())

    assert path.read_bytes() == b"safe\n"
    assert len(result.diversions) == 1
    assert value not in result.diversions[0].reason


def test_text_sink_missing_field_diverts(tmp_path: Path) -> None:
    sink = inject_write_failure(TextSink(_config(tmp_path / "out.txt")))

    result = sink.write([{"other": "value"}], _sink_context())

    assert len(result.diversions) == 1
    assert "line_text" in result.diversions[0].reason


def test_text_sink_encoding_diversion_reason_is_sanitized(tmp_path: Path) -> None:
    secret = "credential-\N{PILE OF POO}"
    sink = inject_write_failure(TextSink(_config(tmp_path / "out.txt", encoding="ascii")))

    result = sink.write([{"line_text": secret}], _sink_context())

    assert len(result.diversions) == 1
    reason = result.diversions[0].reason
    assert secret not in reason
    assert "can't encode" not in reason
    assert "ascii" in reason


def test_text_sink_diversion_fails_closed_without_policy_injection(tmp_path: Path) -> None:
    sink = TextSink(_config(tmp_path / "out.txt"))

    with pytest.raises(FrameworkBugError, match="on_write_failure"):
        sink.write([{"line_text": 1}], _sink_context())


def test_empty_batch_does_not_create_target_and_returns_virtual_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([], _sink_context())

    assert not path.exists()
    assert result.artifact.size_bytes == 0
    assert result.artifact.content_hash == hashlib.sha256(b"").hexdigest()


def test_empty_batch_reports_existing_target_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    original = b"existing\n"
    path.write_bytes(original)
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([], _sink_context())

    assert path.read_bytes() == original
    assert result.artifact.size_bytes == len(original)
    assert result.artifact.content_hash == hashlib.sha256(original).hexdigest()


def test_all_diverted_batch_is_virtual_and_does_not_mutate_target(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    original = b"existing\n"
    path.write_bytes(original)
    sink = inject_write_failure(TextSink(_config(path)))

    result = sink.write([{"line_text": "bad\nline"}], _sink_context())

    assert path.read_bytes() == original
    assert result.artifact.content_hash == hashlib.sha256(original).hexdigest()


def test_fail_if_exists_is_deferred_until_first_real_write(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"winner\n")
    sink = inject_write_failure(TextSink(_config(path, collision_policy="fail_if_exists")))

    with pytest.raises(FileExistsError):
        sink.write([{"line_text": "loser"}], _sink_context())

    assert path.read_bytes() == b"winner\n"


def test_preflight_construction_never_mutates_or_resolves_output(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"winner\n")

    with plugin_preflight_mode(True):
        sink = TextSink(_config(path, collision_policy="auto_increment"))

    assert sink._path == path
    assert list(tmp_path.iterdir()) == [path]


def test_auto_increment_claims_free_sibling_lazily(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"winner\n")
    sink = inject_write_failure(TextSink(_config(path, collision_policy="auto_increment")))

    result = sink.write([{"line_text": "new"}], _sink_context())

    chosen = tmp_path / "out-1.txt"
    assert path.read_bytes() == b"winner\n"
    assert chosen.read_bytes() == b"new\n"
    assert result.artifact.path_or_uri == f"file://{chosen}"


def test_fail_if_exists_race_has_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"

    def attempt(value: str) -> str:
        sink = inject_write_failure(TextSink(_config(path, collision_policy="fail_if_exists")))
        try:
            sink.write([{"line_text": value}], _sink_context())
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ["alpha", "beta"]))

    assert sorted(outcomes) == ["lost", "won"]
    assert path.read_bytes() in {b"alpha\n", b"beta\n"}


def test_auto_increment_race_never_replaces_another_reservation(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"original\n")

    def attempt(value: str) -> None:
        sink = inject_write_failure(TextSink(_config(path, collision_policy="auto_increment")))
        sink.write([{"line_text": value}], _sink_context())

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(attempt, ["alpha", "beta"]))

    assert path.read_bytes() == b"original\n"
    assert {p.read_bytes() for p in tmp_path.glob("out-*.txt")} == {b"alpha\n", b"beta\n"}


@pytest.mark.parametrize(
    ("initial", "message"),
    [
        (b"unterminated", "LF record boundary"),
        (b"one\r\ntwo\r\n", "CR separators"),
        (b"one\rtwo\n", "CR separators"),
        (b"\xff\n", "not valid utf-8"),
    ],
)
def test_append_rejects_noncanonical_existing_target(tmp_path: Path, initial: bytes, message: str) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(initial)
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    validation = sink.validate_output_target()
    assert validation.valid is False
    assert validation.error_message is not None and message in validation.error_message
    with pytest.raises(ValueError, match="Existing text output"):
        sink.write([{"line_text": "new"}], _sink_context())
    assert path.read_bytes() == initial


def test_append_includes_existing_and_new_bytes_in_hash(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    result = sink.write([{"line_text": "new"}], _sink_context())

    expected = b"existing\nnew\n"
    assert path.read_bytes() == expected
    assert result.artifact.content_hash == hashlib.sha256(expected).hexdigest()


def test_append_failure_rolls_back_to_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    def raise_after_write(_: Path) -> os.stat_result:
        raise OSError("stat failed")

    sink._artifact_stat = raise_after_write
    with pytest.raises(OSError, match="stat failed"):
        sink.write([{"line_text": "new"}], _sink_context())

    assert path.read_bytes() == b"existing\n"


def test_append_flush_failure_rolls_back_to_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    sink._sync_stream = lambda _handle: (_ for _ in ()).throw(OSError("flush failed"))
    with pytest.raises(OSError, match="flush failed"):
        sink.write([{"line_text": "new"}], _sink_context())

    assert path.read_bytes() == b"existing\n"


def test_append_hash_failure_rolls_back_to_original_bytes(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    sink._extend_hasher = lambda _staged: (_ for _ in ()).throw(OSError("hash failed"))
    with pytest.raises(OSError, match="hash failed"):
        sink.write([{"line_text": "new"}], _sink_context())

    assert path.read_bytes() == b"existing\n"


def test_failed_append_can_retry_without_replaying_failed_batch(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))
    original_stat = sink._artifact_stat

    def raise_after_write(_: Path) -> os.stat_result:
        raise OSError("stat failed")

    sink._artifact_stat = raise_after_write
    with pytest.raises(OSError):
        sink.write([{"line_text": "failed"}], _sink_context())
    sink._artifact_stat = original_stat

    sink.write([{"line_text": "retried"}], _sink_context())
    assert path.read_bytes() == b"existing\nretried\n"


def test_write_mode_precommit_failure_preserves_preexisting_target(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path)))

    def fail_stat(_: Path) -> os.stat_result:
        raise OSError("stat failed")

    sink._artifact_stat = fail_stat
    with pytest.raises(OSError, match="stat failed"):
        sink.write([{"line_text": "new"}], _sink_context())
    assert path.read_bytes() == b"existing\n"


def test_second_write_precommit_failure_preserves_first_committed_batch(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path)))
    sink.write([{"line_text": "first"}], _sink_context())

    sink._artifact_stat = lambda _: (_ for _ in ()).throw(OSError("stat failed"))
    with pytest.raises(OSError, match="stat failed"):
        sink.write([{"line_text": "second"}], _sink_context())

    assert path.read_bytes() == b"first\n"


def test_exclusive_reservation_is_removed_after_precommit_failure(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path, collision_policy="fail_if_exists")))
    sink._artifact_stat = lambda _: (_ for _ in ()).throw(OSError("stat failed"))

    with pytest.raises(OSError, match="stat failed"):
        sink.write([{"line_text": "new"}], _sink_context())

    assert not path.exists()


def test_exclusive_reservation_is_removed_when_temp_creation_fails_and_retry_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path, collision_policy="fail_if_exists")))
    real_mkstemp = tempfile.mkstemp

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        del args, kwargs
        raise OSError("temp creation failed")

    monkeypatch.setattr("elspeth.plugins.sinks.text_sink.tempfile.mkstemp", fail_mkstemp)
    with pytest.raises(OSError, match="temp creation failed"):
        sink.write([{"line_text": "first"}], _sink_context())

    assert not path.exists()
    assert sink._reservation_owned is False
    assert sink._write_target_claimed is False

    monkeypatch.setattr("elspeth.plugins.sinks.text_sink.tempfile.mkstemp", real_mkstemp)
    sink.write([{"line_text": "retry"}], _sink_context())
    assert path.read_bytes() == b"retry\n"


def test_fdopen_failure_closes_raw_temp_fd_cleans_claim_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "out.txt"
    sink = inject_write_failure(TextSink(_config(path, collision_policy="fail_if_exists")))
    captured: dict[str, object] = {}
    real_mkstemp = tempfile.mkstemp
    real_fdopen = os.fdopen

    def capture_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        fd, name = real_mkstemp(*args, **kwargs)  # type: ignore[arg-type]
        captured.update(fd=fd, name=name)
        return fd, name

    def fail_fdopen(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("fdopen failed")

    monkeypatch.setattr("elspeth.plugins.sinks.text_sink.tempfile.mkstemp", capture_mkstemp)
    monkeypatch.setattr("elspeth.plugins.sinks.text_sink.os.fdopen", fail_fdopen)
    with pytest.raises(OSError, match="fdopen failed"):
        sink.write([{"line_text": "first"}], _sink_context())

    with pytest.raises(OSError):
        os.fstat(captured["fd"])  # type: ignore[arg-type]
    assert not Path(captured["name"]).exists()  # type: ignore[arg-type]
    assert not path.exists()
    assert sink._reservation_owned is False
    assert sink._write_target_claimed is False

    monkeypatch.setattr("elspeth.plugins.sinks.text_sink.os.fdopen", real_fdopen)
    sink.write([{"line_text": "retry"}], _sink_context())
    assert path.read_bytes() == b"retry\n"


def test_temp_creation_failure_never_removes_unowned_preexisting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"preexisting\n")
    sink = inject_write_failure(TextSink(_config(path)))

    monkeypatch.setattr(
        "elspeth.plugins.sinks.text_sink.tempfile.mkstemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("temp creation failed")),
    )
    with pytest.raises(OSError, match="temp creation failed"):
        sink.write([{"line_text": "new"}], _sink_context())

    assert path.read_bytes() == b"preexisting\n"


def test_write_mode_post_replace_failure_keeps_committed_new_target(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path)))

    def fail_parent_fsync() -> None:
        raise OSError("directory fsync failed")

    sink._fsync_parent = fail_parent_fsync
    with pytest.raises(OSError, match="directory fsync failed"):
        sink.write([{"line_text": "new"}], _sink_context())
    assert path.read_bytes() == b"new\n"


def test_append_rollback_failure_names_only_offset(tmp_path: Path) -> None:
    path = tmp_path / "out.txt"
    path.write_bytes(b"existing\n")
    sink = inject_write_failure(TextSink(_config(path, mode="append", collision_policy="append_or_create")))

    sink._artifact_stat = lambda _: (_ for _ in ()).throw(OSError("SENSITIVE primary"))
    sink._truncate_append = lambda _handle, _offset: (_ for _ in ()).throw(OSError("SENSITIVE rollback"))
    with pytest.raises(RuntimeError, match=r"byte offset 9") as exc_info:
        sink.write([{"line_text": "new"}], _sink_context())
    assert "SENSITIVE" not in str(exc_info.value)


def test_configure_for_resume_switches_to_append_contract(tmp_path: Path) -> None:
    sink = inject_write_failure(TextSink(_config(tmp_path / "out.txt")))

    sink.configure_for_resume()

    assert sink._mode == "append"
    assert sink._collision_policy == "append_or_create"


def test_close_is_idempotent_after_append(tmp_path: Path) -> None:
    sink = inject_write_failure(TextSink(_config(tmp_path / "out.txt", mode="append", collision_policy="append_or_create")))
    sink.write([{"line_text": "line"}], _sink_context())

    sink.close()
    sink.close()


def test_text_sink_assistance_pins_strict_line_and_resume_contract() -> None:
    assistance = TextSink.get_agent_assistance(issue_code=None)

    assert assistance is not None
    hints = " ".join(assistance.composer_hints)
    assert "configured field" in hints
    assert "string" in hints
    assert "CR or LF" in hints
    assert "append" in hints
    assert "resume" in hints
