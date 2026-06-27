"""Tests for tutorial run service hardening."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import CallStatus, CallType, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.schema import calls_table
from elspeth.web.composer import tutorial_service as tutorial_service_module
from elspeth.web.composer.tutorial_service import (
    TutorialRunIntegrityError,
    _coalesce_run_source_hashes,
    _count_calls_for_run,
    _count_discarded_rows,
    _parse_rows_file,
    _rows_from_artifacts,
)
from elspeth.web.config import WebSettings
from tests.fixtures.landscape import make_factory, make_landscape_db


def _make_tutorial_settings(data_dir: Path, **overrides: Any) -> WebSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    values.update(overrides)
    return WebSettings(**values)


def test_count_calls_for_run_counts_only_llm_calls() -> None:
    db = make_landscape_db()
    factory = make_factory(db)
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    run_id = "run-llm-count"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    source_node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="inline_blob",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=schema_config,
    )
    transform_node = factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="llm_rate",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema_config,
    )
    _row, token = factory.data_flow.create_row_with_token(
        run_id=run_id,
        source_node_id=source_node.node_id,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"url": "https://example.gov"},
    )
    state = factory.execution.record_completed_node_state(
        token_id=token.token_id,
        node_id=transform_node.node_id,
        run_id=run_id,
        step_index=1,
        input_data={"url": "https://example.gov"},
        output_data={"rating": 5},
        duration_ms=1.0,
    )
    operation = factory.execution.begin_operation(
        run_id=run_id,
        node_id=source_node.node_id,
        operation_type="source_load",
    )

    def insert_call(call_id: str, *, call_type: CallType, state_id: str | None = None, operation_id: str | None = None) -> None:
        with db.write_connection() as conn:
            conn.execute(
                calls_table.insert().values(
                    call_id=call_id,
                    state_id=state_id,
                    operation_id=operation_id,
                    call_index=0 if call_type is CallType.LLM else 1,
                    call_type=call_type.value,
                    status=CallStatus.SUCCESS.value,
                    request_hash=f"{call_id}-request",
                    response_hash=f"{call_id}-response",
                    created_at=datetime.now(UTC),
                )
            )

    insert_call("state-llm", call_type=CallType.LLM, state_id=state.state_id)
    insert_call("state-http", call_type=CallType.HTTP, state_id=state.state_id)
    insert_call("operation-llm", call_type=CallType.LLM, operation_id=operation.operation_id)
    insert_call("operation-sql", call_type=CallType.SQL, operation_id=operation.operation_id)

    with db.connection() as conn:
        assert _count_calls_for_run(conn, run_id) == 2


def test_count_discarded_rows_counts_only_discard_destination() -> None:
    """_count_discarded_rows counts validation_errors whose destination is the
    'discard' sentinel, NOT quarantine-to-a-sink rows (which have a visible
    destination). This is what the tutorial UX surfaces so source-dropped rows
    are not silently invisible."""
    from elspeth.contracts.schema import SchemaConfig

    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "run-discard-count"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )
    for i in range(2):
        factory.data_flow.record_validation_error(
            run_id=run_id,
            node_id="source-0",
            row_data={"i": i},
            error="comma split the row",
            schema_mode="parse",
            destination="discard",
        )
    # A quarantined-to-a-sink row must NOT be counted as discarded.
    factory.data_flow.record_validation_error(
        run_id=run_id,
        node_id="source-0",
        row_data={"i": 99},
        error="bad value",
        schema_mode="parse",
        destination="quarantine_sink",
    )

    with db.connection() as conn:
        assert _count_discarded_rows(conn, run_id) == 2

    other = "run-clean"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=other)
    with db.connection() as conn:
        assert _count_discarded_rows(conn, other) == 0


def test_coalesce_run_source_hashes_aggregates_row_hashes() -> None:
    hashes = ("a" * 64, "b" * 64)

    assert _coalesce_run_source_hashes(hashes, run_id="run-1") == stable_hash({"source_data_hashes": list(hashes)})


# --- _parse_rows_file / _rows_from_artifacts --------------------------------
# Tier-1 audit invariant: row-parsing distinguishes three failure modes, none
# of which may be silently coalesced into ``[]``:
#   1. file format is not a recognised row format → ``None`` (caller skips)
#   2. file IS a row format but content is structurally corrupt → raise
#   3. file IS a row format and parses cleanly → return rows (possibly empty)


def test_parse_rows_file_returns_none_for_unrecognised_suffix(tmp_path: Path) -> None:
    """An auxiliary artifact (.txt, .parquet, .log) is not a row format.

    Returning ``None`` lets ``_rows_from_artifacts`` cleanly skip it and try
    the next artifact, instead of silently treating "I can't read this" as
    "this had zero rows" — which would shadow Tier-1 corruption behind the
    misleading "no readable file artifact with rows" message.
    """
    for name, content in [("notes.txt", "freeform text"), ("dump.parquet", b"\x00\x01")]:
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        assert _parse_rows_file(path) is None, f"{name}: non-row format must return None, not []"


def test_parse_rows_file_raises_for_json_scalar_or_null(tmp_path: Path) -> None:
    """A .json artifact whose top-level is a scalar/null is structurally corrupt.

    The prior implementation fell through and returned [] — Tier-1 corruption
    presented to the operator as "the run finished with no rows".
    """
    for label, content in [
        ("null", "null"),
        ("integer", "42"),
        ("string", '"a string"'),
        ("boolean", "true"),
    ]:
        path = tmp_path / f"output_{label}.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(TutorialRunIntegrityError, match="JSON top-level"):
            _parse_rows_file(path)


def test_parse_rows_file_parses_csv_and_returns_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("url,rating\nato.gov.au,5\ndata.gov.au,4\n", encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == [{"url": "ato.gov.au", "rating": "5"}, {"url": "data.gov.au", "rating": "4"}]


def test_parse_rows_file_parses_json_list_of_objects(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('[{"url": "ato.gov.au", "rating": 5}, {"url": "data.gov.au", "rating": 4}]', encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == [{"url": "ato.gov.au", "rating": 5}, {"url": "data.gov.au", "rating": 4}]


def test_parse_rows_file_parses_empty_csv_as_empty_rows(tmp_path: Path) -> None:
    """A header-only CSV legitimately yields zero rows — that is not corruption."""
    path = tmp_path / "empty.csv"
    path.write_text("url,rating\n", encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == []  # legitimate empty result, distinct from "couldn't parse"


def _fake_artifact(
    artifact_id: str,
    path_or_uri: str,
    artifact_type: str = "file",
    *,
    content_hash: str | None = None,
    size_bytes: int | None = None,
) -> Any:
    from types import SimpleNamespace

    fs_path = Path(path_or_uri.removeprefix("file://"))
    if content_hash is None and fs_path.exists():
        content_hash = hashlib.sha256(fs_path.read_bytes()).hexdigest()
    if size_bytes is None and fs_path.exists():
        size_bytes = fs_path.stat().st_size
    return SimpleNamespace(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        path_or_uri=path_or_uri,
        content_hash=content_hash or "0" * 64,
        size_bytes=0 if size_bytes is None else size_bytes,
    )


def test_rows_from_artifacts_skips_auxiliary_and_returns_row_artifact_rows(tmp_path: Path) -> None:
    """A run that emits a debug .txt log AND a rows .csv must return the .csv rows.

    Auxiliary artifacts (.txt, .log) must be skipped without crashing the
    projection — only row-format artifacts contribute to the row sequence.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    aux = outputs / "debug.txt"
    aux.write_text("composer chain trace", encoding="utf-8")
    rows_file = outputs / "rows.csv"
    rows_file.write_text("url,rating\nato.gov.au,5\n", encoding="utf-8")

    artifacts = [
        _fake_artifact("aux-1", str(aux)),
        _fake_artifact("rows-1", str(rows_file)),
    ]

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-aux")
    assert rows == [{"url": "ato.gov.au", "rating": "5"}]


def test_rows_from_artifacts_skips_auxiliary_before_reading_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    aux = outputs / "debug.bin"
    aux.write_bytes(b"x" * 1024)
    rows_file = outputs / "rows.csv"
    rows_file.write_text("url,rating\nato.gov.au,5\n", encoding="utf-8")
    artifacts = [
        _fake_artifact("aux-1", str(aux)),
        _fake_artifact("rows-1", str(rows_file)),
    ]
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == aux:
            raise AssertionError("auxiliary artifact should be skipped before byte reads")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-aux-skip")

    assert rows == [{"url": "ato.gov.au", "rating": "5"}]


def test_rows_from_artifacts_skips_legacy_percent_encoded_suffix_auxiliary_before_reading_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    legacy_aux = outputs / "debug%2Ecsv"
    decoded_decoy = outputs / "debug.csv"
    legacy_aux.write_bytes(b"x" * 1024)
    decoded_decoy.write_text("url,rating\ndecoy.example,1\n", encoding="utf-8")
    rows_file = outputs / "rows.csv"
    rows_file.write_text("url,rating\nato.gov.au,5\n", encoding="utf-8")
    artifacts = [
        _fake_artifact("aux-legacy", f"file://{outputs}/debug%2Ecsv"),
        _fake_artifact("rows-1", str(rows_file)),
    ]
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path in {legacy_aux, decoded_decoy}:
            raise AssertionError("legacy percent-suffix auxiliary artifact should be skipped before byte reads")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-percent-aux-skip")

    assert rows == [{"url": "ato.gov.au", "rating": "5"}]


def test_rows_from_artifacts_uses_legacy_raw_percent_candidate_when_it_matches_audit(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    legacy_raw_file = outputs / "results%3Ftoken=literal.csv"
    decoded_decoy = outputs / "results?token=literal.csv"
    audited_bytes = b"url,rating\nraw.example,5\n"
    legacy_raw_file.write_bytes(audited_bytes)
    decoded_decoy.write_bytes(b"url,rating\ndecoded.example,1\n")
    artifacts = [
        _fake_artifact(
            "rows-legacy",
            f"file://{outputs}/results%3Ftoken=literal.csv",
            content_hash=hashlib.sha256(audited_bytes).hexdigest(),
            size_bytes=len(audited_bytes),
        )
    ]

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-legacy")

    assert rows == [{"url": "raw.example", "rating": "5"}]


def test_rows_from_artifacts_parses_verified_bytes_when_file_changes_after_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rows_file = outputs / "rows.csv"
    audited_bytes = b"url,rating\nato.gov.au,5\n"
    rows_file.write_bytes(audited_bytes)
    artifacts = [
        _fake_artifact(
            "rows-1",
            str(rows_file),
            content_hash=hashlib.sha256(audited_bytes).hexdigest(),
            size_bytes=len(audited_bytes),
        )
    ]
    real_parse_rows_content = tutorial_service_module._parse_rows_content

    def tampering_parse_rows_content(path: Path, content: bytes) -> list[dict[str, Any]] | None:
        path.write_text("url,rating\ntampered.example,1\n", encoding="utf-8")
        return real_parse_rows_content(path, content)

    monkeypatch.setattr(tutorial_service_module, "_parse_rows_content", tampering_parse_rows_content)

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-snapshot")

    assert rows == [{"url": "ato.gov.au", "rating": "5"}]
    assert rows_file.read_text(encoding="utf-8") == "url,rating\ntampered.example,1\n"


def test_rows_from_artifacts_distinguishes_no_row_format_from_all_empty(tmp_path: Path) -> None:
    """The 'no row-bearing artifact found' message must name the recognised formats.

    Operator action differs between 'pipeline emitted .parquet but we can't
    read parquet' and 'pipeline emitted rows.csv but it was empty' — the
    prior implementation collapsed both into a single misleading message.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    only_aux = outputs / "summary.txt"
    only_aux.write_text("freeform", encoding="utf-8")
    artifacts = [_fake_artifact("aux-1", str(only_aux))]

    with pytest.raises(TutorialRunIntegrityError, match="no row-bearing artifact"):
        _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-only-aux")


def test_rows_from_artifacts_raises_when_all_row_artifacts_yield_zero_rows(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    empty = outputs / "empty.csv"
    empty.write_text("url,rating\n", encoding="utf-8")
    artifacts = [_fake_artifact("rows-1", str(empty))]

    with pytest.raises(TutorialRunIntegrityError, match="yielded zero rows"):
        _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-empty")
