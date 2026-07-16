"""Real-filesystem proofs for recoverable local sink effects."""

from __future__ import annotations

import csv
import fcntl
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectDescriptorMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.engine.orchestrator.preflight import validate_sink_effect_capability
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode
from elspeth.plugins.sinks import _local_file_effects as local_effects
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sinks.text_sink import TextSink

_SCHEMA = {"mode": "observed"}
_CTX = RestrictedSinkEffectContext(
    run_id="run-1",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="operation-1",
    sink_node_id="sink-1",
)


def _member(ordinal: int, row: dict[str, object]) -> SinkEffectMember:
    row_bytes = canonical_json(row).encode("utf-8")
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(row_bytes).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{ordinal}-{row_bytes!r}".encode()).hexdigest(),
    )


def _prepare(
    sink: CSVSink | JSONSink | TextSink,
    *,
    effect_id: str,
    rows: list[dict[str, object]],
    predecessor=None,
):
    members = tuple(_member(index, row) for index, row in enumerate(rows))
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(
            effect_id=effect_id,
            target="{}",
            predecessor_descriptor=predecessor,
        ),
        _CTX,
    )
    return sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id=effect_id,
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=members),
            inspection=inspection,
        ),
        _CTX,
    )


def _stage_path(plan) -> Path:
    return Path(str(plan.safe_evidence["staging_path"]))


def test_abandoned_atomic_replace_reconciles_by_staged_file_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "out.csv"
    sink = CSVSink({"path": str(target), "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="a" * 64, rows=[{"id": 1}, {"id": 2}])

    def crash_after_replace(_target: Path) -> None:
        raise RuntimeError("lost response")

    monkeypatch.setattr(local_effects, "_after_replace", crash_after_replace)
    with pytest.raises(RuntimeError, match="lost response"):
        sink.commit_effect(plan, _CTX)

    fresh = CSVSink({"path": str(target), "schema": _SCHEMA})
    result = fresh.reconcile_effect(plan, _CTX)
    assert result.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert result.descriptor == plan.expected_descriptor


def test_equal_bytes_from_unrelated_inode_are_unknown(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="b" * 64, rows=[{"message": "hello"}])
    target.write_bytes(_stage_path(plan).read_bytes())

    result = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA}).reconcile_effect(plan, _CTX)

    assert result.kind is SinkEffectReconcileKind.UNKNOWN


def test_disjoint_effects_publish_predecessor_then_successor_without_loss(tmp_path: Path) -> None:
    target = tmp_path / "out.csv"
    config = {"path": str(target), "schema": _SCHEMA}
    first_sink = CSVSink(config)
    first = _prepare(first_sink, effect_id="c" * 64, rows=[{"id": 1}])
    first_result = first_sink.commit_effect(first, _CTX)

    second_sink = CSVSink(config)
    second = _prepare(
        second_sink,
        effect_id="d" * 64,
        rows=[{"id": 2}],
        predecessor=first_result.descriptor,
    )
    second_sink.commit_effect(second, _CTX)

    assert target.read_text() == "id\n1\n2\n"


def test_append_effect_includes_validated_initial_baseline(tmp_path: Path) -> None:
    target = tmp_path / "append.csv"
    target.write_text("id\n0\n")
    sink = CSVSink(
        {
            "path": str(target),
            "schema": _SCHEMA,
            "mode": "append",
            "collision_policy": "append_or_create",
        }
    )
    plan = _prepare(sink, effect_id="da" * 32, rows=[{"id": 1}])
    sink.commit_effect(plan, _CTX)
    assert target.read_text() == "id\n0\n1\n"


def test_json_and_text_effect_adapters_publish_cumulative_snapshots(tmp_path: Path) -> None:
    json_target = tmp_path / "out.json"
    json_config = {"path": str(json_target), "format": "json", "schema": _SCHEMA}
    first_json = JSONSink(json_config)
    first_plan = _prepare(first_json, effect_id="e" * 64, rows=[{"id": 1}])
    first_result = first_json.commit_effect(first_plan, _CTX)
    second_json = JSONSink(json_config)
    second_plan = _prepare(
        second_json,
        effect_id="f" * 64,
        rows=[{"id": 2}],
        predecessor=first_result.descriptor,
    )
    second_json.commit_effect(second_plan, _CTX)
    assert json_target.read_text() == '[{"id": 1}, {"id": 2}]'

    text_target = tmp_path / "out.txt"
    text_config = {"path": str(text_target), "field": "message", "schema": _SCHEMA}
    first_text = TextSink(text_config)
    first_text_plan = _prepare(first_text, effect_id="1" * 64, rows=[{"message": "one"}])
    first_text_result = first_text.commit_effect(first_text_plan, _CTX)
    second_text = TextSink(text_config)
    second_text_plan = _prepare(
        second_text,
        effect_id="2" * 64,
        rows=[{"message": "two"}],
        predecessor=first_text_result.descriptor,
    )
    second_text.commit_effect(second_text_plan, _CTX)
    assert text_target.read_bytes() == b"one\ntwo\n"


def test_staging_enforces_streamed_byte_and_row_limits(tmp_path: Path) -> None:
    inspection = local_effects.inspect_local_effect(
        target_path=tmp_path / "bounded.txt",
        request=SinkEffectInspectionRequest(effect_id="3" * 64, target="{}", predecessor_descriptor=None),
    )
    with pytest.raises(local_effects.LocalFileEffectLimitError, match="row limit"):
        local_effects.prepare_local_effect(
            effect_id="3" * 64,
            input_kind=local_effects.SinkEffectInputKind.PIPELINE_MEMBERS,
            inspection=inspection,
            chunks=(b"one",),
            row_count=2,
            accepted_ordinals=(0, 1),
            diverted_ordinals=(),
            encoding="utf-8",
            format_name="text",
            stream_sequence=0,
            max_rows=1,
        )
    with pytest.raises(local_effects.LocalFileEffectLimitError, match="byte limit"):
        byte_inspection = local_effects.inspect_local_effect(
            target_path=tmp_path / "bounded.txt",
            request=SinkEffectInspectionRequest(effect_id="4" * 64, target="{}", predecessor_descriptor=None),
        )
        local_effects.prepare_local_effect(
            effect_id="4" * 64,
            input_kind=local_effects.SinkEffectInputKind.PIPELINE_MEMBERS,
            inspection=byte_inspection,
            chunks=(b"too-large",),
            row_count=1,
            accepted_ordinals=(0,),
            diverted_ordinals=(),
            encoding="utf-8",
            format_name="text",
            stream_sequence=0,
            max_bytes=4,
        )


def test_commit_times_out_on_busy_advisory_lock(tmp_path: Path) -> None:
    target = tmp_path / "locked.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="5" * 64, rows=[{"message": "hello"}])
    lock_path = Path(str(plan.safe_evidence["lock_path"]))
    lock_path.touch()
    with lock_path.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(local_effects.LocalFileLockTimeout):
            local_effects.commit_local_effect(plan, lock_timeout_seconds=0.01)


def test_staging_mismatch_is_unknown_and_cleanup_is_identity_scoped(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="6" * 64, rows=[{"message": "hello"}])
    stage = _stage_path(plan)
    stage.write_bytes(b"tampered")

    assert sink.reconcile_effect(plan, _CTX).kind is SinkEffectReconcileKind.UNKNOWN
    assert local_effects.cleanup_local_effect(plan) is False
    assert stage.exists()


def test_commit_fsyncs_parent_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "out.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="7" * 64, rows=[{"message": "hello"}])
    observed: list[Path] = []
    real_fsync = local_effects._fsync_directory

    def recording_fsync(path: Path) -> None:
        observed.append(path)
        real_fsync(path)

    monkeypatch.setattr(local_effects, "_fsync_directory", recording_fsync)
    sink.commit_effect(plan, _CTX)
    assert target.parent in observed


def test_local_sink_capabilities_are_declared_by_input_kind(tmp_path: Path) -> None:
    csv_sink = CSVSink({"path": str(tmp_path / "out.csv"), "schema": _SCHEMA})
    json_sink = JSONSink({"path": str(tmp_path / "out.json"), "schema": _SCHEMA})
    text_sink = TextSink({"path": str(tmp_path / "out.txt"), "field": "message", "schema": _SCHEMA})

    validate_sink_effect_capability(csv_sink, "write", local_effects.SinkEffectInputKind.PIPELINE_MEMBERS)
    validate_sink_effect_capability(json_sink, "write", local_effects.SinkEffectInputKind.PIPELINE_MEMBERS)
    validate_sink_effect_capability(text_sink, "write", local_effects.SinkEffectInputKind.PIPELINE_MEMBERS)


def test_effect_inspection_applies_deferred_collision_policy(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("occupied\n")
    with plugin_preflight_mode(True):
        fail_sink = TextSink(
            {
                "path": str(target),
                "field": "message",
                "schema": _SCHEMA,
                "collision_policy": "fail_if_exists",
            }
        )
        increment_sink = TextSink(
            {
                "path": str(target),
                "field": "message",
                "schema": _SCHEMA,
                "collision_policy": "auto_increment",
            }
        )
    request = SinkEffectInspectionRequest(effect_id="8" * 64, target="{}", predecessor_descriptor=None)
    with pytest.raises(FileExistsError):
        fail_sink.inspect_effect(request, _CTX)
    inspection = increment_sink.inspect_effect(request, _CTX)
    assert inspection.evidence["target_path"] == str(tmp_path / "out-1.txt")


def test_unsupported_filesystem_identity_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "out.txt"
    target.write_text("existing\n")

    def unsupported(_stat) -> str:
        raise local_effects.LocalFileUnsupportedIdentity("unsupported")

    monkeypatch.setattr(local_effects, "_stable_file_id", unsupported)
    with pytest.raises(local_effects.LocalFileUnsupportedIdentity):
        local_effects.inspect_local_effect(
            target_path=target,
            request=SinkEffectInspectionRequest(effect_id="9" * 64, target="{}", predecessor_descriptor=None),
        )


def test_cleanup_removes_only_exact_abandoned_stage(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="0" * 64, rows=[{"message": "hello"}])
    stage = _stage_path(plan)
    assert local_effects.cleanup_local_effect(plan) is True
    assert not stage.exists()


def test_no_publication_uses_exact_virtual_and_inherited_descriptors(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    virtual_inspection = local_effects.inspect_local_effect(
        target_path=target,
        request=SinkEffectInspectionRequest(effect_id="a1" * 32, target="{}", predecessor_descriptor=None),
    )
    virtual = local_effects.prepare_local_effect(
        effect_id="a1" * 32,
        input_kind=local_effects.SinkEffectInputKind.PIPELINE_MEMBERS,
        inspection=virtual_inspection,
        chunks=(),
        row_count=0,
        accepted_ordinals=(),
        diverted_ordinals=(),
        encoding="utf-8",
        format_name="text",
        stream_sequence=0,
    )
    assert virtual.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION
    assert virtual.safe_evidence["publication_kind"] == "virtual"
    assert virtual.expected_descriptor is not None
    assert virtual.expected_descriptor.size_bytes == 0
    assert not target.exists()
    assert not _stage_path(virtual).exists()

    target.write_bytes(b"existing\n")
    predecessor = ArtifactDescriptor.for_file(
        path=str(target.resolve()),
        content_hash=sha256(b"existing\n").hexdigest(),
        size_bytes=len(b"existing\n"),
    )
    inherited_inspection = local_effects.inspect_local_effect(
        target_path=target,
        request=SinkEffectInspectionRequest(
            effect_id="a2" * 32,
            target="{}",
            predecessor_descriptor=predecessor,
        ),
    )
    inherited = local_effects.prepare_local_effect(
        effect_id="a2" * 32,
        input_kind=local_effects.SinkEffectInputKind.PIPELINE_MEMBERS,
        inspection=inherited_inspection,
        chunks=local_effects.iter_path_chunks(target),
        row_count=0,
        accepted_ordinals=(),
        diverted_ordinals=(),
        encoding="utf-8",
        format_name="text",
        stream_sequence=1,
    )
    assert inherited.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION
    assert inherited.safe_evidence["publication_kind"] == "inherited"
    assert inherited.expected_descriptor == predecessor
    assert target.read_bytes() == b"existing\n"
    assert not _stage_path(inherited).exists()


def test_effect_streaming_preserves_stateful_text_encodings(tmp_path: Path) -> None:
    csv_target = tmp_path / "utf16.csv"
    csv_sink = CSVSink({"path": str(csv_target), "schema": _SCHEMA, "encoding": "utf-16"})
    csv_plan = _prepare(csv_sink, effect_id="b1" * 32, rows=[{"id": 1}, {"id": 2}])
    csv_result = csv_sink.commit_effect(csv_plan, _CTX)
    next_csv_sink = CSVSink({"path": str(csv_target), "schema": _SCHEMA, "encoding": "utf-16"})
    next_csv_plan = _prepare(
        next_csv_sink,
        effect_id="b3" * 32,
        rows=[{"id": 3}],
        predecessor=csv_result.descriptor,
    )
    next_csv_sink.commit_effect(next_csv_plan, _CTX)
    with csv_target.open(encoding="utf-16", newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"id": "1"}, {"id": "2"}, {"id": "3"}]

    json_target = tmp_path / "utf16.json"
    json_sink = JSONSink({"path": str(json_target), "schema": _SCHEMA, "encoding": "utf-16"})
    json_plan = _prepare(json_sink, effect_id="b2" * 32, rows=[{"id": 1}, {"id": 2}])
    json_result = json_sink.commit_effect(json_plan, _CTX)
    next_json_sink = JSONSink({"path": str(json_target), "schema": _SCHEMA, "encoding": "utf-16"})
    next_json_plan = _prepare(
        next_json_sink,
        effect_id="b4" * 32,
        rows=[{"id": 3}],
        predecessor=json_result.descriptor,
    )
    next_json_sink.commit_effect(next_json_plan, _CTX)
    with json_target.open(encoding="utf-16") as stream:
        assert json.load(stream) == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_indented_json_array_effects_preserve_cumulative_format(tmp_path: Path) -> None:
    target = tmp_path / "pretty.json"
    config = {"path": str(target), "format": "json", "indent": 2, "schema": _SCHEMA}
    first_sink = JSONSink(config)
    first = _prepare(first_sink, effect_id="c1" * 32, rows=[{"id": 1}])
    first_result = first_sink.commit_effect(first, _CTX)
    second_sink = JSONSink(config)
    second = _prepare(
        second_sink,
        effect_id="c2" * 32,
        rows=[{"id": 2}],
        predecessor=first_result.descriptor,
    )
    second_sink.commit_effect(second, _CTX)
    assert target.read_text() == json.dumps([{"id": 1}, {"id": 2}], indent=2)
