"""Real-filesystem proofs for recoverable local sink effects."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import time
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.engine._error_hash import compute_error_hash
from elspeth.engine.orchestrator.preflight import validate_sink_effect_capability
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode
from elspeth.plugins.sinks import _local_file_effects as local_effects
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sinks.text_sink import TextSink
from tests.fixtures.base_classes import inject_write_failure

_SCHEMA = {"mode": "observed"}
_CTX = RestrictedSinkEffectContext(
    run_id="run-1",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="operation-1",
    sink_node_id="sink-1",
)


def _member(ordinal: int, row: dict[str, object], *, generation: str = "token") -> SinkEffectMember:
    row_bytes = canonical_json(row).encode("utf-8")
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"{generation}-{ordinal}",
        row_id=f"{generation}-row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(row_bytes).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{generation}-{ordinal}-{row_bytes!r}".encode()).hexdigest(),
    )


def _prepare(
    sink: CSVSink | JSONSink | TextSink,
    *,
    effect_id: str,
    rows: list[dict[str, object]],
    predecessor=None,
    predecessor_rows: list[dict[str, object]] | None = None,
):
    members = tuple(_member(index, row) for index, row in enumerate(rows))
    if predecessor_rows:
        # Mirror the coordinator's cumulative snapshot: finalized predecessor
        # members first, then the current partition, with contiguous ordinals.
        predecessor_members = tuple(_member(index, row, generation="predecessor") for index, row in enumerate(predecessor_rows))
        snapshot = tuple(replace(member, ordinal=ordinal) for ordinal, member in enumerate((*predecessor_members, *members)))
    else:
        snapshot = members
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
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=snapshot),
            inspection=inspection,
        ),
        _CTX,
    )


def _stage_path(plan) -> Path:
    return Path(str(plan.safe_evidence["staging_path"]))


def _expected_diversion_attribution(sink: CSVSink | JSONSink | TextSink) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "ordinal": diversion.row_index,
            "reason_hash": stable_hash({"diversion_reason": diversion.reason}),
            "error_hash": compute_error_hash(diversion.reason),
        }
        for diversion in sink._get_diversions()
    )


@pytest.mark.parametrize(
    ("sink", "rows"),
    [
        (
            lambda path: CSVSink({"path": str(path), "schema": _SCHEMA}),
            [{"id": 1}, {"id": 2, "extra": "divert"}],
        ),
        (
            lambda path: TextSink({"path": str(path), "field": "message", "schema": _SCHEMA}),
            [{"message": "accepted"}, {"message": 42}],
        ),
    ],
)
def test_local_effect_plans_persist_exact_diversion_attribution(tmp_path: Path, sink, rows) -> None:
    configured = inject_write_failure(sink(tmp_path / "out"))
    plan = _prepare(configured, effect_id="ae" * 32, rows=rows)

    assert plan.safe_evidence["accepted_ordinals"] == (0,)
    assert plan.safe_evidence["diverted_ordinals"] == (1,)
    assert plan.safe_evidence["diversion_attribution"] == _expected_diversion_attribution(configured)

    configured.commit_effect(plan, _CTX)
    recovered = configured.reconcile_effect(plan, _CTX)
    assert recovered.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert recovered.accepted_ordinals is None
    assert recovered.diverted_ordinals is None


def test_json_effect_plan_persists_exact_diversion_attribution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = inject_write_failure(JSONSink({"path": str(tmp_path / "out.jsonl"), "format": "jsonl", "schema": _SCHEMA}))
    real_dumps = json.dumps

    def reject_selected_row(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("divert") is True:
            raise ValueError("injected canonical-row serialization failure")
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(json, "dumps", reject_selected_row)
    plan = _prepare(
        sink,
        effect_id="af" * 32,
        rows=[{"id": 1}, {"id": 2, "divert": True}],
    )

    assert plan.safe_evidence["accepted_ordinals"] == (0,)
    assert plan.safe_evidence["diverted_ordinals"] == (1,)
    assert plan.safe_evidence["diversion_attribution"] == _expected_diversion_attribution(sink)


def test_csv_effect_thaws_nested_values_before_serialization(tmp_path: Path) -> None:
    sink = inject_write_failure(CSVSink({"path": str(tmp_path / "nested.csv"), "schema": _SCHEMA}))
    plan = _prepare(sink, effect_id="b1" * 32, rows=[{"id": 1, "metadata": {"tags": ["a", "b"]}}])

    with _stage_path(plan).open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"id": "1", "metadata": "{'tags': ['a', 'b']}"}]


def test_local_effect_evidence_rejects_missing_or_unmatched_diversion_attribution(tmp_path: Path) -> None:
    sink = inject_write_failure(TextSink({"path": str(tmp_path / "out.txt"), "field": "message", "schema": _SCHEMA}))
    plan = _prepare(sink, effect_id="ad" * 32, rows=[{"message": "accepted"}, {"message": 42}])
    evidence = dict(plan.safe_evidence)
    evidence.pop("diversion_attribution")
    with pytest.raises(local_effects.LocalFilePreconditionError, match="diversion attribution"):
        local_effects.LocalFileEffectPlanEvidence.from_mapping(evidence)

    evidence = dict(plan.safe_evidence)
    evidence["diversion_attribution"] = ()
    with pytest.raises(local_effects.LocalFilePreconditionError, match="diversion attribution"):
        local_effects.LocalFileEffectPlanEvidence.from_mapping(evidence)


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
        predecessor_rows=[{"id": 1}],
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


def test_append_successor_effects_preserve_pre_run_baseline(tmp_path: Path) -> None:
    """A cumulative successor snapshot must not erase pre-run append content."""
    target = tmp_path / "append.csv"
    target.write_text("id\n0\n")
    config = {
        "path": str(target),
        "schema": _SCHEMA,
        "mode": "append",
        "collision_policy": "append_or_create",
    }
    first_sink = CSVSink(config)
    first = _prepare(first_sink, effect_id="aa" * 32, rows=[{"id": 1}])
    first_result = first_sink.commit_effect(first, _CTX)
    assert target.read_text() == "id\n0\n1\n"

    second_sink = CSVSink(config)
    second = _prepare(
        second_sink,
        effect_id="ab" * 32,
        rows=[{"id": 2}],
        predecessor=first_result.descriptor,
        predecessor_rows=[{"id": 1}],
    )
    second_sink.commit_effect(second, _CTX)
    assert target.read_text() == "id\n0\n1\n2\n"


def test_append_successor_effects_preserve_text_and_json_baselines(tmp_path: Path) -> None:
    text_target = tmp_path / "append.txt"
    text_target.write_text("zero\n")
    text_config = {
        "path": str(text_target),
        "field": "message",
        "schema": _SCHEMA,
        "mode": "append",
        "collision_policy": "append_or_create",
    }
    first_text = TextSink(text_config)
    first_text_result = first_text.commit_effect(
        _prepare(first_text, effect_id="ba" * 32, rows=[{"message": "one"}]),
        _CTX,
    )
    second_text = TextSink(text_config)
    second_text.commit_effect(
        _prepare(
            second_text,
            effect_id="bb" * 32,
            rows=[{"message": "two"}],
            predecessor=first_text_result.descriptor,
            predecessor_rows=[{"message": "one"}],
        ),
        _CTX,
    )
    assert text_target.read_text() == "zero\none\ntwo\n"

    jsonl_target = tmp_path / "append.jsonl"
    jsonl_target.write_text('{"id": 0}\n')
    jsonl_config = {
        "path": str(jsonl_target),
        "format": "jsonl",
        "schema": _SCHEMA,
        "mode": "append",
        "collision_policy": "append_or_create",
    }
    first_jsonl = JSONSink(jsonl_config)
    first_jsonl_result = first_jsonl.commit_effect(
        _prepare(first_jsonl, effect_id="cc" * 32, rows=[{"id": 1}]),
        _CTX,
    )
    second_jsonl = JSONSink(jsonl_config)
    second_jsonl.commit_effect(
        _prepare(
            second_jsonl,
            effect_id="cd" * 32,
            rows=[{"id": 2}],
            predecessor=first_jsonl_result.descriptor,
            predecessor_rows=[{"id": 1}],
        ),
        _CTX,
    )
    assert jsonl_target.read_text() == '{"id": 0}\n{"id": 1}\n{"id": 2}\n'


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
        predecessor_rows=[{"id": 1}],
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
        predecessor_rows=[{"message": "one"}],
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
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
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
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
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


def test_staging_mismatch_is_unknown(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})
    plan = _prepare(sink, effect_id="6" * 64, rows=[{"message": "hello"}])
    stage = _stage_path(plan)
    stage.write_bytes(b"tampered")

    assert sink.reconcile_effect(plan, _CTX).kind is SinkEffectReconcileKind.UNKNOWN
    assert stage.exists()


def test_stale_sweep_removes_crashed_building_files_but_not_stage_or_lock(tmp_path: Path) -> None:
    effect_id = "6" * 64
    stale_building = tmp_path / f"..out.txt.elspeth-{effect_id}.stage.abc123.building"
    stale_building.write_bytes(b"crashed")
    fresh_building = tmp_path / f"..out.txt.elspeth-{'7' * 64}.stage.def456.building"
    fresh_building.write_bytes(b"in-flight")
    stage = tmp_path / f".out.txt.elspeth-{effect_id}.stage"
    stage.write_bytes(b"staged")
    lock = tmp_path / ".out.txt.elspeth.lock"
    lock.write_bytes(b"")
    unrelated = tmp_path / "..notes.building"
    unrelated.write_bytes(b"user file")
    old = time.time() - 2 * 60 * 60
    for path in (stale_building, stage, lock, unrelated):
        os.utime(path, (old, old))

    removed = local_effects.cleanup_stale_local_effect_building_files(tmp_path)

    # The one-hour mtime bound is the only shield for a concurrent in-flight
    # prepare's live temp in this parent; its writes keep the mtime fresh.
    assert removed == 1
    assert not stale_building.exists()
    assert fresh_building.exists()
    assert stage.exists()
    assert lock.exists()
    assert unrelated.exists()


def test_prepare_sweeps_stale_crashed_building_files(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    stale_building = tmp_path / f"..out.txt.elspeth-{'9' * 64}.stage.abc123.building"
    stale_building.write_bytes(b"crashed")
    old = time.time() - 2 * 60 * 60
    os.utime(stale_building, (old, old))
    sink = TextSink({"path": str(target), "field": "message", "schema": _SCHEMA})

    _prepare(sink, effect_id="8" * 64, rows=[{"message": "hello"}])

    assert not stale_building.exists()


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

    validate_sink_effect_capability(csv_sink, "write", SinkEffectInputKind.PIPELINE_MEMBERS)
    validate_sink_effect_capability(json_sink, "write", SinkEffectInputKind.PIPELINE_MEMBERS)
    validate_sink_effect_capability(text_sink, "write", SinkEffectInputKind.PIPELINE_MEMBERS)


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


def test_no_publication_uses_exact_virtual_and_inherited_descriptors(tmp_path: Path) -> None:
    target = tmp_path / "empty.txt"
    virtual_inspection = local_effects.inspect_local_effect(
        target_path=target,
        request=SinkEffectInspectionRequest(effect_id="a1" * 32, target="{}", predecessor_descriptor=None),
    )
    virtual = local_effects.prepare_local_effect(
        effect_id="a1" * 32,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
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
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
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
        predecessor_rows=[{"id": 1}, {"id": 2}],
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
        predecessor_rows=[{"id": 1}, {"id": 2}],
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
        predecessor_rows=[{"id": 1}],
    )
    second_sink.commit_effect(second, _CTX)
    assert target.read_text() == json.dumps([{"id": 1}, {"id": 2}], indent=2)
