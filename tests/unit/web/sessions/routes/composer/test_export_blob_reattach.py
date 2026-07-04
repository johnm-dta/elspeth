"""Unit tests for _reattach_guided_blob_refs (elspeth-b5ee205720).

A guided blob-backed source has ``blob_ref`` stripped from its committed
options (the manual set_source path can't prove ``path == storage_path``); it
survives only in the GuidedSession snapshot's ``step_1_result``. The export
sidecar and the public-YAML path-omit both key off ``source.options["blob_ref"]``,
so without reattachment the export leaks the raw storage path AND emits no
``source_blob_ids`` (breaking the re-import round-trip). This helper reconstitutes
``blob_ref`` into the export's working copy from the snapshot, mirroring the
cross-reference in redact_guided_snapshot_storage_paths.
"""

from __future__ import annotations

from dataclasses import replace

from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.state import CompositionState, PipelineMetadata, SourceSpec
from elspeth.web.composer.yaml_generator import generate_public_pipeline_dict
from elspeth.web.sessions.routes.composer.state import _reattach_guided_blob_refs

BLOB_PATH = "/data/blobs/sess-1/abc12300-0000-4000-8000-000000000000_data.csv"
BLOB_REF = "abc12300-0000-4000-8000-000000000000"


def _guided_with_snapshot(*, blob_ref: str | None, path: str) -> GuidedSession:
    options: dict[str, object] = {"path": path, "schema": {"mode": "observed"}}
    if blob_ref is not None:
        options["blob_ref"] = blob_ref
    snap = SourceResolved(
        plugin="csv",
        options=options,
        observed_columns=("data.csv",),
        sample_rows=(),
    )
    return replace(GuidedSession.initial(), step_1_result=snap)


def _state(*, source_options: dict[str, object], guided_session: GuidedSession | None) -> CompositionState:
    return CompositionState(
        sources={
            "source": SourceSpec(
                plugin="csv",
                on_success="main",
                options=source_options,
                on_validation_failure="discard",
            )
        },
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
        guided_session=guided_session,
    )


def test_reattaches_blob_ref_from_guided_snapshot() -> None:
    state = _state(
        source_options={"path": BLOB_PATH, "schema": {"mode": "observed"}},
        guided_session=_guided_with_snapshot(blob_ref=BLOB_REF, path=BLOB_PATH),
    )
    out = _reattach_guided_blob_refs(state)
    assert out.sources["source"].options["blob_ref"] == BLOB_REF
    # The reattached working copy keeps the path; omission happens in yaml gen.
    assert out.sources["source"].options["path"] == BLOB_PATH
    # Non-mutating: the input state's source is untouched.
    assert "blob_ref" not in state.sources["source"].options


def test_reattached_state_omits_path_and_yields_sidecar() -> None:
    state = _state(
        source_options={"path": BLOB_PATH, "schema": {"mode": "observed"}},
        guided_session=_guided_with_snapshot(blob_ref=BLOB_REF, path=BLOB_PATH),
    )
    out = _reattach_guided_blob_refs(state)
    # Public YAML now omits the storage path (blob_ref present triggers the
    # existing omit) and strips the web-only blob_ref itself.
    doc = generate_public_pipeline_dict(out)
    src_opts = doc["sources"]["source"]["options"]
    assert "path" not in src_opts
    assert "blob_ref" not in src_opts
    # The export sidecar comprehension now finds a blob_ref to emit.
    sidecar = {name: str(s.options["blob_ref"]) for name, s in out.sources.items() if "blob_ref" in s.options}
    assert sidecar == {"source": BLOB_REF}


def test_untouched_without_guided_session() -> None:
    state = _state(source_options={"path": BLOB_PATH}, guided_session=None)
    assert _reattach_guided_blob_refs(state) is state


def test_untouched_when_snapshot_has_no_blob_ref() -> None:
    state = _state(
        source_options={"path": BLOB_PATH},
        guided_session=_guided_with_snapshot(blob_ref=None, path=BLOB_PATH),
    )
    assert _reattach_guided_blob_refs(state) is state


def test_does_not_touch_operator_typed_source() -> None:
    # The committed source path does not match the snapshot's blob-backed path,
    # so it is an operator-typed path and must NOT be marked blob-backed.
    state = _state(
        source_options={"path": "/tmp/operator/typed.csv"},
        guided_session=_guided_with_snapshot(blob_ref=BLOB_REF, path=BLOB_PATH),
    )
    out = _reattach_guided_blob_refs(state)
    assert "blob_ref" not in out.sources["source"].options


def test_preserves_source_that_already_has_blob_ref() -> None:
    state = _state(
        source_options={"path": BLOB_PATH, "blob_ref": "already-bound"},
        guided_session=_guided_with_snapshot(blob_ref=BLOB_REF, path=BLOB_PATH),
    )
    out = _reattach_guided_blob_refs(state)
    assert out.sources["source"].options["blob_ref"] == "already-bound"
