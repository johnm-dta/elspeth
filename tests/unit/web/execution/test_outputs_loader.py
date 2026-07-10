"""Tests for ``elspeth.web.execution.outputs.load_run_outputs_from_db``.

Unlike the diagnostics endpoint (which caps at _ARTIFACT_PREVIEW_LIMIT=20
for operator-UI pacing), this loader returns the FULL artifact list so a
downstream caller — the eval harness or audit-evidence retrieval flow —
can enumerate and re-fetch every sink output the run produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.web.execution.outputs import load_run_outputs_for_settings, load_run_outputs_from_db, path_or_uri_to_filesystem_path

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


@dataclass(frozen=True)
class _SettingsFake:
    landscape_url: str
    landscape_passphrase: str | None = None

    def get_landscape_url(self) -> str:
        return self.landscape_url


def _register_node(
    factory: RecorderFactory,
    run_id: str,
    node_id: str,
    node_type: NodeType,
    plugin_name: str,
) -> None:
    factory.data_flow.register_node(
        run_id=run_id,
        node_id=node_id,
        plugin_name=plugin_name,
        node_type=node_type,
        plugin_version="1.0",
        config={},
        schema_config=_OBSERVED_SCHEMA,
    )


def _seed_run_with_artifacts(
    db: LandscapeDB,
    run_id: str,
    artifacts: list[tuple[str, str, str, int, str]],
) -> None:
    """Insert a run with one sink + one state-per-artifact, then register
    each artifact. ``artifacts`` is a list of
    ``(artifact_id, path_or_uri, content_hash, size_bytes, sink_node_id)``.
    """
    factory = RecorderFactory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    _register_node(factory, run_id, "src", NodeType.SOURCE, "csv")
    sink_nodes_seen: set[str] = set()
    for i, (artifact_id, path_or_uri, content_hash, size_bytes, sink_node_id) in enumerate(artifacts):
        if sink_node_id not in sink_nodes_seen:
            _register_node(factory, run_id, sink_node_id, NodeType.SINK, "csv")
            sink_nodes_seen.add(sink_node_id)
        row = factory.data_flow.create_row(run_id, "src", i, {"x": i}, row_id=f"row-{i}", source_row_index=i, ingest_sequence=i)
        token = factory.data_flow.create_token(row.row_id, token_id=f"tok-{i}")
        state = factory.execution.begin_node_state(
            token.token_id,
            sink_node_id,
            run_id,
            1,
            {"x": i},
            state_id=f"state-{i}",
        )
        factory.execution.register_artifact(
            run_id,
            state.state_id,
            sink_node_id,
            "file",
            path_or_uri,
            content_hash,
            size_bytes,
            artifact_id=artifact_id,
        )


def test_load_run_outputs_returns_all_artifacts_without_preview_cap(tmp_path: Path) -> None:
    """Diagnostics caps at 20 artifacts; outputs MUST return them all."""
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-1"
        artifacts = [(f"art-{i:02d}", str(tmp_path / f"sink_{i}.csv"), "a" * 64, 100 + i, f"sink_{i}") for i in range(25)]
        for artifact_id, path, _hash, _size, _sink in artifacts:
            Path(path).write_bytes(b"x" * (100 + int(artifact_id.split("-")[1])))
        _seed_run_with_artifacts(db, run_id, artifacts)

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)

        assert response.run_id == run_id
        assert response.landscape_run_id == run_id
        assert len(response.artifacts) == 25


def test_load_run_outputs_records_exists_now_filesystem_check(tmp_path: Path) -> None:
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-2"
        present = tmp_path / "present.csv"
        present.write_bytes(b'{"a":1}\n')
        absent = tmp_path / "deleted.csv"
        absent.write_bytes(b"will-delete\n")
        absent.unlink()

        _seed_run_with_artifacts(
            db,
            run_id,
            [
                ("art-present", str(present), "a" * 64, present.stat().st_size, "sink_p"),
                ("art-absent", str(absent), "b" * 64, 0, "sink_a"),
            ],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)
        by_id = {a.artifact_id: a for a in response.artifacts}
        assert by_id["art-present"].exists_now is True
        assert by_id["art-absent"].exists_now is False


def test_load_run_outputs_for_settings_raises_when_sqlite_audit_db_missing(tmp_path: Path) -> None:
    settings = _SettingsFake(landscape_url=f"sqlite:///{tmp_path / 'missing-audit.db'}")

    with pytest.raises(RuntimeError, match="audit database"):
        load_run_outputs_for_settings(
            settings,
            run_id="web-run-missing-audit",
            landscape_run_id="landscape-run-missing-audit",
        )


def test_load_run_outputs_rejects_corrupt_artifact_types(tmp_path: Path) -> None:
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-corrupt-artifact"
        output_path = tmp_path / "result.csv"
        output_path.write_text("x\n")
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(output_path), "a" * 64, output_path.stat().st_size, "sink_r")],
        )

        with db.connection() as conn:
            conn.execute(text("UPDATE artifacts SET size_bytes = 7.5 WHERE artifact_id = 'art-1'"))

        with pytest.raises(ValidationError, match="size_bytes"):
            load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)


def test_load_run_outputs_strips_file_uri_prefix_in_exists_check(tmp_path: Path) -> None:
    """Sinks record path_or_uri as ``file:///abs/path`` for filesystem outputs."""
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-3"
        target = tmp_path / "results.jsonl"
        target.write_bytes(b'{"x":1}\n')

        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", f"file://{target}", "a" * 64, target.stat().st_size, "sink_r")],
        )
        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)
        assert response.artifacts[0].exists_now is True


def test_load_run_outputs_decodes_file_uri_path_for_exists_check(tmp_path: Path) -> None:
    """Encoded file URI delimiters are literal filename bytes, not URI query params."""
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-encoded-file-uri"
        target = tmp_path / "results?token=literal.jsonl"
        target.write_bytes(b'{"x":1}\n')

        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", f"file://{tmp_path}/results%3Ftoken%3Dliteral.jsonl", "a" * 64, target.stat().st_size, "sink_r")],
        )
        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)
        assert response.artifacts[0].exists_now is True


def test_load_run_outputs_preserves_legacy_raw_percent_file_uri_when_it_exists(tmp_path: Path) -> None:
    """Historical raw file:// rows may contain literal percent-looking filenames."""
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-legacy-raw-percent-file-uri"
        legacy_target = tmp_path / "results%3Ftoken=literal.jsonl"
        decoded_target = tmp_path / "results?token=literal.jsonl"
        legacy_target.write_bytes(b'{"legacy":true}\n')

        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", f"file://{tmp_path}/results%3Ftoken=literal.jsonl", "a" * 64, legacy_target.stat().st_size, "sink_r")],
        )
        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)

        assert response.artifacts[0].exists_now is True
        assert legacy_target.exists()
        assert not decoded_target.exists()


def test_file_uri_path_resolution_prefers_decoded_candidate_when_both_exist(tmp_path: Path) -> None:
    raw_target = tmp_path / "results%3Ftoken%3Dliteral.jsonl"
    decoded_target = tmp_path / "results?token=literal.jsonl"
    raw_target.write_bytes(b'{"raw":true}\n')
    decoded_target.write_bytes(b'{"decoded":true}\n')

    resolved = path_or_uri_to_filesystem_path(f"file://{tmp_path}/results%3Ftoken%3Dliteral.jsonl")

    assert resolved == decoded_target


def test_load_run_outputs_returns_empty_artifacts_for_unknown_run(tmp_path: Path) -> None:
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        response = load_run_outputs_from_db(db, run_id="nope", landscape_run_id="nope")
        assert response.run_id == "nope"
        assert response.artifacts == []


# ── downloadable field tests ─────────────────────────────────────────


def test_downloadable_true_when_file_inside_allowlist_and_exists(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    target = outputs_dir / "results.csv"
    target.write_bytes(b"col1\n1\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-dl-1"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].downloadable is True


def test_downloadable_false_when_file_inside_allowlist_but_missing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    target = outputs_dir / "vanished.csv"
    target.write_bytes(b"col1\n1\n")
    target.unlink()

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-dl-2"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, 7, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].downloadable is False
    assert response.artifacts[0].exists_now is False


def test_downloadable_false_when_file_exists_but_outside_allowlist(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "outputs").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rogue = elsewhere / "rogue.csv"
    rogue.write_bytes(b"escaped\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-dl-3"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(rogue), "a" * 64, rogue.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].exists_now is True
    assert response.artifacts[0].downloadable is False


def test_downloadable_false_when_data_dir_omitted_safe_default(tmp_path: Path) -> None:
    # Legacy callers (eval harness, test fixtures) that don't pass
    # data_dir get downloadable=False everywhere — the safe degraded
    # response: "the UI can't trust this server to serve bytes."
    target = tmp_path / "anywhere.csv"
    target.write_bytes(b"x\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-dl-4"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)

    assert response.artifacts[0].downloadable is False


# ── storage_kind classification tests (elspeth-52af16f9ae) ────────────
#
# storage_kind is the structured discriminator that replaces the
# frontend's regex-based "does this look like a blob-storage path"
# heuristic. It is computed against the REAL storage layouts:
#   * composer blob store:  {data_dir}/blobs/{session_id}/{blob_id}_{filename}
#   * payload store:        {data_dir}/payloads/<2-hex>/<hash>
#   * canonical sink output: {data_dir}/outputs/...
# anything else is "unknown" — the safe default that does not claim a
# path is internal storage it cannot verify.


def test_storage_kind_blob_for_path_under_blobs_directory_even_when_absent(tmp_path: Path) -> None:
    """The critical case: a blob-store path that was PURGED (never
    written, or deleted) must still classify as "blob" so the UI never
    falls back to rendering the raw internal path as the row title or
    the "no longer available on disk" tooltip. Classification must not
    be gated on the file actually existing on disk.
    """
    data_dir = tmp_path / "data"
    session_id = "11111111-1111-1111-1111-111111111111"
    # Deliberately never created — mirrors a blob whose file was purged
    # between the sink write and this manifest read.
    target = data_dir / "blobs" / session_id / "blob-abc_report.json"

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-blob-absent"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, 0, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].exists_now is False
    assert response.artifacts[0].storage_kind == "blob"


def test_storage_kind_blob_for_path_under_blobs_directory_when_present(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    session_id = "22222222-2222-2222-2222-222222222222"
    blob_dir = data_dir / "blobs" / session_id
    blob_dir.mkdir(parents=True)
    target = blob_dir / "blob-xyz_report.json"
    target.write_bytes(b'{"x":1}\n')

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-blob-present"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].exists_now is True
    assert response.artifacts[0].storage_kind == "blob"


def test_storage_kind_payload_for_path_under_payloads_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    payload_dir = data_dir / "payloads" / "ab"
    payload_dir.mkdir(parents=True)
    content_hash = "ab" + "c" * 62
    target = payload_dir / content_hash
    target.write_bytes(b"payload-bytes")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-payload"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].storage_kind == "payload"


def test_storage_kind_payload_honours_payload_root_override(tmp_path: Path) -> None:
    # WebSettings.payload_store_path can relocate the payload store outside
    # data_dir; classification must follow the configured root, not the
    # data_dir-relative default (review-0.7.1 batch E follow-up).
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    override_root = tmp_path / "elsewhere" / "payload-store"
    payload_dir = override_root / "ab"
    payload_dir.mkdir(parents=True)
    content_hash = "ab" + "d" * 62
    target = payload_dir / content_hash
    target.write_bytes(b"payload-bytes")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-payload-override"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        without_override = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)
        with_override = load_run_outputs_from_db(
            db,
            run_id=run_id,
            landscape_run_id=run_id,
            data_dir=data_dir,
            payload_root=override_root,
        )

    assert without_override.artifacts[0].storage_kind == "unknown"
    assert with_override.artifacts[0].storage_kind == "payload"


def test_storage_kind_sink_file_for_path_under_outputs_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    outputs_dir = data_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    target = outputs_dir / "results.csv"
    target.write_bytes(b"col1\n1\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-sink-file"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].storage_kind == "sink_file"
    # Downloadable classification is orthogonal to storage_kind — a
    # sink-allowlisted file keeps its downloadable=True regardless of
    # which kind it was classified as.
    assert response.artifacts[0].downloadable is True


def test_storage_kind_unknown_for_path_outside_known_layouts(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "outputs").mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    rogue = elsewhere / "rogue.csv"
    rogue.write_bytes(b"escaped\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-unknown"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(rogue), "a" * 64, rogue.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].storage_kind == "unknown"


def test_storage_kind_unknown_for_lookalike_directory_name(tmp_path: Path) -> None:
    """A directory that merely starts with "blobs" (e.g. a user-configured
    sink writing into "blobs-extra/") must NOT classify as the internal
    blob store. Regression guard for a naive string-prefix check instead
    of a real path-relative-to test.
    """
    data_dir = tmp_path / "data"
    lookalike_dir = data_dir / "blobs-extra"
    lookalike_dir.mkdir(parents=True)
    target = lookalike_dir / "sha256-abc_report.json"
    target.write_bytes(b"x")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-lookalike"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id, data_dir=data_dir)

    assert response.artifacts[0].storage_kind == "unknown"


def test_storage_kind_unknown_when_data_dir_omitted(tmp_path: Path) -> None:
    # Legacy callers that omit data_dir get the same safe degradation as
    # downloadable=False: no claim can be verified against a real
    # storage layout, so everything reports "unknown".
    target = tmp_path / "anywhere.csv"
    target.write_bytes(b"x\n")

    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        run_id = "web-run-storage-kind-no-data-dir"
        _seed_run_with_artifacts(
            db,
            run_id,
            [("art-1", str(target), "a" * 64, target.stat().st_size, "sink_r")],
        )

        response = load_run_outputs_from_db(db, run_id=run_id, landscape_run_id=run_id)

    assert response.artifacts[0].storage_kind == "unknown"
