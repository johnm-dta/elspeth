"""Tests for ``elspeth.web.execution.outputs.load_run_outputs_from_db``.

Unlike the diagnostics endpoint (which caps at _ARTIFACT_PREVIEW_LIMIT=20
for operator-UI pacing), this loader returns the FULL artifact list so a
downstream caller — the eval harness or audit-evidence retrieval flow —
can enumerate and re-fetch every sink output the run produced.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.web.execution.outputs import load_run_outputs_for_settings, load_run_outputs_from_db

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


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
        row = factory.data_flow.create_row(run_id, "src", i, {"x": i}, row_id=f"row-{i}")
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
    settings = MagicMock()
    settings.get_landscape_url.return_value = f"sqlite:///{tmp_path / 'missing-audit.db'}"
    settings.landscape_passphrase = None

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


def test_load_run_outputs_returns_empty_artifacts_for_unknown_run(tmp_path: Path) -> None:
    with LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}") as db:
        response = load_run_outputs_from_db(db, run_id="nope", landscape_run_id="nope")
        assert response.run_id == "nope"
        assert response.artifacts == []
