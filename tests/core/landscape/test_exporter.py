# tests/core/landscape/test_exporter.py
"""Tests for LandscapeExporter."""

import pytest

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.landscape.exporter import LandscapeExporter


@pytest.fixture
def populated_db():
    """Create a Landscape with one complete run."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={"test": True}, canonical_version="v1")

    recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="csv",
        node_type="source",
        plugin_version="1.0.0",
        config={"path": "input.csv"},
    )

    row = recorder.create_row(
        run_id=run.run_id,
        source_node_id="source_1",
        row_index=0,
        data={"name": "Alice", "value": 100},
    )

    recorder.complete_run(run.run_id, status="completed")

    return db, run.run_id


class TestLandscapeExporterRunMetadata:
    """Exporter extracts run metadata."""

    def test_exporter_extracts_run_metadata(self, populated_db) -> None:
        """Exporter should yield run metadata as first record."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        # Find run record
        run_records = [r for r in records if r["record_type"] == "run"]
        assert len(run_records) == 1
        assert run_records[0]["run_id"] == run_id
        assert run_records[0]["status"] == "completed"

    def test_exporter_run_metadata_has_required_fields(self, populated_db) -> None:
        """Run record should have all required fields."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        run_record = [r for r in records if r["record_type"] == "run"][0]

        required_fields = [
            "record_type",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "canonical_version",
            "config_hash",
        ]
        for field in required_fields:
            assert field in run_record, f"Missing required field: {field}"


class TestLandscapeExporterRows:
    """Exporter extracts row records."""

    def test_exporter_extracts_rows(self, populated_db) -> None:
        """Exporter should yield row records."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        row_records = [r for r in records if r["record_type"] == "row"]
        assert len(row_records) == 1
        assert row_records[0]["row_index"] == 0

    def test_exporter_row_has_required_fields(self, populated_db) -> None:
        """Row record should have all required fields."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        row_record = [r for r in records if r["record_type"] == "row"][0]

        required_fields = [
            "record_type",
            "run_id",
            "row_id",
            "row_index",
            "source_node_id",
            "source_data_hash",
        ]
        for field in required_fields:
            assert field in row_record, f"Missing required field: {field}"


class TestLandscapeExporterNodes:
    """Exporter extracts node records."""

    def test_exporter_extracts_nodes(self, populated_db) -> None:
        """Exporter should yield node records."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        node_records = [r for r in records if r["record_type"] == "node"]
        assert len(node_records) == 1
        assert node_records[0]["node_id"] == "source_1"
        assert node_records[0]["plugin_name"] == "csv"


class TestLandscapeExporterErrors:
    """Exporter error handling."""

    def test_exporter_raises_for_missing_run(self) -> None:
        """Exporter should raise ValueError for missing run."""
        db = LandscapeDB.in_memory()
        exporter = LandscapeExporter(db)

        with pytest.raises(ValueError, match="Run not found"):
            list(exporter.export_run("nonexistent_run_id"))


class TestLandscapeExporterComplexRun:
    """Exporter with complex pipeline data."""

    def test_exporter_extracts_edges(self) -> None:
        """Exporter should yield edge records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source",
            to_node_id="sink",
            label="continue",
            mode="move",
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        edge_records = [r for r in records if r["record_type"] == "edge"]
        assert len(edge_records) == 1
        assert edge_records[0]["from_node_id"] == "source"
        assert edge_records[0]["to_node_id"] == "sink"
        assert edge_records[0]["label"] == "continue"
        assert edge_records[0]["default_mode"] == "move"

    def test_exporter_extracts_tokens(self) -> None:
        """Exporter should yield token records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        token_records = [r for r in records if r["record_type"] == "token"]
        assert len(token_records) == 1
        assert token_records[0]["token_id"] == token.token_id
        assert token_records[0]["row_id"] == row.row_id

    def test_exporter_extracts_node_states(self) -> None:
        """Exporter should yield node_state records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            node_id="transform",
            plugin_name="passthrough",
            node_type="transform",
            plugin_version="1.0.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"x": 1},
        )
        recorder.complete_node_state(
            state_id=state.state_id,
            status="completed",
            output_data={"x": 1},
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        state_records = [r for r in records if r["record_type"] == "node_state"]
        assert len(state_records) == 1
        assert state_records[0]["token_id"] == token.token_id
        assert state_records[0]["node_id"] == node.node_id
        assert state_records[0]["status"] == "completed"

    def test_exporter_extracts_artifacts(self) -> None:
        """Exporter should yield artifact records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        sink = recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=sink.node_id,
            step_index=0,
            input_data={},
        )
        recorder.register_artifact(
            run_id=run.run_id,
            state_id=state.state_id,
            sink_node_id=sink.node_id,
            artifact_type="csv",
            path="/output/result.csv",
            content_hash="abc123",
            size_bytes=1024,
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        artifact_records = [r for r in records if r["record_type"] == "artifact"]
        assert len(artifact_records) == 1
        assert artifact_records[0]["sink_node_id"] == sink.node_id
        assert artifact_records[0]["content_hash"] == "abc123"
        assert artifact_records[0]["artifact_type"] == "csv"

    def test_exporter_extracts_batches(self) -> None:
        """Exporter should yield batch records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            node_id="aggregator",
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0.0",
            config={},
        )
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.complete_batch(
            batch_id=batch.batch_id,
            status="completed",
            trigger_reason="count=10",
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        batch_records = [r for r in records if r["record_type"] == "batch"]
        assert len(batch_records) == 1
        assert batch_records[0]["batch_id"] == batch.batch_id
        assert batch_records[0]["aggregation_node_id"] == agg.node_id
        assert batch_records[0]["status"] == "completed"

    def test_exporter_extracts_batch_members(self) -> None:
        """Exporter should yield batch_member records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        agg = recorder.register_node(
            run_id=run.run_id,
            node_id="aggregator",
            plugin_name="sum",
            node_type="aggregation",
            plugin_version="1.0.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=agg.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.add_batch_member(
            batch_id=batch.batch_id,
            token_id=token.token_id,
            ordinal=0,
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        member_records = [r for r in records if r["record_type"] == "batch_member"]
        assert len(member_records) == 1
        assert member_records[0]["batch_id"] == batch.batch_id
        assert member_records[0]["token_id"] == token.token_id

    def test_exporter_extracts_routing_events(self) -> None:
        """Exporter should yield routing_event records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        gate = recorder.register_node(
            run_id=run.run_id,
            node_id="gate",
            plugin_name="threshold",
            node_type="gate",
            plugin_version="1.0.0",
            config={},
        )
        sink = recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type="sink",
            plugin_version="1.0.0",
            config={},
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink",
            label="high_value",
            mode="move",
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=gate.node_id,
            row_index=0,
            data={},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=gate.node_id,
            step_index=0,
            input_data={},
        )
        recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode="move",
            reason={"rule": "value > 1000"},
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        event_records = [r for r in records if r["record_type"] == "routing_event"]
        assert len(event_records) == 1
        assert event_records[0]["state_id"] == state.state_id
        assert event_records[0]["edge_id"] == edge.edge_id

    def test_exporter_extracts_token_parents(self) -> None:
        """Exporter should yield token_parent records for forks."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type="source",
            plugin_version="1.0.0",
            config={},
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)
        _children = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
        )
        recorder.complete_run(run.run_id, status="completed")

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        parent_records = [r for r in records if r["record_type"] == "token_parent"]
        # Two children, each with one parent relationship
        assert len(parent_records) == 2
        assert all(r["parent_token_id"] == parent_token.token_id for r in parent_records)


class TestLandscapeExporterSigning:
    """Exporter HMAC signing for legal-grade integrity verification."""

    def test_exporter_signs_records_when_enabled(self, populated_db) -> None:
        """When signing enabled, each record should have signature field."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))

        # All records should have signature
        for record in records:
            assert "signature" in record
            assert len(record["signature"]) == 64  # SHA256 hex

    def test_exporter_manifest_contains_final_hash(self, populated_db) -> None:
        """Signed export should include manifest with hash of all records."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))

        manifest_records = [r for r in records if r["record_type"] == "manifest"]
        assert len(manifest_records) == 1

        manifest = manifest_records[0]
        assert "record_count" in manifest
        assert "final_hash" in manifest
        assert "exported_at" in manifest  # Timestamp for forensics

    def test_exporter_unsigned_has_no_signatures(self, populated_db) -> None:
        """When signing disabled, records should not have signature field."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=False))

        for record in records:
            assert "signature" not in record

        # No manifest without signing
        manifest_records = [r for r in records if r.get("record_type") == "manifest"]
        assert len(manifest_records) == 0

    def test_exporter_raises_when_sign_without_key(self, populated_db) -> None:
        """Requesting signing without key should raise ValueError."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)  # No signing_key

        with pytest.raises(ValueError, match="no signing_key provided"):
            list(exporter.export_run(run_id, sign=True))

    def test_exporter_manifest_record_count_matches(self, populated_db) -> None:
        """Manifest record_count should match actual record count (excluding manifest)."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))

        manifest = [r for r in records if r["record_type"] == "manifest"][0]
        non_manifest_count = len([r for r in records if r["record_type"] != "manifest"])

        assert manifest["record_count"] == non_manifest_count

    def test_exporter_signatures_are_deterministic(self, populated_db) -> None:
        """Same data with same key should produce same signatures."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records1 = list(exporter.export_run(run_id, sign=True))
        records2 = list(exporter.export_run(run_id, sign=True))

        # Compare signatures (excluding manifest which has timestamp)
        for r1, r2 in zip(records1, records2, strict=True):
            if r1["record_type"] != "manifest":
                assert r1["signature"] == r2["signature"]

    def test_exporter_different_keys_produce_different_signatures(
        self, populated_db
    ) -> None:
        """Different signing keys should produce different signatures."""
        db, run_id = populated_db
        exporter1 = LandscapeExporter(db, signing_key=b"key-one")
        exporter2 = LandscapeExporter(db, signing_key=b"key-two")

        records1 = list(exporter1.export_run(run_id, sign=True))
        records2 = list(exporter2.export_run(run_id, sign=True))

        # Get first non-manifest record from each
        r1 = [r for r in records1 if r["record_type"] != "manifest"][0]
        r2 = [r for r in records2 if r["record_type"] != "manifest"][0]

        assert r1["signature"] != r2["signature"]

    def test_exporter_manifest_includes_algorithm_metadata(self, populated_db) -> None:
        """Manifest should document algorithms used for forensic verification."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))
        manifest = [r for r in records if r["record_type"] == "manifest"][0]

        assert manifest["hash_algorithm"] == "sha256"
        assert manifest["signature_algorithm"] == "hmac-sha256"
