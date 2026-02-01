# tests/core/landscape/test_exporter.py
"""Tests for LandscapeExporter."""

import pytest

from elspeth.contracts import BatchStatus, NodeStateStatus, NodeType, RoutingMode, RunStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.landscape.exporter import LandscapeExporter

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


@pytest.fixture
def populated_db() -> tuple[LandscapeDB, str]:
    """Create a Landscape with one complete run."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)

    run = recorder.begin_run(config={"test": True}, canonical_version="v1")

    recorder.register_node(
        run_id=run.run_id,
        node_id="source_1",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={"path": "input.csv"},
        schema_config=DYNAMIC_SCHEMA,
    )

    recorder.create_row(
        run_id=run.run_id,
        source_node_id="source_1",
        row_index=0,
        data={"name": "Alice", "value": 100},
    )

    recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

    return db, run.run_id


class TestLandscapeExporterRunMetadata:
    """Exporter extracts run metadata."""

    def test_exporter_extracts_run_metadata(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Exporter should yield run metadata as first record."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        # Find run record
        run_records = [r for r in records if r["record_type"] == "run"]
        assert len(run_records) == 1
        assert run_records[0]["run_id"] == run_id
        assert run_records[0]["status"] == "completed"

    def test_exporter_run_metadata_has_required_fields(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Run record should have all required fields."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        run_record = next(r for r in records if r["record_type"] == "run")

        required_fields = [
            "record_type",
            "run_id",
            "status",
            "started_at",
            "completed_at",
            "canonical_version",
            "config_hash",
            "settings",  # Full resolved settings for audit trail portability
        ]
        for field in required_fields:
            assert field in run_record, f"Missing required field: {field}"

    def test_exporter_run_includes_resolved_settings(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Run record should include parsed settings for audit trail portability."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        run_record = next(r for r in records if r["record_type"] == "run")

        # Settings should be a dict, not a JSON string
        assert isinstance(run_record["settings"], dict), "settings should be parsed dict, not JSON string"
        # Should contain the config we passed to begin_run
        assert run_record["settings"] == {"test": True}


class TestLandscapeExporterRows:
    """Exporter extracts row records."""

    def test_exporter_extracts_rows(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Exporter should yield row records."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        row_records = [r for r in records if r["record_type"] == "row"]
        assert len(row_records) == 1
        assert row_records[0]["row_index"] == 0

    def test_exporter_row_has_required_fields(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Row record should have all required fields."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        row_record = next(r for r in records if r["record_type"] == "row")

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

    def test_exporter_extracts_nodes(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Exporter should yield node records."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))

        node_records = [r for r in records if r["record_type"] == "node"]
        assert len(node_records) == 1
        assert node_records[0]["node_id"] == "source_1"
        assert node_records[0]["plugin_name"] == "csv"

    def test_exporter_node_has_required_fields(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Node record should have all required fields for audit trail portability."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        node_record = next(r for r in records if r["record_type"] == "node")

        required_fields = [
            "record_type",
            "run_id",
            "node_id",
            "plugin_name",
            "node_type",
            "plugin_version",
            "determinism",  # How reproducible is this node?
            "config_hash",
            "config",  # Full resolved config for audit trail portability
            "schema_hash",
            "schema_mode",  # Schema validation mode (dynamic/strict/free/parse/null)
            "schema_fields",  # Explicit field definitions (list or null)
            "sequence_in_pipeline",
        ]
        for field in required_fields:
            assert field in node_record, f"Missing required field: {field}"

    def test_exporter_node_includes_resolved_config(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Node record should include parsed config for audit trail portability."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)

        records = list(exporter.export_run(run_id))
        node_record = next(r for r in records if r["record_type"] == "node")

        # Config should be a dict, not a JSON string
        assert isinstance(node_record["config"], dict), "config should be parsed dict, not JSON string"
        # Should contain the config we passed to register_node
        assert node_record["config"] == {"path": "input.csv"}
        # Determinism should be a string value, not enum object
        assert isinstance(node_record["determinism"], str)
        assert node_record["determinism"] == "deterministic"


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
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_edge(
            run_id=run.run_id,
            from_node_id="source",
            to_node_id="sink",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        token_records = [r for r in records if r["record_type"] == "token"]
        assert len(token_records) == 1
        assert token_records[0]["token_id"] == token.token_id
        assert token_records[0]["row_id"] == row.row_id

    def test_exporter_includes_expand_group_id(self) -> None:
        """Token export includes expand_group_id for deaggregation lineage."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"x": 1},
        )
        parent_token = recorder.create_token(row_id=row.row_id)

        # Use expand_token() to create tokens with expand_group_id set
        expanded_tokens, _expand_group_id = recorder.expand_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            count=2,
            run_id=run.run_id,
            step_in_pipeline=1,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        token_records = [r for r in records if r["record_type"] == "token"]
        # 1 parent + 2 expanded = 3 tokens
        assert len(token_records) == 3

        # Find expanded tokens (they have expand_group_id set)
        expanded_records = [r for r in token_records if r["expand_group_id"] is not None]
        assert len(expanded_records) == 2

        # All expanded tokens should share the same expand_group_id
        expand_group_ids = {r["expand_group_id"] for r in expanded_records}
        assert len(expand_group_ids) == 1

        # Verify the expand_group_id matches what was created
        assert expanded_records[0]["expand_group_id"] == expanded_tokens[0].expand_group_id

    def test_exporter_extracts_node_states(self) -> None:
        """Exporter should yield node_state records."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            node_id="transform",
            plugin_name="passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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
            run_id=run.run_id,
            step_index=0,
            input_data={"x": 1},
        )
        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"x": 1},
            duration_ms=10.0,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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
            run_id=run.run_id,
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
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id=agg.node_id,
        )
        recorder.complete_batch(
            batch_id=batch.batch_id,
            status=BatchStatus.COMPLETED,
            trigger_reason="count=10",
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.AGGREGATION,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
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
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.GATE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id=run.run_id,
            node_id="sink",
            plugin_name="csv",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        edge = recorder.register_edge(
            run_id=run.run_id,
            from_node_id="gate",
            to_node_id="sink",
            label="high_value",
            mode=RoutingMode.MOVE,
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
            run_id=run.run_id,
            step_index=0,
            input_data={},
        )
        recorder.record_routing_event(
            state_id=state.state_id,
            edge_id=edge.edge_id,
            mode=RoutingMode.MOVE,
            reason={"rule": "value > 1000", "matched_value": True},
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

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
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={},
        )
        parent_token = recorder.create_token(row_id=row.row_id)
        _children, _fork_group_id = recorder.fork_token(
            parent_token_id=parent_token.token_id,
            row_id=row.row_id,
            branches=["a", "b"],
            run_id=run.run_id,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        parent_records = [r for r in records if r["record_type"] == "token_parent"]
        # Two children, each with one parent relationship
        assert len(parent_records) == 2
        assert all(r["parent_token_id"] == parent_token.token_id for r in parent_records)


class TestLandscapeExporterSigning:
    """Exporter HMAC signing for legal-grade integrity verification."""

    def test_exporter_signs_records_when_enabled(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """When signing enabled, each record should have signature field."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))

        # All records should have signature
        for record in records:
            assert "signature" in record
            assert len(record["signature"]) == 64  # SHA256 hex

    def test_exporter_manifest_contains_final_hash(self, populated_db: tuple[LandscapeDB, str]) -> None:
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

    def test_exporter_unsigned_has_no_signatures(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """When signing disabled, records should not have signature field."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=False))

        for record in records:
            assert "signature" not in record

        # No manifest without signing
        manifest_records = [r for r in records if r.get("record_type") == "manifest"]
        assert len(manifest_records) == 0

    def test_exporter_raises_when_sign_without_key(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Requesting signing without key should raise ValueError."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db)  # No signing_key

        with pytest.raises(ValueError, match="no signing_key provided"):
            list(exporter.export_run(run_id, sign=True))

    def test_exporter_manifest_record_count_matches(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Manifest record_count should match actual record count (excluding manifest)."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))

        manifest = next(r for r in records if r["record_type"] == "manifest")
        non_manifest_count = len([r for r in records if r["record_type"] != "manifest"])

        assert manifest["record_count"] == non_manifest_count

    def test_exporter_signatures_are_deterministic(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Same data with same key should produce same signatures."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records1 = list(exporter.export_run(run_id, sign=True))
        records2 = list(exporter.export_run(run_id, sign=True))

        # Compare signatures (excluding manifest which has timestamp)
        for r1, r2 in zip(records1, records2, strict=True):
            if r1["record_type"] != "manifest":
                assert r1["signature"] == r2["signature"]

    def test_exporter_different_keys_produce_different_signatures(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Different signing keys should produce different signatures."""
        db, run_id = populated_db
        exporter1 = LandscapeExporter(db, signing_key=b"key-one")
        exporter2 = LandscapeExporter(db, signing_key=b"key-two")

        records1 = list(exporter1.export_run(run_id, sign=True))
        records2 = list(exporter2.export_run(run_id, sign=True))

        # Get first non-manifest record from each
        r1 = next(r for r in records1 if r["record_type"] != "manifest")
        r2 = next(r for r in records2 if r["record_type"] != "manifest")

        assert r1["signature"] != r2["signature"]

    def test_exporter_manifest_includes_algorithm_metadata(self, populated_db: tuple[LandscapeDB, str]) -> None:
        """Manifest should document algorithms used for forensic verification."""
        db, run_id = populated_db
        exporter = LandscapeExporter(db, signing_key=b"test-key-for-hmac")

        records = list(exporter.export_run(run_id, sign=True))
        manifest = next(r for r in records if r["record_type"] == "manifest")

        assert manifest["hash_algorithm"] == "sha256"
        assert manifest["signature_algorithm"] == "hmac-sha256"

    def test_exporter_final_hash_deterministic_with_multiple_records(self) -> None:
        """Final hash must be identical across multiple exports.

        This test creates multiple records of each type to stress-test the
        ordering guarantees added to recorder query methods. Without
        deterministic ORDER BY clauses, the final_hash would differ between
        exports even for identical data.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Create a run with multiple records of each type
        run = recorder.begin_run(config={"test": True}, canonical_version="v1")

        # Multiple nodes (tests get_nodes ordering)
        for i in range(3):
            recorder.register_node(
                run_id=run.run_id,
                node_id=f"node_{i}",
                plugin_name="test",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0.0",
                config={"index": i},
                schema_config=DYNAMIC_SCHEMA,
            )

        # Multiple edges (tests get_edges ordering)
        edges = []
        for i in range(2):
            edge = recorder.register_edge(
                run_id=run.run_id,
                from_node_id=f"node_{i}",
                to_node_id=f"node_{i + 1}",
                label="continue",
                mode=RoutingMode.MOVE,
            )
            edges.append(edge)

        # Multiple rows (tests get_rows ordering)
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="node_0",
                row_index=i,
                data={"value": i * 10},
            )

            # Multiple tokens per row (tests get_tokens ordering)
            for j in range(2):
                token = recorder.create_token(row_id=row.row_id, branch_name=f"branch_{j}")

                # Multiple states per token (tests get_node_states_for_token ordering)
                for k, node_id in enumerate(["node_0", "node_1"]):
                    state = recorder.begin_node_state(
                        token_id=token.token_id,
                        node_id=node_id,
                        run_id=run.run_id,
                        step_index=k,
                        input_data={"x": i * j},
                    )
                    recorder.complete_node_state(
                        state.state_id,
                        status=NodeStateStatus.COMPLETED,
                        output_data={"result": i * j + k},  # Required for COMPLETED states
                        duration_ms=5.0,
                    )

                    # Multiple routing events (tests get_routing_events ordering)
                    if k == 0:
                        recorder.record_routing_event(
                            state_id=state.state_id,
                            edge_id=edges[0].edge_id,  # Use actual edge
                            mode=RoutingMode.MOVE,
                        )

        # Multiple batches (tests get_batches ordering)
        for _ in range(2):
            batch = recorder.create_batch(
                run_id=run.run_id,
                aggregation_node_id="node_1",
            )
            recorder.complete_batch(batch.batch_id, status=BatchStatus.COMPLETED)

        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        # Export multiple times and verify final_hash is identical
        exporter = LandscapeExporter(db, signing_key=b"determinism-test-key")

        final_hashes = []
        for _ in range(5):  # Export 5 times
            records = list(exporter.export_run(run.run_id, sign=True))
            manifest = next(r for r in records if r["record_type"] == "manifest")
            final_hashes.append(manifest["final_hash"])

        # All final hashes must be identical
        assert len(set(final_hashes)) == 1, (
            f"Non-deterministic export detected! Got {len(set(final_hashes))} different hashes: {final_hashes}"
        )

    def test_exporter_record_order_is_stable(self) -> None:
        """Record order must be identical across multiple exports.

        Beyond just the final hash, verify the actual record sequence
        is stable (important for CSV exports and debugging).
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        for i in range(5):
            recorder.register_node(
                run_id=run.run_id,
                node_id=f"node_{i}",
                plugin_name="test",
                node_type=NodeType.TRANSFORM,
                plugin_version="1.0.0",
                config={},
                schema_config=DYNAMIC_SCHEMA,
            )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db)

        # Get record IDs from multiple exports
        exports = []
        for _ in range(3):
            records = list(exporter.export_run(run.run_id))
            node_ids = [r["node_id"] for r in records if r["record_type"] == "node"]
            exports.append(tuple(node_ids))

        # All exports should produce the same order
        assert len(set(exports)) == 1, f"Record order changed between exports: {exports}"


class TestLandscapeExporterCallRecords:
    """P1: Exporter must export external call records."""

    def test_exporter_extracts_calls(self) -> None:
        """P1: Exporter should yield call records for external calls.

        External call records are explicitly part of the audit trail per
        the Data Manifesto. Export regressions that drop or mis-serialize
        calls would break legal-grade auditability.
        """
        from elspeth.contracts.enums import CallStatus, CallType

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            node_id="llm_transform",
            plugin_name="llm_classifier",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"text": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            run_id=run.run_id,
            step_index=0,
            input_data={"text": "test"},
        )

        # Record an LLM call
        call = recorder.record_call(
            state_id=state.state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "classify this", "model": "gpt-4"},
            response_data={"category": "positive", "confidence": 0.95},
            latency_ms=250.5,
        )

        recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data={"category": "positive"},
            duration_ms=300.0,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db)
        records = list(exporter.export_run(run.run_id))

        # P1: Verify call records are extracted
        call_records = [r for r in records if r["record_type"] == "call"]
        assert len(call_records) == 1, f"Expected 1 call record, got {len(call_records)}"

        call_record = call_records[0]
        assert call_record["call_id"] == call.call_id
        assert call_record["state_id"] == state.state_id
        assert call_record["call_type"] == "llm"  # Enum value
        assert call_record["status"] == "success"  # Enum value
        assert call_record["request_hash"] is not None
        assert call_record["response_hash"] is not None
        assert call_record["latency_ms"] == 250.5


class TestLandscapeExporterManifestIntegrity:
    """P1: Verify manifest hash chain is correct, not just that fields exist."""

    def test_manifest_hash_chain_verified(self) -> None:
        """P1: Recompute expected final_hash and verify against manifest.

        Previous tests only checked that final_hash exists. A broken hash
        chain (wrong order, missing records, wrong algorithm) could ship
        without detection, undermining tamper-evidence.
        """
        import hashlib

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={"test": True}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"value": 42},
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        exporter = LandscapeExporter(db, signing_key=b"hash-chain-test-key")
        records = list(exporter.export_run(run.run_id, sign=True))

        # Separate manifest from content records
        manifest = next(r for r in records if r["record_type"] == "manifest")
        content_records = [r for r in records if r["record_type"] != "manifest"]

        # P1: Recompute final_hash from signatures
        running_hash = hashlib.sha256()
        for record in content_records:
            running_hash.update(record["signature"].encode())

        expected_final_hash = running_hash.hexdigest()
        assert manifest["final_hash"] == expected_final_hash, (
            f"Hash chain mismatch: expected {expected_final_hash}, got {manifest['final_hash']}"
        )


class TestLandscapeExporterTier1Corruption:
    """P1: Exporter must crash on corrupted Tier 1 audit data."""

    def test_exporter_crashes_on_invalid_enum_in_database(self) -> None:
        """P1: Invalid enum value in audit DB should crash, not silently coerce.

        Per the Three-Tier Trust Model, corrupted Tier 1 data must crash
        immediately. If invalid enums are silently coerced, auditors could
        receive garbage data that looks valid.
        """
        from sqlalchemy import text

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="source",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )
        recorder.complete_run(run.run_id, status=RunStatus.COMPLETED)

        # Corrupt the run status directly in the database
        with db.connection() as conn:
            conn.execute(
                text("UPDATE runs SET status = 'INVALID_STATUS_VALUE' WHERE run_id = :run_id"),
                {"run_id": run.run_id},
            )
            conn.commit()

        exporter = LandscapeExporter(db)

        # P1: Exporter must crash on invalid status, not return garbage
        with pytest.raises((ValueError, KeyError)):
            list(exporter.export_run(run.run_id))
