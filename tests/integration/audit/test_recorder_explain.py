"""Tests for RecorderFactory explain functionality and graceful degradation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from elspeth.contracts import NodeType, PipelineRow
from elspeth.contracts.audit import NodeStateCompleted
from elspeth.contracts.errors import AuditIntegrityError, CoalesceCollisionError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.config import CoalesceSettings, GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.engine.orchestrator import Orchestrator
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import _TestSchema
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.plugins import CollectSink
from tests.integration.pipeline.orchestrator.test_branch_transforms import _build_branch_pipeline

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


class TestExplainGracefulDegradation:
    """Tests for explain_row() when payloads are unavailable."""

    def test_explain_with_missing_row_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() succeeds even when row payload is purged."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        row_data = {"name": "test", "value": 42}

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
        )

        # Purge the payload (simulate retention policy)
        payload_store.delete(row.source_data_ref)

        # explain_row should still work
        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None  # Hash preserved
        assert lineage.source_data is None  # Payload unavailable
        assert lineage.payload_available is False

    def test_explain_reports_payload_status(self, tmp_path: Path, payload_store) -> None:
        """explain_row() explicitly reports payload availability."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        row_data = {"name": "test"}

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
        )

        # Purge the payload
        payload_store.delete(row.source_data_ref)

        # Check payload_available attribute
        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.payload_available is False

    def test_explain_with_available_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() returns payload when available."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        # Set up with payload store
        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores payload via configured payload_store
        row_data = {"name": "test", "value": 123}

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
        )

        # Payload NOT purged
        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data is not None  # Payload available
        assert lineage.source_data == row_data
        assert lineage.payload_available is True

    def test_explain_row_not_found(self) -> None:
        """explain_row() returns None when row doesn't exist."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id="nonexistent",
        )

        assert lineage is None

    def test_explain_row_without_payload_store(self) -> None:
        """explain_row() works when no payload store is configured."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        factory = RecorderFactory(db)  # No payload store

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload store
        assert lineage.payload_available is False

    def test_explain_row_with_no_payload_ref(self, tmp_path: Path) -> None:
        """explain_row() handles rows when no payload_store is configured.

        When RecorderFactory is created without a payload_store, rows are
        created without payload storage (payload_ref is None).
        """
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory

        db = LandscapeDB.in_memory()
        # No payload_store configured - payloads won't be stored
        factory = RecorderFactory(db, payload_store=None)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row — no payload_store configured, so source_data_ref will be None
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        lineage = factory.query.explain_row(
            run_id=run.run_id,
            row_id=row.row_id,
        )

        assert lineage is not None
        assert lineage.source_data_hash is not None
        assert lineage.source_data is None  # No payload store configured
        assert lineage.payload_available is False

    def test_explain_row_with_corrupted_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() crashes on corrupted payload — Tier 1 integrity violation."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores valid canonical JSON via payload_store
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Store corrupted (non-JSON) data separately with a valid hash
        bad_ref = payload_store.store(b"this is not valid json {{{{")

        # Point the row's source_data_ref to the corrupted payload
        from elspeth.core.landscape.schema import rows_table

        with db.engine.connect() as conn:
            conn.execute(rows_table.update().where(rows_table.c.row_id == row.row_id).values(source_data_ref=bad_ref))
            conn.commit()

        # Tier 1 violation: corrupted payload store data is OUR data — must crash
        with pytest.raises(AuditIntegrityError, match="Corrupt payload"):
            factory.query.explain_row(
                run_id=run.run_id,
                row_id=row.row_id,
            )

    def test_explain_row_with_non_object_payload(self, tmp_path: Path, payload_store) -> None:
        """explain_row() rejects non-object JSON payloads as corruption."""
        import json

        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # create_row auto-stores valid canonical JSON via payload_store
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data={"name": "test"},
        )

        # Store non-object JSON separately with a valid hash
        bad_ref = payload_store.store(json.dumps([1, 2, 3]).encode())

        # Point the row's source_data_ref to the non-object payload
        from elspeth.core.landscape.schema import rows_table

        with db.engine.connect() as conn:
            conn.execute(rows_table.update().where(rows_table.c.row_id == row.row_id).values(source_data_ref=bad_ref))
            conn.commit()

        with pytest.raises(AuditIntegrityError, match="expected JSON object"):
            factory.query.explain_row(
                run_id=run.run_id,
                row_id=row.row_id,
            )

    def test_explain_row_rejects_run_id_mismatch(self, tmp_path: Path, payload_store) -> None:
        """explain_row() raises ValueError when row belongs to different run."""
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.factory import RecorderFactory
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        factory = RecorderFactory(db, payload_store=payload_store)

        # Create two runs
        run1 = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        run2 = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")

        source = factory.data_flow.register_node(
            run_id=run1.run_id,
            plugin_name="csv_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Create row in run1 (create_row auto-stores payload)
        row_data = {"name": "test"}
        row = factory.data_flow.create_row(
            run_id=run1.run_id,
            source_node_id=source.node_id,
            row_index=0,
            data=row_data,
        )

        # Try to explain using run2's ID — cross-run mismatch raises AuditIntegrityError
        with pytest.raises(AuditIntegrityError, match=f"Row {row.row_id} belongs to run {run1.run_id}, not {run2.run_id}"):
            factory.query.explain_row(
                run_id=run2.run_id,  # Wrong run!
                row_id=row.row_id,
            )

        # Same row with correct run_id should work
        lineage_correct = factory.query.explain_row(
            run_id=run1.run_id,
            row_id=row.row_id,
        )

        assert lineage_correct is not None
        assert lineage_correct.row_id == row.row_id


# ---------------------------------------------------------------------------
# Union-merge field provenance: end-to-end audit trail verification
# ---------------------------------------------------------------------------
#
# These tests exercise the production code path (ExecutionGraph.from_plugin_instances
# → Orchestrator.run) to verify that union merge field provenance
# (union_field_origins, union_field_collision_values) is persisted to the
# Landscape audit trail via node_states.context_after_json. Covers both the
# success path (last_wins default policy) and the failure path (fail policy
# raises CoalesceCollisionError, audit trail still captures the full collision
# record via the executor's cleanup handler).


class _ScoreATransform(BaseTransform):
    """Adds a 'score' field with value 10 — collides with _ScoreBTransform."""

    name = "score_a"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        data["score"] = 10
        data["field_a"] = "from_a"
        return TransformResult.success(
            make_pipeline_row(data),
            success_reason={"action": "score_a"},
        )


class _ScoreBTransform(BaseTransform):
    """Adds a 'score' field with value 99 — collides with _ScoreATransform."""

    name = "score_b"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        data["score"] = 99
        data["field_b"] = "from_b"
        return TransformResult.success(
            make_pipeline_row(data),
            success_reason={"action": "score_b"},
        )


def _load_coalesce_context_after(db: LandscapeDB, run_id: str, coalesce_name: str) -> list[dict[str, Any]]:
    """Return deserialized context_after_json blobs for the coalesce node's states.

    The coalesce executor writes context_after on each consumed branch's pending
    node_state (not a dedicated coalesce node), so we look up node_states whose
    node belongs to the coalesce with the given name. The DAG builder assigns
    `plugin_name='coalesce:<name>'` to coalesce nodes.
    """
    with db.connection() as conn:
        rows = conn.execute(
            text(
                """
                SELECT ns.context_after_json, ns.status
                FROM node_states AS ns
                JOIN nodes AS n
                  ON n.node_id = ns.node_id
                 AND n.run_id = ns.run_id
                WHERE ns.run_id = :run_id
                  AND n.node_type = 'coalesce'
                  AND n.plugin_name = :plugin
                """
            ),
            {"run_id": run_id, "plugin": f"coalesce:{coalesce_name}"},
        ).fetchall()
    return [{"status": row[1], "context": json.loads(row[0]) if row[0] is not None else None} for row in rows]


def _find_merged_token_id(db: LandscapeDB, run_id: str) -> str:
    """Return the token_id of the single merged token produced by a coalesce in this run.

    A merged token is the join-group output — it has `join_group_id` set and
    no `branch_name` (branch_name is set on the consumed parent tokens, not the
    merged output). For tests with a single source row and one coalesce point,
    exactly one such token must exist.
    """
    with db.connection() as conn:
        rows = conn.execute(
            text(
                """
                SELECT token_id
                FROM tokens
                WHERE run_id = :run_id
                  AND join_group_id IS NOT NULL
                  AND branch_name IS NULL
                """
            ),
            {"run_id": run_id},
        ).fetchall()
    if len(rows) != 1:
        raise AssertionError(f"expected exactly 1 merged token, found {len(rows)}: {rows}")
    return str(rows[0][0])


class TestUnionMergeFieldProvenance:
    """Verify union_field_origins reaches the audit trail AND the explain() API.

    Covers both concerns raised by CLAUDE.md's attributability standard:

    * **Audit trail surface**: the underlying ``node_states.context_after_json``
      column captures ``union_field_origins`` and ``union_field_collision_values``.
      Verified via direct SQL — finer-grained than the API contract.
    * **explain() API surface**: ``lineage.explain(run_id, token_id)`` projects
      those fields onto the ``LineageResult.node_states`` tuple so an auditor
      running ``explain(recorder, run_id, token_id)`` can reach field provenance
      without writing raw SQL. This is the attributability-test guard.

    The fail-path test covers only the audit-trail surface because no merged
    token exists when ``union_collision_policy=fail`` trips.
    """

    def test_union_merge_surfaces_field_provenance_via_explain(self, payload_store) -> None:
        """Happy path: union merge populates provenance in audit trail AND explain().

        Pipeline: source -> fork(path_a, path_b) -> [ScoreA | ScoreB] -> union merge -> sink.
        Both branches add a 'score' field with different values, colliding on
        merge. With the default union_collision_policy=last_wins, the pipeline
        completes successfully and:

          * The underlying ``node_states.context_after_json`` column captures
            ``union_field_origins`` (every merged field -> its branch) and
            ``union_field_collision_values`` (both (branch, value) entries for
            the colliding 'score' field in declared order).
          * Calling ``explain(run_id, token_id=merged_token_id)`` and then
            drilling into each consumed parent token's ``LineageResult`` exposes
            the same provenance via ``NodeState.context_after_json`` — the API
            contract auditors rely on under CLAUDE.md's attributability standard.

        This exercises the production code path:
        ExecutionGraph.from_plugin_instances -> Orchestrator.run -> CoalesceExecutor
        -> CoalesceMetadata.with_union_result -> complete_node_state(context_after=...)
        -> node_states.context_after_json -> lineage.explain() -> LineageResult
        """
        db = make_landscape_db()
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_scores",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            # union_collision_policy defaults to "last_wins"
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}],
            branch_transforms={
                "path_a": [_ScoreATransform()],
                "path_b": [_ScoreBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        # Pipeline completed; sink received the merged row
        assert len(output_sink.results) == 1
        merged = output_sink.results[0]
        # last_wins: 'path_b' is second in branches ordering => score=99 wins
        assert merged["score"] == 99
        assert merged["field_a"] == "from_a"
        assert merged["field_b"] == "from_b"

        # Inspect the audit trail directly
        states = _load_coalesce_context_after(db, run_result.run_id, "merge_scores")
        assert states, "expected coalesce node_states to be recorded"

        # At least one branch state carries the union provenance context
        contexts_with_origins = [s["context"] for s in states if s["context"] is not None and "union_field_origins" in s["context"]]
        assert contexts_with_origins, f"no coalesce node_state recorded union_field_origins; got: {states}"

        ctx = contexts_with_origins[0]

        # field_origins maps every field to its originating branch
        origins = ctx["union_field_origins"]
        assert set(origins.keys()) >= {"value", "score", "field_a", "field_b"}
        assert origins["field_a"] == "path_a"
        assert origins["field_b"] == "path_b"
        # last_wins: 'score' collision resolves to path_b
        assert origins["score"] == "path_b"

        # collision_values records every contributing branch in declared order
        assert "union_field_collision_values" in ctx
        collision_values = ctx["union_field_collision_values"]
        assert "score" in collision_values
        # Each entry is a [branch, value] pair; declared order is path_a, path_b
        score_entries = collision_values["score"]
        assert len(score_entries) == 2
        assert score_entries[0][0] == "path_a" and score_entries[0][1] == 10
        assert score_entries[1][0] == "path_b" and score_entries[1][1] == 99

        # Base merge metadata is still present
        assert ctx["policy"] == "require_all"
        assert ctx["merge_strategy"] == "union"

        # ─────────────────────────────────────────────────────────────────
        # explain() API contract: CLAUDE.md attributability standard
        # ─────────────────────────────────────────────────────────────────
        # The audit trail (above) holds the data, but the attributability
        # standard is specifically about the explain(recorder, run_id, token_id)
        # API — auditors reach provenance through that entry point, not via
        # raw SQL. Verify the new fields are reachable via LineageResult.
        #
        # The merged token's node_states don't carry context_after directly
        # (the coalesce executor writes context_after on the CONSUMED parent
        # tokens' pending states, not on the merged token's own states).
        # An auditor navigates from a merged row back through parent_tokens
        # to reach the provenance. Verify both hops work:
        #   1. explain(merged_token_id) -> LineageResult with parent_tokens set
        #   2. explain(parent_token_id) -> LineageResult with a coalesce
        #      NodeState whose context_after_json holds union_field_origins
        factory = RecorderFactory(db, payload_store=payload_store)

        merged_token_id = _find_merged_token_id(db, run_result.run_id)
        merged_lineage = explain(factory.query, factory.data_flow, run_result.run_id, token_id=merged_token_id)
        assert merged_lineage is not None, "explain() returned None for merged token"
        # The merged token has two parent tokens (one per branch) recorded
        # via token_parents with join_group_id set — this is the lineage hop
        # auditors follow from a sink row back to the coalesce point.
        assert len(merged_lineage.parent_tokens) == 2, f"expected 2 parent tokens for merged token, got {len(merged_lineage.parent_tokens)}"
        parent_branch_names = {p.branch_name for p in merged_lineage.parent_tokens}
        assert parent_branch_names == {"path_a", "path_b"}

        # Hop into each parent: their coalesce node_state carries the
        # provenance context. Collect the first context_after_json we find.
        provenance_via_explain: dict[str, Any] | None = None
        for parent in merged_lineage.parent_tokens:
            parent_lineage = explain(factory.query, factory.data_flow, run_result.run_id, token_id=parent.token_id)
            assert parent_lineage is not None
            for ns in parent_lineage.node_states:
                # Coalesce branch states are COMPLETED (happy path) and carry
                # the CoalesceMetadata serialization in context_after_json.
                if not isinstance(ns, NodeStateCompleted):
                    continue
                if ns.context_after_json is None:
                    continue
                candidate = json.loads(ns.context_after_json)
                if "union_field_origins" in candidate:
                    provenance_via_explain = candidate
                    break
            if provenance_via_explain is not None:
                break

        assert provenance_via_explain is not None, (
            "explain() did not surface union_field_origins on any parent token's "
            "node_states. This breaks CLAUDE.md's attributability standard — "
            "auditors cannot reach field provenance through the explain() API."
        )

        # The explain()-surfaced provenance must match the raw audit trail
        # (same serialization, same values). Spot-check the critical fields.
        assert provenance_via_explain["union_field_origins"]["field_a"] == "path_a"
        assert provenance_via_explain["union_field_origins"]["field_b"] == "path_b"
        assert provenance_via_explain["union_field_origins"]["score"] == "path_b"
        explain_collisions = provenance_via_explain["union_field_collision_values"]
        assert "score" in explain_collisions
        assert len(explain_collisions["score"]) == 2
        assert {entry[0] for entry in explain_collisions["score"]} == {"path_a", "path_b"}

    def test_audit_trail_captures_collision_metadata_on_fail_policy(self, payload_store) -> None:
        """Failure path: union_collision_policy=fail raises but persists metadata.

        When the pipeline sets union_collision_policy='fail', a field collision
        raises CoalesceCollisionError. The coalesce executor's cleanup handler
        must still propagate metadata_for_audit to complete_node_state so the
        collision record reaches node_states.context_after_json with FAILED status.

        This is the regression guard for the Task 3 fix: metadata_for_audit
        is hoisted before _merge_data is called so the except-block cleanup
        handler can pass it as context_after even when the raise happens
        after metadata was built.
        """
        db = make_landscape_db()
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_scores_strict",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            union_collision_policy="fail",
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}],
            branch_transforms={
                "path_a": [_ScoreATransform()],
                "path_b": [_ScoreBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)

        # The collision must surface — either raised directly or wrapped in the
        # engine's row-processing error envelope. Either way, no rows reach the sink.
        with pytest.raises(Exception) as excinfo:
            orchestrator.run(
                config,
                graph=graph,
                settings=settings,
                payload_store=payload_store,
            )

        # Ensure the underlying cause is CoalesceCollisionError
        exc_chain: list[BaseException] = []
        cur: BaseException | None = excinfo.value
        seen_exceptions: set[int] = set()
        while cur is not None and id(cur) not in seen_exceptions:
            seen_exceptions.add(id(cur))
            exc_chain.append(cur)
            cur = cur.__cause__ or cur.__context__
        assert any(isinstance(e, CoalesceCollisionError) for e in exc_chain), (
            f"expected CoalesceCollisionError in exception chain, got: {[type(e).__name__ for e in exc_chain]}"
        )
        assert not output_sink.results, "sink must not receive rows when fail policy triggers"

        # The audit trail must still capture the collision record via the
        # cleanup handler (Task 3 regression guard). Find the FAILED node_state
        # and assert its context_after_json has the full collision metadata.
        # Note: we use the orchestrator-assigned run_id, which we recover from
        # the database since orchestrator.run raised.
        with db.connection() as conn:
            run_row = conn.execute(text("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")).fetchone()
        assert run_row is not None
        run_id = run_row[0]

        states = _load_coalesce_context_after(db, run_id, "merge_scores_strict")
        failed_with_ctx = [s["context"] for s in states if s["status"] == "failed" and s["context"] is not None]
        assert failed_with_ctx, f"expected FAILED coalesce node_state with context_after_json; got: {states}"

        ctx = failed_with_ctx[0]
        assert "union_field_origins" in ctx, (
            f"FAILED state missing union_field_origins — cleanup handler did not propagate metadata_for_audit. Got: {ctx}"
        )
        assert ctx["union_field_origins"]["score"] == "path_b"
        assert "union_field_collision_values" in ctx
        score_entries = ctx["union_field_collision_values"]["score"]
        assert len(score_entries) == 2
        assert {entry[0] for entry in score_entries} == {"path_a", "path_b"}

    def test_first_wins_collision_policy_surfaces_provenance_via_explain(self, payload_store) -> None:
        """first_wins policy: first branch value wins, audit trail captures collision.

        QA review identified that first_wins lacked E2E integration coverage through
        the Orchestrator -> audit trail path. This test verifies:

          * The first branch's collision value wins (path_a's score=10)
          * union_field_origins records path_a as the winner for 'score'
          * union_field_collision_values still captures BOTH branches' values

        This is the symmetric counterpart to test_union_merge_surfaces_field_provenance_via_explain
        which covers last_wins (the default).
        """
        db = make_landscape_db()
        gate = GateSettings(
            name="fork_gate",
            input="gate_in",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["path_a", "path_b"],
        )
        coalesce = CoalesceSettings(
            name="merge_scores_first",
            branches={"path_a": "done_a", "path_b": "done_b"},
            policy="require_all",
            merge="union",
            union_collision_policy="first_wins",
            on_success="output",
        )
        output_sink = CollectSink("output")

        config, graph, settings = _build_branch_pipeline(
            source_data=[{"value": 1}],
            branch_transforms={
                "path_a": [_ScoreATransform()],
                "path_b": [_ScoreBTransform()],
            },
            coalesce=coalesce,
            gate=gate,
            sinks={"output": output_sink},
        )

        orchestrator = Orchestrator(db)
        run_result = orchestrator.run(
            config,
            graph=graph,
            settings=settings,
            payload_store=payload_store,
        )

        # Pipeline completed; sink received the merged row
        assert len(output_sink.results) == 1
        merged = output_sink.results[0]
        # first_wins: 'path_a' is first in branches ordering => score=10 wins
        assert merged["score"] == 10, f"first_wins should select path_a's score=10, got {merged['score']}"
        assert merged["field_a"] == "from_a"
        assert merged["field_b"] == "from_b"

        # Inspect the audit trail directly
        states = _load_coalesce_context_after(db, run_result.run_id, "merge_scores_first")
        assert states, "expected coalesce node_states to be recorded"

        contexts_with_origins = [s["context"] for s in states if s["context"] is not None and "union_field_origins" in s["context"]]
        assert contexts_with_origins, f"no coalesce node_state recorded union_field_origins; got: {states}"

        ctx = contexts_with_origins[0]

        # field_origins: first_wins means path_a wins for 'score'
        origins = ctx["union_field_origins"]
        assert origins["score"] == "path_a", f"first_wins should record path_a as winner, got {origins['score']}"
        assert origins["field_a"] == "path_a"
        assert origins["field_b"] == "path_b"

        # collision_values still records BOTH branches in declared order
        assert "union_field_collision_values" in ctx
        collision_values = ctx["union_field_collision_values"]
        assert "score" in collision_values
        score_entries = collision_values["score"]
        assert len(score_entries) == 2
        assert score_entries[0][0] == "path_a" and score_entries[0][1] == 10
        assert score_entries[1][0] == "path_b" and score_entries[1][1] == 99

        # Base merge metadata
        assert ctx["policy"] == "require_all"
        assert ctx["merge_strategy"] == "union"
