# tests/unit/engine/test_tokens.py
"""Tests for TokenManager."""

from typing import Any

import pytest

from elspeth.contracts import NodeType, SourceRow
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.testing import make_field
from tests.unit.engine.conftest import DYNAMIC_SCHEMA
from tests.unit.engine.conftest import make_test_step_resolver as _make_step_resolver


def _make_observed_contract(*field_names: str) -> SchemaContract:
    """Create an OBSERVED mode contract with specified fields."""
    fields = tuple(
        make_field(name=name, original_name=f"'{name}'", python_type=object, required=False, source="inferred") for name in field_names
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def _make_source_row(data: dict[str, Any]) -> SourceRow:
    """Create a SourceRow with an OBSERVED contract containing all data fields."""
    contract = _make_observed_contract(*data.keys())
    return SourceRow.valid(data, contract=contract)


def _make_pipeline_row(data: dict[str, Any], contract: SchemaContract | None = None) -> PipelineRow:
    """Create a PipelineRow with an OBSERVED contract if not provided."""
    if contract is None:
        contract = _make_observed_contract(*data.keys())
    return PipelineRow(data, contract)


def _make_manager_context() -> tuple[Any, Any, str, str]:
    """Create TokenManager + recorder context for unit tests."""
    from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
    from elspeth.engine.tokens import TokenManager

    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run = recorder.begin_run(config={}, canonical_version="v1")
    source = recorder.register_node(
        run_id=run.run_id,
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    manager = TokenManager(recorder, step_resolver=_make_step_resolver())
    return manager, recorder, run.run_id, source.node_id


class TestTokenManager:
    """High-level token management."""

    def test_create_initial_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None
        assert isinstance(token_info.row_data, PipelineRow)
        assert token_info.row_data.to_dict() == {"value": 42}

    def test_fork_token(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        assert len(children) == 2
        assert children[0].branch_name == "stats"
        assert children[1].branch_name == "classifier"
        assert isinstance(children[0].row_data, PipelineRow)
        assert children[0].row_data.to_dict() == {"value": 42}

    def test_update_row_data(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        token_info = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
        )

        new_row = _make_pipeline_row({"x": 1, "y": 2})
        updated = manager.update_row_data(token_info, new_data=new_row)

        assert isinstance(updated.row_data, PipelineRow)
        assert updated.row_data.to_dict() == {"x": 1, "y": 2}
        assert updated.token_id == token_info.token_id


class TestTokenManagerCoalesce:
    """Test token coalescing (join operations)."""

    def test_coalesce_tokens(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        stats_token = manager.update_row_data(
            children[0],
            new_data=_make_pipeline_row({"value": 42, "mean": 10.5}),
        )
        classifier_token = manager.update_row_data(
            children[1],
            new_data=_make_pipeline_row({"value": 42, "label": "A"}),
        )

        merged = manager.coalesce_tokens(
            parents=[stats_token, classifier_token],
            merged_data=_make_pipeline_row({"value": 42, "mean": 10.5, "label": "A"}),
            node_id=NodeID("coalesce_node"),
        )

        assert merged.token_id is not None
        assert merged.row_id == initial.row_id
        assert isinstance(merged.row_data, PipelineRow)
        assert merged.row_data.to_dict() == {"value": 42, "mean": 10.5, "label": "A"}


class TestTokenManagerForkIsolation:
    """Test that forked tokens have isolated row_data (no shared mutable objects)."""

    def test_fork_nested_data_isolation(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"payload": {"x": 1, "y": 2}, "items": [1, 2, 3]}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["branch_a", "branch_b"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        child_a = children[0]
        child_b = children[1]

        dict_a = child_a.row_data.to_dict()
        dict_a["payload"]["x"] = 999
        dict_a["items"].append(4)

        dict_b_fresh = child_b.row_data.to_dict()
        assert dict_b_fresh["payload"]["x"] == 1, "Nested dict mutation leaked to sibling!"
        assert dict_b_fresh["items"] == [1, 2, 3], "Nested list mutation leaked to sibling!"

    def test_fork_with_custom_nested_data_isolation(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
        )

        custom_data = _make_pipeline_row({"nested": {"key": "original"}})
        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
            row_data=custom_data,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["nested"]["key"] = "modified"

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["nested"]["key"] == "original"


class TestTokenManagerExpandIsolation:
    """Test that expanded tokens have isolated row_data (no shared mutable objects)."""

    def test_expand_nested_data_isolation(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
        )

        expanded_rows = [
            {"payload": {"x": 1, "y": 2}, "items": [1, 2, 3]},
            {"payload": {"x": 10, "y": 20}, "items": [10, 20, 30]},
        ]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=_make_observed_contract(*expanded_rows[0].keys()),
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        child_a = children[0]
        child_b = children[1]

        dict_a = child_a.row_data.to_dict()
        dict_a["payload"]["x"] = 999
        dict_a["items"].append(4)

        dict_b = child_b.row_data.to_dict()
        assert dict_b["payload"]["x"] == 10, "Nested dict mutation leaked to sibling!"
        assert dict_b["items"] == [10, 20, 30], "Nested list mutation leaked to sibling!"

    def test_expand_shared_input_isolation(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
        )

        shared_metadata = {"version": 1, "tags": ["a", "b"]}
        expanded_rows = [
            {"id": 1, "meta": shared_metadata},
            {"id": 2, "meta": shared_metadata},
        ]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=_make_observed_contract(*expanded_rows[0].keys()),
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["meta"]["version"] = 999
        dict_0["meta"]["tags"].append("mutated")

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["meta"]["version"] == 1, "Shared object mutation leaked!"
        assert dict_1["meta"]["tags"] == ["a", "b"], "Shared list mutation leaked!"

    def test_expand_deep_nesting_isolation(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
        )

        expanded_rows = [
            {"level1": {"level2": [{"level3": ["deep_value"]}]}},
            {"level1": {"level2": [{"level3": ["deep_value"]}]}},
        ]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=_make_observed_contract(*expanded_rows[0].keys()),
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["level1"]["level2"][0]["level3"][0] = "MUTATED"

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["level1"]["level2"][0]["level3"][0] == "deep_value"


class TestTokenManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_fork_with_custom_row_data(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["branch_a"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
            row_data=_make_pipeline_row({"value": 42, "forked": True}),
        )

        assert children[0].row_data.to_dict() == {"value": 42, "forked": True}

    def test_update_preserves_branch_name(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["my_branch"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        updated = manager.update_row_data(
            children[0],
            new_data=_make_pipeline_row({"x": 1, "y": 2}),
        )

        assert updated.branch_name == "my_branch"

    def test_update_preserves_all_lineage_fields(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
        )

        forked_children, fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats_branch"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )
        forked_token = forked_children[0]

        assert forked_token.fork_group_id == fork_group_id
        assert forked_token.branch_name == "stats_branch"

        updated = manager.update_row_data(
            forked_token,
            new_data=_make_pipeline_row({"x": 1, "y": 2}),
        )

        assert updated.row_data.to_dict() == {"x": 1, "y": 2}, "row_data should be updated"
        assert updated.token_id == forked_token.token_id, "token_id must be preserved"
        assert updated.row_id == forked_token.row_id, "row_id must be preserved"
        assert updated.branch_name == "stats_branch", "branch_name must be preserved"
        assert updated.fork_group_id == fork_group_id, "fork_group_id must be preserved"

    def test_update_preserves_expand_group_id(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
        )

        expanded_children, expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"id": 1}, {"id": 2}],
            output_contract=_make_observed_contract("id"),
            node_id=NodeID("expand_node"),
            run_id=run.run_id,
        )
        expanded_token = expanded_children[0]

        assert expanded_token.expand_group_id == expand_group_id

        updated = manager.update_row_data(
            expanded_token,
            new_data=_make_pipeline_row({"id": 1, "processed": True}),
        )

        assert updated.expand_group_id == expand_group_id, "expand_group_id must be preserved"

    def test_update_preserves_join_group_id(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        merged = manager.coalesce_tokens(
            parents=children,
            merged_data=_make_pipeline_row({"value": 42, "merged": True}),
            node_id=NodeID("coalesce_node"),
        )

        assert merged.join_group_id is not None

        updated = manager.update_row_data(
            merged,
            new_data=_make_pipeline_row({"value": 42, "merged": True, "enriched": "yes"}),
        )

        assert updated.join_group_id == merged.join_group_id, "join_group_id must be preserved"

    def test_multiple_rows_different_tokens(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        token1 = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"id": 1}),
        )
        token2 = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=1,
            source_row=_make_source_row({"id": 2}),
        )

        assert token1.row_id != token2.row_id
        assert token1.token_id != token2.token_id


class TestTokenManagerStepInPipeline:
    """Test that step_in_pipeline flows through to audit trail."""

    def test_fork_stores_step_in_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver({"fork_gate": 2}))
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("fork_gate"),
            run_id=run.run_id,
        )

        token_a = recorder.get_token(children[0].token_id)
        token_b = recorder.get_token(children[1].token_id)
        assert token_a is not None
        assert token_b is not None
        assert token_a.step_in_pipeline == 2
        assert token_b.step_in_pipeline == 2

    def test_coalesce_stores_step_in_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver({"gate_node": 1, "coalesce_node": 3}))
        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )

        merged = manager.coalesce_tokens(
            parents=children,
            merged_data=_make_pipeline_row({"value": 42, "merged": True}),
            node_id=NodeID("coalesce_node"),
        )

        merged_token = recorder.get_token(merged.token_id)
        assert merged_token is not None
        assert merged_token.step_in_pipeline == 3


class TestTokenManagerExpand:
    """Test token expansion (deaggregation: 1 input -> N outputs)."""

    def test_expand_token_creates_children(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
        )

        expanded_rows = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=expanded_rows,
            output_contract=_make_observed_contract(*expanded_rows[0].keys()),
            node_id=NodeID("expand_node"),
            run_id=run.run_id,
        )

        assert len(children) == 3

        for child in children:
            assert child.row_id == parent.row_id
            assert child.token_id != parent.token_id

        assert isinstance(children[0].row_data, PipelineRow)
        assert children[0].row_data.to_dict() == {"id": 1, "value": "a"}
        assert children[1].row_data.to_dict() == {"id": 2, "value": "b"}
        assert children[2].row_data.to_dict() == {"id": 3, "value": "c"}

        for i, child in enumerate(children):
            parents = recorder.get_token_parents(child.token_id)
            assert len(parents) == 1
            assert parents[0].parent_token_id == parent.token_id
            assert parents[0].ordinal == i

    def test_expand_token_inherits_branch_name(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({}),
        )

        forked, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats_branch"],
            node_id=NodeID("gate_node"),
            run_id=run.run_id,
        )
        parent = forked[0]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run.run_id,
        )

        assert all(c.branch_name == "stats_branch" for c in children)

    def test_expand_token_stores_step_in_pipeline(self) -> None:
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.tokens import TokenManager

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        source = recorder.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        manager = TokenManager(recorder, step_resolver=_make_step_resolver({"expand_node": 5}))
        parent = manager.create_initial_token(
            run_id=run.run_id,
            source_node_id=source.node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
        )

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run.run_id,
        )

        for child in children:
            db_token = recorder.get_token(child.token_id)
            assert db_token is not None
            assert db_token.step_in_pipeline == 5


class TestTokenManagerBoundaryPaths:
    """Coverage for error guards and quarantine/resume token paths."""

    def test_create_initial_token_requires_contract(self) -> None:
        manager, _recorder, run_id, source_node_id = _make_manager_context()

        with pytest.raises(ValueError, match="must have contract"):
            manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=0,
                source_row=SourceRow.valid({"value": 42}, contract=None),
            )

    def test_create_quarantine_token_rejects_non_quarantined_source_row(self) -> None:
        manager, _recorder, run_id, source_node_id = _make_manager_context()

        with pytest.raises(ValueError, match="requires a quarantined"):
            manager.create_quarantine_token(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=0,
                source_row=_make_source_row({"value": 42}),
            )

    def test_create_quarantine_token_preserves_dict_payload(self) -> None:
        manager, _recorder, run_id, source_node_id = _make_manager_context()

        token = manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=SourceRow.quarantined(
                row={"raw": "invalid"},
                error="bad data",
                destination="quarantine",
            ),
        )

        assert token.row_data.to_dict() == {"raw": "invalid"}
        assert token.row_data.contract.mode == "OBSERVED"
        assert token.row_data.contract.fields == ()
        assert token.row_data.contract.locked is False

    def test_create_quarantine_token_wraps_non_dict_payload(self) -> None:
        manager, _recorder, run_id, source_node_id = _make_manager_context()

        token = manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=SourceRow.quarantined(
                row=["not", "a", "dict"],
                error="bad row type",
                destination="quarantine",
            ),
        )

        assert token.row_data.to_dict() == {"_raw": ["not", "a", "dict"]}
        assert token.row_data.contract.mode == "OBSERVED"

    def test_create_token_for_existing_row_creates_new_token(self) -> None:
        manager, recorder, run_id, source_node_id = _make_manager_context()

        original = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"id": 1}),
        )
        restored_row = _make_pipeline_row({"id": 1, "restored": True})

        resumed = manager.create_token_for_existing_row(
            row_id=original.row_id,
            row_data=restored_row,
        )

        assert resumed.row_id == original.row_id
        assert resumed.token_id != original.token_id
        assert resumed.row_data is restored_row
        assert recorder.get_token(resumed.token_id) is not None

    def test_expand_token_requires_locked_output_contract(self) -> None:
        manager, recorder, run_id, source_node_id = _make_manager_context()

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
        )
        unlocked_contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)

        tokens_before = recorder.get_all_tokens_for_run(run_id)
        parents_before = recorder.get_all_token_parents_for_run(run_id)
        outcome_before = recorder.get_token_outcome(parent.token_id)

        with pytest.raises(ValueError, match="must be locked"):
            manager.expand_token(
                parent_token=parent,
                expanded_rows=[{"value": 1}],
                output_contract=unlocked_contract,
                node_id=NodeID("expand_node"),
                run_id=run_id,
            )

        tokens_after = recorder.get_all_tokens_for_run(run_id)
        parents_after = recorder.get_all_token_parents_for_run(run_id)
        outcome_after = recorder.get_token_outcome(parent.token_id)

        assert len(tokens_after) == len(tokens_before), "Unlocked contract must not create child tokens"
        assert len(parents_after) == len(parents_before), "Unlocked contract must not create token parent links"
        assert outcome_before is outcome_after is None, "Unlocked contract must not record parent EXPANDED outcome"
