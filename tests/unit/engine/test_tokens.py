# tests/unit/engine/test_tokens.py
"""Tests for TokenManager."""

from typing import Any

import pytest

from elspeth.contracts import SourceRow
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.testing import make_field
from tests.fixtures.landscape import make_recorder_with_run
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
    """Create TokenManager + factory context for unit tests."""
    from elspeth.engine.tokens import TokenManager

    setup = make_recorder_with_run()
    manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
    return manager, setup.factory, setup.run_id, setup.source_node_id


class TestTokenManager:
    """High-level token management."""

    def test_create_initial_token(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver())

        token_info = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None
        assert isinstance(token_info.row_data, PipelineRow)
        assert token_info.row_data.to_dict() == {"value": 42}

    def test_fork_token(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )

        assert len(children) == 2
        assert children[0].branch_name == "stats"
        assert children[1].branch_name == "classifier"
        assert isinstance(children[0].row_data, PipelineRow)
        assert children[0].row_data.to_dict() == {"value": 42}


class TestTokenManagerCoalesce:
    """Test token coalescing (join operations)."""

    def test_coalesce_tokens(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats", "classifier"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )

        stats_token = children[0].with_updated_data(
            _make_pipeline_row({"value": 42, "mean": 10.5}),
        )
        classifier_token = children[1].with_updated_data(
            _make_pipeline_row({"value": 42, "label": "A"}),
        )

        merged = manager.coalesce_tokens(
            parents=[stats_token, classifier_token],
            merged_data=_make_pipeline_row({"value": 42, "mean": 10.5, "label": "A"}),
            node_id=NodeID("coalesce_node"),
            run_id=run_id,
        )

        assert merged.token_id is not None
        assert merged.row_id == initial.row_id
        assert isinstance(merged.row_data, PipelineRow)
        assert merged.row_data.to_dict() == {"value": 42, "mean": 10.5, "label": "A"}


class TestTokenManagerCoalesceValidation:
    """Regression tests: coalesce_tokens validates invariants."""

    def test_coalesce_tokens_empty_parents_raises(self) -> None:
        """Empty parents list raises OrchestrationInvariantError."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        manager, _factory, _run_id, _source_node_id = _make_manager_context()

        with pytest.raises(OrchestrationInvariantError, match="at least one parent"):
            manager.coalesce_tokens(
                parents=[],
                merged_data=_make_pipeline_row({"value": 1}),
                node_id=NodeID("coalesce_node"),
                run_id=_run_id,
            )

    def test_coalesce_tokens_mismatched_row_ids_raises(self) -> None:
        """Parents with different row_ids raises OrchestrationInvariantError."""
        from elspeth.contracts.errors import OrchestrationInvariantError

        manager, _factory, run_id, source_node_id = _make_manager_context()

        # Create two separate source rows (different row_ids)
        token_a = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        token_b = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            source_row_index=1,
            ingest_sequence=1,
        )

        assert token_a.row_id != token_b.row_id

        with pytest.raises(OrchestrationInvariantError, match="mismatched token_ids"):
            manager.coalesce_tokens(
                parents=[token_a, token_b],
                merged_data=_make_pipeline_row({"value": 1}),
                node_id=NodeID("coalesce_node"),
                run_id=run_id,
            )


class TestCoalesceMismatchedRowIdsMutationKill:
    """Kill mutants on coalesce_tokens row_id validation (lines 285-286).

    Mutant 1: ``parents[0].row_id`` → ``parents[1].row_id``
        Survives when all test parents share the same row_id OR when the
        test doesn't verify which row_id appears in the error message.

    Mutant 2: ``p.row_id != row_id`` → ``p.row_id > row_id``
        Survives when the mismatched parent's row_id is alphabetically
        greater than the reference. We need at least one parent whose
        row_id is alphabetically LESS than the reference.
    """

    def test_error_message_contains_first_parent_row_id(self) -> None:
        """Verify the reference row_id in the error is parents[0].row_id.

        Kill mutant: ``parents[0].row_id`` → ``parents[1].row_id``.
        If the mutant is active, the error message will contain parent B's
        row_id instead of parent A's, and this assertion fails.
        """
        manager, _factory, run_id, source_node_id = _make_manager_context()

        token_a = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        token_b = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            source_row_index=1,
            ingest_sequence=1,
        )

        with pytest.raises(OrchestrationInvariantError, match=token_a.row_id):
            manager.coalesce_tokens(
                parents=[token_a, token_b],
                merged_data=_make_pipeline_row({"value": 1}),
                node_id=NodeID("coalesce_node"),
                run_id=run_id,
            )

    def test_mismatched_row_id_less_than_reference_detected(self) -> None:
        """Kill mutant: ``p.row_id != row_id`` → ``p.row_id > row_id``.

        The ``>`` mutant misses parents whose row_id is alphabetically less
        than the reference. We order parents so the second has a lesser row_id.
        """
        manager, _factory, run_id, source_node_id = _make_manager_context()

        token_a = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        token_b = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=1,
            source_row=_make_source_row({"value": 2}),
            source_row_index=1,
            ingest_sequence=1,
        )

        # Order so that parents[1].row_id < parents[0].row_id
        # If row_ids are random UUIDs, sort to guarantee ordering
        if token_a.row_id < token_b.row_id:
            first, second = token_b, token_a
        else:
            first, second = token_a, token_b

        # Now second.row_id < first.row_id
        # With the > mutant: p.row_id > row_id would be False for second,
        # so mismatched would be empty and the error would NOT raise.
        with pytest.raises(OrchestrationInvariantError, match="mismatched token_ids"):
            manager.coalesce_tokens(
                parents=[first, second],
                merged_data=_make_pipeline_row({"value": 1}),
                node_id=NodeID("coalesce_node"),
                run_id=run_id,
            )


class TestTokenManagerForkIsolation:
    """Test that forked tokens have isolated row_data (no shared mutable objects)."""

    def test_fork_nested_data_isolation(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"payload": {"x": 1, "y": 2}, "items": [1, 2, 3]}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["branch_a", "branch_b"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
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
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        custom_data = _make_pipeline_row({"nested": {"key": "original"}})
        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
            row_data=custom_data,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["nested"]["key"] = "modified"

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["nested"]["key"] == "original"


class TestTokenManagerExpandIsolation:
    """Test that expanded tokens have isolated row_data (no shared mutable objects)."""

    def test_expand_nested_data_isolation(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
            source_row_index=0,
            ingest_sequence=0,
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
            run_id=run_id,
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
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
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
            run_id=run_id,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["meta"]["version"] = 999
        dict_0["meta"]["tags"].append("mutated")

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["meta"]["version"] == 1, "Shared object mutation leaked!"
        assert dict_1["meta"]["tags"] == ["a", "b"], "Shared list mutation leaked!"

    def test_expand_deep_nesting_isolation(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 1}),
            source_row_index=0,
            ingest_sequence=0,
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
            run_id=run_id,
        )

        dict_0 = children[0].row_data.to_dict()
        dict_0["level1"]["level2"][0]["level3"][0] = "MUTATED"

        dict_1 = children[1].row_data.to_dict()
        assert dict_1["level1"]["level2"][0]["level3"][0] == "deep_value"


class TestTokenManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_fork_with_custom_row_data(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["branch_a"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
            row_data=_make_pipeline_row({"value": 42, "forked": True}),
        )

        assert children[0].row_data.to_dict() == {"value": 42, "forked": True}

    def test_update_preserves_branch_name(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["my_branch"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )

        updated = children[0].with_updated_data(
            _make_pipeline_row({"x": 1, "y": 2}),
        )

        assert updated.branch_name == "my_branch"

    def test_update_preserves_all_lineage_fields(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        forked_children, fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats_branch"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )
        forked_token = forked_children[0]

        assert forked_token.fork_group_id == fork_group_id
        assert forked_token.branch_name == "stats_branch"

        updated = forked_token.with_updated_data(
            _make_pipeline_row({"x": 1, "y": 2}),
        )

        assert updated.row_data.to_dict() == {"x": 1, "y": 2}, "row_data should be updated"
        assert updated.token_id == forked_token.token_id, "token_id must be preserved"
        assert updated.row_id == forked_token.row_id, "row_id must be preserved"
        assert updated.branch_name == "stats_branch", "branch_name must be preserved"
        assert updated.fork_group_id == fork_group_id, "fork_group_id must be preserved"

    def test_update_preserves_expand_group_id(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
            source_row_index=0,
            ingest_sequence=0,
        )

        expanded_children, expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"id": 1}, {"id": 2}],
            output_contract=_make_observed_contract("id"),
            node_id=NodeID("expand_node"),
            run_id=run_id,
        )
        expanded_token = expanded_children[0]

        assert expanded_token.expand_group_id == expand_group_id

        updated = expanded_token.with_updated_data(
            _make_pipeline_row({"id": 1, "processed": True}),
        )

        assert updated.expand_group_id == expand_group_id, "expand_group_id must be preserved"

    def test_update_preserves_join_group_id(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )

        merged = manager.coalesce_tokens(
            parents=children,
            merged_data=_make_pipeline_row({"value": 42, "merged": True}),
            node_id=NodeID("coalesce_node"),
            run_id=run_id,
        )

        assert merged.join_group_id is not None

        updated = merged.with_updated_data(
            _make_pipeline_row({"value": 42, "merged": True, "enriched": "yes"}),
        )

        assert updated.join_group_id == merged.join_group_id, "join_group_id must be preserved"

    def test_multiple_rows_different_tokens(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())

        token1 = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"id": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        token2 = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=1,
            source_row=_make_source_row({"id": 2}),
            source_row_index=1,
            ingest_sequence=1,
        )

        assert token1.row_id != token2.row_id
        assert token1.token_id != token2.token_id


class TestTokenManagerStepInPipeline:
    """Test that step_in_pipeline flows through to audit trail."""

    def test_fork_stores_step_in_pipeline(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver({"fork_gate": 2}))
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("fork_gate"),
            run_id=run_id,
        )

        token_a = factory.query.get_token(children[0].token_id)
        token_b = factory.query.get_token(children[1].token_id)
        assert token_a is not None
        assert token_b is not None
        assert token_a.step_in_pipeline == 2
        assert token_b.step_in_pipeline == 2

    def test_coalesce_stores_step_in_pipeline(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver({"gate_node": 1, "coalesce_node": 3}))
        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"value": 42}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["a", "b"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )

        merged = manager.coalesce_tokens(
            parents=children,
            merged_data=_make_pipeline_row({"value": 42, "merged": True}),
            node_id=NodeID("coalesce_node"),
            run_id=run_id,
        )

        merged_token = factory.query.get_token(merged.token_id)
        assert merged_token is not None
        assert merged_token.step_in_pipeline == 3


class TestTokenManagerExpand:
    """Test token expansion (deaggregation: 1 input -> N outputs)."""

    def test_expand_token_creates_children(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver())

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"original": "data"}),
            source_row_index=0,
            ingest_sequence=0,
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
            run_id=run_id,
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
            parents = factory.query.get_token_parents(child.token_id)
            assert len(parents) == 1
            assert parents[0].parent_token_id == parent.token_id
            assert parents[0].ordinal == i

    def test_expand_token_inherits_branch_name(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        run_id, source_node_id = setup.run_id, setup.source_node_id

        manager = TokenManager(setup.data_flow, step_resolver=_make_step_resolver())

        initial = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({}),
            source_row_index=0,
            ingest_sequence=0,
        )

        forked, _fork_group_id = manager.fork_token(
            parent_token=initial,
            branches=["stats_branch"],
            node_id=NodeID("gate_node"),
            run_id=run_id,
        )
        parent = forked[0]

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run_id,
        )

        assert all(c.branch_name == "stats_branch" for c in children)

    def test_expand_token_stores_step_in_pipeline(self) -> None:
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        factory, run_id, source_node_id = setup.factory, setup.run_id, setup.source_node_id

        manager = TokenManager(factory.data_flow, step_resolver=_make_step_resolver({"expand_node": 5}))
        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run_id,
        )

        for child in children:
            db_token = factory.query.get_token(child.token_id)
            assert db_token is not None
            assert db_token.step_in_pipeline == 5


class TestTokenManagerBoundaryPaths:
    """Coverage for error guards and quarantine/resume token paths."""

    def test_create_initial_token_requires_contract(self) -> None:
        # Since elspeth-a27e71979f, SourceRow.__post_init__ rejects contract=None
        # at construction time, so the engine's guard is now unreachable via
        # normal construction. Verify the earlier guard fires instead.
        with pytest.raises(TypeError, match="contract"):
            SourceRow.valid({"value": 42})

    def test_create_initial_token_requires_explicit_source_identity(self) -> None:
        manager, _factory, run_id, source_node_id = _make_manager_context()

        with pytest.raises(
            OrchestrationInvariantError,
            match=r"TokenManager requires explicit source row identity.*source_row_index.*ingest_sequence",
        ):
            manager.create_initial_token(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=0,
                source_row=_make_source_row({"value": 42}),
            )

    def test_create_quarantine_token_requires_explicit_source_identity(self) -> None:
        manager, _factory, run_id, source_node_id = _make_manager_context()

        with pytest.raises(
            OrchestrationInvariantError,
            match=r"TokenManager requires explicit source row identity.*source_row_index.*ingest_sequence",
        ):
            manager.create_quarantine_token(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=0,
                source_row=SourceRow.quarantined(
                    row={"raw": "invalid"},
                    error="bad data",
                    destination="quarantine",
                ),
            )

    def test_create_quarantine_token_rejects_non_quarantined_source_row(self) -> None:
        manager, _factory, run_id, source_node_id = _make_manager_context()

        with pytest.raises(OrchestrationInvariantError, match="requires a quarantined"):
            manager.create_quarantine_token(
                run_id=run_id,
                source_node_id=source_node_id,
                row_index=0,
                source_row=_make_source_row({"value": 42}),
                source_row_index=0,
                ingest_sequence=0,
            )

    def test_create_quarantine_token_preserves_dict_payload(self) -> None:
        manager, _factory, run_id, source_node_id = _make_manager_context()

        token = manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=SourceRow.quarantined(
                row={"raw": "invalid"},
                error="bad data",
                destination="quarantine",
            ),
            source_row_index=0,
            ingest_sequence=0,
        )

        assert token.row_data.to_dict() == {"raw": "invalid"}
        assert token.row_data.contract.mode == "OBSERVED"
        assert token.row_data.contract.fields == ()
        assert token.row_data.contract.locked is False

    def test_create_quarantine_token_wraps_non_dict_payload(self) -> None:
        manager, _factory, run_id, source_node_id = _make_manager_context()

        token = manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=SourceRow.quarantined(
                row=["not", "a", "dict"],
                error="bad row type",
                destination="quarantine",
            ),
            source_row_index=0,
            ingest_sequence=0,
        )

        assert token.row_data.to_dict() == {"_raw": ["not", "a", "dict"]}
        assert token.row_data.contract.mode == "OBSERVED"

    def test_create_token_for_existing_row_creates_new_token(self) -> None:
        manager, factory, run_id, source_node_id = _make_manager_context()

        original = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"id": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        restored_row = _make_pipeline_row({"id": 1, "restored": True})

        resumed = manager.create_token_for_existing_row(
            row_id=original.row_id,
            row_data=restored_row,
        )

        assert resumed.row_id == original.row_id
        assert resumed.token_id != original.token_id
        assert resumed.row_data is restored_row
        assert factory.query.get_token(resumed.token_id) is not None

    def test_expand_token_requires_locked_output_contract(self) -> None:
        manager, factory, run_id, source_node_id = _make_manager_context()

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )
        unlocked_contract = SchemaContract(mode="OBSERVED", fields=(), locked=False)

        tokens_before = factory.query.get_all_tokens_for_run(run_id)
        parents_before = factory.query.get_all_token_parents_for_run(run_id)
        outcome_before = factory.data_flow.get_token_outcome(parent.token_id)

        with pytest.raises(OrchestrationInvariantError, match="must be locked"):
            manager.expand_token(
                parent_token=parent,
                expanded_rows=[{"value": 1}],
                output_contract=unlocked_contract,
                node_id=NodeID("expand_node"),
                run_id=run_id,
            )

        tokens_after = factory.query.get_all_tokens_for_run(run_id)
        parents_after = factory.query.get_all_token_parents_for_run(run_id)
        outcome_after = factory.data_flow.get_token_outcome(parent.token_id)

        assert len(tokens_after) == len(tokens_before), "Unlocked contract must not create child tokens"
        assert len(parents_after) == len(parents_before), "Unlocked contract must not create token parent links"
        assert outcome_before is outcome_after is None, "Unlocked contract must not record parent EXPANDED outcome"


class TestExpandTokenDefaultOutcome:
    """Kill mutant: ``record_parent_outcome=True`` default → ``False``.

    Line 330: expand_token(record_parent_outcome=True) is the default.
    Callers relying on the default must get EXPANDED outcome recorded
    for the parent. If the mutant flips the default to False, the
    parent outcome is silently skipped.
    """

    def test_expand_without_explicit_record_parent_outcome_records_expanded(self) -> None:
        """Call expand_token WITHOUT record_parent_outcome arg — parent gets EXPANDED."""
        manager, factory, run_id, source_node_id = _make_manager_context()

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        # Deliberately omit record_parent_outcome — rely on default=True
        _children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}, {"a": 2}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run_id,
        )

        outcome = factory.data_flow.get_token_outcome(parent.token_id)
        assert outcome is not None, (
            "Parent token must have an outcome when using expand_token default. "
            "If record_parent_outcome default mutant (True→False) is active, "
            "outcome will be None."
        )
        assert outcome.outcome == TerminalOutcome.TRANSIENT
        assert outcome.path == TerminalPath.EXPAND_PARENT

    def test_expand_with_explicit_false_skips_parent_outcome(self) -> None:
        """Call expand_token with record_parent_outcome=False — no EXPANDED outcome."""
        manager, factory, run_id, source_node_id = _make_manager_context()

        parent = manager.create_initial_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        _children, _expand_group_id = manager.expand_token(
            parent_token=parent,
            expanded_rows=[{"a": 1}],
            output_contract=_make_observed_contract("a"),
            node_id=NodeID("expand_node"),
            run_id=run_id,
            record_parent_outcome=False,
        )

        outcome = factory.data_flow.get_token_outcome(parent.token_id)
        assert outcome is None, "Parent token must NOT have an outcome when record_parent_outcome=False."


class TestExpandTokenStrictZip:
    """Kill mutant: ``strict=True`` → ``strict=False`` in zip on line 394.

    The strict zip ensures db_children (from data_flow) and expanded_rows
    have the same length. If strict=False, length mismatches are silently
    ignored — extra rows are dropped or extra children get no data.
    """

    def test_zip_strict_catches_length_mismatch(self) -> None:
        """DataFlowRepository returns N children but expanded_rows has M != N items.

        We mock the data_flow's expand_token to return a different count
        of children than the expanded_rows length.
        """
        from unittest.mock import patch

        from elspeth.contracts.audit import Token
        from elspeth.engine.tokens import TokenManager

        setup = make_recorder_with_run()
        data_flow = setup.data_flow
        manager = TokenManager(data_flow, step_resolver=_make_step_resolver())

        parent = manager.create_initial_token(
            run_id=setup.run_id,
            source_node_id=setup.source_node_id,
            row_index=0,
            source_row=_make_source_row({"x": 1}),
            source_row_index=0,
            ingest_sequence=0,
        )

        # Create fake Token objects the data_flow would return
        from datetime import UTC, datetime

        fake_children = [
            Token(
                token_id=f"fake-child-{i}",
                row_id=parent.row_id,
                run_id=setup.run_id,
                created_at=datetime.now(UTC),
                expand_group_id="eg-1",
            )
            for i in range(2)  # Recorder returns 2 children
        ]

        # Patch data_flow.expand_token to return 2 children
        with patch.object(data_flow, "expand_token", return_value=(fake_children, "eg-1")), pytest.raises(ValueError):
            manager.expand_token(
                parent_token=parent,
                expanded_rows=[{"a": 1}, {"a": 2}, {"a": 3}],
                output_contract=_make_observed_contract("a"),
                node_id=NodeID("expand_node"),
                run_id=setup.run_id,
            )


class TestCreateQuarantineTokenFlag:
    """Kill mutant: ``quarantined=True`` → ``quarantined=False``.

    Quarantined rows must be recorded with quarantined=True so the
    audit trail distinguishes quarantined data from normal rows.
    Without this flag, quarantined rows with NaN/Infinity would
    fail canonical hashing instead of being safely stored.
    """

    def test_quarantine_token_passes_quarantined_flag_to_data_flow(self) -> None:
        """create_quarantine_token must pass quarantined=True to data_flow.create_row.

        Kill mutant: quarantined=True → quarantined=False.

        When quarantined=True, the data_flow uses repr_hash fallback for
        data containing NaN/Infinity. If the mutant flips it to False,
        canonical hashing crashes on NaN data.
        """
        manager, factory, run_id, source_node_id = _make_manager_context()

        # Data with NaN — only works if quarantined=True (repr_hash fallback)
        quarantine_row = SourceRow.quarantined(
            {"bad_data": float("nan")},
            error="NaN value",
            destination="quarantine_sink",
        )

        # This must NOT raise — quarantined=True enables repr_hash fallback
        token_info = manager.create_quarantine_token(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            source_row=quarantine_row,
            source_row_index=0,
            ingest_sequence=0,
        )

        assert token_info.row_id is not None
        assert token_info.token_id is not None

        # Verify the row was actually stored (proof quarantined=True worked)
        row_record = factory.query.get_row(token_info.row_id)
        assert row_record is not None
        assert row_record.source_data_hash is not None, (
            "Quarantined row must have a hash (via repr_hash fallback). "
            "If quarantined=False mutant was active, create_row would have "
            "crashed on NaN during canonical hashing."
        )
