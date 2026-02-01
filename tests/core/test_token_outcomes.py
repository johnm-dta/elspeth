# tests/core/test_token_outcomes.py
"""Tests for token outcome recording."""

import pytest

from elspeth.core.landscape import LandscapeDB, LandscapeRecorder


@pytest.fixture(scope="module")
def landscape_db() -> LandscapeDB:
    """Create a module-scoped in-memory landscape database.

    Module scope avoids repeated schema creation (15+ tables, indexes)
    which takes ~5-10ms per instantiation.

    Tests should use unique run_ids to isolate their data.
    """
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(landscape_db: LandscapeDB) -> LandscapeRecorder:
    """Create a LandscapeRecorder with the shared test database.

    Function-scoped because the recorder is a lightweight wrapper.
    Uses the module-scoped landscape_db for actual storage.
    """
    return LandscapeRecorder(landscape_db)


class TestTokenOutcomeDataclass:
    """Test TokenOutcome dataclass structure."""

    def test_token_outcome_has_required_fields(self) -> None:
        from elspeth.contracts import TokenOutcome

        # Should have these fields
        assert hasattr(TokenOutcome, "__dataclass_fields__")
        fields = TokenOutcome.__dataclass_fields__
        assert "outcome_id" in fields
        assert "run_id" in fields
        assert "token_id" in fields
        assert "outcome" in fields
        assert "is_terminal" in fields
        assert "recorded_at" in fields

    def test_token_outcome_instantiation(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.COMPLETED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
        )
        assert outcome.outcome_id == "out_123"
        assert outcome.is_terminal is True

    def test_token_outcome_is_frozen(self) -> None:
        """TokenOutcome should be immutable (frozen dataclass)."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.COMPLETED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            outcome.outcome_id = "different"  # type: ignore[misc]

    def test_token_outcome_optional_fields(self) -> None:
        """TokenOutcome should have optional context fields."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        # All optional fields should default to None
        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.COMPLETED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
        )
        assert outcome.sink_name is None
        assert outcome.batch_id is None
        assert outcome.fork_group_id is None
        assert outcome.join_group_id is None
        assert outcome.expand_group_id is None
        assert outcome.error_hash is None
        assert outcome.context_json is None

    def test_token_outcome_with_sink_context(self) -> None:
        """TokenOutcome can record sink-specific context."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.ROUTED,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
            sink_name="error_sink",
        )
        assert outcome.sink_name == "error_sink"

    def test_token_outcome_with_batch_context(self) -> None:
        """TokenOutcome can record batch-specific context."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=RowOutcome.CONSUMED_IN_BATCH,
            is_terminal=True,
            recorded_at=datetime.now(UTC),
            batch_id="batch_abc",
        )
        assert outcome.batch_id == "batch_abc"

    def test_token_outcome_uses_row_outcome_enum(self) -> None:
        """TokenOutcome.outcome field should accept RowOutcome enum values."""
        from datetime import UTC, datetime

        from elspeth.contracts import RowOutcome, TokenOutcome

        # Test with different RowOutcome values
        for row_outcome in [
            RowOutcome.COMPLETED,
            RowOutcome.ROUTED,
            RowOutcome.FORKED,
            RowOutcome.FAILED,
            RowOutcome.QUARANTINED,
            RowOutcome.CONSUMED_IN_BATCH,
            RowOutcome.COALESCED,
            RowOutcome.EXPANDED,
            RowOutcome.BUFFERED,
        ]:
            outcome = TokenOutcome(
                outcome_id="out_123",
                run_id="run_456",
                token_id="tok_789",
                outcome=row_outcome,
                is_terminal=row_outcome.is_terminal,
                recorded_at=datetime.now(UTC),
            )
            assert outcome.outcome == row_outcome
            assert outcome.is_terminal == row_outcome.is_terminal


class TestTokenOutcomesTableSchema:
    """Test token_outcomes table definition."""

    def test_table_exists_in_metadata(self) -> None:
        from elspeth.core.landscape.schema import metadata, token_outcomes_table

        assert token_outcomes_table is not None
        assert "token_outcomes" in metadata.tables

    def test_table_has_required_columns(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        columns = {c.name for c in token_outcomes_table.columns}
        required = {
            "outcome_id",
            "run_id",
            "token_id",
            "outcome",
            "is_terminal",
            "recorded_at",
            "sink_name",
            "batch_id",
            "fork_group_id",
            "join_group_id",
            "expand_group_id",
            "error_hash",
            "context_json",
        }
        assert required.issubset(columns)

    def test_outcome_id_is_primary_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        pk_columns = [c.name for c in token_outcomes_table.primary_key.columns]
        assert pk_columns == ["outcome_id"]

    def test_run_id_has_foreign_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        run_id_col = token_outcomes_table.c.run_id
        fk_targets = [fk.target_fullname for fk in run_id_col.foreign_keys]
        assert "runs.run_id" in fk_targets

    def test_token_id_has_foreign_key(self) -> None:
        from elspeth.core.landscape.schema import token_outcomes_table

        token_id_col = token_outcomes_table.c.token_id
        fk_targets = [fk.target_fullname for fk in token_id_col.foreign_keys]
        assert "tokens.token_id" in fk_targets


class TestRecordTokenOutcome:
    """Test record_token_outcome() method."""

    @pytest.fixture
    def run_with_token(self, recorder):
        """Create a run with a token for testing."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig

        # Begin run
        run = recorder.begin_run(config={"test": True}, canonical_version="v1")

        # Register source node
        recorder.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Create row and token
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source_1",
            row_index=0,
            data={"id": 1},
        )
        token = recorder.create_token(
            row_id=row.row_id,
        )

        return run, token

    def test_record_completed_outcome(self, recorder, run_with_token) -> None:
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        assert outcome_id is not None
        assert outcome_id.startswith("out_")

    def test_record_routed_outcome(self, recorder, run_with_token) -> None:
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.ROUTED,
            sink_name="errors",
        )

        assert outcome_id is not None

    def test_record_buffered_then_terminal(self, recorder, run_with_token) -> None:
        """BUFFERED followed by terminal should succeed."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig

        run, token = run_with_token

        # Create an aggregation node (required for batches)
        recorder.register_node(
            run_id=run.run_id,
            node_id="agg_node_1",
            plugin_name="test_aggregation",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )

        # Create a batch (required for batch_id FK)
        batch = recorder.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node_1",
        )

        # First record BUFFERED (non-terminal) with required batch_id
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.BUFFERED,
            batch_id=batch.batch_id,
        )

        # Then record terminal outcome with same batch_id
        outcome_id = recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.CONSUMED_IN_BATCH,
            batch_id=batch.batch_id,
        )

        assert outcome_id is not None

    def test_duplicate_terminal_raises(self, recorder, run_with_token) -> None:
        """Two terminal outcomes for same token should raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        # First terminal outcome
        recorder.record_token_outcome(
            run_id=run.run_id,
            token_id=token.token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="output",
        )

        # Second terminal outcome should fail
        with pytest.raises(IntegrityError):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.ROUTED,
                sink_name="errors",
            )


class TestOutcomeContractValidation:
    """Test that record_token_outcome enforces required fields per outcome type.

    See docs/audit/tokens/00-token-outcome-contract.md for the contract.
    """

    @pytest.fixture
    def run_with_token(self, recorder):
        """Create a run with a token for testing validation."""
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)
        return run, token

    def test_completed_requires_sink_name(self, recorder, run_with_token) -> None:
        """COMPLETED outcome must have sink_name."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="COMPLETED outcome requires sink_name"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
            )

    def test_routed_requires_sink_name(self, recorder, run_with_token) -> None:
        """ROUTED outcome must have sink_name."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="ROUTED outcome requires sink_name"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.ROUTED,
            )

    def test_forked_requires_fork_group_id(self, recorder, run_with_token) -> None:
        """FORKED outcome must have fork_group_id."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="FORKED outcome requires fork_group_id"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.FORKED,
            )

    def test_failed_requires_error_hash(self, recorder, run_with_token) -> None:
        """FAILED outcome must have error_hash."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="FAILED outcome requires error_hash"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.FAILED,
            )

    def test_quarantined_requires_error_hash(self, recorder, run_with_token) -> None:
        """QUARANTINED outcome must have error_hash."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="QUARANTINED outcome requires error_hash"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.QUARANTINED,
            )

    def test_coalesced_requires_join_group_id(self, recorder, run_with_token) -> None:
        """COALESCED outcome must have join_group_id."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="COALESCED outcome requires join_group_id"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COALESCED,
            )

    def test_expanded_requires_expand_group_id(self, recorder, run_with_token) -> None:
        """EXPANDED outcome must have expand_group_id."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="EXPANDED outcome requires expand_group_id"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.EXPANDED,
            )

    def test_buffered_requires_batch_id(self, recorder, run_with_token) -> None:
        """BUFFERED outcome must have batch_id."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="BUFFERED outcome requires batch_id"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.BUFFERED,
            )

    def test_consumed_in_batch_requires_batch_id(self, recorder, run_with_token) -> None:
        """CONSUMED_IN_BATCH outcome must have batch_id."""
        from elspeth.contracts import RowOutcome

        run, token = run_with_token

        with pytest.raises(ValueError, match="CONSUMED_IN_BATCH outcome requires batch_id"):
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.CONSUMED_IN_BATCH,
            )


class TestGetTokenOutcome:
    """Test get_token_outcome() method."""

    @pytest.fixture
    def run_with_outcome(self, recorder):
        """Create run, token, and outcome for testing."""
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)
        outcome_id = recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out")
        return run, token, outcome_id

    def test_get_token_outcome_returns_dataclass(self, recorder, run_with_outcome) -> None:
        from elspeth.contracts import TokenOutcome

        _run, token, _ = run_with_outcome
        result = recorder.get_token_outcome(token.token_id)

        assert isinstance(result, TokenOutcome)
        assert result.token_id == token.token_id
        assert result.outcome.value == "completed"

    def test_get_token_outcome_returns_terminal_over_buffered(self, recorder, run_with_outcome) -> None:
        """Should return terminal outcome, not BUFFERED."""
        _run, token, _ = run_with_outcome
        # The fixture already recorded COMPLETED (terminal)
        result = recorder.get_token_outcome(token.token_id)
        assert result.is_terminal is True

    def test_get_nonexistent_returns_none(self, recorder) -> None:
        result = recorder.get_token_outcome("nonexistent_token")
        assert result is None


class TestExplainIncludesOutcome:
    """Test that explain() returns recorded outcomes."""

    def test_explain_returns_outcome(self, recorder) -> None:
        from elspeth.contracts import Determinism, NodeType, RowOutcome
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.lineage import explain

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)
        recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out")

        result = explain(recorder, run.run_id, token_id=token.token_id)

        assert result is not None
        assert result.outcome is not None
        assert result.outcome.outcome == RowOutcome.COMPLETED

    def test_explain_returns_none_outcome_when_not_recorded(self, recorder) -> None:
        from elspeth.contracts import Determinism, NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.lineage import explain

        run = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"fields": "dynamic"}),
        )
        row = recorder.create_row(run.run_id, "src", 0, {"x": 1})
        token = recorder.create_token(row.row_id)
        # No outcome recorded

        result = explain(recorder, run.run_id, token_id=token.token_id)

        assert result is not None
        assert result.outcome is None
