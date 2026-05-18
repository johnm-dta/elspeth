# tests/unit/core/test_token_outcomes.py
"""Tests for token outcome recording."""

import pytest

from elspeth.contracts import Determinism
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory, make_landscape_db


@pytest.fixture
def landscape_db() -> LandscapeDB:
    """Create a function-scoped in-memory landscape database.

    Function scope ensures test isolation — each test gets a fresh database.
    """
    return make_landscape_db()


@pytest.fixture
def factory(landscape_db: LandscapeDB) -> RecorderFactory:
    """Create a RecorderFactory with the test database."""
    return make_factory(landscape_db)


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
        assert "path" in fields
        assert "completed" in fields
        assert "recorded_at" in fields

    def test_token_outcome_instantiation(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts import TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(UTC),
        )
        assert outcome.outcome_id == "out_123"
        assert outcome.completed is True
        assert outcome.path is TerminalPath.DEFAULT_FLOW

    def test_token_outcome_is_frozen(self) -> None:
        """TokenOutcome should be immutable (frozen dataclass)."""
        from datetime import UTC, datetime

        from elspeth.contracts import TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
            recorded_at=datetime.now(UTC),
        )

        with pytest.raises(AttributeError):
            outcome.outcome_id = "different"  # type: ignore[misc]

    def test_token_outcome_optional_fields(self) -> None:
        """TokenOutcome should have optional context fields."""
        from datetime import UTC, datetime

        from elspeth.contracts import TokenOutcome

        # All optional fields should default to None
        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            completed=True,
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

        from elspeth.contracts import TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            completed=True,
            recorded_at=datetime.now(UTC),
            sink_name="error_sink",
        )
        assert outcome.sink_name == "error_sink"

    def test_token_outcome_with_batch_context(self) -> None:
        """TokenOutcome can record batch-specific context."""
        from datetime import UTC, datetime

        from elspeth.contracts import TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            completed=True,
            recorded_at=datetime.now(UTC),
            batch_id="batch_abc",
        )
        assert outcome.batch_id == "batch_abc"

    def test_token_outcome_accepts_terminal_pairs(self) -> None:
        """TokenOutcome accepts the ADR-019 two-axis terminal pair."""
        from datetime import UTC, datetime

        from elspeth.contracts import TokenOutcome

        outcome = TokenOutcome(
            outcome_id="out_123",
            run_id="run_456",
            token_id="tok_789",
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.UNROUTED,
            completed=True,
            recorded_at=datetime.now(UTC),
            error_hash="0123456789abcdef",
        )
        assert outcome.outcome is TerminalOutcome.FAILURE
        assert outcome.path is TerminalPath.UNROUTED
        assert outcome.completed is True


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
            "path",
            "completed",
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

    def test_composite_fk_to_tokens(self) -> None:
        """token_outcomes.(token_id, run_id) is a composite FK to tokens.(token_id, run_id).

        This prevents cross-run contamination at the DB level: token_id and run_id
        must belong together in the tokens table, not just exist independently.
        """
        from elspeth.core.landscape.schema import token_outcomes_table

        # run_id FK target is now tokens.run_id (via composite FK),
        # not runs.run_id (the old independent FK). Referential integrity
        # to runs is enforced transitively: tokens.run_id -> runs.run_id.
        run_id_col = token_outcomes_table.c.run_id
        run_id_fk_targets = [fk.target_fullname for fk in run_id_col.foreign_keys]
        assert "tokens.run_id" in run_id_fk_targets

        token_id_col = token_outcomes_table.c.token_id
        token_id_fk_targets = [fk.target_fullname for fk in token_id_col.foreign_keys]
        assert "tokens.token_id" in token_id_fk_targets


class TestRecordTokenOutcome:
    """Test record_token_outcome() method."""

    @pytest.fixture
    def run_with_token(self, factory):
        """Create a run with a token for testing."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        # Begin run
        run = factory.run_lifecycle.begin_run(config={"test": True}, canonical_version="v1")

        # Register source node
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="source_1",
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )

        # Create row and token
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source_1",
            row_index=0,
            data={"id": 1},
        )
        token = factory.data_flow.create_token(
            row_id=row.row_id,
        )

        return run, token

    def test_record_completed_outcome(self, factory, run_with_token) -> None:
        run, token = run_with_token

        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        assert outcome_id is not None
        assert outcome_id.startswith("out_")

    def test_record_routed_outcome(self, factory, run_with_token) -> None:
        run, token = run_with_token

        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.GATE_ROUTED,
            sink_name="errors",
        )

        assert outcome_id is not None

    def test_record_buffered_then_terminal(self, factory, run_with_token) -> None:
        """BUFFERED followed by terminal should succeed."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        run, token = run_with_token

        # Create an aggregation node (required for batches)
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="agg_node_1",
            plugin_name="test_aggregation",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )

        # Create a batch (required for batch_id FK)
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="agg_node_1",
        )

        # First record BUFFERED (non-terminal) with required batch_id
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=None,
            path=TerminalPath.BUFFERED,
            batch_id=batch.batch_id,
        )

        # Then record terminal outcome with same batch_id
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.TRANSIENT,
            path=TerminalPath.BATCH_CONSUMED,
            batch_id=batch.batch_id,
        )

        assert outcome_id is not None

    def test_duplicate_terminal_raises(self, factory, run_with_token) -> None:
        """Two terminal outcomes for same token should raise LandscapeRecordError.

        The DB unique constraint is the enforcement point, but repository writes
        wrap DB failures so callers get a stable audit-layer exception.
        """
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.errors import LandscapeRecordError

        run, token = run_with_token

        # First terminal outcome
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="output",
        )

        # Second terminal outcome should fail
        with pytest.raises(LandscapeRecordError) as exc_info:
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
                sink_name="errors",
            )
        assert isinstance(exc_info.value.__cause__, IntegrityError)


class TestOutcomeContractValidation:
    """Test that record_token_outcome enforces required fields per outcome type.

    See docs/contracts/token-outcomes/00-token-outcome-contract.md for the contract.
    """

    @pytest.fixture
    def run_with_token(self, factory):
        """Create a run with a token for testing validation."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        row = factory.data_flow.create_row(run.run_id, "src", 0, {"x": 1})
        token = factory.data_flow.create_token(row.row_id)
        return run, token

    def test_completed_requires_sink_name(self, factory, run_with_token) -> None:
        """(SUCCESS, DEFAULT_FLOW) outcome must have sink_name."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(SUCCESS, DEFAULT_FLOW\) outcome requires sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
            )

    def test_routed_requires_sink_name(self, factory, run_with_token) -> None:
        """(SUCCESS, GATE_ROUTED) outcome must have sink_name."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(SUCCESS, GATE_ROUTED\) outcome requires sink_name"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
            )

    def test_forked_requires_fork_group_id(self, factory, run_with_token) -> None:
        """(TRANSIENT, FORK_PARENT) outcome must have fork_group_id."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(TRANSIENT, FORK_PARENT\) outcome requires fork_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
            )

    def test_failed_requires_error_hash(self, factory, run_with_token) -> None:
        """(FAILURE, UNROUTED) outcome must have error_hash."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(FAILURE, UNROUTED\) outcome requires error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.UNROUTED,
            )

    def test_quarantined_requires_error_hash(self, factory, run_with_token) -> None:
        """(FAILURE, QUARANTINED_AT_SOURCE) outcome must have error_hash."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(FAILURE, QUARANTINED_AT_SOURCE\) outcome requires error_hash"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.QUARANTINED_AT_SOURCE,
            )

    def test_coalesced_requires_join_group_id(self, factory, run_with_token) -> None:
        """(SUCCESS, COALESCED) outcome must have join_group_id."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(SUCCESS, COALESCED\) outcome requires join_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.COALESCED,
                sink_name="default",
            )

    def test_expanded_requires_expand_group_id(self, factory, run_with_token) -> None:
        """(TRANSIENT, EXPAND_PARENT) outcome must have expand_group_id."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(TRANSIENT, EXPAND_PARENT\) outcome requires expand_group_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.EXPAND_PARENT,
            )

    def test_buffered_requires_batch_id(self, factory, run_with_token) -> None:
        """(NULL, BUFFERED) outcome must have batch_id."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(NULL, BUFFERED\) outcome requires batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=None,
                path=TerminalPath.BUFFERED,
            )

    def test_consumed_in_batch_requires_batch_id(self, factory, run_with_token) -> None:
        """(TRANSIENT, BATCH_CONSUMED) outcome must have batch_id."""
        run, token = run_with_token

        with pytest.raises(ValueError, match=r"\(TRANSIENT, BATCH_CONSUMED\) outcome requires batch_id"):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.BATCH_CONSUMED,
            )

    def test_dropped_by_filter_requires_no_extra_fields(self, factory, run_with_token) -> None:
        """DROPPED_BY_FILTER is terminal and requires no auxiliary columns."""
        run, token = run_with_token

        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.FILTER_DROPPED,
        )
        assert outcome_id is not None


class TestGetTokenOutcome:
    """Test get_token_outcome() method."""

    @pytest.fixture
    def run_with_outcome(self, factory):
        """Create run, token, and outcome for testing."""
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        row = factory.data_flow.create_row(run.run_id, "src", 0, {"x": 1})
        token = factory.data_flow.create_token(row.row_id)
        outcome_id = factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="out",
        )
        return run, token, outcome_id

    def test_get_token_outcome_returns_dataclass(self, factory, run_with_outcome) -> None:
        from elspeth.contracts import TokenOutcome

        _run, token, _ = run_with_outcome
        result = factory.data_flow.get_token_outcome(token.token_id)

        assert isinstance(result, TokenOutcome)
        assert result.token_id == token.token_id
        assert result.outcome == TerminalOutcome.SUCCESS
        assert result.path == TerminalPath.DEFAULT_FLOW

    def test_get_token_outcome_returns_terminal_over_buffered(self, factory, run_with_outcome) -> None:
        """Should return terminal outcome, not BUFFERED."""
        _run, token, _ = run_with_outcome
        # The fixture already recorded COMPLETED (terminal)
        result = factory.data_flow.get_token_outcome(token.token_id)
        assert result.completed is True

    def test_get_nonexistent_returns_none(self, factory) -> None:
        result = factory.data_flow.get_token_outcome("nonexistent_token")
        assert result is None


class TestExplainIncludesOutcome:
    """Test that explain() returns recorded outcomes."""

    def test_explain_returns_outcome(self, factory) -> None:
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.lineage import explain

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        row = factory.data_flow.create_row(run.run_id, "src", 0, {"x": 1})
        token = factory.data_flow.create_token(row.row_id)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=token.token_id, run_id=run.run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="out",
        )

        result = explain(factory.query, factory.data_flow, run.run_id, token_id=token.token_id)

        assert result is not None
        assert result.outcome is not None
        assert result.outcome.outcome == TerminalOutcome.SUCCESS
        assert result.outcome.path == TerminalPath.DEFAULT_FLOW

    def test_explain_returns_none_outcome_when_not_recorded(self, factory) -> None:
        from elspeth.contracts import NodeType
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.lineage import explain

        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        factory.data_flow.register_node(
            run_id=run.run_id,
            node_id="src",
            plugin_name="test",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        row = factory.data_flow.create_row(run.run_id, "src", 0, {"x": 1})
        token = factory.data_flow.create_token(row.row_id)
        # No outcome recorded

        result = explain(factory.query, factory.data_flow, run.run_id, token_id=token.token_id)

        assert result is not None
        assert result.outcome is None
