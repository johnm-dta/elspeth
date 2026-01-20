# tests/core/test_token_outcomes.py
"""Tests for token outcome recording."""

import pytest


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
