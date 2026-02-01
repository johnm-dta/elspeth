# tests/property/engine/test_token_lifecycle_state_machine.py
"""Property-based stateful tests for token lifecycle state machine.

TOKEN LIFECYCLE STATE MACHINE:
Tokens are row instances that flow through the DAG. They follow a strict
state machine with these properties:

States:
- CREATED: Token just created from source row
- PROCESSING: Token being processed by transforms
- FORKED: Token split into multiple paths (parent becomes terminal)
- COALESCED: Token merged in join point
- TERMINAL: Token reached final state (COMPLETED, QUARANTINED, etc.)

Key Invariants:
1. Token ID is immutable once created
2. Token always links to valid row_id
3. Fork creates exactly N children with parent_token_id set
4. Terminal states are final (no further transitions)
5. Every token eventually reaches a terminal state

These tests use Hypothesis RuleBasedStateMachine to explore all possible
state transitions and verify invariants hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, multiple, rule
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import (
    Determinism,
    NodeStateStatus,
    NodeType,
    RowOutcome,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from tests.property.conftest import (
    multiple_branches,
    row_data,
)

# =============================================================================
# Model Types for State Machine
# =============================================================================


class TokenState(Enum):
    """States in the token lifecycle state machine."""

    CREATED = auto()  # Token just created from source
    PROCESSING = auto()  # Token being processed (passed through transform)
    FORKED = auto()  # Token was forked (terminal for parent)
    COALESCED = auto()  # Token was coalesced (terminal for branch tokens)
    COMPLETED = auto()  # Token reached sink (terminal)
    QUARANTINED = auto()  # Token failed validation (terminal)
    CONSUMED_IN_BATCH = auto()  # Token absorbed by aggregation (terminal)


# Terminal states - no transitions allowed after reaching these
TERMINAL_STATES = {
    TokenState.FORKED,
    TokenState.COALESCED,
    TokenState.COMPLETED,
    TokenState.QUARANTINED,
    TokenState.CONSUMED_IN_BATCH,
}


@dataclass
class ModelToken:
    """Model representation of a token for state machine verification."""

    token_id: str
    row_id: str
    state: TokenState
    parent_token_id: str | None = None
    branch_name: str | None = None
    fork_group_id: str | None = None
    children: list[str] = field(default_factory=list)  # Child token IDs if forked
    processed: bool = False  # Whether token has been processed by transform


# =============================================================================
# Test Helpers
# =============================================================================


def create_dynamic_schema() -> SchemaConfig:
    """Create a dynamic schema config for testing."""
    return SchemaConfig.from_dict({"fields": "dynamic"})


def count_tokens_for_run(db: LandscapeDB, run_id: str) -> int:
    """Count all tokens for a run."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


def get_token_parent_ids(db: LandscapeDB, token_id: str) -> list[str]:
    """Get parent token IDs for a token."""
    with db.connection() as conn:
        results = conn.execute(
            text("SELECT parent_token_id FROM token_parents WHERE token_id = :token_id"),
            {"token_id": token_id},
        ).fetchall()
        return [r[0] for r in results]


def verify_fork_children_have_parents(db: LandscapeDB, run_id: str) -> int:
    """Count fork children missing parent links."""
    with db.connection() as conn:
        result = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM tokens t
                JOIN rows r ON r.row_id = t.row_id
                LEFT JOIN token_parents p ON p.token_id = t.token_id
                WHERE r.run_id = :run_id
                  AND t.fork_group_id IS NOT NULL
                  AND p.token_id IS NULL
            """),
            {"run_id": run_id},
        ).scalar()
        return result or 0


# =============================================================================
# Token Lifecycle State Machine
# =============================================================================


class TokenLifecycleStateMachine(RuleBasedStateMachine):
    """Stateful property tests for token lifecycle.

    This explores the state space of:
    - Token creation from source rows
    - Token processing through transforms
    - Token forking to multiple branches
    - Token coalescing from branches
    - Token reaching terminal states

    Model state tracks expected token states and verifies the database
    state matches after each operation.
    """

    # Bundles for managing tokens in different states
    active_tokens = Bundle("active_tokens")  # Tokens that can still transition

    def __init__(self) -> None:
        super().__init__()

        # Database and recorder
        self.db = LandscapeDB.in_memory()
        self.recorder = LandscapeRecorder(self.db)

        # Begin a run
        self.run = self.recorder.begin_run(
            config={"source": {"plugin": "test"}, "sinks": {"default": {"plugin": "test"}}},
            canonical_version="1.0",
        )

        # Register nodes
        self.source_node = self.recorder.register_node(
            run_id=self.run.run_id,
            plugin_name="test_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0.0",
            config={},
            schema_config=create_dynamic_schema(),
        )

        self.transform_node = self.recorder.register_node(
            run_id=self.run.run_id,
            plugin_name="test_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0.0",
            config={},
            sequence=1,
            schema_config=create_dynamic_schema(),
        )

        self.sink_node = self.recorder.register_node(
            run_id=self.run.run_id,
            plugin_name="test_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0.0",
            config={},
            sequence=2,
            determinism=Determinism.IO_WRITE,
            schema_config=create_dynamic_schema(),
        )

        # Model state - tracks expected state of all tokens
        self.model_tokens: dict[str, ModelToken] = {}
        self.row_index = 0
        self.step_counter = 0

    def teardown(self) -> None:
        """Close the in-memory database between state machine runs."""
        self.db.close()

    # -------------------------------------------------------------------------
    # Rules: Token Creation
    # -------------------------------------------------------------------------

    @rule(target=active_tokens, data=row_data)
    def create_token(self, data: dict[str, Any]) -> str:
        """Create a new token from a source row."""
        # Create row in database
        row = self.recorder.create_row(
            run_id=self.run.run_id,
            source_node_id=self.source_node.node_id,
            row_index=self.row_index,
            data=data,
        )
        self.row_index += 1

        # Create token
        token = self.recorder.create_token(row_id=row.row_id)

        # Update model
        self.model_tokens[token.token_id] = ModelToken(
            token_id=token.token_id,
            row_id=row.row_id,
            state=TokenState.CREATED,
        )

        return token.token_id

    # -------------------------------------------------------------------------
    # Rules: Token Processing
    # -------------------------------------------------------------------------

    @rule(token_id=active_tokens, data=row_data)
    def process_token(self, token_id: str, data: dict[str, Any]) -> None:
        """Process a token through a transform (record node state)."""
        model = self.model_tokens[token_id]

        # Only non-terminal tokens can be processed
        if model.state in TERMINAL_STATES:
            return

        # A token can only be processed once per transform node
        # (the node_states table has unique constraint on token_id, node_id, attempt)
        if model.processed:
            return

        # Begin and complete node state
        self.step_counter += 1
        state = self.recorder.begin_node_state(
            token_id=token_id,
            node_id=self.transform_node.node_id,
            run_id=self.run.run_id,
            step_index=self.step_counter,
            input_data=data,
        )

        self.recorder.complete_node_state(
            state_id=state.state_id,
            status=NodeStateStatus.COMPLETED,
            output_data=data,
            duration_ms=1.0,
        )

        # Update model state
        model.state = TokenState.PROCESSING
        model.processed = True

    # -------------------------------------------------------------------------
    # Rules: Token Forking
    # -------------------------------------------------------------------------

    @rule(target=active_tokens, token_id=active_tokens, branches=multiple_branches)
    def fork_token(self, token_id: str, branches: list[str]) -> Any:
        """Fork a token into multiple child tokens.

        Returns multiple() to add each child individually to the active_tokens bundle.
        """
        model = self.model_tokens[token_id]

        # Only non-terminal tokens can be forked
        if model.state in TERMINAL_STATES:
            return multiple()  # Return empty multiple - no children added

        self.step_counter += 1

        # Fork in database (this also records FORKED outcome for parent)
        children, fork_group_id = self.recorder.fork_token(
            parent_token_id=token_id,
            row_id=model.row_id,
            branches=branches,
            run_id=self.run.run_id,
            step_in_pipeline=self.step_counter,
        )

        # Update parent model state
        model.state = TokenState.FORKED
        model.children = [c.token_id for c in children]

        # Create model entries for children
        child_ids = []
        for child, branch in zip(children, branches, strict=True):
            self.model_tokens[child.token_id] = ModelToken(
                token_id=child.token_id,
                row_id=model.row_id,
                state=TokenState.CREATED,
                parent_token_id=token_id,
                branch_name=branch,
                fork_group_id=fork_group_id,
            )
            child_ids.append(child.token_id)

        return multiple(*child_ids)  # Return each child ID as separate bundle entry

    # -------------------------------------------------------------------------
    # Rules: Token Terminal States
    # -------------------------------------------------------------------------

    @rule(token_id=active_tokens)
    def complete_token(self, token_id: str) -> None:
        """Mark a token as completed (reached sink)."""
        model = self.model_tokens[token_id]

        # Only non-terminal tokens can complete
        if model.state in TERMINAL_STATES:
            return

        # Record outcome in database
        self.recorder.record_token_outcome(
            run_id=self.run.run_id,
            token_id=token_id,
            outcome=RowOutcome.COMPLETED,
            sink_name="default",
        )

        # Update model state
        model.state = TokenState.COMPLETED

    @rule(token_id=active_tokens)
    def quarantine_token(self, token_id: str) -> None:
        """Quarantine a token (mark as failed)."""
        model = self.model_tokens[token_id]

        # Only non-terminal tokens can be quarantined
        if model.state in TERMINAL_STATES:
            return

        # Record outcome in database
        self.recorder.record_token_outcome(
            run_id=self.run.run_id,
            token_id=token_id,
            outcome=RowOutcome.QUARANTINED,
            error_hash="test_error_hash",
        )

        # Update model state
        model.state = TokenState.QUARANTINED

    # -------------------------------------------------------------------------
    # Invariants
    # -------------------------------------------------------------------------

    @invariant()
    def token_id_is_immutable(self) -> None:
        """Invariant: Token IDs in database match model token IDs.

        Once a token is created, its ID never changes.
        """
        with self.db.connection() as conn:
            for token_id, model in self.model_tokens.items():
                result = conn.execute(
                    text("SELECT token_id, row_id FROM tokens WHERE token_id = :token_id"),
                    {"token_id": token_id},
                ).fetchone()

                assert result is not None, f"Token {token_id} missing from database"
                assert result[0] == token_id, f"Token ID mismatch: {result[0]} != {token_id}"
                assert result[1] == model.row_id, f"Row ID mismatch for token {token_id}: {result[1]} != {model.row_id}"

    @invariant()
    def token_links_to_valid_row(self) -> None:
        """Invariant: Every token references a valid row_id."""
        with self.db.connection() as conn:
            # Check for orphan tokens
            orphan_count = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM tokens t
                    LEFT JOIN rows r ON r.row_id = t.row_id
                    WHERE r.row_id IS NULL
                """),
            ).scalar()

            assert orphan_count == 0, f"Found {orphan_count} tokens with invalid row_id"

    @invariant()
    def fork_children_have_parent_links(self) -> None:
        """Invariant: Fork children have parent_token_id recorded."""
        missing = verify_fork_children_have_parents(self.db, self.run.run_id)
        assert missing == 0, f"Found {missing} fork children missing parent links"

    @invariant()
    def fork_creates_correct_number_of_children(self) -> None:
        """Invariant: Forked parents have correct number of children in model."""
        for token_id, model in self.model_tokens.items():
            if model.state == TokenState.FORKED:
                # Verify all expected children exist in database
                for child_id in model.children:
                    assert child_id in self.model_tokens, f"Child {child_id} of forked parent {token_id} missing from model"

                    child_model = self.model_tokens[child_id]
                    assert child_model.parent_token_id == token_id, (
                        f"Child {child_id} has wrong parent: {child_model.parent_token_id} != {token_id}"
                    )

    @invariant()
    def terminal_states_have_outcomes(self) -> None:
        """Invariant: Tokens in terminal states have outcomes recorded.

        Note: FORKED outcome is recorded by fork_token() atomically.
        """
        with self.db.connection() as conn:
            for token_id, model in self.model_tokens.items():
                if model.state in {TokenState.COMPLETED, TokenState.QUARANTINED}:
                    result = conn.execute(
                        text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id"),
                        {"token_id": token_id},
                    ).fetchone()

                    assert result is not None, f"Token {token_id} in state {model.state} has no outcome"

                elif model.state == TokenState.FORKED:
                    # FORKED tokens have outcome recorded by recorder.fork_token()
                    result = conn.execute(
                        text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id"),
                        {"token_id": token_id},
                    ).fetchone()

                    assert result is not None, f"Forked parent {token_id} has no FORKED outcome"
                    assert result[0] == RowOutcome.FORKED.value, f"Wrong outcome for forked token: {result[0]}"

    @invariant()
    def model_count_matches_database(self) -> None:
        """Invariant: Model token count matches database count."""
        db_count = count_tokens_for_run(self.db, self.run.run_id)
        model_count = len(self.model_tokens)

        assert db_count == model_count, f"Token count mismatch: database={db_count}, model={model_count}"


# Create the test class that pytest will discover
TestTokenLifecycleStateMachine = TokenLifecycleStateMachine.TestCase
TestTokenLifecycleStateMachine.settings = settings(max_examples=50, stateful_step_count=30, deadline=None)


# =============================================================================
# Additional Non-Stateful Token Lifecycle Properties
# =============================================================================


class TestTokenLifecycleInvariants:
    """Property tests for token lifecycle invariants using @given decorators."""

    @given(token_count=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_token_id_uniqueness(self, token_count: int) -> None:
        """Property: Token IDs are unique across multiple creations."""
        with LandscapeDB.in_memory() as db:
            recorder = LandscapeRecorder(db)

            run = recorder.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            source_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )

            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data={"value": 1},
            )

            # Create multiple tokens for same row
            token_ids = set()
            for _ in range(token_count):
                token = recorder.create_token(row_id=row.row_id)
                assert token.token_id not in token_ids, f"Duplicate token ID: {token.token_id}"
                token_ids.add(token.token_id)

    @given(branch_count=st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_fork_atomic_parent_outcome(self, branch_count: int) -> None:
        """Property: Fork atomically records FORKED outcome for parent."""
        with LandscapeDB.in_memory() as db:
            recorder = LandscapeRecorder(db)

            run = recorder.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            source_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )

            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data={"value": 1},
            )

            token = recorder.create_token(row_id=row.row_id)

            # Generate branch names based on count
            branches = [f"branch_{i}" for i in range(branch_count)]

            # Fork should record parent outcome atomically
            children, _fork_group_id = recorder.fork_token(
                parent_token_id=token.token_id,
                row_id=row.row_id,
                branches=branches,
                run_id=run.run_id,
                step_in_pipeline=1,
            )

            # Verify parent has FORKED outcome
            with db.connection() as conn:
                result = conn.execute(
                    text("SELECT outcome FROM token_outcomes WHERE token_id = :token_id"),
                    {"token_id": token.token_id},
                ).fetchone()

                assert result is not None, "Parent token missing outcome after fork"
                assert result[0] == RowOutcome.FORKED.value, f"Wrong outcome: {result[0]}"

            # Verify children exist with parent links
            assert len(children) == branch_count
            for child in children:
                parent_ids = get_token_parent_ids(db, child.token_id)
                assert token.token_id in parent_ids, f"Child {child.token_id} missing parent link"

    @given(data=row_data)
    @settings(max_examples=20, deadline=None)
    def test_terminal_state_is_final(self, data: dict[str, Any]) -> None:
        """Property: Once a token reaches terminal state, no new outcomes can be recorded."""
        with LandscapeDB.in_memory() as db:
            recorder = LandscapeRecorder(db)

            run = recorder.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            source_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )

            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data=data,
            )

            token = recorder.create_token(row_id=row.row_id)

            # Record COMPLETED outcome
            recorder.record_token_outcome(
                run_id=run.run_id,
                token_id=token.token_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="default",
            )

            # Second terminal outcome should violate unique constraint
            with pytest.raises(IntegrityError):
                recorder.record_token_outcome(
                    run_id=run.run_id,
                    token_id=token.token_id,
                    outcome=RowOutcome.QUARANTINED,
                    error_hash="test_error_hash",
                )

            # Count outcomes - should be exactly 1
            with db.connection() as conn:
                count = conn.execute(
                    text("SELECT COUNT(*) FROM token_outcomes WHERE token_id = :token_id"),
                    {"token_id": token.token_id},
                ).scalar()

                assert count == 1, f"Token should have exactly 1 outcome, got {count}"

    @given(data=row_data)
    @settings(max_examples=20, deadline=None)
    def test_row_data_preserved_through_lifecycle(self, data: dict[str, Any]) -> None:
        """Property: row_id remains constant through token lifecycle."""
        with LandscapeDB.in_memory() as db:
            recorder = LandscapeRecorder(db)

            run = recorder.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            source_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )

            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data=data,
            )

            token = recorder.create_token(row_id=row.row_id)
            original_row_id = row.row_id

            # Fork token
            children, _ = recorder.fork_token(
                parent_token_id=token.token_id,
                row_id=row.row_id,
                branches=["a", "b"],
                run_id=run.run_id,
                step_in_pipeline=1,
            )

            # All children should have same row_id
            for child in children:
                with db.connection() as conn:
                    result = conn.execute(
                        text("SELECT row_id FROM tokens WHERE token_id = :token_id"),
                        {"token_id": child.token_id},
                    ).fetchone()
                    assert result is not None, f"Child token {child.token_id} not found"
                    assert result[0] == original_row_id, f"Child has wrong row_id: {result[0]} != {original_row_id}"

    @given(parent_count=st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_coalesce_creates_merged_token(self, parent_count: int) -> None:
        """Property: Coalesce creates a new token linking to parent tokens."""
        with LandscapeDB.in_memory() as db:
            recorder = LandscapeRecorder(db)

            run = recorder.begin_run(
                config={"source": {"plugin": "test"}},
                canonical_version="1.0",
            )

            source_node = recorder.register_node(
                run_id=run.run_id,
                plugin_name="test_source",
                node_type=NodeType.SOURCE,
                plugin_version="1.0.0",
                config={},
                schema_config=create_dynamic_schema(),
            )

            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data={"value": 1},
            )

            # Create parent token and fork with variable number of branches
            branches = [chr(ord("a") + i) for i in range(parent_count)]
            parent = recorder.create_token(row_id=row.row_id)
            children, _ = recorder.fork_token(
                parent_token_id=parent.token_id,
                row_id=row.row_id,
                branches=branches,
                run_id=run.run_id,
                step_in_pipeline=1,
            )

            # Coalesce the children
            merged = recorder.coalesce_tokens(
                parent_token_ids=[c.token_id for c in children],
                row_id=row.row_id,
                step_in_pipeline=2,
            )

            # Verify merged token exists and has correct parents
            assert merged.token_id is not None
            assert merged.row_id == row.row_id
            assert merged.join_group_id is not None

            parent_ids = get_token_parent_ids(db, merged.token_id)
            assert len(parent_ids) == parent_count, f"Merged token should have {parent_count} parents, got {len(parent_ids)}"
            for child in children:
                assert child.token_id in parent_ids, f"Merged token missing parent {child.token_id}"
