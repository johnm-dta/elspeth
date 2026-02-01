# tests/cli/conftest.py
"""Shared fixtures and helpers for CLI tests."""

from pathlib import Path

from elspeth.contracts import NodeStateStatus, RowOutcome, RunStatus
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder


def verify_audit_trail(
    audit_db_path: Path,
    *,
    expected_row_count: int,
) -> None:
    """Verify audit trail integrity after a pipeline run.

    This helper validates that the Landscape audit trail contains:
    - Exactly one completed run
    - Expected number of rows with valid source hashes
    - Tokens for each row with valid node states
    - Terminal outcomes for each token
    - Artifacts from the output sink with content hashes

    Args:
        audit_db_path: Path to the Landscape SQLite database
        expected_row_count: Number of rows expected to be processed

    Raises:
        AssertionError: If any audit trail validation fails
    """
    db = LandscapeDB.from_url(f"sqlite:///{audit_db_path}")
    try:
        recorder = LandscapeRecorder(db)

        # 1. Verify run was recorded and completed
        runs = recorder.list_runs()
        assert len(runs) == 1, f"Expected 1 run, got {len(runs)}"
        run = runs[0]
        assert run.status == RunStatus.COMPLETED, f"Expected run status COMPLETED, got {run.status}"
        run_id = run.run_id

        # 2. Verify rows were recorded
        rows = recorder.get_rows(run_id)
        assert len(rows) == expected_row_count, f"Expected {expected_row_count} rows, got {len(rows)}"
        # Each row should have a source data hash
        for row in rows:
            assert row.source_data_hash, f"Row {row.row_id} missing source_data_hash"

        # 3. Verify tokens and node states for each row
        for row in rows:
            tokens = recorder.get_tokens(row.row_id)
            assert tokens, f"Row {row.row_id} has no tokens"

            for token in tokens:
                # Get node states for this token
                states = recorder.get_node_states_for_token(token.token_id)
                assert states, f"Token {token.token_id} has no node states"

                # Verify each state has input_hash and completed successfully
                for state in states:
                    assert state.input_hash, f"NodeState {state.state_id} missing input_hash"
                    # Check status is completed (not failed)
                    if state.status == NodeStateStatus.COMPLETED:
                        # Completed states must have output_hash
                        assert state.output_hash, f"Completed state {state.state_id} missing output_hash"

            # Verify terminal outcome for the row
            outcomes = recorder.get_token_outcomes_for_row(run_id, row.row_id)
            terminal_outcomes = [o for o in outcomes if o.is_terminal]
            assert terminal_outcomes, f"Row {row.row_id} has no terminal outcome"
            # All terminal outcomes should be COMPLETED (reached output sink)
            for outcome in terminal_outcomes:
                assert outcome.outcome == RowOutcome.COMPLETED, f"Expected COMPLETED outcome, got {outcome.outcome}"

        # 4. Verify artifacts were recorded
        artifacts = recorder.get_artifacts(run_id)
        assert artifacts, "No artifacts recorded"
        for artifact in artifacts:
            assert artifact.content_hash, f"Artifact {artifact.artifact_id} missing content_hash"
            assert artifact.artifact_type, f"Artifact {artifact.artifact_id} missing artifact_type"
    finally:
        db.close()
