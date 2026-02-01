"""Identity-based audit trail assertions.

These helpers verify token IDENTITY (which specific tokens have which outcomes),
not just outcome COUNTS. This prevents bugs like elspeth-rapid-nd3 where
count-based tests pass even when wrong tokens get wrong outcomes.
"""

from elspeth.contracts.enums import RowOutcome
from elspeth.core.landscape import LandscapeRecorder


def get_token_outcome(recorder: LandscapeRecorder, run_id: str, token_id: str) -> RowOutcome | None:
    """Get the terminal outcome for a specific token."""
    with recorder._db.connection() as conn:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import token_outcomes_table

        result = conn.execute(select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.token_id == token_id)).fetchone()

        if result is None:
            return None
        return RowOutcome(result.outcome)


def assert_token_outcome(
    recorder: LandscapeRecorder,
    run_id: str,
    token_id: str,
    expected: RowOutcome,
) -> None:
    """Assert a specific token has a specific outcome."""
    actual = get_token_outcome(recorder, run_id, token_id)
    assert actual == expected, f"Token {token_id} has outcome {actual}, expected {expected}"


def assert_all_batch_members_consumed(
    recorder: LandscapeRecorder,
    run_id: str,
    batch_id: str,
) -> None:
    """Assert ALL tokens in a batch have CONSUMED_IN_BATCH outcome."""
    with recorder._db.connection() as conn:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import batch_members_table, token_outcomes_table

        members = conn.execute(
            select(batch_members_table.c.token_id, batch_members_table.c.ordinal)
            .where(batch_members_table.c.batch_id == batch_id)
            .order_by(batch_members_table.c.ordinal)
        ).fetchall()

        assert len(members) > 0, f"Batch {batch_id} has no members"

        for member in members:
            token_id = member.token_id
            ordinal = member.ordinal

            outcome_row = conn.execute(select(token_outcomes_table.c.outcome).where(token_outcomes_table.c.token_id == token_id)).fetchone()

            assert outcome_row is not None, f"Batch member {token_id} (ordinal {ordinal}) has no outcome recorded"

            actual = RowOutcome(outcome_row.outcome)
            assert actual == RowOutcome.CONSUMED_IN_BATCH, (
                f"Batch member {token_id} (ordinal {ordinal}) has outcome {actual}, expected CONSUMED_IN_BATCH."
            )


def assert_output_token_distinct_from_inputs(
    output_token_id: str,
    input_token_ids: list[str],
) -> None:
    """Assert output token has a DIFFERENT token_id from all inputs."""
    assert output_token_id not in input_token_ids, (
        f"Output token {output_token_id} reuses an input token_id! "
        f"Token-producing operations must create NEW tokens for audit lineage. "
        f"Input token_ids: {input_token_ids}"
    )


def assert_batch_output_exists(
    recorder: LandscapeRecorder,
    batch_id: str,
) -> str:
    """Assert batch_outputs table has an entry for this batch. Returns output_id."""
    with recorder._db.connection() as conn:
        from sqlalchemy import select

        from elspeth.core.landscape.schema import batch_outputs_table

        result = conn.execute(select(batch_outputs_table).where(batch_outputs_table.c.batch_id == batch_id)).fetchone()

        assert result is not None, f"Batch {batch_id} has no entry in batch_outputs table."

        return result.output_id
