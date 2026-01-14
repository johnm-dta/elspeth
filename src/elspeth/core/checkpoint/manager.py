"""CheckpointManager for creating and loading checkpoints."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import asc, delete, desc, select

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.models import Checkpoint
from elspeth.core.landscape.schema import checkpoints_table


class CheckpointManager:
    """Manages checkpoint creation and retrieval.

    Checkpoints capture run progress at row boundaries, enabling
    resume after crash. Each checkpoint records:
    - Which token was being processed
    - Which node it was at
    - A monotonic sequence number for ordering
    - Optional aggregation state for stateful plugins
    """

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize with Landscape database.

        Args:
            db: LandscapeDB instance for storage
        """
        self._db = db

    def create_checkpoint(
        self,
        run_id: str,
        token_id: str,
        node_id: str,
        sequence_number: int,
        aggregation_state: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint at current progress point.

        Args:
            run_id: The run being checkpointed
            token_id: Current token being processed
            node_id: Current node in the pipeline
            sequence_number: Monotonic progress marker
            aggregation_state: Optional serializable aggregation buffers

        Returns:
            The created Checkpoint
        """
        checkpoint_id = f"cp-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        agg_json = json.dumps(aggregation_state) if aggregation_state else None

        with self._db.engine.connect() as conn:
            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    token_id=token_id,
                    node_id=node_id,
                    sequence_number=sequence_number,
                    aggregation_state_json=agg_json,
                    created_at=now,
                )
            )
            conn.commit()

        return Checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            token_id=token_id,
            node_id=node_id,
            sequence_number=sequence_number,
            aggregation_state_json=agg_json,
            created_at=now,
        )

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to get checkpoint for

        Returns:
            Latest Checkpoint or None if no checkpoints exist
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
                .order_by(desc(checkpoints_table.c.sequence_number))
                .limit(1)
            ).fetchone()

        if result is None:
            return None

        return Checkpoint(
            checkpoint_id=result.checkpoint_id,
            run_id=result.run_id,
            token_id=result.token_id,
            node_id=result.node_id,
            sequence_number=result.sequence_number,
            aggregation_state_json=result.aggregation_state_json,
            created_at=result.created_at,
        )

    def get_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a run, ordered by sequence.

        Args:
            run_id: The run to get checkpoints for

        Returns:
            List of Checkpoints ordered by sequence_number
        """
        with self._db.engine.connect() as conn:
            results = conn.execute(
                select(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
                .order_by(asc(checkpoints_table.c.sequence_number))
            ).fetchall()

        return [
            Checkpoint(
                checkpoint_id=r.checkpoint_id,
                run_id=r.run_id,
                token_id=r.token_id,
                node_id=r.node_id,
                sequence_number=r.sequence_number,
                aggregation_state_json=r.aggregation_state_json,
                created_at=r.created_at,
            )
            for r in results
        ]

    def delete_checkpoints(self, run_id: str) -> int:
        """Delete all checkpoints for a completed run.

        Called after successful run completion to clean up.

        Args:
            run_id: The run to clean up

        Returns:
            Number of checkpoints deleted
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(
                delete(checkpoints_table).where(checkpoints_table.c.run_id == run_id)
            )
            conn.commit()
            return result.rowcount
