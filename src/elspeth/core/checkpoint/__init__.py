"""Checkpoint subsystem for crash recovery.

Provides:
- CheckpointManager: Create and load checkpoints
- Recovery protocol for resuming crashed runs
"""

from elspeth.core.checkpoint.manager import CheckpointManager

__all__ = ["CheckpointManager"]
