"""Checkpoint subsystem for crash recovery.

Provides:
- CheckpointManager: Create and load checkpoints
- RecoveryManager: Determine if/how to resume failed runs
- CheckpointCompatibilityValidator: Validate checkpoint topology compatibility
- ResumeCheck: Result of checking if a run can be resumed
- ResumePoint: Information needed to resume a run
- checkpoint_dumps/checkpoint_loads: Type-preserving JSON serialization for aggregation state
"""

from elspeth.contracts import ResumeCheck, ResumePoint
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.manager import CheckpointCorruptionError, CheckpointManager, IncompatibleCheckpointError
from elspeth.core.checkpoint.recovery import RecoveryManager
from elspeth.core.checkpoint.serialization import checkpoint_dumps, checkpoint_loads

__all__ = [
    "CheckpointCompatibilityValidator",
    "CheckpointCorruptionError",
    "CheckpointManager",
    "IncompatibleCheckpointError",
    "RecoveryManager",
    "ResumeCheck",
    "ResumePoint",
    "checkpoint_dumps",
    "checkpoint_loads",
]
