"""Checkpoint subsystem for crash recovery.

Provides:
- CheckpointManager: Create and load checkpoints
- RecoveryManager: Determine if/how to resume failed runs
- ResumeCheck: Result of checking if a run can be resumed
- ResumePoint: Information needed to resume a run
"""

from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import RecoveryManager, ResumeCheck, ResumePoint

__all__ = ["CheckpointManager", "RecoveryManager", "ResumeCheck", "ResumePoint"]
