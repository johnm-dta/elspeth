"""Checkpoint subsystem for crash recovery.

Provides:
- CheckpointManager: Create and load checkpoints
- RecoveryManager: Determine if/how to resume failed runs
- CheckpointCompatibilityValidator: Validate checkpoint topology compatibility
- ResumeCheck: Result of checking if a run can be resumed
- ResumePoint: Information needed to resume a run
"""

from elspeth.contracts import ResumeCheck, ResumePoint
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import RecoveryManager

__all__ = ["CheckpointCompatibilityValidator", "CheckpointManager", "RecoveryManager", "ResumeCheck", "ResumePoint"]
