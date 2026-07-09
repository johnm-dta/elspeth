"""Execution-wiring DTOs consumed by graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from elspeth.core.dag.models import GraphValidationError

if TYPE_CHECKING:
    from elspeth.contracts import TransformProtocol
    from elspeth.core.config import TransformSettings


@dataclass(frozen=True, slots=True)
class WiredTransform:
    """Pair a transform plugin instance with its wiring settings."""

    plugin: TransformProtocol
    settings: TransformSettings

    def __post_init__(self) -> None:
        """Ensure wiring metadata matches the instantiated plugin."""
        if self.plugin.name != self.settings.plugin:
            raise GraphValidationError(
                f"WiredTransform mismatch: settings.plugin='{self.settings.plugin}' but plugin instance name='{self.plugin.name}'.",
                component_id=self.settings.name,
                component_type="transform",
            )
