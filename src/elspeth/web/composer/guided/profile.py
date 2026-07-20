"""Workflow profile value type and server-owned profile presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from elspeth.web.composer.guided.errors import InvariantError


class WorkflowProfileKind(StrEnum):
    """Closed discriminator for server-owned guided workflow profiles."""

    LIVE = "live"
    TUTORIAL = "tutorial"


_PROFILE_KEYS = frozenset(
    {
        "coaching",
        "bookends",
    }
)


@dataclass(frozen=True, slots=True)
class WorkflowProfile:
    """Server-owned behavior toggles for guided workflow variants."""

    coaching: bool
    bookends: bool

    def __post_init__(self) -> None:
        for field_name in ("coaching", "bookends"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be bool, got {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all direct profile fields for Tier-1 persistence."""

        return {
            "coaching": self.coaching,
            "bookends": self.bookends,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowProfile:
        """Deserialize a Tier-1 persisted profile record with strict shape checks."""

        if type(d) is not dict:
            raise InvariantError(f"WorkflowProfile.from_dict: malformed record {d!r}")

        keys = set(d)
        unexpected_keys = keys - _PROFILE_KEYS
        if unexpected_keys:
            raise InvariantError(f"WorkflowProfile.from_dict: unexpected keys {sorted(unexpected_keys)!r} in record {d!r}")

        missing_keys = _PROFILE_KEYS - keys
        if missing_keys:
            raise InvariantError(f"WorkflowProfile.from_dict: malformed record {d!r}")

        try:
            return cls(
                coaching=d["coaching"],
                bookends=d["bookends"],
            )
        except (TypeError, ValueError) as exc:
            raise InvariantError(f"WorkflowProfile.from_dict: {exc}") from exc


EMPTY_PROFILE = WorkflowProfile(
    coaching=False,
    bookends=False,
)

TUTORIAL_PROFILE = WorkflowProfile(
    coaching=True,
    bookends=True,
)


def profile_for_kind(kind: WorkflowProfileKind) -> WorkflowProfile:
    """Return the server-owned profile constant for a closed profile kind."""

    if kind is WorkflowProfileKind.LIVE:
        return EMPTY_PROFILE
    if kind is WorkflowProfileKind.TUTORIAL:
        return TUTORIAL_PROFILE
    raise InvariantError(f"profile_for_kind: unhandled profile kind {kind!r}")
