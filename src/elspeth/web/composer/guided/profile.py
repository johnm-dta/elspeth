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
        "entry_seed",
        "coaching",
        "advisor_checkpoints",
        "recipe_match",
        "bookends",
    }
)

_TUTORIAL_ENTRY_SEED = "Rate how 'cool' each Australian government web page is on a 1-10 scale, reading each URL from the list below."


@dataclass(frozen=True, slots=True)
class WorkflowProfile:
    """Server-owned behavior toggles for guided workflow variants."""

    entry_seed: str | None
    coaching: bool
    advisor_checkpoints: bool
    recipe_match: bool
    bookends: bool

    def __post_init__(self) -> None:
        if self.entry_seed is not None and type(self.entry_seed) is not str:
            raise TypeError(f"entry_seed must be str | None, got {type(self.entry_seed).__name__}")
        if self.entry_seed is not None and self.entry_seed.strip() == "":
            raise ValueError("entry_seed must be non-empty when provided")
        for field_name in ("coaching", "advisor_checkpoints", "recipe_match", "bookends"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise TypeError(f"{field_name} must be bool, got {type(value).__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize all direct profile fields for Tier-1 persistence."""

        return {
            "entry_seed": self.entry_seed,
            "coaching": self.coaching,
            "advisor_checkpoints": self.advisor_checkpoints,
            "recipe_match": self.recipe_match,
            "bookends": self.bookends,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowProfile:
        """Deserialize a Tier-1 persisted profile record with strict shape checks."""

        try:
            if type(d) is not dict:
                raise TypeError(f"expected dict, got {type(d).__name__}")
            if set(d) != _PROFILE_KEYS:
                raise KeyError(f"expected keys {_PROFILE_KEYS!r}, got {set(d)!r}")
            return cls(
                entry_seed=d["entry_seed"],
                coaching=d["coaching"],
                advisor_checkpoints=d["advisor_checkpoints"],
                recipe_match=d["recipe_match"],
                bookends=d["bookends"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvariantError(f"WorkflowProfile.from_dict: malformed record {d!r}") from exc


EMPTY_PROFILE = WorkflowProfile(
    entry_seed=None,
    coaching=False,
    advisor_checkpoints=False,
    recipe_match=True,
    bookends=False,
)

TUTORIAL_PROFILE = WorkflowProfile(
    entry_seed=_TUTORIAL_ENTRY_SEED,
    coaching=True,
    advisor_checkpoints=True,
    recipe_match=True,
    bookends=True,
)


def profile_for_kind(kind: WorkflowProfileKind) -> WorkflowProfile:
    """Return the server-owned profile constant for a closed profile kind."""

    if kind is WorkflowProfileKind.LIVE:
        return EMPTY_PROFILE
    if kind is WorkflowProfileKind.TUTORIAL:
        return TUTORIAL_PROFILE
    raise InvariantError(f"profile_for_kind: unhandled profile kind {kind!r}")
