"""Resolved guided-mode source and sink DTOs.

These dataclasses are a leaf dependency shared by the guided state machine and
recipe matcher. Keeping them out of ``state_machine.py`` prevents recipe matching
from importing the full persisted session model just to read source/sink shape.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.composer.guided.errors import InvariantError


@dataclass(frozen=True, slots=True)
class SourceResolved:
    """Source plugin state after Step 1."""

    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]
    # Source NODE's invalid-row routing: a configured sink name, or "discard".
    # The guided composer sets this when it resolves the source; manual /
    # schema_form-submission paths keep the "discard" default.
    on_validation_failure: str = "discard"

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "observed_columns", "sample_rows")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
            "observed_columns": list(deep_thaw(self.observed_columns)),
            "sample_rows": [dict(deep_thaw(r)) for r in self.sample_rows],
            "on_validation_failure": self.on_validation_failure,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceResolved:
        """Reconstruct from a plain dict. Tier 1 strict: crash on bad data."""
        try:
            return cls(
                plugin=d["plugin"],
                options=d["options"],
                observed_columns=tuple(d["observed_columns"]),
                sample_rows=tuple(dict(r) for r in d["sample_rows"]),
                # Absent on a legacy record (serialized before this field existed):
                # its historically-correct value was the hardcoded "discard" the
                # commit handler always applied, so absence is deterministic, not a
                # defect — default it rather than crashing the session rehydrate.
                on_validation_failure=d.get("on_validation_failure", "discard"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SourceResolved.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkOutputResolved:
    """A single sink output after Step 2."""

    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str  # "fixed" | "flexible" | "observed"

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "required_fields")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {
            "plugin": self.plugin,
            "options": deep_thaw(self.options),
            "required_fields": list(deep_thaw(self.required_fields)),
            "schema_mode": self.schema_mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkOutputResolved:
        """Reconstruct from a plain dict. Tier 1 strict: crash on bad data."""
        try:
            return cls(
                plugin=d["plugin"],
                options=d["options"],
                required_fields=tuple(d["required_fields"]),
                schema_mode=d["schema_mode"],
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SinkOutputResolved.from_dict: malformed record {d!r}") from exc


@dataclass(frozen=True, slots=True)
class SinkResolved:
    """Sink configuration after Step 2."""

    outputs: Sequence[SinkOutputResolved]

    def __post_init__(self) -> None:
        freeze_fields(self, "outputs")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict."""
        return {"outputs": [o.to_dict() for o in self.outputs]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SinkResolved:
        """Reconstruct from a plain dict. Tier 1 strict: crash on bad data."""
        try:
            return cls(outputs=tuple(SinkOutputResolved.from_dict(o) for o in d["outputs"]))
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"SinkResolved.from_dict: malformed record {d!r}") from exc
