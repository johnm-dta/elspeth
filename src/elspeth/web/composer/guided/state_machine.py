"""Guided-mode state machine: audit-tier (Tier 1) dataclasses.

Frozen dataclasses representing session state at the SDA wizard boundary.
Every field is immutable. Container fields undergo deep-freeze in __post_init__.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §3.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.protocol import GuidedStep, Turn, TurnResponse


class TerminalKind(StrEnum):
    """The outcome classification when a guided session ends."""

    COMPLETED = "completed"
    EXITED_TO_FREEFORM = "exited_to_freeform"
    FAILED = "failed"


class TerminalReason(StrEnum):
    """The reason a guided session ended."""

    USER_CHOSE_FREEFORM = "user_chose_freeform"
    USER_REJECTED_PROPOSAL = "user_rejected_proposal"
    RECIPE_NOT_FOUND = "recipe_not_found"
    SOLVER_ERROR = "solver_error"
    STEP_HANDLER_ERROR = "step_handler_error"


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Marks the end of a guided session.

    Frozen, audit-tier. Persisted in GuidedSession.terminal.
    """

    kind: TerminalKind
    reason: TerminalReason | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """A single turn in the guided-mode dialog.

    Frozen, audit-tier. Appended to GuidedSession.history.
    Carries both the wizard turn (what was emitted) and the user's response.
    """

    turn: Turn
    response: TurnResponse


@dataclass(frozen=True, slots=True)
class SourceResolved:
    """Source plugin state after Step 1.

    Frozen, audit-tier. Stored in GuidedSession.step_1_result.
    """

    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "observed_columns", "sample_rows")


@dataclass(frozen=True, slots=True)
class SinkOutputResolved:
    """A single sink output after Step 2.

    Frozen, audit-tier. Part of SinkResolved.outputs sequence.
    """

    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str  # "fixed" | "flexible" | "observed"

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "required_fields")


@dataclass(frozen=True, slots=True)
class SinkResolved:
    """Sink configuration after Step 2.

    Frozen, audit-tier. Stored in GuidedSession.step_2_result.
    """

    outputs: Sequence[SinkOutputResolved]

    def __post_init__(self) -> None:
        freeze_fields(self, "outputs")


@dataclass(frozen=True, slots=True)
class ChainProposal:
    """A transform chain proposed by Step 3 LLM.

    Frozen, audit-tier. Stored in GuidedSession.step_3_proposal.
    """

    steps: Sequence[Mapping[str, Any]]  # each step: {plugin, options, rationale}
    why: str

    def __post_init__(self) -> None:
        freeze_fields(self, "steps")


@dataclass(frozen=True, slots=True)
class GuidedSession:
    """The guided-mode session state.

    Persisted in CompositionState.guided_session. `terminal` becomes non-None
    when the wizard ends; subsequent freeform turns honour progressive
    disclosure (see §8.2 of the spec).
    """

    step: GuidedStep
    history: tuple[TurnRecord, ...]
    step_1_result: SourceResolved | None
    step_2_result: SinkResolved | None
    step_3_proposal: ChainProposal | None
    terminal: TerminalState | None

    @classmethod
    def initial(cls) -> GuidedSession:
        return cls(
            step=GuidedStep.STEP_1_SOURCE,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
