"""Guided-mode state-machine data: GuidedSession, TerminalState, TurnRecord.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §5.

Trust tier: Tier 1 (audit). Coercion forbidden — every field crashes on
malformed input. The freeze_fields contract applies because these structures
are persisted and re-read across the audit trail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType


class TerminalKind(StrEnum):
    COMPLETED = "completed"
    EXITED_TO_FREEFORM = "exited_to_freeform"


class TerminalReason(StrEnum):
    USER_PRESSED_EXIT = "user_pressed_exit"
    PROTOCOL_VIOLATION = "protocol_violation"
    SOLVER_EXHAUSTED = "solver_exhausted"


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Outcome of a guided session.

    `reason` is None when `kind == COMPLETED`; required when
    `kind == EXITED_TO_FREEFORM`. `pipeline_yaml` is set only on COMPLETED.
    Callers must construct consistently — invariants enforced by step_advance().
    """

    kind: TerminalKind
    reason: TerminalReason | None
    pipeline_yaml: str | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One emitted turn + its (optional) user response, recorded for audit."""

    step: GuidedStep
    turn_type: TurnType
    payload_hash: str
    response_hash: str | None
    emitter: str  # "server" | "llm"


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
