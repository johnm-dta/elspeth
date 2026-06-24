"""Local minimal STEP_4_WIRE fixtures for the P5.6 sign-off gate tests.

The brief's preferred helper ``make_wire_ready_session_and_state`` is owned by
P3; until P3 exports it this module provides a minimal local version that builds
a STEP_4_WIRE-positioned :class:`GuidedSession` (with the caller-supplied
``profile``) plus a single-source / single-sink VALID :class:`CompositionState`.
"""

from __future__ import annotations

import dataclasses

from elspeth.web.composer.guided.profile import WorkflowProfile
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import GuidedSession, TurnRecord
from elspeth.web.composer.state import (
    CompositionState,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _valid_state() -> CompositionState:
    """A single-source / single-sink composition that ``validate().is_valid``."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="out",
            options={"path": "in.csv"},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="out",
                plugin="csv",
                options={"path": "out.csv"},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=2,
    )


def make_wire_ready_session_and_state(
    *,
    profile: WorkflowProfile,
) -> tuple[GuidedSession, CompositionState]:
    """Build a STEP_4_WIRE-positioned session + a valid composition state.

    The session history carries a single server-emitted CONFIRM_WIRING
    ``TurnRecord`` so the dispatcher sees a confirm answer for the wire turn.
    """
    state = _valid_state()
    wire_record = TurnRecord(
        step=GuidedStep.STEP_4_WIRE,
        turn_type=TurnType.CONFIRM_WIRING,
        payload_hash="wire-payload-hash",
        response_hash=None,
        emitter="server",
    )
    session = dataclasses.replace(
        GuidedSession.initial(),
        step=GuidedStep.STEP_4_WIRE,
        history=(wire_record,),
        profile=profile,
    )
    return session, state
