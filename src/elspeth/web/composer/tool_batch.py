"""Per-tool-call dispatch pipeline for the composer compose loop.

Extracted verbatim from ComposerServiceImpl._dispatch_tool_batch (service.py)
to take the single largest method out of the god class. The loop body is
UNCHANGED; only its enclosing context is made explicit via the two carriers
below, replacing the prior nested-closure capture of loop-invariant inputs and
loop-carried accumulators.

Behaviour-preservation contract: every terminal arm's
recorder.record(finish_*) / anti_anchor.record_* / llm_messages.append /
budget-class side-effect is identical to the pre-extraction method. Pinned by
tests/unit/web/composer/test_dispatch_arms_characterization.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID

from elspeth.contracts.composer_progress import ComposerProgressSink
from elspeth.web.composer.anti_anchor import AntiAnchorTracker
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.protocol import ComposerPluginCrashError
from elspeth.web.composer.state import CompositionState, ValidationSummary
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.sessions._persist_payload import _ToolOutcome

if TYPE_CHECKING:
    from elspeth.web.composer.service import (
        ComposerServiceImpl,
        _CachedDiscoveryPayload,
        _RuntimePreflightCache,
    )
    from elspeth.web.sessions.protocol import (
        ComposerSessionPreferencesRecord,
        SessionServiceProtocol,
    )


@dataclass(frozen=True, slots=True)
class ToolBatchContext:
    """Loop-invariant inputs to the dispatch loop, built once per batch.

    ``frozen=True`` prevents rebinding the field *references* only. The
    ``discovery_cache`` and ``runtime_preflight_cache`` fields are
    intentionally-mutable shared caches the dispatch loop writes into; they
    are deliberately exempt from the project's ``deep_freeze`` contract
    because deep-freezing them would break the loop's cache-write behaviour.
    No ``__post_init__`` freeze guard is added for that reason.
    """

    service: ComposerServiceImpl
    recorder: BufferingRecorder
    anti_anchor: AntiAnchorTracker
    discovery_cache: dict[str, _CachedDiscoveryPayload]
    runtime_preflight_cache: _RuntimePreflightCache
    session_id: str | None
    user_id: str | None
    user_message_id: str | None
    user_message_content: str | None
    current_state_id: str | None
    actor: str
    initial_version: int
    deadline: float
    progress: ComposerProgressSink | None
    session_scope: str
    turn_sessions_service: SessionServiceProtocol | None
    turn_session_uuid: UUID | None
    turn_preferences: ComposerSessionPreferencesRecord | None


@dataclass(slots=True)
class BatchAccumulator:
    """Loop-carried state that rebinds per iteration."""

    state: CompositionState
    last_validation: ValidationSummary | None
    last_runtime_preflight: ValidationResult | None
    advisor_calls_used: int
    turn_has_mutation: bool = False
    turn_has_discovery: bool = False
    all_cache_hits: bool = True
    proposals_this_turn: int = 0
    mutation_success_observed: bool = False
    plugin_crash: ComposerPluginCrashError | None = None
    plugin_crash_cause: BaseException | None = None
    tool_outcomes: list[_ToolOutcome] = field(default_factory=list)
    decoded_args_by_call_id: dict[str, dict[str, Any]] = field(default_factory=dict)
