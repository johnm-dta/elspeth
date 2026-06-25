"""Transient-LLM-failure wrapper for ``solve_step_chat`` (Phase A slice 3).

Lives outside ``routes.py`` for the same reason as
:mod:`elspeth.web.sessions._guided_solve_chain`:
the ``trust_tier.tier_model`` analyzer fingerprints findings by AST sibling
index from the module root. Adding a new top-level function to ``routes.py``
would shift the indices of every allowlisted ``isinstance`` check in that
file and force unrelated allowlist churn. Extracting the helper into its own
module is the structural fix.

Contract: see :func:`solve_step_chat_with_auto_drop`.

Difference from ``solve_chain_with_auto_drop``: chat is a **non-load-bearing**
helper. A failed chain solver blocks the wizard, so its auto-drop marks the
session ``solver_exhausted`` and exits to freeform. Chat failure must NOT
terminate the session — the user can still drive the wizard via widgets.
Instead, the helper returns a synthetic "I'm unavailable" message; the
session is unchanged. The Phase-A.5 plan extends this to proactive openers
with the same contract.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import structlog

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.chat_solver import (
    Step1SourceChatResolution,
    maybe_resolve_step_1_source_chat,
    maybe_resolve_step_2_sink_chat,
    solve_step_chat,
)
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkResolved, SourceResolved

slog = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class StepChatResult:
    """Result of one ``solve_step_chat_with_auto_drop`` call.

    Carries enough metadata for the route handler to build a
    :class:`elspeth.contracts.composer_llm_audit.ComposerChatTurn` audit
    record without re-deriving timing or status.  Phase A surfaces only
    ``assistant_message`` to the wire; the rest stays internal to the
    server-side audit path.
    """

    assistant_message: str
    status: ComposerChatTurnStatus
    latency_ms: int
    error_class: str | None


@dataclass(frozen=True, slots=True)
class Step1SourceChatResult:
    """Guarded result of the Step-1 source resolver branch.

    ``source_resolution`` carries a valid ``resolve_source`` tool result. A
    ``None`` value means the model replied in ordinary prose and the route may
    continue to the normal guided chat path. ``fallback_chat`` carries the
    synthetic-unavailable chat result when the resolver itself failed with a
    transient or malformed model response; in that case the route must not call
    the model again or commit source state from the invalid tool arguments.
    """

    source_resolution: Step1SourceChatResolution | None
    fallback_chat: StepChatResult | None


# Synthetic message returned to the user when the LLM is transiently
# unavailable. Phrase chosen to match the Phase-A.5 opener-drop wording
# (plan line 147) so the frontend renders a consistent "LLM is offline"
# experience whether the failure was on a user message or a step-entry
# opener. The wizard widgets remain functional; chat is best-effort.
_SYNTHETIC_UNAVAILABLE_MESSAGE = "I'm unavailable right now; you can still use the wizard controls."


def _safe_frame_strings(
    exc: BaseException,
    *,
    max_frames: int = 16,
) -> tuple[str, ...]:
    """Frame-only traceback strings; mirror of the routes.py / _guided_solve_chain helper.

    Duplicated here rather than imported from ``routes.py`` to avoid a layer
    cycle (this module is imported by ``routes.py``). The capture rule is
    identical: file path + line + function name only, walking the
    ``__cause__`` chain. No source lines or local-variable reprs that could
    carry Tier-3 data. Cap of 16 frames matches the routes.py default for
    audit-row size predictability.
    """
    frames: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(frames) < max_frames:
        if id(current) in seen:
            break
        seen.add(id(current))
        tb = current.__traceback__
        while tb is not None and len(frames) < max_frames:
            f = tb.tb_frame
            frames.append(f"frame={f.f_code.co_filename}:{tb.tb_lineno}:{f.f_code.co_name}")
            tb = tb.tb_next
        current = current.__cause__
    return tuple(frames)


async def resolve_step_1_source_chat_with_auto_drop(
    *,
    site: str,
    session_id: str,
    user_id: str,
    model: str,
    user_message: str,
    plugin_hint: str | None,
    current_source: SourceResolved | None = None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
) -> Step1SourceChatResult:
    """Wrap Step-1 ``resolve_source`` chat with the guided-chat fallback contract."""
    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
    from litellm.exceptions import (
        BlockedPiiEntityError,
        BudgetExceededError,
        GuardrailInterventionNormalStringError,
        GuardrailRaisedException,
    )

    started = time.perf_counter()
    try:
        source_resolution = await maybe_resolve_step_1_source_chat(
            model=model,
            user_message=user_message,
            plugin_hint=plugin_hint,
            current_source=current_source,
            temperature=temperature,
            seed=seed,
            recorder=recorder,
        )
        return Step1SourceChatResult(
            source_resolution=source_resolution,
            fallback_chat=None,
        )
    except (
        LiteLLMAPIError,
        LiteLLMAuthError,
        LiteLLMBadRequestError,
        BudgetExceededError,
        BlockedPiiEntityError,
        GuardrailRaisedException,
        GuardrailInterventionNormalStringError,
        TimeoutError,
        IndexError,
        AttributeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_1_source_chat_transient_failure",
            session_id=session_id,
            user_id=user_id,
            site=site,
            step=GuidedStep.STEP_1_SOURCE.value,
            exc_class=type(exc).__name__,
            latency_ms=latency_ms,
            frames=_safe_frame_strings(exc),
        )
        return Step1SourceChatResult(
            source_resolution=None,
            fallback_chat=StepChatResult(
                assistant_message=_SYNTHETIC_UNAVAILABLE_MESSAGE,
                status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                latency_ms=latency_ms,
                error_class=type(exc).__name__,
            ),
        )


@dataclass(frozen=True, slots=True)
class Step2SinkChatResult:
    """Outcome of a Step-2 sink chat attempt with auto-drop fall-back.

    ``sink_resolution`` carries a valid ``resolve_sink`` tool result, or
    ``None`` when the model replied in prose (the route continues to the
    advisory guided-chat path). ``assistant_message`` carries the LLM's reply
    that accompanied the tool call (``None`` on prose/failure).
    ``fallback_chat`` carries the synthetic unavailable message on transient
    LLM failure.
    """

    sink_resolution: SinkResolved | None
    assistant_message: str | None
    fallback_chat: StepChatResult | None


async def resolve_step_2_sink_chat_with_auto_drop(
    *,
    site: str,
    session_id: str,
    user_id: str,
    model: str,
    user_message: str,
    current_sink: SinkResolved | None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
) -> Step2SinkChatResult:
    """Wrap Step-2 ``resolve_sink`` chat with the guided-chat fallback contract."""
    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
    from litellm.exceptions import (
        BlockedPiiEntityError,
        BudgetExceededError,
        GuardrailInterventionNormalStringError,
        GuardrailRaisedException,
    )

    started = time.perf_counter()
    try:
        resolved = await maybe_resolve_step_2_sink_chat(
            model=model,
            user_message=user_message,
            current_sink=current_sink,
            temperature=temperature,
            seed=seed,
            recorder=recorder,
        )
        if resolved is None:
            return Step2SinkChatResult(sink_resolution=None, assistant_message=None, fallback_chat=None)
        sink, assistant_message = resolved
        return Step2SinkChatResult(sink_resolution=sink, assistant_message=assistant_message, fallback_chat=None)
    except (
        LiteLLMAPIError,
        LiteLLMAuthError,
        LiteLLMBadRequestError,
        BudgetExceededError,
        BlockedPiiEntityError,
        GuardrailRaisedException,
        GuardrailInterventionNormalStringError,
        TimeoutError,
        IndexError,
        AttributeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_2_sink_chat_transient_failure",
            session_id=session_id,
            user_id=user_id,
            site=site,
            step=GuidedStep.STEP_2_SINK.value,
            exc_class=type(exc).__name__,
            latency_ms=latency_ms,
            frames=_safe_frame_strings(exc),
        )
        return Step2SinkChatResult(
            sink_resolution=None,
            assistant_message=None,
            fallback_chat=StepChatResult(
                assistant_message=_SYNTHETIC_UNAVAILABLE_MESSAGE,
                status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                latency_ms=latency_ms,
                error_class=type(exc).__name__,
            ),
        )


async def solve_step_chat_with_auto_drop(
    *,
    site: str,
    session_id: str,
    user_id: str,
    model: str,
    step: GuidedStep,
    user_message: str,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
) -> StepChatResult:
    """Wrap ``solve_step_chat`` with the synthetic-message-on-transient contract.

    On success, returns the LLM's assistant reply verbatim. On transient
    LLM failure (LiteLLM API/auth/bad-request, timeouts, malformed-response
    shape from ``response.choices[0].message``), returns the synthetic
    unavailable message and emits a slog event with safe frame strings.
    **The session is not modified** — unlike the chain-solver auto-drop,
    chat failure does not terminate guided mode.

    ``InvariantError`` and ``ValueError`` from ``solve_step_chat`` are NOT
    absorbed — they propagate so real programming bugs (the solver
    violating its own contract, or a bad caller payload) continue to raise.

    The transient exception set mirrors the project canonical at
    ``composer/service.py`` / ``_guided_solve_chain.py``:
    ``LiteLLMAPIError``, ``LiteLLMAuthError``, ``LiteLLMBadRequestError``,
    plus the non-``APIError`` operational classes ``BudgetExceededError``,
    ``BlockedPiiEntityError``, ``GuardrailRaisedException`` and
    ``GuardrailInterventionNormalStringError`` (direct ``Exception``
    subclasses — provider budget / content-policy failures that must be
    absorbed, matching ``_explain_run_diagnostics``), and ``TimeoutError``
    for asyncio timeouts. ``IndexError``,
    ``AttributeError``, and ``json.JSONDecodeError`` cover malformed-
    response shape from the LiteLLM response unpacking inside
    ``solve_step_chat`` (empty ``choices``, missing ``message``).

    ``asyncio.CancelledError`` is deliberately NOT in the set — client
    disconnects must propagate, not be silently absorbed.

    ``str(exc)`` is NEVER captured into the slog payload: LiteLLM error
    reprs can carry the request body (system prompt). Only
    ``type(exc).__name__`` and frame strings land in the log; the synthetic
    response is what the client sees.

    Args:
        site: Stable identifier for the call site, used in the slog event
            (e.g. ``"post_guided_chat"``). Required for triage so on-call
            can distinguish which call path failed.
        session_id: Session identifier (string form of the UUID) for slog
            correlation.
        user_id: Requesting user's id; correlates failure to user.
        model: LiteLLM model identifier from ``settings.composer_model``.
        step: User's current wizard step; determines the skill briefing.
        user_message: User's typed chat message; route handler is
            responsible for non-empty / length validation before this call.

    Returns:
        :class:`StepChatResult` carrying the assistant message plus the
        latency / status / error-class metadata the route handler needs
        to build a :class:`ComposerChatTurn` audit record.  The status
        discriminator (``SUCCESS`` vs ``SYNTHETIC_UNAVAILABLE``) is what
        downstream consumers use to tell a real LLM reply apart from the
        synthetic "I'm unavailable" fallback — slice 3 left this gap on
        the wire by design; slice 5 closes it in the audit path.
    """
    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
    from litellm.exceptions import (
        BlockedPiiEntityError,
        BudgetExceededError,
        GuardrailInterventionNormalStringError,
        GuardrailRaisedException,
    )

    started = time.perf_counter()
    try:
        message = await solve_step_chat(
            model=model,
            step=step,
            user_message=user_message,
            temperature=temperature,
            seed=seed,
            recorder=recorder,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return StepChatResult(
            assistant_message=message,
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=latency_ms,
            error_class=None,
        )
    except (
        LiteLLMAPIError,
        LiteLLMAuthError,
        LiteLLMBadRequestError,
        BudgetExceededError,
        BlockedPiiEntityError,
        GuardrailRaisedException,
        GuardrailInterventionNormalStringError,
        TimeoutError,
        IndexError,
        AttributeError,
        json.JSONDecodeError,
    ) as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_chat_transient_failure",
            session_id=session_id,
            user_id=user_id,
            site=site,
            step=step.value,
            exc_class=type(exc).__name__,
            latency_ms=latency_ms,
            frames=_safe_frame_strings(exc),
        )
        return StepChatResult(
            assistant_message=_SYNTHETIC_UNAVAILABLE_MESSAGE,
            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
            latency_ms=latency_ms,
            error_class=type(exc).__name__,
        )
