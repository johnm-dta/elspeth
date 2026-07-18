"""Transient-LLM-failure wrapper for ``solve_step_chat`` (Phase A slice 3).

Contract: see :func:`solve_step_chat_with_auto_drop`.

Chat failure must not terminate the session because the user can still drive
the wizard via widgets. The helper returns a synthetic unavailable message and
leaves the session unchanged.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import structlog

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.contracts.composer_progress import ComposerProgressSink
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.chat_solver import (
    AssistantScaffoldLeakError,
    Step1SourceChatResolution,
    maybe_resolve_step_1_source_chat,
    maybe_resolve_step_2_sink_chat,
    solve_step_chat,
)
from elspeth.web.composer.guided.errors import GuidedSolverResponseShapeError
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkResolved, SourceResolved
from elspeth.web.composer.state import CompositionState
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot

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
    ``None`` value means the model replied in ordinary prose (``prose_chat``)
    or the resolver failed (``fallback_chat``); either way the route must not
    call the model again or commit source state from invalid tool arguments.

    ``fallback_chat`` carries the fallback chat result when the resolver
    itself failed — a transient or malformed model response (unavailable
    copy) or a scaffold leak in the tool's own ``assistant_message`` argument
    (honest quality-check copy, distinguished via ``error_class``).

    ``prose_chat`` carries the model's own SUCCESS reply when it declined the
    tool call and answered in prose instead — captured directly from the
    resolve-equipped call so the route never needs a second, tool-less call
    to obtain an answer to show the user. Defaults to ``None`` for
    construction-site compatibility with callers outside this module that do
    not know about the salvage path; every call site inside this module sets
    it explicitly.
    """

    source_resolution: Step1SourceChatResolution | None
    fallback_chat: StepChatResult | None
    prose_chat: StepChatResult | None = None


# Synthetic message returned to the user when the LLM is transiently
# unavailable. Phrase chosen to match the Phase-A.5 opener-drop wording
# (plan line 147) so the frontend renders a consistent "LLM is offline"
# experience whether the failure was on a user message or a step-entry
# opener. The wizard widgets remain functional; chat is best-effort.
_SYNTHETIC_UNAVAILABLE_MESSAGE = "I'm unavailable right now; you can still use the wizard controls."

# Message returned when the strict source/sink commit seam rejects a resolved
# Step-1/Step-2 chat action. The service and model are not unavailable; the
# proposed configuration simply was not applied, and the state is unchanged.
_COMMIT_REJECTED_MESSAGE = (
    "I couldn't apply that configuration, so I didn't change your pipeline. "
    "Review the wizard fields and try again, or keep going with the wizard controls."
)

# Message returned when ``AssistantScaffoldLeakError`` rejects a reply.
# Deliberately NOT the unavailability copy above: the service is fine — a
# quality guard rejected THIS reply — so the message must not claim
# unavailability (that miscalibration was the C-1 finding). Honest and
# retryable: the Send is safe to try again, and the wizard controls remain a
# fallback. ``status`` stays ``SYNTHETIC_UNAVAILABLE`` (no dedicated audit
# enum member for this cause); ``error_class`` carries the distinction
# (``AssistantScaffoldLeakError`` vs a transient exception class) for anyone
# reading the audit trail.
_SCAFFOLD_LEAK_MESSAGE = (
    "That reply didn't pass a quality check, so it wasn't shown — nothing is "
    "wrong with the service. Try sending your message again, or keep going "
    "with the wizard controls."
)


def _safe_frame_strings(
    exc: BaseException,
    *,
    max_frames: int = 16,
) -> tuple[str, ...]:
    """Frame-only traceback strings matching the guided route diagnostic boundary.

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
    timeout_seconds: float,
    context_block: str | None = None,
) -> Step1SourceChatResult:
    """Wrap Step-1 ``resolve_source`` chat with the guided-chat fallback contract.

    ``context_block`` threads straight through to
    :func:`maybe_resolve_step_1_source_chat` so a declined-to-prose reply
    (``prose_chat`` below) is grounded in the same "current build" context a
    second, tool-less call would otherwise have supplied.
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
        outcome = await maybe_resolve_step_1_source_chat(
            model=model,
            user_message=user_message,
            plugin_hint=plugin_hint,
            current_source=current_source,
            temperature=temperature,
            seed=seed,
            recorder=recorder,
            timeout_seconds=timeout_seconds,
            context_block=context_block,
        )
        prose_chat: StepChatResult | None = None
        if outcome.prose_reply is not None:
            # The resolve-equipped call declined the tool and answered in
            # prose instead — a genuine SUCCESS, not a fallback. Building the
            # StepChatResult here (not in the route) keeps latency/status
            # computation colocated with the other outcomes of this call.
            prose_chat = StepChatResult(
                assistant_message=outcome.prose_reply,
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_class=None,
            )
        return Step1SourceChatResult(
            source_resolution=outcome.resolution,
            fallback_chat=None,
            prose_chat=prose_chat,
        )
    except AssistantScaffoldLeakError as exc:
        # Distinct branch from the transient set below: the model can leak
        # scaffolding INTO resolve_source's own assistant_message argument
        # (observed live, tutorial resolve_source path), not only in the
        # tool-less advisory call. Same honest-copy outcome as
        # ``solve_step_chat_with_auto_drop``'s dedicated branch — a scaffold
        # leak is a model register violation, not provider weather, and must
        # NOT be folded into the generic ValueError catch below (which would
        # mislabel it "unavailable").
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_1_source_chat_scaffold_leak",
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
                assistant_message=_SCAFFOLD_LEAK_MESSAGE,
                status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                latency_ms=latency_ms,
                error_class=type(exc).__name__,
            ),
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
    ``None`` when the model replied in prose (``prose_chat``) or the resolver
    failed (``fallback_chat``). ``assistant_message`` carries the LLM's reply
    that accompanied the tool call (``None`` unless ``sink_resolution`` is set).

    ``fallback_chat`` carries the unavailable-copy message on transient LLM
    failure, or the honest quality-check copy on a scaffold leak in the
    tool's own ``assistant_message`` argument (distinguished via
    ``error_class``).

    ``prose_chat`` carries the model's own SUCCESS reply when it declined the
    tool call and answered in prose instead — captured directly from the
    resolve-equipped call so the route never needs a second, tool-less call
    to obtain an answer to show the user. Defaults to ``None`` for
    construction-site compatibility with callers outside this module that do
    not know about the salvage path; every call site inside this module sets
    it explicitly.
    """

    sink_resolution: SinkResolved | None
    assistant_message: str | None
    fallback_chat: StepChatResult | None
    prose_chat: StepChatResult | None = None


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
    state: CompositionState | None = None,
    catalog: PolicyCatalogView | None = None,
    plugin_snapshot: PluginAvailabilitySnapshot | None = None,
    secret_service: WebSecretResolver | None = None,
    max_discovery_iters: int | None = None,
    timeout_seconds: float,
    context_block: str | None = None,
    progress: ComposerProgressSink | None = None,
) -> Step2SinkChatResult:
    """Wrap Step-2 ``resolve_sink`` chat with the guided-chat fallback contract.

    ``state`` + ``catalog`` (and optional ``secret_service``) activate the
    sink discovery-tool loop in :func:`maybe_resolve_step_2_sink_chat`; the
    route always threads them so the composer model can ``list_sinks`` /
    ``get_plugin_schema`` before resolving. ``max_discovery_iters`` bounds the
    loop (the route passes ``settings.composer_max_discovery_turns``); ``None``
    defers to the solver's own default. ``context_block`` threads straight
    through so a declined-to-prose reply (``prose_chat`` below) is grounded in
    the same "current build" context a second, tool-less call would
    otherwise have supplied.
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
        outcome = await maybe_resolve_step_2_sink_chat(
            model=model,
            user_message=user_message,
            current_sink=current_sink,
            temperature=temperature,
            seed=seed,
            recorder=recorder,
            state=state,
            catalog=catalog,
            plugin_snapshot=plugin_snapshot,
            secret_service=secret_service,
            user_id=user_id,
            max_discovery_iters=max_discovery_iters,
            timeout_seconds=timeout_seconds,
            context_block=context_block,
            progress=progress,
        )
        if outcome.sink is not None:
            return Step2SinkChatResult(
                sink_resolution=outcome.sink,
                assistant_message=outcome.assistant_message,
                fallback_chat=None,
            )
        prose_chat: StepChatResult | None = None
        if outcome.assistant_message is not None:
            # The resolve-equipped call declined the tool and answered in
            # prose instead — a genuine SUCCESS, not a fallback. Building the
            # StepChatResult here (not in the route) keeps latency/status
            # computation colocated with the other outcomes of this call.
            prose_chat = StepChatResult(
                assistant_message=outcome.assistant_message,
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=int((time.perf_counter() - started) * 1000),
                error_class=None,
            )
        return Step2SinkChatResult(
            sink_resolution=None,
            assistant_message=None,
            fallback_chat=None,
            prose_chat=prose_chat,
        )
    except AssistantScaffoldLeakError as exc:
        # Distinct branch from the transient set below: the model can leak
        # scaffolding INTO resolve_sink's own assistant_message argument, not
        # only in the tool-less advisory call. Same honest-copy outcome as
        # ``solve_step_chat_with_auto_drop``'s dedicated branch — a scaffold
        # leak is a model register violation, not provider weather, and must
        # NOT be folded into the generic ValueError catch below (which would
        # mislabel it "unavailable").
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_2_sink_chat_scaffold_leak",
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
                assistant_message=_SCAFFOLD_LEAK_MESSAGE,
                status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                latency_ms=latency_ms,
                error_class=type(exc).__name__,
            ),
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
        # A malformed discovery-tool dispatch deep in the sink loop raises this
        # (via ``_execute_discovery_call``); absorb it into the advisory fallback
        # instead of letting malformed model output escape as a 500.
        GuidedSolverResponseShapeError,
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
    timeout_seconds: float,
    context_block: str | None = None,
) -> StepChatResult:
    """Wrap ``solve_step_chat`` with the synthetic-message-on-transient contract.

    On success, returns the LLM's assistant reply verbatim. On transient
    LLM failure (LiteLLM API/auth/bad-request, timeouts, malformed-response
    shape from ``response.choices[0].message``), returns the synthetic
    unavailable message and emits a slog event with safe frame strings.
    **The session is not modified**: advisory chat failure does not terminate
    guided mode.

    ``InvariantError`` and bare ``ValueError`` from ``solve_step_chat`` are
    NOT absorbed — they propagate so real programming bugs (the solver
    violating its own contract, or a bad caller payload) continue to raise.
    The one ``ValueError`` subclass that IS absorbed is
    ``AssistantScaffoldLeakError``: the model writing tool-call scaffolding
    into its user-facing reply is model behaviour (Tier 3), not a caller
    bug — it maps to the honest quality-check-rejection message
    (``_SCAFFOLD_LEAK_MESSAGE``, NOT the unavailability copy — the service is
    fine, a guard rejected this one reply) so the user's Send stays retryable
    instead of 500ing (observed live 2026-07-03, guided step_1 advisory
    reply).

    The transient exception set mirrors the project canonical in
    ``composer/service.py``:
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
        downstream consumers use to tell a real LLM reply apart from a
        fallback message — slice 3 left this gap on the wire by design;
        slice 5 closes it in the audit path. ``error_class`` distinguishes
        the two fallback causes (``AssistantScaffoldLeakError`` vs a
        transient exception class) since ``status`` alone does not.
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
            timeout_seconds=timeout_seconds,
            context_block=context_block,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return StepChatResult(
            assistant_message=message,
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=latency_ms,
            error_class=None,
        )
    except AssistantScaffoldLeakError as exc:
        # Distinct slog event from the transient set: a scaffold leak is a
        # model register violation worth counting separately in triage, not
        # provider weather. Honest outcome — the guard rejected THIS reply,
        # the service is not down — not the unavailability copy; Send stays
        # retryable either way.
        latency_ms = int((time.perf_counter() - started) * 1000)
        slog.error(
            "guided.step_chat_scaffold_leak",
            session_id=session_id,
            user_id=user_id,
            site=site,
            step=step.value,
            exc_class=type(exc).__name__,
            latency_ms=latency_ms,
            frames=_safe_frame_strings(exc),
        )
        return StepChatResult(
            assistant_message=_SCAFFOLD_LEAK_MESSAGE,
            status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
            latency_ms=latency_ms,
            error_class=type(exc).__name__,
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
