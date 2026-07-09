"""Transient-LLM-failure wrapper for ``solve_chain`` (I2 — PR-review finding).

Lives outside ``routes.py`` deliberately. The ``trust_tier.tier_model`` analyzer
fingerprints findings by AST path from the module root, so adding a new
top-level function to ``routes.py`` between ``_dispatch_guided_respond`` and
``create_session_router`` would shift the AST sibling index of every existing
allowlisted ``isinstance`` check in those functions and invalidate the
fingerprints. Extracting this helper into its own module is the only way to
introduce the wrap without churning a dozen unrelated allowlist entries.

Contract: see :func:`solve_chain_with_auto_drop`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import structlog

from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.audit import emit_dropped_to_freeform
from elspeth.web.composer.guided.chain_solver import ChainSolverResponseShapeError, solve_chain
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    GuidedSession,
    SinkResolved,
    SourceResolved,
    TerminalReason,
    mark_solver_exhausted,
)
from elspeth.web.composer.state import CompositionState

slog = structlog.get_logger()


def _safe_frame_strings(
    exc: BaseException,
    *,
    max_frames: int = 16,
) -> tuple[str, ...]:
    """Mirror of ``routes._safe_frame_strings`` — frame-only traceback strings.

    Duplicated here rather than imported from ``routes.py`` to avoid a layer
    cycle (this module is imported by ``routes.py``). The capture rule is
    identical: file path + line + function name only, walking the
    ``__cause__`` chain, no source lines or local-variable reprs that could
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


async def solve_chain_with_auto_drop(
    *,
    site: str,
    session: GuidedSession,
    session_id: str,
    user_id: str,
    composition_version: int,
    recorder: BufferingRecorder,
    model: str,
    source: SourceResolved,
    sink: SinkResolved,
    intent: str | None = None,
    repair_context: str | None = None,
    temperature: float | None,
    seed: int | None,
    state: CompositionState | None = None,
    catalog: CatalogService | None = None,
    secret_service: WebSecretResolver | None = None,
    max_discovery_iters: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[ChainProposal | None, GuidedSession]:
    """Wrap ``solve_chain`` with the auto-drop-on-transient contract (I2).

    On success, returns ``(proposal, session)`` unchanged — the caller proceeds
    with the proposal. On transient LLM failure (LiteLLM API/auth/bad-request,
    non-APIError LiteLLM operational/policy failures, the LiteLLM OpenAIError
    base, timeouts, malformed-response shape from ``response.choices[0].message``),
    marks the session ``solver_exhausted`` via :func:`mark_solver_exhausted`,
    fans the ``guided_dropped_to_freeform`` directive to the audit emitter,
    and returns ``(None, new_terminal_session)`` so the caller can short-circuit
    with ``return state, new_session, None``.

    ``InvariantError`` and ``ValueError`` are NOT absorbed — they propagate so
    real programming bugs (the chain solver violating its own contract, or a
    bad client payload) continue to raise to the outer dispatch handlers.

    The transient exception set mirrors the project canonical at
    ``composer/service.py:3173-3268`` (the advisor invocation path):
    LiteLLM ``APIError``, ``AuthenticationError``, ``BadRequestError``,
    ``BudgetExceededError``, ``BlockedPiiEntityError``,
    ``GuardrailRaisedException``, ``GuardrailInterventionNormalStringError``,
    and ``OpenAIError`` plus ``TimeoutError`` for asyncio timeouts.
    ``IndexError``, ``AttributeError``, and ``json.JSONDecodeError`` cover
    malformed-response shape from ``chain_solver._extract_tool_call`` (empty
    ``choices``, missing ``message``, invalid tool-call JSON).
    ``ChainSolverResponseShapeError`` covers shape failures one layer deeper:
    wrong tool name, wrong ``turn_type``, missing or malformed
    ``payload.steps`` / ``payload.why``.  All such failures are
    external-system (LLM) shape misbehaviour, not server bugs -- routing
    them through the auto-drop path matches what the codebase already does
    for sibling shape failures at the LiteLLM-response level.

    ``asyncio.CancelledError`` is deliberately NOT in the set — client
    disconnects must propagate, not be silently absorbed as a "drop".

    ``str(exc)`` is NEVER captured into the audit payload: LiteLLM error
    reprs can carry the request body (which embeds the system prompt and
    the GUIDED CONTEXT block, including Tier-3 ``sample_rows``). Only
    ``type(exc).__name__`` lands in ``validation_result.errors[].message``;
    diagnostic frames go to slog (which is operator-scoped, not
    Landscape-archived).

    The ``recorder`` is forwarded into ``solve_chain`` so each LLM invocation
    records exactly one :class:`ComposerLLMCall` row regardless of outcome
    (success, transient failure absorbed by this wrapper, or escaping
    exception). The wrapper's own audit emission (``guided_dropped_to_freeform``
    on the transient-failure branch) lives on the :class:`ComposerToolInvocation`
    channel — a separate audit primitive from the LLM-call row. Both channels
    must drain into persistence at the route handler exit; that drain currently
    fires ``_persist_tool_invocations`` AND ``_persist_llm_calls`` in the
    guided dispatch ``finally`` blocks.

    Args:
        site: Stable identifier for the call site, used in the slog event
            (e.g. ``"step_2_sink_initial_solve"``). Required for triage so
            on-call can distinguish which solve attempt failed.
        composition_version: ``state.version`` at the dispatch site; passed
            verbatim into ``emit_dropped_to_freeform`` so the audit row
            stamps the version the operator was working against.
        recorder: Receives both the :class:`ComposerLLMCall` from
            ``solve_chain`` (one per invocation) and the
            ``guided_dropped_to_freeform`` :class:`ComposerToolInvocation` on
            the transient-failure branch (one per drop).
    """
    from litellm.exceptions import (
        APIError,
        AuthenticationError,
        BadRequestError,
        BlockedPiiEntityError,
        BudgetExceededError,
        GuardrailInterventionNormalStringError,
        GuardrailRaisedException,
        OpenAIError,
    )

    try:
        proposal = await solve_chain(
            model=model,
            source=source,
            sink=sink,
            intent=intent,
            repair_context=repair_context,
            recorder=recorder,
            temperature=temperature,
            seed=seed,
            state=state,
            catalog=catalog,
            secret_service=secret_service,
            user_id=user_id,
            max_discovery_iters=max_discovery_iters,
            timeout_seconds=timeout_seconds,
        )
        return proposal, session
    except (
        APIError,
        AuthenticationError,
        BadRequestError,
        BudgetExceededError,
        BlockedPiiEntityError,
        GuardrailRaisedException,
        GuardrailInterventionNormalStringError,
        OpenAIError,
        TimeoutError,
        IndexError,
        AttributeError,
        json.JSONDecodeError,
        ChainSolverResponseShapeError,
    ) as exc:
        slog.error(
            "guided.chain_solver_transient_failure",
            session_id=session_id,
            user_id=user_id,
            site=site,
            exc_class=type(exc).__name__,
            frames=_safe_frame_strings(exc),
        )
        # Record the wrapper class name in the structured ``error_class`` field
        # (NOT free-form ``message``): the guided audit emitter sanitises
        # validation_result by allowlist and drops ``message``, because that
        # channel also carries repair-validation text that can leak paths /
        # raw exception strings. ``type(exc).__name__`` is a safe class name and
        # survives the allowlist, so the auto-drop reason stays auditable.
        validation_result_payload: Mapping[str, Any] = {
            "is_valid": False,
            "errors": [
                {"error_class": type(exc).__name__},
            ],
        }
        new_session, _terminal, directives = mark_solver_exhausted(
            session=session,
            validation_result=validation_result_payload,
        )
        for directive in directives:
            if directive.tool_name == "guided_dropped_to_freeform":
                args = dict(directive.arguments)
                emit_dropped_to_freeform(
                    recorder,
                    prev=GuidedStep(args["prev_step"]),
                    drop_reason=TerminalReason(args["drop_reason"]),
                    validation_result=args["validation_result"],
                    composition_version=composition_version,
                    actor=user_id,
                )
        return None, new_session
