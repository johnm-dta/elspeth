"""No-tool-response finalizer for the composer compose loop.

Extracted verbatim from ComposerServiceImpl._finalize_no_tool_response
(service.py) to reduce the god-class surface. The logic is UNCHANGED;
the enclosing self reference is made explicit via the ``service`` parameter.

Behaviour-preservation contract: all augmentation exit shapes and the
state-claim grounding correction path are identical to the pre-extraction
method. Pinned by the compose-loop integration tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.contracts.composer_audit import ComposerToolInvocation
from elspeth.contracts.composer_llm_audit import ComposerLLMCall
from elspeth.web.composer.discovery_cache import RuntimePreflightCache as _RuntimePreflightCache
from elspeth.web.composer.no_tool_policy import (
    blocking_result_from_tool_invocations as _blocking_result_from_tool_invocations,
)
from elspeth.web.composer.no_tool_policy import (
    compose_empty_state_message as _compose_empty_state_message,
)
from elspeth.web.composer.no_tool_policy import (
    compose_preflight_failure_message as _compose_preflight_failure_message,
)
from elspeth.web.composer.no_tool_policy import (
    enforce_augmentation_prefix_invariant as _enforce_augmentation_prefix_invariant,
)
from elspeth.web.composer.no_tool_policy import (
    is_pending_interpretation_handoff as _is_pending_interpretation_handoff,
)
from elspeth.web.composer.no_tool_policy import (
    last_mutation_was_pending_proposal as _last_mutation_was_pending_proposal,
)
from elspeth.web.composer.no_tool_policy import (
    no_mutation_empty_state_validation as _no_mutation_empty_state_validation,
)
from elspeth.web.composer.no_tool_policy import (
    state_is_structurally_empty as _state_is_structurally_empty,
)
from elspeth.web.composer.no_tool_policy import (
    user_request_expects_pipeline_mutation as _user_request_expects_pipeline_mutation,
)
from elspeth.web.composer.protocol import ComposerResult
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.state_claim_grounding import (
    check_state_claim_grounding,
    compose_grounded_message,
)
from elspeth.web.execution.schemas import ValidationResult
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl


async def finalize_no_tool_response(
    service: ComposerServiceImpl,
    *,
    content: str,
    state: CompositionState,
    initial_version: int,
    user_id: str | None,
    session_id: str | None,
    last_runtime_preflight: ValidationResult | None,
    runtime_preflight_cache: _RuntimePreflightCache,
    session_scope: str,
    user_message: str = "",
    mutation_success_seen: bool = False,
    tool_invocations: tuple[ComposerToolInvocation, ...] = (),
    llm_calls: tuple[ComposerLLMCall, ...] = (),
    plugin_snapshot: PluginAvailabilitySnapshot | None = None,
) -> ComposerResult:
    """Apply the deterministic final-gate check and build a ComposerResult.

    Four augmentation exit shapes are produced (the
    ``routes._composer_history_content`` discriminator depends on the
    ``content`` / ``raw_assistant_content`` relationship below):

    1. **No-mutation empty-state augmentation** — user asked for a
       build-style action, no successful mutation has been seen this
       turn, and the state is structurally empty. The model's prose
       is passed through verbatim with an operator-facing suffix
       appended (concrete blocker if a tool failed). ``content``
       starts with ``raw_assistant_content`` so the LLM keeps seeing
       its own prose on subsequent turns.
    2. **Preflight-invalid empty-state augmentation** — preflight
       is invalid AND the state is structurally empty. Same
       augmentation shape as (1); ``raw_assistant_content`` carries
       the unaugmented prose so the LLM context is unaffected by the
       operator-facing suffix.
    3. **Preflight-invalid non-empty-state augmentation** —
       preflight is invalid and the state is non-empty. Panel-evals
       evidence (issue elspeth-9cfbad6901) showed the model's prose
       in this case is typically substantive disclosure rather than
       a false completion claim — the model names removed nodes,
       chosen operational semantics, and its own diagnosis. The
       prose is preserved verbatim with a system-attributed suffix
       naming the validator's specific objection appended;
       ``raw_assistant_content`` carries the unaugmented prose so
       the LLM-context channel sees the model's own prose on
       subsequent turns. Self-correction continues to land via the
       model's next ``preview_pipeline`` call.

    When preflight is valid (or skipped because state did not
    change and ``last_runtime_preflight`` is ``None``) the
    ``check_state_claim_grounding`` content check runs to detect
    un-grounded prose (state-field contradictions or unmotivated
    action claims — see issues elspeth-c028f7d186 and
    elspeth-905fe2a3d8). When the check returns violations, the
    response is augmented with an ``[ELSPETH-SYSTEM]`` correction
    suffix and ``raw_assistant_content`` carries the unaugmented
    prose. Otherwise the response passes through unchanged and
    ``raw_assistant_content`` is left unset.

    Gate logic (no regex on natural-language text governs the
    routing-shape decision):
    - If ``state.version > initial_version`` (state changed this turn),
      run ``_cached_runtime_preflight`` for the current state.
    - Otherwise, reuse ``last_runtime_preflight`` from the most recent
      ``preview_pipeline`` call (may be ``None``).
    - Whichever of the two paths populated ``runtime_result``, the
      state-claim grounding check runs after preflight-invalid
      branches are dispatched. The grounding check's regex
      patterns are content-grounding (additive correction suffix),
      not routing-shape decisions.

    Args:
        content: The model's assistant prose for this turn.
        state: The post-tool composition state. ``state.version``
            versus ``initial_version`` decides whether preflight
            must re-run.
        initial_version: The composition version at turn start.
        user_id: Authenticated user identity for cache scoping.
        last_runtime_preflight: Most recent preflight outcome from
            ``preview_pipeline`` calls this turn; ``None`` if none.
        runtime_preflight_cache: Per-turn cache to avoid redundant
            preflight invocations on identical state.
        session_scope: Scope identifier for cache + telemetry.
        user_message: The user's message that triggered this turn.
            Used to detect "build-style" requests for the
            no-mutation empty-state augmentation path.
        mutation_success_seen: Whether any mutating tool call
            succeeded this turn. Suppresses the no-mutation
            augmentation path.
        tool_invocations: Tool calls made this turn; the most
            recent failure feeds the operator-facing blocker
            suffix.
        llm_calls: LLM call audit rows for this turn.

    Unexpected preflight exceptions (anything other than a
    ``RuntimePreflightFailure`` caught inside ``_cached_runtime_preflight``)
    are already converted to ``ComposerRuntimePreflightError`` with
    partial-state preservation by the coordinator's exception handling
    path — they are not caught here.
    """
    if (
        _user_request_expects_pipeline_mutation(user_message)
        and not mutation_success_seen
        and _state_is_structurally_empty(state)
        and not _last_mutation_was_pending_proposal(tool_invocations)
    ):
        # No-mutation empty-state augmentation. The model produced
        # honest diagnostic prose about what it tried and what blocked
        # convergence (audit-DB inspection across 2026-05-08 panel-cohort
        # cells confirms this). Pass the prose through and append a
        # system-attributed suffix carrying the concrete blocker — the
        # earlier full-replacement behavior hid the model's actual
        # output from both the user and (via routes._composer_history_content)
        # from the model itself on subsequent turns
        # (cf. elspeth-861b0c58f5).
        #
        # Exception: APPROVAL_REQUIRED proposals are not failures. Under
        # explicit_approve trust mode the build SUCCEEDED — the work is
        # queued for human approval and the state will advance on accept.
        # The model's prose ("approval pending state, not applied") is
        # already truthful; injecting "[ELSPETH-SYSTEM] pipeline is still
        # empty" over it corrupts both operator framing and (via the
        # synthesized suffix being re-read on subsequent turns) the
        # model's state model.
        blocker = _blocking_result_from_tool_invocations(tool_invocations)
        empty_state_runtime_result = _no_mutation_empty_state_validation(blocker)
        augmented_message = _compose_empty_state_message(content, blocker=blocker)
        _enforce_augmentation_prefix_invariant(
            branch="no_mutation_empty_state_augmentation",
            content=content,
            augmented=augmented_message,
        )
        return ComposerResult(
            message=augmented_message,
            state=state,
            runtime_preflight=empty_state_runtime_result,
            raw_assistant_content=content,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
        )

    runtime_result: ValidationResult | None = last_runtime_preflight
    if state.version > initial_version:
        runtime_result = await service._cached_runtime_preflight(
            state,
            user_id=user_id,
            session_id=session_id,
            cache=runtime_preflight_cache,
            initial_version=initial_version,
            session_scope=session_scope,
            llm_calls=llm_calls,
            plugin_snapshot=plugin_snapshot,
        )

    if runtime_result is not None and not runtime_result.is_valid and not _is_pending_interpretation_handoff(runtime_result):
        # Two finalize shapes for invalid preflight, dispatched on
        # state structure. Both augment — the difference is suffix
        # wording (issue elspeth-9cfbad6901 unified the policy after
        # the original replacement-on-non-empty-state branch was
        # found to discard substantive model disclosure):
        #
        # 1. Preflight-invalid empty-state augmentation: state is
        #    structurally empty. The model produced honest diagnostic
        #    prose about what it tried and what blocked convergence.
        #    Pass the prose through and append a system suffix asking
        #    the operator to refine or retry. raw_assistant_content
        #    carries the unaugmented prose so
        #    routes._composer_history_content replays it to the LLM
        #    on subsequent turns without the synthetic system text
        #    (cf. elspeth-861b0c58f5 — the original synthesizer-
        #    replaces-prose behavior corrupted both user view and
        #    LLM context).
        #
        # 2. Preflight-invalid non-empty-state augmentation: state
        #    has been populated AND preflight failed. Panel-evals
        #    evidence (fork_coalesce__p4_adversarial_engineer,
        #    boolean_routing__p3_marketingops; see issue
        #    elspeth-9cfbad6901) shows the model's prose in this
        #    case is typically substantive disclosure — what was
        #    attempted, removed nodes, chosen semantics — rather
        #    than a false completion claim. Pass the prose through
        #    and append a system suffix naming the validator's
        #    specific objection. The model's next preview_pipeline
        #    call surfaces the failure on the self-correction loop;
        #    the operator-facing suffix is stripped from LLM history
        #    so the LLM sees only its own prose.
        if _state_is_structurally_empty(state):
            augmented_message = _compose_empty_state_message(content)
            _enforce_augmentation_prefix_invariant(
                branch="preflight_invalid_empty_state_augmentation",
                content=content,
                augmented=augmented_message,
            )
            return ComposerResult(
                message=augmented_message,
                state=state,
                runtime_preflight=runtime_result,
                raw_assistant_content=content,
                tool_invocations=tool_invocations,
                llm_calls=llm_calls,
            )
        augmented_message = _compose_preflight_failure_message(content, runtime_result=runtime_result)
        _enforce_augmentation_prefix_invariant(
            branch="preflight_invalid_non_empty_state_augmentation",
            content=content,
            augmented=augmented_message,
        )
        return ComposerResult(
            message=augmented_message,
            state=state,
            runtime_preflight=runtime_result,
            raw_assistant_content=content,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
        )

    # State-claim grounding correction (Path 3 of issue elspeth-c028f7d186,
    # widened by issue elspeth-905fe2a3d8 to also catch verbal
    # acknowledgement without state mutation). The check runs in both
    # of the remaining cases:
    #
    #   - runtime_result is_valid: state has reached passing preflight
    #     with non-empty contents (T4 forward-contradiction case
    #     covered here).
    #   - runtime_result is None: state did not change AND no
    #     preview_pipeline was called this turn. Without grounding,
    #     the prior version of this function bare-passed-through —
    #     the panel-evals T5 case (model said "I just fixed it" with
    #     state unchanged from T4) and the cells #2/#4 cases (model
    #     said "you're right, I'll change that" with no mutation
    #     tool call) both landed in this hole. Running the grounding
    #     check here is what closes both bugs.
    #
    # The model's prose may contradict state — claiming a field has
    # its old value when the mutation already landed (forward
    # contradiction), claiming a fresh action when state was
    # unchanged (backward contradiction), or agreeing with the user
    # without acting (verbal acknowledgement). Detect and append an
    # [ELSPETH-SYSTEM] correction so amateur personas (compliance,
    # marketing-ops) cannot read confidently-wrong prose as
    # authoritative.
    #
    # Augmentation shape preserves the model's prose verbatim per the
    # ``_enforce_augmentation_prefix_invariant`` contract; the
    # raw_assistant_content field carries the original prose for the
    # LLM history-replay path (consistent with the empty-state
    # augmentation branches above). The natural-language regex used
    # by ``check_state_claim_grounding`` is content-grounding, not
    # gate routing — the gate decision (state non-empty, preflight
    # not failed) was already taken above without consulting prose.
    grounding_violations = check_state_claim_grounding(
        prose=content,
        state=state,
        mutation_success_seen=mutation_success_seen,
        state_changed=state.version > initial_version,
    )
    if grounding_violations:
        augmented_message = compose_grounded_message(
            prose=content,
            violations=grounding_violations,
        )
        _enforce_augmentation_prefix_invariant(
            branch="state_claim_grounding_correction",
            content=content,
            augmented=augmented_message,
        )
        return ComposerResult(
            message=augmented_message,
            state=state,
            runtime_preflight=runtime_result,
            raw_assistant_content=content,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
        )

    return ComposerResult(
        message=content,
        state=state,
        runtime_preflight=runtime_result,
        tool_invocations=tool_invocations,
        llm_calls=llm_calls,
    )
