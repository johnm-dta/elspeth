"""No-tool finalization policy and user-facing augmentation messages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Final, Literal

from opentelemetry import metrics

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import is_discovery_tool
from elspeth.web.execution.schemas import (
    CHECK_STATE_EXISTS,
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
    ValidationResult,
)
from elspeth.web.interpretation_state import INTERPRETATION_REVIEW_PENDING_CODE

_COMPOSER_AUDIT_INTEGRITY_VIOLATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.audit_integrity_violation.total",
    description="Producer-side audit-integrity invariant violations (augmentation/replacement prefix contracts)",
)

_EMPTY_STATE_FINALIZE_SUFFIX = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] The pipeline is still empty — the composer did not "
    "complete a valid build this turn. To continue: refine your request "
    "with more specifics, or reply telling the composer to retry with the "
    "plan it described above."
)
_EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] The pipeline is still empty — the composer did not "
    "complete a valid build this turn.\n\nCause: {blocker}\n\n"
    "To continue: refine your request with more specifics, or reply telling "
    "the composer to retry with the plan it described above."
)

_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] Runtime preflight failed before this build could be "
    "marked complete.\n\nCause: {detail}{suggestion_block}\n\n"
    "The composer's analysis above is preserved verbatim; the validator's "
    "objection is recorded here."
)
_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE = (
    "\n\n---\n\n"
    "[ELSPETH-SYSTEM] Runtime preflight failed before this build could be "
    "marked complete.\n\nThe composer's analysis above is preserved verbatim; "
    "the validator's objection is recorded here."
)

_BUILD_INTENT_PHRASES: Final[tuple[str, ...]] = (
    "set up",
    "setup",
    "build",
    "create",
    "make",
    "wire",
    "add",
    "update",
    "modify",
    "change",
    "run",
    "execute",
    "process",
    "route",
    "split",
    "save",
    "workflow",
    "automation",
    "pipeline",
    "runnable",
)
_INFORMATION_ONLY_PREFIXES: Final[tuple[str, ...]] = (
    "what ",
    "what's ",
    "which ",
    "why ",
    "how ",
    "explain ",
    "tell me ",
    "show me ",
    "list ",
)

_AugmentationBranch = Literal[
    "no_mutation_empty_state_augmentation",
    "preflight_invalid_empty_state_augmentation",
    "preflight_invalid_non_empty_state_augmentation",
    "state_claim_grounding_correction",
    "orphaned_interpretation_review_augmentation",
    "advisor_signoff_blocked_augmentation",
]


def state_is_structurally_empty(state: CompositionState) -> bool:
    """Return True when no composition tools have produced visible state."""
    return not state.sources and not state.nodes and not state.outputs


def enforce_augmentation_prefix_invariant(
    *,
    branch: _AugmentationBranch,
    content: str,
    augmented: str,
) -> None:
    """Mechanically enforce the producer-side augmentation prefix contract."""
    if not augmented.startswith(content):
        _COMPOSER_AUDIT_INTEGRITY_VIOLATION_COUNTER.add(1, {"invariant": "augmentation_prefix", "branch": branch})
        raise AuditIntegrityError(
            f"Tier 1: composer augmentation contract violated on branch={branch!r}. "
            "Producer constructed an augmented message that does not have the "
            "model's pre-synthesis prose as a strict prefix. The consumer-side "
            "discriminator at routes._composer_history_content uses "
            "content.startswith(raw_content) to detect synthesis and strip "
            "the operator-facing suffix from LLM history; a producer that "
            "breaks the prefix property misroutes synthesized operator-facing "
            "text into the model's prior-turn history. Fix the augmented-"
            "message constructor so the prose appears verbatim at the start "
            "of message."
        )


def compose_empty_state_message(content: str, *, blocker: str | None = None) -> str:
    """Build the user-facing message for the empty-state finalize path."""
    suffix = _EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER.format(blocker=blocker) if blocker else _EMPTY_STATE_FINALIZE_SUFFIX
    if not content:
        return suffix.lstrip("\n").lstrip("-").lstrip()
    return content + suffix


def compose_preflight_failure_message(content: str, *, runtime_result: ValidationResult) -> str:
    """Build the user-facing message for the non-empty-state preflight-invalid path."""
    detail: str | None = None
    suggestion: str | None = None
    if runtime_result.errors:
        first_error = runtime_result.errors[0]
        detail = first_error.message
        suggestion = first_error.suggestion
    else:
        failed_checks = [check for check in runtime_result.checks if not check.passed]
        if failed_checks:
            detail = failed_checks[0].detail
    if detail:
        suggestion_block = f"\n\nSuggested fix: {suggestion}" if suggestion else ""
        suffix = _PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL.format(
            detail=detail,
            suggestion_block=suggestion_block,
        )
    else:
        suffix = _PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE
    if not content:
        return suffix.lstrip("\n").lstrip("-").lstrip()
    return content + suffix


def user_request_expects_pipeline_mutation(message: str) -> bool:
    """Return True when the user is asking the composer to build/edit/run."""
    normalized = " ".join(message.lower().split())
    if not normalized:
        return False
    if normalized.startswith(_INFORMATION_ONLY_PREFIXES) and "set up" not in normalized and "setup" not in normalized:
        return False
    return any(f" {phrase} " in f" {normalized} " for phrase in _BUILD_INTENT_PHRASES)


def _tool_failure_detail(payload: Mapping[str, Any]) -> str:
    """Extract a concise semantic failure detail from a ToolResult payload."""
    match payload:
        case {"data": {"error": error}}:
            return f": {error}"
        case {"validation": {"errors": [first_error, *_]}}:
            match first_error:
                case {"message": message}:
                    return f": {message}"
    if "error" in payload:
        return f": {payload['error']}"
    return "."


def last_mutation_was_pending_proposal(tool_invocations: tuple[ComposerToolInvocation, ...]) -> bool:
    """Return True iff the most recent non-discovery dispatch was an approval proposal."""
    for invocation in reversed(tool_invocations):
        if is_discovery_tool(invocation.tool_name):
            continue
        if invocation.status is not ComposerToolStatus.SUCCESS:
            return False
        if invocation.result_canonical is None:
            return False
        payload = json.loads(invocation.result_canonical)
        if type(payload) is not dict:
            return False
        data = payload["data"] if "data" in payload else None
        if type(data) is not dict:
            return False
        return ("status" in data) and data["status"] == "APPROVAL_REQUIRED"
    return False


def blocking_result_from_tool_invocations(tool_invocations: tuple[ComposerToolInvocation, ...]) -> str:
    """Name the most recent failed build/edit tool result, if one exists."""
    for invocation in reversed(tool_invocations):
        if invocation.status is ComposerToolStatus.ARG_ERROR:
            return f"{invocation.tool_name} failed before mutation ({invocation.error_class}: {invocation.error_message})."
        if invocation.status is ComposerToolStatus.PLUGIN_CRASH:
            return (
                f"{invocation.tool_name} crashed before a safe mutation completed ({invocation.error_class}: {invocation.error_message})."
            )
        if invocation.status is ComposerToolStatus.SUCCESS and invocation.result_canonical is not None:
            payload = json.loads(invocation.result_canonical)
            if type(payload) is dict and "success" in payload and payload["success"] is False:
                return f"{invocation.tool_name} returned success=false{_tool_failure_detail(payload)}"
            if (
                type(payload) is dict
                and "success" in payload
                and payload["success"] is True
                and invocation.version_after == invocation.version_before
                and not is_discovery_tool(invocation.tool_name)
            ):
                return f"{invocation.tool_name} succeeded without mutating CompositionState (version stayed {invocation.version_before})."
    return "the model ended the turn without calling any build/edit tool."


def is_pending_interpretation_handoff(result: ValidationResult) -> bool:
    readiness = result.readiness
    return (
        readiness.authoring_valid
        and readiness.completion_ready
        and not readiness.execution_ready
        and any(blocker.code == INTERPRETATION_REVIEW_PENDING_CODE for blocker in readiness.blockers)
    )


def no_mutation_empty_state_validation(blocker: str) -> ValidationResult:
    """Build the synthetic final-gate result for empty-state no-mutation replies."""
    detail = f"No composition-state mutation completed successfully; state_exists=false. Blocking result: {blocker}"
    suggestion = (
        "Call set_pipeline with source.blob_id or source.inline_blob, call "
        "set_source_from_blob/set_source plus set_output, or ask for the specific "
        "missing file/configuration."
    )
    return ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name=CHECK_STATE_EXISTS, passed=False, detail=detail, affected_nodes=(), outcome_code=None)],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message=detail,
                suggestion=suggestion,
                error_code=None,
            )
        ],
        readiness=ValidationReadiness(
            authoring_valid=False,
            execution_ready=False,
            completion_ready=False,
            blockers=[
                ValidationReadinessBlocker(
                    code="state_exists",
                    component_id=None,
                    component_type=None,
                    detail=detail,
                )
            ],
        ),
    )


def last_failure_was_pre_state_interpretation_review(tool_invocations: tuple[ComposerToolInvocation, ...]) -> bool:
    """Return True when review was called before any state row existed."""
    for invocation in reversed(tool_invocations):
        if is_discovery_tool(invocation.tool_name):
            continue
        return (
            invocation.tool_name == "request_interpretation_review"
            and invocation.status is ComposerToolStatus.ARG_ERROR
            and invocation.error_message is not None
            and "composition_state_id" in invocation.error_message
            and "missing current_state_id" in invocation.error_message
        )
    return False


def pre_state_interpretation_review_repair_message(*, next_turn: int, max_repair_turns: int) -> str:
    return (
        "[composer-system] You called request_interpretation_review before a "
        "persisted composition state existed. Do not reply to the user yet. "
        "First call set_pipeline or another state-staging tool to create the "
        "affected LLM transform with prompt_template containing the "
        "{{interpretation:<term>}} placeholder. Wait for that tool result, "
        "then call request_interpretation_review again. "
        f"This is forced repair turn {next_turn} of {max_repair_turns}."
    )
