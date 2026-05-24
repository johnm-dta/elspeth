"""ComposerServiceImpl — bounded LLM tool-use loop for pipeline composition.

Uses LiteLLM for provider abstraction. Model configured via
WebSettings.composer_model. Tool calls are executed against
CompositionState + CatalogService.

Dual-counter budget: separate limits for discovery and composition turns.
Discovery cache: cacheable discovery tool results cached per-compose-call
in a local dict variable (not an instance field) to avoid concurrent-request
races.

Layer: L3 (application).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, cast
from uuid import UUID

if TYPE_CHECKING:
    from elspeth.web.composer.guided.state_machine import TerminalState
    from elspeth.web.composer.redaction_telemetry import RedactionTelemetry
    from elspeth.web.sessions.protocol import SessionServiceProtocol
    from elspeth.web.sessions.telemetry import _SessionsTelemetry

import structlog
from opentelemetry import metrics
from sqlalchemy import Engine, update
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.composer_audit import ComposerToolInvocation, ComposerToolStatus
from elspeth.contracts.composer_interpretation import InterpretationKind
from elspeth.contracts.composer_llm_audit import (
    ComposerLLMCall,
    ComposerLLMCallStatus,
)
from elspeth.contracts.composer_progress import ComposerProgressEvent, ComposerProgressSink
from elspeth.contracts.errors import AuditIntegrityError, FailedTurnMetadata
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.plugins.transforms.llm.model_catalog import OPENROUTER_LITELLM_PREFIX
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer import yaml_generator
from elspeth.web.composer._compose_loop_carriers import (
    _CallModelOutcome,
    _ClassifyOutcome,
    _DispatchOutcome,
    _PersistOutcome,
    _TerminateOutcome,
    _ToolOutcome,
)
from elspeth.web.composer.anti_anchor import AntiAnchorTracker
from elspeth.web.composer.audit import (
    BufferingRecorder,
    DispatchAudit,
    begin_dispatch_or_arg_error,
    canonicalize_pydantic_cause,
    finish_arg_error,
    finish_success,
)
from elspeth.web.composer.availability import ComposerAvailability as ComposerAvailability  # re-export; genuine home is availability.py
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.llm_response_parsing import (
    apply_anthropic_cache_markers,
    attach_llm_calls,
    build_llm_call_record,
    safe_response_model,
    supports_anthropic_prompt_cache_markers,
    token_usage_from_response,
)
from elspeth.web.composer.progress import (
    convergence_progress_event,
    emit_progress,
    model_call_progress_event,
)
from elspeth.web.composer.prompts import build_messages, build_run_diagnostics_messages, build_system_prompt
from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerPluginCrashError,
    ComposerResult,
    ComposerRuntimePreflightError,
    ComposerServiceError,
    ComposerSettings,
    ToolArgumentError,
)
from elspeth.web.composer.recipe_intent_routing import match_freeform_recipe_intent
from elspeth.web.composer.skills import assert_skill_hash_unchanged_on_disk, load_skill_with_hash
from elspeth.web.composer.state import CompositionState, ValidationSummary
from elspeth.web.composer.tools import (
    _SESSION_AWARE_TOOL_HANDLERS,
    ADVISOR_TRIGGER_REACTIVE,
    ADVISOR_TRIGGER_VALUES,
    RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE,
    ToolResult,
    _sync_list_blobs,
    compute_proof_diagnostics,
    execute_tool,
    get_tool_definitions,
    is_discovery_tool,
)
from elspeth.web.execution.preflight import runtime_preflight_settings_hash
from elspeth.web.execution.runtime_preflight import (
    RuntimePreflightCoordinator,
    RuntimePreflightEntry,
    RuntimePreflightFailure,
    RuntimePreflightKey,
)
from elspeth.web.execution.schemas import (
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
    ValidationResult,
)
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.interpretation_state import (
    INTERPRETATION_REVIEW_PENDING_CODE,
    PROMPT_SHIELD_USER_TERM,
    PROMPT_SHIELD_WARNING_DRAFT,
    RAW_HTML_CLEANUP_REVIEW_DRAFT,
    RAW_HTML_CLEANUP_USER_TERM,
    interpretation_sites,
    vague_term_wiring_count,
)
from elspeth.web.sessions._persist_payload import AuditOutcome, RedactedToolRow
from elspeth.web.sessions.models import sessions_table

slog = structlog.get_logger()

_LLM_API_MAX_ATTEMPTS = 3
_LLM_API_RETRY_BASE_DELAY_SECONDS = 1.0
_INVALID_TOOL_ARGUMENTS_REDACTION_STATUS: Final[str] = "invalid_tool_arguments"

# Composer LLM sampling constants. Hardcoded for deterministic tool-call
# construction (RGR investigation 2026-05-06 §4.4 traced ~33% hard-GREEN
# ceiling on URL→download→line-explode primarily to LiteLLM/OpenRouter
# default sampling at ~1.0). Pipeline composition is closer to "extraction"
# than "creative writing" in the LLM-debugging-skill temperature guide;
# 0.0 is the right point for tool-construction tasks.
#
# The temperature value is recorded on every audit row via ComposerLLMCall.
# The seed is recorded when LiteLLM advertises support for the configured
# provider/model, otherwise it is omitted from the provider request and
# recorded as ``None`` so the audit row mirrors the actual request shape.
#
# Configurability is Tier 2 — do not read from settings/env without an ADR.
_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0
_COMPOSER_LLM_SEED: Final[int] = 42
_COMPOSER_LLM_SEED_PARAM: Final[str] = "seed"

# Bounded set of exception class names emitted as `exception_class` attribute on
# the runtime-preflight counter. Anything not in this set is bucketed as "other"
# to prevent unbounded cardinality from plugin class names leaking into metric labels.
_KNOWN_PREFLIGHT_EXCEPTION_CLASSES: frozenset[str] = frozenset(
    {
        "TimeoutError",
        "PluginNotFoundError",
        "PluginConfigError",
        "GraphValidationError",
        "ValidationError",  # pydantic.ValidationError
    }
)


@trust_boundary(
    tier=3,
    source="LLM composer tool-call payload (request_interpretation_review arguments)",
    source_param="arguments",
    suppresses=("R5",),
    invariant="raises AuditIntegrityError on a non-string or non-member kind; never coerces or writes a fabricated audit-row discriminator",
    test_ref="tests/unit/web/composer/test_request_interpretation_review_kind_boundary.py::test_non_str_kind_raises_audit_integrity_error",
    test_fingerprint="69e4bec4d82790adb9f3dfd104b1a504755721cd7aff2f56982e7cc7b86f0621",
)
def _request_interpretation_review_kind_from_arguments(arguments: Mapping[str, Any]) -> InterpretationKind:
    # `arguments` is the LLM tool-call payload (Tier 3); `kind` becomes the
    # interpretation-kind discriminator on an audit row, so a non-member or
    # non-string value must NOT be written. The InterpretationKind constructor
    # is itself the boundary check: a missing/non-string/unhashable value raises
    # ValueError (and TypeError on exotic inputs) from the enum lookup, which we
    # convert to a typed AuditIntegrityError rather than coercing or writing a
    # bad row. The uncaught AuditIntegrityError is the intended crash (we refuse
    # to record an audit row under a fabricated kind).
    raw_kind = arguments["kind"] if "kind" in arguments else None
    if not isinstance(raw_kind, str):
        raise AuditIntegrityError(f"request_interpretation_review rate-cap row has invalid kind {raw_kind!r}")
    try:
        return InterpretationKind(raw_kind)
    except ValueError as exc:
        raise AuditIntegrityError(f"request_interpretation_review rate-cap row has invalid kind {raw_kind!r}") from exc


# Module-level OTel counter for runtime preflight outcomes.
# Attributes: outcome (success | failure), exception_class (bounded closed-list | other)
_RUNTIME_PREFLIGHT_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.runtime_preflight.total",
    description="Total runtime-equivalent preflight invocations in the composer service",
)

# Module-level OTel counter for producer-side audit-integrity invariant violations.
# Increments before the AuditIntegrityError raise so an SRE has a count even if the
# uncaught exception kills the request before any other telemetry flushes. The crash
# is the right operational signal (CLAUDE.md: "Crash > silent wrong result"), but
# operators benefit from a counter trend ("how often is this firing?") to decide
# whether the field-level paydown at elspeth-7ae1732ab2 is becoming urgent.
# Attributes: invariant (augmentation_prefix | replacement_non_prefix), branch (the
# specific producer branch that violated; closed-list at type level via Literal).
_COMPOSER_AUDIT_INTEGRITY_VIOLATION_COUNTER = metrics.get_meter(__name__).create_counter(
    "composer.audit_integrity_violation.total",
    description="Producer-side audit-integrity invariant violations (augmentation/replacement prefix contracts)",
)


class _MalformedLLMResponseError(ComposerServiceError):
    """Internal carrier for malformed provider responses after the call returned."""

    def __init__(self, message: str, *, response: Any) -> None:
        super().__init__(message)
        self.response = response


class _BadRequestLLMError(ComposerServiceError):
    """Internal carrier for provider bad-request failures.

    Carries the raw provider message and HTTP status code on dedicated
    attributes so the route layer can surface them under
    ``expose_provider_error=True`` without having to re-parse the wrapped
    LiteLLM exception. ``str(self)`` is unchanged from the parent class —
    only the wrap message is rendered there.
    """

    def __init__(
        self,
        message: str,
        *,
        provider_detail: str | None = None,
        provider_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_detail = provider_detail
        self.provider_status_code = provider_status_code


def _apply_openrouter_app_identity(kwargs: dict[str, Any]) -> None:
    """Brand OpenRouter-routed composer calls as ELSPETH, not LiteLLM.

    LiteLLM injects its own OpenRouter attribution headers on every request
    unless the caller overrides them — ``HTTP-Referer: https://litellm.ai`` and
    ``X-Title: liteLLM`` (litellm/main.py). Without this the OpenRouter
    dashboard attributes all composer ("orchestrator") traffic to LiteLLM. The
    LLM transform plugins speak raw HTTP and set the same identity directly
    (``OPENROUTER_APP_REFERER`` / ``OPENROUTER_APP_TITLE`` in
    ``plugins/transforms/llm/providers/openrouter.py``); this brings the
    composer's LiteLLM-routed calls to parity using the one canonical source.

    Scoped to OpenRouter by the ``openrouter/`` routing prefix so the headers
    are never sent to other providers. ``HTTP-Referer`` is OpenRouter's primary
    ranking identifier; ``X-OpenRouter-Title`` is its current display-name
    header (what the plugins send) and ``X-Title`` is the legacy spelling
    LiteLLM defaults to ``liteLLM`` — we override both so no LiteLLM branding
    survives whichever one OpenRouter honours. Caller-supplied headers win: we
    only fill the identity keys we own (``setdefault``).
    """
    model = kwargs["model"] if "model" in kwargs else None
    if model is None or not model.startswith(OPENROUTER_LITELLM_PREFIX):
        return

    # Lazy import: providers/openrouter.py pulls httpx and the provider stack,
    # and the composer keeps that off the app-startup path.
    from elspeth.plugins.transforms.llm.providers.openrouter import (
        OPENROUTER_APP_REFERER,
        OPENROUTER_APP_TITLE,
    )

    existing = kwargs["extra_headers"] if "extra_headers" in kwargs else None
    # Caller-supplied headers win: our three attribution identity keys are laid
    # down first, then any caller headers overlay them (a key the caller already
    # set survives the merge). This expresses the precedence explicitly instead
    # of relying on setdefault's "fill-if-absent" side effect.
    headers: dict[str, str] = {
        "HTTP-Referer": OPENROUTER_APP_REFERER,
        "X-OpenRouter-Title": OPENROUTER_APP_TITLE,
        "X-Title": OPENROUTER_APP_TITLE,
        **(dict(existing) if existing else {}),
    }
    kwargs["extra_headers"] = headers


async def _litellm_acompletion(**kwargs: Any) -> Any:
    """Call LiteLLM lazily so app startup never imports provider machinery.

    Brands OpenRouter-routed calls with ELSPETH's app-attribution headers (see
    :func:`_apply_openrouter_app_identity`) so the OpenRouter dashboard credits
    composer traffic to ELSPETH rather than LiteLLM's defaults.
    """
    import litellm

    _apply_openrouter_app_identity(kwargs)
    return await litellm.acompletion(**kwargs)


def _litellm_completion_supports_param(model: str, param: str) -> bool:
    """Return whether LiteLLM advertises chat-completion support for ``param``."""
    import litellm

    supported_params = litellm.get_supported_openai_params(model=model)
    return isinstance(supported_params, list) and param in supported_params


def _composer_llm_seed_for_model(model: str) -> int | None:
    """Seed value to send to LiteLLM, or ``None`` when the provider rejects it."""
    if _litellm_completion_supports_param(model, _COMPOSER_LLM_SEED_PARAM):
        return _COMPOSER_LLM_SEED
    return None


def _state_is_structurally_empty(state: CompositionState) -> bool:
    """Return True when no composition tools have produced visible state.

    Used by ``_finalize_no_tool_response`` to short-circuit the synthetic
    preflight-failed message: when the model gave up after failing to
    converge on a valid pipeline build, its prose is more truthful than
    the synthesizer's Pydantic-noise replacement, and we surface the prose
    instead.

    "Structurally empty" means no sources, no nodes, no outputs — the three
    user-meaningful state fields. ``edges`` is implied (edges only exist
    between declared nodes) and ``metadata`` is not load-bearing for this
    check.
    """
    return not state.sources and not state.nodes and not state.outputs


def _tool_result_mutated_composition_state(
    *,
    version_before: int,
    result: ToolResult,
) -> bool:
    """Return True when a successful tool advanced the CompositionState version."""
    return result.success and result.updated_state.version > version_before


# Suffix appended to the model's prose when finalize-time runtime preflight
# fails on a structurally-empty state. Single source of truth so tests can
# pin the contract without duplicating the prose. Stable, system-attributed
# so a UI can detect and re-style it if desired.
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

# Suffix appended to the model's prose when finalize-time runtime preflight
# fails on a non-empty state. Companion to the empty-state pair above; the
# difference is the suffix wording — empty-state asks the operator to refine
# or retry, non-empty-state names the validator's specific objection so the
# operator sees both the model's analysis and the validator's reason.
# Single source of truth for the contract (issue elspeth-9cfbad6901).
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
]


def _enforce_augmentation_prefix_invariant(
    *,
    branch: _AugmentationBranch,
    content: str,
    augmented: str,
) -> None:
    """Mechanically enforce the producer-side augmentation prefix contract.

    All composer synthesis shapes are augmentations — the consumer-side
    discriminator at ``routes._composer_history_content`` strips the
    operator-facing suffix from LLM history by detecting the structural
    property ``message.startswith(raw_content)``. A producer that breaks
    the prefix property would emit a row whose suffix cannot be
    distinguished from the model's own prose at read time; the LLM
    would see synthesized operator-facing text in its prior-turn
    history. Crash here rather than commit a corrupt audit row.

    Empty ``content`` is permitted — ``"".startswith("")`` is trivially
    True and the augmentation builders degenerate to suffix-only output
    for empty inputs. Non-empty ``content`` MUST appear at the start of
    ``augmented``.
    """
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


def _compose_empty_state_message(content: str, *, blocker: str | None = None) -> str:
    """Build the user-facing message for the empty-state finalize path.

    Surfaces the model's content (which audit-DB inspection shows is
    typically an honest report of what the model tried and what blocked
    convergence) and appends a system-attributed suffix telling the user
    how to proceed.

    Two suffix templates are interpolated:
    ``_EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER`` when ``blocker`` is a
    non-empty string (carries ``Cause: {blocker}``);
    ``_EMPTY_STATE_FINALIZE_SUFFIX`` otherwise. Future maintainers
    extending the function MUST keep the two templates in sync —
    edits should generally apply to both.

    Args:
        content: The model's actual prose. Preserved verbatim at the start
            of the message. If the model produced no content at all (empty
            string), the suffix alone becomes the message — better than
            silence.
        blocker: When set to a non-empty string, the concrete cause that
            prevented mutation (e.g., a failed tool call's error). Included
            in the suffix so the operator gets the cause from the suffix
            even if the model's prose did not mention it. Used by the
            no-mutation empty-state augmentation; the preflight-invalid
            empty-state augmentation passes ``None`` because
            ``runtime_result`` already carries multi-error structured data.
            ``None`` and empty-string are treated identically (no blocker)
            to avoid emitting a degenerate ``Cause: \\n\\n`` suffix.
    """
    suffix = _EMPTY_STATE_FINALIZE_SUFFIX_WITH_BLOCKER.format(blocker=blocker) if blocker else _EMPTY_STATE_FINALIZE_SUFFIX
    if not content:
        return suffix.lstrip("\n").lstrip("-").lstrip()
    return content + suffix


def _compose_preflight_failure_message(content: str, *, runtime_result: ValidationResult) -> str:
    """Build the user-facing message for the non-empty-state preflight-invalid path.

    Surfaces the model's prose verbatim (panel-evals evidence shows it
    typically carries substantive disclosure: removed nodes, chosen
    operational semantics, the model's own diagnosis — see issue
    elspeth-9cfbad6901). Appends a system-attributed suffix carrying the
    technical preflight error so the operator sees both the model's
    analysis and the validator's objection.

    Companion to ``_compose_empty_state_message`` (preflight-invalid
    empty-state branch). Both branches augment; the difference is the
    suffix wording — empty-state asks the operator to refine or retry,
    non-empty-state names the validator's specific objection.

    Suffix template selection (longest-detail-wins fallback chain):
    ``_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_WITH_DETAIL`` when
    ``runtime_result`` carries a ValidationError or a failed check;
    ``_PREFLIGHT_INVALID_NONEMPTY_FINALIZE_SUFFIX_BARE`` when neither.
    Future maintainers extending this function MUST keep the templates
    in sync — edits should generally apply to both.

    Args:
        content: The model's actual prose. Preserved verbatim at the
            start of the message. If empty, the suffix becomes the whole
            message (matches ``_compose_empty_state_message``); the
            augmentation prefix invariant holds trivially because every
            string startswith "".
        runtime_result: The failed ValidationResult from the final-gate
            preflight. The first ValidationError's ``message`` and
            optional ``suggestion`` populate the suffix; falls back to
            the first failed check's ``detail``; falls back to the bare
            template when the result has neither.

    Boundary contract: the suffix is echoed verbatim into chat history
    and the OpenAI-format chat completion. Secrets are guaranteed
    absent because validate_pipeline() resolves secret refs before
    settings load and validation errors are derived from typed plugin
    configs, never from raw secret values. However, the suffix MAY
    carry operator-supplied path fragments (the operator's own
    configured CSV paths, sink output paths) and exception text that
    names plugin config field values — both surface verbatim from
    ``ValidationError.message`` / ``ValidationCheck.detail``. In the
    current single-operator deployment model this is acceptable: the
    operator already knows their own paths and config. If ELSPETH ever
    takes a multi-tenant deployment shape, the suffix builder must be
    sanitized before merge.
    """
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


def _user_request_expects_pipeline_mutation(message: str) -> bool:
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


def _last_mutation_was_pending_proposal(tool_invocations: tuple[ComposerToolInvocation, ...]) -> bool:
    """Return True iff the most recent non-discovery dispatch was an APPROVAL_REQUIRED proposal.

    Under ``trust_mode == "explicit_approve"`` the compose loop intercepts
    mutation tools, writes a ``composition_proposals`` row with
    ``status="pending"``, and returns ``success=True`` with an
    ``APPROVAL_REQUIRED`` payload while leaving ``CompositionState.version``
    unchanged. The work is complete from the model's perspective — only the
    human-approval step remains, and that step is intentionally outside the
    composer's reach. Treating this as a no-mutation failure was the
    convergence-blocking bug: the empty-state augmentation gate would
    synthesize ``[ELSPETH-SYSTEM] The pipeline is still empty`` over a turn
    that successfully landed a proposal, derailing both the operator's
    framing and (via the same content the LLM reads back on subsequent
    turns) the model's own state model.

    Discovery tools are transparent — the model interleaves them between
    real mutations — so we skip past them. The first non-discovery dispatch
    we find decides the answer.
    """
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
        # Tool results are heterogeneous: only proposal-shaped results carry a
        # "data" mapping with a "status" key, so absence is an expected state
        # (this result is simply not a pending proposal), not a missing-key bug.
        # Membership-test before subscript makes that expectation explicit.
        data = payload["data"] if "data" in payload else None
        if type(data) is not dict:
            return False
        return ("status" in data) and data["status"] == "APPROVAL_REQUIRED"
    return False


def _last_failure_was_pre_state_interpretation_review(tool_invocations: tuple[ComposerToolInvocation, ...]) -> bool:
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


def _blocking_result_from_tool_invocations(tool_invocations: tuple[ComposerToolInvocation, ...]) -> str:
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


def _pre_state_interpretation_review_repair_message(*, next_turn: int) -> str:
    return (
        "[composer-system] You called request_interpretation_review before a "
        "persisted composition state existed. Do not reply to the user yet. "
        "First call set_pipeline or another state-staging tool to create the "
        "affected LLM transform with prompt_template containing the "
        "{{interpretation:<term>}} placeholder. Wait for that tool result, "
        "then call request_interpretation_review again. "
        f"This is forced repair turn {next_turn} of {_MAX_REPAIR_TURNS}."
    )


def _pending_interpretation_review_repair_message(
    missing_sites: tuple[tuple[str, str, InterpretationKind], ...],
    *,
    next_turn: int,
) -> str:
    sites = ", ".join(f"{kind.value}:{component_id}:{term}" for component_id, term, kind in missing_sites)
    return (
        "[composer-system] The current pipeline contains pending assumption-review "
        "site(s) that are missing a matching pending interpretation event, or a "
        "vague-term handoff that is unresolvable: "
        f"{sites}. Do not reply to the user yet. For each listed handoff, "
        "call request_interpretation_review with the listed affected_node_id, "
        "kind, and user_term. If more than one handoff is listed, issue one "
        "request_interpretation_review tool call per listed handoff in this same "
        "assistant turn before stopping. For vague_term handoffs, first make sure "
        "the target LLM node both (a) contains exactly one matching pending vague_term "
        "interpretation_requirements entry and (b) wires that requirement into the prompt — "
        "either a single prompt_template_parts entry "
        '{"kind": "interpretation_ref", "requirement_id": "<the requirement id>"} '
        "referencing it, or exactly one legacy {{interpretation:<term>}} token in "
        "options.prompt_template. A requirement with no wiring cannot be resolved, so the "
        "review would dead-end; if either is missing, patch the node before "
        "calling request_interpretation_review. Use the matching interpretation_requirements "
        "draft as llm_draft. For llm_prompt_template, llm_draft must equal the current "
        "options.prompt_template. For invented_source, llm_draft must equal the "
        "source requirement draft. For pipeline_decision, llm_draft must equal "
        "the target node's requirement draft. If a pipeline_decision site has no "
        f"matching requirement and user_term is {RAW_HTML_CLEANUP_USER_TERM!r}, patch "
        "the target field_mapper node first with an interpretation_requirements "
        "entry whose kind is 'pipeline_decision', status is 'pending', and draft is "
        f"{RAW_HTML_CLEANUP_REVIEW_DRAFT!r}. "
        f"If user_term is {PROMPT_SHIELD_USER_TERM!r}, patch the target LLM node first "
        "with an interpretation_requirements entry whose kind is 'pipeline_decision', "
        f"status is 'pending', and draft is {PROMPT_SHIELD_WARNING_DRAFT!r}; if the "
        "workflow cannot add the shield, keep going with the warning instead of blocking. "
        f"This is forced repair turn {next_turn} of {_MAX_REPAIR_TURNS}."
    )


def _resolvable_vague_term_count(
    state: CompositionState,
    *,
    node_id: str,
    term: str,
) -> int:
    """Count *resolvable* vague_term wirings for ``term`` on LLM node ``node_id``.

    Delegates to :func:`vague_term_wiring_count` so the repair loop's
    resolvability test cannot drift from the tool-boundary gate or the resolver
    contract. A pending requirement counts only when its substitution wiring (a
    ``prompt_template_parts`` ``interpretation_ref`` or a legacy
    ``{{interpretation:<term>}}`` placeholder) is present — which is what lets
    the loop catch a requirement whose wiring was stripped by a later mutation
    *after* its review event already existed (drift the tool boundary cannot
    re-check once the event is persisted).
    """
    for node in state.nodes:
        if node.id != node_id or node.plugin != "llm":
            continue
        return vague_term_wiring_count(node.options, user_term=term)
    return 0


def _is_pending_interpretation_handoff(result: ValidationResult) -> bool:
    readiness = result.readiness
    return (
        readiness.authoring_valid
        and readiness.completion_ready
        and not readiness.execution_ready
        and any(blocker.code == INTERPRETATION_REVIEW_PENDING_CODE for blocker in readiness.blockers)
    )


def _no_mutation_empty_state_validation(blocker: str) -> ValidationResult:
    """Build the synthetic final-gate result for empty-state no-mutation replies."""
    detail = f"No composition-state mutation completed successfully; state_exists=false. Blocking result: {blocker}"
    suggestion = (
        "Call set_pipeline with source.blob_id or source.inline_blob, call "
        "set_source_from_blob/set_source plus set_output, or ask for the specific "
        "missing file/configuration."
    )
    return ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name="state_exists", passed=False, detail=detail, affected_nodes=(), outcome_code=None)],
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


_PROVIDER_REQUIRED_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_API_KEY",),
    "azure_ai": ("AZURE_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


@dataclass(frozen=True, slots=True)
class _SessionAwareDispatchOutcome:
    """Return value of ``_dispatch_session_aware_tool``.

    Carries the post-dispatch signals the compose loop needs to update
    its loop-local accounting:

    - ``result``: the SUCCESS ``ToolResult`` when the handler returned
      cleanly; ``None`` when the dispatch ended in an ARG_ERROR path
      (rate cap or generic) — the audit record was already written and
      the LLM-facing tool message already appended to ``llm_messages``.
    - ``is_discovery``: whether the loop should charge this turn to the
      discovery or composition budget. Session-aware tools that mutate
      composition state report ``False`` so they count as composition
      turns regardless of the success/failure shape.
    """

    result: ToolResult | None
    is_discovery: bool


@dataclass(frozen=True, slots=True)
class _CachedDiscoveryPayload:
    """State-independent portion of a cacheable discovery tool result."""

    success: bool
    affected_nodes: tuple[str, ...]
    data: Any


_RuntimePreflightCache = dict[RuntimePreflightKey, RuntimePreflightEntry]


# The per-dispatch audit envelope (DispatchAudit, begin_dispatch, finish_*)
# and the structural enforcement helper (dispatch_with_audit) live in
# web/composer/audit.py next to the BufferingRecorder. Hoisting them out of
# this module localises the "audit fires before return on every path"
# invariant inside a single helper rather than spreading it across seven
# procedural recorder.record() call sites in _compose_loop. See the audit.py
# module docstring for the contract details.


# Hard cap on proof-step-driven repair turns. When the assistant claims
# completion but preview_pipeline's proof_diagnostics still has blocking
# entries, the loop may inject a synthetic repair message and continue for
# at most this many additional iterations. After the cap, the original
# termination path runs — preventing indefinite spin against a model that
# refuses to apply the suggested repair.
_MAX_REPAIR_TURNS: Final[int] = 2
_MAX_PENDING_PROPOSALS_PER_TURN: Final[int] = 10


def _proof_repair_is_applicable(state: CompositionState) -> bool:
    """Return True iff the proof step has any input it can inspect.

    The forced-repair gate must fire whenever ``compute_proof_diagnostics``
    might find blocking diagnostics. The proof step is a no-op for sources
    that aren't blob-backed (no bytes to read), so the gate's predicate is
    "at least one source is present AND options carries a ``blob_ref``" — *not* "state
    changed this turn", because a blocker can survive session resume into
    a turn where the LLM does no mutations.

    ``SourceSpec.options`` is internally typed as ``Mapping[str, Any]``
    (Tier-1 dataclass invariant — no isinstance probe needed). ``blob_ref``
    is an optional, well-known key set by the binding tools; its absence
    is a documented part of the contract (path-based sources don't have
    one), so containment checking is the appropriate primitive here.
    """
    return any("blob_ref" in source.options for source in state.sources.values())


def _empty_state_uploaded_blob_repair_message(ready_blobs: tuple[Mapping[str, Any], ...], *, next_turn: int) -> str:
    """Build a bounded repair prompt for empty-state stalls with ready uploads.

    The message contains the same metadata exposed by ``list_blobs``: blob id,
    filename, MIME type, byte size, creator, and status. It never includes raw
    blob bytes, storage paths, or full content hashes.
    """
    rendered_blobs = []
    for blob in ready_blobs[:5]:
        rendered_blobs.append(
            "- "
            f"id={blob['id']}; "
            f"filename={blob['filename']}; "
            f"mime_type={blob['mime_type']}; "
            f"size_bytes={blob['size_bytes']}; "
            f"created_by={blob['created_by']}; "
            f"status={blob['status']}"
        )
    remaining = len(ready_blobs) - len(rendered_blobs)
    if remaining > 0:
        rendered_blobs.append(f"- ... {remaining} more ready blob(s) omitted from this bounded repair prompt.")

    blob_block = "\n".join(rendered_blobs)
    return (
        "[composer-system] No composition-state mutation completed successfully, "
        "but this session has ready uploaded blob(s). Do not reply with another conceptual plan. "
        "Continue by calling a build/edit tool: prefer set_pipeline with source.blob_id, "
        "or set_source_from_blob followed by the needed nodes and outputs. "
        "Use inspect_source(blob_id) when you need headers, sample_row_count, or inferred types. "
        "Do not infer that a CSV is header-only from metadata, filename, prior prose, or a failed attempt; "
        "only inspect_source can establish the observed row count. "
        f"This is forced repair turn {next_turn} of {_MAX_REPAIR_TURNS}.\n\n"
        f"Ready uploaded blob(s):\n{blob_block}"
    )


class ComposerServiceImpl:
    """LLM-driven pipeline composer with dual-counter budget and discovery caching.

    Runs a bounded tool-use loop with separate budgets for discovery
    and composition turns. Cacheable discovery tool results are cached
    per-compose-call in a local dict (not an instance field) to avoid
    concurrent-request races.

    Budget classification: a turn containing at least one mutation tool
    call charges the composition budget. A turn containing only discovery
    tool calls charges the discovery budget. Cache hits do not charge
    any budget.

    Args:
        catalog: CatalogService for discovery tool delegation.
        settings: ComposerSettings with composer_max_composition_turns,
            composer_max_discovery_turns, composer_timeout_seconds,
            composer_model, data_dir.
    """

    def __init__(
        self,
        catalog: CatalogService,
        settings: ComposerSettings,
        *,
        sessions_service: SessionServiceProtocol | None = None,
        session_engine: Engine | None = None,
        secret_service: WebSecretResolver | None = None,
        runtime_preflight_coordinator: RuntimePreflightCoordinator | None = None,
    ) -> None:
        self._catalog = catalog
        self._sessions_service = sessions_service
        self._model = settings.composer_model
        self._max_composition_turns = settings.composer_max_composition_turns
        self._max_discovery_turns = settings.composer_max_discovery_turns
        self._timeout_seconds = settings.composer_timeout_seconds
        self._data_dir: str = str(settings.data_dir)
        self._session_engine = session_engine
        self._secret_service = secret_service
        self._settings = settings
        self._runtime_preflight_timeout_seconds = settings.composer_runtime_preflight_timeout_seconds
        self._runtime_preflight_coordinator = runtime_preflight_coordinator or RuntimePreflightCoordinator()
        self._availability = self._compute_availability()
        from elspeth.web.composer.redaction_telemetry import OtelRedactionTelemetry
        from elspeth.web.sessions.telemetry import build_sessions_telemetry

        self._max_tool_calls_per_turn: int = self._settings.composer_max_tool_calls_per_turn
        self._telemetry: _SessionsTelemetry = build_sessions_telemetry(meter=metrics.get_meter("elspeth.web.composer"))
        self._redaction_telemetry: RedactionTelemetry = OtelRedactionTelemetry()
        self._phase3_last_tool_outcomes: tuple[_ToolOutcome, ...] = ()
        self._phase3_last_expected_current_state_id: str | None = None
        self._phase3_last_redacted_assistant_tool_calls: tuple[Mapping[str, Any], ...] = ()
        self._phase3_last_redacted_tool_rows: tuple[RedactedToolRow, ...] = ()
        self._phase3_last_audit_outcome: AuditOutcome | None = None

        # F-5a. Re-read the composer skill markdown from disk and assert
        # its SHA-256 still matches the
        # ``PIPELINE_COMPOSER_SKILL_HASH`` captured atomically at module
        # import. A mismatch means the on-disk file was edited after the
        # LRU cache populated — the LLM would be prompted with the cached
        # (older) text while any audit row written this process would
        # record that older hash. The audit trail and the actual file
        # would then disagree, which is an operator-actionable Tier-1
        # anomaly. The assertion raises ``RuntimeError`` with restart
        # guidance. Performed once per service instantiation; subsequent
        # in-process drift is bounded by the LRU cache lifetime (process
        # death restarts the cache).
        from elspeth.web.composer.prompts import (
            PIPELINE_COMPOSER_SKILL_HASH,
            PIPELINE_COMPOSER_SKILL_NAME,
        )

        assert_skill_hash_unchanged_on_disk(
            PIPELINE_COMPOSER_SKILL_NAME,
            PIPELINE_COMPOSER_SKILL_HASH,
        )
        self._composer_skill_hash: str = PIPELINE_COMPOSER_SKILL_HASH
        self._composer_skill_name: str = PIPELINE_COMPOSER_SKILL_NAME
        # F-5c gate: ensures the first ``compose()`` call upserts
        # the skill markdown into ``skill_markdown_history`` exactly once
        # per service instance. Subsequent compose() calls observe the
        # flag set and skip the upsert.
        self._skill_markdown_history_upserted: bool = False
        # Per-session set of ``(kind, plugin_name)`` pairs for which
        # ``get_plugin_schema`` has returned successfully in this service
        # instance. Surfaced in the per-turn system context as
        # ``schemas_loaded_this_session`` so the LLM can see at a glance
        # which plugins it has already introspected and which schemas it
        # still needs to read before constructing a config (see
        # ``prompts.build_context_string``). A new session_id transparently
        # gets an empty set on first access; in-memory only because the
        # tracker is convergence guidance, not auditable state.
        # Concurrency: a single session is driven serially through one
        # compose() call at a time; a plain dict is sufficient.
        self._schemas_loaded_by_session: dict[str, set[tuple[str, str]]] = {}

    async def _run_one_turn_for_test(
        self,
        *,
        llm: Any | None = None,
        session_id: str | None = None,
        current_state_id: str | None = None,
        initial_state: CompositionState | None = None,
        user_message_id: str | None = None,
        message: str = "one-turn compose-loop test driver",
    ) -> ComposeLoopTestResult:
        """Drive exactly one compose-loop turn for compose-loop tests.

        Test-only helper: it bypasses HTTP route setup but exercises the
        same ``_compose_loop`` body, including ``_require_sessions_service()``.
        Missing ``sessions_service`` must therefore fail with
        ``RuntimeError("sessions_service not wired")``, not ``AttributeError``
        or a constructor ``TypeError``.
        """

        from elspeth.web.composer.state import PipelineMetadata

        del user_message_id
        self._require_sessions_service()
        state = initial_state or CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        resolved_session_id = session_id or "00000000-0000-0000-0000-000000000000"
        original_call_llm = self._call_llm

        async def _call_fake_llm(messages: Any, tools: Any) -> Any:
            if llm is None:
                return await original_call_llm(messages, tools)
            return await llm(messages, tools)

        self._call_llm = _call_fake_llm  # type: ignore[method-assign]
        try:
            result = await self._compose_loop(
                message,
                [],
                state,
                session_id=resolved_session_id,
                initial_current_state_id=current_state_id,
                deadline=asyncio.get_event_loop().time() + self._timeout_seconds,
            )
        finally:
            self._call_llm = original_call_llm  # type: ignore[method-assign]

        return ComposeLoopTestResult(
            assistant_message=result.message,
            tool_outcomes=tuple(self._phase3_last_tool_outcomes),
            persisted_assistant_tool_calls=tuple(self._phase3_last_redacted_assistant_tool_calls),
            persisted_tool_row_content=tuple(row.content for row in self._phase3_last_redacted_tool_rows),
            tool_invocations=result.tool_invocations,
        )

    def _serialize_response_via_walker(
        self,
        outcome: _ToolOutcome,
        *,
        telemetry: Any,
    ) -> str:
        """Serialize one Step 1 outcome through the redaction response walker."""

        # Keep redaction imports local to the redaction paths; service.py is
        # already load-order sensitive and these walkers are cold-path helpers.
        from elspeth.contracts.freeze import deep_thaw
        from elspeth.core.canonical import canonical_json
        from elspeth.web.composer.redaction import MANIFEST, redact_tool_call_response

        if outcome.error_class is None:
            response = outcome.response
            if isinstance(response, Mapping):
                response_payload = deep_thaw(response)
            else:
                result = cast(ToolResult, response)
                response_payload = result.to_dict()
            if outcome.call.function.name not in MANIFEST:
                return canonical_json(response_payload)
            redacted = redact_tool_call_response(
                tool_name=outcome.call.function.name,
                response=response_payload,
                telemetry=telemetry,
            )
            return canonical_json(redacted)
        return canonical_json(
            {
                "error_class": outcome.error_class,
                "error_message": outcome.error_message,
            }
        )

    def _state_payload_for_compose_turn_for_test(
        self,
        response: Any,
    ) -> Any:
        """Build a StatePayload for the current interim Step 2 redacted row."""

        del self
        from elspeth.web.sessions._persist_payload import StatePayload
        from elspeth.web.sessions.protocol import CompositionStateData

        result = cast(ToolResult, response)
        state_d = result.updated_state.to_dict()
        return StatePayload(
            data=CompositionStateData(
                sources=state_d["sources"],
                nodes=state_d["nodes"],
                edges=state_d["edges"],
                outputs=state_d["outputs"],
                metadata_=state_d["metadata"],
                is_valid=result.validation.is_valid,
                validation_errors=tuple(error.message for error in result.validation.errors),
                composer_meta=None,
            ),
            # persist_compose_turn inserts composition state rows under
            # the session write lock and re-derives
            # lineage from per-session version ordering when this is None
            # (spec §5.7.1). The async loop deliberately does not fabricate a
            # predecessor id for a row that has not been allocated yet.
            derived_from_state_id=None,
        )

    def _require_sessions_service(self) -> SessionServiceProtocol:
        """Return the wired sessions service or fail at the persistence boundary."""

        if self._sessions_service is None:
            raise RuntimeError("sessions_service not wired")
        return self._sessions_service

    async def _maybe_upsert_skill_markdown_history(self) -> None:
        """Best-effort first-use upsert of the composer skill markdown (F-5c).

        On the first ``_compose_loop`` entry of this service instance,
        archive the exact skill markdown text into
        ``skill_markdown_history`` keyed by SHA-256. Subsequent calls are
        a cheap in-process branch (flag check) and never touch the DB.

        No-op when ``sessions_service`` is not wired (CLI / unit-test
        paths) — the upsert is meaningful only on deployments that
        persist interpretation events. Per-instance flag, not per-process:
        a service rebuild (test fixture, lifespan restart) re-runs the
        upsert on the new instance, which is harmless under
        ``INSERT OR IGNORE``.

        Failures are NOT silenced: the upsert is best-effort with
        respect to the audit-event row (we don't gate the interpretation
        write on it succeeding), but a real DB failure here indicates
        the session DB is unreachable, in which case the broader compose
        loop is also unable to function — letting the exception escape
        surfaces the failure at the start of the request instead of
        midway through.
        """
        if self._skill_markdown_history_upserted:
            return
        if self._sessions_service is None:
            return
        # Re-read the in-memory cached text (the same atomic pair fed to
        # the LLM). ``load_skill_with_hash`` is @lru_cache'd so this is a
        # cheap dict hit; the assert in __init__ already verified the
        # on-disk file still matches.
        text, sha256_hex = load_skill_with_hash(self._composer_skill_name)
        # Defensive Tier-1 consistency check: the cached hash on the
        # service instance and the hash returned by the cache MUST agree.
        # A mismatch would mean a race between this method and a manual
        # cache invalidation; the audit trail's join semantics would
        # break.
        if sha256_hex != self._composer_skill_hash:
            raise RuntimeError(
                f"Composer skill hash drift detected: service instance cached "
                f"{self._composer_skill_hash!r} but load_skill_with_hash now returns "
                f"{sha256_hex!r}. The LRU cache was invalidated mid-process; restart "
                f"elspeth-web.service so the in-memory skill prompt and the audit "
                f"row's composer_skill_hash agree."
            )
        await self._sessions_service.upsert_skill_markdown_history(
            skill_hash=sha256_hex,
            filename=f"{self._composer_skill_name}.md",
            content=text,
        )
        self._skill_markdown_history_upserted = True

    def get_availability(self) -> ComposerAvailability:
        """Return the boot-time composer availability snapshot."""
        return self._availability

    def _runtime_preflight(self, state: CompositionState, user_id: str | None) -> ValidationResult:
        return validate_pipeline(
            state,
            self._settings,
            yaml_generator,
            secret_service=self._secret_service,
            user_id=user_id,
        )

    async def _missing_pending_interpretation_review_sites(
        self,
        state: CompositionState,
        *,
        session_id: str | None,
    ) -> tuple[tuple[str, str, InterpretationKind], ...]:
        """Return pending interpretation handoffs that cannot be resolved."""

        sites = interpretation_sites(state)
        if session_id is None:
            return ()
        sessions_service = self._require_sessions_service()
        events = await sessions_service.list_interpretation_events(UUID(session_id), status="pending")
        pending_sites = {
            (event.affected_node_id, event.user_term.strip(), event.kind)
            for event in events
            if event.affected_node_id is not None and event.user_term is not None and event.kind is not None
        }
        missing_or_unresolvable: dict[tuple[str, str, InterpretationKind], None] = {}
        for site in sites:
            site_key = (site.component_id, site.user_term, site.kind)
            if site_key not in pending_sites:
                missing_or_unresolvable[site_key] = None
        for event in events:
            if event.kind is not InterpretationKind.VAGUE_TERM or event.affected_node_id is None or event.user_term is None:
                continue
            event_site_key = (event.affected_node_id, event.user_term.strip(), event.kind)
            wiring_count = _resolvable_vague_term_count(
                state,
                node_id=event_site_key[0],
                term=event_site_key[1],
            )
            if wiring_count != 1:
                missing_or_unresolvable[event_site_key] = None
        return tuple(missing_or_unresolvable)

    def _new_runtime_preflight_cache(self) -> _RuntimePreflightCache:
        return {}

    def _raise_cached_runtime_preflight_failure(
        self,
        failure: RuntimePreflightFailure,
        *,
        state: CompositionState,
        initial_version: int,
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> NoReturn:
        raise ComposerRuntimePreflightError.capture(
            failure.original_exc,
            state=state,
            initial_version=initial_version,
            llm_calls=llm_calls,
        ) from failure.original_exc

    async def _cached_runtime_preflight(
        self,
        state: CompositionState,
        *,
        user_id: str | None,
        cache: _RuntimePreflightCache,
        initial_version: int,
        session_scope: str,
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> ValidationResult:
        key = RuntimePreflightKey(
            session_scope=session_scope,
            state_version=state.version,
            settings_hash=runtime_preflight_settings_hash(self._settings),
        )
        # A cache miss is the normal, expected state on the first preflight for
        # this key — absence is not a missing-key bug, so membership-test then
        # subscript instead of relying on .get's implicit-None default.
        cached = cache[key] if key in cache else None
        if isinstance(cached, ValidationResult):
            return cached
        if isinstance(cached, RuntimePreflightFailure):
            self._raise_cached_runtime_preflight_failure(
                cached,
                state=state,
                initial_version=initial_version,
                llm_calls=llm_calls,
            )

        async def worker() -> ValidationResult:
            return await asyncio.wait_for(
                run_sync_in_worker(self._runtime_preflight, state, user_id),
                timeout=self._runtime_preflight_timeout_seconds,
            )

        entry = await self._runtime_preflight_coordinator.run(key, worker)
        cache[key] = entry
        if isinstance(entry, RuntimePreflightFailure):
            exc_name = type(entry.original_exc).__name__
            exc_class = exc_name if exc_name in _KNOWN_PREFLIGHT_EXCEPTION_CLASSES else "other"
            _RUNTIME_PREFLIGHT_COUNTER.add(
                1,
                {"outcome": "failure", "exception_class": exc_class},
            )
            self._raise_cached_runtime_preflight_failure(
                entry,
                state=state,
                initial_version=initial_version,
                llm_calls=llm_calls,
            )
        _RUNTIME_PREFLIGHT_COUNTER.add(1, {"outcome": "success"})
        return entry

    async def _attempt_empty_state_uploaded_blob_repair(
        self,
        *,
        state: CompositionState,
        llm_messages: list[dict[str, Any]],
        session_id: str | None,
        repair_turns_used: int,
    ) -> bool:
        """Continue once when the model stalls despite ready uploaded blobs.

        This catches the uploaded-file happy path failure mode: the user has
        provided data, the session blob inventory has a ready blob, but the
        LLM emits prose and no build/edit tool calls while CompositionState is
        still empty. The repair message gives the model concrete blob ids and
        permitted next tools, then reuses the capped repair-turn budget.
        """
        if repair_turns_used >= _MAX_REPAIR_TURNS:
            return False
        if not _state_is_structurally_empty(state):
            return False
        if self._session_engine is None or session_id is None:
            return False

        blobs = await run_sync_in_worker(_sync_list_blobs, self._session_engine, session_id)
        ready_blobs = tuple(blob for blob in blobs if blob["status"] == "ready" and blob["created_by"] == "user")
        if not ready_blobs:
            return False

        llm_messages.append(
            {
                "role": "user",
                "content": _empty_state_uploaded_blob_repair_message(
                    ready_blobs,
                    next_turn=repair_turns_used + 1,
                ),
            }
        )
        return True

    def _attempt_proof_repair(
        self,
        *,
        state: CompositionState,
        llm_messages: list[dict[str, Any]],
        session_id: str | None,
        repair_turns_used: int,
    ) -> bool:
        """Pre-finalize proof gate.

        When the assistant emits no tool_calls (claiming completion), check
        ``preview_pipeline``'s ``proof_diagnostics`` for blocking entries.
        If any are found AND the repair-turn budget has not been exhausted,
        synthesize a user-attributed message describing each diagnostic plus
        its ``suggested_repair`` and append it to ``llm_messages``. The
        outer compose loop then continues for one more iteration so the
        model can apply the suggested fix.

        Returns True when a repair message was injected (the loop should
        ``continue`` and skip finalization). Returns False when there are
        no blocking diagnostics OR the repair budget is exhausted.

        Boundary contract: this helper NEVER catches plugin exceptions.
        It only repairs *configurations* via composer-tool calls. Plugin
        bugs (transform.process raising) propagate to the operator per the
        Plugin Ownership policy in CLAUDE.md.

        The synthesised message is appended verbatim into chat history. It
        contains operator-supplied column names and the diagnostic-message
        text (which may name CSV paths the operator wrote). No secrets are
        carried — proof_diagnostics never reads source bytes through any
        path that retains decoded content; only inspect_blob_content's
        bounded-summary facts are surfaced.
        """
        if repair_turns_used >= _MAX_REPAIR_TURNS:
            return False

        diagnostics = compute_proof_diagnostics(
            state,
            session_engine=self._session_engine,
            session_id=session_id,
        )
        # The diagnostic dict shape is the documented contract of
        # ``compute_proof_diagnostics`` (see ``tools.py``): every entry
        # has ``severity``, ``code``, ``message``, ``suggested_repair``,
        # ``evidence_locator``. This is an internal-package invariant,
        # not a Tier-3 trust boundary — a missing key is a bug in the
        # diagnostic builder, not malformed external data, so direct
        # subscript access is correct and a ``KeyError`` here is the
        # right failure mode (informative crash) per CLAUDE.md
        # offensive-programming policy. ``.get()`` fallbacks would bury
        # contract drift and ship ``[unknown]`` codes / empty messages
        # into the audit trail and the LLM's repair-message context.
        blocking = [d for d in diagnostics if d["severity"] == "blocking"]
        if not blocking:
            return False

        # Cap at 3 blocking entries in the synthesised message to keep the
        # context window manageable. The model can call preview_pipeline to
        # see the full list.
        rendered = []
        for i, d in enumerate(blocking[:3], start=1):
            rendered.append(f"{i}. [{d['code']}] {d['message']}\n   Suggested repair: {d['suggested_repair']}")

        next_turn = repair_turns_used + 1
        budget_note = (
            f"This is forced repair turn {next_turn} of {_MAX_REPAIR_TURNS}. "
            "Apply the suggested repair via the appropriate composer tool, then call "
            "preview_pipeline to verify the diagnostics are cleared before finalising again."
        )

        message = (
            "[composer-system] Pre-finalisation proof step found blocking "
            "diagnostic(s) — the pipeline cannot run as currently configured. "
            "Do not respond to the user yet; resolve these first.\n\n" + "\n\n".join(rendered) + "\n\n" + budget_note
        )

        llm_messages.append({"role": "user", "content": message})
        return True

    async def _finalize_no_tool_response(
        self,
        *,
        content: str,
        state: CompositionState,
        initial_version: int,
        user_id: str | None,
        last_runtime_preflight: ValidationResult | None,
        runtime_preflight_cache: _RuntimePreflightCache,
        session_scope: str,
        user_message: str = "",
        mutation_success_seen: bool = False,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
    ) -> ComposerResult:
        """Apply the deterministic final-gate check and build a ComposerResult.

        Delegates to :func:`no_tool_finalize.finalize_no_tool_response`.
        """
        from elspeth.web.composer.no_tool_finalize import finalize_no_tool_response

        return await finalize_no_tool_response(
            self,
            content=content,
            state=state,
            initial_version=initial_version,
            user_id=user_id,
            last_runtime_preflight=last_runtime_preflight,
            runtime_preflight_cache=runtime_preflight_cache,
            session_scope=session_scope,
            user_message=user_message,
            mutation_success_seen=mutation_success_seen,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
        )

    async def explain_run_diagnostics(self, snapshot: Mapping[str, object]) -> str:
        """Return a plain-language explanation of a bounded run snapshot.

        The explanation is advisory UI text only: it does not call composer
        tools, mutate CompositionState, or persist chat messages.
        """
        if not self._availability.available:
            raise ComposerServiceError(self._availability.reason or "Composer is unavailable.")

        try:
            messages = build_run_diagnostics_messages(snapshot, data_dir=self._data_dir)
        except OSError as exc:
            raise ComposerServiceError(f"Failed to load deployment skill ({type(exc).__name__})") from exc

        try:
            from litellm.exceptions import APIError as LiteLLMAPIError
            from litellm.exceptions import (
                BlockedPiiEntityError,
                BudgetExceededError,
                GuardrailRaisedException,
            )

            response = await asyncio.wait_for(
                self._call_text_llm(messages),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            raise ComposerServiceError("Run diagnostics explanation timed out") from None
        except (
            LiteLLMAPIError,
            BudgetExceededError,
            BlockedPiiEntityError,
            GuardrailRaisedException,
        ) as exc:
            raise ComposerServiceError(f"LLM unavailable ({type(exc).__name__})") from exc

        content = cast(str | None, response.choices[0].message.content)
        if content is None or not content.strip():
            raise ComposerServiceError("LLM returned an empty diagnostics explanation")
        return content.strip()

    async def compose(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
        session_id: str | None = None,
        current_state_id: str | None = None,
        user_id: str | None = None,
        progress: ComposerProgressSink | None = None,
        guided_terminal: TerminalState | None = None,
        user_message_id: str | None = None,
    ) -> ComposerResult:
        """Run the LLM composition loop with dual-counter budget.

        Args:
            message: The user's chat message.
            messages: Chat history as plain dicts (pre-converted from
                ChatMessageRecord by route handler; seam contract B).
            state: The current CompositionState.
            current_state_id: Database id of ``state`` when it came from a
                persisted session row. Used as the stale-state guard for
                compose-loop tool-call audit persistence.
            guided_terminal: When set, the resolved TerminalState from the
                completed guided session; triggers the layered mode-transition
                prompt for this first freeform turn (spec §8.2). The caller
                is responsible for gate logic and ``transition_consumed`` flip.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If a budget is exhausted or
                the timeout is exceeded.
        """
        if not self._availability.available:
            raise ComposerServiceError(self._availability.reason or "Composer is unavailable.")

        deadline = asyncio.get_event_loop().time() + self._timeout_seconds
        from litellm.exceptions import APIError as LiteLLMAPIError

        try:
            routed_result = await self._try_apply_freeform_recipe_intent(
                message=message,
                state=state,
                session_id=session_id,
                user_id=user_id,
                progress=progress,
                user_message_id=user_message_id,
            )
            if routed_result is not None:
                return routed_result
            return await self._compose_loop(
                message,
                messages,
                state,
                session_id,
                current_state_id,
                user_id,
                deadline,
                progress,
                guided_terminal,
                user_message_id,
            )
        except ComposerConvergenceError as exc:
            await emit_progress(
                progress,
                convergence_progress_event(budget_exhausted=exc.budget_exhausted),
            )
            # Has its own partial_state; route handler persists. Do not intercept.
            raise
        except ComposerPluginCrashError as crash:
            # Plugin-bug crash path. The exception already carries
            # partial_state (populated by _compose_loop at the execute_tool
            # site when state.version > initial_version), so the route
            # handler can persist the accumulated mutations into
            # composition_states symmetrically with the convergence path.
            #
            # Here we only add the session-row audit breadcrumb (updated_at
            # bump — richer crash-marker columns tracked as a follow-up
            # migration: elspeth-23b0987938).
            if self._session_engine is not None and session_id is not None:
                try:
                    # Offload to a worker — _persist_crashed_session
                    # executes a synchronous SQLAlchemy ``Engine.begin()``
                    # + UPDATE, which would otherwise block the event
                    # loop for the duration of the DB round-trip,
                    # stalling websocket heartbeats, rate-limit checks,
                    # and concurrent progress broadcasts. Symmetric with
                    # the execute_tool offload at the top of
                    # _compose_loop: every other sync DB path in this
                    # file runs through run_sync_in_worker, and this
                    # crash-path call was missed when it was hoisted
                    # out of the main loop.
                    await run_sync_in_worker(self._persist_crashed_session, session_id)
                except (SQLAlchemyError, OSError) as audit_failure:
                    # Audit-persistence is best-effort on the crash path —
                    # failure to persist MUST NOT mask the original plugin
                    # bug. Log via slog.error (audit system itself is failing
                    # here, which is one of the three permitted slog use
                    # cases per the logging-telemetry-policy skill).
                    #
                    # Catch is narrowed to (SQLAlchemyError, OSError) so that
                    # programmer-bug exceptions in _persist_crashed_session
                    # itself — RuntimeError from the engine guard,
                    # AttributeError from a drifted table column, TypeError
                    # from a signature change — propagate instead of being
                    # laundered as "audit failure". Mirrors the cleanup-
                    # rollback pattern in the ``fork_from_message`` route
                    # handler (web/sessions/routes.py); see also the
                    # tier-model enforcer entry for this call site.
                    #
                    # exc_info is deliberately omitted: the original plugin
                    # exception's message / __cause__ chain may carry DB
                    # URLs, filesystem paths, or secret fragments from
                    # deeper layers (the response-body redaction in
                    # routes.py exists for the same reason). The two
                    # exc_class fields give the operator enough correlation
                    # to triage from structured logs alone.
                    slog.error(
                        "composer_crash_persistence_failed",
                        session_id=session_id,
                        original_exc_class=crash.exc_class,
                        audit_exc_class=type(audit_failure).__name__,
                    )
            await emit_progress(
                progress,
                ComposerProgressEvent(
                    phase="failed",
                    headline="The composer could not safely finish this request.",
                    evidence=("A pipeline tool failed on the server side.",),
                    likely_next="Review the visible error message, then retry after the issue is resolved.",
                    reason="plugin_crash",
                ),
            )
            raise
        except (ComposerServiceError, LiteLLMAPIError):
            # Generic service-level failure (prompt prep, availability check,
            # or a LiteLLMAPIError surfacing through the inner loop). The
            # route handlers further narrow LiteLLMAPIError into the
            # provider_unavailable progress code; here the service emits the
            # safe catch-all because we may not know which class fired.
            await emit_progress(
                progress,
                ComposerProgressEvent(
                    phase="failed",
                    headline="The composer could not finish this request.",
                    evidence=("The model call or prompt preparation failed safely.",),
                    likely_next="Retry once the composer service is available.",
                    reason="service_setup_failed",
                ),
            )
            raise

    async def _try_apply_freeform_recipe_intent(
        self,
        *,
        message: str,
        state: CompositionState,
        session_id: str | None,
        user_id: str | None,
        progress: ComposerProgressSink | None,
        user_message_id: str | None,
    ) -> ComposerResult | None:
        """Apply a deterministic registered recipe before invoking the cheap model.

        This is intentionally narrow: it only handles empty-state freeform
        requests that exactly match a server-known recipe and carry inline
        content that must first be materialised as a session blob. Existing
        non-empty pipelines and explicit-approval sessions continue through
        the normal LLM/tool proposal path.
        """
        if not _state_is_structurally_empty(state):
            return None
        if self._session_engine is None or session_id is None or self._data_dir is None:
            return None

        match = match_freeform_recipe_intent(message)
        if match is None or match.inline_blob is None:
            return None

        sessions_service = self._sessions_service
        if sessions_service is not None:
            preferences = await sessions_service.get_composer_preferences(UUID(session_id))
            if preferences.trust_mode == "explicit_approve":
                return None

        await emit_progress(
            progress,
            ComposerProgressEvent(
                phase="using_tools",
                headline="I found a registered recipe for this request.",
                evidence=(f"Recipe: {match.recipe_name}",),
                likely_next="ELSPETH will apply the recipe with the supplied inline data.",
            ),
        )

        actor = f"composer-web:user-{user_id}" if user_id is not None else "composer-web:anonymous"
        create_args = {
            "filename": match.inline_blob.filename,
            "mime_type": match.inline_blob.mime_type,
            "content": match.inline_blob.content,
            "description": f"Inline content materialised for recipe {match.recipe_name}",
        }
        create_result = await run_sync_in_worker(
            execute_tool,
            "create_blob",
            create_args,
            state,
            self._catalog,
            data_dir=self._data_dir,
            session_engine=self._session_engine,
            session_id=session_id,
            secret_service=self._secret_service,
            user_id=user_id,
            user_message_id=user_message_id,
            user_message_content=message,
        )
        if not create_result.success or not isinstance(create_result.data, Mapping):
            return None
        blob_id = create_result.data["blob_id"]

        recipe_args = {
            "recipe_name": match.recipe_name,
            "slots": {**match.slots, "source_blob_id": blob_id},
        }
        recipe_result = await run_sync_in_worker(
            execute_tool,
            "apply_pipeline_recipe",
            recipe_args,
            state,
            self._catalog,
            data_dir=self._data_dir,
            session_engine=self._session_engine,
            session_id=session_id,
            secret_service=self._secret_service,
            user_id=user_id,
        )
        if not recipe_result.success:
            return None

        # Record a compact synthetic audit trail for the deterministic server
        # routing decision. The actual tool handlers above still own state
        # validation and blob persistence; this trail makes the bypass visible.
        recorder = BufferingRecorder()
        create_audit, create_canonicalization_failed = begin_dispatch_or_arg_error(
            "server_recipe_create_blob",
            "create_blob",
            create_args,
            version_before=state.version,
            actor=actor,
        )
        if create_canonicalization_failed is None:
            recorder.record(
                finish_success(
                    create_audit,
                    result_payload=create_result.to_dict(),
                    version_after=create_result.updated_state.version,
                )
            )
        recipe_audit, recipe_canonicalization_failed = begin_dispatch_or_arg_error(
            "server_recipe_apply_pipeline_recipe",
            "apply_pipeline_recipe",
            recipe_args,
            version_before=state.version,
            actor=actor,
        )
        if recipe_canonicalization_failed is None:
            recorder.record(
                finish_success(
                    recipe_audit,
                    result_payload=recipe_result.to_dict(),
                    version_after=recipe_result.updated_state.version,
                )
            )

        return ComposerResult(
            message=(
                f"I built the `{match.recipe_name}` recipe and materialised the inline CSV as a session blob before wiring the pipeline."
            ),
            state=recipe_result.updated_state,
            runtime_preflight=None,
            tool_invocations=recorder.invocations,
            llm_calls=(),
        )

    async def _call_model_turn(
        self,
        *,
        llm_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        state: CompositionState,
        initial_version: int,
        deadline: float,
        recorder: BufferingRecorder,
        progress: ComposerProgressSink | None,
        message: str,
        composition_turns_used: int,
        discovery_turns_used: int,
    ) -> _CallModelOutcome:
        """Phase P1 of the compose loop — one LLM call with cap enforcement.

        Emits the model-call progress event, calls the provider before the
        cooperative deadline, and enforces ``_max_tool_calls_per_turn``.
        A cap breach raises :class:`ComposerConvergenceError` with the
        ``tool_call_cap_exceeded`` reason directly; no carrier is returned
        in that case.
        """
        await emit_progress(progress, model_call_progress_event(message))
        response = await self._call_llm_before_deadline(
            llm_messages,
            tools,
            state,
            initial_version,
            deadline,
            recorder=recorder,
        )
        assistant_message = response.choices[0].message
        raw_assistant_content = assistant_message.content
        assistant_tool_calls = assistant_message.tool_calls or ()
        if len(assistant_tool_calls) > self._max_tool_calls_per_turn:
            self._telemetry.tool_call_cap_exceeded_total.add(1)
            raise ComposerConvergenceError.capture(
                max_turns=composition_turns_used + discovery_turns_used,
                budget_exhausted="composition",
                state=state,
                initial_version=initial_version,
                tool_invocations=recorder.invocations,
                llm_calls=recorder.llm_calls,
                reason="tool_call_cap_exceeded",
                evidence={
                    "observed": len(assistant_tool_calls),
                    "cap": self._max_tool_calls_per_turn,
                },
            )
        return _CallModelOutcome(
            response=response,
            assistant_message=assistant_message,
            raw_assistant_content=raw_assistant_content,
            assistant_tool_calls=tuple(assistant_tool_calls),
            has_tool_calls=bool(assistant_message.tool_calls),
        )

    async def _persist_turn_audit(
        self,
        *,
        tool_outcomes: tuple[_ToolOutcome, ...],
        decoded_args_by_call_id: Mapping[str, Mapping[str, Any]],
        assistant_message: Any,
        raw_assistant_content: str | None,
        assistant_tool_calls: tuple[Any, ...],
        plugin_crash: ComposerPluginCrashError | None,
        session_id: str | None,
        current_state_id: str | None,
        persisted_tool_call_turn: bool,
        persisted_assistant_message_id: str | None,
    ) -> _PersistOutcome:
        """Phase P4 of the compose loop — delegates to :func:`turn_audit.persist_turn_audit`."""
        from elspeth.web.composer.turn_audit import persist_turn_audit

        return await persist_turn_audit(
            self,
            tool_outcomes=tool_outcomes,
            decoded_args_by_call_id=decoded_args_by_call_id,
            assistant_message=assistant_message,
            raw_assistant_content=raw_assistant_content,
            assistant_tool_calls=assistant_tool_calls,
            plugin_crash=plugin_crash,
            session_id=session_id,
            current_state_id=current_state_id,
            persisted_tool_call_turn=persisted_tool_call_turn,
            persisted_assistant_message_id=persisted_assistant_message_id,
        )

    async def _dispatch_tool_batch(
        self,
        *,
        call_model: _CallModelOutcome,
        state: CompositionState,
        last_validation: ValidationSummary | None,
        last_runtime_preflight: ValidationResult | None,
        llm_messages: list[dict[str, Any]],
        recorder: BufferingRecorder,
        anti_anchor: AntiAnchorTracker,
        discovery_cache: dict[str, _CachedDiscoveryPayload],
        runtime_preflight_cache: _RuntimePreflightCache,
        session_id: str | None,
        user_id: str | None,
        user_message_id: str | None,
        user_message_content: str | None,
        current_state_id: str | None,
        actor: str,
        initial_version: int,
        deadline: float,
        progress: ComposerProgressSink | None,
        session_scope: str,
        advisor_calls_used: int,
    ) -> tuple[_DispatchOutcome, int]:
        """Phase P3 of the compose loop — delegates to :func:`tool_batch.run_tool_batch`."""
        from elspeth.web.composer.tool_batch import (
            BatchAccumulator,
            ToolBatchContext,
            run_tool_batch,
        )

        turn_sessions_service = self._require_sessions_service() if session_id is not None else None
        turn_session_uuid = UUID(session_id) if session_id is not None else None
        turn_preferences = (
            await turn_sessions_service.get_composer_preferences(turn_session_uuid)
            if turn_sessions_service is not None and turn_session_uuid is not None
            else None
        )
        ctx = ToolBatchContext(
            service=self,
            recorder=recorder,
            anti_anchor=anti_anchor,
            discovery_cache=discovery_cache,
            runtime_preflight_cache=runtime_preflight_cache,
            session_id=session_id,
            user_id=user_id,
            user_message_id=user_message_id,
            user_message_content=user_message_content,
            current_state_id=current_state_id,
            actor=actor,
            initial_version=initial_version,
            deadline=deadline,
            progress=progress,
            session_scope=session_scope,
            turn_sessions_service=turn_sessions_service,
            turn_session_uuid=turn_session_uuid,
            turn_preferences=turn_preferences,
        )
        acc = BatchAccumulator(
            state=state,
            last_validation=last_validation,
            last_runtime_preflight=last_runtime_preflight,
            advisor_calls_used=advisor_calls_used,
        )
        return await run_tool_batch(
            call_model=call_model,
            ctx=ctx,
            acc=acc,
            llm_messages=llm_messages,
        )

    async def _classify_and_budget_turn(
        self,
        *,
        dispatch: _DispatchOutcome,
        persist: _PersistOutcome,
        llm_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        recorder: BufferingRecorder,
        anti_anchor: AntiAnchorTracker,
        progress: ComposerProgressSink | None,
        message: str,
        initial_version: int,
        deadline: float,
        runtime_preflight_cache: _RuntimePreflightCache,
        session_scope: str,
        user_id: str | None,
        mutation_success_seen: bool,
        composition_turns_used: int,
        discovery_turns_used: int,
    ) -> _ClassifyOutcome:
        """Phase P5 of the compose loop — anti-anchor + budget classify.

        Three concerns share this phase because their decision flow is
        sequential:

        1. **Anti-anchor hint (§7.7).** When the last three failed tool
           calls share the same (tool_name, arguments_hash), inject a
           role="user" hint into ``llm_messages`` so the model breaks
           the anchored retry. The hint is persisted via the normal
           ``chat_messages`` path.
        2. **Cache-hit short-circuit.** When every tool call this turn
           was a discovery cache hit, no budget charge: continue.
        3. **Budget classify.** Charge the composition counter (with
           the B-4D-3 last-chance LLM call on exhaustion) or the
           discovery counter (no bonus call). Advisor-only turns are
           neither — return to the driver without charging.

        Returns:
            ``_ClassifyOutcome(action="continue", composition_turns_delta=...,
            discovery_turns_delta=...)`` on the normal path, or
            ``_ClassifyOutcome(action="return", result=...)`` when the
            B-4D-3 bonus call terminated the loop. Convergence raises
            (composition / discovery budget exhausted without bonus
            success) leave through the exception channel.
        """
        state = dispatch.state
        last_runtime_preflight = dispatch.last_runtime_preflight
        turn_has_mutation = dispatch.turn_has_mutation
        turn_has_discovery = dispatch.turn_has_discovery
        all_cache_hits = dispatch.all_cache_hits
        persisted_tool_call_turn = persist.persisted_tool_call_turn
        persisted_assistant_message_id = persist.persisted_assistant_message_id
        failed_turn = persist.failed_turn

        # §7.7 anti-anchor hint: if the last 3 failed tool calls share the
        # same (tool_name, arguments_hash), the model has stopped reading
        # validator feedback. Inject a synthetic role="user" hint before
        # the next LLM turn so the model breaks the anchor. consume_fire()
        # clears the deque so the hint cannot re-fire on the same anchor.
        # Persisted via the normal llm_messages → chat_messages path; the
        # operator-visible audit row carries the [ELSPETH-SYSTEM-HINT]
        # marker so its system origin is unambiguous.
        if anti_anchor.should_fire():
            hint_text = anti_anchor.build_hint()
            anti_anchor.consume_fire()
            llm_messages.append({"role": "user", "content": hint_text})
            await emit_progress(
                progress,
                ComposerProgressEvent(
                    phase="using_tools",
                    headline="ELSPETH detected an anchored retry pattern.",
                    evidence=(
                        "The last 3 tool calls used identical arguments and produced the same error.",
                        "A structural hint was injected to help the model converge.",
                    ),
                    likely_next="The model will see the hint and try a different argument shape.",
                ),
            )

        # If ALL tool calls in this turn were cache hits, no budget
        # charge — continue to next turn without incrementing.
        if all_cache_hits:
            return _ClassifyOutcome(action="continue")

        # Classify turn and charge the appropriate budget.
        # The current turn has already been executed (tool results
        # are in the message history). We increment first, then
        # check whether the budget is now exhausted. If so, we give
        # the LLM one last chance (B-4D-3) for composition, or
        # raise immediately for discovery (discovery exhaustion
        # doesn't benefit from a bonus call — no state was mutated).
        if turn_has_mutation:
            new_composition_turns_used = composition_turns_used + 1
            if new_composition_turns_used >= self._max_composition_turns:
                # B-4D-3 fix: give the LLM one last chance to see the
                # tool results and produce a text response.
                await emit_progress(progress, model_call_progress_event(message))
                response = await self._call_llm_before_deadline(
                    llm_messages,
                    tools,
                    state,
                    initial_version,
                    deadline,
                    recorder=recorder,
                )
                assistant_message = response.choices[0].message
                if not assistant_message.tool_calls:
                    await emit_progress(
                        progress,
                        ComposerProgressEvent(
                            phase="complete",
                            headline="The composer response is ready.",
                            evidence=("The model stopped requesting pipeline tools.",),
                            likely_next="ELSPETH will save any accepted pipeline update.",
                            reason="composer_complete",
                        ),
                    )
                    result = await self._finalize_no_tool_response(
                        content=assistant_message.content or "",
                        state=state,
                        initial_version=initial_version,
                        user_id=user_id,
                        last_runtime_preflight=last_runtime_preflight,
                        runtime_preflight_cache=runtime_preflight_cache,
                        session_scope=session_scope,
                        user_message=message,
                        mutation_success_seen=mutation_success_seen,
                        tool_invocations=recorder.invocations,
                        llm_calls=recorder.llm_calls,
                    )
                    threaded = replace(
                        result,
                        persisted_assistant_message_id=persisted_assistant_message_id,
                        persisted_tool_call_turn=persisted_tool_call_turn,
                    )
                    return _ClassifyOutcome(
                        action="return",
                        result=threaded,
                        composition_turns_delta=1,
                    )
                raise ComposerConvergenceError.capture(
                    max_turns=new_composition_turns_used + discovery_turns_used,
                    budget_exhausted="composition",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=() if persisted_tool_call_turn else recorder.invocations,
                    llm_calls=recorder.llm_calls,
                    failed_turn=failed_turn,
                )
            return _ClassifyOutcome(action="continue", composition_turns_delta=1)
        if turn_has_discovery:
            new_discovery_turns_used = discovery_turns_used + 1
            if new_discovery_turns_used >= self._max_discovery_turns:
                raise ComposerConvergenceError.capture(
                    max_turns=composition_turns_used + new_discovery_turns_used,
                    budget_exhausted="discovery",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=() if persisted_tool_call_turn else recorder.invocations,
                    llm_calls=recorder.llm_calls,
                    failed_turn=failed_turn,
                )
            return _ClassifyOutcome(action="continue", discovery_turns_delta=1)
        # The only non-cache tool currently handled outside the
        # discovery/mutation registries is request_advisor_hint. It
        # has its own per-compose budget above, so give the primary
        # model the returned guidance instead of charging discovery.
        return _ClassifyOutcome(action="continue")

    async def _try_terminate_no_tools(
        self,
        *,
        assistant_message: Any,
        message: str,
        llm_messages: list[dict[str, Any]],
        state: CompositionState,
        session_id: str | None,
        initial_version: int,
        user_id: str | None,
        last_runtime_preflight: ValidationResult | None,
        runtime_preflight_cache: _RuntimePreflightCache,
        session_scope: str,
        mutation_success_seen: bool,
        recorder: BufferingRecorder,
        progress: ComposerProgressSink | None,
        repair_turns_used: int,
        persisted_assistant_message_id: str | None,
        persisted_tool_call_turn: bool,
    ) -> _TerminateOutcome:
        """Phase P2 of the compose loop — handle the no-tool-calls branch.

        Called only when the assistant emitted no tool calls. Either:

        * Appends a repair-prompt to ``llm_messages`` and returns a
          ``_TerminateOutcome(action="continue", repair_turns_delta=1)``,
          asking the driver to bump its repair counter and re-enter P1.
        * Or finalizes the response via ``_finalize_no_tool_response`` and
          returns ``_TerminateOutcome(action="return", result=...)``. The
          ``result`` is already threaded with ``repair_turns_used``,
          ``persisted_assistant_message_id`` and ``persisted_tool_call_turn``
          so the driver only has to ``return outcome.result``.
        """
        if (
            repair_turns_used < _MAX_REPAIR_TURNS
            and _user_request_expects_pipeline_mutation(message)
            and _state_is_structurally_empty(state)
            and _last_failure_was_pre_state_interpretation_review(recorder.invocations)
        ):
            llm_messages.append(
                {
                    "role": "user",
                    "content": _pre_state_interpretation_review_repair_message(
                        next_turn=repair_turns_used + 1,
                    ),
                }
            )
            return _TerminateOutcome(action="continue", repair_turns_delta=1)

        if repair_turns_used < _MAX_REPAIR_TURNS:
            missing_interpretation_sites = await self._missing_pending_interpretation_review_sites(
                state,
                session_id=session_id,
            )
            if missing_interpretation_sites:
                llm_messages.append(
                    {
                        "role": "user",
                        "content": _pending_interpretation_review_repair_message(
                            missing_interpretation_sites,
                            next_turn=repair_turns_used + 1,
                        ),
                    }
                )
                return _TerminateOutcome(action="continue", repair_turns_delta=1)

        if await self._attempt_empty_state_uploaded_blob_repair(
            state=state,
            llm_messages=llm_messages,
            session_id=session_id,
            repair_turns_used=repair_turns_used,
        ):
            return _TerminateOutcome(action="continue", repair_turns_delta=1)

        # Forced-repair gate: when the model claims completion but
        # the proof step still has blocking diagnostics, inject a
        # repair message and continue. Capped at _MAX_REPAIR_TURNS so
        # the loop can never spin indefinitely. NEVER catches plugin
        # exceptions — only repairs configurations.
        #
        # The gate fires whenever the proof step is applicable —
        # i.e. there is a blob-backed source to inspect. The earlier
        # ``state.version > initial_version`` guard skipped the gate
        # on the first compose turn of a resumed session whose
        # blob-backed source was bound on a prior turn (state already
        # carries the source, no mutation this turn). That is exactly
        # the cross-turn scenario the gate exists to catch (e.g.
        # ``csv_fixed_schema_omits_observed_columns`` blockers
        # surviving session resume). For chat-only turns where the
        # source is absent or not blob-backed, ``_attempt_proof_repair``
        # short-circuits cheaply via ``compute_proof_diagnostics``'s
        # own early return.
        if _proof_repair_is_applicable(state) and self._attempt_proof_repair(
            state=state,
            llm_messages=llm_messages,
            session_id=session_id,
            repair_turns_used=repair_turns_used,
        ):
            return _TerminateOutcome(action="continue", repair_turns_delta=1)

        await emit_progress(
            progress,
            ComposerProgressEvent(
                phase="complete",
                headline="The composer response is ready.",
                evidence=("The model did not request any more pipeline tools.",),
                likely_next="ELSPETH will save any accepted pipeline update.",
                reason="composer_complete",
            ),
        )
        result = await self._finalize_no_tool_response(
            content=assistant_message.content or "",
            state=state,
            initial_version=initial_version,
            user_id=user_id,
            last_runtime_preflight=last_runtime_preflight,
            runtime_preflight_cache=runtime_preflight_cache,
            session_scope=session_scope,
            user_message=message,
            mutation_success_seen=mutation_success_seen,
            tool_invocations=recorder.invocations,
            llm_calls=recorder.llm_calls,
        )
        # Thread repair_turns_used through to the result so the
        # route handler can persist it onto the new
        # ``composition_states.composer_meta`` row (and the API state
        # response can surface ``composer_meta.repair_turns_used``)
        # — see web/sessions/routes.py::_state_data_from_composer_state
        # call sites in the compose / recompose paths.
        threaded = replace(
            result,
            repair_turns_used=repair_turns_used,
            persisted_assistant_message_id=persisted_assistant_message_id,
            persisted_tool_call_turn=persisted_tool_call_turn,
        )
        return _TerminateOutcome(action="return", result=threaded)

    async def _compose_loop(
        self,
        message: str,
        messages: list[dict[str, Any]],
        state: CompositionState,
        session_id: str | None = None,
        initial_current_state_id: str | None = None,
        user_id: str | None = None,
        deadline: float = 0.0,
        progress: ComposerProgressSink | None = None,
        guided_terminal: TerminalState | None = None,
        user_message_id: str | None = None,
    ) -> ComposerResult:
        """Inner composition loop with dual-counter budget tracking.

        The loop body is decomposed into five phases (see the carrier
        module ``_compose_loop_carriers`` for the dataclasses that
        thread state between them):

        * P1 :meth:`_call_model_turn`        — one LLM call, cap check
        * P2 :meth:`_try_terminate_no_tools` — handle the no-tool-calls
          branch (repair injections or finalize-and-return)
        * P3 :meth:`_dispatch_tool_batch`    — execute every tool call,
          accumulate ``_ToolOutcome`` records, rebind ``state``
        * P4 :meth:`_persist_turn_audit`     — redact, persist the turn
          audit row, raise plugin-crash propagation if applicable
        * P5 :meth:`_classify_and_budget_turn` — anti-anchor hint,
          cache-hit short-circuit, dual-counter budget classify,
          B-4D-3 last-chance LLM call

        Uses cooperative timeout: the deadline is checked at safe
        checkpoints (before LLM calls, after tool batches) rather
        than using asyncio.wait_for() cancellation.  This ensures
        tool calls that have filesystem/DB side effects always run
        to completion with their state published — no split between
        committed side effects and the response.

        LLM calls are wrapped in per-call asyncio.wait_for(remaining)
        because they are pure network I/O with no side effects and
        can be safely cancelled.

        Args:
            guided_terminal: When set, this is the first freeform turn after
                guided-mode exit; the layered transition prompt is used.
        """
        initial_version = state.version
        # F-5c. On the first compose-loop entry of this service instance,
        # upsert the composer skill markdown into
        # ``skill_markdown_history`` so an auditor inspecting a future
        # interpretation_events row can join via ``composer_skill_hash``
        # to retrieve the exact text the LLM was prompted with. The
        # ``INSERT OR IGNORE`` semantics make repeated calls cheap; we
        # still gate behind a per-instance flag so steady-state compose()
        # calls don't churn the connection pool.
        await self._maybe_upsert_skill_markdown_history()
        llm_messages = self._build_messages(messages, state, message, guided_terminal, session_id=session_id)
        tools = self._get_litellm_tools()
        # Per-call audit recorder. Surfaced on ComposerResult and on
        # the three partial-state-carrier exceptions so the route handler
        # always has the per-call decision trail — including failure paths.
        recorder = BufferingRecorder()
        # Stable actor string for every invocation in this compose() call.
        # Falls back to "anonymous" when user_id is None (CLI/test paths);
        # the real web composer always has user_id from auth dependency.
        actor = f"composer-web:user-{user_id}" if user_id is not None else "composer-web:anonymous"
        await emit_progress(
            progress,
            ComposerProgressEvent(
                phase="starting",
                headline="I'm reading your request and current pipeline.",
                evidence=(
                    "The current pipeline state is prepared for the composer.",
                    "The pipeline composer skill pack and deployment overlay are included.",
                ),
                likely_next="ELSPETH will ask the model for the next safe pipeline action.",
            ),
        )

        composition_turns_used = 0
        discovery_turns_used = 0
        mutation_success_seen = False

        # Discovery cache: local variable scoped to this compose() call.
        # Keyed by (tool_name, canonical_args_json). Each concurrent
        # compose() call gets its own independent cache dict.
        discovery_cache: dict[str, _CachedDiscoveryPayload] = {}

        # Validation threading: compute once for the initial state, then
        # carry forward from each ToolResult.validation. Avoids redundant
        # validate() calls — CompositionState is immutable so validation
        # is deterministic for a given state object.
        last_validation: ValidationSummary | None = None

        # Runtime preflight cache: scoped to this compose() call. Keyed by
        # (session_scope, state_version, settings_hash). A timeout or failure
        # is cached for the lifetime of this compose call so subsequent
        # preview_pipeline calls don't re-fire an already-failed worker.
        runtime_preflight_cache = self._new_runtime_preflight_cache()
        last_runtime_preflight: ValidationResult | None = None
        session_scope = f"session:{session_id}" if session_id is not None else "session:unsaved"

        # §7.7 anti-anchor tracker: detects 3-in-a-row identical failed tool
        # calls and injects a STRUCTURAL HINT before the next LLM turn so the
        # model breaks out of the anchored-loop pattern observed in the Tier 1
        # final cohort's residual RED. Per-compose-call instance — never
        # shared across requests.
        anti_anchor = AntiAnchorTracker()

        # Advisor escape-hatch budget. Local to this compose() call —
        # each fresh user request starts with the full configured budget.
        # Per-compose-request scope (matching the setting name
        # ``composer_advisor_max_calls_per_compose``) is the useful budget:
        # an LLM that breaks out of an anchored loop in one request should
        # not have its budget penalised in the next. There is intentionally
        # no session-lifetime cap; ``composer_rate_limit_per_minute`` and
        # the per-compose budget together bound advisor cost. When the
        # toggle is disabled the counter is never read.
        advisor_calls_used = 0

        # Forced-repair counter. When the assistant emits no tool_calls but
        # the proof step found blocking diagnostics, the loop synthesises a
        # repair message and continues for at most _MAX_REPAIR_TURNS
        # additional iterations. NEVER catches plugin exceptions — only
        # configuration diagnostics.
        repair_turns_used = 0
        persisted_assistant_message_id: str | None = None
        persisted_tool_call_turn = False
        failed_turn: FailedTurnMetadata | None = None
        current_state_id: str | None = initial_current_state_id

        while True:
            # The compose-loop audit path captures the state id observed
            # before the provider call and passes this exact value to
            # persist_compose_turn_async as expected_current_state_id.
            call_model = await self._call_model_turn(
                llm_messages=llm_messages,
                tools=tools,
                state=state,
                initial_version=initial_version,
                deadline=deadline,
                recorder=recorder,
                progress=progress,
                message=message,
                composition_turns_used=composition_turns_used,
                discovery_turns_used=discovery_turns_used,
            )
            # If no tool calls, the LLM is done — apply the final gate and return
            if not call_model.has_tool_calls:
                terminate = await self._try_terminate_no_tools(
                    assistant_message=call_model.assistant_message,
                    message=message,
                    llm_messages=llm_messages,
                    state=state,
                    session_id=session_id,
                    initial_version=initial_version,
                    user_id=user_id,
                    last_runtime_preflight=last_runtime_preflight,
                    runtime_preflight_cache=runtime_preflight_cache,
                    session_scope=session_scope,
                    mutation_success_seen=mutation_success_seen,
                    recorder=recorder,
                    progress=progress,
                    repair_turns_used=repair_turns_used,
                    persisted_assistant_message_id=persisted_assistant_message_id,
                    persisted_tool_call_turn=persisted_tool_call_turn,
                )
                if terminate.action == "return":
                    # Offensive guard (explicit raise, not assert): ``python -O``
                    # strips assert statements. The contract between
                    # ``_dispatch_terminate_phase`` and this caller is that
                    # ``result`` is non-None whenever ``action == "return"``;
                    # a None here would be a compose-loop bug, not a recoverable
                    # state. Routed to the HTTP-500 static-detail handler at
                    # ``routes/composer.py:905`` via :class:`InvariantError`
                    # (B1-sanitised response body).
                    if terminate.result is None:
                        raise InvariantError(
                            "_dispatch_terminate_phase returned action='return' with result=None — "
                            "the terminate-phase contract requires result to be set whenever the "
                            "phase signals a return."
                        )
                    return terminate.result
                repair_turns_used += terminate.repair_turns_delta
                continue

            dispatch, advisor_calls_used = await self._dispatch_tool_batch(
                call_model=call_model,
                state=state,
                last_validation=last_validation,
                last_runtime_preflight=last_runtime_preflight,
                llm_messages=llm_messages,
                recorder=recorder,
                anti_anchor=anti_anchor,
                discovery_cache=discovery_cache,
                runtime_preflight_cache=runtime_preflight_cache,
                session_id=session_id,
                user_id=user_id,
                user_message_id=user_message_id,
                user_message_content=message,
                current_state_id=current_state_id,
                actor=actor,
                initial_version=initial_version,
                deadline=deadline,
                progress=progress,
                session_scope=session_scope,
                advisor_calls_used=advisor_calls_used,
            )
            # State the driver still owns across iterations updates from
            # the dispatch carrier; persist + classify consume the rest
            # of the dispatch fields directly.
            state = dispatch.state
            last_validation = dispatch.last_validation
            last_runtime_preflight = dispatch.last_runtime_preflight
            if dispatch.mutation_success_observed:
                mutation_success_seen = True
            self._phase3_last_tool_outcomes = dispatch.tool_outcomes
            persist = await self._persist_turn_audit(
                tool_outcomes=dispatch.tool_outcomes,
                decoded_args_by_call_id=dispatch.decoded_args_by_call_id,
                assistant_message=dispatch.assistant_message,
                raw_assistant_content=dispatch.raw_assistant_content,
                assistant_tool_calls=dispatch.assistant_tool_calls,
                plugin_crash=dispatch.plugin_crash,
                session_id=session_id,
                current_state_id=current_state_id,
                persisted_tool_call_turn=persisted_tool_call_turn,
                persisted_assistant_message_id=persisted_assistant_message_id,
            )
            current_state_id = persist.current_state_id
            persisted_assistant_message_id = persist.persisted_assistant_message_id
            persisted_tool_call_turn = persist.persisted_tool_call_turn
            failed_turn = persist.failed_turn
            if dispatch.plugin_crash is not None:
                # Plugin-crash propagation discipline (plan §5.7): the
                # capture in P3 already snapshotted `state` after every
                # prior successful tool-call mutation. If persistence
                # succeeded, re-capture with the post-persist failed_turn
                # so the route layer's _handle_plugin_crash sees the
                # complete partial-state story; otherwise raise the
                # original capture as-is.
                if persisted_tool_call_turn:
                    persisted_plugin_crash = ComposerPluginCrashError.capture(
                        dispatch.plugin_crash.original_exc,
                        state=state,
                        initial_version=initial_version,
                        tool_invocations=(),
                        llm_calls=recorder.llm_calls,
                        failed_turn=failed_turn,
                    )
                    if dispatch.plugin_crash_cause is None:
                        raise persisted_plugin_crash
                    raise persisted_plugin_crash from dispatch.plugin_crash_cause
                if dispatch.plugin_crash_cause is None:
                    raise dispatch.plugin_crash
                raise dispatch.plugin_crash from dispatch.plugin_crash_cause

            classify = await self._classify_and_budget_turn(
                dispatch=dispatch,
                persist=persist,
                llm_messages=llm_messages,
                tools=tools,
                recorder=recorder,
                anti_anchor=anti_anchor,
                progress=progress,
                message=message,
                initial_version=initial_version,
                deadline=deadline,
                runtime_preflight_cache=runtime_preflight_cache,
                session_scope=session_scope,
                user_id=user_id,
                mutation_success_seen=mutation_success_seen,
                composition_turns_used=composition_turns_used,
                discovery_turns_used=discovery_turns_used,
            )
            composition_turns_used += classify.composition_turns_delta
            discovery_turns_used += classify.discovery_turns_delta
            if classify.action == "return":
                # Offensive guard (explicit raise, not assert): ``python -O``
                # strips assert statements. The contract between
                # ``_dispatch_classify_phase`` and this caller is that
                # ``result`` is non-None whenever ``action == "return"``
                # (the B-4D-3 last-chance branch sets it). A None here would
                # be a compose-loop bug. Routed to the HTTP-500 static-detail
                # handler at ``routes/composer.py:905`` via
                # :class:`InvariantError` (B1-sanitised response body).
                if classify.result is None:
                    raise InvariantError(
                        "_dispatch_classify_phase returned action='return' with result=None — "
                        "the classify-phase contract requires result to be set whenever the "
                        "phase signals a return."
                    )
                return classify.result
            continue

    def _persist_crashed_session(self, session_id: str) -> None:
        """Best-effort timestamp bump to mark that a compose session crashed.

        NOTE: The sessions-table schema does not yet have a dedicated crash
        marker column. Bumping updated_at is the minimum viable breadcrumb
        until a migration adds (e.g.) a ``status`` or ``crashed_at`` column.
        The schema addition is tracked separately as elspeth-23b0987938;
        when that lands, this method expands to populate the new columns
        and its signature gains ``exc_class``.

        The crash's exc_class is NOT written to the session row — no column
        exists to hold it. The operator correlates the updated_at bump with
        the crash via the slog.error emission at the call site, which
        includes session_id and exc_class in structured fields.

        Signature intentionally minimal — only the data that actually gets
        persisted is accepted. When the schema migration lands, this
        method's signature expands to take last_state and exc_class, and
        callers are updated at that point. Today, the caller passes
        session_id and logs the rest via slog.

        The caller's outer try/except absorbs any failure — this method
        MUST NOT mask the original plugin-bug exception if persistence
        itself fails.
        """
        # Offensive guard (explicit raise, not assert): ``python -O`` strips
        # assert statements, so a caller that somehow reaches this method
        # with ``_session_engine is None`` would silently no-op under the
        # optimised interpreter — turning a recoverable audit failure into
        # a missed ``updated_at`` write with no trace.  A typed
        # ``RuntimeError`` always fires.
        if self._session_engine is None:
            raise RuntimeError("_persist_crashed_session must only be called when session_engine is set")
        now = datetime.now(UTC)
        with self._session_engine.begin() as conn:
            conn.execute(update(sessions_table).where(sessions_table.c.id == session_id).values(updated_at=now))

    def _schemas_loaded_for_session(self, session_id: str | None) -> frozenset[tuple[str, str]]:
        """Return the immutable view of plugins whose schema has loaded.

        Returns an empty frozenset when ``session_id`` is None (the
        unsaved-session fast path) or when no ``get_plugin_schema`` call
        has yet succeeded for this session. The returned frozenset is a
        snapshot — subsequent ``_mark_plugin_schema_loaded`` calls do not
        mutate it.
        """
        if session_id is None:
            return frozenset()
        if session_id not in self._schemas_loaded_by_session:
            return frozenset()
        return frozenset(self._schemas_loaded_by_session[session_id])

    def _mark_plugin_schema_loaded(
        self,
        session_id: str | None,
        plugin_type: str,
        plugin_name: str,
    ) -> None:
        """Record that ``get_plugin_schema`` returned successfully for this plugin.

        No-op when ``session_id`` is None (unsaved sessions have no
        persistent identity for the tracker; the next turn would not see
        the marking anyway).
        """
        if session_id is None:
            return
        if session_id not in self._schemas_loaded_by_session:
            self._schemas_loaded_by_session[session_id] = set()
        self._schemas_loaded_by_session[session_id].add((plugin_type, plugin_name))

    def _build_messages(
        self,
        chat_history: list[dict[str, Any]],
        state: CompositionState,
        user_message: str,
        guided_terminal: TerminalState | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the message list. Returns a NEW list on every call.

        This is critical: the tool-use loop appends to this list during
        iteration. Returning a cached reference would cause cross-turn
        contamination.

        OSError from deployment skill loading (PermissionError,
        IsADirectoryError) is translated into ComposerServiceError so
        the route handler returns a structured 502 rather than a raw 500.

        The HTTP body carries only ``type(exc).__name__`` — NOT
        ``str(exc)`` — because ``OSError.__str__`` expands to a string
        that includes the absolute filename (``[Errno 13] Permission
        denied: '/var/lib/elspeth/data/skills/...'``) which would
        leak filesystem layout and the operator's data-dir path into
        the 502 response body.  Full detail including the filename is
        preserved via ``raise ... from exc`` for the ASGI / server-log
        machinery only.  Mirrors the redaction contract landed by
        commits 1a30d985 (SQLAlchemy 422 path) and 127417cb (sibling
        HTTP-path slog sites) — both narrow the HTTP surface to
        class-name-only while preserving structured server-side detail.

        Args:
            guided_terminal: When set, forward to ``build_messages`` so the
                layered mode-transition prompt is used for this turn.
        """
        try:
            return build_messages(
                chat_history=chat_history,
                state=state,
                user_message=user_message,
                catalog=self._catalog,
                data_dir=self._data_dir,
                advisor_enabled=self._settings.composer_advisor_enabled,
                guided_terminal=guided_terminal,
                schemas_loaded=self._schemas_loaded_for_session(session_id),
            )
        except OSError as exc:
            raise ComposerServiceError(f"Failed to load deployment skill ({type(exc).__name__})") from exc

    def _get_litellm_tools(self) -> list[dict[str, Any]]:
        """Convert tool definitions to LiteLLM function format.

        When ``composer_advisor_enabled`` is False (the default), the
        ``request_advisor_hint`` tool is filtered out of the LLM-visible
        list. This is the strongest off-switch available — the composer
        LLM never sees the tool name or description, so it cannot call
        it even if instructed to. The CLI MCP server (composer_mcp/) is
        not affected; advisor is web-composer only by design.
        """
        definitions = get_tool_definitions()
        if not self._settings.composer_advisor_enabled:
            definitions = [defn for defn in definitions if defn["name"] != "request_advisor_hint"]
        return [
            {
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn["description"],
                    "parameters": defn["parameters"],
                },
            }
            for defn in definitions
        ]

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> Any:
        """Call the LLM via LiteLLM. Separated for test mocking."""
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "tools": tools,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
        except LiteLLMBadRequestError as exc:
            raise _BadRequestLLMError(
                f"LLM request rejected ({type(exc).__name__})",
                provider_detail=str(exc) or None,
                provider_status_code=exc.status_code,
            ) from exc
        # Tier 3 boundary: LiteLLM can return empty choices on content-filter,
        # rate-limit, or malformed upstream responses.  Validate before callers
        # index into choices[0].
        if not response.choices:
            raise _MalformedLLMResponseError("LLM returned empty choices array — cannot continue composition", response=response)
        return response

    async def _call_text_llm(
        self,
        messages: list[dict[str, str]],
    ) -> Any:
        """Call the LLM for non-tool text generation."""
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        try:
            seed = _composer_llm_seed_for_model(self._model)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": messages,
                "temperature": _COMPOSER_LLM_TEMPERATURE,
            }
            if seed is not None:
                kwargs[_COMPOSER_LLM_SEED_PARAM] = seed
            response = await _litellm_acompletion(
                **kwargs,
            )
        except LiteLLMBadRequestError as exc:
            raise _BadRequestLLMError(
                f"LLM request rejected ({type(exc).__name__})",
                provider_detail=str(exc) or None,
                provider_status_code=exc.status_code,
            ) from exc
        if not response.choices:
            raise _MalformedLLMResponseError("LLM returned empty choices array — cannot explain run diagnostics", response=response)
        return response

    def _validate_advisor_arguments(self, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Validate advisor tool arguments at the Tier-3 trust boundary.

        Returns ``None`` if valid; otherwise returns an ARG_ERROR payload
        ready to embed in the outer tool-result envelope.

        The compose-loop's ``_TOOL_REQUIRED_PATHS`` check upstream guarantees
        ``trigger``, ``problem_summary``, ``recent_errors``, and
        ``attempted_actions`` are present in ``arguments`` — but only their
        *presence*, not their
        type or size. Without this validator:

        - A non-list ``recent_errors`` would be silently iterated by Python
          (string → char-by-char, int → TypeError, dict → keys), producing
          a corrupt prompt that we would still pay full provider cost for.
        - A megabyte-scale value would be sent verbatim to LiteLLM,
          rendering ``composer_advisor_max_prompt_tokens`` (declared as a
          cap) into dead config — operators would believe they had a cap
          and they would not.

        Both are Tier-3 trust-boundary failures: the LLM is providing
        external input, and CLAUDE.md's tier model permits ``isinstance``
        checks (and other defensive validation) at this boundary. Anti-
        anchor tracking on the caller side ensures repeated identical
        ARG_ERRORs surface the §7.7 structural hint.
        """
        trigger = arguments["trigger"]
        if not isinstance(trigger, str):
            return {
                "status": "ARG_ERROR",
                "error": "trigger must be a string",
                "error_class": "TypeError",
            }
        if trigger not in ADVISOR_TRIGGER_VALUES:
            return {
                "status": "ARG_ERROR",
                "error": f"trigger must be one of: {', '.join(ADVISOR_TRIGGER_VALUES)}",
                "error_class": "ValueError",
            }

        if not isinstance(arguments["problem_summary"], str):
            return {
                "status": "ARG_ERROR",
                "error": "problem_summary must be a string",
                "error_class": "TypeError",
            }

        recent = arguments["recent_errors"]
        if not isinstance(recent, list) or not all(isinstance(e, str) for e in recent):
            return {
                "status": "ARG_ERROR",
                "error": "recent_errors must be a list of strings",
                "error_class": "TypeError",
            }

        attempted = arguments["attempted_actions"]
        if not isinstance(attempted, list) or not all(isinstance(a, str) for a in attempted):
            return {
                "status": "ARG_ERROR",
                "error": "attempted_actions must be a list of strings",
                "error_class": "TypeError",
            }

        if trigger == ADVISOR_TRIGGER_REACTIVE and (len(recent) < 2 or len(attempted) < 2):
            return {
                "status": "ARG_ERROR",
                "error": (
                    "reactive_validation_loop trigger requires at least two recent_errors "
                    "and two attempted_actions showing the unchanged validation loop"
                ),
                "error_class": "ValueError",
            }

        if "schema_excerpt" in arguments and arguments["schema_excerpt"] is not None:
            candidate = arguments["schema_excerpt"]
            if not isinstance(candidate, str):
                return {
                    "status": "ARG_ERROR",
                    "error": "schema_excerpt must be a string when provided",
                    "error_class": "TypeError",
                }

        # Approximate provider cost cap: rough 4 chars / token. Compute the
        # exact formatted user-message char count we would emit if the call
        # proceeded, including section labels, bullets, and newlines. The
        # fixed system side is bounded separately by the packaged skill plus
        # load_deployment_skill's byte cap; this setting bounds the
        # LLM-controlled variable part.
        total_chars = len(_build_advisor_user_message(arguments))
        char_cap = self._settings.composer_advisor_max_prompt_tokens * 4
        if total_chars > char_cap:
            return {
                "status": "ARG_ERROR",
                "error": (
                    f"prompt size {total_chars} chars exceeds cap {char_cap} chars "
                    f"(composer_advisor_max_prompt_tokens={self._settings.composer_advisor_max_prompt_tokens}). "
                    "Truncate your error/action lists or schema excerpt and retry."
                ),
                "error_class": "ValueError",
            }

        return None

    async def _dispatch_session_aware_tool(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        state: CompositionState,
        audit: DispatchAudit,
        recorder: BufferingRecorder,
        session_id: str | None,
        current_state_id: str | None,
        response: Any,
        llm_messages: list[dict[str, Any]],
        anti_anchor: AntiAnchorTracker,
    ) -> _SessionAwareDispatchOutcome:
        """Dispatch a session-aware async composer tool.

        Mirrors the structural envelope discipline of ``dispatch_with_audit``
        used for the sync ``execute_tool`` path:

        * SUCCESS → ``finish_success`` and a serialized ToolResult appended
          to ``llm_messages``.
        * ARG_ERROR (generic) → ``finish_arg_error`` and the standard
          ``_arg_error_payload`` echo.
        * ARG_ERROR with ``code in RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE``
          (F-6 / F-15) → emit ``interpretation_rate_cap_exceeded`` operational
          telemetry, await ``record_auto_interpreted_no_surfaces_event`` to
          write the AUTO_INTERPRETED_NO_SURFACES audit row, THEN
          ``finish_arg_error`` and echo the standard ARG_ERROR payload so
          the LLM is nudged into the fallback path from the composer
          skill (bake the interpretation directly into the prompt
          template).
        * Plugin crash → propagate; outer compose loop wraps with
          ``ComposerPluginCrashError`` exactly as for the sync path.

        Pre-conditions:

        * ``session_id`` is not None — session-aware tools are reachable
          only from authenticated compose-loop calls. ``RuntimeError`` is
          raised on a missing session id (interpreter-level invariant,
          not Tier-3).
        * ``current_state_id`` is not None for tools that need a
          composition_state foreign key (currently every session-aware
          tool). If the LLM calls the tool before a successful state-staging
          tool has created that row, this returns ARG_ERROR so the model can
          retry after staging the state instead of crashing the request.

        Per-tool dispatch is performed by reading the handler from
        ``_SESSION_AWARE_TOOL_HANDLERS`` and awaiting it with the
        keyword-arguments dict built by ``_build_session_aware_kwargs``.
        Adding a new session-aware tool extends that dict; this dispatch
        method itself does not need to change shape.
        """
        if session_id is None:
            # Compose-loop invariant: session-aware tools are advertised
            # to the LLM only when the loop is running against a
            # persisted session. Reaching this branch with no
            # ``session_id`` means the LLM somehow named a session-aware
            # tool in an unsaved-session compose call. That is a
            # plumbing bug, not a Tier-3 LLM error, so crash with a
            # diagnostic message.
            raise RuntimeError(
                f"Session-aware tool {tool_name!r} dispatched without a session_id. "
                f"_get_litellm_tools() should not advertise session-aware tools to "
                f"the LLM on unsaved-session compose calls."
            )
        if current_state_id is None:
            # Fresh chat sessions legitimately start without a
            # composition_states row. A session-aware tool can only write
            # its audit row after a successful state-staging tool
            # (set_pipeline/upsert_node/etc.) has advanced and persisted
            # the state. Treat an earlier call as LLM-correctable
            # sequencing, not a server crash: the request reached this
            # branch through a valid authenticated compose session, but
            # the LLM called the review tool before its FK target exists.
            exc = ToolArgumentError(
                argument="composition_state_id",
                expected=(
                    "a persisted composition state; call set_pipeline or another "
                    "state-staging tool successfully, wait for its tool result, "
                    "then call request_interpretation_review"
                ),
                actual_type="missing current_state_id",
            )
            arg_error_payload = _arg_error_payload(exc, tool_name)
            recorder.record(
                finish_arg_error(
                    audit,
                    error_class="ToolArgumentError",
                    error_message=str(exc.args[0] if exc.args else "ToolArgumentError"),
                    error_payload=arg_error_payload,
                )
            )
            anti_anchor.record_failure(tool_name, audit.arguments_hash)
            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(arg_error_payload),
                }
            )
            return _SessionAwareDispatchOutcome(result=None, is_discovery=False)

        handler = _SESSION_AWARE_TOOL_HANDLERS[tool_name]
        kwargs = self._build_session_aware_kwargs(
            tool_name=tool_name,
            arguments=arguments,
            state=state,
            session_id=session_id,
            current_state_id=current_state_id,
            tool_call_id=tool_call_id,
            response=response,
        )

        try:
            result = await handler(**kwargs)
        except ToolArgumentError as exc:
            # Two sub-paths: rate-cap (write F-6 row + emit F-15 telemetry
            # BEFORE raising the LLM-facing ARG_ERROR) vs. generic
            # ARG_ERROR (no extra side effects, standard echo).
            cap_type = (
                RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE[exc.code]
                if exc.code is not None and exc.code in RATE_CAP_CODE_TO_TELEMETRY_CAP_TYPE
                else None
            )
            if cap_type is not None:
                # F-15 telemetry FIRST (the spec is explicit: emit BEFORE
                # the ARG_ERROR returns). Operational-only — no
                # ``user_term`` attribute, PII risk.
                self._telemetry.interpretation_rate_cap_exceeded_total.add(
                    1,
                    attributes={
                        "cap_type": cap_type,
                        "session_id": session_id,
                    },
                )
                # F-6 writer SECOND. Best-effort with respect to the
                # interpretation_events row for the rejected call — the
                # handler already declined to insert a row, so this
                # AUTO_INTERPRETED_NO_SURFACES row is the only record of
                # the cap event. Exceptions here are NOT swallowed: a DB
                # failure at this site is a Tier-1 audit anomaly.
                sessions_service = self._require_sessions_service()
                await sessions_service.record_auto_interpreted_no_surfaces_event(
                    session_id=UUID(session_id),
                    # ``audit.actor`` is the loop-local ``composer-web:user-…``
                    # actor string assembled at the top of ``_compose_loop``;
                    # it is the truthful caller identity for this dispatch
                    # and matches the audit envelope's ``actor`` field.
                    actor=audit.actor,
                    kind=_request_interpretation_review_kind_from_arguments(arguments),
                    model_identifier=self._model,
                    model_version=safe_response_model(response) or self._model,
                    provider=self._availability.provider or "unknown",
                    composer_skill_hash=self._composer_skill_hash,
                )

            # Audit envelope: ARG_ERROR. Truthful — the handler returned
            # a ToolArgumentError; the rate-cap subtype is recorded
            # elsewhere (F-6 row + F-15 telemetry).
            arg_error_payload = _arg_error_payload(exc, tool_name)
            recorder.record(
                finish_arg_error(
                    audit,
                    error_class="ToolArgumentError",
                    error_message=str(exc.args[0] if exc.args else "ToolArgumentError"),
                    error_payload=arg_error_payload,
                )
            )
            anti_anchor.record_failure(tool_name, audit.arguments_hash)
            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(arg_error_payload),
                }
            )
            # Session-aware tools currently all carry composition-state
            # mutation intent (interpretation review stages a future
            # /resolve patch). Count toward composition turns regardless
            # of the SUCCESS/ARG_ERROR outcome, matching the sync
            # ARG_ERROR handling for mutation tools.
            return _SessionAwareDispatchOutcome(result=None, is_discovery=False)

        # SUCCESS path. The handler returned a clean ToolResult; record
        # ``finish_success`` and serialise the result for the LLM. The
        # ``result_payload`` matches the sync path's ToolResult.to_dict()
        # so the audit table's ``result_canonical`` column is shape-
        # consistent across dispatch paths.
        recorder.record(
            finish_success(
                audit,
                result_payload=result.to_dict(),
                version_after=result.updated_state.version,
            )
        )
        # Don't claim mutation success when the handler intentionally
        # returns state.version unchanged (interpretation_review_pending
        # stages a future /resolve patch; the version advances at
        # resolve-time, not at staging-time). Treat as a structural
        # success for anti-anchor tracking but not as a version-advance
        # mutation.
        if result.updated_state.version > state.version:
            anti_anchor.record_success()
        llm_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": _serialize_tool_result(result),
            }
        )
        return _SessionAwareDispatchOutcome(result=result, is_discovery=False)

    def _build_session_aware_kwargs(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        state: CompositionState,
        session_id: str,
        current_state_id: str,
        tool_call_id: str,
        response: Any,
    ) -> dict[str, Any]:
        """Build the kwarg dict for a session-aware tool handler.

        Each session-aware tool's handler signature is closed-form
        (the session-aware tool contract documents the required shape); the kwargs differ per
        tool because the injected service methods and snapshot fields
        vary. Adding a new session-aware tool adds a branch here.
        """
        if tool_name == "request_interpretation_review":
            sessions_service = self._require_sessions_service()
            return {
                "arguments": arguments,
                "state": state,
                "session_id": UUID(session_id),
                "composition_state_id": UUID(current_state_id),
                "tool_call_id": tool_call_id,
                "now": datetime.now(UTC),
                "per_term_cap": self._settings.composer_interpretation_rate_limit_per_term,
                "per_session_day_cap": self._settings.composer_interpretation_rate_limit_per_session_day,
                "model_identifier": self._model,
                # ``model_version`` is the actual provider-returned model
                # string (``response.model``) when available; LiteLLM
                # populates this for Anthropic/OpenAI with the dated
                # variant (e.g. ``claude-opus-4-7-20260101``). When the
                # provider does not return one we fall back to the
                # requested identifier — keeps the column NOT NULL
                # without fabricating a value.
                "model_version": safe_response_model(response) or self._model,
                "provider": self._availability.provider or "unknown",
                "composer_skill_hash": self._composer_skill_hash,
                "create_pending_interpretation_event": sessions_service.create_pending_interpretation_event,
                "list_interpretation_events": sessions_service.list_interpretation_events,
            }
        # Defensive: a session-aware tool registered without a kwarg
        # branch here would silently fail at dispatch. Crash loudly so
        # the registration is wired completely before the LLM can
        # invoke it.
        raise RuntimeError(
            f"_build_session_aware_kwargs has no branch for {tool_name!r}; "
            f"every entry in _SESSION_AWARE_TOOL_HANDLERS must add a kwarg-build "
            f"branch here."
        )

    async def _call_advisor_with_audit(
        self,
        arguments: dict[str, Any],
        *,
        recorder: BufferingRecorder | None,
        timeout: float | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Phone the configured advisor (frontier) model for a hint.

        Builds a structured prompt from the composer LLM's stuck-message
        arguments (``problem_summary``, ``recent_errors``, ``attempted_actions``,
        optional ``schema_excerpt``), forwards it to ``composer_advisor_model``
        via LiteLLM as a text-only completion (no tools), and returns
        ``(guidance_text, metadata)``. The caller may pass ``timeout`` to
        bound the advisor-specific timeout by the compose-loop deadline. The
        metadata dict carries inner-LLM accounting (model returned,
        prompt/completion tokens, cached prompt tokens, latency) so the outer
        tool-result envelope can embed it for audit-trail completeness.

        A :class:`ComposerLLMCall` record is fired into ``recorder`` in
        the ``finally`` block so the audit captures failure modes
        (timeouts, auth errors, malformed responses) just as cleanly as
        the success path. The outer ``ComposerToolInvocation`` record is
        the caller's responsibility — the compose-loop interception
        wraps this call with ``finish_success`` either way.

        Anthropic prompt-cache markers are deliberately NOT applied here.
        Advisor calls now include the same composer skill stack as normal
        composer requests, but their model and accounting are independent
        from the primary composer path. If advisor prompt caching becomes
        required, add it with focused usage-accounting tests rather than
        inheriting the primary-composer marker placement by accident.
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError
        from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

        advisor_model = self._settings.composer_advisor_model
        configured_timeout = self._settings.composer_advisor_timeout_seconds
        effective_timeout = configured_timeout if timeout is None else min(configured_timeout, timeout)
        max_completion = self._settings.composer_advisor_max_completion_tokens

        system_msg = build_system_prompt(self._data_dir, advisor_enabled=True) + "\n\n" + _ADVISOR_SYSTEM_INSTRUCTIONS
        # Required fields (trigger, problem_summary, recent_errors,
        # attempted_actions) are validated by _TOOL_REQUIRED_PATHS before this
        # method runs, so direct dict access is sound. schema_excerpt is the
        # only optional field — we test "in arguments" rather than .get() to
        # keep the Tier-3 trust-boundary rules clean.
        user_msg = _build_advisor_user_message(arguments)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        advisor_seed = _composer_llm_seed_for_model(advisor_model)
        kwargs: dict[str, Any] = {
            "model": advisor_model,
            "messages": messages,
            "temperature": _COMPOSER_LLM_TEMPERATURE,
            "max_tokens": max_completion,
        }
        if advisor_seed is not None:
            kwargs[_COMPOSER_LLM_SEED_PARAM] = advisor_seed
        try:
            response = await asyncio.wait_for(
                _litellm_acompletion(**kwargs),
                timeout=effective_timeout,
            )
            if not response.choices:
                raise _MalformedLLMResponseError(
                    "Advisor returned empty choices array",
                    response=response,
                )
            # F4: validate content BEFORE marking SUCCESS. None / empty /
            # whitespace-only content (content-filter triggered, malformed
            # provider output, tool-call-only response) must classify as
            # MALFORMED_RESPONSE rather than fall through to SUCCESS-with-
            # empty-guidance. Empty success would consume budget and tell
            # the composer LLM "you got advice" while no information was
            # actually produced.
            raw_content = response.choices[0].message.content
            if raw_content is None or not str(raw_content).strip():
                raise _MalformedLLMResponseError(
                    "Advisor returned empty or whitespace-only content",
                    response=response,
                )
            guidance = raw_content
            status = ComposerLLMCallStatus.SUCCESS
            usage = token_usage_from_response(response)
            metadata = {
                "model": safe_response_model(response) or advisor_model,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cached_prompt_tokens": usage.cached_prompt_tokens,
                "latency_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
            }
            return guidance, metadata
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMBadRequestError as exc:
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except _MalformedLLMResponseError as exc:
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            response = exc.response
            error_class = type(exc).__name__
            error_message = "malformed_response"
            raise
        except Exception as exc:
            # F5: catch-all so the inner ComposerLLMCall record always
            # lands in the audit trail, even for exception classes not
            # in the typed clauses above (httpx ConnectionError, codec
            # ValueError, etc.). Without this, ``status`` would stay
            # ``None`` and the finally block would skip
            # ``record_llm_call``, leaving an audit gap for exactly the
            # broad-except failure path the compose-loop interception
            # relies on. API_ERROR is the closest semantic for "unknown
            # provider-side / transport failure"; the exception class
            # name is preserved in ``error_class`` for forensic detail.
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=advisor_model,
                        messages=messages,
                        tools=None,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=_COMPOSER_LLM_TEMPERATURE,
                        seed=advisor_seed,
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                    )
                )
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    attach_llm_calls(current_exc, recorder)

    async def _call_llm_with_audit(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        timeout: float,
        recorder: BufferingRecorder | None,
    ) -> Any:
        """Call the composer LLM once and record an audit sidecar.

        For Anthropic-family providers, ``cache_control`` markers are
        applied to the stable first system message and the trailing tool
        before the call. Dynamic composer state lives in the later context
        system message, outside the stable prompt-cache breakpoint. The
        transformed payload is what flows to LiteLLM and what the audit
        ``messages_hash`` / ``tools_spec_hash`` record — the hash is over
        the bytes actually sent, so the audit row is truthful about the
        wire payload (elspeth-4e79436719).
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        if supports_anthropic_prompt_cache_markers(self._model):
            messages, tools_or_none = apply_anthropic_cache_markers(messages, tools)
            tools = tools_or_none if tools_or_none is not None else tools

        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any | None = None
        error_class: str | None = None
        error_message: str | None = None
        try:
            response = await asyncio.wait_for(
                self._call_llm(messages, tools),
                timeout=timeout,
            )
            status = ComposerLLMCallStatus.SUCCESS
            return response
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            attach_llm_calls(exc, recorder)
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            attach_llm_calls(exc, recorder)
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            attach_llm_calls(exc, recorder)
            raise
        except _MalformedLLMResponseError as exc:
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            response = exc.response
            error_class = type(exc).__name__
            error_message = "malformed_response"
            attach_llm_calls(exc, recorder)
            raise
        except _BadRequestLLMError as exc:
            cause = exc.__cause__
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(cause).__name__ if cause is not None else type(exc).__name__
            error_message = error_class
            attach_llm_calls(exc, recorder)
            raise
        except Exception as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            attach_llm_calls(exc, recorder)
            raise
        finally:
            if recorder is not None and status is not None:
                recorder.record_llm_call(
                    build_llm_call_record(
                        model_requested=self._model,
                        messages=messages,
                        tools=tools,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        temperature=_COMPOSER_LLM_TEMPERATURE,
                        seed=_composer_llm_seed_for_model(self._model),
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                    )
                )
                current_exc = sys.exc_info()[1]
                if current_exc is not None:
                    attach_llm_calls(current_exc, recorder)

    async def _call_llm_before_deadline(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        state: CompositionState,
        initial_version: int,
        deadline: float,
        recorder: BufferingRecorder | None = None,
    ) -> Any:
        """Call the LLM with a per-call timeout derived from the deadline.

        LLM calls are pure network I/O with no side effects, so they
        are safe to cancel via asyncio.wait_for.  If the deadline has
        already passed or the call exceeds the remaining budget, raise
        ComposerConvergenceError with the current partial state.

        ``recorder`` is the in-flight :class:`BufferingRecorder` from
        :meth:`_compose_loop` (or ``None`` from test paths). When set,
        timeout-based ``ComposerConvergenceError`` raises include the
        buffer's ``tool_invocations`` so the route handler's audit
        persistence has the per-call decision trail even when the
        budget exhaustion was a wall-clock timeout (no LLM mutation
        in this final call).
        """
        from litellm.exceptions import APIError as LiteLLMAPIError
        from litellm.exceptions import AuthenticationError as LiteLLMAuthError

        def _captured_invocations() -> tuple[ComposerToolInvocation, ...]:
            return recorder.invocations if recorder is not None else ()

        def _captured_llm_calls() -> tuple[ComposerLLMCall, ...]:
            return recorder.llm_calls if recorder is not None else ()

        attempt = 0
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise ComposerConvergenceError.capture(
                    max_turns=0,
                    budget_exhausted="timeout",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=_captured_invocations(),
                    llm_calls=_captured_llm_calls(),
                )
            try:
                return await self._call_llm_with_audit(
                    messages,
                    tools,
                    timeout=remaining,
                    recorder=recorder,
                )
            except TimeoutError:
                raise ComposerConvergenceError.capture(
                    max_turns=0,
                    budget_exhausted="timeout",
                    state=state,
                    initial_version=initial_version,
                    tool_invocations=_captured_invocations(),
                    llm_calls=_captured_llm_calls(),
                ) from None
            except LiteLLMAuthError:
                raise
            except _BadRequestLLMError:
                # Bad-request from provider: never retry. 400s are not transient,
                # and the carrier holds the provider's status code + detail on
                # dedicated attributes for the outer handler to build the HTTP
                # detail. The redacted str(exc) intentionally does NOT leak
                # provider text; only ``expose_provider_error=True`` surfaces
                # ``provider_detail``/``provider_status_code``.
                #
                # Reciprocal contract: the route layer reads those two
                # attributes via ``_litellm_error_detail`` in
                # ``web/sessions/routes/_helpers.py`` (and the parallel call
                # site in ``web/execution/routes.py:evaluate_run_diagnostics``).
                # Any future bad-request carrier that subclasses this or
                # supersedes it MUST populate both ``provider_detail`` and
                # ``provider_status_code`` for the HTTP surface to remain
                # useful — otherwise the route falls back to the redacted
                # class-name wrap and operators lose triage data.
                raise
            except LiteLLMAPIError:
                attempt += 1
                if attempt >= _LLM_API_MAX_ATTEMPTS:
                    raise
                delay_seconds = _LLM_API_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))
                remaining_after_error = deadline - asyncio.get_event_loop().time()
                if remaining_after_error <= delay_seconds:
                    raise
                await asyncio.sleep(delay_seconds)

    def _compute_availability(self) -> ComposerAvailability:
        """Infer whether the configured model has the required env at boot.

        Delegates to :func:`availability.compute_availability`.
        The monkeypatch target ``ComposerServiceImpl._compute_availability``
        is preserved here so test fixtures using
        ``monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", ...)``
        continue to work without modification.
        """
        from elspeth.web.composer.availability import compute_availability

        return compute_availability(self)


def _infer_provider_from_model_name(model: str) -> str | None:
    """Infer provider from a provider-prefixed model string."""
    if "/" not in model:
        return None
    return model.split("/", 1)[0]


def _infer_provider_from_unprefixed_model_name(model: str) -> str | None:
    """Infer provider for common unprefixed model families."""
    normalized = model.lower()
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    return None


def _pydantic_default(obj: Any) -> Any:
    """JSON serializer fallback for Pydantic models in tool results."""
    try:
        return obj.model_dump()
    except AttributeError:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable") from None


def _serialize_tool_result(result: Any) -> str:
    """Serialize a ToolResult to JSON, handling Pydantic models in data."""
    return json.dumps(result.to_dict(), default=_pydantic_default)


def _cached_discovery_payload(result: ToolResult) -> _CachedDiscoveryPayload:
    """Extract the state-independent fields from a cacheable discovery result."""
    return _CachedDiscoveryPayload(
        success=result.success,
        affected_nodes=result.affected_nodes,
        data=result.data,
    )


def _result_from_cached_discovery_payload(
    state: CompositionState,
    cached: _CachedDiscoveryPayload,
) -> ToolResult:
    """Rebuild a cached discovery result with the current state envelope."""
    return ToolResult(
        success=cached.success,
        updated_state=state,
        validation=state.validate(),
        affected_nodes=cached.affected_nodes,
        data=cached.data,
    )


def _make_cache_key(tool_name: str, arguments: dict[str, Any]) -> str:
    """Build a deterministic cache key from tool name + arguments."""
    # Sort keys for determinism. Arguments are simple JSON-serializable
    # dicts from the LLM — no MappingProxyType or frozen containers.
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"


_ADVISOR_SYSTEM_INSTRUCTIONS: Final[str] = (
    "Advisor mode:\n"
    "- You are advising another LLM (a pipeline composer) that is stuck while building an ELSPETH pipeline.\n"
    "- Use the composer skill context and any deployment overlay above as binding local policy.\n"
    "- Read the problem summary, the verbatim validator errors, and the actions already attempted.\n"
    "- Return ONE concrete actionable hint: name specific fields, suggest values, and point at schema sections if provided.\n"
    "- Do not write YAML, do not produce final configuration, do not claim authority.\n"
    "- Your response is ADVICE; the composer LLM will decide what to apply.\n"
    "- Be specific and brief: under 250 words."
)


def _build_advisor_user_message(arguments: Mapping[str, Any]) -> str:
    """Build the exact variable user message sent to the advisor LLM.

    The validation path uses this same helper for prompt-size accounting, so
    bullets, section labels, and newlines cannot drift from the wire payload.
    Callers validate the Tier-3 argument shapes before invoking this helper.
    """
    user_msg_parts: list[str] = [
        f"Advisor trigger: {arguments['trigger']}",
        f"Problem: {arguments['problem_summary']}",
    ]
    recent = cast(list[str], arguments["recent_errors"])
    if recent:
        joined = "\n".join(f"- {e}" for e in recent)
        user_msg_parts.append(f"\nRecent validator errors (most recent first):\n{joined}")
    attempted = cast(list[str], arguments["attempted_actions"])
    if attempted:
        joined = "\n".join(f"- {a}" for a in attempted)
        user_msg_parts.append(f"\nAlready attempted:\n{joined}")
    if "schema_excerpt" in arguments and arguments["schema_excerpt"]:
        user_msg_parts.append(f"\nRelevant schema excerpt:\n{cast(str, arguments['schema_excerpt'])}")
    return "\n".join(user_msg_parts)


# ---------------------------------------------------------------------------
# F2 — module-level ARG_ERROR payload helper.
#
# Defined at module scope (not nested inside the compose loop) so the helper
# is directly importable and testable from
# ``tests/unit/web/composer/test_audit_arg_error_validation_errors.py``.
# Placed at end-of-file so callers above it use a forward reference (resolved
# at call time, never at import); inserting a new module-level def in the
# middle of the file would rotate every downstream fingerprint and force a
# churn of allowlist re-keying that has nothing to do with this change.
# ---------------------------------------------------------------------------


def _arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]:
    """Build the structured payload for an ARG_ERROR audit record + LLM tool message.

    The payload serves two consumers (spec §4.2.6, F2 disposition):

    1. ``dispatch_with_audit`` canonicalizes this into ``result_canonical``
       on the ARG_ERROR audit record. Persisted in Tier-1 Landscape.
    2. The compose loop's ARG_ERROR handler ``json.dumps`` this and sends
       it back to the LLM as the ``role=tool`` content. Tier-3 echo.

    Fields
    ------
    ``error``
        Operator-safe, LLM-facing message. Composed from the
        ``ToolArgumentError`` ``args[0]`` (structurally safe by
        construction — see ``ToolArgumentError`` docstring) plus the
        operator-chosen tool name. NEVER contains a raw LLM-supplied
        value.

    ``validation_errors`` (present iff ``exc.__cause__`` is a
        ``pydantic.ValidationError``)
        Leak-safe canonicalization of the Pydantic chained cause —
        ``loc``/``msg``/``type`` only, ``input``/``url``/``ctx``
        stripped. Provides field-name detail for recovery flows in
        recovery flows that need to know which specific Pydantic field
        failed validation. Absent (key omitted) when there is no
        chained Pydantic cause or the chained cause is not a
        ``ValidationError``; recording an empty list has no audit
        value.
    """
    safe_message = exc.args[0] if exc.args else "tool argument error"
    payload: dict[str, Any] = {"error": f"Tool '{tool_name}' failed: {safe_message}"}
    validation_errors = canonicalize_pydantic_cause(exc.__cause__)
    if validation_errors is not None:
        payload["validation_errors"] = validation_errors
    return payload


# ---------------------------------------------------------------------------
# Test-only compose-loop driver result carrier.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComposeLoopTestResult:
    """Structured result returned by the one-turn compose-loop test driver."""

    assistant_message: str
    tool_outcomes: tuple[Any, ...] = ()
    persisted_assistant_row: Any | None = None
    persisted_assistant_tool_calls: tuple[Any, ...] = ()
    persisted_tool_row_content: tuple[Any, ...] = ()
    # Buffered per-call audit invocations so dispatch-branch tests can
    # assert recorder state without
    # touching the persistence machinery directly.
    tool_invocations: tuple[Any, ...] = ()

    @property
    def tool_outcomes_for_assertion(self) -> tuple[Any, ...]:
        """Backward-compatible assertion surface for compose-loop tests."""

        return self.tool_outcomes
