"""ComposerService protocol and result types.

Layer: L3 (application). Defines the service boundary for LLM-driven
pipeline composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

if TYPE_CHECKING:
    from elspeth.web.composer.guided.state_machine import TerminalState

from elspeth.contracts.composer_audit import ComposerToolInvocation
from elspeth.contracts.composer_llm_audit import ComposerLLMCall
from elspeth.contracts.composer_progress import ComposerProgressReason, ComposerProgressSink
from elspeth.contracts.errors import FailedTurnMetadata
from elspeth.web.composer.state import CompositionState
from elspeth.web.execution.schemas import ValidationResult


@dataclass(frozen=True, slots=True)
class ComposerResult:
    """Result of a compose() call.

    Attributes:
        message: The assistant's text response. When synthesis fires
            (preflight invalid, or state-claim grounding correction on
            the happy-path), ``message`` is *augmented*: the model's
            prose is preserved verbatim and an operator-facing suffix is
            appended. The model's pre-synthesis prose is preserved in
            ``raw_assistant_content`` for the audit trail and LLM
            history replay.
        state: The (possibly updated) CompositionState.
        runtime_preflight: The ValidationResult from the final-gate
            runtime preflight run, or ``None`` if no preflight was
            triggered (e.g. the state was unchanged and no preview
            preflight was available to reuse).
        raw_assistant_content: The model's pre-synthesis prose whenever
            ``message`` is *synthesized* (augmented relative to raw LLM
            output), regardless of why synthesis fired. Synthesis
            triggers include preflight failure (empty-state and
            non-empty-state augmentation) and state-claim grounding
            correction on the happy-path. ``None`` when ``message`` is
            the verbatim LLM response.

    Field-pairing invariant:
        - When ``runtime_preflight`` is non-None, not ``is_valid``, and
          not a typed authoring-valid handoff, ``raw_assistant_content``
          MUST be set (the model's pre-synthesis prose must be recoverable
          from the audit row).
        - When ``raw_assistant_content`` is set with passing preflight,
          ``message`` MUST differ from ``raw_assistant_content``
          (otherwise raw is set spuriously — the audit row would falsely
          imply synthesis happened on a verbatim response).
        Enforced mechanically by ``__post_init__`` because this object
        flows into the audit trail — a violating pairing would silently
        misattribute a verbatim LLM response as if synthesis intervened,
        or lose the original LLM output when synthesis actually did
        happen.

    Synthesis shapes (all augmentation):
        Producers (see ``service._finalize_no_tool_response``) emit
        four synthesis shapes — all of them augmentations
        (``message == raw_assistant_content + operator_suffix``):

        1. No-mutation empty-state augmentation — preflight invalid,
           build-style request received no successful mutation, state
           is structurally empty.
        2. Preflight-invalid empty-state augmentation — preflight
           invalid, state is structurally empty.
        3. Preflight-invalid non-empty-state augmentation (issue
           elspeth-9cfbad6901) — preflight invalid, state has been
           populated. Suffix names the validator's specific objection.
        4. State-claim grounding correction (issue elspeth-c028f7d186) —
           preflight VALID. Fires when the model's prose contradicts
           persisted state (forward contradiction) or claims a fresh
           action when state was unchanged this turn (backward
           contradiction).

        Consumers (see ``routes._composer_history_content``) detect
        synthesis structurally at read time via
        ``message.startswith(raw_assistant_content)`` — true for all
        four shapes. The structural property lets consumers strip the
        operator-facing suffix from LLM history without parsing it.

    Producer contract (mechanical):
        ``message`` MUST start with ``raw_assistant_content``. Enforced
        at construction by
        ``service._enforce_augmentation_prefix_invariant``. A producer
        that violates the contract crashes with ``AuditIntegrityError``
        rather than commit a corrupt audit row that the consumer-side
        discriminator would silently misroute. The field-level
        decoupling that would obviate the structural contract is
        tracked at ``elspeth-7ae1732ab2``.
    """

    message: str
    state: CompositionState
    runtime_preflight: ValidationResult | None = None
    raw_assistant_content: str | None = None
    # Per-tool-call audit trail produced during this compose() invocation.
    # Populated by ComposerServiceImpl._compose_loop via a BufferingRecorder.
    # The route handler persists each entry as a role=tool chat message row.
    # Empty tuple when the compose loop made no tool calls (e.g. the LLM
    # returned text-only).
    tool_invocations: tuple[ComposerToolInvocation, ...] = ()
    llm_calls: tuple[ComposerLLMCall, ...] = ()
    # Set when _compose_loop has already committed assistant+tool rows via
    # SessionServiceProtocol.persist_compose_turn_async. Routes must not drain
    # tool_invocations again for that turn.
    persisted_assistant_message_id: str | None = None
    persisted_tool_call_turn: bool = False
    # Number of forced repair turns the proof step injected into this compose
    # invocation. Capped at 2 by the loop. 0 means first-pass success; 1 or 2
    # means the model was given proof_diagnostics back as a synthesized
    # message and asked to iterate. Surfaced on the compose-produced
    # ``composition_states`` row via the ``composer_meta`` JSON column (see
    # web/sessions/models.py and
    # web/sessions/protocol.py::CompositionStateData.composer_meta) and
    # returned by ``GET /api/sessions/{id}/state`` under
    # ``composer_meta.repair_turns_used`` for the convergence-suite eval
    # scorer.
    repair_turns_used: int = 0

    def __post_init__(self) -> None:
        # Two directions of the field-pairing invariant. Both matter:
        #
        # 1. preflight failed with no raw_assistant_content →
        #    audit trail would carry an augmented or synthetic message
        #    but the model's pre-synthesis prose is irrecoverably lost,
        #    breaking the consumer-side structural discriminator in
        #    routes._composer_history_content.
        # 2. raw_assistant_content set with passing preflight AND
        #    ``message == raw_assistant_content`` →
        #    audit trail would falsely imply synthesis happened on a
        #    verbatim response. The narrowing on ``message ==
        #    raw_assistant_content`` allows non-preflight synthesis
        #    triggers (state-claim grounding correction, issue
        #    elspeth-c028f7d186) where preflight is valid but the
        #    message is augmented relative to raw LLM output. The
        #    consumer-side discriminator at
        #    ``routes._composer_history_content`` does not depend on
        #    *why* synthesis happened, only on whether it did
        #    (``message.startswith(raw)`` distinguishes augmentation
        #    from replacement structurally).
        preflight_failed = (
            self.runtime_preflight is not None
            and not self.runtime_preflight.is_valid
            and not (
                self.runtime_preflight.readiness.authoring_valid
                and self.runtime_preflight.readiness.completion_ready
                and not self.runtime_preflight.readiness.execution_ready
            )
        )
        if preflight_failed and self.raw_assistant_content is None:
            raise ValueError(
                "ComposerResult field-pairing invariant violated: "
                "runtime_preflight failed but raw_assistant_content is None — "
                "the failed preflight should have augmented or replaced "
                "message and parked the model's pre-synthesis prose in "
                "raw_assistant_content for audit-trail recovery and for the "
                "consumer-side augment-vs-replace discriminator at "
                "routes._composer_history_content."
            )
        if self.raw_assistant_content is not None and not preflight_failed and self.message == self.raw_assistant_content:
            raise ValueError(
                "ComposerResult field-pairing invariant violated: "
                "raw_assistant_content is set with passing preflight, but "
                "message is identical to raw_assistant_content — the audit "
                "row would falsely imply synthesis (augmentation or "
                "replacement) happened on a verbatim LLM response. "
                "raw_assistant_content must only be set when message has "
                "actually been synthesized (preflight-failure shapes, or "
                "the state-claim grounding correction shape on the "
                "happy-path)."
            )
        # Cap-assert on repair_turns_used. The loop enforces the bound
        # informally via ``_MAX_REPAIR_TURNS`` (web/composer/service.py),
        # but the field flows into the audit trail via
        # ``composition_states.composer_meta.repair_turns_used`` and is
        # consumed by the convergence-suite eval scorer. An out-of-range
        # value here would land in the legal record and be silently
        # accepted by downstream consumers. Literal[0, 1, 2] would have
        # been the mechanical mypy-checked form, but the compose loop
        # mutates a plain ``int`` counter and threads it via
        # ``dataclasses.replace`` — mypy cannot narrow ``int`` to a
        # Literal at that site. The runtime check is the fallback that
        # still mechanically rejects the bad value at the construction
        # boundary. Keep this bound aligned with ``_MAX_REPAIR_TURNS``
        # in web/composer/service.py.
        if not 0 <= self.repair_turns_used <= 2:
            raise ValueError(
                "ComposerResult.repair_turns_used must be 0, 1, or 2 "
                "(capped by _MAX_REPAIR_TURNS in web/composer/service.py); "
                f"got {self.repair_turns_used}."
            )


class ComposerServiceError(Exception):
    """Base exception for composer service errors."""


def _convergence_reason_for_budget(
    budget_exhausted: Literal["composition", "discovery", "timeout"],
) -> ComposerProgressReason:
    """Map the private convergence budget discriminator to public reason code."""

    if budget_exhausted == "composition":
        return "convergence_composition_budget"
    if budget_exhausted == "discovery":
        return "convergence_discovery_budget"
    return "convergence_wall_clock_timeout"


class ComposerConvergenceError(ComposerServiceError):
    """Raised when the LLM tool-use loop exhausts its budget or times out.

    Declared attributes are frozen after construction (see ``__setattr__``
    below): this exception instance flows into the 422 HTTP response body
    and — when ``partial_state`` is non-None — into the immutable
    ``composition_states`` audit table. Allowing post-construction
    reassignment would let any intermediate layer silently rewrite what
    downstream consumers see. Exception-chain dunders
    (``__cause__``/``__context__``/``__traceback__``/``__notes__``) remain
    writable so ``raise ... from ...`` and ``add_note()`` work normally.

    Attributes:
        max_turns: Total turns used before exhaustion.
        budget_exhausted: Which budget was exhausted — one of
            "composition", "discovery", or "timeout".
        partial_state: The last CompositionState with
            ``version > initial_version``, or None if no mutations
            occurred. Production raise sites MUST go through
            :meth:`capture`, which encapsulates the rule as a single
            source of truth. The direct constructor exists for tests
            that inject specific ``partial_state`` shapes to exercise
            route-handler branches.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"max_turns", "budget_exhausted", "partial_state", "tool_invocations", "llm_calls", "reason", "evidence", "failed_turn"}
    )

    def __init__(
        self,
        max_turns: int,
        *,
        budget_exhausted: Literal["composition", "discovery", "timeout"] = "composition",
        partial_state: CompositionState | None = None,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        reason: ComposerProgressReason | None = None,
        evidence: Mapping[str, Any] | None = None,
        failed_turn: FailedTurnMetadata | None = None,
    ) -> None:
        super().__init__(
            f"Composer did not converge within {max_turns} turns "
            f"(budget exhausted: {budget_exhausted}). "
            f"The LLM kept making tool calls without producing a final response."
        )
        self.max_turns = max_turns
        self.budget_exhausted = budget_exhausted
        self.partial_state = partial_state
        self.reason = reason or _convergence_reason_for_budget(budget_exhausted)
        self.evidence = MappingProxyType(dict(evidence or {}))
        # Per-tool-call audit trail accumulated up to the convergence
        # event. Includes the tool calls that did NOT cause a state
        # mutation (cache hits, ARG_ERROR, discovery-only). The route
        # handler persists this regardless of whether ``partial_state``
        # is set — an audit gap on a no-state-change failure is still
        # an audit gap.
        self.tool_invocations = tool_invocations
        self.llm_calls = llm_calls
        self.failed_turn = failed_turn

    def __setattr__(self, name: str, value: object) -> None:
        # Guard only the declared attributes; exception-chain machinery
        # (__cause__, __context__, __suppress_context__, __traceback__,
        # __notes__) must remain writable so `raise ... from ...` and
        # `add_note()` continue to work. First-time write during
        # ``__init__`` is allowed; subsequent reassignment raises.
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is frozen after construction; exception attributes flow into HTTP responses and Landscape."
            )
        super().__setattr__(name, value)

    @classmethod
    def capture(
        cls,
        max_turns: int,
        *,
        budget_exhausted: Literal["composition", "discovery", "timeout"] = "composition",
        state: CompositionState,
        initial_version: int,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        reason: ComposerProgressReason | None = None,
        evidence: Mapping[str, Any] | None = None,
        failed_turn: FailedTurnMetadata | None = None,
    ) -> ComposerConvergenceError:
        """Build from compose-loop locals, applying the partial-state rule.

        ``partial_state`` is set to ``state`` iff ``state.version >
        initial_version`` — i.e. at least one tool call successfully
        committed a mutation before the budget was hit. Otherwise
        ``partial_state`` is ``None`` so the route handler does not
        append an identity-copy row to ``composition_states`` (which
        would pollute the audit history with zero-delta entries).

        ``tool_invocations`` carries the per-tool-call audit trail. It
        is populated unconditionally — even when ``partial_state`` is
        None — so failed runs without state mutations still leave a
        record of what the LLM tried.

        This classmethod is the SINGLE source of truth for the rule.
        Every production compose-loop raise site MUST use it so the
        invariant cannot drift between sites.
        """
        partial = state if state.version > initial_version else None
        return cls(
            max_turns,
            budget_exhausted=budget_exhausted,
            partial_state=partial,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
            reason=reason,
            evidence=evidence,
            failed_turn=failed_turn,
        )


class ComposerPluginCrashError(ComposerServiceError):
    """Raised when an exception escapes ``execute_tool()`` inside the compose loop.

    Signals a plugin (tier 1/2) bug — distinct from ``ToolArgumentError``
    (which is a tier 3 boundary signal and is caught inside the loop).

    Symmetric with :class:`ComposerConvergenceError`: both transport a
    ``partial_state`` field from service to route so the route handler can
    persist the accumulated in-memory mutations into ``composition_states``
    before returning the failure response. Without this carrier, any tool
    call that successfully mutated state prior to a later crash would be
    silently dropped from the immutable-append state history, and recompose
    would restart from the stale pre-request state.

    Attributes:
        original_exc: The underlying plugin-bug exception. Preserved on
            ``__cause__`` via ``raise ... from`` so the ASGI error machinery
            still has the full traceback, but the route handler redacts
            ``str(original_exc)`` / its ``__cause__`` chain from the HTTP
            response because those may carry DB URLs, filesystem paths, or
            secret fragments.
        partial_state: The last :class:`CompositionState` with ``version >
            initial_version``, or ``None`` if no mutations occurred before
            the crash.
        exc_class: ``type(original_exc).__name__`` — the only safe
            exception-identity hint for structured logs.

    Route ordering: this class inherits from ``ComposerServiceError`` so the
    compose/recompose endpoints in ``web/sessions/routes.py`` must catch
    ``ComposerPluginCrashError`` BEFORE the generic ``except
    ComposerServiceError`` block, mirroring the ordering already used for
    ``ComposerConvergenceError``. If the ordering is inverted the generic
    handler would launder the crash into a 502, reintroducing the
    silent-laundering behaviour the narrowed catch was designed to
    eliminate. The invariant is mechanically enforced by
    ``scripts/cicd/enforce_composer_catch_order.py`` (rule CCO1), which
    scans ``web/`` for any ``try`` block where a superclass handler
    precedes one of its ``ComposerServiceError`` subclasses.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"original_exc", "partial_state", "exc_class", "tool_invocations", "llm_calls", "failed_turn"}
    )

    def __init__(
        self,
        original_exc: Exception,
        *,
        partial_state: CompositionState | None = None,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        failed_turn: FailedTurnMetadata | None = None,
    ) -> None:
        super().__init__(f"Composer plugin crash: {type(original_exc).__name__}")
        self.original_exc = original_exc
        self.partial_state = partial_state
        self.exc_class = type(original_exc).__name__
        # Per-tool-call audit trail accumulated up to the crashing call.
        # The crashing call itself appears as the LAST entry with
        # status=PLUGIN_CRASH so the audit trail records what the LLM
        # tried that triggered the bug.
        self.tool_invocations = tool_invocations
        self.llm_calls = llm_calls
        self.failed_turn = failed_turn

    def __setattr__(self, name: str, value: object) -> None:
        # Guard only the declared attributes; exception-chain dunders
        # (__cause__, __context__, __suppress_context__, __traceback__,
        # __notes__) must remain writable so `raise ... from ...`,
        # structured-log capture, and `add_note()` continue to work.
        # First-time write during ``__init__`` is allowed; subsequent
        # reassignment raises — the three declared fields are consumed
        # verbatim by the HTTP response body, Landscape partial-state
        # persistence, and structured-log exc_class correlation.
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is frozen after construction; exception attributes flow into HTTP responses and Landscape."
            )
        super().__setattr__(name, value)

    @classmethod
    def capture(
        cls,
        original_exc: Exception,
        *,
        state: CompositionState,
        initial_version: int,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        failed_turn: FailedTurnMetadata | None = None,
    ) -> ComposerPluginCrashError:
        """Build from compose-loop locals, applying the partial-state rule.

        ``partial_state`` is set to ``state`` iff ``state.version >
        initial_version`` — i.e. at least one tool call successfully
        committed a mutation before the crash. Otherwise ``partial_state``
        is ``None`` so the route handler does not append an identity-copy
        row to ``composition_states`` (polluting the audit history with
        zero-delta entries).

        ``tool_invocations`` is the per-tool-call audit trail accumulated
        before the crash. The crashing call itself appears as the LAST
        entry with ``status=PLUGIN_CRASH``.

        This classmethod is the SINGLE source of truth for the rule.
        Every production compose-loop raise site MUST use it so the
        invariant cannot drift between sites (mirrors
        :meth:`ComposerConvergenceError.capture`).
        """
        partial = state if state.version > initial_version else None
        return cls(
            original_exc,
            partial_state=partial,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
            failed_turn=failed_turn,
        )


class ComposerRuntimePreflightError(ComposerServiceError):
    """Unexpected internal failure while running composer runtime preflight."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset(
        {"original_exc", "partial_state", "exc_class", "tool_invocations", "llm_calls", "failed_turn"}
    )

    def __init__(
        self,
        *,
        original_exc: Exception,
        partial_state: CompositionState | None,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        failed_turn: FailedTurnMetadata | None = None,
    ) -> None:
        super().__init__("Composer runtime preflight failed internally.")
        self.original_exc = original_exc
        self.partial_state = partial_state
        self.exc_class = type(original_exc).__name__
        # Per-tool-call audit trail accumulated before the runtime
        # preflight failure. Path-1 (cached preflight raised from inside
        # ``compose()``) populates this from the in-flight buffer at
        # raise time. Path-2 (post-compose state-save preflight raised
        # from ``_state_data_from_composer_state``) is populated by
        # the caller threading ``result.tool_invocations`` through.
        self.tool_invocations = tool_invocations
        self.llm_calls = llm_calls
        self.failed_turn = failed_turn

    def __setattr__(self, name: str, value: object) -> None:
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(f"{type(self).__name__}.{name} is frozen after construction")
        super().__setattr__(name, value)

    @classmethod
    def capture(
        cls,
        exc: Exception,
        *,
        state: CompositionState,
        initial_version: int,
        tool_invocations: tuple[ComposerToolInvocation, ...] = (),
        llm_calls: tuple[ComposerLLMCall, ...] = (),
        failed_turn: FailedTurnMetadata | None = None,
    ) -> ComposerRuntimePreflightError:
        partial = state if state.version > initial_version else None
        return cls(
            original_exc=exc,
            partial_state=partial,
            tool_invocations=tool_invocations,
            llm_calls=llm_calls,
            failed_turn=failed_turn,
        )


class ToolArgumentError(Exception):
    """Raised by a tool handler when LLM-supplied arguments are unusable.

    Signals a Tier-3 boundary failure: the LLM provided arguments of the
    wrong type, or semantically invalid values that the handler cannot
    coerce. The compose loop catches this exception and returns the
    message to the LLM as a tool error so it can retry.

    This is the ONLY exception class the compose loop catches around
    execute_tool(). Any other TypeError/ValueError/UnicodeError/KeyError
    escaping a tool handler is a plugin bug and MUST crash — per
    CLAUDE.md, plugin bugs that silently produce wrong results are worse
    than a crash because they pollute the audit trail with confidently
    wrong data.

    Inheritance rationale: this class inherits from ``Exception`` directly,
    NOT from ``ComposerServiceError``. A handler-internal signal caught by
    the compose loop must not be absorbed by the route-level generic
    ``except ComposerServiceError`` block (present in both the compose and
    recompose endpoints of ``web/sessions/routes.py``), which would
    silently convert an escaped ToolArgumentError into a 502 — recreating
    the laundering pattern the compose-loop narrowing is designed to
    eliminate. If a ToolArgumentError ever escapes ``_compose_loop``, that
    is a compose-loop bug: FastAPI's default handler will surface it as
    an unstructured 500 for investigation, which is the correct failure
    mode for an invariant violation.

    Structural safety (leak prevention)
    -----------------------------------
    The message composed by this class is echoed verbatim to the LLM
    API by the compose loop AND recorded in the Landscape audit trail
    via the synthetic ``role: tool`` chat message. Free-form f-string
    construction — ``ToolArgumentError(f"bad value: {user_input!r}")``
    — would be a direct leak channel for secrets, PII, and
    attacker-controlled strings, because Tier-3 argument values are
    by definition untrusted.

    To make that leak structurally impossible, this class accepts ONLY
    three keyword-only, safe-by-construction fields:

    - ``argument``: the parameter name as declared in the tool schema
      (operator-chosen — safe for echo/audit).
    - ``expected``: a brief description of the required shape, e.g.
      ``"a string"`` or ``"a non-empty list"`` (operator-chosen — safe).
    - ``actual_type``: typically ``type(value).__name__`` — carries
      only the class name, never the value.

    There is deliberately no field that can carry the LLM-supplied
    value. The ``__cause__`` chain still carries full debugging
    context for auditors (inspectable via ``exc.__cause__`` on the
    captured exception record) but is NEVER echoed to the LLM: the
    compose loop reads ``exc.args[0]`` only, and ``args[0]`` is
    composed from the structured fields above.

    The three declared fields are frozen after construction, matching
    the pattern used by ``ComposerConvergenceError`` and
    ``ComposerPluginCrashError``: each exception flows into an
    immutable audit artefact, so allowing post-construction mutation
    would let an intermediate layer silently rewrite what downstream
    consumers see. Exception-chain dunders
    (``__cause__``/``__context__``/``__traceback__``/``__notes__``)
    remain writable so ``raise ... from ...`` and ``add_note()`` work
    normally.

    Usage::

        raise ToolArgumentError(
            argument="content",
            expected="a string",
            actual_type=type(content).__name__,
        ) from exc

    The ``from exc`` clause preserves the underlying cause on
    ``__cause__`` so it survives ``asyncio.to_thread`` re-raise for
    audit, without leaking into the LLM echo.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"argument", "expected", "actual_type", "code"})

    def __init__(
        self,
        *,
        argument: str,
        expected: str,
        actual_type: str,
        code: str | None = None,
    ) -> None:
        # Reject empty strings at construction time: a blank field
        # would produce a nonsensical LLM echo ("'' must be , got ")
        # and — more importantly — undermines the audit record the
        # exception lands in (the three fields appear as structured
        # columns alongside the composed message).
        if not argument:
            raise ValueError("ToolArgumentError.argument must be a non-empty identifier")
        if not expected:
            raise ValueError("ToolArgumentError.expected must be a non-empty description")
        if not actual_type:
            raise ValueError("ToolArgumentError.actual_type must be a non-empty type name")
        super().__init__(f"'{argument}' must be {expected}, got {actual_type}")
        self.argument = argument
        self.expected = expected
        self.actual_type = actual_type
        # Optional internal discriminant for compose-loop dispatch logic. The
        # ``code`` field is operator-controlled (a fixed string constant chosen
        # by the handler raising the exception, e.g.
        # ``"RATE_CAP_PER_TERM"``) — it is NEVER an LLM- or user-supplied
        # value and is NOT included in ``args[0]`` / the LLM echo. The
        # compose loop reads it to distinguish branches that need extra
        # bookkeeping (write an AUTO_INTERPRETED_NO_SURFACES row, emit
        # operational telemetry) from generic ARG_ERROR. ``None`` means
        # "no specific dispatch hook" — the default for all existing
        # raise sites pre Phase 5b Task 5 follow-on.
        self.code = code

    def __setattr__(self, name: str, value: object) -> None:
        # Guard only the three declared attributes; exception-chain
        # dunders (__cause__, __context__, __suppress_context__,
        # __traceback__, __notes__) must remain writable so
        # ``raise ... from ...``, structured-log capture, and
        # ``add_note()`` continue to work. First-time write during
        # ``__init__`` is allowed; subsequent reassignment raises.
        if name in type(self)._FROZEN_ATTRS and name in self.__dict__:
            raise AttributeError(
                f"{type(self).__name__}.{name} is frozen after construction; exception attributes flow into the LLM echo and Landscape."
            )
        super().__setattr__(name, value)


class ComposerSettings(Protocol):
    """Protocol for the settings the composer service needs.

    Allows ComposerServiceImpl to depend on a structural type rather than
    the concrete WebSettings class. Properties are read-only to match
    frozen Pydantic models.
    """

    @property
    def composer_model(self) -> str: ...

    @property
    def composer_temperature(self) -> float | None: ...

    @property
    def composer_seed(self) -> int | None: ...

    @property
    def composer_max_composition_turns(self) -> int: ...

    @property
    def composer_max_discovery_turns(self) -> int: ...

    @property
    def composer_max_tool_calls_per_turn(self) -> int: ...

    @property
    def composer_timeout_seconds(self) -> float: ...

    @property
    def composer_runtime_preflight_timeout_seconds(self) -> float: ...

    @property
    def composer_advisor_model(self) -> str: ...

    @property
    def composer_advisor_max_calls_per_compose(self) -> int: ...

    @property
    def composer_advisor_checkpoint_max_passes(self) -> int: ...

    @property
    def composer_advisor_max_prompt_tokens(self) -> int: ...

    @property
    def composer_advisor_max_completion_tokens(self) -> int: ...

    @property
    def composer_advisor_timeout_seconds(self) -> float: ...

    @property
    def composer_interpretation_rate_limit_per_term(self) -> int: ...

    @property
    def composer_interpretation_rate_limit_per_session_day(self) -> int: ...

    @property
    def max_blob_storage_per_session_bytes(self) -> int: ...

    @property
    def data_dir(self) -> Any: ...


class ComposerService(Protocol):
    """Protocol for the LLM-driven pipeline composer.

    Accepts a user message, pre-fetched chat history, and current state.
    Runs the LLM tool-use loop. Returns the assistant's response
    and the (possibly updated) state. Tool-call turns may persist their
    assistant/tool audit rows through SessionServiceProtocol; terminal
    no-tool assistant messages are still route-persisted (seam contract B).
    """

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
        """Run the LLM composition loop.

        Args:
            message: The user's chat message.
            messages: Chat history as plain dicts (role/content keys).
                The route handler fetches ChatMessageRecord from
                session_service.get_messages(), converts each to a dict,
                and passes the result here. ComposerService may depend on
                SessionServiceProtocol for tool-call turn persistence (seam
                contract B), but does not fetch history itself.
            state: The current CompositionState.
            current_state_id: Database id of ``state`` when it came from a
                persisted session row. Used as the stale-state guard for
                compose-loop tool-call audit persistence.
            user_id: Current user ID. Passed through to secret tools.
            guided_terminal: When set, the resolved TerminalState from the
                completed guided session; triggers the layered mode-transition
                prompt for this first freeform turn (spec §8.2).
            user_message_id: Database id of the just-persisted user
                ``chat_messages`` row that triggered this compose call
                (Phase 5a Task 2.5). Threaded through the compose loop into
                inline-blob writers so any blob materialised by a tool call
                this turn records ``created_from_message_id`` pointing back
                at this id. Defaults to ``None`` for test paths and
                non-route callers; the composite FK on
                ``(created_from_message_id, session_id)`` in ``blobs_table``
                rejects cross-session lineage, so a wrong id surfaces as
                IntegrityError rather than silent provenance corruption.

        Returns:
            ComposerResult with assistant message and updated state.

        Raises:
            ComposerConvergenceError: If the loop exceeds max_turns.
        """

    async def surface_pending_interpretation_reviews(
        self,
        state: CompositionState,
        *,
        session_id: str | None,
        current_state_id: str | None,
    ) -> None:
        """Kind-general backend surfacer for the GUIDED commit path (B1).

        Surfaces a resolvable pending interpretation EVENT for every
        interpretation site on ``state`` whose writer-boundary precondition
        holds (all five ``InterpretationKind`` members). Called by the guided
        route persistence seam (``post_guided_respond``) after every committed
        source / transform / recipe-apply, because the guided dispatch path
        never reaches the freeform fail-closed orphan gate. Advisory polarity:
        the run-time ``UnresolvedInterpretationPlaceholderError`` gate stays the
        hard backstop. Idempotent; a no-op when there is no session/persisted
        state. See P3.1 for the concrete implementation.
        """
        ...

    async def explain_run_diagnostics(self, snapshot: Mapping[str, object]) -> str:
        """Explain a bounded run diagnostics snapshot without mutating state."""
        ...
