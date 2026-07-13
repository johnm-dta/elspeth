"""OTel counters for SessionServiceImpl and the compose loop.

Production code uses the real OTel meter from
``opentelemetry.metrics.get_meter``; tests use
``build_sessions_telemetry()`` with no meter, which returns
``_FakeCounter`` instances. The ``_Counter`` and ``_Meter`` Protocols
match the OTel API exactly (``add(amount, attributes=None,
context=None)`` and ``create_counter(name, ...)``), so production
wiring type-checks without ``# type: ignore`` and the real meter
satisfies the structural contract.

The fake counter records every ``add`` call as ``(amount,
attributes, context)`` tuples. Tests inspect via the ``observed_value(counter)``
helper (cumulative sum) or directly through the ``calls`` attribute
after type-narrowing with ``isinstance(counter, _FakeCounter)``.

The authenticated Prometheus reader retains these existing point attributes.
In AWS ECS mode, ``web.operator_telemetry`` independently sanitizes the OTLP
copy against a closed attribute-key allowlist, so run/session/row identities
never become CloudWatch dimensions while local exposition stays compatible.
``observed_value`` is intentionally NOT on the ``_Counter`` Protocol
— production OTel counters do not expose observation, and adding it
would force a structural lie.

**Ownership-vs-metric namespace.** The container type is named
``_SessionsTelemetry`` and the module lives under
``web/sessions/telemetry.py`` because ``SessionServiceImpl`` owns
the persistence counters. The OTel metric strings remain
``composer.audit.*`` because operators consume them as part of the
composer-progress surface — naming reflects the dashboard/consumer
view, not the import direction.

Import direction is unrestricted within the L3 application layer.
Both composer code AND sessions code may import this container, and
sessions code may import composer-owned helpers (e.g. the
``record_*`` emit helpers in ``web/composer/telemetry_phase8.py``).
An earlier draft of this docstring said "sessions code must not
import composer-owned modules"; that prohibition lived only as
docstring text (no CI gate, no enforcement script — verified via
``config/cicd/`` and ``scripts/cicd/`` for the retirement) and was
contradicted by 31+ pre-existing ``from elspeth.web.composer.*``
imports across ``src/elspeth/web/sessions/`` from the day the rule
was first written. The Phase 8b-1b cohort emits made the dead-
letter rule visibly inconsistent, so it has been retired. The
retirement is recorded in the cohort's docstring-corrections
commit (search ``git log -S 'must not import composer'`` to walk
the history).
The practical constraint is the tier-model layer rule
(``contracts/`` → ``core/`` → ``engine/`` → ``plugins/`` + L3
application). Inside L3, composer / sessions / audit_readiness /
preferences / etc. may import each other freely as long as imports
flow within L3 and respect the contract-layer boundaries above.

Phase 2 (redaction counters) and Phase 3 (compose-loop counters) may
extend this container only if ownership still belongs to the sessions
persistence surface; otherwise those phases add composer-owned
telemetry separately (a naming/discoverability convention, not an
import-direction rule).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from opentelemetry.context import Context

type _AttributeValue = str | bool | int | float | Sequence[str] | Sequence[bool] | Sequence[int] | Sequence[float]
type _Attributes = Mapping[str, _AttributeValue]


class _Counter(Protocol):
    """Subset of ``opentelemetry.metrics.Counter`` that production
    code uses.

    The real OTel signature is ``add(amount, attributes=None,
    context=None)``. Keep this Protocol as broad as the SDK surface that
    callers may legally use: ``amount`` may be ``int`` or ``float``;
    attributes may include every OTel scalar/sequence value type; and
    ``context`` is accepted even though Phase 1 callers omit it. This
    avoids a fake-narrow structural type that passes local tests but
    rejects a real Counter-compatible call shape.
    """

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None: ...


class _Meter(Protocol):
    """Subset of ``opentelemetry.metrics.Meter`` that
    ``build_sessions_telemetry`` uses for production wiring.

    ``description`` is keyword-only and optional — the real OTel
    ``Meter.create_counter(name, unit="", description="")`` signature
    accepts it, and the Phase 8 counters wire descriptions to match
    the existing ``_PREFERENCES_PATCH_COUNTER`` precedent in
    ``preferences/service.py``. The Protocol stays minimal otherwise;
    ``unit=`` is not used by this surface today.
    """

    def create_counter(self, name: str, *, description: str = "") -> _Counter: ...


class _FakeCounter:
    """Test-only counter that records every ``add`` call.

    Implements the ``_Counter`` Protocol structurally and adds the
    test inspection surface (``calls`` and the ``observed_value``
    helper). Production code MUST NOT depend on this class — it is
    re-exported only so the telemetry test module and the
    audit-failure-primacy tests can construct it directly.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int | float, dict[str, _AttributeValue] | None, Context | None]] = []

    def add(
        self,
        amount: int | float,
        attributes: _Attributes | None = None,
        context: Context | None = None,
    ) -> None:
        # Defensive copy of the attributes mapping so later mutation
        # of the caller's dict cannot rewrite recorded history.
        recorded_attrs = dict(attributes) if attributes is not None else None
        self.calls.append((amount, recorded_attrs, context))


def observed_value(counter: _Counter) -> int | float:
    """Return the cumulative ``add`` total for a fake counter.

    Test-only helper. Raises ``TypeError`` if ``counter`` is not a
    ``_FakeCounter`` — production OTel counters do not expose
    observation, so a misuse (test running against a
    ``build_sessions_telemetry`` that was wired with a real meter)
    fails loudly rather than producing a confusing attribute error.
    """
    if not isinstance(counter, _FakeCounter):
        raise TypeError(
            f"observed_value: expected _FakeCounter, got "
            f"{type(counter).__name__}. Tests must call "
            f"build_sessions_telemetry() without a meter argument so "
            f"the container is populated with fake counters."
        )
    return sum(amount for amount, _attrs, _context in counter.calls)


@dataclass(frozen=True, slots=True)
class _SessionsTelemetry:
    """Container for the named counters introduced by composer progress
    persistence. All counters default to fakes so tests can assert without
    wiring the real OTel SDK; production wiring replaces them at startup.

    Note: no ``__post_init__`` deep-freeze guard is required even though
    this is ``frozen=True``.  Every field is a ``_Counter`` Protocol
    reference whose internal state (call list / aggregated value) is
    mutable by design — that's what makes a counter useful.  ``frozen``
    blocks slot reassignment, which is the only invariant we want.  The
    CLAUDE.md ``deep_freeze`` contract applies to ``Mapping/Sequence/Set``
    container fields; ``_Counter`` is neither.
    """

    tool_row_tier1_violation_total: _Counter
    state_rolled_back_during_persist_total: _Counter
    tool_row_persist_failed_during_unwind_total: _Counter
    tool_row_integrity_violation_total: _Counter
    tool_call_cap_exceeded_total: _Counter
    audit_grade_view_total: _Counter
    audit_access_log_write_failed_total: _Counter
    # Phase 5b Task 5 follow-on (F-15). Counts ``request_interpretation_review``
    # invocations rejected by the per-term or per-session-day rate cap. Carries
    # attributes ``{"cap_type": "per_term" | "per_session_day", "session_id":
    # str}`` (NO ``user_term`` — Tier-3 / PII risk). Operational telemetry,
    # not audit-primary: the AUTO_INTERPRETED_NO_SURFACES interpretation_events
    # row written alongside is the legal record. Telemetry exists so the
    # operator notices unusual cap-breach rates without trawling the audit DB.
    interpretation_rate_cap_exceeded_total: _Counter
    # Phase 5b Task 5 follow-on (F-17 / F-21). Counts /execute attempts that
    # fail the runtime interpretation-review gate. Carries only non-content
    # attributes ``{"component_id": str, "component_type": str, "kind": str}``
    # — no raw ``user_term``, prompt_template, or source text, all of which may
    # contain user-supplied content. Operational telemetry, not audit-primary:
    # the user-actionable RuntimeError surfaced to the frontend is the primary
    # record (an audit-primary ``interpretation_events`` row WOULD have been
    # the primary record had the LLM fired ``request_interpretation_review`` —
    # its absence is what this counter catches). Purpose: catches LLM
    # under-firing after a model upgrade without waiting for offline eval
    # refresh.
    interpretation_placeholder_unresolved_at_runtime_total: _Counter
    # Execution progress broadcast drops are telemetry-only operational
    # degradation: the Landscape run row remains primary, while WebSocket
    # progress delivery is an ephemeral client-notification channel.
    progress_broadcast_dropped_total: _Counter
    # Orphan cleanup is operational lifecycle telemetry: the run row is the
    # permanent record of the cancellation, while this counter exposes cleanup
    # pressure by source without turning logs into a lifecycle channel.
    orphaned_runs_cancelled_total: _Counter
    # ── Phase 8 (mode / session-switched / tutorial / B3 cohort / B5) ──
    # Counters added unconditionally to the container even when the
    # consuming emit-site is conditional on an earlier-phase surface
    # shipping (B3 cohort a/b1/b2, B5 dynamic-source). The cost of an
    # unused counter slot is effectively zero; the cost of branching
    # the container shape on probe outcomes is real bootstrap-order
    # complexity. Probe gates live at the emit sites, not here.
    mode_opted_out_total: _Counter
    mode_opted_in_total: _Counter
    session_switched_total: _Counter
    # Tutorial counters wired by Task 6 (conditional on Phase 4 ship).
    # tutorial_completed_total — DELIBERATELY ABSENT here: completions are
    # counted by composer/tutorial_telemetry.py's attribute-carrying
    # composer.tutorial.completed_total (completion_path ∈ {first_time, skip,
    # retake, repeat}); a second bare registration of the same counter name
    # double-counted completions. Do NOT re-add a slot for it.
    tutorial_started_total: _Counter
    # tutorial_replayed_total — DELIBERATELY ABSENT (Phase 9 deferred per
    # Decision 2 / Option C in 20-phase-8-polish-and-telemetry.md §"Phase
    # 9 follow-ups"). The replay button ships without the counter slot;
    # Phase 9 adds both the slot and the boundary-question resolution
    # (audit-row vs telemetry-only). Do NOT add this field without
    # re-opening Decision 2.
    session_completed_total: _Counter
    # B3 cohort (a) — Phase 6 share-counter emits (Sub-task 7d). Added
    # unconditionally; emit fires only if the Phase 6 token-verify path
    # has shipped (probe in Task 0).
    share_token_verify_failure_total: _Counter
    share_link_expiry_hit_total: _Counter
    # B3 cohort (b1) — Phase 5b interpretation opt-out (Sub-task 7e).
    interpretation_opt_out_total: _Counter
    # B3 cohort (b2) — Phase 2C audit-readiness fetch failure (Sub-task 7f).
    # Telemetry-only signal; superset exception for non-decision read.
    audit_fetch_failure_total: _Counter
    # B4 (W8-r2 module-local counter) — DELIBERATELY ABSENT: the Task 0
    # probe-failure counter (``composer.phase_8.probe_failed_total``)
    # is NOT a field on this container. Per W8-r2 / A5, it lives as a
    # module-local OTel counter in ``telemetry_phase8.py``
    # (``_PHASE_8_PROBE_FAILED_COUNTER``), constructed at module import
    # time via ``meter.create_counter``. Matches the existing
    # ``_PREFERENCES_PATCH_COUNTER`` pattern in
    # ``src/elspeth/web/preferences/service.py``. Motivation: remove the
    # bootstrap-order coupling that would otherwise require Task 1
    # Step 5 (this container extension) to land before Task 0 Step 2
    # could emit. See 20-phase-8-polish-and-telemetry.md Task 1 Step 4
    # module shape + §Risks "Phase 8 probe-failure counter bootstrap-
    # order coupling (W8-r2)".
    # B5 — Phase 5a dynamic-source emit (Sub-task 7b).
    source_dynamic_created_total: _Counter


def build_sessions_telemetry(*, meter: _Meter | None = None) -> _SessionsTelemetry:
    """Build a telemetry container.

    With ``meter=None`` (the default) returns ``_FakeCounter``
    instances; tests use this path. Production callers pass an OTel
    ``Meter`` (typed structurally as ``_Meter`` so we don't import
    ``opentelemetry.metrics`` here unnecessarily — the structural
    Protocol is satisfied by the real meter at runtime).
    """

    if meter is None:
        return _SessionsTelemetry(
            tool_row_tier1_violation_total=_FakeCounter(),
            state_rolled_back_during_persist_total=_FakeCounter(),
            tool_row_persist_failed_during_unwind_total=_FakeCounter(),
            tool_row_integrity_violation_total=_FakeCounter(),
            tool_call_cap_exceeded_total=_FakeCounter(),
            audit_grade_view_total=_FakeCounter(),
            audit_access_log_write_failed_total=_FakeCounter(),
            interpretation_rate_cap_exceeded_total=_FakeCounter(),
            interpretation_placeholder_unresolved_at_runtime_total=_FakeCounter(),
            progress_broadcast_dropped_total=_FakeCounter(),
            orphaned_runs_cancelled_total=_FakeCounter(),
            # Phase 8 counters.
            mode_opted_out_total=_FakeCounter(),
            mode_opted_in_total=_FakeCounter(),
            session_switched_total=_FakeCounter(),
            tutorial_started_total=_FakeCounter(),
            session_completed_total=_FakeCounter(),
            share_token_verify_failure_total=_FakeCounter(),
            share_link_expiry_hit_total=_FakeCounter(),
            interpretation_opt_out_total=_FakeCounter(),
            audit_fetch_failure_total=_FakeCounter(),
            source_dynamic_created_total=_FakeCounter(),
        )

    # Production wiring against the real OTel meter. The ``_Meter``
    # Protocol satisfies mypy without ``# type: ignore`` decorations
    # — the real OTel ``Meter.create_counter`` matches the structural
    # contract, and the returned counter satisfies ``_Counter``.
    return _SessionsTelemetry(
        tool_row_tier1_violation_total=meter.create_counter("composer.audit.tool_row_tier1_violation_total"),
        state_rolled_back_during_persist_total=meter.create_counter("composer.audit.state_rolled_back_during_persist_total"),
        tool_row_persist_failed_during_unwind_total=meter.create_counter("composer.audit.tool_row_persist_failed_during_unwind_total"),
        tool_row_integrity_violation_total=meter.create_counter("composer.audit.tool_row_integrity_violation_total"),
        tool_call_cap_exceeded_total=meter.create_counter("composer.tool_call_cap_exceeded_total"),
        audit_grade_view_total=meter.create_counter("composer.audit.audit_grade_view_total"),
        audit_access_log_write_failed_total=meter.create_counter("composer.audit.audit_access_log_write_failed_total"),
        interpretation_rate_cap_exceeded_total=meter.create_counter("composer.interpretation_rate_cap_exceeded_total"),
        interpretation_placeholder_unresolved_at_runtime_total=meter.create_counter(
            "composer.interpretation_placeholder_unresolved_at_runtime_total"
        ),
        progress_broadcast_dropped_total=meter.create_counter(
            "execution.progress.broadcast_dropped_total",
            description=(
                "Execution progress WebSocket broadcasts dropped or drained under client backpressure. Attributes: reason, run_id when available."
            ),
        ),
        orphaned_runs_cancelled_total=meter.create_counter(
            "execution.orphaned_runs_cancelled_total",
            description=(
                "Runs cancelled by startup or periodic orphan cleanup. Attributes: source, excluded_live_runs for periodic cleanup."
            ),
        ),
        # ── Phase 8 wire names (real-meter branch) ──
        # Naming: ``composer.<domain>.<verb>_total``. The
        # ``description`` keyword argument is supplied so the
        # Prometheus exposition (B1-r3 MeterProvider) carries
        # operator-readable HELP text for each metric.
        mode_opted_out_total=meter.create_counter(
            "composer.mode.opted_out_total",
            description=(
                "Composer account-level opt-outs from guided mode "
                "(default_mode set to 'freeform' on PATCH /api/composer-preferences). "
                "Post-state counter: fires on every PATCH whose body sets "
                "default_mode=freeform, regardless of prior state. "
                "Denominator is composer.preferences.patch_total."
            ),
        ),
        mode_opted_in_total=meter.create_counter(
            "composer.mode.opted_in_total",
            description=(
                "Composer account-level opt-ins to guided mode "
                "(default_mode set to 'guided' on PATCH /api/composer-preferences). "
                "Post-state counter; symmetric to composer.mode.opted_out_total."
            ),
        ),
        session_switched_total=meter.create_counter(
            "composer.session.switched_total",
            description=(
                "Per-session trust-mode switch on "
                "PATCH /api/sessions/{session_id}/composer/preferences. "
                "Attributes: from_mode, to_mode ∈ {explicit_approve, auto_commit} "
                "drawn from the session row's trust_mode CHECK constraint."
            ),
        ),
        tutorial_started_total=meter.create_counter(
            "composer.tutorial.started_total",
            description="Composer first-run tutorial started (Phase 4 surface; helper wired in Phase 8 for forward-fit).",
        ),
        session_completed_total=meter.create_counter(
            "composer.session.completed_total",
            description=(
                "Composer session terminated via a completion gesture "
                "(Phase 6). Attributes: completion_verb ∈ "
                "{mark_ready_for_review, export_yaml} — sourced from "
                "the CHECK constraint on "
                "composer_completion_events_table.event_type (see "
                "src/elspeth/web/sessions/models.py:735). The counter "
                "aggregates over committed audit rows in that table; "
                "no UI-only verbs (e.g. save_for_review) and no "
                "run-completion verbs (e.g. run_pipeline — see runs/) "
                "appear here, per the CLAUDE.md superset rule."
            ),
        ),
        share_token_verify_failure_total=meter.create_counter(
            "composer.share.token_verify_failure_total",
            description="Shareable-review token verification failure (B3 cohort a — Phase 6 surface).",
        ),
        share_link_expiry_hit_total=meter.create_counter(
            "composer.share.link_expiry_hit_total",
            description="Shareable-review link visited after expiry (B3 cohort a — Phase 6 surface).",
        ),
        interpretation_opt_out_total=meter.create_counter(
            "composer.interpretation.opt_out_total",
            description=(
                "Interpretation auto-opt-out (B3 cohort b1 — Phase 5b surface). "
                "Fires alongside the audit row whose interpretation_source "
                "is 'auto_interpreted_opt_out'."
            ),
        ),
        audit_fetch_failure_total=meter.create_counter(
            "composer.audit.fetch_failure_total",
            description=(
                "Audit-readiness panel fetch failure (B3 cohort b2 — Phase 2C surface). "
                "Telemetry-only signal; superset exception for non-decision read."
            ),
        ),
        source_dynamic_created_total=meter.create_counter(
            "composer.source.dynamic_created_total",
            description="Dynamic source created from chat (B5 — Phase 5a surface).",
        ),
    )
