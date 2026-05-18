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
``observed_value`` is intentionally NOT on the ``_Counter`` Protocol
— production OTel counters do not expose observation, and adding it
would force a structural lie.

**Ownership-vs-metric namespace.** The container type is named
``_SessionsTelemetry`` and the module lives under
``web/sessions/telemetry.py`` because ``SessionServiceImpl`` owns
the persistence counters. The OTel metric strings remain
``composer.audit.*`` because operators consume them as part of the
composer-progress surface, but metric naming does not justify an
import from ``web/sessions/service.py`` up into ``web/composer``.
Composer code may import the sessions-owned container or receive it
from app wiring; sessions code must not import composer-owned modules.
Phase 2 (redaction counters) and Phase 3 (compose-loop counters) may
extend this container only if ownership still belongs to the sessions
persistence surface; otherwise those phases add composer-owned
telemetry separately.
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
    ``build_sessions_telemetry`` uses for production wiring."""

    def create_counter(self, name: str) -> _Counter: ...


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
    # fail the runtime placeholder gate — the LLM transform's prompt_template
    # still carries one or more ``{{interpretation:<term>}}`` placeholders at
    # execute time, meaning the compose-loop never called
    # ``request_interpretation_review`` to resolve them. Carries attributes
    # ``{"node_id": str, "term": str}`` — explicitly NOT the prompt_template
    # value (may carry user-supplied content). Operational telemetry, not
    # audit-primary: the user-actionable RuntimeError surfaced to the
    # frontend is the primary record (an audit-primary
    # ``interpretation_events`` row WOULD have been the primary record had
    # the LLM fired ``request_interpretation_review`` — its absence is what
    # this counter catches). Purpose: catches LLM under-firing after a model
    # upgrade without waiting for offline eval refresh.
    interpretation_placeholder_unresolved_at_runtime_total: _Counter


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
    )
