"""Phase 8 emit helpers for composer mode and tutorial telemetry.

Wraps the ``_SessionsTelemetry`` counter container; helpers are pure
functions and call-sites pass the container in.

Import direction is not restricted: both composer code AND sessions
code may import these helpers. The pre-Phase-8 docstrings on
``sessions/telemetry.py`` and an earlier draft of this module said
"sessions code must not import composer-owned modules"; that
prohibition was a forward-looking aspiration that never matched
codebase reality (there are 31+ pre-existing ``from
elspeth.web.composer.*`` imports across ``src/elspeth/web/sessions/``).
The Phase 8b-1b cohort emit at ``sessions/service.py`` made the
dead-letter rule visibly inconsistent, so it has been retired. The
practical constraint is the tier-model layer rule (all three of
composer, sessions, audit_readiness are L3 application code; they
may import each other freely as long as imports flow within L3 and
respect the contract layer boundaries above).

Trust tier: Tier 2 throughout; inputs are Literal-typed; runtime guards
are offensive.

Module-local counter (W8-r2). ``_PHASE_8_PROBE_FAILED_COUNTER`` is
constructed at import time via ``metrics.get_meter(__name__)`` rather
than added as a slot on ``_SessionsTelemetry``. Rationale: this counter
is emitted by Task 0 Step 2's probe-failure path, which runs before any
container fixture exists; making it a container slot would have coupled
Task 0's emit ordering to Task 1's container-extension landing. The
module-local placement mirrors the existing ``_PREFERENCES_PATCH_COUNTER``
pattern in ``src/elspeth/web/preferences/service.py``. See
``docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md``
§"Phase 8 probe-failure counter bootstrap-order coupling (W8-r2)".

Vocabulary discipline (B1-r2). The ``_SessionTrustMode`` Literal is
intentionally scoped to per-session ``trust_mode`` values
(``explicit_approve`` / ``auto_commit``) drawn from the CHECK
constraint at ``sessions/models.py`` table-definition time. The
account-level helpers (``record_mode_opted_out`` /
``record_mode_opted_in``) take no mode kwarg and use no Literal —
they're post-state-only per §"Account-level scope narrowing
(B2.b — load-bearing)". The completion helper
(``record_session_completed``) uses a separate ``_CompletionVerb``
Literal whose value set MIRRORS the CHECK constraint on
``composer_completion_events_table.event_type`` at
``src/elspeth/web/sessions/models.py:735`` —
``mark_ready_for_review`` / ``export_yaml``. This is the DB-authoritative
audit vocabulary (Tier 1 trust-store; the audit row is the legal record
the counter aggregates over per CLAUDE.md superset rule). A pre-wire
draft of this helper carried a UI-facing vocabulary
(``save_for_review`` / ``run_pipeline`` / ``export_yaml``) plus an
``_AccountMode`` (``guided`` / ``freeform``) attribute; the overall-plan
reviewer for Sub-task 7c surfaced that ``save_for_review`` was UI vocab
that drifts from the DB audit row, and ``run_pipeline`` is a UX-level
verb that does NOT write a ``composer_completion_events_table`` row
(its audit lives under ``runs/``). Per the superset rule the counter
attributes must be a strict subset of audit-recorded reality; both
mismatches were resolved by aligning to the DB CHECK constraint. If a
future phase needs an aggregate over run-completions, file a separate
``composer.run.started_total`` counter that mirrors the ``runs``-table
event — do NOT re-broaden ``_CompletionVerb`` to include
``run_pipeline``.

OTel exporter failure handling (W5). Every ``record_*`` helper wraps
the underlying ``.add(...)`` call in ``try / except Exception`` that
swallows and returns ``None``. Telemetry is best-effort per CLAUDE.md
"Telemetry and Logging" — a broken exporter must not 500 a PATCH whose
audit row already wrote. The ``_assert_session_trust_mode`` /
``_assert_completion_verb`` ValueErrors are programmer-error guards
and intentionally escape (they fire BEFORE the try/except so input
validation still crashes loudly).
"""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics

from elspeth.web.sessions.telemetry import _SessionsTelemetry as SessionsTelemetry

__all__ = [
    "SessionsTelemetry",
    "record_audit_fetch_failure",
    "record_interpretation_opt_out",
    "record_mode_opted_in",
    "record_mode_opted_out",
    "record_session_completed",
    "record_session_switched",
    "record_share_link_expiry_hit",
    "record_share_token_verify_failure",
    "record_source_dynamic_created",
    "record_tutorial_started",
]


# ── Module-local probe-failure counter (W8-r2) ──────────────────────────
# Constructed at module import time via ``metrics.get_meter``. NOT a
# slot on ``_SessionsTelemetry`` — see module docstring rationale.
# Naming mirrors ``_PREFERENCES_PATCH_COUNTER`` in
# ``preferences/service.py``: SCREAMING_SNAKE_CASE constant name; the
# OTel metric name ends in ``_total`` per project convention.
_meter = metrics.get_meter(__name__)
_PHASE_8_PROBE_FAILED_COUNTER = _meter.create_counter(
    name="composer.phase_8.probe_failed_total",
    description=(
        "Phase 8 conditional probe missed an upstream phase surface. "
        "Attributes: phase (the Phase 8 task / sub-task identifier), "
        "probe (the specific upstream symbol or path the probe searched for). "
        "Signals 'this conditional task could not run', not an error."
    ),
)


# ── Per-session trust_mode vocabulary (B1-r2) ───────────────────────────
# Sourced from the CHECK constraint on ``user_sessions_table.trust_mode``
# (see ``src/elspeth/web/sessions/models.py``). Do NOT add ``guided`` /
# ``freeform`` / ``unknown`` — those are wrong vocabulary (account-level
# column) or fabricated values the column does not admit.
_SessionTrustMode = Literal["explicit_approve", "auto_commit"]
_KNOWN_SESSION_TRUST_MODES: frozenset[str] = frozenset({"explicit_approve", "auto_commit"})


# ── Completion verb vocabulary (Phase 6) ────────────────────────────────
# Sourced from the CHECK constraint on
# ``composer_completion_events_table.event_type`` at
# ``src/elspeth/web/sessions/models.py:735``:
#
#     CheckConstraint(
#         "event_type IN ('mark_ready_for_review', 'export_yaml')",
#         name="ck_composer_completion_events_type",
#     )
#
# The two values that appear in audit rows are the only two valid
# attribute values for ``composer.session.completed_total`` — the counter
# aggregates over those rows per the CLAUDE.md superset rule. UI-facing
# vocabulary (``save_for_review``) MUST NOT appear here; it would drift
# from the audit row and silently break aggregation. The Phase 6 UX
# verb ``run_pipeline`` is also intentionally absent: a pipeline run is
# recorded under the ``runs`` table, not the
# ``composer_completion_events_table``, so it has no audit row this
# counter could aggregate over. A future phase wanting a run-started
# aggregate must define a SEPARATE counter (e.g.
# ``composer.run.started_total``) over the runs table.
_CompletionVerb = Literal["mark_ready_for_review", "export_yaml"]
_KNOWN_COMPLETION_VERBS: frozenset[str] = frozenset({"mark_ready_for_review", "export_yaml"})


def _assert_session_trust_mode(name: str, value: str) -> None:
    """Offensive guard for the per-session ``trust_mode`` Literal.

    Raised as ``ValueError`` (programmer error, not a runtime/exporter
    failure). Wrapped BEFORE the W5 try/except so input validation
    propagates rather than being silently swallowed.
    """
    if value not in _KNOWN_SESSION_TRUST_MODES:
        raise ValueError(f"{name} must be one of {sorted(_KNOWN_SESSION_TRUST_MODES)!r}; got {value!r}")


def _assert_completion_verb(name: str, value: str) -> None:
    """Offensive guard for the Phase 6 completion-verb Literal.

    The accepted value set is the CHECK constraint on
    ``composer_completion_events_table.event_type`` (DB-authoritative).
    See module docstring §"Vocabulary discipline" for the superset-rule
    rationale.
    """
    if value not in _KNOWN_COMPLETION_VERBS:
        raise ValueError(f"{name} must be one of {sorted(_KNOWN_COMPLETION_VERBS)!r}; got {value!r}")


# ── Account-level mode opt-out / opt-in (B2.b post-state-only) ──────────


def record_mode_opted_out(tel: SessionsTelemetry) -> None:
    """Account-level opt-out of guided mode.

    Fires when ``default_mode=freeform`` is set on
    ``PATCH /api/composer-preferences``, regardless of the prior
    state. Post-state-only per §"Account-level scope narrowing
    (B2.b — load-bearing)" — no ``from_mode`` attribute, no
    transition-shaped audit event accompanies this emit, and the
    helper takes no kwargs. The design-doc-10 "opt-out rate" is a
    ratio over ``composer.preferences.patch_total``, not a
    transition-conditional count.
    """
    try:
        tel.mode_opted_out_total.add(1, attributes={})
    except Exception:
        return None
    return None


def record_mode_opted_in(tel: SessionsTelemetry) -> None:
    """Account-level opt-in to guided mode. Symmetric to
    ``record_mode_opted_out``; post-state-only, kwarg-free,
    attribute-free.
    """
    try:
        tel.mode_opted_in_total.add(1, attributes={})
    except Exception:
        return None
    return None


# ── Per-session trust_mode switch (transition-shaped, B1-extended) ──────


def record_session_switched(
    tel: SessionsTelemetry,
    *,
    from_mode: _SessionTrustMode,
    to_mode: _SessionTrustMode,
) -> None:
    """Per-session ``trust_mode`` transition.

    Fires on ``PATCH /api/sessions/{session_id}/composer/preferences``
    when the session's ``trust_mode`` column changes. Carries
    ``from_mode`` and ``to_mode`` attributes drawn from the per-session
    ``trust_mode`` CHECK constraint vocabulary — NOT the account-level
    ``default_composer_mode`` vocabulary (``guided`` / ``freeform``
    will be rejected by the assert below). The companion
    ``trust_mode.changed`` audit event carries both prior and new
    state under the B1 extension; this counter mirrors the audit shape
    (superset rule satisfied).
    """
    _assert_session_trust_mode("from_mode", from_mode)
    _assert_session_trust_mode("to_mode", to_mode)
    try:
        tel.session_switched_total.add(
            1,
            attributes={"from_mode": from_mode, "to_mode": to_mode},
        )
    except Exception:
        return None
    return None


# ── Tutorial helpers (Phase 4 surface; helpers shipped in Phase 8) ──────


def record_tutorial_started(tel: SessionsTelemetry) -> None:
    """Composer first-run tutorial started.

    Phase 4 emit site; the helper ships in Phase 8 for forward-fit.
    Attribute-free; no vocabulary to assert.
    """
    try:
        tel.tutorial_started_total.add(1, attributes={})
    except Exception:
        return None
    return None


# ── Completion gestures (Phase 6 surface) ───────────────────────────────


def record_session_completed(
    tel: SessionsTelemetry,
    *,
    completion_verb: _CompletionVerb,
) -> None:
    """Composer session terminated via a completion gesture.

    ``completion_verb`` is sourced from the CHECK constraint on
    ``composer_completion_events_table.event_type`` (see
    ``src/elspeth/web/sessions/models.py:735``). The valid values are
    ``mark_ready_for_review`` and ``export_yaml`` — those are the two
    completion gestures that write an audit row in the
    ``composer_completion_events_table``. Per the CLAUDE.md superset
    rule, the counter aggregates over committed audit rows, so its
    attribute set MUST be a strict subset of the audit-row vocabulary.

    No ``mode`` attribute is carried. An earlier draft tagged each emit
    with the user's account-level ``default_composer_mode``
    (``guided`` / ``freeform``); the overall-plan reviewer for Sub-task
    7c flagged that as additional state read at emit-time with no
    corresponding column on ``composer_completion_events_table`` — i.e.
    a telemetry attribute not present in the audit row, which is the
    exact superset-rule violation the rule exists to prevent.

    Audit primacy. Call sites MUST place this helper AFTER the
    ``engine.begin()`` block that writes the corresponding
    ``composer_completion_events_table`` row has exited. If the audit
    write raises, control never reaches the helper and the counter
    stays at zero — that's the structural enforcement of the primacy
    invariant.
    """
    _assert_completion_verb("completion_verb", completion_verb)
    try:
        tel.session_completed_total.add(
            1,
            attributes={"completion_verb": completion_verb},
        )
    except Exception:
        return None
    return None


# ── B3 cohort emits — wired by Phase 8b-1b at the upstream phases' sites


def record_share_token_verify_failure(tel: SessionsTelemetry) -> None:
    """B3 cohort (a) — Phase 6 shareable-review token-verify failure."""
    try:
        tel.share_token_verify_failure_total.add(1, attributes={})
    except Exception:
        return None
    return None


def record_share_link_expiry_hit(tel: SessionsTelemetry) -> None:
    """B3 cohort (a) — Phase 6 shareable-review link expiry hit."""
    try:
        tel.share_link_expiry_hit_total.add(1, attributes={})
    except Exception:
        return None
    return None


def record_interpretation_opt_out(tel: SessionsTelemetry) -> None:
    """B3 cohort (b1) — Phase 5b interpretation auto-opt-out.

    Fires alongside the audit row whose ``interpretation_source`` is
    ``auto_interpreted_opt_out``. Superset rule satisfied via the
    audit row.
    """
    try:
        tel.interpretation_opt_out_total.add(1, attributes={})
    except Exception:
        return None
    return None


def record_audit_fetch_failure(tel: SessionsTelemetry) -> None:
    """B3 cohort (b2) — Phase 2C audit-readiness fetch failure.

    Telemetry-only signal; superset exception for non-decision read
    (CLAUDE.md primacy rule). No audit row companion.
    """
    try:
        tel.audit_fetch_failure_total.add(1, attributes={})
    except Exception:
        return None
    return None


# ── B5 — Phase 5a dynamic-source emit ────────────────────────────────────


def record_source_dynamic_created(tel: SessionsTelemetry) -> None:
    """B5 — Phase 5a dynamic-source-from-chat creation.

    Fires when the composer creates a dynamic 1-row source from chat
    text (per ``project_composer_dynamic_source_from_chat`` memory).
    """
    try:
        tel.source_dynamic_created_total.add(1, attributes={})
    except Exception:
        return None
    return None
