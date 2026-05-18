"""Phase 8 emit helpers for composer mode and tutorial telemetry.

Wraps the ``_SessionsTelemetry`` counter container; helpers are pure
functions and call-sites pass the container in. Composer imports this
module; sessions code must not (ownership-vs-metric rule per
``sessions/telemetry.py``).

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
(``record_session_completed``) uses a separate ``_AccountMode``
Literal (``guided`` / ``freeform``) because completion is reported on
account-level mode, not per-session trust_mode. A pass-1 draft used a
single shared mode Literal across both surfaces; B1-r2 caught that the
per-session helper would assert-fail on every emit because its inputs
come from a different column with a disjoint value set.

OTel exporter failure handling (W5). Every ``record_*`` helper wraps
the underlying ``.add(...)`` call in ``try / except Exception`` that
swallows and returns ``None``. Telemetry is best-effort per CLAUDE.md
"Telemetry and Logging" — a broken exporter must not 500 a PATCH whose
audit row already wrote. The ``_assert_session_trust_mode`` /
``_assert_account_mode`` ValueErrors are programmer-error guards and
intentionally escape (they fire BEFORE the try/except so input
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
    "record_tutorial_completed",
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


# ── Account-level mode vocabulary (completion gestures, Phase 6) ────────
# Sourced from the CHECK constraint on
# ``user_preferences_table.default_composer_mode``. Distinct from
# ``_SessionTrustMode`` — see module docstring vocabulary discipline.
_AccountMode = Literal["guided", "freeform"]
_KNOWN_ACCOUNT_MODES: frozenset[str] = frozenset({"guided", "freeform"})


# ── Completion verb vocabulary (Phase 6) ────────────────────────────────
_CompletionVerb = Literal["save_for_review", "run_pipeline", "export_yaml"]
_KNOWN_COMPLETION_VERBS: frozenset[str] = frozenset({"save_for_review", "run_pipeline", "export_yaml"})


def _assert_session_trust_mode(name: str, value: str) -> None:
    """Offensive guard for the per-session ``trust_mode`` Literal.

    Raised as ``ValueError`` (programmer error, not a runtime/exporter
    failure). Wrapped BEFORE the W5 try/except so input validation
    propagates rather than being silently swallowed.
    """
    if value not in _KNOWN_SESSION_TRUST_MODES:
        raise ValueError(f"{name} must be one of {sorted(_KNOWN_SESSION_TRUST_MODES)!r}; got {value!r}")


def _assert_account_mode(name: str, value: str) -> None:
    """Offensive guard for the account-level ``default_composer_mode`` Literal."""
    if value not in _KNOWN_ACCOUNT_MODES:
        raise ValueError(f"{name} must be one of {sorted(_KNOWN_ACCOUNT_MODES)!r}; got {value!r}")


def _assert_completion_verb(name: str, value: str) -> None:
    """Offensive guard for the Phase 6 completion-verb Literal."""
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


def record_tutorial_completed(tel: SessionsTelemetry) -> None:
    """Composer first-run tutorial completed.

    Phase 4 emit site; helper shipped in Phase 8 for forward-fit.
    NOTE: there is no ``record_tutorial_replayed`` counterpart —
    ``composer.tutorial.replayed_total`` is Phase 9 deferred per
    Decision 2 / Option C. See
    ``docs/composer/ux-redesign-2026-05/21-phase-9-followups.md``.
    """
    try:
        tel.tutorial_completed_total.add(1, attributes={})
    except Exception:
        return None
    return None


# ── Completion gestures (Phase 6 surface) ───────────────────────────────


def record_session_completed(
    tel: SessionsTelemetry,
    *,
    mode: _AccountMode,
    completion_verb: _CompletionVerb,
) -> None:
    """Composer session terminated via a completion gesture.

    ``mode`` is account-level vocabulary (``guided`` / ``freeform``)
    because completion is reported on account-level mode (the user's
    persisted default at completion time), NOT on the per-session
    ``trust_mode``. See module docstring vocabulary discipline.

    ``completion_verb`` enumerates the Phase 6 completion gestures
    documented in ``09-completion-gestures.md``.
    """
    _assert_account_mode("mode", mode)
    _assert_completion_verb("completion_verb", completion_verb)
    try:
        tel.session_completed_total.add(
            1,
            attributes={"mode": mode, "completion_verb": completion_verb},
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
