"""Type contracts for composer progress events.

Defines the L0 surface of the composer progress channel:

- ``ComposerProgressPhase`` — the lifecycle-phase discriminator Literal
- ``ComposerProgressReason`` — the machine-readable failure-code Literal
- ``ComposerProgressEvent`` — the provider-safe pydantic event model
- ``ComposerProgressSink`` — the async event-emission callable shape

These live at L0 so ``web/composer/protocol.py`` can reference them
without forming the cycle ``web/composer/progress ->
web/composer/tools -> web/composer/protocol -> web/composer/progress``.

Pre-2026-05-23 this module held only the ``Reason`` / ``Sink`` aliases
and a ``TYPE_CHECKING`` import of ``ComposerProgressEvent`` from
``web/composer/progress.py`` to type the callable's parameter.  CodeQL
``py/unsafe-cyclic-import`` (error-severity, gated per
``.github/codeql/codeql-config.yml``) flagged the TYPE_CHECKING-deferred
import as a static-analysis-visible cycle even though no runtime cycle
exists.  Moving the event class down to L0 dissolves the cycle entirely
and is the architecturally-clean fix preferred by CLAUDE.md's "Layer
Dependency Rules → When a New Cross-Layer Need Arises" guidance: "Move
the code down. If the needed code has no upward dependencies, move it
to the lower layer."

The L3-dependent residue (``ComposerProgressSnapshot``, ``ComposerProgressRegistry``,
event-factory functions, tool-name helpers) stays in
``web/composer/progress.py`` because it pulls in threading, the
``is_discovery_tool`` tool helper, and per-tool headline copy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# No ``__all__`` declared. Several public names below are PEP 695 ``type``
# aliases; listing them in ``__all__`` trips CodeQL ``py/undefined-export``
# because that rule (as of 2026-05) does not model PEP 695 type-alias
# bindings as definitions. The project's convention for type-alias modules
# (``contracts/declaration_contracts.py``, ``contracts/enums.py``,
# ``plugins/transforms/*.py``) is to omit ``__all__`` entirely — every
# consumer imports by name, no wildcard imports exist anywhere in the
# project, so ``__all__`` carries no functional load on these modules.

COMPOSER_PROGRESS_MAX_EVIDENCE = 4
_MAX_PROGRESS_TEXT_CHARS = 180


type ComposerProgressPhase = Literal[
    "idle",
    "starting",
    "calling_model",
    "using_tools",
    "validating",
    "saving",
    "complete",
    "failed",
    "cancelled",
]


# Phases that indicate the composer is actively working on a request. The
# in-flight enumeration endpoint filters on this set so an operator polling
# /_active sees the live composer requests for their own sessions even if
# the SPA tab that posted them is no longer connected.
NON_TERMINAL_PROGRESS_PHASES: frozenset[ComposerProgressPhase] = frozenset(
    {
        "starting",
        "calling_model",
        "using_tools",
        "validating",
        "saving",
    }
)


# Stable machine-readable reason codes for composer progress events.
#
# Public taxonomy distinct from ComposerConvergenceError.budget_exhausted —
# the exception models which budget tripped (a private engine concept), this
# Literal is the public-facing UX/observability discriminator. They map but
# they are not the same enum: the convergence error contributes three of
# these codes; the others come from sibling exception classes or from
# success/idle sentinels.
#
# Required when phase == "failed" (enforced by the model_validator on
# ComposerProgressEvent) so a new failure site cannot ship without carrying
# a stable code. The frontend, structured logs, and the 422 response body
# all branch on this value.
type ComposerProgressReason = Literal[
    # Convergence sub-causes — split out from the single
    # ComposerConvergenceError class via its budget_exhausted discriminator.
    "convergence_composition_budget",
    "convergence_discovery_budget",
    "convergence_wall_clock_timeout",
    "tool_call_cap_exceeded",
    # Provider-side failures — LiteLLM exception families.
    "provider_auth_failed",
    "provider_unavailable",
    # Server-side plugin bug escaping execute_tool.
    "plugin_crash",
    # Runtime preflight failure (cached path-1 or post-compose path-2 —
    # users cannot act on the path distinction, so a single code).
    "runtime_preflight_failed",
    # Generic ComposerServiceError — prompt prep / availability / catch-all.
    "service_setup_failed",
    # Client closed the HTTP connection or operator cancelled the request
    # before the composer returned. Distinct from convergence_wall_clock_timeout
    # (server budget exceeded) so dashboards and audit can tell apart "the
    # client gave up" from "the server gave up". Required when phase ==
    # "cancelled" by the same model_validator that requires it on "failed".
    "client_cancelled",
    # Non-failure sentinels — every snapshot carries a code so observability
    # and the SPA never have to special-case None.
    "composer_idle",
    "composer_complete",
]


def _clean_required_text(value: str, *, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"composer progress {field_name} must contain visible text")
    if len(cleaned) > _MAX_PROGRESS_TEXT_CHARS:
        return cleaned[: _MAX_PROGRESS_TEXT_CHARS - 1].rstrip() + "."
    return cleaned


class _StrictProgressModel(BaseModel):
    """Strict model for system-owned progress snapshots."""

    model_config = ConfigDict(strict=True, extra="forbid")


class ComposerProgressEvent(_StrictProgressModel):
    """Provider-safe progress event emitted by the composer path."""

    phase: ComposerProgressPhase
    headline: str
    evidence: tuple[str, ...] = ()
    likely_next: str | None = None
    reason: ComposerProgressReason | None = None

    @field_validator("headline")
    @classmethod
    def _validate_headline(cls, value: str) -> str:
        return _clean_required_text(value, field_name="headline")

    @field_validator("likely_next")
    @classmethod
    def _validate_likely_next(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _clean_required_text(value, field_name="likely_next")

    @field_validator("evidence")
    @classmethod
    def _bound_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        bounded: list[str] = []
        for item in value:
            cleaned = _clean_required_text(item, field_name="evidence")
            bounded.append(cleaned)
            if len(bounded) == COMPOSER_PROGRESS_MAX_EVIDENCE:
                break
        return tuple(bounded)

    @model_validator(mode="after")
    def _require_reason_when_terminal_non_success(self) -> Self:
        # Mechanically forbids the drift the original bug exhibited: a
        # phase="failed" event was emitted with text-only differentiation
        # at three distinct sites and three sub-causes collapsed into one
        # generic message because nothing in the contract required a
        # discriminator. With this validator, a new failure site cannot
        # be added without choosing a code from ComposerProgressReason.
        # The same rule applies to phase == "cancelled" (added with the
        # in-flight observability work) — operator dashboards branch on
        # reason to distinguish client_cancelled from a future operator-
        # initiated cancel without parsing the headline.
        # Other phases keep reason optional — they're status pings, not
        # routing decisions.
        if self.phase in ("failed", "cancelled") and self.reason is None:
            raise ValueError(
                "ComposerProgressEvent.reason is required when phase is 'failed' or "
                "'cancelled' so the frontend, audit logs, and HTTP response body can "
                "branch on a stable taxonomy instead of free-text headline parsing."
            )
        return self


type ComposerProgressSink = Callable[[ComposerProgressEvent], Awaitable[None]]
