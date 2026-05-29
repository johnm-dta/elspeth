"""Decay-sweep judge across existing allowlist entries (Slice 3 of cicd-judge-cli prototype).

``reaudit`` re-runs the Opus judge against entries the allowlist already
carries. It is *read-only* on the allowlist — it does not mutate YAML.
Its output is a structured report classifying, for each entry, whether
the fresh judge verdict still agrees with the entry's stored verdict.

Why this exists (from the prototype plan, "Decay sweep (Slice 3 of
prototype)"): the largest failure mode of the pre-judge allowlist
practice was that self-attested entries were never re-reviewed. Code
changes around a suppression site but the entry stays. Override
verdicts pile up. The decay sweep closes that loop: at renewal time
(or on a periodic schedule), every entry has to survive a fresh judge
pass.

Read-only by design. The operator decides what to do with the report —
this module never writes back to YAML. Three reasons:

1. Mutating entries on the basis of a single fresh judge response
   without human review would defeat the audit purpose: the original
   model verdict is itself a load-bearing audit primitive, and silently
   overwriting it with a fresher one erases history.
2. The fresh judge call is itself stochastic; a single divergent
   verdict is not enough signal to re-classify an entry. The operator
   may want to inspect multiple verdicts or refresh the rationale
   before acting.
3. A reaudit run for the entire ~700-entry allowlist is expensive
   (Opus calls). Operators sweep incrementally with ``--limit`` and
   ``--since`` — the report is the durable artefact they triage
   against, not a transient screen.

Boundary clarification: reaudit handles the *read* side of an entry's
lifecycle. The complementary *write* path lives in ``cli._run_justify``
(new entry, judge gates the write). The two paths share the judge call
shape (``JudgeRequest`` / ``JudgeResponse``) but their entry-lifecycle
roles are distinct.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    load_allowlist,
)
from elspeth_lints.core.judge import JudgeContractError, JudgeRequest, JudgeTransportError, call_judge

if TYPE_CHECKING:
    # ``RedactionRecord`` is the audit primitive emitted by the
    # source-excerpt scrubber; we type the ``ReauditOutcome.excerpt_redactions``
    # field precisely without paying the import at runtime. The runtime
    # path imports the scrubber lazily inside ``_reaudit_one_entry`` (the
    # only site that constructs ``RedactionRecord`` instances) so the
    # circular-import surface this TYPE_CHECKING guard previously
    # protected against — module-init time `reaudit` -> `source_excerpt`
    # -> (any future shared types) -> back to `reaudit` — stays closed.
    from elspeth_lints.core.source_excerpt import RedactionRecord


class ReauditDivergence(StrEnum):
    """How a fresh judge verdict relates to an entry's stored verdict.

    Each value is a distinct operator-actionable signal. The enum is
    StrEnum (not auto-numbered) so the JSON / markdown serialisation is
    stable and human-readable.

    Mapping (entry.judge_verdict, entry.judge_model_verdict, fresh) →
    divergence (referenced for code review against the docstring):

    * (None, _,                ACCEPTED) → PRE_JUDGE_FRESH_ACCEPT
    * (None, _,                BLOCKED)  → PRE_JUDGE_FRESH_BLOCK
    * (ACCEPTED, None,         ACCEPTED) → STILL_AGREES
    * (ACCEPTED, None,         BLOCKED)  → WAS_ACCEPTED_NOW_BLOCKED
    * (OVERRIDDEN, BLOCKED,    ACCEPTED) → OVERRIDE_NO_LONGER_NEEDED
    * (OVERRIDDEN, BLOCKED,    BLOCKED)  → OVERRIDE_STILL_NEEDED
    * (OVERRIDDEN, ACCEPTED,   ACCEPTED) → STILL_AGREES
    * (OVERRIDDEN, ACCEPTED,   BLOCKED)  → WAS_ACCEPTED_NOW_BLOCKED
      (the entry's effective verdict is the override-of-an-ACCEPTED,
      which behaves identically to a plain ACCEPTED for divergence
      classification purposes)

    ``JudgeVerdict.BLOCKED`` never appears on the *prior* side: it is an
    in-memory runtime verdict that means the entry was rejected at write
    time and never persisted. The allowlist loader rejects BLOCKED on
    load (see ``_optional_judge_verdict`` / ``_validate_judge_metadata_
    atomic``), so reaudit can rely on prior verdicts being one of
    ``None``, ``ACCEPTED``, or ``OVERRIDDEN_BY_OPERATOR``.

    ``ENTRY_OBSOLETE`` is the off-tree path: the entry's underlying
    finding no longer exists in the source. The judge is *not* called
    for these — there is nothing for the model to evaluate.

    ``JUDGE_CALL_FAILED`` is the judge-boundary failure path: the judge
    call raised an SDK-level transport error (network, timeout,
    rate-limit, 5xx) or the model response violated the output
    contract. The entry could not be re-judged on this sweep. The entry
    is *not* obsolete and its prior verdict is *not* refreshed — the
    operator must rerun once the transport/contract problem is resolved.
    The exception classname + message is captured in ``fresh_rationale``
    so the report carries the diagnostic without needing the original
    stderr. Closes elspeth-9a4e54cc01 / C3-2.
    """

    STILL_AGREES = "STILL_AGREES"
    OVERRIDE_NO_LONGER_NEEDED = "OVERRIDE_NO_LONGER_NEEDED"
    OVERRIDE_STILL_NEEDED = "OVERRIDE_STILL_NEEDED"
    WAS_ACCEPTED_NOW_BLOCKED = "WAS_ACCEPTED_NOW_BLOCKED"
    PRE_JUDGE_FRESH_BLOCK = "PRE_JUDGE_FRESH_BLOCK"
    PRE_JUDGE_FRESH_ACCEPT = "PRE_JUDGE_FRESH_ACCEPT"
    ENTRY_OBSOLETE = "ENTRY_OBSOLETE"
    JUDGE_CALL_FAILED = "JUDGE_CALL_FAILED"
    # The entry's path resolved outside ``--root`` after symlink and
    # ``..`` normalisation. The only way to reach this branch is a
    # forged or tampered allowlist entry key — an exfiltration attempt
    # via the source-excerpt channel. Distinct from JUDGE_CALL_FAILED
    # (which the operator reads as "rerun later"); this signal means
    # "investigate the YAML for tampering" and never goes away on a
    # naive rerun. Closes elspeth-ebb2b88753 / C3-4.
    SOURCE_EXCERPT_REJECTED = "SOURCE_EXCERPT_REJECTED"
    # The judge returned a fresh verdict, but the stored/fresh verdict
    # tuple could not be mapped by the divergence matrix. This is
    # per-entry data/schema corruption, not a system-level sweep failure.
    JUDGE_CLASSIFICATION_FAILED = "JUDGE_CLASSIFICATION_FAILED"
    # The entry carries a judge_recorded_at later than the reaudit
    # reference time. A future-dated audit timestamp can evade --since
    # windows, so it is surfaced as a tampering/clock-skew signal before
    # the judge is called.
    FUTURE_DATED_ENTRY = "FUTURE_DATED_ENTRY"
    # The current scanner returned more than one finding whose canonical key
    # matches the allowlist entry. Picking one would hide duplicate-key drift.
    AMBIGUOUS_FINDING_MATCH = "AMBIGUOUS_FINDING_MATCH"


class ReauditCause(StrEnum):
    """Triage cause axis for a reaudit outcome.

    ``ReauditDivergence`` remains the stored/fresh verdict transition.
    This enum is a separate operator-facing cause class. It is deliberately
    conservative: a single fresh model response cannot prove whether a verdict
    flip came from policy drift, model drift, or residual sampling noise, so
    those cases stay grouped as ``MODEL_NOISE_OR_POLICY_DRIFT`` rather than
    fabricating a more precise root cause.
    """

    NO_CHANGE = "NO_CHANGE"
    MODEL_NOISE_OR_POLICY_DRIFT = "MODEL_NOISE_OR_POLICY_DRIFT"
    OPERATOR_OVERRIDE_RECHECK = "OPERATOR_OVERRIDE_RECHECK"
    PRE_JUDGE_BASELINE = "PRE_JUDGE_BASELINE"
    CODE_DRIFT = "CODE_DRIFT"
    SOURCE_EVIDENCE_REJECTED = "SOURCE_EVIDENCE_REJECTED"
    JUDGE_BOUNDARY_FAILURE = "JUDGE_BOUNDARY_FAILURE"
    AUDIT_METADATA_ANOMALY = "AUDIT_METADATA_ANOMALY"
    SCANNER_AMBIGUITY = "SCANNER_AMBIGUITY"

    @classmethod
    def for_divergence(cls, divergence: ReauditDivergence) -> ReauditCause:
        """Return the conservative cause class for a divergence label."""
        if divergence is ReauditDivergence.STILL_AGREES:
            return cls.NO_CHANGE
        if divergence is ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED:
            return cls.MODEL_NOISE_OR_POLICY_DRIFT
        if divergence in {
            ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED,
            ReauditDivergence.OVERRIDE_STILL_NEEDED,
        }:
            return cls.OPERATOR_OVERRIDE_RECHECK
        if divergence in {
            ReauditDivergence.PRE_JUDGE_FRESH_BLOCK,
            ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT,
        }:
            return cls.PRE_JUDGE_BASELINE
        if divergence is ReauditDivergence.ENTRY_OBSOLETE:
            return cls.CODE_DRIFT
        if divergence is ReauditDivergence.SOURCE_EXCERPT_REJECTED:
            return cls.SOURCE_EVIDENCE_REJECTED
        if divergence is ReauditDivergence.JUDGE_CALL_FAILED:
            return cls.JUDGE_BOUNDARY_FAILURE
        if divergence in {
            ReauditDivergence.JUDGE_CLASSIFICATION_FAILED,
            ReauditDivergence.FUTURE_DATED_ENTRY,
        }:
            return cls.AUDIT_METADATA_ANOMALY
        if divergence is ReauditDivergence.AMBIGUOUS_FINDING_MATCH:
            return cls.SCANNER_AMBIGUITY
        raise ReauditError(f"no reaudit cause mapping for divergence {divergence!r}")


# Operator-actionable severity ranking. Lower values surface first in
# the markdown report so the most urgent debt is at the top.
#
# JUDGE_CALL_FAILED ranks high: the entry was *not* re-judged. Until
# the operator resolves the transport/contract failure and reruns, the
# entry's decay status is unknown — that ignorance is more urgent than
# any verdict-change signal below it. Closes elspeth-9a4e54cc01 / C3-2.
_DIVERGENCE_ORDER: dict[ReauditDivergence, int] = {
    # SOURCE_EXCERPT_REJECTED ranks first: a forged path is a SECURITY
    # signal (exfiltration attempt). It outranks JUDGE_CALL_FAILED
    # because a transport hiccup may resolve on rerun; a forged path
    # will not, and the operator's response to it is qualitatively
    # different (investigate the YAML, not the network). Closes
    # elspeth-ebb2b88753 / C3-4.
    ReauditDivergence.SOURCE_EXCERPT_REJECTED: -1,
    ReauditDivergence.FUTURE_DATED_ENTRY: 0,
    ReauditDivergence.JUDGE_CALL_FAILED: 1,
    ReauditDivergence.JUDGE_CLASSIFICATION_FAILED: 2,
    ReauditDivergence.AMBIGUOUS_FINDING_MATCH: 3,
    ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED: 4,
    ReauditDivergence.PRE_JUDGE_FRESH_BLOCK: 5,
    ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED: 6,
    ReauditDivergence.ENTRY_OBSOLETE: 7,
    ReauditDivergence.OVERRIDE_STILL_NEEDED: 8,
    ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT: 9,
    ReauditDivergence.STILL_AGREES: 10,
}


@dataclass(frozen=True, slots=True)
class ReauditOutcome:
    """One entry's reaudit result.

    ``fresh_verdict`` / ``fresh_recorded_at`` / ``fresh_model_id`` are ``None`` for
    ``ENTRY_OBSOLETE`` (no judge call was made), ``FUTURE_DATED_ENTRY``
    (the timestamp failed before the judge call), and for
    ``JUDGE_CALL_FAILED`` (judge call was attempted but raised a
    transport/contract error). ``fresh_rationale`` is ``None`` for
    ``ENTRY_OBSOLETE`` but carries the captured exception classname +
    message for ``JUDGE_CALL_FAILED`` so the failure diagnostic is
    durable on the report. All other divergence values carry a fully
    populated fresh-verdict triple.

    ``code_snapshot`` is the surrounding code the judge saw, recorded
    verbatim so the report is independently re-readable months later
    without needing the source tree at the same commit. For
    ``JUDGE_CALL_FAILED`` the snapshot is still populated (we extracted
    it before the failing call), preserving "what the operator would
    have judged" even though no verdict landed.
    """

    entry: AllowlistEntry
    original_verdict: JudgeVerdict | None
    original_model_verdict: JudgeVerdict | None
    fresh_verdict: JudgeVerdict | None
    fresh_rationale: str | None
    fresh_recorded_at: datetime | None
    divergence: ReauditDivergence
    cause: ReauditCause
    code_snapshot: str
    fresh_model_id: str | None = None
    # Cost telemetry for the fresh judge boundary. ``judge_call_attempted``
    # is true even when the transport/contract boundary failed before
    # returning token usage; token fields remain ``None`` in that case.
    judge_call_attempted: bool = False
    fresh_prompt_tokens_total: int | None = None
    fresh_prompt_tokens_cached: int | None = None
    # Secrets-scrubber audit record (closes elspeth-ebb2b88753 / C2-2
    # on the sweep path). Each entry's source excerpt passes through
    # ``source_excerpt.scrub_secrets`` before reaching the judge; any
    # redactions made are captured here so the sidecar trail preserves
    # "n bytes scrubbed for pattern Y" without persisting the secret
    # bytes themselves. Empty tuple = scrubber ran clean. Populated for
    # SOURCE_EXCERPT_REJECTED is by definition impossible (the path
    # check fails before the scrubber runs); the field stays empty in
    # that case.
    excerpt_redactions: tuple[RedactionRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ReauditReport:
    """Aggregate report of a reaudit sweep.

    ``outcomes`` is a tuple (immutable container field) so the report
    can be freely passed around without defensive copies. ``summary``
    is a frozen mapping of divergence-name to count.

    ``entries_dispatched`` is the count of entries the sweep started
    processing. It is incremented as the per-entry loop *enters* each
    iteration, before any work (scan, judge call, classification) runs,
    so it reflects intent rather than success. ``total_entries`` is the
    count of entries surviving the upstream filters — the planned
    dispatch ceiling. A sweep that ran to completion has
    ``entries_dispatched == total_entries``. A sweep killed mid-loop
    (process killed, machine rebooted, uncaught exception above the
    per-entry boundary) has ``entries_dispatched < total_entries``: the
    delta is the count of entries that were never reaudited at all.

    Renderers surface that delta as an "INCOMPLETE SWEEP" banner on the
    text / markdown reports and as a discrete ``incomplete_sweep``
    object on the JSON report. Closes elspeth-9a4e54cc01 / C3-3.
    """

    outcomes: tuple[ReauditOutcome, ...]
    summary: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    entries_dispatched: int = 0
    total_entries: int = 0
    entries_skipped_by_rule_filter: int = 0
    judge_calls_attempted: int = 0
    max_judge_calls: int | None = None
    prompt_tokens_total: int = 0
    prompt_tokens_cached: int | None = 0

    @property
    def prompt_tokens_uncached(self) -> int | None:
        """Return uncached prompt tokens, or ``None`` when cache telemetry is absent."""
        if self.prompt_tokens_cached is None:
            return None
        return max(self.prompt_tokens_total - self.prompt_tokens_cached, 0)

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[ReauditOutcome],
        *,
        entries_dispatched: int | None = None,
        total_entries: int | None = None,
        entries_skipped_by_rule_filter: int = 0,
        max_judge_calls: int | None = None,
    ) -> ReauditReport:
        """Build a report from a sequence of outcomes, computing the summary.

        ``entries_dispatched`` and ``total_entries`` default to
        ``len(outcomes)`` when omitted, which is the "sweep ran to
        completion and every dispatched entry produced an outcome"
        case. Callers tracking partial-sweep state pass them
        explicitly: ``entries_dispatched`` is the dispatch count
        observed at the point the loop exited (whether cleanly or
        otherwise), and ``total_entries`` is the planned dispatch
        ceiling computed before the loop ran.
        """
        counts: dict[str, int] = {member.value: 0 for member in ReauditDivergence}
        for outcome in outcomes:
            counts[outcome.divergence.value] += 1
        # Sort summary by severity order (most urgent first) for stable
        # display in JSON / markdown.
        ordered = sorted(
            counts.items(),
            key=lambda kv: _DIVERGENCE_ORDER[ReauditDivergence(kv[0])],
        )
        dispatched = len(outcomes) if entries_dispatched is None else entries_dispatched
        total = len(outcomes) if total_entries is None else total_entries
        judge_calls_attempted = sum(1 for outcome in outcomes if outcome.judge_call_attempted)
        prompt_tokens_total = sum(outcome.fresh_prompt_tokens_total or 0 for outcome in outcomes)
        prompt_tokens_cached = _aggregate_cached_prompt_tokens(outcomes)
        return cls(
            outcomes=tuple(outcomes),
            summary=tuple(ordered),
            entries_dispatched=dispatched,
            total_entries=total,
            entries_skipped_by_rule_filter=entries_skipped_by_rule_filter,
            judge_calls_attempted=judge_calls_attempted,
            max_judge_calls=max_judge_calls,
            prompt_tokens_total=prompt_tokens_total,
            prompt_tokens_cached=prompt_tokens_cached,
        )


@dataclass(frozen=True, slots=True)
class ReauditProgress:
    """Operator-visible progress snapshot emitted after each durable outcome."""

    entries_dispatched: int
    total_entries: int
    judge_calls_attempted: int
    max_judge_calls: int | None
    prompt_tokens_total: int
    prompt_tokens_cached: int | None

    @property
    def prompt_tokens_uncached(self) -> int | None:
        if self.prompt_tokens_cached is None:
            return None
        return max(self.prompt_tokens_total - self.prompt_tokens_cached, 0)

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[ReauditOutcome],
        *,
        entries_dispatched: int,
        total_entries: int,
        max_judge_calls: int | None,
    ) -> ReauditProgress:
        return cls(
            entries_dispatched=entries_dispatched,
            total_entries=total_entries,
            judge_calls_attempted=sum(1 for outcome in outcomes if outcome.judge_call_attempted),
            max_judge_calls=max_judge_calls,
            prompt_tokens_total=sum(outcome.fresh_prompt_tokens_total or 0 for outcome in outcomes),
            prompt_tokens_cached=_aggregate_cached_prompt_tokens(outcomes),
        )


def _aggregate_cached_prompt_tokens(outcomes: Sequence[ReauditOutcome]) -> int | None:
    """Aggregate cached-token telemetry without fabricating unknown values."""
    total = 0
    for outcome in outcomes:
        if outcome.fresh_prompt_tokens_total is None:
            continue
        if outcome.fresh_prompt_tokens_cached is None:
            return None
        total += outcome.fresh_prompt_tokens_cached
    return total


# =========================================================================
# Orchestrator
# =========================================================================


# Rules a per-rule reaudit sweep is allowed to target via ``--rule``.
#
# Originally this set was ``frozenset({"trust_tier.tier_model"})`` (the
# prototype's single supported rule). Convergent panel finding C2 flagged
# the artificial restriction: judge gating tier_model entries while
# leaving the other twelve ``enforce_*`` allowlist surfaces ungated creates
# a friction-displacement loop — new suppression activity routes to the
# unguarded rules, tier_model's metric goes green while aggregate debt
# grows.
#
# The expanded set is derived from ``BUILTIN_RULES`` so additions stay in
# lockstep with the registered rule catalogue. Exclusions are explicit
# and load-bearing:
#
# * ``audit_evidence.nominal_base`` keeps a private allowlist format
#   (``allow_classes:`` consumed by ``ClassAllowlist``) that is NOT the
#   standard ``AllowlistEntry`` shape ``load_allowlist`` parses. Its
#   entries carry no ``judge_*`` fields by construction; reaudit has
#   nothing to re-judge until the format migrates.
# * ``meta.no-new-bespoke-cicd-enforcer`` is a project-policy gate, not
#   a code-pattern lint. It does not emit per-site findings with the
#   key shape reaudit can dispatch against.
#
# Adding a new rule package: ensure its YAML files use ``entries:`` (not
# a private legacy format) and that ``Rule.analyze`` produces findings
# whose ``canonical_key`` matches the entry key shape. Listing in
# ``_EXCLUDED_FROM_REAUDIT`` is the only place a rule needs to be
# carved out.
_EXCLUDED_FROM_REAUDIT: frozenset[str] = frozenset(
    {
        "audit_evidence.nominal_base",
        "meta.no-new-bespoke-cicd-enforcer",
    }
)


def _supported_rules() -> frozenset[str]:
    """Return the set of rule ids reaudit will accept via ``--rule``.

    Lazily computed: importing ``BUILTIN_RULES`` is non-trivial (each
    rule package's ``__init__`` runs), so this is called by
    ``reaudit_entries`` once per invocation rather than at module
    import time. The result is stable per-process: the rule registry
    is immutable for the lifetime of the interpreter.
    """
    from elspeth_lints.rules import BUILTIN_RULES

    return frozenset(rule.id for rule in BUILTIN_RULES) - _EXCLUDED_FROM_REAUDIT


class ReauditError(RuntimeError):
    """The reaudit run cannot proceed.

    Distinct from ``JudgeConfigurationError`` (which signals missing
    API setup) — this signals an operator-actionable misconfiguration
    of the reaudit command itself (unknown rule package, bad allowlist
    directory, malformed entry key).
    """


class AmbiguousFindingMatchError(ReauditError):
    """The current scan produced multiple findings for one allowlist key."""


class _JudgeCallBudgetExhausted(RuntimeError):
    """Internal control-flow signal: this invocation's judge-call budget is spent."""


@dataclass(slots=True)
class _JudgeCallBudget:
    """Mutable per-invocation budget for expensive judge-boundary calls."""

    max_calls: int | None
    calls_attempted: int = 0

    def reserve_call(self) -> None:
        if self.max_calls is not None and self.calls_attempted >= self.max_calls:
            raise _JudgeCallBudgetExhausted
        self.calls_attempted += 1


@dataclass(frozen=True, slots=True)
class _VocabularySpec:
    """Where a rule package exposes the finding ids its allowlist may contain."""

    module_path: str
    attr_names: tuple[str, ...]
    use_mapping_keys: bool = False
    use_collection_values: bool = False


_RULE_VOCABULARY_REGISTRY: dict[str, _VocabularySpec] = {
    "trust_tier.tier_model": _VocabularySpec(
        "elspeth_lints.rules.trust_tier.tier_model.rule",
        ("RULES",),
        use_mapping_keys=True,
    ),
    "immutability.freeze_guards": _VocabularySpec(
        "elspeth_lints.rules.immutability.freeze_guards.rule",
        ("RULES",),
        use_mapping_keys=True,
    ),
    "immutability.frozen_annotations": _VocabularySpec(
        "elspeth_lints.rules.immutability.frozen_annotations.rule",
        ("RULE_ID",),
    ),
    "audit_evidence.guard_symmetry": _VocabularySpec(
        "elspeth_lints.rules.audit_evidence.guard_symmetry.rule",
        ("LEGACY_RULE_ID",),
    ),
    "audit_evidence.gve_attribution": _VocabularySpec(
        "elspeth_lints.rules.audit_evidence.gve_attribution.rule",
        ("LEGACY_RULE_ID",),
    ),
    "audit_evidence.tier_1_decoration": _VocabularySpec(
        "elspeth_lints.rules.audit_evidence.tier_1_decoration.metadata",
        ("RULE_TDE1", "RULE_TDE2"),
    ),
    "plugin_contract.component_type": _VocabularySpec(
        "elspeth_lints.rules.plugin_contract.component_type.rule",
        ("LEGACY_RULE_ID",),
    ),
    "plugin_contract.options_metadata": _VocabularySpec(
        "elspeth_lints.rules.plugin_contract.options_metadata.rule",
        ("RULE_ID",),
    ),
    "plugin_contract.plugin_hashes": _VocabularySpec(
        "elspeth_lints.rules.plugin_contract.plugin_hashes.rule",
        ("RULE_ID",),
    ),
    "composer.catch_order": _VocabularySpec(
        "elspeth_lints.rules.composer.catch_order.rule",
        ("LEGACY_RULE_ID",),
    ),
    "composer.exception_channel": _VocabularySpec(
        "elspeth_lints.rules.composer.exception_channel.rule",
        ("LEGACY_RULE_ID",),
    ),
    "manifest.contract_manifest": _VocabularySpec(
        "elspeth_lints.rules.manifest.contract_manifest.rule",
        ("RULE_ID",),
    ),
    "manifest.symbol_inventory": _VocabularySpec(
        "elspeth_lints.rules.manifest.symbol_inventory.rule",
        ("RULE_ID",),
    ),
    "manifest.test_to_source_mapping": _VocabularySpec(
        "elspeth_lints.rules.manifest.test_to_source_mapping.rule",
        ("RULE_ID",),
    ),
    "trust_boundary.tests": _VocabularySpec(
        "elspeth_lints.rules.trust_boundary.tests.rule",
        ("_ALLOWLIST_RULE_IDS",),
        use_collection_values=True,
    ),
    "trust_boundary.scope": _VocabularySpec(
        "elspeth_lints.rules.trust_boundary.scope.rule",
        ("_ALLOWLIST_RULE_IDS",),
        use_collection_values=True,
    ),
    "trust_boundary.tier": _VocabularySpec(
        "elspeth_lints.rules.trust_boundary.tier.rule",
        ("_ALLOWLIST_RULE_IDS",),
        use_collection_values=True,
    ),
}


def reaudit_entries(
    *,
    root: Path,
    allowlist_dir: Path,
    rule_filter: str,
    since: datetime | None,
    limit: int | None,
    include_pre_judge: bool,
    max_calls: int | None = None,
    sidecar_writer: Any = None,
    pre_classified_keys: frozenset[str] | None = None,
    pre_classified_outcomes: Sequence[ReauditOutcome] = (),
    reference_time: datetime | None = None,
    progress_callback: Callable[[ReauditProgress], None] | None = None,
) -> ReauditReport:
    """Re-judge entries in ``allowlist_dir`` against current source in ``root``.

    Iteration order: filename-sorted within the directory, YAML order
    within each file (the order ``load_allowlist`` returns). Filters
    apply in the documented order:

    1. ``rule_filter`` — entries whose key encodes a rule outside the
       supported scanner are silently skipped (defensive at boundary
       since the YAML may carry entries from other rule packages once
       wardline-style multi-rule allowlists land).
    2. ``include_pre_judge`` — entries with ``judge_verdict is None``
       are skipped unless this flag is set. Default off because the
       pre-judge corpus is ~700 entries and routine sweeps target the
       newer judge-gated ones.
    3. ``since`` — entries whose ``judge_recorded_at`` is at or after
       this datetime are skipped (their judgment is fresh; no decay).
       Pre-judge entries (no ``judge_recorded_at``) always pass this
       filter because there is no fresh judgment to defer to.
    4. ``limit`` — only the first N entries that survived prior filters
       are reaudited.

    ``max_calls`` is a separate per-invocation spend guard. It caps
    actual judge-boundary calls, not the filtered entry list. Entries
    classified before the judge boundary (obsolete, future-dated,
    unsafe source-excerpt path) do not consume this budget. When the
    budget is exhausted, the report is intentionally incomplete
    (``entries_dispatched < total_entries``) so the sidecar remains
    resumable for another bounded chunk.

    Each surviving entry has its underlying finding re-derived from
    the current source. If the finding no longer exists, the entry is
    classified ``ENTRY_OBSOLETE`` without calling the judge. Otherwise
    the judge runs and the response is classified against the entry's
    stored verdict.

    ``sidecar_writer`` (T6b) is an optional
    :class:`~elspeth_lints.core.reaudit_sidecar.SidecarWriter` already
    entered as a context manager by the caller. When supplied, every
    classified outcome is appended to the sidecar with fsync, so a
    process killed mid-sweep leaves a recoverable trail. The CLI driver
    always supplies one; tests and internal callers may omit it for
    pure in-memory invocations.

    ``pre_classified_keys`` + ``pre_classified_outcomes`` (T6b
    ``--resume`` support) carry forward the keys + outcomes already
    classified by an earlier killed sweep. Entries whose key appears in
    ``pre_classified_keys`` are skipped (already classified, already
    durable on the sidecar). The returned report's ``outcomes`` is the
    concatenation of the prior outcomes and this run's outcomes, in
    iteration order; ``entries_dispatched`` and ``total_entries`` are
    summed across both halves so the report reflects the unified sweep.
    """
    supported = _supported_rules()
    if reference_time is None:
        reference_time = datetime.now(UTC)
    elif reference_time.tzinfo is None:
        raise ReauditError("reference_time must be timezone-aware")
    if max_calls is not None and max_calls <= 0:
        raise ReauditError("--max-calls must be a positive integer when provided")
    if rule_filter not in supported:
        raise ReauditError(
            f"--rule {rule_filter!r} is not supported by reaudit. "
            f"Supported: {sorted(supported)}. "
            f"(Excluded rules: {sorted(_EXCLUDED_FROM_REAUDIT)}; see "
            "_EXCLUDED_FROM_REAUDIT in reaudit.py for the rationale "
            "behind each exclusion.)"
        )
    if not allowlist_dir.is_dir():
        raise ReauditError(f"--allowlist-dir {allowlist_dir} is not a directory")
    if not root.is_dir():
        raise ReauditError(f"--root {root} is not a directory")

    # ``valid_rule_ids`` gates ``per_file_rules`` validation inside
    # ``load_allowlist``. Reaudit only iterates ``allowlist.entries``,
    # but the directory's YAML files may carry sibling ``per_file_rules``
    # sections whose sub-rule ids must be recognised by the loader. The
    # dispatch returns the sub-rule vocabulary for the rule package
    # named by ``rule_filter``.
    valid_rule_ids = _valid_rule_ids_for(rule_filter)
    # Pass source_root so the C8-3 binding gate verifies file_fingerprint
    # against the live source. Reaudit re-judges entries against current
    # code anyway, so the load-time check just ensures the entries we're
    # about to re-judge weren't tampered with since the original verdict.
    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=valid_rule_ids, source_root=root)

    entries_skipped_by_rule_filter = _count_rule_filter_skips(allowlist.entries, valid_rule_ids)
    filtered = _apply_filters(
        entries=allowlist.entries,
        valid_rule_ids=valid_rule_ids,
        include_pre_judge=include_pre_judge,
        since=since,
        limit=limit,
        reference_time=reference_time,
    )

    # Resume support (T6b): the caller may have already classified some
    # prefix of ``filtered`` in a prior sweep that was killed. We strip
    # those entries from the iteration list and carry their outcomes
    # forward into the returned report. Pre-classified entries are NOT
    # rewritten to the sidecar — they're already durable on disk from
    # the original sweep.
    if pre_classified_keys:
        from elspeth_lints.core.reaudit_sidecar import filter_already_classified

        to_dispatch = filter_already_classified(filtered, pre_classified_keys)
    else:
        to_dispatch = list(filtered)

    outcomes: list[ReauditOutcome] = list(pre_classified_outcomes)
    # Cache scanned-file findings keyed by file_path so we don't re-run
    # the scanner for every entry on a file that has many entries
    # (web/composer/tools/* clusters dozens of entries per file).
    # ``rule_filter`` is closure-captured by ``_scan_findings_for_file``
    # via the cache; one ``rule_filter`` is active per ``reaudit_entries``
    # call so a single-keyed cache suffices.
    findings_cache: dict[str, list[Any]] = {}

    # Track dispatched-vs-planned so an aborted-mid-sweep run surfaces
    # the deficit on the report (closes elspeth-9a4e54cc01 / C3-3).
    # Incremented *before* the per-entry work runs so an uncaught
    # exception above the per-entry try/except still leaves
    # ``entries_dispatched`` equal to "how many we tried". On resume,
    # the count starts at the prior sweep's contribution so the unified
    # report's ratio reflects both halves of the work.
    total_entries = len(filtered)
    entries_dispatched = len(pre_classified_outcomes)
    judge_call_budget = _JudgeCallBudget(max_calls=max_calls)
    for entry in to_dispatch:
        entries_dispatched += 1
        try:
            outcome = _reaudit_one_entry(
                entry=entry,
                root=root,
                rule_filter=rule_filter,
                findings_cache=findings_cache,
                reference_time=reference_time,
                judge_call_budget=judge_call_budget,
            )
        except _JudgeCallBudgetExhausted:
            # This is a graceful operator-requested stop, not a
            # per-entry outcome. Leave the current entry unrecorded so
            # --resume reprocesses it from the beginning in the next
            # bounded invocation.
            entries_dispatched -= 1
            break
        outcomes.append(outcome)
        if sidecar_writer is not None:
            sidecar_writer.write_outcome(outcome)
        if progress_callback is not None:
            progress_callback(
                ReauditProgress.from_outcomes(
                    outcomes,
                    entries_dispatched=entries_dispatched,
                    total_entries=total_entries,
                    max_judge_calls=max_calls,
                )
            )

    return ReauditReport.from_outcomes(
        outcomes,
        entries_dispatched=entries_dispatched,
        total_entries=total_entries,
        entries_skipped_by_rule_filter=entries_skipped_by_rule_filter,
        max_judge_calls=max_calls,
    )


def _apply_filters(
    *,
    entries: Sequence[AllowlistEntry],
    valid_rule_ids: frozenset[str],
    include_pre_judge: bool,
    since: datetime | None,
    limit: int | None,
    reference_time: datetime | None = None,
) -> list[AllowlistEntry]:
    """Apply reaudit filters after deterministic oldest-judgment ordering."""
    if reference_time is None:
        reference_time = datetime.now(UTC)
    result: list[AllowlistEntry] = []
    for entry in sorted(entries, key=_entry_filter_order):
        if not _entry_matches_rule_filter(entry, valid_rule_ids):
            continue
        if entry.judge_verdict is None and not include_pre_judge:
            continue
        if since is not None and entry.judge_recorded_at is not None:
            if entry.judge_recorded_at > reference_time:
                # Future-dated entries are tampering/clock-skew signals.
                # They must survive --since so _reaudit_one_entry can emit
                # a structured FUTURE_DATED_ENTRY outcome instead of
                # disappearing from the sweep.
                pass
            elif entry.judge_recorded_at >= since:
                continue
        result.append(entry)
        if limit is not None and len(result) >= limit:
            break
    return result


def _entry_filter_order(entry: AllowlistEntry) -> tuple[int, float]:
    """Sort pre-judge first, then oldest recorded judgment first."""
    if entry.judge_recorded_at is None:
        return (0, float("-inf"))
    return (1, entry.judge_recorded_at.timestamp())


def _count_rule_filter_skips(entries: Sequence[AllowlistEntry], valid_rule_ids: frozenset[str]) -> int:
    """Count entries skipped by the rule-filter phase."""
    return sum(1 for entry in entries if not _entry_matches_rule_filter(entry, valid_rule_ids))


def _entry_matches_rule_filter(entry: AllowlistEntry, valid_rule_ids: frozenset[str]) -> bool:
    """Return True when ``entry`` belongs to the selected rule package.

    Malformed keys stay in the sweep so the per-entry classifier can surface
    them as ENTRY_OBSOLETE. Only well-formed keys with a rule id outside the
    selected package are skipped by the documented rule-filter phase.
    """
    parsed = _parse_entry_key(entry.key)
    if parsed is None:
        return True
    _file_path, rule_id, _symbol_context, _fingerprint = parsed
    return rule_id in valid_rule_ids


def _reaudit_one_entry(
    *,
    entry: AllowlistEntry,
    root: Path,
    rule_filter: str,
    findings_cache: dict[str, list[Any]],
    reference_time: datetime,
    judge_call_budget: _JudgeCallBudget,
) -> ReauditOutcome:
    """Classify one entry, calling the judge unless the entry is obsolete."""
    if entry.judge_recorded_at is not None and entry.judge_recorded_at > reference_time:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=(
                f"judge_recorded_at {entry.judge_recorded_at.isoformat()} is after "
                f"reaudit reference_time {reference_time.isoformat()}; future-dated judge "
                "timestamps indicate clock skew or deliberate tampering."
            ),
            fresh_recorded_at=None,
            divergence=ReauditDivergence.FUTURE_DATED_ENTRY,
            cause=ReauditCause.for_divergence(ReauditDivergence.FUTURE_DATED_ENTRY),
            code_snapshot="<future-dated judge_recorded_at; judge not called>",
        )

    parsed = _parse_entry_key(entry.key)
    if parsed is None:
        # Malformed key — should never happen for entries written by
        # this codebase, but the allowlist YAML is text we don't
        # control end-to-end. Treat as obsolete (no judge call) and
        # surface as a divergence for operator inspection.
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=None,
            fresh_recorded_at=None,
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            cause=ReauditCause.for_divergence(ReauditDivergence.ENTRY_OBSOLETE),
            code_snapshot="<entry key could not be parsed; finding cannot be re-located>",
        )

    file_path, rule_id, symbol_part, fingerprint = parsed

    # Path-containment gate (closes elspeth-ebb2b88753 / C3-4). The
    # ``file_path`` segment is attacker-controllable in the same sense
    # the YAML key is: any actor with write access to the allowlist
    # could forge a ``../../../etc/passwd`` key. ``resolve_safe_excerpt_path``
    # raises ``SourceExcerptPathOutsideRootError`` (subclass of
    # ``ValueError`` so the T6b ``RuntimeError`` net below does NOT
    # catch it) when the resolved path escapes root. We classify the
    # entry as SOURCE_EXCERPT_REJECTED and continue the sweep — a
    # single forged key must not abort 700 legitimate entries, but it
    # must not silently downgrade to JUDGE_CALL_FAILED ("rerun later")
    # either. The catch is scoped tightly to the path-resolution call
    # so a misconfigured ``--root`` (which would manifest as
    # FileNotFoundError on the root itself) still propagates fatally.
    from elspeth_lints.core.source_excerpt import (
        SourceExcerptPathOutsideRootError,
        extract_safe_excerpt,
        resolve_safe_excerpt_path,
    )

    candidate = root / file_path
    try:
        target_file = resolve_safe_excerpt_path(root=root, target_file=candidate)
    except FileNotFoundError:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=None,
            fresh_recorded_at=None,
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            cause=ReauditCause.for_divergence(ReauditDivergence.ENTRY_OBSOLETE),
            code_snapshot=f"<source file {file_path!r} no longer exists>",
        )
    except SourceExcerptPathOutsideRootError as exc:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=str(exc),
            fresh_recorded_at=None,
            divergence=ReauditDivergence.SOURCE_EXCERPT_REJECTED,
            cause=ReauditCause.for_divergence(ReauditDivergence.SOURCE_EXCERPT_REJECTED),
            code_snapshot=f"<source-excerpt path rejected for {entry.key!r}; see fresh_rationale>",
        )

    findings = _scan_findings_for_file(
        target_file=target_file,
        root=root,
        rule_filter=rule_filter,
        cache=findings_cache,
    )
    try:
        matching_finding = _find_matching_finding(findings=findings, entry_key=entry.key)
    except AmbiguousFindingMatchError as exc:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=str(exc),
            fresh_recorded_at=None,
            divergence=ReauditDivergence.AMBIGUOUS_FINDING_MATCH,
            cause=ReauditCause.for_divergence(ReauditDivergence.AMBIGUOUS_FINDING_MATCH),
            code_snapshot=f"<ambiguous current findings match {entry.key!r}; see fresh_rationale>",
        )

    if matching_finding is None:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=None,
            fresh_recorded_at=None,
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            cause=ReauditCause.for_divergence(ReauditDivergence.ENTRY_OBSOLETE),
            code_snapshot=f"<no current finding matches {entry.key!r}>",
        )

    # Secrets-scrubber gate (closes elspeth-ebb2b88753 / C2-2 on the
    # sweep path). ``extract_safe_excerpt`` re-runs containment (cheap)
    # AND scrubs inline secrets from the ±15-line window before the
    # judge call. The scrubbed text enters the prompt; the redactions
    # land on the outcome for the sidecar trail.
    safe_excerpt = extract_safe_excerpt(
        root=root,
        target_file=target_file,
        line=matching_finding.line,
        context_lines=15,
    )
    surrounding_code = safe_excerpt.text
    symbol_for_request = ".".join(symbol_part) if symbol_part else "_module_"
    request = JudgeRequest(
        file_path=file_path,
        rule_id=rule_id,
        symbol=symbol_for_request,
        fingerprint=fingerprint,
        rationale=entry.reason,
        surrounding_code=surrounding_code,
    )
    # Per-entry judge-boundary isolation (closes elspeth-9a4e54cc01 /
    # C3-2). The judge call is a Tier-3 external boundary: a transient
    # network failure or malformed model response on entry 423 of 700
    # must not discard the prior 422 outcomes. The catch covers two
    # failure classes:
    #
    # 1. ``JudgeTransportError`` — wraps SDK connection / timeout /
    #    rate-limit / 5xx errors with a stable project-local type.
    # 2. ``JudgeContractError`` raised from inside ``call_judge`` itself
    #    — these are emitted by ``_parse_judge_payload`` (malformed JSON
    #    from the model), ``_extract_text_block`` (unexpected response
    #    shape), and related response-contract checks. The model's output
    #    is itself a stochastic Tier-3 boundary; a single malformed
    #    response is the same class of failure as a 5xx and must not abort
    #    a 700-entry sweep. T6b extends T6 commit 6b33ee5b3 to close this
    #    gap.
    #
    # The catch is scoped to the ``call_judge`` invocation only:
    # ``_classify_divergence`` and friends emit their own ``RuntimeError``
    # subclasses on registry corruption and those MUST propagate (bugs
    # in our code, not transport failures). ``JudgeConfigurationError``
    # (missing API key / SDK) is also not caught — it's an operator
    # misconfiguration and remains sweep-fatal.
    #
    from elspeth_lints.core.judge import JudgeConfigurationError

    judge_call_budget.reserve_call()
    try:
        response = call_judge(request)
    except JudgeConfigurationError:
        # JudgeConfigurationError is NOT a transport failure — it's an
        # operator misconfiguration (missing API key, missing SDK). Keep
        # the sweep-fatal semantics the T6 commit established.
        raise
    except (JudgeTransportError, JudgeContractError) as exc:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=f"{type(exc).__name__}: {exc}",
            fresh_recorded_at=None,
            divergence=ReauditDivergence.JUDGE_CALL_FAILED,
            cause=ReauditCause.for_divergence(ReauditDivergence.JUDGE_CALL_FAILED),
            code_snapshot=surrounding_code,
            judge_call_attempted=True,
            excerpt_redactions=safe_excerpt.redactions,
        )
    try:
        divergence = _classify_divergence(
            entry_verdict=entry.judge_verdict,
            entry_model_verdict=entry.judge_model_verdict,
            fresh_verdict=response.verdict,
        )
    except ReauditError as exc:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=response.verdict,
            fresh_rationale=f"{type(exc).__name__}: {exc}",
            fresh_recorded_at=response.recorded_at,
            fresh_model_id=response.model_id,
            judge_call_attempted=True,
            fresh_prompt_tokens_total=response.prompt_tokens_total,
            fresh_prompt_tokens_cached=response.prompt_tokens_cached,
            divergence=ReauditDivergence.JUDGE_CLASSIFICATION_FAILED,
            cause=ReauditCause.for_divergence(ReauditDivergence.JUDGE_CLASSIFICATION_FAILED),
            code_snapshot=surrounding_code,
            excerpt_redactions=safe_excerpt.redactions,
        )
    return ReauditOutcome(
        entry=entry,
        original_verdict=entry.judge_verdict,
        original_model_verdict=entry.judge_model_verdict,
        fresh_verdict=response.verdict,
        fresh_rationale=response.judge_rationale,
        fresh_recorded_at=response.recorded_at,
        fresh_model_id=response.model_id,
        judge_call_attempted=True,
        fresh_prompt_tokens_total=response.prompt_tokens_total,
        fresh_prompt_tokens_cached=response.prompt_tokens_cached,
        divergence=divergence,
        cause=ReauditCause.for_divergence(divergence),
        code_snapshot=surrounding_code,
        excerpt_redactions=safe_excerpt.redactions,
    )


def _classify_divergence(
    *,
    entry_verdict: JudgeVerdict | None,
    entry_model_verdict: JudgeVerdict | None,
    fresh_verdict: JudgeVerdict,
) -> ReauditDivergence:
    """Map (stored verdict triple, fresh verdict) → divergence.

    The classification matrix is exhaustive over the legal combinations
    of stored verdicts; an unknown combination raises so a future
    change to the enum surface fails loudly.
    """
    if entry_verdict is None:
        if fresh_verdict is JudgeVerdict.ACCEPTED:
            return ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT
        if fresh_verdict is JudgeVerdict.BLOCKED:
            return ReauditDivergence.PRE_JUDGE_FRESH_BLOCK
        raise ReauditError(f"unexpected fresh verdict {fresh_verdict!r} for pre-judge entry")

    if entry_verdict is JudgeVerdict.ACCEPTED:
        if fresh_verdict is JudgeVerdict.ACCEPTED:
            return ReauditDivergence.STILL_AGREES
        if fresh_verdict is JudgeVerdict.BLOCKED:
            return ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED
        raise ReauditError(f"unexpected fresh verdict {fresh_verdict!r} after stored ACCEPTED")

    if entry_verdict is JudgeVerdict.BLOCKED:
        # Defense-in-depth: the loader rejects BLOCKED on persisted
        # entries, so reaudit should never observe one. If it does, the
        # loader was bypassed — crash rather than guess.
        raise ReauditError(
            "entry_verdict is BLOCKED; BLOCKED is an in-memory runtime verdict and "
            "must never appear on a persisted entry. The allowlist loader should have "
            "rejected this on load — registry corruption or a loader bypass."
        )

    if entry_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
        # The override sits on top of an underlying model verdict
        # (judge_model_verdict). Classify against that underlying
        # verdict so the divergence captures "is the override still
        # load-bearing?".
        if entry_model_verdict is JudgeVerdict.BLOCKED:
            if fresh_verdict is JudgeVerdict.ACCEPTED:
                return ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED
            if fresh_verdict is JudgeVerdict.BLOCKED:
                return ReauditDivergence.OVERRIDE_STILL_NEEDED
        if entry_model_verdict is JudgeVerdict.ACCEPTED:
            # Operator override of an ACCEPTED entry behaves
            # divergence-wise like a plain ACCEPTED — there was no
            # original BLOCK to lift. Operator probably overrode to
            # change the audit signal (e.g. record manual review) but
            # the suppression's legitimacy is the same question.
            if fresh_verdict is JudgeVerdict.ACCEPTED:
                return ReauditDivergence.STILL_AGREES
            if fresh_verdict is JudgeVerdict.BLOCKED:
                return ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED
        # judge_model_verdict is None on overrides is a schema
        # inconsistency the loader should already have rejected; if
        # it slips through, crash rather than guess.
        raise ReauditError(f"override entry has unexpected judge_model_verdict={entry_model_verdict!r}; expected ACCEPTED or BLOCKED")

    raise ReauditError(f"unknown entry_verdict={entry_verdict!r}")


# =========================================================================
# Entry-key parsing + finding lookup
# =========================================================================


def _parse_entry_key(key: str) -> tuple[str, str, tuple[str, ...], str] | None:
    """Split an entry key into (file_path, rule_id, symbol_context, fingerprint).

    Format (defined by ``FindingKey.canonical_key``):
        ``{file_path}:{rule_id}:{symbol_part}:fp={fingerprint}``

    where ``symbol_part`` is either ``_module_`` (sentinel for no
    enclosing symbol) or colon-joined dotted parts (e.g.
    ``Class:method``). File paths use ``/`` and contain no ``:``;
    rule_ids are short tokens (R1, R2, TC, ...) with no ``:``.

    Returns ``None`` when the key is malformed (no ``:fp=`` suffix or
    fewer than three colon-separated segments). Malformed keys become
    ``ENTRY_OBSOLETE`` divergences upstream — we don't crash because
    YAML is a partly-external trust boundary and a stray bad entry
    shouldn't kill the whole sweep.
    """
    prefix, sep, fingerprint = key.rpartition(":fp=")
    if not sep or not fingerprint:
        return None
    parts = prefix.split(":")
    if len(parts) < 3:
        return None
    file_path = parts[0]
    rule_id = parts[1]
    symbol_parts = tuple(parts[2:])
    if symbol_parts == ("_module_",):
        symbol_context: tuple[str, ...] = ()
    else:
        symbol_context = symbol_parts
    return file_path, rule_id, symbol_context, fingerprint


def _scan_findings_for_file(
    *,
    target_file: Path,
    root: Path,
    rule_filter: str,
    cache: dict[str, list[Any]],
) -> list[Any]:
    """Re-run the scanner for ``rule_filter`` against ``target_file``.

    Cached by ``str(target_file)`` so a directory with many entries on
    one file scans that file once. The cache is owned by a single
    ``reaudit_entries`` invocation, which carries one ``rule_filter``
    end-to-end; one rule per cache means the cache key is the file
    path alone.

    Two scanner shapes exist. ``trust_tier.tier_model`` has bespoke
    ``scan_file`` + ``scan_layer_imports_file`` entry points (layer
    imports cross file boundaries in a way ``Rule.analyze`` does not
    model; the merge mirrors ``cli._scan_single_file_findings`` so
    reaudit sees the same finding set the CI run would see). Every
    other supported rule uses the standard ``Rule.analyze`` protocol;
    ``_scan_via_rule_analyze`` dispatches to the matching
    ``BUILTIN_RULES`` entry.
    """
    cache_key = str(target_file)
    if cache_key in cache:
        return cache[cache_key]

    if rule_filter == "trust_tier.tier_model":
        findings = _scan_tier_model(target_file=target_file, root=root)
    else:
        findings = _scan_via_rule_analyze(
            rule_filter=rule_filter,
            target_file=target_file,
            root=root,
        )

    cache[cache_key] = findings
    return findings


def _scan_tier_model(*, target_file: Path, root: Path) -> list[Any]:
    """Run tier_model's bespoke scanners (R1-R7, TC, L1) against ``target_file``.

    Mirrors ``cli._scan_single_file_findings`` so the reaudit finding
    set matches the CI finding set on the same file. Lazy import:
    tier_model is heavy and importing it at module scope would slow
    every ``elspeth-lints --help`` invocation.
    """
    from elspeth_lints.rules.trust_tier.tier_model.rule import (
        scan_file,
        scan_layer_imports_file,
    )

    findings: list[Any] = list(scan_file(target_file, root))
    layer_violations, layer_tc = scan_layer_imports_file(target_file, root)
    findings.extend(layer_violations)
    findings.extend(layer_tc)
    return findings


def _scan_via_rule_analyze(
    *,
    rule_filter: str,
    target_file: Path,
    root: Path,
) -> list[Any]:
    """Run the ``BUILTIN_RULES`` entry whose ``.id == rule_filter`` against ``target_file``.

    The generic path for every supported rule except tier_model. The
    standard ``Rule.analyze(tree, file_path, context)`` protocol returns
    findings with a ``canonical_key`` shape that matches the entry key
    format reaudit's ``_find_matching_finding`` compares against.

    Syntax errors in the source surface as zero findings (the matching
    finding will be ``None``, classified ``ENTRY_OBSOLETE``). This is
    consistent with the CI run's behaviour: a file that doesn't parse
    can't be analysed; we don't pretend its prior findings are still
    valid.
    """
    from elspeth_lints.core.ast_walker import (
        ParsedPythonFile,
        PythonFileReadError,
        PythonSyntaxError,
        parse_python_file,
    )
    from elspeth_lints.core.protocols import RuleContext
    from elspeth_lints.rules import BUILTIN_RULES

    rule = next((r for r in BUILTIN_RULES if r.id == rule_filter), None)
    if rule is None:
        # Should never happen: ``rule_filter`` was already validated
        # against ``_supported_rules()`` (which derives from
        # ``BUILTIN_RULES``). A mismatch here means the registry was
        # mutated mid-run or ``_EXCLUDED_FROM_REAUDIT`` is stale.
        raise ReauditError(
            f"Rule {rule_filter!r} passed _supported_rules() but is not registered in BUILTIN_RULES. Registry drift detected."
        )

    parsed = parse_python_file(target_file)
    if isinstance(parsed, PythonSyntaxError):
        return []
    if isinstance(parsed, PythonFileReadError):
        # Reaudit replays prior findings on a single file. If that file
        # is unreadable now (UTF-8 corruption, permission change, file
        # vanished), we cannot pretend the prior findings are still
        # valid — return empty, matching the syntax-error policy.
        return []
    if not isinstance(parsed, ParsedPythonFile):  # narrow for type checker
        raise ReauditError(f"parse_python_file returned unexpected type {type(parsed).__name__}")

    context = RuleContext(root=root)
    return list(rule.analyze(parsed.tree, parsed.path, context))


def _find_matching_finding(*, findings: Sequence[Any], entry_key: str) -> Any | None:
    """Return the finding whose ``canonical_key`` equals ``entry_key``.

    Matching on the full canonical key (rather than reconstructing it
    from parts) sidesteps any subtle drift in the formatter — if the
    rule's key shape changes, this comparison still works.
    """
    matches = [finding for finding in findings if _canonical_key_for_finding(finding) == entry_key]
    if len(matches) > 1:
        raise AmbiguousFindingMatchError(f"{len(matches)} finding(s) match allowlist key {entry_key!r}; refusing to pick one silently.")
    return matches[0] if matches else None


def _canonical_key_for_finding(finding: Any) -> str:
    """Return a finding's canonical key across both finding contract shapes."""
    canonical_key = finding.canonical_key
    if callable(canonical_key):
        canonical_key = canonical_key()
    if not isinstance(canonical_key, str):
        raise ReauditError(
            f"finding.canonical_key must be a string or zero-argument callable returning a string; got {type(canonical_key).__name__}"
        )
    return canonical_key


def _valid_rule_ids_for(rule_filter: str) -> frozenset[str]:
    """Return the sub-rule vocabulary for ``rule_filter``.

    ``load_allowlist`` validates the ``rules:`` lists inside
    ``per_file_rules`` entries against this set. Reaudit itself only
    iterates ``allowlist.entries`` (not ``per_file_rules``), but the
    allowlist directory's YAML files commonly carry both shapes
    side-by-side, and the loader fails on unknown sub-rule ids.

    The registry reads each rule's emission-vocabulary constant lazily
    via :func:`_lookup_module_attr` so importing one rule's vocabulary
    does not load every rule package. Constants vary in shape:

    * ``RULES`` — mapping whose keys are sub-rule ids.
    * ``RULE_ID`` — package id emitted directly as a finding id.
    * ``LEGACY_RULE_ID`` — finding id emitted by rules ported from
      earlier bespoke scripts.
    * ``_ALLOWLIST_RULE_IDS`` — scanner-owned collection of exact
      sub-rule ids accepted by a sibling allowlist loader.

    Adding a new rule: add one ``_RULE_VOCABULARY_REGISTRY`` entry. The
    tests assert every supported rule resolves a non-empty vocabulary,
    so a supported rule with no registry row fails at collection time
    instead of silently skipping production entries.
    """
    spec = _RULE_VOCABULARY_REGISTRY.get(rule_filter)
    if spec is None:
        raise ReauditError(
            f"No sub-rule vocabulary is registered for rule {rule_filter!r}. "
            "Add an entry to _RULE_VOCABULARY_REGISTRY so load_allowlist can "
            "validate per_file_rules entries that reference this rule's "
            "sub-rule ids."
        )

    if spec.use_mapping_keys and spec.use_collection_values:
        raise ReauditError(f"{rule_filter}: vocabulary specs cannot use both mapping keys and collection values")

    if spec.use_mapping_keys:
        if len(spec.attr_names) != 1:
            raise ReauditError(f"{rule_filter}: mapping-key vocabulary specs must name exactly one attribute")
        rules_mapping = _lookup_module_attr(spec.module_path, spec.attr_names[0])
        if not isinstance(rules_mapping, Mapping):
            raise ReauditError(
                f"{rule_filter}: {spec.module_path}.{spec.attr_names[0]} must be a mapping, got {type(rules_mapping).__name__}"
            )
        return frozenset(str(rule_id) for rule_id in rules_mapping)

    if spec.use_collection_values:
        if len(spec.attr_names) != 1:
            raise ReauditError(f"{rule_filter}: collection-value vocabulary specs must name exactly one attribute")
        values_collection = _lookup_module_attr(spec.module_path, spec.attr_names[0])
        if isinstance(values_collection, str) or not isinstance(values_collection, Collection):
            raise ReauditError(
                f"{rule_filter}: {spec.module_path}.{spec.attr_names[0]} must be a non-string collection, "
                f"got {type(values_collection).__name__}"
            )
        if any(not isinstance(value, str) for value in values_collection):
            bad = [type(value).__name__ for value in values_collection if not isinstance(value, str)]
            raise ReauditError(f"{rule_filter}: collection vocabulary values must be strings, got {bad}")
        return frozenset(values_collection)

    values = tuple(_lookup_module_attr(spec.module_path, attr_name) for attr_name in spec.attr_names)
    if any(not isinstance(value, str) for value in values):
        bad = [type(value).__name__ for value in values if not isinstance(value, str)]
        raise ReauditError(f"{rule_filter}: vocabulary constants must be strings, got {bad}")
    return frozenset(values)


def _lookup_module_attr(module_path: str, attr_name: str) -> Any:
    """Import ``module_path`` and return ``getattr(module, attr_name)``.

    Offensive lookup: a missing attribute raises ``AttributeError``
    naming both the module and the constant, surfacing "you renamed
    the rule's vocabulary constant" as a clear diagnostic. The
    ``getattr`` here is not defensive (no default supplied) — it is
    a dynamic lookup of a constant whose name is known but whose
    static import would force mypy to inspect each rule package's
    ``__all__``, requiring per-rule edits that are out of scope for
    this gate.
    """
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


# =========================================================================
# Rendering
# =========================================================================


def render_report_text(report: ReauditReport) -> str:
    """One line per outcome + a summary block.

    Format: ``{file}:{rule}:{symbol}  {divergence}  fresh={verdict}``.
    Stable column ordering so diffs across runs are easy to read.

    When the sweep is incomplete (``entries_dispatched <
    total_entries``), an "INCOMPLETE SWEEP" banner is prepended so the
    operator cannot miss that some entries were never reaudited at all
    (distinct from JUDGE_CALL_FAILED, which records entries the sweep
    *attempted* but the transport rejected). Closes
    elspeth-9a4e54cc01 / C3-3.
    """
    lines: list[str] = []
    banner = _incomplete_sweep_banner(report)
    if banner is not None:
        lines.append(banner)
        lines.append("")
    for outcome in report.outcomes:
        symbol = outcome.entry.key
        fresh = outcome.fresh_verdict.value if outcome.fresh_verdict is not None else "<no judge call>"
        fresh_model = outcome.fresh_model_id if outcome.fresh_model_id is not None else "<no judge call>"
        lines.append(f"{symbol}  {outcome.divergence.value}  fresh={fresh}  model={fresh_model}  cause={outcome.cause.value}")
    lines.append("")
    lines.append("Summary:")
    for name, count in report.summary:
        lines.append(f"  {name:<32} {count}")
    lines.append(f"  {'entries skipped by rule_filter':<32} {report.entries_skipped_by_rule_filter}")
    lines.append(f"  {'judge calls attempted':<32} {_format_judge_call_count(report.judge_calls_attempted, report.max_judge_calls)}")
    lines.append(
        f"  {'prompt tokens':<32} "
        f"total={report.prompt_tokens_total} "
        f"cached={_format_optional_int(report.prompt_tokens_cached)} "
        f"uncached={_format_optional_int(report.prompt_tokens_uncached)}"
    )
    return "\n".join(lines) + "\n"


def _incomplete_sweep_banner(report: ReauditReport) -> str | None:
    """Return the textual "INCOMPLETE SWEEP" banner, or ``None`` if complete.

    A sweep is "complete" when every planned dispatch produced an
    outcome — ``entries_dispatched == total_entries``. JUDGE_CALL_FAILED
    outcomes still count as completed dispatches (the sweep reached the
    entry, attempted the judge call, and recorded the transport failure
    as a divergence). The banner fires only for entries the sweep never
    reached — process killed, machine rebooted, uncaught exception in
    the per-entry orchestration above the call-judge boundary.
    """
    missing = report.total_entries - report.entries_dispatched
    if missing <= 0:
        return None
    return (
        f"!! INCOMPLETE SWEEP: {missing} entry/entries of {report.total_entries} "
        f"were never reaudited (dispatched {report.entries_dispatched}). "
        "The sweep aborted before reaching them. Rerun reaudit to cover them."
    )


def _format_judge_call_count(attempted: int, max_calls: int | None) -> str:
    if max_calls is None:
        return str(attempted)
    return f"{attempted}/{max_calls}"


def _format_optional_int(value: int | None) -> str:
    return str(value) if value is not None else "n/a"


def render_report_json(report: ReauditReport) -> str:
    """Full report as JSON.

    Uses ``dataclasses.asdict`` then walks the resulting tree to
    encode enums as their string values and datetimes as ISO-8601
    strings. The ``AllowlistEntry`` field becomes a nested object.
    The ``matched`` runtime-only field is deliberately omitted: it is
    loader/scanner state, not part of the report contract.
    """
    import json

    payload: dict[str, Any] = {
        "outcomes": [
            {
                "entry": _entry_to_json(outcome.entry),
                "original_verdict": _verdict_value(outcome.original_verdict),
                "original_model_verdict": _verdict_value(outcome.original_model_verdict),
                "fresh_verdict": _verdict_value(outcome.fresh_verdict),
                "fresh_model_id": outcome.fresh_model_id,
                "fresh_rationale": outcome.fresh_rationale,
                "fresh_recorded_at": (outcome.fresh_recorded_at.isoformat() if outcome.fresh_recorded_at is not None else None),
                "judge_call_attempted": outcome.judge_call_attempted,
                "fresh_prompt_tokens_total": outcome.fresh_prompt_tokens_total,
                "fresh_prompt_tokens_cached": outcome.fresh_prompt_tokens_cached,
                "divergence": outcome.divergence.value,
                "cause": outcome.cause.value,
                "code_snapshot": outcome.code_snapshot,
                "excerpt_redactions": [
                    {
                        "pattern_name": r.pattern_name,
                        "byte_count": r.byte_count,
                        "redacted_hash": r.redacted_hash,
                    }
                    for r in outcome.excerpt_redactions
                ],
            }
            for outcome in report.outcomes
        ],
        "summary": [{"divergence": name, "count": count} for name, count in report.summary],
        # Dispatch accounting — always present so consumers don't have
        # to defensively probe; the fields are the machine-readable
        # counterpart to the "INCOMPLETE SWEEP" banner the text /
        # markdown renderers emit.
        "entries_dispatched": report.entries_dispatched,
        "total_entries": report.total_entries,
        "entries_skipped_by_rule_filter": report.entries_skipped_by_rule_filter,
        "cost_telemetry": {
            "judge_calls_attempted": report.judge_calls_attempted,
            "max_judge_calls": report.max_judge_calls,
            "prompt_tokens_total": report.prompt_tokens_total,
            "prompt_tokens_cached": report.prompt_tokens_cached,
            "prompt_tokens_uncached": report.prompt_tokens_uncached,
        },
    }
    if report.entries_dispatched < report.total_entries:
        payload["incomplete_sweep"] = {
            "missing": report.total_entries - report.entries_dispatched,
            "message": _incomplete_sweep_banner(report),
        }
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def _entry_to_json(entry: AllowlistEntry) -> dict[str, Any]:
    """Convert one AllowlistEntry to a JSON-safe dict."""
    raw = dataclasses.asdict(entry)
    raw.pop("matched", None)
    raw["judge_verdict"] = _verdict_value(entry.judge_verdict)
    raw["judge_model_verdict"] = _verdict_value(entry.judge_model_verdict)
    raw["expires"] = entry.expires.isoformat() if entry.expires is not None else None
    raw["judge_recorded_at"] = entry.judge_recorded_at.isoformat() if entry.judge_recorded_at is not None else None
    return raw


def _verdict_value(verdict: JudgeVerdict | None) -> str | None:
    return verdict.value if verdict is not None else None


def render_report_markdown(report: ReauditReport) -> str:
    """Markdown report grouped by divergence severity.

    Layout:

    * ``# Reaudit report`` header with summary counts as a table.
    * One section per divergence kind, in severity order. Each section
      has a table with columns Entry, Original Verdict, Fresh Verdict,
      Cause, and Notes. The "Entry" column is the canonical key (file:rule:symbol
      :fp=...) — the most useful column for operator triage because it
      grep-matches against the YAML on disk.

    The markdown is the most operator-readable surface; pasted into a
    PR or wiki, it survives without re-running the tool.
    """
    lines: list[str] = []
    lines.append("# Reaudit report")
    lines.append("")
    banner = _incomplete_sweep_banner(report)
    if banner is not None:
        # Use a markdown blockquote so the banner stands out visually
        # and survives plain-text rendering in PR comments / wiki
        # paste-throughs.
        lines.append(f"> **{banner}**")
        lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Divergence | Count |")
    lines.append("| --- | --- |")
    for name, count in report.summary:
        lines.append(f"| {name} | {count} |")
    lines.append(f"| _entries dispatched_ | {report.entries_dispatched} / {report.total_entries} |")
    lines.append(f"| _entries skipped by rule_filter_ | {report.entries_skipped_by_rule_filter} |")
    lines.append(f"| _judge calls attempted_ | {_format_judge_call_count(report.judge_calls_attempted, report.max_judge_calls)} |")
    lines.append(
        f"| _prompt tokens_ | total={report.prompt_tokens_total}; "
        f"cached={_format_optional_int(report.prompt_tokens_cached)}; "
        f"uncached={_format_optional_int(report.prompt_tokens_uncached)} |"
    )
    lines.append("")

    grouped: dict[ReauditDivergence, list[ReauditOutcome]] = {member: [] for member in ReauditDivergence}
    for outcome in report.outcomes:
        grouped[outcome.divergence].append(outcome)

    for divergence in sorted(ReauditDivergence, key=lambda d: _DIVERGENCE_ORDER[d]):
        bucket = grouped[divergence]
        if not bucket:
            continue
        lines.append(f"## {divergence.value} ({len(bucket)})")
        lines.append("")
        lines.append("| Entry | Original Verdict | Fresh Verdict | Fresh Model | Cause | Notes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for outcome in bucket:
            entry_key = _md_escape(outcome.entry.key)
            original = _format_original_verdict(outcome)
            fresh = outcome.fresh_verdict.value if outcome.fresh_verdict is not None else "<no judge call>"
            fresh_model = _md_escape(outcome.fresh_model_id if outcome.fresh_model_id is not None else "<no judge call>")
            cause = outcome.cause.value
            notes = _md_escape(_outcome_notes(outcome))
            lines.append(f"| `{entry_key}` | {original} | {fresh} | {fresh_model} | {cause} | {notes} |")
        lines.append("")

    return "\n".join(lines)


def _format_original_verdict(outcome: ReauditOutcome) -> str:
    """Render the entry's stored verdict for the markdown Original column.

    For overrides we surface both the entry verdict and the underlying
    model verdict because the divergence semantics depend on the model
    verdict, not the override flag itself.
    """
    if outcome.original_verdict is None:
        return "<pre-judge>"
    if outcome.original_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
        underlying = outcome.original_model_verdict.value if outcome.original_model_verdict is not None else "?"
        return f"OVERRIDDEN_BY_OPERATOR (model: {underlying})"
    return outcome.original_verdict.value


def _outcome_notes(outcome: ReauditOutcome) -> str:
    """Short operator-facing note for the rightmost markdown column.

    For divergence classes that need operator attention, the note
    summarises the recommended action. For STILL_AGREES we include the
    fresh rationale's first sentence so the column carries some signal
    even on no-change rows.
    """
    if outcome.divergence is ReauditDivergence.ENTRY_OBSOLETE:
        # ENTRY_OBSOLETE outcomes carry their explanation in
        # code_snapshot when there's no live finding; surface it here
        # so the operator can act without opening the JSON dump.
        return outcome.code_snapshot
    if outcome.divergence is ReauditDivergence.JUDGE_CALL_FAILED:
        # JUDGE_CALL_FAILED outcomes carry the exception classname +
        # message in fresh_rationale; surface the full text (not just
        # the first line) so the operator can diagnose without opening
        # the JSON dump.
        return outcome.fresh_rationale if outcome.fresh_rationale is not None else "<no diagnostic captured>"
    if outcome.divergence is ReauditDivergence.JUDGE_CLASSIFICATION_FAILED:
        return outcome.fresh_rationale if outcome.fresh_rationale is not None else "<no diagnostic captured>"
    if outcome.divergence is ReauditDivergence.FUTURE_DATED_ENTRY:
        return outcome.fresh_rationale if outcome.fresh_rationale is not None else "<no diagnostic captured>"
    if outcome.divergence is ReauditDivergence.AMBIGUOUS_FINDING_MATCH:
        return outcome.fresh_rationale if outcome.fresh_rationale is not None else "<no diagnostic captured>"
    if outcome.divergence is ReauditDivergence.SOURCE_EXCERPT_REJECTED:
        # SOURCE_EXCERPT_REJECTED carries the path-containment error
        # message in fresh_rationale. Surface it verbatim so the
        # operator sees the offending path without opening the JSON
        # dump. Triage signal: investigate the YAML for tampering.
        return outcome.fresh_rationale if outcome.fresh_rationale is not None else "<no diagnostic captured>"
    if outcome.fresh_rationale is None:
        return ""
    return outcome.fresh_rationale.splitlines()[0] if outcome.fresh_rationale else ""


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _md_escape(text: str) -> str:
    """Escape model/operator text for markdown table cells.

    The report is commonly pasted into PR comments and wiki pages. Treat
    notes as untrusted text: strip terminal controls, HTML-escape tag
    delimiters, and backslash-escape markdown link/code/table characters.
    """
    escaped = _ANSI_ESCAPE_RE.sub("", text)
    escaped = escaped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    escaped = escaped.replace("\\", "\\\\")
    for char in ("`", "|", "[", "]", "(", ")"):
        escaped = escaped.replace(char, f"\\{char}")
    return escaped.replace("\n", " ")
