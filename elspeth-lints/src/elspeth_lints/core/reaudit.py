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
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    JudgeVerdict,
    load_allowlist,
)
from elspeth_lints.core.judge import JudgeRequest, call_judge


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

    ``JUDGE_CALL_FAILED`` is the transport-failure path: the judge call
    raised an SDK-level transport error (network, timeout, rate-limit,
    5xx) and the entry could not be re-judged on this sweep. The entry
    is *not* obsolete and its prior verdict is *not* refreshed — the
    operator must rerun once the transport problem is resolved. The
    exception classname + message is captured in ``fresh_rationale``
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


# Operator-actionable severity ranking. Lower values surface first in
# the markdown report so the most urgent debt is at the top.
#
# JUDGE_CALL_FAILED ranks first: the entry was *not* re-judged. Until
# the operator resolves the transport failure (network, rate-limit, 5xx)
# and reruns, the entry's decay status is unknown — that ignorance is
# more urgent than any verdict-change signal below it. Closes
# elspeth-9a4e54cc01 / C3-2.
_DIVERGENCE_ORDER: dict[ReauditDivergence, int] = {
    ReauditDivergence.JUDGE_CALL_FAILED: 0,
    ReauditDivergence.WAS_ACCEPTED_NOW_BLOCKED: 1,
    ReauditDivergence.PRE_JUDGE_FRESH_BLOCK: 2,
    ReauditDivergence.OVERRIDE_NO_LONGER_NEEDED: 3,
    ReauditDivergence.ENTRY_OBSOLETE: 4,
    ReauditDivergence.OVERRIDE_STILL_NEEDED: 5,
    ReauditDivergence.PRE_JUDGE_FRESH_ACCEPT: 6,
    ReauditDivergence.STILL_AGREES: 7,
}


@dataclass(frozen=True, slots=True)
class ReauditOutcome:
    """One entry's reaudit result.

    ``fresh_verdict`` / ``fresh_recorded_at`` are ``None`` for
    ``ENTRY_OBSOLETE`` (no judge call was made) and for
    ``JUDGE_CALL_FAILED`` (judge call was attempted but raised a
    transport error). ``fresh_rationale`` is ``None`` for
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
    code_snapshot: str


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

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[ReauditOutcome],
        *,
        entries_dispatched: int | None = None,
        total_entries: int | None = None,
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
        return cls(
            outcomes=tuple(outcomes),
            summary=tuple(ordered),
            entries_dispatched=dispatched,
            total_entries=total,
        )


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


def reaudit_entries(
    *,
    root: Path,
    allowlist_dir: Path,
    rule_filter: str,
    since: datetime | None,
    limit: int | None,
    include_pre_judge: bool,
    sidecar_writer: Any = None,
    pre_classified_keys: frozenset[str] | None = None,
    pre_classified_outcomes: Sequence[ReauditOutcome] = (),
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

    filtered = _apply_filters(
        entries=allowlist.entries,
        include_pre_judge=include_pre_judge,
        since=since,
        limit=limit,
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
    for entry in to_dispatch:
        entries_dispatched += 1
        outcome = _reaudit_one_entry(
            entry=entry,
            root=root,
            rule_filter=rule_filter,
            findings_cache=findings_cache,
        )
        outcomes.append(outcome)
        if sidecar_writer is not None:
            sidecar_writer.write_outcome(outcome)

    return ReauditReport.from_outcomes(
        outcomes,
        entries_dispatched=entries_dispatched,
        total_entries=total_entries,
    )


def _apply_filters(
    *,
    entries: Sequence[AllowlistEntry],
    include_pre_judge: bool,
    since: datetime | None,
    limit: int | None,
) -> list[AllowlistEntry]:
    """Apply the reaudit filters in documented order, preserving entry order."""
    result: list[AllowlistEntry] = []
    for entry in entries:
        if entry.judge_verdict is None and not include_pre_judge:
            continue
        if since is not None and entry.judge_recorded_at is not None and entry.judge_recorded_at >= since:
            continue
        result.append(entry)
        if limit is not None and len(result) >= limit:
            break
    return result


def _reaudit_one_entry(
    *,
    entry: AllowlistEntry,
    root: Path,
    rule_filter: str,
    findings_cache: dict[str, list[Any]],
) -> ReauditOutcome:
    """Classify one entry, calling the judge unless the entry is obsolete."""
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
            code_snapshot="<entry key could not be parsed; finding cannot be re-located>",
        )

    file_path, rule_id, symbol_part, fingerprint = parsed
    target_file = (root / file_path).resolve()
    if not target_file.exists():
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=None,
            fresh_recorded_at=None,
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            code_snapshot=f"<source file {file_path!r} no longer exists>",
        )

    findings = _scan_findings_for_file(
        target_file=target_file,
        root=root,
        rule_filter=rule_filter,
        cache=findings_cache,
    )
    matching_finding = _find_matching_finding(findings=findings, entry_key=entry.key)

    if matching_finding is None:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=None,
            fresh_recorded_at=None,
            divergence=ReauditDivergence.ENTRY_OBSOLETE,
            code_snapshot=f"<no current finding matches {entry.key!r}>",
        )

    surrounding_code = _extract_surrounding_code(target_file, matching_finding.line, context_lines=15)
    symbol_for_request = ".".join(symbol_part) if symbol_part else "_module_"
    request = JudgeRequest(
        file_path=file_path,
        rule_id=rule_id,
        symbol=symbol_for_request,
        fingerprint=fingerprint,
        rationale=entry.reason,
        surrounding_code=surrounding_code,
    )
    # Per-entry transport-failure isolation (closes elspeth-9a4e54cc01 /
    # C3-2). The judge call is a Tier-3 external boundary: a transient
    # network failure on entry 423 of 700 must not discard the prior
    # 422 outcomes. The catch covers two failure classes:
    #
    # 1. ``openai.APIError`` — the SDK umbrella for connection / timeout /
    #    rate-limit / 5xx. Genuine transport tier.
    # 2. ``RuntimeError`` raised from inside ``call_judge`` itself —
    #    today these are emitted by ``_parse_judge_payload`` (malformed
    #    JSON from the model) and ``_extract_text_block`` (unexpected
    #    response shape). The model's output is itself a stochastic
    #    Tier-3 boundary; a single malformed response is the same class
    #    of failure as a 5xx and must not abort a 700-entry sweep. T6b
    #    extends T6 commit 6b33ee5b3 to close this gap.
    #
    # The catch is scoped to the ``call_judge`` invocation only:
    # ``_classify_divergence`` and friends emit their own ``RuntimeError``
    # subclasses on registry corruption and those MUST propagate (bugs
    # in our code, not transport failures). ``JudgeConfigurationError``
    # (missing API key / SDK) is also not caught — it's an operator
    # misconfiguration and remains sweep-fatal.
    #
    # Lazy import so ``elspeth-lints --help`` doesn't pay the SDK cost
    # on every invocation (mirrors the pattern in ``call_judge`` itself).
    from openai import APIError

    from elspeth_lints.core.judge import JudgeConfigurationError

    try:
        response = call_judge(request)
    except JudgeConfigurationError:
        # JudgeConfigurationError subclasses RuntimeError but is NOT a
        # transport failure — it's an operator misconfiguration (missing
        # API key, missing SDK). Re-raise before the broader RuntimeError
        # branch can swallow it, preserving the sweep-fatal semantics
        # the T6 commit established.
        raise
    except (APIError, RuntimeError) as exc:
        return ReauditOutcome(
            entry=entry,
            original_verdict=entry.judge_verdict,
            original_model_verdict=entry.judge_model_verdict,
            fresh_verdict=None,
            fresh_rationale=f"{type(exc).__name__}: {exc}",
            fresh_recorded_at=None,
            divergence=ReauditDivergence.JUDGE_CALL_FAILED,
            code_snapshot=surrounding_code,
        )
    divergence = _classify_divergence(
        entry_verdict=entry.judge_verdict,
        entry_model_verdict=entry.judge_model_verdict,
        fresh_verdict=response.verdict,
    )
    return ReauditOutcome(
        entry=entry,
        original_verdict=entry.judge_verdict,
        original_model_verdict=entry.judge_model_verdict,
        fresh_verdict=response.verdict,
        fresh_rationale=response.judge_rationale,
        fresh_recorded_at=response.recorded_at,
        divergence=divergence,
        code_snapshot=surrounding_code,
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
    for finding in findings:
        if finding.canonical_key == entry_key:
            return finding
    return None


def _extract_surrounding_code(target_file: Path, line: int, *, context_lines: int) -> str:
    """Return ~30 lines of code centered on ``line``.

    Mirrors ``cli._extract_surrounding_code`` so the judge sees an
    identically-shaped excerpt on reaudit as on the original write.
    """
    text = target_file.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = max(1, line - context_lines)
    end = min(len(lines), line + context_lines)
    out: list[str] = []
    for line_num in range(start, end + 1):
        marker = ">>" if line_num == line else "  "
        out.append(f"{marker} {line_num:5d}  {lines[line_num - 1]}")
    return "\n".join(out)


def _valid_rule_ids_for(rule_filter: str) -> frozenset[str]:
    """Return the sub-rule vocabulary for ``rule_filter``.

    ``load_allowlist`` validates the ``rules:`` lists inside
    ``per_file_rules`` entries against this set. Reaudit itself only
    iterates ``allowlist.entries`` (not ``per_file_rules``), but the
    allowlist directory's YAML files commonly carry both shapes
    side-by-side, and the loader fails on unknown sub-rule ids.

    Each branch reads the rule's emission-vocabulary constant lazily
    via :func:`_lookup_module_attr` (so the import cost of one rule
    doesn't leak into reaudit invocations targeting a different
    rule). Constants vary in shape:

    * ``RULES`` — a ``dict[str, ...]`` whose keys are the sub-rule ids
      (tier_model, freeze_guards, frozen_annotations pattern).
    * ``RULE_ID`` — a single ``str`` (single-emission rules).
    * ``LEGACY_RULE_ID`` — single-emission rules ported from the
      pre-elspeth-lints CI scripts; the shared.py modules expose them.

    The ``getattr``-based lookup sidesteps strict-``__all__``
    enforcement in static type checkers — these constants are
    private-by-convention to each rule package and not all of them
    are surfaced via ``__all__``. The lookup is offensive: a missing
    constant raises ``AttributeError`` (clear "you renamed the
    constant" diagnostic) rather than silently substituting an
    empty vocabulary.

    Adding a new rule: register a branch here and the rule becomes
    immediately reaudit-targetable. Forgetting to register means the
    rule passes ``_supported_rules()`` (which is derived from
    ``BUILTIN_RULES``) but ``reaudit_entries`` fails with a clear
    "no vocabulary registered" error, not silent miscounting.
    """
    if rule_filter == "trust_tier.tier_model":
        rules_dict: dict[str, Any] = _lookup_module_attr("elspeth_lints.rules.trust_tier.tier_model.rule", "RULES")
        return frozenset(rules_dict.keys())
    if rule_filter == "immutability.freeze_guards":
        rules_dict = _lookup_module_attr("elspeth_lints.rules.immutability.freeze_guards.rule", "RULES")
        return frozenset(rules_dict.keys())
    if rule_filter == "immutability.frozen_annotations":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.immutability.frozen_annotations.rule",
                    "RULE_ID",
                )
            }
        )
    if rule_filter == "audit_evidence.guard_symmetry":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.audit_evidence.guard_symmetry.rule",
                    "LEGACY_RULE_ID",
                )
            }
        )
    if rule_filter == "audit_evidence.gve_attribution":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.audit_evidence.gve_attribution.rule",
                    "LEGACY_RULE_ID",
                )
            }
        )
    if rule_filter == "audit_evidence.tier_1_decoration":
        # tier_1_decoration emits two sub-rule ids (TDE1, TDE2); the
        # constants live in the rule's metadata module.
        module = "elspeth_lints.rules.audit_evidence.tier_1_decoration.metadata"
        return frozenset(
            {
                _lookup_module_attr(module, "RULE_TDE1"),
                _lookup_module_attr(module, "RULE_TDE2"),
            }
        )
    if rule_filter == "plugin_contract.component_type":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.plugin_contract.component_type.rule",
                    "LEGACY_RULE_ID",
                )
            }
        )
    if rule_filter == "plugin_contract.options_metadata":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.plugin_contract.options_metadata.rule",
                    "RULE_ID",
                )
            }
        )
    if rule_filter == "plugin_contract.plugin_hashes":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.plugin_contract.plugin_hashes.rule",
                    "RULE_ID",
                )
            }
        )
    if rule_filter == "composer.catch_order":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.composer.catch_order.rule", "RULE_ID")})
    if rule_filter == "composer.exception_channel":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.composer.exception_channel.rule", "RULE_ID")})
    if rule_filter == "manifest.contract_manifest":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.manifest.contract_manifest.rule", "RULE_ID")})
    if rule_filter == "manifest.symbol_inventory":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.manifest.symbol_inventory.rule", "RULE_ID")})
    if rule_filter == "manifest.test_to_source_mapping":
        return frozenset(
            {
                _lookup_module_attr(
                    "elspeth_lints.rules.manifest.test_to_source_mapping.rule",
                    "RULE_ID",
                )
            }
        )
    if rule_filter == "trust_boundary.tests":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.trust_boundary.tests.rule", "RULE_ID")})
    if rule_filter == "trust_boundary.scope":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.trust_boundary.scope.rule", "RULE_ID")})
    if rule_filter == "trust_boundary.tier":
        return frozenset({_lookup_module_attr("elspeth_lints.rules.trust_boundary.tier.rule", "RULE_ID")})

    raise ReauditError(
        f"No sub-rule vocabulary is registered for rule {rule_filter!r}. "
        "Add a branch to _valid_rule_ids_for() so load_allowlist can "
        "validate per_file_rules entries that reference this rule's "
        "sub-rule ids."
    )


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
        lines.append(f"{symbol}  {outcome.divergence.value}  fresh={fresh}")
    lines.append("")
    lines.append("Summary:")
    for name, count in report.summary:
        lines.append(f"  {name:<32} {count}")
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


def render_report_json(report: ReauditReport) -> str:
    """Full report as JSON.

    Uses ``dataclasses.asdict`` then walks the resulting tree to
    encode enums as their string values and datetimes as ISO-8601
    strings. The ``AllowlistEntry`` field becomes a nested object;
    the ``matched`` runtime-only field is included for round-trip
    fidelity even though reaudit never matches entries against
    findings (we re-scan instead).
    """
    import json

    payload: dict[str, Any] = {
        "outcomes": [
            {
                "entry": _entry_to_json(outcome.entry),
                "original_verdict": _verdict_value(outcome.original_verdict),
                "original_model_verdict": _verdict_value(outcome.original_model_verdict),
                "fresh_verdict": _verdict_value(outcome.fresh_verdict),
                "fresh_rationale": outcome.fresh_rationale,
                "fresh_recorded_at": (outcome.fresh_recorded_at.isoformat() if outcome.fresh_recorded_at is not None else None),
                "divergence": outcome.divergence.value,
                "code_snapshot": outcome.code_snapshot,
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
      Notes. The "Entry" column is the canonical key (file:rule:symbol
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
        lines.append("| Entry | Original Verdict | Fresh Verdict | Notes |")
        lines.append("| --- | --- | --- | --- |")
        for outcome in bucket:
            entry_key = _md_escape(outcome.entry.key)
            original = _format_original_verdict(outcome)
            fresh = outcome.fresh_verdict.value if outcome.fresh_verdict is not None else "<no judge call>"
            notes = _md_escape(_outcome_notes(outcome))
            lines.append(f"| `{entry_key}` | {original} | {fresh} | {notes} |")
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
    if outcome.fresh_rationale is None:
        return ""
    return outcome.fresh_rationale.splitlines()[0] if outcome.fresh_rationale else ""


def _md_escape(text: str) -> str:
    """Escape pipes and backslashes for markdown table cells.

    Tables break on bare ``|``; backticks in cells are fine because the
    Entry column is already inline-coded.
    """
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
