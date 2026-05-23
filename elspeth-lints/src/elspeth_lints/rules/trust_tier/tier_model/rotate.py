"""Mechanical fingerprint rotation for tier_model allowlist entries.

When an AST shape change (e.g., adding a top-of-file ``ImportFrom``) shifts
``Module.body[N]`` indices, the fingerprint suffix embedded in every
canonical allowlist key downstream of the insertion rotates even though no
underlying violation site moved. Today the operator (or an agent under
deadline pressure) hand-edits the YAML and tends to bundle the rotation
together with substantive exemption changes; that masks the substantive
changes in CI review.

This module separates the two. Rotation is mechanical, identity-preserving,
and requires no judgment. It is the first slice of the CICD-judge CLI
prototype; the judge path (``justify``) is layered on top after this exists.

Identity model
--------------

Two findings or two entries share an *identity prefix* iff their canonical
keys agree up to but not including the ``:fp=<hex>`` suffix. The identity
prefix is ``<path>:<rule>:<symbol-segments>``. Within a single source file,
governance forbids duplicate canonical keys, so a single ``(file, rule,
symbol-segments)`` triple corresponds to at most one logical violation site.

Classification
--------------

For every identity prefix that appears in the current scan output, the
allowlist, or both:

* **1 finding, 1 entry, fp differs** -> rotation (safe to auto-apply)
* **1 finding, 1 entry, fp matches** -> unchanged
* **>=1 finding, 0 entries** -> new finding (judge path, not our concern)
* **0 findings, >=1 entries** -> stale entry (operator-confirmed cleanup)
* **N:N where N>=2** -> symmetric ambiguous; safe to auto-pair *for this
  rule* because tier_model's ``reason:`` discipline keeps metadata at the
  qname level (e.g. "Tier 3 boundary - JWT header kid is optional"), not
  the line level. Any consistent pairing yields the same yaml end-state.
  Gated by ``allow_symmetric_pairing`` because the property is rule-local;
  another rule whose ``reason:`` text cited specific lines could not share
  this assumption.
* **N:M where N!=M and {N,M} >= 1** -> genuinely ambiguous; surfaced to the
  operator for manual resolution. Auto-pairing here is unsafe regardless
  of metadata discipline.

Out of scope
------------

* Adding new allowlist entries: the prior tooling auto-created ``owner:
  TODO`` placeholder entries for any unmatched finding. That behaviour is
  removed PERMANENTLY (audit-integrity bug: it silently added unreviewed
  exemptions on every refactor). When new findings exist after a rotation
  scan, the tool surfaces them and exits non-zero; the agent fixes the
  code or writes a real allowlist entry with substantive justification.
  The forthcoming ``justify`` subcommand will gate manual entries through
  an LLM judge.
* Updating ``per_file_rules`` (they have no fingerprint).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from elspeth_lints.core.allowlist import AllowlistEntry, FindingKey, PerFileRule
from elspeth_lints.core.atomic_io import atomic_write_text

from .rule import (
    Finding,
    _load_tier_model_allowlist,
    scan_directory,
    scan_layer_imports_directory,
)

_FP_TAG: Final = ":fp="


def identity_prefix(canonical_key: str) -> str:
    """Return the canonical key minus its ``:fp=<hex>`` suffix.

    Raises ValueError if the input is not a canonical key (no ``:fp=``).
    """
    if _FP_TAG not in canonical_key:
        raise ValueError(f"missing {_FP_TAG!r} suffix in canonical key: {canonical_key!r}")
    return canonical_key.rsplit(_FP_TAG, 1)[0]


def _finding_covered_by_per_file_rule(finding: Finding, rules: list[PerFileRule]) -> bool:
    """Return True if any wildcard ``per_file_rule`` matches the finding.

    Mirrors the matching semantics of ``rule._match_finding`` so that the
    rotate command's "new finding" classification agrees with the rule's
    own CI-pass/fail decision. Without this, wildcard-covered findings
    appear as "new" and the rotate command exits non-zero on a tree that
    the production check would call clean.
    """
    key = FindingKey(
        file_path=finding.file_path,
        rule_id=finding.rule_id,
        symbol_context=finding.symbol_context,
        fingerprint=finding.fingerprint,
    )
    return any(rule.matches(key) for rule in rules)


def _refuse_rotation_of_judge_gated_entry(entry: AllowlistEntry) -> None:
    """Crash if a judge-gated entry would be rotated.

    Rotation rebinds an entry's canonical key (and its embedded
    fingerprint) to a new AST signature. For a judge-gated entry the
    persisted ``file_fingerprint`` + ``ast_path`` bind the quartet to
    the exact bytes and AST node the judge inspected; the rotated key
    would point at different code while the binding fields still
    described the original code. Auto-rotating the binding fields
    requires re-running the judge against the new code — which defeats
    the point of rotation (a mechanical, no-judgement refactor sweep).
    The audit-honest response is to refuse: the operator deletes the
    stale entry, re-justifies against the rotated code, and the new
    quartet records what the judge actually said about the new
    location.

    Raises ``RuntimeError`` (matching the existing rotate-error
    convention in :func:`apply_plan`) with a message that names the
    entry key and source file so the operator can locate and resolve
    it.
    """
    if entry.judge_verdict is None:
        return
    raise RuntimeError(
        f"refusing to rotate judge-gated allowlist entry in {entry.source_file}: "
        f"{entry.key!r} carries judge_verdict={entry.judge_verdict.value!r}. "
        "Rotation would silently rebind the persisted file_fingerprint + ast_path "
        "to different code than the judge actually inspected. Re-run justify against "
        "the rotated location (deleting this entry first) so the new quartet records "
        "what the judge says about the new code."
    )


def fingerprint_of(canonical_key: str) -> str:
    """Return the fingerprint hex from a canonical key's ``:fp=<hex>`` suffix.

    Raises ValueError if the input is not a canonical key (no ``:fp=``).
    """
    if _FP_TAG not in canonical_key:
        raise ValueError(f"missing {_FP_TAG!r} suffix in canonical key: {canonical_key!r}")
    return canonical_key.rsplit(_FP_TAG, 1)[1]


@dataclass(frozen=True, slots=True)
class Rotation:
    """A mechanical fingerprint rotation: same identity prefix, new fingerprint."""

    old_key: str
    new_key: str
    entry_source_file: str

    @property
    def prefix(self) -> str:
        return identity_prefix(self.old_key)


@dataclass(frozen=True, slots=True)
class AmbiguousGroup:
    """An identity prefix where 1:1 auto-pairing would be unsafe.

    Surfaced to the operator for manual resolution rather than silently
    guessed. Example: a method that previously had one R1 violation but now
    has two (or vice versa) lands here, because the rotate engine cannot
    know which finding inherits which entry without semantic context.
    """

    prefix: str
    finding_count: int
    entry_count: int
    entry_keys: tuple[str, ...]
    finding_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NewFinding:
    """A finding with no matching allowlist entry by identity prefix.

    The prior tool auto-created an ``owner: TODO`` allowlist entry for these.
    That behaviour has been removed permanently; new findings are surfaced
    and gate the run with a non-zero exit so the agent fixes the code or
    writes a substantive allowlist entry.
    """

    canonical_key: str
    file_path: str
    rule_id: str
    line: int
    message: str


@dataclass(frozen=True, slots=True)
class StaleEntry:
    """An allowlist entry whose identity prefix has no corresponding finding.

    The violation site has been fixed, refactored away, or never existed in
    this configuration. Stale entries are surfaced for operator-confirmed
    cleanup (not auto-removed) because the absence of a finding can also
    mean the file was excluded from this scan rather than truly fixed.
    """

    key: str
    source_file: str
    owner: str
    reason: str


@dataclass(frozen=True, slots=True)
class RotationPlan:
    """Pure data structure produced by ``scan_for_rotations``.

    Apply with ``apply_plan(plan)`` once the operator (or an automated
    workflow) has confirmed the absence of ambiguities or accepted the
    rotation set despite them.
    """

    rotations: tuple[Rotation, ...]
    ambiguous: tuple[AmbiguousGroup, ...]
    stale_entries: tuple[StaleEntry, ...]
    todo_entries: tuple[StaleEntry, ...]
    new_findings: tuple[NewFinding, ...]
    unchanged_count: int

    @property
    def has_rotations(self) -> bool:
        return bool(self.rotations)

    @property
    def has_ambiguity(self) -> bool:
        return bool(self.ambiguous)

    @property
    def stale_entry_count(self) -> int:
        return len(self.stale_entries)

    @property
    def todo_entry_count(self) -> int:
        return len(self.todo_entries)

    @property
    def new_finding_count(self) -> int:
        return len(self.new_findings)

    @property
    def has_new_findings(self) -> bool:
        return bool(self.new_findings)


def scan_for_rotations(
    *,
    source_root: Path,
    allowlist_path: Path,
    allow_symmetric_pairing: bool = True,
) -> RotationPlan:
    """Scan + classify wrapper around ``plan_rotations``.

    Reads the source tree and the allowlist YAML(s), then delegates the pure
    classification to ``plan_rotations``. Keep ``plan_rotations`` separate
    so unit tests can exercise the algorithm without touching the
    filesystem or running the rule's full visitor.
    """
    findings: list[Finding] = list(scan_directory(source_root))
    layer_violations, layer_warnings = scan_layer_imports_directory(source_root)
    findings.extend(layer_violations)
    findings.extend(layer_warnings)

    allowlist = _load_tier_model_allowlist(allowlist_path)

    return plan_rotations(
        findings=findings,
        allowlist_entries=allowlist.entries,
        per_file_rules=allowlist.per_file_rules,
        allow_symmetric_pairing=allow_symmetric_pairing,
    )


def plan_rotations(
    *,
    findings: list[Finding],
    allowlist_entries: list[AllowlistEntry],
    per_file_rules: list[PerFileRule] | None = None,
    allow_symmetric_pairing: bool = True,
) -> RotationPlan:
    """Classify findings against allowlist entries by identity prefix.

    Pure function: takes pre-collected findings and entries, returns a
    ``RotationPlan``. No I/O.

    Args:
        findings: All tier_model findings produced by the current scan.
            Both R1-R7 violations and TC/layer findings belong here.
        allowlist_entries: Entries loaded from the tier_model YAML(s).
        per_file_rules: Wildcard ``per_file_rules`` from the same allowlist.
            Findings covered by a per_file_rule pattern are *not* classified
            as "new" -- they are already allowlisted by wildcard. Defaults
            to no rules (every unmatched finding is "new"). Must be passed
            in production to match the rule's own matching semantics.
        allow_symmetric_pairing: See ``scan_for_rotations``.

    Returns:
        ``RotationPlan`` covering rotations, ambiguous groups, stale
        entries, and the TODO-stub debt count.
    """
    rules = per_file_rules or []
    findings_by_prefix: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        findings_by_prefix[identity_prefix(finding.canonical_key)].append(finding)

    entries_by_prefix: dict[str, list[AllowlistEntry]] = defaultdict(list)
    todo_entries: list[StaleEntry] = []
    for entry in allowlist_entries:
        entries_by_prefix[identity_prefix(entry.key)].append(entry)
        # Detect placeholder-rationale entries created by historical
        # tooling that auto-allowlisted unmatched new findings with
        # owner=TODO and a placeholder reason. These pollute the audit
        # trail and need judge review; we surface the count but never
        # auto-act on them.
        if entry.owner.strip().upper() == "TODO" or entry.reason.strip().upper().startswith("TODO"):
            todo_entries.append(
                StaleEntry(
                    key=entry.key,
                    source_file=entry.source_file,
                    owner=entry.owner,
                    reason=entry.reason,
                )
            )

    rotations: list[Rotation] = []
    ambiguous: list[AmbiguousGroup] = []
    stale_entries: list[StaleEntry] = []
    new_findings_list: list[NewFinding] = []
    unchanged = 0
    # todo_entries already populated above during the allowlist walk

    all_prefixes = set(findings_by_prefix) | set(entries_by_prefix)
    for prefix in all_prefixes:
        fs = findings_by_prefix[prefix] if prefix in findings_by_prefix else []
        es = entries_by_prefix[prefix] if prefix in entries_by_prefix else []
        if len(fs) == 1 and len(es) == 1:
            new_key = fs[0].canonical_key
            old_key = es[0].key
            if new_key == old_key:
                unchanged += 1
            else:
                _refuse_rotation_of_judge_gated_entry(es[0])
                rotations.append(
                    Rotation(
                        old_key=old_key,
                        new_key=new_key,
                        entry_source_file=es[0].source_file,
                    )
                )
        elif fs and not es:
            for finding in fs:
                # Skip findings covered by a wildcard per_file_rule: they
                # are already allowlisted, just not via an entry-level key.
                # Without this filter we'd classify ~700 wildcard-covered
                # findings as "new" and exit non-zero on every clean run.
                if _finding_covered_by_per_file_rule(finding, rules):
                    continue
                new_findings_list.append(
                    NewFinding(
                        canonical_key=finding.canonical_key,
                        file_path=finding.file_path,
                        rule_id=finding.rule_id,
                        line=finding.line,
                        message=finding.message,
                    )
                )
        elif es and not fs:
            for entry in es:
                stale_entries.append(
                    StaleEntry(
                        key=entry.key,
                        source_file=entry.source_file,
                        owner=entry.owner,
                        reason=entry.reason,
                    )
                )
        elif allow_symmetric_pairing and len(fs) == len(es):
            # Symmetric N:N — deterministic sorted-fingerprint pairing.
            # See the module docstring for why this is safe for tier_model.
            old_keys_sorted = sorted(es, key=lambda e: e.key)
            new_keys_sorted = sorted(fs, key=lambda f: f.canonical_key)
            for entry, finding in zip(old_keys_sorted, new_keys_sorted, strict=True):
                if entry.key == finding.canonical_key:
                    unchanged += 1
                else:
                    _refuse_rotation_of_judge_gated_entry(entry)
                    rotations.append(
                        Rotation(
                            old_key=entry.key,
                            new_key=finding.canonical_key,
                            entry_source_file=entry.source_file,
                        )
                    )
        else:
            ambiguous.append(
                AmbiguousGroup(
                    prefix=prefix,
                    finding_count=len(fs),
                    entry_count=len(es),
                    entry_keys=tuple(e.key for e in es),
                    finding_keys=tuple(f.canonical_key for f in fs),
                )
            )

    return RotationPlan(
        rotations=tuple(rotations),
        ambiguous=tuple(ambiguous),
        stale_entries=tuple(stale_entries),
        todo_entries=tuple(todo_entries),
        new_findings=tuple(new_findings_list),
        unchanged_count=unchanged,
    )


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Per-file summary of what ``apply_plan`` did."""

    rotations_applied: int = 0
    stale_entries_removed: int = 0


def apply_plan(
    plan: RotationPlan,
    *,
    allowlist_dir: Path | None = None,
    remove_stale: bool = True,
) -> dict[str, ApplyResult]:
    """Apply rotations (and optionally stale-entry removals) to YAML files.

    Rotations use surgical full-string replacement keyed on the entire
    ``old_key``. Governance forbids duplicate canonical keys within a YAML
    file, so a single ``str.replace`` is sufficient and preserves all
    surrounding structure (comments, ordering, indentation).

    Stale removal walks the YAML line-by-line, deletes the ``- key: <STALE>``
    line and the indented child lines that belong to that entry block,
    leaving all other entries (and all comments) untouched. The block-end
    heuristic is "until the next line that starts a new entry (``- key:``)
    or leaves the ``allow_hits`` indentation level."

    If a rotation's ``old_key`` is not present or appears more than once,
    this raises ``RuntimeError`` rather than silently no-op'ing or rewriting
    the wrong site -- the former hides drift, the latter corrupts the audit
    trail.

    Args:
        plan: The rotation plan produced by ``plan_rotations`` /
            ``scan_for_rotations``.
        allowlist_dir: Directory containing the per-module YAML files.
            ``source_file`` on each entry is just the YAML filename (e.g.
            ``"web.yaml"``) -- the allowlist loader stores ``yaml_file.name``,
            not the full path. We resolve against ``allowlist_dir`` here.
            When ``None`` (legacy single-file callers and tests that pass
            absolute paths), ``source_file`` is used verbatim.
        remove_stale: When ``True`` (default), entries listed in
            ``plan.stale_entries`` are removed from their source YAML files.
            Set ``False`` to apply rotations only and review stale entries
            manually.

    Returns:
        Map of source file path -> ``ApplyResult`` with per-file counts.
    """
    by_file_rotations: dict[str, list[Rotation]] = defaultdict(list)
    for rotation in plan.rotations:
        by_file_rotations[rotation.entry_source_file].append(rotation)

    by_file_stale: dict[str, set[str]] = defaultdict(set)
    if remove_stale:
        for stale in plan.stale_entries:
            by_file_stale[stale.source_file].add(stale.key)

    touched_files = set(by_file_rotations) | set(by_file_stale)
    result: dict[str, ApplyResult] = {}
    for source_file in touched_files:
        candidate = Path(source_file)
        path = candidate if candidate.is_absolute() or allowlist_dir is None else allowlist_dir / candidate
        text = path.read_text(encoding="utf-8")

        rotations = by_file_rotations[source_file] if source_file in by_file_rotations else []
        for rotation in rotations:
            occurrences = text.count(rotation.old_key)
            if occurrences == 0:
                raise RuntimeError(
                    f"rotation old_key not found in {source_file}: {rotation.old_key!r} -- "
                    "entry may have been deleted between scan and apply"
                )
            if occurrences > 1:
                raise RuntimeError(
                    f"rotation old_key occurs {occurrences}x in {source_file}: "
                    f"{rotation.old_key!r} -- refusing to rotate to avoid wrong-target "
                    "update; manual resolution required"
                )
            text = text.replace(rotation.old_key, rotation.new_key, 1)

        stale_keys = by_file_stale[source_file] if source_file in by_file_stale else set()
        removed = 0
        if stale_keys:
            text, removed = _remove_stale_entries(text, stale_keys)

        atomic_write_text(path, text, encoding="utf-8")
        result[source_file] = ApplyResult(
            rotations_applied=len(rotations),
            stale_entries_removed=removed,
        )

    return result


def _remove_stale_entries(text: str, stale_keys: set[str]) -> tuple[str, int]:
    """Surgically remove ``allow_hits`` blocks whose ``key:`` is in stale_keys.

    Pure function; takes YAML text + the set of keys to drop, returns
    (new_text, removed_count). Comments and all non-stale entries are
    preserved byte-identical.

    Algorithm: walk lines. When we see ``- key: <STALE>``, drop that line and
    every following line whose indentation is deeper than the ``-`` marker,
    up to (but not including) the next entry-start or the next less-indented
    structural line.
    """
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("- key:"):
            key_value = stripped[len("- key:") :].strip()
            if key_value in stale_keys:
                # Skip this line and indented continuation lines.
                entry_indent = len(line) - len(stripped)
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.lstrip()
                    if not next_stripped:
                        # Blank line — treat as still inside the block; skip.
                        i += 1
                        continue
                    next_indent = len(next_line) - len(next_stripped)
                    if next_indent <= entry_indent:
                        # Hit a new entry or a less-indented structural line.
                        break
                    i += 1
                removed += 1
                continue
        output.append(line)
        i += 1
    return "".join(output), removed
