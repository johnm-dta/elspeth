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
* **N:N where N>=2** -> symmetric ambiguous; safe to auto-pair only when
  all entries in the group carry equivalent audit metadata
  (``owner``/``reason``/``safety``). If the metadata differs, any sorted
  pairing could attach one site's rationale to a different finding, so
  the group stays ambiguous. Gated by ``allow_symmetric_pairing`` because
  the property is rule-local; another rule whose ``reason:`` text cited
  specific lines could not share this assumption.
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

import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, cast

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from elspeth_lints.core.allowlist import AllowlistEntry, PerFileRule
from elspeth_lints.core.atomic_io import atomic_update_text

from .rule import (
    Finding,
    _finding_key_for,
    _load_tier_model_allowlist,
    _match_per_file_rule,
    scan_directory,
    scan_layer_imports_directory,
)

_FP_TAG: Final = ":fp="
DEFAULT_ROTATION_LOG_PATH: Final = Path(".elspeth/rotations.log")
_JUDGE_METADATA_SIGNATURE_PREFIX: Final = "hmac-sha256:v1:"
_REJUDGE_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "judge_verdict",
        "judge_recorded_at",
        "judge_model",
        "judge_policy_hash",
        "judge_rationale",
        "file_fingerprint",
        "ast_path",
        "judge_metadata_signature",
    }
)


@dataclass(frozen=True, slots=True)
class _AllowHitKeyRecord:
    """Raw allow-hit identity used by the rotation-audit diff."""

    key: str
    source_binding: tuple[str, object, object] | None
    judge_metadata_signature: object | None


def identity_prefix(canonical_key: str) -> str:
    """Return the canonical key minus its ``:fp=<hex>`` suffix.

    Raises ValueError if the input is not a canonical key (no ``:fp=``).
    """
    if _FP_TAG not in canonical_key:
        raise ValueError(f"missing {_FP_TAG!r} suffix in canonical key: {canonical_key!r}")
    return canonical_key.rsplit(_FP_TAG, 1)[0]


def _finding_covered_by_per_file_rule(finding: Finding, rules: list[PerFileRule]) -> bool:
    """Return True if any wildcard ``per_file_rule`` matches the finding.

    Delegates to tier_model's shared per-file matcher so rotate cannot
    drift from the production check path.
    """
    return _match_per_file_rule(rules, _finding_key_for(finding)) is not None


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


def _rotation_metadata_signature(entry: AllowlistEntry) -> tuple[str, str, str]:
    """Return the audit metadata that must survive symmetric pairing intact."""
    return (entry.owner, entry.reason, entry.safety)


def _entries_share_rotation_metadata(entries: list[AllowlistEntry]) -> bool:
    """Return True when every entry has identical rotation-carry metadata."""
    if not entries:
        return True
    first = _rotation_metadata_signature(entries[0])
    return all(_rotation_metadata_signature(entry) == first for entry in entries[1:])


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
        if _finding_covered_by_per_file_rule(finding, rules):
            continue
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
            # Safe only when all entries carry equivalent audit metadata;
            # otherwise sorted pairing can swap owner/reason/safety between
            # logical sites.
            if not _entries_share_rotation_metadata(es):
                ambiguous.append(
                    AmbiguousGroup(
                        prefix=prefix,
                        finding_count=len(fs),
                        entry_count=len(es),
                        entry_keys=tuple(e.key for e in es),
                        finding_keys=tuple(f.canonical_key for f in fs),
                    )
                )
                continue
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


@dataclass(frozen=True, slots=True)
class RotationAuditViolation:
    """One fingerprint rotation in the PR diff without a manifest record."""

    allowlist_file: str
    allowlist_dir: str
    source_file: str
    old_key: str
    new_key: str


@dataclass(frozen=True, slots=True)
class RotationAuditCoverageReport:
    """Result of checking PR rotations against ``.elspeth/rotations.log``."""

    baseline_ref: str
    rotation_log_path: str
    checked_rotation_count: int
    violations: tuple[RotationAuditViolation, ...]

    @property
    def passes(self) -> bool:
        return not self.violations


class RotationAuditError(RuntimeError):
    """The rotation audit check could not proceed."""


def apply_plan(
    plan: RotationPlan,
    *,
    allowlist_dir: Path | None = None,
    remove_stale: bool = False,
    accept_todo_debt: bool = False,
    rotation_log_path: Path | None = None,
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

    The full read → surgical mutation → durable replace sequence runs under
    one ``atomic_update_text`` lock per YAML file. If a rotation's ``old_key``
    is not present or appears more than once, this raises ``RuntimeError``
    rather than silently no-op'ing or rewriting the wrong site -- the former
    hides drift, the latter corrupts the audit trail.

    Args:
        plan: The rotation plan produced by ``plan_rotations`` /
            ``scan_for_rotations``.
        allowlist_dir: Directory containing the per-module YAML files.
            ``source_file`` on each entry is just the YAML filename (e.g.
            ``"web.yaml"``) -- the allowlist loader stores ``yaml_file.name``,
            not the full path. We resolve against ``allowlist_dir`` here.
            When ``None`` (legacy single-file callers and tests that pass
            absolute paths), ``source_file`` is used verbatim.
        remove_stale: When ``True``, entries listed in
            ``plan.stale_entries`` are removed from their source YAML files.
            The default is ``False`` because stale classification can be a
            partial-scan artefact; deletion requires an explicit operator
            choice.
        accept_todo_debt: When ``False`` (default), refuse to apply any
            plan that surfaced historical TODO-stub allowlist debt. The
            operator must either clean that debt first or pass the explicit
            override.
        rotation_log_path: Optional JSONL manifest path. CLI applies pass
            ``.elspeth/rotations.log`` so each real rotation leaves a durable
            audit record; unit tests and library callers may leave it unset.

    Returns:
        Map of source file path -> ``ApplyResult`` with per-file counts.
    """
    if plan.todo_entries and not accept_todo_debt:
        raise RuntimeError(
            f"refusing to apply rotation plan while {len(plan.todo_entries)} TODO-stub allowlist "
            "entry/entries remain. Resolve the placeholder owner/reason debt first, or pass "
            "accept_todo_debt=True / --accept-todo-debt to make the debt acceptance explicit."
        )

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
        rotations = by_file_rotations[source_file] if source_file in by_file_rotations else []
        stale_keys = by_file_stale[source_file] if source_file in by_file_stale else set()
        source_file_for_update = source_file
        rotations_for_update = tuple(rotations)
        stale_keys_for_update = frozenset(stale_keys)

        removed = 0

        def mutate(
            current: str | None,
            *,
            source_file: str = source_file_for_update,
            rotations: tuple[Rotation, ...] = rotations_for_update,
            stale_keys: frozenset[str] = stale_keys_for_update,
        ) -> str:
            nonlocal removed
            if current is None:
                raise RuntimeError(f"allowlist file not found during apply: {source_file}")

            text = current
            for rotation in rotations:
                text = _replace_allow_hit_key(
                    text,
                    old_key=rotation.old_key,
                    new_key=rotation.new_key,
                    source_file=source_file,
                )

            if stale_keys:
                text, removed = _remove_stale_entries(text, stale_keys)
            return text

        atomic_update_text(path, mutate, encoding="utf-8")
        result[source_file] = ApplyResult(
            rotations_applied=len(rotations),
            stale_entries_removed=removed,
        )

    if rotation_log_path is not None and result:
        _append_rotation_manifest(
            plan=plan,
            result=result,
            allowlist_dir=allowlist_dir,
            rotation_log_path=rotation_log_path,
        )

    return result


def _append_rotation_manifest(
    *,
    plan: RotationPlan,
    result: dict[str, ApplyResult],
    allowlist_dir: Path | None,
    rotation_log_path: Path,
) -> None:
    """Append one JSONL record after a successful apply."""
    rotations_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rotation in plan.rotations:
        rotations_by_file[rotation.entry_source_file].append(
            {
                "source_file": rotation.entry_source_file,
                "old_key": rotation.old_key,
                "new_key": rotation.new_key,
            }
        )

    removed_stale = [
        {"source_file": stale.source_file, "key": stale.key}
        for stale in plan.stale_entries
        if stale.source_file in result and result[stale.source_file].stale_entries_removed
    ]
    record = {
        "schema_version": 1,
        "kind": "tier_model_rotation",
        "recorded_at": datetime.now(UTC).isoformat(),
        "allowlist_dir": "" if allowlist_dir is None else allowlist_dir.as_posix(),
        "rotations": [item for source_file in sorted(rotations_by_file) for item in rotations_by_file[source_file]],
        "stale_entries_removed": removed_stale,
        "applied": {
            source_file: {
                "rotations_applied": apply_result.rotations_applied,
                "stale_entries_removed": apply_result.stale_entries_removed,
            }
            for source_file, apply_result in sorted(result.items())
        },
    }
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"

    def append(current: str | None) -> str:
        return (current or "") + line

    atomic_update_text(rotation_log_path, append, encoding="utf-8", create_parent=True)


def check_rotation_audit_coverage(
    *,
    allowlist_root: Path,
    baseline_ref: str,
    repo_root: Path,
    rotation_log_path: Path = DEFAULT_ROTATION_LOG_PATH,
) -> RotationAuditCoverageReport:
    """Verify fingerprint-only allowlist rotations have manifest records."""
    repo_root = repo_root.resolve()
    allowlist_root = _resolve_under_repo(path=allowlist_root, repo_root=repo_root)
    if not repo_root.is_dir():
        raise RotationAuditError(f"--repo-root {repo_root} is not a directory")
    if not allowlist_root.is_dir():
        raise RotationAuditError(f"--allowlist-root {allowlist_root} is not a directory")

    expected_rotations = _rotation_requirements_from_git_diff(
        allowlist_root=allowlist_root,
        baseline_ref=baseline_ref,
        repo_root=repo_root,
    )
    recorded_rotations = _load_rotation_manifest_records(rotation_log_path=rotation_log_path, repo_root=repo_root)
    violations = [
        rotation
        for rotation in expected_rotations
        if not _rotation_is_covered_by_manifest(rotation=rotation, recorded_rotations=recorded_rotations)
    ]
    resolved_log = rotation_log_path if rotation_log_path.is_absolute() else repo_root / rotation_log_path
    return RotationAuditCoverageReport(
        baseline_ref=baseline_ref,
        rotation_log_path=resolved_log.as_posix(),
        checked_rotation_count=len(expected_rotations),
        violations=tuple(violations),
    )


def _rotation_requirements_from_git_diff(
    *,
    allowlist_root: Path,
    baseline_ref: str,
    repo_root: Path,
) -> tuple[RotationAuditViolation, ...]:
    rel_root = _relative_to_repo(allowlist_root, repo_root)
    result = _run_git(["diff", "--name-only", "-z", "--diff-filter=ACMRT", baseline_ref, "HEAD", "--", rel_root], repo_root=repo_root)
    if result.returncode != 0:
        raise RotationAuditError(f"git diff could not inspect baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")

    requirements: list[RotationAuditViolation] = []
    for rel_path in (path for path in result.stdout.split("\0") if path.endswith((".yaml", ".yml"))):
        head_path = repo_root / rel_path
        if not head_path.is_file():
            continue
        baseline_text = _git_show(baseline_ref=baseline_ref, rel_path=rel_path, repo_root=repo_root)
        if baseline_text is None:
            continue
        head_text = head_path.read_text(encoding="utf-8")
        old_by_prefix = _allow_hit_key_records_by_prefix(baseline_text, source_label=f"baseline {baseline_ref}:{rel_path}")
        new_by_prefix = _allow_hit_key_records_by_prefix(head_text, source_label=rel_path)
        allowlist_dir = Path(rel_path).parent.as_posix()
        source_file = Path(rel_path).name
        for prefix in sorted(set(old_by_prefix) & set(new_by_prefix)):
            old_records = sorted(old_by_prefix[prefix], key=lambda record: record.key)
            new_records = sorted(new_by_prefix[prefix], key=lambda record: record.key)
            if len(old_records) != len(new_records):
                continue
            for old_record, new_record in zip(old_records, new_records, strict=True):
                if old_record.key == new_record.key:
                    continue
                if _is_rejudged_key_change(old_record, new_record):
                    continue
                requirements.append(
                    RotationAuditViolation(
                        allowlist_file=rel_path,
                        allowlist_dir=allowlist_dir,
                        source_file=source_file,
                        old_key=old_record.key,
                        new_key=new_record.key,
                    )
                )
    return tuple(requirements)


def _rotation_is_covered_by_manifest(
    *,
    rotation: RotationAuditViolation,
    recorded_rotations: set[tuple[str, str, str, str]],
) -> bool:
    """Return whether manifest records cover ``rotation`` directly or via adjacent hops."""

    direct_record = (rotation.allowlist_dir, rotation.source_file, rotation.old_key, rotation.new_key)
    if direct_record in recorded_rotations:
        return True

    next_keys_by_old_key: dict[str, set[str]] = defaultdict(set)
    for allowlist_dir, source_file, old_key, new_key in recorded_rotations:
        if allowlist_dir != rotation.allowlist_dir or source_file != rotation.source_file:
            continue
        next_keys_by_old_key[old_key].add(new_key)

    visited: set[str] = set()
    frontier = [rotation.old_key]
    while frontier:
        current_key = frontier.pop()
        if current_key in visited:
            continue
        visited.add(current_key)
        for next_key in sorted(next_keys_by_old_key.get(current_key, ())):
            if next_key == rotation.new_key:
                return True
            if next_key not in visited:
                frontier.append(next_key)
    return False


def _allow_hit_key_records_by_prefix(text: str, *, source_label: str) -> dict[str, list[_AllowHitKeyRecord]]:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise RotationAuditError(f"{source_label}: could not parse YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise RotationAuditError(f"{source_label}: YAML root must be a mapping")
    raw_entries = data.get("allow_hits", [])
    if raw_entries is None:
        return {}
    if not isinstance(raw_entries, list):
        raise RotationAuditError(f"{source_label}: allow_hits must be a list")

    by_prefix: dict[str, list[_AllowHitKeyRecord]] = defaultdict(list)
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise RotationAuditError(f"{source_label}: allow_hits[{index}] must be a mapping")
        key = raw_entry.get("key")
        if not isinstance(key, str) or _FP_TAG not in key:
            continue
        by_prefix[identity_prefix(key)].append(_allow_hit_key_record(key=key, raw_entry=raw_entry))
    return by_prefix


def _allow_hit_key_record(*, key: str, raw_entry: dict[object, object]) -> _AllowHitKeyRecord:
    if not _has_complete_rejudge_metadata(raw_entry):
        return _AllowHitKeyRecord(
            key=key,
            source_binding=None,
            judge_metadata_signature=None,
        )
    return _AllowHitKeyRecord(
        key=key,
        source_binding=(key, raw_entry["file_fingerprint"], raw_entry["ast_path"]),
        judge_metadata_signature=raw_entry["judge_metadata_signature"],
    )


def _has_complete_rejudge_metadata(raw_entry: dict[object, object]) -> bool:
    for field in _REJUDGE_REQUIRED_FIELDS:
        value = raw_entry.get(field)
        if value is None or value == "":
            return False
    signature = raw_entry.get("judge_metadata_signature")
    if not isinstance(signature, str):
        return False
    digest = signature.removeprefix(_JUDGE_METADATA_SIGNATURE_PREFIX)
    return (
        signature.startswith(_JUDGE_METADATA_SIGNATURE_PREFIX) and len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    )


def _is_rejudged_key_change(old_record: _AllowHitKeyRecord, new_record: _AllowHitKeyRecord) -> bool:
    """Return whether a same-prefix key change is a fresh judge write, not rotation."""
    if old_record.source_binding is None or new_record.source_binding is None:
        return False
    if old_record.source_binding == new_record.source_binding:
        return False
    return old_record.judge_metadata_signature != new_record.judge_metadata_signature


def _load_rotation_manifest_records(*, rotation_log_path: Path, repo_root: Path) -> set[tuple[str, str, str, str]]:
    path = rotation_log_path if rotation_log_path.is_absolute() else repo_root / rotation_log_path
    if not path.exists():
        return set()

    records: set[tuple[str, str, str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw_record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RotationAuditError(f"{path}:{line_number}: malformed JSONL record: {exc}") from exc
        if not isinstance(raw_record, dict):
            raise RotationAuditError(f"{path}:{line_number}: rotation manifest record must be a JSON object")
        if raw_record.get("kind") != "tier_model_rotation":
            continue
        if raw_record.get("schema_version") != 1:
            raise RotationAuditError(f"{path}:{line_number}: unsupported rotation manifest schema_version")
        allowlist_dir = _normalise_manifest_allowlist_dir(raw_record.get("allowlist_dir"), repo_root=repo_root)
        raw_rotations = raw_record.get("rotations", [])
        if not isinstance(raw_rotations, list):
            raise RotationAuditError(f"{path}:{line_number}: rotations must be a list")
        for index, raw_rotation in enumerate(raw_rotations):
            if not isinstance(raw_rotation, dict):
                raise RotationAuditError(f"{path}:{line_number}: rotations[{index}] must be an object")
            source_file = raw_rotation.get("source_file")
            old_key = raw_rotation.get("old_key")
            new_key = raw_rotation.get("new_key")
            if not all(isinstance(value, str) and value for value in (source_file, old_key, new_key)):
                raise RotationAuditError(f"{path}:{line_number}: rotations[{index}] must include source_file, old_key, and new_key")
            assert isinstance(source_file, str)
            assert isinstance(old_key, str)
            assert isinstance(new_key, str)
            records.add((allowlist_dir, source_file, old_key, new_key))
    return records


def _normalise_manifest_allowlist_dir(value: object, *, repo_root: Path) -> str:
    if not isinstance(value, str):
        raise RotationAuditError("rotation manifest allowlist_dir must be a string")
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix().rstrip("/")


def _resolve_under_repo(*, path: Path, repo_root: Path) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    return candidate.resolve()


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise RotationAuditError(f"{path} is not inside repo root {repo_root}") from exc


def _git_show(*, baseline_ref: str, rel_path: str, repo_root: Path) -> str | None:
    result = _run_git(["show", f"{baseline_ref}:{rel_path}"], repo_root=repo_root)
    if result.returncode == 0:
        return result.stdout
    if not _git_commit_exists(baseline_ref=baseline_ref, repo_root=repo_root):
        raise RotationAuditError(f"git show could not resolve baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")
    if not _git_path_exists(baseline_ref=baseline_ref, rel_path=rel_path, repo_root=repo_root):
        return None
    raise RotationAuditError(f"git show failed for baseline {baseline_ref}:{rel_path}: {_git_failure_detail(result)}")


def _run_git(args: list[str], *, repo_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
            env=env,
        )
    except OSError as exc:
        raise RotationAuditError(f"git command failed to start: {exc}") from exc


def _git_commit_exists(*, baseline_ref: str, repo_root: Path) -> bool:
    result = _run_git(["rev-parse", "--verify", "--quiet", f"{baseline_ref}^{{commit}}"], repo_root=repo_root)
    return result.returncode == 0


def _git_path_exists(*, baseline_ref: str, rel_path: str, repo_root: Path) -> bool:
    result = _run_git(["cat-file", "-e", f"{baseline_ref}:{rel_path}"], repo_root=repo_root)
    return result.returncode == 0


def _git_failure_detail(result: subprocess.CompletedProcess[str]) -> str:
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    if stderr:
        return stderr
    if stdout:
        return stdout
    return f"git exited with status {result.returncode}"


def _replace_allow_hit_key(text: str, *, old_key: str, new_key: str, source_file: str) -> str:
    """Replace one ``allow_hits`` key scalar, never arbitrary prose."""
    spans = _allow_hit_key_value_spans(text, old_key)
    if not spans:
        raise RuntimeError(
            f"rotation old_key not found in {source_file}: {old_key!r} -- entry may have been deleted between scan and apply"
        )
    if len(spans) > 1:
        raise RuntimeError(
            f"rotation old_key occurs {len(spans)}x in {source_file}: "
            f"{old_key!r} -- refusing to rotate to avoid wrong-target "
            "update; manual resolution required"
        )
    start, end = spans[0]
    return text[:start] + new_key + text[end:]


def _allow_hit_key_value_spans(text: str, match_key: str) -> list[tuple[int, int]]:
    """Return source spans for ``key`` scalar values equal to ``match_key``."""
    root = yaml.compose(text)
    if root is None:
        return []

    if isinstance(root, MappingNode):
        allow_hits = _mapping_value_node(root, "allow_hits")
        if allow_hits is None:
            return []
        if not isinstance(allow_hits, SequenceNode):
            raise RuntimeError(f"allow_hits must be a sequence, got {type(allow_hits).__name__}")
        entries = allow_hits.value
    elif isinstance(root, SequenceNode):
        # Legacy single-file fixtures historically used a bare sequence.
        entries = root.value
    else:
        raise RuntimeError(f"allowlist YAML root must be a mapping or sequence, got {type(root).__name__}")

    spans: list[tuple[int, int]] = []
    for item in entries:
        if not isinstance(item, MappingNode):
            raise RuntimeError(f"allow_hits entries must be mappings, got {type(item).__name__}")
        key_node = _mapping_value_node(item, "key")
        if isinstance(key_node, ScalarNode) and key_node.value == match_key:
            spans.append((key_node.start_mark.index, key_node.end_mark.index))
    return spans


def _remove_stale_entries(text: str, stale_keys: Collection[str]) -> tuple[str, int]:
    """Surgically remove ``allow_hits`` blocks whose ``key:`` is in stale_keys.

    Pure function; takes YAML text + the set of keys to drop, returns
    (new_text, removed_count). Comments and all non-stale entries are
    preserved byte-identical.

    Algorithm: parse the YAML event tree and use node source spans for
    entries whose ``key`` scalar matches. This avoids hand-parsing
    indentation, which misclassifies valid flow-style mappings and other
    non-canonical YAML layouts.
    """
    if not stale_keys:
        return text, 0

    root = yaml.compose(text)
    if root is None:
        return text, 0
    if not isinstance(root, MappingNode):
        raise RuntimeError(f"allowlist YAML root must be a mapping, got {type(root).__name__}")

    allow_hits = _mapping_value_node(root, "allow_hits")
    if allow_hits is None:
        return text, 0
    if not isinstance(allow_hits, SequenceNode):
        raise RuntimeError(f"allow_hits must be a sequence, got {type(allow_hits).__name__}")

    spans: list[tuple[int, int]] = []
    for item in allow_hits.value:
        if not isinstance(item, MappingNode):
            raise RuntimeError(f"allow_hits entries must be mappings, got {type(item).__name__}")
        key_node = _mapping_value_node(item, "key")
        if isinstance(key_node, ScalarNode) and key_node.value in stale_keys:
            spans.append((_line_start(text, item.start_mark.index), _node_span_end(text, item.end_mark.index)))

    if not spans:
        return text, 0

    rewritten = text
    for start, end in sorted(spans, reverse=True):
        rewritten = rewritten[:start] + rewritten[end:]
    return rewritten, len(spans)


def _mapping_value_node(mapping: MappingNode, key: str) -> Node | None:
    """Return the value node for ``key`` in a YAML mapping node."""
    for key_node, value_node in mapping.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            return cast(Node, value_node)
    return None


def _line_start(text: str, index: int) -> int:
    """Return the start offset for the line containing ``index``."""
    return text.rfind("\n", 0, index) + 1


def _line_end_after(text: str, index: int) -> int:
    """Return the offset just after the line containing ``index``."""
    newline = text.find("\n", index)
    if newline == -1:
        return len(text)
    return newline + 1


def _node_span_end(text: str, index: int) -> int:
    """Return a deletion end offset for a YAML node end mark."""
    if index >= len(text):
        return len(text)
    if index == _line_start(text, index):
        return index
    return _line_end_after(text, index)
