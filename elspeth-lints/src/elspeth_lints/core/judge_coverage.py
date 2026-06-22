"""New-entry judge-coverage CI gate (convergent finding C1).

Convergent panel finding C1: judge enforcement is voluntary, not
architectural. The judge primitive ships in ``elspeth-lints justify``
and is recorded in entry metadata, but nothing in CI mechanically
rejects a hand-edited allowlist entry that lacks the judge fields.
Pre-judge and judge-skipped entries are observationally identical to
the loader (both have ``judge_verdict is None``), so an agent who
appends an entry by editing YAML directly produces an honest-looking
"pre-judge" entry that erodes the audit trail.

This module closes that loop. On a diff against the PR merge base (or
the previous commit for protected-branch pushes), every new
``allow_hits`` entry MUST carry the signed judge metadata cluster
(``judge_verdict + judge_recorded_at + judge_model + judge_policy_hash
+ judge_rationale + judge_metadata_signature``). Entries already
present in baseline are grandfathered — including true pre-judge
entries that pre-date the gate. The grandfathering is *rotation-stable*:
an entry whose fingerprint shifted because of an upstream AST refactor
still matches its baseline counterpart, so refactors are not penalised
by demanding fresh judge runs. Existing judged entries are not allowed
to mutate their judge metadata under that grandfathering identity.

**Rotation policy (operator-confirmed 2026-05-23):** rotation
grandfathers. The discriminator is ``(file_path, rule_id,
symbol_part, owner, reason, expires)`` — the fingerprint ``fp=<hex>``
segment is stripped because it is the AST-position artefact rotation
mutates. ``owner`` and ``reason`` discriminate between two distinct
entries that happen to share the parsed-key triple (an unusual but
legal shape that the triple alone cannot disambiguate). ``expires`` is
part of the identity (as it already is for ``per_file_rules``): renewing
a suppression by editing only the expiry date is a fresh decision that
must re-acquire judge metadata, not silently grandfather.

**Count-limited grandfathering:** because ``fp=`` is stripped, an
*additional* unjudged entry sharing an existing discriminator (e.g. a
second ``.get()`` in the same method) would otherwise grandfather for
free. Pre-judge grandfathering is therefore limited to the number of
pre-judge baseline entries per discriminator — a rotation (count
unchanged) passes, but the excess is a new suppression that must carry
judge metadata. ``reaudit`` remains the surface for periodic re-judging. A judged entry whose
source binding changed because the operator re-ran ``justify`` after
source/fingerprint drift is not a grandfathered metadata mutation: it is
counted as a fresh judged record, provided the HEAD entry carries the complete
judge metadata cluster and its ``key`` / ``file_fingerprint`` / ``ast_path``
binding no longer matches any judged baseline binding for the same
discriminator.

In fork PR CI the HMAC key is unavailable, so a "complete" newly-signed entry
is still only shape-valid. ``forbid_unverified_judge_metadata`` lets that lane
reuse this diff logic while rejecting fresh judged records outright; unchanged
baseline entries remain inspectable in shape-only mode, but forks cannot add or
re-sign entries that the runner cannot cryptographically verify.

Boundary discipline: this module routes directories into C1 only when
they contain the standard judge-covered ``allow_hits:`` shape parsed by
``elspeth_lints.core.allowlist._parse_allow_hits``. Newly-added
``per_file_rules:`` entries are surfaced as their own coverage category
because wildcard suppressions cannot carry the judge quartet today; existing
baseline rules are grandfathered but remain counted. Private legacy entry
shapes such as ``allow_classes:`` or custom ``entries:`` blocks do not
route otherwise-standalone directories into C1 because they have no judge
metadata representation. If such a shape appears inside an already
judge-covered directory, it is still reported as
``UNRECOGNIZED_ENTRY_SHAPE`` so it cannot be used as an escape hatch from
the gate.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    _verify_judge_metadata_signature_at_load,
)
from elspeth_lints.core.allowlist_io import (
    AllowlistIOError,
    entry_shape_count,
    iter_yaml_documents,
    load_yaml_mapping_text,
    parse_allow_hits,
)

UNRECOGNIZED_ENTRY_SHAPE = "UNRECOGNIZED_ENTRY_SHAPE"
PER_FILE_RULE_REQUIRES_JUDGE = "PER_FILE_RULE_REQUIRES_JUDGE"
JUDGE_METADATA_MUTATED = "JUDGE_METADATA_MUTATED"
UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC = "UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC"
_ALLOWLIST_ENTRY_KEYS = frozenset({"allow_hits", "allow_classes", "entries", "per_file_rules"})
_JUDGE_COVERED_ENTRY_KEYS = frozenset({"allow_hits", "per_file_rules"})


@dataclass(frozen=True, slots=True)
class JudgeCoverageViolation:
    """One new entry that lacks one or more required judge fields.

    ``entry_key`` is the entry's full YAML ``key:`` value (with the
    ``fp=<hex>`` suffix) — useful for operator search and for
    cross-referencing with ``elspeth-lints justify`` output.
    ``source_file`` is the YAML filename relative to the allowlist
    directory. ``missing_fields`` enumerates which of the atomic
    quartet is absent; an entry can fail for any subset.
    """

    entry_key: str
    source_file: str
    missing_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerFileRuleCoverageEntry:
    """One ``per_file_rules`` entry tracked by C1 coverage.

    Per-file wildcard suppressions are valid allowlist records but they cannot
    carry per-finding judge metadata. C1 therefore diffs them separately from
    ``allow_hits``: baseline rules are grandfathered, while newly-added or
    behavior-changing rules are reported with
    :data:`PER_FILE_RULE_REQUIRES_JUDGE`.
    """

    source_file: str
    index: int
    pattern: str
    rules: tuple[str, ...]
    reason: str
    expires: str | None
    max_hits: int | None

    @property
    def label(self) -> str:
        return _per_file_rule_label(self.index, self.pattern, self.rules)


@dataclass(frozen=True, slots=True)
class JudgeCoverageReport:
    """Result of one judge-coverage diff."""

    head_entry_count: int
    grandfathered_count: int
    new_entry_count: int
    violations: tuple[JudgeCoverageViolation, ...]

    @property
    def passes(self) -> bool:
        return not self.violations


class JudgeCoverageError(RuntimeError):
    """The judge-coverage check cannot proceed.

    Distinguished from ``JudgeCoverageReport(violations=...)``: a
    ``JudgeCoverageError`` means the check itself could not run
    (missing baseline, git failure, malformed YAML at HEAD). A
    populated ``violations`` tuple means the check ran successfully
    and the operator's PR introduced un-judged new entries.
    """


def check_judge_coverage(
    *,
    allowlist_root: Path,
    baseline_ref: str,
    repo_root: Path,
    forbid_unverified_judge_metadata: bool = False,
) -> dict[str, JudgeCoverageReport]:
    """Diff every judge-covered allowlist under ``allowlist_root``.

    ``allowlist_root`` is typically ``config/cicd``; every
    ``enforce_*`` subdirectory whose YAML files carry judge-covered entry
    blocks is checked. ``allow_hits:`` entries are diffed against the
    baseline; newly-added ``per_file_rules:`` entries produce
    ``PER_FILE_RULE_REQUIRES_JUDGE`` violations. Non-empty legacy entry
    shapes produce ``UNRECOGNIZED_ENTRY_SHAPE`` violations only inside a
    directory already routed into this C1 surface.

    The returned mapping keys are the enforce-directory names (e.g.
    ``"enforce_tier_model"``). Callers aggregate the per-directory
    reports for the final pass/fail decision and the operator-facing
    summary.
    """
    if not allowlist_root.is_dir():
        raise JudgeCoverageError(f"--allowlist-root {allowlist_root} is not a directory")
    if not repo_root.is_dir():
        raise JudgeCoverageError(f"--repo-root {repo_root} is not a directory")

    if allowlist_root.name.startswith("enforce_"):
        candidate_dirs = [allowlist_root]
    else:
        candidate_dirs = sorted(
            entry_dir for entry_dir in allowlist_root.iterdir() if entry_dir.is_dir() and entry_dir.name.startswith("enforce_")
        )

    reports: dict[str, JudgeCoverageReport] = {}
    for entry_dir in candidate_dirs:
        if not _directory_has_allowlist_entries(entry_dir):
            continue
        reports[entry_dir.name] = check_one_directory(
            allowlist_dir=entry_dir,
            baseline_ref=baseline_ref,
            repo_root=repo_root,
            forbid_unverified_judge_metadata=forbid_unverified_judge_metadata,
        )
    return reports


def check_one_directory(
    *,
    allowlist_dir: Path,
    baseline_ref: str,
    repo_root: Path,
    forbid_unverified_judge_metadata: bool = False,
) -> JudgeCoverageReport:
    """Diff the allow_hits entries in one enforce_* directory."""
    head_entries, head_per_file_rules, shape_violations = _load_head_from_disk(allowlist_dir)
    baseline_entries, baseline_per_file_rules = _load_entries_from_git(
        allowlist_dir=allowlist_dir,
        baseline_ref=baseline_ref,
        repo_root=repo_root,
    )

    baseline_by_discriminator: dict[tuple[str, str, str, str, str, date | None], list[AllowlistEntry]] = {}
    for entry in baseline_entries:
        baseline_by_discriminator.setdefault(_discriminator(entry), []).append(entry)
    baseline_per_file_discriminators = {_per_file_rule_discriminator(entry) for entry in baseline_per_file_rules}

    # Pre-judge grandfathering is COUNT-LIMITED per discriminator. The fp=<hex>
    # suffix is stripped from the discriminator for rotation stability, so a
    # second unjudged suppression at the same (file, rule, symbol, owner, reason,
    # expires) — e.g. a second ``.get()`` in the same method — collides with the
    # baseline discriminator. Without a budget, both grandfather and the new
    # suppression lands with no judge run. We grandfather only as many pre-judge
    # HEAD entries per discriminator as there were pre-judge baseline entries: a
    # rotation (count unchanged) passes, but the EXCESS is a new suppression that
    # must carry judge metadata. Judged entries are matched by exact payload
    # (below) and are not budget-limited.
    baseline_pre_judge_remaining: dict[tuple[str, str, str, str, str, date | None], int] = {
        discriminator: sum(1 for entry in entries if _judge_metadata_payload(entry) is None)
        for discriminator, entries in baseline_by_discriminator.items()
    }

    violations: list[JudgeCoverageViolation] = list(shape_violations)
    new_count = 0
    grandfathered_count = 0
    for entry in head_entries:
        discriminator = _discriminator(entry)
        baseline_matches = baseline_by_discriminator.get(discriminator)
        if baseline_matches is not None:
            if not _judge_metadata_matches_any_baseline(entry, baseline_matches):
                if _is_fresh_judge_record_after_binding_drift(entry, baseline_matches):
                    new_count += 1
                    if forbid_unverified_judge_metadata:
                        violations.append(_unverified_judge_metadata_violation(entry))
                    continue
                violations.append(
                    JudgeCoverageViolation(
                        entry_key=entry.key,
                        source_file=entry.source_file,
                        missing_fields=(JUDGE_METADATA_MUTATED,),
                    )
                )
                continue
            head_is_pre_judge = _judge_metadata_payload(entry) is None
            if head_is_pre_judge and baseline_pre_judge_remaining.get(discriminator, 0) <= 0:
                # Excess unjudged entry beyond the rotation-stable baseline count:
                # a genuinely new suppression masquerading as a rotation. Require
                # judge metadata.
                new_count += 1
                missing = _missing_judge_fields(entry)
                if missing:
                    violations.append(
                        JudgeCoverageViolation(
                            entry_key=entry.key,
                            source_file=entry.source_file,
                            missing_fields=missing,
                        )
                    )
                continue
            if head_is_pre_judge:
                baseline_pre_judge_remaining[discriminator] -= 1
            grandfathered_count += 1
            continue
        new_count += 1
        missing = _missing_judge_fields(entry)
        if missing:
            violations.append(
                JudgeCoverageViolation(
                    entry_key=entry.key,
                    source_file=entry.source_file,
                    missing_fields=missing,
                )
            )
        elif forbid_unverified_judge_metadata and _judge_metadata_payload(entry) is not None:
            violations.append(_unverified_judge_metadata_violation(entry))
    for per_file_entry in head_per_file_rules:
        if _per_file_rule_discriminator(per_file_entry) in baseline_per_file_discriminators:
            grandfathered_count += 1
            continue
        new_count += 1
        violations.append(
            JudgeCoverageViolation(
                entry_key=per_file_entry.label,
                source_file=per_file_entry.source_file,
                missing_fields=(PER_FILE_RULE_REQUIRES_JUDGE,),
            )
        )

    return JudgeCoverageReport(
        head_entry_count=len(head_entries) + len(head_per_file_rules) + len(shape_violations),
        grandfathered_count=grandfathered_count,
        new_entry_count=new_count + len(shape_violations),
        violations=tuple(violations),
    )


# =========================================================================
# Discriminator + judge-field validation
# =========================================================================


def _discriminator(entry: AllowlistEntry) -> tuple[str, str, str, str, str, date | None]:
    """Return the rotation-stable identity tuple for ``entry``.

    Format:
    ``(file_path, rule_id, symbol_part, owner_norm, reason_norm, expires)``.
    The ``fp=<hex>`` segment of the YAML key is stripped — that is the
    fingerprint, the AST-position artefact rotation mutates. Two
    entries that match on this tuple are considered the same audit
    record across rotations.

    Owner and reason are whitespace-normalised so YAML formatting
    tweaks (line-wrap changes, trailing spaces) do not look like new
    entries. Normalisation collapses every run of whitespace
    (including newlines) to one space.

    ``expires`` is part of the identity (matching the ``per_file_rules``
    discriminator, which already includes it). A suppression renewed by
    editing only the expiry date is a fresh suppression decision and must
    re-acquire judge metadata, not silently grandfather — otherwise the
    bounded-expiry forcing function is defeatable with a one-character diff.
    """
    parts = entry.key.split(":")
    if parts and parts[-1].startswith("fp="):
        positional = parts[:-1]
    else:
        # Pre-fingerprint-era entry or malformed key; use whole key.
        positional = parts
    file_path = positional[0] if positional else entry.key
    rule_id = positional[1] if len(positional) > 1 else ""
    symbol_part = ":".join(positional[2:]) if len(positional) > 2 else ""
    owner_norm = _normalize_text(entry.owner)
    reason_norm = _normalize_text(entry.reason)
    _require_substantive_discriminator_anchor("owner", owner_norm, entry)
    _require_substantive_discriminator_anchor("reason", reason_norm, entry)
    return (
        file_path,
        rule_id,
        symbol_part,
        owner_norm,
        reason_norm,
        entry.expires,
    )


def _per_file_rule_discriminator(entry: PerFileRuleCoverageEntry) -> tuple[str, str, tuple[str, ...], str, str | None, int | None]:
    """Return the identity tuple for one per-file wildcard suppression."""
    return (
        entry.source_file,
        entry.pattern,
        entry.rules,
        _normalize_text(entry.reason),
        entry.expires,
        entry.max_hits,
    )


def _normalize_text(text: str) -> str:
    """Collapse whitespace runs (including newlines) to one space."""
    return " ".join(text.split())


def _judge_metadata_matches_any_baseline(entry: AllowlistEntry, baseline_entries: list[AllowlistEntry]) -> bool:
    """Return whether ``entry`` preserves the judge metadata for its baseline identity."""
    head_payload = _judge_metadata_payload(entry)
    return any(head_payload == _judge_metadata_payload(baseline_entry) for baseline_entry in baseline_entries)


def _is_fresh_judge_record_after_binding_drift(entry: AllowlistEntry, baseline_entries: list[AllowlistEntry]) -> bool:
    """Return whether ``entry`` is a newly-judged replacement for the same audit claim.

    A same-discriminator entry with changed judge metadata is normally a
    mutation. The legitimate exception is the documented source-drift lifecycle:
    the operator re-runs ``justify`` for the same owner/reason after the source
    binding changed, producing a complete fresh record with a different
    ``key`` / ``file_fingerprint`` / ``ast_path`` tuple. Treat that shape as a
    judged new record so operators can refresh stale bindings without inventing
    a different human rationale.
    """
    if _missing_judge_fields(entry):
        return False

    judged_baseline_entries = [baseline_entry for baseline_entry in baseline_entries if _judge_metadata_payload(baseline_entry) is not None]
    if not judged_baseline_entries:
        return False

    head_binding = _judge_binding_identity(entry)
    if any(baseline_entry.judge_metadata_signature == entry.judge_metadata_signature for baseline_entry in judged_baseline_entries):
        return False
    return _has_authoritative_judge_metadata_signature(entry) and all(
        _judge_binding_identity(baseline_entry) != head_binding for baseline_entry in judged_baseline_entries
    )


def _judge_binding_identity(entry: AllowlistEntry) -> tuple[str, str | None, str | None, str | None]:
    """Return source-binding fields that a fresh ``justify`` run may legitimately change."""
    return (entry.key, entry.file_fingerprint, entry.scope_fingerprint, entry.ast_path)


def _judge_metadata_payload(entry: AllowlistEntry) -> tuple[object, ...] | None:
    """Return the authenticity-bearing judge metadata for mutation comparison.

    ``None`` means "pre-judge entry": there is no judge metadata to protect,
    so rotation grandfathering remains keyed only by the discriminator.
    Once an entry carries any judge metadata, the full binding/signature cluster
    must remain byte-for-byte equivalent across the baseline identity.
    """
    if (
        entry.judge_verdict is None
        and entry.judge_recorded_at is None
        and entry.judge_model is None
        and entry.judge_policy_hash is None
        and entry.judge_rationale is None
        and entry.judge_metadata_signature is None
        and entry.scope_fingerprint is None
    ):
        return None
    return (
        entry.key,
        entry.file_fingerprint,
        entry.scope_fingerprint,
        entry.judge_signature_version,
        entry.ast_path,
        entry.judge_verdict,
        entry.judge_model_verdict,
        entry.judge_recorded_at,
        entry.judge_model,
        entry.judge_transport,
        entry.judge_policy_hash,
        entry.judge_confidence,
        entry.judge_rationale,
        entry.judge_excerpt_redactions,
        entry.judge_metadata_signature,
    )


def _require_substantive_discriminator_anchor(field: str, value: str, entry: AllowlistEntry) -> None:
    """Fail closed when C1 grandfathering would rely on a spoofable text anchor."""
    if sum(1 for char in value if char.isalnum()) >= 2:
        return
    raise JudgeCoverageError(
        f"{entry.source_file or '<memory>'}: allowlist entry {entry.key!r} has "
        f"non-substantive {field} value {value!r}; judge-coverage grandfathering "
        "relies on owner/reason as discriminator anchors and cannot safely compare "
        "entries whose anchors are empty or trivial."
    )


def _missing_judge_fields(entry: AllowlistEntry) -> tuple[str, ...]:
    """Return the names of judge-metadata fields absent from ``entry``.

    The atomic metadata cluster (``judge_verdict`` + ``judge_recorded_at`` +
    ``judge_model`` + ``judge_policy_hash`` + ``judge_rationale`` +
    ``judge_metadata_signature``) is the C1 contract. The optional override field
    ``judge_model_verdict`` is required only when ``judge_verdict ==
    OVERRIDDEN_BY_OPERATOR``; that invariant is already enforced at load time by
    ``_validate_judge_metadata_atomic`` (so a partially-filled override entry
    would have crashed during HEAD parsing and never reach this check). C1
    therefore validates only the common signed metadata cluster.

    A return value of ``()`` means the entry satisfies C1.
    """
    missing: list[str] = []
    if entry.judge_verdict is None:
        missing.append("judge_verdict")
    if entry.judge_recorded_at is None:
        missing.append("judge_recorded_at")
    if entry.judge_model is None:
        missing.append("judge_model")
    if entry.judge_policy_hash is None:
        missing.append("judge_policy_hash")
    if entry.judge_rationale is None:
        missing.append("judge_rationale")
    if entry.judge_metadata_signature is None:
        missing.append("judge_metadata_signature")
    return tuple(missing)


def _has_authoritative_judge_metadata_signature(entry: AllowlistEntry) -> bool:
    """Return whether ``entry`` has a real HMAC, not only signature-shaped text."""
    try:
        _verify_judge_metadata_signature_at_load(
            entry,
            context=f"judge-coverage {entry.source_file}:{entry.key}",
            allow_shape_only=False,
        )
    except ValueError:
        return False
    return True


def _unverified_judge_metadata_violation(entry: AllowlistEntry) -> JudgeCoverageViolation:
    """Return the keyless-fork violation for a new signed allowlist entry."""
    return JudgeCoverageViolation(
        entry_key=entry.key,
        source_file=entry.source_file,
        missing_fields=(UNVERIFIED_JUDGE_METADATA_WITHOUT_HMAC,),
    )


# =========================================================================
# Filesystem + git plumbing
# =========================================================================


def _directory_has_allowlist_entries(directory: Path) -> bool:
    """Structural check: does any YAML file carry a judge-covered entry key?

    Avoids the cost of full parsing downstream for directories that
    contain only defaults or non-judge custom allowlist formats.

    **Fail-closed (C7-4a):** an ``OSError`` while reading a YAML file
    is NOT silently skipped. The gate cannot distinguish "directory
    has no allowlist entries" from "directory has unreadable allowlist
    entries"; the second case is the silent-failure mode the gate exists
    to prevent. We raise ``JudgeCoverageError`` so the CLI surfaces
    exit-2 (gate-broken) rather than exit-0 (gate-passed).

    **Fail-closed (C7-4b):** the routing check used to be a substring
    match for ``"\\nallow_hits:"``. That missed CRLF line endings
    (``"\\r\\nallow_hits:"``) and ``allow_hits:`` at start-of-file
    after a UTF-8 BOM. A missed routing decision silently excludes
    the whole directory from the gate. Replaced with a YAML parse
    and a structural top-level-key check over all known entry shapes.

    A YAML parse failure here is *also* a fail-closed condition:
    HEAD content is our data and corruption cannot be silently
    routed away from the gate. We raise ``JudgeCoverageError``.
    """
    try:
        documents = iter_yaml_documents(directory)
    except AllowlistIOError as exc:
        raise JudgeCoverageError(f"while routing for allowlist entries: {exc}") from exc
    for document in documents:
        try:
            if any(entry_shape_count(document.data, key, source_file=document.source_file) > 0 for key in _JUDGE_COVERED_ENTRY_KEYS):
                return True
        except AllowlistIOError as exc:
            raise JudgeCoverageError(str(exc)) from exc
    return False


def _load_head_from_disk(allowlist_dir: Path) -> tuple[list[AllowlistEntry], list[PerFileRuleCoverageEntry], list[JudgeCoverageViolation]]:
    """Parse ``allow_hits`` and ``per_file_rules`` entries from HEAD.

    Bypasses ``load_allowlist`` to avoid the rule-specific
    ``valid_rule_ids`` coupling: C1 is rule-agnostic at the
    per-entry level, and the directory may carry sibling
    ``per_file_rules`` whose validation requires a rule-aware
    vocabulary. Reading ``allow_hits`` directly sidesteps that.

    A YAML file that fails to parse propagates as a
    ``JudgeCoverageError`` (HEAD content is our data — bad shape is
    corruption and must crash, not silently skip).
    """
    return _iterate_head_entries_from_directory(allowlist_dir)


def _iterate_head_entries_from_directory(
    directory: Path,
) -> tuple[list[AllowlistEntry], list[PerFileRuleCoverageEntry], list[JudgeCoverageViolation]]:
    entries: list[AllowlistEntry] = []
    per_file_rules: list[PerFileRuleCoverageEntry] = []
    shape_violations: list[JudgeCoverageViolation] = []
    try:
        documents = iter_yaml_documents(directory)
    except AllowlistIOError as exc:
        raise JudgeCoverageError(str(exc)) from exc
    for document in documents:
        # source_root=None: judge_coverage audits the *aggregate* judge-
        # gated-fraction of persisted entries; it does not have access to
        # the source tree and would not benefit from per-entry binding
        # verification. Co-presence invariants still fire from
        # _validate_judge_metadata_atomic.
        try:
            entries.extend(parse_allow_hits(document.data, source_file=document.source_file))
            per_file_rules.extend(_parse_per_file_rules_for_coverage(document.data, source_file=document.source_file))
        except AllowlistIOError as exc:
            raise JudgeCoverageError(str(exc)) from exc
        shape_violations.extend(_unrecognized_shape_violations(document.data, source_file=document.source_file))
    return entries, per_file_rules, shape_violations


def _unrecognized_shape_violations(data: dict[str, Any], *, source_file: str) -> list[JudgeCoverageViolation]:
    """Return violations for non-empty legacy entry shapes in one YAML mapping."""
    violations: list[JudgeCoverageViolation] = []
    for shape_key in sorted(_ALLOWLIST_ENTRY_KEYS - {"allow_hits", "per_file_rules"}):
        raw_entries = data.get(shape_key, [])
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise JudgeCoverageError(f"{source_file}: {shape_key} must be a list if present")
        for index, raw_entry in enumerate(raw_entries):
            violations.append(
                JudgeCoverageViolation(
                    entry_key=_entry_shape_label(shape_key, index, raw_entry),
                    source_file=source_file,
                    missing_fields=(UNRECOGNIZED_ENTRY_SHAPE,),
                )
            )
    return violations


def _parse_per_file_rules_for_coverage(data: dict[str, Any], *, source_file: str) -> list[PerFileRuleCoverageEntry]:
    """Parse ``per_file_rules`` enough for judge-coverage diffing."""
    raw_entries = data.get("per_file_rules", [])
    if raw_entries is None:
        return []
    if not isinstance(raw_entries, list):
        raise AllowlistIOError(f"{source_file}: per_file_rules must be a list if present")

    entries: list[PerFileRuleCoverageEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        context = f"per_file_rules[{index}]"
        if not isinstance(raw_entry, dict):
            raise AllowlistIOError(f"{source_file}: {context} must be a mapping")
        pattern = _required_coverage_string(raw_entry, "pattern", context=context, source_file=source_file)
        rules = tuple(_required_coverage_string_list(raw_entry, "rules", context=context, source_file=source_file))
        reason = _required_coverage_string(raw_entry, "reason", context=context, source_file=source_file)
        expires = _optional_coverage_date(raw_entry, "expires", context=context, source_file=source_file)
        max_hits = _optional_coverage_int(raw_entry, "max_hits", context=context, source_file=source_file)
        entries.append(
            PerFileRuleCoverageEntry(
                source_file=source_file,
                index=index,
                pattern=pattern,
                rules=rules,
                reason=reason,
                expires=expires,
                max_hits=max_hits,
            )
        )
    return entries


def _required_coverage_string(data: dict[str, Any], key: str, *, context: str, source_file: str) -> str:
    if key not in data:
        raise AllowlistIOError(f"{source_file}: {context} must include {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise AllowlistIOError(f"{source_file}: {context}.{key} must be a non-empty string")
    return value


def _required_coverage_string_list(data: dict[str, Any], key: str, *, context: str, source_file: str) -> list[str]:
    raw_values = data.get(key, [])
    if not isinstance(raw_values, list):
        raise AllowlistIOError(f"{source_file}: {context}.{key} must be a list")
    values: list[str] = []
    for index, raw_value in enumerate(raw_values):
        if not isinstance(raw_value, str) or not raw_value:
            raise AllowlistIOError(f"{source_file}: {context}.{key}[{index}] must be a non-empty string")
        values.append(raw_value)
    return values


def _optional_coverage_date(data: dict[str, Any], key: str, *, context: str, source_file: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and value:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise AllowlistIOError(f"{source_file}: {context}.{key} must be YYYY-MM-DD, null, or absent") from exc
        return value
    raise AllowlistIOError(f"{source_file}: {context}.{key} must be YYYY-MM-DD, null, or absent")


def _optional_coverage_int(data: dict[str, Any], key: str, *, context: str, source_file: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool):
        raise AllowlistIOError(f"{source_file}: {context}.{key} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    raise AllowlistIOError(f"{source_file}: {context}.{key} must be an integer, null, or absent")


def _per_file_rule_label(index: int, pattern: str, rules: tuple[str, ...]) -> str:
    return f"per_file_rules[{index}]::pattern={pattern}::rules={','.join(rules)}"


def _entry_shape_label(shape_key: str, index: int, raw_entry: Any) -> str:
    if isinstance(raw_entry, dict):
        key = raw_entry.get("key")
        if isinstance(key, str) and key:
            return f"{shape_key}[{index}]::{key}"
        commit_sha = raw_entry.get("commit_sha")
        if isinstance(commit_sha, str) and commit_sha:
            return f"{shape_key}[{index}]::commit_sha={commit_sha}"
    return f"{shape_key}[{index}]"


def _load_entries_from_git(
    *,
    allowlist_dir: Path,
    baseline_ref: str,
    repo_root: Path,
) -> tuple[list[AllowlistEntry], list[PerFileRuleCoverageEntry]]:
    """Materialise the baseline allowlist files and parse their ``allow_hits``.

    Uses ``git ls-tree`` to enumerate YAML files in the baseline tree
    under ``allowlist_dir``, ``git show <ref>:<path>`` to read each.
    Files absent from baseline (directory or file added in this PR)
    produce no baseline entries — every entry in such a file is
    treated as new and must be judged.

    **Fail-closed (C7-4c).** A baseline file that fails to parse used
    to be silently treated as contributing zero baseline entries.
    That swaps the entire grandfathering signal for "every HEAD
    entry is new" — sounds conservative, but in practice the
    operator sees a flood of "missing judge metadata" violations
    that hides the actual problem (baseline corruption) and creates
    enormous pressure to slap judge stubs onto entries that were
    already grandfathered. We raise ``JudgeCoverageError`` instead,
    so the CLI emits exit-2 ("gate broken — surface to operator")
    with the offending baseline path and parse diagnostic. The
    operator fixes the baseline (or the ref), not the symptoms.

    Same discipline applies to ``_parse_allow_hits`` invariant
    violations: a baseline whose entry shape no longer parses under
    the current schema is a structural anomaly that the operator
    must see, not silently route around.
    """
    rel_dir = _relative_to_repo(allowlist_dir, repo_root)
    file_names = _ls_tree_yaml_files(
        baseline_ref=baseline_ref,
        rel_dir=rel_dir,
        repo_root=repo_root,
    )

    entries: list[AllowlistEntry] = []
    per_file_rules: list[PerFileRuleCoverageEntry] = []
    for rel_path in file_names:
        if Path(rel_path).name == "_defaults.yaml":
            continue
        if not rel_path.endswith(".yaml"):
            continue
        content = _git_show(
            baseline_ref=baseline_ref,
            rel_path=rel_path,
            repo_root=repo_root,
        )
        if content is None:
            continue
        # C7-4c + C7-5: catch both ``yaml.YAMLError`` and ``ValueError``,
        # and raise rather than silently dropping the baseline entries
        # for this file. A silent empty baseline destroys grandfathering.
        try:
            data = load_yaml_mapping_text(content, source_label=f"baseline {baseline_ref}:{rel_path}")
        except AllowlistIOError as exc:
            raise JudgeCoverageError(str(exc)) from exc
        try:
            # source_root=None: baseline entries come from a historical
            # git ref where the source tree at that ref isn't on disk —
            # binding verification is meaningless here.
            entries.extend(parse_allow_hits(data, source_file=Path(rel_path).name))
            per_file_rules.extend(_parse_per_file_rules_for_coverage(data, source_file=Path(rel_path).name))
        except AllowlistIOError as exc:
            raise JudgeCoverageError(f"baseline {baseline_ref}:{rel_path}: {exc}") from exc
    return entries, per_file_rules


def _relative_to_repo(allowlist_dir: Path, repo_root: Path) -> str:
    try:
        return str(allowlist_dir.resolve().relative_to(repo_root.resolve()))
    except ValueError as exc:
        raise JudgeCoverageError(f"{allowlist_dir} is not inside repo root {repo_root}") from exc


def _ls_tree_yaml_files(
    *,
    baseline_ref: str,
    rel_dir: str,
    repo_root: Path,
) -> list[str]:
    """Return baseline YAML file paths under ``rel_dir`` (or ``[]``)."""
    result = _run_git(["ls-tree", "-r", "--name-only", baseline_ref, "--", rel_dir], repo_root=repo_root)
    if result.returncode != 0:
        raise JudgeCoverageError(f"git ls-tree could not inspect baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")
    return [line for line in result.stdout.splitlines() if line]


def _git_show(
    *,
    baseline_ref: str,
    rel_path: str,
    repo_root: Path,
) -> str | None:
    """Return file content at ``baseline_ref`` or ``None`` if not in tree."""
    result = _run_git(["show", f"{baseline_ref}:{rel_path}"], repo_root=repo_root)
    if result.returncode == 0:
        return result.stdout

    if not _git_commit_exists(baseline_ref=baseline_ref, repo_root=repo_root):
        raise JudgeCoverageError(f"git show could not resolve baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")
    if not _git_path_exists(baseline_ref=baseline_ref, rel_path=rel_path, repo_root=repo_root):
        return None
    raise JudgeCoverageError(f"git show failed for baseline {baseline_ref}:{rel_path}: {_git_failure_detail(result)}")


def _run_git(args: list[str], *, repo_root: Path) -> subprocess.CompletedProcess[str]:
    """Run git with a stable C locale so diagnostics are not localized."""
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
        raise JudgeCoverageError(f"git command failed to start: {exc}") from exc


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
