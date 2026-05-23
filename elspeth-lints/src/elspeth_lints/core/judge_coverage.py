"""New-entry judge-coverage CI gate (convergent finding C1).

Convergent panel finding C1: judge enforcement is voluntary, not
architectural. The judge primitive ships in ``elspeth-lints justify``
and is recorded in entry metadata, but nothing in CI mechanically
rejects a hand-edited allowlist entry that lacks the judge fields.
Pre-judge and judge-skipped entries are observationally identical to
the loader (both have ``judge_verdict is None``), so an agent who
appends an entry by editing YAML directly produces an honest-looking
"pre-judge" entry that erodes the audit trail.

This module closes that loop. On a PR diff against the merge base,
every new ``allow_hits`` entry MUST carry the atomic judge quartet
(``judge_verdict + judge_recorded_at + judge_model + judge_rationale``).
Entries already present in baseline are grandfathered — including
true pre-judge entries that pre-date the gate. The grandfathering is
*rotation-stable*: an entry whose fingerprint shifted because of an
upstream AST refactor still matches its baseline counterpart, so
refactors are not penalised by demanding fresh judge runs.

**Rotation policy (operator-confirmed 2026-05-23):** rotation
grandfathers. The discriminator is ``(file_path, rule_id,
symbol_part, owner, reason)`` — the fingerprint ``fp=<hex>`` segment
is stripped because it is the AST-position artefact rotation
mutates. ``owner`` and ``reason`` discriminate between two distinct
entries that happen to share the parsed-key triple (an unusual but
legal shape that the triple alone cannot disambiguate).
``reaudit`` remains the surface for periodic re-judging.

Boundary discipline: this module reads the standard
``allow_hits:`` YAML shape parsed by
``elspeth_lints.core.allowlist._parse_allow_hits``. It does NOT
read the private legacy shapes used by some rules
(``allow_classes:`` for ``audit_evidence.nominal_base``,
``entries:`` for ``plugin_contract.options_metadata``,
``entries: - commit_sha:`` for
``enforce_telemetry_backfill_trailer``). Those shapes do not carry
``judge_*`` fields by construction; gating them is out of scope and
is a separate decision (file a wardline ticket if the migration
becomes desirable).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from elspeth_lints.core.allowlist import (
    AllowlistEntry,
    _parse_allow_hits,
)


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
) -> dict[str, JudgeCoverageReport]:
    """Diff every ``allow_hits`` allowlist under ``allowlist_root``.

    ``allowlist_root`` is typically ``config/cicd``; every
    ``enforce_*`` subdirectory whose YAML files carry
    ``allow_hits:`` blocks is checked. Directories that use the
    private legacy formats (``allow_classes:``, custom ``entries:``)
    are silently skipped — see module docstring for the rationale.

    The returned mapping keys are the enforce-directory names (e.g.
    ``"enforce_tier_model"``). Callers aggregate the per-directory
    reports for the final pass/fail decision and the operator-facing
    summary.
    """
    if not allowlist_root.is_dir():
        raise JudgeCoverageError(f"--allowlist-root {allowlist_root} is not a directory")
    if not repo_root.is_dir():
        raise JudgeCoverageError(f"--repo-root {repo_root} is not a directory")

    reports: dict[str, JudgeCoverageReport] = {}
    for entry_dir in sorted(allowlist_root.iterdir()):
        if not entry_dir.is_dir():
            continue
        if not entry_dir.name.startswith("enforce_"):
            continue
        if not _directory_has_allow_hits(entry_dir):
            continue
        reports[entry_dir.name] = check_one_directory(
            allowlist_dir=entry_dir,
            baseline_ref=baseline_ref,
            repo_root=repo_root,
        )
    return reports


def check_one_directory(
    *,
    allowlist_dir: Path,
    baseline_ref: str,
    repo_root: Path,
) -> JudgeCoverageReport:
    """Diff the allow_hits entries in one enforce_* directory."""
    head_entries = _load_entries_from_disk(allowlist_dir)
    baseline_entries = _load_entries_from_git(
        allowlist_dir=allowlist_dir,
        baseline_ref=baseline_ref,
        repo_root=repo_root,
    )

    baseline_discriminators = {_discriminator(entry) for entry in baseline_entries}

    violations: list[JudgeCoverageViolation] = []
    new_count = 0
    grandfathered_count = 0
    for entry in head_entries:
        if _discriminator(entry) in baseline_discriminators:
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

    return JudgeCoverageReport(
        head_entry_count=len(head_entries),
        grandfathered_count=grandfathered_count,
        new_entry_count=new_count,
        violations=tuple(violations),
    )


# =========================================================================
# Discriminator + judge-field validation
# =========================================================================


def _discriminator(entry: AllowlistEntry) -> tuple[str, str, str, str, str]:
    """Return the rotation-stable identity tuple for ``entry``.

    Format: ``(file_path, rule_id, symbol_part, owner_norm, reason_norm)``.
    The ``fp=<hex>`` segment of the YAML key is stripped — that is the
    fingerprint, the AST-position artefact rotation mutates. Two
    entries that match on this tuple are considered the same audit
    record across rotations.

    Owner and reason are whitespace-normalised so YAML formatting
    tweaks (line-wrap changes, trailing spaces) do not look like new
    entries. Normalisation collapses every run of whitespace
    (including newlines) to one space.
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
    return (
        file_path,
        rule_id,
        symbol_part,
        _normalize_text(entry.owner),
        _normalize_text(entry.reason),
    )


def _normalize_text(text: str) -> str:
    """Collapse whitespace runs (including newlines) to one space."""
    return " ".join(text.split())


def _missing_judge_fields(entry: AllowlistEntry) -> tuple[str, ...]:
    """Return the names of judge-metadata fields absent from ``entry``.

    The atomic quartet (``judge_verdict`` + ``judge_recorded_at`` +
    ``judge_model`` + ``judge_rationale``) is the C1 contract. The
    optional fifth field ``judge_model_verdict`` is required only
    when ``judge_verdict == OVERRIDDEN_BY_OPERATOR``; that invariant
    is already enforced at load time by
    ``_validate_judge_metadata_atomic`` (so a partially-filled
    override entry would have crashed during HEAD parsing and never
    reach this check). C1 therefore validates only the quartet.

    A return value of ``()`` means the entry satisfies C1.
    """
    missing: list[str] = []
    if entry.judge_verdict is None:
        missing.append("judge_verdict")
    if entry.judge_recorded_at is None:
        missing.append("judge_recorded_at")
    if entry.judge_model is None:
        missing.append("judge_model")
    if entry.judge_rationale is None:
        missing.append("judge_rationale")
    return tuple(missing)


# =========================================================================
# Filesystem + git plumbing
# =========================================================================


def _directory_has_allow_hits(directory: Path) -> bool:
    """Structural check: does ANY YAML file here carry a top-level ``allow_hits`` key?

    Avoids the cost of full parsing downstream for directories that
    exclusively use other shapes (``allow_classes:``,
    per_file_rules-only).

    **Fail-closed (C7-4a):** an ``OSError`` while reading a YAML file
    is NOT silently skipped. The gate cannot distinguish "directory
    has no allow_hits" from "directory has unreadable allow_hits";
    the second case is the silent-failure mode the gate exists to
    prevent. We raise ``JudgeCoverageError`` so the CLI surfaces
    exit-2 (gate-broken) rather than exit-0 (gate-passed).

    **Fail-closed (C7-4b):** the routing check used to be a substring
    match for ``"\\nallow_hits:"``. That missed CRLF line endings
    (``"\\r\\nallow_hits:"``) and ``allow_hits:`` at start-of-file
    after a UTF-8 BOM. A missed routing decision silently excludes
    the whole directory from the gate. Replaced with a YAML parse
    and a structural top-level-key check.

    A YAML parse failure here is *also* a fail-closed condition:
    HEAD content is our data and corruption cannot be silently
    routed away from the gate. We raise ``JudgeCoverageError``.
    """
    for yaml_file in directory.glob("*.yaml"):
        if yaml_file.name == "_defaults.yaml":
            continue
        try:
            text = yaml_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise JudgeCoverageError(f"could not read {yaml_file} while routing for allow_hits: {exc}") from exc
        try:
            data = _load_yaml_strict(text)
        except (yaml.YAMLError, ValueError) as exc:
            raise JudgeCoverageError(f"{yaml_file}: failed to parse as YAML mapping while routing for allow_hits: {exc}") from exc
        if "allow_hits" in data:
            return True
    return False


def _load_entries_from_disk(allowlist_dir: Path) -> list[AllowlistEntry]:
    """Parse every ``allow_hits`` entry in ``allowlist_dir``.

    Bypasses ``load_allowlist`` to avoid the rule-specific
    ``valid_rule_ids`` coupling: C1 is rule-agnostic at the
    per-entry level, and the directory may carry sibling
    ``per_file_rules`` whose validation requires a rule-aware
    vocabulary. Reading ``allow_hits`` directly sidesteps that.

    A YAML file that fails to parse propagates as a
    ``JudgeCoverageError`` (HEAD content is our data — bad shape is
    corruption and must crash, not silently skip).
    """
    return list(_iterate_entries_from_directory(allowlist_dir))


def _iterate_entries_from_directory(directory: Path) -> list[AllowlistEntry]:
    entries: list[AllowlistEntry] = []
    for yaml_file in sorted(directory.glob("*.yaml")):
        if yaml_file.name == "_defaults.yaml":
            continue
        # C7-5: ``yaml.safe_load`` raises ``yaml.YAMLError`` (a sibling
        # of ``Exception``, not a ``ValueError`` subclass). Catching
        # only ``ValueError`` let malformed YAML escape as an uncaught
        # traceback (exit 1, "gate broken"), masquerading as the
        # documented exit-2 ("gate broken — surface to operator") via
        # different output shape. We catch both so the CLI gets a
        # ``JudgeCoverageError`` and emits a clean exit-2 with a
        # structured message.
        try:
            data = _load_yaml_strict(yaml_file.read_text(encoding="utf-8"))
        except (yaml.YAMLError, ValueError) as exc:
            raise JudgeCoverageError(f"{yaml_file}: failed to parse as YAML mapping: {exc}") from exc
        entries.extend(_parse_allow_hits(data, source_file=yaml_file.name))
    return entries


def _load_entries_from_git(
    *,
    allowlist_dir: Path,
    baseline_ref: str,
    repo_root: Path,
) -> list[AllowlistEntry]:
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
            data = _load_yaml_strict(content)
        except (yaml.YAMLError, ValueError) as exc:
            raise JudgeCoverageError(f"baseline {baseline_ref}:{rel_path}: failed to parse as YAML mapping: {exc}") from exc
        try:
            entries.extend(_parse_allow_hits(data, source_file=Path(rel_path).name))
        except (ValueError, TypeError) as exc:
            raise JudgeCoverageError(
                f"baseline {baseline_ref}:{rel_path}: allow_hits entry shape violated loader invariants: {exc}"
            ) from exc
    return entries


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
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", baseline_ref, "--", rel_dir],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        # Baseline ref invalid OR directory absent at baseline.
        # Distinguish: invalid ref is a CI input error and must
        # surface as an error; absent directory is legitimate "new
        # directory in this PR" and yields empty baseline.
        #
        # git's stderr taxonomy is unstable across versions; the
        # robust discriminator is "fatal: ..." which git emits for
        # ref-resolution failures but not for "path absent from
        # tree" (the latter produces returncode 0 with empty stdout
        # in modern git, or a non-fatal stderr in older versions).
        stderr = (result.stderr or "").strip()
        if stderr.lower().startswith("fatal:"):
            raise JudgeCoverageError(f"git ls-tree could not resolve baseline-ref {baseline_ref!r}: {stderr}")
        return []
    return [line for line in result.stdout.splitlines() if line]


def _git_show(
    *,
    baseline_ref: str,
    rel_path: str,
    repo_root: Path,
) -> str | None:
    """Return file content at ``baseline_ref`` or ``None`` if not in tree."""
    result = subprocess.run(
        ["git", "show", f"{baseline_ref}:{rel_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _load_yaml_strict(text: str) -> dict[str, Any]:
    """Parse a YAML string into a mapping; reject non-mapping shapes.

    Mirrors ``elspeth_lints.core.allowlist._load_yaml_file`` but takes
    text rather than a path, so it composes with ``git show``.
    """
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML root must be a mapping, got {type(raw).__name__}")
    return raw
