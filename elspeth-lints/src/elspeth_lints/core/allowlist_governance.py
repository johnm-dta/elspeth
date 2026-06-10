"""Convert shared allowlist governance state into lint findings."""

from __future__ import annotations

import hashlib
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, AllowlistBudgetViolation, AllowlistEntry, PerFileRule
from elspeth_lints.core.protocols import Finding, Severity

RULE_STALE_ENTRY = "allowlist.stale_entry"
RULE_UNUSED_RULE = "allowlist.unused_rule"
RULE_EXPIRED_ENTRY = "allowlist.expired_entry"
RULE_EXPIRED_RULE = "allowlist.expired_rule"
RULE_MAX_HITS_EXCEEDED = "allowlist.max_hits_exceeded"
RULE_BUDGET_EXCEEDED = "allowlist.budget_exceeded"


def allowlist_governance_findings(
    allowlist: Allowlist,
    allowlist_dir: Path,
    *,
    emitted_dirs: set[str] | None = None,
    enabled: bool = True,
) -> list[Finding]:
    """Return lint findings for shared allowlist governance failures.

    The shared allowlist loader tracks match counts and exposes stale, expired,
    max-hit, and budget state. Rule implementations call this after applying
    suppressions so CI sees policy debt even when all code findings were
    successfully suppressed.
    """
    if not enabled:
        return []

    if emitted_dirs is not None:
        key = allowlist_dir.resolve().as_posix()
        if key in emitted_dirs:
            return []
        emitted_dirs.add(key)

    findings: list[Finding] = []
    if allowlist.fail_on_stale:
        findings.extend(_stale_entry_finding(entry, allowlist_dir) for entry in allowlist.get_unused_entries())
        findings.extend(_unused_rule_finding(rule, allowlist_dir) for rule in allowlist.get_unused_rules())
    if allowlist.fail_on_expired:
        findings.extend(_expired_entry_finding(entry, allowlist_dir) for entry in allowlist.get_expired_entries())
        findings.extend(_expired_rule_finding(rule, allowlist_dir) for rule in allowlist.get_expired_rules())
    findings.extend(_max_hits_finding(rule, allowlist_dir) for rule in allowlist.get_exceeded_rules())
    findings.extend(_budget_finding(violation, allowlist_dir) for violation in allowlist.get_budget_violations())
    return findings


def allowlist_governance_findings_for_root(
    allowlist: Allowlist,
    allowlist_dir: Path,
    *,
    root: Path,
    allowlist_dir_override: Path | None,
    emitted_dirs: set[str] | None = None,
    enabled: bool = True,
) -> list[Finding]:
    """Return governance findings when the allowlist belongs to this scan.

    Several legacy allowlist resolvers fall back to ``Path.cwd()``. That is
    correct for real repository scans such as ``--root src/elspeth``, but unit
    tests often scan temporary fixture roots outside the repository and should
    not inherit governance failures from the live checkout's allowlist.
    """
    if not enabled or not _governance_applies_to_root(root, allowlist_dir, allowlist_dir_override=allowlist_dir_override):
        return []
    return allowlist_governance_findings(allowlist, allowlist_dir, emitted_dirs=emitted_dirs)


def _governance_applies_to_root(
    root: Path,
    allowlist_dir: Path,
    *,
    allowlist_dir_override: Path | None,
) -> bool:
    if allowlist_dir_override is not None:
        return True
    resolved_root = root.resolve()
    resolved_allowlist_dir = allowlist_dir.resolve()
    if _is_relative_to(resolved_allowlist_dir, resolved_root):
        return True
    return _is_relative_to(resolved_root, Path.cwd().resolve())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _stale_entry_finding(entry: AllowlistEntry, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_STALE_ENTRY,
        file_path=_source_file(allowlist_dir, entry.source_file),
        message=f"Shared allowlist entry no longer matches any finding: {entry.key}",
        fingerprint_payload=f"{RULE_STALE_ENTRY}|{entry.source_file}|{entry.key}",
        suggestion="Remove the stale allow_hits entry or rotate it through the rule's reviewed process.",
    )


def _unused_rule_finding(rule: PerFileRule, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_UNUSED_RULE,
        file_path=_source_file(allowlist_dir, rule.source_file),
        message=f"Shared allowlist per-file rule no longer matches any finding: {rule.pattern} ({', '.join(rule.rules)})",
        fingerprint_payload=f"{RULE_UNUSED_RULE}|{rule.source_file}|{rule.pattern}|{','.join(rule.rules)}",
        suggestion="Remove the stale per_file_rules entry or update it through the rule's reviewed process.",
    )


def _expired_entry_finding(entry: AllowlistEntry, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_EXPIRED_ENTRY,
        file_path=_source_file(allowlist_dir, entry.source_file),
        message=f"Shared allowlist entry expired on {entry.expires}: {entry.key}",
        fingerprint_payload=f"{RULE_EXPIRED_ENTRY}|{entry.source_file}|{entry.key}|{entry.expires}",
        suggestion="Remove the expired allow_hits entry or renew it through the rule's reviewed process.",
    )


def _expired_rule_finding(rule: PerFileRule, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_EXPIRED_RULE,
        file_path=_source_file(allowlist_dir, rule.source_file),
        message=f"Shared allowlist per-file rule expired on {rule.expires}: {rule.pattern} ({', '.join(rule.rules)})",
        fingerprint_payload=f"{RULE_EXPIRED_RULE}|{rule.source_file}|{rule.pattern}|{','.join(rule.rules)}|{rule.expires}",
        suggestion="Remove the expired per_file_rules entry or renew it through the rule's reviewed process.",
    )


def _max_hits_finding(rule: PerFileRule, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_MAX_HITS_EXCEEDED,
        file_path=_source_file(allowlist_dir, rule.source_file),
        message=f"Shared allowlist per-file rule exceeded max_hits: {rule.pattern} matched {rule.matched_count}/{rule.max_hits}",
        fingerprint_payload=f"{RULE_MAX_HITS_EXCEEDED}|{rule.source_file}|{rule.pattern}|{','.join(rule.rules)}|{rule.max_hits}",
        suggestion="Review the new matches and either fix them, split the rule, or deliberately raise max_hits.",
    )


def _budget_finding(violation: AllowlistBudgetViolation, allowlist_dir: Path) -> Finding:
    return _finding(
        rule_id=RULE_BUDGET_EXCEEDED,
        file_path=_source_file(allowlist_dir, "_defaults.yaml"),
        message=f"Shared allowlist budget exceeded: {violation.category} is {violation.current}/{violation.max_allowed}",
        fingerprint_payload=f"{RULE_BUDGET_EXCEEDED}|{violation.category}|{violation.max_allowed}",
        suggestion="Delete stale entries or update the allowlist budget deliberately.",
    )


def _finding(
    *,
    rule_id: str,
    file_path: str,
    message: str,
    fingerprint_payload: str,
    suggestion: str,
) -> Finding:
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line=1,
        column=0,
        message=message,
        fingerprint=_fingerprint(fingerprint_payload),
        severity=Severity.ERROR,
        suggestion=suggestion,
    )


def _source_file(allowlist_dir: Path, source_file: str) -> str:
    if not source_file:
        return allowlist_dir.as_posix()
    return (allowlist_dir / source_file).as_posix()


def _fingerprint(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
