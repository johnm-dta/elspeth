"""Allowlist loading and matching for elspeth-lints."""

from __future__ import annotations

import fnmatch
import hashlib
import hmac
import json
import os
import sys
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import yaml

from elspeth_lints.core.source_excerpt import RedactionRecord

_JUDGE_METADATA_SIGNATURE_ENV_VAR = "ELSPETH_JUDGE_METADATA_HMAC_KEY"
_JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR = "ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE"
_JUDGE_METADATA_SIGNATURE_VERIFY_REQUIRED = "required"
_JUDGE_METADATA_SIGNATURE_VERIFY_SHAPE_ONLY_WHEN_KEY_MISSING = "shape-only-when-key-missing"
_JUDGE_METADATA_SIGNATURE_PREFIX_V1 = "hmac-sha256:v1:"
_JUDGE_METADATA_SIGNATURE_PREFIX_V2 = "hmac-sha256:v2:"
JUDGE_METADATA_SIGNATURE_PREFIXES = (_JUDGE_METADATA_SIGNATURE_PREFIX_V1, _JUDGE_METADATA_SIGNATURE_PREFIX_V2)
_MIN_JUDGE_METADATA_HMAC_KEY_BYTES = 32
_MAX_ALLOWLIST_YAML_BYTES = 5 * 1024 * 1024
_MIN_AUDIT_ANCHOR_ALNUM_CHARS = 2


class DanglingAllowlistEntry(ValueError):
    """Raised when a judge-gated allow_hits entry's bound source file no longer exists.

    The entry is inert (the file that contained the violation it exempted is
    gone), so callers may skip it with a warning rather than crashing. This
    sentinel is distinct from plain ``ValueError`` to ensure that the separate
    fingerprint-mismatch crash path (which signals tampering, not staleness) is
    never accidentally swallowed by a broad ``except ValueError`` handler.
    """


class JudgeVerdict(StrEnum):
    """The verdict an allowlist entry carries from the cicd-judge.

    Members:
        ACCEPTED: The judge read the agent's rationale and the
            surrounding code and approved the suppression.
        BLOCKED: The judge rejected the rationale. Entries with this
            verdict should NOT exist in the allowlist — the CLI declines
            to write them. Reserved for in-memory representation only.
        OVERRIDDEN_BY_OPERATOR: A human operator chose to bypass the
            judge's verdict. This is distinct from ACCEPTED and is the
            audit signal that a human used their authority to override
            the gate. The rotation rate of this verdict is itself a
            meta-metric.

    Entries written before the judge existed carry ``judge_verdict=None``
    on the dataclass — that absence is the honest representation of
    "pre-judge era" rather than a fabricated default verdict.
    """

    ACCEPTED = "ACCEPTED"
    BLOCKED = "BLOCKED"
    OVERRIDDEN_BY_OPERATOR = "OVERRIDDEN_BY_OPERATOR"


class AuditReviewVerdict(StrEnum):
    """Human post-review verdicts for an already persisted judge decision.

    This enum deliberately does not reuse :class:`JudgeVerdict`: the
    original judge verdict remains the record of what the model decided
    at write time, while ``audit_review`` records later human evidence
    about whether that accepted decision was wrong.
    """

    JUDGE_ACCEPTED_WRONG = "JUDGE_ACCEPTED_WRONG"


@dataclass(frozen=True, slots=True)
class FindingKey:
    """Stable key material for matching a finding against an allowlist."""

    file_path: str
    rule_id: str
    symbol_context: tuple[str, ...]
    fingerprint: str

    @property
    def canonical_key(self) -> str:
        """Return the exact suppression key."""
        symbol_part = ":".join(self.symbol_context) if self.symbol_context else "_module_"
        return f"{self.file_path}:{self.rule_id}:{symbol_part}:fp={self.fingerprint}"


@dataclass(frozen=True, slots=True)
class AuditReview:
    """Operator review attached after a judge-accepted entry is found wrong."""

    verdict: AuditReviewVerdict
    reviewer: str
    reviewed_at: datetime
    rationale: str


@dataclass(slots=True)
class AllowlistEntry:
    """An exact allowlist entry for one finding fingerprint.

    Judge metadata (``judge_verdict``, ``judge_recorded_at``,
    ``judge_model``, ``judge_rationale``) is ``None`` for entries
    written before the cicd-judge gate existed. Per the project's
    fabrication-decision test, ``None`` is the honest representation
    of "this entry predates the judge" — filling in a synthetic
    "UNKNOWN" verdict or epoch timestamp would invent audit data the
    judge never produced. Entries written through ``elspeth-lints
    justify`` carry the full judge-metadata tuple.

    ``judge_model_verdict`` is the *model's* verdict, distinct from
    ``judge_verdict`` (the *entry's* verdict). They diverge only when
    the operator used ``--operator-override``: ``judge_verdict`` then
    becomes ``OVERRIDDEN_BY_OPERATOR`` and ``judge_model_verdict``
    preserves what the model originally said (typically ``BLOCKED``,
    less commonly ``ACCEPTED``). For non-override entries this field
    is ``None``: the model's verdict and the entry's verdict are
    identical, so duplicating it would be fabrication of a divergence
    that doesn't exist. The field's value is "override-rate-by-
    underlying-verdict is queryable" — a downstream aggregator can
    distinguish overrides of ACCEPTED entries (harmless) from
    overrides of BLOCKED entries (the dangerous signal).

    ``judge_metadata_signature`` is the versioned HMAC over the judge
    verdict, model verdict, recorded_at timestamp, model id, model
    rationale, source fingerprint, AST path, excerpt-redaction records,
    and entry key. The binding scheme is selected by
    ``judge_signature_version`` and recorded inside the signed payload:
    v1 binds the whole-file ``file_fingerprint``; v2 binds the enclosing-
    scope ``scope_fingerprint``. A v1 entry carries ``file_fingerprint``
    and no ``scope_fingerprint``; a v2 entry the reverse. It is verified
    on production source-root loads so in-place edits to the judge quartet,
    its binding, or its redaction audit trail do not remain invisible.

    ``audit_review`` is a post-judge human review record. It is nullable:
    absence means no later falsification has been recorded. Presence is
    valid only for ``judge_verdict=ACCEPTED`` entries because the review
    answers one question: "was the prior accepted suppression wrong?"
    It does not rewrite the original judge verdict.
    """

    key: str
    owner: str
    reason: str
    safety: str
    expires: date | None
    file_fingerprint: str | None = None
    ast_path: str | None = None
    scope_fingerprint: str | None = None
    judge_signature_version: int | None = None
    judge_transport: str | None = None
    pattern: str | None = None
    source_file: str = ""
    matched: bool = field(default=False, compare=False)
    judge_verdict: JudgeVerdict | None = None
    judge_recorded_at: datetime | None = None
    judge_model: str | None = None
    judge_rationale: str | None = None
    judge_confidence: float | None = None
    judge_model_verdict: JudgeVerdict | None = None
    judge_policy_hash: str | None = None
    judge_metadata_signature: str | None = None
    judge_excerpt_redactions: tuple[RedactionRecord, ...] = ()
    audit_review: AuditReview | None = None

    def matches(self, finding: FindingKey) -> bool:
        """Return whether this exact entry suppresses the finding."""
        return self.key == finding.canonical_key


@dataclass(slots=True)
class PerFileRule:
    """A per-file allowlist rule for one or more rule ids."""

    pattern: str
    rules: tuple[str, ...]
    reason: str
    expires: date | None
    max_hits: int | None = None
    source_file: str = ""
    matched_count: int = field(default=0, compare=False)

    def matches(self, finding: FindingKey) -> bool:
        """Return whether this per-file rule suppresses the finding."""
        if finding.rule_id not in self.rules:
            return False
        return fnmatch.fnmatch(finding.file_path, self.pattern)


@dataclass(frozen=True, slots=True)
class AllowlistBudgetViolation:
    """A loaded allowlist count exceeded a configured ratchet ceiling.

    Emitted by ``Allowlist.get_budget_violations()`` when the loaded entry
    counts exceed any of the six ratchet ceilings declared under
    ``defaults.allowlist_budget`` in the YAML. Permanent entries are those
    with no ``expires`` field; the permanent_* counts let callers ratchet
    against debt that was never bounded.
    """

    category: str
    current: int
    max_allowed: int


@dataclass(slots=True)
class Allowlist:
    """Loaded allowlist entries and per-file rules."""

    entries: list[AllowlistEntry]
    per_file_rules: list[PerFileRule] = field(default_factory=list)
    fail_on_stale: bool = True
    fail_on_expired: bool = True
    max_allow_hits: int | None = None
    max_per_file_rules: int | None = None
    max_total_entries: int | None = None
    max_permanent_allow_hits: int | None = None
    max_permanent_per_file_rules: int | None = None
    max_permanent_total_entries: int | None = None

    def match(self, finding: FindingKey) -> AllowlistEntry | PerFileRule | None:
        """Return the first matching suppression, if any.

        This method mutates match accounting on the returned object:
        exact entries get ``matched=True`` and per-file rules increment
        ``matched_count``. An ``Allowlist`` instance is therefore a
        single analysis-run accumulator, not a thread-safe immutable
        lookup table shared across concurrent checks.
        """
        for entry in self.entries:
            if entry.matches(finding):
                entry.matched = True
                return entry
        for rule in self.per_file_rules:
            if rule.matches(finding):
                rule.matched_count += 1
                return rule
        return None

    def get_unused_entries(self) -> list[AllowlistEntry]:
        """Return exact allowlist entries that did not match any finding."""
        return [entry for entry in self.entries if not entry.matched]

    def get_unused_rules(self) -> list[PerFileRule]:
        """Return per-file rules that did not match any finding."""
        return [rule for rule in self.per_file_rules if rule.matched_count == 0]

    def get_expired_entries(self) -> list[AllowlistEntry]:
        """Return exact allowlist entries whose expiry is in the past."""
        today = datetime.now(UTC).date()
        return [entry for entry in self.entries if entry.expires is not None and entry.expires < today]

    def get_expired_rules(self) -> list[PerFileRule]:
        """Return per-file allowlist rules whose expiry is in the past."""
        today = datetime.now(UTC).date()
        return [rule for rule in self.per_file_rules if rule.expires is not None and rule.expires < today]

    def get_exceeded_rules(self) -> list[PerFileRule]:
        """Return per-file allowlist rules that matched more than their cap."""
        return [rule for rule in self.per_file_rules if rule.max_hits is not None and rule.matched_count > rule.max_hits]

    def get_budget_violations(self) -> list[AllowlistBudgetViolation]:
        """Return configured allowlist-count ratchet overruns."""
        total_entries = len(self.entries) + len(self.per_file_rules)
        permanent_allow_hits = sum(1 for entry in self.entries if entry.expires is None)
        permanent_per_file_rules = sum(1 for rule in self.per_file_rules if rule.expires is None)
        permanent_total_entries = permanent_allow_hits + permanent_per_file_rules
        checks = (
            ("allow_hits", len(self.entries), self.max_allow_hits),
            ("per_file_rules", len(self.per_file_rules), self.max_per_file_rules),
            ("total_entries", total_entries, self.max_total_entries),
            ("permanent_allow_hits", permanent_allow_hits, self.max_permanent_allow_hits),
            ("permanent_per_file_rules", permanent_per_file_rules, self.max_permanent_per_file_rules),
            ("permanent_total_entries", permanent_total_entries, self.max_permanent_total_entries),
        )
        return [
            AllowlistBudgetViolation(category=category, current=current, max_allowed=max_allowed)
            for category, current, max_allowed in checks
            if max_allowed is not None and current > max_allowed
        ]


def load_allowlist(
    path: Path,
    *,
    valid_rule_ids: Collection[str],
    source_root: Path | None = None,
) -> Allowlist:
    """Load an allowlist from a YAML file or directory of YAML files.

    ``source_root`` enables the C8-3 quartet-transplant defence at load
    time: when an entry carries judge metadata, its persisted
    ``file_fingerprint`` is recomputed from the bytes of the source file
    (the path encoded in the entry's key, resolved against
    ``source_root``) and must match. A mismatch ⇒ either the source
    drifted under a still-trusted judge verdict (the rationale no
    longer describes the live code), or the quartet was transplanted
    from a different file. Either case is corruption and must crash on
    load rather than silently propagate into the gate's decision.

    Cross-file transplants are caught here. In-file transplants
    (quartet pasted onto a different AST node within the same file,
    with the entry's key rebound to match) keep the file_fingerprint
    intact and are caught at match time via
    :func:`verify_entry_binding_against_finding`.

    When ``source_root`` is ``None`` the file-fingerprint recompute is
    skipped (call sites that don't know the source root, e.g. some
    diagnostics loaders, still get co-presence enforcement). The HMAC
    over the judge metadata is also verified only on source-root loads:
    aggregate/report loaders can inspect historical entries without a
    deployment secret, while production rule loaders MUST pass
    ``source_root`` so both the source binding and metadata-tamper gate
    are live.
    """
    # No-silent-failures: a source-root load that is in shape-only mode with the
    # HMAC key absent will skip the cryptographic recompute for every judged
    # entry (validating signature *shape* only). That is the intended fork-PR
    # degradation, but degrading a security control silently is itself a defect.
    # Emit one prominent warning per load so the downgrade is visible in the CI
    # log and a reviewer never mistakes a green shape-only run for a verified one.
    if source_root is not None and _can_skip_judge_metadata_hmac_recompute_for_missing_key():
        sys.stderr.write(
            "WARNING: tier_model allowlist loaded with judge-metadata HMAC verification "
            f"DOWNGRADED to shape-only ({_JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR}="
            f"{_JUDGE_METADATA_SIGNATURE_VERIFY_SHAPE_ONLY_WHEN_KEY_MISSING}, "
            f"{_JUDGE_METADATA_SIGNATURE_ENV_VAR} absent). Judge-metadata signatures are "
            "shape-checked only; this load CANNOT detect forged or tampered judge metadata. "
            "This mode is intended for untrusted fork-PR CI only — a trusted context (key "
            "present, verify mode 'required') must re-verify before any merge is authoritative.\n"
        )
    if path.is_dir():
        defaults = _load_defaults(path / "_defaults.yaml")
        entries: list[AllowlistEntry] = []
        per_file_rules: list[PerFileRule] = []
        for yaml_file in sorted(file for file in path.glob("*.yaml") if file.name != "_defaults.yaml"):
            data = _load_yaml_file(yaml_file)
            entries.extend(_parse_allow_hits(data, source_file=yaml_file.name, source_root=source_root))
            per_file_rules.extend(_parse_per_file_rules(data, valid_rule_ids=valid_rule_ids, source_file=yaml_file.name))
        return _allowlist_from_defaults(entries=entries, per_file_rules=per_file_rules, defaults=defaults)

    data = _load_yaml_file(path)
    defaults = _defaults_from_mapping(data)
    return _allowlist_from_defaults(
        entries=_parse_allow_hits(data, source_file=path.name, source_root=source_root),
        per_file_rules=_parse_per_file_rules(data, valid_rule_ids=valid_rule_ids, source_file=path.name),
        defaults=defaults,
    )


def _allowlist_from_defaults(
    *,
    entries: list[AllowlistEntry],
    per_file_rules: list[PerFileRule],
    defaults: _Defaults,
) -> Allowlist:
    return Allowlist(
        entries=entries,
        per_file_rules=per_file_rules,
        fail_on_stale=defaults.fail_on_stale,
        fail_on_expired=defaults.fail_on_expired,
        max_allow_hits=defaults.budget.max_allow_hits,
        max_per_file_rules=defaults.budget.max_per_file_rules,
        max_total_entries=defaults.budget.max_total_entries,
        max_permanent_allow_hits=defaults.budget.max_permanent_allow_hits,
        max_permanent_per_file_rules=defaults.budget.max_permanent_per_file_rules,
        max_permanent_total_entries=defaults.budget.max_permanent_total_entries,
    )


@dataclass(frozen=True, slots=True)
class _BudgetCeilings:
    max_allow_hits: int | None = None
    max_per_file_rules: int | None = None
    max_total_entries: int | None = None
    max_permanent_allow_hits: int | None = None
    max_permanent_per_file_rules: int | None = None
    max_permanent_total_entries: int | None = None


@dataclass(frozen=True, slots=True)
class _Defaults:
    fail_on_stale: bool = True
    fail_on_expired: bool = True
    budget: _BudgetCeilings = field(default_factory=_BudgetCeilings)


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path}: allowlist YAML file is required")
    size = path.stat().st_size
    if size > _MAX_ALLOWLIST_YAML_BYTES:
        raise ValueError(
            f"{path}: allowlist YAML file is {size} bytes, exceeds maximum allowlist YAML size {_MAX_ALLOWLIST_YAML_BYTES} bytes"
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: allowlist YAML must be a mapping")
    return raw


def _load_defaults(path: Path) -> _Defaults:
    if not path.exists():
        return _Defaults()
    return _defaults_from_mapping(_load_yaml_file(path))


def _defaults_from_mapping(data: dict[str, Any]) -> _Defaults:
    defaults = _mapping_or_empty(data, "defaults")
    return _Defaults(
        fail_on_stale=_bool_value(defaults, "fail_on_stale", default=True),
        fail_on_expired=_bool_value(defaults, "fail_on_expired", default=True),
        budget=_parse_allowlist_budget(defaults),
    )


def _parse_allowlist_budget(defaults: dict[str, Any]) -> _BudgetCeilings:
    """Parse the optional ``defaults.allowlist_budget`` block.

    A missing ``allowlist_budget`` key, missing per-ceiling keys, and
    explicit per-ceiling ``null`` values all produce ``None`` (permissive).
    Non-negative integers tighten the corresponding ceiling. A block-level
    ``allowlist_budget: null`` raises ``ValueError`` — write either an empty
    mapping (``{}``) or omit the key entirely if you want permissive
    defaults. Any other value also raises ``ValueError``.
    """
    budget = _mapping_or_empty(defaults, "allowlist_budget")
    return _BudgetCeilings(
        max_allow_hits=_optional_nonneg_int(budget, "max_allow_hits", context="allowlist_budget"),
        max_per_file_rules=_optional_nonneg_int(budget, "max_per_file_rules", context="allowlist_budget"),
        max_total_entries=_optional_nonneg_int(budget, "max_total_entries", context="allowlist_budget"),
        max_permanent_allow_hits=_optional_nonneg_int(budget, "max_permanent_allow_hits", context="allowlist_budget"),
        max_permanent_per_file_rules=_optional_nonneg_int(budget, "max_permanent_per_file_rules", context="allowlist_budget"),
        max_permanent_total_entries=_optional_nonneg_int(budget, "max_permanent_total_entries", context="allowlist_budget"),
    )


def _parse_allow_hits(
    data: dict[str, Any],
    *,
    source_file: str,
    source_root: Path | None,
) -> list[AllowlistEntry]:
    entries_raw = _list_value(data, "allow_hits")
    entries: list[AllowlistEntry] = []
    for index, raw_entry in enumerate(entries_raw):
        entry = _mapping_value(raw_entry, f"allow_hits[{index}]")
        ctx = f"allow_hits[{index}]"
        allowlist_entry = AllowlistEntry(
            key=_required_string(entry, "key", context=ctx),
            owner=_required_audit_anchor_string(entry, "owner", context=ctx),
            reason=_required_audit_anchor_string(entry, "reason", context=ctx),
            safety=_required_string(entry, "safety", context=ctx),
            expires=_optional_date_alias(entry, "expires", "expires_at", context=ctx),
            file_fingerprint=_optional_string(entry, "file_fingerprint", context=ctx),
            ast_path=_optional_string(entry, "ast_path", context=ctx),
            scope_fingerprint=_optional_string(entry, "scope_fingerprint", context=ctx),
            judge_signature_version=_optional_signature_version(entry, "judge_signature_version", context=ctx),
            judge_transport=_optional_string(entry, "judge_transport", context=ctx),
            pattern=_optional_string(entry, "pattern", context=ctx),
            source_file=source_file,
            judge_verdict=_optional_judge_verdict(entry, "judge_verdict", context=ctx, allow_blocked=False),
            judge_recorded_at=_optional_datetime(entry, "judge_recorded_at", context=ctx),
            judge_model=_optional_string(entry, "judge_model", context=ctx),
            judge_rationale=_optional_string(entry, "judge_rationale", context=ctx),
            judge_confidence=_optional_confidence(entry, "judge_confidence", context=ctx),
            judge_model_verdict=_optional_judge_verdict(
                entry,
                "judge_model_verdict",
                context=ctx,
                allow_blocked=True,
                allow_operator_override=False,
            ),
            judge_policy_hash=_optional_string(entry, "judge_policy_hash", context=ctx),
            judge_metadata_signature=_optional_string(entry, "judge_metadata_signature", context=ctx),
            judge_excerpt_redactions=_parse_judge_excerpt_redactions(entry, context=ctx),
            audit_review=_parse_audit_review(entry, context=ctx),
        )
        _validate_judge_metadata_atomic(allowlist_entry, context=ctx)
        _validate_audit_review_context(allowlist_entry, context=ctx)
        if source_root is not None and allowlist_entry.judge_verdict is not None:
            try:
                _verify_source_binding_at_load(allowlist_entry, source_root=source_root, context=ctx)
            except DanglingAllowlistEntry as exc:
                sys.stderr.write(
                    f"WARNING: stale allowlist entry {ctx} binds to a deleted source file — "
                    f"operator should remove it and re-sign. Detail: {exc}\n"
                )
                continue  # skip dangling entry; do NOT append; do NOT verify signature
            _verify_judge_metadata_signature_at_load(allowlist_entry, context=ctx)
        entries.append(allowlist_entry)
    return entries


def _parse_audit_review(data: dict[str, Any], *, context: str) -> AuditReview | None:
    """Parse the optional post-judge audit review block."""
    if "audit_review" not in data or data["audit_review"] is None:
        return None
    item_context = f"{context}.audit_review"
    item = _mapping_value(data["audit_review"], item_context)
    return AuditReview(
        verdict=_required_audit_review_verdict(item, "verdict", context=item_context),
        reviewer=_required_audit_anchor_string(item, "reviewer", context=item_context),
        reviewed_at=_required_datetime(item, "reviewed_at", context=item_context),
        rationale=_required_audit_anchor_string(item, "rationale", context=item_context),
    )


def _parse_judge_excerpt_redactions(data: dict[str, Any], *, context: str) -> tuple[RedactionRecord, ...]:
    raw_redactions = _list_value(data, "judge_excerpt_redactions")
    redactions: list[RedactionRecord] = []
    for index, raw_redaction in enumerate(raw_redactions):
        item_context = f"{context}.judge_excerpt_redactions[{index}]"
        item = _mapping_value(raw_redaction, item_context)
        redacted_hash = _required_string(item, "redacted_hash", context=item_context)
        if len(redacted_hash) != 16 or any(ch not in "0123456789abcdef" for ch in redacted_hash):
            raise ValueError(f"{item_context}.redacted_hash must be 16 lowercase hex characters")
        redactions.append(
            RedactionRecord(
                pattern_name=_required_string(item, "pattern", context=item_context),
                byte_count=_required_positive_int(item, "byte_count", context=item_context),
                redacted_hash=redacted_hash,
            )
        )
    return tuple(redactions)


def _file_path_from_canonical_key(key: str) -> str:
    """Extract the ``file_path`` segment from a canonical allowlist key.

    The canonical key shape is
    ``<file_path>:<rule_id>:<symbol_context_joined_by_colons>:fp=<hash>``
    (see :attr:`FindingKey.canonical_key`). The symbol_context tuple is
    joined by ``:`` — a method symbol like ``("Widget", "lookup")``
    becomes ``Widget:lookup`` — so rsplit-from-the-right does not
    recover file_path. We anchor on the ``.py:`` boundary instead: the
    file path is a Python source file, so the FIRST occurrence of
    ``.py:`` delimits its end.

    Returns the segment before ``.py:`` plus the ``.py`` suffix.
    Raises ``ValueError`` if the key does not match the canonical shape
    (missing ``:fp=`` suffix, or no ``.py:`` boundary).
    """
    if ":fp=" not in key:
        raise ValueError(f"allowlist key is not in canonical form (missing ':fp=' suffix): {key!r}")
    marker = ".py:"
    py_idx = key.find(marker)
    if py_idx < 0:
        raise ValueError(f"allowlist key is not in canonical form (no '.py:' boundary delimiting file_path from rule_id): {key!r}")
    return key[: py_idx + len(".py")]


def _key_without_fp(key: str) -> str:
    """Return the ``file_path:rule_id:symbol`` prefix of a canonical key.

    The canonical shape is ``<file>:<rule>:<symbol>:fp=<hash>``. The fingerprint
    is hex and the symbol is a ``:``-joined tuple of Python identifiers, neither
    of which can contain ``:fp=``, so a single split is unambiguous.
    """
    if ":fp=" not in key:
        raise ValueError(f"allowlist key is not in canonical form (missing ':fp=' suffix): {key!r}")
    return key.split(":fp=", 1)[0]


def find_scope_fallback_entry(
    entries: list[AllowlistEntry],
    *,
    canonical_key: str,
    scope_fingerprint: str,
    ast_path: str,
    scope_depth: int,
) -> AllowlistEntry | None:
    """Match a finding whose exact key missed against a scope-stable v2 entry.

    Rescues a judge-gated **v2** entry whose module-rooted ``ast_path`` drifted
    (a module-level statement shifted the leading index) but whose enclosing
    scope and within-scope position are unchanged — the same suppression,
    relocated by an unrelated module-body edit.

    An entry is a candidate iff ALL hold:

    1. same ``file_path:rule_id:symbol`` as the finding;
    2. it is judge-gated and v2 (carries ``scope_fingerprint``); v1 entries bind
       ``file_fingerprint`` and are skipped;
    3. its persisted ``scope_fingerprint`` equals the finding's (the enclosing
       scope body is byte-identical — a real body edit fails here);
    4. its ``ast_path`` has the same component count (a depth-changing relocation
       fails here);
    5. its within-scope suffix ``ast_path.split("/")[scope_depth:]`` equals the
       finding's (the within-scope position is identical — the fallback-path
       replacement for the exact ``ast_path`` transplant defence).

    Returns the unique candidate, or ``None`` when there are zero or **two or
    more** (ambiguity must never silently bind — fail closed). An empty
    ``scope_fingerprint`` on the finding (un-stamped / non-tier_model finding)
    yields ``None``: there is nothing to bind against.

    Match-only: this never rewrites the entry's key, ast_path, or signature. The
    matched entry keeps its pre-shift key (still self-consistent at load); only
    a *new* finding is allowed to match an *old* key.
    """
    if not scope_fingerprint:
        return None
    finding_prefix = _key_without_fp(canonical_key)
    live_components = ast_path.split("/")
    # scope_depth == len(live_components) (empty suffix) is REACHABLE and safe:
    # a finding whose node IS its enclosing scope (e.g. an R_TB_MALFORMED
    # diagnostic on a FunctionDef) has K == component count. Do NOT assert
    # scope_depth < len(...) — that would crash the gate on such diagnostics.
    # At most one finding per scope is the scope node itself, so an empty suffix
    # cannot wrong-bind two distinct findings; non-judge-gated findings match 0
    # candidates and return None.
    live_suffix = live_components[scope_depth:]
    candidates: list[AllowlistEntry] = []
    for entry in entries:
        if entry.judge_verdict is None or entry.scope_fingerprint is None or entry.ast_path is None:
            continue
        if entry.judge_verdict is JudgeVerdict.BLOCKED:
            # Defense-in-depth: BLOCKED is an in-memory-only verdict meaning the entry
            # was rejected and NOT written. The loader (_validate_judge_metadata_atomic
            # invariant 5) already crashes on any persisted BLOCKED entry, so this cannot
            # occur from a real load — but a suppression gate must never honour a BLOCKED
            # verdict even if one is constructed in-memory.
            continue
        if _key_without_fp(entry.key) != finding_prefix:
            continue
        if entry.scope_fingerprint != scope_fingerprint:
            continue
        stored_components = entry.ast_path.split("/")
        if len(stored_components) != len(live_components):
            continue
        if stored_components[scope_depth:] != live_suffix:
            continue
        candidates.append(entry)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _compute_file_fingerprint(source_path: Path) -> str:
    """Return the SHA-256 hex digest of ``source_path``'s bytes.

    This is the C8-3 binding primitive: a judge inspected the file's
    bytes at a given moment, and the persisted ``file_fingerprint`` is
    that moment's digest. Recomputing on load and comparing detects
    both source-drift (file modified after judgment, judge's rationale
    no longer describes live code) and cross-file quartet transplant
    (quartet copied onto an entry whose file_path differs from the
    file the judge originally inspected).
    """
    return hashlib.sha256(source_path.read_bytes()).hexdigest()


def _resolve_allowlist_source_path(file_path: str, *, source_root: Path, context: str) -> Path:
    """Resolve an allowlist key path without allowing reads outside ``source_root``."""
    raw_path = Path(file_path)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(
            f"{context}: judge-gated entry binds to {file_path!r} outside source_root "
            f"{source_root}; allowlist keys must use normalized relative .py paths. "
            "Refusing to read source bytes."
        )

    resolved_root = source_root.resolve(strict=False)
    resolved_source = (resolved_root / raw_path).resolve(strict=False)
    if not resolved_source.is_relative_to(resolved_root):
        raise ValueError(
            f"{context}: judge-gated entry binds to {file_path!r} outside source_root "
            f"{source_root} after path resolution ({resolved_source}). Refusing to read source bytes."
        )
    return resolved_source


def compute_judge_metadata_signature(
    *,
    key: str,
    ast_path: str,
    judge_verdict: JudgeVerdict,
    judge_recorded_at: datetime,
    judge_model: str,
    judge_rationale: str,
    judge_policy_hash: str,
    signature_version: int = 1,
    file_fingerprint: str | None = None,
    scope_fingerprint: str | None = None,
    judge_transport: str | None = None,
    judge_model_verdict: JudgeVerdict | None = None,
    judge_confidence: float | None = None,
    judge_excerpt_redactions: tuple[RedactionRecord, ...] = (),
    hmac_key: bytes | None = None,
) -> str:
    """Return the versioned HMAC binding for a post-judge allowlist entry.

    The signature binds the audit-significant judge metadata cluster
    (verdict/rationale/model/timestamp/policy hash) and excerpt-redaction
    records to the source identity the entry was judged against. Two
    binding schemes exist, selected by ``signature_version`` and recorded
    *inside* the signed payload (``"version"``) so a v1<->v2 flip is itself
    unforgeable:

    * **v1** binds the whole-file ``file_fingerprint`` (the SHA-256 of the
      source file the judge inspected). Prefix ``hmac-sha256:v1:``.
    * **v2** binds the enclosing-scope ``scope_fingerprint`` (the AST
      fingerprint of the scope the finding lives in). Prefix
      ``hmac-sha256:v2:``.

    ``signature_version`` defaults to ``1`` for back-compatibility with
    callers that omit it; the ``justify`` write path passes
    ``signature_version=2`` explicitly and mints v2 (scope-bound) entries.
    Operators may still edit administrative fields such as owner/expiry,
    but changing what the judge supposedly said, which source location it
    judged, which secrets were scrubbed, or which binding scheme was used
    requires re-running ``justify`` with the deployment-held HMAC key.
    """
    if hmac_key is None:
        hmac_key = _judge_metadata_hmac_key()
    if signature_version == 2:
        if scope_fingerprint is None:
            raise ValueError("compute_judge_metadata_signature: scope_fingerprint is required for signature_version 2")
        if judge_transport is None:
            raise ValueError("compute_judge_metadata_signature: judge_transport is required for signature_version 2")
        # ``judge_transport`` is bound into the v2 payload ONLY — never the v1
        # branch below. v1 is legacy (deleted in a later task) and its signed
        # payload shape must not change. Required (no default): the validator
        # enforces PRESENCE but not CORRECTNESS, so a silent "openrouter"
        # default would let a forgetful future v2 sign site mislabel a signed
        # audit field undetectably. Mirroring ``scope_fingerprint``'s
        # required-for-v2 contract closes that seam (offensive programming).
        binding: dict[str, str] = {
            "scope_fingerprint": scope_fingerprint,
            "judge_transport": judge_transport,
        }
        prefix = _JUDGE_METADATA_SIGNATURE_PREFIX_V2
    elif signature_version == 1:
        if file_fingerprint is None:
            raise ValueError("compute_judge_metadata_signature: file_fingerprint is required for signature_version 1")
        binding = {"file_fingerprint": file_fingerprint}
        prefix = _JUDGE_METADATA_SIGNATURE_PREFIX_V1
    else:
        raise ValueError(f"compute_judge_metadata_signature: unknown signature_version {signature_version!r}")
    payload: dict[str, Any] = {
        "version": signature_version,
        "key": key,
        **binding,
        "ast_path": ast_path,
        "judge_verdict": judge_verdict.value,
        "judge_model_verdict": judge_model_verdict.value if judge_model_verdict is not None else None,
        "judge_recorded_at": judge_recorded_at.isoformat(),
        "judge_model": judge_model,
        "judge_rationale": judge_rationale,
        "judge_policy_hash": judge_policy_hash,
        "judge_excerpt_redactions": [
            {
                "pattern": record.pattern_name,
                "byte_count": record.byte_count,
                "redacted_hash": record.redacted_hash,
            }
            for record in judge_excerpt_redactions
        ],
    }
    if judge_confidence is not None:
        payload["judge_confidence"] = judge_confidence
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hmac.new(hmac_key, canonical, hashlib.sha256).hexdigest()
    return f"{prefix}{digest}"


def _judge_metadata_hmac_key() -> bytes:
    """Load the deployment-held key used to sign judge metadata."""
    raw = os.environ.get(_JUDGE_METADATA_SIGNATURE_ENV_VAR)
    if raw is None or raw == "":
        raise ValueError(
            f"{_JUDGE_METADATA_SIGNATURE_ENV_VAR} is required to verify or write "
            "judge_metadata_signature for post-judge allowlist entries. This key "
            "must be held outside the allowlist YAML; without it, verdict metadata "
            "would be unsigned editable text."
        )
    key = raw.encode("utf-8")
    if len(key) < _MIN_JUDGE_METADATA_HMAC_KEY_BYTES:
        raise ValueError(
            f"{_JUDGE_METADATA_SIGNATURE_ENV_VAR} must be at least "
            f"{_MIN_JUDGE_METADATA_HMAC_KEY_BYTES} bytes when UTF-8 encoded; "
            "shorter HMAC keys are not acceptable for audit metadata binding."
        )
    return key


def _hmac_key_bytes_from_env(var_name: str) -> bytes:
    """Load + validate an HMAC key from an operator-NAMED env var (min 32 bytes).

    Mirrors :func:`_judge_metadata_hmac_key`'s validation but reads a caller-chosen
    variable name: ``rekey`` holds the OLD and NEW keys in two distinct env vars
    (``--old-key-env`` / ``--new-key-env``), never the single production
    ``ELSPETH_JUDGE_METADATA_HMAC_KEY``. Raises ``ValueError`` when the named var is
    unset/empty or the key is too short — the fail-closed gate ``rekey`` runs before
    any tree read.
    """
    raw = os.environ.get(var_name)
    if raw is None or raw == "":
        raise ValueError(f"{var_name} is required (HMAC key environment variable) but is unset or empty.")
    key = raw.encode("utf-8")
    if len(key) < _MIN_JUDGE_METADATA_HMAC_KEY_BYTES:
        raise ValueError(
            f"{var_name} must be at least {_MIN_JUDGE_METADATA_HMAC_KEY_BYTES} bytes when UTF-8 encoded; "
            "shorter HMAC keys are not acceptable for audit metadata binding."
        )
    return key


def _judge_metadata_signature_verify_mode() -> str:
    raw = os.environ.get(_JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR, _JUDGE_METADATA_SIGNATURE_VERIFY_REQUIRED)
    if raw not in {
        _JUDGE_METADATA_SIGNATURE_VERIFY_REQUIRED,
        _JUDGE_METADATA_SIGNATURE_VERIFY_SHAPE_ONLY_WHEN_KEY_MISSING,
    }:
        raise ValueError(
            f"{_JUDGE_METADATA_SIGNATURE_VERIFY_MODE_ENV_VAR} must be "
            f"{_JUDGE_METADATA_SIGNATURE_VERIFY_REQUIRED!r} or "
            f"{_JUDGE_METADATA_SIGNATURE_VERIFY_SHAPE_ONLY_WHEN_KEY_MISSING!r}; got {raw!r}"
        )
    return raw


def _can_skip_judge_metadata_hmac_recompute_for_missing_key() -> bool:
    mode = _judge_metadata_signature_verify_mode()
    raw_key = os.environ.get(_JUDGE_METADATA_SIGNATURE_ENV_VAR)
    return mode == _JUDGE_METADATA_SIGNATURE_VERIFY_SHAPE_ONLY_WHEN_KEY_MISSING and (raw_key is None or raw_key == "")


def _verify_judge_metadata_signature_at_load(entry: AllowlistEntry, *, context: str, allow_shape_only: bool = True) -> None:
    """Assert a post-judge entry's verdict metadata has not been edited."""
    if entry.judge_metadata_signature is None:
        raise ValueError(
            f"{context}: judge_metadata_signature is missing; source-root loads of "
            "post-judge entries require an HMAC over the judge verdict, rationale, "
            "model, recorded_at, file_fingerprint, ast_path, key, and excerpt "
            "redaction records. Unsigned judge metadata is editable plain text and "
            "cannot be trusted by the gate."
        )
    _validate_judge_metadata_signature_shape(entry.judge_metadata_signature, context=context)

    # The binding field required depends on the entry's signature version.
    # Co-presence/forbidden-other was already enforced by invariant 8 in
    # ``_validate_judge_metadata_atomic`` (runs before this in _parse_allow_hits),
    # so exactly one of file_fingerprint/scope_fingerprint is present here.
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 2:
        assert entry.scope_fingerprint is not None
    else:
        assert entry.file_fingerprint is not None
    assert entry.ast_path is not None
    assert entry.judge_verdict is not None
    assert entry.judge_recorded_at is not None
    assert entry.judge_model is not None
    assert entry.judge_rationale is not None
    assert entry.judge_policy_hash is not None
    if allow_shape_only and _can_skip_judge_metadata_hmac_recompute_for_missing_key():
        return
    expected = _recompute_entry_signature(entry, hmac_key=_judge_metadata_hmac_key())
    if not hmac.compare_digest(entry.judge_metadata_signature, expected):
        raise ValueError(
            f"{context}: judge_metadata_signature mismatch for entry {entry.key!r}; "
            "the persisted judge verdict/rationale/model/recorded_at or binding "
            "fields were edited without the deployment HMAC key. Re-run "
            "`elspeth-lints justify` to produce a fresh signed entry."
        )


def _recompute_entry_signature(entry: AllowlistEntry, *, hmac_key: bytes) -> str:
    """Recompute ``entry``'s HMAC signature under ``hmac_key`` (the ONE marshalling).

    This is the single field-marshalling shared by the keyless load-time verifier
    (:func:`_verify_judge_metadata_signature_at_load`), the keyed verify
    (:func:`verify_entry_signature_with_key`), and the ``rekey`` keyed write. It
    passes the FULL signed field set off the already-persisted parsed
    ``AllowlistEntry`` — including the version branch (``file_fingerprint`` vs
    ``scope_fingerprint``) and the ``judge_transport`` fallback — VERBATIM. ``hmac_key``
    is the *only* thing callers vary, so a keyed recompute is field-identical to the
    production verifier and cannot drift out of parity with it.

    No write-time transform runs here: ``judge_confidence`` is passed off the parsed
    entry as-is (the round-at-write-time transform in ``cli.py`` is not re-applied;
    the parsed entry already carries the rounded value).
    """
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    assert entry.ast_path is not None
    assert entry.judge_verdict is not None
    assert entry.judge_recorded_at is not None
    assert entry.judge_model is not None
    assert entry.judge_rationale is not None
    assert entry.judge_policy_hash is not None
    return compute_judge_metadata_signature(
        key=entry.key,
        ast_path=entry.ast_path,
        judge_verdict=entry.judge_verdict,
        judge_model_verdict=entry.judge_model_verdict,
        judge_recorded_at=entry.judge_recorded_at,
        judge_model=entry.judge_model,
        judge_rationale=entry.judge_rationale,
        judge_policy_hash=entry.judge_policy_hash,
        judge_excerpt_redactions=entry.judge_excerpt_redactions,
        judge_confidence=entry.judge_confidence,
        signature_version=version,
        file_fingerprint=entry.file_fingerprint,
        scope_fingerprint=entry.scope_fingerprint,
        # The v2 atomic validator guarantees ``judge_transport`` is present on
        # valid v2 entries, so this fallback only ever applies to v1 entries —
        # where the signer's v2 branch never reads it. Passing it
        # unconditionally keeps the recompute call uniform across versions.
        judge_transport=entry.judge_transport if entry.judge_transport is not None else "openrouter",
        hmac_key=hmac_key,
    )


def verify_entry_signature_with_key(entry: AllowlistEntry, *, hmac_key: bytes) -> None:
    """Authoritatively verify ``entry``'s signature under an EXPLICIT key.

    Unlike :func:`_verify_judge_metadata_signature_at_load` (keyless — env var only,
    with a shape-only escape hatch when the deployment key is absent), this takes the
    key bytes directly and has NO shape-only path: an explicit key is always
    authoritative. Raises ``ValueError`` if the entry carries no signature or the
    recomputed signature does not match. Used by ``rekey``'s dual-key verify window
    (verify-under-OLD-or-NEW) and shares the one marshalling with the production
    loader via :func:`_recompute_entry_signature`, so a pass here is a pass there.
    """
    if entry.judge_metadata_signature is None:
        raise ValueError(f"verify_entry_signature_with_key: entry {entry.key!r} has no judge_metadata_signature to verify")
    expected = _recompute_entry_signature(entry, hmac_key=hmac_key)
    if not hmac.compare_digest(entry.judge_metadata_signature, expected):
        raise ValueError(f"verify_entry_signature_with_key: judge_metadata_signature mismatch for entry {entry.key!r}")


def _validate_judge_metadata_signature_shape(signature: str, *, context: str) -> None:
    if not signature.startswith(JUDGE_METADATA_SIGNATURE_PREFIXES):
        raise ValueError(
            f"{context}: judge_metadata_signature must start with one of {JUDGE_METADATA_SIGNATURE_PREFIXES}; got {signature!r}"
        )
    prefix = next(p for p in JUDGE_METADATA_SIGNATURE_PREFIXES if signature.startswith(p))
    digest = signature.removeprefix(prefix)
    if len(digest) != 64:
        raise ValueError(f"{context}: judge_metadata_signature digest must be 64 lowercase hex characters")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{context}: judge_metadata_signature digest must be valid hex") from exc
    if digest.lower() != digest:
        raise ValueError(f"{context}: judge_metadata_signature digest must use lowercase hex")


def _validate_judge_policy_hash_shape(policy_hash: str, *, context: str) -> None:
    prefix = "sha256:"
    if not policy_hash.startswith(prefix):
        raise ValueError(f"{context}: judge_policy_hash must start with {prefix!r}; got {policy_hash!r}")
    digest = policy_hash.removeprefix(prefix)
    if len(digest) != 64:
        raise ValueError(f"{context}: judge_policy_hash digest must be 64 lowercase hex characters")
    try:
        bytes.fromhex(digest)
    except ValueError as exc:
        raise ValueError(f"{context}: judge_policy_hash digest must be valid hex") from exc
    if digest.lower() != digest:
        raise ValueError(f"{context}: judge_policy_hash digest must use lowercase hex")


def _verify_source_binding_at_load(entry: AllowlistEntry, *, source_root: Path, context: str) -> None:
    """Verify a judge-gated entry's source binding at load, dispatched by version.

    All versions: the source file the entry's key points at must exist
    (a binding to a deleted file is audit-broken). v1 additionally
    recomputes the whole-file byte hash (its binding primitive). v2's
    binding primitive (scope_fingerprint) is parse-dependent and is
    verified at match time in ``verify_entry_binding_against_finding``,
    reusing the scanner's parse — not re-parsed here.

    Crashes on mismatch (Tier-1 doctrine: silently propagating a stale
    or transplanted judge verdict into the gate's decision is evidence
    tampering). Crashes on missing source file (the entry binds to a
    file that no longer exists — the rotation/deletion was not
    accompanied by removing the dependent judge-gated entry).
    """
    file_path = _file_path_from_canonical_key(entry.key)
    source_path = _resolve_allowlist_source_path(file_path, source_root=source_root, context=context)
    if not source_path.exists():
        raise DanglingAllowlistEntry(
            f"{context}: judge-gated entry binds to {file_path!r} which does not exist "
            f"under {source_root}; either the source file was removed without removing "
            f"the dependent allowlist entry, or the entry's key was transplanted from a "
            f"different repository layout. Refusing to load."
        )
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 2:
        # v2 binds via ``scope_fingerprint``, which is parse-dependent and is
        # verified at match time in ``verify_entry_binding_against_finding``
        # (reusing the scanner's parse). There is no whole-file hash to
        # recompute at load; the file-exists guard above is the only load-time
        # source-binding check for v2.
        return
    live_fingerprint = _compute_file_fingerprint(source_path)
    # ``entry.file_fingerprint`` is non-None here: invariant 8 in
    # ``_validate_judge_metadata_atomic`` (binding co-presence) enforced its
    # presence for v1 judge-gated entries, and the v2 branch returned above, so
    # only v1 reaches this line. The assert is genuine narrowing for mypy.
    assert entry.file_fingerprint is not None  # v1: invariant 8 guarantees presence (mypy narrowing)
    if entry.file_fingerprint != live_fingerprint:
        raise ValueError(
            f"{context}: file_fingerprint mismatch for {file_path!r}: persisted "
            f"{entry.file_fingerprint!r} but live source hashes to "
            f"{live_fingerprint!r}. Either the source file was modified after the "
            f"judge accepted the suppression (the rationale no longer describes the "
            f"live code, re-justify is required) or the judge quartet was "
            f"transplanted from a different file (corruption / tampering)."
        )


def verify_entry_binding_against_finding(entry: AllowlistEntry, *, file_path: str, ast_path: str, scope_fingerprint: str) -> None:
    """Assert a matched judge-gated entry still binds to the live finding.

    Checks ``ast_path`` (all versions — the C8-3 in-file transplant
    defence) and, for v2 entries, ``scope_fingerprint`` (the enclosing-
    scope content the judge actually inspected). The live
    ``scope_fingerprint`` is the value the scanner stamped onto the
    finding; an empty value means the finding was emitted by a
    construction site that did not compute it, which for a v2 entry is a
    defect (we must never accept a v2 binding on an unverifiable empty
    value) and crashes here.

    Pre-judge entries (``judge_verdict is None``) carry no binding fields
    and are not checked — their only binding signal is the canonical_key
    the caller already matched against.
    """
    if entry.judge_verdict is None:
        return
    assert entry.ast_path is not None
    if entry.ast_path != ast_path:
        raise ValueError(
            f"ast_path mismatch on judge-gated entry for {file_path!r}: persisted "
            f"{entry.ast_path!r} but live finding's ast_path is {ast_path!r}. The "
            f"judge accepted a different AST node — either the code was refactored "
            f"and the entry needs re-justification, or the judge quartet was "
            f"transplanted onto a different finding within the same file "
            f"(corruption / tampering). Entry key: {entry.key!r}."
        )
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    if version == 2:
        assert entry.scope_fingerprint is not None  # invariant 8 (v2) guarantees presence
        if not scope_fingerprint:
            raise ValueError(
                f"scope_fingerprint missing on the live finding for judge-gated v2 entry "
                f"{entry.key!r} ({file_path!r}); the scanner must stamp scope_fingerprint on "
                "every finding. An empty value cannot verify a v2 binding."
            )
        if entry.scope_fingerprint != scope_fingerprint:
            raise ValueError(
                f"scope_fingerprint mismatch on judge-gated entry for {file_path!r}: persisted "
                f"{entry.scope_fingerprint!r} but the live enclosing scope hashes to "
                f"{scope_fingerprint!r}. The function/class the judge inspected changed; "
                f"re-justify is required. Entry key: {entry.key!r}."
            )


def _validate_judge_metadata_atomic(entry: AllowlistEntry, *, context: str) -> None:
    """Crash if an entry's judge-metadata cluster is internally inconsistent.

    The judge-metadata fields are a tightly coupled audit record: they
    must be either *all absent* (the pre-judge era representation, where
    ``judge_verdict`` is ``None`` and every other ``judge_*`` field is
    ``None``) or *all present* (a fully-recorded judge interaction, where
    ``judge_verdict``, ``judge_recorded_at``, ``judge_model``,
    ``judge_policy_hash``, and ``judge_rationale`` are all set, with
    ``judge_model_verdict`` set iff the entry's verdict is
    ``OVERRIDDEN_BY_OPERATOR``).

    The fields are our own data (Tier 1 in the project's trust model):
    an allowlist on disk was written by ``elspeth-lints justify`` or
    hand-edited by an operator, both of which are inside our trust
    boundary. Inconsistent shape is corruption — half-written audit
    state, a partial revert, a botched merge — and must crash on load
    rather than silently propagate into the gate's decision. Per
    CLAUDE.md Tier-1 doctrine: "wrong type = crash, NULL where
    unexpected = crash, invalid enum value = crash."

    The invariants enforced:

    1. ``judge_verdict == OVERRIDDEN_BY_OPERATOR`` implies
       ``judge_model_verdict is not None`` — an override entry must
       record what the underlying model said so the meta-metric
       "override-rate-by-underlying-verdict" remains queryable; an
       override with no recorded model verdict is fabrication of "we
       overrode something" without recording what.
    2. Non-override entries with ``judge_verdict is not None`` must have
       ``judge_model_verdict is None`` — for a non-override entry the
       model's verdict and the entry's verdict are identical by
       construction; carrying a divergent value fabricates a divergence
       signal that doesn't exist.
    3. ``judge_verdict is not None`` implies recorded_at + model +
       policy hash + rationale are also non-None — these fields are
       atomic; a partial write is corruption.
    4. ``judge_verdict is None`` implies every other ``judge_*`` field
       and every binding field (``file_fingerprint``,
       ``scope_fingerprint``, ``ast_path``, ``judge_signature_version``)
       is also None — a pre-judge-era entry MUST NOT carry stray model
       metadata or binding fields; that would be evidence of partial
       revert / merge corruption.
    5. ``judge_verdict`` and ``judge_model_verdict`` must never be
       ``BLOCKED``. ``BLOCKED`` is the in-memory runtime verdict the
       cicd-judge gate uses to reject a candidate suppression; by
       contract, a ``BLOCKED`` verdict means the entry was NOT written.
       Defense-in-depth against ``_optional_judge_verdict``: even if the
       loader were ever loosened, the atomic validator must still catch
       the corrupt shape.
    6. ``judge_verdict is not None`` implies ``judge_rationale`` is
       non-empty after whitespace strip. The rationale is the "why" of
       the audit record; an empty or whitespace-only rationale is an
       audit-broken verdict. Defense-in-depth against ``_optional_string``:
       even if that helper were loosened to accept empty strings, the
       atomic validator must still catch the corrupt shape.

    7. ``judge_policy_hash`` must be a ``sha256:<64 lowercase hex>``
       digest. The hash records which static judge policy block the
       verdict interpreted so later CLAUDE.md / policy drift cannot make
       old verdicts uninterpretable.

    Invariant 8 (C8-3 binding co-presence, version-aware): when
    ``judge_verdict`` is set, ``ast_path`` and the version's source
    fingerprint must both be present, and the *other* version's
    fingerprint must be absent. The binding field is selected by
    ``judge_signature_version``: v1 (or absent version) binds
    ``file_fingerprint`` (whole-file SHA-256) and must not carry
    ``scope_fingerprint``; v2 binds ``scope_fingerprint`` (enclosing-scope
    AST fingerprint) and must not carry ``file_fingerprint``. These fields
    bind the judge quartet to the source the judge actually inspected;
    without them the quartet is transplantable (copy from a safe entry
    onto an entry keyed at dangerous code; loader can't tell the
    difference), and a stray cross-version fingerprint advertises a
    binding the signed payload did not sign. The live-source recompute for
    v1 is performed by :func:`_verify_source_binding_at_load` in
    ``_parse_allow_hits`` (load-time, catches cross-file transplant and
    source drift) and by :func:`verify_entry_binding_against_finding` at
    match time (catches in-file AST-node transplant). A v2 entry has no
    whole-file load-time recompute: :func:`_verify_source_binding_at_load`
    checks only that the bound source file exists, and the v2
    scope_fingerprint is verified at match time by
    :func:`verify_entry_binding_against_finding`, which compares the
    persisted scope_fingerprint against the value the scanner stamped onto
    the live finding (drift ⇒ re-justify; an empty live value ⇒ crash, since
    an unverifiable binding must never silently pass).
    Both verifications require the fields' presence, which this invariant
    guarantees.

    ``judge_metadata_signature`` is deliberately verified in
    ``_parse_allow_hits`` only when ``source_root`` is provided, because
    report-only loaders do not necessarily have access to the deployment
    HMAC key. Shape validation still treats it as part of the judge
    cluster: a pre-judge entry must not carry a stray signature, and a
    present signature must have a v1 or v2 prefix + 64-hex digest shape.
    """
    # Invariant 5 (defense-in-depth vs _optional_judge_verdict): neither
    # judge_verdict nor judge_model_verdict may be BLOCKED on a persisted
    # entry. BLOCKED is an in-memory runtime verdict; persisted BLOCKED
    # is corruption.
    if entry.judge_verdict is JudgeVerdict.BLOCKED:
        raise ValueError(
            f"{context}: judge_verdict is BLOCKED; BLOCKED is an in-memory runtime "
            "verdict that means the entry was rejected and NOT written. A persisted "
            "BLOCKED entry is corruption (botched hand-edit, partial revert, tampering)."
        )
    if entry.judge_model_verdict is JudgeVerdict.BLOCKED and entry.judge_verdict is not JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
        # An override entry legitimately carries judge_model_verdict=BLOCKED
        # to record what the model originally said before the operator
        # overrode. Outside that one shape, a BLOCKED in this field is corrupt.
        raise ValueError(
            f"{context}: judge_model_verdict is BLOCKED but judge_verdict is "
            f"{entry.judge_verdict.value if entry.judge_verdict is not None else 'None'!r}; "
            "BLOCKED on judge_model_verdict is only valid when the entry is "
            "OVERRIDDEN_BY_OPERATOR (recording what the model said pre-override)."
        )

    # Invariant 4: pre-judge entries are fully empty in the judge cluster
    # AND must not carry binding fields (file_fingerprint / ast_path).
    # Binding fields are written ONLY by ``justify`` alongside a judge
    # verdict; their presence on a verdict-less entry is corruption from
    # the same class of partial revert / merge accident.
    if entry.judge_verdict is None:
        stray: list[str] = []
        if entry.judge_recorded_at is not None:
            stray.append("judge_recorded_at")
        if entry.judge_model is not None:
            stray.append("judge_model")
        if entry.judge_rationale is not None:
            stray.append("judge_rationale")
        if entry.judge_confidence is not None:
            stray.append("judge_confidence")
        if entry.judge_model_verdict is not None:
            stray.append("judge_model_verdict")
        if entry.judge_policy_hash is not None:
            stray.append("judge_policy_hash")
        if entry.file_fingerprint is not None:
            stray.append("file_fingerprint")
        if entry.scope_fingerprint is not None:
            stray.append("scope_fingerprint")
        if entry.judge_transport is not None:
            stray.append("judge_transport")
        if entry.judge_signature_version is not None:
            stray.append("judge_signature_version")
        if entry.ast_path is not None:
            stray.append("ast_path")
        if entry.judge_metadata_signature is not None:
            stray.append("judge_metadata_signature")
        if stray:
            raise ValueError(
                f"{context}: judge_verdict is absent but other judge metadata is present "
                f"({', '.join(stray)}); pre-judge entries must omit every judge_* field "
                "and every binding field. "
                "This shape indicates a partial revert or merge corruption."
            )
        return

    if entry.judge_metadata_signature is not None:
        _validate_judge_metadata_signature_shape(entry.judge_metadata_signature, context=context)
    if entry.judge_policy_hash is not None:
        _validate_judge_policy_hash_shape(entry.judge_policy_hash, context=context)

    # Invariant 3: post-judge entries record the full quartet.
    missing: list[str] = []
    if entry.judge_recorded_at is None:
        missing.append("judge_recorded_at")
    if entry.judge_model is None:
        missing.append("judge_model")
    if entry.judge_rationale is None:
        missing.append("judge_rationale")
    if entry.judge_policy_hash is None:
        missing.append("judge_policy_hash")
    if missing:
        raise ValueError(
            f"{context}: judge_verdict is set ({entry.judge_verdict.value!r}) but the "
            f"required companion fields are missing ({', '.join(missing)}); the judge-"
            "metadata cluster (judge_verdict + judge_recorded_at + judge_model + "
            "judge_rationale + judge_policy_hash) is atomic. A partial write is corruption."
        )

    # Invariant 8 (C8-3 binding co-presence): post-judge entries must
    # carry their version's binding field plus ast_path. Without them the
    # quartet is transplantable — an attacker can copy the verdict +
    # rationale + model + policy hash + recorded_at from a safe entry onto
    # an entry keyed at dangerous code, and the gate has no way to detect
    # the rebind. The binding field is version-specific: v1 (or absent
    # version) binds the whole-file ``file_fingerprint``; v2 binds the
    # enclosing-scope ``scope_fingerprint``. An entry must carry exactly
    # the field its version signs and must NOT carry the other — a v1
    # entry with a stray scope_fingerprint, or a v2 entry with a stray
    # file_fingerprint, is corruption (the signed payload would bind one
    # field while the entry advertises the other). The load-time and
    # match-time binding checks downstream require these fields' presence,
    # which this invariant guarantees.
    version = entry.judge_signature_version if entry.judge_signature_version is not None else 1
    missing_binding: list[str] = []
    if entry.ast_path is None:
        missing_binding.append("ast_path")
    if version == 2:
        if entry.scope_fingerprint is None:
            missing_binding.append("scope_fingerprint")
        # ``judge_transport`` is a v2-only signed binding field (which transport
        # produced the verdict). It is inside the signed v2 payload, so its
        # absence means the entry cannot be re-verified — require it.
        if entry.judge_transport is None:
            missing_binding.append("judge_transport")
        if entry.file_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is 2 but file_fingerprint is present; "
                "v2 entries bind via scope_fingerprint and must not carry the v1 file_fingerprint."
            )
    else:
        if entry.file_fingerprint is None:
            missing_binding.append("file_fingerprint")
        if entry.scope_fingerprint is not None:
            raise ValueError(
                f"{context}: judge_signature_version is absent/1 but scope_fingerprint is present; "
                "v1 entries bind via file_fingerprint. A scope_fingerprint on a v1 entry is corruption."
            )
        # ``judge_transport`` is signed into the v2 payload only; a v1 payload
        # never bound it. Its presence on a v1 entry is a partial revert / merge
        # corruption (the entry advertises a field its signature does not cover).
        if entry.judge_transport is not None:
            raise ValueError(
                f"{context}: judge_signature_version is absent/1 but judge_transport is present; "
                "judge_transport is a v2-only signed field. A judge_transport on a v1 entry is corruption."
            )
    if missing_binding:
        raise ValueError(
            f"{context}: judge_verdict is set ({entry.judge_verdict.value!r}) but the "
            f"required binding fields are missing ({', '.join(missing_binding)}); "
            "judge-gated entries must record their version's source fingerprint "
            "(v1: file_fingerprint, the SHA-256 of the source file; v2: "
            "scope_fingerprint, the AST fingerprint of the enclosing scope) and "
            "ast_path (the AST address of the finding the judge accepted) so the gate "
            "can detect quartet transplant and source drift. An entry whose verdict "
            "cannot be bound to the code it judged is audit-broken."
        )

    # Invariant 7: rationale must be non-empty after whitespace strip.
    # _optional_string already rejects empty strings at parse-time, but a
    # whitespace-only rationale ("   " or "\n\n") would slip past it. An
    # empty/whitespace rationale destroys the "why" of the audit record.
    # ``entry.judge_rationale`` is non-None here per invariant 3 above.
    if entry.judge_rationale is None or not entry.judge_rationale.strip():
        raise ValueError(
            f"{context}: judge_verdict is set ({entry.judge_verdict.value!r}) but "
            "judge_rationale is empty or whitespace-only; a judge verdict without a "
            "rationale is audit-broken (the 'why' is missing)."
        )
    if entry.judge_confidence is not None and not 0.0 <= entry.judge_confidence <= 1.0:
        raise ValueError(f"{context}: judge_confidence must be between 0.0 and 1.0; got {entry.judge_confidence!r}")

    # Invariants 1 and 2: judge_model_verdict's presence is gated by
    # whether this is an override entry.
    if entry.judge_verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR:
        if entry.judge_model_verdict is None:
            raise ValueError(
                f"{context}: judge_verdict is OVERRIDDEN_BY_OPERATOR but "
                "judge_model_verdict is absent; override entries must record what the "
                "underlying model said so override-rate-by-underlying-verdict remains "
                "queryable. An override with no recorded model verdict is fabrication."
            )
    else:
        if entry.judge_model_verdict is not None:
            raise ValueError(
                f"{context}: judge_verdict is {entry.judge_verdict.value!r} (non-"
                f"override) but judge_model_verdict is set "
                f"({entry.judge_model_verdict.value!r}); for non-override entries the "
                "model's verdict and the entry's verdict are identical by "
                "construction, so recording a separate judge_model_verdict "
                "fabricates a divergence that doesn't exist."
            )


def _validate_audit_review_context(entry: AllowlistEntry, *, context: str) -> None:
    """Crash if a post-judge audit review is attached to the wrong entry shape."""
    if entry.audit_review is None:
        return
    if entry.judge_verdict is not JudgeVerdict.ACCEPTED:
        actual = "None" if entry.judge_verdict is None else entry.judge_verdict.value
        raise ValueError(
            f"{context}.audit_review is present but judge_verdict is {actual!r}; "
            "audit_review records a human falsification of a prior "
            "judge_verdict=ACCEPTED decision and is invalid on pre-judge, "
            "BLOCKED, or OVERRIDDEN_BY_OPERATOR entries."
        )


def _parse_per_file_rules(data: dict[str, Any], *, valid_rule_ids: Collection[str], source_file: str) -> list[PerFileRule]:
    rules_raw = _list_value(data, "per_file_rules")
    rules: list[PerFileRule] = []
    for index, raw_rule in enumerate(rules_raw):
        item = _mapping_value(raw_rule, f"per_file_rules[{index}]")
        rule_ids = tuple(_string_list(item, "rules", context=f"per_file_rules[{index}]"))
        unknown = sorted(set(rule_ids).difference(valid_rule_ids))
        if unknown:
            raise ValueError(f"per_file_rules[{index}] uses unknown rule id(s): {', '.join(unknown)}")
        rules.append(
            PerFileRule(
                pattern=_required_string(item, "pattern", context=f"per_file_rules[{index}]"),
                rules=rule_ids,
                reason=_required_string(item, "reason", context=f"per_file_rules[{index}]"),
                expires=_optional_date(item, "expires", context=f"per_file_rules[{index}]"),
                max_hits=_optional_int(item, "max_hits", context=f"per_file_rules[{index}]"),
                source_file=source_file,
            )
        )
    return rules


def _mapping_or_empty(data: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        return {}
    return _mapping_value(data[key], key)


def _mapping_value(value: object, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _list_value(data: dict[str, Any], key: str) -> list[object]:
    if key not in data:
        return []
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _string_list(data: dict[str, Any], key: str, *, context: str) -> list[str]:
    raw = _list_value(data, key)
    values: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context}.{key}[{index}] must be a non-empty string")
        values.append(value)
    return values


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _required_audit_anchor_string(data: dict[str, Any], key: str, *, context: str) -> str:
    value = _required_string(data, key, context=context)
    if not is_substantive_audit_anchor(value):
        raise ValueError(
            f"{context}.{key} must be a substantive audit discriminator anchor; "
            f"got {value!r}. Values like 'x', '.', or whitespace cannot safely "
            "distinguish rotation-grandfathered entries."
        )
    return value


def is_substantive_audit_anchor(value: str) -> bool:
    """Return True if ``value`` can distinguish a rotation-grandfathered entry."""
    alnum_count = sum(1 for char in value if char.isalnum())
    return alnum_count >= _MIN_AUDIT_ANCHOR_ALNUM_CHARS


def _required_positive_int(data: dict[str, Any], key: str, *, context: str) -> int:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context}.{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{context}.{key} must be positive")
    return cast(int, value)


def _optional_date(data: dict[str, Any], key: str, *, context: str) -> date | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be YYYY-MM-DD, null, or absent")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be YYYY-MM-DD") from exc


def _optional_date_alias(data: dict[str, Any], primary: str, alias: str, *, context: str) -> date | None:
    if primary in data and alias in data:
        raise ValueError(f"{context} must not include both {primary!r} and {alias!r}")
    if primary in data:
        return _optional_date(data, primary, context=context)
    return _optional_date(data, alias, context=context)


def _optional_string(data: dict[str, Any], key: str, *, context: str) -> str | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"{context}.{key} must be a non-empty string, null, or absent")


def _optional_confidence(data: dict[str, Any], key: str, *, context: str) -> float | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{context}.{key} must be a number from 0.0 to 1.0, null, or absent")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{context}.{key} must be between 0.0 and 1.0")
    return confidence


def _optional_int(data: dict[str, Any], key: str, *, context: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    value = data[key]
    # Reject bool explicitly — Python's bool is an int subclass, so a YAML
    # `true`/`false` would otherwise parse as 1/0 and silently pass through.
    # At a Tier-3 boundary the audit posture is to reject ambiguous coercions.
    if isinstance(value, bool):
        raise ValueError(f"{context}.{key} must be an integer, not a boolean")
    if isinstance(value, int):
        return value
    raise ValueError(f"{context}.{key} must be an integer, null, or absent")


def _optional_signature_version(data: dict[str, Any], key: str, *, context: str) -> int | None:
    """Parse the optional judge signature version (1 or 2).

    Absent / null yield ``None`` (the pre-judge era and v1 legacy entries
    written before the version field existed are treated as v1 at the
    dispatch site). Any present value must be exactly 1 or 2 — an unknown
    version is corruption and crashes on load per Tier-1 doctrine.
    """
    value = _optional_int(data, key, context=context)
    if value is not None and value not in (1, 2):
        raise ValueError(f"{context}.{key} must be 1 or 2; got {value!r}")
    return value


def _optional_nonneg_int(data: dict[str, Any], key: str, *, context: str) -> int | None:
    """Like ``_optional_int`` but rejects negatives."""
    value = _optional_int(data, key, context=context)
    if value is not None and value < 0:
        raise ValueError(f"{context}.{key} must be non-negative")
    return value


def _bool_value(data: dict[str, Any], key: str, *, default: bool) -> bool:
    if key not in data:
        return default
    value = data[key]
    if not isinstance(value, bool):
        raise ValueError(f"defaults.{key} must be a boolean")
    return value


def _optional_judge_verdict(
    data: dict[str, Any],
    key: str,
    *,
    context: str,
    allow_blocked: bool,
    allow_operator_override: bool = True,
) -> JudgeVerdict | None:
    """Parse an optional judge-verdict field.

    Absent / null both yield ``None`` (the "pre-judge era" representation).
    Any other value MUST be one of the enum members, exact string match.
    Rejecting unknown values keeps the audit trail's set of recorded
    verdicts bounded to the schema; a YAML carrying a garbage verdict is
    a corruption signal that should crash on load, not silently round-
    trip.

    ``allow_blocked`` toggles whether the ``BLOCKED`` enum value is a
    legal disk-side representation. ``JudgeVerdict.BLOCKED`` is an
    in-memory runtime verdict the cicd-judge gate produces to reject a
    candidate suppression; by contract, a ``BLOCKED`` *entry* verdict
    means the entry was NOT written, so ``judge_verdict: BLOCKED`` on
    disk is corruption (botched hand-edit, partial revert, tampering)
    and must be rejected (``allow_blocked=False``). The same value on
    ``judge_model_verdict`` is legitimate on an OVERRIDDEN entry — it
    records what the model said before the operator overrode it — so
    that field's loader passes ``allow_blocked=True`` and the
    cross-field validity is checked in ``_validate_judge_metadata_atomic``.

    Per CLAUDE.md Tier-1 doctrine ("invalid enum value = crash"), reject
    loudly so corruption is visible. The mechanical enforcement here is
    what makes the docstring on :class:`JudgeVerdict.BLOCKED` ("Reserved
    for in-memory representation only") load-bearing rather than
    aspirational for the ``judge_verdict`` field.
    """
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string, null, or absent")
    try:
        verdict = JudgeVerdict(value)
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be one of {[m.value for m in JudgeVerdict]}; got {value!r}") from exc
    if verdict is JudgeVerdict.BLOCKED and not allow_blocked:
        raise ValueError(
            f"{context}.{key} is BLOCKED; BLOCKED is an in-memory runtime verdict that "
            "means the entry was rejected and NOT written. A BLOCKED value on disk for "
            "this field is corruption (botched hand-edit, partial revert, or tampering) "
            "and must not silently propagate into the gate's decision."
        )
    if verdict is JudgeVerdict.OVERRIDDEN_BY_OPERATOR and not allow_operator_override:
        raise ValueError(
            f"{context}.{key} is OVERRIDDEN_BY_OPERATOR; {key} records the model verdict "
            "only, so it may be ACCEPTED or BLOCKED but never the operator override action."
        )
    return verdict


def _required_audit_review_verdict(data: dict[str, Any], key: str, *, context: str) -> AuditReviewVerdict:
    if key not in data:
        raise ValueError(f"{context} must include {key!r}")
    value = data[key]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string")
    try:
        return AuditReviewVerdict(value)
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be one of {[m.value for m in AuditReviewVerdict]}; got {value!r}") from exc


def _required_datetime(data: dict[str, Any], key: str, *, context: str) -> datetime:
    if key not in data or data[key] is None:
        raise ValueError(f"{context} must include {key!r}")
    parsed = _optional_datetime(data, key, context=context)
    assert parsed is not None
    return parsed


def _optional_datetime(data: dict[str, Any], key: str, *, context: str) -> datetime | None:
    """Parse an optional ISO-8601 datetime field.

    Absent / null both yield ``None``. Strings are parsed via
    ``datetime.fromisoformat``; PyYAML may also pre-parse a timestamp
    to a ``datetime`` (when the YAML scalar is a bare ISO timestamp), in
    which case we accept it. Naive datetimes (no tzinfo) are rejected —
    the judge always records UTC-aware timestamps and a naive value is
    a corruption signal.
    """
    if key not in data or data[key] is None:
        return None
    value = data[key]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"{context}.{key} must be a timezone-aware ISO-8601 timestamp")
        return value
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be an ISO-8601 timestamp string, null, or absent")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{context}.{key} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{context}.{key} must include a timezone offset")
    return parsed
