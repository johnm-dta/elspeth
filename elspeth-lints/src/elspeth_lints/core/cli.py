"""Command-line interface for elspeth-lints."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import secrets
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth_lints.core.allowlist import AuditReviewVerdict, JudgeVerdict, is_substantive_audit_anchor
from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    walk_python_files,
)
from elspeth_lints.core.atomic_io import atomic_update_text
from elspeth_lints.core.emitters.github import render_github
from elspeth_lints.core.emitters.json import render_json
from elspeth_lints.core.emitters.sarif import render_sarif
from elspeth_lints.core.emitters.text import render_text
from elspeth_lints.core.protocols import Finding, Rule, RuleContext, RuleScope, Severity
from elspeth_lints.core.registry import DEFAULT_REGISTRY, RuleRegistry

if TYPE_CHECKING:
    from elspeth_lints.rules.trust_tier.tier_model.rotate import RotationPlan


JUSTIFY_RATIONALE_MAX_BYTES = 8 * 1024
OPERATOR_OVERRIDE_TOKEN_ENV = "ELSPETH_JUDGE_OVERRIDE_TOKEN"
OPERATOR_OVERRIDE_TOKEN_SHA256_ENV = "ELSPETH_JUDGE_OVERRIDE_TOKEN_SHA256"
OPERATOR_OVERRIDE_MIN_TOKEN_BYTES = 20


_MAX_AUDIT_IDENTITY_LENGTH = 200


def _non_empty_string(value: str) -> str:
    """Argparse ``type=`` callable that rejects empty / whitespace-only input.

    Used for ``justify --owner``: the owner field is the audit signal for
    who claimed responsibility for a suppression and must be a real
    identity. ``argparse`` swallows the raised ``ArgumentTypeError`` and
    prints the message inline with its standard "argument --owner:
    invalid value" frame, so the operator sees a single clear error and
    the call exits 2.
    """
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError(
            "must be a non-empty audit identity (e.g. an agent name or "
            "human operator name); empty / whitespace-only values are "
            "rejected because the owner field is the audit signal for "
            "who claimed responsibility for the suppression."
        )
    if any(char in value for char in ("\n", "\r", "\x85")):
        raise argparse.ArgumentTypeError(
            "must be a single-line audit identity; embedded line breaks cannot be represented safely in the inline YAML owner field."
        )
    if not is_substantive_audit_anchor(value):
        raise argparse.ArgumentTypeError(
            "must be a substantive audit identity with at least two "
            "alphanumeric characters; values like 'x', '.', '~', or '-' "
            "cannot safely distinguish rotation-grandfathered entries."
        )
    if len(value) > _MAX_AUDIT_IDENTITY_LENGTH:
        raise argparse.ArgumentTypeError(
            f"must be at most {_MAX_AUDIT_IDENTITY_LENGTH} characters; an audit "
            "identity is a short owner/reviewer name, not free text. An unbounded "
            "value bloats the entry, the grandfathering discriminator, and the "
            "text replayed verbatim into future judge prompts."
        )
    return value


def _positive_int(value: str) -> int:
    """Argparse ``type=`` callable that rejects non-positive integers."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _timezone_aware_datetime_arg(value: str) -> datetime:
    """Argparse ``type=`` callable that accepts only timezone-aware timestamps."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"must be a valid ISO-8601 timestamp; got {value!r}") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("must include a timezone offset")
    return parsed


def _bounded_rationale_string(value: str) -> str:
    """Argparse ``type=`` callable for operator-supplied judge rationales."""
    stripped = value.strip()
    if not stripped:
        raise argparse.ArgumentTypeError(
            "must be a non-empty rationale; empty / whitespace-only values would produce an audit entry with no operator explanation."
        )
    byte_count = len(value.encode("utf-8"))
    if byte_count > JUSTIFY_RATIONALE_MAX_BYTES:
        raise argparse.ArgumentTypeError(
            f"must be at most {JUSTIFY_RATIONALE_MAX_BYTES} bytes when UTF-8 "
            f"encoded; got {byte_count} bytes. Shorten the rationale and keep "
            "supporting detail in the linked issue or review notes."
        )
    return value


def _operator_override_authorization_error() -> str | None:
    """Return a refusal reason when ``--operator-override`` is not authorized.

    The override token is secret; the SHA-256 fingerprint is non-secret policy
    material supplied by CI/operator environment. Requiring both prevents the
    CLI flag alone from becoming the authority while keeping the token out of
    command lines, YAML entries, and logs.
    """

    token = os.environ.get(OPERATOR_OVERRIDE_TOKEN_ENV)
    if token is None or not token.strip():
        return f"{OPERATOR_OVERRIDE_TOKEN_ENV} is not set; --operator-override requires an operator-controlled token in the environment."
    if len(token.encode("utf-8")) < OPERATOR_OVERRIDE_MIN_TOKEN_BYTES:
        return f"{OPERATOR_OVERRIDE_TOKEN_ENV} is too short; expected at least {OPERATOR_OVERRIDE_MIN_TOKEN_BYTES} UTF-8 bytes."

    expected_fingerprint = os.environ.get(OPERATOR_OVERRIDE_TOKEN_SHA256_ENV)
    if expected_fingerprint is None or not expected_fingerprint.strip():
        return f"{OPERATOR_OVERRIDE_TOKEN_SHA256_ENV} is not set; --operator-override requires the token's non-secret SHA-256 fingerprint."
    expected = expected_fingerprint.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        return f"{OPERATOR_OVERRIDE_TOKEN_SHA256_ENV} must be a 64-character lowercase or uppercase SHA-256 hex digest."

    actual = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not secrets.compare_digest(actual, expected):
        return f"{OPERATOR_OVERRIDE_TOKEN_ENV} fingerprint does not match {OPERATOR_OVERRIDE_TOKEN_SHA256_ENV}; refusing override."
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the elspeth-lints CLI."""
    DEFAULT_REGISTRY.load_builtin_rules()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _run_check(args, registry=DEFAULT_REGISTRY)
    if args.command == "dump-edges":
        return _run_dump_edges(args)
    if args.command == "rotate":
        return _run_rotate(args)
    if args.command == "justify":
        return _run_justify(args)
    if args.command == "audit-verdict":
        return _run_audit_verdict(args)
    if args.command == "reaudit":
        return _run_reaudit(args)
    if args.command == "migrate-judge-scope":
        return _run_migrate_judge_scope(args)
    if args.command == "check-judge-coverage":
        return _run_check_judge_coverage(args)
    if args.command == "check-judge-quality":
        return _run_check_judge_quality(args)
    if args.command == "check-trust-boundary-diff":
        return _run_check_trust_boundary_diff(args)
    if args.command == "check-rotation-audit":
        return _run_check_rotation_audit(args)
    if args.command == "check-override-rate":
        return _run_check_override_rate(args)
    sys.stderr.write(f"Unknown command {args.command!r}\n")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elspeth-lints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run static-analysis rules")
    check.add_argument("--rules", default="nothing", help="Comma-separated rule ids, or 'nothing' for the empty skeleton run")
    check.add_argument("--rule-set", choices=("static", "full"), default="static")
    check.add_argument("--format", choices=("text", "json", "sarif", "github"), default="text")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument(
        "--repo-root",
        type=Path,
        help=(
            "Repository working tree root for rules that resolve repository-relative "
            "evidence paths. When omitted, rules derive it from --root using their "
            "documented fallback; when supplied, this explicit value wins."
        ),
    )
    check.add_argument(
        "--allowlist-dir",
        type=Path,
        help=(
            "Override every rule's per-rule allowlist directory with this one. "
            "Useful for shadow runs and cross-branch comparisons. "
            "When unset, each rule resolves its own per-rule default directory."
        ),
    )
    check.add_argument("--files", nargs="*", type=Path)

    rotate = subparsers.add_parser(
        "rotate",
        help="Rotate stale fingerprints in tier_model allowlist entries (mechanical, no judge)",
    )
    rotate.add_argument("--root", type=Path, required=True, help="Source tree to scan (e.g. src/elspeth)")
    rotate.add_argument(
        "--allowlist-dir",
        type=Path,
        required=True,
        help="Directory of per-module tier_model allowlist YAML files",
    )
    rotate.add_argument("--dry-run", action="store_true", help="Plan only; do not modify files")
    rotate.add_argument(
        "--rotation-log",
        type=Path,
        default=Path(".elspeth/rotations.log"),
        help=("JSONL audit manifest written on non-dry-run rotation applies. Default: .elspeth/rotations.log."),
    )
    rotate.add_argument(
        "--no-auto-pair-symmetric",
        dest="auto_pair_symmetric",
        action="store_false",
        default=True,
        help=(
            "Surface N:N prefix groups (same count of findings and entries, "
            "N>=2) as ambiguous instead of auto-pairing them by sorted "
            "fingerprint. Default is to auto-pair, which is safe for "
            "tier_model because its reason metadata is qname-level."
        ),
    )
    rotate.add_argument(
        "--remove-stale",
        dest="remove_stale",
        action="store_true",
        default=False,
        help=(
            "Remove stale allowlist entries (entries whose underlying "
            "violation site no longer exists). Default is to keep stale "
            "entries and surface them for operator-confirmed cleanup."
        ),
    )
    rotate.add_argument(
        "--no-remove-stale",
        dest="remove_stale",
        action="store_false",
        help=("Deprecated compatibility flag. Stale removal is already off by default; pass --remove-stale to opt into deletion."),
    )
    rotate.add_argument(
        "--accept-todo-debt",
        action="store_true",
        help=(
            "Allow apply even when the plan reports historical TODO-stub "
            "allowlist entries. Default is to refuse so placeholder debt "
            "cannot be carried through a rotation silently."
        ),
    )
    rotate.add_argument(
        "--format",
        dest="rotate_format",
        choices=("text", "json"),
        default="text",
        help="Output format for the rotation plan summary",
    )

    justify = subparsers.add_parser(
        "justify",
        help="Gate a proposed allowlist entry through the cicd-judge (Opus)",
    )
    justify.add_argument(
        "--root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source tree to scan (used to re-run the rule on --file-path)",
    )
    justify.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repository root for trust-boundary honesty rules that resolve "
            "repo-relative evidence such as test_ref nodeids. Defaults to the "
            "rule's source-root heuristic."
        ),
    )
    justify.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module allowlist YAML files where the entry will be written",
    )
    justify.add_argument(
        "--file-path",
        type=str,
        required=True,
        help="Source file (path relative to --root) containing the finding being suppressed",
    )
    justify.add_argument(
        "--rule",
        type=str,
        default="trust_tier.tier_model",
        help="Rule id whose finding is being suppressed",
    )
    justify.add_argument(
        "--symbol",
        type=str,
        required=True,
        help=(
            "Qualified symbol name that must uniquely identify one current finding "
            "(e.g. 'MyClass._method'). Use the literal '_module_' for module-scope "
            "findings (matches the canonical-key sentinel)."
        ),
    )
    justify.add_argument(
        "--fingerprint",
        type=str,
        default=None,
        help=(
            "Exact finding fingerprint used to disambiguate when --symbol "
            "matches multiple current findings. Accepts the bare fingerprint "
            "or fp=<fingerprint>."
        ),
    )
    justify.add_argument(
        "--rationale",
        type=_bounded_rationale_string,
        required=True,
        help=(f"Agent's proposed justification for the suppression (max {JUSTIFY_RATIONALE_MAX_BYTES} UTF-8 bytes)."),
    )
    justify.add_argument(
        "--owner",
        type=_non_empty_string,
        required=True,
        help=(
            "Audit identity (agent name or human operator) requesting this "
            "exemption. Recorded verbatim in the entry's `owner` field; this "
            "is the audit signal for who took responsibility for the "
            "suppression."
        ),
    )
    justify.add_argument(
        "--operator-override",
        action="store_true",
        help=(
            "Bypass the judge's verdict. The judge is still called (to record "
            "what the model would have said) but the entry is written with "
            "verdict=OVERRIDDEN_BY_OPERATOR. Requires "
            f"{OPERATOR_OVERRIDE_TOKEN_ENV} and "
            f"{OPERATOR_OVERRIDE_TOKEN_SHA256_ENV} to authorize the override."
        ),
    )
    justify.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help=(
            "Maximum completion tokens for the judge response. Defaults to "
            "the judge module's DEFAULT_JUDGE_MAX_TOKENS; increase when the "
            "provider returns finish_reason=length."
        ),
    )
    justify.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the proposed entry and the judge verdict; do not write",
    )
    justify.add_argument(
        "--format",
        dest="justify_format",
        choices=("text", "json"),
        default="text",
        help="Output format for the verdict",
    )

    audit_verdict = subparsers.add_parser(
        "audit-verdict",
        help="Attach a human post-review verdict to a judge-accepted allowlist entry",
    )
    audit_verdict.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module allowlist YAML files containing the entry",
    )
    audit_verdict.add_argument(
        "--key",
        required=True,
        help="Exact allow_hits key whose prior ACCEPTED judge verdict is being reviewed",
    )
    audit_verdict.add_argument(
        "--verdict",
        choices=tuple(verdict.value for verdict in AuditReviewVerdict),
        required=True,
        help="Post-review verdict to attach",
    )
    audit_verdict.add_argument(
        "--reviewer",
        type=_non_empty_string,
        required=True,
        help="Audit identity of the human/operator recording the review",
    )
    audit_verdict.add_argument(
        "--rationale",
        type=_bounded_rationale_string,
        required=True,
        help=(f"Reason the prior judge-accepted suppression is being marked wrong (max {JUSTIFY_RATIONALE_MAX_BYTES} UTF-8 bytes)."),
    )
    audit_verdict.add_argument(
        "--reviewed-at",
        type=_timezone_aware_datetime_arg,
        default=None,
        help="Timezone-aware ISO-8601 review timestamp. Defaults to current UTC.",
    )
    audit_verdict.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the review block; do not write",
    )

    reaudit = subparsers.add_parser(
        "reaudit",
        help=("Re-run the cicd-judge across existing allowlist entries to detect decay (read-only on YAML; emits a triage report)"),
    )
    reaudit.add_argument(
        "--root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source tree to scan for the entries' underlying findings",
    )
    reaudit.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module allowlist YAML files to reaudit",
    )
    reaudit.add_argument(
        "--rule",
        type=str,
        default="trust_tier.tier_model",
        help=("Rule-package selector (controls which scanner re-derives findings). Currently only 'trust_tier.tier_model' is supported."),
    )
    reaudit.add_argument(
        "--since",
        type=str,
        default=None,
        help=(
            "ISO-8601 date or timestamp; entries whose judge_recorded_at is at "
            "or after this point are skipped (their judgment is fresh; no decay). "
            "Pre-judge entries (no judge_recorded_at) are never filtered out by "
            "--since — they are gated by --include-pre-judge instead."
        ),
    )
    reaudit.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only reaudit the first N entries surviving the other filters. "
            "Use for incremental sweeps — the ~700-entry allowlist should "
            "not be re-judged in one pass."
        ),
    )
    reaudit.add_argument(
        "--max-calls",
        type=_positive_int,
        default=None,
        help=(
            "Stop this invocation after N actual judge calls. Unlike --limit, "
            "this is a spend guard: entries resolved before the judge boundary "
            "do not consume it, and an exhausted budget leaves the sweep "
            "incomplete so it can be resumed later."
        ),
    )
    reaudit.add_argument(
        "--include-pre-judge",
        action="store_true",
        help=(
            "Also reaudit entries with no stored judge verdict (the historical "
            "pre-judge corpus). Off by default because the pre-judge entry set "
            "is too large for routine sweeps."
        ),
    )
    reaudit.add_argument(
        "--format",
        dest="reaudit_format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Report format. Markdown is the most operator-readable for triage.",
    )
    reaudit.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write report to this path instead of stdout.",
    )
    # T6b crash-recovery surfaces — mutually exclusive with each other.
    # Without either flag, the command starts a fresh sweep and prints
    # the new run_id to stderr so the operator can recover from a crash.
    reaudit_recovery = reaudit.add_mutually_exclusive_group()
    reaudit_recovery.add_argument(
        "--resume",
        dest="resume_run_id",
        type=str,
        default=None,
        help=(
            "Resume a prior killed/crashed sweep by run_id. Reads the "
            "<allowlist-dir>/.reaudit-state/<run_id>.jsonl sidecar, skips "
            "entries already classified, continues from the first un-classified "
            "entry. Crashes if the allowlist or filter arguments have drifted "
            "between the original sweep and this resume."
        ),
    )
    reaudit_recovery.add_argument(
        "--render-incomplete",
        dest="render_incomplete_run_id",
        type=str,
        default=None,
        help=(
            "Render the partial report captured by a killed/crashed sweep "
            "without re-running the judge. Reads the sidecar at "
            "<allowlist-dir>/.reaudit-state/<run_id>.jsonl, reconstructs the "
            "outcomes, and renders via the same formatter as a normal run. "
            "Exits non-zero because the rendered sweep is incomplete by "
            "definition (a clean sweep would not need this flag)."
        ),
    )

    migrate_judge_scope = subparsers.add_parser(
        "migrate-judge-scope",
        help=(
            "Re-sign currently-valid v1 (file_fingerprint) judge-gated entries as "
            "v2 (scope_fingerprint) without re-running the LLM judge. OPERATOR-ONLY: "
            "writes signed metadata; requires ELSPETH_JUDGE_METADATA_HMAC_KEY (same "
            "custody constraint as `justify` — an agent may PROPOSE this invocation, "
            "only an operator-held environment runs and signs it)."
        ),
    )
    migrate_judge_scope.add_argument(
        "--root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source tree to scan for the entries' underlying findings",
    )
    migrate_judge_scope.add_argument(
        "--allowlist-dir",
        type=Path,
        default=Path("config/cicd/enforce_tier_model"),
        help="Directory of per-module allowlist YAML files to migrate in place",
    )
    migrate_judge_scope.add_argument(
        "--owner",
        type=_non_empty_string,
        required=True,
        help=(
            "Audit identity (operator) running the mechanical migration. Recorded "
            "verbatim in the report; the migration carries each entry's existing "
            "`owner` field forward unchanged (it does not rewrite ownership)."
        ),
    )
    migrate_judge_scope.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report what WOULD migrate; write nothing.",
    )

    dump_edges = subparsers.add_parser("dump-edges", help="Dump import edges for architecture review")
    dump_edges.add_argument("--root", type=Path, default=Path.cwd())
    dump_edges.add_argument("--format", choices=("json", "mermaid", "dot"), default="json")
    dump_edges.add_argument("--output", type=Path)
    dump_edges.add_argument("--include-layer", action="append", choices=("L0", "L1", "L2", "L3"))
    dump_edges.add_argument("--collapse-to-subsystem", dest="collapse_to_subsystem", action="store_true", default=True)
    dump_edges.add_argument("--no-collapse", dest="collapse_to_subsystem", action="store_false")
    dump_edges.add_argument("--no-timestamp", action="store_true", default=False)
    dump_edges.add_argument("--exclude", action="append", default=[])

    # CI gate: every new ``allow_hits`` entry in this PR must carry the
    # atomic judge metadata quartet. See ``judge_coverage.py`` docstring
    # for the convergent finding C1 context, the rotation-grandfather
    # policy, and the scope boundary (``allow_hits:`` is the only
    # judge-covered entry schema; non-empty legacy entry schemas are
    # reported as unrecognized rather than silently skipped).
    check_coverage = subparsers.add_parser(
        "check-judge-coverage",
        help=(
            "Fail if any new allowlist entry in this PR is missing judge metadata. "
            "Diffs HEAD against --baseline-ref; entries present in baseline (modulo "
            "fingerprint rotation) are grandfathered."
        ),
    )
    check_coverage.add_argument(
        "--baseline-ref",
        required=True,
        help=(
            "Git ref to diff HEAD against. CI sets this to the PR's merge-base "
            "(${{ github.event.pull_request.base.sha }}); local devs typically use "
            "origin/main. The ref is consumed by 'git ls-tree' and 'git show'; "
            "any rev-spec git can resolve is acceptable."
        ),
    )
    check_coverage.add_argument(
        "--allowlist-root",
        type=Path,
        default=Path("config/cicd"),
        help=(
            "Directory whose 'enforce_*' subdirectories are checked. Default: "
            "config/cicd. Non-empty legacy entry shapes without an 'allow_hits:' "
            "block produce UNRECOGNIZED_ENTRY_SHAPE violations."
        ),
    )
    check_coverage.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help=(
            "Repository working tree root. Required for git commands. Defaults to "
            "the current working directory (the repo root in standard CI usage)."
        ),
    )

    check_quality = subparsers.add_parser(
        "check-judge-quality",
        help=(
            "Run the labelled judge-quality corpus through the live cicd-judge "
            "and fail if verdict/decorator accuracy drops below threshold."
        ),
    )
    check_quality.add_argument(
        "--corpus",
        type=Path,
        default=Path("config/cicd/judge-quality-corpus/v1.jsonl"),
        help="Strict JSONL corpus of labelled judge-quality cases.",
    )
    check_quality.add_argument(
        "--min-accuracy",
        type=float,
        default=0.90,
        help="Minimum exact-match accuracy required, as a fraction in [0.0, 1.0]. Default: 0.90.",
    )
    check_quality.add_argument(
        "--min-cases",
        type=int,
        default=10,
        help="Minimum accepted corpus size. Default: 10.",
    )
    check_quality.add_argument(
        "--max-cases",
        type=int,
        default=30,
        help="Maximum accepted corpus size. Default: 30.",
    )
    check_quality.add_argument(
        "--model",
        default=None,
        help="OpenRouter model id. Defaults to the judge module's DEFAULT_JUDGE_MODEL.",
    )
    check_quality.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Maximum completion tokens for each judge response. Defaults to DEFAULT_JUDGE_MAX_TOKENS.",
    )
    check_quality.add_argument(
        "--format",
        dest="judge_quality_format",
        choices=("text", "json"),
        default="text",
        help="Output format for the quality report.",
    )

    check_trust_boundary_diff = subparsers.add_parser(
        "check-trust-boundary-diff",
        help=(
            "Report @trust_boundary decorators added in this PR. "
            "Diffs HEAD against --baseline-ref and exits 0 when the report "
            "is produced; this is a review-surfacing step, not an enforcing gate."
        ),
    )
    check_trust_boundary_diff.add_argument(
        "--baseline-ref",
        required=True,
        help="Git ref to diff HEAD against (typically the PR merge-base).",
    )
    check_trust_boundary_diff.add_argument(
        "--root",
        type=Path,
        default=Path("src/elspeth"),
        help="Source tree to scan for changed Python files.",
    )
    check_trust_boundary_diff.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository working tree root. Required for git commands.",
    )

    check_rotation_audit = subparsers.add_parser(
        "check-rotation-audit",
        help=("Fail if this PR rotates tier_model allowlist fingerprints without matching .elspeth/rotations.log manifest records."),
    )
    check_rotation_audit.add_argument(
        "--baseline-ref",
        required=True,
        help="Git ref to diff HEAD against (typically the PR merge-base).",
    )
    check_rotation_audit.add_argument(
        "--allowlist-root",
        type=Path,
        default=Path("config/cicd"),
        help="Directory whose enforce_* allowlist YAML files are checked.",
    )
    check_rotation_audit.add_argument(
        "--rotation-log",
        type=Path,
        default=Path(".elspeth/rotations.log"),
        help="JSONL manifest produced by elspeth-lints rotate.",
    )
    check_rotation_audit.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository working tree root. Required for git commands.",
    )

    # CI gate: override-rate drift (convergent finding C3).
    check_overrides = subparsers.add_parser(
        "check-override-rate",
        help=(
            "Fail if the rolling-window OVERRIDDEN_BY_OPERATOR rate across all "
            "allowlists exceeds --max-rate. Insufficient-data windows pass with a "
            "notice; see override_rate.py docstring for the small-sample policy."
        ),
    )
    check_overrides.add_argument(
        "--allowlist-root",
        type=Path,
        default=Path("config/cicd"),
        help="Directory whose 'enforce_*' subdirectories are aggregated.",
    )
    check_overrides.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling-window length in days (default: 30).",
    )
    check_overrides.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help=(
            "Pass the gate when fewer than this many judge-recorded entries fall "
            "inside the window (default: 10). Prevents small-N noise from "
            "tripping the rate threshold during the first weeks of operation."
        ),
    )
    check_overrides.add_argument(
        "--max-rate",
        type=float,
        default=0.10,
        help=(
            "Maximum tolerated override rate as a fraction in [0.0, 1.0] "
            "(default: 0.10 = 10%%). Computed as "
            "OVERRIDDEN_BY_OPERATOR / (ACCEPTED + OVERRIDDEN_BY_OPERATOR) "
            "across the window."
        ),
    )
    check_overrides.add_argument(
        "--max-overrides",
        type=int,
        default=None,
        help=(
            "Optional absolute cap on OVERRIDDEN_BY_OPERATOR entries inside "
            "the window. When set, the gate fails if the count exceeds this "
            "value even when the ratio remains below --max-rate."
        ),
    )
    check_overrides.add_argument(
        "--reference-time",
        type=str,
        default=None,
        help=(
            "Window anchor as a timezone-aware ISO-8601 timestamp (e.g. "
            "'2026-05-23T00:00:00Z'). Defaults to current UTC; override for "
            "reproducibility in CI replays."
        ),
    )
    check_overrides.add_argument(
        "--counter-snapshot",
        type=Path,
        default=None,
        help=(
            "Path to the hash-bound override-rate counter snapshot. Default: <allowlist-root>/.judge-metrics/override-rate-counters.json."
        ),
    )

    return parser


def _run_check(args: argparse.Namespace, *, registry: RuleRegistry) -> int:
    requested_tokens = _parse_rules(args.rules)
    if not requested_tokens:
        return _emit_findings([], output_format=args.format, rules=[])

    available = set(registry.ids())
    requested_rules = _expand_rule_tokens(requested_tokens, available)
    unknown = sorted(set(requested_rules).difference(available))
    if unknown:
        sys.stderr.write(f"Unknown rule id(s): {', '.join(unknown)}\n")
        return 2

    findings: list[Finding] = []
    allowlist_dir = args.allowlist_dir
    if allowlist_dir is not None and not allowlist_dir.is_dir():
        sys.stderr.write(f"--allowlist-dir: {allowlist_dir} is not a directory\n")
        return 2
    repo_root = getattr(args, "repo_root", None)
    if repo_root is not None and not repo_root.is_dir():
        sys.stderr.write(f"--repo-root: {repo_root} is not a directory\n")
        return 2
    context = RuleContext(root=args.root, allowlist_dir_override=allowlist_dir, repo_root=repo_root)
    selected_rules = [registry.get(rule_id) for rule_id in requested_rules]
    whole_repo_rules = [rule for rule in selected_rules if rule.scope == RuleScope.WHOLE_REPO]
    incremental_rules = [rule for rule in selected_rules if rule.scope == RuleScope.INCREMENTAL]

    empty_tree = ast.Module(body=[], type_ignores=[])
    for rule in whole_repo_rules:
        findings.extend(rule.analyze(empty_tree, args.root, context))

    if incremental_rules:
        explicit_files = tuple(args.files or ())
        if explicit_files:
            out_of_scope = _out_of_scope_explicit_files(explicit_files, root=args.root, rules=incremental_rules)
            if out_of_scope:
                for file_path in out_of_scope:
                    sys.stderr.write(
                        f"{file_path}: no selected incremental rule path_filter applies "
                        f"({', '.join(rule.id for rule in incremental_rules)})\n"
                    )
                return 2
        for item in walk_python_files(args.root, explicit_files or None):
            item_path = item.path
            applicable_rules = _rules_for_path(item_path, root=args.root, rules=incremental_rules)
            if not applicable_rules:
                continue
            if isinstance(item, PythonSyntaxError):
                findings.append(
                    Finding(
                        rule_id="parse-error",
                        file_path=str(item.path),
                        line=item.line,
                        column=item.column,
                        message=item.message,
                        fingerprint=f"syntax:{item.line}:{item.column}",
                    )
                )
                continue
            if isinstance(item, PythonFileReadError):
                # Surface I/O / decoding failures as per-file diagnostic
                # findings so the operator sees that the file was not
                # analysed. A silent ``continue`` here would mean a
                # permission-denied or invalid-UTF-8 file became an
                # invisible gap in coverage — exactly the failure mode
                # ticket C6-6 prevents on the upstream side (no
                # whole-scan abort) needs to remain visible at the
                # report surface. line/column are 0 because an I/O
                # failure has no position inside the (unreadable) file.
                findings.append(
                    Finding(
                        rule_id="read-error",
                        file_path=str(item.path),
                        line=0,
                        column=0,
                        message=f"{item.error_type}: {item.message}",
                        fingerprint=f"read:{item.error_type}",
                    )
                )
                continue
            findings.extend(_run_rules(item, applicable_rules, context=context))

    return _emit_findings(findings, output_format=args.format, rules=selected_rules)


def _parse_rules(raw: str) -> tuple[str, ...]:
    rules = tuple(part.strip() for part in raw.split(",") if part.strip())
    if rules == ("nothing",):
        return ()
    return rules


def _expand_rule_tokens(tokens: tuple[str, ...], available: set[str]) -> tuple[str, ...]:
    expanded: list[str] = []
    unknown: list[str] = []
    for token in tokens:
        if token.endswith("/*"):
            prefix = token[:-2].replace("/", ".")
            matches = sorted(rule_id for rule_id in available if rule_id.startswith(f"{prefix}."))
            if matches:
                expanded.extend(matches)
            else:
                unknown.append(token)
        else:
            expanded.append(token)
    return tuple(dict.fromkeys([*expanded, *unknown]))


def _run_rules(item: ParsedPythonFile, rules: list[Rule], *, context: RuleContext) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rules:
        findings.extend(rule.analyze(item.tree, item.path, context))
    return findings


def _out_of_scope_explicit_files(files: Sequence[Path], *, root: Path, rules: list[Rule]) -> list[str]:
    out_of_scope: list[str] = []
    for file_path in files:
        if not any(_path_matches_rule(file_path, root=root, rule=rule) for rule in rules):
            out_of_scope.append(_display_path(_candidate_path(root, file_path), root))
    return out_of_scope


def _rules_for_path(file_path: Path, *, root: Path, rules: list[Rule]) -> list[Rule]:
    return [rule for rule in rules if _path_matches_rule(file_path, root=root, rule=rule)]


def _path_matches_rule(file_path: Path, *, root: Path, rule: Rule) -> bool:
    return re.search(rule.metadata.path_filter, _display_path(_candidate_path(root, file_path), root)) is not None


def _candidate_path(root: Path, file_path: Path) -> Path:
    if file_path.is_absolute() or file_path.exists():
        return file_path
    return root / file_path


def _display_path(file_path: Path, root: Path) -> str:
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return file_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return file_path.as_posix()


def _run_rotate(args: argparse.Namespace) -> int:
    from elspeth_lints.rules.trust_tier.tier_model.rotate import apply_plan, scan_for_rotations

    if not args.root.is_dir():
        sys.stderr.write(f"--root: {args.root} is not a directory\n")
        return 2
    if not args.allowlist_dir.exists():
        sys.stderr.write(f"--allowlist-dir: {args.allowlist_dir} does not exist\n")
        return 2

    plan = scan_for_rotations(
        source_root=args.root,
        allowlist_path=args.allowlist_dir,
        allow_symmetric_pairing=args.auto_pair_symmetric,
    )

    # Exit-code policy:
    #   - 1 if ambiguity present (operator must resolve before re-running)
    #   - 1 if new findings present (auto-TODO-stub creation has been removed
    #     permanently; the agent must fix the code or write a real allowlist
    #     entry)
    #   - 0 otherwise
    exit_code = 1 if (plan.has_ambiguity or plan.has_new_findings) else 0

    if args.rotate_format == "json":
        import json

        payload = {
            "rotations": [{"old_key": r.old_key, "new_key": r.new_key, "source_file": r.entry_source_file} for r in plan.rotations],
            "ambiguous": [
                {
                    "prefix": g.prefix,
                    "finding_count": g.finding_count,
                    "entry_count": g.entry_count,
                    "entry_keys": list(g.entry_keys),
                    "finding_keys": list(g.finding_keys),
                }
                for g in plan.ambiguous
            ],
            "stale_entries": [
                {"key": s.key, "source_file": s.source_file, "owner": s.owner, "reason": s.reason} for s in plan.stale_entries
            ],
            "todo_entries": [{"key": s.key, "source_file": s.source_file, "owner": s.owner, "reason": s.reason} for s in plan.todo_entries],
            "new_findings": [
                {
                    "canonical_key": nf.canonical_key,
                    "file_path": nf.file_path,
                    "rule_id": nf.rule_id,
                    "line": nf.line,
                    "message": nf.message,
                }
                for nf in plan.new_findings
            ],
            "unchanged_count": plan.unchanged_count,
            "applied": {},
        }
        if not args.dry_run and (plan.rotations or (args.remove_stale and plan.stale_entries)):
            applied = apply_plan(
                plan,
                allowlist_dir=args.allowlist_dir,
                remove_stale=args.remove_stale,
                accept_todo_debt=args.accept_todo_debt,
                rotation_log_path=args.rotation_log,
            )
            payload["applied"] = {
                src: {"rotations_applied": r.rotations_applied, "stale_entries_removed": r.stale_entries_removed}
                for src, r in applied.items()
            }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return exit_code

    _emit_rotation_text_summary(plan)
    if plan.has_new_findings:
        _emit_new_findings_removal_notice(plan)
    if args.dry_run:
        return exit_code
    if plan.has_rotations or (args.remove_stale and plan.stale_entries):
        applied = apply_plan(
            plan,
            allowlist_dir=args.allowlist_dir,
            remove_stale=args.remove_stale,
            accept_todo_debt=args.accept_todo_debt,
            rotation_log_path=args.rotation_log,
        )
        sys.stdout.write("\nApplied:\n")
        for src, r in sorted(applied.items()):
            parts: list[str] = []
            if r.rotations_applied:
                parts.append(f"{r.rotations_applied} rotation(s)")
            if r.stale_entries_removed:
                parts.append(f"{r.stale_entries_removed} stale removal(s)")
            sys.stdout.write(f"  {src}: {', '.join(parts)}\n")
        sys.stdout.write(f"Rotation audit log: {args.rotation_log}\n")
    return exit_code


def _emit_new_findings_removal_notice(plan: RotationPlan) -> None:
    """Surface new findings + the explicit auto-TODO-stub removal notice.

    The prior tool auto-created ``owner: TODO`` allowlist entries for these.
    That behaviour was removed permanently because it silently added
    unreviewed exemptions, polluting the audit trail with placeholder
    rationale. The new tool surfaces the findings and gates the run with a
    non-zero exit so the agent fixes the code or writes a real allowlist
    entry with substantive justification.
    """
    sys.stdout.write(
        "\nNew findings (no matching allowlist entry):\n"
        "  Auto-creation of `owner: TODO` placeholder entries has been REMOVED\n"
        "  PERMANENTLY. For each finding below, the agent must either fix the\n"
        "  code or add a real allowlist entry with substantive owner/reason/\n"
        "  safety fields and an explicit expiry. The forthcoming `justify`\n"
        "  subcommand will gate manual entries through an LLM judge.\n\n"
    )
    for nf in plan.new_findings[:50]:
        sys.stdout.write(f"  {nf.file_path}:{nf.line}  {nf.rule_id}  {nf.message}\n")
    if len(plan.new_findings) > 50:
        sys.stdout.write(f"  ... and {len(plan.new_findings) - 50} more\n")


def _emit_rotation_text_summary(plan: RotationPlan) -> None:
    from elspeth_lints.rules.trust_tier.tier_model.rotate import fingerprint_of

    sys.stdout.write(f"Rotations:        {len(plan.rotations)}\n")
    sys.stdout.write(f"Ambiguous:        {len(plan.ambiguous)}\n")
    sys.stdout.write(f"Unchanged:        {plan.unchanged_count}\n")
    sys.stdout.write(f"New findings:     {plan.new_finding_count}  (judge path, not handled here)\n")
    sys.stdout.write(f"Stale entries:    {plan.stale_entry_count}  (operator-confirmed cleanup, not handled here)\n")
    sys.stdout.write(f"TODO-stub entries:{plan.todo_entry_count:>4}  (placeholder-rationale debt; judge review needed)\n")
    if plan.rotations:
        sys.stdout.write("\nProposed rotations:\n")
        preview = plan.rotations[:20]
        for rotation in preview:
            old_fp = fingerprint_of(rotation.old_key)
            new_fp = fingerprint_of(rotation.new_key)
            sys.stdout.write(f"  {rotation.entry_source_file}: ...:fp={old_fp} -> ...:fp={new_fp}\n")
        if len(plan.rotations) > len(preview):
            sys.stdout.write(f"  ... and {len(plan.rotations) - len(preview)} more\n")
    if plan.ambiguous:
        sys.stdout.write("\nAmbiguous groups (manual resolution required):\n")
        for group in plan.ambiguous:
            sys.stdout.write(f"  prefix={group.prefix}  ({group.finding_count} finding(s), {group.entry_count} entry/entries)\n")
    if plan.stale_entries:
        sys.stdout.write("\nStale entries (operator-confirmed cleanup, not auto-removed):\n")
        for stale in plan.stale_entries:
            sys.stdout.write(f"  {stale.source_file}: {stale.key}\n")
            sys.stdout.write(f"    owner={stale.owner}  reason={stale.reason}\n")


def _run_dump_edges(args: argparse.Namespace) -> int:
    from elspeth_lints.rules.trust_tier.tier_model.rule import (
        _LAYER_NAME_TO_INT,
        LAYER_NAMES,
        render_dump_edges_dot,
        render_dump_edges_json,
        render_dump_edges_mermaid,
        scan_dump_edges,
    )

    root = args.root.resolve()
    if not root.is_dir():
        sys.stderr.write(f"Error: {root} is not a directory\n")
        return 1
    if args.format in ("json", "dot") and args.output is None:
        sys.stderr.write(f"Error: --output is required for --format {args.format}\n")
        return 1

    layer_strs = args.include_layer or ["L3"]
    include_layers = frozenset(_LAYER_NAME_TO_INT[layer] for layer in layer_strs)
    nodes, edges, sccs = scan_dump_edges(
        root=root,
        include_layers=include_layers,
        collapse_to_subsystem=args.collapse_to_subsystem,
        exclude_patterns=args.exclude,
    )

    if args.format == "json":
        rendered = render_dump_edges_json(
            root=args.root,
            include_layers=include_layers,
            collapse_to_subsystem=args.collapse_to_subsystem,
            nodes=nodes,
            edges=edges,
            sccs=sccs,
            use_stable_placeholder=args.no_timestamp,
        )
    elif args.format == "mermaid":
        rendered = render_dump_edges_mermaid(nodes, edges)
    elif args.format == "dot":
        rendered = render_dump_edges_dot(nodes, edges)
    else:
        raise ValueError(f"unknown dump-edges format: {args.format}")

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")

    if sccs:
        sys.stderr.write(
            f"WARNING: {len(sccs)} non-trivial strongly-connected component(s) detected at "
            f"{','.join(sorted(LAYER_NAMES[layer] for layer in include_layers))}.\n"
        )
        if args.output is not None:
            sys.stderr.write(f"         See {args.output} stats.scc_count for details.\n")
    return 0


def _run_justify(args: argparse.Namespace) -> int:
    """Drive the cicd-judge gate for one proposed allowlist entry.

    Re-runs the tier_model rule against ``--file-path`` to locate the
    finding matching ``--symbol``, extracts the surrounding code,
    calls the judge, and (unless ``--dry-run``) writes the entry to the
    appropriate per-module YAML under ``--allowlist-dir``.
    """
    # Lazy imports: the judge module pulls in the anthropic SDK on
    # ``call_judge``, the tier_model rule is heavy. Keep the CLI
    # responsive on subcommands that don't need them.
    from elspeth_lints.core.allowlist import JudgeVerdict, _judge_metadata_hmac_key
    from elspeth_lints.core.judge import (
        DEFAULT_JUDGE_MAX_TOKENS,
        DEFAULT_JUDGE_MODEL,
        JUDGE_EXCERPT_CONTEXT_LINES,
        JudgeConfigurationError,
        JudgeContractError,
        JudgeRequest,
        JudgeResponse,
        JudgeTransportError,
        call_judge,
    )
    from elspeth_lints.core.source_excerpt import (
        SourceExcerptPathOutsideRootError,
        extract_safe_excerpt,
        resolve_safe_excerpt_path,
    )

    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.stderr.write(f"--root: {root} is not a directory\n")
        return 2
    repo_root: Path | None = args.repo_root.resolve() if args.repo_root is not None else None
    if repo_root is not None and not repo_root.is_dir():
        sys.stderr.write(f"--repo-root: {repo_root} is not a directory\n")
        return 2
    allowlist_dir: Path = args.allowlist_dir
    if not allowlist_dir.is_dir():
        sys.stderr.write(f"--allowlist-dir: {allowlist_dir} is not a directory\n")
        return 2
    if args.operator_override:
        authorization_error = _operator_override_authorization_error()
        if authorization_error is not None:
            sys.stderr.write(f"operator override refused: {authorization_error}\n")
            return 2

    # Path-containment gate (closes elspeth-9bbb9df9a5 / C1-2(c)). The
    # ``--file-path`` arg is operator-supplied here but the same code
    # path serves writes derived from allowlist YAML elsewhere, and the
    # principle is identical: the source-excerpt path is a Tier-3 trust
    # boundary because it determines what bytes are shipped to a
    # third-party LLM. ``resolve_safe_excerpt_path`` raises
    # ``SourceExcerptPathOutsideRootError`` if the resolved path escapes
    # ``--root`` (an absolute ``/etc/passwd`` or a relative
    # ``../../../etc/passwd``). The check runs before the scanner so
    # an attacker-supplied path is never even parsed by the rule, let
    # alone exfiltrated.
    candidate = Path(args.file_path) if Path(args.file_path).is_absolute() else (root / args.file_path)
    try:
        target_file = resolve_safe_excerpt_path(root=root, target_file=candidate)
    except FileNotFoundError:
        sys.stderr.write(f"--file-path: {args.file_path!r} does not exist under {root}\n")
        return 2
    except SourceExcerptPathOutsideRootError as exc:
        sys.stderr.write(f"--file-path security violation: {exc}\n")
        return 2

    try:
        symbol_tuple = _parse_symbol(args.symbol)
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    try:
        findings, package_rule_assertions, valid_rule_ids = _scan_single_file_findings_for_justify(
            target_file=target_file,
            root=root,
            repo_root=repo_root,
            asserted_rule=args.rule,
        )
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    symbol_matches = [f for f in findings if _finding_symbol_matches(f, symbol_tuple)]

    if not symbol_matches:
        sys.stderr.write(
            f"No findings on {args.file_path} match symbol {args.symbol!r}. "
            f"Scanned {len(findings)} finding(s) on that file. "
            "Either the symbol path is wrong, the file is clean, or the "
            "finding is already covered by a per_file_rule.\n"
        )
        return 2

    if args.rule in package_rule_assertions:
        matching = list(symbol_matches)
    else:
        matching = [f for f in symbol_matches if f.rule_id == args.rule]
        if not matching:
            reported = ", ".join(sorted({f.rule_id for f in symbol_matches}))
            sys.stderr.write(
                f"--rule mismatch: operator asserted {args.rule!r} but the scanner "
                f"reported {reported} for symbol {args.symbol!r} in "
                f"{args.file_path}. Either correct --rule to one of the reported "
                "rule ids (if you intended to suppress the finding the scanner "
                "actually flagged), or pick a different --symbol (if you intended "
                f"to suppress a {args.rule!r} finding elsewhere in this file). "
                "Refusing to write the entry — silently rebinding the operator's "
                "--rule to the scanner's rule_id would corrupt the audit attribution.\n"
            )
            return 2

    if args.fingerprint is not None:
        requested_fingerprint = args.fingerprint.strip().removeprefix("fp=")
        if not requested_fingerprint:
            sys.stderr.write("--fingerprint must be a non-empty finding fingerprint.\n")
            return 2
        fingerprint_candidates = list(matching)
        matching = [f for f in fingerprint_candidates if f.fingerprint == requested_fingerprint]
        if not matching:
            sys.stderr.write(
                f"No findings on {args.file_path} match symbol {args.symbol!r}, "
                f"rule {args.rule!r}, and fingerprint {requested_fingerprint!r}. "
                "Matching symbol/rule findings:\n"
            )
            for finding in fingerprint_candidates:
                sys.stderr.write(f"  {_finding_canonical_key(finding)}  ({finding.rule_id} at line {finding.line})\n")
            return 2

    if len(matching) > 1:
        sys.stderr.write(
            f"Ambiguous: {len(matching)} findings on {args.file_path} match "
            f"symbol {args.symbol!r}. The judge gate requires a unique "
            "finding per entry. Pass --fingerprint with one of the listed "
            "fingerprints, narrow the symbol path, or run `elspeth-lints rotate` "
            "first if these are stale fingerprints. "
            "Matching findings:\n"
        )
        for finding in matching:
            sys.stderr.write(f"  {_finding_canonical_key(finding)}  ({finding.rule_id} at line {finding.line})\n")
        return 2

    finding = matching[0]
    finding_key = _finding_canonical_key(finding)

    # Binding preconditions, hoisted BEFORE the paid judge call and the
    # source-excerpt read. Both accessors depend only on ``finding`` (just
    # bound above), not on the excerpt or the judge verdict, so we evaluate
    # them here to fail fast: a finding that can't bind a v2 entry is
    # unjustifiable regardless of what the judge would say.
    try:
        scope_fingerprint = _finding_scope_fingerprint(finding)
        ast_path = _finding_ast_path(finding)
    except ValueError as exc:
        # Fail-closed BEFORE the paid judge call: a finding that can't bind
        # a v2 entry (no scope_fingerprint — e.g. a trust_boundary
        # ``protocols.Finding``, which its scanner does not stamp — or an
        # ast_path-absent finding) is unjustifiable, so reject it before
        # spending a judge call or reading the source excerpt. We do NOT
        # fall back to a v1 whole-file binding, which would defeat the
        # migration; v2 justify is tier-model-only today.
        sys.stderr.write(f"Cannot justify finding: {exc}\n")
        return 2

    # Cross-check the operator-asserted --rule against the rule_id the
    # scanner actually reported for the chosen symbol. The default
    # ``trust_tier.tier_model`` is a rule-PACKAGE selector (the same
    # token reaudit accepts) and is the no-op case: it asserts only
    # that we're working inside the tier_model rule set, which is
    # already structurally true here. When the operator passes a
    # specific sub-rule id (e.g. ``--rule R5``), it must match
    # ``finding.rule_id`` exactly — otherwise the audit primitive lies
    # (operator says "I'm suppressing R5", scanner picked R1, judge
    # sees R1, the YAML entry binds to R1, and a future reaudit
    # against R5 misses it). Refuse rather than silently rebinding.
    # Closes elspeth-98c06d159f (C2-3).
    if args.rule not in package_rule_assertions and args.rule != finding.rule_id:
        sys.stderr.write(
            f"--rule mismatch: operator asserted {args.rule!r} but the scanner "
            f"reported {finding.rule_id!r} for symbol {args.symbol!r} in "
            f"{args.file_path}. Either correct --rule to {finding.rule_id!r} (if "
            f"you intended to suppress the finding the scanner actually flagged), "
            f"or pick a different --symbol (if you intended to suppress a "
            f"{args.rule!r} finding elsewhere in this file). Refusing to write the "
            "entry — silently rebinding the operator's --rule to the scanner's "
            "rule_id would corrupt the audit attribution.\n"
        )
        return 2

    if not args.dry_run:
        try:
            _judge_metadata_hmac_key()
        except ValueError as exc:
            sys.stderr.write(f"Judge metadata signature configuration error: {exc}\n")
            return 2

    # Source-excerpt secrets-scrubber gate (closes elspeth-9bbb9df9a5 /
    # C2-2). ``extract_safe_excerpt`` is the single chokepoint between
    # local source bytes and OpenRouter; it path-contains, reads, and
    # scrubs in one call. The scrubbed ``text`` is what enters the
    # judge prompt; the ``redactions`` tuple becomes an audit record on
    # the persisted YAML entry so "n bytes redacted for pattern Y" is
    # post-hoc inspectable. The path was already proved in-root above,
    # but extract_safe_excerpt re-validates as defense-in-depth: any
    # future caller that funnels through this helper alone is also
    # safe.
    safe_excerpt = extract_safe_excerpt(
        root=root,
        target_file=target_file,
        line=finding.line,
        context_lines=JUDGE_EXCERPT_CONTEXT_LINES,
    )
    try:
        rationale_duplicate_count, similar_entries = _find_similar_allowlist_entries(
            allowlist_dir=allowlist_dir,
            rationale=args.rationale,
            valid_rule_ids=valid_rule_ids,
            exclude_key=finding_key,
        )
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"allowlist similarity scan failed: {exc}\n")
        return 2
    from elspeth_lints.rules.trust_tier.tier_model.rule import describe_rule

    request = JudgeRequest(
        file_path=finding.file_path,
        rule_id=finding.rule_id,
        rule_definition=describe_rule(finding.rule_id),
        symbol=args.symbol,
        fingerprint=finding.fingerprint,
        rationale=args.rationale,
        surrounding_code=safe_excerpt.text,
        rationale_duplicate_count=rationale_duplicate_count,
        similar_entries=similar_entries,
    )

    try:
        response: JudgeResponse = call_judge(
            request,
            model_id=DEFAULT_JUDGE_MODEL,
            max_tokens=args.max_tokens or DEFAULT_JUDGE_MAX_TOKENS,
        )
    except JudgeConfigurationError as exc:
        sys.stderr.write(f"Judge configuration error: {exc}\n")
        return 2
    except JudgeTransportError as exc:
        sys.stderr.write(f"Judge transport error: {exc}\n")
        return 2
    except JudgeContractError as exc:
        sys.stderr.write(f"Judge contract error: {exc}\n")
        return 2

    # Resolve the verdict the entry will carry. If the operator
    # supplied --operator-override, the entry records
    # OVERRIDDEN_BY_OPERATOR regardless of what the model said. The
    # model's actual rationale text is preserved in `judge_rationale` so
    # the bypass is visible in audit.
    # When the operator overrides, the entry's verdict diverges from
    # the model's. We preserve both — judge_verdict carries the override
    # signal (entry-level), judge_model_verdict preserves what the model
    # said so override-rate-by-underlying-verdict is queryable. For non-
    # override entries the model's verdict and the entry's verdict are
    # identical; we record judge_model_verdict=None rather than duplicate
    # it (fabrication-decision test: don't write data that synthesises a
    # divergence that doesn't exist).
    if args.operator_override:
        write_verdict = JudgeVerdict.OVERRIDDEN_BY_OPERATOR
        model_verdict: JudgeVerdict | None = response.verdict
    else:
        write_verdict = response.verdict
        model_verdict = None

    # C8-3 binding (v2 scheme): bind the persisted entry to the
    # enclosing-scope AST fingerprint the judge inspected, not the whole
    # file. ``scope_fingerprint`` and ``ast_path`` were computed up-front
    # (the fail-closed precondition block above the judge call) because
    # they depend only on ``finding``. They survive edits elsewhere in the
    # same file that the v1 whole-file ``file_fingerprint`` did not (an
    # unrelated edit to a neighbouring function invalidated a v1 binding
    # and crashed the load — see the v2 migration). The single
    # read-and-hash ``extract_safe_excerpt`` performed remains the source
    # of truth for the per-file scrubber salt baked into
    # ``RedactionRecord.redacted_hash``; that salt is independent of the
    # binding scheme. The persisted entry stays cryptographically bound to
    # the scope + AST node the judge inspected; the loader/matcher pair
    # (allowlist.load_allowlist and
    # allowlist.verify_entry_binding_against_finding) reads the
    # scope_fingerprint back and asserts the binding still holds at match
    # time, closing the entry-transplant attack vector.
    target_yaml = _suggest_yaml_target(finding=finding, allowlist_dir=allowlist_dir)

    def build_signed_yaml_entry() -> str:
        return _build_yaml_entry_text(
            key=finding_key,
            owner=args.owner,
            reason=args.rationale,
            verdict=write_verdict,
            recorded_at=response.recorded_at,
            model_id=response.model_id,
            judge_rationale=response.judge_rationale,
            judge_confidence=response.confidence,
            policy_hash=response.policy_hash,
            model_verdict=model_verdict,
            scope_fingerprint=scope_fingerprint,
            judge_transport=response.judge_transport,
            ast_path=ast_path,
            excerpt_redactions=safe_excerpt.redactions,
        )

    yaml_entry = ""

    # BLOCKED without override is the terminal-failure branch: print the
    # judge's rationale + the model that produced it, do not write, exit
    # non-zero. This is the "judge does not fix" load-bearing constraint.
    if response.verdict == JudgeVerdict.BLOCKED and not args.operator_override:
        if not args.dry_run:
            _append_judge_decision_event_after_judge(
                allowlist_dir=allowlist_dir,
                finding=finding,
                effective_verdict=write_verdict,
                model_verdict=response.verdict,
                recorded_at=response.recorded_at,
                write_disposition="blocked_without_override",
            )
        _emit_justify_output(
            args=args,
            verdict=write_verdict,
            judge_response=response,
            target_yaml=target_yaml,
            yaml_entry=yaml_entry,
            wrote=False,
            blocked=True,
            excerpt_redactions=safe_excerpt.redactions,
        )
        return 1

    if args.dry_run:
        try:
            _judge_metadata_hmac_key()
        except ValueError:
            yaml_entry = ""
        else:
            yaml_entry = build_signed_yaml_entry()
        _emit_justify_output(
            args=args,
            verdict=write_verdict,
            judge_response=response,
            target_yaml=target_yaml,
            yaml_entry=yaml_entry,
            wrote=False,
            blocked=False,
            excerpt_redactions=safe_excerpt.redactions,
        )
        return 0

    yaml_entry = build_signed_yaml_entry()
    _append_entry_to_yaml(target_yaml, yaml_entry)
    _append_judge_decision_event_after_judge(
        allowlist_dir=allowlist_dir,
        finding=finding,
        effective_verdict=write_verdict,
        model_verdict=response.verdict,
        recorded_at=response.recorded_at,
        write_disposition="written",
    )
    _refresh_override_rate_counter_snapshot_after_allowlist_write(target_yaml)
    _emit_justify_output(
        args=args,
        verdict=write_verdict,
        judge_response=response,
        target_yaml=target_yaml,
        yaml_entry=yaml_entry,
        wrote=True,
        blocked=False,
        excerpt_redactions=safe_excerpt.redactions,
    )
    return 0


def _append_judge_decision_event_after_judge(
    *,
    allowlist_dir: Path,
    finding: Any,
    effective_verdict: JudgeVerdict,
    model_verdict: JudgeVerdict | None,
    recorded_at: datetime,
    write_disposition: str,
) -> None:
    from elspeth_lints.core.override_rate import OverrideRateError, append_judge_decision_event

    try:
        append_judge_decision_event(
            allowlist_dir,
            source_file=finding.file_path,
            entry_key=_finding_canonical_key(finding),
            rule_id=finding.rule_id,
            effective_verdict=effective_verdict,
            model_verdict=model_verdict,
            recorded_at=recorded_at,
            write_disposition=write_disposition,
        )
    except (OSError, OverrideRateError) as exc:
        sys.stderr.write(f"judge metrics: decision event append failed: {exc}\n")


def _refresh_override_rate_counter_snapshot_after_allowlist_write(target_yaml: Path) -> None:
    """Best-effort telemetry refresh after the allowlist audit write succeeds."""
    from elspeth_lints.core.override_rate import (
        OverrideRateError,
        default_counter_snapshot_path,
        write_override_rate_counter_snapshot,
    )

    # target_yaml is <allowlist-root>/<enforce_dir>/<file>.yaml for the
    # shipped CI layout. Custom one-off allowlist dirs are outside C3's
    # aggregation contract, so skip the snapshot rather than emitting a
    # misleading empty-root metric.
    if not target_yaml.parent.name.startswith("enforce_"):
        return
    allowlist_root = target_yaml.parent.parent
    snapshot_path = default_counter_snapshot_path(allowlist_root)
    try:
        write_override_rate_counter_snapshot(allowlist_root, snapshot_path=snapshot_path)
    except (OSError, OverrideRateError) as exc:
        sys.stderr.write(f"judge metrics: counter snapshot refresh failed: {exc}\n")
        return
    sys.stderr.write(f"judge metrics: refreshed counter snapshot {snapshot_path}\n")


def _run_audit_verdict(args: argparse.Namespace) -> int:
    """Attach a post-judge audit review to one accepted allowlist entry."""
    from elspeth_lints.core.allowlist import load_allowlist
    from elspeth_lints.rules.trust_tier.tier_model.rule import RULES as TIER_MODEL_RULES

    allowlist_dir = args.allowlist_dir
    if not allowlist_dir.is_dir():
        sys.stderr.write(f"audit-verdict: --allowlist-dir {allowlist_dir} is not a directory\n")
        return 2

    valid_rule_ids = frozenset(TIER_MODEL_RULES)
    try:
        allowlist = load_allowlist(allowlist_dir, valid_rule_ids=valid_rule_ids)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"audit-verdict: cannot load allowlist: {exc}\n")
        return 2

    matches = [entry for entry in allowlist.entries if entry.key == args.key]
    if not matches:
        sys.stderr.write(f"audit-verdict: no allow_hits entry found for key {args.key!r}\n")
        return 2
    if len(matches) > 1:
        sources = ", ".join(sorted(entry.source_file for entry in matches))
        sys.stderr.write(f"audit-verdict: key {args.key!r} appears in multiple allowlist files ({sources}); refusing ambiguous write\n")
        return 2

    entry = matches[0]
    if entry.judge_verdict is not JudgeVerdict.ACCEPTED:
        actual = "None" if entry.judge_verdict is None else entry.judge_verdict.value
        sys.stderr.write(
            "audit-verdict: audit_review can only be attached to "
            f"judge_verdict=ACCEPTED entries; key {entry.key!r} has judge_verdict={actual}.\n"
        )
        return 2

    verdict = AuditReviewVerdict(args.verdict)
    reviewed_at = args.reviewed_at or datetime.now(UTC).replace(microsecond=0)
    review_text = _build_audit_review_text(
        verdict=verdict,
        reviewer=args.reviewer,
        reviewed_at=reviewed_at,
        rationale=args.rationale,
    )
    target_yaml = allowlist_dir / entry.source_file

    if args.dry_run:
        sys.stdout.write(review_text)
        return 0

    try:
        _upsert_audit_review_in_yaml(target_yaml, entry_key=entry.key, review_text=review_text)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"audit-verdict: cannot write audit_review: {exc}\n")
        return 2

    sys.stdout.write(f"audit-verdict: recorded {verdict.value} for {entry.key} in {target_yaml}\n")
    return 0


def _parse_symbol(symbol_arg: str) -> tuple[str, ...]:
    """Convert ``--symbol`` (dot-joined) to a ``symbol_context`` tuple.

    The literal ``_module_`` maps to the empty tuple — that is the
    canonical-key sentinel used when a finding has no enclosing symbol.
    Any other value is split on ``.`` and each part must be non-empty.
    """
    if symbol_arg == "_module_":
        return ()
    parts = tuple(part for part in symbol_arg.split("."))
    if not parts or any(not part for part in parts):
        raise ValueError(f"--symbol: {symbol_arg!r} is not a valid dotted name")
    return parts


def _scan_single_file_findings(
    *,
    target_file: Path,
    root: Path,
    scan_file: Any,
    scan_layer_imports_file: Any,
) -> list[Any]:
    """Re-run both tier_model scanners against a single file.

    Merges the R1-R7 findings from ``scan_file`` with the layer-import
    violations + TC warnings from ``scan_layer_imports_file``. Mirrors
    the way ``scan_for_rotations`` combines them, so the symbol-match
    pass below sees the same set of findings the CI run would see.
    """
    findings: list[Any] = list(scan_file(target_file, root))
    layer_violations, layer_tc = scan_layer_imports_file(target_file, root)
    findings.extend(layer_violations)
    findings.extend(layer_tc)
    return findings


def _scan_single_file_findings_for_justify(
    *,
    target_file: Path,
    root: Path,
    repo_root: Path | None,
    asserted_rule: str,
) -> tuple[list[Any], frozenset[str], frozenset[str]]:
    """Return current findings for the rule family selected by ``justify``.

    ``justify`` historically served only ``trust_tier.tier_model``. The
    trust-boundary honesty gates now share the same judged allowlist protocol,
    so this dispatcher keeps the scanner choice explicit while preserving the
    existing exact ``--rule`` cross-check.
    """
    from elspeth_lints.core.ast_walker import parse_python_file
    from elspeth_lints.rules.trust_boundary.scope import rule as scope_rule
    from elspeth_lints.rules.trust_boundary.scope.metadata import RULE_DEAD, RULE_NOPARAM
    from elspeth_lints.rules.trust_boundary.scope.metadata import RULE_NONLITERAL as SCOPE_NONLITERAL
    from elspeth_lints.rules.trust_boundary.shared import display_path, repository_root
    from elspeth_lints.rules.trust_boundary.tests import rule as tests_rule
    from elspeth_lints.rules.trust_boundary.tests.metadata import (
        RULE_FILE_MISSING,
        RULE_FINGERPRINT_MISMATCH,
        RULE_FINGERPRINT_MISSING,
        RULE_FUNCTION_MISSING,
        RULE_INPUT_IRRELEVANT,
        RULE_INVARIANT_MISMATCH,
        RULE_MISSING,
        RULE_NOTFOUND,
        RULE_PARSE_ERROR,
        RULE_TOO_LARGE,
        RULE_WEAK,
    )
    from elspeth_lints.rules.trust_boundary.tests.metadata import (
        RULE_NONLITERAL as TESTS_NONLITERAL,
    )
    from elspeth_lints.rules.trust_boundary.tier import rule as tier_rule
    from elspeth_lints.rules.trust_boundary.tier.metadata import RULE_INVALID
    from elspeth_lints.rules.trust_boundary.tier.metadata import RULE_NONLITERAL as TIER_NONLITERAL
    from elspeth_lints.rules.trust_tier.tier_model.rule import RULES, scan_file, scan_layer_imports_file

    scope_rule_ids = frozenset({RULE_NOPARAM, RULE_DEAD, SCOPE_NONLITERAL})
    tests_rule_ids = frozenset(
        {
            RULE_MISSING,
            RULE_NOTFOUND,
            RULE_WEAK,
            TESTS_NONLITERAL,
            RULE_FILE_MISSING,
            RULE_PARSE_ERROR,
            RULE_FUNCTION_MISSING,
            RULE_TOO_LARGE,
            RULE_INVARIANT_MISMATCH,
            RULE_INPUT_IRRELEVANT,
            RULE_FINGERPRINT_MISSING,
            RULE_FINGERPRINT_MISMATCH,
        }
    )
    tier_rule_ids = frozenset({RULE_INVALID, TIER_NONLITERAL})
    trust_boundary_rule_ids = frozenset().union(scope_rule_ids, tests_rule_ids, tier_rule_ids)

    if asserted_rule == "trust_tier.tier_model" or asserted_rule in RULES:
        return (
            _scan_single_file_findings(
                target_file=target_file,
                root=root,
                scan_file=scan_file,
                scan_layer_imports_file=scan_layer_imports_file,
            ),
            frozenset({"trust_tier.tier_model"}),
            frozenset(RULES.keys()),
        )

    parsed = parse_python_file(target_file)
    if isinstance(parsed, PythonSyntaxError):
        raise ValueError(f"--file-path: {target_file} cannot be parsed at {parsed.line}:{parsed.column}: {parsed.message}")
    if isinstance(parsed, PythonFileReadError):
        raise ValueError(f"--file-path: {target_file} cannot be read: {parsed.error_type}: {parsed.message}")

    display = display_path(target_file, root)
    if asserted_rule == "trust_boundary.scope" or asserted_rule in scope_rule_ids:
        return (
            scope_rule.analyze_tree(parsed.tree, display),
            frozenset({"trust_boundary.scope"}),
            trust_boundary_rule_ids,
        )
    if asserted_rule == "trust_boundary.tests" or asserted_rule in tests_rule_ids:
        return (
            tests_rule.analyze_tree(parsed.tree, display, repo_root=repository_root(root, repo_root)),
            frozenset({"trust_boundary.tests"}),
            trust_boundary_rule_ids,
        )
    if asserted_rule == "trust_boundary.tier" or asserted_rule in tier_rule_ids:
        return (
            tier_rule.analyze_tree(parsed.tree, display),
            frozenset({"trust_boundary.tier"}),
            trust_boundary_rule_ids,
        )

    raise ValueError(
        f"--rule {asserted_rule!r} is not supported by justify; use a tier_model rule id, "
        "trust_tier.tier_model, a trust_boundary.* package id, or a concrete "
        "trust-boundary honesty-gate finding id."
    )


def _find_similar_allowlist_entries(
    *,
    allowlist_dir: Path,
    rationale: str,
    valid_rule_ids: frozenset[str],
    exclude_key: str,
    limit: int = 5,
) -> tuple[int, tuple[Any, ...]]:
    """Return exact duplicate-rationale context for the judge prompt.

    Exact normalized duplicates are a strong, auditable signal of
    copy/paste rationale drift. Fuzzier similarity can be added later
    without weakening this hard duplicate count.
    """
    from elspeth_lints.core.allowlist import load_allowlist
    from elspeth_lints.core.judge import SimilarAllowlistEntry

    normalized = _normalize_rationale_for_similarity(rationale)
    if not normalized:
        return 0, ()

    allowlist = load_allowlist(allowlist_dir, valid_rule_ids=valid_rule_ids)
    duplicates = [
        entry for entry in allowlist.entries if entry.key != exclude_key and _normalize_rationale_for_similarity(entry.reason) == normalized
    ]
    similar_entries = tuple(
        SimilarAllowlistEntry(
            key=entry.key,
            owner=entry.owner,
            reason_excerpt=_reason_excerpt(entry.reason),
        )
        for entry in duplicates[:limit]
    )
    return len(duplicates), similar_entries


def _normalize_rationale_for_similarity(text: str) -> str:
    """Normalize rationale text for exact duplicate detection."""
    return " ".join(text.casefold().split())


def _reason_excerpt(text: str, *, limit: int = 240) -> str:
    """Return a compact single-line rationale excerpt for prompt context."""
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def _finding_symbol_matches(finding: Any, symbol_tuple: tuple[str, ...]) -> bool:
    """Return True iff the finding's ``symbol_context`` equals the tuple."""
    return _finding_symbol_context(finding) == symbol_tuple


def _finding_symbol_context(finding: Any) -> tuple[str, ...]:
    raw = getattr(finding, "symbol_context", ())
    return tuple(raw)


def _finding_canonical_key(finding: Any) -> str:
    canonical_key = finding.canonical_key
    if callable(canonical_key):
        canonical_key = canonical_key()
    if not isinstance(canonical_key, str):
        raise TypeError(
            f"finding.canonical_key must be a string or zero-argument callable returning a string; got {type(canonical_key).__name__}"
        )
    return canonical_key


def _finding_ast_path(finding: Any) -> str:
    ast_path = getattr(finding, "ast_path", "")
    if not isinstance(ast_path, str) or not ast_path:
        raise ValueError(
            f"finding {_finding_canonical_key(finding)} has no ast_path; "
            "judge-gated allowlist entries must bind to the AST node the judge inspected."
        )
    return ast_path


def _finding_scope_fingerprint(finding: Any) -> str:
    """Return the finding's enclosing-scope fingerprint for a v2 binding.

    Tier-model findings carry ``scope_fingerprint`` (stamped by the scanner).
    A trust_boundary ``protocols.Finding`` does NOT — so v2 justify is
    TIER-MODEL-ONLY today; justifying a trust_boundary rule raises here
    (fail-closed) until that scanner stamps the field. Do not fabricate a
    value: an empty/absent scope_fingerprint cannot bind a v2 entry.
    """
    scope_fingerprint = getattr(finding, "scope_fingerprint", "")
    if not isinstance(scope_fingerprint, str) or not scope_fingerprint:
        raise ValueError(
            f"finding {_finding_canonical_key(finding)} has no scope_fingerprint; "
            "judge-gated v2 entries must bind to the enclosing scope the judge inspected. "
            "The scanner must stamp scope_fingerprint on every finding (v2 justify is "
            "tier-model-only until trust_boundary's scanner stamps the field)."
        )
    return scope_fingerprint


def _suggest_yaml_target(*, finding: Any, allowlist_dir: Path) -> Path:
    """Pick the per-module YAML file under ``--allowlist-dir`` for this finding.

    Mirrors ``rule._suggest_module_file``: the first path segment of
    the finding's file_path is the module key (``foo/bar.py`` ->
    ``foo.yaml``). Bare filenames (no ``/``) map to ``<stem>.yaml``,
    with ``cli*`` collapsing to ``cli.yaml``. If the resulting file
    does not exist yet, the caller's writer will create it.
    """
    file_path = finding.file_path
    if "/" not in file_path:
        stem = file_path.removesuffix(".py")
        if stem.startswith("cli"):
            return allowlist_dir / "cli.yaml"
        return allowlist_dir / f"{stem}.yaml"
    module = file_path.split("/", 1)[0]
    return allowlist_dir / f"{module}.yaml"


def _build_yaml_entry_text(
    *,
    key: str,
    owner: str,
    reason: str,
    verdict: Any,  # JudgeVerdict; typed Any to avoid a top-level import
    recorded_at: Any,  # datetime
    model_id: str,
    judge_rationale: str,
    judge_confidence: float | None,
    policy_hash: str,
    scope_fingerprint: str,
    judge_transport: str,
    ast_path: str,
    excerpt_redactions: tuple[Any, ...] = (),  # tuple[RedactionRecord, ...]
    model_verdict: Any = None,  # JudgeVerdict | None; populated only on override
) -> str:
    """Render one ``allow_hits`` entry as YAML text.

    We emit the YAML by hand (rather than ``yaml.dump``) to keep the
    formatting byte-identical to the surrounding file's style:
    ``- key: ...`` at col 0, child keys at 2-space indent, multi-line
    strings as YAML block scalars. ``yaml.dump`` would round-trip-strip
    file-level comments and re-quote already-unquoted scalars.

    The entry carries the judge-metadata fields and a 90-day expiry
    from ``recorded_at``. ``safety`` is intentionally a placeholder —
    the agent typically supplies it as part of the rationale text; a
    follow-up could split this off as a separate CLI arg.

    ``model_verdict`` is the model's verdict when it diverges from the
    entry's verdict (operator override). It is emitted as
    ``judge_model_verdict`` only when non-None; the absence of the
    field on a non-override entry is the honest representation of
    "model's verdict and entry's verdict agree, consult judge_verdict".
    """
    from datetime import timedelta  # local import to keep top-of-file footprint small

    from elspeth_lints.core.allowlist import compute_judge_metadata_signature

    # Writer-side parity with loader invariant 7 in
    # allowlist._validate_judge_metadata_atomic: a whitespace-only rationale
    # would be rejected at load time, but we refuse to persist it in the
    # first place so a corrupt entry never reaches the YAML on disk. Today
    # judge._required_str_field strips at parse-time, but the writer is the
    # last gate before the audit record lands; offensive-programming policy
    # says the gate should hold even if the upstream guard regresses.
    if not judge_rationale.strip():
        raise ValueError(
            "_build_yaml_entry_text: judge_rationale is empty or whitespace-only; "
            "a judge verdict without a rationale is audit-broken (the 'why' is "
            "missing) and would be rejected by the loader's invariant 7."
        )

    persisted_judge_confidence = _persisted_judge_confidence(judge_confidence)
    judge_metadata_signature = compute_judge_metadata_signature(
        key=key,
        signature_version=2,
        scope_fingerprint=scope_fingerprint,
        judge_transport=judge_transport,
        ast_path=ast_path,
        judge_verdict=verdict,
        judge_model_verdict=model_verdict,
        judge_recorded_at=recorded_at,
        judge_model=model_id,
        judge_rationale=judge_rationale,
        judge_policy_hash=policy_hash,
        judge_confidence=persisted_judge_confidence,
        judge_excerpt_redactions=excerpt_redactions,
    )
    expiry = (recorded_at + timedelta(days=90)).date()
    lines: list[str] = []
    lines.append(f"- key: {key}")
    lines.append(f"  owner: {_yaml_inline_scalar(owner)}")
    lines.append("  reason: |-")
    for reason_line in reason.splitlines() or [""]:
        lines.append(f"    {reason_line}")
    lines.append("  safety: |-")
    lines.append("    Suppression gated by cicd-judge; see judge_rationale below.")
    lines.append(f"  expires: '{expiry.isoformat()}'")
    lines.append(f"  judge_verdict: {verdict.value}")
    if model_verdict is not None:
        lines.append(f"  judge_model_verdict: {model_verdict.value}")
    lines.append(f"  judge_recorded_at: '{recorded_at.isoformat()}'")
    lines.append(f"  judge_model: {_yaml_inline_scalar(model_id)}")
    lines.append(f"  judge_policy_hash: {_yaml_inline_scalar(policy_hash)}")
    if persisted_judge_confidence is not None:
        lines.append(f"  judge_confidence: {persisted_judge_confidence!r}")
    lines.append("  judge_rationale: |-")
    for rationale_line in judge_rationale.splitlines() or [""]:
        lines.append(f"    {rationale_line}")
    # C8-3 binding fields (v2 scheme). ``scope_fingerprint`` is the
    # AST fingerprint of the enclosing scope the judge inspected (the
    # finding's containing function/class/module body), not the whole
    # file — so an unrelated edit elsewhere in the file no longer
    # invalidates this binding. ``ast_path`` is the AST field/index path
    # from the module root to the finding's subject node. Together they
    # bind the persisted entry to scope+AST so the loader/matcher pair
    # can detect entry transplant and scope drift. ``judge_signature_version``
    # selects the v2 binding scheme and is signed inside the HMAC payload
    # so a v1<->v2 flip is itself unforgeable. All are scalars; emit
    # inline (the version is a bare int).
    lines.append("  judge_signature_version: 2")
    lines.append(f"  scope_fingerprint: {_yaml_inline_scalar(scope_fingerprint)}")
    lines.append(f"  judge_transport: {_yaml_inline_scalar(judge_transport)}")
    lines.append(f"  ast_path: {_yaml_inline_scalar(ast_path)}")
    lines.append(f"  judge_metadata_signature: {_yaml_inline_scalar(judge_metadata_signature)}")
    # Excerpt-redaction audit record (closes elspeth-9bbb9df9a5 / C2-2).
    # The judge's prompt may have had inline secrets scrubbed by
    # ``source_excerpt.scrub_secrets`` before transit to OpenRouter;
    # we persist a per-pattern count + 16-char hash here so an auditor
    # can reconstruct what was redacted without re-shipping the
    # original bytes. Absence of the block means "scrubber ran clean".
    if excerpt_redactions:
        lines.append("  judge_excerpt_redactions:")
        for record in excerpt_redactions:
            lines.append(f"    - pattern: {_yaml_inline_scalar(record.pattern_name)}")
            lines.append(f"      byte_count: {record.byte_count}")
            lines.append(f"      redacted_hash: {_yaml_inline_scalar(record.redacted_hash)}")
    return "\n".join(lines) + "\n"


def _persisted_judge_confidence(judge_confidence: float | None) -> float | None:
    if judge_confidence is None:
        return None
    return float(repr(judge_confidence))


def _build_audit_review_text(
    *,
    verdict: AuditReviewVerdict,
    reviewer: str,
    reviewed_at: datetime,
    rationale: str,
) -> str:
    """Render the nested ``audit_review`` block for one allow_hits entry."""
    if not is_substantive_audit_anchor(reviewer):
        raise ValueError("_build_audit_review_text: reviewer must be a substantive audit identity")
    if not rationale.strip() or not is_substantive_audit_anchor(rationale):
        raise ValueError("_build_audit_review_text: rationale must be a substantive non-empty audit explanation")
    if reviewed_at.tzinfo is None:
        raise ValueError("_build_audit_review_text: reviewed_at must include a timezone")

    lines = [
        "  audit_review:",
        f"    verdict: {verdict.value}",
        f"    reviewer: {_yaml_inline_scalar(reviewer)}",
        f"    reviewed_at: {_yaml_inline_scalar(reviewed_at.isoformat())}",
        "    rationale: |-",
    ]
    for rationale_line in rationale.splitlines() or [""]:
        lines.append(f"      {rationale_line}")
    return "\n".join(lines) + "\n"


# YAML 1.1 implicit-resolver regexes lifted verbatim from PyYAML's
# ``yaml.resolver.Resolver`` registrations (site-packages/yaml/resolver.py).
# PyYAML's safe_load is the loader we round-trip through; any plain scalar
# that one of these patterns matches will be coerced from str to a non-str
# Python value on reload. For a Tier-1 audit field that operator-supplied
# strings flow into (e.g. ``--owner yes``), that silent type change is
# evidence corruption. The writer's job is to quote the string in those
# cases so the loader receives an explicit ``tag:yaml.org,2002:str``.
#
# Case matters: PyYAML's bool regex is NOT case-insensitive — only the
# three variants lower/Title/UPPER per word — so ``yEs`` round-trips fine
# unquoted. We replicate that exactly rather than over-quoting on a
# case-insensitive match.
_YAML11_BOOL_RE = re.compile(
    r"""^(?:yes|Yes|YES|no|No|NO
            |true|True|TRUE|false|False|FALSE
            |on|On|ON|off|Off|OFF)$""",
    re.VERBOSE,
)
_YAML11_NULL_RE = re.compile(
    r"""^(?: ~
            |null|Null|NULL
            | )$""",
    re.VERBOSE,
)
_YAML11_INT_RE = re.compile(
    r"""^(?:[-+]?0b[0-1_]+
            |[-+]?0[0-7_]+
            |[-+]?(?:0|[1-9][0-9_]*)
            |[-+]?0x[0-9a-fA-F_]+
            |[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$""",
    re.VERBOSE,
)
_YAML11_FLOAT_RE = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
            |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
    re.VERBOSE,
)
_YAML11_MERGE_RE = re.compile(r"^(?:<<)$")
_YAML11_VALUE_RE = re.compile(r"^(?:=)$")
_YAML11_TIMESTAMP_RE = re.compile(
    r"""^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]
            |[0-9][0-9][0-9][0-9] -[0-9][0-9]? -[0-9][0-9]?
             (?:[Tt]|[ \t]+)[0-9][0-9]?
             :[0-9][0-9] :[0-9][0-9] (?:\.[0-9]*)?
             (?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$""",
    re.VERBOSE,
)

_YAML11_IMPLICIT_NON_STR_RESOLVERS: tuple[re.Pattern[str], ...] = (
    _YAML11_BOOL_RE,
    _YAML11_NULL_RE,
    _YAML11_INT_RE,
    _YAML11_FLOAT_RE,
    _YAML11_MERGE_RE,
    _YAML11_VALUE_RE,
    _YAML11_TIMESTAMP_RE,
)

# Plain scalars that are syntactically legal characters in our narrow bare
# set but are not legal values in ``key: <value>`` position. PyYAML parses a
# bare ``-`` as a block-sequence entry and raises ``ScannerError`` for
# ``x: -``; quote it before it can corrupt Tier-1 audit YAML.
_YAML11_PLAIN_SCALAR_SYNTAX_SENTINELS = frozenset({"-"})


def _value_resolves_to_non_str(value: str) -> bool:
    """Return True iff PyYAML safe_load would coerce ``value`` (as a plain
    scalar) to something other than ``str``.

    The empty string is included because PyYAML's null resolver matches it
    (see the regex: alternation ``| `` with an empty alternative); an
    unquoted empty value reloads as ``None``.
    """
    if value == "":
        return True
    return any(pattern.match(value) is not None for pattern in _YAML11_IMPLICIT_NON_STR_RESOLVERS)


def _yaml_inline_scalar(value: str) -> str:
    """Quote a string for inline YAML scalar emission when it needs it.

    Two reasons to quote:

    1. The value contains characters outside the bare-scalar safe set
       (``[A-Za-z0-9._/-]``) — the historic reason this helper existed.
    2. The value, while bare-safe, would be coerced to a non-string by
       PyYAML's safe_load implicit resolvers — e.g. ``yes`` → ``True``,
       ``null`` → ``None``, ``42`` → ``42``, ``2024-01-01`` → ``date``.
       This is the C2-5 fix: an operator passing ``--owner yes`` must
       not have the audit record's ``owner`` field reload as the Python
       boolean ``True``.
    3. The value is a YAML plain-scalar syntax sentinel in mapping-value
       position. A single bare ``-`` is not a string value in ``x: -``;
       PyYAML treats it as a block-sequence indicator and rejects the
       document. This is the C2-6 single-hyphen round-trip fix.

    Quoting style is single-quoted with internal ``'`` doubled, per YAML
    1.1 §9.3.2. Single-quoted scalars cannot represent C0 control
    characters (NUL through US, excluding TAB/LF/CR per YAML 1.1 §5.3);
    rather than silently producing un-loadable YAML, we raise. Pathological
    inputs in a Tier-1 audit field are a programmer-error signal, not a
    user-data-fault to coerce around.
    """
    # PyYAML's Reader.NON_PRINTABLE pattern (yaml/reader.py) defines the
    # exact set of codepoints safe_load accepts. Anything outside it raises
    # ReaderError at load time with a stack trace that doesn't point at the
    # writer. We replicate the rejection at write time so the failure
    # surfaces with a message that names the audit field and the offending
    # codepoint. The accepted set is:
    #   TAB (\x09), LF (\x0A), CR (\x0D),
    #   \x20-\x7E (printable ASCII),
    #   \x85 (NEL), \xA0-퟿, -�,
    #   \U00010000-\U0010FFFF
    # All other codepoints are unrepresentable in a YAML 1.1 single-quoted
    # scalar that the loader will accept.
    # Inside the reader-accepted set, single-quoted *flow* (inline) scalars
    # further fold LF / CR / NEL to a single space at load time (YAML 1.1
    # §6.3.2 line-break normalisation); the bytes survive the reader but
    # are not preserved on round-trip. Reject both classes here so the
    # failure surfaces at write time with a message naming the codepoint,
    # rather than as a silent reload corruption or a downstream ReaderError
    # with an unhelpful stack trace.
    #
    # The multi-paragraph audit fields (``reason``, ``safety``,
    # ``judge_rationale``) legitimately contain newlines; they go through
    # block-scalar emission (``|-``) elsewhere in _build_yaml_entry_text
    # and never reach this helper.
    folded_line_breaks = {0x0A, 0x0D, 0x85}
    for ch in value:
        codepoint = ord(ch)
        reader_accepts = (
            ch in "\t\n\r"
            or 0x20 <= codepoint <= 0x7E
            or codepoint == 0x85
            or 0xA0 <= codepoint <= 0xD7FF
            or 0xE000 <= codepoint <= 0xFFFD
            or 0x10000 <= codepoint <= 0x10FFFF
        )
        if not reader_accepts:
            raise ValueError(
                f"_yaml_inline_scalar: value contains non-printable character "
                f"U+{codepoint:04X} which PyYAML's safe_load rejects; "
                f"refusing to emit corrupt audit data. Caller must "
                f"sanitise upstream."
            )
        if codepoint in folded_line_breaks:
            raise ValueError(
                f"_yaml_inline_scalar: value contains line-break character "
                f"U+{codepoint:04X} which YAML's single-quoted flow scalar "
                f"folds to a single space on reload; refusing to emit a "
                f"value that won't round-trip. Use a block scalar for "
                f"multi-line audit fields."
            )
    needs_quote = (
        not value
        or value in _YAML11_PLAIN_SCALAR_SYNTAX_SENTINELS
        or any(not (ch.isalnum() or ch in "._/-") for ch in value)
        or _value_resolves_to_non_str(value)
    )
    if not needs_quote:
        return value
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _entry_key_from_yaml_entry(entry_text: str) -> str:
    """Extract the canonical key from writer-produced allow_hits YAML."""
    for line in entry_text.splitlines():
        if line.startswith("- key: "):
            key = line.removeprefix("- key: ").strip()
            if key:
                return key
            break
    raise ValueError("_append_entry_to_yaml: entry_text must start with a non-empty '- key: ...' line")


def _allow_hit_entry_ranges(lines: list[str], *, start: int, end: int) -> list[tuple[int, int]]:
    """Return ``[start, end)`` ranges for top-level entries in allow_hits."""
    ranges: list[tuple[int, int]] = []
    idx = start
    while idx < end:
        if not lines[idx].startswith("- "):
            idx += 1
            continue
        entry_start = idx
        idx += 1
        while idx < end and not lines[idx].startswith("- "):
            idx += 1
        ranges.append((entry_start, idx))
    return ranges


def _is_allow_hits_block_line(line: str) -> bool:
    """Return True when a line still belongs to the ``allow_hits`` block."""
    return line.startswith("- ") or line.startswith(" ") or line.lstrip().startswith("#")


def _upsert_audit_review_in_yaml(target_yaml: Path, *, entry_key: str, review_text: str) -> None:
    """Insert or replace one nested ``audit_review`` block in an allowlist entry."""

    def upsert_in(current: str | None) -> str:
        if current is None:
            raise ValueError(f"{target_yaml}: allowlist YAML file is required")

        lines = current.splitlines(keepends=True)
        header_index = None
        for idx, line in enumerate(lines):
            if line.rstrip("\r\n") == "allow_hits:":
                header_index = idx
                break
        if header_index is None:
            raise ValueError(f"{target_yaml}: no allow_hits block found")

        block_end = len(lines)
        for idx in range(header_index + 1, len(lines)):
            line = lines[idx]
            if not line.strip():
                continue
            if _is_allow_hits_block_line(line):
                continue
            block_end = idx
            break

        key_line = f"- key: {entry_key}"
        matching_ranges = [
            (entry_start, entry_end)
            for entry_start, entry_end in _allow_hit_entry_ranges(lines, start=header_index + 1, end=block_end)
            if lines[entry_start].rstrip("\r\n") == key_line
        ]
        if not matching_ranges:
            raise ValueError(f"{target_yaml}: no allow_hits entry found for key {entry_key!r}")
        if len(matching_ranges) > 1:
            raise ValueError(f"{target_yaml}: duplicate allow_hits entries found for key {entry_key!r}")

        entry_start, entry_end = matching_ranges[0]
        entry_lines = lines[entry_start:entry_end]
        cleaned_entry: list[str] = []
        cursor = 0
        while cursor < len(entry_lines):
            line = entry_lines[cursor]
            if cursor > 0 and line.rstrip("\r\n") == "  audit_review:":
                cursor += 1
                while cursor < len(entry_lines) and (entry_lines[cursor].startswith("    ") or not entry_lines[cursor].strip()):
                    cursor += 1
                continue
            cleaned_entry.append(line)
            cursor += 1

        if cleaned_entry and not cleaned_entry[-1].endswith("\n"):
            cleaned_entry[-1] = f"{cleaned_entry[-1]}\n"
        cleaned_entry.append(review_text if review_text.endswith("\n") else f"{review_text}\n")
        new_lines = [*lines[:entry_start], *cleaned_entry, *lines[entry_end:]]
        return "".join(new_lines)

    atomic_update_text(target_yaml, upsert_in, encoding="utf-8", create_parent=False)


def _append_entry_to_yaml(target_yaml: Path, entry_text: str) -> None:
    """Insert one ``allow_hits`` entry into the per-module YAML file.

    Append-only at the END of the existing ``allow_hits:`` block. If
    the file does not exist, create it with an ``allow_hits:`` header.
    If the file exists but has no ``allow_hits:`` block, append one at
    the bottom. The full read → locate block → insert → write sequence
    runs under ``atomic_update_text`` so concurrent justify invocations
    cannot compute updates from the same old YAML. We deliberately do
    NOT round-trip through PyYAML — the per-module YAMLs carry
    multi-paragraph comments that ``yaml.dump`` would erase, and the
    rotate command's surgical text-edit approach is the established
    pattern in this codebase.
    """
    entry_key = _entry_key_from_yaml_entry(entry_text)

    def append_to(current: str | None) -> str:
        if current is None:
            return f"allow_hits:\n{entry_text}"

        lines = current.splitlines(keepends=True)

        # Locate the ``allow_hits:`` header line; if absent, append the
        # whole block at the bottom.
        header_index = None
        for idx, line in enumerate(lines):
            if line.rstrip("\r\n") == "allow_hits:":
                header_index = idx
                break

        if header_index is None:
            prefix = "" if not current or current.endswith("\n") else "\n"
            return current + f"{prefix}\nallow_hits:\n{entry_text}"

        # Find the end of the allow_hits block: the next line at col 0 that
        # is non-empty and not part of the block (i.e. starts a new top-
        # level key). Lines that start with ``- `` or are indented belong
        # to the block; blank lines and comments inside the block stay.
        block_end = len(lines)
        for idx in range(header_index + 1, len(lines)):
            line = lines[idx]
            if not line.strip():
                continue
            if _is_allow_hits_block_line(line):
                continue
            # Hit a new top-level key (e.g. ``defaults:``); the block ends here.
            block_end = idx
            break

        # Ensure the existing block ends with a newline before our insert.
        if block_end > header_index + 1 and not lines[block_end - 1].endswith("\n"):
            lines[block_end - 1] = lines[block_end - 1] + "\n"

        key_line = f"- key: {entry_key}"
        matching_ranges = [
            (entry_start, entry_end)
            for entry_start, entry_end in _allow_hit_entry_ranges(lines, start=header_index + 1, end=block_end)
            if lines[entry_start].rstrip("\r\n") == key_line
        ]
        if matching_ranges:
            replacement = entry_text if entry_text.endswith("\n") else f"{entry_text}\n"
            replaced: list[str] = []
            cursor = 0
            inserted = False
            for entry_start, entry_end in matching_ranges:
                replaced.extend(lines[cursor:entry_start])
                if not inserted:
                    replaced.append(replacement)
                    inserted = True
                cursor = entry_end
            replaced.extend(lines[cursor:])
            return "".join(replaced)

        new_lines = [*lines[:block_end], entry_text, *lines[block_end:]]
        return "".join(new_lines)

    atomic_update_text(target_yaml, append_to, encoding="utf-8", create_parent=True)


def _emit_justify_output(
    *,
    args: argparse.Namespace,
    verdict: Any,  # JudgeVerdict
    judge_response: Any,  # JudgeResponse
    target_yaml: Path,
    yaml_entry: str,
    wrote: bool,
    blocked: bool,
    excerpt_redactions: tuple[Any, ...] = (),  # tuple[RedactionRecord, ...]
) -> None:
    """Render the justify result as text or JSON to stdout.

    ``excerpt_redactions`` (closes elspeth-9bbb9df9a5 / C2-2) surfaces
    the per-pattern scrubber audit record to operator-visible output
    in both JSON and text formats. The redactions are already
    persisted into the YAML entry by ``_build_yaml_entry_text``; this
    surface is the immediate "what did the scrubber do on this call?"
    signal so the operator notices without re-reading the YAML.
    """
    # ``should_use_decorator`` is the structured "use @trust_boundary
    # instead" nudge from the judge. It is meaningful only paired with a
    # BLOCKED verdict (the parser in ``call_judge`` enforces that
    # invariant). The JSON output exposes it as a top-level field so
    # tooling can branch on the structured value; the text output adds
    # a human-readable BLOCKED-with-decorator message in addition to the
    # usual rationale.
    should_use_decorator = judge_response.should_use_decorator

    if args.justify_format == "json":
        import json

        payload = {
            "verdict": verdict.value,
            "model": judge_response.model_id,
            "confidence": judge_response.confidence,
            "policy_hash": judge_response.policy_hash,
            "recorded_at": judge_response.recorded_at.isoformat(),
            "judge_rationale": judge_response.judge_rationale,
            "should_use_decorator": should_use_decorator,
            "target_yaml": str(target_yaml),
            "wrote": wrote,
            "blocked": blocked,
            "dry_run": args.dry_run,
            "operator_override": args.operator_override,
            "proposed_entry": yaml_entry,
            # Cache accounting (OpenRouter -> Anthropic). cached may be
            # null when the provider didn't report a cached-tokens count
            # (caching off, or older transport without the field). 0
            # means caching was on but produced no hit on this call.
            "prompt_tokens_total": judge_response.prompt_tokens_total,
            "prompt_tokens_cached": judge_response.prompt_tokens_cached,
            "excerpt_redactions": [
                {
                    "pattern_name": r.pattern_name,
                    "byte_count": r.byte_count,
                    "redacted_hash": r.redacted_hash,
                }
                for r in excerpt_redactions
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    sys.stdout.write(f"Verdict:          {verdict.value}\n")
    sys.stdout.write(f"Judge model:      {judge_response.model_id}\n")
    sys.stdout.write(f"Confidence:       {judge_response.confidence:.2f}\n")
    sys.stdout.write(f"Policy hash:      {judge_response.policy_hash}\n")
    sys.stdout.write(f"Recorded at:      {judge_response.recorded_at.isoformat()}\n")
    sys.stdout.write(f"Target YAML:      {target_yaml}\n")
    sys.stdout.write(f"Operator override:{'  yes' if args.operator_override else '  no'}\n")
    sys.stdout.write(f"Dry run:          {'yes' if args.dry_run else 'no'}\n")
    sys.stdout.write(f"Wrote:            {'yes' if wrote else 'no'}\n")
    # Cache accounting (one-line, text-format only). Per the project
    # telemetry primacy order this is operator-facing CLI output, not
    # slog — auditors don't read it; the operator does, to see whether
    # the static policy block is cache-hitting across repeat calls.
    # ``cached=None`` is meaningfully distinct from ``cached=0`` (the
    # provider didn't report a count at all vs reported zero hits).
    sys.stdout.write(
        f"Cache:            prompt_tokens={judge_response.prompt_tokens_total} "
        f"cached={judge_response.prompt_tokens_cached if judge_response.prompt_tokens_cached is not None else 'n/a'}"
    )
    if judge_response.prompt_tokens_cached is not None and judge_response.prompt_tokens_total > 0:
        ratio = judge_response.prompt_tokens_cached / judge_response.prompt_tokens_total
        sys.stdout.write(f" ({ratio:.0%} hit)")
    sys.stdout.write("\n")
    if excerpt_redactions:
        sys.stdout.write(f"\nExcerpt redactions: {len(excerpt_redactions)}\n")
        for r in excerpt_redactions:
            sys.stdout.write(f"  - pattern={r.pattern_name}  bytes={r.byte_count}  hash={r.redacted_hash}\n")
    sys.stdout.write("\nJudge rationale:\n")
    for line in judge_response.judge_rationale.splitlines() or [""]:
        sys.stdout.write(f"  {line}\n")
    if blocked:
        if should_use_decorator is not None:
            sys.stdout.write(
                "\nBLOCKED: use the @trust_boundary decorator on the enclosing "
                f"function with source_param={should_use_decorator!r} instead "
                "of adding this allowlist entry. The decorator is the "
                "structural replacement for per-line YAML allowlist entries "
                "covering Tier-3 external-data trust boundaries; see "
                "src/elspeth/contracts/trust_boundary.py for the decorator "
                "signature and notes/cicd-judge-cli-prototype-plan.md "
                "(Pillar B) for the rationale.\n"
            )
        else:
            sys.stdout.write(
                "\nBLOCKED: the judge rejected this suppression. The judge does "
                "not propose fixes; remediation (refactor, broaden a per-file "
                "rule, abandon the suppression) is the agent's responsibility.\n"
            )
    else:
        sys.stdout.write("\nProposed entry:\n")
        for line in yaml_entry.splitlines():
            sys.stdout.write(f"  {line}\n")


def _emit_findings(findings: list[Finding], *, output_format: str, rules: list[Rule]) -> int:
    if output_format == "json":
        sys.stdout.write(render_json(findings))
    elif output_format == "text":
        sys.stdout.write(render_text(findings))
    elif output_format == "github":
        sys.stdout.write(render_github(findings))
    elif output_format == "sarif":
        sys.stdout.write(render_sarif(findings, metadata=[rule.metadata for rule in rules]))
    else:
        raise ValueError(f"unknown output format: {output_format}")
    return 1 if any(finding.severity is not Severity.NOTE for finding in findings) else 0


def _run_reaudit(args: argparse.Namespace) -> int:
    """Drive a reaudit (decay-sweep) pass over an existing allowlist.

    Lazy-imports the reaudit module so subcommands that don't reach for
    the judge / tier_model scanner don't pay their import cost.

    Three paths (mutually exclusive at argparse level):

    * ``--render-incomplete <run_id>`` — read the sidecar for a prior
      killed sweep and render its partial outcomes. No judge calls. No
      sidecar mutation. Exits non-zero because the rendered sweep is
      incomplete by definition.
    * ``--resume <run_id>`` — read the prior sidecar, skip already-
      classified entries, continue from the first un-classified entry.
      Appends to the SAME sidecar; writes the trailer on clean
      completion. Load + trailer-check + header drift validation all
      run inside the writer's flock-held window (T6c TOCTOU fix) — two
      concurrent ``--resume`` invocations can no longer both pass
      validation and then race on the lock.
    * (no flag) — fresh sweep. Generates a new run_id, prints it to
      stderr, creates a fresh sidecar, classifies every filtered entry.
    """
    from elspeth_lints.core.judge import JudgeConfigurationError
    from elspeth_lints.core.reaudit import (
        ReauditDivergence,
        ReauditError,
        reaudit_entries,
    )
    from elspeth_lints.core.reaudit_sidecar import (
        SidecarHeader,
        SidecarWriter,
        compute_allowlist_hash,
        generate_run_id,
        load_sidecar,
        report_from_loaded_sidecar,
        sidecar_path_for,
        validate_header_for_resume,
    )

    allowlist_dir: Path = args.allowlist_dir

    if args.render_incomplete_run_id is not None:
        sidecar_path = sidecar_path_for(allowlist_dir, args.render_incomplete_run_id)
        try:
            loaded = load_sidecar(sidecar_path)
        except ReauditError as exc:
            sys.stderr.write(f"reaudit error: {exc}\n")
            return 2
        report = report_from_loaded_sidecar(loaded)
        _write_report(report, args)
        # --render-incomplete always exits non-zero (1 if the sweep was
        # killed mid-process, also 1 if the sweep happened to complete
        # but the operator asked for the partial view anyway — they're
        # asking for a failure-mode surface, exit code matches intent).
        return 1

    since: Any = None
    if args.since is not None:
        since = _parse_since(args.since)
        if since is None:
            sys.stderr.write(f"--since: {args.since!r} is not a valid ISO-8601 date or timestamp\n")
            return 2

    # Normalised since-iso for sidecar header binding. The raw
    # ``--since`` string may carry millisecond precision the parser
    # doesn't surface back; we record the *parsed* value's ISO form so
    # the resume comparison sees a canonical representation.
    since_iso_for_header = since.isoformat() if since is not None else None

    from elspeth_lints.core.reaudit import ReauditOutcome

    is_resume = args.resume_run_id is not None
    pre_classified_keys: frozenset[str] | None = None
    pre_classified_outcomes: tuple[ReauditOutcome, ...] = ()
    # T6c TOCTOU fix: the resume path no longer touches the sidecar
    # outside the flock window. The writer's ``on_resume_locked``
    # callback runs load + trailer-check + header-drift-check AFTER the
    # exclusive lock is held, so two concurrent --resume processes can no
    # longer both pass validation, serialise on the lock, then both
    # append past each other's trailers. The state captured here is
    # populated by the callback when validation succeeds; if validation
    # raises, the writer releases the lock and the CLI surfaces the
    # error via the existing ReauditError branch.
    resume_state: dict[str, Any] = {}
    on_resume_locked: Callable[[Any], None] | None = None
    if is_resume:
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        sidecar_path = sidecar_path_for(allowlist_dir, args.resume_run_id)
        run_id = args.resume_run_id
        # Placeholder header — never written. Resume mode (append=True)
        # never writes a header line; the constructor needs a non-None
        # value but only the lock-held callback's loaded.header is
        # authoritative.
        header = SidecarHeader(
            run_id=run_id,
            started_at=_datetime.now(_UTC),
            total_entries=0,
            allowlist_path=str(allowlist_dir.resolve()),
            allowlist_hash="",
            rule_filter=args.rule,
            since_iso=since_iso_for_header,
            limit=args.limit,
            include_pre_judge=args.include_pre_judge,
        )

        def on_resume_locked(loaded: Any) -> None:
            if loaded.trailer is not None:
                raise ReauditError(
                    f"--resume {args.resume_run_id}: this run already completed "
                    f"(trailer present, {loaded.trailer.outcomes_written} outcomes "
                    "written). Render the existing report via --render-incomplete, "
                    "or start a fresh sweep with a new run_id."
                )
            validate_header_for_resume(
                header=loaded.header,
                allowlist_dir=allowlist_dir,
                rule_filter=args.rule,
                since_iso=since_iso_for_header,
                limit=args.limit,
                include_pre_judge=args.include_pre_judge,
            )
            resume_state["header"] = loaded.header
            resume_state["classified_keys"] = loaded.classified_keys
            resume_state["outcomes"] = loaded.outcomes

    else:
        run_id = generate_run_id()
        from datetime import UTC as _UTC
        from datetime import datetime as _datetime

        reference_time = _datetime.now(_UTC)
        # The header is constructed before we know ``total_entries`` —
        # which requires loading the allowlist and applying the filters.
        # We defer constructing the SidecarHeader until inside the
        # ``reaudit_entries`` boundary by computing the filtered count
        # here through a peek. Simpler to centralise: do the load up
        # front, count, then enter the writer.
        from elspeth_lints.core.allowlist import load_allowlist
        from elspeth_lints.core.reaudit import _apply_filters, _supported_rules, _valid_rule_ids_for

        supported = _supported_rules()
        if args.rule not in supported:
            sys.stderr.write(f"reaudit error: --rule {args.rule!r} is not supported by reaudit. Supported: {sorted(supported)}.\n")
            return 2
        try:
            valid_rule_ids = _valid_rule_ids_for(args.rule)
        except ReauditError as exc:
            sys.stderr.write(f"reaudit error: {exc}\n")
            return 2
        try:
            preview = load_allowlist(
                allowlist_dir,
                valid_rule_ids=valid_rule_ids,
                source_root=args.root.resolve(),
            )
        except (FileNotFoundError, NotADirectoryError, ValueError, ReauditError) as exc:
            sys.stderr.write(f"reaudit error: {exc}\n")
            return 2
        filtered_preview = _apply_filters(
            entries=preview.entries,
            valid_rule_ids=valid_rule_ids,
            include_pre_judge=args.include_pre_judge,
            since=since,
            limit=args.limit,
            reference_time=reference_time,
        )

        header = SidecarHeader(
            run_id=run_id,
            started_at=reference_time,
            total_entries=len(filtered_preview),
            allowlist_path=str(allowlist_dir.resolve()),
            allowlist_hash=compute_allowlist_hash(allowlist_dir),
            rule_filter=args.rule,
            since_iso=since_iso_for_header,
            limit=args.limit,
            include_pre_judge=args.include_pre_judge,
        )
        sys.stderr.write(
            f"reaudit: starting sweep run_id={run_id} ({header.total_entries} entries). "
            f"If the sweep is killed before completion, recover via "
            f"`elspeth-lints reaudit --resume {run_id}` or "
            f"`elspeth-lints reaudit --render-incomplete {run_id}`.\n"
        )
        sidecar_path = sidecar_path_for(allowlist_dir, run_id)

    try:
        with SidecarWriter(
            sidecar_path,
            header,
            append=is_resume,
            on_resume_locked=on_resume_locked,
        ) as writer:
            if is_resume:
                # Populated by the in-lock callback. If it had raised,
                # SidecarWriter.__enter__ would have re-raised before
                # this point and the outer except branch would handle
                # it. Reaching here implies validation succeeded.
                pre_classified_keys = resume_state["classified_keys"]
                pre_classified_outcomes = resume_state["outcomes"]
            report = reaudit_entries(
                root=args.root.resolve(),
                allowlist_dir=allowlist_dir,
                rule_filter=args.rule,
                since=since,
                limit=args.limit,
                include_pre_judge=args.include_pre_judge,
                max_calls=args.max_calls,
                sidecar_writer=writer,
                pre_classified_keys=pre_classified_keys,
                pre_classified_outcomes=pre_classified_outcomes,
                reference_time=(resume_state["header"].started_at if is_resume else header.started_at),
                progress_callback=_emit_reaudit_progress,
            )
            if report.entries_dispatched >= report.total_entries:
                writer.commit_trailer()
    except (ValueError, ReauditError) as exc:
        sys.stderr.write(f"reaudit error: {exc}\n")
        return 2
    except JudgeConfigurationError as exc:
        sys.stderr.write(f"Judge configuration error: {exc}\n")
        return 2

    _write_report(report, args)

    # Exit-code policy: 0 only when the sweep is complete and every reached
    # entry still agrees. Any non-STILL_AGREES divergence is gate-firing signal
    # for CI; exit 2 remains reserved for command/configuration errors above.
    non_agreeing_outcomes = sum(1 for outcome in report.outcomes if outcome.divergence is not ReauditDivergence.STILL_AGREES)
    incomplete = report.entries_dispatched < report.total_entries
    if non_agreeing_outcomes > 0 or incomplete:
        return 1
    return 0


def _emit_reaudit_progress(progress: Any) -> None:
    """Write one operator-facing cost-progress line for reaudit."""
    if progress.max_judge_calls is None:
        calls = str(progress.judge_calls_attempted)
    else:
        calls = f"{progress.judge_calls_attempted}/{progress.max_judge_calls}"
    cached = progress.prompt_tokens_cached if progress.prompt_tokens_cached is not None else "n/a"
    uncached = progress.prompt_tokens_uncached if progress.prompt_tokens_uncached is not None else "n/a"
    sys.stderr.write(
        "reaudit progress: "
        f"entries={progress.entries_dispatched}/{progress.total_entries} "
        f"judge_calls={calls} "
        f"prompt_tokens_total={progress.prompt_tokens_total} "
        f"prompt_tokens_cached={cached} "
        f"prompt_tokens_uncached={uncached}\n"
    )


def _run_migrate_judge_scope(args: argparse.Namespace) -> int:
    """Mechanically re-sign currently-valid v1 judge-gated entries as v2.

    v1 entries bind the whole-file ``file_fingerprint``: any byte edit
    anywhere in the source file invalidates the binding at load time and
    forces an operator-only re-sign. v2 entries bind only the enclosing-
    scope ``scope_fingerprint``, so an unrelated edit elsewhere in the file
    no longer churns the entry. This command migrates the byte-drifted-but-
    scope-stable v1 entries to v2 WITHOUT re-running the (paid) LLM judge:
    the judge already inspected and accepted this exact suppressed node, and
    the suppressed node is unchanged, so re-judging would add no information.

    Two INDEPENDENT gates run per v1 entry — they are not conflated:

    * **Integrity gate (always, first):** the entry's EXISTING v1
      ``judge_metadata_signature`` is verified with the operator HMAC key.
      A mismatch (or missing/malformed signature) means the persisted
      verdict/rationale/binding was edited without the key — i.e. tampering.
      A key-holder running this migration must NEVER mint a clean v2
      signature over content whose v1 signature was never checked, so any
      integrity failure STOPS THE WHOLE RUN (not a per-entry skip): the
      offending entry is reported as TAMPERED and nothing is written.

    * **Relevance gate:** the entry's canonical key must locate a live
      finding in a fresh scan of ``--root``. If it does, the suppressed
      node is unchanged (the judge's inspection still applies) and the
      entry is migrated. If it does not, the entry is already stale and is
      refused (re-justify required) — left untouched.

    We deliberately do NOT gate on byte-freshness (``file_fingerprint`` ==
    live source hash). That would refuse exactly the byte-drifted-but-scope-
    stable entries this command exists to relieve. Selection is
    ``judge_signature_version in (None, 1)``; the integrity gate proves no
    tampering; the relevance gate proves the suppressed node is unchanged.
    Byte equality is irrelevant to either.
    """
    from elspeth_lints.core.allowlist import (
        _file_path_from_canonical_key,
        _judge_metadata_hmac_key,
        _verify_judge_metadata_signature_at_load,
        compute_judge_metadata_signature,
        load_allowlist,
    )
    from elspeth_lints.core.source_excerpt import (
        SourceExcerptPathOutsideRootError,
        resolve_safe_excerpt_path,
    )
    from elspeth_lints.rules.trust_tier.tier_model.rule import (
        RULES,
        scan_file,
        scan_layer_imports_file,
    )

    root: Path = args.root.resolve()
    if not root.is_dir():
        sys.stderr.write(f"--root: {root} is not a directory\n")
        return 2
    allowlist_dir: Path = args.allowlist_dir
    if not allowlist_dir.is_dir():
        sys.stderr.write(f"--allowlist-dir: {allowlist_dir} is not a directory\n")
        return 2

    # Operator-only fail-closed gate, hoisted to the front (mirrors justify
    # at the top of its write path). Two reasons it MUST run before any
    # entry is inspected:
    #   1. The migration writes signed metadata; without the key it cannot
    #      sign and must not pretend to succeed.
    #   2. The integrity gate below depends on real HMAC recomputation.
    #      ``_verify_judge_metadata_signature_at_load`` has a documented
    #      escape hatch that no-ops when the key is absent AND verify-mode is
    #      shape-only (the fork-PR degradation). If we let the command run
    #      keyless, that escape hatch would silently turn the integrity gate
    #      into a no-op and we could launder tampering into a clean v2
    #      signature. Requiring the key here guarantees the gate actually
    #      recomputes the HMAC. (Even --dry-run requires it: a dry run that
    #      cannot verify integrity reports a verdict it did not actually
    #      check, which is dishonest audit output.)
    try:
        _judge_metadata_hmac_key()
    except ValueError as exc:
        sys.stderr.write(f"Judge metadata signature configuration error: {exc}\n")
        return 2

    valid_rule_ids = frozenset(RULES.keys())

    # Load WITHOUT source_root. With source_root set, the loader fires BOTH
    # the v1 file_fingerprint live-source gate AND the HMAC signature gate
    # (allowlist._parse_allow_hits: ``if source_root is not None and
    # judge_verdict is not None``). We are deliberately migrating byte-
    # drifted-but-scope-stable entries, so the file_fingerprint live gate
    # must NOT block us. But loading without source_root ALSO skips the HMAC
    # check — therefore this command MUST verify each v1 signature itself
    # (the integrity gate below) before re-signing, or it would launder
    # tampering into a clean v2 signature.
    try:
        allowlist = load_allowlist(allowlist_dir, valid_rule_ids=valid_rule_ids, source_root=None)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        sys.stderr.write(f"migrate-judge-scope error: {exc}\n")
        return 2

    # Select judge-gated v1 entries. Pre-judge entries (judge_verdict is
    # None) carry no signature and no binding — they are neither migratable
    # nor tamperable here, so they are skipped silently. A v1 entry is one
    # whose judge_signature_version is unset (legacy default == 1) or
    # explicitly 1; v2 entries are already migrated.
    v1_entries = [
        entry
        for entry in allowlist.entries
        if entry.judge_verdict is not None and (entry.judge_signature_version is None or entry.judge_signature_version == 1)
    ]

    # Per-file finding cache for the relevance gate: scanning a file is
    # expensive, and several entries may target the same file. Keyed by the
    # resolved target file.
    findings_by_file: dict[Path, list[Any]] = {}

    def _findings_for_file(target_file: Path) -> list[Any]:
        cached = findings_by_file.get(target_file)
        if cached is None:
            cached = _scan_single_file_findings(
                target_file=target_file,
                root=root,
                scan_file=scan_file,
                scan_layer_imports_file=scan_layer_imports_file,
            )
            findings_by_file[target_file] = cached
        return cached

    migrated: list[str] = []  # canonical keys migrated (or would-migrate under --dry-run)
    refused: list[tuple[str, str]] = []  # (key, reason) for relevance-gate refusals

    # ---- Pass 1: integrity gate over EVERY selected v1 entry ----------------
    # The integrity gate runs to completion across all entries BEFORE any
    # write happens. This makes tamper-detection atomic: on tampering the
    # allowlist is left wholly untouched (zero writes), so the "NO entries
    # were written" claim below is true in production (where the command
    # iterates the whole ~hundreds-of-entry allowlist), not just in
    # single-entry tests. A one-pass integrity→write loop would re-sign and
    # write entries 1..N-1 before discovering tampering at entry N, then
    # falsely report that nothing was written — an audit-broken lie in a tool
    # held to the withstands-formal-inquiry bar.
    #
    # Each verify uses the EXISTING v1 signature and the operator key (present
    # in env, gated above so the HMAC recompute cannot be silently skipped).
    # Any ValueError — signature mismatch, missing signature, or malformed
    # shape — is an integrity failure: a key-holder must never re-sign content
    # whose prior signature did not verify (that would launder tampering into
    # a clean v2 signature). This is qualitatively different from the
    # relevance-gate's routine "stale" refusal: tampering is an attack /
    # corruption signal, not benign development drift.
    for entry in v1_entries:
        try:
            _verify_judge_metadata_signature_at_load(entry, context=f"migrate-judge-scope {entry.key!r}")
        except ValueError as exc:
            sys.stderr.write(
                f"migrate-judge-scope: TAMPERED entry refused — {entry.key!r}: {exc}\n"
                "The existing v1 judge_metadata_signature does not verify. Refusing to "
                "re-sign content whose prior signature was never validated (that would "
                "launder tampering into a clean v2 signature). Investigate the entry's "
                "edit history before any migration. Stopping the run; NO entries were "
                "written.\n"
            )
            return 1

    # ---- Pass 2: relevance gate + re-sign, then ONE atomic write per file ---
    # Reached only when every selected v1 entry's existing signature verified.
    # We resolve every entry's v2 rewrite spec first and group the specs by
    # their source YAML file; the actual writes happen in a final per-file
    # pass below, each under a single atomic_update_text. This gives per-file
    # atomicity: a mid-run crash cannot leave one YAML file half-rewritten
    # (worst case is a later file left untouched — a valid idempotent state a
    # re-run resumes from), and it scans/writes each file exactly once.
    specs_by_yaml: dict[Path, list[_V2RewriteSpec]] = {}
    for entry in v1_entries:
        # ---- Relevance gate ----------------------------------------------
        # Re-locate the suppressed finding by canonical key in a fresh scan of
        # the source tree. The file path is the key's leading segment; resolve
        # it through the path-containment guard (a forged ``../../etc/passwd``
        # key must not be scanned).
        try:
            file_path = _file_path_from_canonical_key(entry.key)
        except ValueError as exc:
            refused.append((entry.key, f"entry key is not in canonical form ({exc})"))
            continue
        candidate = root / file_path
        try:
            target_file = resolve_safe_excerpt_path(root=root, target_file=candidate)
        except FileNotFoundError:
            refused.append((entry.key, f"source file {file_path!r} no longer exists → re-justify required"))
            continue
        except SourceExcerptPathOutsideRootError as exc:
            refused.append((entry.key, f"source path escapes --root ({exc}) → re-justify required"))
            continue

        findings = _findings_for_file(target_file)
        matching = [f for f in findings if _finding_canonical_key(f) == entry.key]
        if not matching:
            refused.append((entry.key, "no matching finding → re-justify required"))
            continue
        if len(matching) > 1:
            # The canonical key includes the finding fingerprint, so a unique
            # key SHOULD match a unique finding. More than one is a scanner /
            # allowlist invariant violation: refuse rather than guess which
            # finding the judge inspected.
            refused.append((entry.key, f"{len(matching)} findings match this key (expected exactly one) → re-justify required"))
            continue
        finding = matching[0]

        # Both gates passed. Bind v2 to the LIVE finding's scope_fingerprint
        # and ast_path — these are the values the next CI run's match-time
        # check (verify_entry_binding_against_finding) will compare against.
        try:
            scope_fingerprint = _finding_scope_fingerprint(finding)
            ast_path = _finding_ast_path(finding)
        except ValueError as exc:
            refused.append((entry.key, f"cannot bind v2 entry: {exc} → re-justify required"))
            continue

        migrated.append(entry.key)
        if args.dry_run:
            continue

        # The judge-metadata cluster is co-present-or-co-absent (loader
        # invariant 8) and ``judge_verdict is not None`` was already filtered
        # for; ``_verify_judge_metadata_signature_at_load`` re-asserted every
        # member non-None above. Assert it here too so the invariant is
        # load-bearing at the signing site (offensive programming: a future
        # refactor that broke the filter would crash here, not silently sign a
        # None-bearing payload) and so the static types narrow.
        assert entry.judge_verdict is not None
        assert entry.judge_recorded_at is not None
        assert entry.judge_model is not None
        assert entry.judge_rationale is not None
        assert entry.judge_policy_hash is not None

        # Re-sign as v2 from the LOADED ENTRY's parsed audit fields, so the
        # non-binding lines on disk stay byte-identical and the reload
        # re-verifies. (verdict/model_verdict/recorded_at/model/rationale/
        # policy_hash/confidence/excerpt_redactions are unchanged audit data;
        # only the binding flips from file_fingerprint to scope_fingerprint.)
        # Backfill judge_transport="openrouter": every existing corpus entry was
        # OpenRouter-produced (the only transport before the agent-SDK work), so
        # this is a truthful backfill, not fabrication. It must be bound into the
        # re-signed v2 payload AND written to disk (below) for the reload to
        # re-verify.
        v2_signature = compute_judge_metadata_signature(
            key=entry.key,
            signature_version=2,
            scope_fingerprint=scope_fingerprint,
            judge_transport="openrouter",
            ast_path=ast_path,
            judge_verdict=entry.judge_verdict,
            judge_model_verdict=entry.judge_model_verdict,
            judge_recorded_at=entry.judge_recorded_at,
            judge_model=entry.judge_model,
            judge_rationale=entry.judge_rationale,
            judge_policy_hash=entry.judge_policy_hash,
            judge_confidence=entry.judge_confidence,
            judge_excerpt_redactions=entry.judge_excerpt_redactions,
        )
        target_yaml = allowlist_dir / entry.source_file
        specs_by_yaml.setdefault(target_yaml, []).append(
            _V2RewriteSpec(
                entry_key=entry.key,
                scope_fingerprint=scope_fingerprint,
                ast_path=ast_path,
                v2_signature=v2_signature,
            )
        )

    # Final write pass: one atomic write per source YAML file. Skipped under
    # --dry-run (specs_by_yaml stays empty: the dry-run branch ``continue``s
    # before any spec is appended).
    for target_yaml, specs in specs_by_yaml.items():
        _rewrite_v1_entries_as_v2_in_yaml(target_yaml, specs)

    _emit_migrate_report(args=args, migrated=migrated, refused=refused)

    # Exit-code policy: 1 if anything was refused (the integrity-gate
    # tampering path returns 1 above before reaching here). 0 only when every
    # selected v1 entry migrated cleanly (or, under --dry-run, would).
    return 1 if refused else 0


def _emit_migrate_report(
    *,
    args: argparse.Namespace,
    migrated: list[str],
    refused: list[tuple[str, str]],
) -> None:
    """Write the migrate-judge-scope summary.

    ``migrated`` lists the keys that were re-signed as v2 (or, under
    ``--dry-run``, would be) — the success surface, written to stdout.
    ``refused`` lists ``(key, reason)`` pairs for relevance-gate refusals
    (already-stale entries needing re-justify). Refusals are actionable
    NON-success output and the command exits non-zero on any refusal, so
    they go to STDERR — otherwise ``2>/dev/null`` would hide the very lines
    that explain why the run failed. (The integrity-gate tampering path
    already reports to stderr and stops the run before this is reached, so
    the only refusals surfaced here are stale entries.)
    """
    verb = "WOULD migrate" if args.dry_run else "migrated"
    sys.stdout.write(f"migrate-judge-scope (owner={args.owner}, dry_run={args.dry_run})\n")
    sys.stdout.write(f"  {verb}: {len(migrated)} entr{'y' if len(migrated) == 1 else 'ies'}\n")
    for key in migrated:
        sys.stdout.write(f"    + {key}\n")
    refused_summary = f"  refused / re-justify required: {len(refused)} entr{'y' if len(refused) == 1 else 'ies'}\n"
    sys.stdout.write(refused_summary)
    if refused:
        sys.stderr.write(refused_summary)
        for key, reason in refused:
            sys.stderr.write(f"    - {key}: {reason}\n")


@dataclass(frozen=True)
class _V2RewriteSpec:
    """One entry's v1→v2 binding rewrite, resolved before any file write."""

    entry_key: str
    scope_fingerprint: str
    ast_path: str
    v2_signature: str


def _rewrite_v1_entries_as_v2_in_yaml(target_yaml: Path, specs: list[_V2RewriteSpec]) -> None:
    """Rewrite a batch of v1 allow_hits entries' binding lines to v2, in one write.

    Byte-preserving line surgery (the established pattern in this file — see
    ``_upsert_audit_review_in_yaml``): we locate each entry by its ``- key:``
    line and operate ONLY on its line range, leaving every other byte of the
    file — and every non-binding line of the entry, including multi-line
    block-scalar ``reason`` / ``safety`` / ``judge_rationale`` bodies —
    BYTE-IDENTICAL. The three binding mutations per entry are:

    * the ``file_fingerprint:`` line is REPLACED in place by three lines —
      ``judge_signature_version: 2`` then ``scope_fingerprint: …`` then
      ``judge_transport: openrouter`` — matching the canonical field order
      ``_build_yaml_entry_text`` emits for a fresh v2 entry, so a migrated
      entry is indistinguishable from a justified one (``judge_transport`` is
      backfilled openrouter, truthful for the OpenRouter-produced corpus);
    * the ``ast_path:`` line is rewritten to the live finding's ast_path (a
      binding field; in the no-drift case it is byte-identical, so the line
      does not actually change); and
    * the ``judge_metadata_signature:`` line's value is replaced with the
      recomputed v2 signature.

    Binding lines are matched on the writer's EXACT 2-space indent
    (``line.startswith("  file_fingerprint:")``), NOT on ``str.strip()``.
    Strip-based matching is indentation-blind: a 4-space-indented block-scalar
    BODY line inside a ``reason`` / ``safety`` / ``judge_rationale`` field
    (operator/LLM free text) that happens to strip to ``ast_path:`` or
    ``file_fingerprint:`` would be misidentified as a binding line and
    rewritten — destroying audit text and/or injecting a duplicate YAML key
    (PyYAML accepts duplicate keys last-wins, so the corruption reloads
    silently when it lands in an unsigned field). ``_build_yaml_entry_text``
    emits binding fields at exactly 2 spaces and block-scalar bodies at 4
    spaces, so the indent-exact prefix is an unambiguous discriminator.

    All ``specs`` for ``target_yaml`` are applied to ONE in-memory copy and
    written ONCE under ``atomic_update_text``: the file is migrated wholly or
    not at all (per-file atomicity — a crash mid-run cannot leave a single
    YAML file half-rewritten; the worst case is a later file untouched, which
    is a valid idempotent state a re-run resumes from). New scalar values go
    through ``_yaml_inline_scalar`` so quoting matches the writer exactly.
    """
    specs_by_key = {spec.entry_key: spec for spec in specs}
    if len(specs_by_key) != len(specs):
        raise ValueError(f"{target_yaml}: duplicate entry keys in rewrite batch")

    def rewrite_in(current: str | None) -> str:
        if current is None:
            raise ValueError(f"{target_yaml}: allowlist YAML file is required")

        lines = current.splitlines(keepends=True)
        header_index = None
        for idx, line in enumerate(lines):
            if line.rstrip("\r\n") == "allow_hits:":
                header_index = idx
                break
        if header_index is None:
            raise ValueError(f"{target_yaml}: no allow_hits block found")

        block_end = len(lines)
        for idx in range(header_index + 1, len(lines)):
            line = lines[idx]
            if not line.strip():
                continue
            if _is_allow_hits_block_line(line):
                continue
            block_end = idx
            break

        # Resolve every spec's entry range up front, against the SAME source
        # snapshot, so we can splice them all in one pass without index drift.
        entry_ranges = _allow_hit_entry_ranges(lines, start=header_index + 1, end=block_end)
        range_by_key: dict[str, tuple[int, int]] = {}
        for entry_start, entry_end in entry_ranges:
            key_line = lines[entry_start].rstrip("\r\n")
            if not key_line.startswith("- key: "):
                continue
            entry_key = key_line.removeprefix("- key: ")
            if entry_key not in specs_by_key:
                continue
            if entry_key in range_by_key:
                raise ValueError(f"{target_yaml}: duplicate allow_hits entries found for key {entry_key!r}")
            range_by_key[entry_key] = (entry_start, entry_end)

        for entry_key in specs_by_key:
            if entry_key not in range_by_key:
                raise ValueError(f"{target_yaml}: no allow_hits entry found for key {entry_key!r}")

        # Splice from the bottom up so earlier ranges keep their indices while
        # later ranges are replaced.
        ordered = sorted(range_by_key.items(), key=lambda item: item[1][0])
        result_lines = list(lines)
        for entry_key, (entry_start, entry_end) in reversed(ordered):
            spec = specs_by_key[entry_key]
            rewritten = _rewrite_entry_binding_lines(
                target_yaml=target_yaml,
                entry_key=entry_key,
                entry_lines=lines[entry_start:entry_end],
                spec=spec,
            )
            result_lines[entry_start:entry_end] = rewritten

        return "".join(result_lines)

    atomic_update_text(target_yaml, rewrite_in, encoding="utf-8", create_parent=False)


def _rewrite_entry_binding_lines(
    *,
    target_yaml: Path,
    entry_key: str,
    entry_lines: list[str],
    spec: _V2RewriteSpec,
) -> list[str]:
    """Return ``entry_lines`` with its v1 binding lines rewritten to v2.

    Matching is indent-exact on the writer's 2-space prefix (see
    ``_rewrite_v1_entries_as_v2_in_yaml``'s docstring for why strip-based
    matching is a corruption hazard). Every non-binding line — including
    4-space-indented block-scalar bodies — is passed through unchanged.

    Rewritten binding lines emit ``\\n``: the allowlist toolchain is LF-only
    end to end. ``atomic_update_text`` reads via ``Path.read_text`` (universal
    newlines), so this callback only ever sees LF-normalised text — a CRLF
    input file is normalised to LF on read before we get here — and the
    canonical writer ``_build_yaml_entry_text`` emits LF unconditionally. So
    there is no line ending to "sample and preserve"; the file is LF by
    construction. (Do NOT re-add CRLF sampling: it would be dead code — the
    callback can never observe a ``\\r``.)
    """
    rewritten: list[str] = []
    saw_file_fingerprint = False
    saw_signature = False
    for line in entry_lines:
        if line.startswith("  file_fingerprint:"):
            saw_file_fingerprint = True
            rewritten.append("  judge_signature_version: 2\n")
            rewritten.append(f"  scope_fingerprint: {_yaml_inline_scalar(spec.scope_fingerprint)}\n")
            # judge_transport is backfilled "openrouter" for every migrated
            # entry (the corpus is OpenRouter-produced) and is positioned right
            # after scope_fingerprint — the SAME field order ``_build_yaml_entry_text``
            # emits — so a migrated entry is byte-congruent with a justified one.
            rewritten.append(f"  judge_transport: {_yaml_inline_scalar('openrouter')}\n")
            continue
        if line.startswith("  ast_path:"):
            rewritten.append(f"  ast_path: {_yaml_inline_scalar(spec.ast_path)}\n")
            continue
        if line.startswith("  judge_metadata_signature:"):
            saw_signature = True
            rewritten.append(f"  judge_metadata_signature: {_yaml_inline_scalar(spec.v2_signature)}\n")
            continue
        rewritten.append(line)

    if not saw_file_fingerprint:
        raise ValueError(
            f"{target_yaml}: entry {entry_key!r} has no file_fingerprint line; "
            "migrate-judge-scope only rewrites v1 (file_fingerprint-bound) entries."
        )
    if not saw_signature:
        raise ValueError(f"{target_yaml}: entry {entry_key!r} has no judge_metadata_signature line to re-sign.")
    return rewritten


def _write_report(report: Any, args: argparse.Namespace) -> None:
    """Render ``report`` per ``args.reaudit_format`` to stdout or ``args.output``."""
    from elspeth_lints.core.reaudit import (
        render_report_json,
        render_report_markdown,
        render_report_text,
    )

    if args.reaudit_format == "json":
        rendered = render_report_json(report)
    elif args.reaudit_format == "markdown":
        rendered = render_report_markdown(report)
    else:
        rendered = render_report_text(report)

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


def _parse_since(value: str) -> Any:
    """Parse the ``--since`` argument as ISO-8601 date or timestamp.

    Accepts:
    * ``YYYY-MM-DD`` (interpreted as midnight UTC on that date)
    * Full ISO-8601 timestamp with timezone (``YYYY-MM-DDTHH:MM:SS+00:00``)

    Returns ``None`` on malformed input so the caller can surface a
    user-actionable error. Naive timestamps (no tzinfo) are rejected
    because judge ``recorded_at`` values are always timezone-aware;
    comparing tz-naive against tz-aware would raise at runtime.
    """
    from datetime import UTC, date, datetime, time

    try:
        parsed_date = date.fromisoformat(value)
    except ValueError:
        parsed_date = None
    if parsed_date is not None and len(value) == 10:
        # Bare date — anchor at midnight UTC so the comparison is well-defined.
        return datetime.combine(parsed_date, time.min, tzinfo=UTC)
    try:
        parsed_dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed_dt.tzinfo is None:
        return None
    return parsed_dt


def _run_check_judge_coverage(args: argparse.Namespace) -> int:
    """Handle ``elspeth-lints check-judge-coverage`` — convergent finding C1.

    Lazy-imports ``judge_coverage`` so other subcommands don't pay the
    yaml + subprocess import cost. The exit-code contract:

    * 0 — every new entry in this PR carries the judge metadata
      quartet (or this PR introduced no new entries at all).
    * 1 — at least one new entry is missing one or more required
      judge fields; the report is printed to stdout for the CI log.
    * 2 — the check itself could not run (missing baseline ref,
      ``allowlist-root`` not a directory, etc.). Distinct exit code
      so CI distinguishes "gate fired" from "gate broken".
    """
    from elspeth_lints.core.judge_coverage import (
        JudgeCoverageError,
        check_judge_coverage,
    )

    try:
        reports = check_judge_coverage(
            allowlist_root=args.allowlist_root,
            baseline_ref=args.baseline_ref,
            repo_root=args.repo_root,
        )
    except JudgeCoverageError as exc:
        sys.stderr.write(f"check-judge-coverage: cannot run: {exc}\n")
        return 2

    total_head = sum(r.head_entry_count for r in reports.values())
    total_new = sum(r.new_entry_count for r in reports.values())
    total_grandfathered = sum(r.grandfathered_count for r in reports.values())
    total_violations = sum(len(r.violations) for r in reports.values())

    sys.stdout.write(
        f"check-judge-coverage: {len(reports)} directory(ies) inspected, "
        f"{total_head} entries at HEAD ({total_grandfathered} grandfathered, "
        f"{total_new} new); {total_violations} violation(s).\n"
    )

    if total_violations == 0:
        return 0

    sys.stdout.write("\nNew entries missing required judge metadata:\n")
    for dir_name in sorted(reports):
        report = reports[dir_name]
        if not report.violations:
            continue
        sys.stdout.write(f"\n  {dir_name}/:\n")
        for violation in report.violations:
            sys.stdout.write(f"    {violation.source_file} :: {violation.entry_key}\n")
            sys.stdout.write(f"      missing: {', '.join(violation.missing_fields)}\n")

    sys.stdout.write(
        "\nResolve by running 'elspeth-lints justify' on each new "
        "allow_hits entry to obtain an LLM-judged verdict. For new "
        "per_file_rules violations, replace the wildcard with exact "
        "allow_hits entries that can carry judge metadata, or split the "
        "change into a reviewed migration. Use '--operator-override' only "
        "to record an OVERRIDDEN_BY_OPERATOR verdict when the judge would "
        "BLOCK an exact entry incorrectly.\n"
    )
    return 1


def _run_check_judge_quality(args: argparse.Namespace) -> int:
    """Handle ``elspeth-lints check-judge-quality``.

    Exit-code contract:

    * 0 — live judge exact-match accuracy is at or above threshold.
    * 1 — the corpus ran, but verdict/decorator accuracy is too low.
    * 2 — the measurement itself could not run (bad corpus, missing
      judge dependencies/key, transport failure, malformed judge
      response, or invalid evaluator configuration).
    """
    from elspeth_lints.core.judge import (
        DEFAULT_JUDGE_MAX_TOKENS,
        DEFAULT_JUDGE_MODEL,
        JudgeConfigurationError,
        JudgeContractError,
        JudgeTransportError,
    )
    from elspeth_lints.core.judge_quality import (
        JudgeQualityError,
        evaluate_judge_quality_corpus,
        load_judge_quality_corpus,
        render_judge_quality_report_json,
        render_judge_quality_report_text,
    )

    try:
        cases = load_judge_quality_corpus(args.corpus)
        report = evaluate_judge_quality_corpus(
            cases=cases,
            corpus_path=args.corpus,
            min_accuracy=args.min_accuracy,
            min_cases=args.min_cases,
            max_cases=args.max_cases,
            model_id=args.model or DEFAULT_JUDGE_MODEL,
            max_tokens=args.max_tokens or DEFAULT_JUDGE_MAX_TOKENS,
        )
    except JudgeQualityError as exc:
        sys.stderr.write(f"check-judge-quality: cannot run: {exc}\n")
        return 2
    except JudgeConfigurationError as exc:
        sys.stderr.write(f"check-judge-quality: judge configuration error: {exc}\n")
        return 2
    except JudgeTransportError as exc:
        sys.stderr.write(f"check-judge-quality: judge transport error: {exc}\n")
        return 2
    except JudgeContractError as exc:
        sys.stderr.write(f"check-judge-quality: judge contract error: {exc}\n")
        return 2

    if args.judge_quality_format == "json":
        sys.stdout.write(render_judge_quality_report_json(report))
    else:
        sys.stdout.write(render_judge_quality_report_text(report))
    return 0 if report.passes else 1


def _run_check_trust_boundary_diff(args: argparse.Namespace) -> int:
    """Handle ``elspeth-lints check-trust-boundary-diff``.

    Exit-code contract:

    * 0 — report produced, regardless of whether new decorators exist.
    * 2 — the diff itself could not run (bad baseline, bad root, git failure).
    """
    from elspeth_lints.core.trust_boundary_diff import (
        TrustBoundaryDiffError,
        find_new_trust_boundary_decorators,
        render_trust_boundary_diff_summary,
    )

    try:
        report = find_new_trust_boundary_decorators(
            root=args.root,
            baseline_ref=args.baseline_ref,
            repo_root=args.repo_root,
        )
    except TrustBoundaryDiffError as exc:
        sys.stderr.write(f"check-trust-boundary-diff: cannot run: {exc}\n")
        return 2

    sys.stdout.write(render_trust_boundary_diff_summary(report))
    return 0


def _run_check_rotation_audit(args: argparse.Namespace) -> int:
    """Handle ``elspeth-lints check-rotation-audit``."""
    from elspeth_lints.rules.trust_tier.tier_model.rotate import (
        RotationAuditError,
        check_rotation_audit_coverage,
    )

    try:
        report = check_rotation_audit_coverage(
            allowlist_root=args.allowlist_root,
            baseline_ref=args.baseline_ref,
            repo_root=args.repo_root,
            rotation_log_path=args.rotation_log,
        )
    except RotationAuditError as exc:
        sys.stderr.write(f"check-rotation-audit: cannot run: {exc}\n")
        return 2

    sys.stdout.write(
        f"check-rotation-audit: {report.checked_rotation_count} rotation(s) "
        f"detected against {report.baseline_ref}; {len(report.violations)} "
        f"unrecorded. manifest={report.rotation_log_path}\n"
    )
    if report.passes:
        return 0

    sys.stdout.write("\nUnrecorded tier_model allowlist rotations:\n")
    for violation in report.violations:
        sys.stdout.write(f"  {violation.allowlist_file}\n")
        sys.stdout.write(f"    old: {violation.old_key}\n")
        sys.stdout.write(f"    new: {violation.new_key}\n")
    sys.stdout.write(
        "\nResolve by applying rotations through 'elspeth-lints rotate' so "
        ".elspeth/rotations.log records the old/new key mapping, or add a "
        "reviewed manifest record that matches the mechanical rotation.\n"
    )
    return 1


def _run_check_override_rate(args: argparse.Namespace) -> int:
    """Handle ``elspeth-lints check-override-rate`` — convergent finding C3.

    Exit-code contract:

    * 0 — rate within budget OR insufficient data inside the window.
    * 1 — rate exceeds ``--max-rate``; the contributing overrides are
      listed for triage.
    * 2 — the check itself could not run (root not a directory,
      bad ``--reference-time`` shape, etc.).
    """
    from datetime import datetime as _datetime

    from elspeth_lints.core.override_rate import (
        OverrideRateError,
        compute_override_rate,
        default_counter_snapshot_path,
        write_override_rate_counter_snapshot,
    )

    reference_time = None
    if args.reference_time is not None:
        try:
            reference_time = _datetime.fromisoformat(args.reference_time)
        except ValueError as exc:
            sys.stderr.write(f"check-override-rate: --reference-time must be ISO-8601 (got {args.reference_time!r}): {exc}\n")
            return 2

    counter_snapshot_path = args.counter_snapshot or default_counter_snapshot_path(args.allowlist_root)

    try:
        detail = compute_override_rate(
            allowlist_root=args.allowlist_root,
            window_days=args.window_days,
            min_samples=args.min_samples,
            max_rate=args.max_rate,
            max_overrides=args.max_overrides,
            reference_time=reference_time,
            counter_snapshot_path=counter_snapshot_path,
        )
    except OverrideRateError as exc:
        sys.stderr.write(f"check-override-rate: cannot run: {exc}\n")
        return 2

    if detail.counter_source == "yaml":
        try:
            write_override_rate_counter_snapshot(args.allowlist_root, snapshot_path=counter_snapshot_path)
        except OverrideRateError as exc:
            sys.stderr.write(f"check-override-rate: counter snapshot refresh failed: {exc}\n")
        except OSError as exc:
            sys.stderr.write(f"check-override-rate: counter snapshot refresh failed: {exc}\n")
        else:
            sys.stderr.write(f"check-override-rate: refreshed counter snapshot {counter_snapshot_path}\n")

    report = detail.report
    pct = report.rate * 100.0
    max_pct = report.max_rate * 100.0

    sys.stdout.write(
        f"check-override-rate: window={report.window_days}d, "
        f"reference={report.reference_time.isoformat()}, "
        f"judged_in_window={report.judged_in_window}, "
        f"accepted_in_window={report.accepted_in_window}, "
        f"overrides_in_window={report.overrides_in_window}, "
        f"model_accepted_in_window={report.model_accepted_in_window}, "
        f"model_blocked_in_window={report.model_blocked_in_window}, "
        f"blocked_without_override_in_window={report.blocked_without_override_in_window}, "
        f"rate={pct:.2f}% (max {max_pct:.2f}%), "
        f"counter_source={detail.counter_source}\n"
    )
    if report.max_overrides is not None:
        sys.stdout.write(f"check-override-rate: max_overrides={report.max_overrides}\n")
    if detail.per_rule_reports:
        sys.stdout.write("per-rule override rates:\n")
        for per_rule in detail.per_rule_reports:
            sys.stdout.write(
                f"  {per_rule.rule_id}: judged={per_rule.judged_in_window}, "
                f"accepted={per_rule.accepted_in_window}, "
                f"overrides={per_rule.overrides_in_window}, "
                f"model_accepted={per_rule.model_accepted_in_window}, "
                f"model_blocked={per_rule.model_blocked_in_window}, "
                f"rate={per_rule.rate * 100.0:.2f}%\n"
            )

    if report.absolute_budget_exceeded:
        assert report.max_overrides is not None
        sys.stdout.write(f"FAIL: override count {report.overrides_in_window} exceeds absolute budget {report.max_overrides}.\n")
    elif report.insufficient_data:
        sys.stdout.write(
            f"PASS: insufficient data (denominator={report.judged_in_window} < "
            f"min_samples={report.min_samples}). Override-rate gate cannot "
            "produce a stable signal at this sample size; treat the result as "
            "informational until the denominator grows.\n"
        )
        return 0
    elif report.passes:
        sys.stdout.write("PASS: override rate within budget.\n")
        return 0
    if report.ratio_budget_exceeded:
        sys.stdout.write(f"FAIL: override rate {pct:.2f}% exceeds budget {max_pct:.2f}%.\n")
    sys.stdout.write(f"\n{len(detail.override_entries)} OVERRIDDEN_BY_OPERATOR entries in window:\n")
    for record in sorted(
        detail.override_entries,
        key=lambda r: (r.judge_recorded_at, r.entry_key),
    ):
        sys.stdout.write(f"  {record.judge_recorded_at.isoformat()}  {record.source_file} :: {record.entry_key}\n")
    sys.stdout.write(
        "\nResolution paths:\n"
        "  1. Run 'elspeth-lints reaudit' against the override-heavy "
        "directories; entries classified WAS_ACCEPTED_NOW_BLOCKED can be "
        "actioned (refactor away the suppression).\n"
        "  2. If overrides are legitimate, the operator must adjust "
        "--max-rate via an ADR — the threshold is policy, not a quota.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
