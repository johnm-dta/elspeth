"""Command-line interface for elspeth-lints."""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth_lints.core.ast_walker import (
    ParsedPythonFile,
    PythonFileReadError,
    PythonSyntaxError,
    walk_python_files,
)
from elspeth_lints.core.emitters.github import render_github
from elspeth_lints.core.emitters.json import render_json
from elspeth_lints.core.emitters.sarif import render_sarif
from elspeth_lints.core.emitters.text import render_text
from elspeth_lints.core.protocols import Finding, Rule, RuleContext, RuleScope
from elspeth_lints.core.registry import DEFAULT_REGISTRY, RuleRegistry

if TYPE_CHECKING:
    from elspeth_lints.rules.trust_tier.tier_model.rotate import RotationPlan


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
    return value


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
    if args.command == "reaudit":
        return _run_reaudit(args)
    if args.command == "check-judge-coverage":
        return _run_check_judge_coverage(args)
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
        "--no-remove-stale",
        dest="remove_stale",
        action="store_false",
        default=True,
        help=(
            "Skip removal of stale allowlist entries (entries whose "
            "underlying violation site no longer exists). Default is to "
            "remove them, mirroring the prior tool's behaviour."
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
            "Qualified symbol name (e.g. 'MyClass._method'). Use the literal "
            "'_module_' for module-scope findings (matches the canonical-key "
            "sentinel)."
        ),
    )
    justify.add_argument(
        "--rationale",
        type=str,
        required=True,
        help="Agent's proposed justification for the suppression",
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
            "verdict=OVERRIDDEN_BY_OPERATOR."
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
    # policy, and the scope boundary (only ``allow_hits:`` shape; other
    # legacy YAML shapes are out of scope).
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
            "config/cicd. Subdirectories that use legacy YAML shapes (no "
            "'allow_hits:' block) are silently skipped."
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
        "--reference-time",
        type=str,
        default=None,
        help=(
            "Window anchor as a timezone-aware ISO-8601 timestamp (e.g. "
            "'2026-05-23T00:00:00Z'). Defaults to current UTC; override for "
            "reproducibility in CI replays."
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
    context = RuleContext(root=args.root, allowlist_dir_override=allowlist_dir)
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
            applied = apply_plan(plan, allowlist_dir=args.allowlist_dir, remove_stale=args.remove_stale)
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
        applied = apply_plan(plan, allowlist_dir=args.allowlist_dir, remove_stale=args.remove_stale)
        sys.stdout.write("\nApplied:\n")
        for src, r in sorted(applied.items()):
            parts: list[str] = []
            if r.rotations_applied:
                parts.append(f"{r.rotations_applied} rotation(s)")
            if r.stale_entries_removed:
                parts.append(f"{r.stale_entries_removed} stale removal(s)")
            sys.stdout.write(f"  {src}: {', '.join(parts)}\n")
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
    from elspeth_lints.core.allowlist import JudgeVerdict
    from elspeth_lints.core.judge import (
        DEFAULT_JUDGE_MODEL,
        JudgeConfigurationError,
        JudgeRequest,
        JudgeResponse,
        call_judge,
    )
    from elspeth_lints.rules.trust_tier.tier_model.rule import (
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

    target_file = _resolve_target_file(root=root, file_path_arg=args.file_path)
    if target_file is None:
        sys.stderr.write(f"--file-path: {args.file_path!r} does not exist under {root}\n")
        return 2

    symbol_tuple = _parse_symbol(args.symbol)
    findings = _scan_single_file_findings(
        target_file=target_file, root=root, scan_file=scan_file, scan_layer_imports_file=scan_layer_imports_file
    )
    matching = [f for f in findings if _finding_symbol_matches(f, symbol_tuple)]

    if not matching:
        sys.stderr.write(
            f"No findings on {args.file_path} match symbol {args.symbol!r}. "
            f"Scanned {len(findings)} finding(s) on that file. "
            "Either the symbol path is wrong, the file is clean, or the "
            "finding is already covered by a per_file_rule.\n"
        )
        return 2
    if len(matching) > 1:
        sys.stderr.write(
            f"Ambiguous: {len(matching)} findings on {args.file_path} match "
            f"symbol {args.symbol!r}. The judge gate requires a unique "
            "finding per entry. Either narrow the symbol path or run "
            "`elspeth-lints rotate` first if these are stale fingerprints. "
            "Matching findings:\n"
        )
        for finding in matching:
            sys.stderr.write(f"  {finding.canonical_key}  ({finding.rule_id} at line {finding.line})\n")
        return 2

    finding = matching[0]

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
    if args.rule != "trust_tier.tier_model" and args.rule != finding.rule_id:
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

    surrounding_code = _extract_surrounding_code(target_file, finding.line, context_lines=15)
    request = JudgeRequest(
        file_path=finding.file_path,
        rule_id=finding.rule_id,
        symbol=args.symbol,
        fingerprint=finding.fingerprint,
        rationale=args.rationale,
        surrounding_code=surrounding_code,
    )

    try:
        response: JudgeResponse = call_judge(request, model_id=DEFAULT_JUDGE_MODEL)
    except JudgeConfigurationError as exc:
        sys.stderr.write(f"Judge configuration error: {exc}\n")
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

    # C8-3 binding: compute the source-file fingerprint and capture the
    # finding's ast_path at write time, alongside the judge metadata.
    # These two fields make the persisted quartet cryptographically
    # bound to the bytes + AST node the judge actually inspected — the
    # loader/matcher pair (allowlist.load_allowlist and
    # allowlist.verify_entry_binding_against_finding) reads them back
    # and asserts the binding still holds, closing the quartet-
    # transplant attack vector.
    file_fingerprint = hashlib.sha256(target_file.read_bytes()).hexdigest()
    yaml_entry = _build_yaml_entry_text(
        key=finding.canonical_key,
        owner=args.owner,
        reason=args.rationale,
        verdict=write_verdict,
        recorded_at=response.recorded_at,
        model_id=response.model_id,
        judge_rationale=response.judge_rationale,
        model_verdict=model_verdict,
        file_fingerprint=file_fingerprint,
        ast_path=finding.ast_path,
    )
    target_yaml = _suggest_yaml_target(finding=finding, allowlist_dir=allowlist_dir)

    # BLOCKED without override is the terminal-failure branch: print the
    # judge's rationale + the model that produced it, do not write, exit
    # non-zero. This is the "judge does not fix" load-bearing constraint.
    if response.verdict == JudgeVerdict.BLOCKED and not args.operator_override:
        _emit_justify_output(
            args=args,
            verdict=write_verdict,
            judge_response=response,
            target_yaml=target_yaml,
            yaml_entry=yaml_entry,
            wrote=False,
            blocked=True,
        )
        return 1

    if args.dry_run:
        _emit_justify_output(
            args=args,
            verdict=write_verdict,
            judge_response=response,
            target_yaml=target_yaml,
            yaml_entry=yaml_entry,
            wrote=False,
            blocked=False,
        )
        return 0

    _append_entry_to_yaml(target_yaml, yaml_entry)
    _emit_justify_output(
        args=args,
        verdict=write_verdict,
        judge_response=response,
        target_yaml=target_yaml,
        yaml_entry=yaml_entry,
        wrote=True,
        blocked=False,
    )
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


def _resolve_target_file(*, root: Path, file_path_arg: str) -> Path | None:
    """Resolve ``--file-path`` against ``--root``, returning an absolute Path or None.

    Accepts both forms the user might supply: a path relative to
    ``--root`` (the same shape that appears in finding.file_path) or
    an absolute path. Returns ``None`` if the resolved file does not
    exist.
    """
    candidate = Path(file_path_arg)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = (root / candidate).resolve()
    return resolved if resolved.exists() else None


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


def _finding_symbol_matches(finding: Any, symbol_tuple: tuple[str, ...]) -> bool:
    """Return True iff the finding's ``symbol_context`` equals the tuple."""
    return tuple(finding.symbol_context) == symbol_tuple


def _extract_surrounding_code(target_file: Path, line: int, *, context_lines: int) -> str:
    """Return ~30 lines of code centered on ``line`` from ``target_file``.

    The judge reads this to verify the rationale honestly describes the
    site. We include line-number prefixes so the model can correlate
    its observations with the finding's reported line.
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
    file_fingerprint: str,
    ast_path: str,
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
    lines.append("  judge_rationale: |-")
    for rationale_line in judge_rationale.splitlines() or [""]:
        lines.append(f"    {rationale_line}")
    # C8-3 binding fields. ``file_fingerprint`` is the SHA-256 of the
    # source-file bytes the judge inspected; ``ast_path`` is the AST
    # field/index path from the module root to the finding's subject
    # node. Together they bind the persisted quartet to source+AST so
    # the loader/matcher pair can detect quartet transplant and
    # source drift. Both are scalars; emit inline.
    lines.append(f"  file_fingerprint: {_yaml_inline_scalar(file_fingerprint)}")
    lines.append(f"  ast_path: {_yaml_inline_scalar(ast_path)}")
    return "\n".join(lines) + "\n"


def _yaml_inline_scalar(value: str) -> str:
    """Quote a string for inline YAML scalar emission when it needs it.

    Conservative: any character outside ``[A-Za-z0-9._/-]`` triggers
    single-quoting. Single quotes inside the value are escaped by
    doubling them, per YAML 1.2.
    """
    if value and all(ch.isalnum() or ch in "._/-" for ch in value):
        return value
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _append_entry_to_yaml(target_yaml: Path, entry_text: str) -> None:
    """Insert one ``allow_hits`` entry into the per-module YAML file.

    Append-only at the END of the existing ``allow_hits:`` block. If
    the file does not exist, create it with an ``allow_hits:`` header.
    If the file exists but has no ``allow_hits:`` block, append one at
    the bottom. We deliberately do NOT round-trip through PyYAML —
    the per-module YAMLs carry multi-paragraph comments that
    ``yaml.dump`` would erase, and the rotate command's surgical
    text-edit approach is the established pattern in this codebase.
    """
    if not target_yaml.exists():
        target_yaml.parent.mkdir(parents=True, exist_ok=True)
        target_yaml.write_text(f"allow_hits:\n{entry_text}", encoding="utf-8")
        return

    text = target_yaml.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Locate the ``allow_hits:`` header line; if absent, append the
    # whole block at the bottom.
    header_index = None
    for idx, line in enumerate(lines):
        if line.rstrip("\r\n") == "allow_hits:":
            header_index = idx
            break

    if header_index is None:
        prefix = "" if not text or text.endswith("\n") else "\n"
        target_yaml.write_text(text + f"{prefix}\nallow_hits:\n{entry_text}", encoding="utf-8")
        return

    # Find the end of the allow_hits block: the next line at col 0 that
    # is non-empty and not part of the block (i.e. starts a new top-
    # level key). Lines that start with ``- `` or are indented belong
    # to the block; blank lines inside the block stay.
    block_end = len(lines)
    for idx in range(header_index + 1, len(lines)):
        line = lines[idx]
        if not line.strip():
            continue
        if line.startswith("- ") or line.startswith(" "):
            continue
        # Hit a new top-level key (e.g. ``defaults:``); the block ends here.
        block_end = idx
        break

    # Ensure the existing block ends with a newline before our insert.
    if block_end > header_index + 1 and not lines[block_end - 1].endswith("\n"):
        lines[block_end - 1] = lines[block_end - 1] + "\n"

    new_lines = [*lines[:block_end], entry_text, *lines[block_end:]]
    target_yaml.write_text("".join(new_lines), encoding="utf-8")


def _emit_justify_output(
    *,
    args: argparse.Namespace,
    verdict: Any,  # JudgeVerdict
    judge_response: Any,  # JudgeResponse
    target_yaml: Path,
    yaml_entry: str,
    wrote: bool,
    blocked: bool,
) -> None:
    """Render the justify result as text or JSON to stdout."""
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
        }
        sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    sys.stdout.write(f"Verdict:          {verdict.value}\n")
    sys.stdout.write(f"Judge model:      {judge_response.model_id}\n")
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
    return 1 if findings else 0


def _run_reaudit(args: argparse.Namespace) -> int:
    """Drive a reaudit (decay-sweep) pass over an existing allowlist.

    Lazy-imports the reaudit module so subcommands that don't reach for
    the judge / tier_model scanner don't pay their import cost.
    """
    from elspeth_lints.core.judge import JudgeConfigurationError
    from elspeth_lints.core.reaudit import (
        ReauditDivergence,
        ReauditError,
        reaudit_entries,
        render_report_json,
        render_report_markdown,
        render_report_text,
    )

    since: Any = None
    if args.since is not None:
        since = _parse_since(args.since)
        if since is None:
            sys.stderr.write(f"--since: {args.since!r} is not a valid ISO-8601 date or timestamp\n")
            return 2

    try:
        report = reaudit_entries(
            root=args.root.resolve(),
            allowlist_dir=args.allowlist_dir,
            rule_filter=args.rule,
            since=since,
            limit=args.limit,
            include_pre_judge=args.include_pre_judge,
        )
    except ReauditError as exc:
        sys.stderr.write(f"reaudit error: {exc}\n")
        return 2
    except JudgeConfigurationError as exc:
        sys.stderr.write(f"Judge configuration error: {exc}\n")
        return 2

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

    # Exit-code policy (closes elspeth-9a4e54cc01 / C3-2 + C3-3):
    #   0 — sweep complete, every entry produced a verdict-based
    #       divergence (including ENTRY_OBSOLETE).
    #   1 — sweep had operator-actionable data-collection gaps:
    #       either entries the sweep never reached
    #       (entries_dispatched < total_entries) OR entries whose
    #       judge call raised a transport error (JUDGE_CALL_FAILED).
    #       Distinct from the verdict-change cases (which are *signal*,
    #       not failure) — those still exit 0.
    judge_call_failures = sum(1 for outcome in report.outcomes if outcome.divergence is ReauditDivergence.JUDGE_CALL_FAILED)
    incomplete = report.entries_dispatched < report.total_entries
    if judge_call_failures > 0 or incomplete:
        return 1
    return 0


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
        "entry to obtain an LLM-judged verdict, or '--operator-override' "
        "to record an OVERRIDDEN_BY_OPERATOR verdict if the judge would "
        "BLOCK incorrectly.\n"
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
    )

    reference_time = None
    if args.reference_time is not None:
        try:
            reference_time = _datetime.fromisoformat(args.reference_time)
        except ValueError as exc:
            sys.stderr.write(f"check-override-rate: --reference-time must be ISO-8601 (got {args.reference_time!r}): {exc}\n")
            return 2

    try:
        detail = compute_override_rate(
            allowlist_root=args.allowlist_root,
            window_days=args.window_days,
            min_samples=args.min_samples,
            max_rate=args.max_rate,
            reference_time=reference_time,
        )
    except OverrideRateError as exc:
        sys.stderr.write(f"check-override-rate: cannot run: {exc}\n")
        return 2

    report = detail.report
    pct = report.rate * 100.0
    max_pct = report.max_rate * 100.0

    sys.stdout.write(
        f"check-override-rate: window={report.window_days}d, "
        f"reference={report.reference_time.isoformat()}, "
        f"judged_in_window={report.judged_in_window}, "
        f"overrides_in_window={report.overrides_in_window}, "
        f"rate={pct:.2f}% (max {max_pct:.2f}%)\n"
    )

    if report.insufficient_data:
        sys.stdout.write(
            f"PASS: insufficient data (denominator={report.judged_in_window} < "
            f"min_samples={report.min_samples}). Override-rate gate cannot "
            "produce a stable signal at this sample size; treat the result as "
            "informational until the denominator grows.\n"
        )
        return 0

    if report.passes:
        sys.stdout.write("PASS: override rate within budget.\n")
        return 0

    sys.stdout.write(
        f"FAIL: override rate {pct:.2f}% exceeds budget {max_pct:.2f}%.\n"
        f"\n{len(detail.override_entries)} OVERRIDDEN_BY_OPERATOR entries "
        "in window:\n"
    )
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
