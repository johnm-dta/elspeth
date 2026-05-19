"""Command-line interface for elspeth-lints."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Sequence
from pathlib import Path

from elspeth_lints.core.ast_walker import ParsedPythonFile, PythonSyntaxError, walk_python_files
from elspeth_lints.core.emitters.github import render_github
from elspeth_lints.core.emitters.json import render_json
from elspeth_lints.core.emitters.sarif import render_sarif
from elspeth_lints.core.emitters.text import render_text
from elspeth_lints.core.protocols import Finding, Rule, RuleContext, RuleScope
from elspeth_lints.core.registry import DEFAULT_REGISTRY, RuleRegistry


def main(argv: Sequence[str] | None = None) -> int:
    """Run the elspeth-lints CLI."""
    DEFAULT_REGISTRY.load_builtin_rules()
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        return _run_check(args, registry=DEFAULT_REGISTRY)
    if args.command == "dump-edges":
        return _run_dump_edges(args)
    parser.error(f"unknown command {args.command!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="elspeth-lints")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Run static-analysis rules")
    check.add_argument("--rules", default="nothing", help="Comma-separated rule ids, or 'nothing' for the empty skeleton run")
    check.add_argument("--rule-set", choices=("static", "full"), default="static")
    check.add_argument("--format", choices=("text", "json", "sarif", "github"), default="text")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--allowlist-dir", type=Path)
    check.add_argument("--files", nargs="*", type=Path)

    dump_edges = subparsers.add_parser("dump-edges", help="Dump import edges for architecture review")
    dump_edges.add_argument("--root", type=Path, default=Path.cwd())
    dump_edges.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _run_check(args: argparse.Namespace, *, registry: RuleRegistry) -> int:
    requested_rules = _parse_rules(args.rules)
    if not requested_rules:
        return _emit_findings([], output_format=args.format, rules=[])

    available = set(registry.ids())
    unknown = sorted(set(requested_rules).difference(available))
    if unknown:
        sys.stderr.write(f"Unknown rule id(s): {', '.join(unknown)}\n")
        return 2

    findings: list[Finding] = []
    context = RuleContext(root=args.root)
    selected_rules = [registry.get(rule_id) for rule_id in requested_rules]
    whole_repo_rules = [rule for rule in selected_rules if rule.scope == RuleScope.WHOLE_REPO]
    incremental_rules = [rule for rule in selected_rules if rule.scope == RuleScope.INCREMENTAL]

    empty_tree = ast.Module(body=[], type_ignores=[])
    for rule in whole_repo_rules:
        findings.extend(rule.analyze(empty_tree, args.root, context))

    if incremental_rules:
        for item in walk_python_files(args.root, args.files):
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
            findings.extend(_run_rules(item, incremental_rules, context=context))

    return _emit_findings(findings, output_format=args.format, rules=selected_rules)


def _parse_rules(raw: str) -> tuple[str, ...]:
    rules = tuple(part.strip() for part in raw.split(",") if part.strip())
    if rules == ("nothing",):
        return ()
    return rules


def _run_rules(item: ParsedPythonFile, rules: list[Rule], *, context: RuleContext) -> list[Finding]:
    findings: list[Finding] = []
    for rule in rules:
        findings.extend(rule.analyze(item.tree, item.path, context))
    return findings


def _run_dump_edges(args: argparse.Namespace) -> int:
    if args.format == "json":
        sys.stdout.write("[]\n")
        return 0
    if args.format == "text":
        return 0
    raise ValueError(f"unknown dump-edges format: {args.format}")


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


if __name__ == "__main__":
    raise SystemExit(main())
