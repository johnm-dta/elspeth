"""Command-line interface for elspeth-lints."""

from __future__ import annotations

import argparse
import ast
import re
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
    check.add_argument("--allowlist-dir", type=Path)
    check.add_argument("--files", nargs="*", type=Path)

    dump_edges = subparsers.add_parser("dump-edges", help="Dump import edges for architecture review")
    dump_edges.add_argument("--root", type=Path, default=Path.cwd())
    dump_edges.add_argument("--format", choices=("json", "mermaid", "dot"), default="json")
    dump_edges.add_argument("--output", type=Path)
    dump_edges.add_argument("--include-layer", action="append", choices=("L0", "L1", "L2", "L3"))
    dump_edges.add_argument("--collapse-to-subsystem", dest="collapse_to_subsystem", action="store_true", default=True)
    dump_edges.add_argument("--no-collapse", dest="collapse_to_subsystem", action="store_false")
    dump_edges.add_argument("--no-timestamp", action="store_true", default=False)
    dump_edges.add_argument("--exclude", action="append", default=[])
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
    context = RuleContext(root=args.root)
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
