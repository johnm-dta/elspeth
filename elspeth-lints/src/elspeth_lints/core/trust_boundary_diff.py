"""PR-diff surfacing for newly-added ``@trust_boundary`` decorators.

This module is intentionally report-only: adding a decorator is not, by
itself, a CI failure. The operator-facing value is that PR review gets a
compact list of newly introduced in-source suppression claims, alongside the
existing allowlist judge gates.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from elspeth_lints.rules.trust_boundary.shared import extract_keywords
from elspeth_lints.rules.trust_tier.tier_model.trust_boundary_suppress import find_trust_boundary_calls


@dataclass(frozen=True, slots=True)
class TrustBoundaryDecoratorRecord:
    """One ``@trust_boundary`` decorator found in a source file."""

    source_file: str
    line: int
    symbol: str
    source_param: str | None
    suppresses: tuple[str, ...]
    source: str | None
    test_ref: str | None
    metadata_readable: bool
    identity_hash: str


@dataclass(frozen=True, slots=True)
class TrustBoundaryDiffReport:
    """Diff report for trust-boundary decorators in one source root."""

    baseline_ref: str
    root: str
    new_decorators: tuple[TrustBoundaryDecoratorRecord, ...]


class TrustBoundaryDiffError(RuntimeError):
    """The trust-boundary decorator diff could not be computed."""


def find_new_trust_boundary_decorators(
    *,
    root: Path,
    baseline_ref: str,
    repo_root: Path,
) -> TrustBoundaryDiffReport:
    """Return ``@trust_boundary`` decorators present in HEAD but not baseline.

    ``root`` may be absolute or repository-relative. ``baseline_ref`` is any
    git rev-spec that resolves to the PR comparison base. The comparison uses
    changed Python files only; deleted files are ignored because they cannot
    introduce new decorators.
    """

    repo_root = repo_root.resolve()
    root = _resolve_root(root=root, repo_root=repo_root)
    if not repo_root.is_dir():
        raise TrustBoundaryDiffError(f"--repo-root {repo_root} is not a directory")
    if not root.is_dir():
        raise TrustBoundaryDiffError(f"--root {root} is not a directory")

    rel_root = _relative_to_repo(root, repo_root)
    new_records: list[TrustBoundaryDecoratorRecord] = []
    for rel_path in _changed_python_files(baseline_ref=baseline_ref, rel_root=rel_root, repo_root=repo_root):
        head_path = repo_root / rel_path
        if not head_path.is_file():
            continue
        head_records = _records_from_file(head_path=head_path, rel_path=rel_path)
        if not head_records:
            continue
        baseline_text = _git_show(baseline_ref=baseline_ref, rel_path=rel_path, repo_root=repo_root)
        baseline_records = () if baseline_text is None else _records_from_source(source=baseline_text, source_file=rel_path)
        baseline_identities = {record.identity_hash for record in baseline_records}
        new_records.extend(record for record in head_records if record.identity_hash not in baseline_identities)

    return TrustBoundaryDiffReport(
        baseline_ref=baseline_ref,
        root=rel_root,
        new_decorators=tuple(sorted(new_records, key=lambda record: (record.source_file, record.line, record.symbol))),
    )


def render_trust_boundary_diff_summary(report: TrustBoundaryDiffReport) -> str:
    """Render a compact text summary suitable for ``GITHUB_STEP_SUMMARY``."""

    if not report.new_decorators:
        return f"No new @trust_boundary decorators detected against {report.baseline_ref}.\n"

    lines = [
        f"New @trust_boundary decorators: {len(report.new_decorators)}",
        f"Baseline: {report.baseline_ref}",
        f"Root: {report.root}",
    ]
    for record in report.new_decorators:
        parts = [f"- {record.source_file}:{record.line} {record.symbol}"]
        if record.source_param is not None:
            parts.append(f"source_param={record.source_param}")
        if record.suppresses:
            parts.append(f"suppresses={','.join(record.suppresses)}")
        if record.source is not None:
            parts.append(f"source={record.source}")
        if record.test_ref is not None:
            parts.append(f"test_ref={record.test_ref}")
        if not record.metadata_readable:
            parts.append("metadata=unreadable")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def _resolve_root(*, root: Path, repo_root: Path) -> Path:
    if root.is_absolute():
        return root.resolve()
    return (repo_root / root).resolve()


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise TrustBoundaryDiffError(f"{path} is not inside repo root {repo_root}") from exc
    if rel == Path("."):
        return "."
    return rel.as_posix()


def _changed_python_files(*, baseline_ref: str, rel_root: str, repo_root: Path) -> tuple[str, ...]:
    result = _run_git(
        ["diff", "--name-only", "-z", "--diff-filter=ACMRT", baseline_ref, "HEAD", "--", rel_root],
        repo_root=repo_root,
    )
    if result.returncode != 0:
        raise TrustBoundaryDiffError(f"git diff could not inspect baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")
    return tuple(path for path in result.stdout.split("\0") if path.endswith(".py"))


def _records_from_file(*, head_path: Path, rel_path: str) -> tuple[TrustBoundaryDecoratorRecord, ...]:
    try:
        source = head_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrustBoundaryDiffError(f"could not read {rel_path}: {exc}") from exc
    return _records_from_source(source=source, source_file=rel_path)


def _records_from_source(*, source: str, source_file: str) -> tuple[TrustBoundaryDecoratorRecord, ...]:
    try:
        tree = ast.parse(source, filename=source_file)
    except SyntaxError as exc:
        raise TrustBoundaryDiffError(f"could not parse {source_file}: {exc}") from exc
    visitor = _TrustBoundaryDecoratorVisitor(source_file=source_file)
    visitor.visit(tree)
    return tuple(visitor.records)


class _TrustBoundaryDecoratorVisitor(ast.NodeVisitor):
    def __init__(self, *, source_file: str) -> None:
        self.source_file = source_file
        self.records: list[TrustBoundaryDecoratorRecord] = []
        self._symbol_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._symbol_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self._symbol_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        symbol = ".".join([*self._symbol_stack, node.name])
        for call in find_trust_boundary_calls(node.decorator_list):
            self.records.append(_record_from_call(source_file=self.source_file, symbol=symbol, call=call))
        self._symbol_stack.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self._symbol_stack.pop()


def _record_from_call(*, source_file: str, symbol: str, call: ast.Call) -> TrustBoundaryDecoratorRecord:
    extraction = extract_keywords(call)
    metadata_readable = extraction.kwargs is not None
    kwargs = extraction.kwargs or {}
    source_param = _optional_string(kwargs.get("source_param"))
    suppresses = _string_tuple(kwargs.get("suppresses"))
    source = _optional_string(kwargs.get("source"))
    test_ref = _optional_string(kwargs.get("test_ref"))
    identity_hash = _identity_hash(
        {
            "source_file": source_file,
            "symbol": symbol,
            "source_param": source_param,
            "suppresses": suppresses,
            "source": source,
            "test_ref": test_ref,
            "metadata_readable": metadata_readable,
        }
    )
    return TrustBoundaryDecoratorRecord(
        source_file=source_file,
        line=call.lineno,
        symbol=symbol,
        source_param=source_param,
        suppresses=suppresses,
        source=source,
        test_ref=test_ref,
        metadata_readable=metadata_readable,
        identity_hash=identity_hash,
    )


def _optional_string(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, set, tuple)):
        return ()
    strings = [item for item in value if isinstance(item, str)]
    return tuple(sorted(strings))


def _identity_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _git_show(*, baseline_ref: str, rel_path: str, repo_root: Path) -> str | None:
    result = _run_git(["show", f"{baseline_ref}:{rel_path}"], repo_root=repo_root)
    if result.returncode == 0:
        return result.stdout
    if not _git_commit_exists(baseline_ref=baseline_ref, repo_root=repo_root):
        raise TrustBoundaryDiffError(f"git show could not resolve baseline-ref {baseline_ref!r}: {_git_failure_detail(result)}")
    if not _git_path_exists(baseline_ref=baseline_ref, rel_path=rel_path, repo_root=repo_root):
        return None
    raise TrustBoundaryDiffError(f"git show failed for baseline {baseline_ref}:{rel_path}: {_git_failure_detail(result)}")


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
        raise TrustBoundaryDiffError(f"git command failed to start: {exc}") from exc


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
