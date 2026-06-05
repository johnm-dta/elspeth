"""Shared AST helpers for ``@trust_boundary`` honesty-gate rules.

These helpers duplicate the decorator-recognition shape already used by
``trust_tier.tier_model.trust_boundary_suppress`` deliberately: the honesty
gates must remain independent of the suppression walk so a refactor in either
direction cannot accidentally couple the two. The recognition shape itself is
small and stable (matches the runtime decorator's import surface in
``src/elspeth/contracts/trust_boundary.py``).

Recognised decorator spellings (matching the runtime import surface):

* ``@trust_boundary(...)`` after ``from elspeth.contracts.trust_boundary import trust_boundary``;
* ``@elspeth.contracts.trust_boundary(...)`` (fully qualified attribute chain);
* ``@contracts.trust_boundary(...)`` (shortened attribute chain).
* ``@tb_mod.trust_boundary(...)`` after importing the decorator module as an alias.

Bare-decorator usage (``@trust_boundary`` without a call) is not recognised:
the runtime decorator is keyword-only and never appears without arguments.

Async functions decorated with ``@trust_boundary`` are recognised
identically — the decorator composes with any callable. Both
:class:`ast.FunctionDef` and :class:`ast.AsyncFunctionDef` are yielded by
:func:`iter_trust_boundary_decorators`.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from elspeth_lints.core.allowlist import Allowlist, AllowlistEntry, FindingKey, load_allowlist, verify_entry_binding_against_finding
from elspeth_lints.core.allowlist_governance import allowlist_governance_findings
from elspeth_lints.core.protocols import Finding, RuleMetadata

_TRUST_BOUNDARY_EXPORT = "elspeth.contracts.trust_boundary"
_TRUST_BOUNDARY_FUNCTION = "elspeth.contracts.trust_boundary.trust_boundary"
_TRUST_BOUNDARY_QUALIFIED_NAMES: frozenset[str] = frozenset(
    {
        _TRUST_BOUNDARY_EXPORT,
        _TRUST_BOUNDARY_FUNCTION,
    }
)
TRUST_BOUNDARY_ALLOWLIST_DIR = "enforce_trust_boundary_honesty"


def _dotted_name(expr: ast.expr) -> tuple[str, ...] | None:
    if isinstance(expr, ast.Name):
        return (expr.id,)
    if isinstance(expr, ast.Attribute):
        base = _dotted_name(expr.value)
        if base is None:
            return None
        return (*base, expr.attr)
    return None


def _matches_elspeth_trust_boundary_import(func: ast.expr, import_aliases: Mapping[str, str]) -> bool:
    parts = _dotted_name(func)
    if parts is None:
        return False

    alias_target = import_aliases.get(parts[0])
    dotted = ".".join(parts)
    if alias_target is None:
        return dotted in _TRUST_BOUNDARY_QUALIFIED_NAMES

    target_parts = tuple(alias_target.split("."))
    if parts[: len(target_parts)] == target_parts:
        resolved = dotted
    else:
        resolved = ".".join((*target_parts, *parts[1:]))
    return resolved in _TRUST_BOUNDARY_QUALIFIED_NAMES


def _is_trust_boundary_decorator(decorator: ast.expr, *, import_aliases: Mapping[str, str]) -> ast.Call | None:
    """Return the ``ast.Call`` if ``decorator`` references ``trust_boundary``.

    See module docstring for the recognised spellings. Returns ``None`` for any
    other decorator shape; callers iterate the full decorator list.
    """
    if not isinstance(decorator, ast.Call):
        return None
    return decorator if _matches_elspeth_trust_boundary_import(decorator.func, import_aliases) else None


class _TrustBoundaryDecoratorVisitor(ast.NodeVisitor):
    """Find Elspeth trust-boundary decorators with lexical import awareness."""

    def __init__(self) -> None:
        self.matches: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]] = []
        self._import_alias_stack: list[dict[str, str]] = [{}]

    @property
    def _import_aliases(self) -> dict[str, str]:
        return self._import_alias_stack[-1]

    def _push_scope(self) -> None:
        self._import_alias_stack.append(dict(self._import_aliases))

    def _pop_scope(self) -> None:
        self._import_alias_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root_name = alias.name.split(".", 1)[0]
            self._import_aliases[alias.asname or root_name] = alias.name

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is None or node.level != 0:
            return
        for alias in node.names:
            if alias.name == "*":
                continue
            self._import_aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._push_scope()
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._pop_scope()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            call = _is_trust_boundary_decorator(decorator, import_aliases=self._import_aliases)
            if call is not None:
                self.matches.append((node, call))
                # A function should have at most one @trust_boundary; if a
                # malformed decorator stack accidentally repeats it, we still
                # yield the first only — duplicate findings would be noise.
                break

        self._push_scope()
        try:
            for statement in node.body:
                self.visit(statement)
        finally:
            self._pop_scope()


def iter_trust_boundary_decorators(
    tree: ast.AST,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.Call]]:
    """Yield ``(function_node, decorator_call)`` for every ``@trust_boundary`` in ``tree``.

    Walks the whole tree (module, classes, nested functions). The decorator
    call yielded is the ``ast.Call`` node — callers use
    :func:`extract_keywords` to read its kwargs.

    Ordering is deterministic: statements are visited in lexical order while
    preserving import aliases visible at each decorator site.
    """
    visitor = _TrustBoundaryDecoratorVisitor()
    visitor.visit(tree)
    yield from visitor.matches


def _literal_value(node: ast.expr) -> tuple[bool, object]:
    """Try to extract a static literal from an AST expression.

    Returns ``(ok, value)``. Allowed shapes:

    * :class:`ast.Constant` scalars (``str``, ``int``, ``bool``, ``None``,
      ``float``, ``bytes``);
    * :class:`ast.Tuple`, :class:`ast.List`, :class:`ast.Set`,
      :class:`ast.Dict` composed recursively of allowed shapes.

    Anything referencing a name, a call, an attribute, an f-string, or a
    comprehension is treated as non-literal. The honesty-gate rules need
    static metadata; an unverifiable value is a rule violation in itself —
    each honesty rule self-enforces by emitting its own
    ``R_TB_NONLITERAL`` finding on top of any duplicate finding emitted by
    ``trust_tier.tier_model``. The redundancy is deliberate per the
    honesty-gate cross-rule-bypass remediation (epic elspeth-2ed3bb0f7d,
    ticket elspeth-1f4634235a / C6-4): suppressing ``trust_tier.tier_model``
    on a file must NOT grant honesty-gate immunity.
    """
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.Tuple):
        items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            items.append(value)
        return True, tuple(items)
    if isinstance(node, ast.List):
        list_items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            list_items.append(value)
        return True, list_items
    if isinstance(node, ast.Set):
        set_items: list[object] = []
        for elt in node.elts:
            ok, value = _literal_value(elt)
            if not ok:
                return False, None
            set_items.append(value)
        return True, set(set_items)
    if isinstance(node, ast.Dict):
        result: dict[object, object] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=False):
            if key_node is None:
                return False, None
            ok_key, key_val = _literal_value(key_node)
            if not ok_key:
                return False, None
            ok_val, val_val = _literal_value(value_node)
            if not ok_val:
                return False, None
            result[key_val] = val_val
        return True, result
    return False, None


@dataclass(frozen=True, slots=True)
class KeywordExtraction:
    """Outcome of parsing a ``@trust_boundary(...)`` call's kwargs.

    Exactly one of ``kwargs`` and ``nonliteral_message`` is non-``None``:

    * ``kwargs`` is the literal-value dict when every kwarg parsed
      successfully — the rule walks the dict as before.
    * ``nonliteral_message`` is a human-readable diagnostic when any kwarg
      cannot be statically evaluated (a name reference, a call, a
      comprehension), uses ``**kwargs``-unpacking, or the call has
      positional arguments. The caller emits this as an
      ``R_TB_NONLITERAL`` finding on the decorator call site.

    Why a tagged result rather than ``dict | None``: the previous
    ``None`` sentinel made every caller ``continue`` silently, deferring
    to the ``trust_tier.tier_model`` rule. That created a cross-rule
    bypass — suppressing ``tier_model`` on a file granted honesty-gate
    immunity for non-literal decorators on that same file. The honesty
    gates must self-enforce literal-only kwargs; see
    epic elspeth-2ed3bb0f7d, ticket elspeth-1f4634235a (C6-4). The
    redundant finding when ``tier_model`` is also active is deliberate.
    """

    kwargs: dict[str, object] | None
    nonliteral_message: str | None


def extract_keywords(call: ast.Call) -> KeywordExtraction:
    """Return the decorator's kwargs as a tagged result.

    Positional arguments are rejected (the runtime decorator is keyword-only;
    a positional-arg call is malformed). ``**kwargs``-style unpacking is
    rejected (``keyword.arg is None``). Any kwarg whose value cannot be
    literal-evaluated is also rejected. In all three rejection cases the
    returned :class:`KeywordExtraction` has ``kwargs is None`` and
    ``nonliteral_message`` populated with a human-readable diagnostic;
    callers must emit an ``R_TB_NONLITERAL`` finding rather than silently
    skipping (which would defer to ``trust_tier.tier_model`` and create a
    cross-rule bypass — see :class:`KeywordExtraction`).

    Returns:
        :class:`KeywordExtraction` — ``kwargs`` populated on success,
        ``nonliteral_message`` populated when any kwarg is non-literal, the
        call has positional args, or ``**kwargs`` unpacking is used.
    """
    if call.args:
        return KeywordExtraction(
            kwargs=None,
            nonliteral_message=(
                "@trust_boundary received positional arguments; the decorator is keyword-only and its honesty-gate metadata cannot be read."
            ),
        )
    parsed: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return KeywordExtraction(
                kwargs=None,
                nonliteral_message=(
                    "@trust_boundary uses **-unpacking for its arguments; static analysis cannot read the honesty-gate metadata."
                ),
            )
        ok, value = _literal_value(keyword.value)
        if not ok:
            return KeywordExtraction(
                kwargs=None,
                nonliteral_message=(
                    f"@trust_boundary kwarg {keyword.arg!r} is not a static "
                    "literal (e.g. it references a name, a call, an attribute, "
                    "or a comprehension); the honesty-gate metadata is "
                    "unverifiable so the decorator cannot be trusted to mark a "
                    "real trust boundary."
                ),
            )
        parsed[keyword.arg] = value
    return KeywordExtraction(kwargs=parsed, nonliteral_message=None)


def display_path(file_path: Path, root: Path) -> str:
    """Return the path format used by elspeth-lints rules for this scan root.

    Matches the convention in
    :func:`elspeth_lints.rules.immutability.shared.display_path`: a
    forward-slash POSIX path relative to ``root``, or the absolute path if the
    file is outside the scan root.
    """
    try:
        return file_path.relative_to(root).as_posix()
    except ValueError:
        return file_path.as_posix()


def repository_root(root: Path, explicit_repo_root: Path | None = None) -> Path:
    """Return the repository root from a scan root.

    If ``explicit_repo_root`` is provided, it wins. This lets CI use a
    non-canonical scan root while still resolving repository-relative
    evidence paths such as ``tests/...`` deterministically.

    When the scan root is ``<repo>/src/elspeth`` (the canonical invocation),
    the repository root is ``root.parent.parent``. Otherwise the scan root
    itself is the repository root. Same heuristic as
    :func:`elspeth_lints.rules.audit_evidence.shared.repo_relative_display_path`.
    The ``trust_boundary.tests`` rule needs this to resolve ``test_ref``
    pytest nodeids that point to ``tests/...`` paths even though the scan
    root is ``src/elspeth/``.
    """
    if explicit_repo_root is not None:
        return explicit_repo_root
    if root.name == "elspeth" and root.parent.name == "src":
        return root.parent.parent
    return root


def allowlist_path_for_root(root: Path) -> Path:
    """Return the shared trust-boundary honesty-gate allowlist path.

    The canonical scan root in CI is ``<repo>/src/elspeth`` while allowlist
    governance lives under ``<repo>/config/cicd``. This mirrors tier_model's
    upward discovery, but all three trust-boundary honesty gates share one
    directory so escape entries stay in one audited surface.
    """
    relative_dir = Path("config") / "cicd" / TRUST_BOUNDARY_ALLOWLIST_DIR
    local_dir = root / relative_dir
    if local_dir.is_dir():
        return local_dir

    if root.name == "elspeth" and root.parent.name == "src":
        for candidate in root.parents:
            dir_path = candidate / relative_dir
            if dir_path.is_dir():
                return dir_path
    return local_dir


def load_honesty_gate_allowlist(
    root: Path,
    *,
    allowlist_dir_override: Path | None,
    valid_rule_ids: frozenset[str],
) -> Allowlist:
    """Load the narrow allowlist surface for trust-boundary honesty gates.

    Honesty-gate findings are suppressible only as exact, judge-attested
    entries. Pre-judge grandfathering and per-file blanket rules are rejected
    here because these rules guard the decorator's own truthfulness; an escape
    must be a reviewed exception to one finding, not a new suppression language.
    """
    allowlist_path = allowlist_dir_override if allowlist_dir_override is not None else allowlist_path_for_root(root)
    if not allowlist_path.exists():
        if allowlist_dir_override is not None:
            raise FileNotFoundError(f"{allowlist_path}: trust-boundary allowlist directory does not exist")
        return Allowlist(entries=[])
    allowlist = load_allowlist(allowlist_path, valid_rule_ids=valid_rule_ids, source_root=root)
    _validate_honesty_gate_allowlist(allowlist, valid_rule_ids=valid_rule_ids)
    return allowlist


def filter_allowlisted_findings(
    findings: list[Finding],
    allowlist: Allowlist,
    *,
    allowlist_dir: Path | None = None,
    governance_emitted_dirs: set[str] | None = None,
    emit_allowlist_governance: bool = True,
) -> list[Finding]:
    """Return findings not covered by the trust-boundary allowlist."""
    active = [finding for finding in findings if _allowlist_match(allowlist, finding) is None]
    if allowlist_dir is None:
        return active
    return [
        *active,
        *allowlist_governance_findings(
            allowlist,
            allowlist_dir,
            emitted_dirs=governance_emitted_dirs,
            enabled=emit_allowlist_governance,
        ),
    ]


def _validate_honesty_gate_allowlist(allowlist: Allowlist, *, valid_rule_ids: frozenset[str]) -> None:
    if allowlist.per_file_rules:
        source_files = ", ".join(sorted({rule.source_file for rule in allowlist.per_file_rules if rule.source_file}))
        suffix = f" in {source_files}" if source_files else ""
        raise ValueError(f"trust-boundary honesty gates only allow exact allow_hits; per_file_rules are forbidden{suffix}")

    for entry in allowlist.entries:
        rule_id = _rule_id_from_canonical_key(entry.key)
        source_ctx = f" in {entry.source_file}" if entry.source_file else ""
        if rule_id not in valid_rule_ids:
            raise ValueError(f"allow_hits entry has unknown trust-boundary honesty rule id {rule_id!r}{source_ctx}: {entry.key}")
        if entry.judge_verdict is None:
            raise ValueError(f"trust-boundary honesty allow_hits entry must carry judge_verdict metadata{source_ctx}: {entry.key}")
        if entry.expires is None:
            raise ValueError(f"trust-boundary honesty allow_hits entry must expire{source_ctx}: {entry.key}")


def _allowlist_match(allowlist: Allowlist, finding: Finding) -> object | None:
    matched = allowlist.match(
        FindingKey(
            file_path=finding.file_path,
            rule_id=finding.rule_id,
            symbol_context=finding.symbol_context,
            fingerprint=finding.fingerprint,
        )
    )
    if isinstance(matched, AllowlistEntry) and _is_expired_entry(matched):
        matched.matched = False
        return None
    if isinstance(matched, AllowlistEntry) and matched.judge_verdict is not None:
        if not finding.ast_path:
            raise ValueError(
                f"trust-boundary finding {finding.canonical_key()} has no ast_path; "
                "signed allowlist entries require binding to the inspected AST node."
            )
        # trust_boundary findings are protocols.Finding, which has no scope_fingerprint
        # field. They are v1-only today (no signed v2 trust_boundary entries on disk), so
        # the v1 branch ignores this value. A FUTURE v2 trust_boundary entry would hit the
        # "scope_fingerprint missing on the live finding" crash above — correct fail-closed
        # behaviour and a clear signal to wire trust_boundary's scanner to stamp the field.
        # Do NOT fabricate a value; the empty string is the honest "not computed" signal.
        verify_entry_binding_against_finding(
            matched,
            file_path=finding.file_path,
            ast_path=finding.ast_path,
            scope_fingerprint=getattr(finding, "scope_fingerprint", ""),
        )
    return matched


def _is_expired_entry(entry: AllowlistEntry) -> bool:
    return entry.expires is not None and entry.expires < datetime.now(UTC).date()


def _rule_id_from_canonical_key(key: str) -> str:
    marker = ".py:"
    py_index = key.find(marker)
    if py_index < 0 or ":fp=" not in key:
        raise ValueError(f"allowlist key is not in canonical finding form: {key!r}")
    remainder = key[py_index + len(marker) :]
    rule_id, sep, _symbol_and_fingerprint = remainder.partition(":")
    if not sep or not rule_id:
        raise ValueError(f"allowlist key is missing its rule-id segment: {key!r}")
    return rule_id


def make_decorator_finding(
    *,
    metadata: RuleMetadata,
    rule_id: str,
    file_path: str,
    call: ast.Call,
    message: str,
    suggestion: str,
    symbol_context: tuple[str, ...] = (),
) -> Finding:
    """Build a finding for a ``@trust_boundary`` decorator call site.

    All trust-boundary honesty gates fingerprint decorator findings with
    the same stable payload: ``rule_id|file_path|lineno|col``. Keeping this
    helper in ``shared`` prevents the three rules from drifting if the
    fingerprint shape changes again.
    """
    fingerprint = hashlib.sha256(f"{rule_id}|{file_path}|{call.lineno}|{call.col_offset}".encode()).hexdigest()[:16]
    return Finding(
        rule_id=rule_id,
        file_path=file_path,
        line=call.lineno,
        column=call.col_offset,
        message=message,
        fingerprint=fingerprint,
        severity=metadata.severity,
        suggestion=suggestion,
        symbol_context=symbol_context,
        ast_path=f"decorator:{call.lineno}:{call.col_offset}",
    )
