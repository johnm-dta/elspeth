"""Trust-boundary scope rule implementation.

Honesty gate: ``@trust_boundary(source_param=X)`` must name a parameter of
the decorated function, AND that parameter must be read in a way that
propagates the value somewhere (a "taint-receiving" operation). The runtime
decorator already raises :class:`TypeError` for the "not a parameter" case
at import time; this static rule catches the wider failure modes where
``source_param`` is a parameter but the body either never references it OR
only references it in a dead context — a bare expression-statement, an
assignment whose only target is the bare ``_``, or an assignment whose
only target is a leading-underscore name. Both dead-context patterns are
structurally inert: the decorator's suppression scope would cover nothing.

This rule deliberately does not prove the decorator's ``source`` prose against
a whole-repository external-data call graph. ``source`` is documentation for
reviewers; the mechanical contract here is local suppression scope plus the
companion tests rule's direct subject-call check.

A "taint-receiving read" is any :class:`ast.Name` in :class:`ast.Load`
context whose immediate parent propagates the value: a :class:`ast.Subscript`,
:class:`ast.Attribute`, :class:`ast.Call`, :class:`ast.For`/
:class:`ast.AsyncFor` (as the ``iter`` side), :class:`ast.comprehension`
(as the ``iter`` side), :class:`ast.Starred` unpacking, an
:class:`ast.AugAssign` / :class:`ast.AnnAssign` value, a binary /
comparison / boolean / conditional / formatted-string / await /
return / yield / raise / assert / match-subject expression, or an
:class:`ast.Assign` whose ``value`` is the Name and whose targets are not
all dead (bare ``_`` or a single leading-underscore :class:`ast.Name`).

Explicitly rejected as inert (NOT taint-receiving):

* a bare expression-statement (:class:`ast.Expr` whose ``value`` is the
  :class:`ast.Name` — e.g. ``source_param  # noqa: B018``);
* an assignment whose ``value`` is the bare :class:`ast.Name` and whose
  every target is a leading-underscore :class:`ast.Name`
  (e.g. ``_ = source_param`` or ``_unused = source_param``).

The dead-context exclusions are load-bearing for the C6-2 honesty-gate
hardening (epic elspeth-2ed3bb0f7d, ticket elspeth-9bbf3b66e9): without
them a single ``_ = source_param`` no-op line silently satisfied the
rule, defeating the entire scope honesty gate.

Annotations, default values, and decorator expressions on inner functions
are excluded — they belong to the enclosing scope, not to the boundary's
data flow.

Non-literal kwargs on the decorator now produce a TBS3 finding rather
than silently deferring to ``trust_tier.tier_model`` — see the
``trust_boundary.shared.KeywordExtraction`` docstring and ticket
elspeth-1f4634235a (C6-4).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import (
    PythonFileReadError,
    PythonSyntaxError,
    iter_own_scope,
    walk_python_files,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.scope.metadata import (
    RULE_DEAD,
    RULE_ID,
    RULE_METADATA,
    RULE_NONLITERAL,
    RULE_NOPARAM,
    SUGGESTION_DEAD,
    SUGGESTION_NONLITERAL,
    SUGGESTION_NOPARAM,
)
from elspeth_lints.rules.trust_boundary.shared import (
    display_path,
    extract_keywords,
    filter_allowlisted_findings,
    iter_trust_boundary_decorators,
    load_honesty_gate_allowlist,
    make_decorator_finding,
)

_ALLOWLIST_RULE_IDS = frozenset({RULE_NOPARAM, RULE_DEAD, RULE_NONLITERAL})


@dataclass(frozen=True, slots=True)
class TrustBoundaryScopeRule:
    """Detect ``@trust_boundary`` decorators with absent or dead ``source_param``."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one tree directly (for focused tests) or walk the scan root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(tree, display_path(file_path, context.root))
        return scan_root(context.root, allowlist_dir_override=context.allowlist_dir_override)


def analyze_tree(tree: ast.AST, file_path: str) -> list[Finding]:
    """Return ``trust_boundary.scope`` findings for one parsed syntax tree."""
    findings: list[Finding] = []
    for func_node, call in iter_trust_boundary_decorators(tree):
        extraction = extract_keywords(call)
        if extraction.kwargs is None:
            # Non-literal kwarg (or **-unpacking / positional args). Per the
            # C6-4 honesty-gate hardening (epic elspeth-2ed3bb0f7d, ticket
            # elspeth-1f4634235a) we self-enforce instead of deferring to
            # ``trust_tier.tier_model`` — suppressing tier_model on a file
            # must NOT grant honesty-gate immunity here. The redundant
            # finding when tier_model is also active is deliberate.
            assert extraction.nonliteral_message is not None  # tagged union invariant
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_NONLITERAL,
                    file_path=file_path,
                    call=call,
                    message=extraction.nonliteral_message,
                    suggestion=SUGGESTION_NONLITERAL,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        kwargs = extraction.kwargs
        source_param = kwargs.get("source_param")
        if not isinstance(source_param, str) or not source_param:
            # Missing or malformed source_param is the tier_model rule's
            # R_TB_MALFORMED territory; do not double-report here.
            continue
        param_names = _parameter_names(func_node)
        if source_param not in param_names:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_NOPARAM,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary(source_param={source_param!r}) does not name a parameter of "
                        f"{func_node.name!r}; declared parameters are {tuple(param_names)!r}."
                    ),
                    suggestion=SUGGESTION_NOPARAM,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        if not _body_reads_name(func_node, source_param):
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_DEAD,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary(source_param={source_param!r}) names a parameter of "
                        f"{func_node.name!r} but the function body never reads it; "
                        "the decorator is structurally inert."
                    ),
                    suggestion=SUGGESTION_DEAD,
                    symbol_context=(func_node.name,),
                )
            )
    return findings


def scan_root(root: Path, *, allowlist_dir_override: Path | None = None) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings."""
    allowlist = load_honesty_gate_allowlist(
        root,
        allowlist_dir_override=allowlist_dir_override,
        valid_rule_ids=_ALLOWLIST_RULE_IDS,
    )
    findings: list[Finding] = []
    for item in walk_python_files(root):
        # Skip non-analysable per-file results — see the analogous comment
        # in trust_boundary/tests/rule.py::scan_root.
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(analyze_tree(item.tree, display_path(item.path, root)))
    return filter_allowlisted_findings(findings, allowlist)


def _parameter_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Collect every parameter name in declaration order.

    Includes positional-only, regular, ``*args``, keyword-only, and ``**kwargs``
    parameters. ``self`` and ``cls`` are NOT filtered out — the runtime
    decorator validates against :func:`inspect.signature`, which sees them too,
    so the static rule must match that surface to avoid divergence.
    """
    args = func_node.args
    names: list[str] = []
    names.extend(arg.arg for arg in args.posonlyargs)
    names.extend(arg.arg for arg in args.args)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(arg.arg for arg in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _body_reads_name(func_node: ast.FunctionDef | ast.AsyncFunctionDef, name: str) -> bool:
    """Return True if the function body reads ``name`` in a taint-receiving context.

    A "taint-receiving read" is any :class:`ast.Name` with :class:`ast.Load`
    context whose ``id`` equals ``name`` AND whose immediate AST parent
    propagates the value — see the module docstring for the full
    enumeration. Bare expression-statements (``source_param  # noqa: B018``)
    and dead assignments (``_ = source_param``, ``_unused = source_param``)
    are NOT taint-receiving and do not satisfy the contract.

    Walks every statement in the body's own lexical scope, **excluding**
    the bodies of any nested ``FunctionDef``, ``AsyncFunctionDef``,
    ``Lambda``, or ``ClassDef``. Annotations on the function itself,
    decorators, and default-value expressions are also out of scope
    (they belong to the enclosing definition, not to the body's flow).

    Excluding nested-scope bodies is load-bearing for the DEAD-parameter
    check: an outer function that takes ``data`` as a parameter but never
    reads it, while containing an inner function (or class with a
    class-attribute assignment) that reads ``data`` lexically inherited
    from the outer scope, must still be flagged as DEAD. The outer
    parameter would still be inert from the outer function's body's
    perspective — the inner read happens inside a different lexical scope
    that the @trust_boundary decorator does not apply to. The previous
    implementation used ``ast.walk(statement)``, which descended into the
    inner function's body and silently treated the inner read as if it
    satisfied the outer's parameter contract, masking the bug.

    ``ClassDef`` was added to the short-circuit set after the audit
    found the symmetric leak: a nested class body that read the outer
    parameter (``class Helper: raw = data["x"]``) registered as a
    body-read for the outer function, falsely clearing TBS2.

    The taint-receiving restriction was added in C6-2 (epic
    elspeth-2ed3bb0f7d, ticket elspeth-9bbf3b66e9): before that, ANY
    Load-context Name reference satisfied the predicate, so a single
    ``_ = source_param`` no-op line defeated the entire scope honesty
    gate.
    """
    # Walk only the body statements (not the function's own decorators,
    # parameter annotations, or return annotation — those belong to the
    # enclosing definition, not to the body's flow). For each statement,
    # use scope-respecting iteration so nested-scope bodies (function,
    # async function, lambda, class) don't leak their reads of the outer
    # parameter into the outer body's "did we read it?" decision.
    #
    # We build the parent map per statement (cheap; typical statements
    # have small AST subtrees) so the predicate can classify each
    # candidate Name by its immediate AST parent.
    for statement in func_node.body:
        parents = _build_parent_map(statement)
        for child in iter_own_scope(statement):
            if isinstance(child, ast.Name) and child.id == name and isinstance(child.ctx, ast.Load) and _is_taint_receiving(child, parents):
                return True
    return False


def _build_parent_map(root: ast.AST) -> dict[int, ast.AST]:
    """Return ``{id(child): parent_node}`` for every node reachable in ``root``'s own scope.

    Scope-respecting in the same sense as :func:`iter_own_scope`: the parent
    map covers ``root`` and its descendants, short-circuiting at any nested
    ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` / ``ClassDef`` —
    the nested-scope node itself is included as a child of its parent, but
    its own children are not mapped. Keys are :func:`id` of the child node
    (AST nodes are not hashable by value; the identity-based key is safe
    because each AST node has a unique :func:`id` within the tree).
    """
    parents: dict[int, ast.AST] = {}
    for node in iter_own_scope(root):
        if isinstance(node, _NESTED_SCOPE_TYPES_FOR_PARENTS):
            # Do not descend into the nested-scope node's children for the
            # parent map; ``iter_own_scope`` already skipped them, so we
            # match its behaviour explicitly here.
            continue
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


# Mirror of ``_NESTED_SCOPE_TYPES`` in ``elspeth_lints.core.ast_walker``;
# kept private here to avoid an import surface change for a one-line tuple.
_NESTED_SCOPE_TYPES_FOR_PARENTS: tuple[type, ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)


def _is_taint_receiving(name_node: ast.Name, parents: dict[int, ast.AST]) -> bool:
    """Return True if ``name_node`` participates in a taint-receiving operation.

    The classification is done by inspecting the immediate AST parent — see
    the module docstring for the full enumeration. The two explicit
    rejections (bare expression-statement and dead assignment) are the
    C6-2 honesty-gate hardening: without them a ``_ = source_param``
    no-op line silently satisfied the scope rule.
    """
    parent = parents.get(id(name_node))
    if parent is None:
        # Defensive: a Name with no parent in our map means it was the
        # root we iterated (e.g. a top-level expression-statement value
        # passed directly). Without a parent context we cannot prove it
        # propagates, so reject — matches the bare-expression-statement
        # rejection semantically.
        return False
    # Bare expression-statement: ``source_param`` on its own line.
    if isinstance(parent, ast.Expr) and parent.value is name_node:
        return False
    # Dead assignment: ``_ = source_param`` or ``_unused = source_param``.
    # The Name must BE the assignment value (not nested inside something
    # else) and every target must be a leading-underscore plain Name. A
    # nested context (``_ = source_param["k"]``) lands on the Subscript
    # branch first; this branch handles only the bare-Name value.
    if isinstance(parent, ast.Assign) and parent.value is name_node:
        return not all(_is_dead_target(target) for target in parent.targets)
    # Every other parent context propagates: Subscript, Attribute, Call
    # (callee or argument), For.iter, comprehension.iter, Starred,
    # AugAssign / AnnAssign value, BinOp, BoolOp, Compare, IfExp, JoinedStr
    # / FormattedValue (f-string), Await, Return, Yield, YieldFrom, Raise,
    # Assert, Match, NamedExpr, plus collection literals (Tuple, List,
    # Set, Dict) which can be returned / bound / passed. The closed list of
    # rejections above is exhaustive — if neither matches, the Name flows.
    return True


def _is_dead_target(target: ast.expr) -> bool:
    """Return True if ``target`` is a leading-underscore plain :class:`ast.Name`.

    The two recognised dead-target shapes are ``Name("_")`` and
    ``Name("_unused")`` / any other id whose first character is ``_``.
    Subscript / attribute / tuple-unpacking targets are NOT dead — they
    bind to a real object location or to multiple names, propagating the
    value somewhere observable.
    """
    return isinstance(target, ast.Name) and target.id.startswith("_")


RULE = TrustBoundaryScopeRule()
