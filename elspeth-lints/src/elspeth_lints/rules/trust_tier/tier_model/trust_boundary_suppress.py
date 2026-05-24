"""@trust_boundary decorator awareness for the tier_model rule.

This module implements decorator-driven suppression of Tier-3 defensive-pattern
findings inside functions marked with the project's
``elspeth.contracts.trust_boundary.trust_boundary`` decorator. Suppression is
strictly scoped:

* the finding's ``rule_id`` must appear in the decorator's ``suppresses`` tuple;
* the AST subject the finding is reported against must be **rooted at** the
  parameter named by the decorator's ``source_param`` (or at any name derived
  from it through subscript, attribute, ``.get(...)``, iteration, unpacking,
  walrus, or assignment).

Findings outside decorated functions are unaffected, and findings inside a
decorated function whose subject is not rooted at the boundary parameter
remain visible — the decorator is **not** a whole-function exemption cloak.

Auditability
------------

This rule has no telemetry surface, so the suppression decision IS the audit
signal: the absence of a finding in the rule's report corresponds to a
``@trust_boundary`` decorator that a human reader can locate via the
function's qualified name. The decorator's ``source``, ``invariant``,
``test_ref``, and ``test_fingerprint`` fields stand as the in-source
justification — they are the permanent record of "why no R1 was emitted here".
The companion ``enforce_trust_boundary_*`` CI gates (separate scripts) verify
the local test and scope claims; ``source`` remains reviewer-facing
documentation, not a whole-repository external-data graph proof. This module
only consumes the metadata that those gates also read.

Malformed metadata is **not silently honoured**: a decorator that fails to
expose literal kwargs (``R_TB_NONLITERAL``), has wrong-shaped literals
(``R_TB_MALFORMED``), contains unknown kwargs (``R_TB_UNKNOWN_KWARG``), or
appears more than once on a function (``R_TB_STACKED``) is treated as inert
(no suppression) AND emits a finding of its own so the malformed decorator
cannot accidentally hide real violations.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass

from elspeth_lints.core.ast_walker import walk_function_own_scope

# Closed set of rule IDs the ``@trust_boundary`` decorator is permitted to
# silence. Mirrors the runtime decorator's ``BoundaryRule = Literal["R1",
# "R5"]`` type — that constraint is enforced by mypy at the decorator's call
# site but NOT at the analyzer's parse path (the analyzer reads YAML-like
# kwargs from a static AST and would otherwise honour any string the author
# typed). A decorator that names an unauthorised rule ID is treated as
# malformed: the metadata becomes inert (the decorator does NOT suppress
# anything) AND an ``R_TB_MALFORMED`` finding is emitted so the malformed
# decorator cannot quietly hide real violations.
_ALLOWED_BOUNDARY_RULES: frozenset[str] = frozenset({"R1", "R5"})
_ALLOWED_TRUST_BOUNDARY_KWARGS: frozenset[str] = frozenset(
    {
        "tier",
        "source",
        "source_param",
        "suppresses",
        "invariant",
        "test_ref",
        "test_fingerprint",
    }
)


@dataclass(frozen=True, slots=True)
class BoundaryMetadata:
    """Parsed metadata for a ``@trust_boundary``-decorated function.

    ``source_param`` and ``suppresses`` drive the suppression decision.
    ``source``, ``invariant``, ``test_ref``, and ``test_fingerprint`` are
    carried for the non-failing ``R_TB_SUPPRESSED`` observation stream, so an
    operator can audit which decorator metadata justified the hidden R1/R5
    finding without re-reading the source file by hand.
    """

    suppresses: frozenset[str]
    source_param: str
    decorator_node: ast.Call
    source: str | None
    invariant: str | None
    test_ref: str | None
    test_fingerprint: str | None


@dataclass(frozen=True, slots=True)
class BoundaryFinding:
    """Diagnostic about a malformed ``@trust_boundary`` decorator.

    Returned alongside ``BoundaryMetadata`` (or in its place) by
    :func:`extract_boundary_metadata`. The caller surfaces these as ordinary
    rule findings via the visitor's ``_add_finding`` helper so they appear in
    the same JSON/text report as R1/R5/etc.
    """

    rule_id: str  # "R_TB_NONLITERAL", "R_TB_MALFORMED", "R_TB_UNKNOWN_KWARG", or "R_TB_STACKED"
    message: str
    node: ast.Call


_TRUST_BOUNDARY_NAME = "trust_boundary"


def _is_trust_boundary_decorator(decorator: ast.expr) -> ast.Call | None:
    """Return the ``ast.Call`` if ``decorator`` references ``trust_boundary``.

    Recognises all three documented spellings:

    * ``@trust_boundary(...)`` after ``from elspeth.contracts import trust_boundary``;
    * ``@elspeth.contracts.trust_boundary(...)``;
    * ``@contracts.trust_boundary(...)``.

    All three reduce to either ``ast.Name(id="trust_boundary")`` or an
    ``ast.Attribute`` chain terminating in ``attr="trust_boundary"`` at the
    ``Call.func`` site. Bare-decorator usage (``@trust_boundary`` without a
    call) is not recognised — the decorator is keyword-only and never appears
    without arguments — so we only accept ``ast.Call`` here.
    """
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if isinstance(func, ast.Name) and func.id == _TRUST_BOUNDARY_NAME:
        return decorator
    if isinstance(func, ast.Attribute) and func.attr == _TRUST_BOUNDARY_NAME:
        return decorator
    return None


def find_trust_boundary_calls(decorator_list: Iterable[ast.expr]) -> list[ast.Call]:
    """Locate all ``@trust_boundary(...)`` calls in a function's decorator list.

    Order-independent: one trust-boundary decorator may appear anywhere in the
    stack (above or below other decorators). More than one trust-boundary
    decorator is ambiguous because only one ``source_param`` / ``suppresses``
    tuple can be the audit signal for the function, so callers must reject
    duplicate matches.
    """
    calls: list[ast.Call] = []
    for decorator in decorator_list:
        call = _is_trust_boundary_decorator(decorator)
        if call is not None:
            calls.append(call)
    return calls


def _literal_value(node: ast.expr) -> tuple[bool, object]:
    """Try to literal-eval an AST expression.

    Returns ``(ok, value)``. We deliberately avoid :func:`ast.literal_eval` on
    the raw text because we already have the parsed AST; we walk it directly
    so we can be precise about what we accept. Allowed shapes: ``ast.Constant``
    scalars, and tuples/lists/sets/dicts/frozensets composed recursively of
    allowed shapes.

    Anything else (a ``Name``, a ``Call``, a comprehension, an attribute
    access, an f-string) is treated as non-literal — the static analyzer
    cannot prove its value, so the decorator's metadata is unverifiable and
    the decorator must be treated as inert.
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


def extract_boundary_metadata(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[BoundaryMetadata | None, list[BoundaryFinding]]:
    """Parse a function's ``@trust_boundary`` decorator, if any.

    Returns ``(metadata, diagnostics)``. ``metadata`` is ``None`` either
    because no decorator is present OR because the decorator was malformed
    (in which case ``diagnostics`` contains the appropriate finding); a
    malformed decorator is treated as inert so the rest of the visitor walk
    behaves identically to the un-decorated case.

    Diagnostic rule IDs:

    * ``R_TB_NONLITERAL`` — a kwarg's value isn't a static literal (a Name,
      a Call, a comprehension, etc.). Static analysis cannot read the
      metadata, so the suppression is rejected.
    * ``R_TB_MALFORMED`` — kwargs are present and literal, but the shape is
      wrong: ``suppresses`` is not a tuple of strings, ``source_param`` is
      not a string, etc.
    * ``R_TB_UNKNOWN_KWARG`` — kwargs are literal but include names outside
      the runtime decorator's documented signature.
    * ``R_TB_STACKED`` — the function carries multiple ``@trust_boundary``
      decorators, so the analyzer refuses to choose one suppression claim.
    """
    calls = find_trust_boundary_calls(func_node.decorator_list)
    if not calls:
        return None, []
    if len(calls) > 1:
        return None, [
            BoundaryFinding(
                rule_id="R_TB_STACKED",
                message=(
                    "multiple @trust_boundary decorators found on one function; "
                    "trust-boundary metadata is ambiguous, so static analysis "
                    "will not suppress findings under this function."
                ),
                node=calls[1],
            )
        ]

    call = calls[0]

    diagnostics: list[BoundaryFinding] = []
    parsed: dict[str, object] = {}

    if call.args:
        # The runtime decorator is keyword-only; positional args mean the
        # author bypassed the signature. Treat as malformed.
        diagnostics.append(
            BoundaryFinding(
                rule_id="R_TB_MALFORMED",
                message=(
                    "@trust_boundary received positional arguments; the decorator is keyword-only. "
                    "Static analysis cannot suppress findings under this decorator."
                ),
                node=call,
            )
        )
        return None, diagnostics

    for keyword in call.keywords:
        if keyword.arg is None:
            # **kwargs unpacking — non-static.
            diagnostics.append(
                BoundaryFinding(
                    rule_id="R_TB_NONLITERAL",
                    message=("@trust_boundary uses **-unpacking for its arguments; static analysis cannot read the metadata."),
                    node=call,
                )
            )
            return None, diagnostics
        ok, value = _literal_value(keyword.value)
        if not ok:
            diagnostics.append(
                BoundaryFinding(
                    rule_id="R_TB_NONLITERAL",
                    message=(
                        f"@trust_boundary kwarg {keyword.arg!r} is not a static literal "
                        "(e.g. it references a name, a call, or a comprehension). "
                        "Static analysis cannot read the metadata and cannot suppress findings."
                    ),
                    node=call,
                )
            )
            return None, diagnostics
        parsed[keyword.arg] = value

    unknown_kwargs = sorted(set(parsed) - _ALLOWED_TRUST_BOUNDARY_KWARGS)
    if unknown_kwargs:
        diagnostics.append(
            BoundaryFinding(
                rule_id="R_TB_UNKNOWN_KWARG",
                message=(
                    "@trust_boundary received unknown kwarg(s) "
                    f"{tuple(unknown_kwargs)!r}; static analysis treats the "
                    "decorator as inert because the metadata does not match "
                    "the runtime decorator signature."
                ),
                node=call,
            )
        )
        return None, diagnostics

    # Shape validation of the two fields suppression depends on.
    suppresses_raw = parsed.get("suppresses")
    source_param_raw = parsed.get("source_param")

    shape_errors: list[str] = []
    if suppresses_raw is None:
        shape_errors.append("missing kwarg 'suppresses'")
    elif not isinstance(suppresses_raw, tuple):
        shape_errors.append(f"'suppresses' must be a tuple of rule IDs, got {type(suppresses_raw).__name__}")
    elif not all(isinstance(item, str) for item in suppresses_raw):
        shape_errors.append("'suppresses' tuple contains non-string entries")
    else:
        # Closed-set membership check. The runtime decorator's signature
        # already constrains ``suppresses`` to ``tuple[Literal["R1", "R5"],
        # ...]`` via mypy, but the analyzer's parse path reads the AST
        # directly and must enforce the same set itself. Any rule ID outside
        # the authorised set is a malformed metadata signal — the decorator
        # is treated as inert (no suppression) and an R_TB_MALFORMED finding
        # is emitted naming the offending IDs.
        unauthorised = sorted(str(item) for item in suppresses_raw if str(item) not in _ALLOWED_BOUNDARY_RULES)
        if unauthorised:
            shape_errors.append(
                "'suppresses' contains unauthorised rule id(s) "
                f"{tuple(unauthorised)!r}; @trust_boundary may only suppress "
                f"{tuple(sorted(_ALLOWED_BOUNDARY_RULES))!r}"
            )

    if source_param_raw is None:
        shape_errors.append("missing kwarg 'source_param'")
    elif not isinstance(source_param_raw, str):
        shape_errors.append(f"'source_param' must be a string, got {type(source_param_raw).__name__}")
    elif not source_param_raw:
        shape_errors.append("'source_param' is an empty string")

    if shape_errors:
        diagnostics.append(
            BoundaryFinding(
                rule_id="R_TB_MALFORMED",
                message="@trust_boundary metadata is malformed: " + "; ".join(shape_errors),
                node=call,
            )
        )
        return None, diagnostics

    # mypy: the isinstance/shape checks above narrow these.
    assert isinstance(suppresses_raw, tuple)
    assert isinstance(source_param_raw, str)
    suppresses_strs: tuple[str, ...] = tuple(str(item) for item in suppresses_raw)
    source = parsed.get("source")
    invariant = parsed.get("invariant")
    test_ref = parsed.get("test_ref")
    test_fingerprint = parsed.get("test_fingerprint")
    source_value = source if isinstance(source, str) else None
    invariant_value = invariant if isinstance(invariant, str) else None
    test_ref_value = test_ref if isinstance(test_ref, str) else None
    test_fingerprint_value = test_fingerprint if isinstance(test_fingerprint, str) else None
    metadata = BoundaryMetadata(
        suppresses=frozenset(suppresses_strs),
        source_param=source_param_raw,
        decorator_node=call,
        source=source_value,
        invariant=invariant_value,
        test_ref=test_ref_value,
        test_fingerprint=test_fingerprint_value,
    )
    return metadata, diagnostics


# =============================================================================
# Dataflow walk: which names are "derived from source_param"?
# =============================================================================


def _assignment_targets(target: ast.expr) -> list[str]:
    """Collect bound name(s) from an assignment LHS.

    Handles plain names and tuple/list unpacking, recursively. Subscript and
    attribute targets do NOT bind a new name in the local scope (they mutate
    an existing object), so they are intentionally ignored.

    A starred unpacking target (``*rest``) binds ``rest``; we follow the
    ``ast.Starred.value`` to the inner name.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_assignment_targets(elt))
        return names
    if isinstance(target, ast.Starred):
        return _assignment_targets(target.value)
    return []


def subject_is_rooted(node: ast.AST, derived_names: frozenset[str]) -> bool:
    """Return True if ``node``'s value bottoms out at a name in ``derived_names``.

    Recursively descends through:

    * ``ast.Subscript.value`` — ``arguments["x"]`` is rooted at ``arguments``;
    * ``ast.Attribute.value`` — ``raw.id`` is rooted at ``raw``;
    * ``ast.Call.func`` and (for method calls) the receiver of the call —
      ``arguments.get("x")`` and ``arguments.get("x", default)`` are rooted
      at ``arguments``; ``foo(arguments)`` is rooted at the *function*, not
      at ``arguments``, so plain positional-arg passing does NOT root the
      call at ``arguments``. The ``Call.args[0]`` descent the original task
      description mentions only applies to method-style calls like
      ``arguments.get(...)``, where ``Call.func`` already chains through
      the receiver.
    * ``ast.Starred.value`` — ``*arguments`` is rooted at ``arguments``;
    * ``ast.IfExp`` / ``ast.BoolOp`` — bottoming-out on any operand that is
      rooted counts (an expression like ``arguments.get("x") or default``
      is reported by R1 against the ``Call`` node, not the ``BoolOp``, so
      this branch is mostly defensive).

    Returns False for anything that doesn't terminate at a Name. Function
    calls whose receiver isn't a Name (e.g. ``self.x.get(...)``) are NOT
    rooted at any source_param.
    """
    if isinstance(node, ast.Name):
        return node.id in derived_names
    if isinstance(node, ast.Subscript):
        return subject_is_rooted(node.value, derived_names)
    if isinstance(node, ast.Attribute):
        return subject_is_rooted(node.value, derived_names)
    if isinstance(node, ast.Call):
        # For ``x.method(...)`` the receiver chain is encoded as
        # ``Call.func = Attribute(value=x, attr="method")``. Descending
        # through ``func`` reaches the receiver.
        return subject_is_rooted(node.func, derived_names)
    if isinstance(node, ast.Starred):
        return subject_is_rooted(node.value, derived_names)
    if isinstance(node, ast.IfExp):
        return subject_is_rooted(node.body, derived_names) or subject_is_rooted(node.orelse, derived_names)
    if isinstance(node, ast.BoolOp):
        return any(subject_is_rooted(value, derived_names) for value in node.values)
    if isinstance(node, ast.NamedExpr):
        # The value of a walrus expression is the assigned value — root through it.
        return subject_is_rooted(node.value, derived_names)
    return False


def _expr_contains_derived_reference(node: ast.AST, derived_names: frozenset[str]) -> bool:
    """Return True if ``node`` mentions any name in ``derived_names``.

    Unlike :func:`subject_is_rooted`, this scans the entire subtree — useful
    for "does this RHS expression depend on the boundary parameter at all?"
    Used during the fixed-point derivation walk to propagate derived-ness
    through arbitrary RHS shapes (``raw = a + arguments["x"]`` should mark
    ``raw`` as derived, because removing the ``arguments`` reference would
    change the value).
    """
    return any(isinstance(sub, ast.Name) and sub.id in derived_names for sub in ast.walk(node))


def _comprehension_target_names(generator: ast.comprehension) -> list[str]:
    return _assignment_targets(generator.target)


def compute_derived_names(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_param: str,
) -> frozenset[str]:
    """Compute the set of local names derived from ``source_param``.

    Two-pass / fixed-point: a name assigned at line 30 from another derived
    name should make a reference at line 15 "rooted" if the analyzer asks
    later, so we don't rely on lexical assignment order. Iterate until the
    set stops growing.

    Propagation rules:

    * ``Assign`` / ``AugAssign`` / ``AnnAssign``: if the RHS expression
      mentions any derived name (anywhere in the subtree), every LHS Name
      target becomes derived. (We use *contains-reference*, not
      *rooted-at*, because ``raw = something(arguments["x"])`` should still
      taint ``raw``.)
    * ``For`` / ``AsyncFor``: if the iter is rooted at a derived name, the
      loop target name(s) become derived.
    * ``With`` / ``AsyncWith``: if the context expression is rooted, the
      ``as`` target becomes derived.
    * ``NamedExpr`` (walrus): if the value is rooted/contains-derived, the
      target name becomes derived.
    * Comprehensions (list/set/dict/gen): if the generator's iter is rooted
      at a derived name, the comprehension target becomes derived within
      that comprehension scope. (We add it to the global set; comprehensions
      in Python 3 have their own scope but we conservatively treat the
      target name as derived if it's later referenced.)

    The iteration is monotone over a finite local-name set. Instead of an
    arbitrary constant cap, the loop is bounded by the number of local names
    that can be marked derived plus one final no-change pass.
    """
    derived: set[str] = {source_param}
    local_names = _local_bindable_names(func_node) | {source_param}
    max_iterations = len(local_names) + 1

    for _iteration in range(max_iterations):
        before = frozenset(derived)
        snapshot = frozenset(derived)

        # Walk only the outer function's own lexical scope. The previous
        # implementation used ``ast.walk(func_node)`` with a ``sub is not
        # func_node`` guard, but ``ast.walk`` had already yielded every
        # descendant of any nested function *before* the guard could fire —
        # only the inner FunctionDef AST node itself was skipped, not its
        # body. That leaked taint from inner-scope ``raw = arguments["x"]``
        # assignments into the outer function's ``derived`` set, falsely
        # suppressing R1/R5 findings on unrelated outer-scope ``raw``
        # variables. ``walk_function_own_scope`` short-circuits at the
        # nested-scope AST boundary so inner bodies are not visited at all.
        for sub in walk_function_own_scope(func_node):
            if isinstance(sub, ast.Assign):
                value = sub.value
                if _expr_contains_derived_reference(value, snapshot):
                    for target in sub.targets:
                        for name in _assignment_targets(target):
                            derived.add(name)
            elif isinstance(sub, ast.AnnAssign):
                if sub.value is not None and _expr_contains_derived_reference(sub.value, snapshot):
                    for name in _assignment_targets(sub.target):
                        derived.add(name)
            elif isinstance(sub, ast.AugAssign):
                # ``x += derived`` does not REBIND x to derived in a
                # boundary-parameter sense (x already existed). But for
                # taint-style propagation, x's new value depends on derived,
                # so be conservative and mark it.
                if _expr_contains_derived_reference(sub.value, snapshot):
                    for name in _assignment_targets(sub.target):
                        derived.add(name)
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                if subject_is_rooted(sub.iter, snapshot):
                    for name in _assignment_targets(sub.target):
                        derived.add(name)
            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                for item in sub.items:
                    if item.optional_vars is None:
                        continue
                    if subject_is_rooted(item.context_expr, snapshot):
                        for name in _assignment_targets(item.optional_vars):
                            derived.add(name)
            elif isinstance(sub, ast.NamedExpr):
                if _expr_contains_derived_reference(sub.value, snapshot):
                    for name in _assignment_targets(sub.target):
                        derived.add(name)
            elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for generator in sub.generators:
                    if subject_is_rooted(generator.iter, snapshot):
                        for name in _comprehension_target_names(generator):
                            derived.add(name)

        if frozenset(derived) == before:
            return frozenset(derived)

    # Function bodies are bounded and derivation is monotone over a finite
    # name space, so this loop must converge. If we hit the iteration cap,
    # something has gone wrong — raise so the bug doesn't silently underrun
    # the suppression analysis (per the project's offensive-programming
    # doctrine, an unconverged fixed-point is a bug, not a recoverable case).
    raise RuntimeError(
        f"compute_derived_names did not converge within {max_iterations} iterations for "
        f"function {func_node.name!r}; this is a bug in the dataflow walk."
    )


def _local_bindable_names(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return names this function's own scope can bind through supported flows."""
    names: set[str] = set()
    for sub in walk_function_own_scope(func_node):
        if isinstance(sub, ast.Assign):
            for target in sub.targets:
                names.update(_assignment_targets(target))
        elif isinstance(sub, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            names.update(_assignment_targets(sub.target))
        elif isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                if item.optional_vars is not None:
                    names.update(_assignment_targets(item.optional_vars))
        elif isinstance(sub, ast.NamedExpr):
            names.update(_assignment_targets(sub.target))
        elif isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            for generator in sub.generators:
                names.update(_comprehension_target_names(generator))
    return names
