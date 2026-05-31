"""Trust-boundary tests rule implementation.

Honesty gate: every ``@trust_boundary`` must carry a ``test_ref`` that points
to a real pytest node whose own body asserts on raising behaviour, calls the
decorated symbol through the declared ``source_param``, and is bound by a
recorded AST fingerprint. If the decorator's ``invariant`` text names a
Python exception class, the referenced test's raising assertion must name the
same exception type. This is still not a complete semantic proof that the
operator prose covers every malformed-input class, but it mechanically rejects
the common attestation lies where the test raises something unrelated or no
longer exercises the decorated symbol at all. Sub-findings:

* ``TBE1`` (MISSING): ``test_ref`` is absent or ``None``.
* ``TBE2`` (NOTFOUND): the nodeid itself is malformed and cannot name a
  pytest function.
* ``R_TB_TESTS_FILE_MISSING``: the nodeid path is absent or resolves outside
  the repository root.
* ``R_TB_TESTS_PARSE_ERROR``: the referenced test file exists but cannot be
  parsed or read.
* ``R_TB_TESTS_FUNCTION_MISSING``: the file exists but the named function or
  method is missing.
* ``R_TB_TESTS_FILE_TOO_LARGE``: the referenced file exceeds the analyzer's
  bounded read limit.
* ``TBE3`` (WEAK): the named test exists but its body contains no
  ``pytest.raises(...)``, ``with pytest.raises(...)``, or
  ``unittest.TestCase.assertRaises(...)`` pattern.
* ``TBE4`` (NONLITERAL): a kwarg on the decorator is not a static literal
  (a name reference, a call, **-unpacking, or a positional arg). Emitted
  by self-enforcement rather than deferring to ``trust_tier.tier_model``
  — see ``trust_boundary.shared.KeywordExtraction`` and ticket
  elspeth-1f4634235a (C6-4).
* ``R_TB_TESTS_INVARIANT_MISMATCH``: the invariant text names an exception
  class, but the raising assertion names a different exception class.
* ``R_TB_TESTS_IRRELEVANT_INPUT``: the raising assertion does not call the
  decorated symbol with the declared ``source_param`` as a positional or
  keyword argument.
* ``R_TB_TESTS_FINGERPRINT_MISSING`` / ``R_TB_TESTS_FINGERPRINT_MISMATCH``:
  the referenced test body is not bound to the decorator by its canonical AST
  fingerprint, or the recorded fingerprint has drifted.

Path resolution: ``test_ref`` is a pytest nodeid of the form
``tests/path/to/file.py::test_func`` or ``tests/path/to/file.py::TestCls::test_method``.
Paths are interpreted relative to the repository root (NOT the scan root,
which is typically ``src/elspeth``). The repository root is derived via
:func:`elspeth_lints.rules.trust_boundary.shared.repository_root`, unless
``RuleContext.repo_root`` is supplied by the CLI/operator; explicit context
wins over scan-root derivation.

This rule does NOT walk imports, fixtures, or helpers. A test that delegates
its raising-assertion or subject call to a helper function would be reported
as WEAK / IRRELEVANT; that is a deliberate strictness choice — the honesty
signal must be visible in the test the decorator names, not buried one
indirection away.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from elspeth_lints.core.ast_walker import (
    PythonFileReadError,
    PythonSyntaxError,
    iter_own_scope,
    parse_python_file,
    walk_python_files,
)
from elspeth_lints.core.protocols import Finding, RuleContext, RuleMetadata, RuleScope
from elspeth_lints.rules.trust_boundary.shared import (
    display_path,
    extract_keywords,
    filter_allowlisted_findings,
    iter_trust_boundary_decorators,
    load_honesty_gate_allowlist,
    make_decorator_finding,
    repository_root,
)
from elspeth_lints.rules.trust_boundary.tests.metadata import (
    RULE_FILE_MISSING,
    RULE_FINGERPRINT_MISMATCH,
    RULE_FINGERPRINT_MISSING,
    RULE_FUNCTION_MISSING,
    RULE_ID,
    RULE_INPUT_IRRELEVANT,
    RULE_INVARIANT_MISMATCH,
    RULE_METADATA,
    RULE_MISSING,
    RULE_NONLITERAL,
    RULE_NOTFOUND,
    RULE_PARSE_ERROR,
    RULE_TOO_LARGE,
    RULE_WEAK,
    SUGGESTION_FILE_MISSING,
    SUGGESTION_FINGERPRINT_MISMATCH,
    SUGGESTION_FINGERPRINT_MISSING,
    SUGGESTION_FUNCTION_MISSING,
    SUGGESTION_INPUT_IRRELEVANT,
    SUGGESTION_INVARIANT_MISMATCH,
    SUGGESTION_MISSING,
    SUGGESTION_NONLITERAL,
    SUGGESTION_NOTFOUND,
    SUGGESTION_PARSE_ERROR,
    SUGGESTION_TOO_LARGE,
    SUGGESTION_WEAK,
)

_MAX_TEST_REF_BYTES = 5 * 1024 * 1024
_EXCEPTION_NAME_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b")
_ALLOWLIST_RULE_IDS = frozenset(
    {
        RULE_MISSING,
        RULE_NOTFOUND,
        RULE_WEAK,
        RULE_NONLITERAL,
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


@dataclass(frozen=True, slots=True)
class TrustBoundaryTestsRule:
    """Detect ``@trust_boundary`` decorators with missing, broken, or weak ``test_ref``."""

    id: str = RULE_ID
    scope: RuleScope = RuleScope.WHOLE_REPO
    metadata: RuleMetadata = RULE_METADATA

    def analyze(self, tree: ast.AST, file_path: Path, context: RuleContext) -> list[Finding]:
        """Analyze one tree directly (for focused tests) or walk the scan root."""
        if isinstance(tree, ast.Module) and tree.body and file_path.suffix == ".py":
            return analyze_tree(
                tree,
                display_path(file_path, context.root),
                repo_root=repository_root(context.root, context.repo_root),
            )
        return scan_root(
            context.root,
            repo_root=context.repo_root,
            allowlist_dir_override=context.allowlist_dir_override,
        )


def analyze_tree(tree: ast.AST, file_path: str, *, repo_root: Path) -> list[Finding]:
    """Return ``trust_boundary.tests`` findings for one parsed syntax tree."""
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
        test_ref = kwargs.get("test_ref")
        source_param = kwargs.get("source_param")
        invariant = kwargs.get("invariant")
        test_fingerprint = kwargs.get("test_fingerprint")
        if test_ref is None:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_MISSING,
                    file_path=file_path,
                    call=call,
                    message=("@trust_boundary has no test_ref; a direct raising-shape test is mandatory for trust-boundary suppressions."),
                    suggestion=SUGGESTION_MISSING,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        if not isinstance(source_param, str) or not source_param:
            # Malformed source_param is enforced by trust_boundary.scope and
            # trust_tier.tier_model. Avoid double-reporting from this rule.
            continue
        if not isinstance(test_ref, str) or not test_ref:
            # Malformed (non-string or empty) — surface as MISSING; the
            # tier_model rule will additionally surface R_TB_MALFORMED.
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_MISSING,
                    file_path=file_path,
                    call=call,
                    message=(f"@trust_boundary test_ref must be a non-empty string, got {test_ref!r}."),
                    suggestion=SUGGESTION_MISSING,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        resolution = _resolve_test_ref(test_ref, repo_root)
        if isinstance(resolution, _TestRefResolutionError):
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=resolution.rule_id,
                    file_path=file_path,
                    call=call,
                    message=resolution.message,
                    suggestion=resolution.suggestion,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        if not isinstance(test_fingerprint, str) or not test_fingerprint:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_FINGERPRINT_MISSING,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} must include "
                        f"test_fingerprint={resolution.fingerprint!r} to bind the resolved test body."
                    ),
                    suggestion=SUGGESTION_FINGERPRINT_MISSING,
                    symbol_context=(func_node.name,),
                )
            )
        elif test_fingerprint != resolution.fingerprint:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_FINGERPRINT_MISMATCH,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} resolves to "
                        f"fingerprint {resolution.fingerprint!r}, but the decorator records "
                        f"{test_fingerprint!r}."
                    ),
                    suggestion=SUGGESTION_FINGERPRINT_MISMATCH,
                    symbol_context=(func_node.name,),
                )
            )

        semantics = _test_ref_semantics(
            resolution.test_function,
            subject_name=func_node.name,
            source_param=source_param,
            source_param_index=_source_param_index(func_node, source_param),
            module_class_names=resolution.module_class_names,
        )
        if not semantics.has_raising_assertion:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_WEAK,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} resolves to "
                        f"{resolution.test_function.name!r} but the test body has "
                        "no pytest.raises / with pytest.raises / assertRaises pattern."
                    ),
                    suggestion=SUGGESTION_WEAK,
                    symbol_context=(func_node.name,),
                )
            )
            continue
        invariant_exception_names = _exception_names_in_text(invariant) if isinstance(invariant, str) else frozenset()
        if (
            invariant_exception_names
            and semantics.raised_exception_names
            and not invariant_exception_names.intersection(semantics.raised_exception_names)
        ):
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_INVARIANT_MISMATCH,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary invariant names exception type(s) "
                        f"{tuple(sorted(invariant_exception_names))!r}, but test_ref {test_ref!r} "
                        f"raises {tuple(sorted(semantics.raised_exception_names))!r}."
                    ),
                    suggestion=SUGGESTION_INVARIANT_MISMATCH,
                    symbol_context=(func_node.name,),
                )
            )
        if invariant_exception_names and not semantics.raised_exception_names:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_INVARIANT_MISMATCH,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary invariant names exception type(s) "
                        f"{tuple(sorted(invariant_exception_names))!r}, but test_ref {test_ref!r} "
                        "does not name the exception type in its raising assertion."
                    ),
                    suggestion=SUGGESTION_INVARIANT_MISMATCH,
                    symbol_context=(func_node.name,),
                )
            )
        if not semantics.subject_call_uses_source_param:
            findings.append(
                make_decorator_finding(
                    metadata=RULE_METADATA,
                    rule_id=RULE_INPUT_IRRELEVANT,
                    file_path=file_path,
                    call=call,
                    message=(
                        f"@trust_boundary test_ref {test_ref!r} contains a raising assertion, "
                        f"but that assertion does not directly call {func_node.name!r} with "
                        f"source_param={source_param!r}."
                    ),
                    suggestion=SUGGESTION_INPUT_IRRELEVANT,
                    symbol_context=(func_node.name,),
                )
            )
    return findings


def scan_root(
    root: Path,
    *,
    repo_root: Path | None = None,
    allowlist_dir_override: Path | None = None,
) -> list[Finding]:
    """Walk every Python file under ``root`` and aggregate findings."""
    resolved_repo_root = repository_root(root, repo_root)
    allowlist = load_honesty_gate_allowlist(
        root,
        allowlist_dir_override=allowlist_dir_override,
        valid_rule_ids=_ALLOWLIST_RULE_IDS,
    )
    findings: list[Finding] = []
    for item in walk_python_files(root):
        # Skip non-analysable per-file results. Syntax errors and I/O
        # failures are surfaced by the CLI driver (which converts them to
        # ``parse-error`` / ``read-error`` findings); the per-rule scan
        # loop has nothing rule-specific to add for an unparseable or
        # unreadable file, so it skips them rather than attributing a
        # rule-id-specific finding.
        if isinstance(item, (PythonSyntaxError, PythonFileReadError)):
            continue
        findings.extend(analyze_tree(item.tree, display_path(item.path, root), repo_root=resolved_repo_root))
    return filter_allowlisted_findings(findings, allowlist)


@dataclass(frozen=True, slots=True)
class _ResolvedTestRef:
    """The pytest nodeid resolved to an AST FunctionDef and its host file."""

    test_function: ast.FunctionDef | ast.AsyncFunctionDef
    file_path: Path
    fingerprint: str
    module_class_names: frozenset[str]


@dataclass(frozen=True, slots=True)
class _TestRefResolutionError:
    """A precise reason a pytest nodeid cannot be used as an honesty gate."""

    rule_id: str
    message: str
    suggestion: str


def _resolve_test_ref(test_ref: str, repo_root: Path) -> _ResolvedTestRef | _TestRefResolutionError:
    """Resolve a pytest nodeid against the repository root.

    Supports both shapes:

    * ``tests/.../test_file.py::test_function``
    * ``tests/.../test_file.py::TestClass::test_method``

    Returns a specific resolution error if the nodeid is malformed, points
    outside the repository, points to an absent/oversized/unparseable file,
    or names a function/method that the file does not define.
    """
    parts = test_ref.split("::")
    if len(parts) < 2:
        return _TestRefResolutionError(
            rule_id=RULE_NOTFOUND,
            message=(f"@trust_boundary test_ref {test_ref!r} is not a pytest nodeid with a file and function segment separated by '::'."),
            suggestion=SUGGESTION_NOTFOUND,
        )
    relative_file = parts[0]
    name_segments = [*parts[1:-1], _strip_pytest_parametrize_suffix(parts[-1])]
    resolved_root = repo_root.resolve()
    file_path = (resolved_root / relative_file).resolve()
    if not file_path.is_relative_to(resolved_root):
        return _TestRefResolutionError(
            rule_id=RULE_FILE_MISSING,
            message=(
                f"@trust_boundary test_ref {test_ref!r} points outside the repository root "
                f"{resolved_root}; file is missing from the analysable repository."
            ),
            suggestion=SUGGESTION_FILE_MISSING,
        )
    if not file_path.is_file():
        return _TestRefResolutionError(
            rule_id=RULE_FILE_MISSING,
            message=(f"@trust_boundary test_ref {test_ref!r} file is missing under repository root {resolved_root}."),
            suggestion=SUGGESTION_FILE_MISSING,
        )
    try:
        if file_path.stat().st_size > _MAX_TEST_REF_BYTES:
            return _TestRefResolutionError(
                rule_id=RULE_TOO_LARGE,
                message=(
                    f"@trust_boundary test_ref {test_ref!r} points to {file_path}, which exceeds the {_MAX_TEST_REF_BYTES} byte read limit."
                ),
                suggestion=SUGGESTION_TOO_LARGE,
            )
    except OSError as exc:
        return _TestRefResolutionError(
            rule_id=RULE_PARSE_ERROR,
            message=(f"@trust_boundary test_ref {test_ref!r} file cannot be statted/read by the analyzer: {exc}."),
            suggestion=SUGGESTION_PARSE_ERROR,
        )
    parsed = parse_python_file(file_path)
    if isinstance(parsed, (PythonSyntaxError, PythonFileReadError)):
        return _TestRefResolutionError(
            rule_id=RULE_PARSE_ERROR,
            message=(f"@trust_boundary test_ref {test_ref!r} points to a test file that cannot be parsed or read."),
            suggestion=SUGGESTION_PARSE_ERROR,
        )
    func = _lookup_named_function(parsed.tree, name_segments)
    if func is None:
        return _TestRefResolutionError(
            rule_id=RULE_FUNCTION_MISSING,
            message=(f"@trust_boundary test_ref {test_ref!r} file exists, but the named function/method does not resolve."),
            suggestion=SUGGESTION_FUNCTION_MISSING,
        )
    return _ResolvedTestRef(
        test_function=func,
        file_path=file_path,
        fingerprint=_fingerprint_test_function(func),
        module_class_names=_module_class_names(parsed.tree),
    )


def _fingerprint_test_function(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return a stable fingerprint for the referenced test function AST."""
    canonical = ast.dump(func, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _module_class_names(tree: ast.Module) -> frozenset[str]:
    """Return class-like names visible in the referenced test module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name.rsplit(".", 1)[-1]
                if _is_type_name(bound_name):
                    names.add(bound_name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                if _is_type_name(bound_name):
                    names.add(bound_name)
    return frozenset(names)


def _strip_pytest_parametrize_suffix(segment: str) -> str:
    """Return the function name portion of a parametrized pytest nodeid segment."""
    if "[" not in segment:
        return segment
    return segment.split("[", 1)[0]


def _lookup_named_function(tree: ast.Module, name_segments: list[str]) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Walk ``name_segments`` against a parsed module to find the target function.

    A single segment is a module-level function. Multiple segments are
    interpreted as a class chain ending in a method
    (``TestClass::test_method`` or ``TestOuter::TestInner::test_method``).
    Returns ``None`` if any segment doesn't resolve.
    """
    namespace: list[ast.stmt] = list(tree.body)
    for index, segment in enumerate(name_segments):
        is_last = index == len(name_segments) - 1
        next_namespace: list[ast.stmt] | None = None
        for stmt in namespace:
            if is_last and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name == segment:
                return stmt
            if not is_last and isinstance(stmt, ast.ClassDef) and stmt.name == segment:
                next_namespace = list(stmt.body)
                break
        if next_namespace is None:
            return None
        namespace = next_namespace
    return None


@dataclass(frozen=True, slots=True)
class _TestRefSemantics:
    """Mechanically checkable claims extracted from the referenced test body."""

    has_raising_assertion: bool
    raised_exception_names: frozenset[str]
    subject_call_uses_source_param: bool


def _test_ref_semantics(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    subject_name: str,
    source_param: str,
    source_param_index: int | None,
    module_class_names: frozenset[str],
) -> _TestRefSemantics:
    """Return the direct raising, exception, and subject-call signals in ``func``.

    Detects three shapes:

    * ``pytest.raises(...)`` call (bare or inside any expression);
    * ``with pytest.raises(...):`` context manager (also ``async with``);
    * ``self.assertRaises(...)`` / ``cls.assertRaises(...)`` /
      bare ``assertRaises(...)`` calls (covers unittest-style tests).

    The walk is scope-respecting and body-only: decorators, return
    annotations, default values, and the bodies of nested helper
    functions, lambdas, or classes are out of scope. The honesty
    contract for TBE3 is that the raising assertion must be visible in
    the test function the decorator names — buried one indirection away
    in a helper does NOT count. This mirrors the documented behaviour
    in this module's header docstring ("does NOT walk imports,
    fixtures, or helpers; a test that delegates its raising-assertion
    to a helper function would be reported as WEAK").

    Before the C6-3 fix, ``_walk_statements`` used :func:`ast.walk`, which
    descends into nested ``FunctionDef`` bodies. A decorated test whose
    only ``pytest.raises(...)`` lived in a nested helper falsely passed
    TBE3 — the rule reported "the body contains a raising assertion"
    when, by the contract, the body did not. The fix uses
    :func:`iter_own_scope` (same scope-respecting walker the B1 /
    C5-1 fixes use in the tier_model and scope rules), which
    short-circuits at nested-scope boundaries so reads inside an inner
    helper / lambda / class body don't satisfy the outer's contract.
    The later M6-11 hardening widened the contract from "some direct raising
    shape exists" to "the direct raising shape calls the decorated subject
    through the declared ``source_param`` and names the claimed exception
    type". We still stay deliberately local: imports, fixtures, and helper
    functions are not followed.
    """
    has_raising = False
    raised_exception_names: set[str] = set()
    subject_call_uses_source_param = False
    for statement in func.body:
        for child in iter_own_scope(statement):
            if isinstance(child, ast.Call) and _is_raising_call(child.func):
                has_raising = True
                raised_exception_names.update(_exception_names_from_raises_call(child))
                if _raising_call_invokes_subject(
                    child,
                    subject_name=subject_name,
                    source_param=source_param,
                    source_param_index=source_param_index,
                    module_class_names=module_class_names,
                ):
                    subject_call_uses_source_param = True
            if isinstance(child, (ast.With, ast.AsyncWith)):
                for item in child.items:
                    if isinstance(item.context_expr, ast.Call) and _is_raising_call(item.context_expr.func):
                        has_raising = True
                        raised_exception_names.update(_exception_names_from_raises_call(item.context_expr))
                        if _body_calls_subject_with_source_param(
                            child.body,
                            subject_name=subject_name,
                            source_param=source_param,
                            source_param_index=source_param_index,
                            module_class_names=module_class_names,
                        ):
                            subject_call_uses_source_param = True
    return _TestRefSemantics(
        has_raising_assertion=has_raising,
        raised_exception_names=frozenset(raised_exception_names),
        subject_call_uses_source_param=subject_call_uses_source_param,
    )


def _has_raising_assertion(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function body contains a raising-assertion pattern."""
    return _test_ref_semantics(
        func,
        subject_name="",
        source_param="",
        source_param_index=None,
        module_class_names=frozenset(),
    ).has_raising_assertion


def _is_raising_call(func_expr: ast.expr) -> bool:
    """Return True if ``func_expr`` references a raising-assertion helper."""
    if isinstance(func_expr, ast.Attribute):
        if func_expr.attr == "raises":
            return isinstance(func_expr.value, ast.Name) and func_expr.value.id == "pytest"
        if func_expr.attr in {"assertRaises", "assertRaisesRegex"}:
            return True
    return isinstance(func_expr, ast.Name) and func_expr.id in {"assertRaises", "assertRaisesRegex"}


def _exception_names_in_text(text: str) -> frozenset[str]:
    """Extract exception class names declared in invariant prose."""
    return frozenset(_EXCEPTION_NAME_RE.findall(text))


def _exception_names_from_raises_call(call: ast.Call) -> frozenset[str]:
    """Extract exception class names from the first argument to a raises call."""
    if not call.args:
        return frozenset()
    return frozenset(_exception_names_from_expr(call.args[0]))


def _exception_names_from_expr(expr: ast.expr) -> set[str]:
    """Return simple exception class names from an AST expression."""
    if isinstance(expr, ast.Name):
        return {expr.id}
    if isinstance(expr, ast.Attribute):
        return {expr.attr}
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        names: set[str] = set()
        for item in expr.elts:
            names.update(_exception_names_from_expr(item))
        return names
    return set()


def _source_param_index(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_param: str,
) -> int | None:
    """Return the positional parameter index for ``source_param`` if it has one."""
    positional = [*func_node.args.posonlyargs, *func_node.args.args]
    for index, arg in enumerate(positional):
        if arg.arg == source_param:
            return index
    return None


def _raising_call_invokes_subject(
    call: ast.Call,
    *,
    subject_name: str,
    source_param: str,
    source_param_index: int | None,
    module_class_names: frozenset[str],
) -> bool:
    """Return True for ``pytest.raises(ValueError, subject, ...)`` style calls."""
    if len(call.args) < 2:
        return False
    if not _is_subject_reference(call.args[1], subject_name):
        return False
    pseudo_call = ast.Call(
        func=call.args[1],
        args=list(call.args[2:]),
        keywords=list(call.keywords),
    )
    return _call_uses_source_param(
        pseudo_call,
        source_param=source_param,
        source_param_index=source_param_index,
        module_class_names=module_class_names,
    )


def _body_calls_subject_with_source_param(
    body: list[ast.stmt],
    *,
    subject_name: str,
    source_param: str,
    source_param_index: int | None,
    module_class_names: frozenset[str],
) -> bool:
    """Return True if the with-body directly calls the subject through source_param."""
    for statement in body:
        for child in iter_own_scope(statement):
            if (
                isinstance(child, ast.Call)
                and _is_subject_call(child, subject_name)
                and _call_uses_source_param(
                    child,
                    source_param=source_param,
                    source_param_index=source_param_index,
                    module_class_names=module_class_names,
                )
            ):
                return True
    return False


def _is_subject_call(call: ast.Call, subject_name: str) -> bool:
    """Return True if ``call`` calls the decorated function by name or attribute."""
    return _is_subject_reference(call.func, subject_name)


def _is_subject_reference(expr: ast.expr, subject_name: str) -> bool:
    """Return True if ``expr`` references the decorated function."""
    if not subject_name:
        return False
    if isinstance(expr, ast.Name):
        return expr.id == subject_name
    return isinstance(expr, ast.Attribute) and expr.attr == subject_name


def _call_uses_source_param(
    call: ast.Call,
    *,
    source_param: str,
    source_param_index: int | None,
    module_class_names: frozenset[str],
) -> bool:
    """Return True if a call supplies the source parameter by name or position."""
    if any(keyword.arg == source_param for keyword in call.keywords):
        return True
    positional_index = source_param_index
    if (
        isinstance(call.func, ast.Attribute)
        and positional_index is not None
        and positional_index > 0
        and not _is_unbound_method_call(call.func, module_class_names=module_class_names)
    ):
        # Bound method call: ``service.foo(payload)`` omits ``self`` / ``cls``.
        positional_index -= 1
    return positional_index is not None and len(call.args) > positional_index


def _is_unbound_method_call(func: ast.Attribute, *, module_class_names: frozenset[str]) -> bool:
    receiver_name = _terminal_attribute_name(func.value)
    return receiver_name is not None and (receiver_name in module_class_names or _is_type_name(receiver_name))


def _terminal_attribute_name(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _is_type_name(name: str) -> bool:
    return bool(name) and name[0].isupper()


RULE = TrustBoundaryTestsRule()
