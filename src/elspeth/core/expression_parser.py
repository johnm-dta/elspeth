"""Safe expression parser for gate conditions.

Uses Python's ast module to parse and evaluate expressions in a restricted
subset of Python. This is NOT eval() - it's a secure whitelist-based parser.

The parser operates in two phases:
1. Parse-time validation: Reject forbidden constructs at construction
2. Evaluation: Safely execute the validated AST against a context namespace

Security model:
- Plugins are system code (trusted), but config expressions could come from
  config files that might be misconfigured or tampered with
- This parser is defense-in-depth - restricting what expressions can do
"""

from __future__ import annotations

import ast
import math
import operator
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import elspeth.contracts.errors as contract_errors

if TYPE_CHECKING:
    from elspeth.contracts import PipelineRow


class ExpressionSecurityError(Exception):
    """Raised when expression contains forbidden constructs."""


class ExpressionSyntaxError(Exception):
    """Raised when expression is not valid Python syntax."""


class ExpressionEvaluationError(Exception):
    """Raised when expression evaluation fails at runtime.

    This wraps operational errors (KeyError, ZeroDivisionError, TypeError)
    that occur when evaluating expressions against a context. Unlike
    ExpressionSecurityError (rejected at parse time) or ExpressionSyntaxError
    (invalid Python syntax), this occurs when the expression is valid but
    fails during evaluation.

    The original exception is chained via __cause__ for debugging.
    """


# Allowed comparison operators (immutable to prevent runtime tampering)
_COMPARISON_OPS: MappingProxyType[type[ast.cmpop], Any] = MappingProxyType(
    {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Is: operator.is_,
        ast.IsNot: operator.is_not,
        ast.In: lambda a, b: a in b,
        ast.NotIn: lambda a, b: a not in b,
    }
)

# Allowed binary operators (immutable to prevent runtime tampering)
_BINARY_OPS: MappingProxyType[type[ast.operator], Any] = MappingProxyType(
    {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
)

# Allowed unary operators (immutable to prevent runtime tampering)
_UNARY_OPS: MappingProxyType[type[ast.unaryop], Any] = MappingProxyType(
    {
        ast.Not: operator.not_,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
)

# Allowed boolean operators (immutable to prevent runtime tampering)
_BOOL_OPS: MappingProxyType[type[ast.boolop], str] = MappingProxyType(
    {
        ast.And: "and",
        ast.Or: "or",
    }
)

# Safe built-in functions allowed in expressions (immutable to prevent runtime tampering).
# Only non-coercive operations are permitted. Coercive builtins (str, int, float, bool)
# are forbidden because they silently normalize Tier 2 data in gate expressions,
# masking upstream contract bugs and weakening audit attributability.
_SAFE_BUILTINS: MappingProxyType[str, Any] = MappingProxyType(
    {
        "len": len,
        "abs": abs,
    }
)


_SAFE_CONSTANTS: frozenset[str] = frozenset({"True", "False", "None"})


class _ExpressionValidator(ast.NodeVisitor):
    """AST visitor that validates expressions for security.

    Raises ExpressionSecurityError if any forbidden construct is found.
    """

    def __init__(self, *, allowed_names: frozenset[str] = frozenset({"row"})) -> None:
        self.errors: list[str] = []
        self._in_call_func: bool = False  # Track if currently visiting a Call's func
        self._allowed_names = allowed_names
        self._allow_allowed_name_reference = 0
        self._allow_safe_builtin_reference = 0

    def _is_none_constant(self, node: ast.expr) -> bool:
        """Check if node is a None literal (ast.Constant or ast.Name)."""
        return (isinstance(node, ast.Constant) and node.value is None) or (isinstance(node, ast.Name) and node.id == "None")

    def _is_allowed_derived(self, node: ast.expr) -> bool:
        """Check if node is an allowed name or derived from allowed name access.

        Handles: row, row['x'], row['x']['y'], row.get('x')['y'],
        and any name in self._allowed_names with subscript chains.
        """
        if isinstance(node, ast.Name) and node.id in self._allowed_names:
            return True
        if isinstance(node, ast.Subscript):
            return self._is_allowed_derived(node.value)
        # row.get(...) calls return row-derived data
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._allowed_names
            and node.func.attr == "get"
        )

    def visit_Name(self, node: ast.Name) -> None:
        """Allow only allowed names, boolean/None literals, and safe builtin names."""
        if node.id in _SAFE_CONSTANTS:
            return
        if node.id in _SAFE_BUILTINS:
            if self._allow_safe_builtin_reference > 0:
                return
            self.errors.append(f"Bare builtin name: {node.id!r} (only direct calls are allowed)")
            return
        if node.id in self._allowed_names:
            if self._allow_allowed_name_reference > 0:
                return
            self.errors.append(f"Bare allowed name: {node.id!r} (use {node.id}['field'] or {node.id}.get('field'))")
            return
        self.errors.append(f"Forbidden name: {node.id!r}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Allow subscript access on allowed-name-derived data only."""
        # Reject slice syntax (defense-in-depth, also caught by visit_Slice)
        if isinstance(node.slice, ast.Slice):
            self.errors.append("Slice syntax (e.g., [1:3]) is forbidden")
        # Restrict subscript to allowed-name-derived data
        if not self._is_allowed_derived(node.value):
            self.errors.append(f"Subscript access is only allowed on allowed names; got subscript on {ast.dump(node.value)}")
        self._allow_allowed_name_reference += 1
        try:
            self.visit(node.value)
        finally:
            self._allow_allowed_name_reference -= 1
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice) -> None:
        """Reject slice syntax."""
        self.errors.append("Slice syntax (e.g., [1:3]) is forbidden")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Allow only .get method access on allowed names when called."""
        if isinstance(node.value, ast.Name) and node.value.id in self._allowed_names:
            self._allow_allowed_name_reference += 1
            try:
                self.visit(node.value)
            finally:
                self._allow_allowed_name_reference -= 1
            if node.attr != "get":
                self.errors.append(f"Forbidden attribute: {node.attr!r} (only 'get' is allowed)")
            elif not self._in_call_func:
                # name.get without a call is forbidden - returns method object
                self.errors.append(f"Bare '{node.value.id}.get' is forbidden; use '{node.value.id}.get(key)'")
            return
        self.errors.append(f"Forbidden attribute access: {node.attr!r}")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        """Allow .get() calls on allowed names and safe builtin calls."""
        # Allow name.get() with exactly 1 argument (for any allowed name)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self._allowed_names
            and node.func.attr == "get"
        ):
            caller_name = node.func.value.id
            if len(node.args) != 1:
                self.errors.append(
                    f"{caller_name}.get() requires exactly 1 argument (key only), got {len(node.args)}. "
                    f"Default values are forbidden — they fabricate data the source never provided. "
                    f"Use '{caller_name}.get(key) is not None' to test for field presence."
                )
            if node.keywords:
                self.errors.append(f"{caller_name}.get() does not accept keyword arguments")
            # Visit func with context flag set to allow row.get attribute
            self._in_call_func = True
            try:
                self.visit(node.func)
            finally:
                self._in_call_func = False
            # Visit arguments normally
            for arg in node.args:
                self.visit(arg)
            return

        # Allow safe builtin calls: len(), abs()
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS:
            if node.keywords:
                self.errors.append(f"{node.func.id}() does not accept keyword arguments")
            # Visit func name (validated by visit_Name)
            self._allow_safe_builtin_reference += 1
            try:
                self.visit(node.func)
            finally:
                self._allow_safe_builtin_reference -= 1
            # Visit arguments normally
            for arg in node.args:
                self.visit(arg)
            return

        self.errors.append(f"Forbidden function call: {ast.dump(node.func)}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Validate comparison operators."""
        # Build list of all operands for is/is not validation
        all_operands = [node.left, *node.comparators]

        for i, op in enumerate(node.ops):
            if type(op) not in _COMPARISON_OPS:
                self.errors.append(f"Forbidden comparison operator: {type(op).__name__}")
            # Restrict is/is not to None checks only
            elif isinstance(op, ast.Is | ast.IsNot):
                left_operand = all_operands[i]
                right_operand = all_operands[i + 1]
                if not (self._is_none_constant(left_operand) or self._is_none_constant(right_operand)):
                    self.errors.append("'is' and 'is not' operators are only allowed for None checks")
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        """Validate boolean operators (and, or)."""
        if type(node.op) not in _BOOL_OPS:
            self.errors.append(f"Forbidden boolean operator: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        """Validate binary operators."""
        if type(node.op) not in _BINARY_OPS:
            self.errors.append(f"Forbidden binary operator: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        """Validate unary operators."""
        if type(node.op) not in _UNARY_OPS:
            self.errors.append(f"Forbidden unary operator: {type(node.op).__name__}")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Allow literals: strings, numbers, booleans, None."""
        if node.value is None:
            return  # None is allowed
        if isinstance(node.value, float) and not math.isfinite(node.value):
            self.errors.append("Non-finite float literal is forbidden in expressions")
            return
        if isinstance(node.value, str | int | float | bool):
            return  # Primitives allowed
        self.errors.append(f"Forbidden constant type: {type(node.value).__name__}")

    def visit_List(self, node: ast.List) -> None:
        """Allow list literals for membership checks."""
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Allow dict literals for membership checks, but reject spread syntax."""
        # None keys indicate **spread syntax which we don't support
        for key in node.keys:
            if key is None:
                self.errors.append("Dict spread (**) is forbidden")
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        """Allow tuple literals for membership checks."""
        self.generic_visit(node)

    def visit_Set(self, node: ast.Set) -> None:
        """Allow set literals for membership checks."""
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        """Allow ternary expressions: x if condition else y."""
        self.generic_visit(node)

    # Explicitly forbidden constructs

    def visit_Lambda(self, node: ast.Lambda) -> None:
        """Lambda expressions are forbidden."""
        self.errors.append("Lambda expressions are forbidden")

    def visit_ListComp(self, node: ast.ListComp) -> None:
        """List comprehensions are forbidden."""
        self.errors.append("List comprehensions are forbidden")

    def visit_DictComp(self, node: ast.DictComp) -> None:
        """Dict comprehensions are forbidden."""
        self.errors.append("Dict comprehensions are forbidden")

    def visit_SetComp(self, node: ast.SetComp) -> None:
        """Set comprehensions are forbidden."""
        self.errors.append("Set comprehensions are forbidden")

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        """Generator expressions are forbidden."""
        self.errors.append("Generator expressions are forbidden")

    def visit_Await(self, node: ast.Await) -> None:
        """Await expressions are forbidden."""
        self.errors.append("Await expressions are forbidden")

    def visit_Yield(self, node: ast.Yield) -> None:
        """Yield expressions are forbidden."""
        self.errors.append("Yield expressions are forbidden")

    def visit_YieldFrom(self, node: ast.YieldFrom) -> None:
        """Yield from expressions are forbidden."""
        self.errors.append("Yield from expressions are forbidden")

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        """Assignment expressions (:=) are forbidden."""
        self.errors.append("Assignment expressions (:=) are forbidden")

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        """F-strings are forbidden."""
        self.errors.append("F-strings are forbidden")

    def visit_FormattedValue(self, node: ast.FormattedValue) -> None:
        """Formatted values (f-string expressions) are forbidden."""
        self.errors.append("F-string expressions are forbidden")

    def visit_Starred(self, node: ast.Starred) -> None:
        """Starred expressions (*x) are forbidden."""
        self.errors.append("Starred expressions (*) are forbidden")

    # Explicit allowset of handled expression node types.  Adding a new
    # visit_* method is NOT sufficient — the type must also appear here.
    # This prevents brute-force bypass where defining a handler silently
    # whitelists a new AST construct.
    _HANDLED_EXPR_TYPES: frozenset[type] = frozenset(
        {
            # Allowed constructs (handlers recurse via generic_visit)
            ast.Name,
            ast.Subscript,
            ast.Slice,
            ast.Attribute,
            ast.Call,
            ast.Compare,
            ast.BoolOp,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.List,
            ast.Dict,
            ast.Tuple,
            ast.Set,
            ast.IfExp,
            # Explicitly forbidden constructs (handlers append errors)
            ast.Lambda,
            ast.ListComp,
            ast.DictComp,
            ast.SetComp,
            ast.GeneratorExp,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.NamedExpr,
            ast.JoinedStr,
            ast.FormattedValue,
            ast.Starred,
        }
    )

    def visit(self, node: ast.AST) -> None:
        """Dispatch with fail-closed default for unhandled expression nodes.

        Rejects any expression node type not in _HANDLED_EXPR_TYPES,
        regardless of whether a visit_* method exists.  This is
        defense-in-depth: if a future Python version adds a new AST
        expression type, it will be rejected here rather than silently
        passing validation.

        Non-expression AST nodes (operator types like ast.Eq, context
        markers like ast.Load, the wrapper ast.Expression) are allowed
        through because they are structural metadata, not executable
        constructs.
        """
        if isinstance(node, ast.expr) and type(node) not in self._HANDLED_EXPR_TYPES:
            self.errors.append(f"Unsupported expression construct: {type(node).__name__}")
            return
        super().visit(node)


# ── Module-level coupling enforcement ─────────────────────────────────
# Every type in _HANDLED_EXPR_TYPES must have a visit_* method, and every
# visit_* method must have a corresponding type.  Without this check,
# adding a type without a handler silently falls through to generic_visit,
# bypassing security validation.
_handler_type_names = {t.__name__ for t in _ExpressionValidator._HANDLED_EXPR_TYPES}
_visitor_method_names = {
    name.removeprefix("visit_") for name in vars(_ExpressionValidator) if name.startswith("visit_") and name != "visit"
}

_missing_handlers = _handler_type_names - _visitor_method_names
_orphan_visitors = _visitor_method_names - _handler_type_names

if _missing_handlers or _orphan_visitors:
    _parts: list[str] = []
    if _missing_handlers:
        _parts.append(f"types without visit_* handler: {sorted(_missing_handlers)}")
    if _orphan_visitors:
        _parts.append(f"visit_* handlers without type entry: {sorted(_orphan_visitors)}")
    raise TypeError(f"_ExpressionValidator handler/type coupling violation: {'; '.join(_parts)}")

del _handler_type_names, _visitor_method_names, _missing_handlers, _orphan_visitors


class _ExpressionEvaluator(ast.NodeVisitor):
    """AST visitor that evaluates validated expressions."""

    def __init__(
        self,
        context: dict[str, Any] | PipelineRow,
        *,
        allowed_names: frozenset[str] = frozenset({"row"}),
        single_name_mode: bool = True,
    ) -> None:
        self._context = context
        self._allowed_names = allowed_names
        # In single-name mode the context IS the value (e.g. row={"x": 5}).
        # In multi-name mode the context is a namespace dict where each
        # allowed name maps to its value (e.g. {"collections": {...}, "env": {...}}).
        self._single_name_mode = single_name_mode

    def _ensure_finite_float(self, value: Any, *, context: str) -> Any:
        """Reject non-finite floats produced inside expression evaluation."""
        if isinstance(value, float) and not math.isfinite(value):
            raise ExpressionEvaluationError(f"{context} produced non-finite float: {value!r}")
        return value

    def visit_Expression(self, node: ast.Expression) -> Any:
        """Evaluate the top-level expression."""
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        """Evaluate name references."""
        if node.id in self._allowed_names:
            if self._single_name_mode:
                # Context IS the value (e.g. evaluate({"x": 5}) for row['x'])
                return self._context
            # Namespace mode: look up name in context dict
            return self._context[node.id]
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        if node.id in _SAFE_BUILTINS:
            return _SAFE_BUILTINS[node.id]
        # Should not reach here if validation passed
        msg = f"Unknown name: {node.id}"
        raise ExpressionSecurityError(msg)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Evaluate constants."""
        return self._ensure_finite_float(node.value, context="expression literal")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Evaluate subscript access."""
        value = self.visit(node.value)
        key = self.visit(node.slice)
        try:
            return value[key]
        except KeyError as e:
            # Provide helpful error message with available fields
            if isinstance(value, dict):
                available = list(value.keys())
                msg = f"Field '{key}' not found. Available fields: {available}"
            else:
                msg = f"Key '{key}' not found in {type(value).__name__}"
            raise ExpressionEvaluationError(msg) from e
        except IndexError as e:
            # Handle out-of-range index on lists/tuples
            msg = f"Index {key} out of range for {type(value).__name__} of length {len(value)}"
            raise ExpressionEvaluationError(msg) from e
        except TypeError as e:
            # Handle cases like subscripting None or non-subscriptable types
            msg = f"Cannot access '{key}' on {type(value).__name__}: {e}"
            raise ExpressionEvaluationError(msg) from e

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Evaluate attribute access (only .get on allowed-name values)."""
        value = self.visit(node.value)
        if node.attr == "get":
            if self._single_name_mode and value is self._context:
                # Single-name mode: context is the row/dict itself
                return value.get
            if not self._single_name_mode and isinstance(node.value, ast.Name) and node.value.id in self._allowed_names:
                # Namespace mode: .get() on a top-level allowed name
                return value.get
        msg = f"Forbidden attribute access: {node.attr}"
        raise ExpressionSecurityError(msg)

    def visit_Call(self, node: ast.Call) -> Any:
        """Evaluate function calls (row.get and safe builtins)."""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]

        # Determine function label for error messages from the AST node
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            func_label = "row.get"
        elif isinstance(node.func, ast.Name):
            func_label = node.func.id
        else:
            func_label = "unknown"

        try:
            return func(*args)
        except TypeError as e:
            msg = f"invalid argument to {func_label}(): {e}"
            raise ExpressionEvaluationError(msg) from e
        except (ValueError, OverflowError) as e:
            msg = f"{func_label}() evaluation error: {e}"
            raise ExpressionEvaluationError(msg) from e

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Evaluate comparison chains."""
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            op_func = _COMPARISON_OPS[type(op)]
            try:
                if not op_func(left, right):
                    return False
            except TypeError as e:
                op_name = type(op).__name__
                msg = f"type error in comparison ({op_name}): cannot compare {type(left).__name__} and {type(right).__name__}"
                raise ExpressionEvaluationError(msg) from e
            left = right
        return True

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Evaluate boolean operations (and, or)."""
        if isinstance(node.op, ast.And):
            for value in node.values:
                result = self.visit(value)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            for value in node.values:
                result = self.visit(value)
                if result:
                    return result
            return result
        msg = f"Unknown boolean operator: {type(node.op)}"
        raise ExpressionSecurityError(msg)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Evaluate binary operations."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_name = type(node.op).__name__
        op_func = _BINARY_OPS[type(node.op)]
        try:
            result = op_func(left, right)
        except ZeroDivisionError as e:
            msg = f"division by zero in {op_name} operation"
            raise ExpressionEvaluationError(msg) from e
        except TypeError as e:
            msg = f"type error in {op_name}: cannot apply to {type(left).__name__} and {type(right).__name__}"
            raise ExpressionEvaluationError(msg) from e
        return self._ensure_finite_float(result, context=f"{op_name} operation")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Evaluate unary operations."""
        operand = self.visit(node.operand)
        op_name = type(node.op).__name__
        op_func = _UNARY_OPS[type(node.op)]
        try:
            result = op_func(operand)
        except TypeError as e:
            msg = f"type error in unary {op_name}: cannot apply to {type(operand).__name__}"
            raise ExpressionEvaluationError(msg) from e
        return self._ensure_finite_float(result, context=f"unary {op_name}")

    def visit_List(self, node: ast.List) -> Any:
        """Evaluate list literals."""
        return [self.visit(elt) for elt in node.elts]

    def visit_Dict(self, node: ast.Dict) -> Any:
        """Evaluate dict literals."""
        try:
            keys: list[ast.expr] = []
            for k in node.keys:
                if k is None:
                    raise ExpressionSecurityError("Dict spread (**) reached evaluator — validation bypass detected")
                keys.append(k)
            return {self.visit(k): self.visit(v) for k, v in zip(keys, node.values, strict=True)}
        except TypeError as e:
            # Unhashable key type in dict literal
            msg = f"cannot create dict literal: {e}"
            raise ExpressionEvaluationError(msg) from e

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        """Evaluate tuple literals."""
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        """Evaluate set literals."""
        try:
            return {self.visit(elt) for elt in node.elts}
        except TypeError as e:
            # Unhashable type in set literal (e.g., {[1]})
            msg = f"cannot create set literal: {e}"
            raise ExpressionEvaluationError(msg) from e

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Evaluate ternary expressions."""
        if self.visit(node.test):
            return self.visit(node.body)
        return self.visit(node.orelse)


class ExpressionParser:
    """Safe expression parser for gate conditions.

    Parses and validates expressions at construction time, then evaluates
    them against a context namespace. Only a restricted subset of Python
    is allowed.

    The set of permitted top-level names is configurable via ``allowed_names``.
    By default only ``"row"`` is allowed (gate conditions on pipeline rows).
    Commencement gates pass ``["collections", "dependency_runs", "env"]``
    for dict-context evaluation.

    Allowed operations:
    - Subscript access: name['field'], name['key1']['key2']
    - Method: name.get('field') (single-arg only — defaults are fabrication)
    - Safe builtins: len(), abs()
    - Comparisons: ==, !=, <, >, <=, >=
    - Boolean operators: and, or, not
    - Membership: in, not in
    - Identity: is, is not (for None checks)
    - Literals: strings, numbers, booleans, None
    - List/dict/tuple/set literals for membership checks
    - Ternary expressions: x if condition else y
    - Basic arithmetic: +, -, *, /, //, %

    Forbidden operations:
    - Function calls (except name.get() and safe builtins)
    - Lambda expressions
    - Comprehensions (list, dict, set, generator)
    - Assignment expressions (:=)
    - await, yield
    - f-strings with expressions
    - Attribute access (except name.get)
    - Names not in allowed_names (and not True, False, None)

    Example:
        parser = ExpressionParser("row['confidence'] >= 0.85")
        result = parser.evaluate({"confidence": 0.9})  # Returns True

        parser = ExpressionParser(
            "collections['facts']['count'] > 0",
            allowed_names=["collections"],
        )
        result = parser.evaluate({"collections": {"facts": {"count": 42}}})
    """

    def __init__(self, expression: str, *, allowed_names: list[str] | None = None) -> None:
        """Parse and validate expression at construction time.

        Args:
            expression: The expression string to parse
            allowed_names: Top-level names permitted in the expression.
                Defaults to ["row"] for gate conditions evaluated against
                row data. Commencement gates use ["collections",
                "dependency_runs", "env"] for dict-context evaluation.

        Raises:
            ExpressionSecurityError: If expression contains forbidden constructs
            ExpressionSyntaxError: If expression is not valid Python syntax
        """
        self._expression = expression
        if allowed_names is not None and len(allowed_names) == 0:
            raise ValueError("allowed_names must not be empty")
        self._allowed_names = frozenset(allowed_names) if allowed_names is not None else frozenset({"row"})
        # Single-name mode: default (allowed_names=None) — the caller passes the
        # row value directly as context. This is the standard gate condition path.
        # Namespace mode: caller explicitly passes allowed_names and provides a
        # namespace dict keyed by those names (even if only one name is allowed).
        self._single_name_mode = allowed_names is None

        # Phase 1: Parse the expression
        try:
            self._ast = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            msg = f"Invalid syntax: {e.msg}"
            raise ExpressionSyntaxError(msg) from e

        # Phase 2: Validate for security
        validator = _ExpressionValidator(allowed_names=self._allowed_names)
        validator.visit(self._ast)

        if validator.errors:
            msg = "; ".join(validator.errors)
            raise ExpressionSecurityError(msg)

    @property
    def expression(self) -> str:
        """Return the original expression string."""
        return self._expression

    def is_boolean_expression(self) -> bool:
        """Check if the expression statically returns a boolean.

        Returns True if the expression is guaranteed to return a boolean value:
        - Comparison expressions (==, !=, <, >, <=, >=, in, not in, is, is not)
        - Boolean operators (and, or)
        - Unary not
        - Boolean literals (True, False)
        - Ternary expressions where both branches are boolean expressions

        This is used for config validation: boolean expressions must have
        routes labeled "true"/"false", not arbitrary labels like "above"/"below".
        """
        return self._is_boolean_node(self._ast.body)

    def _is_boolean_node(self, node: ast.expr) -> bool:
        """Recursively check if an AST node returns a boolean."""
        # Comparisons always return bool
        if isinstance(node, ast.Compare):
            return True

        # Boolean operators (and, or) only return bool if ALL operands are boolean
        # Python's `x and y` returns y if x is truthy, not necessarily bool
        # So `row.get('label') or 'default'` returns a string, not bool
        if isinstance(node, ast.BoolOp):
            return all(self._is_boolean_node(v) for v in node.values)

        # Unary not always returns bool
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return True

        # Boolean literals
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return True

        # Name references to True/False
        if isinstance(node, ast.Name) and node.id in ("True", "False"):
            return True

        # Ternary: boolean if both branches are boolean
        if isinstance(node, ast.IfExp):
            return self._is_boolean_node(node.body) and self._is_boolean_node(node.orelse)

        # Everything else (field access, arithmetic, etc.) is not guaranteed boolean
        return False

    def evaluate(self, context: dict[str, Any] | PipelineRow) -> Any:
        """Evaluate expression against context data.

        Args:
            context: Row data dictionary, PipelineRow, or dict namespace
                when using custom allowed_names.

        Returns:
            Result of expression evaluation (typically bool for gate conditions)
        """
        evaluator = _ExpressionEvaluator(
            context,
            allowed_names=self._allowed_names,
            single_name_mode=self._single_name_mode,
        )
        try:
            return evaluator.visit(self._ast)
        except (ExpressionEvaluationError, ExpressionSecurityError):
            raise
        except contract_errors.TIER_1_ERRORS:
            raise  # Framework bugs must not be wrapped as evaluation errors
        except (TypeError, AttributeError, KeyError, NameError, AssertionError, RecursionError):
            raise  # Programming errors in the evaluator must crash through
        except Exception as exc:
            raise ExpressionEvaluationError(
                f"Unexpected error evaluating expression {self._expression!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return f"ExpressionParser({self._expression!r})"
