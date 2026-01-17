# src/elspeth/engine/expression_parser.py
"""Safe expression parser for gate conditions.

Uses Python's ast module to parse and evaluate expressions in a restricted
subset of Python. This is NOT eval() - it's a secure whitelist-based parser.

The parser operates in two phases:
1. Parse-time validation: Reject forbidden constructs at construction
2. Evaluation: Safely execute the validated AST against row data

Security model:
- Plugins are system code (trusted), but config expressions could come from
  config files that might be misconfigured or tampered with
- This parser is defense-in-depth - restricting what expressions can do
"""

from __future__ import annotations

import ast
import operator
from typing import Any


class ExpressionSecurityError(Exception):
    """Raised when expression contains forbidden constructs."""


class ExpressionSyntaxError(Exception):
    """Raised when expression is not valid Python syntax."""


# Allowed comparison operators
_COMPARISON_OPS: dict[type[ast.cmpop], Any] = {
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

# Allowed binary operators
_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

# Allowed unary operators
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.Not: operator.not_,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed boolean operators
_BOOL_OPS: dict[type[ast.boolop], str] = {
    ast.And: "and",
    ast.Or: "or",
}


class _ExpressionValidator(ast.NodeVisitor):
    """AST visitor that validates expressions for security.

    Raises ExpressionSecurityError if any forbidden construct is found.
    """

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Name(self, node: ast.Name) -> None:
        """Allow only 'row' as a name."""
        if node.id not in ("row", "True", "False", "None"):
            self.errors.append(f"Forbidden name: {node.id!r}")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Allow row['field'] subscript access."""
        # Subscript access is allowed - the evaluator will enforce row access
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Allow only row.get method access."""
        if isinstance(node.value, ast.Name) and node.value.id == "row":
            if node.attr != "get":
                self.errors.append(
                    f"Forbidden row attribute: {node.attr!r} (only 'get' is allowed)"
                )
        else:
            self.errors.append(f"Forbidden attribute access: {node.attr!r}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Allow only row.get() calls."""
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "row"
            and node.func.attr == "get"
        ):
            # row.get() is allowed with 1 or 2 arguments
            if len(node.args) < 1 or len(node.args) > 2:
                self.errors.append(
                    f"row.get() requires 1 or 2 arguments, got {len(node.args)}"
                )
            if node.keywords:
                self.errors.append("row.get() does not accept keyword arguments")
            self.generic_visit(node)
            return
        self.errors.append(f"Forbidden function call: {ast.dump(node.func)}")
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Validate comparison operators."""
        for op in node.ops:
            if type(op) not in _COMPARISON_OPS:
                self.errors.append(
                    f"Forbidden comparison operator: {type(op).__name__}"
                )
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
        if isinstance(node.value, str | int | float | bool):
            return  # Primitives allowed
        self.errors.append(f"Forbidden constant type: {type(node.value).__name__}")

    def visit_List(self, node: ast.List) -> None:
        """Allow list literals for membership checks."""
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        """Allow dict literals for membership checks."""
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


class _ExpressionEvaluator(ast.NodeVisitor):
    """AST visitor that evaluates validated expressions."""

    def __init__(self, row: dict[str, Any]) -> None:
        self._row = row

    def visit_Expression(self, node: ast.Expression) -> Any:
        """Evaluate the top-level expression."""
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name) -> Any:
        """Evaluate name references."""
        if node.id == "row":
            return self._row
        if node.id == "True":
            return True
        if node.id == "False":
            return False
        if node.id == "None":
            return None
        # Should not reach here if validation passed
        msg = f"Unknown name: {node.id}"
        raise ExpressionSecurityError(msg)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Evaluate constants."""
        return node.value

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Evaluate subscript access."""
        value = self.visit(node.value)
        key = self.visit(node.slice)
        return value[key]

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Evaluate attribute access (only row.get allowed)."""
        value = self.visit(node.value)
        if value is self._row and node.attr == "get":
            return value.get
        msg = f"Forbidden attribute access: {node.attr}"
        raise ExpressionSecurityError(msg)

    def visit_Call(self, node: ast.Call) -> Any:
        """Evaluate function calls (only row.get allowed)."""
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return func(*args)

    def visit_Compare(self, node: ast.Compare) -> Any:
        """Evaluate comparison chains."""
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            op_func = _COMPARISON_OPS[type(op)]
            if not op_func(left, right):
                return False
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
        op_func = _BINARY_OPS[type(node.op)]
        return op_func(left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Evaluate unary operations."""
        operand = self.visit(node.operand)
        op_func = _UNARY_OPS[type(node.op)]
        return op_func(operand)

    def visit_List(self, node: ast.List) -> Any:
        """Evaluate list literals."""
        return [self.visit(elt) for elt in node.elts]

    def visit_Dict(self, node: ast.Dict) -> Any:
        """Evaluate dict literals."""
        return {
            self.visit(k): self.visit(v)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None  # Handle **spread (not allowed, but be safe)
        }

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        """Evaluate tuple literals."""
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Set(self, node: ast.Set) -> Any:
        """Evaluate set literals."""
        return {self.visit(elt) for elt in node.elts}

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Evaluate ternary expressions."""
        if self.visit(node.test):
            return self.visit(node.body)
        return self.visit(node.orelse)


class ExpressionParser:
    """Safe expression parser for gate conditions.

    Parses and validates expressions at construction time, then evaluates
    them against row data. Only a restricted subset of Python is allowed.

    Allowed operations:
    - Field access: row['field'], row.get('field'), row.get('field', default)
    - Comparisons: ==, !=, <, >, <=, >=
    - Boolean operators: and, or, not
    - Membership: in, not in
    - Identity: is, is not (for None checks)
    - Literals: strings, numbers, booleans, None
    - List/dict/tuple/set literals for membership checks
    - Ternary expressions: x if condition else y
    - Basic arithmetic: +, -, *, /, //, %

    Forbidden operations:
    - Function calls (except row.get())
    - Lambda expressions
    - Comprehensions (list, dict, set, generator)
    - Assignment expressions (:=)
    - await, yield
    - f-strings with expressions
    - Attribute access (except row.get)
    - Names other than 'row', 'True', 'False', 'None'

    Example:
        parser = ExpressionParser("row['confidence'] >= 0.85")
        result = parser.evaluate({"confidence": 0.9})  # Returns True
    """

    def __init__(self, expression: str) -> None:
        """Parse and validate expression at construction time.

        Args:
            expression: The expression string to parse

        Raises:
            ExpressionSecurityError: If expression contains forbidden constructs
            ExpressionSyntaxError: If expression is not valid Python syntax
        """
        self._expression = expression

        # Phase 1: Parse the expression
        try:
            self._ast = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            msg = f"Invalid syntax: {e.msg}"
            raise ExpressionSyntaxError(msg) from e

        # Phase 2: Validate for security
        validator = _ExpressionValidator()
        validator.visit(self._ast)

        if validator.errors:
            msg = "; ".join(validator.errors)
            raise ExpressionSecurityError(msg)

    @property
    def expression(self) -> str:
        """Return the original expression string."""
        return self._expression

    def evaluate(self, row: dict[str, Any]) -> Any:
        """Evaluate expression against row data.

        Args:
            row: Row data dictionary

        Returns:
            Result of expression evaluation (typically bool for gate conditions)
        """
        evaluator = _ExpressionEvaluator(row)
        return evaluator.visit(self._ast)

    def __repr__(self) -> str:
        return f"ExpressionParser({self._expression!r})"
