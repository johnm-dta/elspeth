# tests/engine/test_expression_parser.py
"""Tests for safe expression parser."""

import pytest

from elspeth.engine.expression_parser import (
    ExpressionParser,
    ExpressionSecurityError,
    ExpressionSyntaxError,
)


class TestExpressionParserBasicOperations:
    """Test basic allowed operations."""

    def test_simple_equality(self) -> None:
        parser = ExpressionParser("row['status'] == 'active'")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "inactive"}) is False

    def test_numeric_comparison(self) -> None:
        parser = ExpressionParser("row['confidence'] >= 0.85")
        assert parser.evaluate({"confidence": 0.9}) is True
        assert parser.evaluate({"confidence": 0.85}) is True
        assert parser.evaluate({"confidence": 0.8}) is False

    def test_less_than(self) -> None:
        parser = ExpressionParser("row['count'] < 10")
        assert parser.evaluate({"count": 5}) is True
        assert parser.evaluate({"count": 10}) is False
        assert parser.evaluate({"count": 15}) is False

    def test_greater_than(self) -> None:
        parser = ExpressionParser("row['value'] > 100")
        assert parser.evaluate({"value": 150}) is True
        assert parser.evaluate({"value": 100}) is False
        assert parser.evaluate({"value": 50}) is False

    def test_less_than_or_equal(self) -> None:
        parser = ExpressionParser("row['priority'] <= 3")
        assert parser.evaluate({"priority": 2}) is True
        assert parser.evaluate({"priority": 3}) is True
        assert parser.evaluate({"priority": 4}) is False

    def test_not_equal(self) -> None:
        parser = ExpressionParser("row['status'] != 'deleted'")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "deleted"}) is False


class TestExpressionParserBooleanOperations:
    """Test boolean and/or/not operations."""

    def test_and_operator(self) -> None:
        parser = ExpressionParser("row['status'] == 'active' and row['balance'] > 0")
        assert parser.evaluate({"status": "active", "balance": 100}) is True
        assert parser.evaluate({"status": "active", "balance": 0}) is False
        assert parser.evaluate({"status": "inactive", "balance": 100}) is False
        assert parser.evaluate({"status": "inactive", "balance": 0}) is False

    def test_or_operator(self) -> None:
        parser = ExpressionParser(
            "row['status'] == 'active' or row['override'] == True"
        )
        assert parser.evaluate({"status": "active", "override": False}) is True
        assert parser.evaluate({"status": "inactive", "override": True}) is True
        assert parser.evaluate({"status": "inactive", "override": False}) is False

    def test_not_operator(self) -> None:
        parser = ExpressionParser("not row['disabled']")
        assert parser.evaluate({"disabled": False}) is True
        assert parser.evaluate({"disabled": True}) is False

    def test_complex_boolean_expression(self) -> None:
        parser = ExpressionParser(
            "(row['status'] == 'active' or row['status'] == 'pending') "
            "and row['score'] >= 0.5"
        )
        assert parser.evaluate({"status": "active", "score": 0.7}) is True
        assert parser.evaluate({"status": "pending", "score": 0.6}) is True
        assert parser.evaluate({"status": "active", "score": 0.3}) is False
        assert parser.evaluate({"status": "deleted", "score": 0.9}) is False


class TestExpressionParserMembership:
    """Test membership operations (in, not in)."""

    def test_in_list(self) -> None:
        parser = ExpressionParser("row['status'] in ['active', 'pending']")
        assert parser.evaluate({"status": "active"}) is True
        assert parser.evaluate({"status": "pending"}) is True
        assert parser.evaluate({"status": "deleted"}) is False

    def test_not_in_list(self) -> None:
        parser = ExpressionParser("row['category'] not in ['spam', 'trash']")
        assert parser.evaluate({"category": "inbox"}) is True
        assert parser.evaluate({"category": "spam"}) is False

    def test_in_tuple(self) -> None:
        parser = ExpressionParser("row['code'] in (1, 2, 3)")
        assert parser.evaluate({"code": 2}) is True
        assert parser.evaluate({"code": 5}) is False

    def test_in_set(self) -> None:
        parser = ExpressionParser("row['tag'] in {'a', 'b', 'c'}")
        assert parser.evaluate({"tag": "b"}) is True
        assert parser.evaluate({"tag": "d"}) is False


class TestExpressionParserRowGet:
    """Test row.get() method access."""

    def test_row_get_basic(self) -> None:
        parser = ExpressionParser("row.get('status') == 'active'")
        assert parser.evaluate({"status": "active"}) is True

    def test_row_get_missing_key_returns_none(self) -> None:
        parser = ExpressionParser("row.get('missing') is None")
        assert parser.evaluate({}) is True

    def test_row_get_with_default(self) -> None:
        parser = ExpressionParser("row.get('status', 'unknown') == 'unknown'")
        assert parser.evaluate({}) is True
        assert parser.evaluate({"status": "active"}) is False

    def test_row_get_with_default_when_key_exists(self) -> None:
        parser = ExpressionParser("row.get('status', 'default') == 'active'")
        assert parser.evaluate({"status": "active"}) is True


class TestExpressionParserNoneChecks:
    """Test is/is not for None checks."""

    def test_is_none(self) -> None:
        parser = ExpressionParser("row.get('optional') is None")
        assert parser.evaluate({}) is True
        assert parser.evaluate({"optional": None}) is True
        assert parser.evaluate({"optional": "value"}) is False

    def test_is_not_none(self) -> None:
        parser = ExpressionParser("row.get('required') is not None")
        assert parser.evaluate({"required": "value"}) is True
        assert parser.evaluate({"required": 0}) is True  # 0 is not None
        assert parser.evaluate({}) is False


class TestExpressionParserArithmetic:
    """Test arithmetic operations."""

    def test_addition(self) -> None:
        parser = ExpressionParser("row['a'] + row['b'] > 10")
        assert parser.evaluate({"a": 5, "b": 6}) is True
        assert parser.evaluate({"a": 5, "b": 4}) is False

    def test_subtraction(self) -> None:
        parser = ExpressionParser("row['x'] - row['y'] == 5")
        assert parser.evaluate({"x": 10, "y": 5}) is True

    def test_multiplication(self) -> None:
        parser = ExpressionParser("row['qty'] * row['price'] >= 100")
        assert parser.evaluate({"qty": 5, "price": 25}) is True
        assert parser.evaluate({"qty": 2, "price": 10}) is False

    def test_division(self) -> None:
        parser = ExpressionParser("row['total'] / row['count'] > 5")
        assert parser.evaluate({"total": 30, "count": 5}) is True

    def test_floor_division(self) -> None:
        parser = ExpressionParser("row['value'] // 10 == 4")
        assert parser.evaluate({"value": 45}) is True
        assert parser.evaluate({"value": 49}) is True
        assert parser.evaluate({"value": 50}) is False

    def test_modulo(self) -> None:
        parser = ExpressionParser("row['number'] % 2 == 0")
        assert parser.evaluate({"number": 4}) is True
        assert parser.evaluate({"number": 5}) is False

    def test_unary_minus(self) -> None:
        parser = ExpressionParser("-row['value'] < 0")
        assert parser.evaluate({"value": 5}) is True
        assert parser.evaluate({"value": -5}) is False


class TestExpressionParserTernary:
    """Test ternary (conditional) expressions."""

    def test_ternary_true_branch(self) -> None:
        parser = ExpressionParser("'high' if row['score'] >= 0.8 else 'low'")
        assert parser.evaluate({"score": 0.9}) == "high"
        assert parser.evaluate({"score": 0.5}) == "low"

    def test_ternary_in_comparison(self) -> None:
        parser = ExpressionParser(
            "(row.get('priority', 'normal') if row.get('urgent') else 'low') == 'high'"
        )
        assert parser.evaluate({"urgent": True, "priority": "high"}) is True
        assert parser.evaluate({"urgent": False, "priority": "high"}) is False


class TestExpressionParserComparisonChains:
    """Test comparison chains."""

    def test_chained_comparison(self) -> None:
        parser = ExpressionParser("0 < row['value'] < 100")
        assert parser.evaluate({"value": 50}) is True
        assert parser.evaluate({"value": 0}) is False
        assert parser.evaluate({"value": 100}) is False
        assert parser.evaluate({"value": -5}) is False

    def test_double_equals_chain(self) -> None:
        parser = ExpressionParser("row['a'] == row['b'] == 1")
        assert parser.evaluate({"a": 1, "b": 1}) is True
        assert parser.evaluate({"a": 1, "b": 2}) is False


class TestExpressionParserSecurityRejections:
    """Test that forbidden constructs are rejected at parse time."""

    def test_reject_import(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("__import__('os')")

    def test_reject_eval(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("eval('malicious')")

    def test_reject_exec(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("exec('malicious')")

    def test_reject_compile(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden"):
            ExpressionParser("compile('code', 'file', 'exec')")

    def test_reject_lambda(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Lambda expressions"):
            ExpressionParser("(lambda: True)()")

    def test_reject_list_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="List comprehensions"):
            ExpressionParser("[x for x in range(10)]")

    def test_reject_dict_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Dict comprehensions"):
            ExpressionParser("{k: v for k, v in items}")

    def test_reject_set_comprehension(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Set comprehensions"):
            ExpressionParser("{x for x in items}")

    def test_reject_generator_expression(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Generator expressions"):
            ExpressionParser("list(x for x in range(10))")

    def test_reject_attribute_access_dunder(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.__class__")

    def test_reject_attribute_access_arbitrary(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.items()")

    def test_reject_assignment_expression(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Assignment expressions"):
            ExpressionParser("(x := 5)")

    def test_reject_fstring(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="F-string"):
            ExpressionParser("f\"value: {row['x']}\"")

    def test_reject_arbitrary_function_call(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden function call"):
            ExpressionParser("len(row)")

    def test_reject_method_call_not_get(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden row attribute"):
            ExpressionParser("row.keys()")

    def test_reject_builtin_access(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("open('/etc/passwd')")

    def test_reject_arbitrary_name(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="Forbidden name"):
            ExpressionParser("some_var == 'value'")

    def test_reject_row_get_too_few_args(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="requires 1 or 2 arguments"):
            ExpressionParser("row.get()")

    def test_reject_row_get_too_many_args(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="requires 1 or 2 arguments"):
            ExpressionParser("row.get('a', 'b', 'c')")

    def test_reject_row_get_with_kwargs(self) -> None:
        with pytest.raises(ExpressionSecurityError, match="keyword arguments"):
            ExpressionParser("row.get(key='field')")

    def test_reject_starred_expression(self) -> None:
        """Starred expressions (*x) must be rejected at parse time."""
        with pytest.raises(ExpressionSecurityError, match="Starred expressions"):
            ExpressionParser("[*row['items']]")

    def test_reject_starred_expression_in_tuple(self) -> None:
        """Starred expressions in tuples must be rejected."""
        with pytest.raises(ExpressionSecurityError, match="Starred expressions"):
            ExpressionParser("(*row['items'],)")

    def test_reject_dict_spread(self) -> None:
        """Dict spread (**x) must be rejected at parse time."""
        with pytest.raises(ExpressionSecurityError, match="Dict spread"):
            ExpressionParser("{**row['data']}")

    def test_reject_dict_spread_mixed(self) -> None:
        """Dict spread mixed with regular keys must be rejected."""
        with pytest.raises(ExpressionSecurityError, match="Dict spread"):
            ExpressionParser("{'key': 1, **row['data']}")


class TestExpressionParserSyntaxErrors:
    """Test that syntax errors are handled correctly."""

    def test_invalid_syntax(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("row['field ==")

    def test_incomplete_expression(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("row['field'] ==")

    def test_mismatched_parens(self) -> None:
        with pytest.raises(ExpressionSyntaxError, match="Invalid syntax"):
            ExpressionParser("(row['field'] == 'value'")


class TestExpressionParserEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_string_comparison(self) -> None:
        parser = ExpressionParser("row['name'] == ''")
        assert parser.evaluate({"name": ""}) is True
        assert parser.evaluate({"name": "value"}) is False

    def test_zero_comparison(self) -> None:
        parser = ExpressionParser("row['count'] == 0")
        assert parser.evaluate({"count": 0}) is True
        assert parser.evaluate({"count": 1}) is False

    def test_false_boolean_comparison(self) -> None:
        parser = ExpressionParser("row['flag'] == False")
        assert parser.evaluate({"flag": False}) is True
        assert parser.evaluate({"flag": True}) is False

    def test_nested_subscript(self) -> None:
        parser = ExpressionParser("row['data']['nested'] == 'value'")
        assert parser.evaluate({"data": {"nested": "value"}}) is True

    def test_expression_property(self) -> None:
        parser = ExpressionParser("row['x'] == 1")
        assert parser.expression == "row['x'] == 1"

    def test_repr(self) -> None:
        parser = ExpressionParser("row['x'] == 1")
        assert repr(parser) == "ExpressionParser(\"row['x'] == 1\")"

    def test_dict_literal_in_expression(self) -> None:
        parser = ExpressionParser("row['key'] in {'a': 1, 'b': 2}")
        assert parser.evaluate({"key": "a"}) is True
        assert parser.evaluate({"key": "c"}) is False

    def test_negative_number_literal(self) -> None:
        parser = ExpressionParser("row['value'] > -10")
        assert parser.evaluate({"value": 0}) is True
        assert parser.evaluate({"value": -20}) is False

    def test_float_literal(self) -> None:
        parser = ExpressionParser("row['ratio'] < 0.5")
        assert parser.evaluate({"ratio": 0.3}) is True
        assert parser.evaluate({"ratio": 0.7}) is False

    def test_multiple_and_conditions(self) -> None:
        parser = ExpressionParser("row['a'] == 1 and row['b'] == 2 and row['c'] == 3")
        assert parser.evaluate({"a": 1, "b": 2, "c": 3}) is True
        assert parser.evaluate({"a": 1, "b": 2, "c": 4}) is False

    def test_multiple_or_conditions(self) -> None:
        parser = ExpressionParser(
            "row['status'] == 'a' or row['status'] == 'b' or row['status'] == 'c'"
        )
        assert parser.evaluate({"status": "b"}) is True
        assert parser.evaluate({"status": "d"}) is False


class TestExpressionParserRealWorldExamples:
    """Test real-world gate condition examples."""

    def test_confidence_threshold_gate(self) -> None:
        """Classic confidence threshold routing."""
        parser = ExpressionParser("row['confidence'] >= 0.85")
        assert parser.evaluate({"confidence": 0.9, "label": "positive"}) is True
        assert parser.evaluate({"confidence": 0.7, "label": "positive"}) is False

    def test_status_routing_gate(self) -> None:
        """Route based on status field."""
        parser = ExpressionParser(
            "row['status'] in ['approved', 'verified'] and row.get('errors') is None"
        )
        assert parser.evaluate({"status": "approved", "data": "..."}) is True
        assert parser.evaluate({"status": "approved", "errors": ["err"]}) is False
        assert parser.evaluate({"status": "pending", "data": "..."}) is False

    def test_multi_field_validation_gate(self) -> None:
        """Validate multiple required fields present."""
        parser = ExpressionParser(
            "row.get('name') is not None and row.get('email') is not None"
        )
        assert parser.evaluate({"name": "John", "email": "john@example.com"}) is True
        assert parser.evaluate({"name": "John"}) is False

    def test_amount_range_gate(self) -> None:
        """Check if amount is within acceptable range."""
        parser = ExpressionParser("row['amount'] > 0 and row['amount'] <= 10000")
        assert parser.evaluate({"amount": 5000}) is True
        assert parser.evaluate({"amount": 0}) is False
        assert parser.evaluate({"amount": 15000}) is False

    def test_category_with_score_gate(self) -> None:
        """Route high-confidence items in specific categories."""
        parser = ExpressionParser(
            "(row['category'] == 'urgent' and row['score'] >= 0.9) or "
            "(row['category'] == 'normal' and row['score'] >= 0.7)"
        )
        assert parser.evaluate({"category": "urgent", "score": 0.95}) is True
        assert parser.evaluate({"category": "urgent", "score": 0.8}) is False
        assert parser.evaluate({"category": "normal", "score": 0.75}) is True
        assert parser.evaluate({"category": "normal", "score": 0.5}) is False
