"""Tests for error/reason schema contracts.

Tests for:
- ExecutionError TypedDict (exception, type, traceback fields)
- RoutingReason TypedDict (rule, matched_value, threshold fields)
- TransformReason TypedDict (action, fields_modified fields)
"""


class TestExecutionError:
    """Tests for ExecutionError TypedDict."""

    def test_execution_error_has_required_fields(self) -> None:
        """ExecutionError defines exception and type fields."""
        from elspeth.contracts import ExecutionError

        error: ExecutionError = {
            "exception": "ValueError: invalid input",
            "type": "ValueError",
        }

        assert error["exception"] == "ValueError: invalid input"
        assert error["type"] == "ValueError"

    def test_execution_error_accepts_optional_traceback(self) -> None:
        """ExecutionError can include traceback."""
        from elspeth.contracts import ExecutionError

        error: ExecutionError = {
            "exception": "KeyError: 'foo'",
            "type": "KeyError",
            "traceback": "Traceback (most recent call last):\n...",
        }

        assert "traceback" in error


class TestRoutingReason:
    """Tests for RoutingReason TypedDict."""

    def test_routing_reason_has_rule_field(self) -> None:
        """RoutingReason defines rule field."""
        from elspeth.contracts import RoutingReason

        reason: RoutingReason = {
            "rule": "value > threshold",
            "matched_value": 42,
        }

        assert reason["rule"] == "value > threshold"

    def test_routing_reason_accepts_threshold(self) -> None:
        """RoutingReason can include threshold."""
        from elspeth.contracts import RoutingReason

        reason: RoutingReason = {
            "rule": "value > threshold",
            "matched_value": 42,
            "threshold": 10.0,
        }

        assert reason["threshold"] == 10.0


class TestTransformReason:
    """Tests for TransformReason TypedDict."""

    def test_transform_reason_has_action_field(self) -> None:
        """TransformReason defines action field."""
        from elspeth.contracts import TransformReason

        reason: TransformReason = {
            "action": "normalized_field",
        }

        assert reason["action"] == "normalized_field"

    def test_transform_reason_accepts_optional_fields(self) -> None:
        """TransformReason can include optional fields_modified and validation_errors."""
        from elspeth.contracts import TransformReason

        reason: TransformReason = {
            "action": "validated_and_normalized",
            "fields_modified": ["name", "email"],
            "validation_errors": ["missing_phone"],
        }

        assert reason["fields_modified"] == ["name", "email"]
        assert reason["validation_errors"] == ["missing_phone"]
