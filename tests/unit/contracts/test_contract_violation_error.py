"""Tests for contract violation to error reason conversion.

Tests for:
- ContractViolation.to_error_reason(): base conversion with reason, violation_type, field, original_field
- TypeMismatchViolation.to_error_reason(): adds expected, actual, value
- MissingFieldViolation.to_error_reason(): uses base implementation
- ExtraFieldViolation.to_error_reason(): uses base implementation
- violations_to_error_reason(): single violation returns direct dict, multiple wraps with count

These methods enable converting schema contract violations to TransformErrorReason dicts
suitable for TransformResult.error().
"""

import pytest


class TestContractViolationToErrorReason:
    """Tests for ContractViolation.to_error_reason() base method."""

    def test_to_error_reason_returns_dict(self) -> None:
        """to_error_reason() returns a dict."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        result = exc.to_error_reason()
        assert isinstance(result, dict)

    def test_to_error_reason_has_reason_key(self) -> None:
        """to_error_reason() result has 'reason' key."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        result = exc.to_error_reason()
        assert "reason" in result
        assert result["reason"] == "contract_violation"

    def test_to_error_reason_has_violation_type(self) -> None:
        """to_error_reason() result has 'violation_type' key with class name."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        result = exc.to_error_reason()
        assert result["violation_type"] == "ContractViolation"

    def test_to_error_reason_has_field(self) -> None:
        """to_error_reason() result has 'field' key with normalized_name."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        result = exc.to_error_reason()
        assert result["field"] == "customer_id"

    def test_to_error_reason_has_original_field(self) -> None:
        """to_error_reason() result has 'original_field' key with original_name."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        result = exc.to_error_reason()
        assert result["original_field"] == "Customer ID"


class TestMissingFieldViolationToErrorReason:
    """Tests for MissingFieldViolation.to_error_reason()."""

    def test_missing_field_to_error_reason_has_reason(self) -> None:
        """MissingFieldViolation.to_error_reason() has 'reason' key."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="amount", original_name="Amount")
        result = exc.to_error_reason()
        assert result["reason"] == "contract_violation"

    def test_missing_field_to_error_reason_has_violation_type(self) -> None:
        """MissingFieldViolation.to_error_reason() has correct violation_type."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="amount", original_name="Amount")
        result = exc.to_error_reason()
        assert result["violation_type"] == "MissingFieldViolation"

    def test_missing_field_to_error_reason_has_field(self) -> None:
        """MissingFieldViolation.to_error_reason() has field from normalized_name."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="amount", original_name="Amount")
        result = exc.to_error_reason()
        assert result["field"] == "amount"

    def test_missing_field_to_error_reason_has_original_field(self) -> None:
        """MissingFieldViolation.to_error_reason() has original_field from original_name."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="amount", original_name="Amount")
        result = exc.to_error_reason()
        assert result["original_field"] == "Amount"


class TestTypeMismatchViolationToErrorReason:
    """Tests for TypeMismatchViolation.to_error_reason() override."""

    def test_type_mismatch_to_error_reason_has_reason(self) -> None:
        """TypeMismatchViolation.to_error_reason() has 'reason' key."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        result = exc.to_error_reason()
        assert result["reason"] == "contract_violation"

    def test_type_mismatch_to_error_reason_has_violation_type(self) -> None:
        """TypeMismatchViolation.to_error_reason() has correct violation_type."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        result = exc.to_error_reason()
        assert result["violation_type"] == "TypeMismatchViolation"

    def test_type_mismatch_to_error_reason_has_expected(self) -> None:
        """TypeMismatchViolation.to_error_reason() includes expected as string."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        result = exc.to_error_reason()
        assert result["expected"] == "int"

    def test_type_mismatch_to_error_reason_has_actual(self) -> None:
        """TypeMismatchViolation.to_error_reason() includes actual as string."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        result = exc.to_error_reason()
        assert result["actual"] == "str"

    def test_type_mismatch_to_error_reason_has_value_as_repr(self) -> None:
        """TypeMismatchViolation.to_error_reason() includes value as repr."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        result = exc.to_error_reason()
        assert result["value"] == "'not_a_number'"

    def test_type_mismatch_to_error_reason_repr_handles_complex_values(self) -> None:
        """TypeMismatchViolation.to_error_reason() uses repr for complex values."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="data",
            original_name="Data",
            expected_type=str,
            actual_type=list,
            actual_value=[1, 2, 3],
        )
        result = exc.to_error_reason()
        assert result["value"] == "[1, 2, 3]"


class TestExtraFieldViolationToErrorReason:
    """Tests for ExtraFieldViolation.to_error_reason()."""

    def test_extra_field_to_error_reason_has_reason(self) -> None:
        """ExtraFieldViolation.to_error_reason() has 'reason' key."""
        from elspeth.contracts.errors import ExtraFieldViolation

        exc = ExtraFieldViolation(normalized_name="unknown_col", original_name="Unknown Col")
        result = exc.to_error_reason()
        assert result["reason"] == "contract_violation"

    def test_extra_field_to_error_reason_has_violation_type(self) -> None:
        """ExtraFieldViolation.to_error_reason() has correct violation_type."""
        from elspeth.contracts.errors import ExtraFieldViolation

        exc = ExtraFieldViolation(normalized_name="unknown_col", original_name="Unknown Col")
        result = exc.to_error_reason()
        assert result["violation_type"] == "ExtraFieldViolation"

    def test_extra_field_to_error_reason_has_field(self) -> None:
        """ExtraFieldViolation.to_error_reason() has field from normalized_name."""
        from elspeth.contracts.errors import ExtraFieldViolation

        exc = ExtraFieldViolation(normalized_name="unknown_col", original_name="Unknown Col")
        result = exc.to_error_reason()
        assert result["field"] == "unknown_col"


class TestViolationsToErrorReason:
    """Tests for violations_to_error_reason() helper function."""

    def test_single_violation_returns_direct_dict(self) -> None:
        """violations_to_error_reason() with single violation returns its to_error_reason()."""
        from elspeth.contracts.errors import MissingFieldViolation, violations_to_error_reason

        violation = MissingFieldViolation(normalized_name="id", original_name="ID")
        result = violations_to_error_reason([violation])
        assert result["reason"] == "contract_violation"
        assert result["violation_type"] == "MissingFieldViolation"
        assert result["field"] == "id"

    def test_multiple_violations_returns_wrapped_dict(self) -> None:
        """violations_to_error_reason() with multiple violations returns wrapped dict."""
        from elspeth.contracts.errors import (
            MissingFieldViolation,
            TypeMismatchViolation,
            violations_to_error_reason,
        )

        violations = [
            MissingFieldViolation(normalized_name="id", original_name="ID"),
            TypeMismatchViolation(
                normalized_name="amount",
                original_name="Amount",
                expected_type=int,
                actual_type=str,
                actual_value="bad",
            ),
        ]
        result = violations_to_error_reason(violations)
        assert result["reason"] == "multiple_contract_violations"

    def test_multiple_violations_has_count(self) -> None:
        """violations_to_error_reason() with multiple violations includes count."""
        from elspeth.contracts.errors import (
            ExtraFieldViolation,
            MissingFieldViolation,
            violations_to_error_reason,
        )

        violations = [
            MissingFieldViolation(normalized_name="id", original_name="ID"),
            ExtraFieldViolation(normalized_name="extra", original_name="Extra"),
        ]
        result = violations_to_error_reason(violations)
        assert result["count"] == 2

    def test_multiple_violations_has_violations_list(self) -> None:
        """violations_to_error_reason() with multiple violations includes violations list."""
        from elspeth.contracts.errors import (
            ExtraFieldViolation,
            MissingFieldViolation,
            violations_to_error_reason,
        )

        violations = [
            MissingFieldViolation(normalized_name="id", original_name="ID"),
            ExtraFieldViolation(normalized_name="extra", original_name="Extra"),
        ]
        result = violations_to_error_reason(violations)
        assert "violations" in result
        assert len(result["violations"]) == 2

    def test_multiple_violations_list_contains_error_reasons(self) -> None:
        """violations_to_error_reason() violations list contains to_error_reason() results."""
        from elspeth.contracts.errors import (
            ExtraFieldViolation,
            MissingFieldViolation,
            violations_to_error_reason,
        )

        violations = [
            MissingFieldViolation(normalized_name="id", original_name="ID"),
            ExtraFieldViolation(normalized_name="extra", original_name="Extra"),
        ]
        result = violations_to_error_reason(violations)
        violation_types = [v["violation_type"] for v in result["violations"]]
        assert "MissingFieldViolation" in violation_types
        assert "ExtraFieldViolation" in violation_types

    def test_empty_list_raises_value_error(self) -> None:
        """violations_to_error_reason() with empty list raises ValueError."""
        from elspeth.contracts.errors import violations_to_error_reason

        with pytest.raises(ValueError, match="cannot be empty"):
            violations_to_error_reason([])

    def test_three_violations_has_correct_count(self) -> None:
        """violations_to_error_reason() with three violations has count=3."""
        from elspeth.contracts.errors import (
            ExtraFieldViolation,
            MissingFieldViolation,
            TypeMismatchViolation,
            violations_to_error_reason,
        )

        violations = [
            MissingFieldViolation(normalized_name="id", original_name="ID"),
            TypeMismatchViolation(
                normalized_name="amount",
                original_name="Amount",
                expected_type=int,
                actual_type=str,
                actual_value="x",
            ),
            ExtraFieldViolation(normalized_name="extra", original_name="Extra"),
        ]
        result = violations_to_error_reason(violations)
        assert result["count"] == 3
        assert len(result["violations"]) == 3


class TestTransformErrorReasonAlignment:
    """Ensure helper-emitted keys are declared in TransformErrorReason."""

    def test_contract_violation_helper_keys_are_declared(self) -> None:
        """All keys emitted by violation helpers must exist in TransformErrorReason."""
        from elspeth.contracts.errors import (
            ContractViolation,
            MissingFieldViolation,
            TransformErrorReason,
            TypeMismatchViolation,
            violations_to_error_reason,
        )

        declared_keys = set(TransformErrorReason.__annotations__)

        reasons = [
            ContractViolation(normalized_name="id", original_name="ID").to_error_reason(),
            TypeMismatchViolation(
                normalized_name="amount",
                original_name="Amount",
                expected_type=int,
                actual_type=str,
                actual_value="bad",
            ).to_error_reason(),
            violations_to_error_reason(
                [
                    MissingFieldViolation(normalized_name="id", original_name="ID"),
                    TypeMismatchViolation(
                        normalized_name="amount",
                        original_name="Amount",
                        expected_type=int,
                        actual_type=str,
                        actual_value="bad",
                    ),
                ]
            ),
        ]

        emitted_keys = set().union(*(set(reason) for reason in reasons))
        # multiple_contract_violations embeds per-violation reasons under "violations"
        nested_reasons = reasons[-1]["violations"]
        emitted_keys.update(set().union(*(set(reason) for reason in nested_reasons)))

        missing_keys = emitted_keys - declared_keys
        assert missing_keys == set(), f"Undeclared TransformErrorReason keys: {sorted(missing_keys)}"
