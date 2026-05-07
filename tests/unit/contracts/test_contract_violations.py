"""Tests for schema contract violation exception types.

Tests for:
- ContractViolation: base exception with normalized_name and original_name
- MissingFieldViolation: error message shows both names with "missing" or "required"
- TypeMismatchViolation: stores expected_type, actual_type, actual_value; message shows types
- ExtraFieldViolation: message mentions the field
- ContractMergeError: ValueError subclass, message shows field and conflicting types

These are Tier 3 data violations (external data issues), not system errors.
They result in row quarantine, not crashes.
Error messages follow the "'original' (normalized)" format for debuggability.
"""


class TestContractViolationBase:
    """Tests for ContractViolation base exception."""

    def test_contract_violation_is_exception(self) -> None:
        """ContractViolation is an Exception subclass."""
        from elspeth.contracts.errors import ContractViolation

        assert issubclass(ContractViolation, Exception)

    def test_contract_violation_has_message(self) -> None:
        """ContractViolation produces a message from str()."""
        from elspeth.contracts.errors import ContractViolation

        exc = ContractViolation(normalized_name="customer_id", original_name="Customer ID")
        msg = str(exc)
        # Base class message should mention both names
        assert "customer_id" in msg and "Customer ID" in msg


class TestMissingFieldViolation:
    """Tests for MissingFieldViolation exception."""

    def test_missing_field_is_contract_violation(self) -> None:
        """MissingFieldViolation is a ContractViolation subclass."""
        from elspeth.contracts.errors import ContractViolation, MissingFieldViolation

        assert issubclass(MissingFieldViolation, ContractViolation)

    def test_missing_field_message_contains_required(self) -> None:
        """MissingFieldViolation message mentions 'required' or 'missing'."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="amount", original_name="Amount")
        msg = str(exc).lower()
        assert "required" in msg or "missing" in msg

    def test_missing_field_message_shows_original_normalized_format(self) -> None:
        """MissingFieldViolation message shows 'original' (normalized) format."""
        from elspeth.contracts.errors import MissingFieldViolation

        exc = MissingFieldViolation(normalized_name="customer_id", original_name="Customer ID")
        msg = str(exc)
        # Should show 'Customer ID' (customer_id) or similar format
        assert "Customer ID" in msg
        assert "customer_id" in msg


class TestTypeMismatchViolation:
    """Tests for TypeMismatchViolation exception."""

    def test_type_mismatch_is_contract_violation(self) -> None:
        """TypeMismatchViolation is a ContractViolation subclass."""
        from elspeth.contracts.errors import ContractViolation, TypeMismatchViolation

        assert issubclass(TypeMismatchViolation, ContractViolation)

    def test_type_mismatch_message_shows_types(self) -> None:
        """TypeMismatchViolation message shows expected and actual types."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )
        msg = str(exc)
        assert "int" in msg
        assert "str" in msg

    def test_type_mismatch_message_shows_original_normalized_format(self) -> None:
        """TypeMismatchViolation message shows 'original' (normalized) format."""
        from elspeth.contracts.errors import TypeMismatchViolation

        exc = TypeMismatchViolation(
            normalized_name="customer_id",
            original_name="Customer ID",
            expected_type=int,
            actual_type=str,
            actual_value="abc",
        )
        msg = str(exc)
        assert "Customer ID" in msg
        assert "customer_id" in msg


class TestExtraFieldViolation:
    """Tests for ExtraFieldViolation exception."""

    def test_extra_field_is_contract_violation(self) -> None:
        """ExtraFieldViolation is a ContractViolation subclass."""
        from elspeth.contracts.errors import ContractViolation, ExtraFieldViolation

        assert issubclass(ExtraFieldViolation, ContractViolation)

    def test_extra_field_message_mentions_field(self) -> None:
        """ExtraFieldViolation message mentions the field name."""
        from elspeth.contracts.errors import ExtraFieldViolation

        exc = ExtraFieldViolation(normalized_name="unknown_field", original_name="Unknown Field")
        msg = str(exc)
        assert "Unknown Field" in msg or "unknown_field" in msg

    def test_extra_field_message_mentions_fixed_mode(self) -> None:
        """ExtraFieldViolation message mentions FIXED mode."""
        from elspeth.contracts.errors import ExtraFieldViolation

        exc = ExtraFieldViolation(normalized_name="extra_col", original_name="Extra Col")
        msg = str(exc).upper()
        assert "FIXED" in msg


class TestContractMergeError:
    """Tests for ContractMergeError exception."""

    def test_contract_merge_error_is_value_error(self) -> None:
        """ContractMergeError is a ValueError subclass."""
        from elspeth.contracts.errors import ContractMergeError

        assert issubclass(ContractMergeError, ValueError)

    def test_contract_merge_error_message_shows_field(self) -> None:
        """ContractMergeError message shows the field name."""
        from elspeth.contracts.errors import ContractMergeError

        exc = ContractMergeError(field="amount", type_a="int", type_b="str")
        msg = str(exc)
        assert "amount" in msg

    def test_contract_merge_error_message_shows_conflicting_types(self) -> None:
        """ContractMergeError message shows both conflicting types."""
        from elspeth.contracts.errors import ContractMergeError

        exc = ContractMergeError(field="score", type_a="float", type_b="int")
        msg = str(exc)
        assert "float" in msg
        assert "int" in msg
