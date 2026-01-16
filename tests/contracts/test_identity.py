"""Tests for identity contracts."""


class TestTokenInfo:
    """Tests for TokenInfo."""

    def test_create_token_info(self) -> None:
        """Can create TokenInfo with required fields."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(
            row_id="row-123",
            token_id="tok-456",
            row_data={"field": "value"},
        )

        assert token.row_id == "row-123"
        assert token.token_id == "tok-456"
        assert token.row_data == {"field": "value"}
        assert token.branch_name is None

    def test_token_info_with_branch(self) -> None:
        """Can create TokenInfo with branch_name."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(
            row_id="row-123",
            token_id="tok-456",
            row_data={},
            branch_name="sentiment",
        )

        assert token.branch_name == "sentiment"

    def test_token_info_row_data_mutable(self) -> None:
        """TokenInfo.row_data can be modified (not frozen)."""
        from elspeth.contracts import TokenInfo

        token = TokenInfo(row_id="r", token_id="t", row_data={"a": 1})
        token.row_data["b"] = 2

        assert token.row_data == {"a": 1, "b": 2}
