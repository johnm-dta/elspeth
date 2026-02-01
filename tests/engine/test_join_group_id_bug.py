"""Minimal failing test for join_group_id propagation bug.

This test will FAIL until TokenInfo has join_group_id field and
TokenManager propagates it from Token to TokenInfo.

Per TDD: Write test first, watch it fail, then fix.
"""


def test_token_info_has_join_group_id_field() -> None:
    """TokenInfo must have join_group_id field to propagate canonical value."""
    from elspeth.contracts.identity import TokenInfo

    # This will FAIL because TokenInfo currently has no join_group_id field
    token = TokenInfo(
        row_id="test_row",
        token_id="test_token",
        row_data={},
        join_group_id="canonical_id_from_recorder",
    )

    assert token.join_group_id == "canonical_id_from_recorder"
