from elspeth.contracts.enums import RowOutcome


def test_old(result):
    assert result.outcome == RowOutcome.COMPLETED
