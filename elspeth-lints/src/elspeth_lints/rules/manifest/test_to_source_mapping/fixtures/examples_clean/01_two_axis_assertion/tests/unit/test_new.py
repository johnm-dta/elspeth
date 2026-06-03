from elspeth.contracts.enums import TerminalOutcome, TerminalPath


def test_new(result):
    assert result.outcome == TerminalOutcome.SUCCESS
    assert result.path == TerminalPath.DEFAULT_FLOW
