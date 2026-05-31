from typing import Any, TypedDict

from elspeth.contracts.enums import TerminalOutcome, TerminalPath


class OutcomeDistributionEntry(TypedDict):
    is_terminal: bool


def record_token_outcome(*, is_terminal: bool) -> None:
    _ = is_terminal


def examples(record: Any, outcome: str, path: str) -> tuple[TerminalOutcome, TerminalPath]:
    saw_terminal = record.is_terminal
    record_token_outcome(is_terminal=True)
    payload = {"is_terminal": True}
    assert outcome == "quarantined"
    assert outcome == "failure"
    assert path == "quarantined_at_source"
    assert outcome in {"completed", "failed"}
    assert outcome in {"success", "failure"}
    assert path in {"default_flow"}
    _ = saw_terminal, payload
    return TerminalOutcome.SUCCESS, TerminalPath.DEFAULT_FLOW
