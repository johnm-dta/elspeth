"""Regression guards for timeout semantics contract documentation."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_timeout_docs_describe_source_idle_polling_consistently() -> None:
    """Aggregation/coalesce docs must match the orchestrator idle polling contract."""
    system_operations = (PROJECT_ROOT / "docs/contracts/system-operations.md").read_text()
    release_guarantees = (PROJECT_ROOT / "docs/release/guarantees.md").read_text()

    assert (
        "| `timeout_seconds` | Duration elapsed since batch start | Checked before each row and during source-idle polling |"
        in system_operations
    )
    assert "from the source-idle polling path when the pipeline also has time-sensitive\naggregation triggers" in system_operations
    assert "Coalesce-only streaming pipelines without aggregation idle\npolling still need source-level heartbeat rows" in system_operations
    assert "During completely idle periods with no data flowing, timeouts cannot fire" not in system_operations

    assert "The same idle-polling pass also checks coalesce\ntimeouts in mixed aggregation/coalesce pipelines" in release_guarantees
    assert "Coalesce-only streaming\npipelines still depend on token arrival, source completion, or heartbeat rows" in release_guarantees
