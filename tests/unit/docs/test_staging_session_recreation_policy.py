"""Current pre-release session-store cutover documentation contract."""

from pathlib import Path


def test_current_cutover_requires_epoch_36_cleanup_reason_and_forbids_downgrade_repair() -> None:
    runbook = Path("docs/runbooks/staging-session-db-recreation.md").read_text(encoding="utf-8")
    current_cutover = runbook.split("## Current Cutover:", maxsplit=1)[1].split("## Historical Cutover:", maxsplit=1)[0]
    normalized = " ".join(runbook.split())

    assert "0.7.2 blob deletion cleanup" in current_cutover
    assert "session epoch 36" in current_cutover
    assert "0.7.2 advances `SESSION_SCHEMA_EPOCH` from 35 to 36" in current_cutover
    assert "0.7.1 advances the session store from epoch 26 through epoch 35" in current_cutover
    assert "blob-deletion" in current_cutover
    assert "tombstone unlink or directory fsync fails remains retryable" in current_cutover
    assert "exclusive guided-confirmation proposal admission" in current_cutover
    assert "quota_exceeded" in current_cutover
    assert "stable HTTP 413" in current_cutover
    assert "restore the epoch-29 database" not in current_cutover.lower()
    assert "downgrade to epoch 29" not in current_cutover.lower()
    assert "Do not restore predecessor source or databases as the repair path." in normalized
