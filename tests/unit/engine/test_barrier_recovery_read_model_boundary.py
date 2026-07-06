from __future__ import annotations

import ast
from pathlib import Path

RESTORE_READ_METHODS = {
    "find_duplicate_live_buffered_outcomes",
    "get_failed_unrouted_terminal_token_ids",
    "get_live_buffered_outcomes",
}

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_barrier_recovery_uses_restore_read_model_for_token_outcome_policy() -> None:
    source_path = REPO_ROOT / "src/elspeth/engine/barrier_coordination.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    recovery = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "BarrierRecoveryCoordinator"
    )

    data_flow_restore_reads: list[str] = []
    for node in ast.walk(recovery):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in RESTORE_READ_METHODS:
            continue
        receiver = func.value
        if isinstance(receiver, ast.Attribute) and receiver.attr == "_data_flow":
            data_flow_restore_reads.append(func.attr)

    assert data_flow_restore_reads == []
