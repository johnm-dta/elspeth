"""Unit tests for resume audit snapshot boundary immutability."""

from __future__ import annotations

from types import MappingProxyType
from unittest.mock import Mock

import pytest

from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.checkpoint.recovery import IncompleteTokenSpec
from elspeth.engine.orchestrator.resume import _ResumeAuditSnapshot


def _incomplete_token(token_id: str = "token-1", row_id: str = "row-1") -> IncompleteTokenSpec:
    return IncompleteTokenSpec(
        token_id=token_id,
        row_id=row_id,
        branch_name=None,
        fork_group_id=None,
        join_group_id=None,
        expand_group_id=None,
        token_data_ref=None,
        step_in_pipeline=1,
        max_attempt=0,
    )


def test_resume_audit_snapshot_deep_freezes_container_fields() -> None:
    """Caller-owned audit maps must not remain mutable through the snapshot."""
    source_id = NodeID("source-1")
    source_id_after_creation = NodeID("source-2")
    token = _incomplete_token()
    contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)

    incomplete_by_row = {"row-1": [token]}
    schema_contracts_by_source = {source_id: contract}
    source_names_by_source = {source_id: "source"}
    source_lifecycle_by_source = {source_id: "exhausted"}
    source_schema_classes = {source_id: dict}

    snapshot = _ResumeAuditSnapshot(
        factory=Mock(),
        recovery=Mock(),
        run_id="run-1",
        worker_id="worker-1",
        incomplete_by_row=incomplete_by_row,
        schema_contracts_by_source=schema_contracts_by_source,
        source_names_by_source=source_names_by_source,
        source_lifecycle_by_source=source_lifecycle_by_source,
        source_schema_classes=source_schema_classes,
    )

    incomplete_by_row["row-1"].append(_incomplete_token("token-2"))
    incomplete_by_row["row-2"] = [_incomplete_token("token-3", "row-2")]
    schema_contracts_by_source[source_id_after_creation] = contract
    source_names_by_source[source_id_after_creation] = "source-2"
    source_lifecycle_by_source[source_id_after_creation] = "loaded"
    source_schema_classes[source_id_after_creation] = list

    assert isinstance(snapshot.incomplete_by_row, MappingProxyType)
    assert isinstance(snapshot.incomplete_by_row["row-1"], tuple)
    assert snapshot.incomplete_by_row["row-1"] == (token,)
    assert "row-2" not in snapshot.incomplete_by_row

    for frozen_map in (
        snapshot.schema_contracts_by_source,
        snapshot.source_names_by_source,
        snapshot.source_lifecycle_by_source,
        snapshot.source_schema_classes,
    ):
        assert isinstance(frozen_map, MappingProxyType)
        assert source_id_after_creation not in frozen_map

    with pytest.raises(TypeError):
        snapshot.incomplete_by_row["row-2"] = ()
