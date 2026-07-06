"""Unit tests for resume audit snapshot boundary immutability."""

from __future__ import annotations

from collections.abc import MutableMapping
from types import MappingProxyType
from typing import Any, cast
from unittest.mock import Mock

import pytest

from elspeth.contracts.schema_contract import SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.checkpoint.recovery import RecoveryManager
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator.resume import _ResumeAuditSnapshot


def test_resume_audit_snapshot_freezes_container_fields() -> None:
    """Caller-owned audit maps must not remain mutable through the snapshot.

    The snapshot carries only topology-stable reconstruction (per-source
    schema/contract/lifecycle maps); the resume work set is deliberately NOT a
    field here — it is race-sensitive and is recomputed AFTER the seat CAS in
    ``reconstruct_resume_state``. So this pins the freeze contract for exactly
    the fields the read-only snapshot still owns.
    """
    source_id = NodeID("source-1")
    source_id_after_creation = NodeID("source-2")
    contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)

    schema_contracts_by_source = {source_id: contract}
    source_names_by_source = {source_id: "source"}
    source_lifecycle_by_source = {source_id: "exhausted"}
    source_schema_classes: dict[NodeID, type[Any]] = {source_id: dict}

    snapshot = _ResumeAuditSnapshot(
        factory=Mock(spec=RecorderFactory),
        recovery=Mock(spec=RecoveryManager),
        run_id="run-1",
        worker_id="worker-1",
        schema_contracts_by_source=schema_contracts_by_source,
        source_names_by_source=source_names_by_source,
        source_lifecycle_by_source=source_lifecycle_by_source,
        source_schema_classes=source_schema_classes,
    )

    # Mutating the caller-owned dicts after construction must not leak in.
    schema_contracts_by_source[source_id_after_creation] = contract
    source_names_by_source[source_id_after_creation] = "source-2"
    source_lifecycle_by_source[source_id_after_creation] = "loaded"
    source_schema_classes[source_id_after_creation] = list

    for frozen_map in (
        snapshot.schema_contracts_by_source,
        snapshot.source_names_by_source,
        snapshot.source_lifecycle_by_source,
        snapshot.source_schema_classes,
    ):
        assert isinstance(frozen_map, MappingProxyType)
        assert source_id_after_creation not in frozen_map

    with pytest.raises(TypeError):
        cast(MutableMapping[NodeID, str], snapshot.source_names_by_source)[source_id_after_creation] = "nope"
