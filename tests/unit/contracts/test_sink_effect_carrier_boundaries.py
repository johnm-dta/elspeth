"""Contract ownership and legacy import compatibility for sink-effect carriers."""

import elspeth.contracts as contracts
from elspeth.contracts.audit_export import (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotWinner,
    AuditExportTerminalWitness,
)
from elspeth.contracts.sink_effects import (
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
    SinkEffectFinalizationMember,
    SinkEffectFinalizationResult,
    SinkEffectFinalizeRequest,
    SinkEffectLease,
    SinkEffectReservationRequest,
)
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotCandidate as LegacyAuditExportSnapshotCandidate,
)
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotReadLimits as LegacyAuditExportSnapshotReadLimits,
)
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotRegistryKey as LegacyAuditExportSnapshotRegistryKey,
)
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotWinner as LegacyAuditExportSnapshotWinner,
)
from elspeth.core.landscape.execution.sink_effect_finalization import (
    SinkEffectFinalizationMember as LegacySinkEffectFinalizationMember,
)
from elspeth.core.landscape.execution.sink_effect_finalization import (
    SinkEffectFinalizationResult as LegacySinkEffectFinalizationResult,
)
from elspeth.core.landscape.execution.sink_effect_finalization import (
    SinkEffectFinalizeRequest as LegacySinkEffectFinalizeRequest,
)
from elspeth.core.landscape.execution.sink_effect_lifecycle import (
    SinkEffectAttemptRequest as LegacySinkEffectAttemptRequest,
)
from elspeth.core.landscape.execution.sink_effect_lifecycle import (
    SinkEffectAttemptResult as LegacySinkEffectAttemptResult,
)
from elspeth.core.landscape.execution.sink_effect_lifecycle import (
    SinkEffectLease as LegacySinkEffectLease,
)
from elspeth.core.landscape.execution.sink_effect_reservation import (
    SinkEffectReservationRequest as LegacySinkEffectReservationRequest,
)
from elspeth.core.landscape.export_read_model import AuditExportTerminalWitness as LegacyAuditExportTerminalWitness

CONTRACT_CARRIERS = (
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotWinner,
    AuditExportTerminalWitness,
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
    SinkEffectFinalizationMember,
    SinkEffectFinalizationResult,
    SinkEffectFinalizeRequest,
    SinkEffectLease,
    SinkEffectReservationRequest,
)


def test_sink_effect_carriers_are_contract_owned_and_public() -> None:
    for carrier in CONTRACT_CARRIERS:
        assert carrier.__module__.startswith("elspeth.contracts.")
        assert getattr(contracts, carrier.__name__) is carrier
        assert carrier.__name__ in contracts.__all__


def test_core_carrier_imports_remain_identity_preserving_re_exports() -> None:
    assert LegacyAuditExportSnapshotCandidate is AuditExportSnapshotCandidate
    assert LegacyAuditExportSnapshotReadLimits is AuditExportSnapshotReadLimits
    assert LegacyAuditExportSnapshotRegistryKey is AuditExportSnapshotRegistryKey
    assert LegacyAuditExportSnapshotWinner is AuditExportSnapshotWinner
    assert LegacyAuditExportTerminalWitness is AuditExportTerminalWitness
    assert LegacySinkEffectAttemptRequest is SinkEffectAttemptRequest
    assert LegacySinkEffectAttemptResult is SinkEffectAttemptResult
    assert LegacySinkEffectFinalizationMember is SinkEffectFinalizationMember
    assert LegacySinkEffectFinalizationResult is SinkEffectFinalizationResult
    assert LegacySinkEffectFinalizeRequest is SinkEffectFinalizeRequest
    assert LegacySinkEffectLease is SinkEffectLease
    assert LegacySinkEffectReservationRequest is SinkEffectReservationRequest
