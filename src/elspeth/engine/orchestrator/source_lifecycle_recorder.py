"""SourceLifecycleRecorder: record source lifecycle / field-resolution evidence.

Extracted from ``SourceIterationDriver`` (filigree elspeth-27d7bfc14b). These
two methods are the driver's "write source metadata to Landscape" concern:
the field-resolution mapping (provisional on first row, authoritative at EOF —
elspeth-fb108a77c9) and the run_source lifecycle record with the latest schema
evidence. Both take ``factory`` + ``active_source`` explicitly and hold no
cross-method state beyond the ``RunCeremony`` used for telemetry, so they move
to a focused collaborator the driver delegates to.

Deliberately left on the driver: ``restore_source_iteration_context`` (two-line
node/operation identity restoration threaded through the hot loop and the
finalizer) and the initial ``lifecycle_state="loading"`` record inside
``run_main_processing_loop`` setup — moving either would only churn hot-path
call sites without narrowing this recorder's single concern.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from elspeth.contracts.events import FieldResolutionApplied
from elspeth.contracts.types import NodeID
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import RunSourceLifecycleState
from elspeth.engine.orchestrator.ceremony import RunCeremony

if TYPE_CHECKING:
    from collections.abc import Mapping

    from elspeth.contracts import SourceProtocol


class SourceLifecycleRecorder:
    """Record source field-resolution and run_source lifecycle evidence.

    Holds only the ``RunCeremony`` used to emit ``FieldResolutionApplied``
    telemetry; all per-source state arrives through method arguments.
    """

    def __init__(self, *, ceremony: RunCeremony) -> None:
        self._ceremony = ceremony

    def record_field_resolution(
        self,
        factory: RecorderFactory,
        run_id: str,
        *,
        active_source: SourceProtocol,
        previously_recorded: tuple[Mapping[str, str], str | None] | None = None,
    ) -> tuple[Mapping[str, str], str | None] | None:
        """Record the source field-resolution mapping if available.

        Called on first iteration (provisional — after the generator body
        executes) and again from ``finalize_source_iteration`` (authoritative —
        sparse sources can extend the mapping on later rows, and the run-level
        column is an overwrite UPDATE; elspeth-fb108a77c9). Empty sources
        (header-only files where the loop never runs) record once at finalize.

        ``previously_recorded`` is the snapshot returned by the earlier call;
        when the source's current resolution is identical, the write and its
        ``FieldResolutionApplied`` telemetry are skipped, so an unchanged
        mapping (fixed-header sources — the common case) emits exactly one
        event and a grown union emits a provisional then an authoritative one.

        Returns:
            The recorded (mapping, normalization_version) snapshot, or None if
            the source has no field resolution.
        """
        field_resolution = active_source.get_field_resolution()
        if field_resolution is None:
            return None
        if previously_recorded is not None and field_resolution == previously_recorded:
            return previously_recorded

        resolution_mapping, normalization_version = field_resolution
        factory.run_lifecycle.record_source_field_resolution(
            run_id=run_id,
            resolution_mapping=resolution_mapping,
            normalization_version=normalization_version,
        )
        # Emit telemetry AFTER Landscape succeeds
        self._ceremony.emit_telemetry(
            FieldResolutionApplied(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                source_plugin=active_source.name,
                field_count=len(resolution_mapping),
                normalization_version=normalization_version,
                resolution_mapping=resolution_mapping,
            )
        )
        return field_resolution

    def record_run_source_lifecycle(
        self,
        factory: RecorderFactory,
        run_id: str,
        source_id: NodeID,
        source_name: str,
        active_source: SourceProtocol,
        lifecycle_state: RunSourceLifecycleState,
    ) -> None:
        """Record source lifecycle with the latest source schema evidence."""

        field_resolution = active_source.get_field_resolution()
        resolution_mapping: Mapping[str, str] | None = None
        normalization_version: str | None = None
        if field_resolution is not None:
            resolution_mapping, normalization_version = field_resolution

        factory.run_lifecycle.record_run_source(
            run_id=run_id,
            source_node_id=source_id,
            source_name=source_name,
            plugin_name=active_source.name,
            config_hash=stable_hash(active_source.config),
            source_schema_json=json.dumps(active_source.output_schema.model_json_schema()),
            schema_contract=active_source.get_schema_contract(),
            field_resolution_mapping=resolution_mapping,
            normalization_version=normalization_version,
            lifecycle_state=lifecycle_state,
        )
