"""Post-run audit export functions.

This module handles:
- Exporting the Landscape audit trail to JSON or CSV format after run completion

The export functions support two modes:
- JSON format: All records written to a single sink (handles heterogeneous records)
- CSV format: Separate files per record type (CSV requires homogeneous schemas)

Resume schema reconstruction lives in schema_reconstruction.py. This module
keeps compatibility re-exports for older import paths.

IMPORTANT: Import Cycle Prevention
----------------------------------
This module uses TYPE_CHECKING for imports that would cause cycles.
LandscapeDB is typed but imported conditionally to avoid circular imports.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol
    from elspeth.contracts.audit_export import AuditExportContentStore, AuditExportContentStoreResolver
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.contracts.sink_effects import SinkEffectRuntimeBinding
    from elspeth.core.config import ElspethSettings
    from elspeth.core.landscape import LandscapeDB

from elspeth.contracts import Determinism
from elspeth.engine.orchestrator.schema_reconstruction import (
    _create_schema_model as _create_schema_model,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    _json_schema_to_python_type as _json_schema_to_python_type,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    _model_name_for_field as _model_name_for_field,
)
from elspeth.engine.orchestrator.schema_reconstruction import (
    reconstruct_schema_from_json as reconstruct_schema_from_json,
)


def prepare_audit_export_binding(
    settings: ElspethSettings,
    sink_factory: Callable[[str], SinkEffectRuntimeBinding],
) -> tuple[SinkEffectRuntimeBinding, object]:
    """Construct and admit the exact delayed sink before export mutations."""
    from elspeth.contracts.sink_effects import AuditExportFormat, SinkEffectInputKind
    from elspeth.engine.orchestrator.preflight import (
        validate_audit_export_sink_type_capability,
        validate_pipeline_sink_effect_capabilities,
    )

    sink_name = settings.landscape.export.sink
    if sink_name is None:
        raise ValueError("Export sink name is None")
    binding = sink_factory(sink_name)
    sink_name, sink, modes = _validate_audit_export_binding_provenance(settings, binding)
    admission = validate_pipeline_sink_effect_capabilities(
        {sink_name: sink},
        configured_modes=modes,
        required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
    )
    export_format = AuditExportFormat(settings.landscape.export.format)
    validate_audit_export_sink_type_capability(type(sink), export_format)
    return binding, admission


def _probe_audit_export_publication(
    settings: ElspethSettings,
    *,
    sink_name: str,
    sink: SinkProtocol,
) -> None:
    """Run the sole bounded non-declarative export probe before snapshot I/O."""
    from pathlib import Path

    from elspeth.contracts.errors import SinkEffectCapabilityError
    from elspeth.contracts.sink_effects import AuditExportFormat
    from elspeth.engine.orchestrator.preflight import validate_audit_export_sink_type_capability

    export_format = AuditExportFormat(settings.landscape.export.format)
    validate_audit_export_sink_type_capability(type(sink), export_format)
    if export_format is not AuditExportFormat.CSV:
        return
    raw_path = settings.sinks[sink_name].options.get("path")
    if type(raw_path) is not str or not raw_path.strip():
        raise SinkEffectCapabilityError("CSV audit export requires an explicit local bundle target path")
    from elspeth.plugins.sinks._audit_export_bundle_effects import preflight_audit_export_bundle

    preflight_audit_export_bundle(Path(raw_path))


def _validate_audit_export_binding_provenance(
    settings: ElspethSettings,
    binding: SinkEffectRuntimeBinding,
) -> tuple[str, SinkProtocol, Mapping[str, str]]:
    """Bind one delayed-export runtime binding to its exact settings authority."""
    from elspeth.engine.orchestrator.preflight import (
        SinkEffectExecutionPurpose,
        SinkEffectRuntimeBinding,
        sink_effect_modes_from_runtime_bindings,
    )

    sink_name = settings.landscape.export.sink
    if sink_name is None:
        raise ValueError("Export sink name is None")
    if type(binding) is not SinkEffectRuntimeBinding:
        raise TypeError("Audit export sink factory must return an exact SinkEffectRuntimeBinding")
    sink = cast("SinkProtocol", binding.sink)
    modes = sink_effect_modes_from_runtime_bindings(
        {sink_name: sink},
        {sink_name: binding},
        purpose=SinkEffectExecutionPurpose.AUDIT_EXPORT,
        configured_options={sink_name: settings.sinks[sink_name].options},
    )
    return sink_name, sink, modes


def export_landscape(
    db: LandscapeDB,
    run_id: str,
    settings: ElspethSettings,
    sink_factory: Callable[[str], SinkEffectRuntimeBinding],
    *,
    payload_store: PayloadStore,
    audit_export_content_store: AuditExportContentStore,
    audit_export_content_store_resolver: AuditExportContentStoreResolver,
    worker_id: str,
    prepared_binding: SinkEffectRuntimeBinding | None = None,
    sink_effect_admission: object | None = None,
) -> None:
    """Export audit trail to configured sink after run completion.

    For JSON format: writes all records to a single sink (records are
    heterogeneous but JSON handles that naturally).

    For CSV format: writes separate files per record_type to a directory,
    since CSV requires homogeneous schemas per file.

    Args:
        db: LandscapeDB instance for reading audit data
        run_id: The completed run ID
        settings: Full settings containing export configuration
        sink_factory: Creates a fresh, unstarted sink instance by name.
            The export path needs its own sink instance because the
            pipeline's sinks have already completed their lifecycle.

    Raises:
        ValueError: If signing requested but ELSPETH_SIGNING_KEY not set,
                   or if sink_factory raises for the configured sink name
    """
    from elspeth.contracts.audit_export import AuditExportContentStore, AuditExportContentStoreResolver
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine.orchestrator.audit_export_effects import execute_audit_export_effect, prepare_audit_export_snapshot

    export_config = settings.landscape.export

    if type(worker_id) is not str or not worker_id.strip():
        raise ValueError("audit export worker_id must be a non-empty exact string")

    if not isinstance(audit_export_content_store, AuditExportContentStore):
        raise TypeError("audit_export_content_store must implement AuditExportContentStore")
    if not audit_export_content_store.is_durable():
        raise ValueError("audit_export_content_store must prove durability")
    if type(audit_export_content_store_resolver) is not AuditExportContentStoreResolver:
        raise TypeError("audit_export_content_store_resolver must be exact AuditExportContentStoreResolver")
    audit_export_content_store_resolver.register(audit_export_content_store)
    configured_store = export_config.content_store
    if configured_store is None:
        raise ValueError("audit export requires an explicit durable content_store policy")
    if audit_export_content_store.content_store_id != configured_store.content_store_id:
        raise ValueError("resolved audit_export_content_store ID does not match configured content_store_id")
    if audit_export_content_store.namespace != configured_store.namespace:
        raise ValueError("resolved audit_export_content_store namespace does not match configured namespace")

    from elspeth.contracts.sink_effects import SinkEffectInputKind
    from elspeth.engine.orchestrator.preflight import require_sink_effect_admission

    if prepared_binding is None:
        prepared_binding, sink_effect_admission = prepare_audit_export_binding(settings, sink_factory)
    sink_name, sink, modes = _validate_audit_export_binding_provenance(settings, prepared_binding)
    require_sink_effect_admission(
        {sink_name: sink},
        configured_modes=modes,
        required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        admission=sink_effect_admission,
    )
    _probe_audit_export_publication(settings, sink_name=sink_name, sink=sink)

    # Get signing key from environment if signing enabled
    signing_key: bytes | None = None
    if export_config.sign:
        signing_secret_ref = export_config.signing_secret_ref
        if signing_secret_ref is None:
            raise ValueError("hmac_sha256 export requires an explicit signing_secret_ref")
        try:
            key_str = os.environ[signing_secret_ref]
        except KeyError as exc:
            raise ValueError(f"{signing_secret_ref} environment variable required for signed export") from exc
        if not key_str.strip():
            raise ValueError(f"{signing_secret_ref} environment variable required for signed export")
        signing_key = key_str.encode("utf-8")

    snapshot = prepare_audit_export_snapshot(
        db,
        run_id=run_id,
        config=export_config,
        signing_key=signing_key,
        content_store=audit_export_content_store,
        content_store_resolver=audit_export_content_store_resolver,
    )

    sink.node_id = f"export:{sink_name}"

    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig

    # Snapshot first: export audit rows can never recurse into their own bytes.
    factory = RecorderFactory(db, payload_store=payload_store)
    factory.data_flow.register_node(
        run_id=run_id,
        node_id=sink.node_id,
        plugin_name=sink.name,
        node_type=NodeType.SINK,
        plugin_version=sink.plugin_version,
        config=dict(sink.config),
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        determinism=Determinism.IO_WRITE,
        source_file_hash=sink.source_file_hash,
    )
    try:
        execute_audit_export_effect(
            factory=factory,
            snapshot=snapshot,
            sink=sink,
            sink_node_id=sink.node_id,
            target_config=dict(settings.sinks[sink_name].options),
            worker_id=worker_id,
        )
    finally:
        sink.close()
