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
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from elspeth.contracts import SinkProtocol
    from elspeth.contracts.audit import Run, SinkEffect
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
    from elspeth.contracts.errors import AuditIntegrityError
    from elspeth.contracts.schema import SchemaConfig

    # Snapshot first: export audit rows can never recurse into their own bytes.
    factory = RecorderFactory(db, payload_store=payload_store)
    # Export-node registration is idempotent (elspeth-08350558e6): the node id
    # is deterministic (``export:<sink>``), so a retry after a failed or lost
    # publication response finds the row a prior attempt registered. Reuse it
    # so the retry reaches SinkEffectCoordinator reconciliation of the durable
    # effect instead of crashing on the nodes composite primary key. Fail
    # closed if the registered identity is not the audit-export sink node this
    # attempt would register.
    existing_node = factory.data_flow.get_node(sink.node_id, run_id)
    if existing_node is None:
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
    elif existing_node.node_type is not NodeType.SINK or existing_node.plugin_name != sink.name:
        raise AuditIntegrityError(
            f"audit export node {sink.node_id!r} for run {run_id!r} is already registered "
            f"with divergent identity ({existing_node.node_type.value!r}/{existing_node.plugin_name!r}); "
            f"refusing to reuse it for export sink {sink.name!r}"
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


def audit_export_resume_refusal(run: object | None, run_id: str) -> str | None:
    """Return why ``run`` cannot have its audit export resumed, or None if it can.

    Fail-closed eligibility gate shared by :func:`resume_audit_export` and its
    production drivers (elspeth-8fd1f415b9): resume applies only to runs that
    are immutable export-terminal and whose export has not already completed.
    """
    from elspeth.contracts import ExportStatus
    from elspeth.core.landscape.export_read_model import _EXPORT_TERMINAL

    if run is None:
        return f"run {run_id!r} not found in the audit database"
    status = run.status  # type: ignore[attr-defined]
    if status not in _EXPORT_TERMINAL:
        return f"run {run_id!r} has status {status.value!r}, which is not export-terminal; audit export resume requires a finalized run"
    if run.export_status is ExportStatus.COMPLETED:  # type: ignore[attr-defined]
        return f"run {run_id!r} audit export already completed; refusing to re-run publication"
    return None


def _audit_export_resume_target_refusal(
    run: Run,
    settings: ElspethSettings,
    effects: Sequence[SinkEffect],
) -> str | None:
    """Refuse a resume whose settings diverge from its durable target identity."""
    from elspeth.contracts.sink_effects import SinkEffectInputKind
    from elspeth.core.landscape.execution.sink_effect_identity import compute_sink_effect_target_hash

    export_config = settings.landscape.export
    if run.export_format is not None and run.export_format != export_config.format:
        return "audit export target identity differs from the persisted export format"
    if run.export_sink is not None and run.export_sink != export_config.sink:
        return "audit export target identity differs from the persisted export sink"
    sink_name = export_config.sink
    if sink_name is None:
        return None
    target_hash = compute_sink_effect_target_hash(dict(settings.sinks[sink_name].options))
    audit_effects = tuple(effect for effect in effects if effect.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT)
    if not audit_effects:
        return None
    if len(audit_effects) != 1:
        return "audit export target identity is ambiguous because multiple durable export effects exist for the run"
    effect = audit_effects[0]
    if effect.sink_node_id != f"export:{sink_name}" or effect.config_hash != target_hash:
        return "audit export target identity differs from the existing durable effect"
    return None


def resume_audit_export(
    db: LandscapeDB,
    run_id: str,
    settings: ElspethSettings,
    sink_factory: Callable[[str], SinkEffectRuntimeBinding],
    *,
    payload_store: PayloadStore,
    audit_export_content_store: AuditExportContentStore,
    audit_export_content_store_resolver: AuditExportContentStoreResolver,
    worker_id: str,
) -> None:
    """Resume audit-export recovery for a finalized run (elspeth-8fd1f415b9).

    Production driver for the window after run finalization where a crash or
    transient target failure left the run's export PENDING/FAILED (or unset)
    and its durable sink effect PREPARED/IN_FLIGHT. Re-drives the export
    pipeline: the immutable snapshot winner is reused (never re-derived), the
    deterministic export node registration is idempotent, and
    SinkEffectCoordinator reconciles the durable effect so publication happens
    exactly once.

    Mirrors ``RunLifecycleCoordinator.execute_export_phase`` status semantics:
    export status transitions PENDING -> COMPLETED on success and
    PENDING -> FAILED (with the error recorded) on failure.

    Raises:
        ValueError: If export is not enabled, the run does not exist, the run
            is not export-terminal, or its export already completed.
        Exception: Re-raises any export failure after recording FAILED status.
    """
    from elspeth.contracts import ExportStatus
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.engine._best_effort import best_effort

    export_config = settings.landscape.export
    if not export_config.enabled:
        raise ValueError("audit export is not enabled in settings; nothing to resume")

    factory = RecorderFactory(db, payload_store=payload_store)
    run = factory.run_lifecycle.get_run(run_id)
    refusal = audit_export_resume_refusal(run, run_id)
    if refusal is not None:
        raise ValueError(refusal)
    assert run is not None
    refusal = _audit_export_resume_target_refusal(
        run,
        settings,
        factory.execution.sink_effects.get_effects_for_run(run_id),
    )
    if refusal is not None:
        raise ValueError(refusal)

    factory.run_lifecycle.set_export_status(
        run_id,
        status=ExportStatus.PENDING,
        export_format=export_config.format,
        export_sink=export_config.sink,
    )
    try:
        export_landscape(
            db,
            run_id,
            settings,
            sink_factory,
            payload_store=payload_store,
            audit_export_content_store=audit_export_content_store,
            audit_export_content_store_resolver=audit_export_content_store_resolver,
            worker_id=worker_id,
        )
    except Exception as export_error:
        with best_effort(
            "Export status FAILED recording on resume",
            run_id=run_id,
            original_error=type(export_error).__name__,
        ):
            factory.run_lifecycle.set_export_status(
                run_id,
                status=ExportStatus.FAILED,
                error=str(export_error),
            )
        raise
    factory.run_lifecycle.set_export_status(run_id, status=ExportStatus.COMPLETED)
