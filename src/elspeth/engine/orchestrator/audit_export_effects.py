"""Durable snapshot materialization and recoverable audit-export effects."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Callable, Mapping
from datetime import UTC, timedelta
from pathlib import Path
from tempfile import TemporaryFile
from typing import Any, BinaryIO, cast
from uuid import uuid4

from elspeth.contracts.audit import AuditExportSnapshot, AuditExportSnapshotChunk
from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_DERIVATION_VERSION,
    AuditExportContentDescriptor,
    AuditExportContentStore,
    AuditExportContentStoreResolver,
    AuditExportDerivationConfig,
    AuditExportSnapshotCandidate,
    AuditExportSnapshotReadLimits,
    AuditExportSnapshotRegistryKey,
    AuditExportSnapshotWinner,
    AuditExportSpooledBundle,
    AuditExportTerminalWitness,
    C,
    ClosedAuditExportJSON,
    derive_audit_export_bundle_to_spool,
    derive_public_export_config_hash,
    derive_registry_key_hash,
)
from elspeth.contracts.sink_effects import (
    AuditExportFormat,
    AuditExportSignedManifestInput,
    AuditExportSigningMode,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectFinalizationResult,
    SinkEffectInputKind,
    SinkEffectReservationRequest,
    SinkEffectRole,
)
from elspeth.core.config import LandscapeExportSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.audit_export_snapshots import (
    AuditExportSnapshotRepository,
)
from elspeth.core.landscape.execution.sink_effect_identity import compute_audit_export_effect_identity
from elspeth.core.landscape.export_read_model import open_export_read_transaction
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionRequest,
    SinkEffectExecutionSeam,
)


def _required_limit(value: int | None, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"enabled audit export requires positive {field_name}")
    return value


def _registry_key(run_id: str, config: LandscapeExportSettings) -> AuditExportSnapshotRegistryKey:
    public_hash = derive_public_export_config_hash(cast(ClosedAuditExportJSON, config.public_snapshot_config()))
    registry_hash = derive_registry_key_hash(
        {
            "export_format": config.format,
            "exporter_version": config.exporter_version,
            "public_export_config_hash": public_hash,
            "serialization_version": config.serialization_version,
            "signer_key_id": config.signer_key_id,
            "signing_mode": config.signing_mode,
            "source_run_id": run_id,
        }
    )
    return AuditExportSnapshotRegistryKey(
        source_run_id=run_id,
        exporter_version=config.exporter_version,
        serialization_version=config.serialization_version,
        export_format=AuditExportFormat(config.format),
        signing_mode=AuditExportSigningMode(config.signing_mode),
        signer_key_id=config.signer_key_id,
        public_export_config_hash=public_hash,
        registry_key_hash=registry_hash,
        chunking_algorithm_version=config.chunking_algorithm_version,
        per_chunk_record_limit=_required_limit(config.per_chunk_record_limit, "per_chunk_record_limit"),
        per_chunk_byte_limit=_required_limit(config.per_chunk_byte_limit, "per_chunk_byte_limit"),
    )


def _derivation_config(
    witness: AuditExportTerminalWitness,
    config: LandscapeExportSettings,
    signing_key: bytes | None,
) -> AuditExportDerivationConfig:
    if config.signing_mode == "hmac_sha256":
        if type(signing_key) is not bytes or not signing_key:
            raise ValueError("hmac_sha256 audit export requires non-empty signing-key bytes")
    elif signing_key is not None:
        raise ValueError("unsigned audit export forbids signing-key bytes")
    completed_at = witness.source_completed_at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return AuditExportDerivationConfig(
        source_run_id=witness.source_run_id,
        source_status=witness.source_status.value,
        source_completed_at=completed_at,
        export_format=config.format,
        exporter_version=config.exporter_version,
        serialization_version=config.serialization_version,
        chunking_algorithm_version=config.chunking_algorithm_version,
        include_raw_error_rows=config.include_raw_error_rows,
        per_chunk_byte_limit=_required_limit(config.per_chunk_byte_limit, "per_chunk_byte_limit"),
        per_chunk_record_limit=_required_limit(config.per_chunk_record_limit, "per_chunk_record_limit"),
        signing_mode=config.signing_mode,
        signer_key_id=config.signer_key_id,
        signing_key=signing_key,
    )


def _private_spool_root(config: LandscapeExportSettings) -> Path:
    root = config.spool_root
    if not isinstance(root, Path):
        raise ValueError("enabled audit export requires an explicit spool_root")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode & 0o077:
        raise ValueError("audit export spool_root must be private")
    return root


def _read_spooled(spool: BinaryIO, offset: int, size: int) -> bytes:
    spool.seek(offset)
    content = spool.read(size)
    if type(content) is not bytes or len(content) != size:
        raise OSError("audit export spool ended before its sealed object boundary")
    return content


def _candidate(
    *,
    bundle: AuditExportSpooledBundle,
    witness: AuditExportTerminalWitness,
    config: LandscapeExportSettings,
    content_store_id: str,
) -> AuditExportSnapshotCandidate:
    snapshot = AuditExportSnapshot(
        snapshot_id=bundle.snapshot_id,
        source_run_id=witness.source_run_id,
        source_status=witness.source_status,
        source_completed_at=witness.source_completed_at,
        exported_at=witness.source_completed_at,
        registry_key_hash=bundle.registry_key_hash,
        exporter_version=config.exporter_version,
        serialization_version=config.serialization_version,
        export_format=AuditExportFormat(config.format),
        signing_mode=AuditExportSigningMode(config.signing_mode),
        signer_key_id=config.signer_key_id,
        derivation_version=AUDIT_EXPORT_DERIVATION_VERSION,
        public_export_config_hash=bundle.public_export_config_hash,
        chunking_algorithm_version=config.chunking_algorithm_version,
        per_chunk_record_limit=_required_limit(config.per_chunk_record_limit, "per_chunk_record_limit"),
        per_chunk_byte_limit=_required_limit(config.per_chunk_byte_limit, "per_chunk_byte_limit"),
        record_count=bundle.record_count,
        total_bytes=bundle.total_bytes,
        chunk_count=len(bundle.chunks),
        terminal_chunk_ordinal=bundle.chunks[-1].ordinal,
        content_store_id=content_store_id,
        manifest_hash=bundle.manifest_hash,
        last_chunk_seal_hash=bundle.last_chunk_seal_hash,
        snapshot_hash=bundle.snapshot_hash,
        snapshot_seal_hash=bundle.snapshot_seal_hash,
        signature_hex=bundle.signed_manifest.signature,
        record_chain_algorithm=bundle.record_chain_algorithm,
        final_hash=bundle.final_hash,
        signed_manifest_schema=bundle.signed_manifest.manifest_schema,
        signed_manifest_hash=bundle.signed_manifest.content_hash,
        signed_manifest_ref=bundle.signed_manifest.content_ref,
        signed_manifest_size_bytes=bundle.signed_manifest.size_bytes,
    )
    chunks = tuple(
        AuditExportSnapshotChunk(
            snapshot_id=bundle.snapshot_id,
            ordinal=chunk.ordinal,
            content_ref=chunk.descriptor.content_ref,
            content_hash=chunk.descriptor.content_hash,
            size_bytes=chunk.descriptor.size_bytes,
            record_count=chunk.record_count,
            predecessor_seal_hash=chunk.predecessor_seal_hash,
            cumulative_records=chunk.cumulative_records,
            cumulative_bytes=chunk.cumulative_bytes,
            chunk_seal_hash=chunk.chunk_seal_hash,
        )
        for chunk in bundle.chunks
    )
    return AuditExportSnapshotCandidate(snapshot=snapshot, chunks=chunks)


def _manifest_verifier(
    signing_mode: AuditExportSigningMode,
    signing_key: bytes | None,
) -> Callable[[bytes, AuditExportSignedManifestInput], None]:
    def verify(content: bytes, descriptor: AuditExportSignedManifestInput) -> None:
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover - reader validates first
            raise ValueError("audit export final manifest is not JSON") from exc
        if type(value) is not dict:
            raise TypeError("audit export final manifest must be an exact object")
        manifest = cast(dict[str, ClosedAuditExportJSON], value)
        signature = manifest.pop("signature", None)
        signing_body = C("audit-export-final-manifest-signing-body-v2", manifest)
        if signing_mode is AuditExportSigningMode.UNSIGNED:
            if signing_key is not None or signature is not None or descriptor.signature is not None:
                raise ValueError("unsigned audit export final manifest must not carry a signature")
            return
        if type(signing_key) is not bytes or not signing_key:
            raise ValueError("HMAC audit export winner cannot be verified without its signing key")
        expected = hmac.new(signing_key, signing_body, hashlib.sha256).hexdigest()
        if type(signature) is not str or not hmac.compare_digest(signature, expected):
            raise ValueError("audit export final-manifest HMAC verification failed")
        if descriptor.signature != signature:
            raise ValueError("audit export final-manifest signature diverges from its descriptor")

    return verify


def _record_signature_verifier(
    signing_mode: AuditExportSigningMode,
    signing_key: bytes | None,
) -> Callable[[bytes, str], None] | None:
    if signing_mode is AuditExportSigningMode.UNSIGNED:
        return None
    if type(signing_key) is not bytes or not signing_key:
        raise ValueError("HMAC audit export records cannot be verified without signing-key bytes")

    def verify(unsigned_bytes: bytes, signature: str) -> None:
        expected = hmac.new(signing_key, unsigned_bytes, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("audit export record HMAC verification failed")

    return verify


def _read_limits(config: LandscapeExportSettings) -> AuditExportSnapshotReadLimits:
    return AuditExportSnapshotReadLimits(
        max_total_bytes=_required_limit(config.total_byte_limit, "total_byte_limit"),
        max_total_records=_required_limit(config.total_record_limit, "total_record_limit"),
        max_chunks=_required_limit(config.chunk_limit, "chunk_limit"),
        max_chunk_bytes=_required_limit(config.per_chunk_byte_limit, "per_chunk_byte_limit"),
        max_chunk_records=_required_limit(config.per_chunk_record_limit, "per_chunk_record_limit"),
    )


def prepare_audit_export_snapshot(
    db: LandscapeDB,
    *,
    run_id: str,
    config: LandscapeExportSettings,
    signing_key: bytes | None,
    content_store: AuditExportContentStore,
    content_store_resolver: AuditExportContentStoreResolver | None = None,
    repository: AuditExportSnapshotRepository | None = None,
) -> SinkEffectAuditExportSnapshotInput:
    """Reuse or durably materialize one immutable export snapshot winner."""
    if not isinstance(db, LandscapeDB):
        raise TypeError("db must be LandscapeDB")
    if type(config) is not LandscapeExportSettings:
        raise TypeError("config must be exact LandscapeExportSettings")
    if not isinstance(content_store, AuditExportContentStore) or not content_store.is_durable():
        raise TypeError("content_store must implement durable AuditExportContentStore")
    resolver = content_store_resolver or AuditExportContentStoreResolver()
    resolver.register(content_store)
    snapshots = repository or AuditExportSnapshotRepository()
    key = _registry_key(run_id, config)

    winner: AuditExportSnapshotWinner | None = None
    bundle: AuditExportSpooledBundle | None = None
    witness: AuditExportTerminalWitness | None = None
    spool: BinaryIO | None = None
    with open_export_read_transaction(db.engine) as read_model:
        winner = snapshots.find_winner(read_model.connection, key)
        if winner is None:
            witness = read_model.get_export_terminal_witness(run_id)
            derivation = _derivation_config(witness, config, signing_key)
            exporter = LandscapeExporter(
                db,
                signing_key=signing_key,
                include_raw_error_rows=config.include_raw_error_rows,
                read_model=read_model,
                signer_key_id=config.signer_key_id,
                export_format=config.format,
                exporter_version=config.exporter_version,
                serialization_version=config.serialization_version,
                chunking_algorithm_version=config.chunking_algorithm_version,
                per_chunk_byte_limit=_required_limit(config.per_chunk_byte_limit, "per_chunk_byte_limit"),
                per_chunk_record_limit=_required_limit(config.per_chunk_record_limit, "per_chunk_record_limit"),
                derivation_config=derivation,
            )
            spool = TemporaryFile(  # noqa: SIM115 - lifetime deliberately crosses the read transaction
                mode="w+b",
                dir=_private_spool_root(config),
            )
            try:
                bundle = derive_audit_export_bundle_to_spool(
                    exporter.iter_unsigned_run_records(run_id),
                    derivation,
                    spool,
                    max_total_records=_required_limit(config.total_record_limit, "total_record_limit"),
                    max_total_bytes=_required_limit(config.total_byte_limit, "total_byte_limit"),
                    max_chunks=_required_limit(config.chunk_limit, "chunk_limit"),
                )
                spool.flush()
                os.fsync(spool.fileno())
            except BaseException:
                spool.close()
                spool = None
                raise

    signing_mode = AuditExportSigningMode(config.signing_mode)
    verifier = _manifest_verifier(signing_mode, signing_key)
    record_verifier = _record_signature_verifier(signing_mode, signing_key)
    if winner is not None:
        return snapshots.bind_winner(
            winner,
            content_store_resolver=resolver,
            limits=_read_limits(config),
            signed_manifest_verifier=verifier,
            record_signature_verifier=record_verifier,
        )

    assert bundle is not None and witness is not None and spool is not None
    candidate_id = uuid4().hex
    descriptors = (
        *(chunk.descriptor for chunk in bundle.chunks),
        AuditExportContentDescriptor(
            content_ref=bundle.signed_manifest.content_ref,
            content_hash=bundle.signed_manifest.content_hash,
            size_bytes=bundle.signed_manifest.size_bytes,
            object_kind="final_manifest",
        ),
    )
    offsets = (*bundle.chunk_offsets, bundle.signed_manifest_offset)
    try:
        for descriptor, (offset, size) in zip(descriptors, offsets, strict=True):
            observed_ref = content_store.put_immutable(
                _read_spooled(spool, offset, size),
                candidate_id=candidate_id,
                object_kind=descriptor.object_kind,
            )
            if observed_ref != descriptor.content_ref:
                raise ValueError("audit export content store returned a divergent immutable reference")
        candidate = _candidate(
            bundle=bundle,
            witness=witness,
            config=config,
            content_store_id=content_store.content_store_id,
        )
        with db.engine.begin() as connection:
            registration = snapshots.register_candidate(
                connection,
                candidate,
                content_store_resolver=resolver,
                limits=_read_limits(config),
                signed_manifest_verifier=verifier,
                record_signature_verifier=record_verifier,
            )
        if not registration.inserted:
            content_store.mark_candidate_orphans(candidate_id, descriptors)
        winner = registration.winner
    except BaseException:
        content_store.mark_candidate_orphans(candidate_id, descriptors)
        raise
    finally:
        spool.close()

    return snapshots.bind_winner(
        winner,
        content_store_resolver=resolver,
        limits=_read_limits(config),
        signed_manifest_verifier=verifier,
        record_signature_verifier=record_verifier,
    )


def execute_audit_export_effect(
    *,
    factory: RecorderFactory,
    snapshot: SinkEffectAuditExportSnapshotInput,
    sink: object,
    sink_node_id: str,
    target_config: Mapping[str, object],
    worker_id: str,
    lease_ttl: timedelta = timedelta(minutes=5),
    fault_hook: Callable[[SinkEffectExecutionSeam], None] | None = None,
) -> SinkEffectFinalizationResult:
    """Reserve and execute one zero-member audit-export snapshot effect."""
    identity = compute_audit_export_effect_identity(
        snapshot,
        target_config,
        sink_node_id=sink_node_id,
        role=SinkEffectRole.PRIMARY,
    )
    request = SinkEffectExecutionRequest(
        reservation=SinkEffectReservationRequest(
            run_id=snapshot.source_run_id,
            sink_node_id=sink_node_id,
            role=SinkEffectRole.PRIMARY,
            input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
            requested_target_hash=identity.requested_target_hash,
            members=(),
            audit_export_snapshot_id=snapshot.snapshot_id,
            config_hash=identity.config_hash,
            replacing_target=False,
            primary_effect_id=None,
        ),
        effect_input=snapshot,
        finalization_members=(),
    )
    coordinator = SinkEffectCoordinator(
        factory=factory,
        worker_id=worker_id,
        lease_ttl=lease_ttl,
        fault_hook=fault_hook,
    )
    # Capability preflight owns structural validation of the delayed adapter.
    return coordinator.execute(request, cast(Any, sink))


__all__ = ["execute_audit_export_effect", "prepare_audit_export_snapshot"]
