"""Closed, immutable sink-effect protocol value contracts."""

from __future__ import annotations

import inspect
from dataclasses import FrozenInstanceError, asdict, fields, replace
from datetime import UTC, datetime
from hashlib import sha256
from types import MappingProxyType

import pytest

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.plugin_protocols import SinkEffectProtocol
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportFormat,
    AuditExportSignedManifestInput,
    AuditExportSigningMode,
    AuditExportSnapshotChunkInput,
    RestrictedAuditExportSnapshotReader,
    RestrictedSinkEffectContext,
    SinkEffectAttemptAction,
    SinkEffectAttemptState,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
    SinkEffectReconcileResult,
    SinkEffectRole,
    SinkEffectState,
    _create_restricted_audit_export_snapshot_reader,
)

EXACT_DESCRIPTOR = ArtifactDescriptor(
    artifact_type="file",
    path_or_uri="/safe/output.csv",
    content_hash="abc123",
    size_bytes=12,
)
SAFE_EVIDENCE = {"content_hash": "abc123", "versions": ["v1", "v2"]}


def _member(ordinal: int = 0) -> SinkEffectMember:
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_key=f"lineage-{ordinal}",
        payload_hash=f"payload-{ordinal}",
        row={"value": ordinal, "nested": {"safe": True}},
    )


def _inspection(
    mode: SinkEffectInspectionMode = SinkEffectInspectionMode.INSPECTED,
) -> SinkEffectInspection:
    return SinkEffectInspection(
        mode=mode,
        reference="https://storage.example.test/object/version/1",
        evidence=SAFE_EVIDENCE,
    )


def _sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _export_input(
    *,
    signing_mode: AuditExportSigningMode = AuditExportSigningMode.UNSIGNED,
    chunk_bytes: bytes = b'{"record":1}\n',
    objects_out: dict[str, bytes] | None = None,
) -> SinkEffectAuditExportSnapshotInput:
    snapshot_id = "1" * 64
    registry_key_hash = "2" * 64
    manifest_hash = "3" * 64
    snapshot_hash = "4" * 64
    signer_key_id = "UNSIGNED" if signing_mode is AuditExportSigningMode.UNSIGNED else "operator-key-1"
    signature = None if signing_mode is AuditExportSigningMode.UNSIGNED else "5" * 64
    record_chain_algorithm = (
        "sha256_concat_record_sha256_v1" if signing_mode is AuditExportSigningMode.UNSIGNED else "sha256_concat_hmac_sha256_signatures_v1"
    )
    chunk_hash = _sha256(chunk_bytes)
    chunk = AuditExportSnapshotChunkInput(
        ordinal=0,
        content_ref=f"sha256:{chunk_hash}",
        content_hash=chunk_hash,
        size_bytes=len(chunk_bytes),
        record_count=1,
    )
    manifest_object = {
        "chunk_count": 1,
        "derivation_version": "audit-export-derivation-v1",
        "export_format": "json",
        "final_hash": "6" * 64,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": record_chain_algorithm,
        "record_count": 1,
        "registry_key_hash": registry_key_hash,
        "run_id": "source-run-1",
        "schema": "elspeth.audit-export-manifest.v2",
        "signature": signature,
        "signature_algorithm": signing_mode.value,
        "signature_key_id": signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "total_bytes": len(chunk_bytes),
    }
    manifest_bytes = canonical_json(manifest_object).encode("utf-8")
    signed_manifest_hash = _sha256(manifest_bytes)
    signed_manifest = AuditExportSignedManifestInput(
        content_ref=f"sha256:{signed_manifest_hash}",
        content_hash=signed_manifest_hash,
        size_bytes=len(manifest_bytes),
        manifest_schema="elspeth.audit-export-manifest.v2",
        derivation_version="audit-export-derivation-v1",
        signature_algorithm=signing_mode,
        signature_key_id=signer_key_id,
        record_chain_algorithm=record_chain_algorithm,
        final_hash="6" * 64,
        signature=signature,
    )
    objects = objects_out if objects_out is not None else {}
    objects.update({chunk.content_ref: chunk_bytes, signed_manifest.content_ref: manifest_bytes})
    reader = _create_restricted_audit_export_snapshot_reader(
        snapshot_id=snapshot_id,
        source_run_id="source-run-1",
        registry_key_hash=registry_key_hash,
        manifest_hash=manifest_hash,
        snapshot_hash=snapshot_hash,
        export_format=AuditExportFormat.JSON,
        signing_mode=signing_mode,
        signer_key_id=signer_key_id,
        record_count=1,
        total_bytes=len(chunk_bytes),
        chunks=(chunk,),
        signed_manifest=signed_manifest,
        store_resolver=objects.__getitem__,
        record_counter=lambda content: content.count(b"\n"),
        signed_manifest_verifier=lambda _content, _descriptor: None,
    )
    return SinkEffectAuditExportSnapshotInput(
        snapshot_id=snapshot_id,
        source_run_id="source-run-1",
        registry_key_hash=registry_key_hash,
        manifest_hash=manifest_hash,
        snapshot_hash=snapshot_hash,
        serialization_version="audit-export-v2",
        export_format=AuditExportFormat.JSON,
        signing_mode=signing_mode,
        signer_key_id=signer_key_id,
        record_count=1,
        total_bytes=len(chunk_bytes),
        chunk_count=1,
        chunks=(chunk,),
        signed_manifest=signed_manifest,
        reader=reader,
    )


def _plan(
    *,
    descriptor_mode: SinkEffectDescriptorMode = SinkEffectDescriptorMode.PRECOMPUTED,
    inspection_mode: SinkEffectInspectionMode = SinkEffectInspectionMode.INSPECTED,
    expected_descriptor: ArtifactDescriptor | None = EXACT_DESCRIPTOR,
) -> SinkEffectPlan:
    return SinkEffectPlan(
        effect_id="effect-1",
        protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        descriptor_mode=descriptor_mode,
        inspection_mode=inspection_mode,
        target="https://storage.example.test/object",
        plan_hash="plan-hash",
        payload_hash="payload-hash",
        expected_descriptor=expected_descriptor,
        safe_evidence=SAFE_EVIDENCE,
    )


def test_public_enums_are_exact_closed_wire_vocabularies() -> None:
    assert SINK_EFFECT_PROTOCOL_VERSION == "sink-effect-v1"
    assert {member.name: member.value for member in SinkEffectRole} == {
        "PRIMARY": "primary",
        "FAILSINK": "failsink",
    }
    assert {member.name: member.value for member in SinkEffectState} == {
        "RESERVED": "reserved",
        "PREPARED": "prepared",
        "IN_FLIGHT": "in_flight",
        "FINALIZED": "finalized",
    }
    assert {member.name: member.value for member in SinkEffectDescriptorMode} == {
        "PRECOMPUTED": "precomputed",
        "RESULT_DERIVED": "result_derived",
        "NO_PUBLICATION": "no_publication",
    }
    assert {member.name: member.value for member in SinkEffectInspectionMode} == {
        "INSPECTED": "inspected",
        "NO_INSPECTION_REQUIRED": "no_inspection_required",
    }
    assert {member.name: member.value for member in SinkEffectInputKind} == {
        "PIPELINE_MEMBERS": "pipeline_members",
        "AUDIT_EXPORT_SNAPSHOT": "audit_export_snapshot",
    }
    assert {member.name: member.value for member in AuditExportFormat} == {"JSON": "json", "CSV": "csv"}
    assert {member.name: member.value for member in AuditExportSigningMode} == {
        "UNSIGNED": "unsigned",
        "HMAC_SHA256": "hmac_sha256",
    }
    assert {member.name: member.value for member in SinkEffectReconcileKind} == {
        "NOT_APPLIED": "not_applied",
        "APPLIED_WITH_EXACT_DESCRIPTOR": "applied_with_exact_descriptor",
        "UNKNOWN": "unknown",
    }
    assert {member.name: member.value for member in SinkEffectAttemptAction} == {
        "INSPECT": "inspect",
        "COMMIT": "commit",
        "RECONCILE": "reconcile",
    }
    assert {member.name: member.value for member in SinkEffectAttemptState} == {
        "INTENT": "intent",
        "RETURNED": "returned",
        "RESPONSE_LOST": "response_lost",
        "ERROR": "error",
    }


def test_sink_effect_protocol_has_independent_kind_capability_and_exact_methods() -> None:
    annotations = SinkEffectProtocol.__annotations__
    assert "effect_protocol_version" in annotations
    assert "supported_effect_modes" in annotations
    assert "supported_effect_input_kinds" in annotations
    assert {
        name
        for name in ("inspect_effect", "prepare_effect", "commit_effect", "reconcile_effect")
        if callable(getattr(SinkEffectProtocol, name, None))
    } == {
        "inspect_effect",
        "prepare_effect",
        "commit_effect",
        "reconcile_effect",
    }


@pytest.mark.parametrize(
    ("record_type", "expected_fields"),
    [
        (
            SinkEffectMember,
            ("ordinal", "token_id", "row_id", "ingest_sequence", "lineage_key", "payload_hash", "row"),
        ),
        (SinkEffectInspectionRequest, ("effect_id", "target", "predecessor_descriptor")),
        (SinkEffectInspection, ("mode", "reference", "evidence")),
        (SinkEffectPipelineMembersInput, ("members", "target_snapshot_members")),
        (AuditExportSnapshotChunkInput, ("ordinal", "content_ref", "content_hash", "size_bytes", "record_count")),
        (
            AuditExportSignedManifestInput,
            (
                "content_ref",
                "content_hash",
                "size_bytes",
                "manifest_schema",
                "derivation_version",
                "signature_algorithm",
                "signature_key_id",
                "record_chain_algorithm",
                "final_hash",
                "signature",
            ),
        ),
        (
            SinkEffectAuditExportSnapshotInput,
            (
                "snapshot_id",
                "source_run_id",
                "registry_key_hash",
                "manifest_hash",
                "snapshot_hash",
                "serialization_version",
                "export_format",
                "signing_mode",
                "signer_key_id",
                "record_count",
                "total_bytes",
                "chunk_count",
                "chunks",
                "signed_manifest",
                "reader",
            ),
        ),
        (SinkEffectPrepareRequest, ("effect_id", "effect_input", "inspection")),
        (
            SinkEffectPlan,
            (
                "effect_id",
                "protocol_version",
                "input_kind",
                "descriptor_mode",
                "inspection_mode",
                "target",
                "plan_hash",
                "payload_hash",
                "expected_descriptor",
                "safe_evidence",
            ),
        ),
        (SinkEffectCommitResult, ("descriptor", "evidence", "accepted_ordinals", "diverted_ordinals")),
        (SinkEffectReconcileResult, ("kind", "descriptor", "evidence")),
        (RestrictedSinkEffectContext, ("run_id", "run_started_at", "operation_id", "sink_node_id")),
    ],
)
def test_public_value_objects_have_exact_field_shapes(record_type: type[object], expected_fields: tuple[str, ...]) -> None:
    assert tuple(field.name for field in fields(record_type)) == expected_fields


def test_reconcile_result_is_closed_and_exact_descriptor_is_required() -> None:
    exact = SinkEffectReconcileResult.applied(EXACT_DESCRIPTOR, evidence=SAFE_EVIDENCE)
    assert exact.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert exact.descriptor == EXACT_DESCRIPTOR

    with pytest.raises(ValueError, match="descriptor"):
        SinkEffectReconcileResult(kind=SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR)


@pytest.mark.parametrize(
    "kind",
    [SinkEffectReconcileKind.NOT_APPLIED, SinkEffectReconcileKind.UNKNOWN],
)
def test_non_applied_reconcile_kinds_cannot_claim_a_descriptor(kind: SinkEffectReconcileKind) -> None:
    with pytest.raises(ValueError, match="descriptor"):
        SinkEffectReconcileResult(kind=kind, descriptor=EXACT_DESCRIPTOR)


def test_only_not_applied_carries_permission_to_commit() -> None:
    not_applied = SinkEffectReconcileResult.not_applied(evidence=SAFE_EVIDENCE)
    unknown = SinkEffectReconcileResult.unknown(evidence=SAFE_EVIDENCE)
    exact = SinkEffectReconcileResult.applied(EXACT_DESCRIPTOR, evidence=SAFE_EVIDENCE)

    assert not_applied.may_commit is True
    assert unknown.may_commit is False
    assert exact.may_commit is False


def test_evidence_and_member_rows_are_deeply_immutable_and_copy_isolated() -> None:
    source = {"nested": {"items": ["v1"]}}
    result = SinkEffectReconcileResult.unknown(evidence=source)
    member = _member()

    source["nested"]["items"].append("mutated")

    assert isinstance(result.evidence, MappingProxyType)
    assert isinstance(result.evidence["nested"], MappingProxyType)
    assert result.evidence["nested"]["items"] == ("v1",)
    assert isinstance(member.row, MappingProxyType)
    assert isinstance(member.row["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        result.evidence["new"] = "forbidden"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        member.ordinal = 2  # type: ignore[misc]


def test_all_evidence_fields_enforce_the_64_kib_canonical_json_bound() -> None:
    oversized = {"payload": "x" * (64 * 1024)}

    constructors = (
        lambda: SinkEffectInspection(
            mode=SinkEffectInspectionMode.INSPECTED,
            reference="safe-reference",
            evidence=oversized,
        ),
        lambda: _plan(expected_descriptor=EXACT_DESCRIPTOR).__class__(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=oversized,
        ),
        lambda: SinkEffectCommitResult(
            descriptor=EXACT_DESCRIPTOR,
            evidence=oversized,
            accepted_ordinals=(0,),
            diverted_ordinals=(),
        ),
        lambda: SinkEffectReconcileResult.unknown(evidence=oversized),
    )

    for construct in constructors:
        with pytest.raises(ValueError, match="64 KiB"):
            construct()


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://user:password@example.test/object",
        "https://example.test/object?token=secret",
        "https://example.test/object#sig=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        "Password=super-secret;Server=db.example.test",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
    ],
)
def test_credential_bearing_target_and_reference_forms_fail_closed(unsafe: str) -> None:
    with pytest.raises(ValueError, match="credential"):
        SinkEffectInspectionRequest(
            effect_id="effect-1",
            target=unsafe,
            predecessor_descriptor=None,
        )
    with pytest.raises(ValueError, match="credential"):
        SinkEffectInspection(
            mode=SinkEffectInspectionMode.INSPECTED,
            reference=unsafe,
            evidence=SAFE_EVIDENCE,
        )
    with pytest.raises(ValueError, match="credential"):
        SinkEffectPlan(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target=unsafe,
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=SAFE_EVIDENCE,
        )


def test_plan_protocol_descriptor_and_inspection_modes_validate_exactly() -> None:
    assert _plan().expected_descriptor == EXACT_DESCRIPTOR
    assert (
        _plan(
            descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED,
            expected_descriptor=None,
        ).expected_descriptor
        is None
    )
    assert (
        _plan(
            descriptor_mode=SinkEffectDescriptorMode.NO_PUBLICATION,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            expected_descriptor=None,
        ).expected_descriptor
        is None
    )

    with pytest.raises(ValueError, match="protocol_version"):
        _plan().__class__(
            effect_id="effect-1",
            protocol_version="sink-effect-v0",
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=SAFE_EVIDENCE,
        )
    with pytest.raises(ValueError, match="expected_descriptor"):
        _plan(expected_descriptor=None)
    for descriptor_mode in (
        SinkEffectDescriptorMode.RESULT_DERIVED,
        SinkEffectDescriptorMode.NO_PUBLICATION,
    ):
        with pytest.raises(ValueError, match="expected_descriptor"):
            _plan(descriptor_mode=descriptor_mode, expected_descriptor=EXACT_DESCRIPTOR)
    with pytest.raises((TypeError, ValueError), match="descriptor_mode"):
        _plan().__class__(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode="precomputed",  # type: ignore[arg-type]
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=SAFE_EVIDENCE,
        )
    with pytest.raises((TypeError, ValueError), match="inspection_mode"):
        _plan().__class__(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode="inspected",  # type: ignore[arg-type]
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=SAFE_EVIDENCE,
        )
    with pytest.raises((TypeError, ValueError), match="input_kind"):
        SinkEffectPlan(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind="pipeline_members",  # type: ignore[arg-type]
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.INSPECTED,
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence=SAFE_EVIDENCE,
        )


def test_prepare_and_commit_ordinals_are_immutable_unique_and_disjoint() -> None:
    pipeline_input = SinkEffectPipelineMembersInput(
        members=[_member(0), _member(1)],
        target_snapshot_members=[_member(0), _member(1)],
    )
    request = SinkEffectPrepareRequest(
        effect_id="effect-1",
        effect_input=pipeline_input,
        inspection=_inspection(),
    )
    result = SinkEffectCommitResult(
        descriptor=EXACT_DESCRIPTOR,
        evidence=SAFE_EVIDENCE,
        accepted_ordinals=[0],
        diverted_ordinals=[1],
    )

    assert pipeline_input.members == (_member(0), _member(1))
    assert pipeline_input.target_snapshot_members == (_member(0), _member(1))
    assert pipeline_input.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS
    assert request.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS
    assert result.accepted_ordinals == (0,)
    assert result.diverted_ordinals == (1,)

    with pytest.raises(ValueError, match="unique"):
        SinkEffectPipelineMembersInput(
            members=(_member(0), _member(0)),
            target_snapshot_members=(),
        )
    with pytest.raises(ValueError, match="non-negative"):
        SinkEffectCommitResult(
            descriptor=EXACT_DESCRIPTOR,
            evidence=SAFE_EVIDENCE,
            accepted_ordinals=(-1,),
            diverted_ordinals=(),
        )
    with pytest.raises(ValueError, match="overlap"):
        SinkEffectCommitResult(
            descriptor=EXACT_DESCRIPTOR,
            evidence=SAFE_EVIDENCE,
            accepted_ordinals=(0,),
            diverted_ordinals=(0,),
        )


@pytest.mark.parametrize(
    "members",
    [(_member(0), _member(2)), (_member(1), _member(0))],
    ids=["gap", "out-of-order"],
)
def test_prepare_member_ordinals_must_be_dense_and_ordered(
    members: tuple[SinkEffectMember, ...],
) -> None:
    with pytest.raises(ValueError, match="dense and ordered"):
        SinkEffectPipelineMembersInput(
            members=members,
            target_snapshot_members=(),
        )


def test_pipeline_input_requires_nonempty_current_members_and_dense_snapshot_members() -> None:
    with pytest.raises(ValueError, match="members must be non-empty"):
        SinkEffectPipelineMembersInput(members=(), target_snapshot_members=())
    with pytest.raises(ValueError, match="dense and ordered"):
        SinkEffectPipelineMembersInput(
            members=(_member(0),),
            target_snapshot_members=(_member(0), _member(2)),
        )


def test_prepare_request_is_a_single_closed_union_with_derived_kind_and_plan_match() -> None:
    pipeline = SinkEffectPipelineMembersInput(members=(_member(0),), target_snapshot_members=())
    request = SinkEffectPrepareRequest(effect_id="effect-1", effect_input=pipeline, inspection=_inspection())
    assert request.input_kind is SinkEffectInputKind.PIPELINE_MEMBERS
    assert "input_kind" not in {field.name for field in fields(request)}
    request.validate_plan(_plan())
    with pytest.raises(ValueError, match="input kind"):
        request.validate_plan(replace(_plan(), input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT))
    with pytest.raises(TypeError, match="closed sink effect input union"):
        SinkEffectPrepareRequest(effect_id="effect-1", effect_input=object(), inspection=_inspection())  # type: ignore[arg-type]


def test_export_input_is_dense_bounded_exact_and_has_no_pipeline_fields() -> None:
    export_input = _export_input()
    assert export_input.input_kind is SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT
    assert isinstance(export_input.chunks, tuple)
    assert not hasattr(export_input, "members")
    assert not hasattr(export_input, "target_snapshot_members")
    assert not hasattr(export_input, "token_id")
    assert list(export_input.reader.iter_verified_chunks()) == [b'{"record":1}\n']

    chunk = export_input.chunks[0]
    with pytest.raises(ValueError, match="dense and ordered"):
        replace(export_input, chunks=(replace(chunk, ordinal=1),))
    with pytest.raises(ValueError, match="chunk_count"):
        replace(export_input, chunk_count=2)
    with pytest.raises(ValueError, match="record_count"):
        replace(export_input, record_count=2)
    with pytest.raises(ValueError, match="total_bytes"):
        replace(export_input, total_bytes=chunk.size_bytes + 1)
    with pytest.raises(ValueError, match="strictly positive"):
        replace(export_input, chunks=(replace(chunk, size_bytes=0),), total_bytes=0)


def test_chunk_and_manifest_references_must_match_exact_lowercase_hashes() -> None:
    export_input = _export_input()
    chunk = export_input.chunks[0]
    manifest = export_input.signed_manifest
    different_hash = "a" * 64 if chunk.content_hash != "a" * 64 else "b" * 64
    with pytest.raises(ValueError, match="content_ref"):
        replace(chunk, content_ref=f"sha256:{different_hash}")
    with pytest.raises(ValueError, match="lowercase 64-character hexadecimal"):
        replace(chunk, content_hash=chunk.content_hash.upper())
    with pytest.raises(ValueError, match="content_ref"):
        replace(manifest, content_ref=f"sha256:{'f' * 64}")
    with pytest.raises(ValueError, match="size_bytes"):
        replace(manifest, size_bytes=0)


@pytest.mark.parametrize("signing_mode", [AuditExportSigningMode.UNSIGNED, AuditExportSigningMode.HMAC_SHA256])
def test_signed_manifest_mapping_is_closed_and_exact(signing_mode: AuditExportSigningMode) -> None:
    export_input = _export_input(signing_mode=signing_mode)
    manifest = export_input.signed_manifest
    assert manifest.signature_algorithm is signing_mode
    assert manifest.signature_key_id == export_input.signer_key_id

    with pytest.raises(ValueError, match="manifest_schema"):
        replace(manifest, manifest_schema="elspeth.audit-export-manifest.v1")
    with pytest.raises(ValueError, match="derivation_version"):
        replace(manifest, derivation_version="audit-export-derivation-v0")
    with pytest.raises(ValueError, match="final_hash"):
        replace(manifest, final_hash="not-a-hash")
    if signing_mode is AuditExportSigningMode.UNSIGNED:
        with pytest.raises(ValueError, match="signature"):
            replace(manifest, signature="5" * 64)
        with pytest.raises(ValueError, match="record_chain_algorithm"):
            replace(manifest, record_chain_algorithm="sha256_concat_hmac_sha256_signatures_v1")
    else:
        with pytest.raises(ValueError, match="signature"):
            replace(manifest, signature=None)
        with pytest.raises(ValueError, match="signature_key_id"):
            replace(manifest, signature_key_id="UNSIGNED")


def test_export_parent_rejects_manifest_or_reader_binding_mismatch() -> None:
    export_input = _export_input()
    with pytest.raises(ValueError, match="signature_key_id"):
        replace(export_input, signer_key_id="other-key")
    with pytest.raises(ValueError, match="reader binding"):
        replace(export_input, snapshot_id="9" * 64)
    with pytest.raises(ValueError, match="reader binding"):
        replace(export_input, manifest_hash="8" * 64)
    with pytest.raises(ValueError, match="reader binding"):
        replace(export_input, snapshot_hash="7" * 64)
    with pytest.raises(ValueError, match="reader binding"):
        replace(export_input, chunks=(replace(export_input.chunks[0], record_count=2),), record_count=2)
    with pytest.raises(ValueError, match="reader binding"):
        replacement = replace(export_input.signed_manifest, final_hash="7" * 64)
        replace(export_input, signed_manifest=replacement)


def test_restricted_reader_keeps_manifest_separate_and_exposes_no_arbitrary_read() -> None:
    export_input = _export_input()
    reader = export_input.reader
    assert list(reader.iter_verified_chunks()) == [b'{"record":1}\n']
    manifest_bytes = reader.read_verified_signed_manifest()
    assert not manifest_bytes.endswith(b"\n")
    assert b'"record":1' not in manifest_bytes
    assert inspect.signature(reader.read_verified_signed_manifest).parameters == {}
    for forbidden in ("read", "read_ref", "resolve", "query", "landscape", "credentials", "signer", "secret"):
        assert not hasattr(reader, forbidden)
    with pytest.raises(TypeError):
        reader.read_verified_signed_manifest("sha256:" + "0" * 64)  # type: ignore[call-arg]


def test_restricted_reader_rechecks_chunk_and_manifest_bytes_on_every_access() -> None:
    objects: dict[str, bytes] = {}
    export_input = _export_input(objects_out=objects)
    chunk = export_input.chunks[0]
    manifest = export_input.signed_manifest

    objects[chunk.content_ref] = b'{"record":2}\n'
    with pytest.raises(ValueError, match="hash"):
        list(export_input.reader.iter_verified_chunks())

    objects[chunk.content_ref] = b'{"record":1}\n'
    objects[manifest.content_ref] = objects[manifest.content_ref] + b"\n"
    with pytest.raises(ValueError, match="size"):
        export_input.reader.read_verified_signed_manifest()


def test_reader_is_factory_only_and_excluded_from_repr_equality_and_serialization() -> None:
    first = _export_input()
    second = _export_input()
    assert first.reader is not second.reader
    assert first == second
    assert "RestrictedAuditExportSnapshotReader" not in repr(first)
    with pytest.raises(TypeError, match="factory"):
        RestrictedAuditExportSnapshotReader()  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="serialized"):
        asdict(first)
    with pytest.raises((TypeError, ValueError)):
        SinkEffectPlan(
            effect_id="effect-1",
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
            descriptor_mode=SinkEffectDescriptorMode.NO_PUBLICATION,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            target="safe-target",
            plan_hash="plan-hash",
            payload_hash="payload-hash",
            expected_descriptor=None,
            safe_evidence={"reader": first.reader},
        )


def test_restricted_context_is_frozen() -> None:
    ctx = RestrictedSinkEffectContext(
        run_id="run-1",
        run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
        operation_id="operation-1",
        sink_node_id="sink-1",
    )
    with pytest.raises(FrozenInstanceError):
        ctx.operation_id = "changed"  # type: ignore[misc]


def test_required_identifiers_reject_whitespace_only_strings() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        SinkEffectMember(
            ordinal=0,
            token_id=" \t ",
            row_id="row-0",
            ingest_sequence=0,
            lineage_key="lineage-0",
            payload_hash="payload-0",
            row={},
        )

    with pytest.raises(ValueError, match="non-empty"):
        RestrictedSinkEffectContext(
            run_id="  ",
            run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
            operation_id="operation-1",
            sink_node_id="sink-1",
        )
