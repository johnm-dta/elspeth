"""Closed, immutable sink-effect protocol value contracts."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
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
    SinkEffectIdentity,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectMemberCandidate,
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
    row = {"value": ordinal, "nested": {"safe": True}}
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(canonical_json(row).encode("utf-8")).hexdigest(),
        row=row,
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
    chunk_record_count: int = 1,
    objects_out: dict[str, bytes] | None = None,
    manifest_mutator: Callable[[dict[str, object]], object] | None = None,
    reader_binding_overrides: dict[str, object] | None = None,
    max_chunk_bytes: int | None = None,
    max_chunk_records: int | None = None,
    record_counter: Callable[[bytes], int] | None = None,
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
        record_count=chunk_record_count,
    )
    manifest_object = {
        "chunk_count": 1,
        "derivation_version": "audit-export-derivation-v1",
        "export_format": "json",
        "exported_at": "2026-07-16T01:02:03.456789Z",
        "final_hash": "6" * 64,
        "hash_algorithm": "sha256",
        "last_chunk_seal_hash": "7" * 64,
        "manifest_hash": manifest_hash,
        "record_chain_algorithm": record_chain_algorithm,
        "record_count": chunk_record_count,
        "record_type": "manifest",
        "registry_key_hash": registry_key_hash,
        "run_id": "source-run-1",
        "schema": "elspeth.audit-export-manifest.v2",
        "signature": signature,
        "signature_algorithm": signing_mode.value,
        "signature_key_id": signer_key_id,
        "snapshot_hash": snapshot_hash,
        "snapshot_id": snapshot_id,
        "snapshot_seal_hash": "8" * 64,
        "source_completed_at": "2026-07-16T01:02:03.456789Z",
        "source_status": "completed",
        "total_bytes": len(chunk_bytes),
    }
    if manifest_mutator is not None:
        manifest_mutator(manifest_object)
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
    reader_kwargs: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "source_run_id": "source-run-1",
        "registry_key_hash": registry_key_hash,
        "manifest_hash": manifest_hash,
        "snapshot_hash": snapshot_hash,
        "export_format": AuditExportFormat.JSON,
        "signing_mode": signing_mode,
        "signer_key_id": signer_key_id,
        "record_count": chunk_record_count,
        "total_bytes": len(chunk_bytes),
        "serialization_version": "audit-export-v2",
        "exported_at": "2026-07-16T01:02:03.456789Z",
        "source_completed_at": "2026-07-16T01:02:03.456789Z",
        "source_status": "completed",
        "last_chunk_seal_hash": "7" * 64,
        "snapshot_seal_hash": "8" * 64,
        "chunks": (chunk,),
        "signed_manifest": signed_manifest,
        "store_resolver": objects.__getitem__,
        "record_counter": record_counter or (lambda content: content.count(b"\n")),
        "signed_manifest_verifier": lambda _content, _descriptor: None,
    }
    if reader_binding_overrides is not None:
        reader_kwargs.update(reader_binding_overrides)
    if max_chunk_bytes is not None:
        reader_kwargs["max_chunk_bytes"] = max_chunk_bytes
    if max_chunk_records is not None:
        reader_kwargs["max_chunk_records"] = max_chunk_records
    reader = _create_restricted_audit_export_snapshot_reader(  # type: ignore[arg-type]
        **reader_kwargs,
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
        record_count=chunk_record_count,
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
            (
                "ordinal",
                "token_id",
                "row_id",
                "ingest_sequence",
                "lineage_json",
                "lineage_hash",
                "payload_hash",
                "row",
                "pending_identity_hash",
                "member_effect_id",
            ),
        ),
        (SinkEffectMemberCandidate, ("token_id", "row", "pending_identity")),
        (
            SinkEffectIdentity,
            (
                "effect_id",
                "artifact_id",
                "artifact_idempotency_key",
                "stream_id",
                "config_hash",
                "requested_target_hash",
                "membership_or_manifest_hash",
                "group_payload_hash",
                "input_kind",
                "members",
                "member_ids",
                "snapshot_hash",
                "final_manifest_identity_hash",
            ),
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


def test_member_row_is_a_detached_closed_canonical_value_tree() -> None:
    source = {"nested": [{"value": 1, "float": 1.25, "null": None}]}
    member = replace(_member(), row=source, payload_hash=sha256(canonical_json(source).encode("utf-8")).hexdigest())
    source["nested"][0]["value"] = 99
    assert member.row["nested"][0]["value"] == 1
    with pytest.raises(TypeError):
        member.row["nested"][0]["value"] = 2  # type: ignore[index]


@pytest.mark.parametrize(
    "bad_value",
    [
        bytearray(b"mutable"),
        object(),
        {"unordered"},
        frozenset({"unordered"}),
        math.inf,
        math.nan,
        2**53,
    ],
)
def test_member_row_rejects_noncanonical_or_authority_values(bad_value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="row"):
        replace(_member(), row={"bad": {"nested": bad_value}})


def test_member_row_rejects_non_string_keys_and_nested_reader_smuggling() -> None:
    with pytest.raises(TypeError, match="string keys"):
        replace(_member(), row={1: "bad"})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="reader"):
        replace(_member(), row={"nested": [{"reader": _export_input().reader}]})


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
            expected_descriptor=EXACT_DESCRIPTOR,
        ).expected_descriptor
        == EXACT_DESCRIPTOR
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
    with pytest.raises(ValueError, match="expected_descriptor"):
        _plan(descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED, expected_descriptor=EXACT_DESCRIPTOR)
    with pytest.raises(ValueError, match="expected_descriptor"):
        _plan(descriptor_mode=SinkEffectDescriptorMode.NO_PUBLICATION, expected_descriptor=None)
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
    missing = object()
    assert inspect.getattr_static(export_input, "members", missing) is missing
    assert inspect.getattr_static(export_input, "target_snapshot_members", missing) is missing
    assert inspect.getattr_static(export_input, "token_id", missing) is missing
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


def test_audit_export_reader_binds_exact_serialization_version() -> None:
    export_input = _export_input()

    assert replace(export_input, serialization_version="audit-export-v2").serialization_version == "audit-export-v2"
    with pytest.raises(ValueError, match=r"serialization_version|reader binding"):
        replace(export_input, serialization_version="audit-export-v3")


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
        assert inspect.getattr_static(reader, forbidden, None) is None
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
            expected_descriptor=EXACT_DESCRIPTOR,
            safe_evidence={"reader": first.reader},
        )


def test_restricted_reader_slots_cannot_be_replaced_or_deleted() -> None:
    export_input = _export_input()
    reader = export_input.reader
    before = (reader.snapshot_id, reader.manifest_hash, reader.chunk_count, list(reader.iter_verified_chunks()))
    for slot in RestrictedAuditExportSnapshotReader.__slots__:
        mangled = f"_RestrictedAuditExportSnapshotReader{slot}" if slot.startswith("__") else slot
        with pytest.raises(TypeError, match="immutable"):
            setattr(reader, mangled, object())
        with pytest.raises(TypeError, match="immutable"):
            delattr(reader, mangled)
    assert (reader.snapshot_id, reader.manifest_hash, reader.chunk_count, list(reader.iter_verified_chunks())) == before


_V2_MANIFEST_FIELDS = {
    "chunk_count",
    "derivation_version",
    "export_format",
    "exported_at",
    "final_hash",
    "hash_algorithm",
    "last_chunk_seal_hash",
    "manifest_hash",
    "record_chain_algorithm",
    "record_count",
    "record_type",
    "registry_key_hash",
    "run_id",
    "schema",
    "signature",
    "signature_algorithm",
    "signature_key_id",
    "snapshot_hash",
    "snapshot_id",
    "snapshot_seal_hash",
    "source_completed_at",
    "source_status",
    "total_bytes",
}


@pytest.mark.parametrize("missing_field", sorted(_V2_MANIFEST_FIELDS))
def test_reader_rejects_each_missing_v2_manifest_field(missing_field: str) -> None:
    with pytest.raises(ValueError, match="exact v2 field set"):
        _export_input(manifest_mutator=lambda manifest: manifest.pop(missing_field))


def test_reader_rejects_extra_v2_manifest_field_and_missing_unsigned_signature() -> None:
    with pytest.raises(ValueError, match="exact v2 field set"):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__("extra", "forbidden"))
    with pytest.raises(ValueError, match="exact v2 field set"):
        _export_input(manifest_mutator=lambda manifest: manifest.pop("signature"))


@pytest.mark.parametrize("integer_field", ["chunk_count", "record_count", "total_bytes"])
def test_reader_rejects_bool_for_v2_manifest_integer(integer_field: str) -> None:
    with pytest.raises(TypeError, match=integer_field):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(integer_field, True))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("schema", "elspeth.audit-export-manifest.v1"),
        ("derivation_version", "audit-export-derivation-v0"),
        ("export_format", "xml"),
        ("hash_algorithm", "sha512"),
        ("record_type", "chunk"),
        ("signature_algorithm", "unknown"),
        ("source_status", "running"),
    ],
)
def test_reader_rejects_wrong_v2_manifest_literals(field_name: str, invalid_value: str) -> None:
    with pytest.raises(ValueError):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(field_name, invalid_value))


@pytest.mark.parametrize(
    "hash_field",
    [
        "final_hash",
        "last_chunk_seal_hash",
        "manifest_hash",
        "registry_key_hash",
        "snapshot_hash",
        "snapshot_id",
        "snapshot_seal_hash",
    ],
)
def test_reader_rejects_noncanonical_v2_manifest_hash(hash_field: str) -> None:
    with pytest.raises(ValueError, match=hash_field):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(hash_field, "A" * 64))


@pytest.mark.parametrize("timestamp_field", ["exported_at", "source_completed_at"])
def test_reader_rejects_noncanonical_v2_manifest_timestamp(timestamp_field: str) -> None:
    with pytest.raises(ValueError, match=timestamp_field):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(timestamp_field, "2026-07-16T01:02:03Z"))


@pytest.mark.parametrize("field_name", sorted(_V2_MANIFEST_FIELDS - {"chunk_count", "record_count", "signature", "total_bytes"}))
def test_reader_rejects_non_string_v2_manifest_field(field_name: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(field_name, 1))


def test_reader_rejects_non_string_non_null_v2_signature() -> None:
    with pytest.raises(TypeError, match="signature"):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__("signature", 1))


@pytest.mark.parametrize(
    ("field_name", "different_valid_value"),
    [
        ("exported_at", "2026-07-16T01:02:04.456789Z"),
        ("source_completed_at", "2026-07-16T01:02:02.456789Z"),
        ("source_status", "empty"),
        ("last_chunk_seal_hash", "9" * 64),
        ("snapshot_seal_hash", "a" * 64),
    ],
)
def test_reader_rejects_valid_but_unregistered_final_manifest_fact(
    field_name: str,
    different_valid_value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _export_input(manifest_mutator=lambda manifest: manifest.__setitem__(field_name, different_valid_value))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("exported_at", "2026-07-16T01:02:03Z"),
        ("source_completed_at", "not-a-timestamp"),
        ("source_status", "running"),
        ("last_chunk_seal_hash", "A" * 64),
        ("snapshot_seal_hash", "short"),
    ],
)
def test_reader_factory_rejects_invalid_registered_final_manifest_fact(field_name: str, invalid_value: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _export_input(reader_binding_overrides={field_name: invalid_value})


@pytest.mark.parametrize(
    ("chunk_bytes", "chunk_records", "max_chunk_bytes", "max_chunk_records", "message"),
    [
        (b"0123456789", 1, 9, 1, "max_chunk_bytes"),
        (b"one\ntwo\n", 2, 8, 1, "max_chunk_records"),
    ],
)
def test_reader_factory_enforces_configured_per_chunk_limits_below_hard_maxima(
    chunk_bytes: bytes,
    chunk_records: int,
    max_chunk_bytes: int,
    max_chunk_records: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _export_input(
            chunk_bytes=chunk_bytes,
            chunk_record_count=chunk_records,
            max_chunk_bytes=max_chunk_bytes,
            max_chunk_records=max_chunk_records,
        )


def test_reader_rechecks_configured_chunk_byte_limit_against_store_observation() -> None:
    objects: dict[str, bytes] = {}
    export_input = _export_input(objects_out=objects, max_chunk_bytes=13, max_chunk_records=1)
    chunk = export_input.chunks[0]
    objects[chunk.content_ref] = b'{"record":10}\n'

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        list(export_input.reader.iter_verified_chunks())


def test_reader_rechecks_configured_chunk_record_limit_against_store_observation() -> None:
    observed_records = [1]
    export_input = _export_input(
        max_chunk_bytes=64,
        max_chunk_records=1,
        record_counter=lambda _content: observed_records[0],
    )
    observed_records[0] = 2

    with pytest.raises(ValueError, match="max_chunk_records"):
        list(export_input.reader.iter_verified_chunks())


def test_reader_rechecks_configured_chunk_limit_after_descriptor_tamper() -> None:
    objects: dict[str, bytes] = {}
    export_input = _export_input(objects_out=objects, max_chunk_bytes=13, max_chunk_records=1)
    chunk = export_input.chunks[0]
    replacement = b"x" * 14
    replacement_hash = _sha256(replacement)
    replacement_ref = f"sha256:{replacement_hash}"
    objects[replacement_ref] = replacement
    object.__setattr__(chunk, "content_ref", replacement_ref)
    object.__setattr__(chunk, "content_hash", replacement_hash)
    object.__setattr__(chunk, "size_bytes", len(replacement))

    with pytest.raises(ValueError, match="max_chunk_bytes"):
        list(export_input.reader.iter_verified_chunks())


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
            lineage_json="[]",
            lineage_hash=sha256(b"[]").hexdigest(),
            payload_hash=sha256(b"{}").hexdigest(),
            row={},
        )


def test_member_refuses_noncanonical_or_divergent_lineage_and_payload_hashes() -> None:
    baseline = _member()
    with pytest.raises(ValueError, match="lineage_hash"):
        replace(baseline, lineage_hash="0" * 64)
    with pytest.raises(ValueError, match="payload_hash"):
        replace(baseline, payload_hash="0" * 64)
    with pytest.raises(ValueError, match="canonical"):
        replace(baseline, lineage_json="[ ]", lineage_hash=sha256(b"[ ]").hexdigest())

    with pytest.raises(ValueError, match="non-empty"):
        RestrictedSinkEffectContext(
            run_id="  ",
            run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
            operation_id="operation-1",
            sink_node_id="sink-1",
        )
