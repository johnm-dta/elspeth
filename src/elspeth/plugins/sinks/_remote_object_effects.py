"""Durable plan primitives shared by conditional remote-object sinks."""

from __future__ import annotations

import base64
import hashlib
import os
import stat
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPlan,
    SinkEffectReconcileResult,
)
from elspeth.plugins.sinks._diversion_attribution import DiversionAttribution, parse_diversion_attribution

_EVIDENCE_SCHEMA: Final = "remote-object-effect-plan-v1"
_INSPECTION_SCHEMA: Final = "remote-object-effect-inspection-v1"
_COPY_BYTES: Final = 64 * 1024


class RemoteObjectEffectError(RuntimeError):
    """Base class for fail-closed remote effect validation errors."""


class RemoteObjectEffectLimitError(RemoteObjectEffectError):
    """Raised before dispatch when a durable body exceeds its bound."""


class RemoteObjectPreconditionError(RemoteObjectEffectError):
    """Raised when durable evidence no longer describes the selected object."""


@dataclass(frozen=True, slots=True)
class RemoteObjectObservation:
    exists: bool
    etag: str | None
    content_hash: str | None
    size_bytes: int | None
    effect_id: str | None = None
    plan_hash: str | None = None
    protocol_version: str | None = None
    checksum_algorithm: str | None = None
    checksum_b64: str | None = None


@dataclass(frozen=True, slots=True)
class RemoteObjectPlanEvidence:
    provider: str
    target: str
    staging_path: str
    precondition: str
    predecessor_etag: str | None
    staged_hash: str
    staged_size: int
    checksum_algorithm: str
    checksum_b64: str
    format_name: str
    accepted_ordinals: tuple[int, ...]
    diverted_ordinals: tuple[int, ...]
    diversion_attribution: tuple[DiversionAttribution, ...]
    publication_kind: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "accepted_ordinals": list(self.accepted_ordinals),
            "diversion_attribution": [item.as_mapping() for item in self.diversion_attribution],
            "diverted_ordinals": list(self.diverted_ordinals),
            "checksum_algorithm": self.checksum_algorithm,
            "checksum_b64": self.checksum_b64,
            "format_name": self.format_name,
            "precondition": self.precondition,
            "predecessor_etag": self.predecessor_etag,
            "provider": self.provider,
            "publication_kind": self.publication_kind,
            "schema": _EVIDENCE_SCHEMA,
            "staged_hash": self.staged_hash,
            "staged_size": self.staged_size,
            "staging_path": self.staging_path,
            "target": self.target,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RemoteObjectPlanEvidence:
        if value.get("schema") != _EVIDENCE_SCHEMA:
            raise RemoteObjectPreconditionError("remote object plan schema is missing or divergent")
        accepted = _ordinals(value.get("accepted_ordinals"), "accepted_ordinals")
        diverted = _ordinals(value.get("diverted_ordinals"), "diverted_ordinals")
        if set(accepted) & set(diverted):
            raise RemoteObjectPreconditionError("accepted and diverted ordinals overlap")
        try:
            diversion_attribution = parse_diversion_attribution(
                value.get("diversion_attribution"),
                diverted_ordinals=diverted,
            )
        except ValueError as exc:
            raise RemoteObjectPreconditionError(str(exc)) from exc
        precondition = _string(value.get("precondition"), "precondition")
        if precondition not in {"if_none_match", "if_match"}:
            raise RemoteObjectPreconditionError("remote object precondition is not closed")
        predecessor_etag = _optional_string(value.get("predecessor_etag"), "predecessor_etag")
        if (precondition == "if_match") != (predecessor_etag is not None):
            raise RemoteObjectPreconditionError("remote object precondition and predecessor ETag diverge")
        publication_kind = _string(value.get("publication_kind"), "publication_kind")
        if publication_kind not in {"conditional_create", "conditional_replace", "inherited", "virtual"}:
            raise RemoteObjectPreconditionError("remote object publication kind is not closed")
        checksum_algorithm = _checksum_algorithm(value.get("checksum_algorithm"))
        return cls(
            provider=_string(value.get("provider"), "provider"),
            target=_string(value.get("target"), "target"),
            staging_path=_string(value.get("staging_path"), "staging_path"),
            precondition=precondition,
            predecessor_etag=predecessor_etag,
            staged_hash=_lower_hex(value.get("staged_hash"), "staged_hash"),
            staged_size=_integer(value.get("staged_size"), "staged_size"),
            checksum_algorithm=checksum_algorithm,
            checksum_b64=_checksum_b64(value.get("checksum_b64"), checksum_algorithm),
            format_name=_string(value.get("format_name"), "format_name"),
            accepted_ordinals=accepted,
            diverted_ordinals=diverted,
            diversion_attribution=diversion_attribution,
            publication_kind=publication_kind,
        )


def _string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise RemoteObjectPreconditionError(f"{field_name} must be a non-empty exact string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise RemoteObjectPreconditionError(f"{field_name} must be a non-negative exact integer")
    return value


def _lower_hex(value: object, field_name: str) -> str:
    result = _string(value, field_name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise RemoteObjectPreconditionError(f"{field_name} must be a lowercase SHA-256 digest")
    return result


def _checksum_algorithm(value: object) -> str:
    algorithm = _string(value, "checksum_algorithm")
    if algorithm not in {"sha256", "md5"}:
        raise RemoteObjectPreconditionError("checksum_algorithm must be sha256 or md5")
    return algorithm


def _checksum_b64(value: object, algorithm: str) -> str:
    encoded = _string(value, "checksum_b64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise RemoteObjectPreconditionError("checksum_b64 must be canonical base64") from exc
    expected_size = 32 if algorithm == "sha256" else 16
    if len(raw) != expected_size or base64.b64encode(raw).decode("ascii") != encoded:
        raise RemoteObjectPreconditionError("checksum_b64 has the wrong digest length or encoding")
    return encoded


def _ordinals(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise RemoteObjectPreconditionError(f"{field_name} must be an ordered sequence")
    result = tuple(_integer(item, field_name) for item in value)
    if result != tuple(sorted(result)) or len(result) != len(set(result)):
        raise RemoteObjectPreconditionError(f"{field_name} must contain unique ascending ordinals")
    return result


def inspect_remote_object(
    *,
    provider: str,
    target: str,
    request: SinkEffectInspectionRequest,
    observation: RemoteObjectObservation,
) -> SinkEffectInspection:
    """Bind the exact remote pre-image and validate any durable predecessor."""
    predecessor = request.predecessor_descriptor
    if observation.exists:
        if observation.etag is None or observation.size_bytes is None:
            raise RemoteObjectPreconditionError("existing remote object lacks a bounded ETag/size observation")
    elif any(
        value is not None
        for value in (
            observation.etag,
            observation.content_hash,
            observation.size_bytes,
            observation.effect_id,
            observation.plan_hash,
            observation.protocol_version,
            observation.checksum_algorithm,
            observation.checksum_b64,
        )
    ):
        raise RemoteObjectPreconditionError("absent remote object carries identity evidence")
    if predecessor is not None:
        if predecessor.path_or_uri != target or predecessor.artifact_type != "file":
            raise RemoteObjectPreconditionError("predecessor descriptor does not identify the selected remote object")
        if not observation.exists:
            raise RemoteObjectPreconditionError("declared predecessor remote object is absent")
        if observation.content_hash != predecessor.content_hash or observation.size_bytes != predecessor.size_bytes:
            raise RemoteObjectPreconditionError("declared predecessor bytes do not match remote metadata")
        if observation.protocol_version != SINK_EFFECT_PROTOCOL_VERSION or observation.checksum_b64 is None:
            raise RemoteObjectPreconditionError("declared predecessor lacks exact effect protocol/checksum evidence")
        expected_checksum = "sha256" if provider == "aws_s3" else "md5" if provider == "azure_blob" else None
        if expected_checksum is None or observation.checksum_algorithm != expected_checksum:
            raise RemoteObjectPreconditionError("declared predecessor checksum algorithm diverges from provider policy")
    return SinkEffectInspection(
        mode=SinkEffectInspectionMode.INSPECTED,
        reference=target,
        evidence={
            "effect_id": request.effect_id,
            "observed_checksum_algorithm": observation.checksum_algorithm,
            "observed_checksum_b64": observation.checksum_b64,
            "observed_content_hash": observation.content_hash,
            "observed_etag": observation.etag,
            "observed_exists": observation.exists,
            "observed_protocol_version": observation.protocol_version,
            "observed_size": observation.size_bytes,
            "predecessor_declared": predecessor is not None,
            "provider": provider,
            "schema": _INSPECTION_SCHEMA,
            "target": target,
        },
    )


def _inspection_values(
    inspection: SinkEffectInspection,
    *,
    effect_id: str,
    provider: str,
) -> tuple[str, bool, str | None, bool]:
    evidence = inspection.evidence
    if evidence.get("schema") != _INSPECTION_SCHEMA or evidence.get("effect_id") != effect_id:
        raise RemoteObjectPreconditionError("remote inspection is missing or bound to another effect")
    if evidence.get("provider") != provider:
        raise RemoteObjectPreconditionError("remote inspection provider diverges")
    exists = evidence.get("observed_exists")
    predecessor_declared = evidence.get("predecessor_declared")
    if type(exists) is not bool or type(predecessor_declared) is not bool:
        raise RemoteObjectPreconditionError("remote inspection boolean evidence is malformed")
    etag = _optional_string(evidence.get("observed_etag"), "observed_etag")
    if exists != (etag is not None):
        raise RemoteObjectPreconditionError("remote inspection existence and ETag diverge")
    return _string(evidence.get("target"), "target"), exists, etag, predecessor_declared


def _spool_root() -> Path:
    configured = os.environ.get("ELSPETH_EFFECT_SPOOL_DIR")
    if configured:
        return Path(configured).resolve(strict=False)
    # Staged bodies are referenced by durable PREPARED plans, so the spool
    # must share the landscape DB's durability the way local-file sinks stage
    # next to their targets. A gettempdir() spool loses parked bodies to
    # reboots and /tmp age-cleaners while the plans that promise them survive.
    return Path(".elspeth", "sink-effect-spool").resolve(strict=False)


def _stage_path(effect_id: str, provider: str) -> Path:
    return _spool_root() / provider / f"{effect_id}.body"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# Crashed-write temp bodies in the code-owned provider spool: hidden name,
# ".body." infix, ".building" suffix — disjoint from the visible
# "{effect_id}.body" stage files the durable plans reference.
_BUILDING_GLOB: Final = ".*.body.*.building"
_BUILDING_STALE_SECONDS: Final = 60 * 60
_BUILDING_CLEANUP_LIMIT: Final = 16


def cleanup_stale_remote_spool_building_files(
    parent: Path,
    *,
    now: float | None = None,
    older_than_seconds: float = _BUILDING_STALE_SECONDS,
    max_entries: int = _BUILDING_CLEANUP_LIMIT,
) -> int:
    """Bound cleanup of crashed-write temp bodies; never scans outside parent.

    A building file only survives a hard crash between mkstemp and its atomic
    replace onto the stage name; every in-process failure unlinks it, and a
    re-drive restages under a fresh temp name, so anything old enough here is
    permanently orphaned.
    """
    if older_than_seconds < 0 or type(max_entries) is not int or max_entries < 0:
        raise ValueError("stale building cleanup bounds must be non-negative")
    current_time = time.time() if now is None else now
    removed = 0
    for path in sorted(parent.glob(_BUILDING_GLOB), key=lambda candidate: candidate.name):
        if removed >= max_entries:
            break
        try:
            result = path.lstat()
        except OSError:
            continue
        if not stat.S_ISREG(result.st_mode) or current_time - result.st_mtime < older_than_seconds:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    return removed


def _write_stage(
    *,
    path: Path,
    chunks: Iterable[bytes],
    max_bytes: int,
    checksum_algorithm: str,
) -> tuple[str, int, str]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("remote effect max_bytes must be a positive exact integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    cleanup_stale_remote_spool_building_files(path.parent)
    descriptor, building_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".building", dir=path.parent)
    building = Path(building_name)
    digest = hashlib.sha256()
    checksum = hashlib.sha256() if checksum_algorithm == "sha256" else hashlib.md5(usedforsecurity=False)
    total = 0
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for chunk in chunks:
                if type(chunk) is not bytes:
                    raise TypeError("remote effect body chunks must be exact bytes")
                total += len(chunk)
                if total > max_bytes:
                    raise RemoteObjectEffectLimitError(f"remote object byte limit exceeded: {total} > {max_bytes}")
                stream.write(chunk)
                digest.update(chunk)
                checksum.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(building, path)
        _fsync_directory(path.parent)
    except BaseException:
        building.unlink(missing_ok=True)
        raise
    return digest.hexdigest(), total, base64.b64encode(checksum.digest()).decode("ascii")


def _plan_hash(
    *,
    effect_id: str,
    descriptor_mode: SinkEffectDescriptorMode,
    descriptor: ArtifactDescriptor,
    evidence: Mapping[str, object],
) -> str:
    return stable_hash(
        {
            "descriptor": {
                "content_hash": descriptor.content_hash,
                "path_or_uri": descriptor.path_or_uri,
                "size_bytes": descriptor.size_bytes,
            },
            "descriptor_mode": descriptor_mode.value,
            "effect_id": effect_id,
            "evidence": dict(evidence),
            "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
            "schema": _EVIDENCE_SCHEMA,
        }
    )


def prepare_remote_object(
    *,
    effect_id: str,
    provider: str,
    inspection: SinkEffectInspection,
    body_chunks: Iterable[bytes],
    format_name: str,
    max_bytes: int,
    accepted_ordinals: Sequence[int],
    diverted_ordinals: Sequence[int],
    predecessor_descriptor: ArtifactDescriptor | None,
    checksum_algorithm: str,
    diversion_attribution: Sequence[DiversionAttribution] = (),
) -> SinkEffectPlan:
    """Persist an effect-addressed body and return a fresh-process-safe plan."""
    _lower_hex(effect_id, "effect_id")
    target, exists, etag, predecessor_declared = _inspection_values(inspection, effect_id=effect_id, provider=provider)
    if predecessor_declared != (predecessor_descriptor is not None):
        raise RemoteObjectPreconditionError("prepare predecessor diverges from inspection")
    path = _stage_path(effect_id, provider)
    algorithm = _checksum_algorithm(checksum_algorithm)
    accepted = tuple(accepted_ordinals)
    diverted = tuple(diverted_ordinals)
    # A zero-accepted effect without a declared predecessor publishes nothing:
    # staging an empty/header-only body here would conditionally REPLACE any
    # object already at the target. Existing targets stay untouched.
    virtual_no_publication = not accepted and predecessor_descriptor is None
    if virtual_no_publication:
        path.unlink(missing_ok=True)
        staged_hash = hashlib.sha256(b"").hexdigest()
        staged_size = 0
        checksum = hashlib.sha256(b"") if algorithm == "sha256" else hashlib.md5(b"", usedforsecurity=False)
        checksum_b64 = base64.b64encode(checksum.digest()).decode("ascii")
    else:
        staged_hash, staged_size, checksum_b64 = _write_stage(
            path=path,
            chunks=body_chunks,
            max_bytes=max_bytes,
            checksum_algorithm=algorithm,
        )
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri=target,
        content_hash=staged_hash,
        size_bytes=staged_size,
    )
    inherited = predecessor_descriptor == descriptor
    if virtual_no_publication:
        publication_kind = "virtual"
        descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
    elif inherited:
        publication_kind = "inherited"
        descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
        path.unlink(missing_ok=True)
    else:
        publication_kind = "conditional_replace" if exists else "conditional_create"
        descriptor_mode = SinkEffectDescriptorMode.PRECOMPUTED
    try:
        attribution = parse_diversion_attribution(
            diversion_attribution,
            diverted_ordinals=diverted,
        )
    except ValueError as exc:
        path.unlink(missing_ok=True)
        raise RemoteObjectPreconditionError(str(exc)) from exc
    value = RemoteObjectPlanEvidence(
        provider=provider,
        target=target,
        staging_path=str(path),
        precondition="if_match" if exists else "if_none_match",
        predecessor_etag=etag,
        staged_hash=staged_hash,
        staged_size=staged_size,
        checksum_algorithm=algorithm,
        checksum_b64=checksum_b64,
        format_name=format_name,
        accepted_ordinals=accepted,
        diverted_ordinals=diverted,
        diversion_attribution=attribution,
        publication_kind=publication_kind,
    )
    evidence = value.as_mapping()
    return SinkEffectPlan(
        effect_id=effect_id,
        protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        descriptor_mode=descriptor_mode,
        inspection_mode=SinkEffectInspectionMode.INSPECTED,
        target=target,
        plan_hash=_plan_hash(
            effect_id=effect_id,
            descriptor_mode=descriptor_mode,
            descriptor=descriptor,
            evidence=evidence,
        ),
        payload_hash=staged_hash,
        expected_descriptor=descriptor,
        safe_evidence=evidence,
    )


def validate_remote_plan(plan: SinkEffectPlan, *, provider: str, require_stage: bool) -> tuple[RemoteObjectPlanEvidence, Path]:
    if plan.protocol_version != SINK_EFFECT_PROTOCOL_VERSION or plan.input_kind is not SinkEffectInputKind.PIPELINE_MEMBERS:
        raise RemoteObjectPreconditionError("remote object plan protocol/input kind diverges")
    evidence = RemoteObjectPlanEvidence.from_mapping(plan.safe_evidence)
    if evidence.provider != provider or plan.target != evidence.target:
        raise RemoteObjectPreconditionError("remote object plan provider/target diverges")
    expected_stage = _stage_path(plan.effect_id, provider)
    stage = Path(evidence.staging_path).resolve(strict=False)
    if stage != expected_stage:
        raise RemoteObjectPreconditionError("remote object stage is not the configured effect-addressed path")
    if evidence.checksum_algorithm == "sha256":
        decoded_checksum = base64.b64decode(evidence.checksum_b64, validate=True)
        if decoded_checksum.hex() != evidence.staged_hash:
            raise RemoteObjectPreconditionError("remote object SHA-256 checksum diverges from payload hash")
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri=evidence.target,
        content_hash=evidence.staged_hash,
        size_bytes=evidence.staged_size,
    )
    if plan.expected_descriptor != descriptor or plan.payload_hash != evidence.staged_hash:
        raise RemoteObjectPreconditionError("remote object plan descriptor diverges")
    expected_hash = _plan_hash(
        effect_id=plan.effect_id,
        descriptor_mode=plan.descriptor_mode,
        descriptor=descriptor,
        evidence=evidence.as_mapping(),
    )
    if plan.plan_hash != expected_hash:
        raise RemoteObjectPreconditionError("remote object plan hash diverges")
    if require_stage:
        digest = hashlib.sha256()
        total = 0
        try:
            with stage.open("rb") as stream:
                for chunk in iter(lambda: stream.read(_COPY_BYTES), b""):
                    total += len(chunk)
                    digest.update(chunk)
        except OSError as exc:
            raise RemoteObjectPreconditionError("durable remote object body is unavailable") from exc
        if digest.hexdigest() != evidence.staged_hash or total != evidence.staged_size:
            raise RemoteObjectPreconditionError("durable remote object body diverges from the plan")
    return evidence, stage


def remote_stage_missing(plan: SinkEffectPlan, *, provider: str) -> bool:
    """Report whether the plan's effect-addressed staged body is absent."""
    _evidence, stage = validate_remote_plan(plan, provider=provider, require_stage=False)
    return not stage.is_file()


def restage_remote_object(
    plan: SinkEffectPlan,
    *,
    provider: str,
    body_chunks: Iterable[bytes],
    max_bytes: int,
    accepted_ordinals: Sequence[int],
    diverted_ordinals: Sequence[int],
) -> None:
    """Re-derive a lost staged body and verify it against the plan seal.

    The spool is repairable state: member payloads persist durably in the
    payload store and the plan seals the exact bytes it promised to publish
    (staged hash, size, and checksum), so a body lost to a reboot or a
    tmp-cleaner is rebuilt in place instead of wedging the stream on every
    re-drive. A divergent re-derivation fails closed before any provider I/O
    and leaves no stage behind.
    """
    evidence, stage = validate_remote_plan(plan, provider=provider, require_stage=False)
    if stage.is_file():
        return
    if tuple(accepted_ordinals) != evidence.accepted_ordinals or tuple(diverted_ordinals) != evidence.diverted_ordinals:
        raise RemoteObjectPreconditionError("re-derived member partition diverges from the durable plan")
    staged_hash, staged_size, checksum_b64 = _write_stage(
        path=stage,
        chunks=body_chunks,
        max_bytes=max_bytes,
        checksum_algorithm=evidence.checksum_algorithm,
    )
    if staged_hash != evidence.staged_hash or staged_size != evidence.staged_size or checksum_b64 != evidence.checksum_b64:
        stage.unlink(missing_ok=True)
        raise RemoteObjectPreconditionError("re-derived remote object body diverges from the durable plan")


def remote_commit_result(plan: SinkEffectPlan, evidence: RemoteObjectPlanEvidence) -> SinkEffectCommitResult:
    """Build the commit result and release the now-durable effect body spool.

    Callers invoke this only after the provider has durably confirmed the
    conditional write, so the effect-addressed stage has served its purpose
    and is removed to keep ordinary batches from accumulating spool files.
    """
    assert plan.expected_descriptor is not None
    Path(evidence.staging_path).unlink(missing_ok=True)
    return SinkEffectCommitResult(
        descriptor=plan.expected_descriptor,
        evidence={
            "effect_id": plan.effect_id,
            "plan_hash": plan.plan_hash,
            "provider": evidence.provider,
            "publication_kind": evidence.publication_kind,
        },
        accepted_ordinals=evidence.accepted_ordinals,
        diverted_ordinals=evidence.diverted_ordinals,
    )


def reconcile_remote_observation(
    plan: SinkEffectPlan,
    evidence: RemoteObjectPlanEvidence,
    observation: RemoteObjectObservation,
) -> SinkEffectReconcileResult:
    exact = (
        observation.exists
        and observation.content_hash == evidence.staged_hash
        and observation.size_bytes == evidence.staged_size
        and observation.effect_id == plan.effect_id
        and observation.plan_hash == plan.plan_hash
        and observation.protocol_version == SINK_EFFECT_PROTOCOL_VERSION
        and observation.checksum_algorithm == evidence.checksum_algorithm
        and observation.checksum_b64 == evidence.checksum_b64
    )
    if exact:
        assert plan.expected_descriptor is not None
        # The remote object carries this exact effect's identity, so the
        # durable body has been applied and its spool can be released. All
        # other outcomes keep the stage: NOT_APPLIED may still commit, and
        # UNKNOWN must preserve evidence for investigation.
        Path(evidence.staging_path).unlink(missing_ok=True)
        return SinkEffectReconcileResult.applied(
            plan.expected_descriptor,
            evidence={"effect_id": plan.effect_id, "provider": evidence.provider, "reconciled": "exact_metadata"},
            accepted_ordinals=evidence.accepted_ordinals,
            diverted_ordinals=evidence.diverted_ordinals,
        )
    if not observation.exists and evidence.precondition == "if_none_match":
        return SinkEffectReconcileResult.not_applied(
            evidence={"effect_id": plan.effect_id, "provider": evidence.provider, "reconciled": "still_absent"}
        )
    if observation.exists and evidence.precondition == "if_match" and observation.etag == evidence.predecessor_etag:
        return SinkEffectReconcileResult.not_applied(
            evidence={"effect_id": plan.effect_id, "provider": evidence.provider, "reconciled": "predecessor_unchanged"}
        )
    return SinkEffectReconcileResult.unknown(
        evidence={"effect_id": plan.effect_id, "provider": evidence.provider, "reconciled": "divergent_remote_state"}
    )


def iter_file_chunks(path: Path) -> Iterable[bytes]:
    with path.open("rb") as stream:
        while chunk := stream.read(_COPY_BYTES):
            yield chunk


__all__ = [
    "RemoteObjectEffectError",
    "RemoteObjectEffectLimitError",
    "RemoteObjectObservation",
    "RemoteObjectPlanEvidence",
    "RemoteObjectPreconditionError",
    "cleanup_stale_remote_spool_building_files",
    "inspect_remote_object",
    "iter_file_chunks",
    "prepare_remote_object",
    "reconcile_remote_observation",
    "remote_commit_result",
    "remote_stage_missing",
    "restage_remote_object",
    "validate_remote_plan",
]
