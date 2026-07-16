"""Bounded, recoverable publication primitives for local-file sinks.

The durable plan records both content identity and filesystem object identity.
Content equality alone cannot prove that an atomic replacement was performed by
the planned effect: an unrelated writer can produce identical bytes on another
inode.  Reconciliation therefore treats that case as UNKNOWN.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import tempfile
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final
from urllib.parse import unquote, urlsplit

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
    SinkEffectMember,
    SinkEffectPlan,
    SinkEffectReconcileResult,
)
from elspeth.plugins.sinks._diversion_attribution import DiversionAttribution, parse_diversion_attribution

DEFAULT_MAX_BYTES: Final = 1024 * 1024 * 1024
DEFAULT_MAX_ROWS: Final = 10_000_000
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 5.0
_COPY_CHUNK_BYTES: Final = 64 * 1024
_EVIDENCE_SCHEMA: Final = "local-file-effect-plan-v1"


class LocalFileEffectError(RuntimeError):
    """Base class for closed local publication failures."""


class LocalFileEffectLimitError(LocalFileEffectError):
    """Raised before publication when configured staging bounds are exceeded."""


class LocalFileLockTimeout(LocalFileEffectError):
    """Raised when the bounded target lock cannot be acquired."""


class LocalFileUnsupportedIdentity(LocalFileEffectError):
    """Raised when the filesystem cannot provide a stable local file ID."""


class LocalFilePreconditionError(LocalFileEffectError):
    """Raised when the durable plan no longer matches its pre-image."""


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    exists: bool
    content_hash: str | None
    size_bytes: int | None
    file_id: str | None


@dataclass(frozen=True, slots=True)
class LocalFileEffectPlanEvidence:
    """Credential-free evidence sufficient for commit and fresh-process recovery."""

    target_path: str
    staging_path: str
    lock_path: str
    predecessor_exists: bool
    predecessor_hash: str | None
    predecessor_size: int | None
    predecessor_file_id: str | None
    predecessor_declared: bool
    staged_hash: str
    staged_size: int
    staged_file_id: str
    encoding: str
    format_name: str
    stream_sequence: int
    publication_kind: str
    accepted_ordinals: tuple[int, ...]
    diverted_ordinals: tuple[int, ...]
    diversion_attribution: tuple[DiversionAttribution, ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "accepted_ordinals": list(self.accepted_ordinals),
            "diversion_attribution": [item.as_mapping() for item in self.diversion_attribution],
            "diverted_ordinals": list(self.diverted_ordinals),
            "encoding": self.encoding,
            "format_name": self.format_name,
            "lock_path": self.lock_path,
            "predecessor_declared": self.predecessor_declared,
            "predecessor_exists": self.predecessor_exists,
            "predecessor_file_id": self.predecessor_file_id,
            "predecessor_hash": self.predecessor_hash,
            "predecessor_size": self.predecessor_size,
            "publication_kind": self.publication_kind,
            "schema": _EVIDENCE_SCHEMA,
            "staged_file_id": self.staged_file_id,
            "staged_hash": self.staged_hash,
            "staged_size": self.staged_size,
            "staging_path": self.staging_path,
            "stream_sequence": self.stream_sequence,
            "target_path": self.target_path,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> LocalFileEffectPlanEvidence:
        if value.get("schema") != _EVIDENCE_SCHEMA:
            raise LocalFilePreconditionError("local-file plan evidence schema is missing or divergent")
        accepted = _exact_ordinals(value.get("accepted_ordinals"), "accepted_ordinals")
        diverted = _exact_ordinals(value.get("diverted_ordinals"), "diverted_ordinals")
        if set(accepted) & set(diverted):
            raise LocalFilePreconditionError("accepted and diverted ordinals overlap")
        try:
            diversion_attribution = parse_diversion_attribution(
                value.get("diversion_attribution"),
                diverted_ordinals=diverted,
            )
        except ValueError as exc:
            raise LocalFilePreconditionError(str(exc)) from exc
        return cls(
            target_path=_exact_string(value.get("target_path"), "target_path"),
            staging_path=_exact_string(value.get("staging_path"), "staging_path"),
            lock_path=_exact_string(value.get("lock_path"), "lock_path"),
            predecessor_exists=_exact_bool(value.get("predecessor_exists"), "predecessor_exists"),
            predecessor_hash=_optional_string(value.get("predecessor_hash"), "predecessor_hash"),
            predecessor_size=_optional_int(value.get("predecessor_size"), "predecessor_size"),
            predecessor_file_id=_optional_string(value.get("predecessor_file_id"), "predecessor_file_id"),
            predecessor_declared=_exact_bool(value.get("predecessor_declared"), "predecessor_declared"),
            staged_hash=_exact_string(value.get("staged_hash"), "staged_hash"),
            staged_size=_exact_int(value.get("staged_size"), "staged_size"),
            staged_file_id=_exact_string(value.get("staged_file_id"), "staged_file_id"),
            encoding=_exact_string(value.get("encoding"), "encoding"),
            format_name=_exact_string(value.get("format_name"), "format_name"),
            stream_sequence=_exact_int(value.get("stream_sequence"), "stream_sequence"),
            publication_kind=_exact_string(value.get("publication_kind"), "publication_kind"),
            accepted_ordinals=accepted,
            diverted_ordinals=diverted,
            diversion_attribution=diversion_attribution,
        )


def _exact_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise LocalFilePreconditionError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _exact_string(value, field_name)


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise LocalFilePreconditionError(f"{field_name} must be an exact bool")
    return value


def _exact_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise LocalFilePreconditionError(f"{field_name} must be a non-negative exact int")
    return value


def _optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _exact_int(value, field_name)


def _exact_ordinals(value: object, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise LocalFilePreconditionError(f"{field_name} must be an ordered sequence")
    result = tuple(_exact_int(item, field_name) for item in value)
    if len(result) != len(set(result)) or result != tuple(sorted(result)):
        raise LocalFilePreconditionError(f"{field_name} must contain unique ascending ordinals")
    return result


def _normalize_path(path: Path) -> Path:
    return path.resolve(strict=False)


def _stable_file_id(stat_result: os.stat_result) -> str:
    """Return a stable same-filesystem object ID or fail closed."""
    if type(stat_result.st_dev) is not int or type(stat_result.st_ino) is not int:
        raise LocalFileUnsupportedIdentity("filesystem stat does not expose integer device/inode identity")
    if stat_result.st_ino <= 0:
        raise LocalFileUnsupportedIdentity("filesystem does not expose a stable non-zero inode identity")
    return f"{stat_result.st_dev:x}:{stat_result.st_ino:x}"


def _snapshot(path: Path) -> _FileSnapshot:
    try:
        before = path.stat()
    except FileNotFoundError:
        return _FileSnapshot(False, None, None, None)
    if not path.is_file():
        raise LocalFilePreconditionError(f"local effect target is not a regular file: {path}")
    file_id = _stable_file_id(before)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    try:
        after = path.stat()
    except FileNotFoundError as exc:
        raise LocalFilePreconditionError("file disappeared while its identity was inspected") from exc
    if _stable_file_id(after) != file_id or after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
        raise LocalFilePreconditionError("file changed while its identity was inspected")
    return _FileSnapshot(True, digest.hexdigest(), after.st_size, file_id)


def _snapshot_mapping(snapshot: _FileSnapshot) -> dict[str, object]:
    return {
        "exists": snapshot.exists,
        "file_id": snapshot.file_id,
        "hash": snapshot.content_hash,
        "size": snapshot.size_bytes,
    }


def inspect_local_effect(*, target_path: Path, request: SinkEffectInspectionRequest) -> SinkEffectInspection:
    """Inspect and bind the exact pre-image without mutating the target."""
    target = _normalize_path(target_path)
    observed = _snapshot(target)
    predecessor = request.predecessor_descriptor
    if predecessor is not None:
        expected_uri = ArtifactDescriptor.for_file(
            path=str(target),
            content_hash=predecessor.content_hash,
            size_bytes=predecessor.size_bytes,
        ).path_or_uri
        if predecessor.artifact_type != "file" or predecessor.path_or_uri != expected_uri:
            raise LocalFilePreconditionError("declared predecessor does not identify the selected local target")
        if not observed.exists:
            raise LocalFilePreconditionError("declared predecessor target is absent")
        if observed.content_hash != predecessor.content_hash or observed.size_bytes != predecessor.size_bytes:
            raise LocalFilePreconditionError("declared predecessor bytes do not match the selected local target")
    return SinkEffectInspection(
        mode=SinkEffectInspectionMode.INSPECTED,
        reference=ArtifactDescriptor.for_file(
            path=str(target),
            content_hash=observed.content_hash or hashlib.sha256(b"").hexdigest(),
            size_bytes=observed.size_bytes or 0,
        ).path_or_uri,
        evidence={
            "effect_id": request.effect_id,
            "observed": _snapshot_mapping(observed),
            "predecessor_declared": predecessor is not None,
            "schema": "local-file-effect-inspection-v1",
            "target_path": str(target),
        },
    )


def predecessor_local_path(request: SinkEffectInspectionRequest) -> Path | None:
    """Recover the exact selected path from a prior local-file descriptor."""
    descriptor = request.predecessor_descriptor
    if descriptor is None:
        return None
    if descriptor.artifact_type != "file":
        raise LocalFilePreconditionError("local-file predecessor must be a file artifact")
    parsed = urlsplit(descriptor.path_or_uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise LocalFilePreconditionError("local-file predecessor must use a local file URI")
    return _normalize_path(Path(unquote(parsed.path)))


def _inspection_snapshot(inspection: SinkEffectInspection, *, effect_id: str) -> tuple[Path, _FileSnapshot, bool]:
    evidence = inspection.evidence
    if evidence.get("schema") != "local-file-effect-inspection-v1":
        raise LocalFilePreconditionError("inspection evidence schema is missing or divergent")
    if evidence.get("effect_id") != effect_id:
        raise LocalFilePreconditionError("inspection evidence is bound to a different effect")
    target = _normalize_path(Path(_exact_string(evidence.get("target_path"), "target_path")))
    observed = evidence.get("observed")
    if not isinstance(observed, Mapping):
        raise LocalFilePreconditionError("inspection observed evidence must be a mapping")
    exists = _exact_bool(observed.get("exists"), "observed.exists")
    snapshot = _FileSnapshot(
        exists=exists,
        content_hash=_optional_string(observed.get("hash"), "observed.hash"),
        size_bytes=_optional_int(observed.get("size"), "observed.size"),
        file_id=_optional_string(observed.get("file_id"), "observed.file_id"),
    )
    if exists and None in (snapshot.content_hash, snapshot.size_bytes, snapshot.file_id):
        raise LocalFilePreconditionError("existing inspection snapshot is incomplete")
    if not exists and any(item is not None for item in (snapshot.content_hash, snapshot.size_bytes, snapshot.file_id)):
        raise LocalFilePreconditionError("absent inspection snapshot carries file identity")
    return target, snapshot, _exact_bool(evidence.get("predecessor_declared"), "predecessor_declared")


def _staging_path(target: Path, effect_id: str) -> Path:
    return target.with_name(f".{target.name}.elspeth-{effect_id}.stage")


def _lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.elspeth.lock")


def _plan_hash(
    *,
    effect_id: str,
    input_kind: SinkEffectInputKind,
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
            "effect_id": effect_id,
            "evidence": dict(evidence),
            "input_kind": input_kind.value,
            "descriptor_mode": descriptor_mode.value,
            "schema": _EVIDENCE_SCHEMA,
        }
    )


def _remove_exact_staging(path: Path, snapshot: _FileSnapshot) -> None:
    """Best-effort cleanup for a prepare that cannot produce a durable plan."""
    try:
        current = _snapshot(path)
        if current == snapshot:
            path.unlink()
            _fsync_directory(path.parent)
    except (LocalFileEffectError, OSError):
        return


def prepare_local_effect(
    *,
    effect_id: str,
    input_kind: SinkEffectInputKind,
    inspection: SinkEffectInspection,
    chunks: Iterable[bytes],
    row_count: int,
    accepted_ordinals: Sequence[int] | Callable[[], Sequence[int]],
    diverted_ordinals: Sequence[int] | Callable[[], Sequence[int]],
    encoding: str,
    format_name: str,
    stream_sequence: int,
    diversion_attribution: Sequence[DiversionAttribution] | Callable[[], Sequence[DiversionAttribution]] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> SinkEffectPlan:
    """Stream an exact candidate into a same-directory effect-addressed file."""
    if len(effect_id) != 64 or any(character not in "0123456789abcdef" for character in effect_id):
        raise ValueError("local-file effect_id must be a lowercase SHA-256 digest")
    if type(row_count) is not int or row_count < 0:
        raise TypeError("row_count must be a non-negative exact int")
    if type(max_bytes) is not int or max_bytes < 1 or type(max_rows) is not int or max_rows < 1:
        raise ValueError("local effect bounds must be positive exact integers")
    if type(stream_sequence) is not int or stream_sequence < 0:
        raise ValueError("stream_sequence must be a non-negative exact int")
    if type(encoding) is not str or not encoding or type(format_name) is not str or not format_name:
        raise ValueError("encoding and format_name must be non-empty exact strings")
    if row_count > max_rows:
        raise LocalFileEffectLimitError(f"row limit exceeded: {row_count} > {max_rows}")
    target, predecessor, predecessor_declared = _inspection_snapshot(inspection, effect_id=effect_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = _staging_path(target, effect_id)
    lock = _lock_path(target)
    descriptor_fd: int | None = None
    building: Path | None = None
    staged: _FileSnapshot | None = None
    try:
        descriptor_fd, building_name = tempfile.mkstemp(prefix=f".{staging.name}.", suffix=".building", dir=target.parent)
        building = Path(building_name)
        digest = hashlib.sha256()
        total = 0
        stream: IO[bytes] = os.fdopen(descriptor_fd, "wb")
        descriptor_fd = None
        with stream:
            for chunk in chunks:
                if type(chunk) is not bytes:
                    raise TypeError("local effect chunks must be exact bytes")
                total += len(chunk)
                if total > max_bytes:
                    raise LocalFileEffectLimitError(f"byte limit exceeded: {total} > {max_bytes}")
                stream.write(chunk)
                digest.update(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(building, staging)
        building = None
        staged = _snapshot(staging)
        if not staged.exists or staged.content_hash != digest.hexdigest() or staged.size_bytes != total or staged.file_id is None:
            raise LocalFilePreconditionError("staged file identity diverged after atomic installation")
        _fsync_directory(target.parent)
    except BaseException:
        if descriptor_fd is not None:
            os.close(descriptor_fd)
        if building is not None:
            building.unlink(missing_ok=True)
        if staged is not None:
            _remove_exact_staging(staging, staged)
        raise

    assert staged.content_hash is not None and staged.size_bytes is not None and staged.file_id is not None

    try:
        accepted = tuple(accepted_ordinals() if callable(accepted_ordinals) else accepted_ordinals)
        diverted = tuple(diverted_ordinals() if callable(diverted_ordinals) else diverted_ordinals)
        attribution_value = tuple(diversion_attribution() if callable(diversion_attribution) else diversion_attribution)
    except BaseException:
        _remove_exact_staging(staging, staged)
        raise
    if input_kind is SinkEffectInputKind.PIPELINE_MEMBERS:
        partition = tuple(sorted((*accepted, *diverted)))
        if partition != tuple(range(row_count)):
            _remove_exact_staging(staging, staged)
            raise LocalFilePreconditionError("pipeline accepted/diverted ordinals must exactly partition every input row")
    elif accepted or diverted:
        _remove_exact_staging(staging, staged)
        raise LocalFilePreconditionError("audit-export effects cannot carry pipeline member dispositions")
    try:
        attribution = parse_diversion_attribution(attribution_value, diverted_ordinals=diverted)
    except ValueError as exc:
        _remove_exact_staging(staging, staged)
        raise LocalFilePreconditionError(str(exc)) from exc
    descriptor_hash = staged.content_hash
    descriptor_size = staged.size_bytes
    if input_kind is SinkEffectInputKind.PIPELINE_MEMBERS and not accepted:
        descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
        if predecessor.exists:
            if predecessor.content_hash is None or predecessor.size_bytes is None:
                _remove_exact_staging(staging, staged)
                raise LocalFilePreconditionError("existing no-publication predecessor has incomplete descriptor evidence")
            publication_kind = "inherited"
            descriptor_hash = predecessor.content_hash
            descriptor_size = predecessor.size_bytes
        else:
            publication_kind = "virtual"
            descriptor_hash = hashlib.sha256(b"").hexdigest()
            descriptor_size = 0
    elif (
        not diverted
        and predecessor.exists
        and predecessor.content_hash == staged.content_hash
        and predecessor.size_bytes == staged.size_bytes
    ):
        publication_kind = "inherited"
        descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
    elif not diverted and not predecessor.exists and staged.size_bytes == 0:
        publication_kind = "virtual"
        descriptor_mode = SinkEffectDescriptorMode.NO_PUBLICATION
    else:
        publication_kind = "atomic_replace"
        descriptor_mode = SinkEffectDescriptorMode.PRECOMPUTED

    evidence_value = LocalFileEffectPlanEvidence(
        target_path=str(target),
        staging_path=str(staging),
        lock_path=str(lock),
        predecessor_exists=predecessor.exists,
        predecessor_hash=predecessor.content_hash,
        predecessor_size=predecessor.size_bytes,
        predecessor_file_id=predecessor.file_id,
        predecessor_declared=predecessor_declared,
        staged_hash=descriptor_hash,
        staged_size=descriptor_size,
        staged_file_id=staged.file_id,
        encoding=encoding,
        format_name=format_name,
        stream_sequence=stream_sequence,
        publication_kind=publication_kind,
        accepted_ordinals=accepted,
        diverted_ordinals=diverted,
        diversion_attribution=attribution,
    )
    evidence = evidence_value.as_mapping()
    descriptor = ArtifactDescriptor.for_file(path=str(target), content_hash=descriptor_hash, size_bytes=descriptor_size)
    if descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION:
        staging.unlink()
        _fsync_directory(staging.parent)
    try:
        return SinkEffectPlan(
            effect_id=effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=input_kind,
            descriptor_mode=descriptor_mode,
            inspection_mode=inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=_plan_hash(
                effect_id=effect_id,
                input_kind=input_kind,
                descriptor_mode=descriptor_mode,
                descriptor=descriptor,
                evidence=evidence,
            ),
            payload_hash=descriptor_hash,
            expected_descriptor=descriptor,
            safe_evidence=evidence,
        )
    except BaseException:
        _remove_exact_staging(staging, staged)
        raise


def _parse_plan(plan: SinkEffectPlan) -> tuple[LocalFileEffectPlanEvidence, Path, Path, Path, ArtifactDescriptor]:
    if (
        plan.descriptor_mode
        not in {
            SinkEffectDescriptorMode.PRECOMPUTED,
            SinkEffectDescriptorMode.NO_PUBLICATION,
        }
        or plan.expected_descriptor is None
    ):
        raise LocalFilePreconditionError("local-file effect requires a precomputed exact descriptor")
    evidence = LocalFileEffectPlanEvidence.from_mapping(plan.safe_evidence)
    target = _normalize_path(Path(evidence.target_path))
    staging = _normalize_path(Path(evidence.staging_path))
    lock = _normalize_path(Path(evidence.lock_path))
    if staging != _staging_path(target, plan.effect_id) or lock != _lock_path(target):
        raise LocalFilePreconditionError("local-file plan paths are not effect-addressed in the target directory")
    expected = ArtifactDescriptor.for_file(path=str(target), content_hash=evidence.staged_hash, size_bytes=evidence.staged_size)
    if plan.target != expected.path_or_uri or plan.expected_descriptor != expected or plan.payload_hash != evidence.staged_hash:
        raise LocalFilePreconditionError("local-file plan descriptor diverges from staged evidence")
    expected_hash = _plan_hash(
        effect_id=plan.effect_id,
        input_kind=plan.input_kind,
        descriptor_mode=plan.descriptor_mode,
        descriptor=expected,
        evidence=evidence.as_mapping(),
    )
    if plan.plan_hash != expected_hash:
        raise LocalFilePreconditionError("local-file plan hash does not bind its exact evidence")
    return evidence, target, staging, lock, expected


def _matches(snapshot: _FileSnapshot, *, exists: bool, content_hash: str | None, size: int | None, file_id: str | None) -> bool:
    return (
        snapshot.exists is exists and snapshot.content_hash == content_hash and snapshot.size_bytes == size and snapshot.file_id == file_id
    )


@contextmanager
def _bounded_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    if timeout_seconds < 0:
        raise ValueError("lock timeout must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LocalFileLockTimeout(f"timed out acquiring local effect lock {path}") from None
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _after_replace(_target: Path) -> None:
    """Crash-test seam after the observable rename and before return."""


def commit_local_effect(
    plan: SinkEffectPlan,
    *,
    lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> SinkEffectCommitResult:
    """CAS the target from its inspected pre-image to the exact staged inode."""
    evidence, target, staging, lock, expected = _parse_plan(plan)
    if plan.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION:
        raise LocalFilePreconditionError("no-publication local effects must be finalized without commit")
    with _bounded_lock(lock, lock_timeout_seconds):
        current = _snapshot(target)
        if not _matches(
            current,
            exists=evidence.predecessor_exists,
            content_hash=evidence.predecessor_hash,
            size=evidence.predecessor_size,
            file_id=evidence.predecessor_file_id,
        ):
            raise LocalFilePreconditionError("target pre-image no longer matches the durable local-file plan")
        candidate = _snapshot(staging)
        if not _matches(
            candidate,
            exists=True,
            content_hash=evidence.staged_hash,
            size=evidence.staged_size,
            file_id=evidence.staged_file_id,
        ):
            raise LocalFilePreconditionError("staging hash, size, path, or file identity no longer matches the durable plan")
        with staging.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, target)
        _after_replace(target)
        _fsync_directory(target.parent)
        published = _snapshot(target)
        if not _matches(
            published,
            exists=True,
            content_hash=evidence.staged_hash,
            size=evidence.staged_size,
            file_id=evidence.staged_file_id,
        ):
            raise LocalFilePreconditionError("published target does not retain the planned staged file identity")
    return SinkEffectCommitResult(
        descriptor=expected,
        evidence={"file_id": evidence.staged_file_id, "publication": "atomic_replace"},
        accepted_ordinals=evidence.accepted_ordinals,
        diverted_ordinals=evidence.diverted_ordinals,
    )


def reconcile_local_effect(plan: SinkEffectPlan) -> SinkEffectReconcileResult:
    """Classify target state using the closed NOT_APPLIED/APPLIED/UNKNOWN set."""
    try:
        evidence, target, staging, _lock, expected = _parse_plan(plan)
        current = _snapshot(target)
        candidate = _snapshot(staging)
    except (LocalFileEffectError, OSError) as exc:
        return SinkEffectReconcileResult.unknown(evidence={"reason": type(exc).__name__})

    applied = _matches(
        current,
        exists=True,
        content_hash=evidence.staged_hash,
        size=evidence.staged_size,
        file_id=evidence.staged_file_id,
    )
    if applied and not candidate.exists:
        return SinkEffectReconcileResult.applied(
            expected,
            evidence={"file_id": evidence.staged_file_id, "publication": "atomic_replace"},
        )

    candidate_exact = _matches(
        candidate,
        exists=True,
        content_hash=evidence.staged_hash,
        size=evidence.staged_size,
        file_id=evidence.staged_file_id,
    )
    predecessor_exact = _matches(
        current,
        exists=evidence.predecessor_exists,
        content_hash=evidence.predecessor_hash,
        size=evidence.predecessor_size,
        file_id=evidence.predecessor_file_id,
    )
    if candidate_exact and predecessor_exact:
        return SinkEffectReconcileResult.not_applied(evidence={"staging": "exact", "target": "predecessor"})
    return SinkEffectReconcileResult.unknown(evidence={"staging": "divergent", "target": "ambiguous"})


def cleanup_local_effect(plan: SinkEffectPlan) -> bool:
    """Remove only the exact still-staged inode named by a durable plan."""
    try:
        evidence, _target, staging, _lock, _expected = _parse_plan(plan)
        candidate = _snapshot(staging)
    except (LocalFileEffectError, OSError):
        return False
    if not _matches(
        candidate,
        exists=True,
        content_hash=evidence.staged_hash,
        size=evidence.staged_size,
        file_id=evidence.staged_file_id,
    ):
        return False
    staging.unlink()
    _fsync_directory(staging.parent)
    return True


def continuation_emission(
    *,
    append_mode: bool,
    predecessor_declared: bool,
    current_member_effect_ids: Collection[str | None],
    target_snapshot_members: Sequence[SinkEffectMember],
) -> tuple[bool, tuple[SinkEffectMember, ...]]:
    """Select baseline inclusion and the members one rebuild must serialize.

    Append-mode rebuilds treat the target file as the durable prefix: at the
    first effect it holds pre-run content, at successors it holds the
    predecessor's exact output (pinned by inspection and the commit CAS), so
    only current members are re-serialized behind it. Re-serializing
    predecessor snapshot members instead would drop the pre-run bytes, which
    member rows cannot represent. Replace-mode rebuilds serialize the full
    cumulative snapshot and never include file bytes.
    """
    if not append_mode:
        return False, tuple(target_snapshot_members)
    if not predecessor_declared:
        return True, tuple(target_snapshot_members)
    return True, tuple(member for member in target_snapshot_members if member.member_effect_id in current_member_effect_ids)


def iter_path_chunks(path: Path) -> Iterator[bytes]:
    """Yield bounded chunks from a baseline file without retaining it in memory."""
    with path.open("rb") as stream:
        yield from iter(lambda: stream.read(_COPY_CHUNK_BYTES), b"")


__all__ = [
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ROWS",
    "LocalFileEffectError",
    "LocalFileEffectLimitError",
    "LocalFileEffectPlanEvidence",
    "LocalFileLockTimeout",
    "LocalFilePreconditionError",
    "LocalFileUnsupportedIdentity",
    "cleanup_local_effect",
    "commit_local_effect",
    "continuation_emission",
    "inspect_local_effect",
    "iter_path_chunks",
    "predecessor_local_path",
    "prepare_local_effect",
    "reconcile_local_effect",
]
