"""Recoverable publication primitives for audit-export CSV directory bundles.

The bundle is create-only.  Preparation builds and fsyncs an effect-addressed
sibling directory; commit publishes it with Linux ``renameat2`` and
``RENAME_NOREPLACE``.  Recovery credits only an exact tree bound by the durable
plan.  A pre-existing divergent tree is never replaced.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryFile
from typing import IO, Final, cast
from uuid import uuid4

from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    AuditExportFormat,
    SinkEffectAuditExportSnapshotInput,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.core.landscape.formatters import CSVFormatter

AUDIT_MANIFEST_NAME: Final = "audit_manifest.v2.json"
BUNDLE_MANIFEST_SCHEMA: Final = "elspeth.audit-export-directory-bundle.v1"
BUNDLE_EVIDENCE_SCHEMA: Final = "audit-export-directory-bundle-plan-v1"
INSPECTION_SCHEMA: Final = "audit-export-directory-bundle-inspection-v1"
MAX_BUNDLE_FILES: Final = 96
MAX_RECORD_TYPE_BYTES: Final = 192
MAX_BUNDLE_BYTES: Final = 1024 * 1024 * 1024 * 1024
_COPY_CHUNK_BYTES: Final = 64 * 1024
_AT_FDCWD: Final = -100
_RENAME_NOREPLACE: Final = 1
_PROBE_PREFIX: Final = ".elspeth-bundle-probe-"
# Crashed-prepare building trees; the trailing ".building-" segment keeps the
# glob disjoint from effect-addressed ".bundle-stage" staging trees.
_BUILDING_GLOB: Final = ".*.elspeth-*.building-*"
_PROBE_STALE_SECONDS: Final = 60 * 60
_PROBE_CLEANUP_LIMIT: Final = 16
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_RECORD_TYPE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CSV_FORMULA_PREFIXES: Final = ("=", "+", "-", "@", "\t", "\r", "\n")

# Explicit Linux local-filesystem allowlist.  Network/distributed filesystems
# are intentionally absent because their rename/durability semantics differ.
SUPPORTED_LOCAL_FILESYSTEM_MAGIC: Final[frozenset[int]] = frozenset(
    {
        0xEF53,  # ext2/ext3/ext4
        0x58465342,  # XFS
        0x9123683E,  # Btrfs
        0xF2F52010,  # F2FS
        0x3153464A,  # JFS
        0x2FC12FC1,  # ZFS
        0x01021994,  # tmpfs
        0x858458F6,  # ramfs
        0x794C7630,  # overlayfs (container-local upper layer)
    }
)


class AuditExportBundleError(RuntimeError):
    """Base class for closed bundle-publication failures."""


class AuditExportBundleInputError(AuditExportBundleError):
    """The registered snapshot cannot be represented as an exact CSV bundle."""


class AuditExportBundlePreflightError(AuditExportBundleError):
    """The local platform cannot prove the required publication semantics."""


class AuditExportBundlePreconditionError(AuditExportBundleError):
    """A durable plan or its private staged tree is divergent."""


class AuditExportBundleCollisionError(AuditExportBundleError):
    """The create-only target exists but is not the planned exact tree."""


@dataclass(frozen=True, slots=True)
class AuditExportBundlePreflight:
    target_path: str
    filesystem_magic: int
    device_id: int


@dataclass(frozen=True, slots=True)
class BundleFileEntry:
    relative_path: str
    content_hash: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_name(self.relative_path)
        if _LOWER_HEX_64.fullmatch(self.content_hash) is None:
            raise AuditExportBundlePreconditionError("bundle file content_hash must be lowercase SHA-256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise AuditExportBundlePreconditionError("bundle file size_bytes must be a non-negative exact int")

    def as_mapping(self) -> dict[str, object]:
        return {
            "content_hash": self.content_hash,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_mapping(cls, value: object) -> BundleFileEntry:
        mapping = _closed_mapping(value, {"content_hash", "relative_path", "size_bytes"}, "bundle file entry")
        return cls(
            relative_path=_exact_string(mapping["relative_path"], "relative_path"),
            content_hash=_exact_string(mapping["content_hash"], "content_hash"),
            size_bytes=_non_negative_int(mapping["size_bytes"], "size_bytes"),
        )


@dataclass(frozen=True, slots=True)
class BundlePlanEvidence:
    target_path: str
    staging_path: str
    snapshot_id: str
    manifest_content_ref: str
    manifest_content_hash: str
    manifest_size_bytes: int
    bundle_hash: str
    bundle_size_bytes: int
    files: tuple[BundleFileEntry, ...]

    def __post_init__(self) -> None:
        if _LOWER_HEX_64.fullmatch(self.snapshot_id) is None:
            raise AuditExportBundlePreconditionError("snapshot_id must be lowercase SHA-256")
        if self.manifest_content_ref != f"sha256:{self.manifest_content_hash}":
            raise AuditExportBundlePreconditionError("manifest content ref/hash binding is divergent")
        if _LOWER_HEX_64.fullmatch(self.manifest_content_hash) is None:
            raise AuditExportBundlePreconditionError("manifest content hash must be lowercase SHA-256")
        if type(self.manifest_size_bytes) is not int or self.manifest_size_bytes < 1:
            raise AuditExportBundlePreconditionError("manifest size must be a positive exact int")
        if _LOWER_HEX_64.fullmatch(self.bundle_hash) is None:
            raise AuditExportBundlePreconditionError("bundle_hash must be lowercase SHA-256")
        if type(self.bundle_size_bytes) is not int or not (1 <= self.bundle_size_bytes <= MAX_BUNDLE_BYTES):
            raise AuditExportBundlePreconditionError("bundle size exceeds the closed bound")
        if not (2 <= len(self.files) <= MAX_BUNDLE_FILES):
            raise AuditExportBundlePreconditionError("bundle requires the manifest plus one or more bounded CSV files")
        names = tuple(entry.relative_path for entry in self.files)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise AuditExportBundlePreconditionError("bundle file manifest must be unique and sorted")
        folded = tuple(name.casefold() for name in names)
        if len(folded) != len(set(folded)):
            raise AuditExportBundlePreconditionError("bundle file manifest contains a case-fold collision")
        manifest_entries = [entry for entry in self.files if entry.relative_path == AUDIT_MANIFEST_NAME]
        if len(manifest_entries) != 1:
            raise AuditExportBundlePreconditionError("bundle must contain exactly one reserved audit manifest")
        manifest_entry = manifest_entries[0]
        if manifest_entry.content_hash != self.manifest_content_hash or manifest_entry.size_bytes != self.manifest_size_bytes:
            raise AuditExportBundlePreconditionError("reserved audit manifest entry is divergent")
        if self.bundle_size_bytes != sum(entry.size_bytes for entry in self.files):
            raise AuditExportBundlePreconditionError("bundle size does not equal the exact file-size sum")
        if self.bundle_hash != _bundle_hash(self.files):
            raise AuditExportBundlePreconditionError("bundle hash does not bind the canonical file manifest")

    def as_mapping(self) -> dict[str, object]:
        return {
            "bundle_hash": self.bundle_hash,
            "bundle_size_bytes": self.bundle_size_bytes,
            "files": [entry.as_mapping() for entry in self.files],
            "manifest_content_hash": self.manifest_content_hash,
            "manifest_content_ref": self.manifest_content_ref,
            "manifest_size_bytes": self.manifest_size_bytes,
            "schema": BUNDLE_EVIDENCE_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "staging_path": self.staging_path,
            "target_path": self.target_path,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> BundlePlanEvidence:
        fields = {
            "bundle_hash",
            "bundle_size_bytes",
            "files",
            "manifest_content_hash",
            "manifest_content_ref",
            "manifest_size_bytes",
            "schema",
            "snapshot_id",
            "staging_path",
            "target_path",
        }
        mapping = _closed_mapping(value, fields, "bundle plan evidence")
        if mapping["schema"] != BUNDLE_EVIDENCE_SCHEMA:
            raise AuditExportBundlePreconditionError("bundle plan evidence schema is divergent")
        raw_files = mapping["files"]
        if not isinstance(raw_files, Sequence) or isinstance(raw_files, (str, bytes, bytearray)):
            raise AuditExportBundlePreconditionError("bundle files must be an ordered sequence")
        files = tuple(BundleFileEntry.from_mapping(entry) for entry in raw_files)
        return cls(
            target_path=_exact_string(mapping["target_path"], "target_path"),
            staging_path=_exact_string(mapping["staging_path"], "staging_path"),
            snapshot_id=_exact_string(mapping["snapshot_id"], "snapshot_id"),
            manifest_content_ref=_exact_string(mapping["manifest_content_ref"], "manifest_content_ref"),
            manifest_content_hash=_exact_string(mapping["manifest_content_hash"], "manifest_content_hash"),
            manifest_size_bytes=_non_negative_int(mapping["manifest_size_bytes"], "manifest_size_bytes"),
            bundle_hash=_exact_string(mapping["bundle_hash"], "bundle_hash"),
            bundle_size_bytes=_non_negative_int(mapping["bundle_size_bytes"], "bundle_size_bytes"),
            files=files,
        )


@dataclass(slots=True)
class _RecordSpool:
    relative_path: str
    stream: IO[str]
    fieldnames: set[str]


def _closed_mapping(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise AuditExportBundlePreconditionError(f"{label} must contain the exact closed field set")
    if any(type(key) is not str for key in value):
        raise AuditExportBundlePreconditionError(f"{label} keys must be exact strings")
    return cast(Mapping[str, object], value)


def _exact_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise AuditExportBundlePreconditionError(f"{field_name} must be a non-empty exact string")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise AuditExportBundlePreconditionError(f"{field_name} must be a non-negative exact int")
    return value


def _validate_effect_id(effect_id: str) -> None:
    if type(effect_id) is not str or _LOWER_HEX_64.fullmatch(effect_id) is None:
        raise AuditExportBundleInputError("bundle effect_id must be a lowercase SHA-256 digest")


def _validate_relative_name(name: str) -> None:
    if type(name) is not str or not name or name in {".", ".."} or "/" in name or "\\" in name or "\x00" in name or Path(name).name != name:
        raise AuditExportBundlePreconditionError("bundle entries must be safe single-component relative paths")


def _absolute_path(path: Path) -> Path:
    if not isinstance(path, Path):
        raise TypeError("target_path must be pathlib.Path")
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _staging_path(target: Path, effect_id: str) -> Path:
    return target.with_name(f".{target.name}.elspeth-{effect_id}.bundle-stage")


def _bundle_manifest(files: Sequence[BundleFileEntry]) -> dict[str, object]:
    return {
        "files": [entry.as_mapping() for entry in files],
        "schema": BUNDLE_MANIFEST_SCHEMA,
    }


def _bundle_hash(files: Sequence[BundleFileEntry]) -> str:
    return hashlib.sha256(canonical_json(_bundle_manifest(files)).encode("utf-8")).hexdigest()


def _plan_hash(
    *,
    effect_id: str,
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
            "input_kind": SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT.value,
            "schema": BUNDLE_EVIDENCE_SCHEMA,
        }
    )


def _assert_no_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            result = current.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(result.st_mode):
            raise AuditExportBundlePreflightError(f"bundle path contains symlink component: {current}")


def _target_kind(target: Path) -> str:
    try:
        result = target.lstat()
    except FileNotFoundError:
        return "absent"
    if stat.S_ISLNK(result.st_mode):
        return "symlink"
    if stat.S_ISDIR(result.st_mode):
        return "directory"
    return "other"


def inspect_audit_export_bundle(
    *,
    target_path: Path,
    request: SinkEffectInspectionRequest,
) -> SinkEffectInspection:
    """Inspect the create-only target without enumerating snapshot content."""
    _validate_effect_id(request.effect_id)
    if request.predecessor_descriptor is not None:
        raise AuditExportBundleInputError("create-only audit-export bundles do not accept predecessor artifacts")
    target = _absolute_path(target_path)
    if not target.name:
        raise AuditExportBundleInputError("bundle target must name a directory beneath an existing parent")
    _assert_no_symlink_components(target)
    reference = ArtifactDescriptor.for_file(
        path=str(target),
        content_hash=hashlib.sha256(b"").hexdigest(),
        size_bytes=0,
    ).path_or_uri
    return SinkEffectInspection(
        mode=SinkEffectInspectionMode.INSPECTED,
        reference=reference,
        evidence={
            "effect_id": request.effect_id,
            "schema": INSPECTION_SCHEMA,
            "target_kind": _target_kind(target),
            "target_path": str(target),
        },
    )


def _inspection_target(request: SinkEffectPrepareRequest, target_path: Path) -> Path:
    evidence = request.inspection.evidence
    if set(evidence) != {"effect_id", "schema", "target_kind", "target_path"}:
        raise AuditExportBundleInputError("bundle inspection evidence is not closed")
    if evidence.get("schema") != INSPECTION_SCHEMA or evidence.get("effect_id") != request.effect_id:
        raise AuditExportBundleInputError("bundle inspection evidence is divergent")
    target = _absolute_path(target_path)
    if evidence.get("target_path") != str(target):
        raise AuditExportBundleInputError("bundle prepare target differs from inspection")
    return target


def _csv_relative_path(record_type: object, seen: dict[str, str]) -> str:
    if type(record_type) is not str or not record_type:
        raise AuditExportBundleInputError("every audit-export data record requires a non-empty record_type")
    if len(record_type.encode("utf-8")) > MAX_RECORD_TYPE_BYTES or _RECORD_TYPE.fullmatch(record_type) is None:
        raise AuditExportBundleInputError("record_type cannot form a safe bounded CSV filename")
    if record_type.casefold() in {"manifest", AUDIT_MANIFEST_NAME.casefold()}:
        raise AuditExportBundleInputError("data records cannot claim the reserved final audit manifest")
    relative_path = f"{record_type}.csv"
    _validate_relative_name(relative_path)
    if relative_path.casefold() == AUDIT_MANIFEST_NAME.casefold():
        raise AuditExportBundleInputError("generated CSV filename collides with the reserved audit manifest")
    prior = seen.get(relative_path.casefold())
    if prior is not None and prior != relative_path:
        raise AuditExportBundleInputError("generated CSV filenames contain a case-fold collision")
    seen[relative_path.casefold()] = relative_path
    return relative_path


def _neutralize_csv_formula(value: object) -> object:
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return f"'{value}"
    return value


def _parse_verified_records(
    effect_input: SinkEffectAuditExportSnapshotInput,
) -> tuple[dict[str, _RecordSpool], bytes]:
    formatter = CSVFormatter()
    spools: dict[str, _RecordSpool] = {}
    seen_names: dict[str, str] = {}
    observed_records = 0
    try:
        for chunk in effect_input.reader.iter_verified_chunks():
            if type(chunk) is not bytes or not chunk.endswith(b"\n"):
                raise AuditExportBundleInputError("verified data chunks must contain complete newline-framed records")
            frames = chunk.split(b"\n")
            if frames[-1] != b"":
                raise AuditExportBundleInputError("verified data chunk has a non-final record frame")
            for frame in frames[:-1]:
                if not frame:
                    raise AuditExportBundleInputError("verified data chunk contains an empty record frame")
                try:
                    value = json.loads(frame)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise AuditExportBundleInputError("verified data chunk contains invalid JSON") from exc
                if type(value) is not dict or canonical_json(value).encode("utf-8") != frame:
                    raise AuditExportBundleInputError("verified data records must be exact canonical JSON objects")
                record = cast(dict[str, object], value)
                relative_path = _csv_relative_path(record.get("record_type"), seen_names)
                flattened = formatter.format(record)
                safe_record = {key: _neutralize_csv_formula(item) for key, item in flattened.items()}
                spool = spools.get(relative_path)
                if spool is None:
                    if len(spools) >= MAX_BUNDLE_FILES - 1:
                        raise AuditExportBundleInputError("CSV record types exceed the bounded bundle-file limit")
                    # The bundle owns this spool across the complete verified
                    # reader pass and closes it in the outer finally block.
                    stream = TemporaryFile("w+", encoding="utf-8", newline="\n")  # noqa: SIM115
                    spool = _RecordSpool(relative_path=relative_path, stream=stream, fieldnames=set())
                    spools[relative_path] = spool
                spool.fieldnames.update(safe_record)
                spool.stream.write(canonical_json(safe_record))
                spool.stream.write("\n")
                observed_records += 1
        if observed_records != effect_input.record_count:
            raise AuditExportBundleInputError("CSV bundle observed record count differs from the registered snapshot")
        if not spools:
            raise AuditExportBundleInputError("CSV audit export requires at least one data record type")
        try:
            manifest_bytes = effect_input.reader.read_verified_signed_manifest()
        except Exception as exc:
            raise AuditExportBundleInputError("final audit manifest is missing or failed verification") from exc
        if manifest_bytes.endswith(b"\n"):
            raise AuditExportBundleInputError("final audit manifest must not carry a trailing newline")
        try:
            manifest = json.loads(manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditExportBundleInputError("final audit manifest is not valid JSON") from exc
        if (
            type(manifest) is not dict
            or manifest.get("record_type") != "manifest"
            or manifest.get("schema") != "elspeth.audit-export-manifest.v2"
            or canonical_json(manifest).encode("utf-8") != manifest_bytes
        ):
            raise AuditExportBundleInputError("final audit manifest is non-final or non-canonical")
        return spools, manifest_bytes
    except BaseException:
        for spool in spools.values():
            spool.stream.close()
        raise


def _write_csv_file(path: Path, spool: _RecordSpool) -> BundleFileEntry:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8", newline="") as text:
        os.chmod(path, 0o600)
        writer = csv.DictWriter(text, fieldnames=sorted(spool.fieldnames), lineterminator="\r\n")
        writer.writeheader()
        spool.stream.seek(0)
        for line in spool.stream:
            value = json.loads(line)
            if type(value) is not dict:
                raise AuditExportBundleInputError("CSV private spool contains a non-object record")
            writer.writerow(value)
        text.flush()
        os.fsync(text.fileno())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return BundleFileEntry(path.name, digest.hexdigest(), path.stat().st_size)


def _write_exact_file(path: Path, content: bytes) -> BundleFileEntry:
    with path.open("xb") as stream:
        os.chmod(path, 0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return BundleFileEntry(path.name, hashlib.sha256(content).hexdigest(), len(content))


def _tree_matches(path: Path, files: Sequence[BundleFileEntry]) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return False
    try:
        try:
            names = os.listdir(directory_fd)
        except OSError:
            return False
        expected_names = {entry.relative_path for entry in files}
        if set(names) != expected_names or len(names) != len(expected_names):
            return False
        if len({name.casefold() for name in names}) != len(names):
            return False
        for entry in files:
            file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(entry.relative_path, file_flags, dir_fd=directory_fd)
            except OSError:
                return False
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size_bytes:
                    return False
                digest = hashlib.sha256()
                while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
                    digest.update(chunk)
                after = os.fstat(descriptor)
                if (
                    before.st_dev != after.st_dev
                    or before.st_ino != after.st_ino
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or digest.hexdigest() != entry.content_hash
                ):
                    return False
            finally:
                os.close(descriptor)
        return True
    finally:
        os.close(directory_fd)


def _remove_exact_tree(path: Path, files: Sequence[BundleFileEntry]) -> bool:
    if not _tree_matches(path, files):
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(path, flags)
    except OSError:
        return False
    try:
        for entry in files:
            os.unlink(entry.relative_path, dir_fd=directory_fd)
    except OSError:
        return False
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(path)
        _fsync_directory(path.parent)
    except OSError:
        return False
    return True


@dataclass(slots=True)
class _PinnedBundleParent:
    """Directory capability pinned beneath its immediate no-follow parent."""

    path: Path
    descriptor: int
    anchor_descriptor: int
    entry_name: str
    identity: os.stat_result

    def close(self) -> None:
        os.close(self.descriptor)
        os.close(self.anchor_descriptor)

    def is_still_bound(self) -> bool:
        try:
            observed = os.stat(self.entry_name, dir_fd=self.anchor_descriptor, follow_symlinks=False)
        except OSError:
            return False
        return stat.S_ISDIR(observed.st_mode) and observed.st_dev == self.identity.st_dev and observed.st_ino == self.identity.st_ino


def _open_pinned_bundle_parent(target: Path) -> _PinnedBundleParent:
    """Open every parent component with ``O_NOFOLLOW`` and retain its anchor."""
    parent = target.parent
    if not parent.is_absolute():
        raise AuditExportBundlePreconditionError("bundle target parent must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    current = os.open(parent.anchor, flags)
    parts = parent.parts[1:]
    if not parts:
        anchor = os.dup(current)
        identity = os.fstat(current)
        return _PinnedBundleParent(parent, current, anchor, ".", identity)
    try:
        for index, part in enumerate(parts):
            child = os.open(part, flags, dir_fd=current)
            if index == len(parts) - 1:
                identity = os.fstat(child)
                return _PinnedBundleParent(parent, child, current, part, identity)
            os.close(current)
            current = child
    except BaseException:
        os.close(current)
        raise
    raise AssertionError("absolute bundle parent traversal must terminate")


def _directory_fd_matches(directory_fd: int, files: Sequence[BundleFileEntry]) -> bool:
    try:
        names = os.listdir(directory_fd)
    except OSError:
        return False
    expected_names = {entry.relative_path for entry in files}
    if set(names) != expected_names or len(names) != len(expected_names):
        return False
    if len({name.casefold() for name in names}) != len(names):
        return False
    for entry in files:
        file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(entry.relative_path, file_flags, dir_fd=directory_fd)
        except OSError:
            return False
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size_bytes:
                return False
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, _COPY_CHUNK_BYTES):
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or digest.hexdigest() != entry.content_hash
            ):
                return False
        finally:
            os.close(descriptor)
    return True


def _tree_matches_at(parent_fd: int, name: str, files: Sequence[BundleFileEntry]) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        return _directory_fd_matches(directory_fd, files)
    finally:
        os.close(directory_fd)


def _target_kind_at(parent_fd: int, name: str) -> str:
    try:
        result = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return "absent"
    if stat.S_ISLNK(result.st_mode):
        return "symlink"
    if stat.S_ISDIR(result.st_mode):
        return "directory"
    return "other"


def _remove_exact_tree_at(parent_fd: int, name: str, files: Sequence[BundleFileEntry]) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        identity = os.fstat(directory_fd)
        if not _directory_fd_matches(directory_fd, files):
            return False
        for entry in files:
            os.unlink(entry.relative_path, dir_fd=directory_fd)
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if observed.st_dev != identity.st_dev or observed.st_ino != identity.st_ino:
            return False
    except OSError:
        return False
    finally:
        os.close(directory_fd)
    try:
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError:
        return False
    return True


def _fsync_tree_at(parent_fd: int, name: str, files: Sequence[BundleFileEntry]) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError:
        return False
    try:
        if not _directory_fd_matches(directory_fd, files):
            return False
        os.fsync(directory_fd)
        return True
    finally:
        os.close(directory_fd)


def prepare_audit_export_bundle(
    *,
    target_path: Path,
    request: SinkEffectPrepareRequest,
) -> SinkEffectPlan:
    """Build and fsync one exact private CSV directory bundle."""
    _validate_effect_id(request.effect_id)
    if type(request.effect_input) is not SinkEffectAuditExportSnapshotInput:
        raise TypeError("CSV audit-export bundle requires exact audit snapshot input")
    effect_input = request.effect_input
    if effect_input.export_format is not AuditExportFormat.CSV:
        raise AuditExportBundleInputError("CSV directory bundle requires a CSV audit snapshot")
    target = _inspection_target(request, target_path)
    _assert_no_symlink_components(target)
    if not target.parent.is_dir():
        raise AuditExportBundlePreflightError("bundle target parent must already exist")
    staging = _staging_path(target, request.effect_id)
    building = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.elspeth-{request.effect_id}.building-",
            dir=target.parent,
        )
    )
    os.chmod(building, 0o700)
    spools: dict[str, _RecordSpool] = {}
    installed = False
    files: tuple[BundleFileEntry, ...] = ()
    try:
        spools, manifest_bytes = _parse_verified_records(effect_input)
        entries: list[BundleFileEntry] = []
        for relative_path in sorted(spools):
            entries.append(_write_csv_file(building / relative_path, spools[relative_path]))
        manifest_entry = _write_exact_file(building / AUDIT_MANIFEST_NAME, manifest_bytes)
        if (
            manifest_entry.content_hash != effect_input.signed_manifest.content_hash
            or manifest_entry.size_bytes != effect_input.signed_manifest.size_bytes
        ):
            raise AuditExportBundleInputError("written final manifest differs from its registered descriptor")
        entries.append(manifest_entry)
        files = tuple(sorted(entries, key=lambda entry: entry.relative_path))
        _fsync_directory(building)
        try:
            _rename_noreplace(building, staging)
            installed = True
        except OSError as exc:
            if exc.errno != errno.EEXIST or not _tree_matches(staging, files):
                raise AuditExportBundlePreconditionError("effect-addressed staging directory is divergent") from exc
            shutil.rmtree(building)
        if not _tree_matches(staging, files):
            raise AuditExportBundlePreconditionError("installed staging tree changed before plan durability")
        _fsync_directory(target.parent)
    except BaseException:
        if building.exists():
            shutil.rmtree(building, ignore_errors=True)
        if installed and files:
            _remove_exact_tree(staging, files)
        raise
    finally:
        for spool in spools.values():
            spool.stream.close()

    bundle_size = sum(entry.size_bytes for entry in files)
    bundle_hash = _bundle_hash(files)
    evidence_value = BundlePlanEvidence(
        target_path=str(target),
        staging_path=str(staging),
        snapshot_id=effect_input.snapshot_id,
        manifest_content_ref=effect_input.signed_manifest.content_ref,
        manifest_content_hash=effect_input.signed_manifest.content_hash,
        manifest_size_bytes=effect_input.signed_manifest.size_bytes,
        bundle_hash=bundle_hash,
        bundle_size_bytes=bundle_size,
        files=files,
    )
    evidence = evidence_value.as_mapping()
    descriptor = ArtifactDescriptor.for_file(path=str(target), content_hash=bundle_hash, size_bytes=bundle_size)
    try:
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=_plan_hash(effect_id=request.effect_id, descriptor=descriptor, evidence=evidence),
            payload_hash=bundle_hash,
            expected_descriptor=descriptor,
            safe_evidence=evidence,
        )
    except BaseException:
        _remove_exact_tree(staging, files)
        raise


def _parse_plan(plan: SinkEffectPlan) -> tuple[BundlePlanEvidence, Path, Path, ArtifactDescriptor]:
    if (
        plan.input_kind is not SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT
        or plan.descriptor_mode is not SinkEffectDescriptorMode.PRECOMPUTED
        or plan.expected_descriptor is None
    ):
        raise AuditExportBundlePreconditionError("bundle plan requires a precomputed audit-export descriptor")
    evidence = BundlePlanEvidence.from_mapping(plan.safe_evidence)
    target = _absolute_path(Path(evidence.target_path))
    staging = _absolute_path(Path(evidence.staging_path))
    if staging != _staging_path(target, plan.effect_id) or staging.parent != target.parent:
        raise AuditExportBundlePreconditionError("bundle staging path is not an effect-addressed target sibling")
    descriptor = ArtifactDescriptor.for_file(
        path=str(target),
        content_hash=evidence.bundle_hash,
        size_bytes=evidence.bundle_size_bytes,
    )
    if (
        plan.target != descriptor.path_or_uri
        or plan.expected_descriptor != descriptor
        or plan.payload_hash != evidence.bundle_hash
        or plan.plan_hash != _plan_hash(effect_id=plan.effect_id, descriptor=descriptor, evidence=evidence.as_mapping())
    ):
        raise AuditExportBundlePreconditionError("bundle plan hash or descriptor is divergent")
    return evidence, target, staging, descriptor


def _commit_result(descriptor: ArtifactDescriptor, publication: str) -> SinkEffectCommitResult:
    return SinkEffectCommitResult(
        descriptor=descriptor,
        evidence={"publication": publication},
        accepted_ordinals=(),
        diverted_ordinals=(),
    )


def _before_rename(_staging: Path, _target: Path) -> None:
    """Crash-test seam before the observable create-only rename."""


def _after_rename_before_bundle_fsync(_target: Path) -> None:
    """Crash-test seam after rename and before published-directory fsync."""


def _after_bundle_fsync_before_parent_fsync(_target: Path) -> None:
    """Crash-test seam after bundle fsync and before parent fsync."""


def _after_parent_fsync_before_return(_target: Path) -> None:
    """Crash-test seam after all durability barriers and before return."""


def commit_audit_export_bundle(plan: SinkEffectPlan) -> SinkEffectCommitResult:
    """Publish the staged bundle once, or converge on an exact existing tree."""
    evidence, target, staging, descriptor = _parse_plan(plan)
    parent = _open_pinned_bundle_parent(target)
    try:
        if not parent.is_still_bound():
            raise AuditExportBundlePreconditionError("bundle target parent changed before commit")
        if _tree_matches_at(parent.descriptor, target.name, evidence.files):
            if _target_kind_at(parent.descriptor, staging.name) != "absent" and not _remove_exact_tree_at(
                parent.descriptor, staging.name, evidence.files
            ):
                raise AuditExportBundlePreconditionError("exact target has a divergent leftover staging path")
            if not parent.is_still_bound():
                raise AuditExportBundlePreconditionError("bundle target parent changed during reconciliation")
            return _commit_result(descriptor, "reconciled_exact_existing")
        if _target_kind_at(parent.descriptor, target.name) != "absent":
            raise AuditExportBundleCollisionError("create-only bundle target exists with divergent content")
        if not _tree_matches_at(parent.descriptor, staging.name, evidence.files):
            raise AuditExportBundlePreconditionError("private bundle staging tree is absent or divergent")
        _before_rename(staging, target)
        if not parent.is_still_bound():
            raise AuditExportBundlePreconditionError("bundle target parent changed before publication")
        try:
            _rename_noreplace_at(parent.descriptor, staging.name, target.name)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            if not _tree_matches_at(parent.descriptor, target.name, evidence.files):
                raise AuditExportBundleCollisionError("rename collision target is not the exact planned bundle") from exc
            if _target_kind_at(parent.descriptor, staging.name) != "absent" and not _remove_exact_tree_at(
                parent.descriptor, staging.name, evidence.files
            ):
                raise AuditExportBundlePreconditionError("rename collision left divergent staging content") from exc
            if not parent.is_still_bound():
                raise AuditExportBundlePreconditionError("bundle target parent changed during collision reconciliation") from exc
            return _commit_result(descriptor, "reconciled_exact_existing")
        _after_rename_before_bundle_fsync(target)
        if not _fsync_tree_at(parent.descriptor, target.name, evidence.files):
            raise AuditExportBundlePreconditionError("published bundle changed before directory fsync")
        _after_bundle_fsync_before_parent_fsync(target)
        os.fsync(parent.descriptor)
        _after_parent_fsync_before_return(target)
        if not parent.is_still_bound():
            raise AuditExportBundlePreconditionError("bundle target parent changed during publication")
        if not _tree_matches_at(parent.descriptor, target.name, evidence.files):
            raise AuditExportBundlePreconditionError("published bundle changed before commit returned")
        return _commit_result(descriptor, "rename_noreplace")
    finally:
        parent.close()


def reconcile_audit_export_bundle(plan: SinkEffectPlan) -> SinkEffectReconcileResult:
    """Classify the target using the closed exact/not-applied/unknown set."""
    try:
        evidence, target, staging, descriptor = _parse_plan(plan)
        parent = _open_pinned_bundle_parent(target)
        try:
            if not parent.is_still_bound():
                return SinkEffectReconcileResult.unknown(evidence={"reason": "parent_replaced"})
            if _tree_matches_at(parent.descriptor, target.name, evidence.files):
                if not parent.is_still_bound():
                    return SinkEffectReconcileResult.unknown(evidence={"reason": "parent_replaced"})
                return SinkEffectReconcileResult.applied(descriptor, evidence={"publication": "exact_tree"})
            target_kind = _target_kind_at(parent.descriptor, target.name)
            if target_kind == "absent" and _tree_matches_at(parent.descriptor, staging.name, evidence.files):
                if not parent.is_still_bound():
                    return SinkEffectReconcileResult.unknown(evidence={"reason": "parent_replaced"})
                return SinkEffectReconcileResult.not_applied(evidence={"staging": "exact", "target": "absent"})
            return SinkEffectReconcileResult.unknown(evidence={"staging": "divergent", "target": target_kind})
        finally:
            parent.close()
    except (AuditExportBundleError, OSError, ValueError, TypeError) as exc:
        return SinkEffectReconcileResult.unknown(evidence={"reason": type(exc).__name__})


_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "libc does not expose renameat2")
    result = _RENAMEAT2(
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(_AT_FDCWD),
        ctypes.c_char_p(os.fsencode(destination)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _rename_noreplace_at(parent_fd: int, source_name: str, destination_name: str) -> None:
    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "libc does not expose renameat2")
    result = _RENAMEAT2(
        ctypes.c_int(parent_fd),
        ctypes.c_char_p(os.fsencode(source_name)),
        ctypes.c_int(parent_fd),
        ctypes.c_char_p(os.fsencode(destination_name)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_name)


def _statfs_type(path: Path) -> int:
    statfs = getattr(_LIBC, "statfs", None)
    if statfs is None:
        raise OSError(errno.ENOSYS, "libc does not expose statfs")
    buffer = ctypes.create_string_buffer(512)
    result = statfs(ctypes.c_char_p(os.fsencode(path)), ctypes.byref(buffer))
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(path))
    return ctypes.c_long.from_buffer(buffer).value & 0xFFFFFFFFFFFFFFFF


def _device_id(path: Path) -> int:
    return path.stat().st_dev


def _fsync_regular_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "fsync probe path is not a regular file", str(path))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.ENOTDIR, "fsync path is not a directory", str(path))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_probe_path(path: Path) -> None:
    try:
        result = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def cleanup_stale_audit_export_bundle_scratch(
    parent: Path,
    *,
    now: float | None = None,
    older_than_seconds: float = _PROBE_STALE_SECONDS,
    max_entries: int = _PROBE_CLEANUP_LIMIT,
) -> int:
    """Bound cleanup of old private probe and crashed-prepare building names.

    Building trees are pre-staging scratch: prepare removes them in-process,
    so one only survives a hard crash and is never credited by recovery.
    Never scans outside parent.
    """
    if older_than_seconds < 0 or type(max_entries) is not int or max_entries < 0:
        raise ValueError("stale scratch cleanup bounds must be non-negative")
    current_time = time.time() if now is None else now
    removed = 0
    scratch = {*parent.glob(f"{_PROBE_PREFIX}*"), *parent.glob(_BUILDING_GLOB)}
    for path in sorted(scratch, key=lambda candidate: candidate.name):
        if removed >= max_entries:
            break
        try:
            age = current_time - path.lstat().st_mtime
        except FileNotFoundError:
            continue
        if age < older_than_seconds:
            continue
        _remove_probe_path(path)
        removed += 1
    return removed


def preflight_audit_export_bundle(target_path: Path) -> AuditExportBundlePreflight:
    """Exercise the exact Linux/filesystem durability primitive before reservation."""
    if sys.platform != "linux":
        raise AuditExportBundlePreflightError("CSV audit-export bundles require Linux renameat2 semantics")
    target = _absolute_path(target_path)
    if not target.name:
        raise AuditExportBundlePreflightError("bundle target must name a child directory")
    _assert_no_symlink_components(target)
    parent = target.parent
    if not parent.is_dir():
        raise AuditExportBundlePreflightError("bundle target parent must already exist")
    try:
        filesystem_magic = _statfs_type(parent)
    except OSError as exc:
        raise AuditExportBundlePreflightError("bundle filesystem statfs probe failed") from exc
    if filesystem_magic not in SUPPORTED_LOCAL_FILESYSTEM_MAGIC:
        raise AuditExportBundlePreflightError(f"bundle filesystem 0x{filesystem_magic:x} is not in the local allowlist")
    try:
        target_parent_device = _device_id(parent)
        staging_parent_device = _device_id(parent)
    except OSError as exc:
        raise AuditExportBundlePreflightError("bundle parent device identity is unavailable") from exc
    if target_parent_device != staging_parent_device:
        raise AuditExportBundlePreflightError("bundle staging and target parents must be on the same device")

    cleanup_stale_audit_export_bundle_scratch(parent)
    token = f"{time.time_ns()}-{os.getpid()}-{uuid4().hex}"
    source = parent / f"{_PROBE_PREFIX}{token}-source"
    destination = parent / f"{_PROBE_PREFIX}{token}-destination"
    collision_source = parent / f"{_PROBE_PREFIX}{token}-collision-source"
    collision_destination = parent / f"{_PROBE_PREFIX}{token}-collision-destination"
    paths = (source, destination, collision_source, collision_destination)
    try:
        source.mkdir(mode=0o700)
        probe_file = source / "probe"
        probe_file.write_bytes(b"elspeth-bundle-probe-v1")
        os.chmod(probe_file, 0o600)
        _fsync_regular_file(probe_file)
        _fsync_directory(source)
        try:
            _rename_noreplace(source, destination)
        except OSError as exc:
            if exc.errno == errno.ENOSYS:
                raise AuditExportBundlePreflightError("Linux renameat2(RENAME_NOREPLACE) is unavailable") from exc
            raise AuditExportBundlePreflightError("renameat2 create-only probe failed") from exc
        _fsync_directory(destination)
        _fsync_directory(parent)

        collision_source.mkdir(mode=0o700)
        collision_destination.mkdir(mode=0o700)
        try:
            _rename_noreplace(collision_source, collision_destination)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                if exc.errno == errno.ENOSYS:
                    raise AuditExportBundlePreflightError("Linux renameat2(RENAME_NOREPLACE) is unavailable") from exc
                raise AuditExportBundlePreflightError("renameat2 forced-EEXIST probe failed") from exc
        else:
            raise AuditExportBundlePreflightError("renameat2 replaced an existing probe directory")
        if not collision_source.is_dir() or not collision_destination.is_dir():
            raise AuditExportBundlePreflightError("renameat2 EEXIST probe changed a collision directory")
    except AuditExportBundlePreflightError:
        raise
    except OSError as exc:
        raise AuditExportBundlePreflightError("bundle regular-file/directory/parent fsync probe failed") from exc
    finally:
        for path in paths:
            _remove_probe_path(path)
    return AuditExportBundlePreflight(
        target_path=str(target),
        filesystem_magic=filesystem_magic,
        device_id=target_parent_device,
    )


__all__ = [
    "AUDIT_MANIFEST_NAME",
    "BUNDLE_MANIFEST_SCHEMA",
    "SUPPORTED_LOCAL_FILESYSTEM_MAGIC",
    "AuditExportBundleCollisionError",
    "AuditExportBundleError",
    "AuditExportBundleInputError",
    "AuditExportBundlePreconditionError",
    "AuditExportBundlePreflight",
    "AuditExportBundlePreflightError",
    "BundleFileEntry",
    "BundlePlanEvidence",
    "cleanup_stale_audit_export_bundle_scratch",
    "commit_audit_export_bundle",
    "inspect_audit_export_bundle",
    "preflight_audit_export_bundle",
    "prepare_audit_export_bundle",
    "reconcile_audit_export_bundle",
]
