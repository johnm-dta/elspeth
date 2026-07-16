"""Production filesystem content store for immutable audit-export snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from uuid import uuid4

from elspeth.contracts.audit_export import (
    AUDIT_EXPORT_MAX_CHUNK_BYTES,
    MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES,
    AuditExportContentDescriptor,
    AuditExportContentStoreResolver,
    AuditExportObjectKind,
    AuditExportOrphanCollectionRequest,
    BoundAuditExportContentReader,
    RegisteredAuditExportContent,
    validate_content_namespace,
    validate_credential_free_identifier,
)
from elspeth.contracts.hashing import canonical_json
from elspeth.core.config import AuditExportContentStoreSettings, LandscapeExportSettings

_DIRECTORY_FLAGS: Final = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_READ_FLAGS: Final = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_FILE_CREATE_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_COPY_BYTES: Final = 64 * 1024


def _write_all(descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise OSError("audit-export content-store write made no progress")
        view = view[written:]


def _ensure_absolute_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            with suppress(FileExistsError):
                os.mkdir(part, mode=0o700, dir_fd=current)
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        mode = stat.S_IMODE(os.fstat(current).st_mode)
        if mode & 0o077:
            raise ValueError("audit-export content-store root must be private")
        return current
    except BaseException:
        os.close(current)
        raise


def _open_absolute_directory(path: Path) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = os.open(absolute.anchor, _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _ensure_directory_at(parent_fd: int, name: str) -> int:
    with suppress(FileExistsError):
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)


def _open_namespace(root_fd: int, namespace: str, *, create: bool) -> int:
    current = os.dup(root_fd)
    try:
        for part in namespace.split("/"):
            child = _ensure_directory_at(current, part) if create else os.open(part, _DIRECTORY_FLAGS, dir_fd=current)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def _read_exact_file(parent_fd: int, name: str, descriptor: AuditExportContentDescriptor) -> bytes:
    file_fd = os.open(name, _FILE_READ_FLAGS, dir_fd=parent_fd)
    try:
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != descriptor.size_bytes:
            raise OSError("registered audit-export content size or type is divergent")
        chunks: list[bytes] = []
        remaining = descriptor.size_bytes
        while remaining:
            chunk = os.read(file_fd, min(_COPY_BYTES, remaining))
            if not chunk:
                raise OSError("registered audit-export content ended before its descriptor")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(file_fd, 1):
            raise OSError("registered audit-export content exceeds its descriptor")
        content = b"".join(chunks)
        after = os.fstat(file_fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or hashlib.sha256(content).hexdigest() != descriptor.content_hash
        ):
            raise OSError("registered audit-export content changed or failed hash verification")
        return content
    finally:
        os.close(file_fd)


class _FilesystemContentReader:
    __slots__ = ("_registration", "_store")

    def __init__(self, store: FilesystemAuditExportContentStore, registration: RegisteredAuditExportContent) -> None:
        self._store = store
        self._registration = registration

    def read(self) -> bytes:
        return self._store._read_registered(self._registration)


class FilesystemAuditExportContentStore:
    """Fsync-backed, content-addressed store rooted in an explicit private path."""

    def __init__(self, settings: AuditExportContentStoreSettings) -> None:
        if type(settings) is not AuditExportContentStoreSettings:
            raise TypeError("settings must be exact AuditExportContentStoreSettings")
        self._settings = settings
        self._root = Path(os.path.abspath(os.fspath(settings.root)))
        self._content_store_id = validate_credential_free_identifier(settings.content_store_id, "content_store_id")
        self._namespace = validate_content_namespace(settings.namespace)
        root_fd = _ensure_absolute_directory(self._root)
        try:
            self._root_identity = os.fstat(root_fd)
            namespace_fd = _open_namespace(root_fd, self._namespace, create=True)
            try:
                objects_fd = _ensure_directory_at(namespace_fd, "objects")
                candidates_fd = _ensure_directory_at(namespace_fd, "candidates")
                os.fsync(objects_fd)
                os.fsync(candidates_fd)
                os.close(objects_fd)
                os.close(candidates_fd)
                os.fsync(namespace_fd)
            finally:
                os.close(namespace_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)

    @property
    def content_store_id(self) -> str:
        return self._content_store_id

    @property
    def namespace(self) -> str:
        return self._namespace

    def is_durable(self) -> bool:
        return self._settings.durability in {"fsync", "replicated"}

    def _open_root(self) -> int:
        descriptor = _open_absolute_directory(self._root)
        observed = os.fstat(descriptor)
        if observed.st_dev != self._root_identity.st_dev or observed.st_ino != self._root_identity.st_ino:
            os.close(descriptor)
            raise OSError("audit-export content-store root identity changed")
        return descriptor

    def _open_area(self, area: str, *, create: bool = False) -> tuple[int, int]:
        root_fd = self._open_root()
        try:
            namespace_fd = _open_namespace(root_fd, self._namespace, create=False)
            area_fd = _ensure_directory_at(namespace_fd, area) if create else os.open(area, _DIRECTORY_FLAGS, dir_fd=namespace_fd)
            os.close(namespace_fd)
            return root_fd, area_fd
        except BaseException:
            os.close(root_fd)
            raise

    @staticmethod
    def _candidate_dir(candidates_fd: int, candidate_id: str, *, create: bool) -> int:
        validate_credential_free_identifier(candidate_id, "candidate_id")
        return (
            _ensure_directory_at(candidates_fd, candidate_id) if create else os.open(candidate_id, _DIRECTORY_FLAGS, dir_fd=candidates_fd)
        )

    def _mark_owned(self, candidate_id: str, content_hash: str) -> None:
        root_fd, candidates_fd = self._open_area("candidates", create=True)
        try:
            candidate_fd = self._candidate_dir(candidates_fd, candidate_id, create=True)
            try:
                owned_fd = _ensure_directory_at(candidate_fd, "owned")
                try:
                    try:
                        marker_fd = os.open(content_hash, _FILE_CREATE_FLAGS, 0o600, dir_fd=owned_fd)
                    except FileExistsError:
                        return
                    try:
                        _write_all(marker_fd, b"owned\n")
                        os.fsync(marker_fd)
                    finally:
                        os.close(marker_fd)
                    os.fsync(owned_fd)
                finally:
                    os.close(owned_fd)
                os.fsync(candidate_fd)
            finally:
                os.close(candidate_fd)
        finally:
            os.close(candidates_fd)
            os.close(root_fd)

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: AuditExportObjectKind) -> str:
        if type(content) is not bytes or not content or len(content) > AUDIT_EXPORT_MAX_CHUNK_BYTES:
            raise ValueError("audit-export immutable content must be non-empty exact bytes within the hard limit")
        validate_credential_free_identifier(candidate_id, "candidate_id")
        if object_kind not in {"data_chunk", "final_manifest"}:
            raise ValueError("object_kind must be data_chunk or final_manifest")
        if object_kind == "final_manifest" and len(content) > MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES:
            raise ValueError("audit-export final manifest exceeds the code-owned hard limit")
        content_hash = hashlib.sha256(content).hexdigest()
        root_fd, objects_fd = self._open_area("objects", create=True)
        installed = False
        try:
            shard_fd = _ensure_directory_at(objects_fd, content_hash[:2])
            try:
                temporary = f".{content_hash}.{candidate_id}.{uuid4().hex}.tmp"
                temporary_fd = os.open(temporary, _FILE_CREATE_FLAGS, 0o600, dir_fd=shard_fd)
                try:
                    _write_all(temporary_fd, content)
                    os.fsync(temporary_fd)
                finally:
                    os.close(temporary_fd)
                try:
                    os.link(temporary, content_hash, src_dir_fd=shard_fd, dst_dir_fd=shard_fd, follow_symlinks=False)
                    installed = True
                except FileExistsError as exc:
                    descriptor = AuditExportContentDescriptor(
                        content_ref=f"sha256:{content_hash}",
                        content_hash=content_hash,
                        size_bytes=len(content),
                        object_kind=object_kind,
                    )
                    if _read_exact_file(shard_fd, content_hash, descriptor) != content:
                        raise OSError("immutable audit-export content address collided with divergent bytes") from exc
                finally:
                    os.unlink(temporary, dir_fd=shard_fd)
                os.fsync(shard_fd)
            finally:
                os.close(shard_fd)
            os.fsync(objects_fd)
        finally:
            os.close(objects_fd)
            os.close(root_fd)
        if installed:
            self._mark_owned(candidate_id, content_hash)
        return f"sha256:{content_hash}"

    def open_registered(self, registration: RegisteredAuditExportContent) -> BoundAuditExportContentReader:
        if type(registration) is not RegisteredAuditExportContent:
            raise TypeError("registration must be exact RegisteredAuditExportContent")
        if registration.content_store_id != self.content_store_id or registration.namespace != self.namespace:
            raise LookupError("registered audit-export content belongs to a different store or namespace")
        return _FilesystemContentReader(self, registration)

    def _read_registered(self, registration: RegisteredAuditExportContent) -> bytes:
        descriptor = registration.descriptor
        root_fd, objects_fd = self._open_area("objects")
        try:
            shard_fd = os.open(descriptor.content_hash[:2], _DIRECTORY_FLAGS, dir_fd=objects_fd)
            try:
                return _read_exact_file(shard_fd, descriptor.content_hash, descriptor)
            finally:
                os.close(shard_fd)
        finally:
            os.close(objects_fd)
            os.close(root_fd)

    def mark_candidate_orphans(self, candidate_id: str, descriptors: tuple[AuditExportContentDescriptor, ...]) -> None:
        validate_credential_free_identifier(candidate_id, "candidate_id")
        if not descriptors or any(type(item) is not AuditExportContentDescriptor for item in descriptors):
            raise ValueError("descriptors must be a non-empty exact descriptor tuple")
        marker = canonical_json(
            {
                "candidate_id": candidate_id,
                "content_refs": sorted({item.content_ref for item in descriptors}),
                "marked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "schema": "elspeth.audit-export-orphan-candidate.v1",
            }
        ).encode("utf-8")
        root_fd, candidates_fd = self._open_area("candidates", create=True)
        try:
            candidate_fd = self._candidate_dir(candidates_fd, candidate_id, create=True)
            try:
                try:
                    marker_fd = os.open("orphan.json", _FILE_CREATE_FLAGS, 0o600, dir_fd=candidate_fd)
                except FileExistsError:
                    return
                try:
                    _write_all(marker_fd, marker)
                    os.fsync(marker_fd)
                finally:
                    os.close(marker_fd)
                os.fsync(candidate_fd)
            finally:
                os.close(candidate_fd)
        finally:
            os.close(candidates_fd)
            os.close(root_fd)

    def garbage_collect_candidate(self, request: AuditExportOrphanCollectionRequest) -> bool:
        if type(request) is not AuditExportOrphanCollectionRequest or request.namespace != self.namespace:
            raise TypeError("request must be an exact orphan request for this namespace")
        if request.grace_period_seconds < self._settings.orphan_grace_period_seconds:
            return False
        if datetime.now(UTC) < request.marked_at + timedelta(seconds=request.grace_period_seconds):
            return False
        if any(request.fresh_winner_reference_check(item.content_ref) for item in request.descriptors):
            return False
        root_fd, candidates_fd = self._open_area("candidates")
        deleted = False
        try:
            candidate_fd = self._candidate_dir(candidates_fd, request.candidate_id, create=False)
            try:
                try:
                    orphan_fd = os.open("orphan.json", _FILE_READ_FLAGS, dir_fd=candidate_fd)
                except FileNotFoundError:
                    return False
                try:
                    marker = json.loads(os.read(orphan_fd, MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES))
                finally:
                    os.close(orphan_fd)
                if type(marker) is not dict or marker.get("candidate_id") != request.candidate_id:
                    return False
                try:
                    marker_time = datetime.strptime(str(marker["marked_at"]), "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
                except (KeyError, ValueError):
                    return False
                if datetime.now(UTC) < marker_time + timedelta(seconds=request.grace_period_seconds):
                    return False
                owned_fd = os.open("owned", _DIRECTORY_FLAGS, dir_fd=candidate_fd)
                try:
                    namespace_fd = _open_namespace(root_fd, self.namespace, create=False)
                    try:
                        objects_fd = os.open("objects", _DIRECTORY_FLAGS, dir_fd=namespace_fd)
                        try:
                            for descriptor in request.descriptors:
                                try:
                                    os.stat(descriptor.content_hash, dir_fd=owned_fd, follow_symlinks=False)
                                except FileNotFoundError:
                                    continue
                                shard_fd = os.open(descriptor.content_hash[:2], _DIRECTORY_FLAGS, dir_fd=objects_fd)
                                try:
                                    if _read_exact_file(shard_fd, descriptor.content_hash, descriptor):
                                        os.unlink(descriptor.content_hash, dir_fd=shard_fd)
                                        os.fsync(shard_fd)
                                        deleted = True
                                finally:
                                    os.close(shard_fd)
                                os.unlink(descriptor.content_hash, dir_fd=owned_fd)
                        finally:
                            os.close(objects_fd)
                    finally:
                        os.close(namespace_fd)
                finally:
                    os.close(owned_fd)
                os.unlink("orphan.json", dir_fd=candidate_fd)
            finally:
                os.close(candidate_fd)
        except FileNotFoundError:
            return False
        finally:
            os.close(candidates_fd)
            os.close(root_fd)
        return deleted


def create_audit_export_content_store(
    export_settings: LandscapeExportSettings,
) -> tuple[FilesystemAuditExportContentStore, AuditExportContentStoreResolver]:
    """Construct the explicit production store and its winning-ID resolver."""
    if type(export_settings) is not LandscapeExportSettings:
        raise TypeError("export_settings must be exact LandscapeExportSettings")
    if not export_settings.enabled or export_settings.content_store is None:
        raise ValueError("enabled audit export with an explicit content-store policy is required")
    store = FilesystemAuditExportContentStore(export_settings.content_store)
    resolver = AuditExportContentStoreResolver()
    resolver.register(store)
    return store, resolver


__all__ = ["FilesystemAuditExportContentStore", "create_audit_export_content_store"]
