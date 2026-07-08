"""
Payload store for separating large blobs from audit tables.

Uses content-addressable storage (hash-based) for:
- Automatic deduplication of identical content
- Integrity verification on retrieval
- Efficient storage of large payloads referenced by multiple rows
"""

import contextlib
import hashlib
import hmac
import os
import re
import secrets
import stat
from pathlib import Path

import elspeth.contracts.payload_store as payload_contracts
from elspeth.contracts.payload_store import PayloadNotFoundError

__all__ = ["FilesystemPayloadStore"]

# SHA-256 hex digest: exactly 64 lowercase hex characters.
# Compiled regex for performance on repeated validation. Used with
# ``fullmatch`` — NOT ``match`` — because Python's ``$`` anchor treats
# "just before a final \n" as end-of-string, so
# ``re.compile(r"^[a-f0-9]{64}$").match("a" * 64 + "\n")`` returns a
# match object and would let a newline-terminated hash slip through.
# A real ``hashlib.sha256().hexdigest()`` never contains a newline;
# any value that does is either externally sourced (Tier 3 — reject)
# or corrupt Tier-1 data (reject).
_SHA256_HEX_PATTERN = re.compile(r"[a-f0-9]{64}")


try:
    _O_DIRECTORY: int = os.O_DIRECTORY
    _O_NOFOLLOW: int = os.O_NOFOLLOW
except AttributeError as exc:
    raise RuntimeError("FilesystemPayloadStore requires O_DIRECTORY and O_NOFOLLOW support for symlink-safe access") from exc


class FilesystemPayloadStore:
    """Filesystem-based payload store.

    Stores payloads in a directory structure using first 2 characters
    of hash as subdirectory for better file distribution.

    Structure: base_path/ab/abcdef123...
    """

    def __init__(self, base_path: Path) -> None:
        """Initialize filesystem store.

        Args:
            base_path: Root directory for payload storage
        """
        raw_base_path = base_path.expanduser()
        if raw_base_path.is_symlink():
            raise ValueError(f"Invalid payload store directory: base path must not be a symlink: {raw_base_path}")
        existed = raw_base_path.exists()
        raw_base_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.base_path = raw_base_path.resolve()
        if not existed:
            os.chmod(self.base_path, 0o700)
        self._validate_store_directory(self.base_path)

    def _validate_store_directory(self, directory: Path) -> Path:
        """Validate a payload-store directory before using it for filesystem IO."""
        try:
            directory_stat = directory.lstat()
        except OSError as exc:
            raise ValueError(f"Invalid payload store directory: cannot stat {directory}") from exc
        if stat.S_ISLNK(directory_stat.st_mode):
            raise ValueError(f"Invalid payload store directory: symlinks are not allowed: {directory}")
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise ValueError(f"Invalid payload store directory: expected directory: {directory}")
        if directory_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError(f"Invalid payload store directory: group/world-writable directory is not allowed: {directory}")

        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(self.base_path):
            raise ValueError(f"Invalid payload store directory: {resolved} is not under {self.base_path}")
        return resolved

    def _open_validated_directory(self, directory: Path) -> int:
        """Open a validated directory and bind later operations to its fd."""
        self._validate_store_directory(directory)
        flags = os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW
        try:
            dir_fd = os.open(directory, flags)
        except OSError as exc:
            raise ValueError(f"Invalid payload store directory: cannot open {directory}") from exc

        try:
            directory_stat = directory.lstat()
            fd_stat = os.fstat(dir_fd)
            if stat.S_ISLNK(directory_stat.st_mode):
                raise ValueError(f"Invalid payload store directory: symlinks are not allowed: {directory}")
            if (directory_stat.st_dev, directory_stat.st_ino) != (fd_stat.st_dev, fd_stat.st_ino):
                raise ValueError(f"Invalid payload store directory: changed during validation: {directory}")
            resolved = directory.resolve(strict=True)
            if not resolved.is_relative_to(self.base_path):
                raise ValueError(f"Invalid payload store directory: {resolved} is not under {self.base_path}")
        except BaseException:
            os.close(dir_fd)
            raise

        return dir_fd

    def _open_payload_parent(self, parent: Path, *, create: bool) -> int | None:
        """Open the hash-prefix directory, optionally creating it first."""
        self._validate_store_directory(self.base_path)
        if not create and not parent.exists() and not parent.is_symlink():
            return None
        if parent.is_symlink():
            raise ValueError(f"Invalid payload store directory: symlinks are not allowed: {parent}")
        if create:
            existed = parent.exists()
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not existed:
                os.chmod(parent, 0o700)
        return self._open_validated_directory(parent)

    def _read_payload_file(self, dir_fd: int, filename: str) -> bytes:
        """Read a payload file through a validated parent directory fd."""
        try:
            file_fd = os.open(filename, os.O_RDONLY | _O_NOFOLLOW, dir_fd=dir_fd)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ValueError(f"Invalid payload store file: cannot open {filename}") from exc

        with os.fdopen(file_fd, "rb", closefd=True) as fd:
            return fd.read()

    def _payload_file_exists(self, dir_fd: int, filename: str) -> bool:
        """Check for a payload file without following a symlink."""
        try:
            file_stat = os.stat(filename, dir_fd=dir_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(file_stat.st_mode):
            raise ValueError(f"Invalid payload store file: symlinks are not allowed: {filename}")
        return True

    def _write_temp_payload_file(self, dir_fd: int, filename: str, content: bytes) -> str:
        """Write content to a temporary file in an already-open shard directory."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW
        for _ in range(100):
            temp_name = f".{filename}.{secrets.token_hex(8)}.tmp"
            try:
                file_fd = os.open(temp_name, flags, 0o600, dir_fd=dir_fd)
            except FileExistsError:
                continue
            break
        else:
            raise FileExistsError(f"Could not allocate unique payload temp file for {filename}")

        try:
            with os.fdopen(file_fd, "wb", closefd=True) as fd:
                fd.write(content)
                fd.flush()
                os.fsync(fd.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temp_name, dir_fd=dir_fd)
            raise

        return temp_name

    def _path_for_hash(self, content_hash: str) -> Path:
        """Get filesystem path for content hash.

        Validates content_hash format and ensures path containment.

        Args:
            content_hash: Must be a valid SHA-256 hex digest (64 lowercase hex chars)

        Returns:
            Path under base_path for the content

        Raises:
            ValueError: If content_hash is not a valid SHA-256 hex digest
                        or if resolved path escapes base_path
        """
        # Validate hash format - must be exactly 64 lowercase hex characters.
        # Per CLAUDE.md Tier 1 rules: crash immediately on invalid audit data.
        # ``fullmatch`` (not ``match``) because Python's ``$`` would accept a
        # trailing newline — see the _SHA256_HEX_PATTERN comment above.
        if not _SHA256_HEX_PATTERN.fullmatch(content_hash):
            raise ValueError(f"Invalid content_hash: must be 64 lowercase hex characters, got {repr(content_hash)[:50]}")

        # Construct path using first 2 chars as subdirectory
        path = self.base_path / content_hash[:2] / content_hash

        # Defense in depth: verify path is contained within base_path
        # This catches any edge cases the regex might miss
        try:
            resolved = path.resolve()
            base_resolved = self.base_path.resolve()
            if not resolved.is_relative_to(base_resolved):
                raise ValueError(f"Invalid content_hash: path traversal detected, resolved path {resolved} is not under {base_resolved}")
        except (OSError, ValueError) as e:
            # Path resolution failed - treat as invalid
            raise ValueError(f"Invalid content_hash: path resolution failed for {repr(content_hash)[:50]}") from e

        return path

    def store(self, content: bytes) -> str:
        """Store content and return its hash.

        If file already exists, verifies integrity before returning hash.
        This prevents corrupted files from being silently accepted.

        Raises:
            IntegrityError: If existing file doesn't match expected hash
        """
        content_hash = hashlib.sha256(content).hexdigest()
        path = self._path_for_hash(content_hash)

        # Try to verify existing file first (EAFP, not LBYL).
        # Using try/read_bytes instead of exists()+read_bytes avoids a TOCTOU
        # race where a concurrent purge deletes the file between the check and
        # the read. If the file disappears, we fall through to the write path.
        dir_fd = self._open_payload_parent(path.parent, create=False)
        if dir_fd is not None:
            try:
                existing_content = self._read_payload_file(dir_fd, path.name)
                actual_hash = hashlib.sha256(existing_content).hexdigest()

                # Use timing-safe comparison (same as retrieve())
                if not hmac.compare_digest(actual_hash, content_hash):
                    raise payload_contracts.IntegrityError(
                        f"Payload integrity check failed on store: existing file has hash {actual_hash}, expected {content_hash}"
                    )
                return content_hash
            except FileNotFoundError:
                pass  # File doesn't exist or was purged — fall through to write
            finally:
                os.close(dir_fd)

        # Atomic write via temp file to prevent partial/corrupted files on
        # crash (Tier 1 integrity requirement).
        dir_fd = self._open_payload_parent(path.parent, create=True)
        if dir_fd is None:
            raise ValueError(f"Invalid payload store directory: cannot open {path.parent}")
        temp_name: str | None = None
        try:
            temp_name = self._write_temp_payload_file(dir_fd, path.name, content)
            os.replace(temp_name, path.name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            # Fsync parent directory to ensure rename survives power loss
            os.fsync(dir_fd)
        except BaseException:
            if temp_name is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(temp_name, dir_fd=dir_fd)
            raise
        finally:
            os.close(dir_fd)

        return content_hash

    def retrieve(self, content_hash: str) -> bytes:
        """Retrieve content by hash with integrity verification.

        Raises:
            PayloadNotFoundError: If content not found
            IntegrityError: If content doesn't match expected hash
        """
        path = self._path_for_hash(content_hash)
        dir_fd = self._open_payload_parent(path.parent, create=False)
        if dir_fd is None:
            raise PayloadNotFoundError(content_hash)
        try:
            content = self._read_payload_file(dir_fd, path.name)
        except FileNotFoundError as exc:
            raise PayloadNotFoundError(content_hash) from exc
        finally:
            os.close(dir_fd)
        actual_hash = hashlib.sha256(content).hexdigest()

        # Use timing-safe comparison to prevent timing attacks that could
        # allow an attacker to incrementally discover expected hashes
        if not hmac.compare_digest(actual_hash, content_hash):
            raise payload_contracts.IntegrityError(f"Payload integrity check failed: expected {content_hash}, got {actual_hash}")

        return content

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        path = self._path_for_hash(content_hash)
        dir_fd = self._open_payload_parent(path.parent, create=False)
        if dir_fd is None:
            return False
        try:
            return self._payload_file_exists(dir_fd, path.name)
        finally:
            os.close(dir_fd)

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash.

        Returns:
            True if content was deleted, False if not found
        """
        path = self._path_for_hash(content_hash)
        dir_fd = self._open_payload_parent(path.parent, create=False)
        if dir_fd is None:
            return False
        try:
            if not self._payload_file_exists(dir_fd, path.name):
                return False
            os.unlink(path.name, dir_fd=dir_fd)
        except FileNotFoundError:
            return False
        finally:
            os.close(dir_fd)
        return True
