"""
Payload store for separating large blobs from audit tables.

Uses content-addressable storage (hash-based) for:
- Automatic deduplication of identical content
- Integrity verification on retrieval
- Efficient storage of large payloads referenced by multiple rows
"""

import hashlib
import hmac
import os
import re
import tempfile
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
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

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
        try:
            existing_content = path.read_bytes()
            actual_hash = hashlib.sha256(existing_content).hexdigest()

            # Use timing-safe comparison (same as retrieve())
            if not hmac.compare_digest(actual_hash, content_hash):
                raise payload_contracts.IntegrityError(
                    f"Payload integrity check failed on store: existing file has hash {actual_hash}, expected {content_hash}"
                )
            return content_hash
        except FileNotFoundError:
            pass  # File doesn't exist or was purged — fall through to write

        # Atomic write via temp file to prevent partial/corrupted files on
        # crash (Tier 1 integrity requirement).
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as fd:
            temp_path = Path(fd.name)
            try:
                fd.write(content)
                fd.flush()
                os.fsync(fd.fileno())
            except BaseException:
                if temp_path.exists():
                    temp_path.unlink()
                raise
        try:
            os.replace(temp_path, path)
            # Fsync parent directory to ensure rename survives power loss
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return content_hash

    def retrieve(self, content_hash: str) -> bytes:
        """Retrieve content by hash with integrity verification.

        Raises:
            PayloadNotFoundError: If content not found
            IntegrityError: If content doesn't match expected hash
        """
        path = self._path_for_hash(content_hash)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise PayloadNotFoundError(content_hash) from exc
        actual_hash = hashlib.sha256(content).hexdigest()

        # Use timing-safe comparison to prevent timing attacks that could
        # allow an attacker to incrementally discover expected hashes
        if not hmac.compare_digest(actual_hash, content_hash):
            raise payload_contracts.IntegrityError(f"Payload integrity check failed: expected {content_hash}, got {actual_hash}")

        return content

    def exists(self, content_hash: str) -> bool:
        """Check if content exists."""
        return self._path_for_hash(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """Delete content by hash.

        Returns:
            True if content was deleted, False if not found
        """
        path = self._path_for_hash(content_hash)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True
