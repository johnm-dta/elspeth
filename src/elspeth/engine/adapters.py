# src/elspeth/engine/adapters.py
"""Adapters for bridging Phase 2 plugins to Phase 3B engine interfaces.

Phase 2 plugins use simpler row-wise interfaces for ease of implementation.
Phase 3B engine expects bulk interfaces for efficiency and audit semantics.
These adapters bridge the gap.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.engine.artifacts import ArtifactDescriptor


@dataclass(frozen=True)
class RawArtifactDescriptor:
    """Typed input descriptor for SinkAdapter construction.

    This is validated at construction time to ensure all required fields
    are present. Internal methods can then use direct attribute access.

    Note: response_code for webhooks is not included here because it's
    set dynamically after requests complete, not at construction time.
    """

    kind: Literal["file", "database", "webhook"]
    # File artifacts
    path: str | None = None
    # Database artifacts
    url: str | None = None
    table: str | None = None

    def __post_init__(self) -> None:
        """Validate that required fields are present for each kind."""
        if self.kind == "file":
            if self.path is None:
                raise ValueError("File artifact requires 'path'")
        elif self.kind == "database":
            if self.url is None:
                raise ValueError("Database artifact requires 'url'")
            if self.table is None:
                raise ValueError("Database artifact requires 'table'")
        elif self.kind == "webhook" and self.url is None:
            raise ValueError("Webhook artifact requires 'url'")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawArtifactDescriptor":
        """Create from a dict, validating required fields.

        Args:
            data: Dict with 'kind' and kind-specific fields

        Returns:
            Validated RawArtifactDescriptor

        Raises:
            ValueError: If 'kind' is missing or invalid, or required fields missing
        """
        if "kind" not in data:
            raise ValueError("artifact_descriptor must include 'kind'")

        kind = data["kind"]
        if kind not in ("file", "database", "webhook"):
            raise ValueError(f"Unknown artifact kind: {kind}")

        return cls(
            kind=kind,
            path=data.get("path"),
            url=data.get("url"),
            table=data.get("table"),
        )


class RowWiseSinkProtocol(Protocol):
    """Protocol for Phase 2 row-wise sinks."""

    def write(self, row: dict[str, Any], ctx: Any) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class SinkAdapter:
    """Adapts Phase 2 row-wise sinks to Phase 3B bulk SinkLike interface.

    Phase 2 SinkProtocol: write(row: dict, ctx) -> None (row-wise)
    Phase 3B SinkLike: write(rows: list[dict], ctx) -> dict (bulk, artifact info)

    Artifact Descriptors:
        Different sink types produce different artifact identities:
        - File sinks: {"kind": "file", "path": "output.csv"}
        - Database sinks: {"kind": "database", "url": "...", "table": "results"}
        - Webhook sinks: {"kind": "webhook", "url": "..."}

    Usage:
        raw_sink = CSVSink({"path": "output.csv"})
        adapter = SinkAdapter(
            raw_sink,
            plugin_name="csv",
            sink_name="output",
            artifact_descriptor={"kind": "file", "path": "output.csv"},
        )

        # Pass adapter to PipelineConfig (implements SinkLike)
        config = PipelineConfig(source=src, transforms=[], sinks={"output": adapter})

        # After orchestrator.run(), close the adapter (fills Phase 3B lifecycle gap)
        adapter.close()
    """

    def __init__(
        self,
        sink: RowWiseSinkProtocol,
        plugin_name: str,
        sink_name: str,
        artifact_descriptor: dict[str, Any],
    ) -> None:
        """Wrap a Phase 2 row-wise sink.

        Args:
            sink: Phase 2 sink implementing write(row, ctx) -> None
            plugin_name: Type of sink plugin (csv, json, database)
            sink_name: Instance name from config (output, flagged, etc.)
            artifact_descriptor: Describes the artifact identity by kind.
                Must include 'kind' (file|database|webhook) and kind-specific fields.

        Raises:
            ValueError: If artifact_descriptor is missing required fields
        """
        self._sink = sink
        self.plugin_name = plugin_name
        self.sink_name = sink_name
        self.node_id: str = ""  # Set by Orchestrator during registration
        # Validate at construction - this fails fast if required fields missing
        self._artifact_descriptor = RawArtifactDescriptor.from_dict(artifact_descriptor)
        self._response_code: int = 0  # Set dynamically for webhook artifacts
        self._rows_written: int = 0
        self._last_batch_rows: list[dict[str, Any]] = []  # For hash computation

    @property
    def name(self) -> str:
        """Return sink_name for SinkLike protocol compatibility."""
        return self.sink_name

    @property
    def rows_written(self) -> int:
        """Return total number of rows written through this adapter."""
        return self._rows_written

    def write(self, rows: list[dict[str, Any]], ctx: Any) -> ArtifactDescriptor:
        """Write rows using the wrapped sink's row-wise interface.

        Loops over rows, calling sink.write() for each, then flushes.
        Does NOT close the sink - close() must be called separately.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            ArtifactDescriptor with unified artifact info for any sink type
        """
        # Store batch for hash computation (database/webhook sinks need this)
        self._last_batch_rows = list(rows)

        # Loop over rows, calling Phase 2 row-wise write
        for row in rows:
            self._sink.write(row, ctx)
            self._rows_written += 1

        # Flush buffered data (but don't close - lifecycle managed separately)
        self._sink.flush()

        # Compute artifact metadata based on descriptor kind
        return self._compute_artifact_info()

    def flush(self) -> None:
        """Flush buffered data to disk.

        Delegates to the wrapped sink's flush() method.
        """
        self._sink.flush()

    def close(self) -> None:
        """Close the wrapped sink.

        Must be called after orchestrator.run() completes.
        Fills the Phase 3B lifecycle gap where Orchestrator doesn't close sinks.
        """
        self._sink.close()

    def on_start(self, ctx: Any) -> None:
        """Delegate on_start to wrapped sink.

        Uses hasattr because wrapped sink may be Phase 2 plugin not
        inheriting from BaseSink (legitimate trust boundary check).
        """
        if hasattr(self._sink, "on_start"):
            self._sink.on_start(ctx)

    def on_complete(self, ctx: Any) -> None:
        """Delegate on_complete to wrapped sink.

        Uses hasattr because wrapped sink may be Phase 2 plugin not
        inheriting from BaseSink (legitimate trust boundary check).
        """
        if hasattr(self._sink, "on_complete"):
            self._sink.on_complete(ctx)

    def _compute_artifact_info(self) -> ArtifactDescriptor:
        """Compute artifact metadata based on descriptor kind.

        Returns:
            ArtifactDescriptor with unified format for all sink types.
            All types have: artifact_type, path_or_uri, content_hash, size_bytes
        """
        # Direct attribute access - kind is validated at construction
        kind = self._artifact_descriptor.kind

        if kind == "file":
            return self._compute_file_artifact()
        elif kind == "database":
            return self._compute_database_artifact()
        else:  # kind == "webhook"
            return self._compute_webhook_artifact()

    def _compute_file_artifact(self) -> ArtifactDescriptor:
        """Compute artifact info for file-based sinks."""
        # Direct attribute access - path is validated as non-None for file kind
        path = self._artifact_descriptor.path
        assert path is not None  # Validated by RawArtifactDescriptor.__post_init__
        size_bytes = 0
        content_hash = ""

        if path and os.path.exists(path):
            size_bytes = os.path.getsize(path)
            content_hash = self._hash_file_chunked(path)

        return ArtifactDescriptor.for_file(
            path=path,
            content_hash=content_hash,
            size_bytes=size_bytes,
        )

    def _compute_database_artifact(self) -> ArtifactDescriptor:
        """Compute artifact info for database sinks.

        For audit integrity, database artifacts compute a hash from the
        serialized payload BEFORE insertion. This proves what was sent
        even if the database modifies the data later.
        """
        # Direct attribute access - url/table validated as non-None for database kind
        url = self._artifact_descriptor.url
        table = self._artifact_descriptor.table
        assert url is not None  # Validated by RawArtifactDescriptor.__post_init__
        assert table is not None  # Validated by RawArtifactDescriptor.__post_init__

        # Compute hash from canonical JSON of rows being written
        payload = canonical_json(self._last_batch_rows)
        content_hash = stable_hash(self._last_batch_rows)
        payload_size = len(payload.encode("utf-8"))

        return ArtifactDescriptor.for_database(
            url=url,
            table=table,
            content_hash=content_hash,
            payload_size=payload_size,
            row_count=len(self._last_batch_rows),
        )

    def _compute_webhook_artifact(self) -> ArtifactDescriptor:
        """Compute artifact info for webhook sinks.

        For audit integrity, webhook artifacts compute a hash from the
        serialized request payload BEFORE sending. This proves what was
        sent even if the external service doesn't log the request.
        """
        # Direct attribute access - url validated as non-None for webhook kind
        url = self._artifact_descriptor.url
        assert url is not None  # Validated by RawArtifactDescriptor.__post_init__
        # Response code is set dynamically via set_response_code() after requests
        response_code = self._response_code

        # Compute hash from canonical JSON of rows being sent
        payload = canonical_json(self._last_batch_rows)
        content_hash = stable_hash(self._last_batch_rows)
        request_size = len(payload.encode("utf-8"))

        return ArtifactDescriptor.for_webhook(
            url=url,
            content_hash=content_hash,
            request_size=request_size,
            response_code=response_code,
        )

    def set_response_code(self, code: int) -> None:
        """Set the HTTP response code for webhook artifacts.

        Called by webhook sinks after a request completes.

        Args:
            code: HTTP response code (e.g., 200, 201, 500)
        """
        self._response_code = code

    @staticmethod
    def _hash_file_chunked(path: str, chunk_size: int = 65536) -> str:
        """Hash a file in chunks to avoid memory issues with large files.

        Args:
            path: Path to file
            chunk_size: Bytes to read per chunk (default 64KB)

        Returns:
            SHA-256 hex digest
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256.update(chunk)
        return sha256.hexdigest()
