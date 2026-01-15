# src/elspeth/engine/adapters.py
"""Adapters for bridging Phase 2 plugins to Phase 3B engine interfaces.

Phase 2 plugins use simpler row-wise interfaces for ease of implementation.
Phase 3B engine expects bulk interfaces for efficiency and audit semantics.
These adapters bridge the gap.
"""

import hashlib
import os
from typing import Any, Protocol


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
            artifact_descriptor: Describes the artifact identity by kind
        """
        self._sink = sink
        self.plugin_name = plugin_name
        self.sink_name = sink_name
        self.node_id: str = ""  # Set by Orchestrator during registration
        self._artifact_descriptor = artifact_descriptor
        self._rows_written: int = 0

    @property
    def name(self) -> str:
        """Return sink_name for SinkLike protocol compatibility."""
        return self.sink_name

    @property
    def rows_written(self) -> int:
        """Return total number of rows written through this adapter."""
        return self._rows_written

    def write(self, rows: list[dict[str, Any]], ctx: Any) -> dict[str, Any]:
        """Write rows using the wrapped sink's row-wise interface.

        Loops over rows, calling sink.write() for each, then flushes.
        Does NOT close the sink - close() must be called separately.

        Args:
            rows: List of row dicts to write
            ctx: Plugin context

        Returns:
            Artifact info dict (structure depends on artifact kind)
        """
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
        """Delegate on_start to wrapped sink if it implements it."""
        if hasattr(self._sink, "on_start"):
            self._sink.on_start(ctx)

    def on_complete(self, ctx: Any) -> None:
        """Delegate on_complete to wrapped sink if it implements it."""
        if hasattr(self._sink, "on_complete"):
            self._sink.on_complete(ctx)

    def _compute_artifact_info(self) -> dict[str, Any]:
        """Compute artifact metadata based on descriptor kind.

        Returns:
            Dict with artifact identity and optional content hash.
            Structure depends on kind:
            - file: {kind, path, size_bytes, content_hash}
            - database: {kind, url, table} (no content hash)
            - webhook: {kind, url} (no content hash)
        """
        kind = self._artifact_descriptor.get("kind", "unknown")

        if kind == "file":
            return self._compute_file_artifact()
        elif kind == "database":
            return self._compute_database_artifact()
        elif kind == "webhook":
            return self._compute_webhook_artifact()
        else:
            # Unknown kind - return descriptor as-is
            return dict(self._artifact_descriptor)

    def _compute_file_artifact(self) -> dict[str, Any]:
        """Compute artifact info for file-based sinks."""
        path = self._artifact_descriptor.get("path", "")
        size_bytes = 0
        content_hash = ""

        if path and os.path.exists(path):
            size_bytes = os.path.getsize(path)
            content_hash = self._hash_file_chunked(path)

        return {
            "kind": "file",
            "path": path,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
        }

    def _compute_database_artifact(self) -> dict[str, Any]:
        """Compute artifact info for database sinks.

        Database artifacts use the table identity, not content hashes.
        The audit trail links to the table; row-level integrity is the DB's job.
        """
        return {
            "kind": "database",
            "url": self._artifact_descriptor.get("url", ""),
            "table": self._artifact_descriptor.get("table", ""),
            # No content_hash - database is the source of truth
        }

    def _compute_webhook_artifact(self) -> dict[str, Any]:
        """Compute artifact info for webhook sinks."""
        return {
            "kind": "webhook",
            "url": self._artifact_descriptor.get("url", ""),
            # No content_hash - webhook response should be in calls table
        }

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
