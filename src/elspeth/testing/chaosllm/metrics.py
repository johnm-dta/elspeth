# src/elspeth/testing/chaosllm/metrics.py
"""Metrics storage and aggregation for ChaosLLM server.

The MetricsRecorder provides thread-safe SQLite storage for request metrics
and time-series aggregation. Data is stored for later analysis by the MCP
server or direct SQL queries.
"""

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from elspeth.testing.chaosllm.config import MetricsConfig

# SQLite schema for metrics tables
_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    deployment TEXT,
    model TEXT,
    outcome TEXT NOT NULL,
    status_code INTEGER,
    error_type TEXT,
    injection_type TEXT,
    latency_ms REAL,
    injected_delay_ms REAL,
    message_count INTEGER,
    prompt_tokens_approx INTEGER,
    response_tokens INTEGER,
    response_mode TEXT
);

CREATE TABLE IF NOT EXISTS timeseries (
    bucket_utc TEXT PRIMARY KEY,
    requests_total INTEGER NOT NULL DEFAULT 0,
    requests_success INTEGER NOT NULL DEFAULT 0,
    requests_rate_limited INTEGER NOT NULL DEFAULT 0,
    requests_capacity_error INTEGER NOT NULL DEFAULT 0,
    requests_server_error INTEGER NOT NULL DEFAULT 0,
    requests_client_error INTEGER NOT NULL DEFAULT 0,
    requests_connection_error INTEGER NOT NULL DEFAULT 0,
    requests_malformed INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL,
    p99_latency_ms REAL
);

CREATE TABLE IF NOT EXISTS run_info (
    run_id TEXT PRIMARY KEY,
    started_utc TEXT NOT NULL,
    config_json TEXT NOT NULL,
    preset_name TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_requests_outcome ON requests(outcome);
CREATE INDEX IF NOT EXISTS idx_requests_endpoint ON requests(endpoint);
"""


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """A single request record for metrics storage.

    This dataclass captures all metrics for a single request to the ChaosLLM
    server. All fields except request_id and timestamp_utc are optional to
    accommodate different request types and outcomes.

    Attributes:
        request_id: Unique identifier for this request
        timestamp_utc: ISO-formatted timestamp in UTC
        endpoint: The API endpoint (e.g., '/chat/completions')
        outcome: Request outcome ('success', 'error_injected', 'error_malformed')
        deployment: Azure deployment name (optional)
        model: Model name (optional)
        status_code: HTTP status code (optional)
        error_type: Type of error if any (optional)
        injection_type: Type of injected behavior (optional)
        latency_ms: Total response latency in milliseconds (optional)
        injected_delay_ms: Artificial delay injected in milliseconds (optional)
        message_count: Number of messages in the request (optional)
        prompt_tokens_approx: Approximate prompt token count (optional)
        response_tokens: Number of response tokens (optional)
        response_mode: Response generation mode (optional)
    """

    request_id: str
    timestamp_utc: str
    endpoint: str
    outcome: str
    deployment: str | None = None
    model: str | None = None
    status_code: int | None = None
    error_type: str | None = None
    injection_type: str | None = None
    latency_ms: float | None = None
    injected_delay_ms: float | None = None
    message_count: int | None = None
    prompt_tokens_approx: int | None = None
    response_tokens: int | None = None
    response_mode: str | None = None


def _get_bucket_utc(timestamp_utc: str, bucket_sec: int) -> str:
    """Calculate the time bucket for a given timestamp.

    Args:
        timestamp_utc: ISO-formatted timestamp string
        bucket_sec: Bucket size in seconds

    Returns:
        ISO-formatted bucket timestamp (truncated to bucket boundary)
    """
    # Parse the timestamp
    dt = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))

    # Truncate to bucket boundary
    # We use seconds since midnight for bucketing
    total_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    bucket_seconds = (total_seconds // bucket_sec) * bucket_sec

    bucket_hour = bucket_seconds // 3600
    bucket_minute = (bucket_seconds % 3600) // 60
    bucket_second = bucket_seconds % 60

    bucket_dt = dt.replace(
        hour=bucket_hour,
        minute=bucket_minute,
        second=bucket_second,
        microsecond=0,
    )

    return bucket_dt.isoformat()


def _classify_outcome(
    outcome: str,
    status_code: int | None,
    error_type: str | None,
) -> tuple[bool, bool, bool, bool, bool, bool, bool]:
    """Classify an outcome for time-series aggregation.

    Returns a tuple of booleans for:
    (success, rate_limited, capacity_error, server_error, client_error,
     connection_error, malformed)
    """
    is_success = outcome == "success"
    is_rate_limited = status_code == 429
    is_capacity_error = status_code == 529
    is_server_error = status_code is not None and 500 <= status_code < 600 and status_code != 529
    is_client_error = status_code is not None and 400 <= status_code < 500 and status_code != 429
    is_connection_error = status_code is None and error_type in (
        "timeout",
        "connection_failed",
        "connection_stall",
        "connection_reset",
    )
    is_malformed = outcome == "error_malformed"

    return (
        is_success,
        is_rate_limited,
        is_capacity_error,
        is_server_error,
        is_client_error,
        is_connection_error,
        is_malformed,
    )


class MetricsRecorder:
    """Thread-safe SQLite metrics recorder for ChaosLLM.

    The MetricsRecorder writes request metrics to SQLite and maintains
    time-series aggregations. It uses connection pooling via per-thread
    connections to ensure thread safety without blocking.

    Usage:
        config = MetricsConfig(database="./metrics.db")
        recorder = MetricsRecorder(config)

        # Record a request
        recorder.record_request(
            request_id="abc123",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            latency_ms=150.5,
        )

        # Get statistics
        stats = recorder.get_stats()

        # Reset for new run
        recorder.reset()
    """

    def __init__(
        self,
        config: MetricsConfig,
        *,
        run_id: str | None = None,
    ) -> None:
        """Initialize the metrics recorder.

        Args:
            config: Metrics configuration
            run_id: Optional run ID (default: auto-generated UUID)
        """
        self._config = config
        self._run_id = run_id if run_id is not None else str(uuid.uuid4())
        self._started_utc = datetime.now(UTC).isoformat()

        # Thread-local storage for connections
        self._local = threading.local()
        self._lock = threading.Lock()

        # Detect in-memory databases and URI usage
        self._use_uri = config.database.startswith("file:")
        self._is_memory_db = config.database == ":memory:" or "mode=memory" in config.database

        # Ensure database directory exists for file-backed databases
        if not self._is_memory_db and not self._use_uri:
            db_path = Path(config.database)
            if db_path.parent != Path("."):
                db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection.

        Returns:
            SQLite connection for the current thread

        Note:
            Uses try/except because threading.local() raises AttributeError
            when the attribute doesn't exist yet for this thread. This is
            the canonical pattern for thread-local storage initialization.
        """
        try:
            # threading.local() returns Any for attributes, but we know the type
            connection: sqlite3.Connection = self._local.connection
            return connection
        except AttributeError:
            # First access in this thread - create new connection
            conn = sqlite3.connect(
                self._config.database,
                check_same_thread=False,
                timeout=30.0,
                uri=self._use_uri,
            )
            # Configure journaling for performance (in-memory uses MEMORY mode)
            if self._is_memory_db:
                conn.execute("PRAGMA journal_mode=MEMORY")
                conn.execute("PRAGMA synchronous=OFF")
            else:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
            return conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        conn.executescript(_SCHEMA)
        conn.commit()

    @property
    def run_id(self) -> str:
        """Get the current run ID."""
        return self._run_id

    @property
    def started_utc(self) -> str:
        """Get the run start time in UTC."""
        return self._started_utc

    def record_request(
        self,
        *,
        request_id: str,
        timestamp_utc: str,
        endpoint: str,
        outcome: str,
        deployment: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        error_type: str | None = None,
        injection_type: str | None = None,
        latency_ms: float | None = None,
        injected_delay_ms: float | None = None,
        message_count: int | None = None,
        prompt_tokens_approx: int | None = None,
        response_tokens: int | None = None,
        response_mode: str | None = None,
    ) -> None:
        """Record a single request to the metrics database.

        This method is thread-safe and non-blocking. It writes the request
        to the requests table and updates the appropriate time-series bucket.

        Args:
            request_id: Unique identifier for this request
            timestamp_utc: ISO-formatted timestamp in UTC
            endpoint: The API endpoint (e.g., '/chat/completions')
            outcome: Request outcome ('success', 'error_injected', 'error_malformed')
            deployment: Azure deployment name (optional)
            model: Model name (optional)
            status_code: HTTP status code (optional)
            error_type: Type of error if any (optional)
            injection_type: Type of injected behavior if any (optional)
            latency_ms: Total response latency in milliseconds (optional)
            injected_delay_ms: Artificial delay injected in milliseconds (optional)
            message_count: Number of messages in the request (optional)
            prompt_tokens_approx: Approximate prompt token count (optional)
            response_tokens: Number of response tokens (optional)
            response_mode: Response generation mode (optional)
        """
        conn = self._get_connection()

        # Insert into requests table
        conn.execute(
            """
            INSERT INTO requests (
                request_id, timestamp_utc, endpoint, outcome, deployment, model,
                status_code, error_type, injection_type, latency_ms, injected_delay_ms,
                message_count, prompt_tokens_approx, response_tokens, response_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                timestamp_utc,
                endpoint,
                outcome,
                deployment,
                model,
                status_code,
                error_type,
                injection_type,
                latency_ms,
                injected_delay_ms,
                message_count,
                prompt_tokens_approx,
                response_tokens,
                response_mode,
            ),
        )

        # Update time-series
        self._update_timeseries(
            conn,
            timestamp_utc,
            outcome,
            status_code,
            error_type,
            latency_ms,
        )

        conn.commit()

    def _update_timeseries(
        self,
        conn: sqlite3.Connection,
        timestamp_utc: str,
        outcome: str,
        status_code: int | None,
        error_type: str | None,
        latency_ms: float | None,
    ) -> None:
        """Update time-series aggregation for a request.

        Args:
            conn: Database connection
            timestamp_utc: Request timestamp
            outcome: Request outcome
            status_code: HTTP status code
            error_type: Error type if any
            latency_ms: Request latency in milliseconds
        """
        bucket = _get_bucket_utc(timestamp_utc, self._config.timeseries_bucket_sec)

        (
            is_success,
            is_rate_limited,
            is_capacity_error,
            is_server_error,
            is_client_error,
            is_connection_error,
            is_malformed,
        ) = _classify_outcome(outcome, status_code, error_type)

        # Upsert the bucket
        conn.execute(
            """
            INSERT INTO timeseries (
                bucket_utc, requests_total, requests_success, requests_rate_limited,
                requests_capacity_error, requests_server_error, requests_client_error,
                requests_connection_error, requests_malformed, avg_latency_ms, p99_latency_ms
            ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(bucket_utc) DO UPDATE SET
                requests_total = requests_total + 1,
                requests_success = requests_success + ?,
                requests_rate_limited = requests_rate_limited + ?,
                requests_capacity_error = requests_capacity_error + ?,
                requests_server_error = requests_server_error + ?,
                requests_client_error = requests_client_error + ?,
                requests_connection_error = requests_connection_error + ?,
                requests_malformed = requests_malformed + ?
            """,
            (
                bucket,
                int(is_success),
                int(is_rate_limited),
                int(is_capacity_error),
                int(is_server_error),
                int(is_client_error),
                int(is_connection_error),
                int(is_malformed),
                latency_ms,
                latency_ms,
                # For the UPDATE part
                int(is_success),
                int(is_rate_limited),
                int(is_capacity_error),
                int(is_server_error),
                int(is_client_error),
                int(is_connection_error),
                int(is_malformed),
            ),
        )

        # Update latency statistics for the bucket
        # We need to recalculate avg and p99 from the raw requests
        if latency_ms is not None:
            self._update_bucket_latency_stats(conn, bucket)

    def _update_bucket_latency_stats(
        self,
        conn: sqlite3.Connection,
        bucket: str,
    ) -> None:
        """Recalculate latency statistics for a time-series bucket.

        Args:
            conn: Database connection
            bucket: The bucket timestamp
        """
        # Get the bucket boundaries for querying requests
        bucket_dt = datetime.fromisoformat(bucket)
        bucket_end_dt = bucket_dt + timedelta(seconds=self._config.timeseries_bucket_sec)
        bucket_end = bucket_end_dt.isoformat()

        # Query latencies for this bucket
        cursor = conn.execute(
            """
            SELECT latency_ms FROM requests
            WHERE timestamp_utc >= ? AND timestamp_utc < ? AND latency_ms IS NOT NULL
            ORDER BY latency_ms
            """,
            (bucket, bucket_end),
        )

        latencies = [row[0] for row in cursor.fetchall()]

        if not latencies:
            return

        avg_latency = sum(latencies) / len(latencies)

        # Calculate p99
        p99_index = int(len(latencies) * 0.99)
        if p99_index >= len(latencies):
            p99_index = len(latencies) - 1
        p99_latency = latencies[p99_index]

        conn.execute(
            """
            UPDATE timeseries SET avg_latency_ms = ?, p99_latency_ms = ?
            WHERE bucket_utc = ?
            """,
            (avg_latency, p99_latency, bucket),
        )

    def update_timeseries(self) -> None:
        """Recalculate all time-series buckets from raw request data.

        This is useful for rebuilding aggregations after data corrections
        or for ensuring consistency.
        """
        conn = self._get_connection()

        # Clear existing time-series data
        conn.execute("DELETE FROM timeseries")

        # Get all unique buckets from requests
        cursor = conn.execute("SELECT DISTINCT timestamp_utc FROM requests ORDER BY timestamp_utc")
        timestamps = [row[0] for row in cursor.fetchall()]

        # Group by bucket and rebuild
        seen_buckets: set[str] = set()
        for ts in timestamps:
            bucket = _get_bucket_utc(ts, self._config.timeseries_bucket_sec)
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)

            # Get bucket boundaries
            bucket_dt = datetime.fromisoformat(bucket)
            bucket_end_dt = bucket_dt + timedelta(seconds=self._config.timeseries_bucket_sec)
            bucket_end = bucket_end_dt.isoformat()

            # Query all requests in this bucket
            cursor = conn.execute(
                """
                SELECT outcome, status_code, error_type, latency_ms
                FROM requests
                WHERE timestamp_utc >= ? AND timestamp_utc < ?
                """,
                (bucket, bucket_end),
            )

            rows = cursor.fetchall()
            if not rows:
                continue

            # Aggregate statistics
            total = len(rows)
            success = 0
            rate_limited = 0
            capacity_error = 0
            server_error = 0
            client_error = 0
            connection_error = 0
            malformed = 0
            latencies: list[float] = []

            for row in rows:
                outcome, status_code, error_type, latency_ms = row
                (
                    is_success,
                    is_rate_limited,
                    is_capacity_error,
                    is_server_error,
                    is_client_error,
                    is_connection_error,
                    is_malformed,
                ) = _classify_outcome(outcome, status_code, error_type)

                if is_success:
                    success += 1
                if is_rate_limited:
                    rate_limited += 1
                if is_capacity_error:
                    capacity_error += 1
                if is_server_error:
                    server_error += 1
                if is_client_error:
                    client_error += 1
                if is_connection_error:
                    connection_error += 1
                if is_malformed:
                    malformed += 1
                if latency_ms is not None:
                    latencies.append(latency_ms)

            avg_latency = sum(latencies) / len(latencies) if latencies else None
            p99_latency = None
            if latencies:
                latencies.sort()
                p99_index = int(len(latencies) * 0.99)
                if p99_index >= len(latencies):
                    p99_index = len(latencies) - 1
                p99_latency = latencies[p99_index]

            conn.execute(
                """
                INSERT INTO timeseries (
                    bucket_utc, requests_total, requests_success, requests_rate_limited,
                    requests_capacity_error, requests_server_error, requests_client_error,
                    requests_connection_error, requests_malformed, avg_latency_ms, p99_latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bucket,
                    total,
                    success,
                    rate_limited,
                    capacity_error,
                    server_error,
                    client_error,
                    connection_error,
                    malformed,
                    avg_latency,
                    p99_latency,
                ),
            )

        conn.commit()

    def reset(
        self,
        *,
        config_json: str | None = None,
        preset_name: str | None = None,
    ) -> None:
        """Reset all metrics tables and start a new run.

        Clears all data from requests and timeseries tables, generates a new
        run_id, and records the new run in run_info when metadata is available.
        """
        with self._lock:
            self._run_id = str(uuid.uuid4())
            self._started_utc = datetime.now(UTC).isoformat()

        conn = self._get_connection()

        if config_json is None:
            cursor = conn.execute("SELECT config_json, preset_name FROM run_info LIMIT 1")
            row = cursor.fetchone()
            if row is not None:
                config_json = row["config_json"]
                if preset_name is None:
                    preset_name = row["preset_name"]

        # Clear data tables
        conn.execute("DELETE FROM requests")
        conn.execute("DELETE FROM timeseries")

        if config_json is not None:
            conn.execute("DELETE FROM run_info")
            conn.execute(
                """
                INSERT INTO run_info (run_id, started_utc, config_json, preset_name)
                VALUES (?, ?, ?, ?)
                """,
                (self._run_id, self._started_utc, config_json, preset_name),
            )
        conn.commit()

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics for the current run.

        Returns:
            Dictionary with summary statistics including:
            - run_id: Current run identifier
            - started_utc: Run start time
            - total_requests: Total number of requests
            - requests_by_outcome: Count by outcome type
            - requests_by_status_code: Count by HTTP status code
            - latency_stats: Latency statistics (avg, p50, p95, p99, max)
            - error_rate: Percentage of non-success requests
        """
        conn = self._get_connection()

        # Total requests
        cursor = conn.execute("SELECT COUNT(*) FROM requests")
        total_requests = cursor.fetchone()[0]

        # Requests by outcome
        cursor = conn.execute("SELECT outcome, COUNT(*) FROM requests GROUP BY outcome")
        requests_by_outcome = {row[0]: row[1] for row in cursor.fetchall()}

        # Requests by status code
        cursor = conn.execute("SELECT status_code, COUNT(*) FROM requests WHERE status_code IS NOT NULL GROUP BY status_code")
        requests_by_status_code = {row[0]: row[1] for row in cursor.fetchall()}

        # Latency statistics
        cursor = conn.execute(
            """
            SELECT
                AVG(latency_ms),
                MAX(latency_ms)
            FROM requests WHERE latency_ms IS NOT NULL
            """
        )
        row = cursor.fetchone()
        avg_latency = row[0]
        max_latency = row[1]

        # Percentiles require sorting
        cursor = conn.execute("SELECT latency_ms FROM requests WHERE latency_ms IS NOT NULL ORDER BY latency_ms")
        latencies = [r[0] for r in cursor.fetchall()]

        p50_latency = None
        p95_latency = None
        p99_latency = None

        if latencies:
            p50_index = int(len(latencies) * 0.50)
            p95_index = int(len(latencies) * 0.95)
            p99_index = int(len(latencies) * 0.99)

            if p50_index >= len(latencies):
                p50_index = len(latencies) - 1
            if p95_index >= len(latencies):
                p95_index = len(latencies) - 1
            if p99_index >= len(latencies):
                p99_index = len(latencies) - 1

            p50_latency = latencies[p50_index]
            p95_latency = latencies[p95_index]
            p99_latency = latencies[p99_index]

        latency_stats = {
            "avg_ms": avg_latency,
            "p50_ms": p50_latency,
            "p95_ms": p95_latency,
            "p99_ms": p99_latency,
            "max_ms": max_latency,
        }

        # Error rate calculation
        # Count non-success requests (all outcomes that aren't "success")
        error_rate = 0.0
        if total_requests > 0:
            error_count = sum(count for outcome, count in requests_by_outcome.items() if outcome != "success")
            error_rate = (error_count / total_requests) * 100

        return {
            "run_id": self._run_id,
            "started_utc": self._started_utc,
            "total_requests": total_requests,
            "requests_by_outcome": requests_by_outcome,
            "requests_by_status_code": requests_by_status_code,
            "latency_stats": latency_stats,
            "error_rate": error_rate,
        }

    def export_data(self) -> dict[str, Any]:
        """Export raw requests and time-series data for pushback."""
        conn = self._get_connection()
        requests = [dict(row) for row in conn.execute("SELECT * FROM requests ORDER BY timestamp_utc")]
        timeseries = [dict(row) for row in conn.execute("SELECT * FROM timeseries ORDER BY bucket_utc")]
        return {
            "run_id": self._run_id,
            "started_utc": self._started_utc,
            "requests": requests,
            "timeseries": timeseries,
        }

    def save_run_info(
        self,
        config_json: str,
        preset_name: str | None = None,
    ) -> None:
        """Save run information to the database.

        Args:
            config_json: JSON string of the run configuration
            preset_name: Name of the preset used (optional)
        """
        conn = self._get_connection()

        # Use INSERT OR REPLACE to handle multiple saves for same run
        conn.execute(
            """
            INSERT OR REPLACE INTO run_info (run_id, started_utc, config_json, preset_name)
            VALUES (?, ?, ?, ?)
            """,
            (self._run_id, self._started_utc, config_json, preset_name),
        )
        conn.commit()

    def get_requests(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        outcome: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get request records from the database.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            outcome: Filter by outcome (optional)

        Returns:
            List of request records as dictionaries
        """
        conn = self._get_connection()

        if outcome is not None:
            cursor = conn.execute(
                """
                SELECT * FROM requests WHERE outcome = ?
                ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?
                """,
                (outcome, limit, offset),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM requests
                ORDER BY timestamp_utc DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

        return [dict(row) for row in cursor.fetchall()]

    def get_timeseries(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Get time-series records from the database.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of time-series records as dictionaries
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT * FROM timeseries
            ORDER BY bucket_utc DESC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )

        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close all database connections.

        Call this when shutting down to ensure clean disconnection.
        """
        try:
            connection: sqlite3.Connection = self._local.connection
            connection.close()
            del self._local.connection
        except AttributeError:
            # No connection was created for this thread - nothing to close
            pass
