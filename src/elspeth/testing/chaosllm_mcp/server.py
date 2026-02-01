# src/elspeth/testing/chaosllm_mcp/server.py
"""MCP server for ChaosLLM metrics analysis.

Provides Claude-optimized analysis tools for investigating ChaosLLM test
results. The tools pre-compute insights and return concise summaries designed
for LLM consumption.

Usage:
    # Direct execution
    python -m elspeth.testing.chaosllm_mcp.server --database ./chaosllm-metrics.db

    # As CLI command
    elspeth chaosllm-mcp --database ./chaosllm-metrics.db
"""

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


class ChaosLLMAnalyzer:
    """Read-only analyzer for ChaosLLM metrics database.

    Provides high-level analysis tools with pre-computed insights designed
    for Claude Code consumption. Tools return concise summaries to minimize
    token usage while providing actionable information.
    """

    def __init__(self, database_path: str) -> None:
        """Initialize analyzer with database connection.

        Args:
            database_path: Path to SQLite metrics database
        """
        self._db_path = database_path
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # === High-Level Analysis Tools (Pre-computed insights) ===

    def diagnose(self) -> dict[str, Any]:
        """One-paragraph diagnostic summary.

        Returns total requests, success rate, top error types, detected
        patterns, and AIMD assessment. Designed for ~100 token output.
        """
        conn = self._get_connection()

        # Total requests and success rate
        cursor = conn.execute("SELECT COUNT(*) FROM requests")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "summary": "No requests recorded. Run some tests first.",
                "status": "NO_DATA",
            }

        cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE outcome = 'success'")
        success_count = cursor.fetchone()[0]
        success_rate = (success_count / total) * 100 if total > 0 else 0.0

        # Top 3 error types
        cursor = conn.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM requests
            WHERE error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 3
            """
        )
        top_errors = [(row["error_type"], row["count"]) for row in cursor.fetchall()]

        # Detect patterns
        patterns = self._detect_patterns(conn)

        # AIMD assessment based on 429 rate and patterns
        cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE status_code = 429")
        rate_limit_count = cursor.fetchone()[0]
        rate_limit_pct = (rate_limit_count / total) * 100 if total > 0 else 0.0

        if rate_limit_pct > 30:
            aimd_status = "STRESSED: High 429 rate indicates AIMD under pressure"
        elif rate_limit_pct > 10:
            aimd_status = "MODERATE: Noticeable 429s, AIMD should be adjusting"
        elif rate_limit_pct > 0:
            aimd_status = "LIGHT: Some 429s, AIMD handling normally"
        else:
            aimd_status = "CLEAN: No rate limiting detected"

        # Build summary paragraph
        error_summary = ", ".join(f"{et} ({c})" for et, c in top_errors) if top_errors else "none"
        pattern_summary = ", ".join(patterns) if patterns else "none"

        summary = (
            f"{total} requests, {success_rate:.1f}% success. Top errors: {error_summary}. Patterns: {pattern_summary}. AIMD: {aimd_status}."
        )

        return {
            "summary": summary,
            "status": "CRITICAL" if success_rate < 50 else "WARNING" if success_rate < 80 else "OK",
            "total_requests": total,
            "success_rate_pct": round(success_rate, 2),
            "rate_limit_pct": round(rate_limit_pct, 2),
            "top_errors": top_errors,
            "patterns_detected": patterns,
            "aimd_assessment": aimd_status,
        }

    def _detect_patterns(self, conn: sqlite3.Connection) -> list[str]:
        """Detect unusual patterns in the data."""
        patterns = []

        # Check for burst patterns (high error concentration in short windows)
        cursor = conn.execute(
            """
            SELECT bucket_utc, requests_rate_limited, requests_total
            FROM timeseries
            WHERE requests_rate_limited > 0
            ORDER BY bucket_utc
            """
        )
        burst_windows = 0
        for row in cursor.fetchall():
            if row["requests_total"] > 0:
                rate = row["requests_rate_limited"] / row["requests_total"]
                if rate > 0.5:  # >50% rate limited in a bucket
                    burst_windows += 1
        if burst_windows >= 3:
            patterns.append(f"burst_periods({burst_windows})")

        # Check for timeout clustering
        cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE error_type = 'timeout'")
        timeout_count = cursor.fetchone()[0]
        if timeout_count > 10:
            patterns.append(f"timeout_cluster({timeout_count})")

        # Check for error type diversity
        cursor = conn.execute("SELECT COUNT(DISTINCT error_type) FROM requests WHERE error_type IS NOT NULL")
        error_types = cursor.fetchone()[0]
        if error_types >= 5:
            patterns.append(f"diverse_errors({error_types}_types)")

        return patterns

    def analyze_aimd_behavior(self) -> dict[str, Any]:
        """Analyze AIMD-related behavior.

        Returns recovery times after bursts, backoff effectiveness, and
        throughput degradation. Designed for ~150 token output.
        """
        conn = self._get_connection()

        # Get time-series data for burst analysis
        cursor = conn.execute(
            """
            SELECT bucket_utc, requests_total, requests_success,
                   requests_rate_limited, requests_capacity_error
            FROM timeseries
            ORDER BY bucket_utc
            """
        )
        buckets = [dict(row) for row in cursor.fetchall()]

        if len(buckets) < 2:
            return {
                "summary": "Insufficient data for AIMD analysis (need multiple time buckets).",
                "status": "NO_DATA",
            }

        # Find burst periods (>30% errors in a bucket)
        burst_starts = []
        burst_ends = []
        in_burst = False
        for i, bucket in enumerate(buckets):
            total = bucket["requests_total"]
            errors = bucket["requests_rate_limited"] + bucket["requests_capacity_error"]
            error_rate = errors / total if total > 0 else 0

            if error_rate > 0.3 and not in_burst:
                in_burst = True
                burst_starts.append(i)
            elif error_rate < 0.1 and in_burst:
                in_burst = False
                burst_ends.append(i)

        if in_burst:
            burst_ends.append(len(buckets) - 1)

        # Calculate recovery times
        recovery_times = []
        for start, end in zip(burst_starts, burst_ends, strict=True):
            recovery_buckets = end - start
            recovery_times.append(recovery_buckets)

        avg_recovery = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # Calculate throughput degradation
        pre_burst_throughput = []
        during_burst_throughput = []

        for i, bucket in enumerate(buckets):
            success = bucket["requests_success"]
            is_burst = any(s <= i < e for s, e in zip(burst_starts, burst_ends, strict=True))
            if is_burst:
                during_burst_throughput.append(success)
            else:
                pre_burst_throughput.append(success)

        avg_normal = sum(pre_burst_throughput) / len(pre_burst_throughput) if pre_burst_throughput else 0
        avg_burst = sum(during_burst_throughput) / len(during_burst_throughput) if during_burst_throughput else 0

        degradation_pct = ((avg_normal - avg_burst) / avg_normal * 100) if avg_normal > 0 else 0

        # Backoff effectiveness (ratio of 429s to total during recovery)
        cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE status_code = 429")
        total_429s = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM requests")
        total_requests = cursor.fetchone()[0]

        backoff_ratio = total_429s / total_requests if total_requests > 0 else 0

        if backoff_ratio < 0.1:
            backoff_assessment = "EFFECTIVE: Low 429 ratio indicates good backoff"
        elif backoff_ratio < 0.3:
            backoff_assessment = "MODERATE: Some 429s, backoff could be more aggressive"
        else:
            backoff_assessment = "INEFFECTIVE: High 429 ratio, backoff not helping"

        summary = (
            f"Detected {len(burst_starts)} burst period(s). "
            f"Avg recovery: {avg_recovery:.1f} buckets. "
            f"Throughput degradation: {degradation_pct:.1f}%. "
            f"Backoff: {backoff_assessment}."
        )

        return {
            "summary": summary,
            "burst_count": len(burst_starts),
            "avg_recovery_buckets": round(avg_recovery, 2),
            "throughput_degradation_pct": round(degradation_pct, 2),
            "backoff_ratio": round(backoff_ratio, 3),
            "backoff_assessment": backoff_assessment,
            "normal_throughput_avg": round(avg_normal, 2),
            "burst_throughput_avg": round(avg_burst, 2),
        }

    def analyze_errors(self) -> dict[str, Any]:
        """Analyze error distribution.

        Returns errors grouped by category with counts, percentages, and
        sample timestamps. Designed for ~120 token output.
        """
        conn = self._get_connection()

        # Total requests
        cursor = conn.execute("SELECT COUNT(*) FROM requests")
        total = cursor.fetchone()[0]

        if total == 0:
            return {"summary": "No requests recorded.", "status": "NO_DATA"}

        # Error breakdown by type
        cursor = conn.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM requests
            WHERE error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY count DESC
            """
        )
        error_breakdown = []
        for row in cursor.fetchall():
            pct = (row["count"] / total) * 100
            error_breakdown.append(
                {
                    "type": row["error_type"],
                    "count": row["count"],
                    "pct": round(pct, 2),
                }
            )

        # Error breakdown by status code
        cursor = conn.execute(
            """
            SELECT status_code, COUNT(*) as count
            FROM requests
            WHERE status_code >= 400
            GROUP BY status_code
            ORDER BY count DESC
            """
        )
        status_breakdown = []
        for row in cursor.fetchall():
            pct = (row["count"] / total) * 100
            status_breakdown.append(
                {
                    "status_code": row["status_code"],
                    "count": row["count"],
                    "pct": round(pct, 2),
                }
            )

        # Sample timestamps for each error type
        samples: dict[str, list[str]] = {}
        for error in error_breakdown[:5]:  # Top 5 error types
            cursor = conn.execute(
                """
                SELECT timestamp_utc FROM requests
                WHERE error_type = ?
                ORDER BY timestamp_utc
                LIMIT 3
                """,
                (error["type"],),
            )
            samples[error["type"]] = [row["timestamp_utc"] for row in cursor.fetchall()]

        total_errors = sum(e["count"] for e in error_breakdown)
        error_rate = (total_errors / total) * 100

        top_types = [f"{e['type']}({e['count']})" for e in error_breakdown[:3]]
        summary = (
            f"Error rate: {error_rate:.1f}%. "
            f"Top types: {', '.join(top_types) if top_types else 'none'}. "
            f"Total errors: {total_errors}/{total}."
        )

        return {
            "summary": summary,
            "total_requests": total,
            "total_errors": total_errors,
            "error_rate_pct": round(error_rate, 2),
            "by_error_type": error_breakdown,
            "by_status_code": status_breakdown,
            "sample_timestamps": samples,
        }

    def analyze_latency(self) -> dict[str, Any]:
        """Analyze latency distribution.

        Returns p50/p95/p99, slow request count, and correlation with
        error periods. Designed for ~80 token output.
        """
        conn = self._get_connection()

        # Get all latencies
        cursor = conn.execute("SELECT latency_ms FROM requests WHERE latency_ms IS NOT NULL ORDER BY latency_ms")
        latencies = [row["latency_ms"] for row in cursor.fetchall()]

        if not latencies:
            return {"summary": "No latency data recorded.", "status": "NO_DATA"}

        # Percentiles
        n = len(latencies)
        p50 = latencies[int(n * 0.50)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        p99 = latencies[min(int(n * 0.99), n - 1)] if n > 0 else 0
        avg = sum(latencies) / n if n > 0 else 0
        max_lat = max(latencies) if latencies else 0

        # Slow request threshold (>2x p95)
        slow_threshold = p95 * 2
        slow_count = sum(1 for lat in latencies if lat > slow_threshold)

        # Correlation with errors: check if slow requests correlate with error periods
        cursor = conn.execute(
            """
            SELECT bucket_utc, avg_latency_ms, requests_rate_limited, requests_total
            FROM timeseries
            WHERE avg_latency_ms IS NOT NULL
            ORDER BY bucket_utc
            """
        )
        buckets = [dict(row) for row in cursor.fetchall()]

        # Calculate correlation between latency and error rate
        if len(buckets) >= 3:
            latency_during_errors = []
            latency_during_clean = []
            for bucket in buckets:
                error_rate = bucket["requests_rate_limited"] / bucket["requests_total"] if bucket["requests_total"] > 0 else 0
                if error_rate > 0.1:
                    latency_during_errors.append(bucket["avg_latency_ms"])
                else:
                    latency_during_clean.append(bucket["avg_latency_ms"])

            avg_error_lat = sum(latency_during_errors) / len(latency_during_errors) if latency_during_errors else 0
            avg_clean_lat = sum(latency_during_clean) / len(latency_during_clean) if latency_during_clean else 0

            if avg_error_lat > avg_clean_lat * 1.5:
                correlation = "HIGH: Latency increases during error periods"
            elif avg_error_lat > avg_clean_lat * 1.2:
                correlation = "MODERATE: Some latency increase during errors"
            else:
                correlation = "LOW: Latency stable regardless of errors"
        else:
            correlation = "INSUFFICIENT_DATA"

        summary = (
            f"p50={p50:.0f}ms, p95={p95:.0f}ms, p99={p99:.0f}ms. "
            f"Slow requests (>{slow_threshold:.0f}ms): {slow_count}. "
            f"Error correlation: {correlation}."
        )

        return {
            "summary": summary,
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "avg_ms": round(avg, 2),
            "max_ms": round(max_lat, 2),
            "slow_threshold_ms": round(slow_threshold, 2),
            "slow_count": slow_count,
            "error_correlation": correlation,
        }

    def find_anomalies(self) -> dict[str, Any]:
        """Auto-detect unusual patterns and anomalies.

        Returns unexpected errors, throughput cliffs, and unusual patterns.
        Designed for ~100 token output.
        """
        conn = self._get_connection()

        anomalies: list[dict[str, Any]] = []

        # Check for unexpected status codes (not 200, 429, 500-504, 529)
        expected_codes = {200, 429, 500, 502, 503, 504, 529}
        cursor = conn.execute(
            """
            SELECT status_code, COUNT(*) as count
            FROM requests
            WHERE status_code IS NOT NULL
            GROUP BY status_code
            """
        )
        for row in cursor.fetchall():
            if row["status_code"] not in expected_codes:
                anomalies.append(
                    {
                        "type": "unexpected_status",
                        "status_code": row["status_code"],
                        "count": row["count"],
                    }
                )

        # Check for throughput cliffs (sudden 50%+ drop)
        cursor = conn.execute(
            """
            SELECT bucket_utc, requests_success
            FROM timeseries
            ORDER BY bucket_utc
            """
        )
        buckets = list(cursor.fetchall())
        for i in range(1, len(buckets)):
            prev_success = buckets[i - 1]["requests_success"]
            curr_success = buckets[i]["requests_success"]
            if prev_success > 10 and curr_success < prev_success * 0.5:
                anomalies.append(
                    {
                        "type": "throughput_cliff",
                        "bucket": buckets[i]["bucket_utc"],
                        "drop_pct": round((1 - curr_success / prev_success) * 100, 1),
                    }
                )

        # Check for unusual error clustering (all errors in <10% of time)
        cursor = conn.execute("SELECT COUNT(*) FROM requests WHERE outcome != 'success'")
        total_errors = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(DISTINCT bucket_utc) FROM timeseries")
        total_buckets = cursor.fetchone()[0]

        if total_errors > 10 and total_buckets > 10:
            cursor = conn.execute(
                """
                SELECT COUNT(DISTINCT bucket_utc) FROM timeseries
                WHERE requests_rate_limited > 0 OR requests_capacity_error > 0
                """
            )
            error_buckets = cursor.fetchone()[0]
            if error_buckets < total_buckets * 0.1:
                anomalies.append(
                    {
                        "type": "error_clustering",
                        "error_buckets": error_buckets,
                        "total_buckets": total_buckets,
                        "concentration_pct": round(error_buckets / total_buckets * 100, 1),
                    }
                )

        # Check for zero-success periods
        cursor = conn.execute(
            """
            SELECT COUNT(*) FROM timeseries
            WHERE requests_success = 0 AND requests_total > 0
            """
        )
        zero_success_buckets = cursor.fetchone()[0]
        if zero_success_buckets > 0:
            anomalies.append(
                {
                    "type": "zero_success_periods",
                    "bucket_count": zero_success_buckets,
                }
            )

        if not anomalies:
            summary = "No anomalies detected. Behavior appears normal."
        else:
            anomaly_types = [a["type"] for a in anomalies]
            summary = f"Detected {len(anomalies)} anomaly(ies): {', '.join(anomaly_types)}."

        return {
            "summary": summary,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
        }

    # === Drill-Down Tools ===

    def get_burst_events(self) -> dict[str, Any]:
        """Get detected burst periods with before/during/after stats."""
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT bucket_utc, requests_total, requests_success,
                   requests_rate_limited, requests_capacity_error,
                   avg_latency_ms
            FROM timeseries
            ORDER BY bucket_utc
            """
        )
        buckets = [dict(row) for row in cursor.fetchall()]

        if len(buckets) < 3:
            return {"burst_events": [], "message": "Insufficient data for burst detection"}

        burst_events = []
        in_burst = False
        burst_start_idx = -1

        for i, bucket in enumerate(buckets):
            total = bucket["requests_total"]
            errors = bucket["requests_rate_limited"] + bucket["requests_capacity_error"]
            error_rate = errors / total if total > 0 else 0

            if error_rate > 0.3 and not in_burst:
                # Burst started
                in_burst = True
                burst_start_idx = i
            elif error_rate < 0.1 and in_burst:
                # Burst ended
                in_burst = False
                burst_end_idx = i

                # Collect stats
                before_buckets = buckets[max(0, burst_start_idx - 3) : burst_start_idx]
                during_buckets = buckets[burst_start_idx:burst_end_idx]
                after_buckets = buckets[burst_end_idx : min(len(buckets), burst_end_idx + 3)]

                def _avg_success(b_list: list[dict[str, Any]]) -> float:
                    if not b_list:
                        return 0.0
                    total: int = sum(b["requests_success"] for b in b_list)
                    return total / len(b_list)

                def _avg_latency(b_list: list[dict[str, Any]]) -> float:
                    lats = [b["avg_latency_ms"] for b in b_list if b["avg_latency_ms"] is not None]
                    return sum(lats) / len(lats) if lats else 0.0

                burst_events.append(
                    {
                        "start_bucket": buckets[burst_start_idx]["bucket_utc"],
                        "end_bucket": buckets[burst_end_idx - 1]["bucket_utc"] if burst_end_idx > burst_start_idx else None,
                        "duration_buckets": burst_end_idx - burst_start_idx,
                        "before": {
                            "avg_success": round(_avg_success(before_buckets), 2),
                            "avg_latency_ms": round(_avg_latency(before_buckets), 2),
                        },
                        "during": {
                            "avg_success": round(_avg_success(during_buckets), 2),
                            "avg_latency_ms": round(_avg_latency(during_buckets), 2),
                            "total_rate_limited": sum(b["requests_rate_limited"] for b in during_buckets),
                            "total_capacity_errors": sum(b["requests_capacity_error"] for b in during_buckets),
                        },
                        "after": {
                            "avg_success": round(_avg_success(after_buckets), 2),
                            "avg_latency_ms": round(_avg_latency(after_buckets), 2),
                        },
                    }
                )

        return {
            "burst_count": len(burst_events),
            "burst_events": burst_events,
        }

    def get_error_samples(self, error_type: str, limit: int = 5) -> dict[str, Any]:
        """Get sample requests for a specific error type.

        Args:
            error_type: Error type to filter by (e.g., 'rate_limit', 'timeout')
            limit: Maximum number of samples (default 5)
        """
        conn = self._get_connection()

        cursor = conn.execute(
            """
            SELECT request_id, timestamp_utc, endpoint, status_code,
                   latency_ms, deployment, model
            FROM requests
            WHERE error_type = ?
            ORDER BY timestamp_utc DESC
            LIMIT ?
            """,
            (error_type, limit),
        )
        samples = [dict(row) for row in cursor.fetchall()]

        return {
            "error_type": error_type,
            "sample_count": len(samples),
            "samples": samples,
        }

    def get_time_window(self, start_sec: float, end_sec: float) -> dict[str, Any]:
        """Get statistics for a specific time window.

        Args:
            start_sec: Start of window as Unix timestamp
            end_sec: End of window as Unix timestamp
        """
        conn = self._get_connection()

        # Convert to ISO format for SQLite comparison
        start_iso = datetime.fromtimestamp(start_sec, tz=UTC).isoformat()
        end_iso = datetime.fromtimestamp(end_sec, tz=UTC).isoformat()

        # Request counts in window
        cursor = conn.execute(
            """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as success,
                   SUM(CASE WHEN status_code = 429 THEN 1 ELSE 0 END) as rate_limited,
                   SUM(CASE WHEN status_code = 529 THEN 1 ELSE 0 END) as capacity_errors,
                   AVG(latency_ms) as avg_latency
            FROM requests
            WHERE timestamp_utc >= ? AND timestamp_utc < ?
            """,
            (start_iso, end_iso),
        )
        row = cursor.fetchone()

        # Error breakdown
        cursor = conn.execute(
            """
            SELECT error_type, COUNT(*) as count
            FROM requests
            WHERE timestamp_utc >= ? AND timestamp_utc < ? AND error_type IS NOT NULL
            GROUP BY error_type
            ORDER BY count DESC
            """,
            (start_iso, end_iso),
        )
        errors = {r["error_type"]: r["count"] for r in cursor.fetchall()}

        return {
            "window": {
                "start": start_iso,
                "end": end_iso,
            },
            "total_requests": row["total"] or 0,
            "success_count": row["success"] or 0,
            "rate_limited_count": row["rate_limited"] or 0,
            "capacity_error_count": row["capacity_errors"] or 0,
            "avg_latency_ms": round(row["avg_latency"], 2) if row["avg_latency"] else None,
            "errors_by_type": errors,
        }

    # === Raw Access ===

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query with auto LIMIT 100.

        Args:
            sql: SQL query (must be SELECT)

        Returns:
            Query results as list of dicts

        Raises:
            ValueError: If query is not SELECT or contains dangerous keywords
        """
        # Safety: only allow SELECT
        sql_normalized = sql.strip().upper()
        if not sql_normalized.startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed")

        # Reject dangerous keywords
        dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", "TRUNCATE"]
        for keyword in dangerous:
            if keyword in sql_normalized:
                raise ValueError(f"Query contains forbidden keyword: {keyword}")

        # Auto-add LIMIT if not present
        if "LIMIT" not in sql_normalized:
            sql = f"{sql.rstrip(';')} LIMIT 100"

        conn = self._get_connection()
        cursor = conn.execute(sql)

        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return [dict(zip(columns, row, strict=True)) for row in rows]

    def describe_schema(self) -> dict[str, Any]:
        """Describe the metrics database schema."""
        return {
            "tables": {
                "requests": {
                    "description": "Individual request records",
                    "columns": [
                        "request_id (TEXT PK)",
                        "timestamp_utc (TEXT)",
                        "endpoint (TEXT)",
                        "deployment (TEXT)",
                        "model (TEXT)",
                        "outcome (TEXT: success/error_injected/error_malformed)",
                        "status_code (INTEGER)",
                        "error_type (TEXT)",
                        "injection_type (TEXT)",
                        "latency_ms (REAL)",
                        "injected_delay_ms (REAL)",
                        "message_count (INTEGER)",
                        "prompt_tokens_approx (INTEGER)",
                        "response_tokens (INTEGER)",
                        "response_mode (TEXT)",
                    ],
                },
                "timeseries": {
                    "description": "Time-bucketed aggregations",
                    "columns": [
                        "bucket_utc (TEXT PK)",
                        "requests_total (INTEGER)",
                        "requests_success (INTEGER)",
                        "requests_rate_limited (INTEGER)",
                        "requests_capacity_error (INTEGER)",
                        "requests_server_error (INTEGER)",
                        "requests_client_error (INTEGER)",
                        "requests_connection_error (INTEGER)",
                        "requests_malformed (INTEGER)",
                        "avg_latency_ms (REAL)",
                        "p99_latency_ms (REAL)",
                    ],
                },
                "run_info": {
                    "description": "Run metadata",
                    "columns": [
                        "run_id (TEXT PK)",
                        "started_utc (TEXT)",
                        "config_json (TEXT)",
                        "preset_name (TEXT)",
                    ],
                },
            },
            "hint": "Use query() with SELECT statements to explore data",
        }


def create_server(database_path: str) -> Server:
    """Create MCP server with ChaosLLM analysis tools.

    Args:
        database_path: Path to SQLite metrics database

    Returns:
        Configured MCP Server
    """
    server = Server("chaosllm-analysis")
    analyzer = ChaosLLMAnalyzer(database_path)

    @server.list_tools()  # type: ignore[misc, no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """List available analysis tools."""
        return [
            # === High-Level Analysis Tools ===
            Tool(
                name="diagnose",
                description=(
                    "One-paragraph summary: total requests, success rate, top 3 error "
                    "types, patterns detected, AIMD assessment. Start here."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="analyze_aimd_behavior",
                description=("AIMD analysis: recovery times after bursts, backoff effectiveness, throughput degradation percentage."),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="analyze_errors",
                description=("Error breakdown: grouped by category with counts, percentages, and sample timestamps."),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="analyze_latency",
                description=("Latency analysis: p50/p95/p99, slow request count, correlation with error periods."),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="find_anomalies",
                description=("Auto-detect unusual patterns: unexpected errors, throughput cliffs, error clustering."),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            # === Drill-Down Tools ===
            Tool(
                name="get_burst_events",
                description="Get detected burst periods with before/during/after stats.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="get_error_samples",
                description="Get sample requests for a specific error type.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "error_type": {
                            "type": "string",
                            "description": "Error type (e.g., 'rate_limit', 'timeout', 'connection_reset')",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max samples to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["error_type"],
                },
            ),
            Tool(
                name="get_time_window",
                description="Get stats for a specific time range (Unix timestamps).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "start_sec": {
                            "type": "number",
                            "description": "Start time as Unix timestamp",
                        },
                        "end_sec": {
                            "type": "number",
                            "description": "End time as Unix timestamp",
                        },
                    },
                    "required": ["start_sec", "end_sec"],
                },
            ),
            # === Raw Access ===
            Tool(
                name="query",
                description="Execute read-only SQL (SELECT only, auto LIMIT 100).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string", "description": "SQL SELECT query"},
                    },
                    "required": ["sql"],
                },
            ),
            Tool(
                name="describe_schema",
                description="Show database schema (tables, columns) for writing queries.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()  # type: ignore[misc, untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Handle tool calls."""
        try:
            result: Any
            if name == "diagnose":
                result = analyzer.diagnose()
            elif name == "analyze_aimd_behavior":
                result = analyzer.analyze_aimd_behavior()
            elif name == "analyze_errors":
                result = analyzer.analyze_errors()
            elif name == "analyze_latency":
                result = analyzer.analyze_latency()
            elif name == "find_anomalies":
                result = analyzer.find_anomalies()
            elif name == "get_burst_events":
                result = analyzer.get_burst_events()
            elif name == "get_error_samples":
                result = analyzer.get_error_samples(
                    error_type=arguments["error_type"],
                    limit=arguments.get("limit", 5),
                )
            elif name == "get_time_window":
                result = analyzer.get_time_window(
                    start_sec=arguments["start_sec"],
                    end_sec=arguments["end_sec"],
                )
            elif name == "query":
                result = analyzer.query(arguments["sql"])
            elif name == "describe_schema":
                result = analyzer.describe_schema()
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e!s}")]

    return server


async def run_server(database_path: str) -> None:
    """Run the MCP server with stdio transport."""
    server = create_server(database_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _find_metrics_databases(search_dir: str, max_depth: int = 3) -> list[str]:
    """Find potential ChaosLLM metrics databases.

    Args:
        search_dir: Directory to search from
        max_depth: Maximum directory depth

    Returns:
        List of database paths sorted by relevance
    """
    found: list[tuple[int, float, str]] = []
    search_path = Path(search_dir).resolve()

    for db_file in search_path.rglob("*.db"):
        parts = db_file.relative_to(search_path).parts
        if any(p.startswith(".") for p in parts):
            continue
        if len(parts) > max_depth:
            continue

        name = db_file.name.lower()

        # Prioritize chaosllm metrics files
        if "chaosllm" in name and "metrics" in name:
            priority = 0
        elif "chaosllm" in name:
            priority = 1
        elif "metrics" in name:
            priority = 2
        else:
            priority = 10

        try:
            mtime = db_file.stat().st_mtime
        except OSError:
            mtime = 0

        found.append((priority, -mtime, str(db_file)))

    found.sort(key=lambda x: (x[0], x[1]))
    return [path for _, _, path in found]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ChaosLLM Metrics MCP Server - Analysis tools for load testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with specific database
    chaosllm-mcp --database ./chaosllm-metrics.db

    # Auto-discover in current directory
    chaosllm-mcp
""",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="Path to SQLite metrics database",
    )
    parser.add_argument(
        "--search-dir",
        default=".",
        help="Directory to search for databases (default: current)",
    )

    args = parser.parse_args()

    database_path: str | None = args.database

    if database_path is None:
        # Auto-discovery
        databases = _find_metrics_databases(args.search_dir)

        if not databases:
            sys.stderr.write(f"No metrics databases found in {Path(args.search_dir).resolve()}\n")
            sys.stderr.write("Use --database to specify a database path directly.\n")
            sys.exit(1)

        database_path = databases[0]
        sys.stderr.write(f"Using database: {database_path}\n")

    if not Path(database_path).exists():
        sys.stderr.write(f"Database not found: {database_path}\n")
        sys.exit(1)

    import asyncio

    asyncio.run(run_server(database_path))


if __name__ == "__main__":
    main()
