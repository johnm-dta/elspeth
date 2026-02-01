# tests/testing/chaosllm_mcp/test_server.py
"""Tests for ChaosLLM MCP server."""

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from elspeth.testing.chaosllm_mcp.server import ChaosLLMAnalyzer, create_server

# === Fixtures ===


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary metrics database with schema."""
    db_path = tmp_path / "test_metrics.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
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
        """
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def empty_analyzer(temp_db: Path) -> ChaosLLMAnalyzer:
    """Create analyzer with empty database."""
    analyzer = ChaosLLMAnalyzer(str(temp_db))
    yield analyzer
    analyzer.close()


@pytest.fixture
def populated_analyzer(temp_db: Path) -> ChaosLLMAnalyzer:
    """Create analyzer with pre-populated test data."""
    conn = sqlite3.connect(str(temp_db))

    # Insert test requests
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)

    requests = [
        # 10 successful requests
        *[
            (
                f"req-success-{i}",
                (base_time + timedelta(seconds=i)).isoformat(),
                "/chat/completions",
                "gpt-4",
                "gpt-4",
                "success",
                200,
                None,
                None,
                100.0 + i * 10,  # latency varies
                None,
                3,
                100,
                50,
                "random",
            )
            for i in range(10)
        ],
        # 3 rate limited requests (429)
        *[
            (
                f"req-429-{i}",
                (base_time + timedelta(seconds=20 + i)).isoformat(),
                "/chat/completions",
                "gpt-4",
                "gpt-4",
                "error_injected",
                429,
                "rate_limit",
                "rate_limit",
                None,
                None,
                3,
                100,
                None,
                None,
            )
            for i in range(3)
        ],
        # 2 capacity errors (529)
        *[
            (
                f"req-529-{i}",
                (base_time + timedelta(seconds=25 + i)).isoformat(),
                "/chat/completions",
                "gpt-4",
                "gpt-4",
                "error_injected",
                529,
                "capacity",
                "capacity",
                None,
                None,
                3,
                100,
                None,
                None,
            )
            for i in range(2)
        ],
        # 1 timeout
        (
            "req-timeout-1",
            (base_time + timedelta(seconds=30)).isoformat(),
            "/chat/completions",
            "gpt-4",
            "gpt-4",
            "error_injected",
            None,
            "timeout",
            "timeout",
            None,
            None,
            3,
            100,
            None,
            None,
        ),
    ]

    conn.executemany(
        """
        INSERT INTO requests (
            request_id, timestamp_utc, endpoint, deployment, model,
            outcome, status_code, error_type, injection_type, latency_ms, injected_delay_ms,
            message_count, prompt_tokens_approx, response_tokens, response_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        requests,
    )

    # Insert timeseries buckets
    timeseries = [
        # First bucket: all success
        (
            (base_time + timedelta(seconds=0)).isoformat(),
            5,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            110.0,
            140.0,
        ),
        # Second bucket: some success
        (
            (base_time + timedelta(seconds=5)).isoformat(),
            5,
            5,
            0,
            0,
            0,
            0,
            0,
            0,
            160.0,
            180.0,
        ),
        # Third bucket: errors
        (
            (base_time + timedelta(seconds=20)).isoformat(),
            3,
            0,
            3,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
        ),
        # Fourth bucket: capacity errors
        (
            (base_time + timedelta(seconds=25)).isoformat(),
            3,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            None,
            None,
        ),
    ]

    conn.executemany(
        """
        INSERT INTO timeseries (
            bucket_utc, requests_total, requests_success, requests_rate_limited,
            requests_capacity_error, requests_server_error, requests_client_error,
            requests_connection_error, requests_malformed, avg_latency_ms, p99_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        timeseries,
    )

    conn.commit()
    conn.close()

    analyzer = ChaosLLMAnalyzer(str(temp_db))
    yield analyzer
    analyzer.close()


# === Test ChaosLLMAnalyzer ===


class TestDiagnose:
    """Tests for diagnose() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns NO_DATA for empty database."""
        result = empty_analyzer.diagnose()
        assert result["status"] == "NO_DATA"
        assert "No requests" in result["summary"]

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns summary for populated database."""
        result = populated_analyzer.diagnose()

        assert result["status"] in ("OK", "WARNING", "CRITICAL")
        assert result["total_requests"] == 16
        assert result["success_rate_pct"] > 0
        assert "summary" in result
        assert len(result["top_errors"]) > 0

    def test_aimd_assessment_present(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """AIMD assessment is present in result."""
        result = populated_analyzer.diagnose()
        assert "aimd_assessment" in result
        assert result["rate_limit_pct"] > 0


class TestAnalyzeAimdBehavior:
    """Tests for analyze_aimd_behavior() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns NO_DATA for empty database."""
        result = empty_analyzer.analyze_aimd_behavior()
        assert result["status"] == "NO_DATA"

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns AIMD analysis for populated database."""
        result = populated_analyzer.analyze_aimd_behavior()

        assert "summary" in result
        assert "burst_count" in result
        assert "backoff_ratio" in result
        assert "backoff_assessment" in result


class TestAnalyzeErrors:
    """Tests for analyze_errors() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns NO_DATA for empty database."""
        result = empty_analyzer.analyze_errors()
        assert result["status"] == "NO_DATA"

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns error breakdown for populated database."""
        result = populated_analyzer.analyze_errors()

        assert result["total_requests"] == 16
        assert result["total_errors"] > 0
        assert "by_error_type" in result
        assert "by_status_code" in result

        # Check error types are present
        error_types = [e["type"] for e in result["by_error_type"]]
        assert "rate_limit" in error_types

    def test_sample_timestamps(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Sample timestamps are provided for error types."""
        result = populated_analyzer.analyze_errors()

        assert "sample_timestamps" in result
        assert len(result["sample_timestamps"]) > 0


class TestAnalyzeLatency:
    """Tests for analyze_latency() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns NO_DATA for empty database."""
        result = empty_analyzer.analyze_latency()
        assert result["status"] == "NO_DATA"

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns latency stats for populated database."""
        result = populated_analyzer.analyze_latency()

        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "avg_ms" in result
        assert "max_ms" in result
        assert result["p50_ms"] > 0


class TestFindAnomalies:
    """Tests for find_anomalies() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns no anomalies for empty database."""
        result = empty_analyzer.find_anomalies()
        assert result["anomaly_count"] == 0

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns anomalies for populated database."""
        result = populated_analyzer.find_anomalies()

        assert "summary" in result
        assert "anomalies" in result


class TestGetBurstEvents:
    """Tests for get_burst_events() tool."""

    def test_empty_database(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns empty list for empty database."""
        result = empty_analyzer.get_burst_events()
        assert result["burst_events"] == []

    def test_populated_database(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns burst events for populated database."""
        result = populated_analyzer.get_burst_events()
        assert "burst_count" in result
        assert "burst_events" in result


class TestGetErrorSamples:
    """Tests for get_error_samples() tool."""

    def test_no_matching_errors(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns empty samples when no errors match."""
        result = empty_analyzer.get_error_samples("nonexistent_type")
        assert result["sample_count"] == 0
        assert result["samples"] == []

    def test_matching_errors(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns samples for matching error type."""
        result = populated_analyzer.get_error_samples("rate_limit", limit=5)

        assert result["error_type"] == "rate_limit"
        assert result["sample_count"] == 3  # We inserted 3 rate limit errors
        assert len(result["samples"]) == 3

    def test_limit_respected(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Limit parameter is respected."""
        result = populated_analyzer.get_error_samples("rate_limit", limit=1)
        assert result["sample_count"] == 1


class TestGetTimeWindow:
    """Tests for get_time_window() tool."""

    def test_empty_window(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns zeros for empty time window."""
        result = empty_analyzer.get_time_window(
            start_sec=0,
            end_sec=1000000000,
        )
        assert result["total_requests"] == 0

    def test_populated_window(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns stats for populated time window."""
        # Use a wide window that includes all test data
        base_ts = datetime(2024, 1, 15, 0, 0, 0, tzinfo=UTC).timestamp()
        result = populated_analyzer.get_time_window(
            start_sec=base_ts,
            end_sec=base_ts + 86400,  # +1 day
        )

        assert result["total_requests"] == 16
        assert result["success_count"] == 10
        assert result["rate_limited_count"] == 3


class TestQuery:
    """Tests for raw SQL query() tool."""

    def test_select_query(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """SELECT queries work."""
        result = populated_analyzer.query("SELECT COUNT(*) as cnt FROM requests")
        assert len(result) == 1
        assert result[0]["cnt"] == 16

    def test_non_select_rejected(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Non-SELECT queries are rejected."""
        with pytest.raises(ValueError, match="Only SELECT"):
            populated_analyzer.query("INSERT INTO requests VALUES ('x')")

    def test_dangerous_keywords_rejected(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Dangerous keywords in SELECT are rejected."""
        with pytest.raises(ValueError, match="forbidden keyword"):
            populated_analyzer.query("SELECT * FROM requests; DROP TABLE requests")

    def test_auto_limit_added(self, populated_analyzer: ChaosLLMAnalyzer) -> None:
        """Auto LIMIT 100 is added when missing."""
        result = populated_analyzer.query("SELECT * FROM requests")
        # Should work and return <= 100 results
        assert len(result) <= 100


class TestDescribeSchema:
    """Tests for describe_schema() tool."""

    def test_returns_schema(self, empty_analyzer: ChaosLLMAnalyzer) -> None:
        """Returns schema description."""
        result = empty_analyzer.describe_schema()

        assert "tables" in result
        assert "requests" in result["tables"]
        assert "timeseries" in result["tables"]
        assert "run_info" in result["tables"]


# === Test MCP Server Creation ===


class TestCreateServer:
    """Tests for MCP server creation."""

    def test_creates_server(self, temp_db: Path) -> None:
        """Server can be created."""
        server = create_server(str(temp_db))
        assert server is not None
        assert server.name == "chaosllm-analysis"


# === Integration Tests with Actual MCP Protocol ===


class TestMCPServerTools:
    """Tests for MCP server creation and basic functionality."""

    def test_server_has_name(self, temp_db: Path) -> None:
        """Server has correct name."""
        server = create_server(str(temp_db))
        assert server.name == "chaosllm-analysis"

    def test_diagnose_via_analyzer(self, temp_db: Path) -> None:
        """Diagnose tool can be called via analyzer."""
        # We test through the analyzer directly since MCP protocol testing
        # requires a full stdio server setup
        from elspeth.testing.chaosllm_mcp.server import ChaosLLMAnalyzer

        analyzer = ChaosLLMAnalyzer(str(temp_db))
        result = analyzer.diagnose()
        assert "status" in result
        analyzer.close()
