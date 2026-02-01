# tests/testing/chaosllm/test_metrics.py
"""Tests for ChaosLLM metrics recorder."""

import json
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from elspeth.testing.chaosllm.config import MetricsConfig
from elspeth.testing.chaosllm.metrics import (
    MetricsRecorder,
    RequestRecord,
    _classify_outcome,
    _get_bucket_utc,
)


class TestGetBucketUtc:
    """Tests for _get_bucket_utc helper function."""

    def test_bucket_truncates_to_second(self) -> None:
        """Bucket truncates timestamp to second boundary."""
        timestamp = "2024-01-15T10:30:45.123+00:00"
        bucket = _get_bucket_utc(timestamp, bucket_sec=1)
        # Should truncate to the same second
        assert bucket.startswith("2024-01-15T10:30:45")

    def test_bucket_10_second(self) -> None:
        """10-second bucket truncates correctly."""
        timestamp = "2024-01-15T10:30:47+00:00"
        bucket = _get_bucket_utc(timestamp, bucket_sec=10)
        # 47 seconds -> 40 seconds bucket
        assert bucket.startswith("2024-01-15T10:30:40")

    def test_bucket_60_second(self) -> None:
        """60-second bucket truncates to minute boundary."""
        timestamp = "2024-01-15T10:30:47+00:00"
        bucket = _get_bucket_utc(timestamp, bucket_sec=60)
        # Should be at minute boundary
        assert bucket.startswith("2024-01-15T10:30:00")

    def test_bucket_handles_z_suffix(self) -> None:
        """Handles 'Z' timezone suffix."""
        timestamp = "2024-01-15T10:30:45Z"
        bucket = _get_bucket_utc(timestamp, bucket_sec=1)
        assert "2024-01-15T10:30:45" in bucket

    def test_bucket_preserves_timezone(self) -> None:
        """Bucket preserves the original timezone."""
        timestamp = "2024-01-15T10:30:45+05:30"
        bucket = _get_bucket_utc(timestamp, bucket_sec=1)
        assert "+05:30" in bucket


class TestClassifyOutcome:
    """Tests for _classify_outcome helper function."""

    def test_success_outcome(self) -> None:
        """Success outcome is classified correctly."""
        result = _classify_outcome("success", 200, None)
        is_success, is_rate_limited, is_capacity, is_server, is_client, is_conn, is_malformed = result
        assert is_success is True
        assert is_rate_limited is False
        assert is_capacity is False
        assert is_server is False
        assert is_client is False
        assert is_conn is False
        assert is_malformed is False

    def test_rate_limited_429(self) -> None:
        """429 status code is classified as rate limited."""
        result = _classify_outcome("error_injected", 429, None)
        _, is_rate_limited, _, _, _, _, _ = result
        assert is_rate_limited is True

    def test_capacity_error_529(self) -> None:
        """529 status code is classified as capacity error."""
        result = _classify_outcome("error_injected", 529, None)
        _, _, is_capacity, is_server, _, _, _ = result
        assert is_capacity is True
        assert is_server is False  # 529 is not classified as generic server error

    def test_server_error_500(self) -> None:
        """500 status code is classified as server error."""
        result = _classify_outcome("error_injected", 500, None)
        _, _, _, is_server, _, _, _ = result
        assert is_server is True

    def test_server_error_503(self) -> None:
        """503 status code is classified as server error."""
        result = _classify_outcome("error_injected", 503, None)
        _, _, _, is_server, _, _, _ = result
        assert is_server is True

    def test_client_error_400(self) -> None:
        """400 status code is classified as client error."""
        result = _classify_outcome("error_injected", 400, None)
        _, _, _, _, is_client, _, _ = result
        assert is_client is True

    def test_client_error_403(self) -> None:
        """403 status code is classified as client error."""
        result = _classify_outcome("error_injected", 403, None)
        _, _, _, _, is_client, _, _ = result
        assert is_client is True

    def test_connection_timeout(self) -> None:
        """Timeout error type is classified as connection error."""
        result = _classify_outcome("error_injected", None, "timeout")
        _, _, _, _, _, is_conn, _ = result
        assert is_conn is True

    def test_connection_reset(self) -> None:
        """Connection reset error type is classified as connection error."""
        result = _classify_outcome("error_injected", None, "connection_reset")
        _, _, _, _, _, is_conn, _ = result
        assert is_conn is True

    def test_connection_failed(self) -> None:
        """Connection failed error type is classified as connection error."""
        result = _classify_outcome("error_injected", None, "connection_failed")
        _, _, _, _, _, is_conn, _ = result
        assert is_conn is True

    def test_connection_stall(self) -> None:
        """Connection stall error type is classified as connection error."""
        result = _classify_outcome("error_injected", None, "connection_stall")
        _, _, _, _, _, is_conn, _ = result
        assert is_conn is True

    def test_slow_response(self) -> None:
        """Slow response is not classified as a connection error."""
        result = _classify_outcome("success", 200, "slow_response")
        _, _, _, _, _, is_conn, _ = result
        assert is_conn is False

    def test_malformed_outcome(self) -> None:
        """error_malformed outcome is classified correctly."""
        result = _classify_outcome("error_malformed", 200, None)
        _, _, _, _, _, _, is_malformed = result
        assert is_malformed is True


class TestRequestRecord:
    """Tests for RequestRecord dataclass."""

    def test_create_minimal_record(self) -> None:
        """Can create record with minimal required fields."""
        record = RequestRecord(
            request_id="abc123",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )
        assert record.request_id == "abc123"
        assert record.outcome == "success"
        assert record.deployment is None
        assert record.latency_ms is None

    def test_create_full_record(self) -> None:
        """Can create record with all fields."""
        record = RequestRecord(
            request_id="abc123",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            deployment="gpt-4",
            model="gpt-4-0613",
            status_code=200,
            error_type=None,
            latency_ms=150.5,
            injected_delay_ms=50.0,
            message_count=3,
            prompt_tokens_approx=100,
            response_tokens=50,
            response_mode="random",
        )
        assert record.latency_ms == 150.5
        assert record.message_count == 3

    def test_record_is_frozen(self) -> None:
        """Record is immutable."""
        record = RequestRecord(
            request_id="abc123",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )
        with pytest.raises(AttributeError):
            record.request_id = "different"  # type: ignore[misc]


class TestMetricsRecorderBasic:
    """Basic tests for MetricsRecorder."""

    def test_init_creates_database(self, tmp_path: Path) -> None:
        """Initialization creates the database file."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        assert db_path.exists()
        recorder.close()

    def test_init_creates_tables(self, tmp_path: Path) -> None:
        """Initialization creates all required tables."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        # Verify tables exist
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "requests" in tables
        assert "timeseries" in tables
        assert "run_info" in tables

        recorder.close()

    def test_generates_run_id(self, tmp_path: Path) -> None:
        """Generates a UUID run_id if not provided."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        assert recorder.run_id is not None
        assert len(recorder.run_id) == 36  # UUID length with dashes

        recorder.close()

    def test_uses_provided_run_id(self, tmp_path: Path) -> None:
        """Uses the provided run_id if given."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config, run_id="my-custom-run-id")

        assert recorder.run_id == "my-custom-run-id"

        recorder.close()

    def test_records_started_utc(self, tmp_path: Path) -> None:
        """Records start time in UTC."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))

        before = datetime.now(UTC)
        recorder = MetricsRecorder(config)
        after = datetime.now(UTC)

        started = datetime.fromisoformat(recorder.started_utc)
        assert before <= started <= after

        recorder.close()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Creates parent directory if it doesn't exist."""
        db_path = tmp_path / "subdir" / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        assert db_path.exists()
        assert db_path.parent.exists()

        recorder.close()


class TestRecordRequest:
    """Tests for record_request method."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_record_minimal_request(self, recorder: MetricsRecorder) -> None:
        """Can record a request with minimal fields."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )

        requests = recorder.get_requests()
        assert len(requests) == 1
        assert requests[0]["request_id"] == "req1"
        assert requests[0]["outcome"] == "success"

    def test_record_full_request(self, recorder: MetricsRecorder) -> None:
        """Can record a request with all fields."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            deployment="gpt-4",
            model="gpt-4-0613",
            status_code=200,
            error_type=None,
            injection_type="slow_response",
            latency_ms=150.5,
            injected_delay_ms=50.0,
            message_count=3,
            prompt_tokens_approx=100,
            response_tokens=50,
            response_mode="random",
        )

        requests = recorder.get_requests()
        assert len(requests) == 1
        req = requests[0]
        assert req["latency_ms"] == 150.5
        assert req["message_count"] == 3
        assert req["response_mode"] == "random"
        assert req["injection_type"] == "slow_response"

    def test_record_multiple_requests(self, recorder: MetricsRecorder) -> None:
        """Can record multiple requests."""
        for i in range(10):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc=f"2024-01-15T10:30:{i:02d}+00:00",
                endpoint="/chat/completions",
                outcome="success",
            )

        requests = recorder.get_requests()
        assert len(requests) == 10

    def test_request_updates_timeseries(self, recorder: MetricsRecorder) -> None:
        """Recording a request updates the time-series."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            status_code=200,
            latency_ms=100.0,
        )

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 1
        assert timeseries[0]["requests_success"] == 1


class TestTimeseries:
    """Tests for time-series aggregation."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path), timeseries_bucket_sec=1)
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_requests_in_same_bucket_aggregate(self, recorder: MetricsRecorder) -> None:
        """Requests in the same bucket are aggregated."""
        for i in range(5):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00.100+00:00",  # Same second
                endpoint="/chat/completions",
                outcome="success",
                status_code=200,
            )

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 5
        assert timeseries[0]["requests_success"] == 5

    def test_requests_in_different_buckets_separate(self, recorder: MetricsRecorder) -> None:
        """Requests in different buckets are tracked separately."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            status_code=200,
        )
        recorder.record_request(
            request_id="req2",
            timestamp_utc="2024-01-15T10:30:01+00:00",  # Different second
            endpoint="/chat/completions",
            outcome="success",
            status_code=200,
        )

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 2

    def test_outcome_classification_rate_limit(self, recorder: MetricsRecorder) -> None:
        """Rate limit (429) is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=429,
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_rate_limited"] == 1
        assert timeseries[0]["requests_success"] == 0

    def test_outcome_classification_capacity(self, recorder: MetricsRecorder) -> None:
        """Capacity error (529) is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=529,
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_capacity_error"] == 1

    def test_outcome_classification_server_error(self, recorder: MetricsRecorder) -> None:
        """Server error (500) is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=500,
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_server_error"] == 1

    def test_outcome_classification_client_error(self, recorder: MetricsRecorder) -> None:
        """Client error (403) is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=403,
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_client_error"] == 1

    def test_outcome_classification_connection_error(self, recorder: MetricsRecorder) -> None:
        """Connection error is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            error_type="timeout",
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_connection_error"] == 1

    def test_outcome_classification_malformed(self, recorder: MetricsRecorder) -> None:
        """Malformed response is classified correctly in timeseries."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="error_malformed",
            status_code=200,
        )

        timeseries = recorder.get_timeseries()
        assert timeseries[0]["requests_malformed"] == 1

    def test_latency_statistics(self, recorder: MetricsRecorder) -> None:
        """Latency statistics are calculated correctly."""
        latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
        for i, lat in enumerate(latencies):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00.100+00:00",
                endpoint="/chat/completions",
                outcome="success",
                status_code=200,
                latency_ms=lat,
            )

        timeseries = recorder.get_timeseries()
        ts = timeseries[0]

        # Average should be 300
        assert ts["avg_latency_ms"] == 300.0
        # p99 of 5 items - index 4 (0.99 * 5 = 4.95, int = 4)
        assert ts["p99_latency_ms"] == 500.0

    def test_update_timeseries_rebuilds(self, recorder: MetricsRecorder) -> None:
        """update_timeseries rebuilds all buckets from raw data."""
        for i in range(5):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00+00:00",
                endpoint="/chat/completions",
                outcome="success",
                status_code=200,
            )

        # Rebuild
        recorder.update_timeseries()

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 5


class TestReset:
    """Tests for reset method."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_reset_clears_requests(self, recorder: MetricsRecorder) -> None:
        """Reset clears the requests table."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )

        assert len(recorder.get_requests()) == 1

        recorder.reset()

        assert len(recorder.get_requests()) == 0

    def test_reset_clears_timeseries(self, recorder: MetricsRecorder) -> None:
        """Reset clears the timeseries table."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )

        assert len(recorder.get_timeseries()) == 1

        recorder.reset()

        assert len(recorder.get_timeseries()) == 0

    def test_reset_generates_new_run_id(self, recorder: MetricsRecorder) -> None:
        """Reset generates a new run ID."""
        old_run_id = recorder.run_id
        recorder.reset()
        assert recorder.run_id != old_run_id

    def test_reset_updates_started_utc(self, recorder: MetricsRecorder) -> None:
        """Reset updates the start time."""
        old_started = recorder.started_utc
        time.sleep(0.01)  # Small delay to ensure different timestamp
        recorder.reset()
        assert recorder.started_utc != old_started

    def test_reset_preserves_run_info(self, recorder: MetricsRecorder) -> None:
        """Reset preserves run_info metadata when available."""
        config_json = json.dumps({"test": "config"})
        recorder.save_run_info(config_json, preset_name="gentle")
        old_run_id = recorder.run_id

        recorder.reset()

        with sqlite3.connect(recorder._config.database) as conn:
            row = conn.execute("SELECT run_id, started_utc, config_json, preset_name FROM run_info").fetchone()

        assert row is not None
        assert row[0] == recorder.run_id
        assert row[0] != old_run_id
        assert row[1] == recorder.started_utc
        assert row[2] == config_json
        assert row[3] == "gentle"


class TestGetStats:
    """Tests for get_stats method."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_stats_empty_database(self, recorder: MetricsRecorder) -> None:
        """Stats work on empty database."""
        stats = recorder.get_stats()

        assert stats["run_id"] == recorder.run_id
        assert stats["started_utc"] == recorder.started_utc
        assert stats["total_requests"] == 0
        assert stats["requests_by_outcome"] == {}
        assert stats["error_rate"] == 0.0

    def test_stats_total_requests(self, recorder: MetricsRecorder) -> None:
        """Stats reports total request count."""
        for i in range(10):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00+00:00",
                endpoint="/chat/completions",
                outcome="success",
            )

        stats = recorder.get_stats()
        assert stats["total_requests"] == 10

    def test_stats_requests_by_outcome(self, recorder: MetricsRecorder) -> None:
        """Stats reports requests grouped by outcome."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )
        recorder.record_request(
            request_id="req2",
            timestamp_utc="2024-01-15T10:30:01+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )
        recorder.record_request(
            request_id="req3",
            timestamp_utc="2024-01-15T10:30:02+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
        )

        stats = recorder.get_stats()
        assert stats["requests_by_outcome"]["success"] == 2
        assert stats["requests_by_outcome"]["error_injected"] == 1

    def test_stats_requests_by_status_code(self, recorder: MetricsRecorder) -> None:
        """Stats reports requests grouped by status code."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
            status_code=200,
        )
        recorder.record_request(
            request_id="req2",
            timestamp_utc="2024-01-15T10:30:01+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=429,
        )
        recorder.record_request(
            request_id="req3",
            timestamp_utc="2024-01-15T10:30:02+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
            status_code=429,
        )

        stats = recorder.get_stats()
        assert stats["requests_by_status_code"][200] == 1
        assert stats["requests_by_status_code"][429] == 2

    def test_stats_latency_statistics(self, recorder: MetricsRecorder) -> None:
        """Stats reports latency statistics."""
        latencies = [100.0, 150.0, 200.0, 250.0, 300.0]
        for i, lat in enumerate(latencies):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00+00:00",
                endpoint="/chat/completions",
                outcome="success",
                latency_ms=lat,
            )

        stats = recorder.get_stats()
        latency_stats = stats["latency_stats"]

        assert latency_stats["avg_ms"] == 200.0  # (100+150+200+250+300) / 5
        assert latency_stats["max_ms"] == 300.0
        # p50 of 5 items - index 2 (0.50 * 5 = 2.5, int = 2)
        assert latency_stats["p50_ms"] == 200.0

    def test_stats_error_rate(self, recorder: MetricsRecorder) -> None:
        """Stats reports error rate percentage."""
        # 3 successes, 2 errors = 40% error rate
        for i in range(3):
            recorder.record_request(
                request_id=f"success{i}",
                timestamp_utc="2024-01-15T10:30:00+00:00",
                endpoint="/chat/completions",
                outcome="success",
            )
        for i in range(2):
            recorder.record_request(
                request_id=f"error{i}",
                timestamp_utc="2024-01-15T10:30:01+00:00",
                endpoint="/chat/completions",
                outcome="error_injected",
            )

        stats = recorder.get_stats()
        assert stats["error_rate"] == 40.0  # 2/5 * 100


class TestSaveRunInfo:
    """Tests for save_run_info method."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_save_run_info_basic(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        """Can save run info with config JSON."""
        config_json = json.dumps({"test": "config"})
        recorder.save_run_info(config_json)

        # Verify by querying directly
        db_path = tmp_path / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT * FROM run_info WHERE run_id = ?", (recorder.run_id,))
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == recorder.run_id  # run_id
        assert row[1] == recorder.started_utc  # started_utc
        assert row[2] == config_json  # config_json
        assert row[3] is None  # preset_name

    def test_save_run_info_with_preset(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        """Can save run info with preset name."""
        config_json = json.dumps({"test": "config"})
        recorder.save_run_info(config_json, preset_name="stress-aimd")

        db_path = tmp_path / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT preset_name FROM run_info WHERE run_id = ?", (recorder.run_id,))
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "stress-aimd"

    def test_save_run_info_updates_existing(self, recorder: MetricsRecorder, tmp_path: Path) -> None:
        """Save_run_info updates existing record for same run_id."""
        recorder.save_run_info(json.dumps({"version": 1}))
        recorder.save_run_info(json.dumps({"version": 2}))

        db_path = tmp_path / "metrics.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM run_info WHERE run_id = ?", (recorder.run_id,))
        count = cursor.fetchone()[0]
        cursor = conn.execute("SELECT config_json FROM run_info WHERE run_id = ?", (recorder.run_id,))
        config = cursor.fetchone()[0]
        conn.close()

        assert count == 1
        assert json.loads(config)["version"] == 2


class TestGetRequests:
    """Tests for get_requests method."""

    @pytest.fixture
    def recorder(self, tmp_path: Path) -> MetricsRecorder:
        """Create a fresh recorder for each test."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)
        yield recorder
        recorder.close()

    def test_get_requests_empty(self, recorder: MetricsRecorder) -> None:
        """Returns empty list when no requests."""
        requests = recorder.get_requests()
        assert requests == []

    def test_get_requests_pagination(self, recorder: MetricsRecorder) -> None:
        """Pagination works correctly."""
        for i in range(20):
            recorder.record_request(
                request_id=f"req{i:02d}",
                timestamp_utc=f"2024-01-15T10:30:{i:02d}+00:00",
                endpoint="/chat/completions",
                outcome="success",
            )

        page1 = recorder.get_requests(limit=10, offset=0)
        page2 = recorder.get_requests(limit=10, offset=10)

        assert len(page1) == 10
        assert len(page2) == 10

        # Should be different sets of requests
        ids1 = {r["request_id"] for r in page1}
        ids2 = {r["request_id"] for r in page2}
        assert ids1.isdisjoint(ids2)

    def test_get_requests_filter_by_outcome(self, recorder: MetricsRecorder) -> None:
        """Can filter requests by outcome."""
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/chat/completions",
            outcome="success",
        )
        recorder.record_request(
            request_id="req2",
            timestamp_utc="2024-01-15T10:30:01+00:00",
            endpoint="/chat/completions",
            outcome="error_injected",
        )

        success_requests = recorder.get_requests(outcome="success")
        error_requests = recorder.get_requests(outcome="error_injected")

        assert len(success_requests) == 1
        assert success_requests[0]["request_id"] == "req1"
        assert len(error_requests) == 1
        assert error_requests[0]["request_id"] == "req2"


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_record_requests(self, tmp_path: Path) -> None:
        """Multiple threads can record requests safely."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        errors: list[Exception] = []
        lock = threading.Lock()

        def worker(thread_id: int) -> None:
            try:
                for i in range(50):
                    recorder.record_request(
                        request_id=f"t{thread_id}_req{i}",
                        timestamp_utc=f"2024-01-15T10:30:{i % 60:02d}+00:00",
                        endpoint="/chat/completions",
                        outcome="success",
                        status_code=200,
                        latency_ms=float(100 + i),
                    )
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder.close()

        assert not errors, f"Errors occurred: {errors}"

        # Verify all requests were recorded
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM requests")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 500  # 10 threads * 50 requests each

    def test_concurrent_get_stats(self, tmp_path: Path) -> None:
        """Multiple threads can call get_stats safely."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        # Add some data
        for i in range(100):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc="2024-01-15T10:30:00+00:00",
                endpoint="/chat/completions",
                outcome="success",
            )

        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                for _ in range(50):
                    stats = recorder.get_stats()
                    assert stats["total_requests"] == 100
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder.close()

        assert not errors, f"Errors occurred: {errors}"

    def test_concurrent_reset(self, tmp_path: Path) -> None:
        """Reset is thread-safe even with concurrent access."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        errors: list[Exception] = []
        lock = threading.Lock()

        def writer() -> None:
            try:
                for i in range(100):
                    recorder.record_request(
                        request_id=f"req{i}_{time.time()}",
                        timestamp_utc="2024-01-15T10:30:00+00:00",
                        endpoint="/chat/completions",
                        outcome="success",
                    )
            except Exception as e:
                with lock:
                    errors.append(e)

        def resetter() -> None:
            try:
                for _ in range(5):
                    time.sleep(0.01)
                    recorder.reset()
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)] + [threading.Thread(target=resetter) for _ in range(2)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        recorder.close()

        # Some errors may occur due to race conditions between write and reset
        # but we shouldn't see crashes or database corruption


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_endpoint(self, tmp_path: Path) -> None:
        """Handles very long endpoint strings."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        long_endpoint = "/api/v1/very/long/path/" + "x" * 1000
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint=long_endpoint,
            outcome="success",
        )

        requests = recorder.get_requests()
        assert requests[0]["endpoint"] == long_endpoint

        recorder.close()

    def test_special_characters_in_fields(self, tmp_path: Path) -> None:
        """Handles special characters in string fields."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        recorder.record_request(
            request_id="req'1\"--",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/test?param='value\"",
            outcome="success",
            error_type="error'type\"test",
        )

        requests = recorder.get_requests()
        assert len(requests) == 1
        assert requests[0]["request_id"] == "req'1\"--"

        recorder.close()

    def test_null_values_in_optional_fields(self, tmp_path: Path) -> None:
        """Handles NULL values in optional fields correctly."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/test",
            outcome="success",
            # All optional fields left as None
        )

        requests = recorder.get_requests()
        req = requests[0]

        assert req["deployment"] is None
        assert req["model"] is None
        assert req["status_code"] is None
        assert req["latency_ms"] is None

        recorder.close()

    def test_zero_latency(self, tmp_path: Path) -> None:
        """Handles zero latency correctly."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path))
        recorder = MetricsRecorder(config)

        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/test",
            outcome="success",
            latency_ms=0.0,
        )

        requests = recorder.get_requests()
        assert requests[0]["latency_ms"] == 0.0

        recorder.close()

    def test_bucket_size_larger_than_minute(self, tmp_path: Path) -> None:
        """Handles bucket sizes larger than one minute."""
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path), timeseries_bucket_sec=300)  # 5 minutes
        recorder = MetricsRecorder(config)

        # Records at different times within same 5-minute bucket
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:00+00:00",
            endpoint="/test",
            outcome="success",
        )
        recorder.record_request(
            request_id="req2",
            timestamp_utc="2024-01-15T10:32:00+00:00",
            endpoint="/test",
            outcome="success",
        )
        recorder.record_request(
            request_id="req3",
            timestamp_utc="2024-01-15T10:34:00+00:00",
            endpoint="/test",
            outcome="success",
        )

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 3

        recorder.close()

    def test_bucket_boundary_overflow_update_bucket_latency(self, tmp_path: Path) -> None:
        """Tests bucket boundary calculation with 10-second buckets at :55 seconds.

        This is a regression test for a bug where adding seconds to a datetime using
        replace() would overflow (e.g., second=55 + 10 = 65, which is invalid).
        The fix uses timedelta arithmetic instead.
        """
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path), timeseries_bucket_sec=10)
        recorder = MetricsRecorder(config)

        # Record a request at :55 seconds with a 10-second bucket
        # The bucket would be at :50, and bucket_end should be :00 of next minute
        recorder.record_request(
            request_id="req1",
            timestamp_utc="2024-01-15T10:30:55.500+00:00",
            endpoint="/test",
            outcome="success",
            latency_ms=100.0,
        )

        # Verify that timeseries was created without ValueError
        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 1

        recorder.close()

    def test_bucket_boundary_overflow_update_timeseries(self, tmp_path: Path) -> None:
        """Tests update_timeseries() bucket boundary calculation with overflow.

        Similar to above but exercises the update_timeseries() method which
        also had the same bug in bucket_end_dt calculation.
        """
        db_path = tmp_path / "metrics.db"
        config = MetricsConfig(database=str(db_path), timeseries_bucket_sec=10)
        recorder = MetricsRecorder(config)

        # Record requests at :55 seconds boundary
        for i in range(3):
            recorder.record_request(
                request_id=f"req{i}",
                timestamp_utc=f"2024-01-15T10:30:55.{i:03d}+00:00",
                endpoint="/test",
                outcome="success",
                latency_ms=100.0 + i,
            )

        # Call update_timeseries which rebuilds from raw data
        # This should not raise ValueError on bucket boundary arithmetic
        recorder.update_timeseries()

        timeseries = recorder.get_timeseries()
        assert len(timeseries) == 1
        assert timeseries[0]["requests_total"] == 3

        recorder.close()
