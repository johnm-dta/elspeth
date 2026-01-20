# tests/plugins/clients/test_verifier.py
"""Tests for CallVerifier."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from elspeth.contracts import Call, CallStatus, CallType
from elspeth.core.canonical import stable_hash
from elspeth.plugins.clients.verifier import (
    CallVerifier,
    VerificationReport,
    VerificationResult,
)


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_creation_with_match(self) -> None:
        """VerificationResult can be created for matching responses."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response={"content": "Hello"},
            is_match=True,
        )

        assert result.request_hash == "abc123"
        assert result.is_match is True
        assert result.differences == {}
        assert result.recorded_call_missing is False

    def test_creation_with_differences(self) -> None:
        """VerificationResult can hold difference details."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "New response"},
            recorded_response={"content": "Old response"},
            is_match=False,
            differences={
                "values_changed": {
                    "root['content']": {
                        "new_value": "New response",
                        "old_value": "Old response",
                    }
                }
            },
        )

        assert result.is_match is False
        assert "values_changed" in result.differences

    def test_creation_with_missing_recording(self) -> None:
        """VerificationResult handles missing recorded calls."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response=None,
            is_match=False,
            recorded_call_missing=True,
        )

        assert result.recorded_response is None
        assert result.recorded_call_missing is True

    def test_has_differences_true_when_mismatch(self) -> None:
        """has_differences is True when there are actual differences."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "New"},
            recorded_response={"content": "Old"},
            is_match=False,
            differences={"values_changed": {}},
        )

        assert result.has_differences is True

    def test_has_differences_false_when_match(self) -> None:
        """has_differences is False when responses match."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Same"},
            recorded_response={"content": "Same"},
            is_match=True,
        )

        assert result.has_differences is False

    def test_has_differences_false_when_recording_missing(self) -> None:
        """has_differences is False when recording is missing (not a real diff)."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response=None,
            is_match=False,
            recorded_call_missing=True,
        )

        # Missing recording is not considered a "difference"
        assert result.has_differences is False


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_default_values(self) -> None:
        """VerificationReport has sensible defaults."""
        report = VerificationReport()

        assert report.total_calls == 0
        assert report.matches == 0
        assert report.mismatches == 0
        assert report.missing_recordings == 0
        assert report.results == []

    def test_success_rate_no_calls(self) -> None:
        """success_rate returns 100% when no calls verified."""
        report = VerificationReport()
        assert report.success_rate == 100.0

    def test_success_rate_all_match(self) -> None:
        """success_rate is 100% when all calls match."""
        report = VerificationReport(
            total_calls=10,
            matches=10,
            mismatches=0,
            missing_recordings=0,
        )
        assert report.success_rate == 100.0

    def test_success_rate_partial_match(self) -> None:
        """success_rate calculates correct percentage."""
        report = VerificationReport(
            total_calls=10,
            matches=7,
            mismatches=2,
            missing_recordings=1,
        )
        assert report.success_rate == 70.0

    def test_success_rate_no_matches(self) -> None:
        """success_rate is 0% when nothing matches."""
        report = VerificationReport(
            total_calls=5,
            matches=0,
            mismatches=5,
        )
        assert report.success_rate == 0.0


class TestCallVerifier:
    """Tests for CallVerifier."""

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock LandscapeRecorder."""
        recorder = MagicMock()
        recorder.find_call_by_request_hash = MagicMock(return_value=None)
        recorder.get_call_response_data = MagicMock(return_value=None)
        return recorder

    def _create_mock_call(
        self,
        *,
        call_id: str = "call_123",
        status: CallStatus = CallStatus.SUCCESS,
        request_hash: str = "hash123",
        latency_ms: float = 150.0,
    ) -> Call:
        """Create a mock Call object."""
        return Call(
            call_id=call_id,
            state_id="state_123",
            call_index=0,
            call_type=CallType.LLM,
            status=status,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=latency_ms,
        )

    def test_verify_matching_response(self) -> None:
        """Verifier detects matching responses."""
        recorder = self._create_mock_recorder()
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request_hash = stable_hash(request_data)

        recorded_response = {
            "content": "Hello, world!",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Live response matches recorded
        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=recorded_response.copy(),
        )

        assert result.is_match is True
        assert result.differences == {}
        assert result.has_differences is False
        assert result.recorded_response == recorded_response

    def test_verify_different_response(self) -> None:
        """Verifier detects differences between responses."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        recorded_response = {"content": "Original response", "model": "gpt-4"}
        live_response = {"content": "Different response", "model": "gpt-4"}

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )

        assert result.is_match is False
        assert result.has_differences is True
        assert "values_changed" in result.differences
        assert result.recorded_response == recorded_response
        assert result.live_response == live_response

    def test_verify_missing_recording(self) -> None:
        """Verifier handles missing baseline gracefully."""
        recorder = self._create_mock_recorder()
        recorder.find_call_by_request_hash.return_value = None

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        request_data = {"model": "gpt-4", "messages": []}

        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response={"content": "Hello"},
        )

        assert result.is_match is False
        assert result.recorded_call_missing is True
        assert result.recorded_response is None
        assert result.has_differences is False  # Missing is not a "difference"

    def test_verify_with_ignore_paths(self) -> None:
        """Verifier excludes specified paths from comparison."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        # Responses differ only in latency
        recorded_response = {"content": "Hello", "latency": 100}
        live_response = {"content": "Hello", "latency": 200}

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        # Ignore latency in comparison
        verifier = CallVerifier(
            recorder,
            source_run_id="run_abc123",
            ignore_paths=["root['latency']"],
        )

        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )

        # Should match because latency is ignored
        assert result.is_match is True
        assert result.differences == {}

    def test_verification_report_tracks_stats(self) -> None:
        """Report accumulates verification statistics."""
        recorder = self._create_mock_recorder()

        # Set up for multiple calls
        request1 = {"id": 1}
        request2 = {"id": 2}
        request3 = {"id": 3}

        def find_call_side_effect(
            run_id: str, call_type: str, request_hash: str
        ) -> Call | None:
            # First two requests have recordings, third doesn't
            if request_hash == stable_hash(request3):
                return None
            return self._create_mock_call(request_hash=request_hash)

        def get_response_side_effect(call_id: str) -> dict[str, object]:
            return {"content": "recorded"}

        recorder.find_call_by_request_hash.side_effect = find_call_side_effect
        recorder.get_call_response_data.side_effect = get_response_side_effect

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # First call: match
        verifier.verify(
            call_type="llm",
            request_data=request1,
            live_response={"content": "recorded"},
        )

        # Second call: mismatch
        verifier.verify(
            call_type="llm",
            request_data=request2,
            live_response={"content": "different"},
        )

        # Third call: missing recording
        verifier.verify(
            call_type="llm",
            request_data=request3,
            live_response={"content": "new"},
        )

        report = verifier.get_report()
        assert report.total_calls == 3
        assert report.matches == 1
        assert report.mismatches == 1
        assert report.missing_recordings == 1
        assert len(report.results) == 3

    def test_success_rate_calculation(self) -> None:
        """Verification report calculates correct success rate."""
        recorder = self._create_mock_recorder()

        # All calls match
        mock_call = self._create_mock_call()
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "match"}

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Make 4 matching calls
        for i in range(4):
            verifier.verify(
                call_type="llm",
                request_data={"id": i},
                live_response={"content": "match"},
            )

        report = verifier.get_report()
        assert report.success_rate == 100.0

        # Now add a mismatch
        recorder.get_call_response_data.return_value = {"content": "different"}
        verifier.verify(
            call_type="llm",
            request_data={"id": 5},
            live_response={"content": "live"},
        )

        report = verifier.get_report()
        assert report.success_rate == 80.0  # 4 out of 5

    def test_reset_report(self) -> None:
        """reset_report clears accumulated statistics."""
        recorder = self._create_mock_recorder()
        mock_call = self._create_mock_call()
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "match"}

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Make a verification
        verifier.verify(
            call_type="llm",
            request_data={"id": 1},
            live_response={"content": "match"},
        )

        report = verifier.get_report()
        assert report.total_calls == 1
        assert report.matches == 1

        # Reset
        verifier.reset_report()

        report = verifier.get_report()
        assert report.total_calls == 0
        assert report.matches == 0
        assert report.results == []

    def test_source_run_id_property(self) -> None:
        """source_run_id property returns the baseline run."""
        recorder = self._create_mock_recorder()
        verifier = CallVerifier(recorder, source_run_id="run_xyz789")

        assert verifier.source_run_id == "run_xyz789"

    def test_verify_http_call_type(self) -> None:
        """Verification works for HTTP call types."""
        recorder = self._create_mock_recorder()
        request_data = {
            "method": "POST",
            "url": "https://api.example.com/data",
            "body": {"key": "value"},
        }
        request_hash = stable_hash(request_data)

        mock_call = Call(
            call_id="call_456",
            state_id="state_123",
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=100.0,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"status": 200, "body": "OK"}

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type="http",
            request_data=request_data,
            live_response={"status": 200, "body": "OK"},
        )

        assert result.is_match is True
        recorder.find_call_by_request_hash.assert_called_once_with(
            run_id="run_abc123",
            call_type="http",
            request_hash=request_hash,
        )

    def test_verify_with_none_recorded_response(self) -> None:
        """Handles calls where recorded response couldn't be retrieved."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None  # Payload purged

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response={"content": "Hello"},
        )

        # Empty dict vs live response will mismatch
        assert result.is_match is False
        assert result.recorded_response == {}

    def test_verify_order_independent_comparison(self) -> None:
        """Verifier ignores order in list comparisons."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {"items": ["a", "b", "c"]}
        live_response = {"items": ["c", "a", "b"]}  # Same items, different order

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )

        # Should match because ignore_order=True
        assert result.is_match is True

    def test_verify_nested_differences(self) -> None:
        """Verifier detects nested differences."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {
            "data": {
                "user": {
                    "name": "Alice",
                    "email": "alice@example.com",
                }
            }
        }
        live_response = {
            "data": {
                "user": {
                    "name": "Alice",
                    "email": "bob@example.com",  # Different email
                }
            }
        }

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )

        assert result.is_match is False
        assert "values_changed" in result.differences

    def test_multiple_ignore_paths(self) -> None:
        """Verifier can ignore multiple paths."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {
            "content": "Hello",
            "timestamp": "2024-01-01T00:00:00Z",
            "request_id": "old-123",
        }
        live_response = {
            "content": "Hello",
            "timestamp": "2024-06-15T12:00:00Z",  # Different
            "request_id": "new-456",  # Different
        }

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = recorded_response

        # Ignore both timestamp and request_id
        verifier = CallVerifier(
            recorder,
            source_run_id="run_abc123",
            ignore_paths=["root['timestamp']", "root['request_id']"],
        )

        result = verifier.verify(
            call_type="llm",
            request_data=request_data,
            live_response=live_response,
        )

        # Should match because differences are in ignored paths
        assert result.is_match is True
