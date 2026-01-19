# tests/plugins/clients/test_replayer.py
"""Tests for CallReplayer."""

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import Call, CallStatus, CallType
from elspeth.core.canonical import stable_hash
from elspeth.plugins.clients.replayer import (
    CallReplayer,
    ReplayedCall,
    ReplayMissError,
)


class TestReplayedCall:
    """Tests for ReplayedCall dataclass."""

    def test_creation_with_required_fields(self) -> None:
        """ReplayedCall can be created with required fields."""
        result = ReplayedCall(
            response_data={"content": "Hello"},
            original_latency_ms=150.0,
            request_hash="abc123",
        )

        assert result.response_data == {"content": "Hello"}
        assert result.original_latency_ms == 150.0
        assert result.request_hash == "abc123"
        assert result.was_error is False
        assert result.error_data is None

    def test_creation_with_error_data(self) -> None:
        """ReplayedCall can hold error data."""
        result = ReplayedCall(
            response_data={},
            original_latency_ms=50.0,
            request_hash="abc123",
            was_error=True,
            error_data={"type": "RateLimitError", "message": "Too many requests"},
        )

        assert result.was_error is True
        assert result.error_data == {
            "type": "RateLimitError",
            "message": "Too many requests",
        }


class TestReplayMissError:
    """Tests for ReplayMissError exception."""

    def test_error_has_request_hash(self) -> None:
        """ReplayMissError includes request hash."""
        error = ReplayMissError("hash123", {"model": "gpt-4"})

        assert error.request_hash == "hash123"
        assert "hash123" in str(error)

    def test_error_has_request_data(self) -> None:
        """ReplayMissError includes request data for debugging."""
        request = {"model": "gpt-4", "messages": []}
        error = ReplayMissError("hash123", request)

        assert error.request_data == request


class TestCallReplayer:
    """Tests for CallReplayer."""

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
        error_json: str | None = None,
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
            error_json=error_json,
        )

    def test_replay_returns_recorded_response(self) -> None:
        """Basic replay returns recorded response data."""
        recorder = self._create_mock_recorder()
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash, latency_ms=150.0)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {
            "content": "Hello, world!",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        assert result.response_data == {
            "content": "Hello, world!",
            "model": "gpt-4",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        assert result.original_latency_ms == 150.0
        assert result.request_hash == request_hash
        assert result.was_error is False

        # Verify correct lookup parameters
        recorder.find_call_by_request_hash.assert_called_once_with(
            run_id="run_abc123",
            call_type="llm",
            request_hash=request_hash,
        )

    def test_replay_miss_raises_error(self) -> None:
        """Missing calls raise ReplayMissError."""
        recorder = self._create_mock_recorder()
        recorder.find_call_by_request_hash.return_value = None

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        request_data = {"model": "gpt-4", "messages": []}

        with pytest.raises(ReplayMissError) as exc_info:
            replayer.replay(call_type="llm", request_data=request_data)

        assert exc_info.value.request_data == request_data
        assert exc_info.value.request_hash == stable_hash(request_data)

    def test_replay_caches_results(self) -> None:
        """Second lookup uses cache, not database."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "cached"}

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # First call - hits database
        result1 = replayer.replay(call_type="llm", request_data=request_data)

        # Second call - should use cache
        result2 = replayer.replay(call_type="llm", request_data=request_data)

        # Only one database lookup should have occurred
        assert recorder.find_call_by_request_hash.call_count == 1
        assert recorder.get_call_response_data.call_count == 1

        # Both results should be the same
        assert result1.response_data == result2.response_data
        assert replayer.cache_size() == 1

    def test_replay_handles_error_calls(self) -> None:
        """Replays error status correctly."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        error_details = {
            "type": "RateLimitError",
            "message": "Rate limit exceeded",
            "retryable": True,
        }
        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.ERROR,
            latency_ms=50.0,
            error_json=json.dumps(error_details),
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        assert result.was_error is True
        assert result.error_data == error_details
        assert result.response_data == {}  # Empty dict, not None

    def test_replayed_call_includes_latency(self) -> None:
        """Original latency is preserved in replay result."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        # Original call took 250ms
        mock_call = self._create_mock_call(request_hash=request_hash, latency_ms=250.0)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "test"}

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        assert result.original_latency_ms == 250.0

    def test_source_run_id_property(self) -> None:
        """source_run_id property returns the run being replayed."""
        recorder = self._create_mock_recorder()
        replayer = CallReplayer(recorder, source_run_id="run_xyz789")

        assert replayer.source_run_id == "run_xyz789"

    def test_clear_cache(self) -> None:
        """clear_cache empties the cache."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "test"}

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # Populate cache
        replayer.replay(call_type="llm", request_data=request_data)
        assert replayer.cache_size() == 1

        # Clear cache
        replayer.clear_cache()
        assert replayer.cache_size() == 0

        # Next call should hit database again
        replayer.replay(call_type="llm", request_data=request_data)
        assert recorder.find_call_by_request_hash.call_count == 2

    def test_replay_with_none_response_data(self) -> None:
        """Handles calls where response data couldn't be retrieved."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None  # Payload purged

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        # Should return empty dict, not None
        assert result.response_data == {}

    def test_replay_http_call_type(self) -> None:
        """Replay works for HTTP call types."""
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

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="http", request_data=request_data)

        assert result.response_data == {"status": 200, "body": "OK"}
        recorder.find_call_by_request_hash.assert_called_once_with(
            run_id="run_abc123",
            call_type="http",
            request_hash=request_hash,
        )
