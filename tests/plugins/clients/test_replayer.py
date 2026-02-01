# tests/plugins/clients/test_replayer.py
"""Tests for CallReplayer."""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import Call, CallStatus, CallType
from elspeth.core.canonical import stable_hash
from elspeth.plugins.clients.replayer import (
    CallReplayer,
    ReplayedCall,
    ReplayMissError,
    ReplayPayloadMissingError,
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
        response_ref: str | None = None,
        response_hash: str | None = None,
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
            response_ref=response_ref,
            response_hash=response_hash,
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

        # Verify correct lookup parameters (now includes sequence_index)
        recorder.find_call_by_request_hash.assert_called_once_with(
            run_id="run_abc123",
            call_type="llm",
            request_hash=request_hash,
            sequence_index=0,
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

    def test_replay_caches_results_per_sequence_index(self) -> None:
        """Cache stores results keyed by (call_type, request_hash, sequence_index).

        With the fix for P1-2026-01-21-replay-request-hash-collisions, each
        replay of the same request increments a sequence counter and looks
        for the Nth recorded call. The cache still works, but it's keyed by
        the full (call_type, request_hash, sequence_index) tuple.

        This test verifies that after clearing the cache and sequence counters,
        replaying the same request again uses the cached first response.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = {"content": "cached"}

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # First call - hits database (sequence_index=0)
        result1 = replayer.replay(call_type="llm", request_data=request_data)
        assert recorder.find_call_by_request_hash.call_count == 1

        # Second call - hits database again (sequence_index=1, different from cache key)
        # This is the correct behavior: same request should return different response
        result2 = replayer.replay(call_type="llm", request_data=request_data)
        assert recorder.find_call_by_request_hash.call_count == 2

        # Both results have same content (mock returns same response for any sequence)
        assert result1.response_data == result2.response_data
        # Cache now has 2 entries (one for each sequence_index)
        assert replayer.cache_size() == 2

        # Clear cache and sequence counters - next replay starts from sequence_index=0
        replayer.clear_cache()
        assert replayer.cache_size() == 0

        # Third call after clear - hits database again (sequence_index=0 again)
        result3 = replayer.replay(call_type="llm", request_data=request_data)
        assert recorder.find_call_by_request_hash.call_count == 3
        assert result3.response_data == {"content": "cached"}

    def test_replay_handles_error_calls_without_response(self) -> None:
        """Error calls that never had a response get empty dict.

        Some error calls legitimately have no response (e.g., connection timeout,
        DNS failure). These have response_ref=None, so we allow empty dict.
        """
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
            response_ref=None,  # Never had a response
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        assert result.was_error is True
        assert result.error_data == error_details
        assert result.response_data == {}  # Empty dict is correct here

    def test_replay_error_call_with_purged_response_raises(self) -> None:
        """Error calls that HAD a response but it was purged must fail.

        Bug: P2-2026-01-31-replayer-missing-payload-fallback

        HTTP error responses (400, 500) often include response bodies that are
        recorded in the audit trail. If the payload is purged, we can't silently
        substitute {} - that would change replay behavior.

        The key distinction is response_ref:
        - response_ref=None → call never had a response (OK to use {})
        - response_ref set but payload missing → response was purged (FAIL)
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        error_details = {
            "type": "BadRequestError",
            "message": "Invalid input format",
        }
        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.ERROR,
            latency_ms=50.0,
            error_json=json.dumps(error_details),
            response_ref="payload_ref_123",  # Response WAS recorded
            response_hash="hash_of_response",  # Hash proves response existed
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None  # But payload is now missing

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # Should raise because response_hash exists but payload is missing
        with pytest.raises(ReplayPayloadMissingError) as exc_info:
            replayer.replay(call_type="llm", request_data=request_data)

        assert exc_info.value.call_id == "call_123"
        assert exc_info.value.request_hash == request_hash

    def test_replay_error_call_with_response_succeeds(self) -> None:
        """Error calls with available response data succeed normally."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        error_details = {
            "type": "BadRequestError",
            "message": "Invalid input format",
        }
        error_response = {
            "error": {"code": 400, "message": "Missing required field 'prompt'"},
        }
        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.ERROR,
            latency_ms=50.0,
            error_json=json.dumps(error_details),
            response_ref="payload_ref_123",  # Response was recorded
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = error_response  # And is available

        replayer = CallReplayer(recorder, source_run_id="run_abc123")
        result = replayer.replay(call_type="llm", request_data=request_data)

        assert result.was_error is True
        assert result.error_data == error_details
        assert result.response_data == error_response  # Actual recorded response

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

    def test_replay_with_purged_response_data(self) -> None:
        """Raises error when payload was recorded but is now missing (purged).

        Missing payload would cause silent incorrect outputs (empty dict),
        so we fail fast. The key is response_ref being set - that indicates
        a response was recorded and should be available.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_abc",  # Response WAS recorded
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = None  # Payload purged

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        with pytest.raises(ReplayPayloadMissingError) as exc_info:
            replayer.replay(call_type="llm", request_data=request_data)

        assert exc_info.value.call_id == "call_123"
        assert exc_info.value.request_hash == request_hash

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
            sequence_index=0,
        )

    def test_different_call_types_same_hash_are_cached_separately(self) -> None:
        """Different call types with same request hash are cached separately."""
        recorder = self._create_mock_recorder()
        # Same request data for both call types
        request_data = {"data": "same"}

        # Set up mock to return different responses based on call_type
        def find_call_side_effect(run_id: str, call_type: str, request_hash: str, *, sequence_index: int = 0) -> Call:
            if call_type == "llm":
                return Call(
                    call_id="call_llm",
                    state_id="state_123",
                    call_index=0,
                    call_type=CallType.LLM,
                    status=CallStatus.SUCCESS,
                    request_hash=request_hash,
                    created_at=datetime.now(UTC),
                    latency_ms=100.0,
                )
            else:
                return Call(
                    call_id="call_http",
                    state_id="state_123",
                    call_index=1,
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_hash=request_hash,
                    created_at=datetime.now(UTC),
                    latency_ms=200.0,
                )

        def get_response_side_effect(call_id: str) -> dict[str, Any]:
            if call_id == "call_llm":
                return {"type": "llm_response"}
            else:
                return {"type": "http_response"}

        recorder.find_call_by_request_hash.side_effect = find_call_side_effect
        recorder.get_call_response_data.side_effect = get_response_side_effect

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # Replay LLM call
        llm_result = replayer.replay(call_type="llm", request_data=request_data)
        assert llm_result.response_data["type"] == "llm_response"

        # Replay HTTP call with same request data
        http_result = replayer.replay(call_type="http", request_data=request_data)
        assert http_result.response_data["type"] == "http_response"

        # Both should be cached separately (different call_type, same sequence_index=0)
        assert replayer.cache_size() == 2

        # Verify both database lookups happened (not returned from cache incorrectly)
        assert recorder.find_call_by_request_hash.call_count == 2

        # Replay LLM again - now this is treated as the 2nd LLM call (sequence_index=1)
        # Since the same request can legitimately return different responses,
        # the replayer now looks for the 2nd recorded call (which doesn't exist in this mock)
        # This matches the fix for P1-2026-01-21-replay-request-hash-collisions
        llm_result2 = replayer.replay(call_type="llm", request_data=request_data)
        # Mock returns the same response regardless of sequence_index, so still llm_response
        assert llm_result2.response_data["type"] == "llm_response"
        # Now makes a 3rd DB lookup because sequence_index is different (0 vs 1)
        assert recorder.find_call_by_request_hash.call_count == 3

    def test_duplicate_requests_return_different_responses_in_order(self) -> None:
        """Duplicate identical requests should return different responses in sequence.

        This tests the scenario where the same request is made multiple times
        (e.g., retries, loops over identical data) and each call returns a
        different response (e.g., non-deterministic LLM output with temperature > 0).

        Bug: P1-2026-01-21-replay-request-hash-collisions
        """
        recorder = self._create_mock_recorder()
        # Same request data made multiple times
        request_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        # Track which call index we're returning
        call_sequence = [
            {
                "call_id": "call_1",
                "response": {"content": "Response 1 - first call"},
            },
            {
                "call_id": "call_2",
                "response": {"content": "Response 2 - second call"},
            },
            {
                "call_id": "call_3",
                "response": {"content": "Response 3 - third call"},
            },
        ]

        def find_call_side_effect(run_id: str, call_type: str, request_hash: str, *, sequence_index: int = 0) -> Call | None:
            """Return the Nth call for duplicate request hashes."""
            idx = sequence_index if sequence_index < len(call_sequence) else len(call_sequence) - 1
            return Call(
                call_id=call_sequence[idx]["call_id"],
                state_id="state_123",
                call_index=idx,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_hash=request_hash,
                created_at=datetime.now(UTC),
                latency_ms=100.0,
            )

        def get_response_side_effect(call_id: str) -> dict[str, Any]:
            """Return response data based on call_id."""
            for call_info in call_sequence:
                if call_info["call_id"] == call_id:
                    return call_info["response"]
            return {}

        recorder.find_call_by_request_hash.side_effect = find_call_side_effect
        recorder.get_call_response_data.side_effect = get_response_side_effect

        replayer = CallReplayer(recorder, source_run_id="run_abc123")

        # First replay - should get Response 1
        result1 = replayer.replay(call_type="llm", request_data=request_data)
        assert result1.response_data["content"] == "Response 1 - first call", (
            f"First replay should return first response, got: {result1.response_data}"
        )

        # Second replay of SAME request - should get Response 2 (not cached Response 1!)
        result2 = replayer.replay(call_type="llm", request_data=request_data)
        assert result2.response_data["content"] == "Response 2 - second call", (
            f"Second replay should return second response, got: {result2.response_data}"
        )

        # Third replay - should get Response 3
        result3 = replayer.replay(call_type="llm", request_data=request_data)
        assert result3.response_data["content"] == "Response 3 - third call", (
            f"Third replay should return third response, got: {result3.response_data}"
        )
