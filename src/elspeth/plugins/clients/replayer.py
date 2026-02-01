# src/elspeth/plugins/clients/replayer.py
"""Call replayer for replay mode - returns recorded responses.

In replay mode, instead of making live external calls, the CallReplayer
returns previously recorded responses from the audit trail. This enables:

- Deterministic re-execution of pipelines with external dependencies
- Testing and debugging without live API calls
- Verifying that code changes produce identical results

The replayer matches calls by request_hash (canonical hash of request data),
so the same request always returns the same recorded response.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from elspeth.contracts import CallStatus
from elspeth.core.canonical import stable_hash

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


@dataclass
class ReplayedCall:
    """A replayed call result.

    Contains the response data from a previously recorded call,
    along with metadata about the original call.

    Attributes:
        response_data: The recorded response payload
        original_latency_ms: How long the original call took (for simulation)
        request_hash: Hash of the request data (for debugging)
        was_error: Whether the original call was an error
        error_data: Error details if was_error is True
    """

    response_data: dict[str, Any]
    original_latency_ms: float | None
    request_hash: str
    was_error: bool = False
    error_data: dict[str, Any] | None = None


class ReplayMissError(Exception):
    """Raised when no matching recorded call is found.

    This indicates the current pipeline is attempting to make a call
    that wasn't recorded in the source run - the pipeline may have
    diverged from the original execution.

    Attributes:
        request_hash: Hash of the request that wasn't found
        request_data: The actual request data (for debugging)
    """

    def __init__(self, request_hash: str, request_data: dict[str, Any]) -> None:
        self.request_hash = request_hash
        self.request_data = request_data
        super().__init__(f"No recorded call found for request hash: {request_hash}")


class ReplayPayloadMissingError(Exception):
    """Raised when recorded call exists but payload is missing.

    This indicates the call was recorded but the response payload
    has been purged from the payload store or the store is disabled.
    In replay mode, we cannot proceed without the actual response data.

    Attributes:
        call_id: ID of the call whose payload is missing
        request_hash: Hash of the request (for debugging)
    """

    def __init__(self, call_id: str, request_hash: str) -> None:
        self.call_id = call_id
        self.request_hash = request_hash
        super().__init__(
            f"Recorded call {call_id} exists but response payload is missing "
            f"(request_hash={request_hash}). Payload may have been purged or "
            "payload store may be disabled."
        )


class CallReplayer:
    """Replays recorded external calls instead of making live calls.

    Used in replay mode to return previously recorded responses.
    Matches calls by request_hash (canonical hash of request data).

    Example:
        replayer = CallReplayer(recorder, source_run_id="run-abc123")

        # Instead of making live call:
        result = replayer.replay(
            call_type="llm",
            request_data={"model": "gpt-4", "messages": [...]}
        )
        # Returns the response that was recorded for this exact request

    Thread Safety:
        The replayer caches results in memory. If used across threads,
        external synchronization may be needed for the cache.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        source_run_id: str,
    ) -> None:
        """Initialize replayer.

        Args:
            recorder: LandscapeRecorder for looking up recorded calls
            source_run_id: The run_id to replay calls from
        """
        self._recorder = recorder
        self._source_run_id = source_run_id
        # Cache: (call_type, request_hash, sequence_index) -> cached data
        # The sequence_index allows multiple calls with same hash to be cached separately
        self._cache: dict[
            tuple[str, str, int],
            tuple[dict[str, Any], float | None, bool, dict[str, Any] | None, str],
        ] = {}
        # Sequence counter: (call_type, request_hash) -> next_index
        # Tracks how many times we've seen each unique request
        # Uses defaultdict to avoid .get() which can hide key bugs
        self._sequence_counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    @property
    def source_run_id(self) -> str:
        """The run ID being replayed from."""
        return self._source_run_id

    def replay(
        self,
        call_type: str,
        request_data: dict[str, Any],
    ) -> ReplayedCall:
        """Replay a recorded call.

        Looks up a previously recorded call by computing the canonical
        hash of the request data and searching the source run's audit trail.

        When the same request is replayed multiple times (same call_type and
        request_data), each replay returns the next recorded response in
        chronological order. This supports scenarios where the original run
        made the same request multiple times (e.g., retries, loops over
        identical data, non-deterministic LLM responses).

        Args:
            call_type: Type of call (llm, http, etc.)
            request_data: The request data (used to compute hash for lookup)

        Returns:
            ReplayedCall with recorded response data

        Raises:
            ReplayMissError: If no matching recorded call is found
            ReplayPayloadMissingError: If call exists but payload is missing/purged
        """
        request_hash = stable_hash(request_data)
        sequence_key = (call_type, request_hash)

        # Get the current sequence index for this request and increment it
        # Using defaultdict(int) ensures missing keys default to 0
        sequence_index = self._sequence_counters[sequence_key]
        self._sequence_counters[sequence_key] = sequence_index + 1

        # Cache key includes sequence index to store multiple responses separately
        cache_key = (call_type, request_hash, sequence_index)

        # Check cache first
        if cache_key in self._cache:
            resp, latency, was_error, error, _call_id = self._cache[cache_key]
            return ReplayedCall(
                response_data=resp,
                original_latency_ms=latency,
                request_hash=request_hash,
                was_error=was_error,
                error_data=error,
            )

        # Look up in database with sequence index to get Nth occurrence
        call = self._recorder.find_call_by_request_hash(
            run_id=self._source_run_id,
            call_type=call_type,
            request_hash=request_hash,
            sequence_index=sequence_index,
        )

        if call is None:
            raise ReplayMissError(request_hash, request_data)

        # Get response data from payload store
        response_data = self._recorder.get_call_response_data(call.call_id)

        # Parse error JSON if present
        error_data: dict[str, Any] | None = None
        if call.error_json:
            error_data = json.loads(call.error_json)

        # Determine if this was an error call
        was_error = call.status == CallStatus.ERROR

        # Fail if a response was expected but payload is unavailable.
        # A response is expected if response_hash is set OR status is SUCCESS.
        # This catches both purged payloads AND runs recorded without a payload store.
        response_expected = call.response_hash is not None or call.status == CallStatus.SUCCESS
        if response_data is None and response_expected:
            raise ReplayPayloadMissingError(call.call_id, request_hash)

        # For calls that never had a response (response_ref is None), use empty dict
        if response_data is None:
            response_data = {}

        # Cache for future lookups (includes call_id for debugging)
        self._cache[cache_key] = (
            response_data,
            call.latency_ms,
            was_error,
            error_data,
            call.call_id,
        )

        return ReplayedCall(
            response_data=response_data,
            original_latency_ms=call.latency_ms,
            request_hash=request_hash,
            was_error=was_error,
            error_data=error_data,
        )

    def clear_cache(self) -> None:
        """Clear the replay cache and reset sequence counters.

        Use this if you need to force re-lookup from the database,
        for example after the source run has been modified.

        Note: This also resets sequence counters, so the next replay
        of any request will start from the first recorded occurrence.
        """
        self._cache.clear()
        self._sequence_counters = defaultdict(int)

    def cache_size(self) -> int:
        """Return the number of cached replayed calls."""
        return len(self._cache)
