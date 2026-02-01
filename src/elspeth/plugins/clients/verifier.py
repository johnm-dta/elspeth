# src/elspeth/plugins/clients/verifier.py
"""Call verifier for verify mode - compares live vs recorded responses.

In verify mode, instead of simply replaying recorded responses, the CallVerifier
makes live external calls and compares them against previously recorded responses
from the audit trail. This enables:

- Detecting drift between current API behavior and recorded baselines
- Validating that external services still return expected results
- Catching breaking changes in API responses before they affect production

The verifier uses DeepDiff for flexible comparison with configurable exclusions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deepdiff import DeepDiff

from elspeth.contracts import CallStatus
from elspeth.core.canonical import stable_hash

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


@dataclass
class VerificationResult:
    """Result of verifying a call against recorded response.

    Attributes:
        request_hash: Hash of the request data (for identification)
        live_response: The response from the live call
        recorded_response: The previously recorded response (None if missing or purged)
        is_match: Whether live and recorded responses match
        differences: DeepDiff results as dict (empty if match)
        recorded_call_missing: True if no recorded call was found
        payload_missing: True if call exists but response payload is missing/purged
    """

    request_hash: str
    live_response: dict[str, Any]
    recorded_response: dict[str, Any] | None
    is_match: bool
    differences: dict[str, Any] = field(default_factory=dict)
    recorded_call_missing: bool = False
    payload_missing: bool = False

    @property
    def has_differences(self) -> bool:
        """Check if there are meaningful differences.

        Returns True only when there are actual differences between
        responses (not when the recording is simply missing or payload is purged).
        """
        return not self.is_match and not self.recorded_call_missing and not self.payload_missing


@dataclass
class VerificationReport:
    """Summary report of all verifications in a run.

    Tracks statistics across all verified calls for reporting
    and alerting on drift.

    Attributes:
        total_calls: Number of calls verified
        matches: Number of calls that matched recorded baseline
        mismatches: Number of calls with differences from baseline
        missing_recordings: Number of calls with no recorded baseline
        missing_payloads: Number of calls where response payload is missing/purged
        results: Individual verification results for inspection
    """

    total_calls: int = 0
    matches: int = 0
    mismatches: int = 0
    missing_recordings: int = 0
    missing_payloads: int = 0
    results: list[VerificationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of calls that matched recorded baseline.

        Returns 100.0 if no calls have been verified.
        """
        if self.total_calls == 0:
            return 100.0
        return (self.matches / self.total_calls) * 100


class CallVerifier:
    """Verifies live API calls against recorded responses.

    Used in verify mode to detect drift between current behavior
    and recorded baseline. Uses DeepDiff for comparison.

    Example:
        verifier = CallVerifier(recorder, source_run_id="run-abc123")

        # After making a live call:
        result = verifier.verify(
            call_type="llm",
            request_data={"model": "gpt-4", "messages": [...]},
            live_response={"content": "Hello!", ...}
        )

        if result.has_differences:
            print(f"Response drift detected: {result.differences}")

    Thread Safety:
        The verifier maintains a report of all verifications.
        If used across threads, external synchronization may be
        needed for the report.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        source_run_id: str,
        *,
        ignore_paths: list[str] | None = None,
        ignore_order: bool = True,
    ) -> None:
        """Initialize verifier.

        Args:
            recorder: LandscapeRecorder for looking up recorded calls
            source_run_id: The run_id containing baseline recordings
            ignore_paths: Paths to ignore in comparison (e.g., ["root['latency']"])
                         These paths will be excluded from DeepDiff comparison.
            ignore_order: If True (default), list ordering differences are ignored.
                         If False, list elements must appear in the same order to match.
                         Set to False for order-sensitive data like ranked results.
        """
        self._recorder = recorder
        self._source_run_id = source_run_id
        self._ignore_paths = ignore_paths or []
        self._ignore_order = ignore_order
        self._report = VerificationReport()
        # Sequence counter: (call_type, request_hash) -> next_index
        # Tracks how many times we've seen each unique request
        # Uses defaultdict to avoid .get() which can hide key bugs
        self._sequence_counters: defaultdict[tuple[str, str], int] = defaultdict(int)

    @property
    def source_run_id(self) -> str:
        """The run ID containing baseline recordings."""
        return self._source_run_id

    def verify(
        self,
        call_type: str,
        request_data: dict[str, Any],
        live_response: dict[str, Any],
    ) -> VerificationResult:
        """Verify a live response against recorded baseline.

        Looks up the previously recorded call by computing the canonical
        hash of the request data and compares responses using DeepDiff.

        When the same request is verified multiple times (same call_type and
        request_data), each verification compares against the next recorded
        response in chronological order. This supports scenarios where the
        original run made the same request multiple times.

        Args:
            call_type: Type of call (llm, http, etc.)
            request_data: The request data (used to find recorded call)
            live_response: The response from the live call

        Returns:
            VerificationResult with comparison details
        """
        request_hash = stable_hash(request_data)
        sequence_key = (call_type, request_hash)

        # Get the current sequence index for this request and increment it
        # Using defaultdict(int) ensures missing keys default to 0
        sequence_index = self._sequence_counters[sequence_key]
        self._sequence_counters[sequence_key] = sequence_index + 1

        # Look up recorded call with sequence index to get Nth occurrence
        call = self._recorder.find_call_by_request_hash(
            run_id=self._source_run_id,
            call_type=call_type,
            request_hash=request_hash,
            sequence_index=sequence_index,
        )

        self._report.total_calls += 1

        if call is None:
            result = VerificationResult(
                request_hash=request_hash,
                live_response=live_response,
                recorded_response=None,
                is_match=False,
                recorded_call_missing=True,
            )
            self._report.missing_recordings += 1
            self._report.results.append(result)
            return result

        # Get recorded response
        recorded_response = self._recorder.get_call_response_data(call.call_id)

        # Handle missing/purged payload explicitly.
        # A response is expected if response_hash is set OR status is SUCCESS.
        # This catches both purged payloads AND runs recorded without a payload store.
        response_expected = call.response_hash is not None or call.status == CallStatus.SUCCESS
        if recorded_response is None and response_expected:
            result = VerificationResult(
                request_hash=request_hash,
                live_response=live_response,
                recorded_response=None,
                is_match=False,
                payload_missing=True,
            )
            self._report.missing_payloads += 1
            self._report.results.append(result)
            return result

        # Call never had a response (e.g., connection timeout, DNS failure)
        # Cannot compare, so not a match, but NOT a missing payload
        if recorded_response is None:
            result = VerificationResult(
                request_hash=request_hash,
                live_response=live_response,
                recorded_response=None,
                is_match=False,
            )
            self._report.results.append(result)
            return result

        # Compare using DeepDiff
        diff = DeepDiff(
            recorded_response,
            live_response,
            ignore_order=self._ignore_order,
            exclude_paths=self._ignore_paths,
        )

        is_match = len(diff) == 0

        if is_match:
            self._report.matches += 1
        else:
            self._report.mismatches += 1

        result = VerificationResult(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=recorded_response,
            is_match=is_match,
            differences=diff.to_dict() if diff else {},
        )
        self._report.results.append(result)
        return result

    def get_report(self) -> VerificationReport:
        """Get the verification report for this session.

        Returns the accumulated report of all verifications performed.
        """
        return self._report

    def reset_report(self) -> None:
        """Reset the verification report and sequence counters.

        Clears all accumulated statistics, results, and sequence counters.
        Use this when starting a new verification session.

        Note: This also resets sequence counters, so the next verification
        of any request will compare against the first recorded occurrence.
        """
        self._report = VerificationReport()
        self._sequence_counters = defaultdict(int)
