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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deepdiff import DeepDiff

from elspeth.contracts import CallType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash

if TYPE_CHECKING:
    from elspeth.core.landscape.execution_repository import ExecutionRepository

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.core.landscape.row_data import CallDataState


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying a call against recorded response.

    Attributes:
        request_hash: Hash of the request data (for identification)
        live_response: The response from the live call
        recorded_response: The previously recorded response (None if missing or purged)
        is_match: Whether live and recorded responses match. True = match,
            False = definitive mismatch, None = indeterminate (cannot verify).
        differences: DeepDiff results as dict (empty if match)
        recorded_call_missing: True if no recorded call was found
        payload_missing: True if call exists but response payload is missing/purged
        no_response_recorded: True if call exists but never had a response (timeout/DNS failure)

    Use factory classmethods (``matched``, ``mismatched``, ``missing_recording``,
    ``missing_payload``) for construction — they prevent contradictory flag
    combinations.
    """

    request_hash: str
    # ``Mapping`` (not ``dict``): ``freeze_fields`` in ``__post_init__``
    # replaces these with ``MappingProxyType`` (and ``None`` passes
    # through untouched for ``recorded_response``). The annotation must
    # describe the read-only runtime type callers observe.
    live_response: Mapping[str, Any]
    recorded_response: Mapping[str, Any] | None
    is_match: bool | None
    differences: Mapping[str, Any] = field(default_factory=dict)
    recorded_call_missing: bool = False
    payload_missing: bool = False
    no_response_recorded: bool = False

    def __post_init__(self) -> None:
        if self.is_match is True and self.recorded_call_missing:
            raise ValueError("Cannot be a match when recorded call is missing")
        if self.recorded_call_missing and self.payload_missing:
            raise ValueError("Cannot have both recorded_call_missing and payload_missing")
        if self.is_match is True and self.differences:
            raise ValueError("Cannot be a match with non-empty differences")
        if self.no_response_recorded and self.is_match is True:
            raise ValueError("Cannot be a match when no response was recorded")
        if self.no_response_recorded and self.payload_missing:
            raise ValueError("Cannot have both no_response_recorded and payload_missing")
        if self.no_response_recorded and self.recorded_call_missing:
            raise ValueError("Cannot have both no_response_recorded and recorded_call_missing")
        if self.is_match is None and not self.payload_missing:
            raise ValueError("Indeterminate is_match (None) is only valid for missing payloads")
        # Tier 3 external API responses (may be None for recorded_response);
        # deep_freeze handles None as an opaque scalar, so no conditional needed.
        freeze_fields(self, "live_response", "recorded_response", "differences")

    # --- Factory classmethods ---

    @classmethod
    def matched(
        cls,
        request_hash: str,
        live_response: Mapping[str, Any],
        recorded_response: Mapping[str, Any],
    ) -> VerificationResult:
        """Live response matches recorded baseline."""
        return cls(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=recorded_response,
            is_match=True,
        )

    @classmethod
    def mismatched(
        cls,
        request_hash: str,
        live_response: Mapping[str, Any],
        recorded_response: Mapping[str, Any],
        differences: Mapping[str, Any],
    ) -> VerificationResult:
        """Live response differs from recorded baseline."""
        return cls(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=recorded_response,
            is_match=False,
            differences=differences,
        )

    @classmethod
    def missing_recording(
        cls,
        request_hash: str,
        live_response: Mapping[str, Any],
    ) -> VerificationResult:
        """No recorded call found for this request."""
        return cls(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=None,
            is_match=False,
            recorded_call_missing=True,
        )

    @classmethod
    def missing_payload(
        cls,
        request_hash: str,
        live_response: Mapping[str, Any],
        *,
        is_match: bool | None = None,
        differences: Mapping[str, Any] | None = None,
    ) -> VerificationResult:
        """Call exists but response payload is missing or purged.

        Args:
            is_match: True = hash-verified match, False = hash-verified mismatch,
                None (default) = indeterminate (cannot verify, e.g. payload purged
                with ignore_paths/ignore_order configured).
        """
        return cls(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=None,
            is_match=is_match,
            payload_missing=True,
            differences=differences or {},
        )

    @classmethod
    def no_recorded_response(
        cls,
        request_hash: str,
        live_response: Mapping[str, Any],
    ) -> VerificationResult:
        """Call never had a response (e.g., connection timeout, DNS failure)."""
        return cls(
            request_hash=request_hash,
            live_response=live_response,
            recorded_response=None,
            is_match=False,
            no_response_recorded=True,
        )

    @property
    def has_differences(self) -> bool:
        """Check if there are meaningful differences.

        Returns True when there are actual differences between responses.
        Missing recordings, missing payloads (without hash mismatch),
        indeterminate results, and calls that never had a response are NOT
        differences — there is no baseline to compare against in these cases.
        """
        if self.recorded_call_missing:
            return False
        if self.no_response_recorded:
            return False
        if self.is_match is None:
            # Indeterminate — cannot assert differences without evidence.
            return False
        if self.payload_missing:
            # Hash-based comparison may have populated differences even
            # though the full payload is missing. Surface those.
            return not self.is_match and bool(self.differences)
        return not self.is_match


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
            (includes both hash-verified and unverifiable cases)
        unverifiable: Number of calls that could not be verified (payload purged
            and comparison settings prevent hash-based verification)
        no_response_recorded: Number of calls where original call had no response (timeout/DNS)
        results: Individual verification results for inspection
    """

    total_calls: int = 0
    matches: int = 0
    mismatches: int = 0
    missing_recordings: int = 0
    missing_payloads: int = 0
    unverifiable: int = 0
    no_response_recorded: int = 0
    results: list[VerificationResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Percentage of verifiable calls that matched recorded baseline.

        Excludes unverifiable calls from the denominator — unknown outcomes
        cannot count for or against success.
        Returns 100.0 if no verifiable calls have been verified.
        """
        verifiable = self.total_calls - self.unverifiable
        if verifiable == 0:
            return 100.0
        return (self.matches / verifiable) * 100

    def snapshot(self) -> VerificationReport:
        """Return a detached copy safe to hand to external callers."""
        return VerificationReport(
            total_calls=self.total_calls,
            matches=self.matches,
            mismatches=self.mismatches,
            missing_recordings=self.missing_recordings,
            missing_payloads=self.missing_payloads,
            unverifiable=self.unverifiable,
            no_response_recorded=self.no_response_recorded,
            results=list(self.results),
        )


class CallVerifier:
    """Verifies live API calls against recorded responses.

    Used in verify mode to detect drift between current behavior
    and recorded baseline. Uses DeepDiff for comparison.

    Example:
        verifier = CallVerifier(execution, source_run_id="run-abc123")

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
        execution: ExecutionRepository,
        source_run_id: str,
        *,
        ignore_paths: list[str] | None = None,
        ignore_order: bool = True,
    ) -> None:
        """Initialize verifier.

        Args:
            execution: ExecutionRepository for looking up recorded calls
            source_run_id: The run_id containing baseline recordings
            ignore_paths: Paths to ignore in comparison (e.g., ["root['latency']"])
                         These paths will be excluded from DeepDiff comparison.
            ignore_order: If True (default), list ordering differences are ignored.
                         If False, list elements must appear in the same order to match.
                         Set to False for order-sensitive data like ranked results.
        """
        self._execution = execution
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

    def _record_result(
        self,
        sequence_key: tuple[str, str],
        sequence_index: int,
        result: VerificationResult,
        *,
        matches: int = 0,
        mismatches: int = 0,
        missing_recordings: int = 0,
        missing_payloads: int = 0,
        unverifiable: int = 0,
        no_response_recorded: int = 0,
    ) -> VerificationResult:
        """Update in-memory verifier state after a concrete result exists."""
        self._report.total_calls += 1
        self._report.matches += matches
        self._report.mismatches += mismatches
        self._report.missing_recordings += missing_recordings
        self._report.missing_payloads += missing_payloads
        self._report.unverifiable += unverifiable
        self._report.no_response_recorded += no_response_recorded
        self._report.results.append(result)
        self._sequence_counters[sequence_key] = sequence_index + 1
        return result

    def verify(
        self,
        call_type: CallType,
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

        # Read without mutating so exceptional exits leave state untouched.
        if sequence_key in self._sequence_counters:
            sequence_index = self._sequence_counters[sequence_key]
        else:
            sequence_index = 0

        # Look up recorded call with sequence index to get Nth occurrence
        call = self._execution.find_call_by_request_hash(
            run_id=self._source_run_id,
            call_type=call_type,
            request_hash=request_hash,
            sequence_index=sequence_index,
        )

        if call is None:
            result = VerificationResult.missing_recording(
                request_hash=request_hash,
                live_response=live_response,
            )
            return self._record_result(
                sequence_key,
                sequence_index,
                result,
                missing_recordings=1,
            )

        # Get recorded response with explicit state discrimination
        call_data = self._execution.get_call_response_data(call.call_id)

        # Handle non-available states explicitly
        if call_data.state != CallDataState.AVAILABLE:
            # Payload was expected but is missing (purged or store not configured)
            if call_data.state in (CallDataState.PURGED, CallDataState.STORE_NOT_CONFIGURED, CallDataState.HASH_ONLY):
                # When response_hash exists, perform hash-based
                # verification even though the full payload is missing. Per CLAUDE.md:
                # "Hashes survive payload deletion — integrity is always verifiable."
                if call.response_hash is not None:
                    # Hash comparison is only valid when comparison settings are
                    # equivalent to exact match (no ignore_paths, no ignore_order).
                    # With ignore_paths or ignore_order, the configured comparison
                    # defines a LOOSER equivalence than raw hash equality.
                    can_verify_by_hash = not self._ignore_paths and not self._ignore_order

                    if can_verify_by_hash:
                        live_hash = stable_hash(live_response)
                        is_match = live_hash == call.response_hash
                        result = VerificationResult.missing_payload(
                            request_hash=request_hash,
                            live_response=live_response,
                            is_match=is_match,
                            differences={
                                "hash_mismatch": {
                                    "recorded_hash": call.response_hash,
                                    "live_hash": live_hash,
                                }
                            }
                            if not is_match
                            else None,
                        )
                        return self._record_result(
                            sequence_key,
                            sequence_index,
                            result,
                            missing_payloads=1,
                            matches=1 if is_match else 0,
                            mismatches=0 if is_match else 1,
                        )
                    else:
                        # Cannot perform meaningful verification: payload is gone and
                        # hash comparison would be stricter than the configured semantic
                        # comparison (ignore_paths/ignore_order not applicable to hashes).
                        result = VerificationResult.missing_payload(
                            request_hash=request_hash,
                            live_response=live_response,
                            is_match=None,
                            differences={
                                "unverifiable": {
                                    "reason": "Payload purged and comparison settings "
                                    "(ignore_paths/ignore_order) prevent hash-based verification",
                                    "response_hash_available": True,
                                    "ignore_paths": self._ignore_paths,
                                    "ignore_order": self._ignore_order,
                                }
                            },
                        )
                        return self._record_result(
                            sequence_key,
                            sequence_index,
                            result,
                            missing_payloads=1,
                            unverifiable=1,
                        )
                else:
                    # No hash available — cannot verify, just mark as missing
                    result = VerificationResult.missing_payload(
                        request_hash=request_hash,
                        live_response=live_response,
                        is_match=None,
                        differences={
                            "unverifiable": {
                                "reason": "Payload purged and no response hash available",
                                "response_hash_available": False,
                            }
                        },
                    )
                    return self._record_result(
                        sequence_key,
                        sequence_index,
                        result,
                        missing_payloads=1,
                        unverifiable=1,
                    )

            # CALL_NOT_FOUND: The call record was found by find_call_by_request_hash
            # but vanished before get_call_response_data — database corruption or
            # TOCTOU race. This is a Tier 1 integrity violation, not a normal state.
            if call_data.state == CallDataState.CALL_NOT_FOUND:
                raise AuditIntegrityError(
                    f"CALL_NOT_FOUND for call_id={call.call_id} after successful "
                    f"find_call_by_request_hash — audit record vanished between queries. "
                    f"Possible database corruption or concurrent modification."
                )

            # NEVER_STORED: Call exists but never had a response (e.g., connection
            # timeout, DNS failure). Cannot compare, not a match, not missing payload.
            if call_data.state == CallDataState.NEVER_STORED:
                result = VerificationResult.no_recorded_response(
                    request_hash=request_hash,
                    live_response=live_response,
                )
                return self._record_result(
                    sequence_key,
                    sequence_index,
                    result,
                    no_response_recorded=1,
                )

            # Unknown state — CallDataState may have gained a new member.
            # Offensive programming: crash rather than silently misclassify.
            raise AuditIntegrityError(
                f"Unexpected CallDataState {call_data.state!r} for call_id={call.call_id}. "
                f"The verifier does not know how to handle this state — "
                f"update the verify() method to handle the new state explicitly."
            )

        # AVAILABLE — compare recorded response with live response using DeepDiff.
        # CallDataResult.data is deep-frozen (dict→MappingProxyType, list→tuple).
        # Thaw back to mutable types so DeepDiff compares content, not container types.
        recorded_response = deep_thaw(call_data.data)
        diff = DeepDiff(
            recorded_response,
            live_response,
            ignore_order=self._ignore_order,
            exclude_paths=self._ignore_paths,
        )

        is_match = len(diff) == 0

        if is_match:
            result = VerificationResult.matched(
                request_hash=request_hash,
                live_response=live_response,
                recorded_response=recorded_response,
            )
            return self._record_result(
                sequence_key,
                sequence_index,
                result,
                matches=1,
            )
        else:
            result = VerificationResult.mismatched(
                request_hash=request_hash,
                live_response=live_response,
                recorded_response=recorded_response,
                differences=diff.to_dict(),
            )
            return self._record_result(
                sequence_key,
                sequence_index,
                result,
                mismatches=1,
            )

    def get_report(self) -> VerificationReport:
        """Get the verification report for this session.

        Returns a defensive snapshot of all verifications performed.
        """
        return self._report.snapshot()

    def reset_report(self) -> None:
        """Reset the verification report and sequence counters.

        Clears all accumulated statistics, results, and sequence counters.
        Use this when starting a new verification session.

        Note: This also resets sequence counters, so the next verification
        of any request will compare against the first recorded occurrence.
        """
        self._report = VerificationReport()
        self._sequence_counters = defaultdict(int)
