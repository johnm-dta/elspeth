# tests/plugins/clients/test_verifier.py
"""Tests for CallVerifier."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.contracts import Call, CallStatus, CallType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.row_data import CallDataResult, CallDataState
from elspeth.plugins.infrastructure.clients.verifier import (
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

    def test_has_differences_false_when_payload_missing(self) -> None:
        """has_differences is False when payload is purged (not a real diff)."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response=None,
            is_match=False,
            payload_missing=True,
        )

        # Missing payload is not considered a "difference"
        assert result.has_differences is False

    def test_indeterminate_is_match_none(self) -> None:
        """is_match=None represents indeterminate verification.

        Bug: elspeth-d943c8f11a — previously, unverifiable calls used
        is_match=False which is indistinguishable from definitive mismatch.
        """
        result = VerificationResult.missing_payload(
            request_hash="abc123",
            live_response={"content": "Hello"},
            is_match=None,
            differences={"unverifiable": {"reason": "test"}},
        )

        assert result.is_match is None
        assert result.payload_missing is True
        assert result.has_differences is False  # Indeterminate is not a difference

    def test_indeterminate_invalid_without_payload_missing(self) -> None:
        """is_match=None is forbidden outside of missing payload context."""
        with pytest.raises(ValueError, match=r"Indeterminate.*only valid for missing payloads"):
            VerificationResult(
                request_hash="abc123",
                live_response={"content": "Hello"},
                recorded_response=None,
                is_match=None,
            )


class TestVerificationResultPostFreezeMutationGuards:
    """Exercise the ``freeze_fields`` post-init guard on ``VerificationResult``.

    ``enforce_freeze_guards.py`` is a static scanner — a typo that drops
    a field name from the ``freeze_fields(self, ...)`` call still parses
    as valid Python and silently defeats the scan.  These tests pin the
    runtime contract by mutating each container field after construction
    and asserting the post-freeze TypeError surfaces.  If a refactor
    drops a field, the mutation succeeds and this test fails loudly.
    """

    def test_live_response_is_immutable_after_construction(self) -> None:
        """Mutating ``VerificationResult.live_response`` must raise TypeError."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response={"content": "Hello"},
            is_match=True,
        )

        with pytest.raises(TypeError):
            result.live_response["content"] = "mutated"  # type: ignore[index]

        with pytest.raises(TypeError):
            result.live_response["new_key"] = "x"  # type: ignore[index]

    def test_recorded_response_is_immutable_after_construction(self) -> None:
        """Mutating ``VerificationResult.recorded_response`` must raise TypeError."""
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "Hello"},
            recorded_response={"content": "Hello"},
            is_match=True,
        )

        assert result.recorded_response is not None
        with pytest.raises(TypeError):
            result.recorded_response["content"] = "mutated"  # type: ignore[index]

        with pytest.raises(TypeError):
            result.recorded_response["new_key"] = "x"  # type: ignore[index]

    def test_differences_is_immutable_after_construction(self) -> None:
        """Mutating ``VerificationResult.differences`` must raise TypeError.

        This is the specific leak that a scanner-defeating typo would
        cause: the differences map is consumed by VerificationReport and
        downstream audit emitters — a mutable reference would allow a
        caller to rewrite the diff after the match decision was made.
        """
        result = VerificationResult(
            request_hash="abc123",
            live_response={"content": "New"},
            recorded_response={"content": "Old"},
            is_match=False,
            differences={"values_changed": {"root['content']": {"new_value": "New", "old_value": "Old"}}},
        )

        with pytest.raises(TypeError):
            result.differences["values_changed"] = {}  # type: ignore[index]

        with pytest.raises(TypeError):
            result.differences["added_key"] = {}  # type: ignore[index]

    def test_recorded_response_none_does_not_crash_freeze(self) -> None:
        """``recorded_response=None`` is a valid state (missing payload);
        freeze must handle None transparently without raising.

        This pins the docstring's claim at verifier.py:77-78 that
        ``deep_freeze`` handles None as an opaque scalar, so no
        conditional guard is needed.  A future refactor that adds an
        ``if self.recorded_response is not None`` guard around the
        freeze call would still work, but a refactor that drops the
        field from ``freeze_fields`` entirely would fail this test via
        the live_response/differences tests above.
        """
        result = VerificationResult.missing_payload(
            request_hash="abc123",
            live_response={"content": "Hello"},
            is_match=None,
            differences={"unverifiable": {"reason": "payload purged"}},
        )

        assert result.recorded_response is None
        # live_response still guarded
        with pytest.raises(TypeError):
            result.live_response["content"] = "mutated"  # type: ignore[index]

    def test_source_detachment_across_all_container_fields(self) -> None:
        """Source mutations to inputs must NOT affect the stored views."""
        live_src = {"content": "Hello"}
        recorded_src = {"content": "Hello"}
        diffs_src: dict[str, Any] = {"values_changed": {}}

        result = VerificationResult(
            request_hash="abc123",
            live_response=live_src,
            recorded_response=recorded_src,
            is_match=False,
            differences=diffs_src,
        )

        live_src["content"] = "mutated"
        recorded_src["content"] = "mutated"
        diffs_src["values_changed"] = {"root['x']": {"new_value": "Z"}}

        assert result.live_response["content"] == "Hello"
        assert result.recorded_response is not None
        assert result.recorded_response["content"] == "Hello"
        assert result.differences["values_changed"] == {}


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_default_values(self) -> None:
        """VerificationReport has sensible defaults."""
        report = VerificationReport()

        assert report.total_calls == 0
        assert report.matches == 0
        assert report.mismatches == 0
        assert report.missing_recordings == 0
        assert report.missing_payloads == 0
        assert report.unverifiable == 0
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

    def test_success_rate_excludes_unverifiable(self) -> None:
        """success_rate denominator excludes unverifiable calls.

        Unverifiable calls are unknown outcomes — they cannot count
        for or against success. 7 matches out of 8 verifiable calls = 87.5%.
        """
        report = VerificationReport(
            total_calls=10,
            matches=7,
            mismatches=1,
            missing_payloads=2,
            unverifiable=2,
        )
        # Denominator = 10 - 2 = 8 verifiable calls
        assert report.success_rate == (7 / 8) * 100

    def test_success_rate_all_unverifiable(self) -> None:
        """success_rate returns 100% when all calls are unverifiable."""
        report = VerificationReport(
            total_calls=3,
            missing_payloads=3,
            unverifiable=3,
        )
        assert report.success_rate == 100.0


class TestCallVerifier:
    """Tests for CallVerifier."""

    def _create_mock_recorder(self) -> MagicMock:
        """Create a mock ExecutionRepository."""
        recorder = MagicMock()
        recorder.find_call_by_request_hash = MagicMock(return_value=None)
        recorder.get_call_response_data = MagicMock(return_value=CallDataResult(state=CallDataState.STORE_NOT_CONFIGURED, data=None))
        return recorder

    def _create_mock_call(
        self,
        *,
        call_id: str = "call_123",
        status: CallStatus = CallStatus.SUCCESS,
        request_hash: str = "hash123",
        latency_ms: float = 150.0,
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
            response_ref=response_ref,
            response_hash=response_hash,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Live response matches recorded
        result = verifier.verify(
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        result = verifier.verify(
            call_type=CallType.LLM,
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
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # Ignore latency in comparison
        verifier = CallVerifier(
            recorder,
            source_run_id="run_abc123",
            ignore_paths=["root['latency']"],
        )

        result = verifier.verify(
            call_type=CallType.LLM,
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

        def find_call_side_effect(run_id: str, call_type: str, request_hash: str, *, sequence_index: int = 0) -> Call | None:
            # First two requests have recordings, third doesn't
            if request_hash == stable_hash(request3):
                return None
            return self._create_mock_call(request_hash=request_hash)

        def get_response_side_effect(call_id: str) -> CallDataResult:
            return CallDataResult(state=CallDataState.AVAILABLE, data={"content": "recorded"})

        recorder.find_call_by_request_hash.side_effect = find_call_side_effect
        recorder.get_call_response_data.side_effect = get_response_side_effect

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # First call: match
        verifier.verify(
            call_type=CallType.LLM,
            request_data=request1,
            live_response={"content": "recorded"},
        )

        # Second call: mismatch
        verifier.verify(
            call_type=CallType.LLM,
            request_data=request2,
            live_response={"content": "different"},
        )

        # Third call: missing recording
        verifier.verify(
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data={"content": "match"})

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Make 4 matching calls
        for i in range(4):
            verifier.verify(
                call_type=CallType.LLM,
                request_data={"id": i},
                live_response={"content": "match"},
            )

        report = verifier.get_report()
        assert report.success_rate == 100.0

        # Now add a mismatch
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data={"content": "different"})
        verifier.verify(
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data={"content": "match"})

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # Make a verification
        verifier.verify(
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data={"status": 200, "body": "OK"})

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.HTTP,
            request_data=request_data,
            live_response={"status": 200, "body": "OK"},
        )

        assert result.is_match is True
        recorder.find_call_by_request_hash.assert_called_once_with(
            run_id="run_abc123",
            call_type=CallType.HTTP,
            request_hash=request_hash,
            sequence_index=0,
        )

    def test_verify_with_purged_response_payload(self) -> None:
        """Handles calls where response payload was recorded but is now purged.

        When response_ref is set but get_call_response_data returns None,
        the payload was purged. This should be tracked as missing_payload.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",  # Response WAS recorded
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Hello"},
        )

        # Should track as missing payload — indeterminate (cannot verify)
        assert result.is_match is None
        assert result.recorded_response is None
        assert result.payload_missing is True
        assert result.has_differences is False  # Indeterminate is not a difference
        report = verifier.get_report()
        assert report.missing_payloads == 1
        assert report.unverifiable == 1
        # Should not increment matches or mismatches
        assert report.mismatches == 0
        assert report.matches == 0

    def test_verify_error_call_without_response_not_missing_payload(self) -> None:
        """Error calls that never had a response should NOT be flagged as missing payload.

        Bug: P2-2026-01-31-verifier-error-misclassification

        Some error calls legitimately have no response (connection timeout, DNS failure).
        These have response_ref=None, meaning no response was ever recorded.
        This should NOT be counted as payload_missing.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.ERROR,
            response_ref=None,  # Never had a response
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.NEVER_STORED, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"error": "timeout"},
        )

        # Should NOT be marked as payload_missing
        assert result.payload_missing is False
        assert verifier.get_report().missing_payloads == 0
        # The call exists but has no response to compare
        assert result.recorded_response is None
        # Cannot match since there's no recorded response
        assert result.is_match is False

    def test_verify_error_call_with_purged_response_is_missing_payload(self) -> None:
        """Error calls that HAD a response but it was purged SHOULD be flagged.

        HTTP error responses (400, 500) include response bodies.
        If that payload was purged, it's a genuine missing payload.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.ERROR,
            response_ref="payload_ref_456",  # Error call with response body
            response_hash="hash_of_response",  # Hash proves response existed
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"error": "bad request"},
        )

        # SHOULD be marked as payload_missing (response was recorded but purged)
        assert result.payload_missing is True
        assert verifier.get_report().missing_payloads == 1

    def test_verify_hash_match_when_payload_purged(self) -> None:
        """When payload is purged but response_hash exists, hash-based match succeeds.

        P1-2026-02-05: Per CLAUDE.md "Hashes survive payload deletion — integrity
        is always verifiable." The verifier must compare stable_hash(live_response)
        against the recorded response_hash even when the full payload is gone.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello, world!", "model": "gpt-4"}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        # ignore_order=False required for hash-based verification
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=original_response,  # Same response
        )

        # Hash-based match should succeed
        assert result.is_match is True
        assert result.payload_missing is True
        assert result.has_differences is False
        assert result.differences == {}
        report = verifier.get_report()
        assert report.matches == 1
        assert report.mismatches == 0
        assert report.missing_payloads == 1

    def test_verify_hash_mismatch_when_payload_purged(self) -> None:
        """When payload is purged and live response differs, hash mismatch is detected.

        P1-2026-02-05: Drift detection must work even in ATTRIBUTABLE_ONLY runs
        where payloads are purged but response_hash survives.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello, world!", "model": "gpt-4"}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        # ignore_order=False required for hash-based verification
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "DIFFERENT response", "model": "gpt-4"},
        )

        # Hash-based mismatch should be detected
        assert result.is_match is False
        assert result.payload_missing is True
        assert result.has_differences is True  # Hash mismatch IS a real difference
        assert "hash_mismatch" in result.differences
        assert result.differences["hash_mismatch"]["recorded_hash"] == response_hash
        report = verifier.get_report()
        assert report.matches == 0
        assert report.mismatches == 1
        assert report.missing_payloads == 1

    def test_verify_success_with_no_response_payload_not_missing(self) -> None:
        """SUCCESS calls with no response evidence should NOT be flagged as missing_payload.

        Bug: P2-2026-02-14-callverifier-verify-misclassifies-legitimate-successful-calls

        Some successful operations legitimately have no response body (e.g., DELETE,
        204 No Content). When response_ref=None and response_hash=None, there is no
        evidence a response was ever recorded, so this is not a missing payload.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            status=CallStatus.SUCCESS,
            response_ref=None,
            response_hash=None,  # No response was ever recorded
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.NEVER_STORED, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Hello"},
        )

        # No evidence of response ever existing — NOT payload_missing
        assert result.payload_missing is False
        assert result.recorded_call_missing is False
        assert result.recorded_response is None
        assert result.is_match is False
        report = verifier.get_report()
        assert report.missing_payloads == 0
        assert report.matches == 0
        assert report.mismatches == 0

    def test_verify_hash_only_uses_hash_based_verification(self) -> None:
        """HASH_ONLY state (response_hash exists, no payload) should use hash-based
        verification — same path as PURGED and STORE_NOT_CONFIGURED."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello, world!", "model": "gpt-4"}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.HASH_ONLY, data=None)

        # ignore_order=False required for hash-based verification
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=original_response,  # Same response — hash should match
        )

        assert result.is_match is True
        assert result.payload_missing is True
        assert result.has_differences is False
        report = verifier.get_report()
        assert report.matches == 1
        assert report.missing_payloads == 1

    def test_verify_hash_only_mismatch(self) -> None:
        """HASH_ONLY state with different live response detects hash mismatch."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello, world!"}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.HASH_ONLY, data=None)

        # ignore_order=False required for hash-based verification
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "DIFFERENT"},
        )

        assert result.is_match is False
        assert result.payload_missing is True
        assert "hash_mismatch" in result.differences
        report = verifier.get_report()
        assert report.mismatches == 1
        assert report.missing_payloads == 1

    def test_verify_hash_skipped_when_ignore_paths_configured(self) -> None:
        """Hash verification must be skipped when ignore_paths is configured.

        Bug: elspeth-116d98f421 — hash comparison is stricter than semantic
        comparison. When ignore_paths or ignore_order is set, a hash mismatch
        could be a false alarm (the difference might be in an ignored path).
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello", "latency": 100}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        # With ignore_paths configured, hash comparison is unreliable
        verifier = CallVerifier(
            recorder,
            source_run_id="run_abc123",
            ignore_paths=["root['latency']"],
        )
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Hello", "latency": 999},  # Only ignored field differs
        )

        # Should NOT attempt hash verification — result is indeterminate
        assert result.payload_missing is True
        assert result.is_match is None  # Indeterminate — cannot verify
        assert "unverifiable" in result.differences
        report = verifier.get_report()
        assert report.matches == 0
        assert report.mismatches == 0  # No false alarm
        assert report.unverifiable == 1

    def test_verify_hash_skipped_when_ignore_order_true(self) -> None:
        """Hash verification must be skipped when ignore_order=True (the default).

        Bug: elspeth-116d98f421 — with ignore_order=True, semantic comparison
        treats [1,2,3] and [3,2,1] as equal, but their hashes differ.
        """
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)
        original_response = {"items": ["a", "b", "c"]}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        # Default ignore_order=True
        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"items": ["c", "a", "b"]},  # Same items, different order
        )

        # Should NOT attempt hash verification — indeterminate
        assert result.payload_missing is True
        assert result.is_match is None  # Indeterminate — cannot verify
        assert "unverifiable" in result.differences
        report = verifier.get_report()
        assert report.matches == 0
        assert report.mismatches == 0
        assert report.unverifiable == 1

    def test_verify_hash_works_when_strict_comparison(self) -> None:
        """Hash verification proceeds when no ignore_paths and ignore_order=False."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)
        original_response = {"content": "Hello"}
        response_hash = stable_hash(original_response)

        mock_call = self._create_mock_call(
            request_hash=request_hash,
            response_ref="payload_ref_123",
            response_hash=response_hash,
        )
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        # Strict: no ignore_paths, ignore_order=False
        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=original_response,
        )

        # Hash verification should succeed
        assert result.is_match is True
        assert result.payload_missing is True
        report = verifier.get_report()
        assert report.matches == 1

    def test_verify_order_independent_with_default_config(self) -> None:
        """Default configuration (ignore_order=True) ignores list ordering."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {"items": ["a", "b", "c"]}
        live_response = {"items": ["c", "a", "b"]}  # Same items, different order

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )

        # Should match because ignore_order=True by default
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        result = verifier.verify(
            call_type=CallType.LLM,
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
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # Ignore both timestamp and request_id
        verifier = CallVerifier(
            recorder,
            source_run_id="run_abc123",
            ignore_paths=["root['timestamp']", "root['request_id']"],
        )

        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )

        # Should match because differences are in ignored paths
        assert result.is_match is True

    def test_duplicate_requests_verify_against_different_recordings(self) -> None:
        """Duplicate identical requests verify against different recorded responses.

        This tests the scenario where the same request is made multiple times
        in the original run and each verification compares against the
        corresponding recorded response.

        Bug: P1-2026-01-21-replay-request-hash-collisions
        """
        recorder = self._create_mock_recorder()
        # Same request data made multiple times
        request_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

        # Track which sequence_index we're being asked for
        recorded_responses = [
            {"content": "Response 1 - first call"},
            {"content": "Response 2 - second call"},
            {"content": "Response 3 - third call"},
        ]

        def find_call_side_effect(run_id: str, call_type: str, request_hash: str, *, sequence_index: int = 0) -> Call | None:
            """Return the Nth call for duplicate request hashes."""
            if sequence_index >= len(recorded_responses):
                return None
            return self._create_mock_call(
                call_id=f"call_{sequence_index}",
                request_hash=request_hash,
            )

        def get_response_side_effect(call_id: str) -> CallDataResult:
            """Return response data based on call_id."""
            idx = int(call_id.split("_")[1])
            return CallDataResult(state=CallDataState.AVAILABLE, data=recorded_responses[idx])

        recorder.find_call_by_request_hash.side_effect = find_call_side_effect
        recorder.get_call_response_data.side_effect = get_response_side_effect

        verifier = CallVerifier(recorder, source_run_id="run_abc123")

        # First verify - matches first recorded response
        result1 = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Response 1 - first call"},
        )
        assert result1.is_match is True, "First verify should match first recorded response"

        # Second verify - should compare against SECOND recorded response
        result2 = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Response 2 - second call"},
        )
        assert result2.is_match is True, "Second verify should match second recorded response"

        # Third verify with mismatched response
        result3 = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Wrong response"},
        )
        assert result3.is_match is False, "Third verify should detect mismatch"

        # Verify report stats
        report = verifier.get_report()
        assert report.total_calls == 3
        assert report.matches == 2
        assert report.mismatches == 1

    def test_verify_order_sensitive_when_configured(self) -> None:
        """Verifier detects order changes when ignore_order=False."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {"items": ["a", "b", "c"]}
        live_response = {"items": ["c", "b", "a"]}  # Same items, different order

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result = verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )

        # Should NOT match because ignore_order=False
        assert result.is_match is False
        assert result.has_differences is True
        # DeepDiff reports position changes as values_changed
        assert "values_changed" in result.differences

    def test_ignore_order_handles_duplicate_elements(self) -> None:
        """List comparisons treat lists as multisets when ignore_order=True."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {"tags": ["a", "a", "b"]}
        live_response = {"tags": ["b", "a", "a"]}  # Same multiset, different order

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # With ignore_order=True (default): should match (same multiset)
        verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
        result_loose = verifier_loose.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_loose.is_match is True

        # With ignore_order=False: should NOT match (different positions)
        verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result_strict = verifier_strict.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_strict.is_match is False

    def test_ignore_order_applies_recursively_to_nested_lists(self) -> None:
        """Document that ignore_order affects ALL list levels recursively."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response = {
            "results": [
                {"id": 1, "tags": ["a", "b"]},
                {"id": 2, "tags": ["x", "y"]},
            ]
        }
        live_response = {
            "results": [
                {"id": 2, "tags": ["y", "x"]},  # Both levels reordered
                {"id": 1, "tags": ["b", "a"]},
            ]
        }

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # With ignore_order=True: matches (recursive order-independence)
        verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
        result_loose = verifier_loose.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_loose.is_match is True

        # With ignore_order=False: does NOT match
        verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result_strict = verifier_strict.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_strict.is_match is False

    def test_ignore_order_does_not_affect_dict_keys(self) -> None:
        """Dict key ordering is always ignored (JSON semantics)."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        # Python dicts maintain insertion order, but JSON treats them as unordered
        recorded_response = {"z": 1, "a": 2, "m": 3}
        live_response = {"a": 2, "m": 3, "z": 1}  # Same keys/values, different order

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # Both with and without ignore_order: dicts should match
        verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
        result_loose = verifier_loose.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_loose.is_match is True

        verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result_strict = verifier_strict.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_strict.is_match is True

    def test_empty_lists_always_match(self) -> None:
        """Empty lists match regardless of ignore_order setting."""
        recorder = self._create_mock_recorder()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)

        recorded_response: dict[str, list[Any]] = {"items": []}
        live_response: dict[str, list[Any]] = {"items": []}

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # Both settings should match
        for ignore_order in [True, False]:
            verifier = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=ignore_order)
            result = verifier.verify(
                call_type=CallType.LLM,
                request_data=request_data,
                live_response=live_response,
            )
            assert result.is_match is True, f"Failed with ignore_order={ignore_order}"

    def test_order_sensitivity_with_realistic_llm_response(self) -> None:
        """Verify order handling with actual LLM tool call structure."""
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}
        request_hash = stable_hash(request_data)

        recorded_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                            {"id": "call_2", "function": {"name": "summarize", "arguments": "{}"}},
                            {"id": "call_3", "function": {"name": "respond", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        }
        live_response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "call_2", "function": {"name": "summarize", "arguments": "{}"}},
                            {"id": "call_1", "function": {"name": "search", "arguments": "{}"}},
                            {"id": "call_3", "function": {"name": "respond", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        }

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.AVAILABLE, data=recorded_response)

        # With ignore_order=True (default): matches despite tool call reordering
        verifier_loose = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=True)
        result_loose = verifier_loose.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_loose.is_match is True

        # With ignore_order=False: tool call reordering is detected as drift
        verifier_strict = CallVerifier(recorder, source_run_id="run_abc123", ignore_order=False)
        result_strict = verifier_strict.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response=live_response,
        )
        assert result_strict.is_match is False, "Tool call reordering should be detected"


# =============================================================================
# Bug fix tests: verifier.py bug cluster
# =============================================================================


class TestCallNotFoundRaisesIntegrityError:
    """Tests for elspeth-fd540d3818: CALL_NOT_FOUND should raise AuditIntegrityError."""

    def _create_mock_recorder(self) -> MagicMock:
        recorder = MagicMock()
        recorder.find_call_by_request_hash = MagicMock(return_value=None)
        recorder.get_call_response_data = MagicMock(return_value=CallDataResult(state=CallDataState.STORE_NOT_CONFIGURED, data=None))
        return recorder

    def _create_mock_call(self, *, request_hash: str) -> Call:
        return Call(
            call_id="call_123",
            state_id="state_123",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=150.0,
            response_ref=None,
            response_hash=None,
        )

    def test_call_not_found_raises_audit_integrity_error(self) -> None:
        """CALL_NOT_FOUND after successful find_call_by_request_hash is corruption.

        If find_call_by_request_hash returns a call record, but get_call_response_data
        then says CALL_NOT_FOUND, the call vanished between the two queries. This
        indicates database corruption or a TOCTOU race and must raise AuditIntegrityError.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.CALL_NOT_FOUND, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        with pytest.raises(AuditIntegrityError, match="CALL_NOT_FOUND"):
            verifier.verify(
                call_type=CallType.LLM,
                request_data=request_data,
                live_response={"content": "Hello"},
            )


class TestCounterAccountingGap:
    """Tests for elspeth-ddb745ea0e: PURGED/no-hash path must increment a counter."""

    def _create_mock_recorder(self) -> MagicMock:
        recorder = MagicMock()
        recorder.find_call_by_request_hash = MagicMock(return_value=None)
        recorder.get_call_response_data = MagicMock(return_value=CallDataResult(state=CallDataState.STORE_NOT_CONFIGURED, data=None))
        return recorder

    def _create_mock_call(self, *, request_hash: str) -> Call:
        return Call(
            call_id="call_123",
            state_id="state_123",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=150.0,
            response_ref=None,
            response_hash=None,  # No hash available
        )

    def test_purged_no_hash_increments_counter(self) -> None:
        """PURGED state with no response_hash must increment unverifiable counter.

        Previously, missing_payloads was incremented but neither matches nor
        mismatches was — the result had is_match=False but should be None
        (indeterminate). Now properly tracked as unverifiable.
        """
        recorder = self._create_mock_recorder()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)

        mock_call = self._create_mock_call(request_hash=request_hash)
        recorder.find_call_by_request_hash.return_value = mock_call
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.PURGED, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "Hello"},
        )

        report = verifier.get_report()
        assert report.total_calls == 1
        assert report.missing_payloads == 1
        assert report.unverifiable == 1
        # Accounting invariant: every call is in exactly one bucket.
        accounted = report.matches + report.mismatches + report.missing_recordings + report.missing_payloads + report.no_response_recorded
        assert accounted == report.total_calls, f"Counter accounting: {accounted} accounted != {report.total_calls} total"
        result = report.results[0]
        assert result.payload_missing is True
        assert result.is_match is None  # Indeterminate, not False


class TestVerifierStateIntegrityOnExceptionalPaths:
    """Regression tests for exceptional paths that must not mutate verifier state."""

    def _create_mock_call(self, *, request_hash: str) -> Call:
        return Call(
            call_id="call_123",
            state_id="state_123",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=150.0,
            response_ref=None,
            response_hash=None,
        )

    def test_call_not_found_leaves_report_and_sequence_state_unchanged(self) -> None:
        """CALL_NOT_FOUND must raise without consuming verifier state."""
        recorder = MagicMock()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        recorder.find_call_by_request_hash.return_value = self._create_mock_call(request_hash=request_hash)
        recorder.get_call_response_data.return_value = CallDataResult(state=CallDataState.CALL_NOT_FOUND, data=None)

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        sequence_key = (CallType.LLM, request_hash)

        with pytest.raises(AuditIntegrityError, match="CALL_NOT_FOUND"):
            verifier.verify(
                call_type=CallType.LLM,
                request_data=request_data,
                live_response={"content": "Hello"},
            )

        report = verifier.get_report()
        assert report.total_calls == 0
        assert report.matches == 0
        assert report.mismatches == 0
        assert report.missing_recordings == 0
        assert report.missing_payloads == 0
        assert report.unverifiable == 0
        assert report.no_response_recorded == 0
        assert report.results == []
        assert sequence_key not in verifier._sequence_counters

    def test_repository_integrity_error_leaves_report_and_sequence_state_unchanged(self) -> None:
        """Repository-raised AuditIntegrityError must not corrupt verifier state."""
        recorder = MagicMock()
        request_data = {"model": "gpt-4", "messages": []}
        request_hash = stable_hash(request_data)
        recorder.find_call_by_request_hash.return_value = self._create_mock_call(request_hash=request_hash)
        recorder.get_call_response_data.side_effect = AuditIntegrityError("payload integrity failure")

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        sequence_key = (CallType.LLM, request_hash)

        with pytest.raises(AuditIntegrityError, match="payload integrity failure"):
            verifier.verify(
                call_type=CallType.LLM,
                request_data=request_data,
                live_response={"content": "Hello"},
            )

        report = verifier.get_report()
        assert report.total_calls == 0
        assert report.matches == 0
        assert report.mismatches == 0
        assert report.missing_recordings == 0
        assert report.missing_payloads == 0
        assert report.unverifiable == 0
        assert report.no_response_recorded == 0
        assert report.results == []
        assert sequence_key not in verifier._sequence_counters


class TestHasDifferencesNoRecordedResponse:
    """Tests for elspeth-3895543969: has_differences misleading for no_recorded_response."""

    def test_no_recorded_response_has_differences_is_false(self) -> None:
        """no_recorded_response should NOT report has_differences=True.

        The original call never had a response (e.g., connection timeout).
        There's nothing to compare against — reporting 'differences' is misleading.
        Callers checking has_differences would interpret it as drift detected.
        """
        result = VerificationResult.no_recorded_response(
            request_hash="abc123",
            live_response={"content": "Hello"},
        )

        assert result.is_match is False  # Correct — can't match without baseline
        assert result.has_differences is False  # Bug: was True, should be False
        assert result.recorded_call_missing is False
        assert result.payload_missing is False


class TestGetReportSnapshots:
    """Regression tests for report snapshot isolation."""

    def test_get_report_returns_defensive_snapshot(self) -> None:
        """Mutating a returned report must not alter verifier internals."""
        recorder = MagicMock()
        request_data = {"id": 1}
        request_hash = stable_hash(request_data)
        recorder.find_call_by_request_hash.return_value = Call(
            call_id="call_123",
            state_id="state_123",
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_hash=request_hash,
            created_at=datetime.now(UTC),
            latency_ms=150.0,
            response_ref=None,
            response_hash=None,
        )
        recorder.get_call_response_data.return_value = CallDataResult(
            state=CallDataState.AVAILABLE,
            data={"content": "recorded"},
        )

        verifier = CallVerifier(recorder, source_run_id="run_abc123")
        verifier.verify(
            call_type=CallType.LLM,
            request_data=request_data,
            live_response={"content": "recorded"},
        )

        snapshot = verifier.get_report()
        snapshot.matches = 99
        snapshot.results.append(
            VerificationResult.missing_recording(
                request_hash="forged",
                live_response={"content": "forged"},
            )
        )

        fresh = verifier.get_report()
        assert fresh is not snapshot
        assert fresh.matches == 1
        assert len(fresh.results) == 1
        assert fresh.results[0].request_hash == request_hash
