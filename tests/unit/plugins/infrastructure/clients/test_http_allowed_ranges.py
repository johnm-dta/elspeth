"""Tests for allowed_ranges parameter threading through AuditedHTTPClient redirect chain.

These tests verify that get_ssrf_safe() and _follow_redirects_safe() correctly
pass allowed_ranges through to validate_url_for_ssrf() at each redirect hop.

The critical risk: allowed_ranges has 4 handoff points in the redirect chain.
A dropped parameter at any hop defaults to () (full blocklist), which silently
breaks the allowlist feature for redirect scenarios. These tests catch that.

Approach: We monkeypatch validate_url_for_ssrf at the module where it's imported
(elspeth.plugins.infrastructure.clients.http) so we can inspect whether
allowed_ranges is passed through during redirect hops. We use a real initial
response with a 301 redirect to trigger _follow_redirects_safe.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
import pytest

from elspeth.core.security.web import SSRFSafeRequest


@pytest.fixture
def fake_execution():
    """Minimal CallRecorder fake for AuditedHTTPClient."""
    return FakeCallRecorder()


@pytest.fixture
def telemetry_sink():
    """No-op telemetry callback."""
    return TelemetrySink()


@dataclass
class RecordedCall:
    state_id: str | None
    operation_id: str | None
    call_index: int
    call_type: object
    status: object
    request_data: object
    response_data: object | None = None
    error: object | None = None
    latency_ms: float | None = None
    request_ref: str | None = None
    response_ref: str | None = None


@dataclass
class FakeCallRecorder:
    calls: list[RecordedCall] = field(default_factory=list)
    _next_state_call_index: int = 0
    _next_operation_call_index: int = 0

    def allocate_call_index(self, state_id: str) -> int:
        self._next_state_call_index += 1
        return self._next_state_call_index

    def allocate_operation_call_index(self, operation_id: str) -> int:
        self._next_operation_call_index += 1
        return self._next_operation_call_index

    def record_call(
        self,
        state_id: str,
        call_index: int,
        call_type: object,
        status: object,
        request_data: object,
        response_data: object | None = None,
        error: object | None = None,
        latency_ms: float | None = None,
        *,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> RecordedCall:
        del resolved_prompt_template_hash
        call = RecordedCall(
            state_id=state_id,
            operation_id=None,
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_data=request_data,
            response_data=response_data,
            error=error,
            latency_ms=latency_ms,
            request_ref=request_ref or f"request-{call_index}",
            response_ref=response_ref or f"response-{call_index}",
        )
        self.calls.append(call)
        return call

    def record_operation_call(
        self,
        operation_id: str,
        call_type: object,
        status: object,
        request_data: object,
        response_data: object | None = None,
        error: object | None = None,
        latency_ms: float | None = None,
        *,
        call_index: int | None = None,
        request_ref: str | None = None,
        response_ref: str | None = None,
        resolved_prompt_template_hash: str | None = None,
    ) -> RecordedCall:
        del resolved_prompt_template_hash
        index = call_index if call_index is not None else self.allocate_operation_call_index(operation_id)
        call = RecordedCall(
            state_id=None,
            operation_id=operation_id,
            call_index=index,
            call_type=call_type,
            status=status,
            request_data=request_data,
            response_data=response_data,
            error=error,
            latency_ms=latency_ms,
            request_ref=request_ref or f"operation-request-{index}",
            response_ref=response_ref or f"operation-response-{index}",
        )
        self.calls.append(call)
        return call


@dataclass
class TelemetrySink:
    events: list[object] = field(default_factory=list)

    def __call__(self, event: object) -> None:
        self.events.append(event)


@dataclass
class FakeHTTPContextClient:
    response: httpx.Response
    get_calls: list[tuple[str, dict[str, object]]] = field(default_factory=list)

    def __enter__(self) -> FakeHTTPContextClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.get_calls.append((url, kwargs))
        return self.response


@dataclass
class FakeHTTPClientFactory:
    responses: list[httpx.Response]
    clients: list[FakeHTTPContextClient] = field(default_factory=list)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = field(default_factory=list)

    def __call__(self, *args: object, **kwargs: object) -> FakeHTTPContextClient:
        if not self.responses:
            raise AssertionError("Unexpected httpx.Client construction")
        self.calls.append((args, kwargs))
        client = FakeHTTPContextClient(self.responses.pop(0))
        self.clients.append(client)
        return client


class TestRedirectAllowedRangesThreading:
    """Verify allowed_ranges is threaded from get_ssrf_safe through _follow_redirects_safe
    to the validate_url_for_ssrf call at each redirect hop.

    Architecture:
      get_ssrf_safe(request, allowed_ranges=X)
        -> _follow_redirects_safe(..., allowed_ranges=X)
          -> validate_url_for_ssrf(redirect_url, allowed_ranges=X)  <-- must receive X

    We patch validate_url_for_ssrf at the import site inside http.py so we can
    capture the kwargs it receives during redirect processing.
    """

    def test_allowed_ranges_passed_to_redirect_validation(self, fake_execution, telemetry_sink) -> None:
        """allowed_ranges from get_ssrf_safe reaches validate_url_for_ssrf in redirect hop."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        allowed = (ipaddress.ip_network("127.0.0.0/8"),)

        # Build a realistic initial SSRFSafeRequest (as if validate_url_for_ssrf already ran)
        initial_request = SSRFSafeRequest(
            original_url="http://example.com/start",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/start",
            scheme="http",
            bare_hostname="example.com",
        )

        # Build a redirect SSRFSafeRequest that validate_url_for_ssrf would return
        redirect_safe_request = SSRFSafeRequest(
            original_url="http://localhost/redirected",
            resolved_ip="127.0.0.1",
            host_header="localhost",
            port=80,
            path="/redirected",
            scheme="http",
            bare_hostname="localhost",
        )

        client = AuditedHTTPClient(
            execution=fake_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=telemetry_sink,
        )

        try:
            # Mock the initial HTTP request to return a 301 redirect
            redirect_response = httpx.Response(
                301,
                headers={"location": "http://localhost/redirected"},
                request=httpx.Request("GET", "http://93.184.216.34:80/start"),
            )
            # Mock the follow-up request to return 200
            final_response = httpx.Response(
                200,
                text="OK",
                request=httpx.Request("GET", "http://127.0.0.1:80/redirected"),
            )
            http_client_factory = FakeHTTPClientFactory([redirect_response, final_response])

            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                    autospec=True,
                    return_value=redirect_safe_request,
                ) as mock_validate,
                patch("httpx.Client", new=http_client_factory),
            ):
                client.get_ssrf_safe(
                    initial_request,
                    follow_redirects=True,
                    allowed_ranges=allowed,
                )

                # Assert validate_url_for_ssrf was called during the redirect hop
                # and received the allowed_ranges parameter
                mock_validate.assert_called_once()
                call_kwargs = mock_validate.call_args
                assert call_kwargs.kwargs.get("allowed_ranges") == allowed or (
                    len(call_kwargs.args) > 1 and call_kwargs.args[1] == allowed
                ), f"validate_url_for_ssrf was called during redirect but allowed_ranges was not passed through. Call args: {call_kwargs}"
        finally:
            client.close()

    def test_empty_allowed_ranges_default_preserved_in_redirect(self, fake_execution, telemetry_sink) -> None:
        """When no allowed_ranges is passed to get_ssrf_safe, redirect hops get default ()."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        initial_request = SSRFSafeRequest(
            original_url="http://example.com/start",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/start",
            scheme="http",
            bare_hostname="example.com",
        )

        redirect_safe_request = SSRFSafeRequest(
            original_url="http://other.example.com/page",
            resolved_ip="93.184.216.35",
            host_header="other.example.com",
            port=80,
            path="/page",
            scheme="http",
            bare_hostname="other.example.com",
        )

        client = AuditedHTTPClient(
            execution=fake_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=telemetry_sink,
        )

        try:
            redirect_response = httpx.Response(
                301,
                headers={"location": "http://other.example.com/page"},
                request=httpx.Request("GET", "http://93.184.216.34:80/start"),
            )
            final_response = httpx.Response(
                200,
                text="OK",
                request=httpx.Request("GET", "http://93.184.216.35:80/page"),
            )
            http_client_factory = FakeHTTPClientFactory([redirect_response, final_response])

            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                    autospec=True,
                    return_value=redirect_safe_request,
                ) as mock_validate,
                patch("httpx.Client", new=http_client_factory),
            ):
                # Call WITHOUT allowed_ranges — should default to ()
                client.get_ssrf_safe(
                    initial_request,
                    follow_redirects=True,
                )

                mock_validate.assert_called_once()
                call_kwargs = mock_validate.call_args
                actual_allowed = call_kwargs.kwargs.get("allowed_ranges", ())
                assert actual_allowed == (), f"Default allowed_ranges should be () but got {actual_allowed}"
        finally:
            client.close()

    def test_allowed_ranges_not_threaded_when_no_redirect(self, fake_execution, telemetry_sink) -> None:
        """When response is not a redirect, validate_url_for_ssrf is not called again."""
        from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient

        allowed = (ipaddress.ip_network("10.0.0.0/8"),)

        initial_request = SSRFSafeRequest(
            original_url="http://example.com/page",
            resolved_ip="93.184.216.34",
            host_header="example.com",
            port=80,
            path="/page",
            scheme="http",
            bare_hostname="example.com",
        )

        client = AuditedHTTPClient(
            execution=fake_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=telemetry_sink,
        )

        try:
            ok_response = httpx.Response(
                200,
                text="<html>Content</html>",
                request=httpx.Request("GET", "http://93.184.216.34:80/page"),
            )
            http_client_factory = FakeHTTPClientFactory([ok_response])

            # Patch httpx.Client globally — get_ssrf_safe uses an ephemeral client
            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                    autospec=True,
                ) as mock_validate,
                patch("httpx.Client", new=http_client_factory),
            ):
                client.get_ssrf_safe(
                    initial_request,
                    follow_redirects=True,
                    allowed_ranges=allowed,
                )

                # No redirect, so validate_url_for_ssrf should NOT be called
                # (it was already called before get_ssrf_safe by the caller)
                mock_validate.assert_not_called()
        finally:
            client.close()
