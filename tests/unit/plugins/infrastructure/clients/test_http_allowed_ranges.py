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
from unittest.mock import Mock, patch

import httpx
import pytest

from elspeth.core.security.web import SSRFSafeRequest


@pytest.fixture
def mock_execution():
    """Minimal ExecutionRepository mock for AuditedHTTPClient."""
    execution = Mock()
    execution.record_call = Mock()
    return execution


@pytest.fixture
def mock_telemetry_emit():
    """No-op telemetry callback."""
    return Mock()


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

    def test_allowed_ranges_passed_to_redirect_validation(self, mock_execution, mock_telemetry_emit) -> None:
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
            execution=mock_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=mock_telemetry_emit,
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

            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                    return_value=redirect_safe_request,
                ) as mock_validate,
                patch("httpx.Client") as MockClient,
            ):
                # First httpx.Client() call is the initial request — returns redirect
                initial_client = Mock()
                initial_client.__enter__ = Mock(return_value=initial_client)
                initial_client.__exit__ = Mock(return_value=False)
                initial_client.get.return_value = redirect_response

                # Second httpx.Client() call is the redirect hop — returns 200
                hop_client = Mock()
                hop_client.__enter__ = Mock(return_value=hop_client)
                hop_client.__exit__ = Mock(return_value=False)
                hop_client.get.return_value = final_response

                MockClient.side_effect = [initial_client, hop_client]

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

    def test_empty_allowed_ranges_default_preserved_in_redirect(self, mock_execution, mock_telemetry_emit) -> None:
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
            execution=mock_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=mock_telemetry_emit,
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

            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                    return_value=redirect_safe_request,
                ) as mock_validate,
                patch("httpx.Client") as MockClient,
            ):
                initial_client = Mock()
                initial_client.__enter__ = Mock(return_value=initial_client)
                initial_client.__exit__ = Mock(return_value=False)
                initial_client.get.return_value = redirect_response

                hop_client = Mock()
                hop_client.__enter__ = Mock(return_value=hop_client)
                hop_client.__exit__ = Mock(return_value=False)
                hop_client.get.return_value = final_response

                MockClient.side_effect = [initial_client, hop_client]

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

    def test_allowed_ranges_not_threaded_when_no_redirect(self, mock_execution, mock_telemetry_emit) -> None:
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
            execution=mock_execution,
            state_id="test-state",
            run_id="test-run",
            telemetry_emit=mock_telemetry_emit,
        )

        try:
            ok_response = httpx.Response(
                200,
                text="<html>Content</html>",
                request=httpx.Request("GET", "http://93.184.216.34:80/page"),
            )

            # Patch httpx.Client globally — get_ssrf_safe uses an ephemeral client
            with (
                patch(
                    "elspeth.plugins.infrastructure.clients.http.validate_url_for_ssrf",
                ) as mock_validate,
                patch("httpx.Client") as MockClient,
            ):
                initial_client = Mock()
                initial_client.__enter__ = Mock(return_value=initial_client)
                initial_client.__exit__ = Mock(return_value=False)
                initial_client.get.return_value = ok_response
                MockClient.return_value = initial_client

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
