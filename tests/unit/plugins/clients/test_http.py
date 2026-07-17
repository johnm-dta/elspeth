"""Tests for AuditedHTTPClient (GET and POST methods).

Coverage goals:
- POST method recording to Landscape
- Header fingerprinting (auth tokens, API keys)
- Response format handling (JSON, text, binary)
- Telemetry emission error handling
- Rate limiting integration
"""

import base64
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.call_data import HTTPCallError, HTTPCallRequest, HTTPCallResponse
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient, HTTPResponseBodyTooLargeError


class _CallArgs:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index: int) -> tuple[Any, ...] | dict[str, Any]:
        if index == 0:
            return self.args
        if index == 1:
            return self.kwargs
        raise IndexError(index)


class _CallRecorder:
    def __init__(self, *, return_value: Any = None, side_effect: Exception | None = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_args: _CallArgs | None = None
        self.call_args_list: list[_CallArgs] = []

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args = _CallArgs(args, kwargs)
        self.call_args_list.append(self.call_args)
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    def assert_called_once(self) -> None:
        assert self.call_count == 1


class _ExecutionRepositoryFake:
    def __init__(self) -> None:
        self._next_call_index = 0
        self.record_call = _CallRecorder(return_value=SimpleNamespace(call_id="call-1"))

    def allocate_call_index(self, _state_id: str) -> int:
        call_index = self._next_call_index
        self._next_call_index += 1
        return call_index

    def allocate_operation_call_index(self, _operation_id: str) -> int:
        return self.allocate_call_index(_operation_id)

    def record_operation_call(self, **kwargs: Any) -> Any:
        return self.record_call(**kwargs)


class _LimiterFake:
    def __init__(self) -> None:
        self.acquire = _CallRecorder()


@pytest.fixture
def mock_execution():
    """Create mock ExecutionRepository."""
    return _ExecutionRepositoryFake()


@pytest.fixture
def mock_telemetry_emit():
    """Create mock telemetry emit callback."""
    return _CallRecorder()


@pytest.fixture
def http_client(mock_execution, mock_telemetry_emit):
    """Create AuditedHTTPClient with mocked dependencies."""
    return AuditedHTTPClient(
        execution=mock_execution,
        state_id="test-state-001",
        run_id="test-run-001",
        telemetry_emit=mock_telemetry_emit,
        timeout=30.0,
    )


class _CountingByteStream(httpx.SyncByteStream):
    """Streaming fixture that exposes how many chunks the client consumed."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.yielded: list[bytes] = []

    def __iter__(self) -> Iterator[bytes]:
        for chunk in self._chunks:
            self.yielded.append(chunk)
            yield chunk


# ============================================================================
# POST Method Tests
# ============================================================================


@respx.mock
def test_post_records_call_to_landscape(http_client, mock_execution):
    """POST should record request/response to Landscape."""
    respx.post("https://api.example.com/v1/process").mock(return_value=httpx.Response(200, json={"result": "success"}))

    response = http_client.post(
        "https://api.example.com/v1/process",
        json={"data": "value"},
    )

    assert response.status_code == 200

    # Verify Landscape recording
    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args[1]

    assert call_args["state_id"] == "test-state-001"
    assert call_args["call_type"] == CallType.HTTP
    assert call_args["status"] == CallStatus.SUCCESS
    assert call_args["request_data"].to_dict()["method"] == "POST"
    assert call_args["request_data"].to_dict()["url"] == "https://api.example.com/v1/process"
    assert call_args["request_data"].to_dict()["json"] == {"data": "value"}
    assert call_args["response_data"].to_dict()["status_code"] == 200
    assert call_args["response_data"].to_dict()["body"] == {"result": "success"}
    assert call_args["latency_ms"] > 0


@respx.mock
def test_post_handles_json_response(http_client):
    """POST should parse JSON responses correctly."""
    respx.post("https://api.example.com/endpoint").mock(
        return_value=httpx.Response(
            200,
            json={"status": "ok", "data": {"id": 123}},
            headers={"content-type": "application/json"},
        )
    )

    response = http_client.post(
        "https://api.example.com/endpoint",
        json={"input": "test"},
    )

    data = response.json()
    assert data["status"] == "ok"
    assert data["data"]["id"] == 123


@respx.mock
def test_post_handles_text_response(http_client, mock_execution):
    """POST should handle text responses (non-JSON)."""
    respx.post("https://api.example.com/endpoint").mock(
        return_value=httpx.Response(
            200,
            text="Plain text response",
            headers={"content-type": "text/plain"},
        )
    )

    response = http_client.post(
        "https://api.example.com/endpoint",
        json={"input": "test"},
    )

    assert response.text == "Plain text response"

    # Verify body stored as text (not dict)
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["response_data"].to_dict()["body"] == "Plain text response"


@respx.mock
def test_post_handles_binary_response(http_client, mock_execution):
    """POST should base64-encode binary responses."""
    binary_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

    respx.post("https://api.example.com/image").mock(
        return_value=httpx.Response(
            200,
            content=binary_data,
            headers={"content-type": "image/png"},
        )
    )

    response = http_client.post(
        "https://api.example.com/image",
        json={"format": "png"},
    )

    assert response.content == binary_data

    # Verify body stored as base64
    call_args = mock_execution.record_call.call_args[1]
    assert "_binary" in call_args["response_data"].to_dict()["body"]
    decoded = base64.b64decode(call_args["response_data"].to_dict()["body"]["_binary"])
    assert decoded == binary_data


@respx.mock
def test_post_fingerprints_auth_headers(http_client, mock_execution):
    """POST should fingerprint sensitive headers (not store raw secrets)."""
    with patch.dict("os.environ", {"ELSPETH_FINGERPRINT_KEY": "test-key-12345", "ELSPETH_ALLOW_RAW_SECRETS": ""}):
        respx.post("https://api.example.com/secure").mock(return_value=httpx.Response(200, json={}))

        http_client.post(
            "https://api.example.com/secure",
            json={},
            headers={"Authorization": "Bearer secret-token-xyz"},
        )

        # Verify auth header was fingerprinted (not stored raw)
        call_args = mock_execution.record_call.call_args[1]
        auth_header = call_args["request_data"].to_dict()["headers"]["Authorization"]

        assert "secret-token-xyz" not in auth_header
        assert auth_header.startswith("<fingerprint:")
        assert len(auth_header) > 20  # HMAC fingerprint is long


@respx.mock
def test_post_fingerprints_compact_secret_headers(http_client, mock_execution):
    """Compact secret header names (apikey/authkey) should be fingerprinted."""
    with patch.dict("os.environ", {"ELSPETH_FINGERPRINT_KEY": "test-key-12345", "ELSPETH_ALLOW_RAW_SECRETS": ""}):
        respx.post("https://api.example.com/secure").mock(return_value=httpx.Response(200, json={}))

        http_client.post(
            "https://api.example.com/secure",
            json={},
            headers={"apikey": "api-secret", "authkey": "auth-secret"},
        )

        call_args = mock_execution.record_call.call_args[1]
        recorded_headers = call_args["request_data"].to_dict()["headers"]

        assert recorded_headers["apikey"].startswith("<fingerprint:")
        assert recorded_headers["authkey"].startswith("<fingerprint:")
        assert "api-secret" not in recorded_headers["apikey"]
        assert "auth-secret" not in recorded_headers["authkey"]


@respx.mock
def test_post_telemetry_failure_doesnt_corrupt_audit(http_client, mock_execution, mock_telemetry_emit):
    """Telemetry emission failure must not prevent Landscape recording."""
    # Make telemetry callback raise
    mock_telemetry_emit.side_effect = RuntimeError("Telemetry service unavailable")

    respx.post("https://api.example.com/endpoint").mock(return_value=httpx.Response(200, json={"ok": True}))

    # POST should succeed despite telemetry failure
    response = http_client.post("https://api.example.com/endpoint", json={})

    assert response.status_code == 200

    # Landscape recording must have happened
    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["status"] == CallStatus.SUCCESS


@respx.mock
def test_post_telemetry_emits_token_id_when_configured(mock_execution, mock_telemetry_emit):
    """Telemetry event should include token_id when client has token context."""
    client = AuditedHTTPClient(
        execution=mock_execution,
        state_id="test-state-001",
        run_id="test-run-001",
        telemetry_emit=mock_telemetry_emit,
        token_id="tok-123",
    )

    respx.post("https://api.example.com/token").mock(return_value=httpx.Response(200, json={"ok": True}))
    response = client.post("https://api.example.com/token", json={"x": 1})

    assert response.status_code == 200
    assert mock_telemetry_emit.call_count == 1
    event = mock_telemetry_emit.call_args[0][0]
    assert event.token_id == "tok-123"


@respx.mock
def test_post_telemetry_token_id_none_when_unset(http_client, mock_telemetry_emit):
    """Telemetry event should allow token_id=None when token is not provided."""
    respx.post("https://api.example.com/no-token").mock(return_value=httpx.Response(200, json={"ok": True}))
    response = http_client.post("https://api.example.com/no-token", json={"x": 1})

    assert response.status_code == 200
    assert mock_telemetry_emit.call_count == 1
    event = mock_telemetry_emit.call_args[0][0]
    assert event.token_id is None


@respx.mock
def test_post_with_error_response(http_client, mock_execution):
    """POST with 4xx/5xx should record as ERROR status."""
    respx.post("https://api.example.com/fail").mock(return_value=httpx.Response(500, text="Internal Server Error"))

    response = http_client.post("https://api.example.com/fail", json={})

    assert response.status_code == 500

    # Verify recorded as ERROR
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["status"] == CallStatus.ERROR
    assert call_args["error"].type == "HTTPError"
    assert "500" in call_args["error"].message


@respx.mock
def test_post_with_network_error(http_client, mock_execution):
    """POST network failure should record error and re-raise."""
    respx.post("https://api.example.com/unreachable").mock(side_effect=httpx.ConnectError("Connection refused"))

    with pytest.raises(httpx.ConnectError, match="Connection refused"):
        http_client.post("https://api.example.com/unreachable", json={})

    # Verify error recorded to Landscape
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["status"] == CallStatus.ERROR
    assert call_args["error"].type == "ConnectError"


@respx.mock
def test_post_with_base_url(mock_execution, mock_telemetry_emit):
    """POST should properly join base_url with path."""
    client = AuditedHTTPClient(
        execution=mock_execution,
        state_id="test-state",
        run_id="test-run",
        telemetry_emit=mock_telemetry_emit,
        base_url="https://api.example.com/v2",
    )

    respx.post("https://api.example.com/v2/users").mock(return_value=httpx.Response(201, json={"id": 42}))

    response = client.post("/users", json={"name": "Alice"})

    assert response.status_code == 201

    # Verify full URL was constructed correctly
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["request_data"].to_dict()["url"] == "https://api.example.com/v2/users"


@respx.mock
def test_post_malformed_json_response(http_client, mock_execution):
    """POST with Content-Type: application/json but invalid body should handle gracefully."""
    # Server claims JSON but sends malformed data
    respx.post("https://api.example.com/broken").mock(
        return_value=httpx.Response(
            200,
            text="Not valid JSON {{{",
            headers={"content-type": "application/json"},
        )
    )

    response = http_client.post("https://api.example.com/broken", json={})

    # Should not crash
    assert response.status_code == 200

    # Verify audit trail records the parse failure
    call_args = mock_execution.record_call.call_args[1]
    body = call_args["response_data"].to_dict()["body"]

    assert body["_json_parse_failed"] is True
    assert "_error" in body
    assert "_raw_text" in body
    assert "Not valid JSON" in body["_raw_text"]


# ============================================================================
# GET Method Tests
# ============================================================================


@respx.mock
def test_get_records_call_to_landscape(http_client, mock_execution):
    """GET should record request/response to Landscape."""
    respx.get("https://api.example.com/status").mock(return_value=httpx.Response(200, json={"status": "healthy"}))

    response = http_client.get("https://api.example.com/status")

    assert response.status_code == 200

    # Verify Landscape recording
    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args[1]

    assert call_args["request_data"].to_dict()["method"] == "GET"
    assert call_args["request_data"].to_dict()["url"] == "https://api.example.com/status"
    assert call_args["status"] == CallStatus.SUCCESS


@respx.mock
def test_get_with_params(http_client, mock_execution):
    """GET should include query parameters in request."""
    respx.get("https://api.example.com/search").mock(return_value=httpx.Response(200, json={"results": []}))

    response = http_client.get(
        "https://api.example.com/search",
        params={"q": "test query", "limit": 10},
    )

    assert response.status_code == 200

    # Verify params recorded
    call_args = mock_execution.record_call.call_args[1]
    assert call_args["request_data"].to_dict()["params"] == {"q": "test query", "limit": 10}


@respx.mock
def test_get_telemetry_failure_doesnt_corrupt_audit(http_client, mock_execution, mock_telemetry_emit):
    """Telemetry emission failure must not prevent Landscape recording (GET)."""
    mock_telemetry_emit.side_effect = Exception("Telemetry down")

    respx.get("https://api.example.com/data").mock(return_value=httpx.Response(200, json={}))

    response = http_client.get("https://api.example.com/data")

    assert response.status_code == 200
    mock_execution.record_call.assert_called_once()


@pytest.mark.parametrize(
    "response_headers",
    [
        {},
        {"content-length": "3"},
        {"transfer-encoding": "chunked"},
    ],
)
@respx.mock
def test_get_body_cap_streams_and_aborts_without_trusting_response_headers(
    response_headers: dict[str, str],
    mock_execution,
    mock_telemetry_emit,
) -> None:
    """Configured body cap aborts during streaming for absent, lying, and chunked lengths."""
    stream = _CountingByteStream([b"abc", b"def", b"must-not-read"])
    headers = {"content-type": "text/plain", **response_headers}
    respx.get("https://api.example.com/huge").mock(return_value=httpx.Response(200, headers=headers, stream=stream))
    client = AuditedHTTPClient(
        execution=mock_execution,
        state_id="test-state-001",
        run_id="test-run-001",
        telemetry_emit=mock_telemetry_emit,
        timeout=30.0,
        max_response_body_bytes=5,
    )

    with pytest.raises(HTTPResponseBodyTooLargeError) as exc_info:
        client.get("https://api.example.com/huge")

    assert exc_info.value.body_size == 6
    assert exc_info.value.max_body_bytes == 5
    assert stream.yielded == [b"abc", b"def"]
    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args.kwargs
    assert call_args["status"] is CallStatus.ERROR
    assert call_args["error"].type == "HTTPResponseBodyTooLargeError"
    response_data = call_args["response_data"].to_dict()
    assert response_data["body_size"] == 6
    assert response_data["body"] == {
        "_truncated": True,
        "_reason": "body_too_large",
        "_captured_body": False,
        "_observed_body_size": 6,
        "_max_body_bytes": 5,
    }


def test_body_cap_error_audit_sanitizes_url_without_changing_raised_error(
    http_client: AuditedHTTPClient,
    mock_execution: _ExecutionRepositoryFake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit errors must not persist URL credentials, fragments, or sensitive query values."""
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-key-for-http-error-audit")
    monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)
    raw_url = "https://o'connor:password@api.example.com/huge?token=!!!!&view=summary#access_token=FRAGMENT_SECRET"
    response_payload = HTTPCallResponse(
        status_code=200,
        headers={},
        body_size=6,
        body={"_truncated": True, "_reason": "body_too_large"},
    )
    failure = HTTPResponseBodyTooLargeError(
        url=raw_url,
        body_size=6,
        max_body_bytes=5,
        response_payload=response_payload,
    )

    def raise_body_cap(*_args: object, **_kwargs: object) -> httpx.Response:
        raise failure

    monkeypatch.setattr(http_client, "_request_with_optional_body_cap", raise_body_cap)

    with pytest.raises(HTTPResponseBodyTooLargeError) as exc_info:
        http_client.get(raw_url)

    assert exc_info.value is failure
    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args
    assert call_args is not None
    recorded_message = call_args.kwargs["error"].message
    assert "response body 6 bytes exceeds max_response_body_bytes 5" in recorded_message
    assert "api.example.com/huge" in recorded_message
    assert "view=summary" in recorded_message
    assert "token=" in recorded_message
    assert "o'connor" not in recorded_message
    assert "password" not in recorded_message
    assert "!!!!" not in recorded_message
    assert "FRAGMENT_SECRET" not in recorded_message


def test_http_error_audit_redacts_url_when_fingerprinting_is_unavailable(
    http_client: AuditedHTTPClient,
    mock_execution: _ExecutionRepositoryFake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit sanitization failures redact the URL instead of aborting persistence."""
    monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
    monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

    http_client._record_call(
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.ERROR,
        request_data=HTTPCallRequest(method="GET", url="https://api.example.com/huge", headers={}),
        error=HTTPCallError(
            type="ConnectError",
            message="connection failed for https://api.example.com/huge?token=!!!!",
        ),
    )

    mock_execution.record_call.assert_called_once()
    call_args = mock_execution.record_call.call_args
    assert call_args is not None
    recorded_message = call_args.kwargs["error"].message
    assert recorded_message == "connection failed for <redacted-http-url>"
    assert "token" not in recorded_message
    assert "!!!!" not in recorded_message


@pytest.mark.parametrize(
    "raw_url",
    [
        "https://api.example.com/huge?token=<RAW_SECRET>",
        "https://alice:pass<word>@api.example.com/huge",
    ],
)
def test_http_error_audit_redacts_full_non_whitespace_url_token_on_ambiguous_delimiters(
    raw_url: str,
    http_client: AuditedHTTPClient,
    mock_execution: _ExecutionRepositoryFake,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Angle brackets inside an attacker URL cannot split off raw credentials."""
    monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
    monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

    http_client._record_call(
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.ERROR,
        request_data=HTTPCallRequest(method="GET", url="https://api.example.com/huge", headers={}),
        error=HTTPCallError(type="ConnectError", message=f"connection failed for {raw_url}"),
    )

    call_args = mock_execution.record_call.call_args
    assert call_args is not None
    persisted = call_args.kwargs["error"].message
    assert persisted.startswith("connection failed for ")
    assert "RAW_SECRET" not in persisted
    assert "alice" not in persisted
    assert "pass<word>" not in persisted


# ============================================================================
# Header Fingerprinting Tests
# ============================================================================


@respx.mock
def test_header_fingerprinting_with_missing_key_raises(http_client, mock_execution, monkeypatch):
    """Missing fingerprint key with sensitive headers raises FrameworkBugError.

    Regression test for elspeth-780f6e39b1: previously this path silently
    dropped the header with only a logger.warning.
    """
    from elspeth.contracts.errors import FrameworkBugError

    # Remove fingerprint key and raw secrets flag without clearing entire env
    monkeypatch.delenv("ELSPETH_FINGERPRINT_KEY", raising=False)
    monkeypatch.delenv("ELSPETH_ALLOW_RAW_SECRETS", raising=False)

    respx.post("https://api.example.com/secure").mock(return_value=httpx.Response(200, json={}))

    with pytest.raises(FrameworkBugError, match="Sensitive header 'Authorization' cannot be fingerprinted"):
        http_client.post(
            "https://api.example.com/secure",
            json={},
            headers={"Authorization": "Bearer secret"},
        )


@respx.mock
def test_header_fingerprinting_in_dev_mode(http_client, mock_execution, monkeypatch):
    """Dev mode (ELSPETH_ALLOW_RAW_SECRETS=true) should remove sensitive headers."""
    monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "true")

    respx.post("https://api.example.com/secure").mock(return_value=httpx.Response(200, json={}))

    http_client.post(
        "https://api.example.com/secure",
        json={},
        headers={"Authorization": "Bearer dev-token", "X-Custom": "value"},
    )

    # Verify sensitive header removed, non-sensitive kept
    call_args = mock_execution.record_call.call_args[1]
    headers = call_args["request_data"].to_dict()["headers"]

    assert "Authorization" not in headers
    assert headers["X-Custom"] == "value"


def test_sensitive_header_detection(http_client):
    """Verify sensitive header detection logic.

    Uses word-boundary matching (dash-delimited segments) to avoid
    false positives from substring matching.
    """
    # Well-known sensitive headers (exact match)
    sensitive_exact = [
        "authorization",
        "Authorization",
        "apikey",
        "authkey",
        "X-API-Key",
        "api-key",
        "X-Auth-Token",
        "proxy-authorization",
        "cookie",
        "Cookie",
        "set-cookie",
        "www-authenticate",
        "proxy-authenticate",
        "x-access-token",
        "x-csrf-token",
        "x-xsrf-token",
        "Ocp-Apim-Subscription-Key",
    ]

    for header in sensitive_exact:
        assert http_client._is_sensitive_header(header), f"{header} should be sensitive (exact)"

    # Sensitive via word-segment matching (dash-delimited)
    sensitive_word = [
        "Custom-Auth-Header",  # segment "auth"
        "Secret-Key",  # segments "secret" and "key"
        "Bearer-Token",  # segment "token"
        "X-Session-Token",  # segment "token"
        "X-Authorization",  # segment "authorization"
        "X-Secret-Value",  # segment "secret"
        "X-Password-Hash",  # segment "password"
        "X-Credential-Id",  # segment "credential"
        "XAuthorization",  # compact x+authorization form
        # Concatenated forms without delimiters (regression: P2 header filtering)
        "X-AuthToken",  # "authtoken" compound word
        "X-AccessToken",  # "accesstoken" compound word
    ]

    for header in sensitive_word:
        assert http_client._is_sensitive_header(header), f"{header} should be sensitive (word)"

    # Non-sensitive headers — must NOT be flagged
    non_sensitive = [
        "Content-Type",
        "User-Agent",
        "Accept",
        "X-Request-ID",
        # Regression: these were false positives with substring matching
        "X-Author",  # "author" ≠ "auth"
        "X-Authored-By",  # "authored" ≠ "auth"
        "X-Monkey-Patch",  # "monkey" ≠ "key"
        "X-Turkey-Id",  # "turkey" ≠ "key"
        "X-Keyboard-Layout",  # "keyboard" ≠ "key"
        "X-Hotkey-Name",  # "hotkey" ≠ "key"
        "X-Secretary-Id",  # "secretary" ≠ "secret"
        "X-Tokenizer-Version",  # "tokenizer" ≠ "token"
    ]

    for header in non_sensitive:
        assert not http_client._is_sensitive_header(header), f"{header} should not be sensitive"


@respx.mock
def test_response_headers_filter_sensitive(http_client, mock_execution):
    """Response headers should have sensitive values filtered."""
    respx.get("https://api.example.com/login").mock(
        return_value=httpx.Response(
            200,
            json={},
            headers={
                "Set-Cookie": "session=secret123; HttpOnly",
                "Content-Type": "application/json",
                "X-Request-ID": "abc-123",
            },
        )
    )

    http_client.get("https://api.example.com/login")

    # Verify Set-Cookie filtered out
    call_args = mock_execution.record_call.call_args[1]
    headers = call_args["response_data"].to_dict()["headers"]

    assert "set-cookie" not in headers
    assert headers["content-type"] == "application/json"
    assert headers["x-request-id"] == "abc-123"


# ============================================================================
# Rate Limiting Tests
# ============================================================================


@respx.mock
def test_rate_limiting_integration(mock_execution, mock_telemetry_emit):
    """Rate limiter should be invoked before HTTP call."""
    mock_limiter = _LimiterFake()

    client = AuditedHTTPClient(
        execution=mock_execution,
        state_id="test-state",
        run_id="test-run",
        telemetry_emit=mock_telemetry_emit,
        limiter=mock_limiter,
    )

    respx.get("https://api.example.com/endpoint").mock(return_value=httpx.Response(200, json={}))

    client.get("https://api.example.com/endpoint")

    # Verify rate limiter was called
    mock_limiter.acquire.assert_called_once()


# ============================================================================
# Provider Extraction Tests (Security)
# ============================================================================


def test_extract_provider_strips_credentials(http_client):
    """Provider extraction must not leak credentials from URL."""
    # URL with embedded credentials (userinfo)
    url = "https://user:password@api.example.com/endpoint"

    provider = http_client._extract_provider(url)

    # Should return only hostname (no credentials)
    assert provider == "api.example.com"
    assert "user" not in provider
    assert "password" not in provider


def test_extract_provider_handles_port(http_client):
    """Provider extraction should return hostname without port."""
    url = "https://api.example.com:8443/endpoint"

    provider = http_client._extract_provider(url)

    assert provider == "api.example.com"


def test_extract_provider_unknown_url(http_client):
    """Provider extraction should return 'unknown' for malformed URLs."""
    url = "not-a-valid-url"

    provider = http_client._extract_provider(url)

    assert provider == "unknown"
