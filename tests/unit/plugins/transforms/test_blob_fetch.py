"""Tests for blob_fetch transform."""

from __future__ import annotations

import hashlib
import urllib.parse
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.audit import Call
from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed", "guaranteed_fields": ["url"]}


class _PayloadStoreFake:
    def __init__(self) -> None:
        self.stored_payloads: list[bytes] = []

    def store(self, content: bytes) -> str:
        self.stored_payloads.append(content)
        return hashlib.sha256(content).hexdigest()


def _safe_request_for(url: str, *, resolved_ip: str = "203.0.113.10") -> SSRFSafeRequest:
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    hostname = parsed.hostname
    assert hostname is not None
    default_port = 443 if scheme == "https" else 80
    host_header = parsed.netloc if port != default_port else hostname
    return SSRFSafeRequest(
        original_url=url,
        resolved_ip=resolved_ip,
        host_header=host_header,
        port=port,
        path=path,
        scheme=scheme,
        bare_hostname=hostname,
    )


def _call() -> Call:
    return Call(
        call_id="test-call-id",
        call_index=0,
        call_type=CallType.HTTP,
        status=CallStatus.SUCCESS,
        request_hash="test-request-hash",
        created_at=datetime.now(UTC),
        state_id="state-123",
        request_ref="test-request-ref-hash",
        response_hash="test-response-hash",
        response_ref="test-response-ref-hash",
        latency_ms=100.0,
    )


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema": DYNAMIC_SCHEMA,
        "url_field": "url",
        "http": {
            "abuse_contact": "ops@example.com",
            "fetch_reason": "unit test",
        },
    }
    config.update(overrides)
    return config


def test_blob_fetch_stores_body_and_emits_blob_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.plugins.transforms.blob_fetch as blob_fetch_module
    from elspeth.plugins.transforms.blob_fetch import BlobFetch

    monkeypatch.setattr(
        blob_fetch_module,
        "validate_url_for_ssrf",
        lambda url, allowed_ranges=(): _safe_request_for(url),
    )

    body = b"id,name\n1,alice\n"
    response = httpx.Response(
        200,
        content=body,
        headers={"content-type": "text/csv; charset=utf-8"},
        request=httpx.Request("GET", "https://203.0.113.10:443/data.csv"),
    )
    transform = BlobFetch(_config())
    payload_store = _PayloadStoreFake()
    transform._payload_store = payload_store
    transform._fetch_url = lambda _safe, _ctx: (response, "https://example.test/data.csv", _call())  # type: ignore[method-assign]

    result = transform.process(
        make_pipeline_row({"url": "https://example.test/data.csv", "batch_id": "run-1"}),
        make_context(),
    )

    assert result.status == "success"
    assert result.row is not None
    output = result.row.to_dict()
    expected_hash = hashlib.sha256(body).hexdigest()
    assert payload_store.stored_payloads == [body]
    assert output["url"] == "https://example.test/data.csv"
    assert output["batch_id"] == "run-1"
    assert output["blob_ref"] == expected_hash
    assert output["blob_sha256"] == expected_hash
    assert output["blob_content_type"] == "text/csv"
    assert output["blob_size_bytes"] == len(body)
    assert output["fetch_status"] == 200
    assert output["fetch_url_final"] == "https://example.test/data.csv"
    assert output["fetch_url_final_ip"] == "203.0.113.10"


def test_blob_fetch_blocks_ssrf_rejected_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.plugins.transforms.blob_fetch as blob_fetch_module
    from elspeth.plugins.transforms.blob_fetch import BlobFetch

    def _blocked(url: str, allowed_ranges=()) -> SSRFSafeRequest:
        raise SSRFBlockedError(f"blocked: {url}")

    monkeypatch.setattr(blob_fetch_module, "validate_url_for_ssrf", _blocked)

    result = BlobFetch(_config()).process(
        make_pipeline_row({"url": "http://127.0.0.1/data.csv"}),
        make_context(),
    )

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "validation_failed"
    assert result.reason["error_type"] == "SSRFBlockedError"


def test_blob_fetch_rejects_unapproved_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    import elspeth.plugins.transforms.blob_fetch as blob_fetch_module
    from elspeth.plugins.transforms.blob_fetch import BlobFetch

    monkeypatch.setattr(
        blob_fetch_module,
        "validate_url_for_ssrf",
        lambda url, allowed_ranges=(): _safe_request_for(url),
    )

    response = httpx.Response(
        200,
        content=b"%PDF-1.7",
        headers={"content-type": "application/pdf"},
        request=httpx.Request("GET", "https://203.0.113.10:443/file.pdf"),
    )
    transform = BlobFetch(_config())
    payload_store = _PayloadStoreFake()
    transform._payload_store = payload_store
    transform._fetch_url = lambda _safe, _ctx: (response, "https://example.test/file.pdf", _call())  # type: ignore[method-assign]

    result = transform.process(make_pipeline_row({"url": "https://example.test/file.pdf"}), make_context())

    assert result.status == "error"
    assert result.reason is not None
    assert result.reason["reason"] == "unsupported_content_type"
    assert payload_store.stored_payloads == []


def test_discovery_registers_blob_fetch() -> None:
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()

    transform = manager.get_transform_by_name("blob_fetch")
    assert transform.name == "blob_fetch"
