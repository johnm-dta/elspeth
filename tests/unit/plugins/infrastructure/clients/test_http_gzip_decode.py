"""Regression: a capped streaming response must not double-decompress gzip bodies.

``_consume_capped_response`` reads the body via ``iter_bytes()`` (which has ALREADY
applied content-decoding) but used to reconstruct the ``httpx.Response`` with the
original ``Content-Encoding`` header. On read, httpx then re-ran the gzip decoder on
the already-decoded HTML -> ``httpx.DecodingError`` ("Error -3 while decompressing
data: incorrect header check"). This surfaced live when GitHub Pages began serving
``content-encoding: gzip`` for the tutorial sample pages, failing every web_scrape row.
"""

import gzip
from unittest.mock import MagicMock

import httpx

from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient


def _make_client() -> AuditedHTTPClient:
    recorder = MagicMock()
    recorder.allocate_call_index.return_value = 0
    return AuditedHTTPClient(
        execution=recorder,
        state_id="state-1",
        run_id="run-1",
        telemetry_emit=lambda event: None,
        timeout=5.0,
        max_response_body_bytes=10 * 1024 * 1024,
    )


def test_capped_response_decodes_gzip_once_not_double() -> None:
    body = b"<html><body>" + b"hello world " * 200 + b"</body></html>"
    gz = gzip.compress(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-encoding": "gzip",
                "content-type": "text/html; charset=utf-8",
            },
            content=gz,
        )

    client = _make_client()
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as hc, hc.stream("GET", "https://example.test/page") as streaming:
        result = client._consume_capped_response(streaming, full_url="https://example.test/page")

    assert result.status_code == 200
    # Content-Encoding must be stripped so .text/.content do not re-decode the
    # already-decoded body.
    assert "content-encoding" not in result.headers
    assert result.text == body.decode()
    assert result.content == body
