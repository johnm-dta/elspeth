"""Regression: a capped streaming response must not double-decompress gzip bodies.

``_consume_capped_response`` reads the body via ``iter_bytes()`` (which has ALREADY
applied content-decoding) but used to reconstruct the ``httpx.Response`` with the
original ``Content-Encoding`` header. On read, httpx then re-ran the gzip decoder on
the already-decoded HTML -> ``httpx.DecodingError`` ("Error -3 while decompressing
data: incorrect header check"). This surfaced live when GitHub Pages began serving
``content-encoding: gzip`` for the tutorial sample pages, failing every web_scrape row.

The companion ``test_capped_response_measures_decoded_size_not_compressed`` guards the
*decoded-size* body cap (the decompression-bomb-relevant metric): ``iter_bytes()`` is
kept deliberately so the cap counts decoded bytes, not compressed wire bytes.
"""

import gzip
import zlib
from unittest.mock import MagicMock

import httpx
import pytest

from elspeth.plugins.infrastructure.clients.http import (
    AuditedHTTPClient,
    HTTPResponseBodyTooLargeError,
)


def _make_client(*, max_response_body_bytes: int = 10 * 1024 * 1024) -> AuditedHTTPClient:
    recorder = MagicMock()
    recorder.allocate_call_index.return_value = 0
    return AuditedHTTPClient(
        execution=recorder,
        state_id="state-1",
        run_id="run-1",
        telemetry_emit=lambda event: None,
        timeout=5.0,
        max_response_body_bytes=max_response_body_bytes,
    )


@pytest.mark.parametrize(
    ("encoding", "compress"),
    [
        ("gzip", gzip.compress),
        ("deflate", zlib.compress),
    ],
)
def test_capped_response_decodes_once_not_double(encoding: str, compress) -> None:
    body = b"<html><body>" + b"hello world " * 200 + b"</body></html>"
    encoded = compress(body)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-encoding": encoding,
                "content-type": "text/html; charset=utf-8",
            },
            content=encoded,
        )

    client = _make_client()
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as hc, hc.stream("GET", "https://example.test/page") as streaming:
        result = client._consume_capped_response(streaming, full_url="https://example.test/page")

    assert result.status_code == 200
    # Content-Encoding must be stripped so .text/.content do not re-decode the
    # already-decoded body.
    assert "content-encoding" not in result.headers
    # The stale *compressed* content-length is dropped; httpx recomputes it from
    # the reconstructed (decoded) content, so it now equals the decoded length
    # rather than the smaller compressed wire size.
    assert result.headers["content-length"] == str(len(body))
    assert len(encoded) < len(body)  # confirms the recomputed length is decoded, not wire
    assert result.text == body.decode()
    assert result.content == body


def test_capped_response_identity_success_path_preserved() -> None:
    """The header-stripping reconstruction must not corrupt the common
    uncompressed (no content-encoding) success path."""
    body = b"<html><body>plain identity body</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=body,
        )

    client = _make_client()
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as hc, hc.stream("GET", "https://example.test/page") as streaming:
        result = client._consume_capped_response(streaming, full_url="https://example.test/page")

    assert result.status_code == 200
    assert result.content == body
    assert result.text == body.decode()


def test_capped_response_measures_decoded_size_not_compressed() -> None:
    """The body cap is the decompression-bomb-relevant metric: it must count the
    DECODED size, not the compressed wire size.

    Sizes are chosen so that ``len(gz) < cap < decoded_size``. A small gzip
    payload that inflates past the cap must trip ``HTTPResponseBodyTooLargeError``
    with ``body_size`` reflecting decoded bytes. If the cap measured compressed
    bytes instead, the ~260-byte payload would slip under the 4096-byte cap and
    the error would never fire.
    """
    cap = 4096
    decoded = b"A" * (256 * 1024)  # 262144 decoded bytes
    gz = gzip.compress(decoded)
    assert len(gz) < cap < len(decoded)  # discriminator precondition

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-encoding": "gzip",
                "content-type": "text/html; charset=utf-8",
            },
            content=gz,
        )

    client = _make_client(max_response_body_bytes=cap)
    transport = httpx.MockTransport(handler)
    with (
        httpx.Client(transport=transport) as hc,
        hc.stream("GET", "https://example.test/page") as streaming,
        pytest.raises(HTTPResponseBodyTooLargeError) as exc_info,
    ):
        client._consume_capped_response(streaming, full_url="https://example.test/page")

    exc = exc_info.value
    assert exc.body_size > cap  # cap tripped on decoded bytes
    # Load-bearing: body_size exceeding the ENTIRE compressed payload proves the
    # cap measured decoded, not compressed, bytes.
    assert exc.body_size > len(gz)
    assert exc.max_body_bytes == cap
    truncated = exc.response_data["body"]
    assert truncated["_truncated"] is True
    assert truncated["_reason"] == "body_too_large"
    assert truncated["_observed_body_size"] == exc.body_size
