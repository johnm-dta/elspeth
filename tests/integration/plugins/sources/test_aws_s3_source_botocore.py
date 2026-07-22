"""Offline real-botocore integration tests for the AWS S3 source."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from elspeth.plugins.aws_s3_common import build_s3_client
from elspeth.plugins.sources.aws_s3_source import AWSS3Source


@dataclass
class _Context:
    calls: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def record_validation_error(self, **kwargs: Any) -> None:
        self.validation_errors.append(kwargs)


@pytest.mark.parametrize(
    "source_format,data,expected",
    [
        ("csv", b"id,name\n1,Ada\n", [{"id": "1", "name": "Ada"}]),
        ("json", b'[{"id":1,"name":"Ada"}]', [{"id": 1, "name": "Ada"}]),
        ("jsonl", b'{"id":1,"name":"Ada"}\n', [{"id": 1, "name": "Ada"}]),
    ],
)
def test_real_botocore_stubber_streaming_body_without_network(
    monkeypatch: pytest.MonkeyPatch,
    source_format: str,
    data: bytes,
    expected: list[dict[str, Any]],
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "dummy-session-token")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    client = build_s3_client("ap-southeast-2", None)
    stubber = Stubber(client)
    etag = '"offline-etag"'
    stubber.add_response(
        "head_object",
        {"ContentLength": len(data), "ETag": etag},
        {"Bucket": "offline-bucket", "Key": f"input.{source_format}"},
    )
    stubber.add_response(
        "get_object",
        {"ContentLength": len(data), "Body": StreamingBody(io.BytesIO(data), len(data))},
        {"Bucket": "offline-bucket", "Key": f"input.{source_format}", "IfMatch": etag},
    )
    source = AWSS3Source(
        {
            "bucket": "offline-bucket",
            "key": f"input.{source_format}",
            "format": source_format,
            "schema": {"mode": "observed"},
            "on_validation_failure": "quarantine",
        }
    )
    source._s3_client = client
    context = _Context()
    with stubber:
        rows = list(source.load(context))
    stubber.assert_no_pending_responses()
    assert [row.row for row in rows] == expected
    assert context.calls[0]["response_data"] == {
        "size_bytes": len(data),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }
    source.close()
