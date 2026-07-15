"""Offline real-botocore integration for conditional S3 sink writes."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from botocore.stub import ANY, Stubber

from elspeth.plugins.aws_s3_common import build_s3_client
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink, S3ConditionalWriteRejectedError
from tests.fixtures.base_classes import inject_write_failure


@dataclass
class _Context:
    run_id: str = "offline-run"
    contract: Any = None
    landscape: Any = None
    operation_id: str = "offline-operation"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _checksum(data: bytes) -> str:
    return base64.b64encode(hashlib.sha256(data).digest()).decode("ascii")


@pytest.mark.parametrize(
    ("format", "first_body", "second_body"),
    [
        ("csv", b"id,name\r\n1,Ada\r\n", b"id,name\r\n1,Ada\r\n2,Grace\r\n"),
        ("json", b'[{"id":1,"name":"Ada"}]', b'[{"id":1,"name":"Ada"},{"id":2,"name":"Grace"}]'),
        ("jsonl", b'{"id":1,"name":"Ada"}\n', b'{"id":1,"name":"Ada"}\n{"id":2,"name":"Grace"}\n'),
    ],
)
def test_real_botocore_stubber_conditional_cumulative_writes_without_network(
    monkeypatch: pytest.MonkeyPatch,
    format: str,
    first_body: bytes,
    second_body: bytes,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret-key")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "dummy-session-token")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    client = build_s3_client("ap-southeast-2", None)
    stubber = Stubber(client)
    common = {"Bucket": "offline-bucket", "Key": f"output.{format}", "Body": ANY}
    stubber.add_response(
        "put_object",
        {"ETag": '"etag-1"'},
        {
            **common,
            "ContentLength": len(first_body),
            "ChecksumSHA256": _checksum(first_body),
            "IfNoneMatch": "*",
        },
    )
    stubber.add_response(
        "put_object",
        {"ETag": '"etag-2"'},
        {
            **common,
            "ContentLength": len(second_body),
            "ChecksumSHA256": _checksum(second_body),
            "IfMatch": '"etag-1"',
        },
    )
    sink = inject_write_failure(
        AWSS3Sink(
            {
                "bucket": "offline-bucket",
                "key": f"output.{format}",
                "format": format,
                "overwrite": False,
                "schema": {"mode": "observed"},
            }
        )
    )
    sink._s3_client = client
    with stubber:
        sink.write([{"id": 1, "name": "Ada"}], _Context())
        sink.write([{"id": 2, "name": "Grace"}], _Context())
    stubber.assert_no_pending_responses()
    sink.close()


@pytest.mark.parametrize(
    ("code", "status"),
    [("PreconditionFailed", 412), ("ConditionalRequestConflict", 409)],
)
def test_real_botocore_modeled_condition_failure_has_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    status: int,
) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy-secret-key")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    client = build_s3_client("ap-southeast-2", None)
    body = b'[{"id":1,"name":"Ada"}]'
    expected = {
        "Bucket": "offline-bucket",
        "Key": "output.json",
        "Body": ANY,
        "ContentLength": len(body),
        "ChecksumSHA256": _checksum(body),
        "IfNoneMatch": "*",
    }
    stubber = Stubber(client)
    stubber.add_client_error(
        "put_object",
        service_error_code=code,
        service_message="raw provider sentinel",
        http_status_code=status,
        expected_params=expected,
    )
    sink = inject_write_failure(
        AWSS3Sink(
            {
                "bucket": "offline-bucket",
                "key": "output.json",
                "format": "json",
                "overwrite": False,
                "schema": {"mode": "observed"},
            }
        )
    )
    sink._s3_client = client
    with stubber, pytest.raises(S3ConditionalWriteRejectedError) as captured:
        sink.write([{"id": 1, "name": "Ada"}], _Context())
    assert captured.value.__cause__ is None and captured.value.__context__ is None
    stubber.assert_no_pending_responses()
    sink.close()
