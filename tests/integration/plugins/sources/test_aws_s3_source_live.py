"""Explicit slow acceptance against a disposable object in a real AWS bucket."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from elspeth.plugins.aws_s3_common import build_s3_client
from elspeth.plugins.sources.aws_s3_source import AWSS3Source

pytestmark = pytest.mark.slow


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
        ("csv", b"id,name\n1,Ada\n", {"id": "1", "name": "Ada"}),
        ("json", b'[{"id":1,"name":"Ada"}]', {"id": 1, "name": "Ada"}),
        ("jsonl", b'{"id":1,"name":"Ada"}\n', {"id": 1, "name": "Ada"}),
    ],
)
def test_real_s3_default_chain_round_trip(source_format: str, data: bytes, expected: dict[str, Any]) -> None:
    bucket = os.environ.get("ELSPETH_TEST_S3_BUCKET")
    if not bucket:
        pytest.fail("ELSPETH_TEST_S3_BUCKET is required when the real AWS S3 acceptance is selected")
    key = f"elspeth-plan06/{uuid.uuid4()}/input.{source_format}"
    client = build_s3_client(None, None)
    client.put_object(Bucket=bucket, Key=key, Body=data)
    try:
        source = AWSS3Source(
            {
                "bucket": bucket,
                "key": key,
                "format": source_format,
                "schema": {"mode": "observed"},
                "on_validation_failure": "quarantine",
            }
        )
        context = _Context()
        rows = list(source.load(context))
        assert [row.row for row in rows] == [expected]
        assert context.calls and not context.validation_errors
        source.close()
    finally:
        client.delete_object(Bucket=bucket, Key=key)
        client.close()
