"""Explicit real-AWS acceptance for conditional S3 sink writes."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from elspeth.plugins.aws_s3_common import build_s3_client
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink, S3ConditionalWriteRejectedError
from tests.fixtures.base_classes import inject_write_failure

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@dataclass
class _Context:
    run_id: str = "live-run"
    contract: Any = None
    landscape: Any = None
    operation_id: str = "live-operation"
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
def test_real_s3_default_chain_conditional_write_and_stale_etag_non_clobber(format: str) -> None:
    bucket = os.environ.get("ELSPETH_TEST_S3_BUCKET")
    if not bucket:
        pytest.fail("ELSPETH_TEST_S3_BUCKET is required when the real AWS S3 acceptance is selected")
    key = f"elspeth-plan07/{uuid.uuid4()}/output.{format}"
    config = {
        "bucket": bucket,
        "key": key,
        "format": format,
        "overwrite": False,
        "schema": {"mode": "observed"},
    }
    client = build_s3_client(None, None)
    primary = inject_write_failure(AWSS3Sink(config))
    fresh = inject_write_failure(AWSS3Sink(config))
    try:
        first = primary.write([{"id": 1, "name": "Ada"}], _Context())
        assert first.artifact.content_hash
        with pytest.raises(S3ConditionalWriteRejectedError):
            fresh.write([{"id": 9, "name": "collision"}], _Context())
        second = primary.write([{"id": 2, "name": "Grace"}], _Context())
        assert second.artifact.content_hash != first.artifact.content_hash
        external = b"external-writer-sentinel"
        client.put_object(Bucket=bucket, Key=key, Body=external)
        with pytest.raises(S3ConditionalWriteRejectedError):
            primary.write([{"id": 3, "name": "stale"}], _Context())
        assert client.get_object(Bucket=bucket, Key=key)["Body"].read() == external
    finally:
        primary.close()
        fresh.close()
        client.delete_object(Bucket=bucket, Key=key)
        client.close()
