"""Shared lazy AWS S3 client construction for optional AWS plugins."""

from __future__ import annotations

from typing import Any


def build_s3_client(region_name: str | None, endpoint_url: str | None) -> Any:
    """Build an S3 client with bounded SDK timeouts and retry attempts."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise ImportError('boto3 is required for aws_s3 plugins; install Elspeth with the "aws" extra') from exc

    config = Config(
        connect_timeout=10,
        read_timeout=30,
        retries={"mode": "standard", "total_max_attempts": 3},
    )
    return boto3.client(
        "s3",
        region_name=region_name,
        endpoint_url=endpoint_url,
        config=config,
    )
