"""Tests for the shared, lazy AWS S3 client builder."""

from __future__ import annotations

import builtins
import sys
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pytest


class _FakeConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.connect_timeout = kwargs["connect_timeout"]
        self.read_timeout = kwargs["read_timeout"]
        self.retries = kwargs["retries"]


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Mock:
    boto3 = ModuleType("boto3")
    boto3.client = Mock(return_value=object())  # type: ignore[attr-defined]
    botocore = ModuleType("botocore")
    config = ModuleType("botocore.config")
    config.Config = _FakeConfig  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", config)
    return boto3.client  # type: ignore[attr-defined,no-any-return]


def test_build_s3_client_passes_region_endpoint_and_exact_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.plugins.aws_s3_common import build_s3_client

    client = _install_fake_sdk(monkeypatch)

    result = build_s3_client("ap-southeast-2", "http://localhost:4566")

    assert result is client.return_value
    client.assert_called_once()
    args, kwargs = client.call_args
    assert args == ("s3",)
    assert kwargs["region_name"] == "ap-southeast-2"
    assert kwargs["endpoint_url"] == "http://localhost:4566"
    config = kwargs["config"]
    assert config.connect_timeout == 10
    assert config.read_timeout == 30
    assert config.retries == {"mode": "standard", "total_max_attempts": 3}


def test_none_args_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.plugins.aws_s3_common import build_s3_client

    client = _install_fake_sdk(monkeypatch)

    build_s3_client(None, None)

    assert client.call_args.kwargs["region_name"] is None
    assert client.call_args.kwargs["endpoint_url"] is None


def test_no_credential_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.plugins.aws_s3_common import build_s3_client

    client = _install_fake_sdk(monkeypatch)

    build_s3_client("ap-southeast-2", None)

    assert set(client.call_args.kwargs) == {"region_name", "endpoint_url", "config"}


def test_missing_sdk_error_names_the_aws_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.plugins.aws_s3_common import build_s3_client

    real_import = builtins.__import__

    def _missing_sdk(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "boto3" or name.startswith("botocore"):
            raise ImportError("raw provider import detail")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_sdk)

    with pytest.raises(ImportError, match='install Elspeth with the "aws" extra'):
        build_s3_client(None, None)
