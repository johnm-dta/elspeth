"""Tests for web-authored provider configuration policy helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from elspeth.web.provider_config_policy import (
    AWS_S3_ENDPOINT_URL_POLICY_ERROR,
    web_aws_s3_endpoint_url_policy_error,
)


class TestWebAwsS3EndpointUrlPolicy:
    @pytest.mark.parametrize(
        ("plugin", "options"),
        [
            ("csv", {"endpoint_url": "http://127.0.0.1:9000"}),
            (None, {"endpoint_url": "http://127.0.0.1:9000"}),
            ("aws_s3", {}),
            ("aws_s3", {"endpoint_url": None}),
        ],
    )
    def test_allows_non_aws_or_absent_endpoint(
        self,
        plugin: str | None,
        options: dict[str, Any],
    ) -> None:
        assert web_aws_s3_endpoint_url_policy_error(plugin, options) is None

    @pytest.mark.parametrize(
        "endpoint_url",
        [
            "http://127.0.0.1:9000",
            "https://storage.attacker.invalid",
            17,
        ],
    )
    def test_rejects_every_non_null_endpoint_without_echoing_it(self, endpoint_url: object) -> None:
        error = web_aws_s3_endpoint_url_policy_error("aws_s3", {"endpoint_url": endpoint_url})

        assert error == AWS_S3_ENDPOINT_URL_POLICY_ERROR
        assert str(endpoint_url) not in error


def test_core_config_has_no_web_runtime_import() -> None:
    repo_root = Path(__file__).parents[3]
    tree = ast.parse((repo_root / "src/elspeth/core/config.py").read_text(encoding="utf-8"))

    imported_modules = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported_modules.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None)

    assert not any(module == "elspeth.web" or module.startswith("elspeth.web.") for module in imported_modules)
