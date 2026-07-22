"""CLI/batch retains trusted aws_s3 endpoint overrides and lazy construction."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from elspeth.core.config import load_settings_from_yaml_string
from elspeth.plugins.infrastructure.runtime_factory import instantiate_plugins_from_config
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
from elspeth.plugins.sources.aws_s3_source import AWSS3Source


def test_cli_accepts_aws_s3_source_endpoint_url_and_instantiates_lazily(tmp_path: Path) -> None:
    yaml_text = textwrap.dedent(
        f"""
        landscape:
          url: "sqlite:///audit.db"
        sources:
          primary:
            plugin: aws_s3
            on_success: output
            options:
              bucket: input-bucket
              key: incoming/data.csv
              endpoint_url: http://localhost:4566
              on_validation_failure: discard
              schema:
                mode: observed
        sinks:
          output:
            plugin: csv
            on_write_failure: discard
            options:
              path: "{tmp_path / "output.csv"}"
              schema:
                mode: observed
        """
    ).strip()
    settings = load_settings_from_yaml_string(yaml_text)
    with patch("elspeth.plugins.sources.aws_s3_source.build_s3_client") as builder:
        bundle = instantiate_plugins_from_config(settings)
    assert isinstance(bundle.sources["primary"], AWSS3Source)
    builder.assert_not_called()


def test_cli_accepts_aws_s3_sink_endpoint_url_and_instantiates_lazily(tmp_path: Path) -> None:
    source_path = tmp_path / "input.csv"
    source_path.write_text("id,name\n1,Ada\n", encoding="utf-8")
    yaml_text = textwrap.dedent(
        f"""
        landscape:
          url: "sqlite:///audit.db"
        sources:
          primary:
            plugin: csv
            on_success: output
            options:
              path: "{source_path}"
              on_validation_failure: discard
              schema:
                mode: observed
        sinks:
          output:
            plugin: aws_s3
            on_write_failure: discard
            options:
              bucket: output-bucket
              key: outgoing/data.csv
              endpoint_url: http://localhost:4566
              schema:
                mode: observed
        """
    ).strip()
    settings = load_settings_from_yaml_string(yaml_text)
    with patch("elspeth.plugins.sinks.aws_s3_sink.build_s3_client") as sink_builder:
        bundle = instantiate_plugins_from_config(settings)
    assert isinstance(bundle.sinks["output"], AWSS3Sink)
    assert bundle.sinks["output"]._endpoint_url == "http://localhost:4566"
    sink_builder.assert_not_called()
