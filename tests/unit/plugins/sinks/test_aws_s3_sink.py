"""Unit tests for the bounded AWS S3 sink primitives and runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError

DYNAMIC_SCHEMA: dict[str, Any] = {"mode": "observed"}


def _base_config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bucket": "example-bucket",
        "key": "runs/{{ run_id }}/output.csv",
        "schema": DYNAMIC_SCHEMA,
    }
    config.update(overrides)
    return config


class TestAWSS3SinkConfig:
    def test_task_one_module_is_not_registered(self) -> None:
        import elspeth.plugins.sinks.aws_s3_sink as module

        assert not hasattr(module, "AWSS3Sink")

    def test_all_registered_fields_have_descriptions(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig, CSVWriteOptions

        expected = {
            "bucket",
            "key",
            "format",
            "overwrite",
            "csv_options",
            "headers",
            "region_name",
            "endpoint_url",
            "max_object_bytes",
            "max_record_chars",
        }
        assert expected <= AWSS3SinkConfig.model_fields.keys()
        assert all(AWSS3SinkConfig.model_fields[name].description for name in expected)
        assert all(CSVWriteOptions.model_fields[name].description for name in ("delimiter", "encoding", "include_header"))
        assert AWSS3SinkConfig._plugin_component_type == "sink"

    @pytest.mark.parametrize(
        "field",
        [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "access_key",
            "secret_key",
            "session_token",
            "credentials",
            "client_config",
            "client_kwargs",
        ],
    )
    def test_credential_and_client_fields_are_forbidden(self, field: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        assert field not in AWSS3SinkConfig.model_fields
        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(**{field: "sentinel"}), plugin_name="aws_s3")

    @pytest.mark.parametrize("field", ["bucket", "key"])
    @pytest.mark.parametrize("value", ["", "   ", "<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_blank_and_operator_placeholder_locations_are_rejected(self, field: str, value: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(**{field: value}), plugin_name="aws_s3")

    @pytest.mark.parametrize(
        ("field", "accepted", "rejected"),
        [
            ("bucket", "b" * 2048, "b" * 2049),
            ("key", "k" * 4096, "k" * 4097),
            ("region_name", "r" * 64, "r" * 65),
            ("max_object_bytes", 1024 * 1024 * 1024, 1024 * 1024 * 1024 + 1),
            ("max_record_chars", 8_000_000, 8_000_001),
        ],
    )
    def test_maximum_boundaries(self, field: str, accepted: Any, rejected: Any) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        cfg = AWSS3SinkConfig.from_dict(_base_config(**{field: accepted}), plugin_name="aws_s3")
        assert getattr(cfg, field) == accepted
        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(**{field: rejected}), plugin_name="aws_s3")

    @pytest.mark.parametrize("field", ["max_object_bytes", "max_record_chars"])
    def test_positive_size_boundaries(self, field: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        assert getattr(AWSS3SinkConfig.from_dict(_base_config(**{field: 1})), field) == 1
        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(**{field: 0}))

    @pytest.mark.parametrize("value", ["bad region", "us_east_1", ""])
    def test_invalid_region_is_rejected(self, value: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(region_name=value))

    @pytest.mark.parametrize(
        "value",
        [
            "ftp://localhost",
            "http://",
            "http://user:pass@localhost",
            "http://localhost?q=sentinel",
            "http://localhost#sentinel",
            "http://local host",
            "http://localhost/\x00",
            "x" * 2049,
        ],
    )
    def test_invalid_endpoint_is_rejected(self, value: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(endpoint_url=value))

    def test_explicit_null_endpoint_is_accepted(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        assert AWSS3SinkConfig.from_dict(_base_config(endpoint_url=None)).endpoint_url is None

    @pytest.mark.parametrize("key", ["bad\x00key", "bad\nkey", "{{ unclosed"])
    def test_invalid_key_template_is_rejected(self, key: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(key=key))

    def test_undefined_template_variable_fails_at_render(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig, _render_key_template

        cfg = AWSS3SinkConfig.from_dict(_base_config(key="{{ missing }}"))
        with pytest.raises(ValueError, match="render"):
            _render_key_template(cfg.key, run_id="run-1", timestamp="2026-07-14T00:00:00+00:00")

    @pytest.mark.parametrize("rendered", ["", " \t", "bad\nkey", "k" * 1025])
    def test_rendered_key_is_revalidated(self, rendered: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import _render_key_template

        with pytest.raises(ValueError, match="rendered key"):
            _render_key_template("{{ run_id }}", run_id=rendered, timestamp="2026-07-14T00:00:00+00:00")

    def test_csv_options_and_headers_match_existing_sink_contract(self) -> None:
        from elspeth.contracts.header_modes import HeaderMode
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        cfg = AWSS3SinkConfig.from_dict(
            _base_config(csv_options={"delimiter": ";", "encoding": "utf-16", "include_header": False}, headers={"a": "A"})
        )
        assert cfg.csv_options.delimiter == ";"
        assert cfg.csv_options.include_header is False
        assert cfg.headers_mode is HeaderMode.CUSTOM
        assert cfg.headers_mapping == {"a": "A"}
        with pytest.raises(PluginConfigError):
            AWSS3SinkConfig.from_dict(_base_config(csv_options={"unknown": True}))


def _serialize(
    rows: list[dict[str, Any]],
    *,
    format: str,
    fieldnames: list[str] | None = None,
    max_object_bytes: int = 1024 * 1024,
    max_record_chars: int = 100_000,
    **csv_overrides: Any,
) -> Any:
    from elspeth.plugins.sinks.aws_s3_sink import CSVWriteOptions, _serialize_rows_to_spool

    return _serialize_rows_to_spool(
        rows,
        format=format,
        csv_options=CSVWriteOptions(**csv_overrides),
        fieldnames=fieldnames or ["id", "name"],
        max_object_bytes=max_object_bytes,
        max_record_chars=max_record_chars,
    )


class TestSerialization:
    @pytest.mark.parametrize(
        ("format", "expected"),
        [
            ("csv", b"id,name\r\n1,Ada\r\n2,Grace\r\n"),
            ("json", b'[{"id":1,"name":"Ada"},{"id":2,"name":"Grace"}]'),
            ("jsonl", b'{"id":1,"name":"Ada"}\n{"id":2,"name":"Grace"}\n'),
        ],
    )
    def test_shapes_hashes_rewind_and_idempotent_close(self, format: str, expected: bytes) -> None:
        serialized = _serialize([{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}], format=format)
        assert serialized.body.tell() == 0
        assert serialized.body.read() == expected
        assert serialized.size_bytes == len(expected)
        digest = hashlib.sha256(expected).digest()
        assert serialized.content_hash == digest.hex()
        assert serialized.checksum_sha256_b64 == base64.b64encode(digest).decode("ascii")
        serialized.close()
        serialized.close()
        assert serialized.body.closed

    @pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
    def test_exact_object_limit_max_and_max_plus_one(self, format: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3ObjectSizeLimitError

        probe = _serialize([{"id": 1, "name": "Ada"}], format=format)
        size = probe.size_bytes
        probe.close()
        exact = _serialize([{"id": 1, "name": "Ada"}], format=format, max_object_bytes=size)
        exact.close()
        with pytest.raises(S3ObjectSizeLimitError) as captured:
            _serialize([{"id": 1, "name": "Ada"}], format=format, max_object_bytes=size - 1)
        assert captured.value.observed_bytes > captured.value.limit_bytes
        assert "Ada" not in str(captured.value)

    @pytest.mark.parametrize("format", ["json", "jsonl"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
    def test_nonfinite_and_nonserializable_values_are_static_failures(self, format: str, value: object) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSerializationError

        with pytest.raises(S3RecordSerializationError) as captured:
            _serialize([{"id": 1, "name": value}], format=format)
        assert "object at" not in str(captured.value)

    def test_csv_unencodable_value_is_static_failure(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSerializationError

        with pytest.raises(S3RecordSerializationError):
            _serialize([{"id": 1, "name": "snowman ☃"}], format="csv", encoding="ascii")

    @pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
    def test_record_limit_rejects_huge_single_value_before_writing(self, format: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSizeLimitError

        with pytest.raises(S3RecordSizeLimitError):
            _serialize([{"id": 1, "name": "x" * 101}], format=format, max_record_chars=100)

    def test_many_small_rows_can_exceed_cumulative_object_limit(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3ObjectSizeLimitError

        rows = [{"id": index, "name": "x" * 10} for index in range(100)]
        with pytest.raises(S3ObjectSizeLimitError):
            _serialize(rows, format="jsonl", max_object_bytes=200)

    def test_spool_rolls_over_at_eight_mib(self) -> None:
        serialized = _serialize(
            [{"id": index, "name": "x" * 70_000} for index in range(125)],
            format="jsonl",
            max_object_bytes=20 * 1024 * 1024,
            max_record_chars=100_000,
        )
        assert serialized.size_bytes > 8 * 1024 * 1024
        assert serialized.body._rolled is True  # type: ignore[attr-defined]
        serialized.close()

    def test_context_manager_owns_and_closes_spool(self) -> None:
        with _serialize([{"id": 1, "name": "Ada"}], format="json") as serialized:
            body = serialized.body
            assert not body.closed
        assert body.closed

    def test_spool_is_closed_when_size_failure_occurs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import elspeth.plugins.sinks.aws_s3_sink as module

        real_factory = module.tempfile.SpooledTemporaryFile
        created: list[Any] = []

        def tracking_factory(*args: Any, **kwargs: Any) -> Any:
            spool = real_factory(*args, **kwargs)
            created.append(spool)
            return spool

        monkeypatch.setattr(module.tempfile, "SpooledTemporaryFile", tracking_factory)
        with pytest.raises(module.S3ObjectSizeLimitError):
            _serialize([{"id": 1, "name": "Ada"}], format="json", max_object_bytes=1)
        assert len(created) == 1
        assert created[0].closed

    def test_no_spool_write_exceeds_64_kib(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import elspeth.plugins.sinks.aws_s3_sink as module

        real_factory = module.tempfile.SpooledTemporaryFile
        writes: list[int] = []

        class TrackingSpool:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._spool = real_factory(*args, **kwargs)

            def write(self, data: bytes) -> int:
                writes.append(len(data))
                return self._spool.write(data)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._spool, name)

        monkeypatch.setattr(module.tempfile, "SpooledTemporaryFile", TrackingSpool)
        serialized = _serialize([{"id": 1, "name": "x" * 200_000}], format="json", max_record_chars=300_000)
        serialized.close()
        assert writes
        assert max(writes) <= 64 * 1024

    def test_json_output_is_valid_without_whole_document_formatting(self) -> None:
        with _serialize([{"id": 1, "name": "Ada"}], format="json") as serialized:
            assert json.load(serialized.body) == [{"id": 1, "name": "Ada"}]
