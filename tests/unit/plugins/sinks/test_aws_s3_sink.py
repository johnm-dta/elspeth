"""Unit tests for the bounded AWS S3 sink primitives and runtime."""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
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
    def test_complete_sink_is_registered_in_task_two(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink

        assert AWSS3Sink.name == "aws_s3"

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

    def test_undefined_template_variable_fails_at_config(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError, match="approved variables"):
            AWSS3SinkConfig.from_dict(_base_config(key="{{ missing }}"))

    @pytest.mark.parametrize("rendered", ["", " \t", "bad\nkey", "k" * 1025])
    def test_rendered_key_is_revalidated(self, rendered: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import _render_key_template

        with pytest.raises(ValueError, match="rendered key"):
            _render_key_template("{{ run_id }}", run_id=rendered, timestamp="2026-07-14T00:00:00+00:00")

    def test_key_template_rejects_expressions_before_rendering(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import _render_key_template

        class MultiplicationSentinel(str):
            multiplied = False

            def __mul__(self, count: int) -> str:
                self.multiplied = True
                raise AssertionError(f"template attempted multiplication by {count}")

        run_id = MultiplicationSentinel("run-1")
        with pytest.raises(ValueError, match="template"):
            _render_key_template("{{ run_id * 1000000000 }}", run_id=run_id, timestamp="2026-07-14T00:00:00+00:00")

        assert run_id.multiplied is False

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

    @pytest.mark.parametrize("encoding", ["rot_13", "base64_codec"])
    def test_non_text_to_bytes_csv_codec_is_rejected_at_config(self, encoding: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3SinkConfig

        with pytest.raises(PluginConfigError, match="text to bytes"):
            AWSS3SinkConfig.from_dict(_base_config(csv_options={"encoding": encoding}), plugin_name="aws_s3")


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
        above = _serialize([{"id": 1, "name": "Ada"}], format=format, max_object_bytes=size + 1)
        above.close()
        with pytest.raises(S3ObjectSizeLimitError) as captured:
            _serialize([{"id": 1, "name": "Ada"}], format=format, max_object_bytes=size - 1)
        assert captured.value.observed_bytes > captured.value.limit_bytes
        assert "Ada" not in str(captured.value)

    @pytest.mark.parametrize(("value_size", "accepted"), [(91, True), (92, True), (93, False)])
    def test_json_record_limit_max_minus_one_max_and_max_plus_one(self, value_size: int, accepted: bool) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSizeLimitError

        row = {"v": "x" * value_size}  # compact JSON record length is value_size + 8
        if accepted:
            serialized = _serialize([row], format="json", fieldnames=["v"], max_record_chars=100)
            serialized.close()
        else:
            with pytest.raises(S3RecordSizeLimitError):
                _serialize([row], format="json", fieldnames=["v"], max_record_chars=100)

    @pytest.mark.parametrize(("value_size", "accepted"), [(97, True), (98, True), (99, False)])
    def test_csv_record_limit_max_minus_one_max_and_max_plus_one(self, value_size: int, accepted: bool) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSizeLimitError

        row = {"v": "x" * value_size}  # one value plus CRLF is value_size + 2
        if accepted:
            serialized = _serialize(
                [row],
                format="csv",
                fieldnames=["v"],
                max_record_chars=100,
                include_header=False,
            )
            serialized.close()
        else:
            with pytest.raises(S3RecordSizeLimitError):
                _serialize(
                    [row],
                    format="csv",
                    fieldnames=["v"],
                    max_record_chars=100,
                    include_header=False,
                )

    @pytest.mark.parametrize("format", ["json", "jsonl"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
    def test_nonfinite_and_nonserializable_values_are_static_failures(self, format: str, value: object) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSerializationError

        with pytest.raises(S3RecordSerializationError) as captured:
            _serialize([{"id": 1, "name": value}], format=format)
        assert "object at" not in str(captured.value)

    def test_csv_unencodable_value_is_static_failure(self) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSerializationError

        with pytest.raises(S3RecordSerializationError) as captured:
            _serialize([{"id": 1, "name": "snowman ☃"}], format="csv", encoding="ascii")
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

    def test_stateful_csv_encoding_uses_one_incremental_encoder_and_one_bom(self) -> None:
        serialized = _serialize(
            [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}],
            format="csv",
            encoding="utf-16",
        )
        payload = serialized.body.read()
        serialized.close()
        assert payload.count(b"\xff\xfe") + payload.count(b"\xfe\xff") == 1
        assert payload.decode("utf-16") == "id,name\r\n1,Ada\r\n2,Grace\r\n"

    @pytest.mark.parametrize("encoding", ["rot_13", "base64_codec"])
    def test_non_text_codec_runtime_failure_is_static_and_cause_free(self, encoding: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import (
            CSVWriteOptions,
            S3RecordSerializationError,
            _serialize_rows_to_spool,
        )

        options = CSVWriteOptions.model_construct(delimiter=",", encoding=encoding, include_header=True)
        with pytest.raises(S3RecordSerializationError) as captured:
            _serialize_rows_to_spool(
                [{"id": 1}],
                format="csv",
                csv_options=options,
                fieldnames=["id"],
                max_object_bytes=1024,
                max_record_chars=100,
            )
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

    @pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
    def test_huge_integer_conversion_is_a_static_serialization_failure(self, format: str) -> None:
        from elspeth.plugins.sinks.aws_s3_sink import S3RecordSerializationError

        with pytest.raises(S3RecordSerializationError) as captured:
            _serialize([{"id": 10**5000, "name": "Ada"}], format=format)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

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


@dataclass
class _SinkContext:
    run_id: str = "run-123"
    contract: Any = None
    landscape: Any = None
    operation_id: str = "operation-123"
    calls: list[dict[str, Any]] = field(default_factory=list)
    recorder_error: BaseException | None = None

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        if self.recorder_error is not None:
            raise self.recorder_error


class _S3Client:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [{"ETag": '"etag-1"'}])
        self.requests: list[dict[str, Any]] = []
        self.bodies: list[bytes] = []
        self.closed = 0
        self.close_error: BaseException | None = None

    def put_object(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        body = kwargs["Body"]
        body.seek(0)
        self.bodies.append(body.read())
        body.seek(0)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _runtime_sink(*, client: _S3Client | None = None, **overrides: Any) -> tuple[Any, _S3Client]:
    from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
    from tests.fixtures.base_classes import inject_write_failure

    sink = inject_write_failure(AWSS3Sink(_base_config(**overrides)))
    actual_client = client or _S3Client()
    sink._s3_client = actual_client
    return sink, actual_client


class _ProviderFailure(Exception):
    pass


class _ConditionalFailure(Exception):
    def __init__(self, code: str, status: int) -> None:
        super().__init__("raw provider sentinel must not escape")
        self.response = {"Error": {"Code": code, "Message": "raw provider sentinel"}, "ResponseMetadata": {"HTTPStatusCode": status}}


class TestAWSS3SinkRuntime:
    def test_protocol_metadata_and_assistance(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink

        assert AWSS3Sink.name == "aws_s3"
        assert AWSS3Sink.determinism is Determinism.IO_WRITE
        assert AWSS3Sink.plugin_version == "1.0.0"
        assert AWSS3Sink.supports_resume is False
        assistance = AWSS3Sink.get_agent_assistance()
        assert assistance is not None and assistance.summary
        assert assistance.composer_hints
        assert all(len(hint) <= 280 for hint in assistance.composer_hints)

    def test_configure_for_resume_uses_direct_not_supported_error(self) -> None:
        sink, _ = _runtime_sink()
        with pytest.raises(NotImplementedError, match="does not support resume") as captured:
            sink.configure_for_resume()
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None

    @pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
    def test_unconditional_upload_has_integrity_metadata_and_audit(self, format: str) -> None:
        sink, client = _runtime_sink(format=format, key=f"output.{format}")
        context = _SinkContext()

        result = sink.write([{"id": 1, "name": "Ada"}], context)

        request = client.requests[0]
        assert request["Bucket"] == "example-bucket"
        assert request["Key"] == f"output.{format}"
        assert request["ContentLength"] == len(client.bodies[0])
        assert request["ChecksumSHA256"] == base64.b64encode(hashlib.sha256(client.bodies[0]).digest()).decode("ascii")
        assert "IfNoneMatch" not in request and "IfMatch" not in request
        assert result.artifact.content_hash == hashlib.sha256(client.bodies[0]).hexdigest()
        assert result.artifact.path_or_uri == f"s3://example-bucket/output.{format}"
        call = context.calls[0]
        assert call["status"].value == "success"
        assert call["provider"] == "aws_s3"
        assert call["request_data"] == {
            "operation": "put_object",
            "bucket": "example-bucket",
            "key": f"output.{format}",
            "overwrite": True,
            "condition": "none",
        }
        assert call["response_data"] == {"size_bytes": len(client.bodies[0]), "content_hash": result.artifact.content_hash}

    def test_if_none_match_then_confirmed_etag_if_match_for_cumulative_rewrite(self) -> None:
        client = _S3Client([{"ETag": '"etag-1"'}, {"ETag": '"etag-2"'}])
        sink, _ = _runtime_sink(client=client, overwrite=False, format="json")
        context = _SinkContext()

        first = sink.write([{"id": 1, "name": "Ada"}], context)
        second = sink.write([{"id": 2, "name": "Grace"}], context)

        assert client.requests[0]["IfNoneMatch"] == "*"
        assert "IfMatch" not in client.requests[0]
        assert client.requests[1]["IfMatch"] == '"etag-1"'
        assert "IfNoneMatch" not in client.requests[1]
        assert json.loads(client.bodies[0]) == [{"id": 1, "name": "Ada"}]
        assert json.loads(client.bodies[1]) == [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Grace"}]
        assert first.artifact.content_hash != second.artifact.content_hash
        assert sink._remote_etag == '"etag-2"'

    @pytest.mark.parametrize("response", [{}, {"ETag": ""}, {"ETag": "bad\nvalue"}, {"ETag": "x" * 1025}, []])
    def test_missing_or_malformed_success_etag_poison_sink(self, response: Any) -> None:
        sink, _ = _runtime_sink(client=_S3Client([response]), overwrite=False)
        context = _SinkContext()

        with pytest.raises(Exception, match="outcome is unknown") as captured:
            sink.write([{"id": 1, "name": "Ada"}], context)
        assert captured.value.__cause__ is None
        assert captured.value.__context__ is None
        assert sink._poisoned is True
        assert sink._buffered_rows == []
        with pytest.raises(Exception, match="poisoned"):
            sink.write([{"id": 2, "name": "Grace"}], context)

    @pytest.mark.parametrize(
        "failure",
        [
            _ConditionalFailure("PreconditionFailed", 412),
            _ConditionalFailure("ConditionalRequestConflict", 409),
        ],
    )
    def test_conditional_failure_is_static_and_never_falls_back(self, failure: BaseException) -> None:
        sink, client = _runtime_sink(client=_S3Client([failure]), overwrite=False)
        context = _SinkContext()

        with pytest.raises(Exception, match="conditional write was rejected") as captured:
            sink.write([{"id": 1, "name": "provider-value-sentinel"}], context)

        assert len(client.requests) == 1
        assert client.requests[0]["IfNoneMatch"] == "*"
        assert captured.value.__cause__ is None and captured.value.__context__ is None
        assert "provider-value-sentinel" not in str(captured.value)
        assert context.calls[0]["error"] == {"type": "_ConditionalFailure"}
        assert sink._poisoned is False

    @pytest.mark.parametrize("code", ["AccessDenied", "NoSuchBucket", "InvalidRequest"])
    def test_definite_request_rejection_is_static_without_poison(self, code: str) -> None:
        sink, _ = _runtime_sink(client=_S3Client([_ConditionalFailure(code, 400)]))
        with pytest.raises(Exception, match="S3 object write was rejected") as captured:
            sink.write([{"id": 1, "name": "Ada"}], _SinkContext())
        assert captured.value.__cause__ is None and captured.value.__context__ is None
        assert sink._poisoned is False

    def test_ambiguous_provider_failure_is_audited_then_poisoned(self) -> None:
        sink, client = _runtime_sink(client=_S3Client([_ProviderFailure("endpoint and credential sentinel")]))
        context = _SinkContext()
        with pytest.raises(Exception, match="outcome is unknown") as captured:
            sink.write([{"id": 1, "name": "row sentinel"}], context)
        assert captured.value.__cause__ is None and captured.value.__context__ is None
        assert "sentinel" not in str(captured.value)
        assert sink._poisoned is True
        assert sink._buffered_rows == []
        assert len(client.requests) == 1
        assert context.calls[0]["error"] == {"type": "_ProviderFailure"}

    @pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit()])
    def test_put_object_process_control_exceptions_are_not_converted(self, control: BaseException) -> None:
        sink, _ = _runtime_sink(client=_S3Client([control]))

        with pytest.raises(type(control)):
            sink.write([{"id": 1, "name": "Ada"}], _SinkContext())

    def test_put_object_import_error_is_ambiguous_audited_and_poisoned(self) -> None:
        sink, client = _runtime_sink(client=_S3Client([ImportError("post-dispatch provider sentinel")]))
        context = _SinkContext()
        with pytest.raises(Exception, match="outcome is unknown") as captured:
            sink.write([{"id": 1, "name": "Ada"}], context)
        assert len(client.requests) == 1
        assert context.calls[0]["error"] == {"type": "ImportError"}
        assert sink._poisoned is True
        assert captured.value.__cause__ is None and captured.value.__context__ is None

    def test_builder_import_error_preserves_optional_dependency_guidance(self) -> None:
        from unittest.mock import patch

        from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
        from tests.fixtures.base_classes import inject_write_failure

        expected = ImportError('boto3 is required; install Elspeth with the "aws" extra')
        sink = inject_write_failure(AWSS3Sink(_base_config()))
        context = _SinkContext()
        with (
            patch("elspeth.plugins.sinks.aws_s3_sink.build_s3_client", side_effect=expected),
            pytest.raises(ImportError) as captured,
        ):
            sink.write([{"id": 1, "name": "Ada"}], context)
        assert captured.value is expected
        assert context.calls == []
        assert sink._poisoned is False

    def test_client_creation_failure_redacts_endpoint_and_provider_details(self, caplog: pytest.LogCaptureFixture) -> None:
        from unittest.mock import patch

        from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
        from tests.fixtures.base_classes import inject_write_failure

        endpoint = "https://endpoint-sentinel.invalid"
        sink = inject_write_failure(AWSS3Sink(_base_config(endpoint_url=endpoint)))
        context = _SinkContext()
        with (
            patch(
                "elspeth.plugins.sinks.aws_s3_sink.build_s3_client",
                side_effect=_ProviderFailure("credential body endpoint provider sentinel"),
            ),
            pytest.raises(Exception, match="outcome is unknown") as captured,
        ):
            sink.write([{"id": 1, "name": "row sentinel"}], context)
        rendered = " ".join((str(captured.value), repr(captured.value), caplog.text, repr(context.calls)))
        assert "sentinel" not in rendered
        assert endpoint not in rendered
        assert captured.value.__cause__ is None and captured.value.__context__ is None

    @pytest.mark.parametrize("status", ["success", "failure"])
    def test_audit_failure_is_static_poison_and_never_commits_buffer(self, status: str) -> None:
        responses: list[Any] = [{"ETag": '"etag-1"'}] if status == "success" else [_ConditionalFailure("AccessDenied", 403)]
        sink, _ = _runtime_sink(client=_S3Client(responses), overwrite=False)
        context = _SinkContext(recorder_error=RuntimeError("recorder sentinel"))
        with pytest.raises(Exception, match="audit trail") as captured:
            sink.write([{"id": 1, "name": "row sentinel"}], context)
        assert captured.value.__cause__ is None and captured.value.__context__ is None
        assert "sentinel" not in str(captured.value)
        assert sink._poisoned is True
        assert sink._buffered_rows == []

    @pytest.mark.parametrize(
        ("format", "value", "reason"),
        [
            ("json", float("nan"), "JSON record could not be serialized safely"),
            ("jsonl", object(), "JSON record could not be serialized safely"),
            ("csv", "snowman ☃", "CSV record could not be encoded safely"),
            ("json", "x" * 101, "record exceeds configured character limit"),
        ],
    )
    def test_bad_incoming_row_diverts_with_static_reason(self, format: str, value: object, reason: str) -> None:
        overrides: dict[str, Any] = {"format": format, "max_record_chars": 100}
        if format == "csv":
            overrides["csv_options"] = {"encoding": "ascii"}
        sink, client = _runtime_sink(**overrides)
        result = sink.write([{"id": 1, "name": value}, {"id": 2, "name": "Grace"}], _SinkContext())
        assert len(result.diversions) == 1
        assert result.diversions[0].row_index == 0
        assert result.diversions[0].reason == reason
        assert "snowman" not in reason and "object" not in reason
        assert len(client.requests) == 1

    def test_cumulative_object_cap_aborts_without_upload_or_state_change(self) -> None:
        sink, client = _runtime_sink(format="json", max_object_bytes=30)
        with pytest.raises(Exception, match="byte limit"):
            sink.write([{"id": 1, "name": "x" * 100}], _SinkContext())
        assert client.requests == []
        assert sink._buffered_rows == []

    def test_rejected_row_fields_do_not_enter_observed_csv_schema(self) -> None:
        sink, client = _runtime_sink(format="csv")
        result = sink.write([{"rejected": object()}, {"id": 1}], _SinkContext())
        assert len(result.diversions) == 1
        assert client.bodies == [b"id\r\n1\r\n"]
        assert sink._buffered_rows == [{"id": 1}]

    @pytest.mark.parametrize("format", ["json", "jsonl"])
    def test_displayed_json_key_record_limit_diverts_before_cumulative_serialization(self, format: str) -> None:
        sink, client = _runtime_sink(
            format=format,
            headers={"id": "x" * 100},
            max_record_chars=50,
        )
        result = sink.write([{"id": 1}], _SinkContext())
        assert len(result.diversions) == 1
        assert result.diversions[0].reason == "record exceeds configured character limit"
        assert client.requests == []
        assert sink._buffered_rows == []

    def test_all_diverted_later_batch_preserves_confirmed_remote_artifact(self) -> None:
        sink, client = _runtime_sink(format="csv")
        context = _SinkContext()
        confirmed = sink.write([{"id": 1}], context)

        all_diverted = sink.write([{"rejected": object()}], context)

        assert len(client.requests) == 1
        assert len(all_diverted.diversions) == 1
        assert all_diverted.artifact == confirmed.artifact
        assert sink._buffered_rows == [{"id": 1}]

    @pytest.mark.parametrize("format", ["csv", "json", "jsonl"])
    def test_huge_integer_is_diverted_instead_of_escaping_write(self, format: str) -> None:
        sink, client = _runtime_sink(format=format)
        result = sink.write([{"id": 10**5000, "name": "Ada"}], _SinkContext())
        assert len(result.diversions) == 1
        assert "could not be" in result.diversions[0].reason
        assert client.requests == []

    def test_close_detaches_clears_and_closes_exactly_once(self) -> None:
        sink, client = _runtime_sink(overwrite=False)
        sink._buffered_rows = [{"id": 1}]
        sink._resolved_key = "output.csv"
        sink._remote_etag = '"etag"'
        sink._poisoned = True
        sink.close()
        sink.close()
        assert client.closed == 1
        assert sink._s3_client is None
        assert sink._buffered_rows == []
        assert sink._resolved_key is None
        assert sink._remote_etag is None
        assert sink._poisoned is False
        assert sink._closed is True

    def test_close_failure_is_static_after_cleanup(self) -> None:
        sink, client = _runtime_sink()
        client.close_error = RuntimeError("close endpoint credential sentinel")
        with pytest.raises(Exception, match="Failed to close S3 client") as captured:
            sink.close()
        assert captured.value.__cause__ is None and captured.value.__context__ is None
        assert "sentinel" not in str(captured.value)
        assert sink._s3_client is None and sink._closed is True
        sink.close()
        assert client.closed == 1

    def test_write_after_close_is_rejected(self) -> None:
        sink, client = _runtime_sink()
        sink.close()
        with pytest.raises(Exception, match="closed"):
            sink.write([{"id": 1, "name": "Ada"}], _SinkContext())
        assert client.requests == []

    def test_phase_error_does_not_expose_provider_sentinels(self) -> None:
        from elspeth.contracts import PhaseError, PipelinePhase

        sink, _ = _runtime_sink(client=_S3Client([_ProviderFailure("credential endpoint body sentinel")]))
        with pytest.raises(Exception) as captured:
            sink.write([{"id": 1, "name": "row sentinel"}], _SinkContext())
        public = PhaseError.from_exception(phase=PipelinePhase.EXPORT, error=captured.value, target="aws_s3")
        assert "sentinel" not in repr(public)

    def test_aws_s3_endpoint_url_is_not_a_secret_ref_field(self) -> None:
        from elspeth.web.secrets.ref_policy import allowed_secret_ref_fields

        assert "endpoint_url" not in allowed_secret_ref_fields("sink", "aws_s3")

    @pytest.mark.parametrize(("endpoint", "expected_passed"), [("http://localhost:4566", False), (None, True)])
    def test_registered_sink_uses_load_bearing_web_endpoint_gate(
        self,
        tmp_path: Any,
        endpoint: str | None,
        expected_passed: bool,
    ) -> None:
        from unittest.mock import MagicMock, patch

        from pydantic import SecretBytes

        from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
        from elspeth.web.config import WebSettings
        from elspeth.web.execution.protocol import YamlGenerator
        from elspeth.web.execution.validation import validate_pipeline_for_trained_operator

        source_path = tmp_path / "blobs" / "input.csv"
        source_path.parent.mkdir()
        source_path.write_text("id,name\n1,Ada\n", encoding="utf-8")
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="archive",
                options={"path": str(source_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(
                OutputSpec(
                    name="archive",
                    plugin="aws_s3",
                    options={
                        "bucket": "example-bucket",
                        "key": "output.csv",
                        "endpoint_url": endpoint,
                        "schema": {"mode": "observed"},
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = WebSettings(
            data_dir=tmp_path,
            composer_max_composition_turns=10,
            composer_max_discovery_turns=5,
            composer_timeout_seconds=30.0,
            composer_rate_limit_per_minute=60,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        )
        yaml_generator = MagicMock(spec=YamlGenerator)
        yaml_generator.generate_yaml.return_value = "sources: {}\nsinks: {}\n"
        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string", side_effect=ValueError("stop")) as load,
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins") as instantiate,
        ):
            result = validate_pipeline_for_trained_operator(state, settings, yaml_generator)
        check = next(check for check in result.checks if check.name == "aws_s3_endpoint_url_policy")
        assert check.passed is expected_passed
        if expected_passed:
            assert all(error.error_code != "aws_s3_endpoint_url_not_allowed" for error in result.errors)
            load.assert_called_once()
        else:
            assert result.errors[0].error_code == "aws_s3_endpoint_url_not_allowed"
            load.assert_not_called()
        instantiate.assert_not_called()
