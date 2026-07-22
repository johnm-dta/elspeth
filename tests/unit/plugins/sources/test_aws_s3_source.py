"""Unit tests for the bounded AWS S3 source primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "bucket": "input-bucket",
        "key": "incoming/data.csv",
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine",
    }
    config.update(overrides)
    return config


class TestAWSS3SourceConfig:
    def test_defaults_and_component_type(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        cfg = AWSS3SourceConfig.from_dict(_config())

        assert cfg.format == "csv"
        assert cfg.csv_options.delimiter == ","
        assert cfg.json_options.encoding == "utf-8"
        assert cfg.max_object_bytes == 256 * 1024 * 1024
        assert cfg.max_record_chars == 1_000_000
        assert cfg._plugin_component_type == "source"

    @pytest.mark.parametrize(
        "name",
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
    def test_forbidden_credentials_and_client_kwargs(self, name: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        assert name not in AWSS3SourceConfig.model_fields
        with pytest.raises(PluginConfigError, match="Extra inputs are not permitted"):
            AWSS3SourceConfig.from_dict(_config(**{name: "credential-sentinel"}))

    @pytest.mark.parametrize("bucket", ["", "   ", "<OPERATOR_REQUIRED>", "operator required", "x" * 2049])
    def test_bucket_rejects_invalid_shape(self, bucket: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError):
            AWSS3SourceConfig.from_dict(_config(bucket=bucket))

    @pytest.mark.parametrize("key", ["", "  ", "<OPERATOR_REQUIRED>", "operator_required", "a\x00b", "a\x1fb", "é" * 513])
    def test_key_rejects_invalid_shape(self, key: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError):
            AWSS3SourceConfig.from_dict(_config(key=key))

    def test_bucket_and_key_accept_exact_limits(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        cfg = AWSS3SourceConfig.from_dict(_config(bucket="b" * 2048, key="é" * 512))
        assert len(cfg.bucket) == 2048
        assert len(cfg.key.encode()) == 1024

    @pytest.mark.parametrize("region", ["", " ", "a" * 65, "us_east_1", "us.east.1", "us east 1"])
    def test_region_rejects_invalid_shape(self, region: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError):
            AWSS3SourceConfig.from_dict(_config(region_name=region))

    @pytest.mark.parametrize("region", [None, "us-east-1", "ap-southeast-2", "A1", "a" * 64])
    def test_region_accepts_bounded_shape(self, region: str | None) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        assert AWSS3SourceConfig.from_dict(_config(region_name=region)).region_name == region

    @pytest.mark.parametrize(
        "endpoint",
        ["https://s3.example.com", "http://localhost:4566", "https://127.0.0.1:9000", None],
    )
    def test_endpoint_accepts_bounded_http_urls(self, endpoint: str | None) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        assert AWSS3SourceConfig.from_dict(_config(endpoint_url=endpoint)).endpoint_url == endpoint

    @pytest.mark.parametrize(
        "endpoint",
        [
            "",
            "ftp://s3.example.com",
            "https://",
            "https://user@s3.example.com",
            "https://s3.example.com/path?secret=x",
            "https://s3.example.com/path#fragment",
            "https://s3.example.com/white space",
            "https://s3.example.com/\x01",
            "x" * 2049,
            "not-a-url",
        ],
    )
    def test_endpoint_rejects_unbounded_or_ambiguous_urls(self, endpoint: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError):
            AWSS3SourceConfig.from_dict(_config(endpoint_url=endpoint))

    @pytest.mark.parametrize(
        "field,valid,invalid", [("max_object_bytes", 1024**3, 1024**3 + 1), ("max_record_chars", 8_000_000, 8_000_001)]
    )
    def test_resource_limits_are_positive_and_capped(self, field: str, valid: int, invalid: int) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        assert getattr(AWSS3SourceConfig.from_dict(_config(**{field: valid})), field) == valid
        for value in (0, -1, invalid):
            with pytest.raises(PluginConfigError):
                AWSS3SourceConfig.from_dict(_config(**{field: value}))

    @pytest.mark.parametrize("field", ["max_object_bytes", "max_record_chars"])
    @pytest.mark.parametrize("value", [True, False, "1024", 1.5])
    def test_resource_limits_require_exact_integers(self, field: str, value: Any) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError):
            AWSS3SourceConfig.from_dict(_config(**{field: value}))

    def test_csv_normalization_options_match_azure_contract(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        cfg = AWSS3SourceConfig.from_dict(
            _config(
                csv_options={"delimiter": "|", "has_header": False, "encoding": "utf-16"},
                columns=["id", "full_name"],
                field_mapping={"full-name": "full_name"},
            )
        )
        assert cfg.columns == ["id", "full_name"]
        with pytest.raises(PluginConfigError, match="columns"):
            AWSS3SourceConfig.from_dict(_config(format="json", columns=["id"]))
        with pytest.raises(PluginConfigError, match="columns"):
            AWSS3SourceConfig.from_dict(_config(columns=["id"]))

    @pytest.mark.parametrize("encoding", ["rot_13", "base64_codec"])
    @pytest.mark.parametrize("field", ["csv_options", "json_options"])
    def test_binary_or_text_transform_codecs_are_rejected(self, field: str, encoding: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import AWSS3SourceConfig

        with pytest.raises(PluginConfigError, match="decode bytes to text"):
            AWSS3SourceConfig.from_dict(_config(**{field: {"encoding": encoding}}))


@dataclass
class _Body:
    chunks: list[bytes]
    read_error: BaseException | None = None
    close_error: BaseException | None = None
    reads: list[int] = field(default_factory=list)
    closed: bool = False

    def read(self, size: int) -> bytes:
        self.reads.append(size)
        if self.read_error is not None and not self.chunks:
            raise self.read_error
        if not self.chunks:
            return b""
        return self.chunks.pop(0)

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@dataclass
class _CloseOnlyBody:
    read: Any = None
    close_error: BaseException | None = None
    closed: bool = False

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@dataclass
class _FaultingSpool:
    write_error: BaseException | None = None
    seek_error: BaseException | None = None
    close_error: BaseException | None = None
    closed: bool = False
    writes: list[bytes] = field(default_factory=list)

    def write(self, data: bytes) -> int:
        if self.write_error is not None:
            raise self.write_error
        self.writes.append(data)
        return len(data)

    def seek(self, _offset: int) -> int:
        if self.seek_error is not None:
            raise self.seek_error
        return 0

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


@dataclass
class _Client:
    head: dict[str, Any]
    get: dict[str, Any]
    head_error: BaseException | None = None
    get_error: BaseException | None = None
    head_calls: list[dict[str, Any]] = field(default_factory=list)
    get_calls: list[dict[str, Any]] = field(default_factory=list)

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.head_calls.append(kwargs)
        if self.head_error is not None:
            raise self.head_error
        return self.head

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        if self.get_error is not None:
            raise self.get_error
        return self.get


def _client(data: bytes, *, head: dict[str, Any] | None = None, get: dict[str, Any] | None = None, body: Any = None) -> tuple[_Client, Any]:
    actual_body = _Body([data]) if body is None else body
    head_response = {"ContentLength": len(data), "ETag": '"etag"'} if head is None else head
    get_response = {"ContentLength": len(data), "Body": actual_body} if get is None else get
    return _Client(head_response, get_response), actual_body


class TestDownloadS3Object:
    def test_streams_bounded_chunks_hashes_rewinds_and_closes_body(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import _download_s3_object

        data = b"a,b\n1,2\n"
        body = _Body([data[:2], data[2:7], data[7:]])
        client, _ = _client(data, body=body)

        downloaded = _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=len(data))

        assert downloaded.size_bytes == len(data)
        assert downloaded.content_hash == hashlib.sha256(data).hexdigest()
        assert downloaded.handle.read() == data
        assert body.reads and all(size == 64 * 1024 for size in body.reads)
        assert body.closed
        assert client.get_calls == [{"Bucket": "bucket", "Key": "key", "IfMatch": '"etag"'}]
        downloaded.close()
        downloaded.close()
        assert downloaded.handle.closed

    @pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit()])
    @pytest.mark.parametrize("phase", ["head", "get", "read"])
    def test_process_control_exceptions_are_not_converted(self, phase: str, control: BaseException) -> None:
        from elspeth.plugins.sources.aws_s3_source import _download_s3_object

        data = b"x"
        body = _Body([], read_error=control if phase == "read" else None)
        client = _Client(
            {"ContentLength": 1, "ETag": '"etag"'},
            {"ContentLength": 1, "Body": body},
            head_error=control if phase == "head" else None,
            get_error=control if phase == "get" else None,
        )

        with pytest.raises(type(control)):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=len(data))

    @pytest.mark.parametrize("length", [True, False, -1, "1", None])
    @pytest.mark.parametrize("phase", ["head", "get"])
    def test_rejects_malformed_content_lengths(self, phase: str, length: Any) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        data = b"x"
        head = {"ContentLength": length if phase == "head" else 1, "ETag": '"etag"'}
        get = {"ContentLength": length if phase == "get" else 1, "Body": _Body([data])}
        client, _ = _client(data, head=head, get=get)
        with pytest.raises(S3SourceReadError):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)

    @pytest.mark.parametrize("etag", [None, b"etag", "", " ", "a\x1f", "é", "x" * 1025])
    def test_rejects_malformed_etag(self, etag: Any) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        client, _ = _client(b"x", head={"ContentLength": 1, "ETag": etag})
        with pytest.raises(S3SourceReadError):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)

    @pytest.mark.parametrize("etag", ["x" * 1023, "x" * 1024])
    def test_accepts_etag_boundary(self, etag: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import _download_s3_object

        client, _ = _client(b"x", head={"ContentLength": 1, "ETag": etag})
        with _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1) as downloaded:
            assert downloaded.size_bytes == 1

    @pytest.mark.parametrize("phase", ["head", "get"])
    def test_rejects_unsupported_content_encoding_before_read(self, phase: str) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        body = _Body([b"x"])
        head = {"ContentLength": 1, "ETag": '"etag"', "ContentEncoding": "gzip" if phase == "head" else ""}
        get = {"ContentLength": 1, "Body": body, "ContentEncoding": "gzip" if phase == "get" else None}
        client, _ = _client(b"x", head=head, get=get, body=body)
        with pytest.raises(S3SourceReadError):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        if phase == "head":
            assert not client.get_calls
        else:
            assert body.closed

    @pytest.mark.parametrize("advertised,body_bytes", [(1, b""), (1, b"xx"), (2, b"x")])
    def test_rejects_short_overlong_and_mismatched_bodies(self, advertised: int, body_bytes: bytes) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        body = _Body([body_bytes])
        client = _Client(
            {"ContentLength": advertised, "ETag": '"etag"'},
            {"ContentLength": advertised, "Body": body},
        )
        with pytest.raises(S3SourceReadError):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=10)
        assert body.closed

    @pytest.mark.parametrize("announced", [11, 12])
    def test_rejects_head_over_limit_before_get(self, announced: int) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3ObjectSizeLimitError, _download_s3_object

        client, _ = _client(b"x", head={"ContentLength": announced, "ETag": '"etag"'})
        with pytest.raises(S3ObjectSizeLimitError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=10)
        assert exc_info.value.observed_bytes == announced
        assert exc_info.value.limit_bytes == 10
        assert not client.get_calls

    def test_rejects_get_over_limit_before_read_and_closes_body(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3ObjectSizeLimitError, _download_s3_object

        body = _Body([b"secret-body"])
        client = _Client(
            {"ContentLength": 10, "ETag": '"etag"'},
            {"ContentLength": 11, "Body": body},
        )
        with pytest.raises(S3ObjectSizeLimitError):
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=10)
        assert not body.reads
        assert body.closed

    def test_rejects_stream_over_limit_immediately(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3ObjectSizeLimitError, _download_s3_object

        body = _Body([b"123456", b"78901"])
        client = _Client({"ContentLength": 10, "ETag": '"etag"'}, {"ContentLength": 10, "Body": body})
        with pytest.raises(S3ObjectSizeLimitError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=10)
        assert exc_info.value.observed_bytes == 11
        assert body.closed

    def test_missing_or_malformed_body_fails_closed(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        for body in (None, object()):
            client = _Client({"ContentLength": 0, "ETag": '"etag"'}, {"ContentLength": 0, "Body": body})
            with pytest.raises(S3SourceReadError):
                _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)

    @pytest.mark.parametrize("read_value", [None, 1, "not-callable"])
    def test_malformed_body_with_callable_close_is_closed(self, read_value: Any) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        body = _CloseOnlyBody(read=read_value)
        client = _Client({"ContentLength": 0, "ETag": '"etag"'}, {"ContentLength": 0, "Body": body})
        with pytest.raises(S3SourceReadError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        assert body.closed
        assert exc_info.value.provider_error_type == "InvalidS3Body"

    def test_malformed_body_close_failure_is_sanitized_cleanup_metadata(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        body = _CloseOnlyBody(read=None, close_error=ValueError("credential endpoint SENTINEL"))
        client = _Client({"ContentLength": 0, "ETag": '"etag"'}, {"ContentLength": 0, "Body": body})
        with pytest.raises(S3SourceReadError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        exc = exc_info.value
        assert body.closed
        assert exc.cleanup_error_type == "ValueError"
        assert "SENTINEL" not in f"{exc!s} {exc!r} {exc.__cause__!r} {exc.__context__!r}"

    def test_spool_constructor_failure_is_safe_and_closes_body(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        sentinel = "credential endpoint body SENTINEL"
        body = _Body([b"x"])
        client, _ = _client(b"x", body=body)

        def fail_spool() -> Any:
            raise OSError(sentinel)

        monkeypatch.setattr(aws_s3_source, "_new_spool", fail_spool)
        with pytest.raises(aws_s3_source.S3SourceReadError) as exc_info:
            aws_s3_source._download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        exc = exc_info.value
        assert body.closed
        assert exc.provider_error_type == "OSError"
        assert sentinel not in f"{exc!s} {exc!r} {exc.__cause__!r} {exc.__context__!r}"

    def test_spool_write_failure_preserves_primary_and_closes_body_and_spool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        sentinel = "credential endpoint body SENTINEL"
        spool = _FaultingSpool(write_error=OSError(sentinel), close_error=LookupError(sentinel))
        body = _Body([b"x"], close_error=ValueError(sentinel))
        client, _ = _client(b"x", body=body)
        monkeypatch.setattr(aws_s3_source, "_new_spool", lambda: spool)
        with pytest.raises(aws_s3_source.S3SourceReadError) as exc_info:
            aws_s3_source._download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        exc = exc_info.value
        assert body.closed and spool.closed
        assert exc.provider_error_type == "OSError"
        assert exc.cleanup_error_type == "ValueError"
        assert sentinel not in f"{exc!s} {exc!r} {exc.__cause__!r} {exc.__context__!r}"

    def test_spool_rewind_failure_is_safe_and_closes_spool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        sentinel = "credential endpoint body SENTINEL"
        spool = _FaultingSpool(seek_error=OSError(sentinel))
        body = _Body([b"x"])
        client, _ = _client(b"x", body=body)
        monkeypatch.setattr(aws_s3_source, "_new_spool", lambda: spool)
        with pytest.raises(aws_s3_source.S3SourceReadError) as exc_info:
            aws_s3_source._download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        exc = exc_info.value
        assert body.closed and spool.closed
        assert exc.provider_error_type == "OSError"
        assert sentinel not in f"{exc!s} {exc!r} {exc.__cause__!r} {exc.__context__!r}"

    def test_provider_and_cleanup_failures_are_redacted_and_unchained(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        sentinel = "credential endpoint body SENTINEL"
        body = _Body([], read_error=RuntimeError(sentinel), close_error=ValueError(sentinel))
        client = _Client({"ContentLength": 1, "ETag": '"etag"'}, {"ContentLength": 1, "Body": body})
        with pytest.raises(S3SourceReadError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        exc = exc_info.value
        surface = f"{exc!s} {exc!r} {exc.__cause__!r} {exc.__context__!r}"
        assert sentinel not in surface
        assert exc.__cause__ is None
        assert exc.__context__ is None
        assert exc.provider_error_type == "RuntimeError"
        assert exc.cleanup_error_type == "ValueError"

    def test_close_failure_on_success_becomes_safe_primary_failure(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError, _download_s3_object

        body = _Body([b"x"], close_error=RuntimeError("credential sentinel"))
        client, _ = _client(b"x", body=body)
        with pytest.raises(S3SourceReadError) as exc_info:
            _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=1)
        assert exc_info.value.provider_error_type == "RuntimeError"
        assert "credential" not in str(exc_info.value)

    def test_spool_rolls_to_disk_beyond_eight_mib(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import _download_s3_object

        data = b"x" * (8 * 1024 * 1024 + 1)
        chunks = [data[index : index + 64 * 1024] for index in range(0, len(data), 64 * 1024)]
        body = _Body(chunks)
        client, _ = _client(data, body=body)
        with _download_s3_object(client, bucket="bucket", key="key", max_object_bytes=len(data)) as downloaded:
            assert downloaded.handle._rolled is True  # type: ignore[attr-defined]
            assert downloaded.content_hash == hashlib.sha256(data).hexdigest()


@dataclass
class _SourceContext:
    calls: list[dict[str, Any]] = field(default_factory=list)
    validation_errors: list[dict[str, Any]] = field(default_factory=list)
    call_error: BaseException | None = None

    def record_call(self, **kwargs: Any) -> None:
        if self.call_error is not None:
            raise self.call_error
        self.calls.append(kwargs)

    def record_validation_error(self, **kwargs: Any) -> None:
        self.validation_errors.append(kwargs)


@dataclass
class _RuntimeClient:
    data: bytes
    chunk_size: int = 64 * 1024
    close_error: BaseException | None = None
    closed: int = 0
    bodies: list[_Body] = field(default_factory=list)

    def head_object(self, **_kwargs: Any) -> dict[str, Any]:
        return {"ContentLength": len(self.data), "ETag": '"runtime-etag"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["IfMatch"] == '"runtime-etag"'
        body = _Body([self.data[index : index + self.chunk_size] for index in range(0, len(self.data), self.chunk_size)])
        self.bodies.append(body)
        return {"ContentLength": len(self.data), "Body": body}

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def _source_for(data: bytes, **config_overrides: Any) -> tuple[Any, _RuntimeClient, _SourceContext]:
    from elspeth.plugins.sources.aws_s3_source import AWSS3Source

    source = AWSS3Source(_config(**config_overrides))
    client = _RuntimeClient(data)
    source._s3_client = client
    return source, client, _SourceContext()


class TestAWSS3SourceRegistrationAndParsing:
    def test_protocol_metadata_and_assistance(self) -> None:
        from elspeth.contracts import Determinism
        from elspeth.plugins.sources.aws_s3_source import AWSS3Source

        assert AWSS3Source.name == "aws_s3"
        assert AWSS3Source.determinism is Determinism.IO_READ
        assert AWSS3Source.plugin_version == "1.0.0"
        assert AWSS3Source.source_file_hash.startswith("sha256:")
        assistance = AWSS3Source.get_agent_assistance()
        assert assistance is not None and assistance.summary
        assert assistance.composer_hints
        assert all(len(hint) <= 280 for hint in assistance.composer_hints)

    def test_aws_s3_endpoint_url_is_not_a_secret_ref_field(self) -> None:
        from elspeth.web.secrets.ref_policy import allowed_secret_ref_fields

        assert "endpoint_url" not in allowed_secret_ref_fields("source", "aws_s3")

    def test_registered_aws_s3_source_is_endpoint_url_gated(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from pydantic import SecretBytes

        from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
        from elspeth.web.config import WebSettings
        from elspeth.web.execution.protocol import YamlGenerator
        from elspeth.web.execution.validation import validate_pipeline_for_trained_operator

        state = CompositionState(
            source=SourceSpec(
                plugin="aws_s3",
                on_success="results",
                options={"endpoint_url": "https://credential-canary.attacker.invalid/private"},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="results", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = WebSettings(
            data_dir=Path("/tmp/test_data"),
            composer_max_composition_turns=10,
            composer_max_discovery_turns=5,
            composer_timeout_seconds=30.0,
            composer_rate_limit_per_minute=60,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        )
        yaml_generator = MagicMock(spec=YamlGenerator)
        yaml_generator.generate_yaml.return_value = "sources: {}\nsinks: {}\n"
        with (
            patch("elspeth.web.execution.validation.load_settings_from_yaml_string") as load_settings,
            patch("elspeth.web.execution.validation.instantiate_runtime_plugins") as instantiate,
        ):
            result = validate_pipeline_for_trained_operator(state, settings, yaml_generator)
        check = next(check for check in result.checks if check.name == "aws_s3_endpoint_url_policy")
        assert check.passed is False
        assert result.errors[0].error_code == "aws_s3_endpoint_url_not_allowed"
        yaml_generator.generate_yaml.assert_called_once_with(state)
        load_settings.assert_not_called()
        instantiate.assert_not_called()

    def test_registered_aws_s3_source_accepts_explicit_null_endpoint(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from pydantic import SecretBytes

        from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
        from elspeth.web.config import WebSettings
        from elspeth.web.execution.protocol import YamlGenerator
        from elspeth.web.execution.validation import validate_pipeline_for_trained_operator

        state = CompositionState(
            source=SourceSpec(
                plugin="aws_s3",
                on_success="results",
                options={"endpoint_url": None},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(),
            outputs=(OutputSpec(name="results", plugin="csv", options={}, on_write_failure="discard"),),
            metadata=PipelineMetadata(),
            version=1,
        )
        settings = WebSettings(
            data_dir=Path("/tmp/test_data"),
            composer_max_composition_turns=10,
            composer_max_discovery_turns=5,
            composer_timeout_seconds=30.0,
            composer_rate_limit_per_minute=60,
            shareable_link_signing_key=SecretBytes(b"\x00" * 32),
        )
        yaml_generator = MagicMock(spec=YamlGenerator)
        yaml_generator.generate_yaml.return_value = "sources: {}\nsinks: {}\n"
        with patch("elspeth.web.execution.validation.load_settings_from_yaml_string", side_effect=ValueError("stop")):
            result = validate_pipeline_for_trained_operator(state, settings, yaml_generator)
        check = next(check for check in result.checks if check.name == "aws_s3_endpoint_url_policy")
        assert check.passed is True

    def test_csv_parity_headers_normalization_and_schema_coercion(self) -> None:
        source, _, ctx = _source_for(b"ID,Full Name,score\n1,Ada,2.5\n")

        rows = list(source.load(ctx))

        assert [row.row for row in rows] == [{"id": "1", "full_name": "Ada", "score": "2.5"}]
        assert rows[0].is_quarantined is False
        resolution = source.get_field_resolution()
        assert resolution is not None and resolution[0] == {"ID": "id", "Full Name": "full_name", "score": "score"}

    def test_headerless_csv_columns_and_custom_delimiter(self) -> None:
        source, _, ctx = _source_for(
            b"1|Ada\n2|Grace\n",
            csv_options={"delimiter": "|", "has_header": False},
            columns=["id", "name"],
        )
        assert [row.row for row in source.load(ctx)] == [{"id": "1", "name": "Ada"}, {"id": "2", "name": "Grace"}]

    @pytest.mark.parametrize(
        "source_format,data,expected",
        [
            (
                "json",
                b'[{"ID":1,"name":"Ada"},{"ID":2,"email":"g@example.test"}]',
                [{"id": 1, "name": "Ada"}, {"id": 2, "email": "g@example.test"}],
            ),
            (
                "jsonl",
                b'{"ID":1,"name":"Ada"}\n\n{"ID":2,"email":"g@example.test"}\n',
                [{"id": 1, "name": "Ada"}, {"id": 2, "email": "g@example.test"}],
            ),
        ],
    )
    def test_json_formats_normalize_sparse_field_union(self, source_format: str, data: bytes, expected: list[dict[str, Any]]) -> None:
        source, _, ctx = _source_for(data, format=source_format)
        assert [row.row for row in source.load(ctx)] == expected
        resolution = source.get_field_resolution()
        assert resolution is not None
        assert set(resolution[0].values()) == {"id", "name", "email"}

    def test_json_exact_literal_data_key_with_dot(self) -> None:
        source, _, ctx = _source_for(
            b'{"rows.v1":[{"id":1}],"rows":{"v1":[{"id":2}]}}', format="json", json_options={"data_key": "rows.v1"}
        )
        assert [row.row for row in source.load(ctx)] == [{"id": 1}]

    @pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
    def test_json_streams_registered_unicode_encodings(self, encoding: str) -> None:
        data = json.dumps([{"name": "café", "value": 1.25}], ensure_ascii=False).encode(encoding)
        source, _, ctx = _source_for(data, format="json", json_options={"encoding": encoding})
        assert [row.row for row in source.load(ctx)] == [{"name": "café", "value": 1.25}]

    def test_json_number_parity_preserves_large_integer_and_converts_decimal(self) -> None:
        source, _, ctx = _source_for(b'[{"integer":123456789012345678901234567890,"decimal":1.25,"scientific":1e2}]', format="json")
        row = next(iter(source.load(ctx))).row
        assert row["integer"] == 123456789012345678901234567890
        assert row["decimal"] == 1.25
        assert row["scientific"] == 100.0

    @pytest.mark.parametrize("source_format,data", [("csv", b""), ("json", b"")])
    def test_empty_structural_formats_quarantine(self, source_format: str, data: bytes) -> None:
        source, _, ctx = _source_for(data, format=source_format)
        rows = list(source.load(ctx))
        assert len(rows) == 1 and rows[0].is_quarantined
        assert ctx.calls[0]["response_data"]["content_hash"] == hashlib.sha256(b"").hexdigest()

    def test_empty_jsonl_yields_no_rows_and_still_audits(self) -> None:
        source, _, ctx = _source_for(b"", format="jsonl")
        assert list(source.load(ctx)) == []
        assert len(ctx.calls) == 1

    @pytest.mark.parametrize(
        "source_format,data",
        [
            ("csv", b"value\n123456\n"),
            ("jsonl", b'{"value":"123456"}\n'),
            ("json", b'[{"value":"123456"}]'),
            ("json", b'{"ignored":"123456","rows":[{"id":1}]}'),
        ],
    )
    def test_record_caps_reject_selected_and_ignored_oversized_tokens(self, source_format: str, data: bytes) -> None:
        options: dict[str, Any] = {"format": source_format, "max_record_chars": 5}
        if source_format == "json" and data.startswith(b"{"):
            options["json_options"] = {"data_key": "rows"}
        source, _, ctx = _source_for(data, **options)
        rows = list(source.load(ctx))
        assert rows and rows[0].is_quarantined

    def test_json_aggregate_cap_rejects_many_small_fields(self) -> None:
        payload = json.dumps([{f"k{i}": "x" for i in range(20)}]).encode()
        source, _, ctx = _source_for(payload, format="json", max_record_chars=40)
        rows = list(source.load(ctx))
        assert len(rows) == 1 and rows[0].is_quarantined

    def test_json_nonfinite_after_decimal_conversion_quarantines(self) -> None:
        source, _, ctx = _source_for(b'[{"value":1e9999}]', format="json")
        rows = list(source.load(ctx))
        assert len(rows) == 1 and rows[0].is_quarantined

    @pytest.mark.parametrize("max_chars,quarantined", [(7, True), (8, False), (9, False)])
    def test_json_final_aggregate_cap_exact_boundary(self, max_chars: int, quarantined: bool) -> None:
        source, _, ctx = _source_for(b'[{"x":""}]', format="json", max_record_chars=max_chars)
        rows = list(source.load(ctx))
        assert bool(rows and rows[0].is_quarantined) is quarantined

    def test_json_budget_rejects_two_empty_fields_before_retention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        retained: list[tuple[str, Any]] = []

        class TrackingDict(dict[str, Any]):
            def __setitem__(self, key: str, value: Any) -> None:
                retained.append((key, value))
                super().__setitem__(key, value)

        monkeypatch.setattr(aws_s3_source, "_new_json_object", TrackingDict, raising=False)
        events = iter([("map_key", "a"), ("string", ""), ("map_key", "b"), ("string", ""), ("end_map", None)])
        with pytest.raises(aws_s3_source._RecordLimitExceeded):
            aws_s3_source._build_json_value(
                ("start_map", None),
                events,
                max_record_chars=5,
                budget=[0],
            )
        assert retained == []

    def test_json_array_budget_rejects_scalar_before_appending_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        retained: list[Any] = []

        class TrackingList(list[Any]):
            def append(self, value: Any) -> None:
                retained.append(value)
                super().append(value)

        monkeypatch.setattr(aws_s3_source, "_new_json_array", TrackingList)
        events = iter([("string", "a\n"), ("string", "b"), ("end_array", None)])
        with pytest.raises(aws_s3_source._RecordLimitExceeded):
            aws_s3_source._build_json_value(
                ("start_array", None),
                events,
                max_record_chars=len('["a\\n","b"]') - 1,
                budget=[0],
            )
        assert retained == ["a\n"]

    @pytest.mark.parametrize(
        "events,exact_cost",
        [
            (
                [("map_key", "a\n"), ("string", "x\t"), ("end_map", None)],
                len(json.dumps({"a\n": "x\t"}, ensure_ascii=False, separators=(",", ":"))),
            ),
            ([("map_key", "a"), ("start_array", None), ("end_array", None), ("end_map", None)], len('{"a":[]}')),
            ([("string", "a\n"), ("string", "b"), ("end_array", None)], len('["a\\n","b"]')),
        ],
    )
    def test_json_budget_accounts_for_escaping_separators_and_container_edges(
        self,
        events: list[tuple[str, Any]],
        exact_cost: int,
    ) -> None:
        from elspeth.plugins.sources import aws_s3_source

        first = ("start_array", None) if events[0][0] == "string" else ("start_map", None)
        for limit, passes in ((exact_cost - 1, False), (exact_cost, True), (exact_cost + 1, True)):
            event_iterator = iter(events)
            if passes:
                result = aws_s3_source._build_json_value(first, event_iterator, max_record_chars=limit, budget=[0])
                assert result is not None
            else:
                with pytest.raises(aws_s3_source._RecordLimitExceeded):
                    aws_s3_source._build_json_value(first, event_iterator, max_record_chars=limit, budget=[0])

    @pytest.mark.parametrize("max_chars,quarantined", [(2, True), (3, False), (4, False)])
    def test_ignored_json_scalar_token_cap_exact_boundary(self, max_chars: int, quarantined: bool) -> None:
        source, _, ctx = _source_for(
            b'{"x":"abc","r":[]}',
            format="json",
            json_options={"data_key": "r"},
            max_record_chars=max_chars,
        )
        rows = list(source.load(ctx))
        assert bool(rows and rows[0].is_quarantined) is quarantined

    def test_ignored_json_excessive_depth_is_rejected(self) -> None:
        nested = "[" * 65 + "0" + "]" * 65
        source, _, ctx = _source_for(
            f'{{"x":{nested},"r":[]}}'.encode(),
            format="json",
            json_options={"data_key": "r"},
        )
        rows = list(source.load(ctx))
        assert len(rows) == 1 and rows[0].is_quarantined

    def test_json_array_non_object_quarantines_and_continues(self) -> None:
        source, _, ctx = _source_for(b'[1,{"id":2}]', format="json")
        rows = list(source.load(ctx))
        assert rows[0].is_quarantined
        assert rows[1].row == {"id": 2}

    def test_jsonl_bad_line_quarantines_and_next_line_survives(self) -> None:
        source, _, ctx = _source_for(b'{bad}\n{"id":2}\n', format="jsonl")
        rows = list(source.load(ctx))
        assert rows[0].is_quarantined
        assert rows[1].row == {"id": 2}

    def test_csv_wrong_width_quarantines_and_next_row_survives(self) -> None:
        source, _, ctx = _source_for(b"id,name\n1\n2,Grace\n")
        rows = list(source.load(ctx))
        assert rows[0].is_quarantined
        assert rows[1].row == {"id": "2", "name": "Grace"}

    def test_sparse_field_mapping_applies_when_field_first_appears_later(self) -> None:
        source, _, ctx = _source_for(
            b'[{"id":1},{"id":2,"Full Name":"Grace"}]',
            format="json",
            field_mapping={"full_name": "display_name"},
        )
        assert [row.row for row in source.load(ctx)] == [{"id": 1}, {"id": 2, "display_name": "Grace"}]

    def test_fixed_schema_quarantines_invalid_row_without_echoing_value(self) -> None:
        sentinel = "credential-body-SENTINEL"
        source, _, ctx = _source_for(
            json.dumps([{"id": sentinel}]).encode(),
            format="json",
            schema={"mode": "fixed", "fields": ["id: int"]},
        )
        rows = list(source.load(ctx))
        assert len(rows) == 1 and rows[0].is_quarantined
        assert sentinel not in rows[0].quarantine_error

    def test_lazy_missing_ijson_is_actionable_and_download_closes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        source, client, ctx = _source_for(b"[]", format="json")
        real_import = builtins.__import__

        def missing(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ijson":
                raise ImportError("provider detail SENTINEL")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", missing)
        with pytest.raises(ImportError, match="aws"):
            list(source.load(ctx))
        assert client.bodies[0].closed

    def test_source_parses_a_spool_that_rolls_to_disk(self) -> None:
        data_rows = [b"x" * 1024 + b"\n"] * 8193
        data = b"value\n" + b"".join(data_rows)
        source, _, ctx = _source_for(data, max_record_chars=2048)
        rows = list(source.load(ctx))
        assert len(rows) == 8193
        assert rows[0].row == {"value": "x" * 1024}


class TestAWSS3SourceAuditAndLifecycle:
    def test_success_audit_is_exact_and_redacted(self) -> None:
        data = b"id,name\n1,Ada\n"
        source, _, ctx = _source_for(data)
        list(source.load(ctx))
        assert len(ctx.calls) == 1
        call = ctx.calls[0]
        assert call["request_data"] == {"operation": "read_object", "bucket": "input-bucket", "key": "incoming/data.csv"}
        assert call["response_data"] == {"size_bytes": len(data), "content_hash": hashlib.sha256(data).hexdigest()}
        assert call["provider"] == "aws_s3"
        assert "endpoint" not in repr(call).lower()

    def test_transport_failure_audits_only_safe_fields(self) -> None:
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError

        source, _, ctx = _source_for(b"x")
        source._s3_client = _Client(
            {"ContentLength": 1, "ETag": '"etag"'},
            {"ContentLength": 1, "Body": _Body([], read_error=RuntimeError("credential endpoint body SENTINEL"))},
        )
        with pytest.raises(S3SourceReadError):
            list(source.load(ctx))
        assert len(ctx.calls) == 1
        assert set(ctx.calls[0]["error"]) <= {"type", "bytes_read", "max_object_bytes", "cleanup_error_type"}
        assert "SENTINEL" not in repr(ctx.calls)

    def test_client_construction_failure_is_static_unchained_and_audited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        sentinel = "credential endpoint body SENTINEL"
        source = aws_s3_source.AWSS3Source(_config())
        ctx = _SourceContext()

        def fail_builder(_region: str | None, _endpoint: str | None) -> Any:
            raise RuntimeError(sentinel)

        monkeypatch.setattr(aws_s3_source, "build_s3_client", fail_builder)
        with pytest.raises(aws_s3_source.S3SourceReadError) as exc_info:
            list(source.load(ctx))
        exc = exc_info.value
        assert exc.provider_error_type == "RuntimeError"
        assert exc.__cause__ is None and exc.__context__ is None
        assert sentinel not in f"{exc!s} {exc!r} {ctx.calls!r}"
        assert len(ctx.calls) == 1
        assert ctx.calls[0]["status"].value == "error"
        assert ctx.calls[0]["error"] == {
            "type": "RuntimeError",
            "bytes_read": 0,
            "max_object_bytes": 256 * 1024 * 1024,
        }

    def test_client_construction_missing_extra_importerror_remains_actionable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from elspeth.plugins.sources import aws_s3_source

        source = aws_s3_source.AWSS3Source(_config())
        ctx = _SourceContext()

        def missing_builder(_region: str | None, _endpoint: str | None) -> Any:
            raise ImportError('install Elspeth with the "aws" extra')

        monkeypatch.setattr(aws_s3_source, "build_s3_client", missing_builder)
        with pytest.raises(ImportError, match="aws"):
            list(source.load(ctx))
        assert ctx.calls == []

    def test_audit_write_failure_is_static_unchained_integrity_error(self) -> None:
        from elspeth.contracts.errors import AuditIntegrityError

        source, _, ctx = _source_for(b"id\n1\n")
        ctx.call_error = RuntimeError("credential endpoint body SENTINEL")
        with pytest.raises(AuditIntegrityError) as exc_info:
            list(source.load(ctx))
        assert "SENTINEL" not in f"{exc_info.value!s} {exc_info.value!r}"
        assert exc_info.value.__cause__ is None
        assert exc_info.value.__context__ is None

    def test_failure_path_audit_write_failure_is_static_integrity_error(self) -> None:
        from elspeth.contracts.errors import AuditIntegrityError

        source, _, ctx = _source_for(b"x")
        source._s3_client = _Client(
            {"ContentLength": 1, "ETag": '"etag"'},
            {"ContentLength": 1, "Body": _Body([], read_error=RuntimeError("provider SENTINEL"))},
        )
        ctx.call_error = RuntimeError("recorder SENTINEL")
        with pytest.raises(AuditIntegrityError) as exc_info:
            list(source.load(ctx))
        surface = f"{exc_info.value!s} {exc_info.value!r} {exc_info.value.__cause__!r} {exc_info.value.__context__!r}"
        assert "SENTINEL" not in surface

    def test_public_provider_failure_is_safe_in_phase_error(self) -> None:
        from structlog.testing import capture_logs

        from elspeth.contracts.events import PhaseError, PipelinePhase
        from elspeth.plugins.sources.aws_s3_source import S3SourceReadError

        source, _, ctx = _source_for(b"x")
        source._s3_client = _Client(
            {"ContentLength": 1, "ETag": '"etag"'},
            {"ContentLength": 1, "Body": _Body([], read_error=RuntimeError("credential endpoint body SENTINEL"))},
        )
        with capture_logs() as logs, pytest.raises(S3SourceReadError) as exc_info:
            list(source.load(ctx))
        event = PhaseError.from_exception(phase=PipelinePhase.SOURCE, error=exc_info.value)
        assert "SENTINEL" not in repr(event)
        assert "SENTINEL" not in repr(logs)

    def test_generator_close_releases_download(self) -> None:
        source, client, ctx = _source_for(b"id\n1\n2\n")
        iterator = source.load(ctx)
        assert next(iterator).row == {"id": "1"}
        iterator.close()
        assert client.bodies[0].closed
        assert source._active_download is None

    def test_concurrent_load_rejected_and_reuse_after_close_rejected(self) -> None:
        source, _, ctx = _source_for(b"id\n1\n2\n")
        iterator = source.load(ctx)
        next(iterator)
        with pytest.raises(RuntimeError, match="already active"):
            next(source.load(ctx))
        iterator.close()
        source.close()
        with pytest.raises(RuntimeError, match="closed"):
            next(source.load(ctx))

    def test_close_while_suspended_detaches_resources_and_resume_stops_cleanly(self) -> None:
        source, client, ctx = _source_for(b"id\n1\n2\n")
        iterator = source.load(ctx)
        assert next(iterator).row == {"id": "1"}
        source.close()
        assert client.closed == 1 and client.bodies[0].closed
        with pytest.raises(StopIteration):
            next(iterator)
        source.close()
        assert client.closed == 1

    def test_client_close_failure_is_redacted_and_not_retried(self) -> None:
        source, client, _ = _source_for(b"")
        client.close_error = RuntimeError("credential endpoint SENTINEL")
        with pytest.raises(RuntimeError) as exc_info:
            source.close()
        assert "SENTINEL" not in str(exc_info.value)
        source.close()
        assert client.closed == 1
