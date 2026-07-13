"""Unit tests for the bounded AWS S3 source primitives."""

from __future__ import annotations

import hashlib
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
