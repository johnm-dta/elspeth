"""Validated, bounded primitives for the optional AWS S3 source plugin."""

from __future__ import annotations

import codecs
import hashlib
import re
import tempfile
from collections.abc import Mapping
from types import TracebackType
from typing import Any, BinaryIO, ClassVar, Literal, Never, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.identifiers import validate_field_names
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.plugins.infrastructure.config_base import DataPluginConfig

_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_ETAG_BYTES = 1024
_MAX_BUCKET_CHARS = 2048
_MAX_KEY_BYTES = 1024
_MAX_ENDPOINT_CHARS = 2048
_MAX_REGION_CHARS = 64
_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


class CSVOptions(BaseModel):
    """CSV parsing options for S3 objects."""

    model_config = {"extra": "forbid", "frozen": True}

    delimiter: str = ","
    has_header: bool = True
    encoding: str = "utf-8"

    @field_validator("delimiter")
    @classmethod
    def _validate_delimiter(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("delimiter must be a single character")
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError as exc:
            raise ValueError("unknown CSV encoding") from exc
        return value


class JSONOptions(BaseModel):
    """JSON and JSONL parsing options for S3 objects."""

    model_config = {"extra": "forbid", "frozen": True}

    encoding: str = "utf-8"
    data_key: str | None = None

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            codecs.lookup(value)
        except LookupError as exc:
            raise ValueError("unknown JSON encoding") from exc
        return value


class AWSS3SourceConfig(DataPluginConfig):
    """Strict AWS S3 source configuration with no credential surface."""

    _plugin_component_type: ClassVar[str | None] = "source"

    bucket: str = Field(..., description="S3 bucket name or access-point identifier")
    key: str = Field(..., description="S3 object key")
    format: Literal["csv", "json", "jsonl"] = Field(default="csv", description="S3 object data format")
    csv_options: CSVOptions = Field(default_factory=CSVOptions, description="CSV parsing options")
    json_options: JSONOptions = Field(default_factory=JSONOptions, description="JSON and JSONL parsing options")
    columns: list[str] | None = Field(default=None, description="Explicit columns for headerless CSV")
    field_mapping: dict[str, str] | None = Field(default=None, description="Overrides for normalized source fields")
    on_validation_failure: str = Field(..., description="Quarantine sink name or explicit discard")
    region_name: str | None = Field(default=None, description="AWS signing region override")
    endpoint_url: str | None = Field(default=None, description="CLI/batch-only S3-compatible HTTP endpoint")
    max_object_bytes: int = Field(
        default=256 * 1024 * 1024,
        gt=0,
        le=1024 * 1024 * 1024,
        strict=True,
        description="Maximum S3 object bytes accepted",
    )
    max_record_chars: int = Field(
        default=1_000_000,
        gt=0,
        le=8_000_000,
        strict=True,
        description="Maximum decoded characters in one source record",
    )

    @field_validator("bucket")
    @classmethod
    def _validate_bucket(cls, value: str) -> str:
        if not value.strip() or len(value) > _MAX_BUCKET_CHARS:
            raise ValueError("bucket must be nonblank and at most 2048 characters")
        return reject_operator_required_placeholder_value(value, field_name="bucket")

    @field_validator("key")
    @classmethod
    def _validate_key(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > _MAX_KEY_BYTES:
            raise ValueError("key must be nonblank and at most 1024 UTF-8 bytes")
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError("key must not contain control characters")
        return reject_operator_required_placeholder_value(value, field_name="key")

    @field_validator("region_name")
    @classmethod
    def _validate_region(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or len(value) > _MAX_REGION_CHARS or re.fullmatch(r"[A-Za-z0-9-]+", value) is None:
            raise ValueError("region_name must be a bounded AWS region identifier")
        return value

    @field_validator("endpoint_url")
    @classmethod
    def _validate_endpoint(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value or len(value) > _MAX_ENDPOINT_CHARS:
            raise ValueError("endpoint_url must be a bounded HTTP(S) URL")
        if any(character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F for character in value):
            raise ValueError("endpoint_url must not contain whitespace or control characters")
        try:
            parsed = urlsplit(value)
            _ = parsed.port
        except ValueError as exc:
            raise ValueError("endpoint_url must be a valid HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("endpoint_url must be an HTTP(S) URL with a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("endpoint_url must not contain userinfo")
        if parsed.query or parsed.fragment:
            raise ValueError("endpoint_url must not contain a query or fragment")
        return value

    @field_validator("on_validation_failure")
    @classmethod
    def _validate_failure_destination(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("on_validation_failure must be a sink name or 'discard'")
        return value.strip()

    @model_validator(mode="after")
    def _validate_field_options(self) -> Self:
        if self.format != "csv" and self.columns is not None:
            raise ValueError("columns is only supported for CSV format")
        if self.format == "csv" and self.csv_options.has_header and self.columns is not None:
            raise ValueError("columns requires csv_options.has_header: false")
        if self.format == "csv" and not self.csv_options.has_header and self.columns is None:
            schema = self.schema_config
            if schema is not None and (schema.is_observed or not schema.fields):
                raise ValueError("headerless CSV requires columns or declared schema fields")
        if self.columns is not None:
            validate_field_names(self.columns, "columns")
        if self.field_mapping:
            validate_field_names(list(self.field_mapping.values()), "field_mapping values")
        return self


AWSS3SourceConfig.model_rebuild()


def _normalize_error_type(exc: BaseException) -> str:
    name = type(exc).__name__
    return name if _SAFE_ERROR_TYPE.fullmatch(name) is not None else "ProviderError"


class S3SourceReadError(RuntimeError):
    """Static, redacted failure to obtain an S3 object's exact bytes."""

    def __init__(
        self,
        *,
        provider_error_type: str,
        bytes_read: int = 0,
        max_object_bytes: int,
        cleanup_error_type: str | None = None,
    ) -> None:
        super().__init__("Failed to read S3 object.")
        self.provider_error_type = provider_error_type
        self.bytes_read = bytes_read
        self.max_object_bytes = max_object_bytes
        self.cleanup_error_type = cleanup_error_type


class S3ObjectSizeLimitError(S3SourceReadError):
    """An S3 object's announced or observed bytes exceed the configured cap."""

    def __init__(
        self,
        *,
        observed_bytes: int,
        limit_bytes: int,
        cleanup_error_type: str | None = None,
    ) -> None:
        RuntimeError.__init__(self, f"S3 object size {observed_bytes} bytes exceeds limit {limit_bytes} bytes.")
        self.provider_error_type = "S3ObjectSizeLimitError"
        self.bytes_read = observed_bytes
        self.max_object_bytes = limit_bytes
        self.cleanup_error_type = cleanup_error_type
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes


class _DownloadedObject:
    """Exclusive owner of a rewound spooled S3 object and its safe metadata."""

    def __init__(self, handle: BinaryIO, *, size_bytes: int, content_hash: str) -> None:
        self.handle = handle
        self.size_bytes = size_bytes
        self.content_hash = content_hash
        self._closed = False

    @property
    def spool(self) -> BinaryIO:
        return self.handle

    @property
    def audit_metadata(self) -> dict[str, int | str]:
        return {"size_bytes": self.size_bytes, "content_hash": self.content_hash}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def _read_error(
    error_type: str,
    *,
    max_object_bytes: int,
    bytes_read: int = 0,
    cleanup_error_type: str | None = None,
) -> S3SourceReadError:
    return S3SourceReadError(
        provider_error_type=error_type,
        bytes_read=bytes_read,
        max_object_bytes=max_object_bytes,
        cleanup_error_type=cleanup_error_type,
    )


def _close_body(body: Any) -> str | None:
    cleanup_type: str | None = None
    try:
        body.close()
    except BaseException as exc:
        cleanup_type = _normalize_error_type(exc)
    return cleanup_type


def _close_spool(spool: BinaryIO) -> str | None:
    cleanup_type: str | None = None
    try:
        spool.close()
    except BaseException as exc:
        cleanup_type = _normalize_error_type(exc)
    return cleanup_type


def _new_spool() -> BinaryIO:
    """Create a spool whose ownership is transferred to ``_DownloadedObject``."""
    return cast(BinaryIO, tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b"))


def _raise_safe(error: S3SourceReadError) -> Never:
    raise error from None


def _validated_length(response: Mapping[str, Any]) -> int | None:
    value = response.get("ContentLength")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _valid_encoding(response: Mapping[str, Any]) -> bool:
    return response.get("ContentEncoding") in (None, "")


def _validated_etag(response: Mapping[str, Any]) -> str | None:
    value = response.get("ETag")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        return None
    if not 1 <= len(encoded) <= _MAX_ETAG_BYTES:
        return None
    if any(byte < 0x20 or byte > 0x7E for byte in encoded):
        return None
    return value


def _download_s3_object(client: Any, *, bucket: str, key: str, max_object_bytes: int) -> _DownloadedObject:
    """Download one immutable S3 object into a bounded spooled file."""
    head_response: Any = None
    head_error_type: str | None = None
    try:
        head_response = client.head_object(Bucket=bucket, Key=key)
    except BaseException as exc:
        head_error_type = _normalize_error_type(exc)
    if head_error_type is not None:
        _raise_safe(_read_error(head_error_type, max_object_bytes=max_object_bytes))
    if not isinstance(head_response, Mapping):
        _raise_safe(_read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes))

    head_length = _validated_length(head_response)
    head_etag = _validated_etag(head_response)
    if head_length is None:
        _raise_safe(_read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes))
    if head_etag is None or not _valid_encoding(head_response):
        _raise_safe(_read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes))
    if head_length > max_object_bytes:
        _raise_safe(S3ObjectSizeLimitError(observed_bytes=head_length, limit_bytes=max_object_bytes))

    get_response: Any = None
    get_error_type: str | None = None
    try:
        get_response = client.get_object(Bucket=bucket, Key=key, IfMatch=head_etag)
    except BaseException as exc:
        get_error_type = _normalize_error_type(exc)
    if get_error_type is not None:
        _raise_safe(_read_error(get_error_type, max_object_bytes=max_object_bytes))
    if not isinstance(get_response, Mapping):
        _raise_safe(_read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes))

    body = get_response.get("Body")
    if body is None or not callable(getattr(body, "read", None)) or not callable(getattr(body, "close", None)):
        _raise_safe(_read_error("InvalidS3Body", max_object_bytes=max_object_bytes))

    get_length = _validated_length(get_response)
    validation_error: S3SourceReadError | None = None
    if get_length is None or not _valid_encoding(get_response):
        validation_error = _read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes)
    elif get_length > max_object_bytes:
        validation_error = S3ObjectSizeLimitError(observed_bytes=get_length, limit_bytes=max_object_bytes)
    elif get_length != head_length:
        validation_error = _read_error("InvalidS3Metadata", max_object_bytes=max_object_bytes)
    if validation_error is not None:
        validation_error.cleanup_error_type = _close_body(body)
        _raise_safe(validation_error)

    spool = _new_spool()
    digest = hashlib.sha256()
    total = 0
    primary_error: S3SourceReadError | None = None
    while primary_error is None:
        chunk: Any = None
        read_error_type: str | None = None
        try:
            chunk = body.read(_DOWNLOAD_CHUNK_BYTES)
        except BaseException as exc:
            read_error_type = _normalize_error_type(exc)
        if read_error_type is not None:
            primary_error = _read_error(read_error_type, max_object_bytes=max_object_bytes, bytes_read=total)
            break
        if not isinstance(chunk, bytes) or len(chunk) > _DOWNLOAD_CHUNK_BYTES:
            primary_error = _read_error("InvalidS3Body", max_object_bytes=max_object_bytes, bytes_read=total)
            break
        if not chunk:
            break
        observed = total + len(chunk)
        if observed > max_object_bytes:
            primary_error = S3ObjectSizeLimitError(observed_bytes=observed, limit_bytes=max_object_bytes)
            break
        spool.write(chunk)
        digest.update(chunk)
        total = observed

    cleanup_error_type = _close_body(body)
    if primary_error is None and total != head_length:
        primary_error = _read_error("S3ContentLengthMismatch", max_object_bytes=max_object_bytes, bytes_read=total)
    if primary_error is None and cleanup_error_type is not None:
        primary_error = _read_error(cleanup_error_type, max_object_bytes=max_object_bytes, bytes_read=total)
    elif primary_error is not None:
        primary_error.cleanup_error_type = cleanup_error_type

    if primary_error is not None:
        spool_cleanup_type = _close_spool(spool)
        if primary_error.cleanup_error_type is None:
            primary_error.cleanup_error_type = spool_cleanup_type
        _raise_safe(primary_error)

    spool.seek(0)
    return _DownloadedObject(spool, size_bytes=total, content_hash=digest.hexdigest())
