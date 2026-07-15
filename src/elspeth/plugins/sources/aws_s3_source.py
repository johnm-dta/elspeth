"""Validated, bounded primitives for the optional AWS S3 source plugin."""

from __future__ import annotations

import codecs
import csv
import enum
import hashlib
import io
import json
import math
import re
import sys
import tempfile
import time
from collections.abc import Iterator as ABCIterator
from collections.abc import Mapping
from contextlib import suppress
from decimal import Decimal
from types import TracebackType
from typing import Any, BinaryIO, ClassVar, Literal, Never, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts import CallStatus, CallType, Determinism, PluginSchema, SourceRow
from elspeth.contracts import errors as contract_errors
from elspeth.contracts.contexts import SourceContext
from elspeth.contracts.contract_builder import ContractBuilder, ContractFieldLimitExceeded
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.identifiers import validate_field_names
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.plugins.aws_s3_common import build_s3_client
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.plugins.infrastructure.config_base import DataPluginConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.sources._safe_validation_errors import safe_validation_error_text
from elspeth.plugins.sources.field_normalization import (
    ExternalHeaderError,
    FieldMappingCollisionError,
    FieldResolution,
    extend_field_resolution,
    normalize_field_name,
    resolve_field_names,
)
from elspeth.plugins.sources.json_source import (
    _contains_surrogateescape_chars,
    _reject_nonfinite_constant,
    _surrogateescape_line_to_bytes,
)

_DOWNLOAD_CHUNK_BYTES = 64 * 1024
_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_MAX_ETAG_BYTES = 1024
_MAX_BUCKET_CHARS = 2048
_MAX_KEY_BYTES = 1024
_MAX_ENDPOINT_CHARS = 2048
_MAX_REGION_CHARS = 64
_MAX_JSON_DEPTH = 64
_SAFE_ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")


class _RowSentinel(enum.Enum):
    EXHAUSTED = enum.auto()


_ROW_EXHAUSTED = _RowSentinel.EXHAUSTED


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
        _validate_text_encoding(value, format_name="CSV")
        return value


class JSONOptions(BaseModel):
    """JSON and JSONL parsing options for S3 objects."""

    model_config = {"extra": "forbid", "frozen": True}

    encoding: str = "utf-8"
    data_key: str | None = None

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        _validate_text_encoding(value, format_name="JSON")
        return value


def _validate_text_encoding(value: str, *, format_name: str) -> None:
    try:
        decoder = codecs.getincrementaldecoder(value)(errors="strict")
        decoded = decoder.decode(b"", final=True)
    except (LookupError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{format_name} encoding must decode bytes to text") from exc
    if not isinstance(decoded, str):
        raise ValueError(f"{format_name} encoding must decode bytes to text")


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
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
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
    body_close = getattr(body, "close", None)
    if body is None or not callable(getattr(body, "read", None)) or not callable(body_close):
        interface_error = _read_error("InvalidS3Body", max_object_bytes=max_object_bytes)
        if callable(body_close):
            interface_error.cleanup_error_type = _close_body(body)
        _raise_safe(interface_error)

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

    spool: BinaryIO | None = None
    spool_create_error_type: str | None = None
    try:
        spool = _new_spool()
    except BaseException as exc:
        spool_create_error_type = _normalize_error_type(exc)
    if spool_create_error_type is not None:
        create_error = _read_error(spool_create_error_type, max_object_bytes=max_object_bytes)
        create_error.cleanup_error_type = _close_body(body)
        _raise_safe(create_error)
    if spool is None:
        raise AssertionError("S3 spool creation completed without a handle or failure")

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
        write_result: Any = None
        write_error_type: str | None = None
        try:
            write_result = spool.write(chunk)
        except BaseException as exc:
            write_error_type = _normalize_error_type(exc)
        if write_error_type is not None:
            primary_error = _read_error(write_error_type, max_object_bytes=max_object_bytes, bytes_read=total)
            break
        if isinstance(write_result, bool) or not isinstance(write_result, int) or write_result != len(chunk):
            primary_error = _read_error("InvalidS3Spool", max_object_bytes=max_object_bytes, bytes_read=total)
            break
        digest.update(chunk)
        total = observed

    cleanup_error_type = _close_body(body)
    if primary_error is None and total != head_length:
        primary_error = _read_error("S3ContentLengthMismatch", max_object_bytes=max_object_bytes, bytes_read=total)
    if primary_error is None and cleanup_error_type is not None:
        primary_error = _read_error(cleanup_error_type, max_object_bytes=max_object_bytes, bytes_read=total)
    elif primary_error is not None:
        primary_error.cleanup_error_type = cleanup_error_type

    if primary_error is None:
        rewind_error_type: str | None = None
        try:
            spool.seek(0)
        except BaseException as exc:
            rewind_error_type = _normalize_error_type(exc)
        if rewind_error_type is not None:
            primary_error = _read_error(rewind_error_type, max_object_bytes=max_object_bytes, bytes_read=total)

    if primary_error is not None:
        spool_cleanup_type = _close_spool(spool)
        if primary_error.cleanup_error_type is None:
            primary_error.cleanup_error_type = spool_cleanup_type
        _raise_safe(primary_error)

    return _DownloadedObject(spool, size_bytes=total, content_hash=digest.hexdigest())


class _RecordLimitExceeded(ValueError):
    """A decoded parser record exceeded its configured character cap."""


class _JSONBoundaryError(ValueError):
    """A JSON document violated a bounded structural parsing rule."""


class _BoundedDecodedLineIterator:
    """Yield physical lines while bounding each logical parser record."""

    def __init__(self, stream: io.TextIOWrapper, max_record_chars: int) -> None:
        self._stream = stream
        self._max_record_chars = max_record_chars
        self._logical_chars = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> str:
        remaining = self._max_record_chars - self._logical_chars
        if remaining < 0:
            raise _RecordLimitExceeded("decoded record exceeds configured character limit")
        line = self._stream.readline(remaining + 1)
        if line == "":
            raise StopIteration
        self._logical_chars += len(line)
        if self._logical_chars > self._max_record_chars:
            raise _RecordLimitExceeded("decoded record exceeds configured character limit")
        return line

    def finish_record(self) -> None:
        self._logical_chars = 0


class _BoundedJSONTokenReader:
    """Bound ijson reads and reject oversized lexical tokens before release."""

    def __init__(self, stream: io.TextIOWrapper, max_token_chars: int) -> None:
        self._stream = stream
        self._max_token_chars = max_token_chars
        self._in_string = False
        self._escape = False
        self._unicode_escape_remaining = 0
        self._token_chars = 0
        self._in_primitive = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        bounded_size = _DOWNLOAD_CHUNK_BYTES if size < 0 else min(size, _DOWNLOAD_CHUNK_BYTES)
        # ijson's stable binary-reader contract expects UTF-8 bytes. Decode the
        # configured source codec first, scan decoded tokens, then re-encode.
        # Four is the maximum UTF-8 byte width of one Unicode scalar.
        decoded_size = max(1, bounded_size // 4)
        chunk = self._stream.read(decoded_size)
        self._scan(chunk)
        return chunk.encode("utf-8")

    def _increment_token(self) -> None:
        self._token_chars += 1
        if self._token_chars > self._max_token_chars:
            raise _RecordLimitExceeded("JSON token exceeds configured character limit")

    def _scan(self, chunk: str) -> None:
        for character in chunk:
            if self._in_string:
                if self._unicode_escape_remaining:
                    self._unicode_escape_remaining -= 1
                    if self._unicode_escape_remaining == 0:
                        self._increment_token()
                    continue
                if self._escape:
                    self._escape = False
                    if character == "u":
                        self._unicode_escape_remaining = 4
                    else:
                        self._increment_token()
                    continue
                if character == "\\":
                    self._escape = True
                elif character == '"':
                    self._in_string = False
                    self._token_chars = 0
                else:
                    self._increment_token()
                continue

            if self._in_primitive:
                if character.isspace() or character in ",]}:":
                    self._in_primitive = False
                    self._token_chars = 0
                else:
                    self._increment_token()
                    continue

            if character == '"':
                self._in_string = True
                self._token_chars = 0
            elif not character.isspace() and character not in "[]{}:,":
                self._in_primitive = True
                self._token_chars = 0
                self._increment_token()


def _next_json_event(events: ABCIterator[tuple[str, Any]]) -> tuple[str, Any]:
    try:
        return next(events)
    except StopIteration:
        raise _JSONBoundaryError("JSON document ended before its structure completed") from None


def _budget_add(budget: list[int], amount: int, limit: int) -> None:
    budget[0] += amount
    if budget[0] > limit:
        raise _RecordLimitExceeded("JSON item exceeds configured character limit")


def _normalize_json_scalar(value: Any) -> Any:
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise _JSONBoundaryError("JSON number must be finite")
        converted = float(value)
        if not math.isfinite(converted):
            raise _JSONBoundaryError("JSON number cannot be represented as a finite float")
        return converted
    if isinstance(value, float) and not math.isfinite(value):
        raise _JSONBoundaryError("JSON number must be finite")
    return value


def _rendered_json_cost(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise _JSONBoundaryError("JSON value cannot be represented safely") from exc


def _new_json_object() -> dict[str, Any]:
    return {}


def _new_json_array() -> list[Any]:
    return []


def _build_json_value(
    first_event: tuple[str, Any],
    events: ABCIterator[tuple[str, Any]],
    *,
    max_record_chars: int,
    budget: list[int],
    depth: int = 0,
) -> Any:
    event, value = first_event
    if event in {"string", "number", "boolean", "null"}:
        normalized = _normalize_json_scalar(value)
        _budget_add(budget, _rendered_json_cost(normalized), max_record_chars)
        return normalized

    if event == "start_map":
        if depth >= _MAX_JSON_DEPTH:
            raise _JSONBoundaryError("JSON nesting exceeds configured depth")
        # Reserve both delimiters before retaining any member. Each member then
        # reserves its optional comma, rendered key (including quotes/escapes),
        # and colon before its value is constructed or stored.
        _budget_add(budget, 2, max_record_chars)
        result = _new_json_object()
        first_member = True
        while True:
            key_event, key = _next_json_event(events)
            if key_event == "end_map":
                return result
            if key_event != "map_key" or not isinstance(key, str):
                raise _JSONBoundaryError("JSON object structure is invalid")
            member_syntax_cost = (0 if first_member else 1) + _rendered_json_cost(key) + 1
            _budget_add(budget, member_syntax_cost, max_record_chars)
            value_event = _next_json_event(events)
            built = _build_json_value(
                value_event,
                events,
                max_record_chars=max_record_chars,
                budget=budget,
                depth=depth + 1,
            )
            result[key] = built
            first_member = False

    if event == "start_array":
        if depth >= _MAX_JSON_DEPTH:
            raise _JSONBoundaryError("JSON nesting exceeds configured depth")
        # Reserve both delimiters before retaining any item. Commas are charged
        # before constructing the following item.
        _budget_add(budget, 2, max_record_chars)
        result_list = _new_json_array()
        first_item = True
        while True:
            item_event = _next_json_event(events)
            if item_event[0] == "end_array":
                return result_list
            if not first_item:
                _budget_add(budget, 1, max_record_chars)
            built_item = _build_json_value(
                item_event,
                events,
                max_record_chars=max_record_chars,
                budget=budget,
                depth=depth + 1,
            )
            result_list.append(built_item)
            first_item = False

    raise _JSONBoundaryError("JSON value structure is invalid")


def _skip_json_value(first_event: tuple[str, Any], events: ABCIterator[tuple[str, Any]], *, depth: int = 0) -> None:
    event, _value = first_event
    if event in {"string", "number", "boolean", "null"}:
        return
    if event not in {"start_map", "start_array"}:
        raise _JSONBoundaryError("JSON value structure is invalid")
    if depth >= _MAX_JSON_DEPTH:
        raise _JSONBoundaryError("JSON nesting exceeds configured depth")
    end_event = "end_map" if event == "start_map" else "end_array"
    while True:
        child = _next_json_event(events)
        if child[0] == end_event:
            return
        if event == "start_map":
            if child[0] != "map_key":
                raise _JSONBoundaryError("JSON object structure is invalid")
            child = _next_json_event(events)
        _skip_json_value(child, events, depth=depth + 1)


def _iter_json_array_items(
    events: ABCIterator[tuple[str, Any]],
    *,
    max_record_chars: int,
) -> ABCIterator[Any]:
    while True:
        event = _next_json_event(events)
        if event[0] == "end_array":
            return
        budget = [0]
        item = _build_json_value(event, events, max_record_chars=max_record_chars, budget=budget)
        try:
            rendered = json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise _JSONBoundaryError("JSON item cannot be represented safely") from exc
        if len(rendered) > max_record_chars:
            raise _RecordLimitExceeded("JSON item exceeds configured character limit")
        yield item


def _iter_selected_json_items(
    events: ABCIterator[tuple[str, Any]],
    *,
    data_key: str | None,
    max_record_chars: int,
) -> ABCIterator[Any]:
    root_event = _next_json_event(events)
    if data_key is None:
        if root_event[0] != "start_array":
            raise _JSONBoundaryError("JSON root must be an array")
        yield from _iter_json_array_items(events, max_record_chars=max_record_chars)
    else:
        if root_event[0] != "start_map":
            raise _JSONBoundaryError("JSON root must be an object when data_key is configured")
        found = False
        while True:
            key_event, key = _next_json_event(events)
            if key_event == "end_map":
                break
            if key_event != "map_key" or not isinstance(key, str):
                raise _JSONBoundaryError("JSON object structure is invalid")
            value_event = _next_json_event(events)
            if key == data_key:
                if found:
                    raise _JSONBoundaryError("JSON data_key appears more than once")
                found = True
                if value_event[0] != "start_array":
                    raise _JSONBoundaryError("JSON data_key must contain an array")
                yield from _iter_json_array_items(events, max_record_chars=max_record_chars)
            else:
                _skip_json_value(value_event, events)
        if not found:
            raise _JSONBoundaryError("JSON data_key was not found")

    try:
        next(events)
    except StopIteration:
        return
    raise _JSONBoundaryError("JSON document contains trailing data")


def _raise_audit_error(error_type: str) -> Never:
    error = AuditIntegrityError(f"Failed to record S3 call in the audit trail ({error_type}).")
    raise error from None


def _record_download_call(
    ctx: SourceContext,
    *,
    status: CallStatus,
    bucket: str,
    key: str,
    latency_ms: float,
    response_data: dict[str, Any] | None = None,
    error_data: dict[str, Any] | None = None,
) -> None:
    recorder_error_type: str | None = None
    try:
        ctx.record_call(
            call_type=CallType.HTTP,
            status=status,
            request_data={"operation": "read_object", "bucket": bucket, "key": key},
            response_data=response_data,
            error=error_data,
            latency_ms=latency_ms,
            provider="aws_s3",
        )
    except contract_errors.TIER_1_ERRORS:
        raise
    except BaseException as exc:
        recorder_error_type = _normalize_error_type(exc)
    if recorder_error_type is not None:
        _raise_audit_error(recorder_error_type)


class AWSS3Source(BaseSource):
    """Load bounded CSV, JSON-array, or JSONL rows from one immutable S3 object."""

    name = "aws_s3"
    determinism = Determinism.IO_READ
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:bf19d531af899115"
    config_model = AWSS3SourceConfig

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is not None:
            return None
        return PluginAssistance(
            plugin_name=cls.name,
            issue_code=None,
            summary="Loads bounded CSV, JSON, or JSONL rows from a pre-existing AWS S3 object.",
            composer_hints=(
                "Use the AWS default credential chain; never place access keys or session tokens in pipeline options.",
                "Use endpoint_url only for trusted CLI or batch S3-compatible endpoints; web-authored endpoint overrides are rejected.",
                "Choose format explicitly; csv uses csv_options, while json and jsonl use json_options.",
                "Reference a real pre-existing S3 bucket and object key; this source does not create source data.",
                "Use field_mapping only to override normalized field names at the source boundary.",
                "Route validation failures to a quarantine sink unless deliberate discard is acceptable.",
            ),
        )

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = AWSS3SourceConfig.from_dict(config, plugin_name=self.name)
        self._bucket = cfg.bucket
        self._key = cfg.key
        self._format = cfg.format
        self._csv_options = cfg.csv_options
        self._json_options = cfg.json_options
        self._columns = cfg.columns
        self._field_mapping = cfg.field_mapping
        self._region_name = cfg.region_name
        self._endpoint_url = cfg.endpoint_url
        self._max_object_bytes = cfg.max_object_bytes
        self._max_record_chars = cfg.max_record_chars
        self._on_validation_failure = cfg.on_validation_failure
        self._schema_config = cfg.schema_config
        self._initialize_declared_guaranteed_fields(self._schema_config)
        self._field_resolution: FieldResolution | None = None
        self._contract_builder: ContractBuilder | None = None
        self._first_valid_row_processed = False

        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "AWSS3RowSchema",
            allow_coercion=True,
        )
        self.output_schema = self._schema_class
        if self._format != "csv":
            initial_contract = create_contract_from_config(self._schema_config)
            if initial_contract.locked:
                self.set_schema_contract(initial_contract)

        self._s3_client: Any | None = None
        self._active_download: _DownloadedObject | None = None
        self._closed = False

    def _get_s3_client(self) -> Any:
        if self._s3_client is None:
            client: Any = None
            client_error_type: str | None = None
            try:
                client = build_s3_client(self._region_name, self._endpoint_url)
            except ImportError:
                raise
            except BaseException as exc:
                client_error_type = _normalize_error_type(exc)
            if client_error_type is not None:
                _raise_safe(_read_error(client_error_type, max_object_bytes=self._max_object_bytes))
            if client is None:
                raise AssertionError("S3 client construction completed without a client or failure")
            self._s3_client = client
        return self._s3_client

    def _is_closed(self) -> bool:
        """Read lifecycle state at generator suspension boundaries."""
        return self._closed

    def load(self, ctx: SourceContext) -> ABCIterator[SourceRow]:
        """Download, audit, incrementally parse, and validate one S3 object."""
        if self._closed:
            raise RuntimeError("aws_s3 source is closed")
        if self._active_download is not None:
            raise RuntimeError("aws_s3 source load is already active")

        self._first_valid_row_processed = False
        started = time.perf_counter()
        download: _DownloadedObject | None = None
        failure: S3SourceReadError | None = None
        try:
            download = _download_s3_object(
                self._get_s3_client(),
                bucket=self._bucket,
                key=self._key,
                max_object_bytes=self._max_object_bytes,
            )
        except S3SourceReadError as exc:
            failure = exc

        latency_ms = (time.perf_counter() - started) * 1000
        if failure is not None:
            error_data: dict[str, Any] = {
                "type": failure.provider_error_type,
                "bytes_read": failure.bytes_read,
                "max_object_bytes": failure.max_object_bytes,
            }
            if failure.cleanup_error_type is not None:
                error_data["cleanup_error_type"] = failure.cleanup_error_type
            _record_download_call(
                ctx,
                status=CallStatus.ERROR,
                bucket=self._bucket,
                key=self._key,
                latency_ms=latency_ms,
                error_data=error_data,
            )
            raise failure from None

        if download is None:
            raise AssertionError("S3 download completed without a result or failure")
        self._active_download = download
        parser: ABCIterator[SourceRow] | None = None
        try:
            _record_download_call(
                ctx,
                status=CallStatus.SUCCESS,
                bucket=self._bucket,
                key=self._key,
                latency_ms=latency_ms,
                response_data=download.audit_metadata,
            )

            if self._format == "csv":
                parser = self._load_csv(download.handle, ctx)
            elif self._format == "json":
                parser = self._load_json_array(download.handle, ctx)
            else:
                parser = self._load_jsonl(download.handle, ctx)

            while not self._is_closed():
                try:
                    row = next(parser)
                except StopIteration:
                    break
                if self._is_closed():
                    break
                yield row

            if not self._first_valid_row_processed:
                if self._contract_builder is not None:
                    self.set_schema_contract(self._contract_builder.contract.with_locked())
                elif self.get_schema_contract() is None:
                    self.set_schema_contract(create_contract_from_config(self._schema_config).with_locked())
        finally:
            primary_active = sys.exc_info()[0] is not None
            cleanup_failed = False
            parser_close = getattr(parser, "close", None)
            if callable(parser_close):
                try:
                    parser_close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    cleanup_failed = True
            if download is not None and self._active_download is download:
                self._active_download = None
            if download is not None:
                try:
                    download.close()
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    cleanup_failed = True
            if cleanup_failed and not primary_active:
                raise RuntimeError("Failed to close aws_s3 parser resources.") from None

    def _file_error(self, ctx: SourceContext, message: str) -> ABCIterator[SourceRow]:
        raw_row = {"bucket": self._bucket, "key": self._key, "error": message}
        ctx.record_validation_error(
            row=raw_row,
            error=message,
            schema_mode="parse",
            destination=self._on_validation_failure,
        )
        if self._on_validation_failure != "discard":
            yield SourceRow.quarantined(
                row=raw_row,
                error=message,
                destination=self._on_validation_failure,
                source_row_index=0,
            )

    def _load_csv(self, handle: BinaryIO, ctx: SourceContext) -> ABCIterator[SourceRow]:
        stream = io.TextIOWrapper(handle, encoding=self._csv_options.encoding, errors="strict", newline="")
        lines = _BoundedDecodedLineIterator(stream, self._max_record_chars)
        reader = csv.reader(lines, delimiter=self._csv_options.delimiter, strict=True)
        try:
            if self._csv_options.has_header:
                try:
                    raw_headers = next(reader, _ROW_EXHAUSTED)
                    lines.finish_record()
                except (_RecordLimitExceeded, csv.Error, UnicodeDecodeError):
                    yield from self._file_error(ctx, "CSV header could not be parsed safely")
                    return
                if raw_headers is _ROW_EXHAUSTED:
                    yield from self._file_error(ctx, "CSV parse error: empty file contains no header row")
                    return
                try:
                    self._field_resolution = resolve_field_names(
                        raw_headers=raw_headers,
                        field_mapping=self._field_mapping,
                        columns=None,
                    )
                except ExternalHeaderError:
                    yield from self._file_error(ctx, "CSV header could not be resolved safely")
                    return
                headers = self._field_resolution.final_headers
            elif self._columns is not None:
                self._field_resolution = resolve_field_names(
                    raw_headers=None,
                    field_mapping=self._field_mapping,
                    columns=self._columns,
                )
                headers = self._field_resolution.final_headers
            elif not self._schema_config.is_observed and self._schema_config.fields:
                schema_names = [field_definition.name for field_definition in self._schema_config.fields]
                self._field_resolution = resolve_field_names(
                    raw_headers=None,
                    field_mapping=self._field_mapping,
                    columns=schema_names,
                )
                headers = self._field_resolution.final_headers
            else:
                raise AssertionError("headerless CSV field names must be validated before loading")

            if self._contract_builder is None:
                initial_contract = create_contract_from_config(
                    self._schema_config,
                    field_resolution=self._field_resolution.resolution_mapping,
                )
                if initial_contract.locked:
                    self.set_schema_contract(initial_contract)
                else:
                    self._contract_builder = ContractBuilder(initial_contract)

            row_count = 0
            while True:
                try:
                    values = next(reader, _ROW_EXHAUSTED)
                    lines.finish_record()
                except _RecordLimitExceeded:
                    row_count += 1
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__row_number__": str(row_count)},
                        "CSV record exceeds configured character limit",
                        row_count - 1,
                    )
                    return
                except (csv.Error, UnicodeDecodeError):
                    row_count += 1
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__row_number__": str(row_count)},
                        "CSV record could not be parsed safely",
                        row_count - 1,
                    )
                    return
                if values is _ROW_EXHAUSTED:
                    return
                if not values:
                    continue
                row_count += 1
                if len(values) != len(headers):
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__row_number__": str(row_count), "__field_count__": len(values)},
                        f"CSV parse error: expected {len(headers)} fields, got {len(values)}",
                        row_count - 1,
                    )
                    continue
                row = dict(zip(headers, values, strict=True))
                yield from self._validate_and_yield(row, ctx, source_row_index=row_count - 1)
        finally:
            with suppress(ValueError, OSError):
                stream.detach()

    def _load_jsonl(self, handle: BinaryIO, ctx: SourceContext) -> ABCIterator[SourceRow]:
        stream = io.TextIOWrapper(
            handle,
            encoding=self._json_options.encoding,
            errors="surrogateescape",
            newline="",
        )
        lines = _BoundedDecodedLineIterator(stream, self._max_record_chars)
        line_number = 0
        try:
            while True:
                try:
                    raw_line = next(lines)
                    lines.finish_record()
                except StopIteration:
                    return
                except _RecordLimitExceeded:
                    line_number += 1
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__line_number__": line_number},
                        "JSONL record exceeds configured character limit",
                        line_number - 1,
                    )
                    return
                except UnicodeDecodeError:
                    line_number += 1
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"bucket": self._bucket, "key": self._key, "__line_number__": line_number},
                        "JSONL record has invalid encoded text",
                        line_number - 1,
                    )
                    return
                line_number += 1
                line = raw_line.strip()
                if not line:
                    continue
                if _contains_surrogateescape_chars(raw_line):
                    raw_bytes = _surrogateescape_line_to_bytes(raw_line.rstrip("\r\n"), self._json_options.encoding)
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__raw_bytes_hex__": raw_bytes.hex(), "__line_number__": line_number},
                        f"JSON parse error at line {line_number}: invalid {self._json_options.encoding} encoding",
                        line_number - 1,
                    )
                    continue
                try:
                    row = json.loads(line, parse_constant=_reject_nonfinite_constant)
                except (json.JSONDecodeError, ValueError):
                    yield from self._quarantine_parse_row(
                        ctx,
                        {"__raw_line__": line, "__line_number__": str(line_number)},
                        f"JSON parse error at line {line_number}",
                        line_number - 1,
                    )
                    continue
                yield from self._validate_and_yield(row, ctx, source_row_index=line_number - 1)
        finally:
            with suppress(ValueError, OSError):
                stream.detach()

    def _load_json_array(self, handle: BinaryIO, ctx: SourceContext) -> ABCIterator[SourceRow]:
        try:
            import ijson
        except ImportError as exc:
            raise ImportError('ijson is required for aws_s3 JSON sources; install Elspeth with the "aws" extra') from exc

        stream = io.TextIOWrapper(handle, encoding=self._json_options.encoding, errors="strict", newline="")
        bounded = _BoundedJSONTokenReader(stream, self._max_record_chars)
        events = cast(ABCIterator[tuple[str, Any]], iter(ijson.basic_parse(bounded, use_float=False)))
        items = _iter_selected_json_items(
            events,
            data_key=self._json_options.data_key,
            max_record_chars=self._max_record_chars,
        )
        source_row_index = 0
        try:
            while True:
                try:
                    row = next(items)
                except StopIteration:
                    return
                except (ijson.JSONError, UnicodeDecodeError, _RecordLimitExceeded, _JSONBoundaryError):
                    yield from self._file_error(ctx, "JSON object could not be parsed within configured bounds")
                    return
                yield from self._validate_and_yield(row, ctx, source_row_index=source_row_index)
                source_row_index += 1
        finally:
            with suppress(ValueError, OSError):
                stream.detach()

    def _quarantine_parse_row(
        self,
        ctx: SourceContext,
        raw_row: Any,
        message: str,
        source_row_index: int,
    ) -> ABCIterator[SourceRow]:
        ctx.record_validation_error(
            row=raw_row,
            error=message,
            schema_mode="parse",
            destination=self._on_validation_failure,
        )
        if self._on_validation_failure != "discard":
            yield SourceRow.quarantined(
                row=raw_row,
                error=message,
                destination=self._on_validation_failure,
                source_row_index=source_row_index,
            )

    def _normalize_row_keys(self, row: Any) -> Mapping[str, Any]:
        try:
            row_items = list(row.items())
        except AttributeError:
            raise ExternalHeaderError(f"Expected JSON object, got {type(row).__name__}") from None

        raw_keys = [key for key, _ in row_items]
        if self._field_resolution is None:
            self._field_resolution = self._resolve_json_field_names(raw_keys)
            if self._contract_builder is None and self.get_schema_contract() is None:
                initial_contract = create_contract_from_config(
                    self._schema_config,
                    field_resolution=self._field_resolution.resolution_mapping,
                )
                self._contract_builder = ContractBuilder(initial_contract)

        mapping = self._field_resolution.resolution_mapping
        normalized: dict[str, Any] = {}
        new_raw_keys: list[str] = []
        for key, value in row_items:
            if not isinstance(key, str):
                raise ExternalHeaderError("JSON object keys must be strings")
            if key in mapping:
                normalized[mapping[key]] = value
            else:
                normalized_key = normalize_field_name(key)
                final_name = (
                    self._field_mapping[normalized_key]
                    if self._field_mapping is not None and normalized_key in self._field_mapping
                    else normalized_key
                )
                normalized[final_name] = value
                new_raw_keys.append(key)

        if new_raw_keys:
            try:
                self._field_resolution = extend_field_resolution(
                    self._field_resolution,
                    raw_headers=new_raw_keys,
                    field_mapping=self._field_mapping,
                )
            except FieldMappingCollisionError as exc:
                raise ExternalHeaderError(str(exc)) from exc
        return normalized

    def _resolve_json_field_names(self, raw_keys: list[str]) -> FieldResolution:
        try:
            return resolve_field_names(
                raw_headers=raw_keys,
                field_mapping=self._field_mapping,
                columns=None,
                require_all_mapping_keys=False,
            )
        except FieldMappingCollisionError as exc:
            raise ExternalHeaderError(str(exc)) from exc

    def _validate_and_yield(self, row: Any, ctx: SourceContext, *, source_row_index: int) -> ABCIterator[SourceRow]:
        try:
            row_to_validate = self._normalize_row_keys(row) if self._format in ("json", "jsonl") else row
        except ExternalHeaderError as exc:
            message = f"Field normalization failed: {exc}"
            ctx.record_validation_error(
                row=row,
                error=message,
                schema_mode="field_normalization",
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=message,
                    destination=self._on_validation_failure,
                    source_row_index=source_row_index,
                )
            return

        try:
            validated = self._schema_class.model_validate(row_to_validate)
            validated_row = validated.to_row()
            if self._contract_builder is not None and not self._first_valid_row_processed:
                if self._field_resolution is None:
                    if self._format in ("json", "jsonl"):
                        raise ValueError("field_resolution must exist before first-row inference")
                    field_resolution_map: Mapping[str, str] = {key: key for key in validated_row}
                else:
                    field_resolution_map = self._field_resolution.resolution_mapping
                self._contract_builder.process_first_row(validated_row, field_resolution_map)
                self.set_schema_contract(self._contract_builder.contract)
                self._first_valid_row_processed = True

            contract = self.require_schema_contract()
            if self._format in ("json", "jsonl") and self._contract_builder is not None and contract.mode in ("OBSERVED", "FLEXIBLE"):
                if self._field_resolution is None:
                    raise ValueError("field_resolution must exist before sparse-field inference")
                contract = self._contract_builder.process_sparse_fields(
                    validated_row,
                    self._field_resolution.resolution_mapping,
                )
                self.set_schema_contract(contract)

            if contract.locked:
                violations = contract.validate(validated_row)
                if violations:
                    message = "; ".join(str(violation) for violation in violations)
                    ctx.record_validation_error(
                        row=validated_row,
                        error=message,
                        schema_mode=self._schema_config.mode,
                        destination=self._on_validation_failure,
                    )
                    if self._on_validation_failure != "discard":
                        yield SourceRow.quarantined(
                            row=validated_row,
                            error=message,
                            destination=self._on_validation_failure,
                            source_row_index=source_row_index,
                        )
                    return
            yield SourceRow.valid(validated_row, contract=contract, source_row_index=source_row_index)
        except ContractFieldLimitExceeded as exc:
            message = str(exc)
            ctx.record_validation_error(
                row=row_to_validate,
                error=message,
                schema_mode=self._schema_config.mode,
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row_to_validate,
                    error=message,
                    destination=self._on_validation_failure,
                    source_row_index=source_row_index,
                )
        except ValidationError as exc:
            message = safe_validation_error_text(exc)
            ctx.record_validation_error(
                row=row_to_validate,
                error=message,
                schema_mode=self._schema_config.mode,
                destination=self._on_validation_failure,
            )
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row_to_validate,
                    error=message,
                    destination=self._on_validation_failure,
                    source_row_index=source_row_index,
                )

    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
        if self._field_resolution is None:
            return None
        return self._field_resolution.resolution_mapping, self._field_resolution.normalization_version

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        download = self._active_download
        self._active_download = None
        client = self._s3_client
        self._s3_client = None

        cleanup_error_type: str | None = None
        if download is not None:
            try:
                download.close()
            except BaseException as exc:
                cleanup_error_type = _normalize_error_type(exc)
        if client is not None:
            close_method = getattr(client, "close", None)
            if not callable(close_method):
                if cleanup_error_type is None:
                    cleanup_error_type = "InvalidS3Client"
            else:
                try:
                    close_method()
                except BaseException as exc:
                    if cleanup_error_type is None:
                        cleanup_error_type = _normalize_error_type(exc)
        if cleanup_error_type is not None:
            error = RuntimeError(f"Failed to close aws_s3 source resources ({cleanup_error_type}).")
            raise error from None
