"""Validated, bounded primitives for the optional AWS S3 sink plugin."""

from __future__ import annotations

import base64
import codecs
import csv
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any, BinaryIO, ClassVar, Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts.header_modes import HeaderMode, parse_header_mode
from elspeth.contracts.wire_visible_identity import reject_operator_required_placeholder_value
from elspeth.plugins.infrastructure.config_base import DataPluginConfig, validate_headers_value

_SPOOL_MEMORY_BYTES = 8 * 1024 * 1024
_WRITE_CHUNK_BYTES = 64 * 1024
_MAX_BUCKET_CHARS = 2048
_MAX_KEY_TEMPLATE_BYTES = 4096
_MAX_RENDERED_KEY_BYTES = 1024
_MAX_ENDPOINT_CHARS = 2048
_MAX_REGION_CHARS = 64


class CSVWriteOptions(BaseModel):
    """CSV output controls for the S3 sink."""

    model_config = {"extra": "forbid", "frozen": True}

    delimiter: str = Field(default=",", description="Single-character CSV field delimiter")
    encoding: str = Field(default="utf-8", description="Character encoding for CSV output")
    include_header: bool = Field(default=True, description="Write the CSV header record")

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


class AWSS3SinkConfig(DataPluginConfig):
    """Strict S3 sink configuration with no credential or raw client surface."""

    _plugin_component_type: ClassVar[str | None] = "sink"

    bucket: str = Field(..., description="S3 bucket name or access-point identifier")
    key: str = Field(..., description="S3 object key template rendered once per run")
    format: Literal["csv", "json", "jsonl"] = Field(default="csv", description="S3 object data format")
    overwrite: bool = Field(default=True, description="Allow replacement of an existing S3 object")
    csv_options: CSVWriteOptions = Field(default_factory=CSVWriteOptions, description="CSV writing options")
    headers: str | dict[str, str] | None = Field(
        default=None,
        description="Normalized, original, or custom output headers",
    )
    region_name: str | None = Field(default=None, description="AWS signing region override")
    endpoint_url: str | None = Field(default=None, description="CLI/batch-only S3-compatible HTTP endpoint")
    max_object_bytes: int = Field(
        default=256 * 1024 * 1024,
        gt=0,
        le=1024 * 1024 * 1024,
        strict=True,
        description="Maximum serialized S3 object bytes",
    )
    max_record_chars: int = Field(
        default=1_000_000,
        gt=0,
        le=8_000_000,
        strict=True,
        description="Maximum characters in one serialized output record",
    )

    @field_validator("bucket")
    @classmethod
    def _validate_bucket(cls, value: str) -> str:
        if not value.strip() or len(value) > _MAX_BUCKET_CHARS:
            raise ValueError("bucket must be nonblank and at most 2048 characters")
        return reject_operator_required_placeholder_value(value, field_name="bucket")

    @field_validator("key")
    @classmethod
    def _validate_key_template(cls, value: str) -> str:
        if not value.strip() or len(value.encode("utf-8")) > _MAX_KEY_TEMPLATE_BYTES:
            raise ValueError("key template must be nonblank and at most 4096 UTF-8 bytes")
        if _has_control_character(value):
            raise ValueError("key template must not contain control characters")
        reject_operator_required_placeholder_value(value, field_name="key")
        _, template_syntax_error, environment_type = _load_jinja()
        environment = environment_type(undefined=_load_jinja()[0])
        try:
            environment.from_string(value)
        except template_syntax_error as exc:
            raise ValueError("key template syntax is invalid") from exc
        return value

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, value: str | dict[str, str] | None) -> str | dict[str, str] | None:
        return validate_headers_value(value)

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

    @model_validator(mode="after")
    def _validate_csv_only_options(self) -> Self:
        if self.format != "csv" and self.csv_options != CSVWriteOptions():
            raise ValueError("csv_options is only supported for CSV format")
        return self

    @property
    def headers_mode(self) -> HeaderMode:
        return parse_header_mode(self.headers) if self.headers is not None else HeaderMode.NORMALIZED

    @property
    def headers_mapping(self) -> dict[str, str] | None:
        return cast("dict[str, str]", self.headers) if self.headers_mode is HeaderMode.CUSTOM else None


AWSS3SinkConfig.model_rebuild()


def _load_jinja() -> tuple[type[Any], type[BaseException], type[Any]]:
    """Load Jinja lazily so the base install can still discover other plugins."""
    try:
        from jinja2 import StrictUndefined, TemplateSyntaxError
        from jinja2.sandbox import SandboxedEnvironment
    except ImportError as exc:
        raise ImportError('Jinja2 is required for aws_s3 sinks; install Elspeth with the "aws" extra') from exc
    return StrictUndefined, TemplateSyntaxError, SandboxedEnvironment


def _has_control_character(value: str) -> bool:
    return any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)


def _validate_rendered_key(value: str) -> str:
    if not value.strip() or len(value.encode("utf-8")) > _MAX_RENDERED_KEY_BYTES:
        raise ValueError("rendered key must be nonblank and at most 1024 UTF-8 bytes")
    if _has_control_character(value):
        raise ValueError("rendered key must not contain control characters")
    return value


def _render_key_template(template_source: str, *, run_id: str, timestamp: str) -> str:
    strict_undefined, _, environment_type = _load_jinja()
    environment = environment_type(undefined=strict_undefined)
    try:
        rendered = environment.from_string(template_source).render(run_id=run_id, timestamp=timestamp)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ValueError("S3 key template could not be rendered safely") from None
    return _validate_rendered_key(rendered)


class S3ObjectSizeLimitError(RuntimeError):
    """The complete candidate object exceeded its configured byte cap."""

    def __init__(self, observed_bytes: int, limit_bytes: int) -> None:
        super().__init__(f"S3 object exceeds configured byte limit ({observed_bytes} > {limit_bytes}).")
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes


class S3RecordSerializationError(ValueError):
    """A row cannot be represented without exposing its value in the error."""

    def __init__(self) -> None:
        super().__init__("S3 output record could not be serialized safely.")


class S3RecordSizeLimitError(S3RecordSerializationError):
    """A row exceeded the configured character budget."""

    def __init__(self) -> None:
        ValueError.__init__(self, "S3 output record exceeds configured character limit.")


@dataclass(slots=True)
class _SerializedObject:
    """Own a rewound temporary object and its stable integrity metadata."""

    body: BinaryIO
    size_bytes: int
    content_hash: str
    checksum_sha256_b64: str
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.body.close()

    def __enter__(self) -> _SerializedObject:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class _BoundedBinaryWriter:
    def __init__(self, body: BinaryIO, max_object_bytes: int) -> None:
        self._body = body
        self._max_object_bytes = max_object_bytes
        self.size_bytes = 0
        self.digest = hashlib.sha256()

    def write(self, data: bytes) -> None:
        view = memoryview(data)
        for offset in range(0, len(view), _WRITE_CHUNK_BYTES):
            chunk = view[offset : offset + _WRITE_CHUNK_BYTES]
            next_size = self.size_bytes + len(chunk)
            if next_size > self._max_object_bytes:
                raise S3ObjectSizeLimitError(next_size, self._max_object_bytes)
            self.digest.update(chunk)
            written = self._body.write(chunk)
            if written != len(chunk):
                raise OSError("temporary S3 object spool accepted a partial write")
            self.size_bytes = next_size


class _EncodedTextWriter:
    def __init__(self, writer: _BoundedBinaryWriter, encoding: str) -> None:
        self._writer = writer
        self._encoding = encoding

    def write(self, value: str) -> int:
        try:
            encoded = value.encode(self._encoding)
        except (UnicodeEncodeError, LookupError):
            raise S3RecordSerializationError from None
        self._writer.write(encoded)
        return len(value)


def _json_string_chars(value: str) -> int:
    total = 2
    for character in value:
        codepoint = ord(character)
        if character in {'"', "\\"} or character in {"\b", "\f", "\n", "\r", "\t"}:
            total += 2
        elif codepoint < 0x20:
            total += 6
        else:
            total += 1
    return total


def _json_value_chars(value: Any, *, seen: set[int]) -> int:
    if value is None:
        return 4
    if value is True:
        return 4
    if value is False:
        return 5
    if isinstance(value, str):
        return _json_string_chars(value)
    if type(value) is int:
        return len(str(value))
    if type(value) is float:
        if not math.isfinite(value):
            raise S3RecordSerializationError
        return len(json.dumps(value, allow_nan=False))
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise S3RecordSerializationError
        seen.add(identity)
        try:
            total = 2
            for index, (key, child) in enumerate(value.items()):
                if not isinstance(key, str):
                    raise S3RecordSerializationError
                if index:
                    total += 1
                total += _json_string_chars(key) + 1 + _json_value_chars(child, seen=seen)
            return total
        finally:
            seen.remove(identity)
    if isinstance(value, list | tuple):
        identity = id(value)
        if identity in seen:
            raise S3RecordSerializationError
        seen.add(identity)
        try:
            return 2 + max(0, len(value) - 1) + sum(_json_value_chars(child, seen=seen) for child in value)
        finally:
            seen.remove(identity)
    raise S3RecordSerializationError


def _check_json_record(row: Mapping[str, Any], max_record_chars: int) -> None:
    if _json_value_chars(row, seen=set()) > max_record_chars:
        raise S3RecordSizeLimitError


def _csv_scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if type(value) is bool:
        return str(value)
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise S3RecordSerializationError
        return str(value)
    raise S3RecordSerializationError


def _csv_record_chars(values: Sequence[str], delimiter: str) -> int:
    total = max(0, len(values) - 1) + 2
    for value in values:
        if delimiter in value or '"' in value or "\r" in value or "\n" in value:
            total += len(value) + value.count('"') + 2
        else:
            total += len(value)
    return total


def _serialize_rows_to_spool(
    rows: Iterable[Mapping[str, Any]],
    *,
    format: Literal["csv", "json", "jsonl"],
    csv_options: CSVWriteOptions,
    fieldnames: Sequence[str],
    max_object_bytes: int,
    max_record_chars: int,
) -> _SerializedObject:
    """Incrementally serialize rows into an owned, bounded 8 MiB spool."""
    body = cast(
        "BinaryIO",
        tempfile.SpooledTemporaryFile(max_size=_SPOOL_MEMORY_BYTES, mode="w+b"),  # noqa: SIM115 - ownership is returned
    )
    writer = _BoundedBinaryWriter(body, max_object_bytes)
    json_encoder = json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        if format == "csv":
            text_writer = _EncodedTextWriter(writer, csv_options.encoding)
            csv_writer = csv.writer(text_writer, delimiter=csv_options.delimiter, lineterminator="\r\n")
            if csv_options.include_header:
                header_values = list(fieldnames)
                if _csv_record_chars(header_values, csv_options.delimiter) > max_record_chars:
                    raise S3RecordSizeLimitError
                csv_writer.writerow(header_values)
            for row in rows:
                if set(row) - set(fieldnames):
                    raise S3RecordSerializationError
                values = [_csv_scalar_text(row.get(field)) for field in fieldnames]
                if _csv_record_chars(values, csv_options.delimiter) > max_record_chars:
                    raise S3RecordSizeLimitError
                csv_writer.writerow(values)
        elif format in {"json", "jsonl"}:
            if format == "json":
                writer.write(b"[")
            for index, row in enumerate(rows):
                _check_json_record(row, max_record_chars)
                if format == "json" and index:
                    writer.write(b",")
                try:
                    for fragment in json_encoder.iterencode(row):
                        _EncodedTextWriter(writer, "utf-8").write(fragment)
                except S3ObjectSizeLimitError:
                    raise
                except (TypeError, ValueError, UnicodeError):
                    raise S3RecordSerializationError from None
                if format == "jsonl":
                    writer.write(b"\n")
            if format == "json":
                writer.write(b"]")
        else:
            raise AssertionError(f"Unsupported S3 sink format: {format}")
        body.seek(0)
        digest = writer.digest.digest()
        return _SerializedObject(
            body=body,
            size_bytes=writer.size_bytes,
            content_hash=digest.hex(),
            checksum_sha256_b64=base64.b64encode(digest).decode("ascii"),
        )
    except BaseException:
        body.close()
        raise
