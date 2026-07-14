"""Strict, line-oriented local text sink."""

from __future__ import annotations

import codecs
import hashlib
import io
import keyword
import os
import tempfile
from pathlib import Path
from typing import IO, Any, Literal

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import ArtifactDescriptor, Determinism, PluginSchema
from elspeth.contracts.contexts import SinkContext
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.sink import OutputValidationResult
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import LocalFileSinkConfig, OutputCollisionPolicy
from elspeth.plugins.infrastructure.output_paths import validate_output_collision_policy_mode
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


class TextSinkConfig(LocalFileSinkConfig):
    """Configuration for one-field, line-oriented text output."""

    field: str = Field(description="String field written as one line per row.")
    encoding: Literal["utf-8", "ascii", "latin-1", "cp1252"] = Field(
        default="utf-8",
        description="Character encoding used for every emitted line.",
    )
    mode: Literal["write", "append"] = Field(
        default="write",
        description="Write a new output or append lines to an existing output.",
    )

    @field_validator("field")
    @classmethod
    def _validate_field(cls, value: str) -> str:
        if not value.isidentifier() or keyword.iskeyword(value):
            raise ValueError(f"field {value!r} must be a non-keyword Python identifier")
        return value

    @field_validator("encoding")
    @classmethod
    def _validate_encoding(cls, value: str) -> str:
        try:
            canonical = codecs.lookup(value).name
        except LookupError as exc:
            raise ValueError(f"unknown encoding: {value!r}") from exc
        if canonical not in {"utf-8", "ascii", "iso8859-1", "cp1252"}:
            raise ValueError("encoding must be one of utf-8, ascii, latin-1, or cp1252")
        return value

    @model_validator(mode="after")
    def _validate_collision_mode(self) -> TextSinkConfig:
        validate_output_collision_policy_mode(
            plugin_name="TextSink",
            mode=self.mode,
            collision_policy=self.collision_policy,
        )
        return self


class TextSink(BaseSink):
    """Write one configured string field per canonical LF-delimited record."""

    name = "text"
    determinism = Determinism.IO_WRITE
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:8d595e1faa8de040"
    config_model = TextSinkConfig
    supports_resume = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TextSinkConfig.from_dict(config, plugin_name=self.name)
        self._path = cfg.resolved_path()
        self._requested_path = self._path
        self._field = cfg.field
        self._encoding = cfg.encoding
        self._mode = cfg.mode
        self._collision_policy: OutputCollisionPolicy | None = cfg.collision_policy
        self._schema_config = cfg.schema_config
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "TextSinkRowSchema",
            allow_coercion=False,
        )
        self.input_schema = self._schema_class
        self.declared_required_fields = self._schema_config.get_effective_required_fields() | {self._field}
        self._file: IO[bytes] | None = None
        self._hasher: hashlib._Hash | None = None
        self._write_target_claimed = False
        self._reservation_owned = False
        self._write_has_committed = False

    def configure_for_resume(self) -> None:
        """Switch a configured write sink to canonical append mode."""
        self._mode = "append"
        self._collision_policy = "append_or_create"

    def validate_output_target(self) -> OutputValidationResult:
        """Validate that an existing target is canonical for safe append."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return OutputValidationResult.success()
        try:
            saw_content = False
            final_character = ""
            with open(self._path, encoding=self._encoding, newline="") as stream:
                for chunk in iter(lambda: stream.read(64 * 1024), ""):
                    if chunk:
                        saw_content = True
                        if "\r" in chunk:
                            return OutputValidationResult.failure(message="Existing text output contains non-canonical CR separators")
                        final_character = chunk[-1]
        except UnicodeError:
            return OutputValidationResult.failure(message=f"Existing text output is not valid {self._encoding}")
        if saw_content and final_character != "\n":
            return OutputValidationResult.failure(message="Existing text output does not end at an LF record boundary")
        return OutputValidationResult.success(target_fields=[self._field])

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        """Write accepted rows as encoded, LF-delimited bytes."""
        del ctx
        staged = self._stage_rows(rows)
        if not staged:
            return SinkWriteResult(artifact=self._describe_current_or_virtual(), diversions=self._get_diversions())

        if self._mode == "append":
            artifact = self._append_bytes(staged)
        else:
            artifact = self._replace_with_bytes(staged)
        return SinkWriteResult(artifact=artifact, diversions=self._get_diversions())

    def _stage_rows(self, rows: list[dict[str, Any]]) -> bytes:
        staged = io.BytesIO()
        missing = object()
        for row_index, row in enumerate(rows):
            value = row.get(self._field, missing)
            if type(value) is not str:
                self._divert_row(
                    row,
                    row_index=row_index,
                    reason=f"Text field {self._field!r} must be a string",
                )
                continue
            if "\r" in value or "\n" in value:
                self._divert_row(
                    row,
                    row_index=row_index,
                    reason="Text values cannot contain CR or LF record separators",
                )
                continue
            try:
                staged.write((value + "\n").encode(self._encoding))
            except UnicodeEncodeError:
                self._divert_row(
                    row,
                    row_index=row_index,
                    reason=f"Text value is not representable in configured codec {self._encoding}",
                )
        return staged.getvalue()

    def _describe_current_or_virtual(self) -> ArtifactDescriptor:
        path = self._path
        if path.exists():
            content_hash = self._hash_path(path).hexdigest()
            size = self._artifact_stat(path).st_size
        else:
            content_hash = hashlib.sha256(b"").hexdigest()
            size = 0
        return ArtifactDescriptor.for_file(path=str(path), content_hash=content_hash, size_bytes=size)

    def _replace_with_bytes(self, staged: bytes) -> ArtifactDescriptor:
        self._claim_write_target()
        temp_fd: int | None = None
        temp_path: Path | None = None
        replaced = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            base = self._path.read_bytes() if self._write_has_committed else b""
            temp_fd, temp_name = tempfile.mkstemp(prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent)
            temp_path = Path(temp_name)
            stream = os.fdopen(temp_fd, "wb")
            temp_fd = None  # fdopen transferred ownership to stream.
            with stream:
                stream.write(base)
                stream.write(staged)
                self._sync_stream(stream)

            candidate_hasher = self._hash_path(temp_path)
            candidate_stat = self._artifact_stat(temp_path)
            os.replace(temp_path, self._path)
            replaced = True
            self._reservation_owned = False
            self._write_has_committed = True
            self._hasher = candidate_hasher
            self._fsync_parent()
            return ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=candidate_hasher.hexdigest(),
                size_bytes=candidate_stat.st_size,
            )
        except BaseException:
            try:
                if temp_fd is not None:
                    os.close(temp_fd)
                if not replaced and temp_path is not None and temp_path.exists():
                    temp_path.unlink()
            finally:
                if not replaced:
                    self._remove_owned_reservation()
            raise

    def _append_bytes(self, staged: bytes) -> ArtifactDescriptor:
        self._ensure_append_open()
        handle = self._file
        if handle is None or self._hasher is None:
            raise RuntimeError("TextSink append handle was not initialized")
        handle.flush()
        offset = os.fstat(handle.fileno()).st_size
        try:
            handle.write(staged)
            self._sync_stream(handle)
            candidate_hasher = self._extend_hasher(staged)
            candidate_stat = self._artifact_stat(self._path)
        except BaseException:
            try:
                self._truncate_append(handle, offset)
            except BaseException:
                try:
                    handle.close()
                finally:
                    self._file = None
                    self._hasher = None
                raise RuntimeError(f"Text append rollback failed at byte offset {offset}") from None
            handle.close()
            self._file = None
            self._hasher = self._hash_path(self._path)
            raise

        self._hasher = candidate_hasher
        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=candidate_hasher.hexdigest(),
            size_bytes=candidate_stat.st_size,
        )

    def _claim_write_target(self) -> None:
        if self._write_target_claimed:
            return
        self._requested_path.parent.mkdir(parents=True, exist_ok=True)
        if self._collision_policy is None:
            self._path = self._requested_path
        elif self._collision_policy == "fail_if_exists":
            self._reserve(self._requested_path)
            self._path = self._requested_path
        elif self._collision_policy == "auto_increment":
            self._path = self._reserve_auto_increment()
        else:
            raise AssertionError(f"Unexpected write collision policy: {self._collision_policy!r}")
        self._write_target_claimed = True

    def _reserve(self, path: Path) -> None:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(descriptor)
        self._reservation_owned = True

    def _reserve_auto_increment(self) -> Path:
        suffix = "".join(self._requested_path.suffixes)
        stem = self._requested_path.name[: -len(suffix)] if suffix else self._requested_path.name
        candidates = (self._requested_path, *(self._requested_path.with_name(f"{stem}-{index}{suffix}") for index in range(1, 10_000)))
        for candidate in candidates:
            try:
                self._reserve(candidate)
            except FileExistsError:
                continue
            return candidate
        raise FileExistsError(f"No free output path found near {self._requested_path}.")

    def _remove_owned_reservation(self) -> None:
        if not self._reservation_owned:
            return
        try:
            if self._path.exists() and self._path.stat().st_size == 0:
                self._path.unlink()
        finally:
            self._reservation_owned = False
            self._write_target_claimed = False

    def _ensure_append_open(self) -> None:
        if self._file is not None:
            return
        validation = self.validate_output_target()
        if not validation.valid:
            raise ValueError(f"Existing text output is incompatible: {validation.error_message}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a+b")  # noqa: SIM115 - streaming lifecycle handle
        self._file.flush()
        self._hasher = self._hash_path(self._path)

    def _truncate_append(self, handle: IO[bytes], offset: int) -> None:
        handle.seek(offset)
        handle.truncate(offset)
        handle.flush()
        os.fsync(handle.fileno())

    def _hash_path(self, path: Path) -> hashlib._Hash:
        hasher = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                hasher.update(chunk)
        return hasher

    def _extend_hasher(self, staged: bytes) -> hashlib._Hash:
        if self._hasher is None:
            raise RuntimeError("TextSink hasher was not initialized")
        candidate = self._hasher.copy()
        candidate.update(staged)
        return candidate

    def _sync_stream(self, stream: IO[bytes]) -> None:
        stream.flush()
        os.fsync(stream.fileno())

    def _artifact_stat(self, path: Path) -> os.stat_result:
        return path.stat()

    def _fsync_parent(self) -> None:
        descriptor = os.open(self._path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def flush(self) -> None:
        """Flush and fsync the persistent append handle, when open."""
        if self._file is not None:
            self._sync_stream(self._file)

    def close(self) -> None:
        """Close the persistent append handle; repeated calls are safe."""
        if self._file is not None:
            self._file.close()
            self._file = None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Write one configured string field per row as canonical LF-delimited text.",
                composer_hints=(
                    "Set field to the configured field whose value should become each output line; every accepted value must be a string.",
                    "Text values containing CR or LF are rejected rather than escaped, so one input row always maps to exactly one output record.",
                    "Choose only utf-8, ascii, latin-1, or cp1252; values not representable in the configured encoding are diverted without leaking their content.",
                    "Use mode: append with collision_policy: append_or_create for append/resume; existing bytes must decode cleanly and end with canonical LF.",
                    "Text is not a generic failure sink because it deliberately writes only one configured field and cannot preserve a rejected row losslessly.",
                ),
            )
        return None
