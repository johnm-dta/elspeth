"""Tests for Azure Blob Storage sink plugin.

Covers config validation, lifecycle, write flow (serialization + upload + artifact),
blob path templating, overwrite protection, and audit trail recording.

Serialization boundary tests (non-finite float rejection) and close() resource
release are in test_azure_blob_sink_serialization.py -- not duplicated here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA: dict[str, Any] = {"mode": "observed"}
FIXED_SCHEMA: dict[str, Any] = {"mode": "fixed", "fields": ["id: str", "name: str"]}
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Minimal valid config dict -- connection_string auth, observed schema."""
    cfg: dict[str, Any] = {
        "connection_string": FAKE_CONN_STRING,
        "container": "test-container",
        "blob_path": "output.csv",
        "schema": DYNAMIC_SCHEMA,
    }
    cfg.update(overrides)
    return cfg


def _mock_blob_upload() -> tuple[MagicMock, MagicMock]:
    """Create mock service client returning (service, blob_client) for upload assertions."""
    mock_blob_client = MagicMock()
    mock_container = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob_client
    mock_container.close = MagicMock()
    mock_service = MagicMock()
    mock_service.get_container_client.return_value = mock_container
    return mock_service, mock_blob_client


def _make_sink_ctx() -> PluginContext:
    """Build a PluginContext suitable for sink.write() calls."""
    from tests.fixtures.factories import make_operation_context

    return make_operation_context(
        operation_type="sink_write",
        node_id="sink",
        node_type="SINK",
        plugin_name="azure_blob",
    )


# ============================================================================
# TestAzureBlobSinkConfig -- Config validation (no Azure SDK calls)
# ============================================================================


class TestAzureBlobSinkConfig:
    """Config validation -- no Azure SDK calls needed."""

    def test_connection_string_auth_sets_name(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        assert sink.name == "azure_blob"

    def test_sas_token_auth_sets_method(self) -> None:
        sink = inject_write_failure(
            AzureBlobSink(
                _base_config(
                    connection_string=None,
                    sas_token="sv=2021-06-08&ss=b&srt=sco&se=2099-01-01",
                    account_url="https://fake.blob.core.windows.net",
                )
            )
        )
        assert sink._auth_config.auth_method == "sas_token"

    def test_no_auth_raises(self) -> None:
        with pytest.raises(PluginConfigError, match="authentication"):
            AzureBlobSink(_base_config(connection_string=None))

    def test_empty_container_raises(self) -> None:
        with pytest.raises(PluginConfigError, match="container"):
            AzureBlobSink(_base_config(container=""))

    def test_whitespace_container_raises(self) -> None:
        with pytest.raises(PluginConfigError, match="container"):
            AzureBlobSink(_base_config(container="   "))

    def test_empty_blob_path_raises(self) -> None:
        with pytest.raises(PluginConfigError, match="blob_path"):
            AzureBlobSink(_base_config(blob_path=""))

    def test_invalid_template_syntax_raises_value_error(self) -> None:
        with pytest.raises(PluginConfigError, match="Invalid blob_path template"):
            AzureBlobSink(_base_config(blob_path="{{ unclosed"))

    def test_csv_delimiter_must_be_single_char(self) -> None:
        with pytest.raises(PluginConfigError, match="single character"):
            AzureBlobSink(_base_config(csv_options={"delimiter": ";;"}))

    def test_resume_not_supported_flag(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        assert sink.supports_resume is False

    def test_configure_for_resume_raises(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        with pytest.raises(NotImplementedError, match="does not support resume"):
            sink.configure_for_resume()


# ============================================================================
# TestAzureBlobSinkLifecycle
# ============================================================================


class TestAzureBlobSinkLifecycle:
    """Lifecycle -- close resets state, flush is noop."""

    def test_close_resets_all_state(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        # Simulate some state
        mock_client = MagicMock()
        sink._container_client = mock_client
        sink._buffered_rows = [{"id": "1"}]
        sink._resolved_blob_path = "some/path.csv"
        sink._has_uploaded = True

        sink.close()

        mock_client.close.assert_called_once()
        assert sink._container_client is None
        assert sink._buffered_rows == []  # type: ignore[unreachable]  # close() mutates _container_client; mypy can't track cross-method state change
        assert sink._resolved_blob_path is None
        assert sink._has_uploaded is False

    def test_close_without_client_is_safe(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        sink.close()  # Should not raise

    def test_flush_is_noop(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        sink.flush()  # Should not raise, returns None


# ============================================================================
# TestAzureBlobSinkWrite -- Write flow tests
# ============================================================================


class TestAzureBlobSinkWrite:
    """Write flow -- serialization, upload, artifact descriptors."""

    def test_write_csv_uploads_correct_content(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="csv", schema=FIXED_SCHEMA)))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1", "name": "alice"}, {"id": "2", "name": "bob"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0]
        reader = csv.DictReader(io.StringIO(uploaded.decode("utf-8")))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["id"] == "1"
        assert rows[1]["name"] == "bob"

    def test_write_json_uploads_array(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="json")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1"}, {"id": "2"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0]
        parsed = json.loads(uploaded)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_write_jsonl_uploads_lines(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="jsonl")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1"}, {"id": "2"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0]
        lines = uploaded.decode("utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"id": "1"}

    def test_write_returns_artifact_descriptor(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="csv", schema=FIXED_SCHEMA)))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            result = sink.write([{"id": "1", "name": "alice"}], ctx)

        artifact = result.artifact
        assert artifact.artifact_type == "file"
        assert artifact.path_or_uri.startswith("azure://test-container/")
        assert artifact.content_hash  # non-empty
        assert artifact.size_bytes > 0

    def test_content_hash_is_sha256_of_uploaded_bytes(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="json")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            result = sink.write([{"id": "1"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0]
        expected_hash = hashlib.sha256(uploaded).hexdigest()
        assert result.artifact.content_hash == expected_hash

    def test_empty_rows_no_upload(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="json")))
        ctx = _make_sink_ctx()

        # No patching needed -- empty rows should not touch Azure at all
        result = sink.write([], ctx)

        assert result.artifact.size_bytes == 0
        assert result.artifact.content_hash == hashlib.sha256(b"").hexdigest()

    def test_cumulative_buffering(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="json")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1"}], ctx)
            sink.write([{"id": "2"}], ctx)

        # Second upload should contain both rows
        uploaded = mock_blob.upload_blob.call_args[0][0]
        parsed = json.loads(uploaded)
        assert len(parsed) == 2
        assert parsed[0]["id"] == "1"
        assert parsed[1]["id"] == "2"

    def test_csv_custom_delimiter(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="csv", schema=FIXED_SCHEMA, csv_options={"delimiter": "|"})))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1", "name": "alice"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0].decode("utf-8")
        assert "|" in uploaded

    def test_csv_no_header(self) -> None:
        sink = inject_write_failure(
            AzureBlobSink(
                _base_config(
                    format="csv",
                    schema=FIXED_SCHEMA,
                    csv_options={"include_header": False},
                )
            )
        )
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": "1", "name": "alice"}], ctx)

        uploaded = mock_blob.upload_blob.call_args[0][0].decode("utf-8")
        lines = uploaded.strip().split("\n")
        # Only data row, no header
        assert len(lines) == 1
        assert "id" not in lines[0] or lines[0].startswith("1")


# ============================================================================
# TestAzureBlobSinkTemplateAndOverwrite -- Blob path + overwrite protection
# ============================================================================


class TestAzureBlobSinkTemplateAndOverwrite:
    """Blob path templating and overwrite protection."""

    def test_template_renders_run_id(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(blob_path="results/{{ run_id }}/out.csv")))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            result = sink.write([{"x": 1}], ctx)

        assert ctx.run_id in result.artifact.path_or_uri

    def test_template_renders_timestamp(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(blob_path="results/{{ timestamp }}/out.csv")))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            result = sink.write([{"x": 1}], ctx)

        # ISO timestamps contain 'T'
        assert "T" in result.artifact.path_or_uri

    def test_blob_path_frozen_after_first_write(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(blob_path="results/{{ run_id }}/out.csv")))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            r1 = sink.write([{"x": 1}], ctx)
            r2 = sink.write([{"x": 2}], ctx)

        assert r1.artifact.path_or_uri == r2.artifact.path_or_uri

    def test_undefined_template_var_raises(self) -> None:
        from jinja2 import UndefinedError

        sink = inject_write_failure(AzureBlobSink(_base_config(blob_path="{{ nonexistent_var }}/out.csv")))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(UndefinedError):
            sink.write([{"x": 1}], ctx)

    def test_overwrite_true_passes_flag(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(overwrite=True)))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"x": 1}], ctx)

        assert mock_blob.upload_blob.call_args.kwargs["overwrite"] is True

    def test_overwrite_false_first_write_sends_false(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(overwrite=False)))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"x": 1}], ctx)

        assert mock_blob.upload_blob.call_args.kwargs["overwrite"] is False

    def test_overwrite_false_second_write_sends_true(self) -> None:
        """In-run rewrite of the same blob is allowed (cumulative buffering)."""
        sink = inject_write_failure(AzureBlobSink(_base_config(overwrite=False)))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"x": 1}], ctx)
            sink.write([{"x": 2}], ctx)

        # Second call should have overwrite=True
        calls = mock_blob.upload_blob.call_args_list
        assert calls[1].kwargs["overwrite"] is True

    def test_resource_exists_error_converted_to_value_error(self) -> None:
        # String-based check in production code: type(e).__name__ == "ResourceExistsError"
        class ResourceExistsError(Exception):
            pass

        sink = inject_write_failure(AzureBlobSink(_base_config(overwrite=False)))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()
        mock_blob.upload_blob.side_effect = ResourceExistsError("blob exists")

        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(ValueError, match="already exists"):
            sink.write([{"x": 1}], ctx)


# ============================================================================
# TestAzureBlobSinkAudit -- Audit trail recording
# ============================================================================


class TestAzureBlobSinkAudit:
    """Audit trail recording and error propagation."""

    def test_upload_failure_raises_runtime_error(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()
        mock_blob.upload_blob.side_effect = ConnectionError("network down")

        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(RuntimeError, match="Failed to upload blob"):
            sink.write([{"x": 1}], ctx)

    def test_audit_integrity_error_on_record_call_failure(self) -> None:
        from elspeth.contracts.errors import AuditIntegrityError

        sink = inject_write_failure(AzureBlobSink(_base_config()))
        ctx = _make_sink_ctx()
        mock_service, _mock_blob = _mock_blob_upload()

        # Upload succeeds, but record_call fails after success
        def failing_record_call(**kwargs: Any) -> Any:
            raise RuntimeError("DB write failed")

        ctx.record_call = failing_record_call  # type: ignore[method-assign,assignment]  # intentional monkey-patch for test

        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(AuditIntegrityError, match="Failed to record"):
            sink.write([{"x": 1}], ctx)

    def test_programming_errors_crash_directly(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()
        mock_blob.upload_blob.side_effect = AttributeError("mock has no attr")

        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(AttributeError, match="mock has no attr"):
            sink.write([{"x": 1}], ctx)

    def test_buffer_not_committed_on_upload_failure(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config(format="json")))
        ctx = _make_sink_ctx()
        mock_service, mock_blob = _mock_blob_upload()

        # First write succeeds — buffer grows to [row1]
        with patch(PATCH_AUTH, return_value=mock_service):
            sink.write([{"id": 1}], ctx)
        assert len(sink._buffered_rows) == 1

        # Second write fails — buffer must stay at [row1], not grow to [row1, row2]
        mock_blob.upload_blob.side_effect = ConnectionError("network down")
        with patch(PATCH_AUTH, return_value=mock_service), pytest.raises(RuntimeError):
            sink.write([{"id": 2}], ctx)

        assert len(sink._buffered_rows) == 1
        assert sink._buffered_rows[0]["id"] == 1
