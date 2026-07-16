"""Tests for Azure Blob Storage sink plugin.

Covers config validation, lifecycle, write flow (serialization + upload + artifact),
blob path templating, overwrite protection, and audit trail recording.

Serialization boundary tests (non-finite float rejection) and close() resource
release are in test_azure_blob_sink_serialization.py -- not duplicated here.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink, AzureBlobSinkConfig
from tests.fixtures.base_classes import inject_write_failure

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FAKE_CONN_STRING = "DefaultEndpointsProtocol=https;AccountName=fake;AccountKey=ZmFrZQ==;EndpointSuffix=core.windows.net"
DYNAMIC_SCHEMA: dict[str, Any] = {"mode": "observed"}
FIXED_SCHEMA: dict[str, Any] = {"mode": "fixed", "fields": ["id: str", "name: str"]}
PATCH_AUTH = "elspeth.plugins.infrastructure.azure_auth.AzureAuthConfig.create_blob_service_client"


class _CallRecord:
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, index: int) -> tuple[Any, ...] | dict[str, Any]:
        if index == 0:
            return self.args
        if index == 1:
            return self.kwargs
        raise IndexError(index)


class _CallRecorder:
    def __init__(self) -> None:
        self.call_args: _CallRecord | None = None
        self.call_args_list: list[_CallRecord] = []
        self.side_effect: Exception | None = None

    @property
    def called(self) -> bool:
        return bool(self.call_args_list)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        record = _CallRecord(args, kwargs)
        self.call_args = record
        self.call_args_list.append(record)
        if self.side_effect is not None:
            raise self.side_effect

    def assert_called_once(self) -> None:
        assert len(self.call_args_list) == 1


class _BlobClientFake:
    def __init__(self) -> None:
        self.upload_blob = _CallRecorder()


class _ContainerClientFake:
    def __init__(self, blob_client: _BlobClientFake) -> None:
        self._blob_client = blob_client
        self.close = _CallRecorder()

    def get_blob_client(self, *_args: Any, **_kwargs: Any) -> _BlobClientFake:
        return self._blob_client


class _BlobServiceClientFake:
    def __init__(self, container_client: _ContainerClientFake) -> None:
        self._container_client = container_client

    def get_container_client(self, *_args: Any, **_kwargs: Any) -> _ContainerClientFake:
        return self._container_client


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


def _mock_blob_upload() -> tuple[_BlobServiceClientFake, _BlobClientFake]:
    """Create service/client fakes returning (service, blob_client) for upload assertions."""
    blob_client = _BlobClientFake()
    container = _ContainerClientFake(blob_client)
    service = _BlobServiceClientFake(container)
    return service, blob_client


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

    @pytest.mark.parametrize("container", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_placeholder_container_raises(self, container: str) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            AzureBlobSink(_base_config(container=container))

    @pytest.mark.parametrize("container", ["todo", "unknown", "unset", "required"])
    def test_plain_placeholder_words_can_be_container_names(self, container: str) -> None:
        cfg = AzureBlobSinkConfig.from_dict(_base_config(container=container))
        assert cfg.container == container

    def test_empty_blob_path_raises(self) -> None:
        with pytest.raises(PluginConfigError, match="blob_path"):
            AzureBlobSink(_base_config(blob_path=""))

    @pytest.mark.parametrize("blob_path", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_placeholder_blob_path_raises(self, blob_path: str) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            AzureBlobSink(_base_config(blob_path=blob_path))

    @pytest.mark.parametrize("blob_path", ["todo", "unknown", "unset", "required"])
    def test_plain_placeholder_words_can_be_blob_paths(self, blob_path: str) -> None:
        cfg = AzureBlobSinkConfig.from_dict(_base_config(blob_path=blob_path))
        assert cfg.blob_path == blob_path

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
        mock_client = _ContainerClientFake(_BlobClientFake())
        sink._container_client = mock_client

        sink.close()

        mock_client.close.assert_called_once()
        assert sink._container_client is None
        assert not hasattr(sink, "_buffered_rows")
        assert not hasattr(sink, "_resolved_blob_path")
        assert not hasattr(sink, "_has_uploaded")

    def test_close_without_client_is_safe(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        sink.close()  # Should not raise

    def test_flush_is_noop(self) -> None:
        sink = inject_write_failure(AzureBlobSink(_base_config()))
        sink.flush()  # Should not raise, returns None


# ============================================================================
# TestAzureBlobSinkWrite -- Write flow tests
# ============================================================================
