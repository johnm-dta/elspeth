"""Tests for Dataverse sink plugin."""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts.errors import SinkTransactionalInvariantError
from elspeth.core.canonical import canonical_json
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseClientError,
    DataversePageResponse,
)
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.sinks.dataverse import DataverseSink, DataverseSinkConfig
from tests.fixtures.base_classes import inject_write_failure

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DYNAMIC_SCHEMA = {"mode": "observed"}

_BASE_AUTH = {
    "method": "service_principal",
    "tenant_id": "tenant-1",
    "client_id": "client-1",
    "client_secret": "secret-1",
}

# alternate_key must be a Dataverse column name (a *value* in field_mapping)
_BASE_CONFIG: dict[str, Any] = {
    "environment_url": "https://myorg.crm.dynamics.com",
    "auth": _BASE_AUTH,
    "entity": "contacts",
    "alternate_key": "emailaddress1",
    "field_mapping": {"email": "emailaddress1", "name": "fullname"},
    "schema": DYNAMIC_SCHEMA,
}


def _config(**overrides: Any) -> dict[str, Any]:
    """Return a base config dict with optional overrides."""
    cfg = dict(_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


def _make_204_response() -> DataversePageResponse:
    """Create a typical 204-No-Content upsert response."""
    return DataversePageResponse(
        status_code=204,
        rows=[],
        latency_ms=12.0,
        headers={"content-length": "0"},
        request_headers={"Authorization": "Bearer fake-token"},
        request_url="https://myorg.crm.dynamics.com/api/data/v9.2/contacts",
        next_link=None,
        paging_cookie=None,
        more_records=None,  # No body → no morerecords field
    )


# ---------------------------------------------------------------------------
# DataverseSinkConfig validation
# ---------------------------------------------------------------------------


class TestDataverseSinkConfig:
    """Tests for DataverseSinkConfig Pydantic validation."""

    def test_valid_config(self) -> None:
        cfg = DataverseSinkConfig.from_dict(_config())
        assert cfg.entity == "contacts"
        assert cfg.alternate_key == "emailaddress1"
        assert cfg.field_mapping == {"email": "emailaddress1", "name": "fullname"}
        assert cfg.mode == "upsert"

    def test_alternate_key_required(self) -> None:
        with pytest.raises(PluginConfigError, match="alternate_key"):
            DataverseSinkConfig.from_dict(_config(alternate_key=""))

    def test_alternate_key_whitespace_only(self) -> None:
        with pytest.raises(PluginConfigError, match="alternate_key"):
            DataverseSinkConfig.from_dict(_config(alternate_key="   "))

    def test_entity_required(self) -> None:
        with pytest.raises(PluginConfigError, match="entity"):
            DataverseSinkConfig.from_dict(_config(entity=""))

    def test_entity_whitespace_only(self) -> None:
        with pytest.raises(PluginConfigError, match="entity"):
            DataverseSinkConfig.from_dict(_config(entity="  "))

    def test_entity_stripped(self) -> None:
        cfg = DataverseSinkConfig.from_dict(_config(entity="  contacts  "))
        assert cfg.entity == "contacts"

    def test_field_mapping_required(self) -> None:
        c = _config()
        del c["field_mapping"]
        with pytest.raises(PluginConfigError, match="field_mapping"):
            DataverseSinkConfig.from_dict(c)

    def test_https_enforcement(self) -> None:
        with pytest.raises(PluginConfigError, match="HTTPS"):
            DataverseSinkConfig.from_dict(_config(environment_url="http://myorg.crm.dynamics.com"))

    def test_https_enforcement_no_scheme(self) -> None:
        with pytest.raises(PluginConfigError):
            DataverseSinkConfig.from_dict(_config(environment_url="myorg.crm.dynamics.com"))

    def test_lookup_config_valid(self) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                lookups={
                    "account_id": {
                        "target_entity": "accounts",
                        "target_field": "parentcustomerid",
                    }
                }
            )
        )
        assert cfg.lookups is not None
        assert "account_id" in cfg.lookups
        assert cfg.lookups["account_id"].target_entity == "accounts"

    def test_lookup_target_entity_required(self) -> None:
        with pytest.raises(PluginConfigError, match="target_entity"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "",
                            "target_field": "parentcustomerid",
                        }
                    }
                )
            )

    def test_lookup_target_entity_whitespace_only(self) -> None:
        with pytest.raises(PluginConfigError, match="target_entity"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "   ",
                            "target_field": "parentcustomerid",
                        }
                    }
                )
            )

    def test_lookup_target_entity_stripped(self) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                lookups={
                    "account_id": {
                        "target_entity": "  accounts  ",
                        "target_field": "parentcustomerid",
                    }
                }
            )
        )

        assert cfg.lookups is not None
        assert cfg.lookups["account_id"].target_entity == "accounts"

    def test_lookup_target_field_required(self) -> None:
        with pytest.raises(PluginConfigError, match="target_field"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "accounts",
                            "target_field": "",
                        }
                    }
                )
            )

    def test_lookup_target_field_whitespace_only(self) -> None:
        with pytest.raises(PluginConfigError, match="target_field"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "accounts",
                            "target_field": "   ",
                        }
                    }
                )
            )

    def test_lookup_target_field_stripped(self) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                lookups={
                    "account_id": {
                        "target_entity": "accounts",
                        "target_field": "  parentcustomerid  ",
                    }
                }
            )
        )

        assert cfg.lookups is not None
        assert cfg.lookups["account_id"].target_field == "parentcustomerid"

    def test_lookup_config_rejects_extra_fields(self) -> None:
        with pytest.raises(PluginConfigError):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "accounts",
                            "target_field": "parentcustomerid",
                            "bogus_field": "nope",
                        }
                    }
                )
            )

    def test_additional_domains_valid(self) -> None:
        cfg = DataverseSinkConfig.from_dict(_config(additional_domains=["*.sub.crm15.dynamics.com"]))
        assert cfg.additional_domains == ["*.sub.crm15.dynamics.com"]

    def test_additional_domains_rejects_non_microsoft(self) -> None:
        with pytest.raises(PluginConfigError, match="rejected"):
            DataverseSinkConfig.from_dict(_config(additional_domains=["*.evil.example.com"]))

    def test_schema_required(self) -> None:
        c = _config()
        del c["schema"]
        with pytest.raises(PluginConfigError, match="schema"):
            DataverseSinkConfig.from_dict(c)

    def test_default_api_version(self) -> None:
        cfg = DataverseSinkConfig.from_dict(_config())
        assert cfg.api_version == "v9.2"

    def test_custom_api_version(self) -> None:
        cfg = DataverseSinkConfig.from_dict(_config(api_version="v9.1"))
        assert cfg.api_version == "v9.1"


# ---------------------------------------------------------------------------
# DataverseSink __init__ validation
# ---------------------------------------------------------------------------


class TestDataverseSinkInit:
    """Tests for DataverseSink constructor validation."""

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_alternate_key_not_in_field_mapping_values_raises(self, _mock_schema: MagicMock) -> None:
        """alternate_key must be a Dataverse column that appears as a value in field_mapping."""
        with pytest.raises(PluginConfigError, match="not found in field_mapping values"):
            DataverseSink(_config(alternate_key="nonexistent_column"))

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_alternate_key_pipeline_field_resolved(self, _mock_schema: MagicMock) -> None:
        """The pipeline field for the alternate key should be resolved from field_mapping."""
        sink = inject_write_failure(DataverseSink(_config()))
        # alternate_key is "emailaddress1" (Dataverse column)
        # field_mapping maps "email" (pipeline) -> "emailaddress1" (Dataverse)
        # So _alternate_key_pipeline_field should be "email"
        assert sink._alternate_key_pipeline_field == "email"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_observed_schema_required_fields_populate_declared_required_fields(self, _mock_schema: MagicMock) -> None:
        """Observed-mode required_fields must become the sink boundary contract."""
        sink = inject_write_failure(
            DataverseSink(
                _config(
                    schema={
                        "mode": "observed",
                        "required_fields": ["must_exist_for_contract"],
                    }
                )
            )
        )

        assert sink.declared_required_fields == frozenset({"must_exist_for_contract"})

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_observed_schema_required_fields_fail_sink_boundary(self, _mock_schema: MagicMock) -> None:
        """Missing observed required fields must fail before Dataverse write I/O."""
        sink = inject_write_failure(
            DataverseSink(
                _config(
                    schema={
                        "mode": "observed",
                        "required_fields": ["must_exist_for_contract"],
                    }
                )
            )
        )

        with pytest.raises(SinkTransactionalInvariantError, match="must_exist_for_contract"):
            SinkExecutor._validate_sink_input(
                sink,
                [{"email": "alice@example.com", "name": "Alice"}],
                skip_schema=True,
            )


# ---------------------------------------------------------------------------
# Field mapping and lookup binding
# ---------------------------------------------------------------------------


class TestFieldMappingAndLookups:
    """Tests for _map_row field mapping and lookup binding syntax."""

    def _make_sink(self, **overrides: Any) -> DataverseSink:
        """Create a DataverseSink without calling on_start."""
        with patch(
            "elspeth.plugins.sinks.dataverse.create_schema_from_config",
            return_value=MagicMock(),
        ):
            return inject_write_failure(DataverseSink(_config(**overrides)))

    def test_simple_field_mapping(self) -> None:
        sink = self._make_sink()
        row = {"email": "a@b.com", "name": "Alice"}
        payload = sink._map_row(row)
        assert payload == {"emailaddress1": "a@b.com", "fullname": "Alice"}

    def test_lookup_bind_syntax(self) -> None:
        sink = self._make_sink(
            field_mapping={
                "email": "emailaddress1",
                "account_id": "ignored_column",
            },
            lookups={
                "account_id": {
                    "target_entity": "accounts",
                    "target_field": "parentcustomerid",
                }
            },
        )
        row = {"email": "a@b.com", "account_id": "some-guid-123"}
        payload = sink._map_row(row)
        assert payload["emailaddress1"] == "a@b.com"
        assert payload["parentcustomerid@odata.bind"] == "/accounts(some-guid-123)"
        # The mapped column name should NOT appear -- it's replaced by the bind key
        assert "ignored_column" not in payload

    def test_lookup_none_value_excluded(self) -> None:
        sink = self._make_sink(
            field_mapping={
                "email": "emailaddress1",
                "account_id": "ignored_column",
            },
            lookups={
                "account_id": {
                    "target_entity": "accounts",
                    "target_field": "parentcustomerid",
                }
            },
        )
        row = {"email": "a@b.com", "account_id": None}
        payload = sink._map_row(row)
        assert "parentcustomerid@odata.bind" not in payload
        assert "ignored_column" not in payload

    def test_missing_field_raises_key_error(self) -> None:
        sink = self._make_sink()
        row = {"email": "a@b.com"}  # missing "name"
        with pytest.raises(KeyError, match="name"):
            sink._map_row(row)


# ---------------------------------------------------------------------------
# URL encoding of alternate key values
# ---------------------------------------------------------------------------


class TestBuildUpsertUrl:
    """Tests for _build_upsert_url URL encoding."""

    def _make_sink(self) -> DataverseSink:
        with patch(
            "elspeth.plugins.sinks.dataverse.create_schema_from_config",
            return_value=MagicMock(),
        ):
            return inject_write_failure(DataverseSink(_config()))

    def test_normal_value(self) -> None:
        sink = self._make_sink()
        url = sink._build_upsert_url("alice@example.com")
        assert url == ("https://myorg.crm.dynamics.com/api/data/v9.2/contacts(emailaddress1='alice%40example.com')")

    def test_special_characters(self) -> None:
        sink = self._make_sink()
        url = sink._build_upsert_url("a/b(c)=d")
        # All special chars should be percent-encoded
        assert "%2F" in url  # /
        assert "%28" in url  # (
        assert "%29" in url  # )
        assert "%3D" in url  # =

    def test_simple_string(self) -> None:
        sink = self._make_sink()
        url = sink._build_upsert_url("simple123")
        assert "simple123" in url
        assert "emailaddress1='simple123'" in url


# ---------------------------------------------------------------------------
# ArtifactDescriptor construction
# ---------------------------------------------------------------------------


class TestArtifactDescriptor:
    """Tests for ArtifactDescriptor construction in write()."""

    def _make_sink_with_mock_client(self) -> tuple[DataverseSink, MagicMock]:
        with patch(
            "elspeth.plugins.sinks.dataverse.create_schema_from_config",
            return_value=MagicMock(),
        ):
            sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client
        return sink, mock_client

    def _make_mock_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.record_call = MagicMock()
        ctx.run_id = "test-run-123"
        return ctx

    def test_empty_rows_returns_empty_hash(self) -> None:
        sink, _ = self._make_sink_with_mock_client()
        ctx = self._make_mock_ctx()

        descriptor = sink.write([], ctx)

        assert descriptor.artifact.artifact_type == "webhook"
        assert descriptor.artifact.content_hash == hashlib.sha256(b"").hexdigest()
        assert descriptor.artifact.size_bytes == 0
        assert descriptor.artifact.metadata is not None
        assert descriptor.artifact.metadata["row_count"] == 0
        assert descriptor.artifact.metadata["entity"] == "contacts"
        assert "dataverse://contacts@" in descriptor.artifact.path_or_uri

    def test_non_empty_rows_returns_correct_descriptor(self) -> None:
        sink, _ = self._make_sink_with_mock_client()
        ctx = self._make_mock_ctx()

        rows = [
            {"email": "a@b.com", "name": "Alice"},
            {"email": "c@d.com", "name": "Bob"},
        ]
        descriptor = sink.write(rows, ctx)

        # Hash should cover the mapped payloads (what was actually sent to
        # Dataverse), not the full pipeline rows.
        mapped_payloads = [
            {"emailaddress1": "a@b.com", "fullname": "Alice"},
            {"emailaddress1": "c@d.com", "fullname": "Bob"},
        ]
        expected_canonical = canonical_json(mapped_payloads).encode("utf-8")
        expected_hash = hashlib.sha256(expected_canonical).hexdigest()

        assert descriptor.artifact.artifact_type == "webhook"
        assert descriptor.artifact.content_hash == expected_hash
        assert descriptor.artifact.size_bytes == len(expected_canonical)
        assert descriptor.artifact.metadata is not None
        assert descriptor.artifact.metadata["row_count"] == 2
        assert descriptor.artifact.metadata["entity"] == "contacts"
        assert descriptor.artifact.metadata["mode"] == "upsert"
        assert "dataverse://contacts@" in descriptor.artifact.path_or_uri


# ---------------------------------------------------------------------------
# Write lifecycle
# ---------------------------------------------------------------------------


class TestWriteLifecycle:
    """Tests for on_start and write lifecycle."""

    def _make_mock_ctx(self) -> MagicMock:
        ctx = MagicMock()
        ctx.record_call = MagicMock()
        ctx.run_id = "test-run-123"
        return ctx

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    @patch("elspeth.plugins.sinks.dataverse.DataverseClient")
    @patch("azure.identity.ClientSecretCredential")
    def test_on_start_constructs_credential_and_client(
        self, mock_cred_cls: MagicMock, mock_client_cls: MagicMock, _mock_schema: MagicMock
    ) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_lifecycle = MagicMock()
        mock_lifecycle.run_id = "test-run-123"
        mock_lifecycle.telemetry_emit = MagicMock()
        mock_lifecycle.rate_limit_registry = None

        sink.on_start(mock_lifecycle)

        mock_cred_cls.assert_called_once_with(
            tenant_id="tenant-1",
            client_id="client-1",
            client_secret="secret-1",
        )
        mock_client_cls.assert_called_once()
        assert sink._client is not None
        assert sink._run_id == "test-run-123"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    @patch("elspeth.plugins.sinks.dataverse.DataverseClient")
    @patch("azure.identity.ManagedIdentityCredential")
    def test_on_start_managed_identity(self, mock_mi_cls: MagicMock, _mock_client_cls: MagicMock, _mock_schema: MagicMock) -> None:
        cfg = _config(auth={"method": "managed_identity"})
        sink = inject_write_failure(DataverseSink(cfg))

        mock_lifecycle = MagicMock()
        mock_lifecycle.run_id = "run-mi"
        mock_lifecycle.telemetry_emit = MagicMock()
        mock_lifecycle.rate_limit_registry = None

        sink.on_start(mock_lifecycle)

        mock_mi_cls.assert_called_once()

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_write_processes_rows_serially(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        rows = [
            {"email": "a@b.com", "name": "Alice"},
            {"email": "c@d.com", "name": "Bob"},
            {"email": "e@f.com", "name": "Charlie"},
        ]

        sink.write(rows, ctx)

        # Each row should trigger one upsert call
        assert mock_client.upsert.call_count == 3

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_each_row_gets_record_call(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        rows = [
            {"email": "a@b.com", "name": "Alice"},
            {"email": "c@d.com", "name": "Bob"},
        ]

        sink.write(rows, ctx)

        # record_call should be invoked once per row
        assert ctx.record_call.call_count == 2

        # All calls should be SUCCESS
        for call in ctx.record_call.call_args_list:
            assert call.kwargs["status"].value == "success"
            assert call.kwargs["provider"] == "dataverse"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_error_raises_runtime_error(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.side_effect = DataverseClientError(
            "Bad request (400)",
            retryable=False,
            status_code=400,
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        rows = [{"email": "a@b.com", "name": "Alice"}]

        with pytest.raises(DataverseClientError, match="Bad request"):
            sink.write(rows, ctx)

        # Error call should still be recorded
        assert ctx.record_call.call_count == 1
        error_call = ctx.record_call.call_args_list[0]
        assert error_call.kwargs["status"].value == "error"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_error_records_error_details(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.side_effect = DataverseClientError(
            "Server error (500)",
            retryable=True,
            status_code=500,
        )
        mock_client.get_auth_headers.return_value = {}
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        rows = [{"email": "a@b.com", "name": "Alice"}]

        with pytest.raises(DataverseClientError):
            sink.write(rows, ctx)

        error_call = ctx.record_call.call_args_list[0]
        error_data = error_call.kwargs["error"]
        assert error_data["status_code"] == 500
        assert error_data["retryable"] is True
        assert error_data["error_type"] == "DataverseClientError"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_record_call_includes_url_and_method(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        rows = [{"email": "a@b.com", "name": "Alice"}]
        sink.write(rows, ctx)

        call_kwargs = ctx.record_call.call_args_list[0].kwargs
        assert call_kwargs["request_data"]["method"] == "PATCH"
        assert "a%40b.com" in call_kwargs["request_data"]["url"]
        assert call_kwargs["response_data"]["status_code"] == 204

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_success_audit_preserves_client_fingerprinted_request_headers(
        self, _mock_schema: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DataverseClient returns audit-ready request_headers; the sink must not fingerprint them again."""
        monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "dataverse-sink-test-key")
        monkeypatch.setenv("ELSPETH_ALLOW_RAW_SECRETS", "false")
        sink = inject_write_failure(DataverseSink(_config()))

        fingerprinted_headers = {
            "Authorization": "<fingerprint:client-produced-fingerprint>",
            "Accept": "application/json",
        }
        mock_client = MagicMock()
        mock_client.upsert.return_value = DataversePageResponse(
            status_code=204,
            rows=[],
            latency_ms=12.0,
            headers={"content-length": "0"},
            request_headers=fingerprinted_headers,
            request_url="https://myorg.crm.dynamics.com/api/data/v9.2/contacts",
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )
        sink._client = mock_client

        ctx = self._make_mock_ctx()
        sink.write([{"email": "a@b.com", "name": "Alice"}], ctx)

        call_kwargs = ctx.record_call.call_args_list[0].kwargs
        assert call_kwargs["request_data"]["headers"] == fingerprinted_headers

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_empty_alternate_key_value_raises(self, _mock_schema: MagicMock) -> None:
        """Empty string key value is caught by offensive guard."""
        sink = inject_write_failure(DataverseSink(_config()))
        sink._client = MagicMock()

        ctx = self._make_mock_ctx()

        with pytest.raises(ValueError, match="empty or non-string value"):
            sink.write([{"email": "", "name": "Alice"}], ctx)

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_none_alternate_key_value_raises(self, _mock_schema: MagicMock) -> None:
        """None key value is caught by offensive guard."""
        sink = inject_write_failure(DataverseSink(_config()))
        sink._client = MagicMock()

        ctx = self._make_mock_ctx()

        with pytest.raises(ValueError, match="empty or non-string value"):
            sink.write([{"email": None, "name": "Alice"}], ctx)

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_numeric_alternate_key_value_raises(self, _mock_schema: MagicMock) -> None:
        """Numeric key value is caught by offensive guard (must be string for URL)."""
        sink = inject_write_failure(DataverseSink(_config()))
        sink._client = MagicMock()

        ctx = self._make_mock_ctx()

        with pytest.raises(ValueError, match="empty or non-string value"):
            sink.write([{"email": 42, "name": "Alice"}], ctx)

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_flush_is_noop(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        sink.flush()  # Should not raise

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_close_releases_client(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        mock_client = MagicMock()
        sink._client = mock_client

        sink.close()

        mock_client.close.assert_called_once()
        assert sink._client is None

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_close_without_client_is_safe(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        assert sink._client is None
        sink.close()  # Should not raise


# ---------------------------------------------------------------------------
# Bug fix: idempotent flag (elspeth-1453d7cfa8)
# ---------------------------------------------------------------------------


class TestIdempotentFlag:
    """Sink idempotent flag must be True for PATCH upsert mode."""

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_idempotent_is_true(self, _mock_schema: MagicMock) -> None:
        """PATCH upsert is idempotent — safe for retries and crash recovery."""
        sink = inject_write_failure(DataverseSink(_config()))
        assert sink.idempotent is True

    def test_non_upsert_mode_rejected(self) -> None:
        """Config rejects modes other than 'upsert' (Literal['upsert'])."""
        with pytest.raises(PluginConfigError):
            DataverseSinkConfig.from_dict(_config(mode="create"))


# ---------------------------------------------------------------------------
# Bug fix: request_data records JSON payload (elspeth review finding)
# ---------------------------------------------------------------------------


class TestRequestDataRecordsJsonPayload:
    """Verify that record_call request_data contains 'json': payload, not 'field_names'."""

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_request_data_contains_json_payload(self, _mock_schema: MagicMock) -> None:
        """request_data must contain 'json' key with the mapped payload dict."""
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = MagicMock()
        ctx.record_call = MagicMock()
        ctx.run_id = "test-run-123"

        rows = [{"email": "alice@example.com", "name": "Alice"}]
        sink.write(rows, ctx)

        call_kwargs = ctx.record_call.call_args_list[0].kwargs
        request_data = call_kwargs["request_data"]

        # "json" key must exist and contain the mapped payload
        assert "json" in request_data
        expected_payload = {"emailaddress1": "alice@example.com", "fullname": "Alice"}
        assert request_data["json"] == expected_payload

        # Old format "field_names" must NOT exist
        assert "field_names" not in request_data

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=MagicMock())
    def test_error_request_data_also_contains_json(self, _mock_schema: MagicMock) -> None:
        """Even on error, request_data must contain 'json' with the mapped payload."""
        sink = inject_write_failure(DataverseSink(_config()))

        mock_client = MagicMock()
        mock_client.upsert.side_effect = DataverseClientError(
            "Bad request (400)",
            retryable=False,
            status_code=400,
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client

        ctx = MagicMock()
        ctx.record_call = MagicMock()
        ctx.run_id = "test-run-123"

        rows = [{"email": "alice@example.com", "name": "Alice"}]
        with pytest.raises(DataverseClientError):
            sink.write(rows, ctx)

        call_kwargs = ctx.record_call.call_args_list[0].kwargs
        request_data = call_kwargs["request_data"]

        assert "json" in request_data
        expected_payload = {"emailaddress1": "alice@example.com", "fullname": "Alice"}
        assert request_data["json"] == expected_payload
        assert "field_names" not in request_data
