"""Tests for Dataverse sink plugin."""

from __future__ import annotations

import hashlib
import urllib.parse
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, create_autospec, patch

import pytest

from elspeth.contracts import PluginSchema
from elspeth.contracts.contexts import LifecycleContext, SinkContext
from elspeth.contracts.errors import SinkTransactionalInvariantError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json as contract_canonical_json
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseClient,
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


class FakeDataverseRowSchema(PluginSchema):
    """Schema factory return value for tests that do not exercise schema coercion."""


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


def _make_dataverse_client_double() -> Any:
    """Create an interaction double specced to the real Dataverse client."""
    return create_autospec(DataverseClient, instance=True, spec_set=True)


def _make_sink_context_double() -> Any:
    """Create a sink context double with a specced record_call method."""
    ctx = create_autospec(SinkContext, instance=True, spec_set=True)
    ctx.run_id = "test-run-123"
    ctx.contract = None
    ctx.landscape = None
    ctx.operation_id = "op-001"
    return ctx


def _make_lifecycle_context_double(*, run_id: str = "test-run-123") -> Any:
    """Create a lifecycle context double for on_start tests."""
    ctx = create_autospec(LifecycleContext, instance=True, spec_set=True)
    ctx.run_id = run_id
    ctx.node_id = "sink-node"
    ctx.operation_id = "op-001"
    ctx.landscape = None
    ctx.payload_store = None
    ctx.rate_limit_registry = None
    ctx.telemetry_emit = lambda event: None
    ctx.concurrency_config = None
    ctx.shutdown_event = None
    return ctx


def _plugin_context_for_operation_calls(*, telemetry_emit: Any) -> Any:
    """Build a real PluginContext configured for sink operation call recording."""
    from elspeth.contracts.plugin_context import PluginContext

    landscape = MagicMock(spec_set=["record_operation_call"])
    landscape.record_operation_call.return_value = SimpleNamespace(
        request_hash="req-hash",
        response_hash="resp-hash",
    )

    return PluginContext(
        run_id="test-run-123",
        config={},
        landscape=landscape,
        node_id="sink-node",
        operation_id="op-001",
        telemetry_emit=telemetry_emit,
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

    @pytest.mark.parametrize("entity", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_entity_placeholder_rejected(self, entity: str) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DataverseSinkConfig.from_dict(_config(entity=entity))

    @pytest.mark.parametrize("entity", ["todo", "unknown", "unset", "required", "<literal>"])
    def test_plain_placeholder_words_can_be_entity_names(self, entity: str) -> None:
        cfg = DataverseSinkConfig.from_dict(_config(entity=entity))
        assert cfg.entity == entity

    @pytest.mark.parametrize("alternate_key", ["<OPERATOR_REQUIRED>", "operator required", "operator_required"])
    def test_alternate_key_placeholder_rejected(self, alternate_key: str) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DataverseSinkConfig.from_dict(_config(alternate_key=alternate_key))

    @pytest.mark.parametrize("alternate_key", ["todo", "unknown", "unset", "required", "<literal>"])
    def test_plain_placeholder_words_can_be_alternate_keys(self, alternate_key: str) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                alternate_key=alternate_key,
                field_mapping={"email": alternate_key, "name": "fullname"},
            )
        )
        assert cfg.alternate_key == alternate_key

    def test_field_mapping_required(self) -> None:
        c = _config()
        del c["field_mapping"]
        with pytest.raises(PluginConfigError, match="field_mapping"):
            DataverseSinkConfig.from_dict(c)

    def test_field_mapping_target_placeholder_rejected(self) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DataverseSinkConfig.from_dict(_config(field_mapping={"email": "operator_required", "name": "fullname"}))

    @pytest.mark.parametrize("target", ["todo", "unknown", "unset", "required", "<literal>"])
    def test_plain_placeholder_words_can_be_field_mapping_targets(self, target: str) -> None:
        cfg = DataverseSinkConfig.from_dict(_config(alternate_key=target, field_mapping={"email": target}))
        assert cfg.field_mapping == {"email": target}

    def test_https_enforcement(self) -> None:
        with pytest.raises(PluginConfigError, match="HTTPS"):
            DataverseSinkConfig.from_dict(_config(environment_url="http://myorg.crm.dynamics.com"))

    def test_https_enforcement_no_scheme(self) -> None:
        with pytest.raises(PluginConfigError):
            DataverseSinkConfig.from_dict(_config(environment_url="myorg.crm.dynamics.com"))

    def test_environment_url_rejects_embedded_credentials(self) -> None:
        with pytest.raises(PluginConfigError, match="embedded credentials"):
            DataverseSinkConfig.from_dict(_config(environment_url="https://user:pass@myorg.crm.dynamics.com"))

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

    def test_lookup_target_entity_placeholder_rejected(self) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "operator_required",
                            "target_field": "parentcustomerid",
                        }
                    }
                )
            )

    @pytest.mark.parametrize("target_entity", ["todo", "unknown", "unset", "required", "<literal>"])
    def test_plain_placeholder_words_can_be_lookup_target_entities(self, target_entity: str) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                lookups={
                    "account_id": {
                        "target_entity": target_entity,
                        "target_field": "parentcustomerid",
                    }
                }
            )
        )
        assert cfg.lookups is not None
        assert cfg.lookups["account_id"].target_entity == target_entity

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

    def test_lookup_target_field_placeholder_rejected(self) -> None:
        with pytest.raises(PluginConfigError, match="placeholder"):
            DataverseSinkConfig.from_dict(
                _config(
                    lookups={
                        "account_id": {
                            "target_entity": "accounts",
                            "target_field": "operator_required",
                        }
                    }
                )
            )

    @pytest.mark.parametrize("target_field", ["todo", "unknown", "unset", "required", "<literal>"])
    def test_plain_placeholder_words_can_be_lookup_target_fields(self, target_field: str) -> None:
        cfg = DataverseSinkConfig.from_dict(
            _config(
                lookups={
                    "account_id": {
                        "target_entity": "accounts",
                        "target_field": target_field,
                    }
                }
            )
        )
        assert cfg.lookups is not None
        assert cfg.lookups["account_id"].target_field == target_field

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

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_alternate_key_not_in_field_mapping_values_raises(self, _mock_schema: MagicMock) -> None:
        """alternate_key must be a Dataverse column that appears as a value in field_mapping."""
        with pytest.raises(PluginConfigError, match="not found in field_mapping values"):
            DataverseSink(_config(alternate_key="nonexistent_column"))

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_alternate_key_pipeline_field_resolved(self, _mock_schema: MagicMock) -> None:
        """The pipeline field for the alternate key should be resolved from field_mapping."""
        sink = inject_write_failure(DataverseSink(_config()))
        # alternate_key is "emailaddress1" (Dataverse column)
        # field_mapping maps "email" (pipeline) -> "emailaddress1" (Dataverse)
        # So _alternate_key_pipeline_field should be "email"
        assert sink._alternate_key_pipeline_field == "email"

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
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

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
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
            return_value=FakeDataverseRowSchema,
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

    def test_lookup_bind_value_rejects_odata_injection(self) -> None:
        """Lookup values with OData/URI structural chars are rejected (elspeth-e7d31117df).

        The @odata.bind value sits in the unquoted /entity(value) key position, so
        a row value like "abc)/contacts(emailaddress1='victim')" could change the
        bind URI's navigation shape (injection). _map_row must reject such values
        clearly at the sink boundary rather than emit an ambiguous/injectable
        outbound payload. Validate-and-reject is safe regardless of whether
        Dataverse percent-decodes the bind reference.
        """
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
        for malicious in (
            "abc)/contacts(emailaddress1='victim')",
            "guid'/x",
            "a b",
            "id=1",
            "../accounts(x)",
        ):
            with pytest.raises(ValueError, match="not a valid record reference"):
                sink._map_row({"email": "a@b.com", "account_id": malicious})

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
            return_value=FakeDataverseRowSchema,
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
            return_value=FakeDataverseRowSchema,
        ):
            sink = inject_write_failure(DataverseSink(_config()))

        mock_client = _make_dataverse_client_double()
        mock_client.upsert.return_value = _make_204_response()
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client
        return sink, mock_client

    def _make_mock_ctx(self) -> MagicMock:
        ctx = _make_sink_context_double()
        return ctx


# ---------------------------------------------------------------------------
# Write lifecycle
# ---------------------------------------------------------------------------


class TestWriteLifecycle:
    """Tests for on_start and write lifecycle."""

    def _make_mock_ctx(self) -> MagicMock:
        ctx = _make_sink_context_double()
        return ctx

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    @patch("elspeth.plugins.sinks.dataverse.DataverseClient")
    @patch("azure.identity.ClientSecretCredential")
    def test_on_start_constructs_credential_and_client(
        self, mock_cred_cls: MagicMock, mock_client_cls: MagicMock, _mock_schema: MagicMock
    ) -> None:
        sink = inject_write_failure(DataverseSink(_config()))

        mock_lifecycle = _make_lifecycle_context_double()

        sink.on_start(mock_lifecycle)

        mock_cred_cls.assert_called_once_with(
            tenant_id="tenant-1",
            client_id="client-1",
            client_secret="secret-1",
        )
        mock_client_cls.assert_called_once()
        assert sink._client is not None

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    @patch("elspeth.plugins.sinks.dataverse.DataverseClient")
    @patch("azure.identity.ManagedIdentityCredential")
    def test_on_start_managed_identity(self, mock_mi_cls: MagicMock, _mock_client_cls: MagicMock, _mock_schema: MagicMock) -> None:
        cfg = _config(auth={"method": "managed_identity"})
        sink = inject_write_failure(DataverseSink(cfg))

        mock_lifecycle = _make_lifecycle_context_double(run_id="run-mi")

        sink.on_start(mock_lifecycle)

        mock_mi_cls.assert_called_once()

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_flush_is_noop(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        sink.flush()  # Should not raise

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_close_releases_client(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        mock_client = _make_dataverse_client_double()
        sink._client = mock_client

        sink.close()

        mock_client.close.assert_called_once()
        assert sink._client is None

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_close_without_client_is_safe(self, _mock_schema: MagicMock) -> None:
        sink = inject_write_failure(DataverseSink(_config()))
        assert sink._client is None
        sink.close()  # Should not raise


# ---------------------------------------------------------------------------
# Per-row on_write_failure routing (audit finding #4)
# ---------------------------------------------------------------------------


class TestPerRowWriteFailureRouting:
    """The dataverse sink advertises per-row on_write_failure routing in its
    composer hint. A per-row-attributable HTTP failure (non-retryable 4xx about
    the row payload/key) must divert that row and continue the batch; a
    batch-integrity failure (auth/authz, rate limit, retryable, 5xx) must raise.
    """

    def _make_mock_ctx(self) -> MagicMock:
        ctx = _make_sink_context_double()
        return ctx

    def _make_sink(self, upsert_side_effect: Any) -> tuple[DataverseSink, MagicMock]:
        with patch(
            "elspeth.plugins.sinks.dataverse.create_schema_from_config",
            return_value=FakeDataverseRowSchema,
        ):
            sink = inject_write_failure(DataverseSink(_config()))
        mock_client = _make_dataverse_client_double()
        mock_client.upsert.side_effect = upsert_side_effect
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake-token"}
        sink._client = mock_client
        return sink, mock_client


# ---------------------------------------------------------------------------
# Bug fix: idempotent flag (elspeth-1453d7cfa8)
# ---------------------------------------------------------------------------


class TestIdempotentFlag:
    """Sink idempotent flag must be True for PATCH upsert mode."""

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
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


def _effect_member(ordinal: int, row: dict[str, object]) -> SinkEffectMember:
    row_json = contract_canonical_json(row)
    lineage_json = "[]"
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json=lineage_json,
        lineage_hash=hashlib.sha256(lineage_json.encode()).hexdigest(),
        payload_hash=hashlib.sha256(row_json.encode()).hexdigest(),
        row=row,
        member_effect_id=stable_hash({"effect": "a" * 64, "ordinal": ordinal}),
    )


def _restricted_effect_context() -> RestrictedSinkEffectContext:
    return RestrictedSinkEffectContext(
        run_id="run-1",
        run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
        operation_id="operation-1",
        sink_node_id="sink-1",
    )


class _RecoverableDataverseClient:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.patch_count_by_key: dict[str, int] = {}

    @staticmethod
    def _key(url: str) -> str:
        return urllib.parse.unquote(url.rsplit("='", 1)[1].split("')", 1)[0])

    def upsert(self, url: str, body: dict[str, object]) -> DataversePageResponse:
        key = self._key(url)
        self.patch_count_by_key[key] = self.patch_count_by_key.get(key, 0) + 1
        self.records[key] = dict(body)
        return _make_204_response()

    def get_page(self, url: str) -> DataversePageResponse:
        key = self._key(url)
        if key not in self.records:
            raise DataverseClientError(
                "missing",
                retryable=False,
                status_code=404,
                error_category="row_data_error",
            )
        return DataversePageResponse(
            status_code=200,
            rows=[dict(self.records[key])],
            latency_ms=1.0,
            headers={},
            request_headers={},
            request_url=url,
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )


class TestDataverseMemberEffects:
    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_partial_batch_recovery_reconciles_exact_and_commits_only_missing(self, _mock_schema: MagicMock) -> None:
        client = _RecoverableDataverseClient()
        sink = inject_write_failure(DataverseSink(_config()))
        sink._client = client  # type: ignore[assignment]
        members = tuple(
            _effect_member(index, {"email": key, "name": name})
            for index, (key, name) in enumerate((("a", "Alice"), ("b", "Bob"), ("c", "Carol")))
        )
        effect_input = SinkEffectPipelineMembersInput(members, members)
        ctx = _restricted_effect_context()
        inspection = sink.inspect_effect(
            SinkEffectInspectionRequest(effect_id="a" * 64, target="{}", predecessor_descriptor=None),
            ctx,
        )
        assert inspection.mode is SinkEffectInspectionMode.NO_INSPECTION_REQUIRED
        plan = sink.prepare_effect(
            SinkEffectPrepareRequest(effect_id="a" * 64, effect_input=effect_input, inspection=inspection),
            ctx,
        )

        sink.commit_member_effect(plan, members[0], effect_input, ctx)

        recovered = inject_write_failure(DataverseSink(_config()))
        recovered._client = client  # type: ignore[assignment]
        states = [recovered.reconcile_member_effect(plan, member, effect_input, ctx).kind for member in members]
        assert states == [
            SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR,
            SinkEffectReconcileKind.NOT_APPLIED,
            SinkEffectReconcileKind.NOT_APPLIED,
        ]
        for member, state in zip(members, states, strict=True):
            if state is SinkEffectReconcileKind.NOT_APPLIED:
                recovered.commit_member_effect(plan, member, effect_input, ctx)

        assert client.patch_count_by_key == {"a": 1, "b": 1, "c": 1}

    @patch("elspeth.plugins.sinks.dataverse.create_schema_from_config", return_value=FakeDataverseRowSchema)
    def test_reconcile_returns_unknown_for_divergent_mapped_field(self, _mock_schema: MagicMock) -> None:
        client = _RecoverableDataverseClient()
        sink = inject_write_failure(DataverseSink(_config()))
        sink._client = client  # type: ignore[assignment]
        member = _effect_member(0, {"email": "a", "name": "Alice"})
        effect_input = SinkEffectPipelineMembersInput((member,), (member,))
        ctx = _restricted_effect_context()
        inspection = sink.inspect_effect(SinkEffectInspectionRequest(effect_id="a" * 64, target="{}", predecessor_descriptor=None), ctx)
        plan = sink.prepare_effect(SinkEffectPrepareRequest(effect_id="a" * 64, effect_input=effect_input, inspection=inspection), ctx)
        client.records["a"] = {"emailaddress1": "a", "fullname": "Mallory"}

        result = sink.reconcile_member_effect(plan, member, effect_input, ctx)

        assert result.kind is SinkEffectReconcileKind.UNKNOWN
        assert deep_thaw(result.evidence)["classification"] == "divergent"
