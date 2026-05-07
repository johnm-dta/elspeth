"""Tests for Dataverse source plugin."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import CallStatus
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseClientError,
    DataversePageResponse,
)
from elspeth.plugins.infrastructure.config_base import PluginConfigError

# Dynamic schema config for tests
DYNAMIC_SCHEMA = {"mode": "observed"}
FIXED_SCHEMA = {
    "mode": "fixed",
    "fields": [
        {"name": "contactid", "type": "string"},
        {"name": "fullname", "type": "string"},
    ],
}
FLEXIBLE_SCHEMA = {
    "mode": "flexible",
    "fields": [
        {"name": "contactid", "type": "string"},
    ],
}

# Standard quarantine routing
QUARANTINE_SINK = "quarantine"

# Minimal valid auth config
VALID_AUTH = {
    "method": "service_principal",
    "tenant_id": "test-tenant",
    "client_id": "test-client",
    "client_secret": "test-secret",
}

# Valid environment URL
VALID_ENV_URL = "https://myorg.crm.dynamics.com"


def _base_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid structured-query config with overrides."""
    config: dict[str, Any] = {
        "environment_url": VALID_ENV_URL,
        "auth": VALID_AUTH,
        "entity": "contacts",
        "schema": DYNAMIC_SCHEMA,
        "on_validation_failure": QUARANTINE_SINK,
    }
    config.update(overrides)
    return config


def _fetchxml_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal valid FetchXML config with overrides."""
    config: dict[str, Any] = {
        "environment_url": VALID_ENV_URL,
        "auth": VALID_AUTH,
        "fetch_xml": '<fetch><entity name="contacts"><attribute name="fullname"/></entity></fetch>',
        "schema": DYNAMIC_SCHEMA,
        "on_validation_failure": QUARANTINE_SINK,
    }
    config.update(overrides)
    return config


def _make_page(
    rows: list[dict[str, Any]],
    *,
    next_link: str | None = None,
    paging_cookie: str | None = None,
    more_records: bool | None = False,
    status_code: int = 200,
    latency_ms: float = 10.0,
) -> DataversePageResponse:
    """Build a canned DataversePageResponse."""
    return DataversePageResponse(
        status_code=status_code,
        rows=rows,
        latency_ms=latency_ms,
        headers={"content-type": "application/json"},
        request_headers={"Authorization": "<fingerprint:test-fake>"},
        request_url="https://test.crm.dynamics.com/api/data/v9.2/contacts",
        next_link=next_link,
        paging_cookie=paging_cookie,
        more_records=more_records,
    )


def _mock_lifecycle_context(run_id: str = "test-run-123") -> MagicMock:
    """Build a mock LifecycleContext for on_start()."""
    ctx = MagicMock()
    ctx.run_id = run_id
    ctx.telemetry_emit = MagicMock()
    ctx.rate_limit_registry = None
    return ctx


def _mock_source_context() -> MagicMock:
    """Build a mock SourceContext for load()."""
    ctx = MagicMock()
    ctx.record_call = MagicMock()
    ctx.record_validation_error = MagicMock()
    return ctx


def _plugin_context_for_operation_calls(*, telemetry_emit: Any) -> Any:
    """Build a real PluginContext configured for source operation call recording."""
    from elspeth.contracts.plugin_context import PluginContext

    landscape = MagicMock()
    landscape.record_operation_call.return_value = SimpleNamespace(
        request_hash="req-hash",
        response_hash="resp-hash",
    )

    return PluginContext(
        run_id="test-run-123",
        config={},
        landscape=landscape,
        node_id="source-node",
        operation_id="op-001",
        telemetry_emit=telemetry_emit,
    )


def _make_source(config: dict[str, Any]) -> Any:
    """Create a DataverseSource with patched dependencies.

    The constructor imports ContractBuilder and create_contract_from_config
    lazily, so we patch them at their origin modules.
    """
    from elspeth.plugins.sources.dataverse import DataverseSource

    mock_contract = MagicMock()
    mock_contract.locked = True

    mock_contract_unlocked = MagicMock()
    mock_contract_unlocked.locked = False
    mock_contract_unlocked.with_locked.return_value = mock_contract

    with (
        patch(
            "elspeth.plugins.sources.dataverse.create_schema_from_config",
            return_value=MagicMock(),
        ),
        patch(
            "elspeth.contracts.schema_contract_factory.create_contract_from_config",
            return_value=mock_contract,
        ),
    ):
        return DataverseSource(config)


def _make_source_unlocked(config: dict[str, Any]) -> Any:
    """Create a DataverseSource with an unlocked contract (for contract builder tests)."""
    from elspeth.plugins.sources.dataverse import DataverseSource

    mock_contract = MagicMock()
    mock_contract.locked = False
    mock_contract.with_locked.return_value = MagicMock()

    mock_schema_cls = MagicMock()

    def mock_validate(row: dict[str, Any]) -> MagicMock:
        m = MagicMock()
        m.to_row.return_value = dict(row)
        return m

    mock_schema_cls.model_validate = mock_validate

    with (
        patch(
            "elspeth.plugins.sources.dataverse.create_schema_from_config",
            return_value=mock_schema_cls,
        ),
        patch(
            "elspeth.contracts.schema_contract_factory.create_contract_from_config",
            return_value=mock_contract,
        ),
    ):
        source = DataverseSource(config)

    return source


def _make_source_for_load(
    pages: list[DataversePageResponse],
    config: dict[str, Any],
    *,
    schema_validate_side_effect: Any = None,
) -> Any:
    """Create DataverseSource for load() tests with mocked client and schema."""
    from elspeth.plugins.sources.dataverse import DataverseSource

    mock_contract = MagicMock()
    mock_contract.locked = False
    mock_contract.with_locked.return_value = MagicMock()

    mock_schema_cls = MagicMock()

    if schema_validate_side_effect is not None:
        mock_schema_cls.model_validate = schema_validate_side_effect
    else:

        def mock_validate(row: dict[str, Any]) -> MagicMock:
            m = MagicMock()
            m.to_row.return_value = dict(row)
            return m

        mock_schema_cls.model_validate = mock_validate

    with (
        patch(
            "elspeth.plugins.sources.dataverse.create_schema_from_config",
            return_value=mock_schema_cls,
        ),
        patch(
            "elspeth.contracts.schema_contract_factory.create_contract_from_config",
            return_value=mock_contract,
        ),
    ):
        source = DataverseSource(config)

    # Inject mock client
    mock_client = MagicMock()
    if source._entity is not None:
        metadata_url = (
            f"{source._environment_url.rstrip('/')}/api/data/{source._api_version}/"
            f"EntityDefinitions(LogicalName='{source._entity}')?$select=LogicalName"
        )
        mock_client.get_page.return_value = DataversePageResponse(
            status_code=200,
            rows=[{"LogicalName": source._entity}],
            latency_ms=5.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "<fingerprint:test-fake>"},
            request_url=metadata_url,
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )
        mock_client.paginate_odata.return_value = iter(pages)
    else:
        mock_client.paginate_fetchxml.return_value = iter(pages)
    mock_client.get_auth_headers.return_value = {"Authorization": "Bearer test"}
    source._client = mock_client

    return source


def _make_source_for_start_and_load(
    config: dict[str, Any],
    *,
    mock_client: MagicMock,
    schema_validate_side_effect: Any = None,
) -> Any:
    """Create DataverseSource for lifecycle + load() tests with a patched client."""
    from elspeth.plugins.sources.dataverse import DataverseSource

    mock_contract = MagicMock()
    mock_contract.locked = False
    mock_contract.with_locked.return_value = MagicMock()

    mock_schema_cls = MagicMock()

    if schema_validate_side_effect is not None:
        mock_schema_cls.model_validate = schema_validate_side_effect
    else:

        def mock_validate(row: dict[str, Any]) -> MagicMock:
            m = MagicMock()
            m.to_row.return_value = dict(row)
            return m

        mock_schema_cls.model_validate = mock_validate

    with (
        patch(
            "elspeth.plugins.sources.dataverse.create_schema_from_config",
            return_value=mock_schema_cls,
        ),
        patch(
            "elspeth.contracts.schema_contract_factory.create_contract_from_config",
            return_value=mock_contract,
        ),
    ):
        source = DataverseSource(config)

    lifecycle_ctx = _mock_lifecycle_context()
    with (
        patch("azure.identity.ClientSecretCredential", return_value=MagicMock()),
        patch(
            "elspeth.plugins.sources.dataverse.DataverseClient",
            return_value=mock_client,
        ),
    ):
        source.on_start(lifecycle_ctx)

    return source


# ─────────────────────────────────────────────────────────────────────────
# Config validation tests
# ─────────────────────────────────────────────────────────────────────────


class TestDataverseSourceConfigValidation:
    """Tests for DataverseSourceConfig validation rules."""

    def test_valid_structured_config(self) -> None:
        """Accept a minimal valid structured-query config."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_base_config())
        assert cfg.entity == "contacts"
        assert cfg.fetch_xml is None

    def test_valid_fetchxml_config(self) -> None:
        """Accept a minimal valid FetchXML config."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_fetchxml_config())
        assert cfg.entity is None
        assert cfg.fetch_xml is not None

    def test_mutual_exclusion_both_present(self) -> None:
        """Reject config with both entity and fetch_xml."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(
            fetch_xml='<fetch><entity name="contacts"/></fetch>',
        )
        with pytest.raises(PluginConfigError, match="exactly one"):
            DataverseSourceConfig.from_dict(config)

    def test_mutual_exclusion_neither_present(self) -> None:
        """Reject config with neither entity nor fetch_xml."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config()
        del config["entity"]
        with pytest.raises(PluginConfigError, match="exactly one"):
            DataverseSourceConfig.from_dict(config)

    def test_select_requires_entity(self) -> None:
        """select is only valid in structured mode (with entity)."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(select=["fullname"])
        with pytest.raises(PluginConfigError, match=r"select.*require entity"):
            DataverseSourceConfig.from_dict(config)

    def test_filter_requires_entity(self) -> None:
        """filter is only valid in structured mode."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(filter="statecode eq 0")
        with pytest.raises(PluginConfigError, match=r"select.*require entity"):
            DataverseSourceConfig.from_dict(config)

    def test_orderby_requires_entity(self) -> None:
        """orderby is only valid in structured mode."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(orderby="createdon desc")
        with pytest.raises(PluginConfigError, match=r"select.*require entity"):
            DataverseSourceConfig.from_dict(config)

    def test_top_requires_entity(self) -> None:
        """top is only valid in structured mode."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(top=100)
        with pytest.raises(PluginConfigError, match=r"select.*require entity"):
            DataverseSourceConfig.from_dict(config)

    def test_https_enforcement_rejects_http(self) -> None:
        """environment_url must use HTTPS."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(environment_url="http://myorg.crm.dynamics.com")
        with pytest.raises(PluginConfigError, match="HTTPS"):
            DataverseSourceConfig.from_dict(config)

    def test_https_enforcement_accepts_https(self) -> None:
        """HTTPS URLs are accepted."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_base_config())
        assert cfg.environment_url.startswith("https://")

    def test_on_validation_failure_required(self) -> None:
        """on_validation_failure cannot be omitted."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config()
        del config["on_validation_failure"]
        with pytest.raises(PluginConfigError):
            DataverseSourceConfig.from_dict(config)

    def test_on_validation_failure_not_empty(self) -> None:
        """on_validation_failure cannot be empty string."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(on_validation_failure="")
        with pytest.raises(PluginConfigError, match="on_validation_failure"):
            DataverseSourceConfig.from_dict(config)

    def test_on_validation_failure_not_whitespace(self) -> None:
        """on_validation_failure cannot be whitespace-only."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(on_validation_failure="   ")
        with pytest.raises(PluginConfigError, match="on_validation_failure"):
            DataverseSourceConfig.from_dict(config)

    def test_on_validation_failure_strips_whitespace(self) -> None:
        """on_validation_failure value is stripped."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_base_config(on_validation_failure="  quarantine  "))
        assert cfg.on_validation_failure == "quarantine"

    def test_fetchxml_bad_xml_syntax(self) -> None:
        """FetchXML with broken XML syntax fails at config time."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(fetch_xml="<fetch><not-closed>")
        with pytest.raises(PluginConfigError, match="invalid XML syntax"):
            DataverseSourceConfig.from_dict(config)

    def test_fetchxml_wrong_root_element(self) -> None:
        """FetchXML with non-<fetch> root element fails at config time."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _fetchxml_config(
            fetch_xml='<query><entity name="contacts"/></query>',
        )
        with pytest.raises(PluginConfigError, match="root element must be <fetch>"):
            DataverseSourceConfig.from_dict(config)

    def test_fetchxml_valid_xml(self) -> None:
        """Valid FetchXML passes syntax validation."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_fetchxml_config())
        assert cfg.fetch_xml is not None
        assert "<fetch>" in cfg.fetch_xml

    def test_additional_domains_valid(self) -> None:
        """Accept valid additional domain patterns."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(additional_domains=["*.crm12.dynamics.com"])
        cfg = DataverseSourceConfig.from_dict(config)
        assert cfg.additional_domains == ["*.crm12.dynamics.com"]

    def test_additional_domains_rejects_non_microsoft_tld(self) -> None:
        """Reject additional domain patterns that don't target Microsoft TLDs."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(additional_domains=["*.evil.example.com"])
        with pytest.raises(PluginConfigError, match="rejected"):
            DataverseSourceConfig.from_dict(config)

    def test_structured_query_with_all_odata_options(self) -> None:
        """Accept structured query with select, filter, orderby, top."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        config = _base_config(
            select=["contactid", "fullname"],
            filter="statecode eq 0",
            orderby="createdon desc",
            top=100,
        )
        cfg = DataverseSourceConfig.from_dict(config)
        assert cfg.select == ["contactid", "fullname"]
        assert cfg.filter == "statecode eq 0"
        assert cfg.orderby == "createdon desc"
        assert cfg.top == 100

    def test_normalize_fields_config_key_rejected(self) -> None:
        """normalize_fields is no longer a valid config key."""
        from elspeth.plugins.infrastructure.config_base import PluginConfigError
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        with pytest.raises(PluginConfigError):
            DataverseSourceConfig.from_dict(_base_config(normalize_fields=True))

    def test_include_formatted_values_defaults_false(self) -> None:
        """include_formatted_values defaults to False."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_base_config())
        assert cfg.include_formatted_values is False

    def test_api_version_defaults_to_v92(self) -> None:
        """api_version defaults to v9.2."""
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        cfg = DataverseSourceConfig.from_dict(_base_config())
        assert cfg.api_version == "v9.2"


# ─────────────────────────────────────────────────────────────────────────
# OData metadata stripping tests
# ─────────────────────────────────────────────────────────────────────────


class TestODataMetadataStripping:
    """Tests for _strip_odata_metadata behavior."""

    def test_strips_odata_fields(self) -> None:
        """@odata.* fields are removed from rows."""
        source = _make_source(_base_config())
        row = {
            "contactid": "abc-123",
            "fullname": "Alice",
            "@odata.etag": 'W/"123"',
            "@odata.context": "https://...",
        }
        cleaned = source._strip_odata_metadata(row)
        assert "contactid" in cleaned
        assert "fullname" in cleaned
        assert "@odata.etag" not in cleaned
        assert "@odata.context" not in cleaned

    def test_strips_microsoft_dynamics_crm_fields(self) -> None:
        """@Microsoft.Dynamics.CRM.* fields are removed."""
        source = _make_source(_base_config())
        row = {
            "contactid": "abc-123",
            "@Microsoft.Dynamics.CRM.totalrecordcount": 42,
        }
        cleaned = source._strip_odata_metadata(row)
        assert "contactid" in cleaned
        assert "@Microsoft.Dynamics.CRM.totalrecordcount" not in cleaned

    def test_formatted_values_excluded_by_default(self) -> None:
        """Formatted value annotations pass through as raw keys when include_formatted_values=False.

        The formatted value suffix is NOT an @odata.* or @Microsoft.Dynamics.CRM.*
        annotation — it lives on the field key itself (e.g., statecode@OData.Community...).
        When include_formatted_values=False, these keys are not extracted into
        __formatted fields but they do pass through as regular keys.
        """
        source = _make_source(_base_config(include_formatted_values=False))
        row = {
            "statecode": 0,
            "statecode@OData.Community.Display.V1.FormattedValue": "Active",
        }
        cleaned = source._strip_odata_metadata(row)
        # The raw annotation key passes through (it doesn't match OData strip patterns)
        assert cleaned["statecode"] == 0
        assert "statecode@OData.Community.Display.V1.FormattedValue" in cleaned
        # No __formatted suffix key is created
        assert "statecode__formatted" not in cleaned

    def test_formatted_values_included_with_suffix(self) -> None:
        """Formatted values get __formatted suffix when include_formatted_values=True."""
        source = _make_source(_base_config(include_formatted_values=True))
        row = {
            "statecode": 0,
            "statecode@OData.Community.Display.V1.FormattedValue": "Active",
        }
        cleaned = source._strip_odata_metadata(row)
        assert cleaned == {"statecode": 0, "statecode__formatted": "Active"}

    def test_formatted_value_collision_detection(self) -> None:
        """Collision detected when both 'field__formatted' and formatted annotation exist."""
        source = _make_source(_base_config(include_formatted_values=True))
        row = {
            "statecode": 0,
            "statecode__formatted": "manually set",
            "statecode@OData.Community.Display.V1.FormattedValue": "Active",
        }
        with pytest.raises(ValueError, match="collision"):
            source._strip_odata_metadata(row)

    def test_multiple_formatted_values(self) -> None:
        """Multiple formatted values are preserved correctly."""
        source = _make_source(_base_config(include_formatted_values=True))
        row = {
            "statecode": 0,
            "statecode@OData.Community.Display.V1.FormattedValue": "Active",
            "statuscode": 1,
            "statuscode@OData.Community.Display.V1.FormattedValue": "Active",
            "@odata.etag": 'W/"123"',
        }
        cleaned = source._strip_odata_metadata(row)
        assert cleaned == {
            "statecode": 0,
            "statecode__formatted": "Active",
            "statuscode": 1,
            "statuscode__formatted": "Active",
        }

    def test_data_fields_preserved(self) -> None:
        """Regular data fields are not stripped."""
        source = _make_source(_base_config())
        row = {
            "contactid": "abc",
            "fullname": "Alice",
            "emailaddress1": "alice@example.com",
            "revenue": 1000.50,
        }
        cleaned = source._strip_odata_metadata(row)
        assert cleaned == row


# ─────────────────────────────────────────────────────────────────────────
# Field normalization tests
# ─────────────────────────────────────────────────────────────────────────


class TestFieldNormalization:
    """Tests for _normalize_row_fields behavior."""

    def test_normalization_always_applies(self) -> None:
        """Field names are always normalized."""
        source = _make_source(_base_config())
        row = {"Contact ID": "abc", "Full Name": "Alice"}
        result = source._normalize_row_fields(row, is_first_row=True)
        # Normalized names should be valid Python identifiers
        for key in result:
            assert key.isidentifier(), f"Key '{key}' is not a valid identifier"

    def test_field_resolution_populated_on_first_row(self) -> None:
        """Field resolution mapping is created on the first row."""
        source = _make_source(_base_config())
        assert source._field_resolution is None
        source._normalize_row_fields({"contactid": "abc"}, is_first_row=True)
        assert source._field_resolution is not None

    def test_field_resolution_reused_on_subsequent_rows(self) -> None:
        """Field resolution mapping is reused (not re-created) for subsequent rows."""
        source = _make_source(_base_config())
        source._normalize_row_fields({"contactid": "abc"}, is_first_row=True)
        resolution1 = source._field_resolution
        source._normalize_row_fields({"contactid": "def"}, is_first_row=False)
        resolution2 = source._field_resolution
        assert resolution1 is resolution2


# ─────────────────────────────────────────────────────────────────────────
# DataverseSource construction and lifecycle tests
# ─────────────────────────────────────────────────────────────────────────


class TestDataverseSourceConstruction:
    """Tests for DataverseSource.__init__ and on_start."""

    def test_construction_structured_query(self) -> None:
        """DataverseSource can be constructed with structured query config."""
        source = _make_source(_base_config())
        assert source.name == "dataverse"
        assert source._entity == "contacts"
        assert source._fetch_xml is None

    def test_construction_fetchxml_query(self) -> None:
        """DataverseSource can be constructed with FetchXML config."""
        source = _make_source(_fetchxml_config())
        assert source._entity is None
        assert source._fetch_xml is not None

    def test_on_start_creates_client(self) -> None:
        """on_start() creates a DataverseClient without probing entity metadata."""
        source = _make_source(_base_config())
        assert source._client is None

        lifecycle_ctx = _mock_lifecycle_context()
        mock_client = MagicMock()

        with (
            patch("azure.identity.ClientSecretCredential") as mock_cred,
            patch("elspeth.plugins.sources.dataverse.DataverseClient") as mock_client_cls,
        ):
            mock_cred.return_value = MagicMock()
            mock_client_cls.return_value = mock_client
            source.on_start(lifecycle_ctx)

        assert source._client is not None
        mock_client.get_page.assert_not_called()

    def test_additional_domains_stored_as_tuple(self) -> None:
        """additional_domains are stored as a tuple for immutability."""
        source = _make_source(_base_config(additional_domains=["*.crm12.dynamics.com"]))
        assert source._additional_domains == ("*.crm12.dynamics.com",)

    def test_no_additional_domains_gives_empty_tuple(self) -> None:
        """No additional_domains results in an empty tuple."""
        source = _make_source(_base_config())
        assert source._additional_domains == ()

    def test_load_metadata_probe_404_raises_descriptive_error(self) -> None:
        """404 metadata probe is audited and raises a descriptive entity-not-found error."""
        mock_client = MagicMock()
        mock_client.get_page.side_effect = DataverseClientError(
            "Not Found",
            retryable=False,
            status_code=404,
        )
        source = _make_source_for_start_and_load(
            _base_config(),
            mock_client=mock_client,
        )
        ctx = _mock_source_context()

        with pytest.raises(DataverseClientError, match="Entity 'contacts' not found"):
            list(source.load(ctx))

        assert ctx.record_call.call_count == 1
        call_kwargs = ctx.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"]["status_code"] == 404

    def test_load_metadata_probe_403_records_error_and_continues(self) -> None:
        """403 metadata probe is audited but remains non-fatal to the load."""
        mock_client = MagicMock()
        mock_client.get_page.side_effect = DataverseClientError(
            "Forbidden",
            retryable=False,
            status_code=403,
        )
        mock_client.paginate_odata.return_value = iter([_make_page([{"contactid": "1", "fullname": "Alice"}])])
        source = _make_source_for_start_and_load(
            _base_config(),
            mock_client=mock_client,
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert ctx.record_call.call_count == 2
        metadata_call = ctx.record_call.call_args_list[0].kwargs
        assert metadata_call["status"] == CallStatus.ERROR
        assert metadata_call["error"]["status_code"] == 403

    def test_load_metadata_probe_5xx_reraises(self) -> None:
        """5xx metadata probe is audited and re-raises the original error."""
        original_error = DataverseClientError(
            "Internal Server Error",
            retryable=True,
            status_code=500,
        )
        mock_client = MagicMock()
        mock_client.get_page.side_effect = original_error
        source = _make_source_for_start_and_load(
            _base_config(),
            mock_client=mock_client,
        )
        ctx = _mock_source_context()

        with pytest.raises(DataverseClientError, match="Internal Server Error") as exc_info:
            list(source.load(ctx))

        assert exc_info.value is original_error
        assert ctx.record_call.call_count == 1
        call_kwargs = ctx.record_call.call_args.kwargs
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"]["status_code"] == 500


# ─────────────────────────────────────────────────────────────────────────
# Build query URL tests
# ─────────────────────────────────────────────────────────────────────────


class TestBuildQueryUrl:
    """Tests for _build_query_url."""

    def test_entity_only(self) -> None:
        """URL with entity only, no query params."""
        source = _make_source(_base_config())
        url = source._build_query_url()
        assert url == "https://myorg.crm.dynamics.com/api/data/v9.2/contacts"

    def test_with_select(self) -> None:
        """URL includes $select parameter."""
        source = _make_source(_base_config(select=["contactid", "fullname"]))
        url = source._build_query_url()
        assert "$select=contactid,fullname" in url

    def test_with_filter(self) -> None:
        """URL includes $filter parameter (percent-encoded)."""
        source = _make_source(_base_config(filter="statecode eq 0"))
        url = source._build_query_url()
        assert "$filter=statecode%20eq%200" in url

    def test_with_orderby(self) -> None:
        """URL includes $orderby parameter (percent-encoded)."""
        source = _make_source(_base_config(orderby="createdon desc"))
        url = source._build_query_url()
        assert "$orderby=createdon%20desc" in url

    def test_with_top(self) -> None:
        """URL includes $top parameter."""
        source = _make_source(_base_config(top=50))
        url = source._build_query_url()
        assert "$top=50" in url

    def test_with_all_params(self) -> None:
        """URL includes all OData parameters."""
        source = _make_source(
            _base_config(
                select=["contactid"],
                filter="statecode eq 0",
                orderby="createdon desc",
                top=10,
            )
        )
        url = source._build_query_url()
        assert "$select=contactid" in url
        assert "$filter=statecode%20eq%200" in url
        assert "$orderby=createdon%20desc" in url
        assert "$top=10" in url

    def test_custom_api_version(self) -> None:
        """URL uses configured API version."""
        source = _make_source(_base_config(api_version="v9.1"))
        url = source._build_query_url()
        assert "/api/data/v9.1/" in url

    def test_trailing_slash_on_env_url(self) -> None:
        """Trailing slash on environment_url is stripped."""
        source = _make_source(_base_config(environment_url="https://myorg.crm.dynamics.com/"))
        url = source._build_query_url()
        assert "//api" not in url
        assert "/api/data/" in url


# ─────────────────────────────────────────────────────────────────────────
# load() tests — structured query mode
# ─────────────────────────────────────────────────────────────────────────


class TestDataverseSourceLoadStructured:
    """Tests for load() with structured OData queries."""

    def test_load_single_page(self) -> None:
        """Load yields valid rows from a single page."""
        pages = [
            _make_page(
                [
                    {"contactid": "1", "fullname": "Alice"},
                    {"contactid": "2", "fullname": "Bob"},
                ]
            ),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)

    def test_load_multiple_pages(self) -> None:
        """Load yields rows from multiple pages."""
        pages = [
            _make_page(
                [{"contactid": "1", "fullname": "Alice"}],
                next_link="https://myorg.crm.dynamics.com/api/data/v9.2/contacts?$skiptoken=1",
            ),
            _make_page([{"contactid": "2", "fullname": "Bob"}]),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 2

    def test_load_strips_odata_metadata(self) -> None:
        """OData annotation fields are stripped before yielding."""
        pages = [
            _make_page(
                [
                    {
                        "contactid": "1",
                        "fullname": "Alice",
                        "@odata.etag": 'W/"123"',
                        "@Microsoft.Dynamics.CRM.lookuplogicalname": "contact",
                    }
                ]
            ),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        row_data = rows[0].row
        assert "contactid" in row_data
        assert "@odata.etag" not in row_data
        assert "@Microsoft.Dynamics.CRM.lookuplogicalname" not in row_data

    def test_load_records_page_calls(self) -> None:
        """Structured load records the metadata probe plus each fetched page."""
        pages = [
            _make_page(
                [{"contactid": "1"}],
                next_link="https://myorg.crm.dynamics.com/next",
            ),
            _make_page([{"contactid": "2"}]),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        list(source.load(ctx))
        assert ctx.record_call.call_count == 3

    def test_load_records_entity_metadata_probe_before_structured_page_fetch(self) -> None:
        """Structured load records the startup entity metadata probe inside load()."""
        metadata_url = f"{VALID_ENV_URL}/api/data/v9.2/EntityDefinitions(LogicalName='contacts')?$select=LogicalName"
        metadata_page = DataversePageResponse(
            status_code=200,
            rows=[{"LogicalName": "contacts"}],
            latency_ms=5.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "<fingerprint:test-fake>"},
            request_url=metadata_url,
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )
        data_page = _make_page([{"contactid": "1", "fullname": "Alice"}])

        mock_client = MagicMock()
        mock_client.get_page.return_value = metadata_page
        mock_client.paginate_odata.return_value = iter([data_page])

        source = _make_source_for_start_and_load(
            _base_config(),
            mock_client=mock_client,
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert ctx.record_call.call_count == 2
        assert "EntityDefinitions(LogicalName='contacts')" in ctx.record_call.call_args_list[0].kwargs["request_data"]["url"]

    def test_load_quarantines_on_formatted_value_collision(self) -> None:
        """Formatted value collision quarantines the row."""
        pages = [
            _make_page(
                [
                    {
                        "statecode": 0,
                        "statecode__formatted": "existing",
                        "statecode@OData.Community.Display.V1.FormattedValue": "Active",
                    }
                ]
            ),
        ]
        source = _make_source_for_load(
            pages,
            _base_config(include_formatted_values=True),
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert "collision" in rows[0].quarantine_error.lower()
        ctx.record_validation_error.assert_called_once()
        call_kwargs = ctx.record_validation_error.call_args.kwargs
        assert call_kwargs["schema_mode"] == "odata_strip"
        assert "collision" in call_kwargs["error"].lower()

    def test_load_discard_does_not_yield_quarantined(self) -> None:
        """When on_validation_failure='discard', quarantined rows are not yielded."""
        pages = [
            _make_page(
                [
                    {
                        "statecode": 0,
                        "statecode__formatted": "existing",
                        "statecode@OData.Community.Display.V1.FormattedValue": "Active",
                    }
                ]
            ),
        ]
        source = _make_source_for_load(
            pages,
            _base_config(
                include_formatted_values=True,
                on_validation_failure="discard",
            ),
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 0
        # Validation error is still recorded
        ctx.record_validation_error.assert_called_once()

    def test_load_schema_validation_failure_quarantines(self) -> None:
        """Rows failing schema validation are quarantined."""
        from pydantic import ValidationError

        def failing_validate(row: dict[str, Any]) -> None:
            raise ValidationError.from_exception_data(
                title="DataverseRowSchema",
                line_errors=[
                    {
                        "type": "missing",
                        "loc": ("required_field",),
                        "input": row,
                    }
                ],
            )

        pages = [_make_page([{"contactid": "1"}])]
        source = _make_source_for_load(
            pages,
            _base_config(),
            schema_validate_side_effect=failing_validate,
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert rows[0].is_quarantined
        assert rows[0].quarantine_destination == QUARANTINE_SINK
        ctx.record_validation_error.assert_called_once()

    def test_load_schema_validation_failure_discard(self) -> None:
        """Schema validation failure with discard yields no rows."""
        from pydantic import ValidationError

        def failing_validate(row: dict[str, Any]) -> None:
            raise ValidationError.from_exception_data(
                title="DataverseRowSchema",
                line_errors=[
                    {
                        "type": "missing",
                        "loc": ("required_field",),
                        "input": row,
                    }
                ],
            )

        pages = [_make_page([{"contactid": "1"}])]
        source = _make_source_for_load(
            pages,
            _base_config(on_validation_failure="discard"),
            schema_validate_side_effect=failing_validate,
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 0

    def test_load_client_error_propagates(self) -> None:
        """DataverseClientError during pagination propagates directly."""
        source = _make_source_for_load([], _base_config())
        # Override client to raise
        source._client.paginate_odata.side_effect = DataverseClientError("Server error", retryable=True, status_code=500, latency_ms=100.0)

        ctx = _mock_source_context()
        with pytest.raises(DataverseClientError, match="Server error"):
            list(source.load(ctx))

    def test_load_client_error_records_audit(self) -> None:
        """Pagination errors are audited after the structured metadata probe succeeds."""
        source = _make_source_for_load([], _base_config())
        error = DataverseClientError("Server error", retryable=True, status_code=500, latency_ms=100.0)
        source._client.paginate_odata.side_effect = error

        ctx = _mock_source_context()
        with pytest.raises(DataverseClientError):
            list(source.load(ctx))

        assert ctx.record_call.call_count == 2
        call_kwargs = ctx.record_call.call_args_list[-1].kwargs
        assert call_kwargs["status"] == CallStatus.ERROR
        assert call_kwargs["error"]["status_code"] == 500

    def test_load_empty_pages(self) -> None:
        """Empty pages produce no rows."""
        pages = [_make_page([])]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 0

    def test_load_valid_and_quarantined_mixed(self) -> None:
        """Valid and quarantined rows can be interleaved."""
        from pydantic import ValidationError

        call_count = 0

        def sometimes_failing_validate(row: dict[str, Any]) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValidationError.from_exception_data(
                    title="DataverseRowSchema",
                    line_errors=[
                        {
                            "type": "missing",
                            "loc": ("field",),
                            "input": row,
                        }
                    ],
                )
            m = MagicMock()
            m.to_row.return_value = dict(row)
            return m

        pages = [
            _make_page(
                [
                    {"contactid": "1", "fullname": "Alice"},
                    {"contactid": "2"},  # Will fail validation
                    {"contactid": "3", "fullname": "Charlie"},
                ]
            ),
        ]
        source = _make_source_for_load(
            pages,
            _base_config(),
            schema_validate_side_effect=sometimes_failing_validate,
        )
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        valid_rows = [r for r in rows if not r.is_quarantined]
        quarantined_rows = [r for r in rows if r.is_quarantined]
        assert len(valid_rows) == 2
        assert len(quarantined_rows) == 1


# ─────────────────────────────────────────────────────────────────────────
# load() tests — FetchXML mode
# ─────────────────────────────────────────────────────────────────────────


class TestDataverseSourceLoadFetchXML:
    """Tests for load() with FetchXML queries."""

    def test_fetchxml_load_single_page(self) -> None:
        """FetchXML load yields rows from a single page."""
        pages = [
            _make_page(
                [
                    {"contactid": "1", "fullname": "Alice"},
                ]
            ),
        ]
        source = _make_source_for_load(pages, _fetchxml_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert not rows[0].is_quarantined

    def test_fetchxml_missing_entity_element(self) -> None:
        """FetchXML without <entity> element raises RuntimeError."""
        source = _make_source_for_load(
            [],
            _fetchxml_config(
                fetch_xml="<fetch><all-attributes/></fetch>",
            ),
        )
        ctx = _mock_source_context()

        with pytest.raises(RuntimeError, match="missing <entity> element"):
            list(source.load(ctx))

    def test_fetchxml_entity_missing_name_attribute(self) -> None:
        """FetchXML <entity> without name attribute raises RuntimeError."""
        source = _make_source_for_load(
            [],
            _fetchxml_config(
                fetch_xml="<fetch><entity><attribute name='fullname'/></entity></fetch>",
            ),
        )
        ctx = _mock_source_context()

        with pytest.raises(RuntimeError, match="missing 'name' attribute"):
            list(source.load(ctx))

    def test_fetchxml_uses_paginate_fetchxml(self) -> None:
        """FetchXML mode calls client.paginate_fetchxml, not paginate_odata."""
        pages = [_make_page([{"contactid": "1"}])]
        source = _make_source_for_load(pages, _fetchxml_config())
        ctx = _mock_source_context()

        list(source.load(ctx))
        source._client.paginate_fetchxml.assert_called_once()
        source._client.paginate_odata.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────
# Schema contract locking tests
# ─────────────────────────────────────────────────────────────────────────


class TestSchemaContractLocking:
    """Tests for schema contract locking behavior during load()."""

    def test_contract_locked_after_first_valid_row(self) -> None:
        """Contract builder processes first valid row and sets the flag."""
        pages = [_make_page([{"contactid": "1", "fullname": "Alice"}])]
        source = _make_source_for_load(pages, _base_config())
        assert source._contract_builder is not None

        ctx = _mock_source_context()
        rows = list(source.load(ctx))

        assert len(rows) == 1
        assert source._first_valid_row_processed is True

    def test_contract_flag_set_on_first_row_only(self) -> None:
        """_first_valid_row_processed is set on first valid row, not re-triggered."""
        pages = [
            _make_page(
                [
                    {"contactid": "1", "fullname": "Alice"},
                    {"contactid": "2", "fullname": "Bob"},
                ]
            )
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 2
        # Flag was set on the first row
        assert source._first_valid_row_processed is True

    def test_schema_lock_scoping_is_instance_level(self) -> None:
        """_first_valid_row_processed is instance-level, not reset per page.

        The flag is set on the first valid row of the first page and
        remains True across subsequent pages.
        """
        pages = [
            _make_page(
                [{"contactid": "1", "fullname": "Alice"}],
                next_link="https://myorg.crm.dynamics.com/next",
            ),
            _make_page([{"contactid": "2", "fullname": "Bob"}]),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 2
        # Flag set on first page, first row — stays True through second page
        assert source._first_valid_row_processed is True

    def test_contract_force_locked_when_no_valid_rows(self) -> None:
        """Contract is force-locked when load() yields no valid rows."""
        pages = [_make_page([])]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        list(source.load(ctx))
        assert source._first_valid_row_processed is False
        # Contract should have been set on source (via set_schema_contract)
        # even though no valid rows were processed
        contract = source.get_schema_contract()
        assert contract is not None

    def test_first_valid_row_flag_resets_per_load_call(self) -> None:
        """_first_valid_row_processed resets at the start of each load() call."""
        pages = [_make_page([{"contactid": "1"}])]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        # First load
        list(source.load(ctx))
        assert source._first_valid_row_processed is True

        # The flag is reset inside load() (line 450)
        source._client.paginate_odata.return_value = iter([_make_page([{"contactid": "2"}])])
        list(source.load(ctx))
        # Second load resets the flag, then processes and sets it again
        assert source._first_valid_row_processed is True

    def test_valid_rows_carry_contract(self) -> None:
        """Each valid SourceRow carries the schema contract."""
        pages = [_make_page([{"contactid": "1", "fullname": "Alice"}])]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))
        assert len(rows) == 1
        assert rows[0].contract is not None


# ─────────────────────────────────────────────────────────────────────────
# get_field_resolution tests
# ─────────────────────────────────────────────────────────────────────────


class TestGetFieldResolution:
    """Tests for get_field_resolution()."""

    def test_returns_none_before_load(self) -> None:
        """get_field_resolution returns None before any rows are processed."""
        source = _make_source(_base_config())
        assert source.get_field_resolution() is None

    def test_returns_mapping_after_normalization(self) -> None:
        """get_field_resolution returns the mapping and version after rows are processed."""
        source = _make_source(_base_config())
        source._normalize_row_fields({"contactid": "abc"}, is_first_row=True)

        result = source.get_field_resolution()
        assert result is not None
        mapping, version = result
        assert isinstance(mapping, Mapping)
        assert version is not None
        assert isinstance(version, str)

    def test_returns_mapping_with_already_clean_names(self) -> None:
        """get_field_resolution works when field names are already clean identifiers."""
        source = _make_source(_base_config())
        source._normalize_row_fields({"contactid": "abc"}, is_first_row=True)

        result = source.get_field_resolution()
        assert result is not None
        mapping, _ = result
        assert mapping["contactid"] == "contactid"


# ─────────────────────────────────────────────────────────────────────────
# close() tests
# ─────────────────────────────────────────────────────────────────────────


class TestDataverseSourceClose:
    """Tests for close() resource cleanup."""

    def test_close_releases_client(self) -> None:
        """close() calls client.close() and sets client to None."""
        source = _make_source(_base_config())
        mock_client = MagicMock()
        source._client = mock_client

        source.close()

        mock_client.close.assert_called_once()
        assert source._client is None

    def test_close_when_no_client(self) -> None:
        """close() is safe when client was never created."""
        source = _make_source(_base_config())
        assert source._client is None
        source.close()  # Should not raise


# ─────────────────────────────────────────────────────────────────────────
# Record page call tests
# ─────────────────────────────────────────────────────────────────────────


class TestRecordPageCall:
    """Tests for _record_page_call audit recording."""

    def _make_source_with_client(self) -> Any:
        source = _make_source(_base_config())
        source._client = MagicMock()
        source._client.get_auth_headers.return_value = {"Authorization": "Bearer test"}
        return source

    def test_record_success(self) -> None:
        """Success page records CallStatus.SUCCESS with row count.

        Headers are already fingerprinted by the client before reaching
        the DataversePageResponse DTO — the source passes them through.
        """
        from elspeth.contracts import CallStatus, CallType

        source = self._make_source_with_client()
        ctx = _mock_source_context()

        page = _make_page([{"id": "1"}, {"id": "2"}], latency_ms=42.0)
        source._record_page_call(ctx, url="https://test.com/api", page=page)

        ctx.record_call.assert_called_once()
        call_kwargs = ctx.record_call.call_args
        assert call_kwargs.kwargs["call_type"] == CallType.HTTP
        assert call_kwargs.kwargs["status"] == CallStatus.SUCCESS
        assert call_kwargs.kwargs["response_data"]["row_count"] == 2
        assert call_kwargs.kwargs["latency_ms"] == 42.0
        # Headers passed through as-is (already fingerprinted by client)
        assert call_kwargs.kwargs["request_data"]["headers"] == page.request_headers

    def test_record_success_includes_odata_pagination_metadata(self) -> None:
        """Success audit payload keeps lightweight pagination metadata."""
        source = self._make_source_with_client()
        ctx = _mock_source_context()

        page = DataversePageResponse(
            status_code=200,
            rows=[{"id": "1"}],
            latency_ms=42.0,
            headers={"content-type": "application/json", "odata-version": "4.0"},
            request_headers={"Authorization": "<fingerprint:test-fake>"},
            request_url="https://test.com/api?$skiptoken=abc",
            next_link="https://test.com/api?$skiptoken=def",
            paging_cookie=None,
            more_records=None,
        )
        source._record_page_call(ctx, url=page.request_url, page=page)

        response_data = ctx.record_call.call_args.kwargs["response_data"]
        assert response_data["headers"] == page.headers
        assert response_data["next_link"] == page.next_link
        assert response_data["paging_cookie"] is None
        assert response_data["more_records"] is None

    def test_record_success_includes_fetchxml_pagination_metadata(self) -> None:
        """FetchXML call records paging cookie and explicit terminal flag."""
        source = self._make_source_with_client()
        ctx = _mock_source_context()

        page = DataversePageResponse(
            status_code=200,
            rows=[{"id": "1"}],
            latency_ms=42.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "<fingerprint:test-fake>"},
            request_url="https://test.com/api",
            next_link=None,
            paging_cookie="cookie-123",
            more_records=False,
        )
        source._record_page_call(ctx, url=page.request_url, page=page)

        response_data = ctx.record_call.call_args.kwargs["response_data"]
        assert response_data["paging_cookie"] == "cookie-123"
        assert response_data["more_records"] is False

    def test_record_error(self) -> None:
        """Error page records CallStatus.ERROR with error details."""
        from elspeth.contracts import CallStatus, CallType

        source = self._make_source_with_client()
        ctx = _mock_source_context()

        error = DataverseClientError("Server error", retryable=True, status_code=500, latency_ms=100.0)
        source._record_page_call(ctx, url="https://test.com/api", error=error, error_reason="pagination_error")

        ctx.record_call.assert_called_once()
        call_kwargs = ctx.record_call.call_args
        assert call_kwargs.kwargs["call_type"] == CallType.HTTP
        assert call_kwargs.kwargs["status"] == CallStatus.ERROR
        assert call_kwargs.kwargs["error"]["status_code"] == 500

    def test_no_recording_when_no_args(self) -> None:
        """No record_call when neither page, error, nor error_reason provided."""
        source = self._make_source_with_client()
        ctx = _mock_source_context()

        source._record_page_call(ctx, url="https://test.com/api")
        ctx.record_call.assert_not_called()

    def test_record_success_emits_single_completion_event_via_plugin_context(self) -> None:
        """Real PluginContext path emits one completion event per Dataverse page call."""
        source = self._make_source_with_client()
        events: list[Any] = []
        ctx = _plugin_context_for_operation_calls(telemetry_emit=events.append)

        page = _make_page([{"id": "1"}], latency_ms=42.0)
        source._record_page_call(ctx, url="https://test.com/api", page=page)

        assert len(events) == 1


# ---------------------------------------------------------------------------
# Bug fix: normalize_field_name quarantine (elspeth-4668a0f43d)
# ---------------------------------------------------------------------------


class TestNormalizeFieldQuarantine:
    """Verify that normalize_field_name errors quarantine the row, not crash the load."""

    def test_normalize_row_fields_raises_on_empty_field_name(self) -> None:
        """The primitive _normalize_row_fields raises ValueError for unnormalizable fields.

        This tests the mechanism — that ValueError propagates from
        normalize_field_name. The quarantine behavior (catching this
        ValueError in load()) is tested separately below.
        """
        source = _make_source(_base_config())

        # Build the initial resolution mapping from a valid row
        source._normalize_row_fields({"contactid": "abc"}, is_first_row=True)

        # A subsequent row with a new field that normalizes to empty raises
        with pytest.raises(ValueError, match="empty"):
            source._normalize_row_fields({"!!!": "value", "contactid": "abc"}, is_first_row=False)

    def test_normalize_error_in_load_yields_quarantined_row(self) -> None:
        """In the full load() path, normalize errors yield quarantined rows."""
        source = _make_source(_base_config())
        ctx = _mock_source_context()

        # Mock client to return a page with a problematic field
        mock_client = MagicMock()
        source._client = mock_client

        # Page with a row that has an unnormalizable field name
        page = _make_page([{"!!!": "trash-field-name"}])
        mock_client.paginate_odata.return_value = iter([page])

        rows = list(source.load(ctx))

        # Row should be quarantined, not crash
        assert len(rows) == 1
        assert rows[0].is_quarantined is True
        assert rows[0].quarantine_error is not None
        assert "Field normalization failed" in rows[0].quarantine_error
        assert rows[0].quarantine_destination == "quarantine"

        # Validation error should be recorded in audit trail
        ctx.record_validation_error.assert_called_once()
        call_kwargs = ctx.record_validation_error.call_args.kwargs
        assert call_kwargs["schema_mode"] == "field_normalization"


# ---------------------------------------------------------------------------
# Bug fix: audit trail records actual URL per page (elspeth-fafd21248c)
# ---------------------------------------------------------------------------


class TestAuditUrlPerPage:
    """Verify audit trail records the actual URL for each page, not the initial URL."""

    def test_record_page_call_uses_page_request_url(self) -> None:
        """_record_page_call receives page.request_url, not rebuilt URL."""
        source = self._make_source_with_client()
        ctx = _mock_source_context()

        next_url = "https://test.crm.dynamics.com/api/data/v9.2/contacts?$skiptoken=abc"
        # Simulate a page 2 response with a different request_url
        page2 = DataversePageResponse(
            status_code=200,
            rows=[{"id": "2"}],
            latency_ms=15.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "[FINGERPRINT]"},
            request_url=next_url,
            next_link=None,
            paging_cookie=None,
            more_records=False,
        )

        source._record_page_call(ctx, url=page2.request_url, page=page2)

        call_kwargs = ctx.record_call.call_args.kwargs
        assert call_kwargs["request_data"]["url"] == next_url

    def _make_source_with_client(self) -> Any:
        source = _make_source(_base_config())
        source._client = MagicMock()
        source._client.get_auth_headers.return_value = {"Authorization": "Bearer test"}
        return source


# ---------------------------------------------------------------------------
# Bug fix: URL percent-encoding (elspeth-5bd2873b3e)
# ---------------------------------------------------------------------------


class TestUrlPercentEncoding:
    """Entity names and filter values are percent-encoded in URLs."""

    def test_filter_with_single_quote_encoded(self) -> None:
        """Single quotes in $filter are percent-encoded."""
        source = _make_source(_base_config(filter="name eq 'O''Brien'"))
        url = source._build_query_url()
        # Single quotes should be encoded as %27
        assert "%27" in url
        assert "'" not in url.split("$filter=")[1]

    def test_filter_with_ampersand_encoded(self) -> None:
        """Ampersand in $filter can't break URL structure."""
        source = _make_source(_base_config(filter="name eq 'A&B'"))
        url = source._build_query_url()
        # The & inside the filter value must be encoded as %26
        filter_part = url.split("$filter=")[1]
        assert "&" not in filter_part  # No raw ampersand in filter value
        assert "%26" in filter_part

    def test_filter_with_hash_encoded(self) -> None:
        """Hash in $filter can't truncate URL as fragment."""
        source = _make_source(_base_config(filter="name eq 'test#1'"))
        url = source._build_query_url()
        assert "%23" in url
        assert "#" not in url.split("$filter=")[1]

    def test_entity_name_in_path_encoded(self) -> None:
        """Entity name with special chars is percent-encoded in path segment."""
        source = _make_source(_base_config(entity="my entity"))
        url = source._build_query_url()
        assert "/my%20entity" in url  # Slash-anchored: confirms it's in the path
        assert "my entity" not in url

    def test_normal_entity_unchanged(self) -> None:
        """Normal entity names (alphanumeric) pass through unmodified."""
        source = _make_source(_base_config(entity="contacts"))
        url = source._build_query_url()
        assert "/contacts" in url

    def test_orderby_with_special_chars_encoded(self) -> None:
        """$orderby values with special characters are percent-encoded."""
        source = _make_source(_base_config(orderby="name desc, 'special'"))
        url = source._build_query_url()
        assert "%27" in url  # single quote encoded
        assert "%20" in url  # space encoded

    def test_select_identifiers_not_encoded(self) -> None:
        """$select column names (Dataverse identifiers) are NOT encoded."""
        source = _make_source(_base_config(select=["contactid", "fullname"]))
        url = source._build_query_url()
        assert "$select=contactid,fullname" in url  # literal, no encoding


# ---------------------------------------------------------------------------
# Bug fix: unmapped fields trigger field resolution rebuild (review finding)
# ---------------------------------------------------------------------------


class TestUnmappedFieldsRebuildResolution:
    """Verify that rows with fields not in the initial mapping trigger a
    _field_resolution rebuild so new fields are correctly normalized."""

    def test_new_field_on_page2_is_normalized(self) -> None:
        """When page 2 introduces a field not in page 1, the new field
        appears correctly normalized in output rows and the field
        resolution mapping is rebuilt to include it."""
        pages = [
            _make_page(
                [{"contactid": "1", "fullname": "Alice"}],
                next_link="https://myorg.crm.dynamics.com/api/data/v9.2/contacts?$skiptoken=1",
            ),
            _make_page(
                [{"contactid": "2", "fullname": "Bob", "jobtitle": "Engineer"}],
            ),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))

        assert len(rows) == 2
        # All rows should be valid (not quarantined)
        assert all(not r.is_quarantined for r in rows)

        # Page 1 row has only contactid and fullname
        assert "contactid" in rows[0].row
        assert "fullname" in rows[0].row

        # Page 2 row has the new field "jobtitle" correctly normalized
        assert "jobtitle" in rows[1].row
        assert rows[1].row["jobtitle"] == "Engineer"

        # Field resolution should have been rebuilt to include jobtitle
        resolution = source.get_field_resolution()
        assert resolution is not None
        mapping, _ = resolution
        assert "jobtitle" in mapping

    def test_new_field_with_non_identifier_name_is_normalized(self) -> None:
        """New fields with non-identifier names (e.g., spaces) are
        normalized through the same algorithm on rebuild."""
        pages = [
            _make_page(
                [{"contactid": "1", "fullname": "Alice"}],
                next_link="https://myorg.crm.dynamics.com/next",
            ),
            _make_page(
                [{"contactid": "2", "fullname": "Bob", "Job Title": "Engineer"}],
            ),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        rows = list(source.load(ctx))

        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)

        # The new field should be normalized (spaces -> underscore, lowered)
        row2 = rows[1].row
        assert "job_title" in row2
        assert row2["job_title"] == "Engineer"

        # Resolution mapping should include the new field after rebuild
        resolution = source.get_field_resolution()
        assert resolution is not None
        mapping, _ = resolution
        assert "Job Title" in mapping
        assert mapping["Job Title"] == "job_title"

    def test_resolution_not_rebuilt_when_fields_match(self) -> None:
        """Field resolution is NOT rebuilt when a subsequent row has the
        same fields as the initial mapping."""
        pages = [
            _make_page(
                [{"contactid": "1", "fullname": "Alice"}],
                next_link="https://myorg.crm.dynamics.com/next",
            ),
            _make_page(
                [{"contactid": "2", "fullname": "Bob"}],
            ),
        ]
        source = _make_source_for_load(pages, _base_config())
        ctx = _mock_source_context()

        # Process first page to establish initial resolution
        rows = list(source.load(ctx))
        assert len(rows) == 2

        # Resolution should not have changed (same fields on both pages)
        resolution = source.get_field_resolution()
        assert resolution is not None
        mapping, _ = resolution
        # Only the original fields should be in the mapping
        assert set(mapping.keys()) == {"contactid", "fullname"}
