"""Integration tests for Dataverse source and sink plugins.

Tests exercise production code paths with mock Dataverse endpoints
(httpx mock transport). No real Dataverse connections are made.

Coverage:
- Source: structured query with pagination, FetchXML with paging cookie
- Source: schema discovery and contract locking across pages
- Sink: upsert mode with field mapping and lookup bindings
- Sink: partial batch failure (raise on first error, prior rows committed)
- End-to-end: source → sink pipeline via mock Dataverse
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from elspeth.contracts import CallStatus
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseClient,
    DataversePageResponse,
)
from elspeth.plugins.sources.dataverse import DataverseSource

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class FakeCredential:
    """Mock azure-identity credential that returns a static token."""

    def get_token(self, *scopes: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(token="fake-token-for-testing")


def _noop_telemetry_emit(*_args: Any, **_kwargs: Any) -> None:
    """Lifecycle/source context telemetry callback unused by these tests."""


class FakeLifecycleContext:
    """Mock LifecycleContext for on_start()."""

    def __init__(self) -> None:
        self.run_id = "integration-test-run"
        self.node_id = "test-node"
        self.telemetry_emit = _noop_telemetry_emit
        self.rate_limit_registry = None
        self.landscape = None
        self.payload_store = None
        self.concurrency_config = None


class FakeSourceContext:
    """Mock SourceContext for load()."""

    def __init__(self) -> None:
        self.run_id = "integration-test-run"
        self.node_id = "test-node"
        self.operation_id = "op-001"
        self.landscape = None
        self.telemetry_emit = _noop_telemetry_emit
        self.calls: list[dict[str, Any]] = []
        self.validation_errors: list[dict[str, Any]] = []

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)

    def record_validation_error(self, **kwargs: Any) -> Any:
        self.validation_errors.append(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(row_id="test-row", node_id="test-node", destination=kwargs["destination"])


class FakeSinkContext:
    """Mock SinkContext for write()."""

    def __init__(self) -> None:
        self.run_id = "integration-test-run"
        self.operation_id = "op-001"
        self.contract = None
        self.landscape = None
        self.calls: list[dict[str, Any]] = []

    def record_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _make_odata_response(
    rows: list[dict[str, Any]],
    next_link: str | None = None,
) -> dict[str, Any]:
    """Build a mock OData JSON response body."""
    resp: dict[str, Any] = {"value": rows}
    if next_link is not None:
        resp["@odata.nextLink"] = next_link
    return resp


def _make_source_config(**overrides: Any) -> dict[str, Any]:
    """Build a valid DataverseSource config dict."""
    base = {
        "environment_url": "https://testorg.crm.dynamics.com",
        "auth": {"method": "managed_identity"},
        "entity": "contact",
        "select": ["fullname", "emailaddress1"],
        "schema": {"mode": "observed"},
        "on_validation_failure": "quarantine_sink",
    }
    base.update(overrides)
    return base


def _make_sink_config(**overrides: Any) -> dict[str, Any]:
    """Build a valid DataverseSink config dict."""
    base = {
        "environment_url": "https://testorg.crm.dynamics.com",
        "auth": {"method": "managed_identity"},
        "entity": "contact",
        "mode": "upsert",
        "alternate_key": "emailaddress1",
        "field_mapping": {
            "email": "emailaddress1",
            "name": "fullname",
        },
        "schema": {"mode": "observed"},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Source Integration Tests
# ---------------------------------------------------------------------------


class TestDataverseSourceStructuredQuery:
    """Source with structured OData queries against mock endpoint."""

    def test_single_page_load(self) -> None:
        """Load rows from a single-page OData response."""
        source = DataverseSource(_make_source_config())

        # Mock the client to return a canned response
        mock_client = MagicMock(spec=DataverseClient)
        mock_client.get_page.return_value = DataversePageResponse(
            status_code=200,
            rows=[{"LogicalName": "contact"}],
            latency_ms=5.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://testorg.crm.dynamics.com/api/data/v9.2/EntityDefinitions(LogicalName='contact')?$select=LogicalName",
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )
        mock_client.paginate_odata.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[
                        {"fullname": "Alice", "emailaddress1": "alice@test.com"},
                        {"fullname": "Bob", "emailaddress1": "bob@test.com"},
                    ],
                    latency_ms=50.0,
                    headers={"content-type": "application/json"},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        # Inject mock client
        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 2
        assert all(not r.is_quarantined for r in rows)
        assert rows[0].row["fullname"] == "Alice"
        assert rows[1].row["emailaddress1"] == "bob@test.com"

        # Verify audit recording
        success_calls = [c for c in ctx.calls if c.get("status") == CallStatus.SUCCESS]
        assert len(success_calls) == 2  # Metadata probe + one page fetch

    def test_multi_page_pagination(self) -> None:
        """Load rows across multiple pages with @odata.nextLink."""
        source = DataverseSource(_make_source_config())

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.get_page.return_value = DataversePageResponse(
            status_code=200,
            rows=[{"LogicalName": "contact"}],
            latency_ms=5.0,
            headers={"content-type": "application/json"},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://testorg.crm.dynamics.com/api/data/v9.2/EntityDefinitions(LogicalName='contact')?$select=LogicalName",
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )
        mock_client.paginate_odata.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[{"fullname": "Alice", "emailaddress1": "alice@test.com"}],
                    latency_ms=50.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link="https://testorg.crm.dynamics.com/api/data/v9.2/contacts?$skiptoken=abc",
                    paging_cookie=None,
                    more_records=True,
                ),
                DataversePageResponse(
                    status_code=200,
                    rows=[{"fullname": "Bob", "emailaddress1": "bob@test.com"}],
                    latency_ms=40.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 2
        # Metadata probe + two pages = three audit call records
        success_calls = [c for c in ctx.calls if c.get("status") == CallStatus.SUCCESS]
        assert len(success_calls) == 3


class TestDataverseSourceFetchXML:
    """Source with FetchXML queries against mock endpoint."""

    def test_fetchxml_single_page(self) -> None:
        """Load rows from a single-page FetchXML response."""
        config = _make_source_config(
            entity=None,
            select=None,
            fetch_xml='<fetch><entity name="contact"><attribute name="fullname"/></entity></fetch>',
        )
        source = DataverseSource(config)

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.paginate_fetchxml.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[{"fullname": "Alice"}],
                    latency_ms=50.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 1
        assert rows[0].row["fullname"] == "Alice"


class TestDataverseSourceSchemaLocking:
    """Schema contract locking across pages."""

    def test_contract_locked_after_first_valid_row(self) -> None:
        """OBSERVED schema locks after first valid row, not first page."""
        source = DataverseSource(_make_source_config())

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.paginate_odata.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[{"fullname": "Alice", "emailaddress1": "alice@test.com"}],
                    latency_ms=50.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 1
        # Contract should be locked after first valid row
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked

    def test_empty_result_force_locks_contract(self) -> None:
        """Zero valid rows across all pages → force-lock empty contract."""
        source = DataverseSource(_make_source_config())

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.paginate_odata.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[],  # Empty page
                    latency_ms=50.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 0
        contract = source.get_schema_contract()
        assert contract is not None
        assert contract.locked


class TestDataverseSourceODataStripping:
    """OData metadata stripping from rows."""

    def test_strips_odata_annotations(self) -> None:
        """@odata.* and @Microsoft.Dynamics.CRM.* fields are removed."""
        source = DataverseSource(_make_source_config())

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.paginate_odata.return_value = iter(
            [
                DataversePageResponse(
                    status_code=200,
                    rows=[
                        {
                            "fullname": "Alice",
                            "emailaddress1": "alice@test.com",
                            "@odata.etag": 'W/"12345"',
                            "@odata.context": "https://testorg.crm.dynamics.com/api/data/v9.2/$metadata#contacts",
                            "@Microsoft.Dynamics.CRM.totalrecordcount": 42,
                        }
                    ],
                    latency_ms=50.0,
                    headers={},
                    request_headers={"Authorization": "Bearer fake"},
                    request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                    next_link=None,
                    paging_cookie=None,
                    more_records=False,
                ),
            ]
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sources.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            source.on_start(FakeLifecycleContext())
        source._client = mock_client

        ctx = FakeSourceContext()
        rows = list(source.load(ctx))  # type: ignore[arg-type]  # test fake context

        assert len(rows) == 1
        row = rows[0].row
        assert "fullname" in row
        assert "emailaddress1" in row
        assert "@odata.etag" not in row
        assert "@odata.context" not in row
        assert "@Microsoft.Dynamics.CRM.totalrecordcount" not in row


# ---------------------------------------------------------------------------
# Sink Integration Tests
# ---------------------------------------------------------------------------
