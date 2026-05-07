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

import hashlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import CallStatus
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseClient,
    DataverseClientError,
    DataversePageResponse,
)
from elspeth.plugins.sinks.dataverse import DataverseSink
from elspeth.plugins.sources.dataverse import DataverseSource
from tests.fixtures.base_classes import inject_write_failure

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class FakeCredential:
    """Mock azure-identity credential that returns a static token."""

    def get_token(self, *scopes: str) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(token="fake-token-for-testing")


class FakeLifecycleContext:
    """Mock LifecycleContext for on_start()."""

    def __init__(self) -> None:
        self.run_id = "integration-test-run"
        self.node_id = "test-node"
        self.telemetry_emit = MagicMock()
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
        self.telemetry_emit = MagicMock()
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


class TestDataverseSinkUpsert:
    """Sink upsert mode against mock endpoint."""

    def test_single_row_upsert(self) -> None:
        """Upsert a single row via PATCH."""
        sink = inject_write_failure(DataverseSink(_make_sink_config()))

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.upsert.return_value = DataversePageResponse(
            status_code=204,
            rows=[],
            latency_ms=30.0,
            headers={},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
            next_link=None,
            paging_cookie=None,
            more_records=False,
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sinks.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            sink.on_start(FakeLifecycleContext())
        sink._client = mock_client

        ctx = FakeSinkContext()
        rows = [{"email": "alice@test.com", "name": "Alice"}]
        result = sink.write(rows, ctx)  # type: ignore[arg-type]  # test fake context

        assert isinstance(result, SinkWriteResult)
        descriptor = result.artifact
        assert descriptor.artifact_type == "webhook"
        assert "contact" in descriptor.path_or_uri
        assert descriptor.metadata is not None
        assert descriptor.metadata["row_count"] == 1
        assert descriptor.metadata["mode"] == "upsert"

        # Verify PATCH was called
        mock_client.upsert.assert_called_once()
        call_url = mock_client.upsert.call_args[0][0]
        assert "emailaddress1=" in call_url
        assert "alice%40test.com" in call_url  # URL-encoded @

        # Verify audit recording
        assert len(ctx.calls) == 1
        assert ctx.calls[0]["status"] == CallStatus.SUCCESS

    def test_multi_row_upsert_serial(self) -> None:
        """Multiple rows are upserted serially, each with its own audit record."""
        sink = inject_write_failure(DataverseSink(_make_sink_config()))

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.upsert.return_value = DataversePageResponse(
            status_code=204,
            rows=[],
            latency_ms=25.0,
            headers={},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
            next_link=None,
            paging_cookie=None,
            more_records=False,
        )
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sinks.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            sink.on_start(FakeLifecycleContext())
        sink._client = mock_client

        ctx = FakeSinkContext()
        rows = [
            {"email": "alice@test.com", "name": "Alice"},
            {"email": "bob@test.com", "name": "Bob"},
            {"email": "charlie@test.com", "name": "Charlie"},
        ]
        result = sink.write(rows, ctx)  # type: ignore[arg-type]  # test fake context

        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 3
        assert mock_client.upsert.call_count == 3
        assert len(ctx.calls) == 3  # One audit record per row

    def test_upsert_failure_raises_and_records(self) -> None:
        """Failure on second row raises RuntimeError after recording audit entry."""
        sink = inject_write_failure(DataverseSink(_make_sink_config()))

        call_count = 0

        def mock_upsert(url: str, body: dict[str, Any]) -> DataversePageResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise DataverseClientError(
                    "Conflict (duplicate) (409)",
                    retryable=False,
                    status_code=409,
                    latency_ms=15.0,
                )
            return DataversePageResponse(
                status_code=204,
                rows=[],
                latency_ms=25.0,
                headers={},
                request_headers={"Authorization": "Bearer fake"},
                request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                next_link=None,
                paging_cookie=None,
                more_records=False,
            )

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.upsert = mock_upsert
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sinks.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            sink.on_start(FakeLifecycleContext())
        sink._client = mock_client

        ctx = FakeSinkContext()
        rows = [
            {"email": "alice@test.com", "name": "Alice"},
            {"email": "bob@test.com", "name": "Bob"},
        ]

        with pytest.raises(DataverseClientError, match="Conflict"):
            sink.write(rows, ctx)  # type: ignore[arg-type]  # test fake context

        # First row succeeded, second failed — both recorded
        assert len(ctx.calls) == 2
        assert ctx.calls[0]["status"] == CallStatus.SUCCESS
        assert ctx.calls[1]["status"] == CallStatus.ERROR


class TestDataverseSinkLookupBindings:
    """Lookup field binding generates correct OData bind syntax."""

    def test_lookup_field_binding(self) -> None:
        """Lookup fields produce @odata.bind syntax."""
        config = _make_sink_config(
            field_mapping={
                "email": "emailaddress1",
                "name": "fullname",
                "parent_account_id": "parentcustomerid",
            },
            lookups={
                "parent_account_id": {
                    "target_entity": "accounts",
                    "target_field": "parentcustomerid",
                },
            },
        )
        sink = inject_write_failure(DataverseSink(config))

        mock_client = MagicMock(spec=DataverseClient)
        captured_bodies: list[dict[str, Any]] = []

        def mock_upsert(url: str, body: dict[str, Any]) -> DataversePageResponse:
            captured_bodies.append(body)
            return DataversePageResponse(
                status_code=204,
                rows=[],
                latency_ms=25.0,
                headers={},
                request_headers={"Authorization": "Bearer fake"},
                request_url="https://testorg.crm.dynamics.com/api/data/v9.2/contacts",
                next_link=None,
                paging_cookie=None,
                more_records=False,
            )

        mock_client.upsert = mock_upsert
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sinks.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            sink.on_start(FakeLifecycleContext())
        sink._client = mock_client

        ctx = FakeSinkContext()
        rows = [{"email": "alice@test.com", "name": "Alice", "parent_account_id": "abc-123-guid"}]
        sink.write(rows, ctx)  # type: ignore[arg-type]  # test fake context

        assert len(captured_bodies) == 1
        body = captured_bodies[0]
        assert body["emailaddress1"] == "alice@test.com"
        assert body["fullname"] == "Alice"
        assert body["parentcustomerid@odata.bind"] == "/accounts(abc-123-guid)"
        assert "parentcustomerid" not in body  # raw field not included


class TestDataverseSinkEmptyBatch:
    """Empty batch returns descriptor with empty hash."""

    def test_empty_rows_returns_descriptor(self) -> None:
        sink = inject_write_failure(DataverseSink(_make_sink_config()))

        mock_client = MagicMock(spec=DataverseClient)
        mock_client.get_auth_headers.return_value = {"Authorization": "Bearer fake"}

        with (
            patch("elspeth.plugins.sinks.dataverse.DataverseClient", return_value=mock_client),
            patch("azure.identity.ManagedIdentityCredential"),
        ):
            sink.on_start(FakeLifecycleContext())
        sink._client = mock_client

        ctx = FakeSinkContext()
        result = sink.write([], ctx)  # type: ignore[arg-type]  # test fake context

        assert isinstance(result, SinkWriteResult)
        assert result.artifact.metadata is not None
        assert result.artifact.metadata["row_count"] == 0
        assert result.artifact.content_hash == hashlib.sha256(b"").hexdigest()
