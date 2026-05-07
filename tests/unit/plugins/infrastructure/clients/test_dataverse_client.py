"""Tests for DataverseClient — OData v4 REST API client for Microsoft Dataverse.

Covers all spec-required scenarios:
- DataverseAuthConfig validation (service principal missing fields, managed identity, mutual exclusion)
- Error classification (status code -> retryable/non-retryable, 401 force-refresh path)
- Pagination link following (mock responses with @odata.nextLink)
- @odata.nextLink SSRF validation (reject cross-host nextLink URLs)
- FetchXML paging cookie injection via ElementTree (cookie with XML metacharacters)
- Retry-After cap enforcement (value exceeding max -> non-retryable)
- DataversePageResponse invariants (next_link and paging_cookie mutually exclusive)
- Domain allowlist validation (valid Dataverse domains, rejected non-Dataverse domains)
- Empty-page guard (3 consecutive empty pages -> error)
- JSON parse strict rejection (NaN/Infinity -> DataverseClientError)
- Redirect rejection (3xx -> non-retryable error)
"""

from __future__ import annotations

import json
import urllib.parse
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest
from elspeth.plugins.infrastructure.clients.dataverse import (
    DataverseAuthConfig,
    DataverseClient,
    DataverseClientError,
    DataversePageResponse,
    _validate_domain_allowlist,
    validate_additional_domain,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


class FakeCredential:
    """Mock azure-identity credential that returns a static token."""

    def get_token(self, *scopes: str) -> SimpleNamespace:
        return SimpleNamespace(token="fake-token-123")


def _make_json_response(
    body: dict[str, Any] | list[Any],
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build an httpx.Response with JSON body."""
    resp_headers = {"content-type": "application/json"}
    if headers:
        resp_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        json=body,
        headers=resp_headers,
    )


def _make_text_response(
    text: str,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build an httpx.Response with raw text body."""
    resp_headers = {"content-type": "application/json"}
    if headers:
        resp_headers.update(headers)
    return httpx.Response(
        status_code=status_code,
        text=text,
        headers=resp_headers,
    )


def _make_empty_response(
    status_code: int = 204,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build an httpx.Response with no body (204 No Content style)."""
    return httpx.Response(
        status_code=status_code,
        text="",
        headers=headers or {},
    )


class MockTransport(httpx.BaseTransport):
    """httpx mock transport that returns pre-configured responses."""

    def __init__(self, responses: list[httpx.Response] | None = None) -> None:
        self._responses = list(responses or [])
        self._call_index = 0
        self.requests: list[httpx.Request] = []

    def add_response(self, response: httpx.Response) -> None:
        self._responses.append(response)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._call_index >= len(self._responses):
            raise RuntimeError(
                f"MockTransport exhausted: got request #{self._call_index + 1} but only {len(self._responses)} responses configured"
            )
        response = self._responses[self._call_index]
        self._call_index += 1
        return response


ENV_URL = "https://myorg.crm.dynamics.com"


def _make_ssrf_safe(url: str) -> SSRFSafeRequest:
    """Build a fake SSRFSafeRequest that connects to the original URL.

    In tests using MockTransport, we don't need actual IP pinning.
    The SSRFSafeRequest just needs to produce a valid URL that the
    transport can handle. We use 127.0.0.1 as the resolved IP since
    MockTransport ignores the actual connection target.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return SSRFSafeRequest(
        original_url=url,
        resolved_ip="127.0.0.1",
        host_header=hostname,
        port=port,
        path=path,
        scheme=parsed.scheme,
        bare_hostname=hostname,
    )


@pytest.fixture(autouse=True)
def _bypass_ssrf_dns():
    """Dataverse unit tests use MockTransport; bypass live DNS pinning by default."""
    with patch(
        "elspeth.plugins.infrastructure.clients.dataverse.validate_url_for_ssrf",
        side_effect=lambda url, **_kwargs: _make_ssrf_safe(url),
    ):
        yield


@pytest.fixture()
def transport() -> MockTransport:
    return MockTransport()


@pytest.fixture()
def client(transport: MockTransport) -> DataverseClient:
    """Create a DataverseClient with mock transport and SSRF validation bypassed."""
    c = DataverseClient(
        environment_url=ENV_URL,
        credential=FakeCredential(),  # type: ignore[arg-type]  # test fake
    )
    # Replace the internal httpx.Client with one using our mock transport
    c._client.close()
    c._client = httpx.Client(transport=transport, timeout=30.0)
    return c


# ---------------------------------------------------------------------------
# DataverseAuthConfig validation
# ---------------------------------------------------------------------------


class TestDataverseAuthConfig:
    """Pydantic model validation for auth configuration."""

    def test_service_principal_valid(self) -> None:
        config = DataverseAuthConfig(
            method="service_principal",
            tenant_id="tid-123",
            client_id="cid-456",
            client_secret="secret-789",
        )
        assert config.method == "service_principal"
        assert config.tenant_id == "tid-123"

    def test_service_principal_missing_tenant_id(self) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            DataverseAuthConfig(
                method="service_principal",
                client_id="cid-456",
                client_secret="secret-789",
            )

    def test_service_principal_missing_client_id(self) -> None:
        with pytest.raises(ValueError, match="client_id"):
            DataverseAuthConfig(
                method="service_principal",
                tenant_id="tid-123",
                client_secret="secret-789",
            )

    def test_service_principal_missing_client_secret(self) -> None:
        with pytest.raises(ValueError, match="client_secret"):
            DataverseAuthConfig(
                method="service_principal",
                tenant_id="tid-123",
                client_id="cid-456",
            )

    def test_service_principal_all_missing(self) -> None:
        with pytest.raises(ValueError, match=r"tenant_id.*client_id.*client_secret"):
            DataverseAuthConfig(method="service_principal")

    def test_service_principal_whitespace_only_rejected(self) -> None:
        """Whitespace-only strings are treated as missing."""
        with pytest.raises(ValueError, match="tenant_id"):
            DataverseAuthConfig(
                method="service_principal",
                tenant_id="   ",
                client_id="cid-456",
                client_secret="secret-789",
            )

    def test_managed_identity_valid(self) -> None:
        config = DataverseAuthConfig(method="managed_identity")
        assert config.method == "managed_identity"
        # SP fields default to None and are not required
        assert config.tenant_id is None

    def test_managed_identity_ignores_sp_fields(self) -> None:
        """managed_identity does not validate SP fields even if provided."""
        config = DataverseAuthConfig(
            method="managed_identity",
            tenant_id="ignored",
        )
        assert config.tenant_id == "ignored"

    def test_invalid_method_rejected(self) -> None:
        with pytest.raises(ValueError):
            DataverseAuthConfig(method="password")

    def test_extra_fields_rejected(self) -> None:
        """extra='forbid' prevents unknown fields."""
        with pytest.raises(ValueError):
            DataverseAuthConfig(
                method="managed_identity",
                unknown_field="value",  # type: ignore[call-arg]
            )


# ---------------------------------------------------------------------------
# DataversePageResponse invariants
# ---------------------------------------------------------------------------


class TestDataversePageResponse:
    """Dataclass invariant validation."""

    def test_next_link_and_paging_cookie_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            DataversePageResponse(
                status_code=200,
                rows=[],
                latency_ms=10.0,
                headers={},
                request_headers={"Authorization": "Bearer fake"},
                request_url="https://example.crm.dynamics.com/api/data/v9.2/contacts",
                next_link="https://example.crm.dynamics.com/next",
                paging_cookie="<cookie/>",
                more_records=True,
            )

    def test_next_link_only(self) -> None:
        page = DataversePageResponse(
            status_code=200,
            rows=[{"id": 1}],
            latency_ms=10.0,
            headers={},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://example.crm.dynamics.com/api/data/v9.2/contacts",
            next_link="https://example.crm.dynamics.com/next",
            paging_cookie=None,
            more_records=True,
        )
        assert page.next_link is not None
        assert page.paging_cookie is None

    def test_paging_cookie_only(self) -> None:
        page = DataversePageResponse(
            status_code=200,
            rows=[{"id": 1}],
            latency_ms=10.0,
            headers={},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://example.crm.dynamics.com/api/data/v9.2/contacts",
            next_link=None,
            paging_cookie="<cookie/>",
            more_records=True,
        )
        assert page.paging_cookie is not None
        assert page.next_link is None

    def test_neither_pagination_marker(self) -> None:
        page = DataversePageResponse(
            status_code=200,
            rows=[{"id": 1}],
            latency_ms=10.0,
            headers={},
            request_headers={"Authorization": "Bearer fake"},
            request_url="https://example.crm.dynamics.com/api/data/v9.2/contacts",
            next_link=None,
            paging_cookie=None,
            more_records=False,
        )
        assert page.more_records is False


# ---------------------------------------------------------------------------
# Domain allowlist validation
# ---------------------------------------------------------------------------


class TestDomainAllowlist:
    """Domain allowlist pre-filtering for SSRF prevention."""

    @pytest.mark.parametrize(
        "hostname",
        [
            "myorg.crm.dynamics.com",
            "myorg.crm2.dynamics.com",
            "myorg.crm9.dynamics.com",
            "myorg.crm.microsoftdynamics.us",
            "myorg.crm.appsplatform.us",
            "myorg.crm.microsoftdynamics.de",
            "myorg.crm.dynamics.cn",
        ],
    )
    def test_valid_dataverse_domains(self, hostname: str) -> None:
        assert _validate_domain_allowlist(hostname) is True

    @pytest.mark.parametrize(
        "hostname",
        [
            "evil.com",
            "crm.dynamics.com.evil.com",
            "169.254.169.254",
            "localhost",
            "myorg.dynamics.com",  # missing crm. prefix
            "myorg.crm99.dynamics.com",  # not in allowlist
        ],
    )
    def test_rejected_non_dataverse_domains(self, hostname: str) -> None:
        assert _validate_domain_allowlist(hostname) is False

    def test_additional_domains_appended(self) -> None:
        hostname = "custom.crm15.dynamics.com"
        assert _validate_domain_allowlist(hostname) is False
        assert _validate_domain_allowlist(hostname, additional_domains=("*.crm15.dynamics.com",)) is True

    def test_constructor_rejects_non_dataverse_url(self) -> None:
        with pytest.raises(DataverseClientError, match="does not match"):
            DataverseClient(
                environment_url="https://evil.example.com",
                credential=FakeCredential(),  # type: ignore[arg-type]  # test fake
            )

    def test_constructor_rejects_unparseable_url(self) -> None:
        with pytest.raises(DataverseClientError, match="Cannot extract hostname"):
            DataverseClient(
                environment_url="not-a-url",
                credential=FakeCredential(),  # type: ignore[arg-type]  # test fake
            )


class TestValidateAdditionalDomain:
    """Safety regex for user-provided additional domain patterns."""

    @pytest.mark.parametrize(
        "pattern",
        [
            "*.crm.dynamics.com",
            "localhost.dynamics.com",
            "*.crm.dynamics.cn",
            "*.crm.microsoftdynamics.us",
            "*.custom.appsplatform.us",
            "myorg.crm.microsoftdynamics.de",
        ],
    )
    def test_valid_patterns_accepted(self, pattern: str) -> None:
        validate_additional_domain(pattern)  # should not raise

    @pytest.mark.parametrize(
        "pattern",
        [
            "*.evil.com",
            "*.example.org",
            "*.dynamics.evil.com",
            "anything",
            "",
            "evil.com",
        ],
    )
    def test_invalid_patterns_rejected(self, pattern: str) -> None:
        """Patterns not targeting a legitimate Microsoft sovereign cloud TLD are rejected."""
        with pytest.raises(ValueError, match="rejected"):
            validate_additional_domain(pattern)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class TestErrorClassification:
    """HTTP status code to retryable/non-retryable classification."""

    def test_429_retryable_within_cap(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(httpx.Response(status_code=429, headers={"retry-after": "5"}, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 429

    def test_429_exceeding_cap_non_retryable(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(httpx.Response(status_code=429, headers={"retry-after": "300"}, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is False
        assert "exceeding cap" in str(exc_info.value)

    def test_429_no_retry_after_header_defaults(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(httpx.Response(status_code=429, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True

    def test_401_retryable_first_time(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(httpx.Response(status_code=401, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True

    def test_401_non_retryable_after_auth_retry(self, client: DataverseClient, transport: MockTransport) -> None:
        client._auth_retried = True
        transport.add_response(httpx.Response(status_code=401, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is False

    @pytest.mark.parametrize(
        ("status_code", "retryable"),
        [
            (400, False),
            (403, False),
            (404, False),
            (409, False),
            (412, False),
            (500, True),
            (502, True),
            (503, True),
        ],
    )
    def test_status_code_classification(
        self,
        client: DataverseClient,
        transport: MockTransport,
        status_code: int,
        retryable: bool,
    ) -> None:
        transport.add_response(httpx.Response(status_code=status_code, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is retryable
        assert exc_info.value.status_code == status_code


# ---------------------------------------------------------------------------
# Redirect rejection
# ---------------------------------------------------------------------------


class TestRedirectRejection:
    """3xx responses are rejected as non-retryable (SSRF protection)."""

    @pytest.mark.parametrize("status_code", [301, 302, 307, 308])
    def test_redirect_non_retryable(
        self,
        client: DataverseClient,
        transport: MockTransport,
        status_code: int,
    ) -> None:
        transport.add_response(
            httpx.Response(
                status_code=status_code,
                headers={"location": "https://evil.com/steal"},
                text="",
            )
        )
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is False
        assert exc_info.value.status_code == status_code
        assert "Redirect" in str(exc_info.value)


# ---------------------------------------------------------------------------
# JSON parse strict rejection
# ---------------------------------------------------------------------------


class TestJsonParseStrictRejection:
    """NaN/Infinity in JSON responses are rejected at the Tier 3 boundary."""

    def test_nan_in_response_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        # Python json.dumps with allow_nan=True produces NaN literal
        nan_json = json.dumps({"value": [{"x": float("nan")}]}, allow_nan=True)
        transport.add_response(
            httpx.Response(
                status_code=200,
                text=nan_json,
                headers={"content-type": "application/json"},
            )
        )
        with pytest.raises(DataverseClientError, match="Invalid JSON"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

    def test_infinity_in_response_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        inf_json = json.dumps({"value": [{"x": float("inf")}]}, allow_nan=True)
        transport.add_response(
            httpx.Response(
                status_code=200,
                text=inf_json,
                headers={"content-type": "application/json"},
            )
        )
        with pytest.raises(DataverseClientError, match="Invalid JSON"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

    def test_malformed_json_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(
            httpx.Response(
                status_code=200,
                text="{not valid json}",
                headers={"content-type": "application/json"},
            )
        )
        with pytest.raises(DataverseClientError, match="Invalid JSON"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")


# ---------------------------------------------------------------------------
# Successful response parsing
# ---------------------------------------------------------------------------


class TestSuccessfulResponses:
    """Valid response parsing and structure extraction."""

    def test_odata_value_array_extracted(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"value": [{"id": 1}, {"id": 2}]}))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert len(page.rows) == 2
        assert dict(page.rows[0]) == {"id": 1}
        assert page.status_code == 200

    def test_single_record_response(self, client: DataverseClient, transport: MockTransport) -> None:
        """Response without 'value' key is treated as single-record."""
        transport.add_response(_make_json_response({"id": 42, "name": "test"}))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/accounts(42)")
        assert [dict(r) for r in page.rows] == [{"id": 42, "name": "test"}]

    def test_204_no_content(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_empty_response(204))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert len(page.rows) == 0
        assert page.more_records is None  # No body → no morerecords field

    def test_non_dict_json_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        """A JSON array at top level is rejected (expected object)."""
        transport.add_response(
            httpx.Response(
                status_code=200,
                text="[1, 2, 3]",
                headers={"content-type": "application/json"},
            )
        )
        with pytest.raises(DataverseClientError, match="Expected JSON object"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

    def test_value_not_array_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"value": "not-a-list"}))
        with pytest.raises(DataverseClientError, match="Expected 'value' to be an array"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

    def test_next_link_non_string_rejected(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"value": [], "@odata.nextLink": 12345}))
        with pytest.raises(DataverseClientError, match=r"@odata.nextLink.*string"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")


# ---------------------------------------------------------------------------
# OData pagination
# ---------------------------------------------------------------------------


class TestPaginateOdata:
    """OData nextLink pagination with SSRF validation."""

    def test_follows_next_link(self, client: DataverseClient, transport: MockTransport) -> None:
        """Pagination follows nextLink across multiple pages."""
        next_url = f"{ENV_URL}/api/data/v9.2/accounts?$skiptoken=abc"
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@odata.nextLink": next_url,
                }
            )
        )
        transport.add_response(_make_json_response({"value": [{"id": 2}]}))

        with patch(
            "elspeth.plugins.infrastructure.clients.dataverse.validate_url_for_ssrf",
            return_value=_make_ssrf_safe(next_url),
        ):
            pages = list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))

        assert len(pages) == 2
        assert [dict(r) for r in pages[0].rows] == [{"id": 1}]
        assert [dict(r) for r in pages[1].rows] == [{"id": 2}]

    def test_ssrf_rejection_on_cross_host_next_link(self, client: DataverseClient, transport: MockTransport) -> None:
        """nextLink pointing to a different host is rejected by domain allowlist."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@odata.nextLink": "https://evil.com/steal-data",
                }
            )
        )
        with pytest.raises(DataverseClientError, match="rejected by domain allowlist"):
            # Consume the iterator to trigger the SSRF check
            list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))

    def test_ssrf_ip_pinning_failure(self, client: DataverseClient, transport: MockTransport) -> None:
        """nextLink failing IP-pinning validation is rejected."""
        next_url = f"{ENV_URL}/api/data/v9.2/accounts?$skiptoken=abc"
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@odata.nextLink": next_url,
                }
            )
        )

        with (
            patch(
                "elspeth.plugins.infrastructure.clients.dataverse.validate_url_for_ssrf",
                side_effect=SSRFBlockedError("DNS rebinding detected"),
            ),
            pytest.raises(DataverseClientError, match="IP-pinning SSRF validation"),
        ):
            list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))

    def test_single_page_no_next_link(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"value": [{"id": 1}]}))
        pages = list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))
        assert len(pages) == 1


# ---------------------------------------------------------------------------
# Empty-page guard
# ---------------------------------------------------------------------------


class TestEmptyPageGuard:
    """3 consecutive empty pages triggers termination."""

    def test_three_consecutive_empty_pages_error(self, client: DataverseClient, transport: MockTransport) -> None:
        next1 = f"{ENV_URL}/api/data/v9.2/accounts?p=2"
        next2 = f"{ENV_URL}/api/data/v9.2/accounts?p=3"
        next3 = f"{ENV_URL}/api/data/v9.2/accounts?p=4"
        transport.add_response(_make_json_response({"value": [], "@odata.nextLink": next1}))
        transport.add_response(_make_json_response({"value": [], "@odata.nextLink": next2}))
        transport.add_response(_make_json_response({"value": [], "@odata.nextLink": next3}))

        with (
            patch(
                "elspeth.plugins.infrastructure.clients.dataverse.validate_url_for_ssrf",
                side_effect=lambda url, **kw: _make_ssrf_safe(url),
            ),
            pytest.raises(DataverseClientError, match="3 consecutive empty pages"),
        ):
            list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))

    def test_non_empty_page_resets_counter(self, client: DataverseClient, transport: MockTransport) -> None:
        """A non-empty page resets the consecutive empty counter."""
        next1 = f"{ENV_URL}/api/data/v9.2/accounts?p=2"
        next2 = f"{ENV_URL}/api/data/v9.2/accounts?p=3"
        # Page 1: empty
        transport.add_response(_make_json_response({"value": [], "@odata.nextLink": next1}))
        # Page 2: non-empty (resets counter)
        transport.add_response(_make_json_response({"value": [{"id": 1}], "@odata.nextLink": next2}))
        # Page 3: empty but only 1 consecutive, and no next link -> done
        transport.add_response(_make_json_response({"value": []}))

        with patch(
            "elspeth.plugins.infrastructure.clients.dataverse.validate_url_for_ssrf",
            side_effect=lambda url, **kw: _make_ssrf_safe(url),
        ):
            pages = list(client.paginate_odata(f"{ENV_URL}/api/data/v9.2/accounts"))

        assert len(pages) == 3


# ---------------------------------------------------------------------------
# FetchXML pagination with paging cookie injection
# ---------------------------------------------------------------------------


class TestPaginateFetchxml:
    """FetchXML paging cookie injection via ElementTree."""

    def test_single_page_fetchxml(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@Microsoft.Dynamics.CRM.morerecords": False,
                }
            )
        )
        fetch_xml = '<fetch count="50"><entity name="account"/></fetch>'
        pages = list(client.paginate_fetchxml("accounts", fetch_xml))
        assert len(pages) == 1

    def test_paging_cookie_injection(self, client: DataverseClient, transport: MockTransport) -> None:
        """Paging cookie is injected into FetchXML via ElementTree (not string concat)."""
        cookie_value = urllib.parse.quote("<cookie page='1'/>")
        # Page 1
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@Microsoft.Dynamics.CRM.fetchxmlpagingcookie": cookie_value,
                    "@Microsoft.Dynamics.CRM.morerecords": True,
                }
            )
        )
        # Page 2 (last)
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 2}],
                    "@Microsoft.Dynamics.CRM.morerecords": False,
                }
            )
        )

        fetch_xml = '<fetch count="50"><entity name="account"/></fetch>'
        pages = list(client.paginate_fetchxml("accounts", fetch_xml))
        assert len(pages) == 2

        # Verify second request URL contains fetchXml with paging-cookie attribute
        second_request = transport.requests[1]
        url_parts = urllib.parse.urlparse(str(second_request.url))
        query_params = urllib.parse.parse_qs(url_parts.query)
        fetch_param = query_params["fetchXml"][0]
        assert "paging-cookie" in fetch_param
        assert 'page="2"' in fetch_param

    def test_paging_cookie_with_xml_metacharacters(self, client: DataverseClient, transport: MockTransport) -> None:
        """Cookie containing XML metacharacters (<>&'") is safely injected via ET."""
        # Cookie with XML metacharacters that would break string interpolation
        raw_cookie = '<cookie page="1" attr="a&b<c>d"/>'
        encoded_cookie = urllib.parse.quote(raw_cookie)

        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@Microsoft.Dynamics.CRM.fetchxmlpagingcookie": encoded_cookie,
                    "@Microsoft.Dynamics.CRM.morerecords": True,
                }
            )
        )
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 2}],
                    "@Microsoft.Dynamics.CRM.morerecords": False,
                }
            )
        )

        fetch_xml = '<fetch count="50"><entity name="account"/></fetch>'
        # This must not raise — ET handles escaping
        pages = list(client.paginate_fetchxml("accounts", fetch_xml))
        assert len(pages) == 2

    def test_invalid_fetchxml_root_element(self, client: DataverseClient, transport: MockTransport) -> None:
        """Non-<fetch> root element is rejected."""
        cookie_value = urllib.parse.quote("<cookie page='1'/>")
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@Microsoft.Dynamics.CRM.fetchxmlpagingcookie": cookie_value,
                    "@Microsoft.Dynamics.CRM.morerecords": True,
                }
            )
        )

        bad_xml = '<query count="50"><entity name="account"/></query>'
        with pytest.raises(DataverseClientError, match="root element must be <fetch>"):
            list(client.paginate_fetchxml("accounts", bad_xml))

    def test_malformed_fetchxml_parse_error(self, client: DataverseClient, transport: MockTransport) -> None:
        """Malformed XML raises DataverseClientError."""
        cookie_value = urllib.parse.quote("<cookie/>")
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    "@Microsoft.Dynamics.CRM.fetchxmlpagingcookie": cookie_value,
                    "@Microsoft.Dynamics.CRM.morerecords": True,
                }
            )
        )

        bad_xml = "<fetch><unclosed"
        with pytest.raises(DataverseClientError, match="Failed to parse FetchXML"):
            list(client.paginate_fetchxml("accounts", bad_xml))

    def test_missing_morerecords_crashes_fetchxml_pagination(self, client: DataverseClient, transport: MockTransport) -> None:
        """FetchXML pagination crashes if morerecords field is absent.

        Absence is a protocol anomaly — we must not silently infer pagination
        state. The field is required for FetchXML; only OData omits it.
        """
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": 1}],
                    # No @Microsoft.Dynamics.CRM.morerecords — anomaly
                    "@Microsoft.Dynamics.CRM.fetchxmlpagingcookie": urllib.parse.quote("<cookie/>"),
                }
            )
        )

        fetch_xml = '<fetch count="50"><entity name="account"/></fetch>'
        with pytest.raises(DataverseClientError, match=r"missing.*morerecords"):
            list(client.paginate_fetchxml("accounts", fetch_xml))


# ---------------------------------------------------------------------------
# Retry-After cap enforcement
# ---------------------------------------------------------------------------


class TestRetryAfterCap:
    """Retry-After header cap enforcement."""

    def test_retry_after_at_cap_is_retryable(self, client: DataverseClient, transport: MockTransport) -> None:
        """Retry-After equal to cap is still retryable."""
        transport.add_response(httpx.Response(status_code=429, headers={"retry-after": "60"}, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True

    def test_retry_after_just_above_cap_non_retryable(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(httpx.Response(status_code=429, headers={"retry-after": "60.1"}, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is False

    def test_custom_retry_after_cap(self, transport: MockTransport) -> None:
        """Custom retry_after_cap is respected."""
        c = DataverseClient(
            environment_url=ENV_URL,
            credential=FakeCredential(),  # type: ignore[arg-type]  # test fake
            retry_after_cap=10.0,
        )
        c._client.close()
        c._client = httpx.Client(transport=transport, timeout=30.0)

        transport.add_response(httpx.Response(status_code=429, headers={"retry-after": "15"}, text=""))
        with pytest.raises(DataverseClientError) as exc_info:
            c.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is False


# ---------------------------------------------------------------------------
# Connection / timeout errors
# ---------------------------------------------------------------------------


class TestConnectionErrors:
    """Network-level errors are retryable."""

    def test_timeout_is_retryable(self, client: DataverseClient) -> None:

        class TimeoutTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.TimeoutException("timed out")

        client._client.close()
        client._client = httpx.Client(transport=TimeoutTransport(), timeout=30.0)

        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True
        assert "timed out" in str(exc_info.value)

    def test_timeout_carries_request_url(self, client: DataverseClient) -> None:
        """Regression: errors must carry the actual failing URL for audit trail."""

        class TimeoutTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.TimeoutException("timed out")

        client._client.close()
        client._client = httpx.Client(transport=TimeoutTransport(), timeout=30.0)

        url = f"{ENV_URL}/api/data/v9.2/accounts?$top=100"
        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(url)
        assert exc_info.value.request_url == url
        assert exc_info.value.request_headers is not None

    def test_connect_error_is_retryable(self, client: DataverseClient) -> None:
        class ConnectErrorTransport(httpx.BaseTransport):
            def handle_request(self, request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("connection refused")

        client._client.close()
        client._client = httpx.Client(transport=ConnectErrorTransport(), timeout=30.0)

        with pytest.raises(DataverseClientError) as exc_info:
            client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        assert exc_info.value.retryable is True


# ---------------------------------------------------------------------------
# Upsert (PATCH)
# ---------------------------------------------------------------------------


class TestUpsert:
    """PATCH upsert operation."""

    def test_upsert_204_no_content(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_empty_response(204))
        page = client.upsert(
            f"{ENV_URL}/api/data/v9.2/accounts(key=val)",
            {"name": "test"},
        )
        assert len(page.rows) == 0
        assert page.status_code == 204

        # Verify it was a PATCH request
        assert transport.requests[0].method == "PATCH"

    def test_upsert_with_return_representation(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"id": 42, "name": "test"}))
        page = client.upsert(
            f"{ENV_URL}/api/data/v9.2/accounts(key=val)",
            {"name": "test"},
        )
        assert [dict(r) for r in page.rows] == [{"id": 42, "name": "test"}]


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


class TestAuthHeaders:
    """Credential token injection into requests."""

    def test_bearer_token_in_headers(self, client: DataverseClient, transport: MockTransport) -> None:
        transport.add_response(_make_json_response({"value": []}))
        client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

        request = transport.requests[0]
        assert request.headers["authorization"] == "Bearer fake-token-123"
        assert request.headers["odata-version"] == "4.0"


# ---------------------------------------------------------------------------
# Rate limiter integration
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Rate limiter is called before each request."""

    def test_rate_limiter_called(self, transport: MockTransport) -> None:
        acquire_calls = []

        class FakeLimiter:
            def acquire(self) -> None:
                acquire_calls.append(True)

        c = DataverseClient(
            environment_url=ENV_URL,
            credential=FakeCredential(),  # type: ignore[arg-type]  # test fake
            limiter=FakeLimiter(),  # type: ignore[arg-type]
        )
        c._client.close()
        c._client = httpx.Client(transport=transport, timeout=30.0)

        transport.add_response(_make_json_response({"value": []}))
        c.get_page(f"{ENV_URL}/api/data/v9.2/accounts")

        assert len(acquire_calls) == 1

    def test_no_rate_limiter(self, client: DataverseClient, transport: MockTransport) -> None:
        """No limiter configured — should not raise."""
        transport.add_response(_make_json_response({"value": []}))
        client.get_page(f"{ENV_URL}/api/data/v9.2/accounts")
        # If we got here, no crash from missing limiter


# ---------------------------------------------------------------------------
# DataverseClientError attributes
# ---------------------------------------------------------------------------


class TestDataverseClientError:
    """Error object attribute coverage."""

    def test_error_attributes(self) -> None:
        err = DataverseClientError(
            "test error",
            retryable=True,
            status_code=500,
            latency_ms=42.5,
        )
        assert str(err) == "test error"
        assert err.retryable is True
        assert err.status_code == 500
        assert err.latency_ms == 42.5

    def test_error_defaults(self) -> None:
        err = DataverseClientError("minimal", retryable=False)
        assert err.status_code is None
        assert err.latency_ms is None


# ---------------------------------------------------------------------------
# Bug fix: bearer token fingerprinted before DTO storage (elspeth-7aa72f1ce2)
# ---------------------------------------------------------------------------


class TestBearerTokenFingerprinting:
    """Verify raw bearer tokens are fingerprinted in the client, not the source."""

    def test_request_headers_fingerprinted_in_response(self, transport: MockTransport, client: DataverseClient) -> None:
        """DataversePageResponse.request_headers must NOT contain raw bearer token.

        In production (FINGERPRINT_KEY set): Authorization becomes <fingerprint:{hmac}>.
        In dev mode (ALLOW_RAW_SECRETS=true): Authorization is removed entirely.
        Both are acceptable — raw token is NOT acceptable.
        """
        transport.add_response(_make_json_response({"value": [{"id": "1"}]}))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

        auth_value = page.request_headers.get("Authorization")
        if auth_value is not None:
            # Header present — must be fingerprinted, not raw
            assert auth_value.startswith("<fingerprint:"), (
                f"Authorization header present but not fingerprinted: {auth_value!r}. "
                f"Expected '<fingerprint:...>' format or header removal."
            )
        # If header is absent, dev mode removed it — also acceptable

    def test_request_headers_fingerprinted_on_204(self, transport: MockTransport, client: DataverseClient) -> None:
        """204 No Content path also fingerprints headers."""
        transport.add_response(_make_empty_response(204))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

        auth_value = page.request_headers.get("Authorization")
        if auth_value is not None:
            assert auth_value.startswith("<fingerprint:"), f"Authorization header present but not fingerprinted on 204 path: {auth_value!r}"

    def test_request_url_captured(self, transport: MockTransport, client: DataverseClient) -> None:
        """DataversePageResponse captures the actual request URL."""
        url = f"{ENV_URL}/api/data/v9.2/contacts?$top=10"
        transport.add_response(_make_json_response({"value": [{"id": "1"}]}))
        page = client.get_page(url)

        assert page.request_url == url


# ---------------------------------------------------------------------------
# Bug fix: value array item type validation (elspeth-1abc0d36d2)
# ---------------------------------------------------------------------------


class TestValueArrayTypeValidation:
    """Verify non-dict items in value array are rejected at Tier 3 boundary."""

    def test_string_item_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """String element in value array raises DataverseClientError."""
        transport.add_response(_make_json_response({"value": [{"id": "1"}, "not-a-dict"]}))
        with pytest.raises(DataverseClientError, match=r"value\[1\].*str"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_int_item_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """Integer element in value array raises DataverseClientError."""
        transport.add_response(_make_json_response({"value": [42]}))
        with pytest.raises(DataverseClientError, match=r"value\[0\].*int"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_null_item_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """null element in value array raises DataverseClientError."""
        transport.add_response(_make_json_response({"value": [None]}))
        with pytest.raises(DataverseClientError, match=r"value\[0\].*NoneType"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_nested_list_item_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """Nested list element in value array raises DataverseClientError."""
        transport.add_response(_make_json_response({"value": [[1, 2, 3]]}))
        with pytest.raises(DataverseClientError, match=r"value\[0\].*list"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_valid_dict_items_accepted(self, transport: MockTransport, client: DataverseClient) -> None:
        """All-dict value array passes validation."""
        transport.add_response(_make_json_response({"value": [{"id": "1"}, {"id": "2"}]}))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert len(page.rows) == 2

    def test_empty_value_array_accepted(self, transport: MockTransport, client: DataverseClient) -> None:
        """Empty value array passes validation (no items to check)."""
        transport.add_response(_make_json_response({"value": []}))
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert len(page.rows) == 0


# ---------------------------------------------------------------------------
# Bug fix: request_url tracked per page (elspeth-fafd21248c)
# ---------------------------------------------------------------------------


class TestRequestUrlTracking:
    """Verify paginated responses carry the actual URL fetched, not the initial URL."""

    def test_paginated_pages_carry_correct_urls(self, transport: MockTransport, client: DataverseClient) -> None:
        """Each page in paginated iteration carries its own request_url."""
        initial_url = f"{ENV_URL}/api/data/v9.2/contacts"
        next_url = f"{ENV_URL}/api/data/v9.2/contacts?$skiptoken=abc123"

        # Page 1: has nextLink
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@odata.nextLink": next_url,
                }
            )
        )
        # Page 2: no nextLink (last page)
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "2"}],
                }
            )
        )

        # Patch SSRF validation since nextLink domain check requires DNS
        with patch.object(
            client,
            "_validate_url_ssrf",
            side_effect=lambda url: _make_ssrf_safe(url),
        ):
            pages = list(client.paginate_odata(initial_url))

        assert len(pages) == 2
        assert pages[0].request_url == initial_url
        assert pages[1].request_url == next_url


# ---------------------------------------------------------------------------
# Bug fix: bool(more_records_raw) coercion (elspeth-6fea320491)
# ---------------------------------------------------------------------------


class TestMoreRecordsTier3Parsing:
    """Tier 3 boundary: morerecords must be parsed explicitly, not via bool()."""

    def test_string_false_is_false(self, transport: MockTransport, client: DataverseClient) -> None:
        """String 'false' must parse as False, not bool('false') == True."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": "false",
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is False

    def test_string_true_is_true(self, transport: MockTransport, client: DataverseClient) -> None:
        """String 'true' must parse as True."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": "true",
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is True

    @pytest.mark.parametrize("value,expected", [("False", False), ("TRUE", True), ("fAlSe", False)])
    def test_string_case_insensitive(self, transport: MockTransport, client: DataverseClient, value: str, expected: bool) -> None:
        """Mixed-case strings are accepted case-insensitively."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": value,
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is expected

    def test_bool_true_accepted(self, transport: MockTransport, client: DataverseClient) -> None:
        """JSON boolean true passes through directly."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": True,
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is True

    def test_bool_false_accepted(self, transport: MockTransport, client: DataverseClient) -> None:
        """JSON boolean false passes through directly."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": False,
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is False

    def test_unexpected_string_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """Non-boolean string like 'yes' is rejected."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": "yes",
                }
            )
        )
        with pytest.raises(DataverseClientError, match="unexpected string value"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_integer_rejected(self, transport: MockTransport, client: DataverseClient) -> None:
        """Integer type for morerecords is rejected."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    "@Microsoft.Dynamics.CRM.morerecords": 1,
                }
            )
        )
        with pytest.raises(DataverseClientError, match="unexpected type"):
            client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")

    def test_absent_records_none(self, transport: MockTransport, client: DataverseClient) -> None:
        """When morerecords absent, more_records is None (absence recorded, not inferred)."""
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    # No @Microsoft.Dynamics.CRM.morerecords
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is None

    def test_absent_with_next_link_still_none(self, transport: MockTransport, client: DataverseClient) -> None:
        """Even with nextLink present, absent morerecords is None — no inference."""
        next_url = f"{ENV_URL}/api/data/v9.2/contacts?$skiptoken=abc"
        transport.add_response(
            _make_json_response(
                {
                    "value": [{"id": "1"}],
                    # No @Microsoft.Dynamics.CRM.morerecords
                    "@odata.nextLink": next_url,
                }
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert page.more_records is None  # Absence is absence, nextLink is separate


# ---------------------------------------------------------------------------
# Bug fix: response headers filtered for sensitive data (elspeth-443f57c78d)
# ---------------------------------------------------------------------------


class TestResponseHeaderFiltering:
    """Sensitive response headers must be stripped before DTO storage."""

    def test_set_cookie_stripped(self, transport: MockTransport, client: DataverseClient) -> None:
        """Set-Cookie must not appear in DataversePageResponse.headers."""
        transport.add_response(
            _make_json_response(
                {"value": [{"id": "1"}]},
                headers={"Set-Cookie": "session=abc123; Path=/; HttpOnly"},
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert "set-cookie" not in {k.lower() for k in page.headers}

    def test_www_authenticate_stripped(self, transport: MockTransport, client: DataverseClient) -> None:
        """WWW-Authenticate must not appear in DataversePageResponse.headers."""
        transport.add_response(
            _make_json_response(
                {"value": [{"id": "1"}]},
                headers={"WWW-Authenticate": "Bearer realm=dataverse"},
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert "www-authenticate" not in {k.lower() for k in page.headers}

    def test_safe_headers_preserved(self, transport: MockTransport, client: DataverseClient) -> None:
        """Non-sensitive headers like content-type are preserved."""
        transport.add_response(
            _make_json_response(
                {"value": [{"id": "1"}]},
                headers={
                    "x-ms-request-id": "req-123",
                    "Set-Cookie": "session=secret",
                },
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert any(k.lower() == "x-ms-request-id" for k in page.headers)
        assert "set-cookie" not in {k.lower() for k in page.headers}

    def test_204_response_headers_also_filtered(self, transport: MockTransport, client: DataverseClient) -> None:
        """Empty response (204) path also filters headers."""
        transport.add_response(
            httpx.Response(
                status_code=204,
                text="",
                headers={"Set-Cookie": "session=abc123"},
            )
        )
        page = client.get_page(f"{ENV_URL}/api/data/v9.2/contacts")
        assert "set-cookie" not in {k.lower() for k in page.headers}
