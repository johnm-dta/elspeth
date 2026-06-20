"""Tests for Azure AI Search provider."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest
import respx

from elspeth.core.security.web import SSRFBlockedError, SSRFSafeRequest
from elspeth.plugins.infrastructure.clients.retrieval.azure_search import (
    AzureSearchProvider,
    AzureSearchProviderConfig,
)
from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalError
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk


class TestAzureSearchProviderConfig:
    def test_requires_https(self):
        with pytest.raises(ValueError, match="HTTPS"):
            AzureSearchProviderConfig(
                endpoint="http://test.search.windows.net",
                index="test",
                api_key="key",
            )

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://10.0.0.1",
            "https://169.254.169.254",
        ],
    )
    def test_rejects_blocked_literal_ip_endpoints(self, endpoint: str) -> None:
        with pytest.raises(ValueError, match=r"blocked|metadata|private|loopback|link-local"):
            AzureSearchProviderConfig(
                endpoint=endpoint,
                index="test",
                api_key="key",
            )

    def test_auth_mutual_exclusion(self):
        with pytest.raises(ValueError, match="only one"):
            AzureSearchProviderConfig(
                endpoint="https://test.search.windows.net",
                index="test",
                api_key="key",
                use_managed_identity=True,
            )

    def test_auth_required(self):
        with pytest.raises(ValueError, match="either"):
            AzureSearchProviderConfig(
                endpoint="https://test.search.windows.net",
                index="test",
            )

    def test_managed_identity_rejects_non_azure_search_hostname(self) -> None:
        with pytest.raises(ValueError, match=r"managed identity.*search\.windows\.net"):
            AzureSearchProviderConfig(
                endpoint="https://attacker.example.com",
                index="test",
                use_managed_identity=True,
            )

    def test_api_key_auth_preserves_existing_public_hostname_support(self) -> None:
        config = AzureSearchProviderConfig(
            endpoint="https://search-proxy.example.com",
            index="test",
            api_key="key",
        )

        assert config.endpoint == "https://search-proxy.example.com"

    def test_semantic_requires_config(self):
        with pytest.raises(ValueError, match="semantic_config"):
            AzureSearchProviderConfig(
                endpoint="https://test.search.windows.net",
                index="test",
                api_key="key",
                search_mode="semantic",
            )

    def test_index_name_validation(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            AzureSearchProviderConfig(
                endpoint="https://test.search.windows.net",
                index="bad/path/../traversal",
                api_key="key",
            )

    def test_valid_index_names(self):
        for name in ["my-index", "index_v2", "MyIndex123"]:
            config = AzureSearchProviderConfig(
                endpoint="https://test.search.windows.net",
                index=name,
                api_key="key",
            )
            assert config.index == name


class TestAzureSearchProviderSearch:
    def _make_provider(self, search_mode: str = "hybrid") -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
            search_mode=search_mode,
        )
        execution = MagicMock()
        telemetry_emit = MagicMock()
        return AzureSearchProvider(
            config=config,
            execution=execution,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

    def test_returns_retrieval_chunks(self):
        provider = self._make_provider()
        mock_response = {
            "value": [
                {"@search.score": 5.0, "content": "Result 1", "id": "doc1"},
                {"@search.score": 3.0, "content": "Result 2", "id": "doc2"},
            ]
        }
        with patch.object(provider, "_execute_search", return_value=mock_response):
            chunks = provider.search(
                "test query",
                top_k=5,
                min_score=0.0,
                state_id="state-1",
                token_id="token-1",
            )
        assert len(chunks) == 2
        assert all(isinstance(c, RetrievalChunk) for c in chunks)
        assert chunks[0].score >= chunks[1].score

    def test_min_score_filtering(self):
        # Use keyword mode: range (0, 50). Score 30.0 -> 0.6, score 2.0 -> 0.04.
        provider = self._make_provider("keyword")
        mock_response = {
            "value": [
                {"@search.score": 30.0, "content": "High", "id": "doc1"},
                {"@search.score": 2.0, "content": "Low", "id": "doc2"},
            ]
        }
        with patch.object(provider, "_execute_search", return_value=mock_response):
            chunks = provider.search(
                "test",
                top_k=5,
                min_score=0.5,
                state_id="state-1",
                token_id=None,
            )
        assert len(chunks) == 1
        assert chunks[0].content == "High"

    def test_malformed_json_raises_retrieval_error(self):
        provider = self._make_provider()
        with patch.object(provider, "_execute_search", side_effect=RetrievalError("bad json", retryable=False)):
            with pytest.raises(RetrievalError) as exc_info:
                provider.search("test", top_k=5, min_score=0.0, state_id="s1", token_id=None)
            assert not exc_info.value.retryable

    def test_server_error_raises_retryable(self):
        provider = self._make_provider()
        with patch.object(
            provider,
            "_execute_search",
            side_effect=RetrievalError("server error", retryable=True, status_code=500),
        ):
            with pytest.raises(RetrievalError) as exc_info:
                provider.search("test", top_k=5, min_score=0.0, state_id="s1", token_id=None)
            assert exc_info.value.retryable
            assert exc_info.value.status_code == 500


class TestScoreNormalization:
    def _make_provider(self, search_mode: str = "hybrid") -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
            search_mode=search_mode,
        )
        return AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )

    def test_keyword_mid_range(self):
        provider = self._make_provider("keyword")
        assert provider._normalize_score(25.0) == pytest.approx(0.5)

    def test_keyword_zero(self):
        provider = self._make_provider("keyword")
        assert provider._normalize_score(0.0) == 0.0

    def test_keyword_max(self):
        provider = self._make_provider("keyword")
        assert provider._normalize_score(50.0) == 1.0

    def test_keyword_exceeds_max_clamped(self):
        provider = self._make_provider("keyword")
        assert provider._normalize_score(200.0) == 1.0

    def test_negative_score_clamped_to_zero(self):
        provider = self._make_provider("keyword")
        assert provider._normalize_score(-5.0) == 0.0

    def test_vector_already_normalized(self):
        provider = self._make_provider("vector")
        assert provider._normalize_score(0.75) == pytest.approx(0.75)

    def test_semantic_range(self):
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
            search_mode="semantic",
            semantic_config="my-semantic-config",
        )
        provider = AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )
        assert provider._normalize_score(2.0) == pytest.approx(0.5)

    def test_nan_score_raises_retrieval_error(self):
        provider = self._make_provider("keyword")
        with pytest.raises(RetrievalError, match="non-finite"):
            provider._normalize_score(float("nan"))

    def test_infinity_score_raises_retrieval_error(self):
        provider = self._make_provider("keyword")
        with pytest.raises(RetrievalError, match="non-finite"):
            provider._normalize_score(float("inf"))

    def test_negative_infinity_raises_retrieval_error(self):
        provider = self._make_provider("keyword")
        with pytest.raises(RetrievalError, match="non-finite"):
            provider._normalize_score(float("-inf"))


class TestBuildRequestBody:
    def _make_provider(self, search_mode: str = "hybrid", **overrides: Any) -> AzureSearchProvider:
        config_data = {
            "endpoint": "https://test.search.windows.net",
            "index": "test-index",
            "api_key": "test-key",
            "search_mode": search_mode,
        }
        config_data.update(overrides)
        if search_mode == "semantic":
            config_data.setdefault("semantic_config", "my-semantic-config")
        config = AzureSearchProviderConfig(**config_data)
        return AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )

    def test_keyword_body(self):
        provider = self._make_provider("keyword")
        body = provider._build_request_body("test query", top_k=5)
        assert body["search"] == "test query"
        assert body["top"] == 5
        assert "vectorQueries" not in body

    def test_vector_body(self):
        provider = self._make_provider("vector")
        body = provider._build_request_body("test query", top_k=3)
        assert "search" not in body
        assert body["vectorQueries"][0]["text"] == "test query"
        assert body["vectorQueries"][0]["k"] == 3

    def test_hybrid_body(self):
        provider = self._make_provider("hybrid")
        body = provider._build_request_body("test query", top_k=5)
        assert body["search"] == "test query"
        assert "vectorQueries" in body

    def test_semantic_body(self):
        provider = self._make_provider("semantic")
        body = provider._build_request_body("test query", top_k=5)
        assert body["queryType"] == "semantic"
        assert body["semanticConfiguration"] == "my-semantic-config"


class TestParseResponse:
    def _make_provider(self) -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
        )
        return AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )

    def test_missing_value_key_raises(self):
        provider = self._make_provider()
        with pytest.raises(RetrievalError, match="missing 'value'"):
            provider._parse_response({}, min_score=0.0)

    def test_skips_items_without_score(self):
        provider = self._make_provider()
        response = {"value": [{"content": "text", "id": "doc1"}]}
        chunks, skipped = provider._parse_response(response, min_score=0.0)
        assert chunks == []
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "missing_score"

    def test_skips_items_without_content(self):
        provider = self._make_provider()
        response = {"value": [{"@search.score": 5.0, "id": "doc1"}]}
        chunks, skipped = provider._parse_response(response, min_score=0.0)
        assert chunks == []
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "missing_content"

    def test_skips_items_without_id(self):
        """Items with no id are skipped — no fabricated "unknown" source_id."""
        provider = self._make_provider()
        response = {
            "value": [
                {"@search.score": 5.0, "content": "text", "id": "doc1"},
                {"@search.score": 5.0, "content": "text", "@search.documentId": "doc2"},
                {"@search.score": 5.0, "content": "text"},  # no id at all
            ]
        }
        chunks, skipped = provider._parse_response(response, min_score=0.0)
        assert len(chunks) == 2
        assert {c.source_id for c in chunks} == {"doc1", "doc2"}
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "missing_id"

    @pytest.mark.parametrize(
        "bad_score,desc",
        [
            ("high", "string"),
            (True, "bool_true"),
            (False, "bool_false"),
            ([1.0], "list"),
            ({"v": 1.0}, "dict"),
        ],
    )
    def test_non_numeric_score_skipped_at_tier3_boundary(self, bad_score, desc):
        """Tier 3 boundary: non-numeric @search.score must be skipped, not crash."""
        provider = self._make_provider()
        response = {
            "value": [
                {"@search.score": bad_score, "content": "text", "id": "doc1"},
                {"@search.score": 5.0, "content": "good", "id": "doc2"},
            ]
        }
        chunks, skipped = provider._parse_response(response, min_score=0.0)
        # Bad score item skipped; good item still returned
        assert len(chunks) == 1
        assert chunks[0].source_id == "doc2"
        assert any(s["reason"] == "invalid_score_type" for s in skipped)

    def test_results_sorted_by_descending_score(self):
        provider = self._make_provider()
        response = {
            "value": [
                {"@search.score": 1.0, "content": "low", "id": "d1"},
                {"@search.score": 40.0, "content": "high", "id": "d2"},
                {"@search.score": 10.0, "content": "mid", "id": "d3"},
            ]
        }
        chunks, _ = provider._parse_response(response, min_score=0.0)
        assert chunks[0].score >= chunks[1].score >= chunks[2].score

    @pytest.mark.parametrize(
        "bad_value,desc",
        [
            ({"id": "doc1"}, "dict"),
            ("a string", "str"),
            (42, "int"),
        ],
    )
    def test_non_list_value_raises(self, bad_value, desc):
        """Tier 3: top-level 'value' present but not an array → RetrievalError, not AttributeError.

        Regression for elspeth-1b138841d3: a non-array 'value' (object/string)
        previously iterated into `item.get(...)` and raised raw AttributeError,
        bypassing the transform's RetrievalError handling.
        """
        provider = self._make_provider()
        with pytest.raises(RetrievalError, match="must be an array"):
            provider._parse_response({"value": bad_value}, min_score=0.0)

    def test_non_dict_array_members_skipped_at_tier3_boundary(self):
        """Tier 3: non-object array members are recorded skips, not AttributeError crashes.

        Regression for elspeth-1b138841d3. Skip-over-raise is deliberate: it
        matches the file's per-item skip pattern and survives the mixed case
        (one garbage member must not nuke the whole valid page).
        """
        provider = self._make_provider()
        response = {
            "value": [
                "not an object",
                {"@search.score": 5.0, "content": "good", "id": "doc1"},
                123,
            ]
        }
        chunks, skipped = provider._parse_response(response, min_score=0.0)
        assert len(chunks) == 1
        assert chunks[0].source_id == "doc1"
        invalid_item_skips = [s for s in skipped if s["reason"] == "invalid_item_type"]
        assert len(invalid_item_skips) == 2
        assert {s["type"] for s in invalid_item_skips} == {"str", "int"}


class TestAzureSearchProviderReadiness:
    """Tests for AzureSearchProvider.check_readiness()."""

    @pytest.fixture(autouse=True)
    def _bypass_ssrf_dns(self):
        """Readiness tests mock httpx; bypass live DNS pinning at that boundary."""
        safe_request = SSRFSafeRequest(
            original_url="https://test.search.windows.net/indexes/test-index/docs/$count?api-version=2024-07-01",
            resolved_ip="93.184.216.34",
            host_header="test.search.windows.net",
            port=443,
            path="/indexes/test-index/docs/$count?api-version=2024-07-01",
            scheme="https",
            bare_hostname="test.search.windows.net",
        )
        with patch("elspeth.core.security.web.validate_url_for_ssrf", return_value=safe_request):
            yield

    def _make_provider(self) -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
        )
        return AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )

    def _mock_response(self, *, status_code: int = 200, text: str = "0") -> MagicMock:
        resp = MagicMock()
        type(resp).status_code = PropertyMock(return_value=status_code)
        type(resp).text = PropertyMock(return_value=text)
        resp.raise_for_status = MagicMock()
        if status_code >= 400:
            import httpx

            resp.raise_for_status.side_effect = httpx.HTTPStatusError(
                f"HTTP {status_code}",
                request=MagicMock(),
                response=resp,
            )
        return resp

    def test_index_with_documents_is_ready(self) -> None:
        """Index exists and has documents."""
        from elspeth.contracts.probes import CollectionReadinessResult

        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(text="42")):
            result = provider.check_readiness()

        assert isinstance(result, CollectionReadinessResult)
        assert result.reachable is True
        assert result.count == 42
        assert "42 documents" in result.message

    def test_empty_index_is_not_ready(self) -> None:
        """Index exists but is empty."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(text="0")):
            result = provider.check_readiness()

        assert result.reachable is True
        assert result.count == 0
        assert "empty" in result.message

    def test_index_not_found_404(self) -> None:
        """Index does not exist — 404 response."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(status_code=404)):
            result = provider.check_readiness()

        assert result.reachable is True
        assert result.count is None
        assert "not found" in result.message.lower()

    def test_connection_error(self) -> None:
        """Azure Search is unreachable."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", side_effect=ConnectionError("Connection refused")):
            result = provider.check_readiness()

        assert result.reachable is False
        assert result.count is None
        assert "Connection refused" in result.message

    def test_auth_header_sent(self) -> None:
        """API key is included in the readiness probe request."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(text="10")) as mock_get:
            provider.check_readiness()

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["headers"]["api-key"] == "test-key"

    def test_count_url_format(self) -> None:
        """Readiness probe uses the correct $count endpoint."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(text="5")) as mock_get:
            provider.check_readiness()

        url = mock_get.call_args.args[0].original_url
        assert "/indexes/test-index/docs/$count" in url
        assert "api-version=2024-07-01" in url

    def test_non_integer_count_body(self) -> None:
        """Malformed $count response distinguishable from network failure."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(text="not-a-number")):
            result = provider.check_readiness()

        # Reachable (HTTP 200) but unparseable — distinct from unreachable
        assert result.reachable is True
        assert result.count is None
        assert "non-integer" in result.message.lower()

    def test_server_error_during_probe(self) -> None:
        """HTTP 500 during readiness probe reports unreachable."""
        provider = self._make_provider()

        with patch.object(provider, "_readiness_get", return_value=self._mock_response(status_code=500)):
            result = provider.check_readiness()

        assert result.reachable is False
        assert result.count is None

    def test_managed_identity_sends_bearer_token(self) -> None:
        """Managed identity readiness probe acquires a Bearer token via DefaultAzureCredential."""
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            use_managed_identity=True,
        )
        provider = AzureSearchProvider(
            config=config,
            execution=MagicMock(),
            run_id="run-1",
            telemetry_emit=MagicMock(),
        )

        mock_token = MagicMock()
        mock_token.token = "managed-identity-token-123"
        mock_credential = MagicMock()
        mock_credential.get_token.return_value = mock_token

        with (
            patch.object(provider, "_readiness_get", return_value=self._mock_response(text="10")) as mock_get,
            patch(
                "azure.identity.DefaultAzureCredential",
                return_value=mock_credential,
            ),
        ):
            result = provider.check_readiness()

        assert result.reachable is True
        assert result.count == 10
        # Bearer token must be in headers, NOT api-key
        call_headers = mock_get.call_args.kwargs["headers"]
        assert "Authorization" in call_headers
        assert call_headers["Authorization"] == "Bearer managed-identity-token-123"
        assert "api-key" not in call_headers

    def test_readiness_uses_pinned_connection_url_with_host_and_sni(self) -> None:
        provider = self._make_provider()
        safe_request = SSRFSafeRequest(
            original_url="https://test.search.windows.net/indexes/test-index/docs/$count?api-version=2024-07-01",
            resolved_ip="93.184.216.34",
            host_header="test.search.windows.net",
            port=443,
            path="/indexes/test-index/docs/$count?api-version=2024-07-01",
            scheme="https",
            bare_hostname="test.search.windows.net",
        )
        mock_response = httpx.Response(
            200,
            text="12",
            request=httpx.Request("GET", safe_request.connection_url),
        )

        with (
            patch("elspeth.core.security.web.validate_url_for_ssrf", return_value=safe_request),
            patch("httpx.get", side_effect=AssertionError("readiness must not request the hostname URL directly")),
            patch("httpx.Client") as mock_client_class,
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = provider.check_readiness()

        assert result.reachable is True
        assert result.count == 12
        call = mock_client.get.call_args
        assert call.args[0] == safe_request.connection_url
        assert call.kwargs["headers"]["Host"] == "test.search.windows.net"
        assert call.kwargs["headers"]["api-key"] == "test-key"
        assert call.kwargs["extensions"]["sni_hostname"] == "test.search.windows.net"

    def test_readiness_blocks_dns_to_private_before_request(self) -> None:
        provider = self._make_provider()

        with (
            patch("elspeth.core.security.web.validate_url_for_ssrf", side_effect=SSRFBlockedError("Blocked IP range")),
            patch.object(provider, "_readiness_get") as mock_get,
        ):
            result = provider.check_readiness()

        assert result.reachable is False
        assert result.count is None
        assert "blocked by SSRF validation" in result.message
        mock_get.assert_not_called()

    def test_uncaught_exception_crashes_through(self) -> None:
        """Programming errors (e.g. TypeError) must NOT be caught by check_readiness."""
        provider = self._make_provider()

        with (
            patch.object(provider, "_readiness_get", side_effect=TypeError("unexpected type")),
            pytest.raises(TypeError, match="unexpected type"),
        ):
            provider.check_readiness()


class TestExecuteSearchHTTP:
    """HTTP-level tests for _execute_search using respx to mock httpx transport.

    These tests exercise the real HTTP call path through AuditedHTTPClient,
    unlike the other test classes which mock _execute_search directly.
    """

    SEARCH_URL = "https://test.search.windows.net/indexes/test-index/docs/search?api-version=2024-07-01"
    PINNED_SEARCH_URL = "https://93.184.216.34:443/indexes/test-index/docs/search?api-version=2024-07-01"

    @pytest.fixture(autouse=True)
    def _pin_search_dns(self):
        with patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
            yield

    def _make_provider(self) -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
        )
        execution = MagicMock()
        telemetry_emit = MagicMock()
        return AzureSearchProvider(
            config=config,
            execution=execution,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

    def _make_managed_identity_provider(self) -> AzureSearchProvider:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            use_managed_identity=True,
        )
        execution = MagicMock()
        telemetry_emit = MagicMock()
        return AzureSearchProvider(
            config=config,
            execution=execution,
            run_id="run-1",
            telemetry_emit=telemetry_emit,
        )

    def test_constructor_wires_audited_http_client_context(self) -> None:
        config = AzureSearchProviderConfig(
            endpoint="https://test.search.windows.net",
            index="test-index",
            api_key="test-key",
            request_timeout=12.5,
        )
        execution = MagicMock()
        telemetry_emit = MagicMock()
        limiter = MagicMock()

        with patch("elspeth.plugins.infrastructure.clients.http.AuditedHTTPClient") as client_cls:
            provider = AzureSearchProvider(
                config=config,
                execution=execution,
                run_id="run-1",
                telemetry_emit=telemetry_emit,
                limiter=limiter,
            )

        client_cls.assert_called_once_with(
            execution=execution,
            state_id="__init__",
            run_id="run-1",
            telemetry_emit=telemetry_emit,
            timeout=12.5,
            limiter=limiter,
            headers={"Content-Type": "application/json", "api-key": "test-key"},
        )
        assert provider._http_client is client_cls.return_value

    @pytest.mark.parametrize("status_code", [401, 403])
    def test_auth_error_response_raises_non_retryable(self, status_code: int) -> None:
        """HTTP 401/403 both map to RetrievalError(retryable=False)."""
        provider = self._make_provider()

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).respond(status_code=status_code, json={"error": "Auth failed"})

            with pytest.raises(RetrievalError) as exc_info:
                provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

            assert not exc_info.value.retryable
            assert exc_info.value.status_code == status_code

    def test_429_response_raises_retryable(self) -> None:
        """HTTP 429 maps to RetrievalError(retryable=True)."""

        provider = self._make_provider()

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).respond(status_code=429, json={"error": "Rate limited"})

            with pytest.raises(RetrievalError) as exc_info:
                provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

            assert exc_info.value.retryable
            assert exc_info.value.status_code == 429

    def test_500_response_raises_retryable(self) -> None:
        """HTTP 500 maps to RetrievalError(retryable=True)."""

        provider = self._make_provider()

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).respond(status_code=500, json={"error": "Internal Server Error"})

            with pytest.raises(RetrievalError) as exc_info:
                provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

            assert exc_info.value.retryable
            assert exc_info.value.status_code == 500

    def test_200_valid_json_returns_parsed_dict(self) -> None:
        """HTTP 200 with valid JSON returns the parsed response dict."""

        provider = self._make_provider()
        response_body = {
            "value": [
                {"@search.score": 5.0, "content": "Result 1", "id": "doc1"},
            ]
        }

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).respond(status_code=200, json=response_body)

            result = provider._execute_search("test query", top_k=5, state_id="s1", token_id="t1")

        assert result == response_body

    def test_execute_search_managed_identity_sends_bearer_token(self) -> None:
        provider = self._make_managed_identity_provider()
        response_body = {
            "value": [
                {"@search.score": 5.0, "content": "Result 1", "id": "doc1"},
            ]
        }
        mock_token = MagicMock()
        mock_token.token = "managed-identity-token-123"
        mock_credential = MagicMock()
        mock_credential.get_token.return_value = mock_token

        with (
            patch("azure.identity.DefaultAzureCredential", return_value=mock_credential),
            respx.mock,
        ):
            route = respx.post(self.PINNED_SEARCH_URL).mock(return_value=httpx.Response(200, json=response_body))

            result = provider._execute_search("test query", top_k=5, state_id="s1", token_id="t1")

        assert result == response_body
        mock_credential.get_token.assert_called_once_with("https://search.azure.com/.default")
        assert route.calls.last is not None
        request_headers = route.calls.last.request.headers
        assert request_headers["Authorization"] == "Bearer managed-identity-token-123"
        assert "api-key" not in request_headers

    def test_execute_search_updates_audit_context_with_token(self) -> None:
        provider = self._make_provider()
        response_body = {
            "value": [
                {"@search.score": 5.0, "content": "Result 1", "id": "doc1"},
            ]
        }

        with (
            patch.object(provider._http_client, "update_call_context", wraps=provider._http_client.update_call_context) as update_context,
            respx.mock,
        ):
            respx.post(self.PINNED_SEARCH_URL).respond(status_code=200, json=response_body)

            result = provider._execute_search("test query", top_k=5, state_id="state-99", token_id="token-99")

        assert result == response_body
        update_context.assert_called_once_with("state-99", "token-99")

    def test_execute_search_uses_ssrf_pinned_post_connection(self) -> None:
        provider = self._make_provider()
        response_body = {
            "value": [
                {"@search.score": 5.0, "content": "Result 1", "id": "doc1"},
            ]
        }

        with (
            patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]),
            respx.mock,
        ):
            ip_route = respx.post(self.PINNED_SEARCH_URL).mock(return_value=httpx.Response(200, json=response_body))
            hostname_route = respx.post(self.SEARCH_URL).mock(return_value=httpx.Response(200, json={"value": []}))

            result = provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

        assert result == response_body
        assert ip_route.called, "Search POST must connect to the validated pinned IP"
        assert not hostname_route.called, "Search POST must not re-resolve/request the hostname URL"

    def test_execute_search_blocks_dns_to_private_before_request(self) -> None:
        provider = self._make_provider()

        with (
            patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.0.0.1", 0))]),
            patch.object(provider._http_client, "request_ssrf_safe") as mock_request,
            pytest.raises(RetrievalError, match="blocked by SSRF validation") as exc_info,
        ):
            provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

        assert not exc_info.value.retryable
        mock_request.assert_not_called()

    @pytest.mark.parametrize(
        "transport_error",
        [
            httpx.TimeoutException("timed out"),
            httpx.ConnectError("connection refused"),
        ],
    )
    def test_transport_errors_raise_retryable_retrieval_error(self, transport_error: httpx.HTTPError) -> None:
        """Connection and timeout failures from the HTTP transport are retryable."""
        provider = self._make_provider()

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).mock(side_effect=transport_error)

            with pytest.raises(RetrievalError, match="Search request failed") as exc_info:
                provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

        assert exc_info.value.retryable
        assert exc_info.value.__cause__ is transport_error

    def test_200_malformed_json_raises_non_retryable(self) -> None:
        """HTTP 200 with unparseable body maps to RetrievalError(retryable=False)."""

        provider = self._make_provider()

        with respx.mock:
            respx.post(self.PINNED_SEARCH_URL).respond(
                status_code=200,
                content=b"this is not json",
                headers={"Content-Type": "text/plain"},
            )

            with pytest.raises(RetrievalError, match="Malformed JSON") as exc_info:
                provider._execute_search("test query", top_k=5, state_id="s1", token_id=None)

            assert not exc_info.value.retryable
