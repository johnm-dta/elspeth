"""Unit tests for the azure_document_intelligence transform."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
import respx

from elspeth.contracts import Determinism
from elspeth.contracts.call_data import HTTPCallResponse
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.core.landscape.execution_repository import ExecutionRepository
from elspeth.plugins.infrastructure.clients.http import HTTPResponseBodyTooLargeError
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.transforms.azure.document_intelligence import (
    AzureDocumentIntelligence,
    AzureDocumentIntelligenceConfig,
)

_ENDPOINT = "https://test.cognitiveservices.azure.com"
OP = f"{_ENDPOINT}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/abc?api-version=2024-11-30"

BASE = {
    "endpoint": _ENDPOINT,
    "api_key": "k",
    "model_id": "prebuilt-layout",
    "source_mode": "url",
    "source_field": "doc_url",
    "content_field": "di_content",
    "schema": {"mode": "observed"},
}


# ── Config tests ───────────────────────────────────────────────────────────


def _cfg(**overrides: Any) -> AzureDocumentIntelligenceConfig:
    data = {**BASE, **overrides}
    return AzureDocumentIntelligenceConfig.from_dict(data, plugin_name="azure_document_intelligence")


def test_valid_minimal_config() -> None:
    cfg = _cfg()
    assert cfg.api_version == "2024-11-30"
    assert cfg.output_content_format == "text"
    assert cfg.max_capacity_retry_seconds == 300
    assert cfg.configured_output_fields() == {}
    assert cfg.all_output_field_names() == ["di_content"]


def test_rejects_http_endpoint() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(endpoint="http://di.cognitiveservices.azure.com")


def test_rejects_empty_api_key() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(api_key="   ")


def test_rejects_unknown_api_version() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(api_version="2023-07-31")


def test_rejects_bad_model_id() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(model_id="bad/model id")


def test_requires_at_least_one_output() -> None:
    data = {key: value for key, value in BASE.items() if key != "content_field"}
    with pytest.raises(PluginConfigError):
        AzureDocumentIntelligenceConfig.from_dict(data, plugin_name="azure_document_intelligence")


def test_rejects_duplicate_output_field_names() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(content_field="dup", extract={"tables": "dup"})


def test_query_fields_requires_feature() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(query_fields=["Total"])


def test_query_fields_feature_requires_list() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(features=["queryFields"])


def test_unknown_feature_rejected() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(features=["nope"])


def test_rejects_bad_pages_pattern() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(pages="1;2")


def test_accepts_good_pages_pattern() -> None:
    assert _cfg(pages="1-3,5,7-9").pages == "1-3,5,7-9"


def test_rejects_poll_max_below_interval() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(poll_interval_seconds=5.0, poll_max_interval_seconds=1.0)


def test_rejects_bad_string_index_type() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(string_index_type="bogus")


def test_rejects_bad_source_mode() -> None:
    with pytest.raises(PluginConfigError):
        _cfg(source_mode="ftp")


def test_base64_mode_and_extract_fields() -> None:
    cfg = _cfg(
        source_mode="base64",
        source_field="doc_b64",
        extract={"tables": "di_tables", "key_value_pairs": "di_kv"},
    )
    assert cfg.configured_output_fields() == {"tables": "di_tables", "keyValuePairs": "di_kv"}
    assert set(cfg.all_output_field_names()) == {"di_content", "di_tables", "di_kv"}


# ── Transform metadata ─────────────────────────────────────────────────────


def _transform(**overrides: Any) -> AzureDocumentIntelligence:
    return AzureDocumentIntelligence({**BASE, **overrides})


def test_transform_metadata_and_declared_fields() -> None:
    t = _transform(extract={"tables": "di_tables"}, page_count_field="di_pages")
    assert t.name == "azure_document_intelligence"
    assert t.determinism is Determinism.EXTERNAL_CALL
    assert t.passes_through_input is True
    assert t.declared_output_fields == frozenset({"di_content", "di_tables", "di_pages"})


def test_probe_config_instantiates() -> None:
    AzureDocumentIntelligence(AzureDocumentIntelligence.probe_config())


def test_process_raises_use_accept() -> None:
    t = _transform()
    with pytest.raises(NotImplementedError):
        t.process(Mock(), Mock())


# ── Credential + URL construction (R12) ────────────────────────────────────


def test_default_request_headers_carry_api_key() -> None:
    assert _transform()._default_request_headers() == {"Ocp-Apim-Subscription-Key": "k"}


def test_analyze_url_includes_overload_and_params() -> None:
    t = _transform(
        output_content_format="markdown",
        pages="1-2",
        locale="en-US",
        features=["ocrHighResolution", "queryFields"],
        query_fields=["Total", "VendorName"],
    )
    url = t._analyze_url()
    assert url.startswith(f"{_ENDPOINT}/documentintelligence/documentModels/prebuilt-layout:analyze?")
    assert "_overload=analyzeDocument" in url
    assert "api-version=2024-11-30" in url
    assert "outputContentFormat=markdown" in url
    assert "pages=1-2" in url or "pages=1-2".replace("-", "-") in url
    assert "locale=en-US" in url
    assert "stringIndexType=textElements" in url
    assert "features=ocrHighResolution%2CqueryFields" in url
    assert "queryFields=Total%2CVendorName" in url


def test_get_http_client_builds_real_client_with_header_and_cap() -> None:
    t = _transform()
    t._recorder = Mock()
    t._run_id = "run-1"
    client = t._get_http_client("state-1")
    try:
        assert client._default_headers["Ocp-Apim-Subscription-Key"] == "k"
        assert client._max_response_body_bytes == 50_000_000
    finally:
        client.close()


# ── LRO fake-client harness ────────────────────────────────────────────────


class _Resp:
    def __init__(self, status_code: int, *, headers: dict[str, str] | None = None, body: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)
        self.content = self.text.encode()

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://x")
            raise httpx.HTTPStatusError("err", request=request, response=httpx.Response(self.status_code, request=request))


class _FakeClient:
    """Scripted client: pops post_resps then get_resps; tracks get calls."""

    def __init__(self, post_resps: list[Any] | Any, get_resps: list[Any]) -> None:
        self._post_resps = post_resps if isinstance(post_resps, list) else [post_resps]
        self._get_resps = list(get_resps)
        self.get_calls = 0
        self.closed = False

    def post(self, url: str, *, json: Any = None, timeout: float | None = None) -> Any:
        del url, json, timeout
        return self._post_resps.pop(0)

    def get(self, url: str, *, timeout: float | None = None) -> Any:
        del url, timeout
        self.get_calls += 1
        return self._get_resps.pop(0)

    def close(self) -> None:
        self.closed = True


def _row(url: str = "https://x/y.pdf", field: str = "doc_url") -> PipelineRow:
    contract = SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name=field,
                original_name=field,
                python_type=str,
                required=False,
                source="inferred",
                nullable=False,
            ),
        ),
        locked=True,
    )
    return PipelineRow({field: url}, contract)


def _t_for_lro(**overrides: Any) -> AzureDocumentIntelligence:
    return AzureDocumentIntelligence({**BASE, **overrides})


def _run_with_fake(t: AzureDocumentIntelligence, fake: _FakeClient, row: PipelineRow | None = None) -> Any:
    row = row if row is not None else _row()
    t._poll_interval_seconds = 0.0
    t._poll_max_interval_seconds = 0.0
    with t._http_clients_lock:
        t._http_clients["s1"] = fake
    try:
        return t._process_single_with_state(row, "s1", token_id=None)
    finally:
        with t._http_clients_lock:
            t._http_clients.pop("s1", None)


def _post_202() -> _Resp:
    return _Resp(202, headers={"operation-location": OP})


# ── LRO happy path + enrichment ────────────────────────────────────────────


def test_lro_happy_path_enriches() -> None:
    t = _t_for_lro(extract={"tables": "di_tables"}, page_count_field="di_pages")
    done = _Resp(
        200,
        body={
            "status": "succeeded",
            "analyzeResult": {"modelId": "prebuilt-layout", "content": "# Doc", "tables": [{"rowCount": 2}], "pages": [{}, {}]},
        },
    )
    result = _run_with_fake(t, _FakeClient(_post_202(), [_Resp(200, body={"status": "running"}), done]))
    assert result.status == "success"
    out = result.row.to_dict()
    assert out["di_content"] == "# Doc"
    assert out["di_tables"] == [{"rowCount": 2}]
    assert out["di_pages"] == 2
    assert out["doc_url"] == "https://x/y.pdf"


def test_success_reason_metadata() -> None:
    t = _t_for_lro()
    done = _Resp(200, body={"status": "succeeded", "analyzeResult": {"modelId": "prebuilt-layout", "content": "x", "pages": [{}]}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [done]))
    meta = result.success_reason["metadata"]
    assert meta["model_id"] == "prebuilt-layout"
    assert meta["api_version"] == "2024-11-30"
    assert meta["operation_id"] == "abc"
    assert meta["page_count"] == 1
    assert meta["result_status"] == "succeeded"


def test_facet_absent_emits_empty_container() -> None:
    t = _t_for_lro(extract={"tables": "di_tables"})
    done = _Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "x"}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [done]))
    assert result.status == "success"
    assert result.row.to_dict()["di_tables"] == []


# ── Error paths (R10) ──────────────────────────────────────────────────────


def test_missing_source_field() -> None:
    t = _t_for_lro()
    empty = PipelineRow({}, _row().contract)
    result = _run_with_fake(t, _FakeClient([], []), row=empty)
    assert result.reason["reason"] == "missing_field"


def test_non_string_source_field() -> None:
    t = _t_for_lro()
    contract = SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name="doc_url", original_name="doc_url", python_type=object, required=False, source="inferred", nullable=False
            ),
        ),
        locked=True,
    )
    bad = PipelineRow({"doc_url": 123}, contract)
    result = _run_with_fake(t, _FakeClient([], []), row=bad)
    assert result.reason["reason"] == "non_string_field"
    assert result.reason["actual_type"] == "int"


def test_invalid_url() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient([], []), row=_row(url="not-a-url"))
    assert result.reason["reason"] == "invalid_input"
    assert result.reason["error_type"] == "invalid_document_url"


def test_base64_too_large() -> None:
    t = _t_for_lro(source_mode="base64", source_field="doc_b64", max_base64_chars=4)
    big = _row(url="AAAAAAAA", field="doc_b64")
    result = _run_with_fake(t, _FakeClient([], []), row=big)
    assert result.reason["reason"] == "invalid_input"
    assert result.reason["error_type"] == "base64_too_large"


def test_submit_non_202() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_Resp(400), []))
    assert result.reason["reason"] == "api_error"
    assert result.reason["error_type"] == "submit_rejected"
    assert result.reason["status_code"] == 400


def test_missing_operation_location() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_Resp(202, headers={}), []))
    assert result.reason["reason"] == "operation_location_missing"


def test_operation_location_host_mismatch_suppresses_get() -> None:
    """Security (R13): a mismatched Operation-Location host must NOT be polled."""
    t = _t_for_lro()
    bad = "https://evil.example.com/x/analyzeResults/abc"
    fake = _FakeClient(_Resp(202, headers={"operation-location": bad}), [])
    result = _run_with_fake(t, fake)
    assert result.reason["reason"] == "operation_location_untrusted"
    assert fake.get_calls == 0  # api-key never sent to the attacker host


def test_poll_request_failed_non_capacity() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_post_202(), [_Resp(404)]))
    assert result.reason["reason"] == "api_error"
    assert result.reason["error_type"] == "poll_request_failed"
    assert result.reason["status_code"] == 404


def test_analysis_failed_no_raw_message() -> None:
    t = _t_for_lro()
    failed = _Resp(200, body={"status": "failed", "error": {"code": "InvalidContent", "message": "secret detail leak"}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [failed]))
    assert result.reason["reason"] == "analysis_failed"
    assert result.reason.get("cause") == "InvalidContent"
    assert "secret detail leak" not in json.dumps(result.reason)


def test_poll_timeout() -> None:
    t = _t_for_lro()
    t._poll_timeout_seconds = -1.0  # force immediate timeout on first running poll
    result = _run_with_fake(t, _FakeClient(_post_202(), [_Resp(200, body={"status": "running"})]))
    assert result.reason["reason"] == "poll_timeout"


def test_malformed_unknown_status() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_post_202(), [_Resp(200, body={"status": "weird"})]))
    assert result.reason["reason"] == "malformed_response"
    assert result.reason["error_type"] == "unknown_status"


def test_malformed_missing_analyze_result() -> None:
    t = _t_for_lro()
    result = _run_with_fake(t, _FakeClient(_post_202(), [_Resp(200, body={"status": "succeeded"})]))
    assert result.reason["reason"] == "malformed_response"
    assert result.reason["error_type"] == "missing_analyze_result"


def test_malformed_invalid_json() -> None:
    t = _t_for_lro()

    class _BadJson(_Resp):
        def __init__(self) -> None:
            super().__init__(200, body={})
            self.text = "{not json"

    result = _run_with_fake(t, _FakeClient(_post_202(), [_BadJson()]))
    assert result.reason["reason"] == "malformed_response"
    assert result.reason["error_type"] == "invalid_json"


def test_facet_wrong_type_is_malformed() -> None:
    t = _t_for_lro(extract={"tables": "di_tables"})
    done = _Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "x", "tables": {"not": "list"}}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [done]))
    assert result.reason["reason"] == "malformed_response"


def test_malformed_pages_with_content_only_config_fails_closed() -> None:
    """Regression: a malformed `pages` (present-but-not-a-list) must fail the row closed even
    when neither page_count_field nor a pages facet is configured. Previously the unguarded
    success-metadata count_pages() raised past the executor and crashed the whole batch."""
    t = _t_for_lro()  # default BASE config: content_field only, no page_count_field, no pages facet
    done = _Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "x", "pages": "not-a-list"}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [done]))
    assert result.status == "error"
    assert result.reason["reason"] == "malformed_response"


def test_success_reason_includes_warning_count() -> None:
    t = _t_for_lro()
    done = _Resp(
        200,
        body={"status": "succeeded", "analyzeResult": {"content": "x", "warnings": [{"code": "w1"}, {"code": "w2"}]}},
    )
    result = _run_with_fake(t, _FakeClient(_post_202(), [done]))
    assert result.status == "success"
    assert result.success_reason["metadata"]["warning_count"] == 2


def test_analysis_failed_error_code_is_length_capped() -> None:
    t = _t_for_lro()
    long_code = "X" * 500
    failed = _Resp(200, body={"status": "failed", "error": {"code": long_code, "message": "leak"}})
    result = _run_with_fake(t, _FakeClient(_post_202(), [failed]))
    assert result.reason["reason"] == "analysis_failed"
    assert len(result.reason["cause"]) == 128


def test_declared_input_fields_includes_source_field() -> None:
    t = _transform(source_field="my_doc_ref")
    assert "my_doc_ref" in t.declared_input_fields


# ── Capacity retry (R14) ───────────────────────────────────────────────────


def test_capacity_retry_then_success() -> None:
    t = _t_for_lro()
    # submit returns 429 then 202; poll succeeds. Capacity retry recovers.
    fake = _FakeClient([_Resp(429), _post_202()], [_Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "ok"}})])
    result = _run_with_fake(t, fake)
    assert result.status == "success"
    assert result.row.to_dict()["di_content"] == "ok"


def test_retry_timeout_on_exhausted_budget() -> None:
    t = _t_for_lro()
    t._max_capacity_retry_seconds = -1  # deadline already passed → first 429 yields retry_timeout
    result = _run_with_fake(t, _FakeClient(_Resp(429), []))
    assert result.reason["reason"] == "retry_timeout"
    assert result.reason["status_code"] == 429


# ── Shutdown + body-too-large (R1) ─────────────────────────────────────────


def test_shutdown_requested() -> None:
    t = _t_for_lro()
    t._shutdown.set()
    result = _run_with_fake(t, _FakeClient(_post_202(), []))
    assert result.reason["reason"] == "shutdown_requested"


def test_body_too_large_is_per_row_not_batch_cancel() -> None:
    t = _t_for_lro()
    payload = HTTPCallResponse(status_code=200, headers={}, body_size=999, body={"_truncated": True})

    class _BodyTooLargeClient(_FakeClient):
        def get(self, url: str, *, timeout: float | None = None) -> Any:
            del url, timeout
            self.get_calls += 1
            raise HTTPResponseBodyTooLargeError(url="https://x", body_size=999, max_body_bytes=10, response_payload=payload)

    result = _run_with_fake(t, _BodyTooLargeClient(_post_202(), []))
    assert result.reason["reason"] == "body_too_large"
    assert result.reason["body_size"] == 999
    assert result.reason["max_body_bytes"] == 10


# ── Worker dispatch + lifecycle (R11) ──────────────────────────────────────


def test_process_row_cleans_up_client() -> None:
    t = _t_for_lro()
    fake = _FakeClient(_post_202(), [_Resp(200, body={"status": "succeeded", "analyzeResult": {"content": "x"}})])
    with t._http_clients_lock:
        t._http_clients["state-9"] = fake
    ctx = Mock()
    ctx.state_id = "state-9"
    ctx.token = None
    result = t._process_row(_row(), ctx)
    assert result.status == "success"
    assert fake.closed  # finally-block popped + closed the client
    assert "state-9" not in t._http_clients


def test_close_is_idempotent_without_batch_init() -> None:
    t = _t_for_lro()
    t.close()
    t.close()  # second close must not raise
    assert t._shutdown.is_set()


# ── Real AuditedHTTPClient streaming path via respx (R12) ──────────────────


@respx.mock
def test_real_client_streaming_lro_sends_apikey_and_overload() -> None:
    """Drives the REAL AuditedHTTPClient (streaming, because max_response_body_bytes is set)
    through respx: verifies the 202 + Operation-Location header survive streaming reconstruction,
    the poll GET completes, and the real POST carried the api-key header and _overload param."""
    t = _t_for_lro(extract={"tables": "di_tables"})
    recorder = Mock(spec=ExecutionRepository)
    recorder.record_call = Mock()
    t._recorder = recorder
    t._run_id = "run-1"
    t._telemetry_emit = Mock()
    t._poll_interval_seconds = 0.0
    t._poll_max_interval_seconds = 0.0

    respx.post(url__regex=r".*/documentModels/prebuilt-layout:analyze.*").mock(
        return_value=httpx.Response(202, headers={"operation-location": OP})
    )
    respx.get(OP).mock(
        return_value=httpx.Response(200, json={"status": "succeeded", "analyzeResult": {"content": "# Real", "tables": [{"rowCount": 1}]}})
    )

    result = t._process_single_with_state(_row(), "s-real", token_id=None)
    try:
        assert result.status == "success"
        out = result.row.to_dict()
        assert out["di_content"] == "# Real"
        assert out["di_tables"] == [{"rowCount": 1}]

        post_request = respx.calls[0].request
        assert post_request.method == "POST"
        assert post_request.headers["Ocp-Apim-Subscription-Key"] == "k"
        assert "_overload=analyzeDocument" in str(post_request.url)
        assert "api-version=2024-11-30" in str(post_request.url)
        assert respx.calls[-1].request.method == "GET"
    finally:
        with t._http_clients_lock:
            client = t._http_clients.pop("s-real", None)
        if client is not None:
            client.close()
