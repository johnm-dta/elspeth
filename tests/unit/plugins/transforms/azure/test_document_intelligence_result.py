"""Unit tests for the pure Azure Document Intelligence analyzeResult parser."""

from __future__ import annotations

import pytest

from elspeth.plugins.transforms.azure.document_intelligence_result import (
    count_pages,
    extract_content,
    extract_facet_list,
    operation_id_from_url,
    operation_location_host_matches,
)
from elspeth.plugins.transforms.azure.errors import MalformedResponseError

_ENDPOINT = "https://di.cognitiveservices.azure.com"
_OP = (
    "https://di.cognitiveservices.azure.com/documentintelligence/documentModels/"
    "prebuilt-layout/analyzeResults/3b31320d-8bab?api-version=2024-11-30"
)


def test_extract_content_present_and_absent() -> None:
    assert extract_content({"content": "# Hi"}) == "# Hi"
    assert extract_content({}) == ""


def test_extract_content_wrong_type_fails_closed() -> None:
    with pytest.raises(MalformedResponseError):
        extract_content({"content": 123})


def test_extract_facet_list_present_absent_and_malformed() -> None:
    assert extract_facet_list({"tables": [{"rowCount": 1}]}, "tables") == [{"rowCount": 1}]
    assert extract_facet_list({}, "tables") == []
    with pytest.raises(MalformedResponseError):
        extract_facet_list({"tables": {"not": "a list"}}, "tables")


def test_count_pages() -> None:
    assert count_pages({"pages": [{}, {}, {}]}) == 3
    assert count_pages({}) == 0
    with pytest.raises(MalformedResponseError):
        count_pages({"pages": "x"})


def test_host_match_accepts_same_host_https() -> None:
    assert operation_location_host_matches(_OP, _ENDPOINT)


def test_host_match_rejects_other_host() -> None:
    assert not operation_location_host_matches("https://evil.example.com/analyzeResults/abc", _ENDPOINT)


def test_host_match_rejects_http_scheme() -> None:
    assert not operation_location_host_matches("http://di.cognitiveservices.azure.com/x", _ENDPOINT)


def test_host_match_rejects_same_host_different_port() -> None:
    # api-key must not be sent to a same-host attacker port (F1).
    assert not operation_location_host_matches("https://di.cognitiveservices.azure.com:8443/analyzeResults/abc", _ENDPOINT)


def test_host_match_accepts_explicit_default_port() -> None:
    # An explicit :443 matches an endpoint with no port (both normalize to 443).
    assert operation_location_host_matches("https://di.cognitiveservices.azure.com:443/analyzeResults/abc", _ENDPOINT)


def test_host_match_rejects_garbage() -> None:
    assert not operation_location_host_matches("not a url", _ENDPOINT)


def test_host_match_is_case_insensitive() -> None:
    assert operation_location_host_matches("https://DI.CognitiveServices.Azure.Com/analyzeResults/abc", _ENDPOINT)


def test_operation_id_from_url() -> None:
    assert operation_id_from_url(_OP) == "3b31320d-8bab"
    assert operation_id_from_url("https://h/no/result/segment") is None
