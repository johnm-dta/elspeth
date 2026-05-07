"""Smoke tests for pass-through invariant probe helpers.

These tests pin the transform-local ``probe_config()`` and
``forward_invariant_probe_rows()`` implementations that the ADR-009 harness
depends on. Without these, a pass-through annotation can exist while the
forward invariant silently exercises only error paths.
"""

from __future__ import annotations

import pytest

from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.keyword_filter import KeywordFilter
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.truncate import Truncate
from elspeth.plugins.transforms.type_coerce import TypeCoerce
from elspeth.plugins.transforms.value_transform import ValueTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


@pytest.mark.parametrize(
    ("transform_cls", "expected_added_fields"),
    [
        pytest.param(Truncate, {"truncate_probe_1"}, id="Truncate"),
        pytest.param(TypeCoerce, {"type_coerce_probe_1"}, id="TypeCoerce"),
        pytest.param(KeywordFilter, {"keyword_filter_probe_1"}, id="KeywordFilter"),
        pytest.param(ValueTransform, {"value_transform_probe_added_1"}, id="ValueTransform"),
        pytest.param(AzureContentSafety, {"content_safety_probe_text"}, id="AzureContentSafety"),
        pytest.param(AzurePromptShield, {"prompt_shield_probe_text"}, id="AzurePromptShield"),
        pytest.param(LLMTransform, {"llm_probe_text", "llm_response"}, id="LLMTransform"),
        pytest.param(
            RAGRetrievalTransform,
            {
                "rag_probe_query",
                "policy__rag_context",
                "policy__rag_score",
                "policy__rag_count",
                "policy__rag_sources",
            },
            id="RAGRetrievalTransform",
        ),
        pytest.param(
            WebScrapeTransform,
            {
                "web_scrape_probe_url",
                "page_content",
                "page_fingerprint",
                "fetch_status",
                "fetch_url_final",
                "fetch_url_final_ip",
            },
            id="WebScrapeTransform",
        ),
    ],
)
def test_pass_through_probe_helpers_drive_success_path(
    transform_cls: type,
    expected_added_fields: set[str],
) -> None:
    transform = transform_cls(transform_cls.probe_config())
    assert transform_cls.passes_through_input is True

    base_row = make_pipeline_row({"baseline": "kept"})
    probe_rows = transform.forward_invariant_probe_rows(base_row)

    assert len(probe_rows) == 1
    result = transform.execute_forward_invariant_probe(
        probe_rows,
        make_context(),
    )

    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"
    assert expected_added_fields.issubset(set(result.row.keys()))
