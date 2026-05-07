from __future__ import annotations

from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.testing import make_pipeline_row
from tests.invariants.test_pass_through_invariants import _probe_context, _probe_instantiate

FORWARD_SCOPE = (
    WebScrapeTransform,
    AzureContentSafety,
    AzurePromptShield,
    LLMTransform,
    RAGRetrievalTransform,
)

BACKWARD_SCOPE = (
    BatchStats,
    FieldMapper,
    JSONExplode,
)


def _assert_successful_probe_execution(
    cls: type[BaseTransform],
    *,
    direction: str,
) -> None:
    transform = _probe_instantiate(cls)
    base_row = make_pipeline_row({"baseline": "kept"})

    if direction == "forward":
        probe_rows = transform.forward_invariant_probe_rows(base_row)
        result = transform.execute_forward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )
    else:
        probe_rows = transform.backward_invariant_probe_rows(base_row)
        result = transform.execute_backward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )

    assert result.status == "success", (
        f"{cls.__name__} canonical {direction} invariant probe must execute successfully; got status={result.status!r}."
    )


def test_in_scope_forward_probes_execute_successfully_without_blind_skips() -> None:
    for cls in FORWARD_SCOPE:
        _assert_successful_probe_execution(cls, direction="forward")


def test_in_scope_backward_probes_execute_successfully_without_blind_skips() -> None:
    for cls in BACKWARD_SCOPE:
        _assert_successful_probe_execution(cls, direction="backward")
