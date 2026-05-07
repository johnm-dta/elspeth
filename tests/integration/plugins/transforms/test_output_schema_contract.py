"""Integration test for output schema contract enforcement.

Tests the full contract chain using real transform instances:
1. Transform construction populates _output_schema_config via helper
2. _validate_output_schema_contract passes for correctly-configured transforms
3. _validate_output_schema_contract raises FrameworkBugError when contract missing
4. Invariant 2: guaranteed_fields is superset of declared_output_fields
"""

import pytest

from elspeth.contracts.errors import FrameworkBugError
from elspeth.core.dag.builder import _validate_output_schema_contract
from elspeth.plugins.transforms.batch_replicate import BatchReplicate
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform


@pytest.fixture(autouse=True)
def _set_fingerprint_key(monkeypatch):
    monkeypatch.setenv("ELSPETH_FINGERPRINT_KEY", "test-fingerprint-key")


def _make_rag_transform(*, output_prefix: str = "sci", query_field: str = "q") -> RAGRetrievalTransform:
    """Construct RAG with the base-install provider path, not optional extras."""
    config = RAGRetrievalTransform.probe_config()
    config["output_prefix"] = output_prefix
    config["query_field"] = query_field
    return RAGRetrievalTransform(config)


class TestContractInvariantsAcrossAllTransforms:
    """Verify Invariant 2 (guaranteed_fields superset of declared_output_fields)
    holds for every field-adding transform with real instances."""

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(
                lambda: _make_rag_transform(),
                id="rag",
            ),
            pytest.param(
                lambda: JSONExplode(
                    {"array_field": "items", "output_field": "item", "include_index": True, "schema": {"mode": "observed"}}
                ),
                id="json_explode",
            ),
            pytest.param(
                lambda: BatchReplicate({"include_copy_index": True, "schema": {"mode": "observed"}}),
                id="batch_replicate",
            ),
            pytest.param(
                lambda: FieldMapper({"mapping": {"a": "b"}, "strict": True, "schema": {"mode": "observed"}}),
                id="field_mapper",
            ),
            pytest.param(
                lambda: BatchStats({"value_field": "amount", "schema": {"mode": "observed"}}),
                id="batch_stats",
            ),
            pytest.param(
                lambda: WebScrapeTransform(
                    {
                        "url_field": "url",
                        "content_field": "page_content",
                        "fingerprint_field": "page_hash",
                        "http": {"abuse_contact": "test@example.com", "scraping_reason": "Integration test"},
                        "schema": {"mode": "observed"},
                    }
                ),
                id="web_scrape",
            ),
        ],
    )
    def test_invariant2_guaranteed_superset_of_declared(self, transform_factory):
        """Every field-adding transform's guaranteed_fields contains all declared_output_fields."""
        transform = transform_factory()
        assert transform._output_schema_config is not None, f"{transform.name}: _output_schema_config is None"
        guaranteed = frozenset(transform._output_schema_config.guaranteed_fields)
        assert transform.declared_output_fields.issubset(guaranteed), (
            f"{transform.name}: declared_output_fields {transform.declared_output_fields} not a subset of guaranteed_fields {guaranteed}"
        )

    @pytest.mark.parametrize(
        "transform_factory",
        [
            pytest.param(
                lambda: _make_rag_transform(),
                id="rag",
            ),
            pytest.param(
                lambda: JSONExplode({"array_field": "items", "output_field": "item", "schema": {"mode": "observed"}}),
                id="json_explode",
            ),
            pytest.param(
                lambda: FieldMapper({"mapping": {"a": "b"}, "strict": True, "schema": {"mode": "observed"}}),
                id="field_mapper",
            ),
        ],
    )
    def test_enforcement_passes_for_valid_transforms(self, transform_factory):
        """Transforms with declared_output_fields AND _output_schema_config pass validation."""
        transform = transform_factory()
        _validate_output_schema_contract(transform)  # Should not raise

    def test_enforcement_fires_on_missing_contract(self):
        """A real transform with cleared _output_schema_config triggers FrameworkBugError."""
        transform = _make_rag_transform()
        transform._output_schema_config = None

        with pytest.raises(FrameworkBugError, match="declares output fields"):
            _validate_output_schema_contract(transform)

    def test_rag_guaranteed_fields_exact(self):
        """RAG transform's guaranteed_fields contains exactly the 4 declared output fields."""
        transform = _make_rag_transform()
        assert frozenset(transform._output_schema_config.guaranteed_fields) == frozenset(
            {"sci__rag_context", "sci__rag_score", "sci__rag_count", "sci__rag_sources"}
        )
