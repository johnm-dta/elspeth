"""Tests for RAGRetrievalConfig validation."""

import pytest

from elspeth.plugins.transforms.rag.config import RAGRetrievalConfig


def _valid_config(**overrides):
    """Build a valid config dict with overrides."""
    base = {
        "output_prefix": "policy",
        "query_field": "question",
        "provider": "azure_search",
        "provider_config": {
            "endpoint": "https://test.search.windows.net",
            "index": "test-index",
            "api_key": "test-key",
        },
        "schema_config": {"mode": "observed"},
    }
    base.update(overrides)
    return base


class TestOutputPrefix:
    def test_valid_identifier(self):
        config = RAGRetrievalConfig(**_valid_config(output_prefix="financial"))
        assert config.output_prefix == "financial"

    def test_rejects_non_identifier(self):
        with pytest.raises(ValueError, match="valid Python identifier"):
            RAGRetrievalConfig(**_valid_config(output_prefix="123invalid"))

    def test_rejects_keyword(self):
        with pytest.raises(ValueError, match="Python keyword"):
            RAGRetrievalConfig(**_valid_config(output_prefix="class"))

    def test_rejects_with_spaces(self):
        with pytest.raises(ValueError, match="valid Python identifier"):
            RAGRetrievalConfig(**_valid_config(output_prefix="has spaces"))


class TestQueryField:
    def test_valid_identifier_is_stripped(self):
        config = RAGRetrievalConfig(**_valid_config(query_field=" question "))
        assert config.query_field == "question"

    @pytest.mark.parametrize("query_field", ["", "   ", "has spaces", "question.text", "class"])
    def test_rejects_invalid_query_field(self, query_field):
        with pytest.raises(ValueError, match="query_field"):
            RAGRetrievalConfig(**_valid_config(query_field=query_field))


class TestQueryModes:
    def test_template_and_pattern_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            RAGRetrievalConfig(
                **_valid_config(
                    query_template="{{ query }}",
                    query_pattern=r"\w+",
                )
            )

    def test_template_only_ok(self):
        config = RAGRetrievalConfig(**_valid_config(query_template="{{ query }}"))
        assert config.query_template == "{{ query }}"

    def test_pattern_only_ok(self):
        config = RAGRetrievalConfig(**_valid_config(query_pattern=r"issue:\s*(.+)"))
        assert config.query_pattern == r"issue:\s*(.+)"

    def test_invalid_regex_rejected(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            RAGRetrievalConfig(**_valid_config(query_pattern=r"(unclosed"))


class TestRetrievalParams:
    def test_top_k_bounds(self):
        config = RAGRetrievalConfig(**_valid_config(top_k=1))
        assert config.top_k == 1

        config = RAGRetrievalConfig(**_valid_config(top_k=100))
        assert config.top_k == 100

        with pytest.raises(ValueError):
            RAGRetrievalConfig(**_valid_config(top_k=0))

        with pytest.raises(ValueError):
            RAGRetrievalConfig(**_valid_config(top_k=101))

    def test_min_score_bounds(self):
        config = RAGRetrievalConfig(**_valid_config(min_score=0.0))
        assert config.min_score == 0.0

        config = RAGRetrievalConfig(**_valid_config(min_score=1.0))
        assert config.min_score == 1.0

        with pytest.raises(ValueError):
            RAGRetrievalConfig(**_valid_config(min_score=-0.1))

        with pytest.raises(ValueError):
            RAGRetrievalConfig(**_valid_config(min_score=1.1))


class TestProviderConfig:
    def test_unknown_provider_rejected(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            RAGRetrievalConfig(**_valid_config(provider="unknown"))

    def test_invalid_provider_config_rejected_eagerly(self):
        """Provider config is validated at YAML load time, not first row."""
        with pytest.raises(ValueError):
            RAGRetrievalConfig(
                **_valid_config(
                    provider_config={"endpoint": "http://no-https.example.com", "index": "test", "api_key": "k"},
                )
            )

    def test_max_context_length_ge_1(self):
        config = RAGRetrievalConfig(**_valid_config(max_context_length=1))
        assert config.max_context_length == 1

        with pytest.raises(ValueError):
            RAGRetrievalConfig(**_valid_config(max_context_length=0))
