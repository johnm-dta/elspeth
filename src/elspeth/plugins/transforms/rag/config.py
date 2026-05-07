"""Configuration for RAG retrieval transform."""

from __future__ import annotations

import keyword
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.infrastructure.config_base import TransformDataConfig

if TYPE_CHECKING:
    from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalProvider

ProviderFactory = Callable[..., "RetrievalProvider"]

# Registry entry: (config class, provider class or factory callable)
_ProviderEntry = tuple[type[Any], Callable[..., Any]]


def _get_providers() -> dict[str, _ProviderEntry]:
    """Lazy provider registry — only imports providers whose deps are installed."""
    providers: dict[str, _ProviderEntry] = {}

    try:
        from elspeth.plugins.infrastructure.clients.retrieval.azure_search import (
            AzureSearchProvider,
            AzureSearchProviderConfig,
        )

        providers["azure_search"] = (AzureSearchProviderConfig, AzureSearchProvider)
    except ModuleNotFoundError:
        pass  # azure SDK not installed — azure_search provider unavailable

    try:
        from elspeth.plugins.infrastructure.clients.retrieval.chroma import (
            ChromaSearchProvider,
            ChromaSearchProviderConfig,
        )

        def _chroma_factory(config: ChromaSearchProviderConfig, *, execution: Any, run_id: Any, **_kwargs: Any) -> ChromaSearchProvider:
            """Chroma uses the SDK directly — passes execution repo and run_id for audit trail.

            execution and run_id are mandatory (not defaulted to None) because Chroma
            search calls must be recorded in the audit trail (B1 fix). If the engine
            ever calls this factory without execution, it should crash at startup, not
            silently skip audit recording at query time.
            """
            return ChromaSearchProvider(config=config, execution=execution, run_id=run_id)

        providers["chroma"] = (ChromaSearchProviderConfig, _chroma_factory)
    except ModuleNotFoundError:
        pass  # chromadb not installed — chroma provider unavailable

    return providers


PROVIDERS: dict[str, _ProviderEntry] = _get_providers()


class RAGRetrievalConfig(TransformDataConfig):
    """Configuration for the rag_retrieval transform plugin."""

    @field_validator("schema_config", mode="before")
    @classmethod
    def coerce_schema_config(cls, v: object) -> object:
        """Accept raw dict for schema_config — convert via SchemaConfig.from_dict."""
        if isinstance(v, dict):
            return SchemaConfig.from_dict(v)
        return v

    output_prefix: str
    query_field: str
    query_template: str | None = None
    query_pattern: str | None = None
    provider: str
    provider_config: dict[str, Any]
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    on_no_results: Literal["quarantine", "continue"] = "quarantine"
    context_format: Literal["numbered", "separated", "raw"] = "numbered"
    context_separator: str = "\n---\n"
    max_context_length: int | None = Field(default=None, ge=1)

    @field_validator("output_prefix")
    @classmethod
    def validate_prefix(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(f"output_prefix must be a valid Python identifier, got {v!r}")
        if keyword.iskeyword(v):
            raise ValueError(
                f"output_prefix must not be a Python keyword, got {v!r}. "
                f"Keywords like 'class', 'return' produce field names that break Jinja2 templates."
            )
        return v

    @model_validator(mode="after")
    def validate_query_modes(self) -> Self:
        if self.query_template and self.query_pattern:
            raise ValueError("query_template and query_pattern are mutually exclusive")
        return self

    @model_validator(mode="after")
    def validate_provider_config(self) -> Self:
        provider_entry = PROVIDERS.get(self.provider)
        if provider_entry is None:
            raise ValueError(f"Unknown provider: {self.provider!r}. Available: {sorted(PROVIDERS)}")
        config_cls, _ = provider_entry
        config_cls(**self.provider_config)
        return self

    @field_validator("query_pattern")
    @classmethod
    def validate_regex(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                re.compile(v)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {e}") from e
        return v
