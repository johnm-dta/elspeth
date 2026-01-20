# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
from elspeth.plugins.llm.azure_batch import AzureBatchConfig, AzureBatchLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, PoolConfig
from elspeth.plugins.llm.batch_errors import BatchPendingError
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform
from elspeth.plugins.llm.templates import PromptTemplate, RenderedPrompt, TemplateError

# Public API exports (sorted alphabetically per RUF022)
# Note: AIMDThrottle, ReorderBuffer, PooledExecutor are internal
# and not exported. Import directly if needed for testing.
__all__ = [
    "AzureBatchConfig",
    "AzureBatchLLMTransform",
    "AzureLLMTransform",
    "AzureOpenAIConfig",
    "BaseLLMTransform",
    "BatchPendingError",
    "CapacityError",
    "LLMConfig",
    "OpenRouterConfig",
    "OpenRouterLLMTransform",
    "PoolConfig",
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateError",
]
