# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
from elspeth.plugins.llm.azure_batch import AzureBatchConfig, AzureBatchLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig
from elspeth.plugins.llm.batch_errors import BatchPendingError
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform
from elspeth.plugins.llm.templates import PromptTemplate, RenderedPrompt, TemplateError

__all__ = [
    "AzureBatchConfig",
    "AzureBatchLLMTransform",
    "AzureLLMTransform",
    "AzureOpenAIConfig",
    "BaseLLMTransform",
    "BatchPendingError",
    "LLMConfig",
    "OpenRouterConfig",
    "OpenRouterLLMTransform",
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateError",
]
