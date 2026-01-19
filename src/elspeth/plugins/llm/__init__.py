# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.azure import AzureLLMTransform, AzureOpenAIConfig
from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig
from elspeth.plugins.llm.openrouter import OpenRouterConfig, OpenRouterLLMTransform
from elspeth.plugins.llm.templates import PromptTemplate, RenderedPrompt, TemplateError

__all__ = [
    "AzureLLMTransform",
    "AzureOpenAIConfig",
    "BaseLLMTransform",
    "LLMConfig",
    "OpenRouterConfig",
    "OpenRouterLLMTransform",
    "PromptTemplate",
    "RenderedPrompt",
    "TemplateError",
]
