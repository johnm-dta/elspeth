"""Provider inference and environment contracts for composer LLM availability."""

from __future__ import annotations

PROVIDER_REQUIRED_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "azure": ("AZURE_API_KEY",),
    "azure_ai": ("AZURE_API_KEY",),
    # LiteLLM's Bedrock provider uses boto3's default AWS credential chain
    # (task role, environment, profile, etc.); Composer must not inject or
    # require a parallel static API-key contract.
    "bedrock": (),
    "openai": ("OPENAI_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
}


def infer_provider_from_model_name(model: str) -> str | None:
    """Infer provider from a provider-prefixed model string."""
    if "/" not in model:
        return None
    return model.split("/", 1)[0]


def infer_provider_from_unprefixed_model_name(model: str) -> str | None:
    """Infer provider for common unprefixed model families."""
    normalized = model.lower()
    if normalized.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if normalized.startswith("claude"):
        return "anthropic"
    return None
