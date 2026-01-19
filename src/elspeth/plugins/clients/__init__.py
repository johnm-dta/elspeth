# src/elspeth/plugins/clients/__init__.py
"""Audited clients that automatically record external calls to the audit trail.

These clients wrap external service calls (LLM, HTTP) and ensure every
request/response is recorded to the Landscape audit trail for complete
traceability.

Example:
    from elspeth.plugins.clients import AuditedLLMClient, AuditedHTTPClient

    # Create audited LLM client
    llm_client = AuditedLLMClient(
        recorder=recorder,
        state_id=state_id,
        underlying_client=openai.OpenAI(),
        provider="openai",
    )

    # All calls are automatically recorded
    response = llm_client.chat_completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )
"""

from elspeth.plugins.clients.base import AuditedClientBase
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
    LLMClientError,
    LLMResponse,
    RateLimitError,
)

__all__ = [
    "AuditedClientBase",
    "AuditedHTTPClient",
    "AuditedLLMClient",
    "LLMClientError",
    "LLMResponse",
    "RateLimitError",
]
