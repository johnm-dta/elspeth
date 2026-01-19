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

For replay mode, use CallReplayer to return recorded responses:

    from elspeth.plugins.clients import CallReplayer, ReplayMissError

    replayer = CallReplayer(recorder, source_run_id="run-abc123")
    result = replayer.replay(call_type="llm", request_data={...})
"""

from elspeth.plugins.clients.base import AuditedClientBase
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.clients.llm import (
    AuditedLLMClient,
    LLMClientError,
    LLMResponse,
    RateLimitError,
)
from elspeth.plugins.clients.replayer import (
    CallReplayer,
    ReplayedCall,
    ReplayMissError,
)

__all__ = [
    "AuditedClientBase",
    "AuditedHTTPClient",
    "AuditedLLMClient",
    "CallReplayer",
    "LLMClientError",
    "LLMResponse",
    "RateLimitError",
    "ReplayedCall",
    "ReplayMissError",
]
