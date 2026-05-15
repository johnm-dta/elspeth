"""First-message-of-session auto-titling.

Generates a 3-6 word session title from the user's first message via a
single LLM completion and persists it via the session service. Runs as
a background task spawned from the send_message route; awaited (with
timeout) after compose returns so the title is visible in the same
response cycle as the assistant reply.

Trust tier: the user message is Tier 3 (external), the LLM response is
Tier 3 (external — model output). Both are sanitized at this boundary
before being written to ``session.title`` (Tier 1, audit DB). No
Landscape audit entry is emitted — this is UI metadata, not a pipeline
decision. See CLAUDE.md "Three-Tier Trust Model" for the rationale.

Known gap: each first-message call is paid LLM traffic that bypasses
``composer_rate_limit_per_minute``. For demo-scale traffic this is
noise; production deployments should add a per-user-per-day cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from elspeth.web.composer.service import _composer_llm_seed_for_model, _litellm_acompletion

if TYPE_CHECKING:
    from elspeth.web.sessions.protocol import SessionServiceProtocol


_AUTO_TITLE_SYSTEM_PROMPT = (
    "You name chat sessions. Given the user's opening message, reply with "
    "a 3-6 word title that captures the topic. Reply with ONLY the title, "
    "no quotes, no punctuation at the end, no preamble. Use title case."
)

_AUTO_TITLE_MAX_LEN = 60
_AUTO_TITLE_MAX_TOKENS = 20
_AUTO_TITLE_TEMPERATURE = 0.0
_SURROUNDING_TITLE_QUOTES = (
    '"',
    "'",
    chr(0x201C),
    chr(0x201D),
    chr(0x2018),
    chr(0x2019),
)


def _sanitize_title(raw: str) -> str:
    """Coerce LLM output to a safe session title at the Tier 3 boundary.

    Strips surrounding whitespace, surrounding quotes (single, double,
    smart), collapses internal whitespace, and truncates to
    ``_AUTO_TITLE_MAX_LEN`` chars. Returns the empty string if nothing
    usable remains — caller skips the title update in that case.
    """
    cleaned = raw.strip()
    for quote in _SURROUNDING_TITLE_QUOTES:
        if cleaned.startswith(quote):
            cleaned = cleaned[1:]
        if cleaned.endswith(quote):
            cleaned = cleaned[:-1]
    cleaned = " ".join(cleaned.split())
    return cleaned[:_AUTO_TITLE_MAX_LEN]


async def maybe_auto_title_session(
    *,
    service: SessionServiceProtocol,
    session_id: UUID,
    user_message: str,
    model: str,
) -> None:
    """Generate and persist an auto-title for ``session_id``.

    One-shot LLM completion (no tools, deterministic temperature). On
    any failure — provider error, empty/garbage response, DB write
    failure — the function returns silently. Auto-titling is a UX
    nicety; surfacing its failures as user-visible errors would punish
    every send_message for an optional feature.
    """
    if not user_message.strip():
        return
    seed = _composer_llm_seed_for_model(model)
    kwargs: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _AUTO_TITLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "temperature": _AUTO_TITLE_TEMPERATURE,
        "max_tokens": _AUTO_TITLE_MAX_TOKENS,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        response = await _litellm_acompletion(**kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return
        title = _sanitize_title(content)
        if not title:
            return
        await service.update_session_title(session_id, title)
    except Exception:
        # Auto-titling is best-effort UI metadata; never poison the
        # send_message response with title-generation failures.
        return
