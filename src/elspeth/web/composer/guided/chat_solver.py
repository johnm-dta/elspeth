"""Per-step chat solver: invoke LLM with step-scoped skill briefing, return advisory text.

Phase A is intentionally **advisory-only** — no tools, no state mutation.  The
LLM receives the base preamble + the playbook for the user's current wizard
step, plus the user's typed message, and replies with prose.  Per the plan
file at /home/john/.claude/plans/please-investigate-the-new-fizzy-kite.md,
Phase B introduces the per-step tool palette + Tier-3 args validation.

Audit: ``solve_step_chat`` itself does not record. The route handler
(``post_guided_chat`` in ``web/sessions/routes.py``) constructs a
``ComposerChatTurn`` from the ``StepChatResult`` returned by
``solve_step_chat_with_auto_drop`` and persists it via the
``BufferingRecorder`` drain. No ``ComposerLLMCall`` row is currently emitted
for chat calls; this is a known asymmetry with the chain-solver path, which
emits ``ComposerLLMCall`` via explicit ``recorder.record_llm_call`` calls in
``_guided_solve_chain.py``. Closing that asymmetry is Phase B work.
"""

from __future__ import annotations

from typing import Any

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.prompts import load_step_chat_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.service import (
    _COMPOSER_LLM_TEMPERATURE,
    _composer_llm_seed_for_model,
    _litellm_acompletion,
)


async def solve_step_chat(
    *,
    model: str,
    step: GuidedStep,
    user_message: str,
) -> str:
    """Send a user chat message to the LLM scoped to *step*; return the assistant reply.

    Args:
        model: LiteLLM model identifier from settings.composer_model.  Required —
            callers must be explicit; no hard-coded default (mirrors solve_chain).
        step: The user's current wizard step.  Determines which playbook the
            LLM receives via ``load_step_chat_skill(step)``.
        user_message: The user's typed message.  Tier 3 by trust model — the
            route handler is responsible for non-empty / length validation
            before this is called.

    Returns:
        The assistant's reply as a plain string (no tool calls in Phase A).

    Raises:
        InvariantError: when the LLM response has no message content (a
            defective response we cannot recover from — surface loudly per
            CLAUDE.md offensive-programming discipline).
    """
    if not user_message:
        # Defensive against empty string only: route handler should have caught
        # this, so reaching here means a server-side caller bug, not user input.
        raise InvariantError("solve_step_chat: user_message is empty (route validation gap)")

    system_prompt = load_step_chat_skill(step)
    seed = _composer_llm_seed_for_model(model)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": _COMPOSER_LLM_TEMPERATURE,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = await _litellm_acompletion(**kwargs)

    message = response.choices[0].message
    # LiteLLM's typed contract: message.content is str | None (None when the
    # response is a tool-call only).  Phase A doesn't attach tools, so a None
    # or empty content is a defective response from the model — crash loudly
    # per CLAUDE.md offensive-programming discipline.  We trust LiteLLM's
    # type contract for "is a string"; if the dependency violates its own
    # typing, .strip() raises AttributeError immediately at this site (still
    # loud, no silent degradation).
    content = message.content
    if content is None or not content.strip():
        raise InvariantError(f"solve_step_chat: LLM response missing message content (step={step.value}, model={model!r})")
    # mypy: LiteLLM's response is `Any`, narrow to str at the trust boundary.
    return str(content)
