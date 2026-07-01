"""Guided-mode skill loading + Step 3 context-block construction.

Skills are split per step:

  skills/base.md                  Always-applies preamble + hard rules.
  skills/step_1_source.md         Step-1 playbook.
  skills/step_2_sink.md           Step-2 playbook.
  skills/step_2_5_recipe_match.md Step-2.5 playbook.
  skills/step_3_transforms.md     Step-3 playbook + sample-value eyeballing.
  skills/step_4_wire.md           Step-4 wiring constraints.

``load_guided_skill()`` composes all five (base + every step) and is consumed
by the chain solver, which serves Step 3 but historically receives the full
playbook for breadth.

``load_step_chat_skill(step)`` composes base + one step, scoped to the user's
current wizard position. Consumed by the per-step chat solver.

All loaders are module-cached via ``@lru_cache``; per project memory, restart
elspeth-web.service after editing any skill markdown for live changes to take
effect.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.web.composer.guided.protocol import GuidedStep

if TYPE_CHECKING:
    from elspeth.web.composer.guided.state_machine import (
        SinkResolved,
        SourceResolved,
    )

_SKILLS_DIR = Path(__file__).parent / "skills"

# CLOSED LIST — must match the GuidedStep enum members.  If a new step is
# added to GuidedStep, add its skill file here in playbook order.
_STEP_FILE_NAMES: dict[GuidedStep, str] = {
    GuidedStep.STEP_1_SOURCE: "step_1_source.md",
    GuidedStep.STEP_2_SINK: "step_2_sink.md",
    GuidedStep.STEP_2_5_RECIPE_MATCH: "step_2_5_recipe_match.md",
    GuidedStep.STEP_3_TRANSFORMS: "step_3_transforms.md",
    GuidedStep.STEP_4_WIRE: "step_4_wire.md",
}

# Playbook order — the order steps appear when composing the full skill.
# Mirrors the natural wizard progression and is asserted at module import to
# match the GuidedStep enum membership exactly (see assertion below).
_STEP_PLAYBOOK_ORDER: tuple[GuidedStep, ...] = (
    GuidedStep.STEP_1_SOURCE,
    GuidedStep.STEP_2_SINK,
    GuidedStep.STEP_2_5_RECIPE_MATCH,
    GuidedStep.STEP_3_TRANSFORMS,
    GuidedStep.STEP_4_WIRE,
)

# Discoverability invariant: the per-step file map and the playbook order
# must cover every GuidedStep member.  If GuidedStep gains a new member and
# either map is not updated, fail loudly at import time rather than silently
# omit the step from the composed skill.
assert set(_STEP_FILE_NAMES.keys()) == set(GuidedStep), (
    f"_STEP_FILE_NAMES out of sync with GuidedStep: "
    f"missing {set(GuidedStep) - set(_STEP_FILE_NAMES)}, "
    f"extra {set(_STEP_FILE_NAMES) - set(GuidedStep)}"
)
assert set(_STEP_PLAYBOOK_ORDER) == set(GuidedStep), (
    f"_STEP_PLAYBOOK_ORDER out of sync with GuidedStep: "
    f"missing {set(GuidedStep) - set(_STEP_PLAYBOOK_ORDER)}, "
    f"extra {set(_STEP_PLAYBOOK_ORDER) - set(GuidedStep)}"
)


@lru_cache(maxsize=1)
def _load_base() -> str:
    """Load the always-applies preamble (intro + hard rules + per-step header)."""
    return (_SKILLS_DIR / "base.md").read_text(encoding="utf-8")


@lru_cache(maxsize=len(GuidedStep))
def _load_step(step: GuidedStep) -> str:
    """Load the per-step playbook fragment for *step*."""
    return (_SKILLS_DIR / _STEP_FILE_NAMES[step]).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_guided_skill() -> str:
    """Compose the full guided skill (base + every step in playbook order).

    Used by the chain solver, which historically receives the full playbook
    for breadth even though it only acts on Step 3.  Cached per process;
    restart elspeth-web.service after editing any skill markdown.
    """
    parts = [_load_base()]
    parts.extend(_load_step(step) for step in _STEP_PLAYBOOK_ORDER)
    return "\n\n".join(part.rstrip() for part in parts) + "\n"


@lru_cache(maxsize=len(GuidedStep))
def load_step_chat_skill(step: GuidedStep) -> str:
    """Compose base + the per-step playbook for *step* only.

    Used by the per-step chat solver, which scopes the LLM's awareness to
    just the step the user is currently on.  Cached per process; restart
    elspeth-web.service after editing skill markdown.
    """
    return f"{_load_base().rstrip()}\n\n{_load_step(step).rstrip()}\n"


def build_mode_transition_system_prompt(*, terminal_reason: str, freeform_skill: str) -> str:
    """Construct the guided→freeform transition prompt: freeform skill + transition message.

    The ``freeform_skill`` parameter must be supplied by the caller — typically via
    ``build_system_prompt(data_dir)`` in ``composer/prompts.py``.  This keeps the
    deployment overlay (``data_dir``) correctly threaded into the transition
    prompt, and avoids a circular import: if this module called ``build_system_prompt``
    directly it would create a guided/prompts ↔ composer/prompts import cycle.

    Args:
        terminal_reason: String reason token from the completed guided session
            (e.g. ``"completed_pipeline"``, ``"user_pressed_exit"``).
        freeform_skill: Fully processed freeform composer skill string — core skill
            with deployment overlay appended.
            Produced by ``build_system_prompt(data_dir)``.

    Returns:
        Layered prompt string: freeform skill \\n\\n transition header.
    """
    transition = (
        f"## Mode Transition — Guided → Freeform\n\n"
        f"You have just exited guided mode (reason: {terminal_reason}).\n\n"
        "Any previous guided-mode protocol restrictions (closed turn taxonomy, "
        "read-only state, legal-turn matrix) are LIFTED for the remainder of this "
        "session. Use the full freeform tool surface described above. The guided "
        "session's outcome is recorded in `composition_state.guided_session` — "
        "do not re-run any work it already accomplished."
    )
    return f"{freeform_skill}\n\n{transition}"


def build_repair_addendum(*, validation_error: str) -> str:
    """Render the REPAIR ATTEMPT addendum appended to a repair solve_chain call.

    Args:
        validation_error: Validation error text, taken verbatim from the failing
            ToolResult; Tier 1 audit data, no paraphrasing.
    """
    return (
        "REPAIR ATTEMPT — your previous proposal failed validation:\n"
        f"{validation_error}\n\n"
        "Propose a corrected chain that fixes the named validation errors."
    )


def build_revise_addendum(*, revise_instruction: str) -> str:
    """Render the REVISE REQUEST addendum appended to a revise solve_chain call.

    Distinct from :func:`build_repair_addendum`: a revise is a user instruction
    to CHANGE the current proposal, not a report that the proposal failed
    validation. Framing it as the latter (the repair addendum) misleads the
    model into "correcting errors" that do not exist.

    Args:
        revise_instruction: The user's revise message, verbatim; Tier 1 audit
            data, no paraphrasing.
    """
    return (
        "REVISE REQUEST — the user wants to change the current proposal as "
        "follows:\n"
        f"{revise_instruction}\n\n"
        "Propose an updated chain that applies this change."
    )


def _looks_secret_like_sample(value: str) -> bool:
    lowered = value.lower()
    secret_markers = (
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "bearer ",
        "client_secret",
        "password",
        "secret",
        "token",
    )
    return lowered.startswith(("sk-", "pk_", "rk_", "xoxb-")) or any(marker in lowered for marker in secret_markers)


def _summarize_sample_value(value: Any) -> str:
    if value is None:
        return "<sample:null>"
    if isinstance(value, bool):
        return "<sample:boolean>"
    if isinstance(value, int | float):
        return "<sample:number>"
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return "<sample:empty-string>"
        lowered = stripped.lower()
        if lowered.startswith(("http://", "https://")):
            return "<sample:url>"
        if "@" in stripped and "." in stripped.rsplit("@", 1)[-1]:
            return "<sample:email-like>"
        if _looks_secret_like_sample(stripped):
            return "<sample:secret-like>"
        return f"<sample:string:{len(stripped)}-chars>"
    if isinstance(value, Mapping):
        return f"<sample:object:{len(value)}-keys>"
    if isinstance(value, (list, tuple)):
        return f"<sample:array:{len(value)}-items>"
    return f"<sample:{type(value).__name__}>"


def _summarize_sample_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): _summarize_sample_value(value) for key, value in row.items()}


def build_step_3_context_block(
    *,
    source: SourceResolved,
    sink: SinkResolved,
) -> str:
    """Render the GUIDED CONTEXT block for the Step 3 LLM prompt."""
    src_payload = {
        "plugin": source.plugin,
        "columns": list(source.observed_columns),
        "sample": [_summarize_sample_row(r) for r in source.sample_rows[:3]],
    }
    sink_payload = {
        "outputs": [
            {
                "plugin": o.plugin,
                "required_fields": list(o.required_fields),
                "schema_mode": o.schema_mode,
            }
            for o in sink.outputs
        ],
    }
    return f"GUIDED CONTEXT (server-resolved):\nsource: {json.dumps(src_payload)}\nsink: {json.dumps(sink_payload)}\n"


@lru_cache(maxsize=1)
def guided_staged_skill_hash() -> str:
    """Hex SHA-256 over base.md + every step playbook in _STEP_PLAYBOOK_ORDER.

    Consumed by the tutorial run-cache key (tutorial_model_id, cache input
    #3). Enumerating the playbook order means appending a GuidedStep member
    (and its skill file) automatically extends the keyed input set — the
    step_4_wire.md add (P1) shifts this hash with no edit to the cache path.

    Cached per process; restart elspeth-web.service after editing skill
    markdown (same lifecycle caveat as the other loaders in this module).
    """
    digest = hashlib.sha256()
    digest.update((_SKILLS_DIR / "base.md").read_bytes())
    for step in _STEP_PLAYBOOK_ORDER:
        digest.update((_SKILLS_DIR / _STEP_FILE_NAMES[step]).read_bytes())
    return digest.hexdigest()
