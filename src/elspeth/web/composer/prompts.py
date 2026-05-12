"""System prompt and message construction for the LLM composer.

build_messages() returns a NEW list on every call — never a cached
reference. This is critical because the tool-use loop appends to the
list during iteration.

Layer: L3 (application).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.guided.prompts import build_mode_transition_system_prompt
from elspeth.web.composer.guided.state_machine import TerminalKind
from elspeth.web.composer.redaction import redact_source_storage_path
from elspeth.web.composer.skills import load_deployment_skill, load_skill
from elspeth.web.composer.state import CompositionState

if TYPE_CHECKING:
    from elspeth.web.composer.guided.state_machine import TerminalState

# Load the pipeline composer skill once at module level (static content).
_PIPELINE_SKILL = load_skill("pipeline_composer")

# SYSTEM_PROMPT is bound below, once the strip helpers are defined — it is the
# advisor-enabled, no-deployment-layer projection of the loaded skill (i.e.,
# what ``build_system_prompt(None, advisor_enabled=True)`` returns). Exported
# for tests that need to assert identity with the core skill; build_messages
# no longer uses it as a fast path (the F1 fix routes every call through
# ``build_system_prompt`` so advisor-strip applies consistently).


def _strip_advisor_content(text: str) -> str:
    r"""Remove advisor-specific content from skill text.

    When ``composer_advisor_enabled`` is False, the system prompt fed to
    the composer LLM must NOT teach it about ``request_advisor_hint``.
    Otherwise the LLM sees the tool name in the skill, attempts to call
    it, and hits the defense-in-depth "disabled" rejection — wasting a
    turn for no reason and leaving the model confused about why a tool
    the skill described doesn't actually exist.

    Removes:

    1. The ``, `request_advisor_hint``` token from the Step-0 Diagnostics
       line (leaves ``- **Diagnostics:** `explain_validation_error```).
    2. The dedicated ``#### When You Are Still Stuck — `request_advisor_hint```
       subsection — from its heading through to (but not including) the
       next ``#### `` heading.
    3. Any content wrapped in ``<!-- ADVISOR-ONLY -->...<!-- /ADVISOR-ONLY -->``
       markers — used for advisor-conditional clauses inline in prose where
       a section-level strip would damage surrounding content (e.g., table
       rows, inline guidance referencing the advisor by name).
    4. The marker tags ``<!-- ADVISOR-DISABLED -->...<!-- /ADVISOR-DISABLED -->``
       are removed but their *content* is kept — these wrap fallback prose
       that should only reach the LLM when advisor is disabled. The mirror
       function ``_strip_advisor_disabled_fallback`` strips both the markers
       *and* the content when advisor is enabled.

    The transformation operates on the loaded skill text without touching
    the on-disk file, so the parity test
    ``TestComposerToolNameDrift::test_skill_step0_matches_get_tool_definitions``
    (which scans the unfiltered file content) is unaffected.
    """
    text = text.replace(", `request_advisor_hint`", "")
    start = text.find("#### When You Are Still Stuck")
    if start != -1:
        end = text.find("\n#### ", start + 1)
        if end == -1:
            # Advisor subsection is the trailing subsection of its parent
            # section. Strip from the heading to the next top-level (``## ``)
            # heading, or to end-of-file if none.
            next_h2 = text.find("\n## ", start + 1)
            text = text[:start].rstrip() + "\n" if next_h2 == -1 else text[:start] + text[next_h2 + 1 :]
        else:
            text = text[:start] + text[end + 1 :]  # +1 to skip the leading newline
    text = re.sub(r"<!-- ADVISOR-ONLY -->.*?<!-- /ADVISOR-ONLY -->", "", text, flags=re.DOTALL)
    text = text.replace("<!-- ADVISOR-DISABLED -->", "").replace("<!-- /ADVISOR-DISABLED -->", "")
    return text


def _strip_advisor_disabled_fallback(text: str) -> str:
    """Inverse of ``_strip_advisor_content``: when advisor IS enabled, strip
    the ``<!-- ADVISOR-DISABLED -->...<!-- /ADVISOR-DISABLED -->`` blocks
    (markers and content) so the LLM doesn't see contradictory fallback
    guidance that only applies on advisor-disabled deployments."""
    return re.sub(r"<!-- ADVISOR-DISABLED -->.*?<!-- /ADVISOR-DISABLED -->", "", text, flags=re.DOTALL)


SYSTEM_PROMPT = _strip_advisor_disabled_fallback(_PIPELINE_SKILL)


@lru_cache(maxsize=8)
def build_system_prompt(data_dir: str | None = None, *, advisor_enabled: bool = True) -> str:
    """Build the full system prompt: core skill + optional deployment skill.

    The deployment skill is loaded from ``{data_dir}/skills/pipeline_composer.md``
    if it exists.  This lets operators inject company-specific knowledge
    (provider mappings, custom patterns, domain vocabulary) without editing
    the core skill pack.

    When ``advisor_enabled`` is False, advisor-specific sections are stripped
    from the core skill before deployment overlay (deployment skills are
    operator-controlled and not stripped — operators must police their own
    overlays).

    Cached per ``(data_dir, advisor_enabled)`` pair — the deployment skill
    is read once from disk per unique combination, not on every LLM call.
    Cache size 8 covers the realistic combinations
    (None x {True, False} plus 3 typical data_dirs x {True, False}).

    Args:
        data_dir: Root data directory.  ``None`` skips the deployment layer.
        advisor_enabled: When False, strip advisor-specific content from
            the core skill so the LLM does not learn about a tool that
            will reject its calls as "disabled".

    Returns:
        Combined system prompt string.
    """
    core = _strip_advisor_disabled_fallback(_PIPELINE_SKILL) if advisor_enabled else _strip_advisor_content(_PIPELINE_SKILL)
    deployment = load_deployment_skill("pipeline_composer", data_dir)
    if deployment:
        return core + "\n\n---\n\n" + deployment
    return core


def build_context_string(
    state: CompositionState,
    catalog: CatalogService,
) -> str:
    """Build the injected context string with current state and plugin summary.

    Args:
        state: Current composition state.
        catalog: For building the plugin summary.

    Returns:
        A string with state and plugin info, suitable for appending to the
        system prompt.
    """
    serialized = state.to_dict()
    serialized = redact_source_storage_path(serialized)  # B4: hide blob storage paths
    validation = state.validate()
    serialized["validation"] = {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
    }

    # Build lightweight plugin summary (names only).
    # CatalogService returns PluginSummary instances — use .name attribute.
    source_names = [p.name for p in catalog.list_sources()]
    transform_names = [p.name for p in catalog.list_transforms()]
    sink_names = [p.name for p in catalog.list_sinks()]

    context = {
        "current_state": serialized,
        "composer_progress": {
            "state_exists": state.source is not None or bool(state.nodes) or bool(state.outputs),
        },
        "available_plugins": {
            "sources": source_names,
            "transforms": transform_names,
            "sinks": sink_names,
        },
    }

    return f"Current pipeline state and available plugins:\n{json.dumps(context, indent=2)}"


def build_messages(
    chat_history: list[dict[str, Any]],
    state: CompositionState,
    user_message: str,
    catalog: CatalogService,
    data_dir: str | None = None,
    *,
    advisor_enabled: bool = True,
    guided_terminal: TerminalState | None = None,
) -> list[dict[str, Any]]:
    """Build the full message list for the LLM.

    IMPORTANT: Returns a NEW list on every call. Never returns a cached
    or shared reference. The tool-use loop appends to this list during
    iteration; returning a cached reference would cause cross-turn
    contamination.

    Message sequence:
    1. Stable system message (core skill + optional deployment skill)
    2. Dynamic context system message (current state + plugin summary)
    3. Chat history (previous messages in this session)
    4. Current user message

    The stable prompt and dynamic context are separate system messages
    deliberately. Anthropic prompt-cache markers attach to the first
    system message, so keeping the mutating state JSON in the second
    message lets follow-up turns reuse the expensive stable skill prefix.

    When ``guided_terminal`` is set, this is the first freeform turn after
    a guided-mode exit.  The system prompt is replaced with a layered
    prompt (guided skill → transition header → freeform skill) per spec §8.2.
    The caller is responsible for the gate logic and the ``transition_consumed``
    flip; this function is pure (no state mutation).

    Args:
        chat_history: Chat history as plain dicts (role/content keys).
        state: Current CompositionState.
        user_message: The user's current message.
        catalog: CatalogService for context injection.
        data_dir: Optional data directory for deployment-specific skill
            overlay.  When provided, the deployment skill at
            ``{data_dir}/skills/pipeline_composer.md`` is appended to
            the core skill in the system prompt.
        guided_terminal: When set, the resolved TerminalState from the
            completed guided session; triggers the layered transition
            prompt instead of the freeform-only prompt.
        advisor_enabled: When False, strip advisor-specific sections from
            the core skill before forming the system prompt — the LLM
            should not learn about the ``request_advisor_hint`` tool when
            the operator has it disabled (otherwise it tries to call the
            tool and hits a "disabled" rejection, wasting a turn).

    Returns:
        A new list of message dicts for the LLM.
    """
    messages: list[dict[str, Any]] = []

    # 1. Stable system prompt only.
    # When guided_terminal is set, this is the first freeform turn after
    # a guided-mode exit — use the layered transition prompt (spec §8.2).
    # Otherwise fall through to the standard freeform-only prompt.
    # F1: route through build_system_prompt unconditionally so the
    # advisor-strip transformation applies consistently — the previous
    # ``data_dir is None → SYSTEM_PROMPT`` fast path bypassed it. The
    # @lru_cache on build_system_prompt makes repeat calls free.
    if guided_terminal is not None:
        if guided_terminal.kind is TerminalKind.COMPLETED:
            reason_str = "completed_pipeline"
        else:
            # EXITED_TO_FREEFORM — reason must be non-None for this kind
            if guided_terminal.reason is None:
                raise RuntimeError(f"EXITED_TO_FREEFORM terminal must have a reason: {guided_terminal!r}")
            reason_str = guided_terminal.reason.value
        # Thread data_dir and advisor_enabled through the transition prompt so the
        # first freeform turn after guided exit carries the same deployment overlay
        # and advisor-strip policy as all subsequent freeform turns (Codex #17).
        # build_system_prompt is @lru_cache'd — this call hits the same cache entry
        # as the non-transition branch below.
        freeform_skill = build_system_prompt(data_dir, advisor_enabled=advisor_enabled)
        prompt = build_mode_transition_system_prompt(
            terminal_reason=reason_str,
            freeform_skill=freeform_skill,
        )
    else:
        prompt = build_system_prompt(data_dir, advisor_enabled=advisor_enabled)
    messages.append({"role": "system", "content": prompt})

    # 2. Dynamic state/plugin context. Keep this outside the first
    # system message so provider prompt-cache markers cover only the
    # stable skill/deployment prefix.
    context_str = build_context_string(state, catalog)
    messages.append({"role": "system", "content": context_str})

    # 3. Chat history
    if chat_history:
        messages.extend(chat_history)

    # 4. Current user message
    messages.append({"role": "user", "content": user_message})

    return messages


def build_run_diagnostics_messages(
    snapshot: Mapping[str, object],
    data_dir: str | None = None,
) -> list[dict[str, str]]:
    """Build messages for run diagnostics explanation.

    Uses the same composer skill pack stack as normal composition so every
    composer LLM engagement carries the structure and MCP-tool guidance.
    """
    prompt = build_system_prompt(data_dir) if data_dir is not None else SYSTEM_PROMPT
    diagnostics_instructions = (
        "Run diagnostics explanation mode:\n"
        "- Explain the provided bounded run diagnostics snapshot to an operator.\n"
        "- Use only visible evidence from tokens, node states, operations, artifacts, and status.\n"
        "- Mention saved artifact paths when present.\n"
        "- If there are no Landscape records yet, say the run may still be setting up.\n"
        "- Return strict JSON only, with this exact object shape: "
        '{"headline": string, "evidence": string[], "meaning": string, "next_steps": string[]}.\n'
        "- Keep the headline and meaning plain-English and useful; avoid cute filler or vague progress claims.\n"
        "- Evidence entries must cite visible evidence from the snapshot, not hidden chain-of-thought.\n"
        "- Do not call tools, invent hidden progress, expose hidden chain-of-thought, or mention secrets."
    )
    return [
        {"role": "system", "content": prompt + "\n\n" + diagnostics_instructions},
        {"role": "user", "content": json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False)},
    ]
