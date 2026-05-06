"""System prompt and message construction for the LLM composer.

build_messages() returns a NEW list on every call — never a cached
reference. This is critical because the tool-use loop appends to the
list during iteration.

Layer: L3 (application).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.redaction import redact_source_storage_path
from elspeth.web.composer.skills import load_deployment_skill, load_skill
from elspeth.web.composer.state import CompositionState

# Load the pipeline composer skill once at module level (static content).
_PIPELINE_SKILL = load_skill("pipeline_composer")

# SYSTEM_PROMPT is the no-deployment-layer fast path.  Used directly by
# build_messages when data_dir is None, avoiding a function call.  Also
# exported for tests that need to assert identity with the core skill.
SYSTEM_PROMPT = _PIPELINE_SKILL


def _strip_advisor_content(text: str) -> str:
    r"""Remove advisor-specific content from skill text.

    When ``composer_advisor_enabled`` is False, the system prompt fed to
    the composer LLM must NOT teach it about ``request_advisor_hint``.
    Otherwise the LLM sees the tool name in the skill, attempts to call
    it, and hits the defense-in-depth "disabled" rejection — wasting a
    turn for no reason and leaving the model confused about why a tool
    the skill described doesn't actually exist.

    Removes two pieces:

    1. The ``, `request_advisor_hint``` token from the Step-0 Diagnostics
       line (leaves ``- **Diagnostics:** `explain_validation_error```).
    2. The dedicated ``#### When You Are Still Stuck — `request_advisor_hint```
       subsection — from its heading through to (but not including) the
       next ``#### `` heading.

    The transformation operates on the loaded skill text without touching
    the on-disk file, so the parity test
    ``TestComposerToolNameDrift::test_skill_step0_matches_get_tool_definitions``
    (which scans the unfiltered file content) is unaffected.
    """
    text = text.replace(", `request_advisor_hint`", "")
    start = text.find("#### When You Are Still Stuck")
    if start == -1:
        return text  # advisor section already absent; nothing to do
    end = text.find("\n#### ", start + 1)
    if end == -1:
        # Advisor subsection is the trailing subsection of its parent
        # section. Strip from the heading to the next top-level (``## ``)
        # heading, or to end-of-file if none.
        next_h2 = text.find("\n## ", start + 1)
        if next_h2 == -1:
            return text[:start].rstrip() + "\n"
        return text[:start] + text[next_h2 + 1 :]
    return text[:start] + text[end + 1 :]  # +1 to skip the leading newline


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
    core = _PIPELINE_SKILL if advisor_enabled else _strip_advisor_content(_PIPELINE_SKILL)
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
) -> list[dict[str, Any]]:
    """Build the full message list for the LLM.

    IMPORTANT: Returns a NEW list on every call. Never returns a cached
    or shared reference. The tool-use loop appends to this list during
    iteration; returning a cached reference would cause cross-turn
    contamination.

    Message sequence:
    1. System message (static prompt + injected context)
    2. Chat history (previous messages in this session)
    3. Current user message

    Args:
        chat_history: Chat history as plain dicts (role/content keys).
        state: Current CompositionState.
        user_message: The user's current message.
        catalog: CatalogService for context injection.
        data_dir: Optional data directory for deployment-specific skill
            overlay.  When provided, the deployment skill at
            ``{data_dir}/skills/pipeline_composer.md`` is appended to
            the core skill in the system prompt.
        advisor_enabled: When False, strip advisor-specific sections from
            the core skill before forming the system prompt — the LLM
            should not learn about the ``request_advisor_hint`` tool when
            the operator has it disabled (otherwise it tries to call the
            tool and hits a "disabled" rejection, wasting a turn).

    Returns:
        A new list of message dicts for the LLM.
    """
    messages: list[dict[str, Any]] = []

    # 1. System prompt with injected context (single system message)
    # F1: route through build_system_prompt unconditionally so the
    # advisor-strip transformation applies consistently — the previous
    # ``data_dir is None → SYSTEM_PROMPT`` fast path bypassed it. The
    # @lru_cache on build_system_prompt makes repeat calls free.
    prompt = build_system_prompt(data_dir, advisor_enabled=advisor_enabled)
    context_str = build_context_string(state, catalog)
    messages.append({"role": "system", "content": prompt + "\n\n" + context_str})

    # 2. Chat history
    if chat_history:
        messages.extend(chat_history)

    # 3. Current user message
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
