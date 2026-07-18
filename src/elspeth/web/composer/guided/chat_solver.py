"""Per-step chat solver: invoke LLM with step-scoped skill briefing.

Most steps remain advisory prose: the solver receives the base preamble + the
playbook for the user's current wizard step, plus the user's typed message, and
replies with prose. Step 1's schema-form chat also has a narrow source/data
schema tool palette so a complete source request can materialise data instead
of stalling as prose.

Audit: when supplied a ``ComposerLLMCallRecorder``, both LLM call sites in this
module append one ``ComposerLLMCall`` row per provider request. The route
handler (``post_guided_chat``) is responsible for draining the recorder after
it persists any guided-session state changes.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, cast

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.composer_progress import ComposerProgressSink
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.contracts.trust_boundary import trust_boundary
from elspeth.web.blobs.protocol import ALLOWED_MIME_TYPES, AllowedMimeType
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided._discovery import _assistant_tool_calls_message, _execute_discovery_call
from elspeth.web.composer.guided.errors import GuidedSolverResponseShapeError, InvariantError
from elspeth.web.composer.guided.prompts import _summarize_sample_row, load_step_chat_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import (
    GUIDED_JSON_MAX_ITEMS,
    GuidedJsonBudget,
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
    freeze_guided_json_mapping,
    freeze_guided_str_sequence,
)
from elspeth.web.composer.llm_response_parsing import (
    apply_anthropic_cache_markers,
    attach_llm_calls,
    build_llm_call_record,
    supports_anthropic_prompt_cache_markers,
)
from elspeth.web.composer.progress import emit_progress, model_call_progress_event, tool_batch_progress_event
from elspeth.web.composer.service import _litellm_acompletion
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._dispatch import get_discovery_tool_definitions
from elspeth.web.interpretation_state import SOURCE_AUTHORING_KEY
from elspeth.web.plugin_policy.models import PluginAvailabilitySnapshot

# Server-owned source-option keys that the LLM must NEVER author. Both are
# stamped authoritatively at proposal settlement (including ``blob_ref``),
# ``source_authoring`` by ``set_source_from_blob`` for LLM-authored/dynamic
# sources) and REJECTED by ``set_source`` if caller-supplied. On an in-place
# re-resolve the committed source is threaded into the resolver prompt, so the
# model parrots these keys straight back; left in, the next Send 400s with
# "Step 1 source commit failed". Mirrors ``_WEB_ONLY_SOURCE_KEYS`` in
# ``composer/tools/_common.py`` (the commit-side stripper for prevalidation).
_RESOLVER_FORBIDDEN_SOURCE_OPTION_KEYS: Final[frozenset[str]] = frozenset({"blob_ref", SOURCE_AUTHORING_KEY})

# Register guard for the user-facing chat message. Models occasionally dump
# their internal agentic scratchpad — pseudo tool-call transcripts in
# ``<tool_call>``/``<tool_response>`` tags — INTO the assistant_message tool
# argument (observed live 2026-07-03: a 2.8KB replay of an invented
# list_sources/build_source loop persisted verbatim into a tutorial chat
# history and rendered as the learner-facing reply). assistant_message is a
# Tier-3 boundary: reject the register violation loudly (routes to
# MALFORMED_RESPONSE → advisory; the user's Send is retryable) rather than
# persisting scratchpad as conversation.
_TOOL_SCAFFOLD_MARKERS: Final[tuple[str, ...]] = (
    "<tool_call",
    "</tool_call",
    "<tool_response",
    "</tool_response",
)

# ``solve_step_chat`` never attaches tools (Phase A advisory-only), but its
# system prompt is ``load_step_chat_skill(step)`` — the SAME per-step skill
# the tool-equipped resolve calls use, and ``base.md`` unconditionally frames
# the model as a tool-caller ("you build the pipeline by calling tools",
# `list_sources`/`get_plugin_schema` lookups). A model primed by that framing
# with no tools on the wire has nothing real to call, so it narrates one as
# literal text instead — the scaffold leak ``_require_prose_assistant_message``
# then correctly rejects. This addendum overrides the framing for THIS call
# only (a fresh system message, not a ``load_step_chat_skill`` edit — the
# resolve calls legitimately keep tool access and must not see this).
_ADVISORY_NO_TOOLS_ADDENDUM: Final[str] = (
    "## No tools in this reply\n\n"
    "You have NO tools available for this reply — not `resolve_source`, not "
    "`list_sources`/`list_sinks`/`list_transforms`/`get_plugin_schema`, nothing. "
    "Answer in plain prose only. Never write tool-call syntax, XML-style "
    "scaffolding (`<tool_call>`, `<tool_response>`), or any text that narrates "
    "invoking a tool — even to describe your reasoning. If the user's message "
    "needs an action you can't take from here (for example, they described "
    "data without giving you the actual rows), say so plainly and ask for "
    "what is missing. Do not ask the user to re-send the same message, say "
    "`go ahead`, or wait for a tool-enabled version of this reply; this path "
    "will remain advisory. If the wizard controls can complete the action, "
    "point to those controls plainly.\n"
)

_STEP_1_FALSE_TOOL_DECLINE_REPLY_MARKERS: Final[tuple[str, ...]] = (
    "don't have my tools",
    "do not have my tools",
    "no tools available",
    "tools available in this reply",
    "tool-enabled",
)

_STEP_1_FALSE_TOOL_DECLINE_RESEND_MARKERS: Final[tuple[str, ...]] = (
    "re-send",
    "resend",
    "send your message",
    "send it again",
    "say 'go ahead'",
    'say "go ahead"',
    "say go ahead",
    "just say go ahead",
)

_STEP_1_NONEXISTENT_INLINE_CONTROL_MARKERS: Final[tuple[str, ...]] = (
    "inline json source option",
    "inline json option",
    "inline source option",
    "paste the rows there",
)

_STEP_1_SOURCE_ACTIONABLE_USER_MARKERS: Final[tuple[str, ...]] = (
    "csv",
    "json",
    "inline",
    "path",
    "file",
    "headers",
    "header",
    "columns",
    "column",
    "rows",
    "row",
    "url",
    "schema",
    "source",
    "invalid",
    "discard",
    "quarantine",
)

_STEP_1_SOURCE_FALSE_DECLINE_RETRY_ADDENDUM: Final[str] = (
    "## Retry after false tool-decline\n\n"
    "Your previous reply said you had no tools and asked the user to re-send, "
    "but this request DOES include the `resolve_source` tool. The user's "
    "message contains source-building details. Do not ask the user to re-send "
    "or say `go ahead`; either call `resolve_source` now, or explain the "
    "specific missing source data in plain prose."
)

_STEP_1_SOURCE_INLINE_CONTROL_RETRY_ADDENDUM: Final[str] = (
    "## Retry after nonexistent inline-source control advice\n\n"
    "Your previous reply told the user to choose an inline JSON/source wizard "
    "control, but this wizard does not expose that control. This request DOES "
    "include the `resolve_source` tool. If the user supplied rows or enough "
    "source content, call `resolve_source` now and include that content. If "
    "data is missing, ask for the specific missing rows or file information; "
    "do not point the user at inline JSON/source controls."
)


class AssistantScaffoldLeakError(ValueError):
    """The model leaked tool-call scaffolding into a user-facing message.

    A ``ValueError`` subclass so the step-1/step-2 resolve wrappers' existing
    ``ValueError`` absorption (synthetic-unavailable fallback) is unchanged.
    The advisory wrapper (``solve_step_chat_with_auto_drop``) catches THIS
    class specifically — a bare ``ValueError`` there still signals a caller
    bug and propagates. Observed live twice (tutorial resolve_source
    2026-07-03, live-guided advisory reply 2026-07-03): the model writes a
    pseudo agentic transcript as literal text, which persists verbatim into
    chat_history and renders as the user-facing reply.
    """


def _require_prose_assistant_message(value: object, *, tool: str) -> str:
    """Validate an LLM-supplied assistant_message is user-facing prose."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool} assistant_message must be a non-empty string")
    lowered = value.lower()
    for marker in _TOOL_SCAFFOLD_MARKERS:
        if marker in lowered:
            raise AssistantScaffoldLeakError(
                f"{tool} assistant_message must be user-facing prose; it contains raw "
                f"tool-call scaffolding ({marker!r}) — the model leaked its internal "
                "transcript into the chat message"
            )
    return value


def _step_1_user_message_has_source_action_signal(user_message: str, *, current_source: SourceResolved | None) -> bool:
    lowered = user_message.lower()
    if any(marker in lowered for marker in _STEP_1_SOURCE_ACTIONABLE_USER_MARKERS):
        return True
    # A terse revision like "same again" can still be actionable after a source
    # exists, but only retry the no-tools loop when it at least reads like a
    # command rather than a general question.
    return current_source is not None and any(
        marker in lowered
        for marker in (
            "same again",
            "use this",
            "make it",
            "change it",
            "update it",
            "replace it",
        )
    )


def _should_retry_step_1_source_false_tool_decline(
    *,
    user_message: str,
    prose_reply: str,
    current_source: SourceResolved | None,
) -> bool:
    """Detect the observed Step-1 loop where a tool-equipped call denies tools."""
    lowered_reply = prose_reply.lower()
    if not any(marker in lowered_reply for marker in _STEP_1_FALSE_TOOL_DECLINE_REPLY_MARKERS):
        return False
    if not any(marker in lowered_reply for marker in _STEP_1_FALSE_TOOL_DECLINE_RESEND_MARKERS):
        return False
    return _step_1_user_message_has_source_action_signal(user_message, current_source=current_source)


def _should_retry_step_1_source_nonexistent_control_advice(
    *,
    user_message: str,
    prose_reply: str,
    current_source: SourceResolved | None,
) -> bool:
    lowered_reply = prose_reply.lower()
    if not any(marker in lowered_reply for marker in _STEP_1_NONEXISTENT_INLINE_CONTROL_MARKERS):
        return False
    return _step_1_user_message_has_source_action_signal(user_message, current_source=current_source)


@dataclass(frozen=True, slots=True)
class Step1SourceChatResolution:
    """A Step-1 chat tool call that can be committed as a source.

    All fields originate from the LLM response and are therefore validated at
    this boundary before the route handler writes a blob or mutates guided
    state.
    """

    assistant_message: str
    plugin: str
    filename: str
    mime_type: AllowedMimeType
    content: str
    options: Mapping[str, Any]
    observed_columns: tuple[str, ...]
    sample_rows: tuple[Mapping[str, Any], ...]
    on_validation_failure: str

    def __post_init__(self) -> None:
        if type(self.sample_rows) is not tuple:
            raise TypeError("Step1SourceChatResolution.sample_rows must be an exact tuple")
        if len(self.sample_rows) > GUIDED_JSON_MAX_ITEMS:
            raise InvariantError(f"Step1SourceChatResolution.sample_rows exceeds the {GUIDED_JSON_MAX_ITEMS}-item limit")
        budget = GuidedJsonBudget()
        object.__setattr__(self, "options", freeze_guided_json_mapping(self.options, "Step1SourceChatResolution.options", budget=budget))
        object.__setattr__(
            self,
            "sample_rows",
            tuple(
                freeze_guided_json_mapping(row, f"Step1SourceChatResolution.sample_rows[{index}]", budget=budget)
                for index, row in enumerate(self.sample_rows)
            ),
        )
        object.__setattr__(
            self,
            "observed_columns",
            freeze_guided_str_sequence(
                self.observed_columns,
                "Step1SourceChatResolution.observed_columns",
                budget=budget,
            ),
        )
        freeze_fields(self, "options", "sample_rows", "observed_columns")


@dataclass(frozen=True, slots=True)
class Step1SourceChatOutcome:
    """Outcome of one ``resolve_source``-equipped LLM call.

    At most one of the two fields is populated. ``resolution`` is a valid
    ``resolve_source`` tool call. ``prose_reply`` is the model's own
    register-guarded plain-language answer when it declined the tool — not
    enough detail to act, or a plain question — captured directly from THIS
    call so the caller never needs a second, tool-less call to obtain an
    answer to show the user (that second call previously reused a
    tool-capable system prompt with no tools attached, which is what caused
    the model to hallucinate ``<tool_call>`` scaffolding — see
    ``_ADVISORY_NO_TOOLS_ADDENDUM``).

    Both fields are ``None`` only when the model's reply had neither a tool
    call nor any content — a genuinely defective/empty response. That rare
    case is deliberately NOT raised here: the caller falls back to the
    tool-less advisory call exactly as before, which already raises
    ``InvariantError`` on empty content.
    """

    resolution: Step1SourceChatResolution | None
    prose_reply: str | None


_STEP_1_SOURCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "resolve_source",
        "description": (
            "Use when the Step 1 chat message contains enough information to create "
            "or bind the source data and schema. Do not use for general advice."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "resolution",
                "plugin",
                "filename",
                "mime_type",
                "content",
                "options",
                "observed_columns",
                "sample_rows",
                "assistant_message",
            ],
            "properties": {
                "resolution": {"type": "string", "enum": ["source"]},
                "plugin": {"type": "string", "minLength": 1},
                "filename": {"type": "string", "minLength": 1},
                "mime_type": {"type": "string", "enum": sorted(ALLOWED_MIME_TYPES)},
                "content": {"type": "string", "minLength": 1},
                "options": {
                    "type": "object",
                    "description": (
                        "Source plugin options. IMPORTANT: when the rows are guaranteed to carry "
                        "specific columns — a column the operator named (e.g. `url`), or columns you "
                        "authored into every row of inline `content` — declare them as a contract: set "
                        '`schema` to `{"mode": "observed", "guaranteed_fields": [<those exact column '
                        "names>]}`. This records the columns the rows are guaranteed to contain so a "
                        "downstream transform that reads one of them (e.g. web_scrape reading `url`) "
                        "wires cleanly at the wiring step; an observed source with no `guaranteed_fields` "
                        "promises nothing and fails that contract. Keep `mode` `observed` so any other "
                        "columns still pass through."
                    ),
                },
                "observed_columns": {"type": "array", "items": {"type": "string"}},
                "sample_rows": {"type": "array", "items": {"type": "object"}},
                "assistant_message": {"type": "string", "minLength": 1},
                # Optional (absent from ``required``): the parser defaults it to
                # "discard" so a passive walk never stalls. Listed here so the
                # model is allowed to send it under ``additionalProperties: false``.
                "on_validation_failure": {
                    "type": "string",
                    "description": (
                        "Where rows that fail the source's schema validation are routed: a "
                        "configured sink name, or 'discard' to drop them. For a synthetic/valid-by-"
                        "construction demo source, 'discard' is correct."
                    ),
                },
            },
        },
    },
}


def _record_llm_call(
    *,
    recorder: BufferingRecorder | None,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    status: ComposerLLMCallStatus | None,
    started_at: datetime,
    started_ns: int,
    temperature: float | None,
    seed: int | None,
    response: Any | None,
    error_class: str | None,
    error_message: str | None,
) -> None:
    if recorder is None or status is None:
        return
    recorder.record_llm_call(
        build_llm_call_record(
            model_requested=model,
            messages=messages,
            tools=tools,
            status=status,
            started_at=started_at,
            started_ns=started_ns,
            temperature=temperature,
            seed=seed,
            response=response,
            error_class=error_class,
            error_message=error_message,
        )
    )
    current_exc = sys.exc_info()[1]
    if current_exc is not None:
        attach_llm_calls(current_exc, recorder)


def _build_step_1_source_dynamic_block(
    *,
    plugin_hint: str | None,
    current_source: SourceResolved | None = None,
) -> str:
    """Compose the DYNAMIC Step-1 source block (hint + revise context + tool instructions).

    Split out of the per-step skill so the stable ``load_step_chat_skill(STEP_1_SOURCE)``
    can be an isolable, byte-stable, markable cache head (``messages[0]``); this
    dynamic block rides in ``messages[1]``. The static tool-instructions tail is
    intentionally part of THIS block (after the dynamic hint/revise content),
    not the marked head — only the ~1199-token skill is in the cached prefix.
    """
    hint = (
        f"The current source plugin selected in the wizard is {plugin_hint!r}."
        if plugin_hint is not None
        else "The current source plugin is not persisted in server state; infer only when the chat message or tool context makes it explicit."
    )
    revise_block = ""
    if current_source is not None:
        revise_block = (
            "\n## Current applied source (revise relative to this)\n\n"
            "A source has already been applied to this phase. The user's message "
            "is a REVISION instruction against it — re-emit the COMPLETE updated "
            "source (not a diff). Current source:\n"
            f"{json.dumps(_source_revision_context_for_llm(current_source), sort_keys=True)}\n"
        )
    return (
        "## Step 1 Source/Data Schema Tool\n\n"
        f"{hint}\n"
        f"{revise_block}"
        "If the user's message provides enough information to create inline source data, "
        "call `resolve_source` with the complete file content, the source plugin, "
        "schema options, observed columns, representative sample rows, and a brief "
        "assistant_message. Whenever the message states the rows carry a specific "
        "named column (a `url`, an id, a key), set the source `schema` to "
        '`{"mode": "observed", "guaranteed_fields": [<those exact column names>]}` — '
        "you are RECORDING the columns the operator told you the rows contain, so a "
        "later transform that reads one of them wires cleanly at the wiring step. "
        "Keep `mode` `observed` so any other columns still pass through. "
        "For CSV data, include a header row in `content` and set "
        "`mime_type` to `text/csv`. When the user wants to FETCH one or more remote "
        "document files from URL(s), make the source an INLINE `json` or `csv` manifest "
        "whose rows carry each URL, then add downstream transform nodes: `blob_fetch` "
        "to create blob refs and a parser transform such as `blob_csv_expand` to expand "
        "the blob into rows. Do not choose a fetcher as the source plugin. When the user "
        "wants to SCRAPE one or more webpages into row content, the source is an INLINE "
        "`json` (or `csv`) dataset whose rows carry each "
        "URL in a `url` column — e.g. json `content` of "
        '`[{"url": "https://example/a"}, {"url": "https://example/b"}]`. Declare that '
        "`url` column as guaranteed on the source `schema` "
        '(`{"mode": "observed", "guaranteed_fields": ["url"]}`) so the downstream '
        "web_scrape transform's required `url` input is satisfied at the wiring step "
        "(an observed-mode source that guarantees nothing fails that contract). You "
        "must NOT choose a `web_scraper`/`web_scrape` source: scraping pages is a "
        "downstream TRANSFORM applied later, never a source plugin. The valid "
        "source plugins are `aws_s3`, `azure_blob`, `csv`, `dataverse`, `json`, `null`, and `text`. "
        "For `aws_s3`, `endpoint_url` is CLI/batch-only; web authors must never set it "
        "because every non-null web-authored value is rejected. "
        "Preserve user-supplied values exactly in the file "
        "content; do not invent hidden pipeline transforms. Also set `on_validation_failure` "
        "when you resolve a source: use `discard` for a demo source that is valid by "
        "construction, or the name of a quarantine sink for production data whose invalid "
        "rows must be kept for inspection. If the message is only a "
        "question or lacks enough source detail, reply in prose and do not call a tool.\n"
    )


@trust_boundary(
    tier=3,
    source="web-authored source schema option value (untrusted mapping)",
    source_param="schema",
    suppresses=("R1", "R5"),
    invariant=(
        "returns None for a non-mapping schema; extracts only string mode and string-list "
        "guaranteed_fields; malformed members are dropped, never raised on"
    ),
    non_raising=True,
)
def _llm_safe_schema_option(schema: Any) -> dict[str, Any] | None:
    if not isinstance(schema, Mapping):
        return None
    safe: dict[str, Any] = {}
    mode = schema.get("mode")
    if isinstance(mode, str):
        safe["mode"] = mode
    guaranteed_fields = schema.get("guaranteed_fields")
    if isinstance(guaranteed_fields, (list, tuple)):
        safe_guaranteed_fields = [field for field in guaranteed_fields if isinstance(field, str)]
        if safe_guaranteed_fields:
            safe["guaranteed_fields"] = safe_guaranteed_fields
    return safe or {"shape": "object"}


@trust_boundary(
    tier=3,
    source="committed SourceResolved carrying web-authored options (untrusted mapping values)",
    source_param="current_source",
    suppresses=("R1", "R5"),
    invariant=(
        "builds the LLM revision-context payload from well-formed option values only; "
        "non-mapping options degrade to empty, malformed rows/schema are dropped, never raised on"
    ),
    non_raising=True,
)
def _source_revision_context_for_llm(current_source: SourceResolved) -> dict[str, Any]:
    options = current_source.options if isinstance(current_source.options, Mapping) else {}
    payload: dict[str, Any] = {
        "plugin": current_source.plugin,
        "observed_columns": list(current_source.observed_columns),
        "sample_rows": [_summarize_sample_row(row) for row in current_source.sample_rows if isinstance(row, Mapping)],
        "on_validation_failure": current_source.on_validation_failure,
        "option_count": len(options),
    }
    schema = _llm_safe_schema_option(options.get("schema"))
    if schema is not None:
        payload["schema"] = schema
    if "blob_ref" in options:
        payload["server_storage_bound"] = True
    return payload


def _sink_revision_context_for_llm(current_sink: SinkResolved) -> dict[str, Any]:
    try:
        (output,) = current_sink.outputs
    except ValueError as exc:
        raise InvariantError("Step 2 chat requires exactly one current output") from exc
    options = output.options if isinstance(output.options, Mapping) else {}
    return {
        "output": {
            "plugin": output.plugin,
            "required_fields": list(output.required_fields),
            "schema_mode": output.schema_mode,
            "option_count": len(options),
        }
    }


def build_step_chat_context_block(
    *,
    step: GuidedStep,
    current_source: SourceResolved | None,
    current_sink: SinkResolved | None,
    state: CompositionState | None,
) -> str:
    """Compose the LLM-safe "current build" block for the advisory chat path.

    The advisory solver previously saw only the step playbook + the user's
    message, so "explain what I'm seeing" questions could only be answered
    generically. This block names the applied artifacts via the SAME LLM-safe
    serializers the revision prompts use (plugin names, schema modes, field
    lists, counts — never raw options, blob paths, or secret-bearing values)
    plus a plugins-only pipeline sketch from the composition state.

    Rides as a SECOND system message in ``solve_step_chat`` — the stable
    per-step skill stays the byte-stable, cache-markable head (the same split
    the step-1 resolve path uses for its dynamic block).
    """
    lines: list[str] = [
        "## Current build (what the user is looking at)",
        "",
        f"The user is on wizard step {step.value}. When they ask what they are "
        "seeing or why, explain from THIS build context: name the concrete "
        "plugins and settings below, why they fit what the user asked for, and "
        "what each setting means in plain language. Do not invent settings that "
        "are not listed here.",
        "",
    ]
    if current_source is not None:
        lines.append(f"Applied source: {json.dumps(_source_revision_context_for_llm(current_source), sort_keys=True)}")
    else:
        lines.append("Applied source: none yet.")
    if current_sink is not None:
        lines.append(f"Applied output: {json.dumps(_sink_revision_context_for_llm(current_sink), sort_keys=True)}")
    else:
        lines.append("Applied output: none yet.")
    if state is not None:
        source_plugins = sorted({spec.plugin for spec in state.sources.values()})
        node_plugins = [node.plugin if node.plugin is not None else "(gate/coalesce)" for node in state.nodes]
        output_plugins = [output.plugin for output in state.outputs]
        lines.append(
            "Pipeline so far: "
            f"sources={json.dumps(source_plugins)}, "
            f"transform_nodes={json.dumps(node_plugins)}, "
            f"outputs={json.dumps(output_plugins)}, "
            f"edge_count={len(state.edges)}."
        )
    return "\n".join(lines) + "\n"


@trust_boundary(
    tier=3,
    source="LLM-emitted resolve_source tool-call arguments (untrusted model output JSON)",
    source_param="arguments",
    suppresses=("R1", "R5"),
    invariant=(
        "raises ValueError on non-object decode, missing keys, mistyped fields, or "
        "scaffold-leaking assistant_message; options, sample rows, and observed columns "
        "must satisfy strict depth, item, aggregate text, UTF-8, and finite-JSON bounds"
    ),
    test_ref=(
        "tests/unit/web/composer/guided/test_chat_solver.py::test_parse_step_1_source_translates_strict_snapshot_failures_to_malformed"
    ),
    test_fingerprint="880bf7f1287428d74961b7678b23c597adcb9b26660123eaf14cbb02dc4f6792",
)
def _parse_step_1_source_tool_arguments(arguments: str, *, plugin_hint: str | None) -> Step1SourceChatResolution:
    """Validate the resolve_source tool arguments from a LiteLLM response."""
    data = json.loads(arguments)
    if not isinstance(data, Mapping):
        raise ValueError(f"resolve_source arguments must decode to an object; got {type(data).__name__}")

    missing = {
        "resolution",
        "plugin",
        "filename",
        "mime_type",
        "content",
        "options",
        "observed_columns",
        "sample_rows",
        "assistant_message",
    } - set(data.keys())
    if missing:
        raise ValueError(f"resolve_source arguments missing required keys: {sorted(missing)}")
    if data["resolution"] != "source":
        raise ValueError(f"resolve_source resolution must be 'source'; got {data['resolution']!r}")

    plugin = data["plugin"]
    if not isinstance(plugin, str) or not plugin:
        raise ValueError(f"resolve_source plugin must be a non-empty string; got {plugin!r}")
    if plugin_hint is not None and plugin != plugin_hint:
        raise ValueError(f"resolve_source plugin {plugin!r} does not match current Step 1 plugin {plugin_hint!r}")

    filename = data["filename"]
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"resolve_source filename must be a non-empty string; got {filename!r}")

    mime_type = data["mime_type"]
    if not isinstance(mime_type, str) or mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(f"resolve_source mime_type must be one of {sorted(ALLOWED_MIME_TYPES)}; got {mime_type!r}")

    content = data["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("resolve_source content must be a non-empty string")

    options = data["options"]
    if not isinstance(options, Mapping):
        raise ValueError(f"resolve_source options must be an object; got {type(options).__name__}")
    # Drop SERVER-OWNED keys the model may have parroted back from the threaded
    # current_source (see _RESOLVER_FORBIDDEN_SOURCE_OPTION_KEYS). Their absence
    # is always correct here — they are re-stamped authoritatively at commit, and
    # set_source rejects them as caller-supplied. ``interpretation_requirements``
    # is intentionally NOT stripped: the resolver legitimately stages *pending*
    # review requirements for invented sources.
    options = {key: value for key, value in dict(options).items() if key not in _RESOLVER_FORBIDDEN_SOURCE_OPTION_KEYS}

    observed_columns_raw = data["observed_columns"]
    if not isinstance(observed_columns_raw, list):
        raise ValueError(f"resolve_source observed_columns must be a list; got {type(observed_columns_raw).__name__}")
    observed_columns: list[str] = []
    for idx, column in enumerate(observed_columns_raw):
        if not isinstance(column, str) or not column:
            raise ValueError(f"resolve_source observed_columns[{idx}] must be a non-empty string; got {column!r}")
        observed_columns.append(column)

    sample_rows_raw = data["sample_rows"]
    if not isinstance(sample_rows_raw, list):
        raise ValueError(f"resolve_source sample_rows must be a list; got {type(sample_rows_raw).__name__}")
    sample_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(sample_rows_raw):
        if not isinstance(row, Mapping):
            raise ValueError(f"resolve_source sample_rows[{idx}] must be an object; got {type(row).__name__}")
        sample_rows.append(dict(row))

    assistant_message = _require_prose_assistant_message(data["assistant_message"], tool="resolve_source")

    # on_validation_failure is OPTIONAL (not in the required set / tool schema).
    # The composer sets it most of the time, but a passive walk must never stall,
    # so absent / None / empty defaults to "discard". When the model DOES send it,
    # require a non-empty string at this Tier-3 boundary.
    on_validation_failure_raw = data.get("on_validation_failure")
    if on_validation_failure_raw is None or (isinstance(on_validation_failure_raw, str) and not on_validation_failure_raw):
        on_validation_failure = "discard"
    elif not isinstance(on_validation_failure_raw, str):
        raise ValueError(
            f"resolve_source on_validation_failure must be a string when provided; got {type(on_validation_failure_raw).__name__}"
        )
    else:
        on_validation_failure = on_validation_failure_raw

    try:
        return Step1SourceChatResolution(
            assistant_message=assistant_message,
            plugin=plugin,
            filename=filename,
            mime_type=cast(AllowedMimeType, mime_type),
            content=content,
            options=dict(options),
            observed_columns=tuple(observed_columns),
            sample_rows=tuple(sample_rows),
            on_validation_failure=on_validation_failure,
        )
    except (InvariantError, TypeError) as exc:
        raise ValueError("resolve_source snapshot is malformed") from exc


async def _bounded_acompletion(kwargs: dict[str, Any], timeout_seconds: float) -> Any:
    """Run ``_litellm_acompletion`` under an ``asyncio.wait_for`` bound.

    Every call supplies the current composer timeout. Invalid bounds fail at
    this seam rather than silently creating an unbounded provider request.
    """
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int | float):
        raise TypeError("timeout_seconds must be a finite positive number")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a finite positive number")
    return await asyncio.wait_for(_litellm_acompletion(**kwargs), timeout=timeout_seconds)


async def maybe_resolve_step_1_source_chat(
    *,
    model: str,
    user_message: str,
    plugin_hint: str | None,
    current_source: SourceResolved | None = None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
    timeout_seconds: float,
    context_block: str | None = None,
) -> Step1SourceChatOutcome:
    """Try to resolve a Step-1 schema-form chat message into source data.

    Returns a :class:`Step1SourceChatOutcome`. ``resolution`` is set on a
    ``resolve_source`` tool call. When the model instead replies in ordinary
    prose, ``prose_reply`` carries that (register-guarded) reply so the
    caller can show it directly without a second, tool-less call — both
    fields are ``None`` only on a genuinely empty/defective response, in
    which case the caller falls back to the advisory chat path exactly as
    before.

    When ``current_source`` is supplied the tool prompt includes the current
    applied source so a revision instruction ("add a url column", "make it
    csv not json") resolves relative to it.

    ``context_block`` (:func:`build_step_chat_context_block`) rides as an
    extra, unmarked system message so a declined-to-prose reply (e.g.
    "explain what I'm seeing") is grounded in the same "current build"
    context the tool-less advisory call would otherwise have supplied —
    parity that keeps the salvaged prose no worse than a second call's.
    """
    if not user_message:
        raise InvariantError("maybe_resolve_step_1_source_chat: user_message is empty (route validation gap)")

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    retry_addendum: str | None = None
    for attempt_index in range(2):
        # SPLIT the system prompt: the stable per-step skill is the byte-stable,
        # markable head (messages[0]); the dynamic hint/revise context + tool
        # instructions ride in messages[1]. Only the ~1199-token skill is in the
        # marked cache prefix.
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()},
            {
                "role": "system",
                "content": _build_step_1_source_dynamic_block(
                    plugin_hint=plugin_hint,
                    current_source=current_source,
                ),
            },
        ]
        if context_block is not None:
            messages.append({"role": "system", "content": context_block})
        if retry_addendum is not None:
            messages.append({"role": "system", "content": retry_addendum})
        messages.append({"role": "user", "content": user_message})
        tools = [_STEP_1_SOURCE_TOOL]
        # Mark BEFORE kwargs so the SAME marked objects feed both the wire call and
        # the audit record (messages / tools below, read in the finally block).
        # Gated on THIS call's model.
        if supports_anthropic_prompt_cache_markers(model):
            messages, marked_tools = apply_anthropic_cache_markers(messages, tools)
            if marked_tools is not None:
                tools = marked_tools
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        try:
            response = await _bounded_acompletion(kwargs, timeout_seconds)

            message = response.choices[0].message
            tool_calls = message.tool_calls or ()
            for tool_call in tool_calls:
                function = tool_call.function
                if function is None:
                    continue
                if function.name != "resolve_source":
                    continue
                arguments = function.arguments
                if not isinstance(arguments, str):
                    raise ValueError(f"resolve_source function.arguments must be a JSON string; got {type(arguments).__name__}")
                result = _parse_step_1_source_tool_arguments(arguments, plugin_hint=plugin_hint)
                status = ComposerLLMCallStatus.SUCCESS
                return Step1SourceChatOutcome(resolution=result, prose_reply=None)
            # No resolve_source call: the model judged the message doesn't carry
            # enough detail to act (or it's a plain question) and answered in
            # prose instead. Validate + return that prose directly — the SAME
            # register guard the tool argument gets — so the caller never needs
            # a second, tool-less call to obtain an answer to show the user.
            # Deliberately gated on ``not tool_calls`` (mirrors the step-2 sink
            # salvage): a response that ALSO carries a hallucinated tool call is a
            # more suspicious shape — its prose narrates an action that never ran —
            # and must not be trusted; it falls through to the advisory fallback
            # (now grounded by _ADVISORY_NO_TOOLS_ADDENDUM) exactly as before.
            if not tool_calls:
                content = message.content
                if content is None or not str(content).strip():
                    # Genuinely empty/defective response (no tool call, no
                    # content): both fields None — the caller falls back to the
                    # advisory chat path exactly as before.
                    status = ComposerLLMCallStatus.SUCCESS
                    return Step1SourceChatOutcome(resolution=None, prose_reply=None)
                prose = _require_prose_assistant_message(str(content), tool="maybe_resolve_step_1_source_chat")
                if attempt_index == 0 and _should_retry_step_1_source_false_tool_decline(
                    user_message=user_message,
                    prose_reply=prose,
                    current_source=current_source,
                ):
                    status = ComposerLLMCallStatus.SUCCESS
                    retry_addendum = _STEP_1_SOURCE_FALSE_DECLINE_RETRY_ADDENDUM
                    continue
                if attempt_index == 0 and _should_retry_step_1_source_nonexistent_control_advice(
                    user_message=user_message,
                    prose_reply=prose,
                    current_source=current_source,
                ):
                    status = ComposerLLMCallStatus.SUCCESS
                    retry_addendum = _STEP_1_SOURCE_INLINE_CONTROL_RETRY_ADDENDUM
                    continue
                status = ComposerLLMCallStatus.SUCCESS
                return Step1SourceChatOutcome(resolution=None, prose_reply=prose)

            # Non-empty tool_calls with no resolve_source (hallucinated tool name
            # or function=None): return the empty outcome so the route falls back
            # to the tool-less advisory call, matching the step-2 contract.
            status = ComposerLLMCallStatus.SUCCESS
            return Step1SourceChatOutcome(resolution=None, prose_reply=None)
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMBadRequestError as exc:
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except (IndexError, AttributeError, json.JSONDecodeError, ValueError) as exc:
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            error_class = type(exc).__name__
            error_message = "malformed_response"
            raise
        except Exception as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        finally:
            _record_llm_call(
                recorder=recorder,
                model=model,
                messages=messages,
                tools=tools,
                status=status,
                started_at=started_at,
                started_ns=started_ns,
                temperature=temperature,
                seed=seed,
                response=response,
                error_class=error_class,
                error_message=error_message,
            )

    raise InvariantError("maybe_resolve_step_1_source_chat: retry loop exhausted without returning")


_STEP_2_SINK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "resolve_sink",
        "description": (
            "Use when the Step 2 chat message contains enough information to configure the pipeline output. Do not use for general advice."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["resolution", "output", "assistant_message"],
            "properties": {
                "resolution": {"type": "string", "enum": ["sink"]},
                "output": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "plugin", "options", "required_fields", "schema_mode", "on_write_failure"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "plugin": {"type": "string", "minLength": 1},
                        # Bare object: option shape varies by sink plugin.
                        # Validated by the canonical proposal candidate.
                        "options": {"type": "object"},
                        "required_fields": {"type": "array", "items": {"type": "string"}},
                        "schema_mode": {"type": "string", "enum": ["fixed", "flexible", "observed"]},
                        "on_write_failure": {"type": "string", "minLength": 1},
                    },
                },
                "assistant_message": {"type": "string", "minLength": 1},
            },
        },
    },
}


def _build_step_2_sink_tool_prompt(*, current_sink: SinkResolved | None) -> str:
    """Compose the Step-2 sink tool prompt."""
    revise_block = ""
    if current_sink is not None:
        revise_block = (
            "\n## Current applied sink (revise relative to this)\n\n"
            "A sink has already been applied. The user's message is a REVISION "
            "instruction against it — re-emit the COMPLETE updated output (not a "
            "diff). Current sink:\n"
            f"{json.dumps(_sink_revision_context_for_llm(current_sink), sort_keys=True)}\n"
        )
    return (
        f"{load_step_chat_skill(GuidedStep.STEP_2_SINK).rstrip()}\n\n"
        "## Step 2 Sink Tool\n\n"
        f"{revise_block}"
        "If the user's message provides enough information to configure the "
        "pipeline output, call `resolve_sink` with the complete output "
        "(name, plugin, options, required_fields, schema_mode, "
        "on_write_failure) and a brief "
        "assistant_message. If the message is only a question or lacks enough "
        "detail, reply in prose and do not call a tool.\n"
    )


@trust_boundary(
    tier=3,
    source="LLM-emitted resolve_sink tool-call arguments (untrusted model output JSON)",
    source_param="arguments",
    suppresses=("R1", "R5"),
    invariant=(
        "raises ValueError on non-object decode, missing keys, mistyped output entries, "
        "or strict snapshot depth/item/aggregate text/UTF-8/finite-JSON "
        "violations; never coerces malformed model output"
    ),
    test_ref=(
        "tests/unit/web/composer/guided/test_chat_solver.py::test_parse_step_2_sink_translates_strict_snapshot_failures_to_malformed"
    ),
    test_fingerprint="283f5a4c664af76b2cc2aa111d84d276e17bbbf25e61a1cbec2ce10a39ff7237",
)
def _parse_step_2_sink_tool_arguments(arguments: str) -> tuple[SinkResolved, str]:
    """Validate the resolve_sink tool arguments. Returns (sink, assistant_message)."""
    data = json.loads(arguments)
    if not isinstance(data, Mapping):
        raise ValueError(f"resolve_sink arguments must decode to an object; got {type(data).__name__}")
    expected_top = {"resolution", "output", "assistant_message"}
    if set(data) != expected_top:
        raise ValueError(f"resolve_sink arguments must contain exactly {sorted(expected_top)}")
    if data["resolution"] != "sink":
        raise ValueError(f"resolve_sink resolution must be 'sink'; got {data['resolution']!r}")
    item = data["output"]
    if not isinstance(item, Mapping):
        raise ValueError(f"resolve_sink output must be an object; got {type(item).__name__}")
    expected = {"name", "plugin", "options", "required_fields", "schema_mode", "on_write_failure"}
    if set(item) != expected:
        raise ValueError(f"resolve_sink output must contain exactly {sorted(expected)}")
    name = item["name"]
    if type(name) is not str or not name:
        raise ValueError("resolve_sink output.name must be a non-empty string")
    plugin = item.get("plugin")
    if not isinstance(plugin, str) or not plugin:
        raise ValueError(f"resolve_sink output.plugin must be a non-empty string; got {plugin!r}")
    options = item.get("options")
    if not isinstance(options, Mapping):
        raise ValueError("resolve_sink output.options must be an object")
    required_fields_raw = item.get("required_fields")
    if not isinstance(required_fields_raw, list):
        raise ValueError("resolve_sink output.required_fields must be a list")
    required_fields: list[str] = []
    for col_idx, col in enumerate(required_fields_raw):
        if not isinstance(col, str) or not col:
            raise ValueError(f"resolve_sink output.required_fields[{col_idx}] must be a non-empty string")
        required_fields.append(col)
    schema_mode = item.get("schema_mode")
    if schema_mode not in ("fixed", "flexible", "observed"):
        raise ValueError(f"resolve_sink output.schema_mode must be fixed/flexible/observed; got {schema_mode!r}")
    on_write_failure = item["on_write_failure"]
    if type(on_write_failure) is not str or not on_write_failure:
        raise ValueError("resolve_sink output.on_write_failure must be a non-empty string")
    try:
        output = SinkOutputResolved(
            name=name,
            plugin=plugin,
            options=dict(options),
            required_fields=tuple(required_fields),
            schema_mode=schema_mode,
            on_write_failure=on_write_failure,
        )
    except (InvariantError, TypeError) as exc:
        raise ValueError("resolve_sink output snapshot is malformed") from exc
    assistant_message = _require_prose_assistant_message(data["assistant_message"], tool="resolve_sink")
    return SinkResolved(outputs=(output,)), assistant_message


_STEP_2_SINK_DISCOVERY_TOOL_NAMES: Final[frozenset[str]] = frozenset({"list_sinks", "get_plugin_schema"})
"""Read-only discovery tools the sink stage offers the composer model.

``list_sinks`` answers "which sink plugins exist" and ``get_plugin_schema``
answers "what options does this sink take" — the two facts a model needs to
build a sink without a hand-maintained inventory baked into the prompt. The
set is deliberately tight: source/transform/model discovery is irrelevant to
choosing an output, and every name here is asserted ``<= _DISCOVERY_TOOL_NAMES``
inside :func:`get_discovery_tool_definitions`.
"""

_DEFAULT_MAX_DISCOVERY_ITERS: Final[int] = 6
"""Fallback discovery-iteration cap when the route does not pass one.

Production threads ``settings.composer_max_discovery_turns``; this default
keeps direct callers (and tests) bounded. Reaching the cap returns ``None``
(advisory fallback), never raises.
"""


@dataclass(frozen=True, slots=True)
class Step2SinkChatOutcome:
    """Outcome of one ``resolve_sink``-equipped discovery-loop round.

    At most one of the two fields is populated. ``sink`` is set on a valid
    ``resolve_sink`` tool call, paired with its ``assistant_message``.
    Otherwise ``assistant_message`` alone carries the model's own
    register-guarded plain-prose reply — captured ONLY on a clean, tool-call-
    free response (see :func:`maybe_resolve_step_2_sink_chat`) — so the
    caller can show it directly without a second, tool-less call. Both
    fields ``None`` covers every other non-terminal case unchanged from
    before this salvage was added: a hallucinated non-discovery tool call, a
    genuinely empty/defective response, or the discovery-iteration cap.
    """

    sink: SinkResolved | None
    assistant_message: str | None


async def maybe_resolve_step_2_sink_chat(
    *,
    model: str,
    user_message: str,
    current_sink: SinkResolved | None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
    state: CompositionState | None = None,
    catalog: PolicyCatalogView | None = None,
    plugin_snapshot: PluginAvailabilitySnapshot | None = None,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    max_discovery_iters: int | None = None,
    timeout_seconds: float,
    context_block: str | None = None,
    progress: ComposerProgressSink | None = None,
) -> Step2SinkChatOutcome:
    """Resolve a Step-2 chat message into a sink config via a discovery loop.

    The composer model is given ``resolve_sink`` plus the read-only sink
    discovery tools (``list_sinks`` / ``get_plugin_schema``). Each round:

    * a ``resolve_sink`` call is terminal — parsed and returned;
    * one or more *allowed discovery* calls are dispatched via ``execute_tool``,
      their results threaded back, and the loop continues;
    * a clean, tool-call-free prose reply ends the loop returning that prose
      directly (register-guarded) so the caller never needs a second,
      tool-less call for an answer to show the user;
    * any tool call that is neither ``resolve_sink`` nor an allowed discovery
      tool ends the loop returning an empty outcome (advisory fallback)
      WITHOUT dispatching — the execution-side safety gate that stops a
      hallucinated mutation/secret call from running, since ``execute_tool``
      itself would otherwise happily dispatch one.

    Returns a :class:`Step2SinkChatOutcome`. ``sink`` (+ ``assistant_message``)
    is set on resolution; ``assistant_message`` alone is set on a clean prose
    decline; both ``None`` covers a hallucinated tool call, an empty/defective
    response, or the iteration cap — the route falls back to advisory chat in
    that case exactly as before.

    Discovery is active only when both ``state`` and ``catalog`` are supplied
    (the guided route always threads them). Without them the loop degrades to
    single-shot: the model sees only ``resolve_sink`` and either resolves or
    replies prose on the first round — the pre-loop behaviour.

    ``context_block`` (:func:`build_step_chat_context_block`) rides as an
    extra system message so a declined-to-prose reply is grounded in the same
    "current build" context the tool-less advisory call would otherwise have
    supplied — parity that keeps the salvaged prose no worse than a second
    call's (mirrors the Step-1 resolve path's same addition).

    Audit: one ``ComposerLLMCall`` is recorded per provider round and one
    ``ComposerToolInvocation`` per executed discovery call; the route drains
    both from *recorder* after it persists guided-session state.
    """
    if not user_message:
        raise InvariantError("maybe_resolve_step_2_sink_chat: user_message is empty (route validation gap)")

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    discovery_enabled = catalog is not None and plugin_snapshot is not None and state is not None
    discovery_defs = get_discovery_tool_definitions(_STEP_2_SINK_DISCOVERY_TOOL_NAMES) if discovery_enabled else []
    allowed_discovery = _STEP_2_SINK_DISCOVERY_TOOL_NAMES if discovery_enabled else frozenset()
    tools = [_STEP_2_SINK_TOOL, *discovery_defs]
    actor = user_id or "guided-composer"
    iteration_cap = max_discovery_iters if max_discovery_iters is not None else _DEFAULT_MAX_DISCOVERY_ITERS

    # NO Anthropic prompt-cache marker here (deliberate skip, not an oversight):
    # the step_2 sink skill is ~915 tokens, below Anthropic's 1024-token cache
    # floor, so a cache_control marker on it would be an inert no-op. Marking the
    # tool array / a cumulative prefix would cache something, but the win is
    # marginal at this size and the discovery-loop tool churn complicates the
    # breakpoint — deferred. Revisit if the step_2 skill grows past the floor.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_step_2_sink_tool_prompt(current_sink=current_sink)},
    ]
    if context_block is not None:
        messages.append({"role": "system", "content": context_block})
    messages.append({"role": "user", "content": user_message})

    for _iteration in range(max(1, iteration_cap)):
        request_messages = list(messages)
        kwargs: dict[str, Any] = {"model": model, "messages": request_messages, "tools": tools}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        started_at = datetime.now(UTC)
        started_ns = time.monotonic_ns()
        status: ComposerLLMCallStatus | None = None
        response: Any = None
        error_class: str | None = None
        error_message: str | None = None
        # Visible before the (slow) provider round-trip so a poller sampling
        # mid-call sees "calling_model", not a stale prior-phase snapshot.
        await emit_progress(progress, model_call_progress_event(user_message))
        try:
            response = await _bounded_acompletion(kwargs, timeout_seconds)
            message = response.choices[0].message
            tool_calls = message.tool_calls or ()

            # resolve_sink is terminal — take it regardless of sibling calls.
            for tool_call in tool_calls:
                function = tool_call.function
                if function is None or function.name != "resolve_sink":
                    continue
                arguments = function.arguments
                if not isinstance(arguments, str):
                    raise ValueError(f"resolve_sink function.arguments must be a JSON string; got {type(arguments).__name__}")
                sink, assistant = _parse_step_2_sink_tool_arguments(arguments)
                status = ComposerLLMCallStatus.SUCCESS
                return Step2SinkChatOutcome(sink=sink, assistant_message=assistant)

            # A clean, tool-call-free reply: the model judged the message
            # doesn't carry enough detail to act (or it's a plain question)
            # and answered in prose instead. Validate + return that prose
            # directly — the SAME register guard the tool argument gets — so
            # the caller never needs a second, tool-less call for an answer
            # to show the user. Deliberately gated on ``not tool_calls``, NOT
            # folded into the safety-gate branch below: a response that ALSO
            # carries a hallucinated tool call is a more suspicious shape and
            # must not have its prose trusted either (falls through instead).
            if not tool_calls:
                content = message.content
                if content is None or not str(content).strip():
                    status = ComposerLLMCallStatus.SUCCESS
                    return Step2SinkChatOutcome(sink=None, assistant_message=None)
                prose = _require_prose_assistant_message(str(content), tool="maybe_resolve_step_2_sink_chat")
                status = ComposerLLMCallStatus.SUCCESS
                return Step2SinkChatOutcome(sink=None, assistant_message=prose)

            # Execution-side safety gate: the only non-terminal calls we
            # dispatch are allowed read-only discovery tools. ANY other tool
            # (a hallucinated mutation / secret call) ends the loop without
            # dispatching anything.
            discovery_calls = [tc for tc in tool_calls if tc.function is not None and tc.function.name in allowed_discovery]
            if not discovery_calls or len(discovery_calls) != len(tool_calls):
                status = ComposerLLMCallStatus.SUCCESS
                return Step2SinkChatOutcome(sink=None, assistant_message=None)

            # Thread the assistant tool-call request once, then answer every
            # call id with its result, or the next round 400s.
            assert state is not None and catalog is not None and plugin_snapshot is not None  # implied by discovery_enabled
            await emit_progress(
                progress,
                tool_batch_progress_event(tuple(tc.function.name for tc in discovery_calls if tc.function is not None)),
            )
            messages.append(_assistant_tool_calls_message(message, tool_calls))
            for tool_call in tool_calls:
                messages.append(
                    _execute_discovery_call(
                        tool_call=tool_call,
                        state=state,
                        catalog=catalog,
                        plugin_snapshot=plugin_snapshot,
                        secret_service=secret_service,
                        user_id=user_id,
                        actor=actor,
                        recorder=recorder,
                    )
                )
            status = ComposerLLMCallStatus.SUCCESS
            # fall through to finally (records this round), then loop again
        except TimeoutError:
            status = ComposerLLMCallStatus.TIMEOUT
            error_class = "TimeoutError"
            error_message = "TimeoutError"
            raise
        except asyncio.CancelledError as exc:
            status = ComposerLLMCallStatus.CANCELLED
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAuthError as exc:
            status = ComposerLLMCallStatus.AUTH_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMBadRequestError as exc:
            status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except LiteLLMAPIError as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        except (IndexError, AttributeError, json.JSONDecodeError, ValueError, GuidedSolverResponseShapeError) as exc:
            # ``GuidedSolverResponseShapeError`` from a malformed discovery-tool
            # dispatch (``_execute_discovery_call``) is a response-shape failure,
            # not an unknown server error — classify it MALFORMED_RESPONSE
            # instead of falling through to the API_ERROR catch-all. It still re-raises; the auto-drop wrapper
            # (``resolve_step_2_sink_chat_with_auto_drop``) turns it into the
            # advisory fallback.
            status = ComposerLLMCallStatus.MALFORMED_RESPONSE
            error_class = type(exc).__name__
            error_message = "malformed_response"
            raise
        except Exception as exc:
            status = ComposerLLMCallStatus.API_ERROR
            error_class = type(exc).__name__
            error_message = type(exc).__name__
            raise
        finally:
            _record_llm_call(
                recorder=recorder,
                model=model,
                messages=request_messages,
                tools=tools,
                status=status,
                started_at=started_at,
                started_ns=started_ns,
                temperature=temperature,
                seed=seed,
                response=response,
                error_class=error_class,
                error_message=error_message,
            )

    # Discovery iteration cap reached without a resolve_sink — advisory fallback.
    return Step2SinkChatOutcome(sink=None, assistant_message=None)


async def solve_step_chat(
    *,
    model: str,
    step: GuidedStep,
    user_message: str,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
    timeout_seconds: float,
    context_block: str | None = None,
) -> str:
    """Send a user chat message to the LLM scoped to *step*; return the assistant reply.

    Args:
        model: LiteLLM model identifier from settings.composer_model.  Required —
            callers must be explicit; there is no hard-coded model default.
        step: The user's current wizard step.  Determines which playbook the
            LLM receives via ``load_step_chat_skill(step)``.
        user_message: The user's typed message.  Tier 3 by trust model — the
            route handler is responsible for non-empty / length validation
            before this is called.
        context_block: Optional LLM-safe "current build" block
            (:func:`build_step_chat_context_block`) so what-am-I-seeing / why
            questions get answers grounded in the actual applied artifacts.
            Rides as a SECOND system message — the per-step skill stays the
            byte-stable, cache-markable head.

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

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    system_prompt = load_step_chat_skill(step)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": _ADVISORY_NO_TOOLS_ADDENDUM},
    ]
    if context_block is not None:
        messages.append({"role": "system", "content": context_block})
    messages.append({"role": "user", "content": user_message})
    # Anthropic-family routes honor an explicit ``cache_control`` marker on the
    # stable skill head (the freeform pattern; ``service.py``). Mark BEFORE
    # kwargs so the SAME marked list feeds both the wire call and the audit
    # ``build_llm_call_record(messages=messages)`` in the finally block — the
    # recorded ``messages_hash`` stays truthful to what was sent. ``solve_step_chat``
    # attaches no tools, so the tools half is ``None``. Below-floor stages
    # (STEP_2_SINK ~915 tok, STEP_4_WIRE ~749 tok) are marked here too but the
    # marker is an inert no-op below Anthropic's 1024-token cache floor.
    if supports_anthropic_prompt_cache_markers(model):
        messages, _ = apply_anthropic_cache_markers(messages, None)
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if seed is not None:
        kwargs["seed"] = seed
    started_at = datetime.now(UTC)
    started_ns = time.monotonic_ns()
    status: ComposerLLMCallStatus | None = None
    response: Any = None
    error_class: str | None = None
    error_message: str | None = None
    try:
        response = await _bounded_acompletion(kwargs, timeout_seconds)

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
        # Same register guard as the resolve-path assistant_message args: this
        # reply persists into chat_history and renders verbatim as the
        # user-facing bubble. Observed live 2026-07-03 (live guided, step_1):
        # the model answered the advisory path with a full pseudo
        # <tool_call>/<tool_response> transcript as literal content. Raises
        # AssistantScaffoldLeakError → MALFORMED_RESPONSE in the audit record;
        # the advisory wrapper absorbs it to the synthetic-unavailable retry.
        prose = _require_prose_assistant_message(str(content), tool="solve_step_chat")
        status = ComposerLLMCallStatus.SUCCESS
        return prose
    except TimeoutError:
        status = ComposerLLMCallStatus.TIMEOUT
        error_class = "TimeoutError"
        error_message = "TimeoutError"
        raise
    except asyncio.CancelledError as exc:
        status = ComposerLLMCallStatus.CANCELLED
        error_class = type(exc).__name__
        error_message = type(exc).__name__
        raise
    except LiteLLMAuthError as exc:
        status = ComposerLLMCallStatus.AUTH_ERROR
        error_class = type(exc).__name__
        error_message = type(exc).__name__
        raise
    except LiteLLMBadRequestError as exc:
        status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
        error_class = type(exc).__name__
        error_message = type(exc).__name__
        raise
    except LiteLLMAPIError as exc:
        status = ComposerLLMCallStatus.API_ERROR
        error_class = type(exc).__name__
        error_message = type(exc).__name__
        raise
    except (IndexError, AttributeError, json.JSONDecodeError, InvariantError, AssistantScaffoldLeakError) as exc:
        status = ComposerLLMCallStatus.MALFORMED_RESPONSE
        error_class = type(exc).__name__
        error_message = "malformed_response"
        raise
    except Exception as exc:
        status = ComposerLLMCallStatus.API_ERROR
        error_class = type(exc).__name__
        error_message = type(exc).__name__
        raise
    finally:
        _record_llm_call(
            recorder=recorder,
            model=model,
            messages=messages,
            tools=None,
            status=status,
            started_at=started_at,
            started_ns=started_ns,
            temperature=temperature,
            seed=seed,
            response=response,
            error_class=error_class,
            error_message=error_message,
        )
