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
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, cast

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.secrets import WebSecretResolver
from elspeth.web.blobs.protocol import ALLOWED_MIME_TYPES, AllowedMimeType
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided._discovery import _assistant_tool_calls_message, _execute_discovery_call
from elspeth.web.composer.guided.errors import ChainSolverResponseShapeError, InvariantError
from elspeth.web.composer.guided.prompts import _summarize_sample_row, load_step_chat_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved
from elspeth.web.composer.llm_response_parsing import (
    apply_anthropic_cache_markers,
    attach_llm_calls,
    build_llm_call_record,
    supports_anthropic_prompt_cache_markers,
)
from elspeth.web.composer.service import _litellm_acompletion
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools._dispatch import get_discovery_tool_definitions
from elspeth.web.interpretation_state import SOURCE_AUTHORING_KEY

# Server-owned source-option keys that the LLM must NEVER author. Both are
# stamped authoritatively at commit (``blob_ref`` by ``handle_step_1_source``,
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
        freeze_fields(self, "options", "sample_rows")


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
        "`mime_type` to `text/csv`. When the user wants to FETCH or SCRAPE one or more "
        "URLs, the source is an INLINE `json` (or `csv`) dataset whose rows carry each "
        "URL in a `url` column — e.g. json `content` of "
        '`[{"url": "https://example/a"}, {"url": "https://example/b"}]`. Declare that '
        "`url` column as guaranteed on the source `schema` "
        '(`{"mode": "observed", "guaranteed_fields": ["url"]}`) so the downstream '
        "web_scrape transform's required `url` input is satisfied at the wiring step "
        "(an observed-mode source that guarantees nothing fails that contract). You "
        "must NOT choose a `web_scraper`/`web_scrape` source: fetching pages is a "
        "downstream TRANSFORM applied later, never a source plugin. The only valid "
        "source plugins are `azure_blob`, `csv`, `dataverse`, `json`, `null`, `text`. "
        "Preserve user-supplied values exactly in the file "
        "content; do not invent hidden pipeline transforms. Also set `on_validation_failure` "
        "when you resolve a source: use `discard` for a demo source that is valid by "
        "construction, or the name of a quarantine sink for production data whose invalid "
        "rows must be kept for inspection. If the message is only a "
        "question or lacks enough source detail, reply in prose and do not call a tool.\n"
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
    outputs: list[dict[str, Any]] = []
    for output in current_sink.outputs:
        options = output.options if isinstance(output.options, Mapping) else {}
        outputs.append(
            {
                "plugin": output.plugin,
                "required_fields": list(output.required_fields),
                "schema_mode": output.schema_mode,
                "option_count": len(options),
            }
        )
    return {"outputs": outputs}


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


async def _bounded_acompletion(kwargs: dict[str, Any], timeout_seconds: float | None) -> Any:
    """Run ``_litellm_acompletion`` under an ``asyncio.wait_for`` bound.

    ``timeout_seconds=None`` preserves the unbounded legacy behaviour for
    direct callers/tests; the guided routes thread
    ``settings.composer_timeout_seconds`` so a guided LLM call is bounded the
    same way freeform compose bounds its calls (``composer/service.py``
    ``asyncio.wait_for(..., timeout=self._timeout_seconds)``). The raised
    ``TimeoutError`` lands in each solver's existing ``except TimeoutError``
    audit branch (status=TIMEOUT) and the auto-drop wrappers turn it into the
    synthetic-unavailable / advisory-fallback contract.
    """
    if timeout_seconds is None:
        return await _litellm_acompletion(**kwargs)
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
    timeout_seconds: float | None = None,
) -> Step1SourceChatResolution | None:
    """Try to resolve a Step-1 schema-form chat message into source data.

    Returns ``None`` when the model replies in ordinary prose without a
    ``resolve_source`` tool call, allowing the route to fall back to the
    advisory chat path.

    When ``current_source`` is supplied the tool prompt includes the current
    applied source so a revision instruction ("add a url column", "make it
    csv not json") resolves relative to it.
    """
    if not user_message:
        raise InvariantError("maybe_resolve_step_1_source_chat: user_message is empty (route validation gap)")

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

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
        {"role": "user", "content": user_message},
    ]
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
            return result
        status = ComposerLLMCallStatus.SUCCESS
        return None
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


_STEP_2_SINK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "resolve_sink",
        "description": (
            "Use when the Step 2 chat message contains enough information to "
            "configure the pipeline output(s). Do not use for general advice."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["resolution", "outputs", "assistant_message"],
            "properties": {
                "resolution": {"type": "string", "enum": ["sink"]},
                "outputs": {
                    "type": "array",
                    "minItems": 1,
                    # MVP single-output constraint enforced at the schema boundary:
                    # handle_step_2_sink loops outputs as sink_name="main"
                    # (last-write-wins) and the from-resolved re-render shows
                    # outputs[0] — so >1 output would silently disagree. Cap at 1.
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["plugin", "options", "required_fields", "schema_mode"],
                        "properties": {
                            "plugin": {"type": "string", "minLength": 1},
                            # Bare object: option shape varies by sink plugin.
                            # Validated downstream by handle_step_2_sink ->
                            # _execute_set_output.
                            "options": {"type": "object"},
                            "required_fields": {"type": "array", "items": {"type": "string"}},
                            "schema_mode": {"type": "string", "enum": ["fixed", "flexible", "observed"]},
                        },
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
            "instruction against it — re-emit the COMPLETE updated outputs (not a "
            "diff). Current sink:\n"
            f"{json.dumps(_sink_revision_context_for_llm(current_sink), sort_keys=True)}\n"
        )
    return (
        f"{load_step_chat_skill(GuidedStep.STEP_2_SINK).rstrip()}\n\n"
        "## Step 2 Sink Tool\n\n"
        f"{revise_block}"
        "If the user's message provides enough information to configure the "
        "pipeline output, call `resolve_sink` with the complete list of outputs "
        "(plugin, options, required_fields, schema_mode) and a brief "
        "assistant_message. If the message is only a question or lacks enough "
        "detail, reply in prose and do not call a tool.\n"
    )


def _parse_step_2_sink_tool_arguments(arguments: str) -> tuple[SinkResolved, str]:
    """Validate the resolve_sink tool arguments. Returns (sink, assistant_message)."""
    data = json.loads(arguments)
    if not isinstance(data, Mapping):
        raise ValueError(f"resolve_sink arguments must decode to an object; got {type(data).__name__}")
    missing = {"resolution", "outputs", "assistant_message"} - set(data.keys())
    if missing:
        raise ValueError(f"resolve_sink arguments missing required keys: {sorted(missing)}")
    if data["resolution"] != "sink":
        raise ValueError(f"resolve_sink resolution must be 'sink'; got {data['resolution']!r}")
    outputs_raw = data["outputs"]
    if not isinstance(outputs_raw, list) or not outputs_raw:
        raise ValueError("resolve_sink outputs must be a non-empty list")
    # Enforce the MVP single-output cap SERVER-SIDE, not only via the schema's
    # advisory `maxItems: 1`. A model emitting 2 outputs would otherwise sail
    # through here and handle_step_2_sink would silently last-write-wins on
    # sink_name="main" — the "silently disagree" the schema comment warns about.
    # ELSPETH doctrine: strict validation at the parse boundary for Tier-3
    # LLM-originated input. The ValueError routes to MALFORMED_RESPONSE -> advisory.
    if len(outputs_raw) > 1:
        raise ValueError(f"resolve_sink accepts at most one output (MVP single-output cap); got {len(outputs_raw)}")
    outputs: list[SinkOutputResolved] = []
    for idx, item in enumerate(outputs_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"resolve_sink outputs[{idx}] must be an object; got {type(item).__name__}")
        plugin = item.get("plugin")
        if not isinstance(plugin, str) or not plugin:
            raise ValueError(f"resolve_sink outputs[{idx}].plugin must be a non-empty string; got {plugin!r}")
        options = item.get("options")
        if not isinstance(options, Mapping):
            raise ValueError(f"resolve_sink outputs[{idx}].options must be an object")
        required_fields_raw = item.get("required_fields")
        if not isinstance(required_fields_raw, list):
            raise ValueError(f"resolve_sink outputs[{idx}].required_fields must be a list")
        required_fields: list[str] = []
        for col_idx, col in enumerate(required_fields_raw):
            if not isinstance(col, str) or not col:
                raise ValueError(f"resolve_sink outputs[{idx}].required_fields[{col_idx}] must be a non-empty string")
            required_fields.append(col)
        schema_mode = item.get("schema_mode")
        if schema_mode not in ("fixed", "flexible", "observed"):
            raise ValueError(f"resolve_sink outputs[{idx}].schema_mode must be fixed/flexible/observed; got {schema_mode!r}")
        outputs.append(
            SinkOutputResolved(
                plugin=plugin,
                options=dict(options),
                required_fields=tuple(required_fields),
                schema_mode=schema_mode,
            )
        )
    assistant_message = _require_prose_assistant_message(data["assistant_message"], tool="resolve_sink")
    return SinkResolved(outputs=tuple(outputs)), assistant_message


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


async def maybe_resolve_step_2_sink_chat(
    *,
    model: str,
    user_message: str,
    current_sink: SinkResolved | None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
    state: CompositionState | None = None,
    catalog: CatalogService | None = None,
    secret_service: WebSecretResolver | None = None,
    user_id: str | None = None,
    max_discovery_iters: int | None = None,
    timeout_seconds: float | None = None,
) -> tuple[SinkResolved, str] | None:
    """Resolve a Step-2 chat message into a sink config via a discovery loop.

    The composer model is given ``resolve_sink`` plus the read-only sink
    discovery tools (``list_sinks`` / ``get_plugin_schema``). Each round:

    * a ``resolve_sink`` call is terminal — parsed and returned;
    * one or more *allowed discovery* calls are dispatched via ``execute_tool``,
      their results threaded back, and the loop continues;
    * a prose reply, OR any tool call that is neither ``resolve_sink`` nor an
      allowed discovery tool, ends the loop returning ``None`` (advisory
      fallback) WITHOUT dispatching — the execution-side safety gate that stops
      a hallucinated mutation/secret call from running, since ``execute_tool``
      itself would otherwise happily dispatch one.

    Returns ``(sink, assistant_message)`` on resolution, or ``None`` (prose /
    gate trip / iteration cap) so the route falls back to advisory chat.

    Discovery is active only when both ``state`` and ``catalog`` are supplied
    (the guided route always threads them). Without them the loop degrades to
    single-shot: the model sees only ``resolve_sink`` and either resolves or
    replies prose on the first round — the pre-loop behaviour.

    Audit: one ``ComposerLLMCall`` is recorded per provider round and one
    ``ComposerToolInvocation`` per executed discovery call; the route drains
    both from *recorder* after it persists guided-session state.
    """
    if not user_message:
        raise InvariantError("maybe_resolve_step_2_sink_chat: user_message is empty (route validation gap)")

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    discovery_enabled = catalog is not None and state is not None
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
        {"role": "user", "content": user_message},
    ]

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
                return sink, assistant

            # Execution-side safety gate: the only non-terminal calls we
            # dispatch are allowed read-only discovery tools. A prose reply
            # (no tool calls) or ANY other tool (a hallucinated mutation /
            # secret call) ends the loop without dispatching anything.
            discovery_calls = [tc for tc in tool_calls if tc.function is not None and tc.function.name in allowed_discovery]
            if not discovery_calls or len(discovery_calls) != len(tool_calls):
                status = ComposerLLMCallStatus.SUCCESS
                return None

            # Thread the assistant tool-call request once, then answer every
            # call id with its result, or the next round 400s.
            assert state is not None and catalog is not None  # implied by discovery_enabled
            messages.append(_assistant_tool_calls_message(message, tool_calls))
            for tool_call in tool_calls:
                messages.append(
                    _execute_discovery_call(
                        tool_call=tool_call,
                        state=state,
                        catalog=catalog,
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
        except (IndexError, AttributeError, json.JSONDecodeError, ValueError, ChainSolverResponseShapeError) as exc:
            # ``ChainSolverResponseShapeError`` from a malformed discovery-tool
            # dispatch (``_execute_discovery_call``) is a response-shape failure,
            # not an unknown server error — classify it MALFORMED_RESPONSE like
            # ``solve_chain`` (chain_solver.py), instead of falling through to the
            # API_ERROR catch-all. It still re-raises; the auto-drop wrapper
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
    return None


async def solve_step_chat(
    *,
    model: str,
    step: GuidedStep,
    user_message: str,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
    timeout_seconds: float | None = None,
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

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    system_prompt = load_step_chat_skill(step)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
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
