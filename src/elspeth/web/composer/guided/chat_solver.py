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
from typing import Any, cast

from elspeth.contracts.composer_llm_audit import ComposerLLMCallStatus
from elspeth.contracts.freeze import freeze_fields
from elspeth.web.blobs.protocol import ALLOWED_MIME_TYPES, AllowedMimeType
from elspeth.web.composer.audit import BufferingRecorder
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.prompts import load_step_chat_skill
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.llm_response_parsing import attach_llm_calls, build_llm_call_record
from elspeth.web.composer.service import _litellm_acompletion


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
                "options": {"type": "object"},
                "observed_columns": {"type": "array", "items": {"type": "string"}},
                "sample_rows": {"type": "array", "items": {"type": "object"}},
                "assistant_message": {"type": "string", "minLength": 1},
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


def _build_step_1_source_tool_prompt(*, plugin_hint: str | None) -> str:
    """Compose the Step-1 source/data-schema tool prompt."""
    hint = (
        f"The current source plugin selected in the wizard is {plugin_hint!r}."
        if plugin_hint is not None
        else "The current source plugin is not persisted in server state; infer only when the chat message or tool context makes it explicit."
    )
    return (
        f"{load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()}\n\n"
        "## Step 1 Source/Data Schema Tool\n\n"
        f"{hint}\n"
        "If the user's message provides enough information to create inline source data, "
        "call `resolve_source` with the complete file content, the source plugin, "
        "schema options, observed columns, representative sample rows, and a brief "
        "assistant_message. For CSV data, include a header row in `content` and set "
        "`mime_type` to `text/csv`. Preserve user-supplied values exactly in the file "
        "content; do not invent hidden pipeline transforms. If the message is only a "
        "question or lacks enough source detail, reply in prose and do not call a tool.\n"
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

    assistant_message = data["assistant_message"]
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        raise ValueError("resolve_source assistant_message must be a non-empty string")

    return Step1SourceChatResolution(
        assistant_message=assistant_message,
        plugin=plugin,
        filename=filename,
        mime_type=cast(AllowedMimeType, mime_type),
        content=content,
        options=dict(options),
        observed_columns=tuple(observed_columns),
        sample_rows=tuple(sample_rows),
    )


async def maybe_resolve_step_1_source_chat(
    *,
    model: str,
    user_message: str,
    plugin_hint: str | None,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
) -> Step1SourceChatResolution | None:
    """Try to resolve a Step-1 schema-form chat message into source data.

    Returns ``None`` when the model replies in ordinary prose without a
    ``resolve_source`` tool call, allowing the route to fall back to the
    advisory chat path.
    """
    if not user_message:
        raise InvariantError("maybe_resolve_step_1_source_chat: user_message is empty (route validation gap)")

    from litellm.exceptions import APIError as LiteLLMAPIError
    from litellm.exceptions import AuthenticationError as LiteLLMAuthError
    from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _build_step_1_source_tool_prompt(plugin_hint=plugin_hint)},
        {"role": "user", "content": user_message},
    ]
    tools = [_STEP_1_SOURCE_TOOL]
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
        response = await _litellm_acompletion(**kwargs)

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


async def solve_step_chat(
    *,
    model: str,
    step: GuidedStep,
    user_message: str,
    temperature: float | None,
    seed: int | None,
    recorder: BufferingRecorder | None = None,
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
        status = ComposerLLMCallStatus.SUCCESS
        # mypy: LiteLLM's response is `Any`, narrow to str at the trust boundary.
        return str(content)
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
    except (IndexError, AttributeError, json.JSONDecodeError, InvariantError) as exc:
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
