"""Skill RGR harness — drive a model through a composer scenario.

Loads the production composer tool definitions from
:mod:`elspeth.web.composer.tools` and the production skill text via
:func:`elspeth.web.composer.skills.load_skill`, then runs a scripted user
prompt against a configurable model with stubbed tool executors.  The
output is a sampling-controlled transcript of the LLM's tool calls and
free text, which we diff between RED (current skill) and GREEN (edited
skill) to verify that a finding is actually fixed by a documentation
change rather than by harness noise.

Why this exists
---------------

The pipeline-composer skill is the cornerstone of an LLM-driven
auditable pipeline composition system.  The Iron Law from
``superpowers:writing-skills`` is "no skill edit without a failing test
first."  For documentation, the test is "an LLM under pressure produces
the wrong tool call."  This harness makes that test repeatable on
routes that honor ``temperature`` / ``seed``. Reasoning routes may reject
those fields; ``drop_params=True`` keeps those routes runnable but makes
their transcripts best-effort rather than strictly deterministic.

Provider routing
----------------

We call ``litellm.completion`` rather than ``openai.chat.completions``
because the composer uses litellm in production.  Routing the same
scenario through both ``gpt-5.5`` and Claude lets us catch findings
that only fail under one model (instruction-following profiles diverge
between providers).  ChatGPT 5.5 is cheaper and is used for the bulk
of RED detection; Claude (the production target) is used for GREEN
verification before any skill edit lands.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo-root import without packaging — matches scripts/cicd convention.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def _load_dotenv() -> None:
    """Load OPENROUTER_API_KEY (and friends) from the project .env.

    The composer service uses the same key; loading it here means the
    harness routes through OpenRouter exactly as production does.  We
    avoid a hard dep on python-dotenv — a 20-line parser is enough for
    KEY=value lines and matches what scripts/deploy-vm.sh expects.
    """

    env_path = _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

import litellm  # noqa: E402

from elspeth.web.composer import skills as composer_skills  # noqa: E402
from elspeth.web.composer.tools import get_tool_definitions  # noqa: E402

TRANSCRIPTS_DIR = Path(__file__).resolve().parent / "transcripts"
SCENARIOS_DIR = Path(__file__).resolve().parent / "scenarios"
_SAMPLING_TEMPERATURE = 0
_SAMPLING_SEED = 0

ToolStub = Callable[[dict[str, Any]], Any]


@dataclass
class Scenario:
    """A single RED/GREEN scenario.

    Attributes:
        name: Stable identifier used in transcript filenames.
        user_prompt: The user message that drives the conversation.
        stubs: Map of tool name -> callable returning the result dict.
            Tools not present in this map are answered with a default
            empty-state stub so the LLM cannot rely on undocumented
            behaviour.
        max_turns: Hard cap on assistant-tool round trips before the
            conversation is forcibly terminated.  20 is generous for
            composition workflows that legitimately call ten or more
            tools in sequence.
        red_predicates: Functions that take a transcript and return
            True if the predicted RED behaviour was observed.  All
            must be True for "RED confirmed."
        green_predicates: Functions that take a transcript and return
            True if the predicted GREEN behaviour was observed.  All
            must be True for "GREEN confirmed."
    """

    name: str
    user_prompt: str
    stubs: dict[str, ToolStub] = field(default_factory=dict)
    max_turns: int = 20
    red_predicates: list[Callable[[list[dict[str, Any]]], bool]] = field(default_factory=list)
    green_predicates: list[Callable[[list[dict[str, Any]]], bool]] = field(default_factory=list)


def _default_stub(tool_name: str) -> ToolStub:
    """Return a stub that produces a plausible empty / valid response.

    This keeps the LLM unblocked for tools the scenario didn't override,
    rather than crashing the run.  Each shape is the smallest valid
    response per the tool's contract in tools.py.
    """

    if tool_name in {"list_sources", "list_transforms", "list_sinks"}:
        kind = tool_name.removeprefix("list_")
        # Return a minimal but realistic plugin list — enough to not
        # appear broken, not so much that scenarios become noisy.
        return lambda _args: {"plugins": [{"name": "csv", "kind": kind}]}
    if tool_name == "get_plugin_schema":
        return lambda args: {
            "plugin": args.get("name"),
            "schema": {"type": "object", "properties": {}, "required": []},
        }
    if tool_name == "list_secret_refs":
        return lambda _args: {"secrets": []}
    if tool_name == "preview_pipeline":
        return lambda _args: {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "edge_contracts": [],
            "suggestions": [],
        }
    if tool_name == "get_pipeline_state":
        return lambda _args: {"source": None, "nodes": [], "outputs": {}}
    return lambda _args: {"status": "ok"}


def _tools_for_litellm(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap composer tool defs in OpenAI tools-API envelope.

    ``get_tool_definitions()`` returns the inner ``{name, description,
    parameters}`` object; the chat-completions tools API expects each
    entry wrapped as ``{type: "function", function: {...}}``.
    """

    return [{"type": "function", "function": td} for td in tool_defs]


def run_scenario(
    scenario: Scenario,
    *,
    skill_text: str,
    model: str = "gpt-5.5",
    label: str,
) -> list[dict[str, Any]]:
    """Run one scenario end-to-end and return the transcript.

    Args:
        scenario: The scenario to execute.
        skill_text: The system-prompt skill to inject.  Pass the
            production text for RED, an edited variant for GREEN.
        model: litellm model identifier.  ``gpt-5.5`` for cheap RED
            detection; ``claude-opus-4-7`` for GREEN verification.
        label: Suffix for the transcript filename — typically "red" or
            "green" or "refactor-<variant>".

    Returns:
        The transcript as an ordered list of message + tool-result
        dicts.  Persisted to disk under ``transcripts/<scenario>/``.
    """

    tools = _tools_for_litellm(get_tool_definitions())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": skill_text},
        {"role": "user", "content": scenario.user_prompt},
    ]
    transcript: list[dict[str, Any]] = [{"role": "user", "content": scenario.user_prompt}]

    for turn in range(scenario.max_turns):
        # Note: drop_params lets litellm strip provider-incompatible
        # fields without failing the call — important when ranging
        # across gpt-5 / Claude / Azure with one harness.
        #
        # max_tokens=4096 is a deliberate, generous budget.  GPT-5
        # family models are *reasoning* models that consume hidden
        # tokens before producing tool calls or visible text.  A
        # 64-token budget leaves zero room for the actual response;
        # 4096 leaves room for several tool calls in one turn while
        # still capping runaway loops.  Claude doesn't need the
        # budget but it doesn't hurt either.
        resp = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096,
            temperature=_SAMPLING_TEMPERATURE,
            seed=_SAMPLING_SEED,
            drop_params=True,
        )
        choice = resp.choices[0].message
        msg_dict = choice.model_dump() if hasattr(choice, "model_dump") else dict(choice)
        transcript.append({"role": "assistant", "turn": turn, **msg_dict})

        tool_calls = msg_dict.get("tool_calls") or []
        if not tool_calls:
            break

        messages.append(msg_dict)
        malformed_tool_call = False
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as exc:
                transcript.append(
                    {
                        "role": "tool_argument_error",
                        "tool_call_id": tc.get("id"),
                        "name": name,
                        "raw_args": raw_args,
                        "error_class": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                malformed_tool_call = True
                break
            stub = scenario.stubs.get(name) or _default_stub(name)
            result = stub(args)
            result_str = json.dumps(result, default=str)
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "args": args,
                    "result": result,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": name,
                    "content": result_str,
                }
            )
        if malformed_tool_call:
            break

    out_dir = TRANSCRIPTS_DIR / scenario.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "scenario": scenario.name,
                "model": model,
                "label": label,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "transcript": transcript,
            },
            indent=2,
            default=str,
        )
    )
    return transcript


def evaluate(transcript: list[dict[str, Any]], scenario: Scenario, *, phase: str) -> dict[str, bool]:
    """Run RED or GREEN predicates against a transcript.

    Args:
        transcript: Output of :func:`run_scenario`.
        scenario: Same scenario object used to produce the transcript.
        phase: ``"red"`` or ``"green"``.

    Returns:
        Map of predicate-index -> bool result.  All True means the
        phase is confirmed.
    """

    predicates = scenario.red_predicates if phase == "red" else scenario.green_predicates
    return {f"p{i}": bool(p(transcript)) for i, p in enumerate(predicates)}


def load_skill(*, override_path: Path | None = None) -> str:
    """Load the current production skill, or an edited override.

    Override paths let GREEN runs use a candidate edit without
    modifying the file in place — important for keeping RED runs
    reproducible.
    """

    if override_path is not None:
        return override_path.read_text(encoding="utf-8")
    return composer_skills.load_skill("pipeline_composer")


# Helper predicates used by scenarios.


def called_tool(transcript: list[dict[str, Any]], tool_name: str) -> bool:
    """True if the LLM called ``tool_name`` at any point."""

    return any(entry.get("role") == "tool" and entry.get("name") == tool_name for entry in transcript)


def tried_to_load_schema(transcript: list[dict[str, Any]], tool_name: str) -> bool:
    """True if the LLM emitted a tool call whose function name is
    ``tool_name`` — used for detecting schema-load attempts on tools
    that don't exist (e.g. ``generate_yaml``).

    Note: when the model calls a tool that isn't in the tools list,
    OpenAI/litellm typically reject the call before it reaches us.
    Detection therefore needs to look at the assistant message's
    ``tool_calls`` array, including malformed/rejected calls — which
    we capture in the transcript via ``model_dump``.
    """

    for entry in transcript:
        if entry.get("role") != "assistant":
            continue
        for tc in entry.get("tool_calls") or []:
            if tc.get("function", {}).get("name") == tool_name:
                return True
    return False


def emitted_text_matching(transcript: list[dict[str, Any]], needle: str) -> bool:
    """True if any assistant free-text content contains ``needle`` (case-insensitive)."""

    needle_lc = needle.lower()
    for entry in transcript:
        if entry.get("role") != "assistant":
            continue
        content = entry.get("content") or ""
        if isinstance(content, str) and needle_lc in content.lower():
            return True
    return False


def tool_call_args_match(transcript: list[dict[str, Any]], tool_name: str, arg_predicate: Callable[[dict[str, Any]], bool]) -> bool:
    """True if any call to ``tool_name`` had arguments matching the predicate.

    Used for detecting specific argument shapes (e.g. ``upsert_node``
    with a route key that's the boolean ``true`` rather than the string
    ``"true"``).
    """

    for entry in transcript:
        if entry.get("role") != "tool" or entry.get("name") != tool_name:
            continue
        args = entry.get("args") or {}
        if arg_predicate(args):
            return True
    return False
