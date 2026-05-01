"""Batch 3 scenario — composer bootstrap / tool-inventory behaviour.

Authored alongside Followup D Scope-A (drift gate landed in
``tests/unit/web/composer/test_skill_drift.py::TestComposerToolNameDrift``).
This scenario exists to gather *empirical evidence* about whether real
models struggle with the composer's per-tool deferred-schema-load pattern
— evidence that conditions whether Scope-B (a single ``composer_bootstrap``
tool, tracked as ``elspeth-4f85bd6652``) is worth the architectural cost.

Harness limitation — read this before interpreting results
----------------------------------------------------------

The harness uses :func:`litellm.completion` with the full
``get_tool_definitions()`` list passed via ``tools=...`` (see
``harness.py::run_scenario`` line 201-208). This is **not** the deferred-
MCP tool surface that the production composer service uses. In the
harness, every tool's schema is always available; the LLM never has to
issue a "load schema for X" call before invoking X.

Consequence: the originally-proposed RED predicate "model hits
``InputValidationError`` on ``wire_secret_ref`` because the schema isn't
loaded" is structurally impossible to reproduce here. The harness cannot
exercise the deferred-load pathology itself — that needs an integration
test that goes through the real MCP service, which is out of scope for
the RGR harness.

What this scenario *can* test (and does):

1. **Secret-handling instruction-following.** Does the model use
   ``wire_secret_ref`` with the marker form, or does it try to embed
   ``${OPENROUTER_API_KEY}`` as a literal string into option values? The
   skill instructs the marker form; a model that interpolates literally
   has misread the instruction. RED.
2. **Tool hallucination.** Does the model invoke a tool name that is not
   in ``get_tool_definitions()``? The harness reflects every assistant
   ``tool_calls`` entry into the transcript, so phantom tool names are
   directly observable. RED.
3. **End-to-end pipeline closure.** Does the model reach
   ``preview_pipeline`` after wiring the secret? The skill instructs
   ``preview_pipeline`` as the final pre-export gate. GREEN.

Predicate decisions
-------------------

The Followup D prompt listed four predicates. Three are kept verbatim
in spirit (literal interpolation, phantom tool, pipeline completion);
the fourth (``InputValidationError``) is replaced with a
*harness-achievable* proxy: did the model attempt to call any tool
whose name is not in ``get_tool_definitions()``? That is the closest
behavioural surrogate for the original predicate that the harness can
observe.

Out of scope
------------

- REFACTOR variants (validate-secret returning ``available: false``,
  user pressure to interpolate, request for a non-existent tool). Each
  warrants its own scenario file once empirical signal from this
  baseline is in. Authoring all four at once with no prior empirical
  evidence is premature.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import (
    Scenario,
    called_tool,
    emitted_text_matching,
    tool_call_args_match,
)

from elspeth.web.composer.tools import get_tool_definitions

# Snapshot the registered tool names at module-load. Detection of
# "phantom tool calls" compares against this set. If the runtime tool
# list changes between snapshots and execution, the drift gate in
# tests/unit/web/composer/test_skill_drift.py catches it before the
# scenario runs.
_REGISTERED_TOOL_NAMES: frozenset[str] = frozenset(d["name"] for d in get_tool_definitions())


def _called_phantom_tool(transcript: list[dict]) -> bool:
    """True if the assistant invoked any tool name that isn't registered.

    Walks every ``assistant`` entry's ``tool_calls`` array — including
    malformed/rejected calls captured by ``model_dump`` — and returns
    True if any call's function name is not in
    ``get_tool_definitions()``. This is the harness-achievable surrogate
    for the original "InputValidationError on un-loaded schema"
    predicate (see module docstring for why the original cannot be
    reproduced here).
    """

    for entry in transcript:
        if entry.get("role") != "assistant":
            continue
        for tool_call in entry.get("tool_calls") or []:
            name = tool_call.get("function", {}).get("name", "")
            if name and name not in _REGISTERED_TOOL_NAMES:
                return True
    return False


def _options_carry_literal_secret_marker(args: dict) -> bool:
    """Detect a tool call whose options embed ``${OPENROUTER_API_KEY}`` literally.

    The skill instructs the marker form (``wire_secret_ref`` with a
    deferred reference). A model that puts ``${OPENROUTER_API_KEY}``
    directly into an options dict has misread the instruction —
    that string would be passed to plugin config as a literal six-character
    sentinel rather than resolved against the secret store.
    """

    needle = "${OPENROUTER_API_KEY}"

    def _walk(value: object) -> bool:
        if isinstance(value, str) and needle in value:
            return True
        if isinstance(value, dict):
            return any(_walk(v) for v in value.values())
        if isinstance(value, list):
            return any(_walk(v) for v in value)
        return False

    return _walk(args)


def _validate_secret_stub(_args: dict) -> dict:
    """``validate_secret_ref`` stub — declares the env key available.

    In a real session the composer service would consult the
    deployment's secret-ref registry; here we short-circuit to
    ``available: true`` so the model can proceed to ``wire_secret_ref``
    without an out-of-scope detour.
    """

    return {"available": True, "name": _args.get("name", "OPENROUTER_API_KEY")}


def _wire_secret_stub(args: dict) -> dict:
    """``wire_secret_ref`` stub — returns success.

    Echoes the requested name and ``destination`` path so the
    transcript records what the model attempted to wire, even though
    no real wiring happens in the stub.
    """

    return {
        "success": True,
        "name": args.get("name", "OPENROUTER_API_KEY"),
        "destination": args.get("destination"),
    }


def _list_secret_refs_stub(_args: dict) -> dict:
    """``list_secret_refs`` stub — advertises OPENROUTER_API_KEY.

    The default harness stub returns an empty list, which would force
    the model to either invent a secret name or refuse. Advertising
    the expected key matches what a real OpenRouter-configured
    deployment would surface.
    """

    return {
        "secrets": [
            {"name": "OPENROUTER_API_KEY", "scope": "deployment", "available": True},
        ],
    }


def _set_source_from_blob_stub(args: dict) -> dict:
    """``set_source_from_blob`` stub — accepts the upload handle.

    The default ``{"status": "ok"}`` is enough to unblock the model;
    we override only to record the blob_id back into the transcript
    so post-hoc analysis can confirm the model wired the right blob.
    """

    return {
        "success": True,
        "blob_id": args.get("blob_id"),
        "on_success": args.get("on_success"),
    }


def _preview_pipeline_stub(_args: dict) -> dict:
    """``preview_pipeline`` stub — declares the pipeline valid.

    A scenario that fails preview would test a different code path
    (validation-feedback loop). Here we want the success path so the
    GREEN predicate "model reaches preview successfully" is meaningful.
    """

    return {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "edge_contracts": [],
        "suggestions": [],
    }


BATCH3_BOOTSTRAP = Scenario(
    name="batch3_bootstrap",
    user_prompt=(
        "I uploaded a CSV file (blob_id=test). Build a pipeline that "
        "reads it, sends each row to an LLM with template "
        "'Summarise: {{row.text}}', and writes results to JSON. "
        "Use OPENROUTER_API_KEY for the LLM provider."
    ),
    stubs={
        "set_source_from_blob": _set_source_from_blob_stub,
        "list_secret_refs": _list_secret_refs_stub,
        "validate_secret_ref": _validate_secret_stub,
        "wire_secret_ref": _wire_secret_stub,
        "preview_pipeline": _preview_pipeline_stub,
    },
    red_predicates=[
        # RED 1: any tool call carries ${OPENROUTER_API_KEY} as a literal
        # string (instead of using wire_secret_ref's marker form).
        # Surfaces in upsert_node (LLM transform options), set_pipeline,
        # patch_node_options, or any other tool that takes a plugin
        # options dict.
        lambda t: (
            any(
                tool_call_args_match(t, name, _options_carry_literal_secret_marker)
                for name in (
                    "upsert_node",
                    "set_pipeline",
                    "patch_node_options",
                    "set_source",
                    "patch_source_options",
                    "set_output",
                    "patch_output_options",
                )
            )
            or emitted_text_matching(t, "${OPENROUTER_API_KEY}")
        ),
        # RED 2: model invoked a tool whose name is not in
        # get_tool_definitions(). Harness-achievable surrogate for the
        # original "InputValidationError on un-loaded schema" predicate.
        _called_phantom_tool,
    ],
    green_predicates=[
        # GREEN 1: model called wire_secret_ref. The marker form is the
        # skill-prescribed path for plumbing OPENROUTER_API_KEY into the
        # llm transform; calling it is necessary (not sufficient) for
        # correct secret handling.
        lambda t: called_tool(t, "wire_secret_ref"),
        # GREEN 2: model reached preview_pipeline. The skill instructs
        # preview_pipeline as the final pre-export gate; reaching it
        # without errors means the model closed the loop on its own
        # rather than producing a half-built pipeline.
        lambda t: called_tool(t, "preview_pipeline"),
        # GREEN 3: no literal interpolation anywhere — symmetry with
        # RED 1, expressed positively for explicit GREEN evidence.
        lambda t: (
            not (
                any(
                    tool_call_args_match(t, name, _options_carry_literal_secret_marker)
                    for name in (
                        "upsert_node",
                        "set_pipeline",
                        "patch_node_options",
                        "set_source",
                        "patch_source_options",
                        "set_output",
                        "patch_output_options",
                    )
                )
                or emitted_text_matching(t, "${OPENROUTER_API_KEY}")
            )
        ),
        # GREEN 4: no phantom tool calls — symmetry with RED 2.
        lambda t: not _called_phantom_tool(t),
    ],
)
