"""Batch 1 scenario — five mechanical fact-fixes share one transcript.

Findings exercised:

* A1 — ``generate_yaml`` is documented as a tool but isn't in
  ``get_tool_definitions()``.
* A2 — line 80-83 example shows ``gate.routes.true = "high"``
  (unquoted boolean) but line 104 requires string keys.
* A3 — ``After Task 5B it becomes a hard backstop`` leaks roadmap
  state into runtime guidance.
* A4 — optional field marker ``?`` is documented in the runtime grammar
  but absent from the skill's field-format section.
* C7 — ``null`` source is presented as a valid plugin even though it's
  internal-only.

The user prompt asks the LLM to build a CSV-driven pipeline with a
gate routing on a boolean expression and an optional field — five
findings, one composition.  Any RED predicate firing means the
current skill misled the LLM on at least one finding; all GREEN
predicates passing means the candidate edit fixed all of them
without breaking the basic flow.
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
    tried_to_load_schema,
)


def _route_uses_boolean_key(args: dict) -> bool:
    """Detect ``upsert_node`` with a boolean-keyed routes object.

    This is impossible to express in JSON (object keys are strings)
    so the model would have to mis-serialise — but we also catch the
    case where it sends a structurally-confused arg shape.  If the
    model's text output mentions ``routes: {true: ...}`` style YAML,
    the text predicate below catches that separately.
    """

    routes = args.get("routes")
    if not isinstance(routes, dict):
        return False
    return any(isinstance(k, bool) for k in routes)


def _route_uses_string_keys(args: dict) -> bool:
    """Detect well-formed ``upsert_node`` with string ``"true"``/``"false"`` keys."""

    routes = args.get("routes")
    if not isinstance(routes, dict):
        return False
    return "true" in routes and "false" in routes


def _suggested_null_source(transcript: list[dict]) -> bool:
    """Detect any free text or call mentioning the ``null`` source plugin."""

    if emitted_text_matching(transcript, "null source") or emitted_text_matching(transcript, '"null"'):
        return True
    return tool_call_args_match(
        transcript,
        "set_source",
        lambda args: args.get("plugin") == "null",
    )


BATCH1 = Scenario(
    name="batch1_mechanical_fact_fixes",
    user_prompt=(
        "I want to build a pipeline that reads a CSV file at "
        "outputs/sample.csv. Each row has fields: id, priority, price. "
        "The price field is sometimes missing. "
        "If priority is the string 'high', send the row to a sink "
        "called 'priority'; otherwise send it to a sink called "
        "'normal'. Both sinks should write JSON to outputs/. "
        "Build it end-to-end and tell me how to verify it before "
        "running."
    ),
    red_predicates=[
        # A1 — model attempts to load generate_yaml schema or calls it as a tool.
        lambda t: tried_to_load_schema(t, "generate_yaml"),
        # A2 — model produces YAML/JSON with a boolean-typed route key
        # (either via tool args or in free text).
        lambda t: (
            tool_call_args_match(t, "upsert_node", _route_uses_boolean_key)
            or emitted_text_matching(t, "routes:\n    true:")
            or emitted_text_matching(t, "{true:")
        ),
        # A3 — model parrots Task-5B language back to the user.
        lambda t: emitted_text_matching(t, "Task 5B") or emitted_text_matching(t, "after task 5b"),
        # A4 — model handles the optional price field by either
        # inventing wrong syntax or dropping the optionality entirely.
        # Heuristic: presence of "price?: float" (correct grammar)
        # would mean A4 is fine even on RED, so we flip the predicate:
        # "no mention of the ? marker anywhere" indicates the skill
        # failed to teach it.
        lambda t: (
            not (
                emitted_text_matching(t, "price: float?")
                or emitted_text_matching(t, "price?:")
                or emitted_text_matching(t, '"price: float?"')
            )
        ),
        # C7 — model proposes the `null` source plugin as a real option.
        _suggested_null_source,
    ],
    green_predicates=[
        # GREEN: model never calls generate_yaml.
        lambda t: not tried_to_load_schema(t, "generate_yaml"),
        # GREEN: any gate node uses string route keys.
        lambda t: (
            tool_call_args_match(t, "upsert_node", _route_uses_string_keys) or not called_tool(t, "upsert_node")
        ),  # vacuously true if no gate emitted yet
        # GREEN: no Task-5B mention.
        lambda t: not (emitted_text_matching(t, "Task 5B") or emitted_text_matching(t, "task 5b")),
        # GREEN: model uses the ? optional marker for the missing-sometimes field.
        lambda t: emitted_text_matching(t, "price: float?") or emitted_text_matching(t, '"price: float?"'),
        # GREEN: model does not propose the null source.
        lambda t: not _suggested_null_source(t),
    ],
)
