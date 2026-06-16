"""Pure scoring function for the composer-rgr harness.

Importable counterpart of `evals/composer-rgr/score.py`. The harness shim
keeps the historical CLI surface (`python3 score.py scenario messages state`)
while this module provides the testable `score(...)` function.

Detection rules (ordered by reliability):
  1. Build-failure sentinels in final assistant content. These are
     server-injected (service.py:_build_runtime_preflight_message) and
     only appear when the model declared completion but the pipeline
     failed preflight. Most reliable RED signal.
  2. is_valid=false on the final composition state. Independent of
     message content, catches cases where the model gets stuck in
     tool loops without ever surfacing a sentinel.
  3. Passivity phrases in final assistant content. Catches the
     'If you want, I can…' / 'Should I…' rationalisation pattern that
     the skill's anti-passivity section explicitly forbids.

Tool sequence shape: with ``include_tool_rows=true`` the harness fetch
returns three role kinds — ``user``, ``assistant``, ``tool``. Assistant
rows that emitted tool calls carry a ``tool_calls`` array in LiteLLM
wire format (``{id, type, function:{name, arguments(JSON-string)}}``).
Each call has a corresponding ``role="tool"`` row whose
``content`` is the canonical-JSON serialisation of the dispatched
ToolResult (``{success: bool, validation: …, …}``) linked back via
``tool_call_id``. Rejection signal: ``content.success == false``.

Tool-sequence rules (added 2026-05-23 for the gov-pages-rate-cool
scenario which targets two failure modes invisible to message-only
scoring):

  max_persisted_tool_calls (red_criteria)
      Integer. RED if the trajectory persisted more than N tool
      invocations (assistant rows' tool_calls union). Catches the
      "model thrashed for many calls" shape — distinct from passivity
      (no calls at all) and from converged success (small number of
      productive calls).

  set_pipeline_rejection_without_success (red_criteria)
      Boolean. RED if at least one set_pipeline invocation returned
      success=false (a rejected mutation) AND no set_pipeline
      invocation ever returned success=true. Catches the "model
      attempted the build but never converged on a valid call",
      distinct from passivity (model never attempted).

  must_discover_schema_before_first_mutation (green_criteria)
      Boolean. AMBER if no get_plugin_schema invocation precedes the
      first state-mutating tool call (set_source, set_output,
      upsert_node, set_pipeline, set_source_from_blob,
      apply_pipeline_recipe). A get_plugin_schema call that fires
      ONLY after a rejection still earns an AMBER — the discover-first
      signal requires the schema lookup before the first mutation.

These rules require the harness to fetch ``messages`` with
``?include_tool_rows=true`` so the role=tool rows carrying the
serialised ToolResult JSON (with the ``success`` discriminator) are
present. Without that, the rules silently no-op as if no tool
sequence is available; ``run_scenario.sh`` was updated in the same
commit to request the audit-grade view.

Convergence-bar GREEN criteria (added 2026-05-08 for the
simple-pipeline-convergence program):

  must_have_node_chain_in_order
      List of plugin/type identifiers that must appear in the flattened workflow sequence:
      source plugin(s), then `state.nodes[*].plugin` (or node_type), then
      `state.outputs[*].plugin`, in the given relative order. Stronger than
      must_have_node_kinds_substring_any_of for chains like
      [csv, web_scrape, line_explode, json] where order matters.

  must_include_observed_columns
      For CSV/text/JSON sources where the operator gave specific
      columns in the prompt: source schema must either be in observed
      or flexible mode, OR be in fixed mode with all listed columns
      present in `fields`. Catches all-row-discard from a fixed
      schema that omits headers.

  must_handle_field_as_numeric
      A field name. Pipeline must contain either a type_coerce node
      converting the field to int/float, OR the source schema declares
      the field as int/float. Catches the numeric-gate-on-string-field
      hazard.

  must_have_output_plugins
      List of output sink plugin names required as a multiset. For example
      ["csv", "csv"] requires at least two CSV sinks and catches JSON sinks
      chosen for a CSV-in/CSV-out routing target.

  max_repair_turns
      Integer. If state has composer_meta.repair_turns_used, that
      value must be <= max_repair_turns. If the field is absent
      (Step 4 not yet shipped, or session bypassed the loop entirely),
      an AMBER reason is added so the absence is visible — never
      silently passing.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _node_plugins(state: Any) -> list[str]:
    """Extract node plugin/type identifiers from a state dict, lower-cased.

    Gates have plugin: null and identify themselves via node_type.
    The legacy "type" key is kept for any scenario that pre-dates the
    node_type/plugin split.
    """
    if not isinstance(state, dict):
        return []
    plugins: list[str] = []
    for n in state.get("nodes") or []:
        if isinstance(n, dict):
            plugins.append((n.get("plugin") or n.get("node_type") or n.get("type") or "").lower())
    return plugins


def _normalise_plugin_token(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _workflow_plugins_for_chain(state: Any) -> list[str]:
    """Extract source, node, and output plugin/type identifiers for chain checks."""
    if not isinstance(state, dict):
        return []

    plugins: list[str] = []
    source = state.get("source")
    if isinstance(source, dict):
        plugin = source.get("plugin")
        if isinstance(plugin, str) and plugin:
            plugins.append(plugin.lower())

    sources = state.get("sources")
    if isinstance(sources, dict):
        for source_spec in sources.values():
            if isinstance(source_spec, dict):
                plugin = source_spec.get("plugin")
                if isinstance(plugin, str) and plugin:
                    plugins.append(plugin.lower())

    plugins.extend(_node_plugins(state))

    for output in state.get("outputs") or []:
        if isinstance(output, dict):
            plugin = output.get("plugin")
            if isinstance(plugin, str) and plugin:
                plugins.append(plugin.lower())
    return plugins


def _check_node_chain_in_order(state: Any, chain: list[str]) -> str | None:
    """Return an AMBER reason if `chain` identifiers don't appear in workflow order.

    Each element in `chain` must be a case-insensitive exact match for some
    source plugin, node plugin/type, or output plugin identifier, and the
    matches must be strictly non-decreasing in index.
    """
    plugins = _workflow_plugins_for_chain(state)
    cursor = 0
    for needle in chain:
        needle_l = _normalise_plugin_token(needle)
        match_idx: int | None = None
        for i in range(cursor, len(plugins)):
            if needle_l == plugins[i]:
                match_idx = i
                break
        if match_idx is None:
            return f"required node chain {chain} not satisfied in order; missing '{needle}' after position {cursor}; workflow plugins: {plugins}"
        cursor = match_idx + 1
    return None


def _source_schemas(state: Any) -> list[dict[str, Any]]:
    """Return every named source's options-schema dict.

    Multi-source composer states (ADR-025) key sources under ``sources`` by
    name; each value carries ``options.schema``. Sources without a schema dict
    are skipped. Returns an empty list when no source declares a schema. (State
    shape is LLM-produced and variable, so the type guards here are boundary
    validation, not defensive suppression of our own bugs.)
    """
    if not isinstance(state, dict):
        return []
    sources = state.get("sources")
    if not isinstance(sources, dict):
        return []
    schemas: list[dict[str, Any]] = []
    for source in sources.values():
        if not isinstance(source, dict):
            continue
        options = source.get("options")
        if not isinstance(options, dict):
            continue
        schema = options.get("schema")
        if isinstance(schema, dict):
            schemas.append(schema)
    return schemas


def _sole_source(state: Any) -> Any:
    """Return the single named source dict, or None unless there is exactly one.

    The per-source option-key checks (hint-uptake scenarios) target "the
    source" and only run against single-source scenarios; with multiple named
    sources the target is ambiguous, so return None and let the check report.
    """
    if not isinstance(state, dict):
        return None
    sources = state.get("sources")
    if not isinstance(sources, dict) or len(sources) != 1:
        return None
    return next(iter(sources.values()))


def _output_plugins(state: Any) -> list[str]:
    """Return output sink plugin identifiers from list or mapping state shapes."""
    if not isinstance(state, dict):
        return []
    outputs = state.get("outputs")
    if isinstance(outputs, dict):
        iterable = list(outputs.values())
    elif isinstance(outputs, list):
        iterable = outputs
    else:
        return []

    plugins: list[str] = []
    for output in iterable:
        if not isinstance(output, dict):
            continue
        plugin = output.get("plugin") or output.get("type")
        if isinstance(plugin, str) and plugin:
            plugins.append(plugin.lower())
    return plugins


def _check_output_plugins(state: Any, expected: list[str]) -> str | None:
    """None when output plugins satisfy the expected lower-case multiset."""
    expected_plugins = [plugin.lower() for plugin in expected if isinstance(plugin, str) and plugin]
    if not expected_plugins:
        return None

    observed = _output_plugins(state)
    observed_counts = Counter(observed)
    missing: list[str] = []
    for plugin, required_count in Counter(expected_plugins).items():
        observed_count = observed_counts[plugin]
        if observed_count < required_count:
            missing.append(f"{plugin} x{required_count} (found {observed_count})")
    if not missing:
        return None
    return f"required output plugins not satisfied: need {missing}; found output plugins {observed}"


def _schema_covers_columns(schema: dict[str, Any], columns: list[str]) -> str | None:
    """None if this single schema covers the columns; else an AMBER reason.

    observed/flexible modes always pass — they accept extra columns by design.
    fixed mode must list every column in `fields` (case-insensitive name match
    on the bit before any ': type' suffix in the field grammar).
    """
    mode = (schema.get("mode") or "").lower()
    if mode in {"observed", "flexible"}:
        return None
    if mode != "fixed":
        return f"source schema mode '{mode}' is not recognised (expected observed/flexible/fixed)"
    fields = schema.get("fields") or []
    if not isinstance(fields, list):
        return f"source schema fields is not a list: {fields!r}"
    declared_names: set[str] = set()
    for entry in fields:
        if isinstance(entry, str):
            name = entry.split(":", 1)[0].strip().lower()
            if name:
                declared_names.add(name)
    missing = [c for c in columns if c.lower() not in declared_names]
    if missing:
        return f"fixed source schema omits observed columns {missing}; declared fields: {sorted(declared_names)}"
    return None


def _check_observed_columns(state: Any, columns: list[str]) -> str | None:
    """AMBER unless some named source's schema covers the expected columns.

    With multiple named sources (ADR-025) the columns are satisfied if ANY
    source covers them; a single-source scenario reduces to checking that one
    source. AMBER reasons from the non-covering sources are surfaced (deduped).
    """
    schemas = _source_schemas(state)
    if not schemas:
        return f"source schema missing; cannot verify observed columns {columns}"
    reasons = [_schema_covers_columns(schema, columns) for schema in schemas]
    if any(reason is None for reason in reasons):
        return None
    return "; ".join(dict.fromkeys(r for r in reasons if r is not None))


def _check_numeric_handling(state: Any, field: str) -> str | None:
    """AMBER if `field` is used by a numeric op without prior type_coerce or numeric-typed schema.

    Two ways to satisfy:
      1. Source schema (fixed/flexible) declares `field` as int or float.
      2. A type_coerce node has a conversion targeting `field` -> int/float.
    """
    for schema in _source_schemas(state):
        fields = schema.get("fields") or []
        if isinstance(fields, list):
            for entry in fields:
                if not isinstance(entry, str):
                    continue
                name, _, type_part = entry.partition(":")
                name = name.strip().lower()
                type_clean = type_part.strip().rstrip("?").lower()
                if name == field.lower() and type_clean in {"int", "float"}:
                    return None
    if isinstance(state, dict):
        for n in state.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            if (n.get("plugin") or "").lower() != "type_coerce":
                continue
            options = n.get("options") or {}
            if not isinstance(options, dict):
                continue
            for conv in options.get("conversions") or []:
                if not isinstance(conv, dict):
                    continue
                if (conv.get("field") or "").lower() == field.lower() and (conv.get("to") or "").lower() in {"int", "float"}:
                    return None
    return f"field '{field}' has no numeric handling: neither source schema declares it int/float nor any type_coerce converts it"


def _check_options_keys(
    container: Any,
    required: list[str],
    forbidden: list[str],
    container_label: str,
) -> str | None:
    """Return an AMBER reason if a container's options dict misses required or contains forbidden keys.

    Supports nested-key paths separated by '.', e.g. 'schema.mode' checks
    options['schema']['mode']. Used by the hint-uptake scenarios authored
    in Phase 1 of composer-jit-hints (csv-headerless-input,
    database-sink-upsert-mode) — see plan file for the rationale.
    """
    if not isinstance(container, dict):
        return f"{container_label} not present on final state"
    options = container.get("options")
    if not isinstance(options, dict):
        return f"{container_label}.options missing or wrong shape"

    def _lookup(path: str) -> tuple[bool, Any]:
        node: Any = options
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return False, None
            node = node[part]
        return True, node

    missing = [k for k in required if not _lookup(k)[0]]
    present = [k for k in forbidden if _lookup(k)[0]]
    if not missing and not present:
        return None
    parts: list[str] = []
    if missing:
        parts.append(f"{container_label}.options missing required keys: {missing}")
    if present:
        parts.append(f"{container_label}.options contains forbidden keys: {present}")
    return "; ".join(parts)


def _check_options_key_value(
    container: Any,
    key_path: str,
    allowed_values: list[Any],
    container_label: str,
) -> str | None:
    """Return an AMBER reason if container.options[key_path] is not in allowed_values.

    Same nested-key path convention as _check_options_keys. None of the
    declared values is a valid AMBER outcome only if the key is present
    AND its value is in the allowed set — the absence of the key is its
    own AMBER reason.
    """
    if not isinstance(container, dict):
        return f"{container_label} not present on final state"
    options = container.get("options")
    if not isinstance(options, dict):
        return f"{container_label}.options missing or wrong shape"
    node: Any = options
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return f"{container_label}.options.{key_path} not set (expected one of {allowed_values})"
        node = node[part]
    if node not in allowed_values:
        return f"{container_label}.options.{key_path} = {node!r} (expected one of {allowed_values})"
    return None


def _check_max_repair_turns(state: Any, max_turns: int) -> str | None:
    """AMBER if composer_meta.repair_turns_used > max_turns OR field absent.

    Field is added by service.py in Step 4 of the convergence program. Until
    that ships, this check produces an explanatory AMBER rather than silently
    passing — absence is itself a finding.
    """
    if not isinstance(state, dict):
        return f"state missing; cannot verify max_repair_turns={max_turns}"
    meta = state.get("composer_meta")
    if not isinstance(meta, dict) or "repair_turns_used" not in meta:
        return (
            f"composer_meta.repair_turns_used not present in state; convergence-bar repair-turn check (max={max_turns}) cannot be verified"
        )
    used = meta["repair_turns_used"]
    if not isinstance(used, int):
        return f"composer_meta.repair_turns_used is not an int: {used!r}"
    if used > max_turns:
        return f"used {used} repair turns (max allowed: {max_turns})"
    return None


# Tool-name set considered "state-mutating" for the discover-first check.
# Sourced from src/elspeth/web/composer/tools/_dispatch.py manifests
# (see _MUTATING_TOOLS membership in the dispatcher) — these are the
# entry points that change CompositionState. Discovery tools
# (get_plugin_schema, list_*, diff_pipeline, preview_pipeline,
# explain_validation_error) do NOT belong here even though they
# successfully complete.
_MUTATING_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "set_pipeline",
        "set_source",
        "set_source_from_blob",
        "upsert_node",
        "set_output",
        "remove_node",
        "remove_output",
        "remove_edge",
        "upsert_edge",
        "patch_source_options",
        "patch_node_options",
        "patch_output_options",
        "apply_pipeline_recipe",
        "set_metadata",
    }
)


def _iter_assistant_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return all tool-call records across all assistant rows, in emission order.

    Each entry is the LiteLLM wire-format ToolCall dict
    (``{id, type, function:{name, arguments(JSON-string)}}``). Assistant
    rows without tool_calls contribute nothing. Multi-tool turns yield
    one entry per call. Used by the tool-sequence scoring rules.
    """
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tcs = msg.get("tool_calls")
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if isinstance(tc, dict):
                calls.append(tc)
    return calls


def _tool_call_name(call: dict[str, Any]) -> str | None:
    """Extract the tool function name from a ToolCall dict, or None if malformed."""
    fn = call.get("function")
    if not isinstance(fn, dict):
        return None
    name = fn.get("name")
    return name if isinstance(name, str) else None


def _tool_result_by_call_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map tool_call_id -> parsed ToolResult dict from role=tool rows.

    Rows whose ``content`` is not parseable JSON or not a dict are
    skipped — they're either an ARG_ERROR envelope
    (``{error_class, error_message}``) or a redaction summary. The
    rejection-vs-success discriminator (``success`` key) only lives on
    the well-formed ToolResult shape, so callers checking for that key
    naturally treat malformed rows as "result unknown" rather than
    silently coerced to success or failure.
    """
    results: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if not isinstance(tcid, str):
            continue
        content_raw = msg.get("content")
        if not isinstance(content_raw, str):
            continue
        try:
            parsed = json.loads(content_raw)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            results[tcid] = parsed
    return results


def _check_set_pipeline_rejection_without_success(messages: list[dict[str, Any]]) -> str | None:
    """Return a RED reason if set_pipeline was attempted but never succeeded.

    "Attempted" = at least one assistant tool_call with name set_pipeline
    whose persisted result row carries success=false. "Succeeded" = any
    set_pipeline whose result row carries success=true. The rule fires
    only when BOTH (1) at least one rejection observed AND (2) zero
    successes observed — that is the empirically-attested failure
    shape (staging session 47cfbb5e-...) where the model retried
    several rejected set_pipeline calls without ever landing a valid
    one.
    """
    calls = _iter_assistant_tool_calls(messages)
    results = _tool_result_by_call_id(messages)
    rejections = 0
    successes = 0
    for call in calls:
        if _tool_call_name(call) != "set_pipeline":
            continue
        tcid = call.get("id")
        if not isinstance(tcid, str):
            continue
        result = results.get(tcid)
        if result is None:
            continue
        success = result.get("success")
        if success is True:
            successes += 1
        elif success is False:
            rejections += 1
    if rejections > 0 and successes == 0:
        return (
            f"set_pipeline was called {rejections} time(s) with all results success=false and 0 successful set_pipeline calls "
            "— the model attempted the build but never converged on a valid construction"
        )
    return None


def _check_max_persisted_tool_calls(messages: list[dict[str, Any]], max_calls: int) -> str | None:
    """Return a RED reason if the trajectory exceeded ``max_calls`` tool invocations.

    Counts assistant-side tool_calls (one entry per LLM-emitted call),
    not role=tool result rows — they're equivalent in number under
    normal operation, but the assistant-side count is what the
    canonical staging trace (13 calls) and the skill's discover-first
    guidance both reason about.
    """
    n = len(_iter_assistant_tool_calls(messages))
    if n > max_calls:
        return f"trajectory persisted {n} tool calls (max allowed: {max_calls})"
    return None


def _check_max_tool_calls_for_green(messages: list[dict[str, Any]], max_calls: int) -> str | None:
    """Return an AMBER reason if the trajectory exceeds the green efficiency target."""
    n = len(_iter_assistant_tool_calls(messages))
    if n > max_calls:
        return f"trajectory persisted {n} tool calls (green target: <= {max_calls})"
    return None


def _check_discover_before_mutation(messages: list[dict[str, Any]]) -> str | None:
    """Return an AMBER reason if no get_plugin_schema precedes the first mutation.

    Walks the assistant tool-call sequence left-to-right. If
    a state-mutating tool fires before any get_plugin_schema, the
    discover-first contract is violated. A discovery that fires only
    AFTER a rejection (e.g. as a panic response) is still a violation
    — the contract is about preparing the call, not recovering from it.

    Returns None (GREEN signal) when:
      - get_plugin_schema appears before the first mutating call, OR
      - there are no mutating calls at all (vacuously satisfied — the
        is_valid criterion catches the empty-build failure shape).
    """
    calls = _iter_assistant_tool_calls(messages)
    first_schema_idx: int | None = None
    first_mutation_idx: int | None = None
    first_mutation_name: str | None = None
    for idx, call in enumerate(calls):
        name = _tool_call_name(call)
        if name is None:
            continue
        if name == "get_plugin_schema" and first_schema_idx is None:
            first_schema_idx = idx
        if name in _MUTATING_TOOL_NAMES and first_mutation_idx is None:
            first_mutation_idx = idx
            first_mutation_name = name
    if first_mutation_idx is None:
        return None
    if first_schema_idx is not None and first_schema_idx < first_mutation_idx:
        return None
    if first_schema_idx is None:
        return (
            f"first mutating call ({first_mutation_name!r} at index {first_mutation_idx}) "
            "was not preceded by any get_plugin_schema discovery"
        )
    return (
        f"first mutating call ({first_mutation_name!r} at index {first_mutation_idx}) "
        f"preceded the first get_plugin_schema (at index {first_schema_idx}) — "
        "discovery as a post-rejection recovery does not satisfy discover-first"
    )


def score(scenario: dict[str, Any], messages: list[dict[str, Any]], state: Any) -> dict[str, Any]:
    """Score a captured composer-rgr run. Pure function — no I/O.

    Returns a dict with verdict (RED/AMBER/GREEN), red_reasons, amber_reasons,
    and stats. Verdict precedence is RED > AMBER > GREEN.
    """
    red = scenario["red_criteria"]
    green = scenario["green_criteria"]

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    final = assistant_msgs[-1] if assistant_msgs else {"content": ""}
    final_body = (final.get("content") or "").lower()

    red_reasons: list[str] = []

    sentinel_hits = [s for s in red.get("build_failure_sentinels", []) if s in final_body]
    if sentinel_hits:
        red_reasons.append(f"build-failure sentinel(s) present in final message: {sentinel_hits}")

    # Degenerate-state RED floor — UNCONDITIONAL, outside the ``must_be_valid``
    # gate. A scenario may relax ``must_be_valid`` to false to accept a
    # structurally-complete-but-not-yet-runnable terminal state (e.g. a
    # model-bearing pipeline whose mandatory interpretation reviews are
    # SURFACED but pending out-of-loop resolution — see the csv-classifier
    # convergence scenario / Branch B). That relaxation must NOT also let an
    # empty or null composition through as non-RED. An empty/null pipeline is a
    # genuine build failure regardless of ``must_be_valid``. The empty-dict
    # condition mirrors ``web/execution/validation.py``'s own empty-pipeline
    # short-circuit (source is None and not nodes and not outputs) so this floor
    # cannot drift from the engine's definition of "empty".
    if state is None or state == "null":
        red_reasons.append("final composition state is null (no committed pipeline)")
    elif isinstance(state, dict) and state.get("source") is None and not state.get("nodes") and not state.get("outputs"):
        red_reasons.append("composition state is structurally empty (no source, nodes, or outputs)")

    is_valid = bool(state.get("is_valid")) if isinstance(state, dict) else None
    if red.get("must_be_valid", True) and is_valid is False:
        # The null/empty case is already handled unconditionally by the
        # degenerate-state floor above (and would otherwise double-append its
        # red_reason here), so this gate only needs the is_valid=false branch.
        red_reasons.append("final composition state has is_valid=false")

    phrase_hits = [p for p in red.get("passivity_phrases", []) if p in final_body]
    if phrase_hits:
        red_reasons.append(f"forbidden passivity phrases in final message: {phrase_hits}")

    credential_hits = [p for p in red.get("credential_misnarration_phrases", []) if p in final_body]
    if credential_hits:
        red_reasons.append(f"credential misnarration phrases in final message: {credential_hits}")

    # Tool-sequence RED rules (require ?include_tool_rows=true on the fetch;
    # see module docstring). Each helper returns None when the rule doesn't
    # apply (e.g. no tool rows fetched, no mutations attempted) so omission
    # of the flag from the scenario is the correct no-op.
    max_tool_calls = red.get("max_persisted_tool_calls")
    if isinstance(max_tool_calls, int):
        too_many_reason = _check_max_persisted_tool_calls(messages, max_tool_calls)
        if too_many_reason is not None:
            red_reasons.append(too_many_reason)

    if red.get("set_pipeline_rejection_without_success") is True:
        rejection_reason = _check_set_pipeline_rejection_without_success(messages)
        if rejection_reason is not None:
            red_reasons.append(rejection_reason)

    amber_reasons: list[str] = []

    if green.get("must_discover_schema_before_first_mutation") is True:
        discover_reason = _check_discover_before_mutation(messages)
        if discover_reason is not None:
            amber_reasons.append(discover_reason)

    max_tool_calls_for_green = green.get("max_tool_calls_for_green")
    if isinstance(max_tool_calls_for_green, int):
        inefficient_reason = _check_max_tool_calls_for_green(messages, max_tool_calls_for_green)
        if inefficient_reason is not None:
            amber_reasons.append(inefficient_reason)

    if isinstance(state, dict):
        node_plugins = _node_plugins(state)

        kind_groups = green.get("must_have_node_kinds_substring_any_of") or []
        if kind_groups:
            ok = False
            node_plugin_set = set(node_plugins)
            for group in kind_groups:
                required_tokens = [_normalise_plugin_token(needle) for needle in group]
                required_tokens = [token for token in required_tokens if token]
                if required_tokens and all(token in node_plugin_set for token in required_tokens):
                    ok = True
                    break
            if not ok:
                amber_reasons.append(f"no expected node combo present (need one of {kind_groups}); found node plugins {node_plugins}")

        chain = green.get("must_have_node_chain_in_order")
        if isinstance(chain, list) and chain:
            chain_reason = _check_node_chain_in_order(state, list(chain))
            if chain_reason is not None:
                amber_reasons.append(chain_reason)

        observed_cols = green.get("must_include_observed_columns")
        if isinstance(observed_cols, list) and observed_cols:
            cols_reason = _check_observed_columns(state, list(observed_cols))
            if cols_reason is not None:
                amber_reasons.append(cols_reason)

        numeric_field = green.get("must_handle_field_as_numeric")
        if isinstance(numeric_field, str) and numeric_field:
            num_reason = _check_numeric_handling(state, numeric_field)
            if num_reason is not None:
                amber_reasons.append(num_reason)

        outputs = state.get("outputs")
        out_count = len(outputs) if isinstance(outputs, (list, dict)) else 0
        min_outputs = green.get("must_have_outputs_min", 0)
        if out_count < min_outputs:
            amber_reasons.append(f"only {out_count} outputs (need >= {min_outputs})")

        output_plugins = green.get("must_have_output_plugins")
        if isinstance(output_plugins, list) and output_plugins:
            output_plugin_reason = _check_output_plugins(state, output_plugins)
            if output_plugin_reason is not None:
                amber_reasons.append(output_plugin_reason)

        # Hint-uptake asserters (Phase 1 of composer-jit-hints).
        # These check that the LLM applied a discovery-time hint
        # without being told. Required/forbidden keys on the
        # source/output options encode the hint's intended effect.
        source_keys_req = green.get("must_have_options_keys_for_source") or []
        source_keys_forb = green.get("must_not_have_options_keys_for_source") or []
        if source_keys_req or source_keys_forb:
            r = _check_options_keys(
                _sole_source(state),
                list(source_keys_req),
                list(source_keys_forb),
                container_label="source",
            )
            if r is not None:
                amber_reasons.append(r)
        # By-name output (sink) option-key checks: green spec is a list
        # of {sink_name, required, forbidden} entries so a scenario can
        # constrain multiple sinks if needed.
        for entry in green.get("must_have_options_keys_for_output") or []:
            sink_name = entry.get("sink_name")
            if not isinstance(sink_name, str):
                continue
            outputs_list = state.get("outputs")
            target = None
            if isinstance(outputs_list, list):
                target = next((o for o in outputs_list if isinstance(o, dict) and o.get("name") == sink_name), None)
            elif isinstance(outputs_list, dict):
                target = outputs_list.get(sink_name)
            r = _check_options_keys(
                target,
                list(entry.get("required") or ()),
                list(entry.get("forbidden") or ()),
                container_label=f"output[{sink_name!r}]",
            )
            if r is not None:
                amber_reasons.append(r)
        # By-name output option-value checks.
        for entry in green.get("must_have_options_value_for_output") or []:
            sink_name = entry.get("sink_name")
            key_path = entry.get("key")
            allowed = entry.get("allowed_values") or []
            if not (isinstance(sink_name, str) and isinstance(key_path, str)):
                continue
            outputs_list = state.get("outputs")
            target = None
            if isinstance(outputs_list, list):
                target = next((o for o in outputs_list if isinstance(o, dict) and o.get("name") == sink_name), None)
            elif isinstance(outputs_list, dict):
                target = outputs_list.get(sink_name)
            r = _check_options_key_value(target, key_path, list(allowed), container_label=f"output[{sink_name!r}]")
            if r is not None:
                amber_reasons.append(r)

    max_repair = green.get("max_repair_turns")
    if isinstance(max_repair, int):
        repair_reason = _check_max_repair_turns(state, max_repair)
        if repair_reason is not None:
            amber_reasons.append(repair_reason)

    verdict = "RED" if red_reasons else ("GREEN" if not amber_reasons else "AMBER")

    return {
        "verdict": verdict,
        "red_reasons": red_reasons,
        "amber_reasons": amber_reasons,
        "stats": {
            "assistant_message_count": len(assistant_msgs),
            "final_content_chars": len(final_body),
            "final_content_preview": (final.get("content") or "")[:300],
            "is_valid": is_valid,
            "state_node_count": len(state.get("nodes", [])) if isinstance(state, dict) else None,
            "state_output_count": (len(state.get("outputs", [])) if isinstance(state, dict) else None),
            "persisted_tool_call_count": len(_iter_assistant_tool_calls(messages)),
        },
    }
