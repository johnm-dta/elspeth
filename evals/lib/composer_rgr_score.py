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

Persisted chat history (`tool_calls` field) is NOT a useful signal —
the composer drops internal tool-call assistant turns before persisting,
so even a successful build shows zero persisted tool calls.

Convergence-bar GREEN criteria (added 2026-05-08 for the
simple-pipeline-convergence program):

  must_have_node_chain_in_order
      List of substrings that must appear in `state.nodes[*].plugin`
      (or node_type) in the given relative order. Stronger than
      must_have_node_kinds_substring_any_of for chains like
      [text, web_scrape, line_explode] where order matters.

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

  max_repair_turns
      Integer. If state has composer_meta.repair_turns_used, that
      value must be <= max_repair_turns. If the field is absent
      (Step 4 not yet shipped, or session bypassed the loop entirely),
      an AMBER reason is added so the absence is visible — never
      silently passing.
"""

from __future__ import annotations

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


def _check_node_chain_in_order(state: Any, chain: list[str]) -> str | None:
    """Return an AMBER reason if `chain` substrings don't appear in order in node plugins.

    Each element in `chain` must be a (case-insensitive) substring of some
    node plugin/type, and the matches must be strictly non-decreasing in index.
    """
    plugins = _node_plugins(state)
    cursor = 0
    for needle in chain:
        needle_l = needle.lower()
        match_idx: int | None = None
        for i in range(cursor, len(plugins)):
            if needle_l in plugins[i]:
                match_idx = i
                break
        if match_idx is None:
            return (
                f"required node chain {chain} not satisfied in order; "
                f"missing '{needle}' after position {cursor}; node plugins: {plugins}"
            )
        cursor = match_idx + 1
    return None


def _source_schema(state: Any) -> dict[str, Any] | None:
    """Return the source-options schema dict, or None if absent."""
    if not isinstance(state, dict):
        return None
    source = state.get("source")
    if not isinstance(source, dict):
        return None
    options = source.get("options")
    if not isinstance(options, dict):
        return None
    schema = options.get("schema")
    if not isinstance(schema, dict):
        return None
    return schema


def _check_observed_columns(state: Any, columns: list[str]) -> str | None:
    """AMBER if source schema is fixed mode AND any expected column is missing.

    observed/flexible modes always pass — they accept extra columns by design.
    fixed mode must list every column in `fields` (case-insensitive name match
    on the bit before any ': type' suffix in the field grammar).
    """
    schema = _source_schema(state)
    if schema is None:
        return f"source schema missing; cannot verify observed columns {columns}"
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
        return (
            f"fixed source schema omits observed columns {missing}; "
            f"declared fields: {sorted(declared_names)}"
        )
    return None


def _check_numeric_handling(state: Any, field: str) -> str | None:
    """AMBER if `field` is used by a numeric op without prior type_coerce or numeric-typed schema.

    Two ways to satisfy:
      1. Source schema (fixed/flexible) declares `field` as int or float.
      2. A type_coerce node has a conversion targeting `field` -> int/float.
    """
    schema = _source_schema(state)
    if isinstance(schema, dict):
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
                if (conv.get("field") or "").lower() == field.lower() and (
                    conv.get("to") or ""
                ).lower() in {"int", "float"}:
                    return None
    return (
        f"field '{field}' has no numeric handling: "
        f"neither source schema declares it int/float nor any type_coerce converts it"
    )


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
            f"composer_meta.repair_turns_used not present in state; "
            f"convergence-bar repair-turn check (max={max_turns}) cannot be verified"
        )
    used = meta["repair_turns_used"]
    if not isinstance(used, int):
        return f"composer_meta.repair_turns_used is not an int: {used!r}"
    if used > max_turns:
        return f"used {used} repair turns (max allowed: {max_turns})"
    return None


def score(
    scenario: dict[str, Any], messages: list[dict[str, Any]], state: Any
) -> dict[str, Any]:
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

    is_valid = bool(state.get("is_valid")) if isinstance(state, dict) else None
    if red.get("must_be_valid", True):
        if is_valid is False:
            red_reasons.append("final composition state has is_valid=false")
        elif state is None or state == "null":
            red_reasons.append("final composition state is null (no committed pipeline)")

    phrase_hits = [p for p in red.get("passivity_phrases", []) if p in final_body]
    if phrase_hits:
        red_reasons.append(f"forbidden passivity phrases in final message: {phrase_hits}")

    credential_hits = [
        p for p in red.get("credential_misnarration_phrases", []) if p in final_body
    ]
    if credential_hits:
        red_reasons.append(
            f"credential misnarration phrases in final message: {credential_hits}"
        )

    amber_reasons: list[str] = []

    if isinstance(state, dict):
        node_plugins = _node_plugins(state)
        node_blob = " ".join(node_plugins)

        kind_groups = green.get("must_have_node_kinds_substring_any_of") or []
        if kind_groups:
            ok = False
            for group in kind_groups:
                if all(needle.lower() in node_blob for needle in group):
                    ok = True
                    break
            if not ok:
                amber_reasons.append(
                    f"no expected node combo present (need one of {kind_groups}); "
                    f"found node plugins {node_plugins}"
                )

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
            "state_output_count": (
                len(state.get("outputs", [])) if isinstance(state, dict) else None
            ),
        },
    }
