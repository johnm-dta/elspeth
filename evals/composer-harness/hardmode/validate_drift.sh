#!/usr/bin/env bash
# evals/composer-harness/hardmode/validate_drift.sh — Channel 1 drift detector.
#
# Usage: validate_drift.sh <run_dir>
#
# For a captured scenario run, scans every user turn for tokens that signal
# vocabulary acquisition from the composer:
#
#   * snake_case identifiers ([a-z]+(_[a-z0-9]+)+)        — never domain-natural English
#   * MCP composer-tool literal names                      — set_pipeline, apply_pipeline_recipe, ...
#   * Plugin kind literal names                            — csv_source, web_scrape, type_coerce, ...
#
# For each hit, classifies whether the token was introduced by the composer
# in the previous response (composer_adopted, the load-bearing drift signal)
# or surfaced first in user text (user_introduced, a scenario-design flag).
#
# Per-persona ceiling is parsed from the persona spec's competence_ceiling
# marker. Personas with `competence_ceiling: **none**` are EXEMPT from
# Channel 1 entirely (Dev), but Channel 2 (judge_persona.sh) still applies.
#
# Why two channels: token-match has high precision but low recall. It catches
# echoed plumbing terms ("yes, please add the type_coerce") but misses tonal
# drift (Linda producing confident architectural sentences without using a
# single banned token). Channel 2 covers that gap with an LLM judge.
#
# Output: writes <run_dir>/drift.json with per-token events, per-turn rollup,
# and a verdict ("in-character" | "drift" | "exempt").
#
# Exit codes:
#   0   verdict written successfully (does NOT fail on drift — verdict is data,
#       not pass/fail. Aggregation rolls verdicts up; CI gating is done at the
#       cohort level, not per-run, so a single drifty run doesn't abort a sweep.)
#   64  usage error
#   67  prerequisite missing (run_dir, scenario.json, persona spec)
#   70  persona spec invalid (missing competence_ceiling marker)

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

if (( $# != 1 )); then
  evals_die 64 "usage: validate_drift.sh <run_dir>"
fi
run_dir=$1
[[ -d "$run_dir" ]] || evals_die 67 "run dir not found: $run_dir"
[[ -s "$run_dir/scenario.json" ]] || evals_die 67 "scenario.json missing in $run_dir"

evals_require_tools

persona_id=$(jq -r '.persona' "$run_dir/scenario.json")
persona_spec="$HARNESS_ROOT/personas/${persona_id}.md"
[[ -f "$persona_spec" ]] || evals_die 67 "persona spec not found: $persona_spec"

python3 - "$run_dir" "$persona_spec" <<'PY'
"""Channel 1 drift detector.

Reads a captured scenario run plus its persona spec, emits drift.json.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
spec_path = pathlib.Path(sys.argv[2])
spec_text = spec_path.read_text()

# ---------- Parse competence_ceiling literal ----------
# Accepts: `competence_ceiling: **amateur**`, `competence_ceiling: **none**`, etc.
# Also accepts inline forms like "Linda's competence_ceiling: **amateur**.".
CEILING_RE = re.compile(
    r"competence_ceiling[:\s]*\*\*([a-z_]+)\*\*",
    re.IGNORECASE,
)
m = CEILING_RE.search(spec_text)
if not m:
    sys.stderr.write(
        f"persona spec {spec_path.name} has no competence_ceiling marker — "
        f"expected `competence_ceiling: **<value>**` (e.g. **amateur**, **none**).\n"
    )
    sys.exit(70)

ceiling = m.group(1).lower()
EXEMPT = ceiling == "none"

# ---------- Token classes ----------
# Snake_case: a lowercase token with at least one internal underscore.
# This regex never matches domain-natural English ("transform", "schema",
# "validate" all single tokens). It DOES match column-name conventions like
# `customer_id`, but those are explicitly classified as `ambiguous_snake_case`
# below and routed to the LLM judge for adjudication.
SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# MCP composer-tool names (without the mcp__elspeth-composer__ prefix).
# Static list; new tools must be added here when added to the MCP server.
MCP_TOOL_NAMES = frozenset({
    "apply_pipeline_recipe",
    "clear_source",
    "delete_session",
    "diff_pipeline",
    "explain_validation_error",
    "generate_yaml",
    "get_expression_grammar",
    "get_pipeline_state",
    "get_plugin_assistance",
    "get_plugin_schema",
    "inspect_source",
    "list_models",
    "list_recipes",
    "list_sessions",
    "list_sinks",
    "list_sources",
    "list_transforms",
    "load_session",
    "new_session",
    "patch_node_options",
    "patch_output_options",
    "patch_source_options",
    "preview_pipeline",
    "remove_edge",
    "remove_node",
    "remove_output",
    "save_session",
    "set_metadata",
    "set_output",
    "set_pipeline",
    "set_source",
    "upsert_edge",
    "upsert_node",
})

# Plugin kind literal strings — extracted from examples/ and the plugin
# catalog. Kept explicit (rather than discovered at runtime) so the detector
# behaves identically across staging deploys with different plugin packs
# loaded.
PLUGIN_KIND_NAMES = frozenset({
    # Sources
    "csv_source", "text_source", "json_source", "blob_source",
    # Transforms — generic
    "type_coerce", "value_transform", "json_explode", "line_explode",
    "deaggregation", "checkpoint",
    # Transforms — LLM
    "llm", "openrouter_llm", "azure_openai_llm", "rate_limited_llm",
    # Transforms — content / web
    "web_scrape", "azure_content_safety", "chroma_rag",
    # Gates / routing
    "threshold_gate", "boolean_gate", "route_to_sink", "fork_to_paths",
    "explicit_routing", "deep_routing", "error_routing", "boolean_routing",
    # Aggregation / batch
    "batch_aggregation", "coalesce", "fork_coalesce",
    # Sinks
    "json_sink", "jsonl_sink", "csv_sink", "database_sink", "azure_blob_sink",
    "landscape_journal",
})

# Column-name allowlist: snake_case tokens that are ambient in domain English
# even for amateur personas (CSV column conventions). These are reported but
# do NOT count toward the verdict. Operator can extend in scenarios where
# the input data uses additional column names.
COLUMN_NAME_ALLOWLIST = frozenset({
    "customer_id", "user_id", "first_name", "last_name", "full_name",
    "email_address", "phone_number", "date_of_birth", "zip_code",
    "postal_code", "street_address", "created_at", "updated_at",
})


def detect_tokens(text: str) -> dict[str, list[str]]:
    """Categorise tokens found in text by class.

    Each token is reported in exactly one bucket — buckets are checked in
    priority order so MCP/plugin literal hits aren't double-counted as
    generic snake_case.
    """
    text_lower = text.lower()
    found = {
        "mcp_tool": [],
        "plugin_kind": [],
        "snake_case": [],
        "column_name": [],
    }
    seen: set[str] = set()

    for tok in MCP_TOOL_NAMES:
        if re.search(rf"\b{re.escape(tok)}\b", text_lower) and tok not in seen:
            found["mcp_tool"].append(tok)
            seen.add(tok)
    for tok in PLUGIN_KIND_NAMES:
        if re.search(rf"\b{re.escape(tok)}\b", text_lower) and tok not in seen:
            found["plugin_kind"].append(tok)
            seen.add(tok)

    # Generic snake_case sweep — strip out tokens already classified above.
    for tok in SNAKE_CASE_RE.findall(text_lower):
        if tok in seen:
            continue
        if tok in COLUMN_NAME_ALLOWLIST:
            found["column_name"].append(tok)
        else:
            found["snake_case"].append(tok)
        seen.add(tok)

    # De-dupe each bucket while preserving order.
    return {k: list(dict.fromkeys(v)) for k, v in found.items()}


def load_composer_response(turn_n: int) -> str:
    """Best-effort read of msg.t<N-1>.resp.json into a single string.

    The harness stores the composer reply as the JSON response body of
    POST /api/sessions/<sid>/messages. Composer text content is at
    .message.content, which may be a string or a list-of-segments. Both
    shapes are flattened to plain text here.
    """
    p = run_dir / f"msg.t{turn_n - 1}.resp.json"
    if not p.exists():
        return ""
    try:
        d = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    msg = d.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict):
                txt = seg.get("text") or seg.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return ""


# ---------- Walk turns ----------
turn_files = sorted(
    run_dir.glob("turn*.user.txt"),
    key=lambda p: int(re.search(r"turn(\d+)", p.name).group(1)),
)

events: list[dict] = []
prior_user_text_lower = ""

for tf in turn_files:
    turn_n = int(re.search(r"turn(\d+)", tf.name).group(1))
    text = tf.read_text()
    classified = detect_tokens(text)

    for kind, tokens in classified.items():
        for tok in tokens:
            already_seen_in_user = tok in prior_user_text_lower
            if turn_n >= 2:
                composer_text_lower = load_composer_response(turn_n).lower()
                introduced_by_composer = (
                    bool(composer_text_lower)
                    and tok in composer_text_lower
                    and not already_seen_in_user
                )
            else:
                introduced_by_composer = False

            if already_seen_in_user:
                category = "user_repeat"
            elif introduced_by_composer:
                category = "composer_adopted"
            else:
                category = "user_introduced"

            events.append({
                "turn": turn_n,
                "token": tok,
                "kind": kind,
                "category": category,
            })

    prior_user_text_lower += "\n" + text.lower()

# ---------- Per-persona verdict ----------
# Tolerance table: number of `composer_adopted` events tolerated before
# the verdict tips to "drift". `none` is sentinel for exempt.
TOLERANCE = {
    "amateur": 0,
    "amateur_overconfident": 0,
    "journeyman_academic": 0,
    "journeyman": 0,
    "expert": 0,    # reserved; same strict bar as journeyman by default
    "none": -1,     # exempt sentinel
}
ceiling_tolerance = TOLERANCE.get(ceiling)
if ceiling_tolerance is None:
    sys.stderr.write(
        f"unknown competence_ceiling value '{ceiling}' in {spec_path.name} — "
        f"expected one of {sorted(TOLERANCE)}\n"
    )
    sys.exit(70)

# `column_name` events never count toward verdict (ambient in domain English).
# `user_introduced` events surface scenario-design issues but don't fail the
# verdict (the persona didn't absorb anything from the composer; the prompt
# itself may have been impure).
def is_actionable(event: dict) -> bool:
    if event["kind"] == "column_name":
        return False
    return event["category"] == "composer_adopted"


actionable = [e for e in events if is_actionable(e)]

if EXEMPT:
    verdict = "exempt"
elif len(actionable) > ceiling_tolerance:
    verdict = "drift"
else:
    verdict = "in-character"

# Scenario-design flags: tokens introduced by the user that probably
# shouldn't appear in an amateur persona's mouth (e.g. opening prompt
# leaked plugin kind names). Surfaced for review even when verdict is
# in-character.
scenario_flags = [
    e for e in events
    if not EXEMPT
    and e["category"] == "user_introduced"
    and e["kind"] in {"mcp_tool", "plugin_kind"}
]

report = {
    "persona": spec_path.stem,
    "competence_ceiling": ceiling,
    "exempt_from_channel_1": EXEMPT,
    "ceiling_tolerance": ceiling_tolerance,
    "events": events,
    "actionable_drift_events": actionable,
    "scenario_design_flags": scenario_flags,
    "summary": {
        "turn_count": len(turn_files),
        "total_token_events": len(events),
        "actionable_drift_count": len(actionable),
        "scenario_design_flag_count": len(scenario_flags),
        "verdict": verdict,
    },
    "note": (
        "Channel 1 is structural-token detection only. Marcus-style drift "
        "(adopting the composer's correct meaning of pseudo-technical "
        "vocabulary like 'schema' or 'trigger') and tonal drift (Dev "
        "becoming polite, Sarah dropping narrative voice) are NOT detectable "
        "by this channel. judge_persona.sh (Channel 2) covers that gap."
    ),
}

(run_dir / "drift.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["summary"], indent=2))
PY
