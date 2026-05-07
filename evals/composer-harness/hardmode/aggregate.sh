#!/usr/bin/env bash
# evals/composer-harness/hardmode/aggregate.sh — cross-scenario summary.
#
# Usage: aggregate.sh [<runs_dir>]
#   <runs_dir> defaults to $ELSPETH_EVAL_RUNS_DIR or the newest evals/composer-harness/runs/*-hardmode/
#
# Inputs: one or more <runs_dir>/<scenario_id>/ledger.json
#
# Outputs (in <runs_dir>):
#   aggregate.json   — list of per-scenario summary objects
#   SCORECARD.md     — human-readable cross-scenario table

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$HARNESS_DIR"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

evals_require_tools

runs_root="${1:-${ELSPETH_EVAL_RUNS_DIR:-}}"
if [[ -z "$runs_root" ]]; then
  runs_root=$(ls -1d "$HARNESS_ROOT"/runs/*-hardmode 2>/dev/null | sort -r | head -1 || true)
fi
[[ -d "$runs_root" ]] || evals_die 67 "runs dir not found: $runs_root"

evals_log INFO "aggregating ledgers under $runs_root"

python3 - "$runs_root" <<'PY'
import json, math, pathlib, sys

runs = pathlib.Path(sys.argv[1])
ledgers = sorted(runs.glob("*/ledger.json"))
agg = []
aggregate_errors = []
summary_usage = {
    "prompt_tokens": 0,
    "cached_prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "models": set(),
    "token_usage_available_scenarios": 0,
    "token_usage_unavailable_scenarios": 0,
}
total_artifact_errors = 0
summary_cost = {
    "actual_usd": 0.0,
    "cost_available_scenarios": 0,
    "cost_unavailable_scenarios": 0,
    "costed_call_count": 0,
    "sources": set(),
}

def as_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else 0

def as_cost(value):
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0 else None

for p in ledgers:
    try:
        L = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        aggregate_errors.append({
            "kind": "malformed_ledger",
            "path": str(p.relative_to(runs)),
            "error": str(exc),
        })
        continue
    raw_provider_usage = L.get("provider_usage")
    provider_usage = raw_provider_usage if isinstance(raw_provider_usage, dict) else {}
    token_fields = ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "total_tokens")
    legacy_usage_available = any(as_int(provider_usage.get(field)) for field in token_fields)
    token_usage_available = bool(provider_usage.get("token_usage_available"))
    if "token_usage_available" not in provider_usage:
        token_usage_available = legacy_usage_available
    usage = {
        "prompt_tokens": as_int(provider_usage.get("prompt_tokens")),
        "cached_prompt_tokens": as_int(provider_usage.get("cached_prompt_tokens")),
        "completion_tokens": as_int(provider_usage.get("completion_tokens")),
        "total_tokens": as_int(provider_usage.get("total_tokens")),
        "models": sorted(m for m in (provider_usage.get("models") or []) if isinstance(m, str)),
        "token_usage_available": token_usage_available,
        "source": provider_usage.get("source") or ("diagnostics" if token_usage_available else "not_available"),
    }
    if usage["token_usage_available"]:
        for field in token_fields:
            summary_usage[field] += usage[field]
        summary_usage["models"].update(usage["models"])
        summary_usage["token_usage_available_scenarios"] += 1
    else:
        summary_usage["token_usage_unavailable_scenarios"] += 1
    artifact_errors = L.get("artifact_collection_errors") or []
    artifact_error_count = len(artifact_errors)
    total_artifact_errors += artifact_error_count
    raw_cost = L.get("cost")
    cost_in = raw_cost if isinstance(raw_cost, dict) else {}
    actual_usd = as_cost(cost_in.get("actual_usd"))
    cost_available = actual_usd is not None
    cost = {
        "actual_usd": actual_usd,
        "source": cost_in.get("source") or ("composer_llm_audit.response_usage.cost" if cost_available else "not_available"),
        "cost_available": cost_available,
        "costed_call_count": as_int(cost_in.get("costed_call_count")),
    }
    if cost_available:
        summary_cost["actual_usd"] += actual_usd
        summary_cost["cost_available_scenarios"] += 1
        summary_cost["costed_call_count"] += cost["costed_call_count"]
        if isinstance(cost["source"], str):
            summary_cost["sources"].add(cost["source"])
    else:
        summary_cost["cost_unavailable_scenarios"] += 1
    agg.append({
        "scenario_id": L.get("scenario_id"),
        "persona": L.get("persona"),
        "task_class": L.get("task_class"),
        "user_turns": L.get("user_turn_count"),
        "is_valid": L.get("is_valid"),
        "ran_engine": L.get("ran_engine"),
        "run_status": L.get("run_status"),
        "poll_status": L.get("poll_status"),
        "rows_processed": L.get("rows_processed"),
        "rows_succeeded": L.get("rows_succeeded"),
        "rows_routed_success": L.get("rows_routed_success"),
        "total_wall_seconds": round(L.get("total_wall_seconds") or 0, 1),
        "total_tool_calls": L.get("total_tool_calls"),
        "total_in_loop_recoveries": L.get("total_in_loop_recoveries"),
        "clarifying_keyword_turns": L.get("clarifying_keyword_turns"),
        "limit_keyword_turns": L.get("limit_keyword_turns"),
        "failed_validate_checks": L.get("failed_validate_checks"),
        "run_error": L.get("run_error"),
        "artifact_collection_error_count": artifact_error_count,
        "artifact_collection_errors": artifact_errors,
        "provider_usage": usage,
        "cost": cost,
    })

agg.sort(key=lambda r: (r.get("persona") or "", r.get("task_class") or "", r.get("scenario_id") or ""))
(runs / "aggregate.json").write_text(json.dumps(agg, indent=2))
(runs / "aggregate_errors.json").write_text(json.dumps(aggregate_errors, indent=2))
(runs / "aggregate_summary.json").write_text(json.dumps({
    "scenario_count": len(agg),
    "total_wall_seconds": round(sum(r.get("total_wall_seconds") or 0 for r in agg), 1),
    "artifact_collection_error_count": total_artifact_errors,
    "aggregate_error_count": len(aggregate_errors),
    "provider_usage": {
        "prompt_tokens": summary_usage["prompt_tokens"] if summary_usage["token_usage_available_scenarios"] else None,
        "cached_prompt_tokens": (
            summary_usage["cached_prompt_tokens"] if summary_usage["token_usage_available_scenarios"] else None
        ),
        "completion_tokens": summary_usage["completion_tokens"] if summary_usage["token_usage_available_scenarios"] else None,
        "total_tokens": summary_usage["total_tokens"] if summary_usage["token_usage_available_scenarios"] else None,
        "models": sorted(summary_usage["models"]),
        "token_usage_available_scenarios": summary_usage["token_usage_available_scenarios"],
        "token_usage_unavailable_scenarios": summary_usage["token_usage_unavailable_scenarios"],
    },
    "cost": {
        "actual_usd": round(summary_cost["actual_usd"], 10) if summary_cost["cost_available_scenarios"] else None,
        "source": "+".join(sorted(summary_cost["sources"])) if summary_cost["sources"] else "not_available",
        "cost_available_scenarios": summary_cost["cost_available_scenarios"],
        "cost_unavailable_scenarios": summary_cost["cost_unavailable_scenarios"],
        "costed_call_count": summary_cost["costed_call_count"],
    },
}, indent=2))

# Human-readable scorecard.
def fmt_outcome(r):
    if r["ran_engine"]:
        st = r.get("run_status") or "?"
        if st == "completed":          return "engine OK"
        if st == "completed_with_failures": return "engine partial"
        if st == "failed":             return f"engine failed ({(r.get('run_error') or '')[:48]}...)"
        if st == "empty":              return "engine empty"
        if st == "interrupted":        return "engine interrupted"
        return f"engine {st}"
    elif r.get("is_valid") is False:
        checks = r.get("failed_validate_checks") or []
        return f"validate failed ({', '.join(checks[:3])})"
    elif r.get("user_turns", 0) >= 5 and not r.get("is_valid"):
        return "convergence-budget exhausted"
    else:
        return "did not reach engine"

lines = [
    "# Hard-Mode Eval Scorecard",
    "",
    f"_Generated from `{runs.name}/aggregate.json` ({len(agg)} scenarios)._",
    "",
    "| Scenario | Persona | Class | Turns | Wall (s) | Tokens | Cost | Artifact errors | Outcome | Rows ok / proc |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in agg:
    rows = f"{r.get('rows_succeeded')}/{r.get('rows_processed')}" if r.get("ran_engine") else "—"
    usage = r.get("provider_usage") or {}
    tokens = usage.get("total_tokens") if usage.get("token_usage_available") else "—"
    cost = r.get("cost") or {}
    cost_cell = f"${cost.get('actual_usd'):.4f}" if cost.get("cost_available") else "—"
    lines.append("| {sid} | {p} | {c} | {t} | {w} | {tokens} | {cost} | {artifact_errors} | {o} | {r} |".format(
        sid=r["scenario_id"] or "?",
        p=r["persona"] or "?",
        c=r["task_class"] or "?",
        t=r["user_turns"] or "?",
        w=r["total_wall_seconds"] or 0,
        tokens=tokens,
        cost=cost_cell,
        artifact_errors=r.get("artifact_collection_error_count") or 0,
        o=fmt_outcome(r),
        r=rows,
    ))

lines += [
    "",
    "## Run Summary",
    "",
    "- Provider tokens: "
    + (
        f"{summary_usage['total_tokens']} total "
        f"({summary_usage['prompt_tokens']} prompt, {summary_usage['completion_tokens']} completion, "
        f"{summary_usage['cached_prompt_tokens']} cached prompt)"
        if summary_usage["token_usage_available_scenarios"]
        else "not available"
    ),
    f"- Provider usage coverage: {summary_usage['token_usage_available_scenarios']} available, "
    f"{summary_usage['token_usage_unavailable_scenarios']} unavailable",
    f"- Models observed: {', '.join(sorted(summary_usage['models'])) if summary_usage['models'] else '—'}",
    f"- Artifact collection errors: {total_artifact_errors}",
    "- Cost: "
    + (
        f"${summary_cost['actual_usd']:.4f} from {summary_cost['costed_call_count']} provider-priced calls"
        if summary_cost["cost_available_scenarios"]
        else "not available from harness metadata"
    ),
    f"- Cost coverage: {summary_cost['cost_available_scenarios']} available, "
    f"{summary_cost['cost_unavailable_scenarios']} unavailable",
]

lines += ["", "## Persona-class matrix", ""]
personas = sorted({r["persona"] for r in agg if r.get("persona")})
classes = sorted({r["task_class"] for r in agg if r.get("task_class")})
lines.append("| | " + " | ".join(classes) + " |")
lines.append("|---|" + "|".join(["---"] * len(classes)) + "|")
for p in personas:
    cells = []
    for c in classes:
        match = [r for r in agg if r["persona"] == p and r["task_class"] == c]
        if not match:
            cells.append("—")
        else:
            outs = [fmt_outcome(r) for r in match]
            cells.append("<br>".join(outs))
    lines.append(f"| **{p}** | " + " | ".join(cells) + " |")

if aggregate_errors:
    lines += ["", "## Aggregate errors", ""]
    lines.append("| Path | Kind | Error |")
    lines.append("|---|---|---|")
    for err in aggregate_errors:
        lines.append(f"| {err['path']} | {err['kind']} | {str(err.get('error') or '')[:120]} |")

(runs / "SCORECARD.md").write_text("\n".join(lines) + "\n")
print(f"wrote {runs/'aggregate.json'}")
print(f"wrote {runs/'aggregate_errors.json'}")
print(f"wrote {runs/'aggregate_summary.json'}")
print(f"wrote {runs/'SCORECARD.md'}")
PY
