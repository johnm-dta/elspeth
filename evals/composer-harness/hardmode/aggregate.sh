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
summary_fidelity = {
    "channel1_in_character": 0,
    "channel1_drift": 0,
    "channel1_exempt": 0,
    "channel1_unavailable": 0,
    "channel2_in_character": 0,
    "channel2_out_of_character": 0,
    "channel2_skipped": 0,
    "channel2_unavailable": 0,
    "total_actionable_drift_events": 0,
    "total_judge_drift_events": 0,
}

def load_fidelity(run_dir: pathlib.Path) -> dict:
    """Load drift.json (Channel 1) and judge.json (Channel 2) for a run dir.

    Each may be absent (validator/judge wasn't run); record `unavailable` so
    the scorecard can distinguish "ran and was clean" from "wasn't checked".
    Verdict is the conservative join: drift if either channel reports drift,
    in_character only if both report clean (or one is exempt).
    """
    out = {
        "channel1": {"verdict": "unavailable"},
        "channel2": {"verdict": "unavailable"},
        "verdict": "unavailable",
    }

    drift_path = run_dir / "drift.json"
    if drift_path.exists():
        try:
            d = json.loads(drift_path.read_text())
            summary = d.get("summary") or {}
            out["channel1"] = {
                "verdict": summary.get("verdict", "unknown"),
                "actionable_drift_count": summary.get("actionable_drift_count", 0),
                "scenario_design_flag_count": summary.get("scenario_design_flag_count", 0),
                "ceiling": d.get("competence_ceiling"),
            }
        except (json.JSONDecodeError, OSError):
            out["channel1"] = {"verdict": "malformed"}

    judge_path = run_dir / "judge.json"
    if judge_path.exists():
        try:
            j = json.loads(judge_path.read_text())
            if j.get("skipped"):
                out["channel2"] = {"verdict": "skipped"}
            elif "error" in j:
                out["channel2"] = {"verdict": "malformed", "error": j["error"][:120]}
            else:
                in_char = bool(j.get("in_character"))
                out["channel2"] = {
                    "verdict": "in-character" if in_char else "out-of-character",
                    "confidence": j.get("confidence"),
                    "drift_event_count": len(j.get("drift_events") or []),
                }
        except (json.JSONDecodeError, OSError):
            out["channel2"] = {"verdict": "malformed"}

    # Joined verdict — conservative.
    c1 = out["channel1"]["verdict"]
    c2 = out["channel2"]["verdict"]
    if c1 == "drift" or c2 == "out-of-character":
        out["verdict"] = "drift"
    elif c1 == "unavailable" and c2 == "unavailable":
        out["verdict"] = "unavailable"
    elif c1 in ("malformed",) or c2 in ("malformed",):
        out["verdict"] = "malformed"
    elif c1 in ("in-character", "exempt") and c2 in ("in-character", "skipped", "unavailable"):
        out["verdict"] = "in-character"
    elif c2 == "in-character" and c1 == "unavailable":
        out["verdict"] = "in-character"
    else:
        out["verdict"] = "mixed"
    return out

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
    fidelity = load_fidelity(p.parent)
    c1v = fidelity["channel1"]["verdict"]
    c2v = fidelity["channel2"]["verdict"]
    if c1v == "in-character":
        summary_fidelity["channel1_in_character"] += 1
    elif c1v == "drift":
        summary_fidelity["channel1_drift"] += 1
    elif c1v == "exempt":
        summary_fidelity["channel1_exempt"] += 1
    else:
        summary_fidelity["channel1_unavailable"] += 1
    if c1v != "exempt":
        summary_fidelity["total_actionable_drift_events"] += int(
            fidelity["channel1"].get("actionable_drift_count") or 0
        )
    if c2v == "in-character":
        summary_fidelity["channel2_in_character"] += 1
    elif c2v == "out-of-character":
        summary_fidelity["channel2_out_of_character"] += 1
    elif c2v == "skipped":
        summary_fidelity["channel2_skipped"] += 1
    else:
        summary_fidelity["channel2_unavailable"] += 1
    summary_fidelity["total_judge_drift_events"] += int(
        fidelity["channel2"].get("drift_event_count") or 0
    )
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
        "persona_fidelity": fidelity,
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
    "persona_fidelity": summary_fidelity,
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
    "| Scenario | Persona | Class | Turns | Wall (s) | Tokens | Cost | Artifact errors | Outcome | Fidelity | Rows ok / proc |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]


def fmt_fidelity(fid: dict) -> str:
    """Format the joined fidelity verdict for the scorecard.

    Format: <joined>(<channel1>/<channel2>) — keeps the source channels
    visible so a "drift" verdict points the reader at which channel fired.
    """
    if not fid:
        return "—"
    joined = fid.get("verdict", "—")
    c1 = (fid.get("channel1") or {}).get("verdict", "?")
    c2 = (fid.get("channel2") or {}).get("verdict", "?")
    glyph = {
        "in-character": "✓",
        "drift": "✗",
        "out-of-character": "✗",
        "exempt": "—",
        "skipped": "·",
        "unavailable": "·",
        "malformed": "?",
        "mixed": "~",
    }
    return f"{glyph.get(joined, '?')} ({glyph.get(c1, '?')}/{glyph.get(c2, '?')})"


for r in agg:
    rows = f"{r.get('rows_succeeded')}/{r.get('rows_processed')}" if r.get("ran_engine") else "—"
    usage = r.get("provider_usage") or {}
    tokens = usage.get("total_tokens") if usage.get("token_usage_available") else "—"
    cost = r.get("cost") or {}
    cost_cell = f"${cost.get('actual_usd'):.4f}" if cost.get("cost_available") else "—"
    fidelity_cell = fmt_fidelity(r.get("persona_fidelity") or {})
    lines.append("| {sid} | {p} | {c} | {t} | {w} | {tokens} | {cost} | {artifact_errors} | {o} | {fid} | {r} |".format(
        sid=r["scenario_id"] or "?",
        p=r["persona"] or "?",
        c=r["task_class"] or "?",
        t=r["user_turns"] or "?",
        w=r["total_wall_seconds"] or 0,
        tokens=tokens,
        cost=cost_cell,
        artifact_errors=r.get("artifact_collection_error_count") or 0,
        o=fmt_outcome(r),
        fid=fidelity_cell,
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
    "",
    "## Persona fidelity",
    "",
    "Cross-scenario character-fidelity rollup. Channel 1 is structural-token "
    "drift detection (`drift.json` from `validate_drift.sh`). Channel 2 is the "
    "Haiku LLM judge (`judge.json` from `judge_persona.sh`). Verdict glyphs: "
    "✓ in-character, ✗ drift, — exempt (Channel 1 only, for Dev), · skipped/"
    "unavailable, ~ mixed, ? malformed.",
    "",
    f"- Channel 1: {summary_fidelity['channel1_in_character']} in-character, "
    f"{summary_fidelity['channel1_drift']} drift, "
    f"{summary_fidelity['channel1_exempt']} exempt, "
    f"{summary_fidelity['channel1_unavailable']} unavailable",
    f"- Channel 2: {summary_fidelity['channel2_in_character']} in-character, "
    f"{summary_fidelity['channel2_out_of_character']} out-of-character, "
    f"{summary_fidelity['channel2_skipped']} skipped, "
    f"{summary_fidelity['channel2_unavailable']} unavailable",
    f"- Total actionable Channel-1 drift events: {summary_fidelity['total_actionable_drift_events']}",
    f"- Total Channel-2 drift events: {summary_fidelity['total_judge_drift_events']}",
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
