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
import json, pathlib, sys

runs = pathlib.Path(sys.argv[1])
ledgers = sorted(runs.glob("*/ledger.json"))
agg = []
for p in ledgers:
    try:
        L = json.loads(p.read_text())
    except json.JSONDecodeError:
        continue
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
    })

agg.sort(key=lambda r: (r.get("persona") or "", r.get("task_class") or "", r.get("scenario_id") or ""))
(runs / "aggregate.json").write_text(json.dumps(agg, indent=2))

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
    "| Scenario | Persona | Class | Turns | Wall (s) | Outcome | Rows ok / proc |",
    "|---|---|---|---|---|---|---|",
]
for r in agg:
    rows = f"{r.get('rows_succeeded')}/{r.get('rows_processed')}" if r.get("ran_engine") else "—"
    lines.append("| {sid} | {p} | {c} | {t} | {w} | {o} | {r} |".format(
        sid=r["scenario_id"] or "?",
        p=r["persona"] or "?",
        c=r["task_class"] or "?",
        t=r["user_turns"] or "?",
        w=r["total_wall_seconds"] or 0,
        o=fmt_outcome(r),
        r=rows,
    ))

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

(runs / "SCORECARD.md").write_text("\n".join(lines) + "\n")
print(f"wrote {runs/'aggregate.json'}")
print(f"wrote {runs/'SCORECARD.md'}")
PY
