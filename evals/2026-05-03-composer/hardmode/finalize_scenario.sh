#!/usr/bin/env bash
# After the persona-subagent says DONE: run /validate, /execute (if valid),
# capture run + outputs, write a per-scenario ledger.
#
# Usage: finalize_scenario.sh <scenario_id>

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario_id=$1
out=$ROOT/results/$scenario_id
sid=$(cat $out/sid.txt)
J=$(cat $out/jwt.txt)

# /validate
curl -fsS -X POST "https://elspeth.foundryside.dev/api/sessions/$sid/validate" \
  -H "Authorization: Bearer $J" -o $out/validate.json 2>/dev/null || echo '{}' > $out/validate.json
is_valid=$(jq -r '.is_valid // false' $out/validate.json)

# /execute only if valid
ran=false
rid=""
if [[ "$is_valid" == "true" ]]; then
  curl -sS -X POST "https://elspeth.foundryside.dev/api/sessions/$sid/execute" \
    -H "Authorization: Bearer $J" -H 'Content-Type: application/json' -d '{}' \
    -o $out/execute.json -w '%{http_code}\n' > $out/execute.code
  http=$(cat $out/execute.code)
  if [[ "$http" == "202" ]]; then
    rid=$(jq -r '.run_id' $out/execute.json)
    ran=true
    sleep 5
    curl -fsS -H "Authorization: Bearer $J" \
      "https://elspeth.foundryside.dev/api/runs/$rid" \
      -o $out/run.json
    curl -fsS -H "Authorization: Bearer $J" \
      "https://elspeth.foundryside.dev/api/runs/$rid/diagnostics" \
      -o $out/diagnostics.json 2>/dev/null || true
  fi
fi

# /state/yaml (final composition)
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/state/yaml" \
  -o $out/final_yaml.json 2>/dev/null || echo '{}' > $out/final_yaml.json

# /messages (full conversation)
curl -fsS -H "Authorization: Bearer $J" \
  "https://elspeth.foundryside.dev/api/sessions/$sid/messages" \
  -o $out/messages.json 2>/dev/null || echo '[]' > $out/messages.json

# Aggregate per-scenario ledger
python3 - <<PY
import json, pathlib, glob
out = pathlib.Path("$out")
ledger = dict(
    scenario_id="$scenario_id",
    sid="$sid",
    ran_engine=("$ran" == "true"),
    run_id="$rid",
    is_valid=("$is_valid" == "true"),
)

# Load scenario metadata
scen = json.loads((out/"scenario.json").read_text())
ledger["persona"] = scen.get("persona")
ledger["task_class"] = scen.get("task_class")
ledger["task_summary"] = scen.get("task_summary")
ledger["pass_criterion"] = scen.get("pass_criterion")

# Per-turn metrics
turn_metrics = []
for p in sorted(out.glob("metrics.t*.json")):
    turn_metrics.append(json.loads(p.read_text()))
ledger["turns"] = turn_metrics
ledger["user_turn_count"] = len(turn_metrics)
ledger["total_tool_calls"] = sum(t["tool_call_count"] for t in turn_metrics)
ledger["total_in_loop_recoveries"] = sum(t["in_loop_recovery_count"] for t in turn_metrics)
ledger["total_wall_seconds"] = sum(t["wall_seconds"] for t in turn_metrics)
ledger["mutating_turns"] = [t["turn"] for t in turn_metrics if t["mutated_state"]]
ledger["clarifying_question_turns"] = [t["turn"] for t in turn_metrics if t["asked_clarifying_question"]]
ledger["volunteered_limit_turns"] = [t["turn"] for t in turn_metrics if t["volunteered_limit"]]

# Run outcome
run_path = out/"run.json"
if run_path.exists() and run_path.read_text().strip():
    run = json.loads(run_path.read_text())
    ledger["run_status"] = run.get("status")
    ledger["rows_processed"] = run.get("rows_processed")
    ledger["rows_succeeded"] = run.get("rows_succeeded")
    ledger["rows_routed_success"] = run.get("rows_routed_success")
    ledger["run_error"] = run.get("error")

# Validate outcome (which checks failed if any)
val = json.loads((out/"validate.json").read_text() or "{}")
checks = val.get("checks") or []
ledger["failed_validate_checks"] = [c.get("name") for c in checks if c.get("passed") is False]

(out/"ledger.json").write_text(json.dumps(ledger, indent=2))
print(json.dumps(ledger, indent=2))
PY
