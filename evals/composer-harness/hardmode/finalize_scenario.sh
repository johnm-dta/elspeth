#!/usr/bin/env bash
# evals/composer-harness/hardmode/finalize_scenario.sh — after-DONE finaliser.
#
# Usage: finalize_scenario.sh <scenario_id>
#
# Calls /validate. If valid, calls /execute, then polls /api/runs/<rid> until
# terminal status (per RunStatus enum: completed, completed_with_failures,
# failed, empty, interrupted) or $ELSPETH_EVAL_RUN_TIMEOUT_SEC (default 300s).
#
# Writes:
#   validate.json
#   execute.json + execute.code (if validate passed)
#   run.json (final)
#   diagnostics.json (if run reached terminal)
#   final_yaml.json
#   messages.json
#   ledger.json (consolidated per-scenario summary)
#
# Exit codes:
#   0  validate ran and ledger written (run may still have failed — read ledger.run_status)
#   64 bad usage
#   67 missing run dir / scenario
#   74 validate returned non-2xx
#   75 run did not reach terminal within timeout (ledger still written, run.json reflects last poll)

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$HARNESS_DIR"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

if (( $# != 1 )); then
  evals_die 64 "usage: finalize_scenario.sh <scenario_id>"
fi
scenario_id=$1

evals_load_env --require-creds
evals_require_tools

runs_root="${ELSPETH_EVAL_RUNS_DIR:-}"
if [[ -z "$runs_root" ]]; then
  runs_root=$(ls -1d "$HARNESS_ROOT"/runs/*-hardmode 2>/dev/null | sort -r | head -1 || true)
fi
[[ -n "$runs_root" ]] || evals_die 67 "no runs dir found under $HARNESS_ROOT/runs"

out="$runs_root/$scenario_id"
[[ -d "$out" ]] || evals_die 67 "scenario run dir not found: $out"

export EVALS_OUT_DIR="$out"
export EVALS_LOG_FILE="$out/harness.log"
export EVALS_JWT_FILE="$out/jwt.txt"

sid=$(cat "$out/sid.txt")
evals_log INFO "finalize scenario=$scenario_id sid=$sid"

# /validate (always)
evals_validate "$sid" "$out/validate.json"
is_valid=$(jq -r '.is_valid // false' "$out/validate.json")
evals_log INFO "validate.is_valid=$is_valid"

ran_engine=false
run_id=""
poll_status=ok

# /execute only if valid
if [[ "$is_valid" == "true" ]]; then
  evals_execute "$sid" "$out/execute.json" "$out/execute.code"
  http=$(cat "$out/execute.code")
  evals_log INFO "execute returned HTTP $http"
  if [[ "$http" == "202" ]]; then
    run_id=$(jq -r '.run_id' "$out/execute.json")
    ran_engine=true
    evals_log INFO "polling run $run_id (timeout=${ELSPETH_EVAL_RUN_TIMEOUT_SEC}s)"
    if ! evals_poll_run_terminal "$run_id" \
           "$ELSPETH_EVAL_RUN_TIMEOUT_SEC" \
           "$ELSPETH_EVAL_RUN_POLL_INTERVAL" \
           "$out/run.json"; then
      poll_status=timeout
    fi
    # Diagnostics — best-effort even if timed out (engine may have written partial state).
    evals_get_diagnostics "$run_id" "$out/diagnostics.json" 2>/dev/null || \
      echo '{}' > "$out/diagnostics.json"
  fi
fi

# /state/yaml (final composition) — always, even if not valid.
evals_get_yaml "$sid" "$out/final_yaml.json" 2>/dev/null || \
  echo '{}' > "$out/final_yaml.json"

# /messages (full conversation)
evals_get_messages "$sid" "$out/messages.json" 2>/dev/null || \
  echo '[]' > "$out/messages.json"

# Ledger
python3 - "$out" "$scenario_id" "$sid" "$ran_engine" "$run_id" "$is_valid" "$poll_status" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
ledger = dict(
    scenario_id=sys.argv[2],
    sid=sys.argv[3],
    ran_engine=(sys.argv[4] == "true"),
    run_id=sys.argv[5] or None,
    is_valid=(sys.argv[6] == "true"),
    poll_status=sys.argv[7],  # ok | timeout
)

scen = json.loads((out / "scenario.json").read_text())
for f in ("persona", "task_class", "task_summary", "pass_criterion",
          "expected_outcome", "product_capability_used", "probe"):
    ledger[f] = scen.get(f)

# Per-turn metrics
turns = []
for p in sorted(out.glob("metrics.t*.json")):
    turns.append(json.loads(p.read_text()))
ledger["turns"] = turns
ledger["user_turn_count"] = len(turns)
ledger["total_tool_calls"] = sum(t.get("tool_call_count", 0) for t in turns)
ledger["total_in_loop_recoveries"] = sum(t.get("in_loop_recovery_count", 0) for t in turns)
ledger["total_wall_seconds"] = sum((t.get("wall_seconds") or 0) for t in turns)
ledger["mutating_turns"] = [t["turn"] for t in turns if t.get("mutated_state")]
ledger["clarifying_keyword_turns"] = [t["turn"] for t in turns if t.get("asked_clarifying_keyword_match")]
ledger["limit_keyword_turns"] = [t["turn"] for t in turns if t.get("volunteered_limit_keyword_match")]

run_path = out / "run.json"
if run_path.exists() and run_path.read_text().strip():
    try:
        run = json.loads(run_path.read_text())
    except json.JSONDecodeError:
        run = {}
    for f in ("status", "rows_processed", "rows_succeeded",
              "rows_routed_success", "rows_routed_failure",
              "rows_failed", "rows_quarantined", "error",
              "started_at", "finished_at", "landscape_run_id"):
        ledger[f"run_{f}" if f in ("status", "error") else f] = run.get(f)

val = json.loads((out / "validate.json").read_text() or "{}")
checks = val.get("checks") or []
ledger["failed_validate_checks"] = [c.get("name") for c in checks if c.get("passed") is False]

(out / "ledger.json").write_text(json.dumps(ledger, indent=2))
print(json.dumps({k: ledger[k] for k in (
    "scenario_id", "is_valid", "ran_engine", "run_status", "poll_status",
    "user_turn_count", "rows_processed", "rows_succeeded")}, indent=2))
PY

if [[ "$poll_status" == "timeout" ]]; then
  evals_log ERROR "run did not reach terminal — ledger written with poll_status=timeout"
  exit 75
fi

if [[ "$is_valid" != "true" ]]; then
  evals_log WARN "validate failed — see validate.json (.failed_validate_checks)"
  # Not an error from the harness's perspective — validate-fail is a valid scenario outcome.
fi

evals_log INFO "finalize complete: $out/ledger.json"
