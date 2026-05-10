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

printf '{}\n' > "$out/run.json"
printf '{}\n' > "$out/diagnostics.json"

artifact_errors_file="$out/artifact_collection_errors.json"
printf '[]\n' > "$artifact_errors_file"

record_artifact_error() {
  local artifact=$1 http_code=$2 path=$3
  python3 - "$artifact_errors_file" "$artifact" "$http_code" "$path" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    errors = json.loads(path.read_text())
except (FileNotFoundError, json.JSONDecodeError):
    errors = []
errors.append({"artifact": sys.argv[2], "http_code": sys.argv[3], "path": sys.argv[4]})
path.write_text(json.dumps(errors, indent=2) + "\n")
PY
}

capture_optional_json() {
  local artifact=$1 url=$2 dest=$3 default_json=$4
  local code_file="$dest.code"
  if evals_try_get "$url" "$dest" "$code_file"; then
    return 0
  fi
  local http
  http=$(cat "$code_file" 2>/dev/null || printf '000')
  evals_log WARN "optional artifact $artifact unavailable (HTTP $http); writing fallback to $dest"
  printf '%s\n' "$default_json" > "$dest"
  record_artifact_error "$artifact" "$http" "$(basename "$dest")"
  return 0
}

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
    capture_optional_json \
      "diagnostics" \
      "$ELSPETH_EVAL_BASE_URL/api/runs/$run_id/diagnostics" \
      "$out/diagnostics.json" \
      '{}'
  fi
fi

# /state/yaml (final composition) — always, even if not valid.
capture_optional_json \
  "final_yaml" \
  "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/state/yaml" \
  "$out/final_yaml.json" \
  '{}'

# /messages (full conversation)
capture_optional_json \
  "messages" \
  "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/messages?include_llm_audit=true" \
  "$out/messages.json" \
  '[]'

# Ledger
python3 - "$out" "$scenario_id" "$sid" "$ran_engine" "$run_id" "$is_valid" "$poll_status" <<'PY'
import json, math, pathlib, sys
out = pathlib.Path(sys.argv[1])
ledger = dict(
    scenario_id=sys.argv[2],
    sid=sys.argv[3],
    ran_engine=(sys.argv[4] == "true"),
    run_id=sys.argv[5] or None,
    is_valid=(sys.argv[6] == "true"),
    poll_status=sys.argv[7],  # ok | timeout
    run_status=None,
    run_error=None,
    rows_processed=None,
    rows_succeeded=None,
    rows_routed_success=None,
    rows_routed_failure=None,
    rows_failed=None,
    rows_quarantined=None,
    started_at=None,
    finished_at=None,
    landscape_run_id=None,
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

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from walk(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk(value)

usage = {
    "prompt_tokens": 0,
    "cached_prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "models": [],
    "token_usage_available": False,
    "source": "not_available",
}
models: set[str] = set()
usage_sources: set[str] = set()

def add_usage(item, source):
    added = False
    for field in ("prompt_tokens", "cached_prompt_tokens", "completion_tokens", "total_tokens"):
        value = item.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            usage[field] += value
            added = True
    model = item.get("model") or item.get("model_returned") or item.get("model_requested")
    if isinstance(model, str) and model:
        models.add(model)
    if added:
        usage_sources.add(source)

diag_path = out / "diagnostics.json"
if diag_path.exists() and diag_path.read_text().strip():
    try:
        diagnostics = json.loads(diag_path.read_text())
    except json.JSONDecodeError:
        diagnostics = {}
    for item in walk(diagnostics):
        add_usage(item, "diagnostics")

provider_costs = []
messages_path = out / "messages.json"
if messages_path.exists() and messages_path.read_text().strip():
    try:
        messages = json.loads(messages_path.read_text())
    except json.JSONDecodeError:
        messages = []
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            tool_calls = message.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or tool_call.get("_kind") != "llm_call_audit":
                    continue
                call = tool_call.get("call")
                if not isinstance(call, dict):
                    continue
                add_usage(call, "llm_call_audit")
                cost = call.get("provider_cost")
                if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(float(cost)) and cost >= 0:
                    provider_costs.append(float(cost))

usage["models"] = sorted(models)
usage["token_usage_available"] = bool(usage_sources)
usage["source"] = "+".join(sorted(usage_sources)) if usage_sources else "not_available"
ledger["provider_usage"] = usage
if provider_costs:
    ledger["cost"] = {
        "actual_usd": round(sum(provider_costs), 10),
        "source": "composer_llm_audit.response_usage.cost",
        "cost_available": True,
        "costed_call_count": len(provider_costs),
    }
else:
    ledger["cost"] = {
        "actual_usd": None,
        "source": "not_available",
        "cost_available": False,
        "costed_call_count": 0,
    }

val = json.loads((out / "validate.json").read_text() or "{}")
checks = val.get("checks") or []
ledger["failed_validate_checks"] = [c.get("name") for c in checks if c.get("passed") is False]

artifact_errors_path = out / "artifact_collection_errors.json"
if artifact_errors_path.exists():
    try:
        ledger["artifact_collection_errors"] = json.loads(artifact_errors_path.read_text())
    except json.JSONDecodeError:
        ledger["artifact_collection_errors"] = [
            {"artifact": "artifact_collection_errors", "http_code": "parse_error", "path": "artifact_collection_errors.json"}
        ]
else:
    ledger["artifact_collection_errors"] = []

(out / "ledger.json").write_text(json.dumps(ledger, indent=2))
print(json.dumps({k: ledger.get(k) for k in (
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

# Panel-cohort scenarios encode red_criteria + green_criteria for RGR-style
# scoring. Hardmode scenarios don't have those fields, so this step is a
# silent no-op for them. When the fields ARE present, score_scenario.sh
# emits rgr_score.json which aggregate.sh rolls into the panel scorecard.
if jq -e 'has("red_criteria") and has("green_criteria")' "$out/scenario.json" >/dev/null 2>&1; then
  evals_log INFO "scoring against scenario red/green criteria"
  if ! "$EVALS_SCRIPT_DIR/score_scenario.sh" "$out"; then
    evals_log WARN "score_scenario.sh exited non-zero — see $out/rgr_score.json"
  fi
fi
