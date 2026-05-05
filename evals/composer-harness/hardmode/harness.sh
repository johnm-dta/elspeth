#!/usr/bin/env bash
# evals/composer-harness/hardmode/harness.sh — bootstrap one hard-mode scenario.
#
# Usage:
#   harness.sh <scenario_id>                   # bootstrap into $RUNS_DIR/<scenario_id>
#   harness.sh --doctor                        # run preflight only, no side effects
#   harness.sh --reuse-sid <scenario_id>       # don't refuse if dir already exists
#   harness.sh --fresh <scenario_id>           # delete prior dir before bootstrapping
#
# Writes (per scenario): jwt.txt, scenario.json, session.json, sid.txt,
#                        blob.req.json + blob.json (if scenario has a CSV),
#                        harness.log.
#
# Required env (load from .env in $EVAL_HARNESS_ROOT or set manually):
#   ELSPETH_EVAL_BASE_URL     - e.g. https://elspeth.foundryside.dev
#   ELSPETH_EVAL_USER         - login username
#   ELSPETH_EVAL_PASS         - login password
#
# Optional env:
#   ELSPETH_EVAL_RUNS_DIR     - where to write per-scenario output (default: <harness>/runs/<utc-date>)
#   ELSPETH_EVAL_RUN_TIMEOUT_SEC, ELSPETH_EVAL_RUN_POLL_INTERVAL — see lib/common.sh
#
# Exit codes:
#   0 ok / 1 generic / 64 bad usage / 65 scenario not found / 66 dir conflict
#   69 missing tool / 70 login failure / 71 HTTP failure / 72 session create failure
#   73 blob upload failure

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$HARNESS_DIR"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

# ---------- Arg parsing ------------------------------------------------------

mode=run
fresh=0
reuse=0
scenario_id=""

while (( $# > 0 )); do
  case "$1" in
    --doctor) mode=doctor ;;
    --fresh)  fresh=1 ;;
    --reuse-sid) reuse=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*) evals_die 64 "unknown flag: $1" ;;
    *)
      if [[ -n "$scenario_id" ]]; then evals_die 64 "extra positional arg: $1"; fi
      scenario_id=$1
      ;;
  esac
  shift
done

# ---------- Doctor mode ------------------------------------------------------

if [[ "$mode" == doctor ]]; then
  exec "$LIB_DIR/preflight.sh"
fi

if [[ -z "$scenario_id" ]]; then
  evals_die 64 "usage: harness.sh [--doctor|--fresh|--reuse-sid] <scenario_id>"
fi

# ---------- Setup ------------------------------------------------------------

evals_load_env --require-creds
evals_require_tools

scen_path=$(evals_resolve_scenario "$HARNESS_ROOT/scenarios/hardmode" "$scenario_id")

# Default RUNS_DIR includes a UTC date so historical runs accumulate side-by-side.
runs_root="${ELSPETH_EVAL_RUNS_DIR:-$HARNESS_ROOT/runs/$(date -u +%Y-%m-%d)-hardmode}"
out="$runs_root/$scenario_id"

if [[ -d "$out" ]]; then
  if (( fresh )); then
    evals_log WARN "deleting existing $out (--fresh)"
    rm -rf "$out"
  elif (( reuse )); then
    evals_log INFO "reusing existing $out (--reuse-sid)"
  else
    evals_die 66 "output dir already exists: $out (use --fresh to delete or --reuse-sid to keep)"
  fi
fi
mkdir -p "$out"

export EVALS_OUT_DIR="$out"
export EVALS_LOG_FILE="$out/harness.log"
export EVALS_JWT_FILE="$out/jwt.txt"

evals_log INFO "scenario=$scenario_id out=$out"

# Snapshot the immutable scenario fixture into the run dir.
cp "$scen_path" "$out/scenario.json"

# ---------- Login ------------------------------------------------------------

if (( reuse )) && [[ -s "$EVALS_JWT_FILE" ]]; then
  evals_log INFO "reusing existing JWT (will auto-refresh if near expiry)"
  evals_login_if_needed
else
  evals_login
fi

# ---------- Session ----------------------------------------------------------

# If reusing and sid exists, skip session creation.
if (( reuse )) && [[ -s "$out/sid.txt" ]]; then
  sid=$(cat "$out/sid.txt")
  evals_log INFO "reusing session sid=$sid (--reuse-sid)"
else
  task_summary=$(jq -r '.task_summary' "$out/scenario.json")
  title="hardmode/$scenario_id $task_summary"
  sid=$(evals_create_session "$title")
fi

# ---------- Blob (CSV) -------------------------------------------------------

csv_filename=$(jq -r '.csv_filename // empty' "$out/scenario.json")
if [[ -n "$csv_filename" ]]; then
  if (( reuse )) && [[ -s "$out/blob.json" ]]; then
    evals_log INFO "reusing existing blob.json (--reuse-sid)"
  else
    csv_tmp=$(mktemp -t evals_csv.XXXXXX.csv)
    trap 'rm -f "$csv_tmp"' EXIT
    jq -r '.csv_content' "$out/scenario.json" > "$csv_tmp"
    evals_upload_blob "$sid" "$csv_filename" "text/csv" "$csv_tmp"
  fi
else
  evals_log INFO "scenario has no csv_content — skipping blob upload"
fi

# ---------- Done ------------------------------------------------------------

evals_log INFO "bootstrap complete: $out (sid=$sid)"
cat <<NEXT
Scaffold ready for scenario '$scenario_id'.
  Output dir: $out
  Session id: $sid

Next steps:
  1. Copy the opening prompt for turn 1:
       jq -r '.opening_prompt' '$out/scenario.json' > '$out/turn1.user.txt'
  2. Send it:
       evals/composer-harness/hardmode/post_message.sh '$scenario_id' 1 '$out/turn1.user.txt'
  3. Read composer's reply:
       jq -r '.message.content' '$out/msg.t1.resp.json'
  4. Spawn a persona-subagent (see evals/lib/dispatch-protocol.md) for turn 2+
     until persona signals DONE or message budget (5) is hit.
  5. Finalize:
       evals/composer-harness/hardmode/finalize_scenario.sh '$scenario_id'
NEXT
