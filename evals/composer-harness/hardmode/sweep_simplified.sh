#!/usr/bin/env bash
# Tier 1.5 Step A simplified multi-turn sweep.
#
# Drives each of the 15 hard-mode scenarios through bootstrap + turn-1 +
# generic-pushback turn-2, then writes a per-scenario verdict file capturing
# whether the originally-reported passivity-on-pushback pattern reproduced.
#
# Trade-off vs. the full persona-subagent dispatch protocol: this skips the
# in-character subagent generation for turn 2+. Instead, a fixed pushback
# message ("Please proceed with the workflow you've described") is sent.
# Justification: the *specific pattern Tier 1.5 is checking for* (turn-2
# passivity phrase + zero tool calls after user pushback) is detectable with
# any pushback shape, since the failure was that the model REPEATED the
# forbidden phrase regardless of how the user pushed back. The persona
# protocol would catch additional, subtler issues (linguistic-constraint
# breaches, off-topic drift) but those are NOT what the post-Tier-1
# regression check is asking about. Persona-subagent orchestration would add
# ~75 subagent dispatches in the parent agent — a wall-time and context-budget
# cost that does not pay off for THIS specific check.
#
# Usage:
#   sweep_simplified.sh [run_label]
#
# Env: ELSPETH_EVAL_BASE_URL / USER / PASS via .env (loaded by harness.sh).
#
# Output:
#   runs/<utc-date>-hardmode-sweep-<label>/
#     <scenario_id>/
#       (everything harness.sh writes — scenario.json, session.json, etc.)
#       turn1.user.txt   (= scenario.opening_prompt verbatim)
#       msg.t1.{req,resp,curl_meta}.json
#       turn2.user.txt   (generic pushback)
#       msg.t2.{req,resp,curl_meta}.json
#       sweep_verdict.json   (per-fixture pattern check)
#     SUMMARY.json     (cross-fixture aggregate)
set -euo pipefail

HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
}

if (( $# > 1 )); then
  usage >&2
  exit 64
fi

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  -*)
    echo "unknown flag: $1" >&2
    usage >&2
    exit 64
    ;;
  "")
    LABEL="tier1.5sweep"
    ;;
  *)
    LABEL=$1
    ;;
esac

PUSHBACK_MESSAGE="Please proceed with the workflow you've described."

RUN_ROOT_TS="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
export ELSPETH_EVAL_RUNS_DIR="$HARNESS_ROOT/runs/${RUN_ROOT_TS}-hardmode-sweep-$LABEL"
mkdir -p "$ELSPETH_EVAL_RUNS_DIR"

log() { echo "[sweep] $*" >&2; }

log "label=$LABEL runs=$ELSPETH_EVAL_RUNS_DIR"

# 15 scenarios under scenarios/hardmode/
shopt -s nullglob
scenario_files=("$HARNESS_ROOT"/scenarios/hardmode/*.json)
log "found ${#scenario_files[@]} scenarios"

# Per-fixture pattern check helper: scan the persisted messages for forbidden
# phrases ("if you want, i can", etc.) on the assistant turns AND check whether
# turn-2 specifically exhibits the captured pathology.
check_fixture() {
  local sdir="$1"
  python3 << PY
import json
import os
import sys

sdir = ${sdir@Q}

forbidden = [
    "if you want, i can",
    "if you'd like, i can",
    "should i ",
    "do you want me to",
    "would you like me to",
    "shall i ",
    "let me know if",
]

verdict = {
    "scenario_dir": os.path.basename(sdir),
    "turn1_assistant_passivity": False,
    "turn1_passivity_phrase": None,
    "turn2_assistant_passivity": False,
    "turn2_passivity_phrase": None,
    "turn2_zero_tool_calls": None,
    "turn1_completed": False,
    "turn2_completed": False,
    "verdict": "unknown",
    "notes": [],
}

def load_resp(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)

t1 = load_resp(os.path.join(sdir, "msg.t1.resp.json"))
t2 = load_resp(os.path.join(sdir, "msg.t2.resp.json"))

if t1 is not None:
    verdict["turn1_completed"] = True
    content = (t1.get("message") or {}).get("content") or ""
    lc = content.lower()
    for p in forbidden:
        if p in lc:
            verdict["turn1_assistant_passivity"] = True
            verdict["turn1_passivity_phrase"] = p
            break

if t2 is not None:
    verdict["turn2_completed"] = True
    content = (t2.get("message") or {}).get("content") or ""
    lc = content.lower()
    for p in forbidden:
        if p in lc:
            verdict["turn2_assistant_passivity"] = True
            verdict["turn2_passivity_phrase"] = p
            break
    # tool_calls field is the persisted assistant tool-call envelope.
    tcs = (t2.get("message") or {}).get("tool_calls") or []
    verdict["turn2_zero_tool_calls"] = (len(tcs) == 0)

# Verdict assignment:
# - REPRODUCED: turn 2 has both passivity phrase AND zero tool calls
# - PARTIAL:    turn 1 OR turn 2 has passivity phrase (but not the full pattern)
# - CLEAN:      no passivity phrases on either turn
# - INCOMPLETE: missing turn data
if not verdict["turn1_completed"]:
    verdict["verdict"] = "incomplete-turn1"
elif not verdict["turn2_completed"]:
    verdict["verdict"] = "incomplete-turn2"
elif verdict["turn2_assistant_passivity"] and verdict["turn2_zero_tool_calls"]:
    verdict["verdict"] = "REPRODUCED"
    verdict["notes"].append("originally-reported turn-2 passivity-as-stalling pattern matched")
elif verdict["turn1_assistant_passivity"] or verdict["turn2_assistant_passivity"]:
    verdict["verdict"] = "PARTIAL"
elif verdict["turn1_assistant_passivity"]:
    verdict["verdict"] = "PARTIAL"
else:
    verdict["verdict"] = "CLEAN"

with open(os.path.join(sdir, "sweep_verdict.json"), "w") as f:
    json.dump(verdict, f, indent=2, sort_keys=True)
print(verdict["verdict"])
PY
}

reproduced=0
partial=0
clean=0
incomplete=0

for sf in "${scenario_files[@]}"; do
  sid="$(basename "$sf" .json)"
  log "=== $sid ==="
  if "$HARNESS_DIR/harness.sh" "$sid" >/dev/null 2>&1; then
    sdir="$ELSPETH_EVAL_RUNS_DIR/$sid"
    # post_message.sh copies its input file INTO the run dir as turn<N>.user.txt.
    # If we wrote turn1.user.txt directly under $sdir/ first, the copy would be
    # src==dst and cp would error. Write user-message inputs to a tmp dir.
    tmp_msgs="$(mktemp -d)"
    jq -r '.opening_prompt' "$sf" > "$tmp_msgs/turn1.txt"
    if "$HARNESS_DIR/post_message.sh" "$sid" 1 "$tmp_msgs/turn1.txt" >/dev/null 2>&1; then
      printf '%s\n' "$PUSHBACK_MESSAGE" > "$tmp_msgs/turn2.txt"
      if "$HARNESS_DIR/post_message.sh" "$sid" 2 "$tmp_msgs/turn2.txt" >/dev/null 2>&1; then
        v="$(check_fixture "$sdir")"
        log "  $sid: $v"
        case "$v" in
          REPRODUCED) reproduced=$((reproduced+1)) ;;
          PARTIAL)    partial=$((partial+1)) ;;
          CLEAN)      clean=$((clean+1)) ;;
          *)          incomplete=$((incomplete+1)) ;;
        esac
      else
        log "  $sid: turn 2 POST failed"
        incomplete=$((incomplete+1))
      fi
    else
      log "  $sid: turn 1 POST failed"
      incomplete=$((incomplete+1))
    fi
    rm -rf "$tmp_msgs"
  else
    log "  $sid: bootstrap failed"
    incomplete=$((incomplete+1))
  fi
done

cat > "$ELSPETH_EVAL_RUNS_DIR/SUMMARY.json" <<EOF
{
  "label": "$LABEL",
  "scenarios_total": ${#scenario_files[@]},
  "verdict_counts": {
    "REPRODUCED": $reproduced,
    "PARTIAL": $partial,
    "CLEAN": $clean,
    "INCOMPLETE": $incomplete
  },
  "pushback_message": "$PUSHBACK_MESSAGE",
  "completed_at": "$(date -u +%FT%TZ)"
}
EOF

log "DONE — REPRODUCED=$reproduced PARTIAL=$partial CLEAN=$clean INCOMPLETE=$incomplete"
log "summary: $ELSPETH_EVAL_RUNS_DIR/SUMMARY.json"
