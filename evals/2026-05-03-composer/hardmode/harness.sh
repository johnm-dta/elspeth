#!/usr/bin/env bash
# Hard-mode harness driver: one scenario at a time.
#
# Usage:  harness.sh <scenario_id>
# Example: harness.sh p1_t1_happy
#
# Per-scenario flow:
#   1. Read scenario JSON (persona, task, optional CSV).
#   2. Login (fresh JWT each scenario).
#   3. Create composer session, upload CSV blob if provided.
#   4. Driver-LLM (parent) loops:
#        (a) Send (or pass-through) the user message to /messages.
#        (b) Capture composer response, /state, /composer-progress events.
#        (c) Hand the composer response back to the persona-subagent (via stdin file).
#        (d) Persona returns next user msg or 'DONE: <reason>'.
#   5. After persona DONE: run /validate, /execute (if state.is_valid), capture run.
#   6. Write per-scenario ledger JSON: messages, metrics, validate, run, output files.
#
# This script is a SCAFFOLD — the parent agent (Claude in the main thread)
# performs the persona-subagent role-locking via the Agent tool. The shell
# script is responsible for the deterministic HTTP plumbing only.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SID_FILE=""
JWT_FILE=""

scenario_id="${1:-}"
if [[ -z "$scenario_id" ]]; then echo "usage: $0 <scenario_id>"; exit 1; fi

scen=$ROOT/scenarios/${scenario_id}_*.json
scen=$(ls $scen 2>/dev/null | head -1)
if [[ -z "$scen" ]]; then echo "scenario not found: $scenario_id"; exit 1; fi

out=$ROOT/results/$scenario_id
mkdir -p $out
cp "$scen" $out/scenario.json

# --- step 1: login fresh ---
curl -fsS -X POST https://elspeth.foundryside.dev/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"dta_user","password":"dta_pass"}' \
  | jq -r '.access_token' > $out/jwt.txt
J=$(cat $out/jwt.txt)
[[ -n "$J" ]] || { echo "login failed"; exit 2; }

# --- step 2: create session ---
title=$(jq -r '.task_summary' $out/scenario.json)
curl -fsS -X POST https://elspeth.foundryside.dev/api/sessions \
  -H "Authorization: Bearer $J" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg t "hardmode/$scenario_id $title" '{title:$t}')" \
  -o $out/session.json
sid=$(jq -r '.id' $out/session.json)
echo "$sid" > $out/sid.txt

# --- step 3: upload blob if scenario has one ---
csv_filename=$(jq -r '.csv_filename // empty' $out/scenario.json)
if [[ -n "$csv_filename" ]]; then
  jq -n \
    --arg fn "$csv_filename" \
    --rawfile body <(jq -r '.csv_content' $out/scenario.json) \
    '{filename:$fn, mime_type:"text/csv", content:$body}' \
    > $out/blob.req.json
  curl -fsS -X POST "https://elspeth.foundryside.dev/api/sessions/$sid/blobs/inline" \
    -H "Authorization: Bearer $J" -H 'Content-Type: application/json' \
    --data @$out/blob.req.json -o $out/blob.json
fi

echo "Scaffold ready: $out (sid=$sid)"
echo "Next: parent agent runs the persona-subagent loop, posts /messages, then runs validate-and-execute.sh"
