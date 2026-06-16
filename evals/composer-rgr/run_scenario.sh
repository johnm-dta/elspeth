#!/usr/bin/env bash
# Minimal RGR harness for the pipeline_composer skill.
#
# One scenario, one turn, one judgment. Drives the staging composer with a
# fixed opening prompt, captures every assistant turn (including in-loop
# tool-call assistant messages), and scores against the scenario's
# red_criteria / green_criteria.
#
# Usage:
#   ELSPETH_EVAL_BASE_URL=https://elspeth.foundryside.dev \
#   ELSPETH_EVAL_USER=dta_user \
#   ELSPETH_EVAL_PASS=dta_pass \
#   ./run_scenario.sh [run_label]
#
# Selecting a scenario:
#   By default the harness uses scenarios/url-download-line-explode/scenario.json.
#   Override via ELSPETH_RGR_SCENARIO=<path-to-scenario.json>. Run dirs land
#   under runs/<utc-ts>-<scenario-basename>-<label>/ when a non-default
#   scenario is selected, so per-scenario verdicts don't muddle.
#
# Output:
#   runs/<utc-ts>-<run_label>/
#     login.json  session.json  blob.req.json/blob.json (if scenario CSV is seeded)
#     send.json  messages.json  scoring.json
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCENARIO_FILE="${ELSPETH_RGR_SCENARIO:-$HERE/scenarios/url-download-line-explode/scenario.json}"
[[ -f "$SCENARIO_FILE" ]] || { echo "ERROR: scenario file not found: $SCENARIO_FILE" >&2; exit 64; }
SCENARIO_NAME="$(basename "$(dirname "$SCENARIO_FILE")")"
LABEL="${1:-rgr}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$HERE/runs/$TS-$SCENARIO_NAME-$LABEL"
mkdir -p "$RUN_DIR"

: "${ELSPETH_EVAL_BASE_URL:?set ELSPETH_EVAL_BASE_URL}"
: "${ELSPETH_EVAL_USER:?set ELSPETH_EVAL_USER}"
: "${ELSPETH_EVAL_PASS:?set ELSPETH_EVAL_PASS}"

log() { echo "[$(date -u +%FT%TZ)] $*" >&2; }

log "login as $ELSPETH_EVAL_USER"
curl -sS -X POST "$ELSPETH_EVAL_BASE_URL/api/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"username\":\"$ELSPETH_EVAL_USER\",\"password\":\"$ELSPETH_EVAL_PASS\"}" \
    -o "$RUN_DIR/login.json"
JWT="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("access_token") or d.get("token") or "")' "$RUN_DIR/login.json")"
[[ -n "$JWT" ]] || { echo "ERROR: empty JWT" >&2; cat "$RUN_DIR/login.json" >&2; exit 1; }

log "create session"
curl -sS -X POST "$ELSPETH_EVAL_BASE_URL/api/sessions" \
    -H "Authorization: Bearer $JWT" \
    -H "Content-Type: application/json" \
    -d "{\"title\":\"composer-rgr $LABEL $TS\"}" \
    -o "$RUN_DIR/session.json"
SID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$RUN_DIR/session.json")"
log "session id: $SID"

HAS_CSV_BLOB="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("csv_filename") else "")' "$SCENARIO_FILE")"
if [[ -n "$HAS_CSV_BLOB" ]]; then
    log "upload scenario CSV blob"
    python3 - "$SCENARIO_FILE" "$RUN_DIR/blob.req.json" <<'PY'
import json
import sys

scenario_path, output_path = sys.argv[1:]
scenario = json.load(open(scenario_path))
payload = {
    "filename": scenario["csv_filename"],
    "mime_type": "text/csv",
    "content": scenario["csv_content"],
}
with open(output_path, "w") as f:
    json.dump(payload, f)
PY
    curl -sS -X POST "$ELSPETH_EVAL_BASE_URL/api/sessions/$SID/blobs/inline" \
        -H "Authorization: Bearer $JWT" \
        -H "Content-Type: application/json" \
        --data @"$RUN_DIR/blob.req.json" \
        -o "$RUN_DIR/blob.json"
fi

PROMPT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["opening_prompt"])' "$SCENARIO_FILE")"
PAYLOAD="$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1]}))" "$PROMPT")"

log "send opening prompt (may take 30-180s for the composer to converge)"
START=$(date +%s)
HTTP_CODE=$(curl -sS -X POST "$ELSPETH_EVAL_BASE_URL/api/sessions/$SID/messages" \
    -H "Authorization: Bearer $JWT" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    --max-time 240 \
    -w "%{http_code}" \
    -o "$RUN_DIR/send.json")
END=$(date +%s)
log "POST /messages -> HTTP $HTTP_CODE (elapsed: $((END-START))s)"

log "fetch full message history (with audit-grade tool rows for tool-sequence scoring)"
# include_tool_rows=true: persisted role=tool rows carry the tool-result envelope
# (success/false on set_pipeline rejection lives here). include_raw_content=true:
# surfaces the model's pre-synthesis prose for assistant turns intercepted by the
# empty-state synthesizer. limit=500 raised from default 100 so trajectories
# longer than a default page (the gov-pages-rate-cool scenario specifically
# scores >12-call trajectories) are not silently truncated.
curl -sS -H "Authorization: Bearer $JWT" \
    "$ELSPETH_EVAL_BASE_URL/api/sessions/$SID/messages?include_tool_rows=true&include_raw_content=true&limit=500" \
    -o "$RUN_DIR/messages.json"

log "fetch final composition state"
curl -sS -H "Authorization: Bearer $JWT" \
    "$ELSPETH_EVAL_BASE_URL/api/sessions/$SID/state" \
    -o "$RUN_DIR/state.json"

log "score against scenario criteria"
python3 "$HERE/score.py" "$SCENARIO_FILE" "$RUN_DIR/messages.json" "$RUN_DIR/state.json" \
    | tee "$RUN_DIR/scoring.json"

# echo session URL for inspection
echo "session URL: $ELSPETH_EVAL_BASE_URL/#/$SID/spec" >&2
echo "$SID" > "$RUN_DIR/session_id.txt"
