#!/usr/bin/env bash
# evals/lib/common.sh — shared HTTP / env / logging / poll primitives for eval harnesses.
#
# Source from a per-mode harness script:
#   source "$(dirname "${BASH_SOURCE[0]}")/../../lib/common.sh"
#
# Provides:
#   evals_load_env [--require-creds]
#   evals_log <level> <msg...>
#   evals_die <exit_code> <msg...>
#   evals_require_tools (jq, python3, curl)
#   evals_jwt_exp_seconds_remaining <jwt>
#   evals_login                     -> writes JWT to $EVALS_JWT_FILE
#   evals_login_if_needed [margin]  -> refreshes JWT if expiry within margin (default 300s)
#   evals_create_session <title>    -> writes $out/session.json, $out/sid.txt
#   evals_upload_blob <sid> <filename> <mime> <content_file>
#   evals_post_message <sid> <turn> <msg_file>
#   evals_get_state <sid> <out_file>
#   evals_get_progress <sid> <out_file>
#   evals_validate <sid> <out_file>
#   evals_execute <sid> <out_file> <code_file>
#   evals_get_run <run_id> <out_file>
#   evals_get_diagnostics <run_id> <out_file>
#   evals_get_messages <sid> <out_file>
#   evals_get_yaml <sid> <out_file>
#   evals_try_get <url> <out_file> [<code_file>]
#   evals_poll_run_terminal <run_id> <timeout_sec> <interval_sec> <out_file>
#
# Required environment (loaded by evals_load_env from $EVALS_ENV_FILE if present, else process env):
#   ELSPETH_EVAL_BASE_URL    - e.g. https://elspeth.foundryside.dev
#   ELSPETH_EVAL_USER        - login username
#   ELSPETH_EVAL_PASS        - login password
# Optional:
#   ELSPETH_EVAL_RUN_TIMEOUT_SEC   default 300
#   ELSPETH_EVAL_RUN_POLL_INTERVAL default 3
#   ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC  default 300 (refresh JWT if exp within this many seconds)
#   ELSPETH_EVAL_CURL_MAX_TIME     default 240 (per-call ceiling)

set -euo pipefail

# ---------- Logging ----------------------------------------------------------

# evals_log <level> <msg...>
# Writes to stderr AND, if EVALS_LOG_FILE is set, appends there too.
evals_log() {
  local level=$1; shift
  local ts
  ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  local line
  printf -v line '[%s] [%s] %s' "$ts" "$level" "$*"
  printf '%s\n' "$line" >&2
  if [[ -n "${EVALS_LOG_FILE:-}" ]]; then
    printf '%s\n' "$line" >> "$EVALS_LOG_FILE"
  fi
}

evals_die() {
  local code=$1; shift
  evals_log ERROR "$*"
  exit "$code"
}

# ---------- Environment ------------------------------------------------------

evals_load_env() {
  local require_creds=0
  while (( $# > 0 )); do
    case "$1" in
      --require-creds) require_creds=1 ;;
      *) evals_die 64 "evals_load_env: unknown flag: $1" ;;
    esac
    shift
  done

  # Source $EVALS_ENV_FILE if set and readable; else fall back to ./.env in caller dir.
  local env_file="${EVALS_ENV_FILE:-}"
  if [[ -z "$env_file" ]]; then
    # Walk upward from caller dir looking for .env (max 5 levels).
    local dir="${EVALS_SCRIPT_DIR:-$PWD}"
    local i=0
    while (( i < 5 )); do
      if [[ -f "$dir/.env" ]]; then
        env_file="$dir/.env"
        break
      fi
      dir=$(dirname "$dir")
      i=$((i + 1))
    done
  fi

  if [[ -n "$env_file" && -f "$env_file" ]]; then
    evals_log INFO "loading env from $env_file"
    # shellcheck disable=SC1090
    set -a; source "$env_file"; set +a
  fi

  : "${ELSPETH_EVAL_BASE_URL:=}"
  : "${ELSPETH_EVAL_RUN_TIMEOUT_SEC:=300}"
  : "${ELSPETH_EVAL_RUN_POLL_INTERVAL:=3}"
  : "${ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC:=300}"
  : "${ELSPETH_EVAL_CURL_MAX_TIME:=240}"

  if [[ -z "$ELSPETH_EVAL_BASE_URL" ]]; then
    evals_die 64 "ELSPETH_EVAL_BASE_URL is not set (export it or put in .env)"
  fi
  # Strip trailing slash
  ELSPETH_EVAL_BASE_URL="${ELSPETH_EVAL_BASE_URL%/}"

  if (( require_creds )); then
    if [[ -z "${ELSPETH_EVAL_USER:-}" || -z "${ELSPETH_EVAL_PASS:-}" ]]; then
      evals_die 64 "ELSPETH_EVAL_USER / ELSPETH_EVAL_PASS must be set when --require-creds"
    fi
  fi

  export ELSPETH_EVAL_BASE_URL ELSPETH_EVAL_RUN_TIMEOUT_SEC \
         ELSPETH_EVAL_RUN_POLL_INTERVAL ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC \
         ELSPETH_EVAL_CURL_MAX_TIME
}

# ---------- Tooling ----------------------------------------------------------

evals_require_tools() {
  local missing=()
  for t in jq python3 curl; do
    command -v "$t" >/dev/null 2>&1 || missing+=("$t")
  done
  if (( ${#missing[@]} > 0 )); then
    evals_die 69 "missing required tools: ${missing[*]}"
  fi
}

# ---------- Curl secret hygiene ----------------------------------------------

_evals_curl_temp_file_from_string() {
  local content=$1
  local tmp
  tmp=$(mktemp -t evals-curl.XXXXXX)
  chmod 600 "$tmp"
  printf '%s' "$content" > "$tmp"
  printf '%s' "$tmp"
}

_evals_auth_header_file() {
  : "${EVALS_JWT_FILE:?EVALS_JWT_FILE not set}"
  local jwt
  jwt=$(cat "$EVALS_JWT_FILE")
  _evals_curl_temp_file_from_string "Authorization: Bearer $jwt"$'\n'
}

# ---------- JWT --------------------------------------------------------------

# evals_jwt_exp_seconds_remaining <jwt> -> echoes seconds until exp, or empty if undecodable.
evals_jwt_exp_seconds_remaining() {
  local jwt=$1
  [[ -n "$jwt" ]] || { echo ""; return 0; }
  local payload b64
  b64=$(printf '%s' "$jwt" | awk -F. '{print $2}')
  [[ -n "$b64" ]] || { echo ""; return 0; }
  # Pad with '=' to multiple of 4
  local pad=$((4 - ${#b64} % 4))
  if (( pad < 4 )); then b64="${b64}$(printf '=%.0s' $(seq 1 $pad))"; fi
  payload=$(printf '%s' "$b64" | tr '_-' '/+' | base64 -d 2>/dev/null || true)
  [[ -n "$payload" ]] || { echo ""; return 0; }
  local exp now
  exp=$(printf '%s' "$payload" | jq -r '.exp // empty' 2>/dev/null || echo "")
  [[ -n "$exp" ]] || { echo ""; return 0; }
  now=$(date +%s)
  echo $((exp - now))
}

# evals_login -> writes JWT to $EVALS_JWT_FILE (must be set by caller)
evals_login() {
  : "${EVALS_JWT_FILE:?EVALS_JWT_FILE not set}"
  : "${ELSPETH_EVAL_USER:?ELSPETH_EVAL_USER not set}"
  : "${ELSPETH_EVAL_PASS:?ELSPETH_EVAL_PASS not set}"

  evals_log INFO "logging in as $ELSPETH_EVAL_USER"
  local body body_file resp http
  body=$(jq -nc --arg u "$ELSPETH_EVAL_USER" --arg p "$ELSPETH_EVAL_PASS" \
           '{username:$u, password:$p}')
  body_file=$(_evals_curl_temp_file_from_string "$body")
  resp=$(curl -sS --max-time "$ELSPETH_EVAL_CURL_MAX_TIME" \
           -X POST "$ELSPETH_EVAL_BASE_URL/api/auth/login" \
           -H 'Content-Type: application/json' \
           --data "@$body_file" \
           -w '\n%{http_code}' || printf '\n000')
  rm -f "$body_file"
  http=$(printf '%s' "$resp" | tail -n1)
  body=$(printf '%s' "$resp" | sed '$d')
  if [[ "$http" != "200" ]]; then
    evals_die 70 "login failed (HTTP $http): $body"
  fi
  local token
  token=$(printf '%s' "$body" | jq -r '.access_token // empty')
  [[ -n "$token" ]] || evals_die 70 "login succeeded but no access_token in response"
  printf '%s' "$token" > "$EVALS_JWT_FILE"
  chmod 600 "$EVALS_JWT_FILE"
  evals_log INFO "JWT written to $EVALS_JWT_FILE"
}

# evals_login_if_needed [margin_sec]
# Refreshes JWT if file missing, empty, or exp within margin (default $ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC).
evals_login_if_needed() {
  local margin=${1:-$ELSPETH_EVAL_JWT_REFRESH_MARGIN_SEC}
  : "${EVALS_JWT_FILE:?EVALS_JWT_FILE not set}"
  if [[ ! -s "$EVALS_JWT_FILE" ]]; then
    evals_login
    return
  fi
  local jwt remaining
  jwt=$(cat "$EVALS_JWT_FILE")
  remaining=$(evals_jwt_exp_seconds_remaining "$jwt")
  if [[ -z "$remaining" ]]; then
    evals_log WARN "could not decode JWT exp; refreshing"
    evals_login
    return
  fi
  if (( remaining < margin )); then
    evals_log INFO "JWT expires in ${remaining}s (margin=${margin}s); refreshing"
    evals_login
  else
    evals_log DEBUG "JWT has ${remaining}s remaining"
  fi
}

# ---------- HTTP helpers -----------------------------------------------------

# _evals_http_get <url> <out_file>
# Writes body to out_file; on non-2xx, dies.
_evals_http_get() {
  local url=$1 out=$2
  evals_login_if_needed
  local auth_header_file http
  auth_header_file=$(_evals_auth_header_file)
  if ! http=$(curl -sS --max-time "$ELSPETH_EVAL_CURL_MAX_TIME" \
                -H "@$auth_header_file" \
                -o "$out" -w '%{http_code}' \
                "$url"); then
    http=000
  fi
  rm -f "$auth_header_file"
  if [[ "$http" != 2* ]]; then
    local snippet
    snippet=$(head -c 500 "$out" 2>/dev/null || true)
    evals_die 71 "GET $url failed (HTTP $http): $snippet"
  fi
}

# evals_try_get <url> <out_file> [<code_file>]
# Writes body to out_file and returns success for 2xx. Unlike _evals_http_get,
# non-2xx responses are data for callers to record; this helper must not exit.
evals_try_get() {
  local url=$1 out=$2 code_out=${3:-}
  evals_login_if_needed
  local auth_header_file http
  auth_header_file=$(_evals_auth_header_file)
  if ! http=$(curl -sS --max-time "$ELSPETH_EVAL_CURL_MAX_TIME" \
                -H "@$auth_header_file" \
                -o "$out" -w '%{http_code}' \
                "$url"); then
    http=000
  fi
  rm -f "$auth_header_file"
  if [[ -n "$code_out" ]]; then
    printf '%s' "$http" > "$code_out"
  fi
  if [[ "$http" == 2* ]]; then
    return 0
  fi
  return 1
}

# _evals_http_post_json <url> <json_body_file_or_string> <out_file> [code_out_file]
# Allows non-2xx without dying (caller decides).
_evals_http_post_json() {
  local url=$1 body=$2 out=$3 code_out=${4:-}
  evals_login_if_needed
  local auth_header_file http data_arg=() body_file=""
  auth_header_file=$(_evals_auth_header_file)
  if [[ -f "$body" ]]; then
    data_arg=(--data "@$body")
  else
    body_file=$(_evals_curl_temp_file_from_string "$body")
    data_arg=(--data "@$body_file")
  fi
  http=$(curl -sS --max-time "$ELSPETH_EVAL_CURL_MAX_TIME" \
           -X POST "$url" \
           -H "@$auth_header_file" \
           -H 'Content-Type: application/json' \
           "${data_arg[@]}" \
           -o "$out" -w '%{http_code}' || echo "000")
  rm -f "$auth_header_file" "$body_file"
  if [[ -n "$code_out" ]]; then
    printf '%s' "$http" > "$code_out"
  fi
  printf '%s' "$http"
}

# ---------- API surface ------------------------------------------------------

# evals_create_session <title> -> writes $out/session.json, echoes session_id
# Required env: $EVALS_OUT_DIR
evals_create_session() {
  local title=$1
  : "${EVALS_OUT_DIR:?EVALS_OUT_DIR not set}"
  local body http sid
  body=$(jq -nc --arg t "$title" '{title:$t}')
  http=$(_evals_http_post_json \
           "$ELSPETH_EVAL_BASE_URL/api/sessions" \
           "$body" \
           "$EVALS_OUT_DIR/session.json")
  if [[ "$http" != 2* ]]; then
    evals_die 72 "create_session failed (HTTP $http): $(head -c 500 "$EVALS_OUT_DIR/session.json" 2>/dev/null)"
  fi
  sid=$(jq -r '.id' "$EVALS_OUT_DIR/session.json")
  [[ -n "$sid" && "$sid" != "null" ]] || evals_die 72 "create_session response missing id"
  printf '%s' "$sid" > "$EVALS_OUT_DIR/sid.txt"
  evals_log INFO "session created: $sid"
  echo "$sid"
}

# evals_upload_blob <sid> <filename> <mime> <content_file>
# Writes $EVALS_OUT_DIR/blob.req.json and blob.json.
evals_upload_blob() {
  local sid=$1 filename=$2 mime=$3 content_file=$4
  : "${EVALS_OUT_DIR:?EVALS_OUT_DIR not set}"
  jq -n --arg fn "$filename" --arg mt "$mime" --rawfile c "$content_file" \
        '{filename:$fn, mime_type:$mt, content:$c}' \
        > "$EVALS_OUT_DIR/blob.req.json"
  local http
  http=$(_evals_http_post_json \
           "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/blobs/inline" \
           "$EVALS_OUT_DIR/blob.req.json" \
           "$EVALS_OUT_DIR/blob.json")
  if [[ "$http" != 2* ]]; then
    evals_die 73 "upload_blob failed (HTTP $http): $(head -c 500 "$EVALS_OUT_DIR/blob.json" 2>/dev/null)"
  fi
  evals_log INFO "blob uploaded: $(jq -r '.id // empty' "$EVALS_OUT_DIR/blob.json")"
}

# evals_post_message <sid> <turn> <msg_file>
# msg_file is plain text; wrapped into {"content":...} envelope.
# Writes msg.t<N>.{req,resp,curl_meta}.json
evals_post_message() {
  local sid=$1 turn=$2 msg_file=$3
  : "${EVALS_OUT_DIR:?EVALS_OUT_DIR not set}"
  local out=$EVALS_OUT_DIR
  jq -n --rawfile c "$msg_file" '{content:$c}' > "$out/msg.t${turn}.req.json"

  evals_login_if_needed
  local auth_header_file http_time start end wall
  auth_header_file=$(_evals_auth_header_file)
  start=$(date +%s.%N)
  http_time=$(curl -sS --max-time "$ELSPETH_EVAL_CURL_MAX_TIME" \
                -X POST "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/messages" \
                -H "@$auth_header_file" \
                -H 'Content-Type: application/json' \
                --data "@$out/msg.t${turn}.req.json" \
                -o "$out/msg.t${turn}.resp.json" \
                -w '%{http_code} %{time_total}\n' || echo "000 0.00")
  rm -f "$auth_header_file"
  end=$(date +%s.%N)
  wall=$(awk "BEGIN{printf \"%.2f\", $end - $start}")
  printf '%s\n%.2f\n' "$http_time" "$wall" > "$out/msg.t${turn}.curl_meta"
  local http=${http_time%% *}
  evals_log INFO "post_message turn=$turn http=$http wall=${wall}s"
  if [[ "$http" != 2* ]]; then
    evals_log WARN "post_message non-2xx (HTTP $http) — body preserved at msg.t${turn}.resp.json"
  fi
}

evals_get_state()       { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/sessions/$1/state" "$2"; }
evals_get_progress()    { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/sessions/$1/composer-progress" "$2"; }
evals_get_messages()    { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/sessions/$1/messages" "$2"; }
# Diagnostic-only variant — returns each assistant message's pre-synthesis raw_content
# (the model's actual prose when the empty-state synthesizer at service.py:_finalize_no_tool_response
# replaced the visible content). Required for diagnosing convergence failures
# where the synthesized blocker hides what the model actually produced.
evals_get_messages_with_raw() { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/sessions/$1/messages?include_raw_content=true" "$2"; }
evals_get_yaml()        { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/sessions/$1/state/yaml" "$2"; }

evals_validate() {
  local sid=$1 out=$2
  local http
  http=$(_evals_http_post_json \
           "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/validate" \
           '{}' \
           "$out")
  if [[ "$http" != 2* ]]; then
    local snippet
    snippet=$(head -c 500 "$out" 2>/dev/null || true)
    evals_die 74 "validate failed (HTTP $http): $snippet"
  fi
}

# evals_execute <sid> <out_file> <code_file>
evals_execute() {
  local sid=$1 out=$2 code_out=$3
  _evals_http_post_json \
    "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/execute" \
    '{}' \
    "$out" \
    "$code_out" >/dev/null
}

evals_get_run()         { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/runs/$1" "$2"; }
evals_get_diagnostics() { _evals_http_get "$ELSPETH_EVAL_BASE_URL/api/runs/$1/diagnostics" "$2"; }

# ---------- Run polling ------------------------------------------------------

# Terminal RunStatus values per src/elspeth/contracts/enums.py
EVALS_TERMINAL_RUN_STATUSES="completed completed_with_failures failed empty interrupted"

# evals_poll_run_terminal <run_id> [<timeout_sec>] [<interval_sec>] [<out_file>]
# Polls /api/runs/<run_id> until status is terminal or timeout.
# Writes the final response to <out_file> (default $EVALS_OUT_DIR/run.json).
# Returns 0 on terminal, 1 on timeout, 2 on hard HTTP failure.
evals_poll_run_terminal() {
  local run_id=$1
  local timeout=${2:-$ELSPETH_EVAL_RUN_TIMEOUT_SEC}
  local interval=${3:-$ELSPETH_EVAL_RUN_POLL_INTERVAL}
  local out=${4:-$EVALS_OUT_DIR/run.json}
  local elapsed=0 status=""
  while (( elapsed < timeout )); do
    if ! evals_get_run "$run_id" "$out" 2>/dev/null; then
      evals_log WARN "transient GET /api/runs/$run_id failure at t=${elapsed}s; retrying"
      sleep "$interval"
      elapsed=$((elapsed + interval))
      continue
    fi
    status=$(jq -r '.status // empty' "$out" 2>/dev/null || echo "")
    if [[ -n "$status" ]] && [[ " $EVALS_TERMINAL_RUN_STATUSES " == *" $status "* ]]; then
      evals_log INFO "run $run_id terminal: $status (elapsed ${elapsed}s)"
      return 0
    fi
    evals_log DEBUG "run $run_id status=${status:-?} (elapsed ${elapsed}s)"
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done
  evals_log ERROR "run $run_id did not reach terminal within ${timeout}s (last status=${status:-?})"
  return 1
}

# ---------- Scenario fixture loader ------------------------------------------

# evals_resolve_scenario <scenarios_dir> <scenario_id> -> echoes path to fixture json
evals_resolve_scenario() {
  local dir=$1 scenario_id=$2
  local matches=()
  shopt -s nullglob
  matches=( "$dir"/"$scenario_id"_*.json )
  shopt -u nullglob
  if [[ -f "$dir/$scenario_id.json" ]]; then
    matches+=( "$dir/$scenario_id.json" )
  fi
  if (( ${#matches[@]} == 0 )); then
    evals_die 65 "scenario not found: $scenario_id (looked in $dir)"
  fi
  if (( ${#matches[@]} > 1 )); then
    evals_die 65 "scenario_id ambiguous: $scenario_id matched ${#matches[@]} fixtures: ${matches[*]}"
  fi
  echo "${matches[0]}"
}
