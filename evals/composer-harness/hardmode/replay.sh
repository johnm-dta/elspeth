#!/usr/bin/env bash
# evals/composer-harness/hardmode/replay.sh — engine-only replay of a captured scenario.
#
# Usage:
#   replay.sh <run_dir>                 # replay validate+execute against final_yaml.json
#   replay.sh <run_dir> --yaml-file F   # replay against a specific YAML (e.g. an edited copy)
#
# Why this exists:
# A full hard-mode scenario costs $1-2 and 60-300s in LLM convergence. When all
# you want to verify is "did the engine fix actually fix the run failure", you
# don't need to redo the persona dialogue — you need to replay the engine half
# against the same YAML. This script does that:
#   1. Reads final_yaml.json (or --yaml-file) from the captured run dir.
#   2. Creates a NEW session and loads the YAML via POST /state/yaml.
#   3. Re-uploads the same blob (from blob.req.json) if applicable.
#   4. Runs validate -> execute -> poll-to-terminal.
#   5. Writes the new run.json + diagnostics.json into <run_dir>/replays/<utc-ts>/
#      so the original capture stays intact.
#
# Exit codes:
#   0 ok / 64 bad usage / 67 missing file / 70 login / 76 yaml import failed

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_SCRIPT_DIR="$HARNESS_DIR"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

run_dir=""
yaml_file=""

while (( $# > 0 )); do
  case "$1" in
    --yaml-file) yaml_file=$2; shift ;;
    -h|--help)
      sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    -*) evals_die 64 "unknown flag: $1" ;;
    *)
      if [[ -n "$run_dir" ]]; then evals_die 64 "extra positional arg: $1"; fi
      run_dir=$1
      ;;
  esac
  shift
done

[[ -n "$run_dir" ]] || evals_die 64 "usage: replay.sh <run_dir> [--yaml-file F]"
[[ -d "$run_dir" ]] || evals_die 67 "run dir not found: $run_dir"

evals_load_env --require-creds
evals_require_tools

# Resolve YAML source.
if [[ -z "$yaml_file" ]]; then
  yaml_file="$run_dir/final_yaml.json"
fi
[[ -s "$yaml_file" ]] || evals_die 67 "YAML source not found / empty: $yaml_file"

# final_yaml.json is the API response { "yaml": "..." } — extract the .yaml field
# unless the file is already a raw YAML doc.
yaml_payload=$(mktemp -t evals_replay_yaml.XXXXXX)
trap 'rm -f "$yaml_payload"' EXIT
source_blob_count=0
source_blob_sidecar_file=""
if jq -e '.yaml' "$yaml_file" >/dev/null 2>&1; then
  jq -r '.yaml' "$yaml_file" > "$yaml_payload"
  source_blob_count=$(jq -r '(.source_blob_ids // {}) | length' "$yaml_file")
  source_blob_sidecar_file="$yaml_file"
else
  cp "$yaml_file" "$yaml_payload"
  if [[ -s "$run_dir/final_yaml.json" ]] && jq -e '.source_blob_ids | type == "object"' "$run_dir/final_yaml.json" >/dev/null 2>&1; then
    source_blob_count=$(jq -r '(.source_blob_ids // {}) | length' "$run_dir/final_yaml.json")
    source_blob_sidecar_file="$run_dir/final_yaml.json"
  fi
fi
[[ -s "$yaml_payload" ]] || evals_die 67 "extracted YAML is empty"

# Replay output dir.
ts=$(date -u +%Y%m%dT%H%M%SZ)
replay_dir="$run_dir/replays/$ts"
mkdir -p "$replay_dir"

export EVALS_OUT_DIR="$replay_dir"
export EVALS_LOG_FILE="$replay_dir/replay.log"
export EVALS_JWT_FILE="$replay_dir/jwt.txt"

evals_log INFO "replay: source=$yaml_file -> $replay_dir"
evals_login

# New session.
scen_meta=""
if [[ -s "$run_dir/scenario.json" ]]; then
  scen_meta=$(jq -r '.scenario_id // "unknown"' "$run_dir/scenario.json")
fi
sid=$(evals_create_session "replay/$scen_meta @ $ts")

# Re-upload blob if original had one.
uploaded_blob_id=""
if [[ -s "$run_dir/blob.req.json" ]]; then
  evals_log INFO "re-uploading blob from $run_dir/blob.req.json"
  filename=$(jq -r '.filename' "$run_dir/blob.req.json")
  mime=$(jq -r '.mime_type' "$run_dir/blob.req.json")
  content_tmp=$(mktemp -t evals_replay_blob.XXXXXX)
  trap 'rm -f "$yaml_payload" "$content_tmp"' EXIT
  jq -r '.content' "$run_dir/blob.req.json" > "$content_tmp"
  evals_upload_blob "$sid" "$filename" "$mime" "$content_tmp"
  uploaded_blob_id=$(jq -r '.id // empty' "$replay_dir/blob.json")
fi
if (( source_blob_count > 0 )) && [[ -z "$uploaded_blob_id" ]]; then
  evals_die 67 "final YAML references source blobs but $run_dir/blob.req.json was not available for replay upload"
fi
if (( source_blob_count > 1 )); then
  evals_die 76 "replay supports one captured source blob; final YAML references $source_blob_count"
fi

# Import YAML into the session.
evals_log INFO "importing YAML into session $sid"
import_resp="$replay_dir/import.json"
if (( source_blob_count == 1 )); then
  import_body=$(jq -nc --rawfile y "$yaml_payload" --arg id "$uploaded_blob_id" --slurpfile src "$source_blob_sidecar_file" \
    '{yaml:$y, source_blob_ids:(($src[0].source_blob_ids // {}) | with_entries(.value=$id))}')
else
  import_body=$(jq -nc --rawfile y "$yaml_payload" '{yaml:$y}')
fi
http=$(_evals_http_post_json \
         "$ELSPETH_EVAL_BASE_URL/api/sessions/$sid/state/yaml" \
         "$import_body" \
         "$import_resp" || echo 000)
if [[ "$http" != 2* ]]; then
  evals_die 76 "YAML import failed (HTTP $http): $(head -c 500 "$import_resp" 2>/dev/null)"
fi
evals_log INFO "YAML import OK"

# Validate
evals_validate "$sid" "$replay_dir/validate.json"
is_valid=$(jq -r '.is_valid // false' "$replay_dir/validate.json")

if [[ "$is_valid" != "true" ]]; then
  evals_log WARN "validate failed on replay — see $replay_dir/validate.json"
  exit 0
fi

# Execute + poll
evals_execute "$sid" "$replay_dir/execute.json" "$replay_dir/execute.code"
http=$(cat "$replay_dir/execute.code")
if [[ "$http" != "202" ]]; then
  evals_log ERROR "execute returned HTTP $http; abort"
  exit 0
fi
run_id=$(jq -r '.run_id' "$replay_dir/execute.json")
evals_poll_run_terminal "$run_id" \
  "$ELSPETH_EVAL_RUN_TIMEOUT_SEC" \
  "$ELSPETH_EVAL_RUN_POLL_INTERVAL" \
  "$replay_dir/run.json" || true

evals_get_diagnostics "$run_id" "$replay_dir/diagnostics.json" 2>/dev/null || \
  echo '{}' > "$replay_dir/diagnostics.json"

# Compact summary
python3 - "$run_dir" "$replay_dir" "$run_id" <<'PY'
import json, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
run_id = sys.argv[3]

def status_of(p):
    if not p.exists() or not p.read_text().strip():
        return None
    try:
        return json.loads(p.read_text()).get("status")
    except Exception:
        return None

orig_status = status_of(src / "run.json")
new_status = status_of(dst / "run.json")
out = dst / "replay_summary.json"
out.write_text(json.dumps({
    "source_run_dir": str(src),
    "replay_dir": str(dst),
    "replay_run_id": run_id,
    "original_run_status": orig_status,
    "replay_run_status": new_status,
    "delta": "regression" if (orig_status and orig_status.startswith("completed") and (new_status or "").startswith("failed"))
             else "improvement" if (orig_status and orig_status.startswith("failed") and (new_status or "").startswith("completed"))
             else "same" if orig_status == new_status
             else "different",
}, indent=2))
print(out.read_text())
PY

evals_log INFO "replay complete: $replay_dir"
