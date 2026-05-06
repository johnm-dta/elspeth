#!/usr/bin/env bash
# Basic-mode finalize step — mirrors hardmode/finalize_scenario.sh.
#
# Captures, for one basic-mode scenario session:
#   - final composed pipeline YAML        (final_yaml.json)
#   - full conversation                   (messages.json)
#   - run summary + diagnostics           (run.json, diagnostics.json)
#   - per-row engine outputs              (outputs/MANIFEST.json + content files)
#
# The basic-mode harness is shape-asymmetric with hardmode:
# basic scenarios were captured per-session as HTTP-driver evidence
# (msg{N}.json, blob.json, etc.) but no run-level finalize step was
# previously wired up. This closes that gap (Phase D.2 of
# elspeth-77d2641032).
#
# Caveat: like hardmode/finalize_scenario.sh, this captures the LATEST
# run for the session via /api/sessions/{sid}/runs. If a session had
# multiple /execute attempts (v1 → v3 fix), only the most recent one
# lands here. See elspeth-obs-e87152484a.
#
# Usage:
#   env JWT=<bearer-token> bash finalize_scenario.sh <scenario_dir_name>
# e.g.
#   env JWT=$(<jwt.txt) bash finalize_scenario.sh s4

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scenario_id=$1
out=$ROOT/$scenario_id

if [[ ! -d "$out" ]]; then
  echo "[$scenario_id] scenario dir $out does not exist" >&2
  exit 1
fi
if [[ ! -f "$out/sid.txt" ]]; then
  echo "[$scenario_id] no sid.txt — cannot finalize" >&2
  exit 1
fi
sid=$(cat "$out/sid.txt")
J=${JWT:?JWT must be exported}

API="https://elspeth.foundryside.dev"

# Final composed pipeline YAML
curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/sessions/$sid/state/yaml" \
  -o "$out/final_yaml.json" 2>/dev/null \
  || { echo "[$scenario_id] state/yaml fetch failed" >&2; echo '{}' > "$out/final_yaml.json"; }

# Full conversation
curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/sessions/$sid/messages" \
  -o "$out/messages.json" 2>/dev/null \
  || echo '[]' > "$out/messages.json"

# Latest run for this session
runs_json=$(curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/sessions/$sid/runs" 2>/dev/null || echo '[]')
rid=$(echo "$runs_json" | jq -r '
  if type == "array" then (.[0].run_id // "")
  elif (.runs // []) | length > 0 then .runs[0].run_id
  else ""
  end
')

if [[ -z "$rid" || "$rid" == "null" ]]; then
  echo "[$scenario_id] no run found for session $sid (session may not have reached /execute)" >&2
  exit 0
fi

echo "[$scenario_id] capturing run $rid" >&2

# Run summary
curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/runs/$rid" -o "$out/run.json" \
  || { echo "[$scenario_id] run.json fetch failed" >&2; exit 1; }

# Diagnostics (per-row state machine + first 20 artifacts)
curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/runs/$rid/diagnostics" \
  -o "$out/diagnostics.json" 2>/dev/null || true

# Outputs manifest (full artifact list, no preview cap)
mkdir -p "$out/outputs"
curl -fsS -H "Authorization: Bearer $J" \
  "$API/api/runs/$rid/outputs" \
  -o "$out/outputs/MANIFEST.json" 2>/dev/null \
  || echo '{"artifacts":[]}' > "$out/outputs/MANIFEST.json"

# Each artifact's bytes via the content endpoint (path-allowlist enforced server-side).
jq -r '.artifacts[]? | "\(.artifact_id)\t\(.path_or_uri | sub("^file://"; "") | split("/") | .[-1])"' \
  "$out/outputs/MANIFEST.json" | while IFS=$'\t' read -r aid name; do
  [ -z "$aid" ] && continue
  safe_name="${aid}__${name}"
  curl -fsS -H "Authorization: Bearer $J" \
    "$API/api/runs/$rid/outputs/$aid/content" \
    -o "$out/outputs/$safe_name" 2>/dev/null \
    || echo "[$scenario_id] failed to fetch artifact $aid" >&2
done

echo "[$scenario_id] done (rid=$rid; artifacts=$(jq '.artifacts | length' "$out/outputs/MANIFEST.json"))"
