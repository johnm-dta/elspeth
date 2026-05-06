#!/usr/bin/env bash
# Wrapper: run all scenarios under scenarios/ for a multi-scenario cohort.
#
# Usage:
#   ./run_all_scenarios.sh [run_label] [runs_per_scenario]
#
# Defaults: run_label="cohort", runs_per_scenario=6.
#
# For each scenario directory under scenarios/, fires N runs via run_scenario.sh
# with ELSPETH_RGR_SCENARIO set to the scenario's scenario.json. Run dirs land
# under runs/<utc-ts>-<scenario-name>-<label>-<n>/.
#
# Cost note: ~$1/run on the current gpt-5-mini deploy (operator confirmed
# 2026-05-06). 4 scenarios x 6 runs = 24 runs ~= $24. Set runs_per_scenario=1
# for a smoke test.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:-cohort}"
N="${2:-6}"

: "${ELSPETH_EVAL_BASE_URL:?set ELSPETH_EVAL_BASE_URL}"
: "${ELSPETH_EVAL_USER:?set ELSPETH_EVAL_USER}"
: "${ELSPETH_EVAL_PASS:?set ELSPETH_EVAL_PASS}"

shopt -s nullglob
scenario_files=("$HERE"/scenarios/*/scenario.json)
if [[ ${#scenario_files[@]} -eq 0 ]]; then
  echo "ERROR: no scenarios under $HERE/scenarios/*/scenario.json" >&2
  exit 64
fi

echo "[run_all] $((${#scenario_files[@]} * N)) total runs (${#scenario_files[@]} scenarios x $N runs)" >&2

for scenario_file in "${scenario_files[@]}"; do
  scenario_name="$(basename "$(dirname "$scenario_file")")"
  echo "[run_all] === $scenario_name ===" >&2
  for ((i=1; i<=N; i++)); do
    echo "[run_all] $scenario_name run $i/$N" >&2
    ELSPETH_RGR_SCENARIO="$scenario_file" "$HERE/run_scenario.sh" "$LABEL-$i" || {
      rc=$?
      echo "[run_all] WARNING: run $i for $scenario_name exited $rc; continuing" >&2
    }
  done
done

echo "[run_all] DONE — see runs/<ts>-<scenario>-$LABEL-* for verdicts" >&2
