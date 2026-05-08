#!/usr/bin/env bash
# evals/composer-harness/hardmode/score_scenario.sh
#
# Usage: score_scenario.sh <run_dir>
#
# Calls evals.lib.composer_rgr_score.score(scenario, messages, state)
# against a captured run directory and writes <run_dir>/rgr_score.json.
#
# Why this exists separately from finalize_scenario.sh:
#   * finalize_scenario.sh captures execution metrics (engine ran, rows
#     processed) and writes ledger.json with hardmode-style observability.
#   * Panel cohort scenarios encode RGR-style red/green criteria
#     (must_have_node_chain_in_order, must_include_observed_columns, etc.)
#     that need a separate scoring pass against the final composition state.
#   * Without this step, green_criteria are decorative — finalize only
#     checks valid+executed, not structural fit.
#
# Inputs (per run dir):
#   scenario.json           the cohort-generated scenario fixture
#   messages.json           captured by finalize_scenario.sh
#   state.after.t<N>.json   per-turn state captures; we use the highest N
#                           (the final composition state)
#
# Output:
#   rgr_score.json          {verdict: RED|AMBER|GREEN, ...}  — written
#                           even on failure paths so aggregate.sh always
#                           has a row.
#
# Exit codes:
#   0   rgr_score.json written (verdict is data, not pass/fail)
#   64  usage error
#   67  prerequisite missing (run_dir, scenario.json, messages, state)

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
REPO_ROOT="$(cd "$HARNESS_ROOT/../.." && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

if (( $# != 1 )); then
  evals_die 64 "usage: score_scenario.sh <run_dir>"
fi
run_dir=$1
[[ -d "$run_dir" ]] || evals_die 67 "run dir not found: $run_dir"
[[ -s "$run_dir/scenario.json" ]] || evals_die 67 "scenario.json missing in $run_dir"
[[ -s "$run_dir/messages.json" ]] || evals_die 67 "messages.json missing in $run_dir (run finalize_scenario.sh first)"

evals_require_tools

python3 - "$run_dir" "$REPO_ROOT" <<'PY'
"""Score a captured panel-cohort run against its scenario's RGR criteria."""
from __future__ import annotations

import json
import pathlib
import re
import sys

run_dir = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2])

# evals.lib is a Python package; add repo root so import resolves.
sys.path.insert(0, str(repo_root))
from evals.lib.composer_rgr_score import score  # noqa: E402

scenario = json.loads((run_dir / "scenario.json").read_text())
try:
    messages = json.loads((run_dir / "messages.json").read_text())
except json.JSONDecodeError as exc:
    (run_dir / "rgr_score.json").write_text(json.dumps({
        "verdict": "ERROR",
        "error": f"messages.json malformed: {exc}",
    }, indent=2))
    sys.stderr.write(f"score_scenario: messages.json malformed: {exc}\n")
    sys.exit(0)

# Pick the highest-numbered state.after.t<N>.json — this is the final
# composition state captured at the end of the conversation. If none
# exists, score with state=None (composer_rgr_score handles this — many
# RED conditions can be detected from messages alone).
state_files = sorted(
    run_dir.glob("state.after.t*.json"),
    key=lambda p: int(re.search(r"t(\d+)", p.name).group(1)),
)
state: object | None = None
if state_files:
    try:
        state = json.loads(state_files[-1].read_text())
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"score_scenario: latest state file {state_files[-1].name} malformed: {exc}; "
            f"scoring without state\n"
        )

result = score(scenario, messages, state)

# Annotate provenance so aggregate.sh can show what state file scoring used.
result["_meta"] = {
    "run_dir": str(run_dir),
    "scenario_id": scenario.get("scenario_id"),
    "state_file_used": state_files[-1].name if state_files else None,
    "state_present": state is not None,
}

(run_dir / "rgr_score.json").write_text(json.dumps(result, indent=2))
print(json.dumps({
    "verdict": result.get("verdict"),
    "scenario_id": scenario.get("scenario_id"),
    "state_file_used": result["_meta"]["state_file_used"],
}, indent=2))
PY
