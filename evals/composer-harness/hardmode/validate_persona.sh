#!/usr/bin/env bash
# evals/composer-harness/hardmode/validate_persona.sh — linguistic-constraint check.
#
# Usage: validate_persona.sh <run_dir>
#
# For a captured scenario run, parses each turn<N>.user.txt against the
# persona spec's MUST USE / MUST AVOID lists and reports per-turn violations.
#
# Why: the persona-subagent is meant to stay in character. Substantial drift
# is a "scenario invalidating" signal — the transcript looks reasonable but
# isn't probative because it's not the persona's voice. This script gives
# you a quick numeric handle on character-fidelity.
#
# Output: writes <run_dir>/persona_check.json with per-turn match/miss counts.

set -euo pipefail
HARNESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$HARNESS_DIR/.." && pwd)"
LIB_DIR="$(cd "$HARNESS_ROOT/../lib" && pwd)"
# shellcheck source=../../lib/common.sh
source "$LIB_DIR/common.sh"

if (( $# != 1 )); then
  evals_die 64 "usage: validate_persona.sh <run_dir>"
fi
run_dir=$1
[[ -d "$run_dir" ]] || evals_die 67 "run dir not found: $run_dir"
[[ -s "$run_dir/scenario.json" ]] || evals_die 67 "scenario.json missing in $run_dir"

evals_require_tools

persona_id=$(jq -r '.persona' "$run_dir/scenario.json")
persona_spec="$HARNESS_ROOT/personas/${persona_id}.md"
[[ -f "$persona_spec" ]] || evals_die 67 "persona spec not found: $persona_spec"

python3 - "$run_dir" "$persona_spec" <<'PY'
import json, pathlib, re, sys

run_dir = pathlib.Path(sys.argv[1])
spec_path = pathlib.Path(sys.argv[2])
spec = spec_path.read_text()

def extract_quoted_phrases(after_marker):
    """Pull out double-quoted phrases from the line(s) following a marker, stopping at a blank line."""
    phrases = []
    m = re.search(re.escape(after_marker) + r"(.*?)(?:\n\n|\Z)", spec, re.DOTALL | re.IGNORECASE)
    if not m:
        return phrases
    block = m.group(1)
    phrases.extend(re.findall(r'"([^"]+)"', block))
    # Also pick up backticked tokens.
    phrases.extend(re.findall(r'`([^`]+)`', block))
    return phrases

must_use = []
for marker in ("MUST USE", "**MUST USE**"):
    must_use.extend(extract_quoted_phrases(marker))
must_avoid = []
for marker in ("MUST AVOID", "**MUST AVOID**"):
    must_avoid.extend(extract_quoted_phrases(marker))

must_use = list(dict.fromkeys(must_use))      # de-dupe preserving order
must_avoid = list(dict.fromkeys(must_avoid))

turn_files = sorted(run_dir.glob("turn*.user.txt"))
report = {
    "persona": spec_path.stem,
    "must_use_phrases_extracted": must_use,
    "must_avoid_phrases_extracted": must_avoid,
    "turns": [],
    "summary": {},
}

def find_phrases(text, phrases):
    text_lower = text.lower()
    return [p for p in phrases if p.lower() in text_lower]

total_avoid_violations = 0
total_use_hits = 0
for tf in turn_files:
    text = tf.read_text()
    avoid_hits = find_phrases(text, must_avoid)
    use_hits = find_phrases(text, must_use)
    total_avoid_violations += len(avoid_hits)
    total_use_hits += len(use_hits)
    report["turns"].append({
        "file": tf.name,
        "char_count": len(text),
        "must_avoid_violations": avoid_hits,
        "must_use_hits": use_hits,
    })

report["summary"] = {
    "turn_count": len(turn_files),
    "total_must_avoid_violations": total_avoid_violations,
    "total_must_use_hits": total_use_hits,
    "must_use_coverage_ratio": (
        total_use_hits / max(len(turn_files), 1) if must_use else None
    ),
    "verdict": (
        "in-character"
        if total_avoid_violations == 0
        else f"{total_avoid_violations} MUST-AVOID violations across {len(turn_files)} turns — review"
    ),
    "note": (
        "Phrase extraction is heuristic — relies on quoted/backticked phrases in MUST USE / "
        "MUST AVOID sections. Personas with prose-style constraints (e.g. 'avoid hedges') "
        "won't be caught by this. Treat as a screening signal, not a final judgment."
    ),
}

(run_dir / "persona_check.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["summary"], indent=2))
PY
