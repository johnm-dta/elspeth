#!/usr/bin/env bash
# Stopgap subtree runner for the codex panel review.
# Plan 1 is single-file; this loops it over a subtree (concurrent, resumable) until
# Plan 2's worker pool lands. DELETE once Plan 2 ships. Untracked by design.
#
# Knobs (env override or edit the defaults):
#   SUBTREE : repo-relative dir to review        (default: src/elspeth/web)
#   WORKERS : concurrent files = concurrent codex calls (pilot sustained 4, no 429)
#   DRYRUN  : 1 = print the plan + per-file actions, make NO codex calls (do this first!)
#
# Safety: codex runs from a CLEAN detached worktree, so untracked secrets
# (.env, data/*.db) are absent from the --cd --sandbox scope. Findings are written
# to a gitignored MAIN-repo path, so they survive worktree teardown and are never
# committed. Re-running skips files already done (resume).
set -uo pipefail

MAIN="$(git rev-parse --show-toplevel)"
SUBTREE="${SUBTREE:-src/elspeth/web}"
WORKERS="${WORKERS:-4}"
DRYRUN="${DRYRUN:-0}"
WT="$MAIN/.worktrees/panel-review"
OUT="$MAIN/docs/quality-audit/findings-panel-raw/${SUBTREE//\//_}"
PY="$MAIN/.venv/bin/python"
REV="$(git -C "$MAIN" rev-parse HEAD)"

# 1) clean worktree at current HEAD (tracked files only); idempotent + kept fresh
if [ -d "$WT" ]; then git -C "$WT" checkout -q "$REV"; else git -C "$MAIN" worktree add --detach "$WT" "$REV"; fi

# 2) enumerate targets (exclude the JS frontend)
mapfile -t FILES < <(cd "$WT" && find "$SUBTREE" -name '*.py' -not -path '*/frontend/*' | sort)
mkdir -p "$OUT/_logs"
echo "subtree=$SUBTREE  files=${#FILES[@]}  (~$(( ${#FILES[@]} * 2 )) codex calls)  workers=$WORKERS  dryrun=$DRYRUN"
echo "worktree=$WT"
echo "findings=$OUT  (gitignored, durable)"
[ "${#FILES[@]}" -eq 0 ] && { echo "no .py files under $SUBTREE"; exit 0; }

review_one() {
  local rel="$1" safe dir
  safe="${rel//\//_}"; dir="$OUT/$safe"
  if find "$dir" -name '*.security-architect.md.structured.json' 2>/dev/null | grep -q . \
     && find "$dir" -name '*.solution-architect.md.structured.json' 2>/dev/null | grep -q . ; then
     echo "skip(done)  $rel"; return 0
  fi
  if [ "$DRYRUN" = "1" ]; then echo "WOULD review  $rel"; return 0; fi
  "$PY" "$WT/scripts/codex_panel_review.py" --file "$WT/$rel" --output-dir "$dir" \
      > "$OUT/_logs/$safe.out" 2>&1
  echo "done($?)  $rel  :: $(tail -1 "$OUT/_logs/$safe.out" 2>/dev/null)"
}
export -f review_one ; export WT PY OUT DRYRUN

printf '%s\n' "${FILES[@]}" | xargs -P "$WORKERS" -I{} bash -c 'review_one "$@"' _ {}

echo
echo "=== complete. findings under: $OUT ==="
echo "    failures/429 scan : grep -rEi 'failed=[1-9]|429|invalid_json_schema' '$OUT'/_logs | head"
echo "    teardown worktree : git -C '$MAIN' worktree remove --force '$WT'"
