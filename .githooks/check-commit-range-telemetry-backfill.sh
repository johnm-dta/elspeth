#!/usr/bin/env bash
# CI backstop: checks every commit in a given range for cohort-attribution
# trailers.  Local enforcement via .githooks/commit-msg-telemetry-backfill
# only fires when the contributor has installed the dispatcher.  This
# script re-checks every PR commit so the rule is enforced even for
# contributors who skipped install.
#
# Usage: ./check-commit-range-telemetry-backfill.sh <git-range>
#   e.g. ./check-commit-range-telemetry-backfill.sh origin/main..HEAD
#
# Spec: docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md
#       §"Cohort attribution via commit trailers (A4 — load-bearing)"

set -euo pipefail

RANGE="${1:?usage: $0 <git-range>}"

# Iterate over commits in the range, oldest first.  --no-merges skips merge
# commits whose trailers live on the merged-in commits.
mapfile -t COMMITS < <(git log --no-merges --reverse --format='%H' "$RANGE")

if [[ ${#COMMITS[@]} -eq 0 ]]; then
    echo "No commits to check in range ${RANGE}."
    exit 0
fi

echo "Checking ${#COMMITS[@]} commit(s) in range ${RANGE} for cohort-attribution trailers..."

fail=0
for sha in "${COMMITS[@]}"; do
    subject=$(git log -1 --format='%s' "$sha")
    body=$(git log -1 --format='%B' "$sha")

    # Files touched by this commit
    mapfile -t files < <(git show --name-only --format='' "$sha")

    cohort_a_hit=0
    cohort_b1_hit=0
    cohort_b2_hit=0

    for f in "${files[@]}"; do
        case "$f" in
            src/elspeth/web/shareable_reviews/*) cohort_a_hit=1 ;;
            src/elspeth/web/audit_readiness/*)   cohort_b2_hit=1 ;;
        esac
    done

    # Cohort (b1) — content-scoped on sessions/{routes,service}.py
    sessions_touched=0
    for f in "${files[@]}"; do
        case "$f" in
            src/elspeth/web/sessions/routes.py | \
            src/elspeth/web/sessions/service.py) sessions_touched=1 ;;
        esac
    done

    if [[ $sessions_touched -eq 1 ]]; then
        if git show "$sha" -- src/elspeth/web/sessions/routes.py \
                              src/elspeth/web/sessions/service.py \
           | grep -qE '^\+.*(interpretation_opt_out_total|record_interpretation_opt_out|auto_interpreted_opt_out)'; then
            cohort_b1_hit=1
        fi
    fi

    check_cohort() {
        local cohort="$1"
        local trailer_token="$2"
        if ! echo "$body" | grep -qE "^telemetry-backfill: ${trailer_token}\$"; then
            echo ""
            echo "✗ Commit ${sha:0:12} '${subject}' touches cohort ${cohort} but lacks the required trailer: telemetry-backfill: ${trailer_token}"
            fail=$(( fail + 1 ))
        fi
    }

    (( cohort_a_hit ))  && check_cohort "(a)  shareable_reviews" "shareable-reviews"
    (( cohort_b1_hit )) && check_cohort "(b1) sessions opt-out"  "interpretation-opt-out"
    (( cohort_b2_hit )) && check_cohort "(b2) audit_readiness"   "audit-readiness"
done

if (( fail > 0 )); then
    cat >&2 <<EOM

CI ENFORCEMENT FAILURE — ${fail} commit-trailer violation(s) detected.

Every commit touching a telemetry-backfill cohort directory MUST include a
stable telemetry-backfill trailer on its own line in the body:

  telemetry-backfill: shareable-reviews
  telemetry-backfill: interpretation-opt-out
  telemetry-backfill: audit-readiness

This makes \`git blame\` + \`git log -1\` reveal the cohort attribution.

See:
  docs/composer/ux-redesign-2026-05/20-phase-8-polish-and-telemetry.md
  §"Cohort attribution via commit trailers (A4 — load-bearing)"

To fix:
  1. Identify the violating commit(s) above.
  2. Either rewrite history to add the trailer (\`git rebase -i\`),
     OR open an allowlist entry under
     config/cicd/enforce_telemetry_backfill_trailer/ if the violation
     is a legitimate exception (see that directory's README for schema).

EOM
    exit 1
fi

echo "✓ All ${#COMMITS[@]} commit(s) in range have correct cohort-attribution trailers (or touch no cohort)."
