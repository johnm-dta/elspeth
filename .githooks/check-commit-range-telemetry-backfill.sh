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

ALLOWLIST_DIR="config/cicd/enforce_telemetry_backfill_trailer"
declare -A ALLOWLISTED_COHORTS=()

load_allowlist() {
    if [[ ! -d "$ALLOWLIST_DIR" ]]; then
        return 0
    fi

    local tmp
    tmp="$(mktemp)"
python3 - "$ALLOWLIST_DIR" > "$tmp" <<'PY'
from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

allowlist_dir = Path(sys.argv[1])
today = datetime.now(UTC).date()
valid_cohorts = {"a", "b1", "b2"}


def _is_block_scalar_indicator(value: str) -> bool:
    """Return True for YAML block scalar indicators like |, |-, >+, or |2."""
    if not value or value[0] not in {"|", ">"}:
        return False

    seen_chomp = False
    seen_indent = False
    for char in value[1:]:
        if char in {"+", "-"}:
            if seen_chomp:
                return False
            seen_chomp = True
        elif char.isdigit() and char != "0":
            if seen_indent:
                return False
            seen_indent = True
        else:
            return False
    return True


def _extract_entries(path: Path) -> list[dict[str, str]]:
    """Parse the constrained allowlist YAML shape without external deps.

    CI runs this shell script directly on a GitHub runner, before the project
    virtualenv exists. Keep the parser intentionally narrow: each entry starts
    with ``- commit_sha:`` and scalar fields of interest are indented beneath it.
    Block scalars such as ``reason: |`` are ignored.
    """
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_entries = False
    block_key: str | None = None
    block_indent = 0
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal block_key, block_indent, block_lines
        if current is not None and block_key is not None:
            current[block_key] = "\n".join(line.strip() for line in block_lines if line.strip()).strip()
        block_key = None
        block_indent = 0
        block_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if block_key is not None:
            if indent > block_indent:
                block_lines.append(stripped)
                continue
            flush_block()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "entries:":
            in_entries = True
            continue
        if not in_entries:
            continue
        if stripped.startswith("- "):
            flush_block()
            if current is not None:
                entries.append(current)
            current = {}
            stripped = stripped[2:].strip()
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in {"commit_sha", "cohort", "reason", "owner", "expires"} and _is_block_scalar_indicator(value):
            block_key = key
            block_indent = indent
            block_lines = []
            continue
        if key in {"commit_sha", "cohort", "reason", "owner", "expires"}:
            current[key] = value

    flush_block()
    if current is not None:
        entries.append(current)
    return entries


for path in sorted(allowlist_dir.glob("*.yaml")):
    entries = _extract_entries(path)
    for index, entry in enumerate(entries):
        commit_sha = str(entry.get("commit_sha", ""))
        cohort = str(entry.get("cohort", ""))
        reason = str(entry.get("reason", ""))
        owner = str(entry.get("owner", ""))
        expires = entry.get("expires")

        if len(commit_sha) != 40:
            raise SystemExit(f"{path}: entries[{index}].commit_sha must be a 40-character SHA")
        if cohort not in valid_cohorts:
            raise SystemExit(f"{path}: entries[{index}].cohort must be one of a, b1, b2")
        if not reason:
            raise SystemExit(f"{path}: entries[{index}].reason is required")
        if not owner:
            raise SystemExit(f"{path}: entries[{index}].owner is required")
        if not expires:
            raise SystemExit(f"{path}: entries[{index}].expires is required")

        expires_date = datetime.strptime(str(expires), "%Y-%m-%d").date()
        if expires_date < today:
            continue

        print(f"{commit_sha} {cohort}")
PY
    while read -r commit_sha cohort; do
        [[ -n "${commit_sha:-}" ]] || continue
        ALLOWLISTED_COHORTS["${commit_sha}:${cohort}"]=1
    done < "$tmp"
    rm -f "$tmp"
}

load_allowlist

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
        local cohort_id="$2"
        local trailer_token="$3"
        if [[ -n "${ALLOWLISTED_COHORTS["${sha}:${cohort_id}"]:-}" ]]; then
            echo "✓ Commit ${sha:0:12} '${subject}' touches cohort ${cohort} and is allowlisted for telemetry-backfill: ${trailer_token}"
            return 0
        fi
        if ! echo "$body" | grep -qE "^telemetry-backfill: ${trailer_token}\$"; then
            echo ""
            echo "✗ Commit ${sha:0:12} '${subject}' touches cohort ${cohort} but lacks the required trailer: telemetry-backfill: ${trailer_token}"
            fail=$(( fail + 1 ))
        fi
    }

    (( cohort_a_hit ))  && check_cohort "(a)  shareable_reviews" "a" "shareable-reviews"
    (( cohort_b1_hit )) && check_cohort "(b1) sessions opt-out"  "b1" "interpretation-opt-out"
    (( cohort_b2_hit )) && check_cohort "(b2) audit_readiness"   "b2" "audit-readiness"
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
