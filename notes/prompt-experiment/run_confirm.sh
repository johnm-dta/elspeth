#!/bin/bash
# Confirmatory FULL battery (no compose-only) on the extremes V0 and B.
# Ground-truth under-flag + over-flag + execution-validation (H2) + graduation.
set -u
ROOT=/home/john/elspeth
BASELINE=/tmp/pipeline_composer.BASELINE.md
CORE="$ROOT/src/elspeth/web/composer/skills/pipeline_composer.md"
FE="$ROOT/src/elspeth/web/frontend"
N="${N:-8}"

restore() {
  echo "=== RESTORE baseline + restart ==="
  cp "$BASELINE" "$CORE"; sudo systemctl restart elspeth-web.service; sleep 10
}
trap restore EXIT

run_full() {
  local name="$1" file="$2"
  echo "=== CONFIRM $name: swap + restart ==="
  cp "$file" "$CORE"; sudo systemctl restart elspeth-web.service; sleep 12
  for i in $(seq 1 12); do curl -sf -o /dev/null https://elspeth.foundryside.dev/ && break; sleep 3; done
  echo "=== CONFIRM $name: FULL battery N=$N ==="
  ( cd "$FE" && \
    STAGING_BASE_URL=https://elspeth.foundryside.dev \
    PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
    HARNESS_BATCH_ID="confirm-$name" HARNESS_BATCH_SIZE="$N" HARNESS_COOLDOWN_MS=4000 \
    npx playwright test --config=playwright.experiment.config.ts tutorial-reliability > "/tmp/confirm-$name.log" 2>&1 )
  echo "=== CONFIRM $name: done (exit $?) ==="
}

run_full v0 "$BASELINE"
run_full B  "$ROOT/notes/prompt-experiment/variant-B-restructured.md"
echo "=== CONFIRM COMPLETE ==="
