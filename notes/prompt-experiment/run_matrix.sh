#!/bin/bash
# Prompt A/B experiment matrix runner (2026-06-07). Compose-only, N per variant,
# model pinned to gpt-5.4-mini. Swaps the CORE skill, restarts, runs the battery.
# Restores the baseline on exit no matter what.
set -u
ROOT=/home/john/elspeth
BASELINE=/tmp/pipeline_composer.BASELINE.md
CORE="$ROOT/src/elspeth/web/composer/skills/pipeline_composer.md"
FE="$ROOT/src/elspeth/web/frontend"
N="${N:-15}"

restore() {
  echo "=== RESTORE baseline skill + restart ==="
  cp "$BASELINE" "$CORE"
  sudo systemctl restart elspeth-web.service
  sleep 10
}
trap restore EXIT

run_variant() {
  local name="$1" file="$2"
  echo "=== VARIANT $name: swap + restart ==="
  cp "$file" "$CORE"
  sudo systemctl restart elspeth-web.service
  sleep 12
  for i in $(seq 1 12); do curl -sf -o /dev/null https://elspeth.foundryside.dev/ && break; sleep 3; done
  echo "=== VARIANT $name: battery N=$N ==="
  ( cd "$FE" && \
    STAGING_BASE_URL=https://elspeth.foundryside.dev \
    PLAYWRIGHT_BACKEND_BASE_URL=https://elspeth.foundryside.dev \
    HARNESS_COMPOSE_ONLY=1 HARNESS_BATCH_ID="exp-$name" HARNESS_BATCH_SIZE="$N" HARNESS_COOLDOWN_MS=2000 \
    npx playwright test --config=playwright.experiment.config.ts tutorial-reliability > "/tmp/exp-$name.log" 2>&1 )
  echo "=== VARIANT $name: done (exit $?) ==="
}

run_variant v0 "$BASELINE"
run_variant A  "$ROOT/notes/prompt-experiment/variant-A-tightened.md"
run_variant B  "$ROOT/notes/prompt-experiment/variant-B-restructured.md"
echo "=== MATRIX COMPLETE ==="
