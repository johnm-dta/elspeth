#!/usr/bin/env bash
# =============================================================================
# ChaosLLM Sentiment Analysis Example
#
# Starts a ChaosLLM fake LLM server with ~20% steady-state error injection,
# then runs an ELSPETH sentiment analysis pipeline against it.
#
# Usage:
#   ./examples/chaosllm_sentiment/run.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAOS_CONFIG="examples/chaosllm_sentiment/chaos_config.yaml"
PIPELINE_CONFIG="examples/chaosllm_sentiment/settings.yaml"
CHAOS_PORT=8199
CHAOS_PID=""

cleanup() {
    if [ -n "$CHAOS_PID" ] && kill -0 "$CHAOS_PID" 2>/dev/null; then
        echo ""
        echo "Stopping ChaosLLM server (PID $CHAOS_PID)..."
        kill "$CHAOS_PID" 2>/dev/null || true
        wait "$CHAOS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Clean previous run artifacts
rm -f examples/chaosllm_sentiment/runs/audit.db examples/chaosllm_sentiment/runs/audit.db-wal examples/chaosllm_sentiment/runs/audit.db-shm
rm -f examples/chaosllm_sentiment/output/results.json

echo "=== ChaosLLM Sentiment Analysis Example ==="
echo ""

# --- Start ChaosLLM ---
echo "Starting ChaosLLM server on port $CHAOS_PORT..."
.venv/bin/chaosllm serve --config "$CHAOS_CONFIG" --port "$CHAOS_PORT" --workers 1 &
CHAOS_PID=$!

# Wait for server to be ready
echo "Waiting for ChaosLLM to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$CHAOS_PORT/health" > /dev/null 2>&1; then
        echo "ChaosLLM is ready."
        echo ""
        break
    fi
    if ! kill -0 "$CHAOS_PID" 2>/dev/null; then
        echo "ERROR: ChaosLLM failed to start."
        exit 1
    fi
    sleep 0.5
done

# Verify it's actually running
if ! curl -sf "http://127.0.0.1:$CHAOS_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: ChaosLLM not responding after 15 seconds."
    exit 1
fi

# --- Run Pipeline ---
echo "Running ELSPETH pipeline against ChaosLLM..."
echo ""
.venv/bin/elspeth run --settings "$PIPELINE_CONFIG" --execute

echo ""
echo "=== Pipeline Complete ==="
echo ""

# Show output
if [ -f examples/chaosllm_sentiment/output/results.json ]; then
    echo "Output (examples/chaosllm_sentiment/output/results.json):"
    head -5 examples/chaosllm_sentiment/output/results.json
    echo "..."
    LINES=$(wc -l < examples/chaosllm_sentiment/output/results.json)
    echo "($LINES rows written)"
fi

# Show ChaosLLM stats
echo ""
echo "ChaosLLM server stats:"
curl -s "http://127.0.0.1:$CHAOS_PORT/admin/stats" | python3 -m json.tool 2>/dev/null || true

echo ""
echo "Done. Audit trail: examples/chaosllm_sentiment/runs/audit.db"
