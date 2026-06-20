#!/usr/bin/env bash
# =============================================================================
# ChaosLLM Endurance Test
#
# Stress tests ELSPETH at scale against a noisy LLM backend:
#   10,000 rows x 2 case studies x 5 criteria = 100,000 LLM calls
#   with ~11% steady-state error injection (8% retryable, 3% quarantine).
#
# Usage:
#   ./examples/chaosllm_endurance/run.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAOS_CONFIG="examples/chaosllm_endurance/chaos_config.yaml"
PIPELINE_CONFIG="examples/chaosllm_endurance/settings.yaml"
DEFAULT_INPUT="examples/chaosllm_endurance/input.csv"
CHAOS_PORT=8199
ROWS="${CHAOSLLM_ENDURANCE_ROWS:-10000}"
SEED="${CHAOSLLM_ENDURANCE_SEED:-42}"
REGENERATE_INPUT="${CHAOSLLM_ENDURANCE_REGENERATE:-0}"
CHAOS_PID=""

if [ -n "${CHAOSLLM_ENDURANCE_INPUT_PATH:-}" ]; then
    INPUT_PATH="$CHAOSLLM_ENDURANCE_INPUT_PATH"
elif [ "$ROWS" = "10000" ]; then
    INPUT_PATH="$DEFAULT_INPUT"
else
    INPUT_PATH="examples/chaosllm_endurance/input.${ROWS}.csv"
fi
export CHAOSLLM_ENDURANCE_INPUT_PATH="$INPUT_PATH"

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
rm -f examples/chaosllm_endurance/runs/audit.db examples/chaosllm_endurance/runs/audit.db-wal examples/chaosllm_endurance/runs/audit.db-shm
rm -f examples/chaosllm_endurance/output/results.csv
rm -f examples/chaosllm_endurance/output/quarantined.json

echo "======================================================="
echo "  ChaosLLM Endurance Test"
echo "  ${ROWS} rows x 2 case studies x 5 criteria = $((ROWS * 10)) LLM calls"
echo "======================================================="
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

input_data_rows() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "0"
        return
    fi
    local line_count
    line_count=$(wc -l < "$path")
    if [ "$line_count" -le 0 ]; then
        echo "0"
        return
    fi
    echo $((line_count - 1))
}

# Generate input data if not present, explicitly requested, or mismatched. Smoke
# overrides write a separate input.<rows>.csv so they cannot poison the default
# 10k endurance input.
EXISTING_ROWS="$(input_data_rows "$INPUT_PATH")"
if [ "$EXISTING_ROWS" != "$ROWS" ]; then
    REGENERATE_INPUT="1"
fi
if [ ! -f "$INPUT_PATH" ] || [ "$REGENERATE_INPUT" = "1" ]; then
    mkdir -p "$(dirname "$INPUT_PATH")"
    echo "Generating ${ROWS} row input CSV at ${INPUT_PATH}..."
    .venv/bin/python -m scripts.generate_test_data multi \
        --rows "$ROWS" --case-studies 2 --fields-per-cs 3 \
        --output "$INPUT_PATH" --seed "$SEED"
    echo ""
fi

START_TIME=$(date +%s)

# --- Run Pipeline ---
echo "Running ELSPETH pipeline against ChaosLLM..."
echo ""
.venv/bin/elspeth run --settings "$PIPELINE_CONFIG" --execute

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "======================================================="
echo "  Endurance Test Complete (${ELAPSED}s)"
echo "======================================================="
echo ""

# Show output stats
if [ -f examples/chaosllm_endurance/output/results.csv ]; then
    RESULT_LINES=$(wc -l < examples/chaosllm_endurance/output/results.csv)
    echo "Results: $((RESULT_LINES - 1)) rows written to output/results.csv"
fi

if [ -f examples/chaosllm_endurance/output/quarantined.json ]; then
    QUARANTINE_LINES=$(wc -l < examples/chaosllm_endurance/output/quarantined.json)
    echo "Quarantined: $QUARANTINE_LINES rows written to output/quarantined.json"
else
    echo "Quarantined: 0 rows"
fi

# Show ChaosLLM stats
echo ""
echo "ChaosLLM server stats:"
curl -s "http://127.0.0.1:$CHAOS_PORT/admin/stats" | python3 -m json.tool 2>/dev/null || true

echo ""
echo "Audit trail: examples/chaosllm_endurance/runs/audit.db"
