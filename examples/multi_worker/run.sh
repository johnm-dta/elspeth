#!/usr/bin/env bash
# =============================================================================
# multi_worker — elspeth join: independent processes share one RUNNING run
#
# Backgrounds a LEADER (elspeth run), polls the audit DB read-only until the run
# is RUNNING with >=1 claimed work item, then launches WORKERS (default 1)
# FOLLOWERS via `elspeth join <run_id>`. After all processes exit it queries
# token_work_items grouped by lease_owner and asserts >=2 distinct workers each
# completed >=1 row.  ADR-030 "One-Host WAL Pack".
#
# Usage:   ./examples/multi_worker/run.sh        # leader + 1 follower (2-way)
#          WORKERS=3 ./examples/multi_worker/run.sh   # leader + 3 followers
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAOS_CONFIG="examples/multi_worker/chaos_config.yaml"
PIPELINE_CONFIG="examples/multi_worker/settings.yaml"
DB="examples/multi_worker/runs/audit.db"
CHAOS_PORT=8199
CHAOS_PID=""
LEADER_PID=""
FOLLOWER_PIDS=()
WORKERS="${WORKERS:-1}"

cleanup() {
    # Kill any still-running followers and the leader, then the ChaosLLM server.
    # Array-length-guarded so an empty FOLLOWER_PIDS under `set -u` does not
    # expand to one empty-string element (which would run `kill -0 ""`).
    if [ ${#FOLLOWER_PIDS[@]} -gt 0 ]; then
        for pid in "${FOLLOWER_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
        done
    fi
    [ -n "$LEADER_PID" ] && kill -0 "$LEADER_PID" 2>/dev/null && kill "$LEADER_PID" 2>/dev/null || true
    if [ -n "$CHAOS_PID" ] && kill -0 "$CHAOS_PID" 2>/dev/null; then
        echo ""
        echo "Stopping ChaosLLM server (PID $CHAOS_PID)..."
        kill "$CHAOS_PID" 2>/dev/null || true
        wait "$CHAOS_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Clean previous run artifacts
rm -f examples/multi_worker/runs/audit.db examples/multi_worker/runs/audit.db-wal examples/multi_worker/runs/audit.db-shm
rm -f examples/multi_worker/output/results.json examples/multi_worker/output/quarantined.json
mkdir -p examples/multi_worker/runs examples/multi_worker/output

echo "=== multi_worker (elspeth join) — leader + $WORKERS follower(s) ==="
echo ""

# --- Start ChaosLLM (errorworks bug: must use --workers 1) ---
echo "Starting ChaosLLM server on port $CHAOS_PORT..."
.venv/bin/chaosllm serve --config "$CHAOS_CONFIG" --port "$CHAOS_PORT" --workers 1 &
CHAOS_PID=$!
echo "Waiting for ChaosLLM to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$CHAOS_PORT/health" > /dev/null 2>&1; then
        echo "ChaosLLM is ready."; echo ""; break
    fi
    if ! kill -0 "$CHAOS_PID" 2>/dev/null; then echo "ERROR: ChaosLLM failed to start."; exit 1; fi
    sleep 0.5
done
if ! curl -sf "http://127.0.0.1:$CHAOS_PORT/health" > /dev/null 2>&1; then
    echo "ERROR: ChaosLLM not responding after 15 seconds."; exit 1
fi

# --- Launch LEADER (background) ---
echo "Launching leader: elspeth run --execute ..."
.venv/bin/elspeth run --settings "$PIPELINE_CONFIG" --execute &
LEADER_PID=$!

# --- Poll audit DB (read-only) for RUNNING run with >=1 claimed work item ---
# Readiness criterion (design D4): RUNNING is not enough; require >=1 'leased'
# token_work_item so the leader has demonstrably begun processing before a
# follower attaches. Bounded retry; no artificial sleep widens the join window.
RUN_ID=""
for attempt in $(seq 1 60); do
    RUN_ID="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT run_id FROM runs WHERE status='running' LIMIT 1;" 2>/dev/null || true)"
    if [ -n "$RUN_ID" ]; then
        CLAIMED="$(sqlite3 "file:${DB}?mode=ro" \
            "PRAGMA query_only=ON; SELECT COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' AND status='leased';" 2>/dev/null || echo 0)"
        if [ "${CLAIMED:-0}" -ge 1 ]; then
            echo "Leader RUNNING (run_id=$RUN_ID) with $CLAIMED claimed item(s); joining."
            break
        fi
    fi
    # Guard: leader may have already finished (degenerate fast-drain race, design risk #1)
    if ! kill -0 "$LEADER_PID" 2>/dev/null; then
        echo "WARNING: leader exited before a follower could join (fast-drain race)." >&2
        break
    fi
    sleep 0.5
done
if [ -z "$RUN_ID" ]; then
    echo "FAIL: leader never reached RUNNING within the poll window." >&2
    exit 1
fi

# --- Launch FOLLOWERS (same --settings => identical config_hash; NO --execute) ---
for i in $(seq 1 "$WORKERS"); do
    echo "Launching follower $i: elspeth join $RUN_ID ..."
    .venv/bin/elspeth join "$RUN_ID" --settings "$PIPELINE_CONFIG" &
    FOLLOWER_PIDS+=("$!")
done

# --- Reap leader + followers ---
wait "$LEADER_PID"; LEADER_EXIT=$?
echo "Leader exited ($LEADER_EXIT)."
for pid in "${FOLLOWER_PIDS[@]}"; do
    wait "$pid" || echo "Follower $pid exited non-zero ($?) (see exit-code semantics in README)."
done

# --- Attribution query (read-only): per-worker completed rows ---
# NOTE: token_work_items.lease_owner is cleared to NULL when an item transitions
# to terminal/failed (the mark_terminal / mark_failed write sets lease_owner=NULL).
# Attribution comes from scheduler_events:
#
#   mark_pending_sink (LEASED→PENDING_SINK): emitted by the worker that ran the
#     transform and handed off to sink.  from_lease_owner = that worker.  This is
#     the correct per-worker attribution source in multi-worker mode because the
#     leader later drains follower PENDING_SINK rows and terminates them under the
#     leader's lease_owner — so mark_pending_sink_terminal always shows the leader
#     even for rows the follower actually processed.
#
#   mark_failed (LEASED→FAILED): emitted by the worker that had the lease when
#     the item failed. from_lease_owner = that worker.
echo ""
echo "Per-worker attribution (scheduler_events grouped by from_lease_owner):"
sqlite3 "file:${DB}?mode=ro" <<SQL
PRAGMA query_only = ON;
SELECT
  se.from_lease_owner,
  w.role,
  COUNT(*) AS completed_rows
FROM scheduler_events se
LEFT JOIN run_workers w ON w.worker_id = se.from_lease_owner AND w.run_id = se.run_id
WHERE se.run_id = '$RUN_ID'
  AND se.event_type IN ('mark_pending_sink', 'mark_failed')
  AND se.from_lease_owner IS NOT NULL
GROUP BY se.from_lease_owner, w.role
ORDER BY w.role DESC, completed_rows DESC;
SQL

# --- Assertion: >=2 distinct workers each completed >=1 row ---
WORKER_COUNT="$(sqlite3 "file:${DB}?mode=ro" \
  "PRAGMA query_only=ON; SELECT COUNT(*) FROM (SELECT from_lease_owner FROM scheduler_events WHERE run_id='$RUN_ID' AND event_type IN ('mark_pending_sink','mark_failed') AND from_lease_owner IS NOT NULL GROUP BY from_lease_owner HAVING COUNT(*) >= 1);" 2>/dev/null || echo 0)"
TOTAL_ROWS="$(sqlite3 "file:${DB}?mode=ro" \
  "PRAGMA query_only=ON; SELECT COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' AND status IN ('terminal','failed');" 2>/dev/null || echo 0)"

echo ""
if [ "${WORKER_COUNT:-0}" -ge 2 ]; then
    echo "✓ PASS: leader + $((WORKER_COUNT-1)) follower(s) shared $TOTAL_ROWS rows across $WORKER_COUNT workers"
    exit 0
else
    echo "✗ FAIL: only ${WORKER_COUNT:-0} worker(s) completed rows; expected >=2 (join-window race? raise WORKERS or row count)" >&2
    exit 1
fi
