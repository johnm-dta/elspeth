#!/usr/bin/env bash
# =============================================================================
# multi_worker_showcase — elspeth join: 4-worker swarm spectacle (demo only)
#
# Backgrounds a LEADER (elspeth run), polls the audit DB read-only until the
# run is RUNNING with >=1 claimed work item, then launches WORKERS (default 3)
# FOLLOWERS via `elspeth join <run_id>`. After all processes exit it renders an
# ASCII stats card (workers spawned, total rows, rows/sec, succeeded/quarantined).
# ADR-030 "One-Host WAL Pack".
#
# This is a DEMONSTRATIVE example — there is no PASS/FAIL assertion.
# For the rigorous correctness proof see examples/multi_worker/.
#
# Usage:   ./examples/multi_worker_showcase/run.sh           # leader + 3 followers (4-way)
#          WORKERS=1 ./examples/multi_worker_showcase/run.sh  # smaller swarm for quick dev
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

CHAOS_CONFIG="examples/multi_worker_showcase/chaos_config.yaml"
PIPELINE_CONFIG="examples/multi_worker_showcase/settings.yaml"
DB="examples/multi_worker_showcase/runs/audit.db"
CHAOS_PORT=8199
CHAOS_PID=""
LEADER_PID=""
FOLLOWER_PIDS=()
WORKERS="${WORKERS:-3}"

START=$(date +%s)

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
rm -f examples/multi_worker_showcase/runs/audit.db \
      examples/multi_worker_showcase/runs/audit.db-wal \
      examples/multi_worker_showcase/runs/audit.db-shm
rm -f examples/multi_worker_showcase/output/results.json \
      examples/multi_worker_showcase/output/quarantined.json
mkdir -p examples/multi_worker_showcase/runs examples/multi_worker_showcase/output

echo "=== multi_worker_showcase — 4-worker swarm (~200 rows, demonstrative) ==="
echo "    leader + $WORKERS follower(s) = $((WORKERS+1))-way pack"
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
    # Guard: leader may have already finished (degenerate fast-drain race)
    if ! kill -0 "$LEADER_PID" 2>/dev/null; then
        echo "WARNING: leader exited before followers could join (fast-drain race)." >&2
        break
    fi
    sleep 0.5
done
if [ -z "$RUN_ID" ]; then
    echo "WARNING: leader never reached RUNNING within poll window — stats card may show 0 rows." >&2
    # Do not exit 1 — this is demonstrative-only, no assertion
fi

# --- Launch FOLLOWERS (same --settings => identical config_hash; NO --execute) ---
# elspeth join executes unconditionally — there is no --execute flag on join.
if [ -n "$RUN_ID" ]; then
    for i in $(seq 1 "$WORKERS"); do
        echo "Launching follower $i: elspeth join $RUN_ID ..."
        .venv/bin/elspeth join "$RUN_ID" --settings "$PIPELINE_CONFIG" &
        FOLLOWER_PIDS+=("$!")
    done
fi

# --- Optionally tail combined progress every ~2s while leader is alive ---
echo ""
echo "--- Live progress (token_work_items status counts) ---"
while kill -0 "$LEADER_PID" 2>/dev/null && [ -n "$RUN_ID" ]; do
    COUNTS="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT status, COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' GROUP BY status;" 2>/dev/null || true)"
    printf "\r  %s" "$(echo "$COUNTS" | tr '\n' '  ')"
    sleep 2
done
echo ""
echo "--- Leader finished ---"
echo ""

# --- Reap leader + followers ---
wait "$LEADER_PID"; LEADER_EXIT=$?
echo "Leader exited ($LEADER_EXIT)."
if [ ${#FOLLOWER_PIDS[@]} -gt 0 ]; then
    for pid in "${FOLLOWER_PIDS[@]}"; do
        wait "$pid" || echo "Follower $pid exited non-zero ($?) (see exit-code semantics in README)."
    done
fi

END=$(date +%s)
ELAPSED=$((END - START))
[ "$ELAPSED" -lt 1 ] && ELAPSED=1

# --- ASCII stats card (read-only queries) ---
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        multi_worker_showcase — run complete          ║"
echo "╚══════════════════════════════════════════════════════╝"

if [ -n "$RUN_ID" ]; then
    WORKERS_SPAWNED="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT COUNT(*) FROM run_workers WHERE run_id='$RUN_ID';" 2>/dev/null || echo 0)"
    TOTAL_ROWS="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' AND status IN ('terminal','failed');" 2>/dev/null || echo 0)"
    SUCCEEDED="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' AND status='terminal';" 2>/dev/null || echo 0)"
    FAILED="$(sqlite3 "file:${DB}?mode=ro" \
        "PRAGMA query_only=ON; SELECT COUNT(*) FROM token_work_items WHERE run_id='$RUN_ID' AND status='failed';" 2>/dev/null || echo 0)"
    ROWS_PER_SEC=$((TOTAL_ROWS / ELAPSED))

    echo ""
    printf "  Workers spawned:   %s\n" "$WORKERS_SPAWNED"
    printf "  Total rows done:   %s\n" "$TOTAL_ROWS"
    printf "  Succeeded:         %s\n" "$SUCCEEDED"
    printf "  Quarantined:       %s\n" "$FAILED"
    printf "  Wall-clock:        %ss\n" "$ELAPSED"
    printf "  Aggregate rows/s:  %s\n" "$ROWS_PER_SEC"
    echo ""
    echo "  Per-worker attribution:"
    sqlite3 "file:${DB}?mode=ro" <<SQL
PRAGMA query_only = ON;
SELECT
  '  ' || COALESCE(w.role, 'unknown') || '  ' || COALESCE(t.lease_owner, 'null') ||
  '  completed=' || COUNT(CASE WHEN t.status IN ('terminal', 'failed') THEN 1 END)
FROM token_work_items t
LEFT JOIN run_workers w ON w.worker_id = t.lease_owner AND w.run_id = t.run_id
WHERE t.run_id = '$RUN_ID'
GROUP BY t.lease_owner, w.role
ORDER BY w.role DESC, COUNT(*) DESC;
SQL
else
    echo ""
    echo "  (run_id not captured — leader may have finished before poll window)"
fi

echo ""
echo "  (Demonstrative only — see examples/multi_worker/ for the rigorous proof.)"
echo ""

exit 0
