#!/usr/bin/env bash
# Sign Cluster A — 10 R6 allowlist entries with corrected, site-specific rationales.
# Generated from notes/cluster-a-rationales-draft-2026-06-02.md (verdicts ACCEPT-ready).
#
# OPERATOR-ONLY: justify SIGNS audit metadata and needs ELSPETH_JUDGE_METADATA_HMAC_KEY.
# That key MUST NOT live in any agent environment — set it yourself before running:
#     ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> bash notes/sign_cluster_a.sh
#
# Each entry is re-judged by the agent transport WITH read-only investigation tools
# (--judge-transport agent --judge-tools readonly); a BLOCKED verdict means that entry's
# rationale still needs work and is NOT written — the script reports it and continues.
#
# NOTE: entry #9 (rate_limit/limiter.py try_acquire) is STALE and handled separately — see
# the deletion command in notes/cluster-a-rationales-draft-2026-06-02.md (do NOT justify it).

set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

if [ -z "${ELSPETH_JUDGE_METADATA_HMAC_KEY:-}" ]; then
  echo "ERROR: set ELSPETH_JUDGE_METADATA_HMAC_KEY (operator-only) before running." >&2
  exit 1
fi
OWNER="${ELSPETH_JUSTIFY_OWNER:-$USER}"
LINTS="env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli"
ok=0; fail=0

# --- 1. mcp/server.py:R6:_find_audit_databases:fp=badb2efd50be18f6 ---
read -r -d '' RATIONALE_1 <<'RATIONALE_EOF' || true
`_find_audit_databases` (mcp/server.py:717–720) stats each discovered `.db` file to sort candidates by recency. `OSError` on `stat()` is an expected filesystem boundary condition (file deleted between `rglob` iteration and stat, or a permission issue on a file we don't own). The catch is not a silent drop: the file is still appended to the results list with `mtime=0` so it appears last in the recency-sorted candidate list. The audit trail is not involved — this helper only populates the interactive database-picker menu in the MCP CLI; any sort-order error affects only UI convenience, not pipeline data or Landscape integrity.
RATIONALE_EOF
echo ">>> signing 1/10: _find_audit_databases (fp=badb2efd50be...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "mcp/server.py" --rule R6 --symbol "_find_audit_databases" --fingerprint "badb2efd50be18f6" \
    --rationale "$RATIONALE_1" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: _find_audit_databases fp=badb2efd50be"; fail=$((fail+1)); fi

# --- 2. mcp/server.py:R6:_prompt_for_database:fp=250dc301eb3eda7c ---
read -r -d '' RATIONALE_2 <<'RATIONALE_EOF' || true
`_prompt_for_database` (mcp/server.py:747–751) attempts to display each candidate database as a relative path for readability. `Path.relative_to()` raises `ValueError` when the candidate is on a different mount point or drive root than the search directory. The `except ValueError` catch falls back to the absolute path string — a display-only difference with no effect on which path is returned to the caller or on any audit record. This is path-display formatting in an interactive CLI menu; the exception is an expected API signal from `pathlib`, not a masked error.
RATIONALE_EOF
echo ">>> signing 2/10: _prompt_for_database (fp=250dc301eb3e...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "mcp/server.py" --rule R6 --symbol "_prompt_for_database" --fingerprint "250dc301eb3eda7c" \
    --rationale "$RATIONALE_2" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: _prompt_for_database fp=250dc301eb3e"; fail=$((fail+1)); fi

# --- 3. mcp/server.py:R6:_prompt_for_database:fp=dbbeb81e161a9a1d ---
read -r -d '' RATIONALE_3 <<'RATIONALE_EOF' || true
`_prompt_for_database` (mcp/server.py:760–764) reads interactive user input from stdin. `EOFError` (Ctrl-D / stdin closed) and `KeyboardInterrupt` (Ctrl-C) are the canonical Python signals for "user wishes to abort an interactive prompt". Both are caught narrowly; the handler writes a cancellation message to stderr and returns `None` — the documented return value meaning "user cancelled, do not open any database". This is not a silent swallow: the absence of a selected database propagates to the caller (`main()`), which exits cleanly. No audit trail is involved; the function only selects a database for the MCP CLI session.
RATIONALE_EOF
echo ">>> signing 3/10: _prompt_for_database (fp=dbbeb81e161a...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "mcp/server.py" --rule R6 --symbol "_prompt_for_database" --fingerprint "dbbeb81e161a9a1d" \
    --rationale "$RATIONALE_3" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: _prompt_for_database fp=dbbeb81e161a"; fail=$((fail+1)); fi

# --- 4. mcp/server.py:R6:_prompt_for_database:fp=b20406c8a83aad1b ---
read -r -d '' RATIONALE_4 <<'RATIONALE_EOF' || true
`_prompt_for_database` (mcp/server.py:769–775) converts user input to an integer index. `int(choice)` raises `ValueError` on non-numeric input (e.g. "abc", empty after stripping). The catch is a loop-continue: the user is prompted again via the enclosing `while True`. This is interactive input validation — `ValueError` from `int()` is the expected Python signal for "unparseable string", not a masked error condition. The loop has no exit other than a valid selection or user cancellation (entries 2/3 above). No audit data is involved.
RATIONALE_EOF
echo ">>> signing 4/10: _prompt_for_database (fp=b20406c8a83a...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "mcp/server.py" --rule R6 --symbol "_prompt_for_database" --fingerprint "b20406c8a83aad1b" \
    --rationale "$RATIONALE_4" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: _prompt_for_database fp=b20406c8a83a"; fail=$((fail+1)); fi

# --- 5. telemetry/manager.py:R6:TelemetryManager.handle_event:fp=0a1bb9f3fb968fdf ---
read -r -d '' RATIONALE_5 <<'RATIONALE_EOF' || true
`TelemetryManager.handle_event` (telemetry/manager.py:360–363) operates under `BackpressureMode.DROP`. `queue.Full` from `put_nowait()` is the expected backpressure signal on a bounded queue at capacity — not an error, but the normal control-flow trigger for the DROP overflow policy. The catch dispatches to `_drop_oldest_and_enqueue_newest()` (line 374), which evicts the oldest item, increments `_events_dropped` for accounting, and enqueues the incoming event. Per ELSPETH's audit>telemetry>logger primacy order, telemetry is explicitly best-effort; `queue.Full` is the designed control signal for bounded-queue overflow handling at this layer. Dropped events are counted (not silently lost) via `_events_dropped` protected by `_dropped_lock`.
RATIONALE_EOF
echo ">>> signing 5/10: TelemetryManager.handle_event (fp=0a1bb9f3fb96...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "telemetry/manager.py" --rule R6 --symbol "TelemetryManager.handle_event" --fingerprint "0a1bb9f3fb968fdf" \
    --rationale "$RATIONALE_5" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: TelemetryManager.handle_event fp=0a1bb9f3fb96"; fail=$((fail+1)); fi

# --- 6. telemetry/manager.py:R6:TelemetryManager.handle_event:fp=64535719a1fd4215 ---
read -r -d '' RATIONALE_6 <<'RATIONALE_EOF' || true
`TelemetryManager.handle_event` (telemetry/manager.py:366–372) under `BackpressureMode.BLOCK` calls `queue.put(event, timeout=30.0)`. A 30-second timeout indicates the export thread is likely dead or severely stuck; raising into pipeline code would crash the pipeline over a telemetry failure, which the audit>telemetry>logger primacy order explicitly forbids. The `except queue.Full` catch records the drop via `_events_dropped` (thread-safe, under `_dropped_lock`) and emits a `logger.error` reporting the telemetry subsystem's own failure — a sanctioned logger use per the primacy policy. The event is counted as dropped, not silently discarded. No pipeline row data and no audit trail writes are affected; this is the telemetry layer protecting pipeline continuity over telemetry completeness.
RATIONALE_EOF
echo ">>> signing 6/10: TelemetryManager.handle_event (fp=64535719a1fd...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "telemetry/manager.py" --rule R6 --symbol "TelemetryManager.handle_event" --fingerprint "64535719a1fd4215" \
    --rationale "$RATIONALE_6" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: TelemetryManager.handle_event fp=64535719a1fd"; fail=$((fail+1)); fi

# --- 7. telemetry/manager.py:R6:TelemetryManager.close:fp=79190bde3fa222f5 ---
read -r -d '' RATIONALE_7 <<'RATIONALE_EOF' || true
`TelemetryManager.close` (telemetry/manager.py:582) must guarantee sentinel insertion to prevent the export thread from hanging. `queue.Full` from the bounded `put(None, timeout=0.1)` is the expected backpressure signal during shutdown — the queue may still hold unprocessed events. The catch implements the sentinel-guarantee invariant: drain one item with `get_nowait()` + `task_done()` to account for queue unfinished-task bookkeeping, then retry within the bounded loop (`maxsize + 10` attempts). This is concurrency-primitive control flow at a shutdown path, not a silent swallow: if all drain attempts are exhausted without sending the sentinel, `logger.error` is emitted at line 597 (telemetry-subsystem failure, sanctioned logger use) and the thread-join timeout at line 601–603 surfaces the hang to the operator. Raising `queue.Full` inside `close()` would defeat the critical sentinel-guarantee invariant documented in the docstring.
RATIONALE_EOF
echo ">>> signing 7/10: TelemetryManager.close (fp=79190bde3fa2...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "telemetry/manager.py" --rule R6 --symbol "TelemetryManager.close" --fingerprint "79190bde3fa222f5" \
    --rationale "$RATIONALE_7" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: TelemetryManager.close fp=79190bde3fa2"; fail=$((fail+1)); fi

# --- 8. telemetry/manager.py:R6:TelemetryManager.close:fp=c87058caa78a1776 ---
read -r -d '' RATIONALE_8 <<'RATIONALE_EOF' || true
`TelemetryManager.close` (telemetry/manager.py:592) uses `get_nowait()` to drain one item from a full queue before retrying sentinel insertion. `queue.Empty` is a benign concurrency race: the queue was drained by the export thread between the `put_nowait`-that-raised-Full and this compensating `get_nowait`. The correct response is to retry `put(None, ...)` on the next outer-loop iteration — which is exactly what `pass` achieves. Raising here would abort the sentinel-guarantee protocol unnecessarily; the outer loop handles the ultimate failure case (sentinel not sent after maxsize+10 attempts) via `logger.error` at line 597. This is concurrency-primitive race handling in a shutdown path, not a silent swallow: the race scenario is documented in the comment ("Queue became empty between put and get"), and the `sentinel_sent` flag tracks whether the protocol succeeded.
RATIONALE_EOF
echo ">>> signing 8/10: TelemetryManager.close (fp=c87058caa78a...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "telemetry/manager.py" --rule R6 --symbol "TelemetryManager.close" --fingerprint "c87058caa78a1776" \
    --rationale "$RATIONALE_8" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: TelemetryManager.close fp=c87058caa78a"; fail=$((fail+1)); fi

# --- 9. plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=188d682d00e17ea1 ---
read -r -d '' RATIONALE_9 <<'RATIONALE_EOF' || true
`BatchTransformMixin._release_loop` (batching/mixin.py:350–352) polls `wait_for_next_release(timeout=1.0)` to emit FIFO-ordered batch results. `TimeoutError` is the documented sentinel from that call meaning "no result ready within the poll window, try again" — a normal condition during low load or pipeline idle periods. `continue` is the correct disposition: the loop must keep running until either a result becomes available or `ShutdownError` is raised (the only clean-exit path, handled separately at line 354). Re-raising `TimeoutError` or returning an error result would incorrectly terminate the release thread. `ShutdownError` and unhandled `Exception` are both handled by distinct branches in the same try/except block (lines 354 and 361), confirming the `TimeoutError` catch is not a catch-all swallow.
RATIONALE_EOF
echo ">>> signing 9/10: BatchTransformMixin._release_loop (fp=188d682d00e1...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "plugins/infrastructure/batching/mixin.py" --rule R6 --symbol "BatchTransformMixin._release_loop" --fingerprint "188d682d00e17ea1" \
    --rationale "$RATIONALE_9" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: BatchTransformMixin._release_loop fp=188d682d00e1"; fail=$((fail+1)); fi

# --- 10. plugins/infrastructure/batching/mixin.py:R6:BatchTransformMixin._release_loop:fp=24666a1e729638d7 ---
read -r -d '' RATIONALE_10 <<'RATIONALE_EOF' || true
`BatchTransformMixin._release_loop` (batching/mixin.py:354–359) is a dedicated release thread that must keep running until all completed results have been drained. `ShutdownError` is ELSPETH's typed signal that `batch_buffer.shutdown()` has been called — the documented and only clean exit path, fired by `shutdown_batch_processing()` after waiting for all workers to complete. `break` is the correct disposition: it exits the `while True` loop cleanly. Re-raising `ShutdownError` would propagate it out of the thread and cause the join to observe an unexpected exception instead of a clean exit. This is not a silent swallow — `ShutdownError` is an intentional, narrowly-typed control-flow signal (an ELSPETH-defined exception, defined in `plugins/infrastructure/batching/row_reorder_buffer.py`) used as a wakeup mechanism, not an error condition. `TimeoutError` and unhandled `Exception` are handled by distinct branches in the same try/except block (lines 350 and 361).
RATIONALE_EOF
echo ">>> signing 10/10: BatchTransformMixin._release_loop (fp=24666a1e7296...)"
if $LINTS justify \
    --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
    --file-path "plugins/infrastructure/batching/mixin.py" --rule R6 --symbol "BatchTransformMixin._release_loop" --fingerprint "24666a1e729638d7" \
    --rationale "$RATIONALE_10" --owner "$OWNER" \
    --judge-transport agent --judge-tools readonly; then ok=$((ok+1)); else echo "  !! BLOCKED/FAILED: BatchTransformMixin._release_loop fp=24666a1e7296"; fail=$((fail+1)); fi

echo
echo "Cluster A signing done: $ok signed, $fail need attention (of 10)."
echo "Remember: entry #9 (try_acquire) is a separate DELETION, not a justify."
