# multi_worker_showcase

A 4-worker swarm spectacle demonstrating `elspeth join` at scale — two named
CSV sources fan in ~200 rows, and four cooperating OS processes (1 epoch-fenced
leader + 3 claim-only followers) divide the work over a single RUNNING run.

**This is a demo to watch, not a proof.** For the rigorous self-verifying
check with a PASS/FAIL assertion, see [`examples/multi_worker/`](../multi_worker/).

---

## What This Shows

ELSPETH 0.6.0 ADR-030 ("One-Host WAL Pack") lets independent OS processes share
a single in-flight run over one WAL SQLite audit DB. One process is the epoch-
fenced **leader** (ingests rows, owns barriers, writes sinks, finalises the run);
the others are **followers** (claim READY token work items from the queue, run
the transform, mark dispositions — never write sinks). All coordination state
lives in the DB; followers discover the live `run_id` by reading the `runs` table
and are admitted if they present an identical `config_hash` (same `settings.yaml`).

### Pipeline DAG

```
feed_a.csv (100 rows) ─┐
                       ├─> [queue: source_out] ─> (llm: sentiment) ─> output/results.json
feed_b.csv (100 rows) ─┘                                            └─> output/quarantined.json
```

The explicit `queues: {source_out: {}}` node makes the two-producer fan-in
legal — without it the DAG validator rejects the pipeline at startup.

---

## Running

```bash
./examples/multi_worker_showcase/run.sh           # leader + 3 followers (4-way)
WORKERS=1 ./examples/multi_worker_showcase/run.sh  # smaller swarm for quick dev
```

The script:
1. Starts a ChaosLLM mock server (`--workers 1`) on port 8199.
2. Backgrounds the **leader** with `elspeth run --settings ... --execute`.
3. Polls the audit DB (read-only) until the run is `RUNNING` with ≥1 claimed
   work item, then launches `WORKERS` **followers** with
   `elspeth join "$RUN_ID" --settings "$PIPELINE_CONFIG"` (no `--execute` —
   `join` executes unconditionally; `--execute` is only a flag on `elspeth run`).
4. Tails live `token_work_items` status counts every ~2s while the leader runs.
5. Reaps all PIDs and renders an ASCII stats card.

Expected output: exit 0 + a stats card showing workers spawned ≈ 4, total rows
≈ 200 (minus any quarantined by fault injection), rows/sec, and per-worker
attribution.

---

## Join-Window Timing

The follower can only attach while the run is `RUNNING`. The script polls for
`RUNNING ∧ ≥1 leased item` before joining, and the 200-row input plus ChaosLLM
latency (`base_ms: 30`, `jitter_ms: 20`, plus `slow_response_pct: 1.0`) keeps
the window wide enough that followers consistently join. If the leader drains
before any follower attaches (rare on typical hardware), the stats card still
renders but may show fewer than 4 workers — this is the intrinsic fast-drain
race described in ADR-030 and is non-fatal here (no assertion).

---

## Exit-Code Semantics for `elspeth join`

| Code | Meaning |
|------|---------|
| `0`  | Clean departure — follower worked until the run ended normally |
| `1`  | JoinRefusedError — admission refused (config-hash mismatch, no live leader, or run not RUNNING) |
| `2`  | FollowerSeatDeadError — leader died mid-drain; run `elspeth resume <run_id>` |
| `3`  | SIGINT / RunWorkerEvictedError |
| `4`  | Framework / Tier-1 error |

*Note: exit 2 (FollowerSeatDeadError) is present in the live CLI but missing
from the 0.6.0 design spec's exit-code list — a spec erratum, intentionally
retained here.*

The `run.sh` cleanup trap kills any still-running followers and the ChaosLLM
server on exit, so non-zero follower exits are logged but do not fail the script.

---

## Output

- `output/results.json` — JSONL of successfully processed rows (sentiment analysis)
- `output/quarantined.json` — JSONL of rows that exhausted retries

Both files plus the audit DB are git-ignored (`examples/**/output/*` +
`examples/**/runs/*`).

---

## Key Concepts

- **ADR-030 One-Host WAL Pack** — epoch-fenced leader + N−1 claim-only followers
  sharing one WAL SQLite audit DB over a single run
- **`elspeth join`** — follower admission: verifies `config_hash` match,
  requires `status='running'` and live leader heartbeat
- **Multi-source fan-in via named queue** — explicit `queues:` node is required
  for two sources publishing to the same connection name
- **Demonstrative, not self-verifying** — for a rigorous `✓ PASS`/`✗ FAIL`
  check, use `examples/multi_worker/`

---

## CI / Dogfood Note

`multi_worker_showcase` is the heaviest example (~200 rows × 4 workers).
Do not gate dogfood completion on this example. Use `examples/multi_worker/`
(leader + 1 follower, 60 rows) for bounded smoke testing.
