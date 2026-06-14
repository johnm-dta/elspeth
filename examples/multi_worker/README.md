# multi_worker — elspeth join: independent OS processes share one RUNNING run

This example demonstrates `elspeth join`: an epoch-fenced **leader** and one or
more claim-only **followers** — independent OS processes — cooperate on a single
RUNNING run over one WAL SQLite audit database (ADR-030 "One-Host WAL Pack").

The `run.sh` launcher backgrounds the leader, polls the audit DB read-only
until the run is RUNNING with at least one claimed work item, then attaches
WORKERS (default 1) followers via `elspeth join <run_id>`. After all processes
exit, it queries `token_work_items` grouped by `lease_owner` and asserts
**≥2 distinct workers each completed ≥1 row**, printing `✓ PASS` on success.

## Pipeline shape

```
input.csv (60 rows)
  └─> [source_out] ─> llm_0 (ChaosLLM sentiment) ─> output/results.json
                                                    ─> output/quarantined.json
```

## Leader vs follower roles

| Role | Responsibilities |
|------|----------------|
| **Leader** | Ingests source rows, manages barriers and checkpoints, writes sinks, finalises the run |
| **Follower** | Claims READY work items, runs transforms, marks dispositions — never writes sinks |

The follower discovers the `run_id` by reading the `runs` table in the shared
WAL SQLite DB. Admission is atomic: the follower computes
`stable_hash(resolve_config(settings))` and is refused unless it matches the
leader's `runs.config_hash`. This is why the leader and every follower must
pass the **same** `settings.yaml`.

## Running

```bash
# Default: leader + 1 follower (2-way pack)
./examples/multi_worker/run.sh

# Scale to 3 followers (4-way pack)
WORKERS=3 ./examples/multi_worker/run.sh
```

The follower invocation inside `run.sh` is:

```bash
.venv/bin/elspeth join "$RUN_ID" --settings "$PIPELINE_CONFIG" &
```

**There is no `--execute` flag on `elspeth join`** — join executes
unconditionally. Only `elspeth run` takes `--execute`.

## Join-window timing (design risk)

A follower can only attach while the run is `running`. The poll loop requires
RUNNING *and* ≥1 `leased` token work item before launching followers, so the
leader is demonstrably processing before any follower joins. The input is sized
to 60 rows and `chaos_config.yaml` includes `slow_response_pct: 1.0` /
`slow_response_sec: [1, 3]` so the leader cannot drain the queue before
followers attach under normal ChaosLLM latency.

If the assertion fails with "only 1 worker completed rows", the leader finished
before the follower joined (fast-drain race). Do not add sleeps — raise
`WORKERS` or the row count in `input.csv` instead, and re-run.

## Output

After the run, `run.sh` prints a per-worker attribution table from the audit
DB and the PASS/FAIL verdict:

```
Per-worker attribution (token_work_items grouped by lease_owner):
<worker_id>|leader|<N>
<worker_id>|follower|<M>

✓ PASS: leader + 1 follower(s) shared 60 rows across 2 workers
```

Success: `output/results.json` (completed rows) and `output/quarantined.json`
(rows that exhausted retries against ChaosLLM faults).

## Exit-code semantics (`elspeth join`)

| Exit | Meaning |
|------|---------|
| `0` | Clean departure — follower finished normally |
| `1` | `JoinRefusedError` — admission refused (config-hash mismatch, no live leader, or run not RUNNING) |
| `2` | `FollowerSeatDeadError` — leader died mid-drain; run `elspeth resume <run_id>` to recover |
| `3` | SIGINT / `RunWorkerEvictedError` — follower was interrupted or evicted |
| `4` | Framework / Tier-1 error |

*Note: exit 2 (`FollowerSeatDeadError`) is present in the live CLI but missing
from the 0.6.0 design spec's exit-code list — a spec erratum, intentionally
retained here.*

## Key concepts

- **ADR-030 "One-Host WAL Pack"** — epoch-fenced multi-worker coordination over
  a single WAL SQLite audit DB; no separate coordination service required.
- **`elspeth join`** — attaches a follower to a RUNNING run; runs unconditionally
  (no `--execute` flag); fails fast on config-hash mismatch or admission refusal.
- **`token_work_items.lease_owner`** — audit-true per-worker attribution; the
  read-only attribution query in `run.sh` uses `file:…?mode=ro` +
  `PRAGMA query_only=ON` so the verification never contends with live worker
  writes.
- **ChaosLLM** (`chaosllm_sentiment` shape) — keyless mock LLM with configurable
  latency and fault injection; `--workers 1` required (errorworks constraint).
