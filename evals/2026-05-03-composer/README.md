# ELSPETH Composer Evals — 2026-05-03

This folder contains the raw evidence files for two composer evals run on the same day. Both eval reports live one level up at `docs/composer/evidence/composer-*-2026-05-03.md`; this folder is what backs the claims in those reports.

## Contents

```
2026-05-03-composer/
├── README.md                 (this file — index)
├── basic/                    Basic-mode eval evidence (LLM-driver, well-formed prompts)
│   ├── REPORT.md             working copy of the basic-mode eval (mirror of docs/composer/evidence/composer-eval-basic-2026-05-03.md)
│   ├── tickets.csv           the test CSV used across S2/S3/S3'/S4
│   ├── openapi.json          OpenAPI snapshot at run time
│   ├── catalog/              snapshot of GET /api/catalog/{sources,transforms,sinks} and per-plugin /schema
│   ├── s1a/                  Scenario 1 monolithic LLM-classify pipeline (blocked at composer-time secret_refs)
│   ├── s1b/                  (older path, prior eval — mostly empty here)
│   ├── s2/                   Scenario 2 incremental classifier (msg4 produced real output post-fingerprint-key fix)
│   ├── s3/                   Scenario 3 aggregation (3 architectural-class runtime errors)
│   ├── s3_prime/             Scenario 3-prime (re-run after f3137ae8 schema_compatibility fix — first-iteration green)
│   ├── s4/                   Scenario 4 gate routing (verified close on elspeth-71520f5e30 and elspeth-5069612f3c)
│   └── s5/                   Scenario 5 vague-Excel refusal probe
└── hardmode/                 Hard-mode eval evidence (persona-driven, 3×3 matrix)
    ├── README.md             detailed file-by-file index for the hardmode workspace
    ├── personas/             3 persona specs (Linda compliance, Sarah researcher, Marcus marketing-ops)
    ├── scenarios/            9 scenario fixtures (3 personas × 3 task classes)
    ├── results/              9 per-scenario subdirectories with full session/message/state/run records
    ├── harness.sh            bootstrap script
    ├── post_message.sh       per-turn driver
    ├── finalize_scenario.sh  after-DONE finaliser
    └── aggregate.json        cross-scenario summary table
```

## Companion documents

| Document | Purpose |
|---|---|
| `docs/composer/evidence/composer-fieldreport-2026-05-03.md` | Boss-facing field report ("the brief") |
| `docs/composer/evidence/composer-eval-hardmode-2026-05-03.md` | Technical eval report for the hard-mode (persona-driven) eval |
| `docs/composer/evidence/composer-eval-basic-2026-05-03.md` | Technical eval report for the basic-mode (LLM-driver) eval |

The brief is the document the CTO reads. Both technical reports back specific claims in the brief and reference files in this evidence tree.

## How to verify any claim

If a claim in the brief or either technical report seems too strong or unclear, the verification path is:

1. **Identify the eval class** (basic-mode if S1/S2/S3/S4/S5 reference; hard-mode if P1-T1, P2-T2, etc.)
2. **Find the scenario directory** under `basic/<sN>/` or `hardmode/results/<scenario_id>/`
3. **Read `scenario.json`** (hard-mode) or the relevant prompt files (basic-mode) for the test definition
4. **Read user-side messages** (`turn{N}.user.txt`) and **composer responses** (`msg.t{N}.resp.json` — extract with `jq -r '.message.content'`)
5. **Read engine artefacts**: `validate.json`, `final_yaml.json` (for the actual config), `run.json` (for the run summary), `diagnostics.json` (for per-row engine errors with verbatim exception text), and **`outputs/`** (per-row engine output streams — `MANIFEST.json` plus the actual JSONL/CSV files written by each sink)
6. **Read `ledger.json`** (hard-mode only) for the consolidated per-scenario summary, including the new `artifacts[]` summary (sink_node_id / size_bytes / exists_now / content_hash for each captured output)

Worked examples for the two most-likely challenges:

```bash
# "Did the composer really pick anthropic/claude-3.5-sonnet?"
jq -r '.yaml' evals/2026-05-03-composer/hardmode/results/p1_t1_happy/final_yaml.json | grep -E '^\s+model:'

# "Did every row really get HTTP 404?"
jq '.. | objects | select(.error?) | .error' evals/2026-05-03-composer/hardmode/results/p1_t1_happy/diagnostics.json | head -2

# "Did the proof-of-fix run really produce real output?"
ls -la data/outputs/q3_*csv  # files from run 023eb897-... timestamped 2026-05-03 13:28
cat data/outputs/q3_regional_compliance_categories.csv

# "What did Linda actually say across her 3 turns?"
cat evals/2026-05-03-composer/hardmode/results/p1_t1_happy/turn{1,2,3}.user.txt

# "Did INT-1002 really route to fraud_only.jsonl?" (uses backfilled outputs/)
jq 'select(.interaction_id=="INT-1002")' \
  evals/2026-05-03-composer/basic/s4/outputs/q3_fraud_security_flags.csv 2>/dev/null \
  || head -3 evals/2026-05-03-composer/basic/s4/outputs/q3_fraud_security_flags.csv

# "Which sinks did the engine actually write for hardmode P1-T1?"
jq '.artifacts[] | {sink_node_id, size_bytes, exists_now}' \
  evals/2026-05-03-composer/hardmode/results/p1_t1_happy/outputs/MANIFEST.json
```

## Backfilled vs live-captured outputs

Per-row engine outputs in this tree fall into two provenance categories:

1. **Retroactive backfill (2026-05-06)** — the original eval harness in
   2026-05-03 captured run summaries and diagnostics but not the
   per-row sink output streams. Files were copied from the staging
   deploy's `data/outputs/` directory and timestamp-correlated against
   each run's start/finish window. Each `outputs/MANIFEST.json` records
   the configured path, the actually-written path (which can differ via
   `auto_increment` collision policy), sha256, mtime, and a
   `correlation_confidence` flag. Only HIGH-confidence files were
   archived; LOW-confidence candidates are recorded in
   `skipped_low_confidence` for forensic completeness but NOT copied —
   their bytes may not be from the run we're trying to evidence.
   See `scripts/eval/backfill_2026_05_03_outputs.py`.

2. **Live capture (future evals)** — `hardmode/finalize_scenario.sh` and
   `basic/finalize_scenario.sh` now call the new
   `GET /api/runs/{rid}/outputs` (full manifest, no preview cap) and
   `GET /api/runs/{rid}/outputs/{artifact_id}/content` (per-artifact
   bytes, path-allowlist guarded) at finalize time. These are run-id
   stamped through the audit DB and do not need timestamp correlation.

**Known limitation** (`elspeth-obs-e87152484a`): the 2026-05-03 hardmode
harness ran `finalize_scenario.sh` once per session, so when an operator
re-executed within the same session after a fix (e.g., the model swap
producing run `023eb897-...` for `p1_t1_happy`), the recapture didn't
happen. `scenario_dir/run.json` for those scenarios reflects the FIRST
/execute attempt (failed v1), not the proof-of-fix v3. The README's
"Audit-trail evidence outside this folder" table below carries the v3
run-ids; recovering their per-row outputs requires a separate
rerun-mode pass against staging using those IDs.

## Provenance and what was redacted

These files were captured from a live evaluation run against the staging deploy at `https://elspeth.foundryside.dev` on 2026-05-03 (HEAD `f3137ae8`).

**Redacted before commit**:
- `jwt*.txt` files in every scenario directory — the original short-lived `dta_user` JWTs are replaced with placeholder text. Tokens were already expired (1-hour TTL) but credential-shaped strings shouldn't land in git regardless.
- `login.json`, `login2.json` — wrapped access/refresh tokens, replaced with `{"redacted": ...}` placeholders.

**Not redacted (and intentionally preserved for audit-trail integrity)**:
- Session IDs, blob IDs, run IDs — these are non-sensitive identifiers traceable through the staging audit DB.
- The CSV input data (`tickets.csv` and inline `csv_content` in scenario fixtures) — synthetic test data, no real PII.
- The composer's responses, including model identifiers it chose (`anthropic/claude-3.5-sonnet`, `openai/gpt-4o-mini`) — load-bearing evidence.
- Engine error messages and diagnostics — load-bearing evidence.

**Original ephemeral location** (where these files were written during the run, now a duplicate that can be cleared):
- `/tmp/elspeth_eval/2026-05-03/` — basic-mode workspace
- `/tmp/elspeth_eval/2026-05-03/hardmode/` — hard-mode workspace

## Filed observations referenced in the eval reports

| ID | Title | Surfaced in |
|---|---|---|
| `elspeth-obs-f3143acba2` | Composer-selected LLM model strings can 404 on OpenRouter | hard-mode (3-of-3 happy-path repro) |
| `elspeth-obs-8f82c91147` | Composer LLM convergence-timeout on multi-step builds | hard-mode (3-of-3 edge-class repro) |
| `elspeth-obs-7382fbabc4` | LLM transform never writes per-call rows to landscape `calls` table | basic-mode (S2 audit-trail audit) |

Each observation is queryable in filigree and includes reproducer paths into this evidence tree.

## Audit-trail evidence outside this folder (in the Landscape DB)

For runs that reached `/execute`, the full Tier-1 audit row lives in `/home/john/elspeth/data/runs/audit.db`. Run IDs from this eval:

| Eval | Scenario | Run ID | Status |
|---|---|---|---|
| basic | S2 (LLM classifier post-fingerprint-key fix) | `45a592e1-7a8e-416b-8650-906118b9c96d` | completed, 8/8 succeeded |
| basic | S3-prime (post-`f3137ae8` validator fix) | `a419c8a8-e07d-4618-b2c1-515b257bbb07` | completed |
| basic | S4 v1 (gate routing) | `f8b35c56-ffd0-4abc-b122-894a401cf548` | completed, 8/8 routed |
| basic | S4 v2 (gate routing patched) | `b50133e4-cf17-45bb-9e83-f04b1011b0cf` | completed, 8/8 routed |
| hardmode | P1-T1 v3 (proof-of-fix model swap) | `023eb897-3049-4ad5-a502-e9eb81a4faee` | completed, 8/8 routed |

For each run above, per-row outputs are now also archived under each
scenario's `outputs/` directory where backfill confidence was HIGH; the
load-bearing P1-T1 v3 outputs require a separate rerun-mode pass (see
`elspeth-obs-e87152484a` and the "Backfilled vs live-captured outputs"
section above).

Query the audit DB with e.g.:
```sql
SELECT run_id, status, reproducibility_grade FROM runs WHERE run_id='023eb897-3049-4ad5-a502-e9eb81a4faee';
SELECT env_var_name, source, fingerprint FROM secret_resolutions WHERE run_id='45a592e1-7a8e-416b-8650-906118b9c96d';
```

Note `data/` is not gitignored in this repo as of this writing — the audit DB and pipeline outputs would commit if you `git add data/`. Out of scope for this eval but worth raising as a separate operator concern.
