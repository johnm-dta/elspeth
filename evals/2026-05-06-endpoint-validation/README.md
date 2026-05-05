# Per-Row Endpoint Validation — 2026-05-06

Lightweight end-to-end validation of the new audit-evidence retrieval
path added in branch `feat/eval-per-row-output-archival` (closing
`elspeth-77d2641032`):

1. `GET /api/runs/{rid}/outputs` — full manifest of every sink-write
2. `GET /api/runs/{rid}/outputs/{aid}/content` — per-artifact bytes
3. `evals/2026-05-03-composer/{hardmode,basic}/finalize_scenario.sh` —
   harness wiring that pulls (1) + (2)

## Why this is here, not a new composer-driven eval

The original 2026-05-03 hardmode harness was driven by the composer LLM
(`gpt-5` at the time). The current staging deploy runs `gpt-5-mini`,
which on this machine, this date, was unable to converge on Linda's
opening prompt within the 300s composer wall-clock budget — see
`evals/2026-05-03-composer/hardmode/results/p1_t1_happy_2026-05-06_aborted_composer_timeout/`
for the captured timeout evidence (replicates `composer-obs-8f82c91147`).

To validate the **per-row endpoint and harness** independently of
composer convergence, this directory captures a CLI-driven engine run
against a known-good pipeline (`examples/boolean_routing/`), pointed at
the live staging audit DB at `/home/john/elspeth/data/runs/audit.db`,
plus the manifest produced by the new loader.

## Files

| File | Source | Purpose |
|---|---|---|
| `MANIFEST.json` | `elspeth.web.execution.outputs.load_run_outputs_from_db` reading audit.db | Proves the loader reads real audit rows, builds correct `RunOutputArtifact[]` with `path_or_uri`, `content_hash`, `size_bytes`, `exists_now=True` |
| `approved.csv` | gate routing → approved sink | The actual rows the engine wrote (5 rows, 188 bytes) |
| `rejected.csv` | gate routing → rejected sink | The actual rows the engine wrote (5 rows, 176 bytes) |

## What was verified end-to-end

```bash
# 1. New /outputs route registered (HTTP-level smoke check)
curl -H "Authorization: Bearer $JWT" \
  https://elspeth.foundryside.dev/api/runs/00000000-0000-0000-0000-000000000000/outputs
# → 404 {"detail":"Run not found"}     # from new route's run-ownership gate
#                                       # (NOT 404 missing route — the difference matters)

# 2. CLI run produces audit rows + on-disk artifacts
elspeth run --settings /tmp/validation_2026-05-06/settings.yaml --execute
# → ✓ Run COMPLETED: 10 rows | →10 routed (approved:5, rejected:5)

# 3. Loader returns the manifest with content_hash + exists_now
python -c "from elspeth.web.execution.outputs import load_run_outputs_from_db; ..."
# → 2 artifacts, both file:// URIs, exists_now=True, sha256 hashes populated

# 4. Harness jq pattern parses the manifest correctly
jq -r '.artifacts[]? | "\(.artifact_id)\t\(.path_or_uri | sub("^file://"; "") | split("/") | .[-1])"' MANIFEST.json
# → tab-separated (id, basename) ready for finalize_scenario.sh's `while read aid name` loop
```

## What was NOT verified (and why it doesn't gate this work)

- **HTTP-layer fetch with `Bearer` auth against this exact run**: the CLI
  run was not session-bound (`/api/sessions/{sid}/runs` is empty for it),
  so `_verify_run_ownership` returns 404. The HTTP layer is exercised by
  `tests/unit/web/execution/test_outputs_routes.py` (10 tests, all pass)
  which mock the loader and verify the route's status-code semantics
  (200 / 403 / 404 / 410 / 415).
- **End-to-end via composer-driven session**: requires composer LLM
  convergence within budget. Currently gated by the gpt-5-mini model
  timing out at 300s on hardmode's verbose Linda prompt. Bumping
  `ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS` to 600s (or upgrading the
  composer model back to gpt-5) would close this. Tracked separately.

## Provenance

- Branch: `feat/eval-per-row-output-archival`
- Run-id: `05614d8e43634adf8d4c56c4777c6424` (in current `data/runs/audit.db`)
- Audit DB: fresh epoch-7 DB created by this run (the previous epoch-6
  DB containing 2026-05-03 evidence is preserved as
  `data/runs/audit.db.bak-pre-2026-05-06-rerun`)
- Service version: post-`feat/eval-per-row-output-archival` restart at
  `2026-05-06 08:42:22 AEST`
