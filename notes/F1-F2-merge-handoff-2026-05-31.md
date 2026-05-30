# F1/F2 resume-fork-reemit — merge hand-off (operator, key-holder)

**Branch:** `fix/resume-fork-reemit` (worktree `/home/john/elspeth/.worktrees/fix-resume-fork-reemit`), HEAD `b516921ee`, off merge-base `fd4830b42`.
**Target:** `RC5.2` (currently `5eacd917c`), merge with `--no-ff`.
**Status:** branch is COMPLETE, end-to-end reviewed, all gates green in the worktree. The only thing blocking the merge is the cicd-judge **HMAC signing coupling** — operator-only. Prepared 2026-05-31.

## Why an agent can't do this merge
RC5.2 adopted cicd-judge signing AFTER this branch's merge-base (commits `1b68e8b3e` signed 221 tier-model suppressions; `c93c58639` made the tier-model pre-commit hook fire). Two consequences:
1. The tier-model gate **`ValueError`s at load** against RC5.2's signed allowlist without `ELSPETH_JUDGE_METADATA_HMAC_KEY` — a keyless env can't pass the pre-commit hook.
2. This branch legitimately edits source files that contain **judge-gated (signed)** tier-model entries. Editing a file invalidates the whole-file `file_fingerprint` of every signed entry in it, and the `rotate` tool **refuses** to mechanically rotate judge-gated entries ("would silently rebind to code the judge didn't inspect"). They must be **re-justified** (delete + re-`justify`) with the key.

A keyless agent updating fingerprints while leaving stale signatures would produce entries that look validly-signed but aren't — forbidden by the custody model, and CI-with-key rejects them. So this is genuinely your action.

## The only merge conflict
`config/cicd/enforce_tier_model/core.yaml` (RC5.2's reify/signing vs this branch's pre-signing rotations). `src/elspeth/engine/orchestrator/core.py` **auto-merges cleanly** (RC5.2's `_quarantine_row` change at ~L1841 is disjoint from this branch's resume-finalization changes at ~L2626+).

## Procedure (with the key in your env)
```bash
cd /home/john/elspeth                       # main checkout, on RC5.2
git merge --no-ff --no-commit fix/resume-fork-reemit
git checkout --ours config/cicd/enforce_tier_model/core.yaml   # keep RC5.2's signed allowlist as the base
```
### 1. Re-justify the 7 invalidated judge-gated entries
Each entry below sits in a file this branch edited, so its `file_fingerprint` (and for some, `ast_path`/`fp`) no longer matches the merged source. Delete each from `core.yaml`, then re-`justify` against the merged source (the tool recomputes `fp`/`ast_path`/`file_fingerprint` and re-signs). The underlying suppressed logic is unchanged by this branch (the code only moved), so the original verdicts still hold — re-justify with the same rationale.

1. `core/checkpoint/recovery.py:R5:RecoveryManager:get_unprocessed_row_data:fp=7a2f9a2b860707a7`
2. `core/landscape/data_flow_repository.py:R6:DataFlowRepository:create_row:fp=8ca3a7ee3cc84783`
3. `core/landscape/data_flow_repository.py:R6:DataFlowRepository:create_row:fp=90984799b1de3e99`
4. `core/landscape/data_flow_repository.py:R6:DataFlowRepository:record_validation_error:fp=79239e021b66fa02`
5. `core/landscape/data_flow_repository.py:R6:DataFlowRepository:record_transform_error:fp=cf71d23d49a057eb`
6. `core/landscape/execution_repository.py:R6:ExecutionRepository:begin_node_state:fp=6e0f00f363e89415`
7. `core/landscape/query_repository.py:R6:QueryRepository:explain_row:fp=5b5158345e746330`

`justify` invocation shape (per CLAUDE.md):
```bash
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  justify --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model \
  --file-path <file> --symbol <Class.method> --rationale "<same as the deleted entry>" --owner "$USER"
```
(If you prefer, `reaudit` the four affected files first to confirm the verdicts still stand, then `justify`.)

### 2. Mechanically rotate the NON-judge entries
After the 7 are re-justified, run the sanctioned rotation tool — it will fix the remaining non-judge entries this branch's AST shifts touched (e.g. `core/checkpoint/recovery.py:R6:RecoveryManager:get_resume_point` `fp=c48ba8486b6c2e03 → 54948171154c08e1`):
```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  rotate --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --dry-run   # inspect, then drop --dry-run
```
(This branch already computed the correct post-edit fps for these in its own unsigned allowlist — e.g. `explain_row` would be `b81f5371dd48636a` — useful as a cross-check, but `justify`/`rotate` recompute them.)

### 3. Verify + commit + push
```bash
git add config/cicd/enforce_tier_model/core.yaml
env ELSPETH_JUDGE_METADATA_HMAC_KEY=<key> PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth      # full signature verify — must pass
git commit --no-edit        # full hooks run (key present → tier-model gate passes)
git push                    # bare push of RC5.2 — non-destructive
```

## POST-MERGE OPERATOR ACTION (do NOT skip before any staging deploy)
Schema epoch is **10 → 11**. The SQLite epoch gate rejects epoch-10 DBs. **Delete the staging audit DB** before the next staging run (`elspeth.foundryside.dev`). This is the documented operator gate.

## Three follow-ups (filigree filing was blocked by SCHEMA_MISMATCH — installed filigree v19 < project DB v21; `uv tool upgrade filigree`, then file)
1. **Item D** — record coalesce-failure as a queryable terminal audit signal so `rows_coalesce_failed` reconstructs cumulatively incl. run-1-pre-interrupt failures (schema/epoch change; no-rows branch already ships 0). P2.
2. **Concern #1** — FAILED-ceremony `partial_result` still builds resume-only counters (telemetry-only; the failure ceremony fires before the sweep — real ordering obstacle to unifying it). P3.
3. **`rows_buffered` semantics** — on a `count==N` mid-stream aggregation trigger, the live counter reports N−1 but audit-derive reports N (records ARE persisted, faithfully reconstructed). No test defines the intended value; needs a decision on which is correct. P2/P3.

## What this branch delivers (for the merge message / PR body)
Stops checkpoint resume from re-emitting completed fork/expand/coalesce branches (drives incomplete child tokens in place via `resume_incomplete_token`, no source restart); schema epoch 11 (`tokens.token_data_ref` envelope + `node_states.resume_checkpoint_id` provenance marker); F2 counter reconciliation with uninterrupted runs. End-to-end review found + fixed two production bugs: `1b9141bf5` (buffered/held tokens were re-driven → coalesce crash / aggregation double-emit) and `4d58cbd22` (`derive` over-counted `rows_coalesced`/`rows_succeeded` on every coalesce-success). Full memory: `project_f1_resume_reemit_fix.md`.
