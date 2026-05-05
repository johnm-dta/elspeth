# Phase 5 — Verification and Filigree Close-Out

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the full verification matrix (mypy + ruff + pytest + tier-model + freeze-guards), execute a dogfood smoke run on a representative non-LLM example to confirm runtime is unchanged, then close out filigree work items: close `elspeth-2799f6ec22`, comment on `elspeth-be398f0bcb` to reflect the scope reduction, and review its six children for redundancy.

**Architecture:** Pure verification. No code changes unless something fails — in which case, route the fix back through the appropriate phase plan and re-run.

**Tech Stack:** mypy, ruff, pytest, tier-model enforcer, ELSPETH CLI for dogfood, filigree MCP for close-out.

---

## Task 1: Static analysis — mypy + ruff

- [ ] **Step 1: Type-check the entire source tree**

```bash
.venv/bin/python -m mypy src/ 2>&1 | tail -20
```

Expected: green. Any failure is a regression introduced in Phases 2–4 — diagnose and fix in the same phase that introduced the regression, then re-run.

- [ ] **Step 2: Lint**

```bash
.venv/bin/python -m ruff check src/ tests/ 2>&1 | tail -20
```

Expected: green or only pre-existing warnings.

- [ ] **Step 3: Run config-contracts verifier**

```bash
.venv/bin/python -m scripts.check_contracts 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 4: Run freeze-guard enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_freeze_guards.py 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 5: Run tier-model enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model 2>&1 | tail -20
```

Expected: green. The retirement should not introduce upward imports; if it does, the cause is a missed handler removal.

---

## Task 2: Test suite

- [ ] **Step 1: Unit tests**

```bash
.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -20
```

Expected: green. Test count should be ~10–15 lower than pre-retirement (deleted plugin-specific tests + trimmed shared test methods).

- [ ] **Step 2: Integration tests**

```bash
.venv/bin/python -m pytest tests/integration -q 2>&1 | tail -20
```

Expected: green. The deleted `test_openrouter_batch_integration.py` should be the only integration test removed.

- [ ] **Step 3: Property tests** (if any)

```bash
ls tests/property/ 2>&1
.venv/bin/python -m pytest tests/property -q 2>&1 | tail -10
```

Expected: green. No batch-LLM-specific property tests are known.

- [ ] **Step 4: Confirm no skips related to batch-LLM remain**

```bash
.venv/bin/python -m pytest tests/ -q --collect-only 2>&1 | grep -i "batch.*llm\|llm.*batch" | head
```

Expected: no hits, or only hits referencing `batch_replicate`/`batch_stats` (the kept transforms).

---

## Task 3: Dogfood smoke run

Pick a representative non-batch-LLM example to confirm runtime is unaffected.

- [ ] **Step 1: Run a standalone example**

```bash
.venv/bin/elspeth run --settings examples/threshold_gate/settings.yaml --execute 2>&1 | tail -10
```

Expected: `Run COMPLETED: 8 rows processed | ✓8 succeeded ...`

- [ ] **Step 2: Run an aggregation example (verifies kept batch-aware path)**

```bash
.venv/bin/elspeth run --settings examples/batch_aggregation/settings.yaml --execute 2>&1 | tail -10
```

Expected: `Run COMPLETED` with rows aggregated successfully. This exercises `batch_stats` (kept) and confirms the aggregation executor's batch-dispatch path still works without `BatchPendingError` handlers.

- [ ] **Step 3: Run a deaggregation example**

```bash
.venv/bin/elspeth run --settings examples/deaggregation/settings.yaml --execute 2>&1 | tail -10
```

Expected: `Run COMPLETED` with row expansion. This exercises `batch_replicate` (kept).

- [ ] **Step 4: Confirm a previously broken `*_batched.yaml` is gone**

```bash
ls examples/openrouter_sentiment/settings_batched.yaml examples/template_lookups/settings_batched.yaml 2>&1
```

Expected: both files report "No such file or directory" (Phase 2 deleted them).

- [ ] **Step 5: Confirm CLI errors gracefully if someone passes an old config**

Create a temporary YAML referencing `azure_batch_llm`:

```bash
cat > /tmp/retired-test.yaml <<'EOF'
source:
  plugin: csv
  options:
    path: /tmp/x.csv
aggregations:
- name: agg
  plugin: azure_batch_llm
  trigger:
    type: count
    count: 5
  options: {}
sinks:
  out:
    plugin: json
    options:
      path: /tmp/out.json
EOF
echo "id,x" > /tmp/x.csv
.venv/bin/elspeth run --settings /tmp/retired-test.yaml --execute 2>&1 | tail -10
rm /tmp/retired-test.yaml /tmp/x.csv
```

Expected: a clear error (probably "Unknown transform 'azure_batch_llm'" or value-source enforcement failure). The CLI should NOT crash with `KeyError`/traceback; the error should be operator-readable and point at the retired plugin name.

If the error is unfriendly, file a follow-up ticket — it's not a retirement blocker but worth tracking.

---

## Task 4: Filigree close-out

- [ ] **Step 1: Confirm dogfood + tests + static analysis are all green**

If anything in Tasks 1–3 failed, do not proceed. Fix the regression and re-run.

- [ ] **Step 2: Comment on `elspeth-be398f0bcb` (parent migration epic) about scope reduction**

Use the filigree MCP `add_comment` tool:

```
Comment text:
"Retirement of azure_batch_llm + openrouter_batch_llm has landed (ADR-020, plan
docs/superpowers/plans/2026-05-06-retire-batch-llm-overview.md, main ticket
elspeth-2799f6ec22).

This epic's skip-set in tests/invariants/test_pass_through_invariants.py
explicitly named both retired transforms. They are no longer migration scope —
the skip-set has been updated and the children of this epic should be reviewed
for redundancy:

- elspeth-6276262180
- elspeth-543bc0e01c
- elspeth-b33bf7012e
- elspeth-db7d3ae046
- elspeth-f6d584d6a1
- elspeth-0ebb8cbe50

Any child whose only deliverable was migrating Azure or OpenRouter batch LLM
to invariant coverage should be closed-as-superseded. Any child covering other
transforms (FieldMapper, JSONExplode, WebScrapeTransform, etc.) keeps its
mandate."
```

- [ ] **Step 3: Review the six child tickets**

For each child of `elspeth-be398f0bcb`:

```bash
mcp__filigree__get_issue elspeth-6276262180
mcp__filigree__get_issue elspeth-543bc0e01c
mcp__filigree__get_issue elspeth-b33bf7012e
mcp__filigree__get_issue elspeth-db7d3ae046
mcp__filigree__get_issue elspeth-f6d584d6a1
mcp__filigree__get_issue elspeth-0ebb8cbe50
```

For each: if the title or description scopes the work to AzureBatchLLM or OpenRouterBatchLLM exclusively, close it with `reason="superseded by ADR-020 retirement of batch-LLM transforms"`. Otherwise, leave open.

- [ ] **Step 4: Close `elspeth-2799f6ec22`**

Use `mcp__filigree__close_issue` with reason:

```
Closed by retirement land. ADR-020 written, plugin sources + examples + tests
+ composer entries deleted, BatchPendingError + BatchCheckpointState +
RowMappingEntry removed, _checkpoint/_batch_checkpoints plumbing removed from
PluginAuditContext, batch_checkpoints parameter removed from orchestrator,
all engine handlers removed, CI allowlists trimmed, documentation swept.
Verification matrix green: mypy, ruff, scripts/check_contracts,
scripts/cicd/enforce_freeze_guards, scripts/cicd/enforce_tier_model,
pytest tests/unit, pytest tests/integration. Dogfood smoke run on threshold_gate,
batch_aggregation, deaggregation all COMPLETED.
```

- [ ] **Step 5: Update `elspeth-07e33e68e5` priority if relevant**

The AGENTS.md row-count fix ticket is unrelated to retirement — it stays as P3.

- [ ] **Step 6: Run `mcp__filigree__get_stats` to confirm clean state**

Expected: `elspeth-2799f6ec22` is closed; `elspeth-be398f0bcb` has the comment; child tickets either closed-as-superseded or still open with reduced scope.

---

## Task 5: Final commit (if any close-out edits required code changes)

If the comment on `elspeth-be398f0bcb` mentioned a skip-set in test code that needs updating, update it now:

- [ ] **Step 1: Inspect the skip-set**

```bash
grep -B2 -A20 "AzureBatchLLM\|OpenRouterBatchLLM\|skip" tests/invariants/test_pass_through_invariants.py 2>&1 | head -40
```

Expected: a `SKIP_INVARIANT_TRANSFORMS` (or similar) constant listing the kept skips. If the retired classes are still listed, remove them.

- [ ] **Step 2: Edit the skip-set**

Use Edit. Remove `"AzureBatchLLMTransform"` and `"OpenRouterBatchLLMTransform"` from the list (the classes no longer exist).

- [ ] **Step 3: Run the invariant test suite**

```bash
.venv/bin/python -m pytest tests/invariants/test_pass_through_invariants.py -x -v 2>&1 | tail -30
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add tests/invariants/test_pass_through_invariants.py
git commit -m "test(invariants): drop retired batch-LLM transforms from skip-set"
```

---

## Phase 5 Exit Criteria

- [ ] `mypy src/` green
- [ ] `ruff check src/ tests/` green
- [ ] `python -m scripts.check_contracts` green
- [ ] `python scripts/cicd/enforce_freeze_guards.py` green
- [ ] `python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` green
- [ ] `pytest tests/unit -q` green
- [ ] `pytest tests/integration -q` green
- [ ] Dogfood smoke run on `threshold_gate`, `batch_aggregation`, `deaggregation` all `Run COMPLETED`
- [ ] `elspeth-2799f6ec22` closed
- [ ] `elspeth-be398f0bcb` has scope-reduction comment; redundant children reviewed and closed-as-superseded where applicable
- [ ] `elspeth-07e33e68e5` (AGENTS.md row-count fix) still open (separate work)
- [ ] `elspeth-1244b4f16e` (lru_cache investigation) still open (separate work)
- [ ] `elspeth-958fe00ed9` (test fixture tightening) still open (separate work)
- [ ] `tests/invariants/test_pass_through_invariants.py` skip-set updated if it named the retired classes

After Phase 5, the retirement is complete and verified end-to-end. The branch is ready for review.

---

## Final commit summary expected

By end of Phase 5, the cumulative `git log --oneline` for the retirement effort should show approximately:

```
docs(adr): adr-020 retire batch-LLM transforms
docs(specs): mark batch-LLM invariant follow-on superseded by ADR-020
chore(batch-llm): delete openrouter_batch.py + azure_batch.py plugin sources
chore(examples): remove broken settings_batched.yaml entries
docs(examples): drop batched-LLM mode from openrouter_sentiment + template_lookups READMEs
test(llm): remove plugin-specific batch-LLM test files
test(discovery): drop batch-LLM names from registration + discovery counts
test(llm): trim batch-LLM-specific cases from shared test files
docs(composer): drop batch-LLM transforms from skill catalogs
refactor(cli): drop azure_batch_llm from batch-aware help text
refactor(engine): remove BatchPendingError handlers from aggregation + orchestrator
refactor(orchestrator): remove batch_checkpoints parameter from run/resume flow
refactor(contracts): remove get_checkpoint/set_checkpoint plumbing from PluginAuditContext
refactor(contracts): remove BatchPendingError, BatchCheckpointState, RowMappingEntry
test(contracts): remove BatchCheckpointState + BatchPendingError test coverage
chore(cicd): drop batch-LLM entries from contracts whitelist
chore(cicd): drop batch-LLM entries from freeze-guards allowlist
docs(architecture): remove batch-LLM components from C4 diagrams + plugin count
docs(user-manual): remove batch-LLM transform documentation
docs(reference): drop batch-LLM rows from config + env-vars reference
docs(release): update inventory + rc3 checklist for batch-LLM retirement
docs: sweep active plans + specs to reflect batch-LLM retirement
test(invariants): drop retired batch-LLM transforms from skip-set
```

~22 commits, all named per `<type>(<scope>): <subject>` convention.
