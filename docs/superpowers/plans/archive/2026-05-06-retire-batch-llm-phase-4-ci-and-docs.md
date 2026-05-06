# Phase 4 — CI Allowlists and Documentation Sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trim CI allowlists that exempted the deleted plugins, then sweep documentation (architecture, user-manual, configuration reference, environment variables, release notes, internal plans/specs) to remove references to the retired transforms. Leave historical documents (CHANGELOGs, archived plans) intact — they are history, not live documentation.

**Architecture:** Pure config + documentation phase. No runtime change. The CI allowlists are enforced by `scripts/cicd/enforce_tier_model.py` and `scripts/cicd/enforce_freeze_guards.py` and would surface "unmatched allowlist entry" warnings on the next CI run if not updated. Documentation drift is the smaller risk but worth doing in the same change for reviewer clarity.

**Tech Stack:** YAML for allowlists, Markdown for docs. Verification via `python scripts/cicd/enforce_tier_model.py check`.

---

## Task 1: Trim `config/cicd/contracts-whitelist.yaml`

**Files:**
- Modify: `config/cicd/contracts-whitelist.yaml:303-304, 486` (remove three entries)

- [ ] **Step 1: Inspect context**

```bash
sed -n '298,310p' config/cicd/contracts-whitelist.yaml
sed -n '480,490p' config/cicd/contracts-whitelist.yaml
```

Expected: clear comment headers identifying the entries as "Azure batch LLM" and "OpenRouter batch LLM".

- [ ] **Step 2: Remove the three entries (and any now-empty section header)**

Use Edit. Delete:

```yaml
  # Azure batch LLM (checkpoint params typed as BatchCheckpointState — only config/return remain)
  - "src/elspeth/plugins/transforms/llm/azure_batch.py:AzureBatchLLMTransform.__init__:config"
  - "src/elspeth/plugins/transforms/llm/azure_batch.py:AzureBatchLLMTransform.azure_config:return"
```

and:

```yaml
  # OpenRouter batch LLM (async batch API)
  - "src/elspeth/plugins/transforms/llm/openrouter_batch.py:OpenRouterBatchLLMTransform.__init__:config"
```

Note the comment "async batch API" was misleading even before retirement (OpenRouter's plugin is concurrent-not-async). Removal is the right resolution.

- [ ] **Step 3: Verify allowlist enforcement**

```bash
.venv/bin/python -m scripts.check_contracts 2>&1 | tail -20
```

Expected: green. If the script reports "unmatched allowlist entry: ..." for any other batch-LLM line we missed, remove that line too.

- [ ] **Step 4: Commit**

```bash
git add config/cicd/contracts-whitelist.yaml
git commit -m "chore(cicd): drop batch-LLM entries from contracts whitelist"
```

---

## Task 2: Trim `config/cicd/enforce_frozen_annotations/existing.yaml`

**Files:**
- Modify: `config/cicd/enforce_frozen_annotations/existing.yaml:28-29` (remove two entries)

- [ ] **Step 1: Inspect**

```bash
sed -n '25,32p' config/cicd/enforce_frozen_annotations/existing.yaml
```

Expected:

```yaml
  # LLM batch results — pipeline/API data
  - key: "src/elspeth/plugins/transforms/llm/openrouter_batch.py:_RowSuccess:row"
  - key: "src/elspeth/plugins/transforms/llm/openrouter_batch.py:_RowFailure:error"
```

- [ ] **Step 2: Delete the two entries plus their comment header**

Use Edit. Remove the three lines above.

- [ ] **Step 3: Verify the freeze-guards enforcer**

```bash
.venv/bin/python scripts/cicd/enforce_freeze_guards.py 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add config/cicd/enforce_frozen_annotations/existing.yaml
git commit -m "chore(cicd): drop batch-LLM entries from freeze-guards allowlist"
```

---

## Task 3: Update `ARCHITECTURE.md`

**Files:**
- Modify: `ARCHITECTURE.md` (remove C4 components for AzureBatchLLM and OpenRouterBatchLLM, update transform-count summary)

- [ ] **Step 1: Inspect context**

```bash
grep -n "azure_batch\|openrouter_batch\|AzureBatchLLM\|OpenRouterBatchLLM" ARCHITECTURE.md
```

Expected hits include:
- C4 component diagrams (`Component(azure_batch, ...)`, `Component(openrouter_batch, ...)`)
- Plugin-summary table row stating "13 plugins" (becomes 11)
- The "LLM Transforms" line ("Unified LLMTransform ... + azure_batch + openrouter_batch")

- [ ] **Step 2: Edit each site**

Use Edit. Remove:
- The two `Component(azure_batch, ...)` and `Component(openrouter_batch, ...)` lines from the C4 container block
- The `+ azure_batch + openrouter_batch` suffix from the LLM Transforms description
- Update the "13 plugins" count to "11 plugins" (verify by counting kept transform names in the existing list)

- [ ] **Step 3: Verify**

```bash
grep -n "azure_batch\|openrouter_batch" ARCHITECTURE.md 2>&1
```

Expected: no hits.

- [ ] **Step 4: Commit**

```bash
git add ARCHITECTURE.md
git commit -m "docs(architecture): remove batch-LLM components from C4 diagrams + plugin count"
```

---

## Task 4: Update `docs/guides/user-manual.md`

**Files:**
- Modify: `docs/guides/user-manual.md` (remove `azure_batch_llm` and `openrouter_batch_llm` plugin descriptions)

- [ ] **Step 1: Locate the entries**

```bash
grep -B1 -A1 "azure_batch_llm\|openrouter_batch_llm" docs/guides/user-manual.md
```

Expected:

```
  azure_batch_llm      - Batch LLM transform using Azure OpenAI Batch API.
  openrouter_batch_llm - Batch-aware LLM transform using OpenRouter API.
```

- [ ] **Step 2: Delete the two entries**

Use Edit. Remove both lines and update any surrounding prose if it counts plugins or references batch-mode.

- [ ] **Step 3: Search for "batch mode" / "batched" sections**

```bash
grep -n -i "batch mode\|batched llm\|batch llm" docs/guides/user-manual.md 2>&1
```

If hits exist, delete or rewrite the surrounding paragraphs.

- [ ] **Step 4: Verify**

```bash
grep -n "azure_batch_llm\|openrouter_batch_llm\|batch mode\|batched" docs/guides/user-manual.md 2>&1
```

Expected: no hits (or only incidental "batch" references for batch_replicate / batch_stats / aggregations).

- [ ] **Step 5: Commit**

```bash
git add docs/guides/user-manual.md
git commit -m "docs(user-manual): remove batch-LLM transform documentation"
```

---

## Task 5: Update reference docs

**Files:**
- Modify: `docs/reference/configuration.md` (remove batch-LLM rows from transform table)
- Modify: `docs/reference/environment-variables.md` (remove Azure batch-related env vars if any are exclusive to azure_batch_llm)

- [ ] **Step 1: Inspect**

```bash
grep -B1 -A1 "azure_batch_llm\|openrouter_batch_llm" docs/reference/configuration.md
grep -n "AZURE.*BATCH\|BATCH.*AZURE\|azure_batch" docs/reference/environment-variables.md
```

Expected: clear table rows in `configuration.md`. `environment-variables.md` may have no batch-exclusive entries; if it does, they likely live under an "Azure Batch API" section.

- [ ] **Step 2: Delete identified rows / sections**

Use Edit. Remove the table rows:

```
| `azure_batch_llm` | Azure Batch API for LLM (50% cost savings) |
| `openrouter_batch_llm` | OpenRouter Batch HTTP API |
```

In `environment-variables.md`, remove only env vars that are *exclusive* to azure_batch_llm. Shared Azure auth env vars (`AZURE_OPENAI_API_KEY` etc.) stay — the regular `llm` transform with `provider: azure` still uses them.

- [ ] **Step 3: Verify**

```bash
grep -n "azure_batch_llm\|openrouter_batch_llm" docs/reference/ 2>&1
```

Expected: no hits.

- [ ] **Step 4: Commit**

```bash
git add docs/reference/configuration.md docs/reference/environment-variables.md
git commit -m "docs(reference): drop batch-LLM rows from config + env-vars reference"
```

---

## Task 6: Sweep release notes

**Files:**
- Modify: `docs/release/feature-inventory.md`
- Modify: `docs/release/rc3-checklist.md`

- [ ] **Step 1: Inspect**

```bash
grep -n "azure_batch_llm\|openrouter_batch_llm\|batch-LLM\|BatchLLM" docs/release/feature-inventory.md docs/release/rc3-checklist.md
```

Expected: feature-inventory will list the two transforms; the rc3 checklist may have a checkbox for them.

- [ ] **Step 2: Edit**

Use Edit. Remove transform inventory entries. For rc3-checklist, replace any "Verify azure_batch_llm" / "Verify openrouter_batch_llm" checkboxes with a "Verify retirement of batch-LLM transforms (ADR-020)" item linked to ADR-020.

- [ ] **Step 3: Verify**

```bash
grep -n "azure_batch_llm\|openrouter_batch_llm" docs/release/ 2>&1
```

Expected: no hits.

- [ ] **Step 4: Commit**

```bash
git add docs/release/feature-inventory.md docs/release/rc3-checklist.md
git commit -m "docs(release): update inventory + rc3 checklist for batch-LLM retirement"
```

---

## Task 7: Sweep active plans and specs

**Files (active — edit):**
- `docs/superpowers/plans/2026-04-15-plugin-source-file-hash.md`
- `docs/superpowers/plans/2026-04-21-remaining-transform-invariant-migration.md` (its scope shrinks)
- `docs/superpowers/specs/2026-04-15-plugin-version-audit-design.md`
- `docs/architecture/adr/001-plugin-level-concurrency.md` (if it cites either transform as a motivating use case, add a note that the use case has been retired — do not modify the historical decision text)
- `docs/architecture/telemetry.md`
- `docs/audits/test-infrastructure-audit-2026-03-01.md`
- `docs/bugs/process/CODEX_LOG.md`
- `docs/guides/tier2-tracing.md`
- `docs/plans/05-quality-assessment-t10-llm-consolidation.md`
- `docs/plans/2026-02-25-llm-plugin-consolidation.md`
- `docs/plans/2026-02-26-t17-plugincontext-protocol-split-design.md`

**Files (historical — LEAVE INTACT):**
- `CHANGELOG.md`, `CHANGELOG-RC1.md`, `CHANGELOG-RC2.md`
- Anything under `docs/plans/completed/` or `docs/superpowers/plans/archive/` or `docs/superpowers/specs/archive/`
- `docs/arch-analysis-2026-04-29-1500/` (a dated analysis snapshot)

- [ ] **Step 1: Inventory hits in active docs**

```bash
grep -rln "azure_batch_llm\|openrouter_batch_llm\|AzureBatchLLM\|OpenRouterBatchLLM" docs/ ARCHITECTURE.md 2>&1 | grep -v "CHANGELOG\|/archive/\|/completed/\|arch-analysis-2026-04-29-1500\|020-retire-batch-llm\|2026-04-21-batch-llm-invariant\|2026-05-06-retire-batch-llm"
```

Expected: a list. Each file gets a separate edit pass.

- [ ] **Step 2: Edit each active doc**

For each file:
- If the doc *describes* the transform as a current feature: remove the description.
- If the doc *cites* the transform as motivation, example, or test target: replace with a note such as "(retired 2026-05-06 per ADR-020)" or remove the citation if it's incidental.
- Do NOT rewrite history. Plans and specs that documented intent at their time of writing should keep that intent visible; just append a status line if needed.

For `2026-04-21-remaining-transform-invariant-migration.md` specifically, update its scope statement and skip-set to reflect that AzureBatchLLMTransform and OpenRouterBatchLLMTransform are no longer migration targets.

- [ ] **Step 3: Verify**

```bash
grep -rln "azure_batch_llm\|openrouter_batch_llm\|AzureBatchLLM\|OpenRouterBatchLLM" docs/ ARCHITECTURE.md 2>&1 | grep -v "CHANGELOG\|/archive/\|/completed/\|arch-analysis-2026-04-29-1500\|020-retire-batch-llm\|2026-04-21-batch-llm-invariant\|2026-05-06-retire-batch-llm"
```

Expected: zero hits in active docs.

- [ ] **Step 4: Commit (one or two commits depending on volume)**

```bash
git add docs/
git commit -m "docs: sweep active plans + specs to reflect batch-LLM retirement"
```

If the diff is unwieldy (>10 files), split into thematic commits (e.g., one for plans, one for specs, one for guides).

---

## Phase 4 Exit Criteria

- [ ] `python -m scripts.check_contracts` green
- [ ] `python scripts/cicd/enforce_freeze_guards.py` green
- [ ] No active doc references `azure_batch_llm`, `openrouter_batch_llm`, `AzureBatchLLM`, or `OpenRouterBatchLLM`
- [ ] CHANGELOG, archived plans, and the dated arch-analysis snapshot are untouched
- [ ] Phase 1 ADR remains the canonical pointer; superseded design doc shows the banner
- [ ] `git log` shows ~5 commits scoped to `config/cicd/`, `docs/`, and `ARCHITECTURE.md`

After Phase 4, the codebase + documentation are consistent with the retirement decision. Phase 5 verifies the whole change end-to-end and closes the filigree work items.
