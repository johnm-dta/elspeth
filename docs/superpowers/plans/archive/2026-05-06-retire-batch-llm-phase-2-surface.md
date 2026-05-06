# Phase 2 — Surface Deletion (Plugin Files, Examples, Composer Skill, Plugin Tests)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the two LLM batch plugin files, their plugin-specific tests, broken example YAMLs, composer skill catalog entries, and update plugin-discovery test counts. After this phase the codebase still imports `BatchPendingError` and `BatchCheckpointState` (engine demolition is Phase 3) but no transform implements them.

**Architecture:** Surface-only changes. Engine and contracts demolition is intentionally deferred to Phase 3 so this phase produces a small, easy-to-review commit set. Tests will pass at the end of this phase if engine code remains compatible — `BatchPendingError` handlers in the orchestrator/aggregation executors are dead code after this phase but not broken. The discovery tests adjust counts (17 → 15 transforms).

**Tech Stack:** Python 3.13, pytest, pluggy.

---

## Task 1: Delete the two plugin source files

**Files:**
- Delete: `src/elspeth/plugins/transforms/llm/openrouter_batch.py`
- Delete: `src/elspeth/plugins/transforms/llm/azure_batch.py`

- [ ] **Step 1: Confirm no internal imports remain (other than the plugins themselves)**

Run:
```bash
grep -rn "from elspeth.plugins.transforms.llm.openrouter_batch\|from elspeth.plugins.transforms.llm.azure_batch\|import openrouter_batch\|import azure_batch" src/ tests/ --include="*.py" 2>&1 | head
```

Expected: zero or only self-references inside the files about to be deleted, and inside test files that we will delete in Task 4. Anything else is a structural import we must address before deleting.

- [ ] **Step 2: Delete the two plugin files**

```bash
git rm src/elspeth/plugins/transforms/llm/openrouter_batch.py
git rm src/elspeth/plugins/transforms/llm/azure_batch.py
```

- [ ] **Step 3: Verify pluggy registration still resolves the kept transforms**

```bash
.venv/bin/python -c "
from elspeth.plugins.infrastructure.manager import PluginManager
m = PluginManager()
m.discover_builtin_plugins()
names = sorted(m.transform_names())
print('Transform count:', len(names))
print('batch transforms still registered:', [n for n in names if 'batch' in n])
print('llm batch transforms still registered:', [n for n in names if 'batch_llm' in n])
"
```

Expected: count is two less than before, `batch_replicate` and `batch_stats` still present, no `batch_llm` names appear.

- [ ] **Step 4: Commit**

```bash
git commit -m "chore(batch-llm): delete openrouter_batch.py + azure_batch.py plugin sources"
```

---

## Task 2: Delete the two example yaml files

**Files:**
- Delete: `examples/openrouter_sentiment/settings_batched.yaml`
- Delete: `examples/template_lookups/settings_batched.yaml`

- [ ] **Step 1: Delete**

```bash
git rm examples/openrouter_sentiment/settings_batched.yaml
git rm examples/template_lookups/settings_batched.yaml
```

- [ ] **Step 2: Verify dependent example READMEs still parse**

```bash
grep -n "settings_batched\|Batched" examples/openrouter_sentiment/README.md examples/template_lookups/README.md 2>&1 | head
```

Expected: hits in `openrouter_sentiment/README.md` (we edit that next) and possibly `template_lookups/README.md`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(examples): remove broken settings_batched.yaml entries"
```

---

## Task 3: Trim example READMEs

**Files:**
- Modify: `examples/openrouter_sentiment/README.md` (remove Batched row from variants table around line 99 and dedicated batched-mode section starting around line 105)
- Modify: `examples/template_lookups/README.md` (if it contains a Batched section)

- [ ] **Step 1: Read the openrouter_sentiment README**

Run: `cat examples/openrouter_sentiment/README.md`

Locate:
- The variants table that mentions "Batched"
- Any dedicated section that documents `settings_batched.yaml` (typically a "## Batched (...)" header with code blocks)

- [ ] **Step 2: Edit out the Batched references**

Use Edit. Remove:
- The Batched row from the variants table
- The entire dedicated Batched mode section (header to next sibling header)

- [ ] **Step 3: Repeat for template_lookups README if needed**

Run: `grep -n "Batched\|settings_batched\|batch_llm" examples/template_lookups/README.md`

If hits exist, remove them analogously.

- [ ] **Step 4: Verify final state**

```bash
grep -n "settings_batched\|openrouter_batch_llm\|azure_batch_llm" examples/openrouter_sentiment/README.md examples/template_lookups/README.md 2>&1
```

Expected: no hits.

- [ ] **Step 5: Commit**

```bash
git add examples/openrouter_sentiment/README.md examples/template_lookups/README.md
git commit -m "docs(examples): drop batched-LLM mode from openrouter_sentiment + template_lookups READMEs"
```

---

## Task 4: Delete plugin-specific test files

**Files (delete entirely):**
- `tests/integration/plugins/llm/test_openrouter_batch_integration.py`
- `tests/unit/plugins/llm/test_openrouter_batch.py`
- `tests/unit/plugins/llm/test_azure_batch.py`
- `tests/unit/plugins/llm/test_batch_single_row_contract.py` (docstring confirms scope: "Batch LLM transforms have a _process_single method")
- `tests/unit/plugins/llm/test_batch_errors.py` (imports `BatchCheckpointState`, scoped to batch LLM behaviour)

- [ ] **Step 1: Confirm scope of `test_batch_single_row_contract.py` and `test_batch_errors.py`**

Run:
```bash
head -20 tests/unit/plugins/llm/test_batch_single_row_contract.py tests/unit/plugins/llm/test_batch_errors.py
```

Expected: both clearly scoped to the LLM batch transforms (docstrings or imports name them).

- [ ] **Step 2: Delete the five test files**

```bash
git rm tests/integration/plugins/llm/test_openrouter_batch_integration.py
git rm tests/unit/plugins/llm/test_openrouter_batch.py
git rm tests/unit/plugins/llm/test_azure_batch.py
git rm tests/unit/plugins/llm/test_batch_single_row_contract.py
git rm tests/unit/plugins/llm/test_batch_errors.py
```

- [ ] **Step 3: Run the LLM unit test suite to confirm nothing else collateral broke**

```bash
.venv/bin/python -m pytest tests/unit/plugins/llm/ -x -q 2>&1 | tail -20
```

Expected: green (the deletions removed standalone tests). If failures appear, they are likely from `conftest.py` fixtures or `test_p1_bug_fixes.py` referencing deleted code — note them for Task 6.

- [ ] **Step 4: Commit**

```bash
git commit -m "test(llm): remove plugin-specific batch-LLM test files"
```

---

## Task 5: Update plugin-discovery test counts and assertions

**Files:**
- Modify: `tests/unit/plugins/test_discovery.py:212-213` — delete two assertions
- Modify: `tests/unit/plugins/test_discovery.py:253` — count `17` → `15` (and update the comment)
- Modify: `tests/unit/plugins/llm/test_plugin_registration.py:70-79` — delete two test methods

- [ ] **Step 1: Inspect the current state**

Run:
```bash
sed -n '208,260p' tests/unit/plugins/test_discovery.py
sed -n '65,85p' tests/unit/plugins/llm/test_plugin_registration.py
```

Expected: clear lines containing the assertions and test methods to remove.

- [ ] **Step 2: Remove the discovery assertions**

Use Edit on `tests/unit/plugins/test_discovery.py`. Delete the two lines:

```python
assert "azure_batch_llm" in transform_names, f"Missing azure_batch_llm in {transform_names}"
assert "openrouter_batch_llm" in transform_names, f"Missing openrouter_batch_llm in {transform_names}"
```

- [ ] **Step 3: Update the count assertion**

Change:

```python
            17  # 11 standard transforms + 2 azure safety + llm + azure_batch_llm + openrouter_batch_llm + rag_retrieval
```

to:

```python
            15  # 11 standard transforms + 2 azure safety + llm + rag_retrieval
```

- [ ] **Step 4: Remove the two test methods in test_plugin_registration.py**

Use Edit. Delete:

```python
    def test_azure_batch_llm_still_resolves(self) -> None:
        """azure_batch_llm transform resolves to a valid Pydantic config model."""
        config_model = get_transform_config_model("azure_batch_llm")
        assert config_model is not None

    def test_openrouter_batch_llm_still_resolves(self) -> None:
        """openrouter_batch_llm transform resolves to a valid Pydantic config model."""
        config_model = get_transform_config_model("openrouter_batch_llm")
        assert config_model is not None
```

- [ ] **Step 5: Run the discovery and registration tests**

```bash
.venv/bin/python -m pytest tests/unit/plugins/test_discovery.py tests/unit/plugins/llm/test_plugin_registration.py -x -v 2>&1 | tail -20
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/plugins/test_discovery.py tests/unit/plugins/llm/test_plugin_registration.py
git commit -m "test(discovery): drop batch-LLM names from registration + discovery counts"
```

---

## Task 6: Trim shared-test files that reference batch-LLM specifically

These tests stay (they cover non-batch-LLM concerns) but lose the batch-LLM-scoped pieces.

**Files:**
- Modify: `tests/unit/plugins/llm/test_p1_bug_fixes.py` — drop batch-LLM-specific test methods/imports
- Modify: `tests/unit/plugins/llm/conftest.py` — drop fixtures that only serve removed tests
- Modify: `tests/unit/plugins/llm/test_llm_config.py` — drop batch references if any
- Modify: `tests/unit/plugins/transforms/llm/test_value_sources.py` — drop BatchLLM-specific cases
- Modify: `tests/unit/telemetry/test_plugin_wiring.py` — drop BatchLLM wiring tests
- Modify: `tests/unit/cli/test_cli_helpers.py` — update help-message assertion (Phase 3 changes the actual help string; this test moves with it but the assertion text must match the new wording)
- Modify: `tests/unit/plugins/test_validation_path_agreement.py` — drop batch-LLM coverage if any

- [ ] **Step 1: Inventory the batch-LLM-specific lines in each file**

Run:
```bash
for f in tests/unit/plugins/llm/test_p1_bug_fixes.py \
         tests/unit/plugins/llm/conftest.py \
         tests/unit/plugins/llm/test_llm_config.py \
         tests/unit/plugins/transforms/llm/test_value_sources.py \
         tests/unit/telemetry/test_plugin_wiring.py \
         tests/unit/cli/test_cli_helpers.py \
         tests/unit/plugins/test_validation_path_agreement.py; do
  echo "=== $f ==="
  grep -n "azure_batch_llm\|openrouter_batch_llm\|AzureBatchLLM\|OpenRouterBatchLLM\|BatchPendingError\|BatchCheckpointState" "$f" 2>&1
done
```

Expected: line numbers per file. For each hit, decide whether the surrounding test method is purely batch-LLM-scoped (delete the method) or mixed (delete the batch portion only).

- [ ] **Step 2: Edit each file**

For each hit, use Edit to:
- Remove a whole `def test_*` method if its sole purpose is batch-LLM coverage
- Remove a single line if it's a stray import or assertion in a method that covers other things
- Remove a fixture from `conftest.py` if it has no surviving consumers (after the deletions in Tasks 4–5)

Re-run the grep from Step 1 after each file to confirm the hit is gone.

- [ ] **Step 3: Run the affected test directories**

```bash
.venv/bin/python -m pytest tests/unit/plugins/llm/ tests/unit/plugins/transforms/llm/ tests/unit/cli/ tests/unit/telemetry/test_plugin_wiring.py tests/unit/plugins/test_validation_path_agreement.py -x -q 2>&1 | tail -30
```

Expected: green. Failures here usually mean a deleted method had a fixture-dependent setup that we didn't fully remove.

- [ ] **Step 4: Commit**

```bash
git add -u tests/
git commit -m "test(llm): trim batch-LLM-specific cases from shared test files"
```

---

## Task 7: Remove composer/MCP skill catalog entries

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md:554-555` (delete two table rows)
- Modify: `.claude/skills/pipeline-composer/SKILL.md` (delete same rows if mirrored)
- Modify: `scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md` (research candidate; drop refs)

- [ ] **Step 1: Inspect the current rows**

```bash
sed -n '550,560p' src/elspeth/web/composer/skills/pipeline_composer.md
grep -n "azure_batch_llm\|openrouter_batch_llm" .claude/skills/pipeline-composer/SKILL.md scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md 2>&1
```

Expected: two adjacent table rows in `pipeline_composer.md` and possibly in `SKILL.md`.

- [ ] **Step 2: Delete the rows**

Use Edit. Remove from `pipeline_composer.md`:

```
| `azure_batch_llm` | Azure OpenAI batch processing | no | yes | yes | Adds response field (batch mode) |
| `openrouter_batch_llm` | OpenRouter batch processing | no | yes | yes | Adds response field (batch mode) |
```

Repeat for `.claude/skills/pipeline-composer/SKILL.md` and `pipeline_composer_v2_5concept.md` if they contain the same rows.

- [ ] **Step 3: Verify**

```bash
grep -n "azure_batch_llm\|openrouter_batch_llm" src/elspeth/web/composer/skills/pipeline_composer.md .claude/skills/pipeline-composer/SKILL.md scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md 2>&1
```

Expected: no hits.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md .claude/skills/pipeline-composer/SKILL.md scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md
git commit -m "docs(composer): drop batch-LLM transforms from skill catalogs"
```

---

## Phase 2 Exit Criteria

- [ ] `src/elspeth/plugins/transforms/llm/openrouter_batch.py` and `azure_batch.py` no longer in tree
- [ ] `examples/openrouter_sentiment/settings_batched.yaml` and `examples/template_lookups/settings_batched.yaml` no longer in tree
- [ ] Plugin-discovery test counts updated; both example READMEs no longer mention batched mode
- [ ] Plugin-specific test files (5) deleted; shared test files trimmed
- [ ] Composer/MCP skill catalogs no longer list the retired transforms
- [ ] `pytest tests/unit -x -q` green
- [ ] `git log` shows ~6 commits scoped to `src/elspeth/plugins/`, `examples/`, `tests/`, and `src/elspeth/web/composer/skills/`

After Phase 2, `BatchPendingError` and `BatchCheckpointState` are still defined and exported from `elspeth.contracts` but have no in-tree consumers. Phase 3 removes them.
