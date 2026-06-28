# Composer Tier 1 reliability — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the five Tier 1 reliability items from `docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md` and verify the canonical URL→download→line-explode RGR scenario reaches ≥4/6 hard-GREEN under deterministic sampling.

**Architecture:** Item 1 changes the composer LLM call site to use `temperature=0.0, seed=42` and threads both values into the existing `ComposerLLMCall` audit envelope (frozen L0 dataclass → JSON column on `role=tool` chat messages — no DB migration). Item 2 fixes a single 404'd preflight URL. Items 3–5 are skill/schema-prose changes targeting the connection-name semantic gap that dominates schema-construction failures.

**Tech Stack:** Python 3.13 (frozen dataclasses, Pydantic via service layer), LiteLLM (`acompletion`), pytest + pytest-asyncio, bash for the eval harness, FastAPI for the catalog router whose route table is being verified.

---

## Reality corrections vs. the tasking document

The tasking instructed: *"If you find a fact in the investigation note that contradicts the live code, trust the live code, update this tasking via comment on the parent epic, and continue."* Three corrections are baked into the tasks below; all are also surfaced to the parent epic `elspeth-1d3be32a8a` as a comment.

| # | Tasking claim | Live reality | Effect on plan |
|---|---------------|--------------|----------------|
| 1 | `recorder.llm_calls` is a table; new columns may need declarative schema change + DB-delete migration | `BufferingRecorder` is in-memory; records serialize to a JSON column via `ComposerLLMCall.to_dict()`. The dataclass lives at `src/elspeth/contracts/composer_llm_audit.py`. **No SQL change, no DB delete, no migration needed.** | Item 1 is a dataclass-field addition + builder kwarg + two test-fixture updates. |
| 2 | Preflight uses `/api/login` and `/api/catalog` | Preflight (real path: `evals/lib/preflight.sh`, NOT `evals/composer-harness/lib/preflight.sh`) already uses `/api/auth/login`. The only broken call is `GET /api/catalog` (line 65) — the live router has no root route, only sub-routes. | Item 2 collapses to a one-line URL change to `/api/catalog/sources`. |
| 3 | Item 5 covers four tools (`set_pipeline`, `upsert_node`, `upsert_edge`, `set_output`) | `upsert_edge` uses **node IDs and `edge_type` enums**, not connection-name strings. Enriching a node-id field with a connection-string description would mislead the model. | Item 5 covers the connection-string-bearing fields across `set_source`, `set_source_from_blob`, `set_pipeline`, `upsert_node`, `set_output`. `upsert_edge` is excluded. `remove_output.sink_name` and `patch_output_options.sink_name` are excluded too — they reference an existing sink, not a fresh wiring decision. |
| 4 | Audit verification SQL: `json_extract(tool_calls, '$.call.temperature')` | `tool_calls` is a JSON **array** (single-element wrapper at `routes.py:756`: `tool_calls=[llm_call_audit_envelope(call)]`). Correct path is `$[0].call.temperature` and `$[0]._kind`. The original SQL silently returns NULL/zero rows. | Cohort verification SQL in steps C1.4 and C2.5 uses `$[0].…` everywhere. |

---

## File structure

| File | Purpose | Touch |
|------|---------|-------|
| `src/elspeth/contracts/composer_llm_audit.py` | L0 frozen dataclass for one LLM-call audit record | Modify — add `temperature: float`, `seed: int` fields |
| `src/elspeth/web/composer/service.py` | Composer service: LLM call sites, record builder, audit wrapping | Modify — add module-level constants, pass kwargs to `_litellm_acompletion`, thread into `_build_llm_call_record` |
| `tests/unit/contracts/test_composer_llm_audit.py` | Unit tests for the dataclass | Modify — extend the fixture default and add `temperature`/`seed` round-trip assertion |
| `tests/unit/web/sessions/test_routes.py` | Session-route fixtures that construct `ComposerLLMCall` | Modify — extend the fixture default |
| `tests/unit/web/composer/test_service.py` | Composer service unit tests; some already mock `_litellm_acompletion` | Modify — assert the LLM call site receives `temperature=0.0, seed=42` |
| `evals/lib/preflight.sh` | Doctor preflight script | Modify — change one URL |
| `src/elspeth/web/composer/skills/pipeline_composer.md` | Composer system prompt (the "skill") | Modify — relocate Connection Model section, add Wiring repair examples |
| `src/elspeth/web/composer/tools.py` | LiteLLM-format tool definitions (JSON Schema) | Modify — enrich `description` + `examples` on connection-name string fields |

**No new files.** Five existing files modified by source code. One harness file modified. One skill file modified. Two test fixture files modified.

---

## Pre-work — once, before any task

- [ ] **Step 0.1: Claim the epic in filigree**

```
mcp__filigree__update_issue id=elspeth-1d3be32a8a status=in_progress
mcp__filigree__add_comment id=elspeth-1d3be32a8a body="Starting Tier 1 implementation. Plan: docs/superpowers/plans/2026-05-06-composer-tier1-reliability-implementation.md. Reality corrections vs tasking: see plan §Reality corrections."
```

- [ ] **Step 0.2: Verify worktree state**

Run: `git status && git log --oneline -1`
Expected: HEAD at `19317366`, working tree may have uncommitted plan files (this plan + the tasking + the investigation note); no other uncommitted source changes.

- [ ] **Step 0.3: Verify Python version on the active venv**

Run: `.venv/bin/python --version`
Expected: `Python 3.13.x` (memory: `project_tier_model_python_version.md` — mismatched venv versions trigger ~300 spurious tier-model violations).

---

## Task 1: temperature=0.0 + seed=42 + audit-record persistence

**Files:**
- Modify: `src/elspeth/contracts/composer_llm_audit.py:32-72` — add two required fields to `ComposerLLMCall`
- Modify: `src/elspeth/web/composer/service.py` — add module constants near top, pass kwargs at the two `_litellm_acompletion` sites (currently lines 1727-1731 and 1749-1752), thread values into `_build_llm_call_record` (currently line 219-250)
- Modify: `tests/unit/contracts/test_composer_llm_audit.py:42` (and any sibling default-fixture functions in that file)
- Modify: `tests/unit/web/sessions/test_routes.py:110` (and any sibling default-fixture functions in that file)
- Modify: `tests/unit/web/composer/test_service.py` — add an assertion that the LLM call kwargs include the new params

**Skill check before edit:** invoke `logging-telemetry-policy` to confirm temperature/seed belong in audit (Landscape primacy), not telemetry or logging. Expected outcome: the `ComposerLLMCall` dataclass docstring already names this as the audit primitive (L0); both fields are request metadata that survive payload deletion via the integrity-hash contract — they unambiguously belong in audit.

- [ ] **Step 1.1: Write the failing dataclass test (TDD-RED)**

Append to `tests/unit/contracts/test_composer_llm_audit.py`:

```python
def test_composer_llm_call_records_temperature_and_seed() -> None:
    """temperature and seed are required audit fields and round-trip through to_dict()."""
    started = datetime(2026, 5, 6, 12, 0, 0, tzinfo=UTC)
    call = ComposerLLMCall(
        model_requested="openrouter/openai/gpt-5.4",
        model_returned="openrouter/openai/gpt-5.4",
        status=ComposerLLMCallStatus.SUCCESS,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=100,
        provider_request_id="req_abc",
        messages_hash="m" * 64,
        tools_spec_hash=None,
        started_at=started,
        finished_at=started,
        error_class=None,
        error_message=None,
        temperature=0.0,
        seed=42,
    )
    assert call.temperature == 0.0
    assert call.seed == 42
    payload = call.to_dict()
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42
```

- [ ] **Step 1.2: Run the test to confirm it fails for the right reason**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py::test_composer_llm_call_records_temperature_and_seed -x`
Expected: `TypeError: ComposerLLMCall.__init__() got an unexpected keyword argument 'temperature'` (or similar). NOT a different failure.

- [ ] **Step 1.3: Add the dataclass fields**

Edit `src/elspeth/contracts/composer_llm_audit.py`. Inside `ComposerLLMCall`, after `error_message: str | None` (line ~69) and before the cache-token defaulted fields (line ~70 onward), add two REQUIRED fields:

```python
    error_message: str | None
    temperature: float
    seed: int
    cached_prompt_tokens: int | None = None
```

Rationale for placement: required fields must precede defaulted fields in `@dataclass`. The cache-token fields already have defaults, so the new required fields go immediately before them.

Update the docstring (line ~33) to add a paragraph after the cache-token paragraph:

```python
    """
    ...
    ``temperature`` and ``seed`` capture the deterministic-sampling
    parameters set on every composer LLM request. Both are constant
    in the current implementation (``0.0`` / ``42``), but the audit
    row records the value actually sent so a reviewer can detect
    drift if the constants are ever changed and tie individual
    failures to the precise sampling regime that produced them.
    """
```

- [ ] **Step 1.4: Run the dataclass test to confirm it passes**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py::test_composer_llm_call_records_temperature_and_seed -x`
Expected: PASS.

- [ ] **Step 1.5: Run the full dataclass test module to confirm fixture defaults still work**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py -x`
Expected: failures in tests that use the default fixture (it omits `temperature`/`seed`). This is the "locked-in expectation" pattern — the test pinned the bug (no audit of sampling params); now we update the fixture.

- [ ] **Step 1.6: Update the L0 dataclass test fixture default**

Edit `tests/unit/contracts/test_composer_llm_audit.py`. Find the helper that builds the default `ComposerLLMCall` kwargs (around line 42 — search for `defaults = {`). Add:

```python
    defaults = {
        ...
        "error_message": None,
        "temperature": 0.0,
        "seed": 42,
    }
```

- [ ] **Step 1.7: Update the session-routes fixture default**

Edit `tests/unit/web/sessions/test_routes.py`. Find the equivalent helper (around line 110, search `defaults = {`). Apply the same two-field addition.

- [ ] **Step 1.8: Run both unit-test files to confirm fixtures pass**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_composer_llm_audit.py tests/unit/web/sessions/test_routes.py -x`
Expected: PASS.

- [ ] **Step 1.9: Add module constants to service.py**

Edit `src/elspeth/web/composer/service.py`. Find the imports / module-level area near the top (after the imports but before the first dataclass/function definition — search for the first `def _` or first class). Add:

```python
# Composer LLM sampling constants. Hardcoded for deterministic tool-call
# construction (RGR investigation 2026-05-06 §4.4). Configurability is
# Tier 2; do not read from settings/env without an ADR.
_COMPOSER_LLM_TEMPERATURE: Final[float] = 0.0
_COMPOSER_LLM_SEED: Final[int] = 42
```

If `Final` is not yet imported from `typing`, add `Final` to the `from typing import` line.

- [ ] **Step 1.10: Pass temperature/seed at the two `_litellm_acompletion` call sites**

Edit `_call_llm` (around line 1727) — change:

```python
            response = await _litellm_acompletion(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=_COMPOSER_LLM_TEMPERATURE,
                seed=_COMPOSER_LLM_SEED,
            )
```

Edit `_call_text_llm` (around line 1749) — same treatment, no `tools` param:

```python
            response = await _litellm_acompletion(
                model=self._model,
                messages=messages,
                temperature=_COMPOSER_LLM_TEMPERATURE,
                seed=_COMPOSER_LLM_SEED,
            )
```

Both call sites already have `try`/`except LiteLLMBadRequestError` handling. Per CLAUDE.md "defensive programming forbidden" + tasking anti-pattern #2, do NOT widen the except to catch new failures from these kwargs — if the provider rejects `seed` we want the existing `LiteLLMBadRequestError` path to surface and the audit row to record `BAD_REQUEST_ERROR`.

- [ ] **Step 1.11: Thread temperature/seed into `_build_llm_call_record`**

Edit `_build_llm_call_record` (around line 219). Add two kwargs and forward them into the constructor:

```python
def _build_llm_call_record(
    *,
    model_requested: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    status: ComposerLLMCallStatus,
    started_at: datetime,
    started_ns: int,
    response: Any | None = None,
    error_class: str | None = None,
    error_message: str | None = None,
    temperature: float,
    seed: int,
) -> ComposerLLMCall:
    usage = _token_usage_from_response(response)
    return ComposerLLMCall(
        ...
        error_message=error_message,
        temperature=temperature,
        seed=seed,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        ...
    )
```

Then update the call site in `_call_llm_with_audit`'s `finally` (around line 1835):

```python
                recorder.record_llm_call(
                    _build_llm_call_record(
                        model_requested=self._model,
                        messages=messages,
                        tools=tools,
                        status=status,
                        started_at=started_at,
                        started_ns=started_ns,
                        response=response,
                        error_class=error_class,
                        error_message=error_message,
                        temperature=_COMPOSER_LLM_TEMPERATURE,
                        seed=_COMPOSER_LLM_SEED,
                    )
                )
```

- [ ] **Step 1.12: Find any other call site of `_build_llm_call_record`**

Run: `grep -n "_build_llm_call_record" src/elspeth/web/composer/service.py`
Expected: a single call site (in `_call_llm_with_audit`'s `finally`). If a second site exists, apply the same kwargs.

- [ ] **Step 1.13: Add a service-layer test asserting kwargs reach litellm**

Edit `tests/unit/web/composer/test_service.py`. Find an existing test that mocks `_litellm_acompletion` for `_call_llm` (search for `_litellm_acompletion` or for mock patches on the LLM call). Add — or extend an existing test — to assert:

```python
    # Verify deterministic sampling params reach LiteLLM. Audit primacy:
    # the audit row records what was sent, so the call site MUST send
    # the constants — not "default" or "whatever the model defaults to."
    captured_kwargs = mock_litellm_acompletion.call_args.kwargs
    assert captured_kwargs["temperature"] == 0.0
    assert captured_kwargs["seed"] == 42
```

If no existing test mocks `_litellm_acompletion`, add a small one — but most likely there are several; pick the canonical one for `_call_llm` happy path and add the two assertions.

- [ ] **Step 1.14: Run the composer service test module**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_service.py -x`
Expected: PASS. Any pre-existing test that previously hardcoded `temperature` to a different value or asserted its absence will fail — that's the locked-in-expectation pattern; update the test rather than reverting the production change.

- [ ] **Step 1.15: Run the full unit test suite under composer + contracts**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/ tests/unit/contracts/test_composer_llm_audit.py tests/unit/web/sessions/test_routes.py -x`
Expected: PASS.

- [ ] **Step 1.16: Run mypy on the touched files**

Run: `.venv/bin/python -m mypy src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/service.py`
Expected: no new errors. `Final` import is the most likely complication; if `from typing import Final` is missing, add it.

- [ ] **Step 1.17: Run the tier-model enforcement on this change**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: no new violations. The dataclass is L0; the service is L3. Adding fields at L0 doesn't change layer dependencies.

- [ ] **Step 1.18: Commit**

```bash
git add src/elspeth/contracts/composer_llm_audit.py \
        src/elspeth/web/composer/service.py \
        tests/unit/contracts/test_composer_llm_audit.py \
        tests/unit/web/sessions/test_routes.py \
        tests/unit/web/composer/test_service.py

git commit -m "$(cat <<'EOF'
fix(composer): set temperature=0.0 + seed=42 on LLM calls; record both as audit fields

The composer skill prescribes temperature=0 to USERS while the host that runs
the composer LLM samples at the upstream-model default (~1.0 for OpenAI/gpt-5
family). RGR investigation 2026-05-06 §4.4 traced ~33% hard-GREEN ceiling on
the URL→download→line-explode scenario primarily to this variance source.

Hardcode 0.0 + 42 at both call sites (_call_llm tool loop, _call_text_llm
diagnostics), thread the values through _build_llm_call_record, and add two
required fields to ComposerLLMCall so each audit row records the exact
sampling regime that produced it. JSON column persistence is unchanged
(to_dict() carries the new fields).

No DB migration: ComposerLLMCall is an in-memory frozen dataclass; persistence
is via the existing JSON tool_calls envelope on role=tool chat messages.

Configurability is Tier 2 — do not read from settings/env without an ADR.

Refs: docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md §4.4 §7.1
Filigree: elspeth-1d3be32a8a

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Cohort #1 — post-temperature-fix re-baseline (6 RGR runs)

**Sequencing rule (tasking §5):** items 2–5 cannot land until this cohort is captured. The post-1 cohort is the new baseline against which items 2–5 are measured cumulatively.

- [ ] **Step C1.1: Restart elspeth-web.service**

```bash
sudo systemctl restart elspeth-web.service
sudo systemctl status elspeth-web.service --no-pager | head -20
```

Expected: `active (running)` and recent startup log.

- [ ] **Step C1.2: Confirm the deploy serves the new code**

```bash
curl -sS https://elspeth.foundryside.dev/api/health | jq .
```

Expected: 200 OK. (If a `/api/health` endpoint doesn't exist, hit `/api/auth/login` with OPTIONS — non-5xx confirms the host is up.)

- [ ] **Step C1.3: Run the RGR cohort (6 runs)**

```bash
cd /home/john/elspeth
for i in 1 2 3 4 5 6; do
  evals/composer-rgr/run_scenario.sh "post-item1-${i}"
done
```

Expected: each run prints a verdict line. Collect them.

- [ ] **Step C1.4: Verify audit-row persistence for one run**

For one of the GREEN runs, pick its `session_id` from `evals/composer-rgr/runs/*post-item1-*/session_id.txt`, then query the JSON column. **Important:** `tool_calls` is stored as a JSON array (single-element wrapper from `routes.py:756`), so the `_kind` discriminator and the call body live at `$[0]._kind` / `$[0].call.<field>`, NOT `$._kind` / `$.call.<field>`:

```bash
sqlite3 /home/john/elspeth/data/sessions.db "
  SELECT json_extract(tool_calls, '\$[0].call.temperature'),
         json_extract(tool_calls, '\$[0].call.seed')
  FROM chat_messages
  WHERE session_id = '<sid>'
    AND role = 'tool'
    AND json_extract(tool_calls, '\$[0]._kind') = 'llm_call_audit'
  LIMIT 5;
"
```

Expected: every row returns `0.0|42`. NOT `null|null` and NOT zero rows.

Cross-check that the existing pre-deploy data already follows this envelope (via `SELECT substr(tool_calls, 1, 80) FROM chat_messages WHERE role='tool' LIMIT 1;` — should start with `[{"_kind": "llm_call_audit", ...`).

- [ ] **Step C1.5: Decide pass/fail per tasking §6**

Compute hard-GREEN count from step C1.3.

- If **post-1 cohort < prior baseline (~3/9 hard-GREEN)** → STOP. Comment on the epic with the cohort table; do not proceed. Possible cause: provider silently rejects `seed` or `temperature`.
- If **post-1 cohort ≥ 4/6** → proceed AND continue with items 2–5 (exit criteria require all five items to land).
- Otherwise → proceed with items 2–5.

- [ ] **Step C1.6: Comment cohort #1 on the epic**

```
mcp__filigree__add_comment id=elspeth-1d3be32a8a body="Cohort #1 (post-item-1 re-baseline, 6 runs):

post-item1-1: <verdict>
post-item1-2: <verdict>
post-item1-3: <verdict>
post-item1-4: <verdict>
post-item1-5: <verdict>
post-item1-6: <verdict>

Hard-GREEN: <n>/6.
Audit verification: temperature=0.0, seed=42 confirmed on <n> rows in session <sid>.

Proceeding with items 2-5."
```

---

## Task 2: preflight URL fix (one line)

**Files:**
- Modify: `evals/lib/preflight.sh:65` — change one URL

**Reality vs tasking:** the tasking points to `evals/composer-harness/lib/preflight.sh` and says login is broken too. Both wrong. The real path is `evals/lib/preflight.sh`; login already uses `/api/auth/login`. Only the catalog GET is broken.

- [ ] **Step 2.1: Verify the broken endpoint**

```bash
grep -n "api/catalog" evals/lib/preflight.sh
```
Expected: line 65 has `"$ELSPETH_EVAL_BASE_URL/api/catalog"`.

- [ ] **Step 2.2: Verify the live route table has no root catalog endpoint**

```bash
grep -n "@catalog_router\|router.get\|router.post" src/elspeth/web/catalog/routes.py
```
Expected: routes are `/sources`, `/transforms`, `/sinks`, `/{plugin_type}/{name}/schema` — none at root. Confirms the bare `GET /api/catalog` returns 404.

- [ ] **Step 2.3: Edit the URL**

Edit `evals/lib/preflight.sh`, change the GET in the post-login section:

```bash
step "post-login API call (/api/catalog/sources)"
EVALS_JWT_FILE="$TMP_JWT" _evals_http_get \
  "$ELSPETH_EVAL_BASE_URL/api/catalog/sources" \
  /dev/null
```

`/api/catalog/sources` is sufficient as a cheap auth round-trip — one of the three per-type endpoints exercises the full router + auth + plugin-loader path. Per tasking item-2 anti-pattern #1 ("surgical URL swap only"), do not extend to all three types.

- [ ] **Step 2.4: Run the preflight against staging**

```bash
EVALS_ENV_FILE=evals/composer-harness/.env evals/composer-harness/hardmode/harness.sh --doctor
```

Expected: ends with `[INFO] all preflight checks passed`. Exit code 0.

- [ ] **Step 2.5: Commit**

```bash
git add evals/lib/preflight.sh

git commit -m "$(cat <<'EOF'
fix(evals): repair preflight catalog endpoint — /api/catalog has no root route

The /api/catalog router is mounted as a prefix with sub-routes only
(/sources, /transforms, /sinks, /{plugin_type}/{name}/schema). A bare
GET /api/catalog returns 404, so `harness.sh --doctor` has been failing
silently for any agent trying to use the multi-turn persona harness.

Switch the post-login auth round-trip to /api/catalog/sources — one
endpoint is sufficient for the doctor's purpose (verify host + auth +
plugin-loader path).

Refs: docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md §2.1
Filigree: elspeth-1d3be32a8a

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: move Connection Model to top of skill

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md` — relocate lines 156-238 (Connection Model, including its sub-sub-sections) to immediately after the TERMINATION GATE section ends (around line 58, before the `---` and `## Schema Vocabulary`)

**Mechanical only.** Per tasking item-3 description: cut and paste, do not edit content, no "see also" cross-ref at the old location.

- [ ] **Step 3.1: Verify exact section boundaries**

```bash
grep -n "^##\|^### \|^---$" src/elspeth/web/composer/skills/pipeline_composer.md | sed -n '1,30p'
```
Confirm:
- TERMINATION GATE starts `### TERMINATION GATE` at line 39
- Section ends at the `---` on or near line 59
- `## Schema Vocabulary — Five Distinct Concepts` begins line 61
- `### Connection Model` begins line 156, ending at line 238 right before `### Node Types` at line 242

- [ ] **Step 3.2: Confirm no incoming references would break**

```bash
grep -n -i "connection model" src/elspeth/web/composer/skills/pipeline_composer.md
```
Expected: only the section heading itself. No "see Connection Model below" / "see Connection Model above" cross-refs that would lose direction. If any cross-ref says "below" and the section is moving up, the cross-ref text needs to change — but per tasking, the cleanest answer is to leave the cross-ref pointing by section name (the runtime LLM resolves "Connection Model" by content, not by spatial position).

- [ ] **Step 3.3: Read the section to move**

Read lines 156-240 of `pipeline_composer.md`. Confirm the section ends with the `Boolean routes` subsection ending around line 238, and that line 240 (`Every pipeline needs: **one source**...`) belongs to the **next** section (Node Types) NOT to Connection Model. The cut should NOT include line 240 onward.

- [ ] **Step 3.4: Perform the relocation**

Use a single Edit call: `old_string` is the Connection Model block from `### Connection Model` through the Boolean routes subsection (ending with the YAML/JSON note about quoted booleans). `new_string` is empty (deletion at that location). Then a second Edit insert: place the same block immediately after the TERMINATION GATE block ends, before the `---` separator.

In practice, this is two `Edit` operations on the same file in a single conversation turn. The simpler shape is:

1. Edit #1: `old_string` = `### Connection Model\n...\nIn YAML: ...\n` → `new_string` = `` (empty). This deletes the section from line 156.
2. Edit #2: `old_string` = the `---\n\n## Schema Vocabulary` boundary at line 59-61 → `new_string` = `<the cut Connection Model block>\n\n---\n\n## Schema Vocabulary`. This inserts the cut block immediately before the existing `---`.

The skill is `@lru_cache`'d at module-import time (see `src/elspeth/web/composer/prompts.py:23`); content reordering is invisible until restart.

- [ ] **Step 3.5: Verify the diff is move-only**

```bash
git diff --stat src/elspeth/web/composer/skills/pipeline_composer.md
```
Expected: `0 insertions(-), 0 deletions(-)` net OR a tiny offset (whitespace-equivalent). The total line count of the file should be unchanged.

```bash
git diff src/elspeth/web/composer/skills/pipeline_composer.md | grep -c '^+\|^-' | head
```
Expected: roughly equal `+` and `-` counts (the same lines moved, not edited).

- [ ] **Step 3.6: Smoke test by restart + spot-check via the live deploy**

```bash
sudo systemctl restart elspeth-web.service
```

Optional manual: log in to staging, open a fresh session, ask "what is a connection in your model?" — the model should answer using the Connection Model content. This is a smoke test, not a regression gate; do not commit time to a full RGR run.

- [ ] **Step 3.7: Commit (do NOT push or run the cohort yet — items 4 and 5 still to land)**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md

git commit -m "$(cat <<'EOF'
chore(composer): move Connection Model section to top of pipeline_composer skill

Reduces context distance to the skill's most-violated rule. The Connection
Model section was at line 156, far from the TERMINATION GATE at line 39.
Models that anchor on the early sections were systematically missing the
producer/consumer string-match contract that this section teaches.

Mechanical move only — no content edits.

Refs: docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md §7.2
Filigree: elspeth-1d3be32a8a

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: add Wiring repair examples

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md` — inside the (now-relocated) Connection Model section, append a `#### Wiring repair examples` subsection BEFORE the `#### Boolean routes — quote them` subsection (so the repair examples sit alongside the Common mistakes table conceptually).

Two examples, exactly. Each is a triplet: broken JSON snippet + the verbatim preview error format from `graph.py` + fixed JSON snippet with one-line explanation.

- [ ] **Step 4.1: Confirm the verbatim error string format**

```bash
grep -n "No producer for connection\|Available connections" src/elspeth/core/dag/graph.py | head
```
Expected: `No producer for connection 'X'. Available connections: Y.` Use this exact wording in the examples.

- [ ] **Step 4.2: Insert the new subsection**

Place the new `#### Wiring repair examples` subsection between the existing `#### Common mistakes` subsection (which ends with the table on line ~228) and the `#### Boolean routes — quote them` subsection.

```markdown
#### Wiring repair examples

**Example A — `input` set to a node ID instead of a connection name.**

Broken `set_pipeline` snippet (only the relevant fields shown):

```json
{
  "source": {"plugin": "text", "on_success": "fetch", ...},
  "nodes": [
    {"id": "fetch", "input": "source", ...}
  ]
}
```

`preview_pipeline` returns:

```
No producer for connection 'source'. Available connections: fetch.
```

Why: `node.input` is the **connection-name string the upstream publishes**, not the upstream node's `id`. `source.on_success` publishes connection `fetch`; nothing publishes connection `source`. The runtime resolves wiring by string match, never by walking from `id` to `id`.

Fixed:

```json
{
  "source": {"plugin": "text", "on_success": "fetch", ...},
  "nodes": [
    {"id": "fetch", "input": "fetch", ...}
  ]
}
```

The id `"fetch"` and the connection name `"fetch"` happen to coincide here — that's allowed but not required. Either rename one (`source.on_success: "raw_url_rows"` and `fetch.input: "raw_url_rows"`) or leave them coincident. What matters is that the string on `input` matches a string published by some upstream `on_success`.

**Example B — sink's `sink_name` doesn't match the publishing upstream's `on_success`.**

Broken `set_pipeline` snippet:

```json
{
  "nodes": [
    {"id": "split_lines", "on_success": "lines_out", ...}
  ],
  "outputs": [
    {"sink_name": "output_lines", "plugin": "json", ...}
  ]
}
```

`preview_pipeline` returns:

```
No producer for connection 'output_lines'. Available connections: lines_out.
```

Why: `outputs[i].sink_name` is the consumer-side connection-name string; it must equal an upstream `on_success` value. `split_lines.on_success: "lines_out"` and `outputs[0].sink_name: "output_lines"` are different strings, so the sink has no producer.

Fixed: change one of them to match the other. The simplest fix is to rename the sink:

```json
{
  "outputs": [
    {"sink_name": "lines_out", "plugin": "json", ...}
  ]
}
```

```

- [ ] **Step 4.3: Restart the service to bust the @lru_cache**

```bash
sudo systemctl restart elspeth-web.service
```

- [ ] **Step 4.4: Commit**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md

git commit -m "$(cat <<'EOF'
docs(composer): add wiring repair examples for connection-name failures

Two empirically-observed schema-construction failure shapes from the RGR
investigation now have explicit before/error/after examples in the skill:

Example A: node.input set to an upstream node's id (most common — model
treats the wiring like a normal graph DSL).
Example B: sink_name doesn't match the publishing upstream's on_success
(string-typo class).

Both examples cite the verbatim 'No producer for connection X. Available
connections: Y.' error so the model can correlate its own retry context
with the canonical fix.

No more than two examples per investigation §8.6 — additional skill prose
past iteration 3 has marginal value until platform-side fixes land.

Refs: docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md §7.3
Filigree: elspeth-1d3be32a8a

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: enrich JSON Schema descriptions on connection-name string fields

**Files:**
- Modify: `src/elspeth/web/composer/tools.py` — enrich `description` and add `examples` on the connection-name-string fields across `set_source`, `set_source_from_blob`, `upsert_node`, `set_output`, and the equivalent fields inside `set_pipeline.source`, `set_pipeline.nodes[]`, `set_pipeline.outputs[]`.

**Reality vs tasking:** the tasking listed `upsert_edge` as one of the four tools. `upsert_edge` uses node IDs (`from_node`, `to_node`) and an `edge_type` enum, not connection-name strings. Enriching it with connection-name prose would mislead. Excluded from this task.

Also excluded (per the same logic): `remove_output.sink_name` (line 587), `patch_output_options.sink_name` (line 636) — both refer to *existing* sinks, not fresh wiring decisions. The model isn't picking a connection name there, it's identifying an existing sink.

The fields to enrich:

| Tool | Field | Role | Line (approx) |
|------|-------|------|---------------|
| `set_source` | `on_success` | producer side | 408 |
| `set_source_from_blob` | `on_success` | producer side | 902 |
| `upsert_node` | `input` | consumer side | 441 |
| `upsert_node` | `on_success` | producer side | 442 |
| `set_output` (top-level) | `sink_name` | consumer side | 563 |
| `set_pipeline.source.on_success` | producer side | (inline) | 666 |
| `set_pipeline.nodes[].input` | consumer side | (inline) | 707 |
| `set_pipeline.nodes[].on_success` | producer side | (inline) | 708 |
| `set_pipeline.outputs[].sink_name` | consumer side | (inline) | 745 |

**Three description shapes** (pick the matching one per field role):

```python
# Producer-side ("on_success"): emits this connection name downstream.
"description": (
    "Connection-name string this node PUBLISHES. Some downstream consumer "
    "(node 'input' or output 'sink_name') MUST equal this value for wiring "
    "to resolve. The runtime matches strings, not graph topology — picking "
    "a name unique within the pipeline is fine; the name does not need to "
    "be the downstream node's id."
),
"examples": ["raw_url_rows", "fetched_text", "scored_rows"],

# Consumer-side ("input"): consumes a connection by name.
"description": (
    "Connection-name string this node CONSUMES. MUST equal the value of some "
    "upstream's on_success (or routes value, or on_error) field. Not the "
    "upstream node's id — connections are matched by string, not by topology. "
    "Example: if source.on_success='raw_url_rows', the next node sets "
    "input='raw_url_rows'."
),
"examples": ["raw_url_rows", "fetched_text", "scored_rows"],

# Sink-side ("sink_name"): both an identifier AND the consumed connection name.
"description": (
    "Sink name. This string is BOTH the sink's identifier (used by "
    "patch_output_options/remove_output) AND the connection-name the sink "
    "consumes — it MUST equal some upstream's on_success value. Pick a name "
    "that describes the data being written; the name does not need to match "
    "an upstream node's id."
),
"examples": ["lines_out", "scored_results", "errors_quarantine"],
```

- [ ] **Step 5.1: Apply the producer-side description to `set_source.on_success` (line 408)**

Replace:
```python
                    "on_success": {"type": "string", "description": "Connection name for downstream."},
```
with the producer-side block (multi-line — note JSON Schema allows extra keys, but pydantic models on the API side may strip unknown keys; verify by running an existing tools.py test that round-trips definitions).

- [ ] **Step 5.2: Apply same to `set_source_from_blob.on_success` (line 902)**

Replace existing description; add `examples`.

- [ ] **Step 5.3: Apply consumer-side to `upsert_node.input` (line 441)**

Replace `"Input connection name."` with the consumer-side description and `examples`.

- [ ] **Step 5.4: Apply producer-side to `upsert_node.on_success` (line 442-445)**

Note this field has `"type": ["string", "null"]` and a longer existing description ("Output connection. Required for transform/aggregation/coalesce. Null for gates..."). Preserve the gate/null guidance and APPEND the connection-name guidance:

```python
                    "on_success": {
                        "type": ["string", "null"],
                        "description": (
                            "Output connection. Required for transform/aggregation/coalesce. "
                            "Null for gates (routing is via condition/routes). When set, this "
                            "is the connection-name string the node PUBLISHES — some downstream "
                            "input/sink_name must equal this value."
                        ),
                        "examples": ["fetched_text", "scored_rows", "lines_out"],
                    },
```

- [ ] **Step 5.5: Apply sink-side to `set_output.sink_name` (line 563)**

Replace existing description (`"Sink name (connection point for edges/routes)."`) with the sink-side block.

- [ ] **Step 5.6: Apply same descriptions to the inline-shape fields in `set_pipeline`**

The inline `nodes[]` and `outputs[]` and `source` schemas inside `set_pipeline` (lines 663-756) currently have many bare `{"type": "string"}` entries. Enrich the four connection-string fields:
- `set_pipeline.source.on_success` (line 666) → producer-side
- `set_pipeline.nodes[].input` (line 707) → consumer-side
- `set_pipeline.nodes[].on_success` (line 708) → producer-side
- `set_pipeline.outputs[].sink_name` (line 745) → sink-side

Do NOT modify other inline-shape fields (`id`, `node_type`, `plugin`, `condition`, `routes`, etc.) — those have different roles and are out of scope for the connection-name fix.

- [ ] **Step 5.7: Verify no JSON Schema breakage**

Run: `.venv/bin/python -c "from elspeth.web.composer.tools import get_tool_definitions; import json; defs = get_tool_definitions(); print(json.dumps(defs, indent=2)[:500])"`
Expected: prints valid JSON. No exceptions.

- [ ] **Step 5.8: Run the tools test module**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_tools.py -x -q`
Expected: PASS. Adding `description` and `examples` keys to schema entries should not break tests; if any test asserts on the *exact dict shape* of a tool definition, update the assertion to match the new shape (this is the locked-in-expectation pattern).

- [ ] **Step 5.9: Restart the service**

```bash
sudo systemctl restart elspeth-web.service
```

- [ ] **Step 5.10: Verify the descriptions surface to the live tools list**

```bash
# Log in to staging, then GET the tools-list endpoint (or trigger a session
# and read the LiteLLM tool definitions sent on the request — investigation
# §10 documents the shape).
```

If a `/api/composer/tools` debug endpoint exists, GET it. Otherwise, start a session and inspect the next outbound LLM request's `tools=[...]` payload via the audit DB:

```bash
sqlite3 /home/john/elspeth/data/sessions.db "
  SELECT json_extract(tool_calls, '$.call.tools_spec_hash')
  FROM chat_messages WHERE role='tool' ORDER BY created_at DESC LIMIT 1;
"
```

Hash will differ from before (new descriptions = new bytes); that's the lightweight verification.

- [ ] **Step 5.11: Commit**

```bash
git add src/elspeth/web/composer/tools.py

git commit -m "$(cat <<'EOF'
docs(composer): enrich connection-name field descriptions in LLM tool schemas

The composer's failure mode is connection-name string-mismatch (RGR investigation
2026-05-06 §4.2). Tool schemas previously gave the model a bare {"type": "string"}
on most connection-bearing fields, with no role context (producer vs consumer vs
sink) and no examples. The richer upsert_node tool had a one-line description
("Input connection name.") while the heavier-trafficked set_pipeline inline
shape had no description at all.

Enrich on_success / input / sink_name across set_source, set_source_from_blob,
set_pipeline (inline source + nodes[] + outputs[]), upsert_node, and set_output.
Three description shapes by role; examples=["raw_url_rows", "fetched_text", ...]
on each. Inline duplication is intentional — refactor to $ref is Tier 2 cleanup.

Excluded: upsert_edge (uses node IDs + edge_type, not connection-name strings),
remove_output.sink_name + patch_output_options.sink_name (refer to existing
sinks, not fresh wiring decisions). Adding connection-name guidance to those
fields would mislead the model.

Refs: docs/composer/evidence/composer-passivity-rgr-investigation-2026-05-06.md §7.4
Filigree: elspeth-1d3be32a8a

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Cohort #2 — final cohort (6 RGR runs)

- [ ] **Step C2.1: Confirm all five items have landed in HEAD**

```bash
git log --oneline -7
```
Expected: 5 new commits + the pre-existing baseline + this plan / tasking. Order: temperature → preflight → relocate → examples → schemas (or similar).

- [ ] **Step C2.2: Restart the service (to ensure all skill / schema changes are loaded)**

```bash
sudo systemctl restart elspeth-web.service
```

- [ ] **Step C2.3: Run the final 6-run cohort**

```bash
cd /home/john/elspeth
for i in 1 2 3 4 5 6; do
  evals/composer-rgr/run_scenario.sh "final-${i}"
done
```

- [ ] **Step C2.4: Tally hard-GREEN / soft-RED / hard-RED**

For each run dir under `evals/composer-rgr/runs/*final-*/`, read `scoring.json` and aggregate.

- [ ] **Step C2.5: Verify audit-row persistence one more time**

Same SQL as Cohort #1 step C1.4 but on a fresh `final-*` session. Confirm `temperature=0.0, seed=42` on every llm_call_audit row.

- [ ] **Step C2.6: Comment cohort #2 on the epic**

```
mcp__filigree__add_comment id=elspeth-1d3be32a8a body="Cohort #2 (final, post all Tier 1 items, 6 runs):

final-1: <verdict>
final-2: <verdict>
final-3: <verdict>
final-4: <verdict>
final-5: <verdict>
final-6: <verdict>

Hard-GREEN: <n>/6.
Soft-RED: <n>/6.
Hard-RED: <n>/6.

Audit verification: temperature=0.0, seed=42 confirmed across <n> rows in session <sid>.

Cohort delta vs cohort #1: <hard-GREEN delta>."
```

- [ ] **Step C2.7: Apply tasking §4 exit-criteria gate**

- If hard-GREEN ≥ 4/6 → close the epic per Step Final.1.
- If hard-GREEN < 4/6 → file a Tier 2 observation (`mcp__filigree__observe`) referencing the cohort numbers; comment on the epic; LEAVE EPIC OPEN; stop.

---

## Final.1 — close the epic (only if cohort #2 ≥ 4/6 hard-GREEN)

- [ ] **Step Final.1.1: Per-item documentation comments on the epic**

Five comments, one per item, each citing file:line of what changed and the diff line count:

```
mcp__filigree__add_comment id=elspeth-1d3be32a8a body="Item 1 (temperature=0.0, seed=42 + audit fields): commit <sha>. Files: src/elspeth/contracts/composer_llm_audit.py:32-72 (+2 fields), src/elspeth/web/composer/service.py (+module constants, +6 kwargs); 3 test files updated. Diff: ~<N> lines."
```

(...repeat for items 2-5).

- [ ] **Step Final.1.2: Final summary comment with cohort table**

(See template in Cohort #2 step C2.6.)

- [ ] **Step Final.1.3: Push branch + open PR**

Per project policy, the user has not authorized auto-push. Stop here; surface the local branch state and let the operator decide on push and PR shape (single squashed PR with 5 commits, or 5 separate PRs).

```bash
git log --oneline RC5-UX..HEAD
```

Then ASK the operator: "5 commits ready locally on `RC5-UX`. Push as a single PR with 5 commits, or 5 separate PRs? Default: single squashed PR per tasking §4.1."

- [ ] **Step Final.1.4: Close the epic only after PR is merged**

```
mcp__filigree__close_issue id=elspeth-1d3be32a8a reason="Tier 1 reliability remediation landed — 5 commits, cohort #2 hard-GREEN <n>/6 (≥4/6 exit criterion met). Tier 2 candidates filed as observations: <links>."
```

---

## Self-review

**Spec coverage (tasking §6 five items):**
- Item 1 → Tasks 1 (3 sub-files: dataclass, service, tests).
- Item 2 → Task 2.
- Item 3 → Task 3.
- Item 4 → Task 4.
- Item 5 → Task 5 (with the upsert_edge exclusion documented).
- Cohort measurements → Cohort #1 + Cohort #2.
- Audit-sidecar verification (tasking §4.5) → Step C1.4 and C2.5.
- No-skill-prose-tests rule (tasking §4.6) → no test added against `pipeline_composer.md` content.
- Per-item epic comments (tasking §4.7) → Step Final.1.1.

**Placeholder scan:**
- No "TBD", "implement later", "fill in details".
- All test code is shown verbatim where added.
- Exact line numbers are given as "around line X" because the investigation note's anchors drifted ~25 lines from the snapshot; agents must re-grep before editing. This is honest pointing, not a placeholder.

**Type consistency:**
- `temperature: float` and `seed: int` are required across L0 dataclass, builder kwargs, and test fixtures.
- `_COMPOSER_LLM_TEMPERATURE: Final[float]` and `_COMPOSER_LLM_SEED: Final[int]` are typed at the constants definition.
- `_build_llm_call_record` signature: `temperature: float, seed: int` (kwargs, no defaults — matches dataclass requirement).

**Reality reconciliation captured:**
- Three corrections vs the tasking are listed up top and re-cited in each affected task.
- Memory entries surfaced: `project_composer_harness_state.md` (skill `@lru_cache`), `feedback_no_tests_for_skill_prompts.md`, `project_staging_deployment.md`, `feedback_locked_in_buggy_expectations.md` (test updates after structural fix).

---

## Out-of-scope reminders (tasking §10)

- No Tier 2 work (strict JSON Schema mode, runtime preflight rewrite, in-loop retry hint, mutation-result wiring echo, set_linear_pipeline, derive-input-from-edges).
- No cross-model RGR.
- No per-failure-mode RGR scenarios.
- No multi-turn RGR sweep.
- No raw_assistant_content next-turn exposure.

If the work overlaps with any of these (e.g., the schema-enrichment in Task 5 touches the same file as a future strict-mode change), file an observation and proceed.

---

## Execution choice

Two execution options:

**1. Inline Execution (recommended for this plan)** — Six commits, all small, all in one branch. Use `superpowers:executing-plans` with checkpoints between Cohort #1 and Cohort #2 (the cohorts are blocking measurement gates, not commits). Inline keeps context coherent across the per-item commits and the locked-in-expectation test updates.

**2. Subagent-Driven** — Dispatch a fresh subagent per Task. Less ideal here because the cohort measurements need the whole-session context (which audit row to query, which session_id to cite) and subagents would re-read setup repeatedly.

Default: Inline.
