# Handover — Composer skill remediation, 2026-05-25

You are continuing a session whose explicit task was: take the composer's
pipeline-building skill from its current pass rate to as close to 100% on
`scripts/staging-tutorial-harness.mjs` as possible, **without hardcoding
results or handholding the prompt via the skill**. Branch: RC5.2. Staging at
`elspeth.foundryside.dev` runs from this very checkout (`/home/john/elspeth`)
as `elspeth-web.service` — every skill/server change requires a service
restart.

The composer model is `openrouter/openai/gpt-5.4`. OpenRouter creds live in
`.env`. Operator credentials for the harness: `STAGING_USERNAME=dta_user`,
`STAGING_PASSWORD=dta_pass`.

## What the baseline showed (3-run, pre-fix)

```
run 1 ok=True  kinds=[invented_source, llm_prompt_template, pipeline_decision×2]
run 2 ok=False kinds=[invented_source, llm_prompt_template, pipeline_decision]   shield missing
run 3 ok=False kinds=[invented_source, llm_prompt_template, pipeline_decision]   shield missing
```

The dominant failure mode (2/3) was the LLM building a structurally-correct
pipeline (CSV source, web_scrape, LLM, cleanup field_mapper, sink), staging
three of the four reviews, and confidently stopping with prose like "Pipeline
is built and waiting on review cards." **It just forgot to also stage the
`prompt_injection_shield_recommendation` pipeline_decision review on the LLM
node.** The skill has ~30 lines telling it to do this — the prose isn't
landing on this model.

A separate, operator-witnessed 500 error (`accepted_value must not contain
control characters`) was also diagnosed: the validator at
`src/elspeth/web/validation.py:_validate_accepted_value_content` rejected
**all** newlines, but `invented_source` (multi-line CSV),
`llm_prompt_template` (multi-line Jinja), and `pipeline_decision` (multi-line
explanations) drafts are all legitimately multi-line. This blocked the
UI Accept-via-amend flow and could intermittently bite the LLM's tool
boundary for `VAGUE_TERM`/`PIPELINE_DECISION` drafts.

## What I changed (all committed)

The shape of the fix follows the operator's "no handholding via skill"
constraint by moving the load-bearing rules from prompt prose to
**server-side enforcement**:

### 1. `src/elspeth/web/validation.py`
Relaxed `_validate_accepted_value_content` to permit `\t \n \r`, blocking
the remaining ASCII control chars (NUL, BEL, VT, FF, ESC, US, DEL). Per-line
length cap (1024 chars) preserved and now applied per `splitlines()` line,
catching multi-line payloads with one pathological line. Updated docstring
to remove the obsolete "single-phrase" assertion (which was true when only
`vague_term` existed in v1, but became wrong when `pipeline_decision`
landed). Test updates in `tests/unit/web/sessions/test_interpretation_schemas.py`:
`test_amended_value_rejects_newline` → `test_amended_value_permits_newline`,
added `test_amended_value_permits_carriage_return` and
`test_amended_value_per_line_length_cap_multi_line`.

### 2. `src/elspeth/web/interpretation_state.py`
Added:
- `PROMPT_SHIELD_USER_TERM = "prompt_injection_shield_recommendation"`
- `PROMPT_SHIELD_REVIEW_DRAFT` — canonical recommendation prose.
- `_UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS = {"web_scrape"}`,
  `_AUTHORIZED_PROMPT_SHIELD_PLUGINS = {"azure_prompt_shield"}` (note:
  `azure_content_safety` is **deliberately not** in the authorized set —
  content moderation ≠ prompt-injection shielding).
- `_producer_by_output_stream`, `_llm_consumes_untrusted_remote_content`
  (upstream graph walker), `_llm_has_shield_recommendation`,
  `_missing_prompt_shield_recommendation_review_sites` (mirroring
  `_missing_raw_html_cleanup_review_sites`).
- `prompt_shield_recommendation_contract_error(state) -> str | None` —
  **HARD-REJECTS** `set_pipeline` if an LLM ingests `web_scrape` output
  without a `azure_prompt_shield` between them AND no
  `pipeline_decision` review with the expected `user_term` is staged on
  that LLM node. The error message contains the exact `user_term` and
  draft the LLM must stage, and explicitly forbids adding a shield or
  `content_safety` node to the graph.
- Extended `interpretation_sites()` to also enumerate
  missing-shield-recommendation sites (so the repair-message-on-stop path
  also surfaces them for belt-and-suspenders).

### 3. `src/elspeth/web/composer/tools/sessions.py`
Added a call to `prompt_shield_recommendation_contract_error(new_state)`
after the existing `raw_html_cleanup_review_contract_error` call. Imports
both functions together.

### 4. `src/elspeth/web/composer/service.py`
Extended `_pending_interpretation_review_repair_message` with a special-case
for the prompt-shield `user_term`, parallel to the existing
`RAW_HTML_CLEANUP_USER_TERM` block — instructs the LLM to patch the LLM
node first with the canonical draft and explicitly forbids adding shield
nodes for this requirement.

### 5. Service restarted at end-of-turn so the new contract is live on
`elspeth.foundryside.dev`.

## Verification status — POST-FIX 3-RUN: 3/3 PASS

```
passed=3 failed=0 runs=3
run 1 ok=True kinds=[invented_source, llm_prompt_template, pipeline_decision×2]  cleanup=True shield=True diagnostics=[]
run 2 ok=True kinds=[invented_source, llm_prompt_template, pipeline_decision×2]  cleanup=True shield=True diagnostics=[]
run 3 ok=True kinds=[invented_source, llm_prompt_template, pipeline_decision×2]  cleanup=True shield=True diagnostics=[]
```

Every run now stages all four reviews — the pre-fix flakiness on
`prompt_injection_shield_recommendation` is gone because the LLM can no
longer submit a `set_pipeline` that violates the contract; the server-side
rejection forces the requirement to be staged.

When you resume, lock the result in with a full 20-run sweep:

```bash
cd /home/john/elspeth/src/elspeth/web/frontend
STAGING_USERNAME=dta_user STAGING_PASSWORD=dta_pass TUTORIAL_RUNS=20 \
  npm run staging:tutorial-harness
```

If any run fails, read the specific failing run's `last_assistant_excerpt`
and pipeline state — most likely the LLM either added a `content_safety`
node (which the harness rejects via `content_safety_inserted`) or chose an
unknown untrusted-content producer plugin the upstream walker doesn't
classify. The classifier sets are
`_UNTRUSTED_REMOTE_CONTENT_PRODUCER_PLUGINS` and
`_AUTHORIZED_PROMPT_SHIELD_PLUGINS` in `web/interpretation_state.py`.

## Things I did NOT do that are reasonable next moves

### A. Trim the now-redundant skill section
The skill at `src/elspeth/web/composer/skills/pipeline_composer.md`
(currently 757 lines) has an "Internet content flowing into LLMs" section
(~30 lines) that the server contract now enforces mechanically. Per the
operator's "no handholding via skill" direction and the advisor's "be
willing to *delete* prescriptions that fail to land their intent" note,
that section can be trimmed to: principle only (why prompt-injection matters,
that shields are a separate control from content moderation), no exact
`user_term` strings, no JSON construction snippets. Same goes for the
"Raw Scraped-Content Cleanup" section now that
`raw_html_cleanup_review_contract_error` enforces it. Restart the service
after each trim.

**Don't trim unless the verification run confirms the contract enforces the
behavior** — the skill is the safety net if the contract has gaps.

### B. INVENTED_SOURCE consolidation (operator-offered, separate body of work)
The operator explicitly asked: "If there's a technical improvement we can
make there with INVENTED_SOURCE (i.e. bringing it into line) let me know
- we can do that as a separate body of work."

The asymmetry I found:
- `_validate_accepted_value_content` lives in `web/validation.py` and is
  the shared validator used at the schema layer (`/resolve`) and the
  composer tool boundary for `VAGUE_TERM` and `PIPELINE_DECISION` drafts.
- `_validate_source_artifact_review_content` lives in
  `src/elspeth/web/composer/tools/sessions.py:150` and is the
  tool-boundary-only validator for `INVENTED_SOURCE` drafts.
- After my validator fix, the two are nearly identical: same control-char
  policy (allow `\t \n \r`), same per-line cap, same template-metachars
  and credential rejection. They differ only in module location and
  message wording.
- `src/elspeth/web/sessions/service.py:2975-2982` artificially blocks
  `AMENDED` for `INVENTED_SOURCE` ("`{kind.value} does not support inline
  amendment in this release`"). That block was almost certainly there
  because the schema-layer `_validate_accepted_value_content` rejected
  newlines — and a CSV without newlines is unusable. **With the validator
  fixed, the block can be lifted**, letting operators edit a generated
  CSV (add a row, fix a URL) before accepting.

Proposed scope:
1. Move `_validate_source_artifact_review_content` into
   `web/validation.py` next to `_validate_accepted_value_content`.
2. Either delete it as redundant, or keep both with a `# Alias of`
   docstring and update callers.
3. Lift the AMENDED block for `INVENTED_SOURCE` in
   `service.py:2975`; the schema-layer validator now handles multi-line.
4. Unit-test the AMENDED-for-INVENTED_SOURCE path end-to-end.

### C. Pre-existing unrelated test failures (NOT mine to fix in this scope)
Two tests fail on RC5.2 head independent of my edits:

1. `tests/unit/web/composer/test_advisor_tool.py::test_skill_advisor_examples_include_required_trigger_values`
   — greps the skill for literal string `"trigger:"`. The v2 skill rewrite
   uses "Valid triggers include `reactive_validation_loop`...", which doesn't
   contain `"trigger:"`. Per memory `feedback_no_tests_for_skill_prompts.md`
   this kind of grep-the-skill test is theatre. Either rewrite to assert the
   *behavior* (a `request_advisor_hint` call with that trigger reaches the
   tool), or delete.
2. `tests/unit/web/composer/test_adequacy_guard.py::test_per_entry_shape_walk_yields_only_sensitive_or_scalar_fields`
   — fails on `GetBlobContentResponseModel.validation: Any`. Real schema
   discipline gap, unrelated to this remediation. Either add `Sensitive()`
   marker to that field or replace with a closed-list scalar.

## Constraints to honor on resume
- **Skill is `@lru_cache`'d at module import** — every edit to
  `pipeline_composer.md` needs `sudo systemctl restart elspeth-web.service`
  to take effect on staging.
- **Default to worktree — but skip for skill/config edits** (memories
  `feedback_default_to_worktree.md`, `feedback_skip_worktree_for_skill_and_config_edits.md`).
  Skill prompt and runtime config edits happen in place on RC5.2 so the
  live service sees them without symlink gymnastics.
- **`git stash` is hard-blocked** at the operator level. Commit WIP or
  use a worktree instead.
- **No tests for skill-prompt content** (memory `feedback_no_tests_for_skill_prompts.md`).
- **OpenRouter spend is limited** — small-N (3-run) before any full 20-run.
- **Auto Mode is on** — don't pause for clarifying questions on
  reversible work; do pause and surface for destructive shared-state
  changes (DB delete, force-push, etc.).
