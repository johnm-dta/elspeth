# Backend Per-Phase LLM Drivers + /guided/chat Apply Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Turn the per-phase guided chat box from advisory-only into a per-phase **driver** that proposes *and applies* config through the same `handle_step_*` commit seams the manual form uses, applying in place (no auto-advance) so the learner can revise the current phase by typing again.

**Architecture:** `POST /guided/chat` already has one working "drive this phase" branch for STEP_1 (resolve a source, commit via `handle_step_1_source`). This plan generalizes that pattern to the sink and transform phases — adding a sink driver and reusing the chain solver — and changes the contract from auto-advance to **apply-in-place**: a successful chat apply writes `step_N_result`, leaves `guided.step` unchanged, and re-renders the *current* phase's editable form. Advancing stays an explicit confirm on `/guided/respond`. This is the first plan in the LLM-primary series; **p2 (frontend)** consumes this plan's apply/response contract, and **p4 (tutorial)** consumes the source-driver output shape. p1 and **p3 (prompt-shield)** can start independently.

**Tech Stack:** Python 3.13, FastAPI, LiteLLM (stubbed at `_litellm_acompletion` in tests), pytest (`pytest.mark.asyncio`), SQLite session store, frozen dataclasses (`SourceResolved` / `SinkResolved` / `ChainProposal`).

## Global Constraints

- All work lands on `release/0.7.0` (the named release branch), NOT a feature
  branch. Verify `git branch --show-current` before committing; feature branches
  get orphaned.
- The agent SIGNS NOTHING. The operator holds the HMAC key and pushes per the
  release-train process. Do not proactively re-sign tier-model fingerprints or
  plugin hashes; surface owed re-signs as an operator chore.
- Editing a plugin file (e.g. `src/elspeth/plugins/transforms/llm/transform.py`)
  trips TWO CI gates, both operator-owed re-sign chores: (a) the plugin
  `source_file_hash` gate (`plugin-contract-plugin-hashes`) — refresh via
  `scripts/cicd/plugin_hash.py` (`compute_source_file_hash`/`fix_source_file_hash`);
  (b) the tier-model fingerprint cascade (`trust-tier-model`; adding imports
  shifts `Module.body` indices) — allowlists `config/cicd/enforce_tier_model/plugins.yaml`
  (plugin files) and `.../web.yaml` (web files: interpretation_state.py, state.py),
  rotated via `elspeth_lints.rules.trust_tier.tier_model.rotate`
  (scripts/cicd/rotate_tier_model_fingerprints.py). Co-land the fingerprint/hash
  updates with the source change; the operator re-signs.
- The canonical tutorial prompt couples FOUR things in lockstep: the backend
  constant `CANONICAL_SEED_PROMPT` (`web/preferences/tutorial_cache.py`), its
  byte-identical FRONTEND MIRROR `CANONICAL_TUTORIAL_PROMPT`
  (`frontend/src/components/tutorial/tutorialMachine.ts`, byte-identity enforced
  by `test_canonical_seed_matches_frontend_constant`), the `composer_skill_hash`
  re-bake (`PIPELINE_COMPOSER_SKILL_HASH` in `composer/prompts.py` +
  `assert_skill_hash_unchanged_on_disk`) when the live `pipeline_composer.md`
  skill changes, AND a live-prompt SERVICE RESTART. Editing the prompt constant
  alone needs the mirror + the two value-assert tests (NO restart). Editing the
  live skill/recipe needs the re-baked hash + restart (the 5-input
  `tutorial_model_id` invalidates the cache). Do not conflate the two.
- Prompt-shield reviews and advisor/checkpoint reviews are ADVISORY and NEVER
  hard-block: emitted into `validate()` `warnings` at "medium", excluded from the
  blocking contract. Do not promote them to errors.
- `entry_seed` (the tutorial framing/dataset seed) is SERVER-SIDE ONLY and never
  rides the wire: it is redacted from `WorkflowProfileResponse`. Do not add it to
  any wire/GET shape, and do not infer "tutorial" from the wire profile booleans
  (use the client-only `isTutorial` React prop).
- Existing tests that assert about-to-change behavior must be UPDATED to the new
  behavior, NOT reverted. A wave of failures after a structural change is the
  change landing visibly; update the assertions, do not roll back the change.
- For code commits use `git commit` with
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier` (the
  operator-owed-re-sign gates) — NEVER a blanket `--no-verify`. Doc-only commits
  may use `--no-verify`. Reconcile the full slice diff at the slice boundary.
- NEVER `git add -A` / `git add .`; stage only the files this plan owns.

---

## Orientation: the seams this plan reuses (read before Task 1)

These are verified anchors on `release/0.7.0` (HEAD `7d7bfaffd`). Every task below
cites them; do not re-discover them.

- **Commit seams** (`src/elspeth/web/composer/guided/steps.py`), each returns
  `StepHandlerResult(state, session, tool_result)`; on `tool_result.success is
  False` returns state+session UNCHANGED:
  - `handle_step_1_source(*, state, session, resolved: SourceResolved, catalog,
    data_dir=None, session_engine=None, session_id=None)` — def at `:65`. Commits
    source; sets `session.step_1_result`. **Does NOT advance the step pointer.**
  - `handle_step_2_sink(*, state, session, resolved: SinkResolved, catalog,
    data_dir=None)` — def at `:147`. Commits each output via `_execute_set_output`;
    sets `session.step_2_result`. **Does NOT advance the step pointer.** Raises
    `InvariantError` on empty `resolved.outputs` (`:176`).
  - `handle_step_3_chain_accept(...)` — def at `:275`. Commits the full pipeline
    and **ADVANCES to `STEP_4_WIRE`** (`:388-393`). The chat driver does NOT call
    this (see Task 4); accept stays on `/guided/respond`.
  - `handle_step_2_5_recipe_apply(...)` — def at `:219`. Also **advances to
    `STEP_4_WIRE`**. No chat driver for recipe-apply (deterministic confirm).
- **Drivers / solvers:**
  - `maybe_resolve_step_1_source_chat(*, model, user_message, plugin_hint,
    temperature, seed, recorder=None) -> Step1SourceChatResolution | None`
    (`chat_solver.py:233`). Returns `None` on prose/no-tool-call.
  - `solve_chain(*, model, source, sink, recipe_match=None, repair_context=None,
    recorder=None, temperature, seed) -> ChainProposal` (`chain_solver.py:130`).
  - `solve_step_chat(...) -> str` (`chat_solver.py:346`) — advisory prose.
- **Auto-drop wrappers** (`src/elspeth/web/sessions/_guided_step_chat.py`):
  - `resolve_step_1_source_chat_with_auto_drop(*, site, session_id, user_id,
    model, user_message, plugin_hint, temperature, seed, recorder=None) ->
    Step1SourceChatResult` — `.source_resolution` (the resolution or `None`) +
    `.fallback_chat` (a `StepChatResult` on transient failure). Def at `:109`.
  - `solve_step_chat_with_auto_drop(...) -> StepChatResult` — def at `:182`;
    absorbs transient LLM failures into `_SYNTHETIC_UNAVAILABLE_MESSAGE`; does NOT
    absorb `InvariantError`/`ValueError`.
  - `StepChatResult` (`:39`): `assistant_message`, `status`, `latency_ms`,
    `error_class`.
- **In-place re-render turn builders** (`src/elspeth/web/composer/guided/emitters.py`):
  - `build_step_1_schema_form_turn(plugin, catalog, *, inspection_facts=None)` —
    `:120`. The populated STEP_1 source form (step stays STEP_1).
  - `build_step_2_schema_form_turn(plugin, catalog)` — `:209`. The STEP_2 sink
    form (step stays STEP_2).
  - `build_step_3_propose_chain_turn(proposal: ChainProposal)` — `:312`. The
    transform proposal (step stays STEP_3; NOT committed).
- **Route** `post_guided_chat` (`src/elspeth/web/sessions/routes/composer/guided.py:1637`):
  guards at unknown-step 400 (`:1683-1689`), terminal 409 (`:1722-1726`),
  step-mismatch 409 (`:1733-1740`), no-guided 400 (`:1714-1718`). The existing
  STEP_1 apply branch is `:1754-1992` (note `_replace(guided,
  step=GuidedStep.STEP_2_SINK)` at `:1863` — the auto-advance Task 3 removes).
  Advisory fall-through is `:2005-2172` (`next_turn=None` at `:2169`).
- **Resolved types** (`src/elspeth/web/composer/guided/resolved.py`):
  `SourceResolved` (`:19`: `plugin`, `options`, `observed_columns`,
  `sample_rows`); `SinkResolved(outputs: Sequence[SinkOutputResolved])` (`:89`);
  `SinkOutputResolved` (`:54`: `plugin`, `options`, `required_fields`,
  `schema_mode`).
- **Scrape routing facts** (no `web_scraper` source plugin exists):
  `_web_scrape_predicate` (`recipe_match.py:230-263`, format-blind) fires when the
  source is an inline `json`/`csv` URL-row source (`blob_ref` in options +
  observed `url` column) and the sink is a single json output; the
  `web-scrape-llm-rate-jsonl` recipe (`recipes.py:_build_web_scrape_recipe`,
  `:648-769`) inserts the `web_scrape` TRANSFORM. `match_recipe` (`:489`) fires at
  STEP_2.5. The source driver produces ONLY the inline URL-row source; it does NOT
  propose a scraper source.

**Apply rule (the decided contract — applies to every driver in this plan):**
Every `POST /guided/chat` submit ATTEMPTS to drive the current phase via the
phase driver → `handle_step_*` (or, for STEP_3, re-render the proposal). It
mutates ONLY when the driver produced an actionable config the strict
`handle_step_*` + `validate_pipeline` seams accept. Non-actionable input (a
question, ambiguous text, prose-only reply, or any LLM failure/timeout/malformed
output) falls back to advisory prose with NO mutation (`next_turn=None`, appends
`chat_history` only, never wizard `history`). Apply is **in-place**: write
`step_N_result`, leave `guided.step` unchanged, re-render the current phase form.
The 400/409 guards above are preserved verbatim on the apply path.

**No schema/epoch bump in this plan.** Apply-in-place + revise write only fields
already serialized in `GuidedSession.to_dict`/`from_dict` (`state_machine.py:364-405`):
`step_1_result`, `step_2_result`, `step_3_proposal`, `step`, `chat_history`,
`chat_turn_seq`. Do NOT add a new "proposed-but-not-advanced" staging field; do
NOT bump `GUIDED_SESSION_SCHEMA_VERSION` (=6) or `SESSION_SCHEMA_EPOCH` (=24).

---

### Task 1: Source driver — accept the current applied source for in-place revise

**Files:**
- Modify: `src/elspeth/web/composer/guided/chat_solver.py`
  (`maybe_resolve_step_1_source_chat`, def `:233`; `_build_step_1_source_tool_prompt`, `:134`)
- Test: `tests/integration/web/composer/guided/test_step_chat_source_driver.py` (Create)

**Interfaces:**
- Consumes: `maybe_resolve_step_1_source_chat(*, model, user_message, plugin_hint,
  temperature, seed, recorder=None) -> Step1SourceChatResolution | None`
  (current signature, `chat_solver.py:233`); `_build_step_1_source_tool_prompt(*,
  plugin_hint)` (`:134`); `Step1SourceChatResolution` (`:37`).
- Produces (later tasks/p4 rely on): `maybe_resolve_step_1_source_chat` gains a
  keyword `current_source: SourceResolved | None = None` (default `None` keeps the
  STEP_1 cold-start call at `_guided_step_chat.py:134` source-compatible until
  Task 3 threads it). When `current_source` is set, the tool prompt includes the
  current applied source so a revise instruction ("add a url column", "make it
  csv not json") resolves relative to it. The applied output stays an inline
  URL-row `SourceResolved` (`plugin in {"json","csv"}`, observed `url` column,
  `blob_ref` materialized downstream) — the p1→p4 source-driver output contract.

The driver still returns `None` on a prose/no-tool-call reply (advisory
fall-back); only the prompt and signature change here, not the parse boundary.

- [ ] **Step 1: Write the failing test that the driver accepts `current_source`
  and threads it into the prompt.**
  Create `tests/integration/web/composer/guided/test_step_chat_source_driver.py`:

  ```python
  """p1 Task 1 — source driver accepts current applied source for in-place revise."""

  from __future__ import annotations

  import json
  from types import SimpleNamespace
  from unittest.mock import AsyncMock, patch

  import pytest

  from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
  from elspeth.web.composer.guided.resolved import SourceResolved


  def _fake_resolve_source_response(args: dict) -> SimpleNamespace:
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
                      content=None,
                      tool_calls=[
                          SimpleNamespace(
                              function=SimpleNamespace(
                                  name="resolve_source",
                                  arguments=json.dumps(args),
                              )
                          )
                      ],
                  )
              )
          ]
      )


  @pytest.mark.asyncio
  async def test_source_driver_includes_current_source_in_prompt() -> None:
      current = SourceResolved(
          plugin="json",
          options={"schema": {"mode": "observed"}, "blob_ref": "abc"},
          observed_columns=("url",),
          sample_rows=({"url": "https://example.test/a"},),
      )
      captured: dict = {}

      async def _capture(**kwargs):
          captured.update(kwargs)
          return _fake_resolve_source_response(
              {
                  "resolution": "source",
                  "plugin": "json",
                  "filename": "urls.json",
                  "mime_type": "application/json",
                  "content": '[{"url": "https://example.test/a"}]',
                  "options": {"schema": {"mode": "observed"}},
                  "observed_columns": ["url"],
                  "sample_rows": [{"url": "https://example.test/a"}],
                  "assistant_message": "Updated the URL list.",
              }
          )

      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(side_effect=_capture),
      ):
          result = await maybe_resolve_step_1_source_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="add a second url",
              plugin_hint="json",
              current_source=current,
              temperature=None,
              seed=None,
          )

      assert result is not None
      assert result.plugin == "json"
      system_prompt = captured["messages"][0]["content"]
      # The current applied source MUST be threaded so "add" resolves relative to it.
      assert "https://example.test/a" in system_prompt
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_source_driver.py -x -q
  ```
  Expected failure: `TypeError: maybe_resolve_step_1_source_chat() got an unexpected keyword argument 'current_source'`.

- [ ] **Step 2: Add the `current_source` keyword + thread it into the prompt.**
  In `chat_solver.py`, change `_build_step_1_source_tool_prompt` (`:134`) to accept
  the current source and append it:

  ```python
  def _build_step_1_source_tool_prompt(
      *,
      plugin_hint: str | None,
      current_source: SourceResolved | None = None,
  ) -> str:
      """Compose the Step-1 source/data-schema tool prompt."""
      hint = (
          f"The current source plugin selected in the wizard is {plugin_hint!r}."
          if plugin_hint is not None
          else "The current source plugin is not persisted in server state; infer only when the chat message or tool context makes it explicit."
      )
      revise_block = ""
      if current_source is not None:
          revise_block = (
              "\n## Current applied source (revise relative to this)\n\n"
              "A source has already been applied to this phase. The user's message "
              "is a REVISION instruction against it — re-emit the COMPLETE updated "
              "source (not a diff). Current source:\n"
              f"{json.dumps(current_source.to_dict(), sort_keys=True)}\n"
          )
      return (
          f"{load_step_chat_skill(GuidedStep.STEP_1_SOURCE).rstrip()}\n\n"
          "## Step 1 Source/Data Schema Tool\n\n"
          f"{hint}\n"
          f"{revise_block}"
          "If the user's message provides enough information to create inline source data, "
          "call `resolve_source` with the complete file content, the source plugin, "
          "schema options, observed columns, representative sample rows, and a brief "
          "assistant_message. For CSV data, include a header row in `content` and set "
          "`mime_type` to `text/csv`. Preserve user-supplied values exactly in the file "
          "content; do not invent hidden pipeline transforms. If the message is only a "
          "question or lacks enough source detail, reply in prose and do not call a tool.\n"
      )
  ```

  Add the import at the top of `chat_solver.py` (the resolved types live in
  `resolved.py`):
  ```python
  from elspeth.web.composer.guided.resolved import SourceResolved
  ```
  Then add the keyword to `maybe_resolve_step_1_source_chat` (`:233`) and pass it
  through. Change the signature line and the prompt call site:
  ```python
  async def maybe_resolve_step_1_source_chat(
      *,
      model: str,
      user_message: str,
      plugin_hint: str | None,
      current_source: SourceResolved | None = None,
      temperature: float | None,
      seed: int | None,
      recorder: BufferingRecorder | None = None,
  ) -> Step1SourceChatResolution | None:
  ```
  and the message build (`:256`):
  ```python
      messages: list[dict[str, Any]] = [
          {"role": "system", "content": _build_step_1_source_tool_prompt(plugin_hint=plugin_hint, current_source=current_source)},
          {"role": "user", "content": user_message},
      ]
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_source_driver.py -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Add the failing test that prose-only still returns `None`
  (advisory fall-back preserved).**
  Append to `test_step_chat_source_driver.py`:

  ```python
  @pytest.mark.asyncio
  async def test_source_driver_returns_none_on_prose() -> None:
      prose = SimpleNamespace(
          choices=[SimpleNamespace(message=SimpleNamespace(content="Here is some advice.", tool_calls=None))]
      )
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=prose),
      ):
          result = await maybe_resolve_step_1_source_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="what is a source?",
              plugin_hint="json",
              current_source=None,
              temperature=None,
              seed=None,
          )
      assert result is None
  ```

  Run to pass (the impl already returns `None` on no-tool-call — this pins it):
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_source_driver.py -x -q
  ```
  Expected: `2 passed`.

- [ ] **Step 4: Thread `current_source` through the auto-drop wrapper.**
  `resolve_step_1_source_chat_with_auto_drop` (`_guided_step_chat.py:109`) calls
  `maybe_resolve_step_1_source_chat` (`:134`). Add the keyword to the wrapper
  signature (after `plugin_hint`):
  ```python
      plugin_hint: str | None,
      current_source: SourceResolved | None = None,
  ```
  and pass it through at the call (`:134`):
  ```python
          source_resolution = await maybe_resolve_step_1_source_chat(
              model=model,
              user_message=user_message,
              plugin_hint=plugin_hint,
              current_source=current_source,
              temperature=temperature,
              seed=seed,
              recorder=recorder,
          )
  ```
  Add the import to `_guided_step_chat.py` if not present:
  ```python
  from elspeth.web.composer.guided.resolved import SourceResolved
  ```
  (Confirm with `grep -n "import SourceResolved" src/elspeth/web/sessions/_guided_step_chat.py`;
  add only if absent.)

  Run the full source-driver test + the existing step-chat suite (cold-start
  callers pass no `current_source`, so the default keeps them green):
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_source_driver.py tests/integration/web/composer/guided/test_step_chat.py -q
  ```
  Expected: all pass EXCEPT `test_step_1_chat_can_commit_generated_csv_source_and_emit_step_2`
  — which is the auto-advance test Task 3 updates. Note it; do not fix it here.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/composer/guided/chat_solver.py src/elspeth/web/sessions/_guided_step_chat.py tests/integration/web/composer/guided/test_step_chat_source_driver.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): source driver accepts current source for in-place revise

maybe_resolve_step_1_source_chat gains a current_source keyword (default
None) and threads the current applied source into the resolve_source tool
prompt so a revision instruction resolves relative to it. The auto-drop
wrapper forwards the keyword. Cold-start callers are unaffected by the
default. Prose-only replies still return None (advisory fall-back).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 2: Sink driver — free text → SinkResolved, applied via handle_step_2_sink

**Files:**
- Modify: `src/elspeth/web/composer/guided/chat_solver.py` (add the sink tool +
  parser + driver, after the STEP_1 source driver block ends at `:343`)
- Modify: `src/elspeth/web/sessions/_guided_step_chat.py` (add a sink auto-drop
  wrapper mirroring `resolve_step_1_source_chat_with_auto_drop`, `:109`)
- Test: `tests/integration/web/composer/guided/test_step_chat_sink_driver.py` (Create)

**Interfaces:**
- Consumes: `SinkResolved` / `SinkOutputResolved` (`resolved.py:89/:54`);
  `_litellm_acompletion` (`composer/service.py`); `BufferingRecorder`;
  `ComposerLLMCallStatus`; the existing `_record_llm_call` helper
  (`chat_solver.py:97`); `StepChatResult` / `_SYNTHETIC_UNAVAILABLE_MESSAGE` /
  `_safe_frame_strings` from `_guided_step_chat.py`.
- Produces (Task 4 consumes): `maybe_resolve_step_2_sink_chat(*, model: str,
  user_message: str, current_sink: SinkResolved | None, temperature: float | None,
  seed: int | None, recorder: BufferingRecorder | None = None) -> tuple[SinkResolved,
  str] | None` — returns `(sink, assistant_message)` on a tool call, `None` on
  prose/no-tool-call. The `assistant_message` is surfaced so STEP_2 chat shows the
  LLM's reply (parity with STEP_1, which surfaces
  `source_resolution.assistant_message`). Plus the wrapper
  `resolve_step_2_sink_chat_with_auto_drop(*, site, session_id, user_id, model,
  user_message, current_sink, temperature, seed, recorder=None) ->
  Step2SinkChatResult` where `Step2SinkChatResult` carries `sink_resolution:
  SinkResolved | None`, `assistant_message: str | None`, and `fallback_chat:
  StepChatResult | None`.

> **Contract supersession (signature).** The shared contract §2.2 lists the sink
> driver as `-> SinkResolved | None` (a stub recommendation). THIS plan body's
> `-> tuple[SinkResolved, str] | None` is the authoritative, more detailed
> signature — it carries the `assistant_message` STEP_2 chat must surface for
> STEP_1 parity. Implementors: code the tuple form. The contract entry is a
> superseded stub; updating it to match is an owner-owed edit to the shared
> contract file (tracked in the review's deferred list — not changed by this
> single-plan revision, since four plans cite that file).
  Create `tests/integration/web/composer/guided/test_step_chat_sink_driver.py`:

  ```python
  """p1 Task 2 — sink driver resolves free text into a SinkResolved."""

  from __future__ import annotations

  import json
  from types import SimpleNamespace
  from unittest.mock import AsyncMock, patch

  import pytest

  from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_2_sink_chat
  from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved


  def _fake_resolve_sink_response(args: dict) -> SimpleNamespace:
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
                      content=None,
                      tool_calls=[
                          SimpleNamespace(
                              function=SimpleNamespace(
                                  name="resolve_sink",
                                  arguments=json.dumps(args),
                              )
                          )
                      ],
                  )
              )
          ]
      )


  _JSON_SINK_ARGS = {
      "resolution": "sink",
      "outputs": [
          {
              "plugin": "json",
              "options": {"path": "out.jsonl", "schema": {"mode": "observed"}},
              "required_fields": [],
              "schema_mode": "observed",
          }
      ],
      "assistant_message": "I set the output to a JSON Lines file.",
  }


  @pytest.mark.asyncio
  async def test_sink_driver_resolves_json_output() -> None:
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_resolve_sink_response(_JSON_SINK_ARGS)),
      ):
          result = await maybe_resolve_step_2_sink_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="write the results to a jsonl file",
              current_sink=None,
              temperature=None,
              seed=None,
          )
      assert result is not None
      sink, assistant_message = result
      assert isinstance(sink, SinkResolved)
      assert len(sink.outputs) == 1
      assert sink.outputs[0].plugin == "json"
      assert sink.outputs[0].options["path"] == "out.jsonl"
      assert assistant_message == "I set the output to a JSON Lines file."
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_sink_driver.py -x -q
  ```
  Expected failure: `ImportError: cannot import name 'maybe_resolve_step_2_sink_chat' from 'elspeth.web.composer.guided.chat_solver'`.

- [ ] **Step 2: Add the sink tool schema, parser, and driver.**
  In `chat_solver.py`, after `maybe_resolve_step_1_source_chat` (ends `:343`), add
  the import (top of file, with the SourceResolved import from Task 1):
  ```python
  from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved
  ```
  Then add:

  ```python
  _STEP_2_SINK_TOOL: dict[str, Any] = {
      "type": "function",
      "function": {
          "name": "resolve_sink",
          "description": (
              "Use when the Step 2 chat message contains enough information to "
              "configure the pipeline output(s). Do not use for general advice."
          ),
          "parameters": {
              "type": "object",
              "additionalProperties": False,
              "required": ["resolution", "outputs", "assistant_message"],
              "properties": {
                  "resolution": {"type": "string", "enum": ["sink"]},
                  "outputs": {
                      "type": "array",
                      "minItems": 1,
                      # MVP single-output constraint enforced at the schema boundary:
                      # handle_step_2_sink loops outputs as sink_name="main"
                      # (last-write-wins) and the from-resolved re-render shows
                      # outputs[0] — so >1 output would silently disagree. Cap at 1.
                      "maxItems": 1,
                      "items": {
                          "type": "object",
                          "additionalProperties": False,
                          "required": ["plugin", "options", "required_fields", "schema_mode"],
                          "properties": {
                              "plugin": {"type": "string", "minLength": 1},
                              # Bare object: option shape varies by sink plugin.
                              # Validated downstream by handle_step_2_sink ->
                              # _execute_set_output.
                              "options": {"type": "object"},
                              "required_fields": {"type": "array", "items": {"type": "string"}},
                              "schema_mode": {"type": "string", "enum": ["fixed", "flexible", "observed"]},
                          },
                      },
                  },
                  "assistant_message": {"type": "string", "minLength": 1},
              },
          },
      },
  }


  def _build_step_2_sink_tool_prompt(*, current_sink: SinkResolved | None) -> str:
      """Compose the Step-2 sink tool prompt."""
      revise_block = ""
      if current_sink is not None:
          revise_block = (
              "\n## Current applied sink (revise relative to this)\n\n"
              "A sink has already been applied. The user's message is a REVISION "
              "instruction against it — re-emit the COMPLETE updated outputs (not a "
              "diff). Current sink:\n"
              f"{json.dumps(current_sink.to_dict(), sort_keys=True)}\n"
          )
      return (
          f"{load_step_chat_skill(GuidedStep.STEP_2_SINK).rstrip()}\n\n"
          "## Step 2 Sink Tool\n\n"
          f"{revise_block}"
          "If the user's message provides enough information to configure the "
          "pipeline output, call `resolve_sink` with the complete list of outputs "
          "(plugin, options, required_fields, schema_mode) and a brief "
          "assistant_message. If the message is only a question or lacks enough "
          "detail, reply in prose and do not call a tool.\n"
      )


  def _parse_step_2_sink_tool_arguments(arguments: str) -> tuple[SinkResolved, str]:
      """Validate the resolve_sink tool arguments. Returns (sink, assistant_message)."""
      data = json.loads(arguments)
      if not isinstance(data, Mapping):
          raise ValueError(f"resolve_sink arguments must decode to an object; got {type(data).__name__}")
      missing = {"resolution", "outputs", "assistant_message"} - set(data.keys())
      if missing:
          raise ValueError(f"resolve_sink arguments missing required keys: {sorted(missing)}")
      if data["resolution"] != "sink":
          raise ValueError(f"resolve_sink resolution must be 'sink'; got {data['resolution']!r}")
      outputs_raw = data["outputs"]
      if not isinstance(outputs_raw, list) or not outputs_raw:
          raise ValueError("resolve_sink outputs must be a non-empty list")
      outputs: list[SinkOutputResolved] = []
      for idx, item in enumerate(outputs_raw):
          if not isinstance(item, Mapping):
              raise ValueError(f"resolve_sink outputs[{idx}] must be an object; got {type(item).__name__}")
          plugin = item.get("plugin")
          if not isinstance(plugin, str) or not plugin:
              raise ValueError(f"resolve_sink outputs[{idx}].plugin must be a non-empty string; got {plugin!r}")
          options = item.get("options")
          if not isinstance(options, Mapping):
              raise ValueError(f"resolve_sink outputs[{idx}].options must be an object")
          required_fields_raw = item.get("required_fields")
          if not isinstance(required_fields_raw, list):
              raise ValueError(f"resolve_sink outputs[{idx}].required_fields must be a list")
          required_fields: list[str] = []
          for col_idx, col in enumerate(required_fields_raw):
              if not isinstance(col, str) or not col:
                  raise ValueError(f"resolve_sink outputs[{idx}].required_fields[{col_idx}] must be a non-empty string")
              required_fields.append(col)
          schema_mode = item.get("schema_mode")
          if schema_mode not in ("fixed", "flexible", "observed"):
              raise ValueError(f"resolve_sink outputs[{idx}].schema_mode must be fixed/flexible/observed; got {schema_mode!r}")
          outputs.append(
              SinkOutputResolved(
                  plugin=plugin,
                  options=dict(options),
                  required_fields=tuple(required_fields),
                  schema_mode=schema_mode,
              )
          )
      assistant_message = data["assistant_message"]
      if not isinstance(assistant_message, str) or not assistant_message.strip():
          raise ValueError("resolve_sink assistant_message must be a non-empty string")
      return SinkResolved(outputs=tuple(outputs)), assistant_message


  async def maybe_resolve_step_2_sink_chat(
      *,
      model: str,
      user_message: str,
      current_sink: SinkResolved | None,
      temperature: float | None,
      seed: int | None,
      recorder: BufferingRecorder | None = None,
  ) -> tuple[SinkResolved, str] | None:
      """Try to resolve a Step-2 chat message into a sink config.

      Returns ``(sink, assistant_message)`` on a ``resolve_sink`` tool call, or
      ``None`` when the model replies in prose, allowing the route to fall back
      to advisory chat. Mirrors :func:`maybe_resolve_step_1_source_chat`.
      """
      if not user_message:
          raise InvariantError("maybe_resolve_step_2_sink_chat: user_message is empty (route validation gap)")

      from litellm.exceptions import APIError as LiteLLMAPIError
      from litellm.exceptions import AuthenticationError as LiteLLMAuthError
      from litellm.exceptions import BadRequestError as LiteLLMBadRequestError

      messages: list[dict[str, Any]] = [
          {"role": "system", "content": _build_step_2_sink_tool_prompt(current_sink=current_sink)},
          {"role": "user", "content": user_message},
      ]
      tools = [_STEP_2_SINK_TOOL]
      kwargs: dict[str, Any] = {"model": model, "messages": messages, "tools": tools}
      if temperature is not None:
          kwargs["temperature"] = temperature
      if seed is not None:
          kwargs["seed"] = seed
      started_at = datetime.now(UTC)
      started_ns = time.monotonic_ns()
      status: ComposerLLMCallStatus | None = None
      response: Any = None
      error_class: str | None = None
      error_message: str | None = None
      try:
          response = await _litellm_acompletion(**kwargs)
          message = response.choices[0].message
          tool_calls = message.tool_calls or ()
          for tool_call in tool_calls:
              function = tool_call.function
              if function is None:
                  continue
              if function.name != "resolve_sink":
                  continue
              arguments = function.arguments
              if not isinstance(arguments, str):
                  raise ValueError(f"resolve_sink function.arguments must be a JSON string; got {type(arguments).__name__}")
              sink, assistant = _parse_step_2_sink_tool_arguments(arguments)
              status = ComposerLLMCallStatus.SUCCESS
              return sink, assistant
          status = ComposerLLMCallStatus.SUCCESS
          return None
      except TimeoutError:
          status = ComposerLLMCallStatus.TIMEOUT
          error_class = "TimeoutError"
          error_message = "TimeoutError"
          raise
      except asyncio.CancelledError as exc:
          status = ComposerLLMCallStatus.CANCELLED
          error_class = type(exc).__name__
          error_message = type(exc).__name__
          raise
      except LiteLLMAuthError as exc:
          status = ComposerLLMCallStatus.AUTH_ERROR
          error_class = type(exc).__name__
          error_message = type(exc).__name__
          raise
      except LiteLLMBadRequestError as exc:
          status = ComposerLLMCallStatus.BAD_REQUEST_ERROR
          error_class = type(exc).__name__
          error_message = type(exc).__name__
          raise
      except LiteLLMAPIError as exc:
          status = ComposerLLMCallStatus.API_ERROR
          error_class = type(exc).__name__
          error_message = type(exc).__name__
          raise
      except (IndexError, AttributeError, json.JSONDecodeError, ValueError) as exc:
          status = ComposerLLMCallStatus.MALFORMED_RESPONSE
          error_class = type(exc).__name__
          error_message = "malformed_response"
          raise
      except Exception as exc:
          status = ComposerLLMCallStatus.API_ERROR
          error_class = type(exc).__name__
          error_message = type(exc).__name__
          raise
      finally:
          _record_llm_call(
              recorder=recorder,
              model=model,
              messages=messages,
              tools=tools,
              status=status,
              started_at=started_at,
              started_ns=started_ns,
              temperature=temperature,
              seed=seed,
              response=response,
              error_class=error_class,
              error_message=error_message,
          )
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_sink_driver.py::test_sink_driver_resolves_json_output -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Add the failing test for prose fall-back, then confirm it passes.**
  Append to `test_step_chat_sink_driver.py`:

  ```python
  @pytest.mark.asyncio
  async def test_sink_driver_returns_none_on_prose() -> None:
      prose = SimpleNamespace(
          choices=[SimpleNamespace(message=SimpleNamespace(content="A sink writes rows out.", tool_calls=None))]
      )
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=prose),
      ):
          result = await maybe_resolve_step_2_sink_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="what is a sink?",
              current_sink=None,
              temperature=None,
              seed=None,
          )
      assert result is None


  @pytest.mark.asyncio
  async def test_sink_driver_revise_threads_current_sink() -> None:
      current = SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": "old.jsonl"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )
      captured: dict = {}

      async def _capture(**kwargs):
          captured.update(kwargs)
          return _fake_resolve_sink_response(_JSON_SINK_ARGS)

      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(side_effect=_capture),
      ):
          await maybe_resolve_step_2_sink_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="rename the file to out.jsonl",
              current_sink=current,
              temperature=None,
              seed=None,
          )
      assert "old.jsonl" in captured["messages"][0]["content"]
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_sink_driver.py -x -q
  ```
  Expected: `3 passed`.

- [ ] **Step 4: Add the sink auto-drop wrapper.**
  In `_guided_step_chat.py`, after `resolve_step_1_source_chat_with_auto_drop`
  (ends `:179`), add the dataclass + wrapper. Mirror the source wrapper's
  transient-exception set and slog discipline exactly.

  Add `maybe_resolve_step_2_sink_chat` and `SinkResolved` to the imports from
  `chat_solver` / `resolved` at the top of `_guided_step_chat.py` (confirm the
  existing import lines with `grep -n "from elspeth.web.composer.guided" src/elspeth/web/sessions/_guided_step_chat.py`):

  ```python
  @dataclass(frozen=True, slots=True)
  class Step2SinkChatResult:
      """Outcome of a Step-2 sink chat attempt with auto-drop fall-back.

      ``sink_resolution`` carries a valid ``resolve_sink`` tool result, or
      ``None`` when the model replied in prose (the route continues to the
      advisory guided-chat path). ``assistant_message`` carries the LLM's reply
      that accompanied the tool call (``None`` on prose/failure).
      ``fallback_chat`` carries the synthetic unavailable message on transient
      LLM failure.
      """

      sink_resolution: SinkResolved | None
      assistant_message: str | None
      fallback_chat: StepChatResult | None


  async def resolve_step_2_sink_chat_with_auto_drop(
      *,
      site: str,
      session_id: str,
      user_id: str,
      model: str,
      user_message: str,
      current_sink: SinkResolved | None,
      temperature: float | None,
      seed: int | None,
      recorder: BufferingRecorder | None = None,
  ) -> Step2SinkChatResult:
      """Wrap Step-2 ``resolve_sink`` chat with the guided-chat fallback contract."""
      from litellm.exceptions import APIError as LiteLLMAPIError
      from litellm.exceptions import AuthenticationError as LiteLLMAuthError
      from litellm.exceptions import BadRequestError as LiteLLMBadRequestError
      from litellm.exceptions import (
          BlockedPiiEntityError,
          BudgetExceededError,
          GuardrailInterventionNormalStringError,
          GuardrailRaisedException,
      )

      started = time.perf_counter()
      try:
          resolved = await maybe_resolve_step_2_sink_chat(
              model=model,
              user_message=user_message,
              current_sink=current_sink,
              temperature=temperature,
              seed=seed,
              recorder=recorder,
          )
          if resolved is None:
              return Step2SinkChatResult(sink_resolution=None, assistant_message=None, fallback_chat=None)
          sink, assistant_message = resolved
          return Step2SinkChatResult(sink_resolution=sink, assistant_message=assistant_message, fallback_chat=None)
      except (
          LiteLLMAPIError,
          LiteLLMAuthError,
          LiteLLMBadRequestError,
          BudgetExceededError,
          BlockedPiiEntityError,
          GuardrailRaisedException,
          GuardrailInterventionNormalStringError,
          TimeoutError,
          IndexError,
          AttributeError,
          json.JSONDecodeError,
          ValueError,
      ) as exc:
          latency_ms = int((time.perf_counter() - started) * 1000)
          slog.error(
              "guided.step_2_sink_chat_transient_failure",
              session_id=session_id,
              user_id=user_id,
              site=site,
              step=GuidedStep.STEP_2_SINK.value,
              exc_class=type(exc).__name__,
              latency_ms=latency_ms,
              frames=_safe_frame_strings(exc),
          )
          return Step2SinkChatResult(
              sink_resolution=None,
              assistant_message=None,
              fallback_chat=StepChatResult(
                  assistant_message=_SYNTHETIC_UNAVAILABLE_MESSAGE,
                  status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                  latency_ms=latency_ms,
                  error_class=type(exc).__name__,
              ),
          )
  ```

  Run to confirm the module imports and the source suite is still green:
  ```
  cd /home/john/elspeth && uv run python -c "from elspeth.web.sessions._guided_step_chat import resolve_step_2_sink_chat_with_auto_drop, Step2SinkChatResult; print('ok')" && uv run pytest tests/integration/web/composer/guided/test_step_chat_sink_driver.py -q
  ```
  Expected: `ok` then `3 passed`.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/composer/guided/chat_solver.py src/elspeth/web/sessions/_guided_step_chat.py tests/integration/web/composer/guided/test_step_chat_sink_driver.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): add free-text sink driver + auto-drop wrapper

Add maybe_resolve_step_2_sink_chat (resolve_sink tool -> SinkResolved,
None on prose) mirroring the source driver, plus
resolve_step_2_sink_chat_with_auto_drop and Step2SinkChatResult with the
same transient-failure / synthetic-unavailable contract. current_sink is
threaded into the prompt for in-place revise. Not yet wired into the
route (next task).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 2.5: Prefill builders + GET /guided resumability for the applied-in-place state

> **This task is a prerequisite for Tasks 3 and 4** and exists because in-place
> apply creates a NEW resumable state that did not exist under the old
> auto-advance contract: `step == STEP_1` with `step_1_result` set but
> `step_1_source_intent` / `step_1_chosen_plugin` both `None` (and the STEP_2
> analogue). Two defects must be closed together or the apply is not durable:
> 1. **The re-rendered form is not populated.** `build_step_1_schema_form_turn` /
>    `build_step_2_schema_form_turn` (`emitters.py:120/:209`) hardcode `prefilled =
>    {"schema": {"mode": "observed"}}` — they carry plugin + knobs but NOT the
>    committed `path`/`options`/`observed_columns`. The spec (lines 124) and the
>    contract (§1.3) require the form to render "populated with the applied
>    config." A near-empty form defeats LLM-primary.
> 2. **GET /guided drops the applied form on refresh.** `_build_get_guided_turn`
>    (`guided.py:155-182`) at STEP_1 with only `step_1_result` set falls through to
>    `build_initial_step_1_turn` (`:168`) — the empty opening turn; at STEP_2 it
>    falls to `build_step_2_single_select_turn` (`:182`). The applied source/sink
>    vanishes on a page refresh.
>
> Both are fixed by ONE pair of from-resolved prefill builders used by BOTH the
> chat-apply branch (Tasks 3/4) AND the GET rebuild. The chat-apply `next_turn`
> and the GET refresh turn must be the SAME populated turn so apply and refresh
> agree.

**Files:**
- Modify: `src/elspeth/web/composer/guided/emitters.py` (add two builders after
  `build_step_2_schema_form_turn`, `:238`; precedent: the from-resolved builder
  `build_step_1_inspect_and_confirm_turn_from_intent`, `:87`)
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
  (`_build_get_guided_turn`, STEP_1 branch `:155-168`, STEP_2 branch
  `:169-182`; + the `from .._helpers import (...)` block `:7-93`)
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (re-export the two new
  builders: emitter import block `:71-83` + `__all__` `:4362-4371`)
- Test: `tests/unit/web/composer/guided/test_prefill_from_resolved.py` (Create)
- Test: `tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py` (Create;
  the HTTP apply-and-refresh `test_guided_refresh_applied.py` is created in Task 3
  Step 3, once in-place apply exists)

**Interfaces:**
- Consumes: `SourceResolved` / `SinkResolved` (`resolved.py`); `CatalogServiceProtocol`;
  `build_step_1_schema_form_turn` / `build_step_2_schema_form_turn` (the bare
  builders, reused for the knobs).
- Produces (Tasks 3 & 4 consume; GET /guided consumes):
  `build_step_1_schema_form_turn_from_resolved(source: SourceResolved, catalog:
  CatalogServiceProtocol) -> Turn` and
  `build_step_2_schema_form_turn_from_resolved(sink: SinkResolved, catalog:
  CatalogServiceProtocol) -> Turn`. Each returns the same SCHEMA_FORM turn as the
  bare builder for the plugin but with `prefilled` carrying the applied
  `options` (merged over the `{"schema": {"mode": "observed"}}` default) so the
  form renders the committed config.

- [ ] **Step 1: Write the failing unit test that the from-resolved builder
  prefills the applied options.**
  Create `tests/unit/web/composer/guided/test_prefill_from_resolved.py`:

  ```python
  """p1 Task 2.5 — from-resolved schema_form builders prefill the applied config."""

  from __future__ import annotations

  from unittest.mock import MagicMock

  from elspeth.web.catalog.protocol import CatalogService
  from elspeth.web.composer.guided.emitters import (
      build_step_1_schema_form_turn_from_resolved,
      build_step_2_schema_form_turn_from_resolved,
  )
  from elspeth.web.composer.guided.resolved import (
      SinkOutputResolved,
      SinkResolved,
      SourceResolved,
  )


  def _catalog() -> CatalogService:
      catalog = MagicMock(spec=CatalogService)
      catalog.get_schema.return_value = MagicMock(knob_schema={"fields": []})
      return catalog


  def test_source_prefill_carries_applied_options() -> None:
      source = SourceResolved(
          plugin="csv",
          options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
          observed_columns=("a", "b"),
          sample_rows=({"a": "1", "b": "2"},),
      )
      turn = build_step_1_schema_form_turn_from_resolved(source, _catalog())
      assert turn["type"] == "schema_form"
      assert turn["step_index"] == 0
      assert turn["payload"]["plugin"] == "csv"
      assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


  def test_sink_prefill_carries_applied_options() -> None:
      sink = SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": "/out/y.jsonl", "collision_policy": "auto_increment"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )
      turn = build_step_2_schema_form_turn_from_resolved(sink, _catalog())
      assert turn["type"] == "schema_form"
      assert turn["step_index"] == 1
      assert turn["payload"]["plugin"] == "json"
      assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
      assert turn["payload"]["prefilled"]["collision_policy"] == "auto_increment"
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/test_prefill_from_resolved.py -x -q
  ```
  Expected failure: `ImportError: cannot import name 'build_step_1_schema_form_turn_from_resolved'`.

- [ ] **Step 2: Add the two from-resolved builders.**
  In `emitters.py`, after `build_step_2_schema_form_turn` (`:238`), add (the
  `SourceResolved`/`SinkResolved` imports may already be present; grep first and
  add to the existing `from ...resolved import` block if not):

  ```python
  def build_step_1_schema_form_turn_from_resolved(
      source: SourceResolved,
      catalog: CatalogServiceProtocol,
  ) -> Turn:
      """Build the STEP_1 ``schema_form`` populated from an APPLIED source.

      Unlike :func:`build_step_1_schema_form_turn` (which seeds an empty
      ``prefilled``), this renders the committed ``source.options`` so the
      editable form shows what the LLM (or the manual path) built. Used by the
      chat-apply in-place re-render and by GET /guided when ``step_1_result`` is
      set on a STEP_1 session.
      """
      schema_info = catalog.get_schema("source", source.plugin)
      prefilled: dict[str, Any] = {"schema": {"mode": "observed"}, **dict(source.options)}
      payload: SchemaFormPayload = {
          "mode": "plugin_options",
          "plugin": source.plugin,
          "knobs": cast(KnobSchema, schema_info.knob_schema),
          "prefilled": prefilled,
      }
      return Turn(
          type=TurnType.SCHEMA_FORM.value,
          step_index=_step_index(GuidedStep.STEP_1_SOURCE),
          payload=payload,
      )


  def build_step_2_schema_form_turn_from_resolved(
      sink: SinkResolved,
      catalog: CatalogServiceProtocol,
  ) -> Turn:
      """Build the STEP_2 ``schema_form`` populated from an APPLIED sink.

      Renders the first output's committed ``options`` (MVP single-output
      constraint, matching ``handle_step_2_sink``'s ``sink_name="main"`` loop).
      Used by the chat-apply in-place re-render and by GET /guided when
      ``step_2_result`` is set on a STEP_2 session.
      """
      if not sink.outputs:
          raise InvariantError("build_step_2_schema_form_turn_from_resolved: sink has no outputs")
      output = sink.outputs[0]
      schema_info = catalog.get_schema("sink", output.plugin)
      prefilled: dict[str, Any] = {"schema": {"mode": "observed"}, **dict(output.options)}
      payload: SchemaFormPayload = {
          "mode": "plugin_options",
          "plugin": output.plugin,
          "knobs": cast(KnobSchema, schema_info.knob_schema),
          "prefilled": prefilled,
      }
      return Turn(
          type=TurnType.SCHEMA_FORM.value,
          step_index=_step_index(GuidedStep.STEP_2_SINK),
          payload=payload,
      )
  ```
  Confirm `InvariantError`, `SourceResolved`, `SinkResolved` are imported in
  `emitters.py` (grep `grep -n "InvariantError\|import SourceResolved\|import
  SinkResolved" src/elspeth/web/composer/guided/emitters.py`); add any missing.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/test_prefill_from_resolved.py -x -q
  ```
  Expected: `2 passed`.

- [ ] **Step 3: Write the failing GET-rebuild test that exercises the from-resolved
  branch directly (no false-green placeholder).**
  The full HTTP-level "apply via chat then refresh" invariant only exists once
  Task 3 lands the in-place apply with staging-field clearing — so do NOT write a
  trivial `next_turn is not None` integration assertion here (it would pass at the
  STEP_2 state the manual path reaches and never touch the from-resolved builder —
  a false-green checkpoint). Instead write a UNIT test that constructs the
  in-place applied STEP_1/STEP_2 session state directly (`step_N_result` set,
  staging fields `None`) and asserts `_build_get_guided_turn` returns the populated
  from-resolved turn. This goes red before Step 4 and green after, with no
  cross-task ordering hazard. The HTTP apply-and-refresh invariant is owned by
  Task 3 Step 3 (which creates `test_guided_refresh_applied.py`).

  Create `tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py`:

  ```python
  """p1 Task 2.5 — _build_get_guided_turn re-renders the applied form in place."""

  from __future__ import annotations

  from unittest.mock import MagicMock

  from elspeth.web.catalog.protocol import CatalogService
  from elspeth.web.composer.guided.protocol import GuidedStep
  from elspeth.web.composer.guided.resolved import (
      SinkOutputResolved,
      SinkResolved,
      SourceResolved,
  )
  from elspeth.web.sessions.routes.composer.guided import _build_get_guided_turn


  def _catalog() -> CatalogService:
      catalog = MagicMock(spec=CatalogService)
      catalog.get_schema.return_value = MagicMock(knob_schema={"fields": []})
      return catalog


  def test_get_rebuild_step_1_uses_from_resolved_when_applied_in_place() -> None:
      source = SourceResolved(
          plugin="csv",
          options={"path": "/data/x.csv", "schema": {"mode": "observed"}},
          observed_columns=("a", "b"),
          sample_rows=({"a": "1", "b": "2"},),
      )
      # In-place applied STEP_1 state: result set, BOTH staging fields cleared.
      guided = MagicMock()
      guided.step = GuidedStep.STEP_1_SOURCE
      guided.step_1_source_intent = None
      guided.step_1_chosen_plugin = None
      guided.step_1_result = source
      turn = _build_get_guided_turn(MagicMock(), guided, catalog=_catalog())
      assert turn["type"] == "schema_form"
      assert turn["step_index"] == 0
      assert turn["payload"]["plugin"] == "csv"
      assert turn["payload"]["prefilled"]["path"] == "/data/x.csv"


  def test_get_rebuild_step_2_uses_from_resolved_when_applied_in_place() -> None:
      sink = SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": "/out/y.jsonl"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )
      guided = MagicMock()
      guided.step = GuidedStep.STEP_2_SINK
      guided.step_2_sink_intent = None
      guided.step_2_chosen_plugin = None
      guided.step_2_result = sink
      turn = _build_get_guided_turn(MagicMock(), guided, catalog=_catalog())
      assert turn["type"] == "schema_form"
      assert turn["step_index"] == 1
      assert turn["payload"]["plugin"] == "json"
      assert turn["payload"]["prefilled"]["path"] == "/out/y.jsonl"
  ```

  > If `_build_get_guided_turn`'s STEP_1/STEP_2 branches dereference an attribute
  > on `guided` that the `MagicMock` does not pin to a concrete value (e.g.
  > `step_1_inspection_facts`), set it to a benign value on the mock — the new
  > sub-case is reached only when both staging fields are `None`, so the
  > chosen-plugin branch that reads `step_1_inspection_facts` is not taken; pin it
  > to `None` anyway to keep the mock from auto-creating a child mock that breaks a
  > truthiness check.

  Run to fail (the from-resolved sub-case does not exist yet, so STEP_1 falls
  through to `build_initial_step_1_turn` → `single_select`, STEP_2 to
  `single_select`):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py -x -q
  ```
  Expected failure: `assert 'single_select' == 'schema_form'` (the populated
  sub-case is missing).

- [ ] **Step 4: Fix `_build_get_guided_turn` to emit the from-resolved turn
  for the applied state.**
  In `guided.py`, in the STEP_1 branch, add a sub-case for the in-place applied
  state BEFORE the `build_initial_step_1_turn` fall-through: when `step_1_result`
  is set and both staging fields are `None`, emit the populated form. The branch
  has THREE existing sub-cases in priority order (intent → chosen-plugin →
  fall-through); the new sub-case goes between chosen-plugin and the fall-through.
  Replace the WHOLE STEP_1 branch (`:155-168`) — keep the intent sub-case
  unchanged so a still-pending INSPECT_AND_CONFIRM is not clobbered:
  ```python
      if step is GuidedStep.STEP_1_SOURCE:
          if guided.step_1_source_intent is not None:
              return build_step_1_inspect_and_confirm_turn_from_intent(guided.step_1_source_intent)
          if guided.step_1_chosen_plugin is not None:
              return build_step_1_schema_form_turn(
                  guided.step_1_chosen_plugin,
                  catalog,
                  inspection_facts=guided.step_1_inspection_facts,
              )
          if guided.step_1_result is not None:
              # In-place applied source (chat apply): re-render the populated form.
              # Reaches here only when the chat-apply branch (Task 3) cleared the
              # staging fields after committing — so a manual in-progress plugin
              # switch (chosen_plugin set) still wins above.
              return build_step_1_schema_form_turn_from_resolved(guided.step_1_result, catalog)
          return build_initial_step_1_turn(state, blob_inspection=None, catalog=catalog)
  ```
  And replace the WHOLE STEP_2 branch (`:169-182`) — preserve ALL THREE existing
  sub-cases (`step_2_sink_intent` → `step_2_chosen_plugin` → fall-through) and
  insert the new from-resolved check before the fall-through:
  ```python
      if step is GuidedStep.STEP_2_SINK:
          if guided.step_2_sink_intent is not None:
              # SCHEMA_FORM submitted; session is waiting for MULTI_SELECT_WITH_CUSTOM.
              observed_columns: tuple[str, ...] = ()
              if guided.step_1_result is not None:
                  observed_columns = tuple(guided.step_1_result.observed_columns)
              return build_step_2_multi_select_turn(observed_columns)
          if guided.step_2_chosen_plugin is not None:
              return build_step_2_schema_form_turn(guided.step_2_chosen_plugin, catalog)
          if guided.step_2_result is not None:
              # In-place applied sink (chat apply): re-render the populated form.
              return build_step_2_schema_form_turn_from_resolved(guided.step_2_result, catalog)
          return build_step_2_single_select_turn(catalog)
  ```
  > **Anchor note:** the STEP_2 branch's `step_2_sink_intent` sub-case
  > (`:172-177`) returns a MULTI_SELECT turn and MUST be preserved — do not
  > collapse the STEP_2 edit to the tail two sub-cases, or a pending sink-schema
  > submission falls through to the wrong turn. Replace the entire `:169-182`
  > block.

  **Import wiring (the new builders reach `guided.py` THROUGH `_helpers.py`,
  not directly).** `guided.py` imports every emitter via `from .._helpers import
  (...)` (`:7-93`); there is no direct `from ...emitters import` block in
  `guided.py`. So thread the two new builders through `_helpers.py` first:
  1. Add `build_step_1_schema_form_turn_from_resolved` and
     `build_step_2_schema_form_turn_from_resolved` to `_helpers.py`'s
     `from elspeth.web.composer.guided.emitters import (...)` block (`:71-83`).
  2. Add both names to `_helpers.py`'s `__all__` list (the emitter entries sit
     at `:4362-4371`, alphabetical).
  3. Add both names to `guided.py`'s `from .._helpers import (...)` block
     (`:7-93`, alphabetical, alongside `build_step_1_schema_form_turn` `:69` /
     `build_step_2_schema_form_turn` `:72`).

  Run to pass (and confirm no existing GET test regressed — the new sub-cases only
  fire when `step_N_result` is set AND the staging fields are None, a state that
  did not previously exist at these steps):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py tests/integration/web/composer/guided/test_get_guided.py -q
  ```
  Expected: green.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/composer/guided/emitters.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/composer/guided/test_prefill_from_resolved.py tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): from-resolved schema_form prefill + GET resumability

Add build_step_{1,2}_schema_form_turn_from_resolved, which render the
SCHEMA_FORM populated with the APPLIED source/sink options (the bare
builders seed an empty prefill). Wire them into _build_get_guided_turn
so GET /guided re-renders the committed config when a phase is applied in
place (step_N_result set, staging fields None) instead of falling through
to the empty opening turn. This makes the in-place apply state (new under
LLM-primary) durable across refresh; the chat-apply branches (next tasks)
emit the same populated turn so apply and refresh agree.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 3: STEP_1 chat applies IN-PLACE (remove auto-advance; re-render the source form)

> **Depends on Task 2.5** (the `build_step_1_schema_form_turn_from_resolved`
> builder + the GET /guided populated-state rebuild). Do Task 2.5 first.

**Files:**
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
  (`post_guided_chat` STEP_1 apply branch, `:1754-1992` — the WHOLE region:
  entry guard `:1754-1758`, answered-record leg `:1843-1861`, auto-advance tail
  `:1863-1893`)
- Modify: `tests/integration/web/composer/guided/test_step_chat.py`
  (`test_step_1_chat_can_commit_generated_csv_source_and_emit_step_2`, the apply
  precedent at `:435` — the test-to-UPDATE; + a phase-entry drive test)
- Test: `tests/integration/web/composer/guided/test_guided_refresh_applied.py`
  (Create — the apply-and-refresh invariant deferred from Task 2.5)

**Interfaces:**
- Consumes: `handle_step_1_source` (`steps.py:65`); `resolve_step_1_source_chat_with_auto_drop`
  (now with `current_source`, Task 1); `build_step_1_schema_form_turn_from_resolved(source,
  catalog)` (Task 2.5, `emitters.py`); the audit emitters
  `emit_turn_answered` / `emit_turn_emitted` (already imported in `guided.py`).
- Produces (p2 consumes via the response contract): on an actionable STEP_1 chat
  apply, the response now carries `guided_session.step == "step_1_source"`
  (UNCHANGED), `composition_state.sources.source` committed, and `next_turn` = the
  re-rendered STEP_1 `schema_form` populated from the applied source. The
  `emit_step_advanced` audit event is GONE on this path.

> **Why in-place (the decided contract).** The spec's revise model is in-place
> revision of the CURRENT phase; Back-nav handles earlier phases. If chat-apply
> auto-advanced, the just-configured phase would instantly become "earlier" and
> could never be revised in place. So the STEP_1 branch must STOP advancing to
> STEP_2 (the `_replace(guided, step=GuidedStep.STEP_2_SINK)` at `:1863`) and
> instead re-render the STEP_1 source form.

- [ ] **Step 1: Update the existing test to assert in-place apply (it currently
  asserts auto-advance to step 2).**
  In `test_step_chat.py`, rename and rewrite the assertions of
  `test_step_1_chat_can_commit_generated_csv_source_and_emit_step_2`. Change the
  method name to `test_step_1_chat_commits_source_in_place_and_rerenders_form` and
  replace the post-call assertion block (the lines asserting `step == "step_2_sink"`
  / `next_turn.type == "single_select"` / `step_index == 1`) with:

  ```python
          assert status == 200, body
          assert body["assistant_message"] == "I set this up as a CSV source."
          # Apply-in-place: the phase stays STEP_1 (revise model), and the form
          # re-renders populated from the committed source.
          assert body["guided_session"]["step"] == "step_1_source"
          assert body["next_turn"]["type"] == "schema_form"
          assert body["next_turn"]["step_index"] == 0
          # The re-rendered form is POPULATED from the committed source (the whole
          # point of LLM-primary: the form shows what was just built).
          assert body["next_turn"]["payload"]["plugin"] == "csv"
          assert body["next_turn"]["payload"]["prefilled"]["path"].endswith("_teal_colours.csv")
          assert body["composition_state"]["sources"]["source"]["plugin"] == "csv"
          source_options = body["composition_state"]["sources"]["source"]["options"]
          assert source_options["schema"]["mode"] == "observed"
          assert source_options["path"].endswith("_teal_colours.csv")
          audits = _llm_call_audit_bodies(composer_test_client, session_id)
          assert len(audits) == 1, audits
          assert audits[0]["status"] == "success"
  ```
  Also update the method docstring's last sentence ("...and return the Step-2 turn
  so the UI can advance.") to read "...and re-render the Step-1 form in place so
  the user can revise or advance explicitly."

  Run to fail (the route still auto-advances):
  ```
  cd /home/john/elspeth && uv run pytest "tests/integration/web/composer/guided/test_step_chat.py::TestStep1SourceResolution::test_step_1_chat_commits_source_in_place_and_rerenders_form" -x -q
  ```
  Expected failure: `assert 'step_2_sink' == 'step_1_source'` (the route still
  advances).

- [ ] **Step 2: Make the STEP_1 apply branch fire at phase entry, commit in place,
  clear staging, and re-render.**
  This is ONE coordinated revision of the contiguous STEP_1 apply region
  (`:1754-1893`), not four independent patches — patching them separately produces
  a self-contradictory branch (a widened guard running against a hardcoded
  SCHEMA_FORM audit leg). Make all four edits below together, then run the tests.

  **2a — Widen the entry guard so the driver fires at phase entry (not only from
  the SCHEMA_FORM sub-state).** The guard at `:1754-1758` currently requires
  `current_turn_type is TurnType.SCHEMA_FORM`. At STEP_1 phase entry the recorded
  turn is `SINGLE_SELECT` (GET /guided records the initial turn before any chat —
  `_build_get_guided_turn` → `build_initial_step_1_turn` emits SINGLE_SELECT,
  recorded via `_append_server_turn_record`, `guided.py:320-342`). So the
  SCHEMA_FORM-only guard never matches at entry, and the LLM driver cannot drive
  STEP_1 until the user manually picks a plugin — which breaks the LLM-primary
  premise (contract §1.2: every chat submit ATTEMPTS to drive the phase) AND the
  p4 passive-drive tutorial (it types no manual selection). Widen the turn-type
  conjunct to the three valid STEP_1 turn types (the same set `_build_get_guided_turn`
  rebuilds):
  ```python
              if (
                  existing_record_for_chat is not None
                  and guided.step is GuidedStep.STEP_1_SOURCE
                  and current_turn_type
                  in (TurnType.SINGLE_SELECT, TurnType.SCHEMA_FORM, TurnType.INSPECT_AND_CONFIRM)
              ):
  ```
  > The `existing_record_for_chat is not None` conjunct is kept: GET /guided
  > always records an initial turn first (verified `:320`), so a normal/tutorial
  > flow reaches `/guided/chat` with a recorded SINGLE_SELECT. If you ever observe
  > a `/guided/chat` POST with NO recorded STEP_1 turn (the conjunct False), that
  > is the genuinely-cold no-turn state — leave it falling through to advisory; do
  > NOT relax the `is not None` conjunct.

  **2b — Gate the answered-record audit leg on SCHEMA_FORM; let the drive run
  unconditionally.** The block at `:1843-1861` builds a SCHEMA_FORM-shaped
  `turn_response`, computes `response_hash`, replaces `existing_record_for_chat`,
  and calls `emit_turn_answered` with a hardcoded `turn_type=TurnType.SCHEMA_FORM`.
  Once 2a lets the branch fire on a SINGLE_SELECT entry record, that hardcode would
  stamp a SINGLE_SELECT record as a SCHEMA_FORM answer — wrong. The
  *update-answered-record* leg is meaningful only when the user is editing an
  already-emitted SCHEMA_FORM; for a SINGLE_SELECT/INSPECT entry there is no
  schema-form answer to stamp. So gate ONLY this audit leg on the turn type, while
  `resolve → handle_step_1_source → re-render` (2c, 2d) runs regardless. Wrap the
  `:1843-1861` answered-record block (`response_hash = ...` through the
  `emit_turn_answered(...)` call) in:
  ```python
                  if current_turn_type is TurnType.SCHEMA_FORM:
                      turn_response: TurnResponse = {
                          "chosen": None,
                          "edited_values": {
                              "plugin": source_resolution.plugin,
                              "options": dict(source_resolution.options),
                              "observed_columns": list(source_resolution.observed_columns),
                              "sample_rows": [dict(row) for row in source_resolution.sample_rows],
                          },
                          "custom_inputs": None,
                          "accepted_step_index": None,
                          "edit_step_index": None,
                          "control_signal": None,
                      }
                      response_hash = stable_hash(turn_response)
                      answered_record = _replace(
                          existing_record_for_chat,
                          response_hash=response_hash,
                          summary=_summarize_guided_response(TurnType.SCHEMA_FORM, turn_response),
                      )
                      answered_history = tuple(answered_record if r is existing_record_for_chat else r for r in guided.history)
                      guided = _replace(
                          handler_result.session,
                          history=answered_history,
                          step_1_chosen_plugin=None,
                          step_1_source_intent=None,
                      )
                      emit_turn_answered(
                          recorder,
                          step=GuidedStep.STEP_1_SOURCE,
                          turn_type=TurnType.SCHEMA_FORM,
                          response_hash=response_hash,
                          response_payload_id=_store_guided_audit_payload(getattr(request.app.state, "payload_store", None), turn_response),
                          control_signal=None,
                          composition_version=state.version,
                          actor=user.user_id,
                      )
                  else:
                      # SINGLE_SELECT / INSPECT_AND_CONFIRM entry: no prior
                      # schema-form answer to stamp. Just adopt the committed session
                      # and clear the staging fields (2c).
                      guided = _replace(
                          handler_result.session,
                          step_1_chosen_plugin=None,
                          step_1_source_intent=None,
                      )
                  state = handler_result.state
  ```
  > The original `:1849` line `guided = _replace(handler_result.session,
  > history=answered_history)` is REPLACED by the guarded form above (note the
  > `state = handler_result.state` assignment at `:1850` moves out of the `if` to
  > run on both legs). Hoist `TurnResponse` into the `if` body as shown.

  **2c — Clear the STEP_1 staging fields on commit (apply/refresh must agree).**
  `handle_step_1_source` sets `step_1_result` but does NOT clear
  `step_1_chosen_plugin` / `step_1_source_intent` (verified `steps.py:142`,
  `dataclasses.replace(session, step_1_result=...)` only). If they stay set,
  `_build_get_guided_turn` (`:162`) returns the EMPTY chosen-plugin form on the
  next GET instead of the populated from-resolved turn (Task 2.5) — apply and
  refresh disagree, and Task 2.5's from-resolved GET sub-case becomes dead code.
  Both `_replace` legs in 2b add `step_1_chosen_plugin=None,
  step_1_source_intent=None`. These fields already round-trip via
  `state_machine` `to_dict`/`from_dict`, so NO epoch bump (contract §3). Do NOT fix
  this by reordering `_build_get_guided_turn` precedence to put `step_1_result`
  first — that would clobber a legitimate in-progress manual plugin switch where a
  prior `step_1_result` is still set.

  **2d — Replace the auto-advance tail with the in-place re-render.** Replace the
  block from `guided = _replace(guided, step=GuidedStep.STEP_2_SINK)` (`:1863`)
  through the end of the `emit_turn_emitted(...)` call (`:1893`) with:
  ```python
                  # Apply-in-place: the source is committed (step_1_result set by
                  # handle_step_1_source); the phase STAYS STEP_1 so the user can
                  # revise by typing again. NO step advance, NO emit_step_advanced.
                  # Re-render the source schema_form POPULATED from the committed
                  # source (Task 2.5 builder) — the same turn GET /guided now emits
                  # for this state, so apply and refresh agree.
                  next_turn = build_step_1_schema_form_turn_from_resolved(handler_result.session.step_1_result, catalog)
                  next_turn_type = TurnType(next_turn["type"])
                  next_payload_hash = stable_hash(next_turn["payload"])
                  new_record = TurnRecord(
                      step=GuidedStep.STEP_1_SOURCE,
                      turn_type=next_turn_type,
                      payload_hash=next_payload_hash,
                      response_hash=None,
                      emitter="server",
                  )
                  emit_turn_emitted(
                      recorder,
                      step=GuidedStep.STEP_1_SOURCE,
                      turn_type=next_turn_type,
                      payload_hash=next_payload_hash,
                      payload_payload_id=_store_guided_audit_payload(
                          getattr(request.app.state, "payload_store", None), next_turn["payload"]
                      ),
                      emitter="server",
                      composition_version=state.version,
                      actor=user.user_id,
                  )
  ```

  The chat-history append block (`:1896-1915`) and the persist + return
  (`:1933-1992`) stay as-is — they already carry `next_turn` and the committed
  state. The return's `next_turn=TurnPayloadResponse(...)` now serializes the
  populated STEP_1 schema_form. `build_step_1_schema_form_turn_from_resolved` was
  added to the `guided.py` `from .._helpers import` block in Task 2.5 Step 4
  (via the `_helpers.py` re-export); confirm:
  ```
  grep -n "build_step_1_schema_form_turn_from_resolved" /home/john/elspeth/src/elspeth/web/sessions/routes/composer/guided.py
  ```
  (`handler_result.session.step_1_result` is the committed `SourceResolved`
  `handle_step_1_source` just set — non-None on the success path, since the branch
  already raised 400 above on `not handler_result.tool_result.success`.)

- [ ] **Step 3: Thread `current_source` into the STEP_1 chat driver call (enables
  revise).**
  The STEP_1 branch calls `resolve_step_1_source_chat_with_auto_drop` (`:1774`).
  Pass the currently-applied source (from `guided.step_1_result`, which is a
  `SourceResolved | None`) so a second chat against an already-applied source is a
  revise:
  ```python
                  source_chat_result = await resolve_step_1_source_chat_with_auto_drop(
                      site="post_guided_chat",
                      session_id=str(session_id),
                      user_id=user.user_id,
                      model=settings.composer_model,
                      user_message=body.message,
                      plugin_hint=plugin_hint,
                      current_source=guided.step_1_result,
                      temperature=settings.composer_temperature,
                      seed=settings.composer_seed,
                      recorder=recorder,
                  )
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest "tests/integration/web/composer/guided/test_step_chat.py::TestStep1SourceResolution" tests/integration/web/composer/guided/test_step_chat_source_driver.py -q
  ```
  Expected: all pass.

- [ ] **Step 4: Add the phase-entry drive test (the guard-widening regression).**
  This pins the mustFix that the LLM driver fires at STEP_1 phase entry — before
  any manual plugin selection — which is the LLM-primary premise and the p4
  passive-drive prerequisite. Append to `test_step_chat.py`'s
  `TestStep1SourceResolution` (it reuses the class's existing `composer_test_client`
  fixture and `_post_chat`-style helper; mirror the sibling test's drive):

  ```python
      def test_step_1_chat_drives_source_from_phase_entry(self, composer_test_client) -> None:
          """A chat submit at STEP_1 entry (no manual plugin pick) drives the source."""
          client = composer_test_client
          session_id = _create_session(client)
          # GET records the initial SINGLE_SELECT turn; NO _respond plugin pick.
          entry = _get_guided(client, session_id)
          assert entry["guided_session"]["step"] == "step_1_source"
          assert entry["next_turn"]["type"] == "single_select"
          with patch(
              "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
              new=AsyncMock(return_value=_fake_resolve_source_response_csv()),
          ):
              status, body = _post_chat(
                  client,
                  session_id,
                  message="make a csv source with a text column",
                  step_index="step_1_source",
              )
          assert status == 200, body
          # Drove the phase from entry: committed in place, stayed STEP_1, populated form.
          assert body["guided_session"]["step"] == "step_1_source"
          assert body["next_turn"]["type"] == "schema_form"
          assert body["next_turn"]["step_index"] == 0
          assert body["next_turn"]["payload"]["plugin"] == "csv"
          assert body["composition_state"]["sources"]["source"]["plugin"] == "csv"
  ```

  > Reuse the same `_litellm_acompletion` source stub the in-place test in Step 1
  > uses (factor it to a module helper `_fake_resolve_source_response_csv()` if the
  > existing test inlines it). `_create_session` / `_get_guided` / `_post_chat`
  > are the verbatim helpers already used by `TestStep1SourceResolution`; if the
  > class drives chat through a different helper, match it.

  Run to pass (this is RED before Step 2's guard widening, GREEN after):
  ```
  cd /home/john/elspeth && uv run pytest "tests/integration/web/composer/guided/test_step_chat.py::TestStep1SourceResolution::test_step_1_chat_drives_source_from_phase_entry" -x -q
  ```
  Expected: `1 passed`. Before the 2a guard widening this fails because the
  SCHEMA_FORM-only guard skips the SINGLE_SELECT entry record and the route falls
  through to advisory (`next_turn` is `None`).

- [ ] **Step 5: Add the HTTP apply-and-refresh invariant test.**
  Create `tests/integration/web/composer/guided/test_guided_refresh_applied.py`.
  This is the strong form deferred from Task 2.5: drive a STEP_1 source via
  `/guided/chat` (which now applies in place and clears staging), then GET /guided
  and assert the refresh turn MATCHES the apply-response turn (the invariant the
  staging-field clearing protects). It can only exist now that Task 3 lands the
  in-place apply + staging clear:

  ```python
  """p1 Task 3 — apply via /guided/chat then GET /guided agree (in-place state)."""

  from __future__ import annotations

  from unittest.mock import AsyncMock, patch

  from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient
  from tests.integration.web.composer.guided.test_step_chat import (
      _fake_resolve_source_response_csv,
  )
  from tests.integration.web.composer.guided.test_step_3_e2e import (
      _create_session,
      _get_guided,
  )


  def _post_chat(client: TestClient, session_id: str, *, message: str, step_index: str):
      resp = client.post(
          f"/api/sessions/{session_id}/guided/chat",
          json={"message": message, "step_index": step_index},
      )
      return resp.status_code, resp.json()


  def test_chat_apply_then_get_render_the_same_step_1_turn(composer_test_client: TestClient) -> None:
      client = composer_test_client
      session_id = _create_session(client)
      _get_guided(client, session_id)  # records the initial SINGLE_SELECT turn
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_resolve_source_response_csv()),
      ):
          status, apply_body = _post_chat(
              client, session_id, message="make a csv source", step_index="step_1_source"
          )
      assert status == 200, apply_body
      assert apply_body["guided_session"]["step"] == "step_1_source"
      apply_turn = apply_body["next_turn"]
      assert apply_turn["type"] == "schema_form"
      # Refresh: GET must re-render the SAME populated turn (staging fields cleared,
      # so _build_get_guided_turn hits the from-resolved sub-case, not the empty form).
      get_body = _get_guided(client, session_id)
      assert get_body["guided_session"]["step"] == "step_1_source"
      assert get_body["next_turn"]["type"] == "schema_form"
      assert get_body["next_turn"]["step_index"] == apply_turn["step_index"]
      assert get_body["next_turn"]["payload"]["plugin"] == apply_turn["payload"]["plugin"]
      assert get_body["next_turn"]["payload"]["prefilled"] == apply_turn["payload"]["prefilled"]
  ```

  > If `_fake_resolve_source_response_csv` is not yet factored out of
  > `test_step_chat.py` (Step 4 created it), define it locally in this file with
  > the same `resolve_source` tool-call shape. The load-bearing assertion is
  > `get_body…prefilled == apply_turn…prefilled` — if it fails, the staging fields
  > were NOT cleared (2c) and the GET fell through to the empty chosen-plugin form.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_guided_refresh_applied.py -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 6: Run the full guided integration suite to surface any other
  assertion that depended on STEP_1 chat auto-advancing.**
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/ -q
  ```
  Expected: green. If another test asserts `step_2_sink` after a STEP_1 chat
  apply, it is a locked-in buggy expectation from the old auto-advance contract —
  update its assertion to in-place (same edit as Step 1), do NOT revert the route
  change. List any such test in the commit body.

- [ ] **Step 7: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/sessions/routes/composer/guided.py tests/integration/web/composer/guided/test_step_chat.py tests/integration/web/composer/guided/test_guided_refresh_applied.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): STEP_1 chat applies source in place at phase entry

The /guided/chat STEP_1 apply branch only fired from the SCHEMA_FORM
sub-state and then auto-advanced to STEP_2. The LLM-primary revise model
requires (a) the driver to fire at phase entry (widen the SCHEMA_FORM-only
guard to the three valid STEP_1 turn types) and (b) applying in place:
commit the source, clear the staging fields, leave guided.step == STEP_1,
and re-render the populated source schema_form so the user can revise by
typing again. The answered-record audit leg stays gated on SCHEMA_FORM.
Drop the auto-advance and emit_step_advanced; pass guided.step_1_result as
current_source so a
repeat chat is a revise. Advancing stays an explicit /guided/respond
confirm. Updates the apply-precedent test to assert in-place behavior.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 4: STEP_2 + STEP_3 chat apply branches in post_guided_chat

> **Depends on Task 2 (sink driver), Task 2.5 (from-resolved builder), and Task 3
> (STEP_1 in-place pattern this mirrors).** Do those first.

**Files:**
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py` (`post_guided_chat`;
  add a STEP_2 apply branch and a STEP_3 apply branch alongside the STEP_1 branch,
  before the advisory fall-through at `:2005`)
- Test: `tests/integration/web/composer/guided/test_step_chat_apply.py` (Create)

**Interfaces:**
- Consumes: `resolve_step_2_sink_chat_with_auto_drop` / `Step2SinkChatResult`
  (Task 2, with `.assistant_message`); `handle_step_2_sink` (`steps.py:147`);
  `build_step_2_schema_form_turn_from_resolved(sink, catalog)` (Task 2.5);
  `solve_chain(*, model, source, sink,
  recipe_match=None, repair_context=None, recorder=None, temperature, seed) ->
  ChainProposal` (`chain_solver.py:130`) — STEP_3 calls THIS directly, NOT
  `solve_chain_with_auto_drop` (`_guided_solve_chain.py:68`), because that wrapper
  marks the session `solver_exhausted` (TERMINAL) and returns `(None,
  new_terminal_session)` on transient failure — which would brick the phase,
  violating the non-load-bearing-on-failure rule. STEP_3 wraps `solve_chain` in a
  local try/except that routes transient failures to advisory (`chat_result =
  None`) WITHOUT terminating;
  `build_step_3_propose_chain_turn(proposal)` (`emitters.py:312`);
  `handle_step_1_source` (already wired). STEP_2 needs `current_sink =
  guided.step_2_result`; STEP_3 needs `guided.step_1_result` (source) +
  `guided.step_2_result` (sink) for `solve_chain`.
- Produces (p2 consumes): STEP_2 actionable apply → `step` stays `"step_2_sink"`,
  `step_2_result` committed, `next_turn` = re-rendered STEP_2 `schema_form`.
  STEP_3 actionable apply → `step` stays `"step_3_transforms"`, `next_turn` = a
  `propose_chain` turn (the proposal is TRANSIENT — recorded as `step_3_proposal`
  for re-render but NOT committed via `handle_step_3_chain_accept`). Non-actionable
  → advisory prose, `next_turn=None`.

> **STEP_3 is propose, not commit.** The chat driver re-runs `solve_chain` and
> re-renders the `propose_chain` turn IN PLACE. It MUST NOT call
> `handle_step_3_chain_accept` (which commits the pipeline and advances to
> `STEP_4_WIRE`). Accepting the proposal stays on `/guided/respond` (the existing
> PROPOSE_CHAIN accept branch, `_helpers.py:3627`). Revise = type again →
> `solve_chain` with `repair_context` set to the user's revision message → new
> proposal turn. There is NO recipe-apply chat driver (deterministic confirm).

- [ ] **Step 1: Write the failing STEP_2 sink apply-in-place integration test.**
  Create `tests/integration/web/composer/guided/test_step_chat_apply.py`. Reuse
  the canonical `composer_test_client` fixture and the request helpers from the
  sibling `test_step_3_e2e.py` (`_create_session`, `_get_guided`, `_respond`,
  `_seed_blob`, `_outputs_path`) — import them or copy the verbatim drive helpers
  exactly as `test_step_3_e2e.py` does. Drive to STEP_2 (source committed via the
  manual single_select+schema_form path), then POST a STEP_2 chat that resolves a
  sink:

  ```python
  """p1 Task 4 — STEP_2/STEP_3 /guided/chat apply branches (in-place)."""

  from __future__ import annotations

  import json
  from types import SimpleNamespace
  from unittest.mock import AsyncMock, patch

  from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

  # Reuse the verbatim drive helpers from the sibling e2e test.
  from tests.integration.web.composer.guided.test_step_3_e2e import (
      _create_session,
      _get_guided,
      _outputs_path,
      _respond,
      _seed_blob,
  )


  def _post_chat(client: TestClient, session_id: str, *, message: str, step_index: str):
      resp = client.post(
          f"/api/sessions/{session_id}/guided/chat",
          json={"message": message, "step_index": step_index},
      )
      return resp.status_code, resp.json()


  def _fake_resolve_sink_response(path: str) -> SimpleNamespace:
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
                      content=None,
                      tool_calls=[
                          SimpleNamespace(
                              function=SimpleNamespace(
                                  name="resolve_sink",
                                  arguments=json.dumps(
                                      {
                                          "resolution": "sink",
                                          "outputs": [
                                              {
                                                  "plugin": "json",
                                                  "options": {"path": path, "schema": {"mode": "observed"}, "mode": "write", "collision_policy": "auto_increment"},
                                                  "required_fields": [],
                                                  "schema_mode": "observed",
                                              }
                                          ],
                                          "assistant_message": "Output set to JSON Lines.",
                                      }
                                  ),
                              )
                          )
                      ],
                  )
              )
          ]
      )


  def _drive_to_step_2(client: TestClient, session_id: str) -> None:
      _blob_id, storage_path = _seed_blob(client, session_id)
      _get_guided(client, session_id)
      _respond(client, session_id, chosen=["csv"])
      _respond(
          client,
          session_id,
          edited_values={
              "plugin": "csv",
              "options": {"path": storage_path, "schema": {"mode": "observed"}},
              "observed_columns": ["text", "note"],
              "sample_rows": [{"text": "Hello world", "note": "greeting"}],
          },
      )
      body = _get_guided(client, session_id)
      assert body["guided_session"]["step"] == "step_2_sink"


  def test_step_2_chat_applies_sink_in_place(composer_test_client: TestClient) -> None:
      client = composer_test_client
      session_id = _create_session(client)
      _drive_to_step_2(client, session_id)
      out = _outputs_path(client, "chat_out.jsonl")
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_resolve_sink_response(out)),
      ):
          status, body = _post_chat(
              client, session_id, message="write the rows to a jsonl file", step_index="step_2_sink"
          )
      assert status == 200, body
      # Apply-in-place: phase stays STEP_2, sink committed, form re-rendered.
      assert body["guided_session"]["step"] == "step_2_sink"
      assert body["next_turn"]["type"] == "schema_form"
      assert body["next_turn"]["step_index"] == 1
      outputs = body["composition_state"]["outputs"]
      assert any(o["plugin"] == "json" for o in outputs.values()) or any(
          o["plugin"] == "json" for o in outputs
      )
  ```

  > The `outputs` shape (dict-by-sink-name vs list) follows the existing
  > `_state_response` serialization; the `any(...)` over both forms tolerates
  > either. If the project's `composition_state.outputs` is unambiguously one
  > shape (confirm with a quick `_get_guided`/`GET state` echo in a scratch run),
  > tighten the assertion to that shape.

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py::test_step_2_chat_applies_sink_in_place -x -q
  ```
  Expected failure: `assert body["next_turn"]["type"] == "schema_form"` fails
  (the route currently falls through to advisory `next_turn=None` for STEP_2).

- [ ] **Step 2: Add the STEP_2 sink apply branch.**
  In `guided.py`, after the STEP_1 apply branch closes (the `return
  GuidedChatResponse(...)` at `:1992`) and BEFORE `if chat_result is None:`
  (`:2005`), add a STEP_2 branch. It mirrors the STEP_1 branch structure but:
  drives the sink driver, calls `handle_step_2_sink`, clears the STEP_2 staging
  fields, re-renders `build_step_2_schema_form_turn_from_resolved` (the POPULATED
  form, Task 2.5 — not the bare builder), stays at STEP_2, and on a resolution
  returns a `GuidedChatResponse`. On `None`/fallback it sets `chat_result` and
  falls through to advisory. Like STEP_1 (after the guard widening), this branch
  fires on ANY STEP_2 turn type — it is an `elif guided.step is
  GuidedStep.STEP_2_SINK:` with no turn-type guard — so the LLM drives the sink
  from STEP_2 phase entry too.

  ```python
              elif guided.step is GuidedStep.STEP_2_SINK:
                  sink_chat_result = await resolve_step_2_sink_chat_with_auto_drop(
                      site="post_guided_chat",
                      session_id=str(session_id),
                      user_id=user.user_id,
                      model=settings.composer_model,
                      user_message=body.message,
                      current_sink=guided.step_2_result,
                      temperature=settings.composer_temperature,
                      seed=settings.composer_seed,
                      recorder=recorder,
                  )
                  chat_result = sink_chat_result.fallback_chat
                  sink_resolution = sink_chat_result.sink_resolution
                  if sink_resolution is not None:
                      finished_at = datetime.now(UTC)
                      latency_ms = int((_perf_counter() - started_perf) * 1000)
                      data_dir = str(settings.data_dir) if settings.data_dir else None
                      handler_result = handle_step_2_sink(
                          state=state,
                          session=guided,
                          resolved=sink_resolution,
                          catalog=catalog,
                          data_dir=data_dir,
                      )
                      if not handler_result.tool_result.success:
                          # Non-actionable: the strict commit seam rejected the
                          # proposal. Fall back to advisory (no mutation).
                          chat_result = StepChatResult(
                              assistant_message=_SYNTHETIC_UNAVAILABLE_MESSAGE,
                              status=ComposerChatTurnStatus.SYNTHETIC_UNAVAILABLE,
                              latency_ms=latency_ms,
                              error_class="StepHandlerRejected",
                          )
                          sink_resolution = None
                  if sink_resolution is not None:
                      # Clear the STEP_2 staging fields on commit so the next GET
                      # hits the from-resolved sub-case (Task 2.5), not the empty
                      # chosen-plugin form. handle_step_2_sink sets step_2_result
                      # but does NOT clear these (verified steps.py:214). Mirrors
                      # the STEP_1 staging clear (Task 3 2c). No epoch bump.
                      guided = _replace(
                          handler_result.session,
                          step_2_chosen_plugin=None,
                          step_2_sink_intent=None,
                      )
                      state = handler_result.state
                      # Audit parity with STEP_1 (guided.py:1843-1861): if the prior
                      # STEP_2 turn was an answerable SCHEMA_FORM (the user was
                      # editing the sink form), stamp it answered so its TurnRecord
                      # does not stay response_hash=None forever. Gated on the turn
                      # type for the same reason STEP_1's leg is (a SINGLE_SELECT/
                      # MULTI_SELECT entry record has no schema-form answer to stamp).
                      if (
                          existing_record_for_chat is not None
                          and current_turn_type is TurnType.SCHEMA_FORM
                      ):
                          sink_turn_response: TurnResponse = {
                              "chosen": None,
                              "edited_values": {
                                  "plugin": sink_resolution.outputs[0].plugin,
                                  "options": dict(sink_resolution.outputs[0].options),
                                  "observed_columns": [],
                                  "sample_rows": [],
                              },
                              "custom_inputs": None,
                              "accepted_step_index": None,
                              "edit_step_index": None,
                              "control_signal": None,
                          }
                          sink_response_hash = stable_hash(sink_turn_response)
                          sink_answered_record = _replace(
                              existing_record_for_chat,
                              response_hash=sink_response_hash,
                              summary=_summarize_guided_response(TurnType.SCHEMA_FORM, sink_turn_response),
                          )
                          guided = _replace(
                              guided,
                              history=tuple(
                                  sink_answered_record if r is existing_record_for_chat else r
                                  for r in guided.history
                              ),
                          )
                          emit_turn_answered(
                              recorder,
                              step=GuidedStep.STEP_2_SINK,
                              turn_type=TurnType.SCHEMA_FORM,
                              response_hash=sink_response_hash,
                              response_payload_id=_store_guided_audit_payload(
                                  getattr(request.app.state, "payload_store", None), sink_turn_response
                              ),
                              control_signal=None,
                              composition_version=state.version,
                              actor=user.user_id,
                          )
                      ts_iso = finished_at.isoformat()
                      # POPULATED re-render from the committed sink (Task 2.5
                      # builder) — same turn GET /guided emits for this state.
                      next_turn = build_step_2_schema_form_turn_from_resolved(handler_result.session.step_2_result, catalog)
                      next_turn_type = TurnType(next_turn["type"])
                      next_payload_hash = stable_hash(next_turn["payload"])
                      new_record = TurnRecord(
                          step=GuidedStep.STEP_2_SINK,
                          turn_type=next_turn_type,
                          payload_hash=next_payload_hash,
                          response_hash=None,
                          emitter="server",
                      )
                      emit_turn_emitted(
                          recorder,
                          step=GuidedStep.STEP_2_SINK,
                          turn_type=next_turn_type,
                          payload_hash=next_payload_hash,
                          payload_payload_id=_store_guided_audit_payload(
                              getattr(request.app.state, "payload_store", None), next_turn["payload"]
                          ),
                          emitter="server",
                          composition_version=state.version,
                          actor=user.user_id,
                      )
                      user_turn = ChatTurn(
                          role=ChatRole.USER,
                          content=body.message,
                          seq=guided.chat_turn_seq,
                          step=GuidedStep.STEP_2_SINK,
                          ts_iso=ts_iso,
                      )
                      assistant_message = sink_chat_result.assistant_message or "Output configured."
                      assistant_turn = ChatTurn(
                          role=ChatRole.ASSISTANT,
                          content=assistant_message,
                          seq=guided.chat_turn_seq + 1,
                          step=GuidedStep.STEP_2_SINK,
                          ts_iso=ts_iso,
                      )
                      guided = _replace(
                          guided,
                          history=(*guided.history, new_record),
                          chat_history=(*guided.chat_history, user_turn, assistant_turn),
                          chat_turn_seq=guided.chat_turn_seq + 2,
                      )
                      recorder.record_chat_turn(
                          ComposerChatTurn(
                              step=GuidedStep.STEP_2_SINK.value,
                              initiator=ComposerChatInitiator.USER,
                              chat_turn_seq=user_turn.seq,
                              user_message_hash=stable_hash(body.message),
                              assistant_message_hash=stable_hash(assistant_message),
                              latency_ms=latency_ms,
                              model=settings.composer_model,
                              status=ComposerChatTurnStatus.SUCCESS,
                              started_at=started_at,
                              finished_at=finished_at,
                              error_class=None,
                          )
                      )
                      new_state = _replace(state, guided_session=guided)
                      sink_existing_meta: dict[str, Any] = {}
                      if state_record is not None and state_record.composer_meta is not None:
                          sink_existing_meta = dict(deep_thaw(state_record.composer_meta))
                      new_composer_meta = {**sink_existing_meta, "guided_session": guided.to_dict()}
                      state_d = new_state.to_dict()
                      state_data = CompositionStateData(
                          sources=state_d["sources"],
                          nodes=state_d["nodes"],
                          edges=state_d["edges"],
                          outputs=state_d["outputs"],
                          metadata_=state_d["metadata"],
                          is_valid=False,
                          validation_errors=None,
                          composer_meta=new_composer_meta,
                      )
                      state_record_out = await service.save_composition_state(
                          session_id, state_data, provenance="convergence_persist"
                      )
                      return GuidedChatResponse(
                          assistant_message=assistant_message,
                          guided_session=GuidedSessionResponse(
                              step=guided.step.value,
                              history=[
                                  TurnRecordResponse(
                                      step=r.step.value,
                                      turn_type=r.turn_type.value,
                                      payload_hash=r.payload_hash,
                                      response_hash=r.response_hash,
                                      summary=r.summary,
                                      emitter=r.emitter,
                                  )
                                  for r in guided.history
                              ],
                              terminal=None,
                              chat_history=[
                                  ChatTurnResponse(
                                      role=t.role.value,
                                      content=t.content,
                                      seq=t.seq,
                                      step=t.step.value,
                                      ts_iso=t.ts_iso,
                                  )
                                  for t in guided.chat_history
                              ],
                              chat_turn_seq=guided.chat_turn_seq,
                              profile=_workflow_profile_response(guided),
                          ),
                          next_turn=TurnPayloadResponse(
                              type=next_turn["type"],
                              step_index=next_turn["step_index"],
                              payload=dict(next_turn["payload"]),
                          ),
                          terminal=None,
                          composition_state=_state_response(state_record_out),
                      )
  ```

  > **DRY note:** the response-build + persist tail (CompositionStateData
  > construction, `save_composition_state`, `GuidedChatResponse` assembly) is
  > duplicated near-verbatim from the STEP_1 branch — ~150 lines tripled across
  > STEP_1/STEP_2/STEP_3 in a 2272-line handler. The extraction is REQUIRED (not
  > optional polish — see Step 6 below), but write the duplicated form FIRST so each
  > branch is test-isolated while you build it, then extract in Step 6. Do not skip
  > Step 6.

  **Import-extension sub-step (do this BEFORE writing the branch — the names below
  are called by the apply blocks and will NameError otherwise).** `guided.py`
  currently imports `handle_step_1_source` but NOT `handle_step_2_sink` (verified:
  the `from .._helpers import (...)` block `:7-93` lists `handle_step_1_source` at
  `:85` only). Add to that block:
  - `handle_step_2_sink` — exported from `_helpers.py` `__all__` (`:4392`).
  - `resolve_step_2_sink_chat_with_auto_drop` + `Step2SinkChatResult` (Task 2,
    `_guided_step_chat.py`) — re-export both through `_helpers.py` (add to its
    import block + `__all__`) if `guided.py` reaches sessions helpers via
    `.._helpers`, OR import directly `from elspeth.web.sessions._guided_step_chat
    import (...)` matching how `resolve_step_1_source_chat_with_auto_drop` reaches
    `guided.py` today (grep `grep -n "resolve_step_1_source_chat_with_auto_drop"
    src/elspeth/web/sessions/routes/_helpers.py` to see which path it uses, then
    mirror it).
  - `build_step_2_schema_form_turn_from_resolved` (Task 2.5) — already added to the
    `.._helpers` block in Task 2.5 Step 4.
  - `StepChatResult` / `_SYNTHETIC_UNAVAILABLE_MESSAGE` / `ComposerChatTurnStatus`
    — grep first (`grep -n "_SYNTHETIC_UNAVAILABLE_MESSAGE\|StepChatResult"
    src/elspeth/web/sessions/routes/composer/guided.py`); add via the same path the
    STEP_1 branch already uses for its fallback.
  Do NOT add `handle_step_3_chain_accept`: the STEP_3 branch is propose-only and
  calls `solve_chain` directly (never the accept handler), so importing it would be
  an unused import that Task 6 Step 2's ruff check (F401) fails on.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py::test_step_2_chat_applies_sink_in_place -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Write the failing STEP_3 propose-in-place test.**
  Append to `test_step_chat_apply.py`. Reaching STEP_3 in this tree REQUIRES a
  chain solve (the STEP_2→STEP_3 advance fires `solve_chain` — verified
  `test_step_3_e2e.py:164,184-189`: `_drive_to_step_3_propose_chain` runs UNDER its
  OWN `chain_solver._litellm_acompletion` patch). So there is no manual no-solve
  drive. To avoid nesting two patches on the same symbol, run the drive (with its
  own patch) and the STEP_3 chat (with this test's patch) in SEPARATE, non-nested
  `with patch(...)` blocks — the drive's patch exits before this test's opens.
  Import the verbatim drive helper:

  ```python
  from tests.integration.web.composer.guided.test_step_3_e2e import (
      _drive_to_step_3_propose_chain,
      _fake_llm_response_for_passthrough,
  )


  def _fake_chain_response() -> SimpleNamespace:
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
                      content=None,
                      tool_calls=[
                          SimpleNamespace(
                              function=SimpleNamespace(
                                  name="emit_turn",
                                  arguments=json.dumps(
                                      {
                                          "turn_type": "propose_chain",
                                          "payload": {
                                              "steps": [
                                                  {
                                                      "plugin": "passthrough",
                                                      "options": {},
                                                      "rationale": "no transform needed; pass rows through",
                                                  }
                                              ],
                                              "why": "the rows already match the sink",
                                          },
                                      }
                                  ),
                              )
                          )
                      ],
                  )
              )
          ]
      )


  def test_step_3_chat_reproposes_in_place_without_committing(composer_test_client: TestClient) -> None:
      client = composer_test_client
      session_id = _create_session(client)
      # Drive to STEP_3 under the HELPER's own chain-solver patch (which exits at
      # the end of this block), so it does NOT nest with this test's patch below.
      with patch(
          "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
          new_callable=AsyncMock,
          return_value=_fake_llm_response_for_passthrough(),
      ):
          _drive_to_step_3_propose_chain(client, session_id)
      # Now a fresh, non-nested patch controls the STEP_3 chat re-solve.
      with patch(
          "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_chain_response()),
      ):
          status, body = _post_chat(
              client, session_id, message="actually just pass the rows through", step_index="step_3_transforms"
          )
      assert status == 200, body
      # In-place: phase stays STEP_3, a fresh propose_chain turn is re-rendered,
      # and the pipeline is NOT committed/advanced to wire.
      assert body["guided_session"]["step"] == "step_3_transforms"
      assert body["next_turn"]["type"] == "propose_chain"
      assert body["terminal"] is None
  ```

  > **Drive helper — avoid the nested-patch hazard (verified).**
  > `_drive_to_step_3_propose_chain` (`test_step_3_e2e.py:121`) reaches STEP_3 by
  > driving the STEP_2→STEP_3 advance, which fires `solve_chain` — so it manages
  > its OWN patch of `chain_solver._litellm_acompletion` (the helper's callers wrap
  > it, `test_step_3_e2e.py:184-189`). There is NO manual no-solve path to STEP_3
  > in this tree. The fix is therefore to keep the drive and the STEP_3-chat
  > re-solve in SEPARATE, sequential `with patch(...)` blocks (as written above) so
  > the two patches never overlap — the drive's patch is torn down before this
  > test's opens. Do NOT nest them. `_fake_llm_response_for_passthrough` is the
  > helper's own stub shape (`test_step_3_e2e.py`); import it alongside the drive
  > helper.

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py::test_step_3_chat_reproposes_in_place_without_committing -x -q
  ```
  Expected failure: the STEP_3 chat falls through to advisory (`next_turn` is
  `None`), so `body["next_turn"]["type"]` raises `TypeError`/assertion fails.

- [ ] **Step 4: Add the STEP_3 propose-in-place branch.**
  In `guided.py`, after the STEP_2 branch and before `if chat_result is None:`,
  add a STEP_3 branch that re-runs `solve_chain` DIRECTLY (passing
  `guided.step_1_result` as source, `guided.step_2_result` as sink, and the user's
  message as `repair_context` so it's a revise) and re-renders the proposal turn.
  It records `step_3_proposal` for re-render but does NOT call
  `handle_step_3_chain_accept`. It does NOT use `solve_chain_with_auto_drop` —
  that wrapper TERMINATES the session (`solver_exhausted`) on transient failure,
  which would brick the phase; STEP_3 chat must be non-load-bearing. Wrap
  `solve_chain` in a local try/except over the same transient set the auto-drop
  wrapper absorbs (LiteLLM API/auth/bad-request, `TimeoutError`, malformed-shape
  `IndexError`/`AttributeError`/`json.JSONDecodeError`, and
  `ChainSolverResponseShapeError`) and route those to advisory (`chat_result =
  None`) WITHOUT terminating; let `InvariantError`/`ValueError` propagate (real
  bugs), matching the existing solver discipline:

  ```python
              elif guided.step is GuidedStep.STEP_3_TRANSFORMS:
                  if guided.step_1_result is None or guided.step_2_result is None:
                      # Cannot propose a chain without a committed source + sink;
                      # fall through to advisory prose (no mutation).
                      chat_result = None
                  else:
                      from elspeth.web.composer.guided.chain_solver import (
                          ChainSolverResponseShapeError,
                          solve_chain,
                      )

                      try:
                          from litellm.exceptions import APIError as _LLMAPIError
                          from litellm.exceptions import AuthenticationError as _LLMAuthError
                          from litellm.exceptions import BadRequestError as _LLMBadReq

                          # repair_context flips solve_chain's prompt to a
                          # repair addendum (chain_solver.py:174). On the FIRST
                          # STEP_3 chat (no prior proposal) this is an initial
                          # propose, not a repair — gate it so a cold start is
                          # not mis-framed as error-correction.
                          repair_context = body.message if guided.step_3_proposal is not None else None
                          proposal = await solve_chain(
                              model=settings.composer_model,
                              source=guided.step_1_result,
                              sink=guided.step_2_result,
                              repair_context=repair_context,
                              recorder=recorder,
                              temperature=settings.composer_temperature,
                              seed=settings.composer_seed,
                          )
                      except (
                          _LLMAPIError,
                          _LLMAuthError,
                          _LLMBadReq,
                          TimeoutError,
                          IndexError,
                          AttributeError,
                          json.JSONDecodeError,
                          ChainSolverResponseShapeError,
                      ):
                          # Non-load-bearing: a transient solve failure on the
                          # STEP_3 chat path must NOT terminate the session (that
                          # is what solve_chain_with_auto_drop would do). Fall back
                          # to advisory prose with NO mutation.
                          proposal = None
                      if proposal is None:
                          chat_result = None  # advisory fall-through
                      else:
                          finished_at = datetime.now(UTC)
                          latency_ms = int((_perf_counter() - started_perf) * 1000)
                          # Record the transient proposal for in-place re-render.
                          # NOT committed: handle_step_3_chain_accept (commit +
                          # advance to WIRE) stays on /guided/respond.
                          guided = _replace(guided, step_3_proposal=proposal)
                          state = _replace(state, guided_session=guided)
                          next_turn = build_step_3_propose_chain_turn(proposal)
                          next_turn_type = TurnType(next_turn["type"])
                          next_payload_hash = stable_hash(next_turn["payload"])
                          new_record = TurnRecord(
                              step=GuidedStep.STEP_3_TRANSFORMS,
                              turn_type=next_turn_type,
                              payload_hash=next_payload_hash,
                              response_hash=None,
                              emitter="server",
                          )
                          emit_turn_emitted(
                              recorder,
                              step=GuidedStep.STEP_3_TRANSFORMS,
                              turn_type=next_turn_type,
                              payload_hash=next_payload_hash,
                              payload_payload_id=_store_guided_audit_payload(
                                  getattr(request.app.state, "payload_store", None), next_turn["payload"]
                              ),
                              emitter="server",
                              composition_version=state.version,
                              actor=user.user_id,
                          )
                          ts_iso = finished_at.isoformat()
                          assistant_message = "Here is an updated proposal."
                          user_turn = ChatTurn(
                              role=ChatRole.USER,
                              content=body.message,
                              seq=guided.chat_turn_seq,
                              step=GuidedStep.STEP_3_TRANSFORMS,
                              ts_iso=ts_iso,
                          )
                          assistant_turn = ChatTurn(
                              role=ChatRole.ASSISTANT,
                              content=assistant_message,
                              seq=guided.chat_turn_seq + 1,
                              step=GuidedStep.STEP_3_TRANSFORMS,
                              ts_iso=ts_iso,
                          )
                          guided = _replace(
                              guided,
                              history=(*guided.history, new_record),
                              chat_history=(*guided.chat_history, user_turn, assistant_turn),
                              chat_turn_seq=guided.chat_turn_seq + 2,
                          )
                          recorder.record_chat_turn(
                              ComposerChatTurn(
                                  step=GuidedStep.STEP_3_TRANSFORMS.value,
                                  initiator=ComposerChatInitiator.USER,
                                  chat_turn_seq=user_turn.seq,
                                  user_message_hash=stable_hash(body.message),
                                  assistant_message_hash=stable_hash(assistant_message),
                                  latency_ms=latency_ms,
                                  model=settings.composer_model,
                                  status=ComposerChatTurnStatus.SUCCESS,
                                  started_at=started_at,
                                  finished_at=finished_at,
                                  error_class=None,
                              )
                          )
                          new_state = _replace(state, guided_session=guided)
                          chain_existing_meta: dict[str, Any] = {}
                          if state_record is not None and state_record.composer_meta is not None:
                              chain_existing_meta = dict(deep_thaw(state_record.composer_meta))
                          new_composer_meta = {**chain_existing_meta, "guided_session": guided.to_dict()}
                          state_d = new_state.to_dict()
                          state_data = CompositionStateData(
                              sources=state_d["sources"],
                              nodes=state_d["nodes"],
                              edges=state_d["edges"],
                              outputs=state_d["outputs"],
                              metadata_=state_d["metadata"],
                              is_valid=False,
                              validation_errors=None,
                              composer_meta=new_composer_meta,
                          )
                          state_record_out = await service.save_composition_state(
                              session_id, state_data, provenance="convergence_persist"
                          )
                          return GuidedChatResponse(
                              assistant_message=assistant_message,
                              guided_session=GuidedSessionResponse(
                                  step=guided.step.value,
                                  history=[
                                      TurnRecordResponse(
                                          step=r.step.value,
                                          turn_type=r.turn_type.value,
                                          payload_hash=r.payload_hash,
                                          response_hash=r.response_hash,
                                          summary=r.summary,
                                          emitter=r.emitter,
                                      )
                                      for r in guided.history
                                  ],
                                  terminal=None,
                                  chat_history=[
                                      ChatTurnResponse(
                                          role=t.role.value,
                                          content=t.content,
                                          seq=t.seq,
                                          step=t.step.value,
                                          ts_iso=t.ts_iso,
                                      )
                                      for t in guided.chat_history
                                  ],
                                  chat_turn_seq=guided.chat_turn_seq,
                                  profile=_workflow_profile_response(guided),
                              ),
                              next_turn=TurnPayloadResponse(
                                  type=next_turn["type"],
                                  step_index=next_turn["step_index"],
                                  payload=dict(next_turn["payload"]),
                              ),
                              terminal=None,
                              composition_state=_state_response(state_record_out),
                          )
  ```

  > **Why a direct `solve_chain` call, not the auto-drop wrapper (verified).**
  > `solve_chain_with_auto_drop` (`_guided_solve_chain.py:68-83`) takes
  > `composition_version: int` + a REQUIRED `recorder`, and on transient failure
  > calls `mark_solver_exhausted` and returns `(None, new_TERMINAL_session)` —
  > i.e. it BRICKS the guided session into freeform-drop. That is correct for the
  > manual STEP_2/STEP_3 dispatch (where exhaustion is the designed outcome) but
  > WRONG for the chat path, which must stay non-load-bearing. So STEP_3 chat calls
  > `solve_chain` directly and catches the same transient set into an advisory
  > fall-through. `solve_chain` itself records exactly one `ComposerLLMCall` via the
  > `recorder` you pass, so audit is preserved. Verify the transient exception set
  > against `solve_chain`'s own `except` clauses (the typed-except block begins at
  > `chain_solver.py:282` — the `except (IndexError, AttributeError,
  > json.JSONDecodeError, ChainSolverResponseShapeError)` clause — and runs through
  > `:328`; the `repair_context` gate is at `:174-175`; the
  > `ChainSolverResponseShapeError` is raised in the validation block at `:215-225`)
  > before writing.

  Add `build_step_3_propose_chain_turn` to the route imports if absent (grep
  first). `solve_chain` + `ChainSolverResponseShapeError` are imported inline in
  the branch above (local import keeps the route's top-of-file import churn down
  and is the pattern the existing branches use for litellm exceptions).

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py -x -q
  ```
  Expected: `2 passed`.

- [ ] **Step 5: Add an advisory-fall-back test for a non-actionable STEP_2 message
  (no mutation, next_turn=None).**
  Append to `test_step_chat_apply.py`:

  ```python
  def test_step_2_chat_prose_is_advisory_no_mutation(composer_test_client: TestClient) -> None:
      client = composer_test_client
      session_id = _create_session(client)
      _drive_to_step_2(client, session_id)
      before = _get_guided(client, session_id)
      prose = SimpleNamespace(
          choices=[SimpleNamespace(message=SimpleNamespace(content="A sink writes rows out.", tool_calls=None))]
      )
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=prose),
      ):
          status, body = _post_chat(
              client, session_id, message="what is a sink?", step_index="step_2_sink"
          )
      assert status == 200, body
      # Advisory: no mutation, no next_turn, phase unchanged.
      assert body["next_turn"] is None
      assert body["guided_session"]["step"] == "step_2_sink"
      # No outputs committed by an advisory message.
      after = _get_guided(client, session_id)
      assert before["composition_state"]["outputs"] == after["composition_state"]["outputs"]


  def test_step_2_chat_from_schema_form_stamps_prior_record_answered(composer_test_client: TestClient) -> None:
      """A STEP_2 chat apply from the SCHEMA_FORM sub-state stamps the prior record
      answered (response_hash non-None) — audit parity with STEP_1."""
      client = composer_test_client
      session_id = _create_session(client)
      _drive_to_step_2(client, session_id)
      # Advance to the STEP_2 SCHEMA_FORM sub-state by picking a sink plugin
      # (SINGLE_SELECT -> SCHEMA_FORM), so the existing record is a SCHEMA_FORM turn.
      _respond(client, session_id, chosen=["json"])
      before = _get_guided(client, session_id)
      assert before["next_turn"]["type"] == "schema_form"
      out = _outputs_path(client, "answered_out.jsonl")
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_resolve_sink_response(out)),
      ):
          status, body = _post_chat(
              client, session_id, message="write the rows to a jsonl file", step_index="step_2_sink"
          )
      assert status == 200, body
      assert body["guided_session"]["step"] == "step_2_sink"
      # The prior STEP_2 SCHEMA_FORM record is now answered (not response_hash=None).
      step_2_schema_records = [
          r
          for r in body["guided_session"]["history"]
          if r["step"] == "step_2_sink" and r["turn_type"] == "schema_form"
      ]
      assert step_2_schema_records, body["guided_session"]["history"]
      # The earliest STEP_2 schema_form record (the answered one) carries a hash;
      # the freshly re-rendered one is the new unanswered turn.
      assert any(r["response_hash"] is not None for r in step_2_schema_records)
  ```

  > If `_drive_to_step_2` already lands at a SCHEMA_FORM sub-state in your tree (it
  > should land at SINGLE_SELECT entry — confirm with the `before` GET), drop the
  > extra `_respond(chosen=["json"])`. The load-bearing assertion is that the
  > SCHEMA_FORM record the user was editing gets `response_hash` set — proving the
  > Step 2 audit leg (`emit_turn_answered`) fired.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py -q
  ```
  Expected: `4 passed`.

- [ ] **Step 6: Extract the shared apply-response tail (REQUIRED, not optional).**
  Tasks 3 and 4 now triple a ~150-line persist+response tail across the STEP_1,
  STEP_2, and STEP_3 apply branches (CompositionStateData construction,
  `save_composition_state`, `GuidedChatResponse`/`GuidedSessionResponse` assembly).
  That is real debt in code this plan writes, and "optional polish with no
  enforcement" means it never happens — so it is in scope here. Extract ONE private
  async helper and rewrite all three branches as thin callers:
  ```python
  async def _build_guided_chat_apply_response(
      *,
      guided: GuidedSession,
      state: CompositionState,
      next_turn: Turn,
      assistant_message: str,
      service: ComposerService,
      session_id: UUID,
      state_record: CompositionStateRecord | None,
  ) -> GuidedChatResponse:
      """Persist the in-place-applied state and build the chat-apply response.

      Shared tail for the STEP_1/STEP_2/STEP_3 /guided/chat apply branches: build
      CompositionStateData from `state` + the committed `guided`, persist via
      `save_composition_state(provenance="convergence_persist")`, and assemble the
      GuidedChatResponse with the populated `next_turn`. `guided` and `state` must
      already carry the committed result, the appended history/chat_history, and
      cleared staging fields — this helper does NOT mutate them.
      """
      # (move the existing STEP_1 persist+return tail body here verbatim, taking
      # next_turn/assistant_message as params; the three branches each build
      # next_turn + guided + state, then `return await _build_guided_chat_apply_response(...)`.)
  ```
  > **Sequencing:** TDD-write each branch with its own duplicated tail first (so the
  > branch tests pass in isolation), then do this extraction as the final code
  > change and re-run all three apply tests + the STEP_1 suite to prove behavior is
  > unchanged. Confirm the helper signature carries everything the three branches
  > need (the STEP_3 branch's `terminal=None` and STEP_1's source-committed
  > `composition_state` are both covered because the tail reads from
  > `state_record_out`, not branch locals). Verify the extracted helper passes mypy.

  Run all three apply tests after extracting:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_step_chat_apply.py tests/integration/web/composer/guided/test_step_chat.py -q
  ```
  Expected: green (no behavior change from the extraction).

- [ ] **Step 7: Run the full guided integration suite + the chat-solver units.**
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/ -q
  ```
  Expected: green. Any failure asserting the OLD advisory-only `/guided/chat`
  behavior for STEP_2/STEP_3 (e.g. a test that posts a STEP_2 chat and asserts
  `next_turn is None` for a message that NOW resolves a sink) is a locked-in
  expectation — update it to the actionable-apply contract. The advisory invariant
  test `test_echoes_unchanged_guided_session` (`test_step_chat.py:202`) MUST stay
  green unchanged (its inputs are advisory). If it fails, your branch is mutating
  on a prose message — fix the branch, not the test.

- [ ] **Step 8: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/sessions/routes/composer/guided.py tests/integration/web/composer/guided/test_step_chat_apply.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): STEP_2 sink + STEP_3 transform /guided/chat apply

Generalize the STEP_1 chat-apply pattern to STEP_2 and STEP_3. STEP_2
drives the sink driver and commits via handle_step_2_sink, re-rendering
the sink form in place. STEP_3 re-runs solve_chain (user message as
repair_context) and re-renders the propose_chain turn IN PLACE without
committing — accept stays on /guided/respond. Both stay at their current
step (apply-in-place); non-actionable prose/failure falls back to advisory
(next_turn=None, no mutation). 400/409 guards preserved.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 5: Source-driver scrape-routing assertion (p1→p4 output contract)

**Files:**
- Test: `tests/integration/web/composer/guided/test_source_driver_scrape_routing.py` (Create)

**Interfaces:**
- Consumes: `maybe_resolve_step_1_source_chat` (Task 1, stubbed at the provider
  chokepoint); `SourceResolved`; `match_recipe(source, sink)` (`recipe_match.py:489`);
  `_web_scrape_predicate` (`recipe_match.py:230`); `SinkResolved` /
  `SinkOutputResolved`.
- Produces: a regression that the source driver's URL-list output routes
  `web_scrape` into the transform stage (no `web_scraper` source). This pins the
  p1→p4 contract: p4's tutorial worked example consumes EXACTLY this shape.

> No new implementation here — this task asserts the existing scrape-routing
> already works for a driver-produced URL-row source. If the assertion fails, the
> source driver is producing the wrong shape (a scraper source) and the defect is
> in Task 1, not in the recipe layer.

- [ ] **Step 1: Write the test that a driver-produced URL-row source matches the
  web_scrape recipe.**
  Create `tests/integration/web/composer/guided/test_source_driver_scrape_routing.py`:

  ```python
  """p1 Task 5 — a driver-produced URL-row source routes web_scrape into transforms."""

  from __future__ import annotations

  import json
  from types import SimpleNamespace
  from unittest.mock import AsyncMock, patch

  import pytest

  from elspeth.web.composer.guided.chat_solver import maybe_resolve_step_1_source_chat
  from elspeth.web.composer.guided.recipe_match import match_recipe
  from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved, SourceResolved


  def _fake_url_source_response() -> SimpleNamespace:
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
                      content=None,
                      tool_calls=[
                          SimpleNamespace(
                              function=SimpleNamespace(
                                  name="resolve_source",
                                  arguments=json.dumps(
                                      {
                                          "resolution": "source",
                                          "plugin": "json",
                                          "filename": "urls.json",
                                          "mime_type": "application/json",
                                          "content": json.dumps(
                                              [
                                                  {"url": "https://example.test/project-1.html"},
                                                  {"url": "https://example.test/project-2.html"},
                                              ]
                                          ),
                                          "options": {"schema": {"mode": "observed"}},
                                          "observed_columns": ["url"],
                                          "sample_rows": [{"url": "https://example.test/project-1.html"}],
                                          "assistant_message": "I set up a URL list source.",
                                      }
                                  ),
                              )
                          )
                      ],
                  )
              )
          ]
      )


  @pytest.mark.asyncio
  async def test_url_source_driver_output_matches_web_scrape_recipe() -> None:
      with patch(
          "elspeth.web.composer.guided.chat_solver._litellm_acompletion",
          new=AsyncMock(return_value=_fake_url_source_response()),
      ):
          resolution = await maybe_resolve_step_1_source_chat(
              model="anthropic/claude-sonnet-4.6",
              user_message="scrape these project pages and pull out the name and top risk",
              plugin_hint="json",
              current_source=None,
              temperature=None,
              seed=None,
          )
      assert resolution is not None
      assert resolution.plugin == "json"
      assert "url" in resolution.observed_columns

      # The driver's output, once blob-backed, is the URL-row source the
      # web_scrape recipe routes. _web_scrape_predicate keys on the json/csv
      # plugin + an observed `url` column + blob_ref in options + a single
      # json sink. Simulate the post-blob source shape:
      source = SourceResolved(
          plugin=resolution.plugin,
          options={**dict(resolution.options), "blob_ref": "blob-123"},
          observed_columns=resolution.observed_columns,
          sample_rows=resolution.sample_rows,
      )
      sink = SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": "out.jsonl"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )
      match = match_recipe(source, sink)
      assert match is not None
      assert match.recipe_name == "web-scrape-llm-rate-jsonl"
  ```

  > `RecipeMatch.recipe_name` is the verified field (`recipe_match.py:104`; the
  > dataclass also carries `slots` and `unsatisfied_slots`). The recipe is
  > registered as `"web-scrape-llm-rate-jsonl"` (`recipes.py:818`). The
  > load-bearing assertion is "the web_scrape recipe matched"; if the registered
  > name string differs in your tree, read `recipes.py` near the `RecipeSpec`
  > registration and use the real id.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_source_driver_scrape_routing.py -x -q
  ```
  Expected: `1 passed`. If `match_recipe` returns `None`, read
  `_web_scrape_predicate` (`recipe_match.py:230-263`) and confirm the exact
  precondition the predicate checks (the `_URL_ROW_SOURCE_PLUGINS` set at `:216`,
  the `blob_ref` key, the `url` observed column, the single-json-output check) —
  then make the simulated `source`/`sink` shape satisfy it. Do NOT change the
  predicate; this test exists to lock the driver→recipe contract.

- [ ] **Step 2: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add tests/integration/web/composer/guided/test_source_driver_scrape_routing.py && git commit -m "$(cat <<'EOF'
test(composer/guided): pin source-driver -> web_scrape recipe routing

Assert a URL-list resolve_source output (json/csv + observed url column +
blob_ref) matches the web-scrape-llm-rate-jsonl recipe so web_scrape lands
in the transform stage, not as a (non-existent) web_scraper source. This is
the p1->p4 source-driver output contract the tutorial worked example
consumes.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 6: Slice-boundary reconciliation — full suite, lint surface, wardline

**Files:** none new (verification + reconciliation only).

**Interfaces:** Consumes everything Tasks 1–5 produced. Produces: a green slice
ready for operator re-sign + push.

- [ ] **Step 1: Run the full composer/guided + sessions test surface.**
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/ tests/unit/web/composer/ -q
  ```
  Expected: green. Any red here is a real regression from this slice — fix it (or,
  if it is a locked-in old-contract expectation, update the assertion per the
  Global Constraints, citing which task changed the behavior).

- [ ] **Step 2: Run the static-analysis surface this slice touches (ruff + mypy on
  the changed files).**
  ```
  cd /home/john/elspeth && uv run ruff check src/elspeth/web/composer/guided/chat_solver.py src/elspeth/web/composer/guided/emitters.py src/elspeth/web/sessions/_guided_step_chat.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py && uv run mypy src/elspeth/web/composer/guided/chat_solver.py src/elspeth/web/composer/guided/emitters.py src/elspeth/web/sessions/_guided_step_chat.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Expected: no new ruff findings; mypy clean on the changed files (a pre-existing
  baseline red in `guided.py` that this slice did not introduce is operator-owed —
  note it, do not chase it). Fix any NEW finding this slice introduced.

- [ ] **Step 3: Trust-boundary scan.** The chat-driven apply path consumes Tier-3
  free text → validated config via the existing strict seams; confirm no new
  unguarded boundary.
  ```
  cd /home/john/elspeth && wardline scan . --fail-on ERROR
  ```
  Expected: exit 0. If a finding fires on the new `resolve_sink` parse boundary,
  fix it at the boundary (the parser is the boundary — it already strictly
  validates every field), not the sink. Re-scan to confirm exit 0.

- [ ] **Step 4: Reconcile the slice diff and surface owed re-signs to the
  operator.**
  ```
  cd /home/john/elspeth && git --no-pager diff --stat release/0.7.0..HEAD
  ```
  Confirm only the files this plan owns changed:
  `chat_solver.py`, `_guided_step_chat.py`, `routes/_helpers.py`,
  `routes/composer/guided.py`, and the new test files. **No re-sign gate trips for
  this plan's file set.** The plugin `source_file_hash` gate keys on `plugins/*`
  (none edited); the tier-model `web.yaml` allowlist tracks
  `interpretation_state.py` and `state.py` (neither edited). `_helpers.py` and
  `routes/composer/guided.py` are not in either tracked set. So the Global
  Constraints re-sign hedging is INAPPLICABLE here — do not re-sign anything, and
  do not flag a re-sign chore unless a later step incidentally edited a plugin or a
  `web.yaml`-tracked web file. (If `git diff --stat` shows any `plugins/*`,
  `interpretation_state.py`, or `state.py` change, THEN surface the owed re-sign;
  otherwise there is none.) State in your completion report: branch is
  `release/0.7.0`, agent signed nothing, no re-sign owed (or the specific file if
  one slipped in), and the test counts added.

---

## Cross-plan interface summary (for p2 and p4)

**p1 PRODUCES (p2 CONSUMES — the apply/response contract):**
- `POST /guided/chat` may now APPLY. Wire shapes unchanged. On an actionable
  apply: `guided_session.step` is UNCHANGED (in-place), `composition_state`
  reflects the committed `step_N_result`, and `next_turn` carries the re-rendered
  current-phase form POPULATED from `step_N_result` (STEP_1 `schema_form` with the
  committed source options prefilled, STEP_2 `schema_form` with the committed sink
  options prefilled, STEP_3 `propose_chain` with the fresh proposal) — the same
  populated turn GET /guided emits for the applied state. On non-actionable input:
  `assistant_message` is advisory prose
  and `next_turn=None`. p2's frontend rides the existing `chatGuided` store action
  (`sessionStore.ts:1333-1392`) which merges `next_turn ?? prev`; the intent box
  uses `/guided/chat`, the structured form uses `/guided/respond`. No new wire
  field.

**p1 PRODUCES (p4 CONSUMES — the source-driver output shape):**
- The source driver emits an inline URL-row `SourceResolved` (`plugin in
  {"json","csv"}`, observed `url` column, `blob_ref` materialized downstream). It
  does NOT propose a `web_scraper` source; `match_recipe` /
  `_web_scrape_predicate` route `web_scrape` into the transform stage via the
  `web-scrape-llm-rate-jsonl` recipe. p4's tutorial worked example feeds the
  runtime-derived synthetic URLs as the concrete dataset into this driver. The
  driver also accepts `current_source` for in-place revise.

**p1 does NOT own:** the frontend reorder/concern-B (p2), the prompt-shield A/B/C
surface (p3), the tutorial base-URL seam / synthetic pages / constants (p4). No
schema/epoch bump in any of those either (see the contract §3/§4).
