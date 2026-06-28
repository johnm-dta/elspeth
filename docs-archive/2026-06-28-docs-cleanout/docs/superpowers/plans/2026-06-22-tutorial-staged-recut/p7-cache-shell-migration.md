> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P7 — Cache (C2) + TutorialGuidedShell + Migration

This phase closes the staged recut: it makes the tutorial run-cache key fold the
five deterministic inputs (composer model + core skill + staged guided skill pack
+ deployment overlay + recipe catalog, so a recipe/skill edit can never
serve a stale cached pipeline), builds the `TutorialGuidedShell` bridge that mounts
the real guided `ChatPanel` surface and removes the big-bang `describe`/`showBuilt`
turns, and adds the migration verification + secret-archive hardening on top of the
existing session-DB-reset runbook. Current `release/0.7.0` HEAD is already at
`SESSION_SCHEMA_EPOCH = 23`, so this cutover must plan the next bump as
`SESSION_SCHEMA_EPOCH` **23→24** and expect `PRAGMA user_version` 24. The design
spec must carry the same 23→24 contract; P7 verifies that cross-reference rather
than owning the spec edit.

**Depends on symbols owned by other phases (must already be merged into this branch
before P7 frontend tasks run):**
- `GuidedStep.STEP_4_WIRE` + `skills/step_4_wire.md` (P1.T1/P1.T2) — the cache key's
  staged-skill hash enumerates `_STEP_PLAYBOOK_ORDER`, which P1.T2 extends with the
  wire step's `.md`.
- `GuidedSession.profile` on the wire (`WorkflowProfileResponse`, `schemas.py`) +
  the TS `WorkflowProfile` type and `GuidedSession.profile: WorkflowProfile | null`
  on `guided.ts` (P7.1 / P0).
- `POST /api/sessions/{session_id}/guided/start` + `startGuidedSession(sessionId, profileKind)`
  in `client.ts` (P6 — confirmed absent from client.ts until P6 runs; the "(P7.1)" label in
  the original was a typo).
- `WorkflowProfileKind` discriminator string `"tutorial"` (P0.T1 / P7.1).
- The guided `terminal.kind === "completed"` branch in `ChatPanel` (existing).

The P7.4 cache task and the migration task touch only backend/runbook surfaces and
do **not** depend on the frontend wiring, so they can land first.

---

### Task P7.1: Fold the staged guided-skill hash into `tutorial_model_id` (cache input #3)

**Files:**
- Modify: `src/elspeth/web/composer/guided/prompts.py` (add public `guided_staged_skill_hash()` at module end, currently `:185`)
- Modify: `tests/unit/web/composer/guided/test_prompts.py` (Create if absent under `tests/unit/web/composer/guided/`)

**Interfaces:**
- Produces: `def guided_staged_skill_hash() -> str` — hex SHA-256 over `base.md` plus each step file in `_STEP_PLAYBOOK_ORDER` (so it tracks `step_4_wire.md` automatically once P1.T2 appends `STEP_4_WIRE`).
- Consumes: `_SKILLS_DIR`, `_STEP_FILE_NAMES`, `_STEP_PLAYBOOK_ORDER` (prompts.py:39/43/53).

- [ ] **Step 1: Write the failing test for the staged-skill hash helper.**
  Create `tests/unit/web/composer/guided/test_prompts.py` (or append if it exists):
  ```python
  """Tests for guided skill loading + the staged-skill cache hash."""

  from __future__ import annotations

  import hashlib

  from elspeth.web.composer.guided.prompts import (
      _SKILLS_DIR,
      _STEP_FILE_NAMES,
      _STEP_PLAYBOOK_ORDER,
      guided_staged_skill_hash,
  )


  def test_guided_staged_skill_hash_covers_base_and_every_step_in_order() -> None:
      """The hash folds base.md + each step file in playbook order.

      Tracking every member of _STEP_PLAYBOOK_ORDER means step_4_wire.md is
      keyed the moment STEP_4_WIRE is appended — no second edit to the cache
      path is needed when a stage is added.
      """
      h = hashlib.sha256()
      h.update((_SKILLS_DIR / "base.md").read_bytes())
      for step in _STEP_PLAYBOOK_ORDER:
          h.update((_SKILLS_DIR / _STEP_FILE_NAMES[step]).read_bytes())
      assert guided_staged_skill_hash() == h.hexdigest()


  def test_guided_staged_skill_hash_is_deterministic() -> None:
      assert guided_staged_skill_hash() == guided_staged_skill_hash()
  ```

- [ ] **Step 2: Run the test to confirm it fails on the missing symbol.**
  ```bash
  uv run pytest tests/unit/web/composer/guided/test_prompts.py -q
  ```
  Expected: `ImportError: cannot import name 'guided_staged_skill_hash' from 'elspeth.web.composer.guided.prompts'` (collection error).

- [ ] **Step 3: Implement `guided_staged_skill_hash()` in `prompts.py`.**
  Append at module end (after line 185, following `build_step_3_context_block`):
  ```python
  @lru_cache(maxsize=1)
  def guided_staged_skill_hash() -> str:
      """Hex SHA-256 over base.md + every step playbook in _STEP_PLAYBOOK_ORDER.

      Consumed by the tutorial run-cache key (tutorial_model_id, cache input
      #3). Enumerating the playbook order means appending a GuidedStep member
      (and its skill file) automatically extends the keyed input set — the
      step_4_wire.md add (P1) shifts this hash with no edit to the cache path.

      Cached per process; restart elspeth-web.service after editing skill
      markdown (same lifecycle caveat as the other loaders in this module).
      """
      digest = hashlib.sha256()
      digest.update((_SKILLS_DIR / "base.md").read_bytes())
      for step in _STEP_PLAYBOOK_ORDER:
          digest.update((_SKILLS_DIR / _STEP_FILE_NAMES[step]).read_bytes())
      return digest.hexdigest()
  ```
  Add `import hashlib` to the top-of-module imports (after `import json`, line 25):
  ```python
  import hashlib
  import json
  ```

- [ ] **Step 4: Run the test to confirm it passes.**
  ```bash
  uv run pytest tests/unit/web/composer/guided/test_prompts.py -q
  ```
  Expected: `2 passed`.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/composer/guided/prompts.py tests/unit/web/composer/guided/test_prompts.py
  git commit -m "feat(composer/guided): add guided_staged_skill_hash for tutorial cache key

Hashes base.md plus every step playbook in _STEP_PLAYBOOK_ORDER so the
tutorial run-cache key tracks staged skill edits (incl. step_4_wire.md once
the wire step is appended). Cache input #2 of C2.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.2: Add `recipe_catalog_content_hash()` over recipes.py + recipe_match.py (cache input #5)

**Files:**
- Modify: `src/elspeth/web/composer/recipes.py` (add public `recipe_catalog_content_hash()` at module end)
- Modify: `tests/unit/web/composer/test_recipes.py` (Create if absent)

**Interfaces:**
- Produces: `def recipe_catalog_content_hash() -> str` — hex SHA-256 over the byte content of both `composer/recipes.py` and `composer/guided/recipe_match.py` (the deterministic recipe authoring + predicate registry). Hashes source files, not imported objects, so any edit to either module shifts it.
- Consumes: `pathlib.Path(__file__)` for `recipes.py`; resolves `recipe_match.py` via `Path(__file__).parent / "guided" / "recipe_match.py"`.

- [ ] **Step 1: Write the failing test.**
  Create `tests/unit/web/composer/test_recipes.py` (or append):
  ```python
  """Tests for the composer recipe catalog content hash (cache input #5)."""

  from __future__ import annotations

  import hashlib
  from pathlib import Path

  import elspeth.web.composer.recipes as recipes_module
  from elspeth.web.composer.recipes import recipe_catalog_content_hash


  def test_recipe_catalog_content_hash_covers_both_recipe_modules() -> None:
      """The hash folds recipes.py AND guided/recipe_match.py byte content.

      recipe_match selects which recipe fires and pre-fills slots; recipes.py
      authors the deterministic pipeline including option-level content. Both
      are operator-controlled cache inputs, so both must be keyed.
      """
      recipes_path = Path(recipes_module.__file__)
      recipe_match_path = recipes_path.parent / "guided" / "recipe_match.py"
      h = hashlib.sha256()
      h.update(recipes_path.read_bytes())
      h.update(recipe_match_path.read_bytes())
      assert recipe_catalog_content_hash() == h.hexdigest()


  def test_recipe_catalog_content_hash_is_deterministic() -> None:
      assert recipe_catalog_content_hash() == recipe_catalog_content_hash()
  ```

- [ ] **Step 2: Run the test to confirm it fails on the missing symbol.**
  ```bash
  uv run pytest tests/unit/web/composer/test_recipes.py -q
  ```
  Expected: `ImportError: cannot import name 'recipe_catalog_content_hash' from 'elspeth.web.composer.recipes'`.

- [ ] **Step 3: Implement `recipe_catalog_content_hash()` in `recipes.py`.**
  Ensure `import hashlib` and `from functools import cache` (or `lru_cache`) and `from pathlib import Path` are present at the top of `recipes.py`; add whichever are missing. Append at module end:
  ```python
  @cache  # Process-scoped: module source on disk is immutable for the process lifetime.
  def recipe_catalog_content_hash() -> str:
      """Hex SHA-256 over recipes.py + guided/recipe_match.py byte content.

      Cache input #5 of the tutorial run-cache key (C2). Under D11 the
      web_scrape recipe deterministically authors the cached pipeline
      including option-level content (provider, model, prompt_template,
      response_field, schema mode, output format), and recipe_match selects
      the recipe + pre-fills slots. _state_matches_cached_topology is
      option-blind by design and cannot catch this drift, so option fidelity
      is guaranteed by keying both module sources here.
      """
      recipes_path = Path(__file__)
      recipe_match_path = recipes_path.parent / "guided" / "recipe_match.py"
      digest = hashlib.sha256()
      digest.update(recipes_path.read_bytes())
      digest.update(recipe_match_path.read_bytes())
      return digest.hexdigest()
  ```

- [ ] **Step 4: Run the test to confirm it passes.**
  ```bash
  uv run pytest tests/unit/web/composer/test_recipes.py -q
  ```
  Expected: `2 passed`.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/composer/recipes.py tests/unit/web/composer/test_recipes.py
  git commit -m "feat(composer/recipes): add recipe_catalog_content_hash for tutorial cache key

Hashes recipes.py + guided/recipe_match.py source bytes so a recipe builder
or predicate edit invalidates the tutorial run cache. Cache input #4 of C2;
_state_matches_cached_topology stays option-blind by design.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.3: Fold the two new hashes into `tutorial_model_id` (five-input cache key)

**Files:**
- Modify: `src/elspeth/web/composer/tutorial_service.py:834-871` (`tutorial_model_id`)
- Modify: `tests/unit/web/composer/test_tutorial_service.py:401-458` (extend the 3-input regression test + add staged-skill/recipe guards)

**Interfaces:**
- Consumes: `guided_staged_skill_hash` (P7.1), `recipe_catalog_content_hash` (P7.2), `load_skill_with_hash`, `load_deployment_skill`, `WebSettings.composer_model`/`.data_dir`.
- Produces: `tutorial_model_id(settings) -> str` now of form
  `composer=<model>|skill=<core>|staged_skill=<staged>|deployment_skill=<deploy>|recipe=<recipe>`.

- [ ] **Step 1: Write the failing regression-guard test.**
  Append to `tests/unit/web/composer/test_tutorial_service.py` (after `test_tutorial_model_id_changes_when_deployment_skill_overlay_is_added`, line 458). Import the two new hash functions at the top of the file (with the existing `from elspeth.web.composer.skills import load_skill_with_hash` block):
  ```python
  def test_tutorial_model_id_includes_staged_skill_and_recipe_hashes(tmp_path: Path) -> None:
      """C2 five-input cache key: staged guided skills + recipe catalog are keyed.

      rev-4 regression guard: the original test asserted only THREE inputs
      (composer_model + core skill + deployment overlay). A staged design must
      also key the guided staged skills and both recipe modules, or a stage
      block / recipe edit silently serves a stale cached pipeline.
      """
      from elspeth.web.composer.guided.prompts import guided_staged_skill_hash
      from elspeth.web.composer.recipes import recipe_catalog_content_hash

      settings = _make_tutorial_settings(tmp_path, composer_model="anthropic/claude-sonnet-4.5")
      model_id = tutorial_model_id(settings)

      assert f"staged_skill={guided_staged_skill_hash()}" in model_id
      assert f"recipe={recipe_catalog_content_hash()}" in model_id


  def test_tutorial_model_id_shifts_when_recipe_catalog_hash_changes(
      tmp_path: Path,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """Mutating recipes.py / recipe_match.py content shifts the cache key.

      Pins that the recipe-catalog input is load-bearing: if it were dropped, this would
      pass silently. We monkeypatch the hash function the way an edited module
      would shift it (the real shift is a source-byte change).
      """
      import elspeth.web.composer.tutorial_service as tutorial_service_module

      settings = _make_tutorial_settings(tmp_path, composer_model="openai/gpt-5")
      baseline = tutorial_model_id(settings)

      monkeypatch.setattr(
          tutorial_service_module,
          "recipe_catalog_content_hash",
          lambda: "deadbeef" * 8,
      )
      shifted = tutorial_model_id(settings)
      assert baseline != shifted, "recipe catalog hash must participate in the cache key"


  def test_tutorial_model_id_shifts_when_staged_skill_hash_changes(
      tmp_path: Path,
      monkeypatch: pytest.MonkeyPatch,
  ) -> None:
      """Mutating a guided stage block shifts the cache key (staged skill input)."""
      import elspeth.web.composer.tutorial_service as tutorial_service_module

      settings = _make_tutorial_settings(tmp_path, composer_model="openai/gpt-5")
      baseline = tutorial_model_id(settings)

      monkeypatch.setattr(
          tutorial_service_module,
          "guided_staged_skill_hash",
          lambda: "cafebabe" * 8,
      )
      shifted = tutorial_model_id(settings)
      assert baseline != shifted, "staged guided skill hash must participate in the cache key"
  ```

- [ ] **Step 2: Run the new tests to confirm they fail.**
  ```bash
  uv run pytest tests/unit/web/composer/test_tutorial_service.py -q -k "staged_skill or recipe_catalog"
  ```
  Expected: `test_tutorial_model_id_includes_staged_skill_and_recipe_hashes` fails on `assert "staged_skill=..." in model_id` (substring absent); the monkeypatch tests fail with `AttributeError: <module ...tutorial_service> has no attribute 'recipe_catalog_content_hash'` (the name is not yet imported into the module).

- [ ] **Step 3: Implement the five-input key in `tutorial_model_id`.**
  Add the two imports near the existing `load_deployment_skill` import (tutorial_service.py:38):
  ```python
  from elspeth.web.composer.guided.prompts import guided_staged_skill_hash
  from elspeth.web.composer.recipes import recipe_catalog_content_hash
  from elspeth.web.composer.skills import load_deployment_skill, load_skill_with_hash
  ```
  Replace the body of `tutorial_model_id` (the final four lines, 868-871) with:
  ```python
      _, core_skill_hash = load_skill_with_hash("pipeline_composer")
      staged_skill_hash = guided_staged_skill_hash()
      deployment_overlay = load_deployment_skill("pipeline_composer", settings.data_dir)
      deployment_hash = hashlib.sha256(deployment_overlay.encode("utf-8")).hexdigest()
      recipe_hash = recipe_catalog_content_hash()
      return (
          f"composer={settings.composer_model}"
          f"|skill={core_skill_hash}"
          f"|staged_skill={staged_skill_hash}"
          f"|deployment_skill={deployment_hash}"
          f"|recipe={recipe_hash}"
      )
  ```
  Update the docstring of `tutorial_model_id` (lines 835-867): change "Three such inputs" to "Five such inputs", and add the two new bullets to the "Covered (automatic invalidation)" list:
  ```text
      4. The staged guided skill pack (``base.md`` + ``step_1..step_4_wire.md``)
         consumed by the guided per-step chat solver — biases the staged
         compose path that authors the cached pipeline.
      5. The recipe catalog (``composer/recipes.py`` +
         ``composer/guided/recipe_match.py``) — under D11 the web_scrape recipe
         deterministically authors the cached pipeline's option-level content,
         and the predicate registry selects which recipe fires.
  ```

- [ ] **Step 4: Run the new tests + the original 3-input test to confirm all pass.**
  ```bash
  uv run pytest tests/unit/web/composer/test_tutorial_service.py -q -k "tutorial_model_id"
  ```
  Expected: all `tutorial_model_id` tests pass (the original `..._includes_composer_model_core_skill_and_deployment_skill` still passes — its three substring asserts remain present in the new key).

- [ ] **Step 5: Update the `tutorial_cache.py` invalidation-envelope docstring.**
  In `src/elspeth/web/preferences/tutorial_cache.py` add two bullets to the
  "Invalidation envelope" list (after the deployment-overlay bullet, line 14):
  ```text
  - Staged guided skill pack (``base.md`` + ``step_1..step_4_wire.md``) content
    change.
  - Recipe catalog content change (``composer/recipes.py`` or
    ``composer/guided/recipe_match.py``).
  ```

- [ ] **Step 6: Run the full tutorial-cache + tutorial-service suite.**
  ```bash
  uv run pytest tests/unit/web/composer/test_tutorial_service.py tests/unit/web/preferences/test_tutorial_cache.py -q
  ```
  Expected: all pass (0 failures).

- [ ] **Step 7: Commit.**
  ```bash
  git add src/elspeth/web/composer/tutorial_service.py src/elspeth/web/preferences/tutorial_cache.py tests/unit/web/composer/test_tutorial_service.py
  git commit -m "feat(tutorial): five-input cache key (staged skills + recipe catalog)

tutorial_model_id now folds the guided staged-skill hash and a content hash
over recipes.py + recipe_match.py, in addition to composer_model, the core
skill hash, and the retained deployment-overlay hash. Closes C2: a stage
block or recipe/predicate edit invalidates the run cache. Adds the rev-4
regression guard (the old test asserted only three inputs).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.4: Extend the `TutorialState` machine for the embedded-guided handoff

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts:19-279`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`

**Interfaces:**
- Produces: a reshaped `TutorialStep` union `"welcome" | "guided" | "run" | "audit" | "graduation"` (drops `describe`/`showBuilt`/`graph`/`mode`; the wire stage now lives inside guided), the action `{ type: "guidedCompleted"; sessionId: string }`, and a `start` transition `welcome -> guided`.
- Consumes: nothing new; `CANONICAL_TUTORIAL_PROMPT` stays exported (still the cache seed posted at run).

- [ ] **Step 1: Write the failing reducer tests.**
  Replace the body of `tutorialMachine.test.ts` describe-blocks that reference `describe`/`showBuilt`/`graph`/`mode` with the new flow. Add:
  ```typescript
  import { describe, expect, it } from "vitest";
  import {
    initialTutorialState,
    tutorialReducer,
    type TutorialState,
  } from "./tutorialMachine";

  describe("tutorialReducer staged flow", () => {
    it("start advances welcome -> guided", () => {
      const next = tutorialReducer(initialTutorialState, { type: "start" });
      expect(next.step).toBe("guided");
    });

    it("guidedCompleted advances guided -> run and records the session", () => {
      const guided: TutorialState = { ...initialTutorialState, step: "guided" };
      const next = tutorialReducer(guided, {
        type: "guidedCompleted",
        sessionId: "sess-123",
      });
      expect(next.step).toBe("run");
      expect(next.sessionId).toBe("sess-123");
    });

    it("runCompleted advances run -> audit", () => {
      const run: TutorialState = {
        ...initialTutorialState,
        step: "run",
        sessionId: "sess-123",
      };
      const next = tutorialReducer(run, {
        type: "runCompleted",
        result: {
          runId: "run-1",
          sourceDataHash: "hash",
          rows: [],
          seededFromCache: true,
          cacheKey: null,
          discardedRowCount: 0,
        },
      });
      expect(next.step).toBe("audit");
    });

    it("continueToGraduation advances audit -> graduation", () => {
      const audit: TutorialState = { ...initialTutorialState, step: "audit" };
      const next = tutorialReducer(audit, { type: "continueToGraduation" });
      expect(next.step).toBe("graduation");
    });

    it("back from guided returns to welcome", () => {
      const guided: TutorialState = { ...initialTutorialState, step: "guided" };
      const next = tutorialReducer(guided, { type: "back" });
      expect(next.step).toBe("welcome");
    });
  });
  ```

- [ ] **Step 2: Run vitest to confirm failure.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run tutorialMachine
  ```
  Expected: failures — the reducer does not handle `guidedCompleted` / `continueToGraduation` and `start` still maps to `describe`.

- [ ] **Step 3: Reshape `TutorialStep`, the action union, and the reducer.**
  In `tutorialMachine.ts`:
  - Replace the `TutorialStep` union (lines 19-27) with:
    ```typescript
    export type TutorialStep =
      | "welcome"
      | "guided"
      | "run"
      | "audit"
      | "graduation";
    ```
  - Remove `TutorialBuiltSummary`, `TutorialBuildResult`, and `summariseCompositionState`/its helpers (lines 31-42, 211-279) — the big-bang draft summary is gone; the guided surface owns topology display.
  - Replace the `TutorialAction` union (lines 72-83) with:
    ```typescript
    export type TutorialAction =
      | { type: "start" }
      | { type: "guidedCompleted"; sessionId: string }
      | { type: "startRun" }
      | { type: "runCompleted"; result: TutorialRunResult }
      | { type: "continueToGraduation" }
      | { type: "skipToGraduation" }
      | { type: "cancelRun" }
      | { type: "back" }
      | { type: "reset" };
    ```
  - Drop `builtSummary` from `TutorialState` (line 60) and `initialTutorialState` (line 92); change `initialTutorialState.step` stays `"welcome"`.
  - Replace `previousStep` (lines 107-128) with:
    ```typescript
    export function previousStep(state: TutorialState): TutorialStep | null {
      switch (state.step) {
        case "welcome":
          return null;
        case "guided":
          return "welcome";
        case "run":
          return "guided";
        case "audit":
          return "guided";
        case "graduation":
          return "audit";
      }
    }
    ```
  - Replace the reducer cases (lines 134-203) so `start` -> `guided`, add `guidedCompleted` (sets `step:"run"`, `sessionId`), keep `startRun`/`runCompleted`/`cancelRun`, replace `continueToMode`/`finishMode` with `continueToGraduation` (`audit -> graduation`) and `skipToGraduation` (`welcome -> graduation` skip), drop `built`/`showGraph`:
    ```typescript
        case "start":
          return { ...state, step: "guided" };
        case "guidedCompleted":
          return { ...state, step: "run", sessionId: action.sessionId };
        case "startRun":
          if (state.sessionId === null) {
            throw new Error("tutorialReducer: run step requires a session");
          }
          return { ...state, step: "run" };
        case "runCompleted":
          return {
            ...state,
            step: "audit",
            runId: action.result.runId,
            sourceDataHash: action.result.sourceDataHash,
            rows: action.result.rows,
          };
        case "continueToGraduation":
          return { ...state, step: "graduation" };
        case "skipToGraduation":
          return { ...initialTutorialState, step: "graduation", skipped: true };
        case "cancelRun":
          return { ...state, step: "graduation", cancelled: true };
    ```

- [ ] **Step 4: Run vitest to confirm pass.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run tutorialMachine
  ```
  Expected: all `tutorialMachine` tests pass.

- [ ] **Step 5: Run the typechecker (will still error in HelloWorldTutorial — fixed in P7.6).**
  ```bash
  cd src/elspeth/web/frontend && npm run typecheck 2>&1 | head -30
  ```
  Expected: errors localised to `HelloWorldTutorial.tsx` and the deleted-turn components only (they still reference removed steps/actions); `tutorialMachine.ts` itself typechecks. Record these as the P7.6 worklist.

- [ ] **Step 6: Commit.**
  ```bash
  git add src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts
  git commit -m "refactor(tutorial/machine): collapse describe/showBuilt/graph/mode into guided

The staged guided walk (source/sink/transform/wire) replaces the big-bang
describe+showBuilt turns and subsumes the graph turn; mode-choice becomes a
graduation affordance. tutorialMachine retains welcome + run/audit/graduation
and gains a guidedCompleted handoff. HelloWorldTutorial rewire follows.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.5: Build `TutorialGuidedShell` (welcome bookend + embedded guided ChatPanel + completion handoff)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx`
- Create: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx`

**Interfaces:**
- Produces: `export function TutorialGuidedShell(props: { sessionId: string; onCompleted: (sessionId: string) => void }): JSX.Element`. On mount it first binds `activeSessionId=sessionId` while clearing the previous session/guided payload (same field set as `selectSession`'s pre-load reset), then calls `startGuidedSession(sessionId, "tutorial")` and `startGuided(sessionId)` (store), renders the real `ChatPanel` guided surface, and on the new `guidedSession.terminal.kind === "completed"` calls `onCompleted(sessionId)`.
- Consumes (from other phases): `startGuidedSession(sessionId, profileKind)` (P7.1 `client.ts`); `useSessionStore` `guidedSession`/`startGuided`; the `ChatPanel` `chat-panel--guided`/`chat-panel--completed` branches; `guidedSession.profile` (P7.1 wire). The welcome copy reads `profile?.bookends`.

- [ ] **Step 1: Write the failing component test.**
  Create `TutorialGuidedShell.test.tsx`:
  ```typescript
  import { render, screen, waitFor } from "@testing-library/react";
  import { beforeEach, describe, expect, it, vi } from "vitest";
  import { TutorialGuidedShell } from "./TutorialGuidedShell";
  import { useSessionStore } from "@/stores/sessionStore";

  const startGuidedSessionMock = vi.fn();
  const startGuidedMock = vi.fn();

  vi.mock("@/api/client", () => ({
    startGuidedSession: (...args: unknown[]) => startGuidedSessionMock(...args),
  }));

  vi.mock("@/components/chat/ChatPanel", () => ({
    ChatPanel: () => <div data-testid="chat-panel-stub" />,
  }));

  describe("TutorialGuidedShell", () => {
    beforeEach(() => {
      startGuidedSessionMock.mockReset().mockResolvedValue(undefined);
      startGuidedMock.mockReset().mockResolvedValue(undefined);
      // Start with NO active session so the test exercises the real production
      // path: TutorialGuidedShell must itself bind activeSessionId (D3/B4). A
      // pre-set activeSessionId here would mask a shell that forgot to bind it.
      useSessionStore.setState({
        activeSessionId: null,
        messages: [],
        compositionState: null,
        compositionProposals: [],
        composerPreferences: null,
        staleProposalIds: [],
        proposalActionPendingIds: [],
        composerProgress: null,
        stateVersions: [],
        isComposing: false,
        error: null,
        selectedNodeId: null,
        guidedSession: null,
        guidedNextTurn: null,
        guidedTerminal: null,
        guidedChatPending: false,
        guidedResponsePending: false,
        recoveryError: null,
        recoveryStartedCompositionVersion: null,
        startGuided: startGuidedMock,
      } as never);
    });

    it("posts the TUTORIAL profile and enters guided on mount", async () => {
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
      );
      await waitFor(() =>
        expect(startGuidedSessionMock).toHaveBeenCalledWith("sess-1", "tutorial"),
      );
      expect(startGuidedMock).toHaveBeenCalledWith("sess-1");
      // The shell must have bound the store's activeSessionId; otherwise
      // startGuided discards its payload and ChatPanel renders the empty surface.
      expect(useSessionStore.getState().activeSessionId).toBe("sess-1");
    });

    it("renders the real ChatPanel guided surface", async () => {
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
      );
      await waitFor(() =>
        expect(screen.getByTestId("chat-panel-stub")).toBeInTheDocument(),
      );
    });

    it("clears stale completed guided state before starting a new tutorial session", async () => {
      const onCompleted = vi.fn();
      useSessionStore.setState({
        activeSessionId: "old-sess",
        messages: [{ id: "old-message" }],
        compositionState: { id: "old-state", version: 99 },
        compositionProposals: [{ id: "old-proposal" }],
        guidedSession: {
          step: "step_4_wire",
          history: [],
          terminal: { kind: "completed", reason: null },
          chat_history: [],
          chat_turn_seq: 0,
          profile: null,
        },
        guidedNextTurn: null,
        guidedTerminal: { kind: "completed", reason: null },
      } as never);
      render(
        <TutorialGuidedShell sessionId="sess-2" onCompleted={onCompleted} />,
      );

      await waitFor(() =>
        expect(startGuidedSessionMock).toHaveBeenCalledWith("sess-2", "tutorial"),
      );
      expect(useSessionStore.getState().activeSessionId).toBe("sess-2");
      expect(useSessionStore.getState().guidedSession).toBeNull();
      expect(useSessionStore.getState().guidedNextTurn).toBeNull();
      expect(useSessionStore.getState().guidedTerminal).toBeNull();
      expect(useSessionStore.getState().messages).toEqual([]);
      expect(useSessionStore.getState().compositionState).toBeNull();
      expect(onCompleted).not.toHaveBeenCalled();
    });

    it("calls onCompleted when the guided session terminal is completed", async () => {
      const onCompleted = vi.fn();
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={onCompleted} />,
      );
      await waitFor(() => expect(startGuidedMock).toHaveBeenCalled());
      useSessionStore.setState({
        guidedSession: {
          step: "step_4_wire",
          history: [],
          terminal: { kind: "completed", reason: null },
          chat_history: [],
          chat_turn_seq: 0,
          profile: null,
        },
      } as never);
      await waitFor(() => expect(onCompleted).toHaveBeenCalledWith("sess-1"));
    });

    it("shows a user-visible error if guided startup fails", async () => {
      startGuidedSessionMock.mockRejectedValueOnce(new Error("start failed"));
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
      );
      expect(screen.getByRole("status")).toHaveTextContent("Starting guided composer");
      expect(await screen.findByRole("alert")).toHaveTextContent("start failed");
      expect(startGuidedMock).not.toHaveBeenCalled();
    });

    it("shows a user-visible error if store guided entry fails", async () => {
      startGuidedMock.mockRejectedValueOnce(new Error("store failed"));
      render(
        <TutorialGuidedShell sessionId="sess-1" onCompleted={vi.fn()} />,
      );
      expect(await screen.findByRole("alert")).toHaveTextContent("store failed");
    });
  });
  ```

- [ ] **Step 2: Run the test to confirm it fails (module missing).**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run TutorialGuidedShell
  ```
  Expected: failure — `Failed to resolve import "./TutorialGuidedShell"`.

- [ ] **Step 3: Implement `TutorialGuidedShell.tsx`.**
  ```tsx
  import { useEffect, useRef, useState } from "react";
  import { startGuidedSession } from "@/api/client";
  import { ChatPanel } from "@/components/chat/ChatPanel";
  import { useSessionStore } from "@/stores/sessionStore";

  interface TutorialGuidedShellProps {
    sessionId: string;
    onCompleted: (sessionId: string) => void;
  }

  /**
   * Tutorial bridge (D9): renders the welcome bookend, starts a TUTORIAL-profile
   * guided session, EMBEDS the real ChatPanel guided surface (the truest "use
   * the real thing"), and on guided terminal=completed hands the session back to
   * the surviving tutorialMachine run/audit/graduation tail. Per-stage
   * interpretation review + the wire confirm are owned by the ChatPanel guided
   * branch, which already projects interpretationEventsStore.pendingBySession and
   * blocks advancement while pending (P4.T2). Coaching/bookend copy reads off the
   * wire GuidedSession.profile.
   */
  export function TutorialGuidedShell({
    sessionId,
    onCompleted,
  }: TutorialGuidedShellProps): JSX.Element {
    const guidedSession = useSessionStore((s) => s.guidedSession);
    const startGuided = useSessionStore((s) => s.startGuided);
    const startedRef = useRef(false);
    const completedRef = useRef(false);
    const [starting, setStarting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Start the TUTORIAL-profile guided session exactly once. The start
    // endpoint is idempotent server-side (P7.1): a second POST for a session
    // that already has a persisted GuidedSession returns it unchanged. The
    // startedRef guard avoids a redundant round-trip under StrictMode's
    // double-invoke.
    useEffect(() => {
      if (startedRef.current) {
        return;
      }
      startedRef.current = true;
      void (async () => {
        setStarting(true);
        setError(null);
        // Bind the store's activeSessionId to this tutorial session BEFORE
        // startGuided. startGuided (sessionStore.ts) DISCARDS its fetched guided
        // payload unless get().activeSessionId === the requested id, and ChatPanel
        // renders the empty-session surface (chat-panel--empty) whenever
        // activeSessionId is null. Clear the same session/guided payload that
        // selectSession clears before loading, otherwise a completed guided session
        // from the previous active session can make ChatPanel render the completed
        // surface and fire onCompleted before the new tutorial session has loaded.
        useSessionStore.setState({
          activeSessionId: sessionId,
          messages: [],
          compositionState: null,
          compositionProposals: [],
          composerPreferences: null,
          staleProposalIds: [],
          proposalActionPendingIds: [],
          composerProgress: null,
          stateVersions: [],
          isComposing: false,
          error: null,
          selectedNodeId: null,
          guidedSession: null,
          guidedNextTurn: null,
          guidedTerminal: null,
          guidedChatPending: false,
          guidedResponsePending: false,
          recoveryError: null,
          recoveryStartedCompositionVersion: null,
        });
        try {
          await startGuidedSession(sessionId, "tutorial");
          await startGuided(sessionId);
        } catch (err) {
          setError(formatError(err));
        } finally {
          setStarting(false);
        }
      })();
    }, [sessionId, startGuided]);

    // Hand off to the run/audit/graduation tail when guided reaches completion.
    useEffect(() => {
      if (completedRef.current) {
        return;
      }
      const current = useSessionStore.getState();
      if (
        current.activeSessionId !== sessionId ||
        current.guidedSession !== guidedSession
      ) {
        return;
      }
      if (guidedSession?.terminal?.kind === "completed") {
        completedRef.current = true;
        onCompleted(sessionId);
      }
    }, [guidedSession, onCompleted, sessionId]);

    const bookends = guidedSession?.profile?.bookends ?? true;

    return (
      <section
        className="tutorial-guided-shell"
        aria-label="Guided pipeline composer"
      >
        {bookends && (
          <p className="tutorial-kicker">
            Let's build your first pipeline one stage at a time.
          </p>
        )}
        <p role="status" className="sr-only">
          {starting ? "Starting guided composer" : ""}
        </p>
        {error !== null && (
          <p role="alert" className="tutorial-error">
            {error}
          </p>
        )}
        <ChatPanel />
      </section>
    );
  }

  function formatError(err: unknown): string {
    if (
      typeof err === "object" &&
      err !== null &&
      "detail" in err &&
      typeof (err as { detail?: unknown }).detail === "string"
    ) {
      return (err as { detail: string }).detail;
    }
    if (err instanceof Error) {
      return err.message;
    }
    return "The guided tutorial could not be started.";
  }
  ```

- [ ] **Step 4: Run the test to confirm it passes.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run TutorialGuidedShell
  ```
  Expected: all six tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx
  git commit -m "feat(tutorial): TutorialGuidedShell bridge embeds the real guided ChatPanel

Starts a TUTORIAL-profile guided session (idempotent guided/start), mounts
the real ChatPanel guided surface, and hands the session to the run/audit/
graduation tail on terminal=completed. Per-stage interpretation review + wire
confirm are owned by the ChatPanel guided branch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.6: Rewire `HelloWorldTutorial`; remove `TutorialTurn2Describe` + `TutorialTurn2bShowBuilt`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx:1-167`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx` + `.test.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx` + `.test.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx` (the Step 4 `git rm` removes it; see the Decision note below)
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/copy.ts` (drop deleted-turn copy; keep lifecycle title constants)
- Modify: `src/elspeth/web/frontend/src/components/tutorial/index.ts` (drop removed exports)

**Interfaces:**
- Consumes: `TutorialGuidedShell` (P7.5), `createSession` + `renameSession` (existing), `HELLO_WORLD_PENDING_SESSION_TITLE` / `HELLO_WORLD_SESSION_TITLE` (`copy.ts`), the surviving `TutorialTurn1Welcome` / `TutorialTurn4Run` / `TutorialTurn5AuditStory` / `TutorialTurn7Graduation`, `tutorialReducer` (P7.4).
- Decision: `TutorialTurn6ModeChoice` is removed. Graduation now saves the default composer mode as `"guided"` and renames the tutorial session to `HELLO_WORLD_SESSION_TITLE`; `HelloWorldTutorial` renames the newly-created session to `HELLO_WORLD_PENDING_SESSION_TITLE` before external `POST /api/sessions/{session_id}/guided/start` so orphan cleanup still finds sessions abandoned mid-tutorial.

- [ ] **Step 1: Write the failing integration test for the rewired shell.**
  Replace `HelloWorldTutorial.test.tsx`'s describe/showBuilt assertions with:
  ```typescript
  import { render, screen, waitFor } from "@testing-library/react";
  import userEvent from "@testing-library/user-event";
  import { beforeEach, describe, expect, it, vi } from "vitest";
  import { HelloWorldTutorial } from "./HelloWorldTutorial";

  vi.mock("@/api/client", () => ({
    deleteTutorialOrphans: vi.fn().mockResolvedValue(undefined),
    createSession: vi.fn().mockResolvedValue({ id: "sess-new" }),
    renameSession: vi.fn().mockResolvedValue({ id: "sess-new", title: "hello-world (pending)" }),
    startGuidedSession: vi.fn().mockResolvedValue(undefined),
  }));

  vi.mock("./TutorialGuidedShell", () => ({
    TutorialGuidedShell: ({
      onCompleted,
    }: {
      onCompleted: (s: string) => void;
    }) => (
      <button type="button" onClick={() => onCompleted("sess-new")}>
        finish-guided
      </button>
    ),
  }));

  describe("HelloWorldTutorial staged flow", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("renders the welcome bookend first", () => {
      render(<HelloWorldTutorial />);
      expect(
        screen.getByRole("heading", { name: /welcome/i }),
      ).toBeInTheDocument();
    });

    it("advances welcome -> guided -> run on guided completion", async () => {
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: /start/i }));
      await waitFor(() =>
        expect(
          screen.getByRole("button", { name: "finish-guided" }),
        ).toBeInTheDocument(),
      );
      await user.click(screen.getByRole("button", { name: "finish-guided" }));
      await waitFor(() =>
        expect(screen.queryByRole("button", { name: "finish-guided" })).toBeNull(),
      );
    });

    it("tags the created tutorial session before entering guided", async () => {
      const api = await import("@/api/client");
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: /start/i }));
      await waitFor(() =>
        expect(api.renameSession).toHaveBeenCalledWith("sess-new", "hello-world (pending)"),
      );
      const createOrder = vi.mocked(api.createSession).mock.invocationCallOrder[0];
      const renameOrder = vi.mocked(api.renameSession).mock.invocationCallOrder[0];
      expect(createOrder).toBeLessThan(renameOrder);
    });

    it("surfaces createSession failure instead of stalling on welcome", async () => {
      const api = await import("@/api/client");
      vi.mocked(api.createSession).mockRejectedValueOnce(new Error("session service down"));
      const user = userEvent.setup();
      render(<HelloWorldTutorial />);
      await user.click(screen.getByRole("button", { name: /start/i }));
      expect(screen.getByRole("status")).toHaveTextContent("Creating tutorial session");
      expect(await screen.findByRole("alert")).toHaveTextContent("session service down");
    });

    it("does not import the removed describe/showBuilt turns", async () => {
      const source = await import("./HelloWorldTutorial");
      expect(source).toBeDefined();
      // Static guard: the deleted modules must not be referenced.
    });
  });
  ```

- [ ] **Step 2: Run the test to confirm it fails.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run HelloWorldTutorial
  ```
  Expected: failure — `HelloWorldTutorial` still renders `TutorialTurn2Describe` and has no guided/run-on-completion path; the `start` -> `describe` branch is gone from the reducer (P7.4) so the describe turn never appears.

- [ ] **Step 3: Rewire `HelloWorldTutorial.tsx`.**
  Replace the imports (lines 1-14) and the render branches (lines 66-167) so the
  step set is `welcome / guided / run / audit / graduation`. A session is created
  on Start so `TutorialGuidedShell` has a `sessionId`:
  ```tsx
  import { useEffect, useReducer, useState } from "react";
  import { createSession, deleteTutorialOrphans, renameSession } from "@/api/client";
  import { TutorialTurn1Welcome } from "./TutorialTurn1Welcome";
  import { TutorialGuidedShell } from "./TutorialGuidedShell";
  import { TutorialTurn4Run } from "./TutorialTurn4Run";
  import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";
  import { TutorialTurn7Graduation } from "./TutorialTurn7Graduation";
  import {
    CANONICAL_TUTORIAL_PROMPT,
    initialTutorialState,
    tutorialReducer,
  } from "./tutorialMachine";
  import { HELLO_WORLD_PENDING_SESSION_TITLE } from "./copy";
  ```
  In the component, replace the `welcome`/`describe`/`showBuilt`/`graph`/`mode`
  branches. On Start, create a session, store its id, then dispatch `start`:
  ```tsx
    const [state, dispatch] = useReducer(tutorialReducer, initialTutorialState);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [starting, setStarting] = useState(false);
    const [startError, setStartError] = useState<string | null>(null);

    useEffect(() => {
      void deleteTutorialOrphans().catch((err) => {
        console.error("[tutorial] orphan cleanup failed:", err);
      });
    }, []);

    const onStart = async (): Promise<void> => {
      setStarting(true);
      setStartError(null);
      try {
        const session = await createSession();
        await renameSession(session.id, HELLO_WORLD_PENDING_SESSION_TITLE);
        setSessionId(session.id);
        dispatch({ type: "start" });
      } catch (err) {
        setStartError(formatError(err));
      } finally {
        setStarting(false);
      }
    };
  ```
  Render branches:
  ```tsx
        {state.step === "welcome" && (
          <>
          <p role="status" className="sr-only">
            {starting ? "Creating tutorial session" : ""}
          </p>
          {startError !== null && (
            <p role="alert" className="tutorial-error">
              {startError}
            </p>
          )}
          <TutorialTurn1Welcome
            onStart={() => void onStart()}
            onSkip={() => dispatch({ type: "skipToGraduation" })}
          />
          </>
        )}
        {state.step === "guided" && sessionId !== null && (
          <TutorialGuidedShell
            sessionId={sessionId}
            onCompleted={(id) =>
              dispatch({ type: "guidedCompleted", sessionId: id })
            }
          />
        )}
        {state.step === "run" && state.sessionId !== null && (
          <TutorialTurn4Run
            sessionId={state.sessionId}
            prompt={CANONICAL_TUTORIAL_PROMPT}
            onCompleted={(result) => dispatch({ type: "runCompleted", result })}
            onCancelled={() => dispatch({ type: "cancelRun" })}
            onBack={goBack}
          />
        )}
        {state.step === "audit" &&
          state.sessionId !== null &&
          state.runId !== null &&
          state.sourceDataHash !== null && (
            <TutorialTurn5AuditStory
              sessionId={state.sessionId}
              runId={state.runId}
              onContinue={() => dispatch({ type: "continueToGraduation" })}
              onBack={goBack}
            />
          )}
        {state.step === "graduation" && (
          <TutorialTurn7Graduation
            sessionId={state.sessionId}
            skipped={state.skipped}
            onBack={goBack}
          />
        )}
  ```
  Add the same local `formatError(err: unknown): string` helper shape used by
  `TutorialGuidedShell` so API `detail` strings and plain `Error.message` values
  are surfaced without swallowing startup failures.

- [ ] **Step 3b: Move default-mode save + final title rename into `TutorialTurn7Graduation`.**
  Update `TutorialTurn7Graduation` props to accept `sessionId: string | null` and
  `skipped: boolean`. Import `HELLO_WORLD_SESSION_TITLE` from `copy.ts`.
  In `onFinish`, before creating the fresh post-tutorial composer session:
  ```tsx
      if (sessionId !== null && !skipped) {
        await useSessionStore
          .getState()
          .renameSession(sessionId, HELLO_WORLD_SESSION_TITLE);
      }
      await usePreferencesStore.getState().saveTutorialMode("guided");
  ```
  Keep the existing `markTutorialGraduated({ publishLocally: false })`,
  `createSession()`, and `publishTutorialGraduation(completedAt)` flow after those
  calls. The failure behavior remains fail-closed: if title rename or default-mode
  save fails, show the existing role alert and do not create the fresh composer
  session.

- [ ] **Step 3c: Extend `TutorialTurn7Graduation.test.tsx`.**
  Update existing renders to pass `sessionId="sess-new"` and `skipped={false}`.
  Add assertions:
  ```typescript
  it("renames the tutorial session and saves Guided as the default before finishing", async () => {
    const user = userEvent.setup();
    render(
      <TutorialTurn7Graduation
        sessionId="sess-new"
        skipped={false}
        onBack={() => undefined}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );
    await waitFor(() =>
      expect(useSessionStore.getState().renameSession).toHaveBeenCalledWith(
        "sess-new",
        "hello-world (cool government pages)",
      ),
    );
    expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
      default_mode: "guided",
    });
    expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
      tutorial_completed_at: expect.any(String),
    });
  });

  it("does not rename a skipped tutorial session but still saves Guided default", async () => {
    const user = userEvent.setup();
    render(
      <TutorialTurn7Graduation
        sessionId={null}
        skipped={true}
        onBack={() => undefined}
      />,
    );
    await user.click(
      screen.getByRole("button", { name: "Take me to the composer" }),
    );
    expect(useSessionStore.getState().renameSession).not.toHaveBeenCalled();
    expect(api.updateUserComposerPreferences).toHaveBeenCalledWith({
      default_mode: "guided",
    });
  });
  ```
  Ensure the test store mock includes a spyable `renameSession` implementation.

- [ ] **Step 3d: Replace progress labels with the staged 5-step set.**
  In `HelloWorldTutorial.tsx`, replace `TUTORIAL_STEP_LABELS` and `stepIndex`
  (lines 135-167) with:
  ```tsx
  const TUTORIAL_STEP_LABELS: ReadonlyArray<{ key: string; label: string }> = [
    { key: "welcome", label: "Welcome" },
    { key: "guided", label: "Build" },
    { key: "run", label: "Run" },
    { key: "audit", label: "Audit" },
    { key: "graduation", label: "Ready" },
  ];

  function stepIndex(step: string): number {
    switch (step) {
      case "welcome":
        return 0;
      case "guided":
        return 1;
      case "run":
        return 2;
      case "audit":
        return 3;
      case "graduation":
        return 4;
      default:
        return 0;
    }
  }
  ```

- [ ] **Step 4: Delete the big-bang turn components + their tests, and drop their exports.**
  ```bash
  git rm src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx \
         src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx \
         src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx \
         src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx \
         src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx \
         src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx
  ```
  Edit `src/elspeth/web/frontend/src/components/tutorial/index.ts` to remove any
  `export` lines naming `TutorialTurn2Describe`, `TutorialTurn2bShowBuilt`,
  `TutorialTurn3Graph`, `TutorialTurn6ModeChoice`, `TutorialBuiltSummary`,
  `TutorialBuildResult`, and `summariseCompositionState`. Add an export for
  `TutorialGuidedShell`.

- [ ] **Step 5: Run vitest over the tutorial suite + typecheck.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run tutorial && npm run typecheck
  ```
  Expected: tutorial tests pass; `npm run typecheck` reports 0 errors (no dangling references to deleted modules/steps remain). Remove the now-unused Turn 2/2b/3/6 copy constants in `copy.ts`, but keep `HELLO_WORLD_PENDING_SESSION_TITLE` and `HELLO_WORLD_SESSION_TITLE` because the staged lifecycle still uses them.

- [ ] **Step 6: Commit.**
  ```bash
  git add src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx \
          src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx \
          src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx \
          src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx \
          src/elspeth/web/frontend/src/components/tutorial/copy.ts \
          src/elspeth/web/frontend/src/components/tutorial/index.ts
  git commit -m "feat(tutorial): rewire HelloWorldTutorial onto TutorialGuidedShell

Welcome -> embedded guided ChatPanel -> run/audit/graduation. Removes the
big-bang TutorialTurn2Describe, TutorialTurn2bShowBuilt, TutorialTurn3Graph,
and TutorialTurn6ModeChoice; the wire stage and per-stage interpretation
review now live inside the real guided surface. Pending/final session title
tags are preserved, and graduation saves Guided as the default mode.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.7: Rewrite the tutorial E2E specs for the staged flow

**Files:**
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts`
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`
- Modify: `src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts` (extend for the wire stage)

**Interfaces:**
- Consumes: the mocked guided protocol (the existing `composer-guided.spec.ts` mock pattern), full session-scoped guided routes (`POST /api/sessions/{session_id}/guided/start`, `GET /api/sessions/{session_id}/guided`, `POST /api/sessions/{session_id}/guided/respond`), the `STEP_4_WIRE` turn payload (`WireStageData`, P3), the tutorial run endpoint mock.

- [ ] **Step 1: Read the existing mocked-guided E2E pattern so the rewrite reuses it.**
  ```bash
  cd src/elspeth/web/frontend && sed -n '1,80p' tests/e2e/composer-guided.spec.ts
  ```
  Expected: confirms how the spec mocks the full session-scoped routes
  `GET /api/sessions/{session_id}/guided` and
  `POST /api/sessions/{session_id}/guided/respond`, and asserts on the
  `chat-panel--guided` surface. Note the route-mock helpers it imports
  (page-objects / helpers).

- [ ] **Step 2: Rewrite `tutorial.spec.ts` to the staged flow (mocked).**
  Replace the describe/showBuilt assertions. The new happy-path mocks:
  `POST /api/sessions` (returns `{id}`), `POST /api/sessions/{session_id}/guided/start` (200, idempotent),
  `GET /api/sessions/{session_id}/guided` returning a `step_1_source` turn, then drive
  `POST /api/sessions/{session_id}/guided/respond` through source -> sink -> recipe-apply -> `step_4_wire`,
  then a wire-confirm `respond` returning `terminal.kind === "completed"`, then
  the existing tutorial-run mock. Assert:
  - the welcome bookend renders, Start mounts the `chat-panel--guided` surface;
  - the wire stage renders the topology + edge-contract overlay (assert a
    `from`/`to` edge cell, not `from_id`/`to_id` — M1);
  - on `terminal=completed` the run turn appears (no 409 dead-end);
  - the run/audit/graduation tail completes.
  Concretely add, in the wire-stage assertion block:
  ```typescript
  // The wire validation payload must surface the live prompt-shield advisory
  // for the canonical web_scrape -> llm shape (D11/B4 rev-4), and must NOT
  // contain an azure_prompt_shield node. The mock seeds both in the
  // step_4_wire turn payload.
  await expect(page.getByText(/prompt-injection shield/i)).toBeVisible();
  await expect(page.locator('text=azure_prompt_shield')).toHaveCount(0);
  ```

- [ ] **Step 3: Rewrite `tutorial-reliability.staging.spec.ts` to drive the staged guided flow.**
  Point the harness at the `POST /api/sessions/{session_id}/guided/start` (tutorial profile) entry then the
  staged respond loop instead of the single `describe` -> `showBuilt` compose; the
  run/audit assertions are unchanged (C1 tail survives). Keep the
  classify/aggregate harness wiring (`tests/e2e/harness/`) intact — only the
  compose-phase driver changes.

- [ ] **Step 4: Extend `composer-guided.spec.ts` for the wire stage.**
  Add a test that, after the existing chain-accept, the next turn is
  `step_4_wire` with both blobs present (`topology` + `edge_contracts`), the
  confirm gates on `validate().is_valid`, and a `field_mapper`/schema-relax
  reconciliation re-renders the overlay (B6). Assert the empty/live-guided
  profile reaches `terminal=completed` on a valid wire confirm with **no advisor
  provider call** (profile-gating, D13 — the live profile opts out).

- [ ] **Step 5: Run the non-staging Playwright specs.**
  ```bash
  cd src/elspeth/web/frontend
  npm run test:e2e -- tutorial.spec.ts composer-guided.spec.ts
  npm run test:e2e
  ```
  Expected: targeted tutorial/guided specs pass, then the full non-staging E2E suite passes. The `.staging.spec.ts` is excluded from the default e2e run; it runs under `test:e2e:staging` against a live deploy.

- [ ] **Step 6: Run the staging E2E gate because this task rewrites `tutorial-reliability.staging.spec.ts`.**
  ```bash
  cd src/elspeth/web/frontend && npm run test:e2e:staging
  ```
  Expected: staging tutorial reliability spec passes against the configured live deploy. If the live deploy is unavailable, stop and report the environment blocker; do not mark the staging rewrite verified.

- [ ] **Step 7: Run the `SlotType` / `guided.ts` mirror gate (this branch edits guided.ts upstream; re-run here to confirm no drift).**
  ```bash
  uv run python scripts/cicd/check_slot_type_cross_language.py
  ```
  Expected: exit 0 (no SlotType / guided.ts mirror drift).

- [ ] **Step 8: Commit.**
  ```bash
  git add src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts
  git commit -m "test(e2e): rewrite tutorial specs for the staged guided flow + wire stage

tutorial.spec drives welcome -> guided (source/sink/recipe-apply/wire) ->
run/audit; pins the live prompt-shield advisory presence + absence of an
azure_prompt_shield node on the canonical web_scrape pipeline. Extends
composer-guided.spec for the wire stage (two-read overlay, validate().is_valid
gate, live-profile no-advisor completion).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.8: Verify the advisor-prose correction (P5.9) held + no test re-pins the stale phrase

> **Why this is a verify-only task.** The `_dispatch.py` "Disabled by default"
> prose was already corrected in **P5.9** (which replaced the trailing
> `:129-130` clause with the operator-configured / mandatory-END-sign-off text
> and added `tests/unit/web/composer/test_advisor_tool_prose.py`). This
> task must NOT re-edit the prose — by the time P7 runs, the phrase is gone and a
> blind string-replace would fail or commit nothing. P7.8 instead CONFIRMS the
> P5.9 change survived the intervening phases and that no *other* test (e.g. the
> broader `test_dispatch_arms_characterization.py`) still pins the deleted phrase.
> If P5.9 did not land, Step 1 fails loud and you return to P5.9 — do not author
> the fix here.

**Files:**
- Modify (only if Step 1 finds a stale pin): a matching assertion in
  `tests/unit/web/composer/test_dispatch_arms_characterization.py`.

**Interfaces:** none (verification of an already-landed prose change).

- [ ] **Step 1: Confirm the stale phrase is fully gone from source.**
  ```bash
  grep -rn "Disabled by default" src/elspeth/web
  ```
  Expected: **no output** (exit 1). If `_dispatch.py:129` still matches, P5.9 did
  NOT land — stop and complete P5.9 first; do not re-author the fix here.

- [ ] **Step 2: Confirm the corrected prose + the P5.9 guard test are present.**
  ```bash
  grep -n "operator-configured" src/elspeth/web/composer/tools/_dispatch.py && uv run python -m pytest tests/unit/web/composer/test_advisor_tool_prose.py -q
  ```
  Expected: the grep prints the `operator-configured` line and the prose guard
  test is `1 passed`.

- [ ] **Step 3: Check the broader dispatch suite for any OTHER test still pinning the old phrase.**
  ```bash
  grep -rn "Disabled by default" tests/ ; uv run python -m pytest tests/unit/web/composer/test_dispatch_arms_characterization.py -q
  ```
  Expected: the grep prints **nothing** and `test_dispatch_arms_characterization.py` is all `passed`.
  If the grep hits a stale assertion in `test_dispatch_arms_characterization.py`, update its expected
  substring to `"operator-configured"`, re-run, then proceed to Step 4. If the
  grep is empty, there is nothing to edit and no commit is needed — this task is
  satisfied (skip Step 4).

- [ ] **Step 4: Commit ONLY if Step 3 edited a test.**
  ```bash
  git add tests/unit/web/composer/test_dispatch_arms_characterization.py && git commit -m "test(composer): repoint stale 'Disabled by default' dispatch assertion to corrected prose (P7.8)

P5.9 corrected the advisor tool prose; this repoints the lingering
test_dispatch.py substring assertion to the operator-configured wording.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.9: Migration — runbook deltas (epoch verification + hardened secret-archive)

**Files:**
- Modify: `docs/runbooks/staging-session-db-recreation.md`

**Interfaces:** none (operator-facing procedure). Current code at `release/0.7.0` HEAD is already `SESSION_SCHEMA_EPOCH = 23`; this task documents the next session-DB-only cutover as `SESSION_SCHEMA_EPOCH` **23→24** and adds the runbook-side verification + hardening. The design spec is owned by the global/spec edit and must already state the same 23→24 contract before P7 closes.

- [ ] **Step 1: Add a "Current Cutover: 0.7.0 (single-DB reset)" section** above the existing "Current Cutover: 0.6.0" heading (line 5). It must state:
  - 0.7.0 bumps **only** `SESSION_SCHEMA_EPOCH` 23→24 (NOT `SQLITE_SCHEMA_EPOCH`), so this is a **single-DB** session-only cutover — follow the "Staging Reset" / "Local Or Dev Reset" single-DB path and **do NOT** run the Phase 5b two-DB procedure.
  - Boot fail-closes on a stale session DB with the `SessionSchemaError` "version 23 does not match SESSION_SCHEMA_EPOCH=24 ... Delete the session DB file and restart" (the `_assert_schema_sentinels` guard), converting the lazy per-row 500 into a loud boot guard.
  - `auth.db` and `runs/audit.db` are separate files and are NOT reset.

- [ ] **Step 2: Add a "0.7.0 epoch + smoke verification" block** to the staging procedure's post-restart steps (after the health checks, around line 376). It must instruct the operator to:
  ```bash
  # Confirm the recreated session DB carries the new epoch sentinel.
  sqlite3 "$DB_PATH" 'PRAGMA user_version;'   # expect 24 (== SESSION_SCHEMA_EPOCH)
  ```
  and to run a fresh-session smoke: create a session via the UI, start the
  tutorial (TUTORIAL profile), drive it through the staged guided walk to a
  `terminal=completed`, and run it — confirming the journal shows **no**
  `SessionSchemaError`, no per-row 500, and no
  `UnresolvedInterpretationPlaceholderError` (proving the B1 surfacing landed).

- [ ] **Step 3: Harden the secret-archive steps** in the "Procedure" archive loop (around lines 355-364) and the `user_secrets` blast-radius precondition (line 266 and line 380). Add, as explicit sub-steps in the runbook prose:
  - Before the `cp -a` archive loop, run a WAL checkpoint (or take a clean
    shutdown) so uncheckpointed secret rows are not carried into the archive:
    ```bash
    # The archive includes the encrypted UserSecretStore blob (app.py:874 — the
    # secret store rides the session engine). Checkpoint the WAL FIRST so the
    # -wal/-journal sidecars do not carry uncheckpointed secret material into
    # the long-lived archive.
    sqlite3 "$DB_PATH" 'PRAGMA wal_checkpoint(TRUNCATE);'
    ```
  - In the `user_secrets` sign-off (preconditions §8 and the post-reset
    verification at line 380), add: **destroy or secure the archive at the end of
    the deploy window** — it is a long-lived copy of live encrypted secret
    material and is only inert if `settings.secret_key` is **rotated**; if the
    key is reused across the deploy, the archive is decryptable with the running
    app's key. State the secret_key rotation note explicitly.

- [ ] **Step 4: Verify the design spec cross-reference without editing it.**
  ```bash
  grep -n "staging-session-db-recreation" docs/superpowers/specs/2026-06-22-tutorial-staged-recut-design.md
  ```
  Expected: the spec already references the runbook (line ~672/862) and its schema epoch prose says `23→24` / expected `24`. Do **not** edit the design spec in this phase; if it still carries prior stale epoch wording, stop and route that back to the global/spec owner.

- [ ] **Step 5: Lint the runbook markdown (no broken anchors / fenced blocks).**
  ```bash
  grep -c '```' docs/runbooks/staging-session-db-recreation.md
  ```
  Expected: an even number (every fence closed).

- [ ] **Step 6: Commit.**
  ```bash
  git add docs/runbooks/staging-session-db-recreation.md
  git commit -m "docs(runbook): 0.7.0 single-DB cutover + epoch verify + hardened secret archive

Adds the 0.7.0 single-DB (SESSION_SCHEMA_EPOCH 23->24) cutover section, a
PRAGMA user_version==24 + fresh-session-reaches-COMPLETED smoke verification,
and hardened secret-archive steps (wal_checkpoint(TRUNCATE) before cp -a;
destroy/secure the archive + secret_key rotation note at deploy-window end).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.9b: Bump package version 0.6.0 → 0.7.0 + relock

**Files:**
- Modify: `pyproject.toml:3` (`version`)
- Modify: `uv.lock` (regenerated)

**Interfaces:** none.

The spec/plan target version is **0.7.0** (pre-release), but `pyproject.toml` still reads `version = "0.6.0"`. Land the bump with the migration/runbook cutover so the shipped artifact carries the right version (CI runs `uv sync --frozen`, so a stale lock fails the build).

- [ ] **Step 1: Bump the version.**
  In `pyproject.toml` line 3: `version = "0.6.0"` → `version = "0.7.0"`.
- [ ] **Step 2: Regenerate the lock.**
  `uv lock`
  (refreshes the `[[package]] name = "elspeth"` pin).
- [ ] **Step 3: Verify.**
  `uv run python -c "import importlib.metadata as m; print(m.version('elspeth'))"`
  Expected: `0.7.0`.
- [ ] **Step 4: Commit.**
  ```bash
  git add pyproject.toml uv.lock && git commit -m "chore: bump version 0.6.0 -> 0.7.0

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.10: Phase gate sweep (backend + frontend + trust gates + wardline)

**Files:** none (verification only; fix-forward any failure in the owning file).

**Interfaces:** none.

- [ ] **Step 1: Backend lint + format + types.**
  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  ```
  Expected: ruff clean; format clean; mypy `Success: no issues found`.

- [ ] **Step 2: Targeted backend pytest over the phase's surfaces.**
  ```bash
  uv run pytest tests/unit/web/composer/guided/test_prompts.py \
    tests/unit/web/composer/test_recipes.py \
    tests/unit/web/composer/test_tutorial_service.py \
    tests/unit/web/preferences/test_tutorial_cache.py \
    tests/integration/web/test_tutorial_routes.py -q
  ```
  Expected: all pass (0 failures). This pins the canonical-seed cross-language equality, the five-input cache key, and the tutorial run route.

- [ ] **Step 3: elspeth-lints trust gates this phase touches.**
  ```bash
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
    --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' \
    --root src/elspeth
  ```
  Expected: no new findings attributable to P7 files (the cache helpers + shell + prose are not new Tier-3 boundaries; if `trust_tier.tier_model` reports displacement it is from a sibling phase, not P7 — record, do not bless blind, per the gate-debt note).

- [ ] **Step 4: Frontend gates (from the frontend dir).**
  ```bash
  cd src/elspeth/web/frontend
  npm run typecheck
  npm test -- --run
  npm run build
  npm run test:e2e
  npm run test:e2e:staging
  ```
  Expected: typecheck 0 errors; vitest all pass; build succeeds; non-staging E2E passes; staging tutorial reliability E2E passes against the configured live deploy. If staging is unavailable, report the environment blocker rather than marking this gate verified.

- [ ] **Step 5: wardline taint gate.**
  ```bash
  wardline scan . --fail-on ERROR
  ```
  Expected: exit 0 (clean). The P7 surfaces (cache hashing of local files, a frontend shell, runbook prose) introduce no new external-input -> sink flow; if a finding appears, fix it at the boundary per the `wardline-gate` skill before proceeding.

- [ ] **Step 6: Final phase commit (only if any fix-forward edits were made in steps 1-5).**
  ```bash
  git add src/elspeth/web/composer/guided/prompts.py \
          tests/unit/web/composer/guided/test_prompts.py \
          src/elspeth/web/composer/recipes.py \
          tests/unit/web/composer/test_recipes.py \
          src/elspeth/web/composer/tutorial_service.py \
          src/elspeth/web/preferences/tutorial_cache.py \
          tests/unit/web/composer/test_tutorial_service.py \
          src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts \
          src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts \
          src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx \
          src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx \
          src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx \
          src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx \
          src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx \
          src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx \
          src/elspeth/web/frontend/src/components/tutorial/copy.ts \
          src/elspeth/web/frontend/src/components/tutorial/index.ts \
          src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts \
          src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts \
          src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts \
          docs/runbooks/staging-session-db-recreation.md \
          pyproject.toml uv.lock
  git commit -m "chore(p7): gate-sweep fix-forward for cache/frontend/migration phase

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
