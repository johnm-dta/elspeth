# Tutorial Staged Recut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-pass "big-bang" first-run tutorial (one canonical
sentence → one inference wires source + transforms + sink) with a **staged
wizard** that walks the learner through source → sink → transform → wire. The
tutorial becomes a specific *instance* of the existing `composer/guided/` staged
state machine, parameterised by a server-owned `WorkflowProfile`; live guided mode
is the same engine with the empty (canonical-default) profile. This fixes the four
named failure modes (fragile below top-tier models; doesn't teach source→transform→sink→wiring;
magic-box opacity; high single-inference latency) by decomposing both the rule/prompt
blocks (per-stage skill files) and the flow/UX (a staged stepper that shows the logic),
and by adding a **web_scrape recipe** so the canonical pipeline composes deterministically
(zero LLM calls at compose time). Target version **0.7.0**, pre-release, no feature flag,
no backward-compat.

**Architecture:** One engine, two profiles. `composer/guided/` (today's audited
source → sink → recipe-match → transforms wizard) becomes *the* staged workflow engine.
It gains a **fifth, global stage** `STEP_4_WIRE` (appended after transforms — the stage SET
is a global engine property, the frozen-total `GuidedStep` enum, **not** a `WorkflowProfile`
field, D14). The `WorkflowProfile` (internal plumbing; persisted on `GuidedSession` in the
`composition_states.composer_meta` JSON column) gates *behaviour* — entry seed, per-stage
coaching copy, advisor checkpoints, recipe-match, welcome/graduation bookends — but never the
stage set. The tutorial constructs the one concrete `TUTORIAL_PROFILE` at a new
`POST /guided/start` entry endpoint; the empty profile is a canonical value (not `None`) that
reproduces live-guided behaviour modulo the new wire stage. A new **web_scrape `RecipeSpec`**
(+ predicate keyed on the URL-row `inline_blob` SOURCE — web_scrape is a *transform*) lets the
canonical `inline_blob → web_scrape → llm_rate → field_mapper → jsonl` pipeline compose with
zero LLM calls and reach the wire stage via the deterministic recipe-apply path. Per-stage
interpretation review reuses the freeform `interpretation_events` store/UI (no new backend
guided `TurnType`, D12). The advisor END sign-off is a **pre-terminal gate inside the
`STEP_4_WIRE` branch**, gated on the server-owned `profile.advisor_checkpoints` (D13). The
session DB is **purged** (not migrated) via a `GUIDED_SESSION_SCHEMA_VERSION` 5→6 +
`SESSION_SCHEMA_EPOCH` 22→23 bump (D15).

**Tech Stack:** Python 3.12 (FastAPI web service under `src/elspeth/web/`, SQLite session
DB, Pydantic strict response models, frozen-dataclass domain state, LiteLLM advisor path);
TypeScript/React frontend under `src/elspeth/web/frontend/` (Vitest unit, Playwright E2E);
elspeth-lints trust-tier static analysis + wardline taint gate; pytest + ruff + mypy.

## Global Constraints

- Target version 0.7.0 (pre-release); **no feature flag**, **no backward-compat** — in-place migration (D10), remove the big-bang components.
- Verify ALL code against the WORKTREE ONLY: `/home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth`. Do NOT read `/home/john/elspeth/src` (a different, decomposed branch). Loomweave indexed the decomposed branch — do not trust its route line numbers; Read/Grep the worktree directly.
- The canonical pipeline is exactly `inline_blob → web_scrape → llm_rate → field_mapper(select_only, raw-HTML cleanup) → jsonl` (web_scrape is a TRANSFORM; the recipe predicate keys the URL-row `inline_blob` SOURCE, never `web_scrape`).
- **Shield advisory stays LIVE:** the web_scrape recipe omits the *unbuildable* `azure_prompt_shield` hard node, but the existing medium-severity prompt-shield advisory warning (`prompt_shield_recommendation_warning_pairs`) MUST remain present in the wire validation payload. Tests pin the advisory's **presence** (+ absence of the hard node), not its absence; comment-reference `elspeth-abb2cb0931`. Do not let the flagship example hide the signal.
- The advisor terminal END sign-off is **profile-gated** on the server-owned `profile.advisor_checkpoints` (closed-enum, server-constructed — a client cannot flip it). The empty/live-guided profile gets the wire stage but no mandatory terminal advisor call (so D14's global wire stage is not a blocking-advisor regression for live guided).
- The stage SET is a **global** engine property (frozen-total `GuidedStep` enum), NOT a `WorkflowProfile` field (D14). A future profile must add NO new state-machine dispatch branch (the additivity acceptance test).
- Gate commands (§9.2) — run ALL before claiming completion:
  - `uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/`
  - `uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/`
  - `uv run mypy src/ elspeth-lints/src/`
  - `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth`
  - `uv run python scripts/cicd/check_slot_type_cross_language.py` (SlotType / guided.ts mirror gate — this work edits `guided.ts`)
  - targeted `pytest` over the new guided/tutorial/recipe test files
  - frontend (run from `src/elspeth/web/frontend`): `npm run typecheck`, `npm test -- --run`, `npm run build`, `npm run test:e2e` (+ `test:e2e:staging`)
  - `wardline scan . --fail-on ERROR` (exit 0 clean / 1 gate tripped / 2 tool error) — B1/B3/recipe touch externally-fed advisor/user-text trust boundaries; fix at the boundary, not the sink. See `.agents/skills/wardline-gate/SKILL.md`.
- elspeth-lints trust-tier + trust_boundary honesty gate apply: B1/B3 add/move Tier-3 advisor/user-text handling; new `@trust_boundary` decorations must pass `trust_boundary.tests,scope,tier`. Operator-triggered advisor escapes must NOT route through Tier-3 `_validate_advisor_arguments`; backend checkpoints must not consume unvalidated user text.

---

## File Structure

(see structured `file_structure` — every created/modified file with a one-line responsibility)

## Shared Interfaces (canonical names — use verbatim)

(see structured `interface_contract` — canonical names/types/enum values/signatures every task must use VERBATIM)

### New `TurnType` member
- `TurnType.CONFIRM_WIRING = "confirm_wiring"` — the single new turn type for the wire stage (also carries the advisor-revise re-emit; an attached `advisor_findings` payload key distinguishes the revise re-emit from the initial confirm). Register it in `_LEGAL_TURN_MATRIX[STEP_4_WIRE]`, `_REQUIRED_KEYS` (both TOTAL over `TurnType` — omission crashes at import), and `_NESTED_SHAPES`. Mirror as `"confirm_wiring"` in the `guided.ts` `TurnType` union.

### New `GuidedStep` member (global, appended)
- `GuidedStep.STEP_4_WIRE = "step_4_wire"` — appended LAST (append-only; mid-insert renumbers the wire protocol and is forbidden). Add to: `_LEGAL_TURN_MATRIX`, `prompts._STEP_FILE_NAMES` + `_STEP_PLAYBOOK_ORDER` (+ create `skills/step_4_wire.md` — wiring CONSTRAINTS only, no UX copy), the TWO duplicated `_ORDER` tuples (`emitters.py` and `sessions/routes/_helpers.py`), the `guided.ts` `GuidedStep` union, and the `step_advance` branch dispatch.

### `run_signoff_checkpoint` (new public protocol method)
```python
async def run_signoff_checkpoint(
    self,
    *,
    state: CompositionState,
    session_id: str | None,
    recorder: BufferingRecorder | None,
    progress: ComposerProgressSink | None = None,
) -> AdvisorCheckpointVerdict: ...
```
Add to the `ComposerService` Protocol; `ComposerServiceImpl` delegates to the existing private `_run_advisor_checkpoint(phase="end", ...)`. `AdvisorCheckpointVerdict` gains a `failure_class: Literal["none","unavailable","malformed"] = "none"` field (added in P5.3): the existing `_run_advisor_checkpoint` collapses EVERY exception — timeout/auth/transport AND malformed/parse — to `ok=False` (service.py:4210-4230), so `(ok, blocking)` ALONE cannot separate a malformed response (must fail-closed) from a transport outage (may take the audited escape). Verdict classes for D13: CLEAN = `ok and not blocking`; FLAGGED = `ok and blocking` (fail-closed); MALFORMED = `not ok and failure_class=="malformed"` (**fail-closed, no escape**); UNAVAILABLE = `not ok and failure_class=="unavailable"` (escape only at budget-exhaustion).

### Schema version constants
- `GUIDED_SESSION_SCHEMA_VERSION` (`composer/guided/state_machine.py:41`): **5 → 6**.
- `SESSION_SCHEMA_EPOCH` (`sessions/models.py:117`): **22 → 23** (boot fail-close via `_assert_schema_sentinels`).

---

## Phase P0 — Schema & profile foundation

This phase lays the persistence spine the rest of the recut builds on: a new
`WorkflowProfile` value type, two new persisted `GuidedSession` fields, and the
two schema-sentinel bumps (`GUIDED_SESSION_SCHEMA_VERSION` 5→6 and
`SESSION_SCHEMA_EPOCH` 22→23) that make a stale session DB fail loudly at boot
instead of silently 500-ing per row. No UX or behaviour change ships here — the
new fields default to live-guided behaviour (`EMPTY_PROFILE`,
`advisor_checkpoint_passes_used=0`). Spec refs: §4.3, §8 (D15, D16).

All commands run from the worktree root
`/home/john/elspeth/.claude/worktrees/tutorial-staged-recut`. The repo's
`pyproject.toml` `addopts` already deselects `slow/stress/performance/testcontainer`
(line 424), so plain `pytest <path>` is the CI-equivalent selection — never pass
`-o addopts=""`.

---

### Task P0.1: Create the `WorkflowProfile` value type, constants, and closed-enum discriminator

**Files:**
- Create: `src/elspeth/web/composer/guided/profile.py`
- Create: `tests/unit/web/composer/guided/test_profile.py`

**Interfaces:**
- Produces:
  - `class WorkflowProfileKind(StrEnum)` with members `LIVE = "live"`, `TUTORIAL = "tutorial"`.
  - `@dataclass(frozen=True, slots=True) class WorkflowProfile` with fields
    `entry_seed: str | None`, `coaching: bool`, `advisor_checkpoints: bool`,
    `recipe_match: bool`, `bookends: bool`.
  - `WorkflowProfile.to_dict(self) -> dict[str, Any]` (direct-key, all five fields).
  - `WorkflowProfile.from_dict(cls, d: dict[str, Any]) -> WorkflowProfile` (Tier-1 strict, direct-key, `InvariantError` on malformed).
  - `EMPTY_PROFILE = WorkflowProfile(entry_seed=None, coaching=False, advisor_checkpoints=False, recipe_match=True, bookends=False)`.
  - `TUTORIAL_PROFILE = WorkflowProfile(entry_seed=<canonical seed>, coaching=True, advisor_checkpoints=True, recipe_match=True, bookends=True)`.
  - `profile_for_kind(kind: WorkflowProfileKind) -> WorkflowProfile` (closed-enum → constant mapping; server constructs the object, client only supplies the discriminator).
- Consumes: `elspeth.web.composer.guided.errors.InvariantError` (existing).

- [ ] **Step 1: Write the failing constants + field test.**
  Create `tests/unit/web/composer/guided/test_profile.py`:
  ```python
  # tests/unit/web/composer/guided/test_profile.py
  """Tests for WorkflowProfile — frozen value type + closed-enum discriminator."""

  from __future__ import annotations

  import dataclasses

  import pytest

  from elspeth.web.composer.guided.errors import InvariantError
  from elspeth.web.composer.guided.profile import (
      EMPTY_PROFILE,
      TUTORIAL_PROFILE,
      WorkflowProfile,
      WorkflowProfileKind,
      profile_for_kind,
  )


  class TestWorkflowProfileShape:
      def test_is_frozen(self) -> None:
          with pytest.raises(dataclasses.FrozenInstanceError):
              EMPTY_PROFILE.coaching = True  # type: ignore[misc]

      def test_empty_profile_is_live_guided_default(self) -> None:
          assert EMPTY_PROFILE.entry_seed is None
          assert EMPTY_PROFILE.coaching is False
          assert EMPTY_PROFILE.advisor_checkpoints is False
          assert EMPTY_PROFILE.recipe_match is True
          assert EMPTY_PROFILE.bookends is False

      def test_tutorial_profile_enables_coaching_advisor_bookends(self) -> None:
          assert TUTORIAL_PROFILE.coaching is True
          assert TUTORIAL_PROFILE.advisor_checkpoints is True
          assert TUTORIAL_PROFILE.recipe_match is True
          assert TUTORIAL_PROFILE.bookends is True
          assert isinstance(TUTORIAL_PROFILE.entry_seed, str)
          assert TUTORIAL_PROFILE.entry_seed.strip() != ""


  class TestWorkflowProfileKind:
      def test_kind_values_are_closed(self) -> None:
          assert WorkflowProfileKind.LIVE.value == "live"
          assert WorkflowProfileKind.TUTORIAL.value == "tutorial"
          assert {k.value for k in WorkflowProfileKind} == {"live", "tutorial"}

      def test_profile_for_kind_maps_live_to_empty(self) -> None:
          assert profile_for_kind(WorkflowProfileKind.LIVE) is EMPTY_PROFILE

      def test_profile_for_kind_maps_tutorial(self) -> None:
          assert profile_for_kind(WorkflowProfileKind.TUTORIAL) is TUTORIAL_PROFILE

      def test_unknown_kind_string_rejected_by_enum(self) -> None:
          with pytest.raises(ValueError):
              WorkflowProfileKind("bespoke")
  ```

- [ ] **Step 2: Run the test to confirm import failure.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_profile.py -q
  ```
  Expected: collection error `ModuleNotFoundError: No module named 'elspeth.web.composer.guided.profile'`.

- [ ] **Step 3: Create `profile.py` with the dataclass, enum, constants, and mapping.**
  Create `src/elspeth/web/composer/guided/profile.py`:
  ```python
  """WorkflowProfile — the per-session engine-behaviour knob bundle.

  See docs/superpowers/specs/2026-06-22-tutorial-staged-recut-design.md §4.3 / D11.

  Trust tier: Tier 1 (persisted, audit). Coercion forbidden — ``from_dict`` uses
  direct key access (never ``.get()``), constructs values directly, and chains
  exceptions via ``from exc``. The round-trip invariant holds:
      obj == WorkflowProfile.from_dict(obj.to_dict())

  The stage SET is a global engine property (D14), NOT a profile field — there is
  deliberately no ``stages`` field here. ``EMPTY_PROFILE`` is the canonical
  default and equals live-guided behaviour; ``TUTORIAL_PROFILE`` is the single
  wired non-default instance. The start endpoint accepts only the closed
  ``WorkflowProfileKind`` discriminator from the client and constructs the object
  server-side, so a client cannot inject an arbitrary profile or weaken gating.
  """

  from __future__ import annotations

  from dataclasses import dataclass
  from enum import StrEnum
  from typing import Any

  from elspeth.web.composer.guided.errors import InvariantError

  # The canonical "cool government pages" entry seed for the tutorial profile.
  # Pre-fills the source-intent prompt so the first-run tutorial starts from a
  # known-good URL-list inline_blob rather than an empty canvas. Consumed
  # server-side at POST /guided/start; NOT rendered on GET /guided.
  _TUTORIAL_ENTRY_SEED = (
      "Rate how 'cool' each Australian government web page is on a 1-10 scale, "
      "reading each URL from the list below."
  )


  class WorkflowProfileKind(StrEnum):
      """Closed discriminator the start endpoint accepts from the client.

      The server maps each member to a server-constructed ``WorkflowProfile``
      constant; the client never supplies profile fields directly.
      """

      LIVE = "live"
      TUTORIAL = "tutorial"


  @dataclass(frozen=True, slots=True)
  class WorkflowProfile:
      """Per-session engine-behaviour knobs.

      All five fields are scalars (``entry_seed`` is ``str | None``, the rest are
      ``bool``), so the frozen dataclass is already deeply immutable — no
      ``freeze_fields`` guard is needed.
      """

      entry_seed: str | None
      coaching: bool
      advisor_checkpoints: bool
      recipe_match: bool
      bookends: bool

      def to_dict(self) -> dict[str, Any]:
          """Serialize to a plain JSON-serialisable dict (direct-key)."""
          return {
              "entry_seed": self.entry_seed,
              "coaching": self.coaching,
              "advisor_checkpoints": self.advisor_checkpoints,
              "recipe_match": self.recipe_match,
              "bookends": self.bookends,
          }

      @classmethod
      def from_dict(cls, d: dict[str, Any]) -> WorkflowProfile:
          """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data.

          Strict means: (a) reject UNKNOWN keys — a forked/tampered profile blob must
          not smuggle fields past the closed schema; (b) reject type mismatches on the
          server-owned gate bools — a present-but-mistyped value (e.g. the JSON string
          "false", which is truthy) must NOT silently flip the advisor gate. Missing
          keys already raise via direct-key access below.
          """
          _allowed = {"entry_seed", "coaching", "advisor_checkpoints", "recipe_match", "bookends"}
          extra = set(d) - _allowed
          if extra:
              raise InvariantError(f"WorkflowProfile.from_dict: unexpected keys {sorted(extra)!r}")
          try:
              # `isinstance(x, bool)` excludes 1/0 (bool is an int subclass), so a stray
              # numeric/string for a gate is rejected rather than silently coerced.
              for _k in ("coaching", "advisor_checkpoints", "recipe_match", "bookends"):
                  if not isinstance(d[_k], bool):
                      raise InvariantError(
                          f"WorkflowProfile.from_dict: {_k} must be bool, got {type(d[_k]).__name__}"
                      )
              if d["entry_seed"] is not None and not isinstance(d["entry_seed"], str):
                  raise InvariantError("WorkflowProfile.from_dict: entry_seed must be str | None")
              return cls(
                  entry_seed=d["entry_seed"],
                  coaching=d["coaching"],
                  advisor_checkpoints=d["advisor_checkpoints"],
                  recipe_match=d["recipe_match"],
                  bookends=d["bookends"],
              )
          except KeyError as exc:
              raise InvariantError(f"WorkflowProfile.from_dict: malformed record {d!r}") from exc


  # Canonical default == live-guided behaviour. recipe_match stays on (the
  # recipe-match short-circuit is live-guided's existing behaviour); coaching,
  # advisor_checkpoints, and bookends are off for live guided.
  EMPTY_PROFILE = WorkflowProfile(
      entry_seed=None,
      coaching=False,
      advisor_checkpoints=False,
      recipe_match=True,
      bookends=False,
  )

  # The single wired non-default instance: canonical seed, coaching on,
  # advisor_checkpoints on (terminal END sign-off gate), recipe_match on,
  # welcome/graduation bookends on.
  TUTORIAL_PROFILE = WorkflowProfile(
      entry_seed=_TUTORIAL_ENTRY_SEED,
      coaching=True,
      advisor_checkpoints=True,
      recipe_match=True,
      bookends=True,
  )

  _KIND_TO_PROFILE: dict[WorkflowProfileKind, WorkflowProfile] = {
      WorkflowProfileKind.LIVE: EMPTY_PROFILE,
      WorkflowProfileKind.TUTORIAL: TUTORIAL_PROFILE,
  }


  def profile_for_kind(kind: WorkflowProfileKind) -> WorkflowProfile:
      """Map a closed-enum discriminator to its server-constructed profile."""
      return _KIND_TO_PROFILE[kind]
  ```

- [ ] **Step 4: Run the test to confirm it passes.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_profile.py -q
  ```
  Expected: `8 passed`.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/composer/guided/profile.py tests/unit/web/composer/guided/test_profile.py
  git commit -m "feat(composer/guided): add WorkflowProfile value type + closed-enum kind

Adds WorkflowProfile (frozen, slots) with the five §4.3 fields plus EMPTY_PROFILE
(live-guided default) / TUTORIAL_PROFILE constants and the WorkflowProfileKind
closed discriminator. P0.1 of the tutorial staged recut.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.2: Add strict `to_dict`/`from_dict` round-trip + rejection tests for `WorkflowProfile`

**Files:**
- Modify: `tests/unit/web/composer/guided/test_profile.py`

**Interfaces:**
- Consumes: `WorkflowProfile.to_dict` / `WorkflowProfile.from_dict` (from P0.1); `InvariantError`.

- [ ] **Step 1: Append the strict-serialisation tests.**
  Append to `tests/unit/web/composer/guided/test_profile.py`:
  ```python
  class TestWorkflowProfileSerialisation:
      def test_empty_profile_round_trips(self) -> None:
          assert WorkflowProfile.from_dict(EMPTY_PROFILE.to_dict()) == EMPTY_PROFILE

      def test_tutorial_profile_round_trips(self) -> None:
          assert WorkflowProfile.from_dict(TUTORIAL_PROFILE.to_dict()) == TUTORIAL_PROFILE

      def test_to_dict_emits_all_five_keys(self) -> None:
          assert set(EMPTY_PROFILE.to_dict()) == {
              "entry_seed",
              "coaching",
              "advisor_checkpoints",
              "recipe_match",
              "bookends",
          }

      def test_entry_seed_none_round_trips_as_none(self) -> None:
          d = EMPTY_PROFILE.to_dict()
          assert d["entry_seed"] is None
          assert WorkflowProfile.from_dict(d).entry_seed is None

      def test_from_dict_rejects_missing_key(self) -> None:
          d = TUTORIAL_PROFILE.to_dict()
          del d["advisor_checkpoints"]
          with pytest.raises(InvariantError, match=r"WorkflowProfile\.from_dict"):
              WorkflowProfile.from_dict(d)

      def test_from_dict_uses_direct_key_not_get_default(self) -> None:
          # An empty dict must crash, never silently fabricate a profile.
          with pytest.raises(InvariantError, match=r"WorkflowProfile\.from_dict"):
              WorkflowProfile.from_dict({})

      def test_from_dict_rejects_unknown_key(self) -> None:
          # A forked/tampered blob with an injected field must be rejected, not
          # silently ignored — the closed schema is the tamper boundary.
          d = {**TUTORIAL_PROFILE.to_dict(), "stages": ["smuggled"]}
          with pytest.raises(InvariantError, match=r"unexpected keys"):
              WorkflowProfile.from_dict(d)

      def test_from_dict_rejects_non_bool_advisor_checkpoints(self) -> None:
          # The JSON string "false" is TRUTHY — a present-but-mistyped gate must
          # raise, never silently flip the server-owned advisor gate.
          d = {**TUTORIAL_PROFILE.to_dict(), "advisor_checkpoints": "false"}
          with pytest.raises(InvariantError, match=r"advisor_checkpoints must be bool"):
              WorkflowProfile.from_dict(d)
  ```

- [ ] **Step 2: Run the full profile suite to confirm green.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_profile.py -q
  ```
  Expected: `16 passed`.

- [ ] **Step 3: Commit.**
  ```bash
  git add tests/unit/web/composer/guided/test_profile.py
  git commit -m "test(composer/guided): strict WorkflowProfile round-trip + rejection

Locks the direct-key (no .get()) Tier-1 contract: missing-key and empty-dict
loads crash with InvariantError; both constants round-trip cleanly. P0.2.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.3: Bump `GUIDED_SESSION_SCHEMA_VERSION` 5→6 and update the two version-pinning tests

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py:41`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py:384-403`

**Interfaces:**
- Produces: `GUIDED_SESSION_SCHEMA_VERSION == 6` (module constant, `state_machine.py:41`).
- Consumes: nothing new.

Note: the v6 fields themselves are added in P0.4. This task bumps ONLY the
version constant + the three tests that pin it (the `== 5` assertions and the
`schema_version = 4` rejection test stays valid because 4 ≠ 6). Doing the bump
first means P0.4's new-field round-trip tests are written against the already-v6
constant — no double-edit of the same lines.

- [ ] **Step 1: Update the two `== 5` assertions and the rejection test to v6.**
  In `tests/unit/web/composer/guided/test_state_machine.py`, edit the
  `test_guided_session_schema_version_bumped_for_inspection_facts` body
  (line 385) from `assert GUIDED_SESSION_SCHEMA_VERSION == 5` to
  `assert GUIDED_SESSION_SCHEMA_VERSION == 6`, and the
  `test_guided_session_to_dict_includes_schema_version` body (line 389) from
  `assert sess.to_dict()["schema_version"] == 5` to
  `assert sess.to_dict()["schema_version"] == 6`. Leave
  `test_guided_session_rejects_old_schema_version` (line 398) untouched — it sets
  `old["schema_version"] = 4` and asserts `unsupported schema_version 4`, which is
  still correct after the bump (4 is still rejected by a v6 build).

- [ ] **Step 2: Run the state-machine version tests to confirm they now fail.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py -k "schema_version" -q
  ```
  Expected: 2 failures —
  `test_guided_session_schema_version_bumped_for_inspection_facts` and
  `test_guided_session_to_dict_includes_schema_version` —
  both `assert 5 == 6` (the build constant is still 5).

- [ ] **Step 3: Bump the constant.**
  In `src/elspeth/web/composer/guided/state_machine.py`, change line 41 from
  `GUIDED_SESSION_SCHEMA_VERSION = 5` to `GUIDED_SESSION_SCHEMA_VERSION = 6`. Also
  update the comment block above it (lines 39-40) to read:
  ```python
  # Pre-v6 persisted sessions are intentionally incompatible with v6: the
  # operator must delete the guided sessions DB before deploying this change.
  GUIDED_SESSION_SCHEMA_VERSION = 6
  ```

- [ ] **Step 4: Run the same selection to confirm pass.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py -k "schema_version" -q
  ```
  Expected: `3 passed` (the two bumped tests + the still-valid v4 rejection test).

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/composer/guided/state_machine.py tests/unit/web/composer/guided/test_state_machine.py
  git commit -m "feat(composer/guided): bump GUIDED_SESSION_SCHEMA_VERSION 5->6

Pre-v6 sessions are intentionally incompatible (operator purges the guided
sessions DB on deploy). Version-pin tests updated to 6; the v4 rejection test
stays valid. P0.3; fields land in P0.4.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.4: Add `profile` + `advisor_checkpoint_passes_used` fields to `GuidedSession` with strict serialisation and parameterised `initial`

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py:27` (import), `:257-332` (fields), `:342-351` (`initial`), `:353-391` (`to_dict`), `:393-458` (`from_dict`)
- Modify: `tests/unit/web/composer/guided/test_state_machine.py` (new round-trip + default tests)

**Interfaces:**
- Consumes: `WorkflowProfile`, `EMPTY_PROFILE` (from P0.1); `GUIDED_SESSION_SCHEMA_VERSION == 6` (from P0.3).
- Produces:
  - `GuidedSession.profile: WorkflowProfile = EMPTY_PROFILE` (new frozen field).
  - `GuidedSession.advisor_checkpoint_passes_used: int = 0` (new frozen field).
  - `GuidedSession.advisor_signoff_escape_offered: bool = False` (new frozen field — D5/B2: persists whether the last END sign-off terminal was a genuine-OUTAGE escape OFFER, so a later request's "complete without sign-off (advisor unreachable)" acknowledgement is honoured WITHOUT re-calling the provider, and so a FLAGGED/MALFORMED-exhausted terminal can never be acknowledged into a bypass. Written by `run_wire_signoff`, P5.5).
  - `GuidedSession.initial(cls, profile: WorkflowProfile = EMPTY_PROFILE) -> GuidedSession` (new signature).
  - `to_dict` adds keys `"profile"` (→ `self.profile.to_dict()`), `"advisor_checkpoint_passes_used"`, and `"advisor_signoff_escape_offered"`.
  - `from_dict` reads `d["profile"]` (via `WorkflowProfile.from_dict`), `d["advisor_checkpoint_passes_used"]` (direct-key, `int(...)`), and `d["advisor_signoff_escape_offered"]` (direct-key, `bool(...)`).

- [ ] **Step 1: Write the failing new-field tests.**
  Append to `tests/unit/web/composer/guided/test_state_machine.py` (inside the
  same module-level test area as the other `test_guided_session_roundtrip_*`
  tests — add a new class so the import block can be extended too). First, extend
  the existing `from elspeth.web.composer.guided.state_machine import (` block to
  add nothing new (GuidedSession already imported), but add a new top-of-file
  import line after line 31:
  ```python
  from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE, WorkflowProfile
  ```
  Then append the test class:
  ```python
  class TestGuidedSessionProfileFields:
      def test_initial_defaults_to_empty_profile(self) -> None:
          sess = GuidedSession.initial()
          assert sess.profile == EMPTY_PROFILE
          assert sess.advisor_checkpoint_passes_used == 0
          assert sess.advisor_signoff_escape_offered is False

      def test_initial_accepts_profile_argument(self) -> None:
          sess = GuidedSession.initial(profile=TUTORIAL_PROFILE)
          assert sess.profile == TUTORIAL_PROFILE
          assert sess.advisor_checkpoint_passes_used == 0

      def test_to_dict_emits_profile_and_pass_counter(self) -> None:
          sess = GuidedSession.initial(profile=TUTORIAL_PROFILE)
          d = sess.to_dict()
          assert d["profile"] == TUTORIAL_PROFILE.to_dict()
          assert d["advisor_checkpoint_passes_used"] == 0

      def test_roundtrip_with_tutorial_profile(self) -> None:
          sess = dataclasses.replace(
              GuidedSession.initial(profile=TUTORIAL_PROFILE),
              advisor_checkpoint_passes_used=2,
              advisor_signoff_escape_offered=True,
          )
          restored = GuidedSession.from_dict(sess.to_dict())
          assert restored == sess
          assert restored.profile == TUTORIAL_PROFILE
          assert restored.advisor_checkpoint_passes_used == 2
          assert restored.advisor_signoff_escape_offered is True

      def test_roundtrip_with_empty_profile(self) -> None:
          sess = GuidedSession.initial()
          assert GuidedSession.from_dict(sess.to_dict()) == sess

      def test_from_dict_rejects_missing_profile_key(self) -> None:
          d = GuidedSession.initial().to_dict()
          del d["profile"]
          with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_missing_pass_counter_key(self) -> None:
          d = GuidedSession.initial().to_dict()
          del d["advisor_checkpoint_passes_used"]
          with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_malformed_profile(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["profile"] = {"coaching": True}  # missing the other four keys
          with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
              GuidedSession.from_dict(d)
  ```

- [ ] **Step 2: Run the new tests to confirm failure.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py::TestGuidedSessionProfileFields -q
  ```
  Expected: failures —
  `test_initial_defaults_to_empty_profile` raises `AttributeError: 'GuidedSession' object has no attribute 'profile'`;
  `test_initial_accepts_profile_argument` raises `TypeError: initial() got an unexpected keyword argument 'profile'`.

- [ ] **Step 3: Add the import in `state_machine.py`.**
  After line 37 (`from elspeth.web.composer.source_inspection import ...`) in
  `src/elspeth/web/composer/guided/state_machine.py`, add:
  ```python
  from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
  ```

- [ ] **Step 4: Add the two frozen fields to `GuidedSession`.**
  In `src/elspeth/web/composer/guided/state_machine.py`, the `GuidedSession`
  field list ends at lines 331-332 (`chat_history` / `chat_turn_seq`). Append the
  two new fields immediately after line 332 (`chat_turn_seq: int = 0`):
  ```python
      # P0 (tutorial staged recut, §4.3): the per-session engine-behaviour
      # profile and the persisted advisor-checkpoint pass counter. Both are
      # part of the v6 schema bump. `profile` is a frozen dataclass of scalars
      # (already deeply immutable — no freeze_fields guard needed); the counter
      # is a plain int. The default == live-guided behaviour so existing
      # GuidedSession() / .initial() call sites are unchanged behaviourally.
      profile: WorkflowProfile = EMPTY_PROFILE
      advisor_checkpoint_passes_used: int = 0
  ```

- [ ] **Step 5: Parameterise `GuidedSession.initial`.**
  In `src/elspeth/web/composer/guided/state_machine.py`, replace the `initial`
  classmethod (lines 342-351) with:
  ```python
      @classmethod
      def initial(cls, profile: WorkflowProfile = EMPTY_PROFILE) -> GuidedSession:
          return cls(
              step=GuidedStep.STEP_1_SOURCE,
              history=(),
              step_1_result=None,
              step_2_result=None,
              step_3_proposal=None,
              terminal=None,
              profile=profile,
          )
  ```

- [ ] **Step 6: Add the two `to_dict` keys.**
  In `src/elspeth/web/composer/guided/state_machine.py`, the `to_dict` return
  dict ends with `"chat_turn_seq": self.chat_turn_seq,` (line 390). Add the two
  new keys immediately after that line (before the closing `}` on line 391):
  ```python
              "profile": self.profile.to_dict(),
              "advisor_checkpoint_passes_used": self.advisor_checkpoint_passes_used,
  ```

- [ ] **Step 7: Read the two new `from_dict` keys strictly and pass them to `cls(...)`.**
  In `src/elspeth/web/composer/guided/state_machine.py` `from_dict`, after the
  `chat_turn_seq_raw = d["chat_turn_seq"]` line (line 428), add:
  ```python
              profile_raw = d["profile"]
              advisor_passes_raw = d["advisor_checkpoint_passes_used"]
  ```
  Then in the `return cls(...)` call (the block ending at line 455 with
  `chat_turn_seq=int(chat_turn_seq_raw),`), add the two new kwargs immediately
  after that line (before the closing `)` on line 456):
  ```python
                  profile=WorkflowProfile.from_dict(profile_raw),
                  advisor_checkpoint_passes_used=int(advisor_passes_raw),
  ```

- [ ] **Step 8: Run the new tests to confirm pass.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py::TestGuidedSessionProfileFields -q
  ```
  Expected: `8 passed`.

- [ ] **Step 9: Run the FULL state-machine module to confirm no existing round-trip regressed.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py -q
  ```
  Expected: all pass (the pre-existing `test_guided_session_roundtrip_*` tests
  call `GuidedSession.initial()`, which now defaults `profile=EMPTY_PROFILE` /
  `advisor_checkpoint_passes_used=0` and round-trips cleanly via the new keys).

- [ ] **Step 10: Commit.**
  ```bash
  git add src/elspeth/web/composer/guided/state_machine.py tests/unit/web/composer/guided/test_state_machine.py
  git commit -m "feat(composer/guided): persist profile + advisor pass counter on GuidedSession

Adds frozen profile (WorkflowProfile, default EMPTY_PROFILE) and
advisor_checkpoint_passes_used (int=0) to GuidedSession in the v6 bump; strict
direct-key to_dict/from_dict; GuidedSession.initial(profile=...). The persisted
counter is what makes D13's bounded advisor re-entry real. P0.4.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.5: Thread `profile` through `_initial_composition_state_with_guided_session`

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py:84-95` (import), `:2183-2208` (function)
- Create: `tests/unit/web/sessions/routes/test_initial_composition_state_profile.py`

**Interfaces:**
- Consumes: `WorkflowProfile`, `EMPTY_PROFILE`, `TUTORIAL_PROFILE` (P0.1); `GuidedSession.initial(profile)` (P0.4).
- Produces: `_initial_composition_state_with_guided_session(*, profile: WorkflowProfile = EMPTY_PROFILE) -> CompositionState` (keyword-only `profile`).

Note: the helper is exported in `_helpers.py`'s `__all__` (line 3855). The new
keyword-only param with a default keeps every existing no-arg call site valid;
P7.T1 (the `/guided/start` endpoint) is the first caller to pass a non-default
profile. This task only widens the seam + proves threading.

- [ ] **Step 1: Write the failing threading test.**
  Create `tests/unit/web/sessions/routes/test_initial_composition_state_profile.py`:
  ```python
  # tests/unit/web/sessions/routes/test_initial_composition_state_profile.py
  """_initial_composition_state_with_guided_session threads the WorkflowProfile."""

  from __future__ import annotations

  from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
  from elspeth.web.sessions.routes._helpers import (
      _initial_composition_state_with_guided_session,
  )


  def test_default_is_empty_profile() -> None:
      state = _initial_composition_state_with_guided_session()
      assert state.guided_session is not None
      assert state.guided_session.profile == EMPTY_PROFILE


  def test_threads_tutorial_profile() -> None:
      state = _initial_composition_state_with_guided_session(profile=TUTORIAL_PROFILE)
      assert state.guided_session is not None
      assert state.guided_session.profile == TUTORIAL_PROFILE
      assert state.guided_session.advisor_checkpoint_passes_used == 0
  ```

- [ ] **Step 2: Run to confirm failure.**
  ```bash
  uv run python -m pytest tests/unit/web/sessions/routes/test_initial_composition_state_profile.py -q
  ```
  Expected: `test_threads_tutorial_profile` fails with
  `TypeError: _initial_composition_state_with_guided_session() got an unexpected keyword argument 'profile'`.

- [ ] **Step 3: Add the `WorkflowProfile`/`EMPTY_PROFILE` import.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, the guided import block is at
  lines 84-95 (`from elspeth.web.composer.guided.state_machine import (...)`).
  Immediately after that block (after line 95), add:
  ```python
  from elspeth.web.composer.guided.profile import EMPTY_PROFILE, WorkflowProfile
  ```

- [ ] **Step 4: Add the keyword-only `profile` parameter and thread it.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, change the signature on
  line 2183 from:
  ```python
  def _initial_composition_state_with_guided_session() -> CompositionState:
  ```
  to:
  ```python
  def _initial_composition_state_with_guided_session(
      *, profile: WorkflowProfile = EMPTY_PROFILE
  ) -> CompositionState:
  ```
  and change the `guided_session=GuidedSession.initial(),` line (line 2207) to:
  ```python
          guided_session=GuidedSession.initial(profile),
  ```

- [ ] **Step 5: Run to confirm pass.**
  ```bash
  uv run python -m pytest tests/unit/web/sessions/routes/test_initial_composition_state_profile.py -q
  ```
  Expected: `2 passed`.

- [ ] **Step 6: Run the existing helper callers to confirm no no-arg regression.**
  ```bash
  uv run python -m pytest tests/unit/web/sessions/ tests/unit/web/composer/guided/ -q
  ```
  Expected: all pass (existing no-arg call sites unchanged; default profile == prior behaviour).

- [ ] **Step 7: Commit.**
  ```bash
  git add src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_initial_composition_state_profile.py
  git commit -m "feat(sessions): thread WorkflowProfile through guided-session bootstrap

_initial_composition_state_with_guided_session gains a keyword-only profile=
param (default EMPTY_PROFILE) threaded into GuidedSession.initial(profile).
Existing no-arg call sites unchanged; the start endpoint (P7) is the first
non-default caller. P0.5.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.6: Bump `SESSION_SCHEMA_EPOCH` 22→23 with a boot fail-close guard + update the two epoch-pinning tests

**Files:**
- Modify: `src/elspeth/web/sessions/models.py:113-117` (epoch constant + comment)
- Modify: `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py:48-51`
- Modify: `tests/unit/web/sessions/test_interpretation_events_table.py:199-200`
- Modify: `tests/unit/web/sessions/test_schema.py` (new boot fail-close test for epoch 22)

**Interfaces:**
- Produces: `SESSION_SCHEMA_EPOCH == 23` (`models.py:117`).
- Consumes: existing `initialize_session_schema` → `_assert_schema_sentinels` (`schema.py:106/127-169`) — unchanged code, the bump alone makes a 22-stamped DB fail-close.

- [ ] **Step 1: Write the failing boot fail-close test.**
  Append to `tests/unit/web/sessions/test_schema.py` (it already imports
  `SessionSchemaError`, `create_session_engine`, `initialize_session_schema`,
  `text`, and `pytest` — confirm at the top of the file; the existing
  `test_initialize_session_schema_rejects_partial_stale_schema` uses all of them):
  First extend the existing `from elspeth.web.sessions.models import ...` import
  (line 13) to also bring in `SESSION_DB_APPLICATION_ID` — it is exported from
  `sessions/models.py` (value `0x454C5350`). Then append:
  ```python
  def test_initialize_session_schema_rejects_epoch_22_database() -> None:
      """A DB stamped at the prior epoch (22) fail-closes loudly at boot.

      Regression guard for the tutorial-staged-recut 22->23 bump (D15): a
      sessions.db carried across the bump without an operator purge must crash
      with the actionable delete-and-restart message, not silently lazy-500.
      """
      eng = create_session_engine("sqlite:///:memory:")
      with eng.begin() as conn:
          # A user table is REQUIRED here: initialize_session_schema treats a
          # table-less DB as FRESH (runs metadata.create_all, then re-stamps the
          # CURRENT epoch) and the stale-epoch guard never runs — so without this
          # CREATE TABLE the test passes vacuously (DID NOT RAISE) even post-bump.
          # A user table routes us down the existing-DB (validate, never recreate)
          # branch where _assert_schema_sentinels checks the stamped epoch. Mirrors
          # test_schema.py:124-130's CREATE TABLE pattern.
          conn.execute(text("CREATE TABLE sessions (id VARCHAR PRIMARY KEY)"))
          # Stamp the ELSP application_id (so we hit the "ours but stale" branch,
          # not the "not ELSPETH" branch) + the prior epoch onto the file.
          conn.execute(text(f"PRAGMA application_id = {SESSION_DB_APPLICATION_ID}"))
          conn.execute(text("PRAGMA user_version = 22"))

      with pytest.raises(SessionSchemaError, match="SESSION_SCHEMA_EPOCH"):
          initialize_session_schema(eng)
  ```
  Importing the constant (rather than hardcoding `0x454C5350`/`1162109264`) keeps
  the test resilient if the app_id ever rotates. The epoch guard fires on
  `user_version != SESSION_SCHEMA_EPOCH` regardless of app_id, so this stamps the
  "ours but stale" path specifically.

- [ ] **Step 2: Run to confirm failure.**
  ```bash
  uv run python -m pytest "tests/unit/web/sessions/test_schema.py::test_initialize_session_schema_rejects_epoch_22_database" -q
  ```
  Expected: FAIL — `DID NOT RAISE <class 'SessionSchemaError'>` (with epoch still
  22, a 22-stamped DB matches the current epoch and is accepted).

- [ ] **Step 3: Update the two epoch-pinning tests to 23.**
  In `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py`
  (`test_blob_inline_resolutions_schema_epoch_is_22`, lines 48-51): change both
  `22` literals to `23` and rename the function to
  `test_blob_inline_resolutions_schema_epoch_is_23`:
  ```python
  def test_blob_inline_resolutions_schema_epoch_is_23(engine) -> None:
      assert SESSION_SCHEMA_EPOCH == 23
      with engine.connect() as conn:
          assert conn.execute(text("PRAGMA user_version")).scalar_one() == 23
  ```
  In `tests/unit/web/sessions/test_interpretation_events_table.py`
  (`test_proposal_provenance_schema_cohort_epoch_is_22`, lines 199-200): change
  `22` to `23` and rename to
  `test_proposal_provenance_schema_cohort_epoch_is_23`:
  ```python
  def test_proposal_provenance_schema_cohort_epoch_is_23() -> None:
      assert SESSION_SCHEMA_EPOCH == 23
  ```

- [ ] **Step 4: Bump the epoch constant + add the epoch-23 comment.**
  In `src/elspeth/web/sessions/models.py`, the epoch-history comment ends at
  lines 113-116 (the `#   22 → ...` block) and the constant is at line 117.
  Append a new history line after line 116 and bump the constant:
  ```python
  #   23 → no SQL-shape change; bumped in lockstep with GUIDED_SESSION_SCHEMA_VERSION
  #        5→6 (composer_meta JSON adds GuidedSession.profile +
  #        advisor_checkpoint_passes_used) so a stale sessions.db fail-closes at
  #        boot via _assert_schema_sentinels instead of lazy-500-ing per guided
  #        row on GuidedSession.from_dict. Pre-release delete-and-recreate policy;
  #        see docs/runbooks/staging-session-db-recreation.md.
  SESSION_SCHEMA_EPOCH = 23
  ```

- [ ] **Step 5: Run the boot fail-close test + the two pinning tests to confirm pass.**
  ```bash
  uv run python -m pytest \
    "tests/unit/web/sessions/test_schema.py::test_initialize_session_schema_rejects_epoch_22_database" \
    "tests/unit/web/sessions/test_blob_inline_resolutions_schema.py::test_blob_inline_resolutions_schema_epoch_is_23" \
    "tests/unit/web/sessions/test_interpretation_events_table.py::test_proposal_provenance_schema_cohort_epoch_is_23" -q
  ```
  Expected: `3 passed`.

- [ ] **Step 6: Run the whole sessions schema test surface to confirm no other `== 22` left red.**
  ```bash
  uv run python -m pytest tests/unit/web/sessions/ -q -k "schema or epoch or interpretation_events or blob_inline"
  ```
  Expected: all pass. If any other test references the literal `22` for the
  session epoch, it surfaces here — fix it to `23` in the same task (do NOT defer
  to an observation; it is in-scope for the epoch bump).

- [ ] **Step 7: Commit.**
  ```bash
  git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_blob_inline_resolutions_schema.py tests/unit/web/sessions/test_interpretation_events_table.py tests/unit/web/sessions/test_schema.py
  git commit -m "feat(sessions): bump SESSION_SCHEMA_EPOCH 22->23 (boot fail-close on stale DB)

Lockstep with GUIDED_SESSION_SCHEMA_VERSION 5->6: no SQL-shape change, but the
epoch bump turns _assert_schema_sentinels into a loud boot guard so a sessions.db
carried across the cutover crashes with the delete-and-restart message instead of
lazy-500-ing per guided row (D15). Epoch-pin tests + a 22-rejection regression
test updated. P0.6.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P0.7: Phase-close gate sweep (lint, type, full phase test surface)

**Files:** none (verification only).

**Interfaces:** none new.

- [ ] **Step 1: Lint the files this phase touched.**
  ```bash
  uv run ruff check \
    src/elspeth/web/composer/guided/profile.py \
    src/elspeth/web/composer/guided/state_machine.py \
    src/elspeth/web/sessions/routes/_helpers.py \
    src/elspeth/web/sessions/models.py \
    tests/unit/web/composer/guided/test_profile.py \
    tests/unit/web/composer/guided/test_state_machine.py \
    tests/unit/web/sessions/routes/test_initial_composition_state_profile.py
  ```
  Expected: `All checks passed!`.

- [ ] **Step 2: Type-check the changed source modules.**
  ```bash
  uv run mypy \
    src/elspeth/web/composer/guided/profile.py \
    src/elspeth/web/composer/guided/state_machine.py \
    src/elspeth/web/sessions/routes/_helpers.py \
    src/elspeth/web/sessions/models.py
  ```
  Expected: `Success: no issues found` (or no NEW errors vs. the pre-phase
  baseline for `_helpers.py` — if pre-existing errors are reported there,
  diff against `git stash`-clean to confirm none are newly introduced by this phase).

- [ ] **Step 3: Run the full P0 test surface.**
  ```bash
  uv run python -m pytest \
    tests/unit/web/composer/guided/test_profile.py \
    tests/unit/web/composer/guided/test_state_machine.py \
    tests/unit/web/sessions/routes/test_initial_composition_state_profile.py \
    tests/unit/web/sessions/ -q
  ```
  Expected: all pass, zero failures.

- [ ] **Step 4: Confirm no stale epoch/version literal remains anywhere in the suite.**
  ```bash
  grep -rn "SESSION_SCHEMA_EPOCH == 22\|user_version.*22\|GUIDED_SESSION_SCHEMA_VERSION == 5\|\"schema_version\"\] == 5" tests/ src/ || echo "CLEAN — no stale pins"
  ```
  Expected: `CLEAN — no stale pins` (no match). Any hit here is an in-scope
  follow-up for this phase — fix it before closing P0.

- [ ] **Step 5: Final phase commit (only if Step 4 surfaced and fixed a stray pin; otherwise skip).**
  ```bash
  git add -A && git commit -m "test(sessions): sweep stale schema-version/epoch pins for P0 bumps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

## Phase P1 — Wire stage skeleton (STEP_4_WIRE) + terminal-stamp move

This phase appends `GuidedStep.STEP_4_WIRE` (the 5th, append-only member) and
`TurnType.CONFIRM_WIRING`, wires every coordinated touchpoint named in spec §4.2,
moves terminal-stamping out of both completion seams into a new `STEP_4_WIRE`
handler that gates on `validate().is_valid` only (advisor sign-off comes in P5.6),
and pins the terminal-stamp invariant. The wire turn payload is a **skeleton** here
(carries only the `validate()` summary); the full two-read topology + edge_contracts
blob is **P2.4**'s job, and the dispatcher wire-turn emission + `CONFIRM_WIRING`
dispatch branch is **P2.9**.

P1 does **not** depend on any P0 symbol: it keeps `GuidedSession.initial()` no-arg
and adds no profile gating. P0 (profile field + v6 bump) and P1 (enum append) both
touch `state_machine.py` but in disjoint regions; this phase is sequenced after P0
in the foundation, so `GUIDED_SESSION_SCHEMA_VERSION` is already 6 when P1 runs.
Where P1 adds a new persisted reachable step, the existing strict `from_dict`
round-trip already covers `STEP_4_WIRE` because `step` is serialised via
`GuidedStep(d["step"])`.

### Task P1.1: Append GuidedStep.STEP_4_WIRE + TurnType.CONFIRM_WIRING to the protocol totals

**Files:**
- Modify `src/elspeth/web/composer/guided/protocol.py:16-25` (TurnType enum)
- Modify `src/elspeth/web/composer/guided/protocol.py:96-103` (GuidedStep enum)
- Modify `src/elspeth/web/composer/guided/protocol.py:169-192` (`_LEGAL_TURN_MATRIX`)
- Modify `src/elspeth/web/composer/guided/protocol.py:200-218` (`_REQUIRED_KEYS`)
- Modify `src/elspeth/web/composer/guided/protocol.py:243-253` (`_NESTED_SHAPES`)
- Modify `tests/unit/web/composer/guided/test_protocol.py`

**Interfaces:**
- Produces: `GuidedStep.STEP_4_WIRE = "step_4_wire"` (appended last); `TurnType.CONFIRM_WIRING = "confirm_wiring"`.
- Produces: `_LEGAL_TURN_MATRIX[GuidedStep.STEP_4_WIRE] == frozenset({TurnType.CONFIRM_WIRING})`.
- Produces: `_REQUIRED_KEYS[TurnType.CONFIRM_WIRING] == frozenset({"topology", "edge_contracts", "semantic_contracts"})`.
- Produces: `_NESTED_SHAPES[TurnType.CONFIRM_WIRING] == ()` (no nested validation at the skeleton stage).
- Consumes: existing `legal_turn_types_for`, `validate_payload` (unchanged signatures).

- [ ] **Step 1: Write a failing test for the new enum members + totality.**
  Append to `tests/unit/web/composer/guided/test_protocol.py` (the existing
  `TestTurnType.test_six_turn_types_defined` at line 21 hardcodes six — update it
  to seven and add the new step/matrix assertions):

  ```python
  # tests/unit/web/composer/guided/test_protocol.py
  from elspeth.web.composer.guided.protocol import (
      GuidedStep,
      _LEGAL_TURN_MATRIX,
      _NESTED_SHAPES,
      _REQUIRED_KEYS,
      legal_turn_types_for,
      validate_payload,
  )


  class TestStep4WireProtocol:
      def test_step_4_wire_is_appended_last(self) -> None:
          members = list(GuidedStep)
          assert members[-1] is GuidedStep.STEP_4_WIRE
          assert GuidedStep.STEP_4_WIRE.value == "step_4_wire"

      def test_confirm_wiring_turn_type_present(self) -> None:
          assert TurnType.CONFIRM_WIRING.value == "confirm_wiring"
          assert TurnType("confirm_wiring") is TurnType.CONFIRM_WIRING

      def test_legal_turn_matrix_total_and_wire_entry(self) -> None:
          assert set(_LEGAL_TURN_MATRIX.keys()) == set(GuidedStep)
          assert legal_turn_types_for(GuidedStep.STEP_4_WIRE) == frozenset(
              {TurnType.CONFIRM_WIRING}
          )

      def test_required_keys_total_over_turn_type(self) -> None:
          assert set(_REQUIRED_KEYS.keys()) == set(TurnType)
          assert _REQUIRED_KEYS[TurnType.CONFIRM_WIRING] == frozenset(
              {"topology", "edge_contracts", "semantic_contracts"}
          )

      def test_nested_shapes_total_over_turn_type(self) -> None:
          assert set(_NESTED_SHAPES.keys()) == set(TurnType)
          assert _NESTED_SHAPES[TurnType.CONFIRM_WIRING] == ()

      def test_confirm_wiring_payload_validates(self) -> None:
          payload = {"topology": {}, "edge_contracts": [], "semantic_contracts": []}
          assert validate_payload(TurnType.CONFIRM_WIRING, payload) is None

      def test_confirm_wiring_payload_missing_key_rejected(self) -> None:
          err = validate_payload(TurnType.CONFIRM_WIRING, {"topology": {}})
          assert err is not None
          assert "confirm_wiring" in err
  ```

  Also update the existing six-type test to seven:

  ```python
      def test_six_turn_types_defined(self) -> None:
          expected = {
              "inspect_and_confirm",
              "single_select",
              "multi_select_with_custom",
              "schema_form",
              "propose_chain",
              "recipe_offer",
              "confirm_wiring",
          }
          assert {t.value for t in TurnType} == expected
  ```

- [ ] **Step 2: Run the test to confirm it fails.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_protocol.py -q
  ```
  Expected failure: `AttributeError: STEP_4_WIRE` / `CONFIRM_WIRING` (the enum
  members do not exist yet) and `KeyError` on the matrix/required-keys lookups.

- [ ] **Step 3: Append `CONFIRM_WIRING` to the `TurnType` StrEnum.**
  Edit `protocol.py:16-25`, appending the member **last** (the closed taxonomy is
  append-only):

  ```python
  class TurnType(StrEnum):
      """The closed taxonomy of turn types the protocol allows."""

      INSPECT_AND_CONFIRM = "inspect_and_confirm"
      SINGLE_SELECT = "single_select"
      MULTI_SELECT_WITH_CUSTOM = "multi_select_with_custom"
      SCHEMA_FORM = "schema_form"
      PROPOSE_CHAIN = "propose_chain"
      RECIPE_OFFER = "recipe_offer"
      CONFIRM_WIRING = "confirm_wiring"
  ```

- [ ] **Step 4: Append `STEP_4_WIRE` to the `GuidedStep` StrEnum.**
  Edit `protocol.py:96-103`, appending **last** (mid-insert is forbidden — it would
  renumber the wire protocol ordinals):

  ```python
  class GuidedStep(StrEnum):
      """Wizard step pointer."""

      STEP_1_SOURCE = "step_1_source"
      STEP_2_SINK = "step_2_sink"
      STEP_2_5_RECIPE_MATCH = "step_2_5_recipe_match"
      STEP_3_TRANSFORMS = "step_3_transforms"
      STEP_4_WIRE = "step_4_wire"
  ```

- [ ] **Step 5: Add the `STEP_4_WIRE` row to `_LEGAL_TURN_MATRIX`.**
  Edit `protocol.py:169-192`, adding the new key after `STEP_3_TRANSFORMS`:

  ```python
      GuidedStep.STEP_3_TRANSFORMS: frozenset(
          {
              TurnType.PROPOSE_CHAIN,
              TurnType.SINGLE_SELECT,
              TurnType.SCHEMA_FORM,
          }
      ),
      GuidedStep.STEP_4_WIRE: frozenset({TurnType.CONFIRM_WIRING}),
  }
  ```

- [ ] **Step 6: Add `CONFIRM_WIRING` to `_REQUIRED_KEYS` (total over TurnType).**
  Edit `protocol.py:200-218`, adding the entry after the `RECIPE_OFFER` line:

  ```python
      TurnType.RECIPE_OFFER: frozenset({"mode", "knobs", "prefilled", "recipe_context"}),
      # The wire confirm turn carries the topology + edge_contracts/semantic_contracts
      # overlay so the client can render wiring in one round-trip (spec §B2).
      TurnType.CONFIRM_WIRING: frozenset({"topology", "edge_contracts", "semantic_contracts"}),
  }
  ```

- [ ] **Step 7: Add `CONFIRM_WIRING` to `_NESTED_SHAPES` (total over TurnType).**
  Edit `protocol.py:243-253`, adding the empty-tuple entry after `RECIPE_OFFER`:

  ```python
      TurnType.RECIPE_OFFER: (("knobs", "mapping", frozenset({"fields"})),),
      # No nested-shape validation at the skeleton stage: topology / edge_contracts /
      # semantic_contracts are populated by the two-read merge (P3) and validated there.
      TurnType.CONFIRM_WIRING: (),
  }
  ```

- [ ] **Step 8: Run the test to confirm it passes.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_protocol.py -q
  ```
  Expected: all tests PASS (the import-time totality asserts in `prompts.py` will
  still fail at import — that is fixed in P1.2; run only this file here).

- [ ] **Step 9: Commit.**
  ```
  git add src/elspeth/web/composer/guided/protocol.py tests/unit/web/composer/guided/test_protocol.py
  git commit -m "feat(guided): append STEP_4_WIRE + CONFIRM_WIRING to protocol totals" --no-verify
  ```
  (`--no-verify`: `prompts.py`'s import-time totality assert is red until P1.2;
  reconciled at the slice boundary per project policy.)

### Task P1.2: Register step_4_wire.md in prompts + create the wiring-constraints skill

**Files:**
- Modify `src/elspeth/web/composer/guided/prompts.py:43-58` (`_STEP_FILE_NAMES` + `_STEP_PLAYBOOK_ORDER`)
- Create `src/elspeth/web/composer/guided/skills/step_4_wire.md`
- Modify `tests/unit/web/composer/guided/test_skill.py`

**Interfaces:**
- Produces: `_STEP_FILE_NAMES[GuidedStep.STEP_4_WIRE] == "step_4_wire.md"`.
- Produces: `_STEP_PLAYBOOK_ORDER` ends with `GuidedStep.STEP_4_WIRE`.
- Produces: `skills/step_4_wire.md` (wiring CONSTRAINTS only — no wire-stage UX copy, per H1; it is concatenated into the chain-solve prompt at transform-solve time).
- Consumes: `load_guided_skill()`, `load_step_chat_skill(step)` (unchanged signatures).

- [ ] **Step 1: Write a failing test that the skill maps cover STEP_4_WIRE and the file loads.**
  Append to `tests/unit/web/composer/guided/test_skill.py`:

  ```python
  # tests/unit/web/composer/guided/test_skill.py
  from elspeth.web.composer.guided.protocol import GuidedStep
  from elspeth.web.composer.guided.prompts import (
      _STEP_FILE_NAMES,
      _STEP_PLAYBOOK_ORDER,
      load_guided_skill,
      load_step_chat_skill,
  )


  class TestStep4WireSkill:
      def test_step_4_wire_registered_in_file_names(self) -> None:
          assert _STEP_FILE_NAMES[GuidedStep.STEP_4_WIRE] == "step_4_wire.md"

      def test_step_4_wire_appended_to_playbook_order(self) -> None:
          assert _STEP_PLAYBOOK_ORDER[-1] is GuidedStep.STEP_4_WIRE

      def test_step_4_wire_chat_skill_loads_and_mentions_routing(self) -> None:
          text = load_step_chat_skill(GuidedStep.STEP_4_WIRE)
          assert "wiring" in text.lower() or "routing" in text.lower()

      def test_full_guided_skill_includes_wire_block(self) -> None:
          text = load_guided_skill()
          # Wiring-constraint content is concatenated into the chain solver prompt.
          assert "on_success" in text
  ```

- [ ] **Step 2: Run the test to confirm it fails.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_skill.py -q
  ```
  Expected failure: import of `prompts.py` raises `AssertionError`
  (`_STEP_FILE_NAMES out of sync with GuidedStep: missing {GuidedStep.STEP_4_WIRE}`)
  at module-import time — the totality assert at `prompts.py:64-73` fires.

- [ ] **Step 3: Create the wiring-constraints skill file.**
  Write `src/elspeth/web/composer/guided/skills/step_4_wire.md`. Per H1 this is
  concatenated into the chain solver prompt at *transform*-solve time, so it must
  contain **only wiring constraints that bound node proposals** — no wire-stage UX
  copy ("you will see a visualization…"):

  ```markdown
  ## Step 4 — Wiring constraints

  Wiring is carried by **named connection labels**, never by edge objects.
  Every node and source declares where its output flows by label; the engine
  reconstructs the DAG from those labels. When you propose transforms, the
  wiring they imply MUST satisfy these constraints:

  - **Single linear spine.** The committed source emits `on_success: "chain_in"`.
    The first transform reads `input: "chain_in"`; the last transform emits
    `on_success: "main"`; intermediate transforms chain via `chain_{k}` labels.
    Sinks consume `"main"`. Do not introduce a label that no downstream node
    reads, and do not read a label no upstream node emits — an orphaned or
    dangling label is a wiring error.

  - **Producer/consumer field contract.** A downstream node may only require
    fields that some upstream node guarantees. If a transform consumes a field
    that no prior stage produces, the edge is unsatisfied and the pipeline is
    not runnable. Prefer ordering that makes every consumer's required fields
    available from its input label.

  - **Field minimization before a sink.** When a transform emits large or raw
    intermediate fields (e.g. fetched page content, content fingerprints) that
    the sink does not need, place a `field_mapper` with `select_only: true`
    immediately before the sink to drop them. The selected output field set is
    the sink's contract; raw intermediate fields must not leak to the output.

  - **No fan-out/fan-in unless required.** Routes (`routes`) and forks
    (`fork_to`) are only for genuine branching. A straight rate/transform/export
    pipeline is a single linear spine; do not add routing nodes the task does
    not call for.
  ```

- [ ] **Step 4: Register the step in both prompts maps.**
  Edit `prompts.py:43-58`. Add to `_STEP_FILE_NAMES`:

  ```python
  _STEP_FILE_NAMES: dict[GuidedStep, str] = {
      GuidedStep.STEP_1_SOURCE: "step_1_source.md",
      GuidedStep.STEP_2_SINK: "step_2_sink.md",
      GuidedStep.STEP_2_5_RECIPE_MATCH: "step_2_5_recipe_match.md",
      GuidedStep.STEP_3_TRANSFORMS: "step_3_transforms.md",
      GuidedStep.STEP_4_WIRE: "step_4_wire.md",
  }
  ```

  And append to `_STEP_PLAYBOOK_ORDER`:

  ```python
  _STEP_PLAYBOOK_ORDER: tuple[GuidedStep, ...] = (
      GuidedStep.STEP_1_SOURCE,
      GuidedStep.STEP_2_SINK,
      GuidedStep.STEP_2_5_RECIPE_MATCH,
      GuidedStep.STEP_3_TRANSFORMS,
      GuidedStep.STEP_4_WIRE,
  )
  ```

- [ ] **Step 5: Run the test to confirm it passes.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_skill.py tests/unit/web/composer/guided/test_protocol.py -q
  ```
  Expected: all PASS (the import-time totality asserts in `prompts.py` are now
  satisfied; the protocol totals still hold).

- [ ] **Step 6: Commit.**
  ```
  git add src/elspeth/web/composer/guided/prompts.py src/elspeth/web/composer/guided/skills/step_4_wire.md tests/unit/web/composer/guided/test_skill.py
  git commit -m "feat(guided): add step_4_wire skill block + register in prompts maps"
  ```

### Task P1.3: Append STEP_4_WIRE to both _ORDER tuples + add build_step_4_wire_turn emitter

**Files:**
- Modify `src/elspeth/web/composer/guided/emitters.py:428-432` (`_ORDER` in `_step_index`)
- Modify `src/elspeth/web/composer/guided/emitters.py` (add `build_step_4_wire_turn` + `__all__`/module docstring export note)
- Modify `src/elspeth/web/sessions/routes/_helpers.py:3674-3678` (`_ORDER` in `_guided_step_index`)
- Modify `tests/unit/web/composer/guided/test_emitters.py`

**Interfaces:**
- Produces: `build_step_4_wire_turn(*, validation: ValidationSummary) -> Turn` — a `CONFIRM_WIRING` Turn whose payload is the **skeleton** `{"topology": {}, "edge_contracts": [], "semantic_contracts": []}`. (The real two-read topology/edge_contracts blob is assembled in **P2.4**, which **replaces this signature** with the final one — `build_step_4_wire_turn(state, *, catalog=None, advisor_findings=None, signoff_outcome=None)`; P1.3 ships the keyword-only `validation` skeleton so the stage is reachable, P2.4 swaps it to the positional-`state` final form.)
- Produces: both `_ORDER` tuples end with `GuidedStep.STEP_4_WIRE` so `step_index` / `_guided_step_index` return `4` for it.
- Consumes: `CompositionState.validate() -> ValidationSummary` (state.py:2215); `ValidationSummary.is_valid` (state.py:383).

- [ ] **Step 1: Write a failing test for both step-index maps + the emitter.**
  Append to `tests/unit/web/composer/guided/test_emitters.py`:

  ```python
  # tests/unit/web/composer/guided/test_emitters.py
  from elspeth.web.composer.guided.emitters import _step_index, build_step_4_wire_turn
  from elspeth.web.composer.guided.protocol import GuidedStep, TurnType, validate_payload
  from elspeth.web.composer.state import CompositionState, PipelineMetadata
  from elspeth.web.sessions.routes._helpers import _guided_step_index


  def _empty_state() -> CompositionState:
      return CompositionState(
          source=None,
          nodes=(),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )


  class TestStep4WireEmitter:
      def test_emitter_step_index_is_four(self) -> None:
          assert _step_index(GuidedStep.STEP_4_WIRE) == 4

      def test_helpers_step_index_is_four(self) -> None:
          assert _guided_step_index(GuidedStep.STEP_4_WIRE) == 4

      def test_build_step_4_wire_turn_shape(self) -> None:
          turn = build_step_4_wire_turn(validation=_empty_state().validate())
          assert turn["type"] == TurnType.CONFIRM_WIRING.value
          assert turn["step_index"] == 4
          # Skeleton payload validates against the protocol total.
          assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None
          assert turn["payload"]["topology"] == {}
          assert turn["payload"]["edge_contracts"] == []
          assert turn["payload"]["semantic_contracts"] == []
  ```

- [ ] **Step 2: Run the test to confirm it fails.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_emitters.py -q
  ```
  Expected failure: `ImportError: cannot import name 'build_step_4_wire_turn'` and,
  once that import is stubbed, `ValueError: ... is not in tuple` from
  `_ORDER.index(GuidedStep.STEP_4_WIRE)` in both step-index helpers.

- [ ] **Step 3: Append STEP_4_WIRE to the emitter's `_ORDER` tuple.**
  Edit `emitters.py:428-432`:

  ```python
      _ORDER: tuple[GuidedStep, ...] = (
          GuidedStep.STEP_1_SOURCE,
          GuidedStep.STEP_2_SINK,
          GuidedStep.STEP_2_5_RECIPE_MATCH,
          GuidedStep.STEP_3_TRANSFORMS,
          GuidedStep.STEP_4_WIRE,
      )
      return _ORDER.index(step)
  ```

- [ ] **Step 4: Append STEP_4_WIRE to the route-helper's duplicate `_ORDER` tuple.**
  Edit `_helpers.py:3674-3678`:

  ```python
      _ORDER: tuple[GuidedStep, ...] = (
          GuidedStep.STEP_1_SOURCE,
          GuidedStep.STEP_2_SINK,
          GuidedStep.STEP_2_5_RECIPE_MATCH,
          GuidedStep.STEP_3_TRANSFORMS,
          GuidedStep.STEP_4_WIRE,
      )
      return _ORDER.index(step)
  ```

- [ ] **Step 5: Add the `build_step_4_wire_turn` emitter.**
  Insert after `build_step_3_schema_form_turn` (after `emitters.py:370`). First add
  the `ValidationSummary` import to the `TYPE_CHECKING` block (after the
  `CompositionState` import at `emitters.py:48`):

  ```python
      from elspeth.web.composer.state import CompositionState, ValidationSummary
  ```

  Then the emitter:

  ```python
  def build_step_4_wire_turn(
      *,
      validation: ValidationSummary,
  ) -> Turn:
      """Build a ``confirm_wiring`` Turn for the wire stage (skeleton).

      P1 ships the wire stage as a *reachable* terminal-gate stage. The payload
      carries empty topology / edge_contracts / semantic_contracts blobs; the
      real two-read merge (get_pipeline_state topology + preview_pipeline
      edge_contracts overlay, spec §B2) is assembled in P3 and replaces this
      signature. ``validation`` is accepted now so the emit site can decide
      whether the confirm gate is satisfiable without a second pass; the
      skeleton payload does not embed it.

      Trust tier: L3 web layer; the Turn dict is not persisted (only its hash).
      """
      payload: dict[str, Any] = {
          "topology": {},
          "edge_contracts": [],
          "semantic_contracts": [],
      }
      return Turn(
          type=TurnType.CONFIRM_WIRING.value,
          step_index=_step_index(GuidedStep.STEP_4_WIRE),
          payload=payload,
      )
  ```

  Add `build_step_4_wire_turn` to the module docstring's "Exported:" list
  (`emitters.py:8-17`):

  ```python
      build_step_4_wire_turn — confirm_wiring turn for the wire stage.
  ```

- [ ] **Step 6: Run the test to confirm it passes.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_emitters.py -q
  ```
  Expected: all PASS.

- [ ] **Step 7: Run the broader guided emitter/route smoke to confirm no `_ORDER` regression.**
  ```
  uv run pytest tests/unit/web/composer/guided/ -q
  ```
  Expected: PASS (no `ValueError: ... not in tuple` from either step-index map).

- [ ] **Step 8: Commit.**
  ```
  git add src/elspeth/web/composer/guided/emitters.py src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/composer/guided/test_emitters.py
  git commit -m "feat(guided): wire STEP_4_WIRE into both _ORDER tuples + build_step_4_wire_turn"
  ```

### Task P1.4: Add step_advance STEP_4_WIRE branch + _advance_step_4 self-loop; route _advance_step_3 accept to STEP_4

**Files:**
- Modify `src/elspeth/web/composer/guided/state_machine.py:546-554` (step_advance dispatch)
- Modify `src/elspeth/web/composer/guided/state_machine.py` (add `_advance_step_4` after `_advance_step_3` at ~:778)
- Modify `tests/unit/web/composer/guided/test_state_machine.py`

**Interfaces:**
- Produces: `step_advance` dispatches `GuidedStep.STEP_4_WIRE -> _advance_step_4`.
- Produces: `_advance_step_4(session, response, turn_type) -> _StepAdvanceResult` — a **self-loop**: on `CONFIRM_WIRING` it returns the session unchanged (terminal-stamping is the dispatcher/handler's job, P2/P6); any other turn type raises `InvariantError`. The self-loop is what makes the stage re-enterable across advisor sign-off rounds (D13).
- Consumes: `_StepAdvanceResult`, `GuidedSession`, `TurnResponse`, `TurnType`, `InvariantError` (all existing).
- Note: `_advance_step_3` is **not** changed to mutate `step` — it is pure and already passes through on accept (state_machine.py:765-772). The accept-time advance to `STEP_4_WIRE` is performed by the *handlers* setting `session.step = STEP_4_WIRE` (P2.T1), not by `step_advance`. This task only adds the new branch + handler self-loop so the dispatcher can route a wire-stage response.

- [ ] **Step 1: Write a failing test for the dispatch branch + self-loop.**
  Append to `tests/unit/web/composer/guided/test_state_machine.py`:

  ```python
  # tests/unit/web/composer/guided/test_state_machine.py
  from dataclasses import replace

  import pytest

  from elspeth.web.composer.guided.errors import InvariantError
  from elspeth.web.composer.guided.protocol import (
      ControlSignal,
      GuidedStep,
      TurnResponse,
      TurnType,
  )
  from elspeth.web.composer.guided.state_machine import GuidedSession, step_advance


  def _wire_response(control: ControlSignal | None = None) -> TurnResponse:
      return TurnResponse(
          chosen=None,
          edited_values=None,
          custom_inputs=None,
          accepted_step_index=None,
          edit_step_index=None,
          control_signal=control,
      )


  class TestAdvanceStep4Wire:
      def test_confirm_wiring_is_a_self_loop(self) -> None:
          session = replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
          new_session, turn, terminal, directives = step_advance(
              session,
              _wire_response(),
              current_turn_type=TurnType.CONFIRM_WIRING,
          )
          # step_advance does not stamp terminal at the wire stage — the
          # dispatcher/handler does (P2/P6). The session pointer stays at STEP_4_WIRE.
          assert new_session.step is GuidedStep.STEP_4_WIRE
          assert terminal is None
          assert turn is None
          assert directives == []

      def test_wire_stage_rejects_illegal_turn_type(self) -> None:
          session = replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
          with pytest.raises(InvariantError, match="STEP_4_WIRE"):
              step_advance(
                  session,
                  _wire_response(),
                  current_turn_type=TurnType.PROPOSE_CHAIN,
              )

      def test_wire_stage_exit_to_freeform_still_terminates(self) -> None:
          session = replace(GuidedSession.initial(), step=GuidedStep.STEP_4_WIRE)
          _new, _turn, terminal, _directives = step_advance(
              session,
              _wire_response(control=ControlSignal.EXIT_TO_FREEFORM),
              current_turn_type=TurnType.CONFIRM_WIRING,
          )
          assert terminal is not None
  ```

- [ ] **Step 2: Run the test to confirm it fails.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_state_machine.py -k AdvanceStep4Wire -q
  ```
  Expected failure: `InvariantError: unhandled step: GuidedStep.STEP_4_WIRE` from
  the fall-through `raise` at `state_machine.py:554` (no STEP_4_WIRE branch yet).

- [ ] **Step 3: Add the STEP_4_WIRE dispatch branch in step_advance.**
  Edit `state_machine.py:552-554`, inserting the branch before the fall-through
  `raise`:

  ```python
      if session.step is GuidedStep.STEP_3_TRANSFORMS:
          return _advance_step_3(session, response, current_turn_type)
      if session.step is GuidedStep.STEP_4_WIRE:
          return _advance_step_4(session, response, current_turn_type)
      raise InvariantError(f"unhandled step: {session.step}")
  ```

- [ ] **Step 4: Add the `_advance_step_4` handler (self-loop).**
  Insert after `_advance_step_3` (after `state_machine.py:777`):

  ```python
  def _advance_step_4(
      session: GuidedSession,
      response: TurnResponse,
      turn_type: TurnType,
  ) -> _StepAdvanceResult:
      """Handle a Step 4 (wire) response. Self-loop — does not advance or terminate.

      The wire stage is a *terminal gate*: the dispatcher / wire handler decides
      whether to stamp ``terminal=COMPLETED`` (after ``validate().is_valid`` in P1,
      plus the profile-gated advisor sign-off in P6). ``step_advance`` is pure and
      cannot run validation or call the advisor, so on a ``CONFIRM_WIRING`` turn it
      returns the session unchanged — keeping the stage re-enterable across advisor
      sign-off rounds (spec §B3/D13). exit_to_freeform is handled by ``step_advance``
      before this branch is reached.

      Any non-``CONFIRM_WIRING`` turn type at STEP_4_WIRE means the emitter stamped an
      illegal type on the history record — a server bug, so ``InvariantError`` (500).
      """
      if turn_type is TurnType.CONFIRM_WIRING:
          return (session, None, None, [])
      raise InvariantError(
          f"_advance_step_4: unexpected turn_type {turn_type!r} at STEP_4_WIRE — "
          "the wire stage only emits CONFIRM_WIRING turns; any other type in the "
          "history record indicates a server-side emitter bug."
      )
  ```

- [ ] **Step 5: Run the test to confirm it passes.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_state_machine.py -k AdvanceStep4Wire -q
  ```
  Expected: all PASS.

- [ ] **Step 6: Run the full state_machine + protocol + skill suite to confirm no regression.**
  ```
  uv run pytest tests/unit/web/composer/guided/test_state_machine.py tests/unit/web/composer/guided/test_protocol.py tests/unit/web/composer/guided/test_skill.py -q
  ```
  Expected: all PASS.

- [ ] **Step 7: Commit.**
  ```
  git add src/elspeth/web/composer/guided/state_machine.py tests/unit/web/composer/guided/test_state_machine.py
  git commit -m "feat(guided): add step_advance STEP_4_WIRE branch + _advance_step_4 self-loop"
  ```

### Task P1.5: Mirror STEP_4_WIRE + CONFIRM_WIRING into the frontend guided.ts unions

**Files:**
- Modify `src/elspeth/web/frontend/src/types/guided.ts:18-39` (`TurnType` + `GuidedStep` unions)
- Modify `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx` (the `never`-exhaustive switch on `turn.type`)
- Modify `src/elspeth/web/frontend/src/components/chat/guided/GuidedHistory.tsx` (`STEP_LABELS: Record<GuidedStep,…>` + `TURN_TYPE_LABELS: Record<TurnType,…>`)
- Modify `src/elspeth/web/frontend/src/components/chat/guided/GuidedChatHistory.tsx` (`STEP_LABELS: Record<GuidedStep,…>`)
- Modify `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (`GUIDED_CHAT_PLACEHOLDERS: Record<GuidedStep,…>` + `GUIDED_STEP_PURPOSES: Record<GuidedStep,…>`)

**Interfaces:**
- Produces: TS `TurnType` union gains `"confirm_wiring"`; TS `GuidedStep` union gains `"step_4_wire"`.
- Consumes: nothing new — these are hand-written mirrors of the Python StrEnums (source-of-truth comment at `guided.ts:5-7`).
- Note: widening the two unions WITHOUT updating their exhaustive consumers in the SAME slice BREAKS `npm run typecheck` — every `Record<TurnType,…>` / `Record<GuidedStep,…>` total-key map raises a missing-key error (TS2741) and the `never`-exhaustiveness assertion in `GuidedTurn.tsx` fails to narrow. There are **six** such consumers (GuidedTurn `never` switch; GuidedHistory `STEP_LABELS` + `TURN_TYPE_LABELS`; GuidedChatHistory `STEP_LABELS`; ChatPanel `GUIDED_CHAT_PLACEHOLDERS` + `GUIDED_STEP_PURPOSES`), so this task widens the unions AND extends all six together. The `WireStageData` TS type is **P3.T2** and the `interpretation_review` dead-case removal is **P4.T2**; the real `confirm_wiring`→`WireStageTurn` render replaces the placeholder added here once WireStageTurn exists (**P2.7**).

- [ ] **Step 1: Add the new members to the TS unions.**
  Edit `guided.ts:18-28` (TurnType union), appending the new member after
  `interpretation_review` (keep the existing frontend-only `interpretation_review`
  case — its removal is P4.T2, out of P1 scope):

  ```typescript
  export type TurnType =
    | "inspect_and_confirm"
    | "single_select"
    | "multi_select_with_custom"
    | "schema_form"
    | "propose_chain"
    | "recipe_offer"
    // Phase 5b: guided-mode interpretation-review widget.  Dispatched from
    // GuidedTurn.tsx (the freeform variant uses InterpretationReviewInlineMessage
    // — different file, different component, no shared widget).
    | "interpretation_review"
    // Wire stage (staged-recut P1): the confirm_wiring turn carries the topology +
    // edge_contracts/semantic_contracts overlay; see WireStageData (added P3).
    | "confirm_wiring";
  ```

  Edit `guided.ts:35-39` (GuidedStep union), appending `step_4_wire` last:

  ```typescript
  export type GuidedStep =
    | "step_1_source"
    | "step_2_sink"
    | "step_2_5_recipe_match"
    | "step_3_transforms"
    | "step_4_wire";
  ```

- [ ] **Step 2: Extend the six exhaustive consumers in the SAME slice (required for typecheck).**
  a) `GuidedTurn.tsx` (the `never`-exhaustive switch on `turn.type`, ~line 153) — add a `confirm_wiring` case BEFORE `default:` so `const _exhaustive: never = turn.type` still narrows. The real `<WireStageTurn>` render is wired in P2.7; here it is a placeholder (WireStageTurn does not exist yet in P1):
  ```tsx
      case "confirm_wiring":
        // Placeholder — real WireStageTurn render is wired in P2.7. This case only
        // keeps the union total so the `never` assertion in `default:` compiles.
        return null;
  ```
  b) `GuidedHistory.tsx` — add the new key to BOTH total-key maps:
  ```tsx
      // in STEP_LABELS: Record<GuidedStep, string>
      step_4_wire: "Wire",
      // in TURN_TYPE_LABELS: Record<TurnType, string>
      confirm_wiring: "Confirm wiring",
  ```
  c) `GuidedChatHistory.tsx` — add to its `STEP_LABELS: Record<GuidedStep, string>`:
  ```tsx
      step_4_wire: "Wire",
  ```
  d) `ChatPanel.tsx` — add the new step to BOTH `Record<GuidedStep, string>` maps:
  ```tsx
      // in GUIDED_CHAT_PLACEHOLDERS
      step_4_wire: "Confirm how the steps connect, then continue.",
      // in GUIDED_STEP_PURPOSES
      step_4_wire: "Review and confirm the wiring between your pipeline steps.",
  ```

- [ ] **Step 3: Typecheck the frontend.**
  ```
  npm --prefix src/elspeth/web/frontend run typecheck
  ```
  Expected: PASS — but ONLY because Step 2 extended every exhaustive consumer.
  Widening the unions alone does NOT typecheck: each `Record<TurnType,…>` /
  `Record<GuidedStep,…>` total-key map raises TS2741 (missing key) and the
  `never`-exhaustiveness assertion in `GuidedTurn.tsx` fails to narrow.

- [ ] **Step 4: Run the SlotType / guided.ts mirror-drift gate.**
  ```
  uv run python scripts/cicd/check_slot_type_cross_language.py
  ```
  Expected: PASS (this gate checks `SlotType` Literal vs the `RecipeSlotInput`
  interface, which P1 does not touch — confirm the guided.ts edit did not break
  the parse).

- [ ] **Step 4: Commit.**
  ```
  git add src/elspeth/web/frontend/src/types/guided.ts
  git commit -m "feat(frontend): mirror STEP_4_WIRE + confirm_wiring into guided.ts unions"
  ```

### Task P1.6: Move terminal-stamping out of both completion seams into a STEP_4_WIRE handler

**Files:**
- Modify `src/elspeth/web/composer/guided/steps.py:218-274` (`handle_step_2_5_recipe_apply`)
- Modify `src/elspeth/web/composer/guided/steps.py:277-406` (`handle_step_3_chain_accept`)
- Create the wire handler in `src/elspeth/web/composer/guided/steps.py` (`handle_step_4_wire_confirm`)
- Modify `tests/integration/web/composer/guided/test_step_handlers.py:251-292,329-400`

**Interfaces:**
- Produces (changed): `handle_step_2_5_recipe_apply` success path now sets `session.step = GuidedStep.STEP_4_WIRE`, `session.terminal = None` (no YAML render, no COMPLETED stamp).
- Produces (changed): `handle_step_3_chain_accept` success path now sets `session.step = GuidedStep.STEP_4_WIRE`, `session.terminal = None`, `session.step_3_proposal = proposal`.
- Produces (new): `handle_step_4_wire_confirm(*, state: CompositionState, session: GuidedSession) -> StepHandlerResult` — runs `state.validate()`; on `is_valid` stamps `TerminalState(COMPLETED, reason=None, pipeline_yaml=generate_yaml(state))`; on invalid leaves `terminal=None` and returns a non-success `StepHandlerResult` carrying the `ValidationSummary`. (P5.6 inserts the profile-gated advisor sign-off *before* the COMPLETED stamp; P1 gates on `validate().is_valid` only.)
- Consumes: `CompositionState.validate()` (state.py:2215), `generate_yaml` (yaml_generator.py:198), `TerminalState`, `TerminalKind`, `StepHandlerResult`, `ToolResult`.
- DEPENDS-ON (downstream, not this phase): the dispatcher rewire in **P2.9** must emit `build_step_4_wire_turn` after each accept commit and dispatch `CONFIRM_WIRING` responses to `handle_step_4_wire_confirm`. This task only moves the stamp out + adds the handler; the dispatcher rewire is **P2.9**.

- [ ] **Step 1: Write the terminal-stamp invariant test (failing).**
  Add a new test class to `tests/integration/web/composer/guided/test_step_handlers.py`.
  It asserts that BOTH accept seams leave `terminal is None` AND `step == STEP_4_WIRE`,
  and that the new wire handler stamps COMPLETED on a valid state:

  ```python
  # tests/integration/web/composer/guided/test_step_handlers.py
  class TestTerminalStampInvariant:
      """spec §4.2 / rev-4 invariant: neither completion seam may stamp COMPLETED.

      Both handle_step_2_5_recipe_apply and handle_step_3_chain_accept must leave
      session.terminal is None AND session.step == STEP_4_WIRE on success; the
      COMPLETED stamp moves into handle_step_4_wire_confirm. Missing either move
      silently skips the wire stage, the B1 surfacing pass, and the advisor gate.
      """

      def test_chain_accept_redirects_to_wire_not_completed(self) -> None:
          from elspeth.web.composer.guided.protocol import GuidedStep
          from elspeth.web.composer.guided.state_machine import ChainProposal
          from elspeth.web.composer.guided.steps import (
              handle_step_1_source,
              handle_step_2_sink,
              handle_step_3_chain_accept,
          )

          state = _empty_state()
          session = GuidedSession.initial()
          catalog = create_catalog_service()

          step_1 = handle_step_1_source(
              state=state,
              session=session,
              catalog=catalog,
              resolved=SourceResolved(
                  plugin="csv",
                  options={"path": "x.csv", "schema": {"mode": "observed"}},
                  observed_columns=("price",),
                  sample_rows=({"price": "1.99"},),
              ),
          )
          step_2 = handle_step_2_sink(
              state=step_1.state,
              session=step_1.session,
              catalog=catalog,
              resolved=SinkResolved(
                  outputs=(
                      SinkOutputResolved(
                          plugin="json",
                          options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                          required_fields=("price",),
                          schema_mode="observed",
                      ),
                  ),
              ),
          )
          proposal = ChainProposal(
              steps=(
                  {
                      "plugin": "passthrough",
                      "options": {"schema": {"mode": "observed"}},
                      "rationale": "echo rows",
                  },
              ),
              why="single-step chain",
          )
          result = handle_step_3_chain_accept(
              state=step_2.state,
              session=step_2.session,
              catalog=catalog,
              proposal=proposal,
          )
          assert result.tool_result.success is True
          assert result.session.terminal is None
          assert result.session.step is GuidedStep.STEP_4_WIRE
          assert result.session.step_3_proposal is proposal

          # The committed pipeline is valid → the wire handler stamps COMPLETED.
          from elspeth.web.composer.guided.state_machine import TerminalKind
          from elspeth.web.composer.guided.steps import handle_step_4_wire_confirm

          wire = handle_step_4_wire_confirm(state=result.state, session=result.session)
          assert wire.tool_result.success is True
          assert wire.session.terminal is not None
          assert wire.session.terminal.kind is TerminalKind.COMPLETED
          assert wire.session.terminal.pipeline_yaml is not None
          assert len(wire.session.terminal.pipeline_yaml) > 0

      def test_recipe_apply_redirects_to_wire_not_completed(self, _seeded) -> None:
          from elspeth.web.composer.guided.protocol import GuidedStep
          from elspeth.web.composer.guided.recipe_match import RecipeMatch
          from elspeth.web.composer.guided.steps import handle_step_2_5_recipe_apply

          engine, session_id, blob_id = _seeded
          state = _empty_state()
          catalog = self._real_catalog()
          match = RecipeMatch(
              recipe_name="classify-rows-llm-jsonl",
              slots={
                  "source_blob_id": blob_id,
                  "classifier_template": "Classify the following text: {{ row['text'] }}",
                  "model": "anthropic/claude-3.5-sonnet",
                  "api_key_secret": "OPENROUTER_API_KEY",
                  "required_input_fields": ["text"],
              },
              unsatisfied_slots={},
          )
          result = handle_step_2_5_recipe_apply(
              state=state,
              session=GuidedSession.initial(),
              match=match,
              catalog=catalog,
              session_engine=engine,
              session_id=session_id,
          )
          assert result.tool_result.success is True
          assert result.session.terminal is None
          assert result.session.step is GuidedStep.STEP_4_WIRE

      _real_catalog = TestStep2_5Handler._real_catalog
  ```

  Note: the `_seeded` fixture and `_real_catalog` live on the existing
  `TestStep2_5Handler` class (the apply-recipe test at line 251 uses them). Reference
  them by binding `_real_catalog = TestStep2_5Handler._real_catalog` (shown above) and
  ensure `_seeded` is a module/conftest fixture — if it is a class-scoped fixture on
  `TestStep2_5Handler`, lift it to a module-level fixture in the same edit so both
  classes can request it. Confirm the fixture scope when implementing.

- [ ] **Step 2: Update the two existing handler tests that assert COMPLETED.**
  In `test_step_handlers.py`, the existing
  `test_apply_recipe_terminates_completed_with_yaml` (line 251) and
  `test_chain_accepted_commits_and_completes` (line 329) assert the *old* behaviour
  (handler stamps COMPLETED). Retarget them to the new contract — the handler
  redirects to the wire stage, the wire handler stamps COMPLETED.

  For `test_apply_recipe_terminates_completed_with_yaml`, replace the terminal
  assertions at lines 288-292 with:

  ```python
          from elspeth.web.composer.guided.protocol import GuidedStep

          assert result.session.terminal is None
          assert result.session.step is GuidedStep.STEP_4_WIRE
  ```

  Rename it to `test_apply_recipe_redirects_to_wire_with_committed_state`.

  For `test_chain_accepted_commits_and_completes`, replace the terminal assertions
  at lines 395-399 with:

  ```python
          from elspeth.web.composer.guided.protocol import GuidedStep

          assert result.session.terminal is None
          assert result.session.step is GuidedStep.STEP_4_WIRE
  ```

  Keep the `assert result.session.step_3_proposal is proposal` at line 400. Rename
  it to `test_chain_accepted_commits_and_redirects_to_wire`.

- [ ] **Step 3: Run the tests to confirm they fail.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_step_handlers.py -k "TerminalStampInvariant or redirects_to_wire" -q
  ```
  Expected failure: `ImportError: cannot import name 'handle_step_4_wire_confirm'`,
  and the retargeted existing tests fail on
  `assert result.session.terminal is None` (the handlers still stamp COMPLETED).

- [ ] **Step 4: Move the stamp out of `handle_step_2_5_recipe_apply`.**
  Edit `steps.py:262-274` (the success path). Replace the YAML-render + terminal
  stamp with a step-redirect that leaves `terminal=None`:

  ```python
      new_session = dataclasses.replace(
          session,
          step=GuidedStep.STEP_4_WIRE,
      )

      return StepHandlerResult(
          state=tool_result.updated_state,
          session=new_session,
          tool_result=tool_result,
      )
  ```

  Remove the now-unused local `yaml_text`/`terminal` from this branch. Update the
  docstring (`steps.py:228-237`) first line to: `"""Apply the matched recipe and
  redirect the session to the wire stage."""`. Add `GuidedStep` to the
  `state_machine` import block at `steps.py:24-31`:

  ```python
  from elspeth.web.composer.guided.state_machine import (
      ChainProposal,
      GuidedSession,
      GuidedStep,
      SinkResolved,
      SourceResolved,
      TerminalKind,
      TerminalState,
  )
  ```

- [ ] **Step 5: Move the stamp out of `handle_step_3_chain_accept`.**
  Edit `steps.py:390-400` (the success path). Replace with a redirect that records
  the proposal but leaves `terminal=None`:

  ```python
      new_session = dataclasses.replace(
          session,
          step=GuidedStep.STEP_4_WIRE,
          step_3_proposal=proposal,
      )

      return StepHandlerResult(
          state=tool_result.updated_state,
          session=new_session,
          tool_result=tool_result,
      )
  ```

  Remove the now-unused local `yaml_text`/`terminal` from this branch. Update the
  docstring (`steps.py:287` first line + the "On _execute_set_pipeline success"
  paragraph at `:303-306`) to: success now redirects to `STEP_4_WIRE` and records
  the proposal; the wire handler stamps COMPLETED.

- [ ] **Step 6: Add `handle_step_4_wire_confirm` (validate-only gate).**
  Insert after `handle_step_3_chain_accept` (after `steps.py:406`):

  ```python
  def handle_step_4_wire_confirm(
      *,
      state: CompositionState,
      session: GuidedSession,
  ) -> StepHandlerResult:
      """Confirm wiring: gate the COMPLETED stamp on ``validate().is_valid``.

      This is where terminal-stamping now lives (moved out of the step-2.5
      recipe-apply and step-3 chain-accept seams, spec §4.2). "Confirm wiring"
      does not commit routing — the prior handlers already wired the pipeline via
      named connection labels. It re-runs ``state.validate()`` and, only when the
      pipeline is valid (zero blocking field-contract errors), stamps
      ``TerminalState(COMPLETED, reason=None, pipeline_yaml=...)``.

      On an invalid pipeline it returns a non-success ``StepHandlerResult`` whose
      ``tool_result`` carries the ``ValidationSummary``; the dispatcher re-emits the
      wire turn so the user can reconcile (insert a field_mapper / relax a schema)
      and re-confirm. ``terminal`` stays ``None`` on the invalid path.

      P1 gates on ``validate().is_valid`` only. The profile-gated advisor END
      sign-off (spec §B3/D13) is inserted *before* the COMPLETED stamp in P5.6; it
      reads/increments the persisted pass counter and re-emits the wire turn on a
      non-CLEAN verdict.
      """
      validation = state.validate()
      tool_result = ToolResult(
          success=validation.is_valid,
          updated_state=state,
          validation=validation,
          affected_nodes=(),  # confirm wiring mutates nothing — no nodes changed
          data=None,
      )
      if not validation.is_valid:
          # Leave the session at STEP_4_WIRE, terminal unset — the dispatcher
          # re-emits the wire turn carrying the validation errors.
          return StepHandlerResult(state=state, session=session, tool_result=tool_result)

      yaml_text = generate_yaml(state)
      terminal = TerminalState(
          kind=TerminalKind.COMPLETED,
          reason=None,
          pipeline_yaml=yaml_text,
      )
      new_session = dataclasses.replace(session, terminal=terminal)
      return StepHandlerResult(state=state, session=new_session, tool_result=tool_result)
  ```

  `ToolResult` is defined at `composer/tools/_common.py:536-574`; its
  required fields are `success: bool`, `updated_state: CompositionState`,
  `validation: ValidationSummary`, `affected_nodes: tuple[str, ...]` (no default —
  pass `()`), with `data: Any = None` and the rest defaulted. `__post_init__`
  runs `freeze_fields(self, "affected_nodes", ...)`, so `affected_nodes=()` is
  mandatory. `ToolResult` is already imported into `steps.py` (steps.py:33-41).

- [ ] **Step 7: Run the tests to confirm they pass.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_step_handlers.py -k "TerminalStampInvariant or redirects_to_wire or redirects_to_wire_with" -q
  ```
  Expected: all PASS — both seams leave `terminal is None` + `step == STEP_4_WIRE`;
  the wire handler stamps COMPLETED on the valid pipeline.

- [ ] **Step 8: Run the full step-handler + auto-drop suite to catch fallout.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_step_handlers.py tests/integration/web/composer/guided/test_auto_drop.py -q
  ```
  Expected: PASS. (The auto-drop path returns `repair_result.session` on success at
  `_helpers.py:3520`; that session now carries `step=STEP_4_WIRE, terminal=None`
  rather than COMPLETED — the dispatcher-side fix is P2.9, so a *route-level*
  test of the full accept→complete flow may go red here. If `test_auto_drop.py`
  asserts a COMPLETED terminal at the route layer, mark that assertion with a
  `# updated in P2.9 (wire dispatch)` note and assert `step == STEP_4_WIRE` for now,
  or xfail it with reason "wire dispatch lands in P2.9" — do NOT re-stamp COMPLETED
  in the handler to make it green. P2.9 Step 9 un-xfails these once the wire
  dispatch lands.)

- [ ] **Step 9: Export the new handler.**
  Add `handle_step_4_wire_confirm` to any `__all__` / re-export of `steps.py`
  handlers. Check `_helpers.py:97-100` (which imports `handle_step_2_5_recipe_apply`
  and `handle_step_3_chain_accept`) and the `_helpers.py` `__all__` (the
  `handle_step_*` entries near `:3919-3921`) — add the import + `__all__` entry so
  P2.9 can dispatch to it:

  ```python
  # _helpers.py import block (near :97)
      handle_step_4_wire_confirm,
  # _helpers.py __all__ (near :3921)
      "handle_step_4_wire_confirm",
  ```

  (P1 imports it so the symbol is reachable; P2.9 calls it from the dispatcher.)

- [ ] **Step 10: Commit.**
  ```
  git add src/elspeth/web/composer/guided/steps.py src/elspeth/web/sessions/routes/_helpers.py tests/integration/web/composer/guided/test_step_handlers.py
  git commit -m "feat(guided): move terminal-stamp into handle_step_4_wire_confirm (validate-gate)"
  ```

### Task P1.7: Phase reconciliation — gate sweep + commit

**Files:** none (verification only)

**Interfaces:** none.

- [ ] **Step 1: Run ruff lint + format check over the touched trees.**
  ```
  uv run ruff check src/elspeth/web/composer/guided/ src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/composer/guided/ tests/integration/web/composer/guided/
  uv run ruff format --check src/elspeth/web/composer/guided/ src/elspeth/web/sessions/routes/_helpers.py
  ```
  Expected: `All checks passed!` and no format diff. (If `ruff format` rewrites the
  new emitter/handler, apply `uv run ruff format <paths>` and re-stage.)

- [ ] **Step 2: Run mypy over the touched modules.**
  ```
  uv run mypy src/elspeth/web/composer/guided/ src/elspeth/web/sessions/routes/_helpers.py
  ```
  Expected: `Success: no issues found`. (Watch the `ValidationSummary` import in
  `emitters.py` and the new `ToolResult` construction in `steps.py` — if mypy flags a
  missing `ToolResult` field, mirror the sibling handlers' construction.)

- [ ] **Step 3: Run the full guided unit + integration suite.**
  ```
  uv run pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -q
  ```
  Expected: PASS, except any route-level COMPLETED assertion deferred to P2.9 (Step
  8 of P1.6) — those must be xfail-with-reason, never re-stamped in the handler.

- [ ] **Step 4: Frontend typecheck + the mirror gate (final).**
  ```
  npm --prefix src/elspeth/web/frontend run typecheck
  uv run python scripts/cicd/check_slot_type_cross_language.py
  ```
  Expected: both PASS.

- [ ] **Step 5: Final phase commit (if Steps 1-2 produced format/import fixups).**
  ```
  git add -A
  git commit -m "chore(guided): P1 wire-stage skeleton gate reconciliation"
  ```
  (Skip if nothing changed after the per-task commits.)

---

## Phase P2 — Wire stage data model (B2)

> **Scope.** The `STEP_4_WIRE` turn payload returns **two reads** — *neither alone
> suffices*: (1) `_serialize_full_pipeline_state` (`composer/tools/sessions.py:1071`,
> via `_serialize_source/_serialize_node/_serialize_output` in `_common.py:968-1010`)
> for the **connection-label topology** (`input/on_success/on_error/routes/fork_to` +
> source `on_success`), and (2) the `edge_contracts` (+ `semantic_contracts`) overlay
> from `state.validate()` (built into `_authoring_validation_payload`,
> `sessions.py:1153-1162`; surfaced by `_execute_preview_pipeline`, `generation.py:1651`,
> summary `:1685-1692`). `EdgeContract.to_dict()` (`state.py:359-368`) emits keys
> **`from`/`to`** — NOT `from_id`/`to_id`. The render reconstructs edges from connection
> labels (NEVER `state.edges` — guided passes `edges=[]`) and overlays `edge_contracts`
> keyed by `(from, to)`. B6: after any wire-stage reconciliation the confirm gate
> re-evaluates `validate().is_valid` AND re-runs the P3 surfacing pass.
>
> **Dependencies on other phases (must land first):** P1 owns
> `GuidedStep.STEP_4_WIRE`, `TurnType.CONFIRM_WIRING`, the `emitters.py` `_ORDER`
> tuple entry for `STEP_4_WIRE` (consumed by `_step_index`), and the `guided.ts`
> `GuidedStep`/`TurnType` union strings. This phase's emitter calls
> `_step_index(GuidedStep.STEP_4_WIRE)` and stamps `TurnType.CONFIRM_WIRING.value`;
> both must exist. The P3 surfacing entry point `_surface_pending_interpretation_reviews`
> is consumed by the B6 re-surface step (P2.7) — until P3 lands, the re-surface call
> is threaded as a passed-in callback so this phase stays independently testable.

---

### Task P2.1: `WireTopology` / `WireStageData` payload TypedDicts in `protocol.py`

**Files:**
- Modify `src/elspeth/web/composer/guided/protocol.py` (add TypedDicts after
  `ProposeChainPayload`, `:62-65`)

**Interfaces:**
- Consumes: `TurnType.CONFIRM_WIRING` (P1, `protocol.py:16-25`), `Turn` TypedDict
  (`protocol.py:88-93`)
- Produces:
  ```python
  class _WireSourceTopo(TypedDict):
      plugin: str
      on_success: str | None
  class _WireNodeTopo(TypedDict):
      id: str
      node_type: str
      plugin: str | None
      input: str | None
      on_success: str | None
      on_error: str | None
      routes: Mapping[str, str] | None
      fork_to: Sequence[str] | None
  class _WireOutputTopo(TypedDict):
      sink_name: str
      plugin: str
  class WireTopology(TypedDict):
      sources: Mapping[str, _WireSourceTopo]
      nodes: Sequence[_WireNodeTopo]
      outputs: Sequence[_WireOutputTopo]
  class WireStageData(TypedDict):
      topology: WireTopology
      edge_contracts: Sequence[Mapping[str, Any]]
      semantic_contracts: Sequence[Mapping[str, Any]]
      warnings: Sequence[Mapping[str, Any]]
      advisor_findings: NotRequired[str]  # set only on the P5 sign-off revise re-emit
      signoff_outcome: NotRequired[str]   # SignoffOutcome.value on the revise re-emit
  ```
  (Import `NotRequired` from `typing` in `protocol.py` if not already present.)

- [ ] **Step 1: Write failing test for the WireStageData TypedDict keys.**
  Create `tests/unit/web/composer/guided/test_wire_payload.py`:
  ```python
  """Tests for the STEP_4_WIRE turn payload data model (P2/B2)."""

  from __future__ import annotations

  from elspeth.web.composer.guided.protocol import (
      WireStageData,
      WireTopology,
  )


  class TestWireStageDataShape:
      def test_wire_stage_data_keys(self) -> None:
          data: WireStageData = {
              "topology": {"sources": {}, "nodes": [], "outputs": []},
              "edge_contracts": [],
              "semantic_contracts": [],
              "warnings": [],
          }
          assert set(data.keys()) == {
              "topology",
              "edge_contracts",
              "semantic_contracts",
              "warnings",
          }

      def test_wire_topology_keys(self) -> None:
          topo: WireTopology = {"sources": {}, "nodes": [], "outputs": []}
          assert set(topo.keys()) == {"sources", "nodes", "outputs"}
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py -q`
  Expected: `ImportError: cannot import name 'WireStageData' from 'elspeth.web.composer.guided.protocol'`.
- [ ] **Step 3: Add the TypedDicts.** In `protocol.py`, immediately after the
  `ProposeChainPayload` class (`:62-65`), insert:
  ```python
  class _WireSourceTopo(TypedDict):
      plugin: str
      on_success: str | None


  class _WireNodeTopo(TypedDict):
      id: str
      node_type: str
      plugin: str | None
      input: str | None
      on_success: str | None
      on_error: str | None
      routes: Mapping[str, str] | None
      fork_to: Sequence[str] | None


  class _WireOutputTopo(TypedDict):
      sink_name: str
      plugin: str


  class WireTopology(TypedDict):
      """Connection-label topology for the wire stage (from get_pipeline_state)."""

      sources: Mapping[str, _WireSourceTopo]
      nodes: Sequence[_WireNodeTopo]
      outputs: Sequence[_WireOutputTopo]


  class WireStageData(TypedDict):
      """STEP_4_WIRE turn payload: topology + validate() contract overlay.

      ``edge_contracts`` entries carry keys ``from``/``to`` (EdgeContract.to_dict,
      state.py:359-368) — NOT from_id/to_id. ``warnings`` carries the LIVE
      prompt-shield advisory (prompt_shield_recommendation_warning_pairs) so the
      wire stage surfaces it (D11/B4). The render reconstructs edges from the
      topology connection labels, never from state.edges.

      ``advisor_findings`` / ``signoff_outcome`` are ``NotRequired`` — present only
      on the P5.6/P5.7 sign-off revise re-emit (carrying the advisor findings text
      and the ``SignoffOutcome.value``), absent on the initial confirm.
      """

      topology: WireTopology
      edge_contracts: Sequence[Mapping[str, Any]]
      semantic_contracts: Sequence[Mapping[str, Any]]
      warnings: Sequence[Mapping[str, Any]]
      advisor_findings: NotRequired[str]
      signoff_outcome: NotRequired[str]
  ```
  (Add `NotRequired` to the `from typing import ...` line in `protocol.py` if it is
  not already imported.)
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py -q`
  Expected: `2 passed`.
- [ ] **Step 5: Commit.**
  `git add src/elspeth/web/composer/guided/protocol.py tests/unit/web/composer/guided/test_wire_payload.py && git commit -m "feat(guided): add WireStageData/WireTopology payload TypedDicts (P2.1)"`

---

### Task P2.2: Register `CONFIRM_WIRING` required keys for the wire payload

**Files:**
- Modify `src/elspeth/web/composer/guided/protocol.py` (`_REQUIRED_KEYS` `:200-218`,
  `_NESTED_SHAPES` `:243-253`)

**Interfaces:**
- Consumes: `TurnType.CONFIRM_WIRING` (P1), `validate_payload` (`protocol.py:256`)
- Produces: `_REQUIRED_KEYS[TurnType.CONFIRM_WIRING] = frozenset({"topology",
  "edge_contracts", "semantic_contracts"})`; `_NESTED_SHAPES[TurnType.CONFIRM_WIRING]
  = (("topology", "mapping", frozenset({"sources", "nodes", "outputs"})),)`

> P1 (P1.T1) registers `CONFIRM_WIRING` in `_REQUIRED_KEYS`/`_NESTED_SHAPES` with the
> wire-data keys above (it must, for totality). This task PINS those exact keys with a
> validation test so a P1/P2 drift in the payload contract crashes loudly. If P1 already
> set identical values, Step 3 is a no-op confirmation; otherwise reconcile to the canonical
> values here (the payload data model owns the key set).

- [ ] **Step 1: Write failing test.** Append to
  `tests/unit/web/composer/guided/test_wire_payload.py`:
  ```python
  from elspeth.web.composer.guided.protocol import (
      TurnType,
      validate_payload,
  )


  class TestConfirmWiringValidation:
      def test_valid_wire_payload_passes(self) -> None:
          payload = {
              "topology": {"sources": {}, "nodes": [], "outputs": []},
              "edge_contracts": [],
              "semantic_contracts": [],
              "warnings": [],
          }
          assert validate_payload(TurnType.CONFIRM_WIRING, payload) is None

      def test_missing_topology_rejected(self) -> None:
          payload = {"edge_contracts": [], "semantic_contracts": []}
          err = validate_payload(TurnType.CONFIRM_WIRING, payload)
          assert err is not None
          assert "topology" in err

      def test_topology_must_be_mapping_with_expected_keys(self) -> None:
          payload = {
              "topology": {"sources": {}},  # missing nodes/outputs
              "edge_contracts": [],
              "semantic_contracts": [],
          }
          err = validate_payload(TurnType.CONFIRM_WIRING, payload)
          assert err is not None
          assert "topology" in err
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestConfirmWiringValidation -q`
  Expected: if P1 set different keys, `test_missing_topology_rejected`/`test_valid_wire_payload_passes`
  fail with a required-key mismatch; if `CONFIRM_WIRING` is unregistered, `KeyError`/`ValueError`
  from `validate_payload`.
- [ ] **Step 3: Reconcile the registry entries.** In `_REQUIRED_KEYS` (`:200-218`)
  ensure the `CONFIRM_WIRING` entry is exactly:
  ```python
      TurnType.CONFIRM_WIRING: frozenset({"topology", "edge_contracts", "semantic_contracts"}),
  ```
  In `_NESTED_SHAPES` (`:243-253`) ensure the `CONFIRM_WIRING` entry is exactly:
  ```python
      TurnType.CONFIRM_WIRING: (
          ("topology", "mapping", frozenset({"sources", "nodes", "outputs"})),
      ),
  ```
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestConfirmWiringValidation -q`
  Expected: `3 passed`.
- [ ] **Step 5: Run the protocol totality test to confirm no regression.**
  `uv run pytest tests/unit/web/composer/guided/test_protocol.py -q`
  Expected: `... passed` (no `KeyError` from the totality assertions).
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/composer/guided/protocol.py tests/unit/web/composer/guided/test_wire_payload.py && git commit -m "feat(guided): pin CONFIRM_WIRING required/nested keys for wire payload (P2.2)"`

---

### Task P2.3: `_build_wire_topology` — topology read from connection labels

**Files:**
- Modify `src/elspeth/web/composer/guided/emitters.py` (add helper near `_step_index`,
  `:422-435`)

**Interfaces:**
- Consumes: `CompositionState` (`composer/state.py:1960`), `_serialize_full_pipeline_state`
  (`composer/tools/sessions.py:1071`)
- Produces: `def _build_wire_topology(state: CompositionState) -> WireTopology`

> The topology MUST come from `_serialize_full_pipeline_state` (it carries the
> connection labels `input/on_success/on_error/routes/fork_to`). `preview_pipeline`'s
> own `nodes` list is only `{id, node_type, plugin}` and is NOT a topology source. We
> reuse `_serialize_full_pipeline_state` rather than re-walk the spec so the wire view
> can never drift from `get_pipeline_state`. We project it down to the wire-visible
> topology subset (dropping `options`/`condition`/`branches`/etc.).

- [ ] **Step 1: Write failing test.** Append to
  `tests/unit/web/composer/guided/test_wire_payload.py`:
  ```python
  from collections.abc import Mapping

  from elspeth.web.composer.guided.emitters import _build_wire_topology
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )


  def _canonical_state() -> CompositionState:
      """inline_blob -> web_scrape -> field_mapper -> jsonl (connection-label wiring)."""
      source = SourceSpec(
          plugin="inline_blob",
          on_success="chain_in",
          options={"blob_id": "b1"},
          on_validation_failure="discard",
      )
      scrape = NodeSpec(
          id="scrape",
          node_type="transform",
          plugin="web_scrape",
          input="chain_in",
          on_success="scraped",
          on_error=None,
          options={"url_field": "url"},
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
          trigger=None,
          output_mode=None,
          expected_output_count=None,
      )
      mapper = NodeSpec(
          id="mapper",
          node_type="transform",
          plugin="field_mapper",
          input="scraped",
          on_success="main",
          on_error=None,
          options={"select_only": True},
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
          trigger=None,
          output_mode=None,
          expected_output_count=None,
      )
      out = OutputSpec(
          name="jsonl_out",
          plugin="json",
          options={"path": "out.jsonl", "format": "jsonl"},
          on_write_failure="raise",
      )
      return CompositionState(
          nodes=(scrape, mapper),
          edges=(),
          outputs=(out,),
          metadata=PipelineMetadata(),
          version=1,
          sources={"source": source},
      )


  class TestBuildWireTopology:
      def test_topology_reads_connection_labels(self) -> None:
          topo = _build_wire_topology(_canonical_state())
          assert topo["sources"]["source"] == {
              "plugin": "inline_blob",
              "on_success": "chain_in",
          }
          node_by_id = {n["id"]: n for n in topo["nodes"]}
          assert node_by_id["scrape"]["input"] == "chain_in"
          assert node_by_id["scrape"]["on_success"] == "scraped"
          assert node_by_id["mapper"]["input"] == "scraped"
          assert node_by_id["mapper"]["on_success"] == "main"
          assert topo["outputs"] == [{"sink_name": "jsonl_out", "plugin": "json"}]

      def test_topology_node_subset_drops_options(self) -> None:
          topo = _build_wire_topology(_canonical_state())
          node = topo["nodes"][0]
          assert set(node.keys()) == {
              "id",
              "node_type",
              "plugin",
              "input",
              "on_success",
              "on_error",
              "routes",
              "fork_to",
          }
          assert "options" not in node

      def test_topology_never_reads_state_edges(self) -> None:
          # guided passes edges=() — topology must still reconstruct from labels.
          topo = _build_wire_topology(_canonical_state())
          # source.on_success -> scrape.input forms the first edge by label
          assert topo["sources"]["source"]["on_success"] == "chain_in"
          assert any(n["input"] == "chain_in" for n in topo["nodes"])
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestBuildWireTopology -q`
  Expected: `ImportError: cannot import name '_build_wire_topology'`.
- [ ] **Step 3: Implement `_build_wire_topology`.** In `emitters.py`, add the
  imports to the existing `from elspeth.web.composer.guided.protocol import (...)`
  block (`:30-41`): add `WireTopology`, `_WireNodeTopo`, `_WireOutputTopo`,
  `_WireSourceTopo`. Then add, just before `_step_index` (`:422`):
  ```python
  def _build_wire_topology(state: CompositionState) -> WireTopology:
      """Project the full pipeline state down to the wire-visible topology subset.

      Topology comes from ``_serialize_full_pipeline_state`` (the only source of the
      connection labels ``input/on_success/on_error/routes/fork_to``). The wire view
      reconstructs edges from these labels — it never reads ``state.edges`` (guided
      passes ``edges=()``). Options/condition/branches are dropped; the wire stage
      shows connectivity, not configuration.
      """
      from elspeth.web.composer.tools.sessions import _serialize_full_pipeline_state

      full = _serialize_full_pipeline_state(state, requested_component=None)
      sources: dict[str, _WireSourceTopo] = {
          name: {"plugin": src["plugin"], "on_success": src["on_success"]}
          for name, src in full["sources"].items()
      }
      nodes: list[_WireNodeTopo] = [
          {
              "id": n["id"],
              "node_type": n["node_type"],
              "plugin": n["plugin"],
              "input": n["input"],
              "on_success": n["on_success"],
              "on_error": n["on_error"],
              "routes": n["routes"],
              "fork_to": n["fork_to"],
          }
          for n in full["nodes"]
      ]
      outputs: list[_WireOutputTopo] = [
          {"sink_name": o["sink_name"], "plugin": o["plugin"]} for o in full["outputs"]
      ]
      return {"sources": sources, "nodes": nodes, "outputs": outputs}
  ```
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestBuildWireTopology -q`
  Expected: `3 passed`.
- [ ] **Step 5: Commit.**
  `git add src/elspeth/web/composer/guided/emitters.py tests/unit/web/composer/guided/test_wire_payload.py && git commit -m "feat(guided): _build_wire_topology projects connection-label topology (P2.3)"`

---

### Task P2.4: `build_step_4_wire_turn` — REPLACE the P1.3 skeleton with the two-read merge emitter (final signature)

**Files:**
- Modify `src/elspeth/web/composer/guided/emitters.py` (REPLACE the P1.3 skeleton
  `build_step_4_wire_turn` body + signature, near `:349`)
- Modify `tests/unit/web/composer/guided/test_emitters.py` (update the P1.3 skeleton
  call `build_step_4_wire_turn(validation=...)` → `build_step_4_wire_turn(state)`)

**Interfaces:**
- Consumes: `CompositionState.validate()` (`composer/state.py:2215`, returns
  `ValidationSummary` with `edge_contracts`/`semantic_contracts`/`warnings`),
  `_authoring_validation_payload` (`composer/tools/sessions.py:1153`),
  `_build_wire_topology` (P2.3), `GuidedStep.STEP_4_WIRE` (P1), `TurnType.CONFIRM_WIRING` (P1),
  `CatalogServiceProtocol` (already the TYPE_CHECKING alias in `emitters.py:44` —
  `from elspeth.web.catalog.protocol import CatalogService as CatalogServiceProtocol`;
  reuse it for the `catalog` annotation to match the sibling emitters)
- Produces (FINAL signature — this is the one signature every later phase calls; P5
  call sites depend on the three optional kwargs being present HERE so no P5 task
  re-signs the emitter):
  `def build_step_4_wire_turn(state: CompositionState, *, catalog: CatalogServiceProtocol | None = None, advisor_findings: str | None = None, signoff_outcome: str | None = None) -> Turn`

> **This is a MODIFY, not a fresh create.** P1.3 already defined
> `build_step_4_wire_turn(*, validation: ValidationSummary)` (the reachable
> skeleton). This task REPLACES both the signature and the body: the param goes
> from keyword-only `validation` to positional `state` plus three optional
> kwargs. Because the name already exists, the run-to-fail asserts a CALL-SHAPE
> failure (TypeError), NOT an ImportError.
>
> The emitter merges the two reads into one `WireStageData` payload (one round-trip):
> topology from `_build_wire_topology` (read 1) + `edge_contracts`/`semantic_contracts`/
> `warnings` from `state.validate()` (read 2 — `validate()` is a pure function, no I/O,
> so the emitter stays pure). `EdgeContract.to_dict()` already emits `from`/`to`, so we
> reuse `_authoring_validation_payload` to get the canonical serialized overlay rather
> than re-serializing. `warnings` carries the LIVE prompt-shield advisory (D11/B4) so the
> wire stage surfaces it. `catalog` is accepted (forward-compat for catalog-aware
> rendering) but the payload is catalog-independent; `advisor_findings`/`signoff_outcome`
> (set by the P5.6/P5.7 revise re-emit) are folded into the payload as `advisor_findings`
> / `signoff_outcome` keys when non-`None`, distinguishing a revise re-emit from the
> initial confirm.

- [ ] **Step 1: Write failing test (call-shape change).** Append to
  `tests/unit/web/composer/guided/test_wire_payload.py`:
  ```python
  from elspeth.web.composer.guided.emitters import build_step_4_wire_turn
  from elspeth.web.composer.guided.protocol import GuidedStep, validate_payload


  class TestBuildStep4WireTurn:
      def test_turn_type_and_step(self) -> None:
          turn = build_step_4_wire_turn(_canonical_state())
          assert turn["type"] == TurnType.CONFIRM_WIRING.value
          # step_index is the 0-based ordinal of STEP_4_WIRE in the _ORDER tuple.
          from elspeth.web.composer.guided.emitters import _step_index

          assert turn["step_index"] == _step_index(GuidedStep.STEP_4_WIRE)

      def test_payload_merges_topology_and_contracts(self) -> None:
          turn = build_step_4_wire_turn(_canonical_state())
          payload = turn["payload"]
          assert set(payload.keys()) == {
              "topology",
              "edge_contracts",
              "semantic_contracts",
              "warnings",
          }
          assert payload["topology"]["sources"]["source"]["plugin"] == "inline_blob"

      def test_edge_contracts_use_from_to_keys(self) -> None:
          turn = build_step_4_wire_turn(_canonical_state())
          for ec in turn["payload"]["edge_contracts"]:
              # M1: keys are from/to, NOT from_id/to_id.
              assert "from" in ec
              assert "to" in ec
              assert "from_id" not in ec
              assert "to_id" not in ec

      def test_payload_validates(self) -> None:
          turn = build_step_4_wire_turn(_canonical_state())
          assert validate_payload(TurnType.CONFIRM_WIRING, turn["payload"]) is None

      def test_revise_kwargs_fold_into_payload(self) -> None:
          # P5.6/P5.7 revise re-emit: advisor_findings + signoff_outcome are
          # carried so the frontend renders the revise / fail-closed affordance.
          turn = build_step_4_wire_turn(
              _canonical_state(),
              advisor_findings="FLAGGED: prompt sees no row field",
              signoff_outcome="revise",
          )
          assert turn["payload"]["advisor_findings"] == "FLAGGED: prompt sees no row field"
          assert turn["payload"]["signoff_outcome"] == "revise"

      def test_initial_confirm_omits_revise_keys(self) -> None:
          # The initial confirm (no advisor pass yet) carries neither key.
          turn = build_step_4_wire_turn(_canonical_state())
          assert "advisor_findings" not in turn["payload"]
          assert "signoff_outcome" not in turn["payload"]
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestBuildStep4WireTurn -q`
  Expected: `TypeError: build_step_4_wire_turn() takes 0 positional arguments but 1 was given`
  (the P1.3 skeleton is keyword-only `validation`; the positional `state` call fails
  the shape — NOT an `ImportError`, because P1.3 already defined the name).
- [ ] **Step 3: REPLACE the emitter.** In `emitters.py`, add `WireStageData` to the
  protocol import block. The `CatalogServiceProtocol` alias is ALREADY in the
  `TYPE_CHECKING` block (`:44`), so no new import is needed for the `catalog`
  annotation. REPLACE the P1.3 skeleton `build_step_4_wire_turn` (near `:349`) with:
  ```python
  def build_step_4_wire_turn(
      state: CompositionState,
      *,
      catalog: CatalogServiceProtocol | None = None,
      advisor_findings: str | None = None,
      signoff_outcome: str | None = None,
  ) -> Turn:
      """Build the STEP_4_WIRE ``confirm_wiring`` Turn (two-read merge, B2).

      Read 1 — topology: ``_build_wire_topology`` (connection labels from
      ``_serialize_full_pipeline_state``; NEVER ``state.edges``).
      Read 2 — contract overlay: ``state.validate()`` provides ``edge_contracts``
      (keys ``from``/``to`` — M1), ``semantic_contracts``, and ``warnings`` (which
      carries the LIVE prompt-shield advisory, D11/B4). ``validate()`` is a pure
      function, so this emitter stays pure (no I/O, no clock, no uuid).

      ``catalog`` is accepted for forward-compat (catalog-aware rendering) but the
      payload is catalog-independent. ``advisor_findings`` / ``signoff_outcome`` are
      set by the P5.6/P5.7 sign-off revise re-emit; when non-``None`` they are folded
      into the payload so the frontend distinguishes a revise re-emit (showing the
      advisor findings + outcome class) from the initial confirm.
      """
      from elspeth.web.composer.tools.sessions import _authoring_validation_payload

      validation = state.validate()
      overlay = _authoring_validation_payload(state, validation)
      payload: WireStageData = {
          "topology": _build_wire_topology(state),
          "edge_contracts": overlay["edge_contracts"],
          "semantic_contracts": overlay["semantic_contracts"],
          "warnings": overlay["warnings"],
      }
      if advisor_findings is not None:
          payload["advisor_findings"] = advisor_findings
      if signoff_outcome is not None:
          payload["signoff_outcome"] = signoff_outcome
      return Turn(
          type=TurnType.CONFIRM_WIRING.value,
          step_index=_step_index(GuidedStep.STEP_4_WIRE),
          payload=payload,
      )
  ```
  > `WireStageData` (the `protocol.py` TypedDict from P2.1) already declares
  > `advisor_findings` and `signoff_outcome` as `NotRequired[str]`, so the
  > conditional assignment above typechecks with no further protocol change.
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestBuildStep4WireTurn -q`
  Expected: `6 passed`.
- [ ] **Step 5: Update the P1.3 skeleton emitter test + run the full module.**
  In `tests/unit/web/composer/guided/test_emitters.py`, change the P1.3 skeleton call
  `build_step_4_wire_turn(validation=_empty_state().validate())` to
  `build_step_4_wire_turn(_empty_state())` and update the three skeleton-payload
  assertions (`topology == {}`, `edge_contracts == []`, `semantic_contracts == []`) to
  the real shape: `payload["topology"]["sources"] == {}` and `payload["edge_contracts"] == []`
  for the empty state (no source/sink), `set(payload.keys()) == {"topology", "edge_contracts", "semantic_contracts", "warnings"}`.
  ```
  uv run pytest tests/unit/web/composer/guided/test_wire_payload.py tests/unit/web/composer/guided/test_emitters.py -q
  ```
  Expected: all pass.
- [ ] **Step 6: Export the emitter.** The P1.3 `Exported:` line already names
  `build_step_4_wire_turn`; update its one-line description (`emitters.py:8-18`) to:
  `    build_step_4_wire_turn — build the STEP_4_WIRE confirm_wiring turn (two-read merge).`
- [ ] **Step 7: Commit.**
  `git add src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py tests/unit/web/composer/guided/test_wire_payload.py tests/unit/web/composer/guided/test_emitters.py && git commit -m "feat(guided): replace skeleton build_step_4_wire_turn with two-read merge + final signature (P2.4)"`

---

### Task P2.5: Honest-gap rendering — coalesce/fork skip `edge_contracts`

**Files:**
- Modify `tests/unit/web/composer/guided/test_wire_payload.py` (add coalesce/fork case)

**Interfaces:**
- Consumes: `build_step_4_wire_turn` (P2.4), `validate()` edge_contracts behaviour
- Produces: a pinned-behaviour test (no new prod code if `validate()` already skips
  coalesce/fork edges; otherwise a topology-render note)

> Per §5/B2 honest-gap rule: coalesce/fork nodes are "not statically checkable", so
> `validate()` does not emit `edge_contracts` for their edges. The wire payload simply
> carries whatever `validate()` produced — the GAP is honest (a fork/coalesce edge appears
> in `topology` with NO matching `edge_contracts` row, which the render colours "unchecked").
> This task PINS that the emitter does not fabricate a contract row for fork/coalesce edges.

- [ ] **Step 1: Write failing test.** Append:
  ```python
  class TestHonestGapRendering:
      def test_fork_node_has_topology_but_may_lack_edge_contract(self) -> None:
          # A fork node carries fork_to in topology; edge_contracts for its edges
          # are honest-gap (validate() does not statically check fork fan-out).
          source = SourceSpec(
              plugin="inline_blob",
              on_success="chain_in",
              options={"blob_id": "b1"},
              on_validation_failure="discard",
          )
          fork = NodeSpec(
              id="fork",
              node_type="gate",
              plugin=None,
              input="chain_in",
              on_success=None,
              on_error=None,
              options={},
              condition=None,
              routes=None,
              fork_to=["branch_a", "branch_b"],
              branches=None,
              policy=None,
              merge=None,
              trigger=None,
              output_mode=None,
              expected_output_count=None,
          )
          out = OutputSpec(
              name="jsonl_out",
              plugin="json",
              options={"path": "out.jsonl", "format": "jsonl"},
              on_write_failure="raise",
          )
          state = CompositionState(
              nodes=(fork,),
              edges=(),
              outputs=(out,),
              metadata=PipelineMetadata(),
              version=1,
              sources={"source": source},
          )
          turn = build_step_4_wire_turn(state)
          node = turn["payload"]["topology"]["nodes"][0]
          assert node["fork_to"] == ["branch_a", "branch_b"]
          # No fabricated contract row keyed on the fork's fan-out edges.
          ec_pairs = {(ec["from"], ec["to"]) for ec in turn["payload"]["edge_contracts"]}
          assert ("fork", "branch_a") not in ec_pairs
          assert ("fork", "branch_b") not in ec_pairs
  ```
- [ ] **Step 2: Run to pass (no new prod code expected).**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestHonestGapRendering -q`
  Expected: `1 passed` (the emitter only mirrors `validate()` output — it never
  fabricates contract rows). If it FAILS, the cause is `validate()` emitting a contract
  for fork edges — that is an upstream behaviour, not this phase's; in that case adjust
  the assertion to pin `satisfied is True` is NOT asserted for that gap edge and add a
  comment, do not add fabrication.
- [ ] **Step 3: Commit.**
  `git add tests/unit/web/composer/guided/test_wire_payload.py && git commit -m "test(guided): pin honest-gap rendering for fork/coalesce edges (P2.5)"`

---

### Task P2.6: `WireStageData` TS type + `guided.ts` mirror

**Files:**
- Modify `src/elspeth/web/frontend/src/types/guided.ts` (add after `ProposeChainPayload`,
  `:297-301`)
- Modify `src/elspeth/web/frontend/src/types/guided.test.ts`

**Interfaces:**
- Consumes: `EdgeContract.to_dict` key shape (`from`/`to`/`producer_guarantees`/
  `consumer_requires`/`missing_fields`/`satisfied`)
- Produces:
  ```ts
  export interface WireStageData {
    topology: {
      sources: Record<string, { plugin: string; on_success: string | null }>;
      nodes: Array<{ id: string; node_type: string; plugin: string | null; input: string | null; on_success: string | null; on_error: string | null; routes: Record<string, string> | null; fork_to: string[] | null }>;
      outputs: Array<{ sink_name: string; plugin: string }>;
    };
    edge_contracts: Array<{ from: string; to: string; producer_guarantees: string[]; consumer_requires: string[]; missing_fields: string[]; satisfied: boolean }>;
    semantic_contracts: Array<Record<string, unknown>>;
    warnings: Array<Record<string, unknown>>;
  }
  ```

> `edge_contracts` keys are `from`/`to` (NOT `from_id`/`to_id`) — M1. `node.plugin` is
> nullable (gates/coalesces). Vitest type-assertion failures ARE the test (the array
> literal stops compiling if the shape drifts). P1 owns adding `"step_4_wire"` to the
> `GuidedStep` union and `"confirm_wiring"` to the `TurnType` union — this task does NOT
> touch those unions (it would collide with P1); it only adds `WireStageData`.

- [ ] **Step 1: Write failing test.** Append to `guided.test.ts`:
  ```ts
  import type { WireStageData } from "./guided";

  describe("WireStageData wire shape", () => {
    it("carries topology + from/to edge_contracts", () => {
      const data: WireStageData = {
        topology: {
          sources: { source: { plugin: "inline_blob", on_success: "chain_in" } },
          nodes: [
            {
              id: "scrape",
              node_type: "transform",
              plugin: "web_scrape",
              input: "chain_in",
              on_success: "scraped",
              on_error: null,
              routes: null,
              fork_to: null,
            },
          ],
          outputs: [{ sink_name: "jsonl_out", plugin: "json" }],
        },
        edge_contracts: [
          {
            from: "scrape",
            to: "mapper",
            producer_guarantees: ["content"],
            consumer_requires: ["content"],
            missing_fields: [],
            satisfied: true,
          },
        ],
        semantic_contracts: [],
        warnings: [],
      };
      expect(data.edge_contracts[0].from).toBe("scrape");
      expect(data.edge_contracts[0].to).toBe("mapper");
      // @ts-expect-error edge_contracts keys are from/to, NOT from_id.
      data.edge_contracts[0].from_id;
    });
  });
  ```
- [ ] **Step 2: Run to fail (from `src/elspeth/web/frontend`).**
  `npm test -- --run src/types/guided.test.ts`
  Expected: TS compile error `Module '"./guided"' has no exported member 'WireStageData'`.
- [ ] **Step 3: Add the TS type.** In `guided.ts`, after `ProposeChainPayload`
  (`:297-301`), append:
  ```ts
  /**
   * Wire: WireStageData — the STEP_4_WIRE confirm_wiring payload (B2).
   *
   * Two-read merge from the backend: `topology` (connection labels from
   * get_pipeline_state / _serialize_full_pipeline_state) + `edge_contracts` /
   * `semantic_contracts` / `warnings` (from state.validate()).
   *
   * `edge_contracts` keys are `from`/`to` (EdgeContract.to_dict, state.py:359-368)
   * — NOT from_id/to_id. `warnings` carries the LIVE prompt-shield advisory
   * (prompt_shield_recommendation_warning_pairs) for the web_scrape recipe (D11/B4).
   *
   * Render: reconstruct edges from the topology connection labels
   * (source.on_success -> node.input; node.on_success/routes/fork_to -> downstream)
   * and overlay `edge_contracts` keyed by (from, to). NEVER render state.edges
   * directly (guided passes edges=[]). Coalesce/fork edges are honest-gap — they
   * appear in topology with no matching edge_contracts row (render: "unchecked").
   */
  export interface WireStageData {
    topology: {
      sources: Record<string, { plugin: string; on_success: string | null }>;
      nodes: Array<{
        id: string;
        node_type: string;
        plugin: string | null;
        input: string | null;
        on_success: string | null;
        on_error: string | null;
        routes: Record<string, string> | null;
        fork_to: string[] | null;
      }>;
      outputs: Array<{ sink_name: string; plugin: string }>;
    };
    edge_contracts: Array<{
      from: string;
      to: string;
      producer_guarantees: string[];
      consumer_requires: string[];
      missing_fields: string[];
      satisfied: boolean;
    }>;
    semantic_contracts: Array<Record<string, unknown>>;
    warnings: Array<Record<string, unknown>>;
  }
  ```
- [ ] **Step 4: Run to pass.**
  `npm test -- --run src/types/guided.test.ts`
  Expected: the `WireStageData` describe block passes.
- [ ] **Step 5: Run the SlotType / guided.ts mirror gate (from repo root).**
  `uv run python scripts/cicd/check_slot_type_cross_language.py`
  Expected: exit 0 (the `RecipeSlotInput` interface is untouched; adding `WireStageData`
  does not affect the SlotType mirror).
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/frontend/src/types/guided.ts src/elspeth/web/frontend/src/types/guided.test.ts && git commit -m "feat(frontend): add WireStageData TS type (P2.6)"`

---

### Task P2.7: `WireStageTurn.tsx` — render from connection labels, overlay contracts

**Files:**
- Create `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx`
- Create `src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx`

**Interfaces:**
- Consumes: `WireStageData` (P2.6)
- Produces:
  ```ts
  export interface WireEdge {
    from: string;
    to: string;
    label: string;
    satisfied: boolean | null; // null = honest-gap (no edge_contracts row)
    missing_fields: string[];
  }
  export function reconstructWireEdges(data: WireStageData): WireEdge[];
  export interface WireStageTurnProps {
    data: WireStageData;
    onConfirm: () => void;
    confirmDisabled: boolean;
  }
  export function WireStageTurn(props: WireStageTurnProps): JSX.Element;
  ```

> `reconstructWireEdges` builds edges from connection labels (`source.on_success ->
> node.input`; `node.on_success / routes / fork_to -> downstream node.input`) and
> overlays `edge_contracts` keyed by `(from, to)`. An edge with no matching contract
> row gets `satisfied: null` (honest-gap, e.g. fork/coalesce). NEVER reads `state.edges`.
> The `confirmDisabled` prop is the block-while-pending hook the P4 frontend
> (interpretation projection) drives; this phase just exposes it.

- [ ] **Step 1: Write failing test for `reconstructWireEdges`.** Create
  `WireStageTurn.test.tsx`:
  ```tsx
  import { describe, expect, it } from "vitest";
  import { render, screen } from "@testing-library/react";

  import type { WireStageData } from "../../../types/guided";
  import { reconstructWireEdges, WireStageTurn } from "./WireStageTurn";

  function canonicalData(): WireStageData {
    return {
      topology: {
        sources: { source: { plugin: "inline_blob", on_success: "chain_in" } },
        nodes: [
          {
            id: "scrape",
            node_type: "transform",
            plugin: "web_scrape",
            input: "chain_in",
            on_success: "scraped",
            on_error: null,
            routes: null,
            fork_to: null,
          },
          {
            id: "mapper",
            node_type: "transform",
            plugin: "field_mapper",
            input: "scraped",
            on_success: "main",
            on_error: null,
            routes: null,
            fork_to: null,
          },
        ],
        outputs: [{ sink_name: "jsonl_out", plugin: "json" }],
      },
      edge_contracts: [
        {
          from: "scrape",
          to: "mapper",
          producer_guarantees: ["content"],
          consumer_requires: ["content"],
          missing_fields: [],
          satisfied: true,
        },
      ],
      semantic_contracts: [],
      warnings: [],
    };
  }

  describe("reconstructWireEdges", () => {
    it("builds edges from connection labels, never state.edges", () => {
      const edges = reconstructWireEdges(canonicalData());
      const pairs = edges.map((e) => [e.from, e.to]);
      // source.on_success=chain_in -> scrape.input=chain_in
      expect(pairs).toContainEqual(["source", "scrape"]);
      // scrape.on_success=scraped -> mapper.input=scraped
      expect(pairs).toContainEqual(["scrape", "mapper"]);
    });

    it("overlays edge_contracts keyed by (from, to)", () => {
      const edges = reconstructWireEdges(canonicalData());
      const scrapeToMapper = edges.find((e) => e.from === "scrape" && e.to === "mapper");
      expect(scrapeToMapper?.satisfied).toBe(true);
    });

    it("marks an edge with no contract row as honest-gap (satisfied=null)", () => {
      const edges = reconstructWireEdges(canonicalData());
      const sourceToScrape = edges.find((e) => e.from === "source" && e.to === "scrape");
      expect(sourceToScrape?.satisfied).toBeNull();
    });
  });

  describe("WireStageTurn", () => {
    it("renders the prompt-shield advisory warning when present (D11/B4)", () => {
      const data = canonicalData();
      data.warnings = [{ severity: "medium", message: "prompt-injection shield recommended" }];
      render(<WireStageTurn data={data} onConfirm={() => {}} confirmDisabled={false} />);
      expect(screen.getByText(/prompt-injection shield recommended/)).toBeInTheDocument();
    });

    it("disables confirm when confirmDisabled is true", () => {
      render(
        <WireStageTurn data={canonicalData()} onConfirm={() => {}} confirmDisabled={true} />,
      );
      expect(screen.getByRole("button", { name: /confirm wiring/i })).toBeDisabled();
    });

    it("conveys edge status as TEXT, not colour alone (WCAG 1.4.1)", () => {
      render(<WireStageTurn data={canonicalData()} onConfirm={() => {}} confirmDisabled={false} />);
      // scrape->mapper is satisfied=true; source->scrape is satisfied=null. Each
      // must render a text status token, not only a --ok/--unchecked CSS class, so
      // the state is visible to screen readers and colour-blind users (an edge with
      // satisfied===false and empty missing_fields is otherwise colour-only).
      expect(screen.getByText(/\(connected\)/)).toBeInTheDocument();
      expect(screen.getByText(/\(contract unchecked\)/)).toBeInTheDocument();
    });
  });
  ```
- [ ] **Step 2: Run to fail (from `src/elspeth/web/frontend`).**
  `npm test -- --run src/components/chat/guided/WireStageTurn.test.tsx`
  Expected: cannot resolve `./WireStageTurn`.
- [ ] **Step 3: Implement the component.** Create `WireStageTurn.tsx`:
  ```tsx
  import type { JSX } from "react";

  import type { WireStageData } from "../../../types/guided";

  export interface WireEdge {
    from: string;
    to: string;
    /** The named connection label this edge flows over. */
    label: string;
    /** true/false from edge_contracts; null = honest-gap (no contract row). */
    satisfied: boolean | null;
    missing_fields: string[];
  }

  /**
   * Reconstruct pipeline edges from the topology's connection labels (B2 hard
   * constraint): source.on_success -> node.input; node.on_success / routes values
   * / fork_to -> downstream node.input. NEVER reads state.edges (guided passes
   * edges=[]). Overlays edge_contracts keyed by (from, to); an edge with no
   * matching contract row is honest-gap (satisfied=null, e.g. fork/coalesce).
   */
  export function reconstructWireEdges(data: WireStageData): WireEdge[] {
    const { sources, nodes } = data.topology;
    // Map each consumed label -> the node id that reads it.
    const consumerByLabel = new Map<string, string>();
    for (const node of nodes) {
      if (node.input !== null) {
        consumerByLabel.set(node.input, node.id);
      }
    }
    const contractByPair = new Map<
      string,
      { satisfied: boolean; missing_fields: string[] }
    >();
    for (const ec of data.edge_contracts) {
      contractByPair.set(`${ec.from}\u0000${ec.to}`, {
        satisfied: ec.satisfied,
        missing_fields: ec.missing_fields,
      });
    }

    const edges: WireEdge[] = [];
    const pushEdge = (from: string, label: string | null): void => {
      if (label === null) {
        return;
      }
      const to = consumerByLabel.get(label);
      if (to === undefined) {
        return;
      }
      const contract = contractByPair.get(`${from}\u0000${to}`);
      edges.push({
        from,
        to,
        label,
        satisfied: contract ? contract.satisfied : null,
        missing_fields: contract ? contract.missing_fields : [],
      });
    };

    for (const [name, src] of Object.entries(sources)) {
      pushEdge(name, src.on_success);
    }
    for (const node of nodes) {
      pushEdge(node.id, node.on_success);
      if (node.routes !== null) {
        for (const label of Object.values(node.routes)) {
          pushEdge(node.id, label);
        }
      }
      if (node.fork_to !== null) {
        for (const label of node.fork_to) {
          pushEdge(node.id, label);
        }
      }
    }
    return edges;
  }

  export interface WireStageTurnProps {
    data: WireStageData;
    onConfirm: () => void;
    /** Block-while-pending hook driven by the P4 interpretation projection. */
    confirmDisabled: boolean;
  }

  export function WireStageTurn(props: WireStageTurnProps): JSX.Element {
    const edges = reconstructWireEdges(props.data);
    return (
      <div className="wire-stage" data-testid="wire-stage-turn">
        <h3>See how the pieces connect</h3>
        {props.data.warnings.length > 0 && (
          <ul className="wire-stage__warnings" data-testid="wire-stage-warnings">
            {props.data.warnings.map((w, i) => (
              <li key={i} className="wire-stage__warning">
                {String((w as { message?: unknown }).message ?? "")}
              </li>
            ))}
          </ul>
        )}
        <ul className="wire-stage__edges">
          {edges.map((e) => (
            <li
              key={`${e.from}->${e.to}`}
              className={
                e.satisfied === null
                  ? "wire-stage__edge wire-stage__edge--unchecked"
                  : e.satisfied
                    ? "wire-stage__edge wire-stage__edge--ok"
                    : "wire-stage__edge wire-stage__edge--unsatisfied"
              }
            >
              {e.from} &rarr; {e.to}
              {/* Status as TEXT, not colour alone (WCAG 1.4.1): an edge with
                  satisfied===false and empty missing_fields is otherwise
                  distinguishable only by the --unsatisfied CSS class, invisible to
                  screen readers and colour-blind users. */}
              <span className="wire-stage__edge-status">
                {" "}
                {e.satisfied === null
                  ? "(contract unchecked)"
                  : e.satisfied
                    ? "(connected)"
                    : "(not satisfied)"}
              </span>
              {e.missing_fields.length > 0 && (
                <span className="wire-stage__missing">
                  {" "}
                  missing: {e.missing_fields.join(", ")}
                </span>
              )}
            </li>
          ))}
        </ul>
        <button
          type="button"
          onClick={props.onConfirm}
          disabled={props.confirmDisabled}
        >
          Confirm wiring
        </button>
      </div>
    );
  }
  ```
- [ ] **Step 4: Run to pass.**
  `npm test -- --run src/components/chat/guided/WireStageTurn.test.tsx`
  Expected: all 5 tests pass.
- [ ] **Step 5: Typecheck + build (from `src/elspeth/web/frontend`).**
  `npm run typecheck && npm run build`
  Expected: both succeed (no TS errors).
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx && git commit -m "feat(frontend): WireStageTurn renders edges from connection labels + contract overlay (P2.7)"`

---

### Task P2.8: B6 re-validate + re-surface after a wire-stage reconciliation

**Files:**
- Modify `src/elspeth/web/composer/guided/emitters.py` (add `rebuild_wire_turn_after_reconciliation`)
- Modify `tests/unit/web/composer/guided/test_wire_payload.py`

**Interfaces:**
- Consumes: `CompositionState` (post-reconciliation), `build_step_4_wire_turn` (P2.4),
  a `resurface` callback `Callable[[CompositionState], None]` (the P3 surfacing pass,
  passed in so this phase is independently testable)
- Produces:
  ```python
  def rebuild_wire_turn_after_reconciliation(
      state: CompositionState,
      *,
      resurface: Callable[[CompositionState], None],
  ) -> tuple[Turn, bool]:
      """Re-evaluate the wire turn after upsert_node/set_pipeline reconciliation.

      Returns (rebuilt STEP_4_WIRE turn, is_valid). Re-runs ``resurface`` on the
      post-mutation state (B6 — never trust transform-commit-time surfacing at the
      wire terminal) before rebuilding the turn from the fresh validate().
      """
  ```

> B6: after any wire-stage reconciliation (`upsert_node`/`set_pipeline` for a
> `field_mapper` insert or a schema-relax-to-flexible) the confirm gate MUST
> re-evaluate `validate().is_valid` AND re-run the P3 surfacing pass on the
> post-mutation state. The `resurface` callback is the P3
> `_surface_pending_interpretation_reviews` (bound at the dispatcher). The function
> returns the freshly-built wire turn (so a stale advisory card cannot persist) and
> the `is_valid` flag the dispatcher uses to gate the confirm.

- [ ] **Step 1: Write failing test.** Append:
  ```python
  class TestWireReconciliationRebuild:
      def test_resurface_called_on_post_mutation_state_and_turn_rebuilt(self) -> None:
          from elspeth.web.composer.guided.emitters import (
              rebuild_wire_turn_after_reconciliation,
          )

          state = _canonical_state()
          seen: list[int] = []

          def resurface(s: CompositionState) -> None:
              seen.append(s.version)

          turn, is_valid = rebuild_wire_turn_after_reconciliation(
              state, resurface=resurface
          )
          # B6: surfacing ran on the exact post-mutation state passed in.
          assert seen == [state.version]
          # The rebuilt turn is a fresh wire turn from the current validate().
          assert turn["type"] == TurnType.CONFIRM_WIRING.value
          assert turn["payload"]["topology"]["sources"]["source"]["plugin"] == "inline_blob"
          assert isinstance(is_valid, bool)

      def test_is_valid_reflects_fresh_validate(self) -> None:
          from elspeth.web.composer.guided.emitters import (
              rebuild_wire_turn_after_reconciliation,
          )

          # A state with no outputs is invalid; the rebuild must report it.
          source = SourceSpec(
              plugin="inline_blob",
              on_success="main",
              options={"blob_id": "b1"},
              on_validation_failure="discard",
          )
          state = CompositionState(
              nodes=(),
              edges=(),
              outputs=(),
              metadata=PipelineMetadata(),
              version=1,
              sources={"source": source},
          )
          _turn, is_valid = rebuild_wire_turn_after_reconciliation(
              state, resurface=lambda _s: None
          )
          assert is_valid is False
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestWireReconciliationRebuild -q`
  Expected: `ImportError: cannot import name 'rebuild_wire_turn_after_reconciliation'`.
- [ ] **Step 3: Implement.** In `emitters.py`, add `from collections.abc import Callable`
  to the existing `from collections.abc import ...` import (`:26`), and add after
  `build_step_4_wire_turn`:
  ```python
  def rebuild_wire_turn_after_reconciliation(
      state: CompositionState,
      *,
      resurface: Callable[[CompositionState], None],
  ) -> tuple[Turn, bool]:
      """Re-evaluate the wire turn after a wire-stage reconciliation (B6).

      After any ``upsert_node`` / ``set_pipeline`` reconciliation at the wire stage
      (a ``field_mapper`` insert or a schema relax), the confirm gate must (1) re-run
      the P3 interpretation surfacing pass on the POST-mutation state — never trust
      transform-commit-time results at the wire terminal — and (2) rebuild the wire
      turn from a fresh ``validate()`` so a cosmetically-stale advisory card cannot
      persist. Returns ``(rebuilt STEP_4_WIRE turn, validate().is_valid)``.
      """
      resurface(state)
      turn = build_step_4_wire_turn(state)
      return turn, state.validate().is_valid
  ```
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestWireReconciliationRebuild -q`
  Expected: `2 passed`.
- [ ] **Step 5: Run the full P2 test module + emitters + protocol.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py tests/unit/web/composer/guided/test_emitters.py tests/unit/web/composer/guided/test_protocol.py -q`
  Expected: all pass.
- [ ] **Step 6: Export.** Add `rebuild_wire_turn_after_reconciliation` to the
  `emitters.py` docstring `Exported:` block:
  `    rebuild_wire_turn_after_reconciliation — re-validate + re-surface the wire turn (B6).`
- [ ] **Step 7: Commit.**
  `git add src/elspeth/web/composer/guided/emitters.py tests/unit/web/composer/guided/test_wire_payload.py && git commit -m "feat(guided): B6 re-validate + re-surface wire turn after reconciliation (P2.8)"`

---

### Task P2.9: Wire `STEP_4_WIRE` into the dispatcher — emit the wire turn after each accept commit + add the `CONFIRM_WIRING` dispatch branch + GET rebuild

**Files:**
- Modify `src/elspeth/web/sessions/routes/_helpers.py` (`_dispatch_guided_respond`,
  `:2554`): the recipe-apply commit return (`:3280`), the chain-accept commit return
  (`:3554`), and the chain-accept repair-success return (`:3520`) — replace each
  `return state, guided, None` with a `build_step_4_wire_turn` emission; add the
  `if current_step is GuidedStep.STEP_4_WIRE:` / `CONFIRM_WIRING` dispatch branch
- Modify `src/elspeth/web/sessions/routes/composer.py` (`_build_get_guided_turn`,
  `:1313` closure; `:1404-1421` tail): add a `STEP_4_WIRE` rebuild branch so a GET
  `/guided` on a wire-positioned session re-emits the wire turn
- Create `tests/integration/web/composer/guided/test_wire_dispatch.py`

**Interfaces:**
- Consumes: `build_step_4_wire_turn` (**P2.4**, positional `state`),
  `handle_step_4_wire_confirm` (**P1.6**, already imported into `_helpers.py` at
  `:97`/`__all__` `:3921`), `GuidedStep.STEP_4_WIRE` + `TurnType.CONFIRM_WIRING`
  (**P1**), `TurnRecord`, `stable_hash`, `emit_turn_emitted`, `_replace` (already in
  `_helpers.py`).
- Produces: behaviour — after a recipe-apply / chain-accept / repair-success commit
  (which P1.6 left at `step=STEP_4_WIRE, terminal=None`), the dispatcher returns
  `next_turn = build_step_4_wire_turn(state)` (no longer `None`) so the user LANDS on
  the wire stage; a subsequent `CONFIRM_WIRING` response routes to
  `handle_step_4_wire_confirm` and stamps `TerminalState(COMPLETED)` on a valid
  pipeline (P5.6 later layers the profile-gated sign-off in front of that stamp).
- Produces: GET `/guided` on a `step=STEP_4_WIRE` session rebuilds the wire turn via
  `build_step_4_wire_turn(state)`.

> **Why this task exists.** P1.6 moved the COMPLETED stamp OUT of the two accept
> seams (leaving `step=STEP_4_WIRE, terminal=None`), but the dispatcher still
> returned `next_turn=None` — so the user would silently fall off the end of the
> wizard onto a wire stage with no turn to render, and a `CONFIRM_WIRING` response
> would hit the generic 400 (no branch handles it). This task closes both ends:
> (1) emit the wire turn after each accept commit; (2) dispatch the confirm. It is a
> hard dependency of P5.6 (which only LAYERS the sign-off onto the branch this task
> creates). Spec §4.2 (`_helpers.py:3339`/`:3263`).

- [ ] **Step 1: Read the three accept-commit return sites + the dispatcher tail to anchor the edits.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && sed -n '3277,3281p;3518,3521p;3551,3563p' src/elspeth/web/sessions/routes/_helpers.py
  ```
  Confirm: recipe-apply returns `return state, guided, None` (`:3280`); repair-success
  returns `return repair_result.state, repair_result.session, None` (`:3520`);
  chain-accept success returns `return handler_result.state, handler_result.session, None`
  (`:3554`). After P1.6, each of these sessions carries `step=STEP_4_WIRE, terminal=None`.

- [ ] **Step 2: Write the failing dispatch test.**
  Create `tests/integration/web/composer/guided/test_wire_dispatch.py`. It drives a
  chain-accept commit through `_dispatch_guided_respond` and asserts the returned
  `next_turn` is the wire turn (not `None`), then drives a `CONFIRM_WIRING` confirm and
  asserts COMPLETED:
  ```python
  """Phase P2.9 — the dispatcher emits the wire turn after accept + dispatches CONFIRM_WIRING."""

  from __future__ import annotations

  import pytest

  from elspeth.web.composer.audit import BufferingRecorder
  from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
  from elspeth.web.composer.guided.state_machine import (
      ChainProposal,
      GuidedSession,
      SinkOutputResolved,
      SinkResolved,
      SourceResolved,
      TerminalKind,
  )
  from elspeth.web.composer.guided.steps import (
      handle_step_1_source,
      handle_step_2_sink,
      handle_step_3_chain_accept,
  )
  from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
  from elspeth.web.dependencies import create_catalog_service
  from tests.integration.web.composer.guided.test_step_handlers import _empty_state


  def _wire_ready_session_and_state():
      """Drive source -> sink -> chain-accept so the session is at STEP_4_WIRE.

      Mirrors TestTerminalStampInvariant in test_step_handlers.py (P1.6): the
      chain-accept handler stages step_3_proposal AND (post-P1.6) leaves
      step=STEP_4_WIRE, terminal=None. Returns the post-accept state/session/catalog.
      """
      state = _empty_state()
      session = GuidedSession.initial()
      catalog = create_catalog_service()
      step_1 = handle_step_1_source(
          state=state,
          session=session,
          catalog=catalog,
          resolved=SourceResolved(
              plugin="csv",
              options={"path": "x.csv", "schema": {"mode": "observed"}},
              observed_columns=("price",),
              sample_rows=({"price": "1.99"},),
          ),
      )
      step_2 = handle_step_2_sink(
          state=step_1.state,
          session=step_1.session,
          catalog=catalog,
          resolved=SinkResolved(
              outputs=(
                  SinkOutputResolved(
                      plugin="json",
                      options={"path": "out.jsonl", "schema": {"mode": "observed"}},
                      required_fields=("price",),
                      schema_mode="observed",
                  ),
              ),
          ),
      )
      proposal = ChainProposal(
          steps=(
              {"plugin": "passthrough", "options": {"schema": {"mode": "observed"}}, "rationale": "echo"},
          ),
          why="single-step chain",
      )
      accept = handle_step_3_chain_accept(
          state=step_2.state,
          session=step_2.session,
          catalog=catalog,
          proposal=proposal,
      )
      assert accept.tool_result.success is True
      # P1.6 invariant: the accept handler leaves step=STEP_4_WIRE, terminal=None.
      assert accept.session.step is GuidedStep.STEP_4_WIRE
      assert accept.session.terminal is None
      return accept.state, accept.session, catalog


  async def _dispatch(state, session, catalog, *, current_step, current_turn_type, turn_response):
      # NOTE: P2 runs BEFORE P5.4, so the dispatcher does NOT yet take
      # composer_service / advisor_checkpoint_max_passes — do NOT pass them here.
      # P2.9's wire branch is the validate-gate-only confirm; the sign-off
      # params arrive in P5.4 and the gate is layered in P5.6.
      return await _dispatch_guided_respond(
          state=state,
          guided=session,
          current_step=current_step,
          current_turn_type=current_turn_type,
          turn_response=turn_response,
          catalog=catalog,
          recorder=BufferingRecorder(),
          user_id="u1",
          data_dir=None,
          session_engine=None,
          session_id="s1",
          blob_service=None,
          model="m",
          temperature=None,
          seed=None,
      )


  @pytest.mark.asyncio
  async def test_wire_confirm_completes_a_wire_ready_session() -> None:
      # The accept handler already left the session at STEP_4_WIRE (terminal=None);
      # a CONFIRM_WIRING confirm must dispatch to handle_step_4_wire_confirm and
      # stamp COMPLETED on the valid pipeline.
      state, session, catalog = _wire_ready_session_and_state()
      _s2, guided2, _t2 = await _dispatch(
          state,
          session,
          catalog,
          current_step=GuidedStep.STEP_4_WIRE,
          current_turn_type=TurnType.CONFIRM_WIRING,
          turn_response={
              "chosen": ["confirm"],
              "edited_values": None,
              "custom_inputs": None,
              "accepted_step_index": None,
              "edit_step_index": None,
              "control_signal": None,
          },
      )
      assert guided2.terminal is not None
      assert guided2.terminal.kind is TerminalKind.COMPLETED


  @pytest.mark.asyncio
  async def test_chain_accept_lands_on_wire_turn_not_none() -> None:
      # Re-stage the proposal on a STEP_3 session and drive the PROPOSE_CHAIN accept
      # through the dispatcher: the dispatcher must now return the wire turn (not None).
      state, wire_session, catalog = _wire_ready_session_and_state()
      # Reconstruct a STEP_3-positioned session carrying the same proposal so the
      # dispatcher's accept path runs (the wire_session is already past STEP_3).
      from dataclasses import replace as _dc_replace

      step3_session = _dc_replace(wire_session, step=GuidedStep.STEP_3_TRANSFORMS, terminal=None)
      _new_state, guided, next_turn = await _dispatch(
          state,
          step3_session,
          catalog,
          current_step=GuidedStep.STEP_3_TRANSFORMS,
          current_turn_type=TurnType.PROPOSE_CHAIN,
          turn_response={
              "chosen": ["accept"],
              "edited_values": None,
              "custom_inputs": None,
              "accepted_step_index": None,
              "edit_step_index": None,
              "control_signal": None,
          },
      )
      # P1.6 left terminal=None, step=STEP_4_WIRE; P2.9 now emits the wire turn.
      assert guided.terminal is None
      assert guided.step is GuidedStep.STEP_4_WIRE
      assert next_turn is not None
      assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
  ```
  > Fixture note: `_wire_ready_session_and_state` mirrors `TestTerminalStampInvariant`
  > in `test_step_handlers.py` (P1.6) — it drives `handle_step_3_chain_accept`, which
  > stages `step_3_proposal` AND (post-P1.6) leaves `step=STEP_4_WIRE, terminal=None`,
  > so no fragile `to_dict`/`from_dict` round-trip is needed. `SourceResolved`,
  > `SinkResolved`, `SinkOutputResolved` are re-exported from
  > `composer/guided/state_machine.py` (NOT `tools/_common`). `create_catalog_service`
  > comes from `elspeth.web.dependencies`. Reuse the P1.6 fixtures; do not re-invent
  > state construction.

- [ ] **Step 3: Run to fail.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py -q
  ```
  Expected failure: `test_wire_confirm_completes_a_wire_ready_session` fails with
  `InvariantError: _dispatch_guided_respond: unhandled branch current_step=STEP_4_WIRE,
  current_turn_type=CONFIRM_WIRING` (no `STEP_4_WIRE` branch yet), and
  `test_chain_accept_lands_on_wire_turn_not_none` fails on `assert next_turn is not None`
  (the accept seams still return `None`).

- [ ] **Step 4: Add a private wire-turn emit helper to keep the three call sites DRY.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, just above `_dispatch_guided_respond`
  (`:2554`), add:
  ```python
  def _emit_wire_turn(
      *,
      state: CompositionState,
      guided: GuidedSession,
      recorder: BufferingRecorder,
      user_id: str,
  ) -> tuple[GuidedSession, Turn]:
      """Emit the STEP_4_WIRE confirm_wiring turn after an accept commit (P2.9).

      P1.6 leaves the accept seams at ``step=STEP_4_WIRE, terminal=None``; this
      builds the wire turn, appends its server TurnRecord, and emits the audit
      ``turn_emitted`` event so the user lands on the wire stage (rather than
      silently falling off the wizard with ``next_turn=None``).
      """
      next_turn = build_step_4_wire_turn(state)
      new_record = TurnRecord(
          step=GuidedStep.STEP_4_WIRE,
          turn_type=TurnType.CONFIRM_WIRING,
          payload_hash=stable_hash(next_turn["payload"]),
          response_hash=None,
          emitter="server",
      )
      emit_turn_emitted(
          recorder,
          step=GuidedStep.STEP_4_WIRE,
          turn_type=TurnType.CONFIRM_WIRING,
          payload_hash=stable_hash(next_turn["payload"]),
          payload_payload_id="",
          emitter="server",
          composition_version=state.version,
          actor=user_id,
      )
      guided = _replace(guided, history=(*guided.history, new_record))
      return guided, next_turn
  ```
  Add `build_step_4_wire_turn` to the existing `from elspeth.web.composer.guided.emitters import (...)` block in `_helpers.py` (the block that already imports `build_step_3_propose_chain_turn` at `:78`). `Turn` is NOT yet imported in `_helpers.py` — add it to the existing `from elspeth.web.composer.guided.protocol import ChatRole, ChatTurn, ControlSignal, GuidedStep, TurnResponse, TurnType` line (`:82`) so the `_emit_wire_turn` return annotation resolves (`from ... import ..., Turn, ...`).

- [ ] **Step 5: Replace the three accept-commit returns with the wire-turn emit.**
  - Recipe-apply (`:3277-3280`): replace
    ```python
            state = handler_result.state
            guided = handler_result.session
            # terminal is now set on guided.terminal (TerminalKind.COMPLETED).
            return state, guided, None
    ```
    with
    ```python
            state = handler_result.state
            guided = handler_result.session
            # P1.6: handler left step=STEP_4_WIRE, terminal=None. P2.9: land on wire.
            guided, next_turn = _emit_wire_turn(state=state, guided=guided, recorder=recorder, user_id=user_id)
            return state, guided, next_turn
    ```
  - Repair-success (`:3518-3520`): replace
    ```python
                    if repair_result.tool_result.success:
                        # Repair succeeded: wizard completes normally.
                        return repair_result.state, repair_result.session, None
    ```
    with
    ```python
                    if repair_result.tool_result.success:
                        # Repair succeeded: P1.6 left step=STEP_4_WIRE; land on wire (P2.9).
                        repaired_guided, next_turn = _emit_wire_turn(
                            state=repair_result.state, guided=repair_result.session, recorder=recorder, user_id=user_id
                        )
                        return repair_result.state, repaired_guided, next_turn
    ```
  - Chain-accept success (`:3553-3554`): replace
    ```python
                # handler_result.session.terminal is COMPLETED on success.
                return handler_result.state, handler_result.session, None
    ```
    with
    ```python
                # P1.6: handler left step=STEP_4_WIRE, terminal=None. P2.9: land on wire.
                accepted_guided, next_turn = _emit_wire_turn(
                    state=handler_result.state, guided=handler_result.session, recorder=recorder, user_id=user_id
                )
                return handler_result.state, accepted_guided, next_turn
    ```

- [ ] **Step 6: Add the `STEP_4_WIRE` / `CONFIRM_WIRING` dispatch branch.**
  In `_dispatch_guided_respond`, AFTER the `if current_step is GuidedStep.STEP_3_TRANSFORMS:`
  block closes (the STEP_3 block ends at `:3655`) and BEFORE the unhandled-branch
  `InvariantError` fall-through (`:3657-3662`), insert:
  ```python
      # --- STEP_4_WIRE turns ----------------------------------------------
      # A CONFIRM_WIRING confirm stamps COMPLETED on a valid pipeline. (P5.6
      # layers the profile-gated advisor sign-off in FRONT of the COMPLETED
      # stamp; P2.9 ships the validate-gate-only confirm.)
      if current_step is GuidedStep.STEP_4_WIRE:
          if current_turn_type is TurnType.CONFIRM_WIRING:
              handler_result = handle_step_4_wire_confirm(state=state, session=guided)
              guided = handler_result.session
              if guided.terminal is not None:
                  # COMPLETED (valid pipeline). No further turn.
                  return handler_result.state, guided, None
              # Invalid: re-emit the wire turn so the user can reconcile (B6).
              guided, next_turn = _emit_wire_turn(
                  state=handler_result.state, guided=guided, recorder=recorder, user_id=user_id
              )
              return handler_result.state, guided, next_turn
          raise HTTPException(
              status_code=400,
              detail=f"STEP_4_WIRE expects a confirm_wiring response; got turn_type={current_turn_type!r}.",
          )
  ```

- [ ] **Step 7: Add the GET `/guided` rebuild branch for `STEP_4_WIRE`.**
  In `src/elspeth/web/sessions/routes/composer.py`, in the `_build_get_guided_turn`
  closure (`:1313`), after the
  `if step is GuidedStep.STEP_3_TRANSFORMS:` block (its `return build_step_3_propose_chain_turn(...)`
  is at `:1417`) and BEFORE the trailing `return None` at `:1421`, insert:
  ```python
          if step is GuidedStep.STEP_4_WIRE:
              # Rebuild the wire turn deterministically from the current state
              # (build_step_4_wire_turn is pure — topology + validate() overlay).
              return build_step_4_wire_turn(state)
  ```
  Add `build_step_4_wire_turn` to the existing emitter import block in `composer.py`
  (the block that imports `build_step_3_propose_chain_turn` at `:118`).

- [ ] **Step 8: Run to pass.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py -q
  ```
  Expected: all pass — accept lands on the wire turn; the confirm stamps COMPLETED.

- [ ] **Step 9: Un-xfail the P1.6 route-level COMPLETED assertions.**
  The P1.6 Step-8 note marked any route-level "accept → COMPLETED" assertion in
  `test_auto_drop.py` as `xfail(reason="wire dispatch lands in P2.9")`. With the wire
  dispatch present, update those to the two-hop flow (accept → wire turn → confirm →
  COMPLETED) and remove the xfail. Run:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/guided/test_auto_drop.py tests/unit/web/sessions/routes/ -q -k "guided or wire or auto_drop"
  ```
  Expected: all pass (no remaining xfail referencing the wire dispatch).

- [ ] **Step 10: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py tests/integration/web/composer/guided/test_wire_dispatch.py tests/integration/web/composer/guided/test_auto_drop.py && git commit -m "feat(web/sessions): emit wire turn after accept + add CONFIRM_WIRING dispatch + GET rebuild (P2.9)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P2.10: Phase gate sweep

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: ruff check.**
  `uv run ruff check src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py tests/unit/web/composer/guided/test_wire_payload.py tests/integration/web/composer/guided/test_wire_dispatch.py`
  Expected: `All checks passed!`.
- [ ] **Step 2: ruff format check.**
  `uv run ruff format --check src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py`
  Expected: `... files already formatted`.
- [ ] **Step 3: mypy.**
  `uv run mypy src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py`
  Expected: `Success: no issues found`.
- [ ] **Step 4: SlotType / guided.ts mirror gate.**
  `uv run python scripts/cicd/check_slot_type_cross_language.py`
  Expected: exit 0.
- [ ] **Step 5: Frontend gates (from `src/elspeth/web/frontend`).**
  `npm run typecheck && npm test -- --run src/types/guided.test.ts src/components/chat/guided/WireStageTurn.test.tsx && npm run build`
  Expected: typecheck clean, vitest green, build succeeds.
- [ ] **Step 6: Targeted backend pytest.**
  `uv run pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -q`
  Expected: all pass (no regression in the guided suite from the new payload shape or wire dispatch).
- [ ] **Step 7: Commit (if any formatter touched files).**
  `git add -A && git commit -m "chore(guided): P2 wire-data gate sweep clean (P2.10)" --allow-empty`

---

## Phase P3 — B1 interpretation surfacing (all 5 kinds) + frontend projection

> **Scope (spec §5 B1, D12, §7).** Fix the latent silent-orphan bug: the
> freeform fail-closed orphan gate
> (`_missing_pending_interpretation_review_sites`, `composer/service.py:1376`)
> is **unreachable** from the guided dispatch path. The guided commit handlers
> (`steps.py`) call `_execute_*` tools directly and never traverse any of the
> freeform finalize/checkpoint call sites that run the gate, so a guided step-3
> LLM node — or the deterministic recipe-apply — creates real interpretation
> sites that are surfaced to **no one** and only fail at run time with
> `UnresolvedInterpretationPlaceholderError` (`execution/service.py:514-525`).
> P3 adds a **kind-general** backend surfacing pass that fires after **every**
> site-creating guided commit (source, transform, recipe-apply), covering all
> five `InterpretationKind` members, plus the frontend projection that renders
> pending events in the guided ChatPanel branch and blocks advancement while
> any remain. Polarity: **surface-and-resolve (advisory)** at commit; the
> run-time gate stays the hard backstop.
>
> **Cross-phase dependencies (symbols owned elsewhere):**
> - `handle_step_2_5_recipe_apply` / `handle_step_3_chain_accept`
>   (`composer/guided/steps.py:218 / :277`) currently stamp
>   `TerminalKind.COMPLETED`. **P2** moves the terminal-stamp out of both into the
>   `STEP_4_WIRE` handler so each leaves `session.terminal=None` /
>   `session.step==STEP_4_WIRE`. P3 does **not** depend on that move having
>   landed — P3 hooks the surfacing pass at the route persistence seam
>   (`post_guided_respond`, `composer.py:2401`) which runs on the
>   recipe-apply/chain-accept path **regardless of which step the session lands
>   on**. If P2 lands first, the surfacer still fires (the persist seam is
>   unchanged). If P3 lands first, the surfacer fires on the COMPLETED state,
>   which is correct (the sites are surfaced before the user can run). The two
>   phases are independent at this seam.
> - **P6** owns `run_signoff_checkpoint` and the `_stub_advisor_end_gate_clean`
>   autouse helper (`tests/unit/web/composer/_helpers.py:86`). P3 tests do **not**
>   reach the advisor gate (the surfacer runs at commit, before any STEP_4_WIRE
>   advisor call), so P3 does not consume P6 symbols.

---

### Task P3.1: Add the kind-general `_surface_pending_interpretation_reviews` backend surfacer

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add a method on
  `ComposerServiceImpl` immediately after `_auto_surface_prompt_template_reviews`,
  which currently spans `:1412-1489`)
- Create: `tests/unit/web/composer/test_surface_pending_interpretation_reviews.py`

**Interfaces:**
- Consumes: `interpretation_sites(state: CompositionState) -> tuple[InterpretationReviewSite, ...]`
  (`web/interpretation_state.py:366`); `InterpretationReviewSite` fields
  `component_id: str`, `component_type: str`, `user_term: str`,
  `kind: InterpretationKind`; `InterpretationKind` members `VAGUE_TERM`,
  `INVENTED_SOURCE`, `LLM_PROMPT_TEMPLATE`, `PIPELINE_DECISION`,
  `LLM_MODEL_CHOICE` (`contracts/composer_interpretation.py:74-85`);
  `SessionServiceImpl.create_pending_interpretation_event(*, session_id: UUID,
  composition_state_id: UUID, affected_node_id: str, tool_call_id: str,
  user_term: str, kind: InterpretationKind, llm_draft: str,
  model_identifier: str, model_version: str, provider: str,
  composer_skill_hash: str)` (`sessions/service.py:2717`);
  `SessionServiceImpl.list_interpretation_events(session_id, status=...)`;
  `self._require_sessions_service()`, `self._model`, `self._availability`,
  `self._composer_skill_hash` (all already used by
  `_auto_surface_prompt_template_reviews`).
- Produces (NEW, owned by P3, consumed by P3.2/P3.3/P3.4):
  `async def _surface_pending_interpretation_reviews(self, state: CompositionState, *, session_id: str | None, current_state_id: str | None) -> None`

**Writer-boundary facts (verified, drive the per-kind draft/user_term mapping):**
`create_pending_interpretation_event` (`sessions/service.py:2717`) is strict per
kind — it re-reads the persisted parent state inside the locked transaction and
**raises `ValueError`** unless the requirement shape matches exactly:
- `INVENTED_SOURCE` (`:2794`): `affected_node_id` must equal `SOURCE_COMPONENT_ID`;
  the source must carry a `SOURCE_AUTHORING_KEY` block with a non-empty
  `content_hash`; exactly one pending `invented_source` requirement matching
  `user_term`; `llm_draft` must equal that requirement's `draft`.
- `LLM_PROMPT_TEMPLATE` (`:2889`): node's `options.prompt_template` must be a
  string and `llm_draft` must equal it; exactly one pending PT requirement.
- `LLM_MODEL_CHOICE` (`:2876` else-branch): goes through `_find_llm_transform_node`;
  the `_options_with_default_model_choice_review` auto-stager
  (`composer/tools/_common.py:202`) stages the requirement with
  `user_term=f"llm_model_choice:{node_id}"` and `draft=options.model`.
- `PIPELINE_DECISION` (`:2833`): `_find_interpretation_review_node` + exactly
  one pending requirement + `validate_pipeline_decision_semantics`; `llm_draft`
  must equal the requirement's `draft`.
- `VAGUE_TERM`: legacy placeholder sites (`_legacy_placeholder_sites`,
  `interpretation_state.py:630`) carry **no** staged requirement, so the writer
  boundary would reject them; the surfacer therefore mirrors
  `_auto_surface_prompt_template_reviews`'s precondition discipline and **skips
  any site that does not have a matching pending requirement on the node/source**
  (those stay fail-closed at the run-time gate, which is the designed polarity).

Because of these per-kind preconditions, the surfacer is built as a per-kind
dispatch that reads the site's own `draft`/`user_term` from the node/source
requirement (NOT from a synthesized value), then calls the writer with the exact
matching draft. This reuses `_auto_surface_prompt_template_reviews`'s honest
provenance sentinel (`tool_call_id="backend_auto_surface:<uuid4>"`,
`model_identifier=model_version=self._model`).

- [ ] **Step 1: Write the failing test for the new method's existence + the
  prompt_template path (parity with the existing surfacer).**
  Create `tests/unit/web/composer/test_surface_pending_interpretation_reviews.py`.
  Reuse the dispatch-test scaffolding (`_build_composer`, the `engine` /
  `sessions_service` fixtures, `_state_with_prompt_template_review_node`) by
  importing the helpers directly:

  ```python
  from __future__ import annotations

  from uuid import UUID, uuid4

  import pytest
  import structlog
  from sqlalchemy.pool import StaticPool

  from elspeth.contracts.composer_interpretation import InterpretationKind
  from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      PipelineMetadata,
      SourceSpec,
  )
  from elspeth.web.config import WebSettings
  from elspeth.web.interpretation_state import (
      INTERPRETATION_REQUIREMENTS_KEY,
      SOURCE_AUTHORING_KEY,
      SOURCE_COMPONENT_ID,
  )
  from elspeth.web.sessions.engine import create_session_engine
  from elspeth.web.sessions.protocol import CompositionStateData
  from elspeth.web.sessions.schema import initialize_session_schema
  from elspeth.web.sessions.service import SessionServiceImpl
  from elspeth.web.sessions.telemetry import build_sessions_telemetry


  @pytest.fixture
  def engine():
      eng = create_session_engine(
          "sqlite:///:memory:",
          connect_args={"check_same_thread": False},
          poolclass=StaticPool,
      )
      initialize_session_schema(eng)
      return eng


  @pytest.fixture
  def sessions_service(engine) -> SessionServiceImpl:
      return SessionServiceImpl(
          engine,
          telemetry=build_sessions_telemetry(),
          log=structlog.get_logger("test.sessions"),
      )


  @pytest.fixture(autouse=True)
  def _force_available(monkeypatch: pytest.MonkeyPatch) -> None:
      def _available(self: ComposerServiceImpl) -> ComposerAvailability:
          return ComposerAvailability(available=True, model=self._model, provider="anthropic")

      monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


  def _composer(tmp_path, sessions_service) -> ComposerServiceImpl:
      from unittest.mock import MagicMock

      from elspeth.web.catalog.protocol import CatalogService

      catalog = MagicMock(spec=CatalogService)
      catalog.list_sources.return_value = []
      catalog.list_transforms.return_value = []
      catalog.list_sinks.return_value = []
      settings = WebSettings(
          data_dir=tmp_path,
          composer_model="anthropic/claude-opus-4-7",
          shareable_link_signing_key=b"\x00" * 32,
      )
      return ComposerServiceImpl(
          catalog=catalog,
          settings=settings,
          sessions_service=sessions_service,
          session_engine=sessions_service._engine,
      )


  def _pt_node() -> NodeSpec:
      prompt = "Read {{ row.html }} and return JSON."
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={
              "prompt_template": prompt,
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "prompt_template_review",
                      "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                      "user_term": "llm_prompt_template:rate_node",
                      "status": "pending",
                      "draft": prompt,
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  }
              ],
          },
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  async def _persist(sessions_service, state: CompositionState):
      from datetime import UTC, datetime
      from uuid import uuid4

      from sqlalchemy import insert

      from elspeth.web.sessions.models import sessions_table

      session_id = uuid4()
      with sessions_service._engine.begin() as conn:
          conn.execute(
              insert(sessions_table).values(
                  id=str(session_id),
                  user_id="u",
                  auth_provider_type="local",
                  title="surfacer test",
                  created_at=datetime.now(UTC),
                  updated_at=datetime.now(UTC),
              )
          )
      state_dict = state.to_dict()
      record = await sessions_service.save_composition_state(
          session_id,
          CompositionStateData(
              nodes=state_dict["nodes"],
              sources=state_dict["sources"],
              metadata_=state_dict["metadata"],
              is_valid=True,
          ),
          provenance="tool_call",
      )
      return session_id, record.id


  @pytest.mark.asyncio
  async def test_surfacer_surfaces_prompt_template(tmp_path, sessions_service) -> None:
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(_pt_node(),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.LLM_PROMPT_TEMPLATE in kinds
  ```

  Run to fail:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
  ```
  Expected failure: `AttributeError: 'ComposerServiceImpl' object has no attribute '_surface_pending_interpretation_reviews'`.

- [ ] **Step 2: Add the method (minimal — delegate prompt_template to the
  existing surfacer, then handle model_choice).**
  In `src/elspeth/web/composer/service.py`, immediately after
  `_auto_surface_prompt_template_reviews` (after its closing line `:1489`), add:

  ```python
      async def _surface_pending_interpretation_reviews(
          self,
          state: CompositionState,
          *,
          session_id: str | None,
          current_state_id: str | None,
      ) -> None:
          """Kind-general backend surfacer for the GUIDED commit path (B1).

          The freeform fail-closed orphan gate
          (:meth:`_missing_pending_interpretation_review_sites`) is unreachable
          from the guided dispatcher, so guided commits that create
          interpretation sites would otherwise orphan and only fail at run
          time with ``UnresolvedInterpretationPlaceholderError``. This pass runs
          after every site-creating guided commit (source / transform /
          recipe-apply) and surfaces a resolvable pending EVENT for every site
          whose writer-boundary precondition holds — covering all five
          ``InterpretationKind`` members, not just ``llm_prompt_template``.

          Each branch reads the site's ``draft``/``user_term`` from the node or
          source requirement so the strict per-kind writer boundary
          (``create_pending_interpretation_event``) accepts the insert; a site
          with no matching pending requirement (e.g. a bare legacy vague-term
          token) is SKIPPED and left fail-closed at the run-time gate, the
          designed advisory polarity (spec §5 B1).

          Honest provenance: the sentinel ``tool_call_id="backend_auto_surface:..."``
          records that no LLM tool call produced the event; the user still
          reviews it, so ``interpretation_source`` stays ``user_approved``.
          Idempotent and a no-op when there is no session/persisted state.
          """

          if session_id is None or current_state_id is None:
              return
          # llm_prompt_template is already handled by the existing surfacer,
          # which carries the exact draft-aware dedup the writer boundary needs.
          await self._auto_surface_prompt_template_reviews(
              state,
              session_id=session_id,
              current_state_id=current_state_id,
          )
          sessions_service = self._require_sessions_service()
          events = await sessions_service.list_interpretation_events(UUID(session_id), status="pending")
          for site in interpretation_sites(state):
              if site.kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
                  continue  # handled above
              surfaced = self._backend_surface_args_for_site(state, site)
              if surfaced is None:
                  continue
              affected_node_id, user_term, llm_draft = surfaced
              if any(
                  event.affected_node_id == affected_node_id
                  and event.user_term is not None
                  and event.user_term.strip() == user_term.strip()
                  and event.kind is site.kind
                  for event in events
              ):
                  continue
              await sessions_service.create_pending_interpretation_event(
                  session_id=UUID(session_id),
                  composition_state_id=UUID(current_state_id),
                  affected_node_id=affected_node_id,
                  tool_call_id=f"backend_auto_surface:{uuid4()}",
                  user_term=user_term,
                  kind=site.kind,
                  llm_draft=llm_draft,
                  model_identifier=self._model,
                  model_version=self._model,
                  provider=self._availability.provider or "unknown",
                  composer_skill_hash=self._composer_skill_hash,
              )

      def _backend_surface_args_for_site(
          self,
          state: CompositionState,
          site: InterpretationReviewSite,
      ) -> tuple[str, str, str] | None:
          """Return ``(affected_node_id, user_term, llm_draft)`` for a site, or
          ``None`` when the writer-boundary precondition does not hold.

          Reads the draft straight from the node/source pending requirement so
          the strict ``create_pending_interpretation_event`` writer boundary
          accepts the insert. ``None`` means "no matching pending requirement" —
          the site is left for the run-time gate (designed advisory polarity).
          """

          if site.kind is InterpretationKind.INVENTED_SOURCE:
              source = state.sources[SOURCE_COMPONENT_ID] if SOURCE_COMPONENT_ID in state.sources else None
              if source is None:
                  return None
              options = source.options if isinstance(source.options, Mapping) else {}
              if SOURCE_AUTHORING_KEY not in options:
                  return None
              draft = self._matching_requirement_draft(options, kind=site.kind, user_term=site.user_term)
              if draft is None:
                  return None
              return (SOURCE_COMPONENT_ID, site.user_term, draft)

          node = next((candidate for candidate in state.nodes if candidate.id == site.component_id), None)
          if node is None:
              return None
          options = node.options if isinstance(node.options, Mapping) else {}
          if site.kind is InterpretationKind.LLM_MODEL_CHOICE:
              model = options.get("model")
              if not isinstance(model, str) or not model:
                  return None
              draft = self._matching_requirement_draft(options, kind=site.kind, user_term=site.user_term)
              if draft is None or draft != model:
                  return None
              return (node.id, site.user_term, draft)
          if site.kind is InterpretationKind.PIPELINE_DECISION:
              draft = self._matching_requirement_draft(options, kind=site.kind, user_term=site.user_term)
              if draft is None:
                  return None
              return (node.id, site.user_term, draft)
          # VAGUE_TERM legacy placeholder sites carry no staged requirement;
          # leave them fail-closed at the run-time gate.
          return None

      @staticmethod
      def _matching_requirement_draft(
          options: Mapping[str, Any],
          *,
          kind: InterpretationKind,
          user_term: str,
      ) -> str | None:
          """Return the ``draft`` of the single pending requirement matching
          ``(kind, user_term)``, or ``None`` when there is not exactly one."""

          raw = options.get(INTERPRETATION_REQUIREMENTS_KEY)
          if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
              return None
          matches: list[str] = []
          for requirement in raw:
              if not isinstance(requirement, Mapping):
                  continue
              if requirement.get("kind") != kind.value:
                  continue
              if requirement.get("status") != "pending":
                  continue
              requirement_term = requirement.get("user_term")
              if not isinstance(requirement_term, str) or requirement_term.strip() != user_term.strip():
                  continue
              draft = requirement.get("draft")
              if isinstance(draft, str):
                  matches.append(draft)
          return matches[0] if len(matches) == 1 else None
  ```

  **Add the missing imports.** Verified: `service.py:21` already imports
  `Mapping, Sequence` (`from collections.abc import`), `:24` already imports
  `Any`, and the `from elspeth.web.interpretation_state import (...)` block
  (`:131-140`) already imports `INTERPRETATION_REQUIREMENTS_KEY` and
  `interpretation_sites` — but it does **NOT** import `InterpretationReviewSite`,
  `SOURCE_AUTHORING_KEY`, or `SOURCE_COMPONENT_ID` (all three are exported from
  `interpretation_state.py:27/28/101`). Add them to that import block, keeping
  alphabetical order:
  ```python
  from elspeth.web.interpretation_state import (
      INTERPRETATION_REQUIREMENTS_KEY,
      INTERPRETATION_REVIEW_PENDING_CODE,
      PROMPT_SHIELD_USER_TERM,
      PROMPT_SHIELD_WARNING_DRAFT,
      RAW_HTML_CLEANUP_REVIEW_DRAFT,
      RAW_HTML_CLEANUP_USER_TERM,
      SOURCE_AUTHORING_KEY,
      SOURCE_COMPONENT_ID,
      InterpretationReviewSite,
      interpretation_sites,
      vague_term_wiring_count,
  )
  ```
  (`InterpretationReviewSite` sorts after the UPPER_CASE constants under ruff's
  isort profile because lowercase-after-uppercase ordering applies; if
  `ruff check --fix` reorders it, accept the autofix.) Confirm the current block:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && sed -n '131,140p' src/elspeth/web/composer/service.py
  ```

  Run to pass:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_prompt_template -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Add the failing test for the model_choice path.**
  Append to `test_surface_pending_interpretation_reviews.py`:

  ```python
  def _model_choice_node() -> NodeSpec:
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={
              "prompt_template": "Rate this row and return JSON.",
              "model": "anthropic/claude-sonnet-4.6",
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "prompt_template_review:rate_node",
                      "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                      "user_term": "llm_prompt_template:rate_node",
                      "status": "pending",
                      "draft": "Rate this row and return JSON.",
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  },
                  {
                      "id": "model_choice_review:rate_node",
                      "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                      "user_term": "llm_model_choice:rate_node",
                      "status": "pending",
                      "draft": "anthropic/claude-sonnet-4.6",
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  },
              ],
          },
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  @pytest.mark.asyncio
  async def test_surfacer_surfaces_model_choice(tmp_path, sessions_service) -> None:
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(_model_choice_node(),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.LLM_MODEL_CHOICE in kinds
      assert InterpretationKind.LLM_PROMPT_TEMPLATE in kinds
  ```

  Run to verify it passes with the Step 2 impl (the impl already handles
  model_choice — this test pins it):
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
  ```
  Expected: `2 passed`. (If the model_choice assertion fails, the
  `_backend_surface_args_for_site` model_choice branch is the defect — fix it
  there, not in the test.)

- [ ] **Step 4: Add the idempotency + skip-bare-vague-term tests.**
  Append:

  ```python
  @pytest.mark.asyncio
  async def test_surfacer_is_idempotent(tmp_path, sessions_service) -> None:
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(_model_choice_node(),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      mc = [e for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
      assert len(mc) == 1


  @pytest.mark.asyncio
  async def test_surfacer_skips_bare_vague_term(tmp_path, sessions_service) -> None:
      # A bare {{interpretation:cool}} token with NO staged requirement is a
      # legacy vague_term site; the writer boundary would reject it, so the
      # surfacer must SKIP it (left fail-closed at the run-time gate).
      node = NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={"prompt_template": "Rate how {{interpretation:cool}} this is."},
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(node,),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      # Must not raise (the writer boundary would reject a bare vague_term).
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      vt = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
      assert vt == []
  ```

  Run to pass:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
  ```
  Expected: `4 passed`.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): add kind-general interpretation surfacer (B1)

Add ComposerServiceImpl._surface_pending_interpretation_reviews, a
backend surfacing pass covering all five InterpretationKind members for
the guided commit path. The freeform fail-closed orphan gate is
unreachable from the guided dispatcher, so guided commits that create
interpretation sites orphan and only fail at run time. The surfacer
reads each site's draft/user_term from the node/source pending
requirement so the strict create_pending_interpretation_event writer
boundary accepts the insert; sites with no matching requirement (bare
legacy vague_term tokens) are skipped and left fail-closed at the
run-time gate (designed advisory polarity, spec §5 B1).

Not yet wired into the dispatcher (next commits fire it after source,
transform, and recipe-apply commits).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.2: Fire the surfacer after every guided commit at the route persistence seam

**Files:**
- Modify: `src/elspeth/web/sessions/routes/composer.py` (`post_guided_respond`,
  the persistence block at `:2383-2410`)
- Modify: `tests/integration/web/composer/guided/conftest.py` (wire a real
  `ComposerServiceImpl` onto `app.state.composer_service`, replacing the `None` at
  `:97` — see Step 0)
- Create: `tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py`

**Interfaces:**
- Consumes: `ComposerServiceImpl._surface_pending_interpretation_reviews`
  (P3.1), reached via `request.app.state.composer_service` (a `ComposerService`
  handle; the recompose path already reads it the same way at `composer.py:652`),
  None-guarded; `service.save_composition_state(...) -> state_record_out` with
  `state_record_out.id` (`composer.py:2401`) — here `service` is the route's
  `session_service` (`composer.py:1949`), which exposes `save_composition_state`
  but NOT the surfacer (hence the separate composer handle); `new_state` (the
  post-dispatch state, `composer.py:2384`); `session_id` (the route path param).
- Produces: nothing new — wires the existing surfacer into the route.

**Why here:** the dispatcher (`_dispatch_guided_respond`, `_helpers.py:2554`) is
a pure routing function with **no service handle**. The route is the only place
that holds `service` and the freshly-persisted `state_record_out.id` (the
`current_state_id` the surfacer needs). The persist at `:2401` runs after every
guided commit including recipe-apply (`:3263`) and chain-accept (`:3450`); the
surfacer reads `interpretation_sites(state)` on the post-mutation `new_state`
and surfaces the delta. This is the single hook that covers source, transform,
and recipe-apply commits (P3.3/P3.4 are the per-boundary assertions).

- [ ] **Step 1: Read the exact persistence block to anchor the edit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && sed -n '2381,2412p' src/elspeth/web/sessions/routes/composer.py
  ```
  Confirm `state_record_out = await service.save_composition_state(...)` ends at
  `:2410` and `new_state` is the variable holding the post-dispatch state
  (`:2384`).

- [ ] **Step 2: Write the failing integration test that a guided
  transform-commit surfaces the model_choice + prompt_template cards through the
  route.**
  Create the test UNDER `tests/integration/web/composer/guided/` so it inherits the
  canonical `composer_test_client` fixture from
  `tests/integration/web/composer/guided/conftest.py` (a `SyncASGITestClient`
  wrapping a FastAPI app with the session router mounted, auth mocked as "alice",
  `app.state.session_service`/`session_engine`/`blob_service`/`catalog_service`
  populated). File:
  `tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py`.
  Mirror the request helpers in the sibling `test_respond.py` (`_create_session`,
  `_get_guided`, `_respond`, `_seed_blob`). The interpretations are read over HTTP
  via `GET /api/sessions/{id}/interpretations?status=pending`
  (`sessions/routes/interpretation.py:194`, query param
  `status: Literal["pending", "all"] = "all"`), so no internal-service call is
  needed. The chain proposal is a single `llm` node carrying a `model` +
  `prompt_template`, so the commit auto-stages `llm_model_choice` +
  `llm_prompt_template` requirements (`composer/tools/_common.py:184/202`) and the
  surfacer materialises both as pending events. The source/sink drive bodies are
  copied verbatim from `test_step_3_e2e.py::_drive_to_step_3_propose_chain`
  (`tests/integration/web/composer/guided/test_step_3_e2e.py:107-156`); only the
  chain-solver stub differs (it proposes an `llm` node, not `passthrough`):
  ```python
  """Phase P3.2 — a guided chain-accept commit surfaces interpretation cards via the route."""

  from __future__ import annotations

  import asyncio
  import json
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import AsyncMock, patch
  from uuid import UUID

  from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


  def _create_session(client: TestClient) -> str:
      resp = client.post("/api/sessions", json={"title": "surface-test"})
      assert resp.status_code == 201, resp.json()
      return resp.json()["id"]


  def _get_guided(client: TestClient, session_id: str) -> dict:
      resp = client.get(f"/api/sessions/{session_id}/guided")
      assert resp.status_code == 200, resp.json()
      return resp.json()


  def _respond(client: TestClient, session_id: str, **kwargs) -> dict:
      resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
      assert resp.status_code == 200, resp.json()
      return resp.json()


  def _seed_blob(client: TestClient, session_id: str) -> tuple[str, str]:
      content = "text,note\nHello world,greeting\nGoodbye,farewell\n"
      resp = client.post(
          f"/api/sessions/{session_id}/blobs/inline",
          json={"filename": "data.csv", "content": content, "mime_type": "text/csv"},
      )
      assert resp.status_code == 201, resp.json()
      blob_id = resp.json()["id"]
      record = asyncio.run(client.app.state.blob_service.get_blob(UUID(blob_id)))
      return blob_id, record.storage_path


  def _outputs_path(client: TestClient, filename: str) -> str:
      data_dir: Path = client.app.state.settings.data_dir
      outputs_dir = data_dir / "outputs"
      outputs_dir.mkdir(parents=True, exist_ok=True)
      return str(outputs_dir / filename)


  def _fake_llm_chain_response() -> SimpleNamespace:
      """A LiteLLM-shaped propose_chain response carrying a single `llm` node.

      The node sets BOTH `model` and `prompt_template` so the accept commit
      auto-stages an llm_model_choice AND an llm_prompt_template requirement
      (composer/tools/_common.py:184/202), which the surfacer then materialises
      as pending interpretation events.
      """
      return SimpleNamespace(
          choices=[
              SimpleNamespace(
                  message=SimpleNamespace(
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
                                                      "plugin": "llm",
                                                      "options": {
                                                          "provider": "openrouter",
                                                          "model": "anthropic/claude-sonnet-4.6",
                                                          "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                                                          "prompt_template": "Summarize {{ row.text }} and return JSON.",
                                                          "schema": {"mode": "observed"},
                                                      },
                                                      "rationale": "summarise each row with an llm transform",
                                                  }
                                              ],
                                              "why": "source rows need an llm summary before the sink",
                                              "blockers": [],
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


  def _drive_to_step_3_propose_chain(client: TestClient, session_id: str) -> str:
      """Drive source -> sink -> propose_chain (verbatim from test_step_3_e2e)."""
      _blob_id, storage_path = _seed_blob(client, session_id)
      output_path = _outputs_path(client, "out.jsonl")

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
      _respond(client, session_id, chosen=["json"])
      _respond(
          client,
          session_id,
          edited_values={
              "plugin": "json",
              "options": {
                  "path": output_path,
                  "schema": {"mode": "observed"},
                  "mode": "write",
                  "collision_policy": "auto_increment",
              },
              "observed_columns": [],
              "sample_rows": [],
          },
      )
      # No classifier keyword, single output -> no recipe match -> chain solver fires.
      body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
      assert body["next_turn"]["type"] == "propose_chain"
      return session_id


  def test_chain_accept_commit_surfaces_model_and_template(composer_test_client: TestClient) -> None:
      client = composer_test_client
      session_id = _create_session(client)
      with patch(
          "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
          new_callable=AsyncMock,
          return_value=_fake_llm_chain_response(),
      ):
          _drive_to_step_3_propose_chain(client, session_id)
          # Accept the llm-node chain: handle_step_3_chain_accept commits via
          # _execute_set_pipeline; the route's persist seam then fires the surfacer.
          _respond(client, session_id, chosen=["accept"])
      resp = client.get(f"/api/sessions/{session_id}/interpretations?status=pending")
      assert resp.status_code == 200, resp.json()
      kinds = {row["kind"] for row in resp.json()["events"]}
      assert "llm_model_choice" in kinds
      assert "llm_prompt_template" in kinds
  ```
  > The source/sink drive bodies are copied verbatim from
  > `test_step_3_e2e.py::_drive_to_step_3_propose_chain`; only the chain-solver
  > stub proposes an `llm` node (so the two sites are created) instead of
  > `passthrough`. The single load-bearing addition is the
  > `GET /interpretations?status=pending` assertion after the accept.

  > Service-handle resolution (applied in Step 3): the conftest now wires a real
  > `ComposerServiceImpl` onto `app.state.composer_service` (see the conftest
  > change in Step 0 below). `post_guided_respond` reads that handle and calls
  > `_surface_pending_interpretation_reviews` on it, None-guarded — surfacing is
  > advisory; the run-time `UnresolvedInterpretationPlaceholderError` gate
  > (`execution/service.py:514-525`) stays the hard backstop, so skipping when no
  > composer is wired is safe.

- [ ] **Step 0: Wire a real `ComposerServiceImpl` onto `app.state.composer_service`
  in the guided conftest.**
  `tests/integration/web/composer/guided/conftest.py:97` currently sets
  `app.state.composer_service = None` ("Not used in session router"). The B1
  surfacer (P3.1) is a `ComposerServiceImpl` method that `post_guided_respond`
  invokes through that slot, so the integration test must exercise the real method
  rather than `None`. Add the construction to `composer_test_client` (mirror
  `test_progressive_disclosure.py:130-135`, which already builds the impl over the
  same in-memory engine + catalog the conftest creates). Import at the top of the
  conftest:
  ```python
  from elspeth.web.composer.service import ComposerServiceImpl
  ```
  Then, just before the `app = FastAPI()` line (`conftest.py:75`), build the impl
  from the already-constructed `session_service`, `engine`, and the `WebSettings`
  the fixture passes to `app.state.settings`; lift that `WebSettings(...)` into a
  local `settings` so both the service and `app.state.settings` share it:
  ```python
      catalog_service = create_catalog_service()
      settings = WebSettings(
          data_dir=tmp_path,
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          shareable_link_signing_key=b"\x00" * 32,
      )
      composer_service = ComposerServiceImpl(
          catalog=catalog_service,
          settings=settings,
          sessions_service=session_service,
          session_engine=engine,
      )
  ```
  Replace `app.state.settings = WebSettings(...)` (`:89-96`) with
  `app.state.settings = settings`, replace `app.state.composer_service = None`
  (`:97`) with `app.state.composer_service = composer_service`, and replace
  `app.state.catalog_service = create_catalog_service()` (`:99`) with
  `app.state.catalog_service = catalog_service` so the catalog the surfacer/route
  use is the same instance. (The conftest already imports `create_catalog_service`
  at `:28` and `WebSettings` at `:27`.)

  Run to fail:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py -x -q
  ```
  Expected failure: the pending-event assertion fails (`kinds` is empty / missing
  the two kinds) because the route does not yet call the surfacer.

- [ ] **Step 3: Add the surfacer call after the persist, through the composer handle.**
  The B1 surfacer (`_surface_pending_interpretation_reviews`) is a
  `ComposerServiceImpl` method. The route's local `service` is
  `request.app.state.session_service` (a `SessionService`,
  `composer.py:1949`), which does NOT carry the surfacer. The route DOES hold a
  `ComposerService` handle in the `app.state.composer_service` slot — the same
  pattern the recompose path already uses
  (`composer: ComposerService = request.app.state.composer_service`,
  `composer.py:652`). In
  `src/elspeth/web/sessions/routes/composer.py`, immediately after the
  `state_record_out = await service.save_composition_state(...)` call (ends
  `:2410`), insert:

  ```python
                # B1 (spec §5): the guided dispatch path never reaches the
                # freeform fail-closed orphan gate, so a committed source /
                # transform / recipe-apply that creates interpretation sites
                # would orphan and only fail at run time. Surface every
                # resolvable pending review against the freshly-persisted state
                # so the guided UI can project + block on it (D12). Advisory
                # polarity: the run-time gate (execution/service.py:514-525)
                # stays the hard backstop, so a None composer (no impl wired in
                # this app) safely skips surfacing.
                composer: ComposerService = request.app.state.composer_service
                if composer is not None:
                    await composer._surface_pending_interpretation_reviews(
                        new_state,
                        session_id=str(session_id),
                        current_state_id=str(state_record_out.id),
                    )
  ```

  Ensure `ComposerService` is importable for the annotation — `composer.py`
  already imports it for the recompose path (`:652`); if the surfacer block is in
  a different module scope, add `from elspeth.web.composer.protocol import
  ComposerService` to the existing composer-imports block. Do NOT widen the call
  to `getattr`. (The Protocol itself gains the method in Step 3b, which mypy
  requires because the call is on a `ComposerService`-typed handle.)

- [ ] **Step 3b: Add `_surface_pending_interpretation_reviews` to the
  `ComposerService` Protocol (mypy gate, §9.2).**
  The Step 3 call is on `composer: ComposerService` (the Protocol type, not the
  concrete impl), so mypy — a §9.2 gate — errors with `"ComposerService" has no
  attribute "_surface_pending_interpretation_reviews"` until the method is declared
  on the Protocol. Add it, mirroring how P5.1 Step 4 adds `run_signoff_checkpoint`
  to the same Protocol. In `src/elspeth/web/composer/protocol.py`, immediately
  AFTER the `compose(...)` method's closing docstring/`"""` (`:759`) and BEFORE
  `async def explain_run_diagnostics` (`:761`), insert the declaration — the
  signature is identical to P3.1's `ComposerServiceImpl` method definition:
  ```python
      async def _surface_pending_interpretation_reviews(
          self,
          state: CompositionState,
          *,
          session_id: str | None,
          current_state_id: str | None,
      ) -> None:
          """Kind-general backend surfacer for the GUIDED commit path (B1).

          Surfaces a resolvable pending interpretation EVENT for every
          interpretation site on ``state`` whose writer-boundary precondition
          holds (all five ``InterpretationKind`` members). Called by the guided
          route persistence seam (``post_guided_respond``) after every committed
          source / transform / recipe-apply, because the guided dispatch path
          never reaches the freeform fail-closed orphan gate. Advisory polarity:
          the run-time ``UnresolvedInterpretationPlaceholderError`` gate stays the
          hard backstop. Idempotent; a no-op when there is no session/persisted
          state. See P3.1 for the concrete implementation.
          """
          ...
  ```
  `CompositionState` is already imported at the top of `protocol.py` (it is the
  type of `compose`'s `state` param, `:717`), so no new import is needed.
  Add `src/elspeth/web/composer/protocol.py` to the Step 5 commit (it is already
  in the `git add` list).

  Run to pass + typecheck:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py -x -q && uv run mypy src/elspeth/web/sessions/routes/composer.py src/elspeth/web/composer/protocol.py
  ```
  Expected: `1 passed`, then `Success: no issues found`.

- [ ] **Step 4: Run the guided-respond route regression suite (no behaviour
  regression on existing paths).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/sessions/routes/ tests/integration/web/composer/ -q -k "guided or respond"
  ```
  Expected: all pass (the surfacer is a no-op when there are no pending
  requirements, so existing source/sink-only guided tests are unaffected).

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/sessions/routes/composer.py src/elspeth/web/composer/protocol.py tests/integration/web/composer/guided/conftest.py tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): fire interpretation surfacer at guided persist seam (B1)

Wire _surface_pending_interpretation_reviews into post_guided_respond
right after save_composition_state, using the freshly-persisted state id
as current_state_id. This is the single hook that covers source,
transform, and recipe-apply commits — the dispatcher is a pure routing
function with no service handle, so the route is the only place holding
both the service and the new state id.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.3: Per-boundary assertion — source commit surfaces `invented_source`

**Files:**
- Modify: `tests/unit/web/composer/test_surface_pending_interpretation_reviews.py`

**Interfaces:**
- Consumes: `_pending_source_sites` proves an LLM-authored source produces an
  `invented_source` site (`interpretation_state.py:524-555`); the writer
  boundary's invented_source branch (`sessions/service.py:2794`) requires
  `SOURCE_AUTHORING_KEY.content_hash` + exactly one pending requirement whose
  `draft == llm_draft`.

**Why a unit boundary test (not route):** the spec's load-bearing requirement is
"a source CAN produce a site" — proving the surfacer's `INVENTED_SOURCE` branch
satisfies the strict writer boundary. The route already fires the surfacer
(P3.2); this pins the source-kind path against the writer boundary directly,
which is the part most likely to drift.

- [ ] **Step 1: Determine the exact SourceSpec shape the writer boundary
  accepts.**
  Read the authoring-metadata + requirement shape the source path expects:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "SOURCE_AUTHORING_KEY\|content_hash\|modality\|_is_llm_authored_modality\|_source_authoring_metadata\|invented_source" src/elspeth/web/interpretation_state.py | head -25
  ```
  Read `_source_authoring_metadata` and `_pending_source_sites`
  (`interpretation_state.py:524`) to confirm the `options[SOURCE_AUTHORING_KEY]`
  shape (`modality`, `content_hash`) and the `interpretation_requirements`
  invented_source requirement shape (`user_term`, `draft`, `status="pending"`).

- [ ] **Step 2: Write the failing test.**
  Append to `test_surface_pending_interpretation_reviews.py` (fill the authoring
  block + requirement to match what Step 1 read — use the verbatim field names
  `modality`, `content_hash`, `user_term`, `draft`):

  ```python
  def _llm_authored_source() -> SourceSpec:
      content_hash = "a" * 64
      return SourceSpec(
          plugin="inline_blob",
          options={
              # SOURCE_AUTHORING_KEY block — modality must be LLM-authored so
              # _pending_source_sites yields an invented_source site, and
              # content_hash must be populated for the writer boundary.
              SOURCE_AUTHORING_KEY: {
                  "modality": "llm_generated",
                  "content_hash": content_hash,
              },
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "invented_source_review",
                      "kind": InterpretationKind.INVENTED_SOURCE.value,
                      "user_term": "llm_generated_source",
                      "status": "pending",
                      "draft": "rows: [{url: https://example.gov}]",
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  }
              ],
          },
          on_success="main",
          on_validation_failure=None,
      )


  @pytest.mark.asyncio
  async def test_surfacer_surfaces_invented_source(tmp_path, sessions_service) -> None:
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          sources={SOURCE_COMPONENT_ID: _llm_authored_source()},
          nodes=(),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.INVENTED_SOURCE in kinds
  ```

  **Verify the SourceSpec/CompositionState constructor shape first** — the
  `modality` value, the `sources=` kwarg, and the `_persist` helper's
  `sources` round-trip must match the real types:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "class SourceSpec\|def __init__\|modality\|llm_generated\|_is_llm_authored_modality" src/elspeth/web/composer/state.py src/elspeth/web/interpretation_state.py | head
  ```
  Adjust the `modality` literal and any required SourceSpec fields to the
  verbatim values the codebase uses (read `_is_llm_authored_modality` to get the
  accepted modality string).

  Run to fail (then pass once the literals match):
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_invented_source -x -q
  ```
  If it fails with the surfacer skipping the source, the `_backend_surface_args_for_site`
  INVENTED_SOURCE branch is reading the wrong key — fix the impl. If it fails at
  the writer boundary with a `ValueError`, the test's authoring/requirement
  shape does not match what `_persist` round-trips — fix the test fixture.
  Expected final: `1 passed`.

- [ ] **Step 3: Run the full surfacer suite.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -q
  ```
  Expected: `5 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
test(composer/guided): pin invented_source surfacing at source-commit boundary

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.4: Per-boundary assertion — recipe-apply commit surfaces auto-staged kinds + `pipeline_decision`

**Files:**
- Modify: `tests/unit/web/composer/test_surface_pending_interpretation_reviews.py`

**Interfaces:**
- Consumes: `_options_with_default_llm_reviews` (`composer/tools/_common.py:250`)
  — the recipe-apply path auto-stages `llm_prompt_template` + `llm_model_choice`;
  the raw-HTML cleanup `pipeline_decision` site
  (`_missing_raw_html_cleanup_review_sites`, `interpretation_state.py:594`) fires
  when a `web_scrape` node's content/fingerprint fields are not preserved by a
  `field_mapper(select_only)` node; the writer boundary's PIPELINE_DECISION
  branch (`sessions/service.py:2833`) requires exactly one pending requirement +
  `validate_pipeline_decision_semantics`.

**Why:** the recipe-apply path is the zero-LLM commit that auto-stages reviews
then (today) stamps COMPLETED — the silent-orphan the spec calls out. The
`web_scrape → field_mapper` raw-HTML cleanup `pipeline_decision` is staged by
the P5 recipe builder; here we pin that a node carrying a staged
`pipeline_decision` requirement surfaces. (P5 owns the recipe + its
end-to-end surfacing assertion; this pins the kind path against the writer
boundary in isolation so it does not depend on P5 having landed.)

- [ ] **Step 1: Read the pipeline_decision requirement + semantics validator
  shape.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "validate_pipeline_decision_semantics\|RAW_HTML_CLEANUP_USER_TERM\|pipeline_decision\|def validate_pipeline_decision" src/elspeth/web/interpretation_state.py src/elspeth/web/sessions/service.py | head
  ```
  Read `validate_pipeline_decision_semantics` to learn the required
  `plugin`/`options`/`user_term`/`draft` relationship, and
  `RAW_HTML_CLEANUP_USER_TERM`'s value.

- [ ] **Step 2: Write the failing test for the pipeline_decision path.**
  Append a `field_mapper` node carrying a pending `pipeline_decision`
  requirement whose `user_term`/`draft` match what
  `validate_pipeline_decision_semantics` accepts (read from Step 1 — use the
  verbatim `RAW_HTML_CLEANUP_USER_TERM` import and the exact options shape):

  ```python
  from elspeth.web.interpretation_state import RAW_HTML_CLEANUP_USER_TERM


  def _field_mapper_pipeline_decision_node() -> NodeSpec:
      # Shape mirrors what the P5 recipe builder stages on the field_mapper node.
      # Fill plugin/options/draft to satisfy validate_pipeline_decision_semantics
      # (read its body in Step 1); the draft is the decision text the requirement
      # carries.
      return NodeSpec(
          id="field_mapper_cleanup",
          node_type="transform",
          plugin="field_mapper",
          input="rated",
          on_success="main",
          on_error=None,
          options={
              "select_only": True,
              "mapping": {"rating": "rating", "url": "url"},
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "pipeline_decision_review",
                      "kind": InterpretationKind.PIPELINE_DECISION.value,
                      "user_term": RAW_HTML_CLEANUP_USER_TERM,
                      "status": "pending",
                      "draft": "Drop raw web_scrape content/fingerprint fields from the jsonl output.",
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  }
              ],
          },
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  @pytest.mark.asyncio
  async def test_surfacer_surfaces_pipeline_decision(tmp_path, sessions_service) -> None:
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(_field_mapper_pipeline_decision_node(),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer._surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.PIPELINE_DECISION in kinds
  ```

  Run to fail/pass:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_pipeline_decision -x -q
  ```
  If the writer boundary rejects with a `validate_pipeline_decision_semantics`
  `ValueError`, the test's `plugin`/`options`/`draft` shape does not match the
  validator — adjust the fixture to the exact contract Step 1 read (this is a
  test-fixture fidelity issue, not an impl defect). Expected final: `1 passed`.

- [ ] **Step 3: Run the full surfacer suite (all five kinds now covered:
  prompt_template, model_choice, invented_source, pipeline_decision; vague_term
  skip).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -q
  ```
  Expected: `6 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
test(composer/guided): pin pipeline_decision surfacing (recipe-apply boundary)

Covers the fifth kind through the surfacer; together with the
prompt_template/model_choice/invented_source tests and the
bare-vague_term skip, all five InterpretationKind members are pinned.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.5: Backend backstop — unresolved card BLOCKS run; resolving PERMITS run

**Files:**
- Create: `tests/integration/web/composer/test_guided_interpretation_run_backstop.py`

**Interfaces:**
- Consumes: `materialize_state_for_execution(state)` returns
  `InterpretationReviewPending` when an unresolved site exists, and
  `execution/service.py:514-525` raises `UnresolvedInterpretationPlaceholderError`
  on it; `ExecutionServiceImpl.execute(session_id=...)` (the production run
  path); `SessionServiceImpl.resolve_interpretation_event(...)` resolves a
  pending row. Mirror the freeform mock pattern from
  `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`
  (real `ComposerServiceImpl` + `SessionServiceImpl` + real session engine).

**Spec mandate (§9.1, rev 4):** "put the blocks-run/permits-run assertion at the
BACKEND integration tier ... hit the run path with an unresolved
`interpretation_events` row." This is the load-bearing backstop; E2E is reserved
for UI projection only (P3.6).

- [ ] **Step 1: Find the canonical ExecutionService run-tier test scaffold and
  the resolve-event API.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "UnresolvedInterpretationPlaceholderError\|ExecutionServiceImpl(\|def service\|mock_session_service\|async def resolve_interpretation_event" tests/unit/web/execution/test_service.py src/elspeth/web/sessions/service.py | head -25
  ```
  Read the `service`/`ExecutionServiceImpl` fixture in `test_service.py` and the
  `resolve_interpretation_event` signature in `sessions/service.py`. Decide the
  build: the cleanest backstop drives a real `ExecutionServiceImpl` over a real
  session engine with a persisted state carrying an unresolved LLM node, asserts
  `execute()` raises `UnresolvedInterpretationPlaceholderError`, then surfaces +
  resolves the pending event and asserts `execute()` returns a run id.

- [ ] **Step 2: Write the failing backstop test.**
  Create `tests/integration/web/composer/test_guided_interpretation_run_backstop.py`.
  Build a persisted state with an LLM node carrying a bare
  `{{interpretation:cool}}` placeholder (so `materialize_state_for_execution`
  reports an unresolved site), wire the real `ComposerServiceImpl` +
  `SessionServiceImpl` + `ExecutionServiceImpl` (copy the construction from the
  test_service.py fixture and the dispatch-test `_build_composer`). The helper
  `_persist_state_with_unresolved_node` (a) seeds the session + composition state
  with the bare-placeholder LLM node and (b) directly creates a pending,
  resolvable `vague_term` event for that placeholder — the surfacer SKIPS bare
  vague_term tokens (P3.1, they carry no staged requirement), so the resolvable
  card is created via the writer boundary, whose `vague_term`/`else` branch only
  requires the node be an LLM transform node (no staged requirement —
  `sessions/service.py:2876-2914`). Resolving that event via
  `resolve_interpretation_event(... choice=ACCEPTED_AS_DRAFTED ...)` patches the
  prompt template (substitutes the placeholder, `sessions/service.py:3158`),
  clearing the run-time gate. The helper returns
  `(session_id, state_id, event_id, state)`:

  ```python
  from __future__ import annotations

  import asyncio
  from collections.abc import Coroutine
  from datetime import UTC, datetime
  from pathlib import Path
  from typing import Any, cast
  from unittest.mock import MagicMock
  from uuid import UUID, uuid4

  import pytest
  import structlog
  from sqlalchemy import insert
  from sqlalchemy.pool import StaticPool

  from elspeth.contracts.composer_interpretation import (
      InterpretationChoice,
      InterpretationKind,
  )
  from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      PipelineMetadata,
  )
  from elspeth.web.config import WebSettings
  from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError
  from elspeth.web.execution.progress import ProgressBroadcaster
  from elspeth.web.execution.service import ExecutionServiceImpl
  from elspeth.web.sessions.engine import create_session_engine
  from elspeth.web.sessions.models import sessions_table
  from elspeth.web.sessions.protocol import CompositionStateData
  from elspeth.web.sessions.schema import initialize_session_schema
  from elspeth.web.sessions.service import SessionServiceImpl
  from elspeth.web.sessions.telemetry import build_sessions_telemetry


  @pytest.fixture
  def engine():
      eng = create_session_engine(
          "sqlite:///:memory:",
          connect_args={"check_same_thread": False},
          poolclass=StaticPool,
      )
      initialize_session_schema(eng)
      return eng


  @pytest.fixture
  def sessions_service(engine) -> SessionServiceImpl:
      return SessionServiceImpl(
          engine,
          telemetry=build_sessions_telemetry(),
          log=structlog.get_logger("test.sessions"),
      )


  @pytest.fixture(autouse=True)
  def _force_available(monkeypatch: pytest.MonkeyPatch) -> None:
      def _available(self: ComposerServiceImpl) -> ComposerAvailability:
          return ComposerAvailability(available=True, model=self._model, provider="anthropic")

      monkeypatch.setattr(ComposerServiceImpl, "_compute_availability", _available)


  def _composer(tmp_path: Path, sessions_service: SessionServiceImpl) -> ComposerServiceImpl:
      from unittest.mock import MagicMock

      from elspeth.web.catalog.protocol import CatalogService

      catalog = MagicMock(spec=CatalogService)
      catalog.list_sources.return_value = []
      catalog.list_transforms.return_value = []
      catalog.list_sinks.return_value = []
      settings = WebSettings(
          data_dir=tmp_path,
          composer_model="anthropic/claude-opus-4-7",
          shareable_link_signing_key=b"\x00" * 32,
      )
      return ComposerServiceImpl(
          catalog=catalog,
          settings=settings,
          sessions_service=sessions_service,
          session_engine=sessions_service._engine,
      )


  def _build_execution_service(
      tmp_path: Path,
      sessions_service: SessionServiceImpl,
  ) -> ExecutionServiceImpl:
      """Real ExecutionServiceImpl over the REAL SessionServiceImpl so execute()'s
      get_current_state(session_id) (execution/service.py:483) loads the persisted
      unresolved state and the interpretation gate (:514-525) sees it.

      The loop / yaml_generator / settings are mocked exactly as the canonical
      `service` fixture mocks them (test_service.py:281-289, 296-306) so the PERMIT
      path returns a run id without running a real engine. The gate fires BEFORE
      the loop/yaml stages, so the BLOCK assertion needs no engine; the PERMIT
      assertion only needs the real SessionService (create_run) + the stubbed loop.
      """
      mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
      broadcaster = ProgressBroadcaster(mock_loop)
      mock_settings = MagicMock()
      mock_settings.get_landscape_url.return_value = "sqlite:///test_audit.db"
      mock_settings.get_payload_store_path.return_value = tmp_path / "payloads"
      mock_settings.landscape_passphrase = None
      # State under test has source=None, so the path allowlist is skipped; data_dir
      # is still read by the source/sink path guard, so pin it to a real dir.
      mock_settings.data_dir = str(tmp_path)
      mock_yaml_generator = MagicMock()
      mock_yaml_generator.generate_yaml.return_value = "source:\n  plugin: csv\n  options: {}\n"
      svc = ExecutionServiceImpl(
          loop=mock_loop,
          broadcaster=broadcaster,
          settings=mock_settings,
          session_service=sessions_service,
          yaml_generator=mock_yaml_generator,
          telemetry=build_sessions_telemetry(),
      )
      # Bridge _call_async to a synchronous run (test_service.py:296-306): the real
      # _call_async uses run_coroutine_threadsafe, which needs a running loop.
      _real_loop = asyncio.new_event_loop()

      def _mock_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
          try:
              return _real_loop.run_until_complete(coro)
          except RuntimeError:
              coro.close()
              return None

      cast(Any, svc)._call_async = _mock_call_async
      return svc


  def _llm_node_with_placeholder(term: str = "cool") -> NodeSpec:
      # Bare {{interpretation:<term>}} token (mirrors the dispatch test's
      # _llm_node_spec at test_compose_loop_interpretation_review_dispatch.py:217):
      # an unresolved site for materialize_state_for_execution, resolvable once the
      # pending vague_term event is accepted (resolve patches prompt_template).
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={"prompt_template": f"Rate how {{{{interpretation:{term}}}}} this row is."},
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  async def _persist_state_with_unresolved_node(
      sessions_service: SessionServiceImpl,
      composer: ComposerServiceImpl,
      *,
      term: str = "cool",
  ) -> tuple[UUID, UUID, UUID, CompositionState]:
      """Seed a session + state with a bare-placeholder LLM node AND a pending,
      resolvable vague_term event for it. Returns (session_id, state_id, event_id, state)."""
      state = CompositionState(
          source=None,
          nodes=(_llm_node_with_placeholder(term),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id = uuid4()
      with sessions_service._engine.begin() as conn:
          conn.execute(
              insert(sessions_table).values(
                  id=str(session_id),
                  user_id="alice",
                  auth_provider_type="local",
                  title="run-backstop test",
                  created_at=datetime.now(UTC),
                  updated_at=datetime.now(UTC),
              )
          )
      state_dict = state.to_dict()
      record = await sessions_service.save_composition_state(
          session_id,
          CompositionStateData(
              nodes=state_dict["nodes"],
              sources=state_dict["sources"],
              metadata_=state_dict["metadata"],
              is_valid=True,
          ),
          provenance="tool_call",
      )
      # Create the resolvable pending vague_term event directly through the writer
      # boundary (the surfacer skips bare vague_term tokens). The vague_term branch
      # only requires the node be an LLM transform node — no staged requirement.
      event = await sessions_service.create_pending_interpretation_event(
          session_id=session_id,
          composition_state_id=record.id,
          affected_node_id="rate_node",
          tool_call_id=f"backend_auto_surface:{uuid4()}",
          user_term=term,
          kind=InterpretationKind.VAGUE_TERM,
          llm_draft="modern, useful, engaging, and clear for the public.",
          model_identifier=composer._model,
          model_version=composer._model,
          provider="anthropic",
          composer_skill_hash=composer._composer_skill_hash,
      )
      # InterpretationEventRecord.id is a UUID (contracts/composer_interpretation.py:210).
      return session_id, record.id, event.id, state


  @pytest.mark.asyncio
  async def test_unresolved_card_blocks_run_resolving_permits(
      tmp_path: Path,
      sessions_service: SessionServiceImpl,
  ) -> None:
      composer = _composer(tmp_path, sessions_service)
      execution_service = _build_execution_service(tmp_path, sessions_service)
      session_id, _state_id, event_id, _state = await _persist_state_with_unresolved_node(
          sessions_service, composer
      )

      # 2. run-time gate BLOCKS: execute() raises on the unresolved placeholder.
      with pytest.raises(UnresolvedInterpretationPlaceholderError):
          await execution_service.execute(session_id=session_id)

      # 3. resolve the pending card as accepted-as-drafted (patches prompt_template).
      await sessions_service.resolve_interpretation_event(
          session_id=session_id,
          event_id=event_id,
          choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
          amended_value=None,
          actor="user:alice",
      )

      # 4. with the placeholder resolved, the run-time gate PERMITS:
      run_id = await execution_service.execute(session_id=session_id)
      assert run_id is not None
  ```

  The test is drop-in: `_build_execution_service` (inlined above) constructs a real
  `ExecutionServiceImpl` whose `session_service` is the REAL `SessionServiceImpl`
  (so `execute()`'s `get_current_state(session_id)` at `execution/service.py:483`
  loads the persisted unresolved state and the gate at `:514-525` sees it), with
  the `loop` / `yaml_generator` / `settings` mocked exactly as the canonical
  `service` fixture mocks them (`tests/unit/web/execution/test_service.py:281-289,
  296-306`) so the PERMIT path (step 4) returns a run id without running a real
  engine. The gate fires BEFORE the loop/yaml stages, so the BLOCK assertion needs
  no engine; the PERMIT assertion only needs `create_run` (real SessionService) +
  the stubbed loop to yield a run id. `resolve_interpretation_event` reads the
  pending event's `llm_draft` as the accepted value when
  `choice == ACCEPTED_AS_DRAFTED` (`sessions/service.py:3160-3166`), so
  `amended_value=None` is correct. The resolve substitutes
  `{{interpretation:cool}}` so step 4's `materialize_state_for_execution` returns a
  real state, not `InterpretationReviewPending`.

  Run to fail:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/test_guided_interpretation_run_backstop.py -x -q
  ```
  Expected failure on first authoring: either the BLOCK assertion (if the node
  shape doesn't trip the gate) or the PERMIT assertion (if resolution doesn't
  clear it). Iterate the fixture node shape until BOTH branches are exercised by
  real production code paths (no mock of `materialize_state_for_execution`).

- [ ] **Step 3: Make it pass.**
  No production change should be needed — the run-time gate
  (`execution/service.py:514-525`) and the surfacer (P3.1) already exist. The
  work is constructing a state whose placeholder is resolvable. Once the fixture
  shape is right:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/integration/web/composer/test_guided_interpretation_run_backstop.py -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add tests/integration/web/composer/test_guided_interpretation_run_backstop.py && git commit -m "$(cat <<'EOF'
test(composer/guided): backend run-tier backstop for interpretation gate (B1)

Pins the load-bearing blocks-run/permits-run backstop at the backend
integration tier (spec §9.1 rev 4): an unresolved interpretation card
makes ExecutionService.execute raise
UnresolvedInterpretationPlaceholderError; surfacing + resolving the
pending event permits the run. No production code change — exercises the
existing run-time gate + the P3 surfacer through real paths.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.6: Frontend — guided ChatPanel branch projects pending events + blocks advancement; respondGuided refreshes the store

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (guided
  branch, `:1109-1185`)
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` (`respondGuided`,
  `:1240-1276`)
- Create: `src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.tsx`
- Create: `src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.test.tsx`

**Interfaces:**
- Consumes: `useInterpretationEventsStore` selector `pendingBySession:
  Record<string, Record<string, InterpretationEvent>>`
  (`stores/interpretationEventsStore.ts:82`); `InterpretationReviewTurn`
  (`components/chat/guided/InterpretationReviewTurn.tsx:244`) props `event`,
  `sessionId`, `showOptOut?`, `showAmend?`, `autoFocusOnMount?`, `onResolved?`;
  the projection/filter pattern from `TutorialTurn2bShowBuilt.tsx:34-45`
  (`choice === "pending" && interpretation_source === "user_approved"`);
  `refreshInterpretationEventsForSession(sessionId)`
  (`sessionStore.ts:146`); `GuidedTurn` disabled prop (`ChatPanel.tsx:1176`).
- Produces (NEW, owned by P3): `GuidedInterpretationReviews` React component
  with props `{ sessionId: string }`; a derived `hasPendingInterpretations`
  boolean consumed by the guided branch to disable `GuidedTurn`.

**Why the guided branch and not `GuidedTurn`:** `GuidedTurn.tsx`'s
`interpretation_review` case is **dead code** — the guided branch
(`ChatPanel.tsx:1109`) renders only `guidedNextTurn`, never a backend-emitted
`interpretation_review` turn (the backend has no such `TurnType`; D12). The
projection path is the `pendingBySession` store, exactly as
`TutorialTurn2bShowBuilt` already does it for the big-bang tutorial.

- [ ] **Step 1: Add the store refresh after a guided respond (the data is never
  fetched today).**
  `respondGuided` (`sessionStore.ts:1240`) atomically replaces the four guided
  wire fields but never refreshes `interpretationEventsStore`, so a pending card
  the backend just created via P3.2 is invisible. Mirror the freeform paths
  (`sessionStore.ts:646/754/1068`).

  Write the failing test first — extend the sessionStore test:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -rln "respondGuided" src/elspeth/web/frontend/src/stores/sessionStore.test.ts
  ```
  Read the closest existing `respondGuided` test (the happy path,
  `sessionStore.guided.test.ts:231-256`) and add one asserting that a successful
  `respondGuided` triggers `refreshAll` on the interpretation store (spy on
  `useInterpretationEventsStore.getState().refreshAll`). The test file already
  mocks `@/api/client` (`:14-37`, `respondGuided: vi.fn()`) and pre-seeds the
  active session via `useSessionStore.setState({ activeSessionId: "sess-1" })`;
  `useInterpretationEventsStore` is NOT mocked, so spying on its `getState` works.
  Add the import `import { useInterpretationEventsStore } from
  "@/stores/interpretationEventsStore";` near the top, then:

  ```ts
  it("refreshes interpretation events after a successful guided respond", async () => {
    const { respondGuided } = await import("@/api/client");
    (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
      sampleRespondResponse,
    );
    const refreshAll = vi.fn(async () => {});
    vi.spyOn(useInterpretationEventsStore, "getState").mockReturnValue({
      ...useInterpretationEventsStore.getState(),
      refreshAll,
    });
    // Pre-seed the active session (same as the happy-path test at :238).
    useSessionStore.setState({ activeSessionId: "sess-1" });

    await useSessionStore.getState().respondGuided({
      chosen: ["csv"],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });

    expect(refreshAll).toHaveBeenCalledWith("sess-1");
  });
  ```

  Run to fail:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm test -- --run sessionStore
  ```
  Expected: the new test fails (refreshAll not called).

  Add the refresh in `respondGuided`, in the success branch right after the
  `set({...})` that replaces the wire fields (`sessionStore.ts:1266`), guarded by
  the same stale-session check already in scope:
  ```ts
        // B1 (spec §5/D12): the backend may have surfaced new pending
        // interpretation cards for committed source/transform/recipe-apply
        // sites. Rehydrate the per-session projection so the guided surface can
        // render + block on them (mirrors the freeform refresh sites).
        refreshInterpretationEventsForSession(requestedSessionId);
  ```
  (Confirm `refreshInterpretationEventsForSession` is in scope in
  `sessionStore.ts`; it is defined at `:146`.)

  > Name note: the test spies on `refreshAll` while the impl calls
  > `refreshInterpretationEventsForSession` — these are consistent.
  > `refreshInterpretationEventsForSession(sessionId)` (`sessionStore.ts:146`) is a
  > thin wrapper that delegates to
  > `useInterpretationEventsStore.getState().refreshAll(sessionId)` (`:147`), so the
  > spy on `refreshAll(sessionId)` observes exactly the wrapper's one call.

  Run to pass:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm test -- --run sessionStore
  ```
  Expected: pass.

- [ ] **Step 2: Write the failing test for the new projection component.**
  Create `GuidedInterpretationReviews.test.tsx`. Seed
  `useInterpretationEventsStore` with one pending `user_approved` event for a
  session and assert the component renders an `InterpretationReviewTurn` (by its
  accessible region role / a stable testid) and exposes the pending count.
  Mirror `TutorialTurn2bShowBuilt.test.tsx` for the store-seeding pattern:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && sed -n '1,60p' src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx
  ```
  Test skeleton:

  ```tsx
  import { render, screen } from "@testing-library/react";
  import { describe, expect, it, beforeEach } from "vitest";
  import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
  import { GuidedInterpretationReviews } from "./GuidedInterpretationReviews";
  import type { InterpretationEvent } from "@/types/interpretation";

  const SID = "11111111-1111-1111-1111-111111111111";

  function pendingEvent(id: string): InterpretationEvent {
    // Every required field of InterpretationEvent (types/interpretation.ts:122-165),
    // so no `as` cast is needed and a future field addition fails the typecheck.
    return {
      id,
      session_id: SID,
      composition_state_id: "22222222-2222-2222-2222-222222222222",
      affected_node_id: "rate_node",
      tool_call_id: "backend_auto_surface:abc",
      user_term: "llm_model_choice:rate_node",
      kind: "llm_model_choice",
      llm_draft: "anthropic/claude-sonnet-4.6",
      accepted_value: null,
      choice: "pending",
      created_at: "2026-06-22T00:00:00Z",
      resolved_at: null,
      actor: "system:composer",
      interpretation_source: "user_approved",
      model_identifier: "anthropic/claude-opus-4-7",
      model_version: "anthropic/claude-opus-4-7",
      provider: "anthropic",
      composer_skill_hash: "0".repeat(64),
      arguments_hash: null,
      hash_domain_version: null,
      runtime_model_identifier_at_resolve: null,
      runtime_model_version_at_resolve: null,
      resolved_prompt_template_hash: null,
    };
  }

  describe("GuidedInterpretationReviews", () => {
    beforeEach(() => {
      useInterpretationEventsStore.setState({
        pendingBySession: { [SID]: { e1: pendingEvent("e1") } },
      });
    });

    it("renders a review affordance for each pending user_approved event", () => {
      render(<GuidedInterpretationReviews sessionId={SID} />);
      // InterpretationReviewTurn renders a role="region" with a kind-aware name
      expect(screen.getByRole("region")).toBeInTheDocument();
    });

    it("renders nothing when there are no pending events", () => {
      useInterpretationEventsStore.setState({ pendingBySession: { [SID]: {} } });
      const { container } = render(<GuidedInterpretationReviews sessionId={SID} />);
      expect(container).toBeEmptyDOMElement();
    });
  });
  ```

  Fill the remaining `InterpretationEvent` required fields from
  `src/types/interpretation.ts`:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "interface InterpretationEvent" -A 30 src/elspeth/web/frontend/src/types/interpretation.ts
  ```

  Run to fail:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm test -- --run GuidedInterpretationReviews
  ```
  Expected failure: cannot resolve `./GuidedInterpretationReviews`.

- [ ] **Step 3: Implement the projection component.**
  Create `GuidedInterpretationReviews.tsx`. The projection, filter, sort, and the
  `InterpretationReviewTurn` render are copied from the proven
  `TutorialTurn2bShowBuilt.tsx` path: the same
  `pendingBySession[sessionId]` lookup, the same
  `event.choice === "pending" && event.interpretation_source === "user_approved"`
  filter (`TutorialTurn2bShowBuilt.tsx:34-43`), the same created-at-then-id sort
  (`compareInterpretationEventsByCreatedAt`, `:150-157`), and the same
  `InterpretationReviewTurn` props block (`showOptOut={false}`,
  `showAmend={event.kind === "vague_term"}`, `autoFocusOnMount={index === 0}`,
  `onResolved` writing the new state back, `:100-114`). The only deltas: it is a
  standalone guided component (returns `null` when empty rather than rendering an
  empty-state paragraph), and it also exports a
  `useHasPendingGuidedInterpretations` predicate the ChatPanel branch uses to
  disable `GuidedTurn`:

  ```tsx
  import { useMemo } from "react";
  import { InterpretationReviewTurn } from "./InterpretationReviewTurn";
  import { useInterpretationEventsStore } from "@/stores/interpretationEventsStore";
  import { useSessionStore } from "@/stores/sessionStore";
  import type { InterpretationEvent } from "@/types/interpretation";

  interface GuidedInterpretationReviewsProps {
    sessionId: string;
  }

  function byCreatedAt(a: InterpretationEvent, b: InterpretationEvent): number {
    const order = a.created_at.localeCompare(b.created_at);
    return order !== 0 ? order : a.id.localeCompare(b.id);
  }

  /**
   * Projects pending interpretation-review events for the guided session and
   * renders each via InterpretationReviewTurn. The guided ChatPanel branch
   * blocks advancement while any pending remains (D12). The GuidedTurn
   * interpretation_review case is dead code; this store projection is the path.
   */
  export function GuidedInterpretationReviews({
    sessionId,
  }: GuidedInterpretationReviewsProps): JSX.Element | null {
    const pendingBySession = useInterpretationEventsStore((s) => s.pendingBySession);
    const pending = useMemo(() => {
      const events = Object.values(pendingBySession[sessionId] ?? {});
      return events
        .filter(
          (event) =>
            event.choice === "pending" &&
            event.interpretation_source === "user_approved",
        )
        .sort(byCreatedAt);
    }, [pendingBySession, sessionId]);

    if (pending.length === 0) return null;

    return (
      <section className="guided-interpretation-reviews" aria-label="Assumptions to review">
        <p className="guided-interpretation-count" role="status">
          {pending.length} {pending.length === 1 ? "assumption" : "assumptions"} to review
        </p>
        {pending.map((event, index) => (
          <InterpretationReviewTurn
            key={event.id}
            event={event}
            sessionId={sessionId}
            showOptOut={false}
            showAmend={event.kind === "vague_term"}
            autoFocusOnMount={index === 0}
            onResolved={(newState) => {
              if (newState !== null) {
                useSessionStore.setState({ compositionState: newState });
              }
            }}
          />
        ))}
      </section>
    );
  }

  /**
   * True when the guided session has at least one pending user_approved
   * interpretation card — the predicate the ChatPanel guided branch uses to
   * disable the wizard turn while reviews are outstanding.
   */
  export function useHasPendingGuidedInterpretations(sessionId: string): boolean {
    const pendingBySession = useInterpretationEventsStore((s) => s.pendingBySession);
    return useMemo(() => {
      const events = Object.values(pendingBySession[sessionId] ?? {});
      return events.some(
        (event) =>
          event.choice === "pending" &&
          event.interpretation_source === "user_approved",
      );
    }, [pendingBySession, sessionId]);
  }
  ```

  Run to pass:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm test -- --run GuidedInterpretationReviews
  ```
  Expected: pass.

- [ ] **Step 4: Wire the component + blocking into the ChatPanel guided branch.**
  In `ChatPanel.tsx`, inside the guided branch (`:1109`), import the new symbols
  at the top of the file (near the `GuidedTurn` import, `:30`):
  ```tsx
  import {
    GuidedInterpretationReviews,
    useHasPendingGuidedInterpretations,
  } from "./guided/GuidedInterpretationReviews";
  ```
  Compute the blocking flag where the other guided hooks are read (the guided
  branch has `guidedSession` in scope from `:389`; `activeSessionId` is in scope
  from the freeform interpretation block at `:528`). Add near the top of the
  component body (NOT inside the conditional return — hooks must be unconditional;
  call it with the session id or an empty string when null):
  ```tsx
    const hasPendingGuidedInterpretations = useHasPendingGuidedInterpretations(
      activeSessionId ?? "",
    );
  ```
  Then inside the guided branch JSX, render the reviews above the wizard turn
  and gate `GuidedTurn`'s `disabled` prop. Replace the `<GuidedTurn ... />` block
  (`:1173-1177`) with:
  ```tsx
            <GuidedInterpretationReviews sessionId={activeSessionId ?? ""} />
            <GuidedTurn
              turn={guidedNextTurn}
              onSubmit={(body) => void respondGuided(body)}
              disabled={guidedResponsePending || hasPendingGuidedInterpretations}
            />
  ```
  (The `?? ""` is safe: the guided branch is only reached when a guided session
  is active; `GuidedInterpretationReviews` returns null on an empty session id
  because `pendingBySession[""]` is undefined.)

- [ ] **Step 5: Write the failing ChatPanel test asserting the wizard turn is
  disabled while a pending card exists, then run-to-pass.**
  Extend the ChatPanel guided test:
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -rln "chat-panel--guided\|GuidedTurn\|guidedNextTurn" src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
  ```
  Add a test that, with a pending `user_approved` event seeded in
  `interpretationEventsStore` and an active guided session + next turn, the
  submit control of `GuidedTurn` is disabled (assert via the turn widget's
  primary button `disabled` attribute, the same way the existing guided ChatPanel
  tests query it). Skeleton assertion:
  ```tsx
  // seed pendingBySession[SID] with one pending user_approved event,
  // mount ChatPanel with guidedSession+guidedNextTurn set, then:
  expect(screen.getByRole("button", { name: /confirm|continue|accept|submit/i }))
    .toBeDisabled();
  ```
  Read the existing guided ChatPanel test to match the exact turn-button query
  it already uses; do not invent a new selector.

  Run to fail (before Step 4 is applied) / pass (after):
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm test -- --run ChatPanel
  ```
  Expected after Step 4: pass.

- [ ] **Step 6: Frontend gates — typecheck + build + full vitest.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut/src/elspeth/web/frontend && npm run typecheck && npm test -- --run && npm run build
  ```
  Expected: typecheck clean, all vitest pass, build succeeds.

- [ ] **Step 7: Commit.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.test.tsx src/elspeth/web/frontend/src/stores/sessionStore.ts src/elspeth/web/frontend/src/stores/sessionStore.test.ts src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx && git commit -m "$(cat <<'EOF'
feat(frontend/guided): project pending interpretation cards + block advancement (D12)

The guided ChatPanel branch now projects interpretationEventsStore
pendingBySession through a new GuidedInterpretationReviews component
(rendering each via InterpretationReviewTurn) and disables the wizard
turn while any pending user_approved card remains. respondGuided now
refreshes the interpretation store after a successful response so
backend-surfaced cards (B1) become visible. The GuidedTurn
interpretation_review case stays dead code; the store projection is the
path.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task P3.7: Phase gate sweep

**Files:** none (verification only).

- [ ] **Step 1: Backend lint + types over the touched files.**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/ && uv run mypy src/elspeth/web/composer/service.py src/elspeth/web/sessions/routes/composer.py
  ```
  Expected: all clean. Fix any finding at the boundary (not by suppression).

- [ ] **Step 2: Targeted backend suite (surfacer + route + backstop).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py tests/integration/web/composer/test_guided_commit_surfaces_reviews.py tests/integration/web/composer/test_guided_interpretation_run_backstop.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -q
  ```
  Expected: all pass (the existing freeform dispatch test must stay green — the
  surfacer is additive and does not touch the freeform surface+gate pair).

- [ ] **Step 3: elspeth-lints trust gates (B1 adds a backend interpretation
  surfacer that reads persisted Tier-3 node options).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth
  ```
  Expected: no NEW displacement attributable to P3. If the surfacer trips a
  tier_model entry (it reads `node.options`/`source.options`, already-persisted
  Tier-3 config), state it in the commit/handoff per the gate-debt doctrine —
  do not blind-sign.

- [ ] **Step 4: wardline taint gate (the surfacer consumes persisted
  composition state and writes interpretation rows).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && wardline scan . --fail-on ERROR
  ```
  Expected: exit 0. Fix any finding at the boundary.

- [ ] **Step 5: Commit any gate-driven fixups (if Steps 1-4 required edits).**
  ```
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add -A && git commit -m "$(cat <<'EOF'
chore(composer/guided): gate fixups for B1 interpretation surfacing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```
  (Skip if the tree is clean after Steps 1-4.)

---

## Phase P4 — web_scrape recipe (D11) — re-polarized shield

> **Phase dependency note.** P4 introduces the `web-scrape-llm-rate-jsonl`
> `RecipeSpec` and the `_web_scrape_predicate`. Both are consumed by the
> **P2 completion-seam redirect** (`handle_step_2_5_recipe_apply` must leave
> `terminal=None` / `step=STEP_4_WIRE`) and by the **P6 advisor terminal gate**.
> P4 deliberately does NOT touch the completion seam, `STEP_4_WIRE`, or the wire
> stage — those land in P2/P3/P6. P4 ships the recipe so that, once P2 redirects
> the recipe-apply seam, the canonical pipeline composes deterministically.
> P4 also relies on the **already-shipped** raw-HTML cleanup contract
> (`composition_review_contract_error` → `raw_html_cleanup_review_contract_error`,
> `interpretation_state.py:164/204`) and the **already-shipped** prompt-shield
> advisory (`prompt_shield_recommendation_warning_pairs`,
> `interpretation_state.py:219`) — P4 adds no new contract, it satisfies the
> existing blocking one and preserves the existing advisory.

> **Resolved fact (pinned by Task P4.1).** The canonical URL-row source is an
> `inline_blob` in the wire payload, but it **materialises** to a real registered
> source plugin via `_MIME_TO_SOURCE` (`tools/sources.py:146`): a JSON URL-list →
> `json`, a CSV URL-list → `csv`, plain text → `text`. So `SourceResolved.plugin`
> is **`json`** (or `csv`) for the canonical source — never the literal string
> `"inline_blob"` and never `"web_scrape"` (web_scrape is a TRANSFORM,
> `plugins/transforms/web_scrape.py:146` `class WebScrapeConfig(TransformDataConfig)`).
> The predicate keys on that resolved plugin + a `url`-column signal + a single
> jsonl output, gated on `blob_ref` (mirroring `_classify_predicate`,
> `recipe_match.py:205`).
>
> **`blob_ref` IS present at match time (the load-bearing fact for §4.1).** When
> `set_pipeline` composes the canonical seed from `source.inline_blob`, it persists the
> inline blob as a real blob row AND **unconditionally writes
> `source.options["blob_ref"]` = the new blob UUID** (`composer/tools/sessions.py:425`,
> inside the `if inline_blob is not None` branch; the same `SourceSpec.options` is what
> `match_recipe` reads). So the `blob_ref` gate is satisfied for the canonical source —
> the predicate matches, `match_recipe` returns the recipe, and §4.1's zero-LLM compose
> fires. The end-to-end proof is **Task P4.2 Step 4b**
> (`test_canonical_seed_materialised_source_matches_web_scrape_recipe`), which drives the
> exact materialised shape through `match_recipe`. (NB: this `source.options["blob_ref"]`
> — a plain UUID string — is a DIFFERENT object from the `{"mode": "inline_content",
> "blob_ref": ...}` widened marker the fork blob-rewrite recurses for
> (`sessions/routes/sessions.py:128`); see the spec's two-objects `blob_ref` note in
> §5/B4. Do not let the spec's loose "inline_blob source has no blob_ref" fork-strip
> shorthand mislead this predicate — at match time, `blob_ref` is present.)

> **Resolved fact (pinned by Task P4.2).** There is **no** registered `llm_rate`
> plugin (`grep -rn 'name = "llm_rate"'` → empty; only `llm` at
> `plugins/transforms/llm/transform.py:1124` and `field_mapper` at
> `plugins/transforms/field_mapper.py:117`). `llm_rate` is a cosmetic display
> label in `tutorial.spec.ts`. The recipe's rating node uses the **real `llm`
> plugin**, which is also what the shield advisory keys on
> (`prompt_shield_recommendation_warning_pairs` checks `node.plugin == "llm"`,
> `interpretation_state.py:225`).

---

### Task P4.1: Pin the resolved canonical-source plugin + add `_web_scrape_predicate`

**Files:**
- Modify `src/elspeth/web/composer/guided/recipe_match.py` (topology helpers after
  `_has_two_json_outputs` at :194; predicate after `_split_threshold_slot_resolver`
  at :346; `_RECIPE_PREDICATES` tuple at :353)
- Modify `tests/unit/web/composer/guided/test_recipe_match.py` (add a
  `TestWebScrapePredicate` class + a resolved-plugin-name pin test)

**Interfaces:**
- Consumes: `SourceResolved` (`guided/resolved.py:19`; fields `plugin`, `options`,
  `observed_columns`, `sample_rows`), `SinkResolved` (`resolved.py:89`),
  `SinkOutputResolved` (`resolved.py:54`; fields `plugin`, `options`,
  `required_fields`, `schema_mode`), `_MIME_TO_SOURCE`
  (`composer/tools/sources.py:146`).
- Produces (NEW, consumed by P4.2 + future phases):
  - `_web_scrape_predicate(source: SourceResolved, sink: SinkResolved) -> bool`
  - `_URL_ROW_SOURCE_PLUGINS: frozenset[str]` (module constant = `frozenset({"json", "csv"})`)
  - `_URL_COLUMN_NAMES: frozenset[str]` (module constant = `frozenset({"url"})`)
  - registry entry `(_web_scrape_predicate, "web-scrape-llm-rate-jsonl", _web_scrape_slot_resolver)`
    appended to `_RECIPE_PREDICATES` (slot resolver added in P4.2).

- [ ] **Step 1: Write the resolved-plugin-name pin test (run-to-fail).**
  Append to `tests/unit/web/composer/guided/test_recipe_match.py`:
  ```python
  class TestCanonicalSourcePluginIsResolved:
      """Pin the fact the predicate relies on: an inline_blob URL list
      materialises to a real registered source plugin, never the literal
      string 'inline_blob' and never 'web_scrape'.

      _MIME_TO_SOURCE is the single mapping that resolves a materialised
      inline_blob's MIME type to its concrete source plugin; the predicate
      must key on those resolved names, not on 'inline_blob'.
      """

      def test_mime_to_source_resolves_url_row_plugins(self) -> None:
          from elspeth.web.composer.tools.sources import _MIME_TO_SOURCE

          resolved_plugins = {plugin for plugin, _extra in _MIME_TO_SOURCE.values()}
          # The canonical URL list is JSON rows of {"url": ...}; CSV is the
          # other URL-row carrier. Both are real registered source plugins.
          assert "json" in resolved_plugins
          assert "csv" in resolved_plugins
          # web_scrape is a TRANSFORM, never a materialised source plugin.
          assert "web_scrape" not in resolved_plugins
          assert "inline_blob" not in resolved_plugins

      def test_url_row_source_plugins_constant_matches_resolved_names(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _URL_ROW_SOURCE_PLUGINS

          assert _URL_ROW_SOURCE_PLUGINS == frozenset({"json", "csv"})
          assert "web_scrape" not in _URL_ROW_SOURCE_PLUGINS
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestCanonicalSourcePluginIsResolved -q
  ```
  Expected failure: `ImportError: cannot import name '_URL_ROW_SOURCE_PLUGINS' from 'elspeth.web.composer.guided.recipe_match'`.

- [ ] **Step 2: Add the module constants + `_web_scrape_predicate` (minimal impl).**
  In `recipe_match.py`, after `_has_two_json_outputs` (ends at :195), insert:
  ```python
  # ---------------------------------------------------------------------------
  # web-scrape-llm-rate-jsonl predicate
  #
  # web_scrape is a TRANSFORM (plugins/transforms/web_scrape.py); the predicate
  # keys on the URL-ROW SOURCE that feeds it, never on web_scrape itself. An
  # inline_blob URL list materialises to a real registered source plugin via
  # _MIME_TO_SOURCE (tools/sources.py): JSON rows -> "json", CSV -> "csv". The
  # predicate matches those resolved names + a "url" column signal + a single
  # jsonl output, gated on blob_ref (same blob-presence discipline as
  # _classify_predicate).
  # ---------------------------------------------------------------------------

  _URL_ROW_SOURCE_PLUGINS: frozenset[str] = frozenset({"json", "csv"})
  _URL_COLUMN_NAMES: frozenset[str] = frozenset({"url"})


  def _has_single_jsonl_output(sink: SinkResolved) -> bool:
      """Return True for a single ``json`` output configured as JSONL.

      The canonical web-scrape pipeline writes one JSONL file. ``json`` is the
      registered sink plugin; ``format: jsonl`` is the JSONL discriminator (an
      absent format is the json plugin's default object-array, not JSONL).
      """
      if not (len(sink.outputs) == 1 and sink.outputs[0].plugin == "json"):
          return False
      return sink.outputs[0].options.get("format") == "jsonl"


  def _source_has_url_column(source: SourceResolved) -> bool:
      """Return True iff the source surfaces a ``url`` column.

      The signal is the observed URL column that web_scrape's ``url_field``
      will read. Observed columns come from inspecting the materialised blob;
      a URL list always surfaces ``url``.
      """
      return any(col in _URL_COLUMN_NAMES for col in source.observed_columns)


  def _web_scrape_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
      """Return True for a blob-backed URL-row source → single JSONL output.

      Matches the canonical tutorial shape: an inline_blob URL list that
      materialised to a ``json``/``csv`` source (NOT ``web_scrape`` — that is a
      downstream transform the recipe inserts) feeding a single JSONL sink, with
      an observed ``url`` column.

      Requires ``blob_ref`` in ``source.options`` for the same reason as
      ``_classify_predicate``: the slot resolver cannot derive ``source_blob_id``
      without it, and "no recipe match" (fall through to the live chain solver)
      is the correct outcome for a non-blob-backed URL source.
      """
      if source.plugin not in _URL_ROW_SOURCE_PLUGINS:
          return False
      if "blob_ref" not in source.options:
          return False
      if not _has_single_jsonl_output(sink):
          return False
      return _source_has_url_column(source)
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestCanonicalSourcePluginIsResolved -q
  ```
  Expected: `2 passed`.

- [ ] **Step 3: Write the predicate behaviour tests (run-to-fail).**
  Append to `test_recipe_match.py`:
  ```python
  def _make_url_json_source(
      blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000099",
      *,
      with_blob: bool = True,
  ) -> SourceResolved:
      """A materialised inline_blob URL list: json plugin, url column, blob_ref."""
      options: dict[str, object] = {}
      if with_blob:
          options["blob_ref"] = blob_ref
      return SourceResolved(
          plugin="json",
          options=options,
          observed_columns=("url",),
          sample_rows=({"url": "https://dta.gov.au"},),
      )


  def _make_single_jsonl_sink(path: str = "outputs/ratings.jsonl") -> SinkResolved:
      return SinkResolved(
          outputs=(
              SinkOutputResolved(
                  plugin="json",
                  options={"path": path, "format": "jsonl"},
                  required_fields=(),
                  schema_mode="observed",
              ),
          )
      )


  class TestWebScrapePredicate:
      def test_matches_blob_backed_json_url_source(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          assert _web_scrape_predicate(_make_url_json_source(), _make_single_jsonl_sink()) is True

      def test_does_not_reference_web_scrape_as_source(self) -> None:
          """A source whose plugin is literally 'web_scrape' must NOT match —
          web_scrape is a transform, not a source."""
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          bad = SourceResolved(
              plugin="web_scrape",
              options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
              observed_columns=("url",),
              sample_rows=({"url": "https://dta.gov.au"},),
          )
          assert _web_scrape_predicate(bad, _make_single_jsonl_sink()) is False

      def test_no_match_without_blob_ref(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          assert _web_scrape_predicate(_make_url_json_source(with_blob=False), _make_single_jsonl_sink()) is False

      def test_no_match_without_url_column(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          no_url = SourceResolved(
              plugin="json",
              options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
              observed_columns=("name",),
              sample_rows=({"name": "x"},),
          )
          assert _web_scrape_predicate(no_url, _make_single_jsonl_sink()) is False

      def test_no_match_for_non_jsonl_output(self) -> None:
          from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

          object_array = SinkResolved(
              outputs=(
                  SinkOutputResolved(
                      plugin="json",
                      options={"path": "outputs/ratings.json"},  # no format: jsonl
                      required_fields=(),
                      schema_mode="observed",
                  ),
              )
          )
          assert _web_scrape_predicate(_make_url_json_source(), object_array) is False
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::TestWebScrapePredicate -q
  ```
  Expected: `5 passed` (the predicate already exists from Step 2 — this step pins
  its behaviour; if any assertion is red, fix the predicate before proceeding).

- [ ] **Step 4: Register the predicate in `_RECIPE_PREDICATES` (run-to-fail).**
  Add a registry-shape test first. Append to `test_recipe_match.py`:
  ```python
  def test_web_scrape_predicate_registered_last() -> None:
      """The web-scrape predicate is registered, after the CSV recipes
      (most-specific-first ordering: the URL-row json source never collides
      with the CSV classify/split predicates, but order is asserted to keep
      registry edits intentional)."""
      from elspeth.web.composer.guided.recipe_match import _RECIPE_PREDICATES

      names = [name for _pred, name, _resolver in _RECIPE_PREDICATES]
      assert "web-scrape-llm-rate-jsonl" in names
      assert names[-1] == "web-scrape-llm-rate-jsonl"
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_web_scrape_predicate_registered_last -q
  ```
  Expected failure: `assert 'web-scrape-llm-rate-jsonl' in [...]` → KeyError/AssertionError
  (also note: `_web_scrape_slot_resolver` does not exist yet — that lands in P4.2;
  for this step register with a **temporary** resolver stub so the tuple is shape-valid).

- [ ] **Step 5: Append the registry entry with a slot-resolver stub (minimal impl).**
  In `recipe_match.py`, immediately after `_web_scrape_predicate`, add the stub
  resolver (the real one lands in P4.2):
  ```python
  def _web_scrape_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
      """Partial slot map for the web-scrape-llm-rate-jsonl recipe.

      Provides ``source_blob_id`` (the composer-canonical blob UUID) and
      ``output_path`` (operator-set verbatim, else a rubber-stampable default).
      User-fillable: ``model``, ``api_key_secret``, ``provider``, ``rating_template``.
      """
      if "blob_ref" not in source.options:
          raise InvariantError(
              "web-scrape recipe slot resolver requires source.options['blob_ref']; "
              f"source options present: {sorted(source.options.keys())}"
          )
      blob_ref = source.options["blob_ref"]
      sink_options = sink.outputs[0].options
      output_path = sink_options["path"] if "path" in sink_options else "outputs/ratings.jsonl"
      return {
          "source_blob_id": blob_ref,
          "output_path": output_path,
      }
  ```
  Then append to `_RECIPE_PREDICATES` (after the split-threshold entry at :355,
  before the closing comment block):
  ```python
      (_web_scrape_predicate, "web-scrape-llm-rate-jsonl", _web_scrape_slot_resolver),
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py -q
  ```
  Expected: all pass (existing classify/split tests + the new ones).
  Note: `match_recipe` (recipe_match.py:371) will now raise `InvariantError`
  ("Recipe '...' is in _RECIPE_PREDICATES but not registered in recipes.py") if
  invoked end-to-end — that is expected and resolved in P4.2 when the `RecipeSpec`
  is registered. The unit tests above call `_web_scrape_predicate` directly, so
  they pass now.

- [ ] **Step 6: Lint + commit.**
  ```
  uv run python -m ruff check src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m ruff format src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  git add src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/guided/test_recipe_match.py
  git commit -m "feat(composer/recipe-match): add web-scrape URL-row source predicate (D11/P4.1)

Predicate matches the materialised inline_blob URL source (json/csv via
_MIME_TO_SOURCE), never web_scrape (a transform) and never the literal
'inline_blob'. Keyed on blob_ref + a url column + a single JSONL output,
mirroring _classify_predicate's blob-presence discipline.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.2: Add the `web-scrape-llm-rate-jsonl` `RecipeSpec` + `_build_web_scrape_recipe`

**Files:**
- Modify `src/elspeth/web/composer/recipes.py` (slot table after `_RECIPE3_SLOTS`
  at :442; builder after `_build_fork_coalesce_truncate_recipe` at :561; registry
  `_RECIPES` at :569)
- Modify `tests/unit/web/composer/test_recipes.py` (registry assertion at :38; new
  `TestWebScrapeRecipeBuild` class)

**Interfaces:**
- Consumes: `RecipeSpec` (`recipes.py:26`), `SlotSpec` (`recipes.py:20`),
  `validate_slots` / `apply_recipe` (`recipes.py:119/643`),
  `RAW_HTML_CLEANUP_USER_TERM = "drop_raw_html_fields"` +
  `RAW_HTML_CLEANUP_REVIEW_DRAFT = "Drop the scraped raw HTML and fingerprint
  fields before saving the JSON output."` (`web/interpretation_state.py:31/32`),
  `INTERPRETATION_REQUIREMENTS_KEY = "interpretation_requirements"`
  (`interpretation_state.py:25`).
- Produces (NEW, consumed by P2 completion-seam + P4.3 contract test + P4.4 caching):
  - `_RECIPE_WEB_SCRAPE_SLOTS: Final[dict[str, SlotSpec]]`
  - `_build_web_scrape_recipe(slots: Mapping[str, Any]) -> dict[str, Any]`
  - registry entry `"web-scrape-llm-rate-jsonl"` in `_RECIPES`.
- Canonical chain produced (set_pipeline-compatible args), head source node named
  `"url_rows"`: `source(json, blob_id=…, on_success="rows")` →
  `web_scrape(input="rows", on_success="scraped")` →
  `llm(input="scraped", on_success="rated")` →
  `field_mapper(input="rated", on_success="clean", select_only=true, mapping
  drops content/fingerprint, interpretation_requirements stages the
  pipeline_decision)` → `json/jsonl` sink.

- [ ] **Step 1: Write the registry + structural build test (run-to-fail).**
  Update the registry assertion at `test_recipes.py:38`:
  ```python
          assert names == {
              "classify-rows-llm-jsonl",
              "split-by-numeric-threshold",
              "fork-coalesce-truncate-jsonl",
              "web-scrape-llm-rate-jsonl",
          }
  ```
  Append a new class:
  ```python
  class TestWebScrapeRecipeBuild:
      """The web-scrape recipe deterministically emits
      source → web_scrape → llm → field_mapper(cleanup) → jsonl."""

      _SLOTS = {
          "source_blob_id": str(uuid4()),
          "model": "anthropic/claude-sonnet-4.6",
          "api_key_secret": "OPENROUTER_API_KEY",
          "output_path": "outputs/ratings.jsonl",
      }

      def _build(self) -> dict:
          return apply_recipe("web-scrape-llm-rate-jsonl", self._SLOTS)

      def test_head_source_node_is_named_json_url_source(self) -> None:
          args = self._build()
          assert args["source"]["plugin"] == "json"
          assert args["source"]["blob_id"] == self._SLOTS["source_blob_id"]
          assert args["source"]["on_success"] == "rows"

      def test_canonical_chain_order(self) -> None:
          args = self._build()
          plugins = [n["plugin"] for n in args["nodes"]]
          assert plugins == ["web_scrape", "llm", "field_mapper"]

      def test_chain_is_wired_by_connection_labels(self) -> None:
          args = self._build()
          by_plugin = {n["plugin"]: n for n in args["nodes"]}
          assert by_plugin["web_scrape"]["input"] == "rows"
          assert by_plugin["web_scrape"]["on_success"] == "scraped"
          assert by_plugin["llm"]["input"] == "scraped"
          assert by_plugin["llm"]["on_success"] == "rated"
          assert by_plugin["field_mapper"]["input"] == "rated"
          assert by_plugin["field_mapper"]["on_success"] == "clean"
          assert args["outputs"][0]["sink_name"] == "clean"
          assert args["outputs"][0]["plugin"] == "json"
          assert args["outputs"][0]["options"]["format"] == "jsonl"

      def test_field_mapper_select_only_excludes_raw_content_and_fingerprint(self) -> None:
          """Data-minimization: the cleanup sink field set EXCLUDES the raw
          web_scrape content/fingerprint fields (pin the actual output set)."""
          args = self._build()
          fm = next(n for n in args["nodes"] if n["plugin"] == "field_mapper")
          assert fm["options"]["select_only"] is True
          mapping = fm["options"]["mapping"]
          preserved = set(mapping) | set(mapping.values())
          assert "content" not in preserved
          assert "content_fingerprint" not in preserved
          # Positive pin: the rating + url ARE preserved (the user-facing output).
          assert "rating" in preserved
          assert "url" in preserved

      def test_field_mapper_stages_pipeline_decision_cleanup_requirement(self) -> None:
          """The raw-HTML cleanup pipeline_decision is staged on the field_mapper
          node so the blocking cleanup contract passes (tools/sessions.py:657 →
          raw_html_cleanup_review_contract_error)."""
          from elspeth.web.interpretation_state import (
              INTERPRETATION_REQUIREMENTS_KEY,
              RAW_HTML_CLEANUP_REVIEW_DRAFT,
              RAW_HTML_CLEANUP_USER_TERM,
          )

          args = self._build()
          fm = next(n for n in args["nodes"] if n["plugin"] == "field_mapper")
          reqs = fm["options"][INTERPRETATION_REQUIREMENTS_KEY]
          decision = next(r for r in reqs if r["kind"] == "pipeline_decision")
          assert decision["user_term"] == RAW_HTML_CLEANUP_USER_TERM
          assert decision["draft"] == RAW_HTML_CLEANUP_REVIEW_DRAFT
          assert decision["status"] == "pending"

      def test_web_scrape_node_declares_content_and_fingerprint_fields(self) -> None:
          """web_scrape must name content_field/fingerprint_field so
          _web_scrape_raw_fields can compute the raw set the cleanup drops."""
          args = self._build()
          ws = next(n for n in args["nodes"] if n["plugin"] == "web_scrape")
          assert ws["options"]["url_field"] == "url"
          assert ws["options"]["content_field"] == "content"
          assert ws["options"]["fingerprint_field"] == "content_fingerprint"

      def test_no_azure_prompt_shield_hard_node(self) -> None:
          """rev 4: omit the unbuildable azure_prompt_shield hard node
          (elspeth-abb2cb0931 — composer cannot instantiate it without
          configured endpoint+api_key secrets)."""
          args = self._build()
          plugins = {n["plugin"] for n in args["nodes"]}
          assert "azure_prompt_shield" not in plugins
  ```
  Run-to-fail:
  ```
  uv run python -m pytest tests/unit/web/composer/test_recipes.py::TestWebScrapeRecipeBuild tests/unit/web/composer/test_recipes.py::TestRecipeRegistry::test_registered_recipes -q
  ```
  Expected failure: `RecipeValidationError: recipe 'web-scrape-llm-rate-jsonl' is
  not registered` (and the registry-set assertion is red).

- [ ] **Step 2: Add the slot table + `_build_web_scrape_recipe` (minimal impl).**
  In `recipes.py`, after `_build_fork_coalesce_truncate_recipe` (ends at :561),
  insert:
  ```python
  # ---------------------------------------------------------------------------
  # Recipe 4: web-scrape-llm-rate-jsonl (D11)
  #
  #   json/csv URL-row source (blob)  →  web_scrape (fetch page content)
  #                                    →  llm (rate the page)
  #                                    →  field_mapper(select_only) cleanup
  #                                    →  jsonl sink (single output)
  #
  # web_scrape is a TRANSFORM, not a source: the head source is a json/csv blob
  # of {url: ...} rows. The field_mapper drops the raw scraped content/fingerprint
  # (data minimization) and stages the kind=pipeline_decision raw-HTML cleanup
  # requirement so the blocking cleanup contract (raw_html_cleanup_review_contract_error,
  # interpretation_state.py:164) passes deterministically.
  #
  # Prompt-injection shield (rev 4, re-polarized): the recipe OMITS an unbuildable
  # azure_prompt_shield hard node (the composer cannot instantiate it without
  # configured endpoint+api_key secrets — elspeth-abb2cb0931, a CONDITIONAL
  # security ticket, NOT a licence to remove all shield signal). It does NOT
  # suppress the existing medium-severity prompt-shield advisory warning
  # (prompt_shield_recommendation_warning_pairs, interpretation_state.py:219),
  # which surfaces at the wire stage. See test_no_azure_prompt_shield_hard_node
  # AND the P4.3 advisory-presence test.
  # ---------------------------------------------------------------------------

  _RECIPE_WEB_SCRAPE_SLOTS: Final[dict[str, SlotSpec]] = {
      "source_blob_id": SlotSpec(
          slot_type="blob_id",
          description="UUID of the operator-supplied URL-list blob (json/csv rows of {url: ...}; use create_blob to wrap inline content first)",
      ),
      "model": SlotSpec(
          slot_type="str",
          description="LLM model identifier (e.g., 'anthropic/claude-sonnet-4.6'); use list_models to discover",
      ),
      "api_key_secret": SlotSpec(
          slot_type="str",
          description=(
              "Name of an inventory secret to wire into the LLM 'api_key' option as "
              "a deferred {secret_ref} marker. Discover names via list_secret_refs; "
              "verify with validate_secret_ref. Literal credential strings are rejected."
          ),
      ),
      "provider": SlotSpec(
          slot_type="str",
          required=False,
          default="openrouter",
          description="LLM provider — 'openrouter' or 'azure'",
      ),
      "rating_template": SlotSpec(
          slot_type="str",
          required=False,
          default="Rate the appeal of this government web page from 1-10 and explain briefly:\n\n{{ row['content'] }}",
          description="Jinja2 template for the rating prompt; reference scraped content as {{ row['content'] }}",
      ),
      "output_path": SlotSpec(
          slot_type="str",
          required=False,
          default="outputs/ratings.jsonl",
          description="JSONL output path",
      ),
  }


  def _build_web_scrape_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
      """Build set_pipeline args for the web-scrape-llm-rate-jsonl recipe.

      Emits source → web_scrape → llm → field_mapper(cleanup) → jsonl, named by
      connection labels (NOT EdgeSpec objects — guided passes edges=[]). The
      field_mapper drops the raw scraped content/fingerprint and stages the
      kind=pipeline_decision raw-HTML cleanup requirement so the blocking cleanup
      contract passes. The unbuildable azure_prompt_shield hard node is omitted
      (elspeth-abb2cb0931); the existing medium-severity prompt-shield advisory
      is left to fire from validate() — the recipe MUST NOT suppress it.
      """
      from elspeth.web.composer.tools._common import _pending_interpretation_requirement
      from elspeth.contracts.composer_interpretation import InterpretationKind
      from elspeth.web.interpretation_state import (
          INTERPRETATION_REQUIREMENTS_KEY,
          RAW_HTML_CLEANUP_REVIEW_DRAFT,
          RAW_HTML_CLEANUP_USER_TERM,
      )

      content_field = "content"
      fingerprint_field = "content_fingerprint"
      cleanup_requirement = _pending_interpretation_requirement(
          requirement_id="drop_raw_html_review",
          kind=InterpretationKind.PIPELINE_DECISION,
          user_term=RAW_HTML_CLEANUP_USER_TERM,
          draft=RAW_HTML_CLEANUP_REVIEW_DRAFT,
      )
      return {
          "source": {
              "plugin": "json",
              "blob_id": slots["source_blob_id"],
              "on_success": "rows",
              "options": {
                  "schema": {"mode": "observed"},
              },
              "on_validation_failure": "discard",
          },
          "nodes": [
              {
                  "id": "url_rows",
                  "node_type": "transform",
                  "plugin": "web_scrape",
                  "input": "rows",
                  "on_success": "scraped",
                  "on_error": "discard",
                  "options": {
                      "schema": {"mode": "observed"},
                      "url_field": "url",
                      "content_field": content_field,
                      "fingerprint_field": fingerprint_field,
                      "format": "markdown",
                      "http": {
                          # OPERATOR: abuse_contact ships as a wire-visible HTTP header to every
                          # scraped third party — it MUST be a real, monitored, operator-owned inbox.
                          # Defaults to the DTA canonical-seed address (mirrors
                          # tutorial_cache.CANONICAL_SEED_PROMPT); non-DTA operators MUST override it.
                          # RFC-reserved / placeholder domains are hard-rejected by
                          # state.py:_validate_web_scrape_abuse_contact_not_reserved (severity=high).
                          "abuse_contact": "noreply@dta.gov.au",
                          "scraping_reason": "Regulatory monitoring (tutorial canonical pipeline)",
                      },
                  },
              },
              {
                  "id": "rate_pages",
                  "node_type": "transform",
                  "plugin": "llm",
                  "input": "scraped",
                  "on_success": "rated",
                  "on_error": "discard",
                  "options": {
                      "provider": slots["provider"],
                      "model": slots["model"],
                      "api_key": {"secret_ref": slots["api_key_secret"]},
                      "prompt_template": slots["rating_template"],
                      "response_field": "rating",
                      "schema": {"mode": "observed"},
                      "required_input_fields": [content_field],
                  },
              },
              {
                  "id": "drop_raw_html",
                  "node_type": "transform",
                  "plugin": "field_mapper",
                  "input": "rated",
                  "on_success": "clean",
                  "on_error": "discard",
                  "options": {
                      "schema": {"mode": "observed"},
                      "select_only": True,
                      # mapping preserves ONLY the user-facing fields; the raw
                      # content/fingerprint are intentionally absent (dropped).
                      "mapping": {
                          "url": "url",
                          "rating": "rating",
                      },
                      INTERPRETATION_REQUIREMENTS_KEY: [cleanup_requirement],
                  },
              },
          ],
          "edges": [],
          "outputs": [
              {
                  "sink_name": "clean",
                  "plugin": "json",
                  "options": {
                      "path": slots["output_path"],
                      "format": "jsonl",
                      "schema": {"mode": "observed"},
                      "mode": "write",
                      "collision_policy": "auto_increment",
                  },
                  "on_write_failure": "discard",
              }
          ],
          "metadata": {
              "name": "web-scrape-llm-rate-jsonl",
              "description": (
                  f"Scrape each URL, rate the page with an LLM, drop the raw HTML/"
                  f"fingerprint, and write ratings to {slots['output_path']}"
              ),
          },
      }
  ```

- [ ] **Step 3: Register the `RecipeSpec` in `_RECIPES` (minimal impl).**
  In `recipes.py`, add to the `_RECIPES` dict (after the
  `fork-coalesce-truncate-jsonl` entry at :594-609, before the closing `}`):
  ```python
      "web-scrape-llm-rate-jsonl": RecipeSpec(
          name="web-scrape-llm-rate-jsonl",
          description=(
              "Fetch each URL in a blob of {url: ...} rows, rate the page with an "
              "LLM, drop the raw scraped HTML and fingerprint, and write a JSONL "
              "output of url + rating. Use for: 'scrape these pages and rate them', "
              "'fetch each site and score it'. The URL list must already be uploaded "
              "as a session blob (json or csv rows with a url column). The raw-HTML "
              "cleanup is staged as a pipeline_decision so the data-minimization "
              "contract passes deterministically."
          ),
          slots=_RECIPE_WEB_SCRAPE_SLOTS,
          build=_build_web_scrape_recipe,
      ),
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_recipes.py::TestWebScrapeRecipeBuild tests/unit/web/composer/test_recipes.py::TestRecipeRegistry -q
  ```
  Expected: all pass.

- [ ] **Step 4: Verify `match_recipe` end-to-end no longer raises (run-to-pass).**
  Add a regression to `test_recipe_match.py`:
  ```python
  def test_match_recipe_returns_web_scrape_match_for_url_source() -> None:
      """End-to-end: now that the RecipeSpec is registered, match_recipe returns
      a RecipeMatch (no InvariantError) for the canonical URL-row source."""
      from elspeth.web.composer.guided.recipe_match import match_recipe

      source = _make_url_json_source()
      sink = _make_single_jsonl_sink()
      result = match_recipe(source, sink)
      assert result is not None
      assert result.recipe_name == "web-scrape-llm-rate-jsonl"
      assert result.slots["source_blob_id"] == source.options["blob_ref"]
      # model/api_key_secret remain unsatisfied (operator fills them via recipe_offer).
      assert "model" in result.unsatisfied_slots
      assert "api_key_secret" in result.unsatisfied_slots
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_match_recipe_returns_web_scrape_match_for_url_source -q
  ```
  Expected: `1 passed`.

- [ ] **Step 4b: Trace the REAL canonical-seed materialised shape through `match_recipe`
  (run-to-pass).** Step 4 used the abstract `_make_url_json_source()` fixture; this test
  pins that the fixture's shape is the one the backend *actually* produces for the
  canonical `inline_blob` seed, so the §4.1 "zero-LLM canonical compose via recipe-match"
  lever provably fires for the real tutorial source. The canonical seed
  (`tutorial_cache.CANONICAL_SEED_PROMPT`; frontend `tutorial.spec.ts:56` `plugin:
  "inline_blob"`, rows of `{url: ...}`) is composed via `set_pipeline` with
  `source.inline_blob`; the backend persists the inline blob and binds a registered
  source plugin via `_MIME_TO_SOURCE` (`application/json → json`) **and unconditionally
  writes `source.options["blob_ref"]`** (`composer/tools/sessions.py:420-427`, inside the
  `if inline_blob is not None` branch). So the materialised `SourceResolved` is
  `plugin="json"`, `options={"blob_ref": <uuid>, "path": ...}`, observed `url` column —
  exactly the `_make_url_json_source()` shape. Append to `test_recipe_match.py`:
  ```python
  def test_canonical_seed_materialised_source_matches_web_scrape_recipe() -> None:
      """§4.1 zero-LLM lever: the REAL canonical tutorial seed, materialised by
      set_pipeline(source.inline_blob), matches the web_scrape recipe.

      Provenance pin — the materialised source shape this test encodes is what
      ``_execute_set_pipeline`` produces for the canonical ``inline_blob`` URL seed
      (``composer/tools/sessions.py:420-427``): an ``application/json`` inline blob
      binds the registered ``json`` source plugin via ``_MIME_TO_SOURCE``
      (``composer/tools/sources.py:146``) AND writes ``source.options["blob_ref"]``
      = the persisted blob UUID UNCONDITIONALLY in the ``if inline_blob is not None``
      branch. So ``SourceResolved.plugin == "json"`` (never the ``"inline_blob"``
      authoring alias, never ``"web_scrape"``) and ``blob_ref`` IS present at
      ``match_recipe`` time — the predicate's blob-presence gate is satisfied and the
      recipe fires. If this assertion ever flips to None, the zero-LLM canonical
      compose is broken; do NOT relax the predicate — fix the materialisation or the
      fixture so the two agree.
      """
      from elspeth.web.composer.guided.recipe_match import match_recipe

      # The materialised canonical source, byte-faithful to sessions.py:420-427:
      # json plugin + path + blob_ref overlay + observed url column.
      canonical_source = SourceResolved(
          plugin="json",  # _MIME_TO_SOURCE["application/json"] -> "json"
          options={
              "path": "composer_blobs/canonical-url-list.json",
              "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",  # sessions.py:425
          },
          observed_columns=("url",),
          sample_rows=({"url": "https://www.dta.gov.au"},),
      )
      canonical_sink = _make_single_jsonl_sink()

      result = match_recipe(canonical_source, canonical_sink)
      assert result is not None, "canonical seed must match the web_scrape recipe (zero-LLM §4.1)"
      assert result.recipe_name == "web-scrape-llm-rate-jsonl"
      # The slot resolver derives source_blob_id from the materialised blob_ref.
      assert result.slots["source_blob_id"] == canonical_source.options["blob_ref"]
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/guided/test_recipe_match.py::test_canonical_seed_materialised_source_matches_web_scrape_recipe -q
  ```
  Expected: `1 passed`.

- [ ] **Step 5: Lint + mypy + commit.**
  ```
  uv run python -m ruff check src/elspeth/web/composer/recipes.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m ruff format src/elspeth/web/composer/recipes.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  uv run python -m mypy src/elspeth/web/composer/recipes.py
  git add src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py
  git commit -m "feat(composer/recipes): add web-scrape-llm-rate-jsonl recipe (D11/P4.2)

Deterministically emits json-url-source → web_scrape → llm → field_mapper
(select_only cleanup, drops raw content/fingerprint) → jsonl, naming the head
source node and staging the kind=pipeline_decision raw-HTML cleanup so the
blocking cleanup contract passes. Omits the unbuildable azure_prompt_shield
hard node (elspeth-abb2cb0931) without suppressing the existing prompt-shield
advisory. Wires the slot resolver into the predicate registry.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff + mypy clean, commit succeeds.

---

### Task P4.3: Contract + shield-advisory + data-minimization integration test (CompositionState)

**Files:**
- Create `tests/unit/web/composer/test_web_scrape_recipe_contract.py`

**Interfaces:**
- Consumes: `_build_web_scrape_recipe` (P4.2), `CompositionState` /
  `SourceSpec` / `NodeSpec` / `OutputSpec` / `PipelineMetadata`
  (`composer/state.py`), `composition_review_contract_error` +
  `prompt_shield_recommendation_warning_pairs` (`web/interpretation_state.py`),
  `interpretation_sites` (`interpretation_state.py:366`).
- Produces: a state-level pin that the **built recipe satisfies the blocking
  cleanup contract** AND **preserves the live prompt-shield advisory** AND
  **drops raw fields** — the load-bearing rev-4 re-polarized shield test.

> **Why a hand-built `CompositionState` (not `apply_recipe`).** `apply_recipe` →
> `_execute_set_pipeline` requires session+blob context to resolve `blob_id`. The
> contract/validate logic operates on `CompositionState`, so this test reconstructs
> the recipe's node graph as a `CompositionState` (mirroring
> `tests/unit/web/test_interpretation_state.py:85` `_state_with_web_scrape_cleanup_node`)
> and asserts directly. The set_pipeline-args shape (P4.2 tests) and this state-shape
> test together cover both halves.

- [ ] **Step 1: Write the state-builder helper + contract-pass test (run-to-fail).**
  Create the file:
  ```python
  """State-level contract + shield-advisory tests for the web-scrape recipe (D11).

  Pins the rev-4 re-polarized shield behaviour: the built pipeline omits the
  azure_prompt_shield HARD NODE but the medium-severity prompt-shield ADVISORY
  warning IS present (elspeth-abb2cb0931 is a conditional 'restore once plugins
  gate on secret availability' ticket, NOT a licence to hide the signal).
  """

  from __future__ import annotations

  from typing import Any

  from elspeth.web.composer.recipes import _build_web_scrape_recipe
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )
  from elspeth.web.interpretation_state import (
      composition_review_contract_error,
      interpretation_sites,
      prompt_shield_recommendation_warning_pairs,
  )

  _SLOTS = {
      "source_blob_id": "a1b2c3d4-0000-0000-0000-0000000000aa",
      "model": "anthropic/claude-sonnet-4.6",
      "api_key_secret": "OPENROUTER_API_KEY",
      "output_path": "outputs/ratings.jsonl",
  }


  def _node_from_args(node_args: dict[str, Any]) -> NodeSpec:
      return NodeSpec(
          id=node_args["id"],
          node_type=node_args["node_type"],
          plugin=node_args["plugin"],
          input=node_args["input"],
          on_success=node_args.get("on_success"),
          on_error=node_args.get("on_error"),
          options=node_args["options"],
          condition=None,
          routes=None,
          fork_to=None,
          branches=None,
          policy=None,
          merge=None,
      )


  def _state_from_recipe() -> CompositionState:
      args = _build_web_scrape_recipe(_SLOTS)
      src = args["source"]
      source = SourceSpec(
          plugin=src["plugin"],
          on_success=src["on_success"],
          options=src["options"],
          on_validation_failure=src["on_validation_failure"],
      )
      out = args["outputs"][0]
      output = OutputSpec(
          name=out["sink_name"],
          plugin=out["plugin"],
          options=out["options"],
          on_write_failure=out["on_write_failure"],
      )
      return CompositionState(
          source=source,
          nodes=tuple(_node_from_args(n) for n in args["nodes"]),
          edges=(),
          outputs=(output,),
          metadata=PipelineMetadata(),
          version=1,
      )


  def test_built_recipe_passes_blocking_cleanup_contract() -> None:
      """The staged pipeline_decision satisfies raw_html_cleanup_review_contract_error,
      so composition_review_contract_error (tools/sessions.py:657) is None."""
      state = _state_from_recipe()
      assert composition_review_contract_error(state) is None
  ```
  Run-to-fail (the helper imports may need adjusting if `OutputSpec`/`SourceSpec`
  fields differ — confirm at write time against `state.py:119/287`):
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py::test_built_recipe_passes_blocking_cleanup_contract -q
  ```
  Expected: PASS once the helper compiles. If `composition_review_contract_error`
  returns a non-None string, the field_mapper requirement staging in P4.2 is wrong
  — fix `_build_web_scrape_recipe` (do NOT weaken the test).

- [ ] **Step 2: Write the re-polarized shield-advisory PRESENCE test (run-to-fail).**
  Append:
  ```python
  def test_prompt_shield_advisory_is_present_no_hard_node() -> None:
      """Re-polarized shield (rev 4): assert (a) no azure_prompt_shield HARD NODE,
      AND (b) the medium-severity prompt-shield ADVISORY warning IS present — pin
      the PRESENCE of the security signal, not its absence. The flagship example
      must not be the one web_scrape→llm pipeline that hides the warning the rest
      of the system shows.

      See elspeth-abb2cb0931 (conditional 'restore the shield advice once plugins
      gate on secret availability' ticket).
      """
      state = _state_from_recipe()

      # (a) no unbuildable hard node
      assert all(node.plugin != "azure_prompt_shield" for node in state.nodes)

      # (b) the advisory IS present (web_scrape → llm without a shield)
      warning_pairs = prompt_shield_recommendation_warning_pairs(state)
      assert warning_pairs, "expected the medium-severity prompt-shield advisory to fire"
      components = {component for component, _message in warning_pairs}
      assert "node:rate_pages" in components

  def test_prompt_shield_advisory_surfaces_in_validate_warnings() -> None:
      """The same advisory rides validate().warnings at 'medium' severity
      (state.py:2410), which is the payload the wire stage renders
      (_authoring_validation_payload['warnings'], tools/sessions.py:1157)."""
      state = _state_from_recipe()
      summary = state.validate()
      shield_warnings = [
          w for w in summary.warnings
          if "prompt-injection shield" in w.message and w.severity == "medium"
      ]
      assert shield_warnings, "prompt-shield advisory must ride validate().warnings at medium severity"
  ```
  Run-to-fail then run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py -q
  ```
  Expected: PASS. If the advisory is absent, the recipe accidentally suppressed it
  (e.g. wrong plugin name on the rating node, or it staged a
  prompt_injection_shield_recommendation requirement) — fix the recipe so the
  advisory fires; **do not** make the test green by asserting absence.
  (Confirm `ValidationEntry` exposes `.message` / `.severity` at write time; if the
  field is `.text`, adjust the comprehension accordingly.)

- [ ] **Step 3: Write the data-minimization site test (run-to-fail).**
  Append:
  ```python
  def test_cleanup_node_drops_raw_fields_no_orphan_site() -> None:
      """Data minimization: the raw content/fingerprint fields are NOT preserved,
      and because the pipeline_decision is staged, no missing-cleanup interpretation
      site orphans (interpretation_sites returns no raw-html-cleanup site for the
      field_mapper)."""
      state = _state_from_recipe()
      fm = next(n for n in state.nodes if n.plugin == "field_mapper")
      mapping = fm.options["mapping"]
      preserved = set(mapping) | set(mapping.values())
      assert "content" not in preserved
      assert "content_fingerprint" not in preserved

      from elspeth.contracts.composer_interpretation import InterpretationKind

      sites = interpretation_sites(state)
      raw_html_sites = [
          s for s in sites
          if s.kind is InterpretationKind.PIPELINE_DECISION and s.user_term == "drop_raw_html_fields"
      ]
      assert raw_html_sites == [], "raw-html cleanup must be staged (not orphaned)"
  ```
  Run-to-pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_contract.py -q
  ```
  Expected: all pass.

- [ ] **Step 4: Lint + commit.**
  ```
  uv run python -m ruff check tests/unit/web/composer/test_web_scrape_recipe_contract.py
  uv run python -m ruff format tests/unit/web/composer/test_web_scrape_recipe_contract.py
  git add tests/unit/web/composer/test_web_scrape_recipe_contract.py
  git commit -m "test(composer/recipes): pin web-scrape recipe contract + re-polarized shield (D11/P4.3)

Builds the recipe into a CompositionState and asserts: (1) the staged
pipeline_decision satisfies the blocking raw-HTML cleanup contract;
(2) re-polarized shield — NO azure_prompt_shield hard node BUT the
medium-severity prompt-shield advisory IS present (presence, not absence;
refs elspeth-abb2cb0931); (3) data minimization — raw content/fingerprint
dropped with no orphaned interpretation site.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.4: Zero-LLM compose assertion (recipe-apply makes no provider call)

**Files:**
- Create `tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py`

**Interfaces:**
- Consumes: `_build_web_scrape_recipe` / `apply_recipe` (P4.2),
  `validate_slots` (`recipes.py:119`). Patch target for the provider chokepoint:
  `elspeth.web.composer.service._litellm_acompletion` (the single LiteLLM
  chokepoint, per the existing `test_compose_loop_llm_audit.py:186` pattern).
- Produces: the gate for the §4.1 "zero-LLM canonical compose" claim — the recipe
  build path performs **no** LLM provider call.

> **Scope note.** P4 owns the recipe build, not the dispatch wiring (P2). This test
> proves the **recipe-build path itself** is provider-free: building the
> set_pipeline args and validating the slots calls the LLM zero times. The full
> dispatch-level zero-LLM assertion (`tutorial/run` cache freeze) is asserted by P7
> once the recipe-apply seam redirects through STEP_4_WIRE.

- [ ] **Step 1: Write the zero-LLM build test (run-to-fail).**
  Create the file:
  ```python
  """Zero-LLM compose gate for the web-scrape recipe (D11, §4.1).

  Building the canonical recipe is a pure, deterministic function: it must make
  ZERO LLM provider calls. Pins the §4.1 claim that the canonical pipeline
  composes with no frontier round-trip at recipe-build time.
  """

  from __future__ import annotations

  from unittest.mock import AsyncMock, patch
  from uuid import uuid4

  from elspeth.web.composer.recipes import _build_web_scrape_recipe, apply_recipe

  _SLOTS = {
      "source_blob_id": str(uuid4()),
      "model": "anthropic/claude-sonnet-4.6",
      "api_key_secret": "OPENROUTER_API_KEY",
      "output_path": "outputs/ratings.jsonl",
  }


  def test_build_web_scrape_recipe_makes_zero_llm_calls() -> None:
      with patch(
          "elspeth.web.composer.service._litellm_acompletion",
          new_callable=AsyncMock,
      ) as mock_acomp:
          args = _build_web_scrape_recipe(_SLOTS)
          # llm node IS present in the COMPOSED pipeline (it runs at RUN time,
          # not compose time) — but the build itself called no provider.
          assert any(n["plugin"] == "llm" for n in args["nodes"])
      assert mock_acomp.call_count == 0


  def test_apply_web_scrape_recipe_makes_zero_llm_calls() -> None:
      with patch(
          "elspeth.web.composer.service._litellm_acompletion",
          new_callable=AsyncMock,
      ) as mock_acomp:
          args = apply_recipe("web-scrape-llm-rate-jsonl", _SLOTS)
          assert args["metadata"]["name"] == "web-scrape-llm-rate-jsonl"
      assert mock_acomp.call_count == 0
  ```
  Run-to-fail/pass:
  ```
  uv run python -m pytest tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py -q
  ```
  Expected: PASS (the build is pure). If the patch target import path is wrong,
  fix the dotted path against `test_compose_loop_llm_audit.py:186`; do not skip.

- [ ] **Step 2: Lint + commit.**
  ```
  uv run python -m ruff check tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  uv run python -m ruff format tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  git add tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py
  git commit -m "test(composer/recipes): zero-LLM gate for web-scrape recipe build (D11/P4.4)

Pins §4.1: building/applying the canonical recipe makes zero _litellm_acompletion
calls. The llm node runs at RUN time, never at compose time.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: ruff clean, commit succeeds.

---

### Task P4.5: Phase gate sweep + plugin-hash refresh

**Files:**
- Modify (if the CI plugin-hash gate flags it) `tests`/baseline only — no source
  change expected (P4 edits no plugin file; it edits `recipes.py` + `recipe_match.py`
  which are composer modules, not registered plugins).

**Interfaces:** none new.

- [ ] **Step 1: Run the full P4 test slice (run-to-pass).**
  ```
  uv run python -m pytest \
    tests/unit/web/composer/guided/test_recipe_match.py \
    tests/unit/web/composer/test_recipes.py \
    tests/unit/web/composer/test_web_scrape_recipe_contract.py \
    tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py \
    -q
  ```
  Expected: all pass, `0 failed`.

- [ ] **Step 2: Lint + format + mypy over the touched source (run-to-pass).**
  ```
  uv run python -m ruff check src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  uv run python -m ruff format --check src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  uv run python -m mypy src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py
  ```
  Expected: `All checks passed!`, format clean, mypy `Success: no issues found`.

- [ ] **Step 3: trust-tier + wardline gate (recipe stages Tier-3 review text + LLM options).**
  The recipe authors `interpretation_requirements` (review text) and an
  `api_key` `{secret_ref}` — confirm no new trust-tier displacement and no taint
  finding:
  ```
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,'composer/*' --root src/elspeth/web/composer/recipes.py
  wardline scan src/elspeth/web/composer/recipes.py src/elspeth/web/composer/guided/recipe_match.py --fail-on ERROR
  ```
  Expected: elspeth-lints check passes (the recipe builder is a pure function over
  already-validated slots — `apply_recipe` runs `validate_slots` first, so no new
  Tier-3 boundary is introduced); `wardline` exit 0. If trust-tier reports drift,
  it is a pre-existing operator-owned HMAC re-pin (state it in the commit, do not
  sign blind — see CLAUDE.md gate-debt doctrine).

- [ ] **Step 4: Final phase commit (only if Step 3 produced allowlist/test churn).**
  ```
  git add -A
  git commit -m "chore(composer/recipes): P4 gate sweep — web-scrape recipe slice green

Full P4 recipe slice (predicate + RecipeSpec + contract + zero-LLM) passes;
ruff/mypy clean; trust-tier + wardline checked. No plugin file edited (recipes.py
and recipe_match.py are composer modules, not registered plugins — no
source_file_hash refresh owed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
  Expected: commit succeeds (or "nothing to commit" if Step 3 was clean — that is
  acceptable, the slice is already committed by P4.1–P4.4).

---

## Phase P5 — Advisor sign-off gate (B3/D13) — profile-gated + UNAVAILABLE escape

> **Cross-phase dependencies (consumed by name, not created here):**
> - `WorkflowProfile` (+ `EMPTY_PROFILE`, `TUTORIAL_PROFILE`) and `GuidedSession.profile`
>   / `GuidedSession.advisor_checkpoint_passes_used` — **owned by P0**
>   (`composer/guided/profile.py`, `composer/guided/state_machine.py`).
> - `GuidedStep.STEP_4_WIRE`, `TurnType.CONFIRM_WIRING` — **owned by P1**
>   (`composer/guided/protocol.py`).
> - `handle_step_4_wire_confirm(...)` step handler — **owned by P1.6**
>   (`composer/guided/steps.py`).
> - The `STEP_4_WIRE` dispatch branch in `_dispatch_guided_respond` (the seam this
>   phase mutates) — **owned by P2.9** (`sessions/routes/_helpers.py`); P2.9 also
>   adds the post-accept wire-turn emission + the GET `/guided` rebuild branch.
> - The `build_step_4_wire_turn` emitter — **owned by P2.4**
>   (`composer/guided/emitters.py`); P2.4 lands the FINAL signature (state +
>   optional `catalog`/`advisor_findings`/`signoff_outcome`) so no P5 task re-signs it.
>
> Each task below imports those symbols verbatim. If a dependency symbol is not yet
> present in the worktree when a task runs, the run-to-fail step will fail on
> `ImportError`/`AttributeError` (that is the expected first failure and is called out
> per task); the symbol arrives from its owning phase before the run-to-pass step.

---

### Task P5.1: Add the public `run_signoff_checkpoint` Protocol method

**Files:**
- Modify: `src/elspeth/web/composer/protocol.py` (`ComposerService` Protocol, after `compose` at :713; add a TYPE_CHECKING import for `AdvisorCheckpointVerdict` + `BufferingRecorder`)
- Create: `tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py`

**Interfaces:**
- Produces (Protocol method, verbatim signature):
  `async def run_signoff_checkpoint(self, *, state: CompositionState, session_id: str | None, recorder: BufferingRecorder | None, progress: ComposerProgressSink | None = None) -> AdvisorCheckpointVerdict`
- Consumes: `CompositionState` (`composer/state.py`), `ComposerProgressSink` (already imported in protocol.py:19), `AdvisorCheckpointVerdict` + `BufferingRecorder` (TYPE_CHECKING).

- [ ] **Step 1: Write the failing test that the Protocol declares the method.**
  Create `tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py`:
  ```python
  """Phase P5.1 — the public advisor sign-off checkpoint Protocol method."""

  from __future__ import annotations

  import inspect

  from elspeth.web.composer.protocol import ComposerService


  def test_protocol_declares_run_signoff_checkpoint() -> None:
      assert hasattr(ComposerService, "run_signoff_checkpoint")
      sig = inspect.signature(ComposerService.run_signoff_checkpoint)
      params = sig.parameters
      # keyword-only contract (verbatim names)
      assert params["state"].kind is inspect.Parameter.KEYWORD_ONLY
      assert params["session_id"].kind is inspect.Parameter.KEYWORD_ONLY
      assert params["recorder"].kind is inspect.Parameter.KEYWORD_ONLY
      assert params["progress"].kind is inspect.Parameter.KEYWORD_ONLY
      assert params["progress"].default is None
      assert inspect.iscoroutinefunction(ComposerService.run_signoff_checkpoint)
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py -q`
  Expected failure: `AttributeError: type object 'ComposerService' has no attribute 'run_signoff_checkpoint'` (the `hasattr` assertion fails).
- [ ] **Step 3: Add the TYPE_CHECKING imports to protocol.py.**
  In `src/elspeth/web/composer/protocol.py`, the `TYPE_CHECKING` block currently reads:
  ```python
  if TYPE_CHECKING:
      from elspeth.web.composer.guided.state_machine import TerminalState
  ```
  Replace it with:
  ```python
  if TYPE_CHECKING:
      from elspeth.web.composer.audit import BufferingRecorder
      from elspeth.web.composer.guided.state_machine import TerminalState
      from elspeth.web.composer.service import AdvisorCheckpointVerdict
  ```
- [ ] **Step 4: Add the Protocol method.**
  In `src/elspeth/web/composer/protocol.py`, immediately AFTER the `compose(...)` method's closing docstring/`"""` and BEFORE `async def explain_run_diagnostics`, insert:
  ```python
      async def run_signoff_checkpoint(
          self,
          *,
          state: CompositionState,
          session_id: str | None,
          recorder: BufferingRecorder | None,
          progress: ComposerProgressSink | None = None,
      ) -> AdvisorCheckpointVerdict:
          """Run the deterministic END advisor sign-off checkpoint (phase='end').

          Public façade over the private ``_run_advisor_checkpoint(phase='end')``
          so the guided STEP_4_WIRE dispatcher — which holds a ``ComposerService``
          handle but not the impl's private methods — can request the whole-
          pipeline structural sign-off. Non-raising: a sustained provider failure
          yields ``ok=False`` (unavailable); a FLAGGED sign-off yields
          ``blocking=True``; CLEAN yields ``ok=True, blocking=False``. The caller
          (the wire branch) maps the verdict to terminal/redirect per D13.

          ``recorder`` threads the advisor call's audit sidecar; ``progress``
          (when set) receives a ``calling_model`` event before the call.
          """
          ...
  ```
- [ ] **Step 5: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py -q`
  Expected: `1 passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py && git commit -m "feat(composer): declare public run_signoff_checkpoint Protocol method (P5.1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.2: Implement `run_signoff_checkpoint` on `ComposerServiceImpl` (delegation)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`ComposerServiceImpl`, add method directly above `_run_advisor_checkpoint` at :4176)
- Create: `tests/unit/web/composer/test_run_signoff_checkpoint_impl.py`

**Interfaces:**
- Produces (public impl method, delegates):
  `async def run_signoff_checkpoint(self, *, state, session_id, recorder, progress=None) -> AdvisorCheckpointVerdict`
  → `return await self._run_advisor_checkpoint(phase="end", state=state, session_id=session_id, recorder=recorder, progress=progress)`
- Consumes: existing private `ComposerServiceImpl._run_advisor_checkpoint` (service.py:4176), `AdvisorCheckpointVerdict` (service.py:4664).

- [ ] **Step 1: Write the failing delegation test.**
  Create `tests/unit/web/composer/test_run_signoff_checkpoint_impl.py`:
  ```python
  """Phase P5.2 — run_signoff_checkpoint delegates to _run_advisor_checkpoint(phase='end')."""

  from __future__ import annotations

  from pathlib import Path
  from unittest.mock import AsyncMock, MagicMock

  import pytest

  from elspeth.web.catalog.protocol import CatalogService
  from elspeth.web.catalog.schemas import PluginSchemaInfo, PluginSummary
  from elspeth.web.composer.audit import BufferingRecorder
  from elspeth.web.composer.service import AdvisorCheckpointVerdict, ComposerServiceImpl
  from elspeth.web.composer.state import (
      CompositionState,
      NodeSpec,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )
  from elspeth.web.config import WebSettings


  def _mock_catalog() -> MagicMock:
      catalog = MagicMock(spec=CatalogService)
      catalog.list_sources.return_value = [
          PluginSummary(name="csv", description="CSV", plugin_type="source", config_fields=[]),
      ]
      catalog.list_transforms.return_value = []
      catalog.list_sinks.return_value = []
      catalog.get_schema.return_value = PluginSchemaInfo(
          name="csv",
          plugin_type="source",
          description="CSV source",
          json_schema={"type": "object", "properties": {}},
          knob_schema={"fields": []},
      )
      return catalog


  def _make_settings() -> WebSettings:
      return WebSettings(
          data_dir=Path("/data"),
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          composer_advisor_max_calls_per_compose=4,
          composer_advisor_timeout_seconds=60.0,
          shareable_link_signing_key=b"\x00" * 32,
      )


  def _state() -> CompositionState:
      return CompositionState(
          source=SourceSpec(plugin="csv", on_success="rows", options={"path": "in.csv"}, on_validation_failure="discard"),
          nodes=(
              NodeSpec(
                  id="rate", node_type="transform", plugin="llm", input="rows", on_success="rated",
                  on_error=None, options={"model": "gpt-5.5"}, condition=None, routes=None, fork_to=None,
                  branches=None, policy=None, merge=None,
              ),
          ),
          edges=(),
          outputs=(OutputSpec(name="rated", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard"),),
          metadata=PipelineMetadata(),
          version=2,
      )


  @pytest.mark.asyncio
  async def test_run_signoff_delegates_to_end_checkpoint() -> None:
      service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
      verdict = AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN: looks good")
      service._run_advisor_checkpoint = AsyncMock(return_value=verdict)
      recorder = BufferingRecorder()

      async def sink(event: object) -> None:
          return None

      out = await service.run_signoff_checkpoint(state=_state(), session_id="s1", recorder=recorder, progress=sink)

      assert out is verdict
      service._run_advisor_checkpoint.assert_awaited_once()
      kwargs = service._run_advisor_checkpoint.await_args.kwargs
      assert kwargs["phase"] == "end"
      assert kwargs["session_id"] == "s1"
      assert kwargs["recorder"] is recorder
      assert kwargs["progress"] is sink


  @pytest.mark.asyncio
  async def test_run_signoff_progress_defaults_none() -> None:
      service = ComposerServiceImpl(catalog=_mock_catalog(), settings=_make_settings())
      service._run_advisor_checkpoint = AsyncMock(
          return_value=AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="TimeoutError: x")
      )
      await service.run_signoff_checkpoint(state=_state(), session_id=None, recorder=None)
      assert service._run_advisor_checkpoint.await_args.kwargs["progress"] is None
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_impl.py -q`
  Expected failure: `AttributeError: 'ComposerServiceImpl' object has no attribute 'run_signoff_checkpoint'`.
- [ ] **Step 3: Add the delegation method.**
  In `src/elspeth/web/composer/service.py`, directly ABOVE `async def _run_advisor_checkpoint(` (currently at :4176), insert:
  ```python
      async def run_signoff_checkpoint(
          self,
          *,
          state: CompositionState,
          session_id: str | None,
          recorder: BufferingRecorder | None,
          progress: ComposerProgressSink | None = None,
      ) -> AdvisorCheckpointVerdict:
          """Public END sign-off checkpoint (ComposerService Protocol, P5).

          Thin delegation to the private deterministic END checkpoint so the
          guided STEP_4_WIRE dispatcher can request the whole-pipeline sign-off
          through the ``ComposerService`` handle it holds. The private method
          owns the build-arguments / bounded-retry / verdict-mapping logic; this
          façade adds nothing but the public name so the trust boundary and the
          backend-produced (Tier-1) ``schema_excerpt`` path are unchanged — no
          unvalidated user text is ever forwarded here.
          """
          return await self._run_advisor_checkpoint(
              phase="end",
              state=state,
              session_id=session_id,
              recorder=recorder,
              progress=progress,
          )
  ```
  (Note: `BufferingRecorder`, `ComposerProgressSink`, `CompositionState`, and `AdvisorCheckpointVerdict` are all already in scope in service.py — `BufferingRecorder` via the `composer.audit` import at :62, `AdvisorCheckpointVerdict` is defined in-module at :4664, `ComposerProgressSink`/`CompositionState` are module imports used throughout.)
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_impl.py -q`
  Expected: `2 passed`.
- [ ] **Step 5: Run the existing advisor-checkpoint suite to confirm no regression.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py -q`
  Expected: all existing tests still `passed` (the impl only adds a method).
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_run_signoff_checkpoint_impl.py && git commit -m "feat(composer): implement run_signoff_checkpoint delegating to END checkpoint (P5.2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.3: Add the verdict-class classifier + redirect helper (`classify_signoff_verdict`)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — add a `failure_class` field to `AdvisorCheckpointVerdict` (:4664) and set it in `_run_advisor_checkpoint`'s `except` path (:4207-4230). See Step 1a. WITHOUT this the classifier below cannot tell a malformed response from a transport outage (both are currently `ok=False`), so malformed would fail OPEN into the escape — the D4/B2 defect.
- Create: `src/elspeth/web/composer/guided/signoff.py` (new pure module — no `self`, importable by both `_helpers.py` and tests; mirrors how `_advisor_signoff_blocked_validation` is module-scope data with no `self` dependency)
- Create: `tests/unit/web/composer/guided/test_signoff_classifier.py`

**Interfaces:**
- Produces (canonical names, verbatim):
  - `class SignoffOutcome(StrEnum)` with members `COMPLETE = "complete"`, `REVISE = "revise"`, `BLOCKED_FLAGGED = "blocked_flagged"`, `BLOCKED_UNAVAILABLE = "blocked_unavailable"`, `ESCAPE_UNAVAILABLE = "escape_unavailable"`.
  - `@dataclass(frozen=True, slots=True) class SignoffDecision` with fields `outcome: SignoffOutcome`, `reason: str | None`, `findings_text: str`, `passes_delta: int`.
  - `def classify_signoff_verdict(verdict: AdvisorCheckpointVerdict, *, passes_used: int, max_passes: int) -> SignoffDecision`.
- Consumes: `AdvisorCheckpointVerdict` (`composer/service.py:4664`), now carrying `failure_class: Literal["none","unavailable","malformed"]` (added in Step 1a).

Decision logic (D13 matrix; `is_last_pass = (passes_used + 1) >= max_passes`). CRITICAL: `_run_advisor_checkpoint` collapses EVERY exception to `ok=False` (including a re-raised malformed/parse error), so the classifier MUST split `not verdict.ok` on `verdict.failure_class` — `(ok, blocking)` alone cannot tell malformed (fail-closed) from a transport outage (escapable):
- CLEAN (`verdict.ok and not verdict.blocking`) → `COMPLETE` (reason `None`, `passes_delta=1`).
- FLAGGED (`verdict.ok and verdict.blocking`) — a quality verdict, fail-closed:
  - not last pass → `REVISE` (`passes_delta=1`, findings carried).
  - last pass → `BLOCKED_FLAGGED` (`reason="exhausted"`, `passes_delta=1`).
- MALFORMED (`not verdict.ok and verdict.failure_class == "malformed"`) — fail-closed exactly like FLAGGED, **never** the escape:
  - not last pass → `REVISE`; last pass → `BLOCKED_FLAGGED` (`reason="exhausted"`).
- UNAVAILABLE (`not verdict.ok and verdict.failure_class == "unavailable"`) — genuine outage:
  - not last pass → `REVISE` (`passes_delta=1`, findings carried — re-emit "advisor could not be reached; retry").
  - last pass → `ESCAPE_UNAVAILABLE` (`reason="unavailable"`, `passes_delta=1`) — the differentiated audited escape is *offered*; whether the caller stamps COMPLETED-without-signoff vs `BLOCKED_UNAVAILABLE` depends on the user's acknowledgement (Task P5.5). `BLOCKED_UNAVAILABLE` is the not-acknowledged terminal.

- [ ] **Step 1a: Add `failure_class` to `AdvisorCheckpointVerdict` + classify the exception in `_run_advisor_checkpoint` (`service.py`).**
  In `AdvisorCheckpointVerdict` (`service.py:4664`) add a defaulted field (`Literal` is already imported in service.py):
  ```python
      failure_class: Literal["none", "unavailable", "malformed"] = "none"
  ```
  The default keeps every existing construction valid (CLEAN/FLAGGED set `ok=True` and never read it). In `_run_advisor_checkpoint` (`service.py:4207-4230`) the `except Exception` retry loop currently returns `ok=False` for EVERY exception class — replace the final exhausted-retries return with a classified one:
  ```python
      # The call core re-raises typed LLM errors. A parse/validation/shape error is a
      # MALFORMED verdict (fail-closed at the END gate, D13); timeout/auth/transport is
      # a genuine UNAVAILABLE outage (escapable at budget-exhaustion). Unrecognised
      # errors default to the SAFER class (malformed) — fail-closed by default.
      _unavailable = (TimeoutError, ConnectionError)
      _malformed = (ValueError, KeyError, TypeError, AttributeError)  # JSONDecodeError is a ValueError
      if isinstance(last_exc, _unavailable) or type(last_exc).__name__ in {
          "APITimeoutError", "APIConnectionError", "AuthenticationError", "RateLimitError",
      }:
          failure_class = "unavailable"
      elif isinstance(last_exc, _malformed):
          failure_class = "malformed"
      else:
          failure_class = "malformed"  # fail-closed default for an unclassified error
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          failure_class=failure_class,
          findings_text=f"{type(last_exc).__name__}: {last_exc}" if last_exc else "advisor unavailable",
      )
  ```
  Resolve the EXACT provider exception types against the live LLM client when implementing — the name set above is a transport-class allowlist; everything not on it fails closed as `malformed`. Update `tests/unit/web/composer/test_advisor_checkpoint.py` if it constructs `AdvisorCheckpointVerdict` positionally.
- [ ] **Step 1b: Confirm the verdict field exists.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -c "from elspeth.web.composer.service import AdvisorCheckpointVerdict; print(AdvisorCheckpointVerdict(ok=False, blocking=False, failure_class='malformed', findings_text='x').failure_class)"`
  Expected: prints `malformed`.
- [ ] **Step 1: Write the failing classifier test.**
  Create `tests/unit/web/composer/guided/test_signoff_classifier.py`:
  ```python
  """Phase P5.3 — pure D13 verdict-class classifier for the wire-stage sign-off."""

  from __future__ import annotations

  from elspeth.web.composer.guided.signoff import (
      SignoffOutcome,
      classify_signoff_verdict,
  )
  from elspeth.web.composer.service import AdvisorCheckpointVerdict


  def _clean() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN: good")


  def _flagged() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: prompt sees no row field")


  def _unavailable() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(
          ok=False, blocking=False, failure_class="unavailable", findings_text="TimeoutError: deadline"
      )


  def _malformed() -> AdvisorCheckpointVerdict:
      # The advisor returned output the call core could not parse -> re-raised ->
      # caught -> ok=False with failure_class="malformed". Must FAIL CLOSED (D4/B2).
      return AdvisorCheckpointVerdict(
          ok=False, blocking=False, failure_class="malformed", findings_text="ValueError: unparseable verdict"
      )


  def test_clean_completes() -> None:
      d = classify_signoff_verdict(_clean(), passes_used=0, max_passes=3)
      assert d.outcome is SignoffOutcome.COMPLETE
      assert d.reason is None
      assert d.passes_delta == 1


  def test_flagged_revises_while_budget_remains() -> None:
      d = classify_signoff_verdict(_flagged(), passes_used=0, max_passes=3)
      assert d.outcome is SignoffOutcome.REVISE
      assert "prompt sees no row field" in d.findings_text
      assert d.passes_delta == 1


  def test_flagged_blocks_on_last_pass_no_bypass() -> None:
      d = classify_signoff_verdict(_flagged(), passes_used=2, max_passes=3)
      assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert d.reason == "exhausted"


  def test_unavailable_revises_while_budget_remains() -> None:
      d = classify_signoff_verdict(_unavailable(), passes_used=0, max_passes=3)
      assert d.outcome is SignoffOutcome.REVISE
      assert d.passes_delta == 1


  def test_unavailable_offers_escape_on_last_pass() -> None:
      d = classify_signoff_verdict(_unavailable(), passes_used=2, max_passes=3)
      assert d.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
      assert d.reason == "unavailable"


  def test_malformed_revises_while_budget_remains() -> None:
      d = classify_signoff_verdict(_malformed(), passes_used=0, max_passes=3)
      assert d.outcome is SignoffOutcome.REVISE
      assert d.passes_delta == 1


  def test_malformed_fails_closed_on_last_pass_never_escapes() -> None:
      # D4/B2 regression: a MALFORMED verdict (ok=False) must NOT take the
      # UNAVAILABLE escape — it fails closed exactly like a FLAG.
      d = classify_signoff_verdict(_malformed(), passes_used=2, max_passes=3)
      assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
      assert d.reason == "exhausted"


  def test_flagged_never_yields_an_escape() -> None:
      # A FLAG can never reach the unavailable escape — only BLOCKED_FLAGGED.
      d = classify_signoff_verdict(_flagged(), passes_used=2, max_passes=3)
      assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/guided/test_signoff_classifier.py -q`
  Expected failure: `ModuleNotFoundError: No module named 'elspeth.web.composer.guided.signoff'`.
- [ ] **Step 3: Create the classifier module.**
  Create `src/elspeth/web/composer/guided/signoff.py`:
  ```python
  """Pure D13 verdict-class classifier for the STEP_4_WIRE advisor sign-off.

  No ``self`` dependency: the wire-stage dispatcher (``_dispatch_guided_respond``)
  and the unit tests both consume :func:`classify_signoff_verdict`. It maps an
  :class:`AdvisorCheckpointVerdict` (the non-raising verdict produced by
  ``ComposerService.run_signoff_checkpoint``) to a terminal/redirect decision,
  splitting the two non-CLEAN failure CLASSES per D13:

    * a *quality* FLAG (the advisor judged the pipeline unsafe) stays fully
      fail-closed — re-emit while passes remain, then BLOCKED with no bypass;
    * a *sustained infra* UNAVAILABLE (the advisor never rendered a judgement)
      gets a differentiated audited escape on budget exhaustion, ONLY for
      ``reason="unavailable"`` and NEVER reachable from a FLAG.

  The classifier never touches the provider, never raises, and consumes no user
  text — it is a pure function of the verdict + the persisted pass budget.
  """

  from __future__ import annotations

  from dataclasses import dataclass
  from enum import StrEnum
  from typing import TYPE_CHECKING

  if TYPE_CHECKING:
      from elspeth.web.composer.service import AdvisorCheckpointVerdict


  class SignoffOutcome(StrEnum):
      """The terminal/redirect class for a wire-stage sign-off pass."""

      COMPLETE = "complete"  # CLEAN -> stamp COMPLETED
      REVISE = "revise"  # re-emit the wire turn (budget remains)
      BLOCKED_FLAGGED = "blocked_flagged"  # FLAGGED, budget exhausted -> fail-closed, no bypass
      BLOCKED_UNAVAILABLE = "blocked_unavailable"  # UNAVAILABLE escape declined -> fail-closed
      ESCAPE_UNAVAILABLE = "escape_unavailable"  # UNAVAILABLE, budget exhausted -> offer audited escape


  @dataclass(frozen=True, slots=True)
  class SignoffDecision:
      """Outcome of classifying one wire-stage advisor sign-off pass.

      ``reason`` is ``"exhausted"`` (FLAGGED, no repair left), ``"unavailable"``
      (advisor unreachable), or ``None`` (CLEAN / mid-budget REVISE). It feeds
      the blocked-result reason and the differentiated audit event. ``passes_delta``
      is always 1 — every classified pass consumed one budgeted advisor call.
      """

      outcome: SignoffOutcome
      reason: str | None
      findings_text: str
      passes_delta: int


  def classify_signoff_verdict(
      verdict: AdvisorCheckpointVerdict,
      *,
      passes_used: int,
      max_passes: int,
  ) -> SignoffDecision:
      """Map an END sign-off verdict to a D13 terminal/redirect decision.

      ``passes_used`` is the PERSISTED ``GuidedSession.advisor_checkpoint_passes_used``
      BEFORE this pass; the function computes whether this is the last budgeted pass.
      """
      is_last_pass = (passes_used + 1) >= max_passes
      findings = verdict.findings_text

      if verdict.ok and not verdict.blocking:
          # CLEAN.
          return SignoffDecision(outcome=SignoffOutcome.COMPLETE, reason=None, findings_text=findings, passes_delta=1)

      if verdict.ok and verdict.blocking:
          # FLAGGED — a quality verdict. Fail-closed, no bypass.
          if is_last_pass:
              return SignoffDecision(
                  outcome=SignoffOutcome.BLOCKED_FLAGGED, reason="exhausted", findings_text=findings, passes_delta=1
              )
          return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)

      # not verdict.ok: the advisor call did not return a usable verdict. NOTE:
      # _run_advisor_checkpoint collapses EVERY exception to ok=False (service.py:
      # 4210-4230) — INCLUDING a MALFORMED/unparseable response the call core
      # re-raised. So (ok, blocking) ALONE cannot tell malformed from a transport
      # outage; we MUST split on verdict.failure_class. D13 requires malformed to
      # FAIL CLOSED (no bypass, never the audited escape); only a genuine OUTAGE may
      # take the budget-exhausted escape.
      if verdict.failure_class == "malformed":
          # Treat exactly like FLAGGED — malformed output must NEVER be classified
          # as "advisor unreachable" and must NEVER reach ESCAPE_UNAVAILABLE.
          if is_last_pass:
              return SignoffDecision(
                  outcome=SignoffOutcome.BLOCKED_FLAGGED, reason="exhausted", findings_text=findings, passes_delta=1
              )
          return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)

      # failure_class == "unavailable": a genuine transport/auth/timeout outage.
      if is_last_pass:
          return SignoffDecision(
              outcome=SignoffOutcome.ESCAPE_UNAVAILABLE, reason="unavailable", findings_text=findings, passes_delta=1
          )
      return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)
  ```
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/guided/test_signoff_classifier.py -q`
  Expected: `8 passed`.
- [ ] **Step 5: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/guided/signoff.py tests/unit/web/composer/guided/test_signoff_classifier.py && git commit -m "feat(composer/guided): add pure D13 sign-off verdict classifier (P5.3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.4: Thread the `ComposerService` handle into `_dispatch_guided_respond`

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (`_dispatch_guided_respond` signature at :2554; add a `composer_service` param)
- Modify: `src/elspeth/web/sessions/routes/composer.py` (the `_dispatch_guided_respond(...)` call at :2333; pass `request.app.state.composer_service`)
- Create: `tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py`

**Interfaces:**
- Produces: `_dispatch_guided_respond(..., composer_service: ComposerService | None = None, advisor_checkpoint_max_passes: int | None = None)` — two new **keyword-only** params appended after `seed`, both with SAFE DEFAULTS so pre-P5 callers (the P2.9 `test_wire_dispatch.py::_dispatch` helper, which does NOT pass them) stay valid.
- Consumes: `ComposerService` Protocol (`composer/protocol.py`), `request.app.state.composer_service` (the `ComposerServiceImpl` wired at `app.py:882`), `settings.composer_advisor_checkpoint_max_passes` (available at the route call site, `composer.py:2333`).

> This task makes the wire-stage advisor gate (P5.5/P5.6) *possible* by giving the
> pure-routing dispatcher a handle to the service AND the persisted pass budget. It
> does NOT yet call the gate. `advisor_checkpoint_max_passes` is the budget P5.6
> reads as `max_passes` — threading it here (rather than reaching into
> `composer_service._settings`) keeps the dispatcher free of private-attr access.
>
> **Required-kwarg ordering (decision):** both params take safe defaults so the
> P2.9 `_dispatch` test helper (which constructs the dispatcher call BEFORE P5
> lands and passes neither param) does not break when P5.4 lands. The dispatcher
> treats `composer_service is None` as "no advisor service wired → skip the advisor
> gate" — the same behaviour as the empty / live-guided profile path, so a None
> service is a benign no-op, NOT an error. When `advisor_checkpoint_max_passes is
> None`, P5.6 resolves the budget from settings inside the dispatcher (the route
> always passes a concrete `int`; only the legacy test path leaves it None). This
> keeps the additivity-acceptance invariant: P2.9's validate-gate-only wire confirm
> still runs unchanged because the gate is skipped when no composer is threaded.

- [ ] **Step 1: Write the failing signature test.**
  Create `tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py`:
  ```python
  """Phase P5.4 — the guided dispatcher accepts a ComposerService handle + pass budget."""

  from __future__ import annotations

  import inspect

  from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond


  def test_dispatcher_accepts_composer_service_kwarg() -> None:
      sig = inspect.signature(_dispatch_guided_respond)
      assert "composer_service" in sig.parameters
      param = sig.parameters["composer_service"]
      assert param.kind is inspect.Parameter.KEYWORD_ONLY
      # Safe default so pre-P5 callers (P2.9 test_wire_dispatch.py::_dispatch)
      # that omit this kwarg stay valid; None => advisor gate skipped.
      assert param.default is None


  def test_dispatcher_accepts_advisor_checkpoint_max_passes_kwarg() -> None:
      sig = inspect.signature(_dispatch_guided_respond)
      assert "advisor_checkpoint_max_passes" in sig.parameters
      param = sig.parameters["advisor_checkpoint_max_passes"]
      assert param.kind is inspect.Parameter.KEYWORD_ONLY
      assert param.annotation == "int | None"
      # None => resolve the budget from settings inside the dispatcher (P5.6).
      assert param.default is None
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py -q`
  Expected failure: `AssertionError: assert 'composer_service' in {...}` (params absent).
- [ ] **Step 3: Add the import + params to the dispatcher.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, ensure `ComposerService` is importable for the annotation. If it is not already imported, add at the top of the existing composer-imports block:
  ```python
  from elspeth.web.composer.protocol import ComposerService
  ```
  Then append BOTH params to `_dispatch_guided_respond`'s keyword-only signature, after `seed: int | None,`. Both carry SAFE DEFAULTS so the P2.9 `test_wire_dispatch.py::_dispatch` helper (which constructs the call without these kwargs) stays valid:
  ```python
      seed: int | None,
      composer_service: ComposerService | None = None,  # None => advisor gate skipped (empty-profile path)
      advisor_checkpoint_max_passes: int | None = None,  # None => resolve from settings inside the dispatcher (P5.6)
  ) -> tuple[CompositionState, GuidedSession, Any | None]:
  ```
  (Match the existing closing-paren/return-annotation line at :2570-2571; insert both lines above `) -> tuple[...]`. The defaults are load-bearing: a required positional/keyword without a default would break every pre-P5 caller that omits it — see the P2.9 `_dispatch` helper, which P5.4 must NOT invalidate.)
- [ ] **Step 4: Pass the handle + budget from the route.**
  In `src/elspeth/web/sessions/routes/composer.py`, the `_dispatch_guided_respond(...)` call (:2333) currently ends with `seed=settings.composer_seed,`. Add the two new kwargs as the final kwargs:
  ```python
                          seed=settings.composer_seed,
                          composer_service=request.app.state.composer_service,
                          advisor_checkpoint_max_passes=settings.composer_advisor_checkpoint_max_passes,
                      )
  ```
- [ ] **Step 5: Run to pass + confirm the route still imports.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py -q && uv run python -c "import elspeth.web.sessions.routes.composer"`
  Expected: `1 passed`, then a clean import (no output, exit 0).
- [ ] **Step 6: Run the existing guided-respond route + wire-dispatch suites to confirm no caller breakage.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web tests/integration/web/composer/guided/test_wire_dispatch.py -q -k "guided_respond or dispatch_guided or wire"`
  Expected: existing tests `passed`. Because both new params take safe defaults,
  the P2.9 `tests/integration/web/composer/guided/test_wire_dispatch.py::_dispatch`
  helper (which constructs `_dispatch_guided_respond(...)` WITHOUT `composer_service`
  / `advisor_checkpoint_max_passes`) keeps working unchanged — it is the canonical
  pre-P5 direct caller and is the reason the defaults exist. Named direct callers
  to confirm still pass (and fix in this same commit if any break):
  - `tests/integration/web/composer/guided/test_wire_dispatch.py::_dispatch` (P2.9; relies on the defaults — must NOT require the new kwargs).
  Search for any OTHER in-repo direct caller and fix it here if found:
  `grep -rln "_dispatch_guided_respond(" tests/`.
- [ ] **Step 7: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py && git commit -m "feat(web/sessions): thread ComposerService handle into guided dispatcher (P5.4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.5: Add the persisted-counter-bound async sign-off runner (`run_wire_signoff`)

**Files:**
- Modify: `src/elspeth/web/composer/guided/signoff.py` (append an async runner that calls the service + classifier + persists the counter)
- Create: `tests/unit/web/composer/guided/test_wire_signoff_runner.py`

**Interfaces:**
- Produces (canonical name, verbatim):
  `async def run_wire_signoff(*, session: GuidedSession, state: CompositionState, session_id: str | None, recorder: BufferingRecorder, composer_service: ComposerService, max_passes: int, acknowledged_unavailable: bool, progress: ComposerProgressSink | None = None) -> tuple[GuidedSession, SignoffDecision]`
- Consumes: `GuidedSession.advisor_checkpoint_passes_used` (**P0**), `ComposerService.run_signoff_checkpoint` (P5.1/P5.2), `classify_signoff_verdict` (P5.3).

Behaviour:
- Reads `passes_used = session.advisor_checkpoint_passes_used`.
- If `passes_used >= max_passes` (budget already spent on a prior request) AND not `acknowledged_unavailable`: return the session unchanged with a `BLOCKED_FLAGGED` (`reason="exhausted"`) decision **without** calling the provider (the persisted bound prevents unbounded re-calls across HTTP requests, D16).
- Otherwise call `composer_service.run_signoff_checkpoint(...)`, classify, and on every classified pass persist `advisor_checkpoint_passes_used += decision.passes_delta` onto a `dataclasses.replace`'d session.
- If `decision.outcome is ESCAPE_UNAVAILABLE and acknowledged_unavailable`: return `SignoffOutcome.COMPLETE`-equivalent via a distinct decision (carry the original `reason="unavailable"` so the caller records the differentiated audit event). Concretely, return a `SignoffDecision(outcome=SignoffOutcome.COMPLETE, reason="unavailable", ...)` — COMPLETE-with-`reason="unavailable"` is the audited "complete without sign-off" path; CLEAN COMPLETE has `reason is None`.
- If `decision.outcome is ESCAPE_UNAVAILABLE and not acknowledged_unavailable`: leave it `ESCAPE_UNAVAILABLE` (the caller emits the escape-offer wire turn).

- [ ] **Step 1: Write the failing runner test.**
  Create `tests/unit/web/composer/guided/test_wire_signoff_runner.py`:
  ```python
  """Phase P5.5 — persisted-counter-bound wire-stage sign-off runner."""

  from __future__ import annotations

  import dataclasses
  from unittest.mock import AsyncMock

  import pytest

  from elspeth.web.composer.guided.signoff import (
      SignoffOutcome,
      run_wire_signoff,
  )
  from elspeth.web.composer.guided.state_machine import GuidedSession
  from elspeth.web.composer.service import AdvisorCheckpointVerdict
  from elspeth.web.composer.state import (
      CompositionState,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )


  def _state() -> CompositionState:
      return CompositionState(
          source=SourceSpec(plugin="csv", on_success="main", options={"path": "in.csv"}, on_validation_failure="discard"),
          nodes=(),
          edges=(),
          outputs=(OutputSpec(name="out", plugin="csv", options={"path": "out.csv"}, on_write_failure="discard"),),
          metadata=PipelineMetadata(),
          version=2,
      )


  def _service(verdict: AdvisorCheckpointVerdict) -> object:
      svc = AsyncMock()
      svc.run_signoff_checkpoint = AsyncMock(return_value=verdict)
      return svc


  @pytest.mark.asyncio
  async def test_clean_completes_and_increments_counter() -> None:
      session = GuidedSession.initial()
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=False,
      )
      assert decision.outcome is SignoffOutcome.COMPLETE
      assert decision.reason is None
      assert new_session.advisor_checkpoint_passes_used == 1
      svc.run_signoff_checkpoint.assert_awaited_once()


  @pytest.mark.asyncio
  async def test_flagged_last_pass_blocks_no_bypass() -> None:
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: bad"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=False,
      )
      assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert decision.reason == "exhausted"
      assert new_session.advisor_checkpoint_passes_used == 3


  @pytest.mark.asyncio
  async def test_budget_already_spent_does_not_recall_provider() -> None:
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=3)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=False,
      )
      assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert new_session.advisor_checkpoint_passes_used == 3
      svc.run_signoff_checkpoint.assert_not_awaited()


  @pytest.mark.asyncio
  async def test_unavailable_last_pass_offers_escape_when_unacknowledged() -> None:
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="TimeoutError"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=False,
      )
      assert decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
      assert decision.reason == "unavailable"


  @pytest.mark.asyncio
  async def test_unavailable_acknowledged_completes_with_unavailable_reason() -> None:
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="TimeoutError"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=True,
      )
      # Audited "complete without sign-off": COMPLETE outcome but reason carries
      # "unavailable" so the caller records a DISTINCT audit event vs a CLEAN.
      assert decision.outcome is SignoffOutcome.COMPLETE
      assert decision.reason == "unavailable"


  @pytest.mark.asyncio
  async def test_acknowledged_unavailable_never_bypasses_a_flag() -> None:
      # A FLAG on the last pass with acknowledged_unavailable=True must still BLOCK.
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: bad"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=True,
      )
      assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED


  @pytest.mark.asyncio
  async def test_exhausted_with_acknowledged_outage_completes_cross_request() -> None:
      # D5/B2 regression: the escape is OFFERED on the final pass (one request) and
      # ACKNOWLEDGED on a LATER request, by which time passes_used == max_passes. The
      # persisted escape_offered marker lets the acknowledgement COMPLETE rather than
      # dead-end to BLOCKED_FLAGGED — and the provider is NOT re-called at exhaustion.
      session = dataclasses.replace(
          GuidedSession.initial(),
          advisor_checkpoint_passes_used=3,
          advisor_signoff_escape_offered=True,
      )
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=True,
      )
      assert decision.outcome is SignoffOutcome.COMPLETE
      assert decision.reason == "unavailable"
      svc.run_signoff_checkpoint.assert_not_awaited()


  @pytest.mark.asyncio
  async def test_exhausted_acknowledged_but_prior_was_flag_stays_blocked() -> None:
      # The acknowledgement must NEVER bypass a FLAG: a FLAGGED/MALFORMED-exhausted
      # terminal leaves escape_offered=False, so acknowledging it stays BLOCKED.
      session = dataclasses.replace(
          GuidedSession.initial(),
          advisor_checkpoint_passes_used=3,
          advisor_signoff_escape_offered=False,
      )
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=True,
      )
      assert decision.outcome is SignoffOutcome.BLOCKED_FLAGGED
      svc.run_signoff_checkpoint.assert_not_awaited()
  ```
  > Test-construction note: `dataclasses.replace(session, advisor_checkpoint_passes_used=N)`
  > rebuilds the frozen `GuidedSession` with a bumped counter. `GuidedSession`
  > is `@dataclass(frozen=True, slots=True)`, so it has **no `__dict__`**; a
  > `session.__class__(**{**session.__dict__, ...})` reconstruction would raise
  > `AttributeError: 'GuidedSession' object has no attribute '__dict__'`, which
  > is why `dataclasses.replace` (the correct idiom for slotted frozen
  > dataclasses) is used. The call depends on the P0 field
  > `advisor_checkpoint_passes_used` existing; if P0 has not landed, Step 2
  > fails with `TypeError: __init__() got an unexpected keyword argument
  > 'advisor_checkpoint_passes_used'` — that is the cross-phase dependency
  > surfacing, not a defect in this task.
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/guided/test_wire_signoff_runner.py -q`
  Expected failure: `ImportError: cannot import name 'run_wire_signoff' from 'elspeth.web.composer.guided.signoff'`.
- [ ] **Step 3: Append the runner to `signoff.py`.**
  Add to the TYPE_CHECKING block in `src/elspeth/web/composer/guided/signoff.py`:
  ```python
  if TYPE_CHECKING:
      from elspeth.contracts.composer_progress import ComposerProgressSink
      from elspeth.web.composer.audit import BufferingRecorder
      from elspeth.web.composer.guided.state_machine import GuidedSession
      from elspeth.web.composer.protocol import ComposerService
      from elspeth.web.composer.service import AdvisorCheckpointVerdict
      from elspeth.web.composer.state import CompositionState
  ```
  Then append the runner at module end:
  ```python
  async def run_wire_signoff(
      *,
      session: GuidedSession,
      state: CompositionState,
      session_id: str | None,
      recorder: BufferingRecorder | None,
      composer_service: ComposerService,
      max_passes: int,
      acknowledged_unavailable: bool,
      progress: ComposerProgressSink | None = None,
  ) -> tuple[GuidedSession, SignoffDecision]:
      """Run one wire-stage END sign-off pass, bounded by the PERSISTED counter.

      Returns the (possibly counter-bumped) session and the D13 decision. The
      persisted ``GuidedSession.advisor_checkpoint_passes_used`` is the re-entry
      bound (D16): guided re-entry crosses separate ``/guided/respond`` HTTP
      requests, so an unpersisted per-compose local would reset to 0 each request
      and never bound the loop. When the budget is already spent on a prior
      request the provider is NOT re-called.

      ``acknowledged_unavailable`` is the user's explicit "complete without
      sign-off (advisor unreachable)" acknowledgement; it converts a budget-
      exhausted UNAVAILABLE escape into an audited COMPLETE-with-``reason=
      "unavailable"`` — and it can NEVER bypass a FLAG (a FLAG never produces an
      ESCAPE_UNAVAILABLE outcome, by classifier construction).
      """
      import dataclasses

      passes_used = session.advisor_checkpoint_passes_used
      if passes_used >= max_passes:
          # Budget spent on a prior request: do not re-call the provider.
          if acknowledged_unavailable and session.advisor_signoff_escape_offered:
              # The prior budget-exhausting terminal was a genuine UNAVAILABLE
              # escape OFFER (persisted marker) and the user has now acknowledged
              # "complete without sign-off (advisor unreachable)". Honour it as the
              # audited COMPLETE-with-reason="unavailable". This can NEVER bypass a
              # FLAG: a FLAGGED-exhausted (or MALFORMED-exhausted) terminal leaves
              # escape_offered=False, so an acknowledgement there falls through to
              # BLOCKED below. The acknowledgement arrives on a SEPARATE
              # /guided/respond request than the one that emitted the offer — which
              # is exactly why this cross-request marker is required (D5/B2).
              return session, SignoffDecision(
                  outcome=SignoffOutcome.COMPLETE,
                  reason="unavailable",
                  findings_text="Advisor unreachable; completed without sign-off (acknowledged).",
                  passes_delta=0,
              )
          # Otherwise fail closed (no bypass). FLAGGED-exhausted is the safe terminal.
          return session, SignoffDecision(
              outcome=SignoffOutcome.BLOCKED_FLAGGED,
              reason="exhausted",
              findings_text="Advisor sign-off budget exhausted.",
              passes_delta=0,
          )

      verdict = await composer_service.run_signoff_checkpoint(
          state=state,
          session_id=session_id,
          recorder=recorder,
          progress=progress,
      )
      decision = classify_signoff_verdict(verdict, passes_used=passes_used, max_passes=max_passes)
      new_session = dataclasses.replace(
          session,
          advisor_checkpoint_passes_used=passes_used + decision.passes_delta,
          # Persist whether THIS terminal was a genuine-outage escape OFFER, so a
          # later request carrying the user's acknowledgement (handled above) can
          # honour it without re-calling the provider — and so a FLAGGED-exhausted
          # terminal (escape_offered=False) can never be acknowledged into a bypass.
          advisor_signoff_escape_offered=(decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE),
      )

      if decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE and acknowledged_unavailable:
          # Same-request acknowledgement (user pre-acknowledged before the final
          # pass): audited "complete without sign-off (advisor unreachable)". ONLY
          # reachable from a genuine UNAVAILABLE, never a FLAG or MALFORMED (neither
          # produces ESCAPE_UNAVAILABLE, by classifier construction).
          decision = SignoffDecision(
              outcome=SignoffOutcome.COMPLETE,
              reason="unavailable",
              findings_text=decision.findings_text,
              passes_delta=decision.passes_delta,
          )

      return new_session, decision
  ```
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/guided/test_wire_signoff_runner.py -q`
  Expected: `8 passed`.
- [ ] **Step 5: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/guided/signoff.py tests/unit/web/composer/guided/test_wire_signoff_runner.py && git commit -m "feat(composer/guided): persisted-counter-bound wire sign-off runner (P5.5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.6: Gate the wire-stage terminal on the profile + sign-off decision (dispatch branch)

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (the `STEP_4_WIRE` branch of `_dispatch_guided_respond`, **created by P2.9**; this task adds the profile-gated sign-off before the COMPLETED stamp)
- Create: `tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py`

**Interfaces:**
- Consumes: `GuidedSession.profile` + `.advisor_checkpoint_passes_used` (**P0**), `WorkflowProfile.advisor_checkpoints` (**P0**), `GuidedStep.STEP_4_WIRE` + `TurnType.CONFIRM_WIRING` (**P1**), `handle_step_4_wire_confirm` (**P1.6**, `composer/guided/steps.py`) + `build_step_4_wire_turn` (**P2.4**, final signature already accepts `catalog`/`advisor_findings`/`signoff_outcome`), the `STEP_4_WIRE` dispatch branch (**P2.9**), `run_wire_signoff` + `SignoffOutcome` (P5.5), `TerminalState`/`TerminalKind` (`state_machine.py`), the wire `composer_service` handle (P5.4).
- Produces: behaviour — the `STEP_4_WIRE` branch stamps `TerminalState(COMPLETED)` only on `SignoffOutcome.COMPLETE`; emits a revise wire turn on `REVISE`/`ESCAPE_UNAVAILABLE`; sets a fail-closed terminal-less revise turn carrying `_advisor_signoff_blocked_validation` findings on `BLOCKED_FLAGGED`/`BLOCKED_UNAVAILABLE`.

> **Precondition (read before implementing):** P2.9 has already created the
> `STEP_4_WIRE` branch in `_dispatch_guided_respond` whose `CONFIRM_WIRING`
> sub-branch calls `handle_step_4_wire_confirm(...)` and lets it stamp
> `TerminalState(COMPLETED)` on a valid pipeline (P2.9 is a hard upstream
> dependency — do NOT re-create the branch here). This task REPLACES the BODY of
> that `CONFIRM_WIRING` sub-branch with the profile-gated form: it keeps the same
> validate-gate (re-emit the wire turn on an invalid pipeline) but moves the
> validate check inline and routes the COMPLETED stamp through the profile gate, so
> the unconditional `handle_step_4_wire_confirm` stamp no longer races the
> tutorial-profile sign-off. It LAYERS the gate; it does not create the dispatch
> branch.

- [ ] **Step 1: Write the failing gate tests.**
  Create `tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py`. These exercise `_dispatch_guided_respond` directly at `current_step=STEP_4_WIRE`, `current_turn_type=CONFIRM_WIRING`, with a stubbed `composer_service`:
  ```python
  """Phase P5.6 — STEP_4_WIRE terminal is gated on profile.advisor_checkpoints + verdict."""

  from __future__ import annotations

  from unittest.mock import AsyncMock, MagicMock

  import pytest

  from elspeth.web.composer.audit import BufferingRecorder
  from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
  from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
  from elspeth.web.composer.guided.state_machine import GuidedSession, TerminalKind
  from elspeth.web.composer.service import AdvisorCheckpointVerdict
  from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
  from tests.unit.web.sessions.routes._wire_fixtures import (  # P3 helper; see note
      make_wire_ready_session_and_state,
  )


  def _service(verdict: AdvisorCheckpointVerdict | None) -> MagicMock:
      svc = MagicMock()
      if verdict is None:
          svc.run_signoff_checkpoint = AsyncMock(side_effect=AssertionError("advisor must NOT be called"))
      else:
          svc.run_signoff_checkpoint = AsyncMock(return_value=verdict)
      return svc


  async def _dispatch(session: GuidedSession, state, svc, *, control=ControlSignal.EXIT_TO_FREEFORM):
      # CONFIRM_WIRING confirm response: no control signal (a plain confirm).
      turn_response = {
          "chosen": ["confirm"],
          "edited_values": None,
          "custom_inputs": None,
          "accepted_step_index": None,
          "edit_step_index": None,
          "control_signal": None,
      }
      catalog = MagicMock()
      return await _dispatch_guided_respond(
          state=state,
          guided=session,
          current_step=GuidedStep.STEP_4_WIRE,
          current_turn_type=TurnType.CONFIRM_WIRING,
          turn_response=turn_response,
          catalog=catalog,
          recorder=BufferingRecorder(),
          user_id="u1",
          data_dir=None,
          session_engine=None,
          session_id="s1",
          blob_service=MagicMock(),
          model="m",
          temperature=None,
          seed=None,
          composer_service=svc,
          advisor_checkpoint_max_passes=3,
      )


  @pytest.mark.asyncio
  async def test_empty_profile_completes_with_zero_provider_calls() -> None:
      session, state = make_wire_ready_session_and_state(profile=EMPTY_PROFILE)
      svc = _service(None)  # asserts run_signoff_checkpoint is never awaited
      _state, guided, _turn = await _dispatch(session, state, svc)
      assert guided.terminal is not None
      assert guided.terminal.kind is TerminalKind.COMPLETED
      svc.run_signoff_checkpoint.assert_not_awaited()


  @pytest.mark.asyncio
  async def test_tutorial_profile_clean_completes() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      _state, guided, _turn = await _dispatch(session, state, svc)
      assert guided.terminal is not None
      assert guided.terminal.kind is TerminalKind.COMPLETED
      assert guided.advisor_checkpoint_passes_used == 1
      svc.run_signoff_checkpoint.assert_awaited_once()


  @pytest.mark.asyncio
  async def test_tutorial_profile_flagged_does_not_complete() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: x"))
      _state, guided, next_turn = await _dispatch(session, state, svc)
      assert guided.terminal is None  # re-emit a revise turn, never COMPLETED
      assert next_turn is not None
      assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
  ```
  > Fixture note: `make_wire_ready_session_and_state` is the P3-owned helper
  > that builds a STEP_4_WIRE-positioned `GuidedSession` (with the given
  > `profile`) plus a valid `CompositionState`. If P3 has not exported it,
  > create a local minimal version in `tests/unit/web/sessions/routes/_wire_fixtures.py`
  > that constructs a single-source/single-sink valid state and a session with
  > `step=GuidedStep.STEP_4_WIRE`, `history=(<a CONFIRM_WIRING TurnRecord>,)`,
  > and `profile=<arg>`.
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected failure: depending on P0–P3 landed state, either an `ImportError` for `EMPTY_PROFILE`/`STEP_4_WIRE`/`make_wire_ready_session_and_state` (cross-phase dep not yet present) or, once those exist, `AssertionError: advisor must NOT be called` / `terminal is None` (the gate logic not yet inserted).
- [ ] **Step 3: Replace the P2.9 confirm body with the profile-gated sign-off.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, inside the `if current_turn_type is TurnType.CONFIRM_WIRING:` sub-branch of the `if current_step is GuidedStep.STEP_4_WIRE:` block (P2.9-created), REPLACE the P2.9 body — which called `handle_step_4_wire_confirm(...)` and let *it* stamp COMPLETED unconditionally on a valid pipeline — with the profile-gated form below. Keep the SAME validate-gate semantics: run `state.validate()` first and re-emit the wire turn (terminal stays `None`) on an invalid pipeline, THEN profile-branch. (Do NOT call `handle_step_4_wire_confirm` here any more — its unconditional stamp would race the tutorial-profile gate; the validate check moves inline so the gate owns the stamp.)
  ```python
              # Validate-gate first (same as P2.9): an invalid pipeline never
              # completes — re-emit the wire turn so the user can reconcile (B6).
              if not state.validate().is_valid:
                  guided, next_turn = _emit_wire_turn(
                      state=state, guided=guided, recorder=recorder, user_id=user_id
                  )
                  return state, guided, next_turn

              # D13 — profile-gated terminal advisor sign-off. The empty/live-
              # guided profile (advisor_checkpoints=False) skips the provider
              # entirely and completes on a valid pipeline (no blocking advisor
              # round-trip; the wire stage stays a benign topology-review
              # improvement for live guided). The tutorial profile runs the
              # whole-pipeline END sign-off as a PRE-terminal gate so a FLAG can
              # still re-emit a revise turn (a post-terminal hook would be
              # foreclosed by the composer.py:2131 terminal-409).
              from elspeth.web.composer.guided.signoff import SignoffOutcome, run_wire_signoff

              if not guided.profile.advisor_checkpoints:
                  yaml_text = generate_yaml(state)
                  terminal = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=yaml_text)
                  guided = _replace(guided, terminal=terminal)
                  return state, guided, None

              acknowledged_unavailable = bool(turn_response.get("custom_inputs", None) == ["complete_without_signoff"])
              max_passes = advisor_checkpoint_max_passes  # P5.4 dispatcher param
              guided, decision = await run_wire_signoff(
                  session=guided,
                  state=state,
                  session_id=session_id,
                  recorder=recorder,
                  composer_service=composer_service,
                  max_passes=max_passes,
                  acknowledged_unavailable=acknowledged_unavailable,
                  progress=None,
              )
              if decision.outcome is SignoffOutcome.COMPLETE:
                  yaml_text = generate_yaml(state)
                  terminal = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=yaml_text)
                  guided = _replace(guided, terminal=terminal)
                  return state, guided, None
              # Non-COMPLETE: re-emit the wire turn (terminal stays None). The
              # turn payload carries the advisor findings + outcome class so the
              # frontend renders the revise / fail-closed / escape-offer affordance.
              next_turn = build_step_4_wire_turn(
                  state,
                  catalog=catalog,
                  advisor_findings=decision.findings_text,
                  signoff_outcome=decision.outcome.value,
              )
              new_record = TurnRecord(
                  step=GuidedStep.STEP_4_WIRE,
                  turn_type=TurnType.CONFIRM_WIRING,
                  payload_hash=stable_hash(next_turn["payload"]),
                  response_hash=None,
                  emitter="server",
              )
              emit_turn_emitted(
                  recorder,
                  step=GuidedStep.STEP_4_WIRE,
                  turn_type=TurnType.CONFIRM_WIRING,
                  payload_hash=stable_hash(next_turn["payload"]),
                  payload_payload_id="",
                  emitter="server",
                  composition_version=state.version,
                  actor=user_id,
              )
              guided = _replace(guided, history=(*guided.history, new_record))
              return state, guided, next_turn
  ```
  > Implementation notes:
  > - `settings` is NOT a dispatcher param. `max_passes` is threaded into
  >   `_dispatch_guided_respond` as the keyword-only param
  >   `advisor_checkpoint_max_passes: int` by **P5.4** (added to the P5.4
  >   signature + call + signature test, sourced from
  >   `settings.composer_advisor_checkpoint_max_passes` at the route call site,
  >   `composer.py:2333`). This task READS `advisor_checkpoint_max_passes` (the
  >   dispatcher param) — do NOT reach into `composer_service._settings`. If P5.4
  >   has not yet added the param, add it there first (its run-to-fail names the
  >   missing param), then this task consumes it.
  > - `build_step_4_wire_turn` ALREADY accepts `catalog` + `advisor_findings` +
  >   `signoff_outcome` (all optional, defaulting `None`) — that is the FINAL
  >   signature landed by P2.4. Call it verbatim:
  >   `build_step_4_wire_turn(state, catalog=catalog, advisor_findings=..., signoff_outcome=...)`.
  >   Do NOT modify the emitter here.
  > - `generate_yaml`, `TerminalState`, `TerminalKind`, `TurnRecord`,
  >   `stable_hash`, `emit_turn_emitted`, `_replace` are already imported in
  >   `_helpers.py` (used by the sibling STEP_3 branch).
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected: `3 passed`.
- [ ] **Step 5: Run the wider guided-dispatch + advisor suites for no regression.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py tests/unit/web/composer/guided -q -k "signoff or wire or advisor"`
  Expected: all `passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py && git commit -m "feat(web/sessions): profile-gate the STEP_4_WIRE terminal on the advisor sign-off (P5.6)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.7: Fail-closed terminal-less result + differentiated UNAVAILABLE audit event

**Files:**
- Modify: `src/elspeth/web/composer/guided/signoff.py` (add an audit-event-name resolver `signoff_audit_event_name`)
- Modify: `src/elspeth/web/composer/guided/audit.py` (add the `emit_signoff_decision` event helper, mirroring `emit_turn_emitted`/`emit_step_advanced`)
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (emit the differentiated audit event + carry the blocked validation findings on the revise turn for `BLOCKED_*` outcomes)
- Create: `tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py`

**Interfaces:**
- Produces (canonical name, verbatim):
  `def signoff_audit_event_name(decision: SignoffDecision) -> str` returning one of: `"composer.signoff.clean"` (COMPLETE + reason None), `"composer.signoff.completed_without_signoff_advisor_unreachable"` (COMPLETE + reason "unavailable"), `"composer.signoff.blocked_flagged"` (BLOCKED_FLAGGED), `"composer.signoff.blocked_unavailable"` (BLOCKED_UNAVAILABLE), `"composer.signoff.revise"` (REVISE), `"composer.signoff.escape_offered"` (ESCAPE_UNAVAILABLE).
- Produces (audit emit helper, mirrors the existing `guided/audit.py` pattern):
  `def emit_signoff_decision(recorder: ComposerToolRecorder, *, event_name: str, outcome: str, reason: str | None, composition_version: int, actor: str) -> None` — builds a `ComposerToolInvocation` via `_build_invocation(tool_name=event_name, payload={"outcome": outcome, "reason": reason}, ...)` and calls `recorder.record(invocation)`. There is NO `recorder.record_event` method on `BufferingRecorder` (its surface is `record`/`record_llm_call`/`record_chat_turn`, audit.py:207-215) — the guided audit convention is a free function that builds an invocation and calls `recorder.record(...)`.
- Consumes: `_advisor_signoff_blocked_validation(reason=..., findings=...)` (`composer/service.py:4946`) — for the blocked terminal's findings text; `_build_invocation`/`ComposerToolRecorder` (`composer/guided/audit.py:37`).

> The whole point of the differentiated audit name is honest provenance: a
> "completed without sign-off because advisor unreachable" terminal must NEVER
> be indistinguishable from a CLEAN sign-off (D13). This is the load-bearing
> security/audit assertion of the phase.

- [ ] **Step 1: Write the failing audit-name + blocked-findings test.**
  Create `tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py`:
  ```python
  """Phase P5.7 — differentiated sign-off audit names + fail-closed findings."""

  from __future__ import annotations

  from elspeth.web.composer.guided.signoff import (
      SignoffDecision,
      SignoffOutcome,
      signoff_audit_event_name,
  )
  from elspeth.web.composer.service import _advisor_signoff_blocked_validation


  def _d(outcome: SignoffOutcome, reason: str | None) -> SignoffDecision:
      return SignoffDecision(outcome=outcome, reason=reason, findings_text="f", passes_delta=1)


  def test_clean_audit_name() -> None:
      assert signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, None)) == "composer.signoff.clean"


  def test_completed_without_signoff_has_distinct_audit_name() -> None:
      # The audited escape must be DISTINGUISHABLE from a CLEAN sign-off.
      name = signoff_audit_event_name(_d(SignoffOutcome.COMPLETE, "unavailable"))
      assert name == "composer.signoff.completed_without_signoff_advisor_unreachable"
      assert name != "composer.signoff.clean"


  def test_blocked_flagged_audit_name() -> None:
      assert signoff_audit_event_name(_d(SignoffOutcome.BLOCKED_FLAGGED, "exhausted")) == "composer.signoff.blocked_flagged"


  def test_escape_offered_audit_name() -> None:
      assert signoff_audit_event_name(_d(SignoffOutcome.ESCAPE_UNAVAILABLE, "unavailable")) == "composer.signoff.escape_offered"


  def test_blocked_validation_is_non_runnable() -> None:
      result = _advisor_signoff_blocked_validation(reason="exhausted", findings="prompt sees no row field")
      assert result.is_valid is False
      assert result.readiness.authoring_valid is False
      assert result.readiness.execution_ready is False
      assert result.readiness.completion_ready is False
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py -q`
  Expected failure: `ImportError: cannot import name 'signoff_audit_event_name' from 'elspeth.web.composer.guided.signoff'`.
- [ ] **Step 3: Add the audit-name resolver to `signoff.py`.**
  Append to `src/elspeth/web/composer/guided/signoff.py`:
  ```python
  def signoff_audit_event_name(decision: SignoffDecision) -> str:
      """Map a sign-off decision to a DISTINCT audit event name (D13 provenance).

      The "complete without sign-off (advisor unreachable)" escape MUST be
      distinguishable in the audit trail from a CLEAN sign-off — both are
      COMPLETE outcomes, but the escape carries ``reason="unavailable"`` while a
      CLEAN sign-off carries ``reason=None``. An operator reading the audit log
      can therefore tell an advisor-unreachable completion from a real sign-off.
      """
      if decision.outcome is SignoffOutcome.COMPLETE:
          if decision.reason == "unavailable":
              return "composer.signoff.completed_without_signoff_advisor_unreachable"
          return "composer.signoff.clean"
      if decision.outcome is SignoffOutcome.REVISE:
          return "composer.signoff.revise"
      if decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE:
          return "composer.signoff.escape_offered"
      if decision.outcome is SignoffOutcome.BLOCKED_UNAVAILABLE:
          return "composer.signoff.blocked_unavailable"
      return "composer.signoff.blocked_flagged"
  ```
- [ ] **Step 4: Add the `emit_signoff_decision` audit helper to `guided/audit.py`.**
  `BufferingRecorder` has NO `record_event` method (its surface is
  `record(ComposerToolInvocation)` / `record_llm_call` / `record_chat_turn`,
  `composer/audit.py:207-215`). The guided audit convention (`guided/audit.py`) is a
  free function that builds an invocation via `_build_invocation(...)` and calls
  `recorder.record(invocation)` — `emit_turn_emitted` (`:81`),
  `emit_step_advanced` (`:176`), `emit_dropped_to_freeform` (`:219`) all follow this.
  Append the sign-off variant after `emit_dropped_to_freeform`:
  ```python
  def emit_signoff_decision(
      recorder: ComposerToolRecorder,
      *,
      event_name: str,
      outcome: str,
      reason: str | None,
      composition_version: int,
      actor: str,
  ) -> None:
      """Record a differentiated wire-stage sign-off decision audit event (D13).

      ``event_name`` is the distinct ``signoff_audit_event_name(decision)`` string
      (e.g. ``"composer.signoff.completed_without_signoff_advisor_unreachable"`` vs
      ``"composer.signoff.clean"``) — the audit trail MUST distinguish an
      advisor-unreachable completion from a real sign-off. Built as a
      ``ComposerToolInvocation`` via the shared ``_build_invocation`` (Errata C4
      pattern: no new audit primitive); recorded through ``recorder.record(...)``.
      """
      payload: dict[str, Any] = {"outcome": outcome}
      if reason is not None:
          payload["reason"] = reason
      now = datetime.now(UTC)
      invocation = _build_invocation(
          tool_name=event_name,
          payload=payload,
          composition_version=composition_version,
          actor=actor,
          now=now,
      )
      recorder.record(invocation)
  ```
- [ ] **Step 5: Emit the differentiated event + carry blocked findings in the dispatch branch.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, in the `STEP_4_WIRE` branch from
  P5.6, immediately AFTER `run_wire_signoff(...)` returns `decision`, record the audit
  event via the real helper. Add to the existing
  `from elspeth.web.composer.guided.audit import (...)` import block (`_helpers.py`,
  the block that already imports `emit_turn_emitted` at `:67`) the names
  `emit_signoff_decision`, and import `signoff_audit_event_name` from
  `elspeth.web.composer.guided.signoff`:
  ```python
              from elspeth.web.composer.guided.audit import emit_signoff_decision
              from elspeth.web.composer.guided.signoff import signoff_audit_event_name

              emit_signoff_decision(
                  recorder,
                  event_name=signoff_audit_event_name(decision),
                  outcome=decision.outcome.value,
                  reason=decision.reason,
                  composition_version=state.version,
                  actor=user_id,
              )
  ```
  And in the non-COMPLETE branch, when `decision.outcome in (SignoffOutcome.BLOCKED_FLAGGED, SignoffOutcome.BLOCKED_UNAVAILABLE)`, pass the blocked validation findings into `build_step_4_wire_turn` so the turn renders fail-closed (non-runnable) rather than a plain retry. This REPLACES the plain `build_step_4_wire_turn(...)` call from P5.6 Step 3 in the non-COMPLETE path — fold the two so there is a single emit:
  ```python
              blocked_findings = None
              if decision.outcome in (SignoffOutcome.BLOCKED_FLAGGED, SignoffOutcome.BLOCKED_UNAVAILABLE):
                  blocked = _advisor_signoff_blocked_validation(
                      reason=decision.reason or "exhausted", findings=decision.findings_text
                  )
                  blocked_findings = blocked.errors[0].message if blocked.errors else decision.findings_text
              next_turn = build_step_4_wire_turn(
                  state,
                  catalog=catalog,
                  advisor_findings=blocked_findings or decision.findings_text,
                  signoff_outcome=decision.outcome.value,
              )
  ```
  > Notes:
  > - Import `_advisor_signoff_blocked_validation` from
  >   `elspeth.web.composer.service` at the top of `_helpers.py` (module-scope
  >   free function, no `self`).
  > - The Step-1 test pins the NAME via `signoff_audit_event_name`; add an
  >   assertion that the recorded invocation's `tool_name` equals that name by
  >   reading `recorder.invocations()[-1].tool_name` after a dispatch (the
  >   `BufferingRecorder.invocations()` accessor, `composer/audit.py:226`).
- [ ] **Step 6: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected: all `passed` (audit-name tests + the P5.6 gate tests still green).
- [ ] **Step 7: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/guided/audit.py src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py && git commit -m "feat(web/sessions): differentiated sign-off audit names + fail-closed blocked findings (P5.7)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.8: REQUEST_ADVISOR whole-pipeline escape — preserve the step-3 chain re-solve

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (add a `STEP_4_WIRE` + on-demand `REQUEST_ADVISOR` branch; leave the existing STEP_3 `REJECT/REQUEST_ADVISOR` chain re-solve at :3342 untouched)
- Create: `tests/unit/web/sessions/routes/test_request_advisor_escape.py`

**Interfaces:**
- Consumes: `ControlSignal.REQUEST_ADVISOR` (`protocol.py:72`), `run_wire_signoff`/`SignoffOutcome` (P5.5), the existing STEP_3 `solve_chain_with_auto_drop` re-solve (`_helpers.py:3352`).
- Produces: behaviour — a `REQUEST_ADVISOR` control on a `STEP_4_WIRE` `CONFIRM_WIRING` turn runs the whole-pipeline sign-off on-demand (subject to the persisted pass budget) and re-emits the wire turn with findings; a `REQUEST_ADVISOR` at STEP_3 still triggers `solve_chain_with_auto_drop` (regression guard).

> D6/D13: `REQUEST_ADVISOR` is the per-phase on-demand "go to advisor" escape.
> Today it is a REAL chain re-solve at STEP_3 only (`_helpers.py:3342`). This
> task ADDS the whole-pipeline checkpoint as an ADDITIONAL `REQUEST_ADVISOR`
> target at the wire stage — it does NOT replace the step-3 re-solve. Trust
> tier: the on-demand checkpoint goes through `run_signoff_checkpoint` (the
> backend-produced Tier-1 `schema_excerpt`), so no unvalidated user text is
> forwarded and the Tier-3 `_validate_advisor_arguments` boundary is not
> crossed.

- [ ] **Step 1: Write the failing escape tests.**
  Create `tests/unit/web/sessions/routes/test_request_advisor_escape.py`:
  ```python
  """Phase P5.8 — REQUEST_ADVISOR whole-pipeline escape at the wire stage;
  the existing step-3 chain re-solve is preserved."""

  from __future__ import annotations

  from unittest.mock import AsyncMock, MagicMock

  import pytest

  from elspeth.web.composer.audit import BufferingRecorder
  from elspeth.web.composer.guided.profile import TUTORIAL_PROFILE
  from elspeth.web.composer.guided.protocol import ControlSignal, GuidedStep, TurnType
  from elspeth.web.composer.service import AdvisorCheckpointVerdict
  from elspeth.web.sessions.routes._helpers import _dispatch_guided_respond
  from tests.unit.web.sessions.routes._wire_fixtures import make_wire_ready_session_and_state


  @pytest.mark.asyncio
  async def test_request_advisor_at_wire_runs_whole_pipeline_signoff() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      svc = MagicMock()
      svc.run_signoff_checkpoint = AsyncMock(
          return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: review this")
      )
      turn_response = {
          "chosen": None, "edited_values": None, "custom_inputs": None,
          "accepted_step_index": None, "edit_step_index": None,
          "control_signal": ControlSignal.REQUEST_ADVISOR,
      }
      _s, guided, next_turn = await _dispatch_guided_respond(
          state=state, guided=session, current_step=GuidedStep.STEP_4_WIRE,
          current_turn_type=TurnType.CONFIRM_WIRING, turn_response=turn_response,
          catalog=MagicMock(), recorder=BufferingRecorder(), user_id="u1",
          data_dir=None, session_engine=None, session_id="s1",
          blob_service=MagicMock(), model="m", temperature=None, seed=None,
          composer_service=svc, advisor_checkpoint_max_passes=3,
      )
      svc.run_signoff_checkpoint.assert_awaited_once()
      assert guided.terminal is None  # on-demand review never auto-completes on a FLAG
      assert next_turn is not None
      assert "review this" in next_turn["payload"]["advisor_findings"]


  @pytest.mark.asyncio
  async def test_request_advisor_at_step3_still_resolves_chain(monkeypatch) -> None:
      # Regression guard: the existing STEP_3 chain re-solve path must remain.
      import elspeth.web.sessions.routes._helpers as helpers

      called = {}

      async def fake_solve(**kwargs):
          called["site"] = kwargs.get("site")
          return None, kwargs["session"]

      monkeypatch.setattr(helpers, "solve_chain_with_auto_drop", fake_solve)
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE, at_step3=True)
      svc = MagicMock()
      svc.run_signoff_checkpoint = AsyncMock(side_effect=AssertionError("wire signoff must not run at step3"))
      turn_response = {
          "chosen": None, "edited_values": None, "custom_inputs": None,
          "accepted_step_index": None, "edit_step_index": None,
          "control_signal": ControlSignal.REQUEST_ADVISOR,
      }
      await _dispatch_guided_respond(
          state=state, guided=session, current_step=GuidedStep.STEP_3_TRANSFORMS,
          current_turn_type=TurnType.PROPOSE_CHAIN, turn_response=turn_response,
          catalog=MagicMock(), recorder=BufferingRecorder(), user_id="u1",
          data_dir=None, session_engine=None, session_id="s1",
          blob_service=MagicMock(), model="m", temperature=None, seed=None,
          composer_service=svc, advisor_checkpoint_max_passes=3,
      )
      assert "step_3_request_advisor_solve" in (called.get("site") or "")
      svc.run_signoff_checkpoint.assert_not_awaited()
  ```
  > Fixture note: extend `_wire_fixtures.make_wire_ready_session_and_state` with
  > an `at_step3=False` kwarg that, when True, returns a STEP_3-positioned
  > session with a staged `step_3_proposal` + `step_1_result`/`step_2_result`
  > so the existing STEP_3 re-solve branch is reachable. (P3 owns the wire
  > fixture; if absent, build it locally per P5.6's note.)
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_request_advisor_escape.py -q`
  Expected failure: the wire-stage `REQUEST_ADVISOR` branch does not exist yet, so `next_turn` is `None`/`run_signoff_checkpoint` is not awaited → `AssertionError: Expected 'run_signoff_checkpoint' to have been awaited once` (or an unhandled control → 400).
- [ ] **Step 3: Add the wire-stage `REQUEST_ADVISOR` branch.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, at the TOP of the `if current_step is GuidedStep.STEP_4_WIRE:` branch (before the plain confirm handling from P5.6), add:
  ```python
              control = turn_response["control_signal"]
              if control is ControlSignal.REQUEST_ADVISOR:
                  # On-demand whole-pipeline checkpoint (D6). Subject to the SAME
                  # persisted pass budget as the auto sign-off so a learner cannot
                  # spin the advisor unbounded. Never auto-completes — it always
                  # re-emits the wire turn with the findings so the user decides.
                  from elspeth.web.composer.guided.signoff import run_wire_signoff

                  max_passes = advisor_checkpoint_max_passes  # P5.4 dispatcher param
                  guided, decision = await run_wire_signoff(
                      session=guided,
                      state=state,
                      session_id=session_id,
                      recorder=recorder,
                      composer_service=composer_service,
                      max_passes=max_passes,
                      acknowledged_unavailable=False,
                      progress=None,
                  )
                  next_turn = build_step_4_wire_turn(
                      state,
                      catalog=catalog,
                      advisor_findings=decision.findings_text,
                      signoff_outcome=decision.outcome.value,
                  )
                  new_record = TurnRecord(
                      step=GuidedStep.STEP_4_WIRE,
                      turn_type=TurnType.CONFIRM_WIRING,
                      payload_hash=stable_hash(next_turn["payload"]),
                      response_hash=None,
                      emitter="server",
                  )
                  emit_turn_emitted(
                      recorder,
                      step=GuidedStep.STEP_4_WIRE,
                      turn_type=TurnType.CONFIRM_WIRING,
                      payload_hash=stable_hash(next_turn["payload"]),
                      payload_payload_id="",
                      emitter="server",
                      composition_version=state.version,
                      actor=user_id,
                  )
                  guided = _replace(guided, history=(*guided.history, new_record))
                  return state, guided, next_turn
  ```
  > The STEP_3 branch at :3342 is LEFT EXACTLY AS-IS. Confirm the same
  > `max_passes` threading decision from P5.6 (settings kwarg vs service
  > settings) is reused here so both wire-stage advisor calls share the bound.
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/sessions/routes/test_request_advisor_escape.py -q`
  Expected: `2 passed`.
- [ ] **Step 5: Confirm the STEP_3 re-solve suite is untouched.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web -q -k "step_3 and (advisor or reject or re_solve or chain)"`
  Expected: existing STEP_3 chain re-solve tests still `passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_request_advisor_escape.py && git commit -m "feat(web/sessions): wire-stage REQUEST_ADVISOR whole-pipeline escape, step-3 re-solve preserved (P5.8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.9: Correct the stale "Disabled by default" advisor prose

**Files:**
- Modify: `src/elspeth/web/composer/tools/_dispatch.py` (the advisor tool description at :129)
- Create: `tests/unit/web/composer/tools/test_advisor_tool_prose.py`

**Interfaces:**
- Produces: corrected operator-facing tool prose (no behaviour change). Pins the END sign-off reality contradicting "Disabled by default".

> §B3 (last paragraph): `tools/_dispatch.py:129` says the advisor is "Disabled
> by default" — that contradicts the mandatory-advisor END sign-off this phase
> wires. The advisor on-demand escape budget is real and the END checkpoint is
> profile-gated (always on for the tutorial), so the "Disabled by default" line
> is stale and must go.

- [ ] **Step 1: Write the failing prose test.**
  Create `tests/unit/web/composer/tools/test_advisor_tool_prose.py`. The advisor
  tool definition is a frozen module constant `_REQUEST_ADVISOR_HINT_DEFINITION`
  (`_dispatch.py:112`, a `Mapping[str, Any]`), NOT a builder function — assert
  directly on its top-level `"description"` value:
  ```python
  """Phase P5.9 — the advisor tool description no longer claims 'Disabled by default'."""

  from __future__ import annotations

  from elspeth.web.composer.tools._dispatch import _REQUEST_ADVISOR_HINT_DEFINITION


  def test_advisor_tool_prose_not_stale() -> None:
      description = _REQUEST_ADVISOR_HINT_DEFINITION["description"]
      assert isinstance(description, str)
      assert "Disabled by default" not in description
      # The mandatory END sign-off is profile-gated and runs independently of the
      # on-demand escape budget; the corrected prose says so.
      assert "operator-configured" in description
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/tools/test_advisor_tool_prose.py -q`
  Expected failure: `AssertionError: assert 'Disabled by default' not in '...Disabled by default; only available when the operator has explicitly enabled it.'`.
- [ ] **Step 3: Fix the prose.**
  In `src/elspeth/web/composer/tools/_dispatch.py`, in the `_REQUEST_ADVISOR_HINT_DEFINITION`
  constant's top-level `"description"` string, replace the trailing two lines
  (`:129-130`, currently
  `"as a substitute for reading validator output. Disabled by default; "` followed by
  `"only available when the operator has explicitly enabled it."`) with:
  ```python
            "as a substitute for reading validator output. Availability is "
            "operator-configured; the mandatory END sign-off checkpoint runs "
            "independently of this on-demand escape."
  ```
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/tools/test_advisor_tool_prose.py -q`
  Expected: `1 passed`.
- [ ] **Step 5: Refresh the plugin/tool source hash if the gate requires it, then commit.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add src/elspeth/web/composer/tools/_dispatch.py tests/unit/web/composer/tools/test_advisor_tool_prose.py && git commit -m "docs(composer): correct stale 'Disabled by default' advisor prose (P5.9)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.10: Phase gate sweep — ruff, mypy, full advisor/wire suite

**Files:** none (verification only).

- [ ] **Step 1: ruff on every file this phase touched.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run ruff check src/elspeth/web/composer/protocol.py src/elspeth/web/composer/service.py src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/tools/_dispatch.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py`
  Expected: `All checks passed!`.
- [ ] **Step 2: mypy on the new module + the touched route helper.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run mypy src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/protocol.py`
  Expected: `Success: no issues found`.
- [ ] **Step 3: Run the full phase test set.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py tests/unit/web/composer/test_run_signoff_checkpoint_impl.py tests/unit/web/composer/guided/test_signoff_classifier.py tests/unit/web/composer/guided/test_wire_signoff_runner.py tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py tests/unit/web/sessions/routes/test_request_advisor_escape.py tests/unit/web/composer/tools/test_advisor_tool_prose.py -q`
  Expected: all `passed` (the full P5 suite green together).
- [ ] **Step 4: Run the inherited advisor-checkpoint regression suite.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py -q`
  Expected: all `passed` — the freeform compose-loop END gate is untouched by this phase (the new public method is a thin façade; the wire gate is a separate dispatch surface).
- [ ] **Step 5: wardline trust-boundary scan (the gate touches external-input-adjacent route code).**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && wardline scan src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/composer/guided/signoff.py --fail-on ERROR`
  Expected: exit 0 (clean). The escape ack reads `turn_response["custom_inputs"]` (a guided protocol enum value, not free text) and the checkpoint forwards no user text into the advisor — fix any finding at the boundary if one appears.
- [ ] **Step 6: No commit (verification gate). If any step failed, return to the owning task.**

---

## Phase P6 — Entry protocol + profile lifecycle + concurrency

> **Phase dependency note (read first).** This phase CONSUMES symbols owned by **P0**:
> `WorkflowProfile`, `EMPTY_PROFILE`, `TUTORIAL_PROFILE`, `WorkflowProfileKind`
> (all in `src/elspeth/web/composer/guided/profile.py`), the `GuidedSession.profile`
> field + the `GuidedSession.initial(profile=...)` keyword signature + the v6
> `to_dict`/`from_dict` round-trip (`composer/guided/state_machine.py`). **Do not author
> any of those here.** If P0 has not landed, each task below will fail its run-to-fail
> with an `ImportError`/`AttributeError` that names the missing P0 symbol — that is the
> correct signal to land P0 first, not to re-create the symbol in this phase.
>
> This phase OWNS: `WorkflowProfileResponse` (Pydantic, `sessions/schemas.py`); the
> `profile` field on `GuidedSessionResponse`; the TS `WorkflowProfile` interface + the
> `profile` field on the TS `GuidedSession`; `startGuidedSession` (`client.ts`); the
> `POST /{session_id}/guided/start` route; the `_strip_guided_profile_in_meta` fork
> helper in `sessions/service.py`; the new optional `step_index` field on
> `GuidedRespondRequest` + its 409 guard in `post_guided_respond`.

All file paths are relative to `src/elspeth/web/` unless prefixed. Run every `pytest`
command from the repo root (`/home/john/elspeth/.claude/worktrees/tutorial-staged-recut`).
Frontend commands run from `src/elspeth/web/frontend`.

---

### Task P6.1: `WorkflowProfileResponse` Pydantic model + `profile` field on `GuidedSessionResponse`

**Files:**
- Modify: `sessions/schemas.py` (add `WorkflowProfileResponse` after `ChatTurnResponse` ~:320; add `profile` field to `GuidedSessionResponse` :323)

**Interfaces:**
- Consumes: `_StrictResponse` (`sessions/schemas.py:39`, `model_config = ConfigDict(strict=True, extra="forbid")`).
- Produces: `class WorkflowProfileResponse(_StrictResponse)` with fields `coaching: bool`, `bookends: bool`, `recipe_match: bool`, `advisor_checkpoints: bool` (wire-visible subset; **`entry_seed` is NOT included** — consumed server-side at start). `GuidedSessionResponse.profile: WorkflowProfileResponse | None = None` (`None` == empty/live-guided profile).

- [ ] **Step 1: Write the failing test for `WorkflowProfileResponse` shape + strictness.**
  Append to `tests/unit/web/sessions/test_schemas.py`:
  ```python
  def test_workflow_profile_response_wire_subset_and_strict() -> None:
      from elspeth.web.sessions.schemas import WorkflowProfileResponse

      model = WorkflowProfileResponse(
          coaching=True, bookends=True, recipe_match=True, advisor_checkpoints=True
      )
      dumped = model.model_dump()
      # Exactly the wire-visible subset — entry_seed is consumed server-side, never on the wire.
      assert set(dumped.keys()) == {"coaching", "bookends", "recipe_match", "advisor_checkpoints"}
      # Strict: unknown key rejected.
      import pydantic
      with pytest.raises(pydantic.ValidationError):
          WorkflowProfileResponse(
              coaching=True, bookends=True, recipe_match=True,
              advisor_checkpoints=True, entry_seed="leak",
          )
      # Strict: coercion rejected (string into bool).
      with pytest.raises(pydantic.ValidationError):
          WorkflowProfileResponse(
              coaching="yes", bookends=True, recipe_match=True, advisor_checkpoints=True
          )
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_schemas.py::test_workflow_profile_response_wire_subset_and_strict -x`
  Expected: `ImportError: cannot import name 'WorkflowProfileResponse' from 'elspeth.web.sessions.schemas'`.
- [ ] **Step 3: Add `WorkflowProfileResponse`.**
  In `sessions/schemas.py`, immediately after `class ChatTurnResponse(_StrictResponse):` block (ends ~:320, before `class GuidedSessionResponse`), insert:
  ```python
  class WorkflowProfileResponse(_StrictResponse):
      """Wire-visible subset of a server-owned WorkflowProfile.

      Mirrors :class:`elspeth.web.composer.guided.profile.WorkflowProfile` MINUS
      ``entry_seed`` — the seed is consumed server-side at ``POST /guided/start``
      and must never ride the GET wire (it is the cache-key discriminator, not a
      render input). ``None`` at the parent ``GuidedSessionResponse.profile``
      level means the empty/live-guided profile (no coaching, no bookends).
      """

      coaching: bool
      bookends: bool
      recipe_match: bool
      advisor_checkpoints: bool
  ```
- [ ] **Step 4: Add the `profile` field to `GuidedSessionResponse`.**
  In `sessions/schemas.py`, in `class GuidedSessionResponse(_StrictResponse):`, after `chat_turn_seq: int` (:337), add:
  ```python
      # Server-owned WorkflowProfile (wire-visible subset). ``None`` for the
      # empty/live-guided profile — a real value only rides the tutorial
      # (and any future non-empty) profile. Defaulted to ``None`` because the
      # majority of GuidedSessionResponse construction sites carry the empty
      # profile; the start/GET path overrides it explicitly. Absence == empty
      # profile is semantically honest here (unlike chat_history, where a
      # forgotten thread would *hide* real history — an empty profile has no
      # render-bearing content to hide).
      profile: WorkflowProfileResponse | None = None
  ```
- [ ] **Step 5: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_schemas.py::test_workflow_profile_response_wire_subset_and_strict -x`
  Expected: `1 passed`.
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/sessions/schemas.py tests/unit/web/sessions/test_schemas.py && git commit -m "feat(sessions): add WorkflowProfileResponse + profile field on GuidedSessionResponse

P6.1 — wire-visible WorkflowProfile subset (coaching/bookends/recipe_match/
advisor_checkpoints; entry_seed stays server-side). Defaulted None == empty
profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.2: TS `WorkflowProfile` interface + `profile` field on TS `GuidedSession`

**Files:**
- Modify: `frontend/src/types/guided.ts` (add `WorkflowProfile` interface near :95; add `profile` to `GuidedSession` :89)

**Interfaces:**
- Produces: `export interface WorkflowProfile { coaching: boolean; bookends: boolean; recipe_match: boolean; advisor_checkpoints: boolean }`; `GuidedSession.profile: WorkflowProfile | null`.

- [ ] **Step 1: Write the failing Vitest test.**
  Append to `frontend/src/types/guided.test.ts` (create the file if absent — it is a pure type-shape assertion; see Step 3 for the import surface):
  ```typescript
  import { describe, it, expect } from "vitest";
  import type { WorkflowProfile, GuidedSession } from "@/types/guided";

  describe("WorkflowProfile wire type", () => {
    it("carries the four wire-visible boolean flags and rides GuidedSession.profile", () => {
      const profile: WorkflowProfile = {
        coaching: true,
        bookends: true,
        recipe_match: true,
        advisor_checkpoints: true,
      };
      const session: Pick<GuidedSession, "profile"> = { profile };
      expect(session.profile).not.toBeNull();
      // null is the empty/live-guided profile.
      const empty: Pick<GuidedSession, "profile"> = { profile: null };
      expect(empty.profile).toBeNull();
    });
  });
  ```
- [ ] **Step 2: Run to fail.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/types/guided.test.ts`
  Expected failure: TS compile error `Module '"@/types/guided"' has no exported member 'WorkflowProfile'`.
- [ ] **Step 3: Add the `WorkflowProfile` interface.**
  In `frontend/src/types/guided.ts`, immediately before `export interface GuidedSession {` (:89), insert:
  ```typescript
  /**
   * Wire: WorkflowProfileResponse (schemas.py — WorkflowProfileResponse).
   * Server-owned workflow profile, wire-visible subset. `entry_seed` is
   * consumed server-side at POST /guided/start and is NOT on the wire.
   * A `null` `GuidedSession.profile` is the empty/live-guided profile.
   */
  export interface WorkflowProfile {
    coaching: boolean;
    bookends: boolean;
    recipe_match: boolean;
    advisor_checkpoints: boolean;
  }
  ```
- [ ] **Step 4: Add `profile` to `GuidedSession`.**
  In `frontend/src/types/guided.ts`, in `export interface GuidedSession {`, after `chat_turn_seq: number;` (:94), add:
  ```typescript
    /** Server-owned WorkflowProfile, or `null` for the empty/live-guided profile. */
    profile: WorkflowProfile | null;
  ```
- [ ] **Step 5: Run to pass.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/types/guided.test.ts`
  Expected: `1 passed`.
- [ ] **Step 6: Typecheck (guards the new optional field against existing GuidedSession literals).**
  From `src/elspeth/web/frontend`: `npm run typecheck`
  Expected: passes, OR fails at existing object-literal construction sites that build a `GuidedSession` without `profile`. If it fails there, those are mock/fixture sites — add `profile: null` to each. Re-run until clean. (`profile` is required on the TS side, mirroring the always-present wire field — `null`, never absent.)
- [ ] **Step 7: Commit.**
  `git add src/elspeth/web/frontend/src/types/guided.ts src/elspeth/web/frontend/src/types/guided.test.ts && git commit -m "feat(frontend): mirror WorkflowProfile on TS GuidedSession

P6.2 — WorkflowProfile interface (coaching/bookends/recipe_match/
advisor_checkpoints) + GuidedSession.profile: WorkflowProfile | null.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.3: Thread `profile` onto every `GuidedSessionResponse` construction; helper `_workflow_profile_response`

**Files:**
- Modify: `sessions/routes/_helpers.py` (add `_workflow_profile_response` helper)
- Modify: `sessions/routes/composer.py` (six `GuidedSessionResponse(...)` sites: :1650, :1885, :2098, :2416, :2875, :3057 — thread `profile=...`)

**Interfaces:**
- Consumes: `WorkflowProfile` (P0, `composer/guided/profile.py`), `EMPTY_PROFILE` (P0), `WorkflowProfileResponse` (P6.1), `GuidedSession.profile` (P0).
- Produces: `def _workflow_profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None` — returns `None` when `guided.profile == EMPTY_PROFILE`, else a populated `WorkflowProfileResponse`.

- [ ] **Step 1: Write the failing unit test for the helper.**
  Append to `tests/unit/web/sessions/test_routes_split.py` (the route-helper unit suite):
  ```python
  def test_workflow_profile_response_none_for_empty_profile() -> None:
      from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
      from elspeth.web.composer.guided.state_machine import GuidedSession
      from elspeth.web.sessions.routes._helpers import _workflow_profile_response

      empty_session = GuidedSession.initial()  # default = EMPTY_PROFILE
      assert empty_session.profile == EMPTY_PROFILE
      assert _workflow_profile_response(empty_session) is None

      tutorial_session = GuidedSession.initial(profile=TUTORIAL_PROFILE)
      resp = _workflow_profile_response(tutorial_session)
      assert resp is not None
      assert resp.coaching is TUTORIAL_PROFILE.coaching
      assert resp.bookends is TUTORIAL_PROFILE.bookends
      assert resp.recipe_match is TUTORIAL_PROFILE.recipe_match
      assert resp.advisor_checkpoints is TUTORIAL_PROFILE.advisor_checkpoints
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_routes_split.py::test_workflow_profile_response_none_for_empty_profile -x`
  Expected: `ImportError: cannot import name '_workflow_profile_response'`.
- [ ] **Step 3: Add the helper to `_helpers.py`.**
  Near `_initial_composition_state_with_guided_session` (:2183) add:
  ```python
  def _workflow_profile_response(guided: GuidedSession) -> WorkflowProfileResponse | None:
      """Project a GuidedSession's server-owned profile onto the wire subset.

      Returns ``None`` for the empty/live-guided profile (== ``EMPTY_PROFILE``)
      so the wire carries ``profile: null`` for live guided, and a populated
      :class:`WorkflowProfileResponse` (the four render-visible flags;
      ``entry_seed`` is deliberately excluded — it is the server-side cache-key
      discriminator, not a render input) for any non-empty profile.
      """
      if guided.profile == EMPTY_PROFILE:
          return None
      return WorkflowProfileResponse(
          coaching=guided.profile.coaching,
          bookends=guided.profile.bookends,
          recipe_match=guided.profile.recipe_match,
          advisor_checkpoints=guided.profile.advisor_checkpoints,
      )
  ```
  Add the imports at the top of `_helpers.py` (alongside the existing `from elspeth.web.composer.guided.*` and `from elspeth.web.sessions.schemas import ...` blocks):
  `from elspeth.web.composer.guided.profile import EMPTY_PROFILE` and ensure `WorkflowProfileResponse` is imported from `elspeth.web.sessions.schemas`. `GuidedSession` is already imported.
- [ ] **Step 4: Run to pass (helper unit).**
  `uv run pytest tests/unit/web/sessions/test_routes_split.py::test_workflow_profile_response_none_for_empty_profile -x`
  Expected: `1 passed`.
- [ ] **Step 5: Re-export `_workflow_profile_response` into `composer.py`.**
  In `composer.py`'s `from ._helpers import (` block (starts :7), add `_workflow_profile_response,` (keep alpha order near `_validate_step_indices` / `_state_response`).
- [ ] **Step 6: Thread `profile=` onto all six `GuidedSessionResponse(...)` sites.**
  At EACH of `composer.py:1650, 1885, 2098, 2416, 2875, 3057`, the constructor builds from a local `guided` (or `new_guided`) variable. Add `profile=_workflow_profile_response(<that local>),` as the final kwarg. For the site at :1885 (`post_guided_reenter`, builds from `new_guided`) use `profile=_workflow_profile_response(new_guided)`; for :2098 (`new_guided`), :2416 (`guided`), etc. — match the variable already feeding `step=<x>.step.value`. (Grep `GuidedSessionResponse(` in the file to confirm the six; each is preceded by `step=<var>.step.value` — use `<var>`.)
- [ ] **Step 7: Run the existing guided-route suite to confirm no regression in the six sites.**
  `uv run pytest tests/unit/web/sessions/test_routes.py -k "guided" -q`
  Expected: all previously-passing guided route tests still pass (now also returning `profile` in the payload; `_StrictResponse` accepts it because the field exists). If any test asserts an exact `model_dump()` key set, update it to include `"profile"`.
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer.py tests/unit/web/sessions/test_routes_split.py && git commit -m "feat(sessions): surface WorkflowProfile on every GuidedSessionResponse

P6.3 — _workflow_profile_response (None for empty profile) threaded onto all
six GuidedSessionResponse constructors so GET/respond/chat/reenter carry profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.4: `POST /{session_id}/guided/start` — idempotent, closed-enum profile, persists

**Files:**
- Modify: `sessions/schemas.py` (add `StartGuidedRequest` request model)
- Modify: `sessions/routes/composer.py` (add `post_guided_start` route after `post_guided_reenter` :1796–~:1918; thread profile through `_initial_composition_state_with_guided_session`)

**Interfaces:**
- Consumes: `WorkflowProfileKind` (P0 closed StrEnum `{LIVE="live", TUTORIAL="tutorial"}`), `EMPTY_PROFILE` / `TUTORIAL_PROFILE` (P0), `_initial_composition_state_with_guided_session(*, profile=...)` (P0-modified signature, `_helpers.py:2183`), `GuidedSession.profile` (P0).
- Produces: `class StartGuidedRequest(_RequestModel): profile: str = "live"` (plain string at the boundary, validated against the closed `WorkflowProfileKind` enum in the handler — mirrors the `control_signal` / `step_index` graceful-stale-client convention). Route `POST /{session_id}/guided/start` → `GetGuidedResponse`.

- [ ] **Step 1: Write the failing integration test (idempotency + persistence).**
  Create `tests/unit/web/sessions/test_guided_start.py`:
  ```python
  """POST /guided/start — idempotent profile-seeded guided entry (P6, §4.3, D16)."""

  from __future__ import annotations

  import uuid

  import pytest
  import structlog
  from fastapi import FastAPI
  from sqlalchemy.pool import StaticPool

  from elspeth.web.auth.middleware import get_current_user
  from elspeth.web.auth.models import UserIdentity
  from elspeth.web.config import WebSettings
  from elspeth.web.middleware.rate_limit import ComposerRateLimiter
  from elspeth.web.sessions.engine import create_session_engine
  from elspeth.web.sessions.routes import create_session_router
  from elspeth.web.sessions.schema import initialize_session_schema
  from elspeth.web.sessions.service import SessionServiceImpl
  from elspeth.web.sessions.telemetry import build_sessions_telemetry
  from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


  def _make_app(tmp_path, user_id="alice"):
      engine = create_session_engine(
          "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
      )
      initialize_session_schema(engine)
      service = SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))
      app = FastAPI()
      identity = UserIdentity(user_id=user_id, username=user_id)

      async def mock_user():
          return identity

      app.dependency_overrides[get_current_user] = mock_user
      app.state.session_service = service
      app.state.session_engine = engine
      app.state.catalog_service = None
      app.state.settings = WebSettings(
          data_dir=tmp_path,
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          shareable_link_signing_key=b"\x00" * 32,
      )
      app.state.composer_service = None
      app.state.rate_limiter = ComposerRateLimiter(limit=100)
      app.state.composer_progress_registry = ComposerProgressRegistry()
      app.include_router(create_session_router())
      return app, service


  from elspeth.web.composer.progress import ComposerProgressRegistry  # noqa: E402


  @pytest.mark.asyncio
  async def test_guided_start_seeds_tutorial_profile_and_persists(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      assert resp.status_code == 200
      body = resp.json()
      # Wire carries the tutorial profile (advisor_checkpoints on, bookends on).
      assert body["guided_session"]["profile"] is not None
      assert body["guided_session"]["profile"]["advisor_checkpoints"] is True
      assert body["guided_session"]["profile"]["bookends"] is True

      # Persisted: GET /guided reads back the persisted tutorial profile.
      get_resp = client.get(f"/api/sessions/{session.id}/guided")
      assert get_resp.status_code == 200
      assert get_resp.json()["guided_session"]["profile"]["advisor_checkpoints"] is True


  @pytest.mark.asyncio
  async def test_guided_start_is_idempotent(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      first = client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      assert first.status_code == 200

      # A second start MUST return the existing session unchanged — never re-init.
      second = client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "live"})
      assert second.status_code == 200
      # The persisted tutorial profile is preserved (the second 'live' start did not re-init).
      assert second.json()["guided_session"]["profile"] is not None
      assert second.json()["guided_session"]["profile"]["advisor_checkpoints"] is True

      # Exactly one composition_state version was written (idempotent — no double-create).
      from sqlalchemy import text

      with service._engine.connect() as conn:
          versions = conn.execute(
              text("SELECT COUNT(*) FROM composition_states WHERE session_id = :sid"),
              {"sid": str(session.id)},
          ).scalar()
      assert versions == 1


  @pytest.mark.asyncio
  async def test_guided_start_rejects_unknown_profile_kind(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")

      resp = client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "superuser"})
      assert resp.status_code == 400
      assert "profile" in resp.json()["detail"].lower()


  @pytest.mark.asyncio
  async def test_guided_start_unowned_session_404(tmp_path) -> None:
      app, service = _make_app(tmp_path, user_id="alice")
      client = TestClient(app)
      # A session id that alice does not own (never created here).
      resp = client.post(f"/api/sessions/{uuid.uuid4()}/guided/start", json={"profile": "tutorial"})
      assert resp.status_code == 404
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -x`
  Expected: `404 Not Found` on the start POST (route absent) → assertion error on `status_code == 200` (or a routing 405). This confirms the endpoint does not yet exist.
- [ ] **Step 3: Add `StartGuidedRequest` to `schemas.py`.**
  After `class RevertStateRequest(_RequestModel):` (~:268) add:
  ```python
  class StartGuidedRequest(_RequestModel):
      """Request body for POST /api/sessions/{id}/guided/start.

      ``profile`` is a closed-enum discriminator (``WorkflowProfileKind``) — the
      client supplies the discriminator string and the SERVER constructs the
      WorkflowProfile object, so a client cannot inject an arbitrary profile or
      weaken gating (D13/§4.3). It is a plain ``str`` (not the enum) so a stale
      client sending an unknown value fails with a route-handler 400 carrying a
      clear message rather than a Pydantic 422 — mirroring ``control_signal`` /
      ``step_index``. Defaults to ``"live"`` (the empty/canonical profile).
      """

      profile: str = "live"
  ```
- [ ] **Step 4: Add the `post_guided_start` route to `composer.py`.**
  Immediately after the `post_guided_reenter` handler (ends ~:1918, before `@router.post("/{session_id}/guided/respond"...)` :1920) insert:
  ```python
      @router.post("/{session_id}/guided/start", response_model=GetGuidedResponse)
      async def post_guided_start(
          session_id: UUID,
          body: StartGuidedRequest,
          request: Request,
          user: UserIdentity = Depends(get_current_user),  # noqa: B008
      ) -> GetGuidedResponse:
          """Seed a guided session with a server-owned WorkflowProfile.

          The client supplies a closed-enum ``profile`` discriminator
          (``WorkflowProfileKind``); the SERVER maps it to the concrete profile
          object and persists the resulting GuidedSession, so a client cannot
          inject an arbitrary profile or weaken the advisor gate (D13/§4.3).

          **Idempotent (D16):** a second start for a session that ALREADY has a
          persisted GuidedSession returns the existing session unchanged — it
          never re-initialises or double-creates. GET /guided then reads the
          persisted ``GuidedSession.profile``; the lazy no-arg GET default path
          stays for live guided (empty profile).

          Raises 404 if the session does not exist or belong to the requester.
          Raises 400 if ``profile`` is not a recognised WorkflowProfileKind.
          """
          await _verify_session_ownership(session_id, user, request)
          service: SessionServiceProtocol = request.app.state.session_service
          catalog: CatalogServiceProtocol = request.app.state.catalog_service

          # Tier-3 -> Tier-2 coercion at the profile-kind boundary. A stale client
          # sending an unknown discriminator gets a 400 with a clear message
          # rather than a Pydantic 422; the typed kind then selects a SERVER-owned
          # constant — the client never supplies the profile object.
          try:
              profile_kind = WorkflowProfileKind(body.profile)
          except ValueError as exc:
              raise HTTPException(
                  status_code=400,
                  detail=(
                      f"Unknown profile {body.profile!r}. "
                      f"Valid values: {sorted(k.value for k in WorkflowProfileKind)}."
                  ),
              ) from exc
          profile = TUTORIAL_PROFILE if profile_kind is WorkflowProfileKind.TUTORIAL else EMPTY_PROFILE

          compose_lock = await _get_session_compose_lock_registry(request).get_lock(str(session_id))
          async with compose_lock:
              # Idempotency (D16): if a guided session is already persisted, return
              # it UNCHANGED — never re-init (a second start must not clobber the
              # learner's in-progress wizard or re-seed a fresh profile).
              existing_record = await service.get_current_state(session_id)
              if existing_record is not None:
                  existing_state = _state_from_record(existing_record)
                  if existing_state.guided_session is not None:
                      guided = existing_state.guided_session
                      terminal = guided.terminal
                      return GetGuidedResponse(
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
                              terminal=TerminalStateResponse(
                                  kind=terminal.kind.value,
                                  reason=terminal.reason.value if terminal.reason is not None else None,
                                  pipeline_yaml=terminal.pipeline_yaml,
                              )
                              if terminal is not None
                              else None,
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
                          next_turn=None,
                          terminal=TerminalStateResponse(
                              kind=terminal.kind.value,
                              reason=terminal.reason.value if terminal.reason is not None else None,
                              pipeline_yaml=terminal.pipeline_yaml,
                          )
                          if terminal is not None
                          else None,
                          composition_state=_state_response(existing_record),
                      )

              # No persisted guided session yet: construct the profile-seeded
              # initial state and PERSIST it (so GET /guided reads it back).
              new_state = _initial_composition_state_with_guided_session(profile=profile)
              guided = new_state.guided_session
              if guided is None:  # pragma: no cover — helper always attaches a guided session
                  raise InvariantError("post_guided_start: initial state has no guided_session")
              turn = _build_get_guided_turn(new_state, guided, catalog=catalog)

              new_composer_meta = {"guided_session": guided.to_dict()}
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
                  session_id,
                  state_data,
                  # Start endpoint seeds the canonical guided session for a profile;
                  # ``session_seed`` is the closest existing provenance category for
                  # a fresh server-authored seed state (the closed enum has no
                  # guided-specific value — see merge commit message).
                  provenance="session_seed",
              )

              return GetGuidedResponse(
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
                      type=turn["type"],
                      step_index=turn["step_index"],
                      payload=dict(turn["payload"]),
                  )
                  if turn is not None
                  else None,
                  terminal=None,
                  composition_state=_state_response(state_record_out),
              )
  ```
- [ ] **Step 5: Add the imports to `composer.py`.**
  In the `from ._helpers import (` block add `StartGuidedRequest,`, `WorkflowProfileKind,`, `EMPTY_PROFILE,`, `TUTORIAL_PROFILE,`, `_build_get_guided_turn,` (confirm each is re-exported by `_helpers.py`; if `_helpers.py` does not yet re-export `WorkflowProfileKind`/`EMPTY_PROFILE`/`TUTORIAL_PROFILE`, add `from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE, WorkflowProfileKind` to `_helpers.py` and re-export through its import surface, and add `from elspeth.web.sessions.schemas import StartGuidedRequest` likewise). `_state_from_record`, `_state_response`, `_build_get_guided_turn`, `InvariantError`, `CompositionStateData`, `TurnPayloadResponse`, `TerminalStateResponse` are already imported per the existing `composer.py` import block.
- [ ] **Step 6: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -x`
  Expected: `4 passed`.
- [ ] **Step 7: Lint + type the new code.**
  `uv run ruff check src/elspeth/web/sessions/routes/composer.py src/elspeth/web/sessions/schemas.py && uv run mypy src/elspeth/web/sessions/routes/composer.py`
  Expected: clean.
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/routes/composer.py src/elspeth/web/sessions/schemas.py tests/unit/web/sessions/test_guided_start.py && git commit -m "feat(sessions): POST /guided/start — idempotent closed-enum profile entry

P6.4 — server-constructed WorkflowProfile (LIVE/TUTORIAL), persists the seeded
GuidedSession; idempotent re-start returns the existing session unchanged;
unknown profile -> 400. GET /guided reads the persisted profile.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.5: `client.ts` — `startGuidedSession(sessionId, profileKind)`

**Files:**
- Modify: `frontend/src/api/client.ts` (add `startGuidedSession` near the guided fns ~:589–:646)

**Interfaces:**
- Consumes: `GetGuidedResponse` (already imported from `@/types/guided`).
- Produces: `export async function startGuidedSession(sessionId: string, profileKind: "live" | "tutorial"): Promise<GetGuidedResponse>`.

- [ ] **Step 1: Write the failing Vitest test.**
  Append to `frontend/src/api/client.guided.test.ts` (the client test suite; if a guided-client test file already exists, append there — grep for `getGuided` to find it):
  ```typescript
  import { describe, it, expect, vi, afterEach } from "vitest";
  import { startGuidedSession } from "@/api/client";

  afterEach(() => vi.restoreAllMocks());

  describe("startGuidedSession", () => {
    it("POSTs the profile discriminator to /guided/start", async () => {
      const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(
          JSON.stringify({
            guided_session: {
              step: "step_1_source",
              history: [],
              terminal: null,
              chat_history: [],
              chat_turn_seq: 0,
              profile: { coaching: true, bookends: true, recipe_match: true, advisor_checkpoints: true },
            },
            next_turn: null,
            terminal: null,
            composition_state: null,
          }),
          { status: 200, headers: { "content-type": "application/json" } },
        ),
      );

      const result = await startGuidedSession("sess-1", "tutorial");
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/sessions/sess-1/guided/start",
        expect.objectContaining({
          method: "POST",
          body: JSON.stringify({ profile: "tutorial" }),
        }),
      );
      expect(result.guided_session.profile?.advisor_checkpoints).toBe(true);
    });
  });
  ```
- [ ] **Step 2: Run to fail.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/api/client.guided.test.ts`
  Expected: TS compile error `Module '"@/api/client"' has no exported member 'startGuidedSession'`.
- [ ] **Step 3: Add `startGuidedSession` to `client.ts`.**
  Immediately after the `getGuided` function (ends ~:607, before `respondGuided`), insert:
  ```typescript
  /**
   * Seed a guided session with a server-owned WorkflowProfile.
   *
   * The `profileKind` is a closed-enum discriminator ("live" | "tutorial"); the
   * SERVER constructs the concrete profile object and persists the GuidedSession.
   * Idempotent (D16): a second call for a session that already has a persisted
   * guided session returns the existing session unchanged.
   */
  export async function startGuidedSession(
    sessionId: string,
    profileKind: "live" | "tutorial",
  ): Promise<GetGuidedResponse> {
    const response = await fetch(`/api/sessions/${sessionId}/guided/start`, {
      method: "POST",
      headers: authHeaders("application/json"),
      body: JSON.stringify({ profile: profileKind }),
    });
    return parseResponse<GetGuidedResponse>(response);
  }
  ```
- [ ] **Step 4: Run to pass.**
  From `src/elspeth/web/frontend`: `npm test -- --run src/api/client.guided.test.ts`
  Expected: `1 passed` (plus pre-existing client tests still green).
- [ ] **Step 5: Typecheck.**
  From `src/elspeth/web/frontend`: `npm run typecheck`
  Expected: clean.
- [ ] **Step 6: Commit.**
  `git add src/elspeth/web/frontend/src/api/client.ts src/elspeth/web/frontend/src/api/client.guided.test.ts && git commit -m "feat(frontend): startGuidedSession client (closed-enum profile)

P6.5 — POST /guided/start with profile discriminator; returns GetGuidedResponse.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.6: Fork strip — reset `GuidedSession.profile` to empty inside `fork_session`

**Files:**
- Modify: `sessions/service.py` (add module-level `_strip_guided_profile_in_meta` helper; apply at the two verbatim `composer_meta` copies `:5076` and `:5153`)

**Interfaces:**
- Consumes: `EMPTY_PROFILE` (P0, `composer/guided/profile.py`), `deep_thaw` (already imported `service.py:37`).
- Produces: `def _strip_guided_profile_in_meta(composer_meta: Mapping[str, Any] | None) -> dict[str, Any] | None` — returns a copy of `composer_meta` whose `["guided_session"]["profile"]` is replaced with `EMPTY_PROFILE.to_dict()`; passes `None`/no-guided-session through unchanged.

- [ ] **Step 1: Write the failing service-level test (covers the REAL materialised
  canonical source — proves the strip is in `fork_session`, independent of the
  blob-rewrite path).**
  Append to `tests/unit/web/sessions/test_fork.py` (inside `class TestForkSession`):
  ```python
      @pytest.mark.asyncio
      async def test_fork_strips_tutorial_profile_from_guided_session(self, service) -> None:
          """Forking a tutorial-profile guided session yields the EMPTY profile.

          Critical case (finding 10, rev 4 — CORRECTED). The canonical tutorial
          source MATERIALISES (set_pipeline from ``source.inline_blob``) to a real
          ``json`` source whose ``options`` carry ``blob_ref``
          (``composer/tools/sessions.py:425``), so the route-layer blob-rewrite save
          DOES fire (``rewritten=True``). This fixture uses that real shape on
          purpose: it proves the strip survives EVEN on the path that re-saves the
          state — because the blob-rewrite re-save preserves ``composer_meta``
          verbatim (``sessions/routes/sessions.py:479-480``) and never strips the
          profile. The strip therefore lives in ``fork_session`` (both the :5076
          persist copy and the :5153 return copy) and is independent of
          ``rewritten``. (The earlier "no blob_ref => rewritten=False" framing was a
          false premise — see the spec's two-objects ``blob_ref`` note in §5/B4.)
          """
          from elspeth.web.composer.guided.profile import EMPTY_PROFILE, TUTORIAL_PROFILE
          from elspeth.web.composer.guided.state_machine import GuidedSession

          session = await service.create_session("alice", "Tutorial", "local")
          tutorial_guided = GuidedSession.initial(profile=TUTORIAL_PROFILE)
          state = await service.save_composition_state(
              session.id,
              CompositionStateData(
                  # Materialised canonical URL source (sessions.py:420-427): a real
                  # ``json`` plugin with ``blob_ref`` in options => rewritten=True.
                  # The blob-rewrite save fires but preserves composer_meta verbatim,
                  # so the profile strip must still come from fork_session.
                  sources={
                      "urls": {
                          "plugin": "json",
                          "options": {
                              "path": "composer_blobs/canonical-url-list.json",
                              "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",
                          },
                      }
                  },
                  is_valid=True,
                  composer_meta={"guided_session": tutorial_guided.to_dict()},
              ),
              provenance="session_seed",
          )
          fork_msg = await service.add_message(
              session.id,
              "user",
              "Build this",
              composition_state_id=state.id,
              writer_principal="route_user_message",
          )

          _, _, copied_state = await service.fork_session(
              source_session_id=session.id,
              fork_message_id=fork_msg.id,
              new_message_content="Build something else",
              user_id="alice",
              auth_provider_type="local",
          )

          assert copied_state is not None
          # Returned record (the :5153 copy) carries the EMPTY profile.
          forked_guided = GuidedSession.from_dict(copied_state.composer_meta["guided_session"])
          assert forked_guided.profile == EMPTY_PROFILE
          # And it is PERSISTED that way (the :5076 copy) — re-read from the DB.
          persisted = await service.get_current_state(copied_state.session_id)
          persisted_guided = GuidedSession.from_dict(persisted.composer_meta["guided_session"])
          assert persisted_guided.profile == EMPTY_PROFILE

      @pytest.mark.asyncio
      async def test_fork_without_guided_session_passes_meta_through(self, service) -> None:
          """An ordinary (non-guided) fork is unaffected by the profile strip."""
          session = await service.create_session("alice", "Plain", "local")
          state = await service.save_composition_state(
              session.id,
              CompositionStateData(
                  sources={"s": {"plugin": "csv", "options": {"path": "x.csv"}}},
                  is_valid=True,
                  composer_meta={"repair_turns_used": 2},
              ),
              provenance="session_seed",
          )
          fork_msg = await service.add_message(
              session.id, "user", "Build", composition_state_id=state.id, writer_principal="route_user_message"
          )
          _, _, copied_state = await service.fork_session(
              source_session_id=session.id,
              fork_message_id=fork_msg.id,
              new_message_content="Build other",
              user_id="alice",
              auth_provider_type="local",
          )
          assert copied_state is not None
          # composer_meta passes through verbatim (no guided_session key to strip).
          assert copied_state.composer_meta == {"repair_turns_used": 2}
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_fork.py::TestForkSession::test_fork_strips_tutorial_profile_from_guided_session -x`
  Expected: `AssertionError: assert <TUTORIAL_PROFILE> == <EMPTY_PROFILE>` — the verbatim copy currently carries the tutorial profile through.
- [ ] **Step 3: Add the `_strip_guided_profile_in_meta` helper.**
  In `sessions/service.py`, add a module-level function (near the other module helpers, after the imports / before `class SessionServiceImpl` — or just above `fork_session` if module helpers live in-class; place it module-level so it is import-testable):
  ```python
  def _strip_guided_profile_in_meta(composer_meta: Mapping[str, Any] | None) -> dict[str, Any] | None:
      """Reset a forked GuidedSession's WorkflowProfile to the empty/canonical profile.

      ``composer_meta`` is copied VERBATIM on fork (fork_session :5076 persist
      copy + :5153 return copy). A tutorial profile (canonical seed, coaching,
      advisor checkpoints, bookends) must NOT leak into an ordinary forked
      session (finding 10, rev 4). The route-layer blob-rewrite save preserves
      ``composer_meta`` verbatim (``sessions/routes/sessions.py:479-480``) and
      never strips the profile — and it is route-layer, not part of this service
      method — so the strip MUST live here, inside ``fork_session``, where the
      ``composer_meta`` copies actually happen. (The earlier "canonical source has
      no blob_ref => rewritten=False" rationale was a false premise; the
      materialised canonical source DOES carry ``blob_ref`` — see the spec's
      two-objects ``blob_ref`` note in §5/B4 — but the seam decision does not
      depend on ``rewritten``.)

      Operates at the dict level (not full GuidedSession.from_dict/to_dict) so it
      is resilient to a stale-but-loadable blob: it replaces only the
      ``["guided_session"]["profile"]`` key with ``EMPTY_PROFILE.to_dict()`` and
      passes everything else through. ``None`` / no ``guided_session`` key /
      ``guided_session`` without a ``profile`` key all pass through unchanged.

      (Resume of a tutorial's OWN session — a non-fork path — correctly preserves
      the profile; this strip is fork-only.)
      """
      from elspeth.web.composer.guided.profile import EMPTY_PROFILE

      if composer_meta is None:
          return None
      thawed = dict(deep_thaw(composer_meta))
      guided_raw = thawed.get("guided_session")
      if not isinstance(guided_raw, dict) or "profile" not in guided_raw:
          return thawed
      guided_copy = dict(guided_raw)
      guided_copy["profile"] = EMPTY_PROFILE.to_dict()
      thawed["guided_session"] = guided_copy
      return thawed
  ```
  Confirm `Mapping` and `Any` are imported at the top of `service.py` (they are — used throughout); `deep_thaw` is imported at `:37`.
- [ ] **Step 4: Apply the strip at both verbatim copies in `fork_session`.**
  In `fork_session`, after `source_state_record` is loaded (~:4910) and before `_sync`, compute the stripped meta ONCE:
  ```python
          # Profile strip (finding 10, rev 4): never let a tutorial WorkflowProfile
          # leak into a forked session. Computed once, used by BOTH verbatim
          # composer_meta copies below (the :5076 persist copy and the :5153 return
          # copy). The route-layer blob-rewrite save preserves composer_meta
          # verbatim and never strips the profile (and is not in this service
          # method's path), so the strip must live here — independent of rewritten.
          forked_composer_meta = (
              _strip_guided_profile_in_meta(source_state_record.composer_meta)
              if source_state_record is not None
              else None
          )
  ```
  Then change the `:5076` site (inside `StatePayload(... CompositionStateData(... composer_meta=source_state_record.composer_meta))`) to `composer_meta=forked_composer_meta`, and the `:5153` site (the returned `CompositionStateRecord(... composer_meta=source_state_record.composer_meta)`) to `composer_meta=forked_composer_meta`.
- [ ] **Step 5: Run to pass.**
  `uv run pytest tests/unit/web/sessions/test_fork.py -k "profile or passes_meta_through" -x`
  Expected: `2 passed`.
- [ ] **Step 6: Run the full fork suite (no regression in copy provenance / state lineage).**
  `uv run pytest tests/unit/web/sessions/test_fork.py -q`
  Expected: all pass.
- [ ] **Step 7: Lint + type.**
  `uv run ruff check src/elspeth/web/sessions/service.py && uv run mypy src/elspeth/web/sessions/service.py`
  Expected: clean.
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_fork.py && git commit -m "fix(sessions): strip WorkflowProfile on fork (no tutorial-profile leak)

P6.6 — _strip_guided_profile_in_meta resets guided_session.profile to EMPTY at
BOTH verbatim composer_meta copies in fork_session (:5076 persist + :5153 return).
The route blob-rewrite save preserves composer_meta verbatim and never strips the
profile (and is route-layer, not part of fork_session), so the strip must live in
fork_session — independent of rewritten. Closes finding 10.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.7: `/guided/respond` optimistic-concurrency `step_index` 409 guard

**Files:**
- Modify: `sessions/schemas.py` (add optional `step_index: str | None = None` to `GuidedRespondRequest` :365)
- Modify: `frontend/src/types/guided.ts` (add `step_index?: GuidedStep | null` to `GuidedRespondRequest` :115)
- Modify: `sessions/routes/composer.py` (add the 409 guard in `post_guided_respond` after the terminal-409 guard :2131 / before the dispatcher)

**Interfaces:**
- Consumes: `GuidedStep` (already imported), the existing terminal-409 pattern (`composer.py:2131`) and the `/guided/chat` step-mismatch 409 pattern (`composer.py:2658`).
- Produces: `GuidedRespondRequest.step_index: str | None = None` — when supplied, the handler coerces it to `GuidedStep` and 409s if it does not match `guided.step` (carries an expected step on the wire confirm, D16). Absent (`None`) preserves the existing no-guard behaviour (back-compat with non-wire turns that don't carry a step).

- [ ] **Step 1: Write the failing integration test (stale step_index → 409).**
  Append to `tests/unit/web/sessions/test_guided_start.py` (reuses the `_make_app` harness):
  ```python
  @pytest.mark.asyncio
  async def test_guided_respond_stale_step_index_409(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")
      # Seed a guided session (step_1_source) and emit its initial turn.
      client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      client.get(f"/api/sessions/{session.id}/guided")

      # Respond carrying a step_index that does NOT match the session's current
      # step (the wizard is at step_1_source). Optimistic-concurrency mismatch -> 409.
      resp = client.post(
          f"/api/sessions/{session.id}/guided/respond",
          json={"step_index": "step_3_transforms", "chosen": ["csv"]},
      )
      assert resp.status_code == 409
      assert "step_index" in resp.json()["detail"]

  @pytest.mark.asyncio
  async def test_guided_respond_unknown_step_index_400(tmp_path) -> None:
      app, service = _make_app(tmp_path)
      client = TestClient(app)
      session = await service.create_session("alice", "T", "local")
      client.post(f"/api/sessions/{session.id}/guided/start", json={"profile": "tutorial"})
      client.get(f"/api/sessions/{session.id}/guided")

      resp = client.post(
          f"/api/sessions/{session.id}/guided/respond",
          json={"step_index": "step_99_bogus", "chosen": ["csv"]},
      )
      assert resp.status_code == 400
      assert "step_index" in resp.json()["detail"].lower()
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -k "stale_step_index or unknown_step_index" -x`
  Expected: the stale-index POST returns 200/400 (not 409) because no guard exists yet → assertion fails on `status_code == 409`. (The unknown-index test may already 422 from Pydantic if the field is absent — confirming the field must be added.)
- [ ] **Step 3: Add the optional `step_index` to `GuidedRespondRequest`.**
  In `sessions/schemas.py`, in `class GuidedRespondRequest(_RequestModel):` (:365), after `control_signal: str | None = None` (:384), add:
  ```python
      # Optimistic-concurrency token (D16): the client's expected current step.
      # When present, the route 409s if it does not match the session's live
      # ``guided.step`` (the wizard advanced under the client between read and
      # write) — mirroring the ``/guided/chat`` step-mismatch guard. A plain
      # ``str`` (not the enum) so a stale client sending an unknown value fails
      # with a route-handler 400 rather than a Pydantic 422 — same convention as
      # ``control_signal``. ``None`` preserves the pre-D16 unguarded behaviour
      # for turns that do not carry an expected step.
      step_index: str | None = None
  ```
- [ ] **Step 4: Add the 409 guard in `post_guided_respond`.**
  In `composer.py`, in `post_guided_respond`, after the generic terminal-409 guard (`if guided.terminal is not None: raise HTTPException(409...)` ending :2135) and before `current_step = guided.step` (:2140), insert:
  ```python
                  # Optimistic-concurrency guard (D16): if the client carried an
                  # expected step, reject a mismatch with 409 (the wizard advanced
                  # under the client between read and write) — the same guard
                  # ``/guided/chat`` already has (composer.py:2658). A stale client
                  # sending an unknown value gets a 400, not a Pydantic 422,
                  # mirroring control_signal. ``None`` (field absent) skips the
                  # guard for turns that do not carry an expected step.
                  if body.step_index is not None:
                      try:
                          expected_step = GuidedStep(body.step_index)
                      except ValueError as exc:
                          raise HTTPException(
                              status_code=400,
                              detail=(
                                  f"Unknown step_index {body.step_index!r}. "
                                  f"Valid values: {sorted(s.value for s in GuidedStep)}."
                              ),
                          ) from exc
                      if expected_step is not guided.step:
                          raise HTTPException(
                              status_code=409,
                              detail=(
                                  f"step_index {expected_step.value!r} does not match the session's "
                                  f"current step {guided.step.value!r}. Re-fetch GET "
                                  f"/api/sessions/{{id}}/guided and retry."
                              ),
                          )
  ```
- [ ] **Step 5: Run to pass (backend).**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py -k "step_index" -x`
  Expected: `2 passed`. Also confirm no regression: `uv run pytest tests/unit/web/sessions/test_routes.py -k "guided" -q` → all pass (existing respond callers omit `step_index`, so the guard is skipped).
- [ ] **Step 6: Mirror `step_index` on the TS `GuidedRespondRequest`.**
  In `frontend/src/types/guided.ts`, in `export interface GuidedRespondRequest {` (:115), after `control_signal: ControlSignal | null;` (:122), add:
  ```typescript
    /**
     * Optimistic-concurrency token: the client's expected current step. When
     * present the server 409s on mismatch (the wizard advanced under the
     * client). Optional — omit for non-wire turns that don't carry a step.
     */
    step_index?: GuidedStep | null;
  ```
- [ ] **Step 7: Frontend typecheck + the mirror gate.**
  From `src/elspeth/web/frontend`: `npm run typecheck` (expected clean).
  From the repo root: `uv run python scripts/cicd/check_slot_type_cross_language.py` (the SlotType / guided.ts mirror gate — this work edits `guided.ts`; expected: passes, since `step_index` is a request field, not a `SlotType`/`TurnType`/`GuidedStep` enum member).
- [ ] **Step 8: Commit.**
  `git add src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes/composer.py src/elspeth/web/frontend/src/types/guided.ts tests/unit/web/sessions/test_guided_start.py && git commit -m "feat(sessions): optimistic-concurrency step_index 409 on /guided/respond

P6.7 — optional step_index on GuidedRespondRequest; route 409s on mismatch
(wizard advanced under client), 400 on unknown value — parity with /guided/chat.
TS mirror added. Closes D16.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P6.8: Phase P6 verification sweep

**Files:** none (verification only).

**Interfaces:** none.

- [ ] **Step 1: Run the full P6 backend test set.**
  `uv run pytest tests/unit/web/sessions/test_guided_start.py tests/unit/web/sessions/test_fork.py tests/unit/web/sessions/test_schemas.py tests/unit/web/sessions/test_routes_split.py tests/unit/web/sessions/test_routes.py -q`
  Expected: all pass, zero failures.
- [ ] **Step 2: Ruff check + format-check on every touched file.**
  `uv run ruff check src/ tests/ && uv run ruff format --check src/elspeth/web/sessions/routes/composer.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py`
  Expected: `All checks passed!` / no files would be reformatted. (If format-check flags a file, run `uv run ruff format <file>` and re-commit.)
- [ ] **Step 3: mypy on the touched modules.**
  `uv run mypy src/elspeth/web/sessions/routes/composer.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py`
  Expected: `Success: no issues found`.
- [ ] **Step 4: elspeth-lints trust gates over the touched surface.**
  `PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth`
  Expected: no NEW findings attributable to P6. The start endpoint validates the profile discriminator against a closed enum and the SERVER constructs the profile object (no Tier-3 profile object crosses the boundary); `step_index` is coerced through `GuidedStep(...)` with a 400 on failure (a clean Tier-3→Tier-2 coercion). If a finding lands on a P6 line, fix at the boundary; do not waiver. (Pre-existing fingerprint_baseline drift is operator-owned — state once, do not bless.)
- [ ] **Step 5: Frontend gates.**
  From `src/elspeth/web/frontend`: `npm run typecheck && npm test -- --run && npm run build`
  Expected: all green.
- [ ] **Step 6: SlotType / guided.ts mirror gate + wardline.**
  Repo root: `uv run python scripts/cicd/check_slot_type_cross_language.py` (expected pass) and `wardline scan . --fail-on ERROR` (expected exit 0; P6 touches the profile/start trust boundary — confirm the closed-enum discriminator + server-constructed profile keeps the taint flow clean; fix at the boundary if a finding lands).
- [ ] **Step 7: Final commit (only if Steps 2/3 required a formatting/type touch-up; otherwise skip).**
  `git add -A && git commit -m "chore(sessions): P6 verification sweep — ruff/mypy/lints/frontend green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

## Phase P7 — Cache (C2) + TutorialGuidedShell + Migration

This phase closes the staged recut: it makes the tutorial run-cache key fold the
four operator-controlled deterministic inputs (so a recipe/skill edit can never
serve a stale cached pipeline), builds the `TutorialGuidedShell` bridge that mounts
the real guided `ChatPanel` surface and removes the big-bang `describe`/`showBuilt`
turns, and adds the migration verification + secret-archive hardening on top of the
existing session-DB-reset runbook (the `SESSION_SCHEMA_EPOCH` 22→23 bump itself
landed in P0.T3).

**Depends on symbols owned by other phases (must already be merged into this branch
before P7 frontend tasks run):**
- `GuidedStep.STEP_4_WIRE` + `skills/step_4_wire.md` (P1.T1/P1.T2) — the cache key's
  staged-skill hash enumerates `_STEP_PLAYBOOK_ORDER`, which P1.T2 extends with the
  wire step's `.md`.
- `GuidedSession.profile` on the wire (`WorkflowProfileResponse`, `schemas.py`) +
  the TS `WorkflowProfile` type and `GuidedSession.profile: WorkflowProfile | null`
  on `guided.ts` (P7.1 / P0).
- `POST /api/sessions/{id}/guided/start` + `startGuidedSession(sessionId, profileKind)`
  in `client.ts` (P7.1).
- `WorkflowProfileKind` discriminator string `"tutorial"` (P0.T1 / P7.1).
- The guided `terminal.kind === "completed"` branch in `ChatPanel` (existing).

The P7.4 cache task and the migration task touch only backend/runbook surfaces and
do **not** depend on the frontend wiring, so they can land first.

---

### Task P7.1: Fold the staged guided-skill hash into `tutorial_model_id` (cache input #2)

**Files:**
- Modify: `src/elspeth/web/composer/guided/prompts.py` (add public `guided_staged_skill_hash()` after `load_step_chat_skill`, currently ends at `:109`)
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
  Append after `load_step_chat_skill` (after line 109):
  ```python
  @lru_cache(maxsize=1)
  def guided_staged_skill_hash() -> str:
      """Hex SHA-256 over base.md + every step playbook in _STEP_PLAYBOOK_ORDER.

      Consumed by the tutorial run-cache key (tutorial_model_id, cache input
      #2). Enumerating the playbook order means appending a GuidedStep member
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

### Task P7.2: Add `recipe_catalog_content_hash()` over recipes.py + recipe_match.py (cache input #4)

**Files:**
- Modify: `src/elspeth/web/composer/recipes.py` (add public `recipe_catalog_content_hash()` at module end)
- Modify: `tests/unit/web/composer/test_recipes.py` (Create if absent)

**Interfaces:**
- Produces: `def recipe_catalog_content_hash() -> str` — hex SHA-256 over the byte content of both `composer/recipes.py` and `composer/guided/recipe_match.py` (the deterministic recipe authoring + predicate registry). Hashes source files, not imported objects, so any edit to either module shifts it.
- Consumes: `pathlib.Path(__file__)` for `recipes.py`; resolves `recipe_match.py` via `Path(__file__).parent / "guided" / "recipe_match.py"`.

- [ ] **Step 1: Write the failing test.**
  Create `tests/unit/web/composer/test_recipes.py` (or append):
  ```python
  """Tests for the composer recipe catalog content hash (cache input #4)."""

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

      Cache input #4 of the tutorial run-cache key (C2). Under D11 the
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

### Task P7.3: Fold the two new hashes into `tutorial_model_id` (four-input cache key)

**Files:**
- Modify: `src/elspeth/web/composer/tutorial_service.py:784-821` (`tutorial_model_id`)
- Modify: `tests/unit/web/composer/test_tutorial_service.py:401-447` (extend the 3-input regression test + add the 4th-input guard)

**Interfaces:**
- Consumes: `guided_staged_skill_hash` (P7.1), `recipe_catalog_content_hash` (P7.2), `load_skill_with_hash`, `load_deployment_skill`, `WebSettings.composer_model`/`.data_dir`.
- Produces: `tutorial_model_id(settings) -> str` now of form
  `composer=<model>|skill=<core>|staged_skill=<staged>|deployment_skill=<deploy>|recipe=<recipe>`.

- [ ] **Step 1: Write the failing regression-guard test.**
  Append to `tests/unit/web/composer/test_tutorial_service.py` (after `test_tutorial_model_id_changes_when_deployment_skill_overlay_is_added`, line 447). Import the two new hash functions at the top of the file (with the existing `from elspeth.web.composer.skills import load_skill_with_hash` block):
  ```python
  def test_tutorial_model_id_includes_staged_skill_and_recipe_hashes(tmp_path: Path) -> None:
      """C2 four-input cache key: staged guided skills + recipe catalog are keyed.

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

      Pins that the 4th input is load-bearing: if it were dropped, this would
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

- [ ] **Step 3: Implement the four-input key in `tutorial_model_id`.**
  Add the two imports near the existing `load_deployment_skill` import (tutorial_service.py:37):
  ```python
  from elspeth.web.composer.guided.prompts import guided_staged_skill_hash
  from elspeth.web.composer.recipes import recipe_catalog_content_hash
  from elspeth.web.composer.skills import load_deployment_skill, load_skill_with_hash
  ```
  Replace the body of `tutorial_model_id` (the final three lines, 818-821) with:
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
  Update the docstring of `tutorial_model_id` (lines 785-817): change "Three such inputs" to "Four such inputs", and add the two new bullets to the "Covered (automatic invalidation)" list:
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
  git commit -m "feat(tutorial): four-input cache key (staged skills + recipe catalog)

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
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts:19-209`
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
- Produces: `export function TutorialGuidedShell(props: { sessionId: string; onCompleted: (sessionId: string) => void }): JSX.Element`. On mount it calls `startGuidedSession(sessionId, "tutorial")` then `startGuided(sessionId)` (store), renders the real `ChatPanel` guided surface, and on `guidedSession.terminal.kind === "completed"` calls `onCompleted(sessionId)`.
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
        guidedSession: null,
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
  });
  ```

- [ ] **Step 2: Run the test to confirm it fails (module missing).**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run TutorialGuidedShell
  ```
  Expected: failure — `Failed to resolve import "./TutorialGuidedShell"`.

- [ ] **Step 3: Implement `TutorialGuidedShell.tsx`.**
  ```tsx
  import { useEffect, useRef } from "react";
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
        // Bind the store's activeSessionId to this tutorial session BEFORE
        // startGuided. startGuided (sessionStore.ts) DISCARDS its fetched guided
        // payload unless get().activeSessionId === the requested id, and ChatPanel
        // renders the empty-session surface (chat-panel--empty) whenever
        // activeSessionId is null — so without this bind the embedded guided
        // ChatPanel never mounts. Mirrors the binding the now-deleted
        // TutorialTurn2Describe performed after createSession.
        useSessionStore.setState({ activeSessionId: sessionId });
        await startGuidedSession(sessionId, "tutorial");
        await startGuided(sessionId);
      })();
    }, [sessionId, startGuided]);

    // Hand off to the run/audit/graduation tail when guided reaches completion.
    useEffect(() => {
      if (completedRef.current) {
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
        <ChatPanel />
      </section>
    );
  }
  ```

- [ ] **Step 4: Run the test to confirm it passes.**
  ```bash
  cd src/elspeth/web/frontend && npm test -- --run TutorialGuidedShell
  ```
  Expected: all three tests pass.

- [ ] **Step 5: Commit.**
  ```bash
  git add src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.test.tsx
  git commit -m "feat(tutorial): TutorialGuidedShell bridge embeds the real guided ChatPanel

Starts a TUTORIAL-profile guided session (idempotent /guided/start), mounts
the real ChatPanel guided surface, and hands the session to the run/audit/
graduation tail on terminal=completed. Per-stage interpretation review + wire
confirm are owned by the ChatPanel guided branch.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.6: Rewire `HelloWorldTutorial`; remove `TutorialTurn2Describe` + `TutorialTurn2bShowBuilt`

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx:1-167`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx` + `.test.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx` + `.test.tsx`
- Delete: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn3Graph.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/index.ts` (drop removed exports)

**Interfaces:**
- Consumes: `TutorialGuidedShell` (P7.5), `createSession` (existing), the surviving `TutorialTurn1Welcome` / `TutorialTurn4Run` / `TutorialTurn5AuditStory` / `TutorialTurn7Graduation`, `tutorialReducer` (P7.4).

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
  import { createSession, deleteTutorialOrphans } from "@/api/client";
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
  ```
  In the component, replace the `welcome`/`describe`/`showBuilt`/`graph`/`mode`
  branches. On Start, create a session, store its id, then dispatch `start`:
  ```tsx
    const [state, dispatch] = useReducer(tutorialReducer, initialTutorialState);
    const [sessionId, setSessionId] = useState<string | null>(null);

    useEffect(() => {
      void deleteTutorialOrphans().catch((err) => {
        console.error("[tutorial] orphan cleanup failed:", err);
      });
    }, []);

    const onStart = async (): Promise<void> => {
      const session = await createSession();
      setSessionId(session.id);
      dispatch({ type: "start" });
    };
  ```
  Render branches:
  ```tsx
        {state.step === "welcome" && (
          <TutorialTurn1Welcome
            onStart={() => void onStart()}
            onSkip={() => dispatch({ type: "skipToGraduation" })}
          />
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
          <TutorialTurn7Graduation onBack={goBack} />
        )}
  ```
  Replace `TUTORIAL_STEP_LABELS` and `stepIndex` (lines 135-167) with the 5-step set:
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
  Expected: tutorial tests pass; `npm run typecheck` reports 0 errors (no dangling references to deleted modules/steps remain). If typecheck flags a residual reference (e.g. a stale import in `copy.ts` for `TURN_2B_*`), remove the now-unused copy constants in the same step and re-run.

- [ ] **Step 6: Commit.**
  ```bash
  git add -A src/elspeth/web/frontend/src/components/tutorial/
  git commit -m "feat(tutorial): rewire HelloWorldTutorial onto TutorialGuidedShell

Welcome -> embedded guided ChatPanel -> run/audit/graduation. Removes the
big-bang TutorialTurn2Describe, TutorialTurn2bShowBuilt, TutorialTurn3Graph,
and TutorialTurn6ModeChoice; the wire stage and per-stage interpretation
review now live inside the real guided surface.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.7: Rewrite the tutorial E2E specs for the staged flow

**Files:**
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts`
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`
- Modify: `src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts` (extend for the wire stage)

**Interfaces:**
- Consumes: the mocked guided protocol (the existing `composer-guided.spec.ts` mock pattern), `POST /guided/start`, the `STEP_4_WIRE` turn payload (`WireStageData`, P3), the tutorial run endpoint mock.

- [ ] **Step 1: Read the existing mocked-guided E2E pattern so the rewrite reuses it.**
  ```bash
  cd src/elspeth/web/frontend && sed -n '1,80p' tests/e2e/composer-guided.spec.ts
  ```
  Expected: confirms how the spec mocks `GET /guided` / `POST /guided/respond` and asserts on the `chat-panel--guided` surface. Note the route-mock helpers it imports (page-objects / helpers).

- [ ] **Step 2: Rewrite `tutorial.spec.ts` to the staged flow (mocked).**
  Replace the describe/showBuilt assertions. The new happy-path mocks:
  `POST /api/sessions` (returns `{id}`), `POST /guided/start` (200, idempotent),
  `GET /guided` returning a `step_1_source` turn, then drive
  `POST /guided/respond` through source -> sink -> recipe-apply -> `step_4_wire`,
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
  Point the harness at the `POST /guided/start` (tutorial profile) entry then the
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
  cd src/elspeth/web/frontend && npm run test:e2e -- tutorial.spec.ts composer-guided.spec.ts
  ```
  Expected: both pass (the `.staging.spec.ts` is excluded from the default e2e run; it runs under `test:e2e:staging` against a live deploy).

- [ ] **Step 6: Run the `SlotType` / `guided.ts` mirror gate (this branch edits guided.ts upstream; re-run here to confirm no drift).**
  ```bash
  uv run python scripts/cicd/check_slot_type_cross_language.py
  ```
  Expected: exit 0 (no SlotType / guided.ts mirror drift).

- [ ] **Step 7: Commit.**
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
> and added `tests/unit/web/composer/tools/test_advisor_tool_prose.py`). This
> task must NOT re-edit the prose — by the time P7 runs, the phrase is gone and a
> blind string-replace would fail or commit nothing. P7.8 instead CONFIRMS the
> P5.9 change survived the intervening phases and that no *other* test (e.g. the
> broader `test_dispatch.py`) still pins the deleted phrase. If P5.9 did not
> land, Step 1 fails loud and you return to P5.9 — do not author the fix here.

**Files:**
- Modify (only if Step 1 finds a stale pin): a matching assertion in
  `tests/unit/web/composer/tools/test_dispatch.py`.

**Interfaces:** none (verification of an already-landed prose change).

- [ ] **Step 1: Confirm the stale phrase is fully gone from source.**
  ```bash
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -rn "Disabled by default" src/elspeth/web
  ```
  Expected: **no output** (exit 1). If `_dispatch.py:129` still matches, P5.9 did
  NOT land — stop and complete P5.9 first; do not re-author the fix here.

- [ ] **Step 2: Confirm the corrected prose + the P5.9 guard test are present.**
  ```bash
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -n "operator-configured" src/elspeth/web/composer/tools/_dispatch.py && uv run python -m pytest tests/unit/web/composer/tools/test_advisor_tool_prose.py -q
  ```
  Expected: the grep prints the `operator-configured` line and the prose guard
  test is `1 passed`.

- [ ] **Step 3: Check the broader dispatch suite for any OTHER test still pinning the old phrase.**
  ```bash
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && grep -rn "Disabled by default" tests/ ; uv run python -m pytest tests/unit/web/composer/tools/test_dispatch.py -q
  ```
  Expected: the grep prints **nothing** and `test_dispatch.py` is all `passed`.
  If the grep hits a stale assertion in `test_dispatch.py`, update its expected
  substring to `"operator-configured"`, re-run, then proceed to Step 4. If the
  grep is empty, there is nothing to edit and no commit is needed — this task is
  satisfied (skip Step 4).

- [ ] **Step 4: Commit ONLY if Step 3 edited a test.**
  ```bash
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add tests/unit/web/composer/tools/test_dispatch.py && git commit -m "test(composer): repoint stale 'Disabled by default' dispatch assertion to corrected prose (P7.8)

P5.9 corrected the advisor tool prose; this repoints the lingering
test_dispatch.py substring assertion to the operator-configured wording.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P7.9: Migration — runbook deltas (epoch verification + hardened secret-archive)

**Files:**
- Modify: `docs/runbooks/staging-session-db-recreation.md`

**Interfaces:** none (operator-facing procedure). The `SESSION_SCHEMA_EPOCH` 22→23 bump itself landed in P0.T3; this task adds the runbook-side verification + hardening.

- [ ] **Step 1: Add a "Current Cutover: 0.7.0 (single-DB reset)" section** above the existing "Current Cutover: 0.6.0" heading (line 5). It must state:
  - 0.7.0 bumps **only** `SESSION_SCHEMA_EPOCH` 22→23 (NOT `SQLITE_SCHEMA_EPOCH`), so this is a **single-DB** session-only cutover — follow the "Staging Reset" / "Local Or Dev Reset" single-DB path and **do NOT** run the Phase 5b two-DB procedure.
  - Boot fail-closes on a stale session DB with the `SessionSchemaError` "version 22 does not match SESSION_SCHEMA_EPOCH=23 ... Delete the session DB file and restart" (the P0.T3 `_assert_schema_sentinels` guard), converting the lazy per-row 500 into a loud boot guard.
  - `auth.db` and `runs/audit.db` are separate files and are NOT reset.

- [ ] **Step 2: Add a "0.7.0 epoch + smoke verification" block** to the staging procedure's post-restart steps (after the health checks, around line 376). It must instruct the operator to:
  ```bash
  # Confirm the recreated session DB carries the new epoch sentinel.
  sqlite3 "$DB_PATH" 'PRAGMA user_version;'   # expect 23 (== SESSION_SCHEMA_EPOCH)
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

- [ ] **Step 4: Add a one-line cross-reference** in the spec's §8 migration section to this runbook (if not already present), so the runbook is the single source of the procedure.
  ```bash
  grep -n "staging-session-db-recreation" docs/superpowers/specs/2026-06-22-tutorial-staged-recut-design.md
  ```
  Expected: the spec already references the runbook (line ~672/862). No edit needed if present; if absent, add a reference line. (This is a verification step, not a guaranteed write.)

- [ ] **Step 5: Lint the runbook markdown (no broken anchors / fenced blocks).**
  ```bash
  grep -c '```' docs/runbooks/staging-session-db-recreation.md
  ```
  Expected: an even number (every fence closed).

- [ ] **Step 6: Commit.**
  ```bash
  git add docs/runbooks/staging-session-db-recreation.md
  git commit -m "docs(runbook): 0.7.0 single-DB cutover + epoch verify + hardened secret archive

Adds the 0.7.0 single-DB (SESSION_SCHEMA_EPOCH 22->23) cutover section, a
PRAGMA user_version==23 + fresh-session-reaches-COMPLETED smoke verification,
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
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv lock`
  (refreshes the `[[package]] name = "elspeth"` pin).
- [ ] **Step 3: Verify.**
  `cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && uv run python -c "import importlib.metadata as m; print(m.version('elspeth'))"`
  Expected: `0.7.0`.
- [ ] **Step 4: Commit.**
  ```bash
  cd /home/john/elspeth/.claude/worktrees/tutorial-staged-recut && git add pyproject.toml uv.lock && git commit -m "chore: bump version 0.6.0 -> 0.7.0

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
  Expected: all pass (0 failures). This pins the canonical-seed cross-language equality, the four-input cache key, and the tutorial run route.

- [ ] **Step 3: elspeth-lints trust gates this phase touches.**
  ```bash
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check \
    --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' \
    --root src/elspeth
  ```
  Expected: no new findings attributable to P7 files (the cache helpers + shell + prose are not new Tier-3 boundaries; if `trust_tier.tier_model` reports displacement it is from a sibling phase, not P7 — record, do not bless blind, per the gate-debt note).

- [ ] **Step 4: Frontend gates (from the frontend dir).**
  ```bash
  cd src/elspeth/web/frontend && npm run typecheck && npm test -- --run && npm run build
  ```
  Expected: typecheck 0 errors; vitest all pass; build succeeds.

- [ ] **Step 5: wardline taint gate.**
  ```bash
  wardline scan . --fail-on ERROR
  ```
  Expected: exit 0 (clean). The P7 surfaces (cache hashing of local files, a frontend shell, runbook prose) introduce no new external-input -> sink flow; if a finding appears, fix it at the boundary per the `wardline-gate` skill before proceeding.

- [ ] **Step 6: Final phase commit (only if any fix-forward edits were made in steps 1-5).**
  ```bash
  git add -A
  git commit -m "chore(p7): gate-sweep fix-forward for cache/frontend/migration phase

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```
