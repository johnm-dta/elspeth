> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P0 — Schema & profile foundation

This phase lays the persistence spine the rest of the recut builds on: a new
`WorkflowProfile` value type, three new persisted `GuidedSession` fields, and the
two schema-sentinel bumps (`GUIDED_SESSION_SCHEMA_VERSION` 5→6 and
`SESSION_SCHEMA_EPOCH` 23→24) that make a stale session DB fail loudly at boot
instead of silently 500-ing per row. No UX or behaviour change ships here — the
new fields default to live-guided behaviour (`EMPTY_PROFILE`,
`advisor_checkpoint_passes_used=0`). Spec refs: §4.3, §8 (D15, D16).

All commands run from the repository root `/home/john/elspeth` — the
`release/0.7.0` checkout, NOT the stale `tutorial-staged-recut` worktree (see the
overview's corrected codebase note). The repo's
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
  # server-side at POST /api/sessions/{session_id}/guided/start; NOT rendered
  # on GET /api/sessions/{session_id}/guided.
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
  Expected: `7 passed`.

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
  Expected: `15 passed` (7 tests from P0.1 + 8 strict-serialisation tests from P0.2).

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
  - `from_dict` reads `d["profile"]` (via `WorkflowProfile.from_dict`), `d["advisor_checkpoint_passes_used"]`, and `d["advisor_signoff_escape_offered"]` by direct-key access. It must not coerce persisted advisor fields: `advisor_checkpoint_passes_used` is accepted only when `type(raw) is int and raw >= 0` (so `True`/`False` are rejected even though `bool` subclasses `int`), and `advisor_signoff_escape_offered` is accepted only when `type(raw) is bool`.

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
          assert d["advisor_signoff_escape_offered"] is False

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

      def test_from_dict_rejects_missing_escape_flag_key(self) -> None:
          d = GuidedSession.initial().to_dict()
          del d["advisor_signoff_escape_offered"]
          with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_malformed_profile(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["profile"] = {"coaching": True}  # missing the other four keys
          with pytest.raises(InvariantError, match=r"GuidedSession\.from_dict"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_string_pass_counter(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["advisor_checkpoint_passes_used"] = "1"
          with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_bool_as_int_pass_counter(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["advisor_checkpoint_passes_used"] = True
          with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_negative_pass_counter(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["advisor_checkpoint_passes_used"] = -1
          with pytest.raises(InvariantError, match=r"advisor_checkpoint_passes_used"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_string_escape_flag(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["advisor_signoff_escape_offered"] = "false"
          with pytest.raises(InvariantError, match=r"advisor_signoff_escape_offered"):
              GuidedSession.from_dict(d)

      def test_from_dict_rejects_number_escape_flag(self) -> None:
          d = GuidedSession.initial().to_dict()
          d["advisor_signoff_escape_offered"] = 1
          with pytest.raises(InvariantError, match=r"advisor_signoff_escape_offered"):
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

- [ ] **Step 4: Add the three frozen fields to `GuidedSession`.**
  In `src/elspeth/web/composer/guided/state_machine.py`, the `GuidedSession`
  field list ends with `chat_turn_seq: int = 0` (grep for it — line numbers are
  advisory). Append the three new fields immediately after `chat_turn_seq: int = 0`:
  ```python
      # P0 (tutorial staged recut, §4.3): the per-session engine-behaviour
      # profile, the persisted advisor-checkpoint pass counter, and the
      # genuine-OUTAGE sign-off escape-offered flag (D5/B2 — written by
      # run_wire_signoff in P5.5; persists whether the last END sign-off
      # terminal offered a genuine-outage escape, so a later acknowledgement is
      # honoured without re-calling the provider and a FLAGGED/MALFORMED
      # terminal can never be acknowledged into a bypass). All three are part of
      # the v6 schema bump. `profile` is a frozen dataclass of scalars (already
      # deeply immutable — no freeze_fields guard needed); the other two are
      # plain scalars. The defaults == live-guided behaviour so existing
      # GuidedSession() / .initial() call sites are unchanged behaviourally.
      profile: WorkflowProfile = EMPTY_PROFILE
      advisor_checkpoint_passes_used: int = 0
      advisor_signoff_escape_offered: bool = False
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

- [ ] **Step 6: Add the three `to_dict` keys.**
  In `src/elspeth/web/composer/guided/state_machine.py`, the `to_dict` return
  dict ends with `"chat_turn_seq": self.chat_turn_seq,` (grep for it). Add the
  three new keys immediately after that line (before the closing `}`):
  ```python
              "profile": self.profile.to_dict(),
              "advisor_checkpoint_passes_used": self.advisor_checkpoint_passes_used,
              "advisor_signoff_escape_offered": self.advisor_signoff_escape_offered,
  ```

- [ ] **Step 7: Read the three new `from_dict` keys strictly and pass them to `cls(...)`.**
  In `src/elspeth/web/composer/guided/state_machine.py` `from_dict`, after the
  `chat_turn_seq_raw = d["chat_turn_seq"]` line (grep for it), add:
  ```python
              profile_raw = d["profile"]
              advisor_passes_raw = d["advisor_checkpoint_passes_used"]
              advisor_signoff_escape_raw = d["advisor_signoff_escape_offered"]
  ```
  Immediately after those direct-key reads, validate the persisted advisor fields
  without coercion and wrap malformed profile records so every bad `GuidedSession`
  load reports the `GuidedSession.from_dict` boundary:
  ```python
              try:
                  profile = WorkflowProfile.from_dict(profile_raw)
              except InvariantError as exc:
                  raise InvariantError("GuidedSession.from_dict: malformed profile") from exc

              if type(advisor_passes_raw) is not int:
                  raise InvariantError(
                      "GuidedSession.from_dict: advisor_checkpoint_passes_used "
                      f"must be a non-negative int, got {type(advisor_passes_raw).__name__}"
                  )
              if advisor_passes_raw < 0:
                  raise InvariantError(
                      "GuidedSession.from_dict: advisor_checkpoint_passes_used "
                      f"must be >= 0, got {advisor_passes_raw!r}"
                  )
              if type(advisor_signoff_escape_raw) is not bool:
                  raise InvariantError(
                      "GuidedSession.from_dict: advisor_signoff_escape_offered "
                      f"must be bool, got {type(advisor_signoff_escape_raw).__name__}"
                  )
  ```
  Then in the `return cls(...)` call (the block ending with
  `chat_turn_seq=int(chat_turn_seq_raw),`), add the three new kwargs immediately
  after that line (before the closing `)`):
  ```python
                  profile=profile,
                  advisor_checkpoint_passes_used=advisor_passes_raw,
                  advisor_signoff_escape_offered=advisor_signoff_escape_raw,
  ```

- [ ] **Step 8: Run the new tests to confirm pass.**
  ```bash
  uv run python -m pytest tests/unit/web/composer/guided/test_state_machine.py::TestGuidedSessionProfileFields -q
  ```
  Expected: `14 passed`.

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
P6.4/P6.5 (the session-scoped
`POST /api/sessions/{session_id}/guided/start` endpoint and frontend caller) are
the first non-test callers to pass a non-default profile. This task only widens
the seam + proves threading.

- [ ] **Step 1: Write the failing threading test.**
  ⚠️ The `tests/unit/web/sessions/routes/` directory does NOT exist yet — create it
  first (`mkdir -p tests/unit/web/sessions/routes && touch tests/unit/web/sessions/routes/__init__.py`
  if the suite uses package-style discovery), or pytest will fail to collect the new file.
  Then create `tests/unit/web/sessions/routes/test_initial_composition_state_profile.py`:
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
          guided_session=GuidedSession.initial(profile=profile),
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

### Task P0.6: Bump `SESSION_SCHEMA_EPOCH` 23→24 with a boot fail-close guard + update the two epoch-pinning tests

**Files:**
- Modify: `src/elspeth/web/sessions/models.py:113-117` (epoch constant + comment)
- Modify: `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py:48-51`
- Modify: `tests/unit/web/sessions/test_interpretation_events_table.py:199-200`
- Modify: `tests/unit/web/sessions/test_schema.py` (new boot fail-close test for prior epoch 23)

**Interfaces:**
- Produces: `SESSION_SCHEMA_EPOCH == 24` (`models.py:120`).

> **Corrected — the epoch is already at 23.** A prior bug-fix already advanced `SESSION_SCHEMA_EPOCH` to **23** (the committed value in main `src/`). The next free epoch is **24**, so this task now bumps **23→24**: the prior epoch is **23**, and the boot-guard regression test below stamps `user_version = 23` so its red→green TDD sequence is genuine. The in-flight epoch-23 work (constant + the two pinning-test assertions) is now COMMITTED at `5e46c226c` (operator-confirmed 2026-06-23), so 23 is settled.
- Consumes: existing `initialize_session_schema` → `_assert_schema_sentinels` (`schema.py:106/127-169`) — unchanged code, the bump alone makes a 23-stamped DB fail-close while preserving epoch 23 as the settled prior epoch.

- [ ] **Step 1: Write the failing boot fail-close test.**
  Append to `tests/unit/web/sessions/test_schema.py` (it already imports
  `SessionSchemaError`, `create_session_engine`, `initialize_session_schema`,
  `text`, and `pytest` — confirm at the top of the file; the existing
  `test_initialize_session_schema_rejects_partial_stale_schema` uses all of them).
  No new imports are needed. Append:
  ```python
  def test_initialize_session_schema_rejects_epoch_23_database() -> None:
      """A VALID full-schema DB stamped at the PRIOR epoch fail-closes at boot.

      Genuine TDD red->green guard for the 23->24 bump (D15), isolated from any
      table-set mismatch. Seed a COMPLETE current-schema DB
      (``initialize_session_schema`` runs ``metadata.create_all`` + stamps the
      CURRENT epoch), then re-stamp ``user_version`` to the prior epoch. Because the
      schema is otherwise valid, the ONLY thing that can fail-close is the epoch
      sentinel (``_assert_schema_sentinels``):
        - pre-bump  (epoch still 23): stamped 23 == current 23 -> accepted, NO raise -> RED
        - post-bump (epoch 24):       stamped 23 != current 24 -> raises -> GREEN

      Do NOT use a partial-table fixture (e.g. a lone ``CREATE TABLE sessions``):
      ``_validate_current_schema`` runs right after the sentinel check
      (``schema.py:107``) and would raise a *table-set mismatch* ``SessionSchemaError``
      at BOTH epochs — and ``_schema_error`` always embeds "SESSION_SCHEMA_EPOCH" in
      its message (``schema.py:570``), so ``match=`` still hits. That masks the epoch
      guard and gives a hollow red phase that "passes" before the bump for the wrong
      reason.
      """
      eng = create_session_engine("sqlite:///:memory:")
      initialize_session_schema(eng)  # full schema + stamps the CURRENT epoch
      with eng.begin() as conn:
          conn.execute(text("PRAGMA user_version = 23"))  # re-stamp to the prior epoch
      with pytest.raises(SessionSchemaError, match="SESSION_SCHEMA_EPOCH"):
          initialize_session_schema(eng)
  ```
  Seeding the full schema (so the epoch sentinel is the SOLE fail-close cause) is
  what makes the pre-bump run a genuine RED (DID NOT RAISE) instead of a false
  green via table mismatch. (`create_session_engine("sqlite:///:memory:")` keeps a
  single shared connection, so the schema + re-stamped `user_version` persist across
  the two `initialize_session_schema` calls.)

- [ ] **Step 2: Run to confirm failure.**
  ```bash
  uv run python -m pytest "tests/unit/web/sessions/test_schema.py::test_initialize_session_schema_rejects_epoch_23_database" -q
  ```
  Expected: FAIL — `DID NOT RAISE <class 'SessionSchemaError'>` (with epoch still
  23, a 23-stamped DB matches the current epoch and is accepted).

- [ ] **Step 3: Update the two epoch-pinning tests to 24.**
  ⚠️ As committed at `5e46c226c`, both tests assert `== 23` but their function names
  still read `…_epoch_is_22` (a cosmetic lag from the epoch-23 fix). Bump the
  assertions 23→24 and rename the functions `…_epoch_is_22` → `…_epoch_is_24`.
  In `tests/unit/web/sessions/test_blob_inline_resolutions_schema.py` (grep
  `def test_blob_inline_resolutions_schema_epoch_is_`): set the assertion and the
  `PRAGMA user_version` check to `24` and rename to
  `test_blob_inline_resolutions_schema_epoch_is_24`:
  ```python
  def test_blob_inline_resolutions_schema_epoch_is_24(engine) -> None:
      assert SESSION_SCHEMA_EPOCH == 24
      with engine.connect() as conn:
          assert conn.execute(text("PRAGMA user_version")).scalar_one() == 24
  ```
  In `tests/unit/web/sessions/test_interpretation_events_table.py` (grep
  `def test_proposal_provenance_schema_cohort_epoch_is_`): set to `24` and rename to
  `test_proposal_provenance_schema_cohort_epoch_is_24`:
  ```python
  def test_proposal_provenance_schema_cohort_epoch_is_24() -> None:
      assert SESSION_SCHEMA_EPOCH == 24
  ```

- [ ] **Step 4: Bump the epoch constant + add the epoch-24 comment.**
  In `src/elspeth/web/sessions/models.py`, the epoch-history comment ends with the
  `#   23 → ...` block and the constant `SESSION_SCHEMA_EPOCH = 23` (grep — the
  constant is ~`:120`). Append a new history line after the `23` block and bump the
  constant to 24:
  ```python
  #   24 → no SQL-shape change; bumped in lockstep with GUIDED_SESSION_SCHEMA_VERSION
  #        5→6 (composer_meta JSON adds GuidedSession.profile +
  #        advisor_checkpoint_passes_used + advisor_signoff_escape_offered) so a
  #        stale sessions.db fail-closes at boot via _assert_schema_sentinels instead
  #        of lazy-500-ing per guided row on GuidedSession.from_dict. Pre-release
  #        delete-and-recreate policy; see docs/runbooks/staging-session-db-recreation.md.
  SESSION_SCHEMA_EPOCH = 24
  ```

- [ ] **Step 5: Run the boot fail-close test + the two pinning tests to confirm pass.**
  ```bash
  uv run python -m pytest \
    "tests/unit/web/sessions/test_schema.py::test_initialize_session_schema_rejects_epoch_23_database" \
    "tests/unit/web/sessions/test_blob_inline_resolutions_schema.py::test_blob_inline_resolutions_schema_epoch_is_24" \
    "tests/unit/web/sessions/test_interpretation_events_table.py::test_proposal_provenance_schema_cohort_epoch_is_24" -q
  ```
  Expected: `3 passed`.

- [ ] **Step 6: Run the whole sessions schema test surface to confirm no stale epoch literal left red.**
  ```bash
  uv run python -m pytest tests/unit/web/sessions/ -q -k "schema or epoch or interpretation_events or blob_inline"
  ```
  Expected: all pass. If any other test references a stale session-epoch literal
  (`22` or `23`), it surfaces here — fix it to `24` in the same task (do NOT defer
  to an observation; it is in-scope for the epoch bump).

- [ ] **Step 7: Commit.**
  ```bash
  git add src/elspeth/web/sessions/models.py tests/unit/web/sessions/test_blob_inline_resolutions_schema.py tests/unit/web/sessions/test_interpretation_events_table.py tests/unit/web/sessions/test_schema.py
  git commit -m "feat(sessions): bump SESSION_SCHEMA_EPOCH 23->24 (boot fail-close on stale DB)

Lockstep with GUIDED_SESSION_SCHEMA_VERSION 5->6: no SQL-shape change, but the
epoch bump turns _assert_schema_sentinels into a loud boot guard so a sessions.db
carried across the cutover crashes with the delete-and-restart message instead of
lazy-500-ing per guided row (D15). Epoch-pin tests + a 23-rejection regression
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

- [ ] **Step 4: Confirm no stale epoch/version literal remains anywhere in the suite, except the intentional prior-epoch regression fixture.**
  ```bash
  rg -n 'SESSION_SCHEMA_EPOCH == 2[23]|GUIDED_SESSION_SCHEMA_VERSION == 5|"schema_version"] == 5' tests/ src/
  rg -n 'user_version.*2[23]' tests/ src/ | rg -v 'tests/unit/web/sessions/test_schema.py:.*PRAGMA user_version = 23'
  ```
  Expected: no output from the first command; no output from the second command
  after the allowlist removes the intentional
  `test_initialize_session_schema_rejects_epoch_23_database` fixture. Any other
  hit is an in-scope follow-up for this phase — fix it before closing P0.

- [ ] **Step 5: Final phase commit (only if Step 4 surfaced and fixed a stray pin; otherwise skip).**
  ```bash
  git add \
    src/elspeth/web/composer/guided/profile.py \
    src/elspeth/web/composer/guided/state_machine.py \
    src/elspeth/web/sessions/routes/_helpers.py \
    src/elspeth/web/sessions/models.py \
    tests/unit/web/composer/guided/test_profile.py \
    tests/unit/web/composer/guided/test_state_machine.py \
    tests/unit/web/sessions/routes/test_initial_composition_state_profile.py \
    tests/unit/web/sessions/test_blob_inline_resolutions_schema.py \
    tests/unit/web/sessions/test_interpretation_events_table.py \
    tests/unit/web/sessions/test_schema.py
  git commit -m "test(sessions): sweep stale schema-version/epoch pins for P0 bumps

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---
