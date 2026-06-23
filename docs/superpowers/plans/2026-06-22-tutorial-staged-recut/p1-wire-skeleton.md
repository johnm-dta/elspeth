> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P1 — Wire stage skeleton (STEP_4_WIRE) + terminal-stamp move

This phase appends `GuidedStep.STEP_4_WIRE` (the 5th, append-only member) and
`TurnType.CONFIRM_WIRING`, wires every coordinated touchpoint named in spec §4.2,
moves terminal-stamping out of both completion seams into a new `STEP_4_WIRE`
handler that gates on `validate().is_valid` only (advisor sign-off comes in P5.6),
and pins the terminal-stamp invariant. The wire turn payload is a **skeleton** here:
empty topology/contract arrays plus the live step index, not the final validation
overlay. The full two-read topology + edge_contracts + warnings blob is **P2.4**'s
job. P1 still owns the minimal route wiring so the phase is self-contained:
after accept, the dispatcher emits `confirm_wiring`; a `confirm_wiring` response
calls `handle_step_4_wire_confirm` and stamps `COMPLETED` on a valid state.

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
      # P1 reserves the skeleton wire payload keys as empty containers. P2.2/P2.4
      # upgrade this to the final WireStageData contract, adding warnings and
      # populated topology + edge_contracts/semantic_contracts.
      TurnType.CONFIRM_WIRING: frozenset({"topology", "edge_contracts", "semantic_contracts"}),
  }
  ```

- [ ] **Step 7: Add `CONFIRM_WIRING` to `_NESTED_SHAPES` (total over TurnType).**
  Edit `protocol.py:243-253`, adding the empty-tuple entry after `RECIPE_OFFER`:

  ```python
      TurnType.RECIPE_OFFER: (("knobs", "mapping", frozenset({"fields"})),),
      # No nested-shape validation at the skeleton stage: topology / edge_contracts /
      # semantic_contracts are populated by the two-read merge (P2.4) and validated there.
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
- Modify `src/elspeth/web/sessions/routes/_helpers.py:3692-3697` (`_ORDER` in `_guided_step_index`)
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
  Edit `_helpers.py:3692-3697`:

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
      edge_contracts overlay, spec §B2) is assembled in P2.4 and replaces this
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
- Produces: `_advance_step_4(session, response, turn_type) -> _StepAdvanceResult` — a **self-loop**: on `CONFIRM_WIRING` it returns the session unchanged (terminal-stamping is the dispatcher/handler's job, P1.6/P5.6); any other turn type raises `InvariantError`. The self-loop is what makes the stage re-enterable across advisor sign-off rounds (D13).
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
          # dispatcher/handler does (P1.6/P5.6). The session pointer stays at STEP_4_WIRE.
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
      plus the profile-gated advisor sign-off in P5). ``step_advance`` is pure and
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
- Modify `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` (`GUIDED_CHAT_PLACEHOLDERS: Record<GuidedStep,…>` + `GUIDED_STEP_PURPOSES: Record<GuidedStep,…>` + `GUIDED_WORKFLOW_STEPS`)
- Modify `src/elspeth/web/frontend/src/types/guided.test.ts` (TurnType/GuidedStep cardinality tests)
- Modify `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx` (workflow stepper renders the Wire step)

**Interfaces:**
- Produces: TS `TurnType` union gains `"confirm_wiring"`; TS `GuidedStep` union gains `"step_4_wire"`.
- Consumes: nothing new — these are hand-written mirrors of the Python StrEnums (source-of-truth comment at `guided.ts:5-7`).
- Note: widening the two unions WITHOUT updating their exhaustive consumers in the SAME slice BREAKS `npm run typecheck` — every `Record<TurnType,…>` / `Record<GuidedStep,…>` total-key map raises a missing-key error (TS2741), `GUIDED_WORKFLOW_STEPS` omits the new step from the visual stepper, and the `never`-exhaustiveness assertion in `GuidedTurn.tsx` fails to narrow. There are **seven** such consumers (GuidedTurn `never` switch; GuidedHistory `STEP_LABELS` + `TURN_TYPE_LABELS`; GuidedChatHistory `STEP_LABELS`; ChatPanel `GUIDED_CHAT_PLACEHOLDERS` + `GUIDED_STEP_PURPOSES` + `GUIDED_WORKFLOW_STEPS`), so this task widens the unions AND extends all seven together. The `WireStageData` TS type is **P2.6** and the `interpretation_review` dead-case removal is **P4.T2**; the real `confirm_wiring`→`WireStageTurn` render replaces the placeholder added here once WireStageTurn exists (**P2.7**).

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
    // Wire stage (staged-recut P1): the placeholder turn is rendered in P2.7.
    // P2.4 replaces the skeleton payload with WireStageData.
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

- [ ] **Step 2: Extend the seven exhaustive consumers in the SAME slice (required for typecheck).**
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
  d) `ChatPanel.tsx` — add the new step to BOTH `Record<GuidedStep, string>` maps
  and to `GUIDED_WORKFLOW_STEPS`:
  ```tsx
      // in GUIDED_CHAT_PLACEHOLDERS
      step_4_wire: "Confirm how the steps connect, then continue.",
      // in GUIDED_STEP_PURPOSES
      step_4_wire: "Review and confirm the wiring between your pipeline steps.",
      // in GUIDED_WORKFLOW_STEPS
      { id: "step_4_wire", label: "Wire" },
  ```
  e) `guided.test.ts` — update the TurnType union test to include
  `"confirm_wiring"` and length `7`; update the GuidedStep union test to include
  `"step_4_wire"` and length `5`.
  f) `ChatPanel.test.tsx` — add or update a workflow-stepper assertion so a guided
  session at `step_4_wire` renders a `Wire` step marker and marks it active.

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

- [ ] **Step 5: Commit.**
  ```
  git add \
    src/elspeth/web/frontend/src/types/guided.ts \
    src/elspeth/web/frontend/src/types/guided.test.ts \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedHistory.tsx \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedChatHistory.tsx \
    src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
    src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
  git commit -m "feat(frontend): mirror STEP_4_WIRE + confirm_wiring into guided.ts unions"
  ```

### Task P1.6: Move terminal-stamping out of both completion seams and wire the minimal STEP_4 route path

**Files:**
- Modify `src/elspeth/web/composer/guided/steps.py:218-274` (`handle_step_2_5_recipe_apply`)
- Modify `src/elspeth/web/composer/guided/steps.py:277-406` (`handle_step_3_chain_accept`)
- Create the wire handler in `src/elspeth/web/composer/guided/steps.py` (`handle_step_4_wire_confirm`)
- Modify `src/elspeth/web/sessions/routes/_helpers.py` (`_dispatch_guided_respond`, `_emit_wire_turn`, handler imports/exports)
- Modify `src/elspeth/web/sessions/routes/composer/guided.py` (`_build_get_guided_turn` rebuild branch for `STEP_4_WIRE`)
- Modify `tests/integration/web/composer/guided/test_step_handlers.py:251-292,329-400`
- Create `tests/integration/web/composer/guided/test_wire_dispatch.py`

**Interfaces:**
- Produces (changed): `handle_step_2_5_recipe_apply` success path now sets `session.step = GuidedStep.STEP_4_WIRE`, `session.terminal = None` (no YAML render, no COMPLETED stamp).
- Produces (changed): `handle_step_3_chain_accept` success path now sets `session.step = GuidedStep.STEP_4_WIRE`, `session.terminal = None`, `session.step_3_proposal = proposal`.
- Produces (new): `handle_step_4_wire_confirm(*, state: CompositionState, session: GuidedSession) -> StepHandlerResult` — runs `state.validate()`; on `is_valid` stamps `TerminalState(COMPLETED, reason=None, pipeline_yaml=generate_yaml(state))`; on invalid leaves `terminal=None` and returns a non-success `StepHandlerResult` carrying the `ValidationSummary`. (P5.6 inserts the profile-gated advisor sign-off *before* the COMPLETED stamp; P1 gates on `validate().is_valid` only.)
- Produces (changed route behaviour): after recipe-apply, chain-accept, or repair-success commits, `_dispatch_guided_respond` emits `build_step_4_wire_turn(validation=state.validate())` and returns it as `next_turn` instead of `None`.
- Produces (changed route behaviour): a `CONFIRM_WIRING` response body must be exactly `chosen=["confirm"]` with `edited_values=None`, `custom_inputs=None`, `accepted_step_index=None`, `edit_step_index=None`, and `control_signal=None`; malformed bodies return HTTP 400 at the boundary.
- Produces (changed route behaviour): a valid `CONFIRM_WIRING` response dispatches to `handle_step_4_wire_confirm` and returns `terminal=COMPLETED`, `next_turn=None`.
- Produces (changed route behaviour): GET `/api/sessions/{session_id}/guided` on a persisted `STEP_4_WIRE` session rebuilds the skeleton wire turn.
- Consumes: `CompositionState.validate()` (state.py:2215), `generate_yaml` (yaml_generator.py:198), `TerminalState`, `TerminalKind`, `StepHandlerResult`, `ToolResult`.

This route wiring is deliberately minimal: P1 emits the skeleton wire payload from
P1.3. P2.4 replaces the emitter signature/body with the final two-read topology
payload; P2 then updates the same route seam to call the final emitter shape.

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

- [ ] **Step 3b: Write route-level tests for the now-self-contained P1 flow.**
  Create `tests/integration/web/composer/guided/test_wire_dispatch.py`. Reuse
  `MockPayloadStore` from `tests.fixtures.stores` and pass it through the dispatcher
  so `_store_guided_audit_payload` is exercised (no empty audit payload ids).
  The tests must cover:

  ```python
  @pytest.mark.asyncio
  async def test_chain_accept_returns_confirm_wiring_turn() -> None:
      state, step3_session, catalog, payload_store = _step3_ready_session()
      _state2, guided2, next_turn = await _dispatch(
          state,
          step3_session,
          catalog,
          payload_store=payload_store,
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
      assert guided2.step is GuidedStep.STEP_4_WIRE
      assert guided2.terminal is None
      assert next_turn is not None
      assert next_turn["type"] == TurnType.CONFIRM_WIRING.value


  @pytest.mark.asyncio
  async def test_confirm_wiring_stamps_completed_terminal() -> None:
      state, wire_session, catalog, payload_store = _wire_ready_session()
      _state2, guided2, next_turn = await _dispatch(
          state,
          wire_session,
          catalog,
          payload_store=payload_store,
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
      assert next_turn is None
      assert guided2.terminal is not None
      assert guided2.terminal.kind is TerminalKind.COMPLETED


  @pytest.mark.parametrize(
      "body",
      [
          {"chosen": None, "edited_values": None, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
          {"chosen": ["accept"], "edited_values": None, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
          {"chosen": ["confirm"], "edited_values": {}, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
          {"chosen": ["confirm"], "edited_values": None, "custom_inputs": [], "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
      ],
  )
  @pytest.mark.asyncio
  async def test_confirm_wiring_rejects_malformed_response_body(body) -> None:
      state, wire_session, catalog, payload_store = _wire_ready_session()
      with pytest.raises(HTTPException) as exc:
          await _dispatch(
              state,
              wire_session,
              catalog,
              payload_store=payload_store,
              current_step=GuidedStep.STEP_4_WIRE,
              current_turn_type=TurnType.CONFIRM_WIRING,
              turn_response=body,
          )
      assert exc.value.status_code == 400
  ```

  `_dispatch(...)` is a small local wrapper around `_dispatch_guided_respond` that
  passes `payload_store=payload_store`. `_step3_ready_session()` should drive
  source → sink and stage a `ChainProposal` on a `STEP_3_TRANSFORMS` session.
  `_wire_ready_session()` may call the step-3 accept handler directly, then assert
  `step=STEP_4_WIRE` and `terminal is None` before returning. Use the exact
  `GuidedRespondRequest` body shown above for a valid confirm:
  `chosen=["confirm"]` and every other response field `None`.

- [ ] **Step 3c: Run the route tests to confirm they fail.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py -q
  ```
  Expected: accept returns `next_turn is None`, `STEP_4_WIRE` confirm hits the
  unhandled-branch `InvariantError`, and malformed confirm bodies are not yet
  rejected by the boundary branch.

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
          # re-emits the skeleton wire turn. P2.4 upgrades that payload to carry
          # the validation overlay/warnings.
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

  `ToolResult` is defined at `composer/tools/_common.py:547-588`; its
  required fields are `success: bool`, `updated_state: CompositionState`,
  `validation: ValidationSummary`, `affected_nodes: tuple[str, ...]` (no default —
  pass `()`), with `data: Any = None` and the rest defaulted. `__post_init__`
  runs `freeze_fields(self, "affected_nodes", ...)`, so `affected_nodes=()` is
  mandatory. `ToolResult` is already imported into `steps.py` (steps.py:33-41).

- [ ] **Step 6b: Add a payload-store-backed `_emit_wire_turn` helper.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, import
  `build_step_4_wire_turn` from `guided.emitters` and `Turn` from
  `guided.protocol`. Add this helper immediately above `_dispatch_guided_respond`:

  ```python
  def _emit_wire_turn(
      *,
      state: CompositionState,
      guided: GuidedSession,
      recorder: BufferingRecorder,
      user_id: str,
      payload_store: Any,
  ) -> tuple[GuidedSession, Turn]:
      """Emit the STEP_4_WIRE confirm_wiring turn after an accept commit (P1.6).

      P1 leaves the accept seams at ``step=STEP_4_WIRE, terminal=None``; this
      builds the skeleton wire turn, appends its server TurnRecord, and emits the
      audited ``turn_emitted`` event so the user lands on the wire stage rather
      than silently falling off the wizard with ``next_turn=None``.
      """
      next_turn = build_step_4_wire_turn(validation=state.validate())
      payload_hash = stable_hash(next_turn["payload"])
      new_record = TurnRecord(
          step=GuidedStep.STEP_4_WIRE,
          turn_type=TurnType.CONFIRM_WIRING,
          payload_hash=payload_hash,
          response_hash=None,
          emitter="server",
      )
      emit_turn_emitted(
          recorder,
          step=GuidedStep.STEP_4_WIRE,
          turn_type=TurnType.CONFIRM_WIRING,
          payload_hash=payload_hash,
          payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
          emitter="server",
          composition_version=state.version,
          actor=user_id,
      )
      return _replace(guided, history=(*guided.history, new_record)), next_turn
  ```

  Do not use a placeholder payload id. The dispatcher already receives
  `payload_store`; thread it into every `_emit_wire_turn` call.

- [ ] **Step 6c: Replace accept-commit `None` returns with `_emit_wire_turn`.**
  In `_dispatch_guided_respond`, replace the three post-commit returns:
  recipe-apply, chain-accept repair-success, and chain-accept success. Each should
  follow this shape:

  ```python
  guided, next_turn = _emit_wire_turn(
      state=handler_result.state,
      guided=handler_result.session,
      recorder=recorder,
      user_id=user_id,
      payload_store=payload_store,
  )
  return handler_result.state, guided, next_turn
  ```

  For the repair-success branch, use `repair_result.state` /
  `repair_result.session`. The point is invariant: after a successful commit, the
  route returns a `confirm_wiring` turn, never `(…, next_turn=None)`.

- [ ] **Step 6d: Add the `STEP_4_WIRE` / `CONFIRM_WIRING` dispatch branch.**
  In `_dispatch_guided_respond`, insert the branch before the final unhandled
  `InvariantError`:

  ```python
      if current_step is GuidedStep.STEP_4_WIRE:
          if current_turn_type is not TurnType.CONFIRM_WIRING:
              raise HTTPException(
                  status_code=400,
                  detail=f"STEP_4_WIRE expects confirm_wiring; got {current_turn_type!r}.",
              )
          if (
              turn_response["chosen"] != ["confirm"]
              or turn_response["edited_values"] is not None
              or turn_response["custom_inputs"] is not None
              or turn_response["accepted_step_index"] is not None
              or turn_response["edit_step_index"] is not None
              or turn_response["control_signal"] is not None
          ):
              raise HTTPException(
                  status_code=400,
                  detail=(
                      "confirm_wiring response must be exactly chosen=['confirm'] "
                      "with edited_values/custom_inputs/step indices/control_signal all null."
                  ),
              )

          handler_result = handle_step_4_wire_confirm(state=state, session=guided)
          guided = handler_result.session
          if guided.terminal is not None:
              return handler_result.state, guided, None

          # Invalid pipeline: re-emit the skeleton wire turn. P2.4 upgrades this
          # payload to include warnings/contract overlays; P3 wires interpretation
          # re-surfacing for reconciliation.
          guided, next_turn = _emit_wire_turn(
              state=handler_result.state,
              guided=guided,
              recorder=recorder,
              user_id=user_id,
              payload_store=payload_store,
          )
          return handler_result.state, guided, next_turn
  ```

- [ ] **Step 6e: Add the `GET /api/sessions/{session_id}/guided` rebuild branch for wire-positioned sessions.**
  In `src/elspeth/web/sessions/routes/composer/guided.py`, import
  `build_step_4_wire_turn` and add this branch in `_build_get_guided_turn` before
  the trailing `return None`:

  ```python
      if step is GuidedStep.STEP_4_WIRE:
          return build_step_4_wire_turn(validation=state.validate())
  ```

- [ ] **Step 7: Run the tests to confirm they pass.**
  ```
  uv run pytest \
    tests/integration/web/composer/guided/test_step_handlers.py \
    tests/integration/web/composer/guided/test_wire_dispatch.py \
    -k "TerminalStampInvariant or redirects_to_wire or confirm_wiring or wire_turn" -q
  ```
  Expected: all PASS — both seams leave `terminal is None` + `step == STEP_4_WIRE`;
  route accept returns `confirm_wiring`; the wire handler stamps COMPLETED on the
  valid pipeline; malformed confirm bodies return 400.

- [ ] **Step 8: Run the full step-handler + auto-drop suite to catch fallout.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_step_handlers.py tests/integration/web/composer/guided/test_auto_drop.py -q
  ```
  Expected: PASS. Any route-level accept→complete assertion must now be a two-hop
  flow: accept returns `confirm_wiring`; a second POST with `chosen=["confirm"]`
  stamps COMPLETED. Do not xfail this to P2.

- [ ] **Step 9: Export the new handler.**
  Add `handle_step_4_wire_confirm` to any `__all__` / re-export of `steps.py`
  handlers. Check `_helpers.py:97-100` (which imports `handle_step_2_5_recipe_apply`
  and `handle_step_3_chain_accept`) and the `_helpers.py` `__all__` (the
  `handle_step_*` entries near `:3938-3940`) — add the import + `__all__` entry so
  this task can dispatch to it:

  ```python
  # _helpers.py import block (near :97)
      handle_step_4_wire_confirm,
  # _helpers.py __all__ (near :3940)
      "handle_step_4_wire_confirm",
  ```

  (P1 imports it so the symbol is reachable and dispatches to it immediately.)

- [ ] **Step 10: Commit.**
  ```
  git add \
    src/elspeth/web/composer/guided/steps.py \
    src/elspeth/web/sessions/routes/_helpers.py \
    src/elspeth/web/sessions/routes/composer/guided.py \
    tests/integration/web/composer/guided/test_step_handlers.py \
    tests/integration/web/composer/guided/test_wire_dispatch.py
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
  Expected: PASS. Route-level completion assertions must use the two-hop flow
  introduced in P1.6 (accept → `confirm_wiring` turn → confirm → COMPLETED);
  there should be no xfail deferring wire dispatch to P2.

- [ ] **Step 4: Frontend typecheck + the mirror gate (final).**
  ```
  npm --prefix src/elspeth/web/frontend run typecheck
  uv run python scripts/cicd/check_slot_type_cross_language.py
  ```
  Expected: both PASS.

- [ ] **Step 5: Final phase commit (if Steps 1-2 produced format/import fixups).**
  ```
  git add \
    src/elspeth/web/composer/guided/protocol.py \
    src/elspeth/web/composer/guided/prompts.py \
    src/elspeth/web/composer/guided/skills/step_4_wire.md \
    src/elspeth/web/composer/guided/emitters.py \
    src/elspeth/web/composer/guided/state_machine.py \
    src/elspeth/web/composer/guided/steps.py \
    src/elspeth/web/sessions/routes/_helpers.py \
    src/elspeth/web/sessions/routes/composer/guided.py \
    src/elspeth/web/frontend/src/types/guided.ts \
    src/elspeth/web/frontend/src/types/guided.test.ts \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedHistory.tsx \
    src/elspeth/web/frontend/src/components/chat/guided/GuidedChatHistory.tsx \
    src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
    src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx \
    tests/unit/web/composer/guided/test_protocol.py \
    tests/unit/web/composer/guided/test_skill.py \
    tests/unit/web/composer/guided/test_emitters.py \
    tests/unit/web/composer/guided/test_state_machine.py \
    tests/integration/web/composer/guided/test_step_handlers.py \
    tests/integration/web/composer/guided/test_wire_dispatch.py
  git commit -m "chore(guided): P1 wire-stage skeleton gate reconciliation"
  ```
  (Skip if nothing changed after the per-task commits.)

---
