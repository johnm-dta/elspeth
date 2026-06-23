> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P2 — Wire stage data model (B2)

> **Scope.** The `STEP_4_WIRE` turn payload returns **two reads** — *neither alone
> suffices*: (1) `_serialize_full_pipeline_state` (`composer/tools/sessions.py:1084`,
> via `_serialize_source/_serialize_node/_serialize_output` in `_common.py:968-1010`)
> for the **connection-label topology** (`input/on_success/on_error/routes/fork_to` +
> source `on_success`), and (2) the `edge_contracts` (+ `semantic_contracts`) overlay
> from `state.validate()` (built into `_authoring_validation_payload`,
> `sessions.py:1166-1175`; surfaced by `_execute_preview_pipeline`, `generation.py:1651`,
> summary `:1685-1692`). `EdgeContract.to_dict()` (`state.py:359-368`) emits keys
> **`from`/`to`** — NOT `from_id`/`to_id`. The render reconstructs edges from connection
> labels (NEVER `state.edges` — guided passes `edges=[]`) and overlays `edge_contracts`
> keyed by `(from, to)`. B6: after any wire-stage reconciliation the confirm gate
> re-evaluates `validate().is_valid` AND re-runs the P3 surfacing pass.
>
> **Dependencies on other phases (must land first):** P1 owns
> `GuidedStep.STEP_4_WIRE`, `TurnType.CONFIRM_WIRING`, the `emitters.py` `_ORDER`
> tuple entry for `STEP_4_WIRE` (consumed by `_step_index`), and the `guided.ts`
> `GuidedStep`/`TurnType` union strings. P1 also owns the minimal dispatcher path
> (accept emits a skeleton `confirm_wiring` turn; confirm dispatches to
> `handle_step_4_wire_confirm`). This phase upgrades that path to the final
> two-read wire payload. This phase's emitter calls `_step_index(GuidedStep.STEP_4_WIRE)`
> and stamps `TurnType.CONFIRM_WIRING.value`; both must exist. The P3 surfacing entry point `_surface_pending_interpretation_reviews`
> is consumed by the B6 re-surface helper (P2.8) and bound in the dispatcher path
> (P2.9) — until P3 lands, the re-surface call is threaded as a passed-in callback
> so this phase stays independently testable.

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
      topology connection labels, never from state.edges. Output sinks do not have
      a separate input field; their ``sink_name`` is the connection label consumed
      by the sink, so renderers create upstream→sink edges by matching
      source/node ``on_success``/route labels to ``outputs[].sink_name``.

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
  "edge_contracts", "semantic_contracts", "warnings"})`; `_NESTED_SHAPES[TurnType.CONFIRM_WIRING]
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
          payload = {"edge_contracts": [], "semantic_contracts": [], "warnings": []}
          err = validate_payload(TurnType.CONFIRM_WIRING, payload)
          assert err is not None
          assert "topology" in err

      def test_topology_must_be_mapping_with_expected_keys(self) -> None:
          payload = {
              "topology": {"sources": {}},  # missing nodes/outputs
              "edge_contracts": [],
              "semantic_contracts": [],
              "warnings": [],
          }
          err = validate_payload(TurnType.CONFIRM_WIRING, payload)
          assert err is not None
          assert "topology" in err

      def test_missing_warnings_rejected(self) -> None:
          payload = {
              "topology": {"sources": {}, "nodes": [], "outputs": []},
              "edge_contracts": [],
              "semantic_contracts": [],
          }
          err = validate_payload(TurnType.CONFIRM_WIRING, payload)
          assert err is not None
          assert "warnings" in err
  ```
- [ ] **Step 2: Run to fail.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestConfirmWiringValidation -q`
  Expected: if P1 set different keys, `test_missing_topology_rejected`/`test_valid_wire_payload_passes`
  fail with a required-key mismatch; if `CONFIRM_WIRING` is unregistered, `KeyError`/`ValueError`
  from `validate_payload`.
- [ ] **Step 3: Reconcile the registry entries.** In `_REQUIRED_KEYS` (`:200-218`)
  ensure the `CONFIRM_WIRING` entry is exactly:
  ```python
      TurnType.CONFIRM_WIRING: frozenset({"topology", "edge_contracts", "semantic_contracts", "warnings"}),
  ```
  In `_NESTED_SHAPES` (`:243-253`) ensure the `CONFIRM_WIRING` entry is exactly:
  ```python
      TurnType.CONFIRM_WIRING: (
          ("topology", "mapping", frozenset({"sources", "nodes", "outputs"})),
      ),
  ```
- [ ] **Step 4: Run to pass.**
  `uv run pytest tests/unit/web/composer/guided/test_wire_payload.py::TestConfirmWiringValidation -q`
  Expected: `4 passed`.
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
- Consumes: `CompositionState` (`composer/state.py:1970`), `_serialize_full_pipeline_state`
  (`composer/tools/sessions.py:1084`)
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
          on_success="jsonl_out",
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
          assert node_by_id["mapper"]["on_success"] == "jsonl_out"
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

      def test_output_sink_name_is_preserved_as_connection_label(self) -> None:
          topo = _build_wire_topology(_canonical_state())
          assert topo["outputs"] == [{"sink_name": "jsonl_out", "plugin": "json"}]
          assert any(n["on_success"] == "jsonl_out" for n in topo["nodes"])
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
  Expected: `4 passed`.
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
- Consumes: `CompositionState.validate()` (`composer/state.py:2225`, returns
  `ValidationSummary` with `edge_contracts`/`semantic_contracts`/`warnings`),
  `_authoring_validation_payload` (`composer/tools/sessions.py:1166`),
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
    advisor_findings?: string;
    signoff_outcome?: string;
  }
  ```

> `edge_contracts` keys are `from`/`to` (NOT `from_id`/`to_id`) — M1. `node.plugin` is
> nullable (gates/coalesces). Output sinks consume `output.sink_name` as their
> connection label, so renderers reconstruct upstream→sink edges as well as
> source/node→node edges. `advisor_findings` and `signoff_outcome` are optional and
> appear only on P5 sign-off revise re-emits. Vitest type-assertion failures ARE the test (the array
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
        advisor_findings: "FLAGGED: prompt omits row field",
        signoff_outcome: "revise",
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
   * (source.on_success -> node.input or output.sink_name;
   * node.on_success/routes/fork_to -> downstream node.input or output.sink_name)
   * and overlay `edge_contracts` keyed by (from, to). NEVER render state.edges
   * directly (guided passes edges=[]). Coalesce/fork edges are honest-gap — they
   * appear in topology with no matching edge_contracts row (render: "unchecked").
   * advisor_findings/signoff_outcome are optional and appear only on P5 sign-off
   * revise re-emits.
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
    advisor_findings?: string;
    signoff_outcome?: string;
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
- Modify `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`

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
> node.input` or `output.sink_name`; `node.on_success / routes / fork_to ->
> downstream node.input` or `output.sink_name`) and overlays `edge_contracts` keyed
> by `(from, to)`. An edge with no matching contract row gets `satisfied: null`
> (honest-gap, e.g. fork/coalesce). NEVER reads `state.edges`.
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
            on_error: "scrape_error",
            routes: null,
            fork_to: null,
          },
          {
            id: "mapper",
            node_type: "transform",
            plugin: "field_mapper",
            input: "scraped",
            on_success: "jsonl_out",
            on_error: null,
            routes: null,
            fork_to: null,
          },
          {
            id: "error_handler",
            node_type: "transform",
            plugin: "field_mapper",
            input: "scrape_error",
            on_success: "jsonl_out",
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
      // scrape.on_error=scrape_error -> error_handler.input=scrape_error
      expect(pairs).toContainEqual(["scrape", "error_handler"]);
      // mapper.on_success=jsonl_out -> output.sink_name=jsonl_out
      expect(pairs).toContainEqual(["mapper", "jsonl_out"]);
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
   * constraint): source.on_success -> node.input or output.sink_name;
   * node.on_success / routes values / fork_to -> downstream node.input or
   * output.sink_name. NEVER reads state.edges (guided passes edges=[]). Overlays
   * edge_contracts keyed by (from, to); an edge with no matching contract row is
   * honest-gap (satisfied=null, e.g. fork/coalesce).
   */
  export function reconstructWireEdges(data: WireStageData): WireEdge[] {
    const { sources, nodes, outputs } = data.topology;
    // Map each consumed label -> the node id OR output sink name that reads it.
    const consumerByLabel = new Map<string, string>();
    for (const node of nodes) {
      if (node.input !== null) {
        consumerByLabel.set(node.input, node.id);
      }
    }
    for (const output of outputs) {
      consumerByLabel.set(output.sink_name, output.sink_name);
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
      pushEdge(node.id, node.on_error);
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
- [ ] **Step 5: Wire `WireStageTurn` into the active `GuidedTurn` dispatcher.**
  Replace the P1 placeholder `case "confirm_wiring": return null;` in
  `GuidedTurn.tsx` with:
  ```tsx
      case "confirm_wiring":
        return (
          <WireStageTurn
            key={turnInstanceKey}
            data={turn.payload as WireStageData}
            confirmDisabled={disabled}
            onConfirm={() =>
              guardedSubmit({
                chosen: ["confirm"],
                edited_values: null,
                custom_inputs: null,
                accepted_step_index: null,
                edit_step_index: null,
                control_signal: null,
              })
            }
          />
        );
  ```
  Add `WireStageData` to the type import from `@/types/guided` and import
  `{ WireStageTurn }` from `./WireStageTurn`. Add a dispatcher assertion in the
  nearest guided-turn test surface (or in `WireStageTurn.test.tsx` if no dedicated
  dispatcher test exists) that rendering `GuidedTurn` with `type="confirm_wiring"`
  renders the WireStageTurn UI, and clicking "Confirm wiring" calls `onSubmit` with
  exactly this `GuidedRespondRequest` body: `chosen: ["confirm"]` and all other
  response fields `null`.
- [ ] **Step 6: Typecheck + build (from `src/elspeth/web/frontend`).**
  `npm run typecheck && npm run build`
  Expected: both succeed (no TS errors).
- [ ] **Step 7: Commit.**
  `git add src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx && git commit -m "feat(frontend): WireStageTurn renders edges from connection labels + contract overlay (P2.7)"`

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

### Task P2.9: Upgrade the P1 dispatcher wire path to the final payload + bind B6 rebuild

**Files:**
- Modify `src/elspeth/web/sessions/routes/_helpers.py` (`_dispatch_guided_respond`,
  `:2571`): update the P1 `_emit_wire_turn` helper to call the final
  `build_step_4_wire_turn(state)` signature, keep payload-store-backed audit ids,
  and bind `rebuild_wire_turn_after_reconciliation` on invalid/reconciliation re-emits
- Modify `src/elspeth/web/sessions/routes/composer/guided.py` (`_build_get_guided_turn`,
  `:91` function): update the P1 `STEP_4_WIRE` rebuild branch to call the final
  `build_step_4_wire_turn(state)` signature
- Modify `tests/integration/web/composer/guided/test_wire_dispatch.py`

**Interfaces:**
- Consumes: `build_step_4_wire_turn` (**P2.4**, positional `state`),
  `rebuild_wire_turn_after_reconciliation` (**P2.8**),
  `handle_step_4_wire_confirm` (**P1.6**), `GuidedStep.STEP_4_WIRE` +
  `TurnType.CONFIRM_WIRING` (**P1**), `TurnRecord`, `stable_hash`,
  `emit_turn_emitted`, `_store_guided_audit_payload`, `_replace`.
- Produces: behaviour — the P1 accept/confirm route path stays live, but all
  emitted/rebuilt wire turns now use the final two-read payload. `_emit_wire_turn`
  accepts `payload_store` and persists the payload via `_store_guided_audit_payload`;
  no placeholder audit payload id is allowed.
- Produces: boundary validation remains exact: `CONFIRM_WIRING` accepts only
  `GuidedRespondRequest` body `chosen=["confirm"]` with every other response field
  `null`; malformed bodies return HTTP 400.
- Produces: invalid/reconciliation re-emits bind `rebuild_wire_turn_after_reconciliation`
  so the B6 helper is not a dead exported utility. Until P3 lands, pass
  `resurface=lambda _state: None`; P3 replaces that no-op with
  `_surface_pending_interpretation_reviews`.

> **Why this task exists.** P1.6 made the route path self-contained using the
> skeleton emitter. P2.4 changes the emitter signature and payload contract, and
> P2.8 adds the B6 rebuild helper. This task updates the live route path to those
> final P2 contracts without reintroducing `next_turn=None`, audit payload gaps, or
> permissive confirm bodies.

- [ ] **Step 1: Read the P1 wire route path and anchor the signature upgrade.**
  ```
  rg -n "_emit_wire_turn|build_step_4_wire_turn|STEP_4_WIRE|CONFIRM_WIRING|handle_step_4_wire_confirm|rebuild_wire_turn_after_reconciliation" /home/john/elspeth/src/elspeth/web/sessions/routes/_helpers.py /home/john/elspeth/src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Confirm: P1.6 already replaced accept-commit `next_turn=None` returns with
  `_emit_wire_turn(...)`, already dispatches `CONFIRM_WIRING` to
  `handle_step_4_wire_confirm`, and already rebuilds
  `GET /api/sessions/{session_id}/guided` for
  `STEP_4_WIRE`. This task upgrades those P1 skeleton calls from
  `build_step_4_wire_turn(validation=state.validate())` to
  `build_step_4_wire_turn(state)` and binds the P2.8 rebuild helper.

> **Pre-condition — P1.6 must be complete before running this step.** The test below
> imports `_empty_state` from
> `tests/integration/web/composer/guided/test_step_handlers.py`, which is a **P1.6
> deliverable** (not yet present in the repo at plan-write time). Confirm that file
> exists and exports `_empty_state` before executing Step 2 onward.

- [ ] **Step 2: Update the P1 dispatch tests for final payload, payload-store audit, and malformed confirm bodies.**
  Modify `tests/integration/web/composer/guided/test_wire_dispatch.py`. It should
  still drive chain-accept through `_dispatch_guided_respond`, but now it also
  asserts the returned wire turn uses the final P2 payload keys, persists its
  payload through `MockPayloadStore`, and rejects malformed `CONFIRM_WIRING`
  response bodies:
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
  from tests.fixtures.stores import MockPayloadStore
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
      return accept.state, accept.session, catalog, MockPayloadStore()


  async def _dispatch(state, session, catalog, *, payload_store, current_step, current_turn_type, turn_response):
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
          payload_store=payload_store,
          model="m",
          temperature=None,
          seed=None,
      )


  @pytest.mark.asyncio
  async def test_wire_confirm_completes_a_wire_ready_session() -> None:
      # The accept handler already left the session at STEP_4_WIRE (terminal=None);
      # a CONFIRM_WIRING confirm must dispatch to handle_step_4_wire_confirm and
      # stamp COMPLETED on the valid pipeline.
      state, session, catalog, payload_store = _wire_ready_session_and_state()
      _s2, guided2, _t2 = await _dispatch(
          state,
          session,
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
      assert guided2.terminal is not None
      assert guided2.terminal.kind is TerminalKind.COMPLETED


  @pytest.mark.asyncio
  async def test_chain_accept_lands_on_wire_turn_not_none() -> None:
      # Re-stage the proposal on a STEP_3 session and drive the PROPOSE_CHAIN accept
      # through the dispatcher: the dispatcher must now return the wire turn (not None).
      state, wire_session, catalog, payload_store = _wire_ready_session_and_state()
      # Reconstruct a STEP_3-positioned session carrying the same proposal so the
      # dispatcher's accept path runs (the wire_session is already past STEP_3).
      from dataclasses import replace as _dc_replace

      step3_session = _dc_replace(wire_session, step=GuidedStep.STEP_3_TRANSFORMS, terminal=None)
      _new_state, guided, next_turn = await _dispatch(
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
      # P1.6 left terminal=None, step=STEP_4_WIRE; P2.9 now emits the wire turn.
      assert guided.terminal is None
      assert guided.step is GuidedStep.STEP_4_WIRE
      assert next_turn is not None
      assert next_turn["type"] == TurnType.CONFIRM_WIRING.value
      assert set(next_turn["payload"]) == {
          "topology",
          "edge_contracts",
          "semantic_contracts",
          "warnings",
      }
      assert payload_store._storage, "wire turn payload must be persisted for audit"


  @pytest.mark.parametrize(
      "body",
      [
          {"chosen": None, "edited_values": None, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
          {"chosen": ["accept"], "edited_values": None, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
          {"chosen": ["confirm"], "edited_values": {}, "custom_inputs": None, "accepted_step_index": None, "edit_step_index": None, "control_signal": None},
      ],
  )
  @pytest.mark.asyncio
  async def test_confirm_wiring_rejects_malformed_response_body(body) -> None:
      from fastapi import HTTPException

      state, session, catalog, payload_store = _wire_ready_session_and_state()
      with pytest.raises(HTTPException) as exc:
          await _dispatch(
              state,
              session,
              catalog,
              payload_store=payload_store,
              current_step=GuidedStep.STEP_4_WIRE,
              current_turn_type=TurnType.CONFIRM_WIRING,
              turn_response=body,
          )
      assert exc.value.status_code == 400
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
  uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py -q
  ```
  Expected failure: P1's route path still calls the skeleton emitter signature
  (`validation=...`), so the final payload key assertion fails or the route raises
  `TypeError` once P2.4 has changed `build_step_4_wire_turn` to positional
  `state`. The malformed-body tests should already pass from P1.6; if they do not,
  fix the boundary branch in this task before proceeding.

- [ ] **Step 4: Upgrade the P1 `_emit_wire_turn` helper to the final emitter signature and audit payload store.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, update the P1 helper to this
  final shape:
  ```python
  def _emit_wire_turn(
      *,
      state: CompositionState,
      guided: GuidedSession,
      recorder: BufferingRecorder,
      user_id: str,
      payload_store: Any,
      next_turn: Turn | None = None,
  ) -> tuple[GuidedSession, Turn]:
      """Emit the STEP_4_WIRE confirm_wiring turn after an accept/rebuild (P2.9).

      P1.6 created this helper with the skeleton emitter. P2.9 upgrades it to
      the final two-read payload and persists the emitted payload through the
      configured payload store for the guided audit row.
      """
      if next_turn is None:
          next_turn = build_step_4_wire_turn(state)
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
      guided = _replace(guided, history=(*guided.history, new_record))
      return guided, next_turn
  ```
  Ensure every caller passes the existing dispatcher `payload_store` argument.

- [ ] **Step 5: Update every `_emit_wire_turn` call site to pass `payload_store`.**
  P1.6 already replaced the recipe-apply, chain-accept repair-success, and
  chain-accept success returns with `_emit_wire_turn(...)`. Update each call to:
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
  Use `repair_result.state` / `repair_result.session` in the repair-success branch.
  No accept-commit path may return `next_turn=None`.

- [ ] **Step 6: Harden the existing `STEP_4_WIRE` / `CONFIRM_WIRING` branch and bind B6 rebuild.**
  P1.6 already added this branch. Keep its exact body validation and update the
  invalid/reconciliation re-emit to use `rebuild_wire_turn_after_reconciliation`:
  ```python
      # --- STEP_4_WIRE turns ----------------------------------------------
      # A CONFIRM_WIRING confirm stamps COMPLETED on a valid pipeline. (P5.6
      # layers the profile-gated advisor sign-off in FRONT of the COMPLETED
      # stamp; P2.9 ships the validate-gate-only confirm.)
      if current_step is GuidedStep.STEP_4_WIRE:
          if current_turn_type is not TurnType.CONFIRM_WIRING:
              raise HTTPException(
                  status_code=400,
                  detail=f"STEP_4_WIRE expects a confirm_wiring response; got turn_type={current_turn_type!r}.",
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
              # COMPLETED (valid pipeline). No further turn.
              return handler_result.state, guided, None

          # Invalid: rebuild the final wire turn from a fresh validate(). P3
          # replaces this no-op with _surface_pending_interpretation_reviews.
          next_turn, _is_valid = rebuild_wire_turn_after_reconciliation(
              handler_result.state,
              resurface=lambda _state: None,
          )
          guided, next_turn = _emit_wire_turn(
              state=handler_result.state,
              guided=guided,
              next_turn=next_turn,
              recorder=recorder,
              user_id=user_id,
              payload_store=payload_store,
          )
          return handler_result.state, guided, next_turn
  ```
  If the implementation keeps `_emit_wire_turn` as the single append/audit helper,
  split its builder and appender internally rather than duplicating TurnRecord
  logic. The invariant is what matters: the invalid/reconciliation path calls
  `rebuild_wire_turn_after_reconciliation`, persists the rebuilt payload via
  `_store_guided_audit_payload`, and appends a server `TurnRecord`.

- [ ] **Step 7: Add the `GET /api/sessions/{session_id}/guided` rebuild branch for `STEP_4_WIRE`.**
  In `src/elspeth/web/sessions/routes/composer/guided.py`, in the `_build_get_guided_turn`
  function (`:91`), after the
  `if step is GuidedStep.STEP_3_TRANSFORMS:` block (its `return build_step_3_propose_chain_turn(...)`
  is at `:195`) and BEFORE the trailing `return None` at `:199`, insert:
  ```python
      if step is GuidedStep.STEP_4_WIRE:
          # Rebuild the wire turn deterministically from the current state
          # (build_step_4_wire_turn is pure: topology + validate() overlay).
          return build_step_4_wire_turn(state)
  ```
  Add `build_step_4_wire_turn` to the existing emitter import block in `composer/guided.py`
  (the block that imports `build_step_3_propose_chain_turn` at `:68`).
  Add a route regression to `tests/integration/web/composer/guided/test_wire_dispatch.py`
  that persists a `GuidedSession` at `step=GuidedStep.STEP_4_WIRE`, calls
  `GET /api/sessions/{session_id}/guided`, and asserts:
  ```python
      assert body["next_turn"]["type"] == "confirm_wiring"
      payload = body["next_turn"]["payload"]
      assert {"topology", "edge_contracts", "semantic_contracts", "warnings"} <= set(payload)
      assert {"sources", "nodes", "outputs"} <= set(payload["topology"])
  ```
  The branch must be a sibling of the STEP_3 branch, before the final
  `return None`; if nested under STEP_3, this GET regression returns
  `next_turn=None`.

- [ ] **Step 8: Run to pass.**
  ```
  uv run pytest tests/integration/web/composer/guided/test_wire_dispatch.py -q
  ```
  Expected: all pass — accept lands on the wire turn; the confirm stamps COMPLETED.

- [ ] **Step 9: Un-xfail the P1.6 route-level COMPLETED assertions.**
  The P1.6 Step-8 note allowed a temporary route-level xfail for "accept →
  COMPLETED" assertions in `test_auto_drop.py`. With the wire dispatch present,
  update those to the two-hop flow (accept → wire turn → confirm → COMPLETED) and
  remove the xfail. Run:
  ```
  uv run pytest tests/integration/web/composer/guided/test_auto_drop.py tests/unit/web/sessions/routes/ -q -k "guided or wire or auto_drop"
  ```
  Expected: all pass (no remaining xfail referencing the wire dispatch).

- [ ] **Step 10: Commit.**
  ```
  git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py tests/integration/web/composer/guided/test_wire_dispatch.py tests/integration/web/composer/guided/test_auto_drop.py && git commit -m "feat(web/sessions): emit wire turn after accept + add CONFIRM_WIRING dispatch + GET rebuild (P2.9)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
  ```

---

### Task P2.10: Phase gate sweep

**Files:** none (verification only)

**Interfaces:** none

- [ ] **Step 1: ruff check.**
  `uv run ruff check src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/composer/guided/test_wire_payload.py tests/integration/web/composer/guided/test_wire_dispatch.py`
  Expected: `All checks passed!`.
- [ ] **Step 2: ruff format check.**
  `uv run ruff format --check src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py`
  Expected: `... files already formatted`.
- [ ] **Step 3: mypy.**
  `uv run mypy src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py`
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
- [ ] **Step 7: Wardline full-root gate.**
  `wardline scan . --fail-on ERROR`
  Expected: exit 0. Exit 1 means the trust-boundary gate tripped; fix findings at
  the boundary and re-run. Exit 2 is a Wardline/tooling error that must be surfaced
  before this phase can close.
- [ ] **Step 8: Commit (if any formatter touched files).**
  ```
  git add src/elspeth/web/composer/guided/emitters.py src/elspeth/web/composer/guided/protocol.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/frontend/src/types/guided.ts src/elspeth/web/frontend/src/types/guided.test.ts src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/WireStageTurn.test.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx tests/unit/web/composer/guided/test_wire_payload.py tests/unit/web/composer/guided/test_emitters.py tests/integration/web/composer/guided/test_wire_dispatch.py tests/integration/web/composer/guided/test_auto_drop.py && git commit -m "chore(guided): P2 wire-data gate sweep clean (P2.10)" --allow-empty
  ```

---
