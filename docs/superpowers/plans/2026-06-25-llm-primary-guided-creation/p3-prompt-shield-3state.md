# Always-on 3-state prompt-shield review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Make the prompt-injection-shield review fire on **every** unshielded LLM node (not only when an untrusted `web_scrape` producer is upstream) as an always-on, advisory, per-node A/B/C determination — State A (already shielded, silent), State B (an authorized shield is deployment-available but not wired, strong "use it" advisory), State C (no shield available, high-risk "reconsider" advisory) — reusing `pipeline_decision` with `user_term="prompt_injection_shield_recommendation"` and adding NO new `InterpretationKind`.

**Architecture:** The runtime substance lives in `src/elspeth/web/interpretation_state.py` (the `prompt_shield_recommendation_warning_pairs` gate + the State-A upstream walk + the draft constants) surfaced through `CompositionState.validate()`'s advisory `warnings` list (`src/elspeth/web/composer/state.py:2414-2421`) — never `errors`, never the blocking contract. `validate()` is a pure, context-free method called from ~15 sites, so availability (the B-vs-C signal) cannot be threaded into it; instead the always-on warning defaults to the **fail-safe State C** draft, and a separate context-aware helper (`prompt_shield_state_for_node`) lets callers that hold a `ToolContext` (secret_service + user_id) refine B-vs-C via the existing discovery surface `secret_unavailable_message` over `azure_prompt_shield`. The agent-prose tuple in `src/elspeth/plugins/transforms/llm/transform.py:1714-1722` is re-worded to describe always-on A/B/C (documentation only; it does not make the review fire). **Dependencies:** p3 has no dependency on p1/p2 and can start immediately; p4 (tutorial) CONSUMES p3's `prompt_shield_state_for_node` + the B/C draft constants (the worked example lands in State C).

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy (SQLite), pytest. Frozen-dataclass `CompositionState`/`NodeSpec`. CI gates: plugin `source_file_hash` + tier-model fingerprints (both operator-owed re-signs).

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

## Design decision this plan OWNS (read before starting)

The contract (§2.3) says p3 DECIDES how shield availability reaches the pure
`CompositionState.validate()` call to `prompt_shield_recommendation_warning_pairs`.
**DECISION:**

1. **Always-on, decoupled from the untrusted-producer precondition.** Today the
   warning fires only when `_llm_consumes_untrusted_remote_content(node, ...)` is
   True (an upstream `web_scrape`). Under always-on, the warning fires for **every**
   `llm` node that does NOT already have an authorized shield upstream (State A) and
   does NOT already carry a pending shield requirement. The "untrusted producer"
   condition is dropped from the *fire-or-not* decision (it survives only as
   colour in the message text when present).

2. **State A is the existing upstream-shield walk, decoupled.** A NEW pure helper
   `_llm_has_authorized_shield_upstream(node, producer_by_output_stream)` returns
   True iff an authorized shield (`_AUTHORIZED_PROMPT_SHIELD_PLUGINS`) is reachable
   upstream — judged on its own, NOT coupled to reaching an untrusted producer
   first. State A ⇒ silent (no warning).

3. **B-vs-C cannot reach the pure `validate()`** (it has no `secret_service`/
   `user_id`; it is called from ~15 context-free sites). So:
   - `prompt_shield_recommendation_warning_pairs(state, *, shield_available: bool | None = None)`
     gains ONE optional keyword. `validate()` calls it **parameterless** ⇒
     `shield_available=None` ⇒ **fail-safe State C** draft (always-on warning still
     fires for every unshielded LLM node; the contract's "default C when
     undeterminable" rule).
   - **The B-vs-C refinement happens at the ONE user-facing surface that DOES hold
     secret context — the wire-stage turn (`build_step_4_wire_turn`).** Task 3 wires
     `azure_prompt_shield_available(ToolContext(...))` at the guided route, passes
     the resolved `shield_available` bool into the wire-turn builder, and the
     builder post-processes `validate()`'s C-default warnings into the B-draft
     per-node via `refine_prompt_shield_warnings_for_availability`. WITHOUT Task 3
     the B-draft is DEAD in production and a user with the Azure key sees a
     factually-wrong State-C advisory. This is "the real new plumbing p3 owns"
     (contract §2.3); it is NOT optional.
   - A SEPARATE context-aware helper
     `prompt_shield_state_for_node(node, all_nodes, *, shield_available: bool) -> str`
     returns `"A"`/`"B"`/`"C"` and is what other context-holding callers (and p4's
     tutorial consumer) use to compute the precise B-vs-C state. p4 does NOT
     re-plumb `validate()`.

4. **Three draft constants.** Keep `PROMPT_SHIELD_WARNING_DRAFT` as the **C-draft**
   (rename is NOT required; its existing text already says "continuing without it
   is allowed" and is the high-risk default — the warning under `None`/C uses it).
   Add `PROMPT_SHIELD_AVAILABLE_DRAFT` (State B, "an authorized shield IS available
   in this deployment — wire it in"). State A has no draft (silent). The hash
   excludes the draft (`_prompt_shield_artifact_hash`, def L961; the hash dict it
   passes to `stable_hash(...)` at ~L995 includes only `review_kind`/`llm_node_id`/
   `upstream_chain`, never the draft), so adding a
   second draft does not invalidate stored/in-flight reviews.

5. **Repair-turn prose** (`composer/service.py:719-722`) selects the
   state-appropriate draft: it must accept EITHER the C-draft
   (`PROMPT_SHIELD_WARNING_DRAFT`) or the B-draft (`PROMPT_SHIELD_AVAILABLE_DRAFT`)
   as the staged requirement draft for `user_term=PROMPT_SHIELD_USER_TERM`.

This keeps `validate()` pure, keeps the always-on warning firing everywhere at the
safe default, surfaces the live State-B advisory at the wire-stage turn (Task 3),
and gives p4 a precise A/B/C surface where context exists.

---

### Task 1: Add the State-B draft constant + decouple State A + make the warning always-on

**Files:**
- Modify: `src/elspeth/web/interpretation_state.py` — add `PROMPT_SHIELD_AVAILABLE_DRAFT` after `PROMPT_SHIELD_WARNING_DRAFT` (`:34-40`); add `_llm_has_authorized_shield_upstream` after `_llm_consumes_untrusted_remote_content` (`:290-310`); rewrite `prompt_shield_recommendation_warning_pairs` (def `:240`).
- Test: `tests/unit/web/test_interpretation_state.py` (helpers at `:939` `_state_with_web_scrape_llm_pair`, `:1072` `_unshielded_review_llm_options`; existing tests `:505`, `:524`, `:1197`).

**Interfaces:**
- Consumes: `_producer_by_output_stream(nodes) -> dict[str, NodeSpec]` (`:265`); `_AUTHORIZED_PROMPT_SHIELD_PLUGINS` (`:53`); `_llm_has_shield_recommendation(node) -> bool` (`:313`); `PROMPT_SHIELD_WARNING_DRAFT` (`:34`); `NodeSpec`.
- Produces: `PROMPT_SHIELD_AVAILABLE_DRAFT: Final[str]` (State-B draft; consumed by Task 3 dict-refiner, Task 4 repair-turn prose, and p4); `_llm_has_authorized_shield_upstream(node: NodeSpec, producer_by_output_stream: Mapping[str, NodeSpec]) -> bool` (NEW, pure; consumed by Task 2's `prompt_shield_state_for_node`); `prompt_shield_recommendation_warning_pairs(state, *, shield_available: bool | None = None)` now fires for every unshielded LLM node (State B/C), silent only for State A.

- [ ] **Step 1: Add the State-B draft constant FIRST (so later steps import-resolve).**
  In `interpretation_state.py`, immediately after the closing `)` of
  `PROMPT_SHIELD_WARNING_DRAFT` (`:40`), add:

  ```python
  PROMPT_SHIELD_AVAILABLE_DRAFT: Final[str] = (
      "An authorized prompt-injection shield (azure_prompt_shield) IS available in "
      "this deployment. Wire it between the external-content fetch step and this LLM: "
      "untrusted remote text routed straight into the LLM is a prompt-injection "
      "exposure, and the shield is configured and ready to use. "
      "[user_term: prompt_injection_shield_recommendation]"
  )
  ```

  > Both drafts MUST end with the `[user_term: prompt_injection_shield_recommendation]`
  > trailer so they parse identically downstream. The hash domain
  > (`_prompt_shield_artifact_hash`, def L961; `stable_hash(...)` at ~L995 hashes
  > only `review_kind`/`llm_node_id`/`upstream_chain`) excludes the draft, so adding this
  > constant cannot invalidate any stored/in-flight review.

  Verify the module still imports:
  ```
  cd /home/john/elspeth && python -c "from elspeth.web.interpretation_state import PROMPT_SHIELD_AVAILABLE_DRAFT; print('ok')"
  ```
  Expected: `ok`.

- [ ] **Step 2: Write the failing test — a plain (no-upstream-producer) LLM node now warns.**
  Append to `tests/unit/web/test_interpretation_state.py` (after the existing
  prompt-shield tests, e.g. after `:529`):

  ```python
  def _state_with_plain_llm_only() -> CompositionState:
      """One llm node with NO upstream producer at all (no web_scrape, no shield)."""
      return CompositionState(
          source=None,
          nodes=(
              NodeSpec(
                  id="rate_node",
                  node_type="transform",
                  plugin="llm",
                  input="rows",
                  on_success="out",
                  on_error="stop",
                  options={
                      "provider": "openrouter",
                      "model": "anthropic/claude-sonnet-4.6",
                      "prompt_template": "Rate {{ row.text }} and return JSON.",
                      "temperature": 0,
                  },
                  condition=None,
                  routes=None,
                  fork_to=None,
                  branches=None,
                  policy=None,
                  merge=None,
              ),
          ),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )


  def test_plain_unshielded_llm_warns_always_on() -> None:
      # Always-on: an llm node with no upstream producer and no shield still
      # surfaces the advisory (State C default). The pre-change code returned () here.
      state = _state_with_plain_llm_only()
      warning_pairs = prompt_shield_recommendation_warning_pairs(state)
      assert warning_pairs
      assert any(component == "node:rate_node" for component, _message in warning_pairs)
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py::test_plain_unshielded_llm_warns_always_on -x -q
  ```
  Expected failure: `assert ()` is falsy — `AssertionError` (the old precondition `_llm_consumes_untrusted_remote_content` returns False for a producerless node, so `warning_pairs == ()`).

- [ ] **Step 3: Add the decoupled State-A helper.**
  In `src/elspeth/web/interpretation_state.py`, immediately after
  `_llm_consumes_untrusted_remote_content` (its closing `return False` at `:310`),
  add:

  ```python
  def _llm_has_authorized_shield_upstream(
      node: NodeSpec,
      producer_by_output_stream: Mapping[str, NodeSpec],
  ) -> bool:
      """Walk upstream from ``node``; return True iff an authorized prompt-injection
      shield is reachable on the input chain.

      This is the always-on **State A** detector: it is judged on its own, NOT
      coupled to first reaching an untrusted producer (the deliberate decoupling
      from :func:`_llm_consumes_untrusted_remote_content`). A shielded LLM stays
      silent regardless of what produces its input.
      """

      if node.plugin != "llm":
          return False
      stream = node.input
      visited: set[str] = set()
      while stream and stream not in visited:
          visited.add(stream)
          if stream not in producer_by_output_stream:
              return False
          producer = producer_by_output_stream[stream]
          if producer.plugin in _AUTHORIZED_PROMPT_SHIELD_PLUGINS:
              return True
          stream = producer.input
      return False
  ```

- [ ] **Step 4: Make `prompt_shield_recommendation_warning_pairs` always-on (drop the producer precondition).**
  Replace the loop body of `prompt_shield_recommendation_warning_pairs` (`:240-262`).
  `PROMPT_SHIELD_AVAILABLE_DRAFT` already exists (Step 1), so this imports cleanly.
  Old:
  ```python
  def prompt_shield_recommendation_warning_pairs(state: CompositionState) -> tuple[tuple[str, str], ...]:
      """Return advisory warnings for unshielded untrusted content entering an LLM."""

      producer_by_output_stream = _producer_by_output_stream(state.nodes)
      warnings: list[tuple[str, str]] = []
      for node in state.nodes:
          if node.plugin != "llm":
              continue
          if not _llm_consumes_untrusted_remote_content(node, producer_by_output_stream):
              continue
          if _llm_has_shield_recommendation(node):
              continue
          warnings.append(
              (
                  f"node:{node.id}",
                  (
                      f"LLM node {node.id!r} consumes externally-fetched content from a web_scrape upstream "
                      "without an authorized prompt-injection shield between them. "
                      f"{PROMPT_SHIELD_WARNING_DRAFT}"
                  ),
              )
          )
      return tuple(warnings)
  ```
  New (note: signature gains `shield_available`; message text now conditional on
  whether an untrusted producer is present, but firing no longer depends on it):
  ```python
  def prompt_shield_recommendation_warning_pairs(
      state: CompositionState,
      *,
      shield_available: bool | None = None,
  ) -> tuple[tuple[str, str], ...]:
      """Return always-on advisory warnings for unshielded LLM nodes.

      The review is now ALWAYS-ON per LLM node, decoupled from whether an
      untrusted remote producer (web_scrape) is upstream:

      - **State A** (an authorized shield is reachable upstream) — silent, no warning.
      - **State B** (``shield_available is True``) — an authorized shield IS
        configured for this deployment; strong "wire it in" advisory.
      - **State C** (``shield_available`` is ``False`` or ``None``) — no shield
        available, or availability is undeterminable; high-risk "reconsider"
        advisory. ``None`` is the FAIL-SAFE default because the pure
        :meth:`CompositionState.validate` caller has no deployment/secret context
        to distinguish B from C.

      Advisory only — the caller appends these at "medium" severity into the
      ``warnings`` list and they are excluded from the blocking contract.
      """

      producer_by_output_stream = _producer_by_output_stream(state.nodes)
      warnings: list[tuple[str, str]] = []
      for node in state.nodes:
          if node.plugin != "llm":
              continue
          if _llm_has_authorized_shield_upstream(node, producer_by_output_stream):
              continue  # State A — already shielded, silent
          if _llm_has_shield_recommendation(node):
              continue  # review already staged on this node
          draft = PROMPT_SHIELD_AVAILABLE_DRAFT if shield_available is True else PROMPT_SHIELD_WARNING_DRAFT
          consumes_untrusted = _llm_consumes_untrusted_remote_content(node, producer_by_output_stream)
          lead = (
              f"LLM node {node.id!r} consumes externally-fetched content from a web_scrape upstream "
              "without an authorized prompt-injection shield between them. "
              if consumes_untrusted
              else f"LLM node {node.id!r} has no authorized prompt-injection shield in front of it. "
          )
          warnings.append((f"node:{node.id}", f"{lead}{draft}"))
      return tuple(warnings)
  ```
  Run to pass (Task 1's test):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py::test_plain_unshielded_llm_warns_always_on -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 5: Add the B-vs-C draft-selection test (proves the parameter works).**
  Append to `tests/unit/web/test_interpretation_state.py`:

  ```python
  def test_prompt_shield_warning_uses_available_draft_in_state_b() -> None:
      from elspeth.web.interpretation_state import PROMPT_SHIELD_AVAILABLE_DRAFT

      state = _state_with_plain_llm_only()
      pairs_b = prompt_shield_recommendation_warning_pairs(state, shield_available=True)
      assert pairs_b
      assert any(PROMPT_SHIELD_AVAILABLE_DRAFT in message for _component, message in pairs_b)

      pairs_c = prompt_shield_recommendation_warning_pairs(state, shield_available=False)
      assert pairs_c
      assert any("continuing without it is allowed" in message for _component, message in pairs_c)
  ```
  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py::test_prompt_shield_warning_uses_available_draft_in_state_b -x -q
  ```
  Expected: `1 passed`.
  > NOTE: this proves the *parameter* selects the B-draft. It does NOT prove a real
  > guided user ever reaches State B — that wiring is Task 4 (the production seam),
  > verified separately there. Do not treat this passing test as evidence the
  > 3-state surface is live end to end.

- [ ] **Step 6: Update the two tests whose firing precondition changed.**
  `test_gate_routed_web_scrape_into_llm_warns_without_prompt_shield` (`:505`) still
  passes (web_scrape upstream still warns) — but its comment now under-describes
  the behavior. Leave its assertions; they remain TRUE.
  `test_gate_routed_web_scrape_through_prompt_shield_emits_no_warning` (`:524`) is
  State A and MUST stay `() == ()` — verify it still passes (the new
  `_llm_has_authorized_shield_upstream` short-circuits it).
  Run the existing prompt-shield cluster to confirm no regression:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py -k "prompt_shield or web_scrape_into_llm or web_scrape_through_prompt_shield or plain_unshielded" -q
  ```
  Expected: all pass (the new always-on test + the State-A silence test + the
  hash-stability tests). If `test_gate_routed_web_scrape_through_prompt_shield_emits_no_warning`
  fails, the State-A walk is the defect — fix `_llm_has_authorized_shield_upstream`,
  not the test.

- [ ] **Step 7: Run the advisory-not-blocking test + the full module.**
  `test_prompt_shield_warning_is_advisory_not_blocking` (`:1197`) asserts the
  warning text contains `prompt_injection_shield_recommendation` AND
  `continuing without it is allowed`, then that `materialize_state_for_execution`
  returns a `CompositionState`. Under always-on, `state.validate()` still calls the
  function parameterless ⇒ State C ⇒ `PROMPT_SHIELD_WARNING_DRAFT` ⇒ both substrings
  present, so it STAYS GREEN with no edit. Run the whole module (the hash-stability
  tests at `:1105`, `:1127`, `:1145`, `:1166` are unaffected — draft excluded from
  the hash):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py -q
  ```
  Expected: all pass. If `test_prompt_shield_warning_is_advisory_not_blocking`
  fails, the `validate()` default is the defect (must be C) — fix Step 4, not the test.

- [ ] **Step 8: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/interpretation_state.py tests/unit/web/test_interpretation_state.py && git commit -m "$(cat <<'EOF'
feat(interpretation): make prompt-shield review always-on (State A decoupled)

prompt_shield_recommendation_warning_pairs now fires for every unshielded
LLM node, not only when an untrusted web_scrape producer is upstream. Add
_llm_has_authorized_shield_upstream (State A) judged on its own, decoupled
from _llm_consumes_untrusted_remote_content. The function gains an optional
shield_available kwarg selecting the State B vs C draft; the pure validate()
caller passes nothing -> fail-safe State C (always-on). Add PROMPT_SHIELD_AVAILABLE_DRAFT
(State B). Advisory only; never blocks.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 2: Add the context-aware A/B/C state helper + B-vs-C availability resolver

**Files:**
- Modify: `src/elspeth/web/interpretation_state.py` — add `prompt_shield_state_for_node` after `_llm_has_authorized_shield_upstream` (Task 1).
- Create: `src/elspeth/web/composer/tools/_shield_availability.py` — `azure_prompt_shield_available(context: ToolContext) -> bool`.
- Test: `tests/unit/web/test_interpretation_state.py`; `tests/unit/web/composer/tools/test_shield_availability.py` (new — the `tests/unit/web/composer/tools/` package directory + its `__init__.py` do NOT exist yet and must be created first, Step 3).

**Interfaces:**
- Consumes: `_llm_has_authorized_shield_upstream` (Task 1); `_producer_by_output_stream` (`:265`); `secret_unavailable_message(*, plugin_type, plugin_name, requirements, context) -> str | None` (`composer/tools/_availability.py:43`); `ToolContext` (`composer/tools/_common.py:1640`, fields `secret_service: WebSecretResolver | None`, `user_id: str | None`); the `azure_prompt_shield` catalog schema (`name="azure_prompt_shield"`, declares `discovery_secret_requirements={"api_key": ("AZURE_CONTENT_SAFETY_KEY",)}`).
- Produces: `prompt_shield_state_for_node(node: NodeSpec, all_nodes: Sequence[NodeSpec], *, shield_available: bool) -> str` returning `"A"`/`"B"`/`"C"` (consumed by p4); `azure_prompt_shield_available(context: ToolContext) -> bool` (the B-vs-C resolver; consumed by Task 3's route choke, p4, and any context-holding caller).

- [ ] **Step 1: Write the failing test for `prompt_shield_state_for_node`.**
  Append to `tests/unit/web/test_interpretation_state.py`:

  ```python
  def test_prompt_shield_state_for_node_returns_A_when_shielded() -> None:
      from elspeth.web.interpretation_state import prompt_shield_state_for_node

      state = _state_with_web_scrape_gate_shield_to_llm()
      llm = next(n for n in state.nodes if n.plugin == "llm")
      assert prompt_shield_state_for_node(llm, state.nodes, shield_available=True) == "A"
      assert prompt_shield_state_for_node(llm, state.nodes, shield_available=False) == "A"


  def test_prompt_shield_state_for_node_B_vs_C() -> None:
      from elspeth.web.interpretation_state import prompt_shield_state_for_node

      state = _state_with_plain_llm_only()
      llm = next(n for n in state.nodes if n.plugin == "llm")
      assert prompt_shield_state_for_node(llm, state.nodes, shield_available=True) == "B"
      assert prompt_shield_state_for_node(llm, state.nodes, shield_available=False) == "C"
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py -k prompt_shield_state_for_node -x -q
  ```
  Expected failure: `ImportError: cannot import name 'prompt_shield_state_for_node'`.

- [ ] **Step 2: Implement `prompt_shield_state_for_node`.**
  In `interpretation_state.py`, after `_llm_has_authorized_shield_upstream`
  (Task 1), add:

  ```python
  def prompt_shield_state_for_node(
      node: NodeSpec,
      all_nodes: Sequence[NodeSpec],
      *,
      shield_available: bool,
  ) -> str:
      """Return the prompt-shield review state for ``node``: ``"A"`` / ``"B"`` / ``"C"``.

      - ``"A"`` — an authorized shield is reachable upstream (silent; no advisory).
      - ``"B"`` — no upstream shield, but an authorized shield IS available in this
        deployment (``shield_available is True``): strong "wire it in" advisory.
      - ``"C"`` — no upstream shield and no shield available (``shield_available is
        False``): high-risk "reconsider" advisory.

      The caller resolves ``shield_available`` from deployment context (see
      :func:`elspeth.web.composer.tools._shield_availability.azure_prompt_shield_available`);
      the contract default when availability is undeterminable is ``False`` (State C,
      fail-safe).
      """

      if node.plugin != "llm":
          return "A"
      producer_by_output_stream = _producer_by_output_stream(all_nodes)
      if _llm_has_authorized_shield_upstream(node, producer_by_output_stream):
          return "A"
      return "B" if shield_available else "C"
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py -k prompt_shield_state_for_node -x -q
  ```
  Expected: `2 passed`.

- [ ] **Step 3: Create the test package directory FIRST (it does not exist).**
  The test tree uses package-style layout (`tests/unit/web/composer/__init__.py`
  exists), but `tests/unit/web/composer/tools/` does NOT exist. Without the directory
  AND its `__init__.py`, pytest raises an import-file-mismatch error and refuses to
  collect the new test — Step 4's TDD red cannot even run.
  ```
  cd /home/john/elspeth && mkdir -p tests/unit/web/composer/tools && touch tests/unit/web/composer/tools/__init__.py
  ```

- [ ] **Step 4: Write the failing test for `azure_prompt_shield_available`.**
  Create `tests/unit/web/composer/tools/test_shield_availability.py`:

  ```python
  from __future__ import annotations

  from elspeth.web.composer.tools._common import ToolContext
  from elspeth.web.composer.tools._shield_availability import azure_prompt_shield_available


  class _FakeSecretService:
      def __init__(self, refs: set[str]) -> None:
          self._refs = refs

      def has_ref(self, user_id: str, name: str) -> bool:
          return name in self._refs


  def _ctx(secret_service, user_id: str | None) -> ToolContext:
      return ToolContext(secret_service=secret_service, user_id=user_id)


  def test_shield_available_when_key_configured() -> None:
      ctx = _ctx(_FakeSecretService({"AZURE_CONTENT_SAFETY_KEY"}), "alice")
      assert azure_prompt_shield_available(ctx) is True


  def test_shield_unavailable_when_key_missing() -> None:
      ctx = _ctx(_FakeSecretService(set()), "alice")
      assert azure_prompt_shield_available(ctx) is False


  def test_shield_undeterminable_defaults_to_false() -> None:
      # No secret service / no user => cannot determine => fail-safe State C.
      assert azure_prompt_shield_available(_ctx(None, "alice")) is False
      assert azure_prompt_shield_available(_ctx(_FakeSecretService({"AZURE_CONTENT_SAFETY_KEY"}), None)) is False
  ```

  > `ToolContext` is a dataclass with `secret_service`/`user_id` defaulting to
  > `None` (`_common.py:1704-1705`); construct it with only those two kwargs. If
  > `ToolContext` has other required fields, build it via its existing test factory
  > — grep `tests/unit/web/composer/tools/` for `ToolContext(` to find the
  > minimal constructor used by sibling tests and mirror it.

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/tools/test_shield_availability.py -x -q
  ```
  Expected failure: `ModuleNotFoundError: No module named 'elspeth.web.composer.tools._shield_availability'`.

- [ ] **Step 5: Implement `azure_prompt_shield_available`.**
  Create `src/elspeth/web/composer/tools/_shield_availability.py`:

  ```python
  """Deployment-availability of the authorized prompt-injection shield (B-vs-C signal)."""

  from __future__ import annotations

  from elspeth.web.catalog.schemas import PluginSecretRequirement
  from elspeth.web.composer.tools._availability import secret_unavailable_message
  from elspeth.web.composer.tools._common import ToolContext

  # The authorized prompt-injection shield and the secret ref that gates its use.
  # Mirrors azure_prompt_shield's discovery_secret_requirements
  # ({"api_key": ("AZURE_CONTENT_SAFETY_KEY",)}); the catalog promotes that to a
  # PluginSecretRequirement(field="api_key", candidates=("AZURE_CONTENT_SAFETY_KEY",)).
  _AZURE_PROMPT_SHIELD_NAME = "azure_prompt_shield"
  _AZURE_PROMPT_SHIELD_REQUIREMENT = PluginSecretRequirement(
      field="api_key",
      candidates=("AZURE_CONTENT_SAFETY_KEY",),
  )


  def azure_prompt_shield_available(context: ToolContext) -> bool:
      """Return True iff an authorized prompt-injection shield is usable here.

      Reuses the existing discovery surface
      (:func:`elspeth.web.composer.tools._availability.secret_unavailable_message`)
      keyed on the shield-specific ``AZURE_CONTENT_SAFETY_KEY`` candidate (NOT a
      coarse any-secret check). Returns the FAIL-SAFE ``False`` (State C) when
      availability is undeterminable — no secret service or no user. Because the
      requirement carries non-empty ``candidates``, ``_requirement_satisfied``
      takes the candidate branch (``has_ref(user_id, "AZURE_CONTENT_SAFETY_KEY")``).
      """

      if context.secret_service is None or context.user_id is None:
          return False
      message = secret_unavailable_message(
          plugin_type="transform",
          plugin_name=_AZURE_PROMPT_SHIELD_NAME,
          requirements=(_AZURE_PROMPT_SHIELD_REQUIREMENT,),
          context=context,
      )
      return message is None
  ```

  > **`plugin_type` is a bare string literal, NOT an enum member.** `PluginKind` is
  > `Literal["source", "transform", "sink"]` (`catalog/schemas.py:18`) — a typing
  > type alias, NOT an `Enum`. It has NO `.TRANSFORM` attribute and is not iterable
  > (`list(PluginKind)` raises). Pass the plain string `"transform"`; every existing
  > caller of `secret_unavailable_message` passes a string-typed value.
  > `secret_unavailable_message` uses `plugin_type` only for the message string, not
  > the satisfaction logic — but `"transform"` is the accurate value for the shield.
  > `PluginKind` is therefore NOT imported (only `PluginSecretRequirement` is); do not
  > add a `PluginKind` import or ruff F401 (Task 6 Step 3) trips.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/tools/test_shield_availability.py -x -q
  ```
  Expected: `3 passed`.

- [ ] **Step 6: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/interpretation_state.py src/elspeth/web/composer/tools/_shield_availability.py tests/unit/web/test_interpretation_state.py tests/unit/web/composer/tools/__init__.py tests/unit/web/composer/tools/test_shield_availability.py && git commit -m "$(cat <<'EOF'
feat(interpretation): add A/B/C shield-state helper + availability resolver

prompt_shield_state_for_node returns A (shielded upstream) / B (shield
available, not wired) / C (no shield) for a given LLM node. azure_prompt_shield_available
computes the B-vs-C deployment signal by reusing secret_unavailable_message
keyed on AZURE_CONTENT_SAFETY_KEY, defaulting to fail-safe False (State C)
when undeterminable. Consumed by the tutorial worked example (p4).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 3: Wire B-vs-C refinement into the user-facing wire-stage warning surface (the real plumbing)

> **Why this task exists (do NOT skip it).** Tasks 1-2 build the `shield_available`
> flag and the `azure_prompt_shield_available` resolver, but NOTHING in production
> passes `True`: the pure `CompositionState.validate()` calls the warning function
> parameterless ⇒ always State C. Without this task, the State-B advisory is DEAD
> in production, and a guided user who HAS `AZURE_CONTENT_SAFETY_KEY` configured is
> told "no shield is available" (factually wrong). The contract (§2.3) names
> threading availability into the warning surface as "the real new plumbing p3
> owns."
>
> **The seam MUST be the route HANDLER, not any single wire-turn builder.** The
> `confirm_wiring` turn is built from MANY `build_step_4_wire_turn` call sites: the
> step→turn RE-RENDER helper (`guided.py:210`) AND the ADVANCE/dispatch machinery
> in `_helpers.py` (`:2682, :3892, :3940, :4023, :4098`). A guided user reaches the
> wire stage by ADVANCING (recipe-apply / chain-accept → STEP_4_WIRE), whose
> `next_turn` comes from the `_helpers.py` dispatch path (via `_dispatch_guided_respond`
> at `guided.py:1358`), NOT from `guided.py:210`. Parametrizing one builder would
> leave State B dead on the path the user actually traverses — the same dead-B bug
> one level up. So this task intercepts ONCE at each user-facing route handler,
> AFTER `next_turn` is computed: if it is a `confirm_wiring` turn, refine its
> `payload["warnings"]` with the request-scoped `shield_available`. One choke per
> handler, context guaranteed, independent of which builder produced the turn.
> `validate()` stays pure; its always-on State-C default is the fail-safe.
>
> **The OTHER context-bearing wire surface needs NO change (verified).**
> `_helpers.py:560` projects `live_validation.warnings` into
> `CompositionStateResponse.validation_warnings` — a separate context-bearing
> surface. The frontend renders the prompt-shield advisory ONLY from the wire
> turn's `payload.warnings` (`WireStageTurn.tsx:110`, `data.warnings`); NO
> component or store reads shield text from `validation_warnings` (verified: zero
> references to `validation_warnings`/`validationWarnings` under
> `frontend/src/components` or `frontend/src/stores`). So `validation_warnings`
> carries shield text at State C but the user never sees it through that channel —
> leave it unrefined. Do NOT add the refinement at `_helpers.py:560`.

**Files:**
- Modify: `src/elspeth/web/interpretation_state.py` — add the pure post-processor `refine_prompt_shield_warnings_for_availability` (operates on the `next_turn` warnings-dict list, the already-`to_dict`'d shape) after `prompt_shield_state_for_node` (Task 2).
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py` — the wire-turn re-render call site (`:210`) AND the route handlers that return `next_turn`: `post_guided_respond` (advance path, response built at `:1488-1528` and idempotent-replay at `:1985-1988`) and `get_guided` (re-render path). Compute `shield_available` once per request from a `ToolContext`.
- Test: `tests/unit/web/test_interpretation_state.py` (pure refiner); `tests/integration/web/composer/guided/` (route choke — locate the wire-stage integration test in Step 4).

**Interfaces:**
- Consumes: `PROMPT_SHIELD_AVAILABLE_DRAFT` / `PROMPT_SHIELD_WARNING_DRAFT` (Task 1); `azure_prompt_shield_available(context: ToolContext) -> bool` (Task 2); `ToolContext` (`composer/tools/_common.py:1640`); the `confirm_wiring` turn dict shape `{"type": TurnType.CONFIRM_WIRING.value, "step_index": int, "payload": {"warnings": [{"component","message","severity"}, ...], ...}}`; `TurnType.CONFIRM_WIRING` (`composer/guided/turns.py` or wherever `TurnType` is defined — confirm in Step 1); the route's `request` + `user` (held at `post_guided_respond` `:919`, `get_guided`).
- Produces: `refine_prompt_shield_warnings_for_availability(warnings: list[dict], *, shield_available: bool) -> list[dict]` (NEW, pure; operates on the already-`to_dict`'d warning dicts in a turn payload); a guided wire-stage `next_turn` whose shield warnings carry the B-draft when a shield IS configured, on BOTH the advance and re-render paths.

- [ ] **Step 1: Confirm the turn dict shape + `TurnType.CONFIRM_WIRING` + the route response sites.**
  ```
  cd /home/john/elspeth && grep -rn "CONFIRM_WIRING\|class TurnType" src/elspeth/web/composer/guided/ | head && sed -n '388,394p' src/elspeth/web/composer/guided/emitters.py && sed -n '1488,1528p' src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Confirm: (a) `TurnType.CONFIRM_WIRING.value` is the wire-turn type string; (b) the
  warnings dict shape is `{"component","message","severity"}` (from `ValidationEntry.to_dict()`);
  (c) `post_guided_respond` builds `next_turn` (a dict) before the
  `GuidedRespondResponse(... next_turn=TurnPayloadResponse(type=next_turn["type"], ...))`
  construction at `:1488-1528` (advance path) and `:1985-1988` (idempotent replay).
  Both read the SAME `next_turn` dict — the single choke is "after `next_turn` is
  finalized, before response construction."

- [ ] **Step 2: Write the failing test for the pure dict-level refiner.**
  Append to `tests/unit/web/test_interpretation_state.py`:

  ```python
  def test_refine_prompt_shield_warnings_rewrites_c_to_b_when_available() -> None:
      from elspeth.web.interpretation_state import (
          PROMPT_SHIELD_AVAILABLE_DRAFT,
          PROMPT_SHIELD_WARNING_DRAFT,
          refine_prompt_shield_warnings_for_availability,
      )

      # The already-to_dict'd warning shape carried on a turn payload.
      c_warnings = [
          {"component": "node:rate_node", "message": f"lead {PROMPT_SHIELD_WARNING_DRAFT}", "severity": "medium"},
          {"component": "node:other", "message": "unrelated warning", "severity": "medium"},
      ]

      refined = refine_prompt_shield_warnings_for_availability(c_warnings, shield_available=True)
      shield = [w for w in refined if w["component"] == "node:rate_node"]
      assert shield
      assert PROMPT_SHIELD_AVAILABLE_DRAFT in shield[0]["message"]
      assert PROMPT_SHIELD_WARNING_DRAFT not in shield[0]["message"]
      # Non-shield warning untouched.
      other = [w for w in refined if w["component"] == "node:other"]
      assert other[0]["message"] == "unrelated warning"

      # shield_available=False ⇒ no rewrite (State C / nothing to upgrade).
      unchanged = refine_prompt_shield_warnings_for_availability(c_warnings, shield_available=False)
      assert any(PROMPT_SHIELD_WARNING_DRAFT in w["message"] for w in unchanged)
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py::test_refine_prompt_shield_warnings_rewrites_c_to_b_when_available -x -q
  ```
  Expected failure: `ImportError: cannot import name 'refine_prompt_shield_warnings_for_availability'`.

- [ ] **Step 3: Implement the pure dict-level refiner.**
  In `interpretation_state.py`, after `prompt_shield_state_for_node` (Task 2), add:

  ```python
  def refine_prompt_shield_warnings_for_availability(
      warnings: Sequence[Mapping[str, Any]],
      *,
      shield_available: bool,
  ) -> list[dict[str, Any]]:
      """Rewrite always-on State-C shield warnings to State-B at a context-bearing
      caller, operating on the already-``to_dict``'d warning shape carried on a
      ``confirm_wiring`` turn payload (``{"component","message","severity"}``).

      ``CompositionState.validate()`` is pure and emits the fail-safe State-C draft
      (``PROMPT_SHIELD_WARNING_DRAFT``) for every unshielded LLM node (always-on).
      The route handler — which HOLDS the request's secret context — post-processes
      the turn payload: each warning whose message contains
      ``PROMPT_SHIELD_WARNING_DRAFT`` is re-emitted with
      ``PROMPT_SHIELD_AVAILABLE_DRAFT`` when ``shield_available`` is True. Non-shield
      warnings and the ``shield_available is False`` case (nothing to upgrade) pass
      through unchanged. Pure; never mutates the input; severity preserved.
      """

      result: list[dict[str, Any]] = [dict(entry) for entry in warnings]
      if not shield_available:
          return result
      for entry in result:
          message = entry.get("message")
          if isinstance(message, str) and PROMPT_SHIELD_WARNING_DRAFT in message:
              entry["message"] = message.replace(PROMPT_SHIELD_WARNING_DRAFT, PROMPT_SHIELD_AVAILABLE_DRAFT)
      return result
  ```

  > `Mapping` / `Sequence` / `Any` imports: confirm with
  > `grep -n "from collections.abc import\|from typing import" src/elspeth/web/interpretation_state.py`.
  > The file already uses `Mapping`, `Sequence`, `Final` — add `Any` if absent.
  > Operating on dicts (not `ValidationEntry`) AVOIDS importing `composer.state`
  > into `interpretation_state.py` (composer.state imports interpretation_state at
  > `:2418` — a `ValidationEntry` import here would risk a cycle). Verify no cycle:
  > `python -c "import elspeth.web.interpretation_state; print('ok')"`.
  >
  > **Provenance the refiner depends on (verified — makes Step 5's integration
  > assertion testable).** The `confirm_wiring` turn's `payload["warnings"]` is the
  > already-`to_dict`'d projection of `validate().warnings`: `build_step_4_wire_turn`
  > (`composer/guided/emitters.py`) calls `validation = state.validate()` (L388) and
  > sets `payload["warnings"] = [w.to_dict() for w in validation.warnings]` (L393).
  > `validate()` (`state.py:2414-2421`) appends each `prompt_shield_recommendation_warning_pairs`
  > pair (always-on State-C default) into `warnings` at "medium". So the State-C
  > shield draft IS present in the turn payload when the route refiner runs — that is
  > what `refine_prompt_shield_warnings_for_availability` rewrites to State B.

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py::test_refine_prompt_shield_warnings_rewrites_c_to_b_when_available -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 4: Locate the wire-stage integration test that drives a guided session to STEP_4_WIRE.**
  ```
  cd /home/john/elspeth && grep -rln "confirm_wiring\|STEP_4_WIRE\|chosen=\[.accept.\]\|propose_chain" tests/integration/web/composer/guided/
  ```
  Identify the test that ADVANCES to the wire stage (recipe-apply / chain-accept →
  STEP_4_WIRE) and reads `next_turn`. Note its `composer_test_client` fixture and
  whether the fixture wires a `secret_service` with `AZURE_CONTENT_SAFETY_KEY` (it
  almost certainly does NOT — so the default path is State C, which is what the
  existing assertions expect).

- [ ] **Step 5: Write the failing integration test that the ADVANCE path surfaces State B when the key is configured.**
  Create `tests/integration/web/composer/guided/test_wire_turn_shield_state.py`.
  Drive a guided session to the wire stage (reuse the sibling test's drive helper —
  e.g. `_drive_to_step_3_propose_chain` then `chosen=["accept"]`), but with the
  fixture's `scoped_secret_resolver` returning
  `has_ref(user, "AZURE_CONTENT_SAFETY_KEY") -> True`. Assert the returned wire
  turn's warnings carry the B-draft:

  > **PRECONDITION — the driven chain MUST contain an unshielded `llm` node, or the
  > test sets up no shield warning and its own assertion is vacuously unsatisfiable.**
  > Before relying on `_drive_to_step_3_propose_chain`, INSPECT the chain it proposes:
  > `test_step_3_e2e.py` proposes a `passthrough`-only chain (NO llm node) — copying
  > that helper verbatim would drive a chain with zero shield warnings, and
  > `assert PROMPT_SHIELD_AVAILABLE_DRAFT in messages` would fail for the wrong reason
  > (no warning to refine, not "refinement is broken"). Either reuse a helper whose
  > proposal contains an unshielded llm node (mirror
  > `test_guided_commit_surfaces_reviews.py`, which stages an llm node), or pass an
  > explicit intent string that guarantees an llm node (e.g. a "rate the pages"
  > intent that proposes an `llm` transform). Confirm the proposed chain has an
  > `llm` node — grep the helper's expected proposal / assert on it — before writing
  > the shield assertions.

  ```python
  """Phase p3 Task 3 — the ADVANCE path surfaces State B when the shield key is configured."""

  from __future__ import annotations

  from elspeth.web.interpretation_state import (
      PROMPT_SHIELD_AVAILABLE_DRAFT,
      PROMPT_SHIELD_WARNING_DRAFT,
  )

  # Reuse the sibling drive helpers (mirror test_guided_commit_surfaces_reviews.py /
  # test_step_3_e2e.py): _create_session, _drive_to_step_3_propose_chain (proposing an
  # llm node so the wire stage has an unshielded LLM), _respond.


  def test_advance_to_wire_surfaces_state_b_with_key(composer_test_client_with_shield_key) -> None:
      client = composer_test_client_with_shield_key
      session_id = _create_session(client)
      _drive_to_step_3_propose_chain(client, session_id)  # llm node, no shield
      body = _respond(client, session_id, chosen=["accept"])  # advance -> STEP_4_WIRE
      assert body["next_turn"]["type"] == "confirm_wiring"
      messages = " ".join(w["message"] for w in body["next_turn"]["payload"]["warnings"])
      assert PROMPT_SHIELD_AVAILABLE_DRAFT in messages  # State B (key configured)
      assert PROMPT_SHIELD_WARNING_DRAFT not in messages


  def test_get_guided_reload_at_wire_surfaces_state_b_with_key(composer_test_client_with_shield_key) -> None:
      # The GET re-render path (page reload while sitting at STEP_4_WIRE) must ALSO
      # surface State B — covering the :476/:707/:906 GET TurnPayloadResponse sites,
      # not just the advance path.
      client = composer_test_client_with_shield_key
      session_id = _create_session(client)
      _drive_to_step_3_propose_chain(client, session_id)
      _respond(client, session_id, chosen=["accept"])  # land at STEP_4_WIRE
      reloaded = _get_guided(client, session_id)  # GET re-render
      assert reloaded["next_turn"]["type"] == "confirm_wiring"
      messages = " ".join(w["message"] for w in reloaded["next_turn"]["payload"]["warnings"])
      assert PROMPT_SHIELD_AVAILABLE_DRAFT in messages
      assert PROMPT_SHIELD_WARNING_DRAFT not in messages
  ```
  > `_get_guided(client, session_id)` — the `GET /api/sessions/{id}/guided` helper
  > (mirror the sibling tests). If `get_guided` returns the wire turn under a
  > different response key than `next_turn`, adapt the path to that key.

  > `composer_test_client_with_shield_key` — add a DEDICATED fixture (mirror the
  > base `composer_test_client` in `tests/integration/web/composer/guided/conftest.py`)
  > that sets **`app.state.scoped_secret_resolver`** (NOT `app.state.secret_service`)
  > to a stub whose 2-arg `has_ref(self, user_id, name) -> bool` returns True for
  > `"AZURE_CONTENT_SAFETY_KEY"` (and `user_id` to the mocked "alice"). The 2-arg
  > stub signature matches the `WebSecretResolver` / `ScopedSecretResolver` protocol
  > the production route (Step 7) reads from `scoped_secret_resolver` — setting
  > `secret_service` instead would be a FALSE GREEN: the test would pass against an
  > attribute the production route does not read, masking the TypeError that the real
  > 3-arg `WebSecretService` raises. Do NOT mutate the base fixture — that would shift
  > State-C assertions in the ~11 sibling tests to State B (W2 blast-radius lesson from
  > the P3 interp-surfacing plan). A dedicated fixture bounds the blast radius to this
  > one test. Reuse the sibling drive helpers by importing or copying them.
  >
  > If the base fixture already wires a real `scoped_secret_resolver` with no refs,
  > the stub only needs to add the one ref. The production attribute is
  > `scoped_secret_resolver` (Task 3 Step 6).

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_wire_turn_shield_state.py -x -q
  ```
  Expected failure: `assert PROMPT_SHIELD_AVAILABLE_DRAFT in messages` — the advance
  path still emits State C (the bug this task fixes).

- [ ] **Step 6: Read the route response sites + secret-service handle + enumerate ALL `TurnPayloadResponse(` sites.**
  ```
  cd /home/john/elspeth && grep -n "scoped_secret_resolver\|secret_service\|app.state\.\|TurnPayloadResponse(" src/elspeth/web/sessions/routes/composer/guided.py && sed -n '1351,1360p' src/elspeth/web/sessions/routes/composer/guided.py
  ```
  > **The canonical secret-availability collaborator is `app.state.scoped_secret_resolver`
  > (a `ScopedSecretResolver`, 2-arg `has_ref(user_id, name)`), NEVER
  > `app.state.secret_service` (a `WebSecretService`, 3-arg keyword-only
  > `has_ref(user_id, name, *, auth_provider_type)` → `TypeError` when called with 2
  > positional args at `_availability.py:124`). app.py:354 documents this; app.py:346/
  > 360/908 all pass `scoped_secret_resolver` as the `secret_service=` ToolContext
  > field. Step 7's `_resolve_shield_available` reads `scoped_secret_resolver`.**
  Confirm: the `app.state` secret-resolver attribute name (`scoped_secret_resolver`); the FULL list of
  `TurnPayloadResponse(` construction sites (expected five: `:476, :707, :906,
  :1522, :1985`); that `next_turn` is the dict from `_dispatch_guided_respond`
  (`:1358`) and is also set on the STEP_4_WIRE/idempotent paths; and that `request`
  + `user` are in scope at each handler (`post_guided_respond` `:919`, `get_guided`,
  and any other handler holding a `TurnPayloadResponse` site).

- [ ] **Step 7: Add a DRY `_turn_payload_response` wrapper and replace EVERY `TurnPayloadResponse(...)` site with it.**
  > **Shape fact (verified):** `Turn` is a `TypedDict` (`composer/guided/protocol.py:137`,
  > keys `type`/`step_index`/`payload`), so a `Turn` is a plain `dict` at runtime on
  > BOTH paths — POST `next_turn` (a dict from `_dispatch_guided_respond`) AND GET
  > `turn` (`build_step_4_wire_turn` returns `Turn(...)`, also a dict). Dict ops
  > (`turn["type"]`, `.get(...)`) are correct on both; there is NO `Turn`-object vs
  > dict mismatch. The route already subscripts both (`:477 turn["type"]`,
  > `:1523 next_turn["type"]`).
  >
  > **Coverage fact (verified):** there are FIVE `TurnPayloadResponse(` construction
  > sites in `guided.py` — `:476, :707, :906` (GET / re-render handlers) and
  > `:1522, :1985` (POST advance + idempotent-replay). "Place once + get_guided"
  > would miss sites. A single DRY wrapper applied at ALL FIVE cannot miss one (it
  > no-ops on every non-`confirm_wiring` turn), and is the mechanical, can't-skip
  > form the dead-B bug demands.
  >
  > **No-op-today note (architecture):** the `post_guided_reenter` (`:592`) and
  > `post_guided_start` (`:718`) handlers — feeding the GET-style sites `:476`/`:707`
  > (and `:906`) — structurally never emit a `confirm_wiring` turn today (STEP_4_WIRE
  > is reached only via recipe-apply / chain-accept through `post_guided_respond`).
  > The wrapper no-ops there safely; wrapping them adds no current behavior but is
  > the can't-skip form (a future caller that DOES emit a wire turn through those
  > sites is covered automatically). Leave a one-line `# no-op for non-confirm_wiring
  > turns today; covers future wire-turn emission` comment at those sites so a future
  > reader does not assume the refiner fires there now.

  In `src/elspeth/web/sessions/routes/composer/guided.py`, add module-level helpers
  (TurnType is ALREADY imported at `:48`; do NOT re-import it):
  ```python
  from elspeth.web.composer.tools._common import ToolContext
  from elspeth.web.composer.tools._shield_availability import azure_prompt_shield_available
  from elspeth.web.interpretation_state import refine_prompt_shield_warnings_for_availability


  def _resolve_shield_available(request: Request, user_id: str) -> bool:
      """Resolve whether an authorized prompt-injection shield is configured for
      this request's user. Fail-safe False (State C) when no secret context.

      The secret resolver MUST come from ``request.app.state.scoped_secret_resolver``
      (a ``ScopedSecretResolver`` whose ``has_ref(user_id, name)`` is the 2-arg
      ``WebSecretResolver`` protocol that the downstream `_requirement_satisfied`
      at `_availability.py:124` calls — `has_ref(context.user_id, name)`, 2 positional
      args). It is NOT ``app.state.secret_service`` (a ``WebSecretService`` whose
      ``has_ref(user_id, name, *, auth_provider_type)`` is keyword-only-3-arg —
      calling it with 2 positional args raises ``TypeError`` on EVERY wire turn in
      any deployment where the Azure key is reachable). app.py:354 documents
      ``scoped_secret_resolver`` as the correct collaborator; every sibling route
      (proposals.py, compose.py, state.py, messages.py) reads it. The ToolContext
      field is still named ``secret_service`` (`WebSecretResolver | None`) — only the
      ``app.state`` attribute that feeds it changes.
      """
      return azure_prompt_shield_available(
          ToolContext(
              secret_service=getattr(request.app.state, "scoped_secret_resolver", None),
              user_id=user_id,
          )
      )


  def _turn_payload_response(
      turn: Mapping[str, Any] | None,
      *,
      shield_available: bool,
  ) -> TurnPayloadResponse | None:
      """Build the wire ``TurnPayloadResponse`` from a turn dict, refining a
      ``confirm_wiring`` turn's prompt-shield warnings to State B when a shield is
      available. No-op on every other turn type. Single DRY choke replacing every
      raw ``TurnPayloadResponse(...)`` so the State-B advisory is reached on the
      advance path, the idempotent-replay path, AND the GET re-render path.

      ``turn`` is a ``Turn`` TypedDict (plain dict at runtime) on every path.
      """
      if turn is None:
          return None
      payload = dict(turn["payload"])
      if turn["type"] == TurnType.CONFIRM_WIRING.value:
          payload["warnings"] = refine_prompt_shield_warnings_for_availability(
              payload.get("warnings") or [],
              shield_available=shield_available,
          )
      return TurnPayloadResponse(
          type=turn["type"],
          step_index=turn["step_index"],
          payload=payload,
      )
  ```
  Then replace EVERY `TurnPayloadResponse(type=..., step_index=..., payload=dict(...))`
  construction (the five sites at `:476, :707, :906, :1522, :1985`) with the wrapper.
  At each site the local turn variable is `turn` (GET sites) or `next_turn` (POST
  sites). Compute `shield_available` once per handler (each handler holds `request`
  + `user`) and pass it:
  ```python
      shield_available = _resolve_shield_available(request, user.user_id)
      # ... in the response construction:
      next_turn=_turn_payload_response(next_turn, shield_available=shield_available),
      # or, at GET sites:
      next_turn=_turn_payload_response(turn, shield_available=shield_available),
  ```
  Find every site mechanically before editing:
  ```
  cd /home/john/elspeth && grep -n "TurnPayloadResponse(" src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Each replaced site drops the inline `if turn is not None else None` (the wrapper
  returns `None` for `None`). Confirm `TurnPayloadResponse` requires only
  `type`/`step_index`/`payload` (no other required field) — if it has more, preserve
  them at each site.

  > Do NOT parametrize `build_step_4_wire_turn` — the route wrapper covers every
  > builder and every construction site uniformly; one seam, guaranteed coverage.

- [ ] **Step 8: Run the failing integration test (now passes) + the full guided surface.**
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_wire_turn_shield_state.py tests/integration/web/composer/guided/ tests/unit/web/test_interpretation_state.py -q
  ```
  Expected: the new test passes (advance path now surfaces State B with the key),
  and the existing guided tests stay green (their fixtures have no Azure key ⇒
  State C ⇒ unchanged). If a sibling test that DOES configure the key now sees the
  B-draft, that is the feature landing visibly — UPDATE its assertion to the B-draft
  (test-to-UPDATE), do not revert.

- [ ] **Step 9: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/interpretation_state.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/test_interpretation_state.py tests/integration/web/composer/guided/ && git commit -m "$(cat <<'EOF'
feat(composer/guided): surface State-B prompt-shield advisory at the route

Wire the B-vs-C availability signal into the user-facing wire-stage warnings
via a DRY _turn_payload_response wrapper at EVERY TurnPayloadResponse site
(advance path, idempotent-replay path, AND the GET re-render / reload path),
so the State-B advisory is not dead on the path a user actually traverses to
STEP_4_WIRE. validate() stays pure and emits the always-on State-C default;
the route resolves azure_prompt_shield_available from the request's ToolContext
and refines the confirm_wiring turn's warnings via
refine_prompt_shield_warnings_for_availability, re-emitting the State-B
"shield is available, wire it in" draft per node. A user with
AZURE_CONTENT_SAFETY_KEY configured now sees State B instead of a
factually-wrong State C. No secret context -> fail-safe State C.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 4: Select the state-appropriate draft in the forced-repair-turn prose

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — repair-turn prose (`:719-722`); import block (`:131-139`).
- Test: `tests/unit/web/composer/test_prompts.py` OR the test that exercises the repair-turn text. Locate it first (Step 1).

**Interfaces:**
- Consumes: `PROMPT_SHIELD_USER_TERM` (`:134`), `PROMPT_SHIELD_WARNING_DRAFT` (`:135`); add `PROMPT_SHIELD_AVAILABLE_DRAFT` to the import (Task 1 produced it).
- Produces: repair-turn prose that names BOTH drafts as acceptable staged-requirement drafts for the shield user_term.

- [ ] **Step 1: Locate the test that asserts the repair-turn prose.**
  ```
  cd /home/john/elspeth && grep -rn "prompt_injection_shield_recommendation\|keep going with the warning\|forced repair turn" tests/unit/web/composer/ tests/unit/web/
  ```
  Note the test file + name that asserts the `:719-722` text. If none asserts the
  shield clause directly, the repair-turn prose is covered only by a broad
  prompt-snapshot test; in that case Step 3's edit will shift that snapshot and the
  snapshot test is the test-to-UPDATE.

- [ ] **Step 2: Read the exact current block to anchor the edit.**
  ```
  cd /home/john/elspeth && sed -n '714,724p' src/elspeth/web/composer/service.py
  ```
  Confirm the verbatim text matches the `old_string` in Step 3.

- [ ] **Step 3: Edit the repair-turn prose to name both drafts.**
  In `src/elspeth/web/composer/service.py`, replace (`:719-722`):
  ```python
          f"If user_term is {PROMPT_SHIELD_USER_TERM!r}, patch the target LLM node first "
          "with an interpretation_requirements entry whose kind is 'pipeline_decision', "
          f"status is 'pending', and draft is {PROMPT_SHIELD_WARNING_DRAFT!r}; if the "
          "workflow cannot add the shield, keep going with the warning instead of blocking. "
  ```
  with:
  ```python
          f"If user_term is {PROMPT_SHIELD_USER_TERM!r}, patch the target LLM node first "
          "with an interpretation_requirements entry whose kind is 'pipeline_decision' and "
          "status is 'pending'. Use the state-appropriate draft: if an authorized "
          f"prompt-injection shield is available in this deployment, draft is {PROMPT_SHIELD_AVAILABLE_DRAFT!r}; "
          f"otherwise (or if availability is unknown) draft is {PROMPT_SHIELD_WARNING_DRAFT!r}. If the "
          "workflow cannot add the shield, keep going with the warning instead of blocking. "
  ```
  > **Repair-budget interaction (architecture, non-blocking).** The repair loop is
  > bounded by `_MAX_REPAIR_TURNS = 2`; asking the LLM to resolve B-vs-C inside the
  > repair turn could consume budget. This is SAFE because the prose defaults to the
  > C-draft when availability is unknown (the fail-safe), AND the live B-vs-C
  > refinement is already done deterministically at the wire-stage route (Task 3) —
  > the repair turn does not need to compute availability to be correct. If a future
  > change tightens the budget, default the repair case to the C-draft outright.

- [ ] **Step 4: Add `PROMPT_SHIELD_AVAILABLE_DRAFT` to the import block.**
  In `src/elspeth/web/composer/service.py`, the `from elspeth.web.interpretation_state import (`
  block (`:131-139`) imports `PROMPT_SHIELD_USER_TERM` then `PROMPT_SHIELD_WARNING_DRAFT`.
  Add `PROMPT_SHIELD_AVAILABLE_DRAFT` in alphabetical position (before
  `PROMPT_SHIELD_USER_TERM`):
  ```python
      PROMPT_SHIELD_AVAILABLE_DRAFT,
      PROMPT_SHIELD_USER_TERM,
      PROMPT_SHIELD_WARNING_DRAFT,
  ```
  (If `ruff check --fix` reorders, accept the autofix.)

  > **W1 — writer draft-equality is satisfied either way.** The writer boundary
  > (`sessions/service.py` `create_pending_interpretation_event`, PIPELINE_DECISION
  > branch) pins `llm_draft == the staged requirement's draft` — it does NOT pin the
  > exact module constant. Whether the agent stages the B-draft or the C-draft, the
  > event-draft equality check accepts it. No writer change is needed.

- [ ] **Step 5: Run the located test (or the snapshot test) and the composer prompt tests.**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_prompts.py -q
  ```
  Expected: pass. If a snapshot/value test fails because the repair-turn text
  changed, UPDATE that test's expected string to the new prose (test-to-UPDATE per
  Global Constraints) — do not revert the prose.

- [ ] **Step 6: Commit.**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_prompts.py && git commit -m "$(cat <<'EOF'
feat(composer): repair-turn selects state-appropriate prompt-shield draft

The forced-repair-turn prose now names BOTH the State-B
(PROMPT_SHIELD_AVAILABLE_DRAFT) and State-C (PROMPT_SHIELD_WARNING_DRAFT)
drafts and instructs the agent to stage whichever matches deployment
availability. The writer draft-equality check accepts either; no writer
change. Advisory polarity ("keep going with the warning") preserved.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```
  > If Step 1 found no test asserting the repair-turn text, drop the test path from
  > the `git add` and commit only `service.py`.

---

### Task 5: Re-word the agent-prose tuple to describe always-on A/B/C (documentation)

**Files:**
- Modify: `src/elspeth/plugins/transforms/llm/transform.py` — `get_agent_assistance()` `composer_hints` tuple (`:1714-1722`).

**Interfaces:**
- Consumes: nothing (prose only).
- Produces: nothing (prose only). This step does NOT make the review fire — that
  is Tasks 1-4. Per project doctrine "no tests for skill-prompt content", this
  change is NOT asserted by a test; it is verified by re-reading and by the
  source_file_hash/tier-model re-sign.

> **Why a separate task:** editing this plugin file trips the plugin
> `source_file_hash` gate (`transform.py:1127` carries
> `source_file_hash="sha256:bd14eb75b0b8ede6"`) AND, because the prose tuple does
> not add imports, does NOT shift `Module.body` import indices — but ANY byte
> change still restages the file hash. Isolating the prose edit in its own commit
> keeps the operator-owed re-sign surface clean and separable from the runtime
> change (Tasks 1-4).

- [ ] **Step 1: Re-word the conditional shield prose to always-on A/B/C.**
  In `src/elspeth/plugins/transforms/llm/transform.py` (`get_agent_assistance()`
  def at L1660, `composer_hints` tuple opens at L1694), replace the three
  conditional lines (`:1714`, `:1717`, `:1718`) with always-on framing.
  Replace `:1714`:
  ```python
                      "If internet content, public web content, search results, crawled pages, or other untrusted remote text will flow into this LLM, prompt-injection shielding is important: surface this to the user as a strong recommendation.",
  ```
  with:
  ```python
                      "Prompt-injection shielding is reviewed for EVERY LLM node, not only when untrusted web content is upstream: an unshielded LLM is always surfaced as an advisory (never blocking).",
  ```
  Replace `:1717`:
  ```python
                      "If the workflow proceeds without an authorized prompt shield, stage a pipeline_decision review on the LLM node with user_term prompt_injection_shield_recommendation.",
  ```
  with:
  ```python
                      "Stage a pipeline_decision review on the LLM node with user_term prompt_injection_shield_recommendation whenever no authorized shield is upstream (State B/C). Skip it only when an authorized shield is already wired upstream (State A).",
  ```
  Replace `:1718`:
  ```python
                      "Its draft should recommend an available authorized prompt-injection shield while stating that internet-controlled text will otherwise flow directly to the LLM.",
  ```
  with:
  ```python
                      "Pick the state-appropriate draft: State B (an authorized shield is available in this deployment) recommends wiring it in; State C (no shield available) is the high-risk reconsider advisory. Default to the State-C draft when availability is unknown.",
  ```
  Leave `:1715` (`Recommend an available authorized prompt-injection shield before the LLM; use azure_prompt_shield only when discovery lists it.`) and `:1719-1722`
  unchanged — they remain accurate under always-on.

- [ ] **Step 2: Verify the file still imports and the tuple is well-formed.**
  ```
  cd /home/john/elspeth && python -c "import ast; ast.parse(open('src/elspeth/plugins/transforms/llm/transform.py').read()); print('ok')"
  ```
  Expected: `ok`.

- [ ] **Step 3: Run the broad LLM-transform + composer-prompt test surface to catch any prose-string assertion.**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_tools.py tests/unit/web/composer/test_prompts.py -q
  ```
  Expected: pass. If a test asserts the OLD conditional shield prose verbatim,
  UPDATE it to the new always-on text (test-to-UPDATE) — but per doctrine, prefer
  to confirm no such brittle prose assertion exists; if one does, that assertion is
  itself suspect (it asserts skill-prompt content). Stage the updated test if you
  must change one.

- [ ] **Step 4: Compute the new plugin source_file_hash and stage it as an operator-owed chore.**
  > The agent SIGNS NOTHING but CO-LANDS the hash so the diff is complete; the
  > operator re-signs the gate. Run the refresh helper to compute (NOT to commit
  > unilaterally):
  ```
  cd /home/john/elspeth && python scripts/cicd/plugin_hash.py --help
  ```
  Then use `fix_source_file_hash` per its help to update the
  `source_file_hash="sha256:..."` literal at `transform.py:1127`. Verify it changed:
  ```
  cd /home/john/elspeth && grep -n 'source_file_hash' src/elspeth/plugins/transforms/llm/transform.py | head -1
  ```
  Expected: the sha differs from `sha256:bd14eb75b0b8ede6`.

- [ ] **Step 5: Commit (with the gate skips; operator re-signs).**
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add src/elspeth/plugins/transforms/llm/transform.py && git commit -m "$(cat <<'EOF'
docs(plugins/llm): reword shield agent-prose to always-on A/B/C

The composer_hints prose now describes the always-on 3-state prompt-shield
review (A shielded upstream / B available-not-wired / C none available)
instead of the old "if untrusted content" conditional. Prose only — the
runtime substance is in interpretation_state.py. Plugin source_file_hash
restaged (operator re-signs the plugin-hash + tier-model gates).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

### Task 6: Co-land tier-model + plugin-hash fingerprint updates; full-slice reconciliation

**Files:**
- Modify (co-land, operator re-signs): `config/cicd/enforce_tier_model/web.yaml` (the web files this slice edited: `interpretation_state.py`, `composer/service.py`, `sessions/routes/composer/guided.py`, and the NEW `composer/tools/_shield_availability.py` if it is fingerprinted), `config/cicd/enforce_tier_model/plugins.yaml` (`transforms/llm/transform.py`). NOTE: this plan does NOT edit `composer/state.py` (`validate()` is unchanged) or `composer/guided/emitters.py` (Task 3 chokes at the route handler, not the builder) — do not restage their fingerprints.
- No new source. This task reconciles the slice and produces the operator hand-off.

**Interfaces:**
- Consumes: every source change from Tasks 1-5.
- Produces: a green local lint surface (modulo the two operator-owed gates) and a stated operator re-sign chore.

- [ ] **Step 1: Enumerate the tests the always-on blast radius can flip BEFORE running the broad surface.**
  > Task 1 changed shield-warning emission from web_scrape-fed-only to **every**
  > unshielded LLM node. The blast radius is wider than this plan's explicit
  > coverage: ANY existing test with a non-scrape→llm pipeline that asserts an empty
  > or counted `warnings` set gains a +1 shield warning. Enumerate the candidates up
  > front so the Step 2 wave is expected, not a surprise red after every prior task
  > looked green:
  ```
  cd /home/john/elspeth && grep -rln "validate().warnings\|prompt_shield\|warning_pairs\|len(.*warnings" tests/integration/web/composer/guided/ tests/unit/web/test_interpretation_state.py tests/unit/web/composer/
  ```
  Resolve each hit against the always-on behavior. Specifically pre-cleared by
  ground-truth inspection:
  - **`test_interpretation_state.py:505` (`test_gate_routed_web_scrape_into_llm_warns_without_prompt_shield`)** —
    its transform node IS an `llm` node fed by `web_scrape` (verified: the helper
    `_state_with_web_scrape_gate_to_llm` builds `web_scrape → gate → llm`). It
    asserts `assert warning_pairs` (non-empty) and
    `any(component == "node:summarise_pages" ...)` — a PRESENCE + component
    assertion, NOT a count. It SURVIVES unchanged under always-on (the node still
    warns; no `len(warnings) == 1 → 2` flip exists in this test). Leave it as-is.
  - **`test_interpretation_state.py:524` (`..._through_prompt_shield_emits_no_warning`)** —
    State A (`assert warning_pairs == ()`); the decoupled `_llm_has_authorized_shield_upstream`
    keeps it silent. SURVIVES.
  - **`test_step_3_e2e.py`** proposes a `passthrough`-only chain (NO llm node) ⇒ no
    new warning. SURVIVES.
  Any OTHER enumerated test that asserts a SPECIFIC `len(warnings) == N` on a
  pipeline containing an unshielded `llm` node MUST be updated to `N + 1` (the
  shield warning) per Global-Constraints test-to-UPDATE — verify each by reading
  whether its transform node is `llm`-type; do not assume.

- [ ] **Step 2: Run the BROAD composer regression surface (Task 1 is a global behavior change).**
  > Task 1 makes EVERY unshielded LLM node warn, not only web_scrape-fed ones. Any
  > existing test with a non-scrape→llm pipeline (e.g. `csv→llm`) that asserts an
  > empty or counted `warnings` set now gains a shield warning. Treat the resulting
  > wave as test-to-UPDATE (the change landing visibly), NOT as a reason to revert.
  > Reconcile against the Step 1 enumeration; each red should already be expected.
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/test_interpretation_state.py tests/unit/web/composer/ tests/integration/web/composer/guided/ -q
  ```
  Expected: all pass after reconciling any non-scrape→llm warning-count assertions
  to the always-on behavior. `test_web_scrape_recipe_apply.py::test_apply_web_scrape_recipe_json_preserves_shield_advisory`
  (`:118`) and `::test_apply_web_scrape_recipe_csv_preserves_shield_advisory`
  (`:145`) assert `prompt_shield_recommendation_warning_pairs(state)` is non-empty
  for the web_scrape→LLM recipe — STILL TRUE under always-on (web_scrape is still
  unshielded). Any test asserting a specific warning COUNT on an llm-containing
  pipeline must be updated for the new +1 shield warning per unshielded llm node.

- [ ] **Step 3: Run the static-analysis surface the operator owes a re-sign for, and capture the deltas.**
  ```
  cd /home/john/elspeth && uv run ruff check src/elspeth/web/interpretation_state.py src/elspeth/web/composer/tools/_shield_availability.py src/elspeth/web/composer/service.py src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/plugins/transforms/llm/transform.py && uv run mypy src/elspeth/web/interpretation_state.py src/elspeth/web/composer/tools/_shield_availability.py src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Expected: ruff clean; mypy clean for the new module (pre-existing mypy reds in
  service.py per project memory are operator-owed — do NOT chase them here).

- [ ] **Step 4: Run the tier-model fingerprint check to see which fingerprints went stale.**
  ```
  cd /home/john/elspeth && uv run python -m elspeth_lints.rules.trust_tier.tier_model.rotate --help 2>/dev/null || cat scripts/cicd/rotate_tier_model_fingerprints.py >/dev/null; echo "rotation tool located"
  ```
  Use the rotation tool's `check --format json` (per project memory) to list stale
  fingerprints in `web.yaml` (interpretation_state.py / service.py / sessions/routes/composer/guided.py) and
  `plugins.yaml` (transform.py). **The agent computes and co-lands the updated
  fingerprints in those YAML files so the diff is complete; the OPERATOR re-signs
  the gate (HMAC key isolation).** Do not invent fingerprints; run the tool.
  > Caveat (project memory): rotate deletes BOTH copies of a duplicate key —
  > restore stale first, git-diff after, re-run if a dup-key data-loss appears. The
  > tooling is Python-version sensitive: the worktree venv MUST match main's
  > Python 3.13 or ~300 spurious violations appear.

- [ ] **Step 5: Stage the co-landed YAML and commit (gates skipped; operator re-signs).**
  ```
  cd /home/john/elspeth && git status --porcelain config/cicd/enforce_tier_model/web.yaml config/cicd/enforce_tier_model/plugins.yaml
  ```
  If both are modified by the rotation tool:
  ```
  cd /home/john/elspeth && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git add config/cicd/enforce_tier_model/web.yaml config/cicd/enforce_tier_model/plugins.yaml && git commit -m "$(cat <<'EOF'
chore(cicd): co-land tier-model fingerprints for always-on prompt-shield

Restage the AST fingerprints shifted by the interpretation_state.py /
service.py / transform.py edits in the always-on prompt-shield slice.
Operator re-signs the trust-tier and plugin-hash gates per the release-train
process; the agent signs nothing.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

- [ ] **Step 6: Run wardline at the trust boundary.**
  ```
  cd /home/john/elspeth && wardline scan . --fail-on ERROR
  ```
  Expected: exit 0. The new code consumes only `ToolContext` (already-validated
  deployment state) and `CompositionState` (already through the validate seams) —
  no new external-input boundary. If wardline flags a finding, fix at the boundary
  per the `wardline-gate` skill; do not waive without cause.

- [ ] **Step 7: State the operator hand-off (in the PR/branch summary, not a file).**
  The slice is complete and green locally except the two operator-owed re-sign
  gates. Surface to the operator: (a) `transform.py` plugin `source_file_hash`
  re-sign (`plugin-contract-plugin-hashes`); (b) tier-model fingerprint re-sign
  for `web.yaml` + `plugins.yaml` (`trust-tier-model`). The agent has co-landed the
  computed hash + fingerprints; the operator holds the HMAC key and re-signs per
  the release-train process. No `GUIDED_SESSION_SCHEMA_VERSION`/`SESSION_SCHEMA_EPOCH`
  bump (contract §3 p3): no new `InterpretationKind`, no DB recreate, no boot
  fail-close — persisted interpretation-row shape already supports
  `pipeline_decision` + `prompt_injection_shield_recommendation`; the B/C draft
  constants are module-constant-only and excluded from the artifact hash.

---

## Cross-plan consumption (for p4)

p4's tutorial worked example CONSUMES, do NOT re-implement:
- `prompt_shield_state_for_node(node, all_nodes, *, shield_available: bool) -> "A"|"B"|"C"` (Task 2) — to assert the worked example lands in State C on the synthetic source (no shield wired).
- `azure_prompt_shield_available(context: ToolContext) -> bool` (Task 2) — the B-vs-C resolver, default fail-safe False.
- `PROMPT_SHIELD_WARNING_DRAFT` (C) / `PROMPT_SHIELD_AVAILABLE_DRAFT` (B) constants (Task 1).
- The always-on warning surfacing through `CompositionState.validate().warnings` at "medium" (Task 1), refined to the live B-draft at the wire-stage turn (Task 3) — p4's "force the override on screen" copy rides this advisory (State C on the synthetic source, since no Azure key is configured for the tutorial); it is never promoted to an error.
