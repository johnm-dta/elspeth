> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P3 — B1 interpretation surfacing (all 5 kinds) + frontend projection

> **Scope (spec §5 B1, D12, §7).** Fix the latent silent-orphan bug: the
> freeform fail-closed orphan gate
> (`_missing_pending_interpretation_review_sites`, `composer/service.py:1390`)
> is **unreachable** from the guided dispatch path. The guided commit handlers
> (`steps.py`) call `_execute_*` tools directly and never traverse any of the
> freeform finalize/checkpoint call sites that run the gate, so a guided step-3
> LLM node — or the deterministic recipe-apply — creates real interpretation
> sites that are surfaced to **no one** and only fail at run time with
> `UnresolvedInterpretationPlaceholderError` (`execution/service.py:515-529`).
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
>   (`post_guided_respond`, `composer/guided.py:704`; persistence seam `:~1185`) which runs on the
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

### Task P3.1: Add the kind-general `surface_pending_interpretation_reviews` backend surfacer

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add a method on
  `ComposerServiceImpl` immediately after `_auto_surface_prompt_template_reviews`,
  which currently starts at `:1426`)
- Create: `tests/unit/web/composer/test_surface_pending_interpretation_reviews.py`

**Interfaces:**
- Consumes: `interpretation_sites(state: CompositionState) -> tuple[InterpretationReviewSite, ...]`
  (`web/interpretation_state.py:403`); `InterpretationReviewSite` fields
  `component_id: str`, `component_type: str`, `user_term: str`,
  `kind: InterpretationKind`; `InterpretationKind` members `VAGUE_TERM`,
  `INVENTED_SOURCE`, `LLM_PROMPT_TEMPLATE`, `PIPELINE_DECISION`,
  `LLM_MODEL_CHOICE` (`contracts/composer_interpretation.py:81-85`);
  `SessionServiceImpl.create_pending_interpretation_event(*, session_id: UUID,
  composition_state_id: UUID, affected_node_id: str, tool_call_id: str,
  user_term: str, kind: InterpretationKind, llm_draft: str,
  model_identifier: str, model_version: str, provider: str,
  composer_skill_hash: str)` (`sessions/service.py:2778`);
  `SessionServiceImpl.list_interpretation_events(session_id, status=...)`;
  `self._require_sessions_service()`, `self._model`, `self._availability`,
  `self._composer_skill_hash` (all already used by
  `_auto_surface_prompt_template_reviews`).
- Produces (NEW, owned by P3, consumed by P3.2/P3.3/P3.4):
  `async def surface_pending_interpretation_reviews(self, state: CompositionState, *, session_id: str | None, current_state_id: str | None) -> None`

**Writer-boundary facts (verified, drive the per-kind draft/user_term mapping):**
`create_pending_interpretation_event` (`sessions/service.py:2778`) is strict per
kind — it re-reads the persisted parent state inside the locked transaction and,
for the three requirement-checked kinds below (`INVENTED_SOURCE`,
`PIPELINE_DECISION`, `LLM_PROMPT_TEMPLATE`), **raises `ValueError`** unless the
staged requirement shape matches exactly. `LLM_MODEL_CHOICE` and a bare
`VAGUE_TERM` fall through the writer's `else`-branch (`:2936`), which only runs
`_find_llm_transform_node` (an llm transform node with a non-empty
`prompt_template`) and performs **no** staged-requirement-shape check:
- `INVENTED_SOURCE` (`:2855`): `affected_node_id` must equal `SOURCE_COMPONENT_ID`;
  the source must carry a `SOURCE_AUTHORING_KEY` block with a non-empty
  `content_hash`; exactly one pending `invented_source` requirement matching
  `user_term`; `llm_draft` must equal that requirement's `draft`.
- `LLM_PROMPT_TEMPLATE` (`:2949`): node's `options.prompt_template` must be a
  string and `llm_draft` must equal it; exactly one pending PT requirement.
- `LLM_MODEL_CHOICE` (`:~2936` else-branch): goes through `_find_llm_transform_node`;
  the `_options_with_default_model_choice_review` auto-stager
  (`composer/tools/_common.py:202`) stages the requirement with
  `user_term=f"llm_model_choice:{node_id}"` and `draft=options.model`.
- `PIPELINE_DECISION` (`:2894`): `_find_interpretation_review_node` + exactly
  one pending requirement + `validate_pipeline_decision_semantics`; `llm_draft`
  must equal the requirement's `draft`.
- `VAGUE_TERM`: two cases must stay separate. A staged vague-term requirement
  with an authored `draft` MAY be surfaced using that exact draft. A bare legacy
  placeholder (`{{interpretation:cool}}`, `{{interpretation:legacy}}`, etc.)
  carries **no** staged requirement; the backend must not infer intent from a
  word list, must not invent a draft, and must not special-case `cool`/`legacy`.
  Bare placeholders stay fail-closed at the run-time gate/backstop.

Because of these per-kind preconditions, the surfacer is built as a per-kind
dispatch that reads the site's own `draft`/`user_term` from the node/source
requirement (NOT from a synthesized value), then calls the writer with the exact
matching draft. This means the plan really covers all five `InterpretationKind`
members by surfacing `VAGUE_TERM` only when it is staged, while explicitly
leaving bare legacy placeholders to the run-time gate. This reuses
`_auto_surface_prompt_template_reviews`'s honest provenance sentinel
(`tool_call_id="backend_auto_surface:<uuid4>"`,
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
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
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
      record = await _save_state_for_session(sessions_service, session_id, state)
      return session_id, record.id


  async def _save_state_for_session(sessions_service, session_id: UUID, state: CompositionState):
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
      return record


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
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.LLM_PROMPT_TEMPLATE in kinds
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
  ```
  Expected failure: `AttributeError: 'ComposerServiceImpl' object has no attribute 'surface_pending_interpretation_reviews'`.

- [ ] **Step 2: Add the public method (minimal — delegate prompt_template to the
  existing surfacer, then handle the non-template staged kinds).**
  In `src/elspeth/web/composer/service.py`, immediately after
  `_auto_surface_prompt_template_reviews` (after its closing line `:1489`), add:

  ```python
      async def surface_pending_interpretation_reviews(
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
          designed advisory polarity (spec §5 B1). No backend word-list heuristic
          and no synthesized "cool"/"legacy" draft are permitted.

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
                  and (event.llm_draft or "").strip() == llm_draft.strip()
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
          if site.kind is InterpretationKind.VAGUE_TERM:
              # Only authored/staged vague-term requirements are surfaced.
              # Bare legacy placeholders carry no requirement and are left
              # fail-closed at the run-time gate; never invent a draft.
              draft = self._matching_requirement_draft(options, kind=site.kind, user_term=site.user_term)
              if draft is None:
                  return None
              return (node.id, site.user_term, draft)
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
  `ruff check --fix` reorders it, accept the autofix.) Confirm the current block (lines 131-140 verified correct in release/0.7.0):
  ```
  sed -n '131,140p' /home/john/elspeth/src/elspeth/web/composer/service.py
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_prompt_template -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Add the failing test for the model_choice path.**
  Append to `test_surface_pending_interpretation_reviews.py`:

  ```python
  def _model_choice_node(model: str = "anthropic/claude-sonnet-4.6") -> NodeSpec:
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={
              "prompt_template": "Rate this row and return JSON.",
              "model": model,
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
                      "draft": model,
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
      await composer.surface_pending_interpretation_reviews(
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
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
  ```
  Expected: `2 passed`. (If the model_choice assertion fails, the
  `_backend_surface_args_for_site` model_choice branch is the defect — fix it
  there, not in the test.)

- [ ] **Step 4: Add idempotency, draft-aware dedup, staged-vague, and
  skip-bare-vague-term tests.**
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
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      mc = [e for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
      assert len(mc) == 1


  @pytest.mark.asyncio
  async def test_model_choice_dedup_is_draft_aware(tmp_path, sessions_service) -> None:
      """A stale pending event for the same node/kind/user_term but an old draft
      must not deadlock the new staged requirement."""
      composer = _composer(tmp_path, sessions_service)
      old_model = "anthropic/claude-sonnet-4.5"
      new_model = "anthropic/claude-sonnet-4.6"
      old_state = CompositionState(
          source=None,
          nodes=(_model_choice_node(old_model),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, old_state_id = await _persist(sessions_service, old_state)
      await composer.surface_pending_interpretation_reviews(
          old_state, session_id=str(session_id), current_state_id=str(old_state_id)
      )

      new_state = CompositionState(
          source=None,
          nodes=(_model_choice_node(new_model),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=2,
      )
      new_record = await _save_state_for_session(sessions_service, session_id, new_state)
      await composer.surface_pending_interpretation_reviews(
          new_state, session_id=str(session_id), current_state_id=str(new_record.id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      drafts = [e.llm_draft for e in events if e.kind is InterpretationKind.LLM_MODEL_CHOICE]
      assert old_model in drafts
      assert new_model in drafts


  def _staged_vague_term_node() -> NodeSpec:
      draft = "modern, useful, engaging, and clear for the public."
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="out",
          on_error=None,
          options={
              "prompt_template": "Rate how {{interpretation:cool}} this is.",
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "vague_term_review:rate_node",
                      "kind": InterpretationKind.VAGUE_TERM.value,
                      "user_term": "cool",
                      "status": "pending",
                      "draft": draft,
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
  async def test_surfacer_surfaces_staged_vague_term(tmp_path, sessions_service) -> None:
      # A staged vague-term requirement with an authored draft is surfaced.
      # This is NOT a backend word-list heuristic and NOT an invented draft.
      composer = _composer(tmp_path, sessions_service)
      state = CompositionState(
          source=None,
          nodes=(_staged_vague_term_node(),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, state_id = await _persist(sessions_service, state)
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      vt = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
      assert len(vt) == 1
      assert vt[0].user_term == "cool"
      assert vt[0].llm_draft == "modern, useful, engaging, and clear for the public."


  @pytest.mark.asyncio
  async def test_surfacer_skips_bare_vague_term(tmp_path, sessions_service) -> None:
      # A bare {{interpretation:cool}} token with NO staged requirement is a
      # legacy vague_term site. The surfacer must SKIP it (left fail-closed
      # at the run-time gate) and must not infer a draft from the word "cool".
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
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      vt = [e for e in events if e.kind is InterpretationKind.VAGUE_TERM]
      assert vt == []
  ```

  Run to pass:
  ```
      cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -x -q
      ```
      Expected: `6 passed`.

  > **On the leftover prior-draft event:** draft-aware dedup
  > (`test_model_choice_dedup_is_draft_aware` /
  > `test_pipeline_decision_dedup_is_draft_aware`) intentionally leaves the prior
  > pending event in place when a node is re-staged with a new draft. This is
  > benign in the integrated D12 flow: P3.6 disables guided submit while ANY card
  > is pending, so the user cannot re-stage a node's decision with a new draft
  > while the old card is still pending — the orphan-accumulation deadlock is
  > therefore unreachable.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): add kind-general interpretation surfacer (B1)

Add ComposerServiceImpl.surface_pending_interpretation_reviews, a
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
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py` (`post_guided_respond`
  at `:704`; main persistence seam — after `_dispatch_guided_respond` — at `:~1185`)

  > **RECONCILE (composer-routes decomposition):** The monolith
  > `sessions/routes/composer.py` was split into the package
  > `sessions/routes/composer/` (merged `0e754a67e` into release/0.7.0).
  > `post_guided_respond` now lives in `guided.py:704`; the persistence seam
  > is `guided.py:~1185`. All subsequent references to `composer.py` in this
  > task have been updated accordingly.

- Modify: `src/elspeth/web/composer/protocol.py` (add
  `surface_pending_interpretation_reviews` to the `ComposerService` Protocol — see
  Step 3b)
- Modify: `tests/integration/web/composer/guided/conftest.py` (wire a real
  `ComposerServiceImpl` onto `app.state.composer_service`, replacing the `None` at
  `:99` — see Step 0)
- Create: `tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py`

**Interfaces:**
- Consumes: `ComposerServiceImpl.surface_pending_interpretation_reviews`
  (P3.1), reached via `request.app.state.composer_service` (a `ComposerService`
  handle; the recompose path uses the same pattern in `compose.py:150`),
  None-guarded; `service.save_composition_state(...) -> state_record_out` with
  `state_record_out.id` (`guided.py:~1185`) — here `service` is the route's
  `session_service` (set at route entry in `guided.py`), which exposes `save_composition_state`
  but NOT the surfacer (hence the separate composer handle); `new_state` (the
  post-dispatch state); `session_id` (the route path param).
- Produces: nothing new — wires the existing surfacer into the route.

**Why here:** the dispatcher (`_dispatch_guided_respond`, `_helpers.py:2571`) is
a pure routing function with **no service handle**. The route is the only place
that holds `service` and the freshly-persisted `state_record_out.id` (the
`current_state_id` the surfacer needs). The persist at `guided.py:~1185` runs after every
guided commit including recipe-apply and chain-accept; the
surfacer reads `interpretation_sites(state)` on the post-mutation `new_state`
and surfaces the delta. This is the single hook that covers source, transform,
and recipe-apply commits (P3.3/P3.4 are the per-boundary assertions).

- [ ] **Step 1: Read the exact persistence block to anchor the edit.**
  ```
  sed -n '1178,1200p' /home/john/elspeth/src/elspeth/web/sessions/routes/composer/guided.py
  ```
  Confirm `state_record_out = await service.save_composition_state(...)` is in this
  range and `new_state` is the variable holding the post-dispatch state.

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
  (`sessions/routes/interpretation.py:198`, query param
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
  > `surface_pending_interpretation_reviews` on it, None-guarded — surfacing is
  > advisory; the run-time `UnresolvedInterpretationPlaceholderError` gate
  > (`execution/service.py:515-529`) stays the hard backstop, so skipping when no
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
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py -x -q
  ```
  Expected failure: the pending-event assertion fails (`kinds` is empty / missing
  the two kinds) because the route does not yet call the surfacer.

- [ ] **Step 3: Add the surfacer call after the persist, through the composer handle.**
  The B1 surfacer (`surface_pending_interpretation_reviews`) is a
  `ComposerServiceImpl` method. The route's local `service` is
  `request.app.state.session_service` (a `SessionService`), which does NOT carry
  the surfacer. The route DOES hold a `ComposerService` handle in the
  `app.state.composer_service` slot — the same pattern the recompose endpoint
  uses (`composer: ComposerService = request.app.state.composer_service`,
  `compose.py:150`). In
  `src/elspeth/web/sessions/routes/composer/guided.py`, immediately after the
  `state_record_out = await service.save_composition_state(...)` call at `:~1185`,
  insert:

  ```python
                # B1 (spec §5): the guided dispatch path never reaches the
                # freeform fail-closed orphan gate, so a committed source /
                # transform / recipe-apply that creates interpretation sites
                # would orphan and only fail at run time. Surface every
                # resolvable pending review against the freshly-persisted state
                # so the guided UI can project + block on it (D12). Advisory
                # polarity: the run-time gate (execution/service.py:515-529)
                # stays the hard backstop, so a None composer (no impl wired in
                # this app) safely skips surfacing.
                composer: ComposerService = request.app.state.composer_service
                if composer is not None:
                    await composer.surface_pending_interpretation_reviews(
                        new_state,
                        session_id=str(session_id),
                        current_state_id=str(state_record_out.id),
                    )
  ```

  **`ComposerService` is NOT currently imported in `guided.py`** — add
  `from elspeth.web.composer.protocol import ComposerService` to `guided.py`'s
  imports. (The `recompose` handler in `compose.py` imports it at `:13`; `guided.py`
  is a separate module and does not inherit that import.) Do NOT widen the call
  to `getattr`. (The Protocol itself gains the method in Step 3b, which mypy
  requires because the call is on a `ComposerService`-typed handle.)

- [ ] **Step 3b: Add `surface_pending_interpretation_reviews` to the
  `ComposerService` Protocol (mypy gate, §9.2).**
  The Step 3 call is on `composer: ComposerService` (the Protocol type, not the
  concrete impl), so mypy — a §9.2 gate — errors with `"ComposerService" has no
  attribute "surface_pending_interpretation_reviews"` until the method is declared
  on the Protocol. Add it, mirroring how P5.1 Step 4 adds `run_signoff_checkpoint`
  to the same Protocol. In `src/elspeth/web/composer/protocol.py`, immediately
  AFTER the `compose(...)` method's closing docstring/`"""` (`:759`) and BEFORE
  `async def explain_run_diagnostics` (`:761`), insert the declaration — the
  signature is identical to P3.1's `ComposerServiceImpl` method definition:
  ```python
      async def surface_pending_interpretation_reviews(
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
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py -x -q && uv run mypy src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/composer/protocol.py
  ```
  Expected: `1 passed`, then `Success: no issues found`.

- [ ] **Step 4: Run the guided-respond route regression suite (no behaviour
  regression on existing paths).**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/sessions/routes/ tests/integration/web/composer/ -q -k "guided or respond"
  ```
  Expected: all pass (the surfacer is a no-op when there are no pending
  requirements, so existing source/sink-only guided tests are unaffected).

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && git add src/elspeth/web/sessions/routes/composer/guided.py src/elspeth/web/composer/protocol.py tests/integration/web/composer/guided/conftest.py tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py && git commit -m "$(cat <<'EOF'
feat(composer/guided): fire interpretation surfacer at guided persist seam (B1)

Wire surface_pending_interpretation_reviews into post_guided_respond
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
  `invented_source` site (`interpretation_state.py:561`); the writer
  boundary's invented_source branch (`sessions/service.py:2855`) requires
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
  cd /home/john/elspeth && grep -n "SOURCE_AUTHORING_KEY\|content_hash\|modality\|_is_llm_authored_modality\|_source_authoring_metadata\|invented_source" src/elspeth/web/interpretation_state.py | head -25
  ```
  Read `_source_authoring_metadata` and `_pending_source_sites`
  (`interpretation_state.py:561`) to confirm the `options[SOURCE_AUTHORING_KEY]`
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
      await composer.surface_pending_interpretation_reviews(
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
  cd /home/john/elspeth && grep -n "class SourceSpec\|def __init__\|modality\|llm_generated\|_is_llm_authored_modality" src/elspeth/web/composer/state.py src/elspeth/web/interpretation_state.py | head
  ```
  Adjust the `modality` literal and any required SourceSpec fields to the
  verbatim values the codebase uses (read `_is_llm_authored_modality` to get the
  accepted modality string).

  Run to fail (then pass once the literals match):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_invented_source -x -q
  ```
  If it fails with the surfacer skipping the source, the `_backend_surface_args_for_site`
  INVENTED_SOURCE branch is reading the wrong key — fix the impl. If it fails at
  the writer boundary with a `ValueError`, the test's authoring/requirement
  shape does not match what `_persist` round-trips — fix the test fixture.
  Expected final: `1 passed`.

- [ ] **Step 3: Run the full surfacer suite.**
  ```
      cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -q
      ```
      Expected: `7 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth && git add tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
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
  (`_missing_raw_html_cleanup_review_sites`, `interpretation_state.py:631`) fires
  when a `web_scrape` node's content/fingerprint fields are not preserved by a
  `field_mapper(select_only)` node; the writer boundary's PIPELINE_DECISION
  branch (`sessions/service.py:2894`) requires exactly one pending requirement +
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
  cd /home/john/elspeth && grep -n "validate_pipeline_decision_semantics\|RAW_HTML_CLEANUP_USER_TERM\|pipeline_decision\|def validate_pipeline_decision" src/elspeth/web/interpretation_state.py src/elspeth/web/sessions/service.py | head
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
  from elspeth.web.interpretation_state import (
      RAW_HTML_CLEANUP_REVIEW_DRAFT,
      RAW_HTML_CLEANUP_USER_TERM,
  )


  def _field_mapper_pipeline_decision_node(
      draft: str = RAW_HTML_CLEANUP_REVIEW_DRAFT,
  ) -> NodeSpec:
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
                      "draft": draft,
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
      await composer.surface_pending_interpretation_reviews(
          state, session_id=str(session_id), current_state_id=str(state_id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      kinds = {e.kind for e in events}
      assert InterpretationKind.PIPELINE_DECISION in kinds


  @pytest.mark.asyncio
  async def test_pipeline_decision_dedup_is_draft_aware(tmp_path, sessions_service) -> None:
      """A stale raw-HTML-cleanup event for the same field_mapper decision must
      not suppress a newly authored decision draft."""
      composer = _composer(tmp_path, sessions_service)
      old_draft = "Drop raw HTML fields before writing output."
      new_draft = RAW_HTML_CLEANUP_REVIEW_DRAFT
      old_state = CompositionState(
          source=None,
          nodes=(_field_mapper_pipeline_decision_node(old_draft),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      session_id, old_state_id = await _persist(sessions_service, old_state)
      await composer.surface_pending_interpretation_reviews(
          old_state, session_id=str(session_id), current_state_id=str(old_state_id)
      )

      new_state = CompositionState(
          source=None,
          nodes=(_field_mapper_pipeline_decision_node(new_draft),),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=2,
      )
      new_record = await _save_state_for_session(sessions_service, session_id, new_state)
      await composer.surface_pending_interpretation_reviews(
          new_state, session_id=str(session_id), current_state_id=str(new_record.id)
      )
      events = await sessions_service.list_interpretation_events(session_id, status="pending")
      drafts = [e.llm_draft for e in events if e.kind is InterpretationKind.PIPELINE_DECISION]
      assert old_draft in drafts
      assert new_draft in drafts
  ```

  Run to fail/pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py::test_surfacer_surfaces_pipeline_decision -x -q
  ```
  If the writer boundary rejects with a `validate_pipeline_decision_semantics`
  `ValueError`, the test's `plugin`/`options`/`draft` shape does not match the
  validator — adjust the fixture to the exact contract Step 1 read (this is a
  test-fixture fidelity issue, not an impl defect). Expected final: `1 passed`.

- [ ] **Step 3: Run the full surfacer suite (all five kinds now covered:
  prompt_template, model_choice, invented_source, pipeline_decision, staged
  vague_term; bare vague_term remains skipped/runtime-gated).**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py -q
  ```
      Expected: `9 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth && git add tests/unit/web/composer/test_surface_pending_interpretation_reviews.py && git commit -m "$(cat <<'EOF'
test(composer/guided): pin pipeline_decision surfacing (recipe-apply boundary)

    Covers the pipeline_decision path and its stale-draft regression; together
    with prompt_template/model_choice/invented_source/staged-vague-term tests
    and the bare-vague_term skip, all five InterpretationKind members are pinned.

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
  `execution/service.py:515-529` raises `UnresolvedInterpretationPlaceholderError`
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
  cd /home/john/elspeth && grep -n "UnresolvedInterpretationPlaceholderError\|ExecutionServiceImpl(\|def service\|mock_session_service\|async def resolve_interpretation_event" tests/unit/web/execution/test_service.py src/elspeth/web/sessions/service.py | head -25
  ```
  Read the `service`/`ExecutionServiceImpl` fixture in `test_service.py` and the
  `resolve_interpretation_event` signature in `sessions/service.py`. Decide the
  build: the cleanest backstop drives a real `ExecutionServiceImpl` over a real
  session engine with a persisted state carrying an unresolved LLM node, asserts
  `execute()` raises `UnresolvedInterpretationPlaceholderError`, then surfaces +
  resolves the pending events and asserts `execute()` returns a run id.

- [ ] **Step 2: Write the backstop test.**
  Create `tests/integration/web/composer/test_guided_interpretation_run_backstop.py`.
  Build a persisted state with a COMPLETE LLM pipeline — a `text` source (reading
  a real blob file, `column: "text"`), a complete `llm` node (provider, model,
  `api_key`, a non-empty `prompt_template`, `required_input_fields`, observed
  schema) carrying TWO pending requirements (`llm_prompt_template` AND
  `llm_model_choice`), a `json` sink, and a source->node edge — so
  `interpretation_sites` yields TWO pending sites and
  `materialize_state_for_execution` reports them as unresolved (BLOCK). Wire the
  real `ComposerServiceImpl` + `SessionServiceImpl` + `ExecutionServiceImpl` with a
  REAL `WebSettings` and the REAL `yaml_generator` (copy the construction from the
  dispatch-test `_build_composer`; the only mock is the loop, with `_call_async`
  bridged to a synchronous run). The helper `_persist_state_with_unresolved_node`
  (a) seeds the session + composition state with that complete pipeline (persisting
  `edges` and `outputs`, not just `nodes`/`sources`) and (b) creates BOTH matching
  pending, resolvable events through the writer boundary
  (`sessions/service.py:2949-2974`): the `llm_prompt_template` event takes the
  requirement-checked branch (which requires `options.prompt_template == llm_draft`
  AND exactly one pending PT requirement for `user_term` — both staged on the
  node), and the `llm_model_choice` event falls through the writer's `else`-branch
  (`:2936`), which only confirms a complete llm transform node and does no
  staged-requirement-shape check. Resolving BOTH events via
  `resolve_interpretation_event(... choice=ACCEPTED_AS_DRAFTED ...)` marks both
  requirements `status="resolved"` (`sessions/service.py:1161-1242`), clearing
  every site — the state then has zero sites, so the gate PERMITS and the REAL
  `validate_pipeline` runs against the complete pipeline and accepts it.

  > **Why a complete pipeline, not a minimal node:** the PERMIT branch runs the
  > REAL `validate_pipeline` (no patch), and `validate_pipeline` builds a runtime
  > graph. An incomplete `llm` node (e.g. no provider/model) fails that graph
  > build, which is why the earlier minimal fixture had to patch
  > `validate_pipeline`. A complete LLM pipeline — `text` source + complete `llm`
  > node + `json` sink + edge — builds a valid runtime graph end-to-end, so once
  > both reviews resolve the gate PERMITS and `validate_pipeline` accepts the
  > pipeline for real; only `_run_pipeline` is stubbed, to avoid a live engine run.

  The helper returns `(session_id, state_id, pt_event_id, mc_event_id, state)`:

  ```python
  from __future__ import annotations

  import asyncio
  from collections.abc import Coroutine
  from datetime import UTC, datetime
  from pathlib import Path
  from typing import Any, cast
  from unittest.mock import MagicMock, patch
  from uuid import UUID, uuid4

  import pytest
  import structlog
  from sqlalchemy import insert
  from sqlalchemy.pool import StaticPool

  from elspeth.contracts.composer_interpretation import (
      InterpretationChoice,
      InterpretationKind,
  )
  from elspeth.web.composer import yaml_generator as real_yaml_generator
  from elspeth.web.composer.service import ComposerAvailability, ComposerServiceImpl
  from elspeth.web.composer.state import (
      CompositionState,
      EdgeSpec,
      NodeSpec,
      OutputSpec,
      PipelineMetadata,
      SourceSpec,
  )
  from elspeth.web.config import WebSettings
  from elspeth.web.execution.errors import UnresolvedInterpretationPlaceholderError
  from elspeth.web.execution.progress import ProgressBroadcaster
  from elspeth.web.execution.service import ExecutionServiceImpl
  from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
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
      # F22 (same class as F1): WebSettings requires these four composer
      # fields; omitting them raises a 4-error pydantic ValidationError before
      # the service is ever built. Values mirror the guided conftest / F1.
      settings = WebSettings(
          data_dir=tmp_path,
          composer_model="anthropic/claude-opus-4-7",
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
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
      get_current_state(session_id) (~execution/service.py:484) loads the persisted
      state and the interpretation gate (:515-529) sees it.

      The interpretation gate fires BEFORE validate_pipeline / generate_yaml /
      create_run and is fully REAL in BOTH branches — it raises in BLOCK and passes
      in PERMIT. Settings and yaml_generator are REAL here (unlike the earlier
      minimal fixture): a REAL WebSettings (validate_pipeline reads data_dir and
      resolves the audit DB via get_landscape_url) and the REAL yaml_generator, so
      the PERMIT path runs the REAL validate_pipeline (NO patch) against the
      complete LLM pipeline and accepts it. Only _run_pipeline is stubbed by the
      test, to avoid a live engine run; create_run still runs for real, so
      execute() returns a real run_id. Only the loop is mocked, with _call_async
      bridged to a synchronous run (mirror the canonical `service` fixture's
      _call_async bridge in test_service.py) because the real _call_async uses
      run_coroutine_threadsafe, which needs a running loop.

      WebSettings mirrors the sibling `_composer` helper (the uniform-byte
      placeholder signing key is accepted under pytest, where WebSettings allows
      insecure test keys) plus the one field the ExecutionService path needs beyond
      it: landscape_url for create_run's audit DB.
      """
      mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
      broadcaster = ProgressBroadcaster(mock_loop)
      settings = WebSettings(
          data_dir=tmp_path,
          composer_model="anthropic/claude-opus-4-7",
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          shareable_link_signing_key=b"\x00" * 32,
          landscape_url=f"sqlite:///{tmp_path}/audit.db",
      )
      svc = ExecutionServiceImpl(
          loop=mock_loop,
          broadcaster=broadcaster,
          settings=settings,
          session_service=sessions_service,
          yaml_generator=real_yaml_generator,
          telemetry=build_sessions_telemetry(),
      )
      _real_loop = asyncio.new_event_loop()

      def _mock_call_async(coro: Coroutine[Any, Any, Any]) -> Any:
          try:
              return _real_loop.run_until_complete(coro)
          except RuntimeError:
              coro.close()
              return None

      cast(Any, svc)._call_async = _mock_call_async
      return svc


  PROMPT = "Rate {{ row.text }} and return JSON."
  MODEL = "anthropic/claude-sonnet-4.6"


  def _llm_node() -> NodeSpec:
      # A COMPLETE LLM node: provider + model + api_key + a non-empty
      # prompt_template + required_input_fields + observed schema, carrying TWO
      # pending requirements — an llm_prompt_template AND an llm_model_choice.
      # interpretation_sites then yields TWO pending sites, so execute()'s gate
      # raises (BLOCK). Resolving BOTH reviews marks both requirements
      # status="resolved", clearing every site, so the state has zero sites and the
      # REAL validate_pipeline accepts the complete pipeline -> execute() reaches
      # create_run (PERMIT). The completeness is what matters: an incomplete llm
      # node (e.g. no provider/model) fails the real validate_pipeline graph build,
      # which is why the earlier minimal fixture had to patch validate_pipeline; a
      # complete node does not.
      return NodeSpec(
          id="rate_node",
          node_type="transform",
          plugin="llm",
          input="rows",
          on_success="main",
          on_error="discard",
          options={
              "provider": "openrouter",
              "model": MODEL,
              "api_key": "test-key-literal",
              "prompt_template": PROMPT,
              "required_input_fields": ["text"],
              "schema": {"mode": "observed"},
              INTERPRETATION_REQUIREMENTS_KEY: [
                  {
                      "id": "pt",
                      "kind": InterpretationKind.LLM_PROMPT_TEMPLATE.value,
                      "user_term": "llm_prompt_template:rate_node",
                      "status": "pending",
                      "draft": PROMPT,
                      "event_id": None,
                      "accepted_value": None,
                      "accepted_artifact_hash": None,
                      "resolved_prompt_template_hash": None,
                  },
                  {
                      "id": "mc",
                      "kind": InterpretationKind.LLM_MODEL_CHOICE.value,
                      "user_term": "llm_model_choice:rate_node",
                      "status": "pending",
                      "draft": MODEL,
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


  async def _persist_state_with_unresolved_node(
      sessions_service: SessionServiceImpl,
      composer: ComposerServiceImpl,
      tmp_path: Path,
  ) -> tuple[UUID, UUID, UUID, UUID, CompositionState]:
      """Seed a session + a COMPLETE LLM pipeline (text source + llm node + json
      sink + edge) AND the two matching pending, resolvable events
      (llm_prompt_template + llm_model_choice). The complete shape is what lets the
      REAL validate_pipeline pass once both reviews resolve. Returns
      (session_id, state_id, pt_event_id, mc_event_id, state)."""
      src = tmp_path / "blobs" / "input.txt"
      src.parent.mkdir(parents=True, exist_ok=True)
      src.write_text("hello world\n", encoding="utf-8")
      out = tmp_path / "outputs" / "out.jsonl"
      out.parent.mkdir(parents=True, exist_ok=True)
      state = CompositionState(
          source=None,
          nodes=(),
          edges=(),
          outputs=(),
          metadata=PipelineMetadata(),
          version=1,
      )
      state = state.with_source(
          SourceSpec(
              plugin="text",
              on_success="rows",
              options={"path": str(src), "column": "text", "schema": {"mode": "observed"}},
              on_validation_failure="discard",
          )
      )
      state = state.with_node(_llm_node())
      state = state.with_output(
          OutputSpec(
              name="main",
              plugin="json",
              options={
                  "path": str(out),
                  "schema": {"mode": "observed"},
                  "mode": "write",
                  "collision_policy": "auto_increment",
              },
              on_write_failure="discard",
          )
      )
      state = state.with_edge(
          EdgeSpec(id="e1", from_node="source", to_node="rate_node", edge_type="on_success", label=None)
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
              edges=state_dict["edges"],
              outputs=state_dict["outputs"],
              metadata_=state_dict["metadata"],
              is_valid=True,
          ),
          provenance="tool_call",
      )
      # Create BOTH resolvable pending events through the writer boundary. The PT
      # event goes through the requirement-checked branch (it requires
      # options.prompt_template == llm_draft AND exactly one pending PT requirement
      # matching user_term — both staged on the node); the model_choice event falls
      # through the writer's else-branch (sessions/service.py:2936), which only
      # confirms an llm transform node with a non-empty prompt_template and does no
      # staged-requirement-shape check. Resolving BOTH marks both requirements
      # resolved, clearing every site (PERMIT).
      pt_event = await sessions_service.create_pending_interpretation_event(
          session_id=session_id,
          composition_state_id=record.id,
          affected_node_id="rate_node",
          tool_call_id=f"backend_auto_surface:{uuid4()}",
          user_term="llm_prompt_template:rate_node",
          kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
          llm_draft=PROMPT,
          model_identifier=composer._model,
          model_version=composer._model,
          provider="anthropic",
          composer_skill_hash=composer._composer_skill_hash,
      )
      mc_event = await sessions_service.create_pending_interpretation_event(
          session_id=session_id,
          composition_state_id=record.id,
          affected_node_id="rate_node",
          tool_call_id=f"backend_auto_surface:{uuid4()}",
          user_term="llm_model_choice:rate_node",
          kind=InterpretationKind.LLM_MODEL_CHOICE,
          llm_draft=MODEL,
          model_identifier=composer._model,
          model_version=composer._model,
          provider="anthropic",
          composer_skill_hash=composer._composer_skill_hash,
      )
      # InterpretationEventRecord.id is a UUID (contracts/composer_interpretation.py:210).
      return session_id, record.id, pt_event.id, mc_event.id, state


  @pytest.mark.asyncio
  async def test_unresolved_card_blocks_run_resolving_permits(
      tmp_path: Path,
      sessions_service: SessionServiceImpl,
  ) -> None:
      composer = _composer(tmp_path, sessions_service)
      execution_service = _build_execution_service(tmp_path, sessions_service)
      session_id, _state_id, pt_event_id, mc_event_id, _state = await _persist_state_with_unresolved_node(
          sessions_service, composer, tmp_path
      )

      # 2. run-time gate BLOCKS: execute() raises on the unresolved placeholder.
      with pytest.raises(UnresolvedInterpretationPlaceholderError):
          await execution_service.execute(session_id=session_id)

      # 3. resolve BOTH pending cards as accepted-as-drafted (the prompt_template
      # and the model_choice). Resolving only one leaves a pending site and the
      # gate keeps raising; both must clear for the state to reach PERMIT.
      for event_id in (pt_event_id, mc_event_id):
          await sessions_service.resolve_interpretation_event(
              session_id=session_id,
              event_id=event_id,
              choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
              amended_value=None,
              actor="user:alice",
          )

      # 4. with BOTH reviews resolved, the interpretation gate PERMITS. The gate
      # (execution/service.py:515-529) runs FIRST and is fully REAL here — it now
      # sees zero sites. The REAL validate_pipeline then runs (NO patch) against the
      # complete LLM pipeline (text source + llm node + json sink + edge) and
      # accepts it; only _run_pipeline is stubbed, to avoid a live engine run.
      # create_run still runs for real, so execute() returns a real run_id.
      with patch.object(execution_service, "_run_pipeline"):
          run_id = await execution_service.execute(session_id=session_id)
      assert run_id is not None
  ```

  The test is drop-in: `_build_execution_service` (inlined above) constructs a real
  `ExecutionServiceImpl` whose `session_service` is the REAL `SessionServiceImpl`
  (so `execute()`'s `get_current_state(session_id)` at `~execution/service.py:484`
  loads the persisted state and the interpretation gate at `:515-529` sees it),
  with a REAL `WebSettings` and the REAL `yaml_generator` so the PERMIT path can
  run the actual `validate_pipeline` against the persisted state. Only the `loop`
  is mocked, and `_call_async` is bridged to a synchronous run (mirroring the
  canonical `service` fixture's `_call_async` bridge at
  `tests/unit/web/execution/test_service.py`). The interpretation gate fires
  BEFORE `validate_pipeline` / `generate_yaml` / `create_run`, so the BLOCK
  assertion is fully REAL and needs no engine. For the PERMIT assertion (step 4)
  the test runs the REAL `validate_pipeline` — no patch — because the complete LLM
  node (provider + model + api_key + prompt_template + a real `text` source and
  `json` sink wired by an edge) builds a valid runtime graph end-to-end; only
  `_run_pipeline` is stubbed, to avoid spinning up a live engine. `create_run`
  still runs for real against the in-memory session engine, so `execute()` returns
  a real `run_id`. `resolve_interpretation_event` reads each pending event's
  `llm_draft` as the accepted value when `choice == ACCEPTED_AS_DRAFTED`
  (`sessions/service.py:3295-3300`), so `amended_value=None` is correct; the PT
  acceptance gate passes because `accepted_value == llm_draft ==
  options.prompt_template`. Resolving BOTH reviews marks the `llm_prompt_template`
  AND `llm_model_choice` requirements `status="resolved"`, so step 4's
  `interpretation_sites` returns empty and `materialize_state_for_execution`
  returns a real state, not `InterpretationReviewPending`.

  Run to verify:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/test_guided_interpretation_run_backstop.py -x -q
  ```
  Expected: with the surfacer/gate already in place, this test PASSES as written
  (the BLOCK and PERMIT shapes are pinned above). If it fails, the failure is
  diagnostic: a failing BLOCK means the node lost one of its two pending
  requirements (`llm_prompt_template` / `llm_model_choice`) so no site is left to
  block on; a failing PERMIT means a review did not resolve (check that each
  `llm_draft == options.prompt_template` / `options.model` and the two
  `user_term`s are `"llm_prompt_template:rate_node"` and
  `"llm_model_choice:rate_node"`) — or the complete pipeline did not pass the REAL
  `validate_pipeline` (read the raised `ValidationError`). The interpretation gate
  stays REAL in both branches (no mock of `materialize_state_for_execution`), and
  `validate_pipeline` is REAL in the PERMIT branch (no patch).

- [ ] **Step 3: Confirm it passes (no production change expected).**
  No production change is needed — the run-time gate
  (`execution/service.py:515-529`) and the surfacer (P3.1) already exist, so the
  test from Step 2 passes as written. The only work is the fixture shape above: a
  complete LLM node that yields two pending sites — `llm_prompt_template` and
  `llm_model_choice` (BLOCK) — and clears to zero sites once BOTH reviews resolve,
  at which point the REAL `validate_pipeline` accepts the complete pipeline
  (PERMIT). With the fixture in place:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/composer/test_guided_interpretation_run_backstop.py -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 4: Commit.**
  ```
  cd /home/john/elspeth && git add tests/integration/web/composer/test_guided_interpretation_run_backstop.py && git commit -m "$(cat <<'EOF'
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
  branch, `:1252-1320`)
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts` (`respondGuided`,
  `:1226`)
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`
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
  (`sessionStore.ts:143`); `GuidedTurn` disabled prop (`ChatPanel.tsx:1319`).
- Produces (NEW, owned by P3): `GuidedInterpretationReviews` React component
  with props `{ sessionId: string }`; a derived `hasPendingInterpretations`
  boolean consumed by the guided branch to disable `GuidedTurn`.

**Why the guided branch and not `GuidedTurn`:** `GuidedTurn.tsx`'s
`interpretation_review` case is **dead code** — the guided branch
(`ChatPanel.tsx:1252`) renders only `guidedNextTurn`, never a backend-emitted
`interpretation_review` turn (the backend has no such `TurnType`; D12). The
projection path is the `pendingBySession` store, exactly as
`TutorialTurn2bShowBuilt` already does it for the big-bang tutorial.

- [ ] **Step 1: Add the store refresh after a guided respond (the data is never
  fetched today).**
  `respondGuided` (`sessionStore.ts:1226`) atomically replaces the four guided
  wire fields but never refreshes `interpretationEventsStore`, so a pending card
  the backend just created via P3.2 is invisible. Mirror the freeform paths
  (`sessionStore.ts:645/753/1040`).

      Write the failing tests first — extend the sessionStore guided test:
  ```
  cd /home/john/elspeth && grep -rln "respondGuided" src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts
  ```
      Read the closest existing `respondGuided` test (the happy path,
      `sessionStore.guided.test.ts:231-256`) and add tests asserting that a successful
      `respondGuided` **awaits** `refreshAll` on the interpretation store before it
      clears `guidedResponsePending` (spy on
      `useInterpretationEventsStore.getState().refreshAll`). The test file already
      mocks `@/api/client` (`:14-37`, `respondGuided: vi.fn()`) and pre-seeds the
      active session via `useSessionStore.setState({ activeSessionId: "sess-1" })`;
      `useInterpretationEventsStore` is NOT mocked, so spying on its `getState` works.
  Add the import `import { useInterpretationEventsStore } from
  "@/stores/interpretationEventsStore";` near the top, then:

  ```ts
      it("awaits interpretation-event refresh after a successful guided respond", async () => {
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
       expect(useSessionStore.getState().guidedResponsePending).toBe(false);
      });

      it("keeps submit disabled while the pending-card refresh is deferred", async () => {
        const { respondGuided } = await import("@/api/client");
        (respondGuided as ReturnType<typeof vi.fn>).mockResolvedValueOnce(
          sampleRespondResponse,
        );
        let releaseRefresh!: () => void;
        const refreshAll = vi.fn(
          () =>
            new Promise<void>((resolve) => {
              releaseRefresh = resolve;
            }),
        );
        vi.spyOn(useInterpretationEventsStore, "getState").mockReturnValue({
          ...useInterpretationEventsStore.getState(),
          refreshAll,
        });
        useSessionStore.setState({ activeSessionId: "sess-1" });

        const promise = useSessionStore.getState().respondGuided({
          chosen: ["csv"],
          edited_values: null,
          custom_inputs: null,
          accepted_step_index: null,
          edit_step_index: null,
          control_signal: null,
        });

        await vi.waitFor(() => expect(refreshAll).toHaveBeenCalledWith("sess-1"));
        expect(useSessionStore.getState().guidedResponsePending).toBe(true);
        releaseRefresh();
        await promise;
        expect(useSessionStore.getState().guidedResponsePending).toBe(false);
      });
      ```

  Run to fail:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm test -- --run sessionStore
  ```
      Expected: the new tests fail (refreshAll not called / pending clears before
      refresh resolves).

      Make `refreshInterpretationEventsForSession` awaitable and then await it in
      `respondGuided` as part of the success barrier. First change the helper:
      ```ts
      async function refreshInterpretationEventsForSession(sessionId: string): Promise<void> {
        await useInterpretationEventsStore.getState().refreshAll(sessionId);
      }
      ```
      Update the helper comment: it is idempotent, but no longer fire-and-forget for
      guided `respondGuided`; the guided submit must remain disabled until the
      pending-card projection refresh completes. Existing freeform callers may use
      `void refreshInterpretationEventsForSession(...)` if they intentionally do not
      block their UI.

      Then update `respondGuided`: the success branch is currently a single atomic
      `set({...})` carrying all four wire fields **and** `guidedResponsePending: false`
      (`sessionStore.ts:1246-1252`). Remove `guidedResponsePending: false` from that
      fused set so it sets only the four wire fields, await the card refresh while the
      pending flag is still true, re-check the active session, then add a trailing
      `set({ guidedResponsePending: false })`:
      ```ts
            set({
              guidedSession: response.guided_session,
              guidedNextTurn: response.next_turn,
              guidedTerminal: response.terminal,
              compositionState: response.composition_state,
            });
            // B1 (spec §5/D12): backend-surfaced pending cards must be in
            // interpretationEventsStore before guidedResponsePending clears, otherwise
            // the submit button can briefly re-enable before the card-block arrives.
            await refreshInterpretationEventsForSession(requestedSessionId);
            if (get().activeSessionId !== requestedSessionId) {
              return;
            }
            set({ guidedResponsePending: false });
      ```
      Keep the `catch` branch unchanged: it clears `guidedResponsePending` only after
      confirming the active session is still the requested one. Confirm
      `refreshInterpretationEventsForSession` is in scope in `sessionStore.ts`; it is
      defined at `:143`.

  > Name note: the test spies on `refreshAll` while the impl calls
  > `refreshInterpretationEventsForSession` — these are consistent.
  > `refreshInterpretationEventsForSession(sessionId)` (`sessionStore.ts:143`) is a
  > thin wrapper that delegates to
  > `useInterpretationEventsStore.getState().refreshAll(sessionId)` (`:144`), so the
  > spy on `refreshAll(sessionId)` observes exactly the wrapper's one call.

  Run to pass:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm test -- --run sessionStore
  ```
  Expected: pass.

- [ ] **Step 2: Write the failing test for the new projection component.**
  Create `GuidedInterpretationReviews.test.tsx`. Seed
  `useInterpretationEventsStore` with one pending `user_approved` event for a
  session and assert the component renders an `InterpretationReviewTurn` (by its
  accessible region role / a stable testid) and exposes the pending count.
  Mirror `TutorialTurn2bShowBuilt.test.tsx` for the store-seeding pattern:
  ```
  cd /home/john/elspeth && sed -n '1,60p' src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx
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
      // With one event there are TWO regions: the wrapper <section
      // aria-label="Assumptions to review"> (a named region) AND the inner
      // InterpretationReviewTurn's own <section role="region"> — so a singular
      // getByRole("region") would throw "Found multiple elements", and
      // InterpretationReviewTurn also carries role="status" so that role is
      // ambiguous too. Assert via the wrapper count line instead, the same way
      // the sibling TutorialTurn2bShowBuilt.test.tsx does (getByText concatenates
      // only the <p>'s direct text-node children, so it matches uniquely).
      expect(screen.getByText("1 assumption to review")).toBeInTheDocument();
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
  cd /home/john/elspeth && grep -n "interface InterpretationEvent" -A 30 src/elspeth/web/frontend/src/types/interpretation.ts
  ```

  Run to fail:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm test -- --run GuidedInterpretationReviews
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
  cd /home/john/elspeth/src/elspeth/web/frontend && npm test -- --run GuidedInterpretationReviews
  ```
  Expected: pass.

- [ ] **Step 4: Wire the component + blocking into the ChatPanel guided branch.**
  In `ChatPanel.tsx`, inside the guided branch (`:1252`), import the new symbols
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
  (`:1316-1320`) with:
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
  cd /home/john/elspeth && grep -rln "chat-panel--guided\|GuidedTurn\|guidedNextTurn" src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
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
  cd /home/john/elspeth/src/elspeth/web/frontend && npm test -- --run ChatPanel
  ```
  Expected after Step 4: pass.

- [ ] **Step 6: Frontend gates — typecheck + build + full vitest + E2E (§9.2).**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run typecheck && npm test -- --run && npm run build && npm run test:e2e
  ```
  Also run `npm run test:e2e:staging` (from the same `src/elspeth/web/frontend`
  dir) when `STAGING_BASE_URL` and staging credentials are available, otherwise
  report it as operator-env blocked rather than silently skipping it.
  Expected: typecheck clean, all vitest pass, build succeeds, `npm run test:e2e`
  passes — P3.6 edits the live guided render/advancement path (ChatPanel.tsx +
  sessionStore.ts `respondGuided`), exactly what E2E covers.

- [ ] **Step 7: Commit.**
  ```
  cd /home/john/elspeth && git add src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.tsx src/elspeth/web/frontend/src/components/chat/guided/GuidedInterpretationReviews.test.tsx src/elspeth/web/frontend/src/stores/sessionStore.ts src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx && git commit -m "$(cat <<'EOF'
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

- [ ] **Step 1: Backend lint + types (§9.2 scope).**
  ```
  cd /home/john/elspeth && uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/ && uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/ && uv run mypy src/ elspeth-lints/src/
  ```
  Expected: all clean. Fix any finding at the boundary (not by suppression).
  This is the full §9.2 ruff/mypy scope (00-overview.md:68-70), not just the
  P3-touched files — `mypy src/` covers `src/elspeth/web/composer/protocol.py`
  (the `ComposerService` Protocol method added in P3.2 Step 3b), which the
  narrower per-file mypy line omitted.

- [ ] **Step 2: Targeted backend suite (surfacer + route + backstop).**
  ```
      cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_surface_pending_interpretation_reviews.py tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py tests/integration/web/composer/test_guided_interpretation_run_backstop.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -q
  ```
  Expected: all pass (the existing freeform dispatch test must stay green — the
  surfacer is additive and does not touch the freeform surface+gate pair).

- [ ] **Step 3: elspeth-lints trust gates (B1 adds a backend interpretation
  surfacer that reads persisted Tier-3 node options).**
  ```
  cd /home/john/elspeth && PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_tier.tier_model,trust_boundary.tests,trust_boundary.scope,trust_boundary.tier,'composer/*' --root src/elspeth
  ```
  Expected: no NEW displacement attributable to P3. If the surfacer trips a
  tier_model entry (it reads `node.options`/`source.options`, already-persisted
  Tier-3 config), state it in the commit/handoff per the gate-debt doctrine —
  do not blind-sign.

- [ ] **Step 4: wardline taint gate (the surfacer consumes persisted
  composition state and writes interpretation rows).**
  ```
  cd /home/john/elspeth && wardline scan . --fail-on ERROR
  ```
  Expected: exit 0. Fix any finding at the boundary.

- [ ] **Step 5: Commit any gate-driven fixups (if Steps 1-4 required edits).**
  ```
      cd /home/john/elspeth && git add src/elspeth/web/composer/service.py src/elspeth/web/composer/protocol.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/composer/test_surface_pending_interpretation_reviews.py tests/integration/web/composer/guided/test_guided_commit_surfaces_reviews.py tests/integration/web/composer/test_guided_interpretation_run_backstop.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py && git commit -m "$(cat <<'EOF'
chore(composer/guided): gate fixups for B1 interpretation surfacing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```
  (Skip if the tree is clean after Steps 1-4.)

---
