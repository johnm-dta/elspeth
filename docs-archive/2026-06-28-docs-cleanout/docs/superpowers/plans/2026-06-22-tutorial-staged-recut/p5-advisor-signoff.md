> **Part of the [Tutorial Staged Recut plan](./00-overview.md).** Read the [overview](./00-overview.md) first — it holds the Global Constraints (§9.2 gate commands) and the "use VERBATIM" Shared Interfaces every task depends on. Phases execute **P0 → P7 in order**.

## Phase P5 — Advisor sign-off gate (B3/D13) — profile-gated + UNAVAILABLE escape

> **Cross-phase dependencies (consumed by name, not created here):**
> - `WorkflowProfile` (+ `EMPTY_PROFILE`, `TUTORIAL_PROFILE`) and `GuidedSession.profile`
>   / `GuidedSession.advisor_checkpoint_passes_used` — **owned by P0**
>   (`composer/guided/profile.py`, `composer/guided/state_machine.py`).
> - `GuidedStep.STEP_4_WIRE`, `TurnType.CONFIRM_WIRING` — **owned by P1**
>   (`composer/guided/protocol.py`).
> - `handle_step_4_wire_confirm(...)` step handler — **owned by P1.6**
>   (`composer/guided/steps.py`).
> - The skeleton `STEP_4_WIRE` dispatch branch in `_dispatch_guided_respond`, the
>   post-accept wire-turn emission, and the `GET /api/sessions/{session_id}/guided`
>   rebuild branch — **owned by P1.6** (`sessions/routes/_helpers.py`,
>   `sessions/routes/composer/guided.py`). P2 upgrades the payload/rebuild shape;
>   P5 mutates the existing terminal-gate branch for advisor sign-off.
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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py -q`
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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py -q`
  Expected: `1 passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py && git commit -m "feat(composer): declare public run_signoff_checkpoint Protocol method (P5.1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.2: Implement `run_signoff_checkpoint` on `ComposerServiceImpl` (delegation)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`ComposerServiceImpl`, add method directly above `_run_advisor_checkpoint` at :4221)
- Create: `tests/unit/web/composer/test_run_signoff_checkpoint_impl.py`

**Interfaces:**
- Produces (public impl method, delegates):
  `async def run_signoff_checkpoint(self, *, state, session_id, recorder, progress=None) -> AdvisorCheckpointVerdict`
  → `return await self._run_advisor_checkpoint(phase="end", state=state, session_id=session_id, recorder=recorder, progress=progress)`
- Consumes: existing private `ComposerServiceImpl._run_advisor_checkpoint` (service.py:4221), `AdvisorCheckpointVerdict` (service.py:4707).

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
  from elspeth.web.composer.service import (
      _ADVISOR_UNAVAILABLE_USER_DETAIL,
      AdvisorCheckpointVerdict,
      ComposerServiceImpl,
              )
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
          return_value=AdvisorCheckpointVerdict(
              ok=False,
              blocking=False,
              findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
          )
      )
      await service.run_signoff_checkpoint(state=_state(), session_id=None, recorder=None)
      assert service._run_advisor_checkpoint.await_args.kwargs["progress"] is None
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_impl.py -q`
  Expected failure: `AttributeError: 'ComposerServiceImpl' object has no attribute 'run_signoff_checkpoint'`.
- [ ] **Step 3: Add the delegation method.**
  In `src/elspeth/web/composer/service.py`, directly ABOVE `async def _run_advisor_checkpoint(` (currently at :4221), insert:
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
  (Note: `BufferingRecorder`, `ComposerProgressSink`, `CompositionState`, and `AdvisorCheckpointVerdict` are all already in scope in service.py — `BufferingRecorder` via the `composer.audit` import at :62, `AdvisorCheckpointVerdict` is defined in-module at :4707, `ComposerProgressSink`/`CompositionState` are module imports used throughout.)
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_impl.py -q`
  Expected: `2 passed`.
- [ ] **Step 5: Run the existing advisor-checkpoint suite to confirm no regression.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py -q`
  Expected: all existing tests still `passed` (the impl only adds a method).
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_run_signoff_checkpoint_impl.py && git commit -m "feat(composer): implement run_signoff_checkpoint delegating to END checkpoint (P5.2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.3: Add the verdict-class classifier + redirect helper (`classify_signoff_verdict`)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — add a `failure_class` field to `AdvisorCheckpointVerdict` (:4707) and set it in `_run_advisor_checkpoint`'s `except` path (:4252-4272). See Step 1a. WITHOUT this the classifier below cannot tell a malformed response from a transport outage (both are currently `ok=False`), so malformed would fail OPEN into the escape — the D4/B2 defect.
- Create: `src/elspeth/web/composer/guided/signoff.py` (new pure module — no `self`, importable by both `_helpers.py` and tests; mirrors how `_advisor_signoff_blocked_validation` is module-scope data with no `self` dependency)
- Create: `tests/unit/web/composer/guided/test_signoff_classifier.py`
- Modify: `tests/unit/web/composer/test_advisor_checkpoint.py` — add unavailable-vs-malformed redaction/classification regressions for the exhausted-retry path.

**Interfaces:**
- Produces (canonical names, verbatim):
  - `class SignoffOutcome(StrEnum)` with members `COMPLETE = "complete"`, `REVISE = "revise"`, `BLOCKED_FLAGGED = "blocked_flagged"`, `BLOCKED_UNAVAILABLE = "blocked_unavailable"`, `ESCAPE_UNAVAILABLE = "escape_unavailable"`.
  - `@dataclass(frozen=True, slots=True) class SignoffDecision` with fields `outcome: SignoffOutcome`, `reason: str | None`, `findings_text: str`, `passes_delta: int`.
  - `def classify_signoff_verdict(verdict: AdvisorCheckpointVerdict, *, passes_used: int, max_passes: int) -> SignoffDecision`.
- Consumes: `AdvisorCheckpointVerdict` (`composer/service.py:4707`), now carrying `failure_class: Literal["none","unavailable","malformed"]` (added in Step 1a). Only the exact value `"unavailable"` is escapable; `"none"`, `"malformed"`, an unknown value, or any omitted/default value on an `ok=False` verdict fails closed.

Decision logic (D13 matrix; `is_last_pass = (passes_used + 1) >= max_passes`). CRITICAL: `_run_advisor_checkpoint` collapses EVERY exception to `ok=False` (including a re-raised malformed/parse error), so the classifier MUST split `not verdict.ok` on `verdict.failure_class` — `(ok, blocking)` alone cannot tell malformed (fail-closed) from a transport outage (escapable):
- CLEAN (`verdict.ok and not verdict.blocking`) → `COMPLETE` (reason `None`, `passes_delta=1`).
- FLAGGED (`verdict.ok and verdict.blocking`) — a quality verdict, fail-closed:
  - not last pass → `REVISE` (`passes_delta=1`, findings carried).
  - last pass → `BLOCKED_FLAGGED` (`reason="exhausted"`, `passes_delta=1`).
- MALFORMED / DEFAULT / UNKNOWN (`not verdict.ok and verdict.failure_class != "unavailable"`) — fail-closed exactly like FLAGGED, **never** the escape:
  - not last pass → `REVISE`; last pass → `BLOCKED_FLAGGED` (`reason="exhausted"`).
- UNAVAILABLE (`not verdict.ok and verdict.failure_class == "unavailable"`) — genuine outage:
  - not last pass → `REVISE` (`passes_delta=1`, findings carried — re-emit "advisor could not be reached; retry").
  - last pass → `ESCAPE_UNAVAILABLE` (`reason="unavailable"`, `passes_delta=1`) — the differentiated audited escape is *offered* and persisted. Only a later route-validated acknowledgement against that persisted marker may stamp COMPLETED-without-signoff (Task P5.5); same-request or raw `custom_inputs` pre-acknowledgements are ignored.

- [ ] **Step 1a: Add `failure_class` to `AdvisorCheckpointVerdict` + classify the exception in `_run_advisor_checkpoint` (`service.py`).**
  In `AdvisorCheckpointVerdict` (`service.py:4707`) add a defaulted field (`Literal` is already imported in service.py):
  ```python
      failure_class: Literal["none", "unavailable", "malformed"] = "none"
  ```
  The default keeps every existing construction valid (CLEAN/FLAGGED set `ok=True` and never read it). In `_run_advisor_checkpoint` (`service.py:4252-4272`) the `except Exception` retry loop currently returns `ok=False` for EVERY exception class — replace the final exhausted-retries return with a classified one:
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
      findings_text = (
          _ADVISOR_UNAVAILABLE_USER_DETAIL
          if failure_class == "unavailable"
          else "advisor response was malformed"
      )
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          failure_class=failure_class,
          findings_text=findings_text,
      )
  ```
  Resolve the EXACT provider exception types against the live LLM client when implementing — the name set above is a transport-class allowlist; everything not on it fails closed as `malformed`. `findings_text` must never contain provider exception class names, messages, URLs, credentials, or SDK text. Preserve the existing `_ADVISOR_UNAVAILABLE_USER_DETAIL` redaction string for unavailable failures and classify the raw exception only into `failure_class`.

  Add a regression to `tests/unit/web/composer/test_advisor_checkpoint.py` that forces the exhausted-retry path with a provider exception such as `TimeoutError("provider deadline details")` and asserts:
  ```python
      assert verdict.failure_class == "unavailable"
      assert verdict.findings_text == _ADVISOR_UNAVAILABLE_USER_DETAIL
      assert "TimeoutError" not in verdict.findings_text
      assert "provider deadline details" not in verdict.findings_text
  ```
  Add a malformed exception companion (`ValueError("raw parse failure")`) and assert `failure_class == "malformed"`, `findings_text == "advisor response was malformed"`, and the raw exception text is absent.
- [ ] **Step 1b: Confirm the verdict field exists.**
  `cd /home/john/elspeth && uv run python -c "from elspeth.web.composer.service import AdvisorCheckpointVerdict; print(AdvisorCheckpointVerdict(ok=False, blocking=False, failure_class='malformed', findings_text='x').failure_class)"`
  Expected: prints `malformed`.
- [ ] **Step 1: Write the failing classifier test.**
  Create `tests/unit/web/composer/guided/test_signoff_classifier.py`:
  ```python
  """Phase P5.3 — pure D13 verdict-class classifier for the wire-stage sign-off."""

  from __future__ import annotations

  from typing import Any, cast

  from elspeth.web.composer.guided.signoff import (
      SignoffOutcome,
      classify_signoff_verdict,
  )
  from elspeth.web.composer.service import (
      _ADVISOR_UNAVAILABLE_USER_DETAIL,
      AdvisorCheckpointVerdict,
  )


  def _clean() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN: good")


  def _flagged() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: prompt sees no row field")


  def _unavailable() -> AdvisorCheckpointVerdict:
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          failure_class="unavailable",
          findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
      )


  def _malformed() -> AdvisorCheckpointVerdict:
      # The advisor returned output the call core could not parse -> re-raised ->
      # caught -> ok=False with failure_class="malformed". Must FAIL CLOSED (D4/B2).
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          failure_class="malformed",
          findings_text="advisor response was malformed",
      )


  def _default_none_failure() -> AdvisorCheckpointVerdict:
      # ok=False with the default failure_class="none" is malformed/blocked by policy.
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          findings_text="advisor response was malformed",
      )


  def _unknown_failure_class() -> AdvisorCheckpointVerdict:
      # Forward-compat guard: only the exact value "unavailable" may escape.
      return AdvisorCheckpointVerdict(
          ok=False,
          blocking=False,
          failure_class=cast(Any, "unknown"),
          findings_text="advisor response was malformed",
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


  def test_default_none_failure_class_fails_closed_on_last_pass() -> None:
      # ok=False + default/none is not a genuine outage; it fails closed.
      d = classify_signoff_verdict(_default_none_failure(), passes_used=2, max_passes=3)
      assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
      assert d.reason == "exhausted"


  def test_unknown_failure_class_fails_closed_on_last_pass() -> None:
      # Future/unknown classes must not accidentally become escapable.
      d = classify_signoff_verdict(_unknown_failure_class(), passes_used=2, max_passes=3)
      assert d.outcome is SignoffOutcome.BLOCKED_FLAGGED
      assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
      assert d.reason == "exhausted"


  def test_flagged_never_yields_an_escape() -> None:
      # A FLAG can never reach the unavailable escape — only BLOCKED_FLAGGED.
      d = classify_signoff_verdict(_flagged(), passes_used=2, max_passes=3)
      assert d.outcome is not SignoffOutcome.ESCAPE_UNAVAILABLE
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/guided/test_signoff_classifier.py -q`
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
      # _run_advisor_checkpoint collapses exceptions to ok=False (service.py:
      # 4252-4272). So (ok, blocking) ALONE cannot tell malformed/default/unknown
      # failures from a genuine outage; only the exact failure_class "unavailable"
      # is allowed to reach the budget-exhausted escape.
      if verdict.failure_class == "unavailable":
          if is_last_pass:
              return SignoffDecision(
                  outcome=SignoffOutcome.ESCAPE_UNAVAILABLE, reason="unavailable", findings_text=findings, passes_delta=1
              )
          return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)

      # MALFORMED, NONE/default, or any unknown future class: fail closed exactly
      # like FLAGGED. This prevents a new/omitted failure_class from silently
      # becoming an audited escape.
      if is_last_pass:
          return SignoffDecision(
              outcome=SignoffOutcome.BLOCKED_FLAGGED, reason="exhausted", findings_text=findings, passes_delta=1
          )
      return SignoffDecision(outcome=SignoffOutcome.REVISE, reason=None, findings_text=findings, passes_delta=1)
  ```
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/guided/test_signoff_classifier.py tests/unit/web/composer/test_advisor_checkpoint.py -q -k "signoff or failure_class or advisor_checkpoint"`
  Expected: classifier tests pass and the advisor-checkpoint regression covers both `failure_class="unavailable"` with `_ADVISOR_UNAVAILABLE_USER_DETAIL` and `failure_class="malformed"` without raw provider exception text.
- [ ] **Step 5: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/service.py src/elspeth/web/composer/guided/signoff.py tests/unit/web/composer/test_advisor_checkpoint.py tests/unit/web/composer/guided/test_signoff_classifier.py && git commit -m "feat(composer/guided): add pure D13 sign-off verdict classifier (P5.3)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.4: Thread the `ComposerService` handle into `_dispatch_guided_respond`

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (`_dispatch_guided_respond` signature at :2571; add a `composer_service` param)
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py` (the `_dispatch_guided_respond(...)` call at :1116-1133; pass `request.app.state.composer_service`) — **note:** `routes/composer.py` was decomposed into the `routes/composer/` package; the guided respond route now lives in `composer/guided.py`.
- Create: `tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py`

**Interfaces:**
- Produces: `_dispatch_guided_respond(..., payload_store: Any, model: str, temperature: float | None, seed: int | None, composer_service: ComposerService | None = None, advisor_checkpoint_max_passes: int | None = None)` — the live dispatcher already requires `payload_store` before `model`; append the two new **keyword-only** params after `seed`, both with SAFE DEFAULTS so pre-P5 callers stay valid.
- Consumes: `ComposerService` Protocol (`composer/protocol.py`), `request.app.state.composer_service` (the `ComposerServiceImpl` wired at `app.py:882`), the existing `payload_store` argument (`_helpers.py:2585`), `settings.composer_advisor_checkpoint_max_passes` (available at the route call site, `composer/guided.py:1129`).

> This task makes the wire-stage advisor gate (P5.5/P5.6) *possible* by giving the
> pure-routing dispatcher a handle to the service AND the persisted pass budget. It
> does NOT yet call the gate. `advisor_checkpoint_max_passes` is the budget P5.6
> reads as `max_passes` — threading it here (rather than reaching into
> `composer_service._settings`) keeps the dispatcher free of private-attr access.
>
> **Required-kwarg ordering (decision):** both new params take defaults so the
> wire-dispatch `_dispatch` test helper (introduced by P1.6 and payload-upgraded
> by P2) and other pre-P5 direct callers do not break when P5.4 lands. The
> defaults are only compatibility defaults. P5.6 must distinguish the
> profile cases: when `guided.profile.advisor_checkpoints` is false, no service is
> needed and the valid wire stage may complete; when it is true (tutorial mandatory
> sign-off), `composer_service is None`, `advisor_checkpoint_max_passes is None`, or
> `advisor_checkpoint_max_passes <= 0` is an invariant failure and must fail closed
> with an explicit blocked/re-emitted wire turn. Tutorial sign-off must never
> silently skip because a caller omitted the service or budget.

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
      # Safe compatibility default so pre-P5 callers can omit this kwarg.
      # P5.6 fails closed when a tutorial/advisor-checkpoint profile sees None.
      assert param.default is None


  def test_dispatcher_accepts_advisor_checkpoint_max_passes_kwarg() -> None:
      sig = inspect.signature(_dispatch_guided_respond)
      assert "advisor_checkpoint_max_passes" in sig.parameters
      param = sig.parameters["advisor_checkpoint_max_passes"]
      assert param.kind is inspect.Parameter.KEYWORD_ONLY
      assert param.annotation == "int | None"
      # Safe compatibility default only; tutorial/advisor-checkpoint profiles
      # fail closed if the route/test did not thread a concrete positive budget.
      assert param.default is None
  ```
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py -q`
  Expected failure: `AssertionError: assert 'composer_service' in {...}` (params absent).
- [ ] **Step 3: Add the import + params to the dispatcher.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, ensure `ComposerService` is importable for the annotation. If it is not already imported, add at the top of the existing composer-imports block:
  ```python
  from elspeth.web.composer.protocol import ComposerService
  ```
  Then append BOTH params to `_dispatch_guided_respond`'s keyword-only signature, after the existing `seed: int | None,`. Preserve the live `payload_store: Any` argument before `model`:
  ```python
      payload_store: Any,
      model: str,
      temperature: float | None,
      seed: int | None,
      composer_service: ComposerService | None = None,  # compatibility default; tutorial profile fails closed on None
      advisor_checkpoint_max_passes: int | None = None,  # compatibility default; tutorial profile requires positive int
  ) -> tuple[CompositionState, GuidedSession, Any | None]:
  ```
  (Match the existing closing-paren/return-annotation line at :2588-2589; insert only the two new lines above `) -> tuple[...]`. The defaults are load-bearing for compatibility, but P5.6 owns the tutorial-profile fail-closed guard.)
- [ ] **Step 4: Pass the handle + budget from the route.**
  In `src/elspeth/web/sessions/routes/composer/guided.py`, the `_dispatch_guided_respond(...)` call (`:1116-1133`) already passes `payload_store=getattr(request.app.state, "payload_store", None)` before `model`. Keep that line and add the two new kwargs after `seed=settings.composer_seed,`:
  ```python
                        payload_store=getattr(request.app.state, "payload_store", None),
                        model=settings.composer_model,
                        temperature=settings.composer_temperature,
                        seed=settings.composer_seed,
                        composer_service=request.app.state.composer_service,
                        advisor_checkpoint_max_passes=settings.composer_advisor_checkpoint_max_passes,
                    )
  ```
- [ ] **Step 5: Run to pass + confirm the route still imports.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py -q && uv run python -c "import elspeth.web.sessions.routes.composer"`
  Expected: `1 passed`, then a clean import (no output, exit 0).
- [ ] **Step 6: Run the existing guided-respond route + wire-dispatch suites to confirm no caller breakage.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web tests/integration/web/composer/guided/test_wire_dispatch.py -q -k "guided_respond or dispatch_guided or wire"`
  Expected: existing tests `passed`. Because both new params take safe defaults,
  the `tests/integration/web/composer/guided/test_wire_dispatch.py::_dispatch`
  helper (introduced by P1.6 and payload-upgraded by P2, constructing
  `_dispatch_guided_respond(...)` WITHOUT `composer_service` /
  `advisor_checkpoint_max_passes`) still imports and runs. That compatibility path
  is valid only for profiles without mandatory advisor checkpoints; the P5.6 tests
  add the tutorial-profile invariant guard so a missing service/budget cannot turn
  into a silent skip. Named direct callers to confirm still pass (and fix in this
  same commit if any break):
  - `tests/integration/web/composer/guided/test_wire_dispatch.py::_dispatch` (P1.6/P2 wire-dispatch helper; relies on the defaults for non-advisor profile coverage — must NOT require the new kwargs).
  Search for any OTHER in-repo direct caller and fix it here if found:
  `grep -rln "_dispatch_guided_respond(" tests/`.
- [ ] **Step 7: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py && git commit -m "feat(web/sessions): thread ComposerService handle into guided dispatcher (P5.4)

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
- If `passes_used >= max_passes`, `acknowledged_unavailable` is true, and `session.advisor_signoff_escape_offered` is true: return the session unchanged with `COMPLETE` + `reason="unavailable"` **without** calling the provider. This is the only completion-without-signoff path: the server must have persisted an `ESCAPE_UNAVAILABLE` offer on a previous request before the user's acknowledgement can complete.
- Otherwise call `composer_service.run_signoff_checkpoint(...)`, classify, and on every classified pass persist `advisor_checkpoint_passes_used += decision.passes_delta` onto a `dataclasses.replace`'d session.
- If `decision.outcome is ESCAPE_UNAVAILABLE`: leave it `ESCAPE_UNAVAILABLE` and persist `advisor_signoff_escape_offered=True` (the caller emits the escape-offer wire turn). Do **not** honor same-request acknowledgements; a pre-sent client value must not convert the same provider response into completion.

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
  from elspeth.web.composer.service import _ADVISOR_UNAVAILABLE_USER_DETAIL, AdvisorCheckpointVerdict
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
      svc = _service(
          AdvisorCheckpointVerdict(
              ok=False,
              blocking=False,
              failure_class="unavailable",
              findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
          )
      )
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=False,
      )
      assert decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
      assert decision.reason == "unavailable"


  @pytest.mark.asyncio
  async def test_same_request_unavailable_acknowledgement_is_not_honored() -> None:
      session = GuidedSession.initial()
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(
          AdvisorCheckpointVerdict(
              ok=False,
              blocking=False,
              failure_class="unavailable",
              findings_text=_ADVISOR_UNAVAILABLE_USER_DETAIL,
          )
      )
      new_session, decision = await run_wire_signoff(
          session=session, state=_state(), session_id="s1", recorder=None,
          composer_service=svc, max_passes=3, acknowledged_unavailable=True,
      )
      # A client must not pre-send the acknowledgement and complete in the same
      # request that first discovers the advisor outage. The server first emits a
      # persisted escape offer; only a later request may acknowledge it.
      assert decision.outcome is SignoffOutcome.ESCAPE_UNAVAILABLE
      assert decision.reason == "unavailable"
      assert new_session.advisor_signoff_escape_offered is True


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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/guided/test_wire_signoff_runner.py -q`
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
      bound (D16): guided re-entry crosses separate
      ``POST /api/sessions/{session_id}/guided/respond`` HTTP
      requests, so an unpersisted per-compose local would reset to 0 each request
      and never bound the loop. When the budget is already spent on a prior
      request the provider is NOT re-called.

      ``acknowledged_unavailable`` is the route-validated "complete without
      sign-off (advisor unreachable)" acknowledgement. It is honored only when
      this frozen session already carries ``advisor_signoff_escape_offered=True``
      from a prior server-emitted ESCAPE_UNAVAILABLE turn and the persisted
      counter is exhausted. It can NEVER bypass a FLAG (a FLAG never sets the
      escape-offered marker).
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
              # POST /api/sessions/{session_id}/guided/respond request than the
              # one that emitted the offer — which
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

      return new_session, decision
  ```
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/guided/test_wire_signoff_runner.py -q`
  Expected: `8 passed`.
- [ ] **Step 5: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/guided/signoff.py tests/unit/web/composer/guided/test_wire_signoff_runner.py && git commit -m "feat(composer/guided): persisted-counter-bound wire sign-off runner (P5.5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.6: Gate the wire-stage terminal on the profile + sign-off decision (dispatch branch)

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (the `STEP_4_WIRE` branch of `_dispatch_guided_respond`, **created by P1.6** and payload-upgraded by P2; this task adds the profile-gated sign-off before the COMPLETED stamp)
- Create: `tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py`

**Interfaces:**
- Consumes: `GuidedSession.profile` + `.advisor_checkpoint_passes_used` (**P0**), `WorkflowProfile.advisor_checkpoints` (**P0**), `GuidedStep.STEP_4_WIRE` + `TurnType.CONFIRM_WIRING` (**P1**), `handle_step_4_wire_confirm` (**P1.6**, `composer/guided/steps.py`) + `build_step_4_wire_turn` (**P2.4**, final signature already accepts `catalog`/`advisor_findings`/`signoff_outcome`), the `STEP_4_WIRE` dispatch branch (**P1.6**, payload-upgraded by P2), `run_wire_signoff` + `SignoffOutcome` (P5.5), `TerminalState`/`TerminalKind` (`state_machine.py`), the wire `composer_service` handle (P5.4).
- Produces: behaviour — the `STEP_4_WIRE` branch stamps `TerminalState(COMPLETED)` only on `SignoffOutcome.COMPLETE`; emits a revise wire turn on `REVISE`/`ESCAPE_UNAVAILABLE`; sets a fail-closed terminal-less revise turn carrying `_advisor_signoff_blocked_validation` findings on `BLOCKED_FLAGGED`/`BLOCKED_UNAVAILABLE`.

> **Precondition (read before implementing):** P1.6 has already created the
> `STEP_4_WIRE` branch in `_dispatch_guided_respond` whose `CONFIRM_WIRING`
> sub-branch calls `handle_step_4_wire_confirm(...)` and lets it stamp
> `TerminalState(COMPLETED)` on a valid pipeline; P2 has upgraded the turn
> payload/rebuild shape. Do NOT re-create the branch here. This task REPLACES the BODY of
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

  import dataclasses
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


  async def _dispatch(
      session: GuidedSession,
      state,
      svc,
      *,
      max_passes: int | None = 3,
      control=ControlSignal.EXIT_TO_FREEFORM,
      turn_response_override=None,
  ):
      # CONFIRM_WIRING confirm response: no control signal (a plain confirm).
      turn_response = turn_response_override or {
          "chosen": ["confirm"],
          "edited_values": None,
          "custom_inputs": None,
          "accepted_step_index": None,
          "edit_step_index": None,
          "control_signal": None,
      }
      catalog = MagicMock()
      payload_store = MagicMock()
      payload_store.store.return_value = "payload-id"
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
          payload_store=payload_store,
          model="m",
          temperature=None,
          seed=None,
          composer_service=svc,
          advisor_checkpoint_max_passes=max_passes,
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
  async def test_custom_inputs_never_acknowledge_unavailable_escape() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      session = dataclasses.replace(session, advisor_checkpoint_passes_used=2)
      svc = _service(
          AdvisorCheckpointVerdict(
              ok=False,
              blocking=False,
              failure_class="unavailable",
              findings_text="advisor unavailable",
          )
      )
      raw_custom_ack = {
          "chosen": ["confirm"],
          "edited_values": None,
          "custom_inputs": ["complete_without_signoff"],
          "accepted_step_index": None,
          "edit_step_index": None,
          "control_signal": None,
      }
      _state, guided, next_turn = await _dispatch(
          session,
          state,
          svc,
          turn_response_override=raw_custom_ack,
      )
      assert guided.terminal is None
      assert next_turn is not None
      assert next_turn["payload"]["signoff_outcome"] == "escape_unavailable"


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


  @pytest.mark.asyncio
  async def test_tutorial_profile_missing_service_fails_closed_invariant() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      _state, guided, next_turn = await _dispatch(session, state, None)
      assert guided.terminal is None
      assert next_turn is not None
      assert (
          "Advisor sign-off service or pass budget is not configured"
          in str(next_turn["payload"])
      )


  @pytest.mark.asyncio
  async def test_tutorial_profile_missing_budget_fails_closed_invariant() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      svc = _service(AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
      _state, guided, next_turn = await _dispatch(session, state, svc, max_passes=None)
      assert guided.terminal is None
      assert next_turn is not None
      assert "Advisor sign-off service or pass budget is not configured" in str(next_turn["payload"])
      svc.run_signoff_checkpoint.assert_not_awaited()
  ```
  > Fixture note: `make_wire_ready_session_and_state` is the P3-owned helper
  > that builds a STEP_4_WIRE-positioned `GuidedSession` (with the given
  > `profile`) plus a valid `CompositionState`. If P3 has not exported it,
  > create a local minimal version in `tests/unit/web/sessions/routes/_wire_fixtures.py`
  > that constructs a single-source/single-sink valid state and a session with
  > `step=GuidedStep.STEP_4_WIRE`, `history=(<a CONFIRM_WIRING TurnRecord>,)`,
  > and `profile=<arg>`.
- [ ] **Step 2: Run to fail.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected failure: depending on P0–P3 landed state, either an `ImportError` for `EMPTY_PROFILE`/`STEP_4_WIRE`/`make_wire_ready_session_and_state` (cross-phase dep not yet present) or, once those exist, `AssertionError: advisor must NOT be called` / `terminal is None` (the gate logic not yet inserted).
- [ ] **Step 3: Replace the existing confirm body with the profile-gated sign-off.**
  In `src/elspeth/web/sessions/routes/_helpers.py`, inside the `if current_turn_type is TurnType.CONFIRM_WIRING:` sub-branch of the `if current_step is GuidedStep.STEP_4_WIRE:` block (created by P1.6 and payload-upgraded by P2), REPLACE the existing body — which called `handle_step_4_wire_confirm(...)` and let *it* stamp COMPLETED unconditionally on a valid pipeline — with the profile-gated form below. Keep the SAME validate-gate semantics: run `state.validate()` first and re-emit the wire turn (terminal stays `None`) on an invalid pipeline, THEN profile-branch. (Do NOT call `handle_step_4_wire_confirm` here any more — its unconditional stamp would race the tutorial-profile gate; the validate check moves inline so the gate owns the stamp.)
  ```python
              # Validate-gate first (same semantics as P1.6/P2): an invalid pipeline never
              # completes — re-emit the wire turn so the user can reconcile (B6).
              if not state.validate().is_valid:
                  guided, next_turn = _emit_wire_turn(
                      state=state,
                      guided=guided,
                      recorder=recorder,
                      user_id=user_id,
                      payload_store=payload_store,
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

              if (
                  composer_service is None
                  or advisor_checkpoint_max_passes is None
                  or advisor_checkpoint_max_passes <= 0
              ):
                  blocked = _advisor_signoff_blocked_validation(
                      reason="invariant",
                      findings="Advisor sign-off service or pass budget is not configured.",
                  )
                  advisor_findings = (
                      blocked.errors[0].message
                      if blocked.errors
                      else "Advisor sign-off service or pass budget is not configured."
                  )
                  next_turn = build_step_4_wire_turn(
                      state,
                      catalog=catalog,
                      advisor_findings=advisor_findings,
                      signoff_outcome=SignoffOutcome.BLOCKED_UNAVAILABLE.value,
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
                      payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                      emitter="server",
                      composition_version=state.version,
                      actor=user_id,
                  )
                  guided = _replace(guided, history=(*guided.history, new_record))
                  return state, guided, next_turn

              acknowledged_unavailable = (
                  guided.advisor_signoff_escape_offered
                  and guided.advisor_checkpoint_passes_used >= advisor_checkpoint_max_passes
                  and tuple(turn_response.get("chosen") or ()) == ("complete_without_signoff",)
                  and turn_response.get("custom_inputs") is None
              )
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
                  payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
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
  >   `advisor_checkpoint_max_passes: int | None` by **P5.4** (added to the P5.4
  >   signature + call + signature test, sourced from
  >   `settings.composer_advisor_checkpoint_max_passes` at the route call site,
  >   `composer/guided.py:1129`). This task validates it is a positive `int` for
  >   tutorial profiles before assigning it to `max_passes`; do NOT reach into
  >   `composer_service._settings`. If P5.4 has not yet added the param, add it
  >   there first (its run-to-fail names the missing param), then this task consumes it.
  > - `build_step_4_wire_turn` ALREADY accepts `catalog` + `advisor_findings` +
  >   `signoff_outcome` (all optional, defaulting `None`) — that is the FINAL
  >   signature landed by P2.4. Call it verbatim:
  >   `build_step_4_wire_turn(state, catalog=catalog, advisor_findings=..., signoff_outcome=...)`.
  >   Do NOT modify the emitter here.
  > - `generate_yaml`, `TerminalState`, `TerminalKind`, `TurnRecord`,
  >   `stable_hash`, `emit_turn_emitted`, `_store_guided_audit_payload`,
  >   `payload_store`, and `_replace` are already present in `_helpers.py`
  >   (used by sibling guided branches).
  > - Import `_advisor_signoff_blocked_validation` from
  >   `elspeth.web.composer.service`; it is module-scope and builds the
  >   non-runnable blocked validation payload for both advisor findings and the
  >   missing-service/budget invariant.
  > - The unavailable escape acknowledgement is NOT read from `custom_inputs`.
  >   `custom_inputs` remains arbitrary Tier-3 text for other guided turns. The
  >   route may pass `acknowledged_unavailable=True` only when the persisted
  >   `GuidedSession` already has `advisor_signoff_escape_offered=True`, its
  >   `advisor_checkpoint_passes_used` counter is exhausted, and the user selected
  >   the server-emitted closed choice `chosen=["complete_without_signoff"]`.
  >   A same-request pre-acknowledgement must re-emit `escape_unavailable`, not
  >   complete the tutorial.
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected: `6 passed`.
- [ ] **Step 5: Run the wider guided-dispatch + advisor suites for no regression.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py tests/unit/web/composer/guided -q -k "signoff or wire or advisor"`
  Expected: all `passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py && git commit -m "feat(web/sessions): profile-gate the STEP_4_WIRE terminal on the advisor sign-off (P5.6)

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
- Consumes: `_advisor_signoff_blocked_validation(reason=..., findings=...)` (`composer/service.py:4989`) — for the blocked terminal's findings text; `_build_invocation`/`ComposerToolRecorder` (`composer/guided/audit.py:37`).

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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py -q`
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
  >   reading `recorder.invocations[-1].tool_name` after a dispatch (the
  >   `BufferingRecorder.invocations` property, `composer/audit.py:225-229`).
- [ ] **Step 6: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py -q`
  Expected: all `passed` (audit-name tests + the P5.6 gate tests still green).
- [ ] **Step 7: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/guided/audit.py src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py && git commit -m "feat(web/sessions): differentiated sign-off audit names + fail-closed blocked findings (P5.7)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.8: REQUEST_ADVISOR whole-pipeline escape — preserve the step-3 chain re-solve

**Files:**
- Modify: `src/elspeth/web/sessions/routes/_helpers.py` (add a `STEP_4_WIRE` + on-demand `REQUEST_ADVISOR` branch; leave the existing STEP_3 `REJECT/REQUEST_ADVISOR` chain re-solve at :3360 untouched)
- Create: `tests/unit/web/sessions/routes/test_request_advisor_escape.py`

**Interfaces:**
- Consumes: `ControlSignal.REQUEST_ADVISOR` (`protocol.py:72`), `run_wire_signoff`/`SignoffOutcome` (P5.5), the existing STEP_3 `solve_chain_with_auto_drop` re-solve (`_helpers.py:3370`).
- Produces: behaviour — a `REQUEST_ADVISOR` control on a `STEP_4_WIRE` `CONFIRM_WIRING` turn runs the whole-pipeline sign-off on-demand (subject to the persisted pass budget) and re-emits the wire turn with findings; a `REQUEST_ADVISOR` at STEP_3 still triggers `solve_chain_with_auto_drop` (regression guard).

> D6/D13: `REQUEST_ADVISOR` is the per-phase on-demand "go to advisor" escape.
> Today it is a REAL chain re-solve at STEP_3 only (`_helpers.py:3360`). This
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
      svc._validate_advisor_arguments = MagicMock(
          side_effect=AssertionError("wire-stage REQUEST_ADVISOR must use run_signoff_checkpoint")
      )
      payload_store = MagicMock()
      payload_store.store.return_value = "payload-id"
      turn_response = {
          "chosen": None, "edited_values": None, "custom_inputs": None,
          "accepted_step_index": None, "edit_step_index": None,
          "control_signal": ControlSignal.REQUEST_ADVISOR,
      }
      recorder = BufferingRecorder()
      _s, guided, next_turn = await _dispatch_guided_respond(
          state=state, guided=session, current_step=GuidedStep.STEP_4_WIRE,
          current_turn_type=TurnType.CONFIRM_WIRING, turn_response=turn_response,
          catalog=MagicMock(), recorder=recorder, user_id="u1",
          data_dir=None, session_engine=None, session_id="s1",
          blob_service=MagicMock(), payload_store=payload_store,
          model="m", temperature=None, seed=None,
          composer_service=svc, advisor_checkpoint_max_passes=3,
      )
      svc.run_signoff_checkpoint.assert_awaited_once()
      svc._validate_advisor_arguments.assert_not_called()
      assert guided.terminal is None  # on-demand review never auto-completes on a FLAG
      assert next_turn is not None
      assert "review this" in next_turn["payload"]["advisor_findings"]
      assert "composer.signoff.revise" in [inv.tool_name for inv in recorder.invocations]


  @pytest.mark.asyncio
  async def test_request_advisor_at_wire_missing_service_or_budget_fails_closed() -> None:
      session, state = make_wire_ready_session_and_state(profile=TUTORIAL_PROFILE)
      payload_store = MagicMock()
      payload_store.store.return_value = "payload-id"
      recorder = BufferingRecorder()
      turn_response = {
          "chosen": None, "edited_values": None, "custom_inputs": None,
          "accepted_step_index": None, "edit_step_index": None,
          "control_signal": ControlSignal.REQUEST_ADVISOR,
      }
      _s, guided, next_turn = await _dispatch_guided_respond(
          state=state, guided=session, current_step=GuidedStep.STEP_4_WIRE,
          current_turn_type=TurnType.CONFIRM_WIRING, turn_response=turn_response,
          catalog=MagicMock(), recorder=recorder, user_id="u1",
          data_dir=None, session_engine=None, session_id="s1",
          blob_service=MagicMock(), payload_store=payload_store,
          model="m", temperature=None, seed=None,
          composer_service=None, advisor_checkpoint_max_passes=None,
      )
      assert guided.terminal is None
      assert next_turn is not None
      assert "Advisor sign-off service or pass budget is not configured" in str(next_turn["payload"])
      assert next_turn["payload"]["signoff_outcome"] == "blocked_unavailable"
      payload_store.store.assert_called()


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
      payload_store = MagicMock()
      payload_store.store.return_value = "payload-id"
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
          blob_service=MagicMock(), payload_store=payload_store,
          model="m", temperature=None, seed=None,
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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_request_advisor_escape.py -q`
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
                  from elspeth.web.composer.guided.audit import emit_signoff_decision
                  from elspeth.web.composer.guided.signoff import (
                      SignoffOutcome,
                      run_wire_signoff,
                      signoff_audit_event_name,
                  )

                  if (
                      composer_service is None
                      or advisor_checkpoint_max_passes is None
                      or advisor_checkpoint_max_passes <= 0
                  ):
                      blocked = _advisor_signoff_blocked_validation(
                          reason="invariant",
                          findings="Advisor sign-off service or pass budget is not configured.",
                      )
                      advisor_findings = (
                          blocked.errors[0].message
                          if blocked.errors
                          else "Advisor sign-off service or pass budget is not configured."
                      )
                      next_turn = build_step_4_wire_turn(
                          state,
                          catalog=catalog,
                          advisor_findings=advisor_findings,
                          signoff_outcome=SignoffOutcome.BLOCKED_UNAVAILABLE.value,
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
                          payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                          emitter="server",
                          composition_version=state.version,
                          actor=user_id,
                      )
                      guided = _replace(guided, history=(*guided.history, new_record))
                      return state, guided, next_turn

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
                  emit_signoff_decision(
                      recorder,
                      event_name=signoff_audit_event_name(decision),
                      outcome=decision.outcome.value,
                      reason=decision.reason,
                      composition_version=state.version,
                      actor=user_id,
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
                      payload_payload_id=_store_guided_audit_payload(payload_store, next_turn["payload"]),
                      emitter="server",
                      composition_version=state.version,
                      actor=user_id,
                  )
                  guided = _replace(guided, history=(*guided.history, new_record))
                  return state, guided, next_turn
  ```
  > The STEP_3 branch at :3360 is LEFT EXACTLY AS-IS. This branch must copy the
  > same fail-closed service/budget guard from P5.6 before calling
  > `run_wire_signoff`, then emit the same differentiated
  > `emit_signoff_decision(signoff_audit_event_name(decision), ...)` event after
  > every real `run_wire_signoff` decision. Confirm the same `max_passes`
  > threading decision from P5.6 (settings kwarg vs service settings) is reused
  > here so both wire-stage advisor calls share the bound.
  > The `emit_turn_emitted` call must use the existing `payload_store` param via
  > `_store_guided_audit_payload(payload_store, next_turn["payload"])`; never
  > emit an empty `payload_payload_id`.
- [ ] **Step 4: Run to pass.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/sessions/routes/test_request_advisor_escape.py -q`
  Expected: `2 passed`.
- [ ] **Step 5: Confirm the STEP_3 re-solve suite is untouched.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web -q -k "step_3 and (advisor or reject or re_solve or chain)"`
  Expected: existing STEP_3 chain re-solve tests still `passed`.
- [ ] **Step 6: Commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/sessions/routes/_helpers.py tests/unit/web/sessions/routes/test_request_advisor_escape.py && git commit -m "feat(web/sessions): wire-stage REQUEST_ADVISOR whole-pipeline escape, step-3 re-solve preserved (P5.8)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.9: Correct the stale "Disabled by default" advisor prose

**Files:**
- Modify: `src/elspeth/web/composer/tools/_dispatch.py` (the advisor tool description at :129)
- Create: `tests/unit/web/composer/test_advisor_tool_prose.py`

**Interfaces:**
- Produces: corrected operator-facing tool prose (no behaviour change). Pins the END sign-off reality contradicting "Disabled by default".

> §B3 (last paragraph): `tools/_dispatch.py:129` says the advisor is "Disabled
> by default" — that contradicts the mandatory-advisor END sign-off this phase
> wires. The advisor on-demand escape budget is real and the END checkpoint is
> profile-gated (always on for the tutorial), so the "Disabled by default" line
> is stale and must go.

- [ ] **Step 1: Write the failing prose test.**
  Create `tests/unit/web/composer/test_advisor_tool_prose.py`. The advisor
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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_advisor_tool_prose.py -q`
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
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_advisor_tool_prose.py -q`
  Expected: `1 passed`.
- [ ] **Step 5: Refresh the plugin/tool source hash if the gate requires it, then commit.**
  `cd /home/john/elspeth && git add src/elspeth/web/composer/tools/_dispatch.py tests/unit/web/composer/test_advisor_tool_prose.py && git commit -m "docs(composer): correct stale 'Disabled by default' advisor prose (P5.9)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"`

---

### Task P5.10: Phase gate sweep — ruff, mypy, full advisor/wire suite

**Files:** none (verification only).

- [ ] **Step 1: ruff on every file this phase touched.**
  `cd /home/john/elspeth && uv run ruff check src/elspeth/web/composer/protocol.py src/elspeth/web/composer/service.py src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/tools/_dispatch.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/routes/composer/guided.py`
  Expected: `All checks passed!`.
- [ ] **Step 2: mypy on the new module + the touched route helper.**
  `cd /home/john/elspeth && uv run mypy src/elspeth/web/composer/guided/signoff.py src/elspeth/web/composer/protocol.py`
  Expected: `Success: no issues found`.
- [ ] **Step 3: Run the full phase test set.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_run_signoff_checkpoint_protocol.py tests/unit/web/composer/test_run_signoff_checkpoint_impl.py tests/unit/web/composer/guided/test_signoff_classifier.py tests/unit/web/composer/guided/test_wire_signoff_runner.py tests/unit/web/sessions/routes/test_dispatch_guided_respond_service_handle.py tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py tests/unit/web/sessions/routes/test_request_advisor_escape.py tests/unit/web/composer/test_advisor_tool_prose.py -q`
  Expected: all `passed` (the full P5 suite green together).
- [ ] **Step 4: Run the inherited advisor-checkpoint regression suite.**
  `cd /home/john/elspeth && uv run python -m pytest tests/unit/web/composer/test_advisor_checkpoint.py -q`
  Expected: all `passed` — the freeform compose-loop END gate is untouched by this phase (the new public method is a thin façade; the wire gate is a separate dispatch surface).
- [ ] **Step 5: wardline trust-boundary scan (the gate touches external-input-adjacent route code).**
  `cd /home/john/elspeth && wardline scan . --fail-on ERROR`
  Expected: exit 0 (clean). The unavailable escape acknowledgement is not read
  from `custom_inputs`; it is accepted only as the server-emitted closed choice
  `chosen=["complete_without_signoff"]` when the persisted
  `advisor_signoff_escape_offered` marker proves a prior `escape_unavailable`
  turn was emitted. The checkpoint forwards no user text into the advisor — fix
  any finding at the boundary if one appears.
- [ ] **Step 6: No commit (verification gate). If any step failed, return to the owning task.**

---
