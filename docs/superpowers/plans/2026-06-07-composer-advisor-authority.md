# Composer Advisor Authority — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the composer advisor a mandatory, model-distinct **final authority** with deterministic (backend-initiated) early + end checkpoints, replacing the reactive self-trigger the weak model never fires.

**Architecture:** Reuse the existing audited advisor call (`_call_advisor_with_audit`) from new backend-initiated checkpoint orchestration. EARLY pass (advisory, once per session) reviews approach; END gate (authoritative) runs as a new repair-branch in `_try_terminate_no_tools` mirroring the proof-repair branch, with a parallel `advisor_passes` budget kept separate from `repair_turns` by gate order. Config makes the advisor mandatory + distinct from the primary, fail-closed at startup.

**Tech Stack:** Python 3.13, FastAPI, Pydantic v2 (`WebSettings`), LiteLLM, pytest+pytest-asyncio+xdist. No DB schema change. Spec: `docs/superpowers/specs/2026-06-07-composer-advisor-authority-design.md` (read §13 amendment — two-tier dropped, frontier-primary pivot).

**Scope note:** This is one subsystem (composer advisor) but a large plan; tasks are sequential with commit points — execute subagent-driven (one task per subagent) so each change stays in a focused context. Deferred to sibling specs: deterministic field-contract checker, reciprocity enforcement, vague_term nudge (spec §2).

---

## File structure

- **Modify** `src/elspeth/web/config.py` — remove `composer_advisor_enabled`; add `composer_advisor_checkpoint_max_passes`; add `_validate_advisor_distinct_from_primary` model_validator.
- **Modify** `src/elspeth/web/composer/protocol.py` — remove `composer_advisor_enabled` property; add `composer_advisor_checkpoint_max_passes` property.
- **Modify** `src/elspeth/web/composer/tools/sessions.py` — drop `reactive_validation_loop` from `ADVISOR_TRIGGER_VALUES`; add `ADVISOR_TRIGGER_DETERMINISTIC_EARLY`/`_END` (backend-only).
- **Modify** `src/elspeth/web/composer/tools/__init__.py` — re-export reconciliation.
- **Modify** `src/elspeth/web/composer/service.py` — remove reactive validation branch + tool filter + `advisor_enabled` plumbing; add `AdvisorCheckpointVerdict`, context builders, `_run_advisor_checkpoint`, the early hook, the end-gate branch, the driver counter threading.
- **Modify** `src/elspeth/web/composer/_compose_loop_carriers.py` — add `advisor_passes_delta` to `_TerminateOutcome`.
- **Modify** `src/elspeth/web/composer/prompts.py` — collapse the advisor-disabled strip surface (advisor always on).
- **Modify** `src/elspeth/web/app.py` — boot-probe always includes the advisor model; drop the `composer_advisor_enabled` telemetry attr.
- **Modify** `src/elspeth/web/composer/tool_batch.py` — drop the `composer_advisor_enabled` defense-in-depth branch.
- **Modify** `src/elspeth/web/composer/skills/pipeline_composer.md` — reframe the advisor section.
- **Modify** `deploy/elspeth-web.env` — Sonnet primary + Opus advisor; drop `COMPOSER_ADVISOR_ENABLED`; add checkpoint budget.
- **Tests** under `tests/unit/web/...` — see each task.

---

## Task 1: Config — mandatory + model-distinct advisor

**Files:**
- Modify: `src/elspeth/web/config.py` (composer block ~53-102; validators ~399-478)
- Modify: `src/elspeth/web/composer/protocol.py:673`
- Test: `tests/unit/web/test_config.py` (or the existing config-validation test module — `git grep -l "WebSettings(" tests | head` to locate; the auth-validator tests live in `tests/unit/web/test_config.py`)

- [ ] **Step 1: Write the failing tests** for the distinctness validator and the new budget field. Use the existing `_make_settings`/`WebSettings(...)` construction helper the config tests already use (it supplies the required no-default fields `composer_max_composition_turns`, `composer_max_discovery_turns`, `composer_timeout_seconds`, `composer_rate_limit_per_minute`, `shareable_link_signing_key`). Mirror an existing config-validator test for the construction boilerplate.

```python
import pytest
from pydantic import ValidationError
from elspeth.web.config import WebSettings

def _settings(**overrides):
    # Mirror the existing test helper: supply required fields + any overrides.
    base = dict(
        composer_max_composition_turns=20,
        composer_max_discovery_turns=20,
        composer_timeout_seconds=300.0,
        composer_rate_limit_per_minute=30,
        shareable_link_signing_key=b"x" * 32,
    )
    base.update(overrides)
    return WebSettings(**base)

def test_advisor_must_differ_from_primary_exact():
    with pytest.raises(ValidationError, match="composer_advisor_model must differ from composer_model"):
        _settings(composer_model="gpt-5.5", composer_advisor_model="gpt-5.5")

def test_advisor_distinct_normalizes_provider_prefix():
    # openrouter/openai/gpt-5.5 and gpt-5.5 denote the same model -> reject.
    with pytest.raises(ValidationError, match="must differ"):
        _settings(composer_model="openrouter/openai/gpt-5.5", composer_advisor_model="gpt-5.5")

def test_advisor_distinct_accepts_different_models():
    s = _settings(composer_model="claude-sonnet-4-6", composer_advisor_model="claude-opus-4-7")
    assert s.composer_advisor_model == "claude-opus-4-7"

def test_advisor_checkpoint_budget_default_and_floor():
    assert _settings().composer_advisor_checkpoint_max_passes == 2
    with pytest.raises(ValidationError):
        _settings(composer_advisor_checkpoint_max_passes=0)
```

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/web/test_config.py -k "advisor_must_differ or advisor_distinct or advisor_checkpoint_budget" -v`. Expected: FAIL (no validator; no field; `composer_advisor_checkpoint_max_passes` AttributeError).

- [ ] **Step 3: Implement the config changes** in `config.py`:

  (a) **Remove** the field at config.py:83 and its 4-line comment above it (79-82):
  ```python
      composer_advisor_enabled: bool = False
  ```

  (b) **Add** the new budget field next to the other advisor budget fields (after `composer_advisor_max_calls_per_compose`, ~line 99):
  ```python
      composer_advisor_checkpoint_max_passes: int = Field(
          default=2,
          ge=1,
          description=(
              "Max deterministic advisor-checkpoint passes per compose request "
              "(early + end + end re-reviews), counted SEPARATELY from "
              "_MAX_REPAIR_TURNS. On the last budgeted pass a still-flagged end "
              "gate fails closed (no repair it cannot re-review). DISTINCT from "
              "composer_advisor_max_calls_per_compose, which remains the hard "
              "ceiling across ALL advisor calls (checkpoints + proactive-security)."
          ),
      )
  ```

  (c) **Add** the distinctness validator next to the other `@model_validator(mode="after")` methods (after `_validate_composer_timeout_transport_headroom`, ~line 478). Normalize by taking the final `/`-segment (handles `openrouter/openai/gpt-5.5`, `openai/gpt-5.5`, `gpt-5.5` alike):
  ```python
      @model_validator(mode="after")
      def _validate_advisor_distinct_from_primary(self) -> WebSettings:
          """The advisor must be a different model from the primary composer.

          Independence of failure modes: a model checking its own work shares
          its blind spots. Exact-string distinctness on the canonical model id
          (final path segment, so provider prefixes like ``openrouter/openai/``
          do not mask a same-model pairing). The advisor is mandatory — there is
          no enable flag — so this runs for every boot.
          """
          def _canonical(model_id: str) -> str:
              return model_id.rsplit("/", 1)[-1].strip()

          if _canonical(self.composer_advisor_model) == _canonical(self.composer_model):
              raise ValueError(
                  "composer_advisor_model must differ from composer_model "
                  f"(both resolve to {_canonical(self.composer_model)!r}); the advisor "
                  "is the independent reviewer and cannot be the primary composer"
              )
          return self
  ```

  (d) In `protocol.py`: **remove** the property at 672-673:
  ```python
      @property
      def composer_advisor_enabled(self) -> bool: ...
  ```
  and **add** next to the other advisor properties (~688):
  ```python
      @property
      def composer_advisor_checkpoint_max_passes(self) -> int: ...
  ```

- [ ] **Step 4: Run to verify they pass** — `pytest tests/unit/web/test_config.py -k "advisor_must_differ or advisor_distinct or advisor_checkpoint_budget" -v`. Expected: PASS. (Other readers of `composer_advisor_enabled` still reference the removed field and will fail to import — those are fixed in Tasks 2-3; do NOT run the full suite yet.)

- [ ] **Step 5: Commit** —
```bash
git add src/elspeth/web/config.py src/elspeth/web/composer/protocol.py tests/unit/web/test_config.py
git commit -m "feat(composer): mandatory model-distinct advisor config + checkpoint budget"
```

---

## Task 2: Retire the reactive trigger (keep the tool for proactive-security)

**Files:**
- Modify: `src/elspeth/web/composer/tools/sessions.py:116-126`
- Modify: `src/elspeth/web/composer/tools/__init__.py:100,124`
- Modify: `src/elspeth/web/composer/service.py:99` (import) and `:3179` (reactive validation branch)
- Test: `tests/unit/web/composer/test_advisor_tool.py`

- [ ] **Step 1: Write/flip the failing test** — the reactive trigger is no longer a valid LLM-supplied trigger; `request_advisor_hint` with `trigger="reactive_validation_loop"` must be rejected as an unknown trigger. In `test_advisor_tool.py`, change the test that exercises the reactive trigger to expect rejection, and assert the two proactive triggers still validate.

```python
@pytest.mark.asyncio
async def test_reactive_trigger_is_retired(make_service):
    service = make_service()
    payload = service._validate_advisor_arguments({
        "trigger": "reactive_validation_loop",
        "problem_summary": "stuck",
        "recent_errors": ["e1", "e2"],
        "attempted_actions": ["a1", "a2"],
    })
    assert payload is not None  # ARG_ERROR
    assert "reactive_validation_loop" not in payload["error"] or "must be one of" in payload["error"]

@pytest.mark.asyncio
async def test_proactive_triggers_still_valid(make_service):
    service = make_service()
    for trig in ("proactive_security_safety", "proactive_red_listed_plugin"):
        payload = service._validate_advisor_arguments({
            "trigger": trig, "problem_summary": "p",
            "recent_errors": [], "attempted_actions": [],
        })
        assert payload is None  # valid
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_advisor_tool.py -k "reactive_trigger_is_retired or proactive_triggers_still_valid" -v`. Expected: FAIL (reactive still accepted).

- [ ] **Step 3: Implement the retirement.** In `tools/sessions.py`, remove the `ADVISOR_TRIGGER_REACTIVE` constant (116) and its entry in the tuple (123); keep the two proactive constants:
```python
ADVISOR_TRIGGER_PROACTIVE_SECURITY: Final[str] = "proactive_security_safety"
ADVISOR_TRIGGER_PROACTIVE_RED_LISTED: Final[str] = "proactive_red_listed_plugin"

ADVISOR_TRIGGER_VALUES: Final[tuple[str, ...]] = (
    ADVISOR_TRIGGER_PROACTIVE_SECURITY,
    ADVISOR_TRIGGER_PROACTIVE_RED_LISTED,
)
```
In `tools/__init__.py`, remove `ADVISOR_TRIGGER_REACTIVE` from the import (100) and from `__all__` (124). In `service.py`, remove `ADVISOR_TRIGGER_REACTIVE` from the import (99) and remove the reactive-specific validation branch at 3179-3187 (the `if trigger == ADVISOR_TRIGGER_REACTIVE and (len(recent) < 2 or len(attempted) < 2):` block) — proactive triggers have no min-length requirement.

- [ ] **Step 4: Run to verify it passes** — same command. Then `pytest tests/unit/web/composer/test_advisor_tool.py -q`. Expected: PASS (reconcile any other test that asserted the reactive trigger in the enum — search `git grep -n reactive_validation_loop tests`).

- [ ] **Step 5: Commit** —
```bash
git add src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/tools/__init__.py src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_tool.py
git commit -m "feat(composer): retire reactive advisor self-trigger (keep proactive-security)"
```

---

## Task 3: Remove advisor-disabled dead code (advisor now mandatory)

**Files:**
- Modify: `src/elspeth/web/composer/service.py:3021,3039-3040` and `_get_litellm_tools` / `_build_messages`
- Modify: `src/elspeth/web/composer/tool_batch.py:659-689`
- Modify: `src/elspeth/web/app.py:387,397`
- Modify: `src/elspeth/web/composer/prompts.py:49-103,110-141`
- Test: `tests/unit/web/composer/test_dispatch_arms_characterization.py`, `tests/unit/web/test_app.py`, `tests/unit/web/composer/test_compose_loop_persistence.py`

- [ ] **Step 1: Write the failing test** — the `request_advisor_hint` tool is ALWAYS present (no enable flag) and the system prompt is always the advisor-enabled projection.

```python
@pytest.mark.asyncio
async def test_advisor_tool_always_present(make_service):
    service = make_service()  # no composer_advisor_enabled anymore
    tools = service._get_litellm_tools()
    names = {t["function"]["name"] for t in tools}
    assert "request_advisor_hint" in names

def test_system_prompt_always_advisor_enabled():
    from elspeth.web.composer.prompts import build_system_prompt
    text = build_system_prompt(None)
    assert "request_advisor_hint" in text  # advisor-enabled projection is the only projection
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_dispatch_arms_characterization.py -k advisor_tool_always_present tests/unit/web/composer/test_prompts.py -k system_prompt_always -v`. Expected: FAIL (import error / signature still takes `advisor_enabled`).

- [ ] **Step 3: Implement the removals.**
  - `service.py` `_get_litellm_tools` (~3028-3051): delete the filter branch (3039-3040):
    ```python
            definitions = get_tool_definitions()
            # (removed) advisor-enabled filter — advisor is mandatory; the tool is always present.
    ```
  - `service.py` `_build_messages` (~3021): remove the `advisor_enabled=self._settings.composer_advisor_enabled` argument; `build_messages` no longer takes it (next bullet).
  - `prompts.py`: change `build_system_prompt(data_dir=None)` to drop the `advisor_enabled` kwarg and always use the enabled projection — body becomes `core = _strip_advisor_disabled_fallback(_PIPELINE_SKILL)`. Delete `_strip_advisor_content` (49-95, now dead). Update `build_messages` (427/433) to drop the `advisor_enabled` param it threads. Keep `_strip_advisor_disabled_fallback` (the enabled path) and `SYSTEM_PROMPT`.
  - `tool_batch.py:659-689`: delete the `if not ctx.service._settings.composer_advisor_enabled:` defense-in-depth disabled-error branch (the tool is always enabled now).
  - `app.py:387`: boot-probe always includes the advisor model — replace `if settings.composer_advisor_enabled:` guard so the advisor model is always appended to the probe list. `app.py:397`: remove the `"composer_advisor_enabled"` telemetry attribute.

- [ ] **Step 4: Run to verify it passes** — the two new tests, then the three named characterization suites:
  `pytest tests/unit/web/composer/test_dispatch_arms_characterization.py tests/unit/web/test_app.py tests/unit/web/composer/test_compose_loop_persistence.py tests/unit/web/composer/test_prompts.py -q`. Reconcile any test that set/asserted `composer_advisor_enabled` (remove the override; the field is gone). Expected: PASS.

- [ ] **Step 5: Commit** —
```bash
git add -A
git commit -m "refactor(composer): drop advisor-disabled code paths (advisor mandatory)"
```

---

## Task 4: Checkpoint primitives — verdict, context builders, `_run_advisor_checkpoint`

**Files:**
- Modify: `src/elspeth/web/composer/tools/sessions.py` (new backend-only trigger constants)
- Modify: `src/elspeth/web/composer/service.py` (verdict dataclass + builders + runner)
- Test: `tests/unit/web/composer/test_advisor_checkpoint.py` (new file)

- [ ] **Step 1: Write the failing test** — `_run_advisor_checkpoint` builds phase-specific arguments, calls the advisor, and maps the guidance to a verdict; bounded retry on failure; phase-tagged trigger.

```python
import pytest
from unittest.mock import AsyncMock
from elspeth.web.composer.service import AdvisorCheckpointVerdict

@pytest.mark.asyncio
async def test_run_advisor_checkpoint_end_returns_verdict(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(return_value=("FLAGGED: the sink drops the rating field", {}))
    verdict = await service._run_advisor_checkpoint(
        phase="end", state=simple_state, session_id="s1", recorder=make_recorder(),
    )
    assert isinstance(verdict, AdvisorCheckpointVerdict)
    assert verdict.ok is True
    assert verdict.blocking is True
    assert "rating field" in verdict.findings_text
    # The synthesized trigger is the backend-only end trigger.
    args = service._call_advisor_with_audit.call_args.args[0]
    assert args["trigger"] == "deterministic_end_checkpoint"

@pytest.mark.asyncio
async def test_run_advisor_checkpoint_clean_verdict(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(return_value=("CLEAN: intent satisfied, contracts consistent", {}))
    verdict = await service._run_advisor_checkpoint(phase="end", state=simple_state, session_id="s1", recorder=make_recorder())
    assert verdict.ok is True and verdict.blocking is False

@pytest.mark.asyncio
async def test_run_advisor_checkpoint_unavailable_after_retries(make_service, simple_state):
    service = make_service()
    service._call_advisor_with_audit = AsyncMock(side_effect=TimeoutError())
    verdict = await service._run_advisor_checkpoint(phase="end", state=simple_state, session_id="s1", recorder=make_recorder())
    assert verdict.ok is False  # unavailable
    assert service._call_advisor_with_audit.await_count >= 2  # bounded retry
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -v`. Expected: FAIL (no `AdvisorCheckpointVerdict`, no `_run_advisor_checkpoint`).

- [ ] **Step 3: Implement.** In `tools/sessions.py`, add backend-only trigger constants (NOT in `ADVISOR_TRIGGER_VALUES` — those are the LLM-selectable set; these are backend-synthesized and trusted):
```python
ADVISOR_TRIGGER_DETERMINISTIC_EARLY: Final[str] = "deterministic_early_checkpoint"
ADVISOR_TRIGGER_DETERMINISTIC_END: Final[str] = "deterministic_end_checkpoint"
```
In `service.py`, add the verdict dataclass near the other compose-loop carriers' usage (module scope, after imports):
```python
@dataclass(frozen=True, slots=True)
class AdvisorCheckpointVerdict:
    """Result of a deterministic advisor checkpoint.

    ``ok`` False => the advisor call failed after bounded retry (unavailable);
    callers decide degrade (early) vs fail-closed (end). ``blocking`` True =>
    the advisor flagged a problem (only meaningful when ``ok``).
    """
    ok: bool
    blocking: bool
    findings_text: str
```
Add the phase-specific argument builders and the runner as `ComposerService` methods. The builders synthesize the `arguments` dict in the shape `_build_advisor_user_message` consumes (`trigger`, `problem_summary`, `recent_errors`, `attempted_actions`, optional `schema_excerpt`) — backend-trusted data, so they BYPASS `_validate_advisor_arguments` (that validates Tier-3 LLM input; this is Tier-1 internal). A compact pipeline summary (topology + node options) goes in `schema_excerpt`.
```python
    def _build_checkpoint_arguments(self, *, phase: str, state: CompositionState) -> dict[str, Any]:
        pipeline_summary = _summarize_pipeline_for_advisor(state)  # topology + node options + field contracts
        if phase == "early":
            return {
                "trigger": ADVISOR_TRIGGER_DETERMINISTIC_EARLY,
                "problem_summary": (
                    "Review this pipeline APPROACH early (it was just established). "
                    "Does the topology fit the user's intent? Are producer->consumer "
                    "field contracts coherent (does each node consume fields its upstream "
                    "actually emits, accounting for subtractive transforms)? Name concrete gaps."
                ),
                "recent_errors": [],
                "attempted_actions": [],
                "schema_excerpt": pipeline_summary,
            }
        return {
            "trigger": ADVISOR_TRIGGER_DETERMINISTIC_END,
            "problem_summary": (
                "Final sign-off. Does this pipeline fulfil the user's intent and is it "
                "sound? Flag any unmet intent, broken field contract, or subjective rubric "
                "that should have been surfaced. Start your reply with CLEAN or FLAGGED."
            ),
            "recent_errors": [],
            "attempted_actions": [],
            "schema_excerpt": pipeline_summary,
        }

    async def _run_advisor_checkpoint(
        self,
        *,
        phase: str,
        state: CompositionState,
        session_id: str | None,
        recorder: BufferingRecorder | None,
    ) -> AdvisorCheckpointVerdict:
        """Backend-initiated deterministic advisor checkpoint (early|end).

        Reuses _call_advisor_with_audit. Bounded retry on call failure. The
        verdict's ``blocking`` is True iff the guidance is a FLAGGED sign-off;
        a leading CLEAN (case-insensitive) is non-blocking.
        """
        arguments = self._build_checkpoint_arguments(phase=phase, state=state)
        attempts = 2  # bounded retry; the call itself is wrapped in asyncio.wait_for
        last_exc: Exception | None = None
        for _ in range(attempts):
            try:
                guidance, _meta = await self._call_advisor_with_audit(arguments, recorder=recorder)
            except Exception as exc:  # noqa: BLE001 - call core re-raises typed LLM errors; treat all as unavailable
                last_exc = exc
                continue
            blocking = not guidance.strip().upper().startswith("CLEAN")
            return AdvisorCheckpointVerdict(ok=True, blocking=blocking, findings_text=guidance.strip())
        return AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text=str(last_exc) if last_exc else "advisor unavailable")
```
Also implement `_summarize_pipeline_for_advisor(state: CompositionState) -> str` at module scope — a compact, redaction-safe rendering (node id, type, key options, and per-node `required_input_fields` vs upstream-emitted fields where derivable). Keep it descriptive text; reuse existing pipeline-summary helpers if one exists (`git grep -n "def .*summari.*pipeline\|def _describe_pipeline" src/elspeth/web/composer`).

- [ ] **Step 4: Run to verify it passes** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -v`. Expected: PASS.

- [ ] **Step 5: Commit** —
```bash
git add src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_checkpoint.py
git commit -m "feat(composer): deterministic advisor checkpoint runner (early/end) reusing the audited call"
```

---

## Task 5: EARLY advisory pass (once per session, never blocks)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — `_compose_loop` driver (counter init ~2731; after P3 dispatch ~2825-2830) and a new `_maybe_run_early_checkpoint` helper.
- Test: `tests/unit/web/composer/test_advisor_checkpoint.py`

- [ ] **Step 1: Write the failing test** — when a turn first establishes a non-empty pipeline and no prior early-checkpoint event exists, the early pass runs and its guidance is appended to `llm_messages`; on a later turn it does NOT re-run; a failed early call degrades silently (loop continues).

The EARLY pass fires on the empty→non-empty pipeline TRANSITION, which is structurally ≤ once per session (request 1 creates the pipeline → fires; later turns have a non-empty prev_state → skip; a resumed session starts non-empty → skip). No counter and no audit-latch is needed — and crucially the early pass does NOT consume the END gate's `advisor_checkpoint_passes_used` budget (Task 6), so it can never starve the end re-review.

```python
@pytest.mark.asyncio
async def test_early_checkpoint_runs_on_transition_and_injects(make_service, empty_state, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(
        return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="Consider a field_mapper before the sink"))
    llm_messages = []
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state, prev_state=empty_state, session_id="s1",
        llm_messages=llm_messages, recorder=make_recorder())
    assert ran is True
    assert any("field_mapper" in m["content"] for m in llm_messages if m["role"] == "user")

@pytest.mark.asyncio
async def test_early_checkpoint_skips_when_pipeline_already_nonempty(make_service, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock()
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state, prev_state=nonempty_state, session_id="s1",
        llm_messages=[], recorder=make_recorder())
    assert ran is False
    service._run_advisor_checkpoint.assert_not_awaited()

@pytest.mark.asyncio
async def test_early_checkpoint_degrades_on_failure(make_service, empty_state, nonempty_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable"))
    llm_messages = []
    ran = await service._maybe_run_early_checkpoint(
        state=nonempty_state, prev_state=empty_state, session_id="s1", llm_messages=llm_messages, recorder=make_recorder())
    assert ran is True  # attempted
    assert llm_messages == []  # nothing injected; degraded silently
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -k early_checkpoint -v`. Expected: FAIL (no `_maybe_run_early_checkpoint`).

- [ ] **Step 3: Implement.** Add the helper — fire purely on the empty→non-empty transition (no audit-latch, no end-budget consumption):
```python
    async def _maybe_run_early_checkpoint(
        self, *, state: CompositionState, prev_state: CompositionState,
        session_id: str | None, llm_messages: list[dict[str, Any]], recorder: BufferingRecorder,
    ) -> bool:
        """Run the EARLY advisory checkpoint on the empty->non-empty pipeline
        TRANSITION (structurally <= once per session). Advisory only: inject the
        guidance as a user message; NEVER block. Degrade silently on failure.
        Does NOT consume the END gate budget. Returns whether it ran."""
        if _state_is_structurally_empty(state):
            return False
        if not _state_is_structurally_empty(prev_state):
            return False  # pipeline was already non-empty before this turn (or resumed session)
        verdict = await self._run_advisor_checkpoint(phase="early", state=state, session_id=session_id, recorder=recorder)
        if verdict.ok and verdict.blocking:
            llm_messages.append({
                "role": "user",
                "content": (
                    "[Early review by the advisor model — advisory, not binding]\n"
                    + verdict.findings_text
                    + "\n\nAddress any concrete gap above, or continue if it does not apply."
                ),
            })
        return True
```
Wire it into `_compose_loop` after P3 dispatch updates `state` (after service.py:2825-2830, where `state = dispatch.state`), capturing `prev_state` BEFORE the dispatch reassignment (`prev_state = state` just before `state = dispatch.state`). Use `_state_is_structurally_empty` (already used for the pre-state branch). NOT gated on the end-gate counter — the transition is its own ≤once-per-session bound.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -k early_checkpoint -v`. Expected: PASS.

- [ ] **Step 5: Commit** —
```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_advisor_checkpoint.py
git commit -m "feat(composer): early advisory advisor checkpoint (once per session, non-blocking)"
```

---

## Task 6: END authoritative gate (re-review loop; fail-closed; separate budget)

**Files:**
- Modify: `src/elspeth/web/composer/_compose_loop_carriers.py` — add `advisor_passes_delta` to `_TerminateOutcome`.
- Modify: `src/elspeth/web/composer/service.py` — `_try_terminate_no_tools` (new branch before the tail call ~2468; signature gains `advisor_checkpoint_passes_used`), `_classify_and_budget_turn` (last-chance gate before tail ~2272), `_compose_loop` driver (counter init + delta application).
- Test: `tests/unit/web/composer/test_advisor_checkpoint.py`

- [ ] **Step 1: Write the failing tests** — the end gate: clean→proceed; flagged with budget→repair-continue (advisor_passes_delta=1, message injected); flagged on last pass→fail-closed; unavailable→fail-closed. Drive `_try_terminate_no_tools` with a non-orphan state.

```python
@pytest.mark.asyncio
async def test_end_gate_clean_proceeds_to_finalize(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=False, findings_text="CLEAN"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight is None or outcome.result.runtime_preflight.is_valid

@pytest.mark.asyncio
async def test_end_gate_flagged_with_budget_repairs(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: sink omits rating"))
    llm_messages = []
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0, llm_messages=llm_messages)
    assert outcome.action == "continue"
    assert outcome.advisor_passes_delta == 1
    assert any("FLAGGED" in m["content"] for m in llm_messages)

@pytest.mark.asyncio
async def test_end_gate_flagged_on_last_pass_fails_closed(make_service, clean_runnable_state):
    service = make_service()  # composer_advisor_checkpoint_max_passes default 2
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED: still wrong"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=1)  # next pass is the last
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight.is_valid is False
    assert outcome.result.runtime_preflight.readiness.execution_ready is False

@pytest.mark.asyncio
async def test_end_gate_unavailable_fails_closed(make_service, clean_runnable_state):
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=False, blocking=False, findings_text="unavailable"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "return"
    assert outcome.result.runtime_preflight.is_valid is False

@pytest.mark.asyncio
async def test_advisor_budget_does_not_consume_repair_budget(make_service, clean_runnable_state):
    """Gate-order invariant: a flagged advisor repair-continue increments
    advisor_passes_delta, NOT repair_turns_delta."""
    service = make_service()
    service._run_advisor_checkpoint = AsyncMock(return_value=AdvisorCheckpointVerdict(ok=True, blocking=True, findings_text="FLAGGED"))
    outcome = await drive_try_terminate(service, clean_runnable_state, advisor_checkpoint_passes_used=0)
    assert outcome.action == "continue"
    assert outcome.repair_turns_delta == 0
    assert outcome.advisor_passes_delta == 1
```
(`drive_try_terminate` is a test helper that calls `service._try_terminate_no_tools(...)` with the full kwarg set; build it once at the top of the test module mirroring the existing dispatch-test setup. `clean_runnable_state` is a fixture whose `_missing_pending_interpretation_review_sites` returns empty — stub it so the orphan pre-check passes.)

- [ ] **Step 2: Run to verify they fail** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -k end_gate -v`. Expected: FAIL (no advisor branch; `_TerminateOutcome` has no `advisor_passes_delta`).

- [ ] **Step 3a: Add the counter field** to `_compose_loop_carriers.py` `_TerminateOutcome` (after `repair_turns_delta` ~line 131):
```python
    advisor_passes_delta: int = 0
```

- [ ] **Step 3b: Add the end-gate branch** in `_try_terminate_no_tools`, inserted AFTER the proof-repair branch (service.py:2435) and BEFORE the `_surface_and_finalize_no_tools` call (service.py:2468). The branch runs a cheap orphan pre-check first (so a frontier advisor call is never spent on a pipeline the orphan gate will block), then the advisor:
```python
        # END authoritative advisor gate (elspeth-dac6602a2b). Runs AFTER the
        # cheap deterministic gates (proof above; orphan pre-check here mirrors
        # the tail's gate) so the frontier advisor only reviews a mechanically
        # valid pipeline. Budget is SEPARATE from _MAX_REPAIR_TURNS: a flagged
        # repair-continue increments advisor_passes_delta, never repair_turns.
        # On the LAST budgeted pass a still-flagged gate FAILS CLOSED (no repair
        # it cannot re-review); an unavailable advisor (after bounded retry)
        # FAILS CLOSED — the advisor is the mandatory final authority.
        max_passes = self._settings.composer_advisor_checkpoint_max_passes
        if advisor_checkpoint_passes_used < max_passes:
            orphaned_precheck = await self._missing_pending_interpretation_review_sites(state, session_id=session_id)
            if not orphaned_precheck:
                verdict = await self._run_advisor_checkpoint(
                    phase="end", state=state, session_id=session_id, recorder=recorder,
                )
                is_last_pass = (advisor_checkpoint_passes_used + 1) >= max_passes
                if not verdict.ok:
                    return _TerminateOutcome(
                        action="return",
                        result=self._advisor_blocked_result(
                            reason="unavailable", verdict=verdict, state=state,
                            assistant_message=assistant_message, recorder=recorder,
                            repair_turns_used=repair_turns_used,
                            persisted_assistant_message_id=persisted_assistant_message_id,
                            persisted_tool_call_turn=persisted_tool_call_turn,
                        ),
                        advisor_passes_delta=1,
                    )
                if verdict.blocking and is_last_pass:
                    return _TerminateOutcome(
                        action="return",
                        result=self._advisor_blocked_result(
                            reason="exhausted", verdict=verdict, state=state,
                            assistant_message=assistant_message, recorder=recorder,
                            repair_turns_used=repair_turns_used,
                            persisted_assistant_message_id=persisted_assistant_message_id,
                            persisted_tool_call_turn=persisted_tool_call_turn,
                        ),
                        advisor_passes_delta=1,
                    )
                if verdict.blocking:
                    llm_messages.append({
                        "role": "user",
                        "content": (
                            "[Advisor sign-off — BLOCKING. Resolve before completing.]\n"
                            + verdict.findings_text
                        ),
                    })
                    return _TerminateOutcome(action="continue", advisor_passes_delta=1)
                # CLEAN -> fall through to the shared tail (auto-surface + final orphan gate + finalize)
```

- [ ] **Step 3c: Add the blocked-result helper** `_advisor_blocked_result(...)` near `_orphaned_interpretation_review_validation` (service.py:758). It builds a fail-closed `ComposerResult` mirroring the orphan-gate shape (2575-2588) — a non-runnable `ValidationResult` with `readiness.authoring_valid/execution_ready/completion_ready = False` and a distinct `error_code` (`_ADVISOR_SIGNOFF_BLOCKED_CODE = "advisor_signoff_blocked"`), the findings carried in the augmented message, threaded with `repair_turns_used`/persisted ids:
```python
    def _advisor_blocked_result(
        self, *, reason: str, verdict: AdvisorCheckpointVerdict, state: CompositionState,
        assistant_message: Any, recorder: BufferingRecorder, repair_turns_used: int,
        persisted_assistant_message_id: str | None, persisted_tool_call_turn: bool,
    ) -> ComposerResult:
        raw_content = assistant_message.content or ""
        runtime_result = _advisor_signoff_blocked_validation(reason=reason, findings=verdict.findings_text)
        augmented = _compose_preflight_failure_message(raw_content, runtime_result=runtime_result)
        _enforce_augmentation_prefix_invariant(
            branch="advisor_signoff_blocked_augmentation", content=raw_content, augmented=augmented,
        )
        return replace(
            ComposerResult(
                message=augmented, state=state, runtime_preflight=runtime_result,
                raw_assistant_content=raw_content, tool_invocations=recorder.invocations,
                llm_calls=recorder.llm_calls,
            ),
            repair_turns_used=repair_turns_used,
            persisted_assistant_message_id=persisted_assistant_message_id,
            persisted_tool_call_turn=persisted_tool_call_turn,
        )
```
Implement `_advisor_signoff_blocked_validation(reason, findings) -> ValidationResult` mirroring `_orphaned_interpretation_review_validation` (service.py:758-841): same `ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[...])` shape with `error_code=_ADVISOR_SIGNOFF_BLOCKED_CODE` and a blocker naming the advisor sign-off + the reason. Read service.py:758-841 verbatim and copy its construction, substituting the code/message.

- [ ] **Step 3d: Thread the counter through the driver.** Add `advisor_checkpoint_passes_used = 0` near service.py:2738 (next to `repair_turns_used = 0`). This counts END-gate passes ONLY (the early pass does not touch it — Task 5). Pass it into `_try_terminate_no_tools` (add the kwarg at the P2 call site ~service.py:2761-2768 and to the method signature ~2326). After the P2 outcome (where `repair_turns_used += terminate.repair_turns_delta` at service.py:2797), add `advisor_checkpoint_passes_used += terminate.advisor_passes_delta`.

- [ ] **Step 3e: Add the P5 last-chance gate** in `_classify_and_budget_turn`, before the `_surface_and_finalize_no_tools` call at service.py:2272. Here repair is NOT possible (composition budget exhausted), so flagged/unavailable → fail-closed; clean → fall through. Thread `advisor_checkpoint_passes_used` into `_classify_and_budget_turn` (signature + call site ~2871) and add `advisor_passes_delta` to `_ClassifyOutcome` (carriers ~214) applied in the driver after P5 (~2890):
```python
        max_passes = self._settings.composer_advisor_checkpoint_max_passes
        if advisor_checkpoint_passes_used < max_passes:
            orphaned_precheck = await self._missing_pending_interpretation_review_sites(state, session_id=session_id)
            if not orphaned_precheck:
                verdict = await self._run_advisor_checkpoint(phase="end", state=state, session_id=session_id, recorder=recorder)
                if (not verdict.ok) or verdict.blocking:
                    return _ClassifyOutcome(
                        action="return",
                        result=self._advisor_blocked_result(
                            reason="unavailable" if not verdict.ok else "exhausted",
                            verdict=verdict, state=state, assistant_message=assistant_message,
                            recorder=recorder, repair_turns_used=0,
                            persisted_assistant_message_id=persist.persisted_assistant_message_id,
                            persisted_tool_call_turn=persist.persisted_tool_call_turn,
                        ),
                        composition_turns_delta=1, advisor_passes_delta=1,
                    )
```
(Note: the P5 path can't repair, so even a non-last flagged verdict fails closed — there is no further composition budget to re-run the model.)

- [ ] **Step 4: Run to verify they pass** — `pytest tests/unit/web/composer/test_advisor_checkpoint.py -k end_gate -v`, then the whole file, then the compose-loop dispatch suite: `pytest tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -q`. Expected: PASS. Reconcile any dispatch test that finalized without an advisor (stub `_run_advisor_checkpoint` to return a CLEAN verdict in the shared test setup so existing finalize tests still finalize — add that stub to the suite's service factory).

- [ ] **Step 5: Commit** —
```bash
git add -A
git commit -m "feat(composer): advisor end gate as final authority (re-review loop, fail-closed, separate budget)"
```

---

## Task 7: Skill prompt — reframe the advisor section

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`

NOTE: skills are LLM prompts, not code — do NOT add a grep-the-text "test" (project doctrine: theatre). The live web service reads this file via an `@lru_cache`'d import; restart `elspeth-web.service` after editing (Task 9). If a `composer_skill_hash` CI gate pins it, refresh per Task 9.

- [ ] **Step 1: Edit** — (a) remove the "When You Are Still Stuck — `request_advisor_hint`" self-trigger subsection's REACTIVE framing (the LLM no longer phones the advisor when stuck for validation — the backend runs deterministic checkpoints). (b) Keep the `request_advisor_hint` tool documented ONLY for the proactive security/red-listed triggers. (c) Add a short note: "An advisor model reviews your work automatically — early (your approach) and at completion (final sign-off). The end review is BINDING: if it flags an issue you will be asked to fix it before the pipeline can complete." (d) Remove any `<!-- ADVISOR-DISABLED -->` fallback blocks (advisor is always on).

- [ ] **Step 2: Commit** —
```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md
git commit -m "docs(composer-skill): backend-run advisor checkpoints; retire reactive self-trigger framing"
```

---

## Task 8: Env — Sonnet primary + Opus advisor

**Files:**
- Modify: `deploy/elspeth-web.env:11,17,18`

- [ ] **Step 1: Edit** — set the frontier pairing, remove the retired enable flag, add the checkpoint budget:
```
ELSPETH_WEB__COMPOSER_MODEL=openrouter/anthropic/claude-sonnet-4-6
# (removed) ELSPETH_WEB__COMPOSER_ADVISOR_ENABLED — advisor is mandatory
ELSPETH_WEB__COMPOSER_ADVISOR_MODEL=openrouter/anthropic/claude-opus-4-7
ELSPETH_WEB__COMPOSER_ADVISOR_CHECKPOINT_MAX_PASSES=2
```
(NOTE: Sonnet + Opus are both Anthropic — exact-string distinctness passes since `claude-sonnet-4-6` != `claude-opus-4-7`. This tests model-distinctness, not vendor-independence — a deliberate operator choice per spec §13.)

- [ ] **Step 2: Commit** —
```bash
git add deploy/elspeth-web.env
git commit -m "config(deploy): frontier composer pairing — Sonnet primary + Opus advisor"
```

---

## Task 9: Gates, local verification, staging harness verification

**Files:** none (verification)

- [ ] **Step 1: Local gate surface** (memory: run lints before push) — `ruff check`, `ruff format --check`, `mypy` on changed files (the `ComposerSettings` Protocol change is mypy-enforced — confirm `WebSettings` still structurally conforms). Then `pytest tests/unit/web tests/integration/web/composer -q`. Fix fallout.

- [ ] **Step 2: composer_skill_hash gate** — `git grep -n "assert_skill_hash_unchanged_on_disk\|expected_sha256"` to find the stored expected hash; recompute and co-land the skill-hash refresh (Task 7 changed the skill). The literal-hash values in unit tests are stubs (safe).

- [ ] **Step 3: tier-model gate** — `env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`. Expected: clean (config.py is not in the allowlist; the new validator uses direct attribute access + explicit raise; the checkpoint builders synthesize trusted Tier-1 dicts). If anything flags, STOP and surface to operator.

- [ ] **Step 4: Deploy + staging verify (the harness)** — `sudo systemctl restart elspeth-web.service` (sudoers-granted; picks up HEAD of /home/john/elspeth + the skill change + the env pairing). Confirm `/healthz` 200 (Sonnet+Opus boot-probe both succeed; if boot fails on a distinctness/mandate error, that's the config validator doing its job — fix the env). Then run the tutorial battery reusing the saved token (temp config dropping globalSetup), e.g. `HARNESS_BATCH_ID=advisor-authority HARNESS_BATCH_SIZE=8 npx playwright test --config=<temp> tutorial-reliability`. SUCCESS CRITERIA: graduation rate at/above the opus-6/6 baseline; advisor end gate fires (audit shows `deterministic_end_checkpoint` advisor calls); zero compose hangs from advisor-budget/repair-budget contention; a deliberately under-specified prompt triggers an end-gate FLAGGED→repair→clean sequence (or a fail-closed surfaced block) rather than a silent bad finalize. Delete the temp config after.

- [ ] **Step 5: Record + close** — write the battery result to `notes/tutorial-reliability/2026-06-07-advisor-authority-verify.md`; comment the commit SHAs + staging evidence on elspeth-dac6602a2b; `filigree close elspeth-dac6602a2b` once the battery meets criteria. (filigree CLI works; the MCP server is stale v26 vs DB v27 — use the CLI.)

---

## Self-review notes

- **Spec coverage:** D-1 distinctness → Task 1; D-2 scope (advisor only, two-tier dropped per §13) → all tasks, no two-tier; D-3 hybrid reuse → Task 4 (`_run_advisor_checkpoint` reuses `_call_advisor_with_audit`); D-4 re-review loop → Task 6 (budget-arithmetic re-review); D-5 early advisory / end authoritative → Tasks 5/6; D-6 mandatory fail-closed at startup → Task 1 (validator) + Task 3 (remove enable flag); D-7 unavailable→fail-closed end / degrade early → Task 6 / Task 5; D-8 separate budget by gate order → Task 6 (`advisor_passes_delta` parallel to `repair_turns_delta`). Retire-trigger reconciliation → Task 2. Frontier pairing → Task 8.
- **Implementer must read verbatim before coding (audit-sensitive, just landed in 5deb34f78):** service.py:758-841 (`_orphaned_interpretation_review_validation` — the exact ValidationResult/readiness shape to mirror in `_advisor_signoff_blocked_validation`); service.py:2326-2495 (`_try_terminate_no_tools` full body — exact kwargs + the `replace(result, ...)` threading); service.py:2153-2296 (`_classify_and_budget_turn` last-chance + `_ClassifyOutcome` wrapping). These are 1-read confirmations, not design gaps — the inserted code above is anchored to exact lines.
- **Type consistency:** `AdvisorCheckpointVerdict(ok, blocking, findings_text)` used identically in Tasks 4/5/6; `_TerminateOutcome.advisor_passes_delta` / `_ClassifyOutcome.advisor_passes_delta` added in Task 6; `composer_advisor_checkpoint_max_passes` (config) read in Tasks 5/6.
- **Budget invariant (the load-bearing test):** `test_advisor_budget_does_not_consume_repair_budget` (Task 6) pins D-8 — gate order means a turn is a correctness repair (`repair_turns_delta`) XOR an advisor repair (`advisor_passes_delta`), never both.
- **Risk:** Task 6 is the audit-sensitive heart (touches the just-landed Case-B finalize doors). Keep changes minimal, run the full composer suite, and stub `_run_advisor_checkpoint`→CLEAN in pre-existing finalize tests so they still exercise the finalize path.
