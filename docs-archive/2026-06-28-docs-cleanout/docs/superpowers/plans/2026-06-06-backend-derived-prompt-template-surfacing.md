# Backend-Derived Prompt-Template Review Surfacing (Case B fix) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the "stale prompt-template review" brick class (elspeth-e51216d305 Case B) by having the BACKEND surface the `llm_prompt_template` review at compose-turn finalization against the frozen final prompt skeleton, instead of the LLM surfacing it mid-build (where the skeleton can still grow and invalidate it).

**Architecture:** The `llm_prompt_template` interpretation REQUIREMENT is already auto-staged on every LLM node (`_options_with_default_prompt_template_review`, tools/_common.py:176). Today the composer LLM also surfaces the EVENT via `request_interpretation_review`; because the LLM can mutate the skeleton AFTER surfacing, the review goes stale (Case B). This plan: (1) add a backend helper that surfaces pending prompt-template review EVENTS at turn finalization against the current (frozen) state, idempotently; (2) call it in `_try_terminate_no_tools` BEFORE the missing-sites/orphan gates; (3) make `request_interpretation_review` REJECT `kind=llm_prompt_template` so the LLM cannot surface it early; (4) update the skill prompt; (5) verify on staging. The committed Case-A fix (22168c227, skeleton-hash resolve gate) stays as the backstop.

**Tech Stack:** Python 3.13, FastAPI, SQLAlchemy/SQLite (sessions DB), Pydantic v2 (Tier-3 boundary models), pytest+pytest-asyncio+xdist. Frontend unchanged (the tutorial batch-refreshes interpretation events after the compose turn via `TutorialTurn2Describe.tsx:222`).

---

## Design decisions (resolve before coding; flag (D1) for operator/judge sign-off)

**(D1) Audit provenance of a backend-surfaced event — REQUIRES operator/judge awareness.**
`create_pending_interpretation_event` requires a non-null `tool_call_id` (sessions/service.py:2667; audit row shape composer_interpretation.py:155-167). A backend-surfaced event has no LLM tool call. Decision: pass an HONEST sentinel `tool_call_id = f"backend_auto_surface:{assistant_message_id or uuid4()}"` and keep `interpretation_source = USER_APPROVED` (the user still reviews it; review semantics are identical). This is a free-text value, NOT a schema/enum change, and records *more* honest provenance than faking an LLM tool-call id. It is the one audit-data-domain choice in this plan — surface it in the PR for judge/operator review (cf. trust-tier custody doctrine).

**(D2) Audit metadata at finalization.** The LLM response object is not in scope inside `_try_terminate_no_tools`. Use the service's own fields (already used by the dispatch builder at service.py:3277-3282): `model_identifier=self._model`, `model_version=self._model` (fallback, mirroring `safe_response_model(response) or self._model`), `provider=self._availability.provider or "unknown"`, `composer_skill_hash=self._composer_skill_hash`.

**(D3) Idempotency.** Trivial under surface-late (D4). The auto-surface runs once per turn — line 2334 (the orphan gate) is reached exactly once, then the turn returns. A defensive skip still applies: fetch pending events (`list_interpretation_events(session_id, status="pending")`) and skip any node that already has a pending `LLM_PROMPT_TEMPLATE` event (guards a stale pending PT event left from a prior turn; within a turn at most one is created). (Rate-limiting does NOT apply — backend-initiated.)

**(D4) Insertion point — SURFACE LATE (corrected 2026-06-07 after advisor review + read-only investigation `wf_9a6727f8-d84`).** Insert the auto-surfacing call IMMEDIATELY BEFORE the orphan gate at service.py:2334 (i.e. at ~2333), NOT at 2257. Established from the investigation:
- **The skeleton is MUTABLE across repair turns** but **FROZEN once line 2334 is reached** (that path makes no further LLM call — reads `state` and returns; investigator `skeleton-growth-timing`). Surfacing at 2257 snapshots attempt-1's skeleton S1; a later repair turn can grow it to S2 → the surfaced review goes stale → **Case B re-introduced inside the repair loop** (advisor catch). Surfacing right before 2334 always captures the FINAL skeleton → staleness impossible by construction.
- **`_missing_pending_interpretation_review_sites` flags PT at BOTH the repair gate (2258/2263) and the orphan gate (2334)** with no kind filter (investigator `missing-sites-logic`). Since the LLM now REJECTS surfacing PT (Task 3), the repair message at 2263 would pester the model for something it cannot provide. Fix: **post-filter `LLM_PROMPT_TEMPLATE` out of the repair message at 2263 only** (Task 2b) — repair-continue only when NON-PT sites remain. Leave the orphan gate (2334) UNFILTERED so a still-missing PT after auto-surface stays **fail-closed** (backstop if the helper ever no-ops).
- **The PT requirement is reliably `pending` at finalization** (replace-semantics on re-stage; investigator `pt-requirement-pending`) → `create_pending`'s gate at sessions/service.py:2839 is satisfied.

**(D5) composition_state_id source — PLAN-REALITY CORRECTION (verified 2026-06-06).** The original draft used `current_state_id = state.id`. **`CompositionState` has NO `.id` field** (verified composer/state.py:1761-1786 — fields are source/nodes/edges/outputs/metadata/version/guided_session). The persisted composition_state_id is the outer-loop local `current_state_id` (service.py:2555, assigned from `persist.current_state_id` at 2655) — it is the DB id that `create_pending_interpretation_event` requires (the gate at sessions/service.py needs a pending requirement at THAT exact composition_state_id). It is **NOT** currently a parameter of `_try_terminate_no_tools` (verified signature 2209-2227). **Therefore Task 2 MUST thread `current_state_id: str | None` into `_try_terminate_no_tools`** (add the param; pass the outer-loop `current_state_id` at the call site) and the helper takes it as an argument. Correctness assumption to confirm during impl: at the no-tool-calls finalization branch the LLM made no further mutation this turn, so `current_state_id` already points at the frozen final skeleton; if a finalization can occur before the final state is persisted, the requirement lookup at that id will miss and the create_pending gate (`llm_draft == options.prompt_template` + pending requirement at id) will reject — verify with the Task 2 test before wiring.

**(D6) Audit surface is JUST D1; freeform parity confirmed.** Surface-late (D4) needs NO pending-event mutation / refresh / supersede primitive — those would have been a SECOND audit-domain choice under surface-early. The ONLY audit-data-domain decision in this plan is the D1 `tool_call_id` sentinel. Freeform parity (investigator `freeform-refresh-parity`): sendMessage / retryMessage / acceptProposal all refresh interpretation events post-turn (sessionStore.ts:643/1065/751) → backend-surfaced PT events reach freeform users too. `forkFromMessage` (sessionStore.ts:1109) omits the refresh — PRE-EXISTING gap, **file as a follow-up observation, out of scope here.**

---

## File structure

- **Modify** `src/elspeth/web/sessions/service.py` — no change to `create_pending_interpretation_event` itself; it already supports a direct backend call with kind=LLM_PROMPT_TEMPLATE. (Read-only reference: gate at 2834 requires `llm_draft == options.prompt_template`; requirement must exist+pending at 2839 — both hold when surfacing against the frozen final state, since the auto-staged requirement is pending and its draft tracks the prompt.)
- **Modify** `src/elspeth/web/composer/service.py` — add `_auto_surface_prompt_template_reviews(...)` method; call it in `_try_terminate_no_tools` at line 2257.
- **Modify** `src/elspeth/web/composer/tools/sessions.py` — reject `kind=llm_prompt_template` in `_handle_request_interpretation_review` (~line 1609).
- **Modify** `src/elspeth/web/composer/skills/pipeline_composer.md` — remove the instruction to surface the prompt-template review (lines ~551-552, ~600-605, ~686-690); state the backend auto-surfaces it.
- **Test** `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py` — new test: backend surfaces the event at finalization; update `test_missing_prompt_template_review_event_forces_review_tool_retry` (~931).
- **Test** `tests/unit/web/composer/test_request_interpretation_review_tool.py` — flip `test_request_interpretation_review_accepts_prompt_template_kind` (~719) to expect rejection; keep stale-draft test or fold into the rejection.
- **Test** `tests/unit/web/sessions/test_interpretation_events_service.py` — add Case-B-style service test if useful (the backend surfacing + the committed resolve gate together let a vague-term-first order graduate).

---

## Task 1: Backend auto-surfacing helper (service-level, idempotent)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add method near `_missing_pending_interpretation_review_sites` ~1293)
- Test: `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`

- [ ] **Step 1: Write the failing test** — a composition with one LLM node carrying a pending auto-staged `llm_prompt_template` requirement and NO pending event; assert the helper creates exactly one pending `llm_prompt_template` event whose `llm_draft == node.options.prompt_template`, and is idempotent on a second call.

```python
@pytest.mark.asyncio
async def test_auto_surface_prompt_template_creates_pending_event_idempotently(...):
    # Build a session + composition state with one llm node whose options carry
    # the auto-staged pending llm_prompt_template requirement (use the same
    # _structured_llm_node-style helper the sessions tests use, with a
    # prompt_template_review requirement, status="pending").
    service = make_composer_service(...)  # mirror existing dispatch-test setup
    # current_state_id is the PERSISTED composition_state_id (D5) — the DB row the
    # auto-staged pending requirement is attached to, NOT state.id (no such field).
    await service._auto_surface_prompt_template_reviews(state, session_id=sid, current_state_id=state_db_id)
    events = await sessions_service.list_interpretation_events(sid, status="pending")
    pt = [e for e in events if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]
    assert len(pt) == 1
    assert pt[0].user_term == f"llm_prompt_template:{node_id}"
    assert pt[0].llm_draft == node_options["prompt_template"]
    assert pt[0].tool_call_id.startswith("backend_auto_surface:")
    # Idempotent: a second call creates no duplicate.
    await service._auto_surface_prompt_template_reviews(state, session_id=sid, current_state_id=state_db_id)
    events2 = await sessions_service.list_interpretation_events(sid, status="pending")
    assert len([e for e in events2 if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE]) == 1
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py::test_auto_surface_prompt_template_creates_pending_event_idempotently -v`. Expected: FAIL (AttributeError: no `_auto_surface_prompt_template_reviews`).

- [ ] **Step 3: Implement the helper** (in `ComposerService`, near line 1293):

```python
async def _auto_surface_prompt_template_reviews(
    self,
    state: CompositionState,
    *,
    session_id: str | None,
    current_state_id: str | None,   # (D5) the persisted composition_state_id — NOT state.id
) -> None:
    """Surface a pending llm_prompt_template review EVENT for every llm node that
    carries a pending auto-staged llm_prompt_template requirement and does not yet
    have a pending event for it (idempotent). Backend-derived: the review is created
    against the FINAL frozen skeleton at turn finalization, so it can never go stale
    against a later skeleton mutation (elspeth-e51216d305 Case B). No-op when there
    is no session or no persisted state id. See (D1)-(D5) in the plan.
    """
    if session_id is None or current_state_id is None:
        return
    sessions_service = self._require_sessions_service()
    pending = await sessions_service.list_interpretation_events(UUID(session_id), status="pending")
    already = {
        (e.affected_node_id, e.user_term)
        for e in pending
        if e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE
    }
    for site in interpretation_sites(state):
        if site.kind is not InterpretationKind.LLM_PROMPT_TEMPLATE:
            continue
        if (site.component_id, site.user_term) in already:
            continue
        node = next((n for n in state.nodes if n.id == site.component_id), None)
        if node is None:
            continue
        options = node.options if isinstance(node.options, Mapping) else {}
        prompt_template = options.get("prompt_template")
        if not isinstance(prompt_template, str) or not prompt_template:
            continue
        await sessions_service.create_pending_interpretation_event(
            session_id=UUID(session_id),
            composition_state_id=UUID(current_state_id),
            affected_node_id=site.component_id,
            tool_call_id=f"backend_auto_surface:{uuid4()}",  # (D1)
            user_term=site.user_term,
            kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
            llm_draft=prompt_template,
            model_identifier=self._model,                                   # (D2)
            model_version=self._model,                                      # (D2)
            provider=self._availability.provider or "unknown",             # (D2)
            composer_skill_hash=self._composer_skill_hash,                  # (D2)
        )
```

Confirm imports at top of service.py: `from uuid import UUID, uuid4`; `from collections.abc import Mapping`; `interpretation_sites` and `InterpretationKind` (already imported — verify). `state.id` and `state.nodes[*].id/.options` exist on `CompositionState` (verify field names against composer/state.py during implementation; the resolve path uses `node["id"]/node["options"]` on dict form — here `state` is a `CompositionState` object, so use attribute access `node.options`).

- [ ] **Step 4: Run to verify it passes** — same command. Expected: PASS.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(composer): backend helper to auto-surface prompt-template reviews at finalization"`

---

## Task 2: Wire the helper into `_try_terminate_no_tools` (before the gates)

**Files:**
- Modify: `src/elspeth/web/composer/service.py:2257`
- Test: `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`

- [ ] **Step 1: Write the failing test** — drive the no-tool-calls finalization with a composition that has a pending auto-staged prompt-template requirement and NO pending event (simulating the model finishing without surfacing it). Assert the returned outcome is NOT the orphan/fail-closed result (the prompt-template site is now satisfied by the backend) AND a pending llm_prompt_template event now exists. Mirror the setup of the existing `test_missing_prompt_template_review_event_forces_review_tool_retry` (~931) but invert the expectation.

```python
@pytest.mark.asyncio
async def test_finalization_auto_surfaces_prompt_template_and_does_not_orphan_block(...):
    outcome = await service._try_terminate_no_tools(... state=state_with_unsurfaced_pt ...)
    # The prompt-template site is satisfied by backend surfacing, so the turn is not
    # blocked as an orphan on account of the prompt-template review.
    events = await sessions_service.list_interpretation_events(sid, status="pending")
    assert any(e.kind is InterpretationKind.LLM_PROMPT_TEMPLATE for e in events)
    assert outcome.action == "return"  # finalized (no prompt-template orphan)
```

- [ ] **Step 2: Run to verify it fails** — `pytest ...::test_finalization_auto_surfaces_prompt_template_and_does_not_orphan_block -v`. Expected: FAIL (today: either an orphan-block result or a repair `continue`, and no backend event).

- [ ] **Step 3: Implement the wiring — SURFACE LATE (D4).** THREE edits:

  **(a) Thread the state id (D5)** — add a `current_state_id: str | None` parameter to `_try_terminate_no_tools` (signature at service.py:2209-2227) and pass the outer-loop local `current_state_id` (service.py:2555/2655) at its call site (grep `_try_terminate_no_tools(` for the caller — the no-tool-calls branch of the main compose loop).

  **(b) Auto-surface immediately BEFORE the orphan gate** — insert right before `orphaned_sites = await self._missing_pending_interpretation_review_sites(` at service.py:2334 (NOT at 2257). At this point every repair branch above has declined/continued, so `state` is the FINAL frozen skeleton:

```python
        # Backend-derived surfacing (elspeth-e51216d305 Case B): surface every
        # llm node's auto-staged llm_prompt_template review against the FINAL
        # frozen skeleton, immediately before the fail-closed orphan gate. The
        # skeleton can only grow across repair turns (each re-invokes the model);
        # here, past every repair branch, no further mutation occurs this turn,
        # so the surfaced review can never go stale (cf. surface-early = Case B
        # in the repair loop). The orphan gate below (unfiltered) then sees the
        # PT event present; if this helper ever no-ops, it stays fail-closed.
        await self._auto_surface_prompt_template_reviews(
            state, session_id=session_id, current_state_id=current_state_id,
        )
```

  **(c) Task 2b — post-filter PT out of the repair message at 2263** — so the LLM (which now rejects PT, Task 3) is not pestered to surface it during the repair loop. Replace the `if missing_interpretation_sites:` block at 2263:

```python
            if missing_interpretation_sites:
                # llm_prompt_template is surfaced by the backend at finalization
                # (before the orphan gate), NOT by the model — exclude it from the
                # repair ask so we don't pester the model for a kind it rejects.
                # The orphan gate at 2334 stays UNFILTERED (fail-closed backstop).
                model_repairable = tuple(
                    site
                    for site in missing_interpretation_sites
                    if site[2] is not InterpretationKind.LLM_PROMPT_TEMPLATE
                )
                if model_repairable:
                    llm_messages.append(
                        {
                            "role": "user",
                            "content": _pending_interpretation_review_repair_message(
                                model_repairable,
                                next_turn=repair_turns_used + 1,
                            ),
                        }
                    )
                    return _TerminateOutcome(action="continue", repair_turns_delta=1)
```

  (Confirm the site tuple shape is `(component_id, user_term, kind)` — investigator `missing-sites-logic` verified `site_key = (site.component_id, site.user_term, site.kind)`; `site[2]` is the kind.)

- [ ] **Step 4: Run to verify it passes** — same command. Then run the whole dispatch suite: `pytest tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -q`. Expected: PASS (update any test that assumed prompt-template orphan-blocking — see Task 5).

- [ ] **Step 5: Commit (EXPLICIT PATHS — never `git add -A`; the tree carries uncommitted foreign files `.claude/settings.json`, `.gitignore`, `.mcp.json`)** — `git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py && git commit -m "feat(composer): surface prompt-template reviews at finalization before the orphan gate"`

---

## Task 3: Reject `kind=llm_prompt_template` at the `request_interpretation_review` boundary

**Files:**
- Modify: `src/elspeth/web/composer/tools/sessions.py` (~1609, after argument parse)
- Test: `tests/unit/web/composer/test_request_interpretation_review_tool.py`

- [ ] **Step 1: Write/flip the failing test** — change `test_request_interpretation_review_accepts_prompt_template_kind` (~719) to assert the tool now REJECTS `kind="llm_prompt_template"` with a `ToolArgumentError` naming the allowed kinds; rename it `test_request_interpretation_review_rejects_prompt_template_kind`.

```python
@pytest.mark.asyncio
async def test_request_interpretation_review_rejects_prompt_template_kind(...):
    with pytest.raises(ToolArgumentError) as exc:
        await _handle_request_interpretation_review(arguments={
            "affected_node_id": node_id, "kind": "llm_prompt_template",
            "user_term": f"llm_prompt_template:{node_id}", "llm_draft": prompt_template,
        }, ...)
    assert "llm_prompt_template" in str(exc.value)
    assert "backend" in str(exc.value).lower()
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/web/composer/test_request_interpretation_review_tool.py::test_request_interpretation_review_rejects_prompt_template_kind -v`. Expected: FAIL (tool currently accepts the kind).

- [ ] **Step 3: Implement the guard** — in `_handle_request_interpretation_review`, immediately after the `_validate_mutation_arguments` parse (~1609), mirroring the existing rejection idiom at ~1279:

```python
    if parsed.kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
        raise ToolArgumentError(
            argument="kind",
            expected="vague_term, invented_source, pipeline_decision, or llm_model_choice",
            actual_type=(
                "llm_prompt_template — the prompt-template review is surfaced "
                "automatically by the backend at turn finalization; do not request it"
            ),
        )
```

- [ ] **Step 4: Run to verify it passes** — same command, then `pytest tests/unit/web/composer/test_request_interpretation_review_tool.py -q`. Expected: PASS (fold/remove the now-moot `test_request_interpretation_review_rejects_stale_prompt_template_draft` — that tool path is unreachable for this kind; keep a comment noting the backend now owns it).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(composer): reject LLM-initiated llm_prompt_template review surfacing (backend owns it)"`

---

## Task 4: Update the composer skill prompt

**Files:**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md` (~551-552, ~600-605, ~686-690)

NOTE: skills are LLM prompts, not code — do NOT add a grep-the-text "test" (project doctrine: that is theatre). Verify by re-running the composer (Task 6). The live web service reads this file via an @lru_cache'd module import; **restart `elspeth-web.service` after editing** so staging picks it up.

- [ ] **Step 1: Edit** — remove/replace the instruction to call `request_interpretation_review` for `kind="llm_prompt_template"` (lines ~551-552), and the "surface both review cards" wording (~600-605), with: "The prompt-template review is auto-staged on every LLM node and is surfaced for you automatically by the backend after you finish composing — do NOT call request_interpretation_review with kind=\"llm_prompt_template\". Continue to author the prompt as prompt_template_parts with interpretation_ref slots for vague terms, and surface vague_term / invented_source / pipeline_decision / llm_model_choice reviews as before." Keep the prompt_template_parts construction guidance (~635-690) but drop the manual llm_prompt_template surfacing step.

- [ ] **Step 2: Refresh the stored expected composer_skill_hash — THE GATE IS REAL (verified 2026-06-06).** `assert_skill_hash_unchanged_on_disk(name, expected_sha256)` exists at composer/skills/__init__.py:65-82 and compares a STORED expected hash against `sha256(on_disk_text)`. Editing `pipeline_composer.md` changes the on-disk hash, so the stored `expected_sha256` MUST be recomputed and co-landed or this gate fails. Find its caller/constant: `git grep -n "assert_skill_hash_unchanged_on_disk\|expected_sha256"`. NOTE — the literal-hash values in unit tests (`composer_skill_hash == "a" * 64`, `== "sha256:composer-skill"`) are injected TEST STUBS, NOT the real on-disk hash; they are unaffected by the skill edit (verified). Commit: `git add -A && git commit -m "docs(composer-skill): backend owns prompt-template review surfacing"` (run hooks — this is NOT a pure-markdown-only change once the stored hash co-lands).

---

## Task 5: Reconcile existing tests that pinned the old behavior

**Files:**
- Modify: `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py` (`test_missing_prompt_template_review_event_forces_review_tool_retry` ~931)
- Search: `git grep -ln "llm_prompt_template" tests/unit/web/composer tests/unit/web/sessions`

- [ ] **Step 1: Enumerate** — `git grep -n "request_interpretation_review\|llm_prompt_template" tests/unit/web/composer tests/unit/web/sessions` and read each hit; identify tests asserting "LLM surfaces / must surface prompt-template via the tool" or "repair loop retries for a missing prompt-template event."

- [ ] **Step 2: Update** — `test_missing_prompt_template_review_event_forces_review_tool_retry`: the backend now surfaces the event, so finalization should NOT force an LLM retry for the prompt-template kind. Rewrite to assert backend surfacing instead (or retarget it to a vague_term site, which DOES still force a retry). For each other pinned test, update the expectation to match backend ownership; do NOT delete coverage — retarget it.

- [ ] **Step 3: Run the full composer + sessions suites** — `pytest tests/unit/web/composer tests/unit/web/sessions -q`. Expected: PASS.

- [ ] **Step 4: Commit** — `git add -A && git commit -m "test(composer): reconcile tests with backend-derived prompt-template surfacing"`

---

## Task 6: Gates, local verification, and staging verification

**Files:** none (verification)

- [ ] **Step 1: Local gate surface** — run the CI-equivalent locally (memory: run lints before push): `ruff check`, `ruff format --check`, `mypy` on changed files; `pytest tests/unit/web tests/integration/web/composer -q`. Fix any fallout.

- [ ] **Step 2: Tier-model gate** — `env PYTHONPATH=elspeth-lints/src ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth`. The new helper reads persisted state + calls create_pending (a sessions-service method) — confirm it is in the unflagged helper family (no new allowlist entry); the pre-existing boot_probe.py crash is operator-owed, unrelated. If the new method IS flagged, STOP and surface to operator (it may need a @trust_boundary decorator or an allowlist entry + judge sign-off).

- [ ] **Step 3: Deploy + staging verify** — `sudo systemctl restart elspeth-web.service` (sudoers-granted; picks up HEAD of /home/john/elspeth + the skill change). Then run the tutorial battery reusing the saved token (temp config dropping globalSetup), e.g. `HARNESS_BATCH_ID=verify-caseB HARNESS_BATCH_SIZE=8 npx playwright test --config=<temp> tutorial-reliability`. SUCCESS CRITERIA: vague_term runs reach graduation (turn 7) with ZERO `frontend-state-machine` bricks; specifically confirm a run that previously matched Case B (composer surfaces the prompt-template review late) now graduates. Delete the temp config after.

- [ ] **Step 4: Record + close** — update notes/tutorial-reliability/2026-06-06-e51216d305-vague_term-card-ROOTCAUSE.md with the verified result; comment the commit SHAs + staging evidence on elspeth-e51216d305; `filigree close elspeth-e51216d305` once a Case-B run graduates clean.

---

## Self-review notes

- **Spec coverage:** Path 2 = backend surfacing (Tasks 1-2) + LLM rejection (Task 3) + skill (Task 4) + test reconciliation (Task 5) + verify (Task 6). Committed Case-A fix (22168c227) is the backstop and unchanged.
- **Open items — RESOLVED by pre-flight verification (2026-06-06):**
  - (i) `CompositionState` field names — **DRIFT FOUND & CORRECTED.** No `.id` on `CompositionState`; composition_state_id must be threaded as `current_state_id` (see D5 + Task 1/2 edits). `NodeSpec.id` (str) and `NodeSpec.options` (Mapping) confirmed correct. `InterpretationReviewSite.{component_id,user_term,kind}` confirmed correct.
  - (ii) `_handle_request_interpretation_review` at sessions.py:1573; `_validate_mutation_arguments(` parse at 1604 — Task 3 insertion "~1609" confirmed; reject idiom at 1279 confirmed.
  - (iii) skill-hash gate — **CONFIRMED REAL** (`assert_skill_hash_unchanged_on_disk`); Task 4 Step 2 upgraded from optional to mandatory; test literal-hashes are stubs (safe).
  - Insertion seam (2257/2258), `create_pending_interpretation_event` (sessions/service.py:2661), `list_interpretation_events` (3406) — all confirmed.
- **Risk:** the (D1) audit-provenance sentinel is the one decision needing operator/judge visibility — call it out in the PR. The compose-loop is audit-sensitive and was last touched today (33f05f186); keep changes minimal and run the full composer suite.

---

## Task 7: Adversarial-review remediation (added 2026-06-07 — workflow `wf_d240d901-e0b` reviews)

The implement+review fleet landed Impl-1/2/3 GREEN but two reviewers returned CHANGES_REQUESTED with TWO confirmed high-severity findings (verified against source). Both must be fixed before gating/commit.

**(HIGH-1) A second no-tool finalize path bypasses auto-surface + the orphan gate.** Confirmed: `_classify_and_budget_turn`'s B-4D-3 budget-exhaustion last-chance finalize (`service.py:2247`, fires when `turn_has_mutation` AND the composition budget is exhausted AND the bonus model call returns no tool calls) calls `_finalize_no_tool_response` directly and returns `_ClassifyOutcome(action="return",...)` at 2265 — running NEITHER `_auto_surface_prompt_template_reviews` NOR `_missing_pending_interpretation_review_sites`. This path always has `turn_has_mutation=True`, so it can carry a fresh LLM node with a pending PT requirement and no surfaced event; with the new LLM-side rejection (Impl-2) the model can no longer surface PT here either → it reliably ORPHANS a required PT review (finalizes runnable-pending with no card). The orphan gate must be universal.
- **Fix:** extract the auto-surface + orphan-gate + finalize tail of `_try_terminate_no_tools` (service.py:2435-2526) into a shared async method `_surface_and_finalize_no_tools(self, *, assistant_message, state, session_id, current_state_id, progress, recorder, initial_version, user_id, last_runtime_preflight, runtime_preflight_cache, session_scope, message, mutation_success_seen) -> ComposerResult` that returns either the blocked ComposerResult (orphan) or the finalized result. Caller threads `repair_turns_used` (only `_try_terminate_no_tools`) + persisted ids. Call it from BOTH finalize sites. Thread `session_id` into `_classify_and_budget_turn` (add param; pass at call site service.py:2798); at 2247 use `current_state_id=persist.current_state_id` (the loop persists the mutation BEFORE classify — dispatch(2727)→persist(2758)→`current_state_id=persist.current_state_id`(2770)→classify(2798) — so `state`==persisted(`persist.current_state_id`), the create_pending gate holds).

**(HIGH-2) Node-id-keyed dedup bricks a PT review after a genuine multi-turn prompt edit.** Confirmed: `already = {event.affected_node_id ...}` (service.py:1357) keys on node id alone. Turn 1 surfaces E_A (skeleton_A); turn 2 edits prompt to B (re-stages the requirement, but no code clears the persisted pending event E_A); turn 2 auto-surface sees E_A → SKIPS → no E_B. The orphan gate matches on (node,term,kind) → satisfied by stale E_A → finalizes; the user resolves E_A → the Case-A skeleton-hash gate rejects forever; the LLM can no longer re-surface (Impl-2). The node's PT review is bricked. No false approval (integrity holds) but a NEW liveness regression this diff introduces.
- **Fix:** make the dedup DRAFT-AWARE — skip a node only if a pending PT event exists whose `llm_draft == the node's current options.prompt_template`. On skeleton drift, surface a fresh event for the current skeleton. (The stale event lingers cosmetically; full supersede/cancel needs the governed SUPERSEDED enum — **file as a follow-up**, out of scope here.)

**(LOW-a) Multiplicity mirror.** `_has_pending_prompt_template_requirement` (service.py:1393) returns True on the FIRST match; the writer-boundary gate `_matching_pending_requirement_index` requires EXACTLY ONE and raises on 0/>1. Tighten the helper to count and return True only on exactly-one (so a duplicate-requirement node is skipped to the fail-closed orphan gate, never an opaque 500). Unreachable today but they must not diverge.

**(LOW-b) Provenance comment.** Add a one-line comment at the D2 site (service.py:1386-1387) noting `model_version == model_identifier == self._model` is intentional: a backend-derived surface has no LLM response object to derive a provider-resolved model from (divergence from the LLM-surfaced path's `safe_response_model(response)` is deliberate and audit-honest).

**Tests (TDD):** (HIGH-1) drive the budget-exhaustion finalize with a pending auto-staged PT requirement + no event → assert the PT event IS surfaced (sentinel) AND a genuine bare-token orphan blocks on that path. (HIGH-2) surface E_A for skeleton A, edit the node prompt to B, run auto-surface → assert a fresh resolvable E_B (draft==B) is created. (LOW-a) two pending PT requirements → helper returns False.
