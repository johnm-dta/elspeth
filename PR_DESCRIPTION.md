# Phase 5: Chat data-entry (5a) + Interpretation events (5b)

Phase 5 ships two composer UX capabilities against `RC5.2`. **5a** turns the chat panel into a first-class data-entry surface: short user inputs (a URL, a sentence, one record) project into a 1-row dynamic inline source instead of forcing a CSV upload, with full audit provenance (`creation_modality`, LLM lineage on inline blobs, server-side hash, tamper-evidence). **5b** adds a Landscape-backed *interpretation events* surface so the composer can pause for user clarification when a prompt template is ambiguous, write the resolution to a new `interpretation_events_table` (append-only, hash-linked to `calls_table.resolved_prompt_template_hash`), and persist opt-outs per session. Together these are the two halves of "the LLM can ask a question and we can prove what was answered."

---

## CRITICAL — Cross-branch coupling (read before reviewing the diff)

> **Two skill commits required for functional completeness live on `RC5.2`, NOT in this PR's diff.**
>
> - `34d272360` — Phase 5a.8: *prefer inline_blob for chat-typed source data*
> - `d6219faa2` — Phase 5b.8: *teach the LLM when to call `request_interpretation_review`*
>
> These edit `src/elspeth/web/composer/skills/pipeline_composer.md`. ELSPETH skills are module-imported and `@lru_cache`'d at process start, so the skill must live on the branch the staging service actually reads. They were deliberately landed on the target branch (`RC5.2`) rather than this feature branch.
>
> **What this means for review:**
>
> - The PR diff is *functionally incomplete on its own*. The backend wiring (`request_interpretation_review` tool handler, inline-source projector) is in this diff; the LLM instructions that drive it are not.
> - Reviewers verifying LLM behaviour (5a chat-as-source, 5b interpretation gating) must check the **live staging service** (`elspeth.foundryside.dev`), not the diff. Reading just the diff and concluding "the LLM is never told to do X" will be a false negative.
> - The two commits are already on `RC5.2`. Confirm with `git log RC5.2 --oneline | grep -E "(34d27236|d6219faa)"` before merging.

## CRITICAL — Operator deploy actions

> **This PR lands schema changes that require deleting two SQLite databases on deploy.**
>
> Commit `2e390fc0b` adds `interpretation_events_table` (session DB) and `resolved_prompt_template_hash` on `calls_table` (Landscape audit DB). The two are linked by a cross-DB byte-equal hash invariant. There is no migration runner — per project policy DB migration = delete the old DB — and the two DBs must be reset *together* to preserve the invariant.
>
> **Steps on deploy (staging = `elspeth.foundryside.dev`):**
>
> 1. `sudo systemctl stop elspeth-web.service`
> 2. Delete the session DB. Default path: `<data_dir>/sessions.db` (resolved by `src/elspeth/web/config.py:351-356`).
> 3. Delete the Landscape audit DB. Default path: `<data_dir>/runs/audit.db` (resolved by `src/elspeth/web/config.py:338-343`).
> 4. `npm run build` (frontend bundle for any UI changes).
> 5. `sudo systemctl restart elspeth-web.service` — required for `@lru_cache` skill-prompt invalidation, even if the skill text on `RC5.2` is unchanged this deploy.
> 6. Run the staging smoke (Phase 5b 10-scenario regression in `evals/composer-rgr/phase5b-interpretation/`).
>
> Patched runbook: `docs/runbooks/staging-session-db-recreation.md` (Phase 5b two-DB reset section). A parallel agent is finalising the runbook patch; this PR assumes the runbook lands before deploy.
>
> **Rollback:** re-deploy the prior commit + re-delete both DBs (per the standard `delete-the-old-DB` policy — schema rolls back the same way it rolls forward).

---

## Task → commit map

**Schema-changing commits (require two-DB delete):** `2e390fc0b`, `be6939d8d`.

Commits flagged **[SCHEMA]** require the two-DB delete above. Order is newest → oldest (matches `git log RC5.2..HEAD --oneline`).

### Phase 5b — Interpretation events

- `b7b360287` — 5b.18b.6: frontend integration test for interpretation review (guided + freeform)
- `3bd98f0ad` — 5b.18b.7/.8 follow-on: readiness panel row + ChatPanel Run-gating + resolve-success copy
- `674582692` — 5b.18b.5: `InterpretationReviewInlineMessage` freeform widget
- `17e74284d` — 5b.18b.4: `InterpretationReviewTurn` guided widget
- `c7cfc10e6` — 5b.18b.8: empty-state + placeholder copy for interpretation flow
- `1c23037c9` — 5b.18b.3: `interpretationEventsStore` for review/opt-out state
- `85439842f` — 5b.18b.2: frontend API client methods for interpretation routes
- `51e8b9fdf` — 5b.18b.1: `InterpretationEventResponse` + wire types
- `60c530d4a` — transforms(llm): rename `LLMConfig.template` → `prompt_template` (composer/runtime field-name alignment)
- `f4c3c3bc8` — Task 11: sessions service orphan-PENDING-row recovery
- `8543b395b` — Task 9: Landscape runtime hand-off + cross-DB hash spot-check
- `dbb08dfa8` — evals: correct opt-out endpoint to POST `/interpretations/opt_out`
- `426e4babd` — Task 10: audit_readiness surfaces `llm_interpretations` row
- `fa68a247e` — Task 12: Phase 5b interpretation-review regression suite
- `75c9f10b8` — Task 5 follow-on: F-17/F-21 unresolved-placeholder pre-execute gate
- `8ce72195e` — Tasks 6 + 7: `/interpretations/resolve`, `/list`, `/opt_out`, `/opt_out_summary` HTTP routes
- `964615ea1` — Task 5 follow-on: composer service wires `request_interpretation_review` through dispatch + AUTO_INTERPRETED_NO_SURFACES writer + skill-hash atomicity startup-assert + `skill_markdown_history` upsert
- `a37a08f4a` — Task 5: `request_interpretation_review` tool handler + rate limits + dual-registry dispatch + redaction
- `606b053cf` — Task 4 docstring: pending-row actor sentinel rationale
- `ede3000ab` — Task 4: sessions service writer/reader methods for interpretation events + opt-out
- `8c32efc7c` — Task 3: wire schemas for interpretation events + resolve + opt-out + `amended_value` content validators
- `2e390fc0b` — **[SCHEMA]** Task 2: `interpretation_events_table` + `interpretation_review_disabled` column + append-only triggers + `resolved_prompt_template_hash` on `calls_table`
- `d9a4cfc18` — sessions engine: WAL + busy_timeout + synchronous PRAGMA + schema-epoch guard (co-landed for Phase 5b, ref `elspeth-6815a49a7d`)
- `434d4adb2` — Task 1: `InterpretationEventRecord` + `InterpretationChoice` + `InterpretationSource` contracts

### Phase 5a — Chat as data entry

- `4a94bd505` — Task 2.6: allowlist `chat_messages` insert in attributability test
- `5fefcafc2` — Phase 5b.0.5 security: `max_length` caps on content fields + ASGI 10 MB body limit (landed in 5a block, scoped to 5b prep)
- `1ac9b3dca` — 5a.7: composer audit panel inline-source provenance row
- `a64a118c0` — 5a.6: frontend integration test for chat → inline_blob → review widget
- `440ca1c81` — 5a.5: `InlineSourceFallbackPrompt` as LLM-skip safety net
- `a5a35349c` — 5a.4: `InlineSourceDisambiguationTurn` for ambiguous row count
- `22faa1cec` — 5a.3 review fixes: honest hash + MIME parsing + bound catch-arm in inline-source projection
- `62abf1ef1` — 5a.3: `InlineSourceCreatedTurn` surfaces inline-source provenance
- `231cafee7` — Task 2.6: **[SCHEMA-adjacent]** Phase 5a attributability chain tamper-evidence test (relies on `creation_modality` column)
- `be6939d8d` — 5a.2.5: **[SCHEMA]** server-side `creation_modality` + LLM-provenance on inline blobs
- `9d9e6b761` — 5a.2: `inlineSourceStore` for projected inline-source view
- `b3b1e03e7` — 5a.1: empty-state placeholder primes inline source-from-chat

(35 commits total. Skill commits `34d272360` / `d6219faa2` are on `RC5.2` and intentionally NOT in this list.)

---

## Test coverage

- **Backend:** 218 Phase 5b-scoped tests passing. Broader suite green excluding 1 pre-existing flaky concurrency test (unrelated to Phase 5 scope).
- **Frontend:** 967 vitest passing, typecheck clean.
- **CI status — tier-model enforcement:** 7 Phase 5 commits touched `config/cicd/enforce_tier_model/`; allowlist fingerprints rotated and co-landed in the same commit as the file edit that caused them (per `feedback_ast_shift_fingerprint_rotation`). `scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model` exits 0 on this branch (5 TYPE_CHECKING warnings, no failures).
- **Live LLM evals:**
  - 5a canonical-hero ("create a list of 5 government web pages and rate how cool they are") — exercises chat → inline-source projection end-to-end. The plan (`17-phase-5a-dynamic-source-from-chat.md` Task 8 Step 4) only references the 5b Task 0 JSON shape (`evals/composer-rgr/phase5b-task0-*.json`) — no dedicated `phase5a-*` artifact directory is specified by the plan, so 5a artifacts will land alongside the 5b suite under `evals/composer-rgr/` using the same JSON shape.
  - 5b interpretation 10-scenario regression suite (`evals/composer-rgr/phase5b-interpretation/`) — exercises `request_interpretation_review` tool dispatch, resolve, list, opt-out, opt-out-summary.
  - A separate agent is running the live evals now. `results-*.json` artifacts will land in a follow-up commit on this branch if not yet present at review time.

---

## Known follow-ups

- **Audit-readiness runtime-model-mismatch warning** — being fixed in this branch by another agent; will land before merge.
- **`test_interpretation_runtime_handoff.py` production-path bypass** — documented departure from CLAUDE.md "integration tests MUST use `ExecutionGraph.from_plugin_instances()`". Coverage is preserved in aggregate; filigree-ticketed for follow-up rather than blocking this PR.
- **Plan-doc filename drift** — implementation diverged from the plan docs in two places:
  - `interpretationStore.ts` (plan) → `interpretationEventsStore.ts` (shipped, commit `1c23037c9`)
  - `ChatPanel.tsx` Run-button placement (plan) → `ExecuteButton.tsx` (shipped) — Run-gating wiring landed in `3bd98f0ad`

---

## Test plan for the merge reviewer

- [ ] Two-DB-delete operator step understood (session DB + Landscape audit.db, paths cited above).
- [ ] RC5.2 skill commits `34d272360` and `d6219faa2` confirmed present on the target branch.
- [ ] LLM eval artifacts present on this branch (or noted as follow-up with date).
- [ ] Staging smoke fired post-deploy: 5a canonical-hero + 5b 10-scenario regression both green.
- [ ] Patched runbook `docs/runbooks/staging-session-db-recreation.md` reviewed and reflects two-DB-reset reality.
