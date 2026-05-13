# Handover: Composer Progress Persistence Phase 1A — Task 9 entry point

**Author:** Claude (Opus 4.7) session 2026-05-09
**Successor entry point:** Task 9 (`_insert_chat_message` helper)
**Branch:** `feat/composer-progress-persistence-1a`
**Worktree:** `/home/john/elspeth/.worktrees/composer-progress-1a`
**Plan:** `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md`
**Sidecar:** `…review.json` — verdict `APPROVED_WITH_WARNINGS / GO_WITH_WARNINGS` as of 2026-05-08; 11 blocking issues resolved, 4 operational warnings preserved (W1 audit_access_log inert in 1A, W2 SQLite single-process locking, W3 PostgreSQL deferred to 1C, W4 IntegrityError retry loops).

---

## TL;DR

- Task 8 (`_reserve_sequence_range`) is **implemented in working tree, not committed**. Per plan §1845 + operator decision, it stacks alongside the four pre-existing held artifacts and the scanner extension produced this session. Task 14 atomic cutover is the landing point.
- HEAD remains 8 commits ahead of `RC5.1` — **no commits this session**.
- The static-guard scanner has been extended in working tree to honor plan §94's negative-test allowlist requirement. The allowlist tuple already covers Tasks 8/9/10 negative tests pre-emptively, so Task 9's negative test lands without scanner change.
- A new mechanical coupling discovered: **tier-model fingerprints in `web.yaml` are tightly coupled to `service.py` AST positions**. They must move together; standalone web.yaml commits create HEAD self-inconsistency. This forced deferral of the scanner extension into Task 14's atomic cutover.

---

## Where we are (verify before starting)

```bash
cd /home/john/elspeth/.worktrees/composer-progress-1a
git status --short
git log --oneline RC5.1..HEAD | wc -l    # → 8
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py \
                            tests/unit/web/sessions/test_static_direct_writers.py \
                            tests/unit/web/sessions/test_chat_messages.py \
                            tests/unit/web/sessions/test_composition_states.py \
                            tests/unit/web/sessions/test_audit_access_log.py \
                            -p no:cacheprovider 2>&1 | tail -3
# Expected: 37 passed
```

Expected `git status --short`:

```
 M src/elspeth/web/sessions/models.py                       # Tasks 1-4 schema (pre-Task-8)
 M src/elspeth/web/sessions/service.py                      # Task 8 helper added 2026-05-09
 M tests/unit/web/sessions/test_persist_compose_turn.py     # Task 7 + Task 8 tests
 M tests/unit/web/sessions/test_static_direct_writers.py    # Scanner extension added 2026-05-09
?? tests/unit/web/sessions/test_audit_access_log.py         # Task 4 (pre-Task-8)
?? tests/unit/web/sessions/test_chat_messages.py            # Tasks 1-2 (pre-Task-8)
?? tests/unit/web/sessions/test_composition_states.py       # Task 3 (pre-Task-8)
```

If the worktree state diverges from this, **stop and reconcile** before continuing — `git diff` against the named commits below to understand drift.

---

## Commits ahead of `RC5.1` (8, unchanged)

```
ec875364  feat(sessions): add session write-lock helpers (Task 7)
5c53a724  docs(composer): extend Phase 1 marker for composition_states.provenance enum
b83cc766  test(web): add shared session row factories
ab320de8  docs(composer): fix Task 0 marker — plan KEEPS session_fork in enum
35b2477a  test(web): add static direct-writer guard
1e38fc38  docs(runbooks): session-DB recreation procedure
3f5f5972  docs(composer): mark Phase 1 persistence spec snippets superseded
dce21743  docs(plan): patch B11 in Phase 1A preflight greps
```

---

## Working-tree artifacts (held until Task 14 atomic cutover)

| Artifact | What's in it | Source |
|---|---|---|
| `src/elspeth/web/sessions/models.py` | Tasks 1+2+3+4 schema deltas: `chat_messages` + 4 columns + biconditional CHECKs + composite FK CASCADE + indices; `composition_states.provenance` NOT NULL + 6-value CHECK; new `audit_access_log_table` | pre-2026-05-09 |
| `src/elspeth/web/sessions/service.py` | **Task 8** `_reserve_sequence_range(self, conn, session_id, *, count) -> int` inserted after `_session_write_lock` (~line 337). Calls `_assert_session_write_lock_held` first; SQLAlchemy 2.x `select(func.coalesce(func.max(...), 0))` idiom; `count < 1` raises `ValueError`. | 2026-05-09 |
| `tests/unit/web/sessions/test_chat_messages.py` | 15 schema tests (Tasks 1+2) | pre-2026-05-09 |
| `tests/unit/web/sessions/test_composition_states.py` | 4 schema tests (Task 3) | pre-2026-05-09 |
| `tests/unit/web/sessions/test_audit_access_log.py` | 3 schema tests (Task 4) | pre-2026-05-09 |
| `tests/unit/web/sessions/test_persist_compose_turn.py` | Task 7 reentrancy tests + **Task 8** 6 new tests + `service` fixture annotated `-> SessionServiceImpl` + 2 thread-worker tests annotated `service: SessionServiceImpl` | 2026-05-09 |
| `tests/unit/web/sessions/test_static_direct_writers.py` | **Scanner extension** (plan §94 compliance): `LockDisciplineNegativeTest` dataclass; `_LOCK_DISCIPLINE_NEGATIVE_TESTS` tuple with 3 pre-populated entries (Tasks 8/9/10 negative tests); `check_lock_discipline` `allowlist` kwarg + filter; 2 new self-tests for the allowlist mechanism | 2026-05-09 |

All seven artifacts are **ruff/mypy clean** as of session end. Pre-commit hooks scan the working tree and will gate them.

---

## ACTIVATED INVARIANTS — read this before writing Task 9 code

### State as of session end (post-Task-8 working-tree state)

The static-guard test (`tests/unit/web/sessions/test_static_direct_writers.py`) is **active in three orthogonal modes**, all currently green:

1. **`check_helper_lock_assertions`** — every helper named `_reserve_sequence_range` / `_insert_chat_message` / `_insert_composition_state` MUST contain `self._assert_session_write_lock_held(conn, session_id, caller=...)` in its body. Currently inventory = 1 (Task 8's helper); will be 2 after Task 9, 3 after Task 10.

2. **`check_inline_state_version_allocation`** — no inline `SELECT MAX(version) + 1` or `SELECT MAX(sequence_no) + 1` outside those helpers. Currently the scanner is dormant on this dimension because Task 14's writer migration hasn't landed; it activates fully when callers stop allocating inline.

3. **`check_lock_discipline`** — every caller of those helpers MUST be wrapped in `with self._session_write_lock(conn, session_id):` in the same transaction, **or** appear in `_LOCK_DISCIPLINE_NEGATIVE_TESTS`. The allowlist already pre-covers Tasks 8/9/10 negative tests. The lookup key is `(path, enclosing_symbol, helper_name)` and matching is exact on all three (regression-tested).

### What this means for Task 9

When you implement `_insert_chat_message`:
- The helper **must** open with `self._assert_session_write_lock_held(conn, session_id, caller="_insert_chat_message")` — otherwise `check_helper_lock_assertions` fails.
- The plan-provided test `test_insert_chat_message_requires_session_write_lock` (plan §1924) calls the helper without the lock to verify the precondition fires. This is **already pre-allowlisted** — no scanner change needed.
- The other plan-provided tests **must** wrap helper calls in `with service._session_write_lock(conn, sid):`. Otherwise `check_lock_discipline` flags them.

If you forget the precondition, `pytest tests/unit/web/sessions/test_static_direct_writers.py` fails before commit. Run it explicitly as your gate — pre-commit pytest doesn't fire on test files generally; the static guard is a normal pytest target that has to be invoked.

---

## Critical: tier-model fingerprint coupling (NEW DISCOVERY 2026-05-09)

The handover from the prior session warned about a "fingerprint cascade" but underspecified its consequence. Here is the precise mechanic, learned the hard way this session:

### Mechanic

`scripts/cicd/enforce_tier_model.py` computes each finding's fingerprint as:

```python
fingerprint = sha256(f"{rule_id}|{ast_path}|{ast.dump(node, include_attributes=False)}").hexdigest()[:16]
```

`ast_path` is `"/".join(path_stack)`, where each entry is `field_name[index]` (e.g., `body[0]/body[12]/body[2]`). When `_reserve_sequence_range` was inserted at body[k] of `SessionServiceImpl`, the `body[N]` index of every subsequent method shifted by 1, recomputing 8 downstream fingerprints in `web.yaml`.

### Coupling consequence

The pre-commit hook entry is:

```yaml
- id: enforce-tier-model
  entry: .venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
  pass_filenames: false
  types: [python]
```

`pass_filenames: false` means the hook scans the **full working tree**, not staged files. `types: [python]` means it activates whenever any `.py` file is staged. So:

- Staging *anything Python* triggers a tier-model scan against the post-Task-8 `service.py` working tree.
- HEAD's `web.yaml` references pre-Task-8 fingerprints.
- → Mismatch → pre-commit fails.

### What this rules out

- **Standalone scanner extension commit** — pre-commit fails because of unrelated working-tree state.
- **Standalone `web.yaml` update commit** — would create HEAD self-inconsistency: `web.yaml` references AST positions that only exist after `service.py` lands. A fresh checkout from HEAD without working-tree state would fail tier-model.
- **Standalone Task 8 helper commit** — also broken by plan §1845 ("`_reserve_sequence_range` depends on `chat_messages.sequence_no`") since that NOT NULL column is in held `models.py`.

### What this means for Task 9

Task 9 will **also** insert a method into `SessionServiceImpl` (`_insert_chat_message`), shifting more fingerprints. The same coupling applies. Don't even attempt a Task 9 standalone commit.

The right cadence is: add Task 9 helper + tests in working tree, run gates, leave uncommitted. The web.yaml update for both Task 8 and Task 9 fingerprint shifts lands in **Task 14's single atomic cutover commit**, alongside service.py / models.py / tests / scanner extension / writer migration.

---

## Filigree observations (still open as of 2026-05-09)

- **`elspeth-obs-5304c41054`** (P2, expires 2026-05-22) — Static guard scanner has a qualified-import coverage gap: matches `Name("chat_messages_table")` but not `Attribute(attr="chat_messages_table")` (e.g., `models.chat_messages_table`). Test files in the uncommitted set use qualified imports. **Resolution**: at Task 14, add `_extract_tracked_table_id(node)` helper accepting both `Name` and `Attribute(attr=...)` forms; land alongside scanner extension, allowlist updates, and atomic cutover.

- **`elspeth-obs-de8b6642a4`** (P2, expires 2026-05-22) — Task 14 cutover sizing: 174 pre-existing test failures from Phase 1A NOT NULL columns, concentrated in `test_service.py` / `test_routes.py`. Production direct-writer inventory in plan §57-68 is 15 sites; test fixture migration is comparable or larger. Use to size Task 14's review effort. **Verified at session end (2026-05-09)**: count is still exactly 174 — unchanged baseline.

- **`elspeth-obs-58c90aef32`** (P3, expires 2026-05-22) — Phase 1A preflight evidence (2026-05-08): inventory matches plan §57-68; W2 single-worker confirmed; B11 fixed in `dce21743`. Use to populate the eventual cutover PR body.

### NEW observation worth filing (CLI didn't expose `observe`; MCP tool is deferred — note here for future filing)

- **Tier-model fingerprint–service.py coupling**. The tight coupling between `web.yaml` AST-positional fingerprints and `service.py` method order forces every `SessionServiceImpl` method addition to bundle a `web.yaml` update. There's no clean path to standalone helper commits while pre-commit's tier-model scope is `pass_filenames: false`. Long-term: consider whether tier-model fingerprints should encode `(class_name, method_name, rule_id)` instead of positional path, removing the cascade. Short-term: this constraint reinforces the §23 atomicity rule — Task 14 lands everything together.

---

## Plan execution map (updated 2026-05-09)

```
Task 0       ✅  spec supersession marker (committed: 3f5f5972)
Task 18      ✅  staging session-DB recreation runbook (committed: 1e38fc38)
Preflight 4  ✅  static direct-writer guard (committed: 35b2477a)
                Note: scanner extension for plan §94 negative-test allowlist
                added 2026-05-09 in working tree; not yet committed.
Task 1       ✅  chat_messages schema (held, models.py + test_chat_messages.py)
Task 2       ✅  partial unique index on (session_id, tool_call_id) (held)
Task 3       ✅  composition_states.provenance (held, models.py + test_composition_states.py)
Task 4       ✅  audit_access_log table (held, models.py + test_audit_access_log.py)
Task 7       ✅  session write-lock helpers (committed: ec875364) — static guard ACTIVATED
Task 8       ✅  _reserve_sequence_range (held in working tree 2026-05-09)
                  6 tests + helper. All gates green.
                  Plan §94 scanner allowlist extended in working tree (held).

   ◄── NEXT: Task 9

Task 9       _insert_chat_message helper. Plan §1850-2111. Real-coding work.
              MUST include _assert_session_write_lock_held precondition.
              Working-tree hold per §1845 — same atomicity rule as Task 8.
              Static guard scanner allowlist already pre-covers the negative test.
              Tier-model fingerprint cascade WILL recur (more methods, more shifts).
              Don't attempt standalone commit.
Task 10      _insert_composition_state helper + minimum-touch updates of
              save_composition_state, set_active_state callers.
              Plan §2112-2752. Same atomicity, same pre-allowlisted negative test.
Task 14      ATOMIC CUTOVER. Bundles:
                - models.py + test_chat_messages/composition_states/audit_access_log
                - service.py with all three helpers
                - test_persist_compose_turn.py with Tasks 8/9/10 tests
                - test_static_direct_writers.py scanner extension (this session's work)
                - web.yaml fingerprint updates (post-Task-8/9/10 cascade)
                - 15-site writer migration (add_message, fork_session, save_composition_state, set_active_state, …)
                - Per-test fixture migration to satisfy NOT NULL columns
                - Scanner qualified-import gap fix (obs-5304c41054)
              Plan §2753+. Largest commit in 1A. Plan to size with obs-de8b6642a4.
Preflight 5  Direct-write inventory in cutover PR body (PR-creation step).
```

---

## Critical context — read all of this before substantive work

### Plan atomicity rule (§23)

Schema-breaking metadata (NOT NULL `chat_messages.sequence_no`, NOT NULL `chat_messages.writer_principal`, NOT NULL `composition_states.provenance`) is held in the working tree. Task 14 commits all of it together with every writer migration. **Don't commit schema piecemeal** — the writers in the codebase don't yet supply the new columns, so committing `models.py` alone produces a runtime broken HEAD.

### Plan §94 — scanner negative-test allowlist (now satisfied in working tree)

The plan §94 paragraph required the static guard to "explicitly allowlist a negative test" for each lock-required helper. `35b2477a` shipped without this mechanism; it was added in working tree on 2026-05-09. The allowlist is `_LOCK_DISCIPLINE_NEGATIVE_TESTS` and matches on `(path, enclosing_symbol, helper_name)` exactly (regression-tested). Tasks 9 and 10 negative tests are **pre-populated**; you don't need to extend the scanner.

### Pre-commit hook scope (gotcha)

Pre-commit hooks (ruff, ruff-format, mypy, enforce-tier-model, check-contracts, enforce-frozen-annotations) all use `pass_filenames: false` and scan the **working tree**. So an unstaged file with lint errors blocks an unrelated staged commit. Run all four gates against the working tree before attempting any commit:

```bash
.venv/bin/python -m ruff check src/ tests/ scripts/ examples/
.venv/bin/python -m ruff format --check src/ tests/ scripts/ examples/
.venv/bin/python -m mypy src/elspeth
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Memory `feedback_doc_only_commits_no_ci.md`: doc-only commits MAY use `--no-verify`. Code commits must NOT.

### Worktree venv hygiene (essential)

Worktree's `.venv` is Python 3.13.1 with `[all]` extras. **Never run `uv pip install` from the worktree without `--python` or `source .venv/bin/activate` first** — uv's venv discovery walks up to `/home/john/elspeth/.venv` (main) and silently modifies it. Memory `feedback_uv_venv_leak.md`. Verify both intact when in doubt:

```bash
.venv/bin/python -c "import elspeth; print(elspeth.__file__)"
# → .../composer-progress-1a/src/elspeth/__init__.py
/home/john/elspeth/.venv/bin/python -c "import elspeth; print(elspeth.__file__)"
# → /home/john/elspeth/src/elspeth/__init__.py
```

### Other memory pointers worth re-reading

- `feedback_eval_attribution_can_mislead.md` — verify section numbers against plan headings before inheriting framing from prior summaries (this is exactly how the prior handover got Task 8's commit cadence wrong).
- `feedback_locked_in_buggy_expectations.md` — wave of test failures after a structural fix is the bug landing visibly; update tests rather than reverting the fix.
- `feedback_no_git_stash.md` — never stash; commit to a branch instead.
- `feedback_repeated_out_of_scope_is_underscoping.md` — third-tasking 'no X' is a flag, not authority. When you find an apparent scope creep, surface it before quietly extending.
- `feedback_fix_errors_you_encounter.md` — "I didn't cause this" is never a stopping condition.
- `project_db_migration_policy.md` — delete the old DB, no Alembic.
- `project_rc5ux_demo_prep_scope.md` — branch is RC5.1 demo prep; Phase 1A IS landing pre-demo per operator decision.

---

## Pitfalls already encountered THIS session (don't repeat)

1. **Prior-handover Task 8 cadence was wrong.** Said "Real commit"; plan §1845 says "Do not commit standalone." Verified by reading the plan section, not the summary. Lesson: when starting from a handover, re-grep `^## Task ` headings against the summary before building on it. Memory: `feedback_eval_attribution_can_mislead.md`.

2. **Tier-model fingerprint cascade is HEAD-coupling, not just a nuisance.** The prior handover described it as "When tier-model fails, sed-substitute fingerprints in `web.yaml` and re-run." That's incomplete — the substituted `web.yaml` cannot be committed alone (HEAD self-inconsistency), so the cascade forces atomicity with the service.py change. Document the coupling at session start so future you doesn't re-discover it.

3. **`35b2477a` shipped 80% of the static guard.** Plan §94 explicitly required the negative-test allowlist; the scanner didn't have it. Found by trying to land Task 8's negative test and getting flagged. Lesson: when a plan paragraph says "must support X" and the implementation doesn't, fix the implementation rather than working around it (memory `feedback_fix_errors_you_encounter.md`).

4. **Pytest fixture annotations don't propagate to test parameters.** Annotating `def service(...) -> SessionServiceImpl` is necessary but not sufficient; the test parameter `service` in `def test_X(service: SessionServiceImpl)` also needs the annotation, otherwise nested-function returns through `service.method()` calls are typed `Any` and trigger `[no-any-return]` from mypy. The fix is per-test annotation.

5. **`# noqa: SIM117` only applies when the outer `with`'s body is solely an inner `with`.** A `with A:` followed by `_make_session(); with B:` (two-statement body) doesn't trigger SIM117, so a noqa there is unused (RUF100). Carryover from the prior handover's pitfall #4 — confirmed live this session.

6. **`filigree observe` doesn't exist on the CLI.** It's MCP-only, and MCP tools are deferred in this harness. Note observations in handover files instead, or use `mcp__filigree__observe` after fetching its schema via `ToolSearch`.

---

## Suggested first action for next session

### 1. Verify state (cheap)

```bash
cd /home/john/elspeth/.worktrees/composer-progress-1a
git status --short
git log --oneline RC5.1..HEAD | wc -l    # → 8
.venv/bin/python -c "import elspeth; print(elspeth.__file__)"  # check venv intact
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py \
                            tests/unit/web/sessions/test_static_direct_writers.py \
                            tests/unit/web/sessions/test_chat_messages.py \
                            tests/unit/web/sessions/test_composition_states.py \
                            tests/unit/web/sessions/test_audit_access_log.py \
                            -p no:cacheprovider 2>&1 | tail -3
# Expected: 37 passed
```

### 2. Read Task 9 briefing in the plan (orientation)

Plan §1850-2111. Pay attention to:
- Helper signature: `_insert_chat_message(self, conn, *, session_id, role, content, …) -> str` (returns `id`).
- Plan §94 protocol: precondition assertion via `_assert_session_write_lock_held` BEFORE any DB read.
- The plan provides ~7 tests for Task 9 (positive/negative/cross-session/parent-assistant/etc.) — count them up front so you know what RED looks like.
- `_assert_parent_assistant_message` helper (Task 9.b in the plan) — also held until Task 14.

### 3. Brief the operator before implementing

The cadence this session (and prior) has been: "fresh briefing → confirm scope → implement → report → next task." Operator prefers explicit checkpoints between tasks, especially when an activated invariant changes shape. Task 9 doesn't change invariant *shape* (the static guard is already activated), but it changes *coverage* (helper inventory grows from 1 to 2; the second negative-test allowlist entry becomes live).

Concrete check-in to propose: "Task 9 working-tree hold per §1845, same as Task 8. ~7 tests + helper. Tier-model fingerprint cascade will recur but stays uncommitted (Task 14 cutover). OK to proceed?"

### 4. Watch for the NEW tier-model failures from Task 9

After Task 9's helper insertion, expect another batch of fingerprint shifts in `web.yaml`. **Do not** try to update `web.yaml` standalone — accumulate the working-tree drift; Task 14 lands the consolidated `web.yaml` update. The pre-commit gate will be RED against working tree until then; this is expected.

If you need to verify gates green during Task 9 work, run them as **manual command invocations** not via `git commit`:

```bash
.venv/bin/python -m ruff check src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
.venv/bin/python -m mypy src/elspeth/web/sessions/service.py tests/unit/web/sessions/test_persist_compose_turn.py
.venv/bin/python -m pytest tests/unit/web/sessions/ -p no:cacheprovider 2>&1 | tail -10
```

The full `enforce_tier_model.py` will fail with the 8 known-cascade fingerprints from Task 8 plus whatever Task 9 adds. That's fine and expected; it's a Task 14 deliverable.

---

## Done-criteria for Schedule 1A (per plan §4237-4250)

1. Direct-write inventory captured in PR body. (Step 5 of preflight, at PR creation.)
2. ✅ Static guard fail-closes on new SQLAlchemy/raw-SQL writers (`35b2477a`).
3. No direct writer can insert without `sequence_no` / `writer_principal` / `provenance`.
4. `add_message` preserves cross-session guards, `updated_at`, `raw_content`, `ChatMessageRecord` return hydration.
5. `fork_session` preserves stored `writer_principal` (no role-keyed default).
6. Public route responses + composer prompt history exclude `role="audit"`.
7. Audit breadcrumb persistence failures fail-soft + class-name-only.
8. ✅ Staging session-DB recreation runbook (`1e38fc38`).
9. SQLite current-behavior tests pass.
10. 1A documented as SQLite-only deployable.
11. Ruff + mypy + tier-model + `pytest tests/unit/web/ tests/integration/web/` all green.
12. Follow-up review confirms 1A doesn't block 1B.

---

## Self-contained property

This handover is intended to be paste-loadable into a fresh session without prior conversation history. If something in here is unclear or contradicts the plan or current code, **trust the plan and the code** and update this handover. Memory `feedback_eval_attribution_can_mislead.md` applies to handovers as much as to plans.
