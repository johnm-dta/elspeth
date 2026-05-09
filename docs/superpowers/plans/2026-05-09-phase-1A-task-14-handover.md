# Phase 1A Task 14 — Entry-point Handover

**Branch:** `feat/composer-progress-persistence-1a`
**Worktree:** `/home/john/elspeth/.worktrees/composer-progress-1a`
**Plan:** `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md`
**Prior handovers (still valid for standing context):**
- `docs/superpowers/plans/2026-05-09-phase-1A-task-9-handover.md`
- `docs/superpowers/plans/2026-05-09-phase-1A-task-10-handover.md`

**Date written:** 2026-05-09
**Author:** continuation session of Task 10

---

## TL;DR

- Task 10 (`_insert_composition_state` helper + call-site sweep + plan §2294 test-side provenance fixes) is **done and held in working tree**. Tests for the helper green (6/6); composition_states writer sweep complete (3 sites); 17 test-side direct inserts now supply `provenance`. Tier-model cascade unchanged at exactly 10 (predicted). No commits.
- **Next: Task 14 (`add_message` rewrite + full call-site sweep).** Plan §2749. This is the **largest** remaining 1A task and is also the **commit boundary** — the atomicity rule held Tasks 0/1/2/3/4/7/8/9/10 in working tree across this and prior sessions; Task 14 is what finally lands them.
- Read the prior handovers (`task-9-handover.md` first, then `task-10-handover.md`) for standing context that this document does not repeat — atomicity rule §23, plan §94 scanner allowlist, pre-commit hook scope, venv hygiene, no-stash discipline (now 5 documented violations), and the fingerprint-cascade mechanics. **This document only diffs forward from Task 10's close.**

## What changed since the Task 10 entry handover

### Done in this session (held in working tree)

| Path | Lines added (net) | What it is |
|---|---|---|
| `src/elspeth/web/sessions/service.py` | +130 | `_enveloped_state_column` (module-level) + `_insert_composition_state` (method on `SessionServiceImpl`) with B1 internal-version-allocation |
| `src/elspeth/web/sessions/service.py` | (mutation) | `fork_session._sync` — composition_states insert refactored to call helper under `_session_write_lock`; local `_enveloped` deleted |
| `src/elspeth/web/sessions/service.py` | (mutation) | `save_composition_state._try_insert_state` — wrapped in `_session_write_lock`, `provenance="session_seed"` added, local `_enveloped` replaced with module helper, retry loop kept as belt-and-suspenders with explanatory comment |
| `src/elspeth/web/sessions/service.py` | (mutation) | `set_active_state._try_insert_revert` — wrapped in `_session_write_lock`, `provenance="session_seed"` added, retry loop kept as belt-and-suspenders |
| `tests/unit/web/sessions/test_persist_compose_turn.py` | +210 | 6 new RED-then-GREEN tests: returns_id, allocates_contiguous_versions (B1 sequential), session_write_lock_serializes_sqlite_same_session_state_version_allocation (B1 concurrent), versions_are_per_session (B1 WHERE filter), requires_session_write_lock (negative — pre-allowlisted), rejects_unknown_provenance (CHECK constraint) |
| `tests/unit/web/sessions/test_static_direct_writers.py` | net +28 | 2 new entries (`_insert_composition_state` raw_string + sqlalchemy_insert_call), 2 rewrites (save_composition_state + set_active_state — purpose strings updated to "lock-retrofit-in-place not helper-route" per plan §2128-2133), 1 stale entry removed (`fork_session._sync` composition_states — writer no longer exists at that site) |
| `tests/unit/web/sessions/test_models.py` | (mutation) | 4 sites — `provenance="session_seed"` added to schema-test direct inserts |
| `tests/unit/web/blobs/test_service.py` | (mutation) | 8 sites — `provenance="session_seed"` added to setup-fixture inserts |
| `tests/unit/web/composer/test_tools.py` | (mutation) | 5 sites — `provenance="session_seed"` added to composer-tools fixtures |

The plan §2294 third bullet (test-side direct inserts) was **caught by the advisor** mid-session, not on the Task 10 handover's done-criteria checklist. The fix landed before Task 10 declared done — see pitfalls §1 below.

### Known state at session close (verify before Task 14 begins)

```bash
git status --short
# Expect:
#  M src/elspeth/web/sessions/models.py            # Task 0/Schema (held)
#  M src/elspeth/web/sessions/service.py           # Tasks 8 + 9 + 10 (held)
#  M tests/unit/web/blobs/test_service.py          # Task 10 §2294 (held)
#  M tests/unit/web/composer/test_tools.py         # Task 10 §2294 (held)
#  M tests/unit/web/sessions/test_models.py        # Task 10 §2294 (held)
#  M tests/unit/web/sessions/test_persist_compose_turn.py    # Tasks 8+9+10 helper tests (held)
#  M tests/unit/web/sessions/test_static_direct_writers.py   # Task 8/9/10 allowlist (held)
# ?? docs/superpowers/plans/2026-05-09-phase-1A-task-9-handover.md   # standing context
# ?? docs/superpowers/plans/2026-05-09-phase-1A-task-10-handover.md  # standing context
# ?? docs/superpowers/plans/2026-05-09-phase-1A-task-14-handover.md  # this handover
# ?? tests/unit/web/sessions/test_audit_access_log.py        # Task 0/Schema
# ?? tests/unit/web/sessions/test_chat_messages.py           # Task 0/Schema (~15 tests)
# ?? tests/unit/web/sessions/test_composition_states.py      # Task 0/Schema
```

**Total uncommitted:** 7 modified, 6 untracked. All held for Task 14 atomic cutover.

```bash
# Confirm helper tests (Tasks 8 + 9 + 10) green:
.venv/bin/python -m pytest tests/unit/web/sessions/test_persist_compose_turn.py
# Expect: 18 passed

# Confirm static-direct-writer guard green:
.venv/bin/python -m pytest tests/unit/web/sessions/test_static_direct_writers.py
# Expect: 7 passed

# Confirm test-side provenance sweep green:
.venv/bin/python -m pytest tests/unit/web/blobs/test_service.py tests/unit/web/composer/test_tools.py
# Expect: all green except 1 chat_messages.sequence_no test that test_models.py owns

# Full sessions baseline — note the new ceiling:
.venv/bin/python -m pytest tests/unit/web/sessions/
# Expect: 80 failed, 323 passed
# (This is down from the 174 baseline observed at Task 10 start. The -94
# delta is plan-prescribed: 6 new helper tests + 85 fixes from provenance
# kwargs in inline writers + 3 fixes from §2294 test-side sweep.)

# Tier-model: STILL 10 violations — predicted cascade, do not fix here
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
# Expect: VIOLATIONS FOUND: 10. Fingerprints drifted again from Task 10's
# helper-add and call-site sweep; line numbers have shifted but the count
# and category set are unchanged. Atomic Task 14 cutover refreshes these.

# Sanity: every chat_messages-side failure should fall in the 80 remaining,
# and they all share the same root cause:
.venv/bin/python -m pytest tests/unit/web/sessions/test_fork.py 2>&1 | grep "NOT NULL" | head -3
# Expect: NOT NULL constraint failed: chat_messages.sequence_no (×many)
```

## Task 14: spec & shape

**Read first:** plan §2749 onwards (Task 14: Atomic `add_message` rewrite — signature, behaviour preservation, protocol, and full call-site sweep). The plan document is ~5000 lines; Task 14's section is the longest of the 1A tasks.

**Why Task 14 is qualitatively different from Tasks 0/7/8/9/10:**

- It is the **commit boundary**. Every prior task's working-tree-only changes get committed here, in carefully ordered commits that preserve atomicity (schema + writers must land together; partial commits between Task 3 and Task 14 would break CI for downstream branches).
- It is the **chat_messages writer cutover**. The remaining 80 baseline failures (all `chat_messages.sequence_no NOT NULL`) get fixed when call sites route through `_insert_chat_message` instead of inserting directly. This mirrors Task 10's composition_states sweep but for chat_messages.
- It is the **fingerprint-cascade refresh point**. The tier-model allowlist (`config/cicd/enforce_tier_model/web.yaml`) gets refreshed atomically as part of the cutover — the 10 deferred violations get either re-fingerprinted to their post-cutover positions or removed (if the underlying defensive pattern got rewritten by the sweep).

**Major moving pieces (from a Task 10-eye-view; the plan is authoritative):**

| Piece | Plan section | What | Indicator |
|---|---|---|---|
| 1. `add_message` signature change | §14.x intro | Rewrite to use `_reserve_sequence_range` + `_insert_chat_message` under `_session_write_lock`. Match the protocol Phase 3 callers want. | Method body shrinks dramatically; retry loop deletes |
| 2. Call-site sweep — `routes.py` | §14.x route sweep | Every HTTP-route caller of legacy chat_messages-insert paths gets routed through the new method | Multiple route handlers updated |
| 3. Call-site sweep — `fork_session` | §14.6 | The fork chat_messages-batch insert at the bottom of `fork_session._sync` (currently still a `conn.execute(insert(chat_messages_table), msg_records_data)` batch) gets routed through the helper. The session-write-lock that Task 10 wrapped just the composition_states portion in extends to cover the chat_messages portion too. | `state_version`-style logic unchanged; the lock scope grows |
| 4. Tier-model allowlist refresh | §94 | `config/cicd/enforce_tier_model/web.yaml` — 10 cascade fingerprints get re-fingerprinted to the post-Task-14 AST shape (or deleted if the defensive pattern got rewritten away) | `enforce_tier_model.py check` exits 0 |
| 5. Static-direct-writer allowlist refresh | scanner | `_REVIEWED_ALLOWLIST` entries for `add_message._sync` (and whatever else routes through the helper) get removed; the `_insert_chat_message` entry's purpose is updated from "the SOLE chat_messages writer the call-site sweep at Task 14 will route every existing writer through" to past tense | Scanner test still green; entry count drops |
| 6. Atomic commit sequence | (operator-driven) | Multiple commits, in dependency order: schema → helpers → call-site sweep → allowlist refresh. CI's "every commit must build" rule means the order matters | Each intermediate commit must independently produce a green test suite |

### What about plan §2294's "either canonical row factory" alternative?

Task 10's session went with the explicit `provenance="session_seed"` kwarg per insert (option B in the plan's "either... or..." phrasing). Task 14 may want to introduce a `_make_composition_state(conn, session_id, **kwargs)` helper in `tests/unit/web/conftest.py` to consolidate, mirroring the existing `_make_session`. **Not a Task 14 requirement** — the inserts already work. File as an OQ-followup if the bulk allowlist updates suggest it.

## Expected gate noise (NOT bugs you should fix on the side)

### 1. The 80 remaining failures are Task 14's targets

Don't dismiss them as baseline. They are the work. Each one falls into one of these categories:

- `chat_messages.sequence_no NOT NULL` — Task 14's call-site sweep fixes by routing through `_insert_chat_message` which calls `_reserve_sequence_range`. Most of test_fork.py (25 failures) and chunks of test_service.py / test_routes.py / test_datetime_timezone.py.
- LiteLLM error-redaction tests — these may require their own call-site adjustments depending on whether the redaction path goes through `add_message`. Check before assuming.
- `test_routes.py::TestRecomposeConvergencePartialState::test_recompose_convergence_save_operational_error_preserves_422_body` — single test that was already failing pre-Task-10; investigate as part of the Task 14 sweep, do not skip.

When the sweep is complete, **the count should drop to 0** for the chat_messages-related failures. Anything that does not is either Phase 1B/Phase 3 territory or a genuine new regression — investigate.

### 2. Tier-model cascade refresh

The 10 deferred violations live at AST positions that have drifted twice now (Task 9's helper-add, Task 10's helper-add + call-site sweep). Task 14's sweep will drift them a third time. **Refresh the allowlist atomically with the cutover commit:**

```bash
# After Task 14's source edits land in working tree, regenerate fingerprints:
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
# Read the "Key:" lines from the failure output, update web.yaml entries
# accordingly. Do this in the SAME commit as the source change so HEAD is
# never internally inconsistent.
```

If after the sweep the violation count drops below 10 (because the defensive pattern got rewritten away by the sweep), DELETE the corresponding allowlist entries. Do not leave dead allowlist entries — the handover's pitfall §5 ("no stale promises") applies bidirectionally.

### 3. Static-direct-writer scanner re-balancing

Several `_REVIEWED_ALLOWLIST` entries describe writers that Task 14 routes through helpers. After the sweep:

- The entry for `SessionServiceImpl.add_message._sync` (chat_messages, line ~668 in test_static_direct_writers.py) gets DELETED if `add_message` no longer contains a direct insert.
- The entry for `SessionServiceImpl.fork_session._sync` (chat_messages, line ~688) gets DELETED if §14.6's chat_messages batch routing through the helper succeeds.
- The entry for `SessionServiceImpl._insert_chat_message` gets a **purpose-string rewrite** from "the SOLE chat_messages writer the call-site sweep at Task 14 will route every existing writer through" to past tense reflecting that the routing has happened.
- New entries may be needed for any NEW direct writer the sweep introduces (unlikely but possible if a route handler has a special-case path).

### 4. Pre-commit hook coordination

The atomic commit sequence has many files moving at once. Pre-commit hooks (ruff format, mypy, etc.) will run on each commit. Plan the commit order so each intermediate commit is independently green:

1. Schema-only commit (models.py + the 3 untracked schema test files: test_audit_access_log, test_chat_messages, test_composition_states). Tests pass because the inline writers haven't been touched yet — they were already failing on the constraint, and the new tests are explicit assertions. Wait, no — pre-Task-14 working tree has the writer fixes already. The commit ordering decision is non-trivial; **read the plan's §94 / §23 commit-ordering guidance before deciding**.
2. Helper-and-sweep commit (service.py + test_persist_compose_turn.py + test_static_direct_writers.py).
3. Test-side §2294 sweep (test_models.py + test_blobs/test_service.py + test_composer/test_tools.py).
4. Tier-model allowlist refresh (config/cicd/enforce_tier_model/web.yaml).

If the operator wants a **single** commit covering all of 1A, that's also legal — atomicity is preserved either way. Confirm intent before commit-shaping.

## Suggested first action for the next session

1. `git status` — verify the working-tree inventory in the "Known state" section above.
2. Run the four baseline gate commands above to confirm starting-state matches this handover. Expect 80 failed / 323 passed in sessions; helper tests 18/18; scanner 7/7; tier-model exactly 10.
3. Read plan §2749 onwards (Task 14 spec) carefully — especially:
   - The `add_message` new signature and how it integrates with `_reserve_sequence_range` + `_insert_chat_message`.
   - §14.6's fork-session sweep ordering (state then messages, both under one lock).
   - The plan's commit-ordering guidance under §23 and §94.
4. Brief operator with a concrete check-in proposal that includes:
   - Whether to land Task 14 as a single mega-commit or as ordered chunks.
   - Whether to introduce a `_make_composition_state` test fixture as part of this task or defer to OQ-followup.
   - The expected end-state failure count (target: 0 chat_messages-related failures; non-zero only if Phase 1B/Phase 3 territory remains).
5. Wait for operator go-ahead before starting RED for the Task 14 changes. **Task 14 is the commit boundary — once started, the team is committed (literally) to landing the full sequence. The operator must explicitly authorise this transition.**

## Pitfalls discovered THIS session (carry forward)

In addition to the prior handovers' pitfalls, these are NEW lessons from the Task 10 cycle:

| Pitfall | What happened | Lesson |
|---|---|---|
| **Handover done-criteria is not the full plan §X.X "Files:" list** | Plan §2293-2294's third bullet (test-side direct inserts in test_models / test_blobs / test_composer) was Task 10's scope but did not appear on the handover's done-criteria checklist. I declared done; the advisor caught it; the fix landed before declaring complete. | Always re-read the plan section's **Files:** preamble in addition to the handover's done-criteria checklist. The handover summarises; the plan is authoritative. When in doubt, run the test paths the plan names rather than relying on the handover's enumeration. |
| **Worktree confusion at session start** | Initial `git status` and gate checks ran in the main repo (`/home/john/elspeth`), not the worktree (`/home/john/elspeth/.worktrees/composer-progress-1a`). The handover's expected-inventory check failed loudly and surfaced the confusion immediately, but had I trusted the wrong tree's diff (which was 8 unrelated files from concurrent panel-evals work) it would have wasted significant time. | Verify the worktree path **before** running any gate. The handover header includes the canonical worktree path; `cd` or use absolute paths from there. The session-start git status snapshot in the system prompt is for the main repo, NOT the worktree, and can mislead. |
| **Call-site sweep chunks legitimately fix many baseline failures — handover ±1 rule applies to helper-add only** | The Task 10 handover said "If [the 174 baseline] does [change by even 1], the implementation has touched something downstream — investigate before proceeding." The call-site sweep dropped 174 → 80 (-94 net). I momentarily worried this was a regression. Investigation confirmed all -94 were plan-prescribed (provenance fix in inline writers + §2294 test-side fixes). | Read the ±1 rule scoped to its phase. The helper-add chunk (where I added `_insert_composition_state` without touching any inline writer) preserved the 174 — that's where the rule binds tightly. The call-site sweep chunk is *intended* to fix tests that the schema CHECK constraint had broken; large failure-count drops are the success signal, not a regression flag. Verify by sampling failure modes (NOT NULL chat_messages.sequence_no = baseline; provenance NULL = your fix) rather than by total counts. |
| **`composition_states_table.insert()` (method form) vs `insert(composition_states_table)` (function form)** | Initial grep used only the function form and missed ~14 sites in test files that use the method form. The static-direct-writer scanner detects both, so the scan itself was not wrong, but my manual enumeration was. | When grepping for SQLAlchemy inserts, search **both** forms: `insert(<table>` and `<table>.insert()`. The two compile to identical SQL; teams converge on one or the other for stylistic reasons but mixed usage exists in this codebase. |
| **Bulk replace_all needs a uniquely-anchorable trailing-context substring** | Task 10's §2294 sweep had 17 sites across 3 files. Eight test_blobs sites with identical trailing pattern (`is_valid=True,\n<spaces>created_at=datetime(2026, 1, 1, tzinfo=UTC),\n<spaces>)`) collapsed into one `replace_all=True` Edit. Five test_composer sites with a different identical trailing pattern collapsed into another. Test_models's 4 sites had three subtly-different patterns that needed individual edits. | When facing N similar edits, read 3-5 of them BEFORE deciding individual-vs-bulk. Look at the *trailing* lines (right before the closing `)`) — those are the most stable anchor across variant function bodies. The leading lines often differ (different field orderings, comments, source dicts) but the closing `is_valid` + `created_at` pair tends to be invariant. |

## Done-criteria for Task 14

Task 14 is "done and committed" when:

- [ ] `add_message` rewritten to the new signature; routes through `_reserve_sequence_range` + `_insert_chat_message` under `_session_write_lock`.
- [ ] All HTTP-route call sites of legacy chat_messages-insert paths route through the new `add_message` (or directly through `_insert_chat_message` where appropriate).
- [ ] §14.6 fork-session sweep complete: chat_messages batch insert routed through helper under the same `_session_write_lock` that already covers the composition_states portion (Task 10 set this up).
- [ ] Tier-model `enforce_tier_model.py check` exits 0 (allowlist refreshed in the same commit as the source changes — no stale fingerprints, no orphaned entries).
- [ ] Static-direct-writer scanner test green; allowlist entries for sites that no longer have direct writers REMOVED (not just have stale purposes); `_insert_chat_message` purpose rewritten to past tense.
- [ ] Sessions test suite: 0 chat_messages-related failures (i.e., no `NOT NULL constraint failed: chat_messages.sequence_no` errors anywhere in the test output).
- [ ] Any LiteLLM-error-redaction or recompose-convergence tests that were in the 80-baseline are either now passing or have been investigated and explicitly attributed to a different phase (with a filigree issue filed).
- [ ] Ruff + format + mypy all clean.
- [ ] All previously-held tasks (0/1/2/3/4/7/8/9/10) committed in the agreed order; CI green at each commit boundary; pre-commit hooks satisfied without `--no-verify` (per CLAUDE.md and feedback memory).
- [ ] Phase 1A done. The branch is mergeable to `main` modulo whatever review gate the operator chooses.

## Self-contained property reminder

Same as the prior handovers: **plan and code are ground truth**. If anything in this document contradicts the plan or current source, the plan and source win. This handover summarises and points; it does not authoritatively define.

Cross-reference points-of-truth:

- **Plan:** `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1A-schema-current-writer-safety.md`
- **Atomicity rule:** plan §23
- **Scanner allowlist requirement:** plan §94
- **Task 14 spec proper:** plan §2749 onwards (the longest 1A task)
- **Project memory:** `~/.claude/projects/-home-john-elspeth/memory/MEMORY.md` (auto-loaded each session)
- **No-stash rule:** `feedback_no_git_stash.md` — **5 documented violations** as of Task 10 close, treat the literal keystroke "stash" as STOP.
- **Worktree convention:** `feedback_worktree_convention.md` — `.worktrees/<name>/` inside the project; verify worktree path before running gates (new pitfall §2 above).
- **Plan-vs-handover authority:** new pitfall §1 — the plan's `Files:` preamble overrides the handover's done-criteria when they disagree.
