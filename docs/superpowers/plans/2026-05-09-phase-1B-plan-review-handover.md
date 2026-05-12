# Phase 1B Implementation Kickoff — Handover (2026-05-09, post-review)

> **For: future Claude Code session that will start implementing the approved Phase 1B plan.**
> **Read this first.** It captures branch state, what's already done, what's next, and the exact verification commands.
> **Supersedes** the earlier plan-review handover that lived at this same path — that handover's work is now complete and the file has been recycled for the implementation phase.

## TL;DR (60 seconds)

Phase 1A is implementation-complete. Phase 1B plan has been through 2 review passes (4 reviewers each) + 2 fix passes; all four reviewer domains now sign off (REALITY-OK / ARCH-APPROVE / QUALITY-APPROVE / SYSTEMS-APPROVE). Plan is committed at `88df6ceb`.

Operator confirmed: Phase 3 (compose-loop integration) is on the pre-demo critical path, so 1B is the gating implementation work.

**Next session's job:** start executing the 1B plan, starting with Task 1.

## Working-tree state (verify first)

```bash
cd /home/john/elspeth/.worktrees/composer-progress-1a
git log --oneline -3
# Expected:
#   88df6ceb docs(composer-1b): apply 4-reviewer feedback to Phase 1B plan
#   af0a2918 test(sessions): thread writer_principal through merged augment/replace tests
#   da702480 Merge remote-tracking branch 'origin/RC5.1' into feat/composer-progress-persistence-1a

git status --short
# Expected: only this file (1B handover) untracked. Working tree otherwise clean.

wc -l docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1B-compose-turn-primitive-audit-semantics.md
# Expected: 3337 lines.
```

If any of those don't match, something has shifted between sessions — investigate before starting Task 1.

## What was done (this session — 2026-05-09)

1. Verified Phase 1A is implementation-complete and `af0a2918` is the right base.
2. Ran 4-reviewer parallel review against the 1B plan (reality / architecture / quality / systems). Found 5 must-fix blockers + minor MAJORs.
3. Dispatched fix-it agent (Pass 3); it applied the 5 must-fix edits, growing plan from 2578 → 3323 lines.
4. Re-ran 4-reviewer parallel re-review (Pass 4). Three domains conditional on 5 mechanical residuals (A1-A5); systems conditional on operator decision (B).
5. Operator answered B: Phase 3 is pre-demo critical → 1B is on the demo path.
6. Dispatched fix-it agent for A1-A5; verified all 5 edits in place, plan now 3337 lines.
7. Re-ran quality + architecture re-reviewers (the two domains touched by A1-A5); both APPROVED.
8. Committed plan + 3 Phase 1A task handover docs as `88df6ceb`.

## What's next (start here)

### A) Execute Phase 1B plan, starting with Task 1

The plan file is the source of truth:
`docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-1B-compose-turn-primitive-audit-semantics.md`

It contains 15 tasks across:

- Compose-turn primitive plumbing
- Validation guards (`_validate_tool_call_id_set_equality` — Task 11 Step 3c)
- Async cancellation contract (Task 11 Step 3d — commit-wins semantics)
- INV-AUDIT-AHEAD backward-direction proof (Task 15)

Read the **Schedule 1B Done When** checklist at the *very end* of the plan (lines ~3317-3337) before starting Task 1 — it tells you what "done" looks like, including the consolidated **pre-merge gate** (item 7). If you can't articulate how Task 1 contributes to one of the Done-When items, you're not ready to start it.

### B) Pre-merge gate (lives in the plan, restated here for prominence)

Before opening or merging the 1B PR, all four must pass from worktree root:

```bash
.venv/bin/python -m pytest tests/unit/web tests/integration/web && \
.venv/bin/python -m mypy src/ && \
.venv/bin/python -m ruff check src/ tests/ && \
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Plus:

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_static_direct_writers.py -v
```

(The static-writer scanner check is item 6 of the Done-When; the `&&` block is item 7.)

### C) Pre-existing pollution to handle before either PR opens

Two items inherited from before this session — neither is 1B-domain but both block the merge gate as written:

1. **Pre-existing ruff I001** in `tests/unit/evals/lib/test_composer_rgr_score.py:14` (exists on `origin/RC5.1`, not introduced by this branch). One-line import-sort fix. **Recommendation:** land as a separate cleanup commit on this branch *before* opening the 1A PR, so the literal `ruff check src/ tests/` command is mechanically green.

2. **This file (1B implementation-kickoff handover)** is currently untracked. Operator decision (this session): **leave ephemeral**, do not commit. If it accumulates noise in `git status` during 1B implementation work, you may move it to `notes/` rather than commit it — but check with the operator before doing so.

### D) Phase 1A PR (independent of 1B)

Phase 1A is mergeable independently. The 1A merge gate has been verified clean at `af0a2918` (2739 web tests passing, mypy 0 issues, tier-model passes, only the pre-existing C.1 ruff finding above).

Open the 1A PR after C.1 is fixed. Do NOT bundle it with the 1B implementation work — that violates the separation of "delivered Phase 1A" from "in-progress Phase 1B."

## Critical context (don't forget)

- **CLAUDE.md is load-bearing.** Three-tier trust model, audit primacy, no legacy code, offensive programming, freeze_fields, 4-layer model.
- **Auto mode may be on or off in the next session.** Either way, `B` (operator decisions) above is already settled — don't re-litigate.
- **Memory pointers worth re-reading:**
  - `feedback_eval_attribution_can_mislead.md` — re-grep symbols and line numbers from the plan against current code; do not inherit framing.
  - `feedback_no_git_stash.md` — never use stash.
  - `feedback_locked_in_buggy_expectations.md` — when post-fix tests fail in a wave, the fix landed visibly; update tests, don't revert the fix.
  - `feedback_correctness_beats_performance.md` — 1B is a correctness/audit-integrity push; do not frame any of its work as a perf trade.
  - `feedback_no_calendar_shipping_commitments.md` — "pre-demo critical" is the operator's framing; do not turn that into ADR-level SLAs.
  - `project_rc5ux_demo_prep_scope.md` — the merge bar is "demo succeeds."
  - `project_adr010_dispatch_shape.md` — background on ADR-010 phase shape.

## Plan modification history (for the audit trail)

- 2578 lines (pre-review)
- 3323 lines (after Pass 3 fix-it agent — closed 5 must-fix blockers)
- 3337 lines (after A1-A5 fix-it agent — closed 5 mechanical residuals)
- Now committed at HEAD `88df6ceb`.

The line numbers cited in the plan have been refreshed twice in this session. Treat them as correct as-of `88df6ceb`. If the plan goes through more revision before implementation, **re-grep before trusting any line number** (the `feedback_eval_attribution_can_mislead` memory applies).

## Recommended next-session opening sequence

1. Run the verification commands in "Working-tree state" above.
2. Open the plan file and re-read the Schedule 1B Done When checklist (lines 3317-3337).
3. Decide: tackle the C.1 ruff cleanup first (5 min, makes the merge gate clean for both 1A and 1B PRs), or dive into 1B Task 1.
4. If starting 1B Task 1: open Task 1 in the plan, read the full step-by-step, then start implementing.
5. Use TodoWrite or filigree to track per-task progress within 1B (15 tasks is too many to hold in working memory).

## Useful agent IDs from prior sessions (likely DEAD after /clear, but recorded for the audit trail)

These agents handled the review/fix passes that produced the current plan state. After session clears they almost certainly won't resolve via `SendMessage`, but if continuity is somehow preserved they're worth trying first:

| Purpose | ID |
|---------|-----|
| Plan-review fix-it agent (Pass 3 must-fixes) | `a8659d83cc25580c6` |
| Plan-review fix-it agent (Pass 5 A1-A5 residuals) | `af964d042444eaa03` |
| Quality re-reviewer | `a588e962cf656ed69` |
| Architecture re-reviewer | `aaed55fe0c53d3d93` |

If `SendMessage` to any of these errors with "no such teammate," fall back to a fresh dispatch with explicit task description (the agent IDs are no substitute for self-contained instructions).
