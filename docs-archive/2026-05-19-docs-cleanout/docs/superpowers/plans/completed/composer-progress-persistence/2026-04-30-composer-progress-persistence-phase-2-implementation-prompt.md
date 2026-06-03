# Implementation Prompt — Composer Progress Persistence, Phase 2 (Redaction Framework)

> **Hand this prompt verbatim to a fresh implementing agent.** It is self-contained: it does not assume any memory of prior conversations or any visible context beyond the project's `CLAUDE.md` and `MEMORY.md`. The plan and review chain it references are committed to the repository.

---

## What you are doing

You are implementing Phase 2 of the ELSPETH composer progress-persistence epic: a manifest-keyed redaction primitive that ensures `Sensitive[T]`-annotated tool-call argument and response fields are replaced with summarizer output or fixed sentinels before any persistence write. Phase 1 (schema + tables) and Phase 1B/1C (persistence + Postgres portability) have landed. Phase 2 is the redaction layer alone — no compose-loop wiring (Phase 3), no frontend (Phase 4).

The work touches `src/elspeth/web/composer/redaction.py` (extension), a new `redaction_telemetry.py`, `tools.py` (handler promotion for ~6–8 tools), and ~13 new test files. CI gets a new GitHub Actions workflow (`composer-redaction-gate.yml`) enforcing a direction-aware PR label gate.

## Where the work lives

- **Worktree:** `/home/john/elspeth/.worktrees/composer-progress-1a`
- **Branch:** `feat/composer-progress-persistence-1a`
- **Plan file:** `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.md`
- **Spec (rev 5):** `docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md`

The plan has been through **five review passes**. Each pass surfaced findings folded into the plan body. The institutional memory chain is non-negotiable reading before you touch any code — see the next section.

## REQUIRED reading sequence before any code change

Read these IN ORDER. Do not skip. Each item is load-bearing for the next.

1. **`/home/john/CLAUDE.md`** (user-level) and **`/home/john/elspeth/CLAUDE.md`** (project-level). The project's auditability standard, three-tier trust model, "no legacy code" policy, "no defensive programming," freeze-guard contract, and layer-import rules govern every line of code you will write. Skipping this section is the #1 cause of in-session rework.

2. **`/home/john/.claude/projects/-home-john-elspeth/memory/MEMORY.md`** plus the linked feedback files. Particularly relevant for this work:
   - `feedback_default_to_worktree.md` — you are already in the right worktree; do not switch.
   - `feedback_skip_worktree_for_skill_and_config_edits.md` — not applicable here (this IS code work).
   - `feedback_uv_venv_leak.md` — `uv pip install` from this worktree without `--python .venv/bin/python` clobbers `main`'s venv. Always activate or pass `--python`.
   - `project_tier_model_python_version.md` — the worktree's venv MUST be Python 3.13 or `enforce_tier_model.py` produces ~300 spurious findings.
   - `feedback_fix_errors_you_encounter.md` — if you hit a defect en route, fix it in the same session.
   - `feedback_default_is_fix_not_ticket.md` — don't defer fixes to tickets unless explicitly asked.
   - `feedback_locked_in_buggy_expectations.md` — when a structural fix produces a wave of test failures, those tests pinned the bug. Update them, don't revert the fix.
   - `feedback_correctness_beats_performance.md` — never frame correctness work as a perf tradeoff.
   - `feedback_no_slog_recommendations.md` — audit/telemetry primacy; `slog` only for audit-system or telemetry-system failures.
   - `feedback_doc_only_commits_no_ci.md` — markdown-only commits use `--no-verify`. This work is NOT doc-only; CI hooks run on commit.
   - `project_db_migration_policy.md` — no Alembic; if the DB schema needs to change (it shouldn't in Phase 2), operator deletes the old DB.

3. **The plan body — read top to bottom in this order:**
   - The header (review history banner + rev-history of fix sets).
   - **Appendices A, B, C, D, E IN ORDER** — these are the institutional memory of why specific tasks are shaped the way they are. Appendix C marks a rev-3 closure as SUPERSEDED → Appendix D; Appendix E is the rev-5 closure surface. **You cannot safely edit Task 19 (the property test) without reading Appendices C, D, and E** — the test has been rewritten twice in response to convergent reviewer findings, and the third write needs the six-item cross-check the task body specifies.
   - The Preflight section (gate-state required before Task 1).
   - The File Structure section.
   - Tasks 1–20 in order. **The TDD discipline applies to EVERY task** — Tasks 1–4 spell out the six-step protocol; Tasks 5+ use compressed notation but the six-step sequence is mandatory regardless (the Preflight section says so explicitly).

4. **The review JSONs** at:
   - `docs/superpowers/plans/completed/composer-progress-persistence/2026-04-30-composer-progress-persistence-phase-2-redaction.review.json` (rev-1)
   - `…-redaction.review-rev2.json`
   - `…-redaction.review-rev3.json`
   - `…-redaction.review-rev4.json`
   - `…-redaction.review-rev5.json`
   You do not need to read every word; skim each `summary` and `blocking_issues`/`recommendations` to understand the structural arc of the closure work. Read in full any JSON whose verdict relates to a task you are about to execute.

## REQUIRED sub-skill

Per the plan body's "For agentic workers" banner: use **`superpowers:subagent-driven-development`** (recommended) or **`superpowers:executing-plans`**. The plan is structured for both — tasks are independent enough that subagent-driven parallelisation works well; tasks are also sequential enough that single-track execution works.

The plan uses checkbox (`- [ ]`) syntax for step tracking. Mark steps complete as you finish them.

## Preflight gate (run BEFORE Task 1)

From the worktree root:

```bash
cd /home/john/elspeth/.worktrees/composer-progress-1a
.venv/bin/python --version  # Confirm Python 3.13
.venv/bin/python -m pytest tests/unit -q  # ~14,948 tests; verify zero failures, not exact count
.venv/bin/python -m pytest tests/integration -q -m "not testcontainer"  # ~785 tests; same
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py
```

If any of the above is red, **stop and surface to operator**. Do not begin Task 1.

If the venv has not been rebound since the worktree was forked, run first:

```bash
uv pip install --python .venv/bin/python -e ".[dev]"
```

(The `--python` flag is mandatory; without it, `uv pip install` finds main's venv and clobbers it — see `feedback_uv_venv_leak`.)

## Critical disciplines (cite-only summaries; full text is in the linked files)

- **Three-tier trust model.** Tool-call arguments are Tier-3 (zero trust) at the LLM boundary; promoted handlers must catch `pydantic.ValidationError` and re-raise as `ToolArgumentError` so the compose loop routes to ARG_ERROR (not crash). The audit table is Tier-1 (full trust) — `arguments_canonical` stays raw, redacted views go only to `chat_messages.tool_calls`. See `CLAUDE.md` "Data Manifesto."

- **Layer discipline.** Composer code is L3. Imports flow downward only (L3 may import L0/L1/L2; never the reverse). The tier-model CI script enforces this. No upward imports; no apologetic lazy imports.

- **Audit primacy.** Audit fires first (sync, crash-on-failure), then telemetry (async, best-effort), then logging (last resort). The new `RedactionTelemetry` is operational telemetry, not logging. Do not introduce any `slog.info` / `logger.warning` calls inside the redaction layer.

- **No legacy / no compatibility shims.** When `_TOOL_REQUIRED_PATHS[tool]` is removed for a promoted tool, it is removed in the SAME COMMIT as the promotion. No conditional removal, no "if redundant" hedging. The `MissingRequiredPaths` test-pin grep-and-update step in Task 4 is the canonical procedure for handling the audit `error_class` schema change.

- **Frozen-dataclass freeze-guards.** Any `@dataclass(frozen=True)` with container fields (`dict`, `list`, `set`, `Mapping`, `Sequence`) MUST use `freeze_fields()` in `__post_init__`. The CI script `scripts/cicd/enforce_freeze_guards.py` will fail the build otherwise. The plan's `TraversalNode` dataclass is documented as an intentional freeze-elision; do not generalise that pattern.

- **Test-path integrity.** Integration tests MUST use `ExecutionGraph.from_plugin_instances()` / `instantiate_plugins_from_config()`. Do not bypass production code paths in tests. The plan's tracer-bullet test (Task 6) is the integration-shape exemplar.

- **TDD discipline.** Every task's first step is "write the failing test." Every task's second step is "run, verify it fails." For tasks compressed to "Standard TDD task; test cases: [list]," expand to the six-step sequence anyway.

## Property-test discipline (Task 19, load-bearing security claim)

**Before touching Task 19, re-read Appendices C, D, and E of the plan.** The Hypothesis property test has been rewritten twice in response to convergent reviewer findings about silent-pass modes. The current shape (rev-4 type-agnostic path-aware `!=` invariant) is the closure mechanism for the entire rev-2/3/4 BLOCKER chain. The task body contains a **six-item implementer cross-check** that MUST be inspected before commit:

1. No `isinstance(...)` filter before any `assert` statement.
2. No `if node.value_provider is None: continue` for a Sensitive-marked node.
3. No `except (TypeError, ValueError):` or any broad `except` wrapping value extraction.
4. The only `continue` is the explicit `raw_value is None` skip with a comment.
5. The `assert sensitive_nodes` guard runs and fails loudly on a misclassified type-driven entry.
6. The key-set equality assertion runs before the per-key value comparison for container descent.

A code-quality reviewer who signs off without re-checking each item is a process gap. **Cite each item in the commit message body for the property-test commit.**

Also: run `pytest --collect-only -q` on the property test file BEFORE the commit that lands it. A `hypothesis.errors.ResolutionFailed` on any parametrize entry is a blocking issue that must be resolved with a custom strategy (registered via `hypothesis.strategies.register_type_strategy(Model, custom)` in a `conftest.py` adjacent to the test file).

## Commit cadence

Each plan task commits once on success. Commit messages use Conventional Commits (`feat(composer/redaction): …`, `test(composer/redaction): …`, `chore(composer/redaction): …`, `ci(composer/redaction): …`). Commit bodies cite the plan task, the spec section, and the rev-N closure reference.

Pre-commit hooks run (ruff, mypy slice, tier-model, freeze-guards, sometimes the relevant unit tests). If a hook fails: **do NOT use `--no-verify`**. Investigate. Fix the underlying issue. Re-stage. Create a NEW commit (never amend, per `CLAUDE.md` git safety).

The `--no-verify` exemption applies ONLY to markdown-only commits (per `feedback_doc_only_commits_no_ci`). This work is not doc-only.

## When to stop and surface to operator

Per the plan's Task 20:

- Adequacy guard runtime > 5 seconds for 38 entries (sanity bound).
- Property test collection errors that need a custom Hypothesis strategy.
- Any rev-history closure mechanism appears to regress.
- A grep for `_TOOL_REQUIRED_PATHS` shows leftover entries for promoted tools after the wave completes.
- `MissingRequiredPaths` grep shows test pins for promoted tools you cannot trace to an obvious update path.
- Any test failure that looks like it might be the property test silently passing (e.g. `event()` counter for empty-container is approaching 100%).

For all five cases: **stop, surface the finding to the operator with citations to the plan task and the relevant Appendix entry, and wait for direction.** Do not invent a workaround.

## Phase 3 / Phase 4 boundaries

Phase 2 does NOT touch:

- `src/elspeth/web/composer/service.py` compose-loop body — Phase 3.
- `src/elspeth/web/frontend/` — Phase 4.
- `src/elspeth/web/sessions/` — Phase 1 owns the schema.
- `src/elspeth/contracts/composer_audit.py` `arguments_canonical` — stays raw (spec §4.2.8); Phase 3 MUST NOT redact this surface.

If a Phase 2 task seems to require editing one of the above, you have misread the task. Re-read the relevant Appendix entry; if still ambiguous, stop and surface.

## Final gate

Task 19 is the gate — the full project gate plus the new tests. Task 20 surfaces to the operator with the PR-open decision. Do NOT run `gh pr create` autonomously; the plan explicitly forbids it (rev-1 B4 closure).

## Pointers if you get stuck

- For trust-model questions: invoke the `tier-model-deep-dive` skill.
- For engine internals (canonical JSON, schema contracts, layer architecture, dependency graphs): invoke the `engine-patterns-reference` skill.
- For logging vs telemetry vs audit choices: invoke the `logging-telemetry-policy` skill.
- For config-contract questions: invoke the `config-contracts-guide` skill.
- For test-strategy disputes: invoke the `superpowers:test-driven-development` skill.

If the plan and a memory entry conflict, the memory wins (memory is the operator's current preference). If a memory entry and `CLAUDE.md` conflict, surface the conflict to the operator before acting.

---

**This prompt was committed alongside the rev-5 closure of the Phase 2 plan at HEAD `fb4614e5`. If the plan has had a rev-6 or later review pass since this prompt was written, read the new Appendix entry before proceeding.**
