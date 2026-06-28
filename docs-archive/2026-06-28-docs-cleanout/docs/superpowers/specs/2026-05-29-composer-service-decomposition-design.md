# Design — `web/composer/service.py` god-class decomposition

- **Date:** 2026-05-29
- **Branch:** `composer-service-decomp` (worktree off `RC5.2`)
- **Target land:** RC5.2 if it fits the window; otherwise 5.3 (operator-approved slip)
- **Status:** design — awaiting operator review before implementation plan

## Problem

`src/elspeth/web/composer/service.py` is the largest file in the repository
(5,313 lines). It hosts `ComposerServiceImpl`, a god class of 41 methods. The
dominant offender is **`_dispatch_tool_batch` — a single method of 1,298
lines**, larger than all but ~12 *files* in the codebase. Four more methods
exceed 200 lines (`_compose_loop` 294, `_finalize_no_tool_response` 279,
`_compute_availability` 216, `_dispatch_session_aware_tool` 213).

This is the audit-critical path: `_dispatch_tool_batch` is the per-tool-call
lifecycle that opens and closes every `ComposerToolInvocation` audit envelope.
Its size makes it nearly unreviewable and dangerous to change — every edit
risks the legal record.

## Goal & non-goals

**Goal.** Decompose `ComposerServiceImpl` by extracting cohesive method
clusters into sibling modules under `web/composer/`, mirroring the
just-merged `engine/orchestrator/` decomposition pattern (`core.py` stays a
thin driver; clusters become plain function-modules it invokes). Take
`service.py` from 5,313 toward ~1,500 lines. **Every change is
behaviour-preserving.**

**Non-goals.**

- **Not** re-opening the async-handler deferral. Two of the four per-tool
  special-cases in `_dispatch_tool_batch` are async (`request_advisor_hint`,
  `request_interpretation_review`) and are *deliberately* excluded from the
  `ToolDeclaration` registry — `tools/declarations.py` and `tools/_registry.py`
  both cite `elspeth-f5da936747` ("widening `ToolDeclaration.handler` to admit
  async is deferred"). Folding those into declaration hooks **is that ticket's
  scope** and stays out of bounds here.
- **Not** a semantic change to dispatch, audit, anti-anchor, budget, caching,
  or proposal-interception behaviour. The compose-loop contract is frozen.
- **Not** a redaction-MANIFEST change. `redaction.py` stays the
  auditor-readable single file for security policy (per `elspeth-6c9972ccbf`
  Refinement 1).

## Context: the ToolDeclaration substrate already exists

The epic `elspeth-6c9972ccbf` ("ToolDeclaration paradigm") is shown
`in_progress` in Filigree but is **substantially merged into RC5.2** (commits
`e9f35b65f` merge + `f896fd0ff` decommission of hand-maintained registries).
`tools/execute_tool` already owns *which* handler runs, derived from
per-plane `TOOLS_IN_MODULE` declaration tuples. The Filigree status is stale
(claim expired 2026-05-25); the code is the source of truth.

Therefore `_dispatch_tool_batch` is **not** the dispatch ladder — it is the
per-call lifecycle *around* the registry call. "Converge to ToolDeclaration"
in this work means finishing only the **sync** leftover (the
`get_plugin_schema` post-success hook; the `_TOOL_REQUIRED_PATHS` lookup is
already its own module `_required_paths_validator.py`). The async carve-outs
remain explicit branches.

## Anatomy of `_dispatch_tool_batch`

A single `for tool_call in assistant_message.tool_calls:` loop whose body is an
11-stage pipeline. Stages, in order:

1. Decode `tool_call.function.arguments` (JSON) — ARG_ERROR site 1.
2. Non-dict arguments guard — ARG_ERROR site 2.
3. Open audit envelope (`begin_dispatch_or_arg_error`) + canonical-JSON check
   — ARG_ERROR site 3.
4. Discovery cache check — early-return on hit (`is_cacheable_discovery_tool`).
5. Required-paths validation (`_TOOL_REQUIRED_PATHS` /
   `_find_missing_required_paths`) — ARG_ERROR site.
6. Approval gate (`trust_mode == "explicit_approve"` + `is_mutation_tool` +
   not `is_blob_store_only_mutation_tool`) → create proposal, intercept.
7. `request_advisor_hint` interception (async) — disabled / budget-exhausted /
   arg-validate / deadline / actual `_call_advisor_with_audit`.
8. Session-aware dispatch (`is_session_aware_tool` →
   `_dispatch_session_aware_tool`, async).
9. `preview_pipeline` runtime-preflight precompute.
10. Generic dispatch: `dispatch_with_audit(do_dispatch=run_sync_in_worker(
    execute_tool, ...))`, with the exception ladder:
    - `ToolArgumentError` → arg-error message, continue.
    - `(AssertionError, MemoryError, RecursionError, SystemError)` → re-raise
      (documented CLAUDE.md divergence — do not launder).
    - `AuditIntegrityError` → re-raise (Tier-1).
    - `Exception` → capture `ComposerPluginCrashError`, `break`.
11. SUCCESS post-process: rebind `state`/`last_validation`/
    `last_runtime_preflight`; `get_plugin_schema` schema-loaded mark; §7.7
    anti-anchor (mutation success vs discovery observation); discovery-cache
    store; `llm_messages` append; budget-class flag.

### The ~13 terminal arms (the discriminating risk surface)

Each arm has *distinct* audit / anti-anchor / budget side-effects. The
recurring idiom is:

```
recorder.record(finish_*)     # close the audit envelope
_append_tool_outcome(...)      # immutable record into the batch
anti_anchor.record_*(...)      # §7.7 tracker — SOME ARMS SKIP THIS
llm_messages.append({role:tool})
continue                       # (or break / raise)
```

Per-arm variations that are the bug surface: budget-exhaustion and
advisor-disabled arms **skip** `anti_anchor`; arms set `turn_has_discovery`
vs `turn_has_mutation`; cache-miss arms set `all_cache_hits=False`; three arms
`raise` and one `break` instead of `continue`.

## Approach (selected: C, A-first)

Mirror `engine/orchestrator/`: `ComposerServiceImpl` stays in `service.py` as
the thin driver; cohesive clusters move to sibling **function-modules** (not
new classes) invoked by the driver.

### Sequence — the load-bearing discipline

"Tests pass unchanged" is **necessary but not sufficient**: a verbatim move
that silently drops or reorders a `recorder.record(finish_*)` on an arm the
suite does not exercise stays green while corrupting the audit trail — the
exact catastrophic-but-invisible failure the Data Manifesto (Tier 1) forbids.
So:

1. **Characterization commit (gates everything).** Enumerate which terminal
   arms the existing composer suite hits; add characterization tests for the
   gaps so every arm's audit-envelope close, anti-anchor call (or deliberate
   skip), budget-class flag, and `llm_messages` shape is pinned **before any
   code moves**. This is the entire safety net (see Operational guardrails —
   per-commit mechanical gates are skipped on this branch).
2. **Verbatim extraction.** Lift the loop body into `tools/tool_batch.py` with
   all ~13 arms intact — no idiom collapse, no logic change. Tests green.
3. **Idiom collapse (separate commit).** Collapse the recurring emit idiom into
   uniform `emit_*` helpers, where each per-arm variation is individually
   reviewable and bisectable. Never combine with step 2.
4. **Sync convergence (optional, separate commit).** Fold the sync
   `get_plugin_schema` post-success hook toward the declaration. Async
   carve-outs untouched.
5. **Remaining oversized methods.** Extract `_compose_loop`,
   `_finalize_no_tool_response`, `_compute_availability`,
   `_dispatch_session_aware_tool`, `_persist_turn_audit` as discrete
   behaviour-preserving commits, each its own review checkpoint.

### Target module structure

```text
web/composer/
  service.py              # thin core: __init__, compose, _compose_loop driver, delegation
  tool_batch.py           # NEW — per-call pipeline (was _dispatch_tool_batch)
  tool_batch_arms.py      # NEW — terminal-arm emit helpers (created at step 3)
  turn_audit.py           # NEW — _persist_turn_audit cluster
  availability.py         # NEW — _compute_availability + ComposerAvailability
  no_tool_finalize.py     # NEW — _finalize_no_tool_response
```

`tool_batch.py` may live under `web/composer/tools/` instead of
`web/composer/` if import topology favours co-location with the registry it
calls; the implementation plan resolves this against the layer-import graph
(no upward L3 import is introduced either way — both are application layer).

### The per-call pipeline interface

The method threads ~20 loop-invariant inputs plus ~10 loop-carried
accumulators. The extraction makes this explicit rather than hiding it in
nested closures:

- **`ToolBatchContext` (frozen dataclass).** Loop-invariant inputs built once
  before the loop: `recorder`, `anti_anchor`, `discovery_cache`,
  `runtime_preflight_cache`, deadline, session/user/message ids, `actor`,
  `initial_version`, `progress`, `session_scope`, `turn_preferences`,
  `turn_sessions_service`, plus the handles read off `self` (model,
  availability, settings, catalog, data_dir, session_engine, secret_service,
  redaction telemetry, composer_skill_hash). Deep-frozen per the
  `freeze_fields` contract.
- **`BatchAccumulator` (mutable).** Loop-carried state that rebinds per
  iteration: `state`, `last_validation`, `last_runtime_preflight`,
  `advisor_calls_used`, `turn_has_mutation`, `turn_has_discovery`,
  `all_cache_hits`, `tool_outcomes`, `plugin_crash`, `plugin_crash_cause`,
  `proposals_this_turn`, `mutation_success_observed`,
  `decoded_args_by_call_id`. Honest about what mutates.
- **Return contract unchanged.** The collaborator still produces
  `(_DispatchOutcome, advisor_calls_used)`; `service.py`'s caller does not
  move. `_DispatchOutcome` / `_ToolOutcome` carriers stay as-is.

Methods the pipeline calls that remain on `self` and are passed in (or invoked
via a thin protocol) rather than moved in this phase: `_call_advisor_with_audit`,
`_dispatch_session_aware_tool`, `_cached_runtime_preflight`,
`_validate_advisor_arguments`, `_mark_plugin_schema_loaded`,
`_require_sessions_service`. The plan decides per-method whether each is
passed as a bound callable or relocated; default is *passed*, to keep the
first extraction minimal.

## Behaviour-preservation & testing strategy

- **Contract:** the existing composer test suite passes unchanged at *every*
  commit, AND the characterization tests from step 1 pin every terminal arm's
  audit/anti-anchor/budget side-effects.
- **Tier-1 discipline (the operator's condition for the `--no-verify`
  exemption):** zero tolerance for a dropped, added, or reordered
  `recorder.record(finish_*)`. Audit-envelope behaviour is identical
  pre/post each commit, verified by the characterization tests, not by a
  passing-but-blind suite.
- **Per-commit verification:** full composer suite + `mypy` + `ruff` +
  `tier_model` check run locally before each push.

## Operational guardrails

- **`--no-verify` is permitted on this branch only** (operator-granted
  2026-05-29, restated as *not* general) until the refactor lands, conditioned
  on the Tier-1 discipline above. A new automated CI system will absorb the
  skipped mechanical bureaucracy by land time. Every other branch keeps full
  hooks. (Memory: `feedback_composer_decomp_noverify_exemption`.)
- **Tier-model fingerprint cascade.** Each module extraction shifts AST
  `body[N]` indices and desyncs the `tier_model` allowlist. Because pre-commit
  hooks are skipped on this branch, fingerprints are **not** rotated per-commit
  (avoiding the dup-key data-loss footgun, memory
  `feedback_rotate_tier_model_dup_key_dataloss`); they are reconciled once at
  the end or by the new CI.
- **Worktree venv must be Python 3.13** to match main, or `enforce_tier_model`
  reports ~300 spurious violations (memory `project_tier_model_python_version`).
  The worktree gets its own `.venv` with `uv pip install -e .` (rebind editable
  install to the worktree's `src/`), avoiding the venv-leak footgun.
- **Plugin-hash gate (PH3)** targets plugin files; `service.py` and the new
  modules are not plugins, so PH3 should not trigger. Verify gate scope before
  the first code commit.

## Risks

| Risk | Mitigation |
|------|------------|
| Silent audit-trail corruption on an untested arm | Step-1 characterization tests pin all ~13 arms before any move; this gates the done-claim |
| Idiom-collapse buries a per-arm side-effect divergence | Collapse is a separate commit (step 3), never combined with the move |
| Scope creep into async declaration-folding | Explicitly deferred to `elspeth-f5da936747`; async branches preserved verbatim |
| Large frozen context object becomes a new god-object | It is data-only (no behaviour); behaviour stays in the pipeline functions |
| Branch slips and RC5.2 must land without it | Worktree isolates the work; RC5.2 unaffected; branch retargets to 5.3 cleanly |

## Open questions for the implementation plan

- Final home of `tool_batch.py` (`web/composer/` vs `web/composer/tools/`) —
  resolve against the layer-import graph.
- Which `self`-methods to pass as callables vs relocate (default: pass).
- Whether `turn_audit.py` / `availability.py` / `no_tool_finalize.py`
  extractions are independent enough to land before or after the idiom
  collapse (default: after, to keep the crown-jewel extraction first).
