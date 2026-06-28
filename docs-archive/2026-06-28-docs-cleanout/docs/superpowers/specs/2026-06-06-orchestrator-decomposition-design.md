# Orchestrator decomposition — design spec

- **Date:** 2026-06-06
- **Branch:** `refactor/orchestrator-decomp` (worktree off `release/0.5.3`)
- **Target:** `src/elspeth/engine/orchestrator/core.py` — the `Orchestrator` god-class
- **Deliverable of the originating session:** this spec + the implementation plan. No
  production code is modified in the planning session; execution is a later session.
- **Status:** approved (design), pending spec review.

## Problem

`engine/orchestrator/core.py` is 2,951 LOC. The `Orchestrator` class spans lines
317–2,951 (~2,634 LOC in a single class) with 35 methods, 13 of them ≥80 lines, and
73 module imports. It was the highest-churn file in the repository over the trailing six
months (165 commits) — though it now sits inside the **frozen** subsystem set (only the
web-orchestration system is under active churn), which is what makes this refactor
tractable now.

The class fuses five responsibilities that the instance-state map shows are separable:
run conducting, telemetry/lifecycle ceremony, checkpoint sequencing, source-iteration
driving, and resume. The cost is concrete: the file is hard to hold in context, every
change risks the audit legal-record path, and method-level test seams are brittle.

## Evidence — state-ownership map

Derived by AST analysis of every `Orchestrator` method (which `self._*` attributes it
touches and which sibling methods it calls). Instance state set in `__init__`:

```
_db  _events  _telemetry  _span_factory  _canonical_version
_checkpoint_manager  _checkpoint_config  _sequence_number  _current_graph
_clock  _rate_limit_registry  _concurrency_config  _coalesce_completed_keys_limit
```

Key findings that dictate the seams:

- **Telemetry cluster touches only `_telemetry` + `_events`.** `_emit_telemetry`,
  `_flush_telemetry`, `_safe_flush_telemetry`, `_emit_phase_error`,
  `_emit_interrupted_ceremony`, `_emit_failed_ceremony` form a closed group with no other
  internal state. Cleanest possible extraction.
- **`_sequence_number` is touched ONLY by the checkpoint cluster** —
  `_reset_checkpoint_sequence`, `_rebase_checkpoint_sequence`, `_maybe_checkpoint`,
  `_checkpoint_interrupted_progress`. A `CheckpointCoordinator` can own it outright.
- **`_current_graph` is a hidden temporal coupling.** Set during `_execute_run` /
  `_process_resumed_rows`, read by `_maybe_checkpoint` / `_checkpoint_interrupted_progress`.
  Any collaborator that checkpoints must receive the graph **explicitly** — the extraction
  turns an implicit ordering hazard into an explicit parameter (an improvement, not just a
  move).
- **Source-iteration depends on telemetry.** `_run_main_processing_loop` and its helpers
  call `_emit_telemetry` / `_emit_phase_error` / `_maybe_emit_progress` heavily, so the
  driver must be extracted *after* the ceremony unit.
- **Resume is not a clean leaf.** `_process_resumed_rows` shares `_initialize_run_context`,
  `_flush_and_write_sinks`, and `_make_checkpoint_after_sink_factory` with the normal path
  (`_execute_run`). See "Shared run-core decision".

## Target architecture (5 units, was 1)

| Unit (new file) | Methods | Owns / holds | ~LOC |
|---|---|---|---|
| `Orchestrator` (`core.py`, stays) — conductor | `run`, `_execute_run`, `_initialize_database_phase`, `_execute_export_phase`, `_register_graph_nodes_and_edges`, `_build_processor`, `_initialize_run_context`, `_flush_and_write_sinks`, `_write_pending_to_sinks` | `_db`, `_canonical_version`, `_span_factory`, run-context config | ~900 |
| `RunCeremony` (`ceremony.py`) | `emit_telemetry`, `flush_telemetry`, `safe_flush_telemetry`, `emit_phase_error`, `emit_interrupted_ceremony`, `emit_failed_ceremony` | `_telemetry`, `_events` | ~300 |
| `CheckpointCoordinator` (`checkpointing.py`) | `reset_sequence`, `rebase_sequence`, `maybe_checkpoint`, `make_checkpoint_after_sink_factory`, `checkpoint_interrupted_progress`, `delete_checkpoints` | **owns `_sequence_number`**; holds `_checkpoint_manager`, `_checkpoint_config`; graph passed in | ~400 |
| `SourceIterationDriver` (`source_iteration.py`) | `load_source_with_events`, `restore_source_iteration_context`, `finalize_source_iteration`, `record_field_resolution`, `handle_quarantine_row`, `maybe_emit_progress`, `run_main_processing_loop` | `_events`, `_span_factory`; depends on `RunCeremony` | ~500 |
| `ResumeCoordinator` (`resume.py`) | `resume`, `reconstruct_resume_state`, `process_resumed_rows` | depends on `RunCeremony` + `CheckpointCoordinator` + shared run-core | ~500 |

`Orchestrator` retains the public surface (`run`, `resume`) and composes the collaborators,
delegating to them. External callers and the public contract are unchanged.

### Shared run-core decision

`_initialize_run_context`, `_flush_and_write_sinks`, `_make_checkpoint_after_sink_factory`
are used by both `_execute_run` (normal) and `_process_resumed_rows` (resume). Resolved in
**Phase 4** with full information. Preferred: **(b) extract a `RunExecutionCore` that both
`Orchestrator.run` and `ResumeCoordinator` compose.** Fallbacks, in order: (c) inject the
shared callables into `ResumeCoordinator`; (a) `ResumeCoordinator` holds a back-ref to
`Orchestrator` — minimal change but reintroduces coupling, used only if (b) threatens
behavior preservation. (Approved: pursue (b).)

## Sequencing — staged extraction, one branch, gates reconciled once

The freeze removes collision pressure, so the work lands as a rapid series on the single
worktree branch rather than four separate PRs. Each phase remains independently verifiable
(full suite green) so any behavioral regression bisects to one extraction. This mirrors the
landed `composer/service.py` decomposition (merge `56b49fa05`).

- **Phase 0 — Characterization safety net.** Inventory existing orchestrator coverage
  (`tests/unit/engine/orchestrator/`, `tests/integration/pipeline/orchestrator/`,
  ADR-019 resume/counter-parity, graceful-shutdown, sink-diversion,
  export-partial-semantics). Fill gaps so these are pinned *before* any move:
  telemetry-emitted-after-Landscape ordering; checkpoint sequence monotonicity across
  run→checkpoint→resume→rebase; interrupted/failed ceremony emission; resume parity. Green.
- **Phase 1 — Extract `RunCeremony`.** Smallest blast radius, zero internal deps.
- **Phase 2 — Extract `CheckpointCoordinator`.** Owns `_sequence_number`; threads
  `_current_graph` explicitly through checkpoint calls (makes the temporal coupling explicit
  without changing *when* checkpoints fire).
- **Phase 3 — Extract `SourceIterationDriver`.** Depends on Phase 1.
- **Phase 4 — Extract `ResumeCoordinator`** + resolve the shared run-core decision (b).
  Coordinate with the in-flight F1 resume fix (see R5).
- **Phase 5 — Gate reconciliation & final verification.** tier-model
  (`@trust_boundary`/allowlist for the 4 new files; run under the Python 3.13 venv —
  worktree venv is symlinked to main's), `fingerprint_baseline.json` regen,
  monkeypatch-seam repoints in the three test files (below), contracts whitelist if any
  referenced path moved, `test_immutability_rules` count if class count changes, full
  `pytest tests/`.

**Verification discipline:** full `pytest tests/` after every phase; behavior-preserving
moves only — no logic edits ride along with a move.

## Behavior preservation

The guardrails already exist; the plan leans on them rather than inventing new ones:

- Dedicated `tests/unit/engine/orchestrator/` and `tests/integration/pipeline/orchestrator/`
  suites; ADR-019 resume/counter-parity is the net under the riskiest extraction.
- **Monkeypatch seams that must repoint when methods move** (mechanical, non-optional —
  the same cost the composer decomp paid with `execute_tool`):
  - `tests/conftest.py` patches `Orchestrator.run` (stays on `Orchestrator` — no repoint).
  - `tests/integration/test_adr_019_sweep_durability.py` patches
    `_initialize_database_phase` (stays), `_process_resumed_rows` (→ `ResumeCoordinator`),
    `_run_main_processing_loop` (→ `SourceIterationDriver`), `_flush_and_write_sinks`
    (stays / run-core).
  - `tests/unit/engine/test_orchestrator_registry_bootstrap.py` patches
    `_reconstruct_resume_state` (→ `ResumeCoordinator`).
- `tests/integration/pipeline/test_aggregation_checkpoint_bug.py` asserts
  `_maybe_checkpoint` passes aggregation state — preserve that call exactly through the
  `CheckpointCoordinator` move.

## Risks

- **R1 — Shared run-core.** Resolved Phase 4, `RunExecutionCore` preferred; (a) fallback.
- **R2 — `_current_graph` threading.** Must change only *how the graph reaches*
  checkpointing, never *when* checkpoints fire.
- **R3 — `safe_flush_telemetry`** reads `sys.exc_info()` in a `finally`; pending-exception
  preservation semantics must survive the move byte-for-byte.
- **R4 — Freeze duration.** The freeze is a process control, not a technical lock; keep the
  branch short-lived and reconcile gates at merge.
- **R5 — F1 resume-fix interaction.** The in-flight F1 resume re-emit fix
  (`fix/resume-fork-reemit`, design+plan done, not yet implemented) adds a
  `node_states.resume_checkpoint_id` provenance column and a 3-writer `token_data_ref`, and
  touches the exact resume path Phase 4 extracts. Phase 4 must land *after* F1, or the two
  must be explicitly sequenced/merged — extracting `ResumeCoordinator` against a moving
  resume target risks a hard rebase and double behavior-preservation review. Surface to the
  operator before starting Phase 4.

## Filigree tracking

Fresh epic — "Orchestrator decomposition (engine/orchestrator/core.py)" — with one task
per phase (0–5), mirroring how the composer decomposition tracked Phase 3/4. Filed after
this spec lands. The F1-resume dependency (R5) is recorded as a blocker on the Phase 4 task.

## Out of scope

- No behavioral change, no new features, no logic "improvements" riding along.
- No change to the public `Orchestrator` surface (`run`, `resume`) or its contract.
- The `engine/processor.py` and `web/sessions/service.py` god-classes (separate targets).
