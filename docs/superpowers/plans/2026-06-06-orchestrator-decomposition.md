# Orchestrator Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the 2,634-LOC `Orchestrator` god-class in `src/elspeth/engine/orchestrator/core.py` into a thin conductor plus four state-disjoint collaborators, with zero behavioral change.

**Architecture:** Extract along the AST state-ownership seams — `RunCeremony` (telemetry/`_events`), `CheckpointCoordinator` (owns `_sequence_number`), `SourceIterationDriver` (the row loop), `ResumeCoordinator` + a shared `RunExecutionCore`. `Orchestrator` retains its public surface (`run`, `resume`) and composes the collaborators. Each phase is behavior-preserving and individually verified by the **existing** test suite (the refactor oracle); a regression bisects to one phase.

**Tech Stack:** Python 3.13, pytest. No new runtime dependencies. Engine = audit legal-record path — correctness is absolute.

---

## Conventions (read before any task)

**Worktree:** `/home/john/elspeth/.worktrees/orchestrator-decomp`, branch `refactor/orchestrator-decomp`, off `release/0.5.3`.

**CRITICAL — import-resolution guard.** The `.venv` here is symlinked to main's venv (for hooks), whose editable `elspeth` points at **main's** `src/`, not this worktree. Before running any test, confirm the worktree's source is the imported package, or you will test the wrong tree:

```bash
cd /home/john/elspeth/.worktrees/orchestrator-decomp
# Create a DEDICATED venv for execution (do NOT uv-pip-install into the symlinked main venv — it leaks/clobbers main):
uv venv .venv-wt --python 3.13
uv pip install -e . --python .venv-wt/bin/python
.venv-wt/bin/python -c "import elspeth, pathlib; assert '.worktrees/orchestrator-decomp' in elspeth.__file__, elspeth.__file__; print('OK', elspeth.__file__)"
```
Use `.venv-wt/bin/python -m pytest …` for all test runs below. (Python 3.13 is also required for the tier-model gate per project convention.)

**MANDATORY env prefix for ALL test commands:** `env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 .venv-wt/bin/python -m pytest …`. The interactive shell has `VIRTUAL_ENV` pointing at main's `.venv`; the suite runs under `pytest-xdist` (parallel workers) which inherit that stale pointer and load a second numpy C-extension → `ImportError: cannot load module more than once per process`. Clearing `VIRTUAL_ENV` fixes it. (The commands below omit the prefix for brevity — always add it.)

**Verification ladder (run in this order each phase):**
- Focused: `.venv-wt/bin/python -m pytest tests/unit/engine/orchestrator tests/integration/pipeline/orchestrator -q`
- Resume-critical (phases touching resume/checkpoint): `.venv-wt/bin/python -m pytest tests/integration/test_adr_019_resume_counter_parity.py tests/integration/test_adr_019_sweep_durability.py tests/integration/test_adr_019_cross_table_invariants.py tests/unit/engine/orchestrator/test_resume_failure.py -q`
- Full (CI-equivalent, before every commit): `.venv-wt/bin/python -m pytest tests/` — plain selection; do **not** pass `-o addopts=""` (that force-runs deselected slow/stress/testcontainer suites).

**Refactor discipline:** behavior-preserving moves ONLY. No logic edits, no "while I'm here" cleanups, no signature changes to public methods. When a step says "move lines X–Y verbatim," copy the body exactly and apply only the named state-rename transform. If the focused suite goes red after a move, the move was not behavior-preserving — revert and re-do, do not "fix forward."

**Per-task reviewer gate (mandatory — green is necessary but NOT sufficient).** A passing suite only proves the *covered* paths survived; an under-tested path can change behavior and stay green. So between every task the reviewer MUST, beyond confirming the suite matches baseline:
1. **Verbatim-diff check.** `git diff` the moved body against its source line range and confirm it is byte-identical modulo the named renames. A fresh subagent moving a body is the actor most likely to silently "tidy" it — the diff, not the green check, protects the uncovered lines (see Phase 0 coverage map).
2. **Negative-control check.** Where a task repoints a monkeypatch seam that *injects a failure* (e.g. `sweep_durability:539`'s corrupting loop, `:444`'s `_fail_if_processed`), confirm the patch STILL FIRES its expected failure after repointing — not merely that the test is green. A repoint to a reference the orchestrator captured at `__init__` (rather than dispatching through the attribute) silently no-ops the patch: test green, durability never exercised. Verify the seam still bites.

**Commits:** full hooks on (this is engine code, not docs). One commit per task. End messages with the `Co-Authored-By` trailer.

**Gate reconciliation** is deferred to Phase 5 (single reconciliation under the freeze), per `notes/tier-model-bulk-remediation-playbook.md` and the landed `composer/service.py` precedent (merge `56b49fa05`).

---

## Phase 0 — Characterization safety net

**Goal:** Prove the existing suite already pins the invariants the extraction can break, and fill only genuine gaps. No production code changes.

### Task 0.1: Establish green baseline + setup execution venv

**Files:** none (environment + verification only).

- [ ] **Step 1: Create the dedicated execution venv** (see Conventions). Run the import-resolution guard assertion; it must print the worktree path.

- [ ] **Step 2: Run the full suite to capture the green baseline**

Run: `.venv-wt/bin/python -m pytest tests/ -q 2>&1 | tail -20`
Expected: all pass except any documented pre-existing flakes. Record the pass/fail/deselected counts in the task's commit message or a scratch note — this is the number every later phase must reproduce.

- [ ] **Step 3: Measure the coverage risk-map of the file being dismembered.**

Run: `.venv-wt/bin/python -m pytest tests/unit/engine/orchestrator tests/integration/pipeline/orchestrator tests/integration/test_adr_019_resume_counter_parity.py tests/integration/test_adr_019_sweep_durability.py --cov=elspeth.engine.orchestrator.core --cov-report=term-missing -q`

Record the **term-missing line numbers**. These are the lines the test net does NOT catch — the
real risk map. For every uncovered line that falls inside a method scheduled to move (cross-reference
the line ranges in Phases 1–4), the verbatim-move discipline is the *only* safety net. Flag any
uncovered line in `maybe_checkpoint`/`_run_main_processing_loop`/`resume`/`_process_resumed_rows`
(the highest-stakes moves) and add a targeted characterization test in Task 0.2 Step 3 before that
method is moved. "Green suite" blesses only covered paths; this step tells you where green is silent.

> **ENV LIMITATION (2026-06-06):** coverage cannot run in this worktree venv — numpy 2.4.6's
> C-extension single-init guard raises `ImportError: cannot load module more than once per process`
> under coverage's import machinery (reproduced with pytest-cov, `coverage run`, xdist on AND off,
> `COVERAGE_CORE=sysmon`). The no-cov gate is unaffected (359 focused tests pass clean). Coverage map
> is therefore SKIPPED; per the advisor it is non-blocking. Compensating control: the two highest-stakes
> uncovered-behavior risks were hand-identified instead — (a) `make_checkpoint_after_sink_factory`
> late graph-binding (confirmed at core.py:618, mitigated by the `set_active_graph` relocation in
> Phase 2) and (b) the injected-failure monkeypatch seams (negative-control check in the reviewer gate).
> The **verbatim-diff reviewer gate is the primary net for uncovered lines.**

- [ ] **Step 4: Commit the baseline + coverage note** (if recorded; otherwise skip)

### Task 0.2: Map invariants → guardian tests; fill gaps

**Files:**
- Test (gap fill, only if needed): `tests/unit/engine/orchestrator/test_decomposition_characterization.py`

- [ ] **Step 1: Confirm each invariant has a guardian.** Read these and verify the named behavior is asserted. Do NOT add a test where one already exists.

| Invariant the extraction risks | Guardian test (verify it covers it) |
|---|---|
| Telemetry emitted AFTER Landscape recording | `tests/integration/telemetry/test_wiring.py` |
| PhaseError emission never masks original exception (R3) | `tests/unit/engine/orchestrator/test_phase_error_masking.py` |
| INTERRUPTED ceremony on graceful shutdown | `tests/unit/engine/orchestrator/test_graceful_shutdown.py`, `tests/integration/pipeline/orchestrator/test_graceful_shutdown.py` |
| FAILED ceremony + finalize on run failure | `tests/unit/engine/orchestrator/test_run_status.py`, `test_resume_failure.py` |
| Checkpoint sequence monotonicity + aggregation state | `tests/integration/pipeline/orchestrator/test_orchestrator_checkpointing.py`, `tests/integration/pipeline/test_aggregation_checkpoint_bug.py` |
| Resume counter parity / cross-table invariants | `tests/integration/test_adr_019_resume_counter_parity.py`, `test_adr_019_cross_table_invariants.py` |
| Source-iteration finalize / field resolution | `tests/unit/engine/orchestrator/test_finalize_source_iteration.py` |
| Sink diversion counters / pending grouping | `tests/integration/pipeline/orchestrator/test_sink_diversion_counters.py`, `tests/unit/engine/orchestrator/test_pending_sink_grouping.py` |

- [ ] **Step 2: Identify gaps.** The likely gap is an explicit assertion of the **telemetry-after-Landscape ordering at the `run()` boundary** and **`_sequence_number` continuity across run→resume→rebase** as a single behavioral statement. If `test_wiring.py` and the ADR-019 suite already assert these end-to-end, declare no gap and skip Step 3.

- [ ] **Step 3 (only if a gap exists): Write the characterization test.** Read the recorder/telemetry doubles in `tests/integration/conftest.py` and `tests/integration/_helpers.py` first; reuse their recording fixtures. Assert the ordering/continuity with the existing double rather than a new mock. Run it green against the current (un-refactored) code — a characterization test must pass BEFORE the refactor.

Run: `.venv-wt/bin/python -m pytest tests/unit/engine/orchestrator/test_decomposition_characterization.py -v`
Expected: PASS (pins current behavior).

- [ ] **Step 4: Commit**

```bash
git add tests/unit/engine/orchestrator/test_decomposition_characterization.py
git commit -m "test(orchestrator): characterization net for decomposition

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 1 — Extract `RunCeremony`

**Goal:** Move the 6 telemetry/ceremony methods (lines 372–536, ~154 LOC) to a collaborator touching only `_telemetry` + `_events`. None of these methods are monkeypatched by tests → zero seam repoints. Cleanest extraction.

### Task 1.1: Create `RunCeremony`

**Files:**
- Create: `src/elspeth/engine/orchestrator/ceremony.py`
- Modify: `src/elspeth/engine/orchestrator/core.py` (delete moved methods, add delegation field, update call sites)

- [ ] **Step 1: Create the collaborator with the exact interface.**

```python
# src/elspeth/engine/orchestrator/ceremony.py
"""Run-lifecycle ceremony: telemetry emission and EventBus lifecycle events.

Extracted from Orchestrator (decomposition Phase 1). Behavior-preserving:
method bodies are moved verbatim from core.py with self._events/self._telemetry
references unchanged (this class now owns those references).
"""
from __future__ import annotations
# ... (move the telemetry/event-related imports that these 6 methods use from core.py)

class RunCeremony:
    def __init__(
        self,
        *,
        events: EventBusProtocol,
        telemetry: TelemetryManagerProtocol | None,
    ) -> None:
        self._events = events
        self._telemetry = telemetry

    def emit_telemetry(self, event: TelemetryEvent) -> None: ...          # body from core.py:372-382
    def flush_telemetry(self) -> None: ...                                # body from core.py:384-390
    def emit_phase_error(self, phase, error, target=None) -> None: ...    # body from core.py:392-411
    def safe_flush_telemetry(self) -> None: ...                           # body from core.py:413-434; self._flush_telemetry() -> self.flush_telemetry()
    def emit_interrupted_ceremony(self, run_id, factory, shutdown_exc, start_time) -> None: ...  # body from core.py:436-476; self._emit_telemetry -> self.emit_telemetry
    def emit_failed_ceremony(self, run_id, factory, start_time, result=None) -> None: ...        # body from core.py:478-536; self._emit_telemetry -> self.emit_telemetry
```

Move each body **verbatim** from the cited line range. The only edits: drop the leading underscore on the public methods, and within bodies rewrite intra-cluster calls `self._emit_telemetry` → `self.emit_telemetry`, `self._flush_telemetry` → `self.flush_telemetry`. Copy the exact type annotations and imports those bodies reference.

- [ ] **Step 2: Wire `Orchestrator` to own and delegate.** In `core.py` `__init__` (after line 362) add:

```python
        self._ceremony = RunCeremony(events=self._events, telemetry=self._telemetry)
```

Delete the 6 methods (lines 372–536). Replace every internal call site within `core.py` mechanically:
- `self._emit_telemetry(` → `self._ceremony.emit_telemetry(`
- `self._flush_telemetry(` → `self._ceremony.flush_telemetry(`
- `self._emit_phase_error(` → `self._ceremony.emit_phase_error(`
- `self._safe_flush_telemetry(` → `self._ceremony.safe_flush_telemetry(`
- `self._emit_interrupted_ceremony(` → `self._ceremony.emit_interrupted_ceremony(`
- `self._emit_failed_ceremony(` → `self._ceremony.emit_failed_ceremony(`

Add `from elspeth.engine.orchestrator.ceremony import RunCeremony` to `core.py` imports.

- [ ] **Step 3: Run focused + full suite (the oracle).**

Run: `.venv-wt/bin/python -m pytest tests/unit/engine/orchestrator tests/integration/pipeline/orchestrator -q`
Expected: identical pass set to Phase 0 baseline.
Then: `.venv-wt/bin/python -m pytest tests/`
Expected: matches Phase 0 baseline counts. If red → the move was not verbatim; revert and redo.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/engine/orchestrator/ceremony.py src/elspeth/engine/orchestrator/core.py
git commit -m "refactor(orchestrator): extract RunCeremony (telemetry + lifecycle events)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Extract `CheckpointCoordinator`

**Goal:** Move the checkpoint cluster (lines 364–370, 538–718) into a collaborator that **owns `_sequence_number`** and receives `_current_graph` **explicitly** (eliminating the hidden temporal coupling). No direct monkeypatch seams on these methods; `test_aggregation_checkpoint_bug.py` asserts aggregation state flows through `maybe_checkpoint` — preserve that argument exactly.

### Task 2.1: Create `CheckpointCoordinator`

**Files:**
- Create: `src/elspeth/engine/orchestrator/checkpointing.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`

- [ ] **Step 1: Create the collaborator.**

```python
# src/elspeth/engine/orchestrator/checkpointing.py
"""Checkpoint sequencing and persistence. Owns the monotonic sequence number;
receives the execution graph explicitly (was the implicit self._current_graph).
Extracted from Orchestrator (decomposition Phase 2), behavior-preserving."""
from __future__ import annotations
# ... (move checkpoint-related imports from core.py)

class CheckpointCoordinator:
    def __init__(
        self,
        *,
        checkpoint_manager: CheckpointManager | None,
        checkpoint_config: RuntimeCheckpointConfig | None,
    ) -> None:
        self._checkpoint_manager = checkpoint_manager
        self._checkpoint_config = checkpoint_config
        self._sequence_number = 0

    def reset_sequence(self) -> None:        # body from core.py:364-366
        self._sequence_number = 0
    def rebase_sequence(self, sequence_number: int) -> None:   # body from core.py:368-370
        self._sequence_number = sequence_number
    def set_active_graph(self, graph: ExecutionGraph) -> None:  # NEW — relocates self._current_graph
        self._active_graph = graph
    def maybe_checkpoint(self, ...) -> None: ...
        # body from core.py:538-600; replace self._current_graph -> self._active_graph (read at fire-time)
    def make_checkpoint_after_sink_factory(self, run_id, processor) -> _CheckpointFactory: ...
        # body from core.py:602-628 VERBATIM; self._maybe_checkpoint(...) -> self.maybe_checkpoint(...)
    def checkpoint_interrupted_progress(self, ...) -> None: ...
        # body from core.py:630-709; replace self._current_graph -> self._active_graph
    def delete_checkpoints(self, run_id: str) -> None: ...     # body from core.py:711-718
```

**CRITICAL — preserve LATE binding (advisor R-graph).** The original `make_checkpoint_after_sink_factory`
(core.py:602-628) does NOT pass `graph`; its inner `callback` reads `self._current_graph` at
*callback-fire time* via `_maybe_checkpoint` (core.py:618 → reads `_current_graph`). Capturing
`graph=` at factory-creation time would be **early binding** — a silent behavior change in the
resume path, invisible to any test that checkpoints against a stable graph. So the transform is a
pure **attribute relocation**, not a signature change: `CheckpointCoordinator` owns mutable
`self._active_graph`, set via `set_active_graph()` at exactly the points the original did
`self._current_graph = graph`, and read at fire-time inside `maybe_checkpoint` /
`checkpoint_interrupted_progress`. This preserves the late-binding semantics byte-for-byte. (Turning
the coupling into an explicit per-call parameter is a *behavior-changing* improvement — explicitly
OUT OF SCOPE for this behavior-preserving decomposition.) Preserve every other parameter of these
methods exactly as in the originals (they carry `run_id`/`token_id`/`node_id`/aggregation/coalesce
state — read the originals, do not drop or reorder). `self._sequence_number` stays (now owned here).

- [ ] **Step 2: Wire `Orchestrator`.** In `__init__`:

```python
        self._checkpoints = CheckpointCoordinator(
            checkpoint_manager=checkpoint_manager,
            checkpoint_config=checkpoint_config,
        )
```

Delete `_reset_checkpoint_sequence`, `_rebase_checkpoint_sequence`, `_maybe_checkpoint`, `_make_checkpoint_after_sink_factory`, `_checkpoint_interrupted_progress`, `_delete_checkpoints` from `core.py`. The `self._checkpoint_manager`/`self._checkpoint_config`/`self._sequence_number` instance attributes move out — delete their assignments from `__init__` (lines 354–355, 360).

Update call sites in `run` (1132–1379), `_execute_run` (2308–2432), `_flush_and_write_sinks` (1662–1723), `resume` (2548–2836), `_process_resumed_rows` (2838–2951):
- `self._reset_checkpoint_sequence()` → `self._checkpoints.reset_sequence()`
- `self._rebase_checkpoint_sequence(n)` → `self._checkpoints.rebase_sequence(n)`
- `self._delete_checkpoints(run_id)` → `self._checkpoints.delete_checkpoints(run_id)`
- `self._make_checkpoint_after_sink_factory(...)` → `self._checkpoints.make_checkpoint_after_sink_factory(...)` (no signature change)
- `self._checkpoint_interrupted_progress(...)` → `self._checkpoints.checkpoint_interrupted_progress(...)` (no signature change)

In `_execute_run` (2308–2432) and `_process_resumed_rows` (2838–2951), every assignment
`self._current_graph = <graph>` becomes `self._checkpoints.set_active_graph(<graph>)` at the **same
line/point** (preserving when the graph becomes visible to checkpointing). Then grep `_current_graph`
across `core.py` to confirm the checkpoint methods were its only readers, and delete the
`self._current_graph` attribute (line 361) and its assignments. **Do not** introduce a `graph=`
parameter — see the late-binding note in Step 1.

- [ ] **Step 3: Verify (focused + resume-critical + full).**

Run: `.venv-wt/bin/python -m pytest tests/integration/pipeline/orchestrator/test_orchestrator_checkpointing.py tests/integration/pipeline/test_aggregation_checkpoint_bug.py tests/integration/test_adr_019_resume_counter_parity.py tests/integration/test_adr_019_sweep_durability.py -q`
Expected: PASS.
Then focused + full as in Phase 1 Step 3. Must match baseline.

- [ ] **Step 4: Commit** (`refactor(orchestrator): extract CheckpointCoordinator; make graph coupling explicit`)

---

## Phase 3 — Extract `SourceIterationDriver`

**Goal:** Move the row-loop cluster (lines 1725–2306) into a driver that depends on `RunCeremony`. `_run_main_processing_loop` IS a monkeypatch seam (`test_adr_019_sweep_durability.py:468,539`) → repoint required.

### Task 3.1: Create `SourceIterationDriver`

**Files:**
- Create: `src/elspeth/engine/orchestrator/source_iteration.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`
- Modify (seam repoint): `tests/integration/test_adr_019_sweep_durability.py`

- [ ] **Step 1: Create the collaborator.**

```python
# src/elspeth/engine/orchestrator/source_iteration.py
"""Source loading and the main row-processing loop. Depends on RunCeremony for
telemetry/event emission. Extracted from Orchestrator (decomposition Phase 3)."""
from __future__ import annotations

class SourceIterationDriver:
    def __init__(self, *, events: EventBusProtocol, span_factory: SpanFactory, ceremony: RunCeremony) -> None:
        self._events = events
        self._span_factory = span_factory
        self._ceremony = ceremony

    def load_source_with_events(self, ...) -> ...: ...          # body from core.py:2081-2117
    def restore_source_iteration_context(self, ...) -> ...: ... # body from core.py:1918-1934
    def record_field_resolution(self, ...) -> ...: ...          # body from core.py:1880-1916
    def handle_quarantine_row(self, ...) -> ...: ...            # body from core.py:1725-1878
    def maybe_emit_progress(self, ...) -> ...: ...              # body from core.py:1939-1985
    def finalize_source_iteration(self, ...) -> ...: ...        # body from core.py:1987-2079
    def run_main_processing_loop(self, ...) -> ...: ...         # body from core.py:2119-2306
```

Transform within bodies: `self._emit_telemetry`/`self._emit_phase_error`/etc. → `self._ceremony.emit_telemetry`/`self._ceremony.emit_phase_error`; intra-cluster `self._finalize_source_iteration` → `self.finalize_source_iteration`, `self._record_field_resolution` → `self.record_field_resolution`, `self._restore_source_iteration_context` → `self.restore_source_iteration_context`, `self._handle_quarantine_row` → `self.handle_quarantine_row`, `self._load_source_with_events` → `self.load_source_with_events`, `self._maybe_emit_progress` → `self.maybe_emit_progress`. Preserve all other parameters exactly (read the originals — the loop takes processor/graph/context args).

- [ ] **Step 2: Wire `Orchestrator`.** In `__init__`, after `_ceremony`:

```python
        self._source_driver = SourceIterationDriver(
            events=self._events, span_factory=self._span_factory, ceremony=self._ceremony,
        )
```

Delete the 7 methods from `core.py`. In `_execute_run` (2308–2432), update `self._run_main_processing_loop(...)` → `self._source_driver.run_main_processing_loop(...)`. Grep for any other call sites of the moved methods and repoint.

- [ ] **Step 3: Repoint the monkeypatch seam.** In `tests/integration/test_adr_019_sweep_durability.py`:
- line ~468: `original_loop = Orchestrator._run_main_processing_loop` → `SourceIterationDriver.run_main_processing_loop`
- line ~539: `monkeypatch.setattr(Orchestrator, "_run_main_processing_loop", _corrupting_loop)` → `monkeypatch.setattr(SourceIterationDriver, "run_main_processing_loop", _corrupting_loop)`
Add the import `from elspeth.engine.orchestrator.source_iteration import SourceIterationDriver`. Adjust the patched function's signature if it referenced `self` as the Orchestrator (it now receives the driver as `self`).

- [ ] **Step 4: Verify.** Run the sweep-durability file specifically (it owns the seam), then focused + full. Must match baseline.

Run: `.venv-wt/bin/python -m pytest tests/integration/test_adr_019_sweep_durability.py -q`

- [ ] **Step 5: Commit** (`refactor(orchestrator): extract SourceIterationDriver; repoint sweep-durability seam`)

---

## Phase 4 — Extract `ResumeCoordinator` + `RunExecutionCore`

**Goal:** Move resume (lines 2434–2546, 2548–2836, 2838–2951) into `ResumeCoordinator`, and extract the run-execution core (`_initialize_run_context` 1551–1660, `_flush_and_write_sinks` 1662–1723, plus the checkpoint-factory wiring) shared by `_execute_run` and `_process_resumed_rows` into `RunExecutionCore` (decision **b**). `Orchestrator.resume()` becomes a one-line delegation (public API preserved). Two seam repoints: `_reconstruct_resume_state` and `_process_resumed_rows`.

> **BLOCKER — confirm before starting (Risk R5):** the in-flight F1 resume fix (`fix/resume-fork-reemit`, adds `node_states.resume_checkpoint_id` + 3-writer `token_data_ref`) touches this exact path. Do NOT begin Phase 4 until the operator confirms F1 has landed OR has been explicitly sequenced after this phase. Extracting against a moving resume target risks a hard rebase and a doubled behavior-preservation review.

### Task 4.1: Extract `RunExecutionCore`

**Files:**
- Create: `src/elspeth/engine/orchestrator/run_core.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`

- [ ] **Step 1: Create `RunExecutionCore`** holding the run-context/sink-flush logic shared by normal and resume paths. Move `_initialize_run_context` (1551–1660) and `_flush_and_write_sinks` (1662–1723) bodies verbatim; thread `checkpoints: CheckpointCoordinator`, `ceremony: RunCeremony`, and config (`_concurrency_config`, `_rate_limit_registry`, `_clock`, `_coalesce_completed_keys_limit`, `_span_factory`) in via the constructor. `_build_processor` (843–989) and `_write_pending_to_sinks` (720–841) move here too — they are pure run-execution helpers used by both paths.

```python
class RunExecutionCore:
    def __init__(self, *, checkpoints, ceremony, span_factory, clock,
                 concurrency_config, rate_limit_registry, coalesce_completed_keys_limit) -> None: ...
    def build_processor(self, ...) -> RowProcessor: ...          # body from core.py:843-989
    def initialize_run_context(self, ...) -> ...: ...            # body from core.py:1551-1660
    def flush_and_write_sinks(self, ...) -> ...: ...             # body from core.py:1662-1723
    def write_pending_to_sinks(self, ...) -> ...: ...            # body from core.py:720-841
```

Transform: `self._build_processor` → `self.build_processor`; `self._checkpoint_interrupted_progress`/`self._write_pending_to_sinks` calls inside `flush_and_write_sinks` → `self._checkpoints.checkpoint_interrupted_progress(..., graph=graph)` / `self.write_pending_to_sinks(...)`; telemetry → `self._ceremony.*`.

- [ ] **Step 2: Wire `Orchestrator`** to construct `self._run_core = RunExecutionCore(...)` and repoint `_execute_run`'s calls. Delete the four moved methods from `core.py`. Verify focused + full green BEFORE proceeding to 4.2 (split the risk).

- [ ] **Step 3: Commit** (`refactor(orchestrator): extract RunExecutionCore shared by run/resume`)

### Task 4.2: Extract `ResumeCoordinator`

**Files:**
- Create: `src/elspeth/engine/orchestrator/resume.py`
- Modify: `src/elspeth/engine/orchestrator/core.py`
- Modify (seam repoints): `tests/integration/test_adr_019_sweep_durability.py`, `tests/unit/engine/test_orchestrator_registry_bootstrap.py`

- [ ] **Step 1: Create `ResumeCoordinator`.**

```python
class ResumeCoordinator:
    def __init__(self, *, db, events, ceremony, checkpoints, run_core) -> None: ...
    def resume(self, ...) -> RunResult: ...                  # body from core.py:2548-2836 (PUBLIC contract — preserve signature exactly)
    def reconstruct_resume_state(self, ...) -> ...: ...      # body from core.py:2434-2546
    def process_resumed_rows(self, ...) -> ...: ...          # body from core.py:2838-2951
```

Transform: `self._delete_checkpoints`/`self._rebase_checkpoint_sequence` → `self._checkpoints.*`; `self._emit_*` → `self._ceremony.*`; `self._initialize_run_context`/`self._flush_and_write_sinks`/`self._make_checkpoint_after_sink_factory` → `self._run_core.*` / `self._checkpoints.make_checkpoint_after_sink_factory(..., graph=graph)`; `self._reconstruct_resume_state` → `self.reconstruct_resume_state`; `self._process_resumed_rows` → `self.process_resumed_rows`. Preserve the `resume()` public signature byte-for-byte.

- [ ] **Step 2: Wire `Orchestrator`.** Construct `self._resume_coordinator = ResumeCoordinator(...)`. Replace `Orchestrator.resume` body with a single delegation that preserves the signature:

```python
    def resume(self, *args, **kwargs):  # keep the REAL explicit signature from core.py:2548
        return self._resume_coordinator.resume(*args, **kwargs)
```
(Use the actual parameter list, not `*args` — copy it from the original `def resume(`.) Delete `_reconstruct_resume_state` and `_process_resumed_rows` from `core.py`.

- [ ] **Step 3: Repoint seams.**
- `tests/integration/test_adr_019_sweep_durability.py:444`: `monkeypatch.setattr(Orchestrator, "_process_resumed_rows", _fail_if_processed)` → `monkeypatch.setattr(ResumeCoordinator, "process_resumed_rows", _fail_if_processed)`.
- `tests/unit/engine/test_orchestrator_registry_bootstrap.py:441`: `monkeypatch.setattr(Orchestrator, "_reconstruct_resume_state", fake_reconstruct_resume_state)` → `monkeypatch.setattr(ResumeCoordinator, "reconstruct_resume_state", fake_reconstruct_resume_state)`. Adjust the fake's `self` to be the coordinator.
- Update patch targets `patch("elspeth.engine.orchestrator.core.RecorderFactory" …)` in `test_resume_failure.py` if `RecorderFactory` is now imported in `resume.py` instead of `core.py` (grep; repoint only if the symbol moved).

- [ ] **Step 4: Verify — full resume-critical ladder + full suite.**

Run: `.venv-wt/bin/python -m pytest tests/integration/test_adr_019_resume_counter_parity.py tests/integration/test_adr_019_sweep_durability.py tests/integration/test_adr_019_cross_table_invariants.py tests/unit/engine/orchestrator/test_resume_failure.py tests/unit/engine/test_orchestrator_registry_bootstrap.py tests/integration/pipeline/test_resume_comprehensive.py -q`
Expected: PASS. Then full suite = baseline.

- [ ] **Step 5: Commit** (`refactor(orchestrator): extract ResumeCoordinator; resume() now delegates`)

---

## Phase 5 — Gate reconciliation & final verification

**Goal:** Reconcile every CI gate the structural moves disturbed, once, and prove the full suite + gates green. Per `notes/tier-model-bulk-remediation-playbook.md`.

### Task 5.1: Reconcile gates

**Files:** allowlist/baseline artifacts (per playbook), the 5 new modules.

- [ ] **Step 1: tier-model.** Run the tier-model enforcement gate (Python 3.13). For each of the 4 new files (`ceremony.py`, `checkpointing.py`, `source_iteration.py`, `resume.py`, `run_core.py`), prefer `@trust_boundary` where a method raises on malformed Tier-3 input with a real source param; otherwise carry forward the per-line allowlist entries that existed for the moved methods (the methods' tier classification did not change — this is a relocation, not a new boundary). Use `scripts/cicd/rotate_tier_model_fingerprints.py` to reconcile fingerprints displaced by the move (read `reference_tier_model_fingerprint_rotation_tool` discipline: restore stale entries first, git-diff + re-run, watch for dup-key data-loss).

- [ ] **Step 2: fingerprint baseline.** Regenerate `fingerprint_baseline.json` (the structural move changes it — regen recurs after every structural change, per the playbook) and confirm `test_baseline_capture_is_self_consistent` passes. NOTE: per project memory this regen may require the operator HMAC key to bless the enforce gate — if so, STOP and hand to the operator; do not bless blind. EXPECT the baseline diff to be **larger than this refactor alone**: RC5.3 already carries pre-existing baseline drift with an owed operator re-sign, so this decomposition's drift batches with it into one HMAC sign run (efficient — but don't be alarmed the diff exceeds your changes).

- [ ] **Step 3: contracts whitelist / immutability counts.** Grep `contracts/` for any path or symbol reference to `orchestrator.core` that moved; repoint. If the class count in `core.py` changed, update `test_immutability_rules` count accordingly.

- [ ] **Step 4: lint surface.** Run the full local lint set mirrored from `ci.yaml` Static-analysis (ruff, mypy, the elspeth_lints checks) per `feedback_run_lints_locally_before_push`. Fix any import-ordering / unused-import fallout from the moves.

- [ ] **Step 5: Full suite, final.**

Run: `.venv-wt/bin/python -m pytest tests/`
Expected: matches the Phase 0 baseline exactly (same pass/deselect counts, no new failures).

- [ ] **Step 6: Confirm `core.py` shrank as designed.**

Run: `wc -l src/elspeth/engine/orchestrator/*.py`
Expected: `core.py` ≈ 900–1000 LOC; four/five new collaborator files; total LOC ≈ unchanged (this is a move, not a rewrite).

- [ ] **Step 7: Commit** (`chore(orchestrator): reconcile tier-model/baseline/lint gates post-decomposition`)

---

## Self-review notes (author)

- **Spec coverage:** every spec unit (RunCeremony/CheckpointCoordinator/SourceIterationDriver/ResumeCoordinator/RunExecutionCore), the shared-run-core decision (b, Phase 4.1), all three monkeypatch seams (Phases 3 & 4.2), R5 (Phase 4 blocker), and gate reconciliation (Phase 5) map to tasks.
- **Honest gaps the executor must close by reading source:** exact parameter lists of `maybe_checkpoint`/`run_main_processing_loop`/`initialize_run_context`/`resume` are cited by line range, not reproduced — the executor MUST copy the real signatures (they carry processor/graph/context/aggregation args this plan abbreviates as `...`). This is deliberate: fabricating those signatures from the AST summary would risk silent arg-drop. The line ranges make the copy unambiguous.
- **Type consistency:** collaborator method names are used consistently across phases (`emit_telemetry`, `maybe_checkpoint(graph=…)`, `run_main_processing_loop`, `reconstruct_resume_state`, `process_resumed_rows`).
