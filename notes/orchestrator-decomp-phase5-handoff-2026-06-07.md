# Orchestrator decomposition — Phase 5 gate-reconciliation handoff (2026-06-07)

**Branch:** `refactor/orchestrator-decomp` (off `release/0.5.3`), worktree `.worktrees/orchestrator-decomp`.
**Filigree:** epic `elspeth-35d0764ad5`; Phase 5 task `elspeth-9c9b2e0b19` (open).

Phases 0–4 are **landed and fully verified** (behavior-preserving; full suite identical to
baseline, 0 new failures across 15.8k tests). `core.py` 2951 → 1033 LOC (−65%). Five
collaborators: `ceremony.py` (211), `checkpointing.py` (236), `source_iteration.py` (670),
`run_core.py` (567), `resume.py` (913, = F1 functions + ResumeCoordinator).

Commits: `67be2ba8d` `519d44e40` `e348a25fd` `c39ca9b57` `1f5214d03` `44197b5b1`.

## Phase 5 punch-list (what the structural moves disturbed)

### 1. tier-model allowlist — ONE repoint, NOT judge-gated (no HMAC needed for this item)
`config/cicd/enforce_tier_model/engine.yaml:32` is a per-file-rule:
```
- pattern: engine/orchestrator/core.py
  rules: [R5]
  reason: Protocol dispatch for batch-aware transforms and re-raise guards
  max_hits: 2
```
The R5 protocol-dispatch hit **definitely moved to `run_core.py`**: `isinstance(t,
TransformProtocol)` at `run_core.py:491` (in `build_processor`, ex-`_build_processor`). The
*second* R5 hit is a "re-raise guard" (the rule's reason) whose destination is **NOT
gate-confirmed**: a bare `raise` re-raise lives at BOTH `run_core.py:484` AND `core.py:463`,
so the split is **either 2→run_core/0→core OR 1→run_core/1→core — ESTIMATE, verify counts via
a clean gate run.** Action: add a `run_core.py` per-file-rule (R5, max_hits 1 or 2 per the
gate) and reduce the `core.py` entry to whatever the gate reports (do NOT assume 0). All other
`raise`s in core.py/resume.py are plain `OrchestrationInvariantError` (Tier-1 explicit), not R5.
Per-file `pattern` rules are not fingerprinted/judge-gated, so this edit does not itself need
the HMAC key — BUT see #3 (the gate won't run clean until the unrelated drift is cleared).

No `@trust_boundary` decorators exist in the orchestrator package; no fingerprinted
(`allow_hits` key with `fp=`) entries reference any of the moved methods. So this is the ONLY
tier-model item the refactor creates.

### 2. fingerprint_baseline.json — regen, HMAC-gated (batches with pre-existing drift)
`test_baseline_capture_is_self_consistent` is RED. The diff is DOMINATED by pre-existing
RC5.3 drift (~90 `core/config.py` entries removed) that predates this branch — see
`project_rc53_fingerprint_baseline_drift_2026-06-04`. The decomposition's structural delta
batches on top. Regen via `scripts/cicd/regen_fingerprint_baseline.py` under the HMAC key,
in the same pass that clears the pre-existing drift. **Do not bless blind.**

### 3. BLOCKER for a clean gate run (pre-existing, not this refactor)
`trust_tier.tier_model` aborts at allowlist load with a `scope_fingerprint mismatch` on the
judge-gated entry for `web/composer/boot_probe.py:R6:probe_composer_config` — its enclosing
scope changed (commit `959d465b3`, the boot-probe max_tokens fix) and the entry needs an
operator re-`justify` (HMAC + judge). Until that is re-justified, the tier-model gate cannot
complete in either this worktree or CI `required` mode, so #1's new counts can't be verified
by the gate. Clear this first, then verify #1.

### 4. Clean / green already (no action)
- `test_immutability_rules`: 46 passed (class-count gate unaffected — new classes in new files).
- `contracts/`: no references to any moved `orchestrator.core` symbol.
- `ruff` + `mypy`: clean on all six orchestrator files.
- Monkeypatch seams: all repointed (verified, both class- and instance-level forms).

### 5. Final verification after #1–#3
`env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 .venv-wt/bin/python -m pytest tests/` must match the
Phase-0 baseline (255-entry failure set, all environmental; `comm` set-diff empty). Then the
branch is merge-ready into `release/0.5.3` (`--no-ff`).

## Worktree env (reusable)
Dedicated `.venv-wt` (Python 3.13.1); NEVER the symlinked `.venv` (→ main, leaks main's
source). ALL pytest needs `env -u VIRTUAL_ENV PYTHONNOUSERSITE=1 .venv-wt/bin/python -m pytest`.
Coverage unusable (numpy C-ext single-init vs coverage machinery). No `.env` → ~248
credential/service tests fail environmentally (the baseline; track the SET, not green).
