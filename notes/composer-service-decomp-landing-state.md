# composer/service.py decomposition — landing state (2026-05-30)

Branch: `composer-service-decomp` (off RC5.2 `4925de5e5`). Worktree-resident.
Design: `docs/superpowers/specs/2026-05-29-composer-service-decomposition-design.md`.
Plan: `docs/superpowers/plans/2026-05-30-composer-service-decomposition.md`.

## What landed (all behaviour-preserving, independently reviewed)

| Phase | Commit(s) | Result |
|-------|-----------|--------|
| Characterization gate | `d6c71d198`→`28173a8f3` | 8 tests pinning every dispatch terminal arm (`test_dispatch_arms_characterization.py`); arm #10 documented as driver-unobservable (raises `ComposerConvergenceError`); pre-covered arms #3/#6/#11–#16 mapped to existing audit-wiring assertions |
| Verbatim move | `e5d0043be`, `3b3898ac1` | `_dispatch_tool_batch` (1,298 LOC) → `tool_batch.run_tool_batch`, byte-identical body (proven by reverse-transform diff = one predicted hunk); `ToolBatchContext`/`BatchAccumulator` carriers; mock patches retargeted `service.execute_tool`→`tool_batch.execute_tool` (40→40, pure relocation) |
| Extractions | `97316e049`, `a2fe3bc52`, `d6e112ed9` | `_persist_turn_audit`→`turn_audit.py` (audit-critical, verbatim verified), `_compute_availability`+`ComposerAvailability`→`availability.py` (re-exported; `_compute_availability` kept as delegating method so conftest monkeypatch binds), `_finalize_no_tool_response`→`no_tool_finalize.py` |
| Lint | `a2e8d61a7` | ruff import-ordering on new modules |
| PR-review remediation | `aea77ecd4` | BatchAccumulator trimmed to its 4 live fields (9 dead fields + false docstring removed); `ComposerAvailability.__post_init__` correlation invariant; arm-#4 cache-hit pinned to ordered `[False, True]`; all stale `service.py:NNNN` citations in the char-test re-cited to `tool_batch.py` anchors |
| Immutability gate | `75f7c13ff` | `ToolBatchContext` FG3 (`enforce_freeze_guards/web.yaml`) + frozen_annotations FA `discovery_cache` (`enforce_frozen_annotations/existing.yaml`) — full gate `--root src/elspeth` now exits 0 |
| Skill-theatre removal | `930db782c` | deleted the 4 `test_skill_*` prompt-grep tests from `test_advisor_tool.py` (3 were the long-standing red failures); behavioural `test_f1_skill_text_*` retained |

**`service.py`: 5,313 → 3,653 lines (−1,660 / −31%).** mypy clean; ruff clean; immutability gate clean.

## Test state
`pytest tests/unit/web/composer/ tests/property/web/composer/` → **2,335 passed, 1 failed**.
The 3 `test_advisor_tool.py` skill-prompt-theatre failures were REMOVED (commit `930db782c`, per the "no test_skill tests" doctrine). The 1 remaining failure is PRE-EXISTING:
- `test_compose_loop_invariants.py::TestComposeLoopAuditMachine::runTest` — Hypothesis cache-sensitive timing flake; its failing example monkeypatches `persist_compose_turn_async` (driver step AFTER `run_tool_batch` returns), wholly outside moved code. Observation `elspeth-obs-edc41f7acc`. DO NOT misattribute to this work.

## RESOLVED this session (was deferred)
- **freeze-guard FG3 + frozen_annotations FA for `ToolBatchContext`** — done in `75f7c13ff` (see table above). The deliberate-mutability exemption is now mechanically registered, not just documented.

## STILL DEFERRED to final reconciliation / new CI (operator-sanctioned; branch has `--no-verify`)
CI allowlists were intentionally NOT updated per-commit (avoids the per-commit fingerprint-rotation tax + dup-key data-loss footgun). Co-land before merge:

**tier-model (`trust_tier.tier_model`) — fingerprint reconciliation, NO code fix:**
- New live findings on `tool_batch.py` (the relocated trust-boundary defensive patterns, previously allowlisted on `service.py`): R6 except-clauses (json-decode, PydanticValidationError, ToolArgumentError), R5 isinstance, R4 broad-except ×2. Port their exemptions to `tool_batch.py`/`run_tool_batch` keys, OR add a `pattern: web/composer/tool_batch.py` block.
- `service.py` live findings now uncovered because fingerprints shifted when the file shrank (R8 setdefault ×2, R1 dict.get ×3, R5 isinstance, R2 getattr ×2) — same pre-existing allowlisted patterns, just re-fingerprint.
- ~12 stale `service.py` allowlist entries (methods moved/shrank) — prune.
- Use Python 3.13 venv (matches main) or the check reports ~300 spurious violations. Do NOT run the rotate tool bare (dup-key data-loss; `git diff` + re-run after rotating).

## NOT done (optional, operator-deferred; now filed as filigree tickets)
- **Phase 3** (collapse the terminal-arm emit idiom into helpers) → **`elspeth-c4a555da1c`** (P3, label `composer-service-decomp`). Note: the verbatim move used an alias-preamble, so the loop-carried state lives in inline locals (not the carrier — `BatchAccumulator` was trimmed to its 4 live fields in `aea77ecd4`); Phase 3 entails a locals→carriers migration first. Higher risk, re-touches audit-critical arms.
- **Phase 4** (sync ToolDeclaration convergence — fold `get_plugin_schema` post-success hook) → **`elspeth-dcb5a6be0b`** (P3, label `composer-service-decomp`). Async carve-outs (`request_advisor_hint`, `request_interpretation_review`) remain deferred to `elspeth-f5da936747` regardless.
