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

**`service.py`: 5,313 → 3,653 lines (−1,660 / −31%).** mypy clean (60 files); ruff clean.

## Test state
`pytest tests/unit/web/composer/ tests/property/web/composer/` → **2,336 passed, 4 failed**.
All 4 failures are PRE-EXISTING (present identically on base `28173a8f3` — verified by reverting the working tree to base with the same `.hypothesis` cache):
- 3× `test_advisor_tool.py` skill-prompt theatre (being fixed on another branch; resolve on rebase).
- 1× `test_compose_loop_invariants.py::TestComposeLoopAuditMachine::runTest` — Hypothesis cache-sensitive timing flake; its failing example monkeypatches `persist_compose_turn_async` (driver step AFTER `run_tool_batch` returns), wholly outside moved code. Observation `elspeth-obs-edc41f7acc`. DO NOT misattribute to this work.

## DEFERRED to final reconciliation / new CI (operator-sanctioned; branch has `--no-verify`)
CI allowlists were intentionally NOT updated per-commit (avoids the per-commit fingerprint-rotation tax + dup-key data-loss footgun). Co-land these before merge:

**freeze-guard (`immutability.freeze_guards`) — 1 entry:**
- `tool_batch.py` `ToolBatchContext` FG3: frozen dataclass with mutable `discovery_cache`/`runtime_preflight_cache` fields, deliberately NOT deep-frozen (they are shared mutable caches the loop writes into). Add a per-file-rule / FG3 allowlist entry in `config/cicd/enforce_freeze_guards/web.yaml` with that justification. Precedent: `ProducerEntry` in `_producer_resolver.py`.

**tier-model (`trust_tier.tier_model`) — fingerprint reconciliation, NO code fix:**
- New live findings on `tool_batch.py` (the relocated trust-boundary defensive patterns, previously allowlisted on `service.py`): R6 except-clauses (json-decode, PydanticValidationError, ToolArgumentError), R5 isinstance, R4 broad-except ×2. Port their exemptions to `tool_batch.py`/`run_tool_batch` keys, OR add a `pattern: web/composer/tool_batch.py` block.
- `service.py` live findings now uncovered because fingerprints shifted when the file shrank (R8 setdefault ×2, R1 dict.get ×3, R5 isinstance, R2 getattr ×2) — same pre-existing allowlisted patterns, just re-fingerprint.
- ~12 stale `service.py` allowlist entries (methods moved/shrank) — prune.
- Use Python 3.13 venv (matches main) or the check reports ~300 spurious violations. Do NOT run the rotate tool bare (dup-key data-loss; `git diff` + re-run after rotating).

## NOT done (optional, operator-deferred this session)
- Phase 3 (collapse the terminal-arm emit idiom into helpers). Note: the verbatim move used an alias-preamble, so `BatchAccumulator`'s mutable fields are semi-vestigial (body works on locals); Phase 3 would entail a locals→carriers migration first. Higher risk, re-touches audit-critical arms.
- Phase 4 (sync ToolDeclaration convergence — fold `get_plugin_schema` post-success hook). Async carve-outs (`request_advisor_hint`, `request_interpretation_review`) remain deferred to `elspeth-f5da936747` regardless.
