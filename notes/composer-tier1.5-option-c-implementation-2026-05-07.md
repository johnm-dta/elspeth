# Tier 1.5 §7.6 Option C Empty-State Recovery Nudge — Implementation Note

**Date:** 2026-05-07
**Status:** SHIPPED on branch `composer-tier1.5-hardening`
**Design doc:** `notes/composer-tier1.5-option-c-design-2026-05-07.md`

## What landed

1. **New module** `src/elspeth/web/composer/empty_state_recovery.py`
   - `EmptyStateRecoveryTracker` (single-fire-per-compose guard, mirrors `AntiAnchorTracker` shape)
   - `_RECOVERY_NUDGE_CONTENT` constant carrying the GENERIC nudge text with the stable `[ELSPETH-RECOVERY-NUDGE]` marker per design §2.

2. **Service.py wiring** in `_compose_loop`:
   - Tracker instantiated at the top of `_compose_loop` next to `AntiAnchorTracker`.
   - New private method `_should_fire_recovery_nudge` encapsulates the conjunctive trigger predicate (state-empty AND prior-mutation-attempt AND tracker-fresh AND budget-remaining >= 1).
   - New module-level helper `_has_prior_mutation_attempt(invocations)` for the pure-Q&A guard.
   - Hook injects the nudge ONLY at the primary no-tool-call site (~line 1129). The bonus-call path (~line 1760) is left untouched per the design's load-bearing constraint.
   - Nudge fires → tracker records → user message appended → `composition_turns_used += 1` → `continue` outer loop.

3. **Tests:**
   - New file `tests/unit/web/composer/test_empty_state_recovery.py` (6 tests on the tracker primitive + content marker contract).
   - New `TestOptionCRecoveryNudge` class in `test_service.py` (5 integration tests covering the full predicate matrix + bonus-call non-injection).
   - 23 existing tests updated to provide one extra `_make_llm_response(content="...")` post-nudge response. Pattern: tests that mutate state with a tool that doesn't populate `source/nodes/outputs` (e.g. `set_metadata`, `set_source` with bad args) trigger the nudge and need an extra LLM-response slot.

## Notable test-update patterns

Tests in this codebase often use `mock_llm.call_args_list[1][0][0][-1]` to grab the tool message. With the nudge injected, `[-1]` of the second LLM call's input messages is now the nudge user message, not the tool message. Where that pattern broke, the fix was to filter the message list for `role == "tool"` and pick the most recent one. Recommended pattern for future tests:

```python
second_call_messages = mock_llm.call_args_list[1][0][0]
tool_msg = next(m for m in reversed(second_call_messages) if m.get("role") == "tool")
```

## Design-doc deviations

None of substance. Three minor things worth noting:

1. The design suggested either a `recovery_turns_used` counter or piggy-backing on `composition_turns_used`. Implemented the latter per the design's "single-fire guard plus budget check" preference. Mechanically: nudge increments `composition_turns_used` by 1, charging the nudged turn as a composition attempt (the design endorses this).

2. The design called out a possible loop-trap (`Trap C`) where the budget-exhaustion bonus-call path could re-enter the loop. Mitigation per design: keep the nudge-injection out of `_finalize_no_tool_response` entirely. Implemented exactly that — the hook lives in `_compose_loop` at the primary no-tool-call site only. The bonus-call site at line ~1760 calls `_finalize_no_tool_response` directly without consulting the trigger predicate, so the bonus-call contract is preserved.

3. The design's "verification" section talked about a 25–50 run cohort. Out of scope here — this commit is the implementation only. Cohort scoring + cohort run is the parent session's responsibility.

## Cohort detection contract

Per design §6, the nudge can be detected post-hoc by querying audit-DB `chat_messages` for the literal string `[ELSPETH-RECOVERY-NUDGE]`. The marker prefix is pinned in `_RECOVERY_NUDGE_CONTENT` and verified by `test_nudge_content_carries_stable_marker`. No new audit column required.

## Signal worth surfacing to the operator

The pre-existing 27 test failures after wiring the trigger reveal that "single failed mutation → empty state → surrender" is the COMMON test pattern, not the drift-many-times pattern the design's evidence section described (sessions 2cf59016 etc had 20+ tool calls before surrender). The N=1-prior-mutation trigger means the nudge fires on every one-shot failure in production where the operator's request was clearly a build attempt. After observing the cohort, the operator may want to revisit whether the threshold should be N=3 mutations (closer to the original surrender-after-many-attempts framing) or stay at N=1 (broader recovery window). This is an empirical question that requires the cohort data, not a design-time decision.

## Verification

- `mypy src/elspeth/web/composer/` clean
- `ruff check src/elspeth/web/composer/` clean
- `pytest tests/unit/web/composer/` 863 passed (was 852 baseline → +11 net new tests; 23 existing tests updated mechanically)
- `pytest tests/unit/web/ tests/integration/web/` 2380 passed, 1 xfail (unrelated `elspeth-879f6de6bd`)

## What did NOT change

- Anti-anchor tracker (§7.7) — orthogonal mechanism, untouched
- `_finalize_no_tool_response` — unchanged; the bonus-call path semantics are preserved
- Audit schema — no new column; cohort scoring keys off the marker string
- Plugin code — no changes to plugins, sources, sinks, transforms, or gates
- Service is restartable — `elspeth-web.service` restart is the only deploy step (parent session task, NOT done here)
