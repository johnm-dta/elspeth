# Tier 1.5 §7.6 Option C — Recovery Nudge Verification (2026-05-07)

## What landed

Commit `2c2a61f2 feat(composer/loop): empty-state recovery nudge`. New
`EmptyStateRecoveryTracker` + predicate in `_compose_loop` injects a
single GENERIC system-attributed nudge when the model surrenders with
no-tool-call reply on a structurally-empty state (with prior mutation
attempt + budget remaining). Mirrors the §7.7 anti-anchor pattern.

## Cohort

25 runs against staging (20 rag-text-llm + 5 url-download). Production
budgets: `composer_max_composition_turns=10`,
`composer_max_discovery_turns=5` (total = 15 turns).

## Numerical results

| Scenario | Pre-Option-C baseline | Post-Option-C |
| --- | --- | --- |
| rag-text-llm | 5/6 (83%) | **11/20 (55%)** |
| url-download | 3/4 (75%) | **4/5 (80%)** |
| combined | 8/10 (80%) | **15/25 (60%)** |

Headline rate dropped. With small-N (8 vs 25) the difference is partly
noise, but three of the missed runs are HTTP 422 (discovery budget
exhausted), which is a real new pattern.

## What the data actually shows

### Three categories of failure observed

| Category | Count | Root cause |
| --- | --- | --- |
| HTTP 422 budget-exhaustion | 3 | Model spent all 5 discovery turns on schema lookups and got cut off before composing — orthogonal to Option C |
| nodes=None (state never committed) | 4 | Model made 1-6 mutation attempts, all rejected, surrendered with prose. Empty-state passthrough fired correctly. |
| nodes=0 or 2 with build-failure sentinel | 3 | Synthesizer fired (model's state was non-empty by predicate). Pre-existing failure path, unchanged. |

### Did Option C fire? Suggestive but not confirmed

The `[ELSPETH-RECOVERY-NUDGE]` marker was added to `llm_messages` (the
in-loop transient list passed to the LLM) but is **not persisted to
`chat_messages`** (the DB-backed user-visible message log). My
verification mechanism (querying `chat_messages` for the marker) is
therefore blind by construction. Both the implementation sub-agent and
the reviewer assumed `chat_messages` would surface this; both were wrong.

**Suggestive evidence the nudge IS firing**: the captured RED-04 final
assistant content reads *"The minimal valid shape for your requested
workflow is: 1. CSV source... 2. web_scrape... 3. llm... 4. JSON
output..."*. The phrasing *"minimal valid shape"* matches the nudge
prose verbatim (*"Fall back to a minimal shape: one source, one
transform, one output sink"*). It is not a stock surrender phrasing.
**Hypothesis**: nudge fires → model adopts the framing in its
second-turn prose → model surrenders again with empty state → empty-
state passthrough fires. This is Trap A from the design doc.

**Counter-evidence**: RED elapsed times are 36-44s (no outlier), but
parallel tool calls within a turn compress wallclock so this isn't
conclusive.

**Verification gap**: confirming nudge-fired requires either (a)
in-loop instrumentation (a counter or telemetry attribute) the current
implementation does not have, or (b) re-running with `journalctl` tail
on the progress event sink — feasible but didn't happen in this cohort.

### Why the cohort rate dropped

Most likely combination of:

1. **Small-sample noise.** 8/10 → 15/25 has wide CI overlap.
2. **Increased discovery exploration.** The §7.6 fixes shipped earlier
   in this session give the model better per-call feedback. Plausibly
   the model is now exploring schemas more thoroughly (read more
   `get_plugin_schema` outputs) before attempting builds — and the
   5-turn discovery cap can't accommodate that. Three of the new REDs
   are HTTP 422 budget-exhaustion, which fits this pattern.
3. **Option C had no observable effect** on the conversion rate of the
   empty-state-surrender slice — the 4 nodes=None REDs all surfaced via
   empty-state passthrough, with no evidence of conversion to GREEN.

## Demo-readiness assessment

- **The earlier fixes (`0b176800` + `2e2c7bc8`) are still load-bearing
  and verified.** Edge-contract messages are LLM-actionable; empty-
  state passthrough surfaces model prose instead of Pydantic noise.
- **Option C as shipped has unproven efficacy.** The mechanism is in
  place. The detection is broken (separate fix needed). The
  conversion-rate signal is weak-to-none on this cohort.
- **Discovery budget at 5 is the most demo-relevant new finding.** If
  3/25 = 12% of runs hit discovery exhaustion, that's a real demo risk.
  Raising to 8 or 10 would buy the model more headroom for schema
  exploration without changing other system behaviour.

## Recommended actions

In rough priority order, none of which are blockers for the demo:

1. **Raise `composer_max_discovery_turns` from 5 to 8 or 10.** Single
   env var change, immediate effect. Targets the new HTTP 422 failure
   mode which appears unrelated to Option C and is likely demo-visible.

2. **Add an audit attribute or telemetry counter for `recovery_nudge_fired`**.
   Without this, future Option C cohorts have the same blindness this
   one did. Small change in `_compose_loop` — increment a counter when
   `tracker.record_fire()` is called and emit it through the existing
   composer telemetry. Makes Option C verifiable.

3. **Decide on Option C's final shape.** Three honest choices:
   - **Keep as-is.** Mechanism is in place, no harm if it never fires,
     possibly helps in cases not visible to this cohort.
   - **Revert.** No measured benefit, demo doesn't need it (the earlier
     fixes carry the demo win), revert returns to a simpler call site.
   - **Pivot to big-brother first.** The MCP advisor tool the operator
     mentioned addresses the same surrender failure mode with a
     stronger mechanism (smart-model advice instead of generic-prose
     nudge). Option C might be redundant once big-brother lands.

## Appendix — run identifiers

- `runs/2026*optionc-{ragtxt,urldl}-*` — the cohort
- HTTP 422 runs: ragtxt-02, ragtxt-07, ragtxt-20
- Empty-state REDs (Option C trigger pattern): ragtxt-04, ragtxt-11,
  ragtxt-17, ragtxt-19
- Synthesizer-path REDs (Option C correctly did NOT fire): ragtxt-12,
  ragtxt-13, ragtxt-14
- Suggestive nudge-fired evidence: ragtxt-04 final content contains
  *"the minimal valid shape"* verbatim from the nudge prose.
