# Tier 1.5 §7.6 — Edge-Contract Error Restructure Verification (2026-05-07)

## What landed

Commit `0b176800 feat(composer): structural fix — LLM-actionable edge-contract
preflight errors`. The structural change preserves `CompatibilityResult`
through the `GraphValidationError → ValidationError` translation so the
runtime-preflight error message surfaced to the composer LLM names
producer/consumer node IDs, lists per-field issues with data-flow
nomenclature, and points at concrete `patch_node_options` call shapes for
the fix.

Pre-existing failure shape (captured Tier 1 RED, session
`20260506T160557Z-url-download-line-explode-postmerge-urldl-4`):

```
I cannot mark this pipeline complete yet because runtime preflight failed:
Edge from 'transform_fetch_rules_46d77f2bcb4a' to
'transform_split_lines_c3322ba122ca' invalid: producer schema
'WebScrapeOutput' incompatible with consumer schema 'LineExplodeInput':
Type mismatches: fetch_status (expected str | None, got int).
```

New failure shape (synthesised — not exercised in this cohort, see below):

```
I cannot mark this pipeline complete yet because runtime preflight failed:
Edge contract violation between producer node 'transform_fetch_rules_...'
(schema 'WebScrapeOutput') and consumer node 'transform_split_lines_...'
(schema 'LineExplodeInput'):
Type mismatches:
  - field 'fetch_status': consumer requires 'str | None', producer emits 'int'
. Suggested fix: Most edge-contract failures come from the consumer
over-declaring fields it doesn't operate on. Try option (a) first.

  (a) Relax the consumer's input schema on node 'transform_split_lines_...'.
      Either:
      - Change the declared field type(s) to match what the producer emits
        (see Type mismatches above).
      - Or switch the consumer's input schema mode to 'flexible' so it
        accepts the producer's full output without redeclaring every field.
      Tool: patch_node_options(node_id='transform_split_lines_...',
        patch={'schema': {...}})

  (b) Patch the producer node 'transform_fetch_rules_...'. Note: plugin
      output schemas are largely baked-in by the plugin's contract — this
      option only works if you mis-declared the producer's schema in your
      initial set_pipeline / upsert_node call.
      Tool: patch_node_options(node_id='transform_fetch_rules_...',
        patch={'schema': {...}}).
```

## Cohort

8 runs (4 url-download-line-explode + 4 rag-text-llm) against staging
running the merged code, fired ~5 minutes after `sudo systemctl restart
elspeth-web.service`. Sequential. Cost ~$8 against OpenRouter credit pool.

## Verdict

| Scenario                  | Pre-fix GREEN | Post-fix GREEN |
| ------------------------- | ------------- | -------------- |
| url-download-line-explode | 5/6 (83%)     | **4/4 (100%)** |
| rag-text-llm              | 3/6 (50%)     | **4/4 (100%)** |
| **combined**              | **8/12 (67%)**| **8/8 (100%)** |

**However — the new error message path was not exercised in this cohort.**

`journalctl -u elspeth-web.service` for the 5-minute cohort window contains
zero `preflight failed`, `Edge contract`, or `Type mismatch` matches. Every
run converged on first try (`is_valid=True`, single assistant message,
`nodes=2`, `outputs=1`). The new prose was on the wire but never displayed
to the LLM because the LLM didn't trigger any edge-contract failures.

## Honest interpretation

The cohort delivers:

- **No regression.** The new code path didn't break the success path.
  Pre-existing first-pass behaviour is preserved.
- **First-pass rate appears healthy.** 8/8 in a 4-run-per-scenario sample.
- **Zero direct evidence of the fix improving recovery.** The fix only
  activates when an edge-contract failure occurs. None did.

The 8/12 → 8/8 delta is most plausibly small-sample noise (4-run binomial
100% has a wide CI; 80–95% of true success-rate space is consistent with
4/4 GREEN). The 25-minute time gap between cohorts also gives OpenRouter
quality variance plenty of room to move.

This is **load-bearing insurance, not measured improvement** for the demo.
When the model does fail an edge contract during the live demo, it now has
actionable recovery info instead of opaque Pydantic noise. The previously
captured failure mode (model surrenders on `Type mismatches: f (expected
X, got Y)` because there's no obvious next move) cannot be eliminated by
prose alone, but the new prose at least gives it a concrete tool call to
attempt.

## What would actually verify the fix's recovery path

Two options, both deliberate:

1. **Forced-failure scenario.** Author a scenario whose opening prompt is
   designed to push the model into an edge-contract failure (e.g., "set up
   a pipeline where line_explode declares fetch_status:str|None as
   required" with operator misframing that lures the model into the bad
   schema). Score whether the model converges on the second turn after
   seeing the new error, vs the historical surrender behaviour.
2. **Audit-DB inspection across a larger cohort.** Query the staging
   audit DB for sessions where runtime preflight rejected a completion
   claim, and inspect the next-turn behaviour. With enough volume from
   live operator usage post-deploy, this becomes a real-world signal.

Neither is on the demo critical path. Recommended post-demo work, not
pre-demo.

## Demo readiness

- Fix is on disk, merged onto RC5-UX, deployed on staging, healthy.
- 1422 unit tests pass (1408 pre-existing + 14 new pinning the new
  message/suggestion shape).
- 8/8 cohort GREEN — at minimum no regression, plausibly above the
  pre-fix baseline.
- No new failure modes introduced.
- The fix is insurance for the live demo: when operator drives a prompt
  that hits an edge-contract drift (which empirically happens for ~20% of
  the demo-relevant scenarios), the model has actionable recovery info.

**Recommendation: keep shipped.** The structural change is safe, well-
tested, and the right shape for the underlying problem. Statistical
verification of the recovery path is genuinely post-demo work.

## Appendix — run identifiers

Post-fix cohort:
- `runs/20260506T16{33,34,35,35,36,36,37,38}*postfix-{urldl,ragtxt}-{1..4}`

Pre-fix baseline (most recent, for comparison):
- `runs/20260506T16{04,05,06,07}*url-download-line-explode-postmerge-urldl-{1..6}`
- `runs/20260506T16{07,08,09,10,11}*rag-text-llm-postmerge-ragtxt-{1..6}`
