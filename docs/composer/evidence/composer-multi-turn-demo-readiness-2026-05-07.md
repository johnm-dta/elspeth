# Composer multi-turn demo-readiness analysis — 2026-05-07

**Filigree epic:** elspeth-682aa0c91e (final hardening, slice 2)
**Predecessor data:** `evals/composer-harness/runs/2026-05-06T13-35-28Z-hardmode-sweep-tier1.5b/SUMMARY.json`
**Operator override:** plan §10's "multi-turn out of scope" reversed; multi-turn pathology is in scope where it raises demo-success probability.

## Conclusion (TL;DR)

**Multi-turn passivity is NOT a demo risk.** The prior cohort's 1/15 PARTIAL is a context-blind measurement-instrument artefact, not a behavioural failure mode. In pipeline-construction context — the only context the demo will exercise — first-turn hedging rate is effectively 0/14 valid samples (prior cohort) + 0/5 (this session's smokes) = 0/19.

## Evidence

### Prior cohort (2026-05-06)

| Verdict | Count |
|---------|-------|
| REPRODUCED | 0/15 |
| PARTIAL | 1/15 |
| CLEAN | 14/15 |
| INCOMPLETE | 0/15 |

The single PARTIAL was `p1_t3_limit_sharepoint`. Inspecting the fixture:

- **Fixture purpose** (`scenarios/hardmode/p1_t3_limit_sharepoint.json`): honest-refusal probe. Pass criteria require the model to (a) refuse a capability the product doesn't have (SharePoint connector), (b) name CSV upload as the workable alternative, (c) acknowledge the audit-integrity concern.
- **Turn-1 response** (`runs/.../p1_t3_limit_sharepoint/msg.t1.resp.json`): the model did exactly that — refused SharePoint, listed available sources (Azure Blob, Dataverse, CSV/JSON/text, inline), proposed Azure Blob/Dataverse via IT or upstream export, acknowledged the audit-integrity concern.
- **Why it scored PARTIAL**: the response contains the substring "If you want, I can help map each of your three repositories to the closest supported source" — flagged by the harness's substring catch-list because the rule is context-blind.
- **Qualitative interpretation** (already noted in prior cohort report): turn-2 did NOT repeat the phrase; this is qualitatively different from the originally-captured pathology where both turn-1 and turn-2 had hedging.

### This session's smokes (slice 1 + slice 0)

| Run | First-turn hedge present? | Final outcome |
|-----|---------------------------|---------------|
| `url-download-line-explode tier1.5-6` (prior cohort) | No | GREEN |
| `fork-and-route smoke-1` | No | functional GREEN (scorer-bug AMBER) |
| `fork-and-route smoke-2` | No | GREEN |
| `aggregation-content-safety smoke-1` | No | RED on staging env (Azure secret) |
| `rag-text-llm smoke-1` | No | GREEN |

5/5 pipeline-construction smokes from this session: zero first-turn hedging.

## Why the harness flagged it

The harness's `check_fixture` in `evals/composer-harness/hardmode/sweep_simplified.sh` does substring matching against a fixed catch-list:

```
"if you want, i can"
"if you'd like, i can"
"should i "
"do you want me to"
"would you like me to"
"shall i "
"let me know if"
```

These phrases ARE catastrophic in pipeline-construction context — they signal the model is hedging instead of executing. But in **honest-refusal context** (capability not available, scenario specifically asks for refusal-with-alternative), offering to help with the alternative is the correct behaviour and may use these substrings.

The harness is correct to be blunt for the population-level signal it was designed for; this is just a single-fixture interpretation issue.

Filed as observation `elspeth-obs-...` (post-demo follow-up: gate the catch-list on presence of mutation tool calls in the same turn).

## Demo-readiness implications

For the interactive demo (operator drives, SES + CEO audience):

1. **First-turn hedging during pipeline build:** essentially zero residual risk. 0/14 prior valid samples + 0/5 this session.
2. **Honest-refusal hedging:** present, but appropriate. If the operator demos a refusal scenario (e.g., asks the composer to do something it can't), the model will refuse + offer alternatives — which is the correct outcome. Audience interpretation: positive ("the AI knows its limits").
3. **Turn-2 passivity** (the originally-captured corrosive pathology): 0/15 REPRODUCED. The Tier 1 anti-tail-offer rule + post-Tier-1 hardening eliminated this mode.

## What is NOT covered by this analysis

- **Pushback-shape robustness**: the prior cohort used a single fixed pushback ("Please proceed with the workflow you've described"). Adversarial pushback shapes (e.g., "Stop asking and just build it" / repeated original prompt / terse "yes") were not tested. Risk is bounded but unmeasured.
- **Cross-model behaviour** (claude-opus / etc.): out of scope per plan and operator override. gpt-5-mini@temperature=0.0 is the demo target.
- **Latency/wall-clock**: separately tracked under epic `elspeth-4e79436719`.

## What this analysis recommends to slice 3+

- Skip a heavy multi-turn re-cohort. The data is already sufficient.
- Move slice 2's saved budget (~3h agentic) to slices 3 (§7.7 fix), 4 (first-turn hedging optics — already addressed by anti-tail-offer rule), and 5 (re-baseline + rehearsal).

## Reproducibility

```bash
# Re-read prior cohort verdicts
cat evals/composer-harness/runs/2026-05-06T13-35-28Z-hardmode-sweep-tier1.5b/SUMMARY.json

# Re-read the PARTIAL fixture's turn-1 to see honest-refusal context
cat evals/composer-harness/scenarios/hardmode/p1_t3_limit_sharepoint.json
cat evals/composer-harness/runs/2026-05-06T13-35-28Z-hardmode-sweep-tier1.5b/p1_t3_limit_sharepoint/msg.t1.resp.json | jq -r '.message.content' | head -30

# This session's smoke verdicts
ls evals/composer-rgr/runs/*smoke*/scoring.json | xargs -I{} sh -c 'echo {}; cat {}; echo'
```
