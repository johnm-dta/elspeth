# Tier 1.5 — Post-Merge Verification Cohort (2026-05-07)

## Scope

After the `composer-tier1.5-hardening` branch was fast-forwarded onto `RC5-UX`
and `elspeth-web.service` was restarted to load the updated
`pipeline_composer.md` skill (commit `7dfa3603` — `web_scrape` required-options
documentation), a 6+6 verification cohort was fired against staging
(`elspeth.foundryside.dev`, `gpt-5-mini@temperature=0.0`) to confirm the skill
change is neutral-or-better on the two scenarios most directly affected.

**Cohort design:**
- `url-download-line-explode` ×6 — primary verification target
  (uses `web_scrape` → `line_explode` → `llm` → `json sink`; exercises the
  required-options block the skill update added).
- `rag-text-llm` ×6 — regression control. Despite the name, this scenario also
  uses `web_scrape` → `llm` → `json sink` (without `line_explode`); it covers
  the same plugin path with a slightly simpler shape.

**Run timing:** All 12 runs completed in ~7 minutes wallclock (sequential).
Cost estimate ~$12 against the dta_user OpenRouter credit pool.

## Results

| Scenario                    | Pre-merge GREEN | Post-merge GREEN | Δ |
| --------------------------- | --------------- | ---------------- | - |
| url-download-line-explode   | 5/6 (83%)       | 5/6 (83%)        | 0 |
| rag-text-llm                | 3/6 (50%)       | 3/6 (50%)        | 0 |
| **combined**                | **8/12 (67%)**  | **8/12 (67%)**   | 0 |

The pre-merge baseline batch is the `seq-rebaseline` / `seq-cohort` runs from
`20260506T1501–1506Z`; the post-merge batch is `postmerge-urldl-*` /
`postmerge-ragtxt-*` from `20260506T1604–1611Z`.

## Failure-mode inventory

### url-download-line-explode RED (run 4)

- Sentinels: `runtime preflight failed`, `i cannot mark this pipeline complete`
- Root cause (from `state.json` preview): edge contract violation
  `WebScrapeOutput → LineExplodeInput` — `fetch_status` type mismatch.
- Pre-merge equivalents hit the same edge-contract pattern, distinct fields.
- **Interpretation:** The skill update added required-options docs for the
  initial config of `web_scrape`, but the residual failure mode is producer/
  consumer schema-contract mismatches *between* `web_scrape` and downstream
  transforms. That is a separate problem space and was correctly identified as
  out of scope for this skill change in `composer-tier1.5-final-cohort-report`.

### rag-text-llm REDs (runs 4, 5, 6)

- All three: `state_node_count: null`, `state_output_count: null`,
  `is_valid: null` — pipeline state was never committed.
- All three: `runtime preflight failed: 2 validation errors for
  ElspethSettings — source: Field required, sinks: Field required`.
- **NOT a new pattern.** Pre-merge runs 1 and 2 of the same scenario had
  identical `nodes=None outputs=None` failure shapes.
- **Why pre-existing:** This is the model claiming completion before the
  in-loop tool-call sequence reaches `set_pipeline` / `set_source` / `set_output`.
  Server-side validation correctly refuses the completion. The reason it
  presents as `0 tool calls in messages.json` is a measurement artefact — the
  composer loop's tool calls run server-side and don't surface in the
  client-visible message stream. GREEN runs also show `0 tool calls` in
  `messages.json` — the difference is whether the server-side state ended up
  populated.

## Interpretation

The skill update is **behaviourally neutral** on both scenarios.

- It did not regress the headline metric for either path.
- It did not measurably improve the headline metric either.
- The dominant residual failure modes (`web_scrape` → consumer schema-contract
  mismatches; rag-text-llm pre-state-commit completion claims) are unchanged
  in shape and frequency from the pre-merge baseline.

This matches the skill-change-as-clarity-uplift framing in
`composer-tier1.5-final-cohort-report`: the prose is more truthful about
required `web_scrape` options, but the dominant cohort failure modes were not
*caused by* missing options-prose; they were caused by drift in
*intermediate-edge schema reasoning*, which is a deeper problem.

## Demo-readiness implication

- The demo's primary BA-reporting path (`url-download-line-explode`-shaped
  workflows) holds at 83% first-pass GREEN, identical to the pre-merge
  baseline. That is well above the demo-readiness threshold.
- The rag-text-llm-shaped path (web-scrape-without-line-explode) sits at 50%,
  unchanged from pre-merge. If the operator's live demo prompt resembles this
  shape, expect ~50% first-pass success and budget for one retry.
- No new failure modes introduced by the skill change.

## Pre-demo recommendations (unchanged from pre-merge cohort report)

1. The schema-contract-drift work in §7.6 (broader runtime-preflight error
   rewrite) remains the highest-leverage next investment for moving these
   numbers up.
2. If a hot-fix window opens before the demo, prefer adding *truthful
   intermediate-edge schema declarations* over more skill prose — the skill
   already names the right plugins; the model's failure is reasoning about
   the inter-node contracts, which can't be fixed purely with prose.

## Appendix — run identifiers

Post-merge runs:
- `runs/20260506T1604*url-download-line-explode-postmerge-urldl-{1..6}`
- `runs/20260506T160{7,8,9,10,11}*rag-text-llm-postmerge-ragtxt-{1..6}`

Pre-merge baseline (most recent, for comparison):
- `runs/20260506T1501*url-download-line-explode-seq-rebaseline-{1..6}`
- `runs/20260506T150{3,4,5,6}*rag-text-llm-seq-cohort-{1..6}`
