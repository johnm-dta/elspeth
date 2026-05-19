# Tier 1.5 §7.6 — Empty-State Finalize Passthrough Verification (2026-05-07)

## What landed

Commit `2e2c7bc8 feat(composer): empty-state finalize-time passthrough —
surface model prose, not Pydantic noise`. Sister commit to the earlier
edge-contract restructure (`0b176800`). Both target the dominant
demo-relevant failure modes identified in the post-merge cohort report.

The structural change: in `_finalize_no_tool_response`, when state is
structurally empty (no source, no nodes, no outputs) AND runtime preflight
is invalid, the synthesizer is *wrong* — the model isn't lying about
completion, it's reporting honest failure to converge after retrying
20+ tool calls. Pass through the model's content with a system-attributed
suffix telling the user how to proceed.

## Cohort

10 runs (6 rag-text-llm + 4 url-download-line-explode), staging running
both edge-contract restructure (`0b176800`) and empty-state passthrough
(`2e2c7bc8`). Cost ~$10. Sequential.

## Numerical results

| Scenario | Pre-fix baseline | Edge-contract only | + Empty-state passthrough |
| --- | --- | --- | --- |
| url-download-line-explode | 5/6 (83%) | 4/4 (100%) | **3/4 (75%)** |
| rag-text-llm | 3/6 (50%) | 4/4 (100%) | **5/6 (83%)** |
| combined | 8/12 (67%) | 8/8 (100%) | **8/10 (80%)** |

The 8/12 → 8/10 (67% → 80%) headline cohort delta is statistically
suggestive but small-sample-noisy. The 4-run binomial CI for 75% includes
a true rate as low as ~30%. Real improvement is only confirmable with a
larger cohort (n≥30).

## Qualitative results — the load-bearing win

For both RED runs, the new code path fired correctly:

| Check | rag-text-llm RED-3 | url-download RED-3 |
| --- | --- | --- |
| `[ELSPETH-SYSTEM]` suffix present | ✓ | ✓ |
| Build-failure sentinel `i cannot mark this pipeline complete` absent | ✓ | ✓ |
| Build-failure sentinel `runtime preflight failed` absent | ✓ | ✓ |
| Model's prose surfaced as the load-bearing message | ✓ | ✓ |

Sample of the new prose (rag-text-llm RED-3, before the suffix):

> "I wasn't able to finish the workflow because the composer rejected the
> atomic setup call: it still needs explicit source and sink configuration
> fields in the pipeline state, and the current tool surfa..."

Sample (url-download RED-3):

> "I wasn't able to complete the workflow because the composer rejected
> the pipeline setup before it could be saved.
>
> What I tried:
> - Built a 3-step flow: text input holding the URL → web fetch step → li..."

vs. the pre-fix synthesized message (byte-identical across all REDs):

> "I cannot mark this pipeline complete yet because runtime preflight
> failed: 2 validation errors for ElspethSettings — source: Field
> required, sinks: Field required."

A demo viewer reading the new RED prose understands the model's intent
and the cause of failure. The pre-fix synthesizer's Pydantic noise looks
like a system bug.

## Cohort metric vs. demo-readiness — honest framing

The cohort scorer's RED→GREEN axis is bound by `must_be_valid` (state
must have `is_valid=True`) — when the model surrenders without
converging, state stays null and the verdict is RED regardless of the
message format. So the empty-state passthrough fix:

- **Eliminates one of two** red_reasons (`build-failure sentinels in
  final message`) when state is empty.
- **Preserves the other** (`final composition state is null`) because
  state is still empty.

The cohort metric therefore shows neither a clean improvement nor a
regression on the headline rate — but the *failure-mode shape* has
transformed. This is the specific kind of fix where existing scoring
underweighs the actual win: demo experience is materially better even
though the score number doesn't shift much in either direction.

## Summary verdict — Tier 1.5 §7.6 hardening

Two structural fixes in this session, both shipped to RC5-UX, both
verified on staging:

1. **Edge-contract error restructure** (`0b176800`) — `EdgeContractError`
   subclass carries `CompatibilityResult` end-to-end; preflight error
   surfaced to the LLM names producer/consumer node IDs and lists
   concrete `patch_node_options` call shapes for the fix. Verified
   neutral-or-better on the cohort (8/8 GREEN, no regression).

2. **Empty-state finalize passthrough** (`2e2c7bc8`) — when the model
   surrenders after failing to converge on a valid build, surface its
   honest prose instead of synthesizing Pydantic-noise. Verified working
   as designed on cohort (RED runs now show model self-reports, not
   system-bug-shaped text).

Combined demo-readiness profile: 80% first-pass success, 20% failures
that explain themselves clearly. Pre-fix profile was 67% first-pass
success and 33% Pydantic-noise output that looked like system bugs.

## What's NOT addressed by these fixes

The remaining ~20% RED rate is *"model couldn't converge on a valid
set_pipeline call after 20+ retries"*. This is the deeper issue and
isn't addressable by error-message work alone. Possible follow-on
investments (post-demo):

1. **Loop continuation with retry hint.** When the model produces a
   no-tool-call reply on empty state, INJECT a system hint with the
   canonical web_scrape / llm option block and let the model retry once.
   This is Option C from the earlier advisor consultation.
2. **Scaffold tool.** A new `scaffold_pipeline_from_intent` tool that
   takes a high-level user intent and produces a pipeline shell with
   sensible defaults. The model fills in the gaps.
3. **Better validation feedback per-tool.** Each `set_pipeline` /
   `upsert_node` rejection currently returns a 174-byte response (per
   audit-DB inspection). Richer per-attempt feedback might break the
   "20+ retries with adjusted options, all rejected" pattern.

None of those are on the demo critical path. Recommended for the next
cycle if the cohort metric continues to suggest a 20% floor.

## Appendix — run identifiers

- `runs/20260506T16{55..71}*emptystate-{ragtxt,urldl}-*` — post-fix cohort.
- Pre-fix baseline = post-merge cohort `runs/20260506T16{04..11}*postmerge-*`.
- Edge-contract-only cohort = `runs/20260506T16{33..38}*postfix-*`.
