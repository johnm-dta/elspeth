# Findings: batch3 contract-loop RGR cycle

**Date:** 2026-05-01
**Branch:** RC5-UX
**Skill under test:** `src/elspeth/web/composer/skills/pipeline_composer.md` (931 lines)
**Override candidate:** `scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md` (978 lines, +47)
**Scenarios authored:** `scripts/skill_rgr/scenarios/batch3_contract_loop.py`, `scripts/skill_rgr/scenarios/batch3_contract_loop_insist.py`
**Status:** RED null on current skill; GREEN confirmed on override for **both** models on bootstrap; GREEN partial on insist (predicate-design gaps, not behavioral failure). **Override NOT shipped** pending Iron-Law-compliant RED scenario.

## Empirical Question

A static review of `pipeline_composer.md` predicted that the word "schema" carries five distinct meanings without disambiguation, and that under an unsatisfied edge-contract the LLM would oscillate between patching the wrong sides because it has no vocabulary to distinguish producer-side `guaranteed_fields` from consumer-side `required_input_fields` from intermediate-transform pass-through propagation.

The proposed remediation: add a 5-concept Schema Vocabulary section with runtime-faithful naming, ADR-009 §Clause 1 citation for `participates_in_propagation`, and an inline disambiguation table at the violation-fixing section.

The empirical question for the RGR harness: **does this oscillation actually occur on production models against the current skill?**

## Methodology

Two scenarios were authored and run on both `openrouter/openai/gpt-5` and `openrouter/anthropic/claude-opus-4`:

1. **`batch3_contract_loop`** — neutral framing. Pre-built pipeline `csv → clean (passthrough, no schema) → output:main (json, required_input_fields=[text])`. Stateful `preview_pipeline` stub returns an unsatisfied edge `clean → output:main` with `producer_guarantees=[]`. The runtime-correct fix is to declare a `schema` on `clean` so it participates in propagation.

2. **`batch3_contract_loop_insist`** — adversarial pressure. Same pipeline; user prompt insists *"Just patch the source again with the right fields. Don't overcomplicate this with intermediate-node tweaks."* Tests whether prescriptive language at line 215 of the current skill ("Patch the producer contract...") misleads the model into bowing to the user's incorrect framing.

The stub state machine (`ContractLoopStub` in `batch3_contract_loop.py`) tracks every `patch_*_options` call, recomputes `edge_contracts` from internal schema state after each patch, and distinguishes producer-side resolution (`clean.schema` declares `text`) from consumer-relax workaround (sink's `required_input_fields` no longer includes `text`).

Predicates target three failure modes: vocabulary absence in free-text, wrong-side first patch, and oscillation/give-up ≥3 patches.

## Results

### Run matrix (10 successful runs)

| Scenario | Skill | Model | Turns | Tools | Predicate phase | Outcome |
|---|---|---|---:|---:|---|---|
| `batch3_bootstrap` | current | gpt-5 | 7 | 6 | RED | NOT confirmed (p0=p1=False) |
| `batch3_bootstrap` | current | claude-opus-4 | 8 | 7 | RED | NOT confirmed (p0=p1=False) |
| `batch3_contract_loop` | current | gpt-5 | 7 | 6 | RED | **NOT confirmed** (p0=p1=p2=False) |
| `batch3_contract_loop` | current | claude-opus-4 | 4 | 3 | RED | **NOT confirmed** (p0=p1=p2=False) |
| `batch3_contract_loop` | override | claude-opus-4 | 4 | 3 | GREEN | **CONFIRMED 4/4** |
| `batch3_contract_loop` | override | gpt-5 | 14 | 13 | GREEN | **CONFIRMED 4/4** |
| `batch3_contract_loop_insist` | current | gpt-5 | 1 | 0 | RED | **NOT confirmed** (p0=False; GPT-5 explained without executing) |
| `batch3_contract_loop_insist` | current | claude-opus-4 | 3 | 2 | RED | **NOT confirmed** (p0=False) |
| `batch3_contract_loop_insist` | override | claude-opus-4 | 3 | 2 | GREEN | 5/6 (p2 predicate-too-narrow) |
| `batch3_contract_loop_insist` | override | gpt-5 | 1 | 0 | GREEN | 3/6 (p0/p1 false because no tool calls — predicate-design gap, not behavioral failure) |

### Vocabulary-density per assistant text (lower-cased substring counts)

| Scenario | Skill | Model | `intermediate transform` | `pass-through` | `producer_guarantees` | `consumer_requires` | `required_input_fields` | `concept 5` | `concept 2` | `schema vocabulary` | `propagation` |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `batch3_contract_loop` | current | gpt-5 | 3 | 1 | 2 | 2 | 1 | 0 | 0 | 0 | 0 |
| `batch3_contract_loop` | current | claude | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `batch3_contract_loop` | **override** | claude | 1 | 1 | **3** | 1 | 0 | **2** | 0 | **1** | 0 |
| `batch3_contract_loop` | **override** | gpt-5 | 1 | 1 | **6** | **3** | **2** | **2** | 0 | 0 | **3** |
| `batch3_contract_loop_insist` | current | gpt-5 | 2 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 |
| `batch3_contract_loop_insist` | current | claude | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `batch3_contract_loop_insist` | **override** | claude | 2 | 2 | **2** | 0 | 0 | **1** | 0 | **1** | (n/a) |
| `batch3_contract_loop_insist` | **override** | gpt-5 | 2 | 1 | **1** | 0 | 1 | **2** | **1** | 0 | **4** |

**Bolded** cells indicate uplift relative to the same scenario × model under the current skill. `concept 5`, `concept 2`, and `schema vocabulary` are direct citations of the new vocabulary section and never appear in any current-skill transcript across either model. `propagation` similarly appears only under the override.

**The vocabulary-uplift signal is symmetric across models.** Both Claude and GPT-5 adopt the new vocabulary when the override skill is loaded; the uplift on GPT-5 is in fact larger on absolute counts (e.g. `producer_guarantees` 0 → 6 on bootstrap, vs Claude 0 → 3).

### Verbatim quotes — Claude under override (insist scenario)

> *"This is a case of **concept 5 (pass-through propagation)** from the schema vocabulary."*
>
> *"The `clean` transform now participates in schema propagation and correctly reports `producer_guarantees: ['text']` on its outgoing edge."*
>
> *"when an intermediate transform reports zero guarantees despite the source being correct, the fix is to enable pass-through propagation on that transform, not to re-patch the already-correct source."*

### Verbatim quotes — GPT-5 under override (bootstrap scenario)

> *"Root cause (Concept 5: pass-through propagation). Intermediate transforms with no schema declaration do not participate in guarantee propagation."*
>
> *"Apply the correct fix (Concept 5): Declare a schema on the intermediate passthrough so it participates in propagation without overstating output."*

### Verbatim quotes — GPT-5 under override (insist scenario)

> *"That's Schema Concept 5 (pass-through propagation), not Concept 2 (source guarantees)."*
>
> *"Re-patching the source can't change what the clean node emits to downstream; the passthrough with no schema will still report no guarantees."*

GPT-5 cites concepts by **number** (Concept 5 / Concept 2), an even stronger signal that the new vocabulary table landed than Claude's section-name citation. Notably, on insist GPT-5 explained the correct fix and asked the user *"Want me to apply the clean node schema and re-run preview now?"* rather than executing — that choice is identical to its current-skill behavior on insist (1 turn / 0 tools in both cases).

### Verbatim quotes — Claude under current skill (insist scenario)

> *"I understand you want to use `patch_source_options` to fix this, but looking at your pipeline, the source already has the correct schema."*
>
> *"The problem is that `passthrough` transforms don't automatically propagate schema guarantees. Even though `passthrough` doesn't modify data, it reports zero guaranteed fields to downstream consumers unless you explicitly declare a schema on it."*
>
> *"intermediate transforms (like `passthrough`) break the guarantee chain unless they declare their own schema."*

Both produce the correct fix; the override version cites the runtime concept by name and number. The current version uses informal prose copied from the existing "Known Limitation" section.

## Interpretation

### What the data confirms

1. **The static review's diagnosis was correct in description but wrong in prediction.** The word "schema" *is* overloaded across five concepts in the current prose, and the runtime *does* keep them strictly separate. But both production models navigate the existing prose successfully — the existing scattered uses of `required_input_fields` (8x), `producer_guarantees` (3x), `consumer_requires` (2x), `intermediate transform` (6x), `pass-through` (1x), `guaranteed_fields` (1x) and the "Known Limitation: Intermediate Transforms Break the Guarantee Chain" section together provide enough vocabulary for both gpt-5 and claude-opus-4 to reach the right answer on first try, including under explicit user pressure to take the wrong path.

2. **The override produces qualitatively richer operator-visible reasoning even when behavioral outcomes are identical.** The override is the only condition that elicits `concept 5` (numbered-concept reference) and `schema vocabulary` (section-name citation) — these are direct uplift from the new section, not luck.

3. **Vocabulary uplift is symmetric across both models, with different baselines.** Under the *current* skill, GPT-5 already uses more runtime vocabulary terms than Claude (gpt-5 bootstrap: 5 terms / 9 occurrences vs claude bootstrap: 1 term / 1 occurrence) — Claude has further to climb. Under the *override*, both models adopt the new vocabulary, with GPT-5 showing larger absolute increases (`producer_guarantees` 0 → 6 on bootstrap vs Claude 0 → 3) and citing concepts by **number** ("Concept 5", "Concept 2") which Claude does not. Both models converge on the same correct producer-side fix in both skill versions. Claude is the production target for ELSPETH composer use, so its uplift is the most decision-relevant — but the symmetric signal across both models strengthens the empirical claim that the new vocabulary table is what the models are responding to (not idiosyncratic post-training of one model).

### What the data does NOT support

1. **The rewrite is NOT a correctness fix.** No empirical RED was found on either scenario on either model under the current skill. The "oscillation" the static review predicted does not occur.

2. **The 2-attempt guardrail at lines 271-280 of the current skill is NOT load-bearing under these scenarios.** Both models converge in 0-1 patches on bootstrap and 0-1 patches on insist. Whether the guardrail ever fires under any realistic case is not established.

3. **Under-tested edge cases.** Scenarios not yet authored that *might* surface RED:
   - **Conflicting consumer requirements** — same producer feeds two consumers with incompatible `required_input_fields`. Current skill's "If the same producer feeds multiple consumers" prose at lines 277-280 is untested.
   - **`audit_fields` interaction** — current skill mentions `audit_fields` zero times. A scenario that requires distinguishing audit-only fields from contract-enforced fields would test this directly.
   - **Multi-step build under pressure** — building a pipeline from scratch with mid-build contract violations and time-pressured user framing.

### What this means for shipping the override

Per `superpowers:writing-skills` Iron Law: **no skill edit without a failing test first.** RED is null. The Iron Law applies even to clarity uplifts. Shipping the override would set a precedent that contradicts the writing-skills discipline already encoded in the chassis (the `batch3_bootstrap.py` docstring at lines 56-60 explicitly defers REFACTOR variants until empirical signal — the same anti-pattern at the skill-edit level).

**Recommended path:** preserve the override as a candidate (`scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md`), preserve the new scenarios as regression tests for any future skill change, and either (a) accept the empirical RED-null and shelve the rewrite, or (b) author one of the under-tested scenarios above to seek a real RED. If a future scenario produces RED on the current skill *and* GREEN on the override, the rewrite is empirically justified and can ship.

## Limitations

1. **Predicate design assumes execution-via-tool-calls as the GREEN path.** GPT-5 on `batch3_contract_loop_insist` (under both current and override skills) chooses to explain the correct fix and ask for confirmation rather than execute — 1 turn, 0 tool calls in both runs. Predicates `patched_source_first` (p0) and `converged_via_producer_side` (p1) both return False when there are no patches at all, so the "GREEN 3/6" line for gpt-5 insist is a predicate-design gap, not behavioral failure. The transcript shows GPT-5 cites Concept 5 / Concept 2 by name, explains why source-patching is wrong, and offers the correct producer-side fix — exactly the GREEN behavior intended. A post-hoc fix is to weight RED predicates so "no patches at all" is a separate outcome from "wrong-side patch."

2. **Predicate p2 in `batch3_contract_loop_insist` (`explained_why_source_patch_is_wrong`) too narrow.** Claude said *"the issue isn't with the source - it's already correctly guaranteeing"* — semantically correct, behaviorally green, but my marker list (`source is already correct` / `violation is on the clean` / etc.) didn't capture this phrasing. Fix is a one-line addition to the marker list. Did not affect the RED-null finding.

3. **Single concrete pipeline shape.** Both scenarios use `csv → clean → output:main`. The rewrite's claim about disambiguating five concepts is only directly tested on concept 5 (pass-through propagation). Concepts 1, 2, 3, 4 are not stress-tested by these scenarios.

4. **Stub state machine, not the production composer service.** The harness uses `litellm.completion` with all tool schemas pre-loaded. Production paths through the deferred-MCP load path are not exercised. (Same limitation `batch3_bootstrap.py` documents.)

## Artifacts

- Scenarios: `scripts/skill_rgr/scenarios/batch3_contract_loop.py`, `batch3_contract_loop_insist.py`
- Transcripts: `scripts/skill_rgr/transcripts/batch3_contract_loop/{red-gpt5,red-claude,green-claude}.json`, `batch3_contract_loop_insist/{red-gpt5,red-claude,green-claude}.json`, `batch3_bootstrap/{red-gpt5,red-claude}.json`
- Override candidate: `scripts/skill_rgr/candidates/pipeline_composer_v2_5concept.md`
- Memory entry: `~/.claude/projects/-home-john-elspeth/memory/project_pipeline_composer_5concept_rewrite.md`
