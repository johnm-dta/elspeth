# Composer Prompt A/B Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or subagent-driven-development) to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **STANDBY:** Another agent is implementing a composer fix. Phase 0 (drafting + harness build against standalone files) is safe now. Phases 1+ (swapping the **live** core skill) MUST NOT start until that fix lands and we rebase variants onto the post-fix baseline. Do not edit `src/elspeth/web/composer/**` while the other agent is active.

**Goal:** Determine experimentally whether restructuring the composer system prompt (not changing the model) lifts the cheap model's (`gpt-5.4-mini`) reliability — and *which* structural lever does it — by A/B testing drafted prompt variants on the existing tutorial-reliability harness.

**Architecture:** Hold the model constant at `gpt-5.4-mini`; vary only the core skill file. Two-tier measurement: a fast high-N **compose-only probe** for the primary metric (vague_term recall — the prompt-navigation class) and the existing **full tutorial battery** (low-N) for end-to-end confirmation + the execution-validation class (the reasoning control). Pre-registered differential prediction so the result is informative whichever way it falls.

**Tech Stack:** Python (httpx probe script), the existing Playwright battery (`tests/e2e/tutorial-reliability.staging.spec.ts`), staging at `elspeth.foundryside.dev` (systemd `elspeth-web.service`), Landscape MCP for run inspection.

---

## Background (why this experiment exists)

The 2026-06-07 investigation (see `notes/web-static-analysis-*` and the session record) established:

- The mini receives a **~12,200-token / 1,078-line** system prompt — ~18–47× over Anthropic's skill-length guidance and the 2026 "question anything over 300 words" bar.
- **Verified structural defects:** vague_term guidance sits at the dead-center attention zone (lines 563–755; lost-in-the-middle → >30% accuracy drop per Chroma 2026); the prompt contradicts its own "no held reference material" principle (lines 6–9 vs the 35-line Mechanical Repairs table); the "never surface llm_prompt_template" rule is repeated 3×; two competing wiring mechanisms are taught (`prompt_template_parts` + the legacy `{{interpretation:}}`); and `<!-- SUPPRESSED -->` blocks reach the model (confirmed: `prompts.py` strips only `<!-- ADVISOR-* -->`).
- **Attribution split (the load-bearing frame):** the opus-vs-mini battery held the bad prompt *constant*, conflating capability with prompt-navigability. Re-attributed: vague_term under-fire (2/6) is a **prompt-navigation** failure (rule present, didn't fire → should respond to restructuring); execution-validation (3/6) is a **reasoning** failure (mini builds invalid pipelines → restructuring won't fix; owned by the deterministic field-contract guard in `elspeth-dac6602a2b`).

This experiment isolates the prompt-navigation lever for the cheap path. It is **complementary** to the tutorial-scoped-model decision and to the advisor-checkpoints design (`elspeth-dac6602a2b`), not a substitute.

## Pre-registered hypotheses (declare before running — makes the result informative either way)

- **H1 (primary):** Variants A/B/C lift vague_term recall on the mini vs the V0 control (baseline ~67%, i.e. 2/6 under-flag), toward ≥90%.
- **H2 (control prediction):** Execution-validation failure rate stays ~flat across variants (reasoning-bound; confirms it belongs to `dac6602a2b`, not the prompt). If it *also* drops, that is a bonus to re-attribute, not the expected result.
- **H3 (polarity guard):** Over-fire (vague_term staged on an *objective*, user-supplied criterion) stays ~0 across all variants. A variant that lifts recall by over-firing is a FAIL (operator doctrine: the residual miss must be a false-negative, never a false-positive — see `feedback_vague_term_is_advisory_not_blocking`).
- **H4 (lower-bound probe):** The minimal variant (D) reveals whether the mini needs the detail (Anthropic: "what works for Opus may need *more* detail for Haiku"). If D regresses vs B, detail matters → don't over-cut.

## File Structure

**Drafted variants (Phase 0 — standalone, NOT the live file):**
- ✓ DONE: `notes/prompt-experiment/variant-A-tightened.md` (~290 lines, L1 conciseness, order preserved)
- ✓ DONE: `notes/prompt-experiment/variant-B-restructured.md` (~300 lines, L1+L2; the Operating Contract is its opening section)
- Phase-1 design only (the "if needed" arms): `variant-C-fewshot.md`, `variant-D-minimal.md` — finalize against the post-fix baseline if A/B warrant.

**Harness (Phase 1):**
- Create: `scripts/experiments/compose_probe.py` — Tier-1 fast compose-only probe (httpx, no browser, no execution).
- Create: `scripts/experiments/run_variant_matrix.sh` — variant runner: backup → swap core skill → restart → hash-verify → probe → restore.
- Reuse: `tests/e2e/tutorial-reliability.staging.spec.ts` — Tier-2 full battery (existing).
- Results (gitignored): `notes/prompt-experiment/results/<variant>-<tier>.jsonl`

**Swap target (verified):** `src/elspeth/web/composer/skills/pipeline_composer.md` (the core `_PIPELINE_SKILL`). The deployment overlay is *appended*, so it cannot replace the core — the core file is the only correct swap point.

---

## Variant designs (the experimental arms)

Each variant is a **full static replacement** of the core skill, isolating one cumulative lever. Levers: **L1** conciseness (dedupe/cut/dead-content), **L2** placement (critical-first, vague_term out of the middle, checklist-last), **L3** demonstration (few-shot), **L4** minimalism (lower-bound probe).

| Variant | Levers | Transformation recipe | Target size |
| --- | --- | --- | --- |
| **V0 Control** | — | Current core skill verbatim. Re-run in-session (the original 3/6 was a different day → controls for staging drift). | 1,078 ln |
| **A — Tightened** | L1 | Same section order & coverage. Dedupe each repeated rule to ONE canonical statement; delete all `<!-- SUPPRESSED -->` blocks; drop the legacy `{{interpretation:}}` option (keep `prompt_template_parts` only); replace the Mechanical Repairs table with "use `explain_validation_error` / `get_plugin_assistance`". | ~55–60% |
| **B — Restructured** | L1+L2 | A's content, reordered: lead with the **Operating Contract** (below) carrying the non-negotiable rules incl. the vague_term self-audit *trigger*; build mechanics in the body; a copy-this **termination checklist** at the end. vague_term *trigger* hoisted to top tier; wiring *detail* stays in body. | ~55–60% |
| **C — Few-shot** | L1+L2+L3 | B + two compact worked examples at a high-attention position: (1) a correctly staged+wired vague_term for the canonical "how cool" scoring task; (2) an *objective* criterion (user supplies the cutoff) that correctly does NOT stage vague_term (over-fire guard). | ~+0.5k tok vs B |
| **D — Minimal** (opt) | L4 | Aggressive cut to ~30%; high-freedom; keep only core triggers + the single wiring example; strip edge-case nuance. Lower-bound / Anthropic-caveat probe. | ~30% |

**Isolation logic:** V0→A measures signal-to-noise alone (order held). A→B measures placement. B→C measures demonstration. D probes the opposite extreme. Run A/B first; C/D only if A/B are promising/inconclusive (matches "A and B, C/D if needed").

**Note on rule 3 (field reconciliation) in the Operating Contract:** included for completeness, but per the attribution split it is belt, not suspenders — the execution-validation class is reasoning-bound and owned by the `dac6602a2b` deterministic guard. Do not expect the prompt rule alone to fix it (that is exactly H2).

---

## Drafted artifact: the shared Operating Contract (top section for B/C/D)

This is the high-attention opener that carries the load-bearing rules out of the dead-center zone. ~230 words. Finalize against the post-fix baseline in Phase 1.

```markdown
# Pipeline Composer — Operating Contract

You build ELSPETH pipelines from the user's request using live tools. The audit
trail is the legal record: every authored decision must be explicit, reviewable,
and backed by tool output.

**Four rules that override convenience:**
1. **Build the requested shape.** Never drop a requested source/transform/sink/
   LLM/cleanup step to pass validation. A smaller pipeline that omits requested
   behaviour is a silent downgrade — repair the node, or refuse with a named gap.
2. **Stage a `vague_term` review whenever you author judgement.** If you chose a
   scale, threshold, category boundary, weighting, or *how* to operationalise a
   subjective user criterion, that authored rule is reviewable. Stage
   `kind="vague_term"` on the LLM node AND wire it into the prompt via a
   `prompt_template_parts` `interpretation_ref` slot, in the same `set_pipeline`.
   (Authorship, not vocabulary — do not scan for "magic words".)
3. **Reconcile fields end-to-end.** Every field a node requires must be produced
   by an upstream node. Before `set_pipeline`, check each consumer's
   `required_input_fields` against what the source and transforms actually emit.
4. **Never surface `llm_prompt_template`.** The backend auto-stages and surfaces
   it; requesting it is rejected.

**Done means** exactly one terminal state: a valid preview; OR all required review
cards surfaced with no other validation errors; OR a named-gap refusal. A
successful mutation is NOT "done".

Mechanics for each step follow. Load plugin schemas, repair prose, and recipes
from the live tools — not from memory.
```

---

## Harness adaptation

### Tier 1 — Compose-only probe (PRIMARY, high-N, fast)

Measures the prompt-navigation class directly, decoupled from execution/reasoning.

- **Drive:** a freeform compose (new session → send the canonical scoring message → poll until the compose turn finalizes). No browser, no execution → ~1–2 min/run → **N≥20 per variant** feasible (real power on recall; N=6 is a screen, not a proof).
- **Two canonical prompts per run set:**
  - *Subjective:* "create a list of 5 government web pages and use an LLM to rate how cool they are" (must stage vague_term → recall).
  - *Objective:* a coherent same-domain criterion with a user-supplied rule, e.g. "create a list of 5 government web pages and rank them by page-load time in milliseconds" (or "...filter to pages updated after 2024"). Must NOT stage vague_term → over-fire guard, H3. (Do not use an incoherent criterion like "rate pages taller than 190cm" — it contaminates the metric.)
- **Capture** from `GET /api/sessions/{id}/interpretations` (or state): vague_term staged? wired via `interpretation_ref`? over-fire? `llm_prompt_template` present (backend auto)? Record per-run, tagged by variant + `composer_skill_hash`.

### Tier 2 — Full tutorial battery (SECONDARY, low-N, end-to-end)

The existing Playwright battery on the **top 1–2 variants only** (expensive). Confirms no e2e regression + measures graduation, execution-validation (H2), invented-source. N=6–10.

### Variant runner mechanics (`run_variant_matrix.sh`)

For each variant, in order:
1. `cp` core skill to a backup path (once, at start).
2. `cp notes/prompt-experiment/variant-X.md` → `src/elspeth/web/composer/skills/pipeline_composer.md`.
3. Pin model: confirm `ELSPETH_WEB__COMPOSER_MODEL=gpt-5.4-mini` in `deploy/elspeth-web.env`.
4. `sudo systemctl restart elspeth-web.service`; wait for `Application startup complete` + `curl` 200.
5. **Hash-verify the swap took:** run one probe, read `composer_skill_hash` off a staged interpretation event; assert it differs from the previous variant. (If it didn't change, the lru_cache/restart failed — abort.)
6. Run Tier-1 probe N times; append results to `results/<variant>-tier1.jsonl`.
7. Next variant. At the end, **restore** the core skill from backup + restart.

### Controls / threats to validity

- **Model constant** (mini) — the whole point. Hold `advisor_enabled` constant too (it changes the effective post-strip prompt).
- **Staging-load noise (resolved — don't double-mitigate):** module-level `lru_cache` makes each variant switch cost a restart, so naive per-run round-robin would mean 60–80 restarts. Instead run **block-by-variant but repeat the whole matrix in N rounds** (A,B,C,D → A,B,C,D → …): ~4 restarts/round, and time-of-day load is spread evenly across all arms. (Alternative: accept block-by-variant at known low-load hours — pick ONE, not both.) Classify `infra_fault`/timeout separately and exclude from the recall denominator (or re-run that cell).
- **Cache masking (critical):** the canonical demo can be cache-served (never composes). The probe MUST force real composition each run — fresh session, cache bypassed. Verify the first probe actually invokes the mini (LLM call count > 0 in audit) before trusting any results.
- **Proxy validity (GATES belief in Phase 2):** the vague_term under-fire was observed in the *tutorial* path, but Tier-1 measures *freeform* compose. Before trusting any Tier-1 ranking, confirm the tutorial's *effective* system prompt is the bare core skill — not core + a tutorial preamble injected by `tutorial_service`. If the tutorial wraps/prepends the core skill, then (a) the effective prompt is longer than 12k tokens and (b) vague_term's lost-in-the-middle *position shifts*, so freeform measures a structurally different prompt and the ranking may not transfer. One-grep Phase-1 check (`grep` how `tutorial_service` builds its system prompt vs `build_system_prompt`). If they diverge, run Tier-1 through the *tutorial* compose path instead of freeform. This does not block drafting or the plan; it blocks believing Phase-2 numbers.
- **Statistical honesty:** report per-class rates; treat as a directional ranking with N≥20 on the primary metric; state the power limit explicitly. No silent truncation of failed runs.
- **Audit churn:** each variant = new `composer_skill_hash` (expected for an experiment). Do NOT commit variants; do NOT sign anything; restore baseline when done.

---

## Decision criteria

- **B or C lifts recall to ≥90% with over-fire ~0** → the restructure is the cost-conscious lever. Promote the winner; Tier-2 confirm; ship as the mini prompt; feed the tutorial-scoped-model decision (maybe the mini becomes viable).
- **No static variant moves recall materially** → vague_term is more reasoning-bound than navigation-bound → lean on the model lever + advisor checkpoints. Prompt cleanup still worth it for cost/latency/clarity, but not the recall fix.
- **Execution-validation also drops (unexpected)** → re-attribute; bonus for the cheap path.
- **D regresses vs B** → the mini needs the detail; restructure ≠ truncate.

---

## Execution sequence (phases)

- **Phase 0 (now, safe during standby):** this plan + draft variants A/B (C/D if reached) as standalone files + the Operating Contract. No live-file edits.
- **Phase 1 (after the other agent's fix lands):** rebase variants onto the post-fix baseline (if the fix is itself a prompt change, it becomes the new V0/control and variants layer on it). Verify skill-load semantics still hold. **Proxy-validity gate:** confirm the tutorial's effective prompt = bare core skill (else route Tier-1 through the tutorial path) — see Controls. Build `compose_probe.py` + `run_variant_matrix.sh`. Smoke-test the cache-bypass + hash-verify.
- **Phase 2:** run Tier-1 across V0/A/B (high-N) → rank. Add C/D if warranted.
- **Phase 3:** Tier-2 confirm on the winner(s).
- **Phase 4:** promote winner; document results; hand off to the model-strategy thread + `elspeth-dac6602a2b`.

---

## Self-Review

- **Coverage:** every identified lever (L1–L4) maps to a variant; both failure classes have a metric; the polarity guard (over-fire) is explicit. ✓
- **Placeholders:** the Operating Contract is fully drafted; full variant files are Phase-1 tasks against the post-fix baseline (deliberate — drafting 4 full rewrites now would be discarded on rebase). Recipes are concrete enough to execute mechanically.
- **Consistency:** swap target, restart requirement, and hash-verify are consistent across the runner and the threats section. Model pinned to `gpt-5.4-mini` throughout.
- **Standby respected:** no live `src/elspeth/web/composer/**` edits in Phase 0; variants live under `notes/`.
```
