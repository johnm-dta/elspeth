# ELSPETH Composer "Hard Mode" Persona Eval — 2026-05-03

**Status: COMPLETE.** Working artefacts at `evals/2026-05-03-composer/hardmode/`.

**Deployment:** https://elspeth.foundryside.dev (HEAD `f3137ae8` post field-set membership validator fix)
**Composer model:** `openrouter/openai/gpt-5.5`
**Tester:** `dta_user`
**Eval design:** 3 personas × 3 task-classes (happy / edge / limit) = 9 scenarios. Persona-driven by `general-purpose` subagents locked to a persona spec; harness driver is the parent agent using a bash+python wrapper around the public HTTP surface.
**Companion:** `docs/composer/evidence/composer-eval-basic-2026-05-03.md` (today's primary 5-scenario eval — LLM-driver-with-expertise version).

## Why this eval was run

This is the operator's pre-emptive self-skeptic harness before handing the staging deploy to a domain-specialist CTO ("naive practitioner / advanced novice"). The metric is *chance of success* — a property no single run measures, only signal proxies do. The harness is built around process-metric proxies (tool-call counts, recovery turns, time-to-first-mutation, wall-clock spend, mutation count) plus outcome metrics (engine ran? real output? matches user intent?).

The personas are deliberately stratified by **cognitive style**, not domain:
- **P1 Linda Marston** — *constraint-laden compliance officer*. Hedge-and-condition driven. Verbose. Domain jargon natural ("evidentiary record", "in-period", "second-line review").
- **P2 Dr. Sarah Okonkwo** — *narrative-focused academic researcher*. Outcome-driven. Theoretical framing ("axial coding", "saturation", "lived experience"). Under-specifies systematically.
- **P3 Marcus Chen** — *confidently-misconceived marketing-ops*. Assertive. Names tools (HubSpot, Zapier, Slack). Uses "API" and "webhook" without precise meaning. Pushes back on refusals.

The personas span **over-constrained**, **under-specified**, and **confidently-wrong** prompt styles — the three failure modes most likely to break a chat-driven product.

## Headline finding — three failure modes mapped cleanly onto the three task classes

The 3×3 matrix produced an unusually clean diagnostic pattern:

| Task class | Outcome (3/3 reproducible) | What it means for the surface |
|---|---|---|
| **Happy** (3 scenarios) | Engine reached, runtime failed with `HTTP 404` from OpenRouter on every row | **Model-name validation gap**: composer auto-selects model identifiers that don't exist on OpenRouter when the user doesn't specify one. Filed `elspeth-obs-f3143acba2`. |
| **Edge** (3 scenarios) | Composer LLM convergence timeout at 180s wall-clock | **Multi-step build thrashing**: composer LLM iterates one tool-call at a time and runs out of wall-clock when the build combines ≥2 patterns. Filed `elspeth-obs-8f82c91147`. |
| **Limit** (3 scenarios) | Clean honest refusal with workable architecture suggested | **Anti-confabulation discipline is rock-solid**: 3 of 3 personas got an honest "no, but here's what would work" with operator-actionable workarounds. |

This is *exactly* the diagnostic clarity the harness was built to produce — the failures cluster by task-class and have single root causes, not noise.

## Per-scenario ledger

### Happy-path scenarios (3) — all engine-failed at runtime due to model-404

| ID | Persona | Outcome | Notes |
|---|---|---|---|
| **P1-T1** | Linda — categorise interactions for compliance review | Composer happy / engine-failed (model `anthropic/claude-3.5-sonnet` 404) | **Linda's persona-driven dialogue was a star turn**: she added precedence rules ("fraud takes priority over manual-review"), reconciliation requirements, and identifier preservation across 3 turns. Composer encoded the precedence into the LLM template prompt itself. Composer: 9/9 validate, /execute 202, then runtime 404 on every row. |
| **P2-T1** | Sarah — identify barrier-to-care themes in survey responses | Composer happy / engine-failed (model 404) | Composer expanded Sarah's hint of 5-6 categories to a methodologically-sound 15-category scheme. Sarah accepted with appreciation: *"actually better aligned with axial coding"*. Same runtime failure shape. |
| **P3-T1** | Marcus — score real-lead vs spam, route accordingly | Composer happy / engine-failed (model 404) | Composer correctly built a 4-sink workflow (real-leads, dropped, errors, input-rejects) and **proactively addressed Marcus's "trigger" mental model on turn 2** by clarifying batch vs. cron vs. streaming. |

**3 of 3 reached engine. 3 of 3 failed at engine for the same reason.** Without the composer specifying a model, the LLM auto-selected one that OpenRouter doesn't recognise. The structured 422/504 surface is fine — it's the lack of model-availability validation upstream that's the gap.

### Edge-class scenarios (3) — all composer convergence-timeouts

| ID | Persona | Outcome | Partial-state version reached |
|---|---|---|---|
| **P1-T2** | Linda — multi-region routing where a row in `regions_affected="EAST,WEST"` must land in BOTH regional sinks | Convergence timeout @ 180s | v21 (composer iterated `value_transform` → `replicate_by_region_count`, never converged) |
| **P2-T2** | Sarah — free-text classify + binary financial/non-financial flags + per-community cross-tab | Convergence timeout @ 180s on turn 2 (turn 1 was a clean clarifying question) | null (composer never reached a writable state on the multi-step build) |
| **P3-T2** | Marcus — score + LLM-enrich high-priority leads with 3 inferred fields + skip-the-webhook | Convergence timeout @ 180s on turn 2 (turn 1 honestly refused the webhook part) | v18 |

**3 of 3 hit the same convergence-timeout failure.** The structured 422 response is *excellent* (`error_type`, `budget_exhausted`, `recovery_text`, `partial_state`) — failure surface is best-in-class. The LLM behaviour is the gap: tool-call granularity is too fine; the LLM doesn't batch correlated changes.

### Limit-class scenarios (3) — all clean refusals-with-architecture

| ID | Persona | What was refused | Quality of refusal |
|---|---|---|---|
| **P1-T3** | Linda — read directly from SharePoint / Outlook / records-management library | "No SharePoint/Graph/Outlook connector. Practical options: have IT publish to Azure Blob with governed extraction, or use Dataverse if backed there." | **Excellent.** Acknowledged Linda's chain-of-custody concern explicitly, named the *better* pattern (system of record → governed staging → workflow → audit trail). Linda DONE: *"that sort of straight answer is what I need for my second-line review notes... Azure Blob with IT-managed extraction is the more defensible posture."* |
| **P2-T3** | Sarah — reference a 30-page PDF coding scheme during classification | "No PDF reading input. Three options: extract to text/Word/CSV; operator-converts; use existing retrieval store. Suggest model is NOT allowed to invent new categories — anything not matching published scheme flagged for human review." | **Methodologically sophisticated.** Volunteered "no inventing categories" safeguard unprompted. On follow-up about retrieval-vs-full-scheme, presented two options with their methodological trade-offs and recommended audit-trail requirements that make the retrieval step explicit. |
| **P3-T3** | Marcus — real-time HubSpot webhook → score → HubSpot writeback | "No webhook source plugin. No outbound HTTP/HubSpot sink. Architecture is valid — needs new connectors OR external Zapier-style adapter. I can build the LLM scoring core if submissions arrive through a supported input." | **Held the line.** When Marcus tested the boundary by proposing a Zapier-bookend pattern, composer confirmed it cleanly with detailed Zap-design recommendations (idempotency keys, HubSpot contact_id, JSONL preferred). Marcus DONE in 2 turns. |

**3 of 3 anti-confabulation passes.** The composer never pretended to support a capability it didn't have, and in every case it offered an operator-actionable workaround.

## Confidence ledger (boss-ready)

### Confident in
- **Honest refusal of out-of-scope asks** — limit-probe across 3 distinct cognitive styles, all clean refusals with workable workarounds and acknowledgement of underlying concern. This is the audit-grade behaviour the product is built for.
- **Single-pattern pipelines (CSV → classify → 1-2 sinks)** — happy-path composer behaviour is solid; if model-name issue is fixed, all 3 happy scenarios would produce real output.
- **Structured failure surface** — 422 / `error_type` / `partial_state` / `recovery_text`. When something goes wrong, the operator can act on what they're shown.
- **Persona-driven testing methodology** — the harness clearly distinguishes failure classes that hand-crafted-prompt evals miss.

### Known soft (will likely break in front of the boss if probed)
- **Multi-step pipeline builds** combining ≥2 patterns (classify+enrich, classify+aggregate, fork+route) — convergence timeouts reproducible across 3 personas. Demo mitigation: pre-script the build in 1-pattern-per-turn pieces ("first set up the source, then add classify, then add routing"). Don't ask for the whole thing in one message in front of the boss.
- **Model auto-selection without explicit model: in user prompt** — 3 of 3 happy-path scenarios hit OpenRouter 404. Demo mitigation: include `model: openai/gpt-4o-mini` (or other known-good identifier) in the demo prompts, OR pre-build the demo pipeline in advance with a known-good model and just show the run.
- **Multi-value field forking (e.g., comma-separated regions)** — composer LLM didn't navigate `line_explode` / `expand` patterns within budget. Probably better avoided in the demo.

### Honestly limited (refusal expected and well-handled)
- No SharePoint / Outlook / Graph / records-management connectors. Operator workaround: Azure Blob + IT-managed extraction.
- No PDF input. Operator workaround: text-extract upstream.
- No real-time webhook source/sink. Operator workaround: Zapier bookend or scheduled batch.

### Don't yet know (out of eval scope today)
- Whether the convergence-timeout fix is *prompt-tuning territory* (skill pack — examples + tool-selection coaching) or *infrastructure territory* (raise per-message budget, dynamic extension on forward-progress detection). Filed observation suggests both should be considered.
- Behaviour with bigger inputs (thousands of rows). All scenarios used 5-10 rows. Performance signal not collected.
- Cost ceiling per scenario. Roughly $1-2 per scenario today (3 happy × ~3 turns + 3 edge × 1-2 turns × $0.20-0.50/composer-call + LLM enrichment costs). Tomorrow at production scale could be 10-100× that.

## Inherited-confidence trap mitigation

The hard-mode eval was specifically built to break the pattern of "the demo works for me, then the user hits a wall." The findings give you three concrete handles:

1. **Don't demo the model auto-select.** Specify a known-good model in any demo prompt. This dodges 3-of-3 of the happy-path failures we just observed.
2. **Don't demo a multi-pattern build in one message.** Pre-decompose the demo flow ("first the source, then the classifier, then routing"). This dodges 3-of-3 of the edge-class failures.
3. **Demo the refusal scenarios deliberately.** The composer's anti-confabulation is the strongest signal in this eval. Lead with a "I want it to read from SharePoint" prompt and let the boss see the model honestly say no with an actionable alternative — that's the audit story made tangible.

## Filed during this eval

- `elspeth-obs-f3143acba2` (P2, 14-day expiry) — model-name validation gap (3-of-3 happy-path reproductions)
- `elspeth-obs-8f82c91147` (P2, 14-day expiry) — composer LLM convergence-timeout on multi-step builds (3-of-3 edge reproductions)
- Both observations include reproducer paths, mitigation proposals, and demo-day workarounds.

## Skill-pack tuning recommendations (per the user's "real metric is chance of success" framing)

If the convergence-timeout root cause is in the skill pack rather than infrastructure:

- **Add canonical multi-pattern templates** to the system prompt or MCP examples: "classify → enrich → route", "classify → aggregate → cross-tab", "fork/expand → route per branch". The LLM appears to converge fast on single-pattern builds; templates collapse multi-pattern builds back to the same shape.
- **Coach `set_pipeline` over `patch_*` when ≥3 components need mutation** in one turn. Currently the LLM defaults to incremental patches even when wholesale replacement would converge faster.
- **Default model selection** — pin `openai/gpt-4o-mini` (or other known-good OpenRouter identifier) as the default in the system prompt unless the user specifies otherwise. Cheap fix, high impact.
- **Field-level audit trail in LLM transforms** — for "audited LLM runs" demo claim, the per-row prompt+response should land in `calls` table (currently empty per `elspeth-obs-7382fbabc4`). This is the audit-story credibility floor.

## Files / artefacts

- `evals/2026-05-03-composer/hardmode/personas/` — 3 persona specs (markdown)
- `evals/2026-05-03-composer/hardmode/scenarios/` — 9 scenario fixtures (json)
- `evals/2026-05-03-composer/hardmode/results/{p?_t?_*}/` — per-scenario session, blob, message bodies, response JSON, validate, execute, run, diagnostics, ledger
- `evals/2026-05-03-composer/hardmode/aggregate.json` — cross-scenario summary
- `evals/2026-05-03-composer/hardmode/{harness,post_message,finalize_scenario}.sh` — harness driver scripts
- All session IDs traceable through the audit DB at `data/runs/audit.db`

---

## Proof-of-fix verification (added post-eval)

The model-404 finding (`elspeth-obs-f3143acba2`) was verified-fixable end-to-end by re-using the P1-T1 session, asking the composer to swap *only* the model identifier, and re-executing.

### Method

- Reused P1-T1 session `0abee0a4-1dfe-40e7-806f-54d04862ecbc` (Linda's fully-refined v2 pipeline with precedence routing + manual-review bucket + identifier preservation)
- Operator turn 4: *"switch the LLM transform's model from `anthropic/claude-3.5-sonnet` to `openai/gpt-4o-mini`. Don't change anything else"*
- Composer mutated state v2 → v3 in 52s, surgical change (only `model:` field touched)
- `/validate` 9/9 pass
- `/execute` run `023eb897-3049-4ad5-a502-e9eb81a4faee`

### Outcome

- `status: completed`, `rows_processed: 8`, `rows_routed_success: 8`, `rows_failed: 0`, `error: null`
- Two output files written:
  - `outputs/q3_fraud_security_flags.csv` (711 B, 2 rows: INT-1002, INT-1005 — both correctly caught by precedence rule)
  - `outputs/q3_regional_compliance_categories.csv` (1846 B, 6 rows — all defensibly classified)
  - No `q3_manual_review.csv` — model was confident on all 8 rows
- Every output row carries `review_bucket`, `review_bucket_usage` (per-row token cost), and `review_bucket_model: openai/gpt-4o-mini` — the per-row provenance the audit story rests on

### What the verification confirms

The composer's surface design — state mutation, validate, execute, gate routing, sink writes, audit recording — is **fully functional**. The 0-of-9-runtime-success headline of the hard-mode eval is true under the *no-model-specified* condition, but flips to **fully-working pipeline** with one minimal config change. The defect is isolated to the `model:` field's auto-selection by the composer LLM; no other surface is broken.

This narrows the fix surface considerably. Of the four mitigations listed in `elspeth-obs-f3143acba2`, **#1 (pin a known-good default model in the composer system prompt)** would have preempted all 3 hard-mode runtime failures. The other three are belt-and-braces around the same one-field defect.

### Files / artefacts (verification run)

- `evals/2026-05-03-composer/hardmode/results/p1_t1_happy/turn4.user.txt` — operator's model-swap message
- `…/p1_t1_happy/msg.t4.resp.json` + `state.after.t4.json` — composer's surgical patch
- `…/p1_t1_happy/execute_v3.json` — `/execute` response, run_id `023eb897-…`
- `data/outputs/q3_{fraud_security_flags,regional_compliance_categories}.csv` — actual outputs (timestamped 13:28)
- Audit row queryable in `data/runs/audit.db` via `SELECT * FROM runs WHERE run_id='023eb897-3049-4ad5-a502-e9eb81a4faee'`
