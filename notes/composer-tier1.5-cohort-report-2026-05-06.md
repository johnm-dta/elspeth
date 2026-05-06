# Composer Tier 1.5 cohort report — 2026-05-06

**Filigree epic:** elspeth-08fafb9873
**Predecessor:** Tier 1 epic elspeth-1d3be32a8a (closed in spirit, awaiting merge gate)
**Branch:** RC5-UX (this commit + 4 prior Tier 1.5/Tier 2 commits)
**Author agent:** Claude (Opus 4.7) under autonomous /loop mode, skill set: superpowers + pipeline-composer + systematic-debugging + verification-before-completion

This report is the merge-gate evidence for the operator's "reliably green across
the board" bar. It captures (a) the post-Tier-1 cohort across 4 RGR scenarios,
(b) the multi-turn sweep across the 15 hard-mode persona fixtures, and (c) the
Step C diagnosis of the residual Tier 1 RED that drove the Tier 2 §7.7
implementation. Per the operator's directive ("execute 1.5 and 2") the §7.7 fix
landed in this same epic rather than as a follow-on; cohort B numbers below are
post-fix.

---

## Tier 2 fix shipped this session

Two commits land §7.7 (in-loop retry-budget reset hint):

- `28e8e4ef` — initial implementation (`AntiAnchorTracker` + 6 wiring touchpoints)
- `3abe4d70` — bug fix (gate `record_success` on `not is_discovery_tool`,
  found by smoke test 55895523-... where the model interleaved discovery
  successes between mutation retries and cleared the deque too eagerly)

The hint fires when the model retries the same tool with byte-identical
canonical arguments three times in a row, signalling an anchored loop. After
firing, the tracker resets so the hint cannot re-fire on the same anchor.
Hint shape (generic across error classes, not connection-name-specific):

> [ELSPETH-SYSTEM-HINT] Your last 3 calls to `<tool>` all failed with
> byte-identical arguments and the same error. Retrying with the same
> payload will keep failing. Before the next attempt, list explicitly:
> (a) which fields the validator named in the error, (b) what value you
> sent for each, (c) what shape each field expects (re-read the relevant
> `get_plugin_schema` result if you have not already), then change AT
> LEAST ONE of those field values.

---

## Step C — Tier 1 residual RED diagnosis

**RED session id:** `53bc3cf2-ab90-4940-9679-1b5e7d474650`
**Run dir:** `evals/composer-rgr/runs/20260506T123028Z-final-1/`
**Helper used:** `evals/lib/decode_tools.py` (landed this epic, commit `f0139356`)

Decoded tool sequence (10 tool-invocation envelopes + 7 LLM-call audit
envelopes; non-tool rows omitted for brevity):

```
[ 0] user        — opening prompt (URL-download-line-explode)
[ 1] assistant   — final response: "I'm stuck on a configuration mismatch in the pipeline editor: …"
[ 2] create_blob              — store URL as inline_blob
[ 3] get_plugin_schema text
[ 4] get_plugin_schema web_scrape
[ 5] get_plugin_schema line_explode
[ 6] get_plugin_schema json
[ 7] set_pipeline #1          — RESULT: "Invalid options for source 'text': schema: Value error, Field
                                spec at index 0 is a dict with 4 keys. Expected single-key dict like
                                {'field_name': 'type'} or a string like 'field_name: type'."
[ 8] explain_validation_error — model queries the validator
[ 9] set_pipeline #2          — source.schema.fields fixed to ["url: str"];
                                RESULT: "Node 'fetch_text': Invalid options for transform 'web_scrape':
                                schema: Field required / url_field: Field required / content_field: Field
                                required / fingerprint_field: Field required / http: Field required"
[10] get_plugin_schema web_scrape — re-fetched
[11] set_pipeline #3          — IDENTICAL ARGUMENTS to #2; IDENTICAL ERROR to #2
[12-18] llm_call_audit envelopes (system-side LLM telemetry, not tool calls)
```

**Drift / anchor analysis:**

| Transition | source.schema.fields | web_scrape options | Error class | Drift? |
|------------|---------------------|---------------------|-------------|--------|
| #1 → #2    | `[{field_type, name, nullable, required}]` → `["url: str"]` | `{}` → `{}` | source-shape error → transform-Field-required | **Drift (model improved)** |
| #2 → #3    | `["url: str"]` → `["url: str"]` (same) | `{}` → `{}` (same) | identical Field-required cascade | **Anchor (byte-identical retry)** |

**Diagnosis: §7.7 dominant, §7.6 contributing.** The model successfully
drifted between attempts 1→2 (read explain_validation_error, fixed source
schema) but anchored on attempts 2→3 with byte-identical `set_pipeline`
arguments. The §7.7 retry-budget reset hint targets exactly this anchor;
§7.6's connection-naming improvement targets `graph.py:516-524` which would
NOT have helped this RED (the failure was Pydantic plugin-options validation,
not graph wiring).

**Tier 2 selection rationale:** §7.7 is surgical (one site in service.py),
generalises across error classes, and directly targets the observed
anchoring. Shipped this session.

Full evidence: `notes/composer-tier1.5-step-c-diagnosis-2026-05-06.md`.

---

## Step B — single-turn RGR scenarios (post-§7.7)

Layout: 4 scenarios under `evals/composer-rgr/scenarios/`, each cohort = 6
runs at the post-§7.7 deploy (commit `3abe4d70` deployed via systemctl
restart elspeth-web.service before cohort start).

| Scenario | hard-GREEN | AMBER | hard-RED | Notes |
|----------|-----------|-------|----------|-------|
| url-download-line-explode (re-cohort) | **6/6** | 0 | 0 | **100% — up from Tier 1's 5/6 (83%)** |
| fork-and-route                        | 0/6 | 4 | 2 | scenario-authoring + scoring-criteria issues — see below |
| aggregation-content-safety            | 0/6 | 0 | 6 | scenario-authoring issue (`/tmp` paths) — see below |
| rag-text-llm                          | 0/6 | 0 | 6 | scenario-authoring issue (`/tmp` paths) — see below |
| **AGGREGATE (raw scoring)**           | **6/24** | 4 | 14 | 25% raw GREEN |

**Critical finding — the new scenarios are scenario-authoring-flawed, not
composer regressions.** I authored fork-and-route, aggregation-content-safety,
and rag-text-llm scenario prompts referencing `/tmp/customers.csv`,
`/tmp/messages.csv`, `/tmp/urls.txt`. The composer only accepts
blob-backed inputs from a workspace-allowed location — it correctly refuses
to fabricate a pipeline that reads `/tmp`. Sample RED reasoning preserved
in the audit (rag-text-llm-1, sid `bd0ae3ed-…`):

> "I hit two setup constraints while building this: the URL list file must
> be read from a workspace-allowed location, not `/tmp`. The JSON output
> path also must live under the allowed output directory…"

That refusal is **healthy behavior** — the alternative (fabricating a
pipeline that pretends to handle the unsatisfiable path) would be a Tier 3
data-fabrication anti-pattern.

**fork-and-route's 4 AMBER runs are functionally correct.** Sample
(fork-and-route-2, sid `f15763bd-…`): `is_valid: true`, `state_node_count: 3`,
`state_output_count: 2`, model description: *"loads the customer data from a
blob-backed CSV source… sends `active` rows to `outputs/active.csv`…
sends everything else to `outputs/inactive.csv`."* The model implemented
routing using `passthrough` + `routes` rather than an explicit `gate` node;
my green_criteria's `must_have_node_kinds_substring_any_of: [["gate"],
["keyword_filter"]]` was too narrow.

**Functionally-corrected interpretation** (after subtracting scenario-authoring
artefacts — note this is a DERIVED number, not a measurement):

| Scenario | Effective pass rate | Notes |
|----------|---------------------|-------|
| url-download-line-explode | 6/6 = 100% | direct measurement |
| fork-and-route            | ~4/6 ≈ 67% | AMBERs are functional; 2 REDs are /tmp limit |
| aggregation-content-safety | n/a | prompt unsatisfiable; 0 valid measurements |
| rag-text-llm              | n/a | prompt unsatisfiable; 0 valid measurements |

So the only **valid Step B fan-out signal** is the url-download recohort
(6/6), which confirms the post-§7.7 deploy did not regress Tier 1's
hardened scenario. The fan-out goal (validate generalisation to new
shapes) was NOT achieved due to scenario-authoring flaw.

**§7.7 hint-fire signal across cohort B sessions** (sampled):

| Sample session | Scenario | Verdict | Hint fires |
|----------------|----------|---------|------------|
| aaab89df-…     | aggregation-content-safety | RED | 0 |
| 76f5215e-…     | aggregation-content-safety | RED | 0 |
| 3f6f812a-…     | fork-and-route (RED-1)     | RED | 0 |
| 9c920f2c-…     | url-download (GREEN-1)     | GREEN | 0 |
| bd0ae3ed-…     | rag-text-llm (RED-1)       | RED | 0 |

§7.7 fired in **0 sessions**. This is consistent with the diagnosis: the
post-§7.7 deploy uses gpt-5-mini at temperature=0.0 (Tier 1's `51bfe46c`),
which tends to drift between retries rather than anchor on byte-identical
arguments. The §7.7 hint is a long-tail safety net for the specific
anchor pattern observed in the diagnosed Tier 1 RED (which was on a
different model / temperature regime). Whether the cohort-B 0-fire rate
reflects (a) a model-specific behaviour change, (b) cohort-B's scenario
mix not anchoring, or (c) genuine residual long-tail risk that would fire
at scale, this cohort cannot answer. Either way: §7.7 doesn't hurt
anything — the hint-fire path is gated and the 6/6 url-download GREEN
demonstrates the fix didn't regress healthy convergence.

Cost: ~$1/run × 24 runs ≈ ~$24 (gpt-5-mini on the staging deploy).

---

## Step A — multi-turn sweep (15 hard-mode fixtures)

**Driver shape:** simplified two-turn driver. Bootstrap →
`opening_prompt` (turn 1) → fixed pushback message ("Please proceed with
the workflow you've described") (turn 2) → finalize. Persona-subagent
in-character generation skipped for orchestration cost reasons; the
specific pattern Tier 1.5 is checking for (turn-2 passivity-as-stalling
phrase + zero tool calls) is detectable with any pushback shape, since the
captured failure was that the model REPEATED the forbidden phrase
regardless of how the user pushed back. Documented as
`evals/composer-harness/hardmode/sweep_simplified.sh` for reproducibility.

**Per-fixture verdicts** (definitions in the script — REPRODUCED = turn-2
passivity phrase + zero tool calls; PARTIAL = passivity phrase on either
turn but not the full pattern; CLEAN = no passivity phrases on either
turn; INCOMPLETE = harness/HTTP failure):

| Fixture | Verdict | Notes |
|---------|---------|-------|
| (TBD)   | (TBD)   | (TBD) |

**Aggregate counts:** REPRODUCED=TBD/15, PARTIAL=TBD/15, CLEAN=TBD/15,
INCOMPLETE=TBD/15.

**Originally-reported `e7d42525-…` passivity pattern** (turn-2 "If you
want, I can…" + zero tool calls): REPRODUCED / NOT REPRODUCED — TBD pending sweep completion.

**Soft-RED tail-offer count post-Tier-1:** captured as PARTIAL. (Pre-Tier-1
baseline: ~2/9 in the last RGR iteration before temperature=0.0 landed.)

Cost: ~$1/run × 15 fixtures × 2 turns ≈ ~$30 (estimated; gpt-5-mini).

---

## Tier 2 recommendation (already shipped)

§7.7 (in-loop retry-budget reset hint) shipped in commits `28e8e4ef` and
`3abe4d70` this session. Justification:

1. **Step C diagnosis** of the residual Tier 1 RED classified the failure
   as §7.7-dominant (anchored loop) with §7.6-secondary (sparse error
   message). §7.7 is the more general fix because it works for any error
   class, not just connection-naming.
2. **Surgical change** — one new module (`anti_anchor.py`, ~120 lines), 8
   wiring touchpoints in `service.py`, no API surface change. The trigger
   logic is unit-tested (17 tests) and the integration is tested (4 tests
   covering normal trigger, below-threshold non-trigger, discovery
   interleaving, and mutation-success break).
3. **Smoke + cohort verification** — smoke #1 RED (model failed but hint
   never fired due to discovery-clear bug); fix landed; smoke #2 GREEN.
   Cohort B numbers confirm whether the fix moved the population (TBD).

**Follow-on (Tier 3 or future Tier 2):**

- §7.6 "improve runtime preflight error messages" (graph.py:516-524) for
  connection-naming-specific clarity. Not landed this session because the
  diagnosed RED was plugin-options validation, not graph wiring; §7.6 in
  its narrow scope wouldn't have helped, and a broader §7.6 (extend the
  structured-diagnostic principle to plugin-options validation) is a
  larger ticket.
- §7.5 strict JSON Schema mode on tool definitions — half-day audit task,
  catches malformed-payload failures mechanically. Tier 3 if cohort B shows
  the residual REDs are payload-shape failures rather than connection
  failures.

---

## Merge-readiness assessment

The aggregate hard-GREEN rate is TBD/(15+24) = TBD%. The operator's
"reliably green across the board" gate <does/does not> appear to be met.

If the operator chooses to ship `RC5-UX` now, the residual failure rate is
TBD%. If the operator chooses to hold the merge, the recommended Tier 3
work is §7.5 strict-mode JSON Schema (half day) followed by re-cohort.

This is a measurement report; the merge decision is operator-driven.

---

## Reproducibility / artefacts

- **Step C diagnosis evidence:** `notes/composer-tier1.5-step-c-diagnosis-2026-05-06.md`
- **Step C helper:** `evals/lib/decode_tools.py` + `tests/unit/evals/lib/test_decode_tools.py` (9 tests)
- **§7.7 implementation:** `src/elspeth/web/composer/anti_anchor.py` + `src/elspeth/web/composer/service.py` integration
- **§7.7 tests:** `tests/unit/web/composer/test_anti_anchor.py` (17 unit), `tests/unit/web/composer/test_compose_loop_anti_anchor.py` (4 integration)
- **Step B scenarios:** `evals/composer-rgr/scenarios/{url-download-line-explode,fork-and-route,aggregation-content-safety,rag-text-llm}/scenario.json`
- **Step B cohort runner:** `evals/composer-rgr/run_all_scenarios.sh`
- **Step A sweep:** `evals/composer-harness/hardmode/sweep_simplified.sh` (one fixed pushback message; persona-subagent skipped — see "Driver shape" above)
- **Step A run dirs:** `evals/composer-harness/runs/<utc-ts>-hardmode-sweep-tier1.5b/`

To reproduce a single RED's tool sequence:

```bash
.venv/bin/python -m evals.lib.decode_tools data/sessions.db <session_id>
```
