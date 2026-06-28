# Tutorial Reliability Harness — Design

- **Date:** 2026-06-06
- **Status:** Design (approved in brainstorm; pending written-spec review)
- **Target system:** staging — `https://elspeth.foundryside.dev`, account `dta_user`
- **Author:** orientation + brainstorm session

## 1. Purpose & foundational principle

Build a **repeatable regression harness** that runs the first-run composer
tutorial end-to-end against the live system, classifies how it fails, and
reports how often it actually worked. The immediate ask is a batch of **10
runs (reset between each)**, but the harness is built for an **iterative loop
that may run hundreds or thousands of times** while we refine composer prompts
and tools toward rock-solid.

**Foundational principle — we cannot fake the tutorial.** The moment the
tutorial ends, the user tries to run a real pipeline in the real system. If
that fails, the tutorial was a lie and we have lost them. Therefore every
"pass" the harness reports must mean *this would actually work in the product*,
exercised through the genuine path: real composition, real LLM, real scraping,
real run.

## 2. Pass criterion (four dimensions)

A run **worked** iff **all four** hold:

- **(a) Tutorial path completes.** The real React tutorial reaches graduation
  (turn 7) driving the live backend — no stuck turn, no render crash; the
  tutorial run reports success with rows and a coherent audit story.
- **(b) Real-system re-run passes.** The *same composed pipeline* — the
  composer's actual output, **not** the tutorial-normalized version — executes
  through the normal `/api/sessions/{sid}/execute` path and completes
  successfully with rows.
- **(c) Assumption handling is correct.** The composer flagged the assumptions
  that genuinely need verification and waved through the ones that don't,
  judged against a fixed per-prompt rubric (§5; deterministic grading).
- **(d) The puzzle is creatively solved.** The composer produced a
  *substantively correct, non-degenerate* solution: invented sources that are
  real and reachable, an extraction that actually pulls the requested attribute,
  and output rows with meaningful values — creative without fabricating
  (§6; LLM-judge grading).

(b) is asserted **separately** from (a) because of a real fidelity gap: the
tutorial calls `_normalise_current_tutorial_state_for_execution`
(`src/elspeth/web/composer/tutorial_service.py:147`) which silently rewrites
the composed pipeline before executing (bare `{{ field }}` → `{{ row.field }}`
in LLM prompt templates; strips interpretation placeholders). The real execute
path does **not** apply that fixup. So a tutorial pass does **not** prove a
real-system pass — and the gap between (a) and (b) is the single most valuable
thing this harness measures, because it is the gap the user falls into.

## 3. Single-run sequence (the walking skeleton)

This sequence is proven **once, fully instrumented**, before the loop or any
generalization is written (de-risk the hard part: 7 React turns through a
multi-minute real compose+run, reliably, with reset between).

1. **Login** — reuse the existing staging auth: `tests/e2e/setup/
   staging-global-setup.ts` logs in with `STAGING_USERNAME` / `STAGING_PASSWORD`
   and writes `tests/e2e/.auth/staging-user.json`.
2. **Reset to first-run state** —
   `PATCH /api/composer-preferences {tutorial_completed_at: null,
   default_mode: "guided"}` (re-arms the first-run gate; confirmed contract,
   `frontend/src/stores/preferencesStore.test.ts:228`), then clean sessions via
   `POST /api/tutorial/abandon` + `DELETE /api/tutorial/orphans`.
3. **Drive turns 1→7** through the real UI with one **fixed, reworded,
   non-canonical prompt** (§4). Capture the tutorial `run_id`.
4. **Real-system re-run** — read the composed pipeline from the session's
   composition state (the composer output, pre-normalization), execute it via
   `/api/sessions/{sid}/execute`, capture that `run_id`.
5. **Deep root-cause** — pull both runs from Landscape (errors, node_states,
   outcomes) via the `mcp__elspeth-landscape__*` tools; capture the
   interpretation events raised during composition.
6. **Reset again** — leave `dta_user` clean for the next iteration.
7. **Emit** one structured JSON record (schema §8).

## 4. The fixed prompt (cache bypass + constant difficulty)

The canonical "cool government pages" prompt is cached by
`SHA-256(prompt + model_id)`; running it unedited makes runs 2–N replay a
deterministic canned pipeline and never touch the LLM (`tutorial_service.py:121`).
Any **non-canonical** prompt bypasses the cache entirely (the
`is_canonical_prompt` gate), forcing fresh live composition every run.

The harness uses **one fixed, non-canonical prompt** reworded from the
canonical task (five government pages, scrape HTML, LLM rates/extracts a
subjective attribute, strip HTML, write JSON). Fixed → constant difficulty →
clean `worked N/N` denominator. Non-canonical → real composition every run.

## 5. Assumption-handling rubric (dimension c)

Because the prompt is fixed, the *expected* set of assumptions is knowable.
Per run, capture the interpretation events the composer raised (`kind`,
`user_term`, and — by absence — what it did **not** raise), and compare to a
golden rubric:

- **Expected-to-verify (composer MUST raise `request_interpretation_review`):**
  - `invented_source` — the URLs the composer chose (it invented them; the user
    should confirm).
  - `vague_term` — the subjective rating/extraction criterion (e.g. "how cool",
    "primary colours"), which must also be wired via `prompt_template_parts` +
    `interpretation_ref` (`pipeline_composer.md:467-490`).
- **Expected-to-wave-through (composer must NOT raise a review):**
  - Values the user stated explicitly (abuse contact, scraping reason).
  - Standard mechanical defaults.

Two error modes, weighted asymmetrically:

- **Under-flagging** (waved through something that needed verification) —
  **serious**; this is the silent-wrong-assumption → disillusionment path.
- **Over-flagging** (flagged something trivial or explicitly provided) —
  friction; erodes the "it just works" feel; lower weight, still a fault.

**Grading method:** deterministic rubric match on `kind` + semantic target
(not exact strings, since LLM phrasing varies). An LLM-judge secondary grader
for borderline cases is a possible **extension**, not the primary path — keep
the rubric deterministic to keep stats clean.

**Fix-target:** the composer skill prompt `pipeline_composer.md` (primary);
`request_interpretation_review` plumbing in `tools/sessions.py` / `_dispatch.py`
(secondary).

## 6. Solution-quality & creativity rubric (dimension d)

"Rock solid" is not "it ran without erroring" — it is "the composer creatively
solved the puzzle *well*." The fixed prompt is a genuine puzzle: invent five
real, reachable government URLs; design an extraction that pulls a meaningful
attribute from raw HTML; build the right DAG (scrape → extract → strip HTML →
JSON sink); and be creative **without fabricating** — the skill already requires
this ("Do not invent facts; if the page does not show a clear brand palette…",
`pipeline_composer.md:442`).

A run can pass (a), (b), and (c) and still produce a **degenerate** result.
Dimension (d) catches that. Graded aspects:

- **Source invention** — the URLs resolve and are scrapeable (not 404s, not
  duplicates, five distinct genuine agencies).
- **Extraction design** — the LLM node's prompt actually extracts the requested
  attribute and degrades honestly ("no clear palette") rather than fabricating.
- **Output substance** — output rows carry meaningful, non-empty, non-error
  values for a majority of rows (not all nulls / all "cannot determine" / all
  diverted).
- **DAG shape** — the composed pipeline is the right shape to accomplish the
  stated goal.

**Grading method:** an **LLM-judge** over the real output rows + the composed
DAG, scoring against the aspects above (creativity is irreducibly a judgment —
unlike the deterministic assumption rubric in §5). Source reachability is
checked mechanically (HTTP status from the run's scrape calls in Landscape) to
keep that aspect objective and off the judge. Judge bias controls apply: fixed
rubric, structured verdict, and the judge sees the *task* and the *output*, not
"is this good?" in the abstract.

**Fix-target:** the composer skill prompt `pipeline_composer.md` (extraction and
generated-source discipline, §§"Generated-source discipline" / extraction
sections); secondarily the `web_scrape` / `llm` plugins if the failure is
mechanical rather than compositional.

## 7. Failure classification (makes the loop converge)

Every run is classed on an axis that separates *what we fix* from *noise we
ignore*. Conflating the two means the refinement loop cannot converge — we
would not know whether a prompt change helped or the noise floor moved.

- `pass`
- `tutorial_fault` — subclass + **fix-target**:
  - `composer-skill-prompt` (bad/empty/invalid pipeline emitted)
  - `specific-tool` (a composer tool misbehaved)
  - `plugin` (web_scrape / llm / sink plugin defect)
  - `frontend-state-machine` (stuck turn, render crash, never advanced)
  - `normalization-gap` (passed **a**, failed **b** — the "tutorial lies" class)
  - `assumption-under-flag` / `assumption-over-flag` (failed **c**)
  - `degenerate-output` / `weak-extraction` / `invented-source-unreachable` /
    `wrong-dag-shape` (failed **d** — the puzzle was not creatively solved)
- `infra_fault` — `llm-5xx-or-ratelimit` | `scrape-target-down-or-throttled` |
  `staging-hiccup` | `timeout`

Note the boundary between `invented-source-unreachable` (d, a composition
choice) and `scrape-target-down-or-throttled` (infra): the first is the composer
picking a bad/dead URL; the second is a good URL the network failed to fetch.
The mechanical reachability check (§6) plus retry-on-different-run disambiguates
them.

**Two headline numbers per batch:** **tutorial-pass-rate** (the one we drive to
100%) and **infra-noise rate** (so a prompt change's effect is not confounded).

## 8. Batch record, version tagging, output schema

The refinement loop is "change a prompt/tool → did the rate move?" — which only
works if each batch records what it ran against.

**Batch stamp:** `composer_skill_hash`, `model_id`, git SHA, harness version,
timestamp, prompt text/hash, run count. The **skill hash is the anchor**: it
lets us prove "skill edit X took fresh-composition pass from 6/10 → 9/10."

**Per-run record (JSONL):**

```json
{
  "batch_id": "...", "run_index": 3,
  "outcome": "tutorial_fault",
  "fault_subclass": "assumption-under-flag",
  "fix_target": "composer-skill-prompt",
  "turn_reached": 7,
  "tutorial_run_id": "...", "realsystem_run_id": "...",
  "dim_a_tutorial_completed": true,
  "dim_b_realsystem_passed": true,
  "dim_c_assumptions_ok": false,
  "dim_d_solution_quality": {"ok": true, "judge_score": 0.82, "source_reachable": "5/5"},
  "assumptions": {
    "raised": [{"kind": "invented_source", "term": "..."}],
    "expected_verify": ["invented_source", "vague_term"],
    "expected_waive": ["abuse_contact", "scraping_reason"],
    "under_flagged": ["vague_term"], "over_flagged": []
  },
  "landscape": {"tutorial_errors": [], "realsystem_errors": ["..."]},
  "timing_s": {"compose": 41.2, "tutorial_run": 88.0, "realsystem_run": 71.5},
  "artifacts": {"screenshot": "...", "console_log": "...", "trace": "..."}
}
```

**Deliverables:**
- The harness: a new **non-mocked** staging Playwright spec + a thin runner that
  loops N× with reset between. (The existing `tests/e2e/tutorial.spec.ts` is a
  *mocked* frontend test — it intercepts every backend route and cannot find
  real failures; it is reused only as a selector/flow reference, not extended.)
- Per-batch **report**: the two headline rates + a failure table (run #, turn
  reached, class, fix-target, the Landscape "why", screenshot/console).
- A **results JSONL** appended per batch for cross-batch trend tracking.

## 9. Scaling story (architected-for, NOT built in the first deliverable)

At thousands of runs, browser-driving every run is slow and hammers real
`.gov.au` sites — genuinely abusive *and* a noise source as those sites
throttle or change. Planned second tier (separate work):

- A fast **API-level** bulk loop for prompt/tool refinement, optionally pointed
  at `src/elspeth/testing/chaosweb/` (a controlled scrape target) to strip
  web-flakiness out of the refinement signal.
- The **browser tier** kept as a periodic frontend-regression check.
- Some real-site runs retained for true fidelity; the bulk loop must not depend
  on `dta.gov.au` being up.
- **Puzzle-variation suite** — a small bank of varied tasks of similar
  difficulty (different domains, attributes, source shapes) to test that the
  composer solves puzzles *creatively and generally*, not just the one fixed
  prompt. Each variant carries its own §5 assumption rubric and §6 judge rubric.
  The fixed prompt remains the trend-stat anchor; the suite is the
  generalization probe.

**First deliverable scope:** build the browser tier, prove one run (skeleton),
then 10 runs on the single fixed prompt. Generalize to parameterized-N and the
puzzle-variation suite only after that holds.

## 10. Scope discipline — findings, not fixes

Two defects this harness will expose but **must not try to fix** in this task
(surface as Filigree findings/issues; redesign is separate work):

1. **Cache-as-fakery** — the cached/canonical demo path can show a green
   tutorial while the real composer path is broken.
2. **Normalization fidelity gap** — `_normalise_current_tutorial_state_for_execution`
   repairs the composed pipeline only on the tutorial path, so the tutorial can
   pass while the real-system run of the same pipeline fails.

## 11. Open risks

- **Selector/timing fragility** of a multi-minute real compose+run in a browser
  — mitigated by the skeleton-first approach and generous, explicit timeouts.
- **Single shared account** (`dta_user`) — runs must be sequential
  (single-worker, per `playwright.staging.config.ts`); reset must be reliable or
  runs contaminate each other.
- **Rubric drift** — if the fixed prompt changes, the §5 *and* §6 rubrics must
  be updated in lockstep; the prompt and its rubrics are a coupled triple.
- **LLM-judge as a noise source (dimension d)** — the judge can mis-grade and
  add variance to the very stat we are trying to stabilize. Mitigations: keep
  the objective aspects (source reachability, output non-emptiness) off the
  judge and compute them mechanically; use a structured rubric verdict; record
  the judge model + version in the batch stamp so judge changes are attributable
  and not confused with composer changes.
- **Fixed-prompt vs creative-generalization tension** — a single fixed prompt
  gives clean stats but can be "solved" by a composer that overfits to it. The
  puzzle-variation suite (§9) is the counterweight; until it exists, a high
  fixed-prompt pass rate is necessary but not sufficient evidence of
  rock-solidity.
