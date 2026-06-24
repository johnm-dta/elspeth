# Design: Synthetic-page scrape tutorial + always-on prompt-shield review

- **Date:** 2026-06-25
- **Status:** Approved (design); ready for implementation planning
- **Branch:** `release/0.7.0`
- **Author:** John Morrissey (with Claude)

## Background

The first-run guided tutorial drives the *real* guided composer with a tutorial
profile (`POST /api/sessions/{id}/guided/start` → staged `guided/respond`). Its
canonical scenario scrapes five live Australian government homepages and asks an
LLM to rate each homepage's visual impressiveness 1–10.

Two problems motivated this change:

1. **Reliability + safety of the scrape target.** Live external pages are flaky
   (network, rate limits, layout drift) and — more importantly — feed
   **untrusted external HTML into an LLM step**, which is a prompt-injection
   surface in the *first thing a new user runs*. We want the tutorial's only
   fetched content to be content we author.
2. **The shield decision is invisible by default.** The LLM transform's
   prompt-injection-shield recommendation is *conditional* and model-judged
   (`src/elspeth/plugins/transforms/llm/transform.py:1714–1722`): it only fires
   when the composer judges that untrusted remote text flows in. A new user is
   never reliably taught that running an LLM over fetched content without a
   shield is a deliberate risk.

This design addresses both. It is intentionally two coupled components with a
clear seam.

## Goals

- The first-run tutorial scrapes **ELSPETH-served synthetic pages only**; no
  external fetch occurs during the tutorial.
- The synthetic pages are unmistakably marked test data and carry realistic
  structure (tables) so the LLM extraction is a genuine demonstration.
- Every LLM node — tutorial and live — **always surfaces the prompt-shield
  decision** in one of three states, replacing the conditional behavior. It
  remains **advisory** (never hard-blocks a run).
- The tutorial teaches the shield trade-off: *our source is trusted here, but
  skipping the shield should always be treated as high-risk.*

## Non-goals

- **No hard-blocking** on a missing shield. The shield review stays advisory,
  consistent with the existing "prompt-shield review is advisory" decision.
- **No new `InterpretationKind`.** We reuse the existing `pipeline_decision`
  review with `user_term=prompt_injection_shield_recommendation`; adding a kind
  is a 5-consumer change (enum + DB CHECK + tool enum + both resolve dispatches
  + drift guard + DB recreate) and is not warranted.
- **No auto-insertion** of a shield node. States B/C *recommend*; the user
  decides. Recommendation prose is not a graph step.
- **No change to `web_scrape` SSRF defaults** globally; only the tutorial
  pipeline sets a tight `allowed_hosts`.

---

## Component 1 — Synthetic-page scrape tutorial

### 1.1 The synthetic pages

- **Count:** 3 pages.
- **Location:** `src/elspeth/web/frontend/public/tutorial-site/project-1.html`,
  `project-2.html`, `project-3.html`. Vite copies `public/` verbatim into
  `dist/`, so they survive every frontend rebuild and are served by the existing
  SPA `StaticFiles` mount (`src/elspeth/web/app.py:1196–1200`) at
  `<app-origin>/tutorial-site/project-N.html`.
- **Appearance:** each page is an *impressive-looking* fake project brief
  (hero header, polished layout/styling) whose extractable content is simple.
- **Safety markings (every page):** a persistent, unmissable banner
  **“SYNTHETIC TEST DATA ONLY — DO NOT USE”** and
  `<meta name="robots" content="noindex">`.
- **Structured content (every page) — three tables a real brief would have, with
  values that differ across the three projects so extracted output varies:**
  - **Risk register:** `risk · likelihood · impact · mitigation`.
  - **Schedule:** `milestone · date · status`.
  - **Cost breakdown:** `line item · cost`.

### 1.2 What the LLM extracts (the "challenging" part)

One JSON row per page. Rather than echoing the tables verbatim, the LLM **reads
each table and derives** one fact from each (light reasoning, not copy-paste):

- `project_name` — from the page heading.
- `top_risk` — the highest-impact risk and its mitigation (model must compare
  rows of the risk register).
- `key_date` — the go-live / final milestone from the schedule.
- `total_cost` — the **sum** of the cost line items.

Because we author the tables, the correct answer is known, so the output is
verifiable and gradable. *Fallback if preferred later: verbatim full-table
extraction (simpler, no reasoning).*

### 1.3 Pipeline shape (unchanged from today)

`inline source (3 synthetic URLs) → web_scrape → llm (derive 3 facts) → json sink`

### 1.4 Source seeding (resolves the source-stage dead-end too)

The tutorial profile must **pre-seed the synthetic-URL source** rather than rely
on the user constructing it through the step-1 `single_select`. (A live walk on
2026-06-25 showed the dynamic-source-from-chat path stalls for a scrape scenario
because the assistant proposes a `web_scraper` *source*, which is not a
registered source plugin — `web_scrape` is a transform.) Pre-seeding the inline
source via the profile's entry-seed materialization (`guided.py`
`_materialize_profile_entry_seed_state`) both sets up the dummy-page scenario and
sidesteps that dead-end.

- **Source plugin:** an inline source carrying the 3 URLs (recommend `json` with
  one `{ "url": ... }` object per row, or `text` one-URL-per-line). Exact plugin
  + seed mechanism is an implementation-plan decision; it must produce a
  composed source whose rows expose a `url` field for `web_scrape.url_field`.

### 1.5 Base-URL resolution (the crux portability problem)

The seeded synthetic URLs must point at *the origin the backend can actually
reach*:

- **Deployed (e.g. staging):** backend is behind Caddy on a unix socket — no
  local TCP port. The reachable origin is the **public domain**
  (`https://elspeth.foundryside.dev`), which passes `web_scrape`'s default
  `public_only` SSRF policy.
- **Local dev:** Vite serves `public/` on `:5173` while the API runs on `:8000`
  — different origin and a private host.

Therefore the base URL **must be derived, not hardcoded**:

- Add a setting `tutorial_sample_base_url` (optional). When unset, derive from
  the request origin / configured public base used to reach the app.
- Build synthetic URLs as `{base}/tutorial-site/project-N.html`.

### 1.6 SSRF allowlist for the tutorial pipeline

`web_scrape.allowed_hosts` is set to **exactly the synthetic-page host**, the
tightest allowlist that works (not blanket `allow_private`):

- public base host → leave default `public_only`;
- loopback/private base host (local dev) → a tight CIDR (e.g. `127.0.0.1/32`,
  `::1/128`).

Derivation is computed from the resolved base URL host at seed time.
`web_scrape` config supports `allowed_hosts: "public_only" | "allow_private" |
list[CIDR]` (`src/elspeth/plugins/transforms/web_scrape.py:128`).

### 1.7 Coupled scenario constants (must change in lockstep)

- `CANONICAL_TUTORIAL_PROMPT` — `src/elspeth/web/preferences/tutorial_cache.py`
  (kept **byte-identical** to a frontend constant of the same name).
- The frontend constant it mirrors.
- `_TUTORIAL_ENTRY_SEED` — `src/elspeth/web/composer/guided/profile.py`.

All three move from "5 gov pages / rate visual impressiveness" to "3 synthetic
project pages / extract 3 derived facts."

---

## Component 2 — Global always-on, 3-state prompt-shield review

### 2.1 Current behavior (to replace)

`src/elspeth/plugins/transforms/llm/transform.py:1714–1722` (composer_hints):
the shield recommendation is conditional ("*If* internet/untrusted remote text
will flow into this LLM…") and, when proceeding unshielded, stages a
`pipeline_decision` review with `user_term=prompt_injection_shield_recommendation`.

### 2.2 New behavior — three states, evaluated for every LLM node

- **State A — already shielded:** an authorized prompt-injection shield is wired
  **upstream of this LLM node** in the composed DAG → **no review** (do not nag).
- **State B — shield available, not wired:** no upstream shield, but plugin
  discovery lists an authorized shield (e.g. `azure_prompt_shield`) → stage a
  **strong "use it" advisory** ("use the prompt shield — we really mean it").
- **State C — no shield available:** no upstream shield and discovery offers no
  authorized shield → stage a **"not available, high-risk, reconsider"
  advisory**.

States B and C **always** fire (unconditional), replacing the model-judged
"only if untrusted text" gate. State A stays silent so a correctly-shielded
pipeline is not nagged. All three are **advisory** — they surface a review;
they never hard-block a run.

### 2.3 Interpretation surface (reuse, do not add a kind)

Both B and C use the existing `pipeline_decision` review with
`user_term=prompt_injection_shield_recommendation`; the **draft prose differs by
state** (B: recommend the discovered shield; C: state that no authorized shield
is available and proceeding sends model input unshielded). State A emits nothing.

### 2.4 Detection (implementation-plan items)

- **Upstream-shield graph check (State A):** enumerate which plugins count as
  authorized shields (`azure_prompt_shield` + deployment equivalents) and detect
  one feeding the LLM node in the composed DAG.
- **Availability check (B vs C):** locate the plugin-discovery surface the
  existing hint refers to ("use `azure_prompt_shield` only when discovery lists
  it") and query whether an authorized shield is offered in this deployment.

### 2.5 Tutorial copy

The tutorial deliberately lands in **State B or C** (synthetic source, no shield
wired) and includes copy teaching the trade-off, e.g.:

> *In this instance the source is our own synthetic content, so we proceed
> without a prompt shield — but running an LLM over fetched content without a
> shield should always be treated as a high-risk decision, not a default.*

---

## Testing & verification

- **Skill/guidance behavior is LLM-driven**, so the shield-state behavior and the
  scenario are verified by **re-running the tutorial against staging** (the
  instrumented probe / staging harness), not by asserting on prompt text.
- **Staging harness rubrics** (`src/elspeth/web/frontend/tests/e2e/`
  `tutorial-reliability.staging.spec.ts` + `harness/prompt-and-rubric.ts`) are
  retargeted:
  - `FIXED_PROMPT` → the synthetic-page extraction scenario.
  - `ASSUMPTION_RUBRIC` → drop the `vague_term` expectation (extraction is not
    subjective); **expect `prompt_injection_shield_recommendation`** now that the
    shield review always fires — the harness thereby verifies Component 2.
  - `JUDGE_RUBRIC` / `substantiveRowCount` → grade the derived fields
    (`top_risk`, `key_date`, `total_cost`).
- **Deterministic unit tests** (real code, worth writing): base-URL + `allowed_hosts`
  derivation; the upstream-shield graph check (A vs B/C); static pages served at
  the expected path; the seeded inline source exposes a `url` field.
- **Trust-boundary:** run `wardline scan . --fail-on ERROR` (the synthetic pages
  are new fetched-content; the boundary is `web_scrape`).

## Risks / open items for the plan

1. **Base-URL/origin split (dev vs deploy)** is the highest-risk detail
   (§1.5/§1.6). The plan must pin exactly how the base is derived and how
   `allowed_hosts` follows it, and test both a public-host and a loopback case.
2. **Entry-seed source materialization** (§1.4): confirm the exact hook and the
   inline-source plugin/shape; verify the seeded source removes the step-1
   single_select dead-end rather than papering over it.
3. **Shield availability discovery API** (§2.4): must be locatable and queryable
   at compose time; if availability is genuinely not determinable in a
   deployment, default to State C (fail-safe: warn).
4. **Harness churn:** the rubric retarget is substantial; treat the harness as a
   first-class deliverable, not an afterthought.

## File-change inventory (anchored, non-exhaustive)

- `src/elspeth/web/frontend/public/tutorial-site/project-{1,2,3}.html` — **new.**
- `src/elspeth/web/preferences/tutorial_cache.py` — `CANONICAL_TUTORIAL_PROMPT`.
- frontend canonical-prompt constant (byte-identical mirror).
- `src/elspeth/web/composer/guided/profile.py` — `_TUTORIAL_ENTRY_SEED`
  (+ entry-seed source materialization wiring as needed).
- `src/elspeth/web/composer/guided/...` — entry-seed source materialization
  (`_materialize_profile_entry_seed_state` in `guided.py` route helpers).
- `src/elspeth/web/config.py` — `tutorial_sample_base_url` setting.
- `src/elspeth/plugins/transforms/llm/transform.py:1714–1722` — replace
  conditional shield hints with the 3-state model.
- tutorial copy component (`src/elspeth/web/frontend/src/components/tutorial/…`).
- `src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts` +
  `tutorial-reliability.staging.spec.ts` — rubric retarget.
