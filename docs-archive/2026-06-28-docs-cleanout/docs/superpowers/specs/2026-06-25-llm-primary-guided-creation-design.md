# Design: LLM-primary guided pipeline creation

- **Date:** 2026-06-25
- **Status:** Approved (design); ready for implementation planning
- **Branch:** `release/0.7.0`
- **Author:** John Morrissey (with Claude)
- **Supersedes / folds in:** `docs/superpowers/specs/2026-06-25-tutorial-synthetic-scrape-page-design.md`
  (`fd530e450`). This spec **keeps** that design's synthetic pages, base-URL/SSRF
  handling, and the always-on 3-state prompt-shield review, and **supersedes** its
  §1.4 source *pre-seeding* approach (see "Relationship to the synthetic-scrape
  spec" below).

## Product thesis (the conceit this restores)

ELSPETH guided/tutorial composition exists to bring the tool to **domain
specialists who do not write config files**: they say what they want in plain
English and an **LLM transforms it into pipeline config**. That LLM transform is
the *product*, not an optional helper.

The 0.7.0 staged-recut (P3–P7, delivered) correctly split composition into
phases (source → sink → transform → wire) — small per-phase prompts are far more
reliable than one big-bang inference, and the stages teach the mental model. But
the delivered surface made each phase a **deterministic widget** (`single_select`
/ `schema_form`) with the per-step LLM chat ("Ask about this step") as an
**advisory-only sidecar** (`solve_step_chat`: *"chat does not advance step
state… a non-load-bearing helper"*). That inverts the thesis — the LLM became
"help if you want" instead of the primary way to build. This design flips it back
**without reverting to big-bang**: every phase stays a phase, but each phase is
LLM-primary.

Two concrete observations motivate this (both seen on the live staged tutorial):

1. **LLM is not primary.** The phase leads with the deterministic widget; the
   LLM box is secondary. A `single_select` of registered source plugins also
   *cannot express* "scrape these pages" — `web_scrape` is a transform, not a
   source — which is the same "source-stage dead-end" the synthetic-scrape spec
   routed around by pre-seeding. LLM-primary fixes that dead-end at the root.
2. **The tutorial can reach a panel-less freeform surface.** The guided
   `ChatPanel` branch renders a *switch-to-freeform* affordance and handles an
   `exit_to_freeform` terminal by falling through to the freeform body — which
   has **no** "Ask about this step" panel. There is no such thing as a "freeform
   tutorial"; a tutorial must never land there.

## Usage modes — 4 ways to use, 3 entry points

| # | Way to use | Entry point | Structure | How config is built |
|---|------------|-------------|-----------|---------------------|
| 1 | **Tutorial** | Tutorial | Staged (guided panel) | **Passive worked example** — pre-scripted intent, live LLM build over the synthetic-scrape scenario; the learner watches, authors nothing |
| 2 | **Guided + LLM** | Guided | Staged (guided panel) | LLM drives each phase from plain English |
| 3 | **Guided + manual** | Guided | Staged (guided panel) | Fill / edit the structured form by hand |
| 4 | **Freeform** | Freeform | Unstaged | *Just LLM* — whole-pipeline LLM compose (existing freeform loop) |

Key facts about this taxonomy:

- **Ways 2 and 3 are the SAME panel.** "Guided + LLM" and "guided + manual" are
  not two modes you pick — they are two ways to operate one guided surface, and
  the learner **switches between them freely, in both directions, at any phase**:
  - *"The LLM got it wrong"* → take over: edit the form, or fill it by hand.
  - *"I'm stuck"* → ask the LLM: type intent and it builds/fixes the phase.
  The intent box and the editable form are **both always live** on every guided
  phase. There is no locked "auto" or "manual" state to toggle.
- **Tutorial (way 1) is a passive worked example on the guided panel** — same
  surface, pre-scoped to the synthetic-scrape scenario, but the learner
  **specifies nothing**: the intent is pre-scripted, the LLM does the transform
  live, and the learner watches the pipeline build phase by phase. It exists to
  *show* the conceit before the learner does it themselves in guided mode.
  Tutorial never exposes freeform (concern B) and requires no authoring.
- **Freeform (way 4) is "just LLM"** — the unstaged, whole-pipeline LLM compose.
  It is a distinct entry point and surface, not reachable from inside a tutorial.
  (There are thus two LLM experiences: *staged* LLM in the guided panel, and
  *unstaged* LLM in freeform.)

## Goals

- Every guided phase **leads with a plain-English intent box**; the LLM
  transforms the learner's words into that phase's config.
- The intent box and the structured form are **both always present** on every
  guided phase; the learner moves between LLM-driven and manual **in both
  directions at any time** (LLM→manual when it's wrong; manual→LLM when stuck).
- The structured form is the **editable result** beneath the intent box: a power
  user can fill/edit it directly, ignore the LLM entirely, or accept the LLM's
  proposal as-is.
- The LLM stays present on **every** phase and can **revise the current phase's
  config at any time** ("unfuck mistakes"), not just produce an initial proposal.
- The tutorial is a **passive worked example**: LLM-primary end to end, requiring
  **no user authoring** (the learner watches), and it **never** exposes a freeform
  affordance or a panel-less surface.
- Keep the staged structure, the per-phase reliability, and the synthetic-page
  tutorial scenario + the 3-state prompt-shield review from the folded-in spec.

## Non-goals

- **No big-bang revert.** We do not re-introduce a single whole-pipeline prompt.
  LLM-primary is **per phase**.
- **No removal of the manual path.** The structured form remains fully usable; a
  power user can click through every phase without touching the LLM.
- **No cross-phase auto-revise.** The intent box revises the **current** phase.
  To change an earlier phase, the learner navigates Back (the per-phase Back nav
  fixed during the P7 end-to-end review). Cross-phase revise is a future item.
- **No LLM-fabricated data.** The LLM constructs *config* from intent; it never
  invents source data (URLs, paths). Concrete data (e.g. the tutorial's synthetic
  URLs) is provided as a dataset, consistent with the Tier-1 "exception not
  implicit fabrication" doctrine.
- **No hard-block from the LLM path.** If the LLM is unavailable/malformed the
  phase degrades to the manual form (advisory polarity; never bricks a phase).
- **No new `InterpretationKind`** (inherited from the folded-in spec).

---

## Core model: LLM-primary per phase

Each phase (source, sink, transform; the wire phase is a confirm — see §"Wire")
follows the same shape:

1. **Intent box on top.** Caption invites plain English, e.g. *"Describe the
   source you want"* / *"…the output you want"* / *"…what each row should
   become."* This is the primary input and the default focus.
2. **Driver.** On submit, the phase's LLM **driver** transforms the text into a
   proposed config and **applies** it through the existing commit seam
   (`handle_step_1_source` / `handle_step_2_sink` / `handle_step_3_chain_accept`
   / recipe-apply). Applying — not just replying — is what makes it "creation."
3. **Form = editable result.** The structured widget renders **beneath** the
   intent box, populated with the applied config, fully editable. The learner
   confirms, tweaks, or rebuilds by typing again.
4. **Revise anytime.** The intent box persists and accepts revision instructions
   against the *current* phase config ("make the rating 1–5 not 1–10", "drop the
   raw HTML field") → the driver re-proposes and re-applies that phase.
5. **Manual is always available.** Typing nothing and filling the form by hand
   (or editing the LLM's result) is a first-class path — never gated.

The interaction is therefore one surface ("intent → result you can edit"), not a
mode toggle. "Auto vs manual" is just "did you type intent or fill the form."

### Error handling / fail-safe

- LLM unavailable / malformed / times out → the phase still renders the manual
  form; the intent box shows a non-blocking "couldn't build that — configure it
  below or try rephrasing" state. The wizard remains fully operable offline
  (the existing `solve_step_chat` already returns a synthetic non-load-bearing
  reply on failure; the driver path adopts the same advisory posture).
- The LLM proposes config; the **existing strict commit seams**
  (`handle_step_*`, `validate_pipeline`, the interpretation writer boundary)
  remain the source of truth — a malformed proposal is rejected by them, not
  silently applied.

---

## Backend: per-phase drivers (the substance)

The work is to turn "advisory chat" into "per-phase driver that proposes **and
applies** config," reusing what exists.

- **Source driver — extend for scrape intent.** Today `maybe_resolve_step_1_source_chat`
  → the `resolve_source` tool resolves a chat message to a *registered* source
  plugin (file/store), which cannot express "scrape these pages." Extend the
  source driver so that a scrape/URL-list intent over a provided URL dataset
  yields the canonical `inline_blob` (URL rows) **source** plus recognises that
  `web_scrape` belongs in the transform stage — i.e. the driver produces the
  source config and routes the scrape to the recipe/transform path rather than
  proposing a non-existent `web_scraper` source. This is the root fix for the
  source-stage dead-end.
- **Transform driver — reuse.** The chain-solver (`propose_chain`) already turns
  free text into proposed transforms; it becomes the transform phase's driver
  behind the intent box.
- **Sink driver — new.** There is no free-text → sink-config driver today
  (recipe_match has slot resolvers, not a chat driver). Add one: free text →
  proposed sink output config, applied via `handle_step_2_sink`.
- **Revise capability.** Each driver must accept a revision instruction against
  the phase's *current* applied config (not only a cold start), re-proposing and
  re-applying. The drivers' prompts include the current config so "change X"
  resolves relative to it.
- **Upgrade the per-step chat path.** The per-phase intent box routes to the
  phase driver (apply) instead of advisory-only `solve_step_chat`. `solve_step_chat`'s
  advisory Q&A remains available for genuine questions, but the **primary** action
  of the box is to drive the phase. The `POST /guided/chat` contract gains the
  ability to advance/mutate the current phase (today it explicitly does not) —
  this is the load-bearing backend change; it must preserve the existing
  step-mismatch 409 / unknown-step 400 guards and the advisory-on-failure posture.

All applies go through the existing commit seams and validation — no new write
boundary, no new trust-tier surface beyond what those seams already enforce.

### Wire phase

The wire phase stays a **confirm** of the derived topology (it is computed, not
authored). The intent box there is available to **explain** the wiring and to
drive the small set of supported wire-stage adjustments already modelled as
interpretation decisions (e.g. the raw-HTML cleanup `pipeline_decision`); it does
not free-form-edit the graph.

---

## Frontend: intent-primary surface + remove the freeform dead-end (concern B)

In the guided `ChatPanel` branch (`src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`):

- **Reorder:** the intent box ("Ask about this step" → recaptioned per phase)
  moves **above** the structured turn; the structured turn renders as the
  editable result beneath it. The intent box is present on **every** guided phase.
- **Tutorial never reaches freeform:** remove the *switch-to-freeform* affordance
  from the guided surface **when the session is a tutorial profile** (or, more
  simply, suppress it whenever it would orphan the panel), and ensure a
  tutorial's `TutorialGuidedShell` never renders the panel-less freeform body. A
  tutorial session has no `exit_to_freeform` path.
- **Freeform mode is untouched** as a separate top-level surface for non-tutorial
  use. The change is scoped to: (a) the guided phase layout, and (b) preventing a
  *tutorial* from leaving guided.

The TS `WorkflowProfile` already rides the wire (P6) so the frontend can tell a
tutorial/guided session from freeform.

---

## Tutorial scenario (folds in `fd530e450`, reconciled to LLM-primary)

The tutorial demonstrates the conceit end-to-end on **controlled** content:

- **Synthetic pages (kept verbatim from the folded-in spec):** 3 ELSPETH-served
  static pages under `frontend/public/tutorial-site/project-{1,2,3}.html`,
  banner-marked "SYNTHETIC TEST DATA ONLY — DO NOT USE" + `noindex`, each with a
  risk register / schedule / cost table whose values differ. Served via the SPA
  `StaticFiles` mount. The LLM **derives** `project_name` / `top_risk` /
  `key_date` / summed `total_cost` (light reasoning, verifiable).
- **Source phase is an LLM-driven worked example (supersedes pre-seeding):** the
  worked-example script supplies **both** the plain-English intent ("scrape these
  project pages and pull out the name, top risk, key date, and total cost") **and**
  the 3 synthetic URLs as the dataset (base-URL-resolved — see below). The learner
  **types nothing**; they watch the **LLM build the pipeline** live (`inline_blob`
  over those URLs → `web_scrape` → `llm` extract → `json`) and the form populate.
  The URLs are concrete scripted data (never LLM-fabricated); the pipeline shape is
  the LLM transform of the scripted intent. This is the conceit *demonstrated* —
  the learner then does it for real in guided mode.
- **Base-URL resolution + SSRF allowlist (kept from the folded-in spec):** the
  synthetic URLs must point at the origin the backend can reach; add
  `tutorial_sample_base_url` (derive from the public base when unset); set
  `web_scrape.allowed_hosts` to exactly the synthetic-page host (public →
  `public_only`; loopback dev → tight CIDR). This is the highest-risk detail and
  must be tested for both a public-host and a loopback case.
- **Always-on 3-state prompt-shield review (kept verbatim from the folded-in
  spec, Component 2):** every LLM node surfaces the shield decision —
  A already-shielded (silent) / B available-not-wired (strong "use it" advisory)
  / C none available (high-risk "reconsider" advisory) — advisory only, reusing
  `pipeline_decision` with `user_term=prompt_injection_shield_recommendation`
  (no new `InterpretationKind`). The worked example lands in State C and *forces
  the override on screen* — see **Teaching moments** below for the required copy
  framing it as safe-here-only-because-we-control-the-inputs.
- **Scenario constants:** `CANONICAL_TUTORIAL_PROMPT`
  (`web/preferences/tutorial_cache.py` + the byte-identical frontend mirror) and
  `_TUTORIAL_ENTRY_SEED` (`web/composer/guided/profile.py`) move to the synthetic
  scenario. With LLM-primary + the P6.4 reality that `entry_seed` is a server-side
  **string** (not topology), `_TUTORIAL_ENTRY_SEED` is the framing/seed prompt for
  the source phase + the dataset hand-off, NOT a materialised source topology.

### Teaching moments (what the worked example narrates)

A worked example is not silent watching — it *narrates the concepts as they
arise*, which is the pedagogical point. The learner authors nothing, but the
tutorial calls out what is happening. Two teaching moments are load-bearing and
must appear:

1. **"The LLM made an assumption about X."** When the LLM infers something the
   scripted intent did not state — deriving `total_cost` by summing the per-line
   costs, choosing which field is the "top risk," picking the extraction schema —
   the tutorial names it explicitly: *"the LLM assumed X here — this is the kind
   of thing you review, and can correct by telling it what you meant."* This
   teaches that an LLM transform makes **reviewable** assumptions, and connects to
   the interpretation-surfacing system the learner will use for real (the
   assumption is revisable via the intent box; it is surfaced, not hidden).
2. **"Forcing an override of the prompt-shield (promptguard) warning."** The
   worked example reaches prompt-shield State C (no shield wired on the synthetic
   source), surfaces the high-risk "reconsider" advisory, and proceeds anyway —
   but with explicit tutorial copy along the lines of: *"in this specific case it
   is allowed because we control the inputs (these are our own synthetic test
   pages). Against real or untrusted web content you would wire the shield."*
   This is the crucial safety lesson: the override is **not** a general "skip the
   shield" habit — it is acceptable here **only because the inputs are
   controlled**. It names the trust assumption out loud rather than letting it
   ride as an invisible default; the override is visible, justified, and scoped.

Both moments are advisory (never blocking) and must be worded so the learner does
**not** over-generalise either lesson into "assumptions are fine, ignore them" or
"the shield is optional." The point of showing the override is to teach *when* it
is and is not safe.

## Relationship to the synthetic-scrape spec (`fd530e450`) and P6.4

- **Kept:** synthetic pages (§1.1–1.3), base-URL/SSRF (§1.5–1.6), scenario
  constants (§1.7), and Component 2 (3-state prompt-shield) in full.
- **Superseded:** §1.4 "pre-seed the source via entry-seed *materialization*."
  P6.4 (delivered) made `entry_seed` a server-side string with a **no-op**
  materializer (it does not build topology), so that mechanism no longer exists.
  The source dead-end is instead fixed at the root by the **LLM-primary source
  driver**: the tutorial provides the URL dataset, the LLM builds the source.
- The standalone synthetic-scrape implementation plan is **not** pursued
  separately; its content is delivered through this spec's plan.

## Testing & verification

- **Per-phase drivers (real, deterministic where possible):** unit/integration
  tests that a phase intent → driver → applied config via the real
  `handle_step_*` seam; sink driver applies a valid sink; source driver produces
  the `inline_blob`+scrape routing for a URL-list intent; revise re-applies the
  current phase. LLM calls stubbed at the provider chokepoint (assert the apply,
  not the model).
- **Fail-safe:** LLM unavailable/malformed → the manual form still renders and is
  operable; no hard-block.
- **Concern B:** a tutorial session never renders the freeform body and never
  exposes a switch-to-freeform affordance; the intent panel is present on every
  guided phase (component + e2e).
- **Tutorial scenario:** base-URL + `allowed_hosts` derivation (public + loopback);
  synthetic pages served at the expected path; the shield review reaches State C —
  the staging harness rubric expects `prompt_injection_shield_recommendation`,
  verifying the prompt-shield component.
- **Teaching-moment copy (deterministic UI):** the prompt-shield override caveat
  ("…allowed because we control the inputs…") and the "the LLM made an assumption"
  callout labels are static tutorial copy — assert they render at the right step
  (component/e2e). This is distinct from asserting LLM *prompt* text: **skill/LLM
  behavior is not asserted on prompt strings** — the actual assumption-surfacing
  and the model's transform are verified by re-running the tutorial against staging
  (operator-env; the live staging run is operator-owed, consistent with the P7
  delivery's known gap).
- **Trust-boundary:** `wardline scan . --fail-on ERROR` (web_scrape fetched
  content boundary; the LLM-driven apply path consumes Tier-3 free text →
  validated config via the existing seams — confirm no new unguarded boundary).

## Open items for the implementation plan

1. **`POST /guided/chat` contract change** (advisory → can-apply) is the
   load-bearing backend change. Pin: how the box distinguishes "drive this phase"
   from "answer a question" (or whether every submit attempts a drive and falls
   back to advisory prose), and preserve the step-mismatch 409 / unknown-step 400
   guards.
2. **How the worked-example script carries both the scripted intent and the
   synthetic URL dataset** (so the live LLM has concrete URLs to build from without
   fabricating them, and the learner authors nothing). Pin the exact mechanism —
   `CANONICAL_TUTORIAL_PROMPT` + `_TUTORIAL_ENTRY_SEED` (a string) as the
   scripted-intent / dataset carrier — and whether the LLM runs live on each
   watch-through or a cached transcript drives it.
3. **Source driver scrape-routing**: the driver must produce an `inline_blob`
   source + route `web_scrape` into the transform/recipe stage (reusing the
   format-blind web_scrape recipe fixed in the P7 review), not propose a
   non-existent `web_scraper` source.
4. **Base-URL / SSRF derivation** (from the folded-in spec) remains the
   highest-risk detail; test public + loopback.
5. **Shield-availability discovery API** (from the folded-in spec, §2.4): locate
   and query at compose time; default to State C (warn) if undeterminable.
6. **Decomposition:** the plan likely phases as (a) backend per-phase drivers +
   the `/guided/chat` apply contract, (b) frontend intent-primary reorder +
   concern-B freeform removal, (c) tutorial scenario (synthetic pages + LLM-source
   + prompt-shield + constants + harness). Each phase ends green; the tutorial
   phase depends on (a) and (b).

## File-change inventory (anchored, non-exhaustive)

- `src/elspeth/web/composer/guided/chat_solver.py` — extend source driver
  (scrape routing) + add sink driver + revise; the per-phase drive entry points.
- `src/elspeth/web/sessions/routes/composer/guided.py` — `post_guided_chat`
  apply contract (advance the current phase), preserving the 409/400 guards.
- `src/elspeth/web/composer/guided/steps.py` — the `handle_step_*` apply seams
  the drivers feed (reused; extended only as needed for apply-from-driver).
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` — intent box above
  the form (per-phase captions); remove tutorial switch-to-freeform; panel on
  every guided phase.
- `src/elspeth/web/frontend/src/components/tutorial/*` — tutorial copy + ensure
  the shell never reaches freeform.
- `src/elspeth/web/frontend/public/tutorial-site/project-{1,2,3}.html` — **new**
  (from the folded-in spec).
- `src/elspeth/web/preferences/tutorial_cache.py` + frontend mirror —
  `CANONICAL_TUTORIAL_PROMPT`; `src/elspeth/web/composer/guided/profile.py` —
  `_TUTORIAL_ENTRY_SEED` (framing/dataset prompt, not topology).
- `src/elspeth/web/config.py` — `tutorial_sample_base_url`.
- `src/elspeth/plugins/transforms/llm/transform.py` — 3-state prompt-shield
  (from the folded-in spec, Component 2).
- `src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts` +
  `tutorial-reliability.staging.spec.ts` — rubric retarget (from the folded-in spec).
