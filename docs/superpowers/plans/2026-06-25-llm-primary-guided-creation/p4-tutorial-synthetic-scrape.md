# Tutorial Synthetic-Scrape Passive Worked Example — Implementation Plan (p4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax.

**Goal:** Retarget the first-run guided tutorial from "scrape 5 live gov homepages and rate them" to a self-contained passive worked example that scrapes 3 ELSPETH-served **synthetic** project pages, derives 4 facts per page (name / top risk / key date / total cost), lands in prompt-shield State C, and teaches the assumption-review and shield-override lessons.

**Architecture:** p4 is the **tutorial-scenario slice** of the 4-plan LLM-primary series. It owns (1) the 3 synthetic HTML pages served by the existing SPA `StaticFiles` mount, (2) a NEW deterministic runtime resolver that turns the app's request origin into the 3 synthetic URLs + an SSRF `allowed_hosts` value (the security control is set by the seam, NEVER by the LLM), (3) the lockstep retarget of the canonical-prompt / entry-seed scenario constants, (4) the tutorial teaching-moment copy, and (5) the staging-harness rubric retarget. p4 **depends on p1** (the source driver that ingests a URL dataset and routes `web_scrape` into the transform stage — contract §2.2), **p2** (the `isTutorial` freeform-suppression so a tutorial never reaches a panel-less surface — contract §1/§4), and **p3** (the always-on 3-state prompt-shield that produces the State-C result p4's copy narrates — contract §2.3). p4 implements NONE of those sibling slices; it consumes them.

**Tech Stack:** Python 3.13 / FastAPI / Starlette `StaticFiles` / Pydantic v2 (frozen `WebSettings`) / pytest; TypeScript / React / Vitest / Playwright (staging e2e harness). No DB schema change.

## Global Constraints

- All work lands on `release/0.7.0` (the named release branch), NOT a feature
  branch. Verify `git branch --show-current` before committing; feature branches
  get orphaned.
- The agent SIGNS NOTHING. The operator holds the HMAC key and pushes per the
  release-train process. Do not proactively re-sign tier-model fingerprints or
  plugin hashes; surface owed re-signs as an operator chore.
- Editing a plugin file (e.g. `src/elspeth/plugins/transforms/llm/transform.py`)
  trips TWO CI gates, both operator-owed re-sign chores: (a) the plugin
  `source_file_hash` gate (`plugin-contract-plugin-hashes`) — refresh via
  `scripts/cicd/plugin_hash.py` (`compute_source_file_hash`/`fix_source_file_hash`);
  (b) the tier-model fingerprint cascade (`trust-tier-model`; adding imports
  shifts `Module.body` indices) — allowlists `config/cicd/enforce_tier_model/plugins.yaml`
  (plugin files) and `.../web.yaml` (web files: interpretation_state.py, state.py),
  rotated via `elspeth_lints.rules.trust_tier.tier_model.rotate`
  (scripts/cicd/rotate_tier_model_fingerprints.py). Co-land the fingerprint/hash
  updates with the source change; the operator re-signs.
- The canonical tutorial prompt couples FOUR things in lockstep: the backend
  constant `CANONICAL_SEED_PROMPT` (`web/preferences/tutorial_cache.py`), its
  byte-identical FRONTEND MIRROR `CANONICAL_TUTORIAL_PROMPT`
  (`frontend/src/components/tutorial/tutorialMachine.ts`, byte-identity enforced
  by `test_canonical_seed_matches_frontend_constant`), the `composer_skill_hash`
  re-bake (`PIPELINE_COMPOSER_SKILL_HASH` in `composer/prompts.py` +
  `assert_skill_hash_unchanged_on_disk`) when the live `pipeline_composer.md`
  skill changes, AND a live-prompt SERVICE RESTART. Editing the prompt constant
  alone needs the mirror + the two value-assert tests (NO restart). Editing the
  live skill/recipe needs the re-baked hash + restart (the 5-input
  `tutorial_model_id` invalidates the cache). Do not conflate the two.
- Prompt-shield reviews and advisor/checkpoint reviews are ADVISORY and NEVER
  hard-block: emitted into `validate()` `warnings` at "medium", excluded from the
  blocking contract. Do not promote them to errors.
- `entry_seed` (the tutorial framing/dataset seed) is SERVER-SIDE ONLY and never
  rides the wire: it is redacted from `WorkflowProfileResponse`. Do not add it to
  any wire/GET shape, and do not infer "tutorial" from the wire profile booleans
  (use the client-only `isTutorial` React prop).
- Existing tests that assert about-to-change behavior must be UPDATED to the new
  behavior, NOT reverted. A wave of failures after a structural change is the
  change landing visibly; update the assertions, do not roll back the change.
- For code commits use `git commit` with
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier` (the
  operator-owed-re-sign gates) — NEVER a blanket `--no-verify`. Doc-only commits
  may use `--no-verify`. Reconcile the full slice diff at the slice boundary.
- NEVER `git add -A` / `git add .`; stage only the files this plan owns.

---

## Scope: what p4 owns vs. what is a SIBLING plan's slice

p4 owns ONLY the tutorial scenario. If you find yourself editing any of the
following, STOP — it belongs to another plan in this series:

- **`src/elspeth/plugins/transforms/llm/transform.py` (3-state shield)** → **p3**.
  The folded-in spec listed the shield under the tutorial component; the contract
  (§2.3) reassigns it to p3. p4 only **consumes** p3's per-node State-C result to
  word the tutorial override copy. p4 touches NO plugin file, so the plugin-hash
  gate does NOT fire for p4.
- **`ChatPanel.tsx` reorder / freeform suppression / `isTutorial` prop (concern B)**
  → **p2**. p4 relies on p2 having made "a tutorial never reaches the panel-less
  freeform body" true; p4 does not re-implement it.
- **`maybe_resolve_step_1_source_chat` scrape routing + `/guided/chat` apply
  contract** → **p1** (contract §2.2). p4 produces the URL dataset + `allowed_hosts`
  that p1's source driver consumes; p4 does not edit the driver.

### Gate reconciliation (do NOT copy ground-truth Q3's "restart" blindly)

The ground truth (Q3) warned that p4 might trip BOTH the canonical-prompt gate
AND the skill/recipe-hash-restart gate. **Mostly false:** retargeting the tutorial
*scenario wording* is scenario-agnostic at the recipe level — the *extraction*
prompt the LLM runs drives name/risk/date/cost, NOT the recipe — so it edits no
recipe and no skill. **The one recipe edit p4 DOES make is the optional
`allowed_hosts` SSRF slot (Task 6)** — an additive, behaviour-preserving slot
(empty default → current `public_only` behaviour for every existing caller). It
adds a slot to `recipes.py` / `recipe_match.py`, which **does** change the 5-input
`tutorial_model_id` `recipe_hash` (`tutorial_service.py:877-888`) → the canonical
cache invalidates (already vestigial per Task 3, so no behaviour change). It does
**NOT** edit the live `pipeline_composer.md` skill, so there is **NO
`composer_skill_hash` re-bake and NO service restart**. **Consequence: p4 fires
the canonical-prompt coupling gate (constant + mirror + two value-asserts) and
invalidates the (already-dormant) tutorial cache; it does NOT need a service
restart or a skill-hash re-bake.** p4 touches no plugin file, so the plugin-hash
gate does not fire.

### EPOCH / schema decision (contract §3 p4)

p4 bumps NEITHER `GUIDED_SESSION_SCHEMA_VERSION` (=6) NOR `SESSION_SCHEMA_EPOCH`
(=24). `WorkflowProfile` shape is unchanged — only the `_TUTORIAL_ENTRY_SEED`
*value* moves. The new `WebSettings.tutorial_sample_base_url: str | None = None`
is a settings-shape addition to a `frozen, extra="forbid"` model, orthogonal to
the session DB epochs; it triggers no DB-delete migration and no boot fail-close.

### The passive auto-drive (DECISION: owned by p4 as Task 8 — confirm the home)

The spec's headline conceit (way 1, §"Usage modes") is that the tutorial is a
**passive** worked example: the learner **specifies nothing**, and the LLM builds
the pipeline live. That behaviour is owned by NO plan in the series — the
contract resolved only the URL *carrier*, not the auto-drive itself. Today
`TutorialGuidedShell.tsx:23-26` embeds the real `ChatPanel` and the learner
drives it by hand; the source driver reads URLs out of `body.message`
(`guided.py:1779`). After p1+p2+p3+p4 land, the tutorial is still learner-driven
unless something composes the scripted intent + resolved URLs and auto-submits.

This is ONE gap (the URL-drive AND the `allowed_hosts`-value supply are the same
tutorial-consumer step). **DECISION: p4 owns it as Task 8** — p4 is the tutorial
slice and already depends on p1+p2+p3. Task 8's *steps* are deferred (they consume
p1's `chatGuided`-apply contract + p2's intent-primary panel, which do not exist
until those plans land — concrete TDD steps would be placeholders, which this
methodology forbids). Task 8 therefore documents the consumer at SIGNATURE level
now and is implemented after p1+p2 land. **Operator: confirm p4 is the right home
(the alternative is a dedicated p5); do not leave it unowned.**

### `allowed_hosts` node injection (OWNER: p4 end-to-end — slot Task 6, value Task 8)

The `web-scrape-llm-rate-jsonl` recipe builder (`recipes.py:687-710`) sets NO
`allowed_hosts`, so the node defaults to `"public_only"` (`web_scrape.py:129`).
**On a loopback dev base the tutorial scrape would fail — the exact loopback case
the spec mandates testing.** p4 adds the OPTIONAL `allowed_hosts` slot (Task 6,
empty-default-omit, behaviour-preserving) AND owns the value supply (Task 8): p4
does NOT deflect the value to p1.

**Concrete owner + seam (verified against the live tree):** the recipe-offer
ACCEPT path is the injection point. At STEP_2.5, `response["chosen"] ==
["accept"]` dispatches through `_dispatch_guided_respond` →
`handle_step_2_5_recipe_apply` (`composer/guided/steps.py:223`), which builds
`arguments["slots"] = dict(match.slots)` from the `RecipeMatch` whose slots were
prefilled by `_web_scrape_slot_resolver` (`recipe_match.py:266`). The resolver
takes only `(source, sink)` and has NO session/origin context, so it CANNOT
prefill `allowed_hosts` (and Task 6 keeps `recipe_match.py` un-edited). The seam
that DOES have context is the STEP_2.5 accept dispatch in
`sessions/routes/composer/guided.py` (it holds `request.app.state.settings` and
the request for the origin, and the loaded tutorial profile). Task 8 injects the
resolver output into the match's `allowed_hosts` slot there for tutorial sessions
— `RecipeMatch` is `frozen=True`, so this is a `dataclasses.replace(match,
slots={**match.slots, "allowed_hosts": hosts})` (or an extra-slots arg threaded
into `handle_step_2_5_recipe_apply`), NOT an in-place mutation of `match.slots`
(see Task 8 Step 3 + its apply-time test). The scalar→list mapping
(`"public_only" → []`, an empty list → omit) lives at this seam too, subsuming
Task 6's `public_only`→`[]` facet. The slot EXISTS and is tested in Task 6; its
tutorial VALUE is injected by Task 8 at the named guided.py accept seam.

> **Note on the finding's stale anchor.** Earlier review framing referenced a
> `_helpers.py` STEP_2.5 seam injecting into `edited_values['slots']`. That path
> does not exist in the live tree: there is no `composer/guided/_helpers.py`, and
> the `recipe_offer` accept response carries `chosen=["accept"]` with no
> `edited_values` slot merge — the slots come from the resolver-prefilled
> `match.slots`, applied in `handle_step_2_5_recipe_apply`. The seam named above
> is the verified-correct equivalent.

---

## Task 1: Serve 3 synthetic project pages (content, not status)

The tutorial's only fetched content must be content ELSPETH itself authors.
Create 3 static HTML pages under the Vite `public/` dir; Vite copies `public/`
verbatim into `dist/`, so they serve through the existing SPA `StaticFiles` mount
(`app.py:1196-1200`, mounted at `/` with `html=True`) at
`<app-origin>/tutorial-site/project-{1,2,3}.html`.

**Files:**
- Create: `src/elspeth/web/frontend/public/tutorial-site/project-1.html`
- Create: `src/elspeth/web/frontend/public/tutorial-site/project-2.html`
- Create: `src/elspeth/web/frontend/public/tutorial-site/project-3.html`
- Create: `tests/integration/web/test_tutorial_site_pages.py`

**Interfaces:**
- Consumes: the SPA `StaticFiles` mount at `app.py:1200`
  (`app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="spa")`).
- Produces (consumed by Task 2 + p1's source driver as the *dataset*): three
  pages at the runtime paths `/tutorial-site/project-{1,2,3}.html`, each
  containing the literal banner `SYNTHETIC TEST DATA ONLY — DO NOT USE`, a
  `<meta name="robots" content="noindex" />` (the test asserts the
  close-form-agnostic substring `content="noindex"`), and three tables (risk register,
  schedule, cost breakdown) whose values DIFFER across the three pages so the
  derived facts (`project_name` / `top_risk` / `key_date` / `total_cost`) vary.

> **`html=True` gotcha — the serving test MUST assert page CONTENT, not the HTTP
> status code.** A *missing* `/tutorial-site/project-N.html` returns `index.html`
> with HTTP 200 (SPA fallback), NOT 404. A status-only test passes against a
> missing file. Assert the `SYNTHETIC TEST DATA ONLY` banner string.

> **The pages must be served from `dist/`, which only exists after
> `npm run build`.** The integration test below reads the **source** files under
> `public/` directly (not via the mount) so it runs in CI without a frontend
> build; it asserts the page CONTENT and structure. The mount-serving path is
> exercised by the operator-owed staging run (P7-pattern known gap).

- [ ] **Step 1: Write the failing test that the 3 pages exist, are banner-marked,
  noindexed, and carry the three tables with differing cost totals.**
  Create `tests/integration/web/test_tutorial_site_pages.py`:

  ```python
  """Phase p4 — the 3 synthetic tutorial-site pages (Component 1 of the spec).

  Reads the SOURCE files under frontend/public/ (Vite copies public/ -> dist/
  verbatim, so what is here is what serves at <origin>/tutorial-site/...). The
  pages must be unmistakably marked test data, noindexed, and carry three tables
  whose values DIFFER across the three projects so the derived facts vary.
  """

  from __future__ import annotations

  from pathlib import Path

  import pytest

  _PUBLIC = Path(__file__).resolve().parents[3] / "src/elspeth/web/frontend/public/tutorial-site"
  _PAGES = ("project-1.html", "project-2.html", "project-3.html")


  @pytest.mark.parametrize("name", _PAGES)
  def test_synthetic_page_is_marked_test_data(name: str) -> None:
      html = (_PUBLIC / name).read_text(encoding="utf-8")
      assert "SYNTHETIC TEST DATA ONLY — DO NOT USE" in html
      # Match either the self-closing (' />') or plain ('>') form so the
      # assertion agrees with the XML self-closing fixtures below.
      assert 'content="noindex"' in html


  @pytest.mark.parametrize("name", _PAGES)
  def test_synthetic_page_has_three_tables(name: str) -> None:
      html = (_PUBLIC / name).read_text(encoding="utf-8").lower()
      # Risk register / schedule / cost breakdown headings.
      assert "risk register" in html
      assert "schedule" in html
      assert "cost breakdown" in html
      # The cost table must be summable (>= 3 explicit dollar figures).
      assert html.count("$") >= 3


  def test_synthetic_pages_have_distinct_cost_totals() -> None:
      # The whole point of differing values: the derived total_cost must vary.
      import re

      totals: list[int] = []
      for name in _PAGES:
          html = (_PUBLIC / name).read_text(encoding="utf-8")
          figures = [int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", html)]
          assert figures, f"{name} has no dollar figures"
          totals.append(sum(figures))
      assert len(set(totals)) == 3, f"cost totals must differ across pages, got {totals}"
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/test_tutorial_site_pages.py -x -q
  ```
  Expected failure: `FileNotFoundError: .../public/tutorial-site/project-1.html`.

- [ ] **Step 2: Create `project-1.html`.**
  `src/elspeth/web/frontend/public/tutorial-site/project-1.html`:

  ```html
  <!doctype html>
  <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="robots" content="noindex" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Project Helios — Brief (SYNTHETIC)</title>
      <style>
        body { font-family: system-ui, sans-serif; margin: 0; color: #1a1a2e; }
        .banner { background: #c0392b; color: #fff; text-align: center; padding: 8px; font-weight: 700; }
        .hero { background: linear-gradient(120deg, #1a1a2e, #16213e); color: #fff; padding: 48px 32px; }
        .hero h1 { margin: 0; font-size: 2.4rem; }
        main { max-width: 880px; margin: 0 auto; padding: 24px 32px; }
        table { border-collapse: collapse; width: 100%; margin: 16px 0 32px; }
        th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
        th { background: #16213e; color: #fff; }
        caption { text-align: left; font-weight: 700; margin-bottom: 8px; }
      </style>
    </head>
    <body>
      <div class="banner">SYNTHETIC TEST DATA ONLY — DO NOT USE</div>
      <header class="hero"><h1>Project Helios</h1><p>Public-records digitisation programme — project brief.</p></header>
      <main>
        <table>
          <caption>Risk register</caption>
          <tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>
          <tr><td>Legacy data corruption on ingest</td><td>Medium</td><td>High</td><td>Checksum every record against the source register</td></tr>
          <tr><td>Vendor scanning delays</td><td>Low</td><td>Medium</td><td>Dual-source the scanning contract</td></tr>
          <tr><td>Access-control misconfiguration</td><td>Low</td><td>Low</td><td>Quarterly IRAP-aligned review</td></tr>
        </table>
        <table>
          <caption>Schedule</caption>
          <tr><th>Milestone</th><th>Date</th><th>Status</th></tr>
          <tr><td>Discovery complete</td><td>2026-02-15</td><td>Done</td></tr>
          <tr><td>Pilot ingest</td><td>2026-05-30</td><td>In progress</td></tr>
          <tr><td>Go-live</td><td>2026-09-30</td><td>Planned</td></tr>
        </table>
        <table>
          <caption>Cost breakdown</caption>
          <tr><th>Line item</th><th>Cost</th></tr>
          <tr><td>Scanning &amp; OCR</td><td>$120,000</td></tr>
          <tr><td>Platform engineering</td><td>$80,000</td></tr>
          <tr><td>Assurance &amp; review</td><td>$25,000</td></tr>
        </table>
      </main>
    </body>
  </html>
  ```
  (Total cost = 225,000; go-live 2026-09-30; top risk = the High-impact legacy
  corruption row.)

- [ ] **Step 3: Create `project-2.html`** — identical structure, DIFFERENT values.
  `src/elspeth/web/frontend/public/tutorial-site/project-2.html`: copy
  `project-1.html` verbatim, then change ONLY: `<title>` → `Project Borealis —
  Brief (SYNTHETIC)`, the hero `<h1>` → `Project Borealis`, the hero `<p>` →
  `Grants-administration modernisation — project brief.`, and the three table
  bodies to:

  ```html
        <table>
          <caption>Risk register</caption>
          <tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>
          <tr><td>Grant-rule misinterpretation</td><td>High</td><td>High</td><td>Codify rules with policy sign-off before build</td></tr>
          <tr><td>Identity-provider outage</td><td>Medium</td><td>Medium</td><td>Cache assertions with a short grace window</td></tr>
          <tr><td>Reporting lag</td><td>Low</td><td>Low</td><td>Nightly incremental aggregation</td></tr>
        </table>
        <table>
          <caption>Schedule</caption>
          <tr><th>Milestone</th><th>Date</th><th>Status</th></tr>
          <tr><td>Rules workshop</td><td>2026-03-10</td><td>Done</td></tr>
          <tr><td>Beta release</td><td>2026-06-20</td><td>In progress</td></tr>
          <tr><td>Go-live</td><td>2026-11-15</td><td>Planned</td></tr>
        </table>
        <table>
          <caption>Cost breakdown</caption>
          <tr><th>Line item</th><th>Cost</th></tr>
          <tr><td>Rules engine</td><td>$200,000</td></tr>
          <tr><td>Identity integration</td><td>$60,000</td></tr>
          <tr><td>Reporting</td><td>$40,000</td></tr>
        </table>
  ```
  (Total cost = 300,000; go-live 2026-11-15; top risk = the High/High
  grant-rule row.)

- [ ] **Step 4: Create `project-3.html`** — identical structure, DIFFERENT values.
  `src/elspeth/web/frontend/public/tutorial-site/project-3.html`: copy
  `project-1.html` verbatim, then change ONLY: `<title>` → `Project Meridian —
  Brief (SYNTHETIC)`, the hero `<h1>` → `Project Meridian`, the hero `<p>` →
  `Service-status telemetry platform — project brief.`, and the three table
  bodies to:

  ```html
        <table>
          <caption>Risk register</caption>
          <tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>
          <tr><td>Telemetry volume overrun</td><td>Medium</td><td>High</td><td>Tiered sampling with a hard storage ceiling</td></tr>
          <tr><td>Dashboard adoption shortfall</td><td>Medium</td><td>Medium</td><td>Embed dashboards in existing ops tooling</td></tr>
          <tr><td>Alert fatigue</td><td>Low</td><td>Low</td><td>Tune thresholds against a four-week baseline</td></tr>
        </table>
        <table>
          <caption>Schedule</caption>
          <tr><th>Milestone</th><th>Date</th><th>Status</th></tr>
          <tr><td>Instrumentation</td><td>2026-01-20</td><td>Done</td></tr>
          <tr><td>Pilot dashboards</td><td>2026-04-25</td><td>In progress</td></tr>
          <tr><td>Go-live</td><td>2026-08-12</td><td>Planned</td></tr>
        </table>
        <table>
          <caption>Cost breakdown</caption>
          <tr><th>Line item</th><th>Cost</th></tr>
          <tr><td>Ingestion pipeline</td><td>$90,000</td></tr>
          <tr><td>Dashboarding</td><td>$55,000</td></tr>
          <tr><td>On-call &amp; SRE</td><td>$30,000</td></tr>
        </table>
  ```
  (Total cost = 175,000; go-live 2026-08-12; top risk = the High-impact telemetry
  overrun row.)

- [ ] **Step 5: Run to pass.**
  ```
  cd /home/john/elspeth && uv run pytest tests/integration/web/test_tutorial_site_pages.py -x -q
  ```
  Expected: `7 passed` (3 banner + 3 table + 1 distinct-totals).

- [ ] **Step 6: Commit.**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/frontend/public/tutorial-site/project-1.html \
    src/elspeth/web/frontend/public/tutorial-site/project-2.html \
    src/elspeth/web/frontend/public/tutorial-site/project-3.html \
    tests/integration/web/test_tutorial_site_pages.py && git commit -m "$(cat <<'EOF'
feat(tutorial): add 3 synthetic scrape pages served from public/tutorial-site

ELSPETH-served fake project briefs (Helios / Borealis / Meridian), each
banner-marked SYNTHETIC TEST DATA ONLY, noindexed, carrying a risk
register / schedule / cost-breakdown table whose values differ so the
LLM-derived project_name / top_risk / key_date / total_cost vary per
page. Vite copies public/ into dist/, so they serve at
<origin>/tutorial-site/project-N.html via the existing SPA StaticFiles
mount. The serving test asserts page CONTENT (the banner), not HTTP
status (html=True returns index.html/200 for a missing file).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 2: Runtime base-URL resolver + `tutorial_sample_base_url` setting + `allowed_hosts` derivation

The synthetic URLs are `{base}/tutorial-site/project-{1,2,3}.html` where `{base}`
is **runtime-derived** (the reachable origin differs: public domain on staging,
loopback in dev). A byte-frozen module constant cannot hold a runtime value, so
the scenario's URLs and the SSRF `allowed_hosts` are produced by a NEW
deterministic resolver — NOT by the LLM, and NOT by overloading the no-op profile
materializer.

> **Do NOT mutate `_materialize_profile_entry_seed_state` (`_helpers.py:2260`).**
> Its no-op behaviour is a deliberate P6.4/D16 invariant (emphatic docstring,
> guarded by `test_guided_start_persists_profile_without_materializing_topology`,
> which the ground truth marked "survives"). The contract's "no such seam exists
> today" describes ABSENCE — it is not licence to overload that function.
> Overloading it regresses the invariant and silently breaks that test. p4 adds a
> standalone resolver instead; **application** of the URLs happens at p1's source
> driver and at the `web_scrape` node (this task ships the resolver + tests; p1
> consumes it per contract §2.2/§3.4).

> **SSRF is set by the seam, never by the LLM.** `allowed_hosts` is an SSRF
> control. The passive tutorial's pipeline is LLM-built, so `allowed_hosts` must
> be a deterministic, host-class-keyed value computed by this resolver and applied
> deterministically — the model must not be trusted to set it (project
> asymmetric-trust doctrine).

**Files:**
- Modify: `src/elspeth/web/config.py` — add `tutorial_sample_base_url` to
  `WebSettings` (after `tutorial_cache_dir`, `:58`).
- Create: `src/elspeth/web/composer/tutorial_sample.py` — the resolver module.
- Create: `tests/unit/web/composer/test_tutorial_sample.py`

**Interfaces:**
- Consumes: `WebSettings.tutorial_sample_base_url: str | None`;
  `web_scrape.allowed_hosts` accepts
  `Literal["public_only","allow_private"] | list[CidrStr]`
  (`plugins/transforms/web_scrape.py:128`).
- Produces (consumed by p1's source driver + the `web_scrape` node, contract
  §3.4):
  - `resolve_tutorial_sample_urls(*, base_url: str) -> tuple[str, str, str]` —
    the 3 absolute synthetic URLs.
  - `resolve_tutorial_allowed_hosts(*, base_url: str) -> Literal["public_only"] | list[str]`
    — `"public_only"` for a public host; a tight CIDR list
    (`["127.0.0.1/32", "::1/128"]`) for a loopback/private host. NEVER
    `"allow_private"`.
  - `tutorial_sample_base_url(*, settings: WebSettings, request_origin: str | None) -> str`
    — returns `settings.tutorial_sample_base_url` when set, else the
    `request_origin` (scheme+host[:port]); raises `ValueError` when neither is
    available (fail-loud, not a fabricated host).

- [ ] **Step 1: Write the failing test for the resolver (public + loopback).**
  Create `tests/unit/web/composer/test_tutorial_sample.py`:

  ```python
  """Phase p4 — deterministic tutorial-sample URL + SSRF allowed_hosts resolver.

  The synthetic URLs and the web_scrape allowed_hosts value are computed by the
  SEAM, never by the LLM (allowed_hosts is an SSRF control). Both a public host
  and a loopback host must be covered (spec mandate, highest-risk detail).
  """

  from __future__ import annotations

  import pytest
  from pydantic import SecretBytes

  from elspeth.web.composer.tutorial_sample import (
      resolve_tutorial_allowed_hosts,
      resolve_tutorial_sample_urls,
      tutorial_sample_base_url,
  )
  from elspeth.web.config import WebSettings


  def _settings(**kw) -> WebSettings:
      base = dict(
          composer_max_composition_turns=15,
          composer_max_discovery_turns=10,
          composer_timeout_seconds=85.0,
          composer_rate_limit_per_minute=10,
          # Required field on WebSettings (config.py:250, strict, no default).
          # Mirror the established fixture pattern at
          # tests/unit/web/preferences/test_tutorial_cache.py:264.
          shareable_link_signing_key=SecretBytes(b"\x00" * 32),
      )
      base.update(kw)
      return WebSettings(**base)


  def test_urls_are_built_under_base() -> None:
      urls = resolve_tutorial_sample_urls(base_url="https://elspeth.foundryside.dev")
      assert urls == (
          "https://elspeth.foundryside.dev/tutorial-site/project-1.html",
          "https://elspeth.foundryside.dev/tutorial-site/project-2.html",
          "https://elspeth.foundryside.dev/tutorial-site/project-3.html",
      )


  def test_urls_strip_trailing_slash_on_base() -> None:
      urls = resolve_tutorial_sample_urls(base_url="http://127.0.0.1:5173/")
      assert urls[0] == "http://127.0.0.1:5173/tutorial-site/project-1.html"


  def test_public_host_uses_public_only() -> None:
      assert resolve_tutorial_allowed_hosts(base_url="https://elspeth.foundryside.dev") == "public_only"


  def test_loopback_host_uses_tight_cidr_not_allow_private() -> None:
      hosts = resolve_tutorial_allowed_hosts(base_url="http://127.0.0.1:5173")
      assert hosts == ["127.0.0.1/32", "::1/128"]
      assert hosts != "allow_private"


  def test_ipv6_loopback_host_uses_tight_cidr() -> None:
      hosts = resolve_tutorial_allowed_hosts(base_url="http://[::1]:8451")
      assert hosts == ["127.0.0.1/32", "::1/128"]


  def test_private_rfc1918_host_uses_tight_cidr() -> None:
      # A private (non-public) host must NOT fall through to public_only, and must
      # land on the SAME tight CIDR list as loopback — never a wider allowlist.
      # Exact equality (not just isinstance list) is the security-critical
      # invariant: a resolver bug returning e.g. ["0.0.0.0/0"] must fail this.
      hosts = resolve_tutorial_allowed_hosts(base_url="http://192.168.1.50:8451")
      assert hosts == ["127.0.0.1/32", "::1/128"]
      assert hosts != "allow_private"


  def test_base_url_prefers_configured_setting() -> None:
      settings = _settings(tutorial_sample_base_url="https://configured.example")
      assert (
          tutorial_sample_base_url(settings=settings, request_origin="https://ignored.example")
          == "https://configured.example"
      )


  def test_base_url_falls_back_to_request_origin() -> None:
      settings = _settings()
      assert (
          tutorial_sample_base_url(settings=settings, request_origin="https://elspeth.foundryside.dev")
          == "https://elspeth.foundryside.dev"
      )


  def test_base_url_raises_when_undeterminable() -> None:
      settings = _settings()
      with pytest.raises(ValueError):
          tutorial_sample_base_url(settings=settings, request_origin=None)


  def test_allowed_hosts_none_host_fails_safe_to_loopback() -> None:
      # A base whose urlsplit().hostname is None (e.g. a bare/relative string)
      # must fail SAFE to the tight loopback CIDR list, NEVER widen egress. This
      # exercises the resolver's `if host is None` fail-safe arm directly — a real
      # branch, not a caller-error-only path.
      assert resolve_tutorial_allowed_hosts(base_url="not-a-url") == ["127.0.0.1/32", "::1/128"]
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_tutorial_sample.py -x -q
  ```
  Expected failure: `ModuleNotFoundError: No module named 'elspeth.web.composer.tutorial_sample'`.

- [ ] **Step 2: Add `tutorial_sample_base_url` to `WebSettings`.**
  In `src/elspeth/web/config.py`, immediately after the `tutorial_cache_dir`
  field (`:58`), add:

  ```python
      # Phase p4: explicit public base URL the tutorial synthetic-scrape pages
      # are reachable at (e.g. "https://elspeth.foundryside.dev"). When None, the
      # base is derived from the inbound request origin at compose time. Used ONLY
      # to build {base}/tutorial-site/project-N.html and the SSRF allowed_hosts
      # for the tutorial's web_scrape node — never a general egress allowlist.
      tutorial_sample_base_url: str | None = Field(default=None)
  ```

- [ ] **Step 3: Write the resolver module.**
  Create `src/elspeth/web/composer/tutorial_sample.py`:

  ```python
  """Deterministic resolver for the tutorial synthetic-scrape scenario (p4).

  Turns the app's reachable origin into the 3 synthetic-page URLs and the SSRF
  ``allowed_hosts`` value for the tutorial's ``web_scrape`` node. Everything here
  is computed by the SEAM — the LLM never sets the URLs (it would otherwise
  fabricate them, violating the Tier-1 "exception not implicit fabrication"
  doctrine) and never sets ``allowed_hosts`` (an SSRF control). The 3 concrete
  URLs are handed to the p1 source driver as the dataset (contract §2.2/§3.4).
  """

  from __future__ import annotations

  import ipaddress
  from typing import Literal
  from urllib.parse import urlsplit

  from elspeth.web.config import WebSettings

  _PAGES = ("project-1.html", "project-2.html", "project-3.html")
  # Tight loopback CIDR for local dev. NOT "allow_private" — the tightest
  # allowlist that lets the backend reach its own SPA mount on loopback.
  _LOOPBACK_CIDRS = ["127.0.0.1/32", "::1/128"]


  def resolve_tutorial_sample_urls(*, base_url: str) -> tuple[str, str, str]:
      """Build the 3 absolute synthetic-page URLs under ``base_url``."""
      root = base_url.rstrip("/")
      urls = tuple(f"{root}/tutorial-site/{page}" for page in _PAGES)
      return (urls[0], urls[1], urls[2])


  def resolve_tutorial_allowed_hosts(*, base_url: str) -> Literal["public_only"] | list[str]:
      """Derive the web_scrape SSRF allowlist from the resolved host class.

      Public host -> the default ``"public_only"`` policy is sufficient and is
      the tightest correct value. Loopback / private host (local dev) -> a tight
      CIDR list, NEVER the blanket ``"allow_private"``.
      """
      host = urlsplit(base_url).hostname
      if host is None:
          # Undeterminable host: fail safe to the tight loopback list rather than
          # widening egress. This is a genuine fail-safe, not dead code:
          # tutorial_sample_base_url returns the configured/origin value VERBATIM
          # with no URL validation, so a malformed base (e.g. a bare/relative
          # string whose urlsplit().hostname is None) reaches this arm. Failing to
          # the tightest list is the safe-by-default choice. Covered by
          # test_allowed_hosts_none_host_fails_safe_to_loopback.
          return list(_LOOPBACK_CIDRS)
      try:
          address = ipaddress.ip_address(host)
      except ValueError:
          # A hostname (e.g. elspeth.foundryside.dev) -> public_only covers it.
          return "public_only"
      if address.is_global:
          return "public_only"
      return list(_LOOPBACK_CIDRS)


  def tutorial_sample_base_url(*, settings: WebSettings, request_origin: str | None) -> str:
      """Resolve the base URL: configured setting wins, else the request origin.

      Raises ``ValueError`` when neither is available — we never fabricate a host
      for an SSRF-controlled fetch.
      """
      if settings.tutorial_sample_base_url:
          return settings.tutorial_sample_base_url
      if request_origin:
          return request_origin
      raise ValueError(
          "tutorial_sample_base_url is unset and no request origin is available; "
          "set WebSettings.tutorial_sample_base_url for this deployment."
      )
  ```

  Run to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_tutorial_sample.py -x -q
  ```
  Expected: `10 passed`. (Note: `192.168.1.50` is `is_global == False`, so the
  resolver returns the SAME tight loopback CIDR list `["127.0.0.1/32", "::1/128"]`
  — never a wider allowlist and never `"allow_private"`. The RFC1918 test asserts
  exact equality against that list, the security-relevant invariant.)

- [ ] **Step 4: Verify `WebSettings` still constructs (frozen/extra=forbid does
  not reject the new field).**
  ```
  cd /home/john/elspeth && ELSPETH_ENV=test uv run python -c "from pydantic import SecretBytes; from elspeth.web.config import WebSettings; s = WebSettings(composer_max_composition_turns=1, composer_max_discovery_turns=1, composer_timeout_seconds=1.0, composer_rate_limit_per_minute=1, shareable_link_signing_key=SecretBytes(b'\x00' * 32)); print('tutorial_sample_base_url=', s.tutorial_sample_base_url)"
  ```
  Expected: `tutorial_sample_base_url= None`.

  > **`ELSPETH_ENV=test` is REQUIRED here.** This `-c` invocation runs OUTSIDE
  > pytest, and the default host `127.0.0.1` is loopback, so without the env var
  > `_allow_insecure_test_keys` is False (`config.py:34` needs `pytest` in
  > `sys.modules` OR `ELSPETH_ENV=="test"`) and the `secret_key` / weak-signing-key
  > production guards (`config.py:540,570`) raise `ValidationError` ("secret_key
  > must be set to a secure value outside explicit test contexts") instead of
  > printing the expected line. `ELSPETH_ENV=test` flips the flag (host is already
  > loopback). This step is largely redundant with Step 1/3's
  > `test_base_url_prefers_configured_setting` (which constructs `WebSettings` under
  > pytest); keep it only as a no-pytest smoke check.

- [ ] **Step 5: Commit.**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/config.py \
    src/elspeth/web/composer/tutorial_sample.py \
    tests/unit/web/composer/test_tutorial_sample.py && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit -m "$(cat <<'EOF'
feat(tutorial): add deterministic synthetic-sample URL + SSRF resolver

Add WebSettings.tutorial_sample_base_url (None -> derive from request
origin) and a tutorial_sample resolver that builds the 3 synthetic-page
URLs and the web_scrape allowed_hosts from the resolved host class:
public host -> public_only; loopback/private -> a tight CIDR list
(127.0.0.1/32, ::1/128), never allow_private. The seam sets these
deterministically; the LLM never sets the URLs (no fabrication) or the
SSRF control. p1's source driver consumes the 3 URLs as the dataset.
Does NOT overload the no-op profile entry-seed materializer (a P6.4
invariant); this is a standalone resolver.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 3: Retarget scenario constants in lockstep + decide TutorialCache fate

The canonical-prompt coupling links the backend `CANONICAL_SEED_PROMPT`, the
byte-identical frontend `CANONICAL_TUTORIAL_PROMPT`, the `_TUTORIAL_ENTRY_SEED`
framing prompt, and the e2e `HELLO_WORLD_SESSION_TITLE`. They MUST move in one
commit or byte-identity CI fails mid-plan. This task also makes the
TutorialCache retire-vs-rekey decision explicit (ground truth Q1b).

> **The constants carry the scripted INTENT only, NOT the runtime URLs.** The
> synthetic URLs are runtime-derived (Task 2), so they cannot live in the
> byte-frozen constants. The prompt/seed describe *what to do* ("scrape these
> project pages and extract name / top risk / key date / total cost"); Task 2's
> resolver supplies the *URLs* to p1's driver.

> **TutorialCache decision — RE-KEY by going vestigial (do not delete the
> module).** Because the synthetic URLs are runtime-derived, the effective prompt
> is no longer byte-stable per deploy, so `effective_prompt == CANONICAL_SEED_PROMPT`
> (`tutorial_service.py:117,126`) no longer cleanly holds and the canonical-prompt
> cache stops engaging — the tutorial runs LIVE on each watch-through (the staged
> tutorial already drives the live guided engine; the replay path also has a
> topology gate that falls through to live on mismatch). p4 does NOT delete
> `tutorial_cache.py` (other code imports `CANONICAL_SEED_PROMPT` /
> `TutorialCache` / `tutorial_cache_key`); it retargets the constant to the new
> intent and leaves the cache machinery in place but effectively dormant for the
> synthetic scenario. **Operator-owed:** stale cache files keyed on the old
> `model_id` simply miss; the operator clears `{data_dir}/tutorial_cache/` (the
> documented artifact-delete pattern, `tutorial_cache.py:20-32`). Surface this as
> an operator chore in the commit body; the agent does not delete operator data.

**Files:**
- Modify: `src/elspeth/web/preferences/tutorial_cache.py:56-64` (`CANONICAL_SEED_PROMPT`).
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts:7-14`
  (`CANONICAL_TUTORIAL_PROMPT` mirror).
- Modify: `src/elspeth/web/composer/guided/profile.py:29` (`_TUTORIAL_ENTRY_SEED`).
- Modify: `src/elspeth/web/frontend/src/components/tutorial/copy.ts:14`
  (`HELLO_WORLD_SESSION_TITLE`).
- Test (UPDATE): `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx:124-127`
  — the `renameSession` assertion hardcodes the OLD title literal
  `"hello-world (cool government pages)"` (`:126`), NOT the imported constant, so
  retargeting `HELLO_WORLD_SESSION_TITLE` (Step 6) makes it fail. Update the
  `toHaveBeenCalledWith` literal to the new title (Step 6a).
- Test (UPDATE): `tests/unit/web/preferences/test_tutorial_cache.py:46-57`
  (`test_canonical_seed_prompt_constant_is_exact`) — retarget the verbatim string.
- Test (UPDATE): `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts:10-20`
  — retarget the `.toBe(...)` value.
- Test (SURVIVES, do not touch): `test_canonical_seed_matches_frontend_constant`
  (`:59-81`, reads the `.ts` mirror — it passes automatically once both move in
  lockstep).

**Interfaces:**
- Consumes: nothing new.
- Produces: the retargeted scripted-intent constants the tutorial + p1 driver use
  as the framing prompt.

> **Use the REAL names (the spec mislabels them):** the BACKEND constant is
> `CANONICAL_SEED_PROMPT` (`tutorial_cache.py:56`); the FRONTEND mirror is
> `CANONICAL_TUTORIAL_PROMPT` (`tutorialMachine.ts:7`). They are kept
> byte-identical.

- [ ] **Step 1: Update the backend value-assert test to the new prompt (fail
  first).**
  In `tests/unit/web/preferences/test_tutorial_cache.py`, replace the body of
  `test_canonical_seed_prompt_constant_is_exact` (`:46-56`):

  ```python
  def test_canonical_seed_prompt_constant_is_exact() -> None:
      """The seed prompt must match the synthetic-scrape canonical prompt verbatim."""
      assert CANONICAL_SEED_PROMPT == (
          "Scrape these three synthetic project-brief pages and, for each page, "
          "have an LLM read the tables and return one JSON row with the project "
          "name, the top risk (the highest-impact risk and its mitigation), the "
          "go-live date, and the total cost (the sum of the cost line items). "
          "Remove the raw HTML and write the rows to a json file."
      )
  ```

  Run to fail (the constant still holds the old value):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/preferences/test_tutorial_cache.py::test_canonical_seed_prompt_constant_is_exact -x -q
  ```
  Expected failure: `AssertionError` comparing the old 5-gov-pages string to the
  new synthetic-scrape string.

- [ ] **Step 2: Retarget `CANONICAL_SEED_PROMPT` (backend).**
  In `src/elspeth/web/preferences/tutorial_cache.py`, replace the
  `CANONICAL_SEED_PROMPT = (...)` literal (`:56-64`) with:

  ```python
  CANONICAL_SEED_PROMPT = (
      "Scrape these three synthetic project-brief pages and, for each page, "
      "have an LLM read the tables and return one JSON row with the project "
      "name, the top risk (the highest-impact risk and its mitigation), the "
      "go-live date, and the total cost (the sum of the cost line items). "
      "Remove the raw HTML and write the rows to a json file."
  )
  ```

  Run the backend value test to pass:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/preferences/test_tutorial_cache.py::test_canonical_seed_prompt_constant_is_exact -x -q
  ```
  Expected: `1 passed`.

- [ ] **Step 3: Retarget `CANONICAL_TUTORIAL_PROMPT` (frontend mirror) — BYTE
  IDENTICAL.**
  In `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`,
  replace the `export const CANONICAL_TUTORIAL_PROMPT = ...;` literal (`:7-14`)
  with the byte-identical string (same words, same spacing):

  ```typescript
  export const CANONICAL_TUTORIAL_PROMPT =
    "Scrape these three synthetic project-brief pages and, for each page, " +
    "have an LLM read the tables and return one JSON row with the project " +
    "name, the top risk (the highest-impact risk and its mitigation), the " +
    "go-live date, and the total cost (the sum of the cost line items). " +
    "Remove the raw HTML and write the rows to a json file.";
  ```

  Run the cross-file byte-identity test (this is the lockstep gate):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/preferences/test_tutorial_cache.py::test_canonical_seed_matches_frontend_constant -x -q
  ```
  Expected: `1 passed`. (If it fails, the two strings differ byte-for-byte — fix
  the `.ts` to match the Python exactly, do not change the Python.)

- [ ] **Step 4: Update the frontend value-assert test (`tutorialMachine.test.ts`).**
  In `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`,
  replace the `.toBe(...)` argument (`:11-19`) with the new byte-identical string:

  ```typescript
    it("pins the canonical tutorial prompt verbatim", () => {
      expect(CANONICAL_TUTORIAL_PROMPT).toBe(
        "Scrape these three synthetic project-brief pages and, for each page, " +
          "have an LLM read the tables and return one JSON row with the project " +
          "name, the top risk (the highest-impact risk and its mitigation), the " +
          "go-live date, and the total cost (the sum of the cost line items). " +
          "Remove the raw HTML and write the rows to a json file.",
      );
    });
  ```

  Run the frontend unit test:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- tutorialMachine.test.ts --run
  ```
  Expected: the `pins the canonical tutorial prompt verbatim` test passes (the
  reducer tests are unaffected).

- [ ] **Step 5: Retarget `_TUTORIAL_ENTRY_SEED` (the framing/dataset seed).**
  In `src/elspeth/web/composer/guided/profile.py`, replace `:29`:

  ```python
  _TUTORIAL_ENTRY_SEED = "Scrape the three synthetic project-brief pages from the list below and, for each, extract the project name, top risk, go-live date, and total cost into one JSON row."
  ```

  Confirm the structural profile test still passes (it asserts non-empty `str`,
  not the value):
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/test_profile.py -x -q
  ```
  Expected: all pass (the test pins shape, not value — ground truth §4 SURVIVES).

- [ ] **Step 6: Retarget `HELLO_WORLD_SESSION_TITLE` (e2e session-title tag).**
  `copy.ts:14` currently reads
  `export const HELLO_WORLD_SESSION_TITLE = "hello-world (cool government pages)";`
  — the "cool government pages" tag is the old scenario. Replace with:

  ```typescript
  export const HELLO_WORLD_SESSION_TITLE = "hello-world (synthetic project briefs)";
  ```

  > The orphan-cleanup prefix is `"hello-world ("` (see the `copy.ts` comment at
  > `:16-22` and `HELLO_WORLD_PENDING_SESSION_TITLE`); the retargeted title keeps
  > that prefix, so cleanup still matches. Do not change the prefix.

  Run the frontend tutorial component tests to confirm nothing pinned the old
  title string:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- src/components/tutorial --run
  ```
  Expected: pass. (If a component test asserts the literal `"cool government
  pages"`, update that assertion to the new title — it is an about-to-change
  value, contract Global Constraint "update, don't revert".)

- [ ] **Step 6a: Update the Turn-7 graduation test's hardcoded title literal.**
  `TutorialTurn7Graduation.test.tsx:124-127` asserts `renameSession` was called
  with the OLD title as a raw string literal (NOT the imported constant), so
  Step 6's retarget makes it fail (the component calls
  `renameSession(sessionId, HELLO_WORLD_SESSION_TITLE)`). Replace the literal:

  ```tsx
      expect(useSessionStore.getState().renameSession).toHaveBeenCalledWith(
        "sess-new",
        "hello-world (synthetic project briefs)",
      ),
  ```

  Run to confirm:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- TutorialTurn7Graduation.test.tsx --run
  ```
  Expected: pass. (The Step 6 `src/components/tutorial` run already exercises this
  file; this step makes the required literal edit explicit so it is not left
  unstaged at Step 8.)

- [ ] **Step 7: Reconcile the backend cache-path tests in `test_tutorial_service.py`.**
  `tests/unit/web/composer/test_tutorial_service.py` constructs cache entries
  with `canonical_prompt=CANONICAL_SEED_PROMPT` (`:247`) and example rows like
  `{"url": "https://example.gov", "rating": 5}` (`:250`). These exercise the
  cache MACHINERY (key derivation, replay), not the scenario wording, so they
  pass with the retargeted constant unchanged (they import the symbol, not the
  literal). Run them to confirm:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_tutorial_service.py -x -q
  ```
  Expected: pass. If a test asserts a literal scenario word ("rate"/"cool"/"gov")
  rather than the imported `CANONICAL_SEED_PROMPT` symbol, update only that
  assertion (about-to-change value). Do NOT weaken a cache-machinery assertion.

- [ ] **Step 8: Commit (the lockstep set in ONE commit).**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/preferences/tutorial_cache.py \
    src/elspeth/web/composer/guided/profile.py \
    src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts \
    src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts \
    src/elspeth/web/frontend/src/components/tutorial/copy.ts \
    src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx \
    tests/unit/web/preferences/test_tutorial_cache.py && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit -m "$(cat <<'EOF'
feat(tutorial): retarget canonical scenario to synthetic-scrape extraction

Move CANONICAL_SEED_PROMPT (backend) + its byte-identical frontend
mirror CANONICAL_TUTORIAL_PROMPT + _TUTORIAL_ENTRY_SEED +
HELLO_WORLD_SESSION_TITLE from "rate 5 gov homepages" to "scrape 3
synthetic project briefs and extract name / top risk / go-live date /
total cost". The constants carry the scripted INTENT only; the runtime
synthetic URLs come from the p4 resolver. The canonical-prompt cache
becomes vestigial (URLs are runtime-derived so the effective prompt is
no longer byte-stable) — kept, not deleted. NO service restart and NO
composer_skill_hash re-bake: the recipe/skill are scenario-agnostic.

Operator-owed: clear stale {data_dir}/tutorial_cache/ files (they miss
on the new model_id; documented artifact-delete pattern).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 4: Tutorial teaching-moment copy (assumption callout + shield-override caveat)

The worked example narrates two load-bearing lessons (spec §"Teaching moments"):
(1) "the LLM made an assumption about X" — naming a reviewable inference (e.g.
summing the cost lines, choosing the top risk); (2) "forcing an override of the
prompt-shield warning" — safe HERE only because we control the inputs. Both are
STATIC tutorial copy (deterministic UI), asserted at the right step. This is
distinct from asserting LLM prompt strings (project doctrine: skill behavior is
verified by re-running staging, not by grepping prompts).

> p4 only adds COPY and asserts it RENDERS. p3 produces the shield State-C result;
> p2 owns the panel layout. p4 does not implement either.

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/copy.ts` — add the two
  copy constants (after `TURN_7_LEARNING_BULLETS`, `:76`).
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.tsx`
  — render the assumption callout (the audit-story turn already narrates what the
  LLM did; it is the natural home for "the LLM made an assumption").
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.tsx`
  — render the shield-override caveat (Turn 4 is the run, where the State-C
  override is forced on screen).
- Create: `src/elspeth/web/frontend/src/components/tutorial/teachingMoments.test.tsx`

**Interfaces:**
- Consumes: existing turn components `TutorialTurn4Run` / `TutorialTurn5AuditStory`
  and their props (no new prop needed — the copy is static).
- Produces: `TUTORIAL_ASSUMPTION_CALLOUT` and `TUTORIAL_SHIELD_OVERRIDE_CAVEAT`
  string constants, rendered at the audit-story and run turns respectively.

- [ ] **Step 1: Write the failing component test that both copies render.**
  The exact prop signatures + required API mocks are VERIFIED against the real
  files (`TutorialTurn4Run.tsx:7-19` requires `sessionId`, `prompt`,
  `onCompleted`, `onCancelled`, optional `onBack`, and calls
  `runTutorialPipeline`; `TutorialTurn5AuditStory.tsx:8-13` requires `sessionId`,
  `runId`, `onContinue`, `onBack`, and calls `getRunAuditSummary`). The mock
  shape mirrors the existing `TutorialTurn4Run.discard.test.tsx:6,16-25`.
  Create `src/elspeth/web/frontend/src/components/tutorial/teachingMoments.test.tsx`:

  ```tsx
  import { render, screen, waitFor } from "@testing-library/react";
  import { beforeEach, describe, expect, it, vi } from "vitest";
  import * as api from "@/api/client";
  import {
    TUTORIAL_ASSUMPTION_CALLOUT,
    TUTORIAL_SHIELD_OVERRIDE_CAVEAT,
  } from "./copy";
  import { TutorialTurn4Run } from "./TutorialTurn4Run";
  import { TutorialTurn5AuditStory } from "./TutorialTurn5AuditStory";

  vi.mock("@/api/client", () => ({
    runTutorialPipeline: vi.fn(),
    getRunAuditSummary: vi.fn(),
  }));

  function noop(): void {}

  describe("tutorial teaching moments — static copy", () => {
    it("the assumption callout names the reviewable inference", () => {
      // Must teach that an LLM transform makes REVIEWABLE assumptions you can
      // correct — not "assumptions are fine, ignore them".
      expect(TUTORIAL_ASSUMPTION_CALLOUT.toLowerCase()).toContain("assum");
      expect(TUTORIAL_ASSUMPTION_CALLOUT.toLowerCase()).toContain("review");
    });

    it("the shield-override caveat scopes the override to controlled inputs", () => {
      // The override is acceptable ONLY because we control the inputs — it must
      // NOT read as a general "skip the shield" habit.
      const lower = TUTORIAL_SHIELD_OVERRIDE_CAVEAT.toLowerCase();
      expect(lower).toContain("control");
      expect(lower).toContain("synthetic");
      expect(lower).toContain("high-risk");
    });
  });

  describe("tutorial teaching moments — render at the right turn", () => {
    beforeEach(() => {
      vi.clearAllMocks();
    });

    it("Turn 5 (audit story) renders the assumption callout", async () => {
      // Realistic RunAuditStoryResponse shape: once it resolves non-null the
      // component renders the summary-gated block (TutorialTurn5AuditStory.tsx:64-92),
      // which reads source_data_hash / llm_call_count / run_id / started_at and
      // calls formatPluginVersions(summary.plugin_versions) -> Object.entries(...).
      // plugin_versions MUST be an object ({}), not undefined, or Object.entries
      // throws a TypeError at render and masks the callout assertion.
      vi.mocked(api.getRunAuditSummary).mockResolvedValue({
        source_data_hash: "abc123",
        llm_call_count: 1,
        run_id: "r1",
        started_at: new Date().toISOString(),
        plugin_versions: {},
      } as unknown as Awaited<ReturnType<typeof api.getRunAuditSummary>>);
      render(
        <TutorialTurn5AuditStory
          sessionId="sess-1"
          runId="run-1"
          onContinue={noop}
          onBack={noop}
        />,
      );
      await waitFor(() =>
        expect(screen.getByText(TUTORIAL_ASSUMPTION_CALLOUT)).toBeInTheDocument(),
      );
    });

    it("Turn 4 (run) renders the shield-override caveat", () => {
      vi.mocked(api.runTutorialPipeline).mockResolvedValue({
        run_id: "run-1",
        output: { rows: [], source_data_hash: "h", discarded_row_count: 0 },
        seeded_from_cache: false,
        cache_key: null,
      } as unknown as Awaited<ReturnType<typeof api.runTutorialPipeline>>);
      render(
        <TutorialTurn4Run
          sessionId="sess-1"
          prompt="any"
          onCompleted={noop}
          onCancelled={noop}
        />,
      );
      // The caveat is static pre-flight copy and must render synchronously,
      // before the run resolves.
      expect(screen.getByText(TUTORIAL_SHIELD_OVERRIDE_CAVEAT)).toBeInTheDocument();
    });
  });
  ```

  Run to fail:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- teachingMoments.test.tsx --run
  ```
  Expected failure: import error — `TUTORIAL_ASSUMPTION_CALLOUT` /
  `TUTORIAL_SHIELD_OVERRIDE_CAVEAT` are not exported from `copy.ts`.

- [ ] **Step 2: Add the two copy constants.**
  In `src/elspeth/web/frontend/src/components/tutorial/copy.ts`, after
  `TURN_7_LEARNING_BULLETS` (`:76`), add:

  ```typescript
  /**
   * Teaching moment 1 (spec §"Teaching moments"): names that the LLM transform
   * made a REVIEWABLE assumption (e.g. summing the cost lines, choosing the
   * top-impact risk). Worded so the learner does not over-generalise into
   * "assumptions are fine, ignore them": the assumption is surfaced, not hidden,
   * and is correctable via the intent box.
   */
  export const TUTORIAL_ASSUMPTION_CALLOUT =
    "The LLM made an assumption here — it summed the per-line costs into a total " +
    "and picked the highest-impact entry as the top risk. This is exactly the " +
    "kind of inference you review: every assumption is surfaced in the audit " +
    "trail, and you can correct it by telling the composer what you meant.";

  /**
   * Teaching moment 2 (spec §"Teaching moments"): the prompt-shield State-C
   * override. Acceptable HERE only because the inputs are controlled (our own
   * synthetic pages). Must NOT read as a general "skip the shield" habit —
   * names the trust assumption out loud rather than letting it ride as an
   * invisible default.
   */
  export const TUTORIAL_SHIELD_OVERRIDE_CAVEAT =
    "We are proceeding without a prompt shield in this one case — and only " +
    "because we control the inputs: these are our own synthetic test pages. " +
    "Running an LLM over fetched content without a shield is always a high-risk " +
    "decision, not a default. Against real or untrusted web content you would " +
    "wire the shield.";
  ```

- [ ] **Step 3: Render the assumption callout in Turn 5 (audit story).**
  In `TutorialTurn5AuditStory.tsx`, extend the `import { TURN_5_PRIMARY_BUTTON }
  from "./copy";` line (`:4`) to also import `TUTORIAL_ASSUMPTION_CALLOUT`, and
  render it as an advisory callout immediately after the existing intro paragraph
  (the `<p>The LLM made a judgment call on each page...</p>` at `:50-53`):
  ```tsx
        <p className="tutorial-callout">{TUTORIAL_ASSUMPTION_CALLOUT}</p>
  ```
  Keep advisory styling; do not add a blocking control.

- [ ] **Step 4: Render the shield-override caveat in Turn 4 (run).**
  In `TutorialTurn4Run.tsx`, extend the `import { TUTORIAL_RUN_PREAMBLE,
  TURN_4_PRIMARY_BUTTON } from "./copy";` line (`:4`) to also import
  `TUTORIAL_SHIELD_OVERRIDE_CAVEAT`, and render it as a static pre-flight advisory
  alongside the existing `TUTORIAL_RUN_PREAMBLE` in the turn's JSX body (find
  where `TUTORIAL_RUN_PREAMBLE` is rendered and add a sibling
  `<p className="tutorial-callout">{TUTORIAL_SHIELD_OVERRIDE_CAVEAT}</p>`). It must
  render synchronously (not gated on the run resolving), so the State-C override
  is justified on screen before the run proceeds.

- [ ] **Step 4a: Retarget the stale OLD-scenario "scoring" prose in the two
  files this task already edits.**
  Both turn components still narrate the retired 5-gov-pages SCORING scenario;
  the synthetic-extraction scenario has no "score". No sibling task retargets this
  in-component prose (Task 3 = constants, Task 5 = harness, Task 8 =
  `TutorialGuidedShell`), so it is p4-Task-4-owned.
  - In `TutorialTurn5AuditStory.tsx` (`:93-97`), the closing audit paragraph reads
    `If someone asks why a page received its score, the run has the prompt,
    response, model details, source hash, and plugin versions tied together.`
    Retarget to extraction-accurate phrasing, e.g.:
    ```tsx
            <p>
              If someone asks how the LLM derived a project's facts — the top
              risk it picked or the total it summed — the run has the prompt,
              response, model details, source hash, and plugin versions tied
              together.
            </p>
    ```
  - In `TutorialTurn4Run.tsx` (`preferredColumns`, `:308`), the curated column
    list is the old rating scenario's
    `["url", "page", "title", "score", "coolness", "rationale", "error"]` — none of
    which are the derived facts. Retarget to the extraction output fields:
    ```tsx
      const preferred = ["url", "project_name", "top_risk", "key_date", "total_cost", "error"];
    ```
  - Add a Turn-5-wording assertion to `teachingMoments.test.tsx` so the
    extraction-accurate phrasing is guarded (the synthetic-extraction worked
    example must not regress to "score" copy): in the Turn-5 render test, after the
    `TUTORIAL_ASSUMPTION_CALLOUT` assertion, also assert the audit paragraph names
    the derived facts and NOT "score", e.g.:
    ```tsx
        expect(screen.queryByText(/received its score/i)).not.toBeInTheDocument();
        expect(screen.getByText(/top risk it picked or the total it summed/i)).toBeInTheDocument();
    ```

- [ ] **Step 5: Run to pass.**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- teachingMoments.test.tsx --run
  ```
  Expected: 4 passed. Then run the existing turn tests to confirm no regression:
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- src/components/tutorial --run
  ```
  Expected: pass (update any sibling turn test only if it asserts the prior body
  text shape — about-to-change value).

- [ ] **Step 6: Commit.**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/frontend/src/components/tutorial/copy.ts \
    src/elspeth/web/frontend/src/components/tutorial/TutorialTurn4Run.tsx \
    src/elspeth/web/frontend/src/components/tutorial/TutorialTurn5AuditStory.tsx \
    src/elspeth/web/frontend/src/components/tutorial/teachingMoments.test.tsx && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit -m "$(cat <<'EOF'
feat(tutorial): add assumption + shield-override teaching-moment copy

Two static teaching moments narrate the worked example: (1) the LLM made
a REVIEWABLE assumption (summing costs, picking top risk) — surfaced, not
hidden, correctable via the intent box; (2) the prompt-shield State-C
override is acceptable HERE only because we control the inputs (synthetic
pages), and is always a high-risk decision otherwise. Rendered at Turn 5
(audit story) and Turn 4 (run); asserted as static UI copy, not LLM
prompt strings (skill behavior is verified by re-running staging).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 5: Retarget the staging-reliability harness rubric + verify

The staging harness drives the live tutorial and grades it. The coupled triple
(`FIXED_PROMPT` + `ASSUMPTION_RUBRIC` + `JUDGE_RUBRIC`) must move from the
5-gov-pages scoring scenario to the 3-synthetic-pages extraction scenario, and
`ASSUMPTION_RUBRIC` must now EXPECT `prompt_injection_shield_recommendation`
(p3's always-on shield fires every run — the harness thereby verifies p3's
component end-to-end).

> The `.staging.spec.ts` cannot "run to PASS" in CI — it requires a live staging
> origin + credentials (operator env). p4's verification for this task is
> TypeScript type-checking + lint of the harness file, plus an operator-owed live
> run (the P7-delivery known-gap pattern). Do NOT add a CI gate that runs the
> staging spec.

**Files:**
- Modify: `src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts`
  (`FIXED_PROMPT` `:17-25`, `ASSUMPTION_RUBRIC` `:30-60`, `JUDGE_RUBRIC` `:63-78`,
  bump `HARNESS_VERSION` `:4`).
- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`
  (consumes the rubric; the 5-source heuristics `:164-178` + `minReachableSources`
  expectations retarget to 3 pages + the shield-state expectation).

**Interfaces:**
- Consumes: the retargeted scenario constants (Task 3) — the harness `FIXED_PROMPT`
  is deliberately a DIFFERENT string from `CANONICAL_TUTORIAL_PROMPT` so its
  SHA-256 misses the cache (fresh live compose each run).
- Produces: the retargeted rubric the staging spec grades against.

- [ ] **Step 1: Retarget `FIXED_PROMPT` + bump `HARNESS_VERSION`.**
  In `prompt-and-rubric.ts`, bump `HARNESS_VERSION` (`:4`) to `"2.0.0"` (a
  scenario change is a major rubric revision), and replace `FIXED_PROMPT`
  (`:17-25`) with a semantically-equal-but-textually-distinct synthetic-scrape
  EXTRACTION prompt (distinct so the cache misses):

  ```typescript
  export const HARNESS_VERSION = "2.0.0";

  // Semantically equal to CANONICAL_TUTORIAL_PROMPT (the synthetic-scrape
  // EXTRACTION task) but a DIFFERENT string, so SHA-256(prompt+model_id) misses
  // the tutorial cache (is_canonical_prompt=false in tutorial_service.py) ->
  // fresh live composition every run.
  //
  // EXTRACTION (not scoring): the composer reads authored tables and derives
  // facts (name / top risk / go-live date / summed total cost). Extraction is
  // not subjective, so vague_term is NOT expected here (contrast the old scoring
  // prompt). The always-on prompt-shield (p3) fires every run regardless, so
  // prompt_injection_shield_recommendation IS expected.
  //
  // The prompt MUST carry the 3 concrete scrape targets (mirroring the old
  // prompt's 5 literal gov URLs): driveGuidedWalk seeds FIXED_PROMPT as the
  // sole driving message (:106/:115), and the Tier-1 no-fabrication source
  // driver (p1, contract §2.2) builds rows ONLY from concrete URLs present IN
  // the message — it never invents them. A URL-less prompt yields zero scrape
  // targets, so the run produces no rows and JUDGE_RUBRIC.minReachableSources:3
  // / minSubstantiveRows:3 become UNSATISFIABLE (dim-c/dim-d permanently fail).
  // These are staging-absolute URLs because this is a .staging.spec.ts driving
  // the live staging origin; there is no harness base-URL constant to resolve
  // {base} from. If a deployment serves the tutorial pages elsewhere, change
  // these three literals to that origin.
  export const FIXED_PROMPT =
    "Fetch each of these three synthetic project-brief pages and use an LLM " +
    "to read the tables and produce, per page, a JSON row containing the " +
    "project name, the single highest-impact risk together with its " +
    "mitigation, the go-live milestone date, and the total cost computed by " +
    "summing the cost line items. Drop the raw HTML and write the rows to a " +
    "JSON file.\n" +
    "https://elspeth.foundryside.dev/tutorial-site/project-1.html\n" +
    "https://elspeth.foundryside.dev/tutorial-site/project-2.html\n" +
    "https://elspeth.foundryside.dev/tutorial-site/project-3.html";
  ```

- [ ] **Step 2: Retarget `ASSUMPTION_RUBRIC` (drop vague_term; expect the shield
  recommendation).**
  Replace `ASSUMPTION_RUBRIC` (`:30-60`) with:

  ```typescript
  // Dimension (c): which interpretation kinds the composer SHOULD raise vs NOT.
  // Graded on kind, not exact wording. InterpretationKind values come from the
  // backend enum (kind field on InterpretationEventResponse):
  // vague_term | invented_source | llm_prompt_template | pipeline_decision | llm_model_choice.
  export const ASSUMPTION_RUBRIC = {
    // EXTRACTION, not scoring: there is no subjective rating criterion, so
    // vague_term must NOT be expected. The always-on prompt-shield (p3) fires
    // for every LLM node over fetched content, so the pipeline_decision review
    // with user_term=prompt_injection_shield_recommendation IS expected — this
    // is how the harness verifies Component 2 (the 3-state shield) end-to-end.
    expectVerify: ["prompt_injection_shield_recommendation"] as const,
    // The composer MAY also stage a prompt-template review for the LLM node;
    // acceptable-but-not-required (neither under- nor over-flagging).
    allowOptional: ["llm_prompt_template"] as const,
    // Over-flagging: the USER named the 3 pages and the 4 fields explicitly, so
    // raising an invented_source review, or a review on those stated targets, is
    // an over-flag. MIXED targets: "invented_source" is an InterpretationKind
    // (matched on the event KIND); "project_name"/"total_cost" are user_terms
    // (matched on the review's user_term). Step 4b below makes the overFlagged
    // grader match EITHER kind OR user_term so the kind-valued entry can fire —
    // without it, the user_term-only grader at :380-383 can NEVER match
    // invented_source (a dead check, the exact class :374-379 was written to
    // eliminate).
    overFlagTerms: ["invented_source", "project_name", "total_cost"] as const,
    // Patterns are ANCHORED so they cannot spuriously fail dim-c on a legitimate
    // review. /\bproject[_\s-]?name\b/i matches only the field name "project_name"
    // (not e.g. "the project's name as written"), and /\btotal[_\s-]?cost\b/i drops
    // the bare "sum" alternative which would substring-match "assume"/"consume"/
    // "summary"/"summarize" in legitimate llm_prompt_template / allowOptional
    // reviews. Confirm the expected prompt_injection_shield_recommendation and the
    // allowOptional llm_prompt_template user_terms match NO over-flag pattern.
    overFlagTermPatterns: [/invent|fabricat/i, /\bproject[_\s-]?name\b/i, /\btotal[_\s-]?cost\b/i] as const,
  };
  ```

  > `expectVerify` is the kind/term the harness REQUIRES the run to surface;
  > `prompt_injection_shield_recommendation` is the `user_term` p3 stages on the
  > `pipeline_decision` review (CONFIRMED: `interpretation_state` user_term +
  > `InterpretationKind.PIPELINE_DECISION`; the live skill pairs them at
  > `pipeline_composer.md:676,920`). The staging spec's dim-c `underFlagged`
  > classifier today matches `expectVerify` ONLY against the event `kind`
  > (`tutorial-reliability.staging.spec.ts:370-372`: `raisedKinds = events.map(e => e.kind)`),
  > so the `user_term`-valued `prompt_injection_shield_recommendation` would NEVER
  > be found and dim-c would PERMANENTLY FAIL even when the shield fires. Step 4a
  > below CHANGES that grader to match each `expectVerify` entry against either the
  > event `kind` OR its `user_term` (symmetric with the existing `overFlagged`
  > `user_term` match at `:380-383`). This keeps `expectVerify` precise (it verifies
  > the SPECIFIC shield review fired, not merely that some `pipeline_decision`
  > occurred), which is what "the harness verifies Component 2 end-to-end" requires.

- [ ] **Step 3: Retarget `JUDGE_RUBRIC` (3 pages; grade derived fields).**
  Replace `JUDGE_RUBRIC` (`:63-78`) with:

  ```typescript
  // Dimension (d): the judge rubric the agent applies to recorded output rows.
  export const JUDGE_RUBRIC = {
    // Mechanical (computed in the spec/aggregator, NOT by the judge):
    minReachableSources: 3, // all 3 synthetic pages must have fetched (scrape op status ok)
    maxDiscardedRows: 0, // discarded_row_count from TutorialRunOutput
    minSubstantiveRows: 3, // all 3 rows carry the derived fields
    // Judge-scored (0..1), pass threshold:
    judgePassThreshold: 0.7,
    // The judge question (structured), applied per batch over recorded rows:
    judgePrompt:
      "Given the task (per synthetic project-brief page, extract project_name, " +
      "top_risk = the highest-impact risk and its mitigation, key_date = the " +
      "go-live milestone, and total_cost = the SUM of the cost line items) and " +
      "these output rows, score 0..1: does each row carry a non-empty " +
      "project_name, a top_risk that names a real risk + mitigation, a plausible " +
      "go-live date, and a total_cost that equals the sum of the page's cost " +
      "lines? Penalise degenerate output — missing/null fields, a total_cost " +
      "that is a single line item rather than the sum, or a top_risk that is not " +
      "the highest-impact row.",
  };
  ```

- [ ] **Step 4a: Fix the dim-c `underFlagged` grader to match `expectVerify`
  against `user_term` as well as `kind` (REQUIRED — without this dim-c
  permanently fails).**
  In `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`,
  the dim-c classifier currently matches `expectVerify` ONLY against the event
  `kind` (`:370-373`):

  ```typescript
  const raisedKinds = events.map((e) => e.kind);
  const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter(
    (k) => !raisedKinds.includes(k),
  );
  ```

  Because `expectVerify = ["prompt_injection_shield_recommendation"]` is a
  `user_term` (its `kind` is `pipeline_decision`), `raisedKinds.includes(k)` can
  never be true and `underFlagged` always contains it → dim-c FAILS even when the
  shield fires correctly. Replace the two statements above with a match against
  EITHER the event `kind` OR its `user_term` (symmetric with the existing
  `overFlagged` `user_term` match at `:380-383`):

  ```typescript
  // expectVerify entries may be an InterpretationKind (e.g. "pipeline_decision")
  // OR a user_term (e.g. "prompt_injection_shield_recommendation", the shield
  // review's discriminating signal whose kind is pipeline_decision). Match on
  // either, so the shield review is recognised by its user_term. Kinds are enum
  // values and user_terms are field-path strings, so the OR cannot false-match.
  const underFlagged = ASSUMPTION_RUBRIC.expectVerify.filter(
    (k) => !events.some((e) => e.kind === k || e.user_term === k),
  );
  ```

  > `raisedKinds` is deleted here; confirm it has no other consumer
  > (`grep -n "raisedKinds" src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`
  > should now report no hits — the diagnostic map at `:447` builds its own
  > `events.map((e) => ({ kind: e.kind, term: e.user_term }))` and is unaffected).
  > Verify with `npx tsc --noEmit` in Step 5.

- [ ] **Step 4b: Fix the dim-c `overFlagged` grader to match kind-valued
  entries against `e.kind` as well as `e.user_term` (REQUIRED — without this the
  `invented_source` over-flag check is dead).**
  In `src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts`,
  the `overFlagged` classifier currently matches the pattern against the event
  `user_term` ONLY (`:380-383`):

  ```typescript
  const overFlagged = ASSUMPTION_RUBRIC.overFlagTerms.filter((_label, i) => {
    const pattern = ASSUMPTION_RUBRIC.overFlagTermPatterns[i];
    return events.some((e) => typeof e.user_term === "string" && pattern.test(e.user_term));
  });
  ```

  `overFlagTerms` now mixes a kind-valued entry (`invented_source`) with
  user_term-valued entries (`project_name`/`total_cost`). A real
  `invented_source` over-flag carries `kind="invented_source"` with a DIFFERENT
  `user_term`, so a user_term-only match can never fire `/invent|fabricat/i` →
  the `invented_source` over-flag is undetectable (a dead check). Replace the
  classifier above with one that tests the pattern against EITHER the event
  `kind` OR its `user_term` (symmetric with the Step 4a `underFlagged` fix):

  ```typescript
  // overFlagTerms mixes InterpretationKind-valued entries (e.g.
  // "invented_source", whose pattern /invent|fabricat/i matches the KIND) with
  // user_term-valued entries (e.g. "project_name"/"total_cost", matched on the
  // review's user_term). Test each term's pattern against EITHER the event kind
  // OR its user_term so the kind-valued entry is recognised. Kinds are enum
  // values and user_terms are field-path strings, so the OR cannot false-match
  // across the two namespaces.
  const overFlagged = ASSUMPTION_RUBRIC.overFlagTerms.filter((_label, i) => {
    const pattern = ASSUMPTION_RUBRIC.overFlagTermPatterns[i];
    return events.some(
      (e) =>
        (typeof e.kind === "string" && pattern.test(e.kind)) ||
        (typeof e.user_term === "string" && pattern.test(e.user_term)),
    );
  });
  ```

  > This is the symmetric counterpart to Step 4a: Step 4a fixed the
  > kind-OR-user_term gap on the `underFlagged` (expectVerify) side; Step 4b
  > fixes it on the `overFlagged` (overFlagTerms) side now that the rubric adds a
  > kind-valued over-flag term. Verify with `npx tsc --noEmit` in Step 5.

- [ ] **Step 4: Retarget the staging spec's input-key heuristic.**
  VERIFIED: the spec already reads source count from `JUDGE_RUBRIC.minReachableSources`
  (`:416,442`, now 3 via Step 3) and `substantiveRowCount` reads the per-run
  `sourceInputKeys` dynamically (`:388`) — so the count assertions follow the
  rubric automatically with NO spec edit. The ONE spec-local edit is the
  `KNOWN_INPUT_KEYS` regex (`:165`), which lists the OLD scenario's input keys:

  ```typescript
  const KNOWN_INPUT_KEYS = /^(?:url|source|agency|abuse_contact|scraping_reason|html|html_content|raw_html)$/i;
  ```

  Replace it with the synthetic scenario's INPUT keys (the keys that are NOT
  extraction output — `substantiveRowCount` must NOT count them):

  ```typescript
  const KNOWN_INPUT_KEYS = /^(?:url|source|html|html_content|raw_html|content|content_fingerprint)$/i;
  ```

  > Do NOT add the DERIVED fields (`project_name`, `top_risk`, `key_date`,
  > `total_cost`) to `KNOWN_INPUT_KEYS` — those are exactly the extraction output
  > `substantiveRowCount` must COUNT. `abuse_contact`/`scraping_reason`/`agency`
  > are dropped because they are no longer in the output rows (the `field_mapper`
  > drops everything but `url` + the rated/derived fields). Confirm no other
  > literal `5`/"five" in the spec means "five sources" (the
  > `420_000`/`360_000`/`900_000` timeouts are unrelated — leave them):
  > ```
  > grep -n "five\| 5 \|of 5\|/5\b" src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts
  > ```
  > Retarget any that remain to 3. Do not change the staging URL/auth plumbing.

- [ ] **Step 5: Verify by type-check + lint (NOT by running the staging spec).**
  ```
  cd /home/john/elspeth/src/elspeth/web/frontend && npx tsc --noEmit
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run lint
  ```
  Expected: `tsc` reports no errors in the harness/spec files; lint passes. The
  live staging run is operator-owed (P7 known-gap pattern): the operator runs
  `tutorial-reliability.staging.spec.ts` against staging to verify the live
  shield-state + extraction behavior. Note this as an operator chore.

- [ ] **Step 6: Confirm the docs harness-plan test still passes (it only checks
  for credential leakage, not rubric shape — should be untouched).**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/docs/test_tutorial_reliability_harness_plan.py -x -q
  ```
  Expected: pass (it asserts only that the plan markdown does not embed
  `dta_user`/`dta_pass`).

- [ ] **Step 7: Commit.**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/frontend/tests/e2e/harness/prompt-and-rubric.ts \
    src/elspeth/web/frontend/tests/e2e/tutorial-reliability.staging.spec.ts && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit -m "$(cat <<'EOF'
test(tutorial): retarget staging harness rubric to synthetic extraction

Move FIXED_PROMPT + ASSUMPTION_RUBRIC + JUDGE_RUBRIC from the 5-gov-pages
scoring scenario to the 3-synthetic-pages extraction scenario. Drop the
vague_term expectation (extraction is not subjective); EXPECT
prompt_injection_shield_recommendation now that p3's shield review fires
every run (the harness thereby verifies the 3-state shield end-to-end).
JUDGE_RUBRIC grades the derived fields (project_name / top_risk /
key_date / summed total_cost); minReachableSources 5 -> 3. Bump
HARNESS_VERSION 1.0.0 -> 2.0.0.

Operator-owed: run tutorial-reliability.staging.spec.ts against staging
to verify live shield-state + extraction (live run is operator-env).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 6: Add the optional `allowed_hosts` SSRF slot to the web-scrape recipe

The `web-scrape-llm-rate-jsonl` recipe builder (`recipes.py:687-710`) emits no
`allowed_hosts`, so the `web_scrape` node defaults to `"public_only"`
(`web_scrape.py:129`). On a public base that is correct; on a **loopback dev
base the tutorial scrape fails** (loopback is blocked by `public_only`) — the
spec-mandated loopback case. This task adds an OPTIONAL `allowed_hosts` slot that
the builder emits into the node ONLY when non-empty, so every existing caller is
unchanged (no slot value → no key → `public_only`) and the tutorial loopback path
gets the tight CIDR list from the Task 2 resolver.

> **This is the one application point p4 owns end-to-end.** Adding an SSRF-control
> slot is p4's security concern (the contract leaves `allowed_hosts` unthreaded).
> The slot is ADDITIVE and behaviour-preserving. The slot VALUE for the tutorial
> run is supplied by p4 itself (NOT deflected to p1): Task 8 threads
> `resolve_tutorial_allowed_hosts(...)` into the match's `allowed_hosts` slot at
> the STEP_2.5 recipe-offer-accept seam (`sessions/routes/composer/guided.py`
> accept dispatch → `handle_step_2_5_recipe_apply`, `composer/guided/steps.py:223`)
> via `dataclasses.replace` on the frozen `RecipeMatch` (or an extra-slots arg),
> mapping the scalar `"public_only"` to an empty list (omit). p4 makes the slot
> EXIST and tested here (Task 6); p4 injects the tutorial value (Task 8).

**Files:**
- Modify: `src/elspeth/web/composer/recipes.py` — add an `allowed_hosts` `SlotSpec`
  to `_RECIPE_WEB_SCRAPE_SLOTS` (`:592-645`); emit it conditionally in
  `_build_web_scrape_recipe` (`:687-710`, the `web_scrape` node's `options`).
- No edit: `src/elspeth/web/composer/guided/recipe_match.py` — `_web_scrape_slot_resolver`
  (`:266-300`) returns only `source_blob_id`/`source_plugin`/`output_path`; the
  new slot is `required=False, default=()`, so `validate_slots` supplies the empty
  default and existing matches are unchanged. NO diff expected here (verified in
  Step 4 by running the recipe-match tests, not by editing this file).
- Create: `tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py`

**Interfaces:**
- Consumes: `SlotType` includes `"str_list"` (`contracts/composer_slots.py:15`);
  `SlotSpec(slot_type=..., required=..., default=..., description=...)`;
  `validate_slots(recipe, raw_slots)` (`recipes.py:122`); `get_recipe(name)`
  (`recipes.py:856`). `resolve_tutorial_allowed_hosts(*, base_url)` (Task 2).
- Produces: an `allowed_hosts: str_list` recipe slot (default `()`); the
  `web_scrape` node carries `options["allowed_hosts"]` IFF the slot list is
  non-empty.

> **Why `str_list` with empty-default-omit rather than a scalar `"public_only"`
> default:** `web_scrape.allowed_hosts` is
> `Literal["public_only","allow_private"] | list[CidrStr]`. The current node omits
> the key entirely and relies on the field default. Emitting `"public_only"`
> explicitly would be equivalent but changes every existing recipe's emitted args
> (and any golden-output test). Empty-list-→-omit preserves the EXACT current
> emitted shape for non-tutorial callers; the tutorial passes a non-empty CIDR
> list, which is the only `list` arm the recipe ever needs (the public case stays
> on the omitted default).

- [ ] **Step 1: Write the failing test (omit by default; emit CIDR list when
  supplied; public resolver → empty → omit).**
  Create `tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py`:

  ```python
  """Phase p4 Task 6 — the optional allowed_hosts SSRF slot on the web-scrape recipe.

  Behaviour-preserving: no slot value -> the web_scrape node omits allowed_hosts
  (the field default "public_only" applies, the current behaviour). A non-empty
  CIDR list (loopback dev) -> the node carries it. The list comes from the Task 2
  resolver, NEVER from the LLM (SSRF control).
  """

  from __future__ import annotations

  from elspeth.web.composer.recipes import get_recipe, validate_slots
  from elspeth.web.composer.tutorial_sample import resolve_tutorial_allowed_hosts


  _BASE_SLOTS = {
      "source_blob_id": "00000000-0000-0000-0000-000000000001",
      "source_plugin": "json",
      "model": "anthropic/claude-sonnet-4.6",
      "api_key_secret": "OPENROUTER_API_KEY",
      "provider": "openrouter",
      "rating_template": "Extract fields and return JSON.",
      "abuse_contact": "noreply@dta.gov.au",
      "scraping_reason": "DTA technical demonstration",
      "output_path": "outputs/out.jsonl",
  }


  def _web_scrape_node(args: dict) -> dict:
      return next(n for n in args["nodes"] if n["plugin"] == "web_scrape")


  def test_default_omits_allowed_hosts() -> None:
      recipe = get_recipe("web-scrape-llm-rate-jsonl")
      assert recipe is not None
      slots = validate_slots(recipe, dict(_BASE_SLOTS))
      args = recipe.build(slots)
      node = _web_scrape_node(args)
      # Behaviour-preserving: no allowed_hosts key -> field default public_only.
      assert "allowed_hosts" not in node["options"]


  def test_loopback_cidr_list_is_emitted() -> None:
      recipe = get_recipe("web-scrape-llm-rate-jsonl")
      assert recipe is not None
      hosts = resolve_tutorial_allowed_hosts(base_url="http://127.0.0.1:5173")
      assert isinstance(hosts, list)
      slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": hosts})
      args = recipe.build(slots)
      node = _web_scrape_node(args)
      assert node["options"]["allowed_hosts"] == ["127.0.0.1/32", "::1/128"]


  def test_public_resolver_yields_omit() -> None:
      # The public resolver returns "public_only" (scalar), which the tutorial
      # threading maps to an EMPTY slot list -> omit (field default applies).
      recipe = get_recipe("web-scrape-llm-rate-jsonl")
      assert recipe is not None
      hosts = resolve_tutorial_allowed_hosts(base_url="https://elspeth.foundryside.dev")
      assert hosts == "public_only"
      # public_only is the field default; the slot stays empty -> omit.
      slots = validate_slots(recipe, {**_BASE_SLOTS, "allowed_hosts": []})
      args = recipe.build(slots)
      node = _web_scrape_node(args)
      assert "allowed_hosts" not in node["options"]
  ```

  Run to fail:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py -x -q
  ```
  Expected failure: `validate_slots` rejects the unknown `allowed_hosts` slot OR
  the node never carries it (`KeyError`/assertion).

- [ ] **Step 2: Add the optional `allowed_hosts` slot.**
  In `src/elspeth/web/composer/recipes.py`, inside `_RECIPE_WEB_SCRAPE_SLOTS`
  (after `output_path`, `:644`), add:

  ```python
      "allowed_hosts": SlotSpec(
          slot_type="str_list",
          required=False,
          # Tuple default (not []): recipes.py warns a mutable list default would
          # silently bypass the frozen contract; the sibling str_list slot
          # required_input_fields uses default=() (recipes.py:190). _coerce_default
          # returns tuple(raw) for str_list, so the validated value is always a tuple.
          default=(),
          description=(
              "SSRF allowlist for the web_scrape node, as a list of CIDR strings. "
              "Empty (the default) omits the key so the web_scrape field default "
              "'public_only' applies — the correct value for a public host. Set by "
              "the tutorial seam to a tight loopback CIDR for local dev; NEVER set "
              "by the LLM (this is an SSRF control)."
          ),
      ),
  ```

- [ ] **Step 3: Emit the slot conditionally in the builder.**
  In `_build_web_scrape_recipe` (`recipes.py:648`), inside the `web_scrape`
  node's `options` dict (`:695-709`, currently ending with the `http` block),
  add a conditional emission. Build the node options, then add `allowed_hosts`
  only when the slot list is non-empty. Replace the `web_scrape` node literal
  (the `{"id": "url_rows", ...}` dict, `:688-710`) so its `options` is assembled
  with the conditional key:

  ```python
      web_scrape_options: dict[str, Any] = {
          "schema": {"mode": "observed"},
          "url_field": "url",
          "content_field": content_field,
          "fingerprint_field": fingerprint_field,
          "format": "markdown",
          "http": {
              # OPERATOR: these values are visible to scraped third parties.
              "abuse_contact": slots["abuse_contact"],
              "scraping_reason": slots["scraping_reason"],
          },
      }
      allowed_hosts = slots.get("allowed_hosts") or []
      if allowed_hosts:
          # SSRF allowlist supplied by the deterministic seam (tutorial loopback
          # CIDR). Empty -> omitted -> the web_scrape field default public_only.
          web_scrape_options["allowed_hosts"] = list(allowed_hosts)
  ```

  and reference `web_scrape_options` in the node:

  ```python
              {
                  "id": "url_rows",
                  "node_type": "transform",
                  "plugin": "web_scrape",
                  "input": "rows",
                  "on_success": "scraped",
                  "on_error": "discard",
                  "options": web_scrape_options,
              },
  ```

  > `slots.get("allowed_hosts")` is the validated, coerced value — `str_list`
  > with `default=[]`, so it is always a list. The `or []` guards a `None` only
  > defensively. Confirm `Any` is imported in `recipes.py` (it is — used
  > throughout, e.g. `:122`).

- [ ] **Step 4: Confirm the resolver needs NO edit (existing matches unchanged).**
  In `src/elspeth/web/composer/guided/recipe_match.py`, `_web_scrape_slot_resolver`
  (begins `:266`; the return map is at `:296-300`) returns a partial slot map. The
  slot is `required=False, default=()`, so `validate_slots` supplies the empty
  default (coerced to a tuple, see Step 2) when the resolver omits it — existing
  `match_recipe` flows are unchanged and this file needs NO edit. Confirm by
  running the existing recipe-match tests:
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/guided/ -k "recipe" -q
  ```
  Expected: pass (the additive optional slot does not change any existing match).
  If a golden-args test asserts the EXACT `web_scrape` node options and now sees a
  diff, it should NOT — empty default omits the key. If it diffs, the Step 3
  conditional is wrong (emitting on empty); fix Step 3, not the test.

- [ ] **Step 5: Run to pass.**
  ```
  cd /home/john/elspeth && uv run pytest tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py -x -q
  ```
  Expected: `3 passed`.

- [ ] **Step 6: Commit.**
  ```
  cd /home/john/elspeth && git add \
    src/elspeth/web/composer/recipes.py \
    tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py && SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier git commit -m "$(cat <<'EOF'
feat(recipe): add optional allowed_hosts SSRF slot to web-scrape recipe

Add an additive, behaviour-preserving allowed_hosts (str_list, default
[]) slot to web-scrape-llm-rate-jsonl. Empty default omits the node key
so the web_scrape field default public_only applies (every existing
caller unchanged); a non-empty CIDR list (the tutorial loopback case
from the p4 resolver) is emitted into the node. allowed_hosts is an SSRF
control set deterministically by the seam, never by the LLM. Without this
the tutorial scrape fails on a loopback dev base. The tutorial VALUE is
injected by p4 itself (Task 8) at the STEP_2.5 recipe-offer-accept seam in
guided.py; this task makes the slot exist and tested.

Note: adds a recipe slot, which changes the tutorial_model_id recipe_hash
(the already-vestigial canonical cache invalidates — no behaviour change).
No skill edit -> no composer_skill_hash re-bake, no service restart.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
  ```

---

## Task 7: Trust-boundary scan + slice reconciliation

The synthetic pages are new fetched content; the `web_scrape` transform is the
boundary. Confirm p4 introduced no new unguarded boundary (the LLM-driven apply
path consumes Tier-3 free text → validated config via the existing seams owned by
p1; p4 only adds the deterministic resolver + static content + copy).

**Files:** none (verification + reconciliation only).

- [ ] **Step 1: Run the wardline trust-boundary gate.**
  ```
  cd /home/john/elspeth && wardline scan . --fail-on ERROR
  ```
  Expected: exit 0 (clean). If a finding fires, fix it at the BOUNDARY (the
  `web_scrape` fetch / the resolver's host handling), not the sink — see the
  `wardline-gate` skill. The resolver deliberately fails loud on an
  undeterminable base (`ValueError`) and never widens egress to `allow_private`;
  if wardline flags the host parsing, that is the place to harden.

- [ ] **Step 2: Reconcile the full p4 slice diff at the slice boundary.**
  ```
  cd /home/john/elspeth && git log --oneline release/0.7.0 -7
  cd /home/john/elspeth && git diff --stat HEAD~6..HEAD
  ```
  Confirm the diff touches ONLY p4's owned files: the 3 synthetic pages,
  `config.py`, `tutorial_sample.py`, the 4 scenario constants + their value-assert
  tests, the 2 teaching-moment copy renders + test, the 2 harness files, and (Task
  6) the `allowed_hosts` slot in `recipes.py` + its test. (`recipe_match.py` is
  NOT edited — Task 6 Step 4 confirmed the resolver returns unchanged keys and the
  optional slot defaults itself; if `recipe_match.py` shows a diff, something is
  wrong.) Confirm NOTHING in the sibling slices is touched: `transform.py` (p3),
  `ChatPanel.tsx` / `isTutorial` (p2), `chat_solver.py` / `maybe_resolve_step_1_source_chat`
  / the `/guided/chat` apply contract in `guided.py` (p1), `pipeline_composer.md`
  (live skill). Task 6's `recipes.py` edit is bounded to the ADDITIVE
  `allowed_hosts` slot — confirm it touches no other recipe and no
  predicate/routing logic. If any sibling-slice change appears, it is out of scope
  — revert it.

  > **Task 7 reconciles the p4-CORE slice ONLY; Task 8 is a SEPARATE
  > post-p1/p2 consumer slice with its own reconciliation boundary.** Task 8
  > (the passive auto-drive) lands AFTER p1+p2 merge and owns files Task 7's
  > p4-core manifest does NOT list:
  > `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx`,
  > the NEW backend GET surface that exposes the Task 2 resolver output, and the
  > STEP_2.5 recipe-offer-accept edit in
  > `src/elspeth/web/sessions/routes/composer/guided.py` (the `allowed_hosts`
  > value injection). Task 7 must NOT flag or revert these — they are Task-8-owned
  > and reconciled at Task 8's own slice boundary (Task 8 Step 4), not here. The
  > "revert any unlisted file" rule above applies ONLY to the p1/p2/p3 SIBLING
  > slices, never to Task 8's deliverables.

- [ ] **Step 3: Run the affected backend + frontend test surface together.**
  ```
  cd /home/john/elspeth && uv run pytest \
    tests/integration/web/test_tutorial_site_pages.py \
    tests/unit/web/composer/test_tutorial_sample.py \
    tests/unit/web/composer/test_web_scrape_recipe_allowed_hosts.py \
    tests/unit/web/composer/test_web_scrape_recipe_apply.py \
    tests/unit/web/composer/test_web_scrape_recipe_contract.py \
    tests/unit/web/composer/test_web_scrape_recipe_zero_llm.py \
    tests/unit/web/composer/test_recipes.py \
    tests/unit/web/preferences/test_tutorial_cache.py \
    tests/unit/web/composer/guided/ \
    tests/unit/web/composer/test_tutorial_service.py \
    tests/integration/web/test_tutorial_routes.py \
    tests/unit/web/sessions/test_guided_start.py -q
  cd /home/john/elspeth/src/elspeth/web/frontend && npm run test -- src/components/tutorial --run
  ```
  > The four web-scrape recipe tests
  > (`test_web_scrape_recipe_apply.py` / `_contract.py` / `_zero_llm.py`) plus
  > `test_recipes.py` directly cover the `recipes.py` code Task 6 changes — a
  > reconciliation gate that excludes them could pass green on a Task 6 recipe
  > regression. `test_tutorial_routes.py` covers the tutorial route surface Task 8
  > eventually touches.

  Expected: all pass. `test_guided_start.py::test_guided_start_persists_profile_without_materializing_topology`
  must still PASS — it confirms p4 did NOT overload the no-op materializer (the
  invariant this plan deliberately preserved).

- [ ] **Step 4: Surface the operator-owed chores (do NOT perform them).**
  In the slice hand-off to the operator, name:
  1. Clear stale `{data_dir}/tutorial_cache/` files (vestigial after the prompt
     retarget; they miss on the new `model_id`).
  2. Run `tutorial-reliability.staging.spec.ts` against staging to verify the
     live shield-State-C + extraction behavior (operator env).
  3. `npm run build` so the synthetic pages land in `dist/` and serve from the
     SPA mount; restart is the existing deploy step, NOT a composer-skill restart.
  The agent signs nothing and re-signs no gate. p4 touches no plugin file (no
  plugin-hash gate) and no live skill (no skill-hash re-bake, no service restart);
  the Task 6 recipe slot changes the `tutorial_model_id` `recipe_hash`, which only
  invalidates the already-vestigial tutorial cache — surface that as part of chore
  (1), not as a restart.

---

## Task 8: Tutorial passive auto-drive (consumer of p1 + p2 — steps deferred until they land)

This is the headline conceit: the learner specifies NOTHING and watches the LLM
build the pipeline (spec way 1). It is the tutorial-specific CONSUMER of p1's
`/guided/chat` apply contract (contract §1–§2.1) and p2's intent-primary panel.
Its concrete TDD steps **cannot be authored until p1 and p2 land** (writing them
now would be placeholders against non-existent seams, which this methodology
forbids). This task documents the consumer at SIGNATURE level so the contract is
explicit and unowned-no-more; implement it after p1 + p2 merge.

> **Operator decision (surface before executing the series):** confirm p4 is the
> home for the passive auto-drive (the alternative is a dedicated p5). It is
> placed here because p4 is the tutorial slice and already depends on p1+p2+p3.

**Files (when implemented):**
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.tsx`
  — replace the learner-driven embed with an auto-drive that composes the scripted
  intent + resolved URLs, seeds it via p2's intent box (p1's `chatGuided` action,
  `src/elspeth/web/frontend/src/stores/sessionStore.ts:1333-1392`, contract §2.1),
  AND then runs the multi-phase auto-confirm walker (a scripted `respondGuided` /
  POST `/guided/respond` sequence through STEP_1 → STEP_2 → STEP_2.5 → STEP_3 →
  STEP_4) so the wizard advances without learner input under p1's no-auto-advance
  contract. See Produces + Step 3.
- Create (likely): a backend GET surface that returns
  `resolve_tutorial_sample_urls(...)` + `resolve_tutorial_allowed_hosts(...)` for
  the active tutorial session's resolved origin (Task 2 owns the pure resolver;
  this exposes it to the shell since `entry_seed` is server-side-only and the URLs
  are runtime-derived — they cannot ride the frozen constants).
- Modify: the STEP_2.5 recipe-offer-ACCEPT seam in
  `src/elspeth/web/sessions/routes/composer/guided.py` (the accept dispatch that
  reaches `handle_step_2_5_recipe_apply`, `composer/guided/steps.py:223`) — for a
  TUTORIAL session, thread the resolved `allowed_hosts` into the match's
  `allowed_hosts` slot (the Task 6 slot) before the apply builds
  `arguments["slots"] = dict(match.slots)`. `RecipeMatch` is `frozen=True`, so do
  this via `dataclasses.replace(match, slots={**match.slots, "allowed_hosts":
  hosts})` (or an extra-slots arg into `handle_step_2_5_recipe_apply`), NOT an
  in-place mutation. This seam holds
  `request.app.state.settings` and the request origin, so it can call
  `tutorial_sample_base_url(settings=..., request_origin=...)` →
  `resolve_tutorial_allowed_hosts(...)`, mapping the scalar `"public_only"` to an
  empty list (omit). This is p4-OWNED (NOT deflected to p1); the resolver
  (`recipe_match.py`) takes only `(source, sink)` and cannot reach this context,
  so the value is supplied at the accept seam, not in the resolver.
- Create: `src/elspeth/web/frontend/src/components/tutorial/TutorialGuidedShell.autodrive.test.tsx`
  + a backend integration / apply-time test (Step 1) that the auto-drive builds
  the `inline_blob → web_scrape → llm → json` pipeline and that, on a loopback
  base, the committed `web_scrape` node carries
  `allowed_hosts == ["127.0.0.1/32", "::1/128"]` (and on a public base the key is
  omitted → `public_only`).

**Interfaces:**
- Consumes (from sibling plans):
  - p1: `POST /guided/chat` apply contract — a chat submit that produces an
    actionable source config commits via `handle_step_1_source` and re-renders
    in place (contract §1.2/§1.3); the source driver builds the inline URL-row
    `json`/`csv` source from URLs in the message (contract §2.2).
  - p2: the intent box + `chatGuided` store action on the guided panel
    (contract §2.1); the `isTutorial` prop that suppresses freeform (contract §4).
  - p3: the State-C shield result narrated by Task 4's copy (contract §2.3).
- Consumes (from p4):
  - `resolve_tutorial_sample_urls(*, base_url) -> tuple[str, str, str]` (Task 2).
  - `resolve_tutorial_allowed_hosts(*, base_url) -> "public_only" | list[str]` (Task 2).
  - `tutorial_sample_base_url(*, settings, request_origin) -> str` (Task 2).
  - the Task 6 recipe `allowed_hosts` slot.
  - `_TUTORIAL_ENTRY_SEED` (Task 3) as the scripted intent.
- Produces: the passive worked example — a multi-phase AUTO-CONFIRM
  orchestration, NOT a single submit. The auto-drive seeds the intent message
  `f"{_TUTORIAL_ENTRY_SEED}\n" + "\n".join(urls)` via `chatGuided`, then drives
  the EXPLICIT advancing confirm at each phase via `respondGuided` (POST
  `/guided/respond`) through STEP_1 → STEP_2 → STEP_2.5 → STEP_3 → STEP_4 — the
  learner clicks nothing. The STEP_2.5 `respondGuided` carries
  `chosen=["accept"]` and is the seam that injects the host-class `allowed_hosts`
  into the applied recipe (see Files). The auto-confirm walker is a NEW
  product-side orchestration (TutorialGuidedShell + a scripted `respondGuided`
  sequence); the existing phase-walker `driveGuidedWalk` is HARNESS-only and
  cannot be reused at runtime.

> **No-auto-advance contract (acknowledged).** Under p1's DECIDED
> apply-in-place / no-auto-advance model, p1 removes the STEP_1 → STEP_2
> auto-advance currently at `guided.py:~1862` (the `user_advanced`
> `_replace(guided, step=GuidedStep.STEP_2_SINK)` + `emit_step_advanced`). After
> p1 lands, each phase transition is an EXPLICIT confirm on POST
> `/guided/respond` — a passive learner is not present to click it, so the
> tutorial MUST drive those confirms itself. The single-`chatGuided`-submit
> framing earlier in this task is insufficient on its own: it seeds the intent
> but does not advance the wizard. The auto-confirm walker (Step 3) closes that
> gap.

- [ ] **Step 1 (deferred — author after p1 + p2 land): Write the failing
  auto-drive test.** Component test: mount `TutorialGuidedShell` for a tutorial
  session with a stubbed loopback origin; assert it (a) calls the `chatGuided`
  action with a message that contains `_TUTORIAL_ENTRY_SEED` and all 3 synthetic
  URLs, (b) then drives the explicit phase confirms via `respondGuided` (the
  auto-confirm walker) through to the run turn WITHOUT the learner clicking any
  affordance — i.e. assert the scripted `respondGuided` sequence fires (including
  the STEP_2.5 `chosen=["accept"]` confirm), not merely a single `chatGuided`
  submit. This is what makes "passive, no input affordance" actually hold under
  the no-auto-advance contract. Backend integration / apply-time test: drive the
  tutorial-consumer accept on a loopback base; assert the committed pipeline is
  `inline source → web_scrape → llm → json` and the `web_scrape` node carries
  `allowed_hosts == ["127.0.0.1/32","::1/128"]`; on a public base assert
  `allowed_hosts` is omitted (`public_only`). This apply-time assertion is the
  test the mustFix for the loopback SSRF value requires; it lives here (Task 8)
  because Task 8 owns the value injection at the accept seam.

- [ ] **Step 2 (deferred): Expose the resolved URLs + host-class to the shell.**
  Add the GET surface returning the Task 2 resolver output for the active
  session's resolved origin (`tutorial_sample_base_url(settings=..., request_origin=...)`).
  Keep `entry_seed` off the wire (Global Constraint); the URLs are a separate
  runtime-derived payload, not the profile seed.

- [ ] **Step 3 (deferred): Auto-drive the panel as a multi-phase auto-confirm
  walker.**
  In `TutorialGuidedShell`, compose `message = f"{_TUTORIAL_ENTRY_SEED}\n" +
  urls.join("\n")` and seed it via the p2 intent box / `chatGuided` action once
  the guided session is started. Then, because p1 removes the auto-advance
  (no-auto-advance contract, see Produces), drive each phase's EXPLICIT confirm
  via `respondGuided` (POST `/guided/respond`) through STEP_1 → STEP_2 → STEP_2.5
  → STEP_3 → STEP_4, with NO learner input (render passively). This is a NEW
  product-side scripted-confirm sequence owned by `TutorialGuidedShell` (the
  harness's `driveGuidedWalk` is not reusable at runtime). At the STEP_2.5
  `respondGuided` (`chosen=["accept"]`), the backend accept seam injects the
  host-class `allowed_hosts` into the recipe apply (Task 6 slot) per the Files
  list — the resolved `allowed_hosts` is supplied server-side at that seam, not
  passed by the shell, because `allowed_hosts` is an SSRF control set by the seam,
  never carried over the wire by the client.

- [ ] **Step 4 (deferred): Run to pass + wardline + commit.**
  Run the component + backend integration tests to green; `wardline scan .
  --fail-on ERROR`; commit the auto-drive with the standard `SKIP=` gates. The
  live staging watch-through (learner-authors-nothing, lands State C) remains the
  operator-owed verification (P7 known-gap pattern).
