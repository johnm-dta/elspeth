# ELSPETH Design System — Implementation Design

- **Date:** 2026-06-27
- **Status:** Approved (scope + open choices confirmed by operator)
- **Branch:** work isolated in a worktree off `release/0.7.0`
- **Source of truth for the design:** claude.ai/design project `85edbbda-2f1b-4330-9c7d-e76891cf93d1` ("ELSPETH Design System"), mirrored read-only to `…/scratchpad/elspeth-design-system-latest/` (76 files). Reconciliation analysis: `…/scratchpad/elspeth-design-system/ANALYSIS.md`.

## 1. Context

The operator imported the "ELSPETH Design System" from claude.ai/design and asked to "implement the designs in this project."

A value-by-value diff against the live frontend (`src/elspeth/web/frontend`) established that **the design system is a faithful reverse-engineered snapshot of the current frontend** — its own readme says so, and the repo remains the source of truth. Specifically:

- **Colour tokens, spacing, radius, sizing, transitions, z-index, type sizes/line-heights: byte-identical** to `src/styles/tokens.css` (including the latest refinements — coalesce `#18c2c0`, status-empty `#888888`, light-mode info `#176d8a`).
- **Primitive CSS** (`tokens/primitives.css`) is a copy of the live `src/styles/shared.css` classes.
- **One real additive token gap:** the DS defines `--font-weight-{regular,medium,semibold,bold}` + `--tracking-wordmark`; the frontend has none of these five and hardcodes the values.
- **No reusable React primitive components** exist in the frontend (no `Button.tsx`, `TypeBadge.tsx`, …; only `catalog/PluginCard.tsx`). The `.btn` class is applied inline across ~49 `.tsx` files.
- A 2026-06-27 re-sync found exactly one **forward-design** addition: a net-new **marketing website** (`ui_kits/website/`, 7 static files) — a surface the product does not have today.

So "implement the designs" is a near-no-op at the token/CSS level; the real, sanctioned work is the four items below.

## 2. Scope (confirmed)

**A. Install the design system as a repo Skill.** Copy the latest DS tree into `.claude/skills/elspeth-design/` so `SKILL.md` (`user-invocable: true`) makes it an on-brand reference for future UI work.

**B. Build a typed React primitive library + prove it with one exemplar migration.**

**C. Tokenize font weights + wordmark tracking** in the live token layer (no visual change).

**D. Stand up the marketing website** as a servable static site at repo-root `website/`.

### Confirmed decisions
| Decision | Choice |
|---|---|
| Primitive library location | `src/elspeth/web/frontend/src/components/ui/` (new), barrel `index.ts` |
| Website location | `website/` at repo root (standalone static) |
| Exemplar migration scope | `LoginPage.tsx` only |
| Website icons | Lucide via CDN, as authored |
| Isolation | one worktree off `release/0.7.0`; merge `--no-ff` when complete |

### Out of scope
- Migrating the other ~48 inline-`.btn` call sites (incremental follow-up; the library is additive and they keep working unchanged).
- Any change to token *values* (the frontend already matches the design).
- Integrating the marketing website into the Vite app build or deployment pipeline (it stays decoupled static HTML).
- Vendoring Lucide locally (deferred; CDN as authored).

## 3. Design

### A. Skill install
- Copy the `-latest` mirror tree verbatim into `.claude/skills/elspeth-design/` (tokens, components, guidelines, `ui_kits/`, `assets/`, `styles.css`, `_ds_bundle.js`, `_ds_manifest.json`, `_adherence.oxlintrc.json`, `readme.md`, `SKILL.md`).
- No tests (skill content is an LLM prompt, not code).
- **Trade-off accepted:** landing in the worktree means the skill is not active in the current Claude Code session until the branch is merged. This is the cost of the requested isolation.

### B. React primitive library (`components/ui/`)
Build **9 primitives** as thin, typed wrappers over the existing CSS classes already in `shared.css` (so there is zero visual change and nothing else needs to move):

`Button`, `TypeBadge`, `StatusBadge`, `Card`, `Tabs`, `AlertBanner`, `Input`, `Textarea`, `WordMark`.

- `ChatBubble` and `PluginCard` are composer-level composites (and `catalog/PluginCard.tsx` already exists) — **excluded** from the primitive set.
- **API contract:** follow the DS `.d.ts` shapes (e.g. `Button` variant: `primary | danger | ghost | default`, size: `default | compact | small`; `TypeBadge` type: the six component types; `StatusBadge` status: the six run statuses). Each component renders the corresponding `.btn`/`.type-badge-*`/`.status-badge-*`/etc. classes and forwards standard DOM props + `className`.
- **Barrel:** `components/ui/index.ts` re-exports all 9; consumers import from `components/ui`, never internals (matches the DS adherence rule).
- **Tests (TDD, Vitest + Testing Library):** per component — renders, applies the right class for each variant/prop, forwards `className`/DOM props, respects disabled/aria where relevant.

**Exemplar migration — `LoginPage.tsx`:** replace its inline `.btn`/input/wordmark/error markup with `Button`, `Input`, `WordMark`, `AlertBanner`. Chosen because it is self-contained, exercises four primitives, and sits away from the in-flight front-page/composer chrome (clean landing). Existing LoginPage tests must continue to pass; update them only where the locked-in expectation was the old inline markup (a structural-fix test update, not a behavioural change).

### C. Tokenize font weights
- Add to `src/styles/tokens.css` `:root`: `--font-weight-regular: 400; --font-weight-medium: 500; --font-weight-semibold: 600; --font-weight-bold: 700; --tracking-wordmark: 0.18em;`.
- Replace the ~14 hardcoded `font-weight:` literals in `common.css`, `shared.css`, `base.css` with the tokens, and the hardcoded `letter-spacing: 0.18em` in `components/header/header.css` with `var(--tracking-wordmark)`.
- The new primitives consume these tokens too.
- **Invariant:** `:root` and `[data-theme="light"]` must remain top-level selectors (parsed by `colorContrast.test.ts` / `statusBadgeAccessibility.test.ts`). Weights are colour-independent, so those tests are unaffected; run the full `src/styles` test set to confirm green.

### D. Marketing website (`website/`)
- Copy the 7 `ui_kits/website/` files into `website/`.
- Make it self-contained: copy the DS `styles.css` + `tokens/` (the token entry the pages depend on) into `website/styles.css` + `website/tokens/`, and change each page's `<link rel="stylesheet" href="../../styles.css">` to `href="styles.css"`. `site.css` + `site.js` links stay relative (same dir).
- Lucide stays loaded from the unpkg CDN as authored. Fonts load from Google Fonts via `tokens/fonts.css` (as the product does).
- **Known duplication:** `website/tokens/` is a snapshot of the frontend token values. They are identical today; a `README` note in `website/` records that the frontend `src/styles/tokens.css` is canonical and the website copy is a static mirror (future enhancement: a copy step). Acceptable for a decoupled, build-free static site.
- **Smoke test:** a lightweight check that every page's internal nav links resolve to existing files and the referenced `styles.css`/`site.css`/`site.js` exist (a small node script or a Playwright page-load assertion). No heavy framework.

## 4. Testing strategy
- **B:** Vitest unit tests for all 9 primitives (TDD — tests first). Existing `LoginPage` tests kept green / minimally updated for the new component markup.
- **C:** existing `src/styles/*.test.ts` (colorContrast, statusBadgeAccessibility, themeInit) must pass unchanged; `npm run lint:css` clean.
- **D:** static link/asset smoke check.
- **Whole frontend:** `npm run typecheck`, `npm run lint`, `npm run test` green before merge.
- **A:** none (prompt content).

## 5. Worktree & landing
- Create `/home/john/elspeth/.worktrees/design-system-impl` off `release/0.7.0`. This spec and the implementation plan live in the worktree too (full isolation per operator instruction).
- **Python hooks:** symlink the worktree `.venv` to main's so pre-commit hooks run.
- **Frontend deps (required before any B/C/D test/typecheck/lint):** `src/elspeth/web/frontend/node_modules` is gitignored and therefore absent in a fresh worktree. Run `npm ci` in the worktree's frontend dir (or symlink `node_modules` from main — same `release/0.7.0` `package-lock`, so safe and instant) before the TDD loop. Without this, `npm run test/typecheck/lint` cannot run.
- Do all four items there. The exemplar (`LoginPage`) and the additive token change do not touch the front-page/composer files under active edit, so the merge should be conflict-free.
- Merge `--no-ff` back to `release/0.7.0` when tests are green.

## 5a. Build-time reality checks (verify before building, not assume)
1. **How are component-type / status badges actually rendered?** `.type-badge-*`/`.status-badge-*` classes exist in CSS but a grep found **0** `.tsx` files referencing `type-badge` — so badges are either built via a dynamic class-name helper the grep missed, or are not yet surfaced in the live UI. Confirm the real render path before fixing `TypeBadge`/`StatusBadge` APIs so they match actual usage. (The `LoginPage` exemplar exercises only `Button`/`Input`/`WordMark`/`AlertBanner`, so these two are validated by unit tests, not by an in-app call site — acceptable, but know it.)
2. **`LoginPage` test coupling.** Before migrating, check whether its tests query by role/text (survive the wrapper swap untouched, since classes are preserved) or by DOM structure/class (these are the "locked-in expectations" that need updating). This sizes the test work.
3. **Website link uniformity.** `grep` all five `website/*.html` pages to confirm they all use the `../../styles.css` link before the path-rewrite step — only `index.html` was read directly.

## 6. Risks & mitigations
- **Token test brittleness (C):** mitigated — weights are not colour tokens; run the style test set.
- **LoginPage test lock-in (B):** a wave of test edits after the structural swap is the change landing visibly, not a regression — update expectations, do not revert.
- **Website token drift (D):** documented duplication with a canonical-source note.
- **Skill not live this session (A):** accepted consequence of worktree isolation.
- **CI gates (tier-model/plugin-hash):** this work is frontend + skill + docs; it should not trip the Python tier-model or plugin-hash gates, but the full lint surface is run locally before merge.
- **Skill-dir lint discovery (A):** `_adherence.oxlintrc.json` lands inside `.claude/skills/elspeth-design/`. Confirm no repo lint run auto-discovers config by tree-walk into `.claude/` (almost certainly fine — verify, don't assume).

## 7. Next step (planning)
The implementation plan (via the writing-plans skill) should be a **light task breakdown over this spec** — not a second design document. The design is settled here; the plan sequences the build (worktree setup → 5a checks → C tokens → B primitives TDD → B LoginPage exemplar → D website → A skill copy → full-suite verify → merge).
