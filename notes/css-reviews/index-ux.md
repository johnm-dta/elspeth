# UX-Engineering Review: `styles/index.css` (barrel file)

**File:** `src/elspeth/web/frontend/src/styles/index.css`
**Reviewer:** UX Critic Agent
**Date:** 2026-05-23
**Scope:** Maintainability UX (developer experience) — not visual design.

---

## Confidence Assessment

- **High** on discoverability, naming consistency, and layered-architecture clarity — the file is short and fully readable.
- **Medium** on performance findings — line counts for referenced files not measured in this pass; flagged as "verify" items.
- **Low** risk of mis-read: cascade is explicit and matches the comment.

## Risk Assessment

- **Low** overall. The file is structurally sound. Gaps are documentation precision issues, not ordering bugs.
- The one naming exception (`common.css` and `shared.css` at the flat level alongside `tokens.css`, `base.css`, `themes.css`) is a latent confusion point for contributors.

---

## Summary

**Overall: Acceptable — one Minor gap, one Improvement.**

The barrel is compact and the header comment documents the two critical ordering constraints (common-before-catalog, themes-last). Three findings below, none blocking.

---

## Findings

### Discoverability

**Strengths:**
- Header comment is present, well-formatted, and explicitly names both load-bearing constraints with a brief rationale for each. A new contributor reading top-to-bottom will understand *why* before they see *what*.
- The two highest-risk ordering rules (common → catalog; themes last) are stated in prose, not just implied by position.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Header lists rules but gives no "where to insert" guidance | Minor | Lines 1–13 (header) | Add a sentence: "New component files belong in the per-area block (after `shared.css`, before `common.css`). New global utilities belong after `shared.css` and before any component that uses them." |
| `themes.css` line has no inline constraint marker | Minor | Line 36 | Append a trailing comment: `/* MUST remain last — see cascade rules above */` This makes the constraint visible even when diffing or viewing without the file header. Same treatment warranted for `common.css` re: catalog ordering (line 26). |

---

### Naming Consistency

**Pattern in use:** `<area>/<area>.css` for component directories (`components/catalog/catalog.css`, etc.).

**Flat-level files:** `tokens.css`, `base.css`, `animations.css`, `shared.css`, `common.css`, `themes.css` — all live in `./styles/` without a subdirectory and have no common prefix schema.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| `shared.css` vs `common.css` — two flat-level files with semantically overlapping names, no comment distinguishing their roles | Minor | Lines 18, 26 | Add a brief parenthetical after each import: `/* shared.css — cross-component primitives (mixins, utility classes) */` and `/* common.css — layout scaffolding shared by catalog and downstream areas */`. Without this, contributors guess which file to extend for a new cross-cutting style. |

No naming inconsistency with the component convention itself — the `<area>/<area>.css` pattern is followed by all seven component entries.

---

### Layered Architecture Clarity

**Strengths:**
- The file is visually divided into three blocks separated by blank lines: foundation layer (lines 15–18), component layer (lines 20–35), override layer (line 36). The structure is immediately parseable.

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| Block boundaries have no section labels | Minor | Between line 18/20 and line 35/36 | Add a one-line comment above each block: `/* --- Foundation --- */`, `/* --- Components --- */`, `/* --- Overrides --- */`. The whitespace alone is easy to miss in a diff, and the labels make the three-layer contract visible at a glance. |
| `common.css` appears mid-component-block (line 26) not at the end of the foundation block | Minor | Line 26 | Position is correct (it must precede catalog), but its placement inside the component block without a comment makes it look like a component file when it is a cross-cutting layout utility. The parenthetical suggested under Naming would resolve this. |

---

### Self-Documenting Constraints

**Issues:**

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| "themes LAST" constraint documented in header but not at the import site | Minor | Line 36 | `@import "./themes.css"; /* MUST remain last — overrides all component selectors via @media (prefers-contrast), (forced-colors) */` |
| "common BEFORE catalog" constraint documented in header but not at either import site | Minor | Lines 26–28 | Trailing comment on line 26: `/* MUST precede catalog.css — provides .command-palette-footer selectors */` |

Both are already documented in the header. The issue is that the header is easy to skip when editing a file you already "know." Inline markers survive context collapse and diff review.

---

### Performance

**No issues found, with one verify item.**

- All imports are per-area structural files; there is no sign of a file small enough (under ~30 lines) that inlining would be meaningfully faster than a separate parse.
- Code-splitting per route: not applicable. All areas are used within a single SPA shell; route-level CSS splitting would require a bundler config change upstream of this barrel, not changes to the barrel itself.
- **Verify:** `shared.css` and `common.css` — if either is under 30 lines, consider whether it warrants its own file or should be merged into `base.css`. This could not be confirmed without reading those files.

---

## Priority Recommendations

**Minor (fix before next contributor onboarding):**

1. Add inline trailing comments at `common.css` (line 26) and `themes.css` (line 36) marking the ordering constraints that cannot be violated. These are the two highest-risk lines; make the danger visible at the site.
2. Add a parenthetical distinguishing `shared.css` from `common.css` — the naming overlap is a silent trap for contributors deciding which file to extend.
3. Add section-label comments above each of the three blocks (foundation / components / overrides).
4. Extend the header with a one-sentence insertion guide ("new component files go in the component block, before `common.css`").

**Verify (no change required unless confirmed):**

- Check line counts of `shared.css` and `common.css`; if either is trivially small, evaluate merging into `base.css`.

---

## Information Gaps

- Did not read the individual imported files. Naming-role distinctions for `shared.css` vs `common.css` are inferred from position; the actual content may already make the distinction obvious (or may confirm the confusion).
- No bundler/Vite config reviewed; route-level splitting assessment is inference from SPA structure mentioned in CLAUDE.md.

## Caveats

- All findings are documentation/discoverability issues. None represent an ordering bug or a broken cascade. The file is structurally correct as written.
- Section-label comments are a matter of convention; if the team already has a stated preference against them (not found in the files read), the recommendation should be dismissed.
