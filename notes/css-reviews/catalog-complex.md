# catalog.css — complex-reviewer fidelity report

**Scope.** Fidelity review of `src/elspeth/web/frontend/src/components/catalog/catalog.css` (543 lines, 70 rule blocks) against the three documented fragments lifted from `src/elspeth/web/frontend/src/App.css`:

- Fragment 1 — `App.css` 3717–3761 → catalog.css 10–43 (shortcuts), EXCLUDING `App.css` 3752–3760 (`.command-palette-footer kbd`, OUT to common.css).
- Fragment 2 — `App.css` 5200–5398 → catalog.css 46–243 (plugin-card / audit-icon).
- Fragment 3 — `App.css` 5642–5940 → catalog.css 246–543 (catalog drawer, search, tabs, filter chips, status messages, inline-chat-source entries).

ADJACENCY MOVES: OUT only — `.command-palette-footer kbd` (App.css 3752–3760) moved to `styles/common.css` (line 327, adjacent to its `.command-palette-footer` ancestor at line 317). No IN.

---

## Intent fit

| Verification | Result | Evidence |
|---|---|---|
| Three fragments present in source order | Pass | catalog.css 13–43 (shortcuts), 49–243 (plugin-card/audit-icon), 249–543 (drawer). Source order preserved; same comment-banner headings. |
| Fragment 1 byte-identical to App.css 3720–3750 | **Pass — byte-identical** | `diff` of App.css 3720–3750 vs catalog.css 13–43 returned no differences. |
| Fragment 2 byte-identical to App.css 5203–5397 | **Pass — byte-identical** | `diff` of App.css 5203–5397 vs catalog.css 49–243 returned no differences. |
| Fragment 3 byte-identical to App.css 5645–5939 | **Pass — byte-identical** | `diff` of App.css 5645–5939 vs catalog.css 249–543 returned no differences. |
| `.command-palette-footer kbd` NOT in catalog.css | Pass | Two grep hits inside the file's leading 8-line comment banner only — both reference the move, no rule body. No `.command-palette-footer kbd {` rule in the file. |
| `.command-palette-footer kbd` IS in common.css | Pass | `styles/common.css:327: .command-palette-footer kbd {`, adjacent to `.command-palette-footer` at line 317. |
| `.plugin-card*` + `.audit-icon*` co-located | Pass | One contiguous block lines 49–243; matches App.css 5200–5398 ordering. Header comment ("PluginCard") covers both — `audit-icon` is the card's visual element, correct per the move plan. |
| `.filter-chip*` and `.catalog-tab*` preserved | Pass | All six `.filter-chip*` selectors at lines 401–478, all five `.catalog-tab*` selectors at lines 363–394 — every selector in App.css 5759–5874 present and identical. |

**Selector count parity.** `App.css` contains 75 instances of catalog-owned top-level selectors (regex `^\.(audit-icon|plugin-card|catalog-(backdrop|drawer|header|close-btn|search|tab|list|status-message)|filter-chip|inline-chat-source-entry|shortcuts-(group|list))`); catalog.css contains the same 75. Zero loss.

**Brace balance.** 70 opening `{` lines, 70 closing `}` lines. Balanced.

---

## Scope discipline

| Declared out-of-scope | Verified preserved |
|---|---|
| `.command-palette-footer kbd` (moved to common.css) | Confirmed absent from catalog.css, confirmed present in common.css line 327. |

| Selector that looks like catalog domain but stays elsewhere | Where it lives | Why correct |
|---|---|---|
| `.catalog-reference-label` (App.css 1649), `.catalog-reference-meta` (App.css 1659) | sidebar.css lines 122 / 132 | Used by `components/sidebar/CatalogButton.tsx`. The "Plugin catalog" reference badge is a sidebar element, not part of the catalog drawer surface. Correct placement. |

No undeclared additions, no opportunistic edits inside catalog.css. The 8-line header banner is new framing but matches the documented owners list exactly.

---

## Structural integrity

| Invariant | Status | Evidence |
|---|---|---|
| Heading hierarchy (section banners) | Pass | Three banners — "Shortcuts Help" (12), "PluginCard" (47), "CatalogDrawer (inline styles migrated)" (247). Same headings as App.css source. |
| Brace balance | Pass | 70/70. |
| Selector preservation | Pass | 75/75 catalog-owned selectors. |
| TSX usage of selectors (dead-rule spot-check) | Pass | Every probed selector has ≥1 TSX consumer in `components/catalog/`, `components/common/ShortcutsHelp.tsx`, or test files. Sampled 13 leaf selectors including `audit-icon-glyph`, `catalog-tab-strip`, `filter-chip-active`, `catalog-status-message--center`, `inline-chat-source-entry-try`. |
| Comment-banner sync | Pass | Owners list in lines 1–8 enumerates exactly the selector families that landed: `.shortcuts-group`, `.shortcuts-list*`, `.plugin-card*`, `.audit-icon*`, `.catalog-backdrop`, `.catalog-drawer`, `.catalog-header*`, `.catalog-close-btn`, `.catalog-search*`, `.catalog-tab*`, `.catalog-list`, `.filter-chip*`, `.catalog-status-message*`, `.inline-chat-source-entry*`. Move-out note for `.command-palette-footer kbd` is explicit. |
| Build / lint | Not run | Pure CSS move; rule bodies are byte-identical to App.css so no behavioural delta is possible from this file's content. |

---

## Cross-reference / call-site integrity

- Selector-rename risk: none — no renames performed, this is a pure relocation.
- `.command-palette-footer kbd` rule body intact at `styles/common.css:327`; no orphan reference in catalog.css (the two hits in lines 6–7 are explanatory comment text, not rule selectors).
- All catalog selectors continue to resolve from their existing TSX class strings (sampled set of 13 confirmed; the comprehensive set is the same 75 names that were live before the split, so no TSX wiring change is required by this move).

---

## Orphan / dead-code / boundary findings

**No dead rules inside the moved fragments.** Spot-check of 13 leaf selectors all returned ≥1 TSX consumer.

**Boundary 1 (top of file).** Lines 1–11 are new file-header comment + first banner. Clean transition into the first rule at line 13. No dangling pronoun, no stale "as described above". The "moved to common.css" note correctly attributes the displaced kbd rule and removes any reader confusion when they hit the missing rule.

**Boundary 2 (Shortcuts → PluginCard, lines 43–49).** Line 43 closes `.shortcuts-list-item dd`; lines 46–48 are the banner; line 49 opens `.plugin-card`. Whitespace: two blank lines between sections (44, 45) — matches App.css convention (3751, blank line then banner at 3762 in the source). Clean.

**Boundary 3 (PluginCard → CatalogDrawer, lines 243–249).** Line 243 closes `.plugin-card-no-fields`; lines 246–248 banner; line 249 opens `.catalog-backdrop`. Two blank lines between. Clean.

**Boundary 4 (end of file, line 543).** Last rule `.inline-chat-source-entry-try` closes at 543; file ends at 543 with no trailing blank line. App.css fragment ends at line 5939 with `}` then a blank 5940 then a new banner at 5941. Trailing newline is fine; no orphan.

---

## Style / behavior continuity

| Edit | Surrounding style | Inserted style | Match? |
|---|---|---|---|
| File header comment (lines 1–8) | Other split files use a short owners-list comment at the top (per task description for animations/base/common etc.). | catalog.css uses the same format: owners list + moved-out note. | Pass |
| Section banners | App.css uses 80-char dashed banners `/* --- … --- */` with the section name on the middle line. | catalog.css uses the same banner format, copied verbatim from App.css 3717–3719, 5200–5202, 5642–5644. | Pass |
| Rule bodies | App.css selector text + declaration block ordering. | All three fragments byte-identical to source. | Pass |

Behavior preservation: this is a relocation refactor. Because each fragment's bytes match App.css and no selector specificity changed, cascade order is the only behavioural variable — and that lives in the entrypoint (`styles/index.css`) where the catalog import order vs the rest is configured. That is out of scope for this file-scoped review; the index-complex reviewer owns it.

---

## Out-of-scope observations (mention only — not for this PR)

These were flagged in the brief; surface here without fixing.

1. **Merge candidate: `.catalog-tab` vs `.filter-chip`.** Both are flex pill-style segmented controls — same `gap`, same `font-size: var(--font-size-xs)` family, same focus-ring treatment (`outline: 2px solid var(--color-focus-ring)` on `.filter-chip:focus-visible`). A shared base class (`.elspeth-segmented-pill` or similar) could DRY this, but the existing implementations differ in `min-height` (28 vs unset), `border` (`.filter-chip` has one, `.catalog-tab` doesn't), and `flex: 1` (catalog-tab only). Worth a follow-up design review; not a fidelity defect.

2. **Broken/missing focus state on `.catalog-tab`.** `.catalog-tab` has no `:focus-visible` rule of its own (lines 369–377). It relies on whatever the underlying `<button>` base styles provide. By contrast `.filter-chip:focus-visible` (line 456) and `.catalog-search-clear:focus-visible` (line 358) both define explicit focus rings. If the catalog tab strip uses native button elements styled away from the default focus ring elsewhere in the cascade, tabs may be unreachable via keyboard navigation. Pre-existing in App.css (lines 5765–5773); not introduced by the split, but worth surfacing.

3. **`.catalog-close-btn` focus state.** Same pattern — no `:focus-visible` declaration (lines 306–312). Pre-existing; same caveat.

4. **Missing forced-colors / high-contrast fallback for `.catalog-backdrop`.** The semi-transparent black `rgba(0, 0, 0, 0.3)` overlay (line 252) is invisible in Windows High Contrast Mode (forced-colors: active). The drawer will float without a visible scrim. A `@media (forced-colors: active) { .catalog-backdrop { background-color: Canvas; opacity: 0.5; } }` block (or use of `Canvas` / `CanvasText` system colors) would address this. Pre-existing in App.css 5645–5650; not introduced by the split.

5. **`.audit-icon-glyph { display: none; }`** (line 110). Looks like dead presentation — but it's actively used by `components/catalog/AuditCharacteristicIcon.tsx` (confirmed via grep) which renders a visually-hidden glyph for screen-reader text. Not dead.

These five items are pre-existing in the App.css source and are preserved unchanged by the move — they are NOT regressions introduced by this PR. File-scope reviewer surfaces them per the brief; remediation belongs to a separate accessibility/design pass and should not block this CSS-split landing.

---

## Confidence Assessment

**Confidence: High.**

Basis: three independent `diff` invocations returned identical output for all three fragments; selector-count parity verified by regex (75 = 75); `command-palette-footer kbd` rule located in common.css at the expected adjacent-to-ancestor position; spot-check of 13 leaf selectors confirmed live TSX consumers; brace balance verified (70/70). The remaining un-probed selectors (62 of 75) were not individually grepped for TSX consumers — but the byte-identical-fragment evidence makes per-selector dead-rule audit redundant: any dead rules here were already dead in App.css and that is not a defect introduced by this PR.

## Risk Assessment

**Residual risk: Low.**

The only risks that could survive this review:

- **Cascade-order regression at the entrypoint.** Out of scope; owned by index-complex review.
- **Editor injected non-UTF8 / BOM / CRLF.** Not checked explicitly, but `diff` over byte streams would have caught it; clean.
- **A selector somewhere in App.css that should also have moved but stayed behind.** App.css still contains `.catalog-reference-label` / `.catalog-reference-meta`; verified those belong to sidebar.css's domain (CatalogButton TSX is in `components/sidebar/`). No other catalog-domain stragglers spotted by the namespace grep.

## Information Gaps

- No build / TSX render verification run — fidelity-only review.
- Did not exhaustively grep all 75 selectors for TSX consumers; sampled 13.
- Did not verify forced-colors behaviour by running the app — observation 4 is based on the CSS source alone.
- Did not verify cascade order at `styles/index.css` import site (out of scope for file-scoped review).

## Caveats

This review covers only `catalog.css` and the OUT-move target (`common.css` line 327). It does NOT validate:
- Cascade ordering at `styles/index.css`.
- That common.css' kbd rule still sits in the correct order relative to other common.css rules (common-complex reviewer's domain).
- TSX-side className correctness.
- Build/test passage.

---

**Verdict: APPROVED.** All three fragments byte-identical to App.css source. The single OUT move is correctly executed with the rule landed at common.css line 327, adjacent to its `.command-palette-footer` ancestor. The five pre-existing accessibility / merge-opportunity observations are surfaced for future work but are not regressions from this split.
