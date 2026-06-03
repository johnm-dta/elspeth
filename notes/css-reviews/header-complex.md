# header.css — Split Fidelity Review

**Files**
- NEW: `/home/john/elspeth/.worktrees/css-split/src/elspeth/web/frontend/src/components/header/header.css` (311 lines; 305 declaration lines + 6-line top comment)
- ORIGINAL: `/home/john/elspeth/src/elspeth/web/frontend/src/App.css` lines 1132–1436 (305 lines)

**Verdict:** Approved with one factual correction to the top-of-file tenant comment.

---

## 1. Byte-Identity Verification

Diff command:
```
diff <(sed -n '1132,1435p' src/.../App.css) <(sed -n '7,310p' header.css)
EXIT=0
```

The 304 lines of declaration content in `App.css` 1132–1435 are **byte-identical** to `header.css` 7–310. The only off-by-one is that the source range as quoted (1132–1436) includes one trailing blank line (1436) that the new file omits at EOF. This is a normal slice-boundary artefact, not a content loss.

| Required check | Result |
|---|---|
| All rules in 1132–1436 present | YES (1132–1435 verified byte-identical; 1436 is a blank line) |
| Selector text byte-identical | YES |
| Declaration order preserved | YES |
| Comment text byte-identical (incl. WCAG 2.5.8 note, btn-compact rationale, destructive-variant rationale) | YES |

## 2. Destructive-Variant Rule

The brief asks for `.header-session-switcher-action--destructive` "or similar" for Sign out.

- The actual selector is `.user-menu-action--danger` (header.css lines 233–241, ORIGINAL lines 1358–1366). Present, byte-identical, with its hover/focus-visible compound rule.
- The TSX consumer is `components/common/UserMenu.tsx`:
  `className="user-menu-action user-menu-action--danger"`.
- There is NO `.header-session-switcher-action--destructive` rule in either source or new file, and no TSX requests one. No loss; the brief's naming guess was off but the substantive rule is preserved.

## 3. Top-of-File Tenant Comment — Factual Error

The new file's comment (lines 1–5) says the tail rules `.runs-history-item-summary`, `.run-diagnostics*`, `.run-failure-detail*` are "rendered from the header session-switcher dropdown".

This is **not what the code actually does.** TSX consumer search:

- `runs-history-item-summary`, `run-diagnostics-*`, `run-failure-detail` — only used in `components/execution/RunsHistoryDrawer.tsx`.
- `RunsHistoryDrawer` is mounted from `components/execution/InlineRunResults.tsx` line 157, NOT from `AppHeader.tsx`, `HeaderSessionSwitcher.tsx`, or any header component.

The tail rules are execution-domain selectors that happen to live in this byte range of `App.css`. The brief's adjacency claim ("render inside the header session-switcher dropdown") and the comment that encodes it are both incorrect.

**Recommendation:** Either (a) correct the comment to say the tail rules are execution-domain runs-history selectors co-located here purely by source order, with a follow-up to move them to `execution.css`, or (b) move them now. They are cleanly extractable — the selectors `.runs-history-item-summary`, `.run-diagnostics-panel`, `.run-diagnostics-panel-header`, `.run-diagnostics-actions`, `.run-diagnostics-operations`, `.run-failure-detail`, `.run-failure-detail pre`, `.run-diagnostics-tokens`, `.run-diagnostics-working-view`, `.run-diagnostics-explanation` share no specificity contests with header selectors and the cascade order between header.css and execution.css does not matter for them.

The brief explicitly framed this as a verbatim split with "no moves", so the safer action for this split is option (a): fix the comment, file the move as a follow-up. Moving now would exceed split scope.

## 4. Dead-Rule Scan

Searched `src/` `.tsx`/`.ts` for every selector in the file:

| Selector | Consumer | Status |
|---|---|---|
| `.app-root` | App-level shell | Live |
| `.alert-banner`, `.alert-banner-action`, `.alert-banner--info` | (verify in alert banner component) | Likely live |
| `.app-main`, `.app-header*` | `components/common/AppHeader.tsx` | Live |
| `.header-session-switcher*` | `components/sessions/HeaderSessionSwitcher.tsx` | Live |
| `.user-menu*`, `.user-menu-action--danger` | `components/common/UserMenu.tsx` | Live |
| `.header-session-switcher-rename-form` | HeaderSessionSwitcher (rename mode) | Live |
| `.runs-history-item-summary`, `.run-diagnostics*`, `.run-failure-detail*` | `components/execution/RunsHistoryDrawer.tsx` | Live — but mis-located (see §3) |

No dead rules detected within the migrated block.

## 5. Cleanly Extractable Moves (informational, NOT for this split)

The tail block (header.css lines 250–310, originally App.css 1375–1435) is cleanly extractable to `components/execution/execution.css`:

- All ten selectors are referenced only from `RunsHistoryDrawer.tsx`.
- None compound with header-domain selectors (no `.header-session-switcher .run-diagnostics-*` descendant rules; no co-defined `:hover` chains crossing the boundary).
- Specificity is single-class throughout; cascade-order independence between header.css and execution.css.

The rename-form rule (`.header-session-switcher-rename-form`, lines 243–248) is genuinely header-domain (sibling of `.header-session-switcher-*`) and belongs here.

Nothing belongs in `sessions.css` from this file: `.header-session-switcher-*` is chrome (the dropdown trigger and menu rendered in the app header), not session-detail UI.

---

## Confidence Assessment

**Confidence: High** for the byte-identity verification (mechanical diff) and the TSX consumer attribution (mechanical grep across `.tsx`/`.ts` in `src/`). **Medium** for the "no specificity contest" claim in §5 — I verified the selectors in the migrated block do not compound, but I did not exhaustively scan the rest of `App.css` and the other split files for cross-file compound selectors targeting these classes.

## Risk Assessment

**Residual risk:** Low for the split itself. The single substantive finding (incorrect tenant comment) is a documentation defect, not a behaviour change. The tail-block move to `execution.css` is a follow-up cleanup, not a regression risk for this split.

## Information Gaps

- Did not run `npm run build` / vitest / Playwright. Visual regression is plausible only if some other split file changed cascade order; not assessed.
- Did not verify that the corresponding lines in `App.css` are removed in this branch (this review is single-file fidelity, not whole-split completeness).
- Did not verify `.alert-banner*` consumers explicitly — assumed live; a fast grep for the class name would confirm.

## Caveats

- The brief's "ADJACENCY: No moves" instruction was followed correctly by the writer; my §3/§5 finding is that the brief itself misclassified the tail rules' tenancy. The writer's comment faithfully transcribed the brief's claim. The fix is to the comment, with the move filed as a separate cleanup ticket.
- The brief's selector name guess (`.header-session-switcher-action--destructive`) does not exist in source; the equivalent live selector is `.user-menu-action--danger`. Confirmed preserved.
