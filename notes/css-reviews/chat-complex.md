# Edit Review: chat.css (largest area, 5-fragment split)

## Summary

- **File**: `src/elspeth/web/frontend/src/components/chat/chat.css` (1407 lines, type: CSS)
- **Source**: `src/elspeth/web/frontend/src/App.css` (7335 lines)
- **Fragments**: 5 (chat-1..chat-5), no moves in/out
- **Verdict**: **Approved**. All five invariants hold; rule bodies byte-identical to source; no orphans or cascade collisions introduced by the split.

## Verification Method

1. Extracted the five declared line ranges from `App.css`:
   - 779–902 → `/tmp/chat-1.css` (124 lines)
   - 2227–2707 → `/tmp/chat-2.css` (481 lines)
   - 4286–4853 → `/tmp/chat-3.css` (568 lines)
   - 4854–4939 → `/tmp/chat-4.css` (86 lines)
   - 5419–5549 → `/tmp/chat-5.css` (131 lines)
   - Concatenated in source order → 1390 lines.
2. Stripped CSS comments and blank lines from both the concatenated source and the new `chat.css`. Both produced **1150 lines, byte-identical** (`diff` returned no output). The 17-line surplus in the new file is fully accounted for by the new header comment block (lines 1–13) plus four section banner comments.

## Intent Fit (5 invariants)

| # | Invariant | Status | Evidence |
|---|---|---|---|
| 1 | Five fragments present in source order | ✓ | chat-1 anchor `.bubble {` at L18; chat-2 `.inline-run-results {` at L143; chat-3 `.message-row {` at L632; chat-4 `.composing-row {` at L1194; chat-5 `.template-cards-container {` at L1281. Monotonic. |
| 2 | Selector text byte-identical | ✓ | Stripped-diff produced 0 lines of output across 1150 normalized lines. |
| 3 | `.chat-panel:has(.inline-run-results) .scroll-to-bottom-btn` precedes `.chat-panel` base | ✓ | `:has` rule at new L114 (originally L877, inside chat-1); `.chat-panel {` base at new L483 (originally L2570, inside chat-2). 114 < 483 ✓. Same-specificity wins for `.chat-panel { ... }` would be later-wins, so the `:has` rule (higher specificity anyway) still wins; ordering is preserved as in source. |
| 4 | Both `(max-width: 760px)` and `(max-width: 520px)` media queries from chat-5 preserved | ✓ | `@media (max-width: 760px)` at L1397 and `@media (max-width: 520px)` at L1403 inside the template-cards block. Also retained: the chat-1 `(hover: none)` at L86, the chat-2 `(max-width: 760px)` at L325, the chat-2 chat-panel `(max-width: 760px)` at L468, and the chat-3 inline-source-fallback-prompt `(max-width: 760px)` at L1144. Five `@media` queries total in the new file, matching the source. |
| 5 | Header comment names tenants | ✓ | Lines 11–13: "Tenants (execution-named but chat-mounted — they render inside the chat panel via inline run results): .inline-run-results*, .run-outputs-panel*, .run-output-artifact*." Owners list (L2–10) covers all chat-owned selectors enumerated in the brief. |

## Scope Discipline

- Declared: no moves in, no moves out, orphaned banner stub at L3592–3595 in App.css deleted.
- Verified: stripped diff between concat(source-ranges) and new file is empty — nothing extra was added beyond the new header/section banners (comment-only, non-functional). No silent inclusion of adjacent source content.

## Structural Integrity

- Brace balance: implicit-pass (CSS comment-strip produced parseable groups; no diff against source).
- Five fragments concatenate cleanly with new section banners as the only insertions.
- Header comment well-formed; closes at L13.

## Cross-Reference / Cascade

- The only cross-fragment ordering invariant called out in the brief — `:has` rule before `.chat-panel` base — is preserved (L114 < L483).
- No selector renames; the new file references the same CSS custom properties as the source rules.

## Duplicate Selectors (pre-existing, not split artifacts)

Five selectors appear in two rule blocks each in the new file. **All five also appear twice in App.css** — pre-existing CSS-grouping patterns, not duplication introduced by the split:

| Selector | New file lines | App.css count |
|---|---|---|
| `.chat-input-cancel-btn:focus-visible` | L434, L438 | 2 |
| `.inline-source-fallback-prompt-detail` | L1117, L1127 | 2 |
| `.runs-pending-proposal` | L1171, L1186 (as group member) | 2 |
| `.tool-call-actions` | L966, L1008 | 2 |
| `.tool-call-stale` | L985, L1017 | 2 |

Inspection of `.runs-pending-proposal` (the only one whose duplication is across two distinct grouped selectors) shows the first rule sets the warning border/padding/background and the second adds `flex-wrap: wrap` to a narrower set — intentional layering. No introduced cascade collision.

## Dead Rules (pre-existing, not split artifacts)

Nine classes are defined in `chat.css` but have **zero references** in any `.tsx`/`.ts` under `src/elspeth/web/frontend/src/`. All nine also have zero TSX references in the original codebase — these are pre-existing dead rules carried forward verbatim:

```
.inline-source-created-turn-title
.message-tools-details
.message-tools-item
.message-tools-pre
.message-tools-summary
.runs-pending-proposal
.spec-pending-proposal
.tool-call-card--committed
.tool-call-card--rejected
```

(Caveat: these grep checks would miss usages constructed via string concatenation, conditional class joins, or external HTML templates — but if those exist, they apply equally to source and new file. The split does not change their dead/live status.)

## Out-of-Scope Observations

- The nine pre-existing dead-rule classes are candidates for a separate cleanup pass; this review does not propose deletions because the brief is fidelity-of-split, not housekeeping.
- The `.message-tools-*` family is structurally consistent (likely a `<details><summary><pre>` pattern from an earlier UI) — if intentionally retired, all five should go together; if pending reintroduction, an entity-binding comment would aid future archaeology.

## Confidence Assessment

**High.** Verified by line-range extraction + stripped-content diff (1150 normalized lines, zero diff), anchor-based source-order check, explicit invariant checks (1–5), and TSX/TS reverse lookup for dead-rule detection. No tooling beyond `python3`/`grep`/`diff` required; results are reproducible.

## Risk Assessment

**Low residual risk.**

- Visual regression risk: minimal. Rule bodies are byte-identical; ordering of the one called-out cross-fragment dependency is preserved; all media queries are retained.
- Cascade risk: the new file is now imported as a separate stylesheet. If the import order in the parent CSS index does **not** preserve the same global cascade position relative to non-chat rules that previously interleaved (none for this fragment set per the brief), some cross-area cascade interactions could shift. This review did not inspect the index/import order — flagged as an information gap.

## Information Gaps

- Did not verify the parent CSS index (`index.css` or equivalent) imports `chat.css` at a cascade position that preserves the original global ordering. Cross-file review territory; flagged for the orchestrator.
- Did not verify runtime appearance in the staging UI — pure source-fidelity review.
- Dead-rule detection used static grep against `.tsx`/`.ts`; classes assembled via string interpolation would be missed.

## Caveats

- This review is fidelity-of-split only. It does not assess whether the original CSS rules are themselves correct, idiomatic, or minimal.
- The duplicate-selector detection treats group members as repeats; manual inspection of each case confirmed none are true semantic duplicates introduced by the split.
