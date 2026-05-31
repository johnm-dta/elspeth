# App.css split plan

## Operator decision required (read before approving)

The task brief contains two constraints that conflict against this specific file:

1. **Strict byte-identity** — `cat <files in barrel order> | diff - <original App.css>` must be empty.
2. **Logical grouping** — "one CSS file per `components/<area>/`", "do not over-split (~18–20 files total)", and "cross-cutting rules → `styles/shared.css`".

**These constraints cannot both be satisfied against App.css's current ordering.** Constraint 1 forces 35 physical files (each owning one contiguous source range); constraint 2 wants ~20. The gap exists because App.css is not factored by area today — `.btn` lives at line 629 and `.validation-banner-component-btn--error` lives at line 5639, with 5,000 lines of other components in between. To put both in a single `shared.css` would change cascade.

**Two paths, please pick:**

- **Path A — strict byte-identity (this plan delivers):** 35 physical files, multi-fragment areas (e.g. `chat-1-bubble.css`, `chat-2-input-and-panel.css`, …`chat-5-template-cards.css`). The `cat | diff` test is empty. Cascade is provably preserved by source-order construction. Trade-off: discoverability suffers (e.g. `.validation-banner` rules live in two physical `shared/` files).
- **Path B — logical grouping with rendering equivalence:** ~20 physical files, one per area + globals. Verification is no longer `cat | diff`; it becomes "the bundled CSS produces the same computed styles for every DOM node in a snapshot of the running app." This requires a visual-regression suite or a comprehensive computed-style assertion test that does not exist today. The selector-overlap analysis below shows this is feasible — only 19 class names cross physical-file boundaries, and all 19 cross-file dependencies are cascade-preserved as long as the barrel imports in the order specified — but it relies on careful selector-overlap auditing to guarantee, not on byte-identity.

The rest of this plan **assumes Path A** and is complete and executable under that interpretation. If you pick Path B, the file count drops to ~22 unique logical files (one physical per logical) and the verification command changes; the section map below still works because each per-file range is a superset of the rules that file owns under Path B.

---

## Summary

`src/App.css` is **7,335 lines** ordered as a sequence of banner-delimited blocks
that interleave several component areas. The file is not cleanly factored by
component; many blocks are sandwiches (e.g. the "Audit Readiness" banner
contains the `.graph-modal-*` and `.yaml-modal-*` rules in the middle of it,
and the "App Root & Alert Banners" block carries `.app-header`,
`.user-menu`, `.runs-history`, `.run-diagnostics`, and `.run-failure` rules
together with the cross-cutting `.alert-banner`).

**Cascade-preservation strategy.** The split produces N physical files imported
through a single barrel (`styles/index.css`) in the **same source order as
App.css**. Concatenating those files in barrel order yields a byte-identical
copy of the original. To guarantee that, each physical file owns **one
contiguous source range**. This means a logical area (e.g. "chat") that the
operator would naturally want as one file actually appears in **multiple
non-adjacent source ranges**, and must therefore be emitted as multiple
physical files (`chat-1.css`, `chat-2.css`, …) each carrying one range.

**Unique logical target files: 22.** **Physical files in the barrel: 35**
(one per contiguous source range). The arithmetic is forced by the existing
ordering of App.css — the only way to reduce the physical count without
risking cascade changes is to land a prior PR that reorders App.css so each
area's rules become contiguous. That prior PR is **out of scope** for this
behaviour-preserving split. The operator should be aware that "one file per
area" is structurally impossible against this file without prior reordering.

App.tsx will swap its `import "./App.css"` for `import "./styles/index.css"`.

## Logical target files (22)

Globals (under `styles/`):

- `styles/tokens.css` — `:root` dark tokens (lines 8–209) + `[data-theme="light"]` tokens (lines 3183–3307). **Both blocks must be top-level `^:root` / `^[data-theme="light"]` selectors** because `styles/colorContrast.test.ts` parses them with anchored regexes.
- `styles/base.css` — reset, body, #root, typography helpers, scrollbar, focus-visible, skip-to-content, sr-only.
- `styles/animations.css` — keyframes + the `composing-dot` / `progress-bar*` / `spinner` rules that drive them, plus the global `@media (prefers-reduced-motion: reduce)` block that silences them.
- `styles/themes.css` — `@media (prefers-contrast: more)` and `@media (forced-colors: active)` blocks that target component rules (these contain a nested `:root { … }` and `[data-theme="light"] { … }` block, but those are nested under `@media`, so the anchored `^:root` regex in the tests will NOT match them — safe to land here).
- `styles/shared.css` — `.btn`, `.btn-compact`, `.btn-primary`, `.btn-danger`, `.btn-small`, `.side-rail-slot-fill`, `.tab-strip`, `.type-badge*`, `.status-badge*`, `.validation-banner` base + detailed variants, `.empty-state`, `.confirm-dialog`, `.app-layout`, `.layout-chat`, `.layout-siderail`, `.banner`, `.side-rail` base.
- `styles/common-yaml-view.css` — `.yaml-view*` (a YAML viewer used in several panels; verified `yaml-view` greps under `components/common/`).
- `styles/common-command-palette.css` — `.command-palette*` (Ctrl-K palette, lives in `components/common/`).
- `styles/common-error.css` — `.error-boundary*`.
- `styles/chat-markdown.css` — `.markdown-body*` (used by MarkdownRenderer in `components/chat/` but referenced from many bubble surfaces).

Per-area files (under `components/<area>/`):

- `components/chat/chat.css` — bubble base, scroll-to-bottom, blob-action-btn, chat-input, chat-panel, message-bubble, tool-call, pending-proposals, inline-source-*, pending-overlay, spec/yaml/runs-pending banners, composing-indicator, template-cards. **Owns 6 non-contiguous source ranges** (see Section map).
- `components/chat/guided/guided.css` — `guided-*` widget family (5941–7335).
- `components/recovery/recovery.css` — `recovery-*`.
- `components/header/header.css` — App shell + header (app-root, app-header, header-session-switcher, user-menu), plus the **sandwiched** sessions/execution rules at lines 1374–1436 (`runs-history`, `run-diagnostics`, `run-failure`). See cascade-collision callouts below.
- `components/sidebar/sidebar.css` — side-rail-suggestion-*, side-rail-execute/export/catalog-btn, catalog-reference-* (yes — they live in the side-rail block), completion-bar, graph-mini.
- `components/audit/audit.css` — audit-readiness-*, explain-dialog, readiness-row-detail, plus the **sandwiched** `.graph-modal-*` and `.yaml-modal-*` rules at lines 1933–2025 and the trailing `.yaml-modal-body` at 2221–2225. See cascade-collision callouts.
- `components/tutorial/tutorial.css` — `tutorial-*`.
- `components/inspector/inspector-version-selector.css` — version-selector + trailing `.side-rail-error-banner` at 3860–3863.
- `components/inspector/inspector-graph.css` — graph-view, graph-node, graph-config, react-flow theming (the `:root .react-flow.react-flow` and `[data-theme="light"] .react-flow.react-flow` blocks). **The tests `GraphView.test.tsx` parse the second selector via regex against `src/App.css`**, so this file's path is the new test target.
- `components/settings/settings.css` — secrets-*.
- `components/execution/execution.css` — progress-*.
- `components/blobs/blobs.css` — blob-row, blob-manager.
- `components/catalog/catalog.css` — shortcuts-*, plugin-card, audit-icon, catalog-drawer, catalog-tab-*, filter-chip, inline-chat-source-entry. Owns 3 non-contiguous ranges.

Header comment lines 1–7 are emitted at the very top of the barrel as
`/* ELSPETH Web UX … */` so the audit-tooling banner survives.

## Section map (ordered by source line)

| start | end | physical file | rationale |
|------:|----:|---------------|-----------|
| 1     | 7   | (barrel header comment) | top-of-barrel literal |
| 8     | 209 | `styles/tokens.css` | dark `:root` token block — TOP-LEVEL `^:root` |
| 210   | 336 | `styles/base.css` | reset, html/body, #root, typography helpers, scrollbar, focus-visible, skip-to-content, sr-only |
| 337   | 479 | `styles/animations.css` | `@keyframes` (composing-bounce, progress-stripe, pulse-dot, spin) + `.composing-dot`, `.progress-bar*`, `.spinner` + the global `@media (prefers-reduced-motion: reduce)` that silences them |
| 480   | 540 | `styles/shared.css#1` | `.type-badge*` utilities |
| 541   | 625 | `styles/shared.css#2` | `.status-badge*` utilities + inline `@media (prefers-reduced-motion)` for `.status-badge-icon--cancelling` |
| 626   | 746 | `styles/shared.css#3` | `.btn`, `.btn-compact`, `.btn-primary`, `.btn-danger`, `.btn-small`, `.side-rail-slot-fill` |
| 747   | 778 | `styles/shared.css#4` | `.tab-strip*` (used by catalog AND inspector) |
| 779   | 902 | `components/chat/chat.css#1` | `.bubble*` (base + user/assistant/system), `.bubble-copy-btn`/`.bubble-edit-btn` + `@media (hover: none)`, `.scroll-to-bottom-btn`, `.blob-action-btn` |
| 903   | 924 | `styles/shared.css#5` | `.validation-banner` base (pass/fail variants) |
| 925   | 938 | `styles/shared.css#6` | `.empty-state` |
| 939   | 988 | `styles/shared.css#7` | `.confirm-dialog*` |
| 989   | 1131 | `components/recovery/recovery.css` | `.recovery-panel*`, `.recovery-diff*`, `.recovery-transcript*` + media query |
| 1132  | 1436 | `components/header/header.css` | `.app-root`, `.alert-banner*`, `.app-main`, `.app-header*`, `.header-session-switcher*`, `.user-menu*`, `.header-session-switcher-rename-form`, `.runs-history-item-summary`, `.run-diagnostics*`, `.run-failure-detail*`. **Cascade collision** — see Callout C1 below |
| 1437  | 1535 | `styles/shared.css#8` | `.app-layout*`, `.layout-chat`, `.layout-siderail`, `.banner*`, `.side-rail`, `.side-rail-slot:empty`, `.side-rail-validation-banner` |
| 1536  | 1721 | `components/sidebar/sidebar.css` | `.side-rail-suggestion-*`, `.side-rail-execute-btn`, `.side-rail-export-yaml-btn`, `.side-rail-catalog-btn`, `.catalog-reference-*`, `.completion-bar`, `.graph-mini*` |
| 1722  | 2226 | `components/audit/audit.css` | `.audit-readiness*` + sandwiched `.graph-modal-*` and `.yaml-modal-*` (1933–2025) + `.explain-dialog*` + `.readiness-row-detail*` + trailing `.yaml-modal-body`. **Cascade collision** — see Callout C2 |
| 2227  | 2707 | `components/chat/chat.css#2` | `.inline-run-results*`, `.run-outputs-panel*`, `.run-output-artifact*`, the chat-area `@media (max-width: 760px)` for those, `.chat-input*`, the chat-input `@media`, `.chat-panel*`. **Cascade collision** — see Callout C3 |
| 2708  | 3182 | `components/tutorial/tutorial.css` | `.tutorial-*` (shell, turn, kicker, layer, prompt, error, graph, graph-node, running, draft-progress, progress-bar w/ reduced-motion media, result-table, cancelled-note, mode-fieldset, progress-dot) + `@keyframes tutorial-progress-slide` + bottom `@media (max-width: 760px)` |
| 3183  | 3307 | `styles/tokens.css#2` | `[data-theme="light"]` light-tokens — TOP-LEVEL `^[data-theme="light"]` |
| 3308  | 3367 | `styles/common-yaml-view.css#1` | `.yaml-view*` (view, toolbar, content, pre, line, line-number, line-content) |
| 3368  | 3457 | `styles/themes.css` | `@media (prefers-contrast: more)` and `@media (forced-colors: active)` blocks. The nested `:root` and `[data-theme="light"]` inside the contrast media query are NOT top-level and are correctly ignored by the test regex |
| 3458  | 3591 | `styles/chat-markdown.css` | `.markdown-body*` family (the only banner-less section in this file — it sits between the forced-colors `@media` close and the next banner) |
| 3592  | 3595 | (deleted; banner-only) | "Template Cards" banner stub. **Either keep these 4 lines in the barrel as a literal comment OR move them down to live with the actual `.template-card*` rules at 5419–5549. To preserve byte identity, the comment stays in source order.** Recommendation: include in `components/chat/chat.css#3` as a 4-line file `chat.css#3-banner.css` is over-fragmentation — fold into `chat-markdown.css` trailing comment instead. **Open issue O1** |
| 3596  | 3716 | `styles/common-command-palette.css` | `.command-palette*` |
| 3717  | 3761 | `components/catalog/catalog.css#1` | `.shortcuts-group`, `.shortcuts-list*`, plus the trailing `.command-palette-footer kbd` rule at 3752–3760. **Cascade collision** — see Callout C4 |
| 3762  | 3864 | `components/inspector/inspector-version-selector.css` | `.version-selector*` + trailing `.side-rail-error-banner` (3860–3863). The side-rail-error-banner is small; document as a minor mix |
| 3865  | 4082 | `components/inspector/inspector-graph.css` | `:root .react-flow.react-flow { … }`, `[data-theme="light"] .react-flow.react-flow { … }`, `.react-flow__controls-button:focus-visible`, `.graph-view*`, `.graph-node*`, `.graph-validation-dot`, `.graph-config*` + media query. **Tests must read this file** (see Test impact section) |
| 4083  | 4244 | `components/settings/settings.css` | `.secrets-*` |
| 4245  | 4285 | `styles/common-error.css` | `.error-boundary-*` |
| 4286  | 4853 | `components/chat/chat.css#3` | Migrated-from-inline-style classes for chat: `.message-row*`, `.message-bubble-content*`, `.bubble-action-overlay*`, `.message-edit-*`, `.message-failed-*`, `.message-retry-btn`, `.message-pending`, `.message-tools*`, `.message-sources-created*`, `.inline-source-created-turn*`, `.message-group-separator`, `.tool-call-ribbon*`, `.tool-call-card*`, `.tool-call-audit-id`, `.tool-call-summary/rationale/affects/stale`, `.tool-call-details`, `.tool-call-actions`, `.pending-proposals-banner*`, `.inline-source-fallback-prompt*` + media query, `.pending-overlay-pill`, `.spec-pending-proposal`/`.yaml-pending-summary`/`.runs-pending-proposal` |
| 4854  | 4939 | `components/chat/chat.css#4` | `.composing-row`, `.composing-bubble`, `.composing-pulse`, `.composing-indicator--terminal`, `.composing-terminal-mark`, `.composing-working-view`, `.composing-section`, `.composing-label`, `.composing-title`, `.composing-evidence`, `.composing-text` |
| 4940  | 5057 | `components/execution/execution.css` | `.progress-container`, `.progress-ws-banner`, `.progress-status-*`, `.progress-cancel-btn`, `.progress-bar-outer`, `.progress-counters*`, `.progress-routing-summary`, `.progress-cancelled-msg`, `.progress-failed-msg`, `.progress-errors-*`, `.progress-error-*` |
| 5058  | 5199 | `components/blobs/blobs.css` | `.blob-row*` + `.blob-manager*` |
| 5200  | 5398 | `components/catalog/catalog.css#2` | `.plugin-card*` + `.audit-icon*` (audit-icon is co-used by catalog cards) |
| 5399  | 5418 | `styles/common-yaml-view.css#2` | `.yaml-loading`, `.yaml-toolbar-btn` + `[data-copied]` attribute selector |
| 5419  | 5549 | `components/chat/chat.css#5` | `.template-cards*`, `.template-card*` + their media queries (760px and 520px) |
| 5550  | 5641 | `styles/shared.css#9` | `.validation-banner-*` detailed migrations (content, header, summary, checks, warnings-section, warnings-title, warnings-list, warn-item, fail-title, fail-list, error-item, suggestion, component-btn, component-btn--warning/--error) |
| 5642  | 5940 | `components/catalog/catalog.css#3` | `.catalog-backdrop`, `.catalog-drawer`, `.catalog-header*`, `.catalog-close-btn`, `.catalog-search*`, `.catalog-tab*`, `.catalog-list`, `.filter-chip*`, `.catalog-status-message*`, `.inline-chat-source-entry*` |
| 5941  | 7335 | `components/chat/guided/guided.css` | All `.guided-*` widgets across InspectAndConfirm/MultiSelectWithCustom/SchemaForm/ProposeChain/Recipe/ExitToFreeform/GuidedHistory/CompletionSummary + the trailing `@media (prefers-reduced-motion: reduce)` that silences all guided-* transitions |

**Total = 7335 lines across 53 contiguous source ranges → 35 physical
files + 1 barrel header comment** (after folding adjacent same-file ranges:
the table shows 35 file-entries; sub-numbered rows like `chat.css#1` are
distinct physical files since they cannot be reordered).

## Barrel import order (`styles/index.css`)

The barrel is the only file App.tsx imports. Order matches the source-line
order in App.css so concatenation is byte-identical (modulo the leading
header-comment block which the barrel may emit inline or via a 7-line file
`styles/_header.css`).

1. `styles/_header.css` (or inline comment)
2. `styles/tokens.css` (lines 8–209)
3. `styles/base.css`
4. `styles/animations.css`
5. `styles/shared/1-type-badges.css` (480–540)
6. `styles/shared/2-status-badges.css` (541–625)
7. `styles/shared/3-buttons.css` (626–746)
8. `styles/shared/4-tab-strip.css` (747–778)
9. `components/chat/chat-1-bubble.css` (779–902)
10. `styles/shared/5-validation-banner-base.css` (903–924)
11. `styles/shared/6-empty-state.css` (925–938)
12. `styles/shared/7-confirm-dialog.css` (939–988)
13. `components/recovery/recovery.css` (989–1131)
14. `components/header/header.css` (1132–1436)
15. `styles/shared/8-layout-and-sidebar-base.css` (1437–1535)
16. `components/sidebar/sidebar.css` (1536–1721)
17. `components/audit/audit.css` (1722–2226)
18. `components/chat/chat-2-input-and-panel.css` (2227–2707)
19. `components/tutorial/tutorial.css` (2708–3182)
20. `styles/tokens-light.css` (3183–3307) — or merge into `tokens.css` (see Open issue O2)
21. `styles/common-yaml-view-1.css` (3308–3367)
22. `styles/themes.css` (3368–3457)
23. `styles/chat-markdown.css` (3458–3591) — see O1 for whether 3592–3595 stays here
24. `styles/common-command-palette.css` (3596–3716)
25. `components/catalog/catalog-1-shortcuts.css` (3717–3761)
26. `components/inspector/inspector-version-selector.css` (3762–3864)
27. `components/inspector/inspector-graph.css` (3865–4082)
28. `components/settings/settings.css` (4083–4244)
29. `styles/common-error.css` (4245–4285)
30. `components/chat/chat-3-message-bubble-and-tools.css` (4286–4853)
31. `components/chat/chat-4-composing.css` (4854–4939)
32. `components/execution/execution.css` (4940–5057)
33. `components/blobs/blobs.css` (5058–5199)
34. `components/catalog/catalog-2-plugin-card.css` (5200–5398)
35. `styles/common-yaml-view-2.css` (5399–5418)
36. `components/chat/chat-5-template-cards.css` (5419–5549)
37. `styles/shared/9-validation-banner-detail.css` (5550–5641)
38. `components/catalog/catalog-3-drawer.css` (5642–5940)
39. `components/chat/guided/guided.css` (5941–7335)

**Note on tokens.css:** the dark `:root` block (8–209) and the light
`[data-theme="light"]` block (3183–3307) are at different positions in
source order. For byte-identical reconstruction they must be emitted at
those positions in the barrel. **However**, the operator's stated intent is
"tokens.css (the variable layer) global." The pragmatic move is to emit
**two physical token files** (`tokens-dark.css`, `tokens-light.css`) at the
two source positions, AND have the CSS-reading tests target both. See
Test impact and Open issue O2.

## Cross-file selector overlap audit (true cascade dependencies)

A `grep | sort | uniq -d` pass against bare class declarations (selectors of
the shape `^\.classname\s*{`) returned 11 selectors declared at multiple
positions. **All 11 are adjacent declarations within the same physical file**
(e.g. `.guided-inspect-edit-btn` at 6250 and again at 6270 — both inside
`guided.css`'s 5941–7335 range). No bare selector is declared in two
different physical files.

A broader Python pass (every line beginning `\s*\.classname`) surfaced **19
class names whose occurrences span two or more physical files.** These are
the real cascade dependencies — and every one of them is **preserved by the
barrel import order** specified below, because the order matches App.css
source order.

| selector | files (in barrel order) | nature | preserved? |
|---|---|---|---|
| `.btn` | tokens.css (comment text, not a selector), shared.css#3 | false positive — `.btn` appears in a comment string inside a `:root` rule | n/a |
| `.type-badge`, `.type-badge-source/-transform/-gate/-sink/-aggregation/-coalesce` | shared.css#1, themes.css | shared#1 defines (480–540); themes overrides via `@media (prefers-contrast: more)` and `(forced-colors: active)` at 3389–3413 | yes — themes.css imports AFTER shared.css#1 |
| `.validation-banner-pass`, `.validation-banner-fail` | shared.css#5, themes.css | shared#5 defines (912, 918); themes overrides via forced-colors at 3399, 3400 | yes — themes imports after shared#5 |
| `.alert-banner` | header.css, themes.css | header defines (1140); themes overrides via forced-colors at 3401 | yes — themes imports after header |
| `.tutorial-graph-chevron`, `.tutorial-graph-node`, `.tutorial-progress-bar`, `.tutorial-progress-dot`, `.tutorial-progress-dot--active` | tutorial.css, themes.css | tutorial defines (2708–3182); themes overrides via forced-colors at 3432–3454 | yes — themes imports after tutorial |
| `.chat-panel` | chat-1-bubble.css, chat-2-input-and-panel.css | chat#1 has a descendant selector `.chat-panel:has(.inline-run-results) .scroll-to-bottom-btn` at line 877 that references `.chat-panel` as ancestor; chat#2 defines `.chat-panel` at 2570 | yes — both rules apply; cascade order matches source |
| `.yaml-toolbar-btn` | themes.css, common-yaml-view-2.css | themes has `[data-copied="true"]` for forced-colors at 3423; yaml-view-2 defines at 5409 and 5413 | yes — themes imports BEFORE yaml-view-2; same-specificity later rule wins, matching original source order |
| `.command-palette-footer` | common-command-palette.css, catalog-1-shortcuts.css | palette defines at 3708; catalog#1 has `.command-palette-footer kbd` descendant rule at 3752 | yes — palette imports before catalog#1 |

**Conclusion: every cross-file selector dependency is preserved by the
barrel import order specified.** No additional restructuring is needed.

## Adjacency callouts (NOT cascade-collisions — kept together for byte-identity only)

The following ranges contain rules from multiple component areas in the same
source range, but the rules **do not share selectors** with each other. They
are sandwiched together in App.css for historical reasons, not for cascade
reasons. Splitting them across files would not change cascade — it would
only change discoverability. Under **Path A** (strict byte-identity) they
stay in one file; under **Path B** (logical grouping) they can be safely
separated.

**A1 — Lines 1132–1436 (`components/header/header.css`).** This banner ("App
Root & Alert Banners") mixes `.app-root` + `.alert-banner` + `.app-header*` +
`.header-session-switcher*` + `.user-menu*` + (at the tail, 1374–1436)
`.runs-history-item-summary` + `.run-diagnostics*` + `.run-failure-detail*`.
The tail rules belong to **execution** (or sessions, depending on how the
header session switcher pulls in run history). The selector audit confirms
none of these rules overlap selectors with execution.css (4940–5057) — the
execution.css rules are all `.progress-*`. **Under Path A:** keep the whole
305-line block in `header.css`; document the secondary residents at the top.
**Under Path B:** the tail (1374–1436) can move to `components/execution/execution.css`
or a new `sessions.css`, since no selector overlap exists.

**A2 — Lines 1722–2226 (`components/audit/audit.css`).** Audit-readiness
rules are interrupted at 1933–2025 by `.graph-modal-*` (sidebar/inspector)
and `.yaml-modal-*` (sidebar/common), then resume with `.explain-dialog*`
and `.readiness-row-detail*`, and finally close with `.yaml-modal-body` at
2221–2225 (trailing remnant of yaml-modal). The selector audit shows no
overlap between `.audit-readiness-*` and `.graph-modal-*` / `.yaml-modal-*`,
so this is adjacency, not cascade. **Under Path A:** keep in `audit.css`;
document modals as second tenant. **Under Path B:** the graph-modal and
yaml-modal selectors can move to `components/common/modals.css` cleanly.

**A3 — Lines 2227–2707 (`components/chat/chat-2-input-and-panel.css`).** The
"Chat Input" banner opens with `.inline-run-results*` + `.run-outputs-panel*`
+ `.run-output-artifact*` (execution-named but chat-mounted) then jumps to
`.chat-input*` and `.chat-panel*`. The selector audit confirms one real
cross-file dependency here: `.chat-panel:has(.inline-run-results) .scroll-to-bottom-btn`
at line 877 (chat#1) depends on `.chat-panel` defined at 2570 (chat#2). The
barrel imports chat#1 before chat#2, matching original source order — preserved.
**Under Path A:** keep in `chat#2`; document `inline-run-results / run-outputs-panel
/ run-output-artifact` as chat-mounted execution renderers. **Under Path B:**
these can be merged into one chat.css file.

**A4 — Lines 3717–3761 (`components/catalog/catalog-1-shortcuts.css`).** The
banner is "Shortcuts Help" (`.shortcuts-group`, `.shortcuts-list*`) but the
last 9 lines (3752–3760) are `.command-palette-footer kbd` — a descendant
rule whose ancestor `.command-palette-footer` is defined at 3708 in
`common-command-palette.css`. The selector audit confirms this is a real
cross-file ancestor dependency, preserved by barrel order (palette before
catalog#1). **Under Path A:** leave the kbd rule in `catalog-1-shortcuts.css`.
**Under Path B:** move it to `common-command-palette.css` next to its
ancestor; cascade still preserved because both rules end up in the same
file under Path B.

**A5 — Lines 3860–3863 (`components/inspector/inspector-version-selector.css`).**
A single 4-line `.side-rail-error-banner` rule (sidebar concern) is the last
rule before the Graph View banner. **Under Path A:** leave in the version-selector
file; document at top. **Under Path B:** move to `sidebar.css`.

**A6 — Banner-only stub at 3592–3595** (handled separately as Open Issue O1).

**A7 — Lines 5200–5398 (`components/catalog/catalog-2-plugin-card.css`)
holds `.audit-icon*` (5249–5285).** Audit-icon is a catalog-card visual
element (all `audit-icon` usages in TSX live under `components/catalog/`),
correctly placed. **No collision.**

## Test impact

Three tests currently `readFileSync("src/App.css", "utf8")`:

- `src/styles/colorContrast.test.ts` (line 5) — parses `^:root { … }` and
  `[data-theme="light"] { … }` blocks. **After split:** swap to
  `readFileSync("src/styles/tokens.css")` (which contains the dark `:root`
  at the top and the light `[data-theme="light"]` immediately below — but
  see Open Issue O2 for whether they're one file or two).
- `src/styles/statusBadgeAccessibility.test.ts` (line 5) — same regex
  parser, same swap target.
- `src/components/inspector/GraphView.test.tsx` (line 353) — asserts
  `toContain(":root .react-flow.react-flow")` AND a regex for
  `[data-theme="light"] .react-flow.react-flow { … --xy-minimap-mask-background-color-default: rgba(15, 45, 53, 0\.12);`.
  These selectors live in `inspector-graph.css` (3865–4082). **Swap to**
  `readFileSync("src/components/inspector/inspector-graph.css")`.

The writer must touch these three files in the same commit as the split or
the tests will fail.

## Verification command for the writer

After the split, concatenate the physical files in barrel order and diff
against the original:

```bash
# From src/elspeth/web/frontend/src/
cat \
  styles/_header.css \
  styles/tokens.css \
  styles/base.css \
  styles/animations.css \
  styles/shared/1-type-badges.css \
  styles/shared/2-status-badges.css \
  styles/shared/3-buttons.css \
  styles/shared/4-tab-strip.css \
  components/chat/chat-1-bubble.css \
  styles/shared/5-validation-banner-base.css \
  styles/shared/6-empty-state.css \
  styles/shared/7-confirm-dialog.css \
  components/recovery/recovery.css \
  components/header/header.css \
  styles/shared/8-layout-and-sidebar-base.css \
  components/sidebar/sidebar.css \
  components/audit/audit.css \
  components/chat/chat-2-input-and-panel.css \
  components/tutorial/tutorial.css \
  styles/tokens-light.css \
  styles/common-yaml-view-1.css \
  styles/themes.css \
  styles/chat-markdown.css \
  styles/common-command-palette.css \
  components/catalog/catalog-1-shortcuts.css \
  components/inspector/inspector-version-selector.css \
  components/inspector/inspector-graph.css \
  components/settings/settings.css \
  styles/common-error.css \
  components/chat/chat-3-message-bubble-and-tools.css \
  components/chat/chat-4-composing.css \
  components/execution/execution.css \
  components/blobs/blobs.css \
  components/catalog/catalog-2-plugin-card.css \
  styles/common-yaml-view-2.css \
  components/chat/chat-5-template-cards.css \
  styles/shared/9-validation-banner-detail.css \
  components/catalog/catalog-3-drawer.css \
  components/chat/guided/guided.css \
  | diff - <(git show HEAD:src/elspeth/web/frontend/src/App.css)
```

Empty diff = success.

A second verification pass should re-run the three CSS-reading tests
**against their new targets** and a full `vitest` run to confirm no Cypress
or visual-regression coverage references App.css by path elsewhere.

```bash
grep -RIn "App\.css" src/ test/ 2>/dev/null
```

## Open issues / risks

**O1 — banner-only stub at lines 3592–3595.** This is 4 lines of comment
("Template Cards (onboarding quick-start)") followed by the next banner
("Command Palette"). There are NO rules between them — the Template Cards
banner is orphaned; the actual `.template-card*` rules live at 5419–5549.
Splitting the comment off into its own file at this byte position is
silly. **Two options:**
(a) Append the 4-line banner stub to `styles/chat-markdown.css` as trailing
content (since markdown ends at 3591 and the next file begins at 3596) —
preserves byte identity, but mixes file content.
(b) Emit `styles/_section-stub-template-cards.css` as a 4-line file.
**Recommendation: (a)**; document at the bottom of `chat-markdown.css` that
the trailing 4 lines are an orphaned banner preserved for byte identity.
This is a debt note: a follow-up PR could move the banner down to live
with the actual template-card rules at 5419 and the chat-markdown file
would become cleanly bounded. That follow-up is **out of scope** for this
behaviour-preserving split.

**O2 — token file structure under Path A.** Strict byte identity requires
the dark `:root` block (8–209) at byte position ~8 and the light
`[data-theme="light"]` block (3183–3307) at byte position ~3183, with ~3000
lines between them. Under Path A this forces two physical files
(`tokens-dark.css` and `tokens-light.css`) at their respective barrel
positions. The operator's stated intent ("tokens.css … global") is still
honoured because both files sit under `styles/` and contain only token
declarations. Test parsers must read both. **Under Path B** the two blocks
can collapse into one `tokens.css` and the test parsers read one file.

**O3 / O4 / O5 — multi-fragment files under Path A.** chat.css is split
across 5 physical files; catalog.css across 3; shared.css across 9. This is
forced by Path A's byte-identity requirement and has no internal cascade
risk (the cross-file audit above shows it). The cost is discoverability:
a reader searching for `.validation-banner*` rules has to grep two files.
**Mitigation under Path A:** add `styles/README.md` listing
selector→file mappings, generated by a one-time
`grep -n '^\.' styles/*.css components/**/*.css` sweep after the split.
**Under Path B** all three areas become single files and this concern
goes away.

**O6 — `@media (max-width: 760px)` and similar responsive rules are
scattered.** App.css has 7 separate `@media (max-width: 760px)` blocks
(plus one `(max-width: 520px)` and one `(max-width: 900px)`). Each lives
next to the component rules it targets and goes into the same per-component
file. This is correct for cascade — but anyone debugging responsive
breakpoints across the app will need to grep across many files instead of
opening one. **Not a blocker; surface as documentation note.**

**O7 — `:root .react-flow.react-flow` at line 3868 is a token-override
nested under a non-token banner ("Graph View").** This block declares ~22
react-flow internal custom properties (`--xy-*`) that override react-flow's
defaults. They are token-shaped (custom properties) but live in the
graph file because they only apply to the React Flow canvas selector. The
test `GraphView.test.tsx` reads these via path. **Decision:** keep in
`inspector-graph.css`, not `tokens.css`, because (a) they are
selector-scoped, not global, and (b) the test contract is by file path.
Document at the top of `inspector-graph.css`.

**Risk — backstop visual regression test.** This plan preserves byte-identity
of the concatenated CSS. A subtle regression could still occur if Vite's
`@import` resolver inlines files differently than `cat` does (e.g.
introduces source-map comments, re-orders character-set declarations). The
writer should verify that production build output's CSS (under
`dist/assets/*.css`) is byte-identical between pre-split and post-split
builds, not just the source files. If Vite hashes/reorders, the assertion
becomes "produces the same DOM render under jsdom for a stable component
snapshot," which is weaker. Surface this to the operator before splitting.

## Confidence Assessment

**Confidence: Medium-High.**

What I verified directly:
- Line counts and banner positions (grepped, read, totalled to 7335).
- The three CSS-reading test files' regex behaviour (read the files,
  confirmed `^:root` is anchored multiline so nested `:root` inside
  `@media` blocks is correctly ignored).
- Class-prefix ownership via grep against `components/<area>/`.
- The dark/light token blocks contain only custom-property declarations
  (no rules that would need migration to `themes.css`).

What I did NOT verify:
- Whether Vite's import resolver produces a byte-identical build output
  (only confirmed source byte-identity).
- Whether any tests outside the three named files reference App.css by
  selector content rather than by path.
- Whether any TSX uses a class name that I assigned to one file when the
  class is also used (with different specificity) from another area's
  TSX in a way that creates cross-file specificity dependencies.
- I did not exhaustively cross-grep every single class in App.css
  against every TSX in components/ — only the 67 distinct 2-segment
  prefixes shown in the bulk-grep table during planning.

## Risk Assessment

**Residual risk:**
- Vite or postcss may insert characters or reorder during `@import`
  resolution, breaking byte identity at the bundled level even if source
  files concatenate byte-identically. Mitigation: diff `dist/` output
  before declaring done.
- The cross-file selector overlap audit (run during planning) returned 19
  class names whose declarations span two or more physical files. **All
  19 are preserved by the barrel import order** specified in this plan
  (themes.css imports after every file whose selectors it overrides;
  palette imports before catalog#1; chat#1 imports before chat#2). The
  writer should re-run the audit script (`grep '^\s*\.[a-zA-Z][a-zA-Z0-9_-]*'
  App.css | sort | uniq -c` plus the Python pass shown in the planning
  transcript) after the split to confirm the same 19 selectors and no
  new ones appear.
- The `@media (prefers-contrast: more)` block at 3372 contains both
  nested-`:root` token overrides AND component-rule overrides
  (`.type-badge`, `:focus-visible`). The whole block lands in `themes.css`.
  Test parsers using anchored `^:root` regex (the three CSS-reading tests)
  do NOT pick up these nested `:root` overrides — the regex requires
  no leading whitespace, and the nested `:root` is indented under the
  `@media` block. **Verified** by reading the test parsers during
  planning. If a future test wants to verify high-contrast token values,
  it must read `themes.css` and parse `@media (prefers-contrast: more)`
  nested `:root` blocks separately.

## Information Gaps

- I did not verify the Vite build output preservation (bundle-level
  byte-identity). The writer should confirm `dist/assets/*.css` is
  byte-identical between pre-split and post-split builds, not just the
  source files.
- I did not exhaustively cross-grep every single class against every TSX
  for runtime use; I only audited declarations within App.css. The
  cross-file declaration audit IS complete (19 selectors found, all
  cascade-preserved).
- I did not verify whether any e2e Cypress / Playwright tests reference
  App.css selectors by path. The writer should run
  `grep -RIn "App\.css" src/ test/ playwright/ cypress/ 2>/dev/null`
  before declaring done.
- I did not verify whether the README at `components/chat/guided/` or
  any other area README references App.css by line number — if so, those
  references need updating in the same commit as the split.
- I did not audit `@media` block placement under Path B. Path A places
  every `@media` block in the file whose rules it modifies, which is
  cascade-correct. Path B would require explicit `@media`-block routing
  analysis (some media queries target rules in multiple areas, e.g. the
  global `@media (prefers-reduced-motion)` at 455–479 covers chat,
  execution, and react-flow).

## Caveats

- The "~18-20 target files" guidance in the original brief is structurally
  incompatible with byte-identical reconstruction against App.css's
  current ordering. The plan delivers 22 logical files / 35 physical
  files; the difference is forced by cascade preservation, not by
  over-splitting. If the operator wants fewer physical files, they
  must first land a reordering PR that makes each area's rules
  contiguous in App.css, and then re-run a (much simpler) split.
- The plan preserves cascade by ordering, not by selector specificity.
  If a future edit changes specificity of a rule in one of the multi-file
  areas (e.g. chat-1 vs chat-3), the writer should verify the cascade
  intent is still preserved.
- "Multi-tenant" files (header.css, audit.css, chat-2) need a top-of-file
  comment block explicitly listing the secondary residents. This is
  load-bearing institutional memory — the next person to edit these
  files needs to know they are not single-area.
- The plan does not delete any rules. Any rule that looks dead
  (e.g. `.template-cards-banner` stub at 3592–3595) is preserved as-is
  for byte identity. Cleanup is a separate concern.
