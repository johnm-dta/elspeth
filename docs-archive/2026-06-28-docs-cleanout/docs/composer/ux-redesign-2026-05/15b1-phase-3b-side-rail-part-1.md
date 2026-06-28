# Phase 3B — IA Cleanup: graph mini, YAML export modal, Catalog button move, header session switcher (additions), hash-router migration, InspectorPanel teardown (Part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the second half of Phase 3 — fill the side-rail slots reserved by 15a with the graph mini-view, the YAML export modal, and the relocated Catalog button; promote the version selector to the header next to the session switcher; rewrite the hash router to drop the inspector-tab vocabulary and add explicit redirects for stale `#/{id}/spec`, `#/{id}/runs`, `#/{id}/graph`, and `#/{id}/yaml` deep links; and finally delete `InspectorPanel.tsx` once nothing is left inside it. Every commit leaves the app launchable and green.

**Architecture:** Frontend chrome refactor — same trust-tier posture as 15a. No new boundary. The graph data, YAML export, version metadata, and catalog data already pass through their respective Tier 3 → Tier 2 boundaries at the backend; this plan only relocates their *rendering surfaces*. No backend changes.

**Tech Stack:** React + Zustand + Vitest + testing-library. The new modal pattern mirrors `SecretsPanel.tsx` (focus-trap, escape-to-close, backdrop click) and `ConfirmDialog.tsx` (dialog role, aria-labelledby). No new dependencies.

> **Phase 3 block notice (added 2026-05-17; target corrected 2026-05-17):** This plan is one of four (15a1, 15a2, 15b1, 15b2) that together comprise the Phase 3 IA-cleanup work. **All four land as a single block on the dedicated Phase 3 worktree/branch for this IA-cleanup block** and **merge as one PR**. The canonical target for this packet is worktree `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on branch `feat/composer-phase-3-ia-cleanup`, created from `RC5.2` with `git worktree add .worktrees/composer-phase-3-ia-cleanup -b feat/composer-phase-3-ia-cleanup RC5.2` if it does not already exist. Do **not** use the old Phase 2A/2B/2C worktree or branch (`.worktrees/phase-2a-backend`, `feat/composer-phase-2a-backend`); those references are stale. Phrases below like "Phase 3A" / "15a" mean "earlier tasks in the same Phase 3 branch," not a prior cycle. The 15a1→15a2→15b1→15b2 split is task sequencing and document organisation, not delivery sequencing — sequencing within the block still matters per task ordering.
>
> **Subagent dispatch discipline.** Every subagent prompt for this packet MUST start with this CWD-discipline preamble as its first Bash call: `cd /home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup && pwd && git rev-parse --abbrev-ref HEAD`; expected branch: `feat/composer-phase-3-ia-cleanup`. If the operator explicitly chooses a different Phase 3 worktree/branch, update this notice in **all four** 15a1/15a2/15b1/15b2 files before dispatch and use the chosen concrete values in every subagent prompt. The prompt must also state that `.worktrees/phase-2a-backend` and `feat/composer-phase-2a-backend` are stale Phase 2 targets and forbidden for Phase 3 work. Use absolute paths only thereafter for every Read/Bash/Grep. Bash `cd` does NOT persist between tool calls — relative paths can silently read the wrong branch.

**Sibling plans:**
- Predecessor (task order, same branch): [15a1-phase-3a-removals-part-1.md](15a1-phase-3a-removals-part-1.md) / [15a2-phase-3a-removals-part-2.md](15a2-phase-3a-removals-part-2.md) — Phase 3B tasks consume artifacts produced by 15a tasks (the `SideRail.tsx` scaffold, `HeaderSessionSwitcher`, `InlineRunResults`, the deletions of `SpecView` / `RunsView` / `SessionSidebar`, and the auto-validate effect). **Within the shared branch, complete 15a tasks before starting 15b tasks.** Since the whole block ships as one PR, "shipped" is not a meaningful gate between 15a and 15b — only task ordering is.

**Directory convention adjudication.** 15a created `SideRail.tsx` under `src/components/common/`. Phase 3B introduces enough new side-rail-specific components (`ExecuteButton`, `GraphMiniView`, `ExportYamlButton`, `CatalogButton`) that they warrant their own directory, matching the existing UI-area convention (`inspector/`, `chat/`, `sessions/`, `auth/`, `audit/`). **Task 0 below** moves `SideRail.tsx` and `SideRail.test.tsx` from `common/` into a new `sidebar/` directory in a single commit (per CLAUDE.md "No Legacy Code Policy" — no re-export shim). All Phase 3B paths thereafter reference `src/components/sidebar/`. The modal components (`GraphModal`, `ExportYamlModal`) are placed under `sidebar/` from the start; they are not parked under `inspector/`.
- Successor: Phase 2 (audit-readiness panel) fills the side-rail's `audit-readiness` slot; Phase 6 (completion gestures) fills the `completion-bar` slot; Phase 7 (catalog reshape) edits `CatalogDrawer`'s internal behaviour. 15b is chrome-only and does not block any of those.

**Part split:** This file covers Tasks 1–5 (all additive work plus the Catalog button move). Tasks 6–10 (hash-router rewrite, CommandPalette updates, keyboard shortcuts, InspectorPanel teardown, cleanup) plus Risks, Memory references, and Review history are in [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md).

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md) §B (Phase 3) and §A (H1, H2 calls).

**Design spec:** [03-target-information-architecture.md](03-target-information-architecture.md).

---

## Scope boundaries

**In scope (this plan, 15b):**

- New `GraphMiniView.tsx` mounted in `SideRail`'s `graph-mini` slot. Renders a compact, non-interactive snapshot of the current `compositionState` DAG (~160px tall, full rail width). Clicking the mini opens `GraphModal.tsx` — a modal hosting the existing full `GraphView.tsx`. H2 is resolved as **(a) — click opens modal** per the open-questions doc.
- New `GraphModal.tsx` (modal wrapper around `GraphView`). Mirrors the `SecretsPanel` modal shape: focus trap, escape-to-close, backdrop click, role=dialog, aria-labelledby. Reuses `GraphView` verbatim — no fork; the modal is a container.
- New `ExportYamlButton.tsx` mounted in `SideRail`'s `export-yaml` slot. Renders a button labelled "⬇ Export YAML"; clicking opens `ExportYamlModal.tsx`.
- New `ExportYamlModal.tsx` — modal hosting the existing `YamlView.tsx` content. Same focus-trap shape as `GraphModal`. `YamlView` already carries the copy-to-clipboard + download-as-file affordances; the modal exposes them at first-class size.
- New `CatalogButton.tsx` mounted in `SideRail`'s `catalog-button` slot. Renders a button labelled "📋 Catalog (reference)". Clicking dispatches the existing `OPEN_CATALOG_EVENT`. The `CatalogDrawer` itself continues to listen on that event and renders unchanged — Phase 7 reshapes its internal content; 15b is purely the button-relocation chrome change.
- New `HeaderVersionSelector.tsx` mounted in `AppHeader` next to `HeaderSessionSwitcher`. Reuses the `VersionSelector` logic currently embedded in `InspectorPanel.tsx` (the inner `VersionSelector` component at lines ~53–313 is **extracted** into its own file). Renamed user-visible label: "Composition history" per design doc 03.
- `useHashRouter` rewrite: `VALID_TABS` is replaced with a `VALID_FRAGMENT_VERBS` set; the verb vocabulary changes from inspector-tab IDs to action verbs:
  - `spec` → no-op (redirect to canonical `#/{id}`); historical fragments are silently rewritten via `replaceState`.
  - `runs` → no-op (same; run results render inline).
  - `graph` → opens the graph modal (`OPEN_GRAPH_MODAL_EVENT`).
  - `yaml` → opens the YAML export modal (`OPEN_YAML_MODAL_EVENT`).
  - Any unrecognized fragment is stripped.
  - Default tab `"spec"` is removed; the canonical no-fragment hash is `#/{sessionId}`.
- Inspector teardown: with Graph + YAML extracted into their own surfaces and Spec / Runs / Validate already removed in 15a, the inspector panel has no content left to host. `InspectorPanel.tsx` is **deleted**. The `Layout` component is simplified from `sidebar / chat / inspector` (the 15a-renamed `siderail` slot) to `chat / siderail` only — the inspector-grid-column and its resize handle are removed; `SideRail` now owns that column directly.
- `App.tsx` cleanup: the `InspectorPanel` import is removed, the `<InspectorPanel />` mount is replaced by `<SideRail />` (already added in 15a as the slot scaffold; in 15b it becomes the direct slot in Layout). Keyboard shortcuts `Alt+1`/`Alt+2`/`Alt+3`/`Alt+4` are retired (`Alt+2` for Graph becomes `Ctrl+Shift+G` opening the modal; `Alt+3` for YAML becomes `Ctrl+Shift+Y` opening the modal; `Alt+1` Spec and `Alt+4` Runs are simply gone). `ShortcutsHelp` is updated.
- `CommandPalette` updates: `tab-spec`, `tab-graph`, `tab-yaml`, `tab-runs` command IDs are renamed/retargeted. `tab-graph` becomes `open-graph-modal` (Ctrl+Shift+G). `tab-yaml` becomes `open-yaml-export` (Ctrl+Shift+Y). `tab-spec` and `tab-runs` are deleted. Sessions section is unchanged (already covers H1 fallback per 15a).
- `SWITCH_TAB_EVENT` **is deleted in Task 10** of this plan — the export and all dispatch/listener sites are removed in the cleanup pass. `CommandPalette.tsx` is updated to dispatch `OPEN_GRAPH_MODAL_EVENT` / `OPEN_YAML_MODAL_EVENT` directly. No deferral to Phase 6.
- `App.test.tsx` smoke render runs at the end of every task.
- New `SideRail.test.tsx` slot-presence assertions are extended to cover the three real components (graph mini, export YAML, catalog button).

**Out of scope (deferred — fills the slots 15b creates):**

- Phase 2: audit-readiness panel content fills the `audit-readiness` slot.
- Phase 6: completion-bar verbs Save-for-review / Run pipeline fill the `completion-bar` slot. The Execute button stays in its 15a transitional location (no longer in `InspectorPanel.tsx` since that's gone — 15a is amended below in Task 2 to relocate Execute) until Phase 6 lands the completion bar.
- Phase 7: Catalog drawer content reshape. 15b only moves the *button*; the drawer's interior is Phase 7.
- Phase 4: first-run tutorial.
- Mode-related layout changes beyond what 15a inherited from Phase 1B.

**Out of scope (other phases — flagged so executors don't preemptively do them):**

- Real-time graph mini updates beyond the existing `compositionState` change subscription. The graph mini re-renders whenever `compositionState.version` increments — same trigger as the full graph. No new push channel.
- Multi-run YAML export. The Export YAML modal exports the *current* composition only. Historical-version YAML is reachable via the Composition history selector (revert + export).
- Side-rail width resize/persistence. 15a keeps the legacy `elspeth_inspector_width` key only during the transitional Layout rename so existing users do not lose width preferences mid-branch. 15b2 Task 9 removes the resize path and all reads/writes of that key in Phase 3B, hardcoding the side rail to `SIDERAIL_WIDTH = 320`. There is no Phase 6 deprecation step for this key.

## Trust tier check

Phase 3B is **frontend chrome only**. Same posture as 15a:

- No new external-data ingestion. `GraphMiniView` reads `useSessionStore.compositionState` (already Tier 2 after the backend boundary). `ExportYamlModal` hosts `YamlView`, which fetches `/api/sessions/{id}/state/yaml` — that endpoint already validates at the backend (Tier 3 → Tier 2) before returning the document. `HeaderVersionSelector` reads the existing `stateVersions` already loaded by `loadStateVersions`.
- No new audit-recorder events. Opening / closing a modal is UI state, not auditable activity. The underlying actions (revert via `revertToVersion`, validate, execute) already record through their existing audit boundaries.
- No new persistent state. The graph mini is not collapsible in Phase 3B; do not add an `elspeth_graph_mini_collapsed` key or any other side-rail preference key. Task 9 removes the transitional `elspeth_inspector_width` use and does not replace it.

Per [CLAUDE.md](../../../CLAUDE.md) "Defensive Programming: Forbidden", store accesses go through typed selectors directly (`useSessionStore((s) => s.compositionState)`). No `try`/`catch` around store calls is introduced. The hash-router rewrite preserves the existing fail-closed behaviour: an unrecognized fragment is rewritten to the canonical no-fragment form via `replaceState`, exactly as 15a's pre-write code did for invalid `VALID_TABS` entries.

## Sequencing and dependencies

15b tasks are ordered so that **every commit leaves the app in a state where**:

1. `npm test` passes (vitest + the existing testing-library suite).
2. `App.test.tsx`'s smoke render succeeds.
3. A human opening `elspeth.foundryside.dev` (`project_staging_deployment`) can compose a pipeline using whichever surfaces survive at that point — and can reach Graph and YAML through both the surviving inspector mount **and** the new side-rail slots until the inspector is deleted in Task 9.

The order is:

```
Task 0 — Move SideRail.tsx + SideRail.test.tsx from components/common/ to components/sidebar/
Task 1 — Extract VersionSelector + Promote to header (additive; old VersionSelector mount survives until Task 9)
Task 2 — Relocate Execute button to a transitional SideRail.executeButton slot (additive removal-from-inspector)
Task 3 — Add GraphModal + GraphMiniView                 (additive; inspector Graph tab still works)
Task 4 — Add ExportYamlModal + ExportYamlButton         (additive; inspector YAML tab still works)
Task 5 — Move Catalog button from inspector to SideRail (chrome relocation; OPEN_CATALOG_EVENT unchanged)
Task 6 — useHashRouter rewrite + redirect stale fragments (deep-link migration; modals replace tab dispatches)
Task 7 — CommandPalette rename / retarget               (palette commands now open modals; tab-* removed)
Task 8 — App.tsx keyboard shortcuts: retire Alt+1/2/3/4, add Ctrl+Shift+G / Y, update ShortcutsHelp
Task 9 — Delete InspectorPanel.tsx + Layout grid simplification (final removal; smoke must still pass)
Task 10 — Cleanup pass: remove now-orphaned imports, dead-code constants, tests of removed surfaces
```

Tasks 0–5 are in this file. Tasks 6–10 are in [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md).

Each task is TDD-shaped: failing test, implementation, passing test, smoke render, commit.

## Open scope questions resolved by this plan

1. **What happens to the Execute button mid-phase?** Resolution: **transitional SideRail slot**. 15a explicitly kept Execute in the inspector header. 15b's Task 9 deletes the inspector entirely, so Execute must move *before* Task 9. The cleanest landing place mid-flight is a `SideRail.executeButton` slot — a `<button onClick={execute}>Run pipeline</button>` styled to match the placeholder until Phase 6 replaces it with the persona-aware completion bar. This is documented in design doc 03's "Execute button" row: target is the completion bar in the side rail. 15b lands the *button in the side rail*; Phase 6 lands the *completion bar around it*. (See Task 2.)
2. **Where does the Validation banner go after the inspector is deleted?** Resolution: **migrate to a new `SideRail.validationBanner` slot in Task 9**. The validation banner currently renders between the inspector tab strip and tab content (`InspectorPanel.tsx:631–638`). With the inspector deleted, it has no host. Phase 2's audit-readiness panel will subsume the dot indicator and per-component clickthrough; until Phase 2 ships, the banner lives in a thin slot inside `SideRail` above the graph mini. The banner's `onComponentClick` still fires `selectNode` for the GraphView highlight, but tab-switch navigation was already nooped in 15a.
3. **Should the graph mini be interactive?** Resolution: **no**. Per design doc 03 "Graph view (mini)" — "Persistent in side rail, small". The mini is a verification surface, not an editing surface. Clicks anywhere on the mini open the full-graph modal; there is no node selection, no edge highlighting, no zoom. The full modal is interactive.
4. **Should the Catalog drawer move to the side rail?** Resolution: **no — only the button moves**. Per design doc 03 "Catalog drawer" — "Same, with reference framing." The drawer continues to slide over from the right edge. The change in 15b is purely *the trigger location*: the button now sits in the side rail beneath the export YAML button. The drawer's positioning, focus trap, and animation are unchanged. (Phase 7 reshapes the drawer's content — what plugins look like, search-first vs select-this — not its container.)
5. **Hash fragment redirect semantics.** Resolution: **silent rewrite via `replaceState` plus modal dispatch where applicable**. A user landing on `#/abc/spec` from a stale bookmark gets the URL rewritten to `#/abc` (no visible re-navigation) on mount; for `#/abc/graph` and `#/abc/yaml` the URL is *also* rewritten to `#/abc` but the corresponding modal *opens*, mirroring what the user would have seen on the old IA. This keeps the bookmark roughly meaningful (they came here for the graph; they get the graph) without polluting the URL with action verbs. Documented in Task 6.
6. **`SWITCH_TAB_EVENT` lifecycle.** Resolution: **deleted in Phase 3B Task 10, not deferred**. An IMPORTANT finding (15b2 §4) supersedes the earlier SUGGESTION to defer — `CommandPalette.tsx` must also update its dispatch to use `OPEN_GRAPH_MODAL_EVENT` / `OPEN_YAML_MODAL_EVENT` directly. Task 10 deletes the export and all dispatch/listener sites. No Phase 6 observation is filed.
7. **Where does `OPEN_CATALOG_EVENT` come from after `InspectorPanel.tsx` is deleted?** Resolution: **re-export from `CatalogButton.tsx`**. The constant currently lives in `InspectorPanel.tsx`. Task 5 moves the constant declaration to `CatalogButton.tsx` (which dispatches it) and updates every importer to point at the new location. The drawer subscriber in `App.tsx` follows the new import path.

---

## Task 0: Move `SideRail.tsx` from `components/common/` to `components/sidebar/`

> **Review finding (CRITICAL — coherence):** 15a placed `SideRail.tsx` under `src/components/common/`. Phase 3B introduces enough side-rail-specific components that they warrant their own directory; every 15b1/15b2 file path assumes `src/components/sidebar/`. This task makes the directory convention consistent in one commit, before any new file is added.

**Files:**
- Move: `src/elspeth/web/frontend/src/components/common/SideRail.tsx` → `src/elspeth/web/frontend/src/components/sidebar/SideRail.tsx`.
- Move: `src/elspeth/web/frontend/src/components/common/SideRail.test.tsx` → `src/elspeth/web/frontend/src/components/sidebar/SideRail.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — update the `SideRail` import path to the new location.

No re-export shim is added at the old path (per CLAUDE.md "No Legacy Code Policy" — `SideRail` has a single caller, `App.tsx`, modified in the same commit).

- [ ] **Step 1: Create the directory and move the files**

```bash
cd src/elspeth/web/frontend
mkdir -p src/components/sidebar
git mv src/components/common/SideRail.tsx src/components/sidebar/SideRail.tsx
git mv src/components/common/SideRail.test.tsx src/components/sidebar/SideRail.test.tsx
```

- [ ] **Step 2: Update the import in `App.tsx`**

```bash
grep -RIn "components/common/SideRail" src
```

For every match, change `from "./components/common/SideRail"` → `from "./components/sidebar/SideRail"` (or `from "@/components/sidebar/SideRail"` when the file uses the path alias). 15a's commit only added one importer (`App.tsx`); confirm by grep that nothing else points at the old path.

- [ ] **Step 3: Run all tests + build**

```bash
cd src/elspeth/web/frontend && npx vitest run src && npm run build
```

Expected: PASS. The relocated `SideRail.test.tsx` continues to exercise the 15a scaffold (slot-presence assertions) — only its path changed.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(web/frontend): move SideRail into its own dir (Phase 3B.0)

15a placed SideRail.tsx under components/common/ alongside Layout and
CommandPalette.  Phase 3B adds GraphMiniView, ExportYamlButton,
CatalogButton, ExecuteButton, and GraphModal / ExportYamlModal — the
volume of side-rail-specific UI warrants its own directory, matching
the existing UI-area convention (inspector/, chat/, sessions/, etc.).
Single commit, no compat shim per the No Legacy Code Policy."
```

---

## Task 1: Extract `VersionSelector` and mount it in the header (additive)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.tsx` — the extracted component, renamed for placement clarity.
- Create: `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/common/AppHeader.tsx` (added in 15a Task 3) — mount `<HeaderVersionSelector />` to the right of `<HeaderSessionSwitcher />` and to the left of `<UserMenu />`.
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — KEEP the inner `VersionSelector` component for now; it is deleted in Task 9 when the whole panel goes. Both surfaces render the same data; both call `revertToVersion`. This is the explicit dual-render transitional state.

This task is **additive**. After Task 1, the version selector appears in two places. Task 9 removes the inspector copy.

- [ ] **Step 1: Read the current `VersionSelector`**

Open `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` and read lines 41–313 (the `VersionSelector` component plus `VersionSelectorProps`). The component owns: a dropdown listing versions, a current-version label, a loading state from `isLoadingVersions`, an `onOpen` callback that triggers `loadStateVersions`, and an `onRevert` callback that takes `(stateId, version)`. Confirm those props before extraction.

- [ ] **Step 2: Write the failing test for `HeaderVersionSelector`**

Create `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { HeaderVersionSelector } from "./HeaderVersionSelector";
import { useSessionStore } from "@/stores/sessionStore";

describe("HeaderVersionSelector", () => {
  beforeEach(() => {
    useSessionStore.setState({
      activeSessionId: "sess-1",
      compositionState: { version: 3, source: null, nodes: [], outputs: [] } as never,
      stateVersions: [],
      isLoadingVersions: false,
      loadStateVersions: vi.fn(),
      revertToVersion: vi.fn(),
    } as never);
  });

  it("renders nothing when no active session", () => {
    useSessionStore.setState({ activeSessionId: null } as never);
    const { container } = render(<HeaderVersionSelector />);
    expect(container.firstChild).toBeNull();
  });

  it("shows the current composition version label", () => {
    render(<HeaderVersionSelector />);
    expect(screen.getByText(/v3|version 3/i)).toBeInTheDocument();
  });

  it("uses the design-spec label 'Composition history' on the dropdown trigger", () => {
    render(<HeaderVersionSelector />);
    const trigger = screen.getByRole("button", { name: /composition history/i });
    expect(trigger).toBeInTheDocument();
  });

  it("calls loadStateVersions when the dropdown opens", () => {
    const loadStateVersions = vi.fn();
    useSessionStore.setState({ loadStateVersions } as never);
    render(<HeaderVersionSelector />);
    fireEvent.click(screen.getByRole("button", { name: /composition history/i }));
    expect(loadStateVersions).toHaveBeenCalled();
  });

  it("calls revertToVersion when the user picks an older version", () => {
    const revertToVersion = vi.fn();
    useSessionStore.setState({
      stateVersions: [
        { id: "st-1", version: 1, created_at: "2026-05-15T10:00:00Z", node_count: 1 } as never,
        { id: "st-2", version: 2, created_at: "2026-05-15T10:10:00Z", node_count: 2 } as never,
        { id: "st-3", version: 3, created_at: "2026-05-15T10:20:00Z", node_count: 3 } as never,
      ],
      revertToVersion,
    } as never);
    render(<HeaderVersionSelector />);
    fireEvent.click(screen.getByRole("button", { name: /composition history/i }));
    // Surface a revert affordance for v2 (the design-spec exact UI is decided
    // in the component; the contract here is "user can choose version 2").
    fireEvent.click(screen.getByRole("button", { name: /revert to version 2/i }));
    expect(revertToVersion).toHaveBeenCalledWith("st-2");
  });
});
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/header/HeaderVersionSelector.test.tsx
```

Expected: FAIL — module not found.

- [ ] **Step 4: Create `HeaderVersionSelector.tsx` by extracting + re-pointing the existing VersionSelector**

Create `src/elspeth/web/frontend/src/components/header/HeaderVersionSelector.tsx`. Copy the `VersionSelector` body from `InspectorPanel.tsx:53–313` (the `useState(false)` for `isOpen`, the dropdown markup, the revert-confirmation dialog) into the new file. Three substantive edits during the copy:

1. The component becomes self-sourcing instead of prop-driven. Pull state from `useSessionStore` directly:

```typescript
const activeSessionId = useSessionStore((s) => s.activeSessionId);
const compositionState = useSessionStore((s) => s.compositionState);
const stateVersions = useSessionStore((s) => s.stateVersions);
const isLoadingVersions = useSessionStore((s) => s.isLoadingVersions);
const loadStateVersions = useSessionStore((s) => s.loadStateVersions);
const revertToVersion = useSessionStore((s) => s.revertToVersion);

if (!activeSessionId || !compositionState) return null;
```

2. The dropdown trigger's accessible name becomes "Composition history" (the design-spec rename). The visual label stays compact (e.g., `v{version} ▾`) but `aria-label="Composition history (currently v{version})"`.

3. The revert-target state and `ConfirmDialog` are kept intact — the confirmation UX is the same. The store signature is `revertToVersion(stateId: string)`, so the confirmed action passes `revertTarget.id` only; the version number is display copy, not a second argument.

The inner `VersionSelector` in `InspectorPanel.tsx` is **left in place** for now (Task 9 deletes the whole file). Both renders subscribe to the same store and mutate the same state; the dual-render is harmless.

- [ ] **Step 5: Mount in `AppHeader`**

Open `src/elspeth/web/frontend/src/components/common/AppHeader.tsx` (the file 15a Task 3 introduced). Inside the header layout, add `<HeaderVersionSelector />` directly to the right of the `<HeaderSessionSwitcher />` slot, separated by a small visual divider (CSS class `app-header-separator`). The order, left-to-right: brand → HeaderSessionSwitcher → divider → HeaderVersionSelector → flex-spacer → UserMenu.

- [ ] **Step 6: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS, including the new header-version-selector test and an `AppHeader.test.tsx` assertion that both `HeaderSessionSwitcher` and `HeaderVersionSelector` mount.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): extract VersionSelector to header (Phase 3B.1)

The dropdown is now reachable as 'Composition history' in the header.
The inspector's inner copy is left in place until Task 9 deletes the
entire inspector panel. Both copies subscribe to the same store; the
dual render is intentional and short-lived."
```

---

## Task 2: Relocate Execute button to `SideRail.executeButtonSlot` (amendment to 15a Task 2 scaffold)

> **Review finding (CRITICAL):** The plan previously stated "15a is amended below in Task 2 to relocate Execute" but the amendment was absent. This rewrite is that amendment.

**What this task does:** Moves the Execute button OUT of `InspectorPanel.tsx` and INTO the `SideRail.executeButtonSlot` prop BEFORE Phase 3B's Task 9 deletes the inspector. The `executeButtonSlot` prop was added to the `SideRail` scaffold in **15a Task 2** (see [15a1 review history](15a1-phase-3a-removals-part-1.md#review-history)). This task is the first time any content fills that slot.

**Slot composition contract** (applies to all slots in SideRail): slots are render-props (`ReactNode | null`). `App.tsx` passes `<ExecuteButton />` as the `executeButtonSlot` prop value; `SideRail` renders `{executeButtonSlot}` inside a wrapper div with `data-testid="siderail-slot-execute-button"`. `SideRail` does NOT import `ExecuteButton` directly — the wiring is always at the call site.

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExecuteButton.tsx` — the extracted execute button component.
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExecuteButton.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — pass `executeButtonSlot={<ExecuteButton />}` to `<SideRail />`.
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — remove the Execute button from the inspector header (its parent flex row in the inspector top toolbar).
- Modify: `src/elspeth/web/frontend/src/components/sidebar/SideRail.test.tsx` — assert only that `executeButtonSlot` content is placed under `siderail-slot-execute-button`. Do not test Execute-button store behaviour through bare `<SideRail />`; `SideRail` must not import or mount `ExecuteButton`.

This task carries forward the 15a §"Open scope questions resolved" 2: Execute lives at the side rail until Phase 6 wraps it in the completion bar.

- [ ] **Step 1: Failing slot-placement test in `SideRail.test.tsx`**

Add this case to `src/elspeth/web/frontend/src/components/sidebar/SideRail.test.tsx`:

```typescript
it("places executeButtonSlot content in the execute-button slot", () => {
  render(<SideRail executeButtonSlot={<button type="button">Run pipeline</button>} />);
  const slot = screen.getByTestId("siderail-slot-execute-button");
  expect(
    within(slot).getByRole("button", { name: /run pipeline/i }),
  ).toBeInTheDocument();
});
```

Import `within` from `@testing-library/react` if the file does not already import it. This is the only `SideRail.test.tsx` coverage for the execute button in this task: prop placement, not behaviour.

- [ ] **Step 1a: Failing behaviour tests in `ExecuteButton.test.tsx`**

Add these cases to `src/elspeth/web/frontend/src/components/sidebar/ExecuteButton.test.tsx`:

```typescript
it("renders a Run pipeline button when validation has passed", () => {
  useExecutionStore.setState({
    validationResult: { is_valid: true, checks: [], errors: [], warnings: [] } as never,
    isExecuting: false,
    progress: null,
  } as never);
  useSessionStore.setState({ activeSessionId: "sess-1" } as never);
  render(<ExecuteButton />);
  expect(
    screen.getByRole("button", { name: /run pipeline/i }),
  ).toBeInTheDocument();
});

it("disables the Run pipeline button when validation is failing", () => {
  useExecutionStore.setState({
    validationResult: {
      is_valid: false,
      checks: [],
      errors: [{ component_type: "source", component_id: "csv_source", message: "x" } as never],
      warnings: [],
    } as never,
    isExecuting: false,
    progress: null,
  } as never);
  useSessionStore.setState({ activeSessionId: "sess-1" } as never);
  render(<ExecuteButton />);
  expect(screen.getByRole("button", { name: /run pipeline/i })).toBeDisabled();
});

it("invokes execute with the active session id when clicked", () => {
  const execute = vi.fn();
  useExecutionStore.setState({
    validationResult: { is_valid: true, checks: [], errors: [], warnings: [] } as never,
    isExecuting: false,
    progress: null,
    execute,
  } as never);
  useSessionStore.setState({ activeSessionId: "sess-1" } as never);
  render(<ExecuteButton />);
  fireEvent.click(screen.getByRole("button", { name: /run pipeline/i }));
  expect(execute).toHaveBeenCalledWith("sess-1");
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/sidebar/SideRail.test.tsx src/components/sidebar/ExecuteButton.test.tsx
```

Expected: FAIL — the slot assertion fails until `SideRail` exposes the wrapper and the behaviour tests fail until `ExecuteButton` exists.

- [ ] **Step 3: Create `ExecuteButton.tsx` and wire via App.tsx**

Create `src/elspeth/web/frontend/src/components/sidebar/ExecuteButton.tsx`. Per the slot composition contract, `SideRail` does NOT import or mount this component internally — `App.tsx` passes it as the `executeButtonSlot` prop.

In `App.tsx`:

```tsx
import { ExecuteButton } from "./components/sidebar/ExecuteButton";

// ...inside the Layout render:
<SideRail
  executeButtonSlot={<ExecuteButton />}
  {/* other slots remain null until their respective tasks fill them */}
>
  <InspectorPanel />
</SideRail>
```

`ExecuteButton`'s body:

```tsx
function ExecuteButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  const validationResult = useExecutionStore((s) => s.validationResult);
  const isExecuting = useExecutionStore((s) => s.isExecuting);
  const progress = useExecutionStore((s) => s.progress);
  const execute = useExecutionStore((s) => s.execute);

  if (!activeSessionId) return null;

  const canExecute =
    validationResult?.is_valid === true &&
    !isExecuting &&
    progress?.status !== "running";

  return (
    <button
      type="button"
      className="side-rail-execute-btn"
      onClick={() => execute(activeSessionId)}
      disabled={!canExecute}
      aria-label="Run pipeline"
    >
      ▶ Run pipeline
    </button>
  );
}
```

- [ ] **Step 4: Remove Execute from `InspectorPanel.tsx`**

In `InspectorPanel.tsx`, find the Execute button in the top toolbar (it sits next to where Validate *was* in 15a Task 8 — the right-hand action area of row 1). Delete the button markup and the `handleExecute` callback if it's no longer used elsewhere in the file. The keyboard shortcut `Ctrl+E` in `App.tsx:154–166` is unchanged — it already operates on the store directly without depending on the button DOM.

- [ ] **Step 5: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS. The `InspectorPanel.test.tsx` may have an assertion asserting the Execute button is present in the inspector — update it to assert *absence* (a single-line change: `expect(...).not.toBeInTheDocument()`).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): move Execute button to SideRail (Phase 3B.2)

The button now lives in the side rail's execute-button slot.  Phase 6
will wrap it in the persona-aware completion bar (Save for review /
Run pipeline →).  Ctrl+E is unchanged."
```

---

## Task 3: Add `GraphModal` and `GraphMiniView`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sidebar/GraphMiniView.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sidebar/GraphMiniView.test.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sidebar/GraphModal.tsx` — placed under `sidebar/` from the start (its trigger and lifecycle belong to the side-rail surface). `GraphView.tsx` itself remains under `inspector/` because Task 9 keeps it (only `InspectorPanel.tsx` and its `.test.tsx` are deleted).
- Create: `src/elspeth/web/frontend/src/components/sidebar/GraphModal.test.tsx`.
- Create or verify: `src/elspeth/web/frontend/src/lib/composer-events.ts` — shared modal-open event constants. This module is the canonical event source for Graph/YAML modal open requests from Task 3 onward.
- Modify: `src/elspeth/web/frontend/src/components/sidebar/SideRail.tsx` — populate the `graph-mini` slot.
- `GraphMiniView.tsx`, `GraphModal.tsx`, `useHashRouter.ts`, `CommandPalette.tsx`, `CompletionSummary.tsx`, and `ReadinessRowDetail.tsx` import modal-open constants from `@/lib/composer-events`; component files do not define their own event-name constants.

- [ ] **Step 1: Failing test for `GraphMiniView`**

Before creating the test, create or verify the constants-only module used by all modal openers:

```typescript
// src/elspeth/web/frontend/src/lib/composer-events.ts
export const OPEN_GRAPH_MODAL_EVENT = "elspeth-open-graph-modal";
export const OPEN_YAML_MODAL_EVENT = "elspeth-open-yaml-modal";
```

Create `src/elspeth/web/frontend/src/components/sidebar/GraphMiniView.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GraphMiniView } from "./GraphMiniView";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useSessionStore } from "@/stores/sessionStore";

// GraphView is heavy (React Flow + dagre + canvas); the mini view does its
// own rendering, but the modal hosts the real component — stubbed here.
vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => <div data-testid="graph-view-stub" />,
}));

describe("GraphMiniView", () => {
  beforeEach(() => {
    useSessionStore.setState({
      compositionState: {
        version: 1,
        source: { type: "csv", id: "src-1", options: {} } as never,
        nodes: [{ id: "tx-1", type: "row_transform", options: {} } as never],
        outputs: [{ id: "out-1", type: "stdout", options: {} } as never],
      } as never,
      selectedNodeId: null,
    } as never);
  });

  it("renders an aria-labelled mini graph", () => {
    render(<GraphMiniView />);
    expect(screen.getByRole("button", { name: /pipeline graph/i })).toBeInTheDocument();
  });

  it("renders an empty state when no composition exists", () => {
    useSessionStore.setState({ compositionState: null } as never);
    render(<GraphMiniView />);
    expect(screen.getByText(/no pipeline yet/i)).toBeInTheDocument();
  });

  it("dispatches OPEN_GRAPH_MODAL_EVENT when clicked", () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
    render(<GraphMiniView />);
    fireEvent.click(screen.getByRole("button", { name: /pipeline graph/i }));
    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handler);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/sidebar/GraphMiniView.test.tsx
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement `GraphMiniView.tsx`**

```typescript
// ============================================================================
// GraphMiniView
//
// Compact, non-interactive snapshot of the current composition's DAG, mounted
// in the side rail.  Clicking anywhere on the mini opens the full GraphModal.
//
// The mini view renders a simplified inline-SVG flow: nodes as rounded
// rectangles laid out left-to-right (source → transforms → outputs), edges
// as straight lines.  No React Flow / dagre — those are heavy and reserved
// for the full modal.  The mini reads compositionState.{source,nodes,outputs}
// and lays them out by type-bucket.
//
// Click semantics: H2 resolved as (a) — the entire mini is a single
// activatable region (role=button, keyboard-Enter).  Sub-element clicks do
// not select nodes; the mini is verification-only.
// ============================================================================

import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

export function GraphMiniView(): JSX.Element {
  const compositionState = useSessionStore((s) => s.compositionState);

  if (
    !compositionState ||
    (!compositionState.source &&
      compositionState.nodes.length === 0 &&
      compositionState.outputs.length === 0)
  ) {
    return (
      <div className="graph-mini graph-mini--empty" data-testid="graph-mini-empty">
        <span>No pipeline yet</span>
      </div>
    );
  }

  const openModal = () => {
    window.dispatchEvent(new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
  };

  return (
    <button
      type="button"
      className="graph-mini"
      onClick={openModal}
      aria-label="Pipeline graph (click to expand)"
    >
      <MiniSvg state={compositionState} />
    </button>
  );
}

function MiniSvg({ state }: { state: NonNullable<ReturnType<typeof useSessionStore.getState>["compositionState"]> }) {
  // Simple left-to-right lane layout. Source (if any) → nodes (in order) →
  // outputs.  Node ids are NOT rendered (they're noisy at this size); each
  // bucket renders a colour-coded rectangle with the count.
  const lanes: { label: string; count: number; colorVar: string }[] = [];
  if (state.source) {
    lanes.push({ label: "src", count: 1, colorVar: "--color-accent" });
  }
  if (state.nodes.length > 0) {
    lanes.push({ label: `${state.nodes.length} tx`, count: state.nodes.length, colorVar: "--color-info" });
  }
  if (state.outputs.length > 0) {
    lanes.push({
      label: state.outputs.length === 1 ? "sink" : `${state.outputs.length} sinks`,
      count: state.outputs.length,
      colorVar: "--color-success",
    });
  }

  const width = 240;
  const height = 80;
  const laneWidth = width / Math.max(lanes.length, 1);

  return (
    <svg width={width} height={height} role="img" aria-hidden="true">
      {lanes.map((lane, i) => {
        const x = i * laneWidth + 8;
        return (
          <g key={lane.label}>
            <rect
              x={x}
              y={height / 2 - 16}
              width={laneWidth - 16}
              height={32}
              rx={6}
              fill={`var(${lane.colorVar})`}
              opacity={0.85}
            />
            <text
              x={x + (laneWidth - 16) / 2}
              y={height / 2 + 4}
              textAnchor="middle"
              fontSize={12}
              fill="white"
            >
              {lane.label}
            </text>
            {i < lanes.length - 1 && (
              <line
                x1={x + laneWidth - 16}
                y1={height / 2}
                x2={(i + 1) * laneWidth + 8}
                y2={height / 2}
                stroke="var(--color-text-muted)"
                strokeWidth={2}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
```

- [ ] **Step 4: Failing test for `GraphModal`**

Create `src/elspeth/web/frontend/src/components/sidebar/GraphModal.test.tsx`:

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { GraphModal } from "./GraphModal";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";

vi.mock("@/components/inspector/GraphView", () => ({
  GraphView: () => <div data-testid="graph-view-stub" />,
}));

describe("GraphModal", () => {
  it("renders nothing when not opened", () => {
    const { container } = render(<GraphModal />);
    expect(container.querySelector("[role='dialog']")).toBeNull();
  });

  it("opens on OPEN_GRAPH_MODAL_EVENT", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    expect(screen.getByRole("dialog", { name: /pipeline graph/i })).toBeInTheDocument();
    expect(screen.getByTestId("graph-view-stub")).toBeInTheDocument();
  });

  it("closes on Escape", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the backdrop is clicked", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    fireEvent.click(screen.getByTestId("graph-modal-backdrop"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes when the close button is clicked", () => {
    render(<GraphModal />);
    fireEvent(window, new CustomEvent(OPEN_GRAPH_MODAL_EVENT));
    fireEvent.click(screen.getByRole("button", { name: /close graph/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
```

- [ ] **Step 5: Implement `GraphModal.tsx`**

```typescript
// ============================================================================
// GraphModal
//
// Modal wrapper around GraphView.  Mirrors SecretsPanel: focus trap, escape
// key, backdrop click, role=dialog, aria-labelledby.  Opens on
// OPEN_GRAPH_MODAL_EVENT; closes via Escape / backdrop / close button.
// ============================================================================

import { useEffect, useRef, useState, useId } from "react";
import { GraphView } from "@/components/inspector/GraphView";
import { OPEN_GRAPH_MODAL_EVENT } from "@/lib/composer-events";
import { useFocusTrap } from "@/hooks/useFocusTrap";

export function GraphModal(): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useFocusTrap(dialogRef, isOpen, ".graph-modal-close");

  useEffect(() => {
    function handleOpen() {
      setIsOpen(true);
    }
    window.addEventListener(OPEN_GRAPH_MODAL_EVENT, handleOpen);
    return () => window.removeEventListener(OPEN_GRAPH_MODAL_EVENT, handleOpen);
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        setIsOpen(false);
      }
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      <div
        className="graph-modal-backdrop"
        data-testid="graph-modal-backdrop"
        onClick={() => setIsOpen(false)}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="graph-modal"
      >
        <header className="graph-modal-header">
          <h2 id={titleId}>Pipeline graph</h2>
          <button
            type="button"
            className="graph-modal-close"
            onClick={() => setIsOpen(false)}
            aria-label="Close graph"
          >
            ✕
          </button>
        </header>
        <div className="graph-modal-body">
          <GraphView />
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 6: Wire via `App.tsx` and mount modal at app root**

Per the slot composition contract, `SideRail` does not import `GraphMiniView` directly. In `App.tsx`:

```tsx
import { GraphMiniView } from "./components/sidebar/GraphMiniView";
import { GraphModal } from "./components/sidebar/GraphModal";

// In the Layout render:
<SideRail
  graphMiniSlot={<GraphMiniView />}
  executeButtonSlot={<ExecuteButton />}
  {/* other slots remain null */}
>
  <InspectorPanel />
</SideRail>

// At the app root alongside CommandPalette / SecretsPanel:
<GraphModal />
```

- [ ] **Step 7: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS, including the new graph-mini and graph-modal tests. The existing inspector Graph tab continues to work — `GraphView` is still mounted under the inspector until Task 9.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): add GraphMiniView + GraphModal (Phase 3B.3)

The side rail now carries a compact left-to-right graph mini.  Clicking
the mini fires OPEN_GRAPH_MODAL_EVENT, which a modal mounted at the app
root listens for and renders the existing GraphView inside.  The
inspector Graph tab still works until Task 9 deletes the inspector
panel."
```

---

## Task 4: Add `ExportYamlModal` and `ExportYamlButton`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExportYamlButton.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExportYamlButton.test.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.tsx` — placed under `sidebar/` from the start; `YamlView.tsx` itself stays under `inspector/` (Task 9 keeps it).
- Create: `src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/sidebar/SideRail.tsx` — populate the `export-yaml` slot.

- [ ] **Step 1: Failing test for `ExportYamlButton`**

```typescript
// src/elspeth/web/frontend/src/components/sidebar/ExportYamlButton.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExportYamlButton } from "./ExportYamlButton";
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

describe("ExportYamlButton", () => {
  it("renders nothing when no active session", () => {
    useSessionStore.setState({ activeSessionId: null } as never);
    const { container } = render(<ExportYamlButton />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the export button when a session is active", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    render(<ExportYamlButton />);
    expect(screen.getByRole("button", { name: /export yaml/i })).toBeInTheDocument();
  });

  it("dispatches OPEN_YAML_MODAL_EVENT on click", () => {
    useSessionStore.setState({ activeSessionId: "sess-1" } as never);
    const handler = vi.fn();
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handler);
    render(<ExportYamlButton />);
    fireEvent.click(screen.getByRole("button", { name: /export yaml/i }));
    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_YAML_MODAL_EVENT, handler);
  });
});
```

- [ ] **Step 2: Implement `ExportYamlButton.tsx`**

```typescript
import { useSessionStore } from "@/stores/sessionStore";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

export function ExportYamlButton(): JSX.Element | null {
  const activeSessionId = useSessionStore((s) => s.activeSessionId);
  if (!activeSessionId) return null;
  return (
    <button
      type="button"
      className="side-rail-export-yaml-btn"
      onClick={() => window.dispatchEvent(new CustomEvent(OPEN_YAML_MODAL_EVENT))}
      aria-label="Export YAML"
    >
      ⬇ Export YAML
    </button>
  );
}
```

- [ ] **Step 3: Failing test for `ExportYamlModal`**

```typescript
// src/elspeth/web/frontend/src/components/sidebar/ExportYamlModal.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ExportYamlModal } from "./ExportYamlModal";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";

vi.mock("@/components/inspector/YamlView", () => ({
  YamlView: () => <div data-testid="yaml-view-stub" />,
}));

describe("ExportYamlModal", () => {
  it("renders nothing until opened", () => {
    const { container } = render(<ExportYamlModal />);
    expect(container.querySelector("[role='dialog']")).toBeNull();
  });

  it("opens on OPEN_YAML_MODAL_EVENT", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));
    expect(screen.getByRole("dialog", { name: /export yaml/i })).toBeInTheDocument();
    expect(screen.getByTestId("yaml-view-stub")).toBeInTheDocument();
  });

  it("closes on Escape", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("closes on backdrop click", () => {
    render(<ExportYamlModal />);
    fireEvent(window, new CustomEvent(OPEN_YAML_MODAL_EVENT));
    fireEvent.click(screen.getByTestId("yaml-modal-backdrop"));
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
```

- [ ] **Step 4: Implement `ExportYamlModal.tsx`**

```typescript
import { useEffect, useRef, useState, useId } from "react";
import { YamlView } from "@/components/inspector/YamlView";
import { OPEN_YAML_MODAL_EVENT } from "@/lib/composer-events";
import { useFocusTrap } from "@/hooks/useFocusTrap";

export function ExportYamlModal(): JSX.Element | null {
  const [isOpen, setIsOpen] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useFocusTrap(dialogRef, isOpen, ".yaml-modal-close");

  useEffect(() => {
    function handleOpen() {
      setIsOpen(true);
    }
    window.addEventListener(OPEN_YAML_MODAL_EVENT, handleOpen);
    return () => window.removeEventListener(OPEN_YAML_MODAL_EVENT, handleOpen);
  }, []);

  useEffect(() => {
    if (!isOpen) return;
    function handleKey(e: KeyboardEvent) {
      if (e.key === "Escape") setIsOpen(false);
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <>
      <div
        className="yaml-modal-backdrop"
        data-testid="yaml-modal-backdrop"
        onClick={() => setIsOpen(false)}
        aria-hidden="true"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className="yaml-modal"
      >
        <header className="yaml-modal-header">
          <h2 id={titleId}>Export YAML</h2>
          <button
            type="button"
            className="yaml-modal-close"
            onClick={() => setIsOpen(false)}
            aria-label="Close export YAML"
          >
            ✕
          </button>
        </header>
        <div className="yaml-modal-body">
          <YamlView />
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 5: Wire via `App.tsx` (slot composition contract)**

Per the slot composition contract, `SideRail` does not import `ExportYamlButton` directly. In `App.tsx`:

```tsx
import { ExportYamlButton } from "./components/sidebar/ExportYamlButton";
import { ExportYamlModal } from "./components/sidebar/ExportYamlModal";

// In the Layout render:
<SideRail
  graphMiniSlot={<GraphMiniView />}
  exportYamlSlot={<ExportYamlButton />}
  executeButtonSlot={<ExecuteButton />}
  {/* other slots remain null */}
>
  <InspectorPanel />
</SideRail>

// At the app root:
<GraphModal />
<ExportYamlModal />
```

- [ ] **Step 6: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS. Inspector YAML tab still works until Task 9.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): add ExportYamlButton + ExportYamlModal (Phase 3B.4)

Side rail now carries an Export YAML button.  Clicking dispatches
OPEN_YAML_MODAL_EVENT; a modal mounted at the app root hosts the
existing YamlView (which already provides copy-to-clipboard +
download-as-file).  The inspector YAML tab continues to work until the
inspector panel is deleted."
```

---

## Task 5: Move Catalog button to `SideRail`

**Files:**
- Create: `src/elspeth/web/frontend/src/components/sidebar/CatalogButton.tsx`.
- Create: `src/elspeth/web/frontend/src/components/sidebar/CatalogButton.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/sidebar/SideRail.tsx` — populate the `catalog-button` slot.
- Modify: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` — remove the inspector header's Catalog button. The `OPEN_CATALOG_EVENT` constant currently exported from `InspectorPanel.tsx` line 39 is **moved** to `CatalogButton.tsx`. Importers throughout the codebase (`App.tsx:11,102`) are repointed.
- Modify: `src/elspeth/web/frontend/src/App.tsx` — import `OPEN_CATALOG_EVENT` from `@/components/sidebar/CatalogButton`.
- The `CatalogDrawer` itself stays mounted where it currently is — at the side-rail / inspector boundary — until Task 9 moves it to the app root as part of the inspector teardown.

- [ ] **Step 1: Failing test for `CatalogButton`**

```typescript
// src/elspeth/web/frontend/src/components/sidebar/CatalogButton.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CatalogButton, OPEN_CATALOG_EVENT } from "./CatalogButton";

describe("CatalogButton", () => {
  it("renders a catalog-reference button", () => {
    render(<CatalogButton />);
    expect(
      screen.getByRole("button", { name: /catalog \(reference\)/i }),
    ).toBeInTheDocument();
  });

  it("dispatches OPEN_CATALOG_EVENT on click", () => {
    const handler = vi.fn();
    window.addEventListener(OPEN_CATALOG_EVENT, handler);
    render(<CatalogButton />);
    fireEvent.click(
      screen.getByRole("button", { name: /catalog \(reference\)/i }),
    );
    expect(handler).toHaveBeenCalled();
    window.removeEventListener(OPEN_CATALOG_EVENT, handler);
  });
});
```

- [ ] **Step 2: Implement `CatalogButton.tsx`**

```typescript
// CatalogButton — side-rail trigger for the catalog reference drawer.  The
// drawer's interior is reshaped by Phase 7; this task only relocates the
// trigger.
export const OPEN_CATALOG_EVENT = "open-catalog";

export function CatalogButton(): JSX.Element {
  return (
    <button
      type="button"
      className="side-rail-catalog-btn"
      onClick={() => window.dispatchEvent(new CustomEvent(OPEN_CATALOG_EVENT))}
      aria-label="Open plugin catalog (reference)"
    >
      📋 Catalog (reference)
    </button>
  );
}
```

- [ ] **Step 3: Update importers**

Run a search for `OPEN_CATALOG_EVENT` across the frontend:

```bash
cd src/elspeth/web/frontend && grep -RIn "OPEN_CATALOG_EVENT" src
```

For every match (currently: `App.tsx:11,102`, `InspectorPanel.tsx:39,...`):
- Change `import { OPEN_CATALOG_EVENT } from "./components/inspector/InspectorPanel";` → `from "./components/sidebar/CatalogButton"`.
- Remove the export from `InspectorPanel.tsx` line 39 (delete the export and the corresponding subscription that opens the inspector's local `<CatalogDrawer>` instance; the drawer is now driven by `SideRail`'s mount — see step 4).

- [ ] **Step 4: Wire `CatalogButton` via `App.tsx` and move `<CatalogDrawer>` to app root**

Per the slot composition contract, `SideRail` does not import `CatalogButton` directly. In `App.tsx`:

```tsx
import { CatalogButton } from "./components/sidebar/CatalogButton";

// In the Layout render (cumulative — all slots filled so far):
<SideRail
  graphMiniSlot={<GraphMiniView />}
  catalogSlot={<CatalogButton />}
  exportYamlSlot={<ExportYamlButton />}
  executeButtonSlot={<ExecuteButton />}
  {/* other slots remain null */}
>
  <InspectorPanel />
</SideRail>
```

Also move `<CatalogDrawer>` from `InspectorPanel.tsx` to `App.tsx` at the app root (next to `GraphModal`, `ExportYamlModal`, etc.). The drawer's `isOpen` state and event listener live in `App.tsx`:

```tsx
const [catalogOpen, setCatalogOpen] = useState(false);
useEffect(() => {
  function handleOpen() { setCatalogOpen(true); }
  window.addEventListener(OPEN_CATALOG_EVENT, handleOpen);
  return () => window.removeEventListener(OPEN_CATALOG_EVENT, handleOpen);
}, []);

// In the render:
<CatalogDrawer isOpen={catalogOpen} onClose={() => setCatalogOpen(false)} />
```

Delete the `<CatalogDrawer>` mount and its `isOpen` state from `InspectorPanel.tsx`.

- [ ] **Step 5: Run all tests + smoke render**

```bash
cd src/elspeth/web/frontend && npx vitest run src
```

Expected: PASS. `Ctrl+Shift+P` still opens the catalog because `App.tsx:96–104` dispatches the same `OPEN_CATALOG_EVENT` — the listener is now in `App.tsx` (moved from `InspectorPanel`), but the event name and dispatch path are unchanged.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(web/frontend): move Catalog button to SideRail (Phase 3B.5)

The catalog trigger now lives in the side rail beneath Export YAML.  The
button is renamed 'Catalog (reference)' per design doc 03 — Phase 7
reshapes the drawer's content; 15b is the chrome relocation only.
OPEN_CATALOG_EVENT is now exported from CatalogButton.tsx; all
importers updated."
```

---

**Tasks 6–10 (hash-router rewrite, CommandPalette updates, keyboard shortcuts, InspectorPanel teardown, cleanup pass) continue in [15b2-phase-3b-side-rail-part-2.md](15b2-phase-3b-side-rail-part-2.md).**

---

## Review history

### 2026-05-15 — Review panel applied

**CRITICAL (Systems):** Task 2 rewritten to be the explicit amendment that was previously missing. The task now clearly states: Execute button moves OUT of `InspectorPanel.tsx` and INTO `SideRail.executeButtonSlot` BEFORE Task 9 deletes the inspector. `ExecuteButton` is extracted to its own file; `App.tsx` passes it as `executeButtonSlot={<ExecuteButton />}`.

**IMPORTANT (Architecture):** `executeButtonSlot` prop was added to the 15a Task 2 scaffold (see [15a1 review history](15a1-phase-3a-removals-part-1.md#review-history)). Tasks 3–5 updated to wire content through `App.tsx` props (`graphMiniSlot`, `exportYamlSlot`, `catalogSlot`) rather than directly inside `SideRail.tsx`. The slot composition contract is now explicit throughout: SideRail does not import or mount slot content; callers pass it as props.

**Cross-file decision (SWITCH_TAB_EVENT):** IMPORTANT (15b2 §4) supersedes SUGGESTION (original 15b1 §3): `SWITCH_TAB_EVENT` is deleted in Task 10, not deferred to Phase 6. §"Open scope questions resolved" item 6 and the scope-boundaries text updated accordingly. No filigree observation filed (the work is done in-phase).

### 2026-05-16 — Cross-phase coherence review applied

**CRITICAL (Coherence #3, #6 — directory paths):** 15a placed `SideRail.tsx` under `src/components/common/`, but every 15b1/15b2 file path assumed `src/components/sidebar/` (new directory). Resolution: **Task 0 added** — moves `SideRail.tsx` + `SideRail.test.tsx` from `common/` to `sidebar/` in one commit (No Legacy Code Policy — no compat shim), updates the single `App.tsx` import, then all subsequent tasks use the consistent `sidebar/` path. Sequencing block and "Tasks in this file" wording updated. Directory convention adjudication added to the plan header.

**IMPORTANT (Coherence #3 — new files under inspector/):** Original Task 3 placed `GraphModal.tsx` under `src/components/inspector/` "for now to minimise reflows; Task 10 moves it under `graph/`" — but Task 10 had no such move step, and the comment created a fresh stale-`inspector/` reference for a brand-new file. Same problem in Task 4 for `ExportYamlModal.tsx`. Resolution: both modal components are placed under `src/components/sidebar/` from the start. The modal mocks of `GraphView` / `YamlView` use absolute paths (`@/components/inspector/GraphView`) since those view files remain under `inspector/` (Task 9 only deletes `InspectorPanel.tsx` + `.test.tsx`, not the entire `inspector/` directory).

### 2026-05-17 — Pre-dispatch NO-GO follow-up

**BLOCKER (Execution target) — Phase 3 worktree/branch made concrete.** Shared header now names `/home/john/elspeth/.worktrees/composer-phase-3-ia-cleanup` on `feat/composer-phase-3-ia-cleanup` from `RC5.2`; the old Phase 2A worktree/branch are explicitly forbidden.

**BLOCKER (Slot contract) — ExecuteButton behaviour tests moved out of SideRail.** Task 2 now keeps `SideRail.test.tsx` to slot-placement assertions only. Run-button enable/disable/execute behaviour lives in `ExecuteButton.test.tsx`, preserving the contract that `SideRail` never imports or mounts slot content directly.

**IMPORTANT (Layout consistency) — Width persistence decision aligned with 15b2.** The 15a transitional `elspeth_inspector_width` read survives only until Task 9; Phase 3B deletes width persistence and hardcodes `SIDERAIL_WIDTH = 320`. No Phase 6 deprecation step remains for that key.

### 2026-05-17 — Authoritative-plan no-go blockers absorbed

**BLOCKER (Version selector contract):** Task 1's `HeaderVersionSelector` test data now uses the live `CompositionStateVersion` shape (`id`, not `state_id`, plus `node_count`) and asserts the live store signature `revertToVersion(stateId)` rather than a nonexistent second `version` argument.

**BLOCKER (Event constant ownership):** Task 3 now creates/verifies `src/lib/composer-events.ts` as the canonical owner of `OPEN_GRAPH_MODAL_EVENT` and `OPEN_YAML_MODAL_EVENT`. Graph/YAML side-rail components, modals, router, command palette, and legacy caller retargets all import from that shared module; component files do not define their own event names.

**IMPORTANT (Persistent-state scope):** The trust-tier section now states that Phase 3B adds no new persistent state. The previously mentioned `elspeth_graph_mini_collapsed` key was removed because no task implements a collapsible graph mini.
