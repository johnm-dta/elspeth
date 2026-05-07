# RC4.2 UX Remediation — Inspector UX Subplan

Date: 2026-03-30 (expanded 2026-03-31, corrected 2026-03-31)
Status: Ready
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Scope

This subplan covers the three inspector-area UX enhancements from Wave 3,
plus a prerequisite bug fix in the frontend edge type model:

- **BUG FIX** — frontend `EdgeSpec` type doesn't match API response (Phase 7Z)
- `REQ-UX-05` — version control restructure (compact selector, revert in
  dropdown)
- `REQ-UX-07` — graph readability improvements (larger nodes, fit button, edge
  labels, minimap)
- `REQ-UX-08` — plugin catalog panel (collapsible reference drawer)

Primary surfaces:

- `types/index.ts` — EdgeSpec type correction (bug fix)
- `SpecView.tsx` — edge field name corrections + existing functionality
- `InspectorPanel.tsx` — header layout restructure
- `GraphView.tsx` — edge field corrections + node sizing, layout, edge labels,
  minimap
- New `CatalogDrawer.tsx` — plugin reference drawer
- `Layout.tsx` — no changes needed (`position: relative` already set on
  inspector grid cell at line 182)

Note: `REQ-UX-06` (validation status tint) is covered in sub-plan 05 and
already implemented.

---

## 2. Goals

- **Fix broken edge rendering** in both GraphView and SpecView caused by a
  frontend type that doesn't match the API response shape.
- Make the version control area compact and self-contained: version selector
  above the tabs, actions (validate, execute) adjacent, revert integrated into
  the version dropdown.
- Make the pipeline graph legible by default without manual zooming.
- Give users a browsable reference of available plugins without leaving the
  main workspace.

Non-goals:

- No interactive graph editing (drag-to-connect, add/remove nodes). The graph
  is a view, not an editor.
- No bidirectional spec-graph highlight sync (desirable follow-on, not in
  this wave).
- No plugin insertion from catalog panel (clicking to insert a chat
  suggestion is a design follow-on).

---

## 3. Architecture Decisions

### AD-0: Fix frontend EdgeSpec to match API response (BUG)

**Root cause:** The frontend `EdgeSpec` type at `types/index.ts:106-111` was
written speculatively with field names and edge_type values that don't match
the backend serialization. No transformation exists between the API response
and the frontend type — `parseResponse<CompositionState>` is a compile-time
type assertion, not a runtime validator.

**Backend serializes** (at `state.py:373-377`):
```json
{
  "id": "e1",
  "from_node": "source",
  "to_node": "t1",
  "edge_type": "on_success",
  "label": null
}
```

**Frontend type declares** (at `types/index.ts:106-111`):
```typescript
interface EdgeSpec {
  source: string;      // WRONG — API sends "from_node"
  target: string;      // WRONG — API sends "to_node"
  edge_type: "continue" | "route" | "error";  // WRONG — API sends "on_success" etc.
}
```

**Blast radius** — 17 broken field references across 2 components:

| File | Broken refs | Impact |
|------|------------|--------|
| `GraphView.tsx` | 6 (`edge.source` ×3, `edge.target` ×3, `edge.edge_type === "error"` ×2) | Edges don't render in graph, error animation never fires |
| `SpecView.tsx` | 11 (`edge.source` ×5, `edge.target` ×5, `edge.edge_type` ×2) | Upstream/downstream highlighting broken, connection indicators broken |

All edge-dependent features in both components are silently broken at
runtime. Nodes render correctly but edges, connection indicators, and
highlight-on-click are non-functional.

**Fix:** Correct `EdgeSpec` to match the API response, then update all 17
field references in GraphView and SpecView.

Corrected type:
```typescript
export interface EdgeSpec {
  id: string;
  from_node: string;
  to_node: string;
  edge_type: "on_success" | "on_error" | "route_true" | "route_false" | "fork";
  label: string | null;
}
```

Note: `id` was missing from the original type but is present in the API
response. Add it.

### AD-1: Two-row inspector header (version row + tab row)

Current layout (`InspectorPanel.tsx:124-168`): a single header row with tabs
on the left and version dropdown + action buttons on the right, all at the
same level.

New layout:

```
+----------------------------------------------+
|  v3 ▾  ● (tint)        [Validate] [Execute]  |  <- Row 1: version + actions
|  +------+-------+------+------+               |
|  | Spec | Graph | YAML | Runs |               |  <- Row 2: tabs
|  +------+-------+------+------+               |
+----------------------------------------------+
```

- Row 1: compact version selector (custom dropdown, not native `<select>`),
  validation tint dot (from sub-plan 05, already implemented), Validate
  button, Execute button.
- Row 2: tab strip (unchanged ARIA structure).
- Revert action moves INTO the version dropdown: selecting an old version
  shows a "Revert to this version" button inside the dropdown.

### AD-2: Custom version dropdown replaces native `<select>`

The native `<select>` (currently at `InspectorPanel.tsx:181-210`) can't show
a revert button or rich formatting (timestamp, node count) per option.
Replace with a custom dropdown:

- Trigger: compact button showing `v{N}` with a chevron.
- Panel: absolutely positioned below trigger, z-index above tab content.
- Each row: `v{N}` (bold) + node count + relative timestamp.
- Selected (non-current) row gets a "Revert" button inline.
- Click outside or Escape closes the panel.
- Keyboard: arrow keys navigate, Enter selects, Escape closes.

Lazy-load `stateVersions` on first open (existing `loadStateVersions()`
pattern, already wired to `onFocus`).

### AD-3: Graph node sizing and layout improvements

Current values (at `GraphView.tsx:29-30,44`): `NODE_WIDTH = 180`,
`NODE_HEIGHT = 50`, `nodesep = 40`, `ranksep = 60`.

New values:

| Parameter | Old | New | Rationale |
|-----------|-----|-----|-----------|
| `NODE_WIDTH` | 180 | 260 | Fits plugin name + type badge + node name comfortably |
| `NODE_HEIGHT` | 50 | 80 | Two-line content: node name (bold) + plugin name (muted) |
| `nodesep` | 40 | 60 | Prevents label overlap at larger node size |
| `ranksep` | 60 | 100 | Gives room for edge labels between ranks |

Node content (inside React Flow custom node):

```
+----------------------------+
|  [TRANSFORM]  classify     |  <- type badge + node name (node.name)
|  llm_transform             |  <- plugin name (node.plugin, muted)
+----------------------------+
```

**Important:** The frontend `NodeSpec` has both `name` and `id` as separate
fields. The graph displays `node.name` (user-visible name, currently used at
`GraphView.tsx:81` as `data: { label: node.name }`), not `node.id` (internal
identifier). The two-line layout shows `node.name` (bold) on line 1 with the
type badge, and `node.plugin` (muted) on line 2.

### AD-4: `fitView` with explicit options for readability

Replace the bare `fitView` prop (currently at `GraphView.tsx:143`) with
`fitViewOptions`:

```tsx
fitView
fitViewOptions={{
  padding: 0.15,
  maxZoom: 1.5,
  minZoom: 0.3,
}}
```

This ensures the graph starts at a readable zoom level rather than shrinking
to fit everything into a tiny viewport. For large graphs, the user pans; the
minimap provides orientation.

### AD-5: Edge labels from edge_type

After the AD-0 fix, `edge.edge_type` will be the backend values:
`"on_success"`, `"on_error"`, `"route_true"`, `"route_false"`, `"fork"`.

Map to readable labels:

| `edge_type` | Label | Notes |
|-------------|-------|-------|
| `on_success` | `success` | Normal flow |
| `on_error` | `error` | Error routing (keep dashed animated style) |
| `route_true` | `true` | Gate routing |
| `route_false` | `false` | Gate routing |
| `fork` | `fork` | Fork edges |
| (other) | `edge.label ?? edge.edge_type` | Fallback |

Implementation:

```tsx
const EDGE_LABEL_MAP: Record<string, string> = {
  on_success: "success",
  on_error: "error",
  route_true: "true",
  route_false: "false",
  fork: "fork",
};

// In edge mapping:
label: EDGE_LABEL_MAP[edge.edge_type] ?? edge.label ?? edge.edge_type,
```

Labels rendered via React Flow's `label` prop on edge objects, styled with
`fontSize: 10`, `fill: EDGE_LABEL_COLOR` (from `tokens.ts:32`). Error edges
keep the existing dashed animated style.

**SpecView also needs updating.** The SpecView `ConnectionIndicator`
component checks `edge.edge_type === "error"` (line 117) and
`edge.edge_type === "route"` (line 125). After the fix:
- `"error"` → `"on_error"`
- `"route"` → check for `"route_true"` or `"route_false"` (or use
  `.startsWith("route")`)

### AD-6: Minimap for graphs with >5 nodes

React Flow includes a built-in `<MiniMap>` component (part of
`@xyflow/react`, already imported). Enable conditionally:

```tsx
{rfNodes.length > 5 && (
  <MiniMap
    nodeStrokeWidth={3}
    zoomable
    pannable
    style={{ bottom: 8, right: 8, width: 120, height: 80 }}
  />
)}
```

Position: bottom-right corner of the graph area. Add `MiniMap` to the
existing `@xyflow/react` import at `GraphView.tsx:17-23`.

### AD-7: Catalog panel as a slide-over drawer, not a grid column

The current layout is a strict 3-column grid (sidebar, chat, inspector). Adding
a fourth column would reduce chat/inspector width and complicate the resize
logic.

Instead: the catalog panel is a **slide-over drawer** that overlays the
inspector area from the right edge. When open, it covers the inspector content
but not the chat or sidebar.

Properties:

- `position: absolute` within the inspector grid cell (not `fixed` over the
  whole viewport — the drawer belongs to the inspector area).
- Width: 320px (or the full inspector width if inspector is narrower).
- Height: 100% of the inspector area.
- Z-index: above tab content, below any modal dialogs.
- Backdrop: semi-transparent overlay over the inspector content.
- Toggle: a "Catalog" button in the inspector header (row 1, near the version
  selector) or a keyboard shortcut.
- Close: click backdrop, click X, or press Escape.

This approach:

- Requires no changes to `Layout.tsx` grid structure (the inspector outer
  `<div>` already has `position: "relative"` at `Layout.tsx:182`).
- Keeps the catalog contextually near the inspector.
- Doesn't steal space from chat.

### AD-8: Catalog data fetching and caching

The catalog is effectively static per server session (plugin list changes
require a server restart). Fetch on first open, cache indefinitely in
component-local state (no Zustand store needed).

Use the existing `client.ts` functions (confirmed present):
- `listSources()` (line 344)
- `listTransforms()` (line 352)
- `listSinks()` (line 360)
- `getPluginSchema()` (line 372)

Fetch strategy:

1. On drawer open: fetch all three lists in parallel (`Promise.all`).
2. Cache the lists in component state.
3. On detail expand: fetch `getPluginSchema(type, name)` lazily.
4. Cache schema details per plugin name in a `Map`.

No `catalogStore.ts` needed — the data is static and UI-local.

### AD-9: Gates are not plugins — no gate entries in the catalog

Gates are config-driven operations, not plugins (see CLAUDE.md). The backend
`PluginKind = Literal["source", "transform", "sink"]` does not include
`"gate"`. The catalog API never returns gate entries. The frontend
`PluginSummary.plugin_type` and `getPluginSchema()` use the same three
values. No gate filter is needed in the catalog drawer.

---

## 4. Detailed Changes

### Phase 7Z: Fix Frontend EdgeSpec (PREREQUISITE — do first)

This phase fixes a pre-existing bug. All subsequent phases depend on correct
edge types.

#### 7Z.1: Correct `EdgeSpec` type declaration

File: `src/elspeth/web/frontend/src/types/index.ts`

Replace (lines 106-111):
```typescript
export interface EdgeSpec {
  source: string;
  target: string;
  label: string | null;
  edge_type: "continue" | "route" | "error";
}
```

With:
```typescript
export interface EdgeSpec {
  id: string;
  from_node: string;
  to_node: string;
  edge_type: "on_success" | "on_error" | "route_true" | "route_false" | "fork";
  label: string | null;
}
```

#### 7Z.2: Update GraphView edge references (6 refs)

File: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`

| Line | Old | New |
|------|-----|-----|
| 50 | `g.setEdge(edge.source, edge.target)` | `g.setEdge(edge.from_node, edge.to_node)` |
| 97 | `id: \`e-${edge.source}-${edge.target}-${i}\`` | `id: \`e-${edge.from_node}-${edge.to_node}-${i}\`` |
| 98 | `source: edge.source` | `source: edge.from_node` |
| 99 | `target: edge.target` | `target: edge.to_node` |
| 101 | `animated: edge.edge_type === "error"` | `animated: edge.edge_type === "on_error"` |
| 103 | `edge.edge_type === "error"` | `edge.edge_type === "on_error"` |

#### 7Z.3: Update SpecView edge references (11 refs)

File: `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx`

| Line | Old | New |
|------|-----|-----|
| 66 | `edge.target === nodeId` | `edge.to_node === nodeId` |
| 67 | `upstream.add(edge.source)` | `upstream.add(edge.from_node)` |
| 69 | `edge.source === nodeId` | `edge.from_node === nodeId` |
| 70 | `downstream.set(edge.target, edge.label)` | `downstream.set(edge.to_node, edge.label)` |
| 114 | `n.id === edge.target` | `n.id === edge.to_node` |
| 115 | `edge.target` (fallback) | `edge.to_node` |
| 117 | `edge.edge_type === "error"` | `edge.edge_type === "on_error"` |
| 125 | `edge.edge_type === "route"` | `edge.edge_type.startsWith("route")` |
| 270 | `nodeDownstream.get(edge.source)` | `nodeDownstream.get(edge.from_node)` |
| 272 | `nodeDownstream.set(edge.source, existing)` | `nodeDownstream.set(edge.from_node, existing)` |
| 434 | `key={\`${edge.source}-${edge.target}-${i}\`}` | `key={\`${edge.from_node}-${edge.to_node}-${i}\`}` |

#### 7Z.4: Update existing tests

Any existing tests that construct `EdgeSpec` objects with `source`/`target`
fields must be updated to use `from_node`/`to_node` and backend edge_type
values. Check `SpecView.test.tsx` and `InspectorPanel.test.tsx`.

#### 7Z.5: Verify

Run `npx tsc --noEmit` (ignoring stale .d.ts warnings) and `npx vitest run`
to confirm all field references and tests are updated.

---

### Phase 7A: Version Control Restructure

#### 7A.1: Two-row header layout

File: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx`

Replace the current single header block (lines 124-319, containing the tab
strip, version dropdown, validation dot, and action buttons) with a two-row
structure:

**Row 1 (version + actions):**

```tsx
<div style={{
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "6px 12px",
  borderBottom: "1px solid var(--color-border)",
}}>
  {/* Left: version selector + validation dot */}
  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
    <VersionSelector />
    {/* ValidationDot — already implemented in sub-plan 05 */}
    {compositionState && compositionState.nodes.length > 0 && (
      <span aria-label={...} style={{ ... 8px dot ... }} />
    )}
  </div>
  {/* Right: action buttons */}
  <div style={{ display: "flex", gap: 6 }}>
    {/* Catalog button (new) */}
    {/* Validate button (existing, move here) */}
    {/* Execute button (existing, move here) */}
  </div>
</div>
```

**Row 2 (tab strip):**

```tsx
<div style={{
  padding: "4px 12px 0",
  borderBottom: "1px solid var(--color-border)",
}}>
  <div role="tablist" aria-label="Inspector tabs" ...>
    {/* Same tab buttons as current */}
  </div>
</div>
```

#### 7A.2: `VersionSelector` component (new, inline in InspectorPanel)

Extract from InspectorPanel into its own component or an inline sub-component.

Props: none (reads from stores directly).

**Trigger button:** compact `v{N} ▾` button. Shows current version
number. `aria-expanded`, `aria-haspopup="listbox"`.

**Dropdown panel:** absolutely positioned below trigger.

- Each row: `v{N}` (bold) + `{node_count} nodes` (muted) + timestamp
- Non-current versions get an inline "Revert" button
- Revert fires `window.confirm()` then `revertToVersion(stateId)`
- Click-outside close via fixed backdrop at z-index one below the panel

**Keyboard support:**

- Escape closes the dropdown
- Arrow keys navigate version rows
- Enter activates the focused row's revert button
- Focus returns to trigger on close

**Lazy loading:** calls `loadStateVersions()` on first open (same as current
`onFocus` pattern).

---

### Phase 7B: Graph Readability

#### 7B.1: Update node sizing and layout constants

File: `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx`

```tsx
const NODE_WIDTH = 260;   // was 180 (line 29)
const NODE_HEIGHT = 80;   // was 50 (line 30)

// In dagre config (line 44):
g.setGraph({ rankdir: "TB", nodesep: 60, ranksep: 100 });
//                           was 40       was 60
```

#### 7B.2: Richer node content

Replace the current single-line node label (`data: { label: node.name }` at
line 81) with a two-line layout showing type badge + node name on line 1,
plugin name on line 2.

In the nodes array mapping (around line 79-94), replace the current
`data.label` with a structured JSX element:

- Line 1: type badge (small pill with `BADGE_BACKGROUNDS[node.type]` +
  `BADGE_COLORS[node.type]`, uppercase text, `fontSize: 10`) + `node.name`
  (`fontSize: 13, fontWeight: 600`)
- Line 2: `node.plugin` in muted text (`fontSize: 11,
  color: var(--color-text-muted)`)

Node container style: `width: NODE_WIDTH`, `height: NODE_HEIGHT`,
`borderRadius: 8`, `border: 1px solid var(--color-border-strong)`,
`backgroundColor: var(--color-surface-elevated)`.

The type badges should use the existing `BADGE_COLORS` and
`BADGE_BACKGROUNDS` maps from `@/styles/tokens` (already imported at line
27). The `NodeSpec.type` field is
`"transform" | "gate" | "aggregation" | "coalesce"` — all four have entries
in both badge maps. (Gates are a node type, not a plugin type.)

#### 7B.3: Edge labels

**Prerequisite:** Phase 7Z must be complete. After the fix, `edge.edge_type`
is the backend value (`"on_success"`, `"on_error"`, etc.) and edge field
names are `from_node`/`to_node`.

Add a label map constant:

```tsx
const EDGE_LABEL_MAP: Record<string, string> = {
  on_success: "success",
  on_error: "error",
  route_true: "true",
  route_false: "false",
  fork: "fork",
};
```

In the edges array mapping, replace the current label assignment with:

```tsx
label: EDGE_LABEL_MAP[edge.edge_type] ?? edge.label ?? edge.edge_type,
```

Edge label styling is already in place at line 106:
`labelStyle: { fontSize: 10, fill: EDGE_LABEL_COLOR }`. Error edges retain
existing `animated: true` and `EDGE_COLORS.error` stroke colour (now
correctly matching `edge.edge_type === "on_error"` after 7Z).

#### 7B.4: fitView options

Replace bare `fitView` prop (line 143) with:

```tsx
fitView
fitViewOptions={{ padding: 0.15, maxZoom: 1.5, minZoom: 0.3 }}
```

#### 7B.5: Minimap

After `<Controls showInteractive={false} />` (line 147), add:

```tsx
{rfNodes.length > 5 && (
  <MiniMap
    nodeStrokeWidth={3}
    zoomable
    pannable
    style={{ bottom: 8, right: 8, width: 120, height: 80 }}
  />
)}
```

Import: add `MiniMap` to the `@xyflow/react` import at line 17-23.

#### 7B.6: Verify fit button

The `<Controls>` component from React Flow (line 147) includes a built-in
fit-to-view button. Verify it's visible and uses the `fitViewOptions`. If
not, add a standalone button. Try the built-in one first.

---

### Phase 7C: Plugin Catalog Panel

#### 7C.1: `CatalogDrawer` component (new file)

File: `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx`

Props: `isOpen: boolean`, `onClose: () => void`.

Structure:

- **Backdrop** (position: absolute, inset: 0, semi-transparent, z-index: 30)
- **Drawer** (position: absolute, top/right/bottom: 0, width: 320px,
  z-index: 31)
  - Header: "Plugin Catalog" title + close button
  - Three tabs: Sources, Transforms, Sinks (role="tablist")
  - Scrollable plugin list per active tab
  - Each plugin renders via `PluginCard`

Data management:

- Fetch on first open: `Promise.all([listSources(), listTransforms(),
  listSinks()])` — these functions exist in `client.ts` at lines 344, 352,
  360.
- Cache in component state (no Zustand store)
- Schema details fetched lazily per plugin on expand via
  `getPluginSchema(type, name)` (line 372 of `client.ts`)
- Schema cache in a `Map<string, PluginSchemaInfo>` in component state

Gates are config-driven operations, not plugins (AD-9). The catalog API
never returns gate entries.

Keyboard: Escape closes drawer.

#### 7C.2: `PluginCard` component (new file)

File: `src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx`

Props: `plugin: PluginSummary`, `schema: PluginSchemaInfo | null`,
`onExpand: () => void`.

Collapsible card:

- **Collapsed:** plugin name (bold, 13px) + one-line description (muted, 11px)
- **Expanded:** triggers `onExpand` (lazy schema fetch), then renders config
  field list from schema:
  - Field name (bold)
  - Type (muted)
  - Required/optional badge
  - Description

#### 7C.3: Catalog toggle in InspectorPanel

In `InspectorPanel.tsx`:

- Add `const [catalogOpen, setCatalogOpen] = useState(false)` state.
- Add "Catalog" button in Row 1 of the header, before the Validate button.
- Render `<CatalogDrawer isOpen={catalogOpen} onClose={() =>
  setCatalogOpen(false)}>` inside the inspector's outer container (OUTSIDE
  the header/tab content flow, so it overlays correctly).
- The inspector's outer `<div>` in `Layout.tsx` already has
  `position: "relative"` (line 182), so the drawer's absolute positioning
  is already scoped correctly. No Layout changes needed.

---

### Phase 7D: Tests

#### 7D.1: EdgeSpec fix tests

Verify edge rendering works after the type fix. Add to existing
`InspectorPanel.test.tsx` or `SpecView.test.tsx` (whichever is more
appropriate):

| Test | Expected |
|------|----------|
| `SpecView renders connection indicators with correct edge fields` | Connection indicators visible for edges using `from_node`/`to_node` |

#### 7D.2: InspectorPanel tests

Extend: `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx`
(file already exists from sub-plan 05 with 5 tests for ValidationDot)

| Test | Expected |
|------|----------|
| `renders two-row header` | Version selector and action buttons in row 1, tabs in row 2 |
| `version selector shows current version` | Button text contains `v{N}` |
| `version dropdown opens on click` | Dropdown panel visible after click |
| `revert button appears for non-current version` | Revert button shown for historical version |
| `catalog button toggles drawer` | CatalogDrawer renders when button clicked |

#### 7D.3: GraphView tests

New file: `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx`

| Test | Expected |
|------|----------|
| `renders nodes with correct dimensions` | Node style includes `width: 260` |
| `renders edge labels for on_success type` | Edge label contains "success" |
| `renders edge labels for route_true type` | Edge label contains "true" |
| `shows minimap for >5 nodes` | MiniMap component renders |
| `hides minimap for <=5 nodes` | MiniMap not in DOM |
| `renders type badge and plugin name` | Node content includes type badge text and plugin text |

#### 7D.4: CatalogDrawer tests

New file: `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx`

| Test | Expected |
|------|----------|
| `renders nothing when closed` | No DOM output |
| `fetches catalog on first open` | API calls fired for sources, transforms, sinks |
| `shows three tabs` | Sources, Transforms, Sinks tabs visible |
| `switching tabs shows correct plugin list` | Tab content changes |
| `expanding plugin card fetches schema` | `getPluginSchema` called |
| `escape key closes drawer` | `onClose` called on Escape |
| `backdrop click closes drawer` | `onClose` called on backdrop click |

---

## 5. Implementation Order

```
Phase 7Z (EdgeSpec fix) ── MUST be first ────────────────────+
  7Z.1  Correct EdgeSpec type in types/index.ts               |
  7Z.2  Update GraphView edge references (6 refs)             |
  7Z.3  Update SpecView edge references (11 refs)             |
  7Z.4  Update existing tests                                 |
  7Z.5  Verify tsc + vitest                                   |
                                                              |
Phase 7A (version restructure) ── after 7Z ──────────────────+
  7A.1  Two-row header layout                                 |
  7A.2  VersionSelector component                             |
                                                              |
Phase 7B (graph readability) -- after 7Z, independent of 7A ─+
  7B.1  Node sizing + layout constants                        |
  7B.2  Richer node content                                   |
  7B.3  Edge labels                                           |
  7B.4  fitView options                                       |
  7B.5  Minimap                                               |
  7B.6  Verify fit button                                     |
                                                              |
Phase 7C (catalog panel) -- independent of 7A/7B ────────────+
  7C.1  CatalogDrawer component                               |
  7C.2  PluginCard component                                  |
  7C.3  Catalog toggle in InspectorPanel (after 7A)           |
                                                              |
Phase 7D (tests) -- after 7Z + 7A + 7B + 7C ─────────────────+
  7D.1  EdgeSpec fix tests
  7D.2  InspectorPanel tests (extend existing file)
  7D.3  GraphView tests (new file)
  7D.4  CatalogDrawer tests (new file)
```

**Phase 7Z is a hard prerequisite** — all edge-dependent work in 7A, 7B, and
7D depends on correct field names. Do it first, verify, then proceed.

**After 7Z:** Phases 7A, 7B, and 7C are independent and can proceed in
parallel on different files. 7C.3 (catalog toggle) depends on 7A (new header
layout). If running sequentially: 7Z → 7A → 7B → 7C → 7D.

**Dependencies:**

- Sub-plan 05 already landed — the `ValidationDot` is in place in the
  header. Phase 7A.1 will move it to the new row 1 layout.
- No backend changes in this sub-plan. All work is frontend-only.

**Estimated scope:** ~100 lines EdgeSpec fix (type + 17 field refs + tests),
~250 lines InspectorPanel restructure, ~100 lines GraphView changes, ~250
lines CatalogDrawer + PluginCard, ~200 lines tests. Two new component files,
two new test files. No backend changes.

---

## 6. Files Affected

### Modified

| File | Changes |
|------|---------|
| `src/elspeth/web/frontend/src/types/index.ts` | `EdgeSpec` corrected: field names + edge_type enum + add `id` |
| `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` | Edge field corrections (6 refs), node sizing, layout config, edge labels, minimap, fitView options |
| `src/elspeth/web/frontend/src/components/inspector/SpecView.tsx` | Edge field corrections (11 refs), edge_type comparisons |
| `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.tsx` | Two-row header, VersionSelector, catalog toggle, drawer mount |

### New

| File | Purpose |
|------|---------|
| `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx` | Slide-over drawer with tabbed plugin listing |
| `src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx` | Collapsible plugin detail card |
| `src/elspeth/web/frontend/src/components/inspector/GraphView.test.tsx` | Graph readability tests |
| `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx` | Catalog drawer tests |

### Existing test file extended

| File | Changes |
|------|---------|
| `src/elspeth/web/frontend/src/components/inspector/InspectorPanel.test.tsx` | New tests for version selector and catalog toggle (alongside existing 5 ValidationDot tests) |
| `src/elspeth/web/frontend/src/components/inspector/SpecView.test.tsx` | Edge field name updates in existing tests |

### NOT modified

| File | Why |
|------|-----|
| `src/elspeth/web/frontend/src/components/common/Layout.tsx` | Inspector outer div already has `position: "relative"` (line 182) |
| Theme CSS / tokens file | All referenced variables exist: `BADGE_COLORS`, `BADGE_BACKGROUNDS`, `EDGE_COLORS`, `EDGE_LABEL_COLOR` in `tokens.ts`; semantic CSS variables in `App.css` |

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **EdgeSpec fix breaks existing edge-dependent code** | **Mitigated** | The existing code is already broken at runtime — `edge.source` is `undefined`. The fix makes the code work. TypeScript will catch any missed references at compile time. |
| Custom dropdown accessibility | Medium | Implement full keyboard navigation (arrow keys, Escape, Enter), `aria-expanded`, `aria-haspopup`, focus management. Test with screen reader. |
| Larger graph nodes overflow small inspector panels | Low | `fitViewOptions` with `minZoom: 0.3` allows shrinking. Minimap aids orientation. Inspector is resizable (drag handle in Layout). |
| Catalog drawer covers inspector content | Low | By design — it's a temporary overlay. Backdrop click and Escape close it immediately. |
| React Flow MiniMap dependency | Low | `<MiniMap>` is included in the `@xyflow/react` package already imported. No new dependency. |
| Gate/plugin confusion in docs and types | **Fixed** | Gates are config-driven, not plugins. Removed `"gate"` from `PluginSummary`, `PluginSchemaInfo`, `getPluginSchema`. Cleaned stale doc references (AD-9). |

---

## 8. Acceptance Criteria

### BUG FIX: EdgeSpec Type Correction

- [ ] `EdgeSpec` type uses `from_node`/`to_node` (not `source`/`target`).
- [ ] `EdgeSpec` type uses backend `edge_type` values (`on_success`,
      `on_error`, `route_true`, `route_false`, `fork`).
- [ ] `EdgeSpec` type includes `id` field.
- [ ] All 6 GraphView edge references use correct field names.
- [ ] All 11 SpecView edge references use correct field names.
- [ ] Edge-dependent features (graph edges, connection indicators,
      upstream/downstream highlighting) work at runtime.

### REQ-UX-05: Version Control Restructure

- [ ] Inspector header has two rows: version + actions on top, tabs below.
- [ ] Version selector is a compact `v{N}` button (not a native select).
- [ ] Version dropdown shows version number, node count, and timestamp per
      entry.
- [ ] Revert button appears inline in the dropdown for non-current versions.
- [ ] Keyboard navigation works in the dropdown (arrows, Enter, Escape).
- [ ] ValidationDot (from sub-plan 05) appears in row 1 next to version
      selector.

### REQ-UX-07: Graph Readability

- [ ] Nodes are 260x80px with type badge and plugin name visible.
- [ ] Node displays `node.name` (bold) + `node.plugin` (muted), not
      `node.id`.
- [ ] Edge labels show readable connection types: `success` for on_success,
      `error` for on_error, `true`/`false` for routes, `fork` for forks.
- [ ] Graphs with >5 nodes show a minimap in the bottom-right corner.
- [ ] Default zoom level is readable without manual adjustment.
- [ ] Fit-to-view control is available and functional.

### REQ-UX-08: Plugin Catalog Panel

- [ ] "Catalog" button in inspector header opens a slide-over drawer.
- [ ] Drawer has three tabs: Sources, Transforms, Sinks.
- [ ] No gate entries in catalog (gates are config-driven, not plugins).
- [ ] Each plugin shows name and one-line description.
- [ ] Expanding a plugin shows config schema details.
- [ ] Drawer closes on backdrop click, X button, or Escape.
- [ ] Catalog data is fetched once and cached.
