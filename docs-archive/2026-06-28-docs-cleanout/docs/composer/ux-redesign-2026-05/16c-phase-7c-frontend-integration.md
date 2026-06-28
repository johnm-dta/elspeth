# Phase 7C — Frontend: Catalog drawer integration (filters, synthetic entry, shortcuts)

> **⚠️ HISTORICAL — trust-tier filter scope rescinded 2026-05-18.**
> The plan body below preserves the OD-C decision NOT to surface trust
> tier as a filter dimension (which proved prescient); commit
> `c76ecc0f2` extended that rationale to delete the underlying
> `data_trust_tier` field entirely. The `data_trust_tier` lines in
> test fixtures below are preserved as historical record; do not copy
> them into new tests — `PluginSummary` no longer carries the field.

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the integration half of Phase 7B's frontend work — build
the `FilterChipStrip` component, the synthetic `InlineChatSourceEntry`
entry, wire them plus the extended search into `CatalogDrawer.tsx`, and
regroup the keyboard shortcuts help under "Actions" / "Reference"
subheadings.

**Architecture:** Frontend-only. Consumes the primitives shipped in
[16b-phase-7b-frontend.md](16b-phase-7b-frontend.md) (the rewritten
`PluginCard`, the `AuditCharacteristicIcon`, the
`auditCharacteristics` metadata table). Adds two filter dimensions
that compose with the existing fuzzy search via AND. The "Inline data
from chat" entry is a synthetic frontend-only catalog row, not a
backend plugin (reconnaissance confirmed no such plugin exists; design
doc 08 frames it as "an option, not a plugin"); it renders as the first
row of the Sources tab with a distinct visual style and a different
action (prefill the chat input rather than expand a schema).

**Tech Stack:** React + TypeScript + Vitest + testing-library.

**Sibling plans:**

- [16a-phase-7a-backend.md](16a-phase-7a-backend.md) — backend metadata
  schema, catalog API extension, audit-characteristic derivation,
  canonical CSV example.
- [16b-phase-7b-frontend.md](16b-phase-7b-frontend.md) — frontend
  primitives: extended types, the `auditCharacteristics` metadata table,
  `AuditCharacteristicIcon`, the `PluginCard` rewrite.

**Roadmap reference:**
[00-implementation-roadmap.md](00-implementation-roadmap.md). Design doc:
[08-catalog-reshape.md](08-catalog-reshape.md).

## Implementation worktree (batched with 16a + 16b)

Phase 7 ships as a single batched PR covering 16a + 16b + 16c. This plan
is the last to run (drawer integration consumes 16b primitives and 16a
API surface). The worktree should already exist from 16a/16b execution:

```bash
cd /home/john/elspeth/.worktrees/phase-7-catalog
source .venv/bin/activate
```

If for some reason the worktree wasn't created earlier:

```bash
git -C /home/john/elspeth worktree add .worktrees/phase-7-catalog -b feat/phase-7-catalog
cd /home/john/elspeth/.worktrees/phase-7-catalog
python3.13 -m venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cd src/elspeth/web/frontend && npm install
```

**Order:** This plan runs LAST. Task 1 Step 0 is a mechanical preflight
that hard-aborts if 16b's primitives (`auditCharacteristics.ts`,
`AuditCharacteristicIcon.tsx`, `capability_tags` on `PluginSummary`) are
absent. Do not edit around the preflight; if it aborts, ship 16b before
returning here.

**Discipline (operator-known gotchas):** activate `.venv` before any
`uv pip install`; keep Python at 3.13 to match main; when delegating to
subagents prefix prompts with absolute paths and CWD discipline
(subagents silently misread the worktree path otherwise); prefer
`mcp__filigree__*` over the `filigree` CLI from inside the worktree.
The Playwright spec added in Task 6 requires a running staging instance —
see Task 6 Step 2 for invocation context.

**Final PR shape:** single PR from `feat/phase-7-catalog` → `RC5.2`
covering 16a + 16b + 16c. The PR description should pull the
rev-history bullets from each plan. See
[16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md#implementation-worktree-batched)
for the full batch protocol.

---

## Scope boundaries

**In scope:**

- New `FilterChipStrip.tsx` component with two filter groups:
  capability tag, audit characteristic. Each is a multi-select pill
  group; empty selection = "all"; filters compose with each other and
  with search via AND. Trust tier is kind-derived internal metadata
  (per 16b OD-C rationale) and is not surfaced as a filter dimension.
- New `InlineChatSourceEntry.tsx` synthetic catalog entry. Rendered as
  the **first** row of the Sources tab only; clicking dispatches
  `PREFILL_CHAT_INPUT_EVENT` with a suggested prompt and closes the
  drawer.
- `CatalogDrawer.tsx` integration: **per-tab** filter state alongside
  `searchQuery`; filter-chip strip between search and tab strip;
  synthetic entry at top of Sources; extended `scorePlugin` that
  fuzzy-matches across the new prose fields and capability tags;
  combined `pluginList` predicate (`matchesFilters AND scoreHit`);
  extended `counts` memo.
- `ShortcutsHelp.tsx` regroup: split shortcuts into "Actions" /
  "Reference" subheadings; move `Ctrl+Shift+P` under Reference per
  roadmap question D1.
- Manual staging smoke exercising the reshape end-to-end.

**Out of scope:**

- **CSS styling for the new class names.** This plan introduces classes
  including `filter-chip`, `filter-chip-active`, `filter-chip-group`,
  `filter-chip-clear`, `inline-chat-source-entry`,
  `inline-chat-source-entry-badge`, `shortcuts-subheading`, etc. used
  by Vitest for structural correctness. Visual styling for these is a
  CSS pass that must land before the staging-smoke verification in
  Task 5; the Vitest tests pass on class presence alone. If 16b shipped
  without its CSS pass either, fold the styling for both phases into
  one CSS commit before Task 5.
- Backend (Phase 7A delivered).
- Primitives (Phase 7B delivered — types, audit metadata, leaf
  components, PluginCard rewrite).
- ~~Per-tab filter state.~~ (Moved into scope per rev-2 review: each
  tab has its own `CatalogFilters` record so an active capability-tag
  filter on Sources does not hide every Transform on tab switch.)
- Telemetry on filter usage (Phase 8 polish).
- Schema-field-name search (lazy-fetch conflict; deferred to a future
  phase per the 16b out-of-scope statement).

## Trust tier check (per CLAUDE.md)

All data flows are frontend-internal. `CatalogFilters` state lives in
React state (`useState`), is never persisted, and is reset on drawer
close. The synthetic entry's `PREFILL_CHAT_INPUT_EVENT` payload is a
constant string compiled into the bundle; no Tier-3 boundary.

## File structure

**Created:**

- `src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.tsx`
- `src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.test.tsx`
- `src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.tsx`
- `src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.test.tsx`

**Modified:**

- `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx`
- `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx`
- `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`
- `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`

## Verification approach

Each task is TDD-shaped: failing test, minimal implementation, passing
test, commit. Vitest runs are scoped to the file under test; a final
regression sweep runs the full frontend suite. Task 5 (the final
regression sweep + staging smoke) gates phase completion.

---

## Task 1: `FilterChipStrip.tsx` — filter chip UI

**Files:**

- Create:
  `src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.tsx`
- Create:
  `src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.test.tsx`

The filter strip has two groups: capability tags and audit
characteristics. Each is a multi-select pill group; an empty selection
means "all." Filters compose with each other and with search via AND.
Trust tier is not a filter dimension (see 16b OD-C rationale).

- [ ] **Step 0: Preflight — verify 16b primitives have shipped**

16c imports from 16b's primitives. Run these checks before writing any
failing test. If any check fails, **abort**: 16b has not shipped; ship
16b first and re-run from this step.

```bash
test -f src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts \
  || { echo "ABORT: auditCharacteristics.ts missing — ship 16b first"; exit 1; }

test -f src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.tsx \
  || { echo "ABORT: AuditCharacteristicIcon.tsx missing — ship 16b first"; exit 1; }

grep -c "capability_tags" src/elspeth/web/frontend/src/types/index.ts \
  | grep -qE '^[1-9]' \
  || { echo "ABORT: capability_tags not present in types/index.ts — ship 16b first"; exit 1; }

echo "Preflight OK — 16b primitives confirmed present."
```

Expected: all three checks pass; "Preflight OK" printed. A missing file
or a zero-count grep exits 1 with an explicit message, preventing a
confusing module-not-found failure later in the task.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";

const ALL_OFF: CatalogFilters = {
  capabilityTags: new Set(),
  auditCharacteristics: new Set(),
};

describe("FilterChipStrip", () => {
  it("renders one chip per capability tag", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv", "file", "http"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /csv/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^http/i })).toBeInTheDocument();
  });

  it("emits an updated filter set when a chip is toggled", async () => {
    const onChange = vi.fn();
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /csv/i }));
    expect(onChange).toHaveBeenCalled();
    const updated: CatalogFilters = onChange.mock.calls[0][0];
    expect(updated.capabilityTags.has("csv")).toBe(true);
  });

  it("toggling an active chip removes it", async () => {
    const onChange = vi.fn();
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={{ ...ALL_OFF, capabilityTags: new Set(["csv"]) }}
        onChange={onChange}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /csv/i }));
    const updated: CatalogFilters = onChange.mock.calls[0][0];
    expect(updated.capabilityTags.has("csv")).toBe(false);
  });

  it("renders 'Clear filters' when any filter is active", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={{ ...ALL_OFF, capabilityTags: new Set(["csv"]) }}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /clear filters/i })).toBeInTheDocument();
  });

  it("does not render 'Clear filters' when no filters are active", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.queryByRole("button", { name: /clear filters/i })).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/FilterChipStrip.test.tsx
```

Expected: FAIL — component doesn't exist.

- [ ] **Step 3: Implement the component**

```typescript
// ============================================================================
// FilterChipStrip
//
// Two groups of filter chips (capability tags and audit characteristics)
// at the top of each catalog tab. Filters compose with each other and
// with search via AND: a plugin must match the search query AND have a
// tag in every active group it appears in. Empty group = "all."
//
// Per design doc 08-§Filters: "The filter strip lets users narrow the
// catalog to 'what works for my sensitive-data pipeline' or 'what
// doesn't make a network call' in one click."
// ============================================================================

import { useCallback } from "react";
import { lookupAuditCharacteristic } from "./auditCharacteristics";

// Trust tier is kind-derived internal metadata (see 16b OD-C rationale):
// it is not surfaced as a filter dimension in the catalog UI.
export interface CatalogFilters {
  capabilityTags: Set<string>;
  auditCharacteristics: Set<string>;
}

interface FilterChipStripProps {
  availableCapabilityTags: string[];
  availableAuditCharacteristics: string[];
  filters: CatalogFilters;
  onChange: (next: CatalogFilters) => void;
}

function toggle<T>(set: Set<T>, value: T): Set<T> {
  const next = new Set(set);
  if (next.has(value)) next.delete(value);
  else next.add(value);
  return next;
}

export function FilterChipStrip({
  availableCapabilityTags,
  availableAuditCharacteristics,
  filters,
  onChange,
}: FilterChipStripProps) {
  const anyActive =
    filters.capabilityTags.size > 0 ||
    filters.auditCharacteristics.size > 0;

  const toggleTag = useCallback(
    (tag: string) => onChange({ ...filters, capabilityTags: toggle(filters.capabilityTags, tag) }),
    [filters, onChange],
  );
  const toggleAudit = useCallback(
    (flag: string) => onChange({ ...filters, auditCharacteristics: toggle(filters.auditCharacteristics, flag) }),
    [filters, onChange],
  );
  const clear = useCallback(
    () =>
      onChange({
        capabilityTags: new Set(),
        auditCharacteristics: new Set(),
      }),
    [onChange],
  );

  return (
    <div className="filter-chip-strip" aria-label="Catalog filters">
      {availableCapabilityTags.length > 0 && (
        <ChipGroup label="Capability">
          {availableCapabilityTags.map((tag) => (
            <Chip
              key={tag}
              active={filters.capabilityTags.has(tag)}
              onToggle={() => toggleTag(tag)}
              label={tag}
            />
          ))}
        </ChipGroup>
      )}
      {availableAuditCharacteristics.length > 0 && (
        <ChipGroup label="Audit">
          {availableAuditCharacteristics.map((flag) => {
            const meta = lookupAuditCharacteristic(flag);
            const label = meta?.label ?? flag;
            return (
              <Chip
                key={flag}
                active={filters.auditCharacteristics.has(flag)}
                onToggle={() => toggleAudit(flag)}
                label={label}
              />
            );
          })}
        </ChipGroup>
      )}
      {anyActive && (
        <button
          type="button"
          className="filter-chip-clear"
          onClick={clear}
          aria-label="Clear filters"
        >
          Clear filters
        </button>
      )}
    </div>
  );
}

function ChipGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="filter-chip-group">
      <span className="filter-chip-group-label">{label}:</span>
      {children}
    </div>
  );
}

function Chip({ active, onToggle, label }: { active: boolean; onToggle: () => void; label: string }) {
  return (
    <button
      type="button"
      className={`filter-chip ${active ? "filter-chip-active" : ""}`}
      aria-pressed={active}
      onClick={onToggle}
    >
      {label}
    </button>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/FilterChipStrip.test.tsx
```

Expected: PASS — all five tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.tsx \
  src/elspeth/web/frontend/src/components/catalog/FilterChipStrip.test.tsx
git commit -m "feat(frontend): FilterChipStrip component for catalog filters (Phase 7C.1)"
```

## Task 2: `InlineChatSourceEntry.tsx` — synthetic source entry

**Files:**

- Create:
  `src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.tsx`
- Create:
  `src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.test.tsx`

Per design doc 08-§"The 'Inline data from chat' entry," this row sits at
the top of the Sources tab. It is **not** a backend plugin (recon
confirmed); it's a synthetic frontend-only affordance with a different
visual style and a different action. Clicking it dispatches
`PREFILL_CHAT_INPUT_EVENT` with a suggested prompt; `ChatInput.tsx`
already listens.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { InlineChatSourceEntry } from "./InlineChatSourceEntry";
import { PREFILL_CHAT_INPUT_EVENT } from "./PluginCard";

describe("InlineChatSourceEntry", () => {
  let handler: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    handler = vi.fn();
    window.addEventListener(PREFILL_CHAT_INPUT_EVENT, handler);
  });
  afterEach(() => {
    window.removeEventListener(PREFILL_CHAT_INPUT_EVENT, handler);
  });

  it("renders the entry title and description", () => {
    render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
    expect(screen.getByText(/type your data directly/i)).toBeInTheDocument();
  });

  it("renders a distinct visual style (not a regular plugin-card)", () => {
    const { container } = render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    // The synthetic-entry class is what differentiates it visually.
    expect(container.firstChild).toHaveClass("inline-chat-source-entry");
  });

  it("dispatches PREFILL_CHAT_INPUT_EVENT and closes the drawer on click", async () => {
    const onCloseDrawer = vi.fn();
    render(<InlineChatSourceEntry onCloseDrawer={onCloseDrawer} />);
    await userEvent.click(screen.getByRole("button", { name: /try it/i }));
    expect(handler).toHaveBeenCalled();
    expect(onCloseDrawer).toHaveBeenCalled();
  });

  it("dispatches a non-empty string detail", async () => {
    render(<InlineChatSourceEntry onCloseDrawer={() => {}} />);
    await userEvent.click(screen.getByRole("button", { name: /try it/i }));
    const event = handler.mock.calls[0][0] as CustomEvent<string>;
    expect(typeof event.detail).toBe("string");
    expect(event.detail.length).toBeGreaterThan(10);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/InlineChatSourceEntry.test.tsx
```

Expected: FAIL — component doesn't exist.

- [ ] **Step 3: Implement the synthetic entry**

```typescript
// ============================================================================
// InlineChatSourceEntry
//
// Synthetic catalog entry rendered as the first row of the Sources tab.
// NOT a backend plugin — it represents "type your data directly in chat;
// no plugin required" per design doc 08-§"The 'Inline data from chat'
// entry" and project_composer_dynamic_source_from_chat.
//
// Clicking the "Try it" action prefills the chat input with a suggested
// prompt and closes the catalog drawer. The composer's chat-driven
// flow takes it from there, creating a one-row dynamic source from the
// user's adapted prompt at runtime.
// ============================================================================

import { useCallback } from "react";
import { PREFILL_CHAT_INPUT_EVENT } from "./PluginCard";

const SUGGESTED_PROMPT =
  "Use the LLM to summarise this article in one sentence: " +
  "https://example.com/article";

interface InlineChatSourceEntryProps {
  onCloseDrawer: () => void;
}

export function InlineChatSourceEntry({ onCloseDrawer }: InlineChatSourceEntryProps) {
  const handleClick = useCallback(() => {
    window.dispatchEvent(
      new CustomEvent(PREFILL_CHAT_INPUT_EVENT, { detail: SUGGESTED_PROMPT }),
    );
    onCloseDrawer();
  }, [onCloseDrawer]);

  return (
    <div className="inline-chat-source-entry" role="region" aria-label="Inline data from chat">
      <div className="inline-chat-source-entry-header">
        <span className="inline-chat-source-entry-title">Inline data from chat</span>
        <span className="inline-chat-source-entry-badge">no plugin needed</span>
      </div>
      <div className="inline-chat-source-entry-desc">
        Type your data directly into chat for small inputs — a URL, a sentence, one record.
        The composer creates a one-row dynamic source from your message. Best for ad-hoc
        runs, demos, and exploring; switch to a real source plugin when you have a
        recurring batch.
      </div>
      <div className="inline-chat-source-entry-example">
        <div className="inline-chat-source-entry-example-label">Suggested prompt:</div>
        <pre className="inline-chat-source-entry-example-code">{SUGGESTED_PROMPT}</pre>
      </div>
      <button
        type="button"
        className="btn btn-small inline-chat-source-entry-try"
        onClick={handleClick}
      >
        Try it
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/InlineChatSourceEntry.test.tsx
```

Expected: PASS — all four tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.tsx \
  src/elspeth/web/frontend/src/components/catalog/InlineChatSourceEntry.test.tsx
git commit -m "feat(frontend): InlineChatSourceEntry synthetic catalog row (Phase 7C.2)"
```

## Task 3: Wire filters, extended search, and synthetic entry into `CatalogDrawer.tsx`

**Files:**

- Modify:
  `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx`.

This task ties the previous tasks together. The drawer now:

1. Maintains a `filters` state in addition to `searchQuery`.
2. Renders a `FilterChipStrip` between the search input and the tab strip.
3. Renders `InlineChatSourceEntry` as the **first** row of the Sources tab
   (and only the Sources tab), unaffected by filters or search.
4. Extends `scorePlugin` to fuzzy-match across the new prose fields and
   capability tags.
5. Composes filters with search (AND): a plugin matches iff it passes
   the search predicate AND every active filter group.

- [ ] **Step 1: Write the failing test additions**

In `CatalogDrawer.test.tsx`, add (or replace where appropriate):

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CatalogDrawer } from "./CatalogDrawer";
import * as api from "@/api/client";

vi.mock("@/api/client", () => ({
  listSources: vi.fn(),
  listTransforms: vi.fn(),
  listSinks: vi.fn(),
  getPluginSchema: vi.fn(),
}));

const csv = {
  name: "csv",
  plugin_type: "source",
  description: "Read CSV files.",
  config_fields: [],
  usage_when_to_use: "When you have a CSV file.",
  usage_when_not_to_use: null,
  example_use: null,
  capability_tags: ["csv", "file"],
  audit_characteristics: ["io_read", "quarantine"],
  data_trust_tier: 3,
};
const azure = {
  name: "azure_blob",
  plugin_type: "source",
  description: "Read Azure blobs.",
  config_fields: [],
  usage_when_to_use: null,
  usage_when_not_to_use: null,
  example_use: null,
  capability_tags: ["azure", "blob", "network"],
  audit_characteristics: ["io_read", "external_call"],
  data_trust_tier: 3,
};

describe("CatalogDrawer — Phase 7B reshape", () => {
  beforeEach(() => {
    vi.mocked(api.listSources).mockResolvedValue([csv, azure] as never);
    vi.mocked(api.listTransforms).mockResolvedValue([] as never);
    vi.mocked(api.listSinks).mockResolvedValue([] as never);
  });

  it("renders InlineChatSourceEntry as the first row of the Sources tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("does NOT render InlineChatSourceEntry on the Transforms or Sinks tabs", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    expect(screen.queryByText(/inline data from chat/i)).not.toBeInTheDocument();
  });

  it("renders capability-tag chips derived from the loaded source list", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    // The chip strip shows tags present in the visible-tab plugins.
    expect(screen.getByRole("button", { name: /^csv$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^azure$/i })).toBeInTheDocument();
  });

  it("filtering by capability tag narrows the visible plugins", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /^csv$/i }));
    // csv plugin still visible; azure_blob filtered out (no "csv" tag).
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.queryByText("azure_blob")).not.toBeInTheDocument();
  });

  it("filtering by audit characteristic narrows the visible plugins", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: /network/i }));
    // azure has "external_call" (rendered as "Network call"); csv doesn't.
    expect(screen.getByText("azure_blob")).toBeInTheDocument();
    expect(screen.queryByText("csv")).not.toBeInTheDocument();
  });

  it("extends search across the when_to_use prose", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    const input = screen.getByPlaceholderText(/search plugins/i);
    await userEvent.type(input, "CSV file");
    // csv's usage_when_to_use mentions "CSV file"; azure's doesn't.
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.queryByText("azure_blob")).not.toBeInTheDocument();
  });

  it("extends search across capability tags", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    const input = screen.getByPlaceholderText(/search plugins/i);
    await userEvent.type(input, "blob");
    expect(screen.queryByText("csv")).not.toBeInTheDocument();
    expect(screen.getByText("azure_blob")).toBeInTheDocument();
  });

  it("filter state is per-tab — switching tabs does not carry filters over", async () => {
    // Regression guard for the cross-tab UX trap. An active capability
    // filter on Sources must NOT silently filter Transforms on tab switch.
    vi.mocked(api.listTransforms).mockResolvedValue([
      {
        name: "uppercase",
        plugin_type: "transform",
        description: "Uppercase strings.",
        config_fields: [],
        usage_when_to_use: null,
        usage_when_not_to_use: null,
        example_use: null,
        capability_tags: ["string"],
        audit_characteristics: ["deterministic"],
        data_trust_tier: 2,
      },
    ] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());

    // Activate "csv" filter on Sources tab.
    await userEvent.click(screen.getByRole("button", { name: /^csv$/i }));
    expect(screen.queryByText("azure_blob")).not.toBeInTheDocument();

    // Switch to Transforms — uppercase must be visible (its tab has no filter).
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    await waitFor(() => expect(screen.getByText("uppercase")).toBeInTheDocument());
  });

  it("shows 'No plugins match the active filters.' when filters are non-empty and match nothing", async () => {
    // B3 regression gate: empty-state message must vary by filter state.
    // Mock a source list where no plugin has the "unused_tag" capability tag.
    vi.mocked(api.listSources).mockResolvedValue([
      { ...csv, capability_tags: ["csv", "file"] },
    ] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());

    // Activate a tag that no plugin carries so the filtered list is empty.
    // Simulate by clicking a chip for "file" (present) and then clearing
    // the real plugin set to zero via an exact-name search. Because this
    // test needs the empty-state path with non-empty filters, the cleaner
    // approach is to mock a stub chip chip via searchQuery + forced filter:
    // use an audit characteristic that no plugin has.
    const input = screen.getByPlaceholderText(/search plugins/i);
    await userEvent.type(input, "zzznomatch");
    // Plugin list is now empty; filters are not active — should show the
    // search-specific empty state. Now activate a filter chip too.
    // Reset search first, then activate chip.
    await userEvent.clear(input);
    // Activate "file" chip to narrow; then mock listSources to return an
    // empty array to force the zero-result path with active filters.
    // Simpler approach: mock listSources to return empty for this test.
    vi.mocked(api.listSources).mockResolvedValue([] as never);
    // Re-render to pick up the new mock.
    const { unmount } = render(<CatalogDrawer isOpen onClose={() => {}} />);
    // In the zero-plugin scenario the drawer won't show any chips so we
    // can't click one. Instead, test the message path directly by verifying
    // the implementation in a focused variant: one plugin but all filtered.
    unmount();

    // Targeted scenario: two plugins, active filter matches only csv,
    // then we need a second filter that matches neither.
    vi.mocked(api.listSources).mockResolvedValue([csv, azure] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    // Click "file" chip (only csv has it) to filter to csv only. Then
    // also click "external_call" audit chip which only azure has — the
    // AND composition means neither matches.
    await userEvent.click(screen.getByRole("button", { name: /^file$/i }));
    await userEvent.click(screen.getByRole("button", { name: /network call/i }));
    // Plugin list is now empty with active filters.
    expect(screen.getByText("No plugins match the active filters.")).toBeInTheDocument();
    // Synthetic entry must STILL be visible — it is a pinned affordance.
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("shows 'No plugins available.' (not the filter variant) when no filters are active and the list is empty", async () => {
    // B3 regression gate: empty-state message with no active filters.
    vi.mocked(api.listSources).mockResolvedValue([] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() =>
      expect(screen.getByText("No plugins available.")).toBeInTheDocument(),
    );
    // Synthetic entry must still be visible — no filters are active and
    // the list is empty, but InlineChatSourceEntry is always pinned.
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });

  it("InlineChatSourceEntry and empty-state message are simultaneously visible when filters eliminate all real plugins", async () => {
    // B3 regression gate: the two sibling renders coexist.
    // This test is a red-green gate: if InlineChatSourceEntry is rendered
    // inside the pluginList.length === 0 conditional, one of the two
    // assertions below will fail (the synthetic entry disappears or the
    // empty state is suppressed). With the Step-3 restructure both pass.
    vi.mocked(api.listSources).mockResolvedValue([csv, azure] as never);
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByText("csv")).toBeInTheDocument());
    // Filter to a combination that eliminates all real plugins (AND logic).
    await userEvent.click(screen.getByRole("button", { name: /^file$/i }));
    await userEvent.click(screen.getByRole("button", { name: /network call/i }));
    // Both must be in the document at the same time.
    expect(screen.getByText("No plugins match the active filters.")).toBeInTheDocument();
    expect(screen.getByText(/inline data from chat/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/CatalogDrawer.test.tsx
```

Expected: FAIL — synthetic entry not present, filter chips not present,
search doesn't hit prose / tags.

- [ ] **Step 3: Wire it up**

The diff to `CatalogDrawer.tsx` is large; the conceptual shape is:

1. Add a **per-tab** `filtersByTab` state alongside `searchQuery`. Each
   tab carries its own `CatalogFilters` record; switching tabs reveals
   the filter set the user last configured for that tab, avoiding the
   "active capability filter on Sources hides every Transform" UX trap.

```typescript
function emptyFilters(): CatalogFilters {
  return {
    capabilityTags: new Set(),
    auditCharacteristics: new Set(),
  };
}

const [filtersByTab, setFiltersByTab] = useState<Record<CatalogTab, CatalogFilters>>({
  sources: emptyFilters(),
  transforms: emptyFilters(),
  sinks: emptyFilters(),
});

// Helpers scoped to the active tab so the downstream code reads as if
// there were a single filter set per render.
const filters = filtersByTab[activeTab];
const setFilters = useCallback(
  (next: CatalogFilters) => setFiltersByTab((prev) => ({ ...prev, [activeTab]: next })),
  [activeTab],
);

// Clear all tabs' filters when drawer closes (mirrors searchQuery handling).
useEffect(() => {
  if (!isOpen) {
    setFiltersByTab({ sources: emptyFilters(), transforms: emptyFilters(), sinks: emptyFilters() });
  }
}, [isOpen]);
```

2. Extend `scorePlugin` to include prose and tags:

```typescript
function scorePlugin(query: string, plugin: PluginSummary): number {
  const target = [
    plugin.name,
    plugin.description ?? "",
    plugin.usage_when_to_use ?? "",
    plugin.usage_when_not_to_use ?? "",
    plugin.capability_tags.join(" "),
  ].join(" ");
  const score = fuzzyMatch(query, target);
  if (score < 0) return -1;
  if (confidenceFromScore(score, target.length) < MIN_FUZZY_CONFIDENCE) {
    return -1;
  }
  return score;
}
```

3. Add a `matchesFilters` predicate:

```typescript
function matchesFilters(plugin: PluginSummary, filters: CatalogFilters): boolean {
  if (filters.capabilityTags.size > 0) {
    const has = plugin.capability_tags.some((t) => filters.capabilityTags.has(t));
    if (!has) return false;
  }
  if (filters.auditCharacteristics.size > 0) {
    const has = plugin.audit_characteristics.some((a) => filters.auditCharacteristics.has(a));
    if (!has) return false;
  }
  return true;
}
```

4. Compose `pluginList` with both predicates:

```typescript
const pluginList = useMemo(() => {
  const query = searchQuery.trim();
  const filtered = allPluginsForTab.filter((p) => matchesFilters(p, filters));
  if (!query) return filtered;
  return filtered
    .map((plugin) => ({ plugin, score: scorePlugin(query, plugin) }))
    .filter((item) => item.score >= 0)
    .sort((a, b) => a.score - b.score)
    .map((item) => item.plugin);
}, [allPluginsForTab, searchQuery, filters]);
```

5. Derive the available filter chip values from the current tab's plugins:

```typescript
const availableCapabilityTags = useMemo(() => {
  const set = new Set<string>();
  for (const p of allPluginsForTab) for (const t of p.capability_tags) set.add(t);
  return [...set].sort();
}, [allPluginsForTab]);

const availableAuditCharacteristics = useMemo(() => {
  const set = new Set<string>();
  for (const p of allPluginsForTab) for (const a of p.audit_characteristics) set.add(a);
  return [...set].sort();
}, [allPluginsForTab]);
```

6. Render `FilterChipStrip` between the search bar and the tab strip:

```typescript
<FilterChipStrip
  availableCapabilityTags={availableCapabilityTags}
  availableAuditCharacteristics={availableAuditCharacteristics}
  filters={filters}
  onChange={setFilters}
/>
```

7. Render `InlineChatSourceEntry` as a **pinned affordance** — outside the
   4-way plugin-list conditional ladder, unconditionally when
   `activeTab === "sources"`. The current `.catalog-list` container holds
   a `fetchError → loading → empty → pluginList.map(...)` conditional
   that controls plugin list visibility. The synthetic entry is not part
   of that ladder and must not be governed by it.

   The post-edit structure for the `.catalog-list` container:

```typescript
<div className="catalog-list">
  {/* Pinned affordance — always rendered on Sources tab regardless of
      filter state, search query, or plugin-list empty state. */}
  {activeTab === "sources" && <InlineChatSourceEntry onCloseDrawer={onClose} />}

  {/* Plugin list — governed by its own 4-way conditional. Empty-state
      applies to the plugin list, not the Sources tab as a whole. */}
  {fetchError ? (
    <div className="catalog-status-message catalog-status-message--error">
      <span>Failed to load plugin catalog.</span>
      <button
        type="button"
        className="btn btn-small"
        onClick={loadCatalog}
        aria-label="Retry loading plugin catalog"
      >
        Retry
      </button>
    </div>
  ) : isLoading || isFetching ? (
    <div role="status" aria-live="polite" className="catalog-status-message">
      Loading...
    </div>
  ) : pluginList.length === 0 ? (
    <div className="catalog-status-message catalog-status-message--center">
      {hasActiveFilters(filters)
        ? "No plugins match the active filters."
        : "No plugins available."}
    </div>
  ) : (
    pluginList.map((plugin) => { /* ... */ })
  )}
</div>
```

   Where `hasActiveFilters` is a small helper:

```typescript
function hasActiveFilters(f: CatalogFilters): boolean {
  return f.capabilityTags.size > 0 || f.auditCharacteristics.size > 0;
}
```

   **Why this structure:** when filters eliminate all real source plugins,
   the user sees BOTH the pinned `InlineChatSourceEntry` (the
   lowest-friction starting point, unaffected by filters) AND the
   empty-state message "No plugins match the active filters." — the
   correct mental model is that the plugin list is empty, not the Sources
   tab. Without this split, either the synthetic entry vanishes with the
   filtered list (wrong — breaks design intent) or the empty-state
   suppresses along with the filtered list (wrong — misleads the user
   into thinking the tab is empty when the inline affordance is still
   usable).

8. Extend the `counts` memo to apply each tab's own filters:

```typescript
const counts = useMemo(() => {
  const query = searchQuery.trim();
  const passes = (p: PluginSummary, tab: CatalogTab) =>
    matchesFilters(p, filtersByTab[tab]) && (query ? scorePlugin(query, p) >= 0 : true);
  return {
    sources: (sources ?? []).filter((p) => passes(p, "sources")).length,
    transforms: (transforms ?? []).filter((p) => passes(p, "transforms")).length,
    sinks: (sinks ?? []).filter((p) => passes(p, "sinks")).length,
  };
}, [sources, transforms, sinks, searchQuery, filtersByTab]);
```

(Filter state is **per-tab** in this design — switching from Sources to
Transforms reveals each tab's own filter set, so an active capability-
tag filter on Sources does not silently hide every plugin on Transforms.
Implemented via the `filtersByTab` record above.)

Add the imports at the top of the file:

```typescript
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";
import { InlineChatSourceEntry } from "./InlineChatSourceEntry";
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/CatalogDrawer.test.tsx
```

Expected: PASS — all new tests green, all existing tests still pass.

- [ ] **Step 5: Run the full catalog suite for regressions**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx \
  src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx
git commit -m "feat(frontend): wire filter chips + extended search + chat-source entry into catalog drawer (Phase 7C.3)"
```

## Task 4: Regroup shortcuts under "Reference" and wire Alt+1-3

**Files:**

- Modify:
  `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx`.

Per [00-implementation-roadmap.md §A](00-implementation-roadmap.md) D1:
"Keep Ctrl+Shift+P, regroup in help." The shortcut moves visually from
the flat list to a "Reference" subgroup, signalling that the catalog is
reference material rather than an action. Alt+1-3 is wired to switch
tabs in the catalog drawer (Sources / Transforms / Sinks); the binding
is drawer-scoped (the handler runs only while `isOpen` is true) so it
is harmless from the main canvas.

- [ ] **Step 1: Write the failing tests**

**Do NOT replace the existing `ShortcutsHelp.test.tsx` content.** The
existing `describe("ShortcutsHelp")` block has three regression guards:
the catalog-shortcut test (lines 7-15), the graph/YAML shortcuts test
(lines 17-31), and the retired-inspector-tab guard (lines 33-38). All
three will still pass after the regroup. Preserve them unchanged.

Append a **new** describe block at the bottom of the file:

```typescript
describe("ShortcutsHelp — Phase 7B regroup", () => {
  it("renders an 'Actions' subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: /actions/i })).toBeInTheDocument();
  });

  it("renders a 'Reference' subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    expect(screen.getByRole("heading", { name: /reference/i })).toBeInTheDocument();
  });

  it("places 'Open plugin catalog' under the Reference subheading", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const referenceHeading = screen.getByRole("heading", { name: /reference/i });
    // The catalog entry sits in the <dl> that follows the Reference heading.
    const referenceList = referenceHeading.nextElementSibling;
    expect(referenceList).not.toBeNull();
    expect(referenceList?.textContent).toMatch(/open plugin catalog/i);
  });

  it("places 'Validate pipeline' and 'Execute pipeline' under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/validate pipeline/i);
    expect(actionsList?.textContent).toMatch(/execute pipeline/i);
  });

  it("lists 'Open graph view' and 'Export YAML' under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/open graph view/i);
    expect(actionsList?.textContent).toMatch(/export yaml/i);
  });

  it("lists Alt+1-3 catalog tab shortcut under Actions", () => {
    render(<ShortcutsHelp onClose={() => {}} />);
    const actionsHeading = screen.getByRole("heading", { name: /actions/i });
    const actionsList = actionsHeading.nextElementSibling;
    expect(actionsList?.textContent).toMatch(/alt\+1-3/i);
    expect(actionsList?.textContent).toMatch(/switch.*catalog.*tab|sources.*transforms.*sinks/i);
  });
});
```

Also add to `CatalogDrawer.test.tsx` (in the existing `describe("CatalogDrawer — Phase 7B reshape")` block or as a sibling describe):

```typescript
describe("CatalogDrawer — Alt+1-3 tab shortcuts", () => {
  it("Alt+1 switches to the Sources tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    // Start on Sources (default); switch away first to make the Alt+1 test meaningful.
    await userEvent.click(screen.getByRole("tab", { name: /transforms/i }));
    expect(screen.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");

    fireEvent.keyDown(document, { key: "1", altKey: true });
    expect(screen.getByRole("tab", { name: /sources/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+2 switches to the Transforms tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    fireEvent.keyDown(document, { key: "2", altKey: true });
    expect(screen.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+3 switches to the Sinks tab", async () => {
    render(<CatalogDrawer isOpen onClose={() => {}} />);
    await waitFor(() => expect(screen.getByRole("tab", { name: /sources/i })).toBeInTheDocument());

    fireEvent.keyDown(document, { key: "3", altKey: true });
    expect(screen.getByRole("tab", { name: /sinks/i })).toHaveAttribute("aria-selected", "true");
  });

  it("Alt+1-3 has no effect when drawer is closed", () => {
    render(<CatalogDrawer isOpen={false} onClose={() => {}} />);
    // No keydown handler registered when closed — this must not throw.
    fireEvent.keyDown(document, { key: "1", altKey: true });
  });
});
```

Add `fireEvent` to the `@testing-library/react` import in `CatalogDrawer.test.tsx` if not already present.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/ShortcutsHelp.test.tsx
```

Expected: FAIL — the headings don't exist yet; current implementation
is a flat `<dl>`.

- [ ] **Step 3: Implement the regroup**

Replace
`src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`
content with:

```typescript
import { useRef } from "react";
import { useFocusTrap } from "@/hooks/useFocusTrap";

interface ShortcutsHelpProps {
  onClose: () => void;
}

interface ShortcutEntry {
  keys: string;
  action: string;
}

// Phase 7B regroup: the flat list becomes two sections so the catalog
// shortcut visually moves out of the "Actions" gravity well and into
// "Reference," signalling its reshape from interactive toolkit to
// searchable system-capability reference. Per design doc 08-§Keyboard
// shortcut placement and roadmap open question D1.
//
// Alt+1-3 is drawer-scoped: the handler lives in CatalogDrawer and runs
// only while isOpen is true. It is harmless from the main canvas.
const ACTION_SHORTCUTS: ShortcutEntry[] = [
  { keys: "Ctrl+K", action: "Command palette" },
  { keys: "Ctrl+N", action: "New session" },
  { keys: "Ctrl+/", action: "Focus chat input" },
  { keys: "Ctrl+Shift+V", action: "Validate pipeline" },
  { keys: "Ctrl+E", action: "Execute pipeline" },
  { keys: "Ctrl/Cmd+Shift+G", action: "Open graph view" },
  { keys: "Ctrl/Cmd+Shift+Y", action: "Export YAML" },
  { keys: "Alt+1-3", action: "Switch catalog tab (Sources / Transforms / Sinks)" },
  { keys: "Escape", action: "Close dialog or drawer" },
];

const REFERENCE_SHORTCUTS: ShortcutEntry[] = [
  { keys: "Ctrl/Cmd+Shift+P", action: "Open plugin catalog" },
  { keys: "?", action: "Keyboard shortcuts" },
];

function ShortcutList({ shortcuts }: { shortcuts: ShortcutEntry[] }) {
  return (
    <dl className="shortcuts-list">
      {shortcuts.map(({ keys, action }) => (
        <div key={keys} className="shortcuts-list-item">
          <dt>
            <kbd className="command-palette-kbd">{keys}</kbd>
          </dt>
          <dd>{action}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ShortcutsHelp({ onClose }: ShortcutsHelpProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useFocusTrap(dialogRef);

  return (
    <>
      <div
        className="confirm-dialog-backdrop"
        onClick={onClose}
        role="presentation"
      />
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Keyboard shortcuts"
        className="confirm-dialog"
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            onClose();
          }
        }}
      >
        <h2 className="confirm-dialog-title">Keyboard Shortcuts</h2>
        <h3 className="shortcuts-subheading">Actions</h3>
        <ShortcutList shortcuts={ACTION_SHORTCUTS} />
        <h3 className="shortcuts-subheading">Reference</h3>
        <ShortcutList shortcuts={REFERENCE_SHORTCUTS} />
        <div className="confirm-dialog-actions">
          <button onClick={onClose} className="btn confirm-dialog-btn">
            Close
          </button>
        </div>
      </div>
    </>
  );
}
```

- [ ] **Step 3b: Wire Alt+1-3 into `CatalogDrawer.tsx`**

Add to the existing keydown effect in `CatalogDrawer.tsx` (the effect at
lines 109-129 of the current source that handles Escape and `/`):

```typescript
// Alt+1 / Alt+2 / Alt+3: Switch catalog tab while drawer is open.
// Binding is drawer-scoped — the handler only runs when isOpen is true.
// Does NOT dispatch a custom event; calls setActiveTab directly so
// App.test.tsx's "does not dispatch retired inspector tab shortcuts on
// Alt+digit" regression guard continues to pass unchanged.
if (e.altKey && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
  if (e.key === "1") { e.preventDefault(); setActiveTab("sources"); return; }
  if (e.key === "2") { e.preventDefault(); setActiveTab("transforms"); return; }
  if (e.key === "3") { e.preventDefault(); setActiveTab("sinks"); return; }
}
```

Place this block inside the existing `handleKeyDown` function, after the
`"/"` focus-search branch and before the closing `}` of `handleKeyDown`.
No new imports are required.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/ShortcutsHelp.test.tsx
```

Expected: PASS — the two existing describe blocks (9 tests total: 3
original + 6 new regroup tests) all green. The retired-inspector-tab
guard (`does not list retired inspector tab shortcuts`) still passes
because `ShortcutsHelp.tsx` does not mention `Alt+1-2` or `Switch
inspector tab`.

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/CatalogDrawer.test.tsx
```

Expected: PASS — all CatalogDrawer tests green including the 4 new
Alt+1-3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx \
  src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx \
  src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.tsx \
  src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx
git commit -m "feat(frontend): regroup shortcuts + wire Alt+1-3 catalog tab switch (Phase 7C.4)"
```

## Task 5: Final regression sweep

- [ ] **Step 1: Run the full frontend test suite**

```bash
cd src/elspeth/web/frontend && npx vitest run
```

Expected: PASS — all suites green. If a previously-passing test now
fails, diagnose before declaring this phase done; per CLAUDE.md "Fix
errors you encounter," do not skip-or-suppress.

- [ ] **Step 2: Run tsc for a final type-check**

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit
```

Expected: PASS.

- [ ] **Step 3: Run the frontend linter if configured**

```bash
cd src/elspeth/web/frontend && npx eslint src/components/catalog/ src/components/common/ShortcutsHelp.tsx
```

Expected: PASS.

- [ ] **Step 4: Manually exercise the drawer in staging (secondary gate)**

The Playwright spec in Task 6 is the **primary gate** for the catalog
reshape. This manual smoke is a secondary gate — run it to catch
environment-specific failures (service restart, CSS regression, network
auth) that the spec harness doesn't exercise. If Task 6's spec passes
and this smoke reveals a regression, diagnose and fix before declaring
done.

Per `project_staging_deployment`, the staging deploy at
`elspeth.foundryside.dev` is a source-checkout systemd/Caddy deploy on
this machine.

```bash
cd src/elspeth/web/frontend && npm run build
sudo systemctl restart elspeth-web.service
```

Then visit `elspeth.foundryside.dev` and:

1. Open the catalog drawer with `Ctrl+Shift+P`.
2. Verify the "Inline data from chat" entry sits at the top of Sources.
3. Click "Try it" on the inline entry — the drawer closes and the chat
   input is prefilled with the suggested prompt.
4. Verify the CSV plugin card shows the canonical Phase-7A reference
   content: audit icons for io_read / quarantine / coerce, and the
   "When you'd use this" / "When you wouldn't" prose.
5. Verify other source plugins (azure_blob, json, text, dataverse)
   render with the fallback "see the technical description" message
   since their prose hasn't been authored yet.
6. Click a capability-tag chip (e.g., "csv") — the visible plugin list
   narrows.
7. Click an audit-characteristic chip (e.g., "Network call", filter id `"external_call"`) — the visible
   list narrows again (or shows the union/intersection of conditions
   per design).
8. Click "Clear filters" — both chips deactivate.
9. Type "URL" in the search bar — verify hits include any plugin whose
   prose mentions URL (the synthetic entry's suggested prompt has a URL,
   so it should appear; the chat source isn't fuzzy-scored — it always
   shows on the Sources tab — so this only proves prose-hit on real
   plugins).
10. Press `Alt+2` while the drawer is open — verify the Transforms tab
    activates. Press `Alt+3` — Sinks tab activates. Press `Alt+1` —
    Sources tab activates.
11. Press `?` to open the shortcuts dialog — verify the "Actions" /
    "Reference" subheadings render, the catalog shortcut is under
    "Reference," and `Alt+1-3` appears under "Actions."

- [ ] **Step 5: Document any manual issues found**

If the manual run surfaces a regression, fix it before declaring the
phase done. Do not file an observation and close the task; per
CLAUDE.md "Observations: when (and when not) to use them," items in
scope must be handled in-scope.


---

## Task 6: Playwright E2E gate + un-fixme llm-provider-schema spec

**Files:**

- New:
  `src/elspeth/web/frontend/tests/e2e/catalog-reshape.spec.ts`.
- Modify:
  `src/elspeth/web/frontend/tests/e2e/llm-provider-schema.spec.ts`.

Task 5's manual smoke is the secondary gate. This task is the **primary
gate**: an automated Playwright spec that covers the demo-critical path
and a standing un-fixme of the `llm-provider-schema` spec whose tested
surface is directly reshaped by Phase 7A/7B/7C.

> Path note: the brief specified `src/elspeth/web/frontend/e2e/` but the
> actual repo convention is `tests/e2e/` (every existing Playwright spec
> lives there). The spec is placed at `tests/e2e/catalog-reshape.spec.ts`
> to match the project convention.

### Step 1: Author `catalog-reshape.spec.ts`

```typescript
// catalog-reshape.spec.ts — E2E demo-path gate for Phase 7A/7B/7C.
//
// Covers the catalog drawer as reshaped by the Phase 7 work:
//   - InlineChatSourceEntry always visible on Sources tab
//   - FilterChipStrip narrows the plugin list
//   - PluginCard renders reference content (no toolkit affordances)
//   - Alt+1-3 tab switching
//   - Clear-filters restores full list
//
// This spec is the PRIMARY gate for the catalog reshape demo path.
// Task 5 manual smoke is a secondary gate. If this spec is fixme'd in
// CI (e.g., during a flake spike), that is a demo-day risk: escalate
// rather than silently deferring.
//
// Playwright invocation:
//   cd src/elspeth/web/frontend && npx playwright test tests/e2e/catalog-reshape.spec.ts

import { expect, test } from "@playwright/test";
import { ComposerPage } from "./page-objects/composer-page";

test.describe("catalog-reshape — Phase 7 demo path", () => {
  test("1: Open catalog drawer via Ctrl+Shift+P", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();
  });

  test("2: Drawer opens with Sources tab active and InlineChatSourceEntry visible", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    // Sources tab active by default.
    await expect(page.getByRole("tab", { name: /sources/i })).toHaveAttribute("aria-selected", "true");
    // Synthetic entry is a pinned affordance — always visible on Sources.
    await expect(page.getByText(/inline data from chat/i)).toBeVisible();
  });

  test("3: Click a FilterChipStrip chip — plugin list narrows", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    // Wait for plugins to load.
    await expect(page.getByRole("button", { name: /^csv$/i })).toBeVisible();
    const initialCount = await page.locator(".plugin-card").count();
    await page.getByRole("button", { name: /^csv$/i }).click();
    const filteredCount = await page.locator(".plugin-card").count();
    expect(filteredCount).toBeLessThan(initialCount);
  });

  test("4: Click InlineChatSourceEntry 'Try it' — drawer closes and chat is prefilled", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByText(/inline data from chat/i)).toBeVisible();
    await page.getByRole("button", { name: /try it/i }).first().click();
    // Drawer closes.
    await expect(page.getByRole("dialog", { name: /plugin catalog/i })).not.toBeVisible();
    // Chat input is prefilled with a non-empty string.
    const chatInput = page.getByRole("textbox", { name: /chat input/i });
    const value = await chatInput.inputValue();
    expect(value.trim().length).toBeGreaterThan(0);
  });

  test("5: Switch to Transforms tab via Alt+2", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();
    await page.keyboard.press("Alt+2");
    await expect(page.getByRole("tab", { name: /transforms/i })).toHaveAttribute("aria-selected", "true");
  });

  test("6: Clear filters button restores full plugin list", async ({ page }) => {
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    await expect(page.getByRole("button", { name: /^csv$/i })).toBeVisible();
    await page.getByRole("button", { name: /^csv$/i }).click();
    const filteredCount = await page.locator(".plugin-card").count();
    await page.getByRole("button", { name: /clear filters/i }).click();
    const restoredCount = await page.locator(".plugin-card").count();
    expect(restoredCount).toBeGreaterThan(filteredCount);
  });

  test("7: PluginCard shows reference content — no toolkit affordance (OD-C regression gate)", async ({ page }) => {
    // Per OD-C: PluginCard is reference-only. The "Use in pipeline" button
    // and TrustTierBadge are removed. "When you'd use this" prose must be
    // present on the CSV source card (which has authored prose from Phase 7A).
    const composer = new ComposerPage(page);
    await composer.goto();
    await page.keyboard.press("Control+Shift+P");
    // Click the CSV source card to expand it.
    await page.getByText("csv").first().click();
    // "When you'd use this" prose section must be visible.
    await expect(page.getByText(/when you.d use this/i)).toBeVisible();
    // "Use in pipeline" button must NOT be present.
    await expect(page.getByRole("button", { name: /use in pipeline/i })).not.toBeVisible();
    // TrustTierBadge must NOT be present (trust tier is internal metadata, not surfaced).
    await expect(page.getByTestId("trust-tier-badge")).not.toBeVisible();
  });
});
```

- [ ] **Step 2: Run the spec**

```bash
cd src/elspeth/web/frontend && npx playwright test tests/e2e/catalog-reshape.spec.ts
```

Expected on first run: some tests may require the staging server to be
running (see Task 5 Step 4 for the `npm run build` + `systemctl restart`
sequence). All 7 tests must pass before declaring Phase 7C complete. If
any test fails, diagnose and fix before closing the task.

- [ ] **Step 3: Un-fixme and author `llm-provider-schema.spec.ts`**

Open `src/elspeth/web/frontend/tests/e2e/llm-provider-schema.spec.ts`.
The current state: a single `test.fixme(true, "...")` gate wraps the
entire `describe` block (line 21). The two `test()` bodies inside are
empty stubs (only comments; no assertions).

This is a two-part step:

**Part a — Remove the `test.fixme` gate.** Delete lines 21–25
(`test.fixme(true, "Expected to fail...")`) entirely. The describe block
now runs both tests on every CI pass.

**Part b — Author the test bodies.** The Phase 7A/7B/7C reshape has
changed the catalog card layout (new aria labels, no "Use in pipeline"
button, rewritten schema-preview surface). The existing test bodies are
empty, so there is nothing to "update for new layout" — the bodies need
to be written from scratch against the post-reshape card.

The existing comments in the file describe the intent:

```typescript
test("llm transform schema enumerates Azure and OpenRouter variants", async ({ page }) => {
  // 1. Open catalog drawer.
  await page.keyboard.press("Control+Shift+P");
  await expect(page.getByRole("dialog", { name: /plugin catalog/i })).toBeVisible();

  // 2. Switch to Transforms tab and click the llm transform card.
  await page.getByRole("tab", { name: /transforms/i }).click();
  await page.getByText("llm").first().click();

  // 3. Assert the schema preview surfaces Azure-specific field (deployment_name)
  //    and OpenRouter-specific field (api_key).
  //    These fields are only present once elspeth-dcf12c061b is fixed.
  await expect(page.getByText(/deployment_name/i)).toBeVisible();
  await expect(page.getByText(/api_key/i)).toBeVisible();
});

test("llm node without provider fields surfaces a Stage-1 error", async ({ page }) => {
  // 1. Open catalog drawer, navigate to llm transform.
  await page.keyboard.press("Control+Shift+P");
  await page.getByRole("tab", { name: /transforms/i }).click();
  await page.getByText("llm").first().click();

  // 2. Add the llm node to the pipeline without configuring a provider.
  //    "Use in pipeline" was removed per OD-C; if no successor affordance
  //    exists yet, the test cannot perform this setup step. The skip
  //    below makes the gap CI-visible rather than producing a vacuous
  //    pass on step 3. Remove this line and add the real add-node
  //    sequence once the affordance lands.
  test.skip(!hasStateSeed, "state-seed gap — see elspeth-dcf12c061b");

  // 3. Assert the validation dot reads "Validation failed".
  await expect(page.getByText(/validation failed/i)).toBeVisible();

  // 4. Assert validation banner mentions a provider-specific field.
  await expect(page.getByText(/deployment_name|provider/i)).toBeVisible();
});
```

Run after authoring:

```bash
cd src/elspeth/web/frontend && npx playwright test tests/e2e/llm-provider-schema.spec.ts
```

Expected: some tests may fail against the current backend if
`elspeth-dcf12c061b` is still open (the provider-union bug). If they
fail for that reason, do NOT re-add `test.fixme`. Instead, convert to
`test.skip` with a condition tied to a feature flag or bug ID, so the
failure is visible in CI output rather than silently suppressed. Per
CLAUDE.md No-Legacy: no `// removed for X` placeholder comments; document
the skip condition inline.

If the test bodies require material rewrites beyond the scaffolding above
(e.g., the add-node affordance doesn't exist yet), treat this as a
separate red-green sub-task: name it in a Filigree observation, link it
to `elspeth-dcf12c061b`, and leave the test body with a `test.skip`
condition as described above.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/tests/e2e/catalog-reshape.spec.ts \
  src/elspeth/web/frontend/tests/e2e/llm-provider-schema.spec.ts
git commit -m "test(e2e): catalog-reshape demo-path spec + un-fixme llm-provider-schema (Phase 7C.6)"
```

---

## What Phase 7C leaves the frontend in

After all six tasks land (and Phase 7B is merged):

- The catalog drawer is reshaped into a reference surface: persona-
  facing prose (from 7B's `PluginCard` rewrite), audit-characteristic
  icons, filter chips, and the synthetic "Inline data from chat" entry
  at the top of Sources.
- The keyboard shortcut for the catalog is regrouped under "Reference"
  in the shortcuts help, signalling the reshape to users who consult
  the help.
- Fuzzy search hits the new prose and capability tags; filter chips
  let users narrow the catalog along two orthogonal dimensions
  (capability tag, audit characteristic). Trust tier is not surfaced
  as a filter dimension — it is kind-derived internal metadata.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Filter chips proliferate as capability tags multiply | Filter strip is rendered from the *visible-tab plugin set*, not from a hardcoded list. As authors add tags to their plugins, chips appear automatically; if the set grows unwieldy, Phase 8 can add a "more..." overflow. |
| Filters across tabs surprise users (active capability filter on Sources hides plugins on Transforms) | Resolved: per-tab `filtersByTab` record (Task 3 Step 3). Each tab carries its own filter set; switching tabs reveals that tab's state, never a stale cross-tab filter. |
| Fuzzy search across long prose strings becomes noisy | The existing `confidenceFromScore` noise floor still applies. Long target strings naturally raise the noise floor; this is the existing behaviour, just applied to a longer target. |
| Synthetic entry confuses users — "is this a plugin?" | The entry's distinct visual style (`inline-chat-source-entry` class with a "no plugin needed" badge per Task 2's implementation) and explanatory description per design doc 08-§Risks. |
| Phase 7B not yet shipped | This plan imports from 7B's primitives (`PluginCard`, `AuditCharacteristicIcon`, `auditCharacteristics`). 7C cannot land before 7B. If both ship together as a stacked PR, the sequence is 7B → 7C; verify by running `npx vitest run src/components/catalog/` between merges. |
| Staging smoke reveals a regression in chat-input prefill | `ChatInput.tsx` still imports `PREFILL_CHAT_INPUT_EVENT` from `@/components/catalog/PluginCard`; the export survived 7B's rewrite. The synthetic-entry dispatch uses the same path, so a regression here would surface in unit tests, not just the smoke. |

## Memory references

- `feedback_catalog_is_reference_not_toolkit` — the design call this implements.
- `project_composer_personas` — informs the persona-facing reads.
- `project_composer_dynamic_source_from_chat` — the inline-data entry
  added in Task 2 reifies this memory in the catalog UI.
- `project_staging_deployment` — informs Task 5 Step 4's manual smoke.
- `feedback_no_calendar_shipping_commitments` — no calendar commitments
  in this plan.

## Review history

- 2026-05-15 reality-review: Changed `audit_characteristics` fixture value from `"network"` to `"external_call"` (capability_tags `"network"` left unchanged); updated comment and smoke-test prose to reflect the `"external_call"` filter id / "Network call" display label distinction.
- 2026-05-16 reality-review (rev-2): Switched filter state from shared-across-tabs to **per-tab** (`filtersByTab` record keyed by `CatalogTab`). The previous design shipped a known UX trap — "Tier 3 active on Sources hides every plugin on Transforms because they're Tier 2" — flagged in the original Task 3 narrative but not resolved. Resolution lands in-scope per CLAUDE.md "default is fix not ticket": added a regression test in `CatalogDrawer.test.tsx` exercising tab-switch isolation; updated risk table.
- 2026-05-18 Round-3 fixes (B2 ShortcutsHelp + Alt+1-3 wiring + B1 preflight + OD-C trust-tier chip removal): (B2 — dropped shortcuts) Restored `Ctrl/Cmd+Shift+G` / "Open graph view" and `Ctrl/Cmd+Shift+Y` / "Export YAML" to `ACTION_SHORTCUTS`; both are present in the current `ShortcutsHelp.tsx:11-12` and wired in `App.tsx:154-173`. (B2 — Alt+1-4 → Alt+1-3 wired) CatalogDrawer has exactly 3 tabs (Sources / Transforms / Sinks) — Alt+4 has no mapping. Binding target is the drawer's `handleKeyDown` effect (gated on `isOpen`), calling `setActiveTab` locally with no custom event; App.test.tsx's `"does not dispatch retired inspector tab shortcuts on Alt+digit"` guard continues to pass because that test asserts `elspeth-switch-tab` is NOT fired, which remains true. Help text updated to `Alt+1-3` with action "Switch catalog tab (Sources / Transforms / Sinks)". Task 4 Modified-files list expanded to include `CatalogDrawer.tsx` and `CatalogDrawer.test.tsx`; Step 3b added for the keydown wiring; Step 4 and Step 5 updated. Task 4 Step 1 changed from "replace file content" to "add a new describe block" — preserving the existing 3-test `describe("ShortcutsHelp")` regression guards; new describe adds 6 tests including graph/YAML presence and Alt+1-3 presence. (B1 — preflight) Added Step 0 before Task 1 Step 1: three shell checks asserting `auditCharacteristics.ts`, `AuditCharacteristicIcon.tsx`, and `capability_tags` in `types/index.ts` are present; hard-exits with an explicit message if any check fails. (OD-C) Removed trust-tier filter dimension throughout: dropped `trustTiers` from `CatalogFilters` interface, `emptyFilters()`, and `matchesFilters()`; dropped `availableTrustTiers` memo; dropped `availableTrustTiers` prop from `FilterChipStrip` and its render site; dropped trust-tier test from Task 1 Step 1; dropped `DataTrustTier` import from Step 3; updated intro, Scope, "Out of scope" prose, "What leaves", Risks table, and Task 5 smoke step 4 to remove all trust-tier references. Rationale: trust tier is kind-derived internal metadata per 16b OD-C; there is nothing per-plugin to filter on.
- 2026-05-18 Round-4 fixes (B3 synthetic-entry cordoning + B5 Playwright spec + B4 un-fixme + R3 carry-overs): (B3) Restructured Task 3 Step 3 item 7 so `InlineChatSourceEntry` is rendered as a pinned affordance unconditionally when `activeTab === "sources"`, outside the 4-way `pluginList`-length conditional ladder. Empty-state message applies to the plugin list only, not the Sources tab as a whole. Added two new tests in Task 3 Step 1: empty-state message uses "No plugins match the active filters." when filters are non-empty vs "No plugins available." when filters are empty; synthetic entry and empty-state message render simultaneously when filters eliminate all real plugins. Empty-state message conditional updated in Step 3 to branch on filter state. (B5) Added Task 6 with a Playwright spec at `tests/e2e/catalog-reshape.spec.ts` covering 7 demo-path assertions; Task 5 Step 4 manual smoke reframed as secondary gate pointing to Task 6 spec as primary. (B4 / OD-A) Added Task 6 Step 3 to un-fixme and author `tests/e2e/llm-provider-schema.spec.ts`; named the two coupled sub-steps (remove the `test.fixme(true, ...)` gate; author the four test bodies); called out red-green sub-task risk if bodies need material work. (R3 carry-overs) Fixed Task 4 Step 4 test count from "7 tests total: 3 original + 4 new" to "9 tests total: 3 original + 6 new"; fixed FilterChipStrip block comment from "Three groups" to "Two groups (capability tags and audit characteristics)".

- 2026-05-18 Worktree batch protocol added: Added the `## Implementation worktree (batched with [siblings])` section near the top of this plan documenting the shared `.worktrees/phase-7-catalog` worktree, the execution order in the batch, the operator-known gotchas (venv leak / Python 3.13 / subagent CWD / filigree CLI), and the single-PR shipping shape. See `16-phase-7-catalog-reshape.md` for the canonical batch protocol.
- **2026-05-18 implementation complete**: All 6 tasks landed on `feat/phase-7-catalog`. 5 commits on top of 16b: `e89cbc8b8` (Task 1 FilterChipStrip), `c32f2bc59` (Task 2 InlineChatSourceEntry), `7e726bde9` (Task 3 drawer wire-up: per-tab filter state + extended search + synthetic entry), `0b3ce3dc5` (Task 4 shortcuts regroup + Alt+1-3 + allPluginsForTab memo carry-forward), `757c9701b` (Task 6 Playwright catalog-reshape.spec.ts + un-fixme llm-provider-schema). Task 5 was verification-only (no commit). Per-plan PR-toolkit review verdict: zero BLOCKERs, zero unaddressed MAJORs. Code review approved with zero issues at any severity. Test analysis approved with two Important Improvements (I-1: drawer-closed test only asserts non-throw rather than non-registration; I-2: Clear-filters payload shape not pinned at unit level) — both have mitigating coverage at higher pyramid levels (App.test.tsx retirement guard for I-1; Playwright Clear-filters E2E for I-2), so non-blocking. Two Playwright deviations from plan-verbatim documented in the spec inline: seeded session via REST helper (chat input not mounted on empty landing), and aria-label regex fix from `/chat input/i` to `/message input/i` (plan text was wrong; spec aligned to live aria-label). OD-A honoured: `test.skip(!flag, "bug-ID gap")` form, NOT test.fixme, NOT placeholder comments — both tests in llm-provider-schema.spec.ts gated on module-level `hasProviderUnionSchema` / `hasStateSeed` constants referencing `elspeth-dcf12c061b`; filigree observation `elspeth-obs-6aff645d4f` filed linking the skips to the bug. Final frontend test count: 787/787 vitest + 7/7 Playwright catalog-reshape + 2/2 Playwright llm-provider-schema (skipped cleanly). `npm run typecheck` clean. 1 pre-existing eslint warning (CatalogDrawer.tsx:260 schemaErrors dep) confirmed not 7C-introduced. Manual staging smoke (Task 5 Step 4) deferred to operator — requires `sudo systemctl restart elspeth-web.service` which is outside subagent territory.
