# Phase 7C — Frontend: Catalog drawer integration (filters, synthetic entry, shortcuts)

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
`PluginCard`, the `TrustTierBadge`, the `AuditCharacteristicIcon`, the
`auditCharacteristics` metadata table). Adds three filter dimensions
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
  `AuditCharacteristicIcon`, `TrustTierBadge`, the `PluginCard` rewrite.

**Roadmap reference:**
[00-implementation-roadmap.md](00-implementation-roadmap.md). Design doc:
[08-catalog-reshape.md](08-catalog-reshape.md).

---

## Scope boundaries

**In scope:**

- New `FilterChipStrip.tsx` component with three filter groups:
  capability tag, trust tier, audit characteristic. Each is a multi-
  select pill group; empty selection = "all"; filters compose with each
  other and with search via AND.
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
  tab has its own `CatalogFilters` record so an active "Tier 3" filter
  on Sources does not hide every Transform on tab switch.)
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

The filter strip has three groups: capability tags, trust tier, audit
characteristic. Each is a multi-select pill group; an empty selection
means "all." Filters compose with each other and with search via AND.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";

const ALL_OFF: CatalogFilters = {
  capabilityTags: new Set(),
  trustTiers: new Set(),
  auditCharacteristics: new Set(),
};

describe("FilterChipStrip", () => {
  it("renders one chip per capability tag", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv", "file", "http"]}
        availableTrustTiers={[]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /csv/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^http/i })).toBeInTheDocument();
  });

  it("renders trust-tier chips with friendly labels", () => {
    render(
      <FilterChipStrip
        availableCapabilityTags={[]}
        availableTrustTiers={[1, 2, 3]}
        availableAuditCharacteristics={[]}
        filters={ALL_OFF}
        onChange={() => {}}
      />,
    );
    expect(screen.getByRole("button", { name: /tier 1/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /tier 3/i })).toBeInTheDocument();
  });

  it("emits an updated filter set when a chip is toggled", async () => {
    const onChange = vi.fn();
    render(
      <FilterChipStrip
        availableCapabilityTags={["csv"]}
        availableTrustTiers={[]}
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
        availableTrustTiers={[]}
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
        availableTrustTiers={[]}
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
        availableTrustTiers={[]}
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
// Three groups of filter chips at the top of each catalog tab. Filters
// compose with each other and with search via AND: a plugin must match
// the search query AND have a tag in every active group it appears in.
// Empty group = "all."
//
// Per design doc 08-§Filters: "The filter strip lets users narrow the
// catalog to 'what works for my sensitive-data pipeline' or 'what
// doesn't make a network call' in one click."
// ============================================================================

import { useCallback } from "react";
import type { DataTrustTier } from "@/types/index";
import { lookupAuditCharacteristic } from "./auditCharacteristics";

export interface CatalogFilters {
  capabilityTags: Set<string>;
  trustTiers: Set<DataTrustTier>;
  auditCharacteristics: Set<string>;
}

interface FilterChipStripProps {
  availableCapabilityTags: string[];
  availableTrustTiers: DataTrustTier[];
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
  availableTrustTiers,
  availableAuditCharacteristics,
  filters,
  onChange,
}: FilterChipStripProps) {
  const anyActive =
    filters.capabilityTags.size > 0 ||
    filters.trustTiers.size > 0 ||
    filters.auditCharacteristics.size > 0;

  const toggleTag = useCallback(
    (tag: string) => onChange({ ...filters, capabilityTags: toggle(filters.capabilityTags, tag) }),
    [filters, onChange],
  );
  const toggleTier = useCallback(
    (tier: DataTrustTier) => onChange({ ...filters, trustTiers: toggle(filters.trustTiers, tier) }),
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
        trustTiers: new Set(),
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
      {availableTrustTiers.length > 0 && (
        <ChipGroup label="Trust tier">
          {availableTrustTiers.map((tier) => (
            <Chip
              key={tier}
              active={filters.trustTiers.has(tier)}
              onToggle={() => toggleTier(tier)}
              label={`Tier ${tier}`}
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

Expected: PASS — all six tests green.

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
   "Tier 3 active on Sources hides every Transform" UX trap.

```typescript
function emptyFilters(): CatalogFilters {
  return {
    capabilityTags: new Set(),
    trustTiers: new Set(),
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
  if (filters.trustTiers.size > 0) {
    if (plugin.data_trust_tier === null) return false;
    if (!filters.trustTiers.has(plugin.data_trust_tier)) return false;
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

const availableTrustTiers = useMemo(() => {
  const set = new Set<DataTrustTier>();
  for (const p of allPluginsForTab) {
    if (p.data_trust_tier !== null) set.add(p.data_trust_tier);
  }
  return [...set].sort((a, b) => a - b);
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
  availableTrustTiers={availableTrustTiers}
  availableAuditCharacteristics={availableAuditCharacteristics}
  filters={filters}
  onChange={setFilters}
/>
```

7. Render `InlineChatSourceEntry` at the top of the Sources list. Place
   it inside the same `.catalog-list` container, before the `pluginList.map(...)`:

```typescript
{activeTab === "sources" && <InlineChatSourceEntry onCloseDrawer={onClose} />}
{pluginList.map((plugin) => { /* ... */ })}
```

The synthetic entry is unaffected by `filters` or `searchQuery` — it's
always the first source row, by design (lowest-friction starting point).

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
Transforms reveals each tab's own filter set, so a "tier 3" filter on
Sources does not silently hide every plugin on Transforms. Implemented
via the `filtersByTab` record above.)

Add the imports at the top of the file:

```typescript
import { FilterChipStrip, type CatalogFilters } from "./FilterChipStrip";
import { InlineChatSourceEntry } from "./InlineChatSourceEntry";
import type { DataTrustTier } from "@/types/index";
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

## Task 4: Regroup shortcuts under "Reference"

**Files:**

- Modify:
  `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`.

Per [00-implementation-roadmap.md §A](00-implementation-roadmap.md) D1:
"Keep Ctrl+Shift+P, regroup in help." The shortcut moves visually from
the flat list to a "Reference" subgroup, signalling that the catalog is
reference material rather than an action.

- [ ] **Step 1: Write the failing test**

Replace
`src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx`
content (preserving its imports) with:

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ShortcutsHelp } from "./ShortcutsHelp";

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
});
```

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
const ACTION_SHORTCUTS: ShortcutEntry[] = [
  { keys: "Ctrl+K", action: "Command palette" },
  { keys: "Ctrl+N", action: "New session" },
  { keys: "Ctrl+/", action: "Focus chat input" },
  { keys: "Ctrl+Shift+V", action: "Validate pipeline" },
  { keys: "Ctrl+E", action: "Execute pipeline" },
  { keys: "Alt+1-4", action: "Switch inspector tab (Spec/Graph/YAML/Runs)" },
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

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/common/ShortcutsHelp.test.tsx
```

Expected: PASS — all four tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/common/ShortcutsHelp.tsx \
  src/elspeth/web/frontend/src/components/common/ShortcutsHelp.test.tsx
git commit -m "feat(frontend): regroup shortcuts under Actions / Reference (Phase 7C.4)"
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

- [ ] **Step 4: Manually exercise the drawer in staging**

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
   content: trust-tier badge "Tier 3", audit icons for io_read /
   quarantine / coerce, and the "When you'd use this" / "When you
   wouldn't" prose.
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
10. Press `?` to open the shortcuts dialog — verify the "Actions" /
    "Reference" subheadings render and the catalog shortcut is under
    "Reference."

- [ ] **Step 5: Document any manual issues found**

If the manual run surfaces a regression, fix it before declaring the
phase done. Do not file an observation and close the task; per
CLAUDE.md "Observations: when (and when not) to use them," items in
scope must be handled in-scope.


---

## What Phase 7C leaves the frontend in

After all five tasks land (and Phase 7B is merged):

- The catalog drawer is reshaped into a reference surface: persona-
  facing prose (from 7B's `PluginCard` rewrite), trust-tier badges,
  audit-characteristic icons, filter chips, and the synthetic "Inline
  data from chat" entry at the top of Sources.
- The keyboard shortcut for the catalog is regrouped under "Reference"
  in the shortcuts help, signalling the reshape to users who consult
  the help.
- Fuzzy search hits the new prose and capability tags; filter chips
  let users narrow the catalog along three orthogonal dimensions
  (capability, trust tier, audit characteristic).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Filter chips proliferate as capability tags multiply | Filter strip is rendered from the *visible-tab plugin set*, not from a hardcoded list. As authors add tags to their plugins, chips appear automatically; if the set grows unwieldy, Phase 8 can add a "more..." overflow. |
| Filters across tabs surprise users (active "tier 3" filter on Sources hides every plugin on Transforms because they're Tier 2) | Resolved: per-tab `filtersByTab` record (Task 3 Step 3). Each tab carries its own filter set; switching tabs reveals that tab's state, never a stale cross-tab filter. |
| Fuzzy search across long prose strings becomes noisy | The existing `confidenceFromScore` noise floor still applies. Long target strings naturally raise the noise floor; this is the existing behaviour, just applied to a longer target. |
| Synthetic entry confuses users — "is this a plugin?" | The entry's distinct visual style (`inline-chat-source-entry` class with a "no plugin needed" badge per Task 2's implementation) and explanatory description per design doc 08-§Risks. |
| Phase 7B not yet shipped | This plan imports from 7B's primitives (`PluginCard`, `TrustTierBadge`, `AuditCharacteristicIcon`, `auditCharacteristics`). 7C cannot land before 7B. If both ship together as a stacked PR, the sequence is 7B → 7C; verify by running `npx vitest run src/components/catalog/` between merges. |
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
