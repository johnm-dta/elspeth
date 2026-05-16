# Phase 7B — Frontend: Catalog primitives + PluginCard rewrite

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps
> use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the frontend primitives for Phase 7's catalog reshape —
extend `PluginSummary` types with the fields shipped by Phase 7A, build
the small leaf components (`AuditCharacteristicIcon`, `TrustTierBadge`)
and their centralised metadata table (`auditCharacteristics.ts`), and
rewrite `PluginCard.tsx` to render the new reference-card layout from
design doc 08 (removing the "Use in pipeline" toolkit affordance in the
process). Drawer integration (filter chips, synthetic chat-source
entry, shortcuts regroup) is in the companion plan,
[16c-phase-7c-frontend-integration.md](16c-phase-7c-frontend-integration.md).

**Architecture:** Frontend-only. The catalog API surface already returns
the new fields after Phase 7A; this plan consumes them, falling back to
graceful defaults for unfilled plugins. The five tasks here build the
**primitives** that 16c then composes inside the drawer. The
`PREFILL_CHAT_INPUT_EVENT` export on `PluginCard.tsx` survives the
rewrite because 16c's `InlineChatSourceEntry` dispatches it and
`ChatInput.tsx` already listens.

**Tech Stack:** React + TypeScript + Vitest + testing-library.

**Sibling plans:**

- [16a-phase-7a-backend.md](16a-phase-7a-backend.md) — backend metadata
  schema, catalog API extension, audit-characteristic derivation,
  canonical CSV example.
- [16c-phase-7c-frontend-integration.md](16c-phase-7c-frontend-integration.md)
  — filter chips, synthetic chat-source entry, drawer wire-up,
  shortcuts regroup, staging smoke.

**Roadmap reference:**
[00-implementation-roadmap.md](00-implementation-roadmap.md). Design doc:
[08-catalog-reshape.md](08-catalog-reshape.md).

---

## Scope boundaries

**In scope:**

- Extend `PluginSummary` interface in `types/index.ts` with the six new
  optional fields shipped by 7A (`usage_when_to_use`,
  `usage_when_not_to_use`, `example_use`, `capability_tags`,
  `audit_characteristics`, `data_trust_tier`). Add the
  `InlineChatSourceEntry` type (consumed by 16c).
- New `auditCharacteristics.ts` metadata table mapping each
  audit-characteristic flag string to display metadata (label, glyph,
  tooltip, tone).
- New `AuditCharacteristicIcon.tsx` — single-flag renderer used by the
  card and (in 16c) the filter chip strip.
- New `TrustTierBadge.tsx` — tier-badge renderer.
- Rewrite `PluginCard.tsx` to render the new card layout from design
  doc 08-§"Plugin card content design" (trust-tier badge,
  audit-characteristic icons, "When you'd use this" / "When you
  wouldn't" / "Example use" / "Schema →" disclosure).
- Remove the "Use in pipeline" button and supporting machinery
  (`buildInsertionPrompt`, `handleUseInPipeline`, the `onCloseDrawer`
  prop on `PluginCard`). The `PREFILL_CHAT_INPUT_EVENT` **export**
  stays — `ChatInput.tsx` listens for it, and 16c's `InlineChatSourceEntry`
  dispatches it for its prefill action.

**Out of scope (handled in 16c or later phases):**

- **CSS styling for the new class names.** This plan introduces several
  new class names (`plugin-card-prose-fallback`, `plugin-card-audit-strip`,
  `plugin-card-prose-section`, `plugin-card-example-code`,
  `trust-tier-badge-{1,2,3}`, `audit-icon-positive`,
  `audit-icon-attention`, `audit-icon-unknown`, etc.) used by the
  Vitest assertions for structural correctness. Visual styling for
  these is a follow-up CSS pass that should land before staging-smoke
  verification in 16c Task 5. The Vitest tests pass on class presence
  alone; the manual smoke at the end of 16c is what surfaces unstyled
  appearance. Plan a CSS-only commit between 16b and 16c (or as part of
  16c) to ship the visual treatment.
- Backend (Phase 7A delivered).
- `FilterChipStrip` component and filter state on the drawer (16c).
- `InlineChatSourceEntry` component (16c — type alone lands here).
- `CatalogDrawer.tsx` integration: filter wiring, synthetic-entry
  rendering, extended `scorePlugin`, extended `counts` (all 16c).
- `ShortcutsHelp.tsx` regroup (16c).
- Schema-field-name search support. Schemas are lazy-fetched per plugin
  on disclosure; extending fuzzy search to match across schema field
  names would require a preload pass and conflicts with the existing
  lazy-load design. Flagged so a future phase can revisit; do not ship
  a preload pass.
- Per-plugin prose authoring. 7A ships only the canonical CSV example;
  every other plugin renders with `usage_when_to_use === null` and the
  "see the technical description" fallback. That is the expected
  staging behaviour.
- Drag handles, "in use" highlighting, "recently used" sections,
  "favorites." Per design doc 08, these reintroduce the toolkit framing
  and are explicitly not added.

## Trust tier check (per CLAUDE.md)

All data in this plan flows through the frontend in one direction:
backend → catalog API → React state → DOM. The catalog API responses are
**Tier 1** (system-owned plugin metadata); the frontend types must match
exactly. New `PluginSummary` fields are typed at the boundary:

- `usage_when_to_use: string | null`
- `usage_when_not_to_use: string | null`
- `example_use: string | null`
- `capability_tags: string[]`
- `audit_characteristics: string[]`
- `data_trust_tier: 1 | 2 | 3 | null`

The `data_trust_tier` field is a **union of three literals** plus
`null`, not `number`. This matches the Pydantic `int | None` with
`ge=1, le=3` constraint on the backend. TypeScript-side, this forces a
discriminator read at render time so future tier additions surface as
type errors rather than silent fallbacks.

The synthetic "Inline data from chat" entry is **not** a `PluginSummary`.
It is a separate type (`InlineChatSourceEntry`) with its own shape.
Mixing it into the `PluginSummary[]` array via a tagged union would
require type-narrowing in every card-render site; instead, 16c's drawer
maintains the array of plugins separately and renders the synthetic
entry as a sibling at the top of the Sources tab.

## File structure

**Modified:**

- `src/elspeth/web/frontend/src/types/index.ts` — Extend `PluginSummary`;
  add `InlineChatSourceEntry` type and `DataTrustTier` union.
- `src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx` —
  Major rewrite: new card layout, remove "Use in pipeline" machinery.
- `src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx`
  — Update tests for the new layout; remove the "Use in pipeline" tests
  and the nested-interactive structural-invariant test (it ceases to
  apply once the action button is gone).

**Created:**

- `src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts`
- `src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.test.ts`
- `src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.tsx`
- `src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.test.tsx`
- `src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.tsx`
- `src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.test.tsx`

**Not modified:**

- `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` — Keeps
  its `PREFILL_CHAT_INPUT_EVENT` listener. The 16c synthetic entry
  dispatches the same event with a tailored prompt.
- `src/elspeth/web/frontend/src/api/client.ts` — No new endpoints; the
  existing wrappers continue to work. The TypeScript types they return
  get richer via the `types/index.ts` extension.

## Verification approach

Each task is TDD-shaped: failing test, minimal implementation, passing
test, commit. Vitest runs are scoped to the file under test for fast
feedback; a final regression sweep runs the full frontend suite.

---

## Task 1: Extend `PluginSummary` types

**Files:**

- Modify: `src/elspeth/web/frontend/src/types/index.ts` — extend
  `PluginSummary`; add `InlineChatSourceEntry`.

This is a pure type-level change. The implementation is small, but the
TDD structure is preserved — the failing test is a *type* assertion
that will fail to compile until the field is added.

- [ ] **Step 1: Write the failing test**

Create
`src/elspeth/web/frontend/src/types/index.test.ts` (or extend the
existing one if present):

```typescript
import { describe, it, expectTypeOf } from "vitest";
import type { PluginSummary, InlineChatSourceEntry, DataTrustTier } from "./index";

describe("PluginSummary type extension (Phase 7B)", () => {
  it("has the Phase-7A reference-content fields", () => {
    const summary: PluginSummary = {
      name: "csv",
      plugin_type: "source",
      description: "Read CSV files",
      config_fields: [],
      usage_when_to_use: "When you have a CSV file.",
      usage_when_not_to_use: "When the data is inline.",
      example_use: "source:\n  plugin: csv",
      capability_tags: ["csv", "file"],
      audit_characteristics: ["io_read", "quarantine", "coerce"],
      data_trust_tier: 3,
    };
    expectTypeOf(summary.usage_when_to_use).toEqualTypeOf<string | null>();
    expectTypeOf(summary.capability_tags).toEqualTypeOf<string[]>();
    expectTypeOf(summary.audit_characteristics).toEqualTypeOf<string[]>();
    expectTypeOf(summary.data_trust_tier).toEqualTypeOf<DataTrustTier | null>();
  });

  it("accepts null / empty defaults for unfilled plugins", () => {
    const summary: PluginSummary = {
      name: "azure_blob",
      plugin_type: "source",
      description: "Read Azure blobs",
      config_fields: [],
      usage_when_to_use: null,
      usage_when_not_to_use: null,
      example_use: null,
      capability_tags: [],
      audit_characteristics: ["io_read"],
      data_trust_tier: null,
    };
    expect(summary.usage_when_to_use).toBeNull();
  });

  it("DataTrustTier is the literal union 1 | 2 | 3", () => {
    expectTypeOf<DataTrustTier>().toEqualTypeOf<1 | 2 | 3>();
  });

  it("InlineChatSourceEntry has its own shape distinct from PluginSummary", () => {
    const entry: InlineChatSourceEntry = {
      kind: "inline-chat-source",
      name: "Inline data from chat",
      description: "Type your data directly; no plugin required.",
      example_prompt: "Use the LLM to summarise this article: https://example.com",
    };
    expectTypeOf(entry.kind).toEqualTypeOf<"inline-chat-source">();
  });
});
```

- [ ] **Step 2: Run test to verify it fails (type error)**

```bash
cd src/elspeth/web/frontend && npx vitest run src/types/index.test.ts
```

Expected: FAIL with a TypeScript compile error noting that
`usage_when_to_use` does not exist on `PluginSummary` and
`InlineChatSourceEntry` is not exported.

- [ ] **Step 3: Extend the types**

In `src/elspeth/web/frontend/src/types/index.ts`, replace the existing
`PluginSummary` interface with:

```typescript
/** Three-tier trust classification surfaced on plugin cards.
 *
 * Reading: "what tier of data does this plugin handle at its boundary?"
 *   1 = our data (audit, checkpoints)
 *   2 = pipeline data (post-source)
 *   3 = external data (source input, external-call response)
 *
 * Sources and external-call transforms surface tier 3; pure row
 * transforms = tier 2; sinks = tier 2. See CLAUDE.md "Data Manifesto"
 * for the underlying tier definitions.
 */
export type DataTrustTier = 1 | 2 | 3;

/** Plugin summary from the catalog listing endpoints.
 *
 * Phase 7A added reference-content fields populated by plugin authors.
 * Unfilled plugins return `null` / empty values; the catalog drawer
 * renders a "see the technical description" fallback for them.
 */
export interface PluginSummary {
  name: string;
  plugin_type: "source" | "transform" | "sink";
  description: string;
  config_fields: { name: string; type: string; required: boolean; description: string; default: unknown }[];

  // Phase 7B reference-content fields
  usage_when_to_use: string | null;
  usage_when_not_to_use: string | null;
  example_use: string | null;
  capability_tags: string[];
  audit_characteristics: string[];
  data_trust_tier: DataTrustTier | null;
}
```

Then immediately after, add the synthetic-entry type:

```typescript
/** Synthetic catalog entry rendered as the first row of the Sources tab.
 *
 * NOT a backend plugin — it is a frontend-only affordance representing
 * "type your data directly in chat; no plugin required" per design doc
 * 08-§"The 'Inline data from chat' entry". The composer creates a one-
 * row dynamic source from the user's chat message at runtime.
 *
 * Distinct shape from PluginSummary so the catalog drawer can render it
 * with different visual style and different action (prefill the chat
 * input rather than expand a schema).
 */
export interface InlineChatSourceEntry {
  kind: "inline-chat-source";
  name: string;
  description: string;
  /** Suggested prompt the user can adapt; clicking the entry prefills
   * this into the chat input via PREFILL_CHAT_INPUT_EVENT. */
  example_prompt: string;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/types/index.test.ts
```

Expected: PASS — types compile and assertions hold.

- [ ] **Step 5: Verify nothing else breaks**

```bash
cd src/elspeth/web/frontend && npx tsc --noEmit
```

Expected: PASS — the existing call sites that construct `PluginSummary`
from API responses don't yet provide the new fields, but they're
populated by the backend (Phase 7A); for any in-test fixtures that
hand-roll `PluginSummary` without the new fields, the compile will
flag them.

If `tsc` reports errors in test fixtures, add the missing fields with
the documented defaults (`null` / `[]` / `null` for `data_trust_tier`).
For *production* call sites that construct `PluginSummary` themselves
(unlikely; they all come from the API), do the same.

**Known fixtures that need updating in this commit (verified by
reconnaissance against current `main`):**

- `src/elspeth/web/frontend/src/components/catalog/CatalogDrawer.test.tsx`
  — `vi.mock("@/api/client", …)` block constructs `PluginSummary`
  fixtures for `listSources` / `listTransforms` / `listSinks`. Add the
  six new fields with `null` / `[]` / `null` defaults so the existing
  rendering tests compile under the stricter interface.
- `src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx`
  — every `makePlugin(...)` factory / inline `PluginSummary` literal
  needs the same default expansion. This file is largely rewritten in
  Task 5 of this plan, but the Task-1 compile must pass first.

If `npx tsc --noEmit` surfaces any other fixture file constructing
`PluginSummary` literally, treat the same way; do not narrow the type
to make existing fixtures compile. Per CLAUDE.md "Fix errors you
encounter," fan the type extension out in the same commit.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/types/index.ts \
  src/elspeth/web/frontend/src/types/index.test.ts
git commit -m "feat(frontend): extend PluginSummary with Phase-7A fields (Phase 7B.1)"
```

If any test fixtures needed updates to provide the new fields, add them
to the same commit.

## Task 2: `auditCharacteristics.ts` — centralised flag metadata

**Files:**

- Create:
  `src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts`
- Create:
  `src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.test.ts`

A single source of truth maps each audit-characteristic flag string to
its display metadata: label, icon glyph (CSS-class or Unicode), tooltip
text, and tone (positive = checkmark, attention = warning sign).
Centralising this prevents drift between the card and the filter chip.

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from "vitest";
import {
  AUDIT_CHARACTERISTICS,
  lookupAuditCharacteristic,
  KNOWN_AUDIT_FLAGS,
} from "./auditCharacteristics";

describe("auditCharacteristics metadata", () => {
  it("exposes a metadata entry for io_read", () => {
    const meta = lookupAuditCharacteristic("io_read");
    expect(meta).not.toBeNull();
    expect(meta?.label).toMatch(/i\/?o.read|reads/i);
    expect(meta?.tone).toBe("positive");
  });

  it("exposes a metadata entry for external_call with attention tone", () => {
    const meta = lookupAuditCharacteristic("external_call");
    expect(meta?.tone).toBe("attention");
    expect(meta?.tooltip).toMatch(/external|network/i);
  });

  it("exposes provenance / retention / quarantine / coerce / signed", () => {
    expect(lookupAuditCharacteristic("emits_provenance")).not.toBeNull();
    expect(lookupAuditCharacteristic("quarantine")).not.toBeNull();
    expect(lookupAuditCharacteristic("coerce")).not.toBeNull();
    expect(lookupAuditCharacteristic("signed")).not.toBeNull();
  });

  it("returns null for an unknown flag rather than crashing", () => {
    // Future flags added on the backend without a frontend metadata
    // entry should render as a small grey "unknown" chip, not crash.
    expect(lookupAuditCharacteristic("future_flag_2027")).toBeNull();
  });

  it("KNOWN_AUDIT_FLAGS lists every metadata key", () => {
    expect(KNOWN_AUDIT_FLAGS).toContain("io_read");
    expect(KNOWN_AUDIT_FLAGS).toContain("external_call");
    expect(KNOWN_AUDIT_FLAGS).toContain("quarantine");
  });

  it("AUDIT_CHARACTERISTICS table includes determinism-derived flags", () => {
    // The Phase-7A derivation rules turn Determinism enum values into
    // flag strings verbatim (io_read, io_write, external_call,
    // deterministic, seeded, non_deterministic). The frontend metadata
    // table must cover each so the inferred-flag case has a renderer.
    for (const flag of ["io_read", "io_write", "external_call", "deterministic", "seeded", "non_deterministic"]) {
      expect(lookupAuditCharacteristic(flag)).not.toBeNull();
    }
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/auditCharacteristics.test.ts
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement the metadata table**

Create
`src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts`:

```typescript
// ============================================================================
// auditCharacteristics — centralised metadata for audit-characteristic flags
//
// Source of truth for how audit-characteristic flag strings render on the
// plugin card and in the filter chip strip. The backend (Phase 7A
// _derive_audit_characteristics) emits flag strings; this table maps each
// to its UI metadata.
//
// Unknown flags (no entry here) render as a small grey "unknown" chip with
// the raw flag string as label; this is the forward-compatibility path
// for flags added on the backend without a corresponding frontend update.
// ============================================================================

/** Visual tone for an audit-characteristic chip / icon. */
export type AuditCharacteristicTone = "positive" | "attention";

/** Display metadata for one audit-characteristic flag. */
export interface AuditCharacteristicMeta {
  /** Canonical flag string (matches the backend wire format). */
  flag: string;
  /** Short label shown next to the icon on the card. */
  label: string;
  /** Single-character or short glyph (Unicode) used as the icon. */
  glyph: string;
  /** Plain-language tooltip explaining the flag to a non-developer. */
  tooltip: string;
  /** Visual tone — "positive" renders as a green checkmark-style chip;
   *  "attention" renders as a yellow warning-style chip. */
  tone: AuditCharacteristicTone;
}

export const AUDIT_CHARACTERISTICS: AuditCharacteristicMeta[] = [
  // ── Determinism-derived ───────────────────────────────────────────────
  {
    flag: "deterministic",
    label: "deterministic",
    glyph: "≡",
    tooltip:
      "Plugin produces identical output on every run with the same input. " +
      "Safe to re-run; no replay machinery needed.",
    tone: "positive",
  },
  {
    flag: "seeded",
    label: "seeded",
    glyph: "🎲",
    tooltip:
      "Plugin uses pseudo-randomness controlled by a seed. Replay captures " +
      "the seed; re-running with the same seed reproduces the output.",
    tone: "positive",
  },
  {
    flag: "io_read",
    label: "reads I/O",
    glyph: "📥",
    tooltip:
      "Plugin reads from an external file, environment variable, or local " +
      "filesystem. Replay captures what was read.",
    tone: "positive",
  },
  {
    flag: "io_write",
    label: "writes I/O",
    glyph: "📤",
    tooltip:
      "Plugin writes to a file, environment, or local filesystem. Be " +
      "careful — replay re-applies the side effects.",
    tone: "attention",
  },
  {
    flag: "external_call",
    label: "Network call",
    glyph: "🌐",
    tooltip:
      "Plugin reaches an external system over the network (HTTP, API, " +
      "service call). Replay records the request and response.",
    tone: "attention",
  },
  {
    flag: "non_deterministic",
    label: "non-deterministic",
    glyph: "⁇",
    tooltip:
      "Plugin output is not reproducible from inputs alone. Replay must " +
      "record the full output verbatim.",
    tone: "attention",
  },

  // ── Source / sink behaviour ───────────────────────────────────────────
  {
    flag: "quarantine",
    label: "quarantines bad rows",
    glyph: "🛡",
    tooltip:
      "Source quarantines malformed rows to a designated sink instead of " +
      "crashing or silently discarding them. The audit trail records " +
      "which rows were quarantined and why.",
    tone: "positive",
  },
  {
    flag: "coerce",
    label: "coerces types",
    glyph: "↔",
    tooltip:
      "Plugin coerces external string-typed cells to typed columns at the " +
      "Tier-3 boundary (e.g., \"42\" → 42). Coercion is meaning-preserving; " +
      "fabrication is not. See CLAUDE.md \"Data Manifesto\" for the rule.",
    tone: "positive",
  },

  // ── Authored characteristics ──────────────────────────────────────────
  {
    flag: "emits_provenance",
    label: "extra provenance",
    glyph: "🔍",
    tooltip:
      "Plugin emits additional provenance records beyond the standard " +
      "pipeline lineage (e.g., per-row hashes of source bytes).",
    tone: "positive",
  },
  {
    flag: "signed",
    label: "HMAC-signed",
    glyph: "🔏",
    tooltip:
      "Plugin output is HMAC-signed for tamper-evidence. The signing key " +
      "is part of the audit trail.",
    tone: "positive",
  },
  {
    flag: "credentials",
    label: "needs credentials",
    glyph: "🔑",
    tooltip:
      "Plugin requires user secrets (API keys, tokens) to operate. " +
      "Credentials are stored via the secret-handling pathway, not in " +
      "pipeline config.",
    tone: "attention",
  },
];

const _byFlag = new Map<string, AuditCharacteristicMeta>(
  AUDIT_CHARACTERISTICS.map((m) => [m.flag, m]),
);

export const KNOWN_AUDIT_FLAGS: string[] = AUDIT_CHARACTERISTICS.map((m) => m.flag);

/** Look up the display metadata for a flag string.
 *  Returns null for unknown flags (forward-compatible). */
export function lookupAuditCharacteristic(
  flag: string,
): AuditCharacteristicMeta | null {
  return _byFlag.get(flag) ?? null;
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/auditCharacteristics.test.ts
```

Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.ts \
  src/elspeth/web/frontend/src/components/catalog/auditCharacteristics.test.ts
git commit -m "feat(frontend): catalog audit-characteristic metadata table (Phase 7B.2)"
```

## Task 3: `AuditCharacteristicIcon.tsx` — single-flag renderer

**Files:**

- Create:
  `src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.tsx`
- Create:
  `src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AuditCharacteristicIcon } from "./AuditCharacteristicIcon";

describe("AuditCharacteristicIcon", () => {
  it("renders the label and glyph for a known flag", () => {
    render(<AuditCharacteristicIcon flag="io_read" />);
    expect(screen.getByText(/reads i\/?o/i)).toBeInTheDocument();
  });

  it("uses a positive-tone class for io_read", () => {
    const { container } = render(<AuditCharacteristicIcon flag="io_read" />);
    expect(container.firstChild).toHaveClass("audit-icon-positive");
  });

  it("uses an attention-tone class for external_call", () => {
    const { container } = render(<AuditCharacteristicIcon flag="external_call" />);
    expect(container.firstChild).toHaveClass("audit-icon-attention");
  });

  it("renders the tooltip on the title attribute", () => {
    render(<AuditCharacteristicIcon flag="quarantine" />);
    const el = screen.getByText(/quarantines/i);
    // Tooltip via title for keyboard / screen-reader access without
    // pulling in a tooltip library.
    expect(el.closest("[title]")?.getAttribute("title")).toMatch(/sink/i);
  });

  it("renders unknown flags as a fallback chip with the raw flag string", () => {
    render(<AuditCharacteristicIcon flag="future_flag_2027" />);
    expect(screen.getByText("future_flag_2027")).toBeInTheDocument();
  });

  it("applies an 'audit-icon-unknown' class for unknown flags", () => {
    const { container } = render(
      <AuditCharacteristicIcon flag="future_flag_2027" />,
    );
    expect(container.firstChild).toHaveClass("audit-icon-unknown");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/AuditCharacteristicIcon.test.tsx
```

Expected: FAIL — component doesn't exist.

- [ ] **Step 3: Implement the component**

```typescript
// ============================================================================
// AuditCharacteristicIcon
//
// Single-flag renderer used by the plugin card and the filter chip strip.
// Looks up the flag in the centralised metadata table; falls back to a
// "unknown" chip for forward compatibility with backend flag additions
// that predate the corresponding frontend metadata.
// ============================================================================

import { lookupAuditCharacteristic } from "./auditCharacteristics";

interface AuditCharacteristicIconProps {
  flag: string;
}

export function AuditCharacteristicIcon({ flag }: AuditCharacteristicIconProps) {
  const meta = lookupAuditCharacteristic(flag);
  if (meta === null) {
    return (
      <span
        className="audit-icon audit-icon-unknown"
        title={`Unknown audit characteristic: ${flag}`}
      >
        {flag}
      </span>
    );
  }
  return (
    <span
      className={`audit-icon audit-icon-${meta.tone}`}
      title={meta.tooltip}
    >
      <span className="audit-icon-glyph" aria-hidden="true">{meta.glyph}</span>
      <span className="audit-icon-label">{meta.label}</span>
    </span>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/AuditCharacteristicIcon.test.tsx
```

Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.tsx \
  src/elspeth/web/frontend/src/components/catalog/AuditCharacteristicIcon.test.tsx
git commit -m "feat(frontend): AuditCharacteristicIcon component (Phase 7B.3)"
```

## Task 4: `TrustTierBadge.tsx` — tier badge renderer

**Files:**

- Create:
  `src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.tsx`
- Create:
  `src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.test.tsx`

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TrustTierBadge } from "./TrustTierBadge";

describe("TrustTierBadge", () => {
  it("renders 'Tier 1' for tier=1", () => {
    render(<TrustTierBadge tier={1} />);
    expect(screen.getByText("Tier 1")).toBeInTheDocument();
  });

  it("renders 'Tier 2' for tier=2", () => {
    render(<TrustTierBadge tier={2} />);
    expect(screen.getByText("Tier 2")).toBeInTheDocument();
  });

  it("renders 'Tier 3' for tier=3", () => {
    render(<TrustTierBadge tier={3} />);
    expect(screen.getByText("Tier 3")).toBeInTheDocument();
  });

  it("renders 'tier unspecified' for tier=null", () => {
    render(<TrustTierBadge tier={null} />);
    expect(screen.getByText(/unspecified/i)).toBeInTheDocument();
  });

  it("includes a tooltip explaining each tier", () => {
    render(<TrustTierBadge tier={3} />);
    const badge = screen.getByText("Tier 3");
    // Tier-3 tooltip references external data.
    expect(badge.closest("[title]")?.getAttribute("title")).toMatch(/external/i);
  });

  it("applies a per-tier class", () => {
    const { container, rerender } = render(<TrustTierBadge tier={1} />);
    expect(container.firstChild).toHaveClass("trust-tier-badge-1");
    rerender(<TrustTierBadge tier={2} />);
    expect(container.firstChild).toHaveClass("trust-tier-badge-2");
    rerender(<TrustTierBadge tier={3} />);
    expect(container.firstChild).toHaveClass("trust-tier-badge-3");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/TrustTierBadge.test.tsx
```

Expected: FAIL — component doesn't exist.

- [ ] **Step 3: Implement the component**

```typescript
// ============================================================================
// TrustTierBadge
//
// Small badge rendering the data trust tier surfaced at a plugin's
// boundary. NOT a measure of plugin trust ("all plugins are Tier 1
// system code" per CLAUDE.md plugin-ownership policy); rather, a
// measure of *what tier of data* this plugin handles at the seam.
// Tooltips explain the tier in plain language; the per-tier class
// drives the colour scheme.
// ============================================================================

import type { DataTrustTier } from "@/types/index";

interface TrustTierBadgeProps {
  tier: DataTrustTier | null;
}

const TIER_TOOLTIPS: Record<1 | 2 | 3, string> = {
  1: "Tier 1: Our own data (audit trail, checkpoints). Fully trusted. " +
    "Plugins that handle Tier-1 data crash on any anomaly.",
  2: "Tier 2: Pipeline data after the source boundary. Types are " +
    "trustworthy; values may still cause operation failures. Plugins do " +
    "not coerce — type errors are upstream plugin bugs.",
  3: "Tier 3: External data crossing into the pipeline (source input, " +
    "external-call response). Zero trust. The plugin validates and " +
    "coerces at the boundary; malformed rows are quarantined.",
};

export function TrustTierBadge({ tier }: TrustTierBadgeProps) {
  if (tier === null) {
    return (
      <span
        className="trust-tier-badge trust-tier-badge-unspecified"
        title="The plugin author has not declared the data trust tier."
      >
        tier unspecified
      </span>
    );
  }
  return (
    <span
      className={`trust-tier-badge trust-tier-badge-${tier}`}
      title={TIER_TOOLTIPS[tier]}
    >
      Tier {tier}
    </span>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/TrustTierBadge.test.tsx
```

Expected: PASS — all six tests green.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.tsx \
  src/elspeth/web/frontend/src/components/catalog/TrustTierBadge.test.tsx
git commit -m "feat(frontend): TrustTierBadge component (Phase 7B.4)"
```

## Task 5: Rewrite `PluginCard.tsx`

This is the load-bearing change for "reference, not toolkit." The card
now renders the reference content shipped by Phase 7A; the "Use in
pipeline" button and its supporting machinery are deleted; the new
layout follows design doc 08-§"Plugin card content design."

**Files:**

- Modify:
  `src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx`.
- Modify:
  `src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx`.

The `PREFILL_CHAT_INPUT_EVENT` export stays — `ChatInput.tsx` listens
for it, and 16c's `InlineChatSourceEntry` (Task 2 in 16c) dispatches it.
The `buildInsertionPrompt` / `handleUseInPipeline` / "Use in pipeline"
JSX go away.

- [ ] **Step 1: Write the failing test**

Replace
`src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx`
content (preserving its imports and basic setup) with the new layout
expectations. The complete test file (rewriting the existing one):

```typescript
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PluginCard } from "./PluginCard";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";

function makePlugin(overrides: Partial<PluginSummary> = {}): PluginSummary {
  return {
    name: "csv",
    plugin_type: "source",
    description: "Read rows from a CSV file.",
    config_fields: [],
    usage_when_to_use: "When you have a CSV file already.",
    usage_when_not_to_use: "When the data is inline; use chat instead.",
    example_use: "source:\n  plugin: csv",
    capability_tags: ["csv", "file"],
    audit_characteristics: ["io_read", "quarantine", "coerce"],
    data_trust_tier: 3,
    ...overrides,
  };
}

describe("PluginCard — Phase 7B reshape", () => {
  it("renders the plugin name and one-line description", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    expect(screen.getByText("csv")).toBeInTheDocument();
    expect(screen.getByText(/Read rows from a CSV file/i)).toBeInTheDocument();
  });

  it("renders the trust-tier badge", () => {
    render(<PluginCard plugin={makePlugin({ data_trust_tier: 3 })} schema={null} onExpand={() => {}} />);
    expect(screen.getByText("Tier 3")).toBeInTheDocument();
  });

  it("renders one audit-characteristic icon per flag", () => {
    render(
      <PluginCard
        plugin={makePlugin({ audit_characteristics: ["io_read", "quarantine"] })}
        schema={null}
        onExpand={() => {}}
      />,
    );
    expect(screen.getByText(/reads i\/?o/i)).toBeInTheDocument();
    expect(screen.getByText(/quarantines/i)).toBeInTheDocument();
  });

  it("renders the 'When you'd use this' prose", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    expect(screen.getByText(/when you'd use this/i)).toBeInTheDocument();
    expect(screen.getByText(/when you have a csv file/i)).toBeInTheDocument();
  });

  it("renders the 'When you wouldn't' prose", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    expect(screen.getByText(/when you wouldn't/i)).toBeInTheDocument();
  });

  it("renders the example use as a code block preserving whitespace", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    const codeBlock = screen.getByText(/plugin: csv/);
    expect(codeBlock.tagName.toLowerCase()).toBe("pre");
  });

  it("falls back to a generic message when prose fields are null", () => {
    render(
      <PluginCard
        plugin={makePlugin({ usage_when_to_use: null, usage_when_not_to_use: null, example_use: null })}
        schema={null}
        onExpand={() => {}}
      />,
    );
    // Per design doc 08-§Risks: "Empty entries fall back to a generic
    // 'see the technical description' message rather than blocking display."
    expect(screen.getByText(/see the technical description/i)).toBeInTheDocument();
  });

  it("does NOT render a 'Use in pipeline' button (toolkit affordance removed)", () => {
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={() => {}} />);
    expect(screen.queryByRole("button", { name: /use in pipeline/i })).not.toBeInTheDocument();
  });

  it("calls onExpand when the 'Schema →' disclosure is activated", async () => {
    const onExpand = vi.fn();
    render(<PluginCard plugin={makePlugin()} schema={null} onExpand={onExpand} />);
    const disclosure = screen.getByRole("button", { name: /schema/i });
    await userEvent.click(disclosure);
    expect(onExpand).toHaveBeenCalled();
  });

  it("renders the expanded schema after onExpand resolves and schema arrives", () => {
    const schema: PluginSchemaInfo = {
      name: "csv",
      plugin_type: "source",
      description: "Read CSV files.",
      json_schema: {
        properties: { path: { type: "string", description: "Path to file" } },
        required: ["path"],
      },
    } as unknown as PluginSchemaInfo;
    render(<PluginCard plugin={makePlugin()} schema={schema} onExpand={() => {}} initialExpanded />);
    expect(screen.getByText("path")).toBeInTheDocument();
  });
});
```

Note: the test introduces an `initialExpanded` prop (or equivalent) for
the schema-render assertion. If you'd prefer to keep the toggle entirely
internal, replace that test with: click the disclosure, await
`onExpand`, then assert the field appears. Either works.

- [ ] **Step 2: Run test to verify it fails**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/PluginCard.test.tsx
```

Expected: FAIL — the current card doesn't have the new layout; many
assertions miss.

- [ ] **Step 3: Rewrite `PluginCard.tsx`**

Replace the full content of
`src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx` with:

```typescript
// ============================================================================
// PluginCard (Phase 7B — reference, not toolkit)
//
// Renders one plugin as a reference card per design doc
// 08-§"Plugin card content design":
//
//   ┌──────────────────────────────────────────────────┐
//   │  csv                                Tier 3       │
//   │  Read rows from a CSV file. ...                  │
//   │                                                  │
//   │  Audit: ✓ reads I/O  ✓ quarantines  ✓ coerces   │
//   │                                                  │
//   │  When you'd use this:   ...                      │
//   │  When you wouldn't:     ...                      │
//   │  Example use:           <pre>...</pre>           │
//   │                                                  │
//   │  [ Schema → ]                                    │
//   └──────────────────────────────────────────────────┘
//
// The "Use in pipeline" button and supporting machinery (from the
// pre-Phase-7B toolkit framing) are deliberately removed. The
// PREFILL_CHAT_INPUT_EVENT export stays — InlineChatSourceEntry uses it
// for its prefill action, and ChatInput.tsx remains the receiver.
// ============================================================================

import { useState, type MouseEvent } from "react";
import type { PluginSummary, PluginSchemaInfo } from "@/types/index";
import { TrustTierBadge } from "./TrustTierBadge";
import { AuditCharacteristicIcon } from "./AuditCharacteristicIcon";

/** Event name dispatched by InlineChatSourceEntry and consumed by
 *  ChatInput.tsx. Re-exported here for backwards compatibility with
 *  the existing import chain in ChatInput.tsx; do not rename. */
export const PREFILL_CHAT_INPUT_EVENT = "composer:prefill-chat-input";

interface JsonSchemaField {
  type?: string;
  description?: string;
}
interface JsonSchemaObject {
  properties?: Record<string, JsonSchemaField>;
  required?: string[];
}
interface DiscriminatedSchema {
  oneOf?: Array<{ $ref?: string }>;
  discriminator?: { propertyName?: string; mapping?: Record<string, string> };
  $defs?: Record<string, JsonSchemaObject>;
}

const DEFS_REF_PREFIX = "#/$defs/";

interface PluginCardProps {
  plugin: PluginSummary;
  schema: PluginSchemaInfo | null;
  schemaError?: boolean;
  onExpand: () => void;
  onRetrySchema?: () => void;
  /** Test-only: start in the expanded state to assert schema rendering. */
  initialExpanded?: boolean;
}

function isDiscriminated(s: DiscriminatedSchema & JsonSchemaObject): boolean {
  return Array.isArray(s.oneOf) && s.$defs !== undefined;
}

function resolveVariants(s: DiscriminatedSchema): Array<{ label: string; def: JsonSchemaObject }> {
  const defs = s.$defs ?? {};
  const mapping = s.discriminator?.mapping ?? {};
  const refToValue = new Map<string, string>();
  for (const [discValue, ref] of Object.entries(mapping)) {
    if (ref.startsWith(DEFS_REF_PREFIX)) {
      refToValue.set(ref.slice(DEFS_REF_PREFIX.length), discValue);
    }
  }
  const discProp = s.discriminator?.propertyName ?? "variant";
  const out: Array<{ label: string; def: JsonSchemaObject }> = [];
  for (const entry of s.oneOf ?? []) {
    const ref = entry.$ref ?? "";
    if (!ref.startsWith(DEFS_REF_PREFIX)) continue;
    const defName = ref.slice(DEFS_REF_PREFIX.length);
    const def = defs[defName];
    if (def === undefined) continue;
    out.push({ label: `${discProp}: ${refToValue.get(defName) ?? defName}`, def });
  }
  return out;
}

function renderFields(properties: Record<string, JsonSchemaField>, required: string[] | undefined): JSX.Element[] {
  const req = new Set(required ?? []);
  return Object.entries(properties).map(([name, field]) => (
    <div key={name}>
      <span className="plugin-card-field-name">{name}</span>
      <span className="plugin-card-field-type">{field.type ?? "any"}</span>
      {req.has(name) && <span className="plugin-card-field-required">required</span>}
      {field.description && <div className="plugin-card-field-desc">{field.description}</div>}
    </div>
  ));
}

const PROSE_FALLBACK = "See the technical description above.";

export function PluginCard({
  plugin,
  schema,
  schemaError,
  onExpand,
  onRetrySchema,
  initialExpanded = false,
}: PluginCardProps) {
  const [expanded, setExpanded] = useState(initialExpanded);

  function handleDisclosureClick(e: MouseEvent<HTMLButtonElement>) {
    e.preventDefault();
    if (!expanded) onExpand();
    setExpanded((p) => !p);
  }

  function handleRetry(e: MouseEvent<HTMLButtonElement>) {
    e.stopPropagation();
    (onRetrySchema ?? onExpand)();
  }

  const configSchema = schema?.json_schema as (DiscriminatedSchema & JsonSchemaObject) | undefined;

  const allFallback =
    plugin.usage_when_to_use === null &&
    plugin.usage_when_not_to_use === null &&
    plugin.example_use === null;

  return (
    <div className="plugin-card">
      {/* Header row: name + tier badge */}
      <div className="plugin-card-header-row">
        <span className="plugin-card-name">{plugin.name}</span>
        <TrustTierBadge tier={plugin.data_trust_tier} />
      </div>

      {/* Short technical description */}
      <div className="plugin-card-desc" title={plugin.description}>
        {plugin.description}
      </div>

      {/* Audit-characteristic strip */}
      {plugin.audit_characteristics.length > 0 && (
        <div className="plugin-card-audit-strip" aria-label="Audit characteristics">
          <span className="plugin-card-audit-label">Audit:</span>
          {/* Stable display order: sort lexically so the same plugin
              always shows the same icon order regardless of frozenset
              serialization order from the backend. */}
          {[...plugin.audit_characteristics].sort().map((flag) => (
            <AuditCharacteristicIcon key={flag} flag={flag} />
          ))}
        </div>
      )}

      {/* Reference prose */}
      {allFallback ? (
        <div className="plugin-card-prose-fallback">{PROSE_FALLBACK}</div>
      ) : (
        <>
          <ProseSection label="When you'd use this" body={plugin.usage_when_to_use} />
          <ProseSection label="When you wouldn't" body={plugin.usage_when_not_to_use} />
          {plugin.example_use !== null && (
            <div className="plugin-card-example">
              <div className="plugin-card-example-label">Example use:</div>
              <pre className="plugin-card-example-code">{plugin.example_use}</pre>
            </div>
          )}
        </>
      )}

      {/* Schema disclosure */}
      <button
        type="button"
        className="btn btn-small plugin-card-disclosure"
        onClick={handleDisclosureClick}
        aria-expanded={expanded}
        aria-label={`Schema for ${plugin.name}`}
      >
        Schema {expanded ? "▾" : "→"}
      </button>

      {expanded && (
        <div className="plugin-card-expanded">
          {schemaError ? (
            <div className="plugin-card-schema-error">
              <span>Failed to load schema.</span>
              <button type="button" className="btn btn-small" onClick={handleRetry} aria-label="Retry loading schema">
                Retry
              </button>
            </div>
          ) : schema === null || configSchema === undefined ? (
            <div role="status" aria-live="polite" className="plugin-card-schema-loading">
              <span className="spinner" aria-hidden="true" /> Loading schema...
            </div>
          ) : isDiscriminated(configSchema) ? (
            <div className="plugin-card-variants">
              {resolveVariants(configSchema).map((v) => (
                <div key={v.label} className="plugin-card-variant">
                  <div className="plugin-card-variant-label">{v.label}</div>
                  {v.def.properties ? (
                    <div className="plugin-card-fields">{renderFields(v.def.properties, v.def.required)}</div>
                  ) : (
                    <span className="plugin-card-no-fields">No configuration fields.</span>
                  )}
                </div>
              ))}
            </div>
          ) : configSchema.properties ? (
            <div className="plugin-card-fields">{renderFields(configSchema.properties, configSchema.required)}</div>
          ) : (
            <span className="plugin-card-no-fields">No configuration fields.</span>
          )}
        </div>
      )}
    </div>
  );
}

function ProseSection({ label, body }: { label: string; body: string | null }) {
  if (body === null) return null;
  return (
    <div className="plugin-card-prose-section">
      <div className="plugin-card-prose-label">{label}:</div>
      <div className="plugin-card-prose-body">{body}</div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/PluginCard.test.tsx
```

Expected: PASS — all 10 tests green.

- [ ] **Step 5: Run the rest of the catalog and chat suites for regressions**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/catalog/ src/components/chat/
```

Expected: PASS. `ChatInput.test.tsx` still imports
`PREFILL_CHAT_INPUT_EVENT` from `PluginCard.tsx`; the existing direct
export at `PluginCard.tsx:17` preserves that contract (no new
re-export module is introduced).

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/frontend/src/components/catalog/PluginCard.tsx \
  src/elspeth/web/frontend/src/components/catalog/PluginCard.test.tsx
git commit -m "$(cat <<'EOF'
refactor(frontend): rewrite PluginCard as reference, not toolkit

Phase 7B.5 of composer UX redesign. Implements design doc 08's "Plugin
card content design" layout: trust-tier badge, audit-characteristic icon
strip, persona-facing prose ("When you'd use this" / "When you wouldn't"
/ "Example use"), and a "Schema →" disclosure. Removes the "Use in
pipeline" button and its supporting buildInsertionPrompt /
handleUseInPipeline machinery — per design doc 08, plugin selection is
the LLM's job driven by user intent, not a toolkit affordance.

PREFILL_CHAT_INPUT_EVENT keeps its existing direct export at
PluginCard.tsx:17 — ChatInput.tsx still listens for it and Phase 7C's
InlineChatSourceEntry will dispatch it. No intermediate re-export
module is needed. The buildInsertionPrompt helper and the onCloseDrawer
prop are removed — neither has a consumer anymore.

Unfilled plugins (prose fields all null) fall back to a "see the
technical description" message per design doc 08-§Risks. See
docs/composer/ux-redesign-2026-05/16b-phase-7b-frontend.md.
EOF
)"
```


---

## What Phase 7B leaves the frontend in

After all five tasks land:

- `PluginSummary` exposes the Phase-7A reference-content fields; the
  TypeScript types match the wire format exactly.
- A centralised `auditCharacteristics.ts` table maps every known flag
  string to display metadata; unknown flags fall back to a forward-
  compatible "unknown" chip.
- `AuditCharacteristicIcon` and `TrustTierBadge` render the
  per-characteristic and per-tier visuals; both are consumed by the
  rewritten `PluginCard` and (in 16c) by `FilterChipStrip`.
- `PluginCard.tsx` is the reference-style card from design doc 08; the
  "Use in pipeline" toolkit affordance is gone.

The catalog drawer is still wired the old way (no filters, no synthetic
entry, old `scorePlugin`, old shortcuts). The reshape is *visible* on
each plugin card but not yet *integrated* in the drawer. Phase 7C
finishes the integration.

If 7B ships alone (without 7C), the user sees:

- A new card layout per plugin (good — it's the load-bearing UX change).
- No filter chips.
- No "Inline data from chat" entry at the top of Sources.
- The keyboard shortcut still grouped flat (with all other shortcuts).

That intermediate state is acceptable but lacks the orientation
affordances; ship 7C immediately after 7B to complete the reshape.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Removing "Use in pipeline" breaks a workflow some user depended on | Pre-RC5 has no documented users; per CLAUDE.md "No Legacy Code Policy," deferring breaking changes is the opposite of what we want. The replacement workflow is described in 16c's `InlineChatSourceEntry` and the chat input prefill. |
| `audit_characteristics` ordering varies per response | Backend now emits `tuple[str, ...]` sorted in `_derive_audit_characteristics` (Phase 7A rev-2), so the wire shape is deterministic. The card also sorts the array lexically before rendering as belt-and-braces against future drift; the same plugin always shows the same icon order regardless of source ordering. |
| Backend missing (Phase 7A not yet shipped) | Backwards-compatible TypeScript types: every new `PluginSummary` field is optional in the type union (`string \| null` / `string[]`). The frontend reads with `?? null` / `?? []` fallbacks. When the backend ships, the data "comes alive." |
| `ChatInput.tsx` test breaks because `PREFILL_CHAT_INPUT_EVENT` moved | The export stays at the same import path (`@/components/catalog/PluginCard`). Verified during the regression sweep in Task 5. |
| The "see the technical description" fallback message blends into the technical description above it | Visually distinct class (`plugin-card-prose-fallback`) per the CSS author's discretion; the message is short and italicised by default. If staging exercise shows it's missed, 7C's smoke can surface and 7B's CSS can be retuned. |

## Memory references

- `feedback_catalog_is_reference_not_toolkit` — the design call this implements.
- `project_composer_personas` — informs the persona-facing reads of the
  new card layout (Linda → "when you'd use this"; Marcus → "when you
  wouldn't"; Dev → "example use" + "Schema →").
- `feedback_no_calendar_shipping_commitments` — no calendar commitments
  in this plan.

## Review history

- 2026-05-15 reality-review: Changed `auditCharacteristics.ts` lookup key and all test flag strings from `"network"` to `"external_call"` (keeping user-facing `label` as `"Network call"`); fixed `KNOWN_AUDIT_FLAGS` and derived-flags loop to include `"external_call"`; clarified `PREFILL_CHAT_INPUT_EVENT` wording from "re-export" to "existing direct export at `PluginCard.tsx:17`".
- 2026-05-16 reality-review (rev-2): Enumerated the existing test fixtures that will fail to compile when `PluginSummary` gains required new fields (`CatalogDrawer.test.tsx`, `PluginCard.test.tsx`). The plan previously said "if tsc reports errors, fix them"; rev-2 names the specific files so the Task-1 commit isn't blocked on rediscovery.
