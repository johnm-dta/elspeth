# Phase 5a — Dynamic-source-from-chat (frontend-only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status header — B1 verified, frontend-only.** The open question B1
([roadmap §A pre-Phase-5a](00-implementation-roadmap.md#pre-phase-5a-blocks-dynamic-source-from-chat))
was resolved with verdict **(a) Yes, already works — proceed**. The backend
mechanism for inline-content sources already exists end-to-end:

- `set_pipeline` accepts `source.inline_blob` —
  `src/elspeth/web/composer/tools.py` lines 4414-4472.
- Inline content is SHA-256 hashed at composition time, stored as a session
  blob, and bound as a normal source.
- Inline-sourced rows go through the same `stable_hash()` path
  (`src/elspeth/contracts/hashing.py` lines 80-93) and the same
  `create_row()` method
  (`src/elspeth/core/landscape/data_flow_repository.py` lines 381-424) as
  CSV-sourced rows. `source_data_hash` (`schema.py` line 165) is populated
  identically.
- Redaction has Pydantic models for `source.inline_blob` —
  `src/elspeth/web/composer/redaction.py` lines 1022-1106.

**Implication:** Phase 5a touches **no backend Python**. It is a
frontend-only plan plus a small composer-skill prompt nudge. The plan
covers (a) an empty-state chat-input placeholder, (b) a turn-widget that
makes the LLM's "I created an inline source from your text" decision
explicit and reviewable, (c) a disambiguation widget for ambiguous inputs
("I read 3 URLs — correct?"), (d) a Vitest integration test that mocks the
API and asserts the `set_pipeline` payload shape after a user types data
into chat, and (e) a prompt-side nudge in the composer skill that biases
the LLM toward `inline_blob` for short user inputs.

**Goal.** Land the user-visible affordance described in design doc
[06 §Feature 1](06-chat-as-data-entry.md#feature-1--dynamic-source-from-chat).
After this phase ships, a user can type a URL, a short list, or a single
record into the empty composer chat and arrive at a working source plugin
without picking a file, configuring a schema, or opening the catalog.

**Tech Stack.** React + Zustand + Vitest + testing-library for the
frontend; Markdown for the composer-skill nudge.

**Roadmap reference:** [00-implementation-roadmap.md](00-implementation-roadmap.md)
phase 5a row.

---

## Scope boundaries

**In scope:**

- A context-aware empty-state placeholder for `ChatInput.tsx`. Empty state
  reads "Describe your pipeline, or paste a URL or sample data to start...".
  Active composition state retains the existing "Describe the pipeline you
  want to build..." placeholder.
- A new turn widget `InlineSourceCreatedTurn.tsx` that surfaces, after the
  LLM calls `set_pipeline` with `source.inline_blob`, a reviewable summary
  of what was just created (filename, MIME type, row count or content
  excerpt, blob_id), plus an "edit list" affordance for the LLM-generated
  multi-row case.
- A new turn widget `InlineSourceDisambiguationTurn.tsx` that surfaces
  ambiguous user input ("check these URLs: a.com, b.com, c.com" interpreted
  as "3 rows, one URL per row") for explicit user confirmation BEFORE the
  inline_blob is created. Uses the existing PendingProposalsBanner pattern
  for visual consistency.
- A Vitest integration test that mocks the `set_pipeline` API and asserts
  the request payload shape (`source.inline_blob.filename / .mime_type /
  .content`) is produced from a chat-input dispatch.
- A composer-skill prompt-nudge that biases the LLM toward `inline_blob`
  (vs asking the user to upload a CSV) when the user provides small typed
  data in the conversation. Empirically validated by re-running the LLM
  against the canonical test prompt from
  `project_composer_canonical_test_case` ("create a list of 5 government
  web pages and use an LLM to rate how cool they are"); not gated by a
  string-grep test against the skill markdown (see
  `feedback_no_tests_for_skill_prompts`).
- A risk-mitigating fallback path: a "Create source from this text?"
  affordance shown above the chat input when the LLM has *not* produced
  an `inline_blob` proposal within N turns of detecting source-shaped data
  in the user's most recent message. This handles the case where the LLM
  ignores the prompt nudge.

**Out of scope:**

- **Backend.** No Python changes. `tools.py`, `redaction.py`,
  `data_flow_repository.py`, `hashing.py`, the Landscape schema, and the
  composer service are all untouched.
- **Catalog reshape "Inline data from chat" entry.** Per design doc
  [08-catalog-reshape.md](08-catalog-reshape.md) and the design-spec
  bullet from this scope brief: the catalog entry is owned by Phase 7
  ([16-phase-7-catalog-reshape.md](16-phase-7-catalog-reshape.md)). Phase
  5a only ensures the *affordance* works from chat; Phase 7 is responsible
  for *advertising* it in the catalog drawer.
- **Surface-the-LLM's-interpretation.** Design doc 06 Feature 2 is
  Phase 5b. Phase 5a does NOT touch the interpretation-acceptance event
  shape (open question B2 — undecided as of 2026-05-15).
- **Hello-world tutorial wiring.** Phase 4
  ([04-first-run-tutorial.md](04-first-run-tutorial.md)) consumes this
  feature in turn 2. Phase 5a is the dependency; Phase 4 is the consumer.
  This plan ships first.
- **Audit-readiness panel integration.** Per the scope brief: the audit
  panel's Provenance row should reflect "✓ inline content hashed" when an
  inline source exists. The audit panel itself is Phase 2. If Phase 2 has
  already shipped at planning time, this plan integrates with it (Task 7
  below). If Phase 2 has not shipped, Task 7 is deferred to a Phase 2
  followup ticket and explicitly noted on the umbrella PR.
- **Threshold-cutoff redirection.** Design doc 06 §Disambiguation
  thresholds describes a "that's a lot of rows — paste as CSV?" prompt
  past ~10-20 rows. This is product-tuning territory (open question
  surfaced in 11-§B and not adjudicated as a Phase 5a blocker). Leave to
  Phase 8 polish.

## Trust-tier check (mandatory before any data-handling work)

User-typed chat content is **Tier 3** (external data; zero trust). The
question this plan must answer up front: does the existing `inline_blob`
path validate Tier-3 input at the boundary?

**Answer (verified, do not re-verify):** Yes.

1. The frontend sends `source.inline_blob.content` as a JSON string. JSON
   transport already enforces that the field is a string at the wire
   boundary; non-string content fails JSON parsing.
2. The redaction model `_InlineBlobModel`
   (`redaction.py` line 1022-1048) declares
   `content: Annotated[str, Sensitive(summarizer=...)]` with
   `model_config = ConfigDict(extra="forbid")`. Pydantic enforces both
   the type and the closed-field-set at decode time.
3. `_prepare_blob_create` (called from the `inline_blob is not None`
   branch in `_execute_set_pipeline`, `tools.py` line 4452) validates
   `mime_type` against `_MIME_TO_SOURCE` (an allowlist; unknown MIME types
   produce a `ToolArgumentError`) and produces a SHA-256 hash of the raw
   bytes before storage. The hash is what flows into `source_data_hash`.
4. Post-storage, the inline blob behaves identically to a user-uploaded
   blob — same allowlist, same hashing, same Landscape provenance.

**What Phase 5a must NOT do:** invent Tier-3 validation on the frontend.
The frontend is allowed to assume the backend will reject malformed input
(missing `mime_type`, non-allowlisted MIME, oversize content, etc.) and
surface the rejection to the user via the standard tool-call-error path.
Defensive frontend pre-validation would duplicate the boundary check and
make the contract ambiguous about which side is authoritative. Per
CLAUDE.md trust-tier model: validate **once** at the boundary; the
boundary here is the backend `set_pipeline` handler.

**What Phase 5a MUST do:** propagate backend rejections back to the user
as a visible, repairable turn-level error. The existing `ToolCallCard`
error-rendering surface handles this. No new error UI required.

## Sequencing and dependencies

- **Upstream:** B1 (verified — frontend-only). No phase blocks 5a's start.
- **Downstream:**
  - Phase 4 (hello-world tutorial) consumes this feature. Phase 5a ships
    before Phase 4 is plannable.
  - Phase 5b (interpretation surfacing) layers on top of this feature.
    Phase 5a does not constrain Phase 5b's event shape.
  - Phase 7 (catalog reshape) advertises this feature. Phase 7 references
    "Inline data from chat" but Phase 5a does not implement the catalog
    entry.

- **Optional integration:** Phase 2 (audit-readiness panel) — if shipped
  before Phase 5a's Task 7, Task 7 wires the inline-source provenance row
  into the panel. If Phase 2 ships *after* Phase 5a, Task 7 is deferred
  and Phase 2's planner adds the wire-up to its scope.

## Verification approach

Each task is TDD-shaped: write the failing test → run it red → implement
→ run it green → commit. One exception: the composer-skill prompt nudge
(Task 8) is not test-gated against grep — skill files are LLM prompts,
not code (`feedback_no_tests_for_skill_prompts`). Empirical validation is
required: re-run the canonical test prompt through the live LLM and
verify the LLM picks `inline_blob` for short user inputs.

**Manual smoke at the end of the plan (Task 9):**

1. Start a fresh session on staging (`elspeth.foundryside.dev` per
   `project_staging_deployment`).
2. Confirm the empty-state placeholder reads "Describe your pipeline, or
   paste a URL or sample data to start...".
3. Type "go to www.finance.gov.au" → confirm the LLM creates an
   inline_blob source and the new turn widget surfaces what was created.
4. Type "check these URLs: a.com, b.com, c.com" → confirm the
   disambiguation widget appears and offers row-count confirmation.
5. Type the canonical hero prompt "create a list of 5 government web
   pages and use an LLM to rate how cool they are" → confirm the LLM
   generates the 5 URLs and presents them via the new turn widget for
   user review before the pipeline is finalised.

Verification is complete when (a) all Vitest suites pass, (b) staging
smoke passes, and (c) the canonical hero prompt produces an inline source
end-to-end without a CSV upload step.

---

## File structure (what changes in this phase)

```text
src/elspeth/web/composer/skills/
  pipeline_composer.md                                          MODIFY    (Task 8)

src/elspeth/web/frontend/src/
  components/chat/
    ChatInput.tsx                                               MODIFY    (Task 1)
    ChatInput.test.tsx                                          MODIFY    (Task 1)
    InlineSourceCreatedTurn.tsx                                 CREATE    (Task 3)
    InlineSourceCreatedTurn.test.tsx                            CREATE    (Task 3)
    InlineSourceDisambiguationTurn.tsx                          CREATE    (Task 4)
    InlineSourceDisambiguationTurn.test.tsx                     CREATE    (Task 4)
    InlineSourceFallbackPrompt.tsx                              CREATE    (Task 5)
    InlineSourceFallbackPrompt.test.tsx                         CREATE    (Task 5)
    ChatPanel.tsx                                               MODIFY    (Tasks 3/4/5)
  stores/
    inlineSourceStore.ts                                        CREATE    (Task 2)
    inlineSourceStore.test.ts                                   CREATE    (Task 2)
  types/
    api.ts                                                      MODIFY    (Task 2)
    index.ts                                                    MODIFY    (Task 2)
  api/
    client.ts                                                   MODIFY    (Task 6 — integration test fixture)
  test/
    inlineSourceIntegration.test.tsx                            CREATE    (Task 6)

src/elspeth/web/frontend/src/components/audit/                  (Phase 2 surface; see Task 7)
  AuditReadinessPanel.tsx                                       MODIFY    (Task 7 — only if Phase 2 shipped)

docs/composer/ux-redesign-2026-05/
  17-phase-5a-dynamic-source-from-chat.md                       THIS FILE
```

No backend files change. No Landscape schema migration. No tool-surface
extension.

---

## Task 1 — Empty-state chat-input placeholder

**Goal.** Change the `ChatInput.tsx` placeholder to read "Describe your
pipeline, or paste a URL or sample data to start..." when the chat has
zero non-system messages and no active composition state. Revert to the
existing wording when either condition flips.

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.tsx` —
  add a derived `effectivePlaceholder` that consults the session's
  message count and composition-state version.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatInput.test.tsx`
  — add two new test cases: (a) empty session shows the new placeholder;
  (b) session with at least one user message OR a non-zero composition
  state version shows the original placeholder.

### Step 1 — Write the failing test (RED)

Open `src/elspeth/web/frontend/src/components/chat/ChatInput.test.tsx`
and add a new `describe("empty-state placeholder", ...)` block. The test
must render `ChatInput` inside the same provider stack the existing
tests use, then assert against the textarea's `placeholder` attribute.

The session-state inputs the component now reads:

- `sessionStore.activeSessionId` — already imported.
- `sessionStore.messages.length` — new read (messages is a flat
  `ChatMessage[]` array on the store singleton, not keyed by session).
- `sessionStore.compositionState?.version ?? 0` — new read
  (`compositionState` is `CompositionState | null` on the store
  singleton, not per-session-keyed).

Verified against `src/elspeth/web/frontend/src/stores/sessionStore.ts`
lines 153–154: `messages: ChatMessage[]` and
`compositionState: CompositionState | null`.

Pseudo-shape of the new test cases:

```typescript
describe("ChatInput empty-state placeholder", () => {
  it("shows the data-priming placeholder when the session has no messages and no composition state", () => {
    // arrange: fresh session, messages=[], version=0
    // act: render ChatInput
    // assert: textarea placeholder == "Describe your pipeline, or paste a URL or sample data to start..."
  });

  it("reverts to the standard placeholder once the user has sent a message", () => {
    // arrange: messages=[{role:"user", content:"hi"}], version=0
    // assert: textarea placeholder == "Describe the pipeline you want to build..."
  });

  it("reverts to the standard placeholder once a composition state exists", () => {
    // arrange: messages=[], version=1
    // assert: textarea placeholder == "Describe the pipeline you want to build..."
  });

  it("respects an explicit `placeholder` prop override even in empty state", () => {
    // arrange: messages=[], version=0
    // act: render with placeholder="custom"
    // assert: textarea placeholder == "custom"
  });
});
```

The last case is the contract with Phase A slice 4 (guided-mode per-step
nudge — already in production); the existing `placeholder` prop must
continue to win when supplied.

### Step 2 — Run RED

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/ChatInput.test.tsx
```

Expected: four new test cases fail; existing cases still pass.

### Step 3 — Implement

In `ChatInput.tsx`:

1. Add two new selectors near the existing `activeSessionId` read:

   ```typescript
   // sessionStore.messages is ChatMessage[] (singleton, not per-session-keyed).
   const messageCount = useSessionStore((s) => s.messages.length);
   // sessionStore.compositionState is CompositionState | null (singleton).
   const compositionVersion = useSessionStore(
     (s) => s.compositionState?.version ?? 0,
   );
   ```

   These selectors are correct for the existing store shape (verified
   against `sessionStore.ts` lines 153–154). Do NOT add new store
   fields just for this — the information already exists.

2. Derive the effective placeholder once, near the existing `canSend`
   computation:

   ```typescript
   const isEmptyState = messageCount === 0 && compositionVersion === 0;
   const defaultPlaceholder = isEmptyState
     ? "Describe your pipeline, or paste a URL or sample data to start..."
     : "Describe the pipeline you want to build...";
   const effectivePlaceholder = placeholder ?? defaultPlaceholder;
   ```

3. Replace the existing
   `placeholder={placeholder ?? "Describe the pipeline you want to build..."}`
   on the `<textarea>` with `placeholder={effectivePlaceholder}`.

The `placeholder?:` prop override remains the explicit highest-precedence
control point — Phase A slice 4's per-step nudges keep working unchanged.

### Step 4 — Run GREEN

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/ChatInput.test.tsx
```

Expected: all `ChatInput` tests pass, including the four new ones.

### Step 5 — Commit

```bash
git add src/elspeth/web/frontend/src/components/chat/ChatInput.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatInput.test.tsx
git commit -m "feat(composer/chat): empty-state placeholder primes inline source-from-chat (Phase 5a.1)"
```

---

## Task 2 — `inlineSourceStore` (Zustand) for derived inline-source state

**Goal.** A small Zustand store that owns, per session, the derived
"current pipeline state has an inline source" view. This is the data
backing the new turn widgets and the audit-panel integration. It is
purely derived from `compositionState` and `messages`; the store is a
caching/projection layer, not a source of truth.

**Why a store, not a hook?** Three consumers (the two new turn widgets
plus the optional Phase 2 audit-panel integration) need the same derived
view across the component tree. A hook would either duplicate the
derivation logic or require prop-drilling. A Zustand store matches the
existing pattern (`preferencesStore`, `sessionStore`, etc.).

**Files:**

- Create: `src/elspeth/web/frontend/src/stores/inlineSourceStore.ts`.
- Create: `src/elspeth/web/frontend/src/stores/inlineSourceStore.test.ts`.
- Modify: `src/elspeth/web/frontend/src/types/api.ts` — add the new
  `InlineSourceSummary` type re-export.
- Modify: `src/elspeth/web/frontend/src/types/index.ts` — add the new
  `InlineSourceSummary` type definition.

### Step 1 — Define the type

In `types/index.ts`, add:

```typescript
/**
 * Frontend-derived projection of an inline-blob source attached to the
 * current composition state. Computed from compositionState.source +
 * blob metadata. Never persisted; recomputed on each composition mutation.
 */
export interface InlineSourceSummary {
  blobId: string;
  filename: string;
  mimeType: string;
  /** Truncated content excerpt for display; never the full payload. */
  contentPreview: string;
  /** Best-effort row count from the parsed source; null if unparseable. */
  rowCount: number | null;
  /** SHA-256 of the raw inline content (from session blob metadata). */
  contentHash: string;
  /**
   * Whether the rows in this inline source were typed verbatim by the
   * user ("verbatim") or generated by the LLM and confirmed
   * ("llm-generated") or interpreted ambiguously and confirmed by the
   * user ("disambiguated"). Drives which review affordance the turn
   * widget surfaces.
   */
  provenance: "verbatim" | "llm-generated" | "disambiguated";
}
```

Then in `types/api.ts`, append `InlineSourceSummary` to the existing
`export type { ... } from "./index";` list.

### Step 2 — Write the failing test (RED)

Create `stores/inlineSourceStore.test.ts`:

```typescript
import { describe, it, expect, beforeEach } from "vitest";
import { useInlineSourceStore } from "./inlineSourceStore";
import { resetStore } from "@/test/store-helpers";

describe("inlineSourceStore", () => {
  beforeEach(() => resetStore(useInlineSourceStore));

  it("returns null when no inline source is bound to the session", () => {
    expect(useInlineSourceStore.getState().getSummary("session-1")).toBeNull();
  });

  it("stores a verbatim summary and retrieves it by session", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-uuid",
      filename: "chat.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://example.com",
      rowCount: 1,
      contentHash: "abc123",
      provenance: "verbatim",
    });
    const summary = useInlineSourceStore.getState().getSummary("session-1");
    expect(summary?.provenance).toBe("verbatim");
    expect(summary?.rowCount).toBe(1);
  });

  it("clears the summary when the source is replaced or removed", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-clear-1",
      filename: "clear.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://example.com",
      rowCount: 1,
      contentHash: "deadbeef01",
      provenance: "verbatim",
    });
    useInlineSourceStore.getState().clearSummary("session-1");
    expect(useInlineSourceStore.getState().getSummary("session-1")).toBeNull();
  });

  it("namespaces summaries per session", () => {
    useInlineSourceStore.getState().setSummary("session-1", {
      blobId: "blob-ns-1",
      filename: "verbatim.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://gov.au",
      rowCount: 1,
      contentHash: "aabbcc0011",
      provenance: "verbatim",
    });
    useInlineSourceStore.getState().setSummary("session-2", {
      blobId: "blob-ns-2",
      filename: "llm-gen.csv",
      mimeType: "text/csv",
      contentPreview: "url\nhttps://a.gov\nhttps://b.gov",
      rowCount: 2,
      contentHash: "112233aabb",
      provenance: "llm-generated",
    });
    expect(useInlineSourceStore.getState().getSummary("session-1")?.provenance).toBe("verbatim");
    expect(useInlineSourceStore.getState().getSummary("session-2")?.provenance).toBe("llm-generated");
  });
});
```

### Step 3 — Run RED

```bash
cd src/elspeth/web/frontend
npx vitest run src/stores/inlineSourceStore.test.ts
```

Expected: fail with module-not-found.

### Step 4 — Implement

```typescript
// src/elspeth/web/frontend/src/stores/inlineSourceStore.ts
import { create } from "zustand";
import type { InlineSourceSummary } from "@/types/api";

interface InlineSourceState {
  summariesBySession: Record<string, InlineSourceSummary>;
  setSummary: (sessionId: string, summary: InlineSourceSummary) => void;
  clearSummary: (sessionId: string) => void;
  getSummary: (sessionId: string) => InlineSourceSummary | null;
}

export const useInlineSourceStore = create<InlineSourceState>((set, get) => ({
  summariesBySession: {},
  setSummary: (sessionId, summary) =>
    set((s) => ({
      summariesBySession: { ...s.summariesBySession, [sessionId]: summary },
    })),
  clearSummary: (sessionId) =>
    set((s) => {
      const next = { ...s.summariesBySession };
      delete next[sessionId];
      return { summariesBySession: next };
    }),
  getSummary: (sessionId) => get().summariesBySession[sessionId] ?? null,
}));
```

The derivation that populates this store (reading `compositionState` and
the session's blob metadata to build the `InlineSourceSummary`) lives in
the ChatPanel wiring landed in Task 3 below — the store itself is a
plain projection container.

### Step 5 — Run GREEN

```bash
cd src/elspeth/web/frontend
npx vitest run src/stores/inlineSourceStore.test.ts
```

Expected: all four tests pass.

### Step 6 — Commit

```bash
git add src/elspeth/web/frontend/src/types/index.ts \
        src/elspeth/web/frontend/src/types/api.ts \
        src/elspeth/web/frontend/src/stores/inlineSourceStore.ts \
        src/elspeth/web/frontend/src/stores/inlineSourceStore.test.ts
git commit -m "feat(composer/frontend): inlineSourceStore for projected inline-source view (Phase 5a.2)"
```

---

## Task 3 — `InlineSourceCreatedTurn.tsx` widget

**Goal.** When `set_pipeline` returns successfully and the resulting
composition state has an `inline_blob`-derived source, surface a
reviewable turn-widget that shows the user what was created (filename,
MIME, row count, content excerpt) and offers an "Edit the list" action
when the source was LLM-generated.

The widget appears inline in the chat (not in the proposal banner) — it
is an *informational* surface that confirms a completed mutation, not a
pending-approval surface. The proposal banner already handles pending
mutations; this widget handles the post-success review of an inline
source specifically.

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceCreatedTurn.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceCreatedTurn.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` —
  derive the `InlineSourceSummary` and feed it into the new component
  when `compositionState.source` has `blob_ref` and the blob's
  `created_via` metadata indicates inline-blob origin.

### Step 1 — Write the failing test (RED)

```typescript
// InlineSourceCreatedTurn.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InlineSourceCreatedTurn } from "./InlineSourceCreatedTurn";

describe("InlineSourceCreatedTurn", () => {
  const verbatim = {
    blobId: "b1",
    filename: "chat.csv",
    mimeType: "text/csv",
    contentPreview: "url\nhttps://finance.gov.au",
    rowCount: 1,
    contentHash: "h1",
    provenance: "verbatim" as const,
  };

  const llmGenerated = {
    ...verbatim,
    blobId: "b2",
    contentPreview: "url\n...gov.au\n...gov.au\n...gov.au\n...gov.au\n...gov.au",
    rowCount: 5,
    provenance: "llm-generated" as const,
  };

  it("renders the filename, MIME type, and row count for a verbatim source", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(screen.getByText(/chat\.csv/)).toBeInTheDocument();
    expect(screen.getByText(/text\/csv/)).toBeInTheDocument();
    expect(screen.getByText(/1 row/)).toBeInTheDocument();
  });

  it("does NOT show an Edit button for verbatim provenance", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /edit the list/i })).toBeNull();
  });

  it("DOES show an Edit button for llm-generated provenance", () => {
    const onEdit = vi.fn();
    render(<InlineSourceCreatedTurn summary={llmGenerated} onEdit={onEdit} />);
    const button = screen.getByRole("button", { name: /edit the list/i });
    fireEvent.click(button);
    expect(onEdit).toHaveBeenCalledWith(llmGenerated);
  });

  it("renders the SHA-256 hash as a small audit signal", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(screen.getByText(/h1/)).toBeInTheDocument();
  });

  it("renders the content preview clipped (no full-payload leak in DOM)", () => {
    const huge = {
      ...verbatim,
      contentPreview: "x".repeat(500),
    };
    render(<InlineSourceCreatedTurn summary={huge} onEdit={vi.fn()} />);
    const preview = screen.getByTestId("inline-source-preview");
    expect(preview.textContent?.length).toBeLessThanOrEqual(280);
  });
});
```

The 280-character clip threshold is illustrative — pick whatever is
visually reasonable; the test just enforces that there IS a clip.

### Step 2 — Run RED

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceCreatedTurn.test.tsx
```

Expected: fail with module-not-found.

### Step 3 — Implement

Component shape (clip preview to ~280 chars; show filename, MIME, row
count, full SHA-256 hash in a small audit signal; render an "Edit the
list" button only when `provenance === "llm-generated"`; wire `onEdit`
to a callback that opens the inline editor). Use a `data-testid` of
`inline-source-preview` on the clipped preview element so the Task 3
test's clip assertion has a stable anchor.

In `ChatPanel.tsx`, derive the `InlineSourceSummary` from the current
composition state's source and the session blob metadata, push it into
`inlineSourceStore`, and render the widget at the appropriate position
in the message stream. Mirror the existing pattern used by
`PendingProposalsBanner`. The exact wiring depends on the message-stream
shape — surface the widget as a *system message* in the stream when the
inline source is created, so it scrolls with the rest of the
conversation.

For "Edit the list" — the v1 implementation opens a textarea pre-filled
with the current inline content and offers a Save action that triggers a
new `set_pipeline` call with a revised `inline_blob.content`. The editor
itself can be a simple modal; defer richer table-style editing to Phase 8.

### Step 4 — Run GREEN

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceCreatedTurn.test.tsx
npx vitest run src/components/chat/ChatPanel.test.tsx
```

Expected: green.

### Step 5 — Commit

```bash
git add src/elspeth/web/frontend/src/components/chat/InlineSourceCreatedTurn.tsx \
        src/elspeth/web/frontend/src/components/chat/InlineSourceCreatedTurn.test.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
git commit -m "feat(composer/chat): InlineSourceCreatedTurn surfaces inline-source provenance (Phase 5a.3)"
```

---

## Task 4 — `InlineSourceDisambiguationTurn.tsx` widget

**Goal.** When the LLM is interpreting ambiguous user input (e.g., "check
these URLs: a.com, b.com, c.com" → 3 rows, one URL each), present the
interpretation for explicit confirmation BEFORE creating the inline_blob.
This is the user-facing realisation of design doc 06's "mixed / inferred"
mode.

The mechanism: the LLM proposes the inline source via the existing
`composition_proposal` channel. Phase 5a does NOT extend the proposal
shape — it adds a specialised UI for proposals whose `source.inline_blob`
field is set AND whose `interpretation` annotation indicates ambiguous
input. (If the proposal channel doesn't yet carry an `interpretation`
annotation, Phase 5a leaves it as a regular pending proposal — the
disambiguation turn is then a "nice to have" overlay that surfaces only
when the LLM's narration says "I read this as N rows"; otherwise the
standard PendingProposalsBanner handles it.)

The widget is a richer per-proposal UI than the standard banner: it
shows the user's original input, the LLM's parsed row breakdown, and
offers "Yes, that's right", "No, treat as 1 row", or "Edit the rows
directly" actions.

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceDisambiguationTurn.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceDisambiguationTurn.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` —
  route inline_blob-bearing proposals through the new widget instead of
  the standard banner when the proposal's content is ambiguous.

### Step 1 — Write the failing test (RED)

```typescript
// InlineSourceDisambiguationTurn.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InlineSourceDisambiguationTurn } from "./InlineSourceDisambiguationTurn";

describe("InlineSourceDisambiguationTurn", () => {
  const props = {
    userInput: "check these URLs: a.com, b.com, c.com",
    proposedRows: ["a.com", "b.com", "c.com"],
    proposalId: "p1",
    onConfirmMultiRow: vi.fn(),
    onTreatAsOneRow: vi.fn(),
    onEditRows: vi.fn(),
  };

  it("renders the user's original input verbatim", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    expect(screen.getByText(/check these URLs: a\.com, b\.com, c\.com/)).toBeInTheDocument();
  });

  it("shows the LLM's row breakdown with one row per item", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    expect(screen.getByText("a.com")).toBeInTheDocument();
    expect(screen.getByText("b.com")).toBeInTheDocument();
    expect(screen.getByText("c.com")).toBeInTheDocument();
  });

  it("calls onConfirmMultiRow when the confirm button is clicked", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /yes.*3 rows/i }));
    expect(props.onConfirmMultiRow).toHaveBeenCalledWith("p1");
  });

  it("calls onTreatAsOneRow when the single-row button is clicked", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /treat as 1 row/i }));
    expect(props.onTreatAsOneRow).toHaveBeenCalledWith("p1");
  });

  it("calls onEditRows when the edit button is clicked", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    fireEvent.click(screen.getByRole("button", { name: /edit the rows/i }));
    expect(props.onEditRows).toHaveBeenCalledWith("p1");
  });

  it("announces itself via role=region with an aria-label", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    expect(screen.getByRole("region", { name: /row count/i })).toBeInTheDocument();
  });
});
```

### Step 2 — Run RED

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceDisambiguationTurn.test.tsx
```

Expected: fail with module-not-found.

### Step 3 — Implement

Component shape: a `<section role="region" aria-label="Confirm row count
interpretation (N rows)">` containing the user's original input
verbatim (in a `<blockquote>`), an ordered list of the LLM's parsed
rows, and three action buttons — `Yes — N rows` (primary; calls
`onConfirmMultiRow(proposalId)`), `No — treat as 1 row`
(`onTreatAsOneRow`), and `Edit the rows` (`onEditRows`). The `role` /
`aria-label` and per-button accessible names are load-bearing for the
Task 4 tests; keep them stable.

Wiring in `ChatPanel.tsx`:

1. When iterating proposals, detect proposals where the source change has
   an `inline_blob` AND the proposal's narration / annotation indicates
   row-count ambiguity. The detection heuristic v1: if the proposal's
   `summary` contains the phrase "I read" or "interpreted as", treat as
   ambiguous; otherwise fall back to the standard banner. (This is
   intentionally heuristic — Phase 5b will design a structured field for
   the annotation; until then, the heuristic is fine and the false
   positives are mild.)
2. For ambiguous proposals, render `InlineSourceDisambiguationTurn`
   inline at the proposal's position in the message stream, NOT in the
   banner. Wire `onConfirmMultiRow` → existing `acceptCompositionProposal`,
   `onTreatAsOneRow` → reject the current proposal then send an LLM
   message "treat the original input as a single row", `onEditRows` →
   open the inline editor from Task 3.

### Step 4 — Run GREEN

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceDisambiguationTurn.test.tsx
npx vitest run src/components/chat/ChatPanel.test.tsx
```

### Step 5 — Commit

```bash
git add src/elspeth/web/frontend/src/components/chat/InlineSourceDisambiguationTurn.tsx \
        src/elspeth/web/frontend/src/components/chat/InlineSourceDisambiguationTurn.test.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
git commit -m "feat(composer/chat): InlineSourceDisambiguationTurn for ambiguous row count (Phase 5a.4)"
```

---

## Task 5 — `InlineSourceFallbackPrompt.tsx` (LLM-skip recovery)

**Goal.** Mitigate the risk that the LLM ignores the prompt nudge from
Task 8 and fails to produce an `inline_blob` for short user inputs. If
the user types source-shaped data (a URL, a short list, a record) and
the LLM responds without proposing a source for N (=2) turns, surface a
small affordance: "I haven't created a source from your text yet. Create
one now?".

This is a *floor* on the affordance — it ensures the user can always
reach the inline-source path even if the prompt nudge proves
insufficient.

**Detection specification (precise — bias toward false negatives):**

The fallback prompt fires when ALL of the following are true:

1. The user has sent ≥1 chat message in the current session.
2. The composer has NOT yet bound a source via the LLM proposal flow
   (`compositionState.source === null` or
   `compositionState.source.plugin === ""`).
3. The user's latest message (most recent `role === "user"` message)
   contains exactly one of:
   - A URL (matches `/https?:\/\//`).
   - A comma-separated list of 2–10 items (matches
     `/[^,]+(?:, [^,]+){1,9}/`).
   - A single short typed phrase under 200 chars containing no
     questions (no `?` character).
4. No "source-related" tool call (`set_pipeline`, `set_source_from_blob`,
   `set_source`) is currently in flight (i.e., the most recent assistant
   message does not contain one of these tool-call names in its
   `toolCalls`).

This specification is conservative — bias toward false negatives (don't
show the prompt) over false positives. An affordance that appears only
when genuinely warranted is more trustworthy than one that appears on
ambiguous inputs.

If all four hold, render the fallback prompt above the chat input.

**Files:**

- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceFallbackPrompt.tsx`.
- Create: `src/elspeth/web/frontend/src/components/chat/InlineSourceFallbackPrompt.test.tsx`.
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` —
  derive the predicate and render the fallback prompt when it's true.

### Step 1 — Failing test (RED)

```typescript
// InlineSourceFallbackPrompt.test.tsx
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { InlineSourceFallbackPrompt } from "./InlineSourceFallbackPrompt";

describe("InlineSourceFallbackPrompt", () => {
  it("does not render when the predicate is false", () => {
    const { container } = render(
      <InlineSourceFallbackPrompt
        shouldRender={false}
        candidateText="https://example.com"
        onAccept={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders the call-to-action when the predicate is true", () => {
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={vi.fn()}
        onDismiss={vi.fn()}
      />,
    );
    expect(screen.getByText(/haven't created a source/i)).toBeInTheDocument();
  });

  it("calls onAccept with the candidate text when the user accepts", () => {
    const onAccept = vi.fn();
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="https://example.com"
        onAccept={onAccept}
        onDismiss={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /create source/i }));
    expect(onAccept).toHaveBeenCalledWith("https://example.com");
  });

  it("calls onDismiss when the user dismisses", () => {
    const onDismiss = vi.fn();
    render(
      <InlineSourceFallbackPrompt
        shouldRender={true}
        candidateText="x"
        onAccept={vi.fn()}
        onDismiss={onDismiss}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));
    expect(onDismiss).toHaveBeenCalled();
  });
});
```

### Step 2 — RED

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceFallbackPrompt.test.tsx
```

### Step 3 — Implement

Component shape: `shouldRender === false` ⇒ render nothing
(`return null`). Otherwise render a `<section role="region"
aria-label="Inline source fallback prompt">` with prose "I haven't
created a source from your text yet. Would you like me to treat your
message as the source data?" and two buttons — "Create source from
this text" (primary; calls `onAccept(candidateText)`) and "Dismiss"
(link-style; calls `onDismiss`).

Predicate derivation lives in `ChatPanel.tsx`:

```typescript
const candidate = useMemo(() => {
  // Walk the last 3 user messages; return the first one whose content
  // matches data-shape regex; null if none.
  const userMsgs = messages.filter((m) => m.role === "user").slice(-3);
  for (const m of userMsgs.reverse()) {
    if (looksLikeData(m.content)) return m.content;
  }
  return null;
}, [messages]);

const hasInflightSourceCall = useMemo(() => {
  const lastAssistant = messages.filter((m) => m.role === "assistant").at(-1);
  return lastAssistant?.toolCalls?.some((tc) =>
    ["set_pipeline", "set_source_from_blob", "set_source"].includes(tc.name)
  ) ?? false;
}, [messages]);

const compositionHasSource = compositionState?.source?.plugin !== undefined &&
                              compositionState?.source?.plugin !== "";

const shouldRender =
  Boolean(candidate) && !hasInflightSourceCall && !compositionHasSource;

const onAccept = (text: string) => {
  // Send a synthetic user message asking the LLM to create an
  // inline_blob source from `text`. The LLM still does the structured
  // mutation — the user's click is interpreted, not the source content.
  sendMessage(
    `Please create the source plugin from this text using set_pipeline ` +
    `with source.inline_blob:\n\n${text}`
  );
};
```

When the user accepts, dispatch a synthetic chat turn that *asks* the
LLM to create the inline source. We deliberately keep the LLM in the
loop — the frontend does not construct the `set_pipeline` payload
itself, because the LLM also needs to choose the plugin (csv vs json vs
url-list etc.) and add the other transforms. This is the safer shape
than the frontend forcing a specific payload.

### Step 4 — GREEN

```bash
cd src/elspeth/web/frontend
npx vitest run src/components/chat/InlineSourceFallbackPrompt.test.tsx
npx vitest run src/components/chat/ChatPanel.test.tsx
```

### Step 5 — Commit

```bash
git add src/elspeth/web/frontend/src/components/chat/InlineSourceFallbackPrompt.tsx \
        src/elspeth/web/frontend/src/components/chat/InlineSourceFallbackPrompt.test.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
git commit -m "feat(composer/chat): InlineSourceFallbackPrompt as LLM-skip safety net (Phase 5a.5)"
```

---

## Task 6 — Vitest integration test: chat input → `set_pipeline` payload

**Goal.** A single Vitest test that asserts the frontend dispatches the
chat message via `sendMessage` with the user's text, and renders
`InlineSourceCreatedTurn` from a mocked tool-call response containing the
`source.inline_blob` shape. The frontend test does not exercise the LLM
round-trip; that path is covered by integration tests in plan 18b.

This is **not** a full LLM integration test (which would be a slow
live-API test gated to nightly). It mocks the LLM response with a
canned `set_pipeline` tool call and verifies the frontend handles the
end-to-end loop correctly: chat input dispatch → mocked LLM response
with `inline_blob` tool call → tool result rendered → composition state
updated → `InlineSourceCreatedTurn` rendered.

**Files:**

- Create: `src/elspeth/web/frontend/src/test/inlineSourceIntegration.test.tsx`.

### Step 1 — Write the failing test (RED)

```typescript
// inlineSourceIntegration.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import * as client from "@/api/client";
import { App } from "@/App";  // or whichever top-level renders ChatPanel

describe("Phase 5a integration: chat input → inline_blob → InlineSourceCreatedTurn", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("dispatches a user message that produces an inline_blob source and renders the created turn", async () => {
    // Mock sendMessage to return a canned tool-call response simulating the LLM.
    const sendMessage = vi.spyOn(client, "sendMessage").mockResolvedValue({
      message: {
        id: "asst-1",
        role: "assistant",
        content: "Created a source from your text.",
        toolCalls: [
          {
            name: "set_pipeline",
            arguments: {
              source: {
                plugin: "csv",
                on_success: "continue",
                inline_blob: {
                  filename: "chat.csv",
                  mime_type: "text/csv",
                  content: "url\nhttps://finance.gov.au",
                  description: "Inline source from chat",
                },
              },
              nodes: [],
              outputs: [{ name: "results", plugin: "csv", options: {} }],
            },
            result: { success: true },
          },
        ],
      },
      compositionState: {
        id: "state-1",
        version: 1,
        source: {
          plugin: "csv",
          options: {
            path: "/sessions/.../blobs/.../chat.csv",
            blob_ref: "blob-uuid",
          },
        },
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: null, description: null },
      },
    });

    // Mock blob metadata fetch so the inline-source projection populates.
    vi.spyOn(client, "fetchBlob").mockResolvedValue({
      id: "blob-uuid",
      filename: "chat.csv",
      mime_type: "text/csv",
      content_hash: "sha256-...",
      created_via: "inline_blob",
      size_bytes: 22,
    });

    // Auth is provided via vi.mock("./hooks/useAuth", ...) at module scope
    // (mirrors App.test.tsx lines 81-97: useAuth returns isAuthenticated=true,
    // user = { user_id: "test-001", username: "test-operator", ... }).
    // The vi.mock call must be placed at the top of the test file alongside
    // the other module-scope vi.mock calls; do not place it inside the it() body.
    vi.spyOn(client, "createSession").mockResolvedValue({
      id: "session-1",
      title: null,
      mode: "freeform",
      created_at: "2026-05-16T00:00:00Z",
      updated_at: "2026-05-16T00:00:00Z",
    });

    render(<App />);

    const textarea = await screen.findByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "go to https://finance.gov.au" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    // Assert the API was called.
    await waitFor(() =>
      expect(sendMessage).toHaveBeenCalledWith(
        expect.any(String),  // sessionId
        expect.objectContaining({
          content: "go to https://finance.gov.au",
        }),
      ),
    );

    // Assert the InlineSourceCreatedTurn rendered with the right summary.
    await waitFor(() => {
      expect(screen.getByText(/Source created from your message/)).toBeInTheDocument();
      expect(screen.getByText(/chat\.csv/)).toBeInTheDocument();
      expect(screen.getByText(/text\/csv/)).toBeInTheDocument();
    });
  });
});
```

The exact mock-shape depends on the real `sendMessage` return type — fit
the test to the wire contract documented in the existing
`client.ts`/`sessionStore.ts`. If the wire shape lacks a `created_via`
metadata field on blobs, the inline-source-detection heuristic for the
projection (Task 3 wiring) is "the source's `blob_ref` matches a blob
whose creator was a `set_pipeline` call with `inline_blob`" — derived
from the session's tool-call history rather than from blob metadata. Use
whichever signal already exists; don't add a new backend field.

### Step 2 — RED, then GREEN, then commit

```bash
cd src/elspeth/web/frontend
npx vitest run src/test/inlineSourceIntegration.test.tsx
# expect RED, implement projection / wiring, then GREEN
git add src/elspeth/web/frontend/src/test/inlineSourceIntegration.test.tsx \
        src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx
git commit -m "test(composer/frontend): integration test for chat → inline_blob → review widget (Phase 5a.6)"
```

---

## Task 7 — Audit-readiness panel surface (CONDITIONAL on Phase 2 shipped)

**Goal.** The audit-readiness panel (Phase 2) has a Provenance row.
When an inline source is bound to the current composition state, the
row should display "✓ Inline content hashed (SHA-256: <prefix>...)"
instead of (or in addition to) the default "Source not configured" /
"File: <filename>" treatments.

**Conditional execution rule:**

- **If Phase 2 has shipped before Phase 5a planning** (i.e.,
  `AuditReadinessPanel.tsx` exists in `main`): Task 7 is in scope.
  Implement and ship as part of Phase 5a's umbrella PR.
- **If Phase 2 has NOT shipped:** Task 7 is deferred. Open a follow-up
  ticket "wire inline-source provenance into audit-readiness panel"
  and link it to the Phase 2 umbrella PR. Mark Phase 5a's umbrella PR
  with a note "Task 7 deferred — depends on Phase 2".

Determine the branch at planning time by checking whether
`src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx`
exists in the current branch. If it does, Task 7 is in scope.

### Step 1 — Failing test (RED) — only if Phase 2 shipped

```typescript
// AuditReadinessPanel.test.tsx (modify existing)
it("renders inline-content-hashed provenance row when the source is inline_blob-backed", () => {
  vi.spyOn(inlineSourceStoreModule, "useInlineSourceStore").mockImplementation(
    (selector) => selector({
      getSummary: (_id: string) => ({
        blobId: "blob-uuid",
        filename: "chat.csv",
        mimeType: "text/csv",
        contentPreview: "url\nhttps://finance.gov.au",
        rowCount: 1,
        contentHash: "abc123def456",
        provenance: "verbatim" as const,
      }),
      setSummary: vi.fn(),
      clearSummary: vi.fn(),
    } as ReturnType<typeof inlineSourceStoreModule.useInlineSourceStore>),
  );
  render(<AuditReadinessPanel sessionId="session-1" />);
  expect(screen.getByText(/inline content hashed/i)).toBeInTheDocument();
  expect(screen.getByText(/abc123/)).toBeInTheDocument();
});

it("renders the default file-based provenance row when the source is not inline_blob-backed", () => {
  vi.spyOn(inlineSourceStoreModule, "useInlineSourceStore").mockImplementation(
    (selector) => selector({
      getSummary: (_id: string) => null,
      setSummary: vi.fn(),
      clearSummary: vi.fn(),
    } as ReturnType<typeof inlineSourceStoreModule.useInlineSourceStore>),
  );
  render(<AuditReadinessPanel sessionId="session-1" />);
  expect(screen.getByText(/file:/i)).toBeInTheDocument();
});
```

### Step 2 — Implement

In `AuditReadinessPanel.tsx`, add a `useInlineSourceStore` selector for
the active session. If non-null, render the inline-hashed provenance
row; otherwise fall back to the existing file-based treatment. Import
`useSessionStore` from `@/stores/sessionStore` if it is not already
imported in this component.

```typescript
const provenanceSource = useSessionStore((s) => s.compositionState?.source ?? null);
const inlineSummary = useInlineSourceStore((s) => s.getSummary(activeSessionId));
const provenanceRowContent = inlineSummary
  ? (
    <>
      <span aria-hidden="true">✓</span>
      Inline content hashed (SHA-256: {inlineSummary.contentHash.slice(0, 12)}…)
    </>
  )
  : (
    <>
      <span aria-hidden="true">📄</span>
      File: {provenanceSource?.options?.path ?? "—"}
    </>
  );
```

### Step 3 — GREEN, commit

```bash
git add src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx \
        src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx
git commit -m "feat(composer/audit-panel): inline-source provenance row (Phase 5a.7)"
```

---

## Task 8 — Composer-skill prompt nudge (LLM-side)

**Goal.** Bias the LLM toward using `set_pipeline` with
`source.inline_blob` (vs prompting the user to upload a CSV) when the
user provides small typed data in the conversation. The skill already
mentions `inline_blob` in its decision rule (line 491-496 of
`pipeline_composer.md`); this task strengthens the bias for the
empty-state / data-typed-into-chat path.

**Important constraint** (per `feedback_no_tests_for_skill_prompts`):
the skill is an LLM prompt, not code. Grepping the skill text for
specific phrases is theatre. The validation gate is empirical:
re-run the canonical test prompts through the live LLM and verify the
LLM picks `inline_blob` for short user inputs.

**Validation prompts (run all three after the edit lands):**

1. `"go to www.finance.gov.au"` → expect a `set_pipeline` call with
   `source.inline_blob`. Pass if true; fail if the LLM asks the user
   to upload a CSV or proposes anything else for the source.
2. `"check these URLs: a.com, b.com, c.com"` → expect EITHER a
   `set_pipeline` call with a 3-row `inline_blob`, OR a narration that
   surfaces the row-count interpretation before any tool call. Pass on
   either.
3. (Canonical hero) `"create a list of 5 government web pages and use
   an LLM to rate how cool they are"` — expect the LLM to generate 5
   URLs and create an `inline_blob` source from them. The LLM-generated
   case. Pass if the LLM ships an inline source; fail if it asks the
   user to provide URLs.

**Files:**

- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`.

### Step 1 — Read the current skill near the inline_blob discussion

Lines 81, 491-496, 708, 1352-1363 already reference `inline_blob`. The
edit adds a new top-level instruction near the beginning of the skill
(probably around the "Choosing the source plugin" section) that
explicitly prefers `inline_blob` for short user-typed data, and adds an
example of the canonical hero pattern.

### Step 2 — Edit

Add a new subsection (somewhere structurally appropriate; the existing
section about "When to use inline_blob vs blob_id vs path" near line
491 is the natural home) that reads roughly:

> ### Default to `inline_blob` when the user types data into chat
>
> When the user provides source data *in their chat message itself* — a
> URL, a sentence, a short list (≤ ~20 items), or a single record —
> create the source with `source.inline_blob` directly. Do NOT ask the
> user to upload a CSV. The audit recorder treats inline content
> identically to file content (the SHA-256 hash flows into
> `source_data_hash` the same way), so there is no auditability cost
> to inline source creation.
>
> Patterns that should trigger `inline_blob`:
>
> | User wrote | Treat as |
> |---|---|
> | `go to https://example.com` | 1-row inline CSV with `url` header |
> | `check these URLs: a.com, b.com, c.com` | 3-row inline CSV, one URL per row (confirm row count with the user before finalising) |
> | `this transaction: $4,200, payee 'Acme Corp', date 2026-04-15` | 1-row inline CSV with parsed fields |
> | `create a list of 5 government web pages` | Generate the 5 URLs yourself, present them for user review, then create the inline CSV |
>
> If the user typed something that LOOKS like data but is genuinely
> ambiguous, prefer surfacing your interpretation to the user (via the
> proposal narration) before committing. The user will confirm or
> redirect.

The exact prose can be tuned during implementation. The key intents:

1. Prefer `inline_blob` for short typed data.
2. The auditability story is preserved.
3. Confirm ambiguous row counts before committing.
4. The LLM-generated-rows case is explicitly named.

### Step 3 — Restart the elspeth-web service

Per `project_composer_harness_state`: the composer skill is loaded via
`@lru_cache` on module import. Edits do not take effect until the web
service restarts.

```bash
sudo systemctl restart elspeth-web.service
```

### Step 4 — Empirical validation

Run the three validation prompts above through the live LLM (against
the staging deploy at `elspeth.foundryside.dev`). For each:

1. Type the prompt into an empty composer session.
2. Inspect the response. Pass criteria documented above.
3. If a prompt fails, iterate on the skill text and re-test.

Validation gate: all three prompts pass.

### Step 5 — Commit

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md
git commit -m "feat(composer/skill): prefer inline_blob for chat-typed source data (Phase 5a.8)"
```

---

## Task 9 — Staging smoke

**Goal.** End-to-end manual smoke against staging
(`elspeth.foundryside.dev`) to confirm the feature works in the same
deployment configuration users will hit.

Per `project_staging_deployment`: source-checkout systemd/Caddy deploy
on the host machine. Frontend = `npm run build`. Backend = `systemctl
restart elspeth-web.service`.

### Step 1 — Deploy

```bash
cd /home/john/elspeth
git checkout <phase-5a-branch>
cd src/elspeth/web/frontend && npm run build && cd -
sudo systemctl restart elspeth-web.service
```

### Step 2 — Smoke checklist

- [ ] Open `https://elspeth.foundryside.dev` in a fresh browser
      profile / private window.
- [ ] Authenticate.
- [ ] Create a new session via the header dropdown.
- [ ] Verify the empty-state chat input placeholder reads "Describe your
      pipeline, or paste a URL or sample data to start...".
- [ ] Type `go to https://finance.gov.au` → send.
- [ ] Wait for the LLM response. Expected: a `set_pipeline` tool call
      with `source.inline_blob`. Verified by:
  - The chat surface renders an `InlineSourceCreatedTurn` widget.
  - The widget shows `filename`, `mime_type`, row count = 1, content
    hash (SHA-256 hex).
- [ ] If Phase 2 audit-readiness panel has shipped, open the panel and
      confirm the Provenance row reads "✓ Inline content hashed".
- [ ] Start a fresh session. Type
      `check these URLs: example.com, foo.bar, baz.qux` → send.
      Expected: an `InlineSourceDisambiguationTurn` surfaces with three
      proposed rows, or the standard PendingProposalsBanner shows a
      pending proposal with the same content. Confirm "3 rows" →
      proposal accepts → InlineSourceCreatedTurn renders.
- [ ] Start a fresh session. Type the canonical hero prompt:
      `create a list of 5 government web pages and use an LLM to rate how cool they are`
      → send. Expected:
  - LLM generates 5 URLs.
  - The 5 URLs are presented for review (either as a pending proposal
    with multi-row inline_blob content, or via the
    `InlineSourceDisambiguationTurn` after acceptance).
  - The user accepts. An `InlineSourceCreatedTurn` renders with row
    count = 5 and provenance = `llm-generated` (visible by the "Edit
    the list" button).
- [ ] Click "Edit the list" → the inline editor opens with the 5 URLs
      pre-filled → edit one → save → new `set_pipeline` call → new
      InlineSourceCreatedTurn with the updated content hash.
- [ ] Test the LLM-skip safety net: in a fresh session, type
      `https://example.com` then send. If the LLM does NOT propose a
      source within 2 turns (it asks for a CSV instead, or natters
      about plugins), the `InlineSourceFallbackPrompt` should appear
      above the chat input. Clicking "Create source from this text"
      should trigger a synthetic chat turn that causes the LLM to
      create the inline source.

### Step 3 — Audit-trail spot-check

For one of the successful sessions: open the Landscape MCP analysis
tool, find the inline-sourced rows, and verify `source_data_hash` is
populated identically to a CSV-sourced equivalent.

```bash
elspeth-mcp --database sqlite:///./examples/.../audit.db
# Then via the MCP client:
explain_token(run_id=<id>, token_id=<id>)
# Expected: source_data_hash present, matches SHA-256 of the inline blob bytes.
```

This confirms the audit trail invariant (B1) holds end-to-end. It is a
*spot* check, not a comprehensive audit — the Landscape repository's
treatment of inline-sourced rows is identical to CSV-sourced rows by
construction (same `create_row` path), so a single spot check is the
verification gate.

### Step 4 — Sign-off

If all checklist items pass, Phase 5a is complete. Update the umbrella
PR description with a "Smoke passed" note and the Landscape spot-check
output.

---

## Risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | LLM ignores the prompt nudge and proposes CSV upload for short inputs anyway | High | Task 5 `InlineSourceFallbackPrompt` as floor; Task 8 empirical validation gate |
| R2 | LLM hallucinates source content in the `inline_blob.content` field (e.g. URLs that don't exist) | Medium | The disambiguation and `InlineSourceCreatedTurn` widgets force user review before the pipeline runs; never auto-commit. Audit trail records both the LLM proposal and the user's acceptance |
| R3 | Heuristic detection in `InlineSourceFallbackPrompt` produces false positives ("hi" looks like data) | Low | Heuristic is opt-in (a prompt, not a forced action) and dismissible. False positive cost = one extra click to dismiss; tune thresholds in Phase 8 |
| R4 | Disambiguation widget's "treat as 1 row" branch produces a CSV with a comma-laden cell that breaks downstream parsing | Medium | The LLM still constructs the inline_blob.content, and CSV-quoting is its responsibility (well within capability). If a turn produces a bad CSV, the source-validator on the next pipeline run quarantines or rejects — Tier-3 boundary behaviour. The frontend does not generate CSVs itself |
| R5 | "Edit the list" inline editor degrades to a 500-line modal that nobody uses | Low | v1 is a plain `<textarea>` pre-filled with the inline content; richer editing is Phase 8 polish if telemetry shows demand |
| R6 | `inlineSourceStore` derivation lags `compositionState` updates and renders a stale `InlineSourceCreatedTurn` | Medium | Derive on every `compositionState` change in ChatPanel; the store is a pure cache. Cover with the Task 6 integration test |
| R7 | Audit-panel integration (Task 7) ships before Phase 2 → unused dead code | Low | Task 7 is conditional on Phase 2 having shipped at planning time. If deferred, the follow-up ticket is filed |
| R8 | The composer skill nudge backfires — the LLM now uses `inline_blob` even when a CSV upload would be more appropriate (e.g., 50 rows in chat) | Low | Threshold-cutoff redirection is explicitly scoped to Phase 8. Phase 5a's risk is one-direction; the failure mode is "inline_blob used where CSV-upload would be marginally better", which is recoverable by the user pasting CSV |
| R9 | Phase 5b's interpretation-acceptance event shape (open question B2) constrains Phase 5a's disambiguation-turn data model | Low | Phase 5a's disambiguation turn is UI-only; it does NOT introduce a new event type. The mutation is still a standard composition_proposal. Phase 5b can layer interpretation events on top without disrupting Phase 5a's UI |
| R10 | Chat input placeholder change creates a regression in Phase A slice 4's per-step guided-mode placeholder | Low | The explicit `placeholder?:` prop override case is covered by a dedicated test (Task 1 step 1 case 4) |

## Memory references

- `project_composer_dynamic_source_from_chat` — feature memory
- `project_composer_canonical_test_case` — the hero validation prompt
- `project_composer_first_run_tutorial` — downstream consumer (Phase 4)
- `project_composer_two_audiences` — frames why this feature exists
- `project_composer_harness_state` — skill loading semantics (Task 8)
- `project_staging_deployment` — Task 9 deployment instructions
- `feedback_no_tests_for_skill_prompts` — Task 8 validation rule
- `feedback_default_is_fix_not_ticket` — Task 7's conditional rule
- `project_db_migration_policy` — no DB changes; not invoked
- `feedback_default_to_worktree` — operator preference for new code work

## Review history

| Date | Reviewer | Verdict | Finding IDs | Notes |
|------|----------|---------|-------------|-------|
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, I1, I2 | Applied in this revision. B1: completed placeholder test fixtures in `inlineSourceStore.test.ts` with concrete `InlineSourceSummary` literals. I1: replaced vague detection heuristic prose with a precise 4-condition specification biased toward false negatives. I2: corrected both store selectors (`messages`, `compositionState`) to use the real singleton shape verified against `sessionStore.ts` lines 153–154. |
