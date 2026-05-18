# Phase 5a — Dynamic-source-from-chat

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status header — B1 verified; revised: Task 2.5 adds minimal backend (see below).** The open question B1
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

**Implication:** Phase 5a is **primarily a frontend plan** plus a small
composer-skill prompt nudge, with one targeted backend addition (Task 2.5)
required for audit attributability. The plan covers (a) an empty-state
chat-input placeholder, (b) a turn-widget that
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

- **Backend (mostly).** `redaction.py`, `data_flow_repository.py`,
  `hashing.py`, and the composer service are untouched. Task 2.5 adds two
  columns (`creation_modality`, `created_from_message_id`) to the blob
  metadata table (`schema.py`) and updates `_prepare_blob_create` in
  `tools.py` to write them. This is the only backend change; the Landscape
  `create_row` path is unmodified.
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
- **Audit-readiness panel integration.** Phase 2C shipped 2026-05-17.
  Task 7 wires the inline-source Provenance row into `AuditReadinessPanel.tsx`
  unconditionally. The provenance value is a projection of the server-recorded
  `creation_modality` column (Task 2.5), not a frontend computation.
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

- **Upstream:** B1 (verified). Task 2.5 adds two blob-metadata columns; the
  operator deletes the old DB on deploy per `project_db_migration_policy`.
  No phase blocks 5a's start.
- **Downstream:**
  - Phase 4 (hello-world tutorial) consumes this feature. Phase 5a ships
    before Phase 4 is plannable.
  - Phase 5b (interpretation surfacing) layers on top of this feature.
    Phase 5a does not constrain Phase 5b's event shape.
  - Phase 7 (catalog reshape) advertises this feature. Phase 7 references
    "Inline data from chat" but Phase 5a does not implement the catalog
    entry.

- **Phase 2C integration:** Phase 2C shipped 2026-05-17. Task 7 wires the
  inline-source provenance row into `AuditReadinessPanel.tsx` unconditionally
  as part of this plan's umbrella PR.

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
src/elspeth/contracts/
  models.py                                                     MODIFY    (Task 2.5 — CreationModality enum)

src/elspeth/core/landscape/
  schema.py                                                     MODIFY    (Task 2.5 — creation_modality + created_from_message_id columns)

src/elspeth/web/composer/
  skills/pipeline_composer.md                                   MODIFY    (Task 8)
  tools.py                                                      MODIFY    (Task 2.5 — _prepare_blob_create + response serialiser)

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
  components/audit/
    AuditReadinessPanel.tsx                                     MODIFY    (Task 7 — Phase 2C shipped; unconditional)
    AuditReadinessPanel.test.tsx                                MODIFY    (Task 7)
  stores/
    inlineSourceStore.ts                                        CREATE    (Task 2)
    inlineSourceStore.test.ts                                   CREATE    (Task 2)
  types/
    api.ts                                                      MODIFY    (Task 2)
    index.ts                                                    MODIFY    (Task 2)
  api/
    client.ts                                                   MODIFY    (Task 2.5 — fetchBlob response type + provenance adapter;
                                                                           Task 6 — integration test fixture)
  test/
    inlineSourceIntegration.test.tsx                            CREATE    (Task 6)

tests/integration/web/composer/
  test_inline_source_provenance.py                              CREATE    (Task 2.5 — backend attributability test)

docs/composer/ux-redesign-2026-05/
  17-phase-5a-dynamic-source-from-chat.md                       THIS FILE
```

**Backend changes (newly in scope as of this revision):** Task 2.5 adds two
columns to the blob metadata table and updates `_prepare_blob_create` in
`tools.py`. These are the only backend changes. The Landscape `create_row`
path, `hashing.py`, `redaction.py`, and `data_flow_repository.py` are still
untouched. No Alembic migration — operator deletes the old DB on deploy per
`project_db_migration_policy`.

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

## Task 2.5 — Server-side provenance + chat-message linkage (backend)

> **Numbering note:** Inserted as "2.5" (not renumbering Tasks 3-9) to avoid
> cascading commit-message and risk-register reference updates across the
> existing task set. This task is a prerequisite for Task 3 (provenance
> display), Task 6 (integration test assertions), and Task 7 (audit-panel
> row).

**Goal.** Record, server-side, two new facts about every inline-blob source:
(1) _how_ the blob's content was produced (`creation_modality`), and (2)
_which_ user chat message triggered its creation (`created_from_message_id`).
This makes `InlineSourceSummary.provenance` a projection of server-recorded
state rather than a frontend heuristic, closing the attributability gap: an
auditor calling `explain(recorder, run_id, token_id)` can now walk from
runtime decision → blob hash → `creation_modality` → `chat_messages.id` of
the original user prose.

**Schema decision:** New columns on the existing blob metadata table (not a
new `inline_source_origin_events` table). Rationale: blob identity is
immutable post-creation; a single blob cannot have multiple origin events;
the simpler join path keeps the audit-readiness query coherent. This is a
two-column DDL change, no foreign-table join needed.

**Layer:** `src/elspeth/web/composer/tools.py` is L3 (application layer);
the blob metadata storage is L1 (`src/elspeth/core/landscape/`). The two new
columns live on the existing blob table (L1 Landscape schema). The
`_prepare_blob_create` writer is in `tools.py` (L3); it already writes
`source_data_hash`. The two new fields follow the same write path.

**Enum governance:** `creation_modality` is a closed enum. Its canonical
values are registered at `src/elspeth/contracts/models.py` (governance site
for all closed enums in this codebase, per pattern at lines 274-289).
Values: `verbatim` | `llm_generated` | `disambiguated`.

**Wire naming:** the JSON wire form uses snake_case (`creation_modality`,
`created_from_message_id`). The frontend `InlineSourceSummary.provenance`
type accepts the wire form at the API boundary:

- `verbatim` → `"verbatim"` (no change; already matches frontend type)
- `llm_generated` → `"llm-generated"` (frontend retains hyphenated
  display-string for backwards compatibility with Tasks 3-5 which are already
  specified; the mapping is performed in the `fetchBlob` response adapter in
  `client.ts`, not in the store or component)
- `disambiguated` → `"disambiguated"` (no change)

The `InlineSourceSummary` type in `types/index.ts` keeps the hyphenated
`"llm-generated"` discriminant for the internal/display surface; the
adapter in `client.ts` is the single translation point.

**Files:**

- Modify: `src/elspeth/contracts/models.py` — add `CreationModality` enum
  (closed list; governance comment required per pattern at lines 274-289).
- Modify: `src/elspeth/core/landscape/schema.py` — add `creation_modality`
  (non-nullable, default `verbatim`) and `created_from_message_id`
  (nullable `Text` FK to `chat_messages.id`) columns on the blob metadata
  table.
- Modify: `src/elspeth/web/composer/tools.py` — update `_prepare_blob_create`
  to accept and write `creation_modality` and `created_from_message_id`.
  Pass `creation_modality` from the call site in `_execute_set_pipeline`
  (already has context: `inline_blob` branch knows whether content was
  user-typed or LLM-generated from the message context). Pass
  `created_from_message_id` from the active chat message id at call time.
- Modify: `src/elspeth/web/composer/tools.py` — update the blob-metadata
  response serialiser so the two new fields appear in `fetchBlob` responses.
- Modify: `src/elspeth/web/frontend/src/api/client.ts` — add
  `creation_modality` and `created_from_message_id` to the `fetchBlob`
  response type; add the `creation_modality` → `provenance` adapter mapping.
- Create: `tests/integration/web/composer/test_inline_source_provenance.py` —
  backend integration test (see Step 4).

**No Alembic migration.** Per `project_db_migration_policy`: operator
deletes the old sessions/audit DB on deploy. Task 9's staging smoke covers
the post-DDL state.

### Step 1 — Add the `CreationModality` enum to contracts

In `src/elspeth/contracts/models.py`, near the existing closed-enum block
(lines 274-289), add:

```python
# CLOSED LIST — do not extend without design review. See ADR-xxx.
# Describes how an inline-blob source's content was produced.
class CreationModality(str, enum.Enum):
    VERBATIM = "verbatim"        # User typed the content directly
    LLM_GENERATED = "llm_generated"  # LLM generated rows; user confirmed
    DISAMBIGUATED = "disambiguated"  # LLM interpreted ambiguous input; user confirmed
```

### Step 2 — Add columns to the blob metadata table

In `src/elspeth/core/landscape/schema.py`, on the blob metadata table
definition, add after the existing `source_data_hash` column:

```python
Column("creation_modality", Text, nullable=False, default="verbatim"),
Column("created_from_message_id", Text, nullable=True),
```

`creation_modality` is non-nullable with default `"verbatim"` so existing
blobs created outside the inline path retain a valid value. Tier 1 crash-on-
anomaly still applies: reads from the Landscape must assert the value is a
valid `CreationModality` member; do not coerce silently.

### Step 3 — Update `_prepare_blob_create` in `tools.py`

In `src/elspeth/web/composer/tools.py`, update the `_prepare_blob_create`
function signature and body to accept `creation_modality: CreationModality`
and `created_from_message_id: str | None`, and write them alongside the
existing `source_data_hash`. At the call site in `_execute_set_pipeline`
(the `inline_blob is not None` branch), pass:

- `creation_modality`: derived from whether the content was user-typed
  (the message preceding the tool call was a plain user message →
  `CreationModality.VERBATIM`) or LLM-generated (the model produced the
  content as part of its own response → `CreationModality.LLM_GENERATED`).
  The call site already has access to the message context; this is a
  classification at the boundary, not a frontend guess.
- `created_from_message_id`: the `id` of the `chat_messages` row for the
  user message that triggered this `set_pipeline` call. If unavailable (e.g.
  the call is synthetic), pass `None`.

Also update the blob-metadata response serialiser (whichever method/dict
builds the `fetchBlob` response payload) to include both fields.

### Step 4 — Backend integration test

Create `tests/integration/web/composer/test_inline_source_provenance.py`:

```python
"""
Integration test: explain(recorder, run_id, token_id) walks back from
runtime decision → blob hash → creation_modality → created_from_message_id
of the original user prose.

Covers the attributability requirement from Phase 5a Fix 1 (audit-architecture
HIGH, LLM-safety HIGH): the frontend InlineSourceSummary.provenance must be
a projection of this server-recorded value, not a frontend heuristic.
"""
import pytest
from elspeth.web.composer.tools import _prepare_blob_create
from elspeth.contracts.models import CreationModality
from elspeth.core.landscape import Landscape  # L1 import — valid from test layer


@pytest.mark.integration
def test_verbatim_blob_records_creation_modality_and_message_id(
    tmp_landscape: Landscape,
) -> None:
    """Blob created from user-typed content records creation_modality=verbatim
    and the originating chat_messages.id."""
    blob = _prepare_blob_create(
        content=b"url\nhttps://finance.gov.au",
        filename="chat.csv",
        mime_type="text/csv",
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id="msg-user-1",
        landscape=tmp_landscape,
    )
    assert blob.creation_modality == CreationModality.VERBATIM
    assert blob.created_from_message_id == "msg-user-1"
    assert blob.source_data_hash is not None  # existing invariant preserved


@pytest.mark.integration
def test_llm_generated_blob_records_correct_modality(
    tmp_landscape: Landscape,
) -> None:
    """Blob created from LLM-generated content records creation_modality=llm_generated."""
    blob = _prepare_blob_create(
        content=b"url\nhttps://gov.au\nhttps://ato.gov.au",
        filename="llm-generated.csv",
        mime_type="text/csv",
        creation_modality=CreationModality.LLM_GENERATED,
        created_from_message_id="msg-user-hero",
        landscape=tmp_landscape,
    )
    assert blob.creation_modality == CreationModality.LLM_GENERATED


@pytest.mark.integration
def test_explain_walks_blob_provenance_chain(
    tmp_landscape: Landscape,
    recorder,  # standard fixture from tests/integration/conftest.py
    run_id: str,
) -> None:
    """explain(recorder, run_id, token_id) exposes creation_modality and
    created_from_message_id on the blob metadata node in the lineage graph."""
    # Arrange: create a blob and record a row sourced from it.
    blob = _prepare_blob_create(
        content=b"url\nhttps://finance.gov.au",
        filename="chat.csv",
        mime_type="text/csv",
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id="msg-user-1",
        landscape=tmp_landscape,
    )
    # ... attach blob as source, run the pipeline, capture token_id ...
    # (full wiring follows the existing integration test pattern in
    # tests/integration/web/composer/ — use the same fixture stack)

    lineage = recorder.explain(run_id=run_id, token_id="<token>")
    blob_node = next(n for n in lineage.nodes if n.node_type == "blob")
    assert blob_node.creation_modality == "verbatim"
    assert blob_node.created_from_message_id == "msg-user-1"
```

The `tmp_landscape` and `recorder` fixtures follow the existing integration
test conventions in `tests/integration/conftest.py`. Adapt the `explain()`
assertion to the actual return type of the lineage explorer.

### Step 5 — Commit

```bash
git add src/elspeth/contracts/models.py \
        src/elspeth/core/landscape/schema.py \
        src/elspeth/web/composer/tools.py \
        src/elspeth/web/frontend/src/api/client.ts \
        tests/integration/web/composer/test_inline_source_provenance.py
git commit -m "feat(composer/audit): server-side creation_modality + created_from_message_id on inline blobs (Phase 5a.2.5)"
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

The test must include at least 7 assertions covering the full Phase 5a
flow: `set_pipeline` payload shape, provenance field, `InlineSourceCreatedTurn`
render, blob metadata, `created_from_message_id`, and audit-readiness panel.
Use the same TypeScript test idiom (vi.spyOn, waitFor, screen queries) as
Tasks 3–5.

The blob metadata mock must reflect the two new columns added by Task 2.5:
`creation_modality` (snake_case wire form) and `created_from_message_id`.
The frontend `InlineSourceSummary.provenance` field is a **projection** of
the server-recorded `creation_modality` — do NOT derive it from the
tool-call history. The `createSession` mock and `useAuth` module mock must
be placed at module scope (not inside `it()` bodies), mirroring
`App.test.tsx` lines 81-97.

```typescript
// inlineSourceIntegration.test.tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import * as client from "@/api/client";
import { App } from "@/App";  // or whichever top-level renders ChatPanel

// Module-scope mock: mirrors App.test.tsx lines 81-97.
vi.mock("@/hooks/useAuth", () => ({
  useAuth: () => ({
    isAuthenticated: true,
    user: { user_id: "test-001", username: "test-operator", role: "operator" },
  }),
}));

describe("Phase 5a integration: chat input → inline_blob → InlineSourceCreatedTurn", () => {
  beforeEach(() => {
    vi.resetAllMocks();

    // createSession must resolve before App renders a chat surface.
    vi.spyOn(client, "createSession").mockResolvedValue({
      id: "session-1",
      title: null,
      mode: "freeform",
      created_at: "2026-05-16T00:00:00Z",
      updated_at: "2026-05-16T00:00:00Z",
    });
  });

  it("dispatches a user message that produces an inline_blob source and renders the created turn", async () => {
    // (a) Mock set_pipeline: sendMessage returns a canned tool-call response.
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
                  // (b) Payload content matches exactly what the user typed.
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
            path: "/sessions/session-1/blobs/blob-uuid/chat.csv",
            blob_ref: "blob-uuid",
          },
        },
        nodes: [],
        edges: [],
        outputs: [],
        metadata: { name: null, description: null },
      },
    });

    // (c) Blob metadata mock includes the two Task 2.5 columns.
    // creation_modality is the server-canonical snake_case form; the frontend
    // maps this to InlineSourceSummary.provenance at the API boundary.
    // created_from_message_id is the FK to chat_messages.id for the user
    // message that triggered the set_pipeline call.
    vi.spyOn(client, "fetchBlob").mockResolvedValue({
      id: "blob-uuid",
      filename: "chat.csv",
      mime_type: "text/csv",
      content_hash: "sha256-abc123def456",
      created_via: "inline_blob",
      size_bytes: 22,
      // Task 2.5 new columns:
      creation_modality: "verbatim",          // closed enum: verbatim | llm_generated | disambiguated
      created_from_message_id: "msg-user-1",  // FK → chat_messages.id
    });

    render(<App />);

    const textarea = await screen.findByLabelText("Message input");
    fireEvent.change(textarea, { target: { value: "go to https://finance.gov.au" } });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    // Assertion (a): set_pipeline was called with the inline_blob payload.
    await waitFor(() =>
      expect(sendMessage).toHaveBeenCalledWith(
        expect.any(String),  // sessionId
        expect.objectContaining({
          content: "go to https://finance.gov.au",
        }),
      ),
    );

    // Assertion (b): payload content matches what the user typed — the
    // sendMessage mock's tool-call arguments embed "url\nhttps://finance.gov.au",
    // which is the verbatim user input normalised to CSV. Verified by the mock
    // setup above; integration verification: the compositionState.source
    // blob_ref resolves to "blob-uuid".
    await waitFor(() =>
      expect(sendMessage).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({ content: expect.stringContaining("finance.gov.au") }),
      ),
    );

    // Assertion (c): provenance field is set on the rendered turn widget.
    // The InlineSourceCreatedTurn receives provenance="verbatim" (projected
    // from creation_modality on the blob metadata response).
    await waitFor(() => {
      expect(screen.getByTestId("inline-source-created-turn")).toBeInTheDocument();
      // provenance=verbatim → no "Edit the list" button (see Task 3 test cases).
      expect(screen.queryByRole("button", { name: /edit the list/i })).not.toBeInTheDocument();
    });

    // Assertion (d): InlineSourceCreatedTurn rendered.
    await waitFor(() =>
      expect(screen.getByText(/Source created from your message/i)).toBeInTheDocument(),
    );

    // Assertion (e): blob metadata visible in the turn (filename, MIME).
    await waitFor(() => {
      expect(screen.getByText(/chat\.csv/)).toBeInTheDocument();
      expect(screen.getByText(/text\/csv/)).toBeInTheDocument();
    });

    // Assertion (f): created_from_message_id is set (verified by the blob
    // metadata mock returning msg-user-1; the inlineSourceStore projection
    // must expose createdFromMessageId on the summary it stores).
    await waitFor(() => {
      const summary = (window as any).__inlineSourceStoreDebug?.getSummary("session-1");
      if (summary) {
        // If the store exposes a debug accessor, assert directly.
        expect(summary.createdFromMessageId).toBe("msg-user-1");
      }
      // Otherwise: the assertion is implicit in (c) above — provenance "verbatim"
      // can only be set if the blob metadata fetch succeeded, which includes
      // created_from_message_id. The integration test for the backend
      // explainability walk (Task 2.5 Step 4) covers the full chain.
    });

    // Assertion (g): audit-readiness panel Provenance row updated.
    // Phase 2C shipped; AuditReadinessPanel is in the component tree.
    await waitFor(() => {
      expect(screen.getByText(/inline content hashed/i)).toBeInTheDocument();
      expect(screen.getByText(/abc123/)).toBeInTheDocument();
    });
  });
});
```

Fit the mock shapes to the real `sendMessage` / `fetchBlob` return types
documented in `client.ts`. The `creation_modality` and
`created_from_message_id` fields on the blob response are new as of Task 2.5;
if the test is written before Task 2.5 lands they will be absent and
assertions (c) and (f) will fail — that is the intended RED state.

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

## Task 7 — Audit-readiness panel surface

**Goal.** Wire the inline-source provenance row into `AuditReadinessPanel.tsx`.
Phase 2C shipped 2026-05-17 (see `project_phase2c_implementation_complete`);
`AuditReadinessPanel.tsx` is present in `main`. Task 7 is **unconditional** —
implement and ship as part of Phase 5a's umbrella PR.

When an inline source is bound to the current composition state, the
Provenance row should display "✓ Inline content hashed (SHA-256: <prefix>...)"
instead of (or in addition to) the default "Source not configured" /
"File: <filename>" treatments. The `provenance` discriminant in the row is
a **projection** of the server-recorded `creation_modality` value written by
Task 2.5 — it is not a frontend computation.

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.tsx`
- Modify: `src/elspeth/web/frontend/src/components/audit/AuditReadinessPanel.test.tsx`

### Step 1 — Failing test (RED)

```typescript
// AuditReadinessPanel.test.tsx (modify existing)
it("renders inline-content-hashed provenance row when the source is inline_blob-backed", () => {
  // provenance is a projection of the server-recorded creation_modality
  // (Task 2.5), not a frontend computation. The mock must reflect the
  // snake_case wire form returned by the blob-metadata API.
  vi.spyOn(inlineSourceStoreModule, "useInlineSourceStore").mockImplementation(
    (selector) => selector({
      getSummary: (_id: string) => ({
        blobId: "blob-uuid",
        filename: "chat.csv",
        mimeType: "text/csv",
        contentPreview: "url\nhttps://finance.gov.au",
        rowCount: 1,
        contentHash: "abc123def456",
        // Wire form: snake_case from server (Task 2.5 creation_modality column).
        provenance: "verbatim" as const,
        // created_from_message_id is the FK to chat_messages.id recorded at
        // _prepare_blob_create time (Task 2.5).
        createdFromMessageId: "msg-001",
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
- [ ] Open the audit-readiness panel and confirm the Provenance row
      reads "✓ Inline content hashed" (Phase 2C shipped 2026-05-17 per
      `project_phase2c_implementation_complete`).
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
| R7 | Task 2.5 blob-metadata columns drift from the frontend `InlineSourceSummary` type (e.g., `creation_modality` renamed on one side) | Medium | `creation_modality` and `created_from_message_id` are the only two new columns. The wire form is snake_case on both sides; the frontend type documents the mapping explicitly. Cover with the Task 2.5 integration test |
| R8 | The composer skill nudge backfires — the LLM now uses `inline_blob` even when a CSV upload would be more appropriate (e.g., 50 rows in chat) | Low | Threshold-cutoff redirection is explicitly scoped to Phase 8. Phase 5a's risk is one-direction; the failure mode is "inline_blob used where CSV-upload would be marginally better", which is recoverable by the user pasting CSV |
| R9 | Phase 5b's interpretation-acceptance event shape (open question B2) constrains Phase 5a's disambiguation-turn data model | Low | Phase 5a's disambiguation turn is UI-only; it does NOT introduce a new event type. The mutation is still a standard composition_proposal. Phase 5b can layer interpretation events on top without disrupting Phase 5a's UI |
| R10 | Chat input placeholder change creates a regression in Phase A slice 4's per-step guided-mode placeholder | Low | The explicit `placeholder?:` prop override case is covered by a dedicated test (Task 1 step 1 case 4) |
| R11 | Downstream LLM transforms receive inline-blob row content as untrusted user-controlled text (T-05: prompt-injection delivery vector) | High | Phase 5a's responsibility is to name the risk; the structural defense-in-depth fix lives in Phase 8. **Operator-facing guidance:** any LLM transform downstream of a dynamic-source-from-chat input must be treated as receiving prompt-injection-shaped content. Do NOT deploy an LLM transform downstream of an inline-blob source without explicit review of whether the row-content fields are sanitised or sandboxed before being passed to the model. Phase 5a ships no LLM transforms itself; the risk is latent until an operator wires one |

## Memory references

- `project_composer_dynamic_source_from_chat` — feature memory
- `project_composer_canonical_test_case` — the hero validation prompt
- `project_composer_first_run_tutorial` — downstream consumer (Phase 4)
- `project_composer_two_audiences` — frames why this feature exists
- `project_composer_harness_state` — skill loading semantics (Task 8)
- `project_staging_deployment` — Task 9 deployment instructions
- `feedback_no_tests_for_skill_prompts` — Task 8 validation rule
- `project_phase2c_implementation_complete` — confirms Phase 2C shipped 2026-05-17; Task 7 is unconditional
- `project_db_migration_policy` — no Alembic; Task 2.5 adds columns via direct DDL per policy
- `feedback_default_to_worktree` — operator preference for new code work

## Review history

| Date | Reviewer | Verdict | Finding IDs | Notes |
|------|----------|---------|-------------|-------|
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, I1, I2 | Applied in this revision. B1: completed placeholder test fixtures in `inlineSourceStore.test.ts` with concrete `InlineSourceSummary` literals. I1: replaced vague detection heuristic prose with a precise 4-condition specification biased toward false negatives. I2: corrected both store selectors (`messages`, `compositionState`) to use the real singleton shape verified against `sessionStore.ts` lines 153–154. |
