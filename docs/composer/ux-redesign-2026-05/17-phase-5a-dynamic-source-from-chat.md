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
  `src/elspeth/web/composer/redaction.py` lines 1021-1106.

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

## Worktree

**Branch:** `feat/composer-phase-5-chat-data-entry`
**Worktree path:** `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`
**Shared with:** the entire Phase 5 umbrella (17-, 18-, 18a-, 18b-). Phase 5a and Phase 5b ship as a coordinated PR; do NOT split into separate branches. This document is one of the four that will be implemented together on this single worktree. Shared with 18-, 18a-, 18b- (the Phase 5b overview, backend, and frontend plans).

### Setup (one-time)

From the main checkout at `/home/john/elspeth`:

```bash
git worktree add .worktrees/composer-phase-5-chat-data-entry -b feat/composer-phase-5-chat-data-entry
cd .worktrees/composer-phase-5-chat-data-entry
uv venv --python 3.13                       # Python 3.13 to match main; mismatched versions produce ~300 spurious tier-model violations
source .venv/bin/activate
uv pip install -e ".[dev,llm]"              # editable install bound to THIS worktree's venv, not main's
```

### Operational notes

- **uv venv discipline:** every `uv pip install` invocation in this worktree MUST be preceded by `source .venv/bin/activate` OR invoked with `--python /home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/.venv/bin/python`. Without this, `uv` resolves to main's `.venv` and clobbers it. (See `feedback_uv_venv_leak`.)
- **filigree CLI:** the bare `filigree` command rejects realpath-escaping DBs from inside a worktree. Prefer the `mcp__filigree__*` tools. If you must use the CLI, run it from the git common dir: `(cd "$(git rev-parse --git-common-dir)/.." && filigree <verb>)`.
- **Subagent dispatch from this worktree:** subagents inherit parent CWD silently. Prefix every dispatch prompt with: "Your CWD is `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`; all file paths must be absolute." Use absolute paths everywhere. (See `feedback_subagents_cant_use_worktrees`.)
- **Composer-skill edits stay on main:** the `src/elspeth/web/composer/skills/pipeline_composer.md` file is read by the live `elspeth-web.service` from main, not from any worktree. Skill-prompt edits in this phase (e.g. 5a Task 8 nudge, 5b Task 8 nudge) must be applied on main and the service restarted, per `feedback_skip_worktree_for_skill_and_config_edits`. Land the rest of the work in the worktree as normal.

### Coordination during implementation

- All four plan docs ship one commit history. The order is: 17- (Phase 5a) lands first; then 18a- (Phase 5b backend); then 18b- (Phase 5b frontend). 18- (overview) carries no code changes — its amendments land alongside whichever backend doc they cross-reference.
- The two-DB deletion requirement (session DB + Landscape audit.db) is operator-visible — surface it in the PR description so the operator can run the deletion before deploy.

---

## Scope boundaries

**In scope:**

- A context-aware empty-state placeholder for `ChatInput.tsx`. Empty state
  reads "Describe your pipeline, paste a URL, or type a few rows of data to start...".
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

- **Backend (mostly).** `data_flow_repository.py`, `hashing.py`, and
  the composer service core are untouched. The Landscape `create_row`
  path is unmodified. Task 2.5 adds provenance columns to
  `web/sessions/models.py:blobs_table` and updates `_prepare_blob_create`
  in `tools.py`. Task 2.5 also adds `max_length=262_144` to
  `_InlineBlobModel.content` in `redaction.py` (correcting an incorrect
  prior claim that oversize content was already rejected). Task 2.6 adds
  an immutability trigger to `chat_messages_table` in
  `web/sessions/models.py`. These are the only backend changes.
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
   (`redaction.py` line 1021-1048) declares
   `content: Annotated[str, Sensitive(summarizer=...)]` with
   `model_config = ConfigDict(extra="forbid")`. Pydantic enforces both
   the type and the closed-field-set at decode time. Task 2.5 Step 3a
   adds `max_length=262_144` to this field (256 KiB cap).
3. `_prepare_blob_create` (called from the `inline_blob is not None`
   branch in `_execute_set_pipeline`, `tools.py` line 4452) validates
   `mime_type` against `_MIME_TO_SOURCE` (an allowlist; unknown MIME types
   produce a `ToolArgumentError`) and produces a SHA-256 hash of the raw
   bytes before storage. The resulting digest is stored as `content_hash` on
   the `blobs_table` row (session DB, `web/sessions/models.py`). This
   `content_hash` subsequently flows into `source_data_hash` on the
   `rows_table` row in the Landscape audit DB (`core/landscape/schema.py`
   line 165) when the row is created — two distinct fields in two distinct
   databases, linked by computation.
4. Post-storage, the inline blob behaves identically to a user-uploaded
   blob — same allowlist, same hashing, same Landscape provenance.

**What Phase 5a must NOT do:** invent Tier-3 validation on the frontend.
The frontend is allowed to assume the backend will reject malformed input
(missing `mime_type`, non-allowlisted MIME, etc.) and surface the rejection
to the user via the standard tool-call-error path. Defensive frontend
pre-validation would duplicate the boundary check and make the contract
ambiguous about which side is authoritative. Per CLAUDE.md trust-tier
model: validate **once** at the boundary; the boundary here is the backend
`set_pipeline` handler.

**Correction (F-6):** An earlier draft of this plan claimed the backend
already rejected oversize content. This was incorrect — prior to Task 2.5,
`_InlineBlobModel.content` had no `max_length` constraint and
`_prepare_blob_create` could allocate unbounded bytes before the session
quota check. Task 2.5/Step 3a adds `max_length=262_144` (256 KiB) to
`_InlineBlobModel.content` in `redaction.py`. After this change, a content
payload exceeding 256 KiB raises `ToolArgumentError` at Pydantic decode
time, before any allocation. The frontend does not add size validation —
it relies on the backend rejection surfaced via the standard error path.

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
2. Confirm the empty-state placeholder reads "Describe your pipeline,
   paste a URL, or type a few rows of data to start...".
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

> All file paths below are relative to the worktree root at `/home/john/elspeth/.worktrees/composer-phase-5-chat-data-entry/`; this is identical to main's tree but isolates working state per the project's worktree-by-default convention (`feedback_default_to_worktree`). Exception: `src/elspeth/web/composer/skills/pipeline_composer.md` is edited on main, not in the worktree — see the Worktree section above.

```text
src/elspeth/contracts/
  enums.py                                                      MODIFY    (Task 2.5 — CreationModality StrEnum; NOT contracts/models.py which does not exist)

src/elspeth/web/sessions/
  models.py                                                     MODIFY    (Task 2.5 — provenance columns on blobs_table;
                                                                           Task 2.6 — immutability trigger on chat_messages_table)

src/elspeth/web/composer/
  redaction.py                                                  MODIFY    (Task 2.5 — max_length=262_144 on _InlineBlobModel.content)
  skills/pipeline_composer.md                                   MODIFY    (Task 8)
  tools.py                                                      MODIFY    (Task 2.5 — _prepare_blob_create signature + UTF-8 guard + response serialiser)

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
  test_inline_source_provenance.py                              CREATE    (Task 2.5 — backend attributability + oversize + non-UTF-8 tests)

tests/integration/web/sessions/
  test_chat_messages_immutability.py                            CREATE    (Task 2.6 — chat_messages immutability trigger test)

docs/composer/ux-redesign-2026-05/
  17-phase-5a-dynamic-source-from-chat.md                       THIS FILE
```

**Backend changes (newly in scope as of this revision):**

- Task 2.5 adds provenance columns to `web/sessions/models.py:blobs_table`,
  adds `max_length=262_144` to `_InlineBlobModel.content` in `redaction.py`,
  and updates `_prepare_blob_create` in `tools.py`. The `CreationModality`
  enum goes to `contracts/enums.py` (not `contracts/models.py` — that file
  does not exist; not `core/landscape/schema.py` — `blobs_table` is a
  session-DB table in `web/sessions/models.py`, not a Landscape schema table).
- Task 2.6 adds an immutability trigger to `chat_messages_table` in
  `web/sessions/models.py`.
- The Landscape `create_row` path, `hashing.py`, `data_flow_repository.py`,
  and `core/landscape/schema.py` are untouched. No Alembic migration —
  operator deletes the old DB on deploy per `project_db_migration_policy`.

---

## Task 1 — Empty-state chat-input placeholder

**Goal.** Change the `ChatInput.tsx` placeholder to read "Describe your
pipeline, paste a URL, or type a few rows of data to start..." when
the chat has zero non-system messages and no active composition state.
Revert to the existing wording when either condition flips.

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
lines 154–155: `messages: ChatMessage[]` and
`compositionState: CompositionState | null`.

Pseudo-shape of the new test cases:

```typescript
describe("ChatInput empty-state placeholder", () => {
  it("shows the data-priming placeholder when the session has no messages and no composition state", () => {
    // arrange: fresh session, messages=[], version=0
    // act: render ChatInput
    // assert: textarea placeholder == "Describe your pipeline, paste a URL, or type a few rows of data to start..."
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
   against `sessionStore.ts` lines 154–155). Do NOT add new store
   fields just for this — the information already exists.

2. Derive the effective placeholder once, near the existing `canSend`
   computation:

   ```typescript
   const isEmptyState = messageCount === 0 && compositionVersion === 0;
   const defaultPlaceholder = isEmptyState
     ? "Describe your pipeline, paste a URL, or type a few rows of data to start..."
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
   * How this inline source's content was produced. Projected from the
   * server-recorded `creation_modality` column (Task 2.5) via the
   * `fetchBlob` response adapter in `client.ts`.
   *
   * - "verbatim"                  — user typed the content directly.
   * - "llm-generated"             — LLM generated rows; user confirmed.
   * - "disambiguated"             — LLM interpreted ambiguous input; user confirmed.
   * - "llm-generated-then-amended" — LLM generated rows, user amended via
   *                                   "Edit the list" (F-4). Drives the Edit
   *                                   button visibility alongside "llm-generated".
   *
   * The frontend uses hyphenated forms; the server uses snake_case
   * (`llm_generated`, `llm_generated_then_amended`). The adapter in
   * `client.ts` is the single translation point.
   */
  provenance: "verbatim" | "llm-generated" | "disambiguated" | "llm-generated-then-amended";
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

  // --- Disambiguation re-fire guard tests (F-11) ---

  it("addUserRequestedSingleRow stores the message ID and prevents re-check", () => {
    useInlineSourceStore.getState().addUserRequestedSingleRow("msg-1");
    expect(
      useInlineSourceStore.getState().userRequestedSingleRowForMessageIds.has("msg-1"),
    ).toBe(true);
    expect(
      useInlineSourceStore.getState().userRequestedSingleRowForMessageIds.has("msg-2"),
    ).toBe(false);
  });

  // --- "Not source data" escape tests (F-10) ---

  it("addNonSourceMessage stores the message ID", () => {
    useInlineSourceStore.getState().addNonSourceMessage("msg-escape-1");
    expect(
      useInlineSourceStore.getState().nonSourceMessageIds.has("msg-escape-1"),
    ).toBe(true);
  });

  // --- Fallback-prompt dismiss persistence tests (F-20) ---

  it("markDismissed records a session-scoped dismissal timestamp", () => {
    const before = Date.now();
    useInlineSourceStore.getState().markDismissed("session-1");
    const ts = useInlineSourceStore.getState().dismissedAt.get("session-1");
    expect(ts).toBeGreaterThanOrEqual(before);
    expect(useInlineSourceStore.getState().isDismissed("session-1")).toBe(true);
  });

  it("isDismissed returns false for sessions that were never dismissed", () => {
    expect(useInlineSourceStore.getState().isDismissed("session-never")).toBe(false);
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
  // --- Primary projection: per-session inline-source summary ---
  summariesBySession: Record<string, InlineSourceSummary>;
  setSummary: (sessionId: string, summary: InlineSourceSummary) => void;
  clearSummary: (sessionId: string) => void;
  getSummary: (sessionId: string) => InlineSourceSummary | null;

  // --- Disambiguation re-fire guard (F-11) ---
  // Message IDs for which the user explicitly chose "treat as 1 row".
  // The disambiguation predicate in ChatPanel skips these message IDs.
  userRequestedSingleRowForMessageIds: Set<string>;
  addUserRequestedSingleRow: (messageId: string) => void;

  // --- "Not source data" escape (F-10) ---
  // Message IDs for which the user explicitly chose "this isn't source data".
  // The disambiguation predicate and fallback-prompt predicate skip these.
  nonSourceMessageIds: Set<string>;
  addNonSourceMessage: (messageId: string) => void;

  // --- Fallback-prompt dismiss persistence (F-20) ---
  // Keyed by sessionId. A dismissed fallback prompt must not re-fire
  // within the same session regardless of predicate re-evaluation.
  dismissedAt: Map<string, number>;
  markDismissed: (sessionId: string) => void;
  isDismissed: (sessionId: string) => boolean;
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

  userRequestedSingleRowForMessageIds: new Set(),
  addUserRequestedSingleRow: (messageId) =>
    set((s) => ({
      userRequestedSingleRowForMessageIds: new Set([
        ...s.userRequestedSingleRowForMessageIds,
        messageId,
      ]),
    })),

  nonSourceMessageIds: new Set(),
  addNonSourceMessage: (messageId) =>
    set((s) => ({
      nonSourceMessageIds: new Set([...s.nonSourceMessageIds, messageId]),
    })),

  dismissedAt: new Map(),
  markDismissed: (sessionId) =>
    set((s) => {
      const next = new Map(s.dismissedAt);
      next.set(sessionId, Date.now());
      return { dismissedAt: next };
    }),
  isDismissed: (sessionId) => get().dismissedAt.has(sessionId),
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

**Goal.** Record, server-side, provenance facts about every inline-blob
source: (1) _how_ the blob's content was produced (`creation_modality`),
(2) _which_ user chat message triggered its creation
(`created_from_message_id`), and (3) _which LLM_ produced the content
for LLM-generated modalities (five `creating_*` provenance columns).
This makes `InlineSourceSummary.provenance` a projection of
server-recorded state rather than a frontend heuristic, closing the
attributability gap: an auditor calling `explain(recorder, run_id,
token_id)` can now walk from runtime decision → blob hash →
`creation_modality` → `chat_messages.id` of the original user prose →
the LLM identifier that generated the content (for `llm_generated`
modalities).

**Schema decision:** New columns on the existing blob metadata table (not
a new `inline_source_origin_events` table). Rationale: blob identity is
immutable post-creation; a single blob cannot have multiple origin events;
the simpler join path keeps the audit-readiness query coherent.

**Layer:** Both the blob table and `_prepare_blob_create` writer live in
the L3 session layer. `src/elspeth/web/sessions/models.py` contains
`blobs_table` (lines 447-519). `src/elspeth/web/composer/tools.py`
contains `_prepare_blob_create` (lines 3026-3145); it is also L3. No L1
Landscape schema changes are needed — `blobs_table` is a session-DB
table, not a Landscape table.

**Enum governance:** `creation_modality` is a closed enum. Its canonical
values are registered at `src/elspeth/contracts/enums.py` — the existing
file where all project-wide `StrEnum` enums live (verified: the file
contains `RunStatus`, `NodeStateStatus`, `BatchStatus`, etc.; `models.py`
does not exist in `contracts/`). Values: `verbatim` | `llm_generated` |
`disambiguated` | `llm_generated_then_amended`.

**Wire naming:** the JSON wire form uses snake_case (`creation_modality`,
`created_from_message_id`). The frontend `InlineSourceSummary.provenance`
type accepts the wire form at the API boundary:

- `verbatim` → `"verbatim"` (no change; already matches frontend type)
- `llm_generated` → `"llm-generated"` (frontend retains hyphenated
  display-string; the mapping is performed in the `fetchBlob` response
  adapter in `client.ts`, not in the store or component)
- `disambiguated` → `"disambiguated"` (no change)
- `llm_generated_then_amended` → `"llm-generated-then-amended"` (F-4;
  frontend may collapse to `"llm-generated"` for the Edit-button
  display discriminant if the distinction is not shown in the UI; the
  adapter in `client.ts` is the single translation point)

The `InlineSourceSummary` type in `types/index.ts` retains hyphenated
discriminants for the internal/display surface.

**Trust-tier check correction (F-6):** The earlier prose claimed the
backend already rejects oversize content. This was incorrect. Prior to
this plan, `_InlineBlobModel.content` had no `max_length` constraint, and
`_prepare_blob_create` could allocate unbounded bytes. Step 3 below adds a
`max_length=262_144` (256 KiB) constraint to `_InlineBlobModel.content` in
`redaction.py`. The backend now rejects oversize content at Pydantic decode
time, before any allocation, raising `ToolArgumentError`. The session quota
check that follows is a secondary defense, not the first.

**Files:**

- Modify: `src/elspeth/contracts/enums.py` — add `CreationModality`
  `StrEnum` (closed list; governance comment required per pattern at
  `web/sessions/models.py` lines 274-289).
- Modify: `src/elspeth/web/sessions/models.py` — add provenance columns
  to `blobs_table`: `creation_modality`, `created_from_message_id`,
  `creating_model_identifier`, `creating_model_version`,
  `creating_provider`, `creating_composer_skill_hash`,
  `creating_arguments_hash`, plus three new constraints
  (`fk_blobs_created_from_message_session`, `ck_blobs_creation_modality`,
  `ck_blobs_creating_llm_provenance_nullability`).
- Modify: `src/elspeth/web/composer/redaction.py` — add
  `max_length=262_144` to `_InlineBlobModel.content` (line 1021).
- Modify: `src/elspeth/web/composer/tools.py` — update
  `_prepare_blob_create` signature and body to accept and write all new
  provenance fields; add UTF-8 encode guard; update the blob-metadata
  response serialiser to include all new fields.
- Modify: `src/elspeth/web/frontend/src/api/client.ts` — add all new
  provenance fields to the `fetchBlob` response type; add the
  `creation_modality` → `provenance` adapter mapping.
- Create: `tests/integration/web/composer/test_inline_source_provenance.py` —
  backend integration test (see Step 4).

**No Alembic migration.** Per `project_db_migration_policy`: operator
deletes the old sessions/audit DB on deploy. Task 9's staging smoke covers
the post-DDL state.

### Step 1 — Add the `CreationModality` enum to contracts

In `src/elspeth/contracts/enums.py`, add alongside the existing `StrEnum`
classes (using `StrEnum`, not `str, enum.Enum`, to match the file's
established pattern):

```python
# CLOSED LIST — do not extend without design review. See ADR-xxx.
# Describes how an inline-blob source's content was produced.
# Adding a fifth value MUST include: (a) a spec amendment documenting the
# new modality and its audit semantics; (b) an integration test; (c) a
# Filigree ticket linking the change back to this enum.
class CreationModality(StrEnum):
    VERBATIM = "verbatim"                          # User typed the content directly
    LLM_GENERATED = "llm_generated"               # LLM generated rows; user confirmed
    DISAMBIGUATED = "disambiguated"               # LLM interpreted ambiguous input; user confirmed
    LLM_GENERATED_THEN_AMENDED = "llm_generated_then_amended"  # LLM generated, user amended via "Edit the list"
```

### Step 2 — Add columns to `blobs_table` in `web/sessions/models.py`

In `src/elspeth/web/sessions/models.py`, on `blobs_table` (currently ends
at line 519), add after the existing `status` column and before the
`ck_blobs_created_by` constraint:

```python
# --- Inline-blob provenance (Phase 5a) ---
# creation_modality: closed enum — how this blob's content was produced.
# Non-nullable with default "verbatim" so pre-5a blobs created outside
# the inline path retain a valid value. Tier 1 crash-on-anomaly applies
# to reads: assert the value is a valid CreationModality member; do not
# coerce silently.
#
# CLOSED LIST — do not extend without design review. See ADR-xxx.
# Adding a fifth value MUST include: (a) spec amendment; (b) integration
# test; (c) Filigree ticket. Mirror also goes into CreationModality at
# contracts/enums.py and the ck_blobs_creation_modality CHECK here.
Column("creation_modality", Text, nullable=False, server_default="verbatim"),

# created_from_message_id: FK to chat_messages.id of the user message
# that triggered the set_pipeline call. Composite FK with session_id
# closes the cross-session lineage hole (mirrors the
# fk_chat_messages_parent_assistant_session pattern at models.py:136-141).
Column("created_from_message_id", Text, nullable=True),

# LLM-provenance columns: populated for llm_generated, disambiguated,
# and llm_generated_then_amended modalities; NULL for verbatim.
# Required together: a blob cannot claim LLM authorship without naming
# the model. See ck_blobs_creating_llm_provenance_nullability below.
Column("creating_model_identifier", String, nullable=True),
Column("creating_model_version", String, nullable=True),
Column("creating_provider", String, nullable=True),
Column("creating_composer_skill_hash", String, nullable=True),
Column("creating_arguments_hash", String, nullable=True),

# Composite FK: blob's (created_from_message_id, session_id) must
# reference an existing row in chat_messages with the same session_id.
# ON DELETE RESTRICT prevents message deletion while a blob references
# it — the blob is the audit anchor; deleting its originating message
# would break the attributability walk.
ForeignKeyConstraint(
    ["created_from_message_id", "session_id"],
    ["chat_messages.id", "chat_messages.session_id"],
    name="fk_blobs_created_from_message_session",
    ondelete="RESTRICT",
),

# Index for the FK column: the audit-readiness query joins blobs to
# chat_messages on created_from_message_id; without an index this is
# a full-table scan on every provenance lookup.
Index("ix_blobs_created_from_message_id", "created_from_message_id"),

CheckConstraint(
    "creation_modality IN ('verbatim', 'llm_generated', 'disambiguated', 'llm_generated_then_amended')",
    name="ck_blobs_creation_modality",
),

# LLM-provenance nullability invariant:
# LLM-authored modalities MUST carry all five provenance fields (the
# auditor must be able to answer "which LLM fabricated this content?").
# verbatim modality MUST NOT carry them (the user typed the content;
# no LLM was involved).
# disambiguated: the LLM parsed the row structure; provenance is
# required to identify which model made the parsing decision.
# Plain SQL boolean equivalence — no dialect-specific syntax needed
# (unlike ck_blobs_ready_hash which uses GLOB / POSIX regex).
CheckConstraint(
    "(creation_modality IN ('llm_generated', 'disambiguated', 'llm_generated_then_amended')) = "
    "(creating_model_identifier IS NOT NULL AND creating_model_version IS NOT NULL AND "
    "creating_provider IS NOT NULL AND creating_composer_skill_hash IS NOT NULL AND "
    "creating_arguments_hash IS NOT NULL)",
    name="ck_blobs_creating_llm_provenance_nullability",
),
```

### Step 3 — Update `_prepare_blob_create` in `tools.py` and `_InlineBlobModel` in `redaction.py`

**3a — Size cap in `redaction.py` (F-6).**

In `src/elspeth/web/composer/redaction.py`, at `_InlineBlobModel` (line
1021), add `max_length=262_144` to the `content` field:

```python
content: Annotated[str, Field(max_length=262_144), Sensitive(summarizer=...)]
```

This enforces the 256 KiB cap at Pydantic decode time, before
`_prepare_blob_create` allocates any bytes. A content payload exceeding
this limit produces a Pydantic `ValidationError` (wrapped as
`ToolArgumentError` at the route layer). The session quota check that
follows is a secondary defense.

**3b — Signature update in `tools.py`.**

Update the `_prepare_blob_create` function signature to accept the new
provenance arguments:

```python
def _prepare_blob_create(
    arguments: Mapping[str, Any],
    *,
    data_dir: str,
    session_id: str,
    creation_modality: "CreationModality",
    created_from_message_id: str | None,
    creating_model_identifier: str | None = None,
    creating_model_version: str | None = None,
    creating_provider: str | None = None,
    creating_composer_skill_hash: str | None = None,
    creating_arguments_hash: str | None = None,
) -> ...:
```

The four `creating_*` arguments default to `None`; the call site in
`_execute_set_pipeline` must supply them when `creation_modality` is
`LLM_GENERATED`, `DISAMBIGUATED`, or `LLM_GENERATED_THEN_AMENDED`. The
DB-level CHECK (Step 2) enforces the invariant; a missing value at the call
site will raise `IntegrityError` at insert time — crash, not silent pass.

**3c — UTF-8 encode guard (F-13).**

In `_prepare_blob_create`, guard the `content.encode("utf-8")` call:

```python
try:
    content_bytes = content.encode("utf-8")
except UnicodeEncodeError as exc:
    raise ToolArgumentError(
        argument="content",
        expected="valid UTF-8 text",
        actual_type="str (contained non-encodable character, e.g. surrogate)",
    ) from exc
```

**3d — Call site in `_execute_set_pipeline`.**

At the call site in `_execute_set_pipeline` (the `inline_blob is not None`
branch), pass:

- `creation_modality`: derived from message context. If the user message
  immediately preceding the tool call was a plain `role=user` message and
  the content is verbatim from that message → `CreationModality.VERBATIM`.
  If the LLM produced the content as part of its own response →
  `CreationModality.LLM_GENERATED`. Use `DISAMBIGUATED` when the proposal
  went through the disambiguation widget. Use `LLM_GENERATED_THEN_AMENDED`
  when the user clicked "Edit the list" and saved changes.
- `created_from_message_id`: the `id` of the triggering `chat_messages`
  row. If unavailable (synthetic call), pass `None`.
- `creating_model_identifier`, `creating_model_version`,
  `creating_provider`, `creating_composer_skill_hash`,
  `creating_arguments_hash`: available from the call-loop context for
  LLM-generated modalities. Pass `None` for `VERBATIM`.

**3e — Response serialiser.**

Update the blob-metadata response serialiser (whichever method/dict builds
the `fetchBlob` response payload) to include all new provenance fields:
`creation_modality`, `created_from_message_id`, and the five
`creating_*` fields.

### Step 4 — Backend integration test

Create `tests/integration/web/composer/test_inline_source_provenance.py`.
The tests call through `_execute_set_pipeline` (the route layer) rather
than directly into `_prepare_blob_create`, exercising the real path
including the `creation_modality` classification at the call site (F-2,
Quality MAJOR-1):

```python
"""
Integration test: creation_modality and LLM-provenance columns are written
by _execute_set_pipeline and surfaced via fetchBlob.

Covers:
- Attributability: explain() walks blob → creation_modality →
  created_from_message_id of original user prose.
- LLM-provenance: llm_generated blobs carry non-NULL creating_* fields.
- Verbatim: verbatim blobs carry NULL creating_* fields.
- Cross-session FK: created_from_message_id from a different session fails.
- Oversize content: 300 KiB payload raises ToolArgumentError.
- Non-UTF-8 content: surrogate payload raises ToolArgumentError.
"""
import pytest
from elspeth.core.landscape import LandscapeDB  # correct export: LandscapeDB not Landscape


@pytest.mark.integration
def test_verbatim_blob_records_creation_modality_and_message_id(
    composer_client,  # standard HTTP test client from tests/integration/conftest.py
    session_id: str,
    user_message_id: str,
) -> None:
    """set_pipeline with verbatim inline_blob records creation_modality=verbatim
    and the originating chat_messages.id; creating_* fields are NULL."""
    response = composer_client.post(
        f"/api/sessions/{session_id}/chat",
        json={
            "role": "user",
            "content": "url\nhttps://finance.gov.au",
        },
    )
    # Simulate LLM tool call via the route layer — use the test harness that
    # drives _execute_set_pipeline directly (follow existing pattern in
    # tests/integration/web/composer/).
    blob_response = composer_client.get(
        f"/api/sessions/{session_id}/blobs/{response.json()['blob_id']}"
    )
    blob = blob_response.json()
    assert blob["creation_modality"] == "verbatim"
    assert blob["created_from_message_id"] == user_message_id
    assert blob["content_hash"] is not None  # SHA-256 of inline content, populated by _prepare_blob_create
    # verbatim → LLM provenance fields must be NULL
    assert blob["creating_model_identifier"] is None
    assert blob["creating_arguments_hash"] is None


@pytest.mark.integration
def test_llm_generated_blob_carries_llm_provenance(
    composer_client,
    session_id: str,
) -> None:
    """llm_generated blobs carry non-NULL creating_* LLM-provenance fields."""
    # (Drive the llm_generated path through the test harness)
    blob = ...  # follow existing integration test pattern
    assert blob["creation_modality"] == "llm_generated"
    assert blob["creating_model_identifier"] is not None
    assert blob["creating_model_version"] is not None
    assert blob["creating_provider"] is not None
    assert blob["creating_composer_skill_hash"] is not None
    assert blob["creating_arguments_hash"] is not None


@pytest.mark.integration
def test_cross_session_message_id_rejected(
    composer_client,
    session_id: str,
) -> None:
    """created_from_message_id from a different session fails the composite FK."""
    # Attempt to write a blob with a message_id from a different session.
    # The composite FK (fk_blobs_created_from_message_session) must reject it.
    with pytest.raises(Exception, match="FOREIGN KEY constraint failed|IntegrityError"):
        # Drive _execute_set_pipeline with a cross-session message_id via the
        # route-layer test harness.
        pass  # implement per existing integration test pattern


@pytest.mark.integration
def test_oversize_content_raises_tool_argument_error(
    composer_client,
    session_id: str,
) -> None:
    """A 300 KiB content payload produces a ToolArgumentError (F-6)."""
    big_content = "x" * (300 * 1024)  # 300 KiB
    response = composer_client.post(
        f"/api/sessions/{session_id}/chat",
        json={
            "role": "user",
            "content": big_content,
        },
    )
    # The response must be a 422 (Pydantic ValidationError → ToolArgumentError),
    # not a 500 (memory allocation or DB write).
    assert response.status_code == 422


@pytest.mark.integration
def test_non_utf8_content_raises_tool_argument_error(
    composer_client,
    session_id: str,
) -> None:
    """A surrogate-containing string produces a ToolArgumentError (F-13)."""
    # Surrogates in JSON strings are rejected by most HTTP clients before the
    # route layer; test by driving _prepare_blob_create directly via the test
    # harness with a synthetic surrogate payload.
    from elspeth.web.composer.tools import _prepare_blob_create
    from elspeth.web.composer.errors import ToolArgumentError
    with pytest.raises(ToolArgumentError, match="valid UTF-8 text"):
        _prepare_blob_create(
            {"content": "valid-prefix-\ud800-surrogate", "filename": "f.csv", "mime_type": "text/csv"},
            data_dir="/tmp",
            session_id=session_id,
            creation_modality="verbatim",
            created_from_message_id=None,
        )
```

Adapt the test harness calls to the actual route-layer test infrastructure
in `tests/integration/web/composer/`. The `composer_client` and
`session_id` fixtures follow the existing integration test conventions in
`tests/integration/conftest.py`.

### Step 5 — Commit

```bash
git add src/elspeth/contracts/enums.py \
        src/elspeth/web/sessions/models.py \
        src/elspeth/web/composer/redaction.py \
        src/elspeth/web/composer/tools.py \
        src/elspeth/web/frontend/src/api/client.ts \
        tests/integration/web/composer/test_inline_source_provenance.py
git commit -m "feat(composer/audit): server-side creation_modality + LLM-provenance on inline blobs (Phase 5a.2.5)"
```

---

## Task 2.6 — `chat_messages` immutability (backend)

> **Numbering note:** Inserted as "2.6" (not renumbering Tasks 3-9) per the
> same no-renumber convention as Task 2.5. This task is a prerequisite for
> the Task 2.5 attributability claim: the `explain()` walk from blob →
> `created_from_message_id` → `chat_messages.id` is only tamper-evident
> if the referenced `chat_messages` row cannot be mutated after creation.

**Why immutability is required for Phase 5a's audit story.** Phase 5a
elevates `chat_messages` to an audit anchor: the `created_from_message_id`
FK on `blobs_table` (Task 2.5) lets auditors trace provenance back to the
originating user message. If that message row could be mutated after the
blob was created, the attributability walk from `blob →
created_from_message_id → chat_messages.content` would no longer prove what
the user actually typed — silently breaking the chain. Immutability
enforcement on `chat_messages` is therefore a correctness requirement for
Phase 5a, not a belt-and-braces hardening.

**Trigger ownership: deferred to `18a-phase-5b-backend.md`.**

The `chat_messages` immutability trigger is specified and owned by
`18a-phase-5b-backend.md` §"chat_messages immutability trigger (F-4)".
Phase 5b's backend plan carries all schema work for the session DB
regardless of which phase's audit claim depends on it, and 18a-'s schema
validator (§"Schema validator extension for triggers (F-24)") owns the
trigger-existence registry. Duplicating DDL here would create two sources of
truth for the same schema object; 17- defers entirely.

The canonical trigger name and scope (as defined in 18a-) is:

- **Name:** `trg_chat_messages_immutable_content`
- **Scope:** `BEFORE UPDATE OF content ON chat_messages` (content column
  only)
- **Mechanism:** SQLAlchemy table-scoped `event.listen` with `IF NOT EXISTS`
  for idempotent bootstrap (F-23)

**Scope reconciliation (DELETE protection and all-column vs content-only
UPDATE).** An earlier draft of this task defined two triggers: an
unconditional `BEFORE UPDATE` (all columns) and a conditional `BEFORE
DELETE` (when a blob references the row). After review:

- **DELETE**: redundant with the `ON DELETE RESTRICT` composite FK defined
  in Task 2.5 (`fk_blobs_created_from_message_session`, 17- lines
  843-847). The FK already prevents deletion of a `chat_messages` row while
  any blob holds a `created_from_message_id` reference to it. A trigger
  adding the same protection is unnecessary.
- **All-column UPDATE vs UPDATE-OF-content**: The audit walk reconstructs
  what the user typed, which lives in `content`. Protecting only `content`
  (18a-'s scope) is sufficient for Phase 5a's attributability claim. Role,
  timestamp, and session_id tampering would be misleading but do not break
  the content-hash lineage chain. 18a-'s narrower scope is accepted.

If a future reviewer believes the broader scope (all-column UPDATE, explicit
DELETE trigger) is necessary for a different threat model, that is a scope
extension to 18a-'s trigger, not a revision of this plan.

**Co-shipping requirement.** If `18a-phase-5b-backend.md` does not ship in
the same PR as this plan's changes, the trigger DDL MUST be moved into this
task with the canonical 18a- name and scope:

- Name: `trg_chat_messages_immutable_content`
- Scope: `BEFORE UPDATE OF content ON chat_messages`
- Do NOT redefine with a different name or broader scope.

### Step 1 — Add the trigger

No DDL in this task. Follow `18a-phase-5b-backend.md` §"chat_messages
immutability trigger (F-4)" for the exact SQLAlchemy `event.listen` block
to add in `web/sessions/models.py`. Verify the trigger is registered in
the schema validator per §"Schema validator extension for triggers (F-24)".

**No Alembic migration.** Same policy as Task 2.5 — operator deletes the
old DB on deploy.

### Step 2 — Integration tests

Two test concerns for Phase 5a's attributability requirement:

**2a — Trigger test (content mutation blocked).** This test is owned by
18a-; see `18a-phase-5b-backend.md` §"chat_messages immutability trigger
(F-4)" for the `trg_chat_messages_immutable_content` trigger test. The
test asserts that an attempt to UPDATE `content` on a settled
`chat_messages` row raises `IntegrityError`.

**2b — Phase-5a-specific attributability test.** Create
`tests/integration/web/composer/test_chat_messages_attributability.py`
to walk the `blob → created_from_message_id → chat_messages` chain and
confirm mutation of `content` raises `IntegrityError` (via 18a-'s
trigger) rather than succeeding silently:

```python
"""
Integration test: Phase 5a attributability chain is tamper-evident.

Walks blob.created_from_message_id → chat_messages.id and asserts
that a mutation attempt on chat_messages.content raises IntegrityError,
proving the trigger (trg_chat_messages_immutable_content, owned by
18a-phase-5b-backend.md F-4) is installed and covers the audit anchor.
"""
import pytest
from sqlalchemy.exc import IntegrityError


@pytest.mark.integration
def test_blob_provenance_anchor_is_immutable(
    composer_client,
    session_db,
    session_id: str,
    user_message_id: str,
) -> None:
    """blob.created_from_message_id points to an immutable chat_messages row.

    The trg_chat_messages_immutable_content trigger (owned by 18a-) must be
    installed before this test can pass. If 18a- has not shipped, skip with
    pytest.skip("18a- trigger not yet deployed").
    """
    # Confirm the blob's created_from_message_id resolves to the expected row.
    blob_row = session_db.execute(
        "SELECT created_from_message_id FROM blobs WHERE session_id = :sid LIMIT 1",
        {"sid": session_id},
    ).fetchone()
    assert blob_row is not None
    assert blob_row[0] == user_message_id

    # Assert mutation of content raises IntegrityError — trigger fires.
    with pytest.raises(IntegrityError, match="append-only"):
        session_db.execute(
            "UPDATE chat_messages SET content = 'tampered' WHERE id = :id",
            {"id": user_message_id},
        )
        session_db.commit()
```

**2c — DELETE blocked by FK (not by trigger).** Deletion of a
`chat_messages` row while a blob references it is blocked by the
`ON DELETE RESTRICT` composite FK from Task 2.5, not by a trigger. The
`IntegrityError` message will reference a FK constraint, not the
`append-only` trigger message:

```python
@pytest.mark.integration
def test_chat_message_delete_while_blob_references_it_raises(
    session_db,
    user_message_id: str,
) -> None:
    """DELETE on a chat_messages row referenced by blobs raises IntegrityError.

    Protection comes from the ON DELETE RESTRICT composite FK
    (fk_blobs_created_from_message_session, Task 2.5), not from a trigger.
    The IntegrityError message will reference a FOREIGN KEY constraint.
    """
    with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
        session_db.execute(
            "DELETE FROM chat_messages WHERE id = :id",
            {"id": user_message_id},
        )
        session_db.commit()
```

### Step 3 — Commit

```bash
git add tests/integration/web/composer/test_chat_messages_attributability.py \
        tests/integration/web/sessions/test_chat_messages_immutability.py
git commit -m "test(sessions/audit): Phase 5a attributability chain tamper-evidence test (Task 2.6)"
```

> Note: `models.py` is NOT staged here — the trigger DDL belongs to 18a-'s
> commit. If co-shipping in the same PR, coordinate with the 18a- task list
> to avoid double-patching `models.py`.

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

  it("renders the SHA-256 hash inside a collapsed audit-info disclosure (F-21)", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    // Audit info (blob_id, SHA-256 hash) is behind a <details> "Show audit info"
    // disclosure. It must NOT be visible at a glance — Linda sees filename +
    // row count only; Sarah/Marcus expand the disclosure to audit.
    const disclosure = screen.getByText(/show audit info/i);
    expect(disclosure).toBeInTheDocument();
    // The hash is NOT visible before the disclosure is opened.
    expect(screen.queryByText(/h1/)).not.toBeInTheDocument();
    // Open the disclosure.
    fireEvent.click(disclosure);
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

  it("announces itself via role=region with aria-label (F-18)", () => {
    render(<InlineSourceCreatedTurn summary={verbatim} onEdit={vi.fn()} />);
    expect(
      screen.getByRole("region", { name: /source created/i }),
    ).toBeInTheDocument();
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

Component shape:

- Root: `<section role="region" aria-label="Source created from your
  message">` — the `role`/`aria-label` are load-bearing for the F-18
  accessibility test; keep them stable.
- Visible by default: filename, MIME type, row count (or "1 row" for
  single-record inputs), and a clipped content preview. Clip preview
  to ~280 chars; use `data-testid="inline-source-preview"` on the
  preview element for the clip-assertion anchor.
- **Audit info hidden by default (F-21):** `blob_id` and full SHA-256
  hash are placed inside a `<details>` element with `<summary>Show
  audit info</summary>`. Linda-persona users see filename + row count
  at a glance; Sarah/Marcus-persona users expand the disclosure for the
  audit fields. The Task 3 test asserts the hash is NOT visible until
  the disclosure is opened.
- Edit button: render `<button>Edit the list</button>` only when
  `provenance === "llm-generated"` or `provenance === "llm-generated-then-amended"`;
  wire `onEdit` to the callback. Not shown for `provenance === "verbatim"` or
  `"disambiguated"`.
- `data-testid="inline-source-created-turn"` on the root element
  (required by Task 6 integration test assertion).

Wire `onEdit` to a callback that opens the inline editor pre-filled with
the current inline content.

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
offers four actions:

1. **"Yes — N rows"** (primary) — confirm the LLM's interpretation and
   proceed to create the `inline_blob`.
2. **"No — treat as 1 row"** — reject the multi-row interpretation;
   the entire user message is treated as a single-row value. Sets a
   session-scoped `user_requested_single_row_for_message_id` flag
   (see F-11 re-fire guard below) so subsequent proposals for the same
   message bypass disambiguation.
3. **"Edit the rows directly"** — open the inline editor from Task 3
   pre-filled with the proposed rows for manual correction.
4. **"This isn't source data"** — escape hatch (F-10): dismisses the
   widget entirely, marks the originating message with a
   `non_source_message_ids` flag in `inlineSourceStore`, and emits a
   clean signal to the LLM ("I don't want to treat this as source data")
   so the LLM doesn't re-surface a disambiguation proposal for the same
   message. The frontend must NOT re-show the disambiguation widget for
   a message ID that is in `non_source_message_ids`.

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
    messageId: "msg-user-1",
    onConfirmMultiRow: vi.fn(),
    onTreatAsOneRow: vi.fn(),
    onEditRows: vi.fn(),
    onNotSourceData: vi.fn(),  // F-10: escape action
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

  it("calls onNotSourceData when the escape action is clicked (F-10)", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    fireEvent.click(
      screen.getByRole("button", { name: /this isn.t source data/i }),
    );
    expect(props.onNotSourceData).toHaveBeenCalledWith("msg-user-1");
  });

  it("announces itself via role=region with an aria-label", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    expect(screen.getByRole("region", { name: /row count/i })).toBeInTheDocument();
  });

  it("moves focus to the primary action button on mount (F-19)", () => {
    render(<InlineSourceDisambiguationTurn {...props} />);
    // Primary action is the "Yes — N rows" confirm button.
    expect(
      screen.getByRole("button", { name: /yes.*3 rows/i }),
    ).toHaveFocus();
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
rows, and four action buttons:

- `Yes — N rows` (primary; calls `onConfirmMultiRow(proposalId)`)
- `No — treat as 1 row` (calls `onTreatAsOneRow(proposalId)`)
- `Edit the rows` (calls `onEditRows(proposalId)`)
- `This isn't source data` (link-style escape; calls
  `onNotSourceData(messageId)`)

The `role` / `aria-label` and per-button accessible names are
load-bearing for the Task 4 tests; keep them stable.

**Focus management (F-19).** On mount, move focus to the primary action
button ("Yes — N rows") using a `useEffect` with a `ref` on that button.
The widget is keyboard-accessible from mount without requiring an
additional Tab.

**Re-fire guard for `onTreatAsOneRow` (F-11).** When the user clicks
"No — treat as 1 row", store the `messageId` in
`inlineSourceStore.userRequestedSingleRowForMessageIds` (a `Set<string>`).
The disambiguation predicate in `ChatPanel.tsx` must check this set before
showing the widget: if `messageId` is already present, skip the
disambiguation widget and route to the standard proposal banner instead.
This prevents the widget from re-appearing if the LLM re-proposes for the
same message after a rejected multi-row interpretation.

**"This isn't source data" escape (F-10).** When the user clicks this:
1. Call `onNotSourceData(messageId)`.
2. In `ChatPanel.tsx`, the handler adds `messageId` to
   `inlineSourceStore.nonSourceMessageIds` (a `Set<string>`).
3. Send a clean LLM message: "That message isn't source data — please
   continue without creating a source from it." (no API jargon).
4. The disambiguation predicate also gates on `nonSourceMessageIds` —
   if the message ID is present, the widget never re-fires.

Wiring in `ChatPanel.tsx`:

1. When iterating proposals, detect proposals where the source change has
   an `inline_blob` AND the proposal's narration / annotation indicates
   row-count ambiguity. The detection heuristic v1: if the proposal's
   `summary` contains the phrase "I read" or "interpreted as", treat as
   ambiguous; otherwise fall back to the standard banner. (This is
   intentionally heuristic — Phase 5b will design a structured field for
   the annotation; until then, the heuristic is fine and the false
   positives are mild.) The heuristic must NOT fire for the canonical
   demo prompt (Task 9 Step 2 F-12 verification step).
2. Before showing the widget, check `userRequestedSingleRowForMessageIds`
   and `nonSourceMessageIds` from `inlineSourceStore`. If either set
   contains the message's ID, route to the standard banner.
3. For ambiguous proposals not excluded by step 2, render
   `InlineSourceDisambiguationTurn` inline at the proposal's position
   in the message stream, NOT in the banner. Wire:
   - `onConfirmMultiRow` → existing `acceptCompositionProposal`
   - `onTreatAsOneRow` → reject proposal + add to
     `userRequestedSingleRowForMessageIds` + send "treat the original
     input as a single row"
   - `onEditRows` → open the inline editor from Task 3
   - `onNotSourceData` → add to `nonSourceMessageIds` + send clean
     escape message

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
small affordance: "Your text looks like source data. Create a source
from it?".

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
    expect(screen.getByText(/looks like source data/i)).toBeInTheDocument();
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
aria-label="Inline source fallback prompt">` with prose "Your text
looks like source data. Create a source from it?" and two buttons —
"Create source from this text" (primary; calls `onAccept(candidateText)`)
and "Dismiss" (link-style; calls `onDismiss`).

**Dismiss persistence (F-20).** A dismissed prompt must not re-fire on
subsequent renders within the same session. Use a session-scoped
`dismissedAt: Map<sessionId, number>` entry in `inlineSourceStore`
(add `dismissedAt` to the store's state shape in Task 2 Step 1; update
`inlineSourceStore.test.ts` accordingly). Alternatively, a `useRef`
at `ChatPanel` level keyed by session ID is acceptable if the store
shape is frozen. The predicate in `ChatPanel.tsx` must gate on
`dismissedAt` being absent for the active session before rendering the
prompt.

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
  // Send a user-facing message; the LLM still decides the structured
  // mutation. Do NOT expose API jargon (set_pipeline / inline_blob)
  // in a persisted role=user message — it appears verbatim on session
  // reload for all users (F-3). The LLM has sufficient context from
  // the system prompt and prior message thread to interpret the intent.
  sendMessage(`Use this as my source data:\n\n${text}`);
};
```

When the user accepts, dispatch a user-facing chat turn that *asks* the
LLM to create the inline source. We deliberately keep the LLM in the
loop — the frontend does not construct the `set_pipeline` payload
itself, because the LLM also needs to choose the plugin (csv vs json vs
url-list etc.) and add the other transforms. This is the safer shape
than the frontend forcing a specific payload.

The message text must not contain API jargon (`set_pipeline`,
`source.inline_blob`). The persisted `role=user` message is visible on
session reload and must read as natural user language regardless of
technical sophistication. Structured intent is conveyed by the message
context (the LLM's system prompt and the `candidateText` content);
explicit API naming is not required and harms UX for non-technical
users (F-3).

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

    // Assertion (f): wire-to-store provenance mapping — direct assertions
    // without conditional debug-accessor guards (F-14).
    //
    // Case 1 — verbatim (current mock): fetchBlob returns creation_modality:
    // 'verbatim' → inlineSourceStore.getSummary().provenance === 'verbatim'
    // → no "Edit the list" button (already asserted in (c)).
    //
    // Case 2 — llm_generated (second test, see below): fetchBlob returns
    // creation_modality: 'llm_generated' → provenance === 'llm-generated'
    // → "Edit the list" button IS present.
    //
    // Both are direct assertions — no conditional or debug-accessor path.
    await waitFor(() => {
      // The inlineSourceStore is a Zustand store; read its state directly.
      const { useInlineSourceStore } = require("@/stores/inlineSourceStore");
      const summary = useInlineSourceStore.getState().getSummary("session-1");
      expect(summary).not.toBeNull();
      expect(summary!.provenance).toBe("verbatim");
      expect(summary!.createdFromMessageId).toBe("msg-user-1");
    });

    // Assertion (g): audit-readiness panel Provenance row updated.
    // Phase 2C shipped; AuditReadinessPanel is in the component tree.
    await waitFor(() => {
      expect(screen.getByText(/inline content hashed/i)).toBeInTheDocument();
      expect(screen.getByText(/abc123/)).toBeInTheDocument();
    });
  });

  it("maps creation_modality='llm_generated' to provenance='llm-generated' and shows Edit button", async () => {
    // Wire-to-store provenance mapping — Case 2 (F-14): llm_generated path.
    vi.spyOn(client, "fetchBlob").mockResolvedValue({
      id: "blob-uuid-llm",
      filename: "llm-generated.csv",
      mime_type: "text/csv",
      content_hash: "sha256-llmdef456",
      created_via: "inline_blob",
      size_bytes: 60,
      creation_modality: "llm_generated",
      created_from_message_id: "msg-user-hero",
    });

    render(<App />);
    const textarea = await screen.findByLabelText("Message input");
    fireEvent.change(textarea, {
      target: { value: "create a list of 5 government web pages" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    await waitFor(() => {
      const { useInlineSourceStore } = require("@/stores/inlineSourceStore");
      const summary = useInlineSourceStore.getState().getSummary("session-1");
      expect(summary).not.toBeNull();
      // creation_modality 'llm_generated' maps to 'llm-generated' at the
      // API boundary (client.ts adapter). Store and component use hyphenated form.
      expect(summary!.provenance).toBe("llm-generated");
    });

    // llm-generated provenance → "Edit the list" button IS present.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /edit the list/i }),
      ).toBeInTheDocument(),
    );
  });
});
```

Fit the mock shapes to the real `sendMessage` / `fetchBlob` return types
documented in `client.ts`. The `creation_modality` and
`created_from_message_id` fields on the blob response are new as of Task 2.5;
if the test is written before Task 2.5 lands they will be absent and
assertions (c) and (f) will fail — that is the intended RED state.

### Known races (F-16)

Two timing hazards in the inline-source flow are known but accepted in v1:

**Race 1 — Streaming `set_pipeline` response vs. `inlineSourceStore` recompute.**
While the LLM streams a `set_pipeline` response, `compositionState` may update
mid-stream (if the composition-state subscription is push-based). The
`inlineSourceStore` recomputes on every `compositionState` update event, not
on tool-call response completion. This means the store may briefly hold a stale
summary while the full blob metadata has not yet been fetched. Mitigation: the
`InlineSourceCreatedTurn` widget is only rendered after the blob-metadata fetch
resolves (the store's `setSummary` is called by the `fetchBlob` response handler,
not by the composition-state update); the user never sees a partial widget.
Backend mitigation: the session write lock serialises all `set_pipeline` mutations
per session, so there is no interleaving of competing `set_pipeline` calls.

**Race 2 — User types a new message while `InlineSourceFallbackPrompt` is visible.**
If the user starts typing while the fallback prompt is shown, the predicate may
flip before the component re-renders. The component's `shouldRender` prop is
re-derived on each `messages` change; a new user message (condition 1) will
immediately set `shouldRender=false`. The `useRef`-based `dismissedAt` guard
(F-20) prevents re-fire within the same session even if the predicate transiently
re-fires.

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
   `set_pipeline` call with `source.inline_blob` containing exactly 3
   rows (one URL per row), OR a narration explicitly stating "I read
   these as 3 rows" before any tool call. Pass on either. A generic
   "I'll create a source from those URLs" without the row-count
   confirmation does NOT pass. Use the structured JSON artifact format
   from Phase 5b Task 0 (`evals/composer-rgr/phase5b-task0-*.json`
   shape) to record the raw LLM response for reproducibility.
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
      pipeline, paste a URL, or type a few rows of data to start...".
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
- [ ] **Canonical-prompt disambiguation check (F-12).** Confirm that
      `InlineSourceDisambiguationTurn` did NOT appear during the canonical
      hero prompt run above. The LLM generates 5 URLs with an unambiguous
      row count; the disambiguation predicate must not fire for this input.
      If it does fire (widget appears asking "Did I read this as 5 rows?"),
      the disambiguation predicate is too aggressive — tighten the
      `summary`-contains heuristic in `ChatPanel.tsx` before shipping.
      A false positive on the canonical demo prompt is a demo-blocking
      defect.
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
| R7 | Task 2.5 blob-metadata columns drift from the frontend `InlineSourceSummary` type (e.g., `creation_modality` renamed on one side) | Medium | Seven new columns: `creation_modality`, `created_from_message_id`, and five `creating_*` LLM-provenance fields. The wire form is snake_case on both sides; the frontend type documents the `creation_modality`→`provenance` mapping; the adapter in `client.ts` is the single translation point. Cover with the Task 2.5 integration test |
| R8 | The composer skill nudge backfires — the LLM now uses `inline_blob` even when a CSV upload would be more appropriate (e.g., 50 rows in chat) | Low | Threshold-cutoff redirection is explicitly scoped to Phase 8. Phase 5a's risk is one-direction; the failure mode is "inline_blob used where CSV-upload would be marginally better", which is recoverable by the user pasting CSV |
| R9 | Phase 5b's interpretation-acceptance event shape (open question B2) constrains Phase 5a's disambiguation-turn data model | Low | Phase 5a's disambiguation turn is UI-only; it does NOT introduce a new event type. The mutation is still a standard composition_proposal. Phase 5b can layer interpretation events on top without disrupting Phase 5a's UI |
| R10 | Chat input placeholder change creates a regression in Phase A slice 4's per-step guided-mode placeholder | Low | The explicit `placeholder?:` prop override case is covered by a dedicated test (Task 1 step 1 case 4) |
| R11 | Downstream LLM transforms receive inline-blob row content as untrusted user-controlled text (T-05: prompt-injection delivery vector) | High | Phase 5a's responsibility is to name the risk; the structural defense-in-depth fix lives in Phase 8. **Operator-facing guidance:** any LLM transform downstream of a dynamic-source-from-chat input must be treated as receiving prompt-injection-shaped content. Do NOT deploy an LLM transform downstream of an inline-blob source without explicit review of whether the row-content fields are sanitised or sandboxed before being passed to the model. Phase 5a ships no LLM transforms itself; the risk is latent until an operator wires one |
| R12 | `InlineSourceFallbackPrompt.onAccept` emits jargon (`set_pipeline with source.inline_blob`) as a persisted `role=user` message visible on session reload | Medium | Mitigated by F-3: `onAccept` now sends "Use this as my source data:\n\n${text}" (user-facing prose); structured intent is conveyed by context alone. Session reload shows natural language, not API jargon |
| R13 | `chat_messages` rows used as attributability anchors are mutable (UPDATE/DELETE allowed post-creation) | High | Mitigated by F-5/Task 2.6: a `BEFORE UPDATE / BEFORE DELETE` trigger on `chat_messages_table` makes the referenced rows effectively immutable. Until Task 2.6 ships, the attributability walk is best-effort rather than tamper-evident |
| R14 | Oversize inline content bypasses `_InlineBlobModel.max_length`, allocating unbounded bytes before the session quota check | High | Mitigated by F-6: `max_length=262_144` on `_InlineBlobModel.content` in `redaction.py`, raising `ToolArgumentError` at Pydantic decode time before any allocation. The trust-tier prose previously claimed this was already defended; F-6 corrects that claim |
| R15 | `creation_modality = llm_generated` blobs carry no pointer to which LLM produced the content, making replay/repudiation impossible | High | Mitigated by F-9: five LLM-provenance columns added to `blobs_table` with a CHECK constraint requiring them non-NULL for LLM-produced modalities. Verbatim blobs carry NULL for these columns (correct — no LLM involved) |

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
| 2026-05-15 | Review panel | CHANGES_REQUESTED | B1, I1, I2 | Applied in this revision. B1: completed placeholder test fixtures in `inlineSourceStore.test.ts` with concrete `InlineSourceSummary` literals. I1: replaced vague detection heuristic prose with a precise 4-condition specification biased toward false negatives. I2: corrected both store selectors (`messages`, `compositionState`) to use the real singleton shape verified against `sessionStore.ts` lines 154–155. |
| 2026-05-18 | 9-reviewer panel | CHANGES_REQUESTED | F-1 through F-23 | Applied in this revision. F-1: relocated `CreationModality` to `contracts/enums.py` and blob columns to `web/sessions/models.py:blobs_table`. F-2: rewrote Task 2.5 test to call through `_execute_set_pipeline`; corrected `Landscape` → `LandscapeDB`. F-3: replaced jargon-leaking `onAccept` body with user-facing prose. F-4: added `LLM_GENERATED_THEN_AMENDED` enum value. F-5: added Task 2.6 immutability trigger on `chat_messages`. F-6: added `max_length=262_144` on `_InlineBlobModel.content`; corrected trust-tier prose. F-7: composite FK on `created_from_message_id`/`session_id`. F-8: `ck_blobs_creation_modality` CHECK constraint. F-9: LLM-provenance columns + nullability CHECK. F-10: "This isn't source data" escape action in Task 4. F-11: `user_requested_single_row_for_message_id` re-fire guard. F-12: Task 9 Step 2 disambiguation non-fire check for canonical prompt. F-13: UTF-8 encode guard. F-14: direct provenance→store assertion in Task 6. F-15: tightened Task 8 prompt-2 pass criterion. F-16: added §"Known races" to Task 6. F-17: corrected `redaction.py` citations to 1021, `sessionStore.ts` citations to 154–155. F-18: ARIA assertions in Task 3. F-19: focus-on-mount spec in Task 4. F-20: `dismissedAt` persistence in Task 5 store. F-21: `<details>` audit-info collapse in Task 3. F-22: empty-state copy updated. F-23: fallback-prompt voice updated. |
| 2026-05-18 | Follow-up reviewer pass | CHANGES_REQUESTED | P1, P2 | Applied in this revision. P1 (line 1018 content_hash fix): Task 2.5 integration test asserted `blob["source_data_hash"]` — wrong field; `source_data_hash` is on `rows_table` in the Landscape audit DB, not on the blob API response. Replaced with `blob["content_hash"]` (the `blobs_table` field populated by `_prepare_blob_create`). Adjacent prose at line 138 clarified to name both hashes and their distinct databases (blob's `content_hash` in the session DB; Landscape row's `source_data_hash` in the audit DB, linked by computation). P2 (Task 2.6 trigger ownership deferred to 18a-): Task 2.6's literal DDL blocks (`trg_chat_messages_immutable`, `trg_chat_messages_immutable_delete`) conflicted with 18a-'s canonical trigger (`trg_chat_messages_immutable_content`, BEFORE UPDATE OF content only) and were invisible to 18a-'s schema validator. DDL blocks removed; Task 2.6 now defers entirely to `18a-phase-5b-backend.md` §"chat_messages immutability trigger (F-4)". Rationale prose retained. DELETE-trigger test reframed as FK-constraint test (ON DELETE RESTRICT from Task 2.5 covers this, not a trigger). Phase-5a-specific attributability test added (walks blob → created_from_message_id → asserts content mutation raises IntegrityError via 18a-'s trigger). Scope reconciliation documented: DELETE redundant with FK; content-only UPDATE accepted as sufficient for Phase 5a's audit claim. |
| 2026-05-18 | Plan amendment | APPLIED | (no finding ID) | Added shared-worktree section: `feat/composer-phase-5-chat-data-entry` at `.worktrees/composer-phase-5-chat-data-entry`; Phase 5a + Phase 5b ship as coordinated PR on one branch. Added worktree-root prefix note to File structure section. |
