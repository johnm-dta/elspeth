# Web UX & Composer Agent — Requirements from User/Agent Feedback

**Date:** 2026-04-08
**Source:** User direct feedback + composer agent operational feedback
**Status:** Draft — needs prioritization review

---

## Overview

This document distills feedback from two sources into actionable requirements:

1. **User direct feedback** — observations from using the web pipeline builder UI
2. **Composer agent feedback** — friction points encountered by the LLM agent operating the composer MCP tools

Requirements are grouped into five areas:
- [A] Frontend UX
- [B] Agent Tooling (composer MCP)
- [C] Skill Pack Knowledge
- [D] Communication & Language
- [E] Architectural / Cross-Cutting

---

## [A] Frontend UX Requirements

### A1. Per-user file management with categorized folders

**Problem:** Users cannot easily see or manage the files associated with their pipelines.

**Requirement:** Each user shall have a visible file area organized into:
- **Source files** — inputs to pipelines (uploaded CSVs, URLs, JSON, etc.)
- **Sink files** — pipeline outputs (generated JSON, CSV, reports)
- **Other files** — prompts, templates, configuration snippets

**Behavior:**
- When the agent creates files (blobs, outputs), they appear immediately in the file panel — not hidden in the background
- Files are scoped to the user account
- File panel supports browse, preview, download, and delete

---

### A2. Markdown and Mermaid rendering in chat

**Problem:** Agent responses contain structured content (pipeline descriptions, diagrams, status summaries) that renders as plain text.

**Requirement:** The chat panel shall render agent responses as rich markdown, including:
- Headings, lists, tables, code blocks
- Mermaid diagrams (flowcharts, sequence diagrams)
- Inline code formatting

**Note:** This applies to the `MessageBubble.tsx` component. A markdown renderer (e.g., `react-markdown` + `remark-mermaid` or `mermaid.js`) is needed.

---

### A3. Visible error/warning routing to the agent

**Problem:** Errors and warnings from ELSPETH pipeline operations go "in the background" — users can't see that the agent is aware of them or acting on them.

**Requirement:** When validation errors or warnings occur:
- They shall be visibly surfaced in the chat as agent-attributed messages
- The agent shall acknowledge errors/warnings and explain them in plain language
- Users shall see that the error went *to* the agent (not just to a log)

**Behavior:**
- Errors appear as distinct message types in the chat (not buried in tool output)
- Warning/error severity is visually distinguished (color, icon)
- The agent's response to an error is clearly linked to the triggering error

---

### A4. Even panel split with persistent layout preference

**Problem:** The two-panel layout (chat + inspector) is not evenly split, and resizing doesn't persist.

**Requirement:**
- Default layout shall be approximately 50/50 split between chat and inspector panels
- Panel split ratio shall be user-adjustable (drag handle)
- The chosen ratio shall persist across sessions, stored in either:
  - User account preferences (if authenticated), or
  - Local storage (browser-side)

---

### A5. Secrets button redesign and relocation

**Problem:** The secrets management button uses a cog icon (implying general settings) and is not co-located with related file/attachment actions.

**Requirement:**
- The secrets button shall be placed next to the file button and paperclip (attachment) button in the input toolbar
- The icon shall visually represent secrets/credentials (e.g., key icon, lock icon, shield icon) — not a cog
- Grouping: `[Attach file] [Browse files] [Secrets]` in the input toolbar area

---

### A6. Inline validation status on pipeline components

**Problem:** Validation state is not visually attached to the pipeline components that caused it.

**Requirement:** The pipeline graph view (`GraphView.tsx`) shall show validation state per node:
- **Green:** valid, no issues
- **Yellow:** valid with warnings
- **Red:** has blocking errors

Warnings and errors shall be attached to the specific component (source, transform, sink) that triggered them, visible on hover or inline.

---

### A7. Pipeline status model: valid / invalid / valid-with-warnings

**Problem:** No clear distinction between "structurally valid," "runnable," and "has non-blocking warnings."

**Requirement:** The UI shall distinguish three pipeline states:
- **Valid** — no errors, no warnings, runnable
- **Valid with warnings** — runnable but has caveats (yellow indicator)
- **Invalid** — has blocking errors, cannot run (red indicator)

This status shall be visible in both the inspector panel and the chat.

---

## [B] Agent Tooling Requirements (Composer MCP)

### B1. In-session blob/file creation tool

**Problem:** Many pipelines need a tiny seed input (one URL, a short JSON object, a few CSV rows). Requiring manual upload breaks conversational flow.

**Requirement:** Add a `create_blob` MCP tool:

```
create_blob:
  filename: string
  mime_type: string (text/plain, application/json, text/csv)
  content: string
Returns:
  blob_id: string
  storage_path: string
  size: integer
  mime_type: string
```

**Behavior:**
- Immediately creates a session-visible file
- Appears in user's file panel (requirement A1) under "Source files"
- Can be passed directly to `set_source_from_blob`
- Supported MIME types: `text/plain`, `application/json`, `text/csv`

**Companion tools:**
- `update_blob` — modify content of existing blob
- `delete_blob` — remove a blob
- `get_blob_content` — retrieve content for inspection

**Nice-to-have:** `create_tabular_blob` — accepts rows as JSON array, serializes to JSON or CSV automatically.

---

### B2. Consistent validation payloads from all mutation tools

**Problem:** Validation state is returned inconsistently across mutation tools.

**Requirement:** Every mutation tool (`set_source`, `upsert_node`, `upsert_edge`, `set_output`, `patch_*`, `remove_*`) shall return a standardized validation payload:

```json
{
  "is_valid": true,
  "errors": [],
  "warnings": [{"component": "source", "message": "...", "severity": "medium"}],
  "suggestions": []
}
```

This is partially implemented in `ToolResult` already — ensure all tools conform consistently.

---

### B3. Source patching parity with node/output patching

**Problem:** Source updates feel more fragile than node/output edits.

**Requirement:** `patch_source_options` shall accept structured patches with the same ergonomics as `patch_node_options` and `patch_output_options`. Specifically:
- Partial updates (not full replacement)
- Clear merge semantics
- Consistent error reporting

---

### B4. Blob ID abstraction over raw file paths

**Problem:** Raw storage paths leak infrastructure details into agent and user workflows.

**Requirement:** All user-facing and agent-facing tools shall prefer blob IDs over raw file paths. Internal storage paths shall be an implementation detail, not exposed in tool inputs/outputs or the UI.

---

### B5. Pipeline diff/change summary after edits

**Problem:** After a series of edits, it's unclear what changed.

**Requirement (nice-to-have):** Add a `diff_pipeline` or `summarize_changes` tool that returns:
- What nodes/edges/outputs were added, modified, or removed
- What warnings were introduced or resolved
- Before/after comparison

---

## [C] Skill Pack Knowledge Requirements

### C1. Plugin capabilities registry

**Problem:** The agent rediscovers plugin capabilities every session through repeated `list_*` and `get_plugin_schema` calls.

**Requirement:** Add a machine-readable capabilities registry per plugin to the skill pack, including:
- Input type (file/blob/path/connection)
- Supports schema config (yes/no)
- Supports batching (yes/no)
- Emits structured fields (field names)
- Requires secrets (yes/no, typical secret name)
- Requires network access (yes/no)
- Typical use cases (1-3 phrases)

**Format:** JSON or structured markdown, loadable by the agent at session start.

---

### C2. Plugin quick reference with minimal examples

**Problem:** Schemas are necessary but not the fastest way to understand operational behavior.

**Requirement:** Add a quick reference per plugin covering:
- What it does (1 sentence)
- Most common options
- Common gotchas
- Minimal valid example configuration
- Typical output shape

---

### C3. Source semantics guide

**Problem:** Recurring confusion about source behavior — which sources infer columns, which require schema, how content maps to fields.

**Requirement:** Add a source behavior guide covering:
- **text source:** emits text column, may need explicit column name, schema recommended
- **json source:** row/object expectations, array vs. single-object handling
- **csv source:** header behavior, delimiter options
- **blob wiring:** when to use `set_source_from_blob` vs. `set_source`, inferred plugin behavior

---

### C4. Validation warning glossary

**Problem:** Warnings are system-specific and the agent must interpret them from scratch each time.

**Requirement:** Add a glossary mapping warning text to:
- Plain-English meaning
- Severity (low / medium / high)
- Likely cause
- Standard fix

Example entry:
- **Warning:** "Source has no explicit schema"
- **Meaning:** Downstream field references depend on runtime column names
- **Severity:** Medium
- **Fix:** Add explicit schema on source with expected field names

---

### C5. Common pipeline pattern library (shape catalog)

**Problem:** Most user requests collapse into a small set of workflow shapes, but the agent builds from primitives each time.

**Requirement:** Add a pattern library with 8-12 high-frequency shapes. For each shape:

1. **Name** (e.g., `url_to_structured_extraction`)
2. **Trigger phrases / intent clues** (e.g., "take this URL and extract...")
3. **Canonical pipeline structure** (source → transforms → sinks)
4. **Required user inputs** (what must be asked)
5. **Safe defaults** (output format, quarantine behavior, model choice)
6. **Known caveats** (e.g., LLM returns JSON string unless parsed)
7. **Clarifying questions** (only the minimum missing fields)

**Initial shape set:**
1. URL → scrape → LLM extract → JSON
2. Search → fetch → LLM extract → CSV
3. File/table → classify → route to sinks
4. File/docs → summarize → save
5. File/docs → structured extraction → JSON/CSV
6. Moderate → quarantine → continue
7. Batch LLM extraction over rows
8. Retrieval + answer generation
9. Transform chain with error diversion
10. Fork/join enrichment pipeline

---

### C6. Output-intent mapping

**Problem:** Users express output desires in business language ("save to Excel") that must be mapped to available sinks.

**Requirement:** Add explicit mapping guidance:
- "Excel" / "spreadsheet" → `csv` sink
- "JSON file" → `json` sink
- "database" / "table" → database sink
- "vector search" → `chroma` sink
- "report" → JSON or text depending on context

---

### C7. Secret/provider mapping hints

**Problem:** LLM node configuration requires knowing provider names, model ID formats, and expected secret names.

**Requirement:** Add a provider reference:
- Provider name → typical secret name → model ID format
- e.g., `openrouter` → `OPENROUTER_API_KEY` → `openrouter/model-name`

---

### C8. Execution shape documentation

**Problem:** Unclear what data shape reaches the sink after transforms — scalar string? parsed JSON? merged fields?

**Requirement:** Document per transform type:
- What the transform emits (scalar/string/object/list)
- Whether outputs are merged into the row or replace it
- Whether downstream sinks serialize rows or nested payloads
- How LLM response fields appear in the final row

---

## [D] Communication & Language Requirements

### D1. Business-friendly vocabulary as default

**Problem:** Pipeline jargon (schema, sink, transform, gate, quarantine) intimidates non-technical users.

**Requirement:** The composer agent (and skill pack guidance) shall use business-friendly terms by default:

| Internal Term | Business-Friendly Term |
|---------------|----------------------|
| source | input |
| sink | output / destination / saved file |
| schema | expected columns / expected fields |
| transform | processing step |
| gate | decision step / routing rule |
| pipeline | workflow |
| validation | setup check / pre-run check |
| quarantine | error file / problem records |
| field mapping | rename/reorganize columns |
| on_error | if something fails |
| blob | uploaded file / stored file |
| connection/edge | handoff between steps |

**Rule:** Use business terms by default. Only introduce technical terms when:
1. The user is demonstrably technical
2. The term is necessary to explain a problem
3. The user asks for implementation details

---

### D2. Two-layer response model

**Problem:** Responses are either too technical or lose useful detail.

**Requirement:** Agent responses shall use a two-layer structure:

**Primary:** Business-friendly explanation
> "I've set up a workflow that reads your URL, downloads the content, asks the model to extract the key facts, and saves the result as a JSON file."

**Optional detail** (on request or for technical users):
> "Internally: text source → web_scrape transform → llm transform → json sink with quarantine output."

---

### D3. Error/warning explanation in plain language

**Problem:** Raw validation messages are jargon-heavy.

**Requirement:** When reporting errors or warnings, the agent shall explain:
1. What it means in plain English
2. Whether it blocks running
3. What the fix is

**Bad:** "Source has no explicit schema. Downstream field references may fail."
**Good:** "The workflow expects an input column called `url`. I can add that explicitly so the workflow is more reliable. This won't stop it from running, but defining it makes things more predictable."

---

### D4. Minimal clarifying questions per workflow shape

**Problem:** Agent asks too many questions, including irrelevant ones.

**Requirement:** For each recognized workflow pattern, the agent shall ask only the minimum missing fields needed to proceed. Shape-specific question sets (from C5) determine what to ask.

For "fetch → extract → save":
- URL?
- What to extract?
- Output format? (default: JSON)

NOT: schema mode? quarantine policy? retry config? edge labels?

---

## [E] Cross-Cutting / Architectural Requirements

### E1. Inline/literal source plugin

**Problem:** Many pipelines need trivial seed data (one URL, one JSON object). Creating a file for this adds friction.

**Requirement (nice-to-have):** Add an `inline_json` or `literal` source plugin that accepts rows directly in configuration:

```yaml
source:
  plugin: inline_json
  rows:
    - url: "https://example.com/data.txt"
```

This eliminates the need for tiny helper files entirely.

**Status: Addressed** — The `create_blob` + `set_source_from_blob` two-step flow provides this capability using existing source plugins. See the "Inline data" section in the pipeline composer skill. A native inline plugin would duplicate schema validation logic with no benefit.

---

### E2. File creation visibility contract

**Problem:** Files created by the agent (blobs, outputs) are invisible to the user until they go looking.

**Requirement:** When the agent creates any file (via `create_blob`, pipeline output, etc.):
- The file shall appear immediately in the user's file panel
- The chat shall include a visible reference to the created file
- The user shall be able to click/tap to preview or download

---

### E3. Session-scoped vs. persistent file lifecycle

**Requirement:** Define and document file lifecycle:
- **Session-scoped blobs:** Created by agent during composition, may be temporary
- **Pipeline outputs:** Persist after execution, belong to the user
- **Uploaded files:** Persist until explicitly deleted

Users need to understand which files are ephemeral and which are permanent.

---

## Priority Recommendation

### P1 — Highest Impact
- **A1** Per-user file folders (visible file management)
- **A2** Markdown/Mermaid rendering in chat
- **A3** Visible error routing to agent in chat
- **B1** `create_blob` tool
- **C5** Common pipeline pattern library
- **D1** Business-friendly vocabulary

### P2 — High Impact
- **A4** Even panel split with persistence
- **A5** Secrets button redesign
- **A6** Inline validation on graph nodes
- **A7** Three-state pipeline status
- **B2** Consistent validation payloads
- **C1** Plugin capabilities registry
- **C4** Validation warning glossary
- **D3** Plain-language error explanations

### P3 — Medium Impact
- **B3** Source patching parity
- **B4** Blob ID abstraction
- **C2** Plugin quick reference
- **C3** Source semantics guide
- **C6** Output-intent mapping
- **C7** Secret/provider hints
- **C8** Execution shape docs
- **D2** Two-layer response model
- **D4** Minimal clarifying questions

### P4 — Nice-to-Have
- **B5** Pipeline diff/change summary
- **E1** Inline/literal source plugin
- **E2** File creation visibility contract
- **E3** File lifecycle documentation
