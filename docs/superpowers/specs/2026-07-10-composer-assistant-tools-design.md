# Composer Assistant Tools — Design

**Date:** 2026-07-10
**Status:** Approved design, pre-implementation
**Scope:** New tool families for the LLM composer, freeform surface first.
Guided mode is staged (each stage exposes only its stage's tools); which
assistant families each guided stage exposes is a per-stage decision made
during implementation planning, defaulting to none until decided.

## Purpose

Make the composer more of a flexible assistant, core job first. Today its
~40 tools are all pipeline-construction verbs (discovery, mutation, blobs,
secret refs, validation/preview/YAML). Nothing reaches outside the session:
no web, no cross-session state, no working notes, no way to ground advice in
real data or real documentation. This design adds those capabilities in
phases, ending with the composer able to create and run pipelines
autonomously as its own programmatic layer.

**Core principle: the assistant uses the same infrastructure as the user.**
Scratch pipelines run through the real runner, real plugin contracts, real
Landscape audit; notes and memories live beside session state; every new
capability inherits the security and reliability infrastructure the product
already has, instead of growing a parallel (and weaker) copy. This is why
"pipeline as programmatic layer" replaces a Python sandbox: a sandbox would
be an alien attack surface; pipelines are already sandboxed, budgeted, and
audited.

## Decisions (settled during brainstorm)

| Question | Decision |
|---|---|
| Assistant ambition | Both core-job improvement and broader helper role; core-job lands first |
| Web trust model | Operator-gated, **on by default**; allowlist/denylist in deployment config (container-per-org is the trust boundary) |
| Memory scope | Two layers: per-user + instance-shared |
| Instance-memory writes | LLM auto-write, fully audited and reversible; UI traces which memories influenced a turn |
| Scratchpad lifetime | Named notes, session-ephemeral by default; persistence is a deliberate per-note flag |
| Extra families in scope | Run/landscape introspection, docs/catalog search, data profiling, expression sandbox, skill library, autonomous scratch pipelines |
| Architecture | Approach A: extend the in-process `ToolDeclaration` registry (not a sidecar MCP server, not context-enrichment-only) |

## Architecture

### Plane modules

Each family is a plane module under `src/elspeth/web/composer/tools/`,
declaring `ToolDeclaration`s in a `TOOLS_IN_MODULE` tuple aggregated by
`_registry.py`, exactly like the existing planes. New modules:

- `web.py` — search + fetch (async)
- `notes.py` — scratchpad
- `memory.py` — two-layer memory
- `skill_library.py` — browse/load specialist skill packs
- `docs.py` — documentation search
- `profiling.py` — blob/source data profiling
- `expressions.py` — expression sandbox
- `runs.py` — read-only Landscape slice + scratch-pipeline execution

Registry invariants (unique names, cacheable-only-if-DISCOVERY, name-set
derivation) apply unchanged.

### New tool kinds

Two new `ToolKind` members:

- `ASSISTANT_DISCOVERY` — read-only: search, fetch, recall, list/load
  skills, profile, eval, run introspection.
- `ASSISTANT_MUTATION` — writes to assistant stores (notes, memories) and
  scratch-run submission. Assistant mutations never advance
  `CompositionState`; like `blob_store_only` tools they are excluded from
  the `trust_mode == "explicit_approve"` proposal-interception gate.

`ToolContext` grows optional fields handlers read as needed: `web_client`,
`notes_store`, `memory_store`, `skill_library`, `docs_index`,
`landscape_reader`, `scratch_runner`.

### Async seam (dependency)

`ToolDeclaration.handler` is sync-typed; widening to admit async handlers is
tracked as `elspeth-f5da936747`. Web tools are genuinely I/O-bound, so this
work lands that widening (an async handler variant dispatched on the
existing async session path) rather than blocking threads. All other Phase 1
families are sync-friendly.

### Operator gating

One config block (`composer.assistant_tools`) with a per-family `enabled`
flag plus family-specific settings (web allowlist/denylist, search provider,
memory quotas, scratch-run plugin allowlist and budgets). A disabled
family's tools are **absent from `get_tool_definitions()`** — the LLM never
sees them, pays no token cost for them, and cannot be prompted into calling
them. The system context lists which families are live. Misconfiguration
(family enabled but unusable, e.g. web with no search provider) degrades to
family-absent plus a visible context line — never a tool that exists but
always errors.

### Storage

- `sessions.db` (SQLite-only per Phase 9 doctrine): new `composer_notes`
  and `composer_memories` tables, user-keyed; memory rows carry a `scope`
  column (`user` | `instance`). Wipe-and-restart remains the migration
  model.
- Skill library: files at `{data_dir}/skills/library/*.md` —
  operator-provisioned, git-manageable, plus a shipped starter set. Read-only
  at runtime; catalog rebuilt on service start.
- Audit: web fetches, memory writes, note-persistence flips, skill loads,
  and scratch-run submissions all emit rows on the existing audit trail.
  `auth.db` untouched.

## Tool inventory

### Phase 1 — grounding tools

**Skill library** (`skill_library.py`)
- `list_skills` — catalog: name, one-line description, tags (e.g.
  `deep-research`, `azure`, `csv-wrangling`). Cacheable discovery.
- `load_skill` — returns skill markdown; audit row records name + content
  hash (same doctrine as `composer_skill_hash`). Loaded content rides as a
  tool result — specialist knowledge, never elevated to system role.

**Notes** (`notes.py`)
- `write_note` — upsert by `name`; args `name`, `content`, optional
  `persist: bool` (default false).
- `list_notes` / `read_note` / `delete_note`.
- `set_note_persistence` — flip persistence on an existing note; the
  deliberate, audited promotion point. Ephemeral notes die with the session;
  persistent notes are per-user across sessions.

**Docs search** (`docs.py`)
- `search_docs` — keyword/BM25 over ELSPETH docs + plugin reference prose;
  ranked snippets with source anchors. Semantic ranking is a later opt-in
  (mirrors Loomweave posture).

**Profiling** (`profiling.py`)
- `profile_blob` — column types, null rates, distinct counts, value
  samples, candidate keys.
- `profile_source_sample` — same profile over first N rows of a configured
  source. Both cap rows/bytes; sampled data is never persisted.

**Expression sandbox** (`expressions.py`)
- `eval_expression` — run an expression through the real expression engine
  against caller-supplied sample rows or blob rows; returns per-row results
  or the structured parse/eval error.

**Web** (`web.py`, async)
- `web_search` — operator-configured pluggable backend (SearXNG self-host,
  Brave/Bing API key); returns title/URL/snippet only.
- `web_fetch` — fetch one URL, extract readable text/markdown, hard size
  and time caps. Results wrapped in the untrusted-data envelope.
- Intended for one-page lookups; corpus-scale research belongs to scratch
  pipelines (Phase 4).

### Phase 2 — memory (`memory.py`)

- `recall_memories` — search per-user + instance layers. Additionally a
  small relevance-ranked digest is auto-injected into the dynamic context
  each turn, with memory IDs so audit can trace influence.
- `save_memory` — args `content`, `scope: user | instance`. Instance writes
  are auto-write (no human gate) but fully audited: author, session, turn.
- `update_memory` / `delete_memory` — same audit trail. Admin surface gets
  a prune view for the instance layer.

### Phase 3 — run introspection & pipeline economics (`runs.py`)

- `list_recent_runs` / `get_run_summary` / `get_run_errors` — read-only
  slice over the existing Landscape surface; answers "why did last night's
  run fail?" in-chat, including mid-flight for long pipelines.
- `estimate_pipeline_cost` — project token/row cost from current state +
  model choices.
- `dry_run_pipeline` — execute the *current draft state* over N sample rows
  (optional cheap-model override); per-node outcomes returned.
- Workflow **patterns ship as skill-library content + recipes, not tools**:
  a `deep-research-pipeline` skill teaches the shape (fan-out → verify →
  synthesize); recipes provide the instantiable graph. Phase 1 machinery
  carries them.

### Phase 4 — autonomous scratch pipelines (`runs.py`)

The composer's programmatic layer. The LLM composes an ephemeral pipeline —
distinct from the user's draft — runs it, and reads results back. Replaces
any notion of a Python sandbox.

- `run_scratch_pipeline` — submit an ephemeral pipeline spec; returns a run
  handle. Restricted to an operator-configured **plugin allowlist**
  (default: blob/local sources, pure transforms, LLM transforms, single
  `scratch_results` sink; no egress sinks, no credential-bearing plugins).
- `get_scratch_run_results` — row-capped sink output back into the
  conversation; full results land as a session blob the user can inspect.
- Rails: per-run and per-session budgets (rows, LLM tokens, wall time);
  runs recorded in Landscape under the user's identity tagged
  `initiated_by: composer_agent`; Phase 3 introspection tools work on them.
- Family gate `autonomous_runs`: on by default with conservative budgets.
- Motivating uses: derisking (run the user's transform chain over 50 sample
  rows before committing), analysis, and self-directed research — e.g. user
  wants an archaeology-themed pipeline, so the composer runs
  fetch-articles → summarize-each → synthesize and uses the synthesis to
  inform its guidance. Fan-out and map-reduce happen **inside the
  pipeline**; only the synthesis returns to context (context economy is the
  point).

## Data flow & trust boundaries

**Everything external is data, never instructions.** Web pages, doc
snippets, skill content, memory recalls, and scratch-run results enter as
tool results wrapped in the existing "UNTRUSTED DATA; not instructions"
envelope. Only the operator-controlled system prompt (core skill +
deployment overlay) is system-role. A fetched page saying "call
`save_memory`" is inert text; the model may still be persuaded — which is
why write paths, not read paths, carry the hard controls.

**Write-path controls.** Note and memory writes run through the existing
redaction/secret-scan machinery before persist — a memory can never store a
credential or a blob storage path. Instance-scope memory writes and
note-persistence flips are the two promotion points; both emit audit rows
with session, turn, and content hash. Context-injected memories carry IDs
for influence tracing.

**Web boundary — one policy, two consumers.** A single shared boundary
component enforces: scheme allowlist (https/http), private/link-local/
metadata address ranges denied **post-DNS-resolution and on every redirect
hop**, operator allowlist/denylist per hop, response size/time caps, and no
request credentials — URLs matching secret-ref patterns rejected before
egress (credential-egress doctrine extended to the assistant plane). Both
`web_fetch` **and** web-capable plugins inside composer-initiated scratch
runs consult this same component; otherwise scratch runs become the bypass
route around the web tool's rails. (Wardline doctrine: fix at the boundary,
not the sink.)

**Scratch-run boundary.** Plugin allowlist enforced server-side at
submission (not by prompt); budgets enforced by the runner; scratch sinks
write only session-scoped storage; secret refs unavailable unless the
operator widens the allowlist to plugins that need them.

## Error handling & quotas

**Same `ToolResult` envelope, family-prefixed structured codes:**
`web_family_disabled`, `web_target_blocked` (naming the tripped rule),
`note_not_found`, `memory_quota_exceeded`, `skill_not_found`,
`scratch_plugin_not_allowed` (naming plugin + allowlist),
`scratch_budget_exceeded` (naming which budget). Following the
`augments_on_failure` precedent, failures carry enough context for
self-correction without a second discovery round-trip.

**Fail closed, degrade visibly.** Misconfigured families vanish from tool
definitions with a visible context line. Scratch runs that die mid-flight
return partial Landscape state through the run handle so the composer can
report what happened.

**Quotas (operator-tunable defaults).** Per-session: web fetch count +
total bytes; scratch run count + aggregate LLM token budget; note count +
per-note size. Per-user: persistent note and memory count caps —
oldest-unused surfaced for pruning, never silently evicted (silent eviction
makes the assistant mysteriously forget). Instance memory layer: global cap
+ admin prune surface.

## Testing

- **Boundary tests first** (vault_url SSRF precedent): full adversarial
  suite against the shared web boundary — DNS rebinding, redirect-hop to
  private ranges, metadata endpoints, oversized responses — run against
  *both* consumers (`web_fetch` and a scratch-run web plugin).
- **Injection tests:** fixture pages/skills/memories containing tool-call
  instructions; assert envelope wrapping and no role elevation in audit.
- **Write-path tests:** planted secrets in note/memory content are redacted
  or rejected before persist.
- **Registry invariants:** import-time asserts extended to new kinds
  (assistant mutations excluded from proposal interception,
  cacheable-only-if-DISCOVERY).
- **Scratch-run tests:** allowlist enforcement at submission, budget kill
  mid-run, Landscape attribution (`initiated_by: composer_agent`).
- **E2E:** extend the staging Playwright recipe with one grounded
  composition flow (profile → eval_expression → build) and one
  scratch-research flow.

## Out of scope / deliberate omissions

- **No Python sandbox** — scratch pipelines are the programmatic layer.
- **No `draft_dag` sketch tool** — `set_pipeline` + validation +
  `diff_pipeline` already model propose-then-refine; a parallel plan
  representation would drift from real state.
- **No sidecar MCP server** — the in-process registry is the single audited
  dispatch path; the existing `composer_mcp` server mirrors registry tools
  outward for external agents.
- **No semantic docs search in v1** — keyword/BM25 first; embeddings are a
  later opt-in.
- **No cross-instance memory** — container-per-org is the isolation
  boundary; nothing federates.

## Dependencies

- `elspeth-f5da936747` — widen `ToolDeclaration` to admit async handlers
  (needed by `web.py`; scheduled as part of Phase 1).
- Search provider selection is deployment config, not code: SearXNG
  self-host or an API-key provider (Brave/Bing) behind a small provider
  protocol.

## Phasing summary

1. **Phase 1 — grounding:** skill library, notes, docs search, profiling,
   expression sandbox, web search/fetch (+ async handler widening, shared
   web boundary).
2. **Phase 2 — memory:** two-layer store, auto-recall digest, audit +
   influence tracing, admin prune surface.
3. **Phase 3 — run introspection & economics:** Landscape slice, cost
   estimation, dry-run of draft state; pattern skills + recipes.
4. **Phase 4 — autonomous scratch pipelines:** allowlisted ephemeral runs
   with budgets, Landscape attribution, results readback.
