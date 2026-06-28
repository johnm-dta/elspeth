# ELSPETH Composer — High-Level UX Specification

**Status:** Draft for planning conversion
**Date:** 2026-05-14
**Audience:** UX planning, frontend engineering, product, audit/compliance review
**Supersedes:** ad-hoc per-phase guided-mode design notes (Phases 1B–10)
**Companion docs:** `docs/architecture/landscape.md`, `docs/architecture/token-lifecycle.md`, `docs/guides/telemetry.md`, `CLAUDE.md`

---

## 0. How to read this document

This specification is a *charter*: it states the design north star, the principles that constrain decisions, the canonical information architecture, the component catalogue with rendering rules, the interaction model, and the phasing recommended for execution. It is deliberately written so a planning agent can derive a milestone tree from it without further clarification.

Where the current shipped code matches the target, the spec says "keep". Where it diverges, the spec says "reframe", "extend", or "replace" and identifies the file as evidence. Where it is silent, the planner has latitude.

Section 9 ("Phasing") is the bridge from spec to plan.

---

## 1. North Star

> **A first-time user describes the problem they need to solve; an experienced user does the work themselves. The composer is the same surface for both. Every change is visible before it is committed, and every committed change is recorded in a form that will survive legal inquiry.**

Three commitments fall out of this:

1. **The composer is one surface, not two modes.** Guided-vs-freeform is a *density of scaffolding*, not a binary state of the session. A user fluently shifts density per decision.
2. **Pending and committed are visually distinct everywhere.** No change to the pipeline is ever surprising. Users see proposed changes before they exist; users see committed changes after they happen.
3. **The audit trail is a UX object, not just a backend artifact.** Decisions, proposals, rejections, approvals, validations, runs — these are the events the user navigates, references, and trusts. The audit trail must be browsable, queryable, and citable from the web.

---

## 2. Design principles

These principles win arguments. Where principles conflict, the ones higher on this list dominate.

1. **Auditability before convenience.** If a frictionless pattern would obscure or fabricate an audit fact, the friction stays. Silent autosave, optimistic-only UI, undo-as-rewind: forbidden. Undo as *reversing event*: encouraged.
2. **No silent state mutation.** Every change to the composition state — by human, by LLM, by recipe expansion — has a visible representation before commit and a visible record after commit. The "Pending → Committed → Audited" triad is the canonical lifecycle.
3. **Decisions need rationale.** Every proposed change carries one sentence of *why*. A user (especially a domain expert) cannot meaningfully consent to an option they cannot understand.
4. **The graph is the truth.** The DAG is the primary view of what the pipeline *is*. YAML, spec table, chat — all are projections. When they disagree, the graph wins.
5. **Trust modes are explicit.** A user can elect "auto-commit LLM tool calls" for fast iteration, but the choice is durable, visible, and recorded in the audit trail as a standing instruction. There is no implicit trust.
6. **Keyboard parity is non-negotiable.** Every action achievable by mouse is achievable by keyboard. The graph is keyboard-navigable. Pending changes are keyboard-approvable.
7. **First-render must orient.** When a user opens a session, they must know within 3 seconds: where they are in the flow, what the pipeline currently looks like, what would happen if they did nothing.
8. **Errors point at decisions, not lines.** Validation, runtime, and execution failures trace back to the *decision that introduced them*, not just a code location.
9. **Discoverability over efficiency for novices; efficiency over discoverability for experts.** Affordances should be visible to first-time users and bypassable by experienced ones (keyboard shortcuts, palette, command lines).
10. **No mode-switch is irreversible.** Any choice the user makes (scaffold density, trust mode, sidebar collapse) can be reversed within one click. The product never traps the user.

---

## 3. Personas

Four primary personas drive design tradeoffs. A design decision should be evaluated against at least two of them.

### 3.1 Avery — the domain expert
A compliance analyst at a council. Knows the data and the question; does not know the words *source*, *transform*, *sink*. Needs the composer to translate intent into a pipeline.
- **Mental model:** "I have data. I have a question. Give me an answer."
- **Success bar:** Can complete idea-to-result in one sitting, unaided.
- **Failure mode:** ProposeChain with no rationale → can't consent → abandons.
- **Anchors decisions on:** rationale, recipes, end-of-guided launchpad, resume summary.

### 3.2 Bri — the pipeline engineer
A senior engineer who has built many ELSPETH pipelines. Wants speed and precision. Knows what they want before they type it.
- **Mental model:** "Compose by composition. I direct, the LLM types."
- **Success bar:** Can build a 12-node pipeline in under 5 minutes, with every change visible.
- **Failure mode:** Silent tool calls → no diff visible → trust collapses → reverts to YAML editor.
- **Anchors decisions on:** tool call visibility, pending-change overlay, keyboard shortcuts, per-edit undo.

### 3.3 Chen — the auditor
Internal audit / external regulator. Doesn't build pipelines. Reviews them. Asks questions like: "show me the pipeline that produced this row's classification" or "who approved the addition of this LLM transform?"
- **Mental model:** "Show me the chain of custody for output Y."
- **Success bar:** Can navigate from a run output back to the originating decision in the conversation without leaving the web UI.
- **Failure mode:** Auditor IA does not exist on the web → forced to TUI → cannot operate at scale.
- **Anchors decisions on:** Ledger as first-class region, run-as-of-version view, audit ID surfacing, citation/export.

### 3.4 Diana — the returning user
Built a pipeline last week. Came back to add a sink. Forgot exactly where they left off.
- **Mental model:** "What was I doing, and what's next?"
- **Success bar:** Within 10 seconds of opening a session, knows current state and next action.
- **Failure mode:** Dropped into raw last-message context → has to reconstruct.
- **Anchors decisions on:** session resume summary, status strip, ledger timeline.

### 3.5 Secondary considerations

Not full personas but design constraints:
- **Esme, the educator** — wants the UI to teach. Folded into Avery (rationale + glossaries serve both).
- **Felix, the operator** — runs pipelines in production. Out of scope for the composer; routed to a separate dashboard surface.
- **Big-brother model** — when a cheap composer model is uncertain, it can escalate to an expensive model for advice. This is a persona-like interaction (the composer is briefly *another user*).

---

## 4. End-to-end journeys

Six canonical journeys cover the user-facing surface. Each is a *test case* for the design: if a journey breaks, the design has failed.

### J1 — First-time pipeline (Avery)
1. Lands on composer; clicks **New Session**.
2. Source elicitation turn (high scaffolding); fills SchemaForm; submits.
3. Recipe is offered; reads recipe summary + rationale; fills missing slot; clicks **Apply recipe**.
4. ProposeChain renders with per-node rationale; clicks **Accept all** (or **Edit step N**).
5. End-of-guided launchpad: three primary actions — **Validate**, **Run with sample**, **Run full**, **Download YAML**.
6. Clicks **Run full**; status strip and ledger show progress; pulse on completion.
7. Clicks the run in ledger; sees outputs; downloads CSV.
- **Demo SLA preserved:** ≤9 clicks, ≤3.5s perceived latency on happy path.

### J2 — Power composition (Bri)
1. New session; session-default scaffolding set to **low**.
2. Types "set source to csv with columns x, y, z"; LLM emits `set_source` tool call.
3. Tool call renders inline as a *pending* card with proposed argument table; Bri sees what the LLM understood; clicks **Accept**.
4. Continues conversationally; multiple tool calls land as pending cards; Bri batch-accepts with keyboard shortcut.
5. Asks "remove that last transform"; LLM emits `remove_node`; pending; Bri accepts.
6. Mistake: LLM proposes invalid transform; validation flag appears on the pending card *before* accept; Bri rejects.
7. Final pipeline; download YAML; execute outside composer.

### J3 — Audit walkthrough (Chen)
1. Lands on composer with a known run ID in hand.
2. Opens ledger search; filters by run ID.
3. Clicks the run entry → opens run-as-of-version view: graph + YAML as they were at execution time.
4. Clicks a token from the run output → token lineage explorer (web port of `elspeth explain`) → traces back through decisions.
5. Clicks a specific decision → opens the conversation thread at that moment, frozen.
6. Exports a citation bundle (audit IDs + composition version + conversation excerpt) for the report.

### J4 — Resume (Diana)
1. Opens composer; sidebar lists sessions with last-activity timestamps and titles.
2. Clicks the relevant session; resume summary card renders: last 3 decisions, current validation state, suggested next action.
3. Dismisses summary; lands in conversation at the most recent decision.
4. Picks up freeform: "add a JSON sink writing to /tmp/output.json"; tool call lands as pending; accepts; runs.

### J5 — Mid-session density change (Avery → Bri behaviour)
1. Avery is in guided flow but is comfortable with the current step.
2. Clicks **Switch to chat for this step** on the turn card.
3. Turn collapses to a chat ribbon: "I was going to ask you about X. Tell me what you want."
4. Avery types freely; LLM proposes; pending card; accepts.
5. Next turn returns to high scaffolding automatically (session-default unchanged).

### J6 — Recovery (any user)
1. User triggers a re-render after a backend crash.
2. Recovery diff panel mounts as a *blocking dialog*: shows current store vs server state, with per-element accept/reject.
3. User reviews; resolves conflicts; clicks **Apply**.
4. Conversation continues from the resolved state.

---

## 5. Information architecture

### 5.1 Top-level layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  STATUS STRIP — pipeline name · validation state · last decision     │
├────────────┬───────────────────────────────────────┬─────────────────┤
│            │                                       │                 │
│  SESSIONS  │            CONVERSATION               │     LEDGER      │
│            │  (chat + guided turns + tool calls)   │  (history of    │
│   ───      │                                       │   decisions,    │
│  CONTEXT   │            CANVAS                     │   validations,  │
│  (current  │  (graph · spec · YAML · data flow)    │   runs,         │
│   session  │                                       │   versions)     │
│   meta)    │                                       │                 │
│            │                                       │                 │
├────────────┴───────────────────────────────────────┴─────────────────┤
│  INPUT — chat field · attachments · scaffold density · trust mode    │
└──────────────────────────────────────────────────────────────────────┘
```

- **Sessions** (left): existing `SessionSidebar.tsx`. Add: last-activity timestamps, pipeline-state preview thumbnail on hover.
- **Conversation** (centre, primary): existing `ChatPanel.tsx`, reframed. See §6.
- **Canvas** (centre, secondary surface, view-switchable): existing inspector tabs reframed into a *single* canvas with view modes. See §7.
- **Ledger** (right): replaces existing `RunsView.tsx` as a tab; promoted to top-level region. See §8.
- **Status strip** (top): new persistent strip. See §5.4.
- **Input bar** (bottom): existing `ChatInput.tsx` reframed with mode chips. See §6.4.

### 5.2 Canvas view modes

The canvas has four interchangeable views of the same underlying state:

| View | Purpose | Key affordances |
|------|---------|-----------------|
| **Graph** | Spatial/structural understanding | DAG render, pending overlay, node click → inline detail, keyboard nav |
| **Spec** | Tabular/configuration understanding | Sortable rows of nodes with options, inline edit, pending rows |
| **YAML** | Textual/export understanding | Syntax-highlighted, diff-highlighted, copy, download |
| **Data flow** *(new)* | Behavioural understanding | Sample rows traced through transforms with intermediate results |

All four views render *pending* and *committed* state with the same colour semantics (§10).

### 5.3 Ledger panel

A chronological stream with filterable categories:

| Event class | Examples | Visual register |
|-------------|----------|-----------------|
| **Decisions** | Guided turn answered, tool call accepted, recipe applied | Conversation bubble icon |
| **Proposals** | Tool call landed as pending, awaiting user action | Amber clock icon |
| **Validations** | Pipeline became valid / invalid | Green check / red cross |
| **Runs** | Run started, succeeded, failed | Spinner / check / cross |
| **Versions** | Composition state version bumped | Tag icon |
| **Audit anchors** | External attestations (signed audit checkpoints) | Lock icon |

Each entry is **clickable** and **citable** (right-click → copy audit ID). The ledger is the auditor's home base (Chen) and the returning user's "what happened while I was away" surface (Diana).

### 5.4 Status strip

A persistent ~32px strip across the top of the centre region. Always visible regardless of scroll.

Layout: `[ pipeline name | validation pill | last decision summary | session-default density chip | trust-mode chip ]`

- **Pipeline name**: editable inline.
- **Validation pill**: green "valid", amber "dirty/unvalidated", red "invalid". Click → opens validation explanation in canvas.
- **Last decision summary**: one line, "added LLM transform 'classifier' (12s ago)". Click → scrolls conversation to that decision.
- **Density chip**: shows current session-default (high / medium / low). Click → menu to change.
- **Trust mode chip**: shows current mode (explicit-approve / auto-commit). Click → confirm change + record in audit.

---

## 6. Conversation surface

The conversation is the *primary* surface. It carries decisions, chat, tool calls, approvals, and meta-events in one chronological thread.

### 6.1 Message types

| Type | Source | Rendering |
|------|--------|-----------|
| **User chat** | User keyboard input | Right-aligned bubble |
| **Assistant chat** | LLM free-form response | Left-aligned bubble, may include tool-call ribbons |
| **Guided turn** | Backend orchestration | Distinct card with header, accent stripe in step-type colour |
| **Tool call (read)** | LLM informational tool | Inline ribbon under message, small badge |
| **Tool call (write, pending)** | LLM mutating tool, awaiting decision | Inline pending card with action summary + Accept/Edit/Reject |
| **Tool call (write, committed)** | After user acceptance | Faded card with ✓ Applied badge and audit ID |
| **Tool call (write, rejected)** | After user rejection | Faded card with × Rejected badge |
| **System meta** | Validation pass/fail, version bump | Centred slim banner |
| **Big-brother escalation** | Cheap model escalating to expensive model | Special card: "I asked a stronger model — here's its take" |

### 6.2 Tool call rendering — the canonical pattern

A tool call has three lifecycle states: **pending**, **committed**, **rejected**. Read tools skip pending (they don't mutate state).

#### Pending card (write tools)

```
┌────────────────────────────────────────────────────────────┐
│ ⏳ Proposed: upsert_node                                    │
│ ─────────────────────────────────────────────────────────  │
│ Add transform "classify_severity"                          │
│ type: llm_classifier · model: claude-haiku-4-5             │
│ inputs: case_summary  ·  outputs: severity (str)           │
│                                                            │
│ Why: This recipe expects a severity classification before  │
│      routing to the SLA threshold gate.                    │
│                                                            │
│ Affects: graph (+1 node, +1 edge), validation (re-run)     │
│                                                            │
│ [ View in graph ]  [ View args ]                           │
│ ────────────────────────────────────────────────────────── │
│ [ Accept ✓ ]  [ Edit… ]  [ Reject × ]  [ Discuss 💬 ]      │
└────────────────────────────────────────────────────────────┘
```

Rules:
- **Action summary** — one sentence in plain language; never raw JSON in the primary view.
- **Rationale** — "Why:" line, one sentence. Sourced from recipe metadata, LLM-emitted reasoning, or "added by user request".
- **Side effects** — explicit list of what will change ("affects: graph, validation, X downstream").
- **Affordances** — Accept, Edit, Reject, Discuss. Accept and Reject are equipotent in visual weight (no dark patterns).
- **Edit** — opens an inline editor pre-filled with the proposed args; user adjusts; submits as a new pending state.
- **Discuss** — opens chat input with context primed ("About this proposal: …") so the user can ask clarifying questions without losing the pending state.

#### Committed card

After acceptance, the same card collapses to a thin band:

```
✓ Applied: added transform "classify_severity" · audit #4F8A · 24s ago
```

Click expands to full diff. Audit ID is copyable.

#### Rejected card

Dimmed, struck-through, but retained in thread for context:

```
× Rejected: proposed transform "classify_severity" · 1m ago · undo
```

The "undo" is genuinely a new proposal (same args, fresh pending) — the audit trail records both events.

#### Read-only tool ribbon

For tools like `list_recipes`, `get_pipeline_state`, `diff_pipeline`:

```
🔎 Looked up: 12 recipes matching "sla"
```

Click to expand the result inline. No accept/reject required.

### 6.3 Guided turn rendering

The existing guided turn widgets (`SchemaFormTurn`, `MultiSelectWithCustomTurn`, `InspectAndConfirmTurn`, `ProposeChainTurn`, `RecipeOfferTurn`, `SingleSelectTurn`) are **retained** but visually unified:

- Card with header: "Step N of M — <step name>".
- Accent stripe colour-coded to step type (source / transform / sink / review).
- Body: the widget itself (unchanged).
- Footer: primary action (e.g. **Apply recipe**, **Submit**), secondary actions (**Skip**, **Switch to chat for this step**, **Get help**).

The "Switch to chat for this step" affordance is new: it collapses the widget to a chat ribbon for this single turn. The session default scaffolding density doesn't change.

### 6.4 Input bar

```
┌──────────────────────────────────────────────────────────────────┐
│ [ density: ▼ ] [ trust: ▼ ] [ 📎 ] [ message…                  ] │
│                                                          [ Send ]│
└──────────────────────────────────────────────────────────────────┘
```

- **Density** — current session default (high / medium / low). Inline change is per-session.
- **Trust** — explicit-approve / auto-commit. Inline change creates an audit event.
- **Attachments** — files, blobs, API keys (existing `ChatInput.tsx` affordances).
- **Send** — submits. Enter sends; Shift-Enter newline.

### 6.5 Streaming and tool-call interleaving

When the LLM is generating with interleaved tool calls:

1. Assistant chat bubble begins streaming text.
2. A tool call is invoked; streaming pauses; tool-call card renders inline below the bubble in "executing" state (animated icon).
3. Tool call returns; card transitions to "pending" (write) or "committed" (read).
4. Assistant chat resumes streaming.

The user never sees a moment of opacity. Tool-call execution is visible *as it happens*.

---

## 7. Canvas surface

### 7.1 Graph view (primary)

Existing `GraphView.tsx` using React-Flow. Reframe to:

- **Pending overlay**: nodes/edges from un-accepted proposals render with dashed 2px outline + 60% fill opacity + small "pending #N" pill in top-right corner. Click pill → opens the originating conversation message + pending card.
- **Committed-recent overlay**: nodes accepted in the last 24h render with a subtle "recent" outline. Helps Diana orient on resume.
- **Validation overlay**: invalid nodes outlined in `--color-error` with a small ⚠ in the corner. Click → validation explanation popover.
- **Selection inspector**: clicking a node opens an inline slide-out *on the canvas itself* with read-only details. Edit takes the user to the Spec view at that node.
- **Keyboard navigation**: Tab focuses the graph; arrow keys traverse adjacent nodes following edges; Enter opens slide-out; Space accepts pending change; Delete proposes removal (creates pending change).
- **Node accessible names**: every node has `aria-label="<type> node '<name>', <input-summary>, <output-summary>"`.

### 7.2 Spec view

Existing tabular/spec UI. Add:

- **Pending rows**: amber left border, "pending" pill, struck-through old value if changing.
- **Inline rationale**: hover row → tooltip with the rationale recorded at proposal time.
- **Bulk operations**: select multiple nodes, propose batch removal.

### 7.3 YAML view

Existing `YamlView.tsx`. Add:

- **Diff highlighting** when there are pending changes: added lines green, changed lines amber, removed lines red strikethrough.
- **Citation export**: a button to export the YAML *with* a citation footer (composition version, audit anchor, signing chain).

### 7.4 Data flow view *(new)*

A per-node visualisation of sample data passing through transforms.

- Pick a sample row (or a small batch) from the source.
- Visualise its transformation step-by-step: input → transform → output, with the row's values shown at each stage.
- Click a stage to see the audit record for that transform's execution against that row (when applicable).
- Useful for Avery ("does this actually do what I want?") and for Chen ("show me how this row became that").

This view depends on the existing `preview_pipeline` MCP tool and the `ChaosLLM`/`ChaosWeb` fixtures for safe deterministic preview.

### 7.5 View switching

Top of canvas region: tab strip with **Graph · Spec · YAML · Data flow**. Alt-1..4 keyboard shortcuts. Selected view persists per session.

Pending overlay state is *shared* across views — accepting a proposal in any view commits it everywhere.

---

## 8. Ledger surface

### 8.1 Layout

```
┌───────────────────────────────────────────┐
│  LEDGER                                   │
│  ───────────────────────────────────────  │
│  [ All ▼ ] [ Filter… ] [ Export… ]        │
│  ───────────────────────────────────────  │
│  ✓ Run completed · 12 rows · 2s ago       │
│    └ output: cases_sla.csv (download)     │
│                                           │
│  ↺ Pipeline validated · valid · 15s ago   │
│                                           │
│  ✓ Decision: accepted upsert_node         │
│    · transform 'classify_severity'        │
│    · 30s ago · audit #4F8A                │
│                                           │
│  ⏳ Pending: 1 unresolved proposal         │
│    · click to review                      │
│                                           │
│  …                                        │
└───────────────────────────────────────────┘
```

### 8.2 Filtering

- **By class**: decisions, proposals, validations, runs, versions, audit anchors.
- **By scope**: current session, current pipeline version, current run.
- **By search**: free-text over event summaries and audit IDs.

### 8.3 Run-as-of-version view

The killer auditor feature. Click a historical run entry → ledger spawns a side-by-side view: the conversation thread frozen at run time, the graph as it was, the YAML as it was. Read-only. Exports as a citation bundle.

This requires the composition state version associated with each run (which the system already tracks). It surfaces what is already true; it does not require new backend.

### 8.4 Token lineage explorer

Click a row in a run's output → token lineage popover. Shows the row's `row_id`, `token_id`, the sequence of nodes it passed through, the audit records for each. Web port of `elspeth explain`.

### 8.5 Comparison

Right-click two ledger entries (two versions, two runs) → "Compare". Opens a side-by-side diff in the canvas region. Useful for change reviews.

### 8.6 Export

Export selected events or filters as:
- **JSON** (machine-readable audit bundle)
- **Markdown** (human-readable report with embedded YAML and links)
- **PDF** (audit-presentable, includes signatures and anchors)

---

## 9. Interaction patterns

### 9.1 The Pending-Committed-Audited triad

Every state-changing user or LLM action follows the same lifecycle:

```
PROPOSED ──[user accept]──► COMMITTED ──[automatic]──► AUDITED
   │                            │                         │
   │                            │                         │
   └─[user reject]─► REJECTED   └─[user reverse]─► REVERSED (as new commit)
```

Rules:
- **Proposed** → audit event `proposal.created` (tool call started, args known).
- **Accepted** → audit event `proposal.accepted` + state mutation.
- **Rejected** → audit event `proposal.rejected`.
- **Reversed** → a *new* proposal that undoes the previous; goes through the same lifecycle.

Never modify or delete prior audit events. Reversal is always forward.

### 9.2 Trust modes

| Mode | Behaviour | Use case |
|------|-----------|----------|
| **Explicit approve** (default) | Every write tool call lands as pending; user accepts/rejects | Avery, Chen review, novice Bri |
| **Auto-commit** | Write tool calls auto-accept; pending card flashes briefly, then commits | Bri in flow, automated test runs |

Mode change is itself an audit event ("user set trust mode to auto-commit at <timestamp>"). The standing instruction is recorded so future-Chen can see who chose what.

### 9.3 Density modes

| Density | Behaviour | Use case |
|---------|-----------|----------|
| **High** (default for first session) | Guided turns where backend offers them; ProposeChain for chain construction | Avery, learners |
| **Medium** | Guided turns collapse to chat ribbons with proposal cards; recipes still offered | Diana returning, occasional Bri |
| **Low** | All decisions arrive as freeform tool-call proposals; recipes still surfaced but as chat suggestions | Bri in flow |

Per-turn override: any turn (high or low) can be temporarily switched via the **Switch to chat / Switch to widget** affordance.

### 9.4 Keyboard model

| Key | Action |
|-----|--------|
| `Ctrl+K` | Command palette (existing) — promote to visible affordance |
| `Ctrl+Shift+P` | Plugin catalog (existing) — promote to visible affordance |
| `Alt+1..4` | Canvas view (existing) |
| `Tab` | Focus moves through major regions in order: Sessions → Conversation → Input → Canvas → Ledger |
| `A` (with pending visible) | Accept currently focused proposal |
| `R` (with pending visible) | Reject currently focused proposal |
| `E` (with pending visible) | Edit currently focused proposal |
| `J` / `K` | Move down/up in conversation or ledger |
| `?` | Show keyboard help overlay |
| `Esc` | Close popover / cancel pending edit |

Keys are remappable in user preferences.

### 9.5 Discoverability

- Command palette is reachable by keyboard *and* by a visible button in the status strip (icon + tooltip "Search commands (Ctrl+K)").
- Plugin catalog and recipe gallery are reachable from a visible **Explore** menu in the sidebar.
- First-time users see a one-shot affordance overlay highlighting the four regions and the density chip.

---

## 10. Visual language

### 10.1 Semantic colour tokens

| Token | Hue | Purpose |
|-------|-----|---------|
| `--color-pending` | amber/orange | Proposed but not committed |
| `--color-committed-recent` | subtle blue | Committed in last 24h |
| `--color-valid` | green | Pipeline is valid |
| `--color-invalid` | red | Pipeline is invalid |
| `--color-error` | red | Runtime / execution error |
| `--color-warning` | yellow | Non-blocking concern (e.g., LLM cost high) |
| `--color-audit` | indigo | Audit-related affordances |
| `--node-source` | blue | Source node (existing) |
| `--node-transform` | green | Transform node (existing) |
| `--node-gate` | orange | Gate node (existing) |
| `--node-aggregation` | purple | Aggregation node (existing) |
| `--node-sink` | cyan | Sink node (existing) |

Existing node-type tokens are preserved. New tokens are *additive*.

### 10.2 Contrast targets

- All text: WCAG AA (4.5:1) minimum; AAA (7:1) on critical paths (validation pills, error messages, accept/reject buttons).
- UI components: 3:1 minimum against background.
- Pending overlay: distinguishable from committed by *both* outline style (dashed vs solid) *and* opacity, not by hue alone.

### 10.3 Typography

- Body: 16px (mobile) / 14px (desktop); line height 1.5.
- Chat / conversation: 16px desktop *and* mobile (it's the primary reading surface).
- Monospace (YAML, audit IDs): equivalent x-height to body.
- Headings: H1 24px, H2 20px, H3 18px.

### 10.4 Motion

- Pending → committed: 800ms green pulse on the affected canvas element.
- Streaming chat: cursor-style indicator at the streaming position.
- Run-in-progress: subtle pulse on the run's ledger entry.
- Respect `prefers-reduced-motion`: pulses become static colour flashes; streaming cursors become static.

### 10.5 Iconography

- Status icons accompany colour everywhere (no colour-only signalling).
- Audit-related icons use the indigo audit token + lock motif.
- Pending uses ⏳ or amber clock; committed uses ✓; rejected uses ×.

---

## 11. Accessibility commitments

### 11.1 WCAG conformance targets

- **AA across the board** at first ship.
- **AAA on critical paths** within the first major polish cycle: pending/committed contrast, validation messaging, run status.

### 11.2 Per-dimension commitments

| Dimension | Commitment |
|-----------|------------|
| **Visual** | Status icons on every colour-coded element; tested under colourblind simulation. |
| **Motor** | Keyboard parity for all functions; 44×44px touch targets; no gestures without keyboard equivalents. |
| **Cognitive** | Rationale on every proposal; plain-language summaries on every tool call; glossary for technical terms (source/transform/sink/gate). |
| **Screen reader** | Landmarks on every region; accessible names on all nodes; live-region updates for streaming chat, run progress, validation outcomes. |
| **Temporal** | No timeouts; user-paced streaming; pausable run progress narration. |
| **Situational** | One-handed mobile layout (stacked regions); single-column collapse below 600px; offline-tolerant for view modes (read-only). |

### 11.3 Testable acceptance criteria

- Keyboard-only end-to-end (J1 happy path) completes in ≤120 seconds.
- NVDA + VoiceOver smoke tests pass for chat thread, guided turns, ledger.
- Colourblind simulations (protanopia, deuteranopia, tritanopia) confirm node types and validation states are distinguishable.
- All chat thread elements are announceable in order.
- Zoom to 200% produces no horizontal scroll in primary regions.

---

## 12. Currently-latent functionality to surface

Each of these exists in the backend or is shipped but undiscoverable. The spec promotes them to first-class affordances.

| Feature | Current state | Spec disposition |
|---------|---------------|------------------|
| **Tool calls** (MCP composer tools the LLM invokes) | Silent — invisible to user | First-class conversation message type with full lifecycle (§6.2). |
| **`explain_validation_error`** (MCP tool) | Reachable only through LLM | Invoked directly when user clicks an invalid-node ⚠. |
| **`preview_pipeline`** (MCP tool) | Used by backend only | Drives Data Flow view (§7.4) and the "Run with sample" launchpad button. |
| **`diff_pipeline`** (MCP tool) | Used by backend only | Drives Compare view (§8.5) for any two versions. |
| **`get_audit_info`** (MCP tool) | Used by backend only | Drives audit ID popovers, citation export, token lineage explorer (§8.4). |
| **`list_recipes`** / recipe metadata | Only surfaced via offer | Recipe gallery in sidebar Explore menu, browsable before session start. |
| **`list_sources` / `list_transforms` / `list_sinks` / `list_models`** | Backend-only | Plugin catalog (§9.5) made visible; promotes to a sidebar affordance. |
| **Command palette** (Ctrl+K) | Keyboard-only | Visible button in status strip (§5.4). |
| **Plugin catalog** (Ctrl+Shift+P) | Keyboard-only | Surfaced through Explore menu. |
| **Version selector** | Buried in inspector | Surface as the ledger's "Versions" filter (§5.3) with one-click rollback that creates a new commit. |
| **Recovery diff** | Crash-only | Available on-demand for any session: "show server state vs my state" command. |
| **Fanout confirmation** | One-off `ConfirmDialog` | Generalised into the canonical "before you do this" pattern for any high-impact action (run-against-real-data, delete-node, batch-reject). |
| **Big-brother MCP escalation** (cheap → expensive model) | Described in roadmap; unbuilt | First-class card type in conversation (§6.1). User can manually invoke "ask the stronger model". |
| **ChaosLLM / ChaosWeb fixtures** | Test-only | Drive the "Run with sample" / preview affordances for safe demo runs. |
| **Per-step chat** (Phase A) | Shipped but undiscoverable | Becomes the natural "Discuss" affordance on guided turn cards. |
| **Token lineage** (`elspeth explain` TUI) | TUI-only | Web port in ledger (§8.4). |
| **Audit anchors / signatures** | Backend-only | Visible in ledger as audit-anchor events; citation export includes them. |
| **API keys / blob manager** | Buttons in input | Promote to status strip dropdown; usage state visible. |
| **MCP server (`elspeth-mcp`)** | Separate CLI | Surface as a "Diagnose" affordance on failed runs that links into the same data. |

---

## 13. Mobile & responsive

The composer is desktop-first, but the spec defines fallback behaviours:

| Breakpoint | Layout |
|------------|--------|
| ≥1200px | Full four-column layout (Sessions + Conversation + Canvas + Ledger). |
| 900–1200px | Three columns; Canvas and Ledger share a tabbed region. |
| 600–900px | Two columns; Sessions becomes drawer; Canvas/Ledger tabbed. |
| <600px | Single column; bottom tab bar switches between Conversation, Canvas, Ledger; Sessions in slide-out drawer. |

Mobile is not a primary target but must not be broken. Critical actions (accept/reject pending, view run output) must work one-handed.

---

## 14. Telemetry for UX (operational, not analytics)

Per CLAUDE.md, telemetry is operational visibility, ephemeral. We measure UX outcomes to know if the design is working, not to track users.

| Metric | Purpose | Target |
|--------|---------|--------|
| Time-to-first-run (TTFR) | Avery's success bar | Median ≤5 min on J1 |
| Decision-acceptance rate | Trust health | 60–85%; outside this range suggests misalignment |
| Mode/density switches per session | Density model success | Trending down → users settled; per-turn switches present |
| Recovery panel invocations | State integrity | Near zero in steady state |
| Validation-fail-after-accept rate | LLM proposal quality | <10% |
| Click distance to common actions | Discoverability | ≤2 clicks for the canonical 6 actions (validate, execute, download YAML, view runs, switch density, view ledger) |
| Accessibility audit pass rate | Inclusive design | 100% on automated checks; manual screen-reader pass each release |

All metrics are *audit-trail-derived* where possible (count events, not synthetic UI clicks), preserving the audit-first principle.

---

## 15. Open questions / explicit non-goals

### 15.1 Open questions for the planner

1. **Collaboration model.** Do two users editing the same session need real-time merge (CRDT, OT) or pessimistic lock (session-checkout)? The spec assumes the latter for v1 but flags as a known constraint.
2. **Mobile depth.** Is the composer a serious mobile target or a desktop-only tool with mobile-survivable fallback? Affects scope of touch-target work.
3. **Auditor authentication.** Chen's persona implies a read-only role distinct from author roles. Does the existing auth model support this, or does it need extension?
4. **Internationalisation.** Is i18n on the v1 horizon? Affects copy choices for rationale, prompts, glossary terms.
5. **Embedded vs standalone.** Will the composer ever embed in a host app (e.g., a council case-management UI)? Affects responsive contracts.

### 15.2 Explicit non-goals

- **Real-time multi-cursor co-editing.** Out of scope for v1. Two users in the same session use pessimistic locking with clear conflict resolution.
- **Pipeline marketplace / sharing.** No public recipe sharing in v1.
- **In-composer scheduling.** Operator/scheduling features belong on the separate dashboard surface (Felix's domain), not the composer.
- **WYSIWYG drag-build of pipelines.** The graph is read-with-pending-overlay; users compose by deciding on proposals, not by dragging connectors. (This is an *intentional* constraint: drag-build defeats the audit-rationale model.)
- **Direct YAML editing in the composer.** YAML view is read-only export. To change YAML, users edit through the conversation. (Power users may still hand-write YAML *outside* the composer and import.)

---

## 16. Phasing for executable planning

A planning agent should be able to take this spec and emit phases approximately as follows. Each phase is *bounded* (ships independent value) and *ordered* (later phases depend on earlier).

### Phase 0 — Audit-event foundations *(no user-visible change)*
- Extend audit schema with `proposal.created` / `proposal.accepted` / `proposal.rejected` / `trust_mode.changed` event types.
- Add pending-change state to the composition state store (server + client).
- Add `trust_mode` and `density_default` to session preferences.

### Phase 1 — Status strip and ledger reframe
- Persistent status strip across centre column.
- Replace `RunsView.tsx` with `LedgerView.tsx` covering decisions, proposals, validations, runs, versions.
- Wire validation state and last decision to status strip.
- Resume summary card on session re-entry.

### Phase 2 — Pending-change overlay
- Graph view: dashed-outline + opacity for pending nodes/edges; "pending #N" pill.
- Spec view: pending rows with amber border.
- YAML view: diff highlighting.
- Chat: pending tool-call cards with Accept/Edit/Reject/Discuss.

### Phase 3 — Tool-call visibility
- Inline tool-call rendering under assistant messages.
- Read-tool ribbon vs write-tool card distinction.
- "Affects" side-effect summary on every write proposal.
- Audit ID surfacing and copy.

### Phase 4 — Per-turn scaffolding (density model)
- Replace one-way `ExitToFreeformButton` with per-turn density override.
- Session-default density preference.
- "Switch to chat for this step" affordance on guided turns.
- "Open as guided" promotion from freeform proposals where applicable.

### Phase 5 — Rationale and resume polish
- ProposeChainTurn and RecipeOfferTurn carry per-item rationale.
- RecipeOfferTurn alternatives become clickable.
- End-of-guided launchpad with Validate / Run / Download primary actions.

### Phase 6 — Audit / review surface
- Run-as-of-version view (clickable run → side-by-side frozen state).
- Token lineage explorer (web port of `elspeth explain`).
- Compare view (any two versions).
- Citation export (JSON / Markdown / PDF).

### Phase 7 — Discoverability
- Visible command-palette button.
- Sidebar Explore menu with Recipes, Plugins, Models.
- One-shot onboarding overlay for first-time users.
- Big-brother escalation card type.

### Phase 8 — Data flow view
- Per-row trace through transforms.
- Sample-row picker.
- Integration with `preview_pipeline` and Chaos fixtures.

### Phase 9 — Accessibility and polish
- Keyboard graph navigation.
- Screen-reader passes on all regions.
- Status icons on all colour-coded badges.
- Focus rings audit.
- Mobile responsive depth.

### Phase 10 — Operational telemetry
- Wire UX metrics into the existing telemetry channel (per CLAUDE.md primacy: audit first, telemetry second, log last).
- Dashboard for TTFR, decision-acceptance rate, density-switches.

---

## 17. Acceptance criteria (spec-level)

The spec is "delivered" when these conditions hold:

1. Personas Avery, Bri, Chen, Diana each complete their canonical journey (J1–J4) unaided.
2. Every state mutation has a visible Pending → Committed → Audited representation.
3. Every tool call invocation appears in the conversation thread.
4. Every committed change has a clickable audit ID.
5. Every ledger entry is citable and exportable.
6. Keyboard-only completion of J1 within 120 seconds.
7. WCAG AA across the board; AAA on validation, accept/reject, and run status.
8. Resume from cold start: orientation within 10 seconds.
9. Mode switch (density / trust) is per-session, reversible, and audit-logged.
10. Recovery diff is available on demand, not only after crash.

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **Composition state** | The current canonical pipeline definition the composer is editing. |
| **Pending change** | A proposed mutation to composition state, not yet accepted. |
| **Committed change** | An accepted mutation, written to composition state, recorded in audit. |
| **Audit event** | An immutable record in the Landscape DB capturing a decision, proposal, validation, or run. |
| **Density** | The scaffolding level of decisions (high = widget, low = chat). |
| **Trust mode** | Whether write tool calls require explicit acceptance (explicit-approve) or auto-commit. |
| **Tool call** | An MCP composer-tool invocation by the LLM (read or write). |
| **Recipe** | A named pre-defined pipeline template applied via slots. |
| **Slot** | A configurable parameter on a recipe. |
| **Ledger** | The chronological audit-aware history surface. |
| **Canvas** | The spatial/structural view region (graph, spec, YAML, data flow). |
| **Conversation** | The chronological dialogue surface (chat, guided turns, tool calls). |
| **Status strip** | The persistent top bar showing pipeline name, validation, last decision. |

---

## 19. Cross-references to existing artifacts

| Existing file or doc | Relevance |
|----------------------|-----------|
| `src/elspeth/web/frontend/src/App.tsx` | Top-level layout; will host the four-region reframe. |
| `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Conversation surface; reframe target. |
| `src/elspeth/web/frontend/src/components/chat/guided/*.tsx` | Guided turn widgets; retained, visually unified. |
| `src/elspeth/web/frontend/src/components/inspector/GraphView.tsx` | Graph view; receives pending overlay. |
| `src/elspeth/web/frontend/src/components/inspector/YamlView.tsx` | YAML view; receives diff highlighting. |
| `src/elspeth/web/frontend/src/components/inspector/RunsView.tsx` | Replaced by `LedgerView.tsx`. |
| `src/elspeth/web/frontend/src/components/recovery/RecoveryPanel.tsx` | Available on-demand, not only on crash. |
| `src/elspeth/mcp/*` | Composer MCP tools — informs tool-call rendering. |
| `docs/architecture/landscape.md` | Audit primacy; informs Ledger and citation export. |
| `docs/architecture/token-lifecycle.md` | Token lineage explorer. |
| `docs/guides/telemetry.md` | UX telemetry channel. |
| `CLAUDE.md` § "Auditability Standard", § "Three-Tier Trust Model" | Foundational principles. |

---

## 20. Closing

This spec is intentionally opinionated. Where it makes choices (per-turn scaffolding, pending-overlay-everywhere, ledger-as-region, tool-calls-as-first-class-messages), those choices are testable against the personas and journeys. If a journey breaks, revisit the choice.

The single most consequential decision in this spec is **Section 6.2 — Tool call rendering**. It dissolves three failures at once (Bri's audit-blind-spot, Avery's silent-state-confusion, Chen's chain-of-custody opacity) with one pattern. A planner short on time who only ships one phase should ship Phase 2 + Phase 3 (overlay + tool-call visibility). Everything else amplifies that core change.
