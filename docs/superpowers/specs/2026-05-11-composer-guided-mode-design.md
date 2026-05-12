# Composer Guided Mode — Design

**Ticket:** TBD (to be created in filigree after spec approval)
**Parent epic:** Likely under [elspeth-de91358c30](filigree:elspeth-de91358c30) (Cluster — RC5-UX frontend evaluation remediation) or sibling cluster epic; final placement decided when ticket is created.
**Date:** 2026-05-11
**Status:** Proposed (rev 1 — initial brainstorm output; not yet reviewed by plan-review panel)
**Branch:** RC5.1 (current), targeting merge once frontend RC5-UX P0/P1 work has stabilised.
**Tier-artifact match:** **M-tier** change delivered as a multi-phase plan. The feature spans backend protocol, server endpoints, frontend chat widgets, a new LLM skill, audit event types, and testing across three rings. Each major component (protocol module, endpoints, frontend widgets, skill, recipe pre-match) is independently reviewable and independently testable, but they ship together because partial guided mode is worse than no guided mode (a wizard that breaks halfway through is hostile UX).

**Path conventions.** Throughout this spec, `web/...` is shorthand for `src/elspeth/web/...`; `composer/...` for `src/elspeth/web/composer/...`; `frontend/...` for `src/elspeth/web/frontend/src/...`. Test paths are written full from repo root (`tests/...`). Existing files cited the first time use full paths; subsequent references in the same section may use shorthand.

**Brainstorm provenance.** This design synthesises a brainstorming session captured under the brainstorming skill on 2026-05-11. The visual companion was used; mockups for the four major decisions (guidance shape, step sequencing, turn taxonomy, mode lifecycle) live under `.superpowers/brainstorm/` and are referenced by filename in §13 for the record.

---

## 1. Goals and Non-Goals

### 1.1 Goal

Lower the barrier to building a first ELSPETH pipeline by replacing the current freeform "tell me what you want" composer entry point with a structured wizard that walks the user through three explicit steps:

1. **Source** — what is the data, what columns does it have
2. **Sink** — what is the final output, what fields must it contain
3. **Transforms** — how do we get from (1) to (2)

Both the human user and the LLM operate inside a closed turn-protocol contract: the user can only answer in the form the wizard offers (no freeform text input field), and the LLM can only emit one of six structured turn types per turn. Pipeline state mutation is **server-side only**, driven deterministically by user actions on chip widgets — the LLM is read-only with respect to pipeline state.

The goal is not to replace freeform mode. It is to provide an alternative entry path that is:
- **Predictable** — same input produces the same conversation
- **Auditable** — every turn and response is recorded as a structured audit event, not free prose
- **Cheap on tokens** — happy paths cost zero LLM calls (recipe pre-match); novel paths cost one (Step 3 chain proposal)
- **Safe by construction** — silent shape downgrades, audit fabrication, and state corruption are structurally impossible (the LLM cannot write to state)

### 1.2 Non-Goals

- **Replacing freeform mode.** Freeform composer remains the default for all re-entry to saved pipelines and is reachable mid-flow via an exit button. Freeform skill, freeform tool surface, freeform endpoints — all unchanged.
- **Freeform → guided switching.** One-way transition only (guided → freeform). See §11.
- **Re-entry to a saved pipeline in guided mode.** Re-entry is the "I know what I'm doing" case; guided is for novice authoring. See §11.
- **Multi-source pipelines.** Engine supports exactly one source per run; guided mirrors that. Multi-source is out of scope for ELSPETH itself, not just for guided mode.
- **Visual graph editing inside guided mode.** Steps render as chip widgets, not as a graph view. If a user needs graph manipulation, they exit to freeform.
- **Tutorial / annotated walkthrough about the wizard.** Guided mode *is* the wizard; a tutorial *about* it is separate scope.
- **New plugin types or new pipeline shapes.** Guided mode is a new construction surface for existing engine capabilities. It does not introduce new runtime concepts.

### 1.3 Success Criteria

- **Demo-path SLA.** A first-time user with a CSV blob in hand can produce a valid, executable pipeline in:
  - ≤2 LLM calls (zero on the recipe-match path; one if Step 3 is reached for a clarifying question)
  - ≤4 user clicks (Step 1 confirm + Step 2 declare-or-skip + Step 2.5 apply recipe + Termination confirm)
  - ≤30 seconds wall-clock under typical demo conditions
- **Audit-replayability.** For any guided session, replaying the recorded `guided_turn_emitted` / `guided_turn_answered` event sequence produces an identical final pipeline.
- **No silent failures.** Every error path either re-emits the current turn with a structured error payload, or auto-drops to freeform mode with a recorded drop reason. Auto-drop must always succeed.
- **Skill-prompt size reduction.** The new guided-mode skill is ≤80 lines (vs. ~250 for the existing freeform skill). The freeform skill is unchanged.

---

## 2. Background

### 2.1 Today's freeform composer

`composer/service.py` (3741 lines) implements a chat-style composer where the user types prose and the LLM has the full ~30-tool surface defined in `composer/tools.py` (5645 lines). The LLM can mutate pipeline state (`set_pipeline`, `upsert_node`, `upsert_edge`, `set_source`, etc.) directly. The skill prompt at `composer/skills/pipeline_composer.md` (~250 lines) governs the LLM's behaviour through a combination of:

- 10 numbered convergence guardrails (rule 0: always include `schema`; rule 4: declare numeric types before gates; rule 7: ask the output-shape question for fork shapes; rule 8: format-check sample values against downstream consumers; etc.)
- A Connection Model essay (named-string wiring, not node IDs)
- A Shape Catalog (intent → tool sequence)
- An Anti-Fabrication block (do not invent capabilities; do not propose audit sinks; do not silently downgrade shapes)
- A Termination Gate rule (do not end your turn while invalid)

The prompt is large because the LLM has full mutation power and the prompt is the **only** place where structural correctness rules can be asserted.

### 2.2 Why guided mode

The guardrails in the freeform skill exist because they fix observed failure modes. Each numbered rule corresponds to a class of past mistakes:

- Rule 0 fixes "LLM submits source without `schema` block, validation rejects."
- Rule 4 fixes "gate condition `row['price'] >= 100` against string field, runtime crash."
- Rule 7 fixes "LLM silently drops fork-output shape into a single sink."
- Rule 8 fixes "LLM passes bare hostnames to `web_scrape.url_field`, run-time `SSRFBlockedError`."

In every case, the rule is needed because **the LLM has the option to skip the relevant step.** Guided mode removes the option: the wizard's state machine *requires* schema declaration in Step 1, *requires* the user to make a sink-fanout decision in Step 2, *requires* numeric coercion to be inserted by the chain solver. Rules 0, 4, 7 become trivially satisfied; rule 8 reduces to a one-line "eyeball samples" prompt in Step 3.

The skill-prompt enforcement model is migrating from prose to code. This is the pattern `CLAUDE.md` insists on: "if it's not mechanically enforced, assume the next session won't know about it." Guided mode is the opportunity to move enforcement where it belongs.

### 2.3 What this design borrows from existing infrastructure

- **`recipes.py`** (682 lines) defines three slot-validated recipe specs and the `apply_recipe()` deterministic pipeline-builder. Guided mode reuses this directly for the Step 2.5 pre-match shortcut.
- **`tools.py` mutation handlers** (`_execute_set_source`, `_execute_upsert_node`, etc.) are reused unchanged. Guided mode drives them via deterministic step-handler code rather than via LLM tool calls.
- **`state.py` `CompositionState`** is extended with one optional field; the existing immutability discipline (`@dataclass(frozen=True, slots=True)` + `freeze_fields` per `CLAUDE.md`'s deep_freeze contract) applies to the new field.
- **`_litellm_acompletion` / `_attach_llm_calls`** in `service.py` is the only LLM-invocation path; guided mode's chain solver wraps this rather than introducing a parallel path. (Required for audit integrity.)
- **`BufferingRecorder`** in the Landscape audit plumbing is the only audit-write path; guided mode's new event types (§9.1) emit through it unchanged.
- **Frontend `ChatPanel.tsx`** scaffolding (focus trap, scroll-on-new-message, `role="log"` `aria-live` announcement, skip link) is reused; only the active-turn render area is replaced.

---

## 3. User-Visible Flow

The wizard runs three steps plus a deterministic recipe-match check between Steps 2 and 3, ending in termination.

### 3.1 Step 1 — Source

The user picks a source plugin. Today the registry has: `csv`, `json`, `text`, `azure_blob`, `dataverse`. (`null_source` is internal-only and not user-facing; the existing skill's anti-fabrication rule covers this.)

The Step 1 turn sequence is typically:
1. **`single_select`** — "What kind of data are you starting with?" Chips: CSV, JSON, Text, Azure Blob, Dataverse.
2. **`schema_form`** — auto-generated form from the chosen plugin's option schema (path, encoding, etc.). The form pre-fills `schema.mode: "observed"` (lowest-friction default per the existing skill's rule 0; user can switch to `fixed` or `flexible`).
3. **`inspect_and_confirm`** — only if a blob has been attached. The wizard runs `inspect_source(blob_id)` and shows the user the observed columns, sample values, and any warnings (duplicate headers, NULL-heavy columns, etc.). User confirms or edits the inferred schema.

The output of Step 1 is a `SourceResolved` value (frozen dataclass) holding `{plugin, options, observed_columns, sample_rows}`. The `composition_state.source` field is mutated server-side via `_execute_set_source` once Step 1 completes.

### 3.2 Step 2 — Sink + required fields

The user picks one or more sinks and declares (or defers) required output fields.

1. **`single_select`** — "What's the final output format?" Chips: JSONL, CSV, Database, Azure Blob, ChromaDB.
2. **`schema_form`** — sink-specific options (path, table name, etc.).
3. **`multi_select_with_custom`** — "What fields must appear in the output?" Pre-populated chips from Step 1's `observed_columns`. User can:
   - Tick fields to require them (the precise C-shape path)
   - Type custom field names that don't exist in the source yet (these become "must-be-produced-by-transforms")
   - Click **"Or: let source decide"** (the Refinement 1 escape) — sets the sink's `schema.mode: "observed"`, declares no required fields, and the chain solver works backward from "ship every row to the sink, transformed however the user wants"

Multi-sink (fan-out) is supported: the Step 2 widget can add additional sinks each with its own format and required-fields decision. Each becomes a separate output in `composition_state.outputs`.

The output of Step 2 is a `SinkResolved` value holding `{outputs: [{plugin, options, required_fields, schema_mode}, ...]}`.

### 3.3 Step 2.5 — Recipe pre-match (server-side, no UI when no match)

Before Step 3, the server runs a deterministic match against `recipes.py`'s registered recipes. The match function is a pure predicate:

```python
def match_recipe(
    source: SourceResolved,
    sink: SinkResolved,
) -> RecipeMatch | None:
    """Return the most-specific recipe matching the (source, sink) tuple, or None."""
```

If a match is found, the user gets a **`recipe_offer`** turn:
- "This shape matches the **classify-rows-llm-jsonl** recipe (CSV → LLM classification → JSONL)."
- Buttons: **Apply recipe** | **Build manually**

Apply-recipe calls `recipes.apply_recipe(name, slots)` (existing function), which deterministically builds the entire pipeline. Step 3 is skipped; the wizard advances to termination.

If no match is found, the wizard silently advances to Step 3 (no UI emitted).

### 3.4 Step 3 — Transforms (LLM-driven)

This is the only step where the LLM does substantive work. The server invokes the chain solver (§6.4) with a structured context block:

```
GUIDED CONTEXT (server-resolved):
source: {plugin: "csv", columns: [...], sample: [...]}
sink: {outputs: [{plugin: "json", required_fields: [...]}]}
recipe_match: null
```

The LLM emits a **`propose_chain`** turn containing the proposed transform sequence with per-step rationale. The user can:
- **Accept all** — server commits the chain via existing mutation handlers
- **Edit step N** — re-emits propose_chain with that step locked
- **Reject** — emits a clarifying `single_select` turn ("I'm trying to bridge X to Y; do you want approach A or approach B?") or escalates to advisor via `request_advisor_hint`

If the LLM cannot produce a valid chain after one repair attempt + advisor consultation, the wizard auto-drops to freeform mode (§5.4).

### 3.5 Termination

The server runs `preview_pipeline`. On `is_valid: true`, the wizard shows the user:
- The generated YAML (rendered server-side via the existing `yaml_generator.py`)
- A **"Looks good — save and exit"** button
- A **"Drop to freeform to keep editing"** button (for users who want to iterate before saving)

On `is_valid: false` after one repair attempt, auto-drop fires (§5.4).

---

## 4. Turn-Protocol Contract

### 4.1 Closed turn taxonomy

The LLM may emit exactly one of six turn types per turn. Each has a server-side payload schema. Anything else is rejected by the server.

| Turn type | Payload schema | Used in |
|---|---|---|
| `inspect_and_confirm` | `{observed: {columns: [str], samples: [Mapping], warnings: [str]}}` | Step 1 |
| `single_select` | `{question: str, options: [{id: str, label: str, hint: str?}], allow_custom: bool}` | Steps 1, 2, 3 |
| `multi_select_with_custom` | `{question: str, options: [...], default_chosen: [str], escape_label: str?}` | Step 2 |
| `schema_form` | `{plugin: str, schema_block: <plugin-schema>, prefilled: Mapping}` | Steps 1, 2 |
| `propose_chain` | `{steps: [{plugin: str, options: Mapping, rationale: str}], why: str, blockers: [str]}` | Step 3 |
| `recipe_offer` | `{recipe_name: str, slots: Mapping, alternatives: [str]}` | Step 2.5 (server-emitted) |

The user's reply is a typed `TurnResponse`:

```python
class TurnResponse(TypedDict):
    chosen: list[str] | None              # for select-type turns
    edited_values: Mapping | None         # for schema_form / inspect_and_confirm edits
    custom_inputs: list[str] | None       # for multi_select_with_custom free-add
    accepted_step_index: int | None       # for propose_chain (None = full accept; -1 = reject)
    edit_step_index: int | None           # for propose_chain step-locked re-emit
    control_signal: ControlSignal | None  # exit_to_freeform | request_advisor | reject
```

There is **no freeform user input field** in guided mode. Every input on screen is bound to the current turn's payload schema.

### 4.2 LLM tool surface in guided mode

The LLM's tool surface shrinks from today's ~30 tools to:

```
emit_turn(turn_type: str, payload: dict)        # the only way to interact with the user
inspect_source(blob_id: str)                    # discovery
list_sources() / list_sinks() / list_transforms()  # discovery
list_models(provider: str?)                     # for LLM-transform proposals in Step 3
get_plugin_schema(kind: str, plugin: str)       # for chain-step rationale
list_recipes()                                  # informational; pre-match runs server-side
request_advisor_hint(question: str)             # escalation
```

All state-mutation tools (`set_pipeline`, `upsert_node`, `set_source`, etc.) are **removed from the LLM's surface in guided mode.** Mutation happens server-side, deterministically, in response to user actions. This is the load-bearing safety property.

### 4.3 Legal-turn matrix

Which turn types are legal at which step:

```
Step 1 (source):           inspect_and_confirm | single_select | schema_form
Step 2 (sink):             single_select | multi_select_with_custom | schema_form
Step 2.5 (recipe match):   recipe_offer  (server-emitted only, never LLM-emitted)
Step 3 (transforms):       propose_chain | single_select   (single_select for clarifications)
```

If the LLM emits a turn type not legal at the current step, the server rejects with a structured error and grants one retry. A second illegal emission triggers auto-drop to freeform (§5.4).

### 4.4 Determinism guarantees

- **Server-emitted turns are deterministic.** `inspect_and_confirm` (after a blob is attached), `recipe_offer` (after the pre-match), the initial `single_select` in Step 1 — these are emitted by the server based on state, not by the LLM. They are byte-identical for byte-identical state.
- **LLM-emitted turns are constrained.** The LLM picks the turn type and constructs the payload, but the server validates the payload against the schema before emitting it to the user. Schema validation is deterministic and catches most LLM drift (wrong field names, missing required fields).
- **User responses are typed.** No free prose enters the protocol from either side.

---

## 5. Mode Lifecycle

### 5.1 Mode discriminator

`CompositionState` gains an optional field:

```python
@dataclass(frozen=True, slots=True)
class CompositionState:
    # ... existing fields ...
    guided_session: GuidedSession | None = None
```

When `guided_session is not None and guided_session.terminal is None`, the composer is in guided mode. When `guided_session is None`, freeform mode. When `guided_session.terminal is not None`, guided mode has ended and the freeform composer is active (with progressive disclosure in effect — see §8.2).

### 5.2 Entry — new sessions default to guided

A new empty session is created with `guided_session = GuidedSession(step=GuidedStep.STEP_1, ...)`. The frontend sees `guided_session != null` and renders the guided UI.

### 5.3 Manual exit-to-freeform mid-flow

Every guided turn is rendered alongside a persistent **"Switch to freeform mode"** button. Clicking it:
1. Sends `{control_signal: "exit_to_freeform"}` to `/composer/guided/respond`.
2. Server sets `guided_session.terminal = TerminalState(kind="exited_to_freeform", reason="user_pressed_exit", ...)`.
3. Server appends progressive-disclosure transition prompt to the next chat message context (§8.2).
4. Frontend re-renders as freeform with all state carried.

State carried across:
- `composition_state.source` / `nodes` / `outputs` — whatever the wizard built so far
- `composition_state.guided_session` — frozen, preserved as audit metadata; the freeform composer ignores it for runtime, but it remains in state forever
- Chat history — the guided turns and responses appear as structured messages in the conversation history

### 5.4 Auto-drop on terminal failure

Two cases:
1. **LLM emits illegal turn types twice** — server triggers auto-drop with `TerminalState(kind="exited_to_freeform", reason="protocol_violation")`.
2. **Step 3 chain proposal fails preview after one repair attempt + advisor consultation** — auto-drop with `TerminalState(kind="exited_to_freeform", reason="solver_exhausted")`.

In both cases, behaviour is identical to manual exit (§5.3): `guided_session.terminal` is set, progressive disclosure activates, frontend re-renders as freeform. The drop reason is recorded in audit (§9.1).

### 5.5 Successful termination

User clicks "Save and exit" after preview is green. `guided_session.terminal = TerminalState(kind="completed", pipeline_yaml=<rendered>)`. The frontend renders the completion summary. If the user subsequently sends a freeform chat message (e.g., "actually change the sink to CSV"), progressive disclosure is in effect.

### 5.6 Re-entry to a saved pipeline

Out of scope for v1 (§11). When a saved session is opened, `guided_session` is loaded from storage but treated as historical metadata; the composer renders in freeform mode regardless of `guided_session.terminal` state.

---

## 6. Server Architecture

### 6.1 Module layout

```
src/elspeth/web/composer/guided/
  __init__.py
  protocol.py        # TurnType enum, TurnPayload TypedDicts, TurnResponse, ControlSignal,
                     # legal-turn matrix, mode_transition_prompt()
  state_machine.py   # GuidedSession dataclass; step_advance() pure function
  steps.py           # Per-step handlers: handle_step_1_source, _step_2_sink,
                     # _step_2_5_recipe, _step_3_transforms
  recipe_match.py    # match_recipe(source, sink) -> RecipeMatch | None
  chain_solver.py    # solve_chain(source, sink, llm_client) -> ChainProposal
                     # Wraps _litellm_acompletion with the guided-mode skill prompt
  prompts.py         # Loads guided_pipeline.md skill; constructs Step 3 context block
  audit.py           # emit_guided_turn_event(), emit_guided_response_event(), etc.
  skills/
    guided_pipeline.md   # The new ≤80-line skill prompt
```

All new code lives under `composer/guided/`. Existing files (`composer/service.py`, `composer/tools.py`, `composer/state.py`, `composer/recipes.py`) are minimally extended:

- `composer/state.py`: add `guided_session: GuidedSession | None` field to `CompositionState`. Update `freeze_fields` per the deep-freeze contract.
- `composer/service.py`: add two endpoint handlers (§6.2) and call `composer.guided.audit.emit_*` at the relevant points.
- `composer/tools.py`: no edits — all mutation handlers reused unchanged.
- `composer/recipes.py`: add a small `match_recipe()` helper, or put it in `composer/guided/recipe_match.py` reading the recipe registry. (Decision: put it in `guided/`; keeps `recipes.py` focused on recipe definitions.)

### 6.2 Two new HTTP endpoints

```
POST /composer/guided/start
  body: {session_id: str}
  response: {
    guided_session: GuidedSession,
    next_turn: Turn,
    composition_state: CompositionState,
  }

POST /composer/guided/respond
  body: {
    session_id: str,
    turn_response: TurnResponse,
  }
  response: {
    guided_session: GuidedSession,
    next_turn: Turn | null,
    terminal: TerminalState | null,
    composition_state: CompositionState,
  }
```

Both are pure functions of `(state, response) → (next_state, next_turn)`. The server's state machine is the single source of truth for what step the user is on; the frontend reads `next_turn.step_index` and renders accordingly.

### 6.3 State machine

`state_machine.py` exposes one pure function:

```python
def step_advance(
    session: GuidedSession,
    response: TurnResponse,
    *,
    current_turn_type: TurnType,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditDirective]]:
    """
    Apply the user's response to the current guided session. Returns:
      (new_session, next_turn_or_None, terminal_state_or_None, directives_to_emit)
    """
```

The function encodes the legal-turn matrix, the step transitions, and the terminal-condition checks. It is unit-testable in isolation (no I/O, no LLM, no DB, no clock, no uuid).

`GuidedAuditDirective` (defined in `state_machine.py`) is the pure-function-return shape for audit instructions: a frozen `(tool_name, arguments)` pair where `tool_name` is one of the four discriminator strings (`guided_turn_emitted`, `guided_turn_answered`, `guided_step_advanced`, `guided_dropped_to_freeform`) and `arguments` is the payload mapping. The route handler (§6.2) fans each directive out to the matching `emit_*` helper in `composer/guided/audit.py`, supplying the runtime-only fields (`tool_call_id`, `started_at`/`finished_at`, `version_before`/`version_after`, `actor`) that bind it to a live `ComposerToolInvocation` record per Errata C4. The on-the-wire audit record is still `ComposerToolInvocation`; no new L0 contract type was introduced.

The `state: CompositionState` parameter from earlier drafts of this section was dropped during Phase 2 implementation — `step_advance` is structurally pure over `(session, response, current_turn_type)` alone, with composition-state mutations deferred to the route handler (which already holds the version snapshot needed for the audit record).

### 6.4 Recipe pre-match

`recipe_match.py` exposes:

```python
def match_recipe(
    source: SourceResolved,
    sink: SinkResolved,
) -> RecipeMatch | None:
    """Return the most-specific recipe matching the (source, sink) tuple, or None."""
```

Implementation is a list of `(predicate, recipe_name, slot_resolver)` tuples, one per registered recipe. Predicates are simple boolean expressions over `source.plugin`, `source.observed_columns`, `sink.outputs[].plugin`, etc. Slot resolvers map the resolved Step 1/2 state into the recipe's slot schema. **No fuzzy matching, no LLM.** If multiple recipes match, the most-specific (longest predicate chain) wins. If none match, return `None`.

The three recipes registered today (`classify-rows-llm-jsonl`, `split-by-numeric-threshold`, `fork-coalesce-truncate-jsonl`) each get a predicate. New recipes added later get matchers added in the same file.

### 6.5 Chain solver

`chain_solver.py` exposes:

```python
async def solve_chain(
    source: SourceResolved,
    sink: SinkResolved,
    composer_state: CompositionState,
    llm_client: LiteLLMClient,
    audit_recorder: BufferingRecorder,
) -> ChainProposal:
    """
    Invoke the LLM with the guided-mode skill + structured context block, expect a
    propose_chain turn back, validate the proposal against the engine's edge-contract
    rules, return ChainProposal (which the caller commits via existing mutation handlers).
    """
```

This is a thin wrapper. It builds the system prompt by concatenating `guided_pipeline.md` + the rendered context block, calls `_litellm_acompletion` (existing), receives the LLM's tool call, validates the payload, and returns `ChainProposal`. All audit, telemetry, token accounting, and advisor escalation use existing plumbing.

If the LLM's proposal fails `preview_pipeline`, the solver gets one repair attempt: feed the validation error back to the LLM and ask for a corrected chain. If still red, the caller (the step handler) escalates to advisor or auto-drops.

### 6.6 Mutation discipline

All state mutations in guided mode go through one of three paths, all of which already exist:

1. **Step-handler direct mutation** — `_execute_set_source`, `_execute_set_output`, etc. from `tools.py`, called from `composer/guided/steps.py` after the user confirms a turn.
2. **Recipe application** — `recipes.apply_recipe(name, slots)`, called when the user accepts a `recipe_offer`.
3. **Chain commit** — when the user accepts a `propose_chain`, the step handler walks the proposed steps and calls `_execute_upsert_node` + `_execute_upsert_edge` for each. All existing pre-validation in those handlers runs.

The LLM is never in the mutation call chain. The server's deterministic step handlers are the only mutators.

---

## 7. Frontend Architecture

### 7.1 Component layout

```
src/elspeth/web/frontend/src/components/chat/guided/
  GuidedTurn.tsx                    # Dispatcher: switches on turn.type
  InspectAndConfirmTurn.tsx
  SingleSelectTurn.tsx
  MultiSelectWithCustomTurn.tsx
  SchemaFormTurn.tsx
  ProposeChainTurn.tsx
  RecipeOfferTurn.tsx
  ExitToFreeformButton.tsx          # Persistent button alongside every turn
  GuidedHistory.tsx                 # Compact list of completed steps (collapsible)
  GuidedTurn.test.tsx
  <one .test.tsx per widget>
```

### 7.2 ChatPanel discriminator

`frontend/components/chat/ChatPanel.tsx` is extended with a top-level mode discriminator:

```tsx
if (guidedSession && !guidedSession.terminal) {
  // Guided mode — render guided history + active turn + exit button
  return (
    <>
      <GuidedHistory session={guidedSession} />
      <GuidedTurn turn={guidedSession.next_turn} />
      <ExitToFreeformButton />
    </>
  );
} else if (guidedSession?.terminal?.kind === "completed") {
  // Just-completed pipeline summary (transient; user clicks through to freeform or saves)
  return <CompletionSummary terminal={guidedSession.terminal} />;
} else {
  // Freeform — unchanged
  return <ExistingChatPanelBody />;
}
```

The freeform path is **unmodified** — guided mode is purely additive. Removing guided mode entirely would leave freeform working.

### 7.3 State management

`frontend/stores/sessionStore.ts` is extended with:

```typescript
interface GuidedSlice {
  guidedSession: GuidedSession | null;
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (turnResponse: TurnResponse) => Promise<void>;
  exitToFreeform: (reason: ExitReason) => Promise<void>;
}
```

Every action posts to the corresponding `/composer/guided/{start,respond}` endpoint and replaces the local `guidedSession` with the server response. **No optimistic updates** — server is authoritative.

### 7.4 Accessibility

Per the in-progress RC5-UX epic ([elspeth-de91358c30](filigree:elspeth-de91358c30)), every guided widget must:
- Use semantic HTML (`<fieldset>` for chip groups; `<button>` not `<div onClick>`)
- Pair every chip group with a visible label and `aria-describedby` for hints
- Honour `prefers-reduced-motion` for step-advance transitions
- Use only project design tokens (no hardcoded colours)
- Fire the existing `role="log"` `aria-live="polite"` announcement when a new turn arrives
- Maintain focus on the first interactive element of the new turn after step advance (reuse existing `useFocusTrap` patterns)

The `ProposeChainTurn` widget renders a step list in the same node-card styling as the freeform graph view's nodes. No new visual primitives are introduced.

### 7.5 Testing

Three layers, all in existing infrastructure:

1. **Unit (Vitest)** — one `.test.tsx` per widget, fed fixtures of the server's payload shape. Verifies render, callback wiring, edge cases.
2. **Store integration (Vitest)** — drives `sessionStore` actions, asserts state shape after each.
3. **End-to-end (Playwright)** — `tests/playwright/composer-guided.spec.ts` with three flows (recipe-match happy path; hand-built path; auto-drop path).

---

## 8. LLM Skill Prompt

### 8.1 New guided-mode skill

`composer/guided/skills/guided_pipeline.md`, target ≤80 lines, sections:

1. **Role and protocol** — closed turn taxonomy, read-only state, legal-turn matrix.
2. **Per-step playbook** — for each of Steps 1, 2, 3, the legal turn types and the expected default.
3. **Step 3 chain proposal** — the substantive section. Receives the `GUIDED CONTEXT` block from the server; instructions are short ("propose a chain that satisfies the contract; if you can't, emit a single_select clarification or request_advisor_hint").
4. **Hard rules that survive** — anti-fabrication (do not invent plugins/options), shape preservation (refuse with a named gap rather than degrade), audit boundary (do not propose audit sinks). Same intent as freeform skill, restated for guided context.

What gets removed (because structurally enforced): the 10 numbered convergence guardrails, the Connection Model essay, the Shape Catalog, the worked example, the anti-permission rule, the termination-gate rule.

### 8.2 Progressive disclosure on mode transition

When `guided_session.terminal` becomes non-null (any reason), the next chat message's system prompt is constructed by appending a transition segment + the freeform skill, **not** by replacing the guided skill:

```
[guided_pipeline.md content]

## Mode Transition — Guided → Freeform

You have just exited guided mode (reason: <user_pressed_exit | protocol_violation | solver_exhausted | completed_pipeline>).

The protocol restrictions above (closed turn taxonomy, read-only state, legal-turn matrix)
are LIFTED for the remainder of this session. You now have the full freeform tool surface
detailed below. The guided session's outcome is recorded in `composition_state.guided_session` —
do not re-run any work it already accomplished.

[pipeline_composer.md content (existing freeform skill)]
```

This applies to the first chat turn after transition. Subsequent turns use the freeform skill alone (the transition has already happened in the conversation history).

Token cost: ~340 lines for the transition turn (vs. ~250 today for freeform-only). Only paid when guided mode exits to freeform; happy-path completed-and-saved sessions don't pay it.

### 8.3 Existing freeform skill — unchanged

`composer/skills/pipeline_composer.md` is not modified. The two skills evolve independently. If the freeform skill is later rewritten or extended, the guided skill is unaffected and vice versa.

---

## 9. Audit, Telemetry, Error Handling

### 9.1 Audit event types

Four new audit-tier events emitted via `BufferingRecorder`:

| Event type | Payload |
|---|---|
| `guided_turn_emitted` | `step_index`, `turn_type`, `payload_hash`, `payload_payload_id` (payload-store ref), emitter (`server` \| `llm`) |
| `guided_turn_answered` | `step_index`, `turn_type`, `response_hash`, `response_payload_id`, `control_signal?` |
| `guided_step_advanced` | `prev_step`, `next_step`, `reason` (recipe_applied \| user_advanced \| auto_advanced) |
| `guided_dropped_to_freeform` | `prev_step`, `drop_reason` (user_pressed_exit \| protocol_violation \| solver_exhausted), `validation_result?` (set when `drop_reason="solver_exhausted"`) |

These are Tier 1 (audit-trust) events. Coercion failures crash; no silent fallbacks.

### 9.2 Telemetry

Operational signals via OpenTelemetry (existing path):
- Per-step latency histograms (Step 1, 2, 3 wall-clock)
- LLM token counts per Step 3 invocation
- Recipe-match hit-rate counter (incremented on every Step 2.5 evaluation; tagged with match_found yes/no)
- Auto-drop counter tagged by drop_reason
- Manual-exit counter

### 9.3 Logging

No new `slog` recommendations. Per `CLAUDE.md` primacy, every observable signal has been routed to audit (Tier 1) or telemetry (Tier 2). `slog.debug` is acceptable for transient debugging during development; no production logging is added.

### 9.4 Error handling matrix

| Failure | Resolution |
|---|---|
| LLM emits a turn type illegal at current step | Server rejects, one retry granted; second illegal emission triggers auto-drop with `drop_reason="protocol_violation"` |
| LLM `propose_chain` produces a chain that fails `preview_pipeline` | One repair attempt (server feeds validation error back to LLM); if still red, advisor escalation; if still red, auto-drop with `drop_reason="solver_exhausted"` |
| User's `multi_select_with_custom` includes a field name that doesn't exist in any plugin and isn't being added by transforms | Server validates, re-emits the turn with an error message attached |
| Recipe pre-match returns a recipe but slot resolution fails | Skip to Step 3 with a `single_select` clarification turn explaining what was incompatible |
| Frontend disconnects mid-step | Server `guided_session` is durable in `composition_states`; reconnect resumes at the same step (reuses existing session-recovery plumbing) |
| `composition_states` write fails during step advance | Existing transactional plumbing (`SessionsTransaction` from the progress-persistence work) covers this; no new mechanism |
| User repeatedly exits to freeform mid-flow | Tracked in telemetry; investigate as a UX issue, not a code bug |

No silent failures. Every error path either re-emits the current turn with structured error payload, or auto-drops with a recorded reason.

---

## 10. Testing Strategy

### 10.1 Three concentric rings

**Ring 1 — Protocol contract tests** (`tests/unit/web/composer/guided/`)
- `test_protocol.py` — turn type → payload schema validation; round-trip serialisation; legal-turn matrix
- `test_state_machine.py` — `step_advance()` pure-function tests; Hypothesis property tests for "any sequence of legal responses produces a valid GuidedSession"
- `test_recipe_match.py` — predicate evaluation; specificity ordering; slot-resolver outputs for each registered recipe
- `test_audit.py` — every audit event's payload conforms to the schema

No I/O, no LLM, no DB. Pure functions over data.

**Ring 2 — Server integration tests** (`tests/integration/web/composer/guided/`)
- `test_endpoints.py` — `/composer/guided/start` + `/composer/guided/respond` driven through realistic step sequences
- `test_chain_solver.py` — chain solver with stubbed LLM (`ChaosLLM` from `src/elspeth/testing/`); asserts produced chains are committed via existing mutation handlers and `preview_pipeline` returns green
- `test_progressive_disclosure.py` — mock LLM call after transition; assert response includes a non-guided tool call (proves the model picked up the "rules lifted" signal)
- `test_audit_emission.py` — full guided session, assert all expected audit events appear in `BufferingRecorder`

**Ring 3 — End-to-end (Playwright)** (`tests/playwright/composer-guided.spec.ts`)
- Happy path: CSV blob attached → recipe-match → user accepts → preview green → YAML emitted. Asserts the demo SLA (≤4 clicks, ≤30 seconds).
- Hand-built path: CSV → CSV with custom transforms; LLM stub returns canned valid chain.
- Auto-drop path: force LLM stub to produce invalid chain; assert auto-drop fires and freeform mode receives partial pipeline + chat history.

### 10.2 LLM token budget for CI

The chain-solver test in Ring 2 has two modes:
- **Cheap CI mode** — uses a stubbed LLM with canned proposals. Runs every PR.
- **Real-LLM mode** — invokes the real LLM with the guided-mode skill, asserts the returned chain passes `preview_pipeline`. Runs once per PR if budget permits, or nightly otherwise.

The ChaosLLM fixture is used for the cheap mode; existing test infrastructure is sufficient.

### 10.3 Demo-path assertion

A single Playwright test asserts the demo SLA from §1.3:

```typescript
test('demo path: CSV → classify-rows-llm-jsonl', async ({ page }) => {
  const start = Date.now();
  let clicks = 0;
  // ... attach blob, walk through guided steps ...
  expect(clicks).toBeLessThanOrEqual(4);
  expect(Date.now() - start).toBeLessThan(30_000);
  // Assert the audit trail shows ≤2 LLM calls
});
```

This is the test that proves guided mode delivered on its goal.

---

## 11. Out of Scope / Future Work

| Item | Reason for deferral | Effort to add later |
|---|---|---|
| Freeform → guided switching | Asymmetric design intentional; rare use case; non-trivial inference of "where in the wizard would this freeform pipeline pick up?" | Medium — would require pipeline-shape analysis to map back into wizard steps |
| Re-entry to a saved pipeline in guided mode | Re-entry is the "I know what I'm doing" case; guided is for novice authoring | Medium — requires step-replay UI |
| Additional turn types beyond the six | Six covers Steps 1/2/3 well; new types should be requirements-driven | Small per type — add to taxonomy + dispatcher + widget |
| Multi-source pipelines | Out of scope for ELSPETH itself (engine supports one source per run) | Engine-level change first |
| Visual graph editing inside guided mode | Linear chip steps are deliberate; if user needs graph view, exit to freeform | Large — would require integrating React Flow into a turn type |
| LLM choosing between multiple matched recipes | Pre-match returns at most one recipe (most-specific wins, deterministic) | Small — relax specificity rule, add disambiguation turn |
| Plugin-pack defaults influencing turn options | Existing plugin discovery already incorporates packs; no special-casing needed | Already covered by reuse of plugin registry |
| Wizard tutorial / annotated walkthrough | Guided mode is the wizard, not a tutorial about it | Large — separate scope |

---

## 12. Open Questions / Risks

These are flagged for plan-review before implementation.

1. **Freeze-guard pattern for `GuidedSession`.** I asserted in §5.1 that `GuidedSession` follows the `freeze_fields` pattern from `CLAUDE.md`'s deep_freeze contract. Need to verify that `tuple[TurnRecord, ...]` and nested `Mapping` payloads are correctly handled by `freeze_fields` — or if `deep_freeze` direct calls are needed for the per-element comprehension cases.

2. **Session-recovery plumbing for mid-step disconnects (§9.4).** I claimed reuse of "existing session-recovery plumbing" without full verification. The progress-persistence design ([elspeth-90b4542b63](filigree:elspeth-90b4542b63), revision 4) introduced `SessionsTransaction`; need to confirm `composition_states` is part of the per-turn write block and that mid-step `guided_session` updates land transactionally.

3. **`_run_sync` interaction.** `composer/service.py` uses `_run_sync` and `run_sync_in_worker` for sync DB dispatch. The new endpoints must follow the same pattern; the spec says "two new endpoints" but doesn't yet specify which sync-dispatch path they use. To be resolved during implementation-planning.

4. **Recipe-match specificity ordering.** With three recipes today, "most specific wins" is unambiguous. As recipes are added, the ordering rule may need a tiebreaker beyond "longest predicate chain" (e.g., recipe registration order, explicit priority). Flag for future-proofing.

5. **`propose_chain` payload size.** A complex chain (5+ steps with full options) could produce a large payload. The audit payload-store handles this, but the frontend's `ProposeChainTurn` widget should virtualise rendering for chains over, say, 10 steps. Not blocking; flag for the widget implementation.

6. **Chain solver cost on the demo path.** The demo path (CSV → recipe-match) costs zero LLM tokens in guided mode. But a *non*-recipe-matching demo (less common) costs one Step 3 invocation. The demo-path SLA in §1.3 says "≤2 LLM calls" — confirm this includes the audit/telemetry calls if any (typically not LLM-billed).

7. **Edit-step-N flow in `propose_chain`.** §3.4 says editing a step "re-emits propose_chain with that step locked" — the exact protocol for this needs spelling out: does the LLM see the full prior chain + which step is locked, or just the locked step + surrounding context? Specify during implementation-planning.

8. **Plugin-pack visibility in guided mode.** If a deployment has Azure pack but not LLM pack (or vice versa), the `single_select` chips in Step 1/2/3 should reflect what's installed. Reuse the existing `list_sources` etc. — verify they correctly filter by enabled packs.

---

## 13. Decision Records

Decisions made during the brainstorm, in order:

| Decision | Rationale | Reference |
|---|---|---|
| Shape: structured chat (B), not pure form or hybrid | "Structured for both human and LLM" — protocol constrains both sides; freeform fallback handled by mode-switch, not in-protocol turn | Brainstorm screen `01-guidance-shape.html` |
| Sequencing: both-ends-pinned (C), not source-first | C surfaces the engine's contract model; both anchors visible to LLM at Step 3; structurally satisfies skill rules 0/4/7/8 | Brainstorm screen `02-step-sequencing.html` |
| Refinement 1: required_fields opt-in with "let source decide" escape | Captures both "deliverable in mind" and "exploring my data" users without forcing premature commitment | Brainstorm response, Q2 |
| Refinement 2: recipe pre-match between Step 2 and Step 3 | Zero LLM tokens for happy paths; demo-bulletproof for shapes the recipes already cover | Brainstorm response, Q2 |
| Turn types: closed taxonomy of six (no `freeform_fallback`) | Auto-drop to freeform handles the "LLM can't proceed" case at the end; in-protocol freeform turn would let the protocol degrade | Brainstorm response, Q3; screen `03-turn-taxonomy.html` |
| LLM is read-only with respect to pipeline state | Categorically eliminates degraded-shape commits, audit fabrication, contract-mismatched wiring | Section §4.2 |
| Freeform → guided switching not supported in v1 | Asymmetric is simpler; gradient-toward-complexity model (training wheels) | Brainstorm response, Q4 |
| Re-entry to saved pipeline always lands in freeform | Re-entry is "I know what I'm doing"; guided is novice-onboarding only | Brainstorm response, Q4 |
| Progressive disclosure on mode transition | Preserves LLM coherence across mode boundaries; layered prompt rather than context swap | Section §8.2 |
| Audit primacy — every guided turn emitted as a Tier 1 event | Reproducible authoring replay; matches engine's "every decision traceable" standard | Section §9.1 |

---

## 14. Implementation Phasing

Recommended phasing for the implementation plan (to be detailed in the writing-plans output):

**Phase 1 — Protocol foundation (no UI, no LLM)**
- `composer/guided/protocol.py`, `state_machine.py`, `recipe_match.py`, `audit.py`
- Ring 1 unit tests
- Endpoint stubs (`/composer/guided/start`, `/respond`) returning canned data
- `CompositionState.guided_session` field + freeze-guard

**Phase 2 — Step 1 + Step 2 (deterministic, no LLM)**
- Step handlers for Steps 1, 2, 2.5
- Recipe pre-match wired up
- Frontend widgets for `inspect_and_confirm`, `single_select`, `multi_select_with_custom`, `schema_form`, `recipe_offer`
- Ring 2 integration tests for the recipe-match happy path

**Phase 3 — Step 3 chain solver + LLM skill**
- `chain_solver.py` invoking real LLM with guided skill
- `ProposeChainTurn` widget
- `guided_pipeline.md` skill written and shipped
- Ring 2 integration tests with stubbed LLM

**Phase 4 — Mode lifecycle**
- Manual exit-to-freeform button + state-carry
- Auto-drop on terminal failure
- Progressive disclosure prompt construction
- Ring 3 Playwright tests for happy / hand-built / auto-drop paths

**Phase 5 — Polish + demo-SLA verification**
- Demo-path Playwright assertion
- Telemetry instrumentation
- Documentation updates (user-manual, troubleshooting)

Each phase is an independently reviewable PR.

---

**End of design.**
