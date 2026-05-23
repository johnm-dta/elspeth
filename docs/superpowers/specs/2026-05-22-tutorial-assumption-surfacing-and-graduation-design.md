# Tutorial assumption-surfacing and graduation design

**Date:** 2026-05-22
**Status:** revised after multi-reviewer pass (operator sign-off pre-revision; revision pending operator review)
**Supersedes (partial):** `project_composer_canonical_test_case` (memory entry)

## Revision history

- **2026-05-22 (rev 0):** initial spec written and committed (`421d178de`)
  after section-by-section operator approval.
- **2026-05-23 (rev 1, this revision):** absorbed findings from a
  five-reviewer pass (UX-critic, solution-design-reviewer, systems-thinking
  leverage-analyst, python-code-reviewer, LLM-safety-reviewer). Material
  changes from rev 0:
  - Added explicit **Contract changes (L0)** subsection — the existing
    `interpretation_sites()` plugin-gate at `interpretation_state.py:78-79`
    early-returns for non-LLM nodes; Class 2 and Class 3 cannot reach the
    runtime preflight without an `InterpretationKind` discriminator added
    to L0 contracts and the gate removed in L3. Rev 0 framed this as a
    planning checkpoint; it is now specified as a contract change.
  - Added **Preflight gate per class** — Class 2 (invented URL list) blocks
    `web_scrape` execution; audit-only recording would be forensic, not
    protective. Class 3 (prompt template) blocks `llm` transform execution.
  - Added **Differentiated review surface** — `InterpretationReviewTurn`
    gains a `kind` prop with per-class headings/copy/amend availability;
    tutorial mount suppresses the opt-out button and the amend entry; a
    progress indicator names how many assumptions remain.
  - Added **Opt-out audit semantics** — `interpretation_review_disabled=true`
    skips the human, not the audit. Each Class 2/3 artefact still emits an
    `AUTO_INTERPRETED_OPT_OUT` row.
  - Revised **Turn 7 graduation copy** — split bullet 1 (factual + invitation,
    not ownership-assertion) and added bullet 4 covering hallucination,
    prompt-injection-from-fetched-content, and data-source verification.
  - Enumerated **downstream readers of `tutorial_completed_at`** affected by
    moving the completion gate from Turn 6 to Turn 7.
  - Added **Deferred follow-ups** section capturing the systems-thinker's
    L10 first-run gate, asymmetric / time-bounded opt-out friction, and the
    Approach-C2 interactive amendment path (rev-0 already deferred the
    last; the other two are new).
  - Closed **Planning Checkpoint 2** affirmatively — `web_scrape` plugin
    confirms `abuse_contact` and `scraping_reason` as required fields at
    `src/elspeth/plugins/transforms/web_scrape.py:82,86` with a
    `@field_validator` at `:100`. Rev 0's "verify they exist" is now "they
    exist; canonical prompt values pass validation."
  - Specified **Planning Checkpoint 3** progress-poller headline strategy
    rather than leaving it open.

## Problem

The first-run tutorial (`HelloWorldTutorial`) presents the LLM-composed
pipeline as a passive summary on Turn 2b ("ShowBuilt"). It actively
suppresses Elspeth's existing interpretation-review mechanism by calling
`optOutOfInterpretations(session.id)` and `resolveTutorialInterpretations(...)`
inside `buildTutorialDraft` in
`src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`.
The composer's vague-term interpretations (e.g. defining what "cool"
means in "rate how cool they are") are auto-resolved with
`accepted_as_drafted` before the user sees Turn 2b. The `InterpretationReviewTurn`
component already mounted in `TutorialTurn2bShowBuilt.tsx:78-89` is dead
code — by the time it renders, nothing is pending.

Two consequences:

1. The user is never shown the assumption-review gesture that the rest of
   the composer relies on. They leave the tutorial believing pipelines are
   atomic LLM outputs rather than assemblies of LLM-authored decisions
   that they (the auditor) need to sign off on.
2. The audit story Turn 5 shows is a fiction in spirit: it claims the
   user approved the pipeline, but every meaningful approval was
   silently auto-resolved.

Separately, the canonical tutorial prompt
(`"create a list of 5 government web pages and use an LLM to rate how cool they are"`)
produces results dominated by raw HTML when the `web_scrape` transform
runs. The output is functional but not demoable — visually overwhelming
and not aligned with a "look how cleanly the pipeline produced a
structured answer" story.

Finally, the tutorial currently ends at Turn 6 (mode choice) with no
explicit reminder that pipelines coming out of the composer are
AI-generated and require human review before execution. This was raised
during operator review as the missing closing message.

## Goals

1. **Stop suppressing the interpretation system in the tutorial.** The
   user explicitly approves each assumption the composer makes.
2. **Surface LLM-invented source data via the same mechanism.** When the
   composer has to invent source content the user did not provide (URL
   lists, parsed columns), surface that draft as a formal
   `request_interpretation_review` requirement. Execution preflight
   refuses to start the pipeline until the user approves — recording
   approval after `web_scrape` already fetched the URLs would be
   forensic, not protective. In-UI editing of the draft is deferred
   (Approach C2); the approve / chat-to-revise gestures are the slice.
3. **Surface every LLM-authored prompt template.** Each `prompt_template`
   the composer writes for an `llm` transform surfaces for the user to
   approve before the pipeline can run. Applies to all composer
   sessions, not just the tutorial. Opt-out continues to be the existing
   `interpretation_review_disabled=true` flag, with one rev-1 addition:
   the opt-out skips the human review surface but still emits an
   `AUTO_INTERPRETED_OPT_OUT` audit row, so headless-harness runs are
   still attributable.
4. **Replace the canonical prompt** with a longer one that produces a
   cleaner output (HTML stripped, JSON sink), exercises explicit scrape
   options (abuse contact, scraping reason), and surfaces a non-trivial
   subjective term ("primary colour") for the user to approve.
5. **Add a Turn 7 graduation page** reminding the user that pipelines
   are AI-generated content, must be read before executing, and that
   asking the composer for plain-English explanations is encouraged.

## Non-goals

- The "tutorial runs on local dev but fails on Azure-backed staging"
  report. Parked; revisit separately.
- Allowing the user to author the source themselves (paste URLs, pick
  a file). The approval gesture is review-and-approve, not
  author-from-scratch.
- Approving routine source / transform / sink proposals. Auto-accept
  via `acceptCompositionProposal` stays for the proposal banner; only
  LLM assumptions go through manual review.
- A new dedicated "Turn 2.5 — Review assumptions" tutorial step.
  Approach B (from brainstorming) was rejected.
- Interactive amendment that mutates source content (Approach C2 from
  brainstorming). In this slice the user approves the LLM's draft as-is
  or chats to revise it; in-UI editing of an invented URL list or a
  prompt template is deferred. C2 is carved off as a follow-up (see
  **Deferred follow-ups** item 1). Note: this non-goal is about the
  *editing affordance*, not the *gating semantics* — execution preflight
  still refuses to start until the user approves (see **Preflight gate
  per class**).
- Touching the freeform / guided default-mode selection.
- Anything about runtime preflight, audit-readiness panel, or composer
  flow surfaces outside Turn 2b and the new Turn 7.

## Design

### Contract changes (L0 + L3)

The rev-0 design implicitly assumed that classes 2 and 3 could flow
through the existing interpretation-review machinery unchanged. The
multi-reviewer pass established that this is false: `interpretation_sites()`
at `src/elspeth/web/interpretation_state.py:78-79` early-`continue`s on
`node.plugin != "llm"`, so a Class 2 requirement attached to a source
node would never reach `materialize_state_for_execution()` and would
never block the runtime preflight. Class 3 has the related problem that
`_materialize_node_for_execution()` always performs slot-substitution
into `prompt_template_parts`, which has no slot for an
approve-this-entire-template requirement. Both gaps require explicit
contract work before implementation begins.

#### New L0 enum: `InterpretationKind`

Add to `src/elspeth/contracts/composer_interpretation.py` next to the
existing `InterpretationChoice` and `InterpretationSource` enums:

```python
class InterpretationKind(str, Enum):
    """Discriminator for the three classes of LLM assumption surfaced
    through the interpretation-review mechanism."""

    VAGUE_TERM = "vague_term"          # Class 1: subjective user term
    INVENTED_SOURCE = "invented_source" # Class 2: LLM-fabricated source data
    LLM_PROMPT_TEMPLATE = "llm_prompt_template"  # Class 3: LLM-authored prompt
```

This is a hash-domain change for `InterpretationEventRecord`. Bump
`INTERPRETATION_HASH_DOMAIN_V1` → `..._V2` and stamp
`hash_domain_version="v2"` on every emitted row. Closed-enum governance
applies (the project's standard four-step checklist: amend contract,
extend enum, update closed-enum tests, audit writer-path).

#### L3 changes to `InterpretationRequirement` and `InterpretationReviewPending`

In `src/elspeth/web/interpretation_state.py`:

- Add `kind: InterpretationKind` to the `InterpretationRequirement`
  `TypedDict` (non-optional; every requirement declares its class).
- Add `kind: InterpretationKind` to the per-site tuple emitted by
  `interpretation_sites()` — currently `tuple[str, str]` (node_id, user_term),
  becomes `tuple[str, str, InterpretationKind]`. The `sites` field on
  `InterpretationReviewPending` widens accordingly. `InterpretationReviewPending`
  remains `frozen=True, slots=True`; the sites tuple is still all-scalar so
  no `freeze_fields()` call is needed.
- Remove the `if node.plugin != "llm": continue` gate at line 78. Replace
  with a structured-requirements walk that runs on every node carrying
  `INTERPRETATION_REQUIREMENTS_KEY`. The legacy `{{interpretation:...}}`
  placeholder path retains its `node.plugin == "llm"` guard inside the
  legacy branch only.
- `_materialize_node_for_execution()` gains a per-kind branch:
  - `VAGUE_TERM` and `LLM_PROMPT_TEMPLATE` resolved via the existing
    slot-substitution into `prompt_template_parts` — but Class 3's
    "accepted value" is the entire template text, so the substitution is
    an identity rewrite. Recording `resolved_prompt_template_hash` is the
    load-bearing step.
  - `INVENTED_SOURCE` is a record-and-pass branch: confirm the
    requirement's `accepted_value` matches the source node's current
    payload (drift detection); no substitution into `prompt_template_parts`
    (a source node doesn't have one).
- `affected_node_id` semantics widen in the L0 contract docstring at
  `composer_interpretation.py:135` from "the LLM-transform node this binds
  into" to "the node this requirement binds to — for `VAGUE_TERM` and
  `LLM_PROMPT_TEMPLATE` this is the consuming `llm` transform; for
  `INVENTED_SOURCE` this is the source node whose payload was invented".
  This widening is captured in the `InterpretationKind` ADR-equivalent
  language in the docstring.

#### Layer-rule compliance

- `InterpretationKind` enum lives in L0 (`contracts/`).
- The `TypedDict` extension lives in L3 (`web/`).
- No upward import is introduced. The tier-model allowlist does not need
  an entry for this change.

### Three classes of assumption

| # | `InterpretationKind` | Class | Mechanism | Status today |
|---|---|---|---|---|
| 1 | `VAGUE_TERM` | Vague-term interpretations ("primary colour", "cool", "important") | `request_interpretation_review` | Already implemented; tutorial currently suppresses it |
| 2 | `INVENTED_SOURCE` | Invented source data (URL lists the LLM made up) | `request_interpretation_review` + new L0/L3 contract work | Currently narrated in chat only; no audit event |
| 3 | `LLM_PROMPT_TEMPLATE` | LLM prompt templates themselves | `request_interpretation_review` + new L0/L3 contract work | Not surfaced at all today |

All three resolve through the same UI surface (`InterpretationReviewTurn`,
extended with a `kind` prop — see **Differentiated review surface** below)
and the same backend opt-out flag (`interpretation_review_disabled`).

#### Approval semantics per class (revised from rev 0)

Rev 0 framed all three classes as "audit-only review" — review records
the user's decision but does not directly mutate the source or template.
The five-reviewer pass established that this framing is unsafe for Class 2:
the LLM invents a URL list, then `web_scrape` fetches every URL on
execution; recording approval after the fetch is forensic, not protective.
Approval semantics now differ by class:

- **Class 1 (vague-term).** Approval substitutes the accepted value into
  `prompt_template_parts` slots. Unchanged from today. Rejection forces
  the user back into chat to revise the term.
- **Class 2 (invented source).** Approval is **gating** — execution
  preflight (see **Preflight gate per class** below) refuses to start the
  pipeline until the requirement is resolved. The accepted value is
  validated against the source node's current payload (drift detection
  with a clear error if the payload has changed). Rejection forces the
  user back into chat to revise the URL list. The point of approval is
  the audit record AND the gate — not just the audit record.
- **Class 3 (LLM prompt template).** Approval is also gating but the
  "accepted value" IS the template text. The materialisation path is an
  identity rewrite; the load-bearing artefact is the
  `resolved_prompt_template_hash` stamped into the requirement. Rejection
  forces the user back into chat to revise the template.

Interactive in-UI editing of the invented source data or the prompt
template — Approach C2 from brainstorming — remains deferred. The user
revises by chat in this slice; the contract leaves room for C2 (the
`draft` field on `InterpretationRequirement` is already shaped for it,
see `interpretation_state.py:223`).

#### Cross-session policy

Classes 2 and 3 are composer-skill changes that affect every composer
session, not just the tutorial. The opt-out lever is the existing
`interpretation_review_disabled=true` flag — same switch that already
governs vague-term reviews. Headless harnesses (eval suites, persona
regression) opt out via that flag exactly as they do today; the tutorial
does not opt out, so the user sees every assumption.

#### Preflight gate per class

The execution preflight (in `src/elspeth/web/sessions/service.py`,
the materialisation path that consumes `InterpretationReviewPending`)
must hard-fail when any `kind`-tagged requirement is pending. Per-class
gate semantics:

| `kind` | Gate scope | Failure mode if missing |
|---|---|---|
| `VAGUE_TERM` | Refuse `llm` transform execution that consumes the unresolved placeholder | Existing behaviour; documented for completeness |
| `INVENTED_SOURCE` | Refuse **source node execution** — pipeline cannot start | LLM-fabricated URL list would be fetched by `web_scrape` before user approval; audit record would post-date the network egress |
| `LLM_PROMPT_TEMPLATE` | Refuse `llm` transform execution | Pipeline would run with an unreviewed prompt the LLM authored for itself |

The tutorial path uses the **same gate**. Rev 0 described the tutorial
review as "audit-only" — that framing applied only to the UI
(approve/reject without C2 in-place editing). The gate is unconditional
and is the load-bearing trust property. If the gate is bypassable
(e.g. because the LLM silently failed to emit the surface), the trust
claim of this spec is aspirational. Specifically:

- The composer's `request_interpretation_review` calls are non-deterministic
  skill guidance, not a runtime guarantee. The preflight must defend in
  depth: for every `llm` node with a non-empty `prompt_template`, there
  must exist a resolved Class 3 requirement with
  `resolved_prompt_template_hash == stable_hash(prompt_template)`. If
  not, the preflight refuses execution with a clear
  `INTERPRETATION_REVIEW_MISSING` error citing the offending node.
- For source nodes whose payload was authored by the composer (rather
  than by the user via `set_source_from_blob` with a user-supplied blob),
  the preflight requires a resolved Class 2 requirement. **The
  composer skill alone cannot enforce this** — it must be enforced
  structurally. Open question for the implementation plan: how does the
  preflight distinguish LLM-authored from user-authored source payloads?
  One workable mechanism is a provenance flag stamped onto `NodeSpec`
  when the composer's MCP tools (`set_source`, `set_source_from_blob`,
  `create_blob`) run without an accompanying user-blob upload — added as
  the implementation plan's first task.

#### Differentiated review surface

The five-reviewer pass converged on a UX risk: a single review surface
rendering N identically-chromed cards for three artefact classes trains
accept-all behaviour. Mitigation:

- `InterpretationReviewTurn` gains a required `kind: InterpretationKind`
  prop. Component branches per kind for heading and body framing:
  - `VAGUE_TERM`: heading "Interpretation review" (today's). Body: "When
    you said *{user_term}*, I read that as roughly *{accepted_value}*."
  - `INVENTED_SOURCE`: heading "Invented source data". Body: "You did not
    provide a list of {what} — I drafted one. Review before the pipeline
    fetches anything." Renders the draft as a list (URLs as `<ol>`,
    one-per-line monospaced).
  - `LLM_PROMPT_TEMPLATE`: heading "LLM prompt template". Body: "This is
    the instruction I wrote for the `{node_id}` transform. Read it before
    approving." Renders the template inside a `<pre>` block with
    visible-whitespace handling. The approve button is disabled until the
    user has scrolled the `<pre>` to its end (`scrollTop + clientHeight >= scrollHeight - 8`).
    This is the differentiated-friction lever; it converts a one-click
    rubber-stamp into an attention-forcing gesture for the highest-risk
    class without blocking expert users.
- The amend entry ("Change it: I meant…") is hidden for `INVENTED_SOURCE`
  and `LLM_PROMPT_TEMPLATE` in this slice — those classes route through
  chat-based revision (C2 deferred). The opt-out button ("Stop reviewing
  interpretations this session") is hidden in every mount of the
  component inside the tutorial flow, regardless of `kind` — it should
  not be a one-click escape from the tutorial's load-bearing trust
  lesson. The component's opt-out section is rendered behind a
  `showOptOut: boolean` prop that defaults to `true`; tutorial mounts
  pass `false`.
- A progress indicator above the card group names the queue: "Reviewing
  assumption 1 of N". Renders only when N > 1. Focus management: on
  initial mount, focus the first unresolved card's heading; on resolve,
  focus the next unresolved card's heading; on last resolve, focus the
  now-enabled "Looks good, run it" CTA.

The same component is used outside the tutorial. The kind-aware
behaviour applies universally — power users in the freeform composer
benefit from the same differentiated framing.

#### Opt-out audit semantics

`interpretation_review_disabled=true` (the existing per-session flag
used by headless harnesses) skips the human review surface but **must
not** skip the audit record. For each Class 1/2/3 surface the composer
would have created, the system emits an `InterpretationEventRecord` with:

- `choice = InterpretationChoice.OPTED_OUT` (existing enum value)
- `source = InterpretationSource.AUTO_INTERPRETED_OPT_OUT` (existing enum value)
- `kind` = the appropriate `InterpretationKind` (rev-1 contract addition)
- `accepted_value` = the LLM's draft (i.e. the value that *would* have been
  presented for review)
- `resolved_prompt_template_hash` populated for `LLM_PROMPT_TEMPLATE`

Rationale: the opt-out path today loses Class 2 and Class 3 audit data
entirely (chat narration is not an audit record). After this change, an
operator querying `explain(..., token_id=...)` against a pipeline run by
a headless harness can still distinguish LLM-invented data from
user-provided data, and can still verify the LLM's authored prompt
template against its resolved hash. The opt-out becomes "skip the
human", not "skip the audit" — exactly per the CLAUDE.md attributability
test.

Implementation locus: the LLM-composer's `request_interpretation_review`
MCP tool. When the session has `interpretation_review_disabled=true`,
the tool short-circuits the user-review path but still writes the
`AUTO_INTERPRETED_OPT_OUT` row. The skill amendment below names this
requirement explicitly.

#### Avoiding redundancy with embedded vague terms (3a)

When a prompt template contains an embedded vague-term interpretation
(e.g. `"Rate how {{interpretation:primary_colour}} this page is. ..."`),
both surfaces fire: the user first approves the term definition ("what
primary colour means"), then approves the prompt template ("is this
the right way to ask the model"). The two reviews are conceptually
distinct — one is "what does this word mean", the other is "is this
the right way to ask the model". Folding them together (variant 3b) was
considered and rejected as confusing the audit story.

### Composer-skill amendment

The chat-narration-only pattern at
`src/elspeth/web/composer/skills/pipeline_composer.md:608-609`
(currently: "surface your column interpretation in the narration so the
user can correct it" and "Generate the 5 URLs yourself, present them in
the proposal narration for user review") is **replaced** by the
following bullet, which subsumes both lines plus the new
prompt-template surfacing requirement:

> **Surface every assumption you make.** Three classes of LLM-authored
> content must be surfaced for the user via `request_interpretation_review`
> before the pipeline can run. Every call carries a `kind` matching the
> `InterpretationKind` enum from
> `elspeth.contracts.composer_interpretation`. When
> `interpretation_review_disabled=true`, the call still emits an audit
> event (`source=AUTO_INTERPRETED_OPT_OUT`) but no human-review surface
> renders. Opt-out skips the human, not the audit — never substitute
> chat narration for an audit event.
>
> 1. **Vague-term interpretations** (`kind=VAGUE_TERM`) — when the user
>    prompt contains a subjective or underspecified term ("cool",
>    "important", "primary colour"), surface your definition as you do
>    today (see the existing "Subjective Interpretation Review" section).
> 2. **Invented source data** (`kind=INVENTED_SOURCE`) — if you had to
>    invent source content the user did not provide (URL lists, parsed
>    column values, illustrative records), call
>    `request_interpretation_review` immediately after `set_source` /
>    `set_source_from_blob` succeeds, with
>    `user_term = "inline_source_url_list"` (or similar stable identifier)
>    and `llm_draft = <human-readable rendering>`. The runtime preflight
>    will refuse to start the pipeline until this is resolved — chat
>    narration does not satisfy the gate.
> 3. **LLM prompt templates** (`kind=LLM_PROMPT_TEMPLATE`) — every
>    `prompt_template` you author for an `llm` transform must surface via
>    `request_interpretation_review` with
>    `user_term = "llm_prompt_template:<node_id>"` and
>    `llm_draft = <the raw template text>`. Surface this in addition to
>    any embedded vague-term interpretations; the term review answers
>    "what does this word mean", the template review answers "is this
>    the right way to ask the model". The runtime preflight will block
>    execution until both are resolved.

(The bullet above is the entire replacement; there is no separate
"second skill edit". The chat-narration-only guidance is gone, not
augmented. The skill is non-deterministic guidance, so it is paired with
defense-in-depth in the preflight: see **Preflight gate per class**
above — if the LLM silently fails to call `request_interpretation_review`
for a `LLM_PROMPT_TEMPLATE`, the preflight detects the missing
resolved-template hash and refuses execution.)

### Canonical prompt change

Replace the value of `CANONICAL_TUTORIAL_PROMPT` (frontend) and
`CANONICAL_SEED_PROMPT` (backend) with:

```
Please go to the following web pages, use abuse contact noreply@dta.gov.au
and scraping reason 'DTA technical demonstration'. Read the HTML for each
page, have an LLM identify the primary colours for each government agency.
Remove the HTML and save the rest to a json file.
```

(Two typos in the operator's original were corrected: `followin gmeb` →
`following web`.)

Locations updated:

- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts:3`
- `src/elspeth/web/preferences/tutorial_cache.py:47`
- `src/elspeth/web/composer/skills/pipeline_composer.md:609` and `:669`
  (example block now demonstrates the new prompt and the three-class
  assumption surfacing)
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx:105-106`
  (anchoring comment)

Test fixtures referencing the old prompt are listed under **Test impact**
below.

#### Expected pipeline shape from the new prompt

| Layer | Plugin | Options from prompt | Surfaced assumption? |
|---|---|---|---|
| Source | `inline_blob` (URL list) | URLs **invented** by LLM | Class 2 |
| Transform 1 | `web_scrape` | `abuse_contact`, `scraping_reason` (explicit) | None |
| Transform 2 | `llm` | `prompt_template` for primary-colour identification | Class 1 ("primary colour") + class 3 (the template) |
| Transform 3 | `field_drop` (or similar) | drops the HTML field | None |
| Sink | `jsonl` | "save to json file" | None |

Total of 3 assumptions surfaced for the canonical prompt (URL list +
primary colour definition + LLM prompt template). The user clicks
through 3 review cards on Turn 2b before "Looks good, run it" is
enabled.

### Tutorial code changes

#### `buildTutorialDraft` (TutorialTurn2Describe.tsx)

- **Remove** the `await api.optOutOfInterpretations(session.id)` call
  before `sendMessage`.
- **Remove** the `resolveTutorialInterpretations(session.id, compositionState)`
  call after the proposal-accept loop.
- **Keep** the proposal auto-accept loop (lines 180–182) — that handles
  routine source/transform/sink proposals.
- **Add** an offensive-programming assertion in the post-build composer-
  preferences fetch: if `composerPreferences.interpretation_review_disabled === true`,
  throw with a clear "tutorial sessions must not have interpretation
  review disabled" message. This catches the invariant violation loudly
  rather than silently degrading to the old auto-resolve behaviour.

#### `TutorialTurn2bShowBuilt.tsx`

- The `InterpretationReviewTurn` mount at lines 78–89 lights up
  naturally, but the wrapper needs the new kind-aware behaviour from
  **Differentiated review surface** (above):
  - Iterate over **every** pending interpretation event for the
    session, not just the first one (today's `useMemo` finds a single
    event). Each interpretation gets its own review card.
  - Render the cards **simultaneously** as a vertical stack (not as a
    sub-wizard). Above the stack: a count header "N assumptions to
    review" (rendered only when N > 1). Per-card chrome and copy
    differs by `kind` per the **Differentiated review surface** spec.
  - Focus management: on initial mount, focus the heading of the first
    unresolved card; on resolve, the next-unresolved card's heading
    receives focus; when all are resolved, the "Looks good, run it" CTA
    receives focus. Avoids the on-mount focus-fight between simultaneous
    `InterpretationReviewTurn` mounts.
  - Tutorial mount passes `showOptOut={false}` (suppresses the "Stop
    reviewing interpretations this session" button) and
    `showAmend={false}` for `INVENTED_SOURCE` and `LLM_PROMPT_TEMPLATE`
    cards (those route through chat-based revision, not in-place edit).
- Disable the "Looks good, run it" / "Show me the graph" CTA while any
  pending interpretation exists. The button label remains the same;
  the disabled state surfaces via `aria-disabled` and a tooltip
  ("Approve the assumptions above first"). The screen-reader announcement
  when the last card resolves is "All assumptions approved; ready to run."
- Update copy on Turn 2b to make the gesture explicit: section heading
  becomes "Here is what the composer drafted — review its assumptions"
  rather than today's passive "Here is what the composer drafted."

#### New: `TutorialTurn7Graduation.tsx`

A new tutorial turn appended after Turn 6 ModeChoice.

State-machine additions in `tutorialMachine.ts`:

- New `TutorialStep`: `"graduation"`
- New `TutorialAction`: `{ type: "finishMode" }`
- `tutorialReducer` handles `"finishMode"` by transitioning from
  `"mode"` to `"graduation"`. (The reducer no longer marks tutorial
  complete on the existing `"saveMode"` action; that now only persists
  the mode preference.)
- `previousStep("graduation")` returns `"mode"`

Page content:

```
Kicker: Graduation

Heading: You're ready to use the composer.

Body (four bullets):

  1. What you built is AI-generated.
     The pipeline you just ran was authored by an LLM that
     interpreted your one-sentence description. Each assumption
     it made — the URL list it invented, the definition of
     "primary colour", the prompt it wrote for itself — is in
     the audit trail with your approval against it. You can come
     back to it from the Audit page at any time.

  2. Read before you run.
     From this point on, when the composer drafts a pipeline for
     you in normal use, glance at the graph and the YAML before
     clicking Run. If anything looks wrong, amend or reject — the
     same gestures you just practised.

  3. Ask Elspeth.
     If anything in a pipeline (a plugin name, a transform's
     effect, a recorded assumption) doesn't make sense, ask in the
     chat panel. The composer can explain the pipeline it just
     built, in plain English, against the actual node options.

  4. LLMs are confident even when they're wrong.
     The composer LLM may invent URLs that look real but aren't,
     or write a prompt that misframes your question. The pipeline
     LLMs you build will treat fetched HTML as instructions even
     when it shouldn't be. Before sharing or acting on a pipeline's
     output, verify the sources are who they claim to be and check
     the output matches what you actually asked for.

CTA: Take me to the composer.
```

Bullet 1 wording note (revised from rev 0): the rev-0 bullet ended with
"that approval is yours" — UX review flagged that as confrontational
for users who clicked through unfamiliar review cards without fully
understanding what they were approving. The revised wording leads with
the factual audit claim ("each assumption is in the audit trail"),
makes the approval invitation explicit ("you can come back to it"), and
defers the ownership framing to bullet 2 ("read before you run") where
it sits as practice guidance rather than a closing accusation.

Bullet 4 is new in rev 1. The canonical tutorial pipeline scrapes HTML
and feeds it to an LLM transform — prompt-injection-from-scraped-content
is directly in scope and the graduation page would be silent on it
without this bullet.

The CTA's click handler PATCHes
`/api/composer-preferences` setting `tutorial_completed_at`, then
navigates to the empty post-tutorial session that
`cleanup_tutorial_orphans` will have created.

Behaviour:

- `<h2>` focused on mount (matches existing tutorial-turn pattern).
- Four bullets are a single `<ul>` for screen-reader clarity (each `<li>`
  contains an inline-strong `<strong>` for the bullet heading and a
  following sentence for the body).
- Single CTA is the only focusable element below the heading.
- No secondary action ("skip", "remind me later"). The page earns
  one click.
- One new telemetry event: `tutorial_graduation_shown` fires on mount.

#### Tutorial-completion semantics moved from Turn 6 to Turn 7

Today, Turn 6's "Save and go" PATCH sets both `default_mode` and
`tutorial_completed_at` in a single request. After this change:

- **Turn 6's PATCH** sets only `default_mode`. The user's mode preference
  persists immediately.
- **Turn 7's CTA** sets `tutorial_completed_at`.
- If the user closes the browser between Turn 6 and Turn 7, they see
  the tutorial again next session (preference saved, completion not
  recorded). This is the intended behaviour: graduation IS completion.

**Downstream readers of `tutorial_completed_at` (enumerated for rev 1):**

The previous-coupled-states invariant (mode-set ↔ tutorial-complete) is
broken by this change. Every site reading `tutorial_completed_at` was
audited; the new mode-saved-but-not-graduated state is safe for each:

| Reader | Behaviour with the new in-between state |
|---|---|
| `src/elspeth/web/composer/tutorial_service.py:115` (`build_tutorial_session_context`) | Reads `prefs.tutorial_completed_at is not None` to decide whether the user has finished the tutorial. Falsy → tutorial entry re-offered. Safe: mode-saved-but-not-graduated correctly re-offers the tutorial. |
| `src/elspeth/web/composer/tutorial_service.py:886` (orphan cleanup gate) | Same check; same safe behaviour. The user's empty post-tutorial session was created on Turn 6 sign-in already; orphan cleanup runs idempotently and doesn't depend on the completion timestamp for its work. |
| `src/elspeth/web/preferences/service.py:280-379` (`update_preferences` resolution path) | Already handles each field independently via `model_fields_set`; the symmetric handling for `default_mode` and `tutorial_completed_at` is documented at `:370`. No change. |
| `src/elspeth/web/app.py:485,669` (validation/log call sites) | Logging only; reads tolerate `None`. No change. |
| `src/elspeth/web/frontend/src/stores/preferencesStore.ts` (and tests at `preferencesStore.test.ts`) | Frontend gates on `tutorial_completed_at == null` to mount the tutorial flow. Mode-saved-but-not-graduated correctly re-enters the tutorial at the saved mode preference's first turn. |
| `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts:88-106` | Asserts the PATCH body shape. Tests must split: one asserting Turn 6 PATCH includes only `default_mode`; one asserting Turn 7 PATCH includes only `tutorial_completed_at`. |

The change is therefore safe but **requires the e2e fixture update**
listed under **Test impact** below. Audited 2026-05-23 via `grep -rn 'tutorial_completed_at' src/elspeth/`.

### Edge cases

| Scenario | Behaviour |
|---|---|
| User rejects every assumption on Turn 2b | "Looks good, run it" stays disabled. Chat panel remains usable; user re-asks composer. On the composer's next turn the rejected assumptions are superseded; new ones surface. |
| User abandons mid-assumption-review (tab close) | Existing orphan-cleanup catches the `hello-world (…)` session on next tutorial entry; pending interpretation events go to `ABANDONED` per existing contract. No new code. |
| Cache hit path (`_replay_cache_entry`) | Cached pipelines were seeded from a run whose assumptions were all resolved. Replay attaches the resolved state; no pending interpretations exist. Turn 2b renders the empty-assumption muted-text path. |
| `interpretation_review_disabled=true` set on the tutorial session | Crash loud in `buildTutorialDraft` (offensive-programming). Tutorial should never see this flag set. |
| Multiple LLM transforms in user's freeform prompt (not the canonical one) | Turn 2b lists every pending interpretation in order. User clicks through each. No upper bound. |
| Skill change lands but composer model silently omits a Class 2 or Class 3 surface | **Class 3** is caught by preflight defense-in-depth (the LLM-node carries a non-empty `prompt_template` with no resolved Class 3 requirement; preflight raises `INTERPRETATION_REVIEW_MISSING`). **Class 2** is caught by preflight when the source-payload provenance flag indicates LLM-authored (see **Preflight gate per class** open question — provenance flag added in implementation plan task 1). If the provenance flag is missed for any reason, telemetry secondary alert fires: count surfaced Class 2 assumptions in tutorial runs; alert on zero for N consecutive tutorial runs as an operator follow-up. |
| Cache-replay run that pre-dates the skill change | `composer_skill_hash` is in the cache key. Skill edit → cache miss → live run path → new assumptions surfaced. Stale entries unreachable. |

### Cache invalidation

- **Automatic at the key level.** Both the prompt change (cache key
  prefix at `src/elspeth/web/preferences/tutorial_cache.py:47`) and the
  skill change invalidate cached entries. The skill-change chain:
  `tutorial_model_id()` at `src/elspeth/web/composer/tutorial_service.py:831`
  loads the skill content + hash via `load_skill_with_hash("pipeline_composer")`,
  folds the hash into `model_id`, which feeds `_compute_key`. A skill
  content change → new hash → new `model_id` → new cache key → cache miss
  → live run path → new assumption surfaces. Verified 2026-05-23.
- **Manual cleanup recommended on staging.** Operator runs
  `rm -rf {data_dir}/tutorial_cache/` once after deploy to evict the
  unreachable old entries from disk. Not required for correctness —
  orphan entries are harmless — but keeps the cache clean. Same
  "operator deletes the artifact" pattern as the DB-migration memory.

### Test impact

**Updated tests (no logic change, only fixture / string updates):**

- `tests/integration/web/test_tutorial_routes.py`
- `tests/unit/web/test_app.py`
- `tests/unit/web/preferences/test_tutorial_cache.py`
- `tests/unit/web/composer/test_tutorial_service.py` (verify
  `_normalise_bare_required_field_templates` still works for the new
  prompt's `llm` node)
- `tests/unit/web/composer/test_tools.py:58`
- `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`
  — drop the `acceptCompositionProposal` mock that asserts auto-accept;
  assert tutorial does **not** opt out of interpretations
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx`
  — assert `buildTutorialDraft` no longer calls
  `optOutOfInterpretations` or `resolveTutorialInterpretations`
- `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts` — update
  happy-path fixture to include pending `user_approved` interpretation
  events (URL list + primary colour + prompt template); add Turn 2b
  click-through assertion; add Turn 7 graduation assertion; update
  assertions on row contents
- `src/elspeth/web/frontend/src/test/inlineSourceIntegration.test.tsx:511`
- `src/elspeth/web/frontend/src/test/interpretationIntegration.test.tsx`
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx:274,305`
- `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx:105`
  (anchoring comment)

**New tests:**

- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx`
  — renders four bullets (rev 1: includes the new hallucination/injection
  bullet); single CTA fires `tutorial_completed_at` PATCH; telemetry
  `tutorial_graduation_shown` emits on mount
- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`
  — extend with `"finishMode"` action / `"graduation"` step;
  `previousStep("graduation")` returns `"mode"`

**New tests for rev-1 contract work:**

- `tests/unit/contracts/test_composer_interpretation.py` — add
  `InterpretationKind` enum closed-set assertion; assert
  `INTERPRETATION_HASH_DOMAIN_V1` is retired and `..._V2` is the only
  active domain; assert every emitted `InterpretationEventRecord`
  carries `hash_domain_version="v2"`.
- `tests/unit/web/test_interpretation_state.py` — add `kind`-aware
  `interpretation_sites()` cases for `INVENTED_SOURCE` (source node)
  and `LLM_PROMPT_TEMPLATE` (llm node); assert the LLM-only gate is
  removed; assert `_materialize_node_for_execution` handles
  `INVENTED_SOURCE` as record-and-pass with drift detection.
- `tests/integration/web/test_preflight_per_class.py` (new file) —
  assert preflight refuses source-node execution when an unresolved
  `INVENTED_SOURCE` requirement exists, refuses llm-transform
  execution when an unresolved `LLM_PROMPT_TEMPLATE` requirement
  exists, and defends-in-depth by raising `INTERPRETATION_REVIEW_MISSING`
  when an `llm` node carries a non-empty `prompt_template` with no
  resolved Class 3 requirement.
- `tests/integration/web/test_interpretation_opt_out_audit.py` (new
  file) — assert `interpretation_review_disabled=true` sessions still
  emit one `AUTO_INTERPRETED_OPT_OUT` row per Class 1/2/3 surface, each
  carrying the appropriate `kind` and `accepted_value`.
- `src/elspeth/web/frontend/src/components/InterpretationReviewTurn.test.tsx`
  — extend with `kind`-aware rendering assertions per the
  **Differentiated review surface** spec (per-kind heading; `<pre>`-scroll
  gate for `LLM_PROMPT_TEMPLATE`; `showOptOut={false}` hides the opt-out
  button; `showAmend={false}` hides the amend entry).
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx`
  — extend with multi-card render assertions (N > 1 shows progress
  header; focus management transitions card-to-card on resolve;
  simultaneous render not sub-wizard).

**Skill prose is not unit-tested.** Per the existing
`feedback_no_tests_for_skill_prompts` memory ("skills are LLM prompts,
not code; grepping skill text is theatre"). Verification of the skill
amendment happens via a manual tutorial run against staging — the
composer either surfaces all three classes or it doesn't.

### Memory updates landing with the change

- `project_composer_canonical_test_case` — replace old prompt with new
  prompt; note "primary colour" as the demoable assumption.
- New: `project_composer_assumption_surfacing_policy` — captures the
  three-class always-surface policy as a load-bearing project
  constraint, so future agents don't reintroduce auto-opt-out. **Must
  include the opt-out audit invariant** (rev 1): opt-out skips the
  human, not the audit; `AUTO_INTERPRETED_OPT_OUT` row is still emitted
  for every Class 1/2/3 surface when `interpretation_review_disabled=true`.
- New: `project_tutorial_turn7_graduation` — records that Turn 7
  marks `tutorial_completed_at`, not Turn 6.
- New: `project_interpretation_kind_contract` (rev 1) — captures the
  `InterpretationKind` enum and its hash-domain bump
  (`INTERPRETATION_HASH_DOMAIN_V1` → `..._V2`), so future contract
  changes know the discriminator already exists.

### Deferred follow-ups

Explicit follow-up backlog from the rev-1 reviewer pass. Each item was
considered for inclusion and deferred with reason. Each should be filed
as a Filigree issue when implementation lands so the audit trail
captures "we considered this, deferred deliberately, here is why."

1. **Approach C2: interactive in-UI editing of invented source data and
   prompt templates.** Currently rejection routes through chat-based
   revision. C2 would let the user edit the URL list inline or revise
   the prompt template in a textarea, then re-resolve the requirement.
   Deferred reason: contract-rippling change (a structured `draft`
   schema, a new materialisation path that mutates source content, a
   validation rule for edited URLs/templates). Carves cleanly into its
   own spec.

2. **First-run gate (systems-thinker's Meadows-L10 recommendation).**
   The Turn 7 graduation page is information (Meadows L6). A
   higher-leverage move is a one-shot mechanical gate: the user's first
   post-tutorial pipeline either must be hand-edited before its first
   execute OR must pass through an explicit "I have read this
   AI-generated content, run it anyway" affordance distinct from the
   normal Run button. Decays after first use. Deferred reason: scope —
   this changes composer behaviour outside the tutorial flow, and
   "first post-tutorial pipeline" is a new state that needs schema and
   UI work. The tutorial spec ships lean; the gate ships as a focused
   follow-up that builds on the graduation page's framing.

3. **Asymmetric opt-out friction (systems-thinker Concern 2).** Flipping
   `interpretation_review_disabled=true` via the per-session opt-out
   button should require a justification string written to the audit
   trail and visible in the readiness panel — friction proportional to
   consequence. Headless harnesses that set the flag at session
   creation are unaffected. Deferred reason: cross-cuts the readiness
   panel and the existing `composer-preferences` PATCH contract; better
   carved as its own spec once the rev-1 contract changes have shipped.

4. **Time-bounded opt-out (systems-thinker Concern 2).** The opt-out
   flag auto-expires (e.g. 30 days) using the same lifecycle as the
   CICD allowlist-expiry pattern. Forces reaffirmation. Deferred reason:
   same as item 3 — cross-cutting and best handled with item 3.

5. **Audit-readiness panel surface for the new audit-only rows.** When
   `interpretation_review_disabled=true` and the LLM authored a Class 2
   or Class 3 artefact, the audit row exists but has no UI presence in
   the readiness panel today. A readiness panel surface would let
   reviewers see "this run used opt-out and the LLM authored these
   three artefacts that you never reviewed." Deferred reason: out of
   scope for the tutorial-fix slice; the audit-row data is there to
   build on later.

6. **Eval-harness verification that the opt-out audit invariant holds.**
   When `interpretation_review_disabled=true` is set programmatically
   by an eval harness (not via the per-session opt-out button), the
   audit-row emission should still fire. An eval-harness regression
   test asserts this. Deferred reason: depends on the audit-row
   emission landing first.

Per the `feedback_no_unilateral_deferral` memory, each deferral above
names a concrete reason and a sensible carve. None are "I find this
complicated and would rather not."

### Planning checkpoints

Rev 0 listed three open verifications. Rev 1 closes all three:

1. **`InterpretationReviewTurn` renders for non-LLM-transform requirements.**
   **CLOSED — specified in Contract changes (L0 + L3) + Differentiated
   review surface above.** The component gains a `kind` prop and
   per-kind branches; the LLM-only gate at `interpretation_state.py:78-79`
   is removed; per-class chrome and friction are specified. No longer an
   open question for the implementation plan.
2. **`web_scrape` plugin supports `abuse_contact` and `scraping_reason`
   options.** **CLOSED affirmatively.** Verified at
   `src/elspeth/plugins/transforms/web_scrape.py:82,86` — both are required
   Pydantic fields, validated by `@field_validator` at `:100`. The
   canonical prompt's values (`noreply@dta.gov.au`,
   `'DTA technical demonstration'`) pass the wire-visible-placeholder
   check in `src/elspeth/contracts/wire_visible_identity.py:41`
   (no angle brackets, no sentinel keywords). No plugin-side work.
3. **`composer-progress` polling headline strategy.** **SPECIFIED.** The
   composer may emit multiple `request_interpretation_review` calls per
   build. The progress poller renders the first pending requirement's
   `kind`-aware headline:
   - `VAGUE_TERM`: "Drafting an interpretation of *{user_term}*…"
   - `INVENTED_SOURCE`: "Drafting source data for review…"
   - `LLM_PROMPT_TEMPLATE`: "Authoring the prompt for the `{node_id}` transform…"
   When more than one is pending: append " (N more queued)". The poller
   already runs once per second; surface order matches the order
   `request_interpretation_review` was called. No new endpoint needed.

(No open checkpoints remain; the implementation plan should treat this
spec as fully scoped.)

## Migration

ELSPETH's No Legacy Code Policy applies. The old `optOutOfInterpretations`
and `resolveTutorialInterpretations` calls are deleted, not
feature-flagged. The old canonical prompt strings are replaced, not
retained alongside. The tutorial cache invalidates automatically; the
operator deletes stale on-disk entries once after deploy.

No database schema migration. The interpretation events / tutorial
cache / composer-preferences tables are unchanged.

## References

Tutorial-side (rev 0 baseline):

- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx`
- `src/elspeth/web/composer/skills/pipeline_composer.md`
- `src/elspeth/web/composer/tutorial_service.py`

Contract-side and runtime gate (rev 1 additions):

- `src/elspeth/contracts/composer_interpretation.py` — `InterpretationKind`
  enum lives here; hash-domain bump applies here.
- `src/elspeth/web/interpretation_state.py:78-79` — LLM-only gate to
  remove; per-kind `_materialize_node_for_execution` branches added here.
- `src/elspeth/web/sessions/service.py` — execution preflight; per-class
  refusal logic.
- `src/elspeth/web/frontend/src/components/InterpretationReviewTurn.tsx`
  — `kind` prop, per-kind chrome, `showOptOut` / `showAmend` props.
- `src/elspeth/web/preferences/service.py:280-379` — `tutorial_completed_at`
  resolution path (already field-independent; no change but cited for
  audit).
- `src/elspeth/plugins/transforms/web_scrape.py:82,86,100` — verified
  `abuse_contact` / `scraping_reason` Pydantic fields and validator.
- `src/elspeth/contracts/wire_visible_identity.py:41` — placeholder check
  the canonical-prompt values pass.

Origin:

- Brainstorming transcript: this design was derived through the
  superpowers:brainstorming flow on 2026-05-22.
- Multi-reviewer pass on 2026-05-23: five reviewers (UX-critic,
  solution-design-reviewer, systems-thinking leverage-analyst,
  python-code-reviewer, LLM-safety-reviewer) produced the findings
  absorbed into rev 1. See conversation transcript for the full review
  output.
