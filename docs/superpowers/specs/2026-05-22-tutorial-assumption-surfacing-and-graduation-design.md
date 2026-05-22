# Tutorial assumption-surfacing and graduation design

**Date:** 2026-05-22
**Status:** approved (operator sign-off; not yet planned)
**Supersedes (partial):** `project_composer_canonical_test_case` (memory entry)

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
   `request_interpretation_review` requirement — audit-only, not
   interactive.
3. **Surface every LLM-authored prompt template.** Each `prompt_template`
   the composer writes for an `llm` transform surfaces for the user to
   approve before the pipeline can run. Applies to all composer
   sessions, not just the tutorial; opt-out continues to be the existing
   `interpretation_review_disabled=true` flag.
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
  brainstorming). Review is audit-only in this slice; rejection forces
  a re-engagement with the composer via chat. C2 is carved off as a
  follow-up.
- Touching the freeform / guided default-mode selection.
- Anything about runtime preflight, audit-readiness panel, or composer
  flow surfaces outside Turn 2b and the new Turn 7.

## Design

### Three classes of assumption

| # | Class | Mechanism | Status today |
|---|---|---|---|
| 1 | Vague-term interpretations ("primary colour", "cool", "important") | `request_interpretation_review` | Already implemented; tutorial currently suppresses it |
| 2 | Invented source data (URL lists the LLM made up) | `request_interpretation_review`, **new pattern** | Currently narrated in chat only; no audit event |
| 3 | LLM prompt templates themselves | `request_interpretation_review`, **new pattern** | Not surfaced at all today |

All three resolve through the same UI surface (`InterpretationReviewTurn`)
and the same backend opt-out flag (`interpretation_review_disabled`).

#### Why audit-only review, not interactive mutation

For class 2 and class 3, approving / amending / rejecting the assumption
**does not** mutate the underlying source or prompt template. The
interpretation event records the user's decision; the audit trail
captures it. To change the underlying content the user re-engages the
composer via chat ("use these URLs instead", "make the prompt more
specific"). This is consistent with how vague-term interpretations work
today (resolving the term doesn't directly rewrite the prompt template
beyond placeholder substitution).

The principled (interactive) alternative — extending the
`InterpretationRequirement` contract with a structured `draft`,
adding a new materialisation path for source content updates, and
teaching `InterpretationReviewTurn` to render and edit a list of URLs —
is deferred. It requires a contract change rippling through L0
(`composer_interpretation.py`), L3 (`interpretation_state.py`), the LLM-
transform materialisation path, and a new validation rule. Bundling
that with the tutorial-prompt change conflates two unrelated risks.

#### Cross-session policy

Classes 2 and 3 are composer-skill changes that affect every composer
session, not just the tutorial. The opt-out lever is the existing
`interpretation_review_disabled=true` flag — same switch that already
governs vague-term reviews. Headless harnesses (eval suites, persona
regression) opt out via that flag exactly as they do today; the tutorial
does not opt out, so the user sees every assumption.

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
> before the pipeline can run (unless
> `interpretation_review_disabled=true`):
>
> 1. **Vague-term interpretations** — when the user prompt contains a
>    subjective or underspecified term ("cool", "important",
>    "primary colour"), surface your definition as you do today (see
>    the existing "Subjective Interpretation Review" section).
> 2. **Invented source data** — if you had to invent source content
>    the user did not provide (URL lists, parsed column values,
>    illustrative records), call `request_interpretation_review`
>    immediately after `set_source` / `set_source_from_blob` succeeds,
>    with `user_term = "inline_source_url_list"` (or similar stable
>    identifier) and `llm_draft = <human-readable rendering>`. Chat
>    narration alone does not satisfy this requirement — it does not
>    create an audit event.
> 3. **LLM prompt templates** — every `prompt_template` you author for
>    an `llm` transform must surface via `request_interpretation_review`
>    with `user_term = "llm_prompt_template:<node_id>"` and `llm_draft = <the raw template text>`.
>    Surface this in addition to any embedded vague-term interpretations;
>    the term review answers "what does this word mean", the template
>    review answers "is this the right way to ask the model". The
>    runtime preflight will block execution until both are resolved.

(The bullet above is the entire replacement; there is no separate
"second skill edit". The chat-narration-only guidance is gone, not
augmented.)

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
  naturally — no code change required to render it.
- Extend the rendering to iterate over **every** pending interpretation
  event for the session, not just the first one (today's `useMemo`
  finds a single event). Each interpretation gets its own review card
  in sequence.
- Disable the "Looks good, run it" / "Show me the graph" CTA while any
  pending interpretation exists. The button label remains the same;
  the disabled state surfaces via `aria-disabled` and a tooltip
  ("Approve the assumptions above first").
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

Body (three bullets):

  1. What you built is AI-generated.
     The pipeline you just ran was authored by an LLM that
     interpreted your one-sentence description. The audit trail
     records that you approved every assumption it made — that
     approval is yours.

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

CTA: Take me to the composer.
```

The CTA's click handler PATCHes
`/api/composer-preferences` setting `tutorial_completed_at`, then
navigates to the empty post-tutorial session that
`cleanup_tutorial_orphans` will have created.

Behaviour:

- `<h2>` focused on mount (matches existing tutorial-turn pattern).
- Three bullets are an `<ul>` for screen-reader clarity.
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

### Edge cases

| Scenario | Behaviour |
|---|---|
| User rejects every assumption on Turn 2b | "Looks good, run it" stays disabled. Chat panel remains usable; user re-asks composer. On the composer's next turn the rejected assumptions are superseded; new ones surface. |
| User abandons mid-assumption-review (tab close) | Existing orphan-cleanup catches the `hello-world (…)` session on next tutorial entry; pending interpretation events go to `ABANDONED` per existing contract. No new code. |
| Cache hit path (`_replay_cache_entry`) | Cached pipelines were seeded from a run whose assumptions were all resolved. Replay attaches the resolved state; no pending interpretations exist. Turn 2b renders the empty-assumption muted-text path. |
| `interpretation_review_disabled=true` set on the tutorial session | Crash loud in `buildTutorialDraft` (offensive-programming). Tutorial should never see this flag set. |
| Multiple LLM transforms in user's freeform prompt (not the canonical one) | Turn 2b lists every pending interpretation in order. User clicks through each. No upper bound. |
| Skill change lands but composer model produces no class-2 surface for invented data | Tutorial still works (one fewer assumption). Skill prose is non-deterministic guidance. Detection: count surfaced assumptions in tutorial telemetry; alert on zero for N consecutive runs as an operator follow-up. |
| Cache-replay run that pre-dates the skill change | `composer_skill_hash` is in the cache key. Skill edit → cache miss → live run path → new assumptions surfaced. Stale entries unreachable. |

### Cache invalidation

- **Automatic at the key level.** Both the prompt change (cache key
  prefix) and the skill change (`composer_skill_hash` inside
  `tutorial_model_id`) invalidate cached entries.
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
  — renders three bullets; single CTA fires `tutorial_completed_at`
  PATCH; telemetry `tutorial_graduation_shown` emits on mount
- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`
  — extend with `"finishMode"` action / `"graduation"` step;
  `previousStep("graduation")` returns `"mode"`

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
  constraint, so future agents don't reintroduce auto-opt-out.
- New: `project_tutorial_turn7_graduation` — records that Turn 7
  marks `tutorial_completed_at`, not Turn 6.

### Planning checkpoints (open before implementation)

1. **`InterpretationReviewTurn` renders for non-LLM-transform requirements.**
   Confirm the component doesn't hard-assume the requirement comes from
   an `llm` node. If it does, a small UI change is required to support
   class-2 (source) and class-3 (prompt template) surfaces. Verify
   during the first implementation task.
2. **`web_scrape` plugin supports `abuse_contact` and `scraping_reason`
   options.** Verify they exist on the registered plugin. If they don't,
   either add them or drop the explicit options from the canonical
   prompt — but the principle (surfacing assumptions for primary-colour
   identification, invented URLs, and the LLM template) is unchanged
   either way.
3. **`composer-progress` polling.** Confirm class-3 prompt-template
   surfacing doesn't make the build look stuck. The composer may emit
   multiple `request_interpretation_review` calls in a row; the progress
   poller should report a sensible headline for each.

## Migration

ELSPETH's No Legacy Code Policy applies. The old `optOutOfInterpretations`
and `resolveTutorialInterpretations` calls are deleted, not
feature-flagged. The old canonical prompt strings are replaced, not
retained alongside. The tutorial cache invalidates automatically; the
operator deletes stale on-disk entries once after deploy.

No database schema migration. The interpretation events / tutorial
cache / composer-preferences tables are unchanged.

## References

- `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`
- `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx`
- `src/elspeth/web/composer/skills/pipeline_composer.md`
- `src/elspeth/web/composer/tutorial_service.py`
- `src/elspeth/contracts/composer_interpretation.py`
- `src/elspeth/web/interpretation_state.py`
- Brainstorming transcript: this design was derived through the
  superpowers:brainstorming flow on 2026-05-22.
