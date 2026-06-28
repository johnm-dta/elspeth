# 06 — Chat as Data Entry

This document covers two related features that turn the composer's chat
input from a chat affordance into a first-class data and intent capture
surface:

1. **Dynamic-source-from-chat** — typing data or a description directly into
   the chat creates a source plugin without file upload or schema setup.
2. **Surface-the-LLM's-interpretation** — when the LLM interprets subjective
   user terms (e.g., "cool"), the interpretation is presented for explicit
   user acceptance or amendment before the pipeline runs.

The two features compose. The user types "rate these for coolness" — the
composer creates the source AND surfaces the interpretation of "coolness"
as a reviewable artifact in one flow.

## Feature 1 — Dynamic-source-from-chat

### Concept

For simple/single-record/short-list inputs, the user can type the data
directly into the composer chat. The LLM creates a dynamic source
containing those rows. No file picker, no schema configuration, no
upload — the chat text **is** the source.

### Canonical examples

| User types | Composer creates |
|---|---|
| `"go to www.finance.gov.au"` | 1-row source with that URL |
| `"check these URLs: a.com, b.com, c.com"` | 3-row source (one URL per row), with explicit confirmation of the row interpretation |
| `"create a list of 5 government web pages and use an LLM to rate how cool they are"` | 5-row source — the LLM selects which URLs qualify as government web pages |
| `"this transaction: $4,200, payee 'Acme Corp', date 2026-04-15"` | 1-row source with the structured fields parsed out |

### Why it matters

- **Removes the single biggest dropoff in pipeline tools**: "I need to
  format my data as a CSV before I can start." For most ad-hoc work, the
  user has one URL, one record, or a short list — file upload is overkill.
- **Aligns the chat metaphor with reality**: the user's prompt *is* the
  data. They don't have to translate intent into a file format the system
  understands.
- **Lowers the entry cost for hello-world**: the tutorial in
  [04-first-run-tutorial.md](04-first-run-tutorial.md) uses this feature
  in turn 2 to get the user to a working pipeline without uploading
  anything.
- **Serves Marcus (URL trigger) and Linda (single record) naturally** —
  see the persona analysis in
  [02-personas-and-audiences.md](02-personas-and-audiences.md).

### Modes of operation

| Mode | User input | Composer behaviour |
|---|---|---|
| **Verbatim transcription** | User types data literally (a URL, a sentence, a record) | Composer creates a source containing exactly what the user typed |
| **LLM-generated rows** | User describes data ("5 government pages") | Composer asks the LLM to generate the source rows, presents them for confirmation, and creates the source from the confirmed list |
| **Mixed / inferred** | User types something ambiguous ("check these URLs: a.com, b.com") | Composer makes a row-count call, presents its interpretation ("I'm treating this as 3 rows — change?"), and creates the source after confirmation |

### Audit treatment

The user's chat message that produced the source content **becomes part
of the audit chain**. Specifically:

- **Verbatim transcription:** the source rows hash to whatever the user
  typed; the chat message itself is recorded as the source's provenance
  ("source content provided by user chat message at [timestamp]").
- **LLM-generated rows:** the source rows are recorded, AND the LLM call
  that produced them is recorded (prompt, response, model, version). An
  auditor asking "why these 5 URLs?" gets a complete answer.
- **Mixed / inferred:** the row-count interpretation is recorded as an
  explicit user decision ("user confirmed 3-row interpretation").

This is critical. Without it, a dynamic source looks like "rows just
appeared from somewhere" — and the auditability story breaks. The
implementation must verify the recorder treats these cases correctly;
see [11-open-questions.md](11-open-questions.md).

### Chat input placeholder

The empty-state chat input should prime the user that dynamic-source-from-chat
is an option:

> Describe your pipeline, or paste a URL or sample data to start...

For an existing session with an active composition, the placeholder reverts
to the standard "Type your message..." since the data step is already done.

### Surface in the Catalog reference

The Catalog (see [08-catalog-reshape.md](08-catalog-reshape.md)) should
list "**Inline data from chat**" as the *first* source option, framed as
the lowest-friction starting point. Subsequent entries (CSV, API,
database, etc.) are for users with batch-sized inputs.

### Disambiguation thresholds

A reasonable cutoff to push the user toward file-source: somewhere around
10-20 rows of typed data. Past that point, the LLM is doing structure
extraction at scale and the user is likely better served by a CSV upload.
The threshold is a product question; see
[11-open-questions.md](11-open-questions.md).

When the threshold is crossed, the composer offers:

> That's a lot of rows to type into chat. Would you like to paste it as
> CSV instead, or keep going inline?

Not blocking — just gentle redirection.

## Feature 2 — Surface the LLM's interpretation

### Concept

When the LLM interprets subjective or ambiguous user terms during pipeline
composition, the interpretation is presented to the user for explicit
acceptance or amendment **before the pipeline runs**. The user's accepted
interpretation is recorded in the audit trail as a discrete decision.

### Canonical example (from the tutorial)

User says: *"rate how cool they are"*

Without this feature: the LLM silently picks an interpretation of "cool",
bakes it into the prompt template, and the pipeline runs. The audit trail
contains the prompt but not the user's review of it.

With this feature:

```text
  Before we run: when you said "cool", I read that as roughly
  "modern design + clear purpose + interactivity". Want to adjust the
  definition, or use mine?

  [ Use my interpretation ]   [ Change it: I meant... ]
```

The user clicks one. If they amend, an inline editor opens pre-filled with
the LLM's draft. Their edited version replaces it. Either choice is
recorded.

### Why it matters

This is the small-but-load-bearing affordance that turns opaque AI
decisions into accountable AI decisions. Without it:

- The audit trail shows what the LLM did, but not whether a human
  reviewed it.
- An auditor asking "why is the prompt template what it is?" gets the
  answer "the LLM wrote it."

With it:

- The audit trail shows the LLM's draft, the user's accepted version,
  and the user's identity attached to the acceptance event.
- An auditor asking the same question gets "the user, having reviewed
  the LLM's suggested interpretation, accepted/amended it on [date]."

That second answer is dramatically stronger. It's the difference between
"the AI did it" and "the human approved what the AI did." For Linda's
persona, this is exactly the kind of accountability transformation that
makes ELSPETH usable in her actual job.

### When the interpretation gets surfaced

Not on every LLM-mediated step — that would be exhausting. Surface the
interpretation when the LLM is operationalizing a **subjective or
underspecified user term**. Heuristics:

| Surface | Don't surface |
|---|---|
| Subjective adjective ("cool", "important", "risky") | Concrete operator ("rate as numeric 1-10") |
| User asked for X but provided no definition | User provided their own definition |
| First time a term appears in a composition | Same term reused later in the same session |
| LLM had >1 plausible interpretation | Single clear path |

The composer is doing some inference here. False positives (surfacing when
not needed) are an annoyance; false negatives (failing to surface when the
LLM is making a real subjective call) are an audit hole. Bias toward
false positives at first; tune down with usage data.

### Recording the interpretation

Each interpretation surface event produces an audit record with:

- The user-provided term ("cool")
- The LLM's draft interpretation
- The user's accepted or amended interpretation
- A timestamp and the user's identity
- A reference to the pipeline state the interpretation applies to

The final accepted interpretation is what flows into the prompt template
used by the LLM transform at run time. Subsequent runs against this
pipeline use the recorded interpretation; the LLM is not asked to
re-interpret.

### Editing the interpretation later

The user can revisit and edit a recorded interpretation by selecting the
relevant transform in the graph (or its turn widget in guided mode). The
edit produces a new audit record; the old interpretation is preserved in
history.

### Composer-mediated, not transform-mediated

This feature is about **what gets baked into the pipeline before it runs**,
not about runtime behaviour. At runtime, the LLM transform is just executing
its (now user-approved) prompt template. The interpretation surfacing
happens once, during composition.

This keeps the runtime path simple and deterministic. Multiple runs of the
same pipeline use the same interpretation; the user doesn't have to
re-confirm.

## How the two features compose

The canonical tutorial example exercises both:

1. User types: *"create a list of 5 government web pages and use an LLM to
   rate how cool they are"*
2. **Dynamic-source-from-chat**: the LLM generates 5 URLs and creates the
   source. The composer presents the 5 URLs and lets the user edit the
   list.
3. **Surface-the-LLM's-interpretation**: the LLM reads "cool" as a
   subjective term and offers its draft definition. The user accepts or
   amends.
4. Pipeline runs with the user-approved source content and the
   user-approved interpretation.

Both feature surfaces are explicit and reviewable. The audit trail records
both: the source URLs (with the LLM call that produced them), and the
interpretation (with the user's acceptance).

## Implementation notes

| Component | Touch-point |
|---|---|
| Composer tool surface (backend) | Needs a tool that creates a source from inline content (likely already exists in some form for blob handling). |
| Composer tool surface (backend) | Needs a tool that records an LLM-interpretation acceptance event as a distinct audit record, separate from the prompt template's plugin config. |
| Frontend chat input | New empty-state placeholder text. |
| Frontend turn / proposal UI | New "use mine / change it" pattern for interpretation surfacing. |
| Audit recorder | Verify dynamic-source-from-chat content is treated identically to file-source content for provenance recording. |
| Audit recorder | Verify interpretation-acceptance events are recorded as discrete events linked to the pipeline state. |

## Risks

| Risk | Mitigation |
|---|---|
| Users feel nagged by interpretation prompts | Heuristic threshold; tune with usage data; offer a per-session "stop asking" toggle (records as "user opted out of interpretation review for this session" — still in audit trail). |
| The LLM hallucinates source content | Surface the generated rows for explicit user confirmation; never auto-commit. |
| Audit trail becomes noisy with small acceptance events | Acceptable cost. Audit trails for high-assurance work are supposed to be detailed. |
| Mode discrepancy between guided and freeform | Both modes use the same backend tool surface and the same interpretation-recording event; the surfacing UI may differ (turn widget in guided, inline message in freeform). |

## Memory references

- `project_composer_dynamic_source_from_chat`
- `project_composer_canonical_test_case` (the hero example of both features)
- `project_composer_first_run_tutorial` (where both features are first taught)
