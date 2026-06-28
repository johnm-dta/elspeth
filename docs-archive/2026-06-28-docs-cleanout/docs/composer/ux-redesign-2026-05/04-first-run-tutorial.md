# 04 — First-Run Tutorial (Hello World)

## Purpose

A short, forced sequential tutorial that runs on a user's very first session.
It accomplishes four things that nothing else in the UI does as well:

1. **Teaches the three-layer mental model** (source / transform / sink) by
   building a real pipeline.
2. **Introduces the dynamic-source-from-chat feature** so users learn the
   composer's lowest-friction input mode immediately.
3. **Demonstrates the audit trail** against the user's own pipeline,
   establishing ELSPETH's defining feature at moment of first contact.
4. **Resolves the default-mode preference** with an informed choice, after
   the user has seen what the system does.

## Why a tutorial

The composer has two reasonable mode-defaults (guided and freeform) and the
right default depends on the user. Forcing a preference choice before the
user has composed anything produces uninformed answers. The tutorial
inverts the order: build first, see the audit trail, then pick a default.

A side benefit: the tutorial doubles as **vocabulary teaching**. Without it,
guided mode has to teach "what's a source?" at every step of every session;
freeform mode assumes the user knows what to ask for. The tutorial teaches
the vocabulary once, in a context where it's grounded in a working artifact.

## The canonical seed prompt

The tutorial uses the operator's canonical composer test case as the
suggested input for turn 2:

> *"create a list of 5 government web pages and use an LLM to rate how
> cool they are"*

This prompt has been validated to work end-to-end on the live composer.
See `project_composer_canonical_test_case` in memory.

The user can edit the prompt or substitute their own; the canonical text is
**pre-populated in the input** so the path of least resistance is "submit
as-is."

## Turn-by-turn arc

Target: ~3 minutes for a quick reader; ~5 minutes for someone who explores.

### Turn 1 — Welcome and frame

```
Welcome to ELSPETH.

In about 3 minutes we'll build and run your first pipeline together. Then
you'll choose how you want to work going forward.

Pipelines have three layers:
  • SENSE — where data comes from (a source)
  • DECIDE — what happens to each row (transforms)
  • ACT — where results go (sinks)

We'll build one of each. Ready?

  [ Let's go → ]
```

The single forward button is intentional. There's no "skip tutorial" — the
tutorial **is** the first session, not an interruption to it. (The tutorial
*can* be retaken later from the settings menu; see Open Questions in
[11-open-questions.md](11-open-questions.md) about how that interacts with
existing sessions.)

### Turn 2 — Describe your pipeline

```
You don't have to build the three layers one at a time. For your first
pipeline, describe what you want in one sentence and I'll wire it up.

Try this (pre-filled — edit if you like):

  ┌────────────────────────────────────────────────────────┐
  │ create a list of 5 government web pages and use an LLM │
  │ to rate how cool they are                               │
  └────────────────────────────────────────────────────────┘

  [ Build it ]
```

System submits the prompt to the composer, which generates a pipeline draft
with a 5-row source (LLM-selected URLs), a `web_scrape` transform, an `llm`
rate transform, and a JSONL sink.

### Turn 2b — Show what was built + surface interpretation

```
Got it. Here's what I drafted:

  SOURCE — 5 government web pages I selected:
    • australia.gov.au
    • finance.gov.au
    • ato.gov.au
    • data.gov.au
    • dta.gov.au
  (these are my pick — you can edit the list)

  TRANSFORM — fetch each page (web_scrape), then call an LLM to rate
  each one for "coolness"

  SINK — write the ratings to a JSONL file in your session

Before we run: when you said "cool", I read that as roughly
"modern design + clear purpose + interactivity". Want to adjust the
definition, or use mine?

  [ Use my interpretation ]   [ Change it: I meant... ]
```

The "Change it" path is important — it teaches the **surface-the-LLM's-
interpretation** affordance from
[06-chat-as-data-entry.md](06-chat-as-data-entry.md). If the user clicks it,
they get a free-text editor pre-filled with the LLM's draft interpretation;
their edited version replaces it.

### Turn 3 — A glance at the graph

```
Here's your pipeline as a graph. Three layers, four steps:

  ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
  │ url_source │ →  │ web_scrape │ →  │  llm_rate  │ →  │ jsonl_sink │
  └────────────┘    └────────────┘    └────────────┘    └────────────┘
        5 rows           fetch              rate           write

Look familiar? That's the source → transform → sink shape we talked
about, just with two transforms instead of one.

  [ Looks good, run it → ]
```

This turn is short on purpose. It introduces the graph view (which will be
persistent in the side rail after the tutorial) without belabouring it.

### Turn 4 — Run

```
Running your pipeline...

  ▓▓▓▓▓░░░░░  fetching pages          [2 of 5 done]
  ░░░░░░░░░░  rating with LLM
  ░░░░░░░░░░  writing output

  ▓▓▓▓▓▓▓▓▓▓  done — 5 rows in, 5 rows out

Here's what came back:

  URL              | Coolness | LLM rationale
  ─────────────────|──────────|──────────────────────────────────
  australia.gov.au |    6     | Clear hierarchy but dated...
  finance.gov.au   |    7     | Functional layout, useful tools...
  ato.gov.au       |    5     | Information-dense, navigation rough...
  data.gov.au      |    8     | Modern, interactive, clear purpose...
  dta.gov.au       |    9     | Bold design, strong narrative...

  [ Continue → ]
```

The result is the actual run output rendered as a table. No JSONL preview;
the user does not need to learn what JSONL is during the tutorial.

### Turn 5 — The audit story (THE LOAD-BEARING TURN)

```
Notice something? The LLM made a judgment call on every page — and you can
see WHY. That's not because we logged its output. It's because ELSPETH
records the full lineage as evidence:

  ✓ Every URL you started with — hash a7f3e2…
    (if you ran this again with different URLs, the trail would show
     different hashes; same URLs, same hashes)

  ✓ Your accepted definition of "cool" — recorded as a prompt template
    (if a colleague disagrees with the rating, they can see exactly
     what definition the LLM was applying)

  ✓ Every LLM call — full prompt, full response, model, version, timestamp

  ✓ The output file — SHA-256-hashed, chain-of-custody recorded

  ✓ The run itself — when, who ran it, plugin versions in use

If someone six months from now asks "why did dta.gov.au score 9?" — the
audit trail can answer. With the prompt. With the page content the LLM
saw. With the model and version.

That's what ELSPETH was built for: AI decisions you can defend.

  [ Explore the full audit trail ]   [ Continue → ]
```

The "Explore the full audit trail" link opens the same audit-evidence view
the user will see when they click "Explain" on the audit-readiness panel in
future sessions. Continuity established.

### Turn 6 — The mode choice

```
You've built and run your first pipeline. Going forward, there are two
ways to compose:

  ● GUIDED — same step-by-step flow you just did. Recommended.
    Best when you're learning what's possible, or when you want a
    clear path through validation and audit checks.

  ○ FREEFORM — describe what you want in chat, I'll build it.
    Best for power users who know exactly what they need.

  You can switch any time from the chat panel. What should new sessions
  default to?

  [ Guided (recommended) ]   [ Freeform ]
```

The selected option writes the `composer.default_mode` user preference. The
tutorial session ends; the user lands on the regular composer in their
selected mode.

## What the tutorial deliberately avoids

| Avoided | Why |
|---|---|
| File upload of any kind | Dynamic-source-from-chat is the lowest-friction path; introducing CSV upload here distracts from the demo. |
| Plugin browsing / catalog | No vocabulary yet to use a catalog. Catalog is introduced in the user's first real session. |
| Multi-step source / transform / sink wizards | The user describes the whole pipeline at once; the LLM does the wiring. Multi-step is what guided mode is for, after the tutorial. |
| Gates, forks, coalesce | Advanced graph concepts beyond hello-world. |
| Validation errors | The seed prompt is curated to produce a clean pipeline. If a real user-edited prompt produces validation errors, surface them — but the canonical path is clean. |
| YAML view | YAML is a power-user surface; not needed for the mental model the tutorial is teaching. |
| Authentication or org concepts | The user logged in to get here; the tutorial is about the composer, not the account. |

## What the tutorial promises and must deliver

| Promise | Implementation requirement |
|---|---|
| "I'll wire it up from your one sentence." | The composer must produce a working pipeline from the canonical seed. Currently does. |
| "Here's what I drafted." | The system must show the user the drafted pipeline structure before running, with editable elements (URL list, interpretation). |
| "Your accepted definition of 'cool' — recorded as a prompt template." | The user's acceptance of the LLM's interpretation must be recorded in the audit trail as a distinct event, not silently baked into the prompt. This is the "surface-the-LLM's-interpretation" feature from [06-chat-as-data-entry.md](06-chat-as-data-entry.md). |
| "Hash a7f3e2… of your URLs." | The audit recorder treats dynamic-source-from-chat content identically to file-source content for provenance. |
| "If someone six months from now asks..." | The Landscape audit trail can answer. (Already true; the tutorial makes it visible.) |

## What happens after the tutorial

The user lands on a fresh empty session in their chosen default mode. The
tutorial pipeline is preserved as a session in their history, named
something like "hello-world (cool government pages)" — they can return to it,
edit it, or run it again. It is a real artifact, not a throwaway.

The tutorial is **not** re-shown on subsequent sessions. The user can opt to
revisit it from the settings menu.

## Implementation notes

- The tutorial is **frontend-only behaviour over existing backend APIs.**
  No new backend endpoints are required if the audit recorder already
  records dynamic-source-from-chat content the same way as other sources
  (verify; flag in [11-open-questions.md](11-open-questions.md) if not).
- The "Change it" interpretation editor requires the backend to accept an
  edited prompt-template override and record that override in the audit
  trail. See [06-chat-as-data-entry.md](06-chat-as-data-entry.md) for the
  feature spec; flag in [11-open-questions.md](11-open-questions.md) if
  the backend doesn't yet support this.
- The audit-trail explanation in turn 5 should pull its hash values from
  the actual run, not from canned text. Otherwise the demonstration is
  theatre.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| User edits the seed prompt to something the composer can't handle | Validation falls back gracefully; show the validation error in the audit-readiness style ("here's what couldn't be built and why") and offer to restore the canonical seed. |
| LLM picks URLs that don't load | The web_scrape transform's error handling is already audit-aware; show the failure on the result table ("page 3: HTTP 503 — recorded in audit") and treat it as a teaching moment about robust pipelines. |
| Run takes longer than ~30 seconds (LLM call x 5) | Show progress, but also: this is a real LLM cost. Consider caching the canonical run's results for first-tutorial seed-prompt cases. See [11-open-questions.md](11-open-questions.md). |
| User wants to skip | Provide a small "I've used ELSPETH before" link in turn 1 that fast-forwards to turn 6 (mode-default choice) without the build/run. The vocabulary teaching is then lost; that's acceptable for returning users. |

## Memory references

- `project_composer_first_run_tutorial`
- `project_composer_canonical_test_case`
- `project_composer_dynamic_source_from_chat`
