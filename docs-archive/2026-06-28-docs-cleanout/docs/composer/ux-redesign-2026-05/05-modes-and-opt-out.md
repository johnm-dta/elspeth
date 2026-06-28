# 05 — Modes and Opt-Out

## Decision

**New sessions default to guided mode**, with a **persistent per-user
opt-out** reachable both from settings and inline. This supersedes commit
`82dd2e73b`, which set freeform as the default.

## Why the change

The personas analysis in [02-personas-and-audiences.md](02-personas-and-audiences.md)
shows guided benefits Linda (compliance) and Sarah (researcher) — the
primary composer audience per the README's positioning — while freeform
benefits Marcus (marketing ops) and Dev (senior engineer). Freeform-default
optimizes for the minority audience at the cost of the majority audience.

The opt-out resolves the tension. Marcus and Dev opt out once after the
hello-world tutorial; Linda and Sarah never need to think about it.

## Three behaviour scopes

The composer's mode behaviour operates at three distinct scopes. Conflating
them is what produces the "ugh, this guided thing again" feeling — fixing
the conflation is most of the design work.

| Scope | Lifetime | Setter | What it governs |
|---|---|---|---|
| **Account default** | Persistent across sessions, per user | Set during hello-world tutorial; changeable via settings or inline opt-out | What mode *new* sessions start in |
| **Per-session mode** | Lifetime of one composing session | "Switch to guided" / "Exit to freeform" buttons in chat panel header | What this session is doing right now |
| **Per-step within guided** | Within one step of the wizard | "Skip this step" / "I know what I want" affordances in turn widgets | Granular escape within guided |

The account default governs the *initial* mode for new sessions only. Within
a session, the user can switch modes freely without affecting their account
preference.

## State flow diagram

```text
                  ┌───────────────────────────┐
                  │   New user logs in        │
                  │   (no sessions yet)       │
                  └────────────┬──────────────┘
                               │
                               ▼
                  ┌───────────────────────────┐
                  │   Hello-world tutorial    │  ← always; not skippable
                  │   (see doc 04)            │     on first run
                  └────────────┬──────────────┘
                               │
                               ▼
                  ┌───────────────────────────┐
                  │   Mode default choice     │
                  │   [Guided] [Freeform]     │
                  └────────────┬──────────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                              ▼
       account.default = guided      account.default = freeform
                │                              │
                ▼                              ▼
       ┌──────────────────┐          ┌──────────────────┐
       │ New session →    │          │ New session →    │
       │ starts guided    │          │ starts freeform  │
       └────────┬─────────┘          └────────┬─────────┘
                │                              │
                │ user can switch              │ user can switch
                │ to freeform                  │ to guided
                │ for this session             │ for this session
                ▼                              ▼
       (per-session mode = whatever the user picked this time)
```

The account preference can be changed at any time:

```text
       ┌──────────────────┐
       │ Settings →       │
       │ Composer prefs   │
       │ Default mode:    │
       │   ⦿ Guided       │  ← single source of truth
       │   ○ Freeform     │
       └──────────────────┘
                ▲
                │ same preference;
                │ same write
                │
       ┌──────────────────┐
       │ Inline opt-out   │  ← on the guided mode UI;
       │ on guided UI     │     "Always start new sessions in
       │ (small, secondary)│    freeform mode" toggle
       └──────────────────┘
```

## Where the opt-out lives

Three reachable surfaces, all writing to the same `composer.default_mode`
preference:

### 1. Inline on the guided mode UI

A small, secondary affordance somewhere in the guided mode chrome (likely
near the stepper or in a "..." overflow). Wording:

> ☐ Always start new sessions in freeform mode

Catches Marcus and Dev at the moment of friction — they're in guided mode,
they don't want it, the affordance is right there.

### 2. User settings menu

Under "Composer preferences" → "Default mode for new sessions":

```text
  Default mode for new sessions
  ─────────────────────────────
  ⦿ Guided (recommended)
      Step-by-step wizard with validation and audit checks
      surfaced at each decision.

  ○ Freeform
      Describe your pipeline in chat; the composer builds it
      from your description.

  ☐ Show the hello-world tutorial again on next session
```

The "honest" home for the preference. Discoverable by users who want to
change without being in the guided UI.

### 3. Tooltip / context on the "Switch to guided" button

If a user has opted out, the freeform mode page has a "Switch to guided"
button (per the per-session mode toggle). The tooltip on that button
includes a small "(or change your default)" link that opens the settings
menu pre-scrolled to composer preferences. Catches users who want to
re-enable guided as their default after having opted out.

## Discoverability after opt-out

The user said: *"disable; I'll go looking for it if I want it."* That
implies the affordance can't be deeply buried (Linda might disable it
accidentally and need to find her way back), but it also shouldn't be in
chrome (defeats the opt-out).

The plan:

- After opting out, the freeform mode page still shows a **persistent but
  unobtrusive "Switch to guided"** button in the chat panel header. Same
  place as today, smaller / quieter visual weight.
- A **one-time gentle banner** appears on the first session after opting
  out: *"Future sessions will start in freeform mode. You can switch to
  guided anytime from this panel, or re-enable as your default in
  Settings."* Dismissible; never returns.
- The settings menu lives in the user dropdown (top right) and is one
  click away — the place "I'll go looking" naturally arrives at.

## Per-session mode switching (unchanged)

The existing "Switch to guided" and "Exit to freeform" buttons in the chat
panel header continue to work as **session-scoped overrides**. They do not
change the account preference.

If the user's account default is freeform and they switch to guided for
this session, the next new session reverts to freeform. The intent of the
session toggle is "I want to do this differently right now," not "change
my mind permanently."

Switching modes mid-session is a real action — the existing implementation
handles transitions between composition states. The only change here is
ensuring the per-session vs account semantics are clear in the UI.

## Returning user, never opted out

A user who took the hello-world tutorial, picked guided as default, and
has been using the composer normally:

- New session → guided mode → workflow stepper, turn widgets, normal flow.
- They can switch to freeform per session if they want; the toggle is in
  the chat panel header.
- They never see the account preference unless they go looking in
  settings.

## Returning user, opted out

A user who at some point set their account default to freeform:

- New session → freeform mode → chat-driven authoring.
- They see a "Switch to guided" button in the chat panel header (small,
  always visible).
- On the very first session after the opt-out, a dismissible banner
  confirms the new default.
- The account preference is in settings if they want to flip back.

## Very first user (tutorial)

The tutorial **always** runs guided-style — sequential, structured — even
if the user will end up picking freeform as their default. Reasoning: the
tutorial is also vocabulary teaching, and that teaching benefits from
structure. The mode choice at the end of the tutorial sets the *post-tutorial*
default, not the tutorial's own mode.

This means there is exactly one moment when a user experiences guided
mode regardless of preference: the first-run tutorial. After that, their
preference governs.

## Telemetry to validate the call later

Worth recording (in operational telemetry, not the audit trail):

| Metric | Useful for |
|---|---|
| Opt-out rate within first N sessions | If >40% opt out, the default is mis-tuned |
| Re-enable rate after opt-out | High rate suggests inline opt-out is too easy to hit accidentally |
| Per-mode completion rate (% of sessions reaching an executable pipeline) | Validates that guided actually helps the audience it's designed for |
| Per-mode session-switch rate (how often users switch mid-session) | High = mode-default is wrong for the task; users are course-correcting |

These metrics are project-level health, not user-attributable. Aggregate
only.

## Implementation notes

- `composer.default_mode` is a new user preference, stored in the same
  table / model as other user-level settings.
- Default value for new accounts (no preference set): `guided`.
- Reading the preference: at session-create time only. Switching it
  mid-session does not affect the current session.
- Writing the preference: from settings, from inline opt-out, or from the
  tutorial's final turn. All three write to the same field.
- Existing users (anyone who composed before this redesign ships): default
  to the *current* freeform-default to avoid mode-change surprise.
  See [11-open-questions.md](11-open-questions.md) for whether to grandfather
  them or migrate.

## Memory references

- `project_composer_default_guided_with_opt_out`
- `project_composer_personas`
