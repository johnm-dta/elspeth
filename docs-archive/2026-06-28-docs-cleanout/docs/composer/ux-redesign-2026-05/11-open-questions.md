# 11 — Open Questions

This document catalogues product / design / implementation questions that
need a decision before or during implementation. Each is presented with
options and (where applicable) a recommended call. None of the items here
is a blocker for starting work; each is a decision that needs to be made
at some point.

Questions are grouped by where they bite. Mark each with one of:

- **Pre-Phase-0**: blocks adjudication
- **Pre-Phase-N**: blocks phase N
- **In-Phase-N**: can be decided during phase N
- **Post-launch**: tune with telemetry

## A. Audience / framing

### A1. Demo persona mix — **Pre-Phase-0**

**Question:** At the upcoming tech demo, who is in the room?

**Why it matters:** The default-mode call (Phase 1), the tutorial content
emphasis (Phase 4), and the result-rendering choices (Phase 6) all tune
differently for compliance/research audiences vs ops/engineer audiences.
This redesign's recommendations assume the former; if the actual demo
audience skews ops/engineer, several recommendations soften.

**Options:**
- (a) Compliance / regulated industry buyers → recommendations stand.
- (b) ML / data engineering audience → consider freeform-default with a
  guided opt-in path. Tutorial still useful but emphasis shifts.
- (c) Mixed audience → recommendations stand; the opt-out covers the
  variance.

**Recommended call:** Operator to decide; this is a product positioning
question.

### A2. Existing-user migration policy for Phase 1 — **Pre-Phase-1**

**Question:** When Phase 1 ships, what happens to users who composed
under the current freeform-default?

**Options:**
- (a) Grandfather them: existing users keep freeform-default; only new
  users get guided-default. Minimal disruption.
- (b) Migrate everyone: existing users get guided-default; opt-out is
  visible. Some surprise on next login.
- (c) Migrate with a banner: existing users get guided-default with a
  one-time "We changed the default — here's why; switch back?" banner.

**Recommended call:** (c). Honest about the change, gives users one-click
restore, doesn't silently impose.

### A3. Is the composer single-user or multi-user? — **Pre-Phase-6**

**Question:** The "Save for review" gesture (Phase 6) assumes a colleague
on the other end can open the saved session. Does the auth model
currently support this?

**Sub-questions:**
- Can a session be shared between two authenticated users?
- Does the recipient see the same composition state, or a copy?
- Can the recipient edit, or only view?

**Recommended call:** Initial implementation: shareable link opens
read-only inspect view. Multi-user collaborative editing is out of scope
for this redesign.

## B. Backend dependencies

### B1. Does the audit recorder treat dynamic-source-from-chat as a real source? — **Pre-Phase-5a**

**Question:** When a user types data into chat and the composer creates
a source from it, is the resulting source's content hash-recorded the
same way as a CSV source's content?

**Why it matters:** If not, the audit story for dynamic sources is
broken. Tutorial turn 5's audit story assumes "hash of your typed
input" is real.

**Options:**
- (a) Yes, already works — proceed.
- (b) No, but it's a small change — add to Phase 5a scope.
- (c) No, and it's a non-trivial change — flag as a blocker for Phase 5a.

**Recommended call:** Verify before starting Phase 5a. Specifically,
test: type a URL into chat, generate a source from it, run the pipeline,
and check the Landscape audit trail for the source-content hash.

### B2. Does the audit recorder support interpretation-acceptance events? — **Pre-Phase-5b**

**Question:** The "surface the LLM's interpretation" feature records the
user's accepted (or amended) interpretation as a discrete event linked
to the pipeline state. Does the audit recorder support this event shape
today?

**Why it matters:** Without it, the feature is theatre — the user reviews
the interpretation but nothing is recorded.

**Options:**
- (a) Yes — proceed.
- (b) No, but the event is similar to existing audit events — small
  extension.
- (c) No, and it's a new event class — Phase 5b scope expands.

**Recommended call:** Treat as a non-trivial change; design the event
shape during Phase 5b planning.

### B3. Should YAML export be audit-recorded? — **In-Phase-6**

**Question:** When a user clicks Export YAML, should that be recorded
in the audit trail as a distinct event?

**Why it matters:** The YAML itself is not an audit-bearing artifact (the
*run* is). But the export is a moment where a composition leaves the
managed surface; recording it could be useful for traceability.

**Options:**
- (a) Don't record. YAML export is not a state change.
- (b) Record as a low-priority operational event. Useful for "who
  exported what when?" questions but not load-bearing for audit.
- (c) Record only when the export is `mark-for-review` -shaped (signed
  artifact). Casual exports unrecorded.

**Recommended call:** (b). Cheap, low cost, supports forensic
investigation.

## C. Tutorial specifics

### C1. Tutorial transform: rename, LLM, or both? — **In-Phase-4**

**Question:** Does the hello-world tutorial use a field-rename transform,
an LLM transform, or both?

**Why it matters:** The canonical seed prompt ("rate how cool")
inherently uses an LLM transform. But an LLM transform requires
credentials and adds run-time cost. A field-rename transform runs in
milliseconds and has no cost.

**Sub-question:** Can we cache the canonical-seed-prompt run results so
tutorial users hit a cache instead of running fresh LLM calls?

**Options:**
- (a) Canonical seed uses LLM; cache results aggressively for the exact
  default prompt; fall back to fresh runs for user-edited prompts. Best
  story; needs cache infrastructure.
- (b) Tutorial uses field-rename only (simpler); the LLM story is
  introduced later. Cheaper; weaker tutorial.
- (c) Two-path tutorial: simple path (rename) and advanced path
  (LLM). User picks. Adds complexity.

**Recommended call:** (a). The LLM-based tutorial is dramatically better;
caching the canonical run is a focused engineering task.

### C2. Tutorial-skip affordance for returning users — **In-Phase-4**

**Question:** Should the tutorial have a "I've used ELSPETH before, skip
this" affordance?

**Options:**
- (a) No skip. Every new account does the tutorial. Strong consistency.
- (b) Subtle skip link in turn 1 that fast-forwards to the mode-choice
  turn. Lose the vocabulary teaching for returning users.
- (c) Detect returning users via login (e.g., this user has a previous
  session on another account / SSO realm) and offer skip.

**Recommended call:** (b). The vocabulary teaching is for first-time
users; returning users who legitimately don't need it shouldn't be
forced through.

### C3. Re-take tutorial from settings — **Post-launch**

**Question:** Should there be a "redo the hello-world tutorial" affordance
in settings?

**Recommended call:** (Optional polish.) Add if telemetry shows users
looking for it. Not required for launch.

## D. Catalog reshape

### D1. Catalog keyboard shortcut neighbourhood — **In-Phase-7**

**Question:** Current `Ctrl+Shift+P` opens the Catalog and sits next to
action shortcuts (`Ctrl+Shift+V` Validate, `Ctrl+E` Execute). Should it
move to a "reference" cluster (`?` for help neighborhood)?

**Options:**
- (a) Keep `Ctrl+Shift+P`. Update the shortcuts-help dialog to group it
  under "Reference" rather than "Actions."
- (b) Move to `Shift+/` (which is `?` on US keyboards) — but that's
  already the help dialog.
- (c) Pick a new shortcut in the help-adjacent space.

**Recommended call:** (a). Minimal disruption; the grouping in the
help dialog fixes the conceptual confusion.

### D2. "Inline data from chat" — plugin or option? — **In-Phase-7**

**Question:** The catalog lists "Inline data from chat" as the first
source option. Is it implemented as a plugin (with a `plugin: inline_data`
identifier in YAML) or as a special-case option that doesn't appear in
YAML?

**Recommended call:** Operator / engine team to confirm. Likely a real
plugin (so it appears in YAML for reproducibility); the catalog presents
it specially because of its low-friction story.

### D3. Plugin metadata: "when you'd use this" prose — **In-Phase-7**

**Question:** Who writes the "when you'd use this" / "when you wouldn't"
prose for each plugin?

**Options:**
- (a) Plugin authors as part of plugin code.
- (b) Documentation contributors via PR.
- (c) LLM-assisted draft from existing schema + description; human
  review.

**Recommended call:** (a) is the long-term answer; (c) is the
short-term bootstrap. Ship Phase 7 with LLM-drafted prose for all
existing plugins; require human-authored prose for new plugins going
forward.

## E. Completion gestures

### E1. Save-for-review signed artifact — **In-Phase-6**

**Question:** Should "Save for review" produce a HMAC-signed snapshot
(using the existing audit-export signing infrastructure)?

**Options:**
- (a) Yes — signed snapshot, the reviewer can verify it came from the
  composer without tampering.
- (b) No — just a flagged session; trust the session-state integrity.

**Recommended call:** (a) for the Linda use case. The compliance audience
benefits from cryptographic provenance; the cost is one HMAC call.

### E2. Save-for-review reviewer surface — **In-Phase-6 or after**

**Question:** What does the reviewer see when they open a save-for-review
session link?

**Options:**
- (a) Read-only inspect view; reviewer reads the YAML, audit-readiness
  panel, graph. Can comment but not edit. (Initial impl.)
- (b) Read-only with explicit "Approve / Request changes" actions.
- (c) Full editor — reviewer can amend and re-save.

**Recommended call:** (a) for initial implementation. (b) is a natural
next step. (c) requires multi-user editing infrastructure that's
out-of-scope for this redesign.

### E3. Run-result rendering: when does narrative show? — **In-Phase-6**

**Question:** The recommendation is that "Run pipeline" shows a narrative
result summary when `batch_*` analytic transforms are in the pipeline.
What's the exact detection logic?

**Options:**
- (a) Any `batch_*` plugin in the pipeline → narrative mode.
- (b) Specific list of `batch_*` plugins (e.g., `batch_classifier_metrics`,
  `batch_distribution_profile`) → narrative mode.
- (c) Per-plugin declaration: each plugin declares whether its output is
  narrative-renderable; the result view picks based on plugins present.

**Recommended call:** (c). Aligns with the existing plugin-declaration
model. (b) is acceptable as a starting point.

## F. Mode / opt-out

### F1. First-tutorial mode preference for kiosk / shared accounts — **In-Phase-1**

**Question:** What if the same account is shared across multiple users
(e.g., a kiosk login for a workshop)? The first-user's tutorial choice
locks in the default for everyone afterwards.

**Recommended call:** Out of scope for this redesign. ELSPETH is not
designed for kiosk use. Real users have real accounts.

### F2. SSO / org-default override — **In-Phase-1**

**Question:** Should an org admin be able to set a default
`composer.default_mode` for all users in their org?

**Recommended call:** Out of scope for this redesign. Per-user preference
suffices for the documented audiences.

## G. Audit-readiness panel

### G1. Plugin-trust tier display format — **In-Phase-2**

**Question:** How are trust tiers (1/2/3) displayed in plain language?

**Options:**
- (a) "Tier 1 — full trust (audit DB), Tier 2 — elevated (pipeline
  data), Tier 3 — zero trust (external)"
- (b) "Audit data" / "Pipeline data" / "External data" — drop the tier
  numbers in the user-facing surface.
- (c) Show both — domain experts can use the tier numbers, casual
  readers can use the plain names.

**Recommended call:** (b) for the panel display; (c) in the Explain
detail view.

### G2. Retention row when not configured — **In-Phase-2**

**Question:** When retention isn't explicitly configured, what does the
Retention row show?

**Options:**
- (a) `— Not configured (using 90-day default)` — informational, no
  warning.
- (b) `⚠ Not configured` — soft warning encouraging explicit choice.
- (c) `⚠ Not configured` for pipelines that handle sensitive data; `—`
  otherwise.

**Recommended call:** (c). Trust-tier-aware. Compliance users get the
nudge; quick-experiment users don't.

## H. IA cleanup

### H1. Session sidebar replacement — **In-Phase-3 or Phase 8**

**Question:** The session sidebar (always-on 200px) is removed; what
replaces it?

**Options:**
- (a) Header dropdown — list of recent sessions; can be expanded with
  filter + archive controls.
- (b) Command-palette-based: `Ctrl+K → Sessions` shows the list.
- (c) Both.

**Recommended call:** (a) for visibility; (c) for power users — Command
Palette already exists, so adding a sessions section is cheap.

### H2. Graph mini-view click target — **In-Phase-3**

**Question:** When a user clicks the mini graph view in the side rail,
what opens?

**Options:**
- (a) A larger graph view in a modal.
- (b) A larger graph view in-place (the mini grows to fill the rail).
- (c) The graph view as a full-screen takeover, dismissible.

**Recommended call:** (a). Modal preserves the rest of the UI context.

## I. Telemetry

### I1. What metrics matter most for Post-launch tuning? — **Post-launch**

**Question:** Of the metrics suggested in
[05-modes-and-opt-out.md](05-modes-and-opt-out.md), which actually get
implemented?

**Recommended call:** Opt-out rate and per-mode completion rate are
load-bearing for the default-mode call. Others are nice-to-have.

## Decision log template

For decisions made during implementation, log them in this file (or in a
follow-up `12-decisions-log.md`) with the format:

```
### [Decision ID] - [Title]
**Date:** YYYY-MM-DD
**Decided by:** [name]
**Question:** [from this doc, or new]
**Decision:** [the call]
**Reasoning:** [why]
**Reversibility:** [easy / hard / one-way]
```

This is a working document. Update it as questions arise; close items as
they're decided.
