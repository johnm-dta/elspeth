# Fix A (elspeth-c09eceade4) — seam conflict found; BLOCKED on operator decision

Date: 2026-06-06. Status: implementation HELD (working tree uncommitted). Fix A
as specified is architecturally wrong at the named seam; the real gap is
elsewhere. Surfacing per "no unilateral deferral / surface with concrete
proposal."

## What Fix A asked for

Authoring-time fail-closed guard in `_common.py` (~line 1296, the
`_mask_pending_interpretation_placeholders_for_authoring_validation` seam):
before masking, for each `_legacy_terms(prompt_template)` token, require a
matching pending wired `vague_term` requirement; else raise `ToolArgumentError`.
Goal: convert the late run-time `UnresolvedInterpretationPlaceholderError` into
an in-loop authoring error.

## Finding 1 — the named seam cannot discriminate (so I moved it)

`_prevalidate_plugin_options` (which invokes the mask at 1296) only ever sees
options AFTER `strip_authoring_options` has removed
`interpretation_requirements` and `prompt_template_parts` (the strip happens in
the call expression of `_prevalidate_transform`, `_common.py`). At 1296 the
wired and orphan cases are byte-identical (just a `prompt_template` with a
token). The mask works there only because it blindly rewrites the token; a guard
that must *distinguish* wired-vs-orphan cannot run there.

The discriminating seam is `_prevalidate_transform` itself, BEFORE the strip —
the one chokepoint shared by all three mutation paths (`set_pipeline`
sessions.py:404, `upsert_node` transforms.py:443, `patch_node_options`
transforms.py:726), each of which passes full `review_options`. I implemented
the guard there (uncommitted). Direct + end-to-end tests pass (5/5 in a new
`TestOrphanInterpretationPlaceholderGuard`).

## Finding 2 — the guard breaks the LEGITIMATE two-step flow (4 regressions)

Running the wider suite: 4 failures in
`tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`.
They encode the supported flow:

1. `set_pipeline` writes a node with a BARE `{{interpretation:cool}}` token (no
   requirement yet) — expected to SUCCEED.
2. `request_interpretation_review(kind="vague_term", ...)` then creates the
   user-resolvable EVENT.

This ordering is MANDATORY, not incidental: `request_interpretation_review`
(sessions.py:1252-1279) requires the token/site to already exist in the prompt
(`_matching_interpretation_sites`) before it will create the event, and the
interpretation EVENT is created ONLY by that tool (`create_pending_interpretation_event`,
sessions.py:1698) — never by `set_pipeline`. So there is no "inline co-staging
in one set_pipeline that yields a resolvable event"; the bare token at step 1 is
a TRANSIENT orphan that step 2 resolves. The bug is a PERSISTENT orphan (token
written, review never called). Fix A's set_pipeline guard cannot tell transient
from persistent — it rejects both, killing the bare-token form outright.

`request_interpretation_review` also has a deliberate legacy fallback
(`vague_term_wiring_count`: when no structured requirement matches, COUNT the
`{{interpretation:<term>}}` placeholders). The bare-token form is a supported,
contract-bearing input, not malformed Tier-3.

## Finding 3 — an in-loop guard for this ALREADY EXISTS, and it overlaps Fix A

`service.py:2153-2168`: when the model emits no tool calls (claims done),
`_missing_pending_interpretation_review_sites` detects a token-without-event and
injects a repair message forcing a `request_interpretation_review` retry
(proven by `test_pending_interpretation_placeholder_without_event_forces_review_tool_retry`).
This is exactly Fix A's stated goal (in-loop self-correct before the user runs).

## Finding 4 — the REAL gap: turn-completion is not fail-closed

The existing in-loop guard at 2153 is gated on
`repair_turns_used < _MAX_REPAIR_TURNS`. When the budget is EXHAUSTED, control
falls through to `_finalize_no_tool_response` (2214), which completes the turn as
a normal "success" — there is NO orphan check at finalization. The only backstop
is `materialize_state_for_execution` at RUN time. Result: turn finalizes green,
UI enables "run", backend rejects with `UnresolvedInterpretationPlaceholderError`
— precisely the symptom in 2026-06-06-interpretation-placeholder-rootcause.md
(removing the tutorial-only normalization exposed it; correctness was always
held by the run-time net, this is a UX/timing gap).

## The fork (operator picks; I will not unilaterally decide or flip the 4 tests)

Fix A as specified (set_pipeline authoring guard) (a) deprecates the supported
bare-token two-step form, (b) requires flipping 4 dispatch tests + reworking
`request_interpretation_review`'s legacy fallback, and (c) duplicates the
existing in-loop guard. I recommend re-scoping to the actual gap:

- **Option 1 (recommended): turn-completion fail-closed.** When the repair
  budget is exhausted AND `_missing_pending_interpretation_review_sites` is still
  non-empty, do NOT finalize as success — surface the unresolved interpretation
  as a turn-level failure/blocking diagnostic (mirror the proof-repair gate's
  fail-closed shape) so the UI never enables "run" on an orphan. This makes
  tutorial == regular run, closes the UX gap, and keeps the bare-token two-step
  flow intact. Seam: `service.py:2137-2214` no-tool-calls finalization.

- **Option 2: deprecate the bare-token form** (Fix A as literally specified,
  generalized): require the STRUCTURED requirement co-staged in `set_pipeline`
  (token + `interpretation_requirements` + `prompt_template_parts` ref), reject
  the bare token. This is a deliberate product change: rewrite the 4 dispatch
  tests to the structured flow and remove/retire `vague_term_wiring_count`'s
  legacy placeholder fallback. Larger blast radius; only worth it if the legacy
  flat-token form is being retired anyway (the "migration window" comments).

Fix B (composer skill hard-rule) is compatible with either and reduces firing.

## Working-tree state

Uncommitted: the Finding-1 guard in `_common.py`
(`_assert_no_orphan_interpretation_placeholders` + wiring into
`_prevalidate_transform`) and its 5-test class in `test_tools.py`. These are the
Fix-A-as-specified artifact, retained for reference/diff. They should be
reverted or reworked once the operator picks Option 1 or 2. NOT committed; the 4
dispatch tests are NOT flipped.
