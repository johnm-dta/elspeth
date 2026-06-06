# Root cause + fix proposal: orphaned interpretation placeholders fail tutorial runs

Date: 2026-06-06. Status: PROPOSAL (no product code changed). Investigation only.

## Symptom

After the backend-parity fix (removed the tutorial-only
`_normalise_current_tutorial_state_for_execution`), a fresh **scoring** tutorial
("rate 1-10 how visually impressive") intermittently fails at the run step with
`UnresolvedInterpretationPlaceholderError: ... {{interpretation:visually impressive}}
in LLM transform`. The tutorial UI reaches "run it" (continue enabled), but the
backend rejects the run. Intermittent because the composer (gpt-5.4-mini) is
non-deterministic about how it expresses the subjective criterion.

## Root cause (precise)

The interpretation system recognises a subjective-criterion review two ways:

1. **Structured + resolvable:** a pending `vague_term` entry in
   `options.interpretation_requirements` **wired** to the prompt via either a
   `prompt_template_parts` `interpretation_ref` or exactly one legacy
   `{{interpretation:<term>}}` token. `request_interpretation_review(kind="vague_term")`
   ENFORCES this wiring (`tools/sessions.py:1279`,
   `vague_term_wiring_count(...) != 1` → `ToolArgumentError`). Resolving such an
   event runs `_patch_llm_transform_prompt` which substitutes the placeholder →
   no leftover → run proceeds.

2. **Text-only / orphaned:** `interpretation_state._legacy_placeholder_sites`
   flags a pending `VAGUE_TERM` site for **every** `{{interpretation:<term>}}`
   found in a flat `prompt_template` — purely by text scan, independent of
   whether any requirement/event exists.

**The gap:** nothing at node-authoring time (`set_pipeline` / `upsert_node`)
rejects a `prompt_template` that contains a bare `{{interpretation:<term>}}`
token **without** a matching staged+wired `vague_term` requirement. The authoring
path only *masks* the token for config validation
(`tools/_common.py:1357 _mask_pending_interpretation_placeholders_for_authoring_validation`
→ replaces it with the literal `"pending interpretation"` so LLM-config
validation passes). The wiring guard at `sessions.py:1279` fires **only when
`request_interpretation_review` is actually called** — it cannot stop the LLM
from writing the placeholder text directly via `set_pipeline` and never calling
the review tool.

So when the composer under-fires — writes `{{interpretation:visually impressive}}`
into the flat prompt but does **not** stage the matching `vague_term` review —
the result is an **orphaned placeholder**: `_legacy_placeholder_sites` flags it
(run-blocker), but there is no requirement/event, so Turn 2b shows no card to
resolve it, the user/harness can't clear it, and the run is correctly rejected
at `materialize_state_for_execution` (the runtime safety net, `service.py:505`,
which the code comments tie to a known "≥8/10 staging runs emit the token"
empirical LLM gate).

The deleted tutorial-only normalization was **regex-stripping these orphaned
placeholders** pre-run (`INTERPRETATION_PLACEHOLDER_RE.sub`), so the tutorial
silently "succeeded" where a regular run fails. That is the parity violation,
and removing it (operator principle) correctly exposed the latent composer bug.
The canonical "rate how cool" demo masks it a second way — it is cache-replayed
and never actually runs the composition.

## Why intermittent

The composer sometimes (a) bakes the criterion into the prompt as plain prose
(no token, no review) → run works; sometimes (b) stages a properly-wired
`vague_term` → resolvable → run works; and sometimes (c) writes a bare token
without staging the review → orphan → run fails. Only (c) fails. Verified:
a diag run took path (a)/(b) and passed end-to-end with real output; the
`verify-allfixes` run took path (c) and failed.

## Fix proposal (at source; no band-aid)

**A. Authoring-time fail-closed guard (primary, durable).** In the node-config
authoring-validation path (`tools/_common.py`, where
`_mask_pending_interpretation_placeholders_for_authoring_validation` is invoked,
~line 1296), BEFORE masking, reject any `prompt_template` whose
`{{interpretation:<term>}}` tokens lack a matching pending `vague_term`
requirement with wiring (reuse `vague_term_wiring_count(options, user_term=term)`
and `_legacy_terms`). Raise a `ToolArgumentError` instructing the composer to
either stage `request_interpretation_review(kind="vague_term", user_term=...)`
wired to the token, or remove the token. This turns the late, user-facing
run-time rejection into an **in-loop authoring error the composer can self-correct
before the user ever runs** — and makes tutorial == regular run (both reject the
orphan; both succeed when wired). This is offensive-programming-correct (reject
malformed Tier-3 composer output at the boundary), NOT the normalization band-aid
(which silently rewrote at run).

**B. Composer skill hard-rule (secondary, reduces firing of A).** Add a forceful,
early hard-rule to `pipeline_composer.md`: a `{{interpretation:<term>}}` token in
any prompt MUST be accompanied — in the same `set_pipeline` — by a staged+wired
`vague_term` requirement; never leave a bare token. (Matches the prior finding
that gpt-5.x needs forceful early hard-rules.) A+B together: the skill reduces
mis-firing; the guard guarantees correctness regardless of the model.

**Not chosen:** reintroducing any strip-at-run behaviour (that is the band-aid we
removed); and the larger migration off the legacy flat-token form to
`prompt_template_parts` everywhere (defensible long-term per the "migration
window" comments, but out of scope for this fix).

## Verification plan

1. Implement A (+ B). 2. Re-run the scoring-prompt battery (≥10, paced). Expect:
runs that take path (c) now either self-correct in the compose loop (A's error
lets the composer re-stage) or never reach the user with an orphan; tutorial-pass
parity holds; `UnresolvedInterpretationPlaceholderError` no longer reaches a
graduated tutorial. 3. Confirm a regular `/execute` of the same composition
behaves identically (true parity). 4. Add a unit test for A (orphan token →
ToolArgumentError; wired token → accepted).
