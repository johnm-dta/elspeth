# LLM-primary guided pipeline creation — Series Overview

> **For agentic workers:** REQUIRED SUB-SKILL: each plan in this series is
> executed with **superpowers:subagent-driven-development** (recommended) or
> superpowers:executing-plans, task-by-task. Steps use checkbox (- [ ]) syntax.

**Series goal:** Make the LLM transform the *primary* way a learner builds each
guided pipeline phase. Today the per-phase guided chat box is advisory-only and
the manual form is the only thing that commits config. This series turns the chat
box into a per-phase **driver** that proposes *and applies* config through the
same `handle_step_*` commit seams the manual form uses — applying **in place** so
a learner can revise the current phase by typing again — leads every guided phase
with a plain-English intent box, makes the prompt-injection-shield review fire on
**every** unshielded LLM node as an always-on 3-state (A/B/C) advisory, and
retargets the first-run tutorial into a self-contained passive worked example that
scrapes 3 ELSPETH-served synthetic pages and lands in prompt-shield State C.

- **Spec:** `docs/superpowers/specs/2026-06-25-llm-primary-guided-creation-design.md`
- **Contract (the authority for all shared decisions):**
  `llm-primary-contract.md` (session scratchpad — the file every plan cites). Where
  a plan body and the contract disagree, the contract DECIDES, except where a plan
  body explicitly self-documents a supersession (see Cross-plan issues #1).

---

## The four plans (one line each)

- **p1 — `p1-backend-drivers.md`** — Backend per-phase LLM drivers + `/guided/chat`
  apply contract: generalize the working STEP_1 source-resolve branch to a sink
  driver and the chain solver, and change the contract from auto-advance to
  **apply-in-place** (commit `step_N_result`, leave `guided.step` unchanged,
  re-render the current phase form).
- **p2 — `p2-frontend-intent-surface.md`** — Frontend intent-primary guided
  surface: move the per-phase intent box ABOVE the structured form and recaption
  per phase (presentation-only), and add a client-only `isTutorial` prop that makes
  a tutorial session structurally incapable of reaching the freeform surface
  (concern B).
- **p3 — `p3-prompt-shield-3state.md`** — Global always-on 3-state prompt-shield
  review: fire on every unshielded LLM node — State A (shielded upstream, silent),
  State B (shield available, "wire it in"), State C (none available, high-risk) —
  reusing `pipeline_decision` + `user_term="prompt_injection_shield_recommendation"`
  with NO new `InterpretationKind`.
- **p4 — `p4-tutorial-synthetic-scrape.md`** — Tutorial synthetic-scrape passive
  worked example: 3 ELSPETH-served synthetic project pages, a deterministic
  runtime base-URL/SSRF resolver, the lockstep retarget of the canonical-prompt /
  entry-seed scenario constants, the teaching-moment copy, and the staging-harness
  rubric retarget.

---

## Dependency / execution order

```
        p1 (backend drivers)          p3 (prompt-shield)
              │  start independently         │  start independently
              ▼                              ▼
        p2 (frontend) ── consumes p1's /guided/chat apply contract
              │
              └────────────┬─────────────────┘
                           ▼
                   p4 (tutorial) — depends on p1 + p2 + p3
```

- **p1 and p3 can start immediately** — neither depends on the other (p3 states
  "no dependency on p1/p2"; p1 and p3 touch disjoint surfaces).
- **p2 consumes p1's apply contract** (behavioral, *same wire shape*): the intent
  box rides the unchanged `chatGuided` store action, which p1 makes apply-capable
  on the backend. p2 is *mechanically* writeable/testable before p1 lands (the
  reorder + concern-B are observable without the apply behavior), but it is
  *logically* downstream of p1 — the surface is incomplete until p1's
  apply-in-place response exists. Order p2 after p1 for a coherent slice.
- **p4 is last** — it consumes p1's source-driver output shape (§2.2), p2's
  `isTutorial` freeform suppression (concern B), and p3's per-node A/B/C surface
  (§2.3). p4 implements none of those; it only consumes them.
- **Open ownership gap to track:** the spec's headline "passive, learner-specifies-
  nothing" auto-drive is owned by **no plan's concrete steps**. p4 claims it as
  **Task 8** but defers the steps (they consume p1's apply contract + p2's panel,
  which do not exist until those land) and asks the operator to confirm p4 is the
  right home vs. a dedicated p5. After all four plans land, the headline passive
  behavior may still not be implemented unless Task 8 is scheduled. See Cross-plan
  issues #3.

---

## Shared cross-plan contract summary

### Apply semantics (the load-bearing decision — contract §1)

- **Same commit seam.** The LLM driver commits a phase through the *identical*
  `handle_step_*` seam the manual form uses, producing the *same-shaped*
  `step_N_result` on `GuidedSession`. The shared contract is the seam + the
  `step_N_result` shape, NOT the route or the dispatcher.
- **Attempt-to-drive, mutate-only-when-actionable.** Every `POST /guided/chat`
  submit attempts to drive the current phase. It mutates ONLY when the driver
  produced an actionable config the strict `handle_step_*` + `validate_pipeline`
  seams accept. Non-actionable input (a question, ambiguous/prose-only reply, or
  any LLM failure/timeout/malformed output) falls back to **advisory prose with NO
  mutation** (`next_turn=None`, appends `chat_history` only — never wizard
  `history`). The wizard remains fully operable offline; a phase is never bricked.
- **Apply is IN-PLACE — no auto-advance.** A successful chat apply writes
  `step_N_result` and leaves `guided.step` UNCHANGED. The editable form re-renders
  populated from `step_N_result`. Revise = type again (the driver re-proposes
  against the current applied config). Advancing is a separate, explicit confirm
  through `/guided/respond`. The existing STEP_1 auto-advance (`guided.py:1863`) is
  REMOVED for the generalized path; `test_step_1_chat_..._emit_step_2` is a
  test-to-UPDATE (assert in-place), not a precedent extended unchanged.
- **Guards preserved verbatim** on the apply path: unknown-step 400, terminal 409,
  step-mismatch 409.
- **Apply/response shape (p1 PRODUCES, p2 CONSUMES):** wire shapes UNCHANGED. On an
  actionable apply: `guided_session.step` unchanged, `composition_state` reflects
  the committed `step_N_result`, `next_turn` = the re-rendered current-phase form
  populated from `step_N_result`. On non-actionable: advisory `assistant_message`,
  `next_turn=None`. p2 rides the existing `chatGuided` merge (`next_turn ?? prev`).
- **Source-driver output (p1 PRODUCES, p4 CONSUMES):** an inline URL-row
  `SourceResolved` (`plugin in {"json","csv"}`, observed `url` column, `blob_ref`
  materialized downstream). It does NOT propose a `web_scraper` source; the scrape
  is routed into the *transform* stage by `_web_scrape_predicate` (format-blind) +
  the `web-scrape-llm-rate-jsonl` recipe (`web_scrape` transform node), firing at
  STEP_2.5.
- **Prompt-shield surface (p3 PRODUCES, p4 CONSUMES):** no new `InterpretationKind`
  — reuse `PIPELINE_DECISION` + `user_term="prompt_injection_shield_recommendation"`.
  p3 builds `prompt_shield_state_for_node(node, all_nodes, *, shield_available) ->
  "A"|"B"|"C"`, the B-vs-C resolver `azure_prompt_shield_available(context)`, and
  the B/C draft constants. p4 consumes the State-C result to word the tutorial
  override copy (p4's copy is hand-written, not an imported draft).

### EPOCH / schema + gate decisions (contract §3)

- **NO plan bumps `GUIDED_SESSION_SCHEMA_VERSION` (=6) or `SESSION_SCHEMA_EPOCH`
  (=24).** Apply-in-place + revise write only fields already serialized in
  `GuidedSession.to_dict`/`from_dict` (`step_N_result`, `step`, `chat_history`,
  `chat_turn_seq`); no new "proposed-but-not-advanced" staging field. p2 is
  presentation-only. p3 adds no new kind/CHECK/tool-enum/resolve-dispatch. p4's new
  `WebSettings.tutorial_sample_base_url` is a settings-shape addition orthogonal to
  the session DB epochs. Consequence: **no DB-delete migration and no boot
  fail-close in this series.**
- **Operator-owed cleanup (not a schema epoch):** p4's canonical-prompt cache goes
  vestigial once synthetic URLs are runtime-derived; stale `{data_dir}/tutorial_cache/`
  files simply miss — the operator clears them (documented artifact-delete pattern).
- **Gate posture:** the plugin `source_file_hash` gate and the tier-model
  fingerprint cascade are operator-owed re-sign chores; co-land the fingerprint/hash
  updates with any source change, the operator re-signs. p2 is `.tsx`/`.ts`-only so
  those Python gates do NOT fire on its diff (p2 commits with a plain `git commit`,
  no `SKIP=`). p4 fires the canonical-prompt coupling gate (constant + byte-identical
  mirror + two value-asserts) and invalidates the dormant cache, but needs NO
  service restart and NO `composer_skill_hash` re-bake (the recipe/skill stay
  scenario-agnostic).

---

## Global Constraints

```
## Global Constraints

- All work lands on `release/0.7.0` (the named release branch), NOT a feature
  branch. Verify `git branch --show-current` before committing; feature branches
  get orphaned.
- The agent SIGNS NOTHING. The operator holds the HMAC key and pushes per the
  release-train process. Do not proactively re-sign tier-model fingerprints or
  plugin hashes; surface owed re-signs as an operator chore.
- Editing a plugin file (e.g. `src/elspeth/plugins/transforms/llm/transform.py`)
  trips TWO CI gates, both operator-owed re-sign chores: (a) the plugin
  `source_file_hash` gate (`plugin-contract-plugin-hashes`) — refresh via
  `scripts/cicd/plugin_hash.py` (`compute_source_file_hash`/`fix_source_file_hash`);
  (b) the tier-model fingerprint cascade (`trust-tier-model`; adding imports
  shifts `Module.body` indices) — allowlists `config/cicd/enforce_tier_model/plugins.yaml`
  (plugin files) and `.../web.yaml` (web files: interpretation_state.py, state.py),
  rotated via `elspeth_lints.rules.trust_tier.tier_model.rotate`
  (scripts/cicd/rotate_tier_model_fingerprints.py). Co-land the fingerprint/hash
  updates with the source change; the operator re-signs.
- The canonical tutorial prompt couples FOUR things in lockstep: the backend
  constant `CANONICAL_SEED_PROMPT` (`web/preferences/tutorial_cache.py`), its
  byte-identical FRONTEND MIRROR `CANONICAL_TUTORIAL_PROMPT`
  (`frontend/src/components/tutorial/tutorialMachine.ts`, byte-identity enforced
  by `test_canonical_seed_matches_frontend_constant`), the `composer_skill_hash`
  re-bake (`PIPELINE_COMPOSER_SKILL_HASH` in `composer/prompts.py` +
  `assert_skill_hash_unchanged_on_disk`) when the live `pipeline_composer.md`
  skill changes, AND a live-prompt SERVICE RESTART. Editing the prompt constant
  alone needs the mirror + the two value-assert tests (NO restart). Editing the
  live skill/recipe needs the re-baked hash + restart (the 5-input
  `tutorial_model_id` invalidates the cache). Do not conflate the two.
- Prompt-shield reviews and advisor/checkpoint reviews are ADVISORY and NEVER
  hard-block: emitted into `validate()` `warnings` at "medium", excluded from the
  blocking contract. Do not promote them to errors.
- `entry_seed` (the tutorial framing/dataset seed) is SERVER-SIDE ONLY and never
  rides the wire: it is redacted from `WorkflowProfileResponse`. Do not add it to
  any wire/GET shape, and do not infer "tutorial" from the wire profile booleans
  (use the client-only `isTutorial` React prop).
- Existing tests that assert about-to-change behavior must be UPDATED to the new
  behavior, NOT reverted. A wave of failures after a structural change is the
  change landing visibly; update the assertions, do not roll back the change.
- For code commits use `git commit` with
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier` (the
  operator-owed-re-sign gates) — NEVER a blanket `--no-verify`. Doc-only commits
  may use `--no-verify`. Reconcile the full slice diff at the slice boundary.
- NEVER `git add -A` / `git add .`; stage only the files this plan owns.
```

---

## Execution guidance

- **Required sub-skill.** Execute each plan with
  **superpowers:subagent-driven-development** (recommended) or
  superpowers:executing-plans, task-by-task. Each task ends green; a phase boundary
  is a reconciliation point.
- **Recorded-baseline attribution.** This repo carries operator-owed red gates
  (tier_model HMAC / freeze_guards / baseline reds the operator holds
  deliberately). Run lints against the **recorded baseline**, not "is the gate
  green now" — a pre-existing red that the operator owns is NOT this series'
  regression. Attribute failures to the change that introduced them; do not adopt
  unrelated red gates as blockers.
- **Commit recipe for code commits:** `git commit` with
  `SKIP=elspeth-lints-freeze-guards,elspeth-lints-trust-tier` (the two
  operator-owed re-sign gates) — NEVER a blanket `--no-verify`. p2 is frontend-only
  and uses a plain `git commit` (those Python gates do not fire on `.tsx`/`.ts`).
  Doc-only commits may use `--no-verify`. Stage only the files the plan owns; never
  `git add -A`/`.`.
- **The agent signs nothing.** The operator holds the HMAC key and pushes per the
  release-train. Co-land any fingerprint/hash refresh with the source change and
  surface the re-sign as an operator chore; do not proactively re-sign.
- **Branch.** All four plans land on **`release/0.7.0`**. Verify
  `git branch --show-current` before each commit.

---

## Cross-plan consistency — findings

The three named shared interfaces all match the contract and each other:

- **Apply-response (p2 ← p1):** p1 Task 3/4 produce `step` unchanged + populated
  `next_turn` from `step_N_result`; p2 consumes via the unchanged `chatGuided`
  (`next_turn ?? prev`). Match.
- **Source-driver output (p4 ← p1):** inline `json`/`csv` URL-row `SourceResolved`
  (url column + `blob_ref`); `web_scrape` routed to the transform stage via
  `_web_scrape_predicate` + the `web-scrape-llm-rate-jsonl` recipe. Match (p1's
  cross-plan summary block restates this verbatim).
- **Prompt-shield (p4 ← p3):** `prompt_shield_state_for_node(...) -> "A"|"B"|"C"` +
  the B/C drafts; p4 consumes the State-C *result* (its override copy is
  hand-written, not an imported constant). Match.

Residual cross-plan items are documented/owner-pending, not blocking — see the
StructuredOutput `crossPlanIssues` list.
