# Plan — Pipeline model policy (allowlist + auto-attested provenance)

**Ticket:** elspeth-065fbd2427 (feature, under composer-correctness epic elspeth-e1ab67e55a)
**Status:** plan for review — NOT implementation-authorised
**Date:** 2026-06-08

## Goal

Let deployment config declare an allowlist of permitted pipeline llm-node models so a
policy-sourced model is auto-attributed to policy (no human review card) while a
genuinely-LLM-authored off-policy model is rejected at preflight (and auto-corrected by
the Fix 2 repair loop) or, opt-in, surfaced for review. Resolves the `llm_model_choice`
provenance conflation; un-blocks the csv-classifier convergence test.

## Background (current machinery)

- `_options_with_default_model_choice_review` (`src/elspeth/web/composer/tools/_common.py:202`)
  unconditionally stages a **pending** `llm_model_choice` requirement for any non-empty
  `options.model` on an llm node.
- `_missing_model_choice_review_sites` / `_validate_model_choice_review`
  (`src/elspeth/web/interpretation_state.py:655,853`) enumerate the orphan and Tier-1-guard
  the resolved choice against `options.model` (hash drift → crash).
- Contrast: `llm_prompt_template` is backend-**auto-surfaced**
  (`_auto_surface_prompt_template_reviews`, `composer/service.py:1376`) so it never orphans;
  model-choice has no such path → it orphans → fail-closed → `is_valid=false`.
- No model-policy config exists today (`WebSettings`).

## Design (decided)

```yaml
pipeline_model_policy:          # absent => today's behaviour (opt-in, non-breaking)
  allowed: [anthropic/claude-opus-4-8, openai/gpt-5.5]
  off_policy: reject            # default; alt: require_review
```

- in `allowed` → requirement staged **auto-resolved**, provenance `policy_attested:<config-ref>`.
- off-policy + `reject` → `ValidationError` in `validate_pipeline` → Fix 2 `_attempt_preflight_repair`
  auto-corrects.
- off-policy + `require_review` → today's pending card.
- absent → today's behaviour (every model → pending card).

## TDD sequence (red → green per step)

**Phase 0 — config surface**
1. RED: `WebSettings` test asserting `pipeline_model_policy` parses (allowed[], off_policy enum,
   defaults). GREEN: add the Pydantic settings model. Confirm absent → `None`.
2. Follow the config-contracts pattern if this needs a Runtime*Config projection (it is consumed
   by composer tools + validator, both of which already take `settings`; likely no new Runtime
   dataclass, but check `scripts.check_contracts`).

**Phase 1 — policy-aware auto-stager (the core)**
3. RED: unit test on `_options_with_default_model_choice_review` — in-allowlist model →
   requirement `status=resolved`, provenance `policy_attested:*`, NOT pending. GREEN: thread the
   policy (from settings) into the stager; auto-resolve when `model in allowed`.
4. RED: in-allowlist model → `_missing_model_choice_review_sites` returns () (no orphan). GREEN
   (should already hold once the requirement is resolved — verify, don't assume).
5. RED: absent policy → unchanged pending behaviour (regression guard). GREEN (no-op path).

**Phase 2 — off-policy rejection + Fix 2 synergy**
6. RED: validator test — off-policy model + `off_policy: reject` → `validate_pipeline` returns a
   `ValidationError` (component=node, error_code e.g. `model_not_permitted`, message names the
   allowed set). GREEN: add the check in `execution/validation.py` (near the other llm-node checks).
7. RED (integration, mirrors TestPreflightRepairContinue): full compose loop, content-aware-free
   real validator, off-policy model on turn1 → Fix 2 repair-continue → composer (scripted) picks
   an allowed model → finalises valid. Proves reject + Fix 2 = auto-correction. GREEN: falls out
   of Phase 1+2 + existing Fix 2.
8. RED: `off_policy: require_review` → pending card (today's path). GREEN.

**Phase 3 — provenance + Tier-1**
9. RED: audit assertion — a policy-sourced run attributes the model to policy
   (`policy_attested:<ref>`) on the run row / requirement. GREEN: stamp provenance.
10. RED: Tier-1 drift — policy-attested resolution + later `options.model` change → still crashes
    in `_validate_model_choice_review` (provenance-agnostic hash check). GREEN (should already
    hold; verify the policy-attested resolution carries the hash).

**Phase 4 — fix csv-classifier + docstring**
11. Set the test's node model to a policy-allowed value (or assert the reject+repair path),
    correct the docstring's `llm_prompt_template`↔`llm_model_choice` conflation. GREEN.

## Cross-cutting

- **Audit doctrine** (`feedback_audit_doctrine_run_config_boundary`): policy is config → auditable;
  recording `policy_attested:<config-ref>` is in-doctrine (config-sourced provenance), not a
  fold of system config into audit-by-precedent. Keep the ref a stable config identifier, not a
  blob of settings.
- **No-legacy / offensive programming**: off-policy is a typed `ValidationError`, not a silent
  drop; absent-config fallback is explicit, not a `.get` default that hides intent.
- **Trust tiers**: `pipeline_model_policy` is persisted config = T3 on read; the allowlist
  membership test is a pure comparison, no new trust boundary, but the new validator check that
  reads `options.model` (LLM-authored) must treat it as the existing tier (already does).

## Out of scope (fast-follow / separate tickets)

- Per-scope overrides (tutorial → stronger model). v1 is a single global allowlist. The scoping
  hook connects to the existing tutorial-model-scoping need (memory
  `project_tutorial_reliability_harness_2026-06-06`).
- The advisor-suppression "symmetry" deferral from the Fix 2 review (separate concern; same code
  region but unrelated decision).

## Verification

`pytest tests/unit/web/composer tests/unit/evals tests/unit/web/execution`; `scripts.check_contracts`
if a Runtime projection is added; mypy + ruff + `wardline scan src/elspeth/web/composer --fail-on ERROR`;
mock-discipline baseline. Adversarial review workflow over the diff (audit/Tier-1 lenses) before landing.
