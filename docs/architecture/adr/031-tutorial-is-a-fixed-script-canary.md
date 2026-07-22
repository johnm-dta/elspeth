# ADR-031: The Tutorial Is a Fixed-Script Canary for the General Guided Surface

**Date:** 2026-07-22
**Status:** Accepted
**Deciders:** John Morrissey, Claude Fable 5
**Tags:** composer, tutorial, guided, backend-parity, testing-doctrine,
          silent-degradation

## Context

The first-run tutorial walks a fixed scenario (summarise web pages) through
the guided composer. Two standing rules make it deliberately rigid:

- **Backend parity.** The tutorial has no tutorial-only backend path, no
  tutorial-only normalisation, and no special-case planner treatment. Its
  only privilege is a frozen prompt. Everything else is the same guided
  machinery every user exercises.
- **No adaptive slack.** The frozen prompt cannot be rephrased, the walk
  cannot re-plan around a broken affordance, and the scripted driver cannot
  improvise. If any step of the general surface is defective, the tutorial
  stops there.

During the 2026-07-22 three-pathway verification campaign this rigidity had
an unplanned but decisive effect. The freeform and guided acceptance tests
put a live LLM in the loop; the tutorial does not. Nearly every defect the
campaign fixed "for the tutorial" was in fact a general guided-surface
defect that the tutorial was merely the first consumer rigid enough to die
on: the empty auto-proposal being silently accepted, interpretation-review
events never materialising at wire-confirm, review cards orphaned on the
completion surface, and a state-binder rewrite that severed legal pipeline
wiring after generation.

The adaptive surfaces masked those same defects. A planner with a repair
budget quietly compensates for machinery bugs — the binder defect was
mis-attributed to model error for four consecutive runs because every
generation "repeated the same mistake" (the binder was rewriting all of
them identically after the fact). A production user on an adaptive surface
experiences this class of defect not as failure but as polite degradation:
extra repair rounds, slower authoring, occasionally a backend that quietly
rescues a mangled result. Nothing pages anyone. The counterfactual is
uncomfortable and concrete: these defects could have shipped, with a large
share of authoring turns being silently absorbed or repaired behind the
scenes, and we would have been indifferent because no surface ever went
visibly red.

## Decision

Keep the tutorial maximally fragile, and treat that fragility as a feature
with doctrine status:

1. **The tutorial stays on the general path.** Backend parity is preserved
   permanently: no tutorial-only normalisation, shortcuts, or defect
   workarounds. When the tutorial breaks on a general-surface defect, the
   fix lands in the general surface.
2. **Tutorial-green is a machinery signal, not a feature signal.** Because
   it is a fixed-input walk with no adaptive shock absorber, the tutorial's
   end-to-end pass is the cheapest honest indicator of the guided
   machinery's true state, and it is weighted accordingly at release
   boundaries.
3. **Adaptive-surface green does not subsume tutorial green.** A passing
   freeform or guided run proves the journey is *achievable with an
   intelligent agent compensating*; it does not prove the machinery is
   sound. Both signals are required.
4. **Repair-round counts are a masked-defect indicator.** Elevated
   planner repair activity on adaptive surfaces is investigated as
   possible machinery defect absorption, not accepted as model
   variance. (The per-attempt planner trail exists to make this legible.)

## Consequences

- The tutorial will continue to break first, and loudly, on defects that
  other surfaces absorb. This is the intended behaviour and the cost is
  paid deliberately: tutorial reds are machinery investigations, not
  tutorial-team tickets.
- Fixed-script walks over general surfaces are a pattern worth repeating:
  any future high-variance authoring surface should carry at least one
  frozen-input end-to-end walk whose only permitted outcome is complete
  success.
- Repair-budget tuning is treated with suspicion as a remedy. Raising the
  budget in response to repeated identical rejections buys more rejections
  of a possibly-correct answer; the identical-rejection fingerprint
  (independent generations, including a stronger escape-hatch model, all
  rejected with the same code) is diagnosed as a boundary contradiction on
  our side first.

## Alternatives considered

- **Tutorial-only normalisation to keep the walk green.** Rejected
  (repeatedly, in the moment, under schedule pressure — which is exactly
  when it is most tempting). Each such patch would have hidden a real
  general-surface defect and converted the canary into a liar.
- **Retiring the fixed walk in favour of adaptive acceptance tests only.**
  Rejected: the campaign demonstrated that adaptive tests structurally
  cannot detect the masked-defect class the tutorial catches.
