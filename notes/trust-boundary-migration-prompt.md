# Task prompt — Migrate honest Tier-3 suppressions to @trust_boundary (~136 candidates)

You are continuing the cicd-judge tier_model backfill. This task is the **decorator-migration**
half of the 272-entry BLOCK queue: entries the cicd-judge confirmed are *honest Tier-3 boundaries*
but rejected because they use a raw per-line allowlist entry instead of the structural
`@trust_boundary` decorator. Migrating them replaces the allowlist line with a decorator that
makes the suppression self-documenting and auditable. This is the deferred migration tracked under
filigree **elspeth-987f902911** — coordinate there.

## Read first (do not skip)
- Memory: `project_cicd_judge_tier_model_backfill_2026-05-30.md`.
- Worklist: `notes/reaudit-rejudge-block-queue-2026-05-30.md` — the **"Migrate to @trust_boundary (~136)"** section.
- Per-entry rationales: `notes/reaudit-rejudge-postprompt-2026-05-30.md` (each says why + names the fix).
- Decorator contract: `src/elspeth/contracts/trust_boundary.py` (signature, validation, stacking ban).
- A worked ACCEPT example of the target shape: corpus case `accept_existing_decorator_r1` in
  `config/cicd/judge-quality-corpus/v1.jsonl`.

## HARD GATE — verify eligibility before migrating (the heuristic over-includes)
The bucket was split by "rationale mentions @trust_boundary", which is imprecise. For each entry,
confirm **all** of these before applying a decorator; if any fails, it is NOT a decorator migration:

1. **Rule is R1 or R5.** `_ALLOWED_BOUNDARY_RULES = {"R1", "R5"}` (see
   `rules/trust_tier/tier_model/trust_boundary_suppress.py`). The decorator can ONLY suppress R1 and R5.
   An R2/R6/R7/R8/R9 entry **cannot** be decorated — route it to reification, a `per_file_rule`, or a
   signed allowlist entry instead. (Many web getattr entries are R2 — watch for this.)
2. **Genuine Tier-3 boundary.** The suppressed value's *contents* originate externally (network / file /
   operator YAML / provider SDK response), per the enhanced judge doctrine ("is the shape contractually
   guaranteed here?"). If it's first-party Tier-1/Tier-2 data, it's reification, not migration.
3. **Data-flow rooted at a parameter.** The decorator's `source_param` names a function parameter, and the
   analyzer checks the finding's subject is rooted at that param. If the value comes from a local file-read
   or computation (e.g. `dependency_resolver._load_depends_on`'s `loaded`), the function must be
   restructured so the external value arrives as a parameter, OR it's not decorator-eligible.
4. **The boundary establishes trust.** Per doctrine, a Tier-3 guard is honest only if it coerces+records
   (absence as `None`, not a fabricated default) or crashes on the upgrade to Tier-2. If the code
   silent-defaults/fabricates, it's reification, not migration — fix the code.

## Method (per entry)
1. Read the rationale + code; run the 4-point eligibility gate. Reclassify ineligible entries (note why).
2. Apply `@trust_boundary` to the enclosing function:
   ```python
   @trust_boundary(
       tier=3,
       source="<what the external source is>",
       source_param="<the parameter carrying external data>",
       suppresses=("R1",),  # or ("R5",) / ("R1","R5")
       invariant="<what the function guarantees on malformed input, e.g. raises X>",
       test_ref="tests/.../test_*.py::test_<the invariant>",
       test_fingerprint="sha256:...",  # per the decorator contract
   )
   ```
   The `invariant` + `test_ref` must be real — write the test if it doesn't exist (TDD).
3. **Delete the per-line allowlist entry** from `config/cicd/enforce_tier_model/<module>.yaml`.
4. Re-run the gate (agent-keyless → shape-only): the R1/R5 finding must now be suppressed *by the
   decorator* (it appears in the `R_TB_SUPPRESSED` audit stream, not as a violation), and the
   decorator-hygiene rules (`R_TB_NONLITERAL`, `R_TB_MALFORMED`, stacking ban) must pass.
5. ruff + mypy on touched files.

## Sequencing & scope
- Group by subsystem; `web/` dominates (~119 candidates, but expect a chunk to be R2-ineligible — reclassify).
- Surface each batch for review. File/track under elspeth-987f902911.
- Worktree: ask the operator first (default yes).

## Constraints
- **Never** hold or use `ELSPETH_JUDGE_METADATA_HMAC_KEY` (operator-only). Migration removes allowlist
  entries and adds decorators; no signing needed.
- One decorator per function (stacking is banned and detected). If a function has multiple suppressible
  findings, list them together in one `suppresses=(...)`.
- A migration must be honest: the `invariant`/`test_ref` must reflect real behaviour, not be theatre.

## Done when
Every eligible entry carries a `@trust_boundary` decorator with a real invariant+test and its allowlist
entry removed (gate green); every ineligible entry is reclassified (reify / per-file-rule / signed
allowlist) with a recorded reason.
