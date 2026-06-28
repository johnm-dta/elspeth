# P4 (web_scrape recipe — re-polarized shield) — retrospective plan review

**Verdict: PLAN_HAD_BLOCKING_DEFECT.** The plan as written would not have produced a working canonical tutorial; delivery's E2E review caught and corrected the defect.
**Reviewed against HEAD `7b8b68f24` (release/0.7.0), DELIVERED.** Plan == `docs/superpowers/plans/2026-06-22-tutorial-staged-recut/p4-web-scrape-recipe.md`.
**Method:** 9-lens reality-weighted retrospective (6 reality/fidelity verifiers + architecture/quality/systems), adversarially verified — each finding re-derived against live source and live test output, not the stale worktree and not Loomweave.

## What the verdict means here

This is a judgement on **plan soundness**, not on the delivered tree. The delivered code is correct: all 133 P4 tests pass at HEAD (re-run during this synthesis: `133 passed in 1.53s`), the recipe fires, and the re-polarized shield holds. But the plan **as authored** gated the `web_scrape` predicate on a sink field (`format == "jsonl"`) that the canonical seed does not carry at match time, so executed literally it ships a green-but-broken deliverable: the flagship §4.1 zero-LLM canonical compose silently degrades to LLM-driven compose while every prescribed unit test passes. That is the rubric's definition of a blocking plan defect. The verdict says the plan was unsound; it does **not** say the current code is broken.

## Plan defects caught only in delivery

**D1 — Format-gated predicate broke the canonical recipe (BLOCKER, the verdict driver).** The plan's P4.1 Step 2 prescribed `_has_single_jsonl_output(sink)` returning `sink.outputs[0].options.get("format") == "jsonl"`, and wired `_web_scrape_predicate` to gate on it (plan:138-179). At `match_recipe` time the resolved json sink's options are the operator's Step-2 SCHEMA_FORM submission taken verbatim (`state_machine.py:697-703` ← `SinkIntent.options`); jsonl is auto-detected from the `.jsonl` filename **only at runtime** (`json_sink.py:57-59`), so `format` is absent at match time. The gate returned False for the real canonical sink, the recipe never matched, and the pipeline silently fell through to LLM-driven compose. **Fix:** commit `29320d291` deleted `_has_single_jsonl_output` and switched the predicate to the pre-existing format-blind `_has_single_json_output` (live `recipe_match.py:190-191`, called at `:261` — verified present, old helper gone). The fix is correct, not a band-aid: the builder force-sets `format: jsonl` on its own output (`recipes.py:752-756`) regardless of the matched sink. *Calibration: a plan review SHOULD have caught this.* A predicate that keys on a downstream/runtime-derived field is reviewable — the rule is "predicates may only key on fields populated in `SourceResolved`/`SinkResolved` at `match_recipe` time." The in-file sibling `_classify_predicate` already used the format-blind helper; the established pattern was on screen and not reused.

**D2 — Self-blinding verification: the plan's own "end-to-end proof" masked D1 (BLOCKER, compounds D1).** P4.2 Step 4b (`test_canonical_seed_materialised_source_matches_web_scrape_recipe`, plan:817-874) was billed as "the end-to-end proof" and carried the comment "If this assertion ever flips to None… do NOT relax the predicate." But it built its sink with `_make_single_jsonl_sink()`, hand-setting `options={"path":…, "format":"jsonl"}` (plan:219-229, 861) — the one field the real resolved sink lacks. The proof passed green against an unrepresentative fixture. Worse, P4.1 Step 3's `test_no_match_for_non_jsonl_output` (plan:288-303) **affirmatively asserted** a no-format sink returns False — the plan's mandated test certified the bug as correct. Under the plan-as-written the unit suite is 7/7 green for the broken predicate and would have gone **red** for the correct fix. **Fix:** `29320d291` added `_make_single_json_sink_no_format()` (docstring warns hand-set `format:jsonl` fixtures "re-mask the bug"), inverted the negative test to `test_matches_json_output_without_explicit_jsonl_format` (asserts True), and added `test_match_recipe_fires_for_real_resolved_shape_without_explicit_format`. *Calibration: SHOULD have caught.* A reachability test that hand-supplies its own discriminator is a no-op guard.

**D3 — Orphan-site test checked the wrong surface (CAUGHT, low blast radius).** P4.3 Step 3's `test_cleanup_node_drops_raw_fields_no_orphan_site` (plan:1069-1088) asserted `interpretation_sites(state)` returns `[]` for the staged raw-html cleanup requirement. False: a *staged* pending requirement and a *truly orphaned* missing-cleanup site emit **identical** `InterpretationReviewSite` coordinates (`interpretation_state.py:595-664`), so the site list cannot distinguish staged from orphaned — the assertion was vacuous. **Fix:** delivery re-pinned to `composition_review_contract_error(state) is None`. *Calibration: borderline* — requires knowing the two emitters share coordinates, non-obvious from the plan text.

**D4 — Raw-slot builder calls would KeyError on execution (CAUGHT, minor).** P4.3/P4.4 fixtures called `_build_web_scrape_recipe(_SLOTS)` directly with `_SLOTS` omitting `provider`/`rating_template`, which the builder reads directly and which only `apply_recipe` injects. Executed literally, `slots["provider"]` raises KeyError. **Fix:** delivery routed P4.3 through `apply_recipe(...)` and added the keys to P4.4's `_SLOTS`. *Calibration: SHOULD have caught* — the slots contract is statically checkable.

## Latent / missed risks (delivery did NOT fully address)

- **L1 — Predicate over-breadth in live guided flow (WARNING).** The only source-side discriminator is now "a column named `url`". Any blob-backed json/csv source with a `url` column + single json sink matches `web-scrape` in **every** live guided Step 2.5 — not tutorial-gated. A generic bookmarks json `{url,title,notes}` would be offered web_scrape. The plan never analysed non-tutorial inputs. Mitigated (not eliminated) by the declinable `build_manually` offer. This breadth is a direct consequence of the correct D1 fix.

- **L2 — Now-load-bearing classify/web_scrape tiebreaker is unpinned by outcome (WARNING).** After the format-blind fix, a CSV source carrying both a classifier-keyword `required_field` (e.g. `category`) **and** an observed `url` column matches both `_classify_predicate` and `_web_scrape_predicate`. Empirically `match_recipe` returns `classify-rows-llm-jsonl` — correct, because classify is registered first. But the only guard is `test_web_scrape_predicate_registered_last`, which asserts **list position**, not the **resolution outcome**; a registry reorder would silently misroute classify-shaped inputs with no failing test. The plan documented the opposite ("order is not load-bearing"). **Follow-up:** add a behavioural test that feeds the genuine overlap input through `match_recipe` and asserts classify wins.

- **L3 — Stale test docstring actively misinforms (NOTE).** `test_recipe_match.py:689` still reads "the URL-row json source never collides with the CSV classify/split predicates" — contradicting the corrected source comment at `recipe_match.py:471-472` ("ORDER IS THE TIEBREAKER for one genuine overlap"). Verified live during synthesis. Fold the docstring fix into the L2 test.

- **L4 — `api_key_secret` "literal credentials are rejected" claim is false (WARNING, propagated).** The slot description (`recipes.py:610`) promises rejection, but `validate_slots → _coerce_slot` for `str` does only `isinstance` (`recipes.py:70-73`); the builder wraps the value as `{"secret_ref": …}` (`recipes.py:721`), and `collect_credential_field_violations` short-circuits on any `{secret_ref}` marker (`core/secrets.py:138`). A literal credential is accepted, wrapped, treated as provisioned, and persisted in plaintext. Harm bounded (becomes an unresolvable secret name failing downstream auth — no silent credential use). Capped at WARNING because the identical false claim already exists on `_RECIPE1_SLOTS` (`recipes.py:172`): P4 **propagated** rather than introduced it. **Recommended:** correct the description or add a name-shape guard; fix `_RECIPE1_SLOTS:172` in the same change.

## Confirmed sound (the load-bearing claims that held)

- **JSON-seed collision safety.** `_classify_predicate` and `_split_threshold_predicate` both short-circuit on `_is_csv(source)` (`recipe_match.py:322, 404`), so a json-plugin source can never reach them — the canonical seed cannot be stolen. The L1/L2 defect is in the reverse CSV direction, not in the JSON-seed claim.
- **Re-polarized shield semantics.** `raw_html_cleanup_review_contract_error(state)` returns None (blocking cleanup satisfied); `prompt_shield_recommendation_warning_pairs(state)` returns exactly one pair, component `node:rate_pages` — medium-severity advisory preserved, `azure_prompt_shield` hard node omitted (rev-4 design). Component format `f"node:{node.id}"` verified at `interpretation_state.py:254`.
- **Constructor + real-catalog validation fidelity.** Every embedded DTO matches the on-disk dataclass field set; `slot_type="blob_id"` is a valid `SlotType` literal (`contracts/composer_slots.py:15`). Every emitted option key is accepted by the **real** Pydantic config classes — proven by the apply test driving the recipe through `_prevalidate_plugin_options` against `create_catalog_service()` with `result.success` asserted for JSON and CSV. The flagged `ValidationEntry` uncertainty resolved in the plan's favour (`.message`/`.severity` exist; `"medium"` valid).
- **P3-surfacer non-raise (cross-phase).** Delivered surfacer wraps `create_pending_interpretation_event` in `try/except ValueError: continue` (`service.py:1612-1627`) — the exact W1 backstop the P3 review flagged. `validate_pipeline_decision_node_semantics` does not raise; `composition_review_contract_error` returns None; llm node carries non-empty `rating_template`. Run-tier backstop (`352936197`) and surfacer pin (`6fead8374`) pass.
- **Architectural additivity.** Recipe #4 touched only `recipes.py`/`recipe_match.py` (`da5191e7f`, `04d494529`); dispatcher loop, guided handlers, completion seam, session model untouched. No migrations, no one-way doors.
- **Empirical linchpin (re-run this synthesis).** All 5 P4 test files: `133 passed in 1.53s`. `match_recipe(json url-row source w/ blob_ref, single json sink)` → `recipe_name='web-scrape-llm-rate-jsonl'`, slots `{source_blob_id, source_plugin='json', output_path}`, unsatisfied `{model, api_key_secret, abuse_contact, scraping_reason}`.

## Cross-phase note (P3 / 3.7)

On the P3-surfacer-interaction axis P4 was retrospectively **sound**: the delivered recipe satisfies every P3 anchor settled at `7b8b68f24`, and the surfacer cannot 500 when the field_mapper `pipeline_decision` is staged. The plan's P4.3 Step-3 premise — and the task prompt's R6-point-3 framing — that `interpretation_sites()` returns `[]` for a staged requirement is **false** (D3); delivery re-pinned the orphan check to `composition_review_contract_error`. No blocking P3-interaction defect survives into the delivered code. Compose-gating (blocking contract, satisfied) is correctly separated from run-gating (interpretation sites, advisory).

## Reviewer rollup

- reality/empirical-linchpins: **NO_GO** — predicate gated on absent `format`; plan's own `test_no_match_for_non_jsonl_output` certified the bug. 133/133 pass at HEAD.
- reality/recipe-match-predicate-collision: **NO_GO** — D1 blocking + self-blinding proof; L1 over-breadth and L2 unpinned tiebreaker survive; json-seed collision safety held.
- reality/recipes-build-fidelity: **GO_WITH_WARNINGS** — node graph/slots/registry/shield faithful; lone warning = false `api_key_secret` rejection claim (L4).
- reality/materialisation-canonical-seed-provenance: **GO_WITH_WARNINGS** — source-side provenance rigorous; sink-side D1 blocker CAUGHT; one residual ordering warning.
- reality/constructor-and-catalog-validation: **GO_WITH_WARNINGS** — full constructor + real-catalog fidelity; D1 the only in-scope plan defect, caught at E2E.
- reality/p3-surfacer-interaction: **GO_WITH_WARNINGS** — surfacer non-raise confirmed; D1 + D3 caught, neither latent.
- design/architecture: **NO_GO** — plan deviated from the established format-blind pattern; the verification loop was self-consistently wrong; additivity sound.
- design/quality: **NO_GO** — D1 + masked fixture + inverted negative test = complete test-coverage failure for the central gate; all CAUGHT at E2E.
- design/systems: **GO_WITH_WARNINGS** — blocker caught at E2E; L1/L2 second-order risks not fully analysed; independence-from-P2 claim operationally true.

Lens split 4 NO_GO / 5 GO_WITH_WARNINGS. The verdict is decided by the rubric, not majority: the format-blind predicate defect (D1) is a unanimously confirmed BLOCKER-grade, plan-level defect that would not have produced working code without delivery's `29320d291` fix — so the retrospective verdict is **PLAN_HAD_BLOCKING_DEFECT**.

---

## Confidence Assessment

**Overall Confidence: High.** D1/D2 were independently confirmed by every reality lens and by architecture and quality, with the empirical crux re-run (plan-as-written predicate returns False on the real match-time shape; delivered predicate returns True). One reality lens attempted active refutation and confirmed instead.

| Finding | Confidence | Basis |
|---------|------------|-------|
| D1 format-gate blocker | High | Plan text (138-179) + `git show 29320d291` + live `recipe_match.py:190-191,261` + empirical predicate run; corroborated by 5 lenses |
| D2 self-blinding proof | High | Plan fixture (219-229, 861) + reconstructed buggy predicate passes all plan unit tests; delivery added the discriminating fixture |
| D3 orphan-site surface | High | `interpretation_state.py:595-664` (shared coordinates) + delivered contract-based replacement |
| L1 over-breadth | Moderate | Live predicate is `url`-column-only; "every guided Step 2.5" is a reachability inference, not a run against the live wizard |
| L2 unpinned tiebreaker | High | Empirical overlap run returns classify; only a position-assert test exists |
| L4 credential claim | Moderate | Verified short-circuit path; "persisted plaintext" inferred from the wrap, not observed in a DB row |

## Risk Assessment

**Implementation Risk (of the plan as written): Critical** — the headline deliverable was silently inoperative. **In the delivered tree: Low** — fixed and regression-pinned. **Reversibility: Easy** — recipe additivity reverts cleanly; the fix was a clean local change once diagnosed.

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Canonical tutorial silently degrades to LLM compose | Critical | Certain (plan-as-written) | Already fixed (29320d291) + reachability test |
| Future registry reorder misroutes classify inputs | Medium | Possible | Add outcome-based tiebreaker test (L2) |
| Operator over-offered web_scrape on non-URL data | Low | Possible | Declinable `build_manually`; tighter url-column heuristic (L1) |
| Literal credential persisted plaintext as secret-ref name | Medium | Possible | Correct description or add name-shape guard (L4) |

## Information Gaps

1. **L1 reachability** — whether the live guided Step-2.5 wizard actually surfaces the web_scrape offer for a non-tutorial `{url,…}` json source was reasoned from the predicate, not driven through the running wizard. Bounds the real-world false-offer rate.
2. **L4 persistence** — "persisted in plaintext" inferred from the `{secret_ref}` wrap path; not observed in an actual `set_pipeline` DB row. Distinguishes a cosmetic description error from a genuine plaintext-at-rest exposure.
3. **Synthesis-specific** — no reviewer drove the full `/api/tutorial/run` dispatch path; the zero-LLM claim is verified at build/apply scope only (P4's stated boundary; dispatch-level freeze is P7).

## Caveats & Required Follow-ups

### Before relying on this synthesis
- This is a **retrospective plan** review. The verdict grades the plan; the delivered tree is correct (133/133 pass) and is **not** under remediation.
- Confirm the L2 behavioural test and L3 docstring fix land before the next registry edit.

### Assumptions made
- The rubric's "blocking defect" axis is decided by whether the plan-as-written would produce working code, independent of whether delivery later caught it (the task framing is explicit).
- Reviewer scope boundaries held; the format-gate defect is the predicate lens's, recorded cross-lens without double-counting toward additional blockers.

### Limitations
- Synthesis does not re-verify every reality finding from scratch; D1/D2/L2/L3 were independently re-run, the remainder inherited from claim-by-claim reviewer verification.
- Covers only the nine declared lenses; no legal/compliance/accessibility angle.

---

*Provenance: 9-lens reality-weighted retrospective workflow `wf_7cab4a5d-7e2` (34 agents, ~4.3M tokens across both runs; first run crashed in result aggregation at agent 33 and was resumed from journal — cached lens agents replayed, only failed adversarial verifiers + synthesis re-ran). 40 findings survived adversarial verification across 9 lenses. Lens verdicts: empirical-linchpins NO_GO · recipe-match/predicate-collision NO_GO · architecture NO_GO · quality NO_GO · recipes-build GO_WITH_WARNINGS · materialisation GO_WITH_WARNINGS · constructor/catalog GO_WITH_WARNINGS · p3-surfacer GO_WITH_WARNINGS · systems GO_WITH_WARNINGS. Suite re-run during synthesis: 133 passed in 1.53s.*

