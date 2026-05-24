  ---
  Convergent findings (multiple lanes — load-bearing)

  C1. Judge enforcement is voluntary, not architectural

  Source: architect B1 + quality (process pattern) + systems R2
  Files: .github/workflows/ci.yaml (grep justify|judge returns nothing); config/cicd/enforce_tier_model/*.yaml (no diff-check between merge base and PR)
  Issue: Agent can hand-edit any YAML under an enforce_*/ directory and add an entry without any of the 5 judge fields; CI passes because the loader treats all-None as honest pre-judge-era representation. Pre-judge and judge-skipped entries
  are observationally identical.
  Fix: CI gate that on PRs touching config/cicd/enforce_*/*.yaml computes the new-entry set against merge base and rejects any new entry missing judge_model + judge_recorded_at + judge_rationale + non-null judge_verdict. Pre-judge entries
  grandfathered (unchanged).

  C2. Commons unfenced — judge guards 1 of 15 allowlist surfaces

  Source: systems R1
  File: elspeth-lints/src/elspeth_lints/core/reaudit.py:168 — _SUPPORTED_RULES = frozenset({"trust_tier.tier_model"})
  Issue: config/cicd/ has 15 enforcement directories. Judge guards exactly one. Friction-displacement reinforcing loop will route new suppression activity to the 14 unguarded surfaces (enforce_freeze_guards, enforce_guard_symmetry,
  enforce_audit_evidence_nominal, etc.). Tier_model's metric goes green while aggregate suppression debt grows.
  Fix: Expand _SUPPORTED_RULES to all enforce_* rules. The judge module is already rule-agnostic in its interface; this is configuration not refactor.

  C3. Override drift has no structural anchor

  Source: architect M1 + systems R2
  Files: elspeth-lints/src/elspeth_lints/core/cli.py (justify handler); CI workflow (no override-rate check)
  Issue: OVERRIDDEN_BY_OPERATOR is documented as "a metric to watch"; watch-it-yourself is not a feedback loop. Long-run equilibrium = whatever rate the worst-week deadline pressure tolerates. Each override sets precedent; no override reverses
   one.
  Fix: CI threshold (e.g. override_rate over rolling 30d > 10% fails CI until reaudit clears the WAS_ACCEPTED_NOW_BLOCKED entries). Converts the metric from informational to enforcing.

  ---
  BLOCKERs (singleton)

  None remaining — all 3 prior independent-reviewer BLOCKERs fixed inline in f6c29e8fe.

  ---
  MAJORs (singleton)

  M1. @classmethod + @trust_boundary stacking crashes at import time (empirically verified)

  Source: quality (with empirical repro)
  File: src/elspeth/contracts/trust_boundary.py decoration-time inspect.signature(func) call
  Issue: @trust_boundary applied below @classmethod raises TypeError: <classmethod(...)> is not a callable object at import. @staticmethod works (callable on 3.10+). Asymmetry will surprise authors with a baffling import-time crash.
  Fix: Either (a) pin with 2 regression tests (@classmethod+below crashes; @staticmethod+below works) so authors learn the requirement from test names, or (b) detect classmethod/staticmethod wrapping and emit diagnostic naming the
  decorator-order requirement.

  M2. Concurrent justify writes = silent last-write-wins corruption

  Source: quality
  File: elspeth-lints/src/elspeth_lints/core/cli.py _append_entry_to_yaml uses Path.write_text
  Issue: Two parallel elspeth-lints justify invocations on the same per-module YAML produce silent corruption. CI agents + local human can both invoke; in-flight CI plus local justify could lose an audit entry. Audit primitive only matters if
  writes are durable.
  Fix: Either document "invoke serially" + add a test pinning current behaviour, or use atomic-rename (os.replace) with a tempfile.

  M3. NFR handwave on reaudit operability (~700 entries, no retry/checkpoint/cost ceiling)

  Source: architect M1
  File: elspeth-lints/src/elspeth_lints/core/reaudit.py:244-258
  Issue: Reaudit iterates linearly with no retry envelope, no checkpoint, no cost ceiling, no resumability. A 600-call run failing on call 599 leaves no durable artefact and the operator pays for 598 wasted calls. Decay-sweep is the only
  mechanism that closes the "self-attestation never re-reviewed" failure mode.
  Fix: Add NFR section to plan doc covering target cost-per-sweep, wall-clock, transient-failure semantics, and resumability via --skip M. Then implement at least checkpoint/resume.

  M4. ADR absence for Pillar A

  Source: architect M2
  Files: notes/cicd-judge-cli-prototype-plan.md (informal plan, not numbered ADR); no ADR for single-vendor + schema + closed-set rule
  Issue: Pillar A commits the project to (a) single-vendor OpenRouter routing for an audit-integrity primitive, (b) 5-field audit-record schema baked into AllowlistEntry, (c) OVERRIDDEN_BY_OPERATOR meta-metric with no observation surface, (d)
  closed-set BoundaryRule literal {R1, R5}. Plan doc captures rationale but isn't ADR-class. Wardline port will re-litigate these from scratch.
  Fix: File an ADR covering the four decisions. Prototype-tier ADRs can be terse.

  M5. Positive-path-biased test design — process pattern, 3 BLOCKERs in one commit

  Source: quality (process-level finding)
  Files: tests/unit/elspeth_lints/test_tier_model_decorator_suppression.py (B1 path), test_allowlist_judge_metadata_integrity.py (B2 path), test_justify.py (B3 path)
  Issue: Implementing agents' green test suites missed three boundary defects that an external reviewer caught. The pattern: verification-of-design ("does the function do what spec says?") not adversarial-probe-of-boundaries ("what shape of
  input breaks the contract?"). 3 datapoints = pattern, not slip.
  Fix: Codify the parameterize-invariant-violations-first template from test_allowlist_judge_metadata_integrity.py as the team default for new invariants. Cultural fix, not a commit edit.

  ---
  MINORs (singleton)

  N1. Tier-artifact disclosure gap in plan doc

  Source: architect M4
  File: notes/cicd-judge-cli-prototype-plan.md
  Issue: Plan describes "two-pillar deliverable"; reader doesn't infer that judge enforcement is voluntary. Tier-artifact mismatch.
  Fix: Add "Delivered vs Enforced" section: two columns showing what ships as code vs what CI fails on.
  Source: architect M4
  File: notes/cicd-judge-cli-prototype-plan.md
  Issue: Plan describes "two-pillar deliverable"; reader doesn't infer that judge enforcement is voluntary. Tier-artifact mismatch.
  Fix: Add "Delivered vs Enforced" section: two columns showing what ships as code vs what CI fails on.

  N2. "Judge does NOT fix" tension with should_use_decorator suggestion

  Source: architect M3
  File: notes/cicd-judge-cli-prototype-plan.md:40
  Issue: Plan says "Judge does NOT fix" flatly; implementation has judge suggesting @trust_boundary as structural alternative. Not contradictory but in tension.
  Fix: 1-line edit: "Judge does not write code fixes; it MAY name a structural alternative (the @trust_boundary decorator) when the finding fits the decorator's preconditions."

  N3. _BoundaryMetadata private name imported across modules

  Source: python
  File: elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/rule.py import block
  Issue: Leading-underscore signals module-private; imported from another module breaks the convention pyright/ruff will report.
  Fix: Rename to BoundaryMetadata (public-by-convention) in trust_boundary_suppress.py; update both import sites.

  N4. Any-typed verdict/datetime params in _build_yaml_entry_text

  Source: python
  File: elspeth-lints/src/elspeth_lints/core/cli.py ~`_build_yaml_entry_text**Issue:** Signature usesverdict: Any  # JudgeVerdictandrecorded_at: Any  # datetime"to avoid top-level import" — but the file already has aTYPE_CHECKINGblock
  forRotationPlan. The rationale is solved one block above. **Fix:** Add JudgeVerdictanddatetimeto the existingTYPE_CHECKING` guard; annotate properly.

  N5. --symbol separator inconsistent with canonical_key encoding

  Source: python
  File: elspeth-lints/src/elspeth_lints/core/cli.py:768 _parse_symbol + line 161 help text
  Issue: Canonical key uses : as symbol-part joiner. justify --symbol expects .. Operator copy-pasting a fragment from rotate's output silently hits "No findings match" with no diagnostic.
  Fix: Accept both via re.split(r"[.:]", symbol_arg), OR update error message to name the separator requirement.

  N6. audit-integrity tests use empty valid_rule_ids

  Source: quality
  File: tests/unit/elspeth_lints/test_allowlist_judge_metadata_integrity.py
  Issue: Every test passes valid_rule_ids=set() — validator runs anyway but reader assumes rule-validation is being exercised. Test scope obscured.
  Fix: One-line comment explaining scope, OR pass {"trust_tier.tier_model"} + add per-file-rule field to fixtures.

  N7. --limit 0 semantics undefined in reaudit

  Source: quality
  File: tests/unit/elspeth_lints/test_reaudit.py
  Issue: Argparse accepts --limit 0; orchestrator behaviour unspecified (zero entries? unbounded?). Audit answer to "how many entries did we reaudit?" should be unambiguous.
  Fix: Add test pinning behaviour, OR change argparse to reject 0.

  N8. CRLF/BOM not tested in rotate's text-level YAML rewriter

  Source: quality
  File: tests/unit/elspeth_lints/test_rotate_tier_model.py
  Issue: Tests use LF-only inputs. Windows-edited operator commits + UTF-8 BOMs will likely produce confusing "key not found" errors in the surgical text rewriter.
  Fix: Add one CRLF test + one BOM test pinning behaviour.

  N9. Multiple @trust_boundary decorators stacked on one function not tested

  Source: quality
  File: tests/unit/contracts/test_trust_boundary.py
  Issue: @trust_boundary(suppresses=("R1",)) @trust_boundary(suppresses=("R5",)) def f(...) behaviour is silent. Outer overwrites inner's __trust_boundary__ via functools.wraps. Static analyzer sees only outermost. Probably right but nowhere
  asserted.
  Fix: Add one test pinning the current behaviour or forcing a discussion about whether to crash on second application.

  N10. Honesty-gate first-migration bootstrap sequence undocumented

  Source: architect O1
  File: notes/cicd-judge-cli-prototype-plan.md "Migration phases" Phase 2
  Issue: First @trust_boundary migration requires the behavioural test to land first (else trust_boundary.tests fails the build). Plan doesn't surface this; first migration attempt will hit avoidable CI fail.
  Fix: Add Phase 2 sub-step: "(2a) write the test that exercises malformed input raising; (2b) add the decorator."

  ---
  OBSERVATIONs (single lane each)

  O1. "Fixes that Fail" pattern on 5-flat-field metadata schema

  Source: systems
  Files: elspeth-lints/src/elspeth_lints/core/allowlist.py AllowlistEntry; loader's atomic-quartet+1 validation
  Issue: Each future verdict primitive (fresh-vs-stored confidence, per-model verdict for multi-model ensemble, re-judge-history append-log) bolts another flat field. At ~8 fields the loader becomes a state machine. Structural fix is a
  versioned sub-record now (cheap) vs later (expensive).
  Fix (deferrable): judge_metadata: {schema_version: 1, verdict, ...} sub-record. Not urgent in prototype; named for wardline port.

  O2. Single-vendor model lock acknowledged but unaddressed for prototype scope

  Source: systems R3
  File: elspeth-lints/src/elspeth_lints/core/judge.py:44 DEFAULT_JUDGE_MODEL = "anthropic/claude-opus-4" (post-fixup)
  Issue: Each accepted entry carries the model identity; switching costs grow with corpus. Plan defers configurability to wardline. Reaudit divergence semantics depend on same model returning comparable verdicts.
  Fix (deferrable to wardline): Commit to a model-equivalence story, or accept re-judge sweep on every model rev.

  O3. Duplicated decorator-recognition logic with no equivalence test

  Source: python + systems both noted (this one is between-categories)
  Files: elspeth-lints/src/elspeth_lints/rules/trust_tier/tier_model/trust_boundary_suppress.py (_is_trust_boundary_decorator, _literal_value); same names in elspeth-lints/src/elspeth_lints/rules/trust_boundary/shared.py
  Issue: Two byte-identical implementations. Documented as deliberate decoupling. With no equivalence test, will drift across the next 2-3 refactors. One side will recognize decorator shapes the other rejects.
  Fix: Property-test that feeds same AST snippets through both implementations and asserts they agree on recognition (covers finite set of decorator spellings).

  O4. walk_function_own_scope over-yields outer-scope expressions (docstring overclaim)

  Source: python
  File: elspeth-lints/src/elspeth_lints/core/ast_walker.py:11
  Issue: Docstring says "every node in function's own lexical scope" but iter_child_nodes yields decorators, default values, return annotations — those execute in enclosing scope, not function's own. Benign for current callers but API surface
  over-promises.
  Fix: Tighten docstring OR iterate only func_node.body, excluding decorator_list/args/returns at top level.

  O5. Tier-1 baseline regen pending

  Source: PM (me)
  Files: tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json; test_baseline_capture_is_self_consistent
  Issue: Baseline doesn't include the 3 new trust_boundary.* rules. Also still failing on upstream composer-service refactor drift (2 pre-existing failures noted throughout). Need to regen baseline to add new rule keys AND absorb upstream
  drift in same commit.
  Fix: Run capture_all() against current tree; commit the new JSON.

  O6. VAL story is wholly future work

  Source: quality
  Issue: Test corpus is overwhelmingly VER (does the code do what spec says?). The VAL layer ("does the judge actually catch shallow rationales in practice?") is deferred to live operation. No canned prompt-response corpus, no live-SDK smoke
  test gated on key. For an audit-integrity tool, this is the highest-leverage VAL gap.
  Fix (deferrable): Build canned-prompt VAL corpus for the eventual wardline port; pin openai exact version in [judge] extra; add opt-in live-SDK smoke gated on OPENROUTER_API_KEY.

  ---
  Polish (deferred MINORs from the prior independent code review, never addressed)

  These were marked MINOR/OBSERVATION by the reviewer that found B1/B2/B3 originally; never relitigated by the panel because they were already documented as out-of-scope.

  - N1 (prior): unreachable branch in _outcome_notes (reaudit.py:687-689) — simplify the redundant null-check.
  - N2 (prior): defensive branch in _classify_divergence now arguably unreachable after B2 fix landed — decide: remove (let AttributeError fire on logic drift) or keep as defense-in-depth with a test.
  - N3 (prior): identical decorator-recognition between trust_boundary_suppress.py and trust_boundary/shared.py — same as O3 above.
  - N4 (prior): _suggest_yaml_target collapses three path levels into one filename; not explained in help text. Either explain in --allowlist-dir help or surface chosen target prominently in dry-run output.
  - N5 (prior): BoundaryRule = Literal["R1", "R5"] alias far from where adoption decisions get made — add # CLOSED LIST — see notes/cicd-judge-cli-prototype-plan.md marker.
  - O1 (prior): should_use_decorator exposed in stdout/JSON but not persisted in YAML — finding-overridden-past-decorator-suggestion only discoverable by text search.
  - O2 (prior): _FP_TAG = ":fp=" constant duplicated between rotate.py and reaudit.py's _parse_entry_key.
  - O3 (prior): compute_derived_names 32-iteration cap — rationale not in comment; pick e.g. len(distinct_names_in_function) + 1 for self-documenting bound.

  ---
  Summary count

  ┌──────────────────────────────────────────┬────────────────────────────────────────────────┐
  │                 Category                 │                     Count                      │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Convergent (multi-lane structural)       │ 3 (C1/C2/C3 — these are the load-bearing ones) │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ MAJOR singletons                         │ 5                                              │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ MINOR singletons                         │ 10                                             │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ OBSERVATION singletons                   │ 6                                              │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Polish carried forward from prior review │ 8                                              │
  ├──────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ Total                                    │ 32                                             │
  └──────────────────────────────────────────┴────────────────────────────────────────────────┘

  ★ Insight ─────────────────────────────────────
  - The 3 convergent items are the only ones I'd recommend gating the operator on. C1+C2+C3 together convert Pillar A from social to architectural; any one alone leaves the others exposed. The rest are real but lower-leverage — each can land
  as an individual small-PR follow-up without the others.
  - The single-lane MAJORs (M1-M5) cluster into two groups: code-correctness (M1 classmethod crash, M2 concurrent writes) and design-documentation (M3 NFR, M4 ADR, M5 test culture). The code-correctness ones are fixable in <30min each; the
  documentation ones are non-trivial because they require operator decisions about scope and SLAs.
  - O5 (baseline regen) is technically a known follow-up I flagged earlier, not a panel finding. Including it in the list because it's the only item that blocks CI green on push — everything else passes CI as committed. Worth running before
  any push to remote.
  ─────────────────────────────────────────────────
