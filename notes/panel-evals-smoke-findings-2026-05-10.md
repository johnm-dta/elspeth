# Panel-evals smoke cohort — where the composer is soft (post-patches, 2026-05-10)

**Cohort**: `panel-broad-2026-05-09` smoke subset (6 of 12 planned cells).
**Reason for partial run**: signal saturated — 6 cells covered all 4 personas across 4 distinct shapes; further cells would replicate findings.
**Total cost**: $0.5576 (well under the $12 ceiling).
**Composer model**: `openai/gpt-5.4-mini-20260317` (staging).

## Demo-critical findings (block these before demo)

### 1. Composer misreports its own pipeline state across turns (CORRECTED 2026-05-10)
**Cell**: `boolean_routing__p1_compliance` (Linda, the compliance officer)

**Initial framing was wrong.** I had claimed "composer hallucinated a state change" based on `mutated_state: false` on T5. Direct diff of `state.after.t4.json` vs `state.after.t5.json` shows them byte-identical; tracing `on_validation_failure` across turns shows: t2/t3 = `discard`, t4/t5 = `rejected_records`.

**What actually happened**: the composer DID apply Linda's no-drop fix — between T3 and T4. What it hallucinated was its own state introspection, in opposite directions across two consecutive turns:

- **T4 prose claimed "the source still uses `on_validation_failure: discard`"** — but state already showed `rejected_records`. (Fix HAD been applied; composer reported it wasn't.)
- **T5 prose claimed "I fixed the workflow behavior"** — but state is unchanged from T4. (Fix had already been applied a turn earlier; composer reported it had just done it.)

So the composer's *behavior* is correct (fix lands on T4 and persists). The *narration* is unreliable across turns — it loses track of what's already in state. Linda the amateur compliance officer cannot distinguish "the fix is applied and the prose is wrong" from "the fix is NOT applied and the prose is right" — both look identical.

This is still demo-relevant (amateur personas read prose as authoritative) but **not** an integrity / fraudulent-confidence framing. Probably **P1** rather than P0.

**Methodology lesson**: I treated `mutated_state` / `state_version_after - state_version_before` as decisive without diffing state files. In cell #2 I had even written down "set_metadata may not bump state_version" and then carried `state_version` forward as decisive in cell #4. Should diff state files for any decisive claim about state mutation.

Run dir: `runs/panel-smoke-2026-05-10/boolean_routing__p1_compliance/observations.md`

### 2. Recipe #10 (fork + coalesce) is still broken
**Cell**: `fork_coalesce__p4_adversarial_engineer` (Dev, fully technical persona)

The recent skill-iteration commits — `e3a484f9` (teach name correspondence), `23c73524` (align canonical pattern), reverts `aa18e2fc` and `688cf7cc`, then `5aff40b6` (mandatory advisor escalation) — have not landed a working fix. The composer still produces fork+coalesce wirings whose branch names don't resolve at preflight. With Dev's explicit fix instruction in turn 2, the composer mutated state but produced a *different* fork-related preflight error rather than converging — the failure surface migrated rather than the bug being fixed.

Run dir: `runs/panel-smoke-2026-05-10/fork_coalesce__p4_adversarial_engineer/observations.md`

### 3. Synthesizer hides the model's substantive prose when preflight fails
**Cell**: `fork_coalesce__p4_adversarial_engineer` (both turns), and partially `boolean_routing__p3_marketingops` turn 1.

`synthesizer_replaced: true` on both turns of cell #6. The model's actual prose was substantive ("I set up the product-data workflow with a CSV input, description truncation, a fork into two parallel passthrough branches, a nested coalesce merge, and a JSONL output... the SQLite audit output was removed during cleanup because it was not actually connected to any data path... Operationally: source validation failures are discarded..."). The synthesizer **replaced** all of this with an opaque preflight error.

Operators (and the persona) lose:
- what the model attempted
- the model's own diagnosis of the fork wiring problem
- the disclosure that audit was silently removed
- the discard semantics

Recommendation: switch to `synthesizer_augmented` shape (suffix the technical error onto the model's prose) when the model's prose is informative. The runbook already distinguishes `synthesizer_replaced` from `synthesizer_augmented` in metrics — the policy should follow.

### 4. Composer cannot configure Landscape audit backends
**Cells**: `batch_aggregation__p4` (Dev) and `fork_coalesce__p4` (Dev)

In both cells, Dev requested SQLite audit logging. The composer:
- (cell #2) tried to satisfy the audit ask as a SQL sink, bailed when it couldn't supply a connection URL, then said "no pipeline-level audit-backend setting in the available composer tools"
- (cell #6) silently removed the SQLite audit output as "unconnected" (it would never be connected — audit log is pipeline-level, not a node)

The composer treats audit as a sink/node category instead of a pipeline-level Landscape backend. This is a substantive coverage gap in the composer's MCP tool surface — `set_metadata` and friends don't expose a path to set the audit backend. Given that Landscape audit is the load-bearing feature of ELSPETH's "every decision must be traceable" positioning, **the composer cannot drive the build that the framework is named for**.

## Substantial composer-quality findings (file as bugs)

### 5. Verbal acknowledgement without state mutation
Recurring across cells #2 and #4. Pattern: composer agrees with a user requirement ("you're right", "I confirmed", "yes I can change that") but `mutated_state: false`. The composer needs an internal tool-call gate that requires state mutation to follow verbal agreement, or surfaces the no-op as an explicit "I cannot do this" rather than implying done.

### 6. Column-name and format fabrication
Cell #4 (Linda): the composer chose column name `approval_status` from Linda's "approval status indicator" — the example's actual column is `approved`. Same shape as Marcus's cell where composer used `approved` correctly. Different cells, different fabrications. Plus: composer chose JSON outputs unprompted (only switched to CSV when Linda asked) when the scenario expected CSV.

### 7. Self-correction whiplash on plugin contracts
Cell #2 (Dev): composer claimed build was correct on T1, said it was incomplete on T2, then said T2's correction was wrong on T3. A user without Dev's domain knowledge would have been pulled into adding non-existent plugin options. Suggests composer is reading its plugin schema differently between turns or is lossy on state inspection.

### 8. Composer's first response to dialogue-mode is passive
Cells #1 (Sarah) and #5 (Sarah): on opening turn the composer offered design sketches but did NOT call any build tool. The synthesizer correctly distinguishes "tried to build, failed" from "did not try to build" via the `[ELSPETH-SYSTEM]` block — but the underlying behavior on under-specified asks is "explain plan, await confirmation" rather than "build a plausible draft, name what's missing".

### 9. Composer placeholders blob_id as `__missing__` instead of asking
Cell #1: composer called `inspect_source` with blob_id `__missing__` rather than asking the user where the file is. The synthesized error message recovers the situation, but the underlying behavior — fabricating an identifier rather than asking — should be a guardrail.

## Patches that ARE working (don't lose these)

- **bug `elspeth-861b0c58f5` (csv-source schema-required loop) is substantially mitigated.** Cell #3 (`boolean_routing__p3_marketingops` — the original bug-discovery cell) now converges in 2 turns when the user pushes "use whatever defaults you've got". The synthesized error block now contains a complete corrective object template (`{"sink_name": "...", "plugin": "csv", "options": {...}, "on_write_failure": "..."}`) plus explicit `schema: {mode: observed}` guidance. The recipes.py + skill patches are doing real work.
- **Composer gives upload UI guidance when asked plainly** (cell #5 turn 3) — improvement over cell #1 where Linda struggled.
- **Synthesizer correctly differentiates failure types**: "tried to build, failed" (full corrective template) vs "didn't call any build tool" (terse).

## Harness gaps (fix before broad cohort)

### 10. `scenario_from_example.py` does not bundle `csv_content`
Affects cells #1, #4, #5 directly (no blob → composer can't bind source → cell dead-ends or extends artificially). For chroma_rag the gap is double — needs both csv_content AND knowledge-base connection details.

### 11. `must_have_node_chain_in_order` scoring criterion is misaligned with ELSPETH data model
The scorer checks `state.after.tN.json.nodes[].plugin` against the chain. ELSPETH puts source plugin in `state.source.plugin` and sink plugins in `state.outputs[].plugin` — not in `nodes`. So a chain like `['csv', 'batch', 'csv']` can never match because `nodes` contains only the transforms/aggregations between them. **All AMBER verdicts in this run are from this artifact, not real composer failures.** Fix: update `scenario_from_example.py` to flatten `source.plugin + nodes[].plugin + outputs[].plugin` into one chain before generating the criterion.

### 12. `passivity_phrases` scored on final-content only
Cell #2 hit "if you want, I can keep working from here to rebuild" on turn 3, but the score didn't trigger RED because turn 4 didn't repeat the phrase. Calibration question: should passivity be checked across all turns, or only the final assistant content? Current behavior biases toward "model recovered" pipelines.

## Persona discipline (Phase A working as intended)

| Persona | Cells | Channel 1 | Channel 2 | Notes |
|---|---|---|---|---|
| p1_compliance (Linda) | 1 | in-character (0 drift) | in-character (0 drift, conf 0.95) | Held discipline through 5 turns of pressure |
| p2_researcher (Sarah) | 2 | in-character (0 actionable) | 1× in-character, 1× out-of-character | The "out" was my dispatch fault — primed her with a snake_case path |
| p3_marketingops (Marcus) | 1 | in-character (0 events) | in-character (0 drift, conf 0.92) | Pseudo-tech vocab held |
| p4_adversarial_engineer (Dev) | 2 | exempt (competence_ceiling: none) | 1× in-character (conf 0.98), 1× out-of-character | The "out" was a tonal shift in turn 4 (judge called it "pedagogical") — calibration question, not a discipline failure |

**Calibration insight**: Channel 1 + Channel 2 agree when I don't path-prime the dispatch. The cell #1 disagreement was caused by my dispatch prompt seeding `examples/schema_contracts_llm_assessment/input.csv` to the Sarah subagent. Future dispatch prompts should not seed snake_case paths or system-vocabulary tokens.

## Recommendations (ordered by demo proximity)

1. **Before demo**: address findings #1 (hallucinated state) and #2 (Recipe #10 broken). Even one demo of a compliance-officer persona ending in fraudulent confidence destroys ELSPETH's positioning.
2. **Before demo**: change synthesizer to `augmented` shape when raw_content is informative (finding #3) — the technical error message is fine to surface, but suppressing the model's prose costs operator context.
3. **Before broad cohort**: fix harness gaps #10 (csv_content bundling) and #11 (node-chain criterion).
4. **Pre-Phase 3**: implement audit-backend tool surface (finding #4) — needed for any compliance-positioned demo.
5. **Track as bugs**: findings #5–#9 (composer quality) — file individually, prioritise after demo.

## Run artifacts

- `runs/panel-smoke-2026-05-10/SCORECARD.md` — per-cell verdict + cost table
- `runs/panel-smoke-2026-05-10/aggregate.json` — machine-readable
- `runs/panel-smoke-2026-05-10/<scenario_id>/observations.md` — per-cell narrative findings
- `runs/panel-smoke-2026-05-10/<scenario_id>/{ledger,score,drift,judge}.json` — structured signals
