# ADR-007: Pass-through contract propagation — declaration, semantics, and composer parity

**Date:** 2026-04-19
**Status:** Accepted
**Deciders:** Architecture Critic (SME agent), Systems Thinker (SME agent), Python Code Reviewer (SME agent), Quality Engineer (SME agent), Claude (synthesis/lead)
**Tags:** dag, schema-contract, plugin-base, composer, propagation, validation

> **Reader orientation (added 2026-04-20; issue `elspeth-3f320398f1` / M7).**
> For the current authoritative framework governing **new** plugin declarations
> (creates_tokens, declared_output_fields, source guaranteed_fields, sink
> required_fields, etc.), start at [ADR-010](010-declaration-trust-framework.md).
> ADR-010's `DeclarationContract` protocol, decorator registry, and dispatcher
> generalise the pattern this ADR pioneered. This ADR (007) remains normative
> for the single-declaration pattern on `passes_through_input` specifically —
> `PassThroughDeclarationContract` is the concrete 2A registration — but new
> declarations do NOT copy this ADR's structure and do NOT get their own
> bespoke walker/executor/harness trio. They register under the ADR-010
> framework. The pre-existing amendment banners below record the technical
> supersession history; this banner exists to route readers to the right
> starting point on the first encounter with the ADR series.

> **Amended by [ADR-009](009-pass-through-pathway-fusion.md) on 2026-04-19.**
> ADR-009 supersedes §Decision 1's "unconditional contract on every row"
> language with an empty-emission carve-out (Clause 3), closes the
> duplicated-walker mitigation in §Negative Consequences #2 with shared
> primitives (`compose_propagation` and `participates_in_propagation`, Clause
> 1), and delivers the invariant harness named in §Neutral Consequences line
> 83 (Clause 4). Track 2 will tighten Clause 3 via a new `can_drop_rows`
> declaration within 90 days of Track 1 merge.

> **Amended by ADR-010 (Declaration-trust framework, 2026-04-19).**
> Normative in this ADR: Decisions 1–3 for `passes_through_input` specifically.
> Superseded: §Negative Consequences #2 (resolved earlier by ADR-009 §Clause 1);
> §Neutral Consequences line 83 (invariant-test placeholder) — now concrete in ADR-009 §Clause 4 and ADR-010 `test_contract_negative_examples_fire.py`.
> New declarations do NOT copy this ADR's pattern — adopt the ADR-010 framework and register a contract.

## Context

`BaseTransform._build_output_schema_config` runs once at `__init__` using the transform's own YAML `schema_config`. Transforms are constructed before edges are wired, so `schema_config.guaranteed_fields` is the transform's *input* declaration from YAML — not the actual upstream predecessor's guarantees.

When a transform declares `mode: observed` with no fields, its computed `output_schema_config.guaranteed_fields` is only `declared_output_fields`. At runtime, however, `process()` emits rows that carry forward the upstream's fields. `BatchReplicate.process`, for example, deep-copies each input row and adds `copy_index`; the runtime contract is strictly broader than the static contract.

The sink validator `_validate_sink_required_fields` honestly reports this as a `GraphValidationError`: the static contract does not declare fields the sink needs. Operators work around this by declaring explicit `guaranteed_fields` on upstream aggregations (see commit `c633a36b` on `examples/deaggregation/settings.yaml`). The workaround is fragile: it requires the operator to know the transform's runtime behaviour and duplicate that knowledge in YAML. Third-time-lucky pattern — this same ticket was filed three times before this ADR landed.

### Symptom: false rejections of valid pipelines

Any pipeline with an observed-mode aggregation upstream of a pass-through-style transform and a sink requiring the aggregated fields is statically rejected even though runtime behaviour is correct. The longer we defer the structural fix, the more explicit-schema workarounds proliferate — and each is load-bearing, hiding the underlying bug from future designers.

### Bug-class, not just a bug

The root cause is that static DAG analysis trusts plugin declarations without verification. The named divergences in the original bug report (fork-to-sink checks, aggregation required-fields, source alias handling) were all *instances* of this class. Fixing the three named instances (commit `d29f8546`) closed the bug report but left the class wide open.

## Decision

Three interrelated decisions are recorded in this single ADR (the v2 plan split them into three; the ADR review concluded they are consequences of one another and should not be decision-fractals).

### Decision 1 — Declaration: opt-in `passes_through_input=False` default

Add a class-level `passes_through_input: bool = False` attribute to `BaseTransform` and to `TransformProtocol`/`BatchTransformProtocol`. The annotation is **unconditional**: setting it to `True` is a contract that `process()` preserves every input field on every row, regardless of row content or runtime state.

The flag applies at both TRANSFORM and AGGREGATION node positions. ELSPETH plugins like `BatchReplicate` live in `plugins/transforms/` but are wired under `aggregations:` in pipeline YAML (by virtue of `is_batch_aware=True`). Both usages execute transform-class plugins and share the propagation semantics. The NodeInfo guard permits `passes_through_input=True` on both NodeType.TRANSFORM and NodeType.AGGREGATION; it is still rejected on SOURCE, COALESCE, SINK, and GATE nodes where the attribute has no meaning.

Conditional pass-through (drop based on row content) is **forbidden** under this annotation. A conditional drop annotated `True` would pass static + Hypothesis tests and crash production via `PassThroughContractViolation` when runtime observes the drop. Making this an unconditional contract closes that gap: authors annotating a class must reason about all rows, not the typical case.

### Decision 2 — Intersection across predecessors

When a pass-through transform has multiple predecessors (diamond topology via coalesce, fork-to-same-target, etc.), the effective guarantees at the transform's output are the **intersection** of participating predecessors' guarantees, unioned with the transform's own declared output fields.

The pinned-semantics table (reproduced below) covers the 11 edge cases the plan review identified. Intersection (not union) because union would mask silent field drops in one branch: if branch A guarantees `{a,b}` and branch B guarantees only `{a}`, a sink requiring `b` downstream is satisfied only sometimes at runtime — the static contract must reflect the worst case.

| Edge case | Behavior | Rationale |
|-----------|----------|-----------|
| Predecessor abstains (`guaranteed_fields=None`) | Skipped in intersection | Matches `_validate_sink_required_fields` abstain-vs-empty predicate |
| Predecessor declares empty (`()`) | Participates → intersection collapses to empty | Producer committed to zero guarantees |
| Coalesce upstream (any strategy) | Coalesce's pre-computed `output_schema_config` is the leaf; recursion does not double-intersect | Builder materialised policy-aware guarantees |
| Aggregation upstream | Recursion descends like any node; observed-mode returns empty; `mode: fixed` with declared fields participates | A non-opt-in aggregation correctly stops the chain |
| `mode: closed` upstream | Only fields in the predecessor's effective set propagate | Mode is invariant under propagation |
| Gate upstream | Single predecessor (gate's sole upstream); gate's `output_schema_config` is the leaf | No multi-predecessor case for gates |
| Multiple predecessors via fork-to-same-target | `predecessors()` yields each source once; intersection across nodes | Matches existing sink validator semantics |
| Source-position pass-through (no predecessors) | Raises `FrameworkBugError` (TIER_1) | Impossible in a built DAG; recovery would be defensive programming |
| `creates_tokens=True` × `passes_through_input=True` | Both compose: each emitted token row carries input ∪ added fields | BatchReplicate has both |
| `is_batch_aware=True` × `passes_through_input=True` | Cross-check applies per-row in executor; static contract treats batch as homogeneous | BatchReplicate is batch-aware |
| Pass-through with `output_schema_config=None` (self abstains) | `own = frozenset()`; `result = inherited` | Transform abstains from its own contract but still propagates inherited |

### Decision 3 — Composer parity: fail-closed on probe failure for pass-through transforms

The composer preview (`src/elspeth/web/composer/state.py::_effective_producer_guarantees`) must mirror runtime rejection for known pass-through plugins. When the constructor probe fails:

- **Known pass-through plugin** (plugin class has `passes_through_input=True`): composer returns `frozenset()` and surfaces a **high**-severity warning. Stage 1 then rejects the pipeline because the transform appears to guarantee no fields, failing the sink's required-fields check. This matches runtime rejection.
- **Non-pass-through plugin**: composer returns `raw_guaranteed` and surfaces a **medium**-severity warning. Prior v2 behaviour preserved — draft configs with incomplete plugin options must not crash preview.

The set of known pass-through plugins is re-derived per call from the live plugin manager (not cached at module load). This avoids a staleness bug when plugins are registered after the composer module imports (dynamic packs, test fixture ordering).

## Consequences

### Positive Consequences

- Pass-through pipelines become statically valid where they are runtime-correct. The `c633a36b` explicit-schema workaround becomes unnecessary for any pipeline whose pass-through transforms are annotated.
- Annotation is greppable (`grep -rn 'passes_through_input = True' src/`), discoverable in code review, and auditable — it surfaces in the plugin's class body next to `creates_tokens`, `declared_output_fields`, and `is_batch_aware`.
- Composer preview matches runtime decision exactly for pass-through plugins. Stage 1 rejection is no longer a UI/runtime divergence waiting to surprise a reviewer.
- The opt-in default (`False`) means existing plugins need no changes. Only transforms that are *provably* pass-through are annotated in Phase B.

### Negative Consequences

- Every transform plugin author must now answer "does this transform unconditionally preserve input fields?" This adds cognitive load to plugin authoring. Mitigated by the `False` default — no decision required for non-pass-through transforms, which remain the majority.
- Two recursion implementations exist: `ExecutionGraph._walk_effective_guaranteed_fields` in `core/dag/graph.py` (L1) and `_intersect_predecessor_guarantees` in `web/composer/state.py` (L3). They must be kept in sync. L3→L1 import would be permitted by the layer rules but is deliberately avoided — coupling preview to runtime through a shared helper ties their evolution together and narrows the composer's ability to adapt to new UI states. Mitigated by integration test #36 (YAML round-trip with mutation) which exercises both paths.
- `get_effective_guaranteed_fields` semantic change (from "same as `get_guaranteed_fields`" to "propagation-aware") may surprise existing call sites. Mitigated by Phase A audit (`grep -rn get_effective_guaranteed_fields src/elspeth tests/`), test #15 (semantic split pin), and the method is renamed in spirit — `get_guaranteed_fields` continues to mean "this node alone declares."

### Neutral Consequences

- `passes_through_input` is declared as bare `bool` (not `ClassVar[bool]`) to match adjacent `creates_tokens: bool = False` and `declared_output_fields: frozenset[str] = frozenset()` pattern on `BaseTransform`. Stylistic consistency with existing class-body-attribute declarations takes priority over typing precision.
- A bidirectional annotation-integrity test (`test_non_pass_through_transforms_do_drop_fields`) fires filigree observations when transforms annotated `False` appear to preserve input fields on probe rows. Fire-and-forget observations expire after 14 days; governance is the shared `STRICT_DATE` constant in the redundancy linter (see §Migration in the implementation plan).

## Alternatives Considered

### Alternative 1: Auto-detect via AST inspection of `process()`

**Description:** Walk the Python AST of `process()` and infer whether the transform preserves all input fields.

**Rejected because:** Magical and brittle. AST inspection cannot reason about transitive helpers, `**row` spreads through conditionals, or row-content-dependent paths. Invisible to `grep`, which CLAUDE.md requires for load-bearing constraints. Silent false positives would propagate guarantees the transform does not honour at runtime.

### Alternative 2: Schema-driven derivation (compare input/output schemas at build time)

**Description:** Derive pass-through-ness from whether the output schema is a superset of the input schema.

**Rejected because:** Hides intent from code readers. A transform could be schema-preserving but not actually pass-through (e.g., a transform that renames fields at the schema level but drops the originals at runtime). Schema is a declaration about shape; pass-through is a declaration about runtime behaviour — they can diverge.

### Alternative 3: Conditional pass-through with row-content predicate

**Description:** Let the annotation be a callable `passes_through_input_for(row) -> bool` or a set of conditions.

**Rejected because:** Audit integrity demands the annotation be a class-level invariant, not a per-row decision. A conditional annotation would be impossible to reason about at build time and would let the DAG statically accept pipelines whose runtime behaviour depends on row content.

### Alternative 4: Union across predecessors (not intersection)

**Description:** For diamond topologies, compute the union of predecessor guarantees.

**Rejected because:** Would mask silent field drop in one branch. Diamond topologies naturally have different branches with different shapes; the DAG validator must assume the worst case at the join point, which is the intersection.

### Alternative 5: Preview-optimistic / runtime-authoritative composer

**Description:** Composer returns `raw_guaranteed` on probe failure even for pass-through plugins; runtime is the source of truth.

**Rejected because:** Creates a UI-driven over-acceptance surface. Operators run the composer during pipeline design; letting them ship pipelines that look valid in the UI and fail at runtime is a trust-destroying UX regression. Composer parity is the minimum correctness bar for the preview feature to remain useful.

## References

- Plan: `/home/john/.claude/plans/elspeth-87f6d5dea5-snazzy-swing.md`
- Companion ADR: `ADR-008: Runtime contract cross-check in TransformExecutor`
- Related bug report: `elspeth-87f6d5dea5` (composer/runtime schema-contract divergence)
- Related commits: `d29f8546` (fix three named divergences), `c633a36b` (explicit-schema workaround for examples/deaggregation)
