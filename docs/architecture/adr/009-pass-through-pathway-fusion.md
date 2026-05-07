# ADR-009: Pass-through pathway fusion and runtime-VAL completeness

**Date:** 2026-04-19
**Status:** Accepted
**Supersedes:** ADR-007 §Decision 1 ("unconditional on every row"), §Negative Consequences #2 (duplicated walkers), §Neutral Consequences line 83 (invariant test placeholder); ADR-008 §Decision scope statement ("cross-check applies per-row in executor")
**Tags:** dag, schema-contract, propagation, runtime-validation, audit-integrity, composer-parity

---

> **Amended by ADR-010 (Declaration-trust framework, 2026-04-19).**
> Normative in this ADR: all four Clauses for `passes_through_input`.
> Extended: §Clause 4 invariant-harness design now covers ALL registered contracts via the ADR-010 `negative_example` requirement, not only `passes_through_input`.

## Context

ADR-007 (pass-through contract propagation) and ADR-008 (runtime contract cross-check) shipped as a partial landing. Two limitations were documented at landing time and are now closed:

1. **Duplicated walkers (ADR-007 §Negative Consequences #2).** `core/dag/graph.py::_walk_effective_guaranteed_fields` and `web/composer/state.py::_effective_producer_guarantees` encoded the same ADR-007 propagation semantics in two independent implementations. Drift was mitigated by one integration test — a Shifting-the-Burden mitigation that catches drift after it is introduced rather than preventing it structurally.
2. **Single-path runtime cross-check (ADR-008 scope).** The runtime cross-check ran only on `TransformExecutor.execute_transform` (single-token). The batch-aware flush path (`processor._process_batch_aggregation_node` and `handle_timeout_flush`) trusted the static `passes_through_input=True` annotation without verifying it. A mis-annotated batch-aware transform (e.g., `BatchReplicate`) could silently drop fields from emitted rows, producing audit records that attested to pass-through preservation when in fact it had not occurred — evidence tampering under the Tier 1 trust model.

Two governance gaps were also present:

3. **ADR-007 contradicted the code.** ADR-007 §Decision 1 stated the contract was unconditional on every row. `engine/executors/transform.py::_cross_check_pass_through` silently exempted empty emission (`result.row is None and result.rows is None → return`). No ADR clause authorised the carve-out.
4. **Vapourware invariant test.** ADR-007 §Neutral Consequences line 83 named `test_non_pass_through_transforms_do_drop_fields` as a governance mechanism. The file did not exist.

## Decision

### Clause 1 — Shared propagation primitives (closes ADR-007 §Negative Consequences #2)

Two pure primitives become the canonical source of truth for the ADR-007 propagation rule:

- **`SchemaConfig.participates_in_propagation: bool`** (`src/elspeth/contracts/schema.py`). The participation predicate. Returns `self.has_effective_guarantees`. Today this is an alias; the named property exists so future changes to propagation participation (e.g., Track 2's `can_drop_rows` declaration) land on this property alone without fragmenting call sites.
- **`compose_propagation(self_fields, predecessor_guarantees)`** (`src/elspeth/contracts/guarantee_propagation.py`). The aggregation rule. Pure, stateless, no graph dependency. Entries in `predecessor_guarantees` are `frozenset[str]` for participating predecessors and `None` for abstaining ones.

Both walkers now consult these primitives:

- `graph.py::_walk_effective_guaranteed_fields` — runtime DAG walker (L1), multi-predecessor.
- `web/composer/state.py::_effective_producer_guarantees` — composer preview walker (L3), single-upstream via `_walk_to_real_producer` (coalesce absorbs fan-in via pre-computed output).

Traversal remains separate: the two walkers legitimately traverse different graph views. Merging traversal would pollute layers without eliminating duplication. The honest shareable units are the aggregation rule and the predicate, which the two walkers now share.

`graph.py::_predecessor_declares_guarantees` (whose name contradicted its implementation — it said "declares" but consulted `has_effective_guarantees`) is deleted. Its sole call site reads `schema.participates_in_propagation`.

### Clause 2 — Runtime cross-check on the batch-aware path (closes ADR-008 scope gap)

`engine/executors/pass_through.py::verify_pass_through` is the single cross-check callable. Module-level rather than class-method: both call sites (single-token executor and batch-flush processor) import the same function and share a module-level OpenTelemetry violations counter.

Invoked from two sites:

- `engine/executors/transform.py::TransformExecutor._cross_check_pass_through` — thin wrapper that applies single-token boundary assertions and delegates.
- `engine/processor.py::RowProcessor._cross_check_flush_output` — batch aggregation flush path.

**Batch-mode semantics.** Two output modes require distinct handling:

- **`OutputMode.TRANSFORM` (batch-homogeneous intersection).** `input_fields` is the intersection of all buffered input contracts (ADR-007 table line 53). Every emitted row must preserve the intersection — the weakest shared guarantee across the batch. A transform claiming `passes_through_input=True` must preserve what every input contributed.
- **`OutputMode.PASSTHROUGH` (1:1 pairing).** Tokens are 1:1 with outputs (routing enforces the count match). Each `(input_token, output_row)` pair is checked independently using that specific input token's contract fields. Using the batch intersection for passthrough would create a correctness hole on heterogeneous batches — a field present on only one input token could be silently dropped on its corresponding output token.

**Call-site placement (critical).** `_cross_check_flush_output` MUST run BEFORE `_emit_transform_completed` and the `_route_*` methods. A failed cross-check must not follow a COMPLETED (telemetry) or CONSUMED_IN_BATCH (Landscape) terminal-state emission on any token, or the audit trail would contain both terminal states for the same token — violating CLAUDE.md's "every row reaches exactly one terminal state" invariant.

**Violation recording.** On `PassThroughContractViolation`, `_record_flush_violation` writes per-token FAILED audit entries for every buffered token. The per-token context payload is rebuilt inside the loop so `$.context.token_id` matches each row's own token, not the triggering token's — triage queries of the form `WHERE exception_type = 'PassThroughContractViolation'` expect every affected token to resolve to its own identifier.

**Audit-write failure.** If `record_token_outcome` raises mid-loop, `_record_flush_violation` raises `AuditIntegrityError`. The audit trail is incomplete; the operator learns about the write failure loudly rather than silently. `AuditIntegrityError` is registered in `TIER_1_ERRORS` — it propagates without on_error absorption. The primary `PassThroughContractViolation` is preserved via Python's implicit `__context__` (the raise sits inside an `except PassThroughContractViolation` block in the caller), and the error message references the original violation.

### Clause 3 — Empty-emission carve-out (supersedes ADR-007 §Decision 1)

`passes_through_input=True` is compatible with emitting zero rows. An empty emission drops no fields because there are no rows to examine; `verify_pass_through` returns early on `not emitted_rows`. This supersedes ADR-007 §Decision 1's language "the contract is unconditional on every row" — the contract is unconditional on every **emitted** row. Filter-style transforms (0-or-1 emission) may annotate `passes_through_input=True` without violating the contract, provided every row they do emit preserves input fields.

**Track 2 SLA trigger.** The carve-out is not semantically airtight: a filter that always emits zero rows can carry the annotation without ever being checked. Track 2 will introduce a separate `can_drop_rows: bool = False` declaration; transforms with `can_drop_rows=False` emitting zero rows will raise a new `UnexpectedEmptyEmission` violation. Track 1 does not ship `can_drop_rows` — the declaration is part of a wider framework pattern that deserves its own ADR with multiple concrete declarations to inform its shape.

**Hard trigger:** Track 2's `can_drop_rows` declaration MUST land within 90 days of Track 1 merge, OR upon registration of a second `passes_through_input=True` transform with external-call dependencies (LLM, HTTP, DB), whichever is sooner. File the trigger as a filigree dependency on the Track 2 epic; the Eroding Goals risk is real and the SLA is the safeguard.

### Clause 4 — Invariant harness (delivers ADR-007 §Neutral Consequences line 83)

`tests/invariants/` is the governance home for declarative-annotation tests. Forward invariant (`test_annotated_transforms_preserve_input_fields`) discovers every registered `passes_through_input=True` transform and asserts on Hypothesis-generated probe rows that every emitted row preserves every input field. Backward invariant (`test_non_pass_through_transforms_do_drop_fields`) fails CI when a non-annotated transform that opted into probing (i.e., implements `probe_config()`) preserves every input field across 15 scalar probes — remediation is either adding the annotation or teaching `probe_config()` to return a shape that exercises the drop path. Non-annotated transforms without `probe_config()` are skipped: the backward invariant only gates transforms that explicitly opted into probing.

Side-effectful governance channels (e.g., firing filigree observations from pytest) are rejected: pytest's contract is pass/fail, and a shell-out to a best-effort CLI suppresses errors behind `check=False` — the "diagnostic, does not fail CI" design documented in an earlier draft of this ADR was governance theatre the harness would never actually exercise. Converting the backward invariant to a hard failure gives it real teeth without creating false positives on the currently-registered plugin set (no non-annotated transform implements `probe_config()` today, so the failure fires only on deliberate future declarations).

Probe instantiation uses a new `BaseTransform.probe_config()` classmethod. Every `passes_through_input=True` transform MUST implement `probe_config()` to declare how it should be instantiated in isolation. A companion `test_harness_skip_rate_budget` asserts `skip_rate ≤ 25%` across the annotated plugin set; Track 2 additions that slip the budget must implement `probe_config()` rather than raising the threshold.

## Consequences

**Positive:**

- Audit integrity restored for batch-aware `passes_through_input=True` transforms. `BatchReplicate`'s contract is now verified at runtime on every emitted row, and mis-annotations on future batch-aware pass-through transforms will be caught.
- Walker duplication eliminated structurally. Both walkers share the aggregation rule and participation predicate; drift is impossible at those layers.
- `graph.py`'s `_predecessor_declares_guarantees` function — whose name contradicted its implementation — is removed.
- ADR-007 language is now honest about filter semantics.
- Template proven against one declaration. Track 2 has a concrete reference implementation for generalizing the pattern.

**Negative:**

- Small per-row overhead on the batch flush path (intersection computation across buffered input contracts plus one `verify_pass_through` call per emitted row). Bounded by the existing NFR benchmark in `tests/performance/benchmarks/test_cross_check_overhead.py`.
- The empty-emission carve-out (Clause 3) is governance-stable but not semantically airtight until Track 2 lands `can_drop_rows`. The 90-day SLA trigger is the safeguard.
- `_output_schema_config` is accessed as a private-by-convention attribute in `_cross_check_flush_output`. The coupling matches the existing pattern in `_cross_check_pass_through` and is named here as known coupling that Track 2 may formalize as part of framework generalization.

**Neutral:**

- Filigree observations with 14-day TTL are the governance channel for backward-invariant signals. Named owner must be specified in the filigree issue when an observation is promoted.
- `pytest.skip` for unprobeable transforms is bounded by the skip-rate budget test; coverage gaps surface loudly.

## Alternatives Considered

1. **`DeclarationContract` protocol up front** — rejected. Designing a protocol against one concrete case is premature abstraction; Track 2 designs the protocol with multiple concrete cases in hand (rule-of-three).
2. **`TransformResult.output_to_input_indices` for precise batch attribution** — rejected. Expanding the plugin API is outside Track 1 scope; batch-homogeneous intersection is ADR-007-sanctioned and catches realistic mis-annotations.
3. **Introduce `can_drop_rows` in Track 1** — rejected. Pulls framework-pattern design in. Forward reference via Clause 3 + Track 2 SLA is the honest deferral.
4. **Amend ADR-007/008 in place** — rejected. Accepted ADRs are audit-trail artifacts; editing their text destroys the record of what was known at acceptance time. ADR-009 supersedes with named clause references; amendment banners point readers from ADR-007/008 to the current authoritative statement.
5. **Single shared walker** — rejected. The two walkers traverse different graph views (runtime DAG vs. composer producer-graph). Unifying traversal would either pollute L1 with composer concerns or force the composer to walk the full DAG. Sharing the aggregation rule and predicate is the honest factoring; sharing the walker is over-reach.

## References

- [ADR-007: Pass-through contract propagation](007-pass-through-contract-propagation.md) — amended by this ADR § Clauses 1, 3, 4.
- [ADR-008: Runtime contract cross-check](008-runtime-contract-cross-check.md) — amended by this ADR § Clause 2.
- Track 2 epic: TBD (follow-up filigree epic with SLA dependency linking Track 1 merge date).
- CLAUDE.md §Three-Tier Trust Model, §Plugin Ownership, §"No Legacy Code Policy".
- Review artefacts: `/tmp/elspeth-track1/plan.md`, `/tmp/elspeth-track1/synthesis.md`, `/tmp/elspeth-track1/spec.md`.
