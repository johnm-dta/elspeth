# Composer Capability Parity Plan 01: Candidate and Approval Seam

**Goal:** Prove one side-effect-free `set_pipeline` candidate boundary and stop
freeform explicit approval from persisting semantically invalid pipeline
proposals.

**Architecture:** Extract the validation and construction portion of
`_execute_set_pipeline()` without changing its error strings, policy/profile
behavior, inline-blob settlement, or public tool result. Both auto-commit and
explicit-approval paths call this boundary; only a valid candidate may become a
durable proposal.

**Prerequisite:** Work from the baseline named in the master plan. Do not change
session or landscape schema constants in this phase.

## Task 1: Characterize the current executor boundary

**Files:**

- Create: `tests/unit/web/composer/test_set_pipeline_candidate.py`
- Modify only if a fixture must be shared: `tests/unit/web/composer/conftest.py`
- Reference: `src/elspeth/web/composer/tools/sessions.py`
- Reference: `src/elspeth/web/composer/tool_batch.py`

- [ ] Add table-driven characterization for a valid linear graph, named
  multi-source queue, fork/coalesce, gate, aggregation, structured LLM, and
  multi-output graph. Capture `success`, validation, affected nodes, normalized
  composition content, and public result data from `_execute_set_pipeline()`.
- [ ] Add semantic failures for unknown/blocked plugins, invalid options,
  escaping paths, invalid gate conditions, manual `blob_ref`, credential-policy
  failure, profile-validation failure, and stale interpretation review.
- [ ] Add an inline source case that records blob rows, registry rows, files,
  quota, audit invocations, and state before and after execution.

Run:

```bash
uv run pytest tests/unit/web/composer/test_set_pipeline_candidate.py -q
```

Expected before extraction: PASS against `_execute_set_pipeline()`; this is the
behavioral baseline, not a snapshot of private implementation structure.

## Task 2: Extract candidate construction

**Files:**

- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/tools/__init__.py`
- Test: `tests/unit/web/composer/test_set_pipeline_candidate.py`
- Test: `tests/integration/web/composer/test_inline_source_provenance.py`

- [ ] First add failing imports and assertions for this contract:

```python
@dataclass(frozen=True, slots=True)
class SetPipelineCandidate:
    result: ToolResult
    prepared_inline_blob: _PreparedBlobCreate | None

    @property
    def acceptable(self) -> bool:
        return self.result.success and self.result.validation is not None and self.result.validation.is_valid


def build_set_pipeline_candidate(
    arguments: Mapping[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> SetPipelineCandidate:
    """Validate and construct a pipeline without publishing or persisting it."""
```

- [ ] Verify the new tests fail because the symbol does not exist.
- [ ] Move Pydantic validation, semantic checks, source/node/edge/output
  construction, profile-aware validation, review reconciliation, and final
  `ToolResult` construction into the builder.
- [ ] Keep `_prepare_blob_create()` in candidate construction, but leave
  `_persist_prepared_blob_create()` and result-payload settlement in
  `_execute_set_pipeline()`.
- [ ] Reduce `_execute_set_pipeline()` to: build candidate; return failures
  unchanged; persist a prepared blob when present; return the settled result.
- [ ] Assert candidate construction changes none of the observables captured in
  Task 1, including for invalid and inline cases.
- [ ] Assert candidate content equals executor content for custody-safe
  arguments. A difference in sources, nodes, edges, outputs, metadata,
  validation, or review reconciliation is a blocking defect.

Run:

```bash
uv run pytest \
  tests/unit/web/composer/test_set_pipeline_candidate.py \
  tests/unit/web/composer/test_promote_set_pipeline.py \
  tests/integration/web/composer/test_inline_source_provenance.py -q
```

Expected: PASS; inline persistence happens once in the executor and never in
the candidate builder.

## Task 3: Validate before creating a freeform proposal

**Files:**

- Modify: `src/elspeth/web/composer/tool_batch.py`
- Modify: `src/elspeth/web/composer/tools/__init__.py`
- Create: `tests/integration/web/composer/test_freeform_proposal_prevalidation.py`
- Modify: `tests/unit/web/sessions/test_composer_proposals.py`

- [ ] Write a failing explicit-approval test in which the model emits a
  schema-valid but semantically invalid `set_pipeline` call. Assert there is no
  `composition_proposals` row, the normal structured tool failure is appended
  to the model conversation, and the next model attempt may repair it.
- [ ] Add valid and invalid non-`set_pipeline` mutation cases proving their
  current approval behavior is unchanged.
- [ ] In the explicit-approval intercept, call
  `build_set_pipeline_candidate()` only for `tool_name == "set_pipeline"`.
  When unacceptable, route its existing `ToolResult` through the normal tool
  outcome/audit/repair path and do not call `create_composition_proposal()`.
- [ ] When acceptable, continue using the existing redaction, summary,
  provenance, base-state, and proposal service. Do not persist the candidate
  state and do not persist the prepared inline blob in this phase.
- [ ] Keep auto-commit dispatch through `_execute_set_pipeline()` and prove it
  traverses the same candidate builder once.

Run:

```bash
uv run pytest \
  tests/integration/web/composer/test_freeform_proposal_prevalidation.py \
  tests/unit/web/sessions/test_composer_proposals.py \
  tests/unit/web/composer/test_dispatch_arms_characterization.py -q
```

Expected: PASS; invalid `set_pipeline` calls create zero proposal rows and valid
ones create exactly one pending row without changing composition state.

## Task 4: Verify the tracer slice

Run:

```bash
uv run pytest tests/unit/web/composer tests/integration/web/composer -q
uv run pytest tests/unit/web/sessions/test_routes.py -q
uv run ruff check src/elspeth/web/composer tests/unit/web/composer tests/integration/web/composer
uv run mypy src/elspeth/web/composer
git diff --check
```

Expected: all commands exit 0.

**Definition of done:** The production executor and explicit-approval path share
one candidate implementation, invalid full-pipeline drafts re-enter the repair
loop before human review, and this phase changes no persisted schema or current
non-pipeline approval behavior.
