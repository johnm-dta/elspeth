# Composer Capability Parity Plan 02: Shared Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one public full-pipeline planner that returns an immutable canonical proposal and validates a non-persisted candidate before any controller can approve it.

**Architecture:** `PipelineProposal` is a hash-verified envelope around exact custody-safe `set_pipeline` arguments. A pure candidate builder is extracted from the existing executor, raw inline content is materialized through an audited idempotent custody seam before hashing, and `plan_pipeline()` runs a read-only discovery loop ending in the canonical proposal schema. The production freeform new-pipeline entrypoint is replaced with this planner immediately; there is no runtime architecture switch.

**Tech Stack:** Python 3.12+, dataclasses, Pydantic v2, LiteLLM tool-call protocol, existing composer discovery/dispatch/audit infrastructure, pytest.

---

## File structure

**Create:**
- `src/elspeth/web/composer/pipeline_proposal.py` — proposal/provenance types and domain-separated hashes.
- `src/elspeth/web/composer/pipeline_custody.py` — audited, idempotent inline-content materialization before proposal hashing.
- `src/elspeth/web/composer/planner.py` — shared discovery/terminal proposal loop.
- `src/elspeth/web/composer/pipeline_commit.py` — reusable audited `set_pipeline` dispatch.
- `tests/unit/web/composer/test_pipeline_proposal.py`
- `tests/unit/web/composer/test_set_pipeline_candidate.py`
- `tests/unit/web/composer/test_pipeline_planner.py`
- `tests/integration/web/composer/test_pipeline_planner.py`
- `tests/integration/web/composer/test_pipeline_custody.py`
- `tests/integration/web/composer/test_freeform_planner_entrypoint.py`
- `tests/integration/web/composer/test_freeform_proposal_acceptance.py`

**Modify:**
- `src/elspeth/web/composer/tools/sessions.py` — extract non-persisting candidate construction.
- `src/elspeth/web/composer/tools/__init__.py` — export candidate builder/result.
- `src/elspeth/contracts/blobs.py` — expose atomic idempotent custody reservation through the blob-service protocol.
- `src/elspeth/web/blobs/service.py` — derive/reuse deterministic pending or ready blob records without bypassing the service boundary.
- `src/elspeth/web/sessions/routes/composer/guided.py` — move only the shared content hash; guided replacement is Plan 03.
- `src/elspeth/web/composer/service.py` — route production freeform new-pipeline authoring through the shared planner while preserving incremental tools.
- `src/elspeth/web/sessions/service.py` — create/reuse and settle the one validated-draft proposal row.
- `src/elspeth/web/sessions/protocol.py` — type the canonical proposal binding.
- `src/elspeth/web/sessions/routes/composer/proposals.py` — accept freeform proposals through the shared commit seam.
- `tests/unit/web/blobs/test_service.py` — cover deterministic custody reservation, reuse, races, and fail-closed mismatches.

### Task 1: Add immutable proposal and provenance contracts

**Files:**
- Create: `src/elspeth/web/composer/pipeline_proposal.py`
- Test: `tests/unit/web/composer/test_pipeline_proposal.py`

- [ ] **Step 1: Write failing hash, immutability, and round-trip tests**

```python
from dataclasses import replace

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.pipeline_proposal import (
    PipelineProposal,
    PlannerSurface,
    ProposalProvenance,
    pipeline_draft_hash,
)


def _pipeline() -> dict[str, object]:
    return {
        "source": {"plugin": "csv", "on_success": "rows", "options": {}},
        "nodes": [],
        "edges": [],
        "outputs": [{"sink_name": "rows", "plugin": "json", "options": {}}],
    }


def _proposal() -> PipelineProposal:
    pipeline = _pipeline()
    return PipelineProposal(
        pipeline=pipeline,
        why="Direct source to output.",
        base_composition_version=3,
        base_composition_content_hash="b" * 64,
        draft_hash=pipeline_draft_hash(pipeline),
        reviewed_anchor_hash="a" * 64,
        provenance=ProposalProvenance(
            surface=PlannerSurface.FREEFORM_BIG_BANG,
            skill_hash="s" * 64,
            proposal_schema_version=1,
            model_identifier="model",
            model_version="model-version",
            provider="provider",
            repair_count=0,
        ),
    )


def test_pipeline_proposal_is_deeply_immutable() -> None:
    proposal = _proposal()
    with pytest.raises(TypeError):
        proposal.pipeline["nodes"] = []
    with pytest.raises(AttributeError):
        proposal.pipeline["outputs"].append({})


def test_pipeline_proposal_rejects_draft_hash_mismatch() -> None:
    with pytest.raises(InvariantError, match="draft_hash mismatch"):
        replace(_proposal(), draft_hash="0" * 64)


def test_pipeline_proposal_strict_round_trip() -> None:
    proposal = _proposal()
    assert PipelineProposal.from_dict(proposal.to_dict()) == proposal
```

- [ ] **Step 2: Run the tests and verify red**

Run: `uv run pytest tests/unit/web/composer/test_pipeline_proposal.py -q`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the exact envelope, without topology submodels**

```python
# src/elspeth/web/composer/pipeline_proposal.py
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.core.canonical import stable_hash
from elspeth.web.composer.guided.errors import InvariantError


class PlannerSurface(StrEnum):
    FREEFORM_BIG_BANG = "freeform_big_bang"
    GUIDED_FULL = "guided_full"
    GUIDED_STAGED = "guided_staged"
    TUTORIAL_PROFILE = "tutorial_profile"


@dataclass(frozen=True, slots=True)
class ProposalProvenance:
    surface: PlannerSurface
    skill_hash: str
    proposal_schema_version: int
    model_identifier: str
    model_version: str
    provider: str
    repair_count: int
    supersedes_payload_hash: str | None = None

    def __post_init__(self) -> None:
        if self.proposal_schema_version != 1:
            raise InvariantError("unsupported proposal schema version")
        if type(self.repair_count) is not int or self.repair_count < 0:
            raise InvariantError("repair_count must be a non-negative int")


def pipeline_draft_hash(pipeline: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "schema": "composer.pipeline-proposal.v1",
            "pipeline": deep_thaw(pipeline),
        }
    )


def reviewed_anchor_hash(facts: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            "schema": "guided.reviewed-anchors.v1",
            "facts": deep_thaw(facts),
        }
    )


@dataclass(frozen=True, slots=True)
class PipelineProposal:
    pipeline: Mapping[str, Any]
    why: str
    base_composition_version: int
    base_composition_content_hash: str
    draft_hash: str
    reviewed_anchor_hash: str
    provenance: ProposalProvenance

    def __post_init__(self) -> None:
        freeze_fields(self, "pipeline", "provenance")
        if self.draft_hash != pipeline_draft_hash(self.pipeline):
            raise InvariantError("PipelineProposal draft_hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline": deep_thaw(self.pipeline),
            "why": self.why,
            "base_composition_version": self.base_composition_version,
            "base_composition_content_hash": self.base_composition_content_hash,
            "draft_hash": self.draft_hash,
            "reviewed_anchor_hash": self.reviewed_anchor_hash,
            "provenance": {
                "surface": self.provenance.surface.value,
                "skill_hash": self.provenance.skill_hash,
                "proposal_schema_version": self.provenance.proposal_schema_version,
                "model_identifier": self.provenance.model_identifier,
                "model_version": self.provenance.model_version,
                "provider": self.provenance.provider,
                "repair_count": self.provenance.repair_count,
                "supersedes_payload_hash": self.provenance.supersedes_payload_hash,
            },
        }
```

Implement strict `from_dict()` by checking the exact top-level and provenance
key sets before construction. Do not default missing fields.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest tests/unit/web/composer/test_pipeline_proposal.py -q
git add src/elspeth/web/composer/pipeline_proposal.py tests/unit/web/composer/test_pipeline_proposal.py
git commit -m "feat(composer): add canonical pipeline proposal envelope"
```

Expected: PASS, then commit succeeds.

### Task 2: Share the composition-content hash

**Files:**
- Modify: `src/elspeth/web/composer/pipeline_proposal.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py:129-139`
- Test: `tests/unit/web/composer/test_pipeline_proposal.py`

- [ ] **Step 1: Write the failing exclusion test**

```python
from dataclasses import replace

from elspeth.web.composer.pipeline_proposal import composition_content_hash


def test_composition_content_hash_ignores_version_and_guided_metadata(empty_state) -> None:  # noqa: ANN001
    changed = replace(empty_state, version=99, guided_session=None)
    assert composition_content_hash(empty_state) == composition_content_hash(changed)
```

- [ ] **Step 2: Move the existing helper without changing its preimage**

```python
def composition_content_hash(state: CompositionState) -> str:
    return stable_hash(
        {
            "sources": deep_thaw(state.sources),
            "nodes": [node.to_dict() for node in state.nodes],
            "edges": [edge.to_dict() for edge in state.edges],
            "outputs": [output.to_dict() for output in state.outputs],
            "metadata": state.metadata.to_dict(),
        }
    )
```

Import it into `guided.py` and replace `_composition_content_hash` call sites.
Do not change any route behavior.

- [ ] **Step 3: Run guided hash regressions and commit**

```bash
uv run pytest tests/unit/web/composer/test_pipeline_proposal.py tests/integration/web/composer/guided/test_get_guided.py -q
git add src/elspeth/web/composer/pipeline_proposal.py src/elspeth/web/sessions/routes/composer/guided.py tests/unit/web/composer/test_pipeline_proposal.py
git commit -m "refactor(composer): share pipeline content hash"
```

Expected: PASS.

### Task 3: Extract a side-effect-free canonical candidate builder

**Files:**
- Modify: `src/elspeth/web/composer/tools/sessions.py:207-718`
- Modify: `src/elspeth/web/composer/tools/__init__.py`
- Test: `tests/unit/web/composer/test_set_pipeline_candidate.py`
- Test: `tests/integration/web/composer/test_inline_source_provenance.py`

- [ ] **Step 1: Write the candidate contract tests**

```python
def test_invalid_candidate_never_replaces_current_state(tool_context, empty_state, invalid_args) -> None:  # noqa: ANN001
    before = snapshot_candidate_observables(tool_context)
    candidate = build_set_pipeline_candidate(invalid_args, empty_state, tool_context)
    assert candidate.result.success is True
    assert candidate.result.validation.is_valid is False
    assert snapshot_candidate_observables(tool_context) == before


def test_candidate_matches_executor_for_custody_safe_args(tool_context, empty_state, valid_args) -> None:  # noqa: ANN001
    candidate = build_set_pipeline_candidate(valid_args, empty_state, tool_context)
    executed = _execute_set_pipeline(valid_args, empty_state, tool_context)
    assert composition_content_hash(candidate.result.updated_state) == composition_content_hash(executed.updated_state)


def test_inline_candidate_has_no_storage_side_effects(tool_context, empty_state, inline_args, tmp_path) -> None:  # noqa: ANN001
    before = snapshot_candidate_observables(tool_context)
    candidate = build_set_pipeline_candidate(inline_args, empty_state, tool_context)
    assert candidate.prepared_inline_blob is not None
    assert snapshot_candidate_observables(tool_context) == before
```

`snapshot_candidate_observables()` must cover the current composition row,
session/blob rows, file registry, configured blob directory, audit recorder,
and published in-memory state. Repeat the exact after-state assertion for
invalid, repairable, and inline candidates.

Add a parameterized semantic-failure parity test comparing `success`, message,
validation, and unchanged state between the builder and executor for unknown
plugin, escaping path, bad gate condition, and invalid output options.

- [ ] **Step 2: Run tests and verify red**

Run: `uv run pytest tests/unit/web/composer/test_set_pipeline_candidate.py -q`

Expected: FAIL because `build_set_pipeline_candidate` does not exist.

- [ ] **Step 3: Extract the builder**

```python
@dataclass(frozen=True, slots=True)
class SetPipelineCandidate:
    result: ToolResult
    prepared_inline_blob: _PreparedBlobCreate | None = None

    @property
    def acceptable(self) -> bool:
        return self.result.success and self.result.validation.is_valid


def build_set_pipeline_candidate(
    arguments: dict[str, Any],
    state: CompositionState,
    context: ToolContext,
) -> SetPipelineCandidate:
    """Validate and construct exactly as set_pipeline, without persistence."""
    # Move existing shape/semantic checks, source/node/edge/output construction,
    # CompositionState construction and composition_review_contract_error here.
    # Stop before _persist_prepared_blob_create(). Return the prepared object.
```

Reduce `_execute_set_pipeline()` to:

```python
candidate = build_set_pipeline_candidate(args, state, context)
if not candidate.result.success:
    return candidate.result
if candidate.prepared_inline_blob is not None:
    quota_error = _persist_prepared_blob_create(...)
    if quota_error is not None:
        return _failure_result(state, quota_error)
return _candidate_result_with_existing_blob_payload(candidate)
```

Preserve every existing error string, affected component, validation payload,
and inline provenance field. Candidate construction may allocate a prepared
blob UUID but may not write a file or database row.

- [ ] **Step 4: Run extraction and inline side-effect regressions**

```bash
uv run pytest \
  tests/unit/web/composer/test_set_pipeline_candidate.py \
  tests/unit/web/composer/test_promote_set_pipeline.py \
  tests/integration/web/composer/test_inline_source_provenance.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/tools/__init__.py tests/unit/web/composer/test_set_pipeline_candidate.py tests/integration/web/composer/test_inline_source_provenance.py
git commit -m "refactor(composer): extract pipeline candidate construction"
```

### Task 4: Finalize inline-content custody before proposal construction

**Files:**
- Create: `src/elspeth/web/composer/pipeline_custody.py`
- Create: `tests/integration/web/composer/test_pipeline_custody.py`
- Modify: `src/elspeth/contracts/blobs.py`
- Modify: `src/elspeth/web/blobs/service.py`
- Modify: `tests/unit/web/blobs/test_service.py`
- Modify: `src/elspeth/web/composer/audit.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`

- [ ] **Step 1: Write custody, retry, and crash-point tests**

Submit terminal planner arguments containing `source.inline_blob.content`.
Assert the reviewable proposal contains only canonical `source.blob_id`, its
hash is computed after replacement, and raw content is absent from checkpoints,
LLM audit, tool audit, logs, and validation errors. Retry the same terminal call
and assert it reuses one blob.

Inject crashes after custody reservation, file write, blob-row finalization,
and before proposal checkpoint save. Retry must reuse custody by the durable
reservation and create neither a second blob nor a duplicate custody audit
event.

- [ ] **Step 2: Implement the audited idempotent seam**

```python
async def finalize_proposal_custody(
    pipeline: Mapping[str, Any],
    *,
    session: SessionCustodyContext,
    user_message_id: UUID,
) -> Mapping[str, Any]:
    """Return canonical arguments with raw inline content materialized."""
```

Use the existing session-blob APIs and quota/retention policy. Materialize
before constructing `PipelineProposal`; replace `source.inline_blob` with
`source.blob_id`; never write `source.options.blob_ref`. Redact raw terminal
arguments from persisted completion audit while retaining request/tool/result
hashes and materialization lineage.

Use the existing internal blob lifecycle as the durable custody reservation;
do not create a `composition_proposals` row before the canonical candidate is
valid. Derive a deterministic blob id/path from the session id, stable user
message id, and inline content hash. Insert/reuse a `blobs` row with
`status="pending"`, write and fsync bytes, then finalize that same row as ready.
On restart, reconcile pending blob rows/files before charging quota or writing.
Cover missing, partial, hash-mismatched, cross-session, and orphaned files
explicitly; do not add a second blob store or expose pending custody as an
accept-capable proposal.

Extend `BlobServiceProtocol` and `BlobServiceImpl` with an atomic
`reserve_inline_custody()` operation. It accepts a domain-separated custody
identity containing session id, originating message id, and expected content
hash; the service derives the deterministic blob id/storage path, inserts the
pending row once, and returns an existing pending/ready row only when ownership,
hash, size, MIME type, and provenance match exactly. A primary-key race is
resolved by reading and validating the winner. Cross-session, hash-mismatched,
metadata-mismatched, and error-state reuse fail closed. `pipeline_custody.py`
uses this protocol only and never reads/writes `blobs_table` directly.

Emit `composer.pipeline.custody.total{result}` with the closed result vocabulary
`reserved|reused|finalized|reconcile_failed`; never attach session/blob ids.

- [ ] **Step 3: Run and commit**

```bash
uv run pytest tests/integration/web/composer/test_pipeline_custody.py tests/integration/web/composer/test_inline_source_provenance.py tests/unit/web/blobs/test_service.py -q
git add src/elspeth/contracts/blobs.py src/elspeth/web/blobs/service.py src/elspeth/web/composer/pipeline_custody.py src/elspeth/web/composer/audit.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/integration/web/composer/test_pipeline_custody.py tests/unit/web/blobs/test_service.py
git commit -m "feat(composer): finalize proposal content custody"
```

### Task 5: Implement the shared read-only planner loop

**Files:**
- Create: `src/elspeth/web/composer/planner.py`
- Test: `tests/unit/web/composer/test_pipeline_planner.py`

- [ ] **Step 1: Write failing response-schema and parser tests**

```python
def test_terminal_planner_schema_embeds_canonical_pipeline_schema() -> None:
    terminal = planner_terminal_tool_definition()
    assert terminal["function"]["parameters"]["properties"]["pipeline"] == canonical_set_pipeline_schema()


async def test_plan_pipeline_parses_complete_canonical_payload(fake_completion, state, context) -> None:  # noqa: ANN001
    payload = fork_coalesce_multi_output_args()
    fake_completion.queue_tool_call("emit_pipeline_proposal", {"pipeline": payload, "why": "Parallel assessment."})
    proposal = await plan_pipeline(
        intent="Assess in parallel and merge.",
        state=state,
        reviewed_facts={},
        config=planner_config(PlannerSurface.GUIDED_FULL),
        tool_context=context,
        complete=fake_completion,
    )
    assert deep_thaw(proposal.pipeline) == payload
    assert proposal.provenance.surface is PlannerSurface.GUIDED_FULL
```

Add tests proving discovery calls are limited to the declared read-only set,
malformed terminal calls fail closed, and one invalid candidate receives an
allowlisted structured repair message before the second candidate succeeds.

- [ ] **Step 2: Run tests and verify red**

Run: `uv run pytest tests/unit/web/composer/test_pipeline_planner.py -q`

Expected: FAIL because `planner.py` does not exist.

- [ ] **Step 3: Implement the public API and terminal declaration**

```python
@dataclass(frozen=True, slots=True)
class PlannerConfig:
    surface: PlannerSurface
    system_prompt: str
    skill_hash: str
    model_identifier: str
    model_version: str
    provider: str
    max_repairs: int
    discovery_tool_names: tuple[str, ...]


PlannerCompletion = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    Awaitable[Any],
]


def planner_terminal_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "emit_pipeline_proposal",
            "description": "Return one complete canonical pipeline for review.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pipeline": canonical_set_pipeline_schema(),
                    "why": {"type": "string"},
                },
                "required": ["pipeline", "why"],
                "additionalProperties": False,
            },
        },
    }


async def plan_pipeline(
    *,
    intent: str,
    state: CompositionState,
    reviewed_facts: Mapping[str, Any],
    config: PlannerConfig,
    tool_context: ToolContext,
    complete: PlannerCompletion,
    recorder: BufferingRecorder | None = None,
) -> PipelineProposal:
    """Discover, propose, and candidate-validate a complete canonical pipeline."""
```

The loop must:

1. advertise only `get_discovery_tool_definitions(config.discovery_tool_names)`
   plus the terminal tool;
2. dispatch discovery with `execute_tool()` and reject any non-discovery name;
3. parse terminal arguments with a strict local Pydantic envelope whose
   `pipeline` field is then validated by `SetPipelineArgumentsModel`;
4. call `finalize_proposal_custody()` with the stable originating user-message
   id; its durable key also includes the inline content hash;
5. deep-freeze and hash only the returned custody-safe arguments;
6. call `build_set_pipeline_candidate()`;
7. return only when `candidate.acceptable` is true;
8. feed allowlisted validation fields back for at most `max_repairs` attempts;
9. record the actual per-call messages/tools hashes through the existing LLM
   audit helper.

- [ ] **Step 4: Run unit tests and commit**

```bash
uv run pytest tests/unit/web/composer/test_pipeline_planner.py -q
git add src/elspeth/web/composer/planner.py src/elspeth/web/composer/pipeline_custody.py tests/unit/web/composer/test_pipeline_planner.py
git commit -m "feat(composer): add shared canonical pipeline planner"
```

Expected: PASS.

### Task 6: Add the reusable audited commit seam

**Files:**
- Create: `src/elspeth/web/composer/pipeline_commit.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Test: `tests/integration/web/composer/test_pipeline_planner.py`

- [ ] **Step 1: Write failing exact-arguments and audit tests**

```python
async def test_commit_dispatches_exact_proposal_arguments(proposal, commit_context, monkeypatch) -> None:  # noqa: ANN001
    observed: list[dict[str, Any]] = []

    def capture(name, arguments, *args, **kwargs):  # noqa: ANN001
        assert name == "set_pipeline"
        observed.append(arguments)
        return execute_tool(name, arguments, *args, **kwargs)

    monkeypatch.setattr("elspeth.web.composer.pipeline_commit.execute_tool", capture)
    result = await commit_pipeline_proposal(proposal=proposal, context=commit_context)
    assert result.validation.is_valid
    assert observed == [deep_thaw(proposal.pipeline)]
    assert commit_context.recorder.invocations[-1].tool_name == "set_pipeline"
```

Add stale base-content and reviewed-anchor mismatch tests; both must return a
typed conflict before dispatch and leave the recorder/state unchanged.

- [ ] **Step 2: Implement the seam using existing dispatch infrastructure**

```python
@dataclass(frozen=True, slots=True)
class PipelineCommitContext:
    state: CompositionState
    tool_context: ToolContext
    recorder: BufferingRecorder
    actor: str
    reviewed_facts: Mapping[str, Any]
    publish: Callable[[ToolResult, CompositionProposalRecord], Awaitable[None]]


async def commit_pipeline_proposal(
    *,
    proposal: PipelineProposal,
    context: PipelineCommitContext,
) -> ToolResult:
    if proposal.base_composition_content_hash != composition_content_hash(context.state):
        raise PipelineProposalConflict("base composition content changed")
    if proposal.reviewed_anchor_hash != reviewed_anchor_hash(context.reviewed_facts):
        raise PipelineProposalConflict("reviewed facts changed")

    arguments = deep_thaw(proposal.pipeline)
    candidate = build_set_pipeline_candidate(arguments, context.state, context.tool_context)
    if not candidate.acceptable:
        raise PipelineProposalValidationError.from_summary(candidate.result.validation)

    audit_arguments = redact_tool_call_arguments(
        "set_pipeline",
        arguments,
        telemetry=NoopRedactionTelemetry(),
    )
    audit = begin_dispatch(
        tool_call_id=f"proposal-{proposal.draft_hash}",
        tool_name="set_pipeline",
        arguments=audit_arguments,
        version_before=context.state.version,
        actor=context.actor,
    )
    binding = await get_or_create_proposal_binding(
        session_id=context.tool_context.session_id,
        deterministic_tool_call_id=f"proposal-{proposal.draft_hash}",
        proposal=proposal,
        redacted_arguments=audit_arguments,
    )
    async def dispatch_and_verify() -> ToolResult:
        result = cast(
            ToolResult,
            await run_sync_in_worker(
                execute_tool,
                "set_pipeline",
                arguments,
                context.state,
                context.tool_context.catalog,
                # pass every field from ToolContext exactly as tool_batch does
            ),
        )
        if not result.validation.is_valid:
            raise PipelineProposalValidationError.from_summary(result.validation)
        if composition_content_hash(candidate.result.updated_state) != composition_content_hash(result.updated_state):
            raise PipelineProposalCommitMismatch("validated candidate differs from dispatched result")
        return result

    outcome = await settle_pipeline_commit_with_deferred_cancellation(
        recorder=context.recorder,
        audit=audit,
        binding=binding,
        do_dispatch=dispatch_and_verify,
        version_after_provider=lambda result: result.updated_state.version,
        arg_error_payload_factory=lambda exc: arg_error_payload(exc, "set_pipeline"),
        publish=context.publish,
    )
    return cast(ToolResult, outcome.result)
```

Factor a shared `execute_tool_from_context()` helper if copying ToolContext
kwargs would otherwise drift from `tool_batch.py`; both call sites must use it.
Validation and content-hash comparison belong inside `do_dispatch`, before the
settlement helper may call `publish`. A mismatch settles the audit/binding as
failed, publishes no current state, and remains idempotent on retry.

Create exactly one `composition_proposals` row after candidate validation, when
the canonical draft becomes reviewable. Its deterministic `tool_call_id` is
derived from `proposal.draft_hash`; the same `(session_id, tool_call_id)` row is
used for review, replay, acceptance, commit, and retry. Keep exact custody-safe
arguments only in its private replay column and redacted arguments in
audit/public surfaces. Assert failed/repaired candidates create zero public
proposal rows and an accepted draft creates exactly one. Generalize the existing deferred-cancellation
critical section so dispatch, audit, proposal acceptance, current-state
publication, answer, and stage advance settle together before cancellation is
re-raised.

Add crash injection after audited dispatch but before current-state/checkpoint
publication. A retry must return the recorded commit or finish publication
without repeating `set_pipeline`. Assert exactly one blob, tool invocation,
commit binding, current-state version, answer event, and stage-advance event at
every crash point, including cancellation while the worker is still running.
Emit `composer.pipeline.commit.total{surface,result}` and
`composer.pipeline.commit_mismatch_total`; result is closed and identifiers are
never metric attributes.

- [ ] **Step 3: Run integration tests and commit**

```bash
uv run pytest tests/integration/web/composer/test_pipeline_planner.py tests/unit/web/composer/test_dispatch_arms_characterization.py -q
git add src/elspeth/web/composer/pipeline_commit.py src/elspeth/web/composer/tool_batch.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py tests/integration/web/composer/test_pipeline_planner.py
git commit -m "feat(composer): share audited pipeline proposal commit"
```

Expected: PASS.

### Task 7: Replace the production freeform new-pipeline entrypoint

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/sessions/routes/composer/proposals.py`
- Create: `tests/integration/web/composer/test_freeform_planner_entrypoint.py`
- Create: `tests/integration/web/composer/test_freeform_proposal_acceptance.py`

- [ ] **Step 1: Write a production-entrypoint test**

Call `ComposerServiceImpl.compose()` through the normal compose route with an
empty composition and a plain-English multi-source/fork/coalesce request. Spy
on `plan_pipeline()` and `commit_pipeline_proposal()` and assert both are called
once. Assert the response is a reviewable proposal before acceptance and the
accepted state is graph-isomorphic to the fixture.

- [ ] **Step 2: Preserve incremental authoring**

With a non-empty composition and an incremental edit request, assert existing
mutation/discovery tools remain available and the service does not force a
full replacement proposal unless the operator asks to replace/rebuild the
pipeline.

- [ ] **Step 3: Route production review acceptance through the shared seam**

Drive the complete route journey: compose English intent -> validated
reviewable proposal -> list/read the single pending proposal -> accept through
`routes/composer/proposals.py` -> audited canonical commit -> current-state
publication. Capture the executor arguments and assert exact equality with the
proposal. Candidate failure, commit mismatch, cancellation, and retry must not
call the route's old direct `execute_tool` arm and must not create another
public proposal row.

- [ ] **Step 4: Replace the old new-build loop and add an architecture guard**

Route every new/rebuild request through `plan_pipeline()` with no old/new
architecture setting. Delete the superseded new-build loop. Add a
static/behavioral test that no production new-pipeline planning loop exists
outside `plan_pipeline()`.

- [ ] **Step 5: Run and commit**

```bash
uv run pytest tests/integration/web/composer/test_freeform_planner_entrypoint.py tests/integration/web/composer/test_freeform_proposal_acceptance.py tests/unit/web/composer/test_compose_service_structure.py -q
git add src/elspeth/web/composer/service.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/routes/composer/proposals.py tests/integration/web/composer/test_freeform_planner_entrypoint.py tests/integration/web/composer/test_freeform_proposal_acceptance.py
git commit -m "feat(composer): route freeform new builds through the shared planner"
```

### Task 8: Verify Plan 02 before the guided replacement slice

- [ ] **Step 1: Run planner and unaffected-guided characterization tests**

Run:

```bash
uv run pytest \
  tests/unit/web/composer/test_pipeline_proposal.py \
  tests/unit/web/composer/test_set_pipeline_candidate.py \
  tests/unit/web/composer/test_pipeline_planner.py \
  tests/unit/web/blobs/test_service.py \
  tests/integration/web/composer/test_pipeline_custody.py \
  tests/integration/web/composer/test_pipeline_planner.py \
  tests/integration/web/composer/test_freeform_planner_entrypoint.py \
  tests/integration/web/composer/test_freeform_proposal_acceptance.py \
  tests/integration/web/composer/guided/test_chain_solver.py \
  tests/integration/web/composer/guided/test_wire_dispatch.py -q
```

Expected: PASS. Guided replacement has not started yet; this is sequencing, not
a supported compatibility mode.

- [ ] **Step 2: Run static checks**

```bash
uv run ruff check src/elspeth/web/composer/pipeline_proposal.py src/elspeth/web/composer/pipeline_custody.py src/elspeth/web/composer/planner.py src/elspeth/web/composer/pipeline_commit.py src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/service.py
uv run mypy src/elspeth/web/composer/pipeline_proposal.py src/elspeth/web/composer/pipeline_custody.py src/elspeth/web/composer/planner.py src/elspeth/web/composer/pipeline_commit.py
git diff --check
```

Expected: all exit 0.

- [ ] **Step 3: Record Plan 02 evidence and commit the ledger update**

```bash
git add docs/superpowers/plans/2026-07-13-composer-capability-parity-implementation-plan.md
git commit -m "docs: record shared composer planner evidence"
```
