# Remaining Transform Invariant Migration Implementation Plan

> **Scope reduction note (2026-05-06):** ADR-020 retired both
> `AzureBatchLLMTransform` and `OpenRouterBatchLLMTransform`. Sections of
> this plan that name them as deferred or in-scope migration targets are
> obsolete; the batch-LLM tranche described as "follow-on" no longer
> exists. Other transform migrations covered by this plan
> (`FieldMapper`, `JSONExplode`, `WebScrapeTransform`,
> `AzureContentSafety`, `AzurePromptShield`, etc.) remain in scope.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the remaining eligible transform plugins in epic `elspeth-be398f0bcb` into truthful ADR-009/ADR-010 invariant coverage by classifying each transform honestly, adding hermetic probe support, eliminating the current blind-skip surface at both probe instantiation and probe execution time, explicitly carving out the batch-LLM blocker that still needs separate contract work, and leaving a bounded spike artifact that defines the follow-on batch-LLM tranche.

**Architecture:** Keep the invariant harness generic and dumb: it should ask transforms for probe config, probe rows, and a probe execution path, but it should not learn every transport stack or concurrency model itself. Local shape-changing transforms stay non-pass-through and opt into backward probes that exercise real field-drop paths; success-path enrichers and validators that preserve all input fields become `passes_through_input=True` and provide deterministic forward probes using local doubles or internal no-network execution seams. Batch-aware transforms only join that lane when the probe seam exercises the real runtime path rather than fabricating a synthetic success row. Scope-level coverage in this tranche must prove more than instantiation: for every in-scope transform it must instantiate the class, shape the canonical probe rows, execute the canonical forward/backward hook successfully, and fail hard if any of those phases regress.

**Tech Stack:** Python 3.12, pytest, Hypothesis, respx/httpx mocks, ELSPETH plugin base classes, BatchTransformMixin, Filigree issue traceability, ADR-009/ADR-010 contract governance.

---

## Context Background

### Scope check

This is one plan, not four separate plans, because the work is one subsystem:
the ADR-009 invariant harness plus the remaining transform families that feed
it. The cross-cutting prerequisite is real: today the harness calls
`process()` directly and still converts execution-time `TypeError` /
`AttributeError` probe rejection into `pytest.skip()`, which means
`BatchTransformMixin` transforms such as `AzureContentSafety`,
`AzurePromptShield`, and `LLMTransform` cannot join truthful forward coverage
without a transform-owned probe execution seam and a hard-fail coverage check
that exercises that seam.

### Repo reality to preserve

- `BatchStats`, `FieldMapper`, and `JSONExplode` are genuinely shape-changing today.
  Their success rows do not preserve all input fields, so this plan keeps them
  non-pass-through and adds backward probes.
- `BatchReplicate` is already the model for the non-pass-through lane:
  `passes_through_input = False`, `probe_config()`, and a custom
  `backward_invariant_probe_rows()` are already implemented in
  `src/elspeth/plugins/transforms/batch_replicate.py`.
- `tests/invariants/test_pass_through_invariants.py` still skips on
  execution-time `TypeError` / `AttributeError` during probe invocation today.
  This plan must delete that skip path rather than only proving probe
  instantiation.
- `PluginManager.create_transform()` instantiates transform classes positionally
  (`plugin_cls(config)`) in the live runtime. The invariant harness must match
  that contract instead of imposing a stricter `cls(config=...)` keyword-only
  constructor assumption that some real transforms do not implement.
- `WebScrapeTransform`, `AzureContentSafety`, `AzurePromptShield`,
  `LLMTransform`, and `RAGRetrievalTransform` preserve input fields on their
  successful emission paths in the current repo. The migration work is not
  “make them preserve fields”; it is “declare that truthfully and prove it
  hermetically.”
- `BaseAzureSafetyTransform` is only a shared infrastructure base today. The
  truthful `passes_through_input=True` declaration should live on
  `AzureContentSafety` and `AzurePromptShield`, not on the base class, so a
  future Azure safety subclass cannot inherit a pass-through promise by
  accident.
- `AzureBatchLLMTransform` and `OpenRouterBatchLLMTransform` are NOT eligible
  for truthful `passes_through_input=True` annotation in this tranche. Their
  fully successful rows preserve input fields, but their mixed-success batch
  path currently emits partial-error rows that omit some currently declared
  guaranteed fields (notably `*_usage` and `*_model`). This plan must not hide
  that mismatch behind fabricated probe hooks that bypass the real batch
  assembly path.
- `RAGRetrievalConfig` validates `provider` names against the lazy `PROVIDERS`
  registry. `chroma` is gated behind the optional `chromadb` extra in this repo,
  so mandatory invariant coverage cannot depend on `provider: "chroma"` being
  available in the test environment.
- `LLMTransform.close()` and `RAGRetrievalTransform.close()` unconditionally
  close any provider left attached to the instance. Probe doubles in this
  tranche must therefore implement `close()` and restore the original provider
  state before returning so the transform lifecycle remains safe after probe
  execution.
- External-call transforms MUST NOT require live credentials, DNS, vendor
  access, or nondeterministic provider behavior in invariant tests. The probe
  path must stay CI-safe and local-only.

### Explicit deferral

- This tranche intentionally does **not** annotate `AzureBatchLLMTransform` or
  `OpenRouterBatchLLMTransform` as pass-through.
- This plan does include a bounded spike for those classes, but the spike is
  design/contract work only. It must not change their runtime behavior or
  pretend the blocker is already fixed.
- Do **not** add hermetic hooks that manually construct a compliant success row
  for those classes. That would only prove the hook can fabricate a
  pass-through-shaped row, not that the real `_process_batch()` /
  checkpoint/result-assembly path is truthful.
- Follow-on work for the batch LLM lane must first resolve the current
  declaration/runtime mismatch on mixed-success outputs, then add a no-network
  seam that exercises the real batch execution path.

### Assumption

The Filigree epic and its child tasks are the approved spec for this plan. Do
not pause to write a separate design doc unless repo reality contradicts the
issue descriptions in a materially new way.

---

## File Structure

### Cross-cutting files

- Modify: `src/elspeth/plugins/infrastructure/base.py`
- Modify: `docs/contracts/plugin-protocol.md`
- Modify: `tests/invariants/test_pass_through_invariants.py`
- Create: `tests/unit/plugins/test_invariant_probe_execution.py`
- Modify: `tests/unit/plugins/transforms/test_forward_invariant_probes.py`
- Create: `tests/unit/plugins/transforms/test_backward_invariant_probes.py`
- Create: `tests/invariants/test_transform_probe_coverage.py`

### Local shape-changing files

- Modify: `src/elspeth/plugins/transforms/batch_stats.py`
- Modify: `src/elspeth/plugins/transforms/field_mapper.py`
- Modify: `src/elspeth/plugins/transforms/json_explode.py`
- Modify: `tests/unit/plugins/transforms/test_batch_stats.py`
- Modify: `tests/unit/plugins/transforms/test_field_mapper.py`
- Modify: `tests/unit/plugins/transforms/test_json_explode.py`

### Web / Azure safety files

- Modify: `src/elspeth/plugins/transforms/web_scrape.py`
- Modify: `src/elspeth/plugins/transforms/azure/base.py`
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`
- Modify: `tests/unit/plugins/transforms/test_web_scrape.py`
- Modify: `tests/unit/plugins/transforms/azure/test_content_safety.py`
- Modify: `tests/unit/plugins/transforms/azure/test_prompt_shield.py`

### LLM family files

- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `tests/unit/plugins/llm/test_transform.py`

### Batch LLM spike files

- Create: `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md`

### Retrieval / suite reconciliation files

- Modify: `src/elspeth/plugins/transforms/rag/transform.py`
- Modify: `tests/unit/plugins/transforms/rag/test_transform.py`
- Modify: `tests/integration/plugins/transforms/test_output_schema_contract.py`
- Modify: `tests/unit/core/test_dag_schema_propagation.py`
- Modify: `tests/unit/web/composer/test_state.py`

---

## Implementation Tasks

### Task 1: Add a transform-owned invariant probe execution seam

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py`
- Modify: `tests/invariants/test_pass_through_invariants.py`
- Modify: `docs/contracts/plugin-protocol.md`
- Create: `tests/unit/plugins/test_invariant_probe_execution.py`

- [ ] **Step 1: Write the failing generic execution-hook tests**

Create `tests/unit/plugins/test_invariant_probe_execution.py`:

```python
from __future__ import annotations

from typing import Any

from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


class _SingleRowEcho(BaseTransform):
    name = "single_row_echo"
    input_schema = None
    output_schema = None

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(row, success_reason={"action": "echo"})

    def close(self) -> None:
        pass


class _BatchEcho(BaseTransform):
    name = "batch_echo"
    input_schema = None
    output_schema = None
    is_batch_aware = True

    def process(self, rows: list[PipelineRow], ctx: Any) -> TransformResult:  # type: ignore[override]
        return TransformResult.success(rows[0], success_reason={"action": "echo"})

    def close(self) -> None:
        pass


class _CustomExecution(BaseTransform):
    name = "custom_execution"
    input_schema = None
    output_schema = None

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        raise AssertionError("default process() path must not be used")

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        return TransformResult.success(probe_rows[0], success_reason={"action": "custom"})

    def close(self) -> None:
        pass


def test_default_forward_execution_calls_process_for_single_row() -> None:
    transform = _SingleRowEcho(config={})
    row = make_pipeline_row({"baseline": "kept"})
    result = transform.execute_forward_invariant_probe([row], make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"


def test_default_forward_execution_calls_process_for_batch_aware() -> None:
    transform = _BatchEcho(config={})
    rows = [make_pipeline_row({"baseline": "kept"})]
    result = transform.execute_forward_invariant_probe(rows, make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"


def test_custom_forward_execution_can_bypass_default_process_path() -> None:
    transform = _CustomExecution(config={})
    row = make_pipeline_row({"baseline": "kept"})
    result = transform.execute_forward_invariant_probe([row], make_context())
    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `uv run pytest tests/unit/plugins/test_invariant_probe_execution.py -q`
Expected: FAIL with `AttributeError: '_SingleRowEcho' object has no attribute 'execute_forward_invariant_probe'`.

- [ ] **Step 3: Add the execution hook to `BaseTransform`, switch the harness to use it, and delete execution-time skip behavior**

Edit `src/elspeth/plugins/infrastructure/base.py` and add two default execution hooks immediately after `backward_invariant_probe_rows()`. While touching the same area, update the `probe_config()` docstring so it documents the real constructor contract: the returned dict is passed positionally to `cls(...)`, matching `PluginManager.create_transform()`, not via `cls(config=...)`.

Add a focused constructor-parity regression to `tests/unit/plugins/test_invariant_probe_execution.py` using a probe-only transform whose `__init__` accepts the config dict positionally but does not expose a `config=` keyword. The test should import `_probe_instantiate()` from `tests/invariants/test_pass_through_invariants.py` and prove the helper instantiates that class successfully. This locks in that invariant probing follows the live runtime contract instead of inventing a stricter keyword-only one.

Then add the default execution hooks:

```python
    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Execute the forward invariant probe using the transform's production path.

        Default behavior:
        - batch-aware transforms receive the full probe row list
        - single-row transforms must receive exactly one probe row

        Transforms with transport or concurrency seams that cannot be exercised
        via plain ``process()`` (for example ``BatchTransformMixin`` classes or
        external-call transforms that need hermetic local doubles) override this
        method rather than teaching the invariant harness about private internals.
        """
        if self.is_batch_aware:
            return self.process(probe_rows, ctx)  # type: ignore[arg-type]
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() "
                f"received {len(probe_rows)} rows for a non-batch transform."
            )
        return self.process(probe_rows[0], ctx)

    def execute_backward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Execute the backward invariant probe.

        Defaults to the same execution path as the forward probe. Non-pass-through
        transforms can override this when their representative drop path needs a
        special local seam.
        """
        return self.execute_forward_invariant_probe(probe_rows, ctx)
```

Then edit `tests/invariants/test_pass_through_invariants.py` so `_probe_instantiate()` also matches the runtime constructor path:

```python
        return cls(config)
```

and so both tests call the execution hooks instead of calling `process()` directly:

```python
        result = transform.execute_forward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )
```

and:

```python
        result = transform.execute_backward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )
```

Delete the current `except (TypeError, AttributeError): pytest.skip(...)` block
entirely for forward probes. A transform that rejects its canonical probe
execution path is still a blind spot and must fail the suite, not consume skip
budget.

- [ ] **Step 4: Update the plugin protocol documentation**

Append this paragraph to the pass-through governance section in `docs/contracts/plugin-protocol.md` near the `passes_through_input` rules:

```markdown
For ADR-009 invariant coverage, a transform participates through three hooks:
`probe_config()`, `forward_invariant_probe_rows()` /
`backward_invariant_probe_rows()`, and
`execute_forward_invariant_probe()` /
`execute_backward_invariant_probe()`. The harness owns discovery and
assertions; the transform owns how to instantiate, shape, and execute a
hermetic representative probe. This keeps networked or `BatchTransformMixin`
transforms from forcing transport-specific branches into the invariant
harness itself.
```

- [ ] **Step 5: Run the focused tests**

Run: `uv run pytest tests/unit/plugins/test_invariant_probe_execution.py tests/invariants/test_pass_through_invariants.py -q -k "PassThrough or BatchReplicate or invariant_probe_execution or positional"`
Expected: PASS. Existing `PassThrough` and `BatchReplicate` behavior should remain unchanged, and the harness should now instantiate probeable transforms with the same positional constructor contract the runtime uses.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py docs/contracts/plugin-protocol.md \
  tests/invariants/test_pass_through_invariants.py tests/unit/plugins/test_invariant_probe_execution.py
git commit -m "test(invariants): add transform-owned probe execution seam"
```

### Task 2: Keep local shape-changing transforms non-pass-through and make them probeable

**Files:**
- Modify: `src/elspeth/plugins/transforms/batch_stats.py`
- Modify: `src/elspeth/plugins/transforms/field_mapper.py`
- Modify: `src/elspeth/plugins/transforms/json_explode.py`
- Create: `tests/unit/plugins/transforms/test_backward_invariant_probes.py`
- Modify: `tests/unit/plugins/transforms/test_batch_stats.py`
- Modify: `tests/unit/plugins/transforms/test_field_mapper.py`
- Modify: `tests/unit/plugins/transforms/test_json_explode.py`

- [ ] **Step 1: Write the failing backward-probe smoke tests**

Create `tests/unit/plugins/transforms/test_backward_invariant_probes.py`:

```python
from __future__ import annotations

import pytest

from elspeth.plugins.transforms.batch_replicate import BatchReplicate
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_context


def _emitted_rows(result):
    if result.row is not None:
        return [result.row]
    assert result.rows is not None
    return list(result.rows)


@pytest.mark.parametrize(
    ("transform_cls", "required_dropped_field"),
    [
        pytest.param(BatchReplicate, "quarantined_only_marker", id="BatchReplicate-control"),
        pytest.param(BatchStats, "baseline", id="BatchStats"),
        pytest.param(FieldMapper, "field_mapper_probe_source", id="FieldMapper"),
        pytest.param(JSONExplode, "json_explode_items", id="JSONExplode"),
    ],
)
def test_backward_probe_helpers_exercise_a_real_drop_path(
    transform_cls,
    required_dropped_field: str,
) -> None:
    transform = transform_cls(transform_cls.probe_config())
    base_row = make_pipeline_row({"baseline": "kept"})

    probe_rows = transform.backward_invariant_probe_rows(base_row)
    result = transform.execute_backward_invariant_probe(probe_rows, make_context())

    assert result.status == "success"
    assert all(required_dropped_field not in row.to_dict() for row in _emitted_rows(result))
```

- [ ] **Step 2: Run the new test to verify it fails on the new transforms**

Run: `uv run pytest tests/unit/plugins/transforms/test_backward_invariant_probes.py -q`
Expected: FAIL because `BatchStats`, `FieldMapper`, and `JSONExplode` do not yet implement `probe_config()`.

- [ ] **Step 3: Implement local probe config and representative backward rows**

Edit `src/elspeth/plugins/transforms/batch_stats.py`:

```python
    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "value_field": "batch_stats_probe_value",
            "compute_mean": True,
        }

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._value_field,
                value=1.0,
            )
        ]
```

Edit `src/elspeth/plugins/transforms/field_mapper.py`:

```python
    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "mapping": {
                "field_mapper_probe_source": "field_mapper_probe_target",
            },
            "strict": True,
        }

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="field_mapper_probe_source",
                value="mapped",
            )
        ]
```

Edit `src/elspeth/plugins/transforms/json_explode.py`:

```python
    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "array_field": "json_explode_items",
            "output_field": "json_explode_item",
            "include_index": True,
        }

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._array_field,
                value=("a", "b"),
            )
        ]
```

Also add one focused assertion in each existing family test file that the new
probe helper drives a success result rather than an error path.

- [ ] **Step 4: Run the local-transform test slice**

Run: `uv run pytest tests/unit/plugins/transforms/test_backward_invariant_probes.py tests/unit/plugins/transforms/test_batch_stats.py tests/unit/plugins/transforms/test_field_mapper.py tests/unit/plugins/transforms/test_json_explode.py tests/invariants/test_pass_through_invariants.py -q -k "BatchStats or FieldMapper or JSONExplode or backward_probe_helpers"`
Expected: PASS. `BatchStats`, `FieldMapper`, and `JSONExplode` should no longer be blind skips in the backward invariant.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/batch_stats.py \
  src/elspeth/plugins/transforms/field_mapper.py \
  src/elspeth/plugins/transforms/json_explode.py \
  tests/unit/plugins/transforms/test_backward_invariant_probes.py \
  tests/unit/plugins/transforms/test_batch_stats.py \
  tests/unit/plugins/transforms/test_field_mapper.py \
  tests/unit/plugins/transforms/test_json_explode.py
git commit -m "test(invariants): probe local shape-changing transforms truthfully"
```

### Task 3: Migrate `WebScrapeTransform` into the pass-through lane with a hermetic probe

**Files:**
- Modify: `src/elspeth/plugins/transforms/web_scrape.py`
- Modify: `tests/unit/plugins/transforms/test_web_scrape.py`
- Modify: `tests/unit/plugins/transforms/test_forward_invariant_probes.py`
- Modify: `tests/integration/plugins/transforms/test_output_schema_contract.py`

- [ ] **Step 1: Extend the forward-probe smoke test with `WebScrapeTransform`**

Edit `tests/unit/plugins/transforms/test_forward_invariant_probes.py`:

```python
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

@pytest.mark.parametrize(
    ("transform_cls", "expected_added_fields"),
    [
        pytest.param(Truncate, {"truncate_probe_1"}, id="Truncate"),
        pytest.param(TypeCoerce, {"type_coerce_probe_1"}, id="TypeCoerce"),
        pytest.param(KeywordFilter, {"keyword_filter_probe_1"}, id="KeywordFilter"),
        pytest.param(ValueTransform, {"value_transform_probe_added_1"}, id="ValueTransform"),
        pytest.param(
            WebScrapeTransform,
            {"web_scrape_probe_url", "page_content", "page_fingerprint", "fetch_status"},
            id="WebScrapeTransform",
        ),
    ],
)
```

and make the test call the new execution hook:

```python
    result = transform.execute_forward_invariant_probe(
        probe_rows,
        make_context(),
    )
```

- [ ] **Step 2: Run the smoke test to verify it fails**

Run: `uv run pytest tests/unit/plugins/transforms/test_forward_invariant_probes.py -q -k WebScrapeTransform`
Expected: FAIL because `WebScrapeTransform` is not yet annotated or probeable.

- [ ] **Step 3: Annotate `WebScrapeTransform` and add a no-network execution hook**

Edit `src/elspeth/plugins/transforms/web_scrape.py`:

```python
class WebScrapeTransform(BaseTransform):
    name = "web_scrape"
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "url_field": "web_scrape_probe_url",
            "content_field": "page_content",
            "fingerprint_field": "page_fingerprint",
            "http": {
                "abuse_contact": "invariants@example.com",
                "scraping_reason": "ADR-009 invariant probe",
                "allowed_hosts": ["93.184.216.34/32"],
            },
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._url_field,
                value="https://93.184.216.34/invariant-probe",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        class _InvariantPayloadStore:
            def store(self, payload: bytes) -> str:
                return "probe-processed-hash"

        class _InvariantCall:
            request_ref = "probe-request-hash"
            response_ref = "probe-response-hash"

        class _InvariantResponse:
            status_code = 200
            text = "<html><body><h1>Probe</h1><p>safe</p></body></html>"
            headers = {"content-type": "text/html"}

        def _fake_fetch_url(safe_request, probe_ctx):
            return _InvariantResponse(), safe_request.original_url, _InvariantCall()

        original_fetch = self._fetch_url
        had_payload_store = hasattr(self, "_payload_store")
        original_payload_store = getattr(self, "_payload_store", None)
        try:
            self._payload_store = _InvariantPayloadStore()
            self._fetch_url = _fake_fetch_url  # type: ignore[assignment]
            return super().execute_forward_invariant_probe(probe_rows, ctx)
        finally:
            self._fetch_url = original_fetch  # type: ignore[assignment]
            if had_payload_store:
                self._payload_store = original_payload_store
            else:
                delattr(self, "_payload_store")
```

Also add a focused unit test in `tests/unit/plugins/transforms/test_web_scrape.py`
that asserts `passes_through_input is True` and the forward probe returns a
successful row containing the original `baseline` field plus the scrape fields.
The probe test should also assert that any injected `_payload_store` state is
restored after execution so the hermetic hook does not leave mutated instance
state behind for later lifecycle operations.

- [ ] **Step 4: Run the web slice, including the invariant harness and the output-schema contract test**

Run: `uv run pytest tests/unit/plugins/transforms/test_web_scrape.py tests/unit/plugins/transforms/test_forward_invariant_probes.py tests/invariants/test_pass_through_invariants.py tests/integration/plugins/transforms/test_output_schema_contract.py -q -k "WebScrapeTransform or web_scrape"`
Expected: PASS. `WebScrapeTransform` should prove successful pass-through without making a real HTTP call, and the shared invariant harness should validate that immediately rather than waiting for Task 6 to expose constructor/execution regressions.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/web_scrape.py \
  tests/unit/plugins/transforms/test_web_scrape.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py \
  tests/integration/plugins/transforms/test_output_schema_contract.py
git commit -m "feat(invariants): annotate web_scrape as pass-through with hermetic probe"
```

### Task 4: Migrate the Azure safety transforms into truthful forward coverage

**Files:**
- Modify: `src/elspeth/plugins/transforms/azure/base.py`
- Modify: `src/elspeth/plugins/transforms/azure/content_safety.py`
- Modify: `src/elspeth/plugins/transforms/azure/prompt_shield.py`
- Modify: `tests/unit/plugins/transforms/azure/test_content_safety.py`
- Modify: `tests/unit/plugins/transforms/azure/test_prompt_shield.py`
- Modify: `tests/unit/plugins/transforms/test_forward_invariant_probes.py`

- [ ] **Step 1: Add failing forward-probe smoke coverage for the Azure transforms**

Extend `tests/unit/plugins/transforms/test_forward_invariant_probes.py`:

```python
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

@pytest.mark.parametrize(
    ("transform_cls", "expected_added_fields"),
    [
        pytest.param(Truncate, {"truncate_probe_1"}, id="Truncate"),
        pytest.param(TypeCoerce, {"type_coerce_probe_1"}, id="TypeCoerce"),
        pytest.param(KeywordFilter, {"keyword_filter_probe_1"}, id="KeywordFilter"),
        pytest.param(ValueTransform, {"value_transform_probe_added_1"}, id="ValueTransform"),
        pytest.param(AzureContentSafety, {"azure_safety_probe_text"}, id="AzureContentSafety"),
        pytest.param(AzurePromptShield, {"azure_safety_probe_text"}, id="AzurePromptShield"),
    ],
)
```

- [ ] **Step 2: Run the Azure smoke slice to verify it fails**

Run: `uv run pytest tests/unit/plugins/transforms/test_forward_invariant_probes.py -q -k "AzureContentSafety or AzurePromptShield"`
Expected: FAIL because the classes are not yet annotated and cannot execute hermetically.

- [ ] **Step 3: Put the shared probe execution hook in the Azure base class, but keep the pass-through truth declaration on the concrete classes**

Edit `src/elspeth/plugins/transforms/azure/base.py`:

```python
class BaseAzureSafetyTransform(BaseTransform, BatchTransformMixin):
    determinism = Determinism.EXTERNAL_CALL

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        field_name = self._fields[0] if isinstance(self._fields, list) else self._fields
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=field_name,
                value="safe invariant content",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        class _InvariantResponse:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload
                self.status_code = 200
                self.headers = {"content-type": "application/json"}
                self.content = b"{}"
                self.text = "{}"

            def json(self) -> dict[str, Any]:
                return self._payload

            def raise_for_status(self) -> None:
                return None

        class _InvariantHTTPClient:
            def __init__(self, payload: dict[str, Any]) -> None:
                self._payload = payload

            def post(self, url: str, json: dict[str, Any]) -> _InvariantResponse:
                return _InvariantResponse(self._payload)

        original_get_http_client = self._get_http_client
        try:
            self._get_http_client = lambda state_id, token_id=None: _InvariantHTTPClient(  # type: ignore[assignment]
                self._invariant_probe_response()
            )
            return self._process_single_with_state(
                probe_rows[0],
                "invariant-state",
                token_id=ctx.token.token_id if ctx.token is not None else None,
            )
        finally:
            self._get_http_client = original_get_http_client  # type: ignore[assignment]

    def _invariant_probe_response(self) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _invariant_probe_response() "
            "for ADR-009 hermetic forward probes."
        )
```

- [ ] **Step 4: Give each concrete Azure transform a probe config and response payload**

Edit `src/elspeth/plugins/transforms/azure/content_safety.py`:

```python
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "endpoint": "https://invariant.example.cognitiveservices.azure.com",
            "api_key": "probe-key",
            "fields": ["azure_safety_probe_text"],
            "thresholds": {"hate": 2, "violence": 2, "sexual": 2, "self_harm": 2},
            "schema": {"mode": "observed"},
        }

    def _invariant_probe_response(self) -> dict[str, Any]:
        return {
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 0},
                {"category": "Violence", "severity": 0},
                {"category": "Sexual", "severity": 0},
                {"category": "SelfHarm", "severity": 0},
            ]
        }
```

Edit `src/elspeth/plugins/transforms/azure/prompt_shield.py`:

```python
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "endpoint": "https://invariant.example.cognitiveservices.azure.com",
            "api_key": "probe-key",
            "fields": ["azure_safety_probe_text"],
            "schema": {"mode": "observed"},
        }

    def _invariant_probe_response(self) -> dict[str, Any]:
        return {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [{"attackDetected": False}],
        }
```

Also add focused tests in each existing family file asserting that:

- the concrete class carries `passes_through_input is True`,
- `BaseAzureSafetyTransform` remains unannotated / default-false, and
- `execute_forward_invariant_probe()` returns a success result preserving the
  original row fields.

- [ ] **Step 5: Run the Azure slice**

Run: `uv run pytest tests/unit/plugins/transforms/azure/test_content_safety.py tests/unit/plugins/transforms/azure/test_prompt_shield.py tests/unit/plugins/transforms/test_forward_invariant_probes.py tests/invariants/test_pass_through_invariants.py -q -k "AzureContentSafety or AzurePromptShield"`
Expected: PASS. Both Azure transforms should join the annotated pass-through set with deterministic local responses.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/transforms/azure/base.py \
  src/elspeth/plugins/transforms/azure/content_safety.py \
  src/elspeth/plugins/transforms/azure/prompt_shield.py \
  tests/unit/plugins/transforms/azure/test_content_safety.py \
  tests/unit/plugins/transforms/azure/test_prompt_shield.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py
git commit -m "feat(invariants): migrate azure safety transforms to pass-through coverage"
```

### Task 5: Migrate `LLMTransform` into truthful forward coverage and keep batch LLMs out of this tranche

**Files:**
- Modify: `src/elspeth/plugins/transforms/llm/transform.py`
- Modify: `tests/unit/plugins/llm/test_transform.py`
- Modify: `tests/unit/plugins/transforms/test_forward_invariant_probes.py`

- [ ] **Step 1: Add failing forward-probe smoke coverage for `LLMTransform` only**

Extend `tests/unit/plugins/transforms/test_forward_invariant_probes.py`:

```python
from elspeth.plugins.transforms.llm.transform import LLMTransform

@pytest.mark.parametrize(
    ("transform_cls", "expected_added_fields"),
    [
        pytest.param(Truncate, {"truncate_probe_1"}, id="Truncate"),
        pytest.param(TypeCoerce, {"type_coerce_probe_1"}, id="TypeCoerce"),
        pytest.param(KeywordFilter, {"keyword_filter_probe_1"}, id="KeywordFilter"),
        pytest.param(ValueTransform, {"value_transform_probe_added_1"}, id="ValueTransform"),
        pytest.param(LLMTransform, {"llm_probe_text", "llm_response"}, id="LLMTransform"),
    ],
)
```

- [ ] **Step 2: Run the `LLMTransform` smoke slice to verify it fails**

Run: `uv run pytest tests/unit/plugins/transforms/test_forward_invariant_probes.py -q -k LLMTransform`
Expected: FAIL because `LLMTransform` is not yet annotated/probeable.

- [ ] **Step 3: Annotate `LLMTransform` and use a lifecycle-safe fake provider through `_process_row()`**

Edit `src/elspeth/plugins/transforms/llm/transform.py`:

```python
class LLMTransform(BaseTransform, BatchTransformMixin):
    name = "llm"
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "api_key": "probe-key",
            "model": "openai/gpt-4o-mini",
            "template": "{{ row.llm_probe_text }}",
            "schema": {"mode": "observed"},
            "required_input_fields": [],
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="llm_probe_text",
                value="probe request",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        class _InvariantProvider:
            def execute_query(self, *args: Any, **kwargs: Any) -> LLMQueryResult:
                return LLMQueryResult(
                    content="probe response",
                    usage=TokenUsage.known(1, 1),
                    model="probe-model",
                    finish_reason=FinishReason.STOP,
                )

            def close(self) -> None:
                return None

        original_provider = self._provider
        probe_provider = _InvariantProvider()
        try:
            self._provider = probe_provider
            return self._process_row(probe_rows[0], ctx)
        finally:
            self._provider = original_provider
            probe_provider.close()
```

In `tests/unit/plugins/llm/test_transform.py`, add a focused multi-query
regression asserting that `passes_through_input=True` remains truthful for a
multi-query config as well:

```python
def test_multi_query_success_preserves_original_input_fields() -> None:
    transform, mock_provider = _make_transform_with_mock_provider(
        _make_multi_query_config(provider="azure")
    )
    mock_provider.execute_query.return_value = LLMQueryResult(
        content='{"score": 1}',
        usage=TokenUsage.known(1, 1),
        model="gpt-4o",
        finish_reason=FinishReason.STOP,
    )

    row = _make_row({"text": "hello", "baseline": "kept"})
    result = transform._process_row(row, _make_ctx())

    assert result.status == "success"
    assert result.row is not None
    assert result.row["baseline"] == "kept"
    assert result.row["text"] == "hello"
```

Also add a focused regression that the forward invariant probe restores any
pre-existing provider state and that calling `close()` after probe execution
does not raise.

- [ ] **Step 4: Keep the batch-LLM classes out of the pass-through lane and hand the blocker to the dedicated spike**

Do **not** modify `src/elspeth/plugins/transforms/llm/azure_batch.py` or
`src/elspeth/plugins/transforms/llm/openrouter_batch.py` in this tranche.

Carry these requirements forward into the dedicated batch-LLM spike task in
this plan:

- The current mixed-success batch path in both classes emits partial-error rows
  that omit some currently declared guaranteed fields (`*_usage`,
  `*_model`), so `passes_through_input=True` would not yet be truthful.
- A helper that directly constructs a happy-path row would be invalid evidence.
  The future probe seam must exercise the real `_process_batch()` and
  checkpoint/result-assembly behavior.
- The future batch-LLM tranche must start by adding regression coverage for the
  current declaration/runtime mismatch, then choose a real fix: either make the
  mixed-success outputs satisfy the currently declared fields, or change the
  declarations/output contract design before attempting pass-through annotation.

- [ ] **Step 5: Run the `LLMTransform` slice**

Run: `uv run pytest tests/unit/plugins/llm/test_transform.py tests/unit/plugins/transforms/test_forward_invariant_probes.py tests/invariants/test_pass_through_invariants.py -q -k LLMTransform`
Expected: PASS. `LLMTransform` should join the annotated forward invariant without vendor calls.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/transforms/llm/transform.py \
  tests/unit/plugins/llm/test_transform.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py
git commit -m "feat(invariants): migrate llm transform into truthful pass-through coverage"
```

### Task 6: Migrate `RAGRetrievalTransform` and make the scope-level coverage explicit

**Files:**
- Modify: `src/elspeth/plugins/transforms/rag/transform.py`
- Modify: `tests/unit/plugins/transforms/rag/test_transform.py`
- Modify: `tests/integration/plugins/transforms/test_output_schema_contract.py`
- Modify: `tests/unit/core/test_dag_schema_propagation.py`
- Modify: `tests/unit/web/composer/test_state.py`
- Create: `tests/invariants/test_transform_probe_coverage.py`
- Modify: `tests/unit/plugins/transforms/test_forward_invariant_probes.py`

- [ ] **Step 1: Add scope-level coverage that proves successful canonical probe execution, plus a failing RAG forward-smoke case**

Create `tests/invariants/test_transform_probe_coverage.py`:

```python
from __future__ import annotations

from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
from elspeth.plugins.transforms.batch_stats import BatchStats
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.transforms.json_explode import JSONExplode
from elspeth.plugins.transforms.llm.transform import LLMTransform
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform
from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
from elspeth.testing import make_pipeline_row
from tests.invariants.test_pass_through_invariants import _probe_context, _probe_instantiate

FORWARD_SCOPE = (
    WebScrapeTransform,
    AzureContentSafety,
    AzurePromptShield,
    LLMTransform,
    RAGRetrievalTransform,
)

BACKWARD_SCOPE = (
    BatchStats,
    FieldMapper,
    JSONExplode,
)


def _assert_successful_probe_execution(
    cls: type[BaseTransform],
    *,
    direction: str,
) -> None:
    transform = _probe_instantiate(cls)
    base_row = make_pipeline_row({"baseline": "kept"})

    if direction == "forward":
        probe_rows = transform.forward_invariant_probe_rows(base_row)
        result = transform.execute_forward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )
    else:
        probe_rows = transform.backward_invariant_probe_rows(base_row)
        result = transform.execute_backward_invariant_probe(
            probe_rows,
            _probe_context(transform),
        )

    assert result.status == "success", (
        f"{cls.__name__} canonical {direction} invariant probe must execute "
        f"successfully; got status={result.status!r}."
    )


def test_in_scope_forward_probes_execute_successfully_without_blind_skips() -> None:
    for cls in FORWARD_SCOPE:
        _assert_successful_probe_execution(cls, direction="forward")


def test_in_scope_backward_probes_execute_successfully_without_blind_skips() -> None:
    for cls in BACKWARD_SCOPE:
        _assert_successful_probe_execution(cls, direction="backward")
```

Extend `tests/unit/plugins/transforms/test_forward_invariant_probes.py`:

```python
from elspeth.plugins.transforms.rag.transform import RAGRetrievalTransform

@pytest.mark.parametrize(
    ("transform_cls", "expected_added_fields"),
    [
        pytest.param(Truncate, {"truncate_probe_1"}, id="Truncate"),
        pytest.param(TypeCoerce, {"type_coerce_probe_1"}, id="TypeCoerce"),
        pytest.param(KeywordFilter, {"keyword_filter_probe_1"}, id="KeywordFilter"),
        pytest.param(ValueTransform, {"value_transform_probe_added_1"}, id="ValueTransform"),
        pytest.param(
            RAGRetrievalTransform,
            {"rag_probe_query", "policy__rag_context", "policy__rag_score", "policy__rag_count", "policy__rag_sources"},
            id="RAGRetrievalTransform",
        ),
    ],
)
```

- [ ] **Step 2: Run the RAG/scope tests to verify they fail**

Run: `uv run pytest tests/unit/plugins/transforms/test_forward_invariant_probes.py tests/invariants/test_transform_probe_coverage.py -q -k "RAGRetrievalTransform or in_scope"`
Expected: FAIL because `RAGRetrievalTransform` is not yet annotated/probeable
and the new scope-level test now fails on canonical probe execution regressions,
not just instantiation gaps.

- [ ] **Step 3: Annotate `RAGRetrievalTransform` and give it a provider-agnostic, lifecycle-safe forward execution path**

Edit `src/elspeth/plugins/transforms/rag/transform.py`:

```python
class RAGRetrievalTransform(BaseTransform):
    name = "rag_retrieval"
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "output_prefix": "policy",
            "query_field": "rag_probe_query",
            # Use a provider name that validates on the base install. The
            # invariant hook injects a local double and never constructs the
            # real provider.
            "provider": "azure_search",
            "provider_config": {
                "endpoint": "https://invariant.example.search.windows.net",
                "index": "invariant-probe",
                "api_key": "probe-key",
            },
            "schema": {"mode": "observed"},
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="rag_probe_query",
                value="What is the policy?",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        class _InvariantProvider:
            last_skipped_count = 0
            last_skipped_reasons: list[dict[str, Any]] = []

            def search(self, *args: Any, **kwargs: Any):
                from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk

                return [
                    RetrievalChunk(
                        content="Probe context",
                        score=0.95,
                        source_id="probe-doc",
                        metadata={"kind": "invariant"},
                    )
                ]

            def close(self) -> None:
                return None

        original_provider = self._provider
        original_on_start_called = self._on_start_called
        probe_provider = _InvariantProvider()
        try:
            self._provider = probe_provider
            self._on_start_called = True
            return super().execute_forward_invariant_probe(probe_rows, ctx)
        finally:
            self._provider = original_provider
            self._on_start_called = original_on_start_called
            probe_provider.close()
```

Add focused assertions in `tests/unit/plugins/transforms/rag/test_transform.py`
that `passes_through_input` is true, the probe path preserves the original query
field, and calling `close()` after probe execution remains safe. Keep these in
the unit suite that already owns `RAGRetrievalTransform` lifecycle behavior
rather than moving probe/lifecycle coverage into
`tests/integration/plugins/transforms/test_rag_pipeline.py`. Do not rely on
`pytest.importorskip("chromadb")` or any optional provider extra in the
forward-probe path; this coverage is mandatory for the base install.

- [ ] **Step 4: Run the full tranche verification, including composer/runtime agreement checks**

Run:

```bash
uv run pytest \
  tests/unit/plugins/test_invariant_probe_execution.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py \
  tests/unit/plugins/transforms/test_backward_invariant_probes.py \
  tests/unit/plugins/transforms/rag/test_transform.py \
  tests/invariants/test_pass_through_invariants.py \
  tests/invariants/test_transform_probe_coverage.py \
  tests/integration/plugins/transforms/test_output_schema_contract.py \
  tests/unit/core/test_dag_schema_propagation.py \
  tests/unit/web/composer/test_state.py -q
```

Expected: PASS. In-scope transforms should now be either annotated-pass-through
with working forward probes or non-pass-through with working backward probes.
`AzureBatchLLMTransform` and `OpenRouterBatchLLMTransform` remain intentionally
out of scope for this tranche.

- [ ] **Step 5: Run lint on the touched files and commit the tranche**

Run:

```bash
uv run ruff check \
  src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/plugins/transforms/batch_stats.py \
  src/elspeth/plugins/transforms/field_mapper.py \
  src/elspeth/plugins/transforms/json_explode.py \
  src/elspeth/plugins/transforms/web_scrape.py \
  src/elspeth/plugins/transforms/azure/base.py \
  src/elspeth/plugins/transforms/azure/content_safety.py \
  src/elspeth/plugins/transforms/azure/prompt_shield.py \
  src/elspeth/plugins/transforms/llm/transform.py \
  src/elspeth/plugins/transforms/rag/transform.py \
  tests/invariants/test_pass_through_invariants.py \
  tests/invariants/test_transform_probe_coverage.py \
  tests/unit/plugins/test_invariant_probe_execution.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py \
  tests/unit/plugins/transforms/test_backward_invariant_probes.py \
  tests/unit/plugins/transforms/rag/test_transform.py
```

Then commit:

```bash
git add src/elspeth/plugins/infrastructure/base.py \
  src/elspeth/plugins/transforms/batch_stats.py \
  src/elspeth/plugins/transforms/field_mapper.py \
  src/elspeth/plugins/transforms/json_explode.py \
  src/elspeth/plugins/transforms/web_scrape.py \
  src/elspeth/plugins/transforms/azure/base.py \
  src/elspeth/plugins/transforms/azure/content_safety.py \
  src/elspeth/plugins/transforms/azure/prompt_shield.py \
  src/elspeth/plugins/transforms/llm/transform.py \
  src/elspeth/plugins/transforms/rag/transform.py \
  docs/contracts/plugin-protocol.md \
  tests/invariants/test_pass_through_invariants.py \
  tests/invariants/test_transform_probe_coverage.py \
  tests/unit/plugins/test_invariant_probe_execution.py \
  tests/unit/plugins/transforms/test_forward_invariant_probes.py \
  tests/unit/plugins/transforms/test_backward_invariant_probes.py \
  tests/unit/plugins/transforms/test_web_scrape.py \
  tests/unit/plugins/transforms/azure/test_content_safety.py \
  tests/unit/plugins/transforms/azure/test_prompt_shield.py \
  tests/unit/plugins/llm/test_transform.py \
  tests/unit/plugins/transforms/rag/test_transform.py \
  tests/integration/plugins/transforms/test_output_schema_contract.py \
  tests/unit/core/test_dag_schema_propagation.py \
  tests/unit/web/composer/test_state.py
git commit -m "feat(invariants): close remaining transform migration blind spots"
```

### Task 7: Spike the batch-LLM follow-on without changing runtime behavior

**Files:**
- Create: `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md`

- [ ] **Step 1: Re-read the batch-LLM contract surfaces and capture the exact mismatch**

Inspect:

- `src/elspeth/plugins/transforms/llm/__init__.py`
- `src/elspeth/plugins/transforms/llm/azure_batch.py`
- `src/elspeth/plugins/transforms/llm/openrouter_batch.py`
- `tests/unit/plugins/llm/test_azure_batch.py`
- `tests/unit/plugins/llm/test_openrouter_batch.py`

Write a short matrix in the spike note covering:

- what `get_llm_guaranteed_fields()` currently declares,
- what each batch class advertises via `declared_output_fields` /
  `_output_schema_config`,
- what the fully successful row path emits,
- what mixed-success / partial-error rows emit today, and
- which fields are currently missing from error-bearing rows.

- [ ] **Step 2: Evaluate fix directions against contract truthfulness and engine behavior**

In the spike note, evaluate at least these options:

- keep the current guaranteed-field contract and populate `*_usage` /
  `*_model` on error-bearing rows with truthful sentinel values such as `None`,
- narrow the guaranteed-field contract for batch LLM outputs and push usage /
  model into a non-guaranteed lane, or
- redesign the batch transform result contract so mixed-success rows are not
  represented as a successful multi-row pass-through output.

For each option, spell out the trade-offs for:

- downstream DAG/schema validation,
- composer/output-schema previews,
- audit semantics and row-level explainability,
- test-path integrity, and
- migration risk for existing pipelines.

- [ ] **Step 3: Define the minimum truthful no-network probe seam for each batch implementation**

The spike must describe a probe seam that exercises the real runtime assembly
path rather than fabricating output rows:

- For `OpenRouterBatchLLMTransform`, the seam must still drive
  `_process_batch()` and the per-row result assembly path, even if HTTP is
  replaced by a local double.
- For `AzureBatchLLMTransform`, the seam must still exercise the checkpoint /
  resume / result-assembly behavior using local doubles or synthetic downloaded
  batch payloads, not a helper that directly returns a prebuilt success row.

The spike note must name the likely test entry points and what would need to be
stubbed or injected to keep the probe deterministic and offline.

- [ ] **Step 4: End the spike with a concrete recommendation and a follow-on implementation checklist**

The spike note must finish with:

- a recommended contract direction,
- the smallest set of files/tests expected to change in the future tranche,
- the characterization tests that should be written first, and
- explicit non-go conditions that should block pass-through annotation if the
  real runtime path still disagrees with the declared contract.

- [ ] **Step 5: Commit the spike artifact**

```bash
git add docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md
git commit -m "docs(invariants): spike batch llm follow-on design"
```

---

## Definition of Done

- [ ] `BatchStats`, `FieldMapper`, and `JSONExplode` remain `passes_through_input = False` and are no longer skipped for missing probe opt-in.
- [ ] `WebScrapeTransform`, `AzureContentSafety`, `AzurePromptShield`, `LLMTransform`, and `RAGRetrievalTransform` are truthfully annotated `passes_through_input = True`.
- [ ] The Azure pass-through declaration lives on `AzureContentSafety` and `AzurePromptShield`, not on `BaseAzureSafetyTransform`.
- [ ] `AzureBatchLLMTransform` and `OpenRouterBatchLLMTransform` remain explicitly deferred until their mixed-success output contract and a real no-network batch probe seam are designed.
- [ ] A spike note exists at `docs/superpowers/specs/2026-04-21-batch-llm-invariant-follow-on-design.md` capturing the batch-LLM contract mismatch, a recommended fix direction, and the minimum truthful no-network probe seam.
- [ ] The invariant harness no longer hardcodes `process()` as the only probe execution path and no longer converts execution-time probe rejection into `pytest.skip()`.
- [ ] Probe doubles for `WebScrapeTransform`, `LLMTransform`, and `RAGRetrievalTransform` restore any replaced collaborators or instance state before returning and leave subsequent lifecycle calls safe after probe execution.
- [ ] `RAGRetrievalTransform` probe configuration validates against a provider entry available in the base install; no invariant coverage depends on optional `chromadb` extras.
- [ ] No in-scope transform requires live credentials, DNS, or network access to satisfy invariant coverage.
- [ ] The scope-level coverage tests instantiate, shape, and successfully execute the canonical forward/backward probes for the exact transforms tracked by epic `elspeth-be398f0bcb`, making both instantiation regressions and execution-time blind skips hard failures.
- [ ] Output-schema and composer-preview agreement tests still pass after the new annotations.
