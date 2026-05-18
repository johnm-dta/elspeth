# Report Assemble Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic `report_assemble` batch-aware transform that collates many input rows into one text/report row per aggregation flush, with ordinary metadata columns for pagination and downstream sink control.

**Architecture:** The aggregation node remains responsible for buffering and triggers. The engine will expose durable aggregation-batch metadata on `PluginContext` so batch-aware transforms can emit truthful pagination columns without relying on uncheckpointed plugin instance state. `report_assemble` will be a pure transform: it formats rows into a string field plus metadata columns and leaves persistence to existing sinks.

**Tech Stack:** Python 3.13, Pydantic v2 config models, Elspeth `BaseTransform`/`BatchTransformProtocol`, `PipelineRow`/`SchemaContract`, pytest, existing plugin hash and contract enforcement scripts.

---

## Scope

In scope:
- A new transform plugin named `report_assemble`.
- Count-trigger pagination and end-of-source whole-report assembly through existing `aggregations:` config.
- Output as normal row columns, not hidden prose metadata.
- Engine-owned aggregation batch context needed for resume-safe `report_index` and `lines_seen_total`.
- Docs and a small example using the existing `text` source plus JSON sink.

Out of scope:
- LLM report writing or report templates.
- A new line-by-line text sink.
- Grouped report emission; add it later once per-group counters are designed.
- Backfilling previously emitted sink rows to mark an exact count-triggered final page as final. The truthful v1 flag is `is_end_of_source_report`, not "no later rows exist".

## File Structure

New files:
- `src/elspeth/plugins/transforms/report_assemble.py` — deterministic batch-aware transform.
- `tests/unit/plugins/transforms/test_report_assemble.py` — plugin config, rendering, metadata, discovery, and contract tests.
- `tests/integration/pipeline/test_report_assemble_aggregation.py` — production-path aggregation tests.
- `examples/report_assemble/input.txt` — line-oriented example input.
- `examples/report_assemble/settings.yaml` — count-trigger pagination example.
- `examples/report_assemble/README.md` — explain how to run and inspect output.

Modified files:
- `src/elspeth/contracts/node_state_context.py` — add aggregation batch context dataclass and extend audit flush context.
- `src/elspeth/contracts/plugin_context.py` — add `aggregation_batch` to plugin execution context.
- `src/elspeth/contracts/contexts.py` — expose `aggregation_batch` on `TransformContext`.
- `src/elspeth/contracts/aggregation_checkpoint.py` — persist aggregation counters, including empty-buffer checkpoints.
- `src/elspeth/engine/executors/aggregation.py` — maintain counters, inject context, checkpoint/restore counters.
- `config/cicd/contracts-whitelist.yaml` — whitelist `ReportAssemble.probe_config` and constructor config.
- `docs/reference/configuration.md` — add `report_assemble` to plugin table and aggregation example.
- `docs/guides/user-manual.md` — add CLI plugin list entry.

## Task 1: Add Durable Aggregation Batch Context

**Files:**
- Modify: `src/elspeth/contracts/node_state_context.py`
- Modify: `src/elspeth/contracts/plugin_context.py`
- Modify: `src/elspeth/contracts/contexts.py`
- Modify: `src/elspeth/contracts/aggregation_checkpoint.py`
- Modify: `src/elspeth/engine/executors/aggregation.py`
- Test: `tests/unit/engine/test_executors.py`
- Test: `tests/integration/pipeline/test_aggregation_recovery.py`

- [ ] **Step 1: Write failing unit tests for context injection**

Add a test that uses a tiny batch-aware transform to capture `ctx.aggregation_batch` during `AggregationExecutor.execute_flush()`.

Expected assertions:
```python
assert captured.trigger_type == "count"
assert captured.batch_id == flushed_batch_id
assert captured.batch_size == 3
assert captured.flush_index == 1
assert captured.rows_seen_total == 3
assert captured.row_start == 1
assert captured.row_end == 3
assert captured.is_end_of_source is False
```

Add a second flush in the same test and assert:
```python
assert second.flush_index == 2
assert second.rows_seen_total == 6
assert second.row_start == 4
assert second.row_end == 6
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:
```bash
.venv/bin/python -m pytest tests/unit/engine/test_executors.py -k aggregation_batch_context -v
```

Expected: FAIL because `PluginContext` has no `aggregation_batch` field.

- [ ] **Step 3: Add the context dataclass**

In `src/elspeth/contracts/node_state_context.py`, add:

```python
@dataclass(frozen=True, slots=True)
class AggregationBatchContext:
    """Metadata about the aggregation flush currently executing."""

    trigger_type: str
    batch_id: str
    batch_size: int
    flush_index: int
    rows_seen_total: int
    row_start: int
    row_end: int
    is_end_of_source: bool

    def __post_init__(self) -> None:
        if not self.trigger_type:
            raise ValueError("AggregationBatchContext.trigger_type must not be empty")
        if not self.batch_id:
            raise ValueError("AggregationBatchContext.batch_id must not be empty")
        require_int(self.batch_size, "batch_size", min_value=1)
        require_int(self.flush_index, "flush_index", min_value=1)
        require_int(self.rows_seen_total, "rows_seen_total", min_value=1)
        require_int(self.row_start, "row_start", min_value=1)
        require_int(self.row_end, "row_end", min_value=1)
        if self.row_end < self.row_start:
            raise ValueError("AggregationBatchContext.row_end must be >= row_start")
        if self.rows_seen_total < self.row_end:
            raise ValueError("AggregationBatchContext.rows_seen_total must be >= row_end")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_type": self.trigger_type,
            "batch_id": self.batch_id,
            "batch_size": self.batch_size,
            "flush_index": self.flush_index,
            "rows_seen_total": self.rows_seen_total,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "is_end_of_source": self.is_end_of_source,
        }
```

Also extend `AggregationFlushContext` with `flush_index`, `rows_seen_total`, `row_start`, `row_end`, and `is_end_of_source`, then update `to_dict()`.

- [ ] **Step 4: Add context fields to plugin protocols**

In `src/elspeth/contracts/plugin_context.py`, import `AggregationBatchContext` under normal imports and add:

```python
aggregation_batch: AggregationBatchContext | None = field(default=None)
```

In `src/elspeth/contracts/contexts.py`, add the import under `TYPE_CHECKING` and add this property to `TransformContext`:

```python
@property
def aggregation_batch(self) -> AggregationBatchContext | None: ...
```

- [ ] **Step 5: Persist aggregation counters**

In `src/elspeth/engine/executors/aggregation.py`, extend `_AggregationNodeState`:

```python
accepted_count_total: int = 0
completed_flush_count: int = 0
```

Increment `accepted_count_total` in `buffer_row()` after the row is accepted into the batch.

In `src/elspeth/contracts/aggregation_checkpoint.py`, bump the version and extend `AggregationNodeCheckpoint`:

```python
accepted_count_total: int
completed_flush_count: int
batch_id: str | None
```

Allow `batch_id=None` only when `tokens` is empty. Keep `batch_id` required when `tokens` is non-empty.

In `AggregationExecutor.get_checkpoint_state()`, include nodes when either they have buffered tokens or their counters are non-zero. This preserves counters in post-sink checkpoints where the buffer is empty.

In `restore_from_checkpoint()`, restore the counters for every checkpointed node. Only call `_reconcile_checkpoint_batch_members()` when `node_checkpoint.tokens` is non-empty.

- [ ] **Step 6: Inject and clear context during flush**

In `AggregationExecutor.execute_flush()`, before `transform.process(...)`, compute:

```python
batch_context = AggregationBatchContext(
    trigger_type=trigger_type.value,
    batch_id=batch_id,
    batch_size=len(buffered_rows),
    flush_index=node.completed_flush_count + 1,
    rows_seen_total=node.accepted_count_total,
    row_start=node.accepted_count_total - len(buffered_rows) + 1,
    row_end=node.accepted_count_total,
    is_end_of_source=trigger_type is TriggerType.END_OF_SOURCE,
)
ctx.aggregation_batch = batch_context
```

Pass the same fields into `AggregationFlushContext` for `context_after`. After a successful batch completion, increment `node.completed_flush_count += 1`. Clear `ctx.aggregation_batch = None` in both success and failure cleanup paths.

- [ ] **Step 7: Verify GREEN**

Run:
```bash
.venv/bin/python -m pytest tests/unit/engine/test_executors.py -k aggregation_batch_context -v
.venv/bin/python -m pytest tests/integration/pipeline/test_aggregation_recovery.py -k aggregation -v
```

Expected: PASS.

## Task 2: Define `report_assemble` Unit Tests

**Files:**
- Create: `tests/unit/plugins/transforms/test_report_assemble.py`

- [ ] **Step 1: Write tests before implementation**

Create tests covering:
- required class attributes:
```python
assert ReportAssemble.name == "report_assemble"
assert ReportAssemble.is_batch_aware is True
```
- simple collation:
```python
transform = ReportAssemble({"schema": {"mode": "observed"}, "text_field": "line"})
result = transform.process([_row({"line": "alpha"}), _row({"line": "beta"})], _ctx(batch_size=2))
assert result.row["report_body"] == "alpha\nbeta"
```
- pagination columns from engine context:
```python
assert result.row["report_index"] == 2
assert result.row["line_start"] == 81
assert result.row["line_end"] == 160
assert result.row["line_count"] == 80
assert result.row["lines_seen_total"] == 160
assert result.row["flush_trigger"] == "count"
assert result.row["is_end_of_source_report"] is False
```
- final partial page:
```python
assert result.row["is_end_of_source_report"] is True
assert result.row["line_count"] == 37
```
- configurable output field:
```python
transform = ReportAssemble({"schema": {"mode": "observed"}, "text_field": "line", "output_field": "page_text"})
assert result.row["page_text"] == "alpha\nbeta"
```
- markdown title:
```python
transform = ReportAssemble({"schema": {"mode": "observed"}, "text_field": "line", "format": "markdown", "title": "Findings"})
assert result.row["report_body"].startswith("# Findings\n\n")
```
- non-string row value crashes as an upstream validation bug:
```python
with pytest.raises(TypeError, match="must be a string"):
    transform.process([_row({"line": 3})], ctx)
```
- empty batch returns a non-retryable `TransformResult.error`.
- `PluginManager.register_builtin_plugins()` discovers `report_assemble`.

- [ ] **Step 2: Run tests and verify RED**

Run:
```bash
.venv/bin/python -m pytest tests/unit/plugins/transforms/test_report_assemble.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `elspeth.plugins.transforms.report_assemble`.

## Task 3: Implement `report_assemble`

**Files:**
- Create: `src/elspeth/plugins/transforms/report_assemble.py`
- Modify: `config/cicd/contracts-whitelist.yaml`
- Test: `tests/unit/plugins/transforms/test_report_assemble.py`

- [ ] **Step 1: Add config model and transform shell**

Use this shape:

```python
class ReportAssembleConfig(TransformDataConfig):
    text_field: str = Field(description="Name of the string field to collate")
    output_field: str = Field(default="report_body", description="Output field for assembled text")
    format: Literal["plain_text", "markdown", "html_fragment"] = Field(default="plain_text")
    join_with: str = Field(default="\n", description="Separator used between input rows")
    title: str | None = Field(default=None, description="Optional report title")

    @field_validator("text_field", "output_field")
    @classmethod
    def _reject_empty(cls, value: str, info: Any) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be empty")
        return stripped

    @field_validator("output_field")
    @classmethod
    def _validate_output_identifier(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError(f"output_field must be a valid Python identifier, got {value!r}")
        return value
```

The class attributes:

```python
class ReportAssemble(BaseTransform):
    name = "report_assemble"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = ReportAssembleConfig
    is_batch_aware = True
```

The hash value is a deliberate stale sentinel. Task 6 refreshes it with the project script after the file is complete.

- [ ] **Step 2: Add schema contract behavior**

In `__init__`, call `ReportAssembleConfig.from_dict(config, plugin_name=self.name)`, initialize declared input fields, and ensure `text_field` is in `schema_config.required_fields`.

Declare output fields:
```python
_REPORT_METADATA_FIELDS = frozenset(
    {
        "report_format",
        "report_index",
        "line_start",
        "line_end",
        "line_count",
        "lines_seen_total",
        "flush_trigger",
        "is_end_of_source_report",
    }
)
```

Set:
```python
self.declared_output_fields = frozenset({cfg.output_field, *_REPORT_METADATA_FIELDS})
```

Build `_output_schema_config` with observed mode and guaranteed fields for every declared output field. The report replaces input rows; do not propagate arbitrary input fields into the output contract.

- [ ] **Step 3: Add rendering helpers**

Implement:
```python
def _render_plain_text(self, lines: list[str]) -> str:
    body = self._join_with.join(lines)
    if self._title:
        return f"{self._title}\n\n{body}" if body else self._title
    return body

def _render_markdown(self, lines: list[str]) -> str:
    body = self._join_with.join(lines)
    if self._title:
        return f"# {self._title}\n\n{body}" if body else f"# {self._title}"
    return body

def _render_html_fragment(self, lines: list[str]) -> str:
    import html
    paragraphs = "\n".join(f"<p>{html.escape(line)}</p>" for line in lines)
    if self._title:
        return f"<h1>{html.escape(self._title)}</h1>\n{paragraphs}" if paragraphs else f"<h1>{html.escape(self._title)}</h1>"
    return paragraphs
```

- [ ] **Step 4: Add `process()`**

Implement batch processing:
```python
def process(self, rows: list[PipelineRow], ctx: TransformContext) -> TransformResult:
    if not rows:
        return TransformResult.error({"reason": "empty_batch"}, retryable=False)
    if ctx.aggregation_batch is None:
        raise RuntimeError("report_assemble must run as an aggregation node with aggregation_batch context")

    lines: list[str] = []
    for index, row in enumerate(rows):
        value = row[self._text_field]
        if type(value) is not str:
            raise TypeError(
                f"Field '{self._text_field}' must be a string, got {type(value).__name__} in batch row {index}. "
                "This indicates an upstream validation bug - check source schema or prior transforms."
            )
        lines.append(value)

    report_text = self._render(lines)
    batch = ctx.aggregation_batch
    output = {
        self._output_field: report_text,
        "report_format": self._format,
        "report_index": batch.flush_index,
        "line_start": batch.row_start,
        "line_end": batch.row_end,
        "line_count": batch.batch_size,
        "lines_seen_total": batch.rows_seen_total,
        "flush_trigger": batch.trigger_type,
        "is_end_of_source_report": batch.is_end_of_source,
    }
    output_contract = self._output_contract_for(output)
    return TransformResult.success(
        PipelineRow(output, output_contract),
        success_reason={
            "action": "assembled_report",
            "fields_added": sorted(output),
            "batch_size": batch.batch_size,
            "flush_trigger": batch.trigger_type,
        },
    )
```

Use `SchemaContract(mode="OBSERVED", fields=...)` plus `_align_output_contract()` following `batch_top_k.py`.

- [ ] **Step 5: Add whitelist entries**

Add to `config/cicd/contracts-whitelist.yaml`:
```yaml
  - "src/elspeth/plugins/transforms/report_assemble.py:ReportAssemble.probe_config:return"
  - "src/elspeth/plugins/transforms/report_assemble.py:ReportAssemble.__init__:config"
```

- [ ] **Step 6: Verify unit GREEN**

Run:
```bash
.venv/bin/python -m pytest tests/unit/plugins/transforms/test_report_assemble.py -v
```

Expected: PASS.

## Task 4: Add Production-Path Aggregation Tests

**Files:**
- Create: `tests/integration/pipeline/test_report_assemble_aggregation.py`

- [ ] **Step 1: Add count-trigger pagination test**

Use a temporary text file:
```text
line 1
line 2
line 3
line 4
line 5
```

Use settings equivalent to:
```yaml
source:
  plugin: text
  on_success: lines
  options:
    path: <tmp>/input.txt
    column: line
    strip_whitespace: false
    skip_blank_lines: false
    schema:
      mode: observed
aggregations:
  - name: pages
    plugin: report_assemble
    input: lines
    on_success: output
    on_error: discard
    trigger:
      count: 2
    output_mode: transform
    options:
      schema:
        mode: observed
      text_field: line
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: <tmp>/output.jsonl
      schema:
        mode: observed
```

Assert output JSONL has three rows:
```python
assert rows[0]["report_body"] == "line 1\nline 2"
assert rows[0]["report_index"] == 1
assert rows[0]["line_start"] == 1
assert rows[0]["line_end"] == 2
assert rows[0]["is_end_of_source_report"] is False
assert rows[2]["report_body"] == "line 5"
assert rows[2]["report_index"] == 3
assert rows[2]["line_start"] == 5
assert rows[2]["line_end"] == 5
assert rows[2]["is_end_of_source_report"] is True
```

- [ ] **Step 2: Add end-of-source whole-report test**

Omit `trigger` from the aggregation config. Assert one output row containing all lines, with:
```python
assert row["report_index"] == 1
assert row["line_count"] == 5
assert row["lines_seen_total"] == 5
assert row["flush_trigger"] == "end_of_source"
assert row["is_end_of_source_report"] is True
```

- [ ] **Step 3: Run production-path tests**

Run:
```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_report_assemble_aggregation.py -v
```

Expected: PASS.

## Task 5: Docs And Example

**Files:**
- Modify: `docs/reference/configuration.md`
- Modify: `docs/guides/user-manual.md`
- Create: `examples/report_assemble/input.txt`
- Create: `examples/report_assemble/settings.yaml`
- Create: `examples/report_assemble/README.md`

- [ ] **Step 1: Update plugin tables**

Add `report_assemble` to the transform plugin tables with this description:

```text
`report_assemble` | Assemble a batch of text rows into one report/text row with pagination metadata
```

- [ ] **Step 2: Add aggregation example**

Add an example under aggregation settings:
```yaml
aggregations:
  - name: pages
    plugin: report_assemble
    input: lines
    on_success: output
    on_error: discard
    trigger:
      count: 80
    output_mode: transform
    expected_output_count: 1
    options:
      schema:
        mode: observed
      text_field: line
      format: markdown
      title: "Run report"
```

Document that omitting `trigger` creates one report at end of source.

- [ ] **Step 3: Add runnable example**

`examples/report_assemble/input.txt`:
```text
Alpha finding
Beta finding
Gamma finding
Delta finding
Epsilon finding
```

`examples/report_assemble/settings.yaml` should use the `text` source, `report_assemble` aggregation with `trigger.count: 2`, and a JSON sink at `examples/report_assemble/output/reports.jsonl`.

`examples/report_assemble/README.md` should include:
```bash
uv run elspeth run --settings examples/report_assemble/settings.yaml
```

and explain that the output rows include `report_body`, `report_index`, `line_start`, `line_end`, `line_count`, `lines_seen_total`, `flush_trigger`, and `is_end_of_source_report`.

## Task 6: Hashes, Contracts, And Verification

**Files:**
- Modify: `src/elspeth/plugins/transforms/report_assemble.py`

- [ ] **Step 1: Refresh plugin hash**

Run:
```bash
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth --fix
```

Expected: updates only the `ReportAssemble.source_file_hash` line.

- [ ] **Step 2: Run focused checks**

Run:
```bash
.venv/bin/python -m pytest tests/unit/plugins/transforms/test_report_assemble.py -v
.venv/bin/python -m pytest tests/integration/pipeline/test_report_assemble_aggregation.py -v
.venv/bin/python -m pytest tests/unit/engine/test_executors.py -k aggregation_batch_context -v
```

Expected: PASS.

- [ ] **Step 3: Run repo policy checks for touched surfaces**

Run:
```bash
.venv/bin/python -m scripts.check_contracts
.venv/bin/python -m scripts.cicd.enforce_tier_model check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python -m scripts.cicd.enforce_plugin_hashes check --root src/elspeth
ruff check src/elspeth/contracts src/elspeth/engine/executors/aggregation.py src/elspeth/plugins/transforms/report_assemble.py tests/unit/plugins/transforms/test_report_assemble.py tests/integration/pipeline/test_report_assemble_aggregation.py
```

Expected: PASS.

- [ ] **Step 4: Run aggregation regression slice**

Run:
```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_aggregation_recovery.py tests/unit/core/test_config_aggregation.py tests/integration/pipeline/test_composer_runtime_agreement.py -k "aggregation or batch_stats" -v
```

Expected: PASS. If this slice is too slow in the local sandbox, record the exact timeout or hang and run the narrower failing-file subset first.

## Notes For Implementers

- Do not add logs for per-row or per-report decisions. The output row, node state success reason, batch records, and sink output are the audit surface.
- `is_end_of_source_report` means "this report was emitted by the end-of-source flush." If the final page was emitted by an exact count trigger, it will be false because the engine cannot update already-written sink rows after source completion.
- Keep `report_assemble` deterministic. No LLM calls, no external calls, no clock reads.
- The transform should crash on non-string `text_field` values, matching the existing `line_explode` trust-boundary pattern.
- The output replaces input rows. Do not preserve arbitrary input fields in the report output contract.
