# Multi-Row Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable transforms to output multiple rows from batch inputs (passthrough/transform modes) and single inputs (deaggregation).

**Architecture:** Extend `TransformResult` to support multi-row output via a new `rows` field. The processor checks `output_mode` from aggregation settings and `can_expand` flag on transforms to determine how to handle multi-row results. Each output row gets proper token lineage.

**Tech Stack:** Python dataclasses, existing TokenManager for child token creation

---

## Background

Currently:
- `TransformResult.row` holds ONE output row
- Aggregation only supports `output_mode: single` (N rows → 1 row)
- Transforms cannot expand rows (1 → N)

After this change:
- `TransformResult.rows` can hold MULTIPLE output rows
- `output_mode: passthrough` returns N enriched rows from N input rows
- `output_mode: transform` returns M rows from N input rows
- `can_expand=True` transforms can return multiple rows from single input

## Token Lineage

Multi-row output creates audit trail challenges. The solution:
- **Passthrough mode**: Each output row inherits the token_id of its corresponding input row
- **Transform mode**: Output rows get new token_ids with parent linkage to triggering token
- **Deaggregation**: Output rows get new token_ids with parent linkage to input token

---

### Task 1: Extend TransformResult for Multi-Row Output

**Files:**
- Modify: `src/elspeth/contracts/results.py:60-98`
- Test: `tests/contracts/test_results.py`

**Step 1: Write the failing test**

```python
# tests/contracts/test_results.py

def test_transform_result_multi_row_success():
    """TransformResult.success_multi returns multiple rows."""
    rows = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
    result = TransformResult.success_multi(rows)

    assert result.status == "success"
    assert result.row is None  # Single row field is None
    assert result.rows == rows
    assert len(result.rows) == 2


def test_transform_result_success_single_sets_rows_none():
    """TransformResult.success() keeps rows as None for backwards compat."""
    result = TransformResult.success({"id": 1})

    assert result.status == "success"
    assert result.row == {"id": 1}
    assert result.rows is None


def test_transform_result_is_multi_row():
    """is_multi_row property distinguishes single vs multi output."""
    single = TransformResult.success({"id": 1})
    multi = TransformResult.success_multi([{"id": 1}, {"id": 2}])

    assert single.is_multi_row is False
    assert multi.is_multi_row is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/contracts/test_results.py::test_transform_result_multi_row_success -v`
Expected: FAIL with "AttributeError: type object 'TransformResult' has no attribute 'success_multi'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/contracts/results.py - modify TransformResult class

@dataclass
class TransformResult:
    """Result of a transform operation.

    Use the factory methods to create instances.

    IMPORTANT: status uses Literal["success", "error"], NOT enum, per architecture.
    Audit fields (input_hash, output_hash, duration_ms) are populated by executors.

    For multi-row output (batch transforms, deaggregation):
    - Use success_multi(rows) instead of success(row)
    - rows field contains list of output dicts
    - row field is None when rows is set
    """

    status: Literal["success", "error"]
    row: dict[str, Any] | None
    reason: dict[str, Any] | None
    retryable: bool = False

    # Multi-row output support
    rows: list[dict[str, Any]] | None = None

    # Audit fields - set by executor, not by plugin
    input_hash: str | None = field(default=None, repr=False)
    output_hash: str | None = field(default=None, repr=False)
    duration_ms: float | None = field(default=None, repr=False)

    @property
    def is_multi_row(self) -> bool:
        """Check if this result contains multiple output rows."""
        return self.rows is not None

    @classmethod
    def success(cls, row: dict[str, Any]) -> "TransformResult":
        """Create successful result with single output row."""
        return cls(status="success", row=row, reason=None, rows=None)

    @classmethod
    def success_multi(cls, rows: list[dict[str, Any]]) -> "TransformResult":
        """Create successful result with multiple output rows.

        Use for:
        - Batch transforms with passthrough/transform output_mode
        - Deaggregation transforms (1 input -> N outputs)

        Args:
            rows: List of output row dicts (must not be empty)
        """
        if not rows:
            raise ValueError("success_multi requires at least one row")
        return cls(status="success", row=None, reason=None, rows=rows)

    @classmethod
    def error(
        cls,
        reason: dict[str, Any],
        *,
        retryable: bool = False,
    ) -> "TransformResult":
        """Create error result with reason."""
        return cls(
            status="error",
            row=None,
            reason=reason,
            retryable=retryable,
            rows=None,
        )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/contracts/test_results.py -v -k "transform_result"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/contracts/results.py tests/contracts/test_results.py
git commit -m "feat(contracts): add multi-row support to TransformResult"
```

---

### Task 2: Add can_expand Flag to BaseTransform

**Files:**
- Modify: `src/elspeth/plugins/base.py:24-60`
- Modify: `src/elspeth/plugins/protocols.py:140-150`
- Test: `tests/plugins/test_base.py`

**Step 1: Write the failing test**

```python
# tests/plugins/test_base.py

def test_base_transform_can_expand_default_false():
    """BaseTransform.can_expand defaults to False."""
    from elspeth.plugins.base import BaseTransform

    class SimpleTransform(BaseTransform):
        name = "simple"
        input_schema = None  # Not needed for this test
        output_schema = None

        def process(self, row, ctx):
            return TransformResult.success(row)

    transform = SimpleTransform({})
    assert transform.can_expand is False


def test_base_transform_can_expand_settable():
    """BaseTransform.can_expand can be overridden to True."""
    from elspeth.plugins.base import BaseTransform

    class ExpandingTransform(BaseTransform):
        name = "expander"
        can_expand = True
        input_schema = None
        output_schema = None

        def process(self, row, ctx):
            return TransformResult.success_multi([row, row])  # Duplicate row

    transform = ExpandingTransform({})
    assert transform.can_expand is True
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plugins/test_base.py::test_base_transform_can_expand_default_false -v`
Expected: FAIL with "AttributeError: 'SimpleTransform' object has no attribute 'can_expand'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/base.py - add to BaseTransform class attributes (after is_batch_aware)

    # Deaggregation support - override to True for transforms that can output multiple rows
    # When True, process() may return TransformResult.success_multi(rows)
    can_expand: bool = False
```

```python
# src/elspeth/plugins/protocols.py - add to TransformProtocol (after is_batch_aware)

    # Deaggregation support
    # When True, process() may return TransformResult.success_multi(rows)
    can_expand: bool
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/plugins/test_base.py -v -k "can_expand"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/base.py src/elspeth/plugins/protocols.py tests/plugins/test_base.py
git commit -m "feat(plugins): add can_expand flag for deaggregation transforms"
```

---

### Task 3: Create TokenManager.expand_token() Method

**Files:**
- Modify: `src/elspeth/engine/tokens.py`
- Test: `tests/engine/test_tokens.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_tokens.py

def test_expand_token_creates_children():
    """expand_token creates child tokens for each expanded row."""
    from elspeth.engine.tokens import TokenManager
    from elspeth.contracts import TokenInfo
    from unittest.mock import MagicMock

    recorder = MagicMock()
    manager = TokenManager(recorder)

    parent = TokenInfo(
        row_id="row_1",
        token_id="token_parent",
        row_data={"original": "data"},
    )

    expanded_rows = [
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
        {"id": 3, "value": "c"},
    ]

    children = manager.expand_token(
        parent_token=parent,
        expanded_rows=expanded_rows,
        step_in_pipeline=2,
    )

    assert len(children) == 3
    # All children share same row_id (same source row)
    for child in children:
        assert child.row_id == "row_1"
        assert child.token_id != parent.token_id
        assert child.token_id.startswith("token_")

    # Each child has its expanded row data
    assert children[0].row_data == {"id": 1, "value": "a"}
    assert children[1].row_data == {"id": 2, "value": "b"}
    assert children[2].row_data == {"id": 3, "value": "c"}

    # Recorder should track parent relationships
    assert recorder.record_token_parent.call_count == 3
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_tokens.py::test_expand_token_creates_children -v`
Expected: FAIL with "AttributeError: 'TokenManager' object has no attribute 'expand_token'"

**Step 3: Read existing TokenManager implementation**

Run: `cat src/elspeth/engine/tokens.py` to understand the existing fork_token method pattern.

**Step 4: Write minimal implementation**

```python
# src/elspeth/engine/tokens.py - add method to TokenManager class

    def expand_token(
        self,
        parent_token: TokenInfo,
        expanded_rows: list[dict[str, Any]],
        step_in_pipeline: int,
    ) -> list[TokenInfo]:
        """Create child tokens for deaggregation (1 input -> N outputs).

        Unlike fork_token (which creates parallel paths through the same DAG),
        expand_token creates sequential children that all continue down the
        same path. Used when a transform outputs multiple rows from single input.

        Args:
            parent_token: The token being expanded
            expanded_rows: List of output row dicts
            step_in_pipeline: Current step (for audit)

        Returns:
            List of child TokenInfo, one per expanded row
        """
        children = []
        for i, row_data in enumerate(expanded_rows):
            child_id = self._generate_token_id()
            child = TokenInfo(
                row_id=parent_token.row_id,  # Same source row
                token_id=child_id,
                row_data=row_data,
                branch_name=parent_token.branch_name,  # Inherit branch
            )
            children.append(child)

            # Record lineage
            self._recorder.record_token_parent(
                child_token_id=child_id,
                parent_token_id=parent_token.token_id,
                step_index=step_in_pipeline,
                reason=f"expand:{i}",
            )

        return children
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_tokens.py::test_expand_token_creates_children -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/engine/tokens.py tests/engine/test_tokens.py
git commit -m "feat(engine): add TokenManager.expand_token for deaggregation"
```

---

### Task 4: Handle Multi-Row Output in Processor (Deaggregation)

**Files:**
- Modify: `src/elspeth/engine/processor.py:400-500` (transform processing section)
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_processor_handles_expanding_transform():
    """Processor creates multiple RowResults for expanding transform."""
    # Setup: Create a transform with can_expand=True that returns success_multi
    # Run a row through the processor
    # Assert: Multiple RowResults returned, each with proper token

    from elspeth.plugins.base import BaseTransform
    from elspeth.plugins.results import TransformResult
    from elspeth.contracts import RowOutcome

    class ExpanderTransform(BaseTransform):
        name = "expander"
        can_expand = True
        input_schema = None
        output_schema = None

        def process(self, row, ctx):
            # Expand each row into 2 rows
            return TransformResult.success_multi([
                {**row, "copy": 1},
                {**row, "copy": 2},
            ])

    # ... full test setup with mocks ...
    # processor.process_row returns list with 2 RowResults
    # Each has outcome=COMPLETED and different token_ids
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_processor_handles_expanding_transform -v`
Expected: FAIL (processor doesn't handle multi-row results yet)

**Step 3: Read existing transform handling in processor**

Examine the section around line 450-460 where transform results are processed.

**Step 4: Write minimal implementation**

```python
# src/elspeth/engine/processor.py - modify transform result handling

# After calling transform.process() and getting result:

if result.status == "success":
    if result.is_multi_row and transform.can_expand:
        # Deaggregation: create child tokens for each output row
        child_tokens = self._token_manager.expand_token(
            parent_token=current_token,
            expanded_rows=result.rows,
            step_in_pipeline=step,
        )

        # Queue each child for continued processing
        for child_token in child_tokens:
            child_items.append(_WorkItem(
                token=child_token,
                start_step=step + 1,
            ))

        # Parent token is consumed (EXPANDED)
        return (
            RowResult(
                token=current_token,
                final_data=current_token.row_data,
                outcome=RowOutcome.EXPANDED,
            ),
            child_items,
        )
    else:
        # Single row output (existing logic)
        ...
```

**Step 5: Add EXPANDED to RowOutcome enum**

```python
# src/elspeth/contracts/enums.py - add to RowOutcome

class RowOutcome(str, Enum):
    # ... existing values ...
    EXPANDED = "expanded"  # Row was expanded into multiple child rows
```

**Step 6: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_processor_handles_expanding_transform -v`
Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/engine/processor.py src/elspeth/contracts/enums.py tests/engine/test_processor.py
git commit -m "feat(engine): handle deaggregation in processor"
```

---

### Task 5: Handle passthrough Output Mode in Aggregation

**Files:**
- Modify: `src/elspeth/engine/processor.py:144-223` (_process_batch_aggregation_node)
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_aggregation_passthrough_mode():
    """Passthrough mode returns N enriched rows from N input rows."""
    # Setup: aggregation with output_mode="passthrough"
    # Buffer 3 rows, trigger fires
    # Transform returns success_multi with 3 enriched rows
    # Assert: 3 RowResults returned, each with original token_id
    pass  # Full test implementation
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_passthrough_mode -v`
Expected: FAIL (only single mode handled)

**Step 3: Write minimal implementation**

```python
# src/elspeth/engine/processor.py - modify _process_batch_aggregation_node

def _process_batch_aggregation_node(self, ...):
    # ... existing buffering logic ...

    if self._aggregation_executor.should_flush(node_id):
        buffered_rows = self._aggregation_executor.flush_buffer(node_id)
        buffered_tokens = self._aggregation_executor.get_buffered_tokens(node_id)

        result = transform.process(buffered_rows, ctx)

        if result.status == "success":
            # Get output_mode from aggregation settings
            agg_settings = self._aggregation_settings.get(node_id)
            output_mode = agg_settings.output_mode if agg_settings else "single"

            if output_mode == "single":
                # Existing: N rows -> 1 row
                final_data = result.row if result.row is not None else {}
                updated_token = TokenInfo(
                    row_id=current_token.row_id,
                    token_id=current_token.token_id,
                    row_data=final_data,
                    branch_name=current_token.branch_name,
                )
                return (
                    RowResult(token=updated_token, final_data=final_data, outcome=RowOutcome.COMPLETED),
                    child_items,
                )

            elif output_mode == "passthrough":
                # N rows -> N enriched rows, preserving original tokens
                if not result.is_multi_row:
                    raise ValueError(
                        f"passthrough mode requires success_multi result, got single row"
                    )
                if len(result.rows) != len(buffered_tokens):
                    raise ValueError(
                        f"passthrough mode requires same number of output rows ({len(result.rows)}) "
                        f"as input rows ({len(buffered_tokens)})"
                    )

                # Queue all but last as work items, return last as result
                for i, (token, row_data) in enumerate(zip(buffered_tokens[:-1], result.rows[:-1])):
                    updated_token = TokenInfo(
                        row_id=token.row_id,
                        token_id=token.token_id,
                        row_data=row_data,
                        branch_name=token.branch_name,
                    )
                    child_items.append(_WorkItem(token=updated_token, start_step=step + 1))

                # Return last row as the immediate result
                last_token = buffered_tokens[-1]
                last_data = result.rows[-1]
                updated_token = TokenInfo(
                    row_id=last_token.row_id,
                    token_id=last_token.token_id,
                    row_data=last_data,
                    branch_name=last_token.branch_name,
                )
                return (
                    RowResult(token=updated_token, final_data=last_data, outcome=RowOutcome.COMPLETED),
                    child_items,
                )

            elif output_mode == "transform":
                # N rows -> M rows (variable)
                # Similar to deaggregation - create new tokens for all output rows
                ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_passthrough_mode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): implement passthrough output mode for aggregation"
```

---

### Task 6: Handle transform Output Mode in Aggregation

**Files:**
- Modify: `src/elspeth/engine/processor.py:144-223`
- Test: `tests/engine/test_processor.py`

**Step 1: Write the failing test**

```python
# tests/engine/test_processor.py

def test_aggregation_transform_mode():
    """Transform mode returns M rows from N input rows."""
    # Setup: aggregation with output_mode="transform"
    # Buffer 5 rows, trigger fires
    # Transform returns success_multi with 2 rows (e.g., split by category)
    # Assert: 2 RowResults with new token_ids, parent lineage to triggering token
    pass
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_transform_mode -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/elspeth/engine/processor.py - add transform mode handling

elif output_mode == "transform":
    # N rows -> M rows with new tokens
    if not result.is_multi_row:
        # Single row output is valid for transform mode
        final_data = result.row if result.row is not None else {}
        updated_token = TokenInfo(
            row_id=current_token.row_id,
            token_id=current_token.token_id,
            row_data=final_data,
            branch_name=current_token.branch_name,
        )
        return (
            RowResult(token=updated_token, final_data=final_data, outcome=RowOutcome.COMPLETED),
            child_items,
        )

    # Multi-row output: create new tokens for each
    expanded_tokens = self._token_manager.expand_token(
        parent_token=current_token,
        expanded_rows=result.rows,
        step_in_pipeline=step,
    )

    # Queue all but last as work items
    for token in expanded_tokens[:-1]:
        child_items.append(_WorkItem(token=token, start_step=step + 1))

    # Return last as immediate result
    last_token = expanded_tokens[-1]
    return (
        RowResult(token=last_token, final_data=last_token.row_data, outcome=RowOutcome.COMPLETED),
        child_items,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/engine/test_processor.py::test_aggregation_transform_mode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/engine/processor.py tests/engine/test_processor.py
git commit -m "feat(engine): implement transform output mode for aggregation"
```

---

### Task 7: Create Example Deaggregation Transform

**Files:**
- Create: `src/elspeth/plugins/transforms/json_explode.py`
- Modify: `src/elspeth/plugins/transforms/hookimpl.py`
- Test: `tests/plugins/transforms/test_json_explode.py`

**Step 1: Write the failing test**

```python
# tests/plugins/transforms/test_json_explode.py

def test_json_explode_expands_array():
    """JSONExplode expands array field into multiple rows."""
    from elspeth.plugins.transforms.json_explode import JSONExplode
    from unittest.mock import MagicMock

    transform = JSONExplode({"array_field": "items", "schema": {"fields": "dynamic"}})
    ctx = MagicMock()

    row = {
        "id": 1,
        "items": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
    }

    result = transform.process(row, ctx)

    assert result.status == "success"
    assert result.is_multi_row
    assert len(result.rows) == 3
    assert result.rows[0] == {"id": 1, "item": {"name": "a"}, "item_index": 0}
    assert result.rows[1] == {"id": 1, "item": {"name": "b"}, "item_index": 1}
    assert result.rows[2] == {"id": 1, "item": {"name": "c"}, "item_index": 2}
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/plugins/transforms/test_json_explode.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/transforms/json_explode.py
"""JSON array explode transform - deaggregation example.

Expands a JSON array field into multiple rows, one per array element.
Useful for flattening nested data structures.

Example:
    Input:  {"id": 1, "items": [{"x": 1}, {"x": 2}]}
    Output: [
        {"id": 1, "item": {"x": 1}, "item_index": 0},
        {"id": 1, "item": {"x": 2}, "item_index": 1},
    ]
"""

from typing import Any

from pydantic import Field

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class JSONExplodeConfig(TransformDataConfig):
    """Configuration for JSON explode transform."""

    array_field: str = Field(description="Name of the array field to explode")
    output_field: str = Field(
        default="item",
        description="Name for the exploded element in output rows"
    )
    include_index: bool = Field(
        default=True,
        description="Whether to include item_index field"
    )


class JSONExplode(BaseTransform):
    """Explode a JSON array field into multiple rows.

    This is a deaggregation transform (can_expand=True) that takes
    one input row and produces N output rows, one per array element.
    """

    name = "json_explode"
    can_expand = True  # CRITICAL: enables multi-row output

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = JSONExplodeConfig.from_dict(config)
        self._array_field = cfg.array_field
        self._output_field = cfg.output_field
        self._include_index = cfg.include_index
        self._on_error = cfg.on_error

        # Schema setup
        assert cfg.schema_config is not None
        schema = create_schema_from_config(cfg.schema_config, "JSONExplodeSchema", allow_coercion=False)
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Explode array field into multiple rows."""
        array_value = row.get(self._array_field)

        if array_value is None:
            return TransformResult.error(
                {"reason": f"Field '{self._array_field}' is None or missing"}
            )

        if not isinstance(array_value, list):
            return TransformResult.error(
                {"reason": f"Field '{self._array_field}' is not an array"}
            )

        if len(array_value) == 0:
            # Empty array - return single row with None item
            output = {k: v for k, v in row.items() if k != self._array_field}
            output[self._output_field] = None
            if self._include_index:
                output["item_index"] = None
            return TransformResult.success(output)

        # Explode array into multiple rows
        output_rows = []
        for i, item in enumerate(array_value):
            output = {k: v for k, v in row.items() if k != self._array_field}
            output[self._output_field] = item
            if self._include_index:
                output["item_index"] = i
            output_rows.append(output)

        return TransformResult.success_multi(output_rows)

    def close(self) -> None:
        pass
```

**Step 4: Register in hookimpl**

```python
# src/elspeth/plugins/transforms/hookimpl.py - add import and registration

from elspeth.plugins.transforms.json_explode import JSONExplode

# In get_transform_plugins():
return [PassThrough, FieldMapper, BatchStats, JSONExplode]
```

**Step 5: Run test to verify it passes**

Run: `uv run pytest tests/plugins/transforms/test_json_explode.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/transforms/json_explode.py src/elspeth/plugins/transforms/hookimpl.py tests/plugins/transforms/test_json_explode.py
git commit -m "feat(plugins): add JSONExplode deaggregation transform"
```

---

### Task 8: Create Integration Test with Example Pipeline

**Files:**
- Create: `examples/deaggregation/settings.yaml`
- Create: `examples/deaggregation/input.json`
- Test: `tests/integration/test_deaggregation.py`

**Step 1: Create example input**

```json
// examples/deaggregation/input.json
[
  {"order_id": 1, "items": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]},
  {"order_id": 2, "items": [{"sku": "C3", "qty": 5}]},
  {"order_id": 3, "items": [{"sku": "A1", "qty": 1}, {"sku": "D4", "qty": 3}, {"sku": "E5", "qty": 2}]}
]
```

**Step 2: Create example settings**

```yaml
# examples/deaggregation/settings.yaml
datasource:
  plugin: json
  options:
    path: examples/deaggregation/input.json
    schema:
      mode: free
      fields: dynamic

row_plugins:
  - plugin: json_explode
    options:
      array_field: items
      output_field: item
      include_index: true
      schema:
        fields: dynamic

sinks:
  output:
    plugin: csv
    options:
      path: examples/deaggregation/output/order_items.csv
      schema:
        fields: dynamic

output_sink: output

landscape:
  url: sqlite:///examples/deaggregation/runs/audit.db
```

**Step 3: Write integration test**

```python
# tests/integration/test_deaggregation.py

def test_deaggregation_pipeline():
    """Full pipeline with deaggregation transform."""
    import subprocess
    import csv
    from pathlib import Path

    # Run pipeline
    result = subprocess.run(
        ["uv", "run", "elspeth", "run", "-s", "examples/deaggregation/settings.yaml", "--execute"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    # Check output
    output_path = Path("examples/deaggregation/output/order_items.csv")
    assert output_path.exists()

    with open(output_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # 3 orders with 2+1+3 = 6 items
    assert len(rows) == 6

    # Verify structure
    assert all("order_id" in row for row in rows)
    assert all("item" in row for row in rows)
    assert all("item_index" in row for row in rows)
```

**Step 4: Run integration test**

Run: `uv run pytest tests/integration/test_deaggregation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/deaggregation/ tests/integration/test_deaggregation.py
git commit -m "feat(examples): add deaggregation example pipeline"
```

---

### Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run type checker**

Run: `uv run mypy src/elspeth/`
Expected: No errors

**Step 3: Run linter**

Run: `uv run ruff check src/elspeth/`
Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: fix any remaining issues from multi-row implementation"
```

---

## Summary

After completing all tasks:

| Feature | Status |
|---------|--------|
| `TransformResult.success_multi()` | ✅ |
| `TransformResult.is_multi_row` | ✅ |
| `BaseTransform.can_expand` | ✅ |
| `TokenManager.expand_token()` | ✅ |
| `RowOutcome.EXPANDED` | ✅ |
| Processor deaggregation handling | ✅ |
| `output_mode: passthrough` | ✅ |
| `output_mode: transform` | ✅ |
| `JSONExplode` example transform | ✅ |
| Integration test | ✅ |

**Total estimated time:** 2-3 hours

**Key architectural decisions:**
1. Multi-row output uses separate `rows` field (not overloading `row`)
2. Deaggregation creates child tokens with parent lineage
3. Passthrough mode preserves original token IDs
4. Transform mode creates new tokens for variable output
