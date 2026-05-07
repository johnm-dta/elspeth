# Quick-Win Deduplication — SpanFactory + Manager Error Formatting

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate duplicated code in `engine/spans.py` (7× no-op guard) and `plugins/infrastructure/manager.py` (3× validate-then-format error pattern).

**Architecture:** Pure extraction refactors — no behavioral changes. Each task extracts a private helper, rewrites callers to use it, and verifies via existing tests.

**Tech Stack:** Python, contextlib, OpenTelemetry (optional dependency)

**Filigree issues:** `elspeth-d5190ef4e1` (SpanFactory), `elspeth-4de05937be` (manager.py)

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/elspeth/engine/spans.py` | Add `_make_span`, rewrite 7 public methods |
| Modify | `src/elspeth/plugins/infrastructure/manager.py` | Add `_raise_if_invalid`, rewrite 3 `create_*` methods |
| Verify | `tests/unit/engine/test_spans.py` | Existing tests — no changes needed |
| Verify | `tests/unit/plugins/test_validation_integration.py` | Existing tests — no changes needed |

---

### Task 1: SpanFactory — extract `_make_span` helper

**Files:**
- Modify: `src/elspeth/engine/spans.py`
- Verify: `tests/unit/engine/test_spans.py`

- [ ] **Step 1: Run existing span tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_spans.py -v`
Expected: All tests PASS

- [ ] **Step 2: Add `_make_span` private contextmanager**

Add this method to `SpanFactory` between `enabled` property (line 72) and `run_span` (line 74):

```python
@contextmanager
def _make_span(
    self,
    name: str,
    attributes: dict[str, Any],
) -> Iterator["Span | NoOpSpan"]:
    """Create a span with attributes, or yield no-op if tracing is disabled.

    Args:
        name: Span name (e.g., "run", "source:csv", "transform:field_mapper")
        attributes: Key-value pairs to set on the span. Values must already
            be in their final form (e.g., token_ids already converted to tuple).
    """
    if self._tracer is None:
        yield self._NOOP_SPAN
        return

    with self._tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span
```

- [ ] **Step 3: Rewrite `run_span` to use `_make_span`**

Replace the body of `run_span` (lines 84–90) with:

```python
@contextmanager
def run_span(self, run_id: str) -> Iterator["Span | NoOpSpan"]:
    """Create a span for the entire run.

    Args:
        run_id: Run identifier

    Yields:
        Span or NoOpSpan if tracing disabled (never None - uniform interface)
    """
    with self._make_span("run", {"run.id": run_id}) as span:
        yield span
```

- [ ] **Step 4: Rewrite `source_span`**

```python
@contextmanager
def source_span(self, source_name: str) -> Iterator["Span | NoOpSpan"]:
    """Create a span for source loading.

    Args:
        source_name: Name of the source plugin

    Yields:
        Span or NoOpSpan
    """
    with self._make_span(f"source:{source_name}", {
        "plugin.name": source_name,
        "plugin.type": "source",
    }) as span:
        yield span
```

- [ ] **Step 5: Rewrite `row_span`**

```python
@contextmanager
def row_span(
    self,
    row_id: str,
    token_id: str,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for processing a row.

    Args:
        row_id: Row identifier
        token_id: Token identifier

    Yields:
        Span or NoOpSpan
    """
    with self._make_span("row", {
        "row.id": row_id,
        "token.id": token_id,
    }) as span:
        yield span
```

- [ ] **Step 6: Rewrite `transform_span`**

This one has mutually-exclusive `token_id`/`token_ids` and multiple optional attrs — build the dict conditionally:

```python
@contextmanager
def transform_span(
    self,
    transform_name: str,
    *,
    node_id: str | None = None,
    input_hash: str | None = None,
    token_id: str | None = None,
    token_ids: Sequence[str] | None = None,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for a transform operation.

    Args:
        transform_name: Name of the transform plugin
        node_id: Unique node identifier for disambiguation
        input_hash: Optional input data hash
        token_id: Token identifier for single-row transforms
        token_ids: Token identifiers for batch transforms (aggregation flush)

    Note:
        Use token_id for single-row transforms (most common case).
        Use token_ids for batch/aggregation transforms that process multiple tokens.
        These are mutually exclusive - if both provided, token_ids takes precedence.

        node_id enables correlation with Landscape node_states when multiple
        instances of the same plugin type exist in a pipeline.

    Yields:
        Span or NoOpSpan
    """
    attrs: dict[str, Any] = {
        "plugin.name": transform_name,
        "plugin.type": "transform",
    }
    if node_id is not None:
        attrs["node.id"] = node_id
    if input_hash is not None:
        attrs["input.hash"] = input_hash
    if token_ids is not None:
        attrs["token.ids"] = tuple(token_ids)
    elif token_id is not None:
        attrs["token.id"] = token_id

    with self._make_span(f"transform:{transform_name}", attrs) as span:
        yield span
```

- [ ] **Step 7: Rewrite `gate_span`**

```python
@contextmanager
def gate_span(
    self,
    gate_name: str,
    *,
    node_id: str | None = None,
    input_hash: str | None = None,
    token_id: str | None = None,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for a gate operation.

    Args:
        gate_name: Name of the gate (from GateSettings)
        node_id: Unique node identifier for disambiguation
        input_hash: Optional input data hash
        token_id: Token identifier for the token being evaluated

    Yields:
        Span or NoOpSpan
    """
    attrs: dict[str, Any] = {
        "plugin.name": gate_name,
        "plugin.type": "gate",
    }
    if node_id is not None:
        attrs["node.id"] = node_id
    if input_hash is not None:
        attrs["input.hash"] = input_hash
    if token_id is not None:
        attrs["token.id"] = token_id

    with self._make_span(f"gate:{gate_name}", attrs) as span:
        yield span
```

- [ ] **Step 8: Rewrite `aggregation_span`**

```python
@contextmanager
def aggregation_span(
    self,
    aggregation_name: str,
    *,
    node_id: str | None = None,
    input_hash: str | None = None,
    batch_id: str | None = None,
    token_ids: Sequence[str] | None = None,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for an aggregation flush.

    Args:
        aggregation_name: Name of the aggregation plugin
        node_id: Unique node identifier for disambiguation
        input_hash: Input data hash for trace-to-audit correlation
        batch_id: Optional batch identifier
        token_ids: Token identifiers in the batch

    Note:
        Aggregation batches process multiple tokens, so this uses token_ids (plural).
        The token.ids attribute is a tuple for OpenTelemetry compatibility.

    Yields:
        Span or NoOpSpan
    """
    attrs: dict[str, Any] = {
        "plugin.name": aggregation_name,
        "plugin.type": "aggregation",
    }
    if node_id is not None:
        attrs["node.id"] = node_id
    if input_hash is not None:
        attrs["input.hash"] = input_hash
    if batch_id is not None:
        attrs["batch.id"] = batch_id
    if token_ids is not None:
        attrs["token.ids"] = tuple(token_ids)

    with self._make_span(f"aggregation:{aggregation_name}", attrs) as span:
        yield span
```

- [ ] **Step 9: Rewrite `sink_span`**

```python
@contextmanager
def sink_span(
    self,
    sink_name: str,
    *,
    node_id: str | None = None,
    token_ids: Sequence[str] | None = None,
) -> Iterator["Span | NoOpSpan"]:
    """Create a span for a sink write.

    Args:
        sink_name: Name of the sink plugin
        node_id: Unique node identifier for disambiguation
        token_ids: Token identifiers being written in this batch

    Note:
        Sinks batch-write multiple tokens, so this uses token_ids (plural).
        The token.ids attribute is a tuple for OpenTelemetry compatibility.

    Yields:
        Span or NoOpSpan
    """
    attrs: dict[str, Any] = {
        "plugin.name": sink_name,
        "plugin.type": "sink",
    }
    if node_id is not None:
        attrs["node.id"] = node_id
    if token_ids is not None:
        attrs["token.ids"] = tuple(token_ids)

    with self._make_span(f"sink:{sink_name}", attrs) as span:
        yield span
```

- [ ] **Step 10: Run all span tests**

Run: `.venv/bin/python -m pytest tests/unit/engine/test_spans.py -v`
Expected: All tests PASS — identical behavior, just deduplicated

- [ ] **Step 11: Commit**

```bash
git add src/elspeth/engine/spans.py
git commit -m "refactor(spans): extract _make_span to eliminate 7× no-op guard duplication"
```

---

### Task 2: manager.py — extract `_raise_if_invalid` helper

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/manager.py`
- Verify: `tests/unit/plugins/test_validation_integration.py`

- [ ] **Step 1: Run existing validation integration tests to establish baseline**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation_integration.py -v`
Expected: All tests PASS

- [ ] **Step 2: Add `_raise_if_invalid` module-level function and update import**

Add `ValidationError` to the import from `validation.py`, then add the helper function above the `PluginManager` class definition (after the import block, before line 26):

```python
from elspeth.plugins.infrastructure.validation import PluginConfigValidator, ValidationError


def _raise_if_invalid(errors: list[ValidationError], label: str, name: str) -> None:
    """Raise ValueError with formatted message if validation errors exist."""
    if errors:
        error_lines = [f"  - {err.field}: {err.message}" for err in errors]
        raise ValueError(
            f"Invalid configuration for {label} '{name}':\n" + "\n".join(error_lines)
        )
```

- [ ] **Step 3: Rewrite `create_source` to use `_raise_if_invalid`**

Replace lines 220–226 with:

```python
        # Validate config first
        errors = self._validator.validate_source_config(source_type, config)
        _raise_if_invalid(errors, "source", source_type)
```

- [ ] **Step 4: Rewrite `create_transform` to use `_raise_if_invalid`**

Replace lines 248–253 with:

```python
        # Validate config first
        errors = self._validator.validate_transform_config(transform_type, config)
        _raise_if_invalid(errors, "transform", transform_type)
```

- [ ] **Step 5: Rewrite `create_sink` to use `_raise_if_invalid`**

Replace lines 275–280 with:

```python
        # Validate config first
        errors = self._validator.validate_sink_config(sink_type, config)
        _raise_if_invalid(errors, "sink", sink_type)
```

- [ ] **Step 6: Run validation integration tests**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation_integration.py -v`
Expected: All tests PASS — error messages are identical

- [ ] **Step 7: Run full plugin test suite as regression check**

Run: `.venv/bin/python -m pytest tests/unit/plugins/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/plugins/infrastructure/manager.py
git commit -m "refactor(manager): extract _raise_if_invalid to eliminate 3× error formatting duplication"
```
