# ELSPETH Plugin Development Guide

Create custom sources, transforms, gates, and sinks for ELSPETH pipelines.

> **Quick Links:**
> - [5-Minute Transform](#5-minute-transform) - Get started fast
> - [Plugin Types](#plugin-types-overview) - Choose the right type
> - [Contract Tests](#contract-testing) - Verify your plugin works

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [5-Minute Transform](#5-minute-transform)
- [Plugin Types Overview](#plugin-types-overview)
- [Creating Transforms](#creating-a-transform-plugin)
- [Creating Sources](#creating-a-source-plugin)
- [Creating Sinks](#creating-a-sink-plugin)
- [Creating Gates](#creating-a-gate-plugin)
- [Plugin Registration](#plugin-registration)
- [Schema Configuration](#schema-configuration)
- [Contract Testing](#contract-testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.12+** with type hints, dataclasses
- **Pydantic v2** for config validation
- **ELSPETH concepts** - Read `CLAUDE.md` for the Three-Tier Trust Model

```bash
git clone https://github.com/johnm-dta/elspeth.git && cd elspeth
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## 5-Minute Transform

The fastest path to a working plugin:

```python
# src/elspeth/plugins/transforms/double_value.py
from typing import Any
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class DoubleValueTransform(BaseTransform):
    """Double a numeric field value."""

    name = "double_value"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TransformDataConfig.from_dict(config)
        self._field = config.get("field", "value")
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config, "DoubleValueSchema", allow_coercion=False
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        if self._field not in row:
            return TransformResult.error({"reason": "missing_field", "field": self._field})

        try:
            result = float(row[self._field]) * 2
        except (TypeError, ValueError) as e:
            return TransformResult.error({"reason": "conversion_failed", "error": str(e)})

        output = dict(row)
        output[self._field] = result
        return TransformResult.success(output)

    def close(self) -> None:
        pass
```

**Register it:**

```python
# src/elspeth/plugins/transforms/hookimpl.py
from elspeth.plugins.transforms.double_value import DoubleValueTransform

class ElspethBuiltinTransforms:
    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        return [..., DoubleValueTransform]
```

**Use it:**

```yaml
transforms:
  - plugin: double_value
    options:
      schema:
        fields: dynamic
      field: price
```

**Test it:**

```python
# tests/contracts/transform_contracts/test_double_value_contract.py
from elspeth.plugins.transforms.double_value import DoubleValueTransform
from .test_transform_protocol import TransformContractPropertyTestBase

class TestDoubleValueContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self):
        return DoubleValueTransform({"schema": {"fields": "dynamic"}, "field": "value"})

    @pytest.fixture
    def valid_input(self):
        return {"id": 1, "value": 10.0}
```

---

## Plugin Types Overview

ELSPETH follows the **Sense/Decide/Act** model:

```
SOURCE (Sense) → TRANSFORM/GATE (Decide) → SINK (Act)
```

| Type | Purpose | Base Class | Key Method |
|------|---------|------------|------------|
| **Source** | Load data from external systems | `BaseSource` | `load()` |
| **Transform** | Process/classify rows | `BaseTransform` | `process()` |
| **Gate** | Route rows to destinations | `BaseGate` | `evaluate()` |
| **Sink** | Output data | `BaseSink` | `write()` |

### The Trust Model: Who Can Coerce Data?

| Plugin Type | Coercion Allowed? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | External data boundary - normalize incoming data |
| **Transform** | ❌ No | Wrong types = upstream bug → crash |
| **Gate** | ❌ No | Wrong types = upstream bug → crash |
| **Sink** | ❌ No | Wrong types = upstream bug → crash |

**Rule:** Only sources touch untrusted external data. Everything else trusts upstream.

---

## Creating a Transform Plugin

Transforms process rows one at a time (or in batches for aggregation).

### Basic Transform

```python
from typing import Any
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class MyTransformConfig(TransformDataConfig):
    """Config with your custom fields."""
    multiplier: float = 1.0
    target_field: str = "value"


class MyTransform(BaseTransform):
    """Multiply a field by a configured factor."""

    name = "my_transform"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MyTransformConfig.from_dict(config)
        self._multiplier = cfg.multiplier
        self._target_field = cfg.target_field
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config, "MyTransformSchema", allow_coercion=False
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        if self._target_field not in row:
            return TransformResult.error({
                "reason": "missing_field",
                "field": self._target_field,
            })

        # Wrap operations on row values (their data can fail)
        try:
            result = float(row[self._target_field]) * self._multiplier
        except (TypeError, ValueError) as e:
            return TransformResult.error({
                "reason": "conversion_failed",
                "error": str(e),
            })

        output = dict(row)
        output[self._target_field] = result
        return TransformResult.success(output)

    def close(self) -> None:
        pass
```

### Required Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Unique plugin identifier (class attribute) |
| `input_schema` | `type[PluginSchema]` | Expected input row schema |
| `output_schema` | `type[PluginSchema]` | Produced output row schema |
| `determinism` | `Determinism` | Reproducibility level (default: `DETERMINISTIC`) |
| `plugin_version` | `str` | Plugin version for audit trail (default: `"0.0.0"`) |

**Determinism levels:**
- `DETERMINISTIC` - Same input always produces same output
- `EXTERNAL_CALL` - Calls external service (LLM, API)
- `IO_READ` - Reads from external source
- `IO_WRITE` - Writes to external sink

### TransformResult Options

```python
# Success - transformed row
TransformResult.success({"id": 1, "value": 20.0})

# Error - row failed processing (routes to on_error sink)
TransformResult.error({"reason": "division_by_zero"})

# Multiple outputs (requires creates_tokens=True)
TransformResult.success_multi([row1, row2, row3])
```

<details>
<summary><strong>Advanced: Batch-Aware Transforms</strong></summary>

For aggregation transforms that process multiple rows together:

```python
class BatchStatsTransform(BaseTransform):
    name = "batch_stats"
    is_batch_aware = True  # Receives list[dict] instead of dict

    def process(self, rows: list[dict[str, Any]], ctx: PluginContext) -> TransformResult:
        if not rows:
            return TransformResult.success({"count": 0, "sum": 0})

        total = sum(r.get("value", 0) for r in rows)
        return TransformResult.success({
            "count": len(rows),
            "sum": total,
            "mean": total / len(rows),
        })
```

**Pipeline config:**
```yaml
aggregations:
  - name: compute_stats
    plugin: batch_stats
    trigger:
      count: 100  # Process every 100 rows
    output_mode: single  # N inputs → 1 output
```

</details>

<details>
<summary><strong>Advanced: Deaggregation Transforms</strong></summary>

For transforms that expand one row into multiple rows:

```python
class ExpandItemsTransform(BaseTransform):
    name = "expand_items"
    creates_tokens = True  # Engine creates new tokens for each output

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        items = row["items"]  # Trust: source validated this is a list

        output_rows = [
            {**row, "item": item, "item_index": i}
            for i, item in enumerate(items)
        ]

        return TransformResult.success_multi(output_rows)
```

**Token semantics:**
- `creates_tokens=True` + `success_multi()` → New tokens per output
- `creates_tokens=False` + `success_multi()` → RuntimeError

</details>

---

## Creating a Source Plugin

Sources load data from external systems. **Only sources can coerce data.**

```python
from collections.abc import Iterator
from typing import Any
from pydantic import ValidationError

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import SourceDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class MySourceConfig(SourceDataConfig):
    """Inherits path, schema, and on_validation_failure."""
    skip_header: bool = True


class MySource(BaseSource):
    """Load data from a custom format."""

    name = "my_source"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MySourceConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._skip_header = cfg.skip_header
        self._on_validation_failure = cfg.on_validation_failure

        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=True for sources
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config, "MySourceSchema", allow_coercion=True
        )
        self.output_schema = self._schema_class

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

        with open(self._path) as f:
            lines = f.readlines()

        if self._skip_header and lines:
            lines = lines[1:]

        for line in lines:
            row = self._parse_line(line)

            try:
                validated = self._schema_class.model_validate(row)
                yield SourceRow.valid(validated.to_row())

            except ValidationError as e:
                ctx.record_validation_error(
                    row=row,
                    error=str(e),
                    schema_mode=self._schema_config.mode or "dynamic",
                    destination=self._on_validation_failure,
                )
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=row,
                        error=str(e),
                        destination=self._on_validation_failure,
                    )

    def _parse_line(self, line: str) -> dict[str, Any]:
        parts = line.strip().split(",")
        return {"id": parts[0], "value": parts[1]} if len(parts) >= 2 else {}

    def close(self) -> None:
        pass
```

### SourceRow Options

```python
# Valid row - proceed to processing
SourceRow.valid({"id": 1, "value": 100})

# Quarantined - route to on_validation_failure sink
SourceRow.quarantined(row=raw_row, error="Invalid type", destination="quarantine_sink")
```

---

## Creating a Sink Plugin

Sinks output data and **must return audit information** including content hashes.

```python
import hashlib
from pathlib import Path
from typing import Any

from elspeth.contracts import ArtifactDescriptor
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class MySinkConfig(PathConfig):
    append: bool = False


class MySink(BaseSink):
    """Write data to a custom format."""

    name = "my_sink"
    idempotent = False  # Appends are not idempotent

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MySinkConfig.from_dict(config)
        self._path = Path(cfg.path)
        self._append = cfg.append

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config, "MySinkSchema", allow_coercion=False
        )
        self.input_schema = schema

        self._file = None
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def on_start(self, ctx: PluginContext) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a" if self._append else "w")
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
        for row in rows:
            line = ",".join(str(v) for v in row.values()) + "\n"
            self._file.write(line)
            self._hasher.update(line.encode())
            self._bytes_written += len(line.encode())

        # REQUIRED: Return artifact with content hash for audit
        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=self._hasher.hexdigest(),
            size_bytes=self._bytes_written,
        )

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
```

### ArtifactDescriptor Requirements

The audit trail requires:
- `content_hash` - SHA-256 hex digest of output content
- `size_bytes` - Output size in bytes

---

## Creating a Gate Plugin

Gates route rows to different destinations. **Most gates should use config expressions:**

```yaml
gates:
  - name: quality_check
    condition: "row['score'] >= 0.8"
    routes:
      "true": high_quality_sink
      "false": continue
```

**Expression parser allows:**
- Field access: `row['field']`, `row.get('field', default)`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`, `in`, `not in`
- Boolean ops: `and`, `or`, `not`
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`
- Literals: strings, numbers, booleans, None, lists, dicts

**Expression parser forbids:**
- Function calls (except `row.get()`) - `int(x)`, `len(x)`, `str(x)` are NOT allowed
- Comprehensions, lambdas, f-strings, attribute access (except `row.get`)

**Create a plugin gate only when expressions can't express your logic:**

```python
from typing import Any
from elspeth.contracts import Determinism, RoutingAction
from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult
from elspeth.plugins.schema_factory import create_schema_from_config


class MLGate(BaseGate):
    """Route based on ML model prediction."""

    name = "ml_gate"
    determinism = Determinism.EXTERNAL_CALL  # ML inference is external
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._threshold = config.get("threshold", 0.7)
        self._model_path = config.get("model_path")
        self._model = None

        schema_config = config.get("schema", {"fields": "dynamic"})
        schema = create_schema_from_config(
            schema_config, "MLGateSchema", allow_coercion=False
        )
        self.input_schema = schema
        self.output_schema = schema

    def on_start(self, ctx: PluginContext) -> None:
        if self._model_path:
            import pickle
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        if self._model:
            features = [row.get("f1", 0), row.get("f2", 0)]
            score = self._model.predict_proba([features])[0][1]

            if score > self._threshold:
                row["ml_score"] = score
                return GateResult(row=row, action=RoutingAction.route("flagged"))

        return GateResult(row=row, action=RoutingAction.continue_())

    def close(self) -> None:
        self._model = None
```

### GateResult Options

```python
# Continue to next node
GateResult(row=row, action=RoutingAction.continue_())

# Route to named sink
GateResult(row=row, action=RoutingAction.route("review_queue"))

# Fork to multiple paths
GateResult(row=row, action=RoutingAction.fork(["path_a", "path_b"]))
```

### When to Use Each Approach

| Scenario | Use |
|----------|-----|
| `row['score'] > threshold` | Config expression |
| Multiple conditions with AND/OR | Config expression |
| ML model inference | Plugin gate |
| External API validation | Plugin gate |
| Rate limiting / quotas | Plugin gate |
| Complex business rules | Plugin gate |

---

## Plugin Registration

Register in **two places**:

### 1. Hook Implementation (pluggy)

```python
# src/elspeth/plugins/transforms/hookimpl.py
from elspeth.plugins.transforms.my_transform import MyTransform

class ElspethBuiltinTransforms:
    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        return [..., MyTransform]  # Add here
```

### 2. CLI Registry

```python
# src/elspeth/cli.py (in _execute_pipeline and _resume_run)
from elspeth.plugins.transforms.my_transform import MyTransform

TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
    ...,
    "my_transform": MyTransform,  # Add here
}
```

---

## Schema Configuration

All data-processing plugins require schema configuration.

### Schema Modes

| Mode | Behavior | Extra Fields |
|------|----------|--------------|
| `dynamic` | Accept any fields | Allowed |
| `strict` | Only declared fields | Rejected |
| `free` | Declared required, extras allowed | Allowed |

### YAML Examples

```yaml
# Accept anything
schema:
  fields: dynamic

# Strict - only these fields
schema:
  mode: strict
  fields:
    - "id: int"
    - "name: str"
    - "active: bool"

# At least these, allow more
schema:
  mode: free
  fields:
    - "id: int"
    - "value: float"
```

### Supported Types

`str`, `int`, `float`, `bool`, `any`

---

## Contract Testing

Every plugin **must** pass protocol contract tests.

### Quick Test Setup

```python
# tests/contracts/transform_contracts/test_my_transform_contract.py
import pytest
from elspeth.plugins.transforms.my_transform import MyTransform
from .test_transform_protocol import TransformContractPropertyTestBase


class TestMyTransformContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self):
        return MyTransform({
            "schema": {"fields": "dynamic"},
            "multiplier": 2.0,
        })

    @pytest.fixture
    def valid_input(self):
        return {"id": 1, "value": 10.0}

    # 15+ contract tests are inherited automatically!
```

### Running Tests

```bash
# All contract tests
.venv/bin/python -m pytest tests/contracts/ -v

# Specific plugin
.venv/bin/python -m pytest tests/contracts/transform_contracts/test_my_transform_contract.py -v
```

### Test Base Classes

| Base Class | Location | Inherited Tests |
|------------|----------|-----------------|
| `SourceContractPropertyTestBase` | `tests/contracts/source_contracts/` | 14 tests |
| `TransformContractPropertyTestBase` | `tests/contracts/transform_contracts/` | 15 tests |
| `SinkDeterminismContractTestBase` | `tests/contracts/sink_contracts/` | 17 tests |

<details>
<summary><strong>Contract Tests Reference</strong></summary>

### Source Contracts

| Contract | Test |
|----------|------|
| Has `name` attribute | `test_source_has_name` |
| Has `output_schema` attribute | `test_source_has_output_schema` |
| `load()` returns iterator | `test_load_returns_iterator` |
| `load()` yields `SourceRow` only | `test_load_yields_source_rows` |
| `close()` is idempotent | `test_close_is_idempotent` |

### Transform Contracts

| Contract | Test |
|----------|------|
| Has `name` attribute | `test_transform_has_name` |
| Has `input_schema` attribute | `test_transform_has_input_schema` |
| Has `output_schema` attribute | `test_transform_has_output_schema` |
| `process()` returns `TransformResult` | `test_process_returns_transform_result` |
| `close()` is idempotent | `test_close_is_idempotent` |

### Sink Contracts

| Contract | Test |
|----------|------|
| Has `name` attribute | `test_sink_has_name` |
| Has `input_schema` attribute | `test_sink_has_input_schema` |
| `write()` returns `ArtifactDescriptor` | `test_write_returns_artifact_descriptor` |
| `content_hash` is valid SHA-256 | `test_content_hash_is_sha256_hex` |
| Same data → same hash | `test_same_data_same_hash` |

</details>

---

## Troubleshooting

### "Plugin not found"

```
KeyError: 'my_transform'
```

**Fix:** Register in both `hookimpl.py` AND `cli.py`.

### "schema is required"

```
ValidationError: schema_config is required
```

**Fix:** Extend `TransformDataConfig`, not `PluginConfig`:

```python
# Wrong
class MyConfig(PluginConfig): ...

# Right
class MyConfig(TransformDataConfig): ...
```

### "has no attribute 'name'"

**Fix:** Make `name` a class attribute, not instance attribute:

```python
class MyTransform(BaseTransform):
    name = "my_transform"  # Class attribute (correct)

    def __init__(self, config):
        self.name = "my_transform"  # Instance attribute (wrong)
```

### Validation failures in transform

**Fix:** Check `allow_coercion` setting:

```python
# Source: allow_coercion=True (external boundary)
# Transform/Sink: allow_coercion=False (trust upstream)
```

---

## Checklist for New Plugins

- [ ] Has `name` class attribute
- [ ] Has required schema attributes (`input_schema`, `output_schema`)
- [ ] Config extends correct base class (`TransformDataConfig`, `SourceDataConfig`)
- [ ] Schema created with correct `allow_coercion`
- [ ] Registered in `hookimpl.py`
- [ ] Registered in `cli.py`
- [ ] Contract tests pass
- [ ] `close()` is idempotent

---

## See Also

- [CLAUDE.md](CLAUDE.md) - Three-Tier Trust Model and data handling philosophy
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and plugin integration points
- [Configuration Reference](docs/reference/configuration.md) - Full configuration options
