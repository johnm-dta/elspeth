# ELSPETH Plugin Development Guide

This document describes how to create plugins for ELSPETH, from initial setup through registration and testing.

> **Related Documentation:**
> - `docs/contracts/plugin-protocol.md` - Authoritative protocol specification
> - `CLAUDE.md` - Project overview and Three-Tier Trust Model

## Prerequisites

Before creating plugins, you should be familiar with:

- **Python 3.11+** - Type hints, dataclasses, context managers
- **Pydantic v2** - Model validation, `Field()`, `model_validate()` ([docs](https://docs.pydantic.dev/))
- **pytest** - Fixtures, parametrization ([docs](https://docs.pytest.org/))
- **ELSPETH concepts** - Read `CLAUDE.md` for the Three-Tier Trust Model

**Development environment:**

```bash
# Clone and setup
git clone https://github.com/your-org/elspeth-rapid.git
cd elspeth-rapid

# Create venv with uv (required)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Verify setup
uv run elspeth plugins list
```

## Table of Contents

1. [Plugin Types Overview](#plugin-types-overview)
2. [Creating a Transform Plugin](#creating-a-transform-plugin)
3. [Creating a Source Plugin](#creating-a-source-plugin)
4. [Creating a Sink Plugin](#creating-a-sink-plugin)
5. [Creating a Gate Plugin](#creating-a-gate-plugin)
6. [Plugin Registration](#plugin-registration)
7. [Schema Configuration](#schema-configuration)
8. [Contract Testing](#contract-testing)

---

## Plugin Types Overview

ELSPETH follows the Sense/Decide/Act (SDA) model:

```
SOURCE (Sense) → TRANSFORM/GATE (Decide) → SINK (Act)
```

| Plugin Type | Purpose | Base Class | Key Methods |
|-------------|---------|------------|-------------|
| **Source** | Load data from external systems | `BaseSource` | `load()`, `close()` |
| **Transform** | Process/classify rows | `BaseTransform` | `process()`, `close()` |
| **Gate** | Route rows to destinations | `BaseGate` | `evaluate()`, `close()` |
| **Sink** | Output data to destinations | `BaseSink` | `write()`, `flush()`, `close()` |

### The Trust Model

All plugins are **system-owned code**, not user-provided extensions. This means:

| Plugin Type | Coercion Allowed? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary |
| **Transform** | ❌ No | Wrong types = upstream bug → crash |
| **Sink** | ❌ No | Wrong types = upstream bug → crash |

---

## Creating a Transform Plugin

Transforms are the most common plugin type. Here's the complete process.

### Step 1: Create the Configuration Class

Configuration classes validate plugin settings using Pydantic:

```python
# src/elspeth/plugins/transforms/my_transform.py
"""My custom transform plugin."""

from typing import Any

from pydantic import Field

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class MyTransformConfig(TransformDataConfig):
    """Configuration for my transform.

    Inherits from TransformDataConfig which:
    - Requires 'schema' configuration
    - Provides optional 'on_error' for error routing
    """

    # Add your custom config fields here
    multiplier: float = Field(
        default=1.0,
        description="Factor to multiply values by",
    )
    target_field: str = Field(
        default="value",
        description="Name of the field to transform",
    )
```

### Step 2: Create the Transform Class

```python
class MyTransform(BaseTransform):
    """Multiply a field value by a configured factor.

    Config options:
        schema: Required. Schema for input/output validation
        multiplier: Factor to multiply by (default: 1.0)
        target_field: Field name to transform (default: "value")
        on_error: Sink for rows that fail processing (optional)

    Example YAML:
        row_plugins:
          - plugin: my_transform
            options:
              schema:
                fields: dynamic
              multiplier: 2.5
              target_field: amount
    """

    # Required: unique plugin identifier
    name = "my_transform"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Parse and validate configuration
        cfg = MyTransformConfig.from_dict(config)
        self._multiplier = cfg.multiplier
        self._target_field = cfg.target_field
        self._on_error = cfg.on_error  # Required for error routing

        # TransformDataConfig guarantees schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Create schema from config
        # CRITICAL: allow_coercion=False - wrong types are upstream bugs
        schema = create_schema_from_config(
            self._schema_config,
            "MyTransformSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> TransformResult:
        """Process a single row.

        Args:
            row: Input row data (types already validated by source)
            ctx: Plugin context with run_id, config, landscape access

        Returns:
            TransformResult.success(row) or TransformResult.error(reason)
        """
        # Check if target field exists
        if self._target_field not in row:
            # This is a processing error, not a bug
            return TransformResult.error({
                "reason": "missing_field",
                "field": self._target_field,
            })

        # Wrap operations on THEIR DATA values
        try:
            original = row[self._target_field]
            result = float(original) * self._multiplier
        except (TypeError, ValueError) as e:
            # Their data caused the error - return error result
            return TransformResult.error({
                "reason": "conversion_failed",
                "field": self._target_field,
                "original_value": str(row[self._target_field]),
                "error": str(e),
            })

        # Return transformed row
        output = dict(row)
        output[self._target_field] = result
        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources (called after all rows processed)."""
        pass  # Nothing to release for this transform
```

### Transform Variants

#### Batch-Aware Transform

For transforms that process multiple rows together (aggregation):

```python
class BatchStatsTransform(BaseTransform):
    name = "batch_stats"
    is_batch_aware = True  # CRITICAL: receive list[dict] at aggregation nodes

    def process(  # type: ignore[override]
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> TransformResult:
        """Process a batch of rows.

        Args:
            rows: List of input rows (batch-aware receives list)
            ctx: Plugin context
        """
        if not rows:
            return TransformResult.success({"count": 0, "sum": 0})

        total = sum(r.get("value", 0) for r in rows)
        return TransformResult.success({
            "count": len(rows),
            "sum": total,
            "mean": total / len(rows),
        })
```

**Usage in pipeline:**
```yaml
aggregations:
  - name: compute_stats
    plugin: batch_stats
    trigger:
      count: 100  # Process every 100 rows
    output_mode: single  # N inputs → 1 output
    options:
      schema:
        fields: dynamic
```

#### Deaggregation Transform

For transforms that expand one row into multiple rows:

```python
class ExpandItemsTransform(BaseTransform):
    name = "expand_items"
    creates_tokens = True  # CRITICAL: engine creates new tokens for outputs

    def process(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> TransformResult:
        """Expand array field into multiple rows."""
        items = row["items"]  # Trust: source validated this is a list

        output_rows = [
            {**row, "item": item, "item_index": i}
            for i, item in enumerate(items)
        ]

        # Return multiple rows - engine creates new tokens for each
        return TransformResult.success_multi(output_rows)
```

**Token creation semantics:**
- `creates_tokens=True` + `success_multi()` → New tokens created per output row
- `creates_tokens=False` + `success_multi()` → RuntimeError (except in aggregation passthrough)

---

## Creating a Source Plugin

Sources load data from external systems and are the **only** place where type coercion is allowed.

### Step 1: Create the Configuration Class

```python
# src/elspeth/plugins/sources/my_source.py
"""My custom source plugin."""

from collections.abc import Iterator
from typing import Any

from pydantic import Field

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.plugins.base import BaseSource
from elspeth.plugins.config_base import SourceDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class MySourceConfig(SourceDataConfig):
    """Configuration for my source.

    Inherits from SourceDataConfig which requires:
    - path: File path (from PathConfig)
    - schema: Schema configuration (from DataPluginConfig)
    - on_validation_failure: Where invalid rows go (REQUIRED)
    """

    # Add custom config fields
    skip_header: bool = Field(
        default=True,
        description="Whether to skip the first line",
    )
```

### Step 2: Create the Source Class

```python
class MySource(BaseSource):
    """Load data from my custom format.

    Config options:
        path: Path to data file (required)
        schema: Schema configuration (required)
        on_validation_failure: Sink name or 'discard' (required)
        skip_header: Whether to skip header line (default: True)
    """

    name = "my_source"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = MySourceConfig.from_dict(config)
        self._path = cfg.resolved_path()
        self._skip_header = cfg.skip_header
        self._on_validation_failure = cfg.on_validation_failure

        # Store schema config for audit trail
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # CRITICAL: allow_coercion=True for sources (external data boundary)
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config,
            "MySourceRowSchema",
            allow_coercion=True,  # Sources MAY coerce external data
        )

        # Set output_schema for protocol compliance
        self.output_schema = self._schema_class

    def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
        """Load rows from the data file.

        Yields:
            SourceRow.valid() for rows that pass validation
            SourceRow.quarantined() for rows that fail (unless 'discard')
        """
        from pydantic import ValidationError

        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

        with open(self._path) as f:
            lines = f.readlines()

        if self._skip_header and lines:
            lines = lines[1:]

        for line_num, line in enumerate(lines, start=1):
            # Parse line into row dict (your format-specific logic)
            row = self._parse_line(line)

            try:
                # Validate and coerce row data
                validated = self._schema_class.model_validate(row)
                yield SourceRow.valid(validated.to_row())

            except ValidationError as e:
                # Record validation failure in audit trail
                ctx.record_validation_error(
                    row=row,
                    error=str(e),
                    schema_mode=self._schema_config.mode or "dynamic",
                    destination=self._on_validation_failure,
                )

                # Route to configured sink (or drop if 'discard')
                if self._on_validation_failure != "discard":
                    yield SourceRow.quarantined(
                        row=row,
                        error=str(e),
                        destination=self._on_validation_failure,
                    )

    def _parse_line(self, line: str) -> dict[str, Any]:
        """Parse a line into a row dict."""
        # Your format-specific parsing logic here
        parts = line.strip().split(",")
        return {"id": parts[0], "value": parts[1]} if len(parts) >= 2 else {}

    def close(self) -> None:
        """Release resources."""
        pass
```

---

## Creating a Sink Plugin

Sinks output data and **must** return audit-relevant information including content hashes.

```python
# src/elspeth/plugins/sinks/my_sink.py
"""My custom sink plugin."""

import hashlib
from pathlib import Path
from typing import Any

from pydantic import Field

from elspeth.contracts import ArtifactDescriptor
from elspeth.plugins.base import BaseSink
from elspeth.plugins.config_base import PathConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config


class MySinkConfig(PathConfig):
    """Configuration for my sink."""

    append: bool = Field(
        default=False,
        description="Append to existing file instead of overwrite",
    )


class MySink(BaseSink):
    """Write data to my custom format.

    Config options:
        path: Output file path (required)
        schema: Schema configuration (required)
        append: Append mode (default: False)
    """

    name = "my_sink"
    idempotent = False  # Appends are not idempotent

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = MySinkConfig.from_dict(config)
        self._path = Path(cfg.path)
        self._append = cfg.append

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "MySinkSchema",
            allow_coercion=False,  # Sinks do NOT coerce
        )
        self.input_schema = schema

        self._file = None
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def on_start(self, ctx: PluginContext) -> None:
        """Open output file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self._append else "w"
        self._file = open(self._path, mode)
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def write(
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> ArtifactDescriptor:
        """Write rows to file.

        Returns:
            ArtifactDescriptor with content_hash (REQUIRED for audit)
        """
        for row in rows:
            line = self._format_row(row)
            self._file.write(line)
            self._hasher.update(line.encode())
            self._bytes_written += len(line.encode())

        return ArtifactDescriptor.for_file(
            path=str(self._path),
            content_hash=self._hasher.hexdigest(),
            size_bytes=self._bytes_written,
        )

    def _format_row(self, row: dict[str, Any]) -> str:
        """Format row for output."""
        return ",".join(str(v) for v in row.values()) + "\n"

    def flush(self) -> None:
        """Flush buffered data."""
        if self._file:
            self._file.flush()

    def close(self) -> None:
        """Close file handle."""
        if self._file:
            self._file.close()
            self._file = None
```

---

## Creating a Gate Plugin

Gates route rows based on conditions. ELSPETH supports two approaches:

| Approach | Use When | Implementation |
|----------|----------|----------------|
| **Config Expression** | Simple field comparisons (`score > 0.8`) | YAML `condition` string |
| **Plugin Gate** | Complex logic (ML models, external lookups, stateful routing) | `BaseGate` subclass |

### Config Expression Gates (Simple Cases)

For simple routing, use config-driven gates:

```yaml
gates:
  - name: quality_check
    condition: "row['score'] >= 0.8"
    routes:
      "true": high_quality_sink
      "false": continue
```

### Plugin Gates (Complex Cases)

For complex routing logic, create a `BaseGate` subclass:

```python
# src/elspeth/plugins/transforms/my_gate.py
"""Custom gate plugin for complex routing decisions."""

from typing import Any

from elspeth.contracts import Determinism, PluginSchema, RoutingAction
from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult
from elspeth.plugins.schema_factory import create_schema_from_config


class MyGateConfig:
    """Configuration for my gate."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.threshold = config.get("threshold", 0.7)
        self.model_path = config.get("model_path")


class MyGate(BaseGate):
    """Route rows based on ML model predictions.

    Use this gate when:
    - Routing logic requires ML inference
    - Multiple fields must be analyzed together
    - External service calls are needed
    - Stateful decisions (rate limiting, quotas)

    Config options:
        threshold: Prediction threshold (default: 0.7)
        model_path: Path to ML model file

    Example YAML:
        gates:
          - name: ml_router
            plugin: my_gate
            threshold: 0.8
            model_path: /models/classifier.pkl
            routes:
              flagged: review_sink
    """

    name = "my_gate"
    determinism = Determinism.EXTERNAL_CALL  # ML model is external
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = MyGateConfig(config)
        self._threshold = cfg.threshold
        self._model_path = cfg.model_path
        self._model = None  # Lazy load

        # Gates typically pass through the same schema
        schema_config = config.get("schema", {"fields": "dynamic"})
        schema = create_schema_from_config(
            schema_config,
            "MyGateSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def on_start(self, ctx: PluginContext) -> None:
        """Load ML model at pipeline start."""
        if self._model_path:
            import pickle
            with open(self._model_path, "rb") as f:
                self._model = pickle.load(f)

    def evaluate(
        self, row: dict[str, Any], ctx: PluginContext
    ) -> GateResult:
        """Evaluate row and decide routing.

        Args:
            row: Input row (types validated by upstream)
            ctx: Plugin context

        Returns:
            GateResult with routing decision
        """
        # Complex logic that can't be a config expression
        if self._model:
            features = self._extract_features(row)
            score = self._model.predict_proba([features])[0][1]

            if score > self._threshold:
                # Optionally add metadata before routing
                row["ml_score"] = score
                return GateResult(
                    row=row,
                    action=RoutingAction.route("flagged"),
                )

        # Default: continue to next node
        return GateResult(row=row, action=RoutingAction.continue_())

    def _extract_features(self, row: dict[str, Any]) -> list[float]:
        """Extract ML features from row."""
        # Your feature extraction logic
        return [row.get("feature1", 0), row.get("feature2", 0)]

    def close(self) -> None:
        """Release resources."""
        self._model = None
```

### GateResult Options

```python
from elspeth.contracts import RoutingAction
from elspeth.plugins.results import GateResult

# Continue to next node in pipeline
GateResult(row=row, action=RoutingAction.continue_())

# Route to named sink
GateResult(row=row, action=RoutingAction.route("sink_name"))

# Fork to multiple parallel paths
GateResult(row=row, action=RoutingAction.fork(["path_a", "path_b"]))
```

### When to Use Each Approach

| Scenario | Recommended |
|----------|-------------|
| `row['score'] > threshold` | Config expression |
| Multiple conditions with AND/OR | Config expression |
| ML model inference | Plugin gate |
| External API validation | Plugin gate |
| Rate limiting / quotas | Plugin gate |
| Complex business rules | Plugin gate |

### Registering Gate Plugins

Gate plugins follow the same registration pattern as transforms:

1. Add to `hookimpl.py` (see [Plugin Registration](#plugin-registration))
2. Add to CLI registries if needed

> **Note:** Built-in gate plugins are rare since most routing is simple enough for config expressions. Create a plugin gate only when expressions can't express your logic.

---

## Plugin Registration

Plugins must be registered in two places:

### 1. Hook Implementation (pluggy)

Add your plugin to the appropriate `hookimpl.py`:

```python
# src/elspeth/plugins/transforms/hookimpl.py

class ElspethBuiltinTransforms:
    """Hook implementer for built-in transform plugins."""

    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        """Return built-in transform plugin classes."""
        from elspeth.plugins.transforms.my_transform import MyTransform
        # ... other imports ...

        return [PassThrough, FieldMapper, MyTransform]  # Add yours here
```

### 2. CLI Registration

Add your plugin to the CLI registries in `src/elspeth/cli.py`:

```python
# In _execute_pipeline() and _resume_run() functions:

from elspeth.plugins.transforms.my_transform import MyTransform

TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
    "passthrough": PassThrough,
    "field_mapper": FieldMapper,
    "my_transform": MyTransform,  # Add yours here
}
```

### File Organization

Place your plugin in the appropriate directory:

```
src/elspeth/plugins/
├── sources/
│   ├── __init__.py
│   ├── hookimpl.py          # Register sources here
│   ├── csv_source.py
│   └── my_source.py         # Your new source
├── transforms/
│   ├── __init__.py
│   ├── hookimpl.py          # Register transforms here
│   ├── passthrough.py
│   └── my_transform.py      # Your new transform
└── sinks/
    ├── __init__.py
    ├── hookimpl.py          # Register sinks here
    ├── csv_sink.py
    └── my_sink.py           # Your new sink
```

---

## Schema Configuration

Schemas control how data is validated. All data-processing plugins require schema configuration.

### Schema Modes

| Mode | Behavior | Extra Fields |
|------|----------|--------------|
| `dynamic` | Accept any fields | Allowed |
| `strict` | Only declared fields allowed | Rejected |
| `free` | Declared fields required, extras allowed | Allowed |

### YAML Configuration

```yaml
# Dynamic - accept anything
schema:
  fields: dynamic

# Strict - only these fields
schema:
  mode: strict
  fields:
    - "id: int"
    - "name: str"
    - "active: bool"

# Free - at least these, allow more
schema:
  mode: free
  fields:
    - "id: int"
    - "value: float"
```

### Supported Types

| Type | Python Type | Notes |
|------|-------------|-------|
| `str` | `str` | Text |
| `int` | `int` | Integer |
| `float` | `float` | Decimal |
| `bool` | `bool` | True/False |
| `any` | `Any` | No type checking |

### Schema Factory

Use `create_schema_from_config()` to create runtime schemas:

```python
from elspeth.plugins.schema_factory import create_schema_from_config

# For sources - coercion allowed
schema = create_schema_from_config(
    schema_config,
    "MySourceSchema",
    allow_coercion=True,
)

# For transforms/sinks - no coercion
schema = create_schema_from_config(
    schema_config,
    "MyTransformSchema",
    allow_coercion=False,
)
```

---

## Contract Testing

Every plugin MUST pass its protocol contract tests. The contract test framework verifies interface guarantees automatically.

### Quick Start: Testing a New Plugin

```python
# tests/contracts/transform_contracts/test_my_transform_contract.py
from typing import TYPE_CHECKING

import pytest

from elspeth.plugins.transforms.my_transform import MyTransform

from .test_transform_protocol import TransformContractPropertyTestBase

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol


class TestMyTransformContract(TransformContractPropertyTestBase):
    """Contract tests for MyTransform plugin."""

    @pytest.fixture
    def transform(self) -> "TransformProtocol":
        """REQUIRED: Return a configured transform instance."""
        return MyTransform({
            "schema": {"fields": "dynamic"},
            "multiplier": 2.0,
            "target_field": "value",
        })

    @pytest.fixture
    def valid_input(self) -> dict:
        """REQUIRED: Return input that should process successfully."""
        return {"id": 1, "value": 10.0}

    # All 15+ protocol contract tests are inherited automatically!
```

### Contract Test Base Classes

| Base Class | Location | Inherited Tests |
|------------|----------|-----------------|
| `SourceContractTestBase` | `tests/contracts/source_contracts/test_source_protocol.py` | 14 tests |
| `SourceContractPropertyTestBase` | Same file | 14 + property tests |
| `TransformContractTestBase` | `tests/contracts/transform_contracts/test_transform_protocol.py` | 15 tests |
| `TransformContractPropertyTestBase` | Same file | 15 + property tests |
| `SinkContractTestBase` | `tests/contracts/sink_contracts/test_sink_protocol.py` | 17 tests |
| `SinkDeterminismContractTestBase` | Same file | 17 + determinism tests |

### Required Fixtures by Plugin Type

#### Source Plugins

```python
class TestMySourceContract(SourceContractPropertyTestBase):

    @pytest.fixture
    def source(self, tmp_path: Path) -> SourceProtocol:
        """REQUIRED: Return a configured source instance."""
        # Create test input file
        input_file = tmp_path / "input.csv"
        input_file.write_text("id,value\n1,100\n2,200\n")

        return MySource({
            "path": str(input_file),
            "schema": {"fields": "dynamic"},
            "on_validation_failure": "discard",
        })
```

#### Transform Plugins

```python
class TestMyTransformContract(TransformContractPropertyTestBase):

    @pytest.fixture
    def transform(self) -> TransformProtocol:
        """REQUIRED: Return a configured transform instance."""
        return MyTransform({
            "schema": {"fields": "dynamic"},
            "multiplier": 2.0,
            "target_field": "value",
        })

    @pytest.fixture
    def valid_input(self) -> dict:
        """REQUIRED: Return input that should process successfully."""
        return {"id": 1, "value": 10.0}
```

#### Sink Plugins

```python
class TestMySinkContract(SinkDeterminismContractTestBase):

    @pytest.fixture
    def sink(self, tmp_path: Path) -> SinkProtocol:
        """REQUIRED: Return a configured sink instance."""
        return MySink({
            "path": str(tmp_path / "output.csv"),
            "schema": {"fields": "dynamic"},
        })

    @pytest.fixture
    def sample_rows(self) -> list[dict]:
        """REQUIRED: Return sample rows to write."""
        return [{"id": 1}, {"id": 2}]
```

### Running Contract Tests

```bash
# Run all contract tests
.venv/bin/python -m pytest tests/contracts/ -v

# Run tests for a specific plugin
.venv/bin/python -m pytest tests/contracts/transform_contracts/test_my_transform_contract.py -v

# Run with Hypothesis nightly profile (more examples)
HYPOTHESIS_PROFILE=nightly .venv/bin/python -m pytest tests/contracts/ -v
```

---

## Protocol Contracts Reference

### Source Protocol Contracts

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute (non-empty string) | `test_source_has_name` |
| Have `output_schema` attribute (class type) | `test_source_has_output_schema` |
| Have `determinism` attribute (Determinism enum) | `test_source_has_determinism` |
| Have `plugin_version` attribute (string) | `test_source_has_plugin_version` |
| `load()` returns an iterator | `test_load_returns_iterator` |
| `load()` yields `SourceRow` objects only | `test_load_yields_source_rows` |
| Valid `SourceRow` has non-None `.row` dict | `test_valid_rows_have_data` |
| Quarantined `SourceRow` has `.quarantine_error` | `test_quarantined_rows_have_error` |
| `close()` is idempotent | `test_close_is_idempotent` |

### Transform Protocol Contracts

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute | `test_transform_has_name` |
| Have `input_schema` attribute | `test_transform_has_input_schema` |
| Have `output_schema` attribute | `test_transform_has_output_schema` |
| Have `is_batch_aware` attribute (bool) | `test_transform_has_batch_awareness_flag` |
| Have `creates_tokens` attribute (bool) | `test_transform_has_creates_tokens_flag` |
| `process()` returns `TransformResult` | `test_process_returns_transform_result` |
| Success result has output data | `test_success_result_has_output_data` |
| `close()` is idempotent | `test_close_is_idempotent` |

### Sink Protocol Contracts

| Contract | Verified By |
|----------|-------------|
| Have `name` attribute | `test_sink_has_name` |
| Have `input_schema` attribute | `test_sink_has_input_schema` |
| Have `idempotent` attribute (bool) | `test_sink_has_idempotent_flag` |
| `write()` returns `ArtifactDescriptor` | `test_write_returns_artifact_descriptor` |
| `ArtifactDescriptor.content_hash` is valid SHA-256 | `test_content_hash_is_sha256_hex` |
| `ArtifactDescriptor.size_bytes` is not None | `test_artifact_has_size_bytes` |
| Same data produces same `content_hash` | `test_same_data_same_hash` |
| `flush()` is idempotent | `test_flush_is_idempotent` |
| `close()` is idempotent | `test_close_is_idempotent` |

---

## Checklist for New Plugins

Before submitting a new plugin:

- [ ] Plugin has all required protocol attributes (`name`, `*_schema`, `determinism`, `plugin_version`)
- [ ] Configuration class extends appropriate base (`TransformDataConfig`, `SourceDataConfig`, `PathConfig`)
- [ ] Schema created with correct `allow_coercion` setting (True for sources, False otherwise)
- [ ] Registered in `hookimpl.py` for the plugin type
- [ ] Registered in `cli.py` plugin registry
- [ ] Contract test class created inheriting appropriate base class
- [ ] All inherited contract tests pass
- [ ] Plugin-specific behavior has additional tests
- [ ] Sources yield `SourceRow.valid()` or `SourceRow.quarantined()`
- [ ] Transforms return `TransformResult.success()` or `TransformResult.error()`
- [ ] Sinks return `ArtifactDescriptor` with valid `content_hash`
- [ ] `close()` method is idempotent
- [ ] No type coercion in transforms/sinks

---

## Troubleshooting Plugin Development

### Common Errors

#### "Plugin not found" when running pipeline

```
KeyError: 'my_transform'
```

**Cause:** Plugin not registered in both required locations.
**Fix:** Ensure plugin is added to both `hookimpl.py` AND `cli.py`:

```python
# 1. src/elspeth/plugins/transforms/hookimpl.py
return [PassThrough, FieldMapper, MyTransform]  # Add here

# 2. src/elspeth/cli.py (in BOTH _execute_pipeline and _resume_run)
TRANSFORM_PLUGINS = {
    "my_transform": MyTransform,  # Add here
}
```

#### "schema is required" validation error

```
ValidationError: schema_config is required
```

**Cause:** Plugin config class missing schema requirement.
**Fix:** Ensure your config extends `TransformDataConfig` (not `PluginConfig`):

```python
# WRONG - PluginConfig doesn't require schema
class MyConfig(PluginConfig): ...

# RIGHT - TransformDataConfig requires schema
class MyConfig(TransformDataConfig): ...
```

#### Contract test fails with "has no attribute 'name'"

```
AttributeError: 'MyTransform' object has no attribute 'name'
```

**Cause:** Missing class attribute.
**Fix:** Add `name` as a class attribute, not instance attribute:

```python
class MyTransform(BaseTransform):
    name = "my_transform"  # Class attribute (correct)

    def __init__(self, config):
        self.name = "my_transform"  # Instance attribute (wrong)
```

#### "allow_coercion" causing validation failures

**Symptom:** Source works but transform fails on same data.
**Cause:** Transform using `allow_coercion=True` (should be False).
**Fix:** Only sources should coerce:

```python
# Source: allow_coercion=True (external data boundary)
# Transform/Sink: allow_coercion=False (trust upstream)
```

### Debugging Tips

1. **Run contract tests first:**
   ```bash
   .venv/bin/python -m pytest tests/contracts/transform_contracts/test_my_transform_contract.py -v
   ```

2. **Test plugin in isolation:**
   ```python
   plugin = MyTransform({"schema": {"fields": "dynamic"}})
   result = plugin.process({"id": 1, "value": 10}, mock_ctx)
   print(result)
   ```

3. **Check plugin attributes:**
   ```python
   print(f"name: {plugin.name}")
   print(f"input_schema: {plugin.input_schema}")
   print(f"is_batch_aware: {plugin.is_batch_aware}")
   ```

---

## Further Reading

- `docs/contracts/plugin-protocol.md` - Complete protocol specification
- `TEST_SYSTEM.md` - Complete test system documentation
- `CLAUDE.md` - Project overview and Three-Tier Trust Model
- `src/elspeth/plugins/protocols.py` - Protocol definitions
- `src/elspeth/contracts/results.py` - Result type definitions
