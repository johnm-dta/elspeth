# ELSPETH Plugin Development Guide

Create custom sources, transforms, and sinks for ELSPETH pipelines.

> **Quick Links:**
>
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
- [Plugin Registration](#plugin-registration)
- [Schema Configuration](#schema-configuration)
- [Contract Testing](#contract-testing)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python 3.12+** with type hints, dataclasses
- **Pydantic v2** for config validation
- **ELSPETH concepts** - Read [Data Trust and Error Handling](docs/guides/data-trust-and-error-handling.md) for the Three-Tier Trust Model

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

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class DoubleValueConfig(TransformDataConfig):
    """Config with custom field."""
    field: str = "value"


class DoubleValueTransform(BaseTransform):
    """Double a numeric field value."""

    name = "double_value"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = DoubleValueConfig

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = DoubleValueConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._field = cfg.field

        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "DoubleValue",
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        if self._field not in row:
            return TransformResult.error({"reason": "missing_field", "field": self._field})

        try:
            result = row[self._field] * 2
        except TypeError as e:
            return TransformResult.error({"reason": "invalid_input", "error": str(e)})

        output = row.to_dict()
        output[self._field] = result
        return TransformResult.success(
            PipelineRow(output, self._align_output_contract(row.contract)),
            success_reason={"action": "doubled_value", "fields_modified": [self._field]},
        )

    def close(self) -> None:
        pass
```

**Make it discoverable:**

Put the file under `src/elspeth/plugins/transforms/`. Built-in plugin discovery
scans that top-level directory automatically. If you add a new subdirectory,
add that subdirectory to `PLUGIN_SCAN_CONFIG` in
`src/elspeth/plugins/infrastructure/discovery.py`.

**Use it:**

```yaml
transforms:
- name: double_price
  plugin: double_value
  input: validated           # Explicit input connection
  on_success: doubled        # Named output connection
  options:
    schema:
      mode: observed
    field: price
```

**Test it:**

```python
# tests/unit/contracts/transform_contracts/test_double_value_contract.py
import pytest

from elspeth.plugins.transforms.double_value import DoubleValueTransform
from .test_transform_protocol import TransformContractPropertyTestBase

class TestDoubleValueContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self):
        return DoubleValueTransform({"schema": {"mode": "observed"}, "field": "value"})

    @pytest.fixture
    def valid_input(self):
        return {"id": 1, "value": 10.0}
```

---

## Plugin Types Overview

ELSPETH follows the **Sense/Decide/Act** model:

```
SOURCE (Sense) → TRANSFORM (Decide) → SINK (Act)
```

| Type | Purpose | Base Class | Key Method | Context |
|------|---------|------------|------------|---------|
| **Source** | Load data from external systems | `BaseSource` | `load()` | `SourceContext` |
| **Transform** | Process/classify rows | `BaseTransform` | `process()` | `TransformContext` |
| **Sink** | Output data | `BaseSink` | `write()` | `SinkContext` |

### The Trust Model: Who Can Coerce Data?

| Plugin Type | Coercion Allowed? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | External data boundary - normalize incoming data |
| **Transform** | ❌ No for pipeline row data; ✅ only for new external responses fetched by the transform | Pipeline data types are already validated; any HTTP/LLM/DB/file response is a fresh external boundary |
| **Sink** | ❌ No | Wrong types = upstream bug → crash |

**Rule:** Trust follows data flow, not plugin type. A transform that calls an
HTTP API, LLM, database, or file creates a new Tier 3 boundary inside the
transform: wrap the external call, validate/coerce the response immediately, and
then treat the validated result as pipeline data.

---

## Creating a Transform Plugin

Transforms process rows one at a time (or in batches for aggregation).

### Basic Transform

```python
from typing import Any

from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult


class MyTransformConfig(TransformDataConfig):
    """Config with your custom fields."""
    multiplier: int = 2
    target_field: str = "value"


class MyTransform(BaseTransform):
    """Multiply a field by a configured factor."""

    name = "my_transform"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = MyTransformConfig

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MyTransformConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._multiplier = cfg.multiplier
        self._target_field = cfg.target_field

        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)
        self.input_schema, self.output_schema = self._create_schemas(
            cfg.schema_config,
            "MyTransform",
        )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        if self._target_field not in row:
            return TransformResult.error({
                "reason": "missing_field",
                "field": self._target_field,
            })

        # Wrap operations on row values (their data can fail)
        try:
            result = row[self._target_field] * self._multiplier
        except TypeError as e:
            return TransformResult.error({
                "reason": "invalid_input",
                "error": str(e),
            })

        output = row.to_dict()
        output[self._target_field] = result
        return TransformResult.success(
            PipelineRow(output, self._align_output_contract(row.contract)),
            success_reason={
                "action": "multiplied_field",
                "fields_modified": [self._target_field],
            },
        )

    def close(self) -> None:
        pass
```

### Required Attributes

| Attribute | Type | Purpose |
|-----------|------|---------|
| `name` | `str` | Unique plugin identifier (class attribute) |
| `config_model` | `type[PluginConfig] \| None` | Pydantic config class rendered by `get_config_schema()` |
| `input_schema` | `type[PluginSchema]` | Expected input row schema |
| `output_schema` | `type[PluginSchema]` | Produced output row schema |
| `declared_input_fields` | `frozenset[str]` | Required input fields from `required_input_fields` |
| `declared_output_fields` | `frozenset[str]` | Fields added to every emitted row, used for collision checks |
| `_output_schema_config` | `SchemaConfig \| None` | Static output guarantee surface for DAG validation |
| `on_error` | `str \| None` | Sink name for error routing; injected by runtime settings |
| `on_success` | `str \| None` | Output connection name; injected by runtime settings |
| `determinism` | `Determinism` | Reproducibility level (default: `DETERMINISTIC`) |
| `plugin_version` | `str` | Plugin version for audit trail (default: `"0.0.0"`) |
| `source_file_hash` | `str \| None` | Entry-point file hash for audit identity; CI enforces concrete plugin values |

**Determinism levels:**

- `DETERMINISTIC` - Same input always produces same output
- `EXTERNAL_CALL` - Calls external service (LLM, API)
- `IO_READ` - Reads from external source
- `IO_WRITE` - Writes to external sink

### TransformResult Options

```python
# Success - transformed row (success_reason is REQUIRED)
TransformResult.success(
    PipelineRow(output_dict, output_contract),
    success_reason={"action": "classified"},
)

# Error - row failed processing (routes to on_error sink)
TransformResult.error({"reason": "division_by_zero"})

# Multiple outputs (requires creates_tokens=True)
TransformResult.success_multi(
    [PipelineRow(row1, output_contract), PipelineRow(row2, output_contract)],
    success_reason={"action": "expanded"},
)

# Intentional zero emission for filters
TransformResult.success_empty(success_reason={"action": "filtered"})
```

<details>
<summary><strong>Advanced: Batch-Aware Transforms</strong></summary>

For aggregation transforms that process multiple rows together, build an output
contract for the aggregate shape in `__init__` and return a `PipelineRow` with
that contract:

```python
from typing import Any

from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.schema_contract_factory import create_contract_from_config


class BatchStatsTransform(BaseTransform):
    name = "batch_stats"
    is_batch_aware = True  # Receives list[PipelineRow] instead of PipelineRow

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._output_schema_config = SchemaConfig(
            mode="fixed",
            fields=(
                FieldDefinition(name="count", field_type="int", required=True),
                FieldDefinition(name="sum", field_type="float", required=True),
                FieldDefinition(name="mean", field_type="float", required=True),
            ),
        )
        self._aggregate_output_contract = create_contract_from_config(self._output_schema_config)

    def process(self, rows: list[PipelineRow], ctx: TransformContext) -> TransformResult:
        if not rows:
            return TransformResult.error({"reason": "invalid_input", "error": "empty batch"})

        total = float(sum(r["value"] for r in rows))
        return TransformResult.success(
            PipelineRow(
                {"count": len(rows), "sum": total, "mean": total / len(rows)},
                self._aggregate_output_contract,
            ),
            success_reason={"action": "batch_stats_computed"},
        )
```

**Pipeline config:**

```yaml
aggregations:
- name: compute_stats
  plugin: batch_stats
  input: processed           # Explicit input connection
  on_success: stats_out      # Named output connection
  on_error: discard          # Sink name for batch errors, or 'discard'
  trigger:
    count: 100  # Process every 100 rows
  output_mode: single  # N inputs → 1 output
```

</details>

<details>
<summary><strong>Advanced: Deaggregation Transforms</strong></summary>

For transforms that expand one row into multiple rows:

```python
from elspeth.contracts.contract_propagation import propagate_contract


class ExpandItemsTransform(BaseTransform):
    name = "expand_items"
    creates_tokens = True  # Engine creates new tokens for each output
    declared_output_fields = frozenset({"item", "item_index"})

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        items = row["items"]  # Trust: source validated this is a list

        output_rows = []
        for i, item in enumerate(items):
            output = {**row.to_dict(), "item": item, "item_index": i}
            output_contract = propagate_contract(
                row.contract,
                output,
                transform_adds_fields=True,
            )
            output_contract = self._apply_declared_output_field_contracts(output_contract)
            output_contract = self._align_output_contract(output_contract)
            output_rows.append(PipelineRow(output, output_contract))

        return TransformResult.success_multi(output_rows, success_reason={"action": "expanded_items"})
```

For production deaggregation code, use `line_explode` as the reference pattern:
it declares output fields, builds `_output_schema_config`, propagates contracts,
and returns homogeneous `PipelineRow` outputs.

**Token semantics:**

- `creates_tokens=True` + `success_multi()` → New tokens per output
- `creates_tokens=False` + `success_multi()` → RuntimeError

</details>

---

## Creating a Source Plugin

Sources load data from external systems. **Sources can coerce input data at the
ingestion boundary.**

```python
from collections.abc import Iterator
from typing import Any
from pydantic import ValidationError

from elspeth.contracts import PluginSchema, SourceRow
from elspeth.contracts.contract_builder import ContractBuilder
from elspeth.contracts.schema_contract_factory import create_contract_from_config
from elspeth.plugins.infrastructure.base import BaseSource
from elspeth.plugins.infrastructure.config_base import SourceDataConfig
from elspeth.contracts.contexts import SourceContext
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


class MySourceConfig(SourceDataConfig):
    """Inherits path, schema, and on_validation_failure."""
    skip_header: bool = True


class MySource(BaseSource):
    """Load data from a custom format."""

    name = "my_source"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = MySourceConfig

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MySourceConfig.from_dict(config, plugin_name=self.name)
        self._path = cfg.resolved_path()
        self._skip_header = cfg.skip_header
        self._on_validation_failure = cfg.on_validation_failure

        self._schema_config = cfg.schema_config
        self._initialize_declared_guaranteed_fields(self._schema_config)

        # CRITICAL: allow_coercion=True for sources
        self._schema_class: type[PluginSchema] = create_schema_from_config(
            self._schema_config, "MySourceSchema", allow_coercion=True
        )
        self.output_schema = self._schema_class

        initial_contract = create_contract_from_config(self._schema_config)
        if initial_contract.locked:
            self.set_schema_contract(initial_contract)
            self._contract_builder: ContractBuilder | None = None
        else:
            self._contract_builder = ContractBuilder(initial_contract)
        self._first_valid_row_processed = False

    def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")
        self._first_valid_row_processed = False

        with open(self._path) as f:
            lines = f.readlines()

        if self._skip_header and lines:
            lines = lines[1:]

        for line in lines:
            row = self._parse_line(line)

            try:
                validated = self._schema_class.model_validate(row)
                validated_row = validated.to_row()

                if self._contract_builder is not None and not self._first_valid_row_processed:
                    field_resolution = {field: field for field in validated_row}
                    self._contract_builder.process_first_row(validated_row, field_resolution)
                    self.set_schema_contract(self._contract_builder.contract)
                    self._first_valid_row_processed = True

                contract = self.require_schema_contract()
                if contract.locked:
                    violations = contract.validate(validated_row)
                    if violations:
                        error_msg = "; ".join(str(v) for v in violations)
                        ctx.record_validation_error(
                            row=validated_row,
                            error=error_msg,
                            schema_mode=self._schema_config.mode,
                            destination=self._on_validation_failure,
                        )
                        if self._on_validation_failure != "discard":
                            yield SourceRow.quarantined(
                                row=validated_row,
                                error=error_msg,
                                destination=self._on_validation_failure,
                            )
                        continue

                yield SourceRow.valid(validated_row, contract=contract)

            except ValidationError as e:
                ctx.record_validation_error(
                    row=row,
                    error=str(e),
                    schema_mode=self._schema_config.mode or "observed",
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
SourceRow.valid({"id": 1, "value": 100}, contract=source_contract)

# Quarantined - route to on_validation_failure sink
SourceRow.quarantined(row=raw_row, error="Invalid type", destination="quarantine_sink")
```

Valid source rows must carry a `SchemaContract`. Create the contract from the
effective source schema, update it through `ContractBuilder` for observed or
flexible first-row inference, then pass it to `SourceRow.valid(..., contract=...)`.

---

## Creating a Sink Plugin

Sinks output data and **must return audit information** including content hashes.

```python
import hashlib
from pathlib import Path
from typing import Any

from elspeth.contracts import ArtifactDescriptor
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.infrastructure.config_base import PathConfig
from elspeth.contracts.contexts import LifecycleContext, SinkContext
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config


class MySinkConfig(PathConfig):
    append: bool = False


class MySink(BaseSink):
    """Write data to a custom format."""

    name = "my_sink"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0000000000000000"
    config_model = MySinkConfig
    idempotent = False  # Appends are not idempotent

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = MySinkConfig.from_dict(config, plugin_name=self.name)
        self._path = Path(cfg.path)
        self._append = cfg.append

        schema = create_schema_from_config(
            cfg.schema_config, "MySinkSchema", allow_coercion=False
        )
        self.input_schema = schema
        self.declared_required_fields = cfg.schema_config.get_effective_required_fields()

        self._file = None
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def on_start(self, ctx: LifecycleContext) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a" if self._append else "w")
        self._bytes_written = 0
        self._hasher = hashlib.sha256()

    def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> SinkWriteResult:
        for row in rows:
            line = ",".join(str(v) for v in row.values()) + "\n"
            self._file.write(line)
            self._hasher.update(line.encode())
            self._bytes_written += len(line.encode())

        # REQUIRED: Return artifact with content hash for audit
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=str(self._path),
                content_hash=self._hasher.hexdigest(),
                size_bytes=self._bytes_written,
            ),
            diversions=self._get_diversions(),
        )

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
```

### SinkWriteResult and ArtifactDescriptor Requirements

`write()` returns `SinkWriteResult`. The `artifact` inside it gives the audit
trail proof of work, and `diversions` carries any per-row write failures created
with `BaseSink._divert_row()`.

The artifact requires:

- `content_hash` - SHA-256 hex digest of output content
- `size_bytes` - Output size in bytes

---

## Plugin Registration

Built-in plugins are discovered dynamically by scanning configured plugin
directories. For a new top-level built-in plugin, put the file in one of these
directories and make the class inherit the correct base:

| Plugin type | Directory |
|-------------|-----------|
| Source | `src/elspeth/plugins/sources/` |
| Transform | `src/elspeth/plugins/transforms/` |
| Sink | `src/elspeth/plugins/sinks/` |

Discovery is non-recursive. If you add a new subdirectory, update
`PLUGIN_SCAN_CONFIG` in `src/elspeth/plugins/infrastructure/discovery.py`.
`PluginManager.register_builtin_plugins()` turns discovered classes into pluggy
hook implementations at startup, so there is no separate CLI registry to edit.

After adding a plugin, run discovery-focused tests. Some tests assert the exact
built-in plugin count, so a new plugin may require updating those expectations.

---

## Schema Configuration

All data-processing plugins require schema configuration.

### Schema Modes

| Mode | Behavior | Extra Fields |
|------|----------|--------------|
| `observed` | Accept any fields (types inferred from data) | Allowed |
| `fixed` | Only declared fields | Rejected |
| `flexible` | Declared required, extras allowed | Allowed |

### YAML Examples

```yaml
# Accept anything
schema:
  mode: observed

# Fixed - only these fields
schema:
  mode: fixed
  fields:
    - "id: int"
    - "name: str"
    - "active: bool"

# At least these, allow more
schema:
  mode: flexible
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
# tests/unit/contracts/transform_contracts/test_my_transform_contract.py
import pytest
from elspeth.plugins.transforms.my_transform import MyTransform
from .test_transform_protocol import TransformContractPropertyTestBase


class TestMyTransformContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self):
        return MyTransform({
            "schema": {"mode": "observed"},
            "multiplier": 2,
        })

    @pytest.fixture
    def valid_input(self):
        return {"id": 1, "value": 10.0}

    # 15+ contract tests are inherited automatically!
```

### Running Tests

```bash
# All contract tests
.venv/bin/python -m pytest tests/unit/contracts/ -v

# Specific plugin
.venv/bin/python -m pytest tests/unit/contracts/transform_contracts/test_my_transform_contract.py -v
```

### Test Base Classes

| Base Class | Location | Inherited Tests |
|------------|----------|-----------------|
| `SourceContractPropertyTestBase` | `tests/unit/contracts/source_contracts/` | 14 tests |
| `TransformContractPropertyTestBase` | `tests/unit/contracts/transform_contracts/` | 15 tests |
| `SinkDeterminismContractTestBase` | `tests/unit/contracts/sink_contracts/` | 17 tests |

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
| `write()` returns `SinkWriteResult` with an `ArtifactDescriptor` | `test_write_returns_artifact_descriptor` |
| `content_hash` is valid SHA-256 | `test_content_hash_is_sha256_hex` |
| Same data → same hash | `test_same_data_same_hash` |

</details>

---

## Troubleshooting

### "Plugin not found"

```
KeyError: 'my_transform'
```

**Fix:** Confirm the plugin file lives in a scanned directory and the class
inherits the correct base class. For plugins in new subdirectories, add the
subdirectory to `PLUGIN_SCAN_CONFIG` in
`src/elspeth/plugins/infrastructure/discovery.py`.

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
# Transform external response: validate/coerce immediately at that call boundary
```

---

## Checklist for New Plugins

- [ ] Has `name` class attribute
- [ ] Has `plugin_version`, `source_file_hash`, and `config_model` class attributes
- [ ] Has required schema attributes (`input_schema`, `output_schema`)
- [ ] Config extends the correct data config base (`TransformDataConfig`, `SourceDataConfig`, `PathConfig`, or a narrower sink/source config)
- [ ] Schema created with correct `allow_coercion`
- [ ] Source valid rows call `SourceRow.valid(row, contract=contract)`
- [ ] Transform successes return `PipelineRow` values, never raw dicts
- [ ] Sink `write()` returns `SinkWriteResult(artifact=ArtifactDescriptor, diversions=...)`
- [ ] Transform field declarations are set (`declared_input_fields`, `declared_output_fields`, `_output_schema_config`) when the plugin requires or adds fields
- [ ] Sink required fields are set with `declared_required_fields`
- [ ] Plugin file lives in a scanned discovery directory, or `PLUGIN_SCAN_CONFIG` was updated
- [ ] `scripts/cicd/enforce_plugin_hashes.py check --root src/elspeth --fix` has refreshed `source_file_hash`
- [ ] Contract tests pass
- [ ] `close()` is idempotent

---

## See Also

- [Data Trust and Error Handling](docs/guides/data-trust-and-error-handling.md) - External boundaries, quarantine, and plugin error handling
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and plugin integration points
- [Configuration Reference](docs/reference/configuration.md) - Full configuration options
