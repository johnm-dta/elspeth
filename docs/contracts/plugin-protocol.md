# ELSPETH Plugin Protocol Contract

> **Status:** FINAL (v1.12)
> **Last Updated:** 2026-05-07
> **Authority:** This document is the master reference for all plugin interactions.

## Overview

ELSPETH follows the Sense/Decide/Act (SDA) model:

```
SOURCE (Sense) → DECIDE → SINK (Act)
```

**Critical distinction:**

| Layer | What | Requires Code? |
|-------|------|----------------|
| **Plugins** | External system integration + business logic | Yes (validated system code) |
| **System Operations** | Token routing, batching, forking, merging | No (config-driven) |

### Plugins (This Document)

Plugins are **system code** developed by the same team that operates ELSPETH. They are independently validated and treated as trusted components. This is NOT a plugin marketplace—arbitrary external code is not accepted.

| Plugin | Purpose | Touches |
|--------|---------|---------|
| **Source** | Load data from external systems | Row contents |
| **Transform** | Apply business logic to rows | Row contents |
| **Sink** | Write data to external systems | Row contents |

### System Operations (NOT Plugins)

These are **config-driven** infrastructure provided by the ELSPETH engine:

| Operation | Purpose | Config Example |
|-----------|---------|----------------|
| **Gate** | Route tokens based on conditions | `condition: "row['score'] > 0.8"` |
| **Aggregation** | Batch tokens until trigger | `trigger: "count >= 100"` |
| **Fork** | Split token to parallel paths | Routing action |
| **Coalesce** | Merge tokens from parallel paths | `policy: require_all` |

System operations work on **wrapped data** (tokens, metadata) and require no custom code.
Plugins work on **row contents** (the actual data) and are validated system code.

---

## Core Principles

### 1. Audit Is Non-Negotiable

Every plugin interaction is recorded. The audit trail must answer:
- What data entered the plugin?
- What did the plugin produce?
- When did it happen?
- Did it succeed or fail?

Plugins MUST return audit-relevant information. "Trust me, I did it" is not acceptable.

### 2. Plugins Control Their Own Schedule

ELSPETH doesn't dictate internal timing. Plugins may:
- Block waiting for external resources (satellite links, APIs, databases)
- Queue work internally and process on their own schedule
- Batch operations for efficiency

The contract specifies WHEN methods are called, not HOW FAST plugins must respond.

### 3. Exception Handling: Three-Tier Trust Model

ELSPETH uses a three-tier trust model that determines how exceptions should be handled:

| Tier | Trust Level | Coercion | Exception Handling |
|------|-------------|----------|-------------------|
| **External Data** (Source input or transform-side external response) | Zero trust | ✅ Allowed at that external boundary | Validate, coerce, quarantine/return error failures |
| **Pipeline Data** (Post-source rows) | Elevated ("probably ok") | ❌ Forbidden | Types trusted, wrap VALUE operations |
| **Our Code** (Plugin internals) | Full trust | ❌ N/A | Let it crash - bugs must surface |

#### The Key Insight: Type-Safe ≠ Operation-Safe

Data that passed source validation has correct **types**, but **values** can still cause operation failures:

```python
# Pipeline data is type-valid (int), but operation fails
row = {"divisor": 0}  # Passed source validation ✓
result = 100 / row["divisor"]  # 💥 ZeroDivisionError - WRAP THIS

# Pipeline data is type-valid (str), but content is problematic
row = {"date": "not-a-date"}  # Passed as str ✓
parsed = datetime.fromisoformat(row["date"])  # 💥 ValueError - WRAP THIS
```

#### The Divide-By-Zero Test

```python
from elspeth.contracts.contract_propagation import propagate_contract
from elspeth.contracts.schema_contract import PipelineRow


# THEIR DATA value caused the error → WRAP AND HANDLE
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    try:
        result = row["value"] / row["divisor"]  # User's divisor=0
    except ZeroDivisionError:
        return TransformResult.error(
            {"reason": "division_by_zero", "field": "divisor"},
            retryable=False,
        )
    output = {**row.to_dict(), "result": result}
    output_contract = propagate_contract(row.contract, output, transform_adds_fields=True)
    return TransformResult.success(
        PipelineRow(output, output_contract),
        success_reason={"action": "calculated"},
    )

# OUR CODE caused the error → LET IT CRASH
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    # If _batch_count is 0, that's MY bug - I should have initialized it
    average = self._total / self._batch_count  # Let it crash!
    output = {**row.to_dict(), "average": average}
    output_contract = propagate_contract(row.contract, output, transform_adds_fields=True)
    return TransformResult.success(
        PipelineRow(output, output_contract),
        success_reason={"action": "averaged"},
    )
```

#### Coercion Rules by Plugin Type

| Plugin Type | May Coerce Types? | Why |
|-------------|-------------------|-----|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary (`"42"` → `42`) |
| **Transform** | ❌ No for pipeline rows; ✅ only for new external responses fetched by the transform | Pipeline data is validated; external API/LLM/DB/file responses are fresh Tier 3 boundaries |
| **Sink** | ❌ No | Receives validated data; wrong types = upstream bug |

If a transform receives `"42"` when its `input_schema` says `int`, that's a bug in the source or upstream transform. The correct fix is to fix the upstream plugin, NOT to coerce in the transform. If the same transform calls an external service and receives `"42"` in that response, that response is Tier 3 and must be validated/coerced immediately at the call boundary before being merged into the pipeline row.

#### The Boundary

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PLUGIN PROCESSING                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THEIR DATA VALUES (row field values, external responses)           │
│  ─────────────────────────────────────────────────────────          │
│  • Types are trusted (source validated them)                        │
│  • Values may still cause operation failures                        │
│  • Wrap OPERATIONS in try/catch                                     │
│  • Return error result on failure                                    │
│  • Row gets quarantined, pipeline continues                          │
│                                                                      │
│  Examples:                                                           │
│  • row["value"] / row["divisor"] → catch ZeroDivisionError          │
│  • datetime.fromisoformat(row["date"]) → catch ValueError           │
│  • external_api.call(row["id"]) → catch ApiError                    │
│  • json.loads(row["payload"]) → catch JSONDecodeError               │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  THEIR DATA TYPES (at any external boundary)                        │
│  ─────────────────────────────────────────────────────────          │
│  • External data may have wrong types                               │
│  • Sources MAY coerce source input: "42" → 42, "true" → True        │
│  • Transforms MAY coerce fresh external responses they fetch        │
│  • Sources MUST quarantine rows that can't be coerced               │
│  • Transforms/Sinks MUST NOT coerce already-validated pipeline rows │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  OUR CODE (internal state, plugin logic)                            │
│  ─────────────────────────────────────────                          │
│  • Do NOT wrap in try/catch                                          │
│  • Let exceptions propagate                                          │
│  • Pipeline crashes - this is a bug to fix                           │
│                                                                      │
│  Examples:                                                           │
│  • self._counter / self._batch_size → if batch_size=0, that's a bug │
│  • self._buffer[index] → if index wrong, that's a bug               │
│  • self._connection.execute() → if connection is None, that's a bug │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

#### Contract

| Zone | On Error | Plugin Action | ELSPETH Action |
|------|----------|---------------|----------------|
| **Their Data Values** | Row value causes operation error | Catch, return error result | Quarantine row, continue |
| **Their Data Types** | Wrong type at Source boundary | Coerce if possible, else error | Quarantine row, continue |
| **Their Data Types** | Wrong type at Transform/Sink | — | This is an upstream bug, should crash |
| **Our Code** | Internal state causes error | Let it crash | Record in audit, halt pipeline |
| **Lifecycle** | `on_start`/`on_complete`/`close` fails | Let it crash | Halt pipeline (config/code bug) |

**Rules:**
- Plugins MUST wrap operations on row values and return error results on failure
- Sources MAY coerce source input; transforms MAY coerce fresh external responses they fetch; transforms and sinks MUST NOT coerce already-validated pipeline rows
- Plugins MUST NOT wrap operations on internal state - let bugs surface
- If you're tempted to add `try/except` around your own logic, you have a bug to fix
- If a transform receives wrong types, that's an upstream bug to fix, not something to coerce

### 4. Forced Pass for Lifecycle Hooks

All lifecycle hooks are REQUIRED in the protocol, even if implementation is `pass`.

**Why:** The audit trail records every lifecycle event. Even an empty `on_start()` produces an audit record: "plugin started at timestamp X". This is defensible under audit.

### 5. Built-In Discovery Is Dynamic

Built-in plugins are discovered by scanning `PLUGIN_SCAN_CONFIG` in
`src/elspeth/plugins/infrastructure/discovery.py`. The scan is non-recursive:
top-level `sources/`, `transforms/`, and `sinks/` modules are discovered
automatically, while new subdirectories must be listed explicitly.
`PluginManager.register_builtin_plugins()` registers the discovered classes via
generated pluggy hook implementations. There is no separate CLI registry for
new built-in plugins.

---

## Plugin Types

### Source

**Purpose:** Load data into the pipeline. Exactly one source per run.

**Data Flow:** Produces rows on its own schedule.

#### Required Attributes

```python
name: str                          # Plugin identifier (e.g., "csv", "api", "satellite")
config_model: type[PluginConfig] | None
output_schema: type[PluginSchema]  # Schema of rows this source produces
node_id: str | None                # Set by orchestrator after registration
determinism: Determinism           # See Determinism Declaration section for all 6 levels
plugin_version: str                # Semantic version for reproducibility
source_file_hash: str | None       # Entry-point module hash for audit identity
declared_guaranteed_fields: frozenset[str]  # Runtime producer guarantee surface (ADR-016)
```

`declared_guaranteed_fields` is the runtime declaration surface used by the
source boundary contract. Sources derive it from their effective
`SchemaConfig` after any source-specific schema rewrite. For example, the text
source's observed-mode column heuristic must be reflected here, not only in the
raw config dict.

#### Required Configuration

```yaml
sources:
  my_source:
    type: csv_source
    path: data/input.csv
    schema:
      mode: fixed
      fields:
        name: {type: string}
        age: {type: integer}
    # REQUIRED: Where do non-conformant rows go?
    on_validation_failure: quarantine_sink  # Sink name, or "discard"
```

**`on_validation_failure`** (REQUIRED):
- Specifies destination for rows that fail schema validation/coercion
- Value must be a sink name or `"discard"` for explicit drop
- Cannot be omitted, empty, or null - operator must acknowledge bad data handling
- Even when `"discard"`, a `QuarantineEvent` is recorded in the audit trail

**Quarantine behavior:**
1. Source attempts to validate/coerce row against `output_schema`
2. If validation fails:
   - `QuarantineEvent` recorded (always, even for discard)
   - Row routed to configured sink OR dropped if `"discard"`
   - Row does NOT enter the pipeline
3. If validation succeeds:
   - Row enters pipeline as normal

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration.

    Called once at pipeline construction.
    Validate config here - fail fast if misconfigured.
    """

def load(self, ctx: PluginContext) -> Iterator[SourceRow]:
    """Yield rows from the source.

    MAY BLOCK internally waiting for data (satellite downlink, API polling).
    Yields rows on source's own schedule.
    Iterator exhausts when source has no more data.

    Returns:
        Iterator yielding SourceRow instances - either SourceRow.valid() for
        valid rows or SourceRow.quarantined() for invalid rows.

    Example:
        try:
            validated = self._schema_contract.validate(row)
            yield SourceRow.valid(validated, contract=self.require_schema_contract())
        except ValidationError as e:
            if self._on_validation_failure != "discard":
                yield SourceRow.quarantined(
                    row=row,
                    error=str(e),
                    destination=self._on_validation_failure,
                )
    """

def close(self) -> None:
    """Release resources.

    Called after on_complete() or on error.
    MUST NOT raise - log errors internally if cleanup fails.
    """
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before load().

    Use for: Opening connections, authenticating, preparing resources.
    Should be reasonably quick - heavy blocking belongs in load().

    If this fails, it's a CODE BUG - pipeline crashes.
    """

def on_complete(self, ctx: PluginContext) -> None:
    """Called after load() exhausts (before close).

    Use for: Finalization, recording completion metrics.

    If this fails, it's a CODE BUG - pipeline crashes.
    """
```

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)           ← Setup connections, auth
    │
    ▼
load(ctx) ─► yields rows on source's schedule
    │        (may block internally)
    │
    ▼ (iterator exhausts)
on_complete(ctx)        ← Finalization
    │
    ▼
close()                 ← Release resources
```

#### SourceRow Contract

```python
@dataclass
class SourceRow:
    row: Any  # dict for valid rows, Any for quarantined (may be malformed)
    is_quarantined: bool
    quarantine_error: str | None
    quarantine_destination: str | None
    contract: SchemaContract | None  # Required for valid rows; absent for quarantined rows

    @classmethod
    def valid(cls, row: dict[str, Any], *, contract: SchemaContract) -> SourceRow:
        """Create a valid source row.

        Args:
            row: Validated row data
            contract: Schema contract for the row

        Returns:
            SourceRow with is_quarantined=False
        """
        ...

    @classmethod
    def quarantined(cls, row: Any, error: str, destination: str) -> SourceRow:
        """Create a quarantined row result.

        Args:
            row: The original row data (may be non-dict for malformed data)
            error: The validation error message
            destination: The sink name to route this row to
        """
        ...

    def to_pipeline_row(self) -> PipelineRow:
        """Convert to PipelineRow for processing.

        Raises:
            ValueError: If row is quarantined or has no contract
        """
        ...
```

**Usage:**
- Valid rows: `SourceRow.valid(row_dict, contract=schema_contract)`
- Invalid rows: `SourceRow.quarantined(row_data, error_message, destination_sink)`
- The `contract` parameter is required for valid rows; without it the engine cannot create downstream `PipelineRow` values
- Quarantined rows never have contracts

#### Audit Records

- `on_start` called timestamp
- Each row yielded (row_id, content_hash)
- `on_complete` called timestamp
- Total rows produced
- **QuarantineEvent** for each validation failure:
  - `run_id`, `source_id`, `row_index`
  - `raw_row` (original data before coercion attempt)
  - `failure_reason` (why validation failed)
  - `field_errors` (per-field error details)
  - `destination` (sink name or "discard")
  - `timestamp`

#### Source row identity — no fabrication

**Constraint.** Every row a source emits must carry two engine-owned
identity fields supplied with their authentic values:

- `source_row_index`: the row's position within *this source's own
  emission stream* (0-indexed, monotone, no gaps the source did not
  itself introduce).
- `ingest_sequence`: a global per-run monotone ordering value the engine
  assigns at the boundary where the source's row enters the orchestrator.

The engine refuses to mint a token without both. `TokenManager` raises
`OrchestrationInvariantError` and the `DataFlowRepository.create_row`
write boundary raises `AuditIntegrityError`. Both error sites carry the
institutional-memory message:

> *Do not fabricate source_row_index or ingest_sequence from row_index.*

**Why.** `source_row_index` and `ingest_sequence` are Tier-1 audit
evidence (per [CLAUDE.md's three-tier trust model](../../CLAUDE.md)).
They are the durable identity primitives that resume, replay, and
cross-source ordering depend on. Substituting an arbitrary value (most
commonly: `row_index` — the orchestrator's positional count, which is
*not* the source's own row index during resume) gives the audit trail a
confident wrong answer. An auditor running `elspeth explain` on a row
with fabricated identity gets back a value the source never actually
asserted; the audit story breaks and the wrong answer is
indistinguishable from a correct one.

This is the fabrication-decision-test from CLAUDE.md applied at the
source boundary: if the source plugin does not know `source_row_index`
— because it doesn't natively track per-source row position — then the
correct value is **not knowable**, and the only honest action is to
crash. The engine deliberately raises rather than defaulting.

**What the audit trail looks like under correct identity.**

```
rows.source_node_id        = "customer_events"
rows.source_row_index      = 42          # 43rd row from THIS source
rows.ingest_sequence       = 1024        # 1025th row from ALL sources in this run
rows.row_index             = 42          # orchestrator's view (may differ on resume)
```

`elspeth explain --row <row_id>` reconstructs the lineage by joining
`rows.source_node_id` → `run_sources.schema_contract_json`. Resume
replays rows ordered by `ingest_sequence`, which is the only value that
remains stable across crash/restart.

**What the audit trail looks like under fabricated identity.** If a
source plugin computes `source_row_index = self._counter` where
`_counter` resets on retry, or — worse — sets `source_row_index =
row_index` because that field happened to be available, the audit DB
ends up with rows whose identity values reflect orchestrator state, not
source state. On resume the engine reads those rows back believing they
are authentic source emissions; the replay order is wrong, the
per-source `(run_id, source_node_id, source_row_index)` unique constraint
collides on second-attempt rows, and the cross-source `ingest_sequence`
ordering — which the durable scheduler uses as its claim-order primitive
(ADR-026 §Decision 3) — becomes nondeterministic. The pipeline appears
to work; the audit trail asserts facts the external system never said.

**What a source plugin must do.** Track per-source row position
explicitly inside the plugin (an instance counter, an offset field on
the upstream record, a database cursor position — whatever the source
genuinely knows). Pass that value as `source_row_index` on every row.
The orchestrator assigns `ingest_sequence` at the boundary, so plugins
do not compute it; they just must not invent it themselves. If the
source genuinely cannot provide `source_row_index` (a streaming API
with no notion of position, for instance), the source plugin is not
yet engine-compatible — file a design ticket, do not paper over the
gap in plugin code.

**Cross-references.**

- Engine refuse points: `src/elspeth/engine/tokens.py:71-92`
  (`_require_source_row_identity`) and
  `src/elspeth/core/landscape/data_flow_repository.py:441-446`.
- Schema invariants on the persisted columns:
  [Landscape — Per-row source identity](../architecture/landscape.md#per-row-source-identity-adr-025).
- Why these fields exist and what they mean for cross-source ordering:
  [ADR-025 §Decision 4](../architecture/adr/025-multi-source-ingestion.md)
  and [ADR-026 §Decision 3](../architecture/adr/026-durable-token-scheduler.md).
- A lint rule that detects `source_row_index = row_index` and similar
  fabricating patterns in Source plugin code is tracked under filigree
  `elspeth-92afea0d23`; until it lands, this contract is the
  authoritative discoverable statement of the rule.

---

### Transform

**Purpose:** Apply business logic to rows. Stateless between calls.

**Data Flow:** Row(s) in → row(s) out (possibly modified).

#### Required Attributes

```python
name: str
config_model: type[PluginConfig] | None
input_schema: type[PluginSchema]
output_schema: type[PluginSchema]
node_id: str | None
determinism: Determinism
plugin_version: str
source_file_hash: str | None
is_batch_aware: bool = False  # Set True for batch processing at aggregation nodes
creates_tokens: bool = False  # Set True for deaggregation (1→N row expansion)
passes_through_input: bool = False  # Set True only when every emitted row preserves input fields
can_drop_rows: bool = False  # Set True only when pass-through transform may intentionally emit zero rows
declared_input_fields: frozenset[str] = frozenset()  # Required input-field declaration (single-row only)
declared_output_fields: frozenset[str] = frozenset()  # Guaranteed per-emitted-row output fields
_output_schema_config: SchemaConfig | None = None  # Required when output fields are declared
```

#### Token Creation (Deaggregation)

Transforms that expand one row into multiple rows must declare `creates_tokens = True`:

```python
from elspeth.contracts.contract_propagation import propagate_contract


class JSONExplode(BaseTransform):
    name = "json_explode"
    creates_tokens = True  # Engine creates new tokens for each output row
    declared_output_fields = frozenset({"item", "item_index"})
    # ...

    def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
        items = row["items"]  # Trust: source validated this is a list
        output_rows = []
        for i, item in enumerate(items):
            output = {**row.to_dict(), "item": item, "item_index": i}
            output_contract = propagate_contract(row.contract, output, transform_adds_fields=True)
            output_contract = self._apply_declared_output_field_contracts(output_contract)
            output_contract = self._align_output_contract(output_contract)
            output_rows.append(PipelineRow(output, output_contract))
        return TransformResult.success_multi(
            output_rows,
            success_reason={"action": "exploded", "output_count": len(items)},
        )
```

**Invariants:**
- `creates_tokens=True` + `success()` → single output (allowed, like passthrough)
- `creates_tokens=True` + `success_multi()` → engine creates new tokens per output row
- `creates_tokens=False` + `success_multi()` → RuntimeError (except in aggregation passthrough mode)
- `success_multi([])` is invalid (no silent drops) - use `success_empty()` for intentional zero emission
- If `passes_through_input=True` and zero emission is intentional, the transform must also declare `can_drop_rows=True`

For ADR-009 invariant coverage, a transform participates through three hooks:
`probe_config()`, `forward_invariant_probe_rows()` /
`backward_invariant_probe_rows()`, and
`execute_forward_invariant_probe()` /
`execute_backward_invariant_probe()`. The harness owns discovery and
assertions; the transform owns how to instantiate, shape, and execute a
hermetic representative probe. This keeps external-call and
`BatchTransformMixin` transforms from forcing transport-specific branches into
the invariant harness itself.

#### Zero-Emission Filters

Transforms that intentionally emit zero rows must use `TransformResult.success_empty(...)`.

- `can_drop_rows=False` (default): zero-emission success is a runtime declaration violation for pass-through transforms
- `can_drop_rows=True`: pass-through transform may intentionally terminate the row with terminal outcome `DROPPED_BY_FILTER`
- `can_drop_rows` is governance, not batching: it does not permit `success_multi([])`

**Engine semantics:**
- When a deaggregation transform (`creates_tokens=True`) returns `success_multi()` for a single input token, the engine creates N child tokens via `expand_token()` and the parent token reaches terminal state `EXPANDED` (children continue).

#### Batch-Aware Transforms

Transforms can declare `is_batch_aware = True` to receive batched rows when used at aggregation nodes:

```python
class SummaryTransform(BaseTransform):
    name = "summary"
    is_batch_aware = True  # Engine may pass aggregated data when used at aggregation node
    # ... set input_schema, output_schema, _output_schema_config, and
    # self._aggregate_output_contract in __init__ ...

    def process(
        self,
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        total = float(sum(r["value"] for r in rows))
        return TransformResult.success(
            PipelineRow(
                {"total": total, "count": len(rows)},
                self._aggregate_output_contract,
            ),
            success_reason={"action": "aggregated", "batch_size": len(rows)},
        )
```

**How it works:**

1. Transform is configured at an aggregation node (see "Aggregation" below)
2. Engine buffers rows until trigger fires (count, timeout, condition)
3. Engine calls `transform.process(rows: list[PipelineRow], ctx)` with the batch
4. Transform returns aggregated result

> **Implementation note:** Batch-aware transforms implement `BatchTransformProtocol` — a separate protocol from `TransformProtocol`. The key difference is the `process()` signature: `BatchTransformProtocol.process(rows: list[PipelineRow], ctx)` receives a list of `PipelineRow`, while `TransformProtocol.process(row: PipelineRow, ctx)` receives a single row. Both protocols share most attributes (`name`, `input_schema`, `output_schema`, `is_batch_aware`, `creates_tokens`, `declared_output_fields`, and `_output_schema_config`).

**Key points:**
- `is_batch_aware = False` (default): Transform implements `TransformProtocol`, receives single `PipelineRow`
- `is_batch_aware = True`: Transform implements `BatchTransformProtocol`, receives `list[PipelineRow]` at aggregation nodes
- The engine decides when to batch based on pipeline configuration
- `declared_input_fields` is a single-row precondition surface today; batch-aware transforms must leave it empty until a batch pre-emission contract exists

#### Optional Configuration

```yaml
transforms:
  price_calculator:
    type: custom_transform
    # OPTIONAL: Where do rows go when transform returns error?
    on_error: failed_calculations  # Sink name, or "discard"
```

**`on_error`** (OPTIONAL):
- Specifies destination for rows where transform returns `TransformResult.error()`
- Value must be a sink name or `"discard"` for explicit drop
- If omitted and transform returns an error → `ConfigurationError` (pipeline crashes)
- Even when `"discard"`, a `TransformErrorEvent` is recorded in the audit trail

**Error vs Bug distinction:**

| Scenario | Signal | Behavior |
|----------|--------|----------|
| **Processing Error** | `TransformResult.error(...)` | Route to `on_error` sink |
| **Transform Bug** | Exception thrown | CRASH immediately |

```python
from elspeth.contracts.contract_propagation import propagate_contract
from elspeth.contracts.schema_contract import PipelineRow


# PROCESSING ERROR - legitimate, uses on_error routing
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    if row["quantity"] == 0:
        return TransformResult.error(
            {"reason": "division_by_zero", "field": "quantity"},
            retryable=False,
        )
    output = {**row.to_dict(), "unit_price": row["total"] / row["quantity"]}
    output_contract = propagate_contract(row.contract, output, transform_adds_fields=True)
    return TransformResult.success(
        PipelineRow(output, output_contract),
        success_reason={"action": "calculated_unit_price"},
    )

# TRANSFORM BUG - crashes, does NOT use on_error routing
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    return TransformResult.success(
        PipelineRow({"value": row["nonexistent"]}, row.contract),  # KeyError = BUG
        success_reason={"action": "processed"},
    )
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration."""

def process(
    self,
    row: PipelineRow,
    ctx: PluginContext,
) -> TransformResult:
    """Process a single row.

    Args:
        row: Input row as PipelineRow (immutable, supports dual-name access)
        ctx: Plugin context

    Returns:
        TransformResult.success(PipelineRow(...), success_reason={...}) - processed row
        TransformResult.success_multi([PipelineRow(...)], success_reason={...}) - multiple output rows
        TransformResult.error(reason, retryable=bool) - processing failed

    Notes:
        - For batch-aware transforms (is_batch_aware=True), the engine may buffer
          rows and call this with aggregated data when used in aggregation nodes
        - PipelineRow is immutable and supports both normalized and original field names
        - MUST be a pure function for DETERMINISTIC transforms

    Exception handling:
        - Data validation errors → return TransformResult.error()
        - Code bugs → let exception propagate (will crash)
    """

def close(self) -> None:
    """Release resources."""
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before any rows are processed."""

def on_complete(self, ctx: PluginContext) -> None:
    """Called after all rows are processed."""
```

#### TransformResult Contract

```python
@dataclass
class TransformResult:
    status: Literal["success", "error"]
    row: PipelineRow | None                            # Single output row (mutually exclusive with rows)
    rows: tuple[PipelineRow, ...] | None               # Multi-row output (mutually exclusive with row)
    reason: TransformErrorReason | None                # Error details or None (success)
    retryable: bool = False                            # Can this operation be retried?

    # Success metadata - REQUIRED for success results, None for error results
    success_reason: TransformSuccessReason | None

    # Audit fields (set by executor, NOT by plugin)
    input_hash: str | None
    output_hash: str | None
    duration_ms: float | None

    # Context snapshot for audit trail (optional)
    # Contains operational metadata like pool stats, ordering info
    context_after: NodeStateContext | None

    @property
    def is_multi_row(self) -> bool:
        """True if this result contains multiple output rows."""
        return self.rows is not None

    @property
    def has_output_data(self) -> bool:
        """True if this result has any output data (single or multi-row)."""
        return self.row is not None or self.rows is not None

    def to_pipeline_row(self) -> PipelineRow:
        """Convert single-row success result to PipelineRow."""
        ...

    def to_pipeline_rows(self) -> list[PipelineRow]:
        """Convert multi-row success result to list of PipelineRows."""
        ...
```

**Invariants:**
- `status == "success"` requires `has_output_data == True` AND `success_reason is not None`
- `row` and `rows` are mutually exclusive (exactly one is set on success)
- `status == "error"` requires `reason is not None`
- If `success_reason` is missing on success, `__post_init__` will crash (plugin bug)

**Factory methods:**

```python
# REQUIRED: success_reason parameter (keyword-only)
TransformResult.success(
    PipelineRow(output_dict, output_contract),
    success_reason={"action": "processed"},
    context_after=None,  # Optional operational metadata
)

output_rows = (PipelineRow(row1, output_contract), PipelineRow(row2, output_contract))
TransformResult.success_multi(
    output_rows,
    success_reason={"action": "split", "count": len(output_rows)},
    context_after=None,  # Optional operational metadata
)

TransformResult.success_empty(
    success_reason={"action": "filtered"},
    context_after=None,  # Optional operational metadata
)

TransformResult.error(
    reason={"reason": "invalid_input", "field": "amount"},
    retryable=False,
    context_after=None,  # Optional operational metadata from partial execution
)
```

**Multi-row usage:**

```python
from elspeth.contracts.contract_propagation import propagate_contract


# Deaggregation: 1 input → N outputs
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    items = row["items"]  # Trust: source validated this is a list
    output_rows = []
    for i, item in enumerate(items):
        output = {**row.to_dict(), "item": item, "item_index": i}
        output_contract = propagate_contract(row.contract, output, transform_adds_fields=True)
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)
        output_rows.append(PipelineRow(output, output_contract))
    return TransformResult.success_multi(
        output_rows,
        success_reason={"action": "exploded", "output_count": len(items)},
    )

# Aggregation passthrough: N inputs → N enriched outputs
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    # When is_batch_aware=True and used in aggregation, engine may pass aggregated data
    rows = row if isinstance(row, list) else [row]
    output_rows = []
    for r in rows:
        output = {**r.to_dict(), "batch_size": len(rows)}
        output_contract = propagate_contract(r.contract, output, transform_adds_fields=True)
        output_contract = self._apply_declared_output_field_contracts(output_contract)
        output_contract = self._align_output_contract(output_contract)
        output_rows.append(PipelineRow(output, output_contract))
    return TransformResult.success_multi(
        output_rows,
        success_reason={"action": "batch_enriched", "batch_size": len(rows)},
    )
```

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)
    │
    ▼
┌─────────────────────────────┐
│ process(row, ctx) × N       │  ← Called for each row
│     │                       │
│     ▼                       │
│ TransformResult             │
└─────────────────────────────┘
    │
    ▼
on_complete(ctx)
    │
    ▼
close()
```

#### Audit Records

- Each `process()` call: input_hash, output_hash, duration_ms, status
- Errors: exception type, message, retryable flag
- **TransformErrorEvent** for each `TransformResult.error()`:
  - `run_id`, `token_id`, `transform_id`
  - `row` (input row data)
  - `error_details` (from TransformResult.error())
  - `destination` (sink name or "discard")
  - `input_hash` (for traceability)
  - `timestamp`

---

### Sink

**Purpose:** Output data to external destination. One or more per run.

**Data Flow:** Receives rows, produces artifacts.

#### Required Attributes

```python
name: str
config_model: type[PluginConfig] | None
input_schema: type[PluginSchema]
node_id: str | None
idempotent: bool                # Can this sink handle duplicate writes safely?
determinism: Determinism
plugin_version: str
source_file_hash: str | None
declared_required_fields: frozenset[str]
```

#### Required Methods

```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize with configuration."""

def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> SinkWriteResult:
    """Receive rows and return proof of work.

    The sink controls its own internal processing:
    - May write immediately (CSV, database)
    - May queue internally and process later (satellite, async API)
    - May batch for efficiency

    MUST return SinkWriteResult describing what was produced/queued.
    SHOULD NOT block for slow operations - queue internally, confirm in on_complete().

    Returns:
        SinkWriteResult with ArtifactDescriptor content_hash and size_bytes
        (REQUIRED for audit) plus optional row diversions
    """

def flush(self) -> None:
    """Ensure all buffered writes are durable.

    MUST guarantee that when this method returns:
    - All data passed to write() is persisted
    - Data survives process crash
    - Data survives power loss (for file/block storage)

    This method MUST block until durability is guaranteed.

    For file-based sinks: Call file.flush() + os.fsync(file.fileno())
    For database sinks: Call connection.commit()
    For async sinks: Await flush completion

    This is called by the orchestrator BEFORE creating checkpoints.
    If this method returns successfully, the checkpoint system assumes
    all data is durable and will NOT replay these writes on resume.

    Called periodically and before on_complete().
    """

def close(self) -> None:
    """Release resources.

    Called after on_complete() or on error.
    """
```

#### Required Lifecycle Hooks

```python
def on_start(self, ctx: PluginContext) -> None:
    """Called before any writes.

    Use for: Opening connections, creating output files, initializing queues.
    """

def on_complete(self, ctx: PluginContext) -> None:
    """Called after all writes, before close.

    CRITICAL: This method MAY BLOCK until all queued work is confirmed.

    Use for:
    - Committing database transactions
    - Waiting for satellite transmission confirmation
    - Finalizing multi-part uploads
    - Any operation that must complete before run is considered done

    The run CANNOT complete until on_complete() returns for ALL sinks.
    """
```

#### SinkWriteResult and ArtifactDescriptor Contract

`write()` returns `SinkWriteResult`. Its `artifact` is the primary audit proof;
its `diversions` tuple carries per-row write failures recorded with
`BaseSink._divert_row()`.

```python
@dataclass(frozen=True)
class SinkWriteResult:
    artifact: ArtifactDescriptor
    diversions: tuple[RowDiversion, ...] = ()
```

```python
@dataclass(frozen=True)
class ArtifactDescriptor:
    artifact_type: Literal["file", "database", "webhook"]
    path_or_uri: str              # Where the artifact lives
    content_hash: str             # SHA-256 of content (REQUIRED)
    size_bytes: int               # Size of artifact (REQUIRED)
    metadata: dict | None = None  # Type-specific metadata
```

**Factory methods:**

```python
ArtifactDescriptor.for_file(path: str, content_hash, size_bytes)
ArtifactDescriptor.for_database(url: SanitizedDatabaseUrl, table, content_hash, payload_size, row_count)
ArtifactDescriptor.for_webhook(url: SanitizedWebhookUrl, content_hash, request_size, response_code)
```

> **URL sanitization is enforced at the type level.** `for_database()` and `for_webhook()` require pre-sanitized URL types, NOT plain strings. Use `SanitizedDatabaseUrl.from_raw_url(url)` or `SanitizedWebhookUrl.from_raw_url(url)` to strip credentials before passing to the factory. This prevents secrets from entering the audit trail. Both factory methods include `isinstance()` enforcement and raise `TypeError` on plain strings.

#### content_hash Requirement

The `content_hash` field is REQUIRED and proves what was written. Hash computation differs by artifact type:

| Artifact Type | What Gets Hashed | Why |
|---------------|------------------|-----|
| **file** | SHA-256 of file contents | Proves exact bytes written |
| **database** | SHA-256 of canonical JSON payload BEFORE insert | Proves intent (DB may transform data) |
| **webhook** | SHA-256 of request body | Proves what was sent (response is separate) |

**Key principle:** Hash what YOU control, not what the destination does with it. For databases, you hash the payload you're sending, not what the DB stores (it may add timestamps, auto-increment IDs, etc.). This proves intent.

#### Lifecycle

```
__init__(config)
    │
    ▼
on_start(ctx)               ← Open connections, prepare
    │
    ▼
┌──────────────────────────────────────────────┐
│ write(rows, ctx) → SinkWriteResult          │  ← May be called multiple times
│     │                                        │
│     ▼                                        │
│ (sink processes on its own schedule)         │
└──────────────────────────────────────────────┘
    │
    ▼
flush()                     ← Flush any buffers
    │
    ▼
on_complete(ctx)            ← BLOCKS until truly done
    │                         (satellite confirms, transaction commits)
    ▼
close()                     ← Release resources
```

#### Idempotency

Sinks declare `idempotent: bool`:
- `True`: Safe to retry writes with same data (uses idempotency key)
- `False`: Retry may cause duplicates (append-only files, non-idempotent APIs)

Idempotency key format: `{run_id}:{token_id}:{sink_name}`

#### Resume Capability

Sinks declare `supports_resume: bool` to indicate they can append to existing output on resume:

```python
supports_resume: bool   # Can this sink append to existing output on resume?
```

Sinks that support resume implement three additional methods:

| Method | Purpose |
|--------|---------|
| `configure_for_resume()` | Switch to append mode (called by engine on resume) |
| `validate_output_target()` | Validate existing output matches configured schema |
| `set_resume_field_resolution(mapping)` | Set source field resolution mapping before validation |

Sinks that don't support resume (`supports_resume=False`) will never have these methods called — the CLI rejects resume before execution.

#### Audit Records

- Each `write()`: SinkWriteResult with ArtifactDescriptor (type, path, hash, size) and any diversions
- `on_complete` timestamp (confirms delivery)
- Idempotency key if applicable

---

## System Operations (Engine-Level)

The following are **engine-level operations** that coordinate token flow through the DAG. They operate on **wrapped data** (tokens, routing metadata) rather than row contents directly.

```
┌──────────────────────────────────────────────────────────────────┐
│                     DATA FLOW ARCHITECTURE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  PLUGINS (touch row contents)                                    │
│  ────────────────────────────                                    │
│  Source ──► [row data] ──► Transform ──► [row data] ──► Sink    │
│                                    │                              │
│                                    ▼                              │
│  ROUTING DECISIONS                                                │
│  ─────────────────                                               │
│  Gate: "where does this token go?" (config-driven)               │
│                                                                   │
│  STRUCTURAL OPERATIONS (config-driven only)                      │
│  ──────────────────────────────────────────                      │
│  Fork: "copy token to parallel paths"                            │
│  Coalesce: "merge tokens from paths"                             │
│  Aggregation: "batch tokens until trigger"                       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

### Gate (Routing Decision)

**Purpose:** Evaluate a condition on row data and decide where the token goes next.

**Key property:** Gates make routing decisions. They evaluate conditions on row data and determine token destinations.

Gates are **config-driven operations** defined in YAML. They are NOT plugins.

#### Config Expression Gates

Config expressions provide declarative routing.

**Configuration:**

```yaml
pipeline:
  - source: csv_input

  - gate: quality_check
    condition: "row['confidence'] >= 0.85"
    routes:
      high: continue          # Continue to next node
      low: review_sink        # Route to named sink

  - transform: enrich_data

  - sink: output
```

**How It Works:**

1. Engine evaluates `condition` expression against row data
2. Expression returns a route label (`high`, `low`, etc.)
3. Engine looks up route label in `routes` config
4. Token is routed to destination (`continue` or sink name)

#### Expression Language

```python
# Simple field comparison
"row['score'] > 0.8"

# Multiple conditions
"row['status'] == 'active' and row['balance'] > 0"

# Field existence check
"'optional_field' in row"

# Null handling
"row.get('nullable_field') is not None"
```

#### Expression Safety

Gate conditions are evaluated using a **restricted expression parser**, NOT Python `eval()`.

**Allowed:**
- Field access: `row['field']`, `row.get('field')`
- Comparisons: `==`, `!=`, `<`, `>`, `<=`, `>=`
- Boolean operators: `and`, `or`, `not`
- Identity: `is`, `is not` (for `None` checks)
- Membership: `in`, `not in`
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`
- Ternary: `x if condition else y`
- Literals: strings, numbers, booleans, `None`
- Collections: lists, dicts, tuples, sets (for membership checks)

**NOT Allowed:**
- Function calls (except `row.get()`)
- Imports
- Attribute access beyond row fields
- Assignment (including `:=`)
- Lambda/comprehensions
- `await`, `yield`
- F-strings with expressions
- Starred expressions (`*x`, `**x`)
- Slice syntax

This prevents code injection. An expression like `"__import__('os').system('rm -rf /')"` will be rejected at config validation time.

#### GateResult Contract

The engine evaluates gate conditions and produces `GateResult` objects internally.

```python
@dataclass
class GateResult:
    row: dict[str, Any]        # Row data (passed through)
    action: RoutingAction      # Where the token goes

    # Schema contract for output (optional)
    # Enables conversion to PipelineRow via to_pipeline_row()
    contract: SchemaContract | None

    # Audit fields (set by executor, NOT by plugin)
    input_hash: str | None
    output_hash: str | None
    duration_ms: float | None

    def to_pipeline_row(self) -> PipelineRow:
        """Convert to PipelineRow for downstream processing."""
        ...

# RoutingAction options:
RoutingAction.continue_()              # Continue to next node in pipeline
RoutingAction.route("label")           # Route to labeled destination (resolved via routes config)
RoutingAction.fork_to_paths(["path1", "path2"]) # Fork to multiple parallel paths
```

#### Gate Audit Trail

Gates record:
- Condition evaluated: expression text
- Route chosen: label + destination
- Timing: evaluation duration

---

### Fork (Token Splitting)

**Purpose:** Copy a single token to multiple parallel paths for concurrent processing.

**Key property:** Creates child tokens with same row data, different branch identities.

#### Configuration

```yaml
pipeline:
  - source: input

  - gate: parallel_analysis
    condition: "True"  # Always fork
    routes:
      all: fork         # Special keyword triggers fork
    fork_to:
      - sentiment_path
      - entity_path
      - summary_path

  # Parallel paths defined separately
  paths:
    sentiment_path:
      - transform: sentiment_analyzer
    entity_path:
      - transform: entity_extractor
    summary_path:
      - transform: summarizer
```

#### How It Works

1. Gate evaluates condition
2. If route is `fork`, engine creates N child tokens
3. Each child token:
   - Has same `row_id` as parent
   - Gets unique `token_id`
   - Records `parent_token_id` for lineage
   - Assigned to specific path/branch
4. Parent token marked with terminal state `FORKED`
5. Child tokens flow through their respective paths

#### Token Lineage

```
Parent Token (T1)
    │ row_id: R1
    │ status: FORKED
    │
    ├──► Child Token (T2)
    │    row_id: R1
    │    parent_token_id: T1
    │    branch: sentiment_path
    │
    ├──► Child Token (T3)
    │    row_id: R1
    │    parent_token_id: T1
    │    branch: entity_path
    │
    └──► Child Token (T4)
         row_id: R1
         parent_token_id: T1
         branch: summary_path
```

#### Audit Trail

- Fork event: parent_token_id, child_token_ids, branch assignments
- Fork group ID: links all children from same fork

---

### Coalesce (Token Merging)

**Purpose:** Merge tokens from parallel paths back into a single token.

**Key property:** Waits for tokens from specified branches, combines based on policy.

> **Note:** Coalesce is fully structural — there is no plugin-level coalesce protocol. The engine uses config-driven merge strategies (union/nested/select) via the `CoalesceExecutor` in `engine/coalesce_executor.py`. Configure coalesce behavior via the `coalesce:` section in pipeline YAML.

#### Configuration

```yaml
pipeline:
  # ... fork happens earlier ...

  # List format (identity branches — no per-branch transforms):
  - coalesce: merge_results
    branches:
      - sentiment_path
      - entity_path
      - summary_path
    policy: require_all    # Wait for all branches
    timeout: 5m            # Max wait time
    merge: union           # How to combine row data
```

```yaml
pipeline:
  # ... fork happens earlier, transforms declared with input/on_success ...

  # Dict format (per-branch transforms — ARCH-15):
  # Keys are branch names (from fork_to), values are the input connections
  # that feed each branch into the coalesce node.
  - coalesce: merge_results
    branches:
      sentiment_path: sentiment_output    # branch routes through sentiment transform chain
      entity_path: entities_output        # branch routes through entity transform chain
      summary_path: summary_output        # branch routes through summary transform chain
    policy: require_all
    timeout: 5m
    merge: union
```

> **Note:** List format is syntactic sugar. `branches: [a, b]` normalizes to `branches: {a: a, b: b}` (identity mapping — tokens skip directly to coalesce with no intermediate transforms).

#### Policies

| Policy | Behavior |
|--------|----------|
| `require_all` | Wait for ALL branches, fail if any missing after timeout |
| `quorum` | Wait for N branches (configurable threshold) |
| `best_effort` | Wait until timeout, merge whatever arrived |
| `first` | Take first arrival, discard others |

#### Merge Strategies

| Strategy | Behavior |
|----------|----------|
| `union` | Combine all fields (later branches overwrite earlier) |
| `nested` | Each branch output becomes nested object: `{branch_name: output}` |
| `select` | Take output from specific branch only |

#### How It Works

1. Engine tracks arriving tokens by `row_id` and branch
2. When policy is satisfied (all arrived, quorum met, timeout):
   - Creates new merged token
   - Combines row data per merge strategy
   - Child tokens marked with terminal state `COALESCED`
3. Merged token continues down pipeline

#### Token Lineage

```
Child Tokens (arriving)
    │
    ├── T2 (sentiment_path): {sentiment: "positive"}
    ├── T3 (entity_path): {entities: ["ACME", "NYC"]}
    └── T4 (summary_path): {summary: "..."}
    │
    ▼ (coalesce with union strategy)
    │
Merged Token (T5)
    row_id: R1
    parent_token_ids: [T2, T3, T4]
    row_data: {
        sentiment: "positive",
        entities: ["ACME", "NYC"],
        summary: "..."
    }
```

#### Audit Trail

- Coalesce event: input_token_ids, output_token_id, policy used
- Timing: wait duration, which branches arrived when
- Merge details: strategy used, any conflicts

---

### Aggregation (Token Batching)

**Purpose:** Collect multiple tokens until a trigger fires, then process as batch.

**Key property:** Converts N input tokens into M output tokens (often N→1 for aggregates).

**Architecture:** Aggregation is **fully structural** - the engine owns the buffer and decides when to flush. There is no plugin-level aggregation protocol. When you need batch processing:

1. Configure an aggregation node in the pipeline
2. Use a batch-aware transform (`is_batch_aware = True`) at that node
3. Engine buffers rows and calls `transform.process(rows: list[PipelineRow], ctx)` when trigger fires

#### Configuration

```yaml
pipeline:
  - source: events

  # Aggregation configuration
  aggregations:
    - node_id: batch_stats      # Links to transform below
      trigger:
        count: 100              # Fire after 100 rows
        timeout_seconds: 3600   # Or after 1 hour
        # condition: "row['type'] == 'flush_signal'"  # Optional: trigger on special row

  transforms:
    - plugin: summary_transform
      node_id: batch_stats      # Same as aggregation node_id
      # Transform must have is_batch_aware = True
```

#### Output Mode

Aggregation supports two output modes that determine how batch results are handled:

```yaml
aggregations:
  - node_id: batch_stats
    trigger: { count: 100 }
    output_mode: transform          # default: N inputs → M outputs
    expected_output_count: 1        # optional: validate N→1 cardinality
```

| Mode | Input → Output | Token Handling | While Buffering | Use Case |
|------|----------------|----------------|-----------------|----------|
| `transform` | N → M | New tokens created | `CONSUMED_IN_BATCH` | Aggregation, group-by, splitting |
| `passthrough` | N → N | Same tokens preserved | `BUFFERED` | Batch enrichment |

**Outcome semantics:**

- **`transform`** (default): Batch transformation. N rows become M rows with new tokens. All input tokens are terminal (`CONSUMED_IN_BATCH`). New tokens are created via `expand_token()` with parent linkage to the triggering token. Use `expected_output_count: 1` for classic N→1 aggregation (sum, count, mean) to validate cardinality.

- **`passthrough`**: Batch enrichment. N rows become N enriched rows with the same token IDs. Buffered tokens get `BUFFERED` (non-terminal) while waiting, then reappear as `COMPLETED` on flush. Transform must return `success_multi()` with exactly N rows.

**Error handling:** All modes are atomic - if the transform returns `error`, ALL buffered rows fail together.

#### Trigger Types

| Trigger | Fires When |
|---------|------------|
| `count` | N tokens accumulated |
| `timeout_seconds` | Duration elapsed since batch start |
| `condition` | Row matches expression |
| `end_of_source` | Source exhausted (implicit, always checked) |

Multiple triggers can be combined (first one to fire wins).

#### How It Works (Engine-Level)

1. Tokens arrive at aggregation node
2. **Engine buffers** the row internally:
   - Assigns `batch_id`
   - Records batch membership in audit trail
   - Updates trigger evaluator
   - Token outcome depends on `output_mode`:
     - `transform`: `CONSUMED_IN_BATCH` (terminal)
     - `passthrough`: `BUFFERED` (non-terminal, will reappear as `COMPLETED`)
3. Engine checks trigger conditions
4. When trigger fires:
   - Engine retrieves buffered rows as `list[PipelineRow]`
   - Engine calls `transform.process(rows, ctx)` with the batch
   - Batch state: `draft` → `executing` → `completed`
   - Output handling depends on `output_mode`:
     - `passthrough`: All buffered tokens continue with enriched data
     - `transform`: New tokens created via `expand_token()` continue
   - Buffer is cleared for next batch

**Important:** The engine owns the buffer, not the transform. Transforms simply receive `list[PipelineRow]` and return a result. This enables:
- Crash recovery (buffers are checkpointed)
- Consistent trigger evaluation
- Clean audit trail

#### Batch Lifecycle

```
Batch B1 (draft) - Engine buffers rows
    │
    ├── Buffer T1.row_data → batch: [row1]
    ├── Buffer T2.row_data → batch: [row1, row2]
    ├── Buffer T3.row_data → batch: [row1, row2, row3]
    │
    ▼ (trigger: count >= 3)
    │
Batch B1 (executing)
    │
    ▼ Engine calls transform.process([row1, row2, row3], ctx)
    │
Batch B1 (completed)
    │
    └──► Output Token T4
         row_data: {count: 3, sum: 150, avg: 50}

Input tokens T1, T2, T3 → terminal state: CONSUMED_IN_BATCH
```

#### Crash Recovery

The engine persists buffer state in checkpoints:
- `get_checkpoint_state()` serializes buffered rows and batch metadata
- `restore_from_checkpoint()` restores buffers after crash
- Trigger evaluators resume from correct count

This means in-progress batches survive crashes and can be resumed.

#### Audit Trail

- Batch created: batch_id, trigger config, aggregation_node_id
- Batch membership: which tokens belong to which batch (ordinal position)
- Batch state transitions: draft → executing → completed/failed
- Trigger event: which condition fired, when
- Transform input/output hashes for the batch call

---

### Why Not Plugins?

These operations are engine-level because:

1. **They don't touch row contents** - A gate evaluates a condition but doesn't modify the row. Fork copies tokens but doesn't change data. Coalesce combines outputs but the merge strategy is config, not code.

2. **They require DAG coordination** - Fork/coalesce semantics span multiple nodes. The engine must track token lineage, manage parallel paths, handle timeouts. This is orchestration, not business logic.

3. **Config is sufficient** - All behavior is expressible via:
   - Expressions: `"row['field'] > value"`
   - Policies: `require_all`, `quorum`, `best_effort`
   - Strategies: `union`, `nested`, `select`

4. **100% our code** - These are ELSPETH internals with comprehensive testing. No user extension points, no defensive programming needed.

---

## Exception Handling Summary

| Method | On Failure |
|--------|------------|
| `__init__` | Raise immediately - misconfiguration |
| `on_start` | CODE BUG → crash |
| `on_complete` | CODE BUG → crash |
| `close` | Log error, don't raise |
| `load` (Source) | Validation failure → quarantine via `on_validation_failure`; Code bug → crash |
| `process` (Transform) | Processing error → route via `on_error`; Code bug → crash |
| `write` (Sink) | Data error → return error result; Code bug → raise |

### Error Routing Summary

| Plugin Type | Config Field | Required? | On Missing Config + Error |
|-------------|--------------|-----------|---------------------------|
| **Source** | `on_validation_failure` | Yes | N/A (config validation fails) |
| **Transform** | `on_error` | No | `ConfigurationError` - pipeline crashes |
| **Sink** | N/A | N/A | Sinks don't route errors |

---

## Determinism Declaration

Plugins declare their determinism level:

```python
class Determinism(StrEnum):
    DETERMINISTIC = "deterministic"          # Same input → same output, always
    SEEDED = "seeded"                        # Same input → same output given captured seed
    IO_READ = "io_read"                      # Reads from external state (files/env/time)
    IO_WRITE = "io_write"                    # Writes side effects (files/db)
    EXTERNAL_CALL = "external_call"          # Network/API calls (record request/response)
    NON_DETERMINISTIC = "non_deterministic"  # Cannot be reproduced (record outputs)
```

**Implications:**
- `DETERMINISTIC`: Safe to replay; verify can recompute and compare
- `SEEDED`: Capture and replay with same seed
- `IO_READ`: Capture what was read (inputs/metadata) for replay/verify
- `IO_WRITE`: Side effects; replay requires care and idempotency
- `EXTERNAL_CALL`: Record request/response for replay; verify can diff responses
- `NON_DETERMINISTIC`: Record outputs; verify cannot recompute deterministically

---

## Engine Concerns (Not Plugin Contract)

The following are handled by the ELSPETH engine, not by plugins. They are documented here for completeness but are NOT part of the plugin contract.

### Schema Validation

The engine validates schema compatibility between connected nodes at pipeline construction time:
- Source `output_schema` → Transform `input_schema`
- Transform `output_schema` → next Transform `input_schema` or Sink `input_schema`

Plugins declare schemas; the engine enforces compatibility. Plugins do NOT validate schemas themselves.

### Retry Policy

Retry behavior is configured at the engine level, not declared by plugins:

```yaml
retry:
  max_attempts: 3
  base_delay: 1.0
  max_delay: 60.0
  jitter: 1.0
```

Plugins indicate whether errors are `retryable` in their result objects. The engine decides whether and when to retry based on policy.

### Rate Limiting

External call rate limits are engine configuration:

```yaml
rate_limit:
  calls_per_second: 10
  burst: 20
```

Plugins make calls; the engine throttles them.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.12 | 2026-05-07 | Accuracy pass for current runtime contracts — documented required `SourceRow.valid(..., contract=...)`, `TransformResult` `PipelineRow`-only outputs, `SinkWriteResult` sink returns, dynamic built-in discovery, `config_model`/`source_file_hash` identity surfaces, and transform-side external Tier 3 boundaries |
| 1.11 | 2026-04-20 | Added `declared_input_fields`, `declared_output_fields`, and `can_drop_rows` transform declarations; documented `TransformResult.success_empty()` and `DROPPED_BY_FILTER` zero-emission semantics |
| 1.10 | 2026-02-13 | RC-3 alignment — Fixed stale "(config OR plugin)" gate description to "(config-driven)" (gate plugins removed 2026-02-11). Corrected gate key property to reflect routing-only behavior (no row modification). |
| 1.9 | 2026-02-08 | Second accuracy pass — Fixed `Determinism` comment (all 6 levels, not 3), `Enum` → `StrEnum`, `list[dict]` → `list[PipelineRow]` in aggregation section, `invalid_data` → `invalid_input` error category, `RoutingAction.route()` parameter clarification, `BatchTransformProtocol` attribute precision |
| 1.8 | 2026-02-08 | Accuracy pass — Fixed `RoutingAction.fork()` → `fork_to_paths()`, documented `SanitizedDatabaseUrl`/`SanitizedWebhookUrl` for ArtifactDescriptor factories, documented `BatchTransformProtocol` as separate protocol, added the legacy adds-fields declaration later superseded by `declared_output_fields`, expanded expression language reference, fixed `ctx.run_started_at` reference, documented sink resume capability, noted `CoalesceProtocol` existence |
| 1.7 | 2026-02-05 | **CRITICAL UPDATES**: (1) `TransformResult.success()` requires `success_reason` parameter (keyword-only, crashes if missing), (2) Row type changed from `dict[str, Any]` to `PipelineRow` in Transform/Gate signatures, (3) Added `context_after` field for operational metadata, (4) Added schema-contract conversion methods on result types, (5) Enhanced `flush()` documentation with durability guarantees, (6) Updated `SourceRow` to document `contract` parameter and `to_pipeline_row()` method, (7) Fixed `load()` return type to `Iterator[SourceRow]` |
| 1.6 | 2026-01-20 | Gate documentation: config expression gates, `GateResult` contract, routing actions |
| 1.5 | 2026-01-19 | Multi-row output: `creates_tokens` attribute, `TransformResult.success_multi()`, `rows` field, aggregation `output_mode` (passthrough/transform), `BUFFERED` and `EXPANDED` outcomes |
| 1.4 | 2026-01-19 | Batch-aware transforms (`is_batch_aware`), structural aggregation (engine owns buffers), crash recovery for batches |
| 1.3 | 2026-01-17 | Source `on_validation_failure` (required), Transform `on_error` (optional), QuarantineEvent, TransformErrorEvent |
| 1.2 | 2026-01-17 | Three-tier trust model, coercion rules by plugin type, type-safe ≠ operation-safe |
| 1.1 | 2026-01-17 | Add content_hash requirements, expression safety, engine concerns |
| 1.0 | 2026-01-17 | Initial contract - Source, Transform, Sink + System Operations |
