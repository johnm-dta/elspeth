# ELSPETH Architecture

**Version:** 1.1
**Date:** 2026-01-12
**Status:** Design

---

## Executive Summary

ELSPETH is a **domain-agnostic Sense/Decide/Act (SDA) framework** designed for high-reliability, auditable data processing workflows. While the initial use case involves LLM-powered decision-making, the architecture supports any decision system: machine learning models, rules engines, threshold-based classifiers, or custom algorithms.

The framework is designed for cases requiring **high-level attributability** - every output must be traceable to its source with complete audit trail.

---

## Design Principles

### 1. Auditability First

Every decision the system makes must be explainable:

> "This output came from transform X, which processed input Y with config Z, at time T, and routed to sink W because of reason R"

This is not optional telemetry - it's the core of the system.

### 2. Domain Agnostic

The framework makes no assumptions about what transforms do:

| Domain | Sense | Decide | Act |
|--------|-------|--------|-----|
| Tender Evaluation | CSV submissions | LLM classification | Results + abuse queue |
| Weather Monitoring | Sensor API | Threshold + ML model | Log, warning, emergency |
| Satellite Ops | Telemetry stream | Anomaly detection | Routine, investigate, intervene |
| Financial Compliance | Transaction feed | Rules + ML fraud | Approved, flagged, blocked |
| Medical Triage | Patient intake | Symptom classifier | Routine, urgent, emergency |

Same framework, different plugins.

### 3. Routing as First-Class Citizen

Rows don't just flow through a pipeline - they can be **routed** to different destinations based on classification decisions. A "gate" transform can send a row to:

- The next transform (continue)
- A named sink (route with reason)

This enables patterns like routing emergency readings to an alert system while continuing to log routine readings.

### 4. Reliability Over Performance

This is a high-reliability system, not a high-throughput system. Design choices favor:

- Correctness over speed
- Auditability over efficiency
- Explicit over implicit

---

## Core Concepts

### The SDA Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                              SENSE                                   │
│                                                                      │
│  Load data from sources into the system                              │
│  Examples: CSV file, database query, API poll, message queue         │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                              DECIDE                                  │
│                                                                      │
│  Transform and classify data through plugin chain                    │
│  Examples: LLM query, ML inference, rules check, threshold gate      │
│                                                                      │
│  Transforms can be:                                                  │
│  - Pass-through: always continue to next stage                       │
│  - Gate: evaluate and route to different destinations                │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                               ACT                                    │
│                                                                      │
│  Output results to sinks                                             │
│  Examples: Write file, call API, insert database, send alert         │
│                                                                      │
│  Sinks can receive:                                                  │
│  - Normal results (end of pipeline)                                  │
│  - Routed rows (from gates, with classification reason)              │
└─────────────────────────────────────────────────────────────────────┘
```

### Plugin Taxonomy

There are exactly **three plugin primitives**:

| Primitive | Role | State |
|-----------|------|-------|
| **Source** | Get data into the system | Stateless |
| **Transform** | Do something with data | Stateless between rows* |
| **Sink** | Act on data | Stateless |

*Aggregation transforms accumulate state across rows until a trigger condition.

The distinction between "LLM plugin" and "ML plugin" is artificial - both are transforms that happen to make external calls. This keeps the mental model clean and the routing model uniform.

### Transform Subtypes

| Type | Behavior | Routing |
|------|----------|---------|
| **Pass-through** | Process row, always continue | No routing decision |
| **Gate** | Evaluate row, decide destination | Returns `continue` or sink name |
| **Aggregation** | Collect rows until trigger, then process batch | May or may not route |

### Conditional Routing

Gates return a routing decision alongside the (possibly modified) row:

```python
@dataclass
class RoutingAction:
    """What the gate decided to do."""
    kind: Literal["continue", "route_to_sink", "fork_to_paths"]
    destinations: list[str]      # Sink names or path names
    mode: Literal["move", "copy"] = "move"
    reason: dict[str, Any]       # Classification metadata

@dataclass
class GateResult:
    row: dict[str, Any]          # Possibly modified row
    action: RoutingAction        # What to do with the row
```

**Routing modes:**

| Mode | Behavior |
|------|----------|
| `move` | Row goes to destination, exits current path |
| `copy` | Row goes to destination AND continues on current path |

Example flow (weather monitoring):

```
Reading → QualityGate ─┬─ "low_confidence" → discard_log
                       └─ "valid" → continue
                                      │
                                      ▼
                       SeverityGate ─┬─ "emergency" → emergency_broadcast
                                     ├─ "warning" → warning_alerts
                                     └─ "normal" → continue
                                                      │
                                                      ▼
                                                [routine_log]
```

---

## The Execution Graph

### Pipeline as DAG

Elspeth compiles pipeline configurations into a **directed acyclic graph (DAG)** of nodes and edges. This is the true execution model; pipelines are syntactic sugar.

```
User writes:                         Engine compiles to:
─────────────                        ───────────────────
row_plugins:                         source
  - quality_gate                        │
  - privacy_gate          ───→          ▼
  - main_eval                        quality_gate ──→ quarantine
output_sink: results                    │ (continue)
                                        ▼
                                     privacy_gate ──→ review
                                        │ (continue)
                                        ▼
                                     main_eval
                                        │
                                        ▼
                                     results_sink
```

### Why DAG?

Once you have:

- Multi-destination routing (`route: [A, B, C]`)
- Parallel paths that merge (coalesce)

...you have a DAG, whether you admit it or not. Building DAG-awareness into the core prevents future rewrites.

### DAG Primitives

| Primitive | Role |
|-----------|------|
| **Node** | Source, Transform, Gate, Aggregation, Coalesce, Sink |
| **Edge** | Connection between nodes with label (route name) and mode (copy/move) |
| **Token** | Instance of a row flowing through a specific path |

### Token Identity

In a linear pipeline, `row_id` is sufficient. In a DAG with parallel branches, we need to track **instances**:

| Concept | Purpose |
|---------|---------|
| `row_id` | Original source row identity (stable, from source) |
| `token_id` | Instance of row flowing through a specific path |
| `parent_token_id` | Lineage across forks and joins |

**Token lifecycle:**

```
Source:  row_id=R1 → token_id=T1

Fork:    token T1 forks to paths [A, B]
         → token T2 (parent=T1, branch=A)
         → token T3 (parent=T1, branch=B)

Join:    tokens [T2, T3] coalesce
         → token T4 (parents=[T2, T3])
```

This enables precise lineage: "This output token came from these parent tokens, which trace back to row R1."

### Linear Pipelines are Degenerate DAGs

Most configurations remain simple:

```yaml
row_plugins:
  - plugin: enrich
  - plugin: classify
  - plugin: validate
output_sink: results
```

This compiles to a DAG where every node has exactly one `continue` edge. Users never see DAG complexity unless they use routing or coalesce.

---

## The Landscape (Audit System)

### Purpose

The Landscape is the **audit backbone** of Elspeth. It captures:

- Every run with its resolved configuration
- Every plugin instance registered for the run
- Every row loaded from the source
- Every transform applied to every row (with before/after state)
- Every external call made by transforms (LLM, HTTP, ML inference)
- Every routing decision (continue, route, halt)
- Every artifact produced by sinks

### Mental Model: Distributed Tracing for Data Pipelines

The Landscape is analogous to OpenTelemetry/Jaeger, where:

- **Rows** are like requests
- **Transform spans** are like spans
- **External calls** are like child spans
- **Routing decisions** are like span events

### Core Invariants

These invariants MUST hold across all operations:

1. **Run Reproducibility**: Every run stores resolved config (not just hash)
2. **Deterministic Linkage**: External calls link to spans that exist at call time
3. **Strict Ordering**: Transforms ordered by (sequence, attempt); calls ordered by (state_id, call_index)
4. **No Orphan Records**: Foreign keys enforced (`PRAGMA foreign_keys=ON` in SQLite)
5. **Uniqueness**: (run_id, row_id) unique; (state_id, call_index) unique
6. **Canonical JSON Contract**: Hash algorithm versioned, never silently changed

### Row Identity

Rows are identified by a stable `row_id` (UUID or content-derived), not by `row_index`:

| Field | Purpose |
|-------|---------|
| `row_id` | Stable identity - survives reprocessing, out-of-order execution |
| `row_index` | Presentation/debugging - position in original source |

This allows concurrent processing and partial reprocessing without identity collisions.

### Data Model (Simplified)

```
runs
  └── nodes (execution graph vertices)
  └── edges (execution graph connections)
  └── rows (source data)
        └── tokens (row instances in DAG)
              └── token_parents (for joins)
              └── node_states (what happened at each node)
                    └── calls (external calls within state)
              └── routing_events (edge selections)
  └── batches (aggregation groups)
        └── batch_members (which tokens fed batch)
        └── batch_outputs (what batch produced)
  └── artifacts (sink outputs)
```

### The Attributability Test

Given any output, prove complete lineage. In a DAG, use `token_id` for precision:

```python
def test_can_explain_any_output():
    """Given any output token, prove complete lineage to source."""
    # Option A (recommended): explain by token_id
    lineage = landscape.explain(run_id, token_id=token_id, field=field)

    # Option B: explain by row_id (only valid for linear paths)
    # lineage = landscape.explain(run_id, row_id=row_id, sink=sink_name, field=field)

    # Verify complete chain exists
    assert lineage.source_row is not None
    assert len(lineage.node_states) > 0

    # For successful transforms, verify both hashes
    for state in lineage.node_states:
        assert state.input_hash is not None
        if state.status == "completed":
            assert state.output_hash is not None

    # Verify token parentage (for joins)
    if lineage.token.parents:
        for parent_token_id in lineage.token.parents:
            parent_lineage = landscape.explain(run_id, token_id=parent_token_id)
            assert parent_lineage is not None

    # Verify call linkage
    for call in lineage.calls:
        assert any(s.state_id == call.state_id for s in lineage.node_states)
```

**explain() API options:**

| Signature | Use Case |
|-----------|----------|
| `explain(run_id, token_id, field)` | Precise - works for any DAG |
| `explain(run_id, row_id, sink, field)` | Convenience - disambiguates by target sink |
| `explain(run_id, row_id, field)` | Only valid when row has single terminal path |

### Audit Trail Export

For compliance and legal inquiry, the Landscape can be exported after a run completes:

```yaml
landscape:
  url: sqlite:///./runs/audit.db
  export:
    enabled: true
    sink: audit_archive       # Reference to configured sink
    format: csv               # csv or json
    sign: true                # HMAC signature per record
```

**Export flow:**
1. Run completes normally
2. Orchestrator queries all audit data for run
3. Records formatted and written to export sink
4. If `sign: true`, each record gets HMAC signature + final manifest

**Signing provides:**
- Per-record integrity verification
- Chain-of-custody proof via running hash
- Manifest with final hash for tamper detection

**Environment:**
- `ELSPETH_SIGNING_KEY`: Required for signed exports (UTF-8 encoded string)

**Redaction note:** Redaction is the responsibility of plugins BEFORE invoking Landscape recording methods. The Landscape is a faithful recorder - it stores what it's given. The export therefore exports exactly what was recorded.

---

## Canonical JSON and Hashing

### Why This Matters

Hashes are only meaningful with deterministic serialization. Python's built-in `hash()` is not stable across processes. We need cryptographic hashes over canonical byte representations.

### The Two-Phase Approach

Canonicalization happens in two phases:

1. **Normalize** (our code): Convert pandas/numpy types to JSON-safe primitives, reject NaN/Infinity
2. **Serialize** (`rfc8785`): Produce deterministic JSON per RFC 8785 (JSON Canonicalization Scheme)

This catches issues early with clear error messages. Phase 1 handles pandas/numpy quirks; Phase 2 uses a standards-compliant implementation for the actual serialization.

### Canonicalization Rules (v1)

```python
import math
import hashlib
import base64
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import rfc8785  # RFC 8785 JSON Canonicalization Scheme

CANONICAL_VERSION = "sha256-rfc8785-v1"


# === Phase 1: Normalization (our code) ===

def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states, not "missing"
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records
    """
    # === Check for NaN/Infinity FIRST (before pd.isna can convert them) ===
    # This ensures consistent rejection regardless of type origin
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Numpy scalar types (common when reading CSVs with pandas)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_normalize_value(x) for x in obj.tolist()]

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        # Naive timestamps are assumed UTC (explicit policy)
        return obj.tz_convert("UTC").isoformat() if obj.tz else obj.tz_localize("UTC").isoformat()

    # Intentional missing values (NOT NaN - that's rejected above)
    if obj is None or obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
        return None

    # Standard library types that need conversion
    if isinstance(obj, datetime):
        if obj.tzinfo is None:
            obj = obj.replace(tzinfo=timezone.utc)
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Decimal):
        return str(obj)  # Preserve precision

    return obj


def _normalize_for_canonical(data: Any) -> Any:
    """Recursively normalize a data structure for canonical JSON.

    Converts pandas/numpy types to JSON-safe primitives.
    Raises ValueError for NaN, Infinity, or other non-serializable values.
    """
    if isinstance(data, dict):
        return {k: _normalize_for_canonical(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_normalize_for_canonical(v) for v in data]
    return _normalize_value(data)


# === Phase 2: Serialization (rfc8785) ===

def canonical_json(obj: Any) -> str:
    """Produce canonical JSON for hashing.

    Two-phase approach:
    1. Normalize pandas/numpy types to JSON-safe primitives (our code)
    2. Serialize per RFC 8785/JCS standard (rfc8785 package)

    Raises:
        ValueError: If data contains NaN, Infinity, or other non-finite values
        TypeError: If data contains types that cannot be serialized
    """
    normalized = _normalize_for_canonical(obj)
    return rfc8785.dumps(normalized)


def stable_hash(obj: Any, version: str = CANONICAL_VERSION) -> str:
    """Compute stable hash of object.

    Args:
        obj: Data structure to hash
        version: Hash algorithm version (stored with runs for verification)

    Returns:
        SHA-256 hex digest of canonical JSON
    """
    canonical = canonical_json(obj)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
```

### Rules

| Type | Canonicalization |
|------|------------------|
| Keys | Sorted alphabetically (recursive) |
| Strings | UTF-8, no escaping beyond JSON spec |
| Numbers | JSON default representation |
| Floats | NaN/Infinity **rejected** (raises ValueError) |
| `numpy.integer` | Converted to Python `int` |
| `numpy.floating` | Converted to Python `float` (checked for finite) |
| `numpy.bool_` | Converted to Python `bool` |
| `pandas.Timestamp` | UTC ISO8601 string (naive assumed UTC) |
| `None`, `pd.NA`, `pd.NaT` | Converted to `null` |
| `numpy.datetime64` | **Unsupported** - normalize upstream to `pd.Timestamp` |
| `numpy.timedelta64`, `pd.Timedelta` | **Unsupported** - normalize upstream |
| `datetime` | UTC ISO8601 string |
| `bytes` | Base64 with `{"__bytes__": ...}` wrapper |
| `Decimal` | String representation (preserves precision) |
| `None` | `null` |

### Why `allow_nan=False` AND Explicit Checks?

The `default=` parameter in `json.dumps()` only fires for types JSON doesn't recognize natively.
Since Python `float` is a native JSON type, the `default` function is **never called** for floats.
This means NaN/Infinity would slip through without `allow_nan=False`.

We check explicitly in `_normalize_value()` for clear error messages, and use `allow_nan=False` as defense-in-depth.

### Version Contract

The `canonical_version` is stored with every run. If canonicalization rules change, the version increments. Old hashes remain valid under their recorded version.

---

## Payload Storage and Retention

### The Problem

Storing full before/after state for every transform on every row, plus full request/response for every external call, will become enormous. A single LLM-heavy run could generate gigabytes of audit data.

### Solution: Payload Store Abstraction

```python
class PayloadStore(Protocol):
    """Separate large payloads from core audit tables."""

    def put(self, data: bytes, content_type: str) -> PayloadRef: ...
    def get(self, ref: PayloadRef) -> bytes: ...
    def exists(self, ref: PayloadRef) -> bool: ...
```

Landscape tables store:

- `payload_ref` - Reference to payload store
- `payload_hash` - Hash for integrity verification
- `payload_size` - Size in bytes
- `content_type` - MIME type or internal type marker

### Retention Strategy

| Data Type | Default Retention | After Expiry |
|-----------|-------------------|--------------|
| Run metadata | Indefinite | Keep |
| Transform hashes | Indefinite | Keep |
| Routing decisions | Indefinite | Keep |
| Row payloads | 90 days | Purge, keep hash |
| Call payloads | 90 days | Purge, keep hash |
| Artifacts | Per artifact policy | Per policy |

After retention expiry, `explain()` still works at the hash+metadata level. The system explicitly reports when full payloads are no longer available.

### Configuration

```yaml
landscape:
  retention:
    row_payloads_days: 90
    call_payloads_days: 90
    compress_after_days: 7
```

---

## Failure Semantics

### Token Terminal States (Derived)

Every token ends in a terminal state. There are no silent drops.

**Important:** Terminal states are *derived* from node_states, routing_events, and batch membership—not stored as a column. This avoids redundant state.

```python
class TokenTerminalState(Enum):
    """Final disposition of a token - derived, not stored."""
    COMPLETED = "completed"           # Reached output sink
    ROUTED = "routed"                 # Sent to named sink by gate (move mode)
    FORKED = "forked"                 # Split into child tokens
    CONSUMED_IN_BATCH = "consumed"    # Fed into aggregation batch
    COALESCED = "coalesced"           # Merged with other tokens
    QUARANTINED = "quarantined"       # Failed, stored for investigation
    FAILED = "failed"                 # Failed, not recoverable
```

**Contrast with `node_states.status`:** The `status` column in `node_states` tracks *processing status at a single node* (`open`, `completed`, `failed`), not the token's terminal disposition.

### Transform Results

Transforms return a structured result:

```python
@dataclass
class TransformResult:
    """Result of a transform operation."""
    status: Literal["success", "error"]  # Note: "route" removed; routing is via GateResult
    row: dict[str, Any] | None           # Modified row (if success)
    reason: dict[str, Any] | None        # Error metadata (if error)
    retryable: bool = False              # Can this be retried?
```

### Retry Semantics

Retries are explicit attempts with ordering:

- `(run_id, row_id, transform_seq, attempt)` is unique
- Each attempt is recorded separately
- Final outcome indicates which attempt succeeded (or all failed)
- Backoff metadata captured (delay, reason, policy)

### Sink Idempotency

Sinks receive idempotency keys to prevent duplicate side effects:

```python
idempotency_key = f"{run_id}:{row_id}:{sink_name}:{artifact_kind}"
```

For sinks that cannot guarantee idempotency (e.g., some webhooks), the system flags this risk in configuration validation.

### Delivery Guarantee

The system provides **at-least-once** delivery with idempotency recommended. If a sink write fails after partial execution, the row may be reprocessed. Sinks should be idempotent or explicitly acknowledge this limitation.

---

## External Call Recording

### Terminology: Recompute vs Replay vs Verify

Precise terminology prevents confusion for implementers and auditors:

| Term | Applies To | Meaning |
|------|------------|---------|
| **Recompute** | Deterministic transforms | Run the code again; expect identical output hashes |
| **Replay** | Non-deterministic calls | Substitute recorded responses; no live calls |
| **Verify** | Non-deterministic calls | Run live AND compare to recorded; flag differences |

**Key distinction:**

- Deterministic transforms don't "replay" - they **recompute**. Same code + same input = same output.
- Non-deterministic calls **replay** recorded responses or **verify** against them.

### Run-Level Reproducibility Grade

Every run is assigned a reproducibility grade based on its transforms:

| Grade | Meaning | `explain()` capability |
|-------|---------|------------------------|
| `FULL_REPRODUCIBLE` | All transforms deterministic | Recompute any output from source |
| `REPLAY_REPRODUCIBLE` | Has non-deterministic calls, but payloads retained | Replay to identical downstream outputs |
| `ATTRIBUTABLE_ONLY` | Payloads purged or absent | Lineage and hashes exist, cannot replay |

The grade is computed at run completion and stored in run metadata. It may degrade over time as payloads are purged.

### Run Modes

| Mode | Behavior |
|------|----------|
| `live` | Call external services, record request/response |
| `replay` | Use recorded responses, no external calls |
| `verify` | Call external, compare against recorded (flag differences) |

**Verify Mode Implementation:**

Use DeepDiff for detailed comparison of recorded vs live responses:

```python
from deepdiff import DeepDiff

def verify_external_call(recorded_response: dict, live_response: dict, call_id: str) -> VerifyResult:
    """Compare recorded and live responses, recording any drift."""
    diff = DeepDiff(
        recorded_response,
        live_response,
        ignore_order=True,  # List order may not matter
        exclude_paths=["root['id']", "root['created']"],  # Non-deterministic fields
    )

    if diff:
        landscape.record_verification_drift(
            call_id=call_id,
            diff_summary=diff.to_dict(),
            severity=classify_drift(diff),
        )
        return VerifyResult(matched=False, diff=diff)

    return VerifyResult(matched=True, diff=None)
```

This provides forensic detail for the audit trail - not just *that* something changed, but *what* changed.

### Recorded Data

For each external call:

- Provider identifier (e.g., `openai`, `azure`, `weather.gov`)
- Model/version if available
- Request hash + payload ref
- Response hash + payload ref
- Latency, status code, error details

---

## Data Governance

### Redaction

The Landscape can become a sensitive data honeypot. Design for minimal disclosure:

| Data Type | Policy |
|-----------|--------|
| Secrets (API keys) | Never stored - hash fingerprint only |
| PII in payloads | Configurable redaction profiles |
| Intermediate data | Full storage with retention limits |

### Secret Handling

Secrets are NEVER written to Landscape. We store a fingerprint for "same secret used" verification.

**Why HMAC, not plain hash?**

Plain hashing (`sha256(secret)`) creates an offline guessing oracle. An attacker with fingerprints can pre-compute hashes for common API keys or passwords.

HMAC with a managed key requires the attacker to have **both** the fingerprint **and** the key.

```python
import hmac
import hashlib

# fingerprint_key is managed like an internal secret (rotated, not stored in Landscape)
def secret_fingerprint(secret_value: str, fingerprint_key: bytes) -> str:
    """Generate a fingerprint for a secret without creating a guessing oracle."""
    return hmac.new(
        fingerprint_key,
        secret_value.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

# Usage in plugin config
config_for_landscape = {
    "model": "gpt-4",
    "api_key": "[REDACTED]",
    "api_key_fingerprint": secret_fingerprint(api_key, FINGERPRINT_KEY),
}
```

**Key management:**

- `fingerprint_key` is loaded from environment or secrets manager
- Rotate like any internal secret
- Never stored in Landscape (only the fingerprints are)

### Access Levels

| Role | Access |
|------|--------|
| Operator | Redacted explain view (default) |
| Auditor | Full explain with payloads (requires explicit grant) |
| Admin | Retention management, purge operations |

```bash
# Default: redacted view
elspeth explain --run abc123 --row xyz

# Full view (requires ELSPETH_AUDIT_ACCESS=full)
elspeth explain --run abc123 --row xyz --full
```

---

## Technology Stack

### Core Framework (Domain-Agnostic)

| Component | Technology | Rationale |
|-----------|------------|-----------|
| CLI | Typer | Type-safe, auto-generated help |
| TUI | Textual | Interactive terminal UI for `explain`, `status` |
| Configuration | Dynaconf + Pydantic | Multi-source precedence + validation |
| Plugin System | pluggy | Battle-tested (pytest uses it) |
| Data | pandas | Standard for tabular data |
| HTTP | httpx | Modern async support |
| Database | SQLAlchemy Core | Multi-backend without ORM overhead |
| Migrations | Alembic | Schema versioning |
| Retries | tenacity | Industry standard backoff |

### Acceleration Stack (Avoid Reinventing)

These libraries replace components that would otherwise become "mini-products" requiring ongoing maintenance:

| Component | Technology | Replaces | Rationale |
|-----------|------------|----------|-----------|
| Canonical JSON | `rfc8785` | Hand-rolled serialization | RFC 8785/JCS standard; we keep our normalization layer |
| DAG Validation | NetworkX | Custom graph algorithms | Acyclicity checks, topological sort, graph queries |
| Observability | OpenTelemetry | Custom tracing | Emit spans from same events as Landscape; OTEL is view, Landscape is truth |
| Tracing UI | Jaeger | Custom dashboards | Immediate visualization while building Landscape UI |
| Logging | structlog | Ad-hoc logging | Structured key/value events alongside Landscape |
| Rate Limiting | pyrate-limiter | Custom leaky buckets | Multi-limit/interval support, SQLite/Redis persistence |
| Diffing | DeepDiff | Custom comparison | Deep nested diffs for verify mode (recorded vs live) |
| Property Testing | Hypothesis | Manual edge cases | Find nasty canonicalization, DAG, and lineage bugs |

### Optional Plugin Packs

| Pack | Technology | Use Case |
|------|------------|----------|
| LLM | LiteLLM | 100+ LLM providers unified |
| ML | scikit-learn, ONNX | Traditional ML inference |
| Azure | azure-storage-blob, azure-identity | Azure cloud integration |

### Landscape Storage

| Environment | Backend | Notes |
|-------------|---------|-------|
| Development | SQLite | WAL mode, foreign keys enforced |
| Production | PostgreSQL | Partitioned by run_id, batch writes |

---

## Configuration

### Precedence Levels

Configurations merge with clear precedence (higher overrides lower):

1. System defaults (lowest)
2. Prompt/plugin packs
3. Profile configuration
4. Suite defaults
5. Runtime overrides (highest)

### Example Configuration

```yaml
datasource:
  plugin: http_poll
  options:
    url: https://api.weather.gov/stations/${STATION}/observations
    interval_seconds: 60

sinks:
  routine_log:
    plugin: database
    options:
      table: weather_readings

  warning_alerts:
    plugin: webhook
    options:
      url: https://alerts.internal/warning
      idempotent: false  # Explicitly flag non-idempotent sink

  emergency_broadcast:
    plugin: multi_sink
    options:
      targets:
        - webhook: https://emergency.gov/broadcast
        - database: emergency_log

row_plugins:
  - plugin: threshold_gate
    type: gate
    options:
      field: sensor_confidence
      min: 0.8
    routes:
      pass: continue
      fail: routine_log

  - plugin: threshold_gate
    type: gate
    options:
      rules:
        - field: wind_speed
          operator: ">="
          value: 150
          result: emergency
    routes:
      emergency: emergency_broadcast
      normal: continue

output_sink: routine_log

landscape:
  enabled: true
  backend: sqlite
  path: ./runs/landscape.db
  retention:
    row_payloads_days: 90
    call_payloads_days: 90
  redaction:
    profile: standard  # Redact known PII patterns
```

---

## Security Model

### Classification

This is a **high-reliability** system, not a high-security system. The threat model is:

- Accountability and auditability (high attributability standard)
- Data integrity verification (HMAC signing)
- Secret management (environment variables, redaction in logs)

Not in scope:

- PROTECTED + data handling
- Supply chain verification
- Container signing

### Artifact Signing

Sinks can sign outputs with HMAC-SHA256:

```python
signature = hmac.new(key, canonical_json(artifact), hashlib.sha256).hexdigest()
```

---

## Implementation Phases

**Design principle:** Prove the DAG infrastructure with deterministic transforms before adding external calls. A previous design iteration failed because LLMs were too tightly coupled to the orchestrator. The framework must be domain-agnostic first.

### Phase 1: Foundation (P0) - Prove the Core

**Goal:** Complete audit trail for deterministic transforms. No external calls yet.

- Project scaffold with dependencies (acceleration stack)
- Canonical JSON and hashing (two-phase with `rfc8785`)
  - Unit tests for "nasty" cases: `numpy.int64`, `numpy.float64`, `pandas.Timestamp`, `NaT`, `NaN`, `Infinity`
  - Cross-process hash stability test
- Landscape core (SQLAlchemy + SQLite)
- Payload store abstraction
- Configuration (Dynaconf + Pydantic)
- DAG validation with NetworkX

### Phase 2: Plugin System (P0) - Domain-Agnostic Extensibility

**Goal:** Plugins handle all decision logic; framework is neutral.

- pluggy hookspecs for Source, Transform, Sink
- TracedTransformPlugin base class with spans
- Gate protocol with routing
- RowOutcome and TransformResult models
- Schema contracts for plugin I/O

### Phase 3: SDA Engine (P0) - Orchestrator Without Opinions

**Goal:** Engine orchestrates flow; knows nothing about what transforms do.

- RowProcessor with span lifecycle
- Retry with attempt tracking (tenacity)
- Artifact pipeline (topological sort via NetworkX)
- Standard orchestrator
- OpenTelemetry span emission

### Phase 4: CLI & Basic I/O (P1)

**Goal:** Usable system with deterministic transforms only.

- CLI (Typer + Textual TUI)
- Basic sources (CSV, JSON)
- Basic sinks (CSV, JSON, database)
- `elspeth explain` with Textual interface
- structlog integration

### Phase 5: Production Hardening (P1)

- Checkpointing with replay support
- Rate limiting (pyrate-limiter)
- Retention and purge jobs
- Redaction profiles

### Phase 6: External Calls (P2) - Add Non-Determinism

**Goal:** Now add LLM/HTTP/ML plugins, with call recording proven on a solid foundation.

- External call recording infrastructure
- Record/replay/verify modes (with DeepDiff)
- LLM plugin pack (LiteLLM)
- HTTP plugin for general APIs
- Property tests (Hypothesis)

### Phase 7: Advanced (P2)

- Experimental orchestrator (A/B testing)
- Multi-destination routing (copy semantics)
- Azure plugin pack

---

## Open Questions

1. **Multi-destination routing**: Should gates support `copy` (send to sink AND continue) vs `move` (send to sink, exit pipeline)?
2. **Aggregation lineage**: How to link batch outputs back to constituent row_ids?
3. **Streaming sources**: Support for continuous data streams vs batch?
4. **Cross-run lineage**: Can outputs from run A be inputs to run B with preserved lineage?
