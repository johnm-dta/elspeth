## Rating Scale
- 1 = Low: small surface, straightforward logic, strong tests, few external interactions
- 3 = Medium: moderate complexity and/or IO/dynamic behavior; tests mitigate but don’t eliminate
- 5 = High: complex stateful logic and tight coupling, tricky invariants, partial/unstable integration, or known mismatch signals

## Subsystem Bug-Likelihood Ratings

### 1) Contracts (`src/elspeth/contracts/`) — **1/5 (Low)**
- Why: Mostly declarative models/enums; failures tend to be immediate and unit-tested.
- Signals: Dense test suite under `tests/contracts/`.

### 2) Configuration (`src/elspeth/core/config.py`, `src/elspeth/plugins/config_base.py`) — **3/5 (Medium)**
- Why: Many interacting settings + cross-field invariants (routes, fork_to, export sink, schema requirements).
- Signals: Config-time expression validation (`ExpressionParser`) and several model validators.

### 3) Canonicalization & Hashing (`src/elspeth/core/canonical.py`) — **3/5 (Medium)**
- Why: Data-type normalization (pandas/numpy/datetime/Decimal/bytes) is edge-case heavy.
- Signals: Explicit NaN/Infinity rejection; correctness-critical for audit integrity.

### 4) Execution Graph / DAG (`src/elspeth/core/dag.py`) — **3/5 (Medium)**
- Why: Route label resolution, gate edge construction, and fork semantics are easy to get subtly wrong.
- Signals: Multiple internal maps (`_route_resolution_map`, `_route_label_map`, id maps) that must stay consistent.

### 5) Landscape Audit DB (`src/elspeth/core/landscape/`) — **4/5 (Med–High)**
- Why: Large schema + many write paths + strict “Tier 1” behavior; correctness depends on consistent enum/string conversions and invariants.
- Signals: `LandscapeRecorder` is a large surface area (runs/nodes/edges/tokens/states/routing/batches/artifacts/errors) and couples to many subsystems.

### 6) Engine Runtime (`src/elspeth/engine/`) — **4/5 (Med–High)**
- Why: Central state machine (token identity, fork/route/coalesce, aggregation triggers, retry semantics, sink batching); high coupling to graph + recorder + plugins.
- Signals: Work-queue DAG execution with an iteration guard, multiple executors, and “terminal state is derived” rules that must stay consistent.

### 7) Plugin Framework (`src/elspeth/plugins/`) — **3/5 (Medium)**
- Why: Dynamic discovery (pluggy), schema hashing, and protocol compliance enforcement are failure-prone at edges.
- Signals: Schema hashing relies on Pydantic `model_fields`; raises hard on protocol violations.

### 8) Built-in Plugins (`src/elspeth/plugins/sources|transforms|sinks/`) — **3/5 (Medium)**
- Why: IO + validation + hashing. Sources sit at the untrusted boundary; sinks enforce strict typing.
- Notable risk: `PluginContext.route_to_sink()` is explicitly a stub; source validation failures currently don’t have an engine-level delivery mechanism (they are not yielded).

### 9) Checkpointing & Recovery (`src/elspeth/core/checkpoint/`) — **5/5 (High)**
- Why: Crash recovery is stateful and invariant-heavy; current implementation has a semantic mismatch risk.
- Signals: `RecoveryManager.get_unprocessed_rows()` treats `sequence_number` like `row_index`, while `checkpoints_table` documents it as a generic monotonic marker and `Orchestrator` increments an internal counter (not `row_index`).

### 10) Rate Limiting (`src/elspeth/core/rate_limit/`) — **4/5 (Med–High)**
- Why: Threading + global `threading.excepthook` override to suppress upstream library race noise is brittle and can have surprising interactions.
- Signals: Multi-bucket logic to work around pyrate-limiter behavior; persistent SQLite mode; background thread cleanup.

### 11) Retention / Purge (`src/elspeth/core/retention/` + `elspeth purge`) — **3/5 (Medium)**
- Why: Deletion logic is operationally risky; needs careful selection criteria and audit-grade guarantees.
- Signals: Depends on correct run status/timestamps and payload-store behavior; currently doesn’t measure bytes freed.

### 12) CLI + Explain TUI (`src/elspeth/cli.py`, `src/elspeth/tui/`) — **3/5 (Medium)**
- Why: CLI is mostly glue but touches many subsystems; TUI has partial implementations and intentional graceful-degradation paths.
- Signals: `ExplainApp` UI is placeholder; `ExplainScreen` has broad exception handling (allowlisted) for incomplete/corrupt runs.

## Highest-Leverage “Bug Smell” Hotspots
- Checkpoint sequence semantics mismatch (`src/elspeth/engine/orchestrator.py`, `src/elspeth/core/checkpoint/recovery.py`, `src/elspeth/core/landscape/schema.py`)
- Source quarantine routing is not end-to-end (`src/elspeth/plugins/context.py`, `src/elspeth/plugins/sources/*.py`)
- Routing/edge integrity invariants (gate labels ↔ edge_map ↔ route_resolution_map) (`src/elspeth/core/dag.py`, `src/elspeth/engine/executors.py`)
- Thread lifecycle suppression in rate limiter (`src/elspeth/core/rate_limit/limiter.py`)
