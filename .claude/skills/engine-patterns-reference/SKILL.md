---
name: engine-patterns-reference
description: >
  Engine internals and implementation patterns — composite primary keys, schema contracts,
  header normalization, aggregation timeouts, canonical JSON, retry semantics, secret
  handling, test path integrity, tech stack, source layout, and the attributability test.
  Use when working on engine code, Landscape queries, DAG validation, hashing, secrets,
  or when you need to understand the source tree structure.
---

# Engine Patterns Reference

> **Model complexity warning:** Engine code touches audit integrity (Tier 1),
> plugin boundaries, and DAG execution. Do not delegate engine work to fast/simple
> models — they will produce plausible-looking code that violates composite PK
> joins, bypasses production code paths in tests, or adds defensive `.get()` on
> Tier 1 data. Use Sonnet or Opus and include this skill in the prompt.

## Quick Decision Table (Mechanical — No Judgment Required)

For engine code, match the situation to this table. First match wins.

| I'm writing... | Key rule | Common mistake |
|----------------|----------|----------------|
| A Landscape query joining `node_states` | Use `node_states.run_id` directly, never join through `nodes` on `node_id` alone | Ambiguous join — `node_id` reused across runs |
| A `json.loads()` on data from our DB | No try/except — crash on failure. This is Tier 1 data. | Adding `except json.JSONDecodeError: return {}` |
| A test that builds an `ExecutionGraph` | Use `ExecutionGraph.from_plugin_instances()`, not manual construction | `graph._branch_to_coalesce = {...}` bypasses production logic |
| A test that loads pipeline config | Use `instantiate_plugins_from_config()`, not manual plugin construction | Building plugins by hand skips validation that production does |
| Canonical JSON for hashing | Use `rfc8785.dumps()` after `_normalize_for_canonical()`. Reject NaN/Infinity. | `json.dumps(sort_keys=True)` is not deterministic |
| A `from_settings()` mapping | Map every Settings field explicitly. Run `scripts.check_contracts`. | Adding a field to Settings but forgetting the Runtime*Config mapping |
| An aggregation trigger | Timeout fires on next row arrival, not during idle. Document this. | Assuming timeout fires in real-time |
| A retry state row | `(token_id, node_id, attempt)` is unique. Each attempt = separate row. | Updating the previous attempt row instead of inserting a new one |
| Secret handling code | HMAC fingerprint only. Never store the actual secret. | Logging the secret value, even at DEBUG level |
| Header/field name logic | Normalize at source boundary only. Use `SchemaContract.original_name` for restoration. | Re-normalizing in transforms or sinks |

## Fix Right, Not Quick

If you are proposing a fix that involves "a patch or temporary workaround," STOP.
We only have one chance to fix things pre-release. Make the fix right, not quick.
This especially includes architectural defects. Lint failures, failing tests, and
CI/CD issues must all be resolved to merge — no exceptions.

## Composite Primary Key Pattern: nodes Table

**CRITICAL:** The `nodes` table has a composite primary key `(node_id, run_id)`. This means the same `node_id` can exist in multiple runs when the same pipeline runs multiple times.

**Queries touching `node_states` must use `node_states.run_id` directly:**

```python
# WRONG - Ambiguous join when node_id is reused across runs
query = (
    select(calls_table)
    .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
    .join(nodes_table, node_states_table.c.node_id == nodes_table.c.node_id)  # BUG!
    .where(nodes_table.c.run_id == run_id)
)

# CORRECT - Use denormalized run_id on node_states
query = (
    select(calls_table)
    .join(node_states_table, calls_table.c.state_id == node_states_table.c.state_id)
    .where(node_states_table.c.run_id == run_id)  # Direct filter, no ambiguous join
)
```

**Why:** The `node_states` table has a denormalized `run_id` column (schema comment: "Added for composite FK"). Use it directly instead of joining through `nodes` table. The join on `node_id` alone would match multiple nodes rows when `node_id` is reused.

**If you MUST access columns from `nodes` table:** Use composite join with BOTH keys:

```python
.join(
    nodes_table,
    (node_states_table.c.node_id == nodes_table.c.node_id) &
    (node_states_table.c.run_id == nodes_table.c.run_id)
)
```

## Schema Contracts (DAG Validation)

Transforms can declare field requirements that are validated at DAG construction:

```yaml
# Source guarantees these fields in output
source:
  plugin: csv
  options:
    schema:
      mode: observed
      guaranteed_fields: [customer_id, amount]

# Transform requires these fields in input
transforms:
  - plugin: llm_classifier
    options:
      required_input_fields: [customer_id, amount]
```

The DAG validates that upstream `guaranteed_fields` satisfy downstream `required_input_fields`. For template-based transforms, use `elspeth.core.templates.extract_jinja2_fields()` to discover template dependencies during development.

## Header Normalization and Engine Custody

**Source field names are normalized to valid Python identifiers at the source boundary.** This is non-negotiable — it's not cosmetic cleanup, it's a language boundary requirement.

User-chosen field names must cross multiple language boundaries during pipeline execution: Python attribute access (`row.field`), Jinja2 templates (`{{ row.field }}`), expression parser AST nodes, SQL column names, and JSON keys. Each has different reserved words, quoting rules, and valid character sets. Normalizing once at the source boundary (Tier 3 -> Tier 2 transition) means every downstream consumer gets names that are valid in every context.

**The engine takes custody of the original headers.** The `SchemaContract` preserves `original_name` metadata on every field throughout the pipeline. At the sink boundary, three header modes control output:

| Mode | Output | Use Case |
|------|--------|----------|
| `NORMALIZED` | `customer_id` | Default — pipeline working names |
| `ORIGINAL` | `Customer ID` | Restore what the source provided |
| `CUSTOM` | Explicit mapping | External system handover with specific naming requirements |

This is a **restoration**, not a transformation — the engine preserved the originals as metadata and gives them back on request. The `CUSTOM` mode deliberately refuses to silently fall back to normalized names for unmapped fields, because wrong column names in an external system handover are worse than a crash.

**Collision detection operates on normalized names.** The transform executor (`engine/executors/transform.py`) centrally enforces field collision detection pre-execution for any transform that declares `declared_output_fields`. This is mandatory and engine-level — plugins don't opt in, they declare their output fields and the engine prevents collisions before the transform runs.

## Aggregation Timeout Behavior

Aggregation triggers fire in two ways:

- **Count trigger**: Fires immediately when row count threshold is reached
- **Timeout trigger**: Checked **before** each row is processed

**Known Limitation (True Idle):** Timeout triggers fire when the next row arrives, not during completely idle periods. If no rows arrive, buffered data won't flush until either:

1. A new row arrives (triggering the timeout check)
2. The source completes (triggering end-of-source flush)

For streaming sources that may never end, combine timeout with count triggers, or implement periodic heartbeat rows at the source level.

## Canonical JSON - Two-Phase with RFC 8785

**NaN and Infinity are strictly rejected, not silently converted.** This is defense-in-depth for audit integrity:

```python
import rfc8785

# Two-phase canonicalization
def canonical_json(obj: Any) -> str:
    normalized = _normalize_for_canonical(obj)  # Phase 1: pandas/numpy -> primitives (ours)
    return rfc8785.dumps(normalized)            # Phase 2: RFC 8785/JCS standard serialization
```

- **Phase 1 (our code)**: Normalize pandas/numpy types, reject NaN/Infinity
- **Phase 2 (`rfc8785`)**: Deterministic JSON per RFC 8785 (JSON Canonicalization Scheme)

Test cases must cover: `numpy.int64`, `numpy.float64`, `pandas.Timestamp`, `NaT`, `NaN`, `Infinity`.

## Retry Semantics

- `(token_id, node_id, attempt)` is unique in the `node_states` table (also `(token_id, step_index, attempt)`)
- Each attempt recorded as a separate `node_states` row
- Backoff metadata captured

## Secret Handling

Never store secrets directly - use HMAC fingerprints for audit:

```python
fingerprint = hmac.new(fingerprint_key, secret.encode(), hashlib.sha256).hexdigest()
```

Secrets can be loaded from environment variables (default) or Azure Key Vault via the `secrets:` section in `settings.yaml`:

```yaml
secrets:
  source: keyvault
  vault_url: https://my-vault.vault.azure.net  # Must be literal URL, not env var reference
  mapping:
    AZURE_OPENAI_KEY: azure-openai-key
    ELSPETH_FINGERPRINT_KEY: elspeth-fingerprint-key
```

`vault_url` must be a literal HTTPS URL because secrets are loaded before environment variable resolution. Secret resolutions are recorded in the `secret_resolutions` Landscape table (vault source, HMAC fingerprint, latency).

## Test Path Integrity

**Never bypass production code paths in tests.** BUG-LINEAGE-01 hid for weeks because tests manually built `ExecutionGraph` objects instead of using `from_plugin_instances()`. Manual construction had the correct mapping; the production path had a different (wrong) one. Tests passed, production was broken.

```python
# WRONG - bypasses production logic
graph = ExecutionGraph()
graph.add_node("source", ...)
graph._branch_to_coalesce = {"path_a": "merge1"}  # Tests pass, production breaks

# CORRECT - exercises the real code path
graph = ExecutionGraph.from_plugin_instances(source=source, transforms=transforms, ...)
```

**Rules:** Integration tests MUST use `ExecutionGraph.from_plugin_instances()` and `instantiate_plugins_from_config()`. Manual construction is acceptable only for unit tests of isolated algorithms (topo sort, cycle detection, visualization).

## The Attributability Test

For any output, the system must prove complete lineage:

```python
from elspeth.core.landscape.lineage import explain

lineage = explain(recorder, run_id, token_id=token_id)
assert lineage is not None
assert lineage.source_row is not None
assert len(lineage.node_states) > 0
```

## Technology Stack

### Core Framework

| Component | Technology | Rationale |
| --------- | ---------- | --------- |
| CLI | Typer | Type-safe, auto-generated help |
| TUI | Textual | Interactive terminal UI for `explain`, `status` |
| Configuration | Dynaconf + Pydantic | Multi-source precedence + validation |
| Plugins | pluggy | Battle-tested (pytest uses it) |
| Data | pandas | Standard for tabular data |
| Database | SQLAlchemy Core | Multi-backend without ORM overhead |
| Migrations | Alembic | Schema versioning |
| Retries | tenacity | Industry standard backoff |
| Telemetry | OpenTelemetry (api, sdk, otlp exporter) | Distributed tracing and metrics |

### Acceleration Stack (avoid reinventing)

| Component | Technology | Replaces |
| --------- | ---------- | -------- |
| Canonical JSON | `rfc8785` | Hand-rolled serialization (RFC 8785/JCS standard) |
| DAG Validation | NetworkX | Custom graph algorithms (acyclicity, topo sort) |
| Logging | structlog | Ad-hoc logging (structured events) |
| Rate Limiting | pyrate-limiter | Custom leaky buckets |
| Diffing | DeepDiff | Custom comparison (for verify mode) |
| Property Testing | Hypothesis | Manual edge-case hunting |

### Optional Plugin Packs

| Pack | Technology | Use Case |
| ---- | ---------- | -------- |
| LLM | LiteLLM | 100+ LLM providers unified |
| Azure | azure-storage-blob, azure-identity, azure-keyvault-secrets | Azure cloud integration |
| Telemetry | ddtrace | Datadog APM integration |
| Web | beautifulsoup4, html2text | Web scraping transforms |
| Security | sqlcipher3 | Audit database encryption at rest |
| MCP | mcp | Landscape analysis server protocol |

## Source Layout

```text
src/elspeth/
+-- core/
|   +-- landscape/      # Audit trail storage (recorder, exporter, schema)
|   +-- checkpoint/     # Crash recovery checkpoints
|   +-- dag/            # DAG construction, validation, graph models (NetworkX)
|   +-- rate_limit/     # Rate limiting for external calls
|   +-- retention/      # Payload purge policies
|   +-- security/       # Secret fingerprinting via HMAC, URL/IP validation
|   +-- config.py       # Configuration loading (Dynaconf + Pydantic)
|   +-- canonical.py    # Deterministic JSON hashing (RFC 8785)
|   +-- events.py       # Synchronous event bus for CLI observability
|   +-- expression_parser.py # AST-based expression parsing (no eval)
|   +-- identifiers.py  # ID generation utilities
|   +-- logging.py      # Structured logging setup
|   +-- operations.py   # Operation type definitions
|   +-- payload_store.py # Content-addressable storage for large blobs
|   +-- templates.py    # Jinja2 field extraction
+-- contracts/          # Type contracts, schemas, protocol definitions, hashing primitives, and phase-typed contexts
+-- engine/
|   +-- orchestrator/   # Full run lifecycle management (core, aggregation, export, outcomes, validation)
|   +-- executors/      # Transform, gate, sink, aggregation executors
|   +-- processor.py    # DAG traversal with work queue
|   +-- dag_navigator.py # DAG path navigation
|   +-- coalesce_executor.py # Fork/join barrier with merge policies
|   +-- batch_adapter.py # Batch windowing logic
|   +-- retry.py        # Tenacity-based retry with backoff
|   +-- tokens.py       # Token identity and lineage management
|   +-- triggers.py     # Aggregation trigger evaluation
|   +-- clock.py        # Clock abstraction for testing
|   +-- spans.py        # Telemetry span management
+-- plugins/
|   +-- infrastructure/ # Shared: base classes, protocols, config, clients, batching, pooling
|   +-- sources/        # CSVSource, JSONSource, NullSource, AzureBlobSource
|   +-- transforms/     # FieldMapper, Passthrough, Truncate, LLM, Azure safety, etc.
|   +-- sinks/          # CSVSink, JSONSink, DatabaseSink, AzureBlobSink
+-- telemetry/          # OpenTelemetry exporters and instrumentation
+-- testing/            # ChaosLLM, ChaosWeb, ChaosEngine test servers
|   +-- chaosllm/       # Fake OpenAI/Azure LLM server with error/latency injection
|   +-- chaosweb/       # Fake web server with failure profiles
|   +-- chaosengine/    # Engine-level test utilities
|   +-- chaosllm_mcp/   # MCP server for ChaosLLM metrics and control
+-- mcp/                # Landscape MCP analysis server
|   +-- analyzers/      # Domain-specific analysis tools (contracts, diagnostics, queries, reports)
|   +-- server.py       # MCP server implementation
|   +-- types.py        # MCP type definitions
+-- tui/                # Terminal UI (Textual) - explain screens and widgets
|   +-- screens/        # TUI screen implementations
|   +-- widgets/        # TUI widget components
|   +-- explain_app.py  # Explain TUI application entry point
|   +-- constants.py    # TUI constants
+-- cli.py              # Typer CLI
+-- cli_helpers.py      # CLI utility functions
+-- cli_formatters.py   # Event formatting for CLI output
```

## Layer Architecture & Dependency Analysis (`enforce_tier_model.py`)

ELSPETH's layer model (L0 contracts → L1 core → L2 engine → L3
plugins/web/mcp/tui/cli/telemetry/testing) is enforced by
`scripts/cicd/enforce_tier_model.py`. This script does **dual duty** (per
ADR-006 Phase 5) — it is both the defensive-pattern scanner *and* the
layer-import enforcer. Treat the two roles as separate even though they
share a process.

The script exposes **two subcommands** with different exit-code semantics:

| Subcommand | Role | Exit code | Use for |
|---|---|---|---|
| `check` | Conformance gate | non-zero on violations | CI build gate, pre-commit hook, regression checks |
| `dump-edges` | Architecture observation | always 0 | L2 cluster analysis, refactor planning, SCC discovery, dependency graph diffing |

Mixing the two roles in one CLI invocation is forbidden — `check` is a
build gate (failure stops CI), `dump-edges` is observational (failure
would be tool-level, not architectural).

### `check` — conformance gate

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check \
  --root src/elspeth \
  --allowlist config/cicd/enforce_tier_model
```

Detects:

- **Upward imports** (rule `L1`) — e.g., `engine/` importing from `plugins/`.
- **TYPE_CHECKING-guarded upward imports** (rule `TC`) — warning, not
  failure. Architecturally impure but no runtime coupling.
- **Defensive-pattern violations** (rules `R1`–`R9`) — `.get()` on Tier 1
  data, `hasattr()` use, silent exception handling, etc. (See the
  skill's tier-model section for the full Tier 1/2/3 distinction.)

Exit non-zero on any finding outside the allowlist. Allowlist entries
live in `config/cicd/enforce_tier_model/` and require a justification
comment. **Do not add allowlist entries to silence findings without a
real architectural reason** — the allowlist is for documented
exemptions, not for hiding bugs.

### `dump-edges` — architecture observation

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py dump-edges \
  --root src/elspeth \
  --format json \
  --output /tmp/l3-import-graph.json \
  --no-timestamp
```

Emits the deterministic intra-layer import graph as a machine-readable
artefact. Three output formats are supported and all derived from the
same AST walk:

| Format | Use for |
|---|---|
| `json` | Programmatic consumption (cite by JSON path); the load-bearing format |
| `mermaid` | Inline visualisation in Markdown / docs |
| `dot` | Graphviz rendering (`dot -Tsvg input.dot`) for high-fidelity diagrams |

Key flags:

| Flag | Effect |
|---|---|
| `--include-layer L3` (repeatable) | Filter edges by source/target layer; default L3 only |
| `--collapse-to-subsystem` | Aggregate file-level edges to subsystem granularity (default ON) |
| `--no-timestamp` | Emit a stable placeholder for the `generated_at` field; produces byte-identical output across runs |

**Determinism contract:** Given the same source tree and `--no-timestamp`,
output is byte-identical. This means the artefact can live in `git` or be
diffed across branches without spurious churn. CI integration is permitted
but optional — `dump-edges` is observational, not a gate.

### JSON schema (v1.0)

```json
{
  "schema_version": "1.0",
  "generated_at": "...",
  "tool_version": "...",
  "scope": { "root": "src/elspeth", "layers_included": ["L3"], "collapsed_to_subsystem": true },
  "nodes": [ { "id": "plugins/transforms/llm", "layer": "L3", "file_count": 12, "loc": 6431 }, ... ],
  "edges": [
    {
      "from": "web/composer", "to": "plugins/transforms/llm",
      "weight": 12,
      "type_checking_only": false, "conditional": false, "reexport": false,
      "sample_sites": [ { "file": "src/elspeth/web/composer/tools.py", "line": 142 } ]
    }, ...
  ],
  "stats": {
    "total_nodes": 33, "total_edges": 77,
    "type_checking_edges": 0, "conditional_edges": 2, "reexport_edges": 0,
    "scc_count": 5, "largest_scc_size": 7
  },
  "strongly_connected_components": [ ["web", "web/auth", ...], ... ]
}
```

**Edge metadata semantics:**

| Field | Meaning | Why it matters |
|---|---|---|
| `weight` | Count of distinct import statements producing this edge | High weight = load-bearing coupling; the edge is unlikely to be incidental |
| `type_checking_only` | All sites are inside `if TYPE_CHECKING:` blocks | No runtime coupling; architecturally impure but not load-bearing |
| `conditional` | All sites are inside `try`/`if`/etc. blocks at runtime | Coupling exists but is gated; investigate the gate condition |
| `reexport` | Edge passes through an `__init__.py` re-export | Hidden coupling — easy to miss because the import statement looks intra-package |

The `AND-across-sites` aggregation rule means a flag is `true` only if
**every** underlying site qualifies. An edge with one TYPE_CHECKING site
and one runtime site is `type_checking_only: false` — it is load-bearing.
This is the right semantics for cluster analysis: "is this edge
unconditionally coupled at runtime?" not "does any import statement
happen to be conditional?"

### Strongly-connected components (SCCs)

Cycles between modules are detected via Tarjan's algorithm
(`networkx.strongly_connected_components`). Non-trivial SCCs (size ≥2)
are surfaced in `strongly_connected_components` and counted in
`stats.scc_count` / `stats.largest_scc_size`.

**Cycles are observational, not enforcement.** The script does not fail
the build on cycle detection. Cycles are an architectural finding to be
triaged: some are benign (Python's `package ↔ subpackage` re-export
pattern), some are tactical accumulation (sub-packages mutually
importing because no clean boundary was drawn).

When you find a non-trivial SCC, the right response depends on the
shape:

| SCC shape | Interpretation | Action |
|---|---|---|
| `pkg ↔ pkg/sub` (size 2) | Re-export pattern — `__init__.py` exposes names from `pkg/sub` | Almost always benign; verify by reading `__init__.py` |
| `pkg/a ↔ pkg/b` peer cycle | Sibling sub-packages mutually coupled | Real cycle; flag for cluster boundary review |
| Multi-node tangle (size ≥4) | No acyclic decomposition possible | Treat as a unit in any cluster analysis; do NOT pretend to break it |

The architecture analysis under
`docs/arch-analysis-2026-04-29-1500/` found a 7-node SCC spanning every
`web/*` sub-package. That cluster is analysed as a unit in
`clusters/composer/`; do not attempt to decompose it without explicit
architectural buy-in.

### When to use which subcommand

| Situation | Subcommand |
|---|---|
| CI build gate / pre-commit hook | `check` |
| Pre-PR sanity scan after a refactor | `check` |
| Investigating "where does X end up coupled to?" | `dump-edges` |
| Planning a cluster split or sub-package extraction | `dump-edges --collapse-to-subsystem` |
| Comparing two branches' coupling shape | `dump-edges --no-timestamp` then `diff` |
| Producing an architecture diagram | `dump-edges --format mermaid` (inline) or `--format dot` (rendered) |

Both subcommands respect the same `--root` and `--allowlist` flags. Both
are AST-based — the script does **not** use grep, regex, or string
matching to derive imports, so conditional, multi-line, and re-exported
imports are all correctly attributed.

### Citation discipline when consuming `dump-edges` output

When writing analysis or design docs that cite the graph, **cite by JSON
path**, not by paraphrasing the artefact's content. This is the same
discipline the L1/L2/Phase 8 archaeology passes used:

```text
[ORACLE: stats.scc_count = 5, stats.largest_scc_size = 7]
[ORACLE: edge plugins/sinks → plugins/infrastructure, weight 45]
[ORACLE: edges with from='mcp' and to startswith 'composer_mcp' = 0]
[ORACLE: strongly_connected_components[4] (7 members)]
```

Citation by path makes claims auditable: any reader can re-derive the
claim by querying the JSON. Paraphrasing produces second-hand claims
that drift from the artefact silently. The `--no-timestamp` flag exists
specifically so the cited values stay stable.

### Extending the script

The script's path→layer table lives at the top of the file (search for
`LAYER_TABLE` or the equivalent constant). Adding a new layer or
re-classifying a path requires:

1. Updating the table.
2. Verifying `check` still passes (regression test against the live
   tree).
3. Re-running `dump-edges` and committing the updated artefact if the
   workspace stores one.

Tests live in `tests/unit/scripts/cicd/test_enforce_tier_model_dump_edges.py`
and cover smoke, layer filtering, TYPE_CHECKING/conditional/reexport
tagging, collapse-to-subsystem, SCC detection, determinism, CLI
validation, and empty-input handling. Adding new metadata fields or
output formats means adding tests alongside.

## Offensive Programming Examples

**Examples of good offensive programming:**

```python
# Detect corruption at read time — don't let bad data propagate
try:
    data = json.loads(stored_json)
except json.JSONDecodeError as exc:
    raise AuditIntegrityError(
        f"Corrupt JSON for run {run_id}: database corruption (Tier 1 violation). "
        f"Parse error: {exc}"
    ) from exc

# Validate write-side invariants at construction — reject garbage before it enters the audit trail
@dataclass(frozen=True)
class SecretResolutionInput:
    fingerprint: str
    def __post_init__(self) -> None:
        if len(self.fingerprint) != 64 or not all(c in "0123456789abcdef" for c in self.fingerprint):
            raise ValueError(f"fingerprint must be 64-char lowercase hex, got {self.fingerprint!r}")

# Use atomic guards to prevent TOCTOU races — detect the anomaly, don't just hope it doesn't happen
result = conn.execute(
    update(table).where(table.c.id == id).where(table.c.field.is_(None)).values(field=value)
)
if result.rowcount == 0:
    # Distinguish "not found" from "already set" — different operator actions needed
    existing = conn.execute(select(table.c.field).where(table.c.id == id)).fetchone()
    if existing is not None and existing.field is not None:
        raise AuditIntegrityError(f"Cannot overwrite: field already exists for {id}")
    raise ValueError(f"Record {id} not found")
```

**When to add offensive checks:**

- **Tier 1 read paths**: Data from our own database should be exactly what we expect. If it isn't, crash immediately with context — this is corruption or tampering.
- **Write-side DTOs**: Validate invariants at construction (via `__post_init__`) before data enters the audit trail. Cheaper to reject garbage at the door than to discover it on read.
- **State transitions**: When a method has preconditions (e.g., "contract must not already exist"), assert them with contextual error messages, not silent no-ops.
- **Exception chains**: Always use `from exc` to preserve the original exception chain. `from None` destroys diagnostic information.

### hasattr Alternatives

**`hasattr()` is unconditionally banned.** Everything `hasattr` does can be done more safely:

| Instead of `hasattr` | Use | Why |
|---------------------|-----|-----|
| `hasattr(obj, "attr")` before access | `try: obj.attr` / `except AttributeError:` | `hasattr` swallows all exceptions from `@property` getters, not just missing attributes |
| `hasattr(self, "method_" + name)` for dispatch | Explicit allowset (`frozenset`) + `isinstance` | `hasattr` lets you bypass the gate just by defining a method — no review required |
| `hasattr(e, "field")` on caught exceptions | `isinstance(e, SpecificType)` or direct access | Exception types define their attributes — if the type is in the `except` clause, the attribute exists |
