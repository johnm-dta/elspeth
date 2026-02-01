# Bug Report: Config gate nodes use hardcoded metadata

## Summary
- ~~Orchestrator registers all nodes with `plugin_version="1.0.0"` and default determinism~~
- **UPDATE 2026-01-19:** Plugin nodes (source, transforms, sinks) now correctly extract metadata from plugin instances. Only **config gates** still use hardcoded `plugin_version="1.0.0"` since they are engine-internal expression evaluators without plugin instances.

## Severity
- Severity: minor (downgraded from major - only affects config gates)
- Priority: P2 (downgraded from P1)

## Reporter
- Name or handle: Codex
- Date: 2026-01-15
- Related run/issue ID: N/A

## Environment
- Commit/branch: 5c27593 (local)
- OS: Linux (dev env)
- Python version: 3.11+ (per project)
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if applicable)
- Goal or task prompt: N/A
- Model/version: N/A
- Tooling and permissions (sandbox/approvals): N/A
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: N/A

## Steps To Reproduce
1. Configure a pipeline that includes at least one **config gate** (the `gates:` section).
2. Run the pipeline.
3. Inspect the `nodes` table for the run and locate the config gate node(s).

## Expected Behavior
- Config gate nodes should have clear, intentional metadata:
  - a deterministic determinism value (likely deterministic), and
  - a version scheme that reflects the engine/config gate evaluator version (not a hard-coded placeholder).

## Actual Behavior
- Config gate nodes use hardcoded `plugin_version="1.0.0"` and `determinism=DETERMINISTIC`, independent of any explicit versioning scheme.

## Evidence
- Config gates use hardcoded metadata at node registration time: `src/elspeth/engine/orchestrator.py:443-447`
- Recorder supports determinism and receives values from the orchestrator: `src/elspeth/engine/orchestrator.py:463-472`
- Config gate semantics are engine-internal (no plugin instance exists): `src/elspeth/core/dag.py:339-344`

## Impact
- User-facing impact: Audit records misrepresent plugin versions and determinism, undermining reproducibility and compliance.
- Data integrity / security impact: Reproducibility grade and audit trail integrity are compromised.
- Performance or cost impact: N/A.

## Root Cause Hypothesis
- Orchestrator builds nodes from graph info only and does not read plugin instance metadata (version/determinism/schema hashes).

## Proposed Fix
- Code changes (modules/files):
  - Treat plugin metadata as a first-class compile-time artifact, not a runtime patch:
    - `src/elspeth/plugins/manager.py`: expose a public helper or `PluginSpec.from_plugin(...)` for metadata resolution (version, determinism, schema hashes) so the CLI and engine share a single source of truth.
    - `src/elspeth/core/dag.py`: extend `NodeInfo` to carry `plugin_version`, `determinism`, and a stable `schema_hash` computed from input/output schema hashes; populate these in `ExecutionGraph.from_config(...)` using the plugin registry.
    - `src/elspeth/engine/orchestrator.py`: register nodes from `NodeInfo` (no hard-coded defaults), ensuring the audit trail matches the validated execution graph.
    - `src/elspeth/plugins/protocols.py` and `src/elspeth/plugins/base.py`: add `plugin_version` and `determinism` to `SourceProtocol`/`BaseSource` to avoid source nodes becoming an audit exception.
  - Avoid any runtime fallback or hard-coded values; if metadata is missing, fail fast during graph compilation.
- Config or schema changes: none.
- Tests to add/update:
  - Orchestrator test asserting `nodes.plugin_version`, `nodes.determinism`, and `nodes.schema_hash` match plugin class metadata for source, transform, gate, and sink.
  - Registry test to assert schema hash composition is stable and deterministic.
- Risks or migration steps:
  - Requires wiring a plugin registry into graph compilation; keep this as the single metadata resolution path.
  - Ensure adapters (e.g., `SinkAdapter`) surface plugin identity without masking the underlying plugin class metadata.

## Architectural Deviations
- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/design/architecture.md:249` (audit trail captures plugin instances with metadata).
- Observed divergence: node records use hard-coded version and default determinism.
- Reason (if known): registration uses graph-only info without plugin metadata.
- Alignment plan or decision needed: define authoritative source for plugin metadata during node registration.

## Acceptance Criteria
- Node records contain actual plugin version and determinism for source/transform/gate/sink nodes.

## Tests
- Suggested tests to run: `.venv/bin/python -m pytest tests/engine/test_orchestrator.py -k node`
- New tests required: yes (node metadata persistence).

## Notes / Links
- Related issues/PRs: N/A
- Related design docs: `docs/design/architecture.md`

## Partial Resolution

**Partially fixed in:** 2026-01-19 (verified during triage)

**What's fixed:**
- Plugin nodes (source, transforms, sinks) now correctly extract `plugin_version` and `determinism` from the plugin instance (`src/elspeth/engine/orchestrator.py:451-456`)

**What remains:**
- Config gates still use hardcoded `plugin_version="1.0.0"` (line 445)
- This may be acceptable since config gates are engine-internal expression evaluators, not user-facing plugins

**Recommendation:** Consider closing as "by design" - config gates don't have plugin instances and their behavior is deterministic by nature. Alternatively, define a version scheme for the config gate "pseudo-plugin".

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 6c

**Current Code Analysis:**

The bug is confirmed present in `/home/john/elspeth-rapid/src/elspeth/engine/orchestrator.py` at lines 646-649:

```python
if node_id in config_gate_node_ids:
    # Config gates are deterministic (expression evaluation is deterministic)
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
```

Similarly, aggregation nodes (lines 650-654) and coalesce nodes (lines 655-658) also use hardcoded `plugin_version="1.0.0"` and `determinism=Determinism.DETERMINISTIC`.

**Context and Scope:**

1. **Plugin nodes (source, transforms, sinks):** These were successfully fixed in commit 7144be3 (2026-01-15) and now correctly extract metadata from plugin instances (lines 663-668).

2. **Config gates:** These are engine-internal nodes created from `GateSettings` configuration objects. They:
   - Use `ExpressionParser` (in `src/elspeth/engine/expression_parser.py`) for condition evaluation
   - Have no plugin instances - they're created directly in the DAG from config
   - Are executed by `GateExecutor.execute_config_gate()` (in `src/elspeth/engine/executors.py:475`)
   - Are named with pattern `config_gate:{name}` in the DAG

3. **Aggregations and Coalesce nodes:** These are covered by separate bug P2-2026-01-21-orchestrator-aggregation-metadata-hardcoded.md (verified as STILL VALID). Aggregations DO have plugin instances but aren't included in the `node_to_plugin` mapping.

**Git History:**

- Commit a7d2099 (2026-01-18): When config gates were first integrated, they were intentionally given hardcoded metadata with this design note: "Determinism: Config gates marked DETERMINISTIC (expression eval is pure)"
- Commit 7144be3 (2026-01-15): Fixed plugin node metadata but explicitly excluded config gates
- No subsequent commits have addressed config gate versioning

**Root Cause Confirmed:**

Yes. Config gates are fundamentally different from plugin-based nodes:

- They have no plugin class or instance to extract metadata from
- They're pure configuration artifacts evaluated by the engine's `ExpressionParser`
- Their behavior is controlled by the expression parser implementation, not a plugin

However, the current `plugin_version="1.0.0"` is a placeholder that doesn't reflect:
- The ELSPETH engine version (currently "0.1.0" in `src/elspeth/__init__.py`)
- The `ExpressionParser` version or capabilities
- Changes to gate evaluation semantics over time

**Design Question:**

The core question is whether config gates SHOULD have version tracking:

**Arguments for "by design" (close as not a bug):**
- Config gates are configuration, not plugins
- Expression evaluation is deterministic and simple (just Python AST parsing)
- Changes to expression parser would likely be breaking changes requiring migration anyway
- The audit trail already distinguishes them with `plugin_name=config_gate:{name}`

**Arguments for fixing:**
- ELSPETH's auditability standard: "Every decision must be traceable to source data, configuration, **and code version**"
- If expression parser semantics change (e.g., adding new operators, changing type coercion), old audit records won't show which version was used
- Reproducibility: To replay a run, you need to know which expression evaluator version was used
- Consistency: All other nodes use real versions, only config gates use placeholders

**Recommendation:**

**Keep open as valid P2 issue.** While config gates don't have plugin instances, they still represent code execution (via `ExpressionParser`) that could change over time. The audit trail should reflect which version of the engine/expression parser was used.

**Suggested Fix:**

Use the ELSPETH engine version from `elspeth.__version__` for all engine-internal nodes (config gates, coalesce):

```python
from elspeth import __version__ as ENGINE_VERSION

if node_id in config_gate_node_ids:
    plugin_version = f"engine:{ENGINE_VERSION}"
    determinism = Determinism.DETERMINISTIC
elif node_id in coalesce_node_ids:
    plugin_version = f"engine:{ENGINE_VERSION}"
    determinism = Determinism.DETERMINISTIC
```

This clearly indicates these are engine operations (not plugins) while still providing version traceability for audit purposes.

**Note:** Aggregations should be handled separately per P2-2026-01-21-orchestrator-aggregation-metadata-hardcoded.md as they DO have plugin instances.

---

## FIX APPLIED: 2026-01-30

**Status:** FIXED

**Fixed By:** Claude Code

**Changes Made:**

1. **`src/elspeth/engine/orchestrator.py`**:
   - Added import: `from elspeth import __version__ as ENGINE_VERSION`
   - Changed config gate `plugin_version` from `"1.0.0"` to `f"engine:{ENGINE_VERSION}"`
   - Changed coalesce node `plugin_version` from `"1.0.0"` to `f"engine:{ENGINE_VERSION}"`

2. **`tests/engine/test_orchestrator_audit.py`**:
   - Added `test_config_gate_node_uses_engine_version()` - verifies config gates use `engine:0.1.0` format
   - Added `test_coalesce_node_uses_engine_version()` - verifies coalesce nodes use `engine:0.1.0` format

**Verification:**
- Both new tests pass
- All 12 tests in `test_orchestrator_audit.py` pass
- All fork/coalesce property tests pass (13/13)
- mypy type check passes
- ruff lint passes

**Result:** Engine-internal nodes (config gates and coalesce) now use `engine:{VERSION}` format (e.g., `engine:0.1.0`) instead of hardcoded `"1.0.0"`. This provides audit trail version traceability while clearly distinguishing engine-internal nodes from user plugins.
