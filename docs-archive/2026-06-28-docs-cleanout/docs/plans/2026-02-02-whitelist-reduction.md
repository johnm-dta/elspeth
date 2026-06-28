# Tier Model Whitelist Reduction Plan

**Status:** IN PROGRESS
**Created:** 2026-02-02
**Owner:** Development Team

## Problem Statement

The tier model enforcement whitelist (`config/cicd/enforce_tier_model.yaml`) has grown to **547 entries** across 2798 lines. This whitelist was intended to document legitimate exceptions to ELSPETH's defensive programming prohibition, but has become a "make CI pass" escape hatch that conceals actual bugs.

### Current State

| Rule | Pattern | Count | Primary Concern |
|------|---------|-------|-----------------|
| R5 | `isinstance()` | 250 | Mostly legitimate AST/type work |
| R1 | `dict.get()` | 167 | **Many are bug-hiding on internal state** |
| R6 | Silent except | 48 | Mixed - need review |
| R4 | Broad except | 47 | Mixed - need review |
| R9 | `dict.pop(default)` | 11 | Likely bug-hiding |
| R3 | `hasattr()` | 9 | Likely bug-hiding |
| R2 | `getattr(default)` | 9 | Likely bug-hiding |
| R8 | `setdefault()` | 4 | Mixed |
| R7 | `contextlib.suppress()` | 2 | Likely bug-hiding |

### Expiration Cliff

- **488 entries** expire on `2026-05-02` (3 months)
- **59 entries** are permanent (`expires: null`)

This creates a maintenance bomb where CI will fail in 3 months unless action is taken.

### Root Cause

Entries were added with boilerplate justifications:
```yaml
reason: Optional field access at trust boundary
safety: Code reviewed - legitimate pattern
```

These provide no insight into WHY the pattern is legitimate, suggesting rubber-stamp approval rather than thoughtful review.

## Goals

1. **Reduce whitelist by ~50%** by fixing actual bug-hiding patterns
2. **Improve whitelist quality** with meaningful justifications
3. **Implement per-file whitelisting** for legitimate trust boundaries
4. **Make AST/type work permanent** with proper documentation

## Non-Goals

- Changing the enforcement tool itself (R1-R9 rules are sound)
- Removing all defensive patterns (some ARE at trust boundaries)
- Achieving zero whitelist entries (unrealistic)

---

## Phase 1: Fix Bug-Hiding Patterns (~120 entries)

**Target:** Core engine code where `.get()` accesses internal state

### 1.1 engine/executors.py (26 R1 entries) ✅ COMPLETE

**Status:** COMPLETE - 2026-02-02

**Problem:** Internal state dictionaries accessed with `.get()`:
```python
self._batch_ids.get(node_id)           # Bug if node_id missing
self._trigger_evaluators.get(node_id)  # Bug if node_id missing
self._buffers.get(node_id, [])         # Silent empty list on bug
self._edge_map.get((typed_node_id, dest))  # Silent None on bug
```

**What Was Done:**
1. Added validation against `aggregation_settings` before `.get()` access
2. Changed internal state access from `.get()` to direct `[]` where appropriate
3. Updated checkpoint restore to use direct `[]` for required v1.1 format fields
4. Fixed tests to include all required fields in checkpoint format

**Pattern Applied:**
```python
# BEFORE (bug-hiding):
return self._buffers.get(node_id, [])  # Invalid node → [] (hides bug)

# AFTER (explicit validation + legitimate default):
if node_id not in self._aggregation_settings:
    raise OrchestrationInvariantError(f"...")
return self._buffers.get(node_id, [])  # Now only handles "valid node, no rows yet"
```

**Results:**
- Whitelist: 547 → 528 (-19 entries)
- All 787 engine tests pass
- Code reviewed for tier model compliance

### 1.2 engine/orchestrator.py (12 R1 entries)

**Problem:** Same pattern as executors - internal state access with `.get()`

**Approach:** Same as 1.1

### 1.3 core/landscape/recorder.py (8 entries)

**Problem:** Audit trail code using defensive patterns. Per CLAUDE.md:
> Tier 1 (Audit Database): Full trust - crash on any anomaly

**Approach:** Crash on unexpected state rather than silent defaults

### 1.4 core/landscape/formatters.py (11 entries)

**Problem:** Formatters reading OUR audit data defensively

**Approach:** Same as 1.3 - trust our data, crash on corruption

### 1.5 testing/chaosllm/* (19 entries)

**Problem:** Even test utilities hide bugs with defensive patterns

**Approach:** Test code should fail fast to reveal issues during development

---

## Phase 2: Per-File/Category Whitelisting (~200 entries)

**Target:** Legitimate trust boundaries that should be whitelisted at file level

### 2.1 Enhance Whitelist Format

Add support for per-file rules:
```yaml
per_file_rules:
  # Tier 3: MCP arguments from external clients
  - pattern: "mcp/server.py"
    rules: [R1]  # dict.get() on MCP arguments is legitimate
    reason: "MCP tool arguments are external client data (Tier 3)"
    expires: null

  # Tier 3: LLM API responses
  - pattern: "plugins/llm/*"
    rules: [R1, R4, R6]
    reason: "LLM responses are external API data (Tier 3)"
    expires: null

  # Tier 3: Azure cloud services
  - pattern: "plugins/azure/*"
    rules: [R1, R4, R6]
    reason: "Azure responses are external API data (Tier 3)"
    expires: null

  # Tier 3: Telemetry exporters (external services)
  - pattern: "telemetry/exporters/*"
    rules: [R1, R4, R6]
    reason: "Telemetry services are external (Tier 3)"
    expires: null
```

### 2.2 Files to Whitelist

| Pattern | Rules | Entries Removed | Justification |
|---------|-------|-----------------|---------------|
| `mcp/server.py` | R1 | ~24 | MCP arguments are Tier 3 |
| `plugins/llm/*` | R1, R4, R6 | ~60 | LLM APIs are Tier 3 |
| `plugins/azure/*` | R1, R4, R6 | ~15 | Azure APIs are Tier 3 |
| `telemetry/exporters/*` | R1, R4, R6 | ~50 | External services |
| `plugins/sources/*` | R1, R4, R6 | ~10 | External data sources |

---

## Phase 3: Make AST/Type Work Permanent (~180 entries)

**Target:** R5 (isinstance) entries that are fundamental to Python type handling

### 3.1 Permanent Whitelist Categories

| File | R5 Count | Reason |
|------|----------|--------|
| `engine/expression_parser.py` | 24 | AST node type discrimination |
| `core/templates.py` | 12 | Jinja2 AST walking |
| `core/canonical.py` | 13 | JSON type normalization |
| `core/config.py` | 31 | Config structure validation |
| `contracts/schema.py` | 7 | Schema type validation |
| `contracts/config/runtime.py` | 9 | Runtime config validation |

### 3.2 Implementation

Move to permanent with meaningful documentation:
```yaml
- key: engine/expression_parser.py:R5:...
  owner: architecture
  reason: "AST parsing requires isinstance() to discriminate ast.Call from ast.Name etc."
  safety: "Fundamental to Python AST - no alternative exists"
  expires: null  # Permanent
```

---

## Phase 4: Review and Clean Up (~50 entries)

### 4.1 TUI Code (18 R1 entries)

`tui/widgets/node_detail.py` has mixed patterns:
- Some `.get()` for legitimately optional audit fields
- Some for presentation convenience ("N/A" display)

**Approach:**
- Optional schema fields → keep whitelist with better docs
- Presentation convenience → consider if crash would be better

### 4.2 Exception Handling (R4, R6)

Review broad exception handling (R4) and silent exceptions (R6):
- External service boundaries → legitimate
- Internal error handling → likely bug-hiding

---

## Success Criteria

| Metric | Before | Target |
|--------|--------|--------|
| Total whitelist entries | 547 | <300 |
| R1 (dict.get) entries | 167 | <50 |
| Boilerplate justifications | ~90% | <10% |
| Per-file whitelists | 0 | 5-10 |
| Entries with `expires: null` | 59 | ~200 |

## Implementation Order

1. **Phase 1.1:** Fix `engine/executors.py` (highest impact)
2. **Phase 1.2:** Fix `engine/orchestrator.py`
3. **Phase 2.1:** Implement per-file whitelist support
4. **Phase 2.2:** Add per-file rules for plugins
5. **Phase 3:** Make AST/type entries permanent
6. **Phase 1.3-1.5:** Fix remaining bug-hiding patterns
7. **Phase 4:** Final review and cleanup

## Risks

1. **Breaking changes:** Removing `.get()` may expose latent bugs
   - Mitigation: Good test coverage, fix bugs as discovered

2. **False positives:** Some patterns may be legitimate but uncategorized
   - Mitigation: Review carefully, keep whitelist if justified

3. **Time investment:** ~120 patterns to review and fix
   - Mitigation: Prioritize by impact (engine > plugins > TUI)

---

## Appendix: Files by Entry Count

```
36  engine/executors.py
36  core/config.py
29  mcp/server.py
27  engine/orchestrator.py
24  engine/expression_parser.py
21  tui/widgets/node_detail.py
20  plugins/llm/openrouter_multi_query.py
19  plugins/llm/azure_batch.py
17  telemetry/exporters/datadog.py
17  telemetry/exporters/azure_monitor.py
15  plugins/llm/azure_multi_query.py
14  core/templates.py
14  core/canonical.py
13  testing/chaosllm/response_generator.py
13  telemetry/exporters/otlp.py
11  telemetry/manager.py
11  telemetry/exporters/console.py
11  core/landscape/formatters.py
11  contracts/schema.py
10  plugins/azure/blob_source.py
```
