# Plugin-Declared Transform Semantics — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `web_scrape → line_explode` framing check with a plugin-owned semantic contract layer where producers declare what they emit, consumers declare what they require, and a generic validator compares the declarations.

**Architecture:** A new L0 contracts module (`contracts/plugin_semantics.py`) defines `FieldSemanticFacts`, `FieldSemanticRequirement`, and `SemanticEdgeContract` types. `BaseTransform` gains three optional declaration methods. A shared `ProducerResolver` (extracted as Phase 0) provides one walk-back implementation that both the existing schema validator and the new semantic validator consume. The generic `validate_semantic_contracts(...)` lives next to `_check_schema_contracts`, runs inside `CompositionState.validate()`, and is wired into `/validate`, `/execute`, MCP, and `ToolResult`. The old hardcoded validator is deleted in Phase 6 after the new system has parity coverage.

**Tech Stack:** Python 3.12+, pluggy 1.6+, Pydantic 2.12+, frozen dataclasses with `freeze_fields(...)` from `contracts/freeze.py`, pytest, hypothesis (property-based tests).

**Supersedes:** `docs/plans/2026-04-27-plugin-declared-transform-semantics.md` (v1). v1 review identified eight blocking issues (B1–B8). This plan resolves them by:

| v1 Blocker | v2 Resolution |
|------------|---------------|
| B1 `PluginAssistance()` no-arg default is `TypeError` | `get_agent_assistance` returns `PluginAssistance \| None` (default `None`), serializers handle `None`. See Phase 1 Task 1.4. |
| B2 `PluginAssistanceExample` missing freeze guard | Added `__post_init__` with conditional `freeze_fields(...)`. Phase 1 Task 1.4. |
| B3 No shared edge-resolution primitive | **Phase 0 Task 0.2** extracts `ProducerResolver` consumed by both validators. |
| B4 Pass-through semantic propagation under-specified | Phase 1 explicitly degrades pass-through to `UNKNOWN` (no `transform_field_semantics` API). Documented in Phase 3 Task 3.1. |
| B5 Secret-leakage test absent | Phase 3 Task 3.5 adds sentinel-string leak test across all five surfaces. |
| B6 WARN-path audit gap | **Eliminated by design**: Phase 1 starts at `unknown_policy=FAIL`. No WARN-path decision exists to record. See "Design decision: FAIL on unknown" below. |
| B7 `/validate` ↔ `/execute` error-surface asymmetry | Phase 4 Task 4.5 introduces `SemanticContractViolationError` carrying structured records; both surfaces consume the shared helper and render structure. |
| B8 Wardline regression surface unconfirmed | **Confirmed web-only**: `data/wardline_line_export_pipeline.yaml:29` already has `text_separator: "\n"`; CLI has never had a framing check. Phase 1 web-only is the right surface. |

**v2 review findings (post-write, addressed in this document):**

| Finding | Resolution |
|---------|------------|
| Fixtures omit required `schema` and `on_error` — validator's tolerant probe path makes tests vacuous | Every fixture in Phases 2/3 includes `schema` and `on_error="errors"`. Phase 3 Task 3.1 validator change: probe-error tolerance applies ONLY to producer construction. Consumer probe failure is a test bug and propagates. |
| `ProducerResolver.walk_to_real_producer` would crash on source-fed consumers via `_node_by_id["source"]` | Resolver short-circuits when producer is source. Semantic validator skips source-fed edges entirely (matching the new "source → transform out of scope" decision below). Phase 0 Task 0.2 adds explicit source-walk tests. |
| `routes.py` catches bare `ValueError` and renders only `detail=str(exc)` — structured payload dropped at the HTTP boundary | Phase 4 Task 4.5b adds a dedicated `SemanticContractViolationError` handler returning a 422 with structured `semantic_contracts` payload. |
| `ValidationResult` Pydantic schema is `extra="forbid"` and lacks `semantic_contracts` field — adding it in `validation.py` alone won't reach the wire | Phase 4 Task 4.3b extends the Pydantic schema with the field + nested response model BEFORE the validator populates it. |
| Assistance lookup is registry-order dependent (loops every transform class) | `SemanticEdgeContract` carries `consumer_plugin: str` and `producer_plugin: str \| None`. `_assistance_suggestion_for` calls the named consumer plugin directly. Updated in Phase 1 Task 1.3 (contract type) and Phase 4 Task 4.3 (lookup). |
| `unknown_policy=FAIL` would break source → transform pipelines that work today (no `BaseSource` semantic API in Phase 1) | Source → transform edges are out of scope for Phase 1. Validator skips them. Phase 0 Task 0.3 audits in-tree pipelines for `line_explode` usage to confirm only the Wardline transform → transform chain is affected. |
| Plan referenced `WebScrape` — actual class is `WebScrapeTransform` | All Phase 2/3/6 references corrected to `WebScrapeTransform`. |
| Task 4.7 only modified preview, not `ToolResult.to_dict()` itself | Task 4.7 expanded to include `ToolResult.to_dict()` validation payload at `tools.py:114`. |
| Phase 7 doc search missed skill files in `src/elspeth/web/composer/skills/pipeline_composer.md` and `.claude/skills/pipeline-composer/SKILL.md` | Phase 7 Task 7.1 grep widened to include all known skill paths. |

**Design decision: start at FAIL, not WARN.** v1 proposed `unknown_policy=WARN` then "tighten to FAIL when coverage improves." Per the project's no-legacy-code policy and the no-users-yet posture, in-codebase migration plans are forbidden. Phase 1 starts at FAIL: every producer that semantically feeds a declared consumer must itself declare semantics. Today only `web_scrape → line_explode` exists, so the requirement is trivially satisfiable. New consumer declarations in future PRs must come with corresponding producer declarations in the same PR.

**Design decision: pass-through degrades to UNKNOWN in Phase 1.** No `transform_field_semantics()` API is added in Phase 1. A pass-through transform between a declared producer and a declared consumer breaks the chain — the consumer sees `outcome=UNKNOWN`. Combined with `unknown_policy=FAIL`, this means pass-through transforms in semantic chains must declare semantics OR the chain must be edited not to include pass-through transforms in semantic-critical edges. This is intentional — it keeps Phase 1 small and forces the next iteration to design a proper propagation API based on real cases rather than speculation.

**Design decision: source → transform edges are out of scope for Phase 1.** `BaseSource` does not expose `output_semantics()` in Phase 1 — extending it is a separate plan. The semantic validator MUST skip any edge whose effective producer is the source (`producer_id == "source"`). Without this skip, every `csv → line_explode` pipeline that works today would hard-fail under `unknown_policy=FAIL`, which contradicts the no-users-yet posture (we still don't break working in-tree pipelines without cause). This also incidentally prevents the resolver from indexing `_node_by_id["source"]`, which has no entry. The Phase 0 in-tree audit (Task 0.3) confirms which existing pipelines this affects, and the audit's findings determine whether `BaseSource` semantic declarations need to land in this plan or the next.

**Out of scope (intentional, do not expand):**
- Runtime graph-level semantic validation (`ExecutionGraph.validate_semantic_compatibility`). Web `/validate` and `/execute` are sufficient for the Wardline regression because that path is web-only. CLI/non-web YAML semantic validation is a separate plan.
- Source → transform semantic edges. `BaseSource` semantic API is a separate plan.
- Semantic propagation through pass-through transforms.
- Telemetry metrics for probe latency or UNKNOWN counts. (Single-line Phase 6 note acknowledges deferral; no implementation.)
- Catalog `PluginSummary.semantic_capabilities` summary field. (`get_plugin_assistance` discovery tool covers the need.)

**File structure:**

| File | Purpose | Action |
|------|---------|--------|
| `src/elspeth/contracts/plugin_semantics.py` | New L0 contract module: enums, fact/requirement/edge dataclasses, comparator function | CREATE |
| `src/elspeth/contracts/plugin_assistance.py` | New L0 module: `PluginAssistance`, `PluginAssistanceExample` | CREATE |
| `src/elspeth/web/composer/_producer_resolver.py` | New L3 module: extracted producer-map + walk-back primitive consumed by both validators | CREATE |
| `src/elspeth/web/composer/_semantic_validator.py` | New L3 module: `validate_semantic_contracts(...)` | CREATE |
| `src/elspeth/web/composer/state.py` | Refactor `_check_schema_contracts` to use `ProducerResolver`; add `ValidationSummary.semantic_contracts`; integrate semantic validator | MODIFY |
| `src/elspeth/web/execution/validation.py` | Replace `_CHECK_TRANSFORM_FRAMING` with `_CHECK_SEMANTIC_CONTRACTS`, call shared helper, populate response semantic_contracts list | MODIFY |
| `src/elspeth/web/execution/schemas.py` | Add `SemanticEdgeContract` Pydantic response model + `semantic_contracts` field on `ValidationResult` (extra="forbid" requires explicit declaration) | MODIFY |
| `src/elspeth/web/execution/routes.py` | Add `SemanticContractViolationError` handler that builds a 422 response with structured `semantic_contracts` payload — currently the route catches bare `ValueError` at line 336 and renders only `detail=str(exc)` | MODIFY |
| `src/elspeth/web/execution/service.py` | Replace direct framing call with shared helper that raises `SemanticContractViolationError` | MODIFY |
| `src/elspeth/web/execution/errors.py` | New: `SemanticContractViolationError` exception with structured payload | CREATE |
| `src/elspeth/web/composer/tools.py` | Serialize `semantic_contracts` in `_execute_preview_pipeline` | MODIFY |
| `src/elspeth/composer_mcp/server.py` | Add `_SemanticEdgeContractPayload` TypedDict; serialize in `_validation_to_dict` | MODIFY |
| `src/elspeth/web/frontend/src/types/index.ts` | Add `semantic_contracts` to `ValidationResult` interface | MODIFY |
| `src/elspeth/plugins/infrastructure/base.py` | Add three default methods on `BaseTransform` (return empty/None) | MODIFY |
| `src/elspeth/plugins/transforms/web_scrape.py` | Implement `output_semantics()`, `get_agent_assistance()` | MODIFY |
| `src/elspeth/plugins/transforms/line_explode.py` | Implement `input_semantic_requirements()`, `get_agent_assistance()` | MODIFY |
| `tests/unit/contracts/test_plugin_semantics.py` | New: contract types, deep-freeze, comparator, layer-purity | CREATE |
| `tests/unit/contracts/test_plugin_assistance.py` | New: assistance dataclasses + freeze | CREATE |
| `tests/unit/web/composer/test_producer_resolver.py` | New: shared resolver behaviour | CREATE |
| `tests/unit/web/composer/test_semantic_validator.py` | New: validator algorithm + property-based tests + parity test | CREATE |
| `tests/unit/web/composer/test_state.py` | Update: replace framing tests with semantic equivalents + Wardline regression pin | MODIFY |
| `tests/unit/web/execution/test_validation.py` | Update: rename check, assert structured records | MODIFY |
| `tests/unit/web/execution/test_service.py` | Update: assert `SemanticContractViolationError` shape | MODIFY |
| `tests/unit/plugins/transforms/test_web_scrape.py` | Add: declaration tests + secret-leak sentinel test | MODIFY |
| `tests/unit/plugins/transforms/test_line_explode.py` | Add: declaration tests + secret-leak sentinel test | MODIFY |
| `tests/unit/composer_mcp/test_server.py` | Add: MCP `semantic_contracts` payload assertions | MODIFY (or create section) |

**Plugin class names — verified:**

- `WebScrapeTransform` at `src/elspeth/plugins/transforms/web_scrape.py:221` (registered as `name = "web_scrape"`). The Phase 2.2 task originally said "WebScrape" — the actual class is `WebScrapeTransform`. Use the verified name in all imports and tests.
- `LineExplode` at `src/elspeth/plugins/transforms/line_explode.py:125` (registered as `name = "line_explode"`).
- `WebScrapeConfig` and `LineExplodeConfig` are TransformDataConfig subclasses, which means `schema_config` (alias `schema`) is REQUIRED at construction (`src/elspeth/plugins/infrastructure/config_base.py:274`). Every test fixture that constructs these via `PluginManager.create_transform(...)` MUST supply a `schema` field or construction raises `PluginConfigError` and the validator's tolerant probe-error path silently skips the case.

**`CompositionState.validate()` requires every transform to have non-empty `on_error`** (`src/elspeth/web/composer/state.py:1438–1448`). Every fixture transform must set `on_error="errors"` (or another valid sink/connection name) — otherwise `state.validate()` short-circuits with a transform-config error before semantic validation runs, making semantic tests vacuous.

**Verified file/symbol references** (line numbers verified against current HEAD; v1 had drift):

- `validate_transform_framing_contracts` defined at `state.py:351`; called at `state.py:1509`, `validation.py:274`, `service.py:325` (three sites — v1 missed `state.py:1509`).
- `_check_schema_contracts` defined at `state.py:543`.
- `_walk_to_real_producer` defined twice: `state.py:391` (inside `validate_transform_framing_contracts`) and `state.py:784` (inside `_check_schema_contracts`).
- `_effective_producer_vote` at `state.py:838`; `_connection_propagation_vote` at `state.py:957`.
- `WebScrapeConfig.content_field`, `format`, `text_separator` at `web_scrape.py:118`.
- `web_scrape` instance fields stored at `web_scrape.py:308–309`; `extract_content(...)` call at `web_scrape.py:481`; output emit at `web_scrape.py:520`.
- `LineExplodeConfig.source_field` at `line_explode.py:33`; instance store at `line_explode.py:147`; `splitlines()` at `line_explode.py:183`.
- `ValidationSummary` at `state.py:281` (`edge_contracts` field at `state.py:295`).
- `_ValidationPayload` TypedDict at `composer_mcp/server.py:64`; `_EdgeContractPayload` at `:51`; `_validation_to_dict` at `:282`.
- `_execute_preview_pipeline` defined at `tools.py:3326`; `edge_contracts` serialized at `tools.py:3345`.
- `BaseTransform` at `plugins/infrastructure/base.py:57`; `passes_through_input` ClassVar present.
- Frontend `ValidationResult` interface at `frontend/src/types/index.ts:264-270` (no `semantic_contracts` field).
- Wardline YAML at `data/wardline_line_export_pipeline.yaml:29` already has `text_separator: "\n"` (regression file is fixed; we will add a test fixture that reproduces the BROKEN version to pin the regression).

---

## Phase 0 — Pre-conditions

These tasks land before any new validator code. They confirm scope and remove structural risk.

### Task 0.1: Confirm Wardline regression surface

**Files:**
- Read: `data/wardline_line_export_pipeline.yaml`
- Read: `src/elspeth/cli_helpers.py`, `src/elspeth/cli/run.py` (or wherever `elspeth run` lives)
- No code changes — investigative task that produces a one-paragraph note in the plan execution log.

**Goal:** Confirm the original Wardline regression manifested through the web composer path (`/validate` or `/execute`), not the CLI. If CLI, this plan's surface choice is wrong and Phase 1 must be revisited.

- [ ] **Step 1: Inspect the Wardline YAML for the broken version in git history**

```bash
git log --all --oneline -- data/wardline_line_export_pipeline.yaml | head -10
git show <commit-where-text-separator-was-added>:data/wardline_line_export_pipeline.yaml | grep -n text_separator
```

Expected: Identify the commit where `text_separator: "\n"` was added. The version before that commit is the broken version that triggered the regression.

- [ ] **Step 2: Search for any CLI-side framing check**

```bash
grep -rn 'framing\|line_explode.*web_scrape\|web_scrape.*line_explode' src/elspeth/cli_helpers.py src/elspeth/core/ src/elspeth/engine/ 2>/dev/null
```

Expected: No matches. The framing check exists only at `state.py:351`, `validation.py:274`, `service.py:325`, `state.py:1509` — all web composer surfaces.

- [ ] **Step 3: Record finding**

Write a one-paragraph note in the execution journal (or PR description): "Wardline regression surface: web composer only. CLI has no framing or semantic validator today; Phase 1 covering `CompositionState.validate()` + `/validate` + `/execute` + MCP is the correct surface. CLI parity is intentionally deferred to a future plan."

If the finding contradicts this — i.e., a CLI framing check is found — STOP and revise the plan before continuing.

- [ ] **Step 4: Commit the journal note**

```bash
git add docs/plans/2026-04-27-plugin-declared-transform-semantics-v2.md  # if you appended a note
git commit -m "chore: confirm Wardline regression surface is web-only (Phase 0)"
```

---

### Task 0.2: Extract `ProducerResolver` as a shared primitive

**Files:**
- Create: `src/elspeth/web/composer/_producer_resolver.py`
- Modify: `src/elspeth/web/composer/state.py:543` (`_check_schema_contracts` constructs the resolver instead of building producer_map inline)
- Test: `tests/unit/web/composer/test_producer_resolver.py`

**Goal:** Eliminate the duplicated walk-back logic. `validate_transform_framing_contracts` (state.py:351) already has its own `_walk_to_real_producer` at state.py:391; `_check_schema_contracts` has another at state.py:784. The new semantic validator MUST NOT add a third. Both existing callers must consume the new resolver before the semantic validator is written.

**Scope discipline:** Extract ONLY producer-map building and connection-keyed walk-back (gates → real producer, with cycle detection and duplicate-connection handling). DO NOT extract pass-through propagation (`_effective_producer_vote`, `_connection_propagation_vote`, `_intersect_predecessor_guarantees`) — those stay schema-specific. Semantic Phase 1 does not propagate through pass-through.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/web/composer/test_producer_resolver.py`:

```python
"""Tests for the shared producer-map resolver primitive."""

from __future__ import annotations

import pytest

from elspeth.web.composer._producer_resolver import ProducerResolver, ProducerEntry
from elspeth.web.composer.state import NodeSpec, SourceSpec


def _node(node_id: str, *, plugin: str | None, node_type: str = "transform",
          input: str = "", on_success: str | None = None,
          on_error: str | None = None, options: dict | None = None,
          routes: dict[str, str] | None = None,
          fork_to: tuple[str, ...] | None = None) -> NodeSpec:
    return NodeSpec(
        id=node_id, node_type=node_type, plugin=plugin, input=input,
        on_success=on_success, on_error=on_error, options=options or {},
        routes=routes, fork_to=fork_to,
    )


class TestProducerResolverBuild:
    def test_source_registers_as_producer_for_on_success(self):
        source = SourceSpec(plugin="csv", on_success="step1", options={}, on_validation_failure="discard")
        nodes = (_node("step1", plugin="t", input="step1", on_success="sink"),)
        resolver = ProducerResolver.build(source=source, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.find_producer_for("step1")
        assert producer is not None
        assert producer.producer_id == "source"
        assert producer.plugin_name == "csv"

    def test_node_on_success_registers_producer(self):
        nodes = (_node("a", plugin="p1", input="src_out", on_success="b_in"),
                 _node("b", plugin="p2", input="b_in", on_success="sink"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.find_producer_for("b_in")
        assert producer is not None
        assert producer.producer_id == "a"
        assert producer.plugin_name == "p1"

    def test_duplicate_producer_for_connection_is_recorded(self):
        nodes = (_node("a", plugin="p1", input="src", on_success="dup"),
                 _node("b", plugin="p2", input="src", on_success="dup"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert "dup" in resolver.duplicate_connections
        assert resolver.find_producer_for("dup") is None  # ambiguous

    def test_routes_register_producers(self):
        nodes = (_node("g", plugin="gate1", node_type="gate", input="src",
                       routes={"yes": "yes_out", "no": "no_out"}),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        for connection in ("yes_out", "no_out"):
            producer = resolver.find_producer_for(connection)
            assert producer is not None and producer.producer_id == "g"

    def test_fork_to_registers_producers(self):
        nodes = (_node("g", plugin="fork1", node_type="gate", input="src",
                       fork_to=("a", "b")),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        for branch in ("a", "b"):
            assert resolver.find_producer_for(branch) is not None

    def test_coalesce_without_on_success_publishes_under_own_id(self):
        nodes = (_node("c", plugin=None, node_type="coalesce", input="branches"),)
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        producer = resolver.find_producer_for("c")
        assert producer is not None and producer.producer_id == "c"


class TestProducerResolverWalkBack:
    def test_walk_through_gate_returns_real_producer(self):
        nodes = (
            _node("scrape", plugin="web_scrape", input="src_out", on_success="gate_in"),
            _node("g", plugin="gate1", node_type="gate", input="gate_in", on_success="explode_in"),
            _node("explode", plugin="line_explode", input="explode_in", on_success="sink"),
        )
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset({"sink"}))

        producer = resolver.walk_to_real_producer("explode_in")
        assert producer is not None
        assert producer.producer_id == "scrape"
        assert producer.plugin_name == "web_scrape"

    def test_walk_returns_none_on_routing_loop(self):
        nodes = (
            _node("g1", plugin=None, node_type="gate", input="loop_b", on_success="loop_a"),
            _node("g2", plugin=None, node_type="gate", input="loop_a", on_success="loop_b"),
        )
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("loop_a") is None

    def test_walk_returns_none_when_connection_is_duplicate(self):
        nodes = (_node("a", plugin="p1", input="src", on_success="dup"),
                 _node("b", plugin="p2", input="src", on_success="dup"))
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("dup") is None

    def test_walk_returns_none_when_connection_unknown(self):
        nodes: tuple[NodeSpec, ...] = ()
        resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

        assert resolver.walk_to_real_producer("nope") is None

    def test_walk_returns_source_producer_without_node_lookup(self):
        # Reviewer-found bug: walk_to_real_producer must NOT index
        # _node_by_id["source"]. Source producers must short-circuit
        # before any node-table lookup.
        source = SourceSpec(
            plugin="csv", on_success="step1",
            options={
                "path": "x.csv",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            on_validation_failure="quarantine",
        )
        resolver = ProducerResolver.build(
            source=source, nodes=(), sink_names=frozenset(),
        )
        producer = resolver.walk_to_real_producer("step1")
        assert producer is not None
        assert producer.producer_id == "source"
        assert producer.plugin_name == "csv"

    def test_walk_through_gate_to_source(self):
        source = SourceSpec(
            plugin="csv", on_success="gate_in",
            options={
                "path": "x.csv",
                "schema": {"mode": "fixed", "fields": ["url: str"]},
            },
            on_validation_failure="quarantine",
        )
        nodes = (_node("g", plugin="gate1", node_type="gate",
                       input="gate_in", on_success="explode_in"),)
        resolver = ProducerResolver.build(
            source=source, nodes=nodes, sink_names=frozenset(),
        )
        producer = resolver.walk_to_real_producer("explode_in")
        assert producer is not None
        assert producer.producer_id == "source"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_producer_resolver.py -v`
Expected: ImportError — module does not exist.

- [ ] **Step 3: Create the resolver module**

Create `src/elspeth/web/composer/_producer_resolver.py`:

```python
"""Shared producer-map and walk-back primitive.

Both the schema-contract validator and the semantic-contract validator
need to: (1) build a map from connection name to producer node, (2) walk
back through structural gates to find the real producer of a connection.
This module provides the single implementation. Pass-through propagation
is intentionally NOT included — that remains schema-specific in
state.py because semantic validation does not propagate through
pass-through transforms in Phase 1.

Layer: L3 (web composer application code). Imports state types from
the same layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.state import NodeSpec, SourceSpec


@dataclass(frozen=True, slots=True)
class ProducerEntry:
    """A producer registered against one or more connection names.

    options is the producer's raw options Mapping — NOT deep-frozen here
    because state.py already deep-freezes node options in __post_init__.
    """

    producer_id: str
    plugin_name: str | None
    options: Mapping[str, Any]


class ProducerResolver:
    """Builds and queries the connection → producer map for a composition.

    Construction is via ``build(...)`` rather than ``__init__`` so the
    primitive can compute and report duplicates as part of its result.
    Once built, ``find_producer_for`` and ``walk_to_real_producer`` are
    pure functions of the resolver state.

    NOT a frozen dataclass: holds derived dicts/sets that are
    construction-time fixed but expensive to deep-freeze. Treat as
    effectively immutable — do not mutate the public attributes.
    """

    __slots__ = (
        "_producer_map",
        "_node_by_id",
        "duplicate_connections",
    )

    def __init__(
        self,
        producer_map: dict[str, ProducerEntry],
        node_by_id: dict[str, NodeSpec],
        duplicate_connections: frozenset[str],
    ) -> None:
        self._producer_map = producer_map
        self._node_by_id = node_by_id
        self.duplicate_connections = duplicate_connections

    @classmethod
    def build(
        cls,
        *,
        source: SourceSpec | None,
        nodes: tuple[NodeSpec, ...],
        sink_names: frozenset[str],
    ) -> ProducerResolver:
        producer_map: dict[str, ProducerEntry] = {}
        duplicates: set[str] = set()
        node_by_id = {node.id: node for node in nodes}

        def register(connection_name: str | None, entry: ProducerEntry) -> None:
            if connection_name is None or connection_name == "discard":
                return
            if connection_name in sink_names:
                # Direct-to-sink edges aren't producers for downstream
                # walk-back; schema-contract code handles them separately.
                return
            if connection_name in producer_map:
                duplicates.add(connection_name)
                return
            producer_map[connection_name] = entry

        if source is not None:
            register(
                source.on_success,
                ProducerEntry(
                    producer_id="source",
                    plugin_name=source.plugin,
                    options=source.options,
                ),
            )

        for node in nodes:
            entry = ProducerEntry(
                producer_id=node.id,
                plugin_name=node.plugin,
                options=node.options,
            )
            if node.node_type == "coalesce" and node.on_success is None:
                register(node.id, entry)
            else:
                register(node.on_success, entry)
            register(node.on_error, entry)
            if node.routes is not None:
                for target in node.routes.values():
                    register(target, entry)
            if node.fork_to is not None:
                for target in node.fork_to:
                    register(target, entry)

        return cls(producer_map, node_by_id, frozenset(duplicates))

    def find_producer_for(self, connection_name: str) -> ProducerEntry | None:
        """Return the immediate producer for a connection, or None.

        Returns None for: unknown connection, duplicate (ambiguous)
        connection, or a connection produced only by a direct-to-sink edge.
        """
        if connection_name in self.duplicate_connections:
            return None
        return self._producer_map.get(connection_name)

    def walk_to_real_producer(self, connection_name: str) -> ProducerEntry | None:
        """Walk back through structural gates to the true producer.

        Returns None on: unknown connection, duplicate connection,
        routing loop, or any structural node that semantic walk-back
        does not traverse (currently: coalesce — its branch semantics
        are handled by callers that need them).

        Source producers (producer_id == "source") return immediately
        WITHOUT a node-table lookup. The source is registered in
        _producer_map but is intentionally absent from _node_by_id
        (it is not a NodeSpec). Any code path that called
        _node_by_id[producer.producer_id] for the source would raise
        KeyError — short-circuit here is load-bearing.
        """
        current = connection_name
        visited: set[str] = set()
        while True:
            if current in visited:
                return None
            visited.add(current)
            if current in self.duplicate_connections:
                return None
            if current not in self._producer_map:
                return None
            producer = self._producer_map[current]
            if producer.producer_id == "source":
                return producer
            producer_node = self._node_by_id[producer.producer_id]
            if producer_node.node_type == "gate":
                current = producer_node.input
                continue
            return producer
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_producer_resolver.py -v`
Expected: All 11 tests pass (9 base + 2 source-walk tests).

- [ ] **Step 5: Refactor `validate_transform_framing_contracts` to consume the resolver**

In `src/elspeth/web/composer/state.py`, change `validate_transform_framing_contracts` (line 351) to use `ProducerResolver`. Keep the function's signature, behaviour, and ALL existing tests green — this is a refactor, not a behaviour change.

```python
def validate_transform_framing_contracts(nodes: tuple[NodeSpec, ...]) -> tuple[ValidationEntry, ...]:
    """Validate cross-transform framing contracts that schemas cannot express.

    Retained as a thin shim during Phase 0–5; deleted in Phase 6 once the
    semantic validator owns this case.
    """
    from elspeth.web.composer._producer_resolver import ProducerResolver

    errors: list[ValidationEntry] = []
    resolver = ProducerResolver.build(source=None, nodes=nodes, sink_names=frozenset())

    for node in nodes:
        if node.node_type != "transform" or node.plugin != "line_explode":
            continue
        if "source_field" not in node.options:
            continue
        source_field = node.options["source_field"]
        upstream_producer = resolver.walk_to_real_producer(node.input)
        if upstream_producer is None or upstream_producer.plugin_name != "web_scrape":
            continue
        if "content_field" not in upstream_producer.options:
            continue
        content_field = upstream_producer.options["content_field"]
        if content_field != source_field:
            continue
        scrape_format = upstream_producer.options["format"] if "format" in upstream_producer.options else "markdown"
        if scrape_format != "text":
            continue
        text_separator = upstream_producer.options["text_separator"] if "text_separator" in upstream_producer.options else " "
        if type(text_separator) is str and "\n" in text_separator:
            continue
        errors.append(
            ValidationEntry(
                f"node:{node.id}",
                f"line_explode '{node.id}' consumes web_scrape text content field '{source_field}' from "
                f"'{upstream_producer.producer_id}', but format='text' requires text_separator to contain '\\n' before "
                "page contents can be split into lines. Set text_separator: '\\n' on the web_scrape transform "
                "or use format: markdown.",
                "high",
            )
        )
    return tuple(errors)
```

Delete the now-unused `_walk_to_real_producer` closure inside the old function (lines 391–406) and the inline `_register_producer` plus `producer_map`/`duplicate_connections` setup (lines 360–389). They are replaced by the resolver.

- [ ] **Step 6: Run framing tests to verify behaviour preserved**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py tests/unit/web/execution/test_validation.py tests/unit/web/execution/test_service.py -v -k 'framing or Framing'`
Expected: All existing framing tests pass unchanged.

- [ ] **Step 7: Refactor `_check_schema_contracts` walk-back to use the resolver**

In `state.py:543`, replace the inline `_register_producer` (lines 579–599) and `_walk_to_real_producer` (lines 784–799) with calls to `ProducerResolver`. Keep `_walk_producer_entry_to_real_producer`'s coalesce/fork-warning behaviour — that is schema-specific and stays inside `_check_schema_contracts`. The resolver provides the basic gate walk; the schema function wraps it with its richer warning emission.

Implementation note: the schema function's walk needs to emit warnings for coalesce and fork-to-sink cases that the resolver's plain walk does not. The cleanest factoring is:
1. Build the resolver up front.
2. Use `resolver.find_producer_for(...)` to seed the walk.
3. Keep a schema-specific walker (`_walk_producer_entry_to_real_producer`) that consults `resolver._node_by_id` (or a new public accessor) for node-type checks and emits warnings as it goes.

Add a public accessor on `ProducerResolver` to support this:

```python
def get_node(self, node_id: str) -> NodeSpec | None:
    """Return the registered NodeSpec for a producer id, or None.

    Returns None when the id is "source" (the source is intentionally
    not a NodeSpec) or when the id is unknown. Schema-contract code
    interpreting source-as-producer must short-circuit on None
    rather than indexing the underlying dict.
    """
    return self._node_by_id.get(node_id)
```

- [ ] **Step 8: Run full schema-contract tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All schema-contract tests pass unchanged. If any test fails, the refactor changed behaviour — investigate before continuing.

- [ ] **Step 9: Run lint + type check**

```bash
.venv/bin/python -m ruff check src/elspeth/web/composer/
.venv/bin/python -m mypy src/elspeth/web/composer/_producer_resolver.py
```

Expected: No new warnings in resolver code; no mypy errors.

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/web/composer/_producer_resolver.py \
        src/elspeth/web/composer/state.py \
        tests/unit/web/composer/test_producer_resolver.py
git commit -m "refactor(composer): extract ProducerResolver shared by schema + semantic validators

Both validate_transform_framing_contracts and _check_schema_contracts had their
own inline producer-map + walk-back. The semantic validator (next phase) needs
the same primitive — extract it to a shared module so we don't end up with
three independent walk-backs.

Pass-through propagation stays in state.py because semantic validation
does not propagate through pass-through in Phase 1.

Phase 0 of plugin-declared semantics."
```

---

### Task 0.3: Audit in-tree pipelines for `line_explode` usage

**Files:**
- Read: `data/`, `examples/`, `tests/integration/`, anything else that ships YAML pipelines
- Output: a one-paragraph note in the execution journal listing every in-tree pipeline that uses `line_explode` and the immediate upstream producer plugin name.

**Goal:** Confirm that with source → transform edges out of scope (per the design decision above), the only in-tree pipeline affected by `unknown_policy=FAIL` is the Wardline transform → transform chain. If other pipelines route source → line_explode without a declaring transform between them, document them — the in-tree pipelines themselves still work today (the validator skips source-fed edges), but operators authoring new pipelines will need to know the limitation.

- [ ] **Step 1: Find every YAML using `line_explode`**

```bash
grep -rln 'plugin: line_explode\|line_explode' data/ examples/ tests/integration/ src/elspeth/web/composer/skills/ 2>/dev/null
```

- [ ] **Step 2: For each match, identify the upstream producer**

For each YAML, locate the transform whose `on_success` matches `line_explode`'s `input` (or the source's `on_success` if the source feeds line_explode directly).

- [ ] **Step 3: Categorize and record**

Write a table in the execution journal:

| Pipeline | Upstream producer | Producer kind | Phase 1 outcome under FAIL |
|----------|-------------------|---------------|---------------------------|
| data/wardline_line_export_pipeline.yaml | web_scrape | transform (declared in Phase 2) | SATISFIED (when text_separator includes \n) |
| ... | ... | source/transform | (skip / SATISFIED / fail) |

- [ ] **Step 4: Decide based on findings**

- If only Wardline-shape (transform → line_explode) chains exist, proceed unmodified.
- If a source → line_explode pipeline exists in-tree, the validator's "skip source-fed edges" rule covers it without breakage; record the limitation and continue.
- If a non-`web_scrape` transform → line_explode chain exists where the transform has no semantics declaration, that chain will fail under FAIL once Phase 6 deletes the legacy validator. Either declare semantics on that transform in this plan (extend Phase 2) or pause the plan to discuss scope expansion.

- [ ] **Step 5: Commit the journal note**

```bash
git add docs/plans/2026-04-27-plugin-declared-transform-semantics-v2.md
git commit -m "chore: in-tree audit of line_explode usage (Phase 0 Task 0.3)

Confirms FAIL policy is safe given source-fed edges are skipped;
records any non-web_scrape transform → line_explode chains that need
semantic declarations added to this plan."
```

---

## Phase 1 — Contract types

Build the L0 types every later phase consumes. No behaviour change to running code yet.

### Task 1.1: `ContentKind`, `TextFraming`, `UnknownSemanticPolicy`, `SemanticOutcome` enums

**Files:**
- Create: `src/elspeth/contracts/plugin_semantics.py` (enums portion)
- Test: `tests/unit/contracts/test_plugin_semantics.py`

- [ ] **Step 1: Write failing tests for enum membership and StrEnum behaviour**

Create `tests/unit/contracts/test_plugin_semantics.py`:

```python
"""Tests for plugin semantics contract types."""

from __future__ import annotations

import pytest

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    SemanticOutcome,
    TextFraming,
    UnknownSemanticPolicy,
)


class TestContentKind:
    def test_known_members(self):
        assert ContentKind.UNKNOWN.value == "unknown"
        assert ContentKind.PLAIN_TEXT.value == "plain_text"
        assert ContentKind.MARKDOWN.value == "markdown"
        assert ContentKind.HTML_RAW.value == "html_raw"
        assert ContentKind.JSON_STRUCTURED.value == "json_structured"
        assert ContentKind.BINARY.value == "binary"

    def test_is_str_subclass(self):
        assert isinstance(ContentKind.PLAIN_TEXT, str)
        assert ContentKind.PLAIN_TEXT == "plain_text"

    def test_membership_is_closed_for_phase_1(self):
        # Phase 1 vocabulary — additions require explicit plan amendment.
        assert {member.value for member in ContentKind} == {
            "unknown", "plain_text", "markdown", "html_raw",
            "json_structured", "binary",
        }


class TestTextFraming:
    def test_known_members(self):
        assert TextFraming.UNKNOWN.value == "unknown"
        assert TextFraming.NOT_TEXT.value == "not_text"
        assert TextFraming.COMPACT.value == "compact"
        assert TextFraming.NEWLINE_FRAMED.value == "newline_framed"
        assert TextFraming.LINE_COMPATIBLE.value == "line_compatible"

    def test_membership_is_closed_for_phase_1(self):
        assert {member.value for member in TextFraming} == {
            "unknown", "not_text", "compact", "newline_framed", "line_compatible",
        }


class TestUnknownSemanticPolicy:
    def test_known_members(self):
        assert UnknownSemanticPolicy.ALLOW.value == "allow"
        assert UnknownSemanticPolicy.WARN.value == "warn"
        assert UnknownSemanticPolicy.FAIL.value == "fail"


class TestSemanticOutcome:
    def test_known_members(self):
        assert SemanticOutcome.SATISFIED.value == "satisfied"
        assert SemanticOutcome.CONFLICT.value == "conflict"
        assert SemanticOutcome.UNKNOWN.value == "unknown"
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: ImportError.

- [ ] **Step 3: Create the enums**

Create `src/elspeth/contracts/plugin_semantics.py`:

```python
"""Plugin-declared semantic contracts.

L0 module (contracts layer). Imports nothing above L0.

Vocabulary is intentionally CLOSED. Additions require design review and
a plan amendment — adding enum values lazily is exactly how the project
ends up rebuilding ad hoc runtime validation as expanding prose.
"""

from __future__ import annotations

from enum import StrEnum


class ContentKind(StrEnum):
    """The kind of content a field carries."""

    UNKNOWN = "unknown"
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML_RAW = "html_raw"
    JSON_STRUCTURED = "json_structured"
    BINARY = "binary"


class TextFraming(StrEnum):
    """How a text-bearing field is framed for downstream line operations."""

    UNKNOWN = "unknown"
    NOT_TEXT = "not_text"
    COMPACT = "compact"
    NEWLINE_FRAMED = "newline_framed"
    LINE_COMPATIBLE = "line_compatible"


class UnknownSemanticPolicy(StrEnum):
    """How a consumer treats an UNKNOWN producer fact for a required field.

    Phase 1 line_explode uses FAIL — every producer that semantically
    feeds it must declare semantics. WARN and ALLOW are present for
    future consumers but are not used in Phase 1.
    """

    ALLOW = "allow"
    WARN = "warn"
    FAIL = "fail"


class SemanticOutcome(StrEnum):
    """Result of comparing producer facts to a consumer requirement."""

    SATISFIED = "satisfied"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"
```

- [ ] **Step 4: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/plugin_semantics.py tests/unit/contracts/test_plugin_semantics.py
git commit -m "feat(contracts): add plugin semantics enum vocabulary

ContentKind, TextFraming, UnknownSemanticPolicy, SemanticOutcome enums.
Closed vocabulary at the L0 contracts layer."
```

---

### Task 1.2: `FieldSemanticFacts`, `OutputSemanticDeclaration`, `FieldSemanticRequirement`, `InputSemanticRequirements` dataclasses

**Files:**
- Modify: `src/elspeth/contracts/plugin_semantics.py` (append the dataclasses)
- Modify: `tests/unit/contracts/test_plugin_semantics.py` (add dataclass tests)

- [ ] **Step 1: Write failing tests for the dataclasses**

Append to `tests/unit/contracts/test_plugin_semantics.py`:

```python
from dataclasses import FrozenInstanceError

from elspeth.contracts.plugin_semantics import (
    FieldSemanticFacts,
    FieldSemanticRequirement,
    InputSemanticRequirements,
    OutputSemanticDeclaration,
)


class TestFieldSemanticFacts:
    def test_construct(self):
        facts = FieldSemanticFacts(
            field_name="content",
            content_kind=ContentKind.PLAIN_TEXT,
            text_framing=TextFraming.COMPACT,
            fact_code="web_scrape.content.compact_text",
            configured_by=("format", "text_separator"),
        )
        assert facts.field_name == "content"
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.COMPACT
        assert facts.fact_code == "web_scrape.content.compact_text"
        assert facts.configured_by == ("format", "text_separator")

    def test_immutable(self):
        facts = FieldSemanticFacts(
            field_name="x", content_kind=ContentKind.PLAIN_TEXT,
            fact_code="t.x.basic",
        )
        with pytest.raises(FrozenInstanceError):
            facts.field_name = "y"  # type: ignore[misc]

    def test_default_configured_by_is_empty_tuple(self):
        facts = FieldSemanticFacts(
            field_name="x", content_kind=ContentKind.UNKNOWN,
            fact_code="t.x.unknown",
        )
        assert facts.configured_by == ()


class TestFieldSemanticRequirement:
    def test_construct_and_compare_against_satisfied_facts(self):
        requirement = FieldSemanticRequirement(
            field_name="content",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE}),
            requirement_code="line_explode.source_field.line_framed_text",
            unknown_policy=UnknownSemanticPolicy.FAIL,
        )
        assert requirement.field_name == "content"
        assert ContentKind.PLAIN_TEXT in requirement.accepted_content_kinds
        assert TextFraming.LINE_COMPATIBLE in requirement.accepted_text_framings
        assert requirement.severity == "high"  # default

    def test_immutable(self):
        requirement = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
            requirement_code="t.x.req",
        )
        with pytest.raises(FrozenInstanceError):
            requirement.field_name = "y"  # type: ignore[misc]


class TestOutputSemanticDeclaration:
    def test_default_is_empty(self):
        decl = OutputSemanticDeclaration()
        assert decl.fields == ()

    def test_carries_facts(self):
        f1 = FieldSemanticFacts("a", ContentKind.PLAIN_TEXT, fact_code="t.a")
        f2 = FieldSemanticFacts("b", ContentKind.MARKDOWN, fact_code="t.b")
        decl = OutputSemanticDeclaration(fields=(f1, f2))
        assert decl.fields == (f1, f2)


class TestInputSemanticRequirements:
    def test_default_is_empty(self):
        reqs = InputSemanticRequirements()
        assert reqs.fields == ()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: ImportError on the new dataclass names.

- [ ] **Step 3: Add the dataclasses**

Append to `src/elspeth/contracts/plugin_semantics.py`:

```python
from collections.abc import Mapping  # noqa: E402  (intentional placement after enums)
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FieldSemanticFacts:
    """Structured facts a producer declares about a field it emits.

    All container fields are tuples / enum values. ``configured_by``
    names option paths that influenced this fact; it MUST contain only
    safe option names, never values, URLs, headers, prompts, row data,
    or exception text.
    """

    field_name: str
    content_kind: ContentKind
    text_framing: TextFraming = TextFraming.UNKNOWN
    fact_code: str = "field_semantics"
    configured_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OutputSemanticDeclaration:
    """A producer's full semantic facts across the fields it emits."""

    fields: tuple[FieldSemanticFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class FieldSemanticRequirement:
    """Structured requirements a consumer declares for a field it consumes."""

    field_name: str
    accepted_content_kinds: frozenset[ContentKind]
    accepted_text_framings: frozenset[TextFraming]
    requirement_code: str
    severity: str = "high"
    unknown_policy: UnknownSemanticPolicy = UnknownSemanticPolicy.FAIL
    configured_by: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class InputSemanticRequirements:
    """A consumer's full semantic requirements across the fields it consumes."""

    fields: tuple[FieldSemanticRequirement, ...] = ()
```

These dataclasses use only `tuple[...]`, `frozenset[...]`, `str`, and enum-typed fields — all natively immutable. No `freeze_fields(self, ...)` is required because there are no `dict`, `list`, `set`, `Mapping`, or `Sequence` fields.

- [ ] **Step 4: Run tests, verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/contracts/plugin_semantics.py tests/unit/contracts/test_plugin_semantics.py
git commit -m "feat(contracts): add semantic facts/requirements/declaration dataclasses"
```

---

### Task 1.3: `SemanticEdgeContract` + comparator function `compare_semantic(...)`

**Files:**
- Modify: `src/elspeth/contracts/plugin_semantics.py`
- Modify: `tests/unit/contracts/test_plugin_semantics.py`

- [ ] **Step 1: Write failing tests for `SemanticEdgeContract` and `compare_semantic`**

Append to `tests/unit/contracts/test_plugin_semantics.py`:

```python
from elspeth.contracts.plugin_semantics import SemanticEdgeContract, compare_semantic


class TestSemanticEdgeContract:
    def test_construct(self):
        facts = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT, fact_code="t.x")
        req = FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
            accepted_text_framings=frozenset({TextFraming.UNKNOWN, TextFraming.LINE_COMPATIBLE}),
            requirement_code="c.x.req",
        )
        edge = SemanticEdgeContract(
            from_id="a", to_id="b",
            consumer_plugin="line_explode", producer_plugin="web_scrape",
            producer_field="x", consumer_field="x",
            producer_facts=facts, requirement=req,
            outcome=SemanticOutcome.SATISFIED,
        )
        assert edge.outcome is SemanticOutcome.SATISFIED
        assert edge.consumer_plugin == "line_explode"
        assert edge.producer_plugin == "web_scrape"


class TestCompareSemantic:
    def _req(self, kinds, framings, policy=UnknownSemanticPolicy.FAIL):
        return FieldSemanticRequirement(
            field_name="x",
            accepted_content_kinds=frozenset(kinds),
            accepted_text_framings=frozenset(framings),
            requirement_code="t.x.req",
            unknown_policy=policy,
        )

    def test_satisfied_when_facts_within_acceptance(self):
        facts = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT,
                                   text_framing=TextFraming.NEWLINE_FRAMED,
                                   fact_code="t.x.nl")
        req = self._req({ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN},
                        {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE})
        assert compare_semantic(facts, req) is SemanticOutcome.SATISFIED

    def test_conflict_on_content_kind_mismatch(self):
        facts = FieldSemanticFacts("x", ContentKind.HTML_RAW,
                                   text_framing=TextFraming.NOT_TEXT,
                                   fact_code="t.x.raw")
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(facts, req) is SemanticOutcome.CONFLICT

    def test_conflict_on_framing_mismatch(self):
        facts = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT,
                                   text_framing=TextFraming.COMPACT,
                                   fact_code="t.x.compact")
        req = self._req({ContentKind.PLAIN_TEXT},
                        {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE})
        assert compare_semantic(facts, req) is SemanticOutcome.CONFLICT

    def test_unknown_when_facts_are_none(self):
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(None, req) is SemanticOutcome.UNKNOWN

    def test_unknown_when_either_dimension_is_unknown(self):
        facts_kind_unknown = FieldSemanticFacts("x", ContentKind.UNKNOWN,
                                                text_framing=TextFraming.NEWLINE_FRAMED,
                                                fact_code="t.x.kindless")
        facts_framing_unknown = FieldSemanticFacts("x", ContentKind.PLAIN_TEXT,
                                                   text_framing=TextFraming.UNKNOWN,
                                                   fact_code="t.x.framingless")
        req = self._req({ContentKind.PLAIN_TEXT}, {TextFraming.NEWLINE_FRAMED})
        assert compare_semantic(facts_kind_unknown, req) is SemanticOutcome.UNKNOWN
        assert compare_semantic(facts_framing_unknown, req) is SemanticOutcome.UNKNOWN
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: ImportError.

- [ ] **Step 3: Add `SemanticEdgeContract` and `compare_semantic`**

Append to `src/elspeth/contracts/plugin_semantics.py`:

```python
@dataclass(frozen=True, slots=True)
class SemanticEdgeContract:
    """Per-edge result of comparing producer facts to consumer requirement.

    consumer_plugin is REQUIRED — assistance lookup MUST address a
    specific plugin class, not iterate every registered transform.
    producer_plugin is optional because some producers (e.g., source)
    are not registered transform classes.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    producer_facts: FieldSemanticFacts | None
    requirement: FieldSemanticRequirement
    outcome: SemanticOutcome


def compare_semantic(
    facts: FieldSemanticFacts | None,
    requirement: FieldSemanticRequirement,
) -> SemanticOutcome:
    """Compare producer facts to a consumer requirement.

    Returns UNKNOWN if facts are absent or any compared dimension is
    UNKNOWN. Returns CONFLICT if either dimension is not in the
    accepted set. Returns SATISFIED only when both dimensions are
    explicitly in the accepted set.
    """
    if facts is None:
        return SemanticOutcome.UNKNOWN
    if facts.content_kind is ContentKind.UNKNOWN:
        return SemanticOutcome.UNKNOWN
    if facts.text_framing is TextFraming.UNKNOWN:
        return SemanticOutcome.UNKNOWN
    if facts.content_kind not in requirement.accepted_content_kinds:
        return SemanticOutcome.CONFLICT
    if facts.text_framing not in requirement.accepted_text_framings:
        return SemanticOutcome.CONFLICT
    return SemanticOutcome.SATISFIED
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: All tests pass.

- [ ] **Step 5: Add property-based test for comparator exhaustiveness**

Append:

```python
from hypothesis import given, strategies as st


_CONTENT_KINDS = list(ContentKind)
_FRAMINGS = list(TextFraming)


@given(
    content_kind=st.sampled_from(_CONTENT_KINDS),
    text_framing=st.sampled_from(_FRAMINGS),
    accepted_kinds=st.sets(st.sampled_from(_CONTENT_KINDS), min_size=1),
    accepted_framings=st.sets(st.sampled_from(_FRAMINGS), min_size=1),
)
def test_compare_semantic_outcome_is_consistent(
    content_kind, text_framing, accepted_kinds, accepted_framings,
):
    facts = FieldSemanticFacts(
        field_name="x",
        content_kind=content_kind,
        text_framing=text_framing,
        fact_code="t.x.gen",
    )
    requirement = FieldSemanticRequirement(
        field_name="x",
        accepted_content_kinds=frozenset(accepted_kinds),
        accepted_text_framings=frozenset(accepted_framings),
        requirement_code="c.x.req",
    )
    outcome = compare_semantic(facts, requirement)

    if content_kind is ContentKind.UNKNOWN or text_framing is TextFraming.UNKNOWN:
        assert outcome is SemanticOutcome.UNKNOWN
    elif (content_kind in accepted_kinds and text_framing in accepted_framings):
        assert outcome is SemanticOutcome.SATISFIED
    else:
        assert outcome is SemanticOutcome.CONFLICT
```

- [ ] **Step 6: Run all contract tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics.py -v`
Expected: All tests pass including hypothesis property test.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/contracts/plugin_semantics.py tests/unit/contracts/test_plugin_semantics.py
git commit -m "feat(contracts): SemanticEdgeContract + compare_semantic comparator

Property test covers the (ContentKind x TextFraming x accepted-sets)
outcome space — guards against future enum additions that could break
exhaustiveness."
```

---

### Task 1.4: `PluginAssistance` and `PluginAssistanceExample` (with freeze guards)

**Files:**
- Create: `src/elspeth/contracts/plugin_assistance.py`
- Test: `tests/unit/contracts/test_plugin_assistance.py`

- [ ] **Step 1: Write failing tests, including the deep-freeze post-construction mutation tests**

Create `tests/unit/contracts/test_plugin_assistance.py`:

```python
"""Tests for plugin assistance contract types — including deep-freeze guards."""

from __future__ import annotations

from types import MappingProxyType

import pytest

from elspeth.contracts.plugin_assistance import (
    PluginAssistance,
    PluginAssistanceExample,
)


class TestPluginAssistanceExampleFreeze:
    def test_dict_fields_become_mapping_proxy(self):
        example = PluginAssistanceExample(
            title="t",
            before={"format": "text", "text_separator": " "},
            after={"format": "text", "text_separator": "\n"},
        )
        assert isinstance(example.before, MappingProxyType)
        assert isinstance(example.after, MappingProxyType)

    def test_none_fields_are_left_none(self):
        example = PluginAssistanceExample(title="t", before=None, after=None)
        assert example.before is None
        assert example.after is None

    def test_inner_mutation_is_blocked(self):
        original = {"format": "text"}
        example = PluginAssistanceExample(title="t", before=original)
        with pytest.raises(TypeError):
            example.before["format"] = "markdown"  # type: ignore[index]
        # And mutating the source dict does NOT affect the frozen field
        original["format"] = "markdown"
        assert example.before["format"] == "text"


class TestPluginAssistanceFreeze:
    def test_examples_field_freezes_inner_dicts(self):
        example = PluginAssistanceExample(title="t", before={"k": "v"})
        assistance = PluginAssistance(
            plugin_name="web_scrape",
            issue_code="line_explode.source_field.line_framed_text",
            summary="Set text_separator to '\\n'.",
            suggested_fixes=("Set text_separator: '\\n'", "Or use format: markdown"),
            examples=(example,),
        )
        # Both examples and inner dicts are frozen.
        assert isinstance(assistance.examples, tuple)
        assert isinstance(assistance.examples[0].before, MappingProxyType)

    def test_required_fields(self):
        # plugin_name, issue_code, summary are REQUIRED (no defaults).
        # suggested_fixes, examples, composer_hints have empty defaults.
        assistance = PluginAssistance(
            plugin_name="p", issue_code=None, summary="s",
        )
        assert assistance.suggested_fixes == ()
        assert assistance.examples == ()
        assert assistance.composer_hints == ()
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_assistance.py -v`
Expected: ImportError.

- [ ] **Step 3: Create `plugin_assistance.py` with freeze guards**

Create `src/elspeth/contracts/plugin_assistance.py`:

```python
"""Plugin assistance: deterministic guidance keyed by issue codes.

L0 module. Carries no plugin runtime references. Consumers (catalog
service, MCP discovery, validators) attach assistance to issue codes;
they do not parse summary/suggested_fixes prose.

Secret discipline: assistance fields MUST contain only safe option
names, plugin names, enum values, and human-readable advice. They MUST
NOT contain raw URLs, headers, prompts, row data, credentials, raw
provider errors, file paths, or exception strings. Enforcement is by
plugin authors and tests (see secret-leakage tests in Phase 3).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True, slots=True)
class PluginAssistanceExample:
    """A before/after configuration sketch."""

    title: str
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.before is not None:
            freeze_fields(self, "before")
        if self.after is not None:
            freeze_fields(self, "after")


@dataclass(frozen=True, slots=True)
class PluginAssistance:
    """Deterministic, side-effect-free guidance for an issue code.

    Returned by ``BaseTransform.get_agent_assistance(issue_code=...)``.
    Validators attach the issue code; they do not parse the prose
    fields. Catalog/MCP discovery surfaces this as structured data.
    """

    plugin_name: str
    issue_code: str | None
    summary: str
    suggested_fixes: tuple[str, ...] = ()
    examples: tuple[PluginAssistanceExample, ...] = ()
    composer_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # examples is tuple[PluginAssistanceExample, ...]; the elements
        # already deep-freeze their own dict fields in their own __post_init__.
        # No additional freeze_fields needed for examples itself, because
        # tuple of frozen-dataclass elements is natively immutable AND
        # element fields are already frozen at element construction time.
        # suggested_fixes / composer_hints are tuple[str, ...]: natively
        # immutable, no guard needed.
        pass
```

Note: The `__post_init__` is a no-op deliberately. Document the reasoning so a future reader does not "helpfully" remove it or "helpfully" add a guard that would shallow-wrap and mask a real bug.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_assistance.py -v`
Expected: All tests pass.

- [ ] **Step 5: Run the freeze-guard CI script locally**

Run: `.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth/contracts/plugin_assistance.py`
Expected: No violations reported.

If the script reports a violation on `PluginAssistance` itself (because it has `tuple[PluginAssistanceExample, ...]`), add `PluginAssistance` to `config/cicd/enforce_freeze_guards/` allowlist with reason: "tuple of frozen-dataclass elements; elements deep-freeze their own dict fields in __post_init__."

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/contracts/plugin_assistance.py \
        tests/unit/contracts/test_plugin_assistance.py \
        config/cicd/enforce_freeze_guards/  # only if allowlist amended
git commit -m "feat(contracts): PluginAssistance + PluginAssistanceExample with freeze guards

PluginAssistanceExample.before/after deep-freeze via freeze_fields with
None-gating. PluginAssistance.examples is tuple-of-frozen-dataclass —
no additional guard needed because elements freeze their own contents."
```

---

### Task 1.5: Layer-purity test for `contracts/plugin_semantics.py` and `contracts/plugin_assistance.py`

**Files:**
- Test: `tests/unit/contracts/test_plugin_semantics_imports.py`

- [ ] **Step 1: Write the layer test**

Create `tests/unit/contracts/test_plugin_semantics_imports.py`:

```python
"""Verify L0 purity of the plugin semantics + assistance contract modules.

These modules sit in src/elspeth/contracts/ which is L0 — they may not
import anything from core/ (L1), engine/ (L2), or plugins/web/mcp/tui/
(L3). The CI script enforce_tier_model.py also catches this; this test
gives faster feedback during development.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJECT_ROOT / "src"

_FORBIDDEN_PREFIXES = (
    "elspeth.core",
    "elspeth.engine",
    "elspeth.plugins",
    "elspeth.web",
    "elspeth.mcp",
    "elspeth.composer_mcp",
    "elspeth.tui",
    "elspeth.cli",
    "elspeth.telemetry",
    "elspeth.testing",
)


def _module_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    return imports


@pytest.mark.parametrize("module_path", [
    "src/elspeth/contracts/plugin_semantics.py",
    "src/elspeth/contracts/plugin_assistance.py",
])
def test_module_does_not_import_above_l0(module_path: str):
    path = _PROJECT_ROOT / module_path
    imports = _module_imports(path)
    violations = [
        imp for imp in imports
        if any(imp == prefix or imp.startswith(f"{prefix}.")
               for prefix in _FORBIDDEN_PREFIXES)
    ]
    assert not violations, (
        f"{module_path} imports L1+ modules: {violations}. "
        f"Contracts must remain L0-pure."
    )
```

- [ ] **Step 2: Run, verify pass**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/contracts/test_plugin_semantics_imports.py -v`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_plugin_semantics_imports.py
git commit -m "test(contracts): layer-purity check for new L0 plugin semantics modules"
```

---

## Phase 2 — Plugin API

Add the three default declaration methods on `BaseTransform`, then implement them for `web_scrape` and `line_explode`.

### Task 2.1: `BaseTransform` default methods

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/base.py:57` (add three methods to `BaseTransform`)
- Test: extend `tests/unit/plugins/infrastructure/test_base.py` (or create a new test if existing test isn't appropriate)

- [ ] **Step 1: Find the existing base-class test file**

```bash
ls tests/unit/plugins/infrastructure/test_base.py 2>/dev/null || \
  find tests -name 'test_base*.py' -path '*plugins/infrastructure*'
```

If no test file exists for the base class methods specifically, create `tests/unit/plugins/infrastructure/test_base_semantics.py`.

- [ ] **Step 2: Write failing tests**

Create or append `tests/unit/plugins/infrastructure/test_base_semantics.py`:

```python
"""Tests for BaseTransform default semantic declaration methods."""

from __future__ import annotations

from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.plugin_semantics import (
    InputSemanticRequirements,
    OutputSemanticDeclaration,
)
from elspeth.plugins.infrastructure.base import BaseTransform


class _StubTransform(BaseTransform):
    name = "stub"

    def process(self, row, ctx):  # pragma: no cover — not exercised
        raise NotImplementedError


def test_default_output_semantics_is_empty():
    instance = _StubTransform.__new__(_StubTransform)
    decl = instance.output_semantics()
    assert isinstance(decl, OutputSemanticDeclaration)
    assert decl.fields == ()


def test_default_input_semantic_requirements_is_empty():
    instance = _StubTransform.__new__(_StubTransform)
    reqs = instance.input_semantic_requirements()
    assert isinstance(reqs, InputSemanticRequirements)
    assert reqs.fields == ()


def test_default_get_agent_assistance_returns_none():
    result = _StubTransform.get_agent_assistance(issue_code=None)
    assert result is None
    result_with_code = _StubTransform.get_agent_assistance(issue_code="any.code")
    assert result_with_code is None
```

- [ ] **Step 3: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_semantics.py -v`
Expected: AttributeError on the new methods.

- [ ] **Step 4: Add the methods to `BaseTransform`**

In `src/elspeth/plugins/infrastructure/base.py`, find the `BaseTransform` class definition (line 57) and add these methods after the existing class-level declarations (after the `passes_through_input` ClassVar — around the bottom of the class, before any final method definitions):

```python
    # ── Plugin-declared semantics (Phase 1: optional, default empty) ──
    # Override on a subclass to declare what the plugin emits / requires.
    # The generic semantic validator compares producer facts to consumer
    # requirements when the configured field names match.

    def output_semantics(self) -> "OutputSemanticDeclaration":
        """Return semantic facts for the fields this transform emits.

        Default returns an empty declaration: the transform makes no
        semantic claims beyond what the schema contract already
        expresses. Override to declare ContentKind/TextFraming for
        configured output fields.
        """
        from elspeth.contracts.plugin_semantics import OutputSemanticDeclaration
        return OutputSemanticDeclaration()

    def input_semantic_requirements(self) -> "InputSemanticRequirements":
        """Return semantic requirements for fields this transform consumes.

        Default returns no requirements. Override to declare that a
        configured input field must satisfy specific ContentKind /
        TextFraming acceptance sets.
        """
        from elspeth.contracts.plugin_semantics import InputSemanticRequirements
        return InputSemanticRequirements()

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> "PluginAssistance | None":
        """Return deterministic guidance keyed by an issue code.

        Default returns None: the plugin offers no specific guidance.
        Override to return a PluginAssistance instance describing fixes
        for this plugin's issue codes. Validators attach the issue
        code; the plugin owns the prose.
        """
        return None
```

Add the necessary type-only imports at the top of the file under `if TYPE_CHECKING:`:

```python
if TYPE_CHECKING:
    ...
    from elspeth.contracts.plugin_assistance import PluginAssistance
    from elspeth.contracts.plugin_semantics import (
        InputSemanticRequirements,
        OutputSemanticDeclaration,
    )
```

The runtime imports inside the methods are intentional: they avoid making `BaseTransform` import-fail in any environment where the contracts module is unavailable (none today, but the indirection costs nothing).

- [ ] **Step 5: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/infrastructure/test_base_semantics.py -v`
Expected: All 3 tests pass.

- [ ] **Step 6: Run the full plugin infrastructure test suite to catch regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/infrastructure/ -v`
Expected: All tests pass; no existing transforms broken by the new defaults.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/plugins/infrastructure/base.py \
        tests/unit/plugins/infrastructure/test_base_semantics.py
git commit -m "feat(plugins): add BaseTransform default semantic-declaration methods

output_semantics() and input_semantic_requirements() return empty
declarations; get_agent_assistance() returns None. Plugins override
to declare semantics. Defaults preserve existing-transform behaviour."
```

---

### Task 2.2: `web_scrape` output semantics + assistance

**Files:**
- Modify: `src/elspeth/plugins/transforms/web_scrape.py`
- Modify: `tests/unit/plugins/transforms/test_web_scrape.py`

- [ ] **Step 1: Write failing declaration tests**

Append to `tests/unit/plugins/transforms/test_web_scrape.py` (use the existing fixtures/helpers in that file for plugin construction):

```python
class TestWebScrapeOutputSemantics:
    def _build(self, **option_overrides):
        # WebScrapeConfig is a TransformDataConfig subclass — schema is
        # REQUIRED at construction. Omitting it raises PluginConfigError
        # which the validator's tolerant probe path silently absorbs;
        # the test would then pass vacuously without exercising
        # output_semantics() at all.
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        defaults = {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": "content",
            "fingerprint_field": "fingerprint",
            "format": "markdown",
            "http": {"abuse_contact": "x@example.com", "scraping_reason": "t",
                     "timeout": 5, "allowed_hosts": "public_only"},
        }
        defaults.update(option_overrides)
        return get_shared_plugin_manager().create_transform("web_scrape", defaults)

    def test_text_compact_separator_declares_plain_text_compact(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming
        ws = self._build(format="text", text_separator=" ")
        decl = ws.output_semantics()
        facts = next(f for f in decl.fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.COMPACT
        assert facts.fact_code == "web_scrape.content.compact_text"
        assert facts.configured_by == ("format", "text_separator")

    def test_text_newline_separator_declares_plain_text_newline_framed(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming
        ws = self._build(format="text", text_separator="\n")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.PLAIN_TEXT
        assert facts.text_framing is TextFraming.NEWLINE_FRAMED

    def test_markdown_declares_markdown_line_compatible(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming
        ws = self._build(format="markdown")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.MARKDOWN
        assert facts.text_framing is TextFraming.LINE_COMPATIBLE

    def test_raw_declares_html_raw_not_text(self):
        from elspeth.contracts.plugin_semantics import ContentKind, TextFraming
        ws = self._build(format="raw")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "content")
        assert facts.content_kind is ContentKind.HTML_RAW
        assert facts.text_framing is TextFraming.NOT_TEXT

    def test_custom_content_field_changes_semantic_field_name(self):
        ws = self._build(format="text", text_separator="\n", content_field="body")
        facts = next(f for f in ws.output_semantics().fields if f.field_name == "body")
        assert facts.field_name == "body"


class TestWebScrapeAssistance:
    def test_returns_assistance_for_compact_text_issue(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
        result = WebScrapeTransform.get_agent_assistance(
            issue_code="web_scrape.content.compact_text",
        )
        assert result is not None
        assert result.plugin_name == "web_scrape"
        assert result.issue_code == "web_scrape.content.compact_text"
        # Suggested fixes mention configuration knobs only — no values.
        assert any("text_separator" in fix for fix in result.suggested_fixes)
        assert any("markdown" in fix.lower() for fix in result.suggested_fixes)

    def test_returns_none_for_unknown_issue(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
        assert WebScrapeTransform.get_agent_assistance(issue_code="nope.unknown") is None

    def test_returns_none_when_no_issue_code(self):
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform
        assert WebScrapeTransform.get_agent_assistance(issue_code=None) is None
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_web_scrape.py -v -k 'OutputSemantics or Assistance'`
Expected: AttributeError or default-empty results.

- [ ] **Step 3: Implement `output_semantics()` and `get_agent_assistance()` on `WebScrapeTransform`**

In `src/elspeth/plugins/transforms/web_scrape.py`, add a module-level helper near `_build_web_scrape_output_schema_config` (around line 173):

```python
def _build_web_scrape_output_semantics(
    *,
    content_field: str,
    format: str,
    text_separator: str,
) -> "OutputSemanticDeclaration":
    """Map WebScrapeConfig values to declared output facts for the content field."""
    from elspeth.contracts.plugin_semantics import (
        ContentKind,
        FieldSemanticFacts,
        OutputSemanticDeclaration,
        TextFraming,
    )

    if format == "markdown":
        kind = ContentKind.MARKDOWN
        framing = TextFraming.LINE_COMPATIBLE
        fact_code = "web_scrape.content.markdown"
    elif format == "raw":
        kind = ContentKind.HTML_RAW
        framing = TextFraming.NOT_TEXT
        fact_code = "web_scrape.content.raw_html"
    elif format == "text":
        kind = ContentKind.PLAIN_TEXT
        if "\n" in text_separator:
            framing = TextFraming.NEWLINE_FRAMED
            fact_code = "web_scrape.content.newline_framed_text"
        else:
            framing = TextFraming.COMPACT
            fact_code = "web_scrape.content.compact_text"
    else:
        # Unknown format value — let the schema layer handle it.
        # Returning UNKNOWN here is honest: we don't know.
        kind = ContentKind.UNKNOWN
        framing = TextFraming.UNKNOWN
        fact_code = "web_scrape.content.unknown_format"

    return OutputSemanticDeclaration(
        fields=(
            FieldSemanticFacts(
                field_name=content_field,
                content_kind=kind,
                text_framing=framing,
                fact_code=fact_code,
                configured_by=("format", "text_separator"),
            ),
        ),
    )
```

Then add the instance method on the `WebScrapeTransform` class:

```python
    def output_semantics(self) -> "OutputSemanticDeclaration":
        return _build_web_scrape_output_semantics(
            content_field=self._content_field,
            format=self._format,
            text_separator=self._text_separator,
        )

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> "PluginAssistance | None":
        from elspeth.contracts.plugin_assistance import (
            PluginAssistance,
            PluginAssistanceExample,
        )

        if issue_code != "web_scrape.content.compact_text":
            return None
        return PluginAssistance(
            plugin_name="web_scrape",
            issue_code="web_scrape.content.compact_text",
            summary=(
                "format='text' with a non-newline text_separator produces a "
                "compact single-line string. Downstream line-oriented "
                "transforms (line_explode) cannot recover line boundaries."
            ),
            suggested_fixes=(
                "Set text_separator: '\\n' to preserve line boundaries.",
                "Or use format: markdown — markdown extraction preserves "
                "line-oriented structure.",
            ),
            examples=(
                PluginAssistanceExample(
                    title="Use newline separator with text format",
                    before={"format": "text", "text_separator": " "},
                    after={"format": "text", "text_separator": "\n"},
                ),
                PluginAssistanceExample(
                    title="Switch to markdown format",
                    before={"format": "text", "text_separator": " "},
                    after={"format": "markdown"},
                ),
            ),
        )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_web_scrape.py -v -k 'OutputSemantics or Assistance'`
Expected: All new tests pass.

- [ ] **Step 5: Run the full web_scrape test suite for regressions**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_web_scrape.py -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/plugins/transforms/web_scrape.py \
        tests/unit/plugins/transforms/test_web_scrape.py
git commit -m "feat(web_scrape): declare output semantics + assistance for content field

Maps format/text_separator to ContentKind + TextFraming.
get_agent_assistance() owns the compact-text gotcha guidance,
replacing the hardcoded validator suggestion."
```

---

### Task 2.3: `line_explode` input semantic requirements + assistance

**Files:**
- Modify: `src/elspeth/plugins/transforms/line_explode.py`
- Modify: `tests/unit/plugins/transforms/test_line_explode.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/unit/plugins/transforms/test_line_explode.py`:

```python
class TestLineExplodeInputSemanticRequirements:
    def _build(self, **opts):
        # LineExplodeConfig is a TransformDataConfig subclass — schema
        # is REQUIRED. Omission raises PluginConfigError → validator
        # silently skips → vacuous test. See web_scrape helper above
        # for the same pattern.
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        defaults = {
            "schema": {"mode": "flexible", "fields": ["content: str"]},
            "source_field": "content",
        }
        defaults.update(opts)
        return get_shared_plugin_manager().create_transform("line_explode", defaults)

    def test_default_source_field_requirement(self):
        from elspeth.contracts.plugin_semantics import (
            ContentKind, TextFraming, UnknownSemanticPolicy,
        )
        plugin = self._build()
        reqs = plugin.input_semantic_requirements()
        assert len(reqs.fields) == 1
        req = reqs.fields[0]
        assert req.field_name == "content"
        assert req.accepted_content_kinds == frozenset(
            {ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN}
        )
        assert req.accepted_text_framings == frozenset(
            {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE}
        )
        assert req.requirement_code == "line_explode.source_field.line_framed_text"
        assert req.unknown_policy is UnknownSemanticPolicy.FAIL
        assert req.configured_by == ("source_field",)

    def test_custom_source_field_changes_requirement_field_name(self):
        plugin = self._build(source_field="body")
        req = plugin.input_semantic_requirements().fields[0]
        assert req.field_name == "body"


class TestLineExplodeAssistance:
    def test_returns_assistance_for_line_framed_requirement(self):
        from elspeth.plugins.transforms.line_explode import LineExplode
        a = LineExplode.get_agent_assistance(
            issue_code="line_explode.source_field.line_framed_text",
        )
        assert a is not None
        assert a.plugin_name == "line_explode"
        assert "splitlines" in a.summary or "line" in a.summary.lower()

    def test_returns_none_for_unknown_issue(self):
        from elspeth.plugins.transforms.line_explode import LineExplode
        assert LineExplode.get_agent_assistance(issue_code="nope") is None
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_line_explode.py -v -k 'InputSemantic or Assistance'`
Expected: AttributeError / empty results.

- [ ] **Step 3: Implement on `LineExplode`**

In `src/elspeth/plugins/transforms/line_explode.py`, add a helper near `_build_line_explode_output_schema_config` (around line 99):

```python
def _build_line_explode_input_requirements(
    *, source_field: str,
) -> "InputSemanticRequirements":
    from elspeth.contracts.plugin_semantics import (
        ContentKind,
        FieldSemanticRequirement,
        InputSemanticRequirements,
        TextFraming,
        UnknownSemanticPolicy,
    )
    return InputSemanticRequirements(
        fields=(
            FieldSemanticRequirement(
                field_name=source_field,
                accepted_content_kinds=frozenset(
                    {ContentKind.PLAIN_TEXT, ContentKind.MARKDOWN},
                ),
                accepted_text_framings=frozenset(
                    {TextFraming.NEWLINE_FRAMED, TextFraming.LINE_COMPATIBLE},
                ),
                requirement_code="line_explode.source_field.line_framed_text",
                severity="high",
                unknown_policy=UnknownSemanticPolicy.FAIL,
                configured_by=("source_field",),
            ),
        ),
    )
```

Then on the `LineExplode` class:

```python
    def input_semantic_requirements(self) -> "InputSemanticRequirements":
        return _build_line_explode_input_requirements(
            source_field=self._source_field,
        )

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> "PluginAssistance | None":
        from elspeth.contracts.plugin_assistance import PluginAssistance

        if issue_code != "line_explode.source_field.line_framed_text":
            return None
        return PluginAssistance(
            plugin_name="line_explode",
            issue_code="line_explode.source_field.line_framed_text",
            summary=(
                "line_explode calls splitlines() on the configured source_field. "
                "Compact text (a single string with no newlines) emits one row "
                "containing the whole content — the opposite of line "
                "deaggregation. The producer must emit newline-framed or "
                "line-compatible text."
            ),
            suggested_fixes=(
                "Configure the upstream producer to emit newline-framed text "
                "(for web_scrape: text_separator: '\\n', or use format: markdown).",
                "Choose a producer whose output_semantics() declares a "
                "text_framing of newline_framed or line_compatible.",
            ),
        )
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_line_explode.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/transforms/line_explode.py \
        tests/unit/plugins/transforms/test_line_explode.py
git commit -m "feat(line_explode): declare input semantic requirement + assistance

Requires source_field to be line-framed text (PLAIN_TEXT|MARKDOWN x
NEWLINE_FRAMED|LINE_COMPATIBLE). UnknownSemanticPolicy.FAIL — every
producer feeding line_explode must declare semantics."
```

---

## Phase 3 — Generic semantic validator

Build the generic validator that compares producer facts to consumer requirements. The validator consumes the `ProducerResolver` from Phase 0 and emits structured `SemanticEdgeContract` records plus `ValidationEntry` errors.

### Task 3.1: `validate_semantic_contracts(...)` core algorithm

**Files:**
- Create: `src/elspeth/web/composer/_semantic_validator.py`
- Test: `tests/unit/web/composer/test_semantic_validator.py`

- [ ] **Step 1: Write failing core-algorithm tests**

Create `tests/unit/web/composer/test_semantic_validator.py`:

```python
"""Tests for validate_semantic_contracts algorithm."""

from __future__ import annotations

import pytest

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    SemanticOutcome,
    TextFraming,
)
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)


def _wardline_state(*, text_separator: str = " ", scrape_format: str = "text"):
    """Build the canonical Wardline-shape composition: scrape -> explode -> sink.

    Required to satisfy real config validation:
    - csv source has schema (DataPluginConfig.schema_config is required)
    - web_scrape transform has schema, on_success, on_error
    - line_explode transform has schema, on_success, on_error
    Without these, plugin construction fails as a "draft config" error,
    the validator's tolerant probe path silently skips, and the test
    becomes vacuous (no error raised, but no contract emitted either).
    """
    return CompositionState(
        metadata=PipelineMetadata(name="wardline"),
        source=SourceSpec(
            plugin="csv", on_success="scrape_in",
            options={"path": "data/url.csv",
                     "schema": {"mode": "fixed", "fields": ["url: str"]}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="scrape", node_type="transform", plugin="web_scrape",
                input="scrape_in", on_success="explode_in", on_error="errors",
                options={
                    "schema": {"mode": "flexible", "fields": ["url: str"]},
                    "required_input_fields": ["url"],
                    "url_field": "url",
                    "content_field": "content",
                    "fingerprint_field": "fingerprint",
                    "format": scrape_format,
                    "text_separator": text_separator,
                    "http": {"abuse_contact": "x@example.com",
                             "scraping_reason": "t", "timeout": 5,
                             "allowed_hosts": "public_only"},
                },
            ),
            NodeSpec(
                id="explode", node_type="transform", plugin="line_explode",
                input="explode_in", on_success="sink", on_error="errors",
                options={
                    "schema": {"mode": "flexible", "fields": ["content: str"]},
                    "source_field": "content",
                },
            ),
        ),
        outputs=(
            OutputSpec(name="sink", plugin="json",
                       options={"path": "out.json"}),
            OutputSpec(name="errors", plugin="json",
                       options={"path": "err.json"}),
        ),
    )


class TestValidateSemanticContracts:
    def test_compact_text_produces_conflict(self):
        state = _wardline_state(text_separator=" ", scrape_format="text")
        errors, contracts = validate_semantic_contracts(state)

        assert len(errors) == 1
        error = errors[0]
        assert error.severity == "high"
        assert "line_explode" in error.message
        assert "web_scrape" in error.message
        assert "content" in error.message
        # Diagnostic must include the requirement_code so a UI / agent
        # can look up plugin-owned assistance.
        assert "line_explode.source_field.line_framed_text" in error.message
        # Generic diagnostic must NOT contain fix-prose tokens — those
        # belong in PluginAssistance, not the validator.
        assert "text_separator" not in error.message
        assert "use markdown" not in error.message.lower()
        assert "set " not in error.message.lower()  # imperative fix-language

        assert len(contracts) == 1
        contract = contracts[0]
        assert contract.from_id == "scrape"
        assert contract.to_id == "explode"
        assert contract.producer_field == "content"
        assert contract.consumer_field == "content"
        assert contract.outcome is SemanticOutcome.CONFLICT
        assert contract.requirement.requirement_code == \
            "line_explode.source_field.line_framed_text"
        assert contract.producer_facts is not None
        assert contract.producer_facts.text_framing is TextFraming.COMPACT

    def test_newline_text_passes(self):
        state = _wardline_state(text_separator="\n", scrape_format="text")
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].producer_facts.text_framing is TextFraming.NEWLINE_FRAMED

    def test_markdown_passes(self):
        state = _wardline_state(scrape_format="markdown")
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].producer_facts.content_kind is ContentKind.MARKDOWN

    def test_source_fed_consumer_emits_no_semantic_contract(self):
        # Phase 1 design: source -> transform edges are out of scope.
        # The validator skips them entirely (no contract, no error).
        # If this test fails, Phase 1 has accidentally re-enabled
        # source-fed semantic checking.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            source=SourceSpec(
                plugin="csv", on_success="explode_in",
                options={"path": "x.csv",
                         "schema": {"mode": "fixed",
                                    "fields": ["content: str"]}},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(id="explode", node_type="transform",
                         plugin="line_explode",
                         input="explode_in", on_success="sink",
                         on_error="errors",
                         options={
                             "schema": {"mode": "flexible",
                                        "fields": ["content: str"]},
                             "source_field": "content",
                         }),
            ),
            outputs=(
                OutputSpec(name="sink", plugin="json",
                           options={"path": "out.json"}),
                OutputSpec(name="errors", plugin="json",
                           options={"path": "err.json"}),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()
        assert contracts == ()

    def test_undeclared_transform_producer_with_fail_policy_emits_error(self):
        # Real registered transform that does NOT declare output_semantics:
        # `passthrough` (src/elspeth/plugins/transforms/passthrough.py:39).
        # passthrough → line_explode is exactly the "pass-through degrades
        # to UNKNOWN" case the design decision documents.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            source=SourceSpec(
                plugin="csv", on_success="pt_in",
                options={"path": "x.csv",
                         "schema": {"mode": "fixed",
                                    "fields": ["content: str"]}},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="pt", node_type="transform", plugin="passthrough",
                    input="pt_in", on_success="explode_in", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["content: str"]},
                    },
                ),
                NodeSpec(
                    id="explode", node_type="transform", plugin="line_explode",
                    input="explode_in", on_success="sink", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["content: str"]},
                        "source_field": "content",
                    },
                ),
            ),
            outputs=(
                OutputSpec(name="sink", plugin="json",
                           options={"path": "out.json"}),
                OutputSpec(name="errors", plugin="json",
                           options={"path": "err.json"}),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)

        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.UNKNOWN
        assert contracts[0].consumer_plugin == "line_explode"
        assert contracts[0].producer_plugin == "passthrough"

        # FAIL policy → UNKNOWN producer fails.
        assert len(errors) == 1
        assert "no semantic facts" in errors[0].message.lower() \
            or "undeclared" in errors[0].message.lower()
        assert errors[0].component == "node:explode"

    def test_gate_between_producer_and_consumer_is_traversed(self):
        # Gates are STRUCTURAL — plugin=None and condition/routes carry
        # the routing logic. Verified via tests/unit/web/composer/
        # test_yaml_generator.py:72 which uses the same shape.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            source=SourceSpec(
                plugin="csv", on_success="src_in",
                options={"path": "x.csv",
                         "schema": {"mode": "fixed",
                                    "fields": ["url: str"]}},
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape", node_type="transform", plugin="web_scrape",
                    input="src_in", on_success="gate_in", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "fingerprint",
                        "format": "markdown",
                        "http": {"abuse_contact": "x@example.com",
                                 "scraping_reason": "t",
                                 "timeout": 5,
                                 "allowed_hosts": "public_only"},
                    },
                ),
                NodeSpec(
                    id="g", node_type="gate", plugin=None,
                    input="gate_in", on_success=None, on_error=None,
                    options={},
                    condition="row['content']",
                    routes={"yes": "explode_in", "no": "errors"},
                    fork_to=None, branches=None, policy=None, merge=None,
                ),
                NodeSpec(
                    id="explode", node_type="transform", plugin="line_explode",
                    input="explode_in", on_success="sink", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["content: str"]},
                        "source_field": "content",
                    },
                ),
            ),
            outputs=(
                OutputSpec(name="sink", plugin="json",
                           options={"path": "out.json"}),
                OutputSpec(name="errors", plugin="json",
                           options={"path": "err.json"}),
            ),
        )
        errors, contracts = validate_semantic_contracts(state)
        assert errors == ()  # markdown is line-compatible
        assert len(contracts) == 1
        assert contracts[0].outcome is SemanticOutcome.SATISFIED
        assert contracts[0].from_id == "scrape"  # walked through gate
        assert contracts[0].consumer_plugin == "line_explode"
        assert contracts[0].producer_plugin == "web_scrape"
```

**Note:** the undeclared-transform test uses the registered `passthrough` plugin (`src/elspeth/plugins/transforms/passthrough.py:39`); the gate test uses `plugin=None` per the structural-gate convention verified at `tests/unit/web/composer/test_yaml_generator.py:72`. Both are real fixtures that exercise production registry lookup.

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_semantic_validator.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the validator**

Create `src/elspeth/web/composer/_semantic_validator.py`:

```python
"""Generic semantic-contract validator.

For each consumer node with declared input_semantic_requirements(),
walks back to the effective upstream producer (using ProducerResolver),
asks the producer for its output_semantics(), and compares facts to
requirements per field. Emits structured SemanticEdgeContract records
and high-severity ValidationEntry errors on CONFLICT or on UNKNOWN +
FAIL policy.

This module reuses ProducerResolver — there must be exactly ONE
walk-back implementation across composer state.

Pass-through propagation is intentionally NOT performed in Phase 1.
A pass-through transform between a declared producer and a declared
consumer breaks the chain — the consumer sees outcome=UNKNOWN, which
combined with line_explode's FAIL policy means the chain rejects.
This is by design: it forces a real propagation API decision rather
than making an ad hoc choice.

Layer: L3.
"""

from __future__ import annotations

from collections.abc import Mapping

from elspeth.contracts.plugin_semantics import (
    FieldSemanticFacts,
    OutputSemanticDeclaration,
    SemanticEdgeContract,
    SemanticOutcome,
    UnknownSemanticPolicy,
    compare_semantic,
)
from elspeth.web.composer._producer_resolver import (
    ProducerEntry,
    ProducerResolver,
)
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    ValidationEntry,
)


def _is_config_probe_exception(exc: Exception) -> bool:
    """Same expected-error set used by _check_schema_contracts for probes."""
    from elspeth.plugins.infrastructure.config_base import PluginConfigError
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError
    from elspeth.plugins.infrastructure.templates import TemplateError
    from elspeth.plugins.infrastructure.validation import UnknownPluginTypeError

    if isinstance(exc, (PluginConfigError, PluginNotFoundError,
                        TemplateError, UnknownPluginTypeError)):
        return True
    return type(exc) is ValueError and \
        str(exc).startswith("Invalid configuration for transform ")


def _instantiate_consumer(node: NodeSpec):
    """Construct a consumer transform instance to read its requirements."""
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    if node.plugin is None:
        return None
    return get_shared_plugin_manager().create_transform(
        node.plugin, deep_thaw(node.options),
    )


def _instantiate_producer(producer: ProducerEntry):
    """Construct a producer transform/source instance to read its facts."""
    from elspeth.contracts.freeze import deep_thaw
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    if producer.plugin_name is None or producer.producer_id == "source":
        # Sources don't expose output_semantics() in Phase 1 — return None.
        # Phase 2 (post-Wardline) can extend BaseSource the same way.
        return None
    return get_shared_plugin_manager().create_transform(
        producer.plugin_name, deep_thaw(producer.options),
    )


def _safe_output_semantics(producer: ProducerEntry) -> OutputSemanticDeclaration | None:
    """Construct producer and read its semantics, tolerating draft-config errors.

    Returns None when:
    - Producer plugin has no instance (source — see _instantiate_producer)
    - Plugin construction fails with an expected draft/config probe error

    Unexpected exceptions PROPAGATE — they indicate a framework bug
    (per CLAUDE.md plugin-as-system-code policy: a plugin method that
    raises is a bug we MUST know about).
    """
    try:
        instance = _instantiate_producer(producer)
    except Exception as exc:
        if _is_config_probe_exception(exc):
            return None
        raise

    if instance is None:
        return None
    return instance.output_semantics()


def _find_producer_facts(
    declaration: OutputSemanticDeclaration,
    field_name: str,
) -> FieldSemanticFacts | None:
    for facts in declaration.fields:
        if facts.field_name == field_name:
            return facts
    return None


def validate_semantic_contracts(
    state: CompositionState,
) -> tuple[tuple[ValidationEntry, ...], tuple[SemanticEdgeContract, ...]]:
    """Validate semantic contracts across the composition.

    Returns (errors, contracts):
    - errors: ValidationEntry records suitable for ValidationSummary.errors
    - contracts: SemanticEdgeContract records for ValidationSummary.semantic_contracts
    """
    errors: list[ValidationEntry] = []
    contracts: list[SemanticEdgeContract] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    sink_names = frozenset(output.name for output in state.outputs)
    resolver = ProducerResolver.build(
        source=state.source, nodes=state.nodes, sink_names=sink_names,
    )

    for node in state.nodes:
        if node.node_type != "transform" or node.plugin is None:
            continue

        # CONSUMER probe failures must NOT be silently tolerated. The
        # consumer is the entry point — if it can't construct, the
        # test/composition has a bug we MUST surface, otherwise the
        # validator silently skips the case it's meant to check.
        # Producer probes are tolerant (separate _safe_output_semantics).
        consumer = _instantiate_consumer(node)
        if consumer is None:
            continue
        requirements = consumer.input_semantic_requirements()
        if not requirements.fields:
            continue

        upstream_producer = resolver.walk_to_real_producer(node.input)
        if upstream_producer is None:
            # No declared producer (coalesce, ambiguous, missing).
            # Schema-contract layer handles missing-field cases; semantic
            # layer treats this as unknown for any FAIL-policy requirement.
            for req in requirements.fields:
                edge_key = ("?source", node.id, req.field_name, req.field_name)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                contract = SemanticEdgeContract(
                    from_id="?", to_id=node.id,
                    consumer_plugin=node.plugin, producer_plugin=None,
                    producer_field=req.field_name,
                    consumer_field=req.field_name,
                    producer_facts=None,
                    requirement=req,
                    outcome=SemanticOutcome.UNKNOWN,
                )
                contracts.append(contract)
                if req.unknown_policy is UnknownSemanticPolicy.FAIL:
                    errors.append(ValidationEntry(
                        f"node:{node.id}",
                        (
                            f"Semantic contract: '{node.plugin}' node '{node.id}' "
                            f"requires field '{req.field_name}' "
                            f"({req.requirement_code}) but the upstream producer is "
                            f"undeclared (coalesce, ambiguous, or unreachable)."
                        ),
                        req.severity,
                    ))
            continue

        # SOURCE → TRANSFORM edges are out of scope for Phase 1.
        # BaseSource has no output_semantics() API yet; treating source
        # producers as UNKNOWN under FAIL would break every working
        # csv → line_explode pipeline. Skip these edges entirely;
        # extending BaseSource is a separate plan.
        if upstream_producer.producer_id == "source":
            continue

        producer_decl = _safe_output_semantics(upstream_producer)

        for req in requirements.fields:
            facts = _find_producer_facts(producer_decl, req.field_name) \
                if producer_decl is not None else None
            outcome = compare_semantic(facts, req)

            edge_key = (upstream_producer.producer_id, node.id,
                        req.field_name, req.field_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            contracts.append(SemanticEdgeContract(
                from_id=upstream_producer.producer_id,
                to_id=node.id,
                consumer_plugin=node.plugin,
                producer_plugin=upstream_producer.plugin_name,
                producer_field=req.field_name,
                consumer_field=req.field_name,
                producer_facts=facts,
                requirement=req,
                outcome=outcome,
            ))

            if outcome is SemanticOutcome.CONFLICT:
                producer_label = (upstream_producer.plugin_name
                                  if upstream_producer.plugin_name is not None
                                  else "(unknown plugin)")
                facts_kind = facts.content_kind.value if facts else "unknown"
                facts_framing = facts.text_framing.value if facts else "unknown"
                # Generic diagnostic: name producer, consumer, field,
                # observed producer facts, and the requirement_code.
                # Do NOT enumerate accepted enum sets — that prints
                # values like "markdown" / "newline_framed" which read
                # as fix prose ("use markdown"). Fix prose belongs in
                # PluginAssistance, addressed by requirement_code.
                errors.append(ValidationEntry(
                    f"node:{node.id}",
                    (
                        f"Semantic contract violation: '{upstream_producer.producer_id}' "
                        f"-> '{node.id}'. Consumer ({node.plugin}) requires field "
                        f"'{req.field_name}' to satisfy {req.requirement_code}, "
                        f"but producer ({producer_label}) declares "
                        f"content_kind={facts_kind}, text_framing={facts_framing}."
                    ),
                    req.severity,
                ))
            elif outcome is SemanticOutcome.UNKNOWN \
                    and req.unknown_policy is UnknownSemanticPolicy.FAIL:
                producer_label = (upstream_producer.plugin_name
                                  if upstream_producer.plugin_name is not None
                                  else "(unknown plugin)")
                errors.append(ValidationEntry(
                    f"node:{node.id}",
                    (
                        f"Semantic contract: '{node.plugin}' node '{node.id}' "
                        f"requires field '{req.field_name}' "
                        f"({req.requirement_code}) but upstream producer "
                        f"'{upstream_producer.producer_id}' ({producer_label}) "
                        f"declares no semantic facts for that field. "
                        f"Producers semantically feeding this consumer must "
                        f"declare output_semantics()."
                    ),
                    req.severity,
                ))

    return tuple(errors), tuple(contracts)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_semantic_validator.py -v`
Expected: All core tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/_semantic_validator.py \
        tests/unit/web/composer/test_semantic_validator.py
git commit -m "feat(composer): generic semantic-contract validator

Walks producer/consumer graph via shared ProducerResolver, asks each
plugin for declared facts/requirements, compares them, emits structured
SemanticEdgeContract records + high-severity ValidationEntry errors
on CONFLICT or FAIL-policy UNKNOWN.

Pass-through degrades to UNKNOWN by design — Phase 1 has no
transform_field_semantics propagation API."
```

---

### Task 3.2: Cross-validator parity test (old vs new on the Wardline shape)

**Files:**
- Modify: `tests/unit/web/composer/test_semantic_validator.py`

**Goal:** Protect the Phase 0–5 window when both `validate_transform_framing_contracts` and `validate_semantic_contracts` are active. They MUST agree on every Wardline-shape input. Without this test, divergence detected at Phase 6 deletion forces backing out four phases.

- [ ] **Step 1: Append the parity test**

Append to `tests/unit/web/composer/test_semantic_validator.py`:

```python
class TestParityWithLegacyFramingValidator:
    """The old hardcoded validator and the new generic one MUST agree on
    Wardline-shape pipelines while both exist (Phase 0–5).

    Deleted in Phase 6 along with the legacy validator.
    """

    @pytest.mark.parametrize("text_separator,scrape_format,expect_blocked", [
        (" ", "text", True),       # Wardline regression case
        ("\t", "text", True),      # tab is not newline
        ("\n", "text", False),     # explicitly newline-framed
        (" \n ", "text", False),   # contains newline
        (" ", "markdown", False),  # markdown is line-compatible
        (" ", "raw", True),        # raw HTML is not text — fails semantic
    ])
    def test_parity(self, text_separator, scrape_format, expect_blocked):
        from elspeth.web.composer.state import (
            validate_transform_framing_contracts,
        )

        state = _wardline_state(text_separator=text_separator,
                                scrape_format=scrape_format)

        legacy_errors = validate_transform_framing_contracts(state.nodes)
        semantic_errors, _ = validate_semantic_contracts(state)

        # Note: the legacy validator only handles format=text; format=raw
        # is a NEW case the semantic validator catches that the legacy
        # validator never did. For format=raw we EXPECT divergence and
        # assert only the semantic side blocks.
        if scrape_format == "raw":
            assert legacy_errors == ()
            assert len(semantic_errors) == 1
            return

        # For format=text and format=markdown, parity is required.
        legacy_blocked = bool(legacy_errors)
        semantic_blocked = bool(semantic_errors)
        assert legacy_blocked == expect_blocked
        assert semantic_blocked == expect_blocked
        assert legacy_blocked == semantic_blocked
```

- [ ] **Step 2: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_semantic_validator.py::TestParityWithLegacyFramingValidator -v`
Expected: All cases pass.

If parity fails for a `format=text` case, investigate: the semantic validator's `_build_web_scrape_output_semantics` mapping or the new comparator has a bug. Do NOT proceed to Phase 4 with parity broken.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_semantic_validator.py
git commit -m "test(composer): cross-validator parity test for legacy + semantic framing

Protects the Phase 0-5 window when both validators run. format=raw is
a new case only the semantic validator catches — explicitly asserted
as expected divergence. Other format=text and format=markdown shapes
must agree."
```

---

### Task 3.3: Secret-leakage sentinel test

**Files:**
- Modify: `tests/unit/plugins/transforms/test_web_scrape.py`
- Modify: `tests/unit/plugins/transforms/test_line_explode.py`
- Modify: `tests/unit/web/composer/test_semantic_validator.py`

**Goal:** The plan asserts that assistance and validation diagnostics must not leak secrets. This test feeds a sentinel string into plugin config and asserts the sentinel never appears in: `OutputSemanticDeclaration` field, `PluginAssistance` fields, `SemanticEdgeContract` serialized form, `ValidationEntry.message`.

- [ ] **Step 1: Write the sentinel test for `web_scrape`**

Append to `tests/unit/plugins/transforms/test_web_scrape.py`:

```python
class TestWebScrapeSecretLeakage:
    SENTINEL = "PASSWORD_SENTINEL_x9q7r3"

    def test_sentinel_url_not_in_output_semantics_or_assistance(self):
        from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
        from elspeth.plugins.transforms.web_scrape import WebScrapeTransform

        ws = get_shared_plugin_manager().create_transform("web_scrape", {
            "schema": {"mode": "flexible", "fields": ["url: str"]},
            "required_input_fields": ["url"],
            "url_field": "url",
            "content_field": f"content_{self.SENTINEL}",  # field name SHOULD appear
            "fingerprint_field": "fingerprint",
            "format": "text",
            "text_separator": " ",
            "http": {
                "abuse_contact": f"x+{self.SENTINEL}@example.com",
                "scraping_reason": f"reason-{self.SENTINEL}",
                "timeout": 5,
                "allowed_hosts": "public_only",
            },
        })

        # Output semantics: the configured content_field name DOES include
        # the sentinel — that's not a leak, the user wrote that field name.
        # What MUST NOT appear: the abuse_contact email, the scraping_reason.
        decl = ws.output_semantics()
        decl_repr = repr(decl)
        assert f"x+{self.SENTINEL}" not in decl_repr
        assert f"reason-{self.SENTINEL}" not in decl_repr

        # Assistance is class-level, no instance state — but verify anyway.
        assistance = WebScrapeTransform.get_agent_assistance(
            issue_code="web_scrape.content.compact_text",
        )
        assistance_repr = repr(assistance)
        assert self.SENTINEL not in assistance_repr
```

- [ ] **Step 2: Write the sentinel test for the validator → ValidationEntry message and SemanticEdgeContract**

Append to `tests/unit/web/composer/test_semantic_validator.py`:

```python
class TestSemanticValidatorSecretLeakage:
    SENTINEL = "PASSWORD_SENTINEL_x9q7r3"

    def test_sentinel_does_not_appear_in_validator_output(self):
        # Build a Wardline state with sentinel in non-field-name options.
        state = CompositionState(
            metadata=PipelineMetadata(name="t"),
            source=SourceSpec(
                plugin="csv", on_success="scrape_in",
                options={
                    "path": f"data/{self.SENTINEL}/url.csv",
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape", node_type="transform", plugin="web_scrape",
                    input="scrape_in", on_success="explode_in",
                    on_error="errors",
                    options={
                        "schema": {"mode": "flexible", "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "fingerprint",
                        "format": "text",
                        "text_separator": " ",
                        "http": {
                            "abuse_contact": f"x+{self.SENTINEL}@example.com",
                            "scraping_reason": f"reason-{self.SENTINEL}",
                            "timeout": 5,
                            "allowed_hosts": "public_only",
                        },
                    },
                ),
                NodeSpec(
                    id="explode", node_type="transform", plugin="line_explode",
                    input="explode_in", on_success="sink", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["content: str"]},
                        "source_field": "content",
                    },
                ),
            ),
            outputs=(
                OutputSpec(name="sink", plugin="json",
                           options={"path": f"out-{self.SENTINEL}.json"}),
                OutputSpec(name="errors", plugin="json",
                           options={"path": "err.json"}),
            ),
        )

        errors, contracts = validate_semantic_contracts(state)
        for entry in errors:
            assert self.SENTINEL not in entry.message, \
                f"Sentinel leaked in error message: {entry.message!r}"
            assert self.SENTINEL not in entry.component
        for contract in contracts:
            assert self.SENTINEL not in repr(contract)
```

- [ ] **Step 3: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/plugins/transforms/test_web_scrape.py::TestWebScrapeSecretLeakage tests/unit/web/composer/test_semantic_validator.py::TestSemanticValidatorSecretLeakage -v`
Expected: Both pass.

If a test fails, the leak is real — fix the implementation, do not weaken the test.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/plugins/transforms/test_web_scrape.py \
        tests/unit/web/composer/test_semantic_validator.py
git commit -m "test: secret-leakage sentinel for semantic facts/assistance/diagnostics

Asserts that values from plugin config (URLs, contact emails, reasons,
file paths) never appear in output_semantics, get_agent_assistance,
SemanticEdgeContract, or ValidationEntry messages."
```

---

## Phase 4 — Wire into surfaces

Connect the validator to `CompositionState.validate()`, `/validate`, `/execute`, MCP, and `ToolResult`. Introduce `SemanticContractViolationError` for the structured `/execute` path.

### Task 4.1: Add `semantic_contracts` to `ValidationSummary`

**Files:**
- Modify: `src/elspeth/web/composer/state.py:281`

- [ ] **Step 1: Find existing serialization tests for `ValidationSummary`**

```bash
grep -rn 'edge_contracts' tests/unit/web/composer/ | head -10
```

- [ ] **Step 2: Add the field**

In `state.py:281`, modify `ValidationSummary`:

```python
@dataclass(frozen=True, slots=True)
class ValidationSummary:
    """Stage 1 validation result.

    errors block execution. warnings are advisory but actionable.
    suggestions are optional improvements. edge_contracts shows
    per-edge schema contract check results. semantic_contracts shows
    per-edge semantic contract check results (Phase 1: line_explode +
    web_scrape only). All are tuples for structured component
    attribution.
    """

    is_valid: bool
    errors: tuple[ValidationEntry, ...]
    warnings: tuple[ValidationEntry, ...] = ()
    suggestions: tuple[ValidationEntry, ...] = ()
    edge_contracts: tuple[EdgeContract, ...] = ()
    semantic_contracts: tuple["SemanticEdgeContract", ...] = ()
```

Add the import at the top of `state.py`:

```python
from elspeth.contracts.plugin_semantics import SemanticEdgeContract
```

- [ ] **Step 3: Run all state tests to confirm no regressions from the new optional field**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests pass; the new field defaults to `()` and is invisible to existing assertions.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/state.py
git commit -m "feat(composer): add semantic_contracts to ValidationSummary"
```

---

### Task 4.2: Wire `validate_semantic_contracts` into `CompositionState.validate()`

**Files:**
- Modify: `src/elspeth/web/composer/state.py:1509` (the existing line that calls `validate_transform_framing_contracts`)

- [ ] **Step 1: Write a failing integration test**

Append to `tests/unit/web/composer/test_state.py`:

```python
class TestCompositionStateValidateEmitsSemanticContracts:
    def test_compact_wardline_yields_semantic_error_in_validate(self):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ", scrape_format="text")
        result = state.validate()

        assert result.is_valid is False
        # Wardline-shape with compact text: at least one error tagged with
        # node:explode reflecting the semantic contract violation.
        explode_errors = [e for e in result.errors if e.component == "node:explode"]
        assert any("Semantic contract" in e.message or "line_explode" in e.message
                   for e in explode_errors)

        # And a SemanticEdgeContract record on the summary.
        assert len(result.semantic_contracts) == 1
        assert result.semantic_contracts[0].outcome.value == "conflict"

    def test_passing_wardline_yields_satisfied_contract(self):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator="\n", scrape_format="text")
        result = state.validate()
        # Other validation may pass or fail; what we assert is that
        # the semantic contract is SATISFIED.
        assert any(c.outcome.value == "satisfied"
                   for c in result.semantic_contracts)
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestCompositionStateValidateEmitsSemanticContracts -v`
Expected: Fails — `semantic_contracts` is empty.

- [ ] **Step 3: Wire it in**

In `state.py`, find the `CompositionState.validate()` method (around line 1509 where `validate_transform_framing_contracts` is called). Replace that single line with:

```python
        # Legacy framing check (deleted in Phase 6).
        errors.extend(validate_transform_framing_contracts(self.nodes))

        # Generic semantic-contract check.
        from elspeth.web.composer._semantic_validator import validate_semantic_contracts
        semantic_errors, semantic_contracts = validate_semantic_contracts(self)
        errors.extend(semantic_errors)
```

Then find where `ValidationSummary(...)` is constructed at the end of `validate()` and pass `semantic_contracts=semantic_contracts`. (Search for `ValidationSummary(` inside `validate` to find the construction site.)

- [ ] **Step 4: Run, verify passes**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py::TestCompositionStateValidateEmitsSemanticContracts -v`
Expected: Both new tests pass.

- [ ] **Step 5: Run full state test suite**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_state.py -v`
Expected: All tests pass. Note: during Phase 0–5, both validators emit errors for the Wardline regression — the test suite should tolerate duplicate errors (one from each validator) on Wardline-shape composition states. If existing tests assert exactly one error count on Wardline cases, they need updating to tolerate `>= 1` until Phase 6 deletion.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "feat(composer): wire validate_semantic_contracts into CompositionState.validate

Both legacy framing validator and new semantic validator emit during
Phase 0-5; legacy is deleted in Phase 6. ValidationSummary.semantic_contracts
carries structured per-edge results."
```

---

### Task 4.3: Replace `_CHECK_TRANSFORM_FRAMING` with `_CHECK_SEMANTIC_CONTRACTS` in `/validate`

**Files:**
- Modify: `src/elspeth/web/execution/validation.py:44`, `:53`, `:274–301`
- Modify: `tests/unit/web/execution/test_validation.py:344, 359, 459` (string literals)

- [ ] **Step 1: Write a failing test for the new check name + structured records**

Append to `tests/unit/web/execution/test_validation.py`:

```python
class TestValidatePipelineSemanticContracts:
    def _make_state(self, text_separator=" "):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state
        return _wardline_state(text_separator=text_separator)

    def test_compact_text_fails_with_semantic_contracts_check_name(self,
                                                                    web_settings,
                                                                    yaml_generator):
        state = self._make_state(text_separator=" ")
        result = validate_pipeline(state, web_settings, yaml_generator)
        assert result.is_valid is False
        assert _check(result, "semantic_contracts").passed is False

    def test_newline_text_passes_semantic_contracts_check(self,
                                                          web_settings,
                                                          yaml_generator):
        state = self._make_state(text_separator="\n")
        result = validate_pipeline(state, web_settings, yaml_generator)
        # Subsequent checks may still fail (depends on fixture); we only
        # assert semantic_contracts itself passed.
        assert _check(result, "semantic_contracts").passed is True
```

(Use whatever fixtures the existing tests at line 240+ use; adapt the helper imports.)

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_validation.py::TestValidatePipelineSemanticContracts -v`
Expected: KeyError or "check not found" — the `semantic_contracts` check doesn't exist yet.

- [ ] **Step 3: Replace the constant and the check logic**

In `src/elspeth/web/execution/validation.py`:

Line 32 — replace import:
```python
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
```
(Stop importing `validate_transform_framing_contracts`.)

Line 44 — replace constant:
```python
_CHECK_SEMANTIC_CONTRACTS = "semantic_contracts"
```

Line 53 — update `_ALL_CHECKS`:
```python
_ALL_CHECKS = [
    _CHECK_PATH_ALLOWLIST,
    _CHECK_SECRET_REFS,
    _CHECK_SEMANTIC_CONTRACTS,
    _CHECK_SETTINGS,
    _CHECK_PLUGINS,
    _CHECK_GRAPH,
    _CHECK_SCHEMA,
]
```

Lines 274–301 — replace the framing block:
```python
    semantic_errors, semantic_contracts = validate_semantic_contracts(state)
    if semantic_errors:
        checks.append(
            ValidationCheck(
                name=_CHECK_SEMANTIC_CONTRACTS,
                passed=False,
                detail="Semantic contract check failed",
            )
        )
        for entry in semantic_errors:
            # entry.message already names plugins, fields, requirement code.
            # Suggestion is plugin-owned — fetch from PluginAssistance.
            errors.append(
                ValidationError(
                    component_id=entry.component.removeprefix("node:"),
                    component_type="transform",
                    message=entry.message,
                    suggestion=_assistance_suggestion_for(entry, semantic_contracts),
                )
            )
        checks.extend(_skipped_checks(_CHECK_SEMANTIC_CONTRACTS))
        return ValidationResult(is_valid=False, checks=checks, errors=errors)

    checks.append(
        ValidationCheck(
            name=_CHECK_SEMANTIC_CONTRACTS,
            passed=True,
            detail=(
                f"All {len(semantic_contracts)} semantic contract(s) satisfied"
                if semantic_contracts else "No semantic contracts to check"
            ),
        )
    )
```

Add a helper above `validate_pipeline`:
```python
def _assistance_suggestion_for(
    entry: Any,
    contracts: tuple[Any, ...],
) -> str | None:
    """Look up plugin-owned guidance for a semantic error.

    Uses SemanticEdgeContract.consumer_plugin (and producer_plugin as
    a fallback) to address a SPECIFIC plugin class. Looping every
    registered transform and returning the first match was registry-
    order dependent — fixed by carrying the plugin names on the
    contract (Phase 1 Task 1.3).
    """
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    component_id = entry.component.removeprefix("node:")
    matching = next((c for c in contracts if c.to_id == component_id), None)
    if matching is None:
        return None

    manager = get_shared_plugin_manager()
    issue_code = matching.requirement.requirement_code

    # Consumer plugin owns the requirement, so it's the authoritative
    # source for guidance about the requirement_code. Verified method
    # name: get_transform_by_name (manager.py:183), NOT get_transform_class.
    consumer_cls = manager.get_transform_by_name(matching.consumer_plugin)
    consumer_assistance = consumer_cls.get_agent_assistance(issue_code=issue_code)
    if consumer_assistance is not None:
        return consumer_assistance.summary

    # Producer plugin may also publish guidance for the producer-side
    # fact_code. The validator could attach that fact_code on the
    # contract in a later phase; for now, only consumer assistance is
    # surfaced as suggestion text.
    return None
```

- [ ] **Step 4: Update existing tests that key on `"transform_framing"` string**

Edit `tests/unit/web/execution/test_validation.py` lines 344, 359, 459: replace `"transform_framing"` with `"semantic_contracts"`. Also rename `class TestValidatePipelineTransformFraming` to `class TestValidatePipelineSemanticContractsLegacy` or merge into the new class; either way no test should still key on the old name.

- [ ] **Step 5: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_validation.py -v`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/execution/validation.py \
        tests/unit/web/execution/test_validation.py
git commit -m "feat(/validate): replace transform_framing with semantic_contracts check

Check name surfaces in ValidationCheck.name (rendered in the frontend
banner). Suggestion text now comes from plugin-owned assistance keyed
by requirement_code, not from a hardcoded web_scrape string."
```

---

### Task 4.3b: Extend `ValidationResult` Pydantic schema with `semantic_contracts`

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py:53` (the `ValidationResult` model)

**Goal:** The `ValidationResult` Pydantic response model is `_StrictResponse` with `extra="forbid"` (`schemas.py:33`). Adding `semantic_contracts` to validator output without declaring the field on the schema will cause Pydantic to drop or reject the field when constructing the HTTP response. This task adds the response model and the field so the wire payload actually carries semantic contract data.

- [ ] **Step 1: Write a failing serialization test**

Append to `tests/unit/web/execution/test_schemas.py` (or wherever schema tests live; create the file if not):

```python
def test_validation_result_accepts_semantic_contracts():
    from elspeth.web.execution.schemas import (
        SemanticEdgeContractResponse,
        ValidationCheck,
        ValidationResult,
    )

    contract = SemanticEdgeContractResponse(
        from_id="scrape", to_id="explode",
        consumer_plugin="line_explode", producer_plugin="web_scrape",
        producer_field="content", consumer_field="content",
        outcome="conflict",
        requirement_code="line_explode.source_field.line_framed_text",
    )
    result = ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name="semantic_contracts",
                                passed=False, detail="failed")],
        errors=[],
        semantic_contracts=[contract],
    )
    payload = result.model_dump()
    assert payload["semantic_contracts"][0]["outcome"] == "conflict"
    assert payload["semantic_contracts"][0]["consumer_plugin"] == "line_explode"


def test_validation_result_rejects_unknown_field():
    # Confirms extra="forbid" still applies — the new field doesn't
    # accidentally weaken strict-mode enforcement.
    from pydantic import ValidationError
    from elspeth.web.execution.schemas import ValidationResult

    with pytest.raises(ValidationError):
        ValidationResult(
            is_valid=True, checks=[], errors=[],
            invented_extra_field="nope",
        )
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py -v`
Expected: ImportError on `SemanticEdgeContractResponse`.

- [ ] **Step 3: Add the response model and field**

In `src/elspeth/web/execution/schemas.py`, near `ValidationError` (line 47) add:

```python
class SemanticEdgeContractResponse(_StrictResponse):
    """Per-edge semantic-contract result for HTTP serialization.

    Field set mirrors composer_mcp/server.py::_SemanticEdgeContractPayload
    so MCP and HTTP clients receive identical shapes.
    """

    from_id: str
    to_id: str
    consumer_plugin: str
    producer_plugin: str | None
    producer_field: str
    consumer_field: str
    outcome: str  # SemanticOutcome value: "satisfied" | "conflict" | "unknown"
    requirement_code: str
```

Then update `ValidationResult` (line 53):

```python
class ValidationResult(_StrictResponse):
    """Result of dry-run validation against real engine code."""

    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[ValidationError]
    semantic_contracts: list[SemanticEdgeContractResponse] = []
```

- [ ] **Step 4: Update `validate_pipeline` to populate the new field**

In `src/elspeth/web/execution/validation.py`, the success branch of the semantic check should attach `semantic_contracts` to the returned `ValidationResult`. In the failure branch too — diagnostics carry the same structured payload.

```python
    return ValidationResult(
        is_valid=False, checks=checks, errors=errors,
        semantic_contracts=[
            SemanticEdgeContractResponse(
                from_id=c.from_id, to_id=c.to_id,
                consumer_plugin=c.consumer_plugin,
                producer_plugin=c.producer_plugin,
                producer_field=c.producer_field,
                consumer_field=c.consumer_field,
                outcome=c.outcome.value,
                requirement_code=c.requirement.requirement_code,
            )
            for c in semantic_contracts
        ],
    )
```

Apply the same population in the success-return paths so callers always see the satisfied contracts (operators want to confirm "yes, semantic_contracts: 1 satisfied" in the UI banner).

- [ ] **Step 5: Run + commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_schemas.py tests/unit/web/execution/test_validation.py -v`
Expected: All pass.

```bash
git add src/elspeth/web/execution/schemas.py \
        src/elspeth/web/execution/validation.py \
        tests/unit/web/execution/test_schemas.py
git commit -m "feat(/validate): add SemanticEdgeContractResponse to Pydantic schema

extra='forbid' would otherwise drop the new field at the response
boundary. Field shape mirrors MCP _SemanticEdgeContractPayload so
HTTP and MCP clients receive identical structures."
```

---

### Task 4.4: Frontend `ValidationResult` interface — add `semantic_contracts`

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/index.ts:264-270`

**Goal:** The new field flows through the API; the frontend type must declare it so the field is not silently dropped at the type layer.

- [ ] **Step 1: Add the field**

In `src/elspeth/web/frontend/src/types/index.ts`:

```typescript
export interface SemanticEdgeContract {
  from_id: string;
  to_id: string;
  consumer_plugin: string;
  producer_plugin: string | null;
  producer_field: string;
  consumer_field: string;
  outcome: "satisfied" | "conflict" | "unknown";
  requirement_code: string;
}

export interface ValidationResult {
  is_valid: boolean;
  summary?: string;
  checks: ValidationCheck[];
  errors: ValidationError[];
  warnings?: ValidationWarning[];
  semantic_contracts?: SemanticEdgeContract[];
}
```

- [ ] **Step 2: Verify the frontend builds**

```bash
cd src/elspeth/web/frontend && npm run build
```

Expected: Build succeeds. (If the project uses `tsc --noEmit` instead, run that.)

- [ ] **Step 3: Commit**

```bash
git add src/elspeth/web/frontend/src/types/index.ts
git commit -m "feat(frontend): declare semantic_contracts on ValidationResult"
```

---

### Task 4.5: `SemanticContractViolationError` exception + `/execute` integration

**Files:**
- Create: `src/elspeth/web/execution/errors.py`
- Modify: `src/elspeth/web/execution/service.py:46`, `:325-327`
- Modify: `tests/unit/web/execution/test_service.py:1974+`

**Goal:** Replace the structure-discarding `raise ValueError("; ".join(...))` at `service.py:325-327` with a richer exception that carries the structured `ValidationEntry` records and `SemanticEdgeContract` records. Both `/validate` and `/execute` consume the same shared helper; only the rendering differs.

- [ ] **Step 1: Write failing tests for the exception type**

Create `tests/unit/web/execution/test_errors.py`:

```python
"""Tests for execution-layer error types."""

from __future__ import annotations

import pytest

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    FieldSemanticFacts,
    FieldSemanticRequirement,
    SemanticEdgeContract,
    SemanticOutcome,
    TextFraming,
    UnknownSemanticPolicy,
)
from elspeth.web.composer.state import ValidationEntry
from elspeth.web.execution.errors import SemanticContractViolationError


def _entry(node_id: str = "x") -> ValidationEntry:
    return ValidationEntry(f"node:{node_id}", "msg", "high")


def _contract() -> SemanticEdgeContract:
    facts = FieldSemanticFacts("c", ContentKind.PLAIN_TEXT,
                               text_framing=TextFraming.COMPACT,
                               fact_code="t.c.compact")
    req = FieldSemanticRequirement(
        field_name="c",
        accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
        accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
        requirement_code="t.c.req",
        unknown_policy=UnknownSemanticPolicy.FAIL,
    )
    return SemanticEdgeContract(
        from_id="a", to_id="b",
        consumer_plugin="line_explode", producer_plugin="web_scrape",
        producer_field="c", consumer_field="c",
        producer_facts=facts, requirement=req,
        outcome=SemanticOutcome.CONFLICT,
    )


class TestSemanticContractViolationError:
    def test_carries_structured_payload(self):
        entries = (_entry("x"),)
        contracts = (_contract(),)
        exc = SemanticContractViolationError(
            entries=entries, contracts=contracts,
        )
        assert exc.entries == entries
        assert exc.contracts == contracts

    def test_str_summarizes_entries(self):
        entries = (
            _entry("x"),
            ValidationEntry("node:y", "second message", "high"),
        )
        exc = SemanticContractViolationError(entries=entries, contracts=())
        message = str(exc)
        assert "msg" in message
        assert "second message" in message

    def test_is_value_error_subclass_for_existing_callers(self):
        # /execute callers that catch ValueError must continue to work
        # during migration. Subclass of ValueError keeps that contract.
        exc = SemanticContractViolationError(entries=(_entry(),), contracts=())
        assert isinstance(exc, ValueError)
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_errors.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the exception**

Create `src/elspeth/web/execution/errors.py`:

```python
"""Structured execution-layer exceptions.

SemanticContractViolationError carries the same structured records as
the /validate endpoint surfaces, so callers of /execute that need to
render structured errors (frontend banner, MCP error payload) can do
so instead of falling back to string parsing.

Subclassing ValueError preserves backward compatibility for any caller
catching ValueError today; new callers should catch the specific type
to access entries and contracts.
"""

from __future__ import annotations

from elspeth.contracts.plugin_semantics import SemanticEdgeContract
from elspeth.web.composer.state import ValidationEntry


class SemanticContractViolationError(ValueError):
    """Raised when /execute pre-run semantic validation rejects the pipeline.

    Subclasses ValueError so existing `except ValueError` paths still
    catch it; new code should catch `SemanticContractViolationError`
    directly to access the structured payload.
    """

    def __init__(
        self,
        *,
        entries: tuple[ValidationEntry, ...],
        contracts: tuple[SemanticEdgeContract, ...],
    ) -> None:
        self.entries = entries
        self.contracts = contracts
        message = "; ".join(entry.message for entry in entries)
        super().__init__(message)
```

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_errors.py -v`
Expected: All 3 tests pass.

- [ ] **Step 5: Update `/execute` to raise the structured exception**

In `src/elspeth/web/execution/service.py`:

Line 46 — replace import:
```python
from elspeth.web.composer._semantic_validator import validate_semantic_contracts
from elspeth.web.execution.errors import SemanticContractViolationError
```
(Stop importing `validate_transform_framing_contracts`.)

Lines 325–327 — replace:
```python
        semantic_errors, semantic_contracts = validate_semantic_contracts(composition_state)
        if semantic_errors:
            raise SemanticContractViolationError(
                entries=semantic_errors, contracts=semantic_contracts,
            )
```

- [ ] **Step 6: Update `/execute` tests**

In `tests/unit/web/execution/test_service.py`, find `class TestTransformFramingRestriction` (around line 1974) and update assertions. Existing tests probably assert `pytest.raises(ValueError, match="...")` — this still works because `SemanticContractViolationError` IS a `ValueError`. Add a new assertion that the structured payload is accessible:

```python
class TestExecuteSemanticContractViolation:
    def test_compact_text_raises_structured_exception(self,
                                                       execute_request_factory):
        from tests.unit.web.composer.test_semantic_validator import _wardline_state
        from elspeth.web.execution.errors import SemanticContractViolationError

        state = _wardline_state(text_separator=" ")

        with pytest.raises(SemanticContractViolationError) as excinfo:
            # adapt to whatever the real /execute test pattern uses
            ...

        exc = excinfo.value
        assert len(exc.entries) >= 1
        assert any("Semantic contract" in e.message for e in exc.entries)
        assert any(c.outcome.value == "conflict" for c in exc.contracts)
```

(Adapt to the existing test pattern in `test_service.py`.)

Old tests that key on `match="line_explode"` substring still work because the new diagnostic also names `line_explode`.

- [ ] **Step 7: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_service.py -v`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/execution/errors.py \
        src/elspeth/web/execution/service.py \
        tests/unit/web/execution/test_errors.py \
        tests/unit/web/execution/test_service.py
git commit -m "feat(/execute): raise SemanticContractViolationError with structured payload

Subclass of ValueError preserves existing 'except ValueError' callers.
New callers can access .entries and .contracts for structured rendering."
```

---

### Task 4.5b: `/execute` route handler renders structured payload

**Files:**
- Modify: `src/elspeth/web/execution/routes.py:336` (add a `SemanticContractViolationError` branch BEFORE the bare `except ValueError`)
- Test: `tests/unit/web/execution/test_routes.py` (or wherever the route tests live)

**Goal:** `routes.py:336` currently catches a bare `ValueError` and renders `detail=str(exc)`. `SemanticContractViolationError` IS a `ValueError`, so without a dedicated handler the structured `entries`/`contracts` payload is dropped at the HTTP boundary. This task adds a handler that surfaces a 422 with the structured payload BEFORE the existing 404-mapping `ValueError` branch.

- [ ] **Step 1: Write failing test**

```python
def test_execute_returns_422_with_structured_semantic_payload(client, ...):
    # Create a session whose composition state will trigger a semantic
    # contract violation (Wardline shape with text_separator=' ').
    ...
    response = client.post(f"/api/sessions/{session_id}/execute", json={...})
    assert response.status_code == 422
    body = response.json()
    assert "semantic_contracts" in body["detail"]
    assert body["detail"]["semantic_contracts"][0]["outcome"] == "conflict"
    assert body["detail"]["semantic_contracts"][0]["consumer_plugin"] == "line_explode"
```

(Adapt to the project's existing route-test fixtures.)

- [ ] **Step 2: Run, verify failing**

Expected: 404 with string detail (the bare ValueError path).

- [ ] **Step 3: Add the handler before the existing `except ValueError` branch**

In `routes.py`, find the `except ValueError as exc` at line 336 inside the `/execute` endpoint and add a more-specific handler immediately above it:

```python
        except SemanticContractViolationError as exc:
            # Structured 422 with the same payload shape as /validate.
            # Status 422 (Unprocessable Entity) — the request was syntactically
            # valid but the composition fails semantic contracts. The
            # bare-ValueError branch below maps to 404 because most other
            # ValueErrors at this site are state-not-found cases; semantic
            # violations are NOT state-not-found and need their own status.
            raise HTTPException(
                status_code=422,
                detail={
                    "kind": "semantic_contract_violation",
                    "errors": [
                        {"component": e.component,
                         "message": e.message,
                         "severity": e.severity}
                        for e in exc.entries
                    ],
                    "semantic_contracts": [
                        {
                            "from_id": c.from_id, "to_id": c.to_id,
                            "consumer_plugin": c.consumer_plugin,
                            "producer_plugin": c.producer_plugin,
                            "producer_field": c.producer_field,
                            "consumer_field": c.consumer_field,
                            "outcome": c.outcome.value,
                            "requirement_code": c.requirement.requirement_code,
                        }
                        for c in exc.contracts
                    ],
                },
            ) from exc
        except ValueError as exc:
            ...  # existing 404 branch unchanged
```

Add the import at the top of `routes.py`:

```python
from elspeth.web.execution.errors import SemanticContractViolationError
```

- [ ] **Step 4: Run + commit**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/execution/test_routes.py -v`
Expected: New test passes; existing 404 mapping for unrelated ValueErrors unchanged.

```bash
git add src/elspeth/web/execution/routes.py tests/unit/web/execution/test_routes.py
git commit -m "feat(/execute): structured 422 for SemanticContractViolationError

Bare 'except ValueError' would otherwise drop the structured entries
and contracts. Status 422 distinguishes from 404 state-not-found."
```

---

### Task 4.6: MCP `_SemanticEdgeContractPayload` + serialize in `_validation_to_dict`

**Files:**
- Modify: `src/elspeth/composer_mcp/server.py:51`, `:64`, `:282`
- Modify: `tests/unit/composer_mcp/test_server.py` (or create the file/section as needed)

- [ ] **Step 1: Write failing MCP serialization test**

Append to `tests/unit/composer_mcp/test_server.py`:

```python
class TestValidationToDictSemanticContracts:
    def test_semantic_contracts_in_payload(self):
        from elspeth.composer_mcp.server import _validation_to_dict
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        validation = state.validate()
        payload = _validation_to_dict(validation)

        assert "semantic_contracts" in payload
        assert isinstance(payload["semantic_contracts"], list)
        assert len(payload["semantic_contracts"]) == 1
        contract = payload["semantic_contracts"][0]
        assert contract["from_id"] == "scrape"
        assert contract["to_id"] == "explode"
        assert contract["producer_field"] == "content"
        assert contract["consumer_field"] == "content"
        assert contract["outcome"] == "conflict"
        assert contract["requirement_code"] == "line_explode.source_field.line_framed_text"
```

- [ ] **Step 2: Run, verify failing**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/composer_mcp/test_server.py::TestValidationToDictSemanticContracts -v`
Expected: KeyError on `semantic_contracts`.

- [ ] **Step 3: Add the payload type and serializer**

In `src/elspeth/composer_mcp/server.py`, near `_EdgeContractPayload` (line 51):

```python
_SemanticEdgeContractPayload = TypedDict(
    "_SemanticEdgeContractPayload",
    {
        "from_id": str,
        "to_id": str,
        "consumer_plugin": str,
        "producer_plugin": str | None,
        "producer_field": str,
        "consumer_field": str,
        "outcome": str,
        "requirement_code": str,
    },
)
```

Update `_ValidationPayload` (line 64):

```python
class _ValidationPayload(TypedDict):
    is_valid: bool
    errors: list[_ValidationEntryPayload]
    warnings: list[_ValidationEntryPayload]
    suggestions: list[_ValidationEntryPayload]
    edge_contracts: list[_EdgeContractPayload]
    semantic_contracts: list[_SemanticEdgeContractPayload]
```

Add a serializer near `_edge_contract_to_payload`:

```python
def _semantic_edge_contract_to_payload(
    contract: Any,
) -> _SemanticEdgeContractPayload:
    """Serialize a SemanticEdgeContract for MCP. Field names + enum values only."""
    return {
        "from_id": contract.from_id,
        "to_id": contract.to_id,
        "consumer_plugin": contract.consumer_plugin,
        "producer_plugin": contract.producer_plugin,
        "producer_field": contract.producer_field,
        "consumer_field": contract.consumer_field,
        "outcome": contract.outcome.value,
        "requirement_code": contract.requirement.requirement_code,
    }
```

Update `_validation_to_dict` (line 282) to include `semantic_contracts`:

```python
def _validation_to_dict(validation: Any) -> _ValidationPayload:
    return {
        "is_valid": validation.is_valid,
        "errors": [entry.to_dict() for entry in validation.errors],
        "warnings": [entry.to_dict() for entry in validation.warnings],
        "suggestions": [entry.to_dict() for entry in validation.suggestions],
        "edge_contracts": [_edge_contract_to_payload(c)
                           for c in validation.edge_contracts],
        "semantic_contracts": [_semantic_edge_contract_to_payload(c)
                               for c in validation.semantic_contracts],
    }
```

- [ ] **Step 4: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/composer_mcp/test_server.py -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/composer_mcp/server.py tests/unit/composer_mcp/test_server.py
git commit -m "feat(mcp): serialize semantic_contracts in MCP validation payloads"
```

---

### Task 4.7: `ToolResult.to_dict()` AND `_execute_preview_pipeline` include `semantic_contracts`

**Files:**
- Modify: `src/elspeth/web/composer/tools.py:114` (`ToolResult.to_dict()` validation payload)
- Modify: `src/elspeth/web/composer/tools.py:3345` (`_execute_preview_pipeline` summary)

**Goal:** Two distinct serialization paths emit validation data to MCP clients. Both must include `semantic_contracts`. The original draft only modified the preview path; `ToolResult.to_dict()` (which carries validation results from EVERY mutation tool — `upsert_node`, `set_source`, `patch_*`, etc.) would silently omit the new field.

- [ ] **Step 1: Write failing tests for both surfaces**

Append to the appropriate tools test file:

```python
class TestToolResultSemanticContracts:
    def test_tool_result_to_dict_includes_semantic_contracts(self):
        from elspeth.web.composer.tools import ToolResult
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        validation = state.validate()
        tr = ToolResult(
            success=True, updated_state=state, validation=validation,
            affected_nodes=(),
        )
        payload = tr.to_dict()
        assert "semantic_contracts" in payload["validation"]
        assert len(payload["validation"]["semantic_contracts"]) == 1
        assert payload["validation"]["semantic_contracts"][0]["outcome"] == "conflict"
        assert payload["validation"]["semantic_contracts"][0]["consumer_plugin"] == "line_explode"


class TestPreviewPipelineSemanticContracts:
    def test_summary_includes_semantic_contracts(self):
        from elspeth.web.composer.tools import _execute_preview_pipeline
        from tests.unit.web.composer.test_semantic_validator import _wardline_state

        state = _wardline_state(text_separator=" ")
        result = _execute_preview_pipeline({}, state, catalog=...)
        assert "semantic_contracts" in result.data
        assert len(result.data["semantic_contracts"]) == 1
        assert result.data["semantic_contracts"][0]["outcome"] == "conflict"
```

- [ ] **Step 2: Run, verify failing**

- [ ] **Step 3: Add a shared serializer and apply at both sites**

In `src/elspeth/web/composer/tools.py`, add a module-level helper:

```python
def _semantic_contracts_payload(
    contracts: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Serialize SemanticEdgeContract tuple to JSON-friendly dicts.

    Centralized so ToolResult.to_dict and _execute_preview_pipeline
    emit identical shapes — and so adding a field updates both surfaces
    in one place.
    """
    return [
        {
            "from_id": sc.from_id,
            "to_id": sc.to_id,
            "consumer_plugin": sc.consumer_plugin,
            "producer_plugin": sc.producer_plugin,
            "producer_field": sc.producer_field,
            "consumer_field": sc.consumer_field,
            "outcome": sc.outcome.value,
            "requirement_code": sc.requirement.requirement_code,
        }
        for sc in contracts
    ]
```

In `ToolResult.to_dict()` (around line 114), update the `validation` dict:

```python
        result: dict[str, Any] = {
            "success": self.success,
            "validation": {
                "is_valid": self.validation.is_valid,
                "errors": [e.to_dict() for e in self.validation.errors],
                "warnings": [e.to_dict() for e in self.validation.warnings],
                "suggestions": [e.to_dict() for e in self.validation.suggestions],
                "semantic_contracts": _semantic_contracts_payload(
                    self.validation.semantic_contracts,
                ),
            },
            "affected_nodes": list(self.affected_nodes),
            "version": self.updated_state.version,
        }
```

In `_execute_preview_pipeline` (around line 3345), update the summary:

```python
    summary: dict[str, Any] = {
        "is_valid": validation.is_valid,
        "errors": [e.to_dict() for e in validation.errors],
        "warnings": [e.to_dict() for e in validation.warnings],
        "suggestions": [e.to_dict() for e in validation.suggestions],
        "edge_contracts": [ec.to_dict() for ec in validation.edge_contracts],
        "semantic_contracts": _semantic_contracts_payload(
            validation.semantic_contracts,
        ),
        ...
    }
```

(Note: `SemanticEdgeContract` does not have a `.to_dict()` method by design — keeping serialization at consumption sites avoids creating an L0 dependency on JSON encoding.)

- [ ] **Step 4: Run**

Expected: Both new tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools.py tests/...
git commit -m "feat(composer): semantic_contracts in ToolResult.to_dict + preview summary

ToolResult is the carrier for every mutation tool's validation result.
Without this, mutation tools silently omit the new field — only the
explicit preview tool would expose it. Shared _semantic_contracts_payload
helper keeps the two surfaces in sync."
```

---

## Phase 5 — Plugin assistance discovery tool

Expose plugin assistance to the composer LLM and other agent consumers as a discovery tool. This task is small but important: it enables Phase 6's deletion of hardcoded suggestion strings.

### Task 5.1: `get_plugin_assistance` MCP tool

**Files:**
- Modify: `src/elspeth/composer_mcp/server.py` (add tool definition + dispatch)
- Modify: `src/elspeth/web/composer/tools.py` (add the underlying composer tool)

- [ ] **Step 1: Write failing test**

```python
def test_get_plugin_assistance_returns_structured_payload():
    from elspeth.web.composer.tools import _execute_get_plugin_assistance

    result = _execute_get_plugin_assistance(
        {"plugin_name": "web_scrape",
         "issue_code": "web_scrape.content.compact_text"},
        state=...,
        catalog=...,
    )
    assert result.success
    payload = result.data
    assert payload["plugin_name"] == "web_scrape"
    assert payload["issue_code"] == "web_scrape.content.compact_text"
    assert "summary" in payload
    assert isinstance(payload["suggested_fixes"], list)
```

- [ ] **Step 2: Implement the tool**

Add a new function `_execute_get_plugin_assistance(...)` in `tools.py` that calls `cls.get_agent_assistance(issue_code=...)` for the named plugin and serializes `PluginAssistance` to a dict (no `.to_dict()` on the contract type — serialize at the boundary, same pattern as `SemanticEdgeContract`).

Register the tool in `_DISCOVERY_TOOLS` and add the MCP definition.

- [ ] **Step 3: Run + commit**

```bash
git add src/elspeth/web/composer/tools.py src/elspeth/composer_mcp/server.py tests/...
git commit -m "feat(mcp): get_plugin_assistance discovery tool"
```

---

## Phase 6 — Delete the legacy validator

Now the new system has parity coverage and is wired through every surface. Delete the old code in one atomic commit so the parallel-validator window closes cleanly.

### Task 6.1: Delete `validate_transform_framing_contracts` and all three call sites

**Files:**
- Modify: `src/elspeth/web/composer/state.py` (delete function at line 351; delete call at line 1509)
- Already removed in Phase 4: `validation.py:274` (Task 4.3) and `service.py:325` (Task 4.5)

- [ ] **Step 1: Verify Task 4.3 and 4.5 removed the legacy imports**

```bash
grep -rn 'validate_transform_framing_contracts' src/elspeth/
```

Expected: Only `state.py` remains (the function definition + the internal call at line 1509).

If `validation.py` or `service.py` still imports the function, those tasks were not completed — go back.

- [ ] **Step 2: Delete the function and the call**

In `state.py`:
- Delete the entire `validate_transform_framing_contracts` function (now refactored to use `ProducerResolver`).
- Delete the call inside `CompositionState.validate()` (the `errors.extend(validate_transform_framing_contracts(self.nodes))` line — Task 4.2 added the new call alongside it; remove the legacy line).

- [ ] **Step 3: Delete legacy framing tests that no longer have an analogue**

In `tests/unit/web/composer/test_state.py`:
- Search for `validate_transform_framing_contracts` references; remove or rename.
- The Wardline regression cases now live in semantic-validator tests (Phase 3 Task 3.1). The legacy state-level test class (`TestStateValidatesFraming` or similar) can be removed if its scenarios are fully covered by `TestCompositionStateValidateEmitsSemanticContracts` in Task 4.2.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ -x --timeout 60`
Expected: All tests pass. If anything fails, the parallel-validator parity test (Task 3.2) was insufficient — investigate and add coverage before re-running deletion.

- [ ] **Step 5: Run the tier-model and freeze-guard CI scripts**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth
```

Expected: No new violations.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/unit/web/composer/test_state.py
git commit -m "refactor(composer): delete legacy validate_transform_framing_contracts

The generic semantic-contract validator now owns the line_explode +
web_scrape framing case (and more — format=raw is also caught now,
which the legacy validator missed). Parity test (Task 3.2) verified
agreement on overlapping cases before deletion."
```

---

### Task 6.2: Pin the Wardline regression in the new system

**Files:**
- Modify: `tests/unit/web/composer/test_semantic_validator.py`

**Goal:** A test fixture that uses the EXACT options shape that originally triggered the Wardline regression — not a synthesized variant. If the new system regresses on this specific shape, the test must catch it.

- [ ] **Step 1: Build the exact-shape fixture from the broken Wardline YAML**

Use `git show <commit-where-text-separator-was-added>~1:data/wardline_line_export_pipeline.yaml` (from Task 0.1) to get the broken version. Translate to a `CompositionState`:

```python
class TestWardlineRegressionPin:
    """Exact options shape from the original Wardline regression YAML.

    Sourced from data/wardline_line_export_pipeline.yaml at the commit
    immediately before text_separator: '\\n' was added. If the new
    system fails to block this shape, the regression has recurred.
    """

    def _wardline_broken_yaml_state(self):
        # OPTIONS COPIED VERBATIM from the broken YAML revision.
        return CompositionState(
            metadata=PipelineMetadata(name="wardline-line-export"),
            source=SourceSpec(
                plugin="csv", on_success="scrape_in",
                options={
                    "schema": {"mode": "fixed", "fields": ["url: str"]},
                    "path": "data/blobs/<uuid>/<uuid>_wardline_url.csv",
                    "on_validation_failure": "quarantine",
                },
                on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="scrape_page", node_type="transform", plugin="web_scrape",
                    input="scrape_in", on_success="explode_in", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["url: str"]},
                        "required_input_fields": ["url"],
                        "url_field": "url",
                        "content_field": "content",
                        "fingerprint_field": "content_fingerprint",
                        "format": "text",
                        # text_separator OMITTED -> defaults to single space
                        "fingerprint_mode": "content",
                        "strip_elements": ["script", "style"],
                        "http": {
                            "abuse_contact": "pipeline@example.com",
                            "scraping_reason": "User requested Wardline contents exported as line-oriented JSON",
                            "timeout": 30,
                            "allowed_hosts": "public_only",
                        },
                    },
                ),
                NodeSpec(
                    id="split_lines", node_type="transform", plugin="line_explode",
                    input="explode_in", on_success="sink", on_error="errors",
                    options={
                        "schema": {"mode": "flexible",
                                   "fields": ["content: str"]},
                        "source_field": "content",
                    },
                ),
            ),
            outputs=(OutputSpec(name="sink", plugin="json",
                                options={"path": "out.json"}),
                     OutputSpec(name="errors", plugin="json",
                                options={"path": "err.json"})),
        )

    def test_wardline_broken_yaml_blocked_by_semantic_validator(self):
        state = self._wardline_broken_yaml_state()
        errors, contracts = validate_semantic_contracts(state)

        assert len(errors) == 1
        assert errors[0].component == "node:split_lines"

        assert len(contracts) == 1
        contract = contracts[0]
        assert contract.from_id == "scrape_page"
        assert contract.to_id == "split_lines"
        assert contract.outcome is SemanticOutcome.CONFLICT
        assert contract.requirement.requirement_code == \
            "line_explode.source_field.line_framed_text"

    def test_wardline_broken_yaml_blocked_by_full_validate(self):
        state = self._wardline_broken_yaml_state()
        result = state.validate()
        assert result.is_valid is False
        assert any(e.component == "node:split_lines" for e in result.errors)
```

- [ ] **Step 2: Run**

Run: `PYTHONPATH=src .venv/bin/python -m pytest tests/unit/web/composer/test_semantic_validator.py::TestWardlineRegressionPin -v`
Expected: Both tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_semantic_validator.py
git commit -m "test(composer): pin Wardline regression with exact-shape fixture

Options copied verbatim from data/wardline_line_export_pipeline.yaml
at the commit immediately before the text_separator fix. Guards against
the new system silently regressing on the original failure shape."
```

---

## Phase 7 — Documentation update

### Task 7.1: Replace skill-prose framing guidance with assistance pointer

**Files (verified to exist; widened from v2-pre-review which only searched docs/):**
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md` (in-repo composer skill prose, line ~458)
- Modify: `.claude/skills/pipeline-composer/SKILL.md` (Claude Code skill copy, line ~253)
- Modify: any other in-repo doc that hardcodes the `web_scrape` + `text_separator: "\n"` advice

- [ ] **Step 1: Find current prose across all known skill paths**

```bash
grep -rn 'text_separator.*\\n\|line_explode.*web_scrape\|web_scrape.*line_explode' \
  docs/ \
  src/elspeth/web/composer/skills/ \
  .claude/skills/ \
  2>/dev/null
```

Expected matches: at least the two skill files above. Treat any other match as in-scope for this task.

- [ ] **Step 2: Replace with pointer**

Replace each prose block with:

> When configuring `line_explode` after `web_scrape`, call `get_plugin_assistance(plugin_name="line_explode", issue_code="line_explode.source_field.line_framed_text")` to get the current guidance from the plugin itself. The skill no longer hardcodes specific framing advice — it lives in the plugin and is exposed via the discovery tool.

- [ ] **Step 3: Commit**

Stage every file the Step 1 grep returned. The two known skill paths
live OUTSIDE `docs/`, so a bare `git add docs/` would silently leave
the in-repo skill prose untracked.

```bash
git add docs/ \
        src/elspeth/web/composer/skills/pipeline_composer.md \
        .claude/skills/pipeline-composer/SKILL.md
# Add any additional paths the Step 1 grep returned.
git commit -m "docs: replace hardcoded line_explode/web_scrape guidance with assistance pointer"
```

---

## Self-review checklist

After completing the plan, verify:

1. **Spec coverage:** Each v1 blocker (B1–B8) is resolved by a numbered task above. ✓ (table at the top maps blockers to phases)
2. **Placeholder scan:** No "TBD," "TODO," "implement later" in any task. Tasks like 4.4, 4.6, 4.7 reference real test fixtures whose paths the implementor must look up — that is intentional, not a placeholder.
3. **Type consistency:** `OutputSemanticDeclaration`, `InputSemanticRequirements`, `FieldSemanticFacts`, `FieldSemanticRequirement`, `SemanticEdgeContract`, `PluginAssistance`, `PluginAssistanceExample`, `SemanticContractViolationError`, `ProducerResolver`, `ProducerEntry` — names are stable across all tasks.
4. **Phase 0 is required:** Tasks 0.1 (confirm surface) and 0.2 (extract resolver) MUST land before Phase 1. The synthesizer's blocker B3 cannot be addressed retroactively.
5. **B6 elimination:** No WARN-policy code path exists in Phase 1 (line_explode is FAIL). No audit-trail recording is needed because no relaxation decision exists. If a future requirement adds WARN, that requirement also adds the audit event — out of scope for this plan.

---

**Plan complete and saved to `docs/plans/2026-04-27-plugin-declared-transform-semantics-v2.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

---

## Execution journal

### Phase 0 — completed 2026-04-28

**Task 0.1 — Wardline regression surface (investigative).**

Wardline regression surface: web composer only. CLI has no framing or
semantic validator today; Phase 1 covering `CompositionState.validate()`
+ `/validate` + `/execute` + MCP is the correct surface. CLI parity is
intentionally deferred to a future plan.

Verification:
- `grep -rn 'framing\|line_explode.*web_scrape\|web_scrape.*line_explode' src/elspeth/cli_helpers.py src/elspeth/core/ src/elspeth/engine/` returned zero matches.
- All four web-surface line numbers verified at HEAD: `state.py:351`
  (`def validate_transform_framing_contracts`), `validation.py:274`
  (`framing_errors = validate_transform_framing_contracts(state.nodes)`),
  `service.py:325` (`framing_errors = validate_transform_framing_contracts(composition_state.nodes)`),
  `state.py:1509` (`errors.extend(validate_transform_framing_contracts(self.nodes))`).
- Bookkeeping: `validation.py` and `service.py` resolve to
  `src/elspeth/web/execution/`, not `web/composer/`. Phase 1's edit list
  must touch the `web/execution/` modules.
- `data/wardline_line_export_pipeline.yaml` is currently untracked in
  git (`git ls-files | grep wardline` returns nothing). The current
  working-tree file at line 29 already contains `text_separator: "\n"`
  (the fixed form). The "broken version" that triggered the regression
  is therefore not recoverable from history — Phase 1's regression-pin
  test must construct the broken version inline rather than via
  `git show <commit>:…`.

**Task 0.3 — In-tree audit of `line_explode` usage (investigative).**

| Pipeline | Upstream producer | Producer kind | Phase 1 outcome under FAIL |
|----------|-------------------|---------------|---------------------------|
| `data/wardline_line_export_pipeline.yaml` | `web_scrape` (`scrape_page`) | transform (declared in Phase 2) | SATISFIED (text_separator includes `\n`) |

No source → `line_explode` chains in tree. No non-`web_scrape`
transform → `line_explode` chains in tree. The validator's
"skip source-fed edges" rule is not exercised by any in-tree
pipeline; Phase 6's deletion of the legacy validator will not break
any in-tree pipeline under `unknown_policy=FAIL`.

**Task 0.2 — Extract `ProducerResolver` (refactor, committed as `d18ebb9d`).**

Status: merged into `RC5-UX`. Resolver lives at
`src/elspeth/web/composer/_producer_resolver.py`; tests at
`tests/unit/web/composer/test_producer_resolver.py` (12 cases — 6 build
+ 6 walk-back, including the two source-walk tests pinning the
reviewer-found bug). Both existing callers
(`validate_transform_framing_contracts`, `_check_schema_contracts`) now
consume the resolver. Schema-specific concerns (pass-through propagation
via `_effective_producer_vote` / `_connection_propagation_vote` /
`_intersect_predecessor_guarantees`, plus coalesce/fork-warning emission)
remain inside `state.py` per the plan's scope discipline.

Two minor behavioural deviations from "no behaviour change" — both
strict improvements, both flagged in the commit body, neither exercised
by any pre-existing test:

1. `ProducerResolver.build` adds a same-node carve-out: a single node
   registering against the same connection more than once
   (e.g. `on_success="x"` plus a `fork_to=("x",)` entry) is now
   idempotent rather than a duplicate. Generalises a routes-only
   carve-out the schema validator already had.
2. Fork-to-sink branches in `_check_schema_contracts` register via
   `direct_sink_producers` (uniform with routes-to-sinks) rather than
   landing in `producer_map`. Edge-contract output is unchanged for
   the existing fork-gate-direct-sink test.

The deleted `_ProducerEntry` NamedTuple in `state.py` was retired under
the no-legacy-code policy; the vote functions now type their parameters
as the new frozen `ProducerEntry` (same field shape).

Tier-model allowlist fingerprints for `_check_schema_contracts` shifted
because line numbers moved; the underlying R5/R6 rationale is unchanged
and entries were updated in place.

Verification gates at merge: 12 resolver tests, 4 framing tests, 168
full state tests, 913 composer+execution tests, 13462 unit tests
(2 unrelated skips) all green. Ruff + mypy clean. Pre-commit suite
(tier-model enforcement, frozen-annotation guards, contract manifest,
composer exception-channel) clean.

Phase 0 is closed. Phase 1 (contract types) is unblocked.

### Phases 1–7 — completed 2026-04-28

All seven phases landed on `RC5-UX` over 27 commits in a single session. Final
verification: `tests/unit/contracts/ tests/unit/web/composer/ tests/unit/web/execution/ tests/unit/composer_mcp/ tests/unit/plugins/transforms/test_web_scrape.py tests/unit/plugins/transforms/test_line_explode.py tests/unit/plugins/infrastructure/`
returns 3862 passed. `grep -rn 'validate_transform_framing_contracts' src/elspeth/ tests/`
returns zero matches — the legacy validator is fully deleted.

| Phase | Commits | Summary |
|-------|---------|---------|
| 1 — Contract types | 5 (5745dfe9 … 8e9d3e4c) | enums, facts/requirements/declaration dataclasses, edge contract + comparator, PluginAssistance with freeze guards, layer-purity test |
| 2 — Plugin API | 3 (8d10d422 … 5a2604f8) | BaseTransform default methods; web_scrape output_semantics + assistance; line_explode input_semantic_requirements + assistance |
| 3 — Generic semantic validator | 3 (a37b27df … bbc0463e) | validate_semantic_contracts core, parity test (gate for Phase 6 deletion), secret-leakage sentinel |
| 4 Part A — composer + /validate + frontend | 5 (555e5e72 … 2035442d) | ValidationSummary.semantic_contracts; integration into CompositionState.validate; /validate route swap; Pydantic schema; frontend type |
| 4 Part B — /execute + MCP + ToolResult | 5 (793fa53b … c67fc0ce) | hoisted shared helpers; SemanticContractViolationError + 422 handler; MCP serialization; ToolResult.to_dict + preview parity |
| 5 — get_plugin_assistance discovery tool | 1 (6b436d3c) | MCP tool exposing PluginAssistance keyed by plugin_name + issue_code |
| 6 — Delete legacy validator + pin Wardline | 2 (824bae65, 684fe4fe) | atomic deletion of validate_transform_framing_contracts and call sites; surface-level Wardline regression pin via state.validate() |
| 7 — Skill prose update | 1 (e806a937) | replaced hardcoded text_separator advice with get_plugin_assistance pointer in pipeline_composer.md, .claude/skills/pipeline-composer/SKILL.md, and docs/reference/web-scrape-transform.md |

**Notable execution-time facts (not changes to plan design):**

- Worktree-harness stale-base behaviour: every phase's subagent had to `git
  merge --ff-only RC5-UX` before starting. The Phase 4+ briefs pre-authorised
  the FF-align so subagents didn't need to round-trip for permission. Worth
  flagging if the harness behaviour changes.
- One process violation: the Phase 2 subagent ran `git stash`/`git stash pop`
  to test whether a freeze-guard finding was pre-existing. CLAUDE.md "Git
  Safety" forbids this. The pop completed cleanly so no data was lost; the
  agent acknowledged the violation, finished the Task 2.3 commit cleanly, and
  committed to using `git diff <base>..HEAD -- <path>` for "is this
  pre-existing?" questions going forward. Phase 3+ briefs added an explicit
  "no `git stash`" line.
- Worktree isolation leak in Phase 5+6+7: pre-commit hooks running from inside
  the worktree appear to have written to `config/cicd/contracts-whitelist.yaml`
  in the main checkout (likely via the shared `.venv` symlink resolving the
  config path through). The leak was benign — diff between worktree and main
  was zero — but the FF-merge had to be preceded by `git restore` on that one
  file. Worth flagging as a harness pattern.
- Phase 4 Part A's consumer-probe widening: the plan's stated invariant
  ("consumer probe failure propagates") was preserved for unexpected
  exceptions but widened to share the named config-error predicate
  (`PluginConfigError`, `PluginNotFoundError`, `TemplateError`,
  `UnknownPluginTypeError`, plus a specific `ValueError`) used by
  `_check_schema_contracts`. Necessary because `state.validate()` has 40+
  pre-existing tests with fictional plugin names that the strict invariant
  would break. Documented in commit `8cfeb1ac`.

The Wardline regression test (`TestWardlineRegressionPin` in
`tests/unit/web/composer/test_semantic_validator.py`) is the single
load-bearing test for this work. It exercises `state.validate()` end-to-end
against an inline broken-shape fixture. If a future refactor breaks the
wiring between `state.validate()` and the semantic validator, this test
catches it before the Wardline shape can regress in production.

Plan status: COMPLETE.
