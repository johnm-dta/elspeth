# Web UX Task-Plan 4C: YAML Generator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Implement deterministic CompositionState → ELSPETH pipeline YAML generation with frozen field unwrapping
**Parent Plan:** `plans/2026-03-28-web-ux-sub4-composer.md`
**Spec:** `specs/2026-03-28-web-ux-sub4-composer-design.md`
**Depends On:** Task-Plan 4A (Data Models — provides CompositionState with to_dict())
**Blocks:** Task-Plan 4D (Composer Service & Wiring)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/composer/yaml_generator.py` | Deterministic YAML generator — CompositionState to ELSPETH pipeline YAML |
| Create | `tests/unit/web/composer/test_yaml_generator.py` | YAML generator tests (linear, gate, aggregation, fork/coalesce, determinism, frozen containers) |

---

### Task 5: YAML Generator

**Files:**
- Create: `src/elspeth/web/composer/yaml_generator.py`
- Create: `tests/unit/web/composer/test_yaml_generator.py`

- [ ] **Step 1: Write YAML generator tests**

```python
# tests/unit/web/composer/test_yaml_generator.py
"""Tests for deterministic YAML generation from CompositionState."""
from __future__ import annotations

import yaml

import pytest

from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.composer.yaml_generator import generate_yaml


def _make_linear_pipeline() -> CompositionState:
    """Source -> transform -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="transform_1",
            options={"path": "/data/input.csv", "schema": {"fields": ["name", "age"]}},
            on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="transform_1",
                node_type="transform",
                plugin="uppercase",
                input="source_out",
                on_success="main_output",
                on_error=None,
                options={"field": "name"},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(
            EdgeSpec(id="e1", from_node="source", to_node="transform_1", edge_type="on_success", label=None),
            EdgeSpec(id="e2", from_node="transform_1", to_node="main_output", edge_type="on_success", label=None),
        ),
        outputs=(
            OutputSpec(
                name="main_output",
                plugin="csv",
                options={"path": "/data/output.csv"},
                on_write_failure="quarantine",
            ),
        ),
        metadata=PipelineMetadata(name="Linear Pipeline", description="A simple pipeline"),
        version=5,
    )


def _make_gate_pipeline() -> CompositionState:
    """Source -> gate -> two sinks."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="quality_check",
            options={"path": "/data/in.csv"},
            on_validation_failure="discard",
        ),
        nodes=(
            NodeSpec(
                id="quality_check",
                node_type="gate",
                plugin=None,
                input="source_out",
                on_success=None,
                on_error=None,
                options={},
                condition="row['confidence'] >= 0.85",
                routes={"high": "good_output", "low": "review_output"},
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="good_output", plugin="csv", options={"path": "/good.csv"}, on_write_failure="quarantine"),
            OutputSpec(name="review_output", plugin="csv", options={"path": "/review.csv"}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(name="Gate Pipeline"),
        version=3,
    )


def _make_aggregation_pipeline() -> CompositionState:
    """Source -> aggregation -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv", on_success="batch_agg",
            options={"path": "/data/in.csv"}, on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="batch_agg", node_type="aggregation", plugin="batch_counter",
                input="source_out", on_success="main_output", on_error=None,
                options={"batch_size": 10}, condition=None, routes=None,
                fork_to=None, branches=None, policy=None, merge=None,
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _make_fork_coalesce_pipeline() -> CompositionState:
    """Source -> fork gate -> two paths -> coalesce -> sink."""
    return CompositionState(
        source=SourceSpec(
            plugin="csv", on_success="fork_gate",
            options={}, on_validation_failure="quarantine",
        ),
        nodes=(
            NodeSpec(
                id="fork_gate", node_type="gate", plugin=None,
                input="source_out", on_success=None, on_error=None,
                options={}, condition="True",
                routes={"all": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None, policy=None, merge=None,
            ),
            NodeSpec(
                id="merge_point", node_type="coalesce", plugin=None,
                input="join", on_success="main_output", on_error=None,
                options={}, condition=None, routes=None, fork_to=None,
                branches=("path_a", "path_b"), policy="require_all", merge="nested",
            ),
        ),
        edges=(),
        outputs=(
            OutputSpec(name="main_output", plugin="csv", options={}, on_write_failure="discard"),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


class TestGenerateYaml:
    def test_linear_pipeline(self) -> None:
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        # Source
        assert parsed["source"]["plugin"] == "csv"
        assert parsed["source"]["on_success"] == "transform_1"
        assert parsed["source"]["options"]["path"] == "/data/input.csv"
        assert parsed["source"]["options"]["on_validation_failure"] == "quarantine"

        # Transform
        assert len(parsed["transforms"]) == 1
        t = parsed["transforms"][0]
        assert t["name"] == "transform_1"
        assert t["plugin"] == "uppercase"
        assert t["input"] == "source_out"
        assert t["on_success"] == "main_output"
        assert t["options"]["field"] == "name"

        # Sink
        assert "main_output" in parsed["sinks"]
        s = parsed["sinks"]["main_output"]
        assert s["plugin"] == "csv"

    def test_gate_pipeline(self) -> None:
        state = _make_gate_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        assert len(parsed["gates"]) == 1
        g = parsed["gates"][0]
        assert g["name"] == "quality_check"
        assert g["condition"] == "row['confidence'] >= 0.85"
        assert g["routes"]["high"] == "good_output"
        assert g["routes"]["low"] == "review_output"

    def test_aggregation_pipeline(self) -> None:
        state = _make_aggregation_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "aggregations" in parsed
        assert len(parsed["aggregations"]) == 1
        a = parsed["aggregations"][0]
        assert a["name"] == "batch_agg"
        assert a["plugin"] == "batch_counter"
        assert a["options"]["batch_size"] == 10

    def test_fork_coalesce_pipeline(self) -> None:
        state = _make_fork_coalesce_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)

        assert "gates" in parsed
        gate = parsed["gates"][0]
        assert gate["fork_to"] == ["path_a", "path_b"]

        assert "coalesce" in parsed
        coal = parsed["coalesce"][0]
        assert coal["branches"] == ["path_a", "path_b"]
        assert coal["policy"] == "require_all"
        assert coal["merge"] == "nested"

    def test_deterministic(self) -> None:
        """Same state produces byte-identical YAML."""
        state = _make_linear_pipeline()
        yaml1 = generate_yaml(state)
        yaml2 = generate_yaml(state)
        assert yaml1 == yaml2

    def test_landscape_key_never_emitted(self) -> None:
        """landscape key is never emitted — URL comes from WebSettings at execution time (S1 fix)."""
        state = _make_linear_pipeline()
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert "landscape" not in parsed

    def test_on_error_emitted_when_set(self) -> None:
        state = CompositionState(
            source=SourceSpec(
                plugin="csv", on_success="t1",
                options={}, on_validation_failure="quarantine",
            ),
            nodes=(
                NodeSpec(
                    id="t1", node_type="transform", plugin="uppercase",
                    input="in", on_success="out", on_error="error_sink",
                    options={}, condition=None, routes=None,
                    fork_to=None, branches=None, policy=None, merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(name="out", plugin="csv", options={}, on_write_failure="discard"),
                OutputSpec(name="error_sink", plugin="csv", options={}, on_write_failure="discard"),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["transforms"][0]["on_error"] == "error_sink"

    def test_frozen_state_serializes_without_error(self) -> None:
        """generate_yaml() handles frozen state objects (MappingProxyType, tuple).

        AC #15: No RepresenterError from PyYAML on frozen containers.
        Verifies that generate_yaml() correctly calls state.to_dict()
        before yaml.dump().
        """
        state = _make_linear_pipeline()
        # State has been through freeze_fields() — options are MappingProxyType
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed["source"]["plugin"] == "csv"
        # Nested frozen options must serialize correctly
        assert parsed["source"]["options"]["schema"]["fields"] == ["name", "age"]

    def test_empty_state_minimal_yaml(self) -> None:
        """Empty state produces minimal valid YAML (no source, no sinks)."""
        state = CompositionState(
            source=None, nodes=(), edges=(), outputs=(),
            metadata=PipelineMetadata(), version=1,
        )
        yaml_str = generate_yaml(state)
        parsed = yaml.safe_load(yaml_str)
        assert parsed.get("source") is None or "source" not in parsed
```

- [ ] **Step 2: Run tests — expect FAIL (module not found)**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement YAML generator**

```python
# src/elspeth/web/composer/yaml_generator.py
"""Deterministic YAML generator — CompositionState to ELSPETH pipeline YAML.

Pure function. Same CompositionState always produces byte-identical YAML.
Uses yaml.dump() with sort_keys=True for determinism.

Layer: L3 (application). Imports from L3 (web/composer/state) only.
"""
from __future__ import annotations

from typing import Any

import yaml

from elspeth.web.composer.state import CompositionState


def generate_yaml(state: CompositionState) -> str:
    """Convert a CompositionState to ELSPETH pipeline YAML.

    The output is deterministic: same state produces byte-identical YAML.
    Maps CompositionState fields to the YAML structure expected by
    ELSPETH's load_settings() parser.

    Calls state.to_dict() to unwrap all frozen containers
    (MappingProxyType -> dict, tuple -> list) before passing to
    yaml.dump(). This avoids RepresenterError from PyYAML on frozen
    types. See spec R4 and AC #15.

    Args:
        state: The pipeline composition state to serialize.

    Returns:
        YAML string representing the pipeline configuration.
    """
    # Unwrap frozen containers to plain Python types (R4).
    # to_dict() recursively converts MappingProxyType -> dict,
    # tuple -> list. Without this, yaml.dump() raises RepresenterError.
    state_dict = state.to_dict()

    doc: dict[str, Any] = {}

    # Source
    source = state_dict.get("source")
    if source is not None:
        source_options = dict(source["options"])
        source_options["on_validation_failure"] = source["on_validation_failure"]
        doc["source"] = {
            "plugin": source["plugin"],
            "on_success": source["on_success"],
            "options": source_options,
        }

    # Transforms
    transforms = [n for n in state_dict["nodes"] if n["node_type"] == "transform"]
    if transforms:
        doc["transforms"] = []
        for t in transforms:
            entry: dict[str, Any] = {
                "name": t["id"],
                "plugin": t["plugin"],
                "input": t["input"],
                "on_success": t["on_success"],
            }
            if t.get("on_error") is not None:
                entry["on_error"] = t["on_error"]
            if t["options"]:
                entry["options"] = t["options"]
            doc["transforms"].append(entry)

    # Gates
    gates = [n for n in state_dict["nodes"] if n["node_type"] == "gate"]
    if gates:
        doc["gates"] = []
        for g in gates:
            entry = {
                "name": g["id"],
                "input": g["input"],
                "condition": g.get("condition"),
                "routes": g.get("routes", {}),
            }
            if g.get("fork_to") is not None:
                entry["fork_to"] = g["fork_to"]
            doc["gates"].append(entry)

    # Aggregations
    aggregations = [n for n in state_dict["nodes"] if n["node_type"] == "aggregation"]
    if aggregations:
        doc["aggregations"] = []
        for a in aggregations:
            entry = {
                "name": a["id"],
                "plugin": a["plugin"],
                "input": a["input"],
                "on_success": a["on_success"],
            }
            if a.get("on_error") is not None:
                entry["on_error"] = a["on_error"]
            if a["options"]:
                entry["options"] = a["options"]
            doc["aggregations"].append(entry)

    # Coalesce
    coalesces = [n for n in state_dict["nodes"] if n["node_type"] == "coalesce"]
    if coalesces:
        doc["coalesce"] = []
        for c in coalesces:
            entry = {
                "name": c["id"],
                "branches": c.get("branches", []),
                "policy": c.get("policy"),
                "merge": c.get("merge"),
            }
            doc["coalesce"].append(entry)

    # Sinks
    if state_dict["outputs"]:
        doc["sinks"] = {}
        for output in state_dict["outputs"]:
            sink_entry: dict[str, Any] = {
                "plugin": output["plugin"],
                "on_write_failure": output["on_write_failure"],
            }
            if output["options"]:
                sink_entry["options"] = output["options"]
            doc["sinks"][output["name"]] = sink_entry

    # landscape key is intentionally omitted — URL comes from
    # WebSettings.get_landscape_url() at execution time (security fix S1).

    return yaml.dump(doc, default_flow_style=False, sort_keys=True)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_yaml_generator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Run mypy**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/yaml_generator.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/yaml_generator.py tests/unit/web/composer/test_yaml_generator.py
git commit -m "feat(web/composer): add deterministic YAML generator"
```

---

## Self-Review Checklist

- [ ] All 6 steps complete with all checkboxes checked
- [ ] `generate_yaml()` calls `state.to_dict()` before `yaml.dump()` (R4 fix -- no RepresenterError on frozen containers)
- [ ] YAML is deterministic: `sort_keys=True`, `default_flow_style=False`
- [ ] `landscape` key is never emitted (security fix S1 -- URL comes from WebSettings at execution time)
- [ ] Linear pipeline test: source, transform, and sink sections all correct
- [ ] Gate pipeline test: condition, routes emitted correctly
- [ ] Aggregation pipeline test: plugin and options emitted correctly
- [ ] Fork/coalesce pipeline test: `fork_to`, `branches`, `policy`, `merge` all present
- [ ] `on_error` only emitted when non-None
- [ ] Empty state produces minimal valid YAML without error
- [ ] Frozen state with nested `MappingProxyType` options serializes without error (AC #15)
- [ ] `on_validation_failure` is nested inside `source.options` (matches ELSPETH settings format)
- [ ] mypy passes on `yaml_generator.py`
- [ ] Commit uses conventional commit format

---

## Round 5 Review Findings

**Warnings (implement during execution):**

- **W-4C-1 (AC #8): `load_settings()` round-trip not tested.** The YAML generator tests verify that `yaml.safe_load()` can parse the output, but AC #8 in the spec requires a round-trip through ELSPETH's own `load_settings()`. If `load_settings()` has stricter requirements (required top-level keys, field validators), the YAML output could be syntactically valid but semantically rejected. Add a test that writes `generate_yaml(state)` to a temp file, calls `load_settings(tmp_path)`, and verifies no exception. This is the authoritative validation that the YAML is engine-compatible.

- **W-4C-2: `on_write_failure` placement in YAML.** The YAML generator emits `on_write_failure` at the sink's top level rather than nested inside `options`. Verify this matches the engine's `load_settings()` parser for sinks. If the engine expects it inside `options`, the generated YAML will silently produce invalid configurations that pass unit tests but fail at execution time. The `load_settings()` round-trip test (W-4C-1) will catch this if implemented.

**Parallelism note (W7):** 4C only imports from `elspeth.web.composer.state` (4A output), not from `tools.py` (4B output). **4B and 4C can run in parallel after 4A completes.** The 4B plan header overstates the dependency.

**No blocking issues for 4C.**
