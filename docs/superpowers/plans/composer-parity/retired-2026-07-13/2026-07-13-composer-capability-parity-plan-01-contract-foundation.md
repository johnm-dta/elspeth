# Composer Capability Parity Plan 01: Contract Foundation Implementation Plan

> **RETIRED (2026-07-17): DO NOT EXECUTE.** See
> [the current disposition](../2026-07-17-current-plan-disposition.md).

**Goal:** Close the canonical coalesce failure-routing gap, fix secret-reference construction probes, and make drift between the LLM-facing `set_pipeline` declaration and the typed validation boundary fail deterministically.

**Architecture:** First complete the canonical runtime contract by giving coalesce failures an audited sink route. Then expose a read-only accessor for the registered declaration, compare its complete structural shape with `SetPipelineArgumentsModel`, and resolve persisted secret-reference markers only inside constructor probes. No planner or guided persistence behavior changes in this slice.

**Tech Stack:** Python 3.12+, Pydantic v2 JSON Schema, pytest, existing plugin manager and composer validation.

---

## File structure

**Create:**
- `src/elspeth/web/composer/tools/schema_contract.py` — immutable canonical schema accessor and structural-shape projection.
- `tests/unit/web/composer/test_set_pipeline_schema_contract.py` — declaration/model parity and immutability tests.
- `tests/integration/pipeline/test_composer_secret_ref_probe.py` — real LLM structured-output/fork/coalesce regression.
- `tests/integration/pipeline/test_coalesce_error_routing.py` — canonical branch/coalesce failure-route regression.

**Modify:**
- `src/elspeth/web/composer/tools/sessions.py` — close any schema discrepancies identified by the parity test.
- `src/elspeth/web/composer/state.py` — normalize secret refs before every composer transform-constructor probe.
- `src/elspeth/core/config.py` — add typed optional coalesce failure destination.
- `src/elspeth/core/dag/builder.py` — validate and bind the coalesce failure sink.
- `src/elspeth/engine/dag_navigator.py` — expose the graph-owned failure destination.
- `src/elspeth/engine/orchestrator/outcomes.py` — route failed coalesce outcomes exactly once.
- `src/elspeth/web/composer/state.py` — preserve the coalesce route in canonical state.
- `src/elspeth/web/composer/yaml_generator.py` — serialize the route.
- `src/elspeth/web/composer/yaml_importer.py` — restore the route.

### Task 0: Complete canonical coalesce failure routing before schema lock

**Files:**
- Modify: `src/elspeth/core/config.py`
- Modify: `src/elspeth/core/dag/builder.py`
- Modify: `src/elspeth/engine/dag_navigator.py`
- Modify: `src/elspeth/engine/orchestrator/outcomes.py`
- Modify: `src/elspeth/web/composer/state.py`
- Modify: `src/elspeth/web/composer/yaml_generator.py`
- Modify: `src/elspeth/web/composer/yaml_importer.py`
- Test: `tests/unit/core/test_config.py`
- Test: `tests/unit/engine/orchestrator/test_outcomes.py`
- Test: `tests/unit/web/composer/test_yaml_generator.py`
- Test: `tests/unit/web/composer/test_yaml_importer.py`
- Create: `tests/integration/pipeline/test_coalesce_error_routing.py`

- [ ] **Step 1: Write the canonical failure-route regression**

Build a `require_all`/`union` coalesce with `on_error="failures"`. Assert a
branch loss/incomplete barrier produces one audited coalesce failure and one
failure-sink row, while successful pairs continue downstream. Add config tests
for unknown, empty, and colliding sink names.

- [ ] **Step 2: Add one typed graph-owned route**

Add optional `on_error: str | None` to `CoalesceSettings`, validate it exactly
like other sink destinations, and bind it in `core/dag/builder.py`. Carry the
binding through `DagNavigator` and canonical composer state/YAML import/export;
do not infer it in the executor or create a composer-only field. Add a
state -> YAML -> state round trip and runtime-agreement assertion.

- [ ] **Step 3: Route all failed outcome arms once**

Use one orchestrator helper for intake, timeout, collision, and end-of-source
flush failures. Preserve existing `rows_coalesce_failed` and durable barrier
audit semantics. A configured failure sink receives one failure row per logical
failed coalesce; provider/branch failures must not duplicate it.

- [ ] **Step 4: Prove compatibility and commit**

```bash
uv run pytest tests/unit/core/test_config.py tests/unit/engine/orchestrator/test_outcomes.py tests/unit/web/composer/test_yaml_generator.py tests/unit/web/composer/test_yaml_importer.py tests/integration/pipeline/test_coalesce_error_routing.py tests/integration/pipeline/test_composer_runtime_agreement.py -q
git add src/elspeth/core/config.py src/elspeth/core/dag/builder.py src/elspeth/engine/dag_navigator.py src/elspeth/engine/orchestrator/outcomes.py src/elspeth/web/composer/state.py src/elspeth/web/composer/yaml_generator.py src/elspeth/web/composer/yaml_importer.py tests/unit/core/test_config.py tests/unit/engine/orchestrator/test_outcomes.py tests/unit/web/composer/test_yaml_generator.py tests/unit/web/composer/test_yaml_importer.py tests/integration/pipeline/test_coalesce_error_routing.py tests/integration/pipeline/test_composer_runtime_agreement.py
git commit -m "feat(coalesce): route failures to a canonical sink"
```

### Task 1: Publish the registered canonical schema without exposing mutable state

**Files:**
- Create: `src/elspeth/web/composer/tools/schema_contract.py`
- Test: `tests/unit/web/composer/test_set_pipeline_schema_contract.py`

- [ ] **Step 1: Write the failing accessor test**

```python
from elspeth.web.composer.tools.schema_contract import canonical_set_pipeline_schema


def test_canonical_set_pipeline_schema_is_a_defensive_copy() -> None:
    first = canonical_set_pipeline_schema()
    second = canonical_set_pipeline_schema()
    assert first == second
    assert first is not second
    first["properties"].pop("nodes")
    assert "nodes" in canonical_set_pipeline_schema()["properties"]
```

- [ ] **Step 2: Run the test and verify the red state**

Run: `uv run pytest tests/unit/web/composer/test_set_pipeline_schema_contract.py::test_canonical_set_pipeline_schema_is_a_defensive_copy -q`

Expected: FAIL with `ImportError: cannot import name 'canonical_set_pipeline_schema'`.

- [ ] **Step 3: Implement the accessor**

```python
# src/elspeth/web/composer/tools/schema_contract.py
from __future__ import annotations

from copy import deepcopy
from typing import Any

def canonical_set_pipeline_schema() -> dict[str, Any]:
    """Return a defensive copy of the registered set_pipeline JSON Schema."""
    # Local import avoids a registry-initialisation cycle. The registered
    # definition is the LLM-facing authority; never import the private
    # declaration object as a second access path.
    from elspeth.web.composer.tools._dispatch import get_tool_definitions

    definition = next(item for item in get_tool_definitions() if item["name"] == "set_pipeline")
    return deepcopy(definition["parameters"])
```

Keep the accessor in its internal schema-contract module. Do not add a dead
facade export; Plan 02's planner imports this module directly as its first
production consumer.

- [ ] **Step 4: Run the test and verify green**

Run: `uv run pytest tests/unit/web/composer/test_set_pipeline_schema_contract.py::test_canonical_set_pipeline_schema_is_a_defensive_copy -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools/schema_contract.py tests/unit/web/composer/test_set_pipeline_schema_contract.py
git commit -m "refactor(composer): expose canonical pipeline schema"
```

### Task 2: Enforce declaration/Pydantic structural compatibility

**Files:**
- Modify: `src/elspeth/web/composer/tools/schema_contract.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py:875-1085`
- Test: `tests/unit/web/composer/test_set_pipeline_schema_contract.py`

- [ ] **Step 1: Add the recursive structural projection and failing equality test**

```python
# add to schema_contract.py
from collections.abc import Mapping
from elspeth.contracts.hashing import canonical_json


def _resolve_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if ref is None:
        return schema
    prefix = "#/$defs/"
    if not isinstance(ref, str) or not ref.startswith(prefix):
        raise ValueError(f"unsupported schema ref {ref!r}")
    return root["$defs"][ref.removeprefix(prefix)]


def structural_schema_shape(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root_schema = schema if root is None else root
    current = _resolve_ref(schema, root_schema)
    combinator = next((key for key in ("anyOf", "oneOf", "allOf") if key in current), None)
    if combinator is not None:
        branches = current[combinator]
        non_null = [branch for branch in branches if branch.get("type") != "null"]
        nullable = len(non_null) != len(branches)
        return {
            combinator: sorted(
                (structural_schema_shape(branch, root=root_schema) for branch in non_null),
                key=canonical_json,
            ),
            "nullable": nullable,
        }
    result: dict[str, Any] = {}
    if "type" in current:
        result["type"] = current["type"]
    if "required" in current:
        result["required"] = sorted(current["required"])
    if current.get("type") == "object":
        result["additionalProperties"] = current.get("additionalProperties", True)
        result["properties"] = {
            name: structural_schema_shape(child, root=root_schema)
            for name, child in sorted(current.get("properties", {}).items())
        }
        additional = current.get("additionalProperties")
        if isinstance(additional, Mapping):
            result["additionalProperties"] = structural_schema_shape(additional, root=root_schema)
    if current.get("type") == "array":
        result["items"] = structural_schema_shape(current["items"], root=root_schema)
    if "enum" in current:
        result["enum"] = list(current["enum"])
    for keyword in (
        "const",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "pattern",
        "uniqueItems",
    ):
        if keyword in current:
            result[keyword] = current[keyword]
    return result
```

```python
# add to test_set_pipeline_schema_contract.py
from elspeth.web.composer.redaction import SetPipelineArgumentsModel
from elspeth.web.composer.tools.schema_contract import structural_schema_shape


def test_llm_declaration_matches_typed_set_pipeline_boundary() -> None:
    declared = canonical_set_pipeline_schema()
    typed = SetPipelineArgumentsModel.model_json_schema()
    assert structural_schema_shape(declared) == structural_schema_shape(typed)
```

- [ ] **Step 2: Run the equality test and capture every real mismatch**

Run: `uv run pytest tests/unit/web/composer/test_set_pipeline_schema_contract.py::test_llm_declaration_matches_typed_set_pipeline_boundary -vv`

Expected: FAIL. The initial diff must identify current declaration omissions such
as nested `additionalProperties: false` and fully typed `trigger` properties;
do not weaken the projection to hide mismatches.

- [ ] **Step 3: Make the registered declaration structurally match the model**

In `_SET_PIPELINE_DECLARATION.json_schema`:

```python
# Add to every closed nested object: source, inline_blob, nodes[*], trigger,
# edges[*], outputs[*], and metadata.
"additionalProperties": False,
```

Replace the untyped trigger object with the model's exact fields:

```python
"trigger": {
    "type": "object",
    "properties": {
        "count": {"type": ["integer", "null"]},
        "timeout_seconds": {"type": ["number", "null"]},
        "condition": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
},
```

Represent Pydantic optionals as nullable only where the declaration currently
advertises explicit JSON `null`; otherwise update the Pydantic-shape projection
with this single rule before comparison:

```python
# Optional field means omission is allowed. It does not imply the LLM schema
# must advertise explicit null unless the declaration includes null.
result.pop("nullable", None)
```

Do not ignore properties, required sets, array item shapes, mapping value
shapes, nested strictness, or supported union branches.

Add a distinct named-source boundary model so the typed model matches the
canonical v1 restriction instead of accepting blob fields and rejecting them
later:

```python
class _SetPipelineNamedSourceModel(BaseModel):
    plugin: str
    on_success: str
    options: _LlmJsonObject = Field(default_factory=dict)
    on_validation_failure: str | None = None

    model_config = ConfigDict(extra="forbid")


class SetPipelineArgumentsModel(BaseModel):
    source: _SetPipelineSourceModel | None = None
    sources: dict[str, _SetPipelineNamedSourceModel] | None = None
    # nodes / edges / outputs / metadata unchanged
```

- [ ] **Step 4: Add mutation-control tests**

```python
import copy
import pytest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["properties"]["nodes"]["items"]["properties"].pop("fork_to"),
        lambda s: s["properties"]["nodes"]["items"]["properties"].pop("merge"),
        lambda s: s["properties"].pop("sources"),
        lambda s: s["properties"]["outputs"]["items"].pop("additionalProperties"),
    ],
)
def test_structural_guard_detects_narrowed_authoring_schema(mutate) -> None:  # noqa: ANN001
    declared = copy.deepcopy(canonical_set_pipeline_schema())
    mutate(declared)
    typed = SetPipelineArgumentsModel.model_json_schema()
    assert structural_schema_shape(declared) != structural_schema_shape(typed)
```

- [ ] **Step 5: Run the contract suite**

Run: `uv run pytest tests/unit/web/composer/test_set_pipeline_schema_contract.py -q`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/tools/schema_contract.py src/elspeth/web/composer/tools/sessions.py tests/unit/web/composer/test_set_pipeline_schema_contract.py
git commit -m "test(composer): lock canonical pipeline schema parity"
```

### Task 3: Resolve secret-reference markers inside composer construction probes

**Files:**
- Modify: `src/elspeth/web/composer/state.py:548-584,1394-1415,1449-1472,1638-1660`
- Test: `tests/integration/pipeline/test_composer_secret_ref_probe.py`

- [ ] **Step 1: Add the real structured-output regression**

Build a `CompositionState` with an observed two-field source, a fork gate, two
`llm` transforms whose `api_key` is `{"secret_ref": "OPENROUTER_API_KEY"}` and
whose structured queries guarantee red/blue fields, a `require_all`/`union`
coalesce, a `field_mapper` requiring the combined fields, and JSON success/error
outputs. Assert:

```python
validation = state.validate()
assert validation.is_valid, [entry.to_dict() for entry in validation.errors]
assert not any("Computed contract probe" in warning.message for warning in validation.warnings)
cleanup = next(node for node in state.nodes if node.id == "cleanup")
expected_fields = {
    "color_name",
    "hex",
    "blue_amount",
    "blue_confidence",
    "blue_reason",
    "red_amount",
    "red_confidence",
    "red_reason",
}
assert set(cleanup.options["required_input_fields"]) == expected_fields
cleanup_contracts = [entry for entry in validation.edge_contracts if entry.to_id == "cleanup"]
assert cleanup_contracts
assert all(set(entry.producer_guarantees) == expected_fields for entry in cleanup_contracts)
```

Add a negative control that removes `red_reason` from the red LLM structured
output declaration and assert the named downstream guarantee error identifies
`cleanup` and `red_reason`.

Use the exact LLM query shape from `tests/unit/plugins/llm/test_transform.py` and
the fork/coalesce node shape from
`src/elspeth/web/composer/recipes.py::_build_fork_coalesce_truncate_recipe`.

- [ ] **Step 2: Run the regression and verify the red state**

Run: `uv run pytest tests/integration/pipeline/test_composer_secret_ref_probe.py -q`

Expected: FAIL because the constructor probe passes a mapping into the LLM
config's string credential field, suppressing computed guarantees.

- [ ] **Step 3: Add one helper and use it at every composer probe site**

```python
# inside CompositionState.validate(), beside _probe_transform_construction
from elspeth.core.secrets import redact_secret_refs_for_validation
from elspeth.web.interpretation_state import strip_authoring_options


def _constructor_probe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    thawed = deep_thaw(options)
    stripped = strip_authoring_options(thawed)
    return redact_secret_refs_for_validation(stripped)
```

Place the helper at module scope so both `_producer_declared_field_type()` and
`CompositionState.validate()` use it. Replace all four resolver-free probe
arguments in `state.py`:

```python
get_shared_plugin_manager().create_transform(
    plugin_name,
    _constructor_probe_options(options),
)
```

The helper is only for resolver-free validation. Persisted state and runtime
secret resolution remain unchanged.

- [ ] **Step 4: Run narrow and neighboring agreement tests**

Run: `uv run pytest tests/integration/pipeline/test_composer_secret_ref_probe.py tests/integration/pipeline/test_composer_runtime_agreement.py -q`

Expected: PASS with no new probe warning.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/state.py tests/integration/pipeline/test_composer_secret_ref_probe.py
git commit -m "fix(composer): resolve secret refs in contract probes"
```

### Task 4: Verify the foundation slice

**Files:** no new files.

- [ ] **Step 1: Run the Plan 01 gate**

Run:

```bash
uv run pytest \
  tests/unit/web/composer/test_set_pipeline_schema_contract.py \
  tests/integration/pipeline/test_coalesce_error_routing.py \
  tests/integration/pipeline/test_composer_secret_ref_probe.py \
  tests/integration/pipeline/test_composer_runtime_agreement.py -q
```

Expected: PASS.

- [ ] **Step 2: Run static checks on touched Python files**

```bash
uv run ruff check \
  src/elspeth/web/composer/tools/schema_contract.py \
  src/elspeth/web/composer/tools/sessions.py \
  src/elspeth/web/composer/state.py \
  src/elspeth/core/config.py \
  src/elspeth/core/dag/builder.py \
  src/elspeth/engine/dag_navigator.py \
  src/elspeth/engine/orchestrator/outcomes.py \
  src/elspeth/web/composer/yaml_generator.py \
  src/elspeth/web/composer/yaml_importer.py \
  tests/unit/web/composer/test_set_pipeline_schema_contract.py \
  tests/integration/pipeline/test_coalesce_error_routing.py \
  tests/integration/pipeline/test_composer_secret_ref_probe.py
uv run mypy src/elspeth/web/composer/tools/schema_contract.py src/elspeth/web/composer/state.py
git diff --check
```

Expected: all commands exit 0.

- [ ] **Step 3: Record Plan 01 evidence in the master ledger**

Update only the Plan 01 row with the commit hashes and `PASS`; then commit:

```bash
git add docs/superpowers/plans/composer-parity/2026-07-13-composer-capability-parity-implementation-plan.md
git commit -m "docs: record composer parity contract foundation"
```
