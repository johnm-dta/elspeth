# Composer One-Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Collapse the three parallel typing systems for composer knobs (plugin Pydantic JSON Schema, recipe `SlotSpec`, frontend `FieldType`) into one canonical wire shape (`KnobSchema`), lowered server-side at catalog load, rendered by a single frontend widget.

**Tracking epic:** `elspeth-a5fbc1ed4a` — "Composer one-knob configuration — single guided knob schema across plugin options and recipes"

**Implementation closeout:** Completed on 2026-05-14 and merged into `RC5.2` at `2135908f6`. The tracking epic `elspeth-a5fbc1ed4a` is closed. All task checkboxes below are marked complete to make this plan usable as an execution record, not just a pre-implementation handoff.

**Verification closeout:** Final verification from `/home/john/elspeth` passed: `ruff check src tests`, `mypy src`, tier-model, freeze-guards, plugin options metadata lint, metadata lint tests, guided/composer backend sweep (`235 passed`), frontend vitest (`478 passed`), frontend typecheck, and frontend build. The frontend build retained the existing Vite dynamic-import/chunk-size warnings.

**Deployment shoe-in:** Before deploying this change, the operator must delete/recreate the guided sessions DB because the guided session schema version intentionally rejects older persisted sessions.

**Architecture:** Backend lowers Pydantic-emitted JSON Schema into a fallback-capable `KnobSchema` at `CatalogServiceImpl.__init__` (startup-time, not per-request). Discriminated-union plugins (LLMTransform) flatten via `visible_when` predicates. Valid rich live schemas lower to explicit `json-object`, `json-array`, or `json-value` fallback knobs rather than failing startup. Frontend `SchemaFormTurn` dispatches on `kind`; recipe rendering folds into the same turn type via a tagged-union `SchemaFormPayload` with `mode` discriminator. Step 1 schema-form is prefilled from persisted `SourceInspectionFacts`. Hidden-field submissions are rejected by the backend with FastAPI's `{"detail": {"code": ...}}` envelope, not silently dropped.

**Tech Stack:** Python 3.13, Pydantic v2, FastAPI, TypeScript, React 18, pytest, Hypothesis, vitest.

**Source documents:**
- Spec: `docs/superpowers/specs/2026-05-14-composer-one-knob-design.md`
- Investigation: `.scratch/composer-knob-wiring-report.md` (kept until commit)

**Readiness repairs applied after no-go review:**
- Live catalog lowering must include `json-object`, `json-array`, and `json-value`; the current catalog contains `$ref`, nullable array/object, object-map, array-of-model, complex-`anyOf`, and top-level-`oneOf` shapes.
- Live symbols are `CatalogServiceImpl`, `get_shared_plugin_manager()`, plugin `config_model`, and `GuidedSession.initial()`.
- `SourceInspectionFacts` must round-trip all current fields and needs a new inverse serializer; delimiter/encoding prefill is out unless facts are extended first.
- Recipe context comes from `get_recipe(match.recipe_name).description`; `RecipeMatch` has no `recipe_description`.
- Recipe rendering must preserve both accept and `build_manually`; if `TurnType.RECIPE_OFFER` remains, protocol validation must be updated for the new payload.
- Route tests must assert `body["detail"]["code"]` for `HTTPException(detail={...})`.

**Plan-review disposition applied on 2026-05-14:** accepted the eight blocking findings from `2026-05-14-composer-one-knob.review.json`. This revision rewrites Task 9, adds protocol validation updates to Tasks 10 and 15, replaces stale APIs across Tasks 5/8/10/12/15/16/17, adds Task 6.5 for property/snapshot coverage, adds nullable return-path tests, makes hidden-field rejection auditable, adds tier/freeze CI gates, and makes Step 1 inspection prefill non-tautological. The structural warnings accepted in the same pass are: extract recipe slot contracts instead of lazy imports, move `knob_schema.py` under `web/catalog/`, execute Phase D in a worktree with WIP commits, and move metadata audit/fill before the atomic wire migration.

---

## File Structure

### Backend — new files
- `src/elspeth/contracts/discriminated.py` — `DiscriminatedPlugin` Protocol (L0 contracts layer)
- `src/elspeth/contracts/composer_slots.py` — `SlotType`, `SlotSpec` contract shared by recipes and catalog lowering
- `src/elspeth/web/catalog/knob_schema.py` — `KnobField`, `KnobSchema`, `VisibilityPredicate`, `RecipeContext`, tagged-union `SchemaFormPayload`, lowering functions
- `scripts/cicd/enforce_options_metadata.py` — CI lint
- `config/cicd/enforce_options_metadata/allowlist.yaml` — empty initial allowlist

### Backend — modified files
- `src/elspeth/web/catalog/schemas.py:44` — add `knob_schema: KnobSchema` to `PluginSchemaInfo`
- `src/elspeth/web/catalog/service.py:36-83` — `CatalogServiceImpl.__init__` pre-materialises `knob_schema` for every registered plugin; `get_schema` reads cache
- `src/elspeth/web/composer/recipes.py` — import `SlotType` and `SlotSpec` from `contracts.composer_slots`; keep recipe registry behavior unchanged
- `src/elspeth/web/composer/guided/protocol.py:53-59` — replace `SchemaFormPayload` with tagged-union from `web.catalog.knob_schema`; update `_REQUIRED_KEYS` / `_NESTED_SHAPES`
- `src/elspeth/web/composer/guided/state_machine.py:397` — add `step_1_inspection_facts: SourceInspectionFacts | None`
- `src/elspeth/web/composer/guided/emitters.py:115-143, 181-209, 326-343` — switch the three `*_schema_form_turn` emitters to `KnobSchema`; thread inspection facts into Step 1
- `src/elspeth/web/sessions/routes.py:2180, 2584-2815, 3037-3112, 5215` — hidden-field rejection; recipe-fold dispatch update; thread inspection-facts through Step 1 dispatch
- `src/elspeth/plugins/transforms/llm/transform.py` — add `discriminated_variants()` classmethod to LLMTransform

### Frontend — modified files
- `src/elspeth/web/frontend/src/types/guided.ts` — replace `SchemaFormPayload` with tagged union; new `KnobField`, `VisibilityPredicate`, `RecipeContext` types
- `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx` — full rewrite: discriminator dispatch on `kind`, `visible_when` evaluation, mode-based composition
- `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx` — rewrite tests for new shape

### Frontend — new files
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeContextHeader.tsx` — small peer component for recipe metadata banner
- Covered by `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx`; no standalone `RecipeContextHeader.test.tsx` was landed.

### Frontend — deleted files (after step 4 lands)
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx`
- `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.test.tsx`

### Test files — new
- `tests/unit/web/composer/test_knob_schema.py`
- `tests/unit/web/composer/test_knob_schema_recipe_adapter.py`
- `tests/unit/web/composer/test_knob_schema_discriminated.py`
- `tests/unit/web/catalog/test_knob_schema_properties.py`
- `tests/golden/web/catalog/knob_schema/*.json`
- `tests/unit/web/catalog/test_eager_lowering.py`
- `tests/integration/web/composer/guided/test_hidden_field_rejection.py`
- `tests/unit/scripts/cicd/test_enforce_options_metadata.py`

### Plugin configuration models — bulk metadata fill (Task 16)
- Every metadata-bearing plugin configuration model under `src/elspeth/plugins/{sources,transforms,sinks}/...` that lacks `title` or `description` on any field, including provider-specific discriminated variants returned by `discriminated_variants()`

---

## Phase A — Backend foundation (commit 1 of §9 step 1)

### Task 1: DiscriminatedPlugin protocol

**Files:**
- Create: `src/elspeth/contracts/discriminated.py`
- Test: `tests/unit/contracts/test_discriminated.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/contracts/test_discriminated.py
from typing import Protocol, runtime_checkable
from pydantic import BaseModel
from elspeth.contracts.discriminated import DiscriminatedPlugin


def test_plugin_with_discriminated_variants_is_recognised():
    class _AzureCfg(BaseModel):
        provider: str = "azure"

    class _OpenRouterCfg(BaseModel):
        provider: str = "openrouter"

    class MyPlugin:
        @classmethod
        def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
            return ("provider", {"azure": _AzureCfg, "openrouter": _OpenRouterCfg})

    assert isinstance(MyPlugin, DiscriminatedPlugin)


def test_plugin_without_method_is_not_discriminated():
    class MyPlugin:
        pass

    assert not isinstance(MyPlugin, DiscriminatedPlugin)
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_discriminated.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'elspeth.contracts.discriminated'`

- [x] **Step 3: Implement the protocol**

```python
# src/elspeth/contracts/discriminated.py
"""Protocol for plugins whose config schema is a Pydantic discriminated union.

Plugins implementing this protocol expose their variant model classes to the
composer's knob-schema lowering. See
docs/superpowers/specs/2026-05-14-composer-one-knob-design.md §5."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class DiscriminatedPlugin(Protocol):
    """Plugin protocol contract for discriminated-union config models.

    Implementing classes return:
        (discriminator_field_name, {literal_value: variant_model_cls})

    The discriminator field name MUST match the field on each variant model
    that carries the variant identifier (commonly ``provider``, ``kind``,
    ``type``). Variant models MUST be ``Annotated[Union[...], Field(discriminator=...)]``
    forms; the pydantic-v2 ``Discriminator(...)`` class form is not supported."""

    @classmethod
    def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]: ...
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_discriminated.py -v`
Expected: PASS — 2 passed

- [x] **Step 5: Commit**

```bash
git add src/elspeth/contracts/discriminated.py tests/unit/contracts/test_discriminated.py
git commit -m "feat(contracts): add DiscriminatedPlugin protocol for composer lowering

Plugins with Pydantic discriminated-union config models expose their variant model
classes via discriminated_variants(). Consumed by KnobSchema.from_discriminated_model
in the upcoming knob_schema.py — see spec §5.
"
```

---

### Task 2: KnobSchema core types

**Files:**
- Create: `src/elspeth/web/catalog/knob_schema.py` (initial — types only)
- Test: `tests/unit/web/composer/test_knob_schema.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/web/composer/test_knob_schema.py
from elspeth.web.catalog.knob_schema import (
    KnobField,
    KnobSchema,
    RecipeContext,
    VisibilityPredicate,
    _PluginOptionsPayload,
    _RecipeDecisionPayload,
)


def test_knob_field_minimal_shape():
    field: KnobField = {
        "name": "path",
        "label": "Input file path",
        "kind": "text",
        "required": True,
        "nullable": False,
    }
    assert field["name"] == "path"


def test_recipe_decision_payload_carries_recipe_context():
    payload: _RecipeDecisionPayload = {
        "mode": "recipe_decision",
        "knobs": {"fields": []},
        "prefilled": {},
        "recipe_context": {
            "recipe_name": "classify-rows-llm-jsonl",
            "description": "Classify each row via an LLM",
            "alternatives": ["build_manually"],
        },
    }
    assert payload["mode"] == "recipe_decision"
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema.py -v`
Expected: FAIL — `ImportError: cannot import name 'KnobField' from ...`

- [x] **Step 3: Implement the types module**

```python
# src/elspeth/web/catalog/knob_schema.py
"""Composer one-knob wire shape — closed JSON-Schema subset for plugin options.

Lowering happens at catalog load time inside CatalogServiceImpl.__init__; this
module exposes the result types and the lowering entry points. See
docs/superpowers/specs/2026-05-14-composer-one-knob-design.md.

Trust tier: L3 web layer. KnobSchema instances are Tier 1 (we wrote them from
plugin models we control); prefilled values from SourceInspectionFacts remain
Tier 3.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


class VisibilityPredicate(TypedDict):
    """Conditional-visibility predicate for a KnobField.

    `field` MUST reference an earlier-declared KnobField in the same KnobSchema
    (forward references rejected at catalog load). `equals` is an exact value
    match against current form state. No other keys are permitted; predicates
    with extra keys raise KnobSchemaLoweringError at catalog load."""

    field: str
    equals: Any


class KnobField(TypedDict):
    name: str
    label: str
    description: NotRequired[str]
    kind: Literal[
        "text",
        "number-int",
        "number-float",
        "checkbox",
        "enum",
        "string-list",
        "blob-ref",
        "json-object",
        "json-array",
        "json-value",
    ]
    tier: NotRequired[Literal["essential", "common", "advanced"]]
    required: bool
    default: NotRequired[object]
    nullable: bool
    enum: NotRequired[list[str]]
    item_kind: NotRequired[Literal["text", "number-int", "number-float"]]
    visible_when: NotRequired[VisibilityPredicate]


class KnobSchema(TypedDict):
    fields: list[KnobField]


class RecipeContext(TypedDict):
    recipe_name: str
    description: str
    alternatives: list[str]


class _PluginOptionsPayload(TypedDict):
    mode: Literal["plugin_options"]
    plugin: str
    knobs: KnobSchema
    prefilled: dict[str, object]


class _RecipeDecisionPayload(TypedDict):
    mode: Literal["recipe_decision"]
    knobs: KnobSchema
    prefilled: dict[str, object]
    recipe_context: RecipeContext


SchemaFormPayload = _PluginOptionsPayload | _RecipeDecisionPayload


class KnobSchemaLoweringError(Exception):
    """Raised at catalog-load time when a plugin's Pydantic schema is malformed
    or violates one-knob invariants.

    Valid-but-rich fields lower to json-object, json-array, or json-value
    fallback knobs. True invariant violations halt startup."""

    def __init__(
        self,
        *,
        plugin_kind: str,
        plugin_name: str,
        field_path: str,
        constraint: str,
        remediation: str,
    ) -> None:
        message = (
            f"Plugin {plugin_kind}/{plugin_name} field {field_path!r}: "
            f"{constraint}. Remediation: {remediation}"
        )
        super().__init__(message)
        self.plugin_kind = plugin_kind
        self.plugin_name = plugin_name
        self.field_path = field_path
        self.constraint = constraint
        self.remediation = remediation
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema.py -v`
Expected: PASS — 2 passed

- [x] **Step 5: Commit**

```bash
git add src/elspeth/web/catalog/knob_schema.py tests/unit/web/composer/test_knob_schema.py
git commit -m "feat(composer): add KnobSchema wire types for one-knob lowering

KnobField, VisibilityPredicate, tagged-union SchemaFormPayload, and the
KnobSchemaLoweringError exception type. Lowering functions land in subsequent
tasks. See spec §4 + §4.1.
"
```

---

### Task 3: KnobSchema.from_model — single-model lowering

**Files:**
- Modify: `src/elspeth/web/catalog/knob_schema.py` (add `from_model` classmethod and helpers)
- Test: `tests/unit/web/composer/test_knob_schema_from_model.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/web/composer/test_knob_schema_from_model.py
from typing import Annotated, Literal
from pydantic import BaseModel, Field

from elspeth.web.catalog.knob_schema import (
    KnobSchema,
    KnobSchemaLoweringError,
    lower_model_to_knob_schema,
)


def test_simple_str_field_lowers_to_text():
    class Opts(BaseModel):
        name: Annotated[str, Field(title="Name", description="Your name")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    assert len(ks["fields"]) == 1
    f = ks["fields"][0]
    assert f["name"] == "name"
    assert f["label"] == "Name"
    assert f["description"] == "Your name"
    assert f["kind"] == "text"
    assert f["required"] is True
    assert f["nullable"] is False


def test_optional_str_lowers_to_nullable_text():
    class Opts(BaseModel):
        encoding: Annotated[str | None, Field(title="Encoding", description="File encoding")] = None

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "text"
    assert f["nullable"] is True
    assert f["required"] is False
    assert "default" in f and f["default"] is None


def test_int_with_default_keeps_default():
    class Opts(BaseModel):
        skip_rows: Annotated[int, Field(title="Skip rows", description="Rows to skip")] = 0

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "number-int"
    assert f["default"] == 0


def test_required_int_omits_default():
    class Opts(BaseModel):
        port: Annotated[int, Field(title="Port", description="TCP port")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert "default" not in f


def test_literal_lowers_to_enum():
    class Opts(BaseModel):
        mode: Annotated[Literal["a", "b"], Field(title="Mode", description="Pick one")]

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "enum"
    assert f["enum"] == ["a", "b"]


def test_tier_annotation_emitted_when_set():
    class Opts(BaseModel):
        debug: Annotated[
            bool,
            Field(
                title="Debug",
                description="Verbose output",
                json_schema_extra={"composer_tier": "advanced"},
            ),
        ] = False

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["tier"] == "advanced"


def test_tier_absent_when_unannotated():
    class Opts(BaseModel):
        debug: Annotated[bool, Field(title="Debug", description="Verbose output")] = False

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert "tier" not in f


def test_string_list_kind_for_list_of_str():
    class Opts(BaseModel):
        tags: Annotated[list[str], Field(title="Tags", description="Tag list")] = []

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "string-list"
    assert f["item_kind"] == "text"


def test_object_map_lowers_to_json_object():
    class Opts(BaseModel):
        weird: Annotated[dict[str, int], Field(title="W", description="W")] = {}

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "json-object"


def test_non_string_array_lowers_to_json_array():
    class Opts(BaseModel):
        rows: Annotated[list[dict[str, str]], Field(title="Rows", description="Rows")] = []

    ks = lower_model_to_knob_schema(Opts, plugin_kind="source", plugin_name="test")
    f = ks["fields"][0]
    assert f["kind"] == "json-array"


def test_optional_int_clear_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}, "skip_rows": None},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert validated.options["skip_rows"] is None


def test_optional_str_absent_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert "encoding" not in validated.options


def test_optional_str_clear_round_trips_through_set_source_validator():
    from elspeth.web.composer.redaction import SetSourceArgumentsModel

    payload = {
        "plugin": "csv",
        "on_success": "continue",
        "on_validation_failure": "quarantine",
        "options": {"schema": {"mode": "observed"}, "encoding": None},
    }
    validated = SetSourceArgumentsModel.model_validate(payload)
    assert validated.options["encoding"] is None
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_from_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'lower_model_to_knob_schema'`

- [x] **Step 3: Implement `lower_model_to_knob_schema`**

Append to `src/elspeth/web/catalog/knob_schema.py`:

```python
import types
from collections.abc import Mapping
from typing import Union, get_args, get_origin
from pydantic import BaseModel
from pydantic.fields import FieldInfo


_TYPE_TO_KIND: dict[type, Literal["text", "number-int", "number-float", "checkbox"]] = {
    str: "text",
    int: "number-int",
    float: "number-float",
    bool: "checkbox",
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner_type, nullable). Handles `T | None` and `Optional[T]` shapes."""
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(annotation):
            return args[0], True
    return annotation, False


def _kind_for_scalar(
    inner: Any,
    *,
    plugin_kind: str,
    plugin_name: str,
    field_path: str,
) -> tuple[Literal["text", "number-int", "number-float", "checkbox", "enum", "json-value"], list[str] | None]:
    """Map a Python scalar/Literal type to a KnobField kind. Returns (kind, enum_values).

    Unknown valid shapes fall back to json-value; only malformed schemas or
    one-knob invariant violations raise KnobSchemaLoweringError.
    """
    if get_origin(inner) is Literal:
        values = [str(v) for v in get_args(inner)]
        return "enum", values
    if inner in _TYPE_TO_KIND:
        return _TYPE_TO_KIND[inner], None
    return "json-value", None


def _lower_field(
    name: str,
    info: FieldInfo,
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str,
) -> KnobField:
    annotation = info.annotation
    inner, nullable = _unwrap_optional(annotation)

    # list[str] → string-list
    if get_origin(inner) is list:
        list_args = get_args(inner)
        if len(list_args) == 1 and list_args[0] is str:
            field: KnobField = {
                "name": name,
                "label": info.title or name,
                "kind": "string-list",
                "item_kind": "text",
                "required": info.is_required(),
                "nullable": nullable,
            }
            if info.description:
                field["description"] = info.description
            _attach_default(field, info)
            _attach_tier(field, info)
            return field
        field: KnobField = {
            "name": name,
            "label": info.title or name,
            "kind": "json-array",
            "required": info.is_required(),
            "nullable": nullable,
        }
        if info.description:
            field["description"] = info.description
        _attach_default(field, info)
        _attach_tier(field, info)
        return field

    if get_origin(inner) in (dict, Mapping) or (isinstance(inner, type) and issubclass(inner, BaseModel)):
        field: KnobField = {
            "name": name,
            "label": info.title or name,
            "kind": "json-object",
            "required": info.is_required(),
            "nullable": nullable,
        }
        if info.description:
            field["description"] = info.description
        _attach_default(field, info)
        _attach_tier(field, info)
        return field

    kind, enum_values = _kind_for_scalar(
        inner, plugin_kind=plugin_kind, plugin_name=plugin_name, field_path=name
    )
    field = {
        "name": name,
        "label": info.title or name,
        "kind": kind,
        "required": info.is_required(),
        "nullable": nullable,
    }
    if info.description:
        field["description"] = info.description
    if enum_values is not None:
        field["enum"] = enum_values
    _attach_default(field, info)
    _attach_tier(field, info)
    return field


def _attach_default(field: KnobField, info: FieldInfo) -> None:
    if info.is_required():
        return  # required → no default on wire
    # PydanticUndefined would mean "no default declared on an optional field" — rare
    from pydantic_core import PydanticUndefined
    if info.default is PydanticUndefined:
        return
    field["default"] = info.default  # type: ignore[typeddict-item]


def _attach_tier(field: KnobField, info: FieldInfo) -> None:
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return
    tier = extra.get("composer_tier")
    if tier in ("essential", "common", "advanced"):
        field["tier"] = tier  # type: ignore[typeddict-item]


def lower_model_to_knob_schema(
    model_cls: type[BaseModel],
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str = "common",
) -> KnobSchema:
    """Lower a single-model Pydantic config class to KnobSchema.

    Discriminated unions go through lower_discriminated_to_knob_schema instead."""
    fields: list[KnobField] = []
    for name, info in model_cls.model_fields.items():
        fields.append(
            _lower_field(
                name,
                info,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                composer_tier_default=composer_tier_default,
            )
        )
    return {"fields": fields}
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_from_model.py -v`
Expected: PASS — includes forward-lowering tests plus nullable return-path validator coverage.

- [x] **Step 5: Commit**

```bash
git add src/elspeth/web/catalog/knob_schema.py tests/unit/web/composer/test_knob_schema_from_model.py
git commit -m "feat(composer): implement KnobSchema.from_model for single-model plugins

Handles scalars (str/int/float/bool), Literal-as-enum, Optional[T] as nullable,
list[str] as string-list, and rich valid shapes as json-object/json-array/json-value.
Tier annotation honoured when set; absent when unset per spec §4.
"
```

---

### Task 4: Extract SlotSpec contract + KnobSchema.from_slot_specs recipe adapter

**Files:**
- Create: `src/elspeth/contracts/composer_slots.py`
- Modify: `src/elspeth/web/composer/recipes.py`
- Modify: `src/elspeth/web/catalog/knob_schema.py`
- Test: `tests/unit/web/composer/test_knob_schema_recipe_adapter.py`

- [x] **Step 1: Extract SlotType and SlotSpec to contracts**

Move the existing `SlotType` and `SlotSpec` definitions from `src/elspeth/web/composer/recipes.py` to `src/elspeth/contracts/composer_slots.py`, then re-export/import them in `recipes.py`. This avoids a `web/catalog -> web/composer` dependency and removes the lazy-import anti-pattern from the adapter.

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_recipes.py tests/unit/web/composer/guided/test_recipe_match.py -v`
Expected: PASS; recipe registry and matching behavior unchanged.

- [x] **Step 2: Write the failing test (parametrised over SlotType)**

```python
# tests/unit/web/composer/test_knob_schema_recipe_adapter.py
import typing
import pytest

from elspeth.contracts.composer_slots import SlotSpec, SlotType
from elspeth.web.catalog.knob_schema import lower_slot_specs_to_knob_schema


def test_blob_id_slot_lowers_to_blob_ref():
    slots = {"source_blob": SlotSpec(slot_type="blob_id", required=True, description="Source CSV")}
    ks = lower_slot_specs_to_knob_schema(slots)
    f = ks["fields"][0]
    assert f["kind"] == "blob-ref"
    assert f["required"] is True


def test_str_list_slot_lowers_to_string_list():
    slots = {"keys": SlotSpec(slot_type="str_list", required=False, description="Merge keys")}
    ks = lower_slot_specs_to_knob_schema(slots)
    f = ks["fields"][0]
    assert f["kind"] == "string-list"
    assert f["item_kind"] == "text"


@pytest.mark.parametrize("slot_type", typing.get_args(SlotType))
def test_every_slot_type_has_a_mapping(slot_type):
    """Parametrised totality test — a new SlotType member fails this collection
    rather than crashing at runtime. Closes systems-thinking #5.3 second-order
    risk + python-eng Warning 3 from first review."""
    slots = {"x": SlotSpec(slot_type=slot_type, required=False, description="X")}
    ks = lower_slot_specs_to_knob_schema(slots)
    assert len(ks["fields"]) == 1
    assert "kind" in ks["fields"][0]
```

- [x] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_recipe_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'lower_slot_specs_to_knob_schema'`

- [x] **Step 4: Implement `lower_slot_specs_to_knob_schema`**

Append to `src/elspeth/web/catalog/knob_schema.py`:

```python
from collections.abc import Mapping
from elspeth.contracts.composer_slots import SlotType

_SLOT_TYPE_TO_KIND: dict[str, Literal["text", "number-int", "number-float", "blob-ref", "string-list"]] = {
    "blob_id": "blob-ref",
    "str": "text",
    "int": "number-int",
    "float": "number-float",
    "str_list": "string-list",
}


def lower_slot_specs_to_knob_schema(slots: Mapping[str, Any]) -> KnobSchema:
    """Adapter — lower recipe SlotSpec map to KnobSchema.

    Total over typing.get_args(SlotType). New slot types must extend the
    mapping and grow the test parametrisation in lockstep."""
    fields: list[KnobField] = []
    for name, spec in slots.items():
        slot_type = spec.slot_type
        kind = _SLOT_TYPE_TO_KIND[slot_type]
        field: KnobField = {
            "name": name,
            "label": name,
            "kind": kind,
            "required": spec.required,
            "nullable": not spec.required,
        }
        if spec.description:
            field["description"] = spec.description
        if kind == "string-list":
            field["item_kind"] = "text"
        if spec.default is not None:
            field["default"] = spec.default  # type: ignore[typeddict-item]
        fields.append(field)
    return {"fields": fields}
```

- [x] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_recipe_adapter.py -v`
Expected: PASS — at least 3 + N (N = number of SlotType members) tests pass.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/contracts/composer_slots.py src/elspeth/web/composer/recipes.py src/elspeth/web/catalog/knob_schema.py tests/unit/web/composer/test_knob_schema_recipe_adapter.py
git commit -m "feat(composer): add recipe-slot adapter to KnobSchema lowering

Recipes without backing plugin models still produce conformant KnobSchema.
Totality test parametrised over typing.get_args(SlotType) so a new member
breaks test collection instead of runtime. The adapter is the supported bridge
for non-plugin-backed recipes; see spec §5 + §12.
"
```

---

### Task 5: KnobSchema.from_discriminated_model + LLMTransform.discriminated_variants

**Files:**
- Modify: `src/elspeth/web/catalog/knob_schema.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py` (add classmethod)
- Test: `tests/unit/web/composer/test_knob_schema_discriminated.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/web/composer/test_knob_schema_discriminated.py
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import pytest

from elspeth.web.catalog.knob_schema import (
    KnobSchemaLoweringError,
    lower_discriminated_to_knob_schema,
)


class _AzureCfg(BaseModel):
    provider: Literal["azure"] = "azure"
    deployment: Annotated[str, Field(title="Deployment", description="Azure deployment name")]


class _OpenRouterCfg(BaseModel):
    provider: Literal["openrouter"] = "openrouter"
    model: Annotated[str, Field(title="Model", description="OpenRouter model id")]


class _StubPlugin:
    @classmethod
    def discriminated_variants(cls):
        return ("provider", {"azure": _AzureCfg, "openrouter": _OpenRouterCfg})


def test_discriminator_emitted_first_as_enum():
    ks = lower_discriminated_to_knob_schema(_StubPlugin, plugin_kind="transform", plugin_name="llm")
    first = ks["fields"][0]
    assert first["name"] == "provider"
    assert first["kind"] == "enum"
    assert set(first["enum"]) == {"azure", "openrouter"}


def test_variant_fields_get_visible_when():
    ks = lower_discriminated_to_knob_schema(_StubPlugin, plugin_kind="transform", plugin_name="llm")
    deployment = next(f for f in ks["fields"] if f["name"] == "deployment")
    assert deployment["visible_when"] == {"field": "provider", "equals": "azure"}
    model = next(f for f in ks["fields"] if f["name"] == "model")
    assert model["visible_when"] == {"field": "provider", "equals": "openrouter"}


def test_non_discriminated_plugin_raises():
    class _NotDiscriminated:
        pass
    with pytest.raises(KnobSchemaLoweringError) as exc:
        lower_discriminated_to_knob_schema(_NotDiscriminated, plugin_kind="transform", plugin_name="x")
    assert "discriminated_variants" in exc.value.constraint
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_discriminated.py -v`
Expected: FAIL — `ImportError: cannot import name 'lower_discriminated_to_knob_schema'`

- [x] **Step 3: Implement `lower_discriminated_to_knob_schema`**

Append to `src/elspeth/web/catalog/knob_schema.py`:

```python
def lower_discriminated_to_knob_schema(
    plugin_cls: type,
    *,
    plugin_kind: str,
    plugin_name: str,
    composer_tier_default: str = "common",
) -> KnobSchema:
    """Lower a discriminated-union plugin per spec §4.1.

    Discriminator emitted first as kind="enum"; each variant's fields receive
    visible_when={"field": <discriminator>, "equals": <variant-value>}."""
    discriminated_variants = getattr(plugin_cls, "discriminated_variants", None)
    if discriminated_variants is None or not callable(discriminated_variants):
        raise KnobSchemaLoweringError(
            plugin_kind=plugin_kind,
            plugin_name=plugin_name,
            field_path="<class>",
            constraint=(
                "plugin lacks discriminated_variants() classmethod required by "
                "DiscriminatedPlugin protocol"
            ),
            remediation=(
                "Implement discriminated_variants() returning "
                "(discriminator_field_name, {literal_value: variant_cls})."
            ),
        )
    discriminator, variants = discriminated_variants()

    # Emit discriminator first as an enum knob.
    fields: list[KnobField] = [
        {
            "name": discriminator,
            "label": discriminator,
            "kind": "enum",
            "enum": list(variants.keys()),
            "required": True,
            "nullable": False,
        }
    ]
    for variant_value, variant_cls in variants.items():
        for fname, info in variant_cls.model_fields.items():
            if fname == discriminator:
                continue  # already emitted
            inner_field = _lower_field(
                fname,
                info,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                composer_tier_default=composer_tier_default,
            )
            inner_field["visible_when"] = {"field": discriminator, "equals": variant_value}
            fields.append(inner_field)
    return {"fields": fields}
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_discriminated.py -v`
Expected: PASS — 3 passed

- [x] **Step 5: Add `discriminated_variants()` to LLMTransform**

Read the current shape:

Run: `grep -n "_PROVIDERS\|get_config_schema\|class LLMTransform" /home/john/elspeth/src/elspeth/plugins/transforms/llm/transform.py | head -20`

Add after the existing `get_config_schema` classmethod (location: same file, locate via grep):

```python
    @classmethod
    def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
        """Expose provider variants to the composer knob-schema lowering.

        See contracts/discriminated.py and the one-knob design spec §5."""
        return ("provider", {provider: config_cls for provider, (config_cls, _) in _PROVIDERS.items()})
```

- [x] **Step 6: Write LLMTransform integration test**

```python
# Append to tests/unit/web/composer/test_knob_schema_discriminated.py
def test_llm_transform_real_lowering():
    from elspeth.plugins.transforms.llm.transform import LLMTransform
    ks = lower_discriminated_to_knob_schema(
        LLMTransform, plugin_kind="transform", plugin_name="llm"
    )
    # First field is the discriminator
    assert ks["fields"][0]["name"] == "provider"
    assert ks["fields"][0]["kind"] == "enum"
    # All variant fields have visible_when predicates
    for f in ks["fields"][1:]:
        assert "visible_when" in f
        assert f["visible_when"]["field"] == "provider"
```

- [x] **Step 7: Run all knob-schema tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema*.py tests/unit/contracts/test_discriminated.py -v`
Expected: all passed

- [x] **Step 8: Commit**

```bash
git add src/elspeth/web/catalog/knob_schema.py src/elspeth/plugins/transforms/llm/transform.py tests/unit/web/composer/test_knob_schema_discriminated.py
git commit -m "feat(composer): support discriminated-union plugins via visible_when

KnobSchema.from_discriminated_model uses the DiscriminatedPlugin protocol
(contracts/discriminated.py) to read variant models without dict-parsing the
JSON schema. LLMTransform implements the protocol via _PROVIDERS. Variant
fields receive visible_when={field, equals} predicates. See spec §4.1.
"
```

---

### Task 6: VisibilityPredicate scope guard

**Files:**
- Modify: `src/elspeth/web/catalog/knob_schema.py` — add validator
- Test: `tests/unit/web/composer/test_knob_schema_visible_when.py`

- [x] **Step 1: Write the failing test (negative cases)**

```python
# tests/unit/web/composer/test_knob_schema_visible_when.py
import pytest
from elspeth.web.catalog.knob_schema import (
    KnobSchema,
    KnobSchemaLoweringError,
    validate_knob_schema,
)


def _ks_with_predicate(predicate):
    return {
        "fields": [
            {"name": "x", "label": "x", "kind": "enum", "enum": ["a", "b"],
             "required": True, "nullable": False},
            {"name": "y", "label": "y", "kind": "text",
             "required": False, "nullable": False, "visible_when": predicate},
        ]
    }


def test_well_formed_predicate_validates():
    ks = _ks_with_predicate({"field": "x", "equals": "a"})
    validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_extra_keys_rejected():
    ks = _ks_with_predicate({"field": "x", "equals": "a", "operator": "and"})
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "keys" in exc.value.constraint.lower()


def test_missing_keys_rejected():
    ks = _ks_with_predicate({"field": "x"})  # missing 'equals'
    with pytest.raises(KnobSchemaLoweringError):
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_forward_reference_rejected():
    # `y` references `x`, but x is declared after y.
    ks = {
        "fields": [
            {"name": "y", "label": "y", "kind": "text",
             "required": False, "nullable": False,
             "visible_when": {"field": "x", "equals": "a"}},
            {"name": "x", "label": "x", "kind": "enum", "enum": ["a"],
             "required": True, "nullable": False},
        ]
    }
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "forward" in exc.value.constraint.lower()


def test_unknown_field_reference_rejected():
    ks = _ks_with_predicate({"field": "nonexistent", "equals": "a"})
    with pytest.raises(KnobSchemaLoweringError):
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")


def test_nested_visibility_rejected():
    # y is visible_when x=a; z is visible_when y=...
    ks = {
        "fields": [
            {"name": "x", "label": "x", "kind": "enum", "enum": ["a"],
             "required": True, "nullable": False},
            {"name": "y", "label": "y", "kind": "text",
             "required": False, "nullable": False,
             "visible_when": {"field": "x", "equals": "a"}},
            {"name": "z", "label": "z", "kind": "text",
             "required": False, "nullable": False,
             "visible_when": {"field": "y", "equals": "anything"}},
        ]
    }
    with pytest.raises(KnobSchemaLoweringError) as exc:
        validate_knob_schema(ks, plugin_kind="t", plugin_name="p")
    assert "nest" in exc.value.constraint.lower()
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_visible_when.py -v`
Expected: FAIL — `ImportError: cannot import name 'validate_knob_schema'`

- [x] **Step 3: Implement `validate_knob_schema`**

Append to `src/elspeth/web/catalog/knob_schema.py`:

```python
_PREDICATE_KEYS: frozenset[str] = frozenset({"field", "equals"})


def validate_knob_schema(
    schema: KnobSchema,
    *,
    plugin_kind: str,
    plugin_name: str,
) -> None:
    """Validate KnobSchema invariants. Called by CatalogServiceImpl.__init__ after
    each lowering. Raises KnobSchemaLoweringError on:

    - visible_when keys outside exactly {field, equals}
    - visible_when.field is a forward reference (declared later)
    - visible_when.field is a non-existent KnobField
    - visible_when.field is itself visible_when-gated (nesting)
    """
    seen_so_far: list[str] = []
    visibility_gated: set[str] = set()
    for f in schema["fields"]:
        if "visible_when" in f:
            pred = f["visible_when"]
            keys = set(pred.keys())
            if keys != _PREDICATE_KEYS:
                raise KnobSchemaLoweringError(
                    plugin_kind=plugin_kind,
                    plugin_name=plugin_name,
                    field_path=f["name"],
                    constraint=f"visible_when has keys {sorted(keys)}; only 'field' and 'equals' permitted",
                    remediation="Remove extra keys; AND/OR predicates are out of scope (spec §4.1)",
                )
            target = pred["field"]
            if target not in seen_so_far:
                if any(g["name"] == target for g in schema["fields"]):
                    raise KnobSchemaLoweringError(
                        plugin_kind=plugin_kind,
                        plugin_name=plugin_name,
                        field_path=f["name"],
                        constraint=f"visible_when references forward field {target!r}",
                        remediation="Re-order fields so the discriminator is declared first",
                    )
                raise KnobSchemaLoweringError(
                    plugin_kind=plugin_kind,
                    plugin_name=plugin_name,
                    field_path=f["name"],
                    constraint=f"visible_when references unknown field {target!r}",
                    remediation="Check the field name; only earlier-declared KnobFields are valid targets",
                )
            if target in visibility_gated:
                raise KnobSchemaLoweringError(
                    plugin_kind=plugin_kind,
                    plugin_name=plugin_name,
                    field_path=f["name"],
                    constraint=(
                        f"visible_when targets {target!r} which is itself visible_when-gated "
                        f"(nested visibility chain)"
                    ),
                    remediation="Flatten the predicate chain; visibility nesting is out of scope",
                )
            visibility_gated.add(f["name"])
        seen_so_far.append(f["name"])
```

- [x] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/test_knob_schema_visible_when.py -v`
Expected: PASS — 6 passed

- [x] **Step 5: Commit**

```bash
git add src/elspeth/web/catalog/knob_schema.py tests/unit/web/composer/test_knob_schema_visible_when.py
git commit -m "feat(composer): mechanically enforce visible_when scope guard

Catalog load rejects predicates with extra/missing keys, forward references,
unknown targets, and nested visibility chains. Converts the §4.1 'redesign
signal' policy into a build gate per systems-thinking second-pass review.
"
```

---

### Task 6.5: Property tests + live catalog golden snapshots

**Files:**
- Test: `tests/unit/web/catalog/test_knob_schema_properties.py`
- Test: `tests/golden/web/catalog/knob_schema/*.json`

- [x] **Step 1: Add Hypothesis property coverage**

Use the repo's existing Hypothesis style (`rg -n "@given|hypothesis" tests/`) and add generated `BaseModel` cases for scalar, nullable, list, map, and complex-union fields. The property is:

- `lower_model_to_knob_schema(...)` always returns a valid `KnobSchema` for valid Pydantic models.
- `kind` is one of the closed Python/TypeScript kind set, including fallback kinds.
- nullable fields preserve `nullable=True`.
- valid rich shapes lower to `json-object`, `json-array`, or `json-value`; they do not raise merely because the UI lacks a bespoke editor.

Run: `.venv/bin/python -m pytest tests/unit/web/catalog/test_knob_schema_properties.py -v`
Expected: FAIL first because the property test does not exist; PASS after implementation.

- [x] **Step 2: Add golden snapshots for the live plugin catalog**

Create a snapshot test that constructs `CatalogServiceImpl(get_shared_plugin_manager())`, walks `get_sources()`, `get_transforms()`, and `get_sinks()`, and writes/compares stable JSON snapshots of each plugin's lowered `knob_schema`. Snapshot keys must use plugin `kind/name`, not class names, so plugin renames are visible.

Run: `.venv/bin/python -m pytest tests/unit/web/catalog/test_knob_schema_properties.py -v -k golden`
Expected: PASS with committed snapshots. Any future snapshot diff must be reviewed intentionally in PR.

- [x] **Step 3: Commit**

```bash
git add tests/unit/web/catalog/test_knob_schema_properties.py tests/golden/web/catalog/knob_schema/
git commit -m "test(catalog): add property and golden coverage for knob schema lowering

Hypothesis covers synthesized Pydantic models and the golden snapshots cover
the live plugin catalog through CatalogServiceImpl(get_shared_plugin_manager()).
This closes the spec §10 coverage gate before catalog integration."
```

---

## Phase B — Catalog integration (commit 2 of §9 step 1)

### Task 7: Extend PluginSchemaInfo with knob_schema field

**Files:**
- Modify: `src/elspeth/web/catalog/schemas.py:44`
- Test: `tests/unit/web/catalog/test_schemas.py` (existing test will need adjustment; verify)

- [x] **Step 1: Read current PluginSchemaInfo shape**

Run: `sed -n '40,60p' /home/john/elspeth/src/elspeth/web/catalog/schemas.py`
Note the current fields. The field `json_schema: dict[str, Any]` is present and stays.

- [x] **Step 2: Add `knob_schema` field**

Edit `src/elspeth/web/catalog/schemas.py` — locate the `PluginSchemaInfo` class and add a new field after `json_schema`:

```python
    knob_schema: dict[str, Any]
    """Lowered composer knob schema (spec §4). Computed once at catalog load
    inside CatalogServiceImpl.__init__ and cached. Always present; raw json_schema
    is preserved for external /catalog routes that need full JSON Schema."""
```

(Use `dict[str, Any]` rather than the TypedDict alias to keep `PluginSchemaInfo` simple — the TypedDict is enforced upstream by `validate_knob_schema`.)

- [x] **Step 3: Run existing catalog tests — they should still pass for the test fixtures, but new ones need adjustment**

Run: `.venv/bin/python -m pytest tests/unit/web/catalog/ -v`
Expected: any failures are tests that instantiate `PluginSchemaInfo` directly without `knob_schema`. Fix those by passing `knob_schema={"fields": []}`.

- [x] **Step 4: Commit**

```bash
git add src/elspeth/web/catalog/schemas.py tests/unit/web/catalog/
git commit -m "feat(catalog): add knob_schema field to PluginSchemaInfo

Carries the lowered composer knob shape alongside the existing raw json_schema.
External /catalog routes keep the full schema; composer reads only knob_schema.
See spec §3 catalog API extension.
"
```

---

### Task 8: CatalogServiceImpl eager lowering + cache

**Files:**
- Modify: `src/elspeth/web/catalog/service.py:36-83`
- Test: `tests/unit/web/catalog/test_eager_lowering.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/web/catalog/test_eager_lowering.py
import pytest
from pydantic import BaseModel

from elspeth.web.catalog.knob_schema import KnobSchemaLoweringError
from elspeth.web.catalog.service import CatalogServiceImpl


def test_catalog_lowering_runs_at_init_not_first_request(plugin_manager):
    """Verify lowering is eager — a bad plugin breaks __init__, not the
    first /api/catalog request."""
    svc = CatalogServiceImpl(plugin_manager)
    # If we got here without an exception, every registered plugin lowered cleanly.
    info = svc.get_schema("source", "csv")
    assert "knob_schema" in info.model_dump()
    assert info.knob_schema["fields"]  # non-empty list


def test_catalog_get_schema_reads_cache(plugin_manager, monkeypatch):
    """Two get_schema calls should not rerun lowering."""
    calls = 0

    def counting_lower(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"fields": []}

    monkeypatch.setattr("elspeth.web.catalog.service.lower_model_to_knob_schema", counting_lower)
    svc = CatalogServiceImpl(plugin_manager)
    calls_after_init = calls
    svc.get_schema("source", "csv")
    svc.get_schema("source", "csv")
    assert calls == calls_after_init


def test_catalog_init_raises_on_bad_plugin(plugin_manager_with_broken_plugin):
    """A plugin whose lowering fails halts startup."""
    with pytest.raises(KnobSchemaLoweringError):
        CatalogServiceImpl(plugin_manager_with_broken_plugin)


@pytest.fixture
def plugin_manager():
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

    return get_shared_plugin_manager()


@pytest.fixture
def plugin_manager_with_broken_plugin():
    class _BrokenOptions(BaseModel):
        missing_metadata: str

    class _BrokenSource:
        name = "broken_source"
        config_model = _BrokenOptions

        @classmethod
        def get_config_schema(cls):
            return cls.config_model.model_json_schema()

    class _FakePluginManager:
        def get_sources(self):
            return [_BrokenSource]

        def get_transforms(self):
            return []

        def get_sinks(self):
            return []

    return _FakePluginManager()
```

Keep these fixtures local to `test_eager_lowering.py` unless the existing catalog test suite already has equivalent fixtures with the same behavior. Do not rely on undefined shared fixtures.

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/catalog/test_eager_lowering.py -v`
Expected: FAIL — `KeyError` on the `knob_schema` field or attribute access fails.

- [x] **Step 3: Modify CatalogServiceImpl**

Edit `src/elspeth/web/catalog/service.py`. The current pattern is:

```python
def get_schema(self, plugin_type, name):
    plugin_cls = self._get_plugin_class(plugin_type, name)
    json_schema = self._catalog_schema(plugin_cls, plugin_type)
    return PluginSchemaInfo(name=name, plugin_type=plugin_type, json_schema=json_schema)
```

Replace with:

```python
class CatalogServiceImpl:
    def __init__(self, plugin_manager: PluginManager) -> None:
        self._pm = plugin_manager
        # Preserve the live constructor's class-list caches before building
        # schema records. Do not move the cache loop above these assignments.
        self._source_classes = plugin_manager.get_sources()
        self._transform_classes = plugin_manager.get_transforms()
        self._sink_classes = plugin_manager.get_sinks()

        # Pre-materialise every plugin's schema at construction time.
        # KnobSchemaLoweringError surfaces here — halts process startup.
        self._schema_cache: dict[tuple[str, str], PluginSchemaInfo] = {}
        for kind in ("source", "transform", "sink"):
            classes = (
                self._source_classes
                if kind == "source"
                else self._transform_classes
                if kind == "transform"
                else self._sink_classes
            )
            for cls in classes:
                name = cls.name
                self._schema_cache[(kind, name)] = self._build_schema_info(kind, name, cls)

    def get_schema(self, plugin_type: PluginKind, name: str) -> PluginSchemaInfo:
        try:
            return self._schema_cache[(plugin_type, name)]
        except KeyError:
            raise ValueError(f"Unknown plugin {plugin_type}/{name}")

    def _build_schema_info(
        self,
        plugin_type: PluginKind,
        name: str,
        plugin_cls: PluginClass,
    ) -> PluginSchemaInfo:
        from elspeth.web.catalog.knob_schema import (
            lower_discriminated_to_knob_schema,
            lower_model_to_knob_schema,
            validate_knob_schema,
        )

        json_schema = self._catalog_schema(plugin_cls, plugin_type)
        discriminated_variants = getattr(plugin_cls, "discriminated_variants", None)

        if discriminated_variants is not None and callable(discriminated_variants):
            knob_schema = lower_discriminated_to_knob_schema(
                plugin_cls, plugin_kind=plugin_type, plugin_name=name
            )
        else:
            options_cls = plugin_cls.config_model
            knob_schema = lower_model_to_knob_schema(
                options_cls, plugin_kind=plugin_type, plugin_name=name
            )
        validate_knob_schema(knob_schema, plugin_kind=plugin_type, plugin_name=name)

        description = (plugin_cls.__doc__ or "").strip() or get_plugin_description(plugin_cls)
        return PluginSchemaInfo(
            name=name,
            plugin_type=plugin_type,
            description=description,
            json_schema=json_schema,
            knob_schema=knob_schema,
        )

    # Use the live `plugin_cls.name`, `_source_classes`, `_transform_classes`,
    # and `_sink_classes` surfaces.
```

(The exact attribute names for source/transform/sink class lists need to match what's in `service.py:36-65` — read it before pasting.)

- [x] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/catalog/ tests/unit/web/composer/test_knob_schema*.py -v`
Expected: all passed

- [x] **Step 5: Smoke-test against a real catalog at startup**

Run: `.venv/bin/python -c "from elspeth.web.catalog.service import CatalogServiceImpl; from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager; svc = CatalogServiceImpl(get_shared_plugin_manager()); print(f'Loaded {len(svc._schema_cache)} plugins')"`
Expected: prints plugin count without raising. If `KnobSchemaLoweringError` raises, it should be for a true malformed schema or invariant violation — not merely a rich-but-valid live shape. The error message names plugin, field, constraint, remediation.

Also add this as a durable integration test, not only a shell smoke:

Run: `.venv/bin/python -m pytest tests/integration/web/catalog/test_startup_lowering.py -v`
Expected: constructs `CatalogServiceImpl(get_shared_plugin_manager())`, asserts `_schema_cache` covers every plugin from `get_sources()/get_transforms()/get_sinks()`, and asserts every cached `knob_schema` has a `fields` list.

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/catalog/service.py tests/unit/web/catalog/test_eager_lowering.py
git commit -m "feat(catalog): pre-materialise knob_schema at CatalogServiceImpl.__init__

Lowering runs once at startup, caching results by (plugin_kind, name).
KnobSchemaLoweringError propagates through DI bootstrap so a bad plugin
halts startup rather than returning 500 on first request. Closes
python-engineering C-2 from second-pass review.
"
```

---

## Phase C — Inspection-fact persistence (commit 3 of §9 step 2)

### Task 9: Persist SourceInspectionFacts on GuidedSession

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py:397, 422, 443, 478, 513`
- Modify: `src/elspeth/web/composer/source_inspection.py`
- Test: `tests/unit/web/composer/guided/test_state_machine.py`

- [x] **Step 1: Write the failing test**

```python
# Append to tests/unit/web/composer/guided/test_state_machine.py
import dataclasses

from elspeth.web.composer.guided.state_machine import GUIDED_SESSION_SCHEMA_VERSION, GuidedSession
from elspeth.web.composer.source_inspection import SourceInspectionFacts, facts_from_dict


def test_guided_session_round_trips_inspection_facts():
    facts = SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"filename": "input.csv"},
        byte_range_inspected=(0, 128),
        observed_headers=("name", "age"),
        inferred_types={"name": "str", "age": "int"},
        url_candidates=(),
        sample_row_count=10,
        warnings=(),
    )
    sess = dataclasses.replace(GuidedSession.initial(), step_1_inspection_facts=facts)
    d = sess.to_dict()
    restored = GuidedSession.from_dict(d)
    assert restored.step_1_inspection_facts == facts


def test_guided_session_inspection_facts_default_none():
    sess = GuidedSession.initial()
    assert sess.step_1_inspection_facts is None


def test_source_inspection_facts_from_dict_is_tier1_strict():
    d = {
        "source_kind": "csv",
        "redacted_identity": {"filename": "input.csv"},
        "byte_range_inspected": [0, 128],
        "sample_row_count": 10,
        "observed_headers": ["name", "age"],
        "inferred_types": {"name": "str", "age": "int"},
        "url_candidates": [],
        "warnings": [],
    }
    restored = facts_from_dict(d)
    assert restored.observed_headers == ("name", "age")


def test_guided_session_schema_version_bumped_for_inspection_facts():
    assert GUIDED_SESSION_SCHEMA_VERSION == 4
```

- [x] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v -k inspection_facts`
Expected: FAIL — `GuidedSession` does not yet persist `step_1_inspection_facts`.

- [x] **Step 3: Add strict facts deserializer**

Edit `source_inspection.py`:

- Add `facts_from_dict(d: Mapping[str, Any]) -> SourceInspectionFacts` next to `facts_to_dict`.
- Use direct subscript access for every field (`d["source_kind"]`, etc.); do not use `.get()` on this Tier-1 persisted path.
- Convert JSON lists back to tuples for `byte_range_inspected`, `observed_headers`, `url_candidates`, and `warnings`.
- Preserve `None` for `observed_headers` and `inferred_types`; do not fabricate empty structures.
- Wrap `KeyError`, `TypeError`, and `ValueError` in `InvariantError` with the malformed record.

- [x] **Step 4: Add the field to GuidedSession**

Edit `state_machine.py`:
- Add `step_1_inspection_facts: SourceInspectionFacts | None = None` after `step_3_proposal` (line ~397).
- Import `SourceInspectionFacts`, `facts_from_dict`, and `facts_to_dict` without creating a runtime cycle.
- Keep using the live constructor helper `GuidedSession.initial()`; do not introduce a new `GuidedSession.new` helper.
- Update `to_dict` to include `step_1_inspection_facts: facts_to_dict(self.step_1_inspection_facts) if self.step_1_inspection_facts else None`.
- Update `from_dict` to read `facts_raw = d["step_1_inspection_facts"]` via direct subscript and restore through `facts_from_dict(facts_raw)` when non-None.
- Bump `GUIDED_SESSION_SCHEMA_VERSION` from 3 to 4 and update the schema-version tests.
- Add `__post_init__` handling or an equivalent freeze guard so `step_1_inspection_facts` cannot carry mutable nested mappings after session construction. `SourceInspectionFacts` already freezes its mappings, but the new `GuidedSession` field must be covered by the existing freeze-guard discipline and tests.
- Ensure the field is included in any `_replace` lifecycle that re-enters Step 1 (e.g., cleared on re-enter to a fresh state).
- Add an operational migration note for staging/prod: existing persisted guided sessions using schema version 3 are intentionally incompatible; the operator deletes the sessions DB before deploy. Do not silently accept both versions.

(See the current shape at `state_machine.py:397, 422, 443, 478, 513` for matching the existing patterns.)

- [x] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v -k "inspection_facts or schema_version or freeze"`
Expected: PASS

- [x] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/source_inspection.py src/elspeth/web/composer/guided/state_machine.py tests/unit/web/composer/guided/test_state_machine.py
git commit -m "feat(composer): persist SourceInspectionFacts on GuidedSession

step_1_inspection_facts is written when the initial INSPECT_AND_CONFIRM turn
is emitted and read by build_step_1_schema_form_turn (Task 11). Survives
session reload so the rebuild path produces an identical prefilled form.
"
```

---

## Phase C2 — Metadata audit before renderer cutover (commit 3b of §9 step 5a, reordered)

Task 16 runs here, before Phase D, so the new renderer never ships against missing titles/descriptions. Task 17 can still wire the permanent CI check after the audit is clean.

## Phase D — Atomic wire-contract migration (commit 4 of §9 step 3 — ATOMIC)

> **Important:** Tasks 10–14 must ship as ONE commit. Each task lands in the working tree, with local tests green, but only the final task 14's commit step is executed. Per spec §9 step 3 and Phase 9 convention 14: wire-contract changes ship in one commit.
>
> **Worktree discipline:** execute Phase D in an isolated worktree or make WIP commits per task and squash/revert at the end. This phase intentionally spans backend + frontend wire changes; do not accumulate the full multi-file change in an unprotected dirty tree.
>
> **Precondition:** Task 16 metadata audit/fill must already be complete, or Phase D must begin with a metadata coverage assertion that every loaded plugin config field has a title. The new renderer must not go live while labels can regress to raw Python attribute names.

### Task 10: Replace SchemaFormPayload with tagged union in protocol.py

**Files:**
- Modify: `src/elspeth/web/composer/guided/protocol.py:53-59, 236-286`

- [x] **Step 1: Locate current SchemaFormPayload**

Run: `sed -n '50,80p' /home/john/elspeth/src/elspeth/web/composer/guided/protocol.py`

- [x] **Step 2: Replace with re-export from knob_schema**

Edit `protocol.py`:

```python
# Replace the existing SchemaFormPayload TypedDict with:
from elspeth.web.catalog.knob_schema import SchemaFormPayload  # noqa: F401

# Remove the old SchemaFormPayload TypedDict definition entirely; do NOT
# leave a deprecated alias per CLAUDE.md no-legacy policy.
```

- [x] **Step 3: Update protocol validation constants**

In the same edit, update `_REQUIRED_KEYS` and `_NESTED_SHAPES`:

- `TurnType.SCHEMA_FORM` required keys become `{"mode", "knobs", "prefilled"}` plus mode-specific validation for `plugin_options.plugin` and `recipe_decision.recipe_context`.
- `knobs` must be a mapping with `fields`.
- If `TurnType.RECIPE_OFFER` is still emitted before Task 15, keep its old required keys until that task; Task 15 performs the recipe-offer payload switch atomically.

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v`
Expected: failing tests first for the old shape, then PASS after updating the constants and fixtures.

- [x] **Step 4: Type-check**

Run: `.venv/bin/python -m mypy src/elspeth/web/composer/`
Expected: errors in `emitters.py` (consumers of old shape). These are fixed in Task 11.

---

### Task 11: Rewrite three emitter functions

**Files:**
- Modify: `src/elspeth/web/composer/guided/emitters.py:115-143, 181-209, 326-343`
- Test: `tests/unit/web/composer/guided/test_emitters.py` (extend existing)

- [x] **Step 1: Update `build_step_1_schema_form_turn`**

```python
def build_step_1_schema_form_turn(
    plugin: str,
    catalog: CatalogServiceProtocol,
    *,
    inspection_facts: SourceInspectionFacts | None = None,
) -> Turn:
    """Build a schema_form Turn for the chosen source plugin.

    `inspection_facts` (when present) is merged into `prefilled` using only
    facts carried by the current SourceInspectionFacts model. When None,
    prefill falls back to the existing constant {"schema": {"mode": "observed"}}.
    """
    schema_info = catalog.get_schema("source", plugin)
    prefilled: dict[str, Any] = {"schema": {"mode": "observed"}}
    if inspection_facts is not None:
        _merge_inspection_into_prefill(prefilled, inspection_facts)
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": schema_info.knob_schema,
        "prefilled": prefilled,
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_1_SOURCE),
        payload=payload,
    )


def _merge_inspection_into_prefill(
    prefilled: dict[str, Any],
    facts: SourceInspectionFacts,
) -> None:
    """Conservative prefill from inspection facts. Only sets keys with
    confident values; never invents (per Tier 3 trust-tier discipline)."""
    if facts.observed_headers and facts.inferred_types:
        fields: list[str] = []
        for header in facts.observed_headers:
            inferred = facts.inferred_types[header]
            field_type = "any" if inferred == "null" else inferred
            fields.append(f"{header}: {field_type}")
        prefilled["schema"] = {"mode": "flexible", "fields": fields}
    elif facts.observed_headers:
        prefilled["schema"] = {"mode": "observed"}
    # Delimiter and encoding are deliberately not prefilled here: the live
    # SourceInspectionFacts model does not carry those fields yet.
```

- [x] **Step 2: Update `build_step_2_schema_form_turn` and `build_step_3_schema_form_turn`**

```python
def build_step_2_schema_form_turn(
    plugin: str,
    catalog: CatalogServiceProtocol,
) -> Turn:
    schema_info = catalog.get_schema("sink", plugin)
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": schema_info.knob_schema,
        "prefilled": {"schema": {"mode": "observed"}},
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_2_SINK),
        payload=payload,
    )


def build_step_3_schema_form_turn(
    *,
    plugin: str,
    options: Mapping[str, Any],
    catalog: CatalogServiceProtocol,
) -> Turn:
    schema_info = catalog.get_schema("transform", plugin)
    payload: SchemaFormPayload = {
        "mode": "plugin_options",
        "plugin": plugin,
        "knobs": schema_info.knob_schema,
        "prefilled": dict(options),
    }
    return Turn(
        type=TurnType.SCHEMA_FORM.value,
        step_index=_step_index(GuidedStep.STEP_3_TRANSFORMS),
        payload=payload,
    )
```

- [x] **Step 3: Update tests**

```python
# tests/unit/web/composer/guided/test_emitters.py — find existing schema_form turn tests
# Replace assertions on `payload["schema_block"]` with `payload["knobs"]`
# and add `assert payload["mode"] == "plugin_options"`
```

- [x] **Step 4: Run emitter tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_emitters.py -v`
Expected: PASS (after the assertion swap)

---

### Task 12: Thread inspection facts through dispatch

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py:2180, 5189` (dispatch/rebuild sites that emit Step 1 after inspection)
- Modify: `src/elspeth/web/composer/guided/emitters.py` (if `build_initial_step_1_turn` needs to route through inspection facts)

- [x] **Step 1: Identify call sites**

Run:

```bash
rg -n "build_step_1_schema_form_turn|build_initial_step_1_turn|build_step_1_inspect_and_confirm_turn_from_intent" src/elspeth/web/sessions/routes.py src/elspeth/web/composer/guided/emitters.py
```

Reality check from the review: there is only one direct `build_step_1_schema_form_turn` call in `routes.py`. The GET rebuild path does not call it; when `step_1_source_intent` is set it currently rebuilds `INSPECT_AND_CONFIRM` through `build_step_1_inspect_and_confirm_turn_from_intent`, otherwise it calls `build_initial_step_1_turn(...)`. Do not implement the old "two direct call sites" premise.

- [x] **Step 2: Update the direct schema-form dispatch**

Locate each site and change:

```python
# Before
build_step_1_schema_form_turn(plugin, catalog)
# After
build_step_1_schema_form_turn(plugin, catalog, inspection_facts=guided.step_1_inspection_facts)
```

- [x] **Step 3: Write the population path**

Find where INSPECT_AND_CONFIRM is emitted (likely calls `_build_inspect_and_confirm_turn`); after that emission, persist the facts on `guided`:

```python
# Wherever INSPECT_AND_CONFIRM is emitted, after computing inspection facts:
guided = _replace(guided, step_1_inspection_facts=inspection_facts)
```

- [x] **Step 4: Fix the rebuild path**

The rebuild path must deliver the same prefilled schema form after refresh. If the session is waiting on the schema form after source selection, the chosen plugin and inspection facts must be persisted in `GuidedSession` and the rebuild branch must call `build_step_1_schema_form_turn(..., inspection_facts=guided.step_1_inspection_facts)`. If that requires a new staging field for the selected source plugin, add it explicitly and include it in the schema-version bump from Task 9. Do not use `build_initial_step_1_turn(...)` for this intra-step rebuild once a plugin has been selected.

- [x] **Step 5: Integration test for real prefill**

```python
# tests/integration/web/composer/test_guided_step1_prefill.py
def test_step1_prefilled_from_inspection_facts(client):
    sess = client.post("/api/sessions", json={"title": "step1-prefill"}).json()["id"]
    # Trigger INSPECT_AND_CONFIRM via blob upload
    seed_csv_blob(client, sess)
    client.post(f"/api/sessions/{sess}/guided/respond", json=single_select_response("csv"))
    # Now the SCHEMA_FORM turn should have inspection-derived prefill
    state = client.get(f"/api/sessions/{sess}/guided").json()
    assert state["turn"]["type"] == "schema_form"
    schema_prefill = state["turn"]["payload"]["prefilled"]["schema"]
    assert schema_prefill["mode"] == "flexible"
    assert "name: str" in schema_prefill["fields"]
    assert "age: int" in schema_prefill["fields"]


def test_step1_prefill_survives_guided_get_rebuild(client, session_with_step1_inspection_facts):
    state = client.get(f"/api/sessions/{session_with_step1_inspection_facts}/guided").json()
    schema_prefill = state["turn"]["payload"]["prefilled"]["schema"]
    assert schema_prefill["fields"] == ["name: str", "age: int"]


def single_select_response(plugin: str) -> dict[str, object]:
    return {
        "chosen": [plugin],
        "edited_values": None,
        "custom_inputs": None,
        "accepted_step_index": None,
        "edit_step_index": None,
        "control_signal": None,
    }


@pytest.fixture
def session_with_step1_inspection_facts(client):
    sess = client.post("/api/sessions", json={"title": "step1-prefill-rebuild"}).json()["id"]
    seed_csv_blob(client, sess)
    resp = client.post(
        f"/api/sessions/{sess}/guided/respond",
        json=single_select_response("csv"),
    )
    assert resp.status_code == 200, resp.json()
    state = client.get(f"/api/sessions/{sess}/guided").json()
    assert state["turn"]["type"] == "schema_form"
    assert state["turn"]["payload"]["prefilled"]["schema"]["mode"] == "flexible"
    return sess


def seed_csv_blob(client, session_id: str) -> str:
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": "name,age\nAda,37\n", "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]
```

The tests must fail if `_merge_inspection_into_prefill` only returns the old constant `{"schema": {"mode": "observed"}}`.
Define `single_select_response(...)`, `seed_csv_blob(...)`, and `session_with_step1_inspection_facts` in this test module using the repo's existing `/api/sessions/{id}/blobs/inline` route; do not assume any of those helpers already exist. Add `pytest` locally for the fixture.

---

### Task 13: Backend hidden-field rejection

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` — add a helper, apply at all three SCHEMA_FORM dispatchers
- Test: `tests/integration/web/composer/guided/test_hidden_field_rejection.py`

- [x] **Step 1: Write the failing test**

```python
# tests/integration/web/composer/guided/test_hidden_field_rejection.py
from fastapi.testclient import TestClient


def test_hidden_field_submission_returns_400(client, session_at_step_3_llm):
    """Operator submits a value for a hidden variant field; backend rejects."""
    sess_id = session_at_step_3_llm
    # The session is at step 3 with provider=azure selected; submit a value
    # for `model` (which is openrouter-only) → should be 400.
    resp = client.post(
        f"/api/sessions/{sess_id}/guided/respond",
        json={
            "chosen": None,
            "edited_values": {
                "plugin": "llm",
                "options": {"provider": "azure", "deployment": "gpt-4", "model": "claude-3"},
                "observed_columns": [],
                "sample_rows": [],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": 0,
            "control_signal": None,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["code"] == "hidden_field_submitted"
    assert body["detail"]["field"] == "model"
    audit_rows = audit_events_for_session(client, sess_id)
    assert any(row.tool_name == "guided_hidden_field_rejected" for row in audit_rows)


@pytest.fixture
def session_at_step_3_llm(client):
    sess = _create_session(client)
    _drive_guided_flow_to_step_3_transform_schema(client, sess, transform_plugin="llm")
    state = client.get(f"/api/sessions/{sess}/guided").json()
    assert state["guided_session"]["step"] == "step_3_transforms"
    assert state["turn"]["type"] == "schema_form"
    assert state["turn"]["payload"]["plugin"] == "llm"
    return sess


def audit_events_for_session(client, session_id: str):
    service = client.app.state.session_service
    msgs = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    rows = []
    for msg in msgs:
        if msg.role not in ("tool", "audit") or not msg.tool_calls:
            continue
        for tool_call in msg.tool_calls:
            invocation = tool_call.get("invocation", {})
            if invocation.get("tool_name"):
                rows.append(SimpleNamespace(**invocation))
    return rows


def _create_session(client) -> str:
    resp = client.post("/api/sessions", json={"title": "hidden-field-rejection"})
    assert resp.status_code == 201, resp.json()
    return resp.json()["id"]


def _drive_guided_flow_to_step_3_transform_schema(client, session_id: str, *, transform_plugin: str) -> None:
    with patch(
        "elspeth.web.composer.guided.chain_solver._litellm_acompletion",
        new_callable=AsyncMock,
        return_value=_fake_llm_response_for_transform(transform_plugin),
    ):
        blob_id, storage_path = _seed_blob(client, session_id)
        output_path = _outputs_path(client, "out.jsonl")

        _get_guided(client, session_id)
        _respond(client, session_id, chosen=["csv"])
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "csv",
                "options": {"path": storage_path, "schema": {"mode": "observed"}, "blob_id": blob_id},
                "observed_columns": ["text", "note"],
                "sample_rows": [{"text": "Hello world", "note": "greeting"}],
            },
        )
        _respond(client, session_id, chosen=["json"])
        _respond(
            client,
            session_id,
            edited_values={
                "plugin": "json",
                "options": {
                    "path": output_path,
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "observed_columns": [],
                "sample_rows": [],
            },
        )
        body = _respond(client, session_id, chosen=["text"], custom_inputs=[])
        assert body["next_turn"]["type"] == "propose_chain"

        edit_resp = client.post(
            f"/api/sessions/{session_id}/guided/respond",
            json={
                "chosen": None,
                "edited_values": None,
                "custom_inputs": None,
                "accepted_step_index": None,
                "edit_step_index": 0,
                "control_signal": None,
            },
        )
        assert edit_resp.status_code == 200, edit_resp.json()


def _fake_llm_response_for_transform(plugin: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="emit_turn",
                                arguments=json.dumps(
                                    {
                                        "turn_type": "propose_chain",
                                        "payload": {
                                            "steps": [
                                                {
                                                    "plugin": plugin,
                                                    "options": {"provider": "azure", "deployment": "gpt-4"},
                                                    "rationale": "exercise variant visibility",
                                                }
                                            ],
                                            "why": "single transform proposal for hidden-field rejection test",
                                            "blockers": [],
                                        },
                                    }
                                ),
                            )
                        )
                    ]
                )
            )
        ]
    )


def _get_guided(client, session_id: str) -> dict:
    resp = client.get(f"/api/sessions/{session_id}/guided")
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _respond(client, session_id: str, **kwargs) -> dict:
    resp = client.post(f"/api/sessions/{session_id}/guided/respond", json=kwargs)
    assert resp.status_code == 200, resp.json()
    return resp.json()


def _seed_blob(client, session_id: str) -> tuple[str, str]:
    resp = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "data.csv", "content": "text,note\nHello,greeting\n", "mime_type": "text/csv"},
    )
    assert resp.status_code == 201, resp.json()
    blob_id = resp.json()["id"]
    record = asyncio.run(client.app.state.blob_service.get_blob(UUID(blob_id)))
    return blob_id, record.storage_path


def _outputs_path(client, filename: str) -> str:
    outputs_dir = client.app.state.settings.data_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return str(outputs_dir / filename)
```

Define `session_at_step_3_llm`, `audit_events_for_session`, and the local route-driving helpers in this integration test module; do not assume they exist. The fixture must construct a real guided session at Step 3 with an LLM schema whose provider discriminator makes `model` hidden for the selected Azure branch. Add the needed local imports (`asyncio`, `json`, `SimpleNamespace`, `AsyncMock`, `patch`, `UUID`, `pytest`) in the same test file.

- [x] **Step 2: Implement `_reject_hidden_field_submissions`**

Add to `routes.py`:

```python
def _reject_hidden_field_submissions(
    knobs: KnobSchema,
    submitted_options: dict[str, Any],
    *,
    recorder: ComposerToolRecorder,
    composition_version: int,
    actor: str,
    session_id: str,
    plugin_kind: str,
    plugin_name: str,
) -> None:
    """Raise HTTPException(400, code=hidden_field_submitted) if any submitted
    option corresponds to a KnobField whose visible_when predicate doesn't
    match current form state."""
    fields_by_name = {f["name"]: f for f in knobs["fields"]}
    for opt_name, _ in submitted_options.items():
        try:
            field = fields_by_name[opt_name]
        except KeyError:
            continue
        if "visible_when" not in field:
            continue
        pred = field["visible_when"]
        # Check predicate against submitted state — the discriminator value
        # is itself in submitted_options.
        try:
            target_val = submitted_options[pred["field"]]
        except KeyError as exc:
            emit_hidden_field_rejected(
                recorder,
                session_id=session_id,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                field=opt_name,
                predicate=pred,
                actual_state={},
                composition_version=composition_version,
                actor=actor,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "hidden_field_submitted",
                    "field": opt_name,
                    "predicate": pred,
                    "message": f"Visibility discriminator {pred['field']!r} is missing from submitted options.",
                },
            ) from exc
        if target_val != pred["equals"]:
            emit_hidden_field_rejected(
                recorder,
                session_id=session_id,
                plugin_kind=plugin_kind,
                plugin_name=plugin_name,
                field=opt_name,
                predicate=pred,
                actual_state={pred["field"]: target_val},
                composition_version=composition_version,
                actor=actor,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "hidden_field_submitted",
                    "field": opt_name,
                    "predicate": pred,
                    "actual_state": {pred["field"]: target_val},
                    "message": (
                        f"Field {opt_name!r} is hidden under current form state "
                        f"({pred['field']}={target_val!r}, predicate expects "
                        f"{pred['field']}={pred['equals']!r}). Hidden fields "
                        "must not appear in edited_values.options."
                    ),
                },
            )
```

Audit policy: hidden-field rejection is security-relevant and must emit an audit event before the 400 response. Add `emit_hidden_field_rejected()` to `src/elspeth/web/composer/guided/audit.py` using the existing `_build_invocation(...)` helper and `tool_name="guided_hidden_field_rejected"`; do not invent a new audit primitive or log this decision. The canonical payload must include session id, plugin kind/name, rejected field, predicate, and submitted discriminator value. Thread `recorder`, `composition_version=state.version`, and `actor=user_id` from `_dispatch_guided_respond` into `_reject_hidden_field_submissions(...)` at every call site.

Also validate `blob-ref` values as UUID strings at the same Tier-3 boundary before dispatching to plugin validators. Invalid UUIDs should return a structured 400 instead of flowing as arbitrary text.

- [x] **Step 3: Call the helper at each SCHEMA_FORM dispatcher site**

At each `SCHEMA_FORM` branch in `_dispatch_guided_respond` (Step 1, Step 2, Step 3), after parsing `options_raw` and before applying it:

```python
schema_info = catalog.get_schema(plugin_kind, plugin_name)
_reject_hidden_field_submissions(
    schema_info.knob_schema,
    options_raw,
    recorder=recorder,
    composition_version=state.version,
    actor=user_id,
    session_id=session_id,
    plugin_kind=plugin_kind,
    plugin_name=plugin_name,
)
```

- [x] **Step 4: Run the test**

Run: `.venv/bin/python -m pytest tests/integration/web/composer/guided/test_hidden_field_rejection.py -v`
Expected: PASS

---

### Task 14: Rewrite SchemaFormTurn.tsx

**Files:**
- Modify: `src/elspeth/web/frontend/src/types/guided.ts` (new types)
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx` (full rewrite)
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx`

- [x] **Step 1: Update TypeScript types**

```typescript
// src/elspeth/web/frontend/src/types/guided.ts (replace SchemaFormPayload block)

export type FieldKind =
  | "text"
  | "number-int"
  | "number-float"
  | "checkbox"
  | "enum"
  | "string-list"
  | "blob-ref"
  | "json-object"
  | "json-array"
  | "json-value";

export type FieldTier = "essential" | "common" | "advanced";

export interface VisibilityPredicate {
  field: string;
  equals: unknown;
}

export interface KnobField {
  name: string;
  label: string;
  description?: string;
  kind: FieldKind;
  tier?: FieldTier;
  required: boolean;
  default?: unknown;
  nullable: boolean;
  enum?: string[];
  item_kind?: "text" | "number-int" | "number-float";
  visible_when?: VisibilityPredicate;
}

export interface KnobSchema {
  fields: KnobField[];
}

export interface RecipeContext {
  recipe_name: string;
  description: string;
  alternatives: string[];
}

export type SchemaFormPayload =
  | {
      mode: "plugin_options";
      plugin: string;
      knobs: KnobSchema;
      prefilled: Record<string, unknown>;
    }
  | {
      mode: "recipe_decision";
      knobs: KnobSchema;
      prefilled: Record<string, unknown>;
      recipe_context: RecipeContext;
    };
```

- [x] **Step 2: Rewrite SchemaFormTurn.tsx**

Replace the file with a kind-dispatched renderer. Structure:

```typescript
// src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx
import { useId, useState } from "react";
import type { GuidedRespondRequest, SchemaFormPayload, KnobField } from "@/types/guided";
import { RecipeContextHeader } from "./RecipeContextHeader";

interface Props {
  payload: SchemaFormPayload;
  onSubmit: (body: GuidedRespondRequest) => void;
  disabled?: boolean;
}

export function SchemaFormTurn({ payload, onSubmit, disabled = false }: Props) {
  const reactId = useId();
  const [values, setValues] = useState<Record<string, unknown>>(() =>
    initialValues(payload.knobs.fields, payload.prefilled),
  );

  function isVisible(field: KnobField): boolean {
    if (!field.visible_when) return true;
    return values[field.visible_when.field] === field.visible_when.equals;
  }

  function visibleFields(): KnobField[] {
    return payload.knobs.fields.filter(isVisible);
  }

  function onChange(name: string, value: unknown) {
    setValues((prev) => {
      const next = { ...prev, [name]: value };
      // If a discriminator changed, drop variant-specific user values
      // for variants now hidden (per spec §4.1 evaluation semantics).
      for (const f of payload.knobs.fields) {
        if (f.visible_when && f.visible_when.field === name) {
          if (f.visible_when.equals !== value) {
            delete next[f.name];
          }
        }
      }
      return next;
    });
  }

  function canSubmit(): boolean {
    for (const f of visibleFields()) {
      if (!f.required) continue;
      const v = values[f.name];
      if (f.kind === "checkbox") continue;  // required bool always satisfied
      if (v === undefined || v === null || v === "") return false;
    }
    return true;
  }

  function handleContinue() {
    if (!canSubmit()) return;
    const submitted: Record<string, unknown> = {};
    for (const f of visibleFields()) {
      submitted[f.name] = values[f.name] ?? f.default ?? null;
    }
    if (payload.mode === "plugin_options") {
      onSubmit({
        chosen: null,
        edited_values: {
          plugin: payload.plugin,
          options: submitted,
          observed_columns: [],
          sample_rows: [],
        },
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });
    } else {
      onSubmit({
        chosen: ["accept"],
        edited_values: {
          recipe_name: payload.recipe_context.recipe_name,
          slots: submitted,
        },
        custom_inputs: null,
        accepted_step_index: null,
        edit_step_index: null,
        control_signal: null,
      });
    }
  }

  return (
    <div className="guided-turn guided-schema-form">
      {payload.mode === "recipe_decision" && (
        <RecipeContextHeader context={payload.recipe_context} />
      )}
      {visibleFields().map((f) => (
        <KnobFieldRenderer
          key={f.name}
          field={f}
          value={values[f.name]}
          onChange={(v) => onChange(f.name, v)}
          idPrefix={reactId}
          disabled={disabled}
        />
      ))}
      <button
        type="button"
        onClick={handleContinue}
        disabled={disabled || !canSubmit()}
      >
        Continue
      </button>
    </div>
  );
}

function initialValues(fields: KnobField[], prefilled: Record<string, unknown>): Record<string, unknown> {
  const v: Record<string, unknown> = {};
  for (const f of fields) {
    if (f.name in prefilled) {
      v[f.name] = prefilled[f.name];
    } else if (f.default !== undefined) {
      v[f.name] = f.default;
    } else {
      v[f.name] = emptyForKind(f.kind);
    }
  }
  return v;
}

function emptyForKind(kind: KnobField["kind"]): unknown {
  if (kind === "checkbox") return false;
  if (kind === "string-list") return [];
  if (kind === "json-object") return {};
  if (kind === "json-array") return [];
  if (kind === "json-value") return null;
  if (kind === "number-int" || kind === "number-float") return "";
  return "";
}

function KnobFieldRenderer({
  field, value, onChange, idPrefix, disabled,
}: {
  field: KnobField;
  value: unknown;
  onChange: (v: unknown) => void;
  idPrefix: string;
  disabled: boolean;
}) {
  const id = `${idPrefix}-${field.name}`;
  switch (field.kind) {
    case "text":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <input id={id} type="text" value={String(value ?? "")}
            placeholder={field.default !== undefined ? String(field.default) : undefined}
            onChange={(e) => onChange(e.target.value)} disabled={disabled} />
          {field.description && <p>{field.description}</p>}
          {field.nullable && value !== null && (
            <button type="button" onClick={() => onChange(null)}>Clear</button>
          )}
        </div>
      );
    case "number-int":
    case "number-float":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <input id={id} type="number"
            step={field.kind === "number-int" ? "1" : "any"}
            value={value === null ? "" : String(value ?? "")}
            onChange={(e) => {
              const raw = e.target.value;
              if (raw === "") { onChange(null); return; }
              const n = field.kind === "number-int" ? parseInt(raw, 10) : parseFloat(raw);
              onChange(Number.isNaN(n) ? null : n);
            }} disabled={disabled} />
        </div>
      );
    case "checkbox":
      return (
        <div>
          <input id={id} type="checkbox" checked={Boolean(value)}
            onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
          <label htmlFor={id}>{field.label}</label>
        </div>
      );
    case "enum":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <select id={id} value={String(value ?? "")}
            onChange={(e) => onChange(e.target.value)} disabled={disabled}>
            {(field.enum ?? []).map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
      );
    case "string-list":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <textarea id={id} value={Array.isArray(value) ? (value as string[]).join("\n") : ""}
            onChange={(e) => onChange(e.target.value.split("\n").filter(Boolean))}
            disabled={disabled} />
          <p>One value per line.</p>
        </div>
      );
    case "blob-ref":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <input id={id} type="text" value={String(value ?? "")}
            placeholder="blob UUID"
            onChange={(e) => onChange(e.target.value)} disabled={disabled} />
        </div>
      );
    case "json-object":
    case "json-array":
    case "json-value":
      return (
        <div>
          <label htmlFor={id}>{field.label}</label>
          <textarea id={id}
            aria-describedby={field.description ? `${id}-description` : undefined}
            value={JSON.stringify(value ?? emptyForKind(field.kind), null, 2)}
            onChange={(e) => {
              try { onChange(JSON.parse(e.target.value)); }
              catch { onChange(e.target.value); }
            }}
            disabled={disabled} />
          {field.description && <p id={`${id}-description`}>{field.description}</p>}
        </div>
      );
  }
  const _exhaustive: never = field.kind;
  return _exhaustive;
}
```

- [x] **Step 3: Add RecipeContextHeader.tsx**

```typescript
// src/elspeth/web/frontend/src/components/chat/guided/RecipeContextHeader.tsx
import type { RecipeContext } from "@/types/guided";

export function RecipeContextHeader({ context }: { context: RecipeContext }) {
  return (
    <div className="recipe-context-header">
      <h3>{context.recipe_name}</h3>
      <p>{context.description}</p>
      {context.alternatives.length > 0 && (
        <div>Alternatives: {context.alternatives.join(", ")}</div>
      )}
    </div>
  );
}
```

- [x] **Step 4: Rewrite SchemaFormTurn.test.tsx**

Replace the existing tests with kind-dispatched cases:

```typescript
// Representative tests; expand for full coverage
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { SchemaFormTurn } from "./SchemaFormTurn";

describe("SchemaFormTurn", () => {
  it("renders enum as select", () => {
    render(<SchemaFormTurn
      payload={{
        mode: "plugin_options", plugin: "x",
        knobs: { fields: [
          { name: "p", label: "P", kind: "enum", enum: ["a", "b"], required: true, nullable: false },
        ]},
        prefilled: {},
      }}
      onSubmit={vi.fn()}
    />);
    expect(screen.getByRole("combobox", { name: /P/i })).toBeInTheDocument();
  });

  it("hides fields whose visible_when does not match", () => {
    render(<SchemaFormTurn
      payload={{
        mode: "plugin_options", plugin: "x",
        knobs: { fields: [
          { name: "k", label: "K", kind: "enum", enum: ["a", "b"], required: true, nullable: false },
          { name: "v", label: "V", kind: "text", required: false, nullable: false,
            visible_when: { field: "k", equals: "a" } },
        ]},
        prefilled: { k: "b" },
      }}
      onSubmit={vi.fn()}
    />);
    expect(screen.queryByLabelText(/V/i)).not.toBeInTheDocument();
  });

  it("drops variant state on discriminator change", () => {
    const onSubmit = vi.fn();
    render(<SchemaFormTurn
      payload={{
        mode: "plugin_options", plugin: "x",
        knobs: { fields: [
          { name: "k", label: "K", kind: "enum", enum: ["a", "b"], required: true, nullable: false },
          { name: "va", label: "Va", kind: "text", required: false, nullable: false,
            visible_when: { field: "k", equals: "a" } },
          { name: "vb", label: "Vb", kind: "text", required: false, nullable: false,
            visible_when: { field: "k", equals: "b" } },
        ]},
        prefilled: { k: "a" },
      }}
      onSubmit={onSubmit}
    />);
    fireEvent.change(screen.getByLabelText(/Va/i), { target: { value: "typed" } });
    fireEvent.change(screen.getByRole("combobox", { name: /K/i }), { target: { value: "b" } });
    // Va is hidden now; va value must not be submitted
    fireEvent.change(screen.getByLabelText(/Vb/i), { target: { value: "bee" } });
    fireEvent.click(screen.getByRole("button", { name: /Continue/i }));
    expect(onSubmit).toHaveBeenCalledWith(expect.objectContaining({
      edited_values: expect.objectContaining({
        options: { k: "b", vb: "bee" },  // no `va`
      }),
    }));
  });
});
```

Required frontend coverage:

- Add parametrised tests for every `FieldKind`, including `number-int`, `number-float`, `json-object`, `json-array`, and `json-value`.
- Add a required-text clear test: type into a required `text` field, clear it, assert the Continue button is disabled and `onSubmit` is not called. This covers spec §7 case 4.
- Add the `never` exhaustiveness check shown above; no default renderer arm.
- Add a11y assertions for `aria-describedby`, clear-button accessible name, heading hierarchy when `RecipeContextHeader` is present, and keyboard editing of `string-list`.
- Add a `blob-ref` test that submits an invalid UUID and verify the backend boundary rejects it; the frontend may show text, but the server owns validation.

- [x] **Step 5: Run frontend tests**

Run: `cd src/elspeth/web/frontend && npm test -- --run SchemaFormTurn`
Expected: PASS

- [x] **Step 6: Run full backend + frontend test suite**

Run backend: `.venv/bin/python -m pytest tests/unit/web/ tests/integration/web/composer/`
Run frontend: `cd src/elspeth/web/frontend && npm test -- --run`
Expected: all passed.

- [x] **Step 7: Run policy gates before the atomic commit**

Run:

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m mypy src
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check --root src/elspeth --allowlist config/cicd/enforce_freeze_guards
```

Expected: all green. If `enforce_tier_model.py` or `enforce_freeze_guards.py` uses a different CLI shape in the live repo, inspect the script help and run the repo's real check form; do not skip the gate.

- [x] **Step 8: Commit the atomic wire-contract change**

```bash
git add \
  src/elspeth/web/composer/guided/protocol.py \
  src/elspeth/web/composer/guided/emitters.py \
  src/elspeth/web/composer/guided/state_machine.py \
  src/elspeth/web/sessions/routes.py \
  src/elspeth/web/frontend/src/types/guided.ts \
  src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx \
  src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.test.tsx \
  src/elspeth/web/frontend/src/components/chat/guided/RecipeContextHeader.tsx \
  tests/
git commit -m "feat(composer): atomic wire-contract migration to KnobSchema

The one-knob wire contract goes live: SCHEMA_FORM payloads carry tagged-union
{mode, knobs, prefilled, [recipe_context]} instead of raw json_schema. Frontend
SchemaFormTurn rewritten to dispatch on KnobField.kind with visible_when
predicates for discriminated-union plugins. Step 1 prefill now reads from
GuidedSession.step_1_inspection_facts. Backend rejects submissions for hidden
fields (400 hidden_field_submitted).

This is the atomic commit mandated by spec §9 step 3 and Phase 9 convention 14
(wire-contract changes ship in one commit). Frontend and backend revert
together via git revert.
"
```

---

## Phase E — Recipe-fold (commit 5 of §9 step 4)

### Task 15: Switch recipe_offer dispatch to SCHEMA_FORM with mode=recipe_decision

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py:2584-2815` (recipe_offer dispatch)
- Modify: `src/elspeth/web/composer/guided/emitters.py:243-277` (recipe-offer emitter)
- Modify: `src/elspeth/web/composer/guided/protocol.py` (`RECIPE_OFFER` required/nested payload shape and docs)
- Delete: `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx`
- Delete: `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx` (remove RECIPE_OFFER case)

- [x] **Step 1: Write failing protocol/emitter tests**

Update `tests/unit/web/composer/guided/test_emitters.py` and `tests/unit/web/composer/guided/test_protocol.py` first:

- `build_step_2_5_recipe_offer_turn` emits `type=recipe_offer` with `payload.mode == "recipe_decision"`.
- The payload includes `knobs`, `prefilled`, and `recipe_context`.
- `validate_payload(TurnType.RECIPE_OFFER, payload)` accepts the new shape and rejects the old recipe-offer shape.
- `recipe_context.description` comes from `get_recipe(match.recipe_name).description`.
- Both accept and `build_manually` submit paths remain valid.

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_emitters.py tests/unit/web/composer/guided/test_protocol.py -v -k recipe`
Expected: FAIL on old payload/protocol shape.

- [x] **Step 2: Update the recipe-offer emitter**

Replace `build_step_2_5_recipe_offer_turn` to emit a SCHEMA_FORM turn with `mode=recipe_decision`:

```python
def build_step_2_5_recipe_offer_turn(match: RecipeMatch) -> Turn:
    """Build a recipe-offer turn as SCHEMA_FORM with mode=recipe_decision.

    The TurnType.RECIPE_OFFER discriminator is RETAINED so the state machine
    routing in _advance_step_2_5 is unchanged; only the rendering is unified
    via the SchemaFormPayload tagged union (spec §4.1 Choice 4).
    """
    from elspeth.web.composer.guided.errors import InvariantError
    from elspeth.web.composer.recipes import get_recipe
    recipe = get_recipe(match.recipe_name)
    if recipe is None:
        raise InvariantError(f"Recipe {match.recipe_name!r} disappeared from registry")
    from elspeth.web.catalog.knob_schema import lower_slot_specs_to_knob_schema
    knobs = lower_slot_specs_to_knob_schema(match.unsatisfied_slots)
    payload: SchemaFormPayload = {
        "mode": "recipe_decision",
        "knobs": knobs,
        "prefilled": dict(match.slots),
        "recipe_context": {
            "recipe_name": match.recipe_name,
            "description": recipe.description,
            "alternatives": ["build_manually"],
        },
    }
    return Turn(
        type=TurnType.RECIPE_OFFER.value,  # state-machine discriminator unchanged
        step_index=_step_index(GuidedStep.STEP_2_5_RECIPE_MATCH),
        payload=payload,
    )
```

- [x] **Step 3: Update protocol validation and docs**

Update `protocol._REQUIRED_KEYS` / `_NESTED_SHAPES` for `TurnType.RECIPE_OFFER` so it validates the tagged `SchemaFormPayload` shape (`mode`, `knobs`, `prefilled`, `recipe_context`) rather than the old `{recipe_name, slots, alternatives, unsatisfied_slots}` shape. Update comments/docstrings to explain the double discriminator: `turn.type == RECIPE_OFFER` routes the state machine, while `payload.mode == "recipe_decision"` routes the shared renderer.

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v -k recipe`
Expected: PASS.

- [x] **Step 4: Update GuidedTurn.tsx dispatcher**

Replace the case that routes RECIPE_OFFER → RecipeOfferTurn with the SCHEMA_FORM renderer:

```typescript
// GuidedTurn.tsx (excerpt)
case "recipe_offer":
  return <SchemaFormTurn payload={turn.payload as SchemaFormPayload} onSubmit={onSubmit} />;
```

`SchemaFormTurn` must render both recipe actions:

- Accept/apply sends `chosen: ["accept"]` with the edited slot values.
- Build manually sends `chosen: ["build_manually"]` and `edited_values: null`, preserving the current route contract.

If `TurnType.RECIPE_OFFER` remains the turn discriminator, update `protocol._REQUIRED_KEYS` and nested payload validation for the new `mode="recipe_decision"` payload shape.

- [x] **Step 5: Run dispatch tests**

Run: `.venv/bin/python -m pytest tests/unit/web/composer/guided/test_emitters.py tests/unit/web/composer/guided/test_protocol.py -v -k recipe`
Expected: tests need updating for the new payload shape; fix until green.

Run: `cd src/elspeth/web/frontend && npm test -- --run`
Expected: PASS.

- [x] **Step 6: Delete RecipeOfferTurn.tsx and its test**

```bash
rm src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx
rm src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.test.tsx
```

Audit imports:

Run: `grep -rn "RecipeOfferTurn" src/elspeth/web/frontend/src/`
Expected: no hits (or only in GuidedTurn.tsx if you forgot to remove an import — fix).

- [x] **Step 7: Pre-commit verification**

Run:

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_emitters.py tests/unit/web/composer/guided/test_protocol.py tests/integration/web/composer/ -v
cd src/elspeth/web/frontend && npm test -- --run
```

Expected: PASS.

- [x] **Step 8: Commit**

```bash
git add src/elspeth/web/composer/guided/emitters.py src/elspeth/web/sessions/routes.py src/elspeth/web/frontend/src/components/chat/guided/
git commit -m "feat(composer): fold recipe rendering into SchemaFormTurn

Recipe-offer turns now emit SCHEMA_FORM payloads with mode=recipe_decision
and a RecipeContextHeader. TurnType.RECIPE_OFFER discriminator retained for
state-machine routing (spec §4.1 Choice 4). RecipeOfferTurn.tsx deleted.
The three integration touchpoints (state machine, route dispatch, protocol
bindings) are unchanged; only the rendering path is unified.
"
```

---

## Phase F — Metadata CI enforcement (Task 16 already executed in Phase C2; Task 17 is §9 step 5b)

### Task 16: Audit and fill plugin config metadata

**Execution order:** run this task in Phase C2 before Phase D, even though it is documented next to the permanent CI lint. The renderer cutover is blocked until this audit is clean.

**Files:**
- Modify: every metadata-bearing plugin configuration model under `src/elspeth/plugins/` lacking `title` or `description` on any field, including provider-specific discriminated variants returned by `discriminated_variants()`

- [x] **Step 1: Discover the gap**

Run a quick audit script:

```bash
.venv/bin/python -c "
from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager

pm = get_shared_plugin_manager()
gaps = []

def iter_metadata_models(kind, cls):
    variants_fn = getattr(cls, 'discriminated_variants', None)
    if variants_fn is not None and callable(variants_fn):
        _, variants = variants_fn()
        for variant_name, model in variants.items():
            yield f'{kind}/{cls.name}[{variant_name}]', model
        return

    opts = getattr(cls, 'config_model', None)
    if opts is not None:
        yield f'{kind}/{cls.name}', opts

for kind in ('source', 'transform', 'sink'):
    listing = pm.get_sources() if kind == 'source' else (pm.get_transforms() if kind == 'transform' else pm.get_sinks())
    for cls in listing:
        for model_id, opts in iter_metadata_models(kind, cls):
            for name, info in opts.model_fields.items():
                if not info.title:
                    gaps.append((model_id, name, 'missing title'))
                if not info.description:
                    gaps.append((model_id, name, 'missing description'))
print(f'{len(gaps)} gaps')
for g in gaps[:50]:
    print(g)
"
```

This produces an initial gap list. Adjust the listing iteration to match the live plugin manager API if a specific list method differs, but keep `get_shared_plugin_manager()`. For discriminated plugins such as `LLMTransform`, audit every model returned by `discriminated_variants()` rather than only the base `config_model`, because the variant-only fields are exactly what become variant-specific knobs.

- [x] **Step 2: Fill the gaps**

For each gap, edit the owning plugin configuration model to add `Field(title=..., description=...)` (preferring the canonical `Annotated[T, Field(...)]` form). For discriminated plugins, this includes provider-specific variant models, not only the base `config_model`. Group commits by plugin package to keep commits reviewable.

- [x] **Step 3: Re-run the audit until empty**

Repeat Step 1 until `0 gaps`.

- [x] **Step 4: Commit (one or more per-plugin-package commits)**

```bash
# Example, per plugin package:
git add src/elspeth/plugins/sources/csv/options.py
git commit -m "chore(plugins): fill config field metadata for csv source

Adds title and description to all fields in CsvSource.config_model so the
composer renderer has authoritative labels. Per spec §11 metadata floor."
```

---

### Task 17: CI lint script + tests

**Files:**
- Create: `scripts/cicd/enforce_options_metadata.py`
- Create: `config/cicd/enforce_options_metadata/allowlist.yaml` (empty list)
- Test: `tests/unit/scripts/cicd/test_enforce_options_metadata.py`

- [x] **Step 1: Write the failing test**

```python
# tests/unit/scripts/cicd/test_enforce_options_metadata.py
import subprocess
import sys


def test_enforce_metadata_succeeds_on_current_plugins():
    """After Task 16 lands, every plugin config field has title and description.
    The lint script exits 0 against the current plugin catalog."""
    result = subprocess.run(
        [sys.executable, "scripts/cicd/enforce_options_metadata.py"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Lint failed: {result.stdout}\n{result.stderr}"


def test_enforce_metadata_fails_on_missing_title(tmp_path, monkeypatch):
    """If a synthetic plugin lacks title, the lint exits non-zero with a
    descriptive message."""
    from scripts.cicd.enforce_options_metadata import run_metadata_lint

    broken_catalog = make_plugin_manager_with_missing_metadata()
    failures = run_metadata_lint(plugin_manager=broken_catalog, allowlist=set())
    assert any("missing title" in failure for failure in failures)


def make_plugin_manager_with_missing_metadata():
    from pydantic import BaseModel, Field

    class _Options(BaseModel):
        missing_title: str = Field(description="Has description but no title")

    class _Source:
        name = "metadata_gap"
        config_model = _Options

    class _FakePluginManager:
        def get_sources(self):
            return [_Source]

        def get_transforms(self):
            return []

        def get_sinks(self):
            return []

    return _FakePluginManager()
```

- [x] **Step 2: Implement the script**

```python
#!/usr/bin/env python
"""CI lint: enforce title + description on every plugin configuration field.

Run from the project root: .venv/bin/python scripts/cicd/enforce_options_metadata.py

Allowlist at config/cicd/enforce_options_metadata/allowlist.yaml — entries are
of the form `<plugin_kind>/<plugin_name>:<field_name>` for single-model plugins
or `<plugin_kind>/<plugin_name>[<variant>]:<field_name>` for discriminated
variants, with a `reason` string. Empty allowlist enforces strictly."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import BaseModel

from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager


def run_metadata_lint(*, plugin_manager: object, allowlist: set[str]) -> list[str]:
    failures: list[str] = []
    pm = plugin_manager

    def iter_metadata_models(kind: str, cls: object) -> Iterator[tuple[str, type[BaseModel]]]:
        variants_fn = getattr(cls, "discriminated_variants", None)
        if variants_fn is not None and callable(variants_fn):
            _, variants = variants_fn()
            for variant_name, model in variants.items():
                yield f"{kind}/{cls.name}[{variant_name}]", model
            return

        opts = getattr(cls, "config_model", None)
        if opts is not None:
            yield f"{kind}/{cls.name}", opts

    for kind, listing in (
        ("source", pm.get_sources()),
        ("transform", pm.get_transforms()),
        ("sink", pm.get_sinks()),
    ):
        for cls in listing:
            for model_id, opts in iter_metadata_models(kind, cls):
                for fname, info in opts.model_fields.items():
                    ident = f"{model_id}:{fname}"
                    if ident in allowlist:
                        continue
                    if not info.title:
                        failures.append(f"{ident}: missing title")
                    if not info.description:
                        failures.append(f"{ident}: missing description")
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    allowlist_path = root / "config" / "cicd" / "enforce_options_metadata" / "allowlist.yaml"
    allowlist: set[str] = set()
    if allowlist_path.exists():
        data = yaml.safe_load(allowlist_path.read_text()) or {}
        for entry in data.get("entries", []):
            allowlist.add(entry["id"])

    failures = run_metadata_lint(plugin_manager=get_shared_plugin_manager(), allowlist=allowlist)

    if failures:
        print("Plugin config metadata lint failed:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print(
            "\nFix by adding title= and description= to each Field(...). "
            "See spec §11 for canonical annotation form.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 3: Create the empty allowlist**

```yaml
# config/cicd/enforce_options_metadata/allowlist.yaml
# Entries override the title+description lint per spec §11.
# Format: { id: "<kind>/<plugin_name>:<field_name>", reason: "<why>" }
entries: []
```

- [x] **Step 4: Run the lint and the tests**

Run: `.venv/bin/python scripts/cicd/enforce_options_metadata.py`
Expected: exit 0.

Run: `.venv/bin/python -m pytest tests/unit/scripts/cicd/test_enforce_options_metadata.py -v`
Expected: PASS.

- [x] **Step 5: Wire into CI**

Locate the CI workflow file (likely `.github/workflows/*.yml` or `pre-commit` config) and add an invocation of the lint as a fail-on-error check. The exact wiring depends on existing CI structure — examine the existing checks before integrating.

- [x] **Step 6: Commit**

```bash
git add scripts/cicd/enforce_options_metadata.py config/cicd/enforce_options_metadata/allowlist.yaml tests/unit/scripts/cicd/test_enforce_options_metadata.py .github/workflows/
git commit -m "ci: enforce config metadata (title + description) per plugin field

Closes the spec §11 governance floor: the renderer relies on Pydantic-emitted
title/description; this lint makes that reliance visible at CI time. Empty
allowlist enforces strictly. Failure messages cite the offending field and
remediation.
"
```

---

## Self-Review

### Spec coverage check

| Spec section | Task(s) implementing it |
|---|---|
| §1 Problem | Implicit (motivates all tasks) |
| §2 Choices 1-6 | Choice 1 (JSON Schema canonical): Task 3, 5; Choice 2 (Pydantic authoring): Task 16; Choice 3 (server lowering): Tasks 3-6; Choice 4 (recipe-fold mode): Task 15; Choice 5 (inspection prefill): Tasks 9, 12; Choice 6 (composer_tier wire-optional): Task 3 (`_attach_tier`) |
| §3 Architecture (catalog API extension) | Tasks 7, 8 |
| §4 Wire shape | Task 2 (types), Task 10 (protocol re-export), Task 14 (TS types) |
| §4.1 Discriminated-union + visible_when | Task 5 (lowering), Task 6 (validator), Task 14 (frontend evaluation), Task 13 (backend rejection) |
| §5 Lowering function | Tasks 3, 4, 5 |
| §6 Failure modes | Task 2 (`KnobSchemaLoweringError`), Tasks 3, 5, 6, 13 |
| §7 Return-path round-trip | Task 3 return-path validator tests (cases 1-3), Task 14 tests (cases 4 and 9), Task 13 test (case 10) |
| §8 Trust-tier table | Implicit in design; emitter doesn't change tier (Task 11) |
| §9 Sequencing | Phase A=step 1; Phase B continues step 1; Phase C=step 2; Phase D=step 3 atomic; Phase E=step 4; Phase F=steps 5a/5b |
| §10 Test strategy | Tasks 3-6 plus Task 6.5 (golden snapshots + Hypothesis), Task 8 integration smoke |
| §11 CI lint | Task 16 before Phase D, Task 17 permanent CI gate |
| §12 Adapter retirement | No task — passive criterion acknowledged in spec |
| §13 Out of scope | No tasks — documented |
| §14 Verification gate | Plan-review blockers accepted; Phase D policy gates added before atomic commit |

No known gaps after the 2026-05-14 review-revision pass.

### Placeholder scan

No "TBD" / "TODO" / "Add appropriate error handling" / "Similar to Task N" patterns. Code blocks must not contain ellipsis placeholders; helper fixtures named in tests must be implemented in the same task.

### Type consistency

- `KnobField` shape consistent across tasks 2, 3, 4, 5 (Python) and Task 14 (TypeScript).
- `lower_model_to_knob_schema`, `lower_discriminated_to_knob_schema`, `lower_slot_specs_to_knob_schema` — all three lowering functions defined and consumed in Task 8 (`CatalogServiceImpl._build_schema_info`).
- `validate_knob_schema` — defined in Task 6, called in Task 8.
- `DiscriminatedPlugin` protocol — defined in Task 1; Tasks 5 and 8 dispatch structurally with `getattr(..., "discriminated_variants", None)` plus `callable(...)`, matching the spec's runtime catalog-load behavior.
- `_reject_hidden_field_submissions` — defined in Task 13, called in Task 13 (and again in Task 15 implicitly via the recipe-fold dispatch).
- `SourceInspectionFacts` — already exists in codebase; persisted on `GuidedSession` in Task 9; consumed in Task 11 emitter.

All signatures and names match across tasks.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-14-composer-one-knob.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best for an 18-task plan with clear inter-task dependencies and verifiable per-task tests.

2. **Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batched with checkpoints at phase boundaries (A, B, C, D, E, F).

Which approach?
