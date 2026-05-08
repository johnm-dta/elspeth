"""Pipeline recipes — deterministic scaffolds for common simple-pipeline intents.

Step 5 of the composer simple-pipeline-convergence program.

A recipe takes operator-supplied slot values, validates them against a
typed schema, and returns a ``set_pipeline`` arguments dict ready for
execution. The resulting pipeline state is identical to one a model would
hand-author via ``set_pipeline`` directly — recipes are accelerators, not
opaque shortcuts. The model still sees the full state via
``get_pipeline_state`` after recipe application and can ``patch_*_options``
to refine.

Slot validation runs *before* scaffolding. This catches the URL-as-path
class of bugs (a string blob path passed where a blob UUID is required,
a numeric threshold passed as a string, etc.) at the recipe boundary
rather than at runtime. Each recipe declares its slot schema as a
mapping of slot name → ``SlotSpec`` describing the required type and
optional default.

Boundary contract:
  * Slot values are operator-supplied — coerce at the boundary (str →
    UUID, str → float for numeric thresholds) but do NOT fabricate
    defaults beyond what the slot schema declares.
  * Recipes never call external services or read blob bytes — they
    construct config only. Source inspection (Step 2) and proof step
    (Step 3) handle blob reads.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal
from uuid import UUID

SlotType = Literal["blob_id", "str", "float", "int"]


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """Declares one input slot for a recipe.

    Attributes:
        slot_type: How to validate the operator-supplied value. ``blob_id``
            requires a parseable UUID string; ``float``/``int`` allow
            numeric strings (coerced) or numeric primitives.
        required: When True, omitting this slot is a validation error.
        default: Used only when ``required=False`` and the slot is absent.
        description: Human-readable help text surfaced to the model.
    """

    slot_type: SlotType
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class RecipeSpec:
    """Declares a recipe — its slot schema and the scaffold function."""

    name: str
    description: str
    slots: Mapping[str, SlotSpec]
    build: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    """Pure function: validated slots → set_pipeline-compatible args dict."""


class RecipeValidationError(ValueError):
    """Raised when operator-supplied slots fail validation."""


def _coerce_slot(name: str, spec: SlotSpec, raw: Any) -> Any:
    """Validate and coerce one slot value against its declared type."""
    if spec.slot_type == "blob_id":
        if not isinstance(raw, str):
            raise RecipeValidationError(
                f"slot '{name}' must be a UUID string for a session blob "
                f"(got type {type(raw).__name__}). To use a URL, first call "
                "create_blob with mime_type='text/plain' to wrap it."
            )
        try:
            UUID(raw)
        except ValueError as exc:
            raise RecipeValidationError(
                f"slot '{name}' must be a valid UUID (got {raw!r}). To use a "
                "URL, first call create_blob with mime_type='text/plain' to "
                "wrap it; the returned blob_id is what this slot accepts."
            ) from exc
        return raw

    if spec.slot_type == "str":
        if not isinstance(raw, str):
            raise RecipeValidationError(f"slot '{name}' must be a string (got type {type(raw).__name__})")
        return raw

    if spec.slot_type == "float":
        if isinstance(raw, bool):
            raise RecipeValidationError(f"slot '{name}' must be a number (got bool — use 0.0 or 1.0 explicitly)")
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError as exc:
                raise RecipeValidationError(f"slot '{name}' must be a number; could not coerce {raw!r} to float") from exc
        raise RecipeValidationError(f"slot '{name}' must be a number (got type {type(raw).__name__})")

    if spec.slot_type == "int":
        if isinstance(raw, bool):
            raise RecipeValidationError(f"slot '{name}' must be an integer (got bool — use 0 or 1 explicitly)")
        if isinstance(raw, int):
            return raw
        if isinstance(raw, str):
            try:
                return int(raw)
            except ValueError as exc:
                raise RecipeValidationError(f"slot '{name}' must be an integer; could not coerce {raw!r}") from exc
        raise RecipeValidationError(f"slot '{name}' must be an integer (got type {type(raw).__name__})")

    raise RecipeValidationError(f"recipe slot type {spec.slot_type!r} is not implemented")


def validate_slots(recipe: RecipeSpec, raw_slots: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a raw slots dict against a recipe's declared schema.

    Returns a new dict containing only the recipe's declared slots,
    coerced to their declared types. Raises ``RecipeValidationError``
    on missing required slots, unknown slot names, or type-coercion
    failures.
    """
    unknown = set(raw_slots) - set(recipe.slots)
    if unknown:
        raise RecipeValidationError(f"recipe '{recipe.name}' does not accept slot(s): {sorted(unknown)}. Accepted: {sorted(recipe.slots)}.")
    coerced: dict[str, Any] = {}
    for slot_name, spec in recipe.slots.items():
        if slot_name in raw_slots:
            coerced[slot_name] = _coerce_slot(slot_name, spec, raw_slots[slot_name])
        elif spec.required:
            raise RecipeValidationError(
                f"recipe '{recipe.name}' is missing required slot '{slot_name}': {spec.description or spec.slot_type}"
            )
        else:
            coerced[slot_name] = spec.default
    return coerced


# ---------------------------------------------------------------------------
# Recipe 1: classify-rows-llm-jsonl
#
#   csv source (blob)  →  llm transform (response stored in label_field)
#                       →  jsonl sink (single output)
# ---------------------------------------------------------------------------


_RECIPE1_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied CSV blob (use create_blob to wrap inline content first)",
    ),
    "classifier_template": SlotSpec(
        slot_type="str",
        description="Jinja2 template for the LLM prompt; reference row fields as {{ row['col'] }}",
    ),
    "model": SlotSpec(
        slot_type="str",
        description="LLM model identifier (e.g., 'anthropic/claude-3.5-sonnet'); use list_models to discover",
    ),
    "provider": SlotSpec(
        slot_type="str",
        required=False,
        default="openrouter",
        description="LLM provider — 'openrouter' or 'azure'",
    ),
    "label_field": SlotSpec(
        slot_type="str",
        required=False,
        default="classification",
        description="Row field name where the LLM response is written",
    ),
    "output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/classified.jsonl",
        description="JSONL output path",
    ),
}


def _build_classify_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the classify-rows-llm-jsonl recipe."""
    # ``blob_id`` is a TOP-LEVEL key of ``source`` (sibling of ``options``),
    # NOT a member of ``options``. ``_execute_set_pipeline`` reads it via
    # ``src_args.get("blob_id")`` and feeds it to ``_resolve_source_blob``,
    # which authoritatively materialises ``options["path"]`` and the
    # canonical ``options["blob_ref"]``. Putting ``blob_id`` inside
    # ``options`` would skip resolution and leave the source unbound — the
    # proof step (``compute_proof_diagnostics`` reads ``options["blob_ref"]``)
    # would then silently report no diagnostics.
    return {
        "source": {
            "plugin": "csv",
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "classifier",
                "node_type": "transform",
                "plugin": "llm",
                "input": "rows",
                "on_success": "labelled",
                "on_error": "discard",
                "options": {
                    "provider": slots["provider"],
                    "model": slots["model"],
                    "template": slots["classifier_template"],
                    "response_field": slots["label_field"],
                },
            }
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "labelled",
                "plugin": "json",
                "options": {
                    "path": slots["output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            }
        ],
        "metadata": {
            "name": "classify-rows-llm-jsonl",
            "description": (
                f"LLM classification of CSV rows; classification stored in field "
                f"'{slots['label_field']}', written to {slots['output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Recipe 2: split-by-numeric-threshold
#
#   csv source (blob)  →  type_coerce (numeric field)
#                       →  gate (row[field] >= threshold)
#                       →  above_output sink + below_output sink
# ---------------------------------------------------------------------------


_RECIPE2_SLOTS: Final[dict[str, SlotSpec]] = {
    "source_blob_id": SlotSpec(
        slot_type="blob_id",
        description="UUID of the operator-supplied CSV blob",
    ),
    "field": SlotSpec(
        slot_type="str",
        description="Column to compare against the threshold (must be numeric or coercible)",
    ),
    "threshold": SlotSpec(
        slot_type="float",
        description="Numeric threshold; rows with field >= threshold go to above_output_path",
    ),
    "above_output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/above.jsonl",
        description="JSONL output for rows meeting/exceeding the threshold",
    ),
    "below_output_path": SlotSpec(
        slot_type="str",
        required=False,
        default="outputs/below.jsonl",
        description="JSONL output for rows below the threshold",
    ),
}


def _build_threshold_recipe(slots: Mapping[str, Any]) -> dict[str, Any]:
    """Build set_pipeline args for the split-by-numeric-threshold recipe."""
    field = slots["field"]
    threshold = slots["threshold"]
    return {
        "source": {
            "plugin": "csv",
            "blob_id": slots["source_blob_id"],
            "on_success": "rows",
            "options": {
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "coerce_numeric",
                "node_type": "transform",
                "plugin": "type_coerce",
                "input": "rows",
                "on_success": "numeric_rows",
                "on_error": "discard",
                "options": {
                    "conversions": [{"field": field, "to": "float"}],
                },
            },
            {
                "id": "threshold_gate",
                "node_type": "gate",
                "input": "numeric_rows",
                "condition": f"row['{field}'] >= {threshold}",
                "routes": {"true": "above", "false": "below"},
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "above",
                "plugin": "json",
                "options": {
                    "path": slots["above_output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
            {
                "sink_name": "below",
                "plugin": "json",
                "options": {
                    "path": slots["below_output_path"],
                    "format": "jsonl",
                    "schema": {"mode": "observed"},
                    "collision_policy": "auto_increment",
                },
                "on_write_failure": "discard",
            },
        ],
        "metadata": {
            "name": "split-by-numeric-threshold",
            "description": (
                f"CSV rows split by {field} >= {threshold}; above → {slots['above_output_path']}, below → {slots['below_output_path']}"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_RECIPES: Final[dict[str, RecipeSpec]] = {
    "classify-rows-llm-jsonl": RecipeSpec(
        name="classify-rows-llm-jsonl",
        description=(
            "Apply an LLM classifier to every row of a CSV blob and write a "
            "JSONL output with each row's classification stored in a named "
            "field. Use for: 'classify these tickets as high/medium/low', "
            "'tag these reviews as positive/negative', 'pick a category for "
            "each row'. The CSV must already be uploaded as a session blob."
        ),
        slots=_RECIPE1_SLOTS,
        build=_build_classify_recipe,
    ),
    "split-by-numeric-threshold": RecipeSpec(
        name="split-by-numeric-threshold",
        description=(
            "Split CSV rows by a numeric threshold into two JSONL outputs. "
            "Coerces the field to float before comparison so a string-typed "
            "CSV column is handled correctly. Use for: 'route prices above "
            "100 to high.jsonl', 'separate scores >= 0.8 from the rest', "
            "'split orders by amount'."
        ),
        slots=_RECIPE2_SLOTS,
        build=_build_threshold_recipe,
    ),
}


def list_recipes() -> list[dict[str, Any]]:
    """Return discovery metadata for every registered recipe."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "slots": {
                slot_name: {
                    "type": s.slot_type,
                    "required": s.required,
                    "default": s.default,
                    "description": s.description,
                }
                for slot_name, s in spec.slots.items()
            },
        }
        for spec in _RECIPES.values()
    ]


def get_recipe(name: str) -> RecipeSpec | None:
    """Return a recipe spec by name, or None if not registered."""
    return _RECIPES.get(name)


def apply_recipe(name: str, raw_slots: Mapping[str, Any]) -> dict[str, Any]:
    """Validate slots and return the set_pipeline args for a recipe.

    Raises ``RecipeValidationError`` if the recipe is unknown or the
    slots fail validation. The returned dict is consumable directly by
    ``set_pipeline``.
    """
    recipe = _RECIPES.get(name)
    if recipe is None:
        raise RecipeValidationError(f"recipe '{name}' is not registered. Available recipes: {sorted(_RECIPES)}.")
    coerced = validate_slots(recipe, raw_slots)
    # Concrete recipe builders return dict; the Mapping return type on the
    # RecipeSpec contract is the looser superset (Mapping ⊇ dict). Convert
    # to the concrete dict the caller (set_pipeline executor) requires.
    return dict(recipe.build(coerced))
