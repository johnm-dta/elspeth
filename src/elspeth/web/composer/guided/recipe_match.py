"""Deterministic recipe matcher: topology-only, pure function, no I/O.

``match_recipe(source, sink)`` runs at Step 2.5 to detect if an existing
registered recipe fits the operator's (source, sink) shape. On a match, it
returns a :class:`RecipeMatch` with a **partial** slot map — only the slots
derivable from observed (source, sink) state are populated. The operator fills
the remaining required slots via a ``recipe_offer`` turn at Step 2.5.

Boundary contract for slot resolvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Two kinds of values are read from ``source.options`` and
``sink.outputs[i].options``:

``blob_ref`` (``source.options["blob_ref"]``)
    The composer-canonical blob UUID, written by
    ``_execute_set_source_from_blob`` (tools.py) **and** by
    ``handle_step_1_source`` (steps.py) when the submitted path resolves to
    a known uploaded blob.  This is Tier-2 data (already validated by our
    own code) — direct subscript access is mandatory, no ``.get()``.  A
    missing ``blob_ref`` crashes with an informative ``ValueError`` (the
    resolver must only be called after blob-backed source commit).

Output paths (``sink.outputs[i].options.get("path", default)``)
    User/LLM-supplied via the SchemaForm, so genuinely Tier-3.  A default
    is the documented coercion: an absent path produces a rubber-stampable
    suggested value that the operator can confirm or change via the
    ``recipe_offer`` turn.

Predicates match on **topology only** (source plugin + sink output count and
plugin). They do NOT return False just because some required slots cannot be
derived — that would make Step 2.5 a dead letter. Slot resolvers are partial
by design.

Recipe coverage in v1
~~~~~~~~~~~~~~~~~~~~~
- ``classify-rows-llm-jsonl``:  CSV → single JSON output with classifier-keyword
  in required_fields.
- ``split-by-numeric-threshold``:  CSV → two JSON outputs.
- ``fork-coalesce-truncate-jsonl``:  intentionally omitted in v1.  The recipe
  requires structural intent (truncate one arm, key-based coalesce) that cannot
  be inferred from (source, sink) alone.  Additionally, its topology (CSV →
  single JSON output) overlaps with classify — ordering alone cannot
  disambiguate; only the classifier-keyword check separates them.  Adding an
  R3 predicate without a clear distinguishing signal would produce ambiguous
  matches.  Users who want this recipe must hand-build via Step 3 or
  pre-select via ``list_recipes``.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §6.4
and Errata C8.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.state_machine import SinkResolved, SourceResolved
from elspeth.web.composer.recipes import SlotSpec, get_recipe

# ---------------------------------------------------------------------------
# Type aliases for the predicate registry
# ---------------------------------------------------------------------------

_Predicate = Callable[[SourceResolved, SinkResolved], bool]
_SlotResolver = Callable[[SourceResolved, SinkResolved], Mapping[str, Any]]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecipeMatch:
    """Result of matching a recipe to a (source, sink) tuple.

    ``slots`` is a *partial* map: only the slots derivable from observed
    state are populated. The route handler emits a ``recipe_offer`` turn
    that asks the operator to fill the remaining required slots before
    ``_execute_apply_pipeline_recipe`` is invoked (see Errata C2).

    ``unsatisfied_slots`` is the schema for the required slots NOT covered by
    ``slots`` — the emitter projects these to a wire-friendly shape so the
    frontend can render an editable form for them.  Only *required* slots are
    included; optional slots with declared defaults are auto-filled by
    ``validate_slots`` at apply time and are not surfaced to the operator.
    """

    recipe_name: str
    slots: Mapping[str, Any]
    unsatisfied_slots: Mapping[str, SlotSpec]

    def __post_init__(self) -> None:
        freeze_fields(self, "slots", "unsatisfied_slots")


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _is_csv(source: SourceResolved) -> bool:
    return source.plugin == "csv"


def _has_single_json_output(sink: SinkResolved) -> bool:
    return len(sink.outputs) == 1 and sink.outputs[0].plugin == "json"


def _has_two_json_outputs(sink: SinkResolved) -> bool:
    return len(sink.outputs) == 2 and all(o.plugin == "json" for o in sink.outputs)


# ---------------------------------------------------------------------------
# Classify-rows-llm-jsonl predicate and slot resolver
# ---------------------------------------------------------------------------

_CLASSIFY_KEYWORDS: frozenset[str] = frozenset({"category", "label", "tag", "classification"})


def _classify_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
    """Return True for CSV → single-JSON with a classifier-keyword required field.

    The required_fields check is the only topology signal that separates
    classify from fork-coalesce-truncate (which also produces single JSON
    output from CSV). Without it, recipe selection would be ambiguous.
    """
    if not (_is_csv(source) and _has_single_json_output(sink)):
        return False
    return any(name in _CLASSIFY_KEYWORDS for name in sink.outputs[0].required_fields)


def _classify_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
    """Partial slot map for the classify-rows-llm-jsonl recipe.

    Provides: ``source_blob_id``, ``output_path``, ``label_field``.
    User-fillable: ``classifier_template``, ``model``, ``api_key_secret``,
    ``provider``, ``required_input_fields``.

    ``label_field`` is the first required_field name that belongs to
    ``_CLASSIFY_KEYWORDS``.  Using the keyword-matching field rather than
    ``required_fields[0]`` produces an honest pre-fill — the operator may
    rubber-stamp the default via the ``recipe_offer`` turn, so it should be
    the most likely correct value.

    ``source.options["blob_ref"]`` is the composer-canonical blob UUID.
    It is written by ``_execute_set_source_from_blob`` for blob-upload flows
    and by ``handle_step_1_source`` (steps.py) for SchemaForm flows where
    the submitted path resolves to an uploaded blob.  A missing ``blob_ref``
    is a state-machine invariant violation — this resolver must only be
    reached for blob-backed sources.
    """
    if "blob_ref" not in source.options:
        raise ValueError(
            "Recipe slot resolver requires source.options['blob_ref'] "
            "(set by _execute_set_source_from_blob or handle_step_1_source "
            "blob enrichment path); source options present: "
            f"{sorted(source.options.keys())}"
        )
    blob_ref = source.options["blob_ref"]
    output_path = sink.outputs[0].options.get("path", "outputs/classified.jsonl")
    # Pick the first keyword-matching required_field; predicate guarantees one exists.
    label_field = next(n for n in sink.outputs[0].required_fields if n in _CLASSIFY_KEYWORDS)
    return {
        "source_blob_id": blob_ref,
        "output_path": output_path,
        "label_field": label_field,
    }


# ---------------------------------------------------------------------------
# Split-by-numeric-threshold predicate and slot resolver
# ---------------------------------------------------------------------------


def _split_threshold_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
    """Return True for CSV → two JSON outputs (regardless of required_fields)."""
    return _is_csv(source) and _has_two_json_outputs(sink)


def _split_threshold_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
    """Partial slot map for the split-by-numeric-threshold recipe.

    Provides: ``source_blob_id``, ``above_output_path``, ``below_output_path``.
    User-fillable: ``field``, ``threshold``.

    ``source.options["blob_ref"]`` is the composer-canonical blob UUID;
    same contract as ``_classify_slot_resolver`` — must be present.
    ``sink.outputs[i].options.get("path", default)`` is Tier-3 with a
    rubber-stampable suggested default.
    """
    if "blob_ref" not in source.options:
        raise ValueError(
            "Recipe slot resolver requires source.options['blob_ref'] "
            "(set by _execute_set_source_from_blob or handle_step_1_source "
            "blob enrichment path); source options present: "
            f"{sorted(source.options.keys())}"
        )
    blob_ref = source.options["blob_ref"]
    above_path = sink.outputs[0].options.get("path", "outputs/above.jsonl")
    below_path = sink.outputs[1].options.get("path", "outputs/below.jsonl")
    return {
        "source_blob_id": blob_ref,
        "above_output_path": above_path,
        "below_output_path": below_path,
    }


# ---------------------------------------------------------------------------
# Predicate registry — most-specific first, first match wins
# ---------------------------------------------------------------------------

_RECIPE_PREDICATES: Sequence[tuple[_Predicate, str, _SlotResolver]] = (
    (_classify_predicate, "classify-rows-llm-jsonl", _classify_slot_resolver),
    (_split_threshold_predicate, "split-by-numeric-threshold", _split_threshold_slot_resolver),
    # fork-coalesce-truncate-jsonl predicate intentionally omitted in v1; that
    # recipe requires structural intent (truncate one arm, key-based coalesce)
    # that the matcher cannot infer from (source, sink) alone.  Even if a
    # predicate were added later, topology cannot disambiguate classify vs R3
    # (both are CSV → single JSON) — only the classifier-keyword check separates
    # them.  Users who want fork-coalesce must hand-build via Step 3 or
    # pre-select via list_recipes.
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def match_recipe(
    source: SourceResolved,
    sink: SinkResolved,
) -> RecipeMatch | None:
    """Return the most-specific recipe matching the tuple, or None.

    Topology-only. Predicates examine source plugin + sink output count
    and plugin. Slot resolvers derive what they can; operator fills the
    rest via the ``recipe_offer`` turn at Step 2.5.

    First match in ``_RECIPE_PREDICATES`` wins; order is most-specific first.
    """
    for predicate, recipe_name, slot_resolver in _RECIPE_PREDICATES:
        if predicate(source, sink):
            resolved_slots = slot_resolver(source, sink)
            # Look up the recipe to derive the unsatisfied required-slot
            # schema. The predicate registry's recipe_name MUST be registered
            # — if not, that's an invariant violation, crash loudly.
            recipe = get_recipe(recipe_name)
            if recipe is None:
                raise ValueError(f"Recipe '{recipe_name}' is in _RECIPE_PREDICATES but not registered in recipes.py — invariant violation.")
            unsatisfied: dict[str, SlotSpec] = {
                name: spec for name, spec in recipe.slots.items() if spec.required and name not in resolved_slots
            }
            return RecipeMatch(
                recipe_name=recipe_name,
                slots=resolved_slots,
                unsatisfied_slots=unsatisfied,
            )
    return None
