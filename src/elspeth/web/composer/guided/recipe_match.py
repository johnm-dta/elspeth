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
    own code) — direct subscript access is mandatory, no ``.get()``.

    **Both predicates gate on ``"blob_ref" in source.options`` before returning
    True**, so a CSV source without a blob upload will produce no recipe match
    (returning ``None``) rather than crashing in the resolver.  CSV-without-blob
    is a legitimate user configuration; the correct outcome is that the flow
    continues to manual chain solving via Step 3.  Slot resolvers retain their
    ``InvariantError`` guard as defence-in-depth for any future caller that
    bypasses the predicate registry.

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
  in required_fields.  Reachable from the real guided flow.
- ``split-by-numeric-threshold``:  CSV → two JSON outputs.  Predicate is in
  the registry but is **currently unreachable from guided-flow Step 2**: the Step 2
  state machine always produces a single-output sink (``_advance_step_2`` hard-codes
  ``SinkResolved(outputs=(output,))``).  Enabling this recipe requires a multi-output
  Step 2 UI and state-machine refactor.  Kept in the registry for forward-compat;
  tracked at elspeth-obs-74a708e3d7.
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
from typing import Any, cast

from elspeth.contracts.composer_slots import SlotSpec, SlotType
from elspeth.contracts.freeze import deep_thaw, freeze_fields
from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.resolved import SinkResolved, SourceResolved
from elspeth.web.composer.recipes import get_recipe

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
        # Offensive invariants: the documented contract on ``unsatisfied_slots``
        # is enforced by ``match_recipe`` at the construction site, but any
        # *other* caller (apply-time reconstruction in routes.py, test fixtures,
        # future code paths) could violate it silently.  Pin the contract here
        # so violations crash at construction with an informative message
        # rather than producing audit-trail garbage at apply time.
        #
        # Invariant 1: key-disjointness — a slot cannot be both "resolved" and
        # "unsatisfied".  The two mappings are complementary by definition.
        overlap = set(self.unsatisfied_slots) & set(self.slots)
        if overlap:
            raise InvariantError(f"RecipeMatch.unsatisfied_slots must be disjoint from slots; overlap on: {sorted(overlap)}")
        # Invariant 2: only required slots belong in ``unsatisfied_slots``.
        # Optional slots with declared defaults are auto-filled by
        # ``validate_slots`` at apply time and must not be surfaced to the
        # operator as fields they need to fill.
        for name, spec in self.unsatisfied_slots.items():
            if not spec.required:
                raise InvariantError(
                    f"RecipeMatch.unsatisfied_slots[{name!r}] is optional; "
                    "only required slots belong in unsatisfied_slots "
                    "(optional slots are auto-filled by validate_slots)"
                )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-serialisable dict.

        ``slots`` values are arbitrary (Tier-2, already validated). ``unsatisfied_slots``
        values are ``SlotSpec`` instances — serialised as plain dicts with scalar fields.
        """
        unsatisfied: dict[str, dict[str, Any]] = {
            name: {
                "slot_type": spec.slot_type,
                "required": spec.required,
                "description": spec.description,
                "default": spec.default,
            }
            for name, spec in self.unsatisfied_slots.items()
        }
        return {
            "recipe_name": self.recipe_name,
            "slots": dict(deep_thaw(self.slots)),
            "unsatisfied_slots": unsatisfied,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecipeMatch:
        """Reconstruct from a plain dict.  Tier 1 strict — crashes on bad data.

        Used when restoring ``step_2_5_recipe_offer`` from ``GuidedSession.from_dict``.
        The dict must have been produced by ``to_dict()``.
        """
        try:
            unsatisfied: dict[str, SlotSpec] = {
                name: SlotSpec(
                    slot_type=cast(SlotType, spec_d["slot_type"]),
                    required=spec_d["required"],
                    description=spec_d["description"],
                    default=spec_d["default"],
                )
                for name, spec_d in d["unsatisfied_slots"].items()
            }
            return cls(
                recipe_name=d["recipe_name"],
                slots=d["slots"],
                unsatisfied_slots=unsatisfied,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise InvariantError(f"RecipeMatch.from_dict: malformed record {d!r}") from exc


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
# web-scrape-llm-rate-jsonl predicate
#
# web_scrape is a TRANSFORM (plugins/transforms/web_scrape.py); the predicate
# keys on the URL-ROW SOURCE that feeds it, never on web_scrape itself. An
# inline_blob URL list materialises to a real registered source plugin via
# _MIME_TO_SOURCE (tools/sources.py): JSON rows -> "json", CSV -> "csv". The
# predicate matches those resolved names + a "url" column signal + a single
# jsonl output, gated on blob_ref (same blob-presence discipline as
# _classify_predicate).
# ---------------------------------------------------------------------------

_URL_ROW_SOURCE_PLUGINS: frozenset[str] = frozenset({"json", "csv"})
_URL_COLUMN_NAMES: frozenset[str] = frozenset({"url"})


def _has_single_jsonl_output(sink: SinkResolved) -> bool:
    """Return True for a single ``json`` output configured as JSONL.

    The canonical web-scrape pipeline writes one JSONL file. ``json`` is the
    registered sink plugin; ``format: jsonl`` is the JSONL discriminator (an
    absent format is the json plugin's default object-array, not JSONL).
    """
    if not (len(sink.outputs) == 1 and sink.outputs[0].plugin == "json"):
        return False
    return sink.outputs[0].options.get("format") == "jsonl"


def _source_has_url_column(source: SourceResolved) -> bool:
    """Return True iff the source surfaces a ``url`` column.

    The signal is the observed URL column that web_scrape's ``url_field``
    will read. Observed columns come from inspecting the materialised blob;
    a URL list always surfaces ``url``.
    """
    return any(col in _URL_COLUMN_NAMES for col in source.observed_columns)


def _web_scrape_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
    """Return True for a blob-backed URL-row source → single JSONL output.

    Matches the canonical tutorial shape: an inline_blob URL list that
    materialised to a ``json``/``csv`` source (NOT ``web_scrape`` — that is a
    downstream transform the recipe inserts) feeding a single JSONL sink, with
    an observed ``url`` column.

    Requires ``blob_ref`` in ``source.options`` for the same reason as
    ``_classify_predicate``: the slot resolver cannot derive ``source_blob_id``
    without it, and "no recipe match" (fall through to the live chain solver)
    is the correct outcome for a non-blob-backed URL source.
    """
    if source.plugin not in _URL_ROW_SOURCE_PLUGINS:
        return False
    if "blob_ref" not in source.options:
        return False
    if not _has_single_jsonl_output(sink):
        return False
    return _source_has_url_column(source)


def _web_scrape_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
    """Partial slot map for the web-scrape-llm-rate-jsonl recipe.

    Provides ``source_blob_id`` (the composer-canonical blob UUID),
    ``source_plugin`` (the real materialised source plugin: json or csv), and
    ``output_path`` (operator-set verbatim, else a rubber-stampable default).
    User-fillable: ``model``, ``api_key_secret``, ``provider``,
    ``rating_template``, ``abuse_contact``, and ``scraping_reason``.

    Auto-derives the three slots inferable from (source, sink); the remaining
    required slots (``model``, ``api_key_secret``, ``abuse_contact``,
    ``scraping_reason``) surface as ``unsatisfied_slots`` for the operator to
    fill via ``recipe_offer``.  The ``RecipeSpec`` this pre-fills against is
    registered in recipes.py (``web-scrape-llm-rate-jsonl``).

    Retains the ``blob_ref`` InvariantError guard for the same defence-in-depth
    reason as the classify/split resolvers: the predicate gates on
    ``blob_ref in source.options``, so this is structurally unreachable via
    ``match_recipe``; any caller that bypasses the registry gets a clear crash.
    """
    if "blob_ref" not in source.options:
        raise InvariantError(
            f"web-scrape recipe slot resolver requires source.options['blob_ref']; source options present: {sorted(source.options.keys())}"
        )
    blob_ref = source.options["blob_ref"]
    sink_options = sink.outputs[0].options
    if "path" in sink_options:
        output_path = sink_options["path"]
    else:
        output_path = "outputs/ratings.jsonl"
    return {
        "source_blob_id": blob_ref,
        "source_plugin": source.plugin,
        "output_path": output_path,
    }


# ---------------------------------------------------------------------------
# Classify-rows-llm-jsonl predicate and slot resolver
# ---------------------------------------------------------------------------

_CLASSIFY_KEYWORDS: frozenset[str] = frozenset({"category", "label", "tag", "classification"})


def _classify_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
    """Return True for blob-backed CSV → single-JSON with a classifier-keyword required field.

    Requires ``blob_ref`` in ``source.options``: a CSV source configured via
    SchemaForm with a direct file path (no blob upload) has no ``blob_ref`` and
    must NOT match — the slot resolver cannot run without it, and "no recipe match"
    is the correct outcome (the flow continues to manual chain solving).

    The required_fields check is the only topology signal that separates
    classify from fork-coalesce-truncate (which also produces single JSON
    output from CSV). Without it, recipe selection would be ambiguous.
    """
    if not (_is_csv(source) and _has_single_json_output(sink)):
        return False
    if "blob_ref" not in source.options:
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
    the submitted path resolves to an uploaded blob.

    The ``_classify_predicate`` gate now requires ``blob_ref in source.options``
    before returning True, so this InvariantError is structurally unreachable from
    any call path through ``match_recipe``.  It is kept as defence-in-depth: any
    future caller that bypasses the predicate registry and invokes this resolver
    directly will get a clear, immediately-informative crash rather than a silent
    ``KeyError`` or opaque downstream failure.
    """
    if "blob_ref" not in source.options:
        raise InvariantError(
            "Recipe slot resolver requires source.options['blob_ref'] "
            "(set by _execute_set_source_from_blob or handle_step_1_source "
            "blob enrichment path); source options present: "
            f"{sorted(source.options.keys())}"
        )
    blob_ref = source.options["blob_ref"]
    # Tier-3 sink options. An operator-configured ``path`` is used verbatim;
    # its absence is an explicit decision to offer the recipe's default JSONL
    # destination as a rubber-stampable pre-fill (not a silent fallback).
    sink_options = sink.outputs[0].options
    if "path" in sink_options:
        output_path = sink_options["path"]
    else:
        output_path = "outputs/classified.jsonl"
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
    """Return True for blob-backed CSV → two JSON outputs (regardless of required_fields).

    Requires ``blob_ref`` in ``source.options`` for the same reason as
    ``_classify_predicate``: a CSV source without a blob upload has no
    ``blob_ref``, and the slot resolver cannot run without it.  Returning
    False here means "no recipe match" — the flow continues to manual chain
    solving, which is the correct outcome for a non-blob-backed CSV source.

    **Currently unreachable from guided-flow Step 2.**

    ``_advance_step_2`` (state_machine.py) hard-codes
    ``SinkResolved(outputs=(output,))`` — the state machine always produces a
    single-output sink. ``_has_two_json_outputs`` therefore never returns True
    from the real flow.  The predicate is kept in ``_RECIPE_PREDICATES`` for
    forward-compatibility: when multi-output Step 2 lands (requires multi-output
    Step 2 UI + ``step_2_sink_intents: tuple[SinkIntent, ...]`` staging field +
    refactored ``_advance_step_2``), this recipe will auto-wire without restoring
    deleted code.  Tracked: elspeth-obs-74a708e3d7.
    """
    if not (_is_csv(source) and _has_two_json_outputs(sink)):
        return False
    return "blob_ref" in source.options


def _split_threshold_slot_resolver(source: SourceResolved, sink: SinkResolved) -> Mapping[str, Any]:
    """Partial slot map for the split-by-numeric-threshold recipe.

    Provides: ``source_blob_id``, ``above_output_path``, ``below_output_path``.
    User-fillable: ``field``, ``threshold``.

    ``source.options["blob_ref"]`` is the composer-canonical blob UUID;
    same contract as ``_classify_slot_resolver`` — must be present.
    ``sink.outputs[i].options.get("path", default)`` is Tier-3 with a
    rubber-stampable suggested default.

    The ``_split_threshold_predicate`` gate now requires ``blob_ref in source.options``
    before returning True, so this InvariantError is structurally unreachable from
    any call path through ``match_recipe``.  It is kept as defence-in-depth: any
    future caller that bypasses the predicate registry will get a clear crash
    rather than a silent ``KeyError`` or opaque downstream failure.
    """
    if "blob_ref" not in source.options:
        raise InvariantError(
            "Recipe slot resolver requires source.options['blob_ref'] "
            "(set by _execute_set_source_from_blob or handle_step_1_source "
            "blob enrichment path); source options present: "
            f"{sorted(source.options.keys())}"
        )
    blob_ref = source.options["blob_ref"]
    # Tier-3 sink options. As in ``_classify_slot_resolver``, an operator-set
    # ``path`` is used verbatim; its absence is an explicit decision to offer
    # the recipe's default destination as a rubber-stampable pre-fill.
    above_options = sink.outputs[0].options
    if "path" in above_options:
        above_path = above_options["path"]
    else:
        above_path = "outputs/above.jsonl"
    below_options = sink.outputs[1].options
    if "path" in below_options:
        below_path = below_options["path"]
    else:
        below_path = "outputs/below.jsonl"
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
    #
    # web-scrape-llm-rate-jsonl is registered LAST: the URL-row json/csv source
    # never collides with the CSV classify/split predicates (those need
    # classifier-keyword required_fields / two outputs), so order is not load-
    # bearing for disambiguation — it is asserted last to keep registry edits
    # intentional.  The RecipeSpec it matches is registered in recipes.py
    # (``web-scrape-llm-rate-jsonl``), so ``match_recipe`` resolves it end-to-end.
    (_web_scrape_predicate, "web-scrape-llm-rate-jsonl", _web_scrape_slot_resolver),
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
                raise InvariantError(
                    f"Recipe '{recipe_name}' is in _RECIPE_PREDICATES but not registered in recipes.py — invariant violation."
                )
            unsatisfied: dict[str, SlotSpec] = {
                name: spec for name, spec in recipe.slots.items() if spec.required and name not in resolved_slots
            }
            return RecipeMatch(
                recipe_name=recipe_name,
                slots=resolved_slots,
                unsatisfied_slots=unsatisfied,
            )
    return None
