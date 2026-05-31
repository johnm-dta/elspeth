# tests/unit/web/composer/guided/test_recipe_match.py
"""Tests for match_recipe() — deterministic recipe matcher (topology-only)."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.recipe_match import (
    RecipeMatch,
    _classify_slot_resolver,
    _split_threshold_slot_resolver,
    match_recipe,
)
from elspeth.web.composer.guided.state_machine import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_csv_source(blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000001") -> SourceResolved:
    """Return a minimal CSV SourceResolved with blob_ref in options.

    ``blob_ref`` is the composer-canonical key for the blob UUID, written by
    ``_execute_set_source_from_blob`` and by the ``handle_step_1_source``
    blob-enrichment path in steps.py.
    """
    return SourceResolved(
        plugin="csv",
        options={"blob_ref": blob_ref},
        observed_columns=("id", "text"),
        sample_rows=({"id": "1", "text": "hello"},),
    )


def _make_csv_source_no_blob() -> SourceResolved:
    """Return a minimal CSV SourceResolved WITHOUT blob_ref in options.

    Models the legitimate case of a CSV source configured via SchemaForm with
    a direct file path and no blob upload.  Neither recipe predicate should
    match this source; ``match_recipe`` must return None.
    """
    return SourceResolved(
        plugin="csv",
        options={"path": "/data/my_file.csv"},
        observed_columns=("id", "text"),
        sample_rows=({"id": "1", "text": "hello"},),
    )


def _make_json_output(
    required_fields: tuple[str, ...] = (),
    path: str = "outputs/out.jsonl",
) -> SinkOutputResolved:
    return SinkOutputResolved(
        plugin="json",
        options={"path": path, "format": "jsonl"},
        required_fields=required_fields,
        schema_mode="observed",
    )


def _make_single_json_sink(
    required_fields: tuple[str, ...] = (),
    path: str = "outputs/classified.jsonl",
) -> SinkResolved:
    return SinkResolved(outputs=(_make_json_output(required_fields, path),))


def _make_two_json_sink(
    above_path: str = "outputs/above.jsonl",
    below_path: str = "outputs/below.jsonl",
) -> SinkResolved:
    return SinkResolved(
        outputs=(
            _make_json_output(("amount",), above_path),
            _make_json_output(("amount",), below_path),
        )
    )


# ---------------------------------------------------------------------------
# RecipeMatch dataclass
# ---------------------------------------------------------------------------


class TestRecipeMatch:
    def test_recipe_match_slots_are_frozen(self) -> None:
        match = RecipeMatch(
            recipe_name="test-recipe",
            slots={"source_blob_id": "blob-abc"},
            unsatisfied_slots={},
        )
        with pytest.raises((AttributeError, TypeError)):
            match.slots["source_blob_id"] = "mutated"  # type: ignore[index]

    def test_recipe_match_name_is_frozen(self) -> None:
        match = RecipeMatch(recipe_name="test-recipe", slots={}, unsatisfied_slots={})
        with pytest.raises(AttributeError):
            match.recipe_name = "other"  # type: ignore[misc]

    def test_recipe_match_unsatisfied_slots_are_frozen(self) -> None:
        from elspeth.web.composer.recipes import SlotSpec

        match = RecipeMatch(
            recipe_name="test-recipe",
            slots={},
            unsatisfied_slots={
                "model": SlotSpec(slot_type="str", description="LLM model"),
            },
        )
        with pytest.raises((AttributeError, TypeError)):
            match.unsatisfied_slots["model"] = SlotSpec(slot_type="str")  # type: ignore[index]


# ---------------------------------------------------------------------------
# Classify-rows-llm-jsonl matching
# ---------------------------------------------------------------------------


class TestClassifyRecipeMatch:
    def test_csv_to_json_with_llm_intent_matches(self) -> None:
        """CSV + single JSON output with a classifier keyword → classify recipe."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("category",))
        match = match_recipe(source, sink)
        assert match is not None
        assert match.recipe_name == "classify-rows-llm-jsonl"
        # C8 slot-name verification
        assert "source_blob_id" in match.slots
        assert "label_field" in match.slots

    def test_classify_slot_label_field_uses_keyword_match(self) -> None:
        """label_field is the keyword-matching field, not necessarily [0]."""
        source = _make_csv_source(blob_ref="a1b2c3d4-0000-0000-0000-00000000abcd")
        sink = _make_single_json_sink(required_fields=("record_id", "label"))
        match = match_recipe(source, sink)
        assert match is not None
        assert match.slots["label_field"] == "label"

    def test_classify_slot_source_blob_id_populated_from_blob_ref(self) -> None:
        """source_blob_id slot is populated from source.options['blob_ref']."""
        blob_uuid = "a1b2c3d4-0000-0000-0000-000000000099"
        source = _make_csv_source(blob_ref=blob_uuid)
        sink = _make_single_json_sink(required_fields=("tag",))
        match = match_recipe(source, sink)
        assert match is not None
        assert match.slots["source_blob_id"] == blob_uuid

    def test_classify_slot_output_path_from_sink_options(self) -> None:
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("classification",), path="my/custom.jsonl")
        match = match_recipe(source, sink)
        assert match is not None
        assert match.slots["output_path"] == "my/custom.jsonl"

    def test_classify_slot_output_path_default_when_absent(self) -> None:
        """When the sink output has no 'path' option, the default applies."""
        source = _make_csv_source()
        output = SinkOutputResolved(
            plugin="json",
            options={},  # no path key
            required_fields=("category",),
            schema_mode="observed",
        )
        sink = SinkResolved(outputs=(output,))
        match = match_recipe(source, sink)
        assert match is not None
        assert match.slots["output_path"] == "outputs/classified.jsonl"

    def test_classify_does_not_match_without_keyword_field(self) -> None:
        """CSV + single JSON output WITHOUT classifier keywords → no classify match."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("record_id", "score"))
        match = match_recipe(source, sink)
        # No recipe matches (threshold needs two outputs, classify needs keyword).
        assert match is None

    def test_classify_does_not_match_non_csv_source(self) -> None:
        """Non-CSV source → classify does not match even with keyword field."""
        source = SourceResolved(
            plugin="api",
            options={},
            observed_columns=("label",),
            sample_rows=(),
        )
        sink = _make_single_json_sink(required_fields=("label",))
        match = match_recipe(source, sink)
        assert match is None

    def test_classify_does_not_match_non_json_sink(self) -> None:
        """CSV + non-JSON single output → no match."""
        source = _make_csv_source()
        output = SinkOutputResolved(
            plugin="csv",
            options={"path": "out.csv"},
            required_fields=("category",),
            schema_mode="observed",
        )
        sink = SinkResolved(outputs=(output,))
        match = match_recipe(source, sink)
        assert match is None


# ---------------------------------------------------------------------------
# Split-by-numeric-threshold matching
# ---------------------------------------------------------------------------


class TestSplitThresholdRecipeMatch:
    def test_csv_to_two_json_outputs_matches_threshold(self) -> None:
        """CSV + two JSON outputs → split-by-numeric-threshold."""
        source = _make_csv_source()
        sink = _make_two_json_sink()
        match = match_recipe(source, sink)
        assert match is not None
        assert match.recipe_name == "split-by-numeric-threshold"

    def test_split_threshold_slots_use_correct_names(self) -> None:
        """Slot map keys must match _RECIPE2_SLOTS declarations (C8 verification)."""
        blob_uuid = "a1b2c3d4-0000-0000-0000-000000000789"
        source = _make_csv_source(blob_ref=blob_uuid)
        sink = _make_two_json_sink(above_path="my/above.jsonl", below_path="my/below.jsonl")
        match = match_recipe(source, sink)
        assert match is not None
        assert match.recipe_name == "split-by-numeric-threshold"
        assert set(match.slots.keys()) >= {
            "source_blob_id",
            "above_output_path",
            "below_output_path",
        }
        assert match.slots["source_blob_id"] == blob_uuid
        assert match.slots["above_output_path"] == "my/above.jsonl"
        assert match.slots["below_output_path"] == "my/below.jsonl"

    def test_threshold_does_not_match_single_output(self) -> None:
        """Single JSON output → threshold does not match (needs two)."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("amount",))
        match = match_recipe(source, sink)
        # No keyword fields → classify also misses; net result: None.
        assert match is None

    def test_threshold_does_not_match_non_csv_source(self) -> None:
        source = SourceResolved(
            plugin="database",
            options={},
            observed_columns=("value",),
            sample_rows=(),
        )
        sink = _make_two_json_sink()
        match = match_recipe(source, sink)
        assert match is None


# ---------------------------------------------------------------------------
# No-match cases
# ---------------------------------------------------------------------------


class TestNoMatch:
    def test_no_match_returns_none(self) -> None:
        """Sink with three outputs → no registered recipe matches."""
        source = _make_csv_source()
        sink = SinkResolved(
            outputs=(
                _make_json_output((), "a.jsonl"),
                _make_json_output((), "b.jsonl"),
                _make_json_output((), "c.jsonl"),
            )
        )
        match = match_recipe(source, sink)
        assert match is None

    def test_empty_sink_outputs_returns_none(self) -> None:
        source = _make_csv_source()
        sink = SinkResolved(outputs=())
        match = match_recipe(source, sink)
        assert match is None


# ---------------------------------------------------------------------------
# CSV source without blob_ref: predicate returns False → no recipe match
# ---------------------------------------------------------------------------


class TestCsvNoBlobRef:
    """CSV sources without a blob_ref (direct file-path configuration) are
    legitimate user configurations.  Both predicates gate on blob_ref presence
    before returning True, so ``match_recipe`` returns None rather than crashing.
    "No recipe match" is the correct outcome — the guided flow continues to
    manual chain solving via Step 3.

    Asymmetry probes: reverting either predicate's blob_ref check causes these
    tests to fail (the predicate returns True and the slot resolver crashes with
    InvariantError, which leaks up through match_recipe as an unhandled exception).
    """

    def test_classify_returns_none_when_blob_ref_absent(self) -> None:
        """CSV + classifier-keyword sink, but no blob_ref → None (no recipe match)."""
        source = _make_csv_source_no_blob()
        sink = _make_single_json_sink(required_fields=("category",))
        # Predicate gates out on missing blob_ref; resolver never reached.
        assert match_recipe(source, sink) is None

    def test_split_threshold_returns_none_when_blob_ref_absent(self) -> None:
        """CSV + two-JSON sink, but no blob_ref → None (no recipe match)."""
        source = _make_csv_source_no_blob()
        sink = _make_two_json_sink()
        # Predicate gates out on missing blob_ref; resolver never reached.
        assert match_recipe(source, sink) is None

    def test_classify_no_blob_does_not_raise(self) -> None:
        """Confirms the predicate change: no InvariantError leaks from match_recipe
        when blob_ref is absent.  If the predicate's blob_ref check were reverted,
        the resolver would raise InvariantError and this test would fail."""
        source = _make_csv_source_no_blob()
        sink = _make_single_json_sink(required_fields=("label",))
        # Must NOT raise — returns None silently.
        result = match_recipe(source, sink)
        assert result is None

    def test_split_threshold_no_blob_does_not_raise(self) -> None:
        """Same probe for the split-threshold predicate."""
        source = _make_csv_source_no_blob()
        sink = _make_two_json_sink()
        result = match_recipe(source, sink)
        assert result is None


# ---------------------------------------------------------------------------
# Resolver-level defence-in-depth (InvariantError still raised directly)
# ---------------------------------------------------------------------------


class TestResolverDefenceInDepth:
    """The slot resolvers retain their InvariantError guard even though the
    predicates now prevent blob_ref-less sources from ever reaching them through
    ``match_recipe``.  These tests call the resolvers directly to prove the
    defence-in-depth guard is intact — any future caller that bypasses the
    predicate registry will get a clear crash rather than a silent KeyError.
    """

    def test_classify_resolver_crashes_directly_without_blob_ref(self) -> None:
        """_classify_slot_resolver called directly with no blob_ref → InvariantError."""
        source = _make_csv_source_no_blob()
        sink = _make_single_json_sink(required_fields=("category",))
        with pytest.raises(InvariantError, match="blob_ref"):
            _classify_slot_resolver(source, sink)

    def test_split_threshold_resolver_crashes_directly_without_blob_ref(self) -> None:
        """_split_threshold_slot_resolver called directly with no blob_ref → InvariantError."""
        source = _make_csv_source_no_blob()
        sink = _make_two_json_sink()
        with pytest.raises(InvariantError, match="blob_ref"):
            _split_threshold_slot_resolver(source, sink)


# ---------------------------------------------------------------------------
# Unsatisfied required-slot schema (Task 10.0 — Gap 6)
# ---------------------------------------------------------------------------


class TestUnsatisfiedSlots:
    """match_recipe populates RecipeMatch.unsatisfied_slots with the required
    slots the resolver could NOT pre-fill.

    The frontend renders editable inputs for these so the operator can supply
    the values before Apply.  Optional slots with declared defaults are NOT
    surfaced — ``validate_slots`` auto-fills them at apply time.
    """

    def test_classify_unsatisfied_lists_three_required_slots(self) -> None:
        """classify-rows-llm-jsonl: classifier_template, model, api_key_secret."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("category",))
        match = match_recipe(source, sink)
        assert match is not None
        unsatisfied_names = set(match.unsatisfied_slots.keys())
        assert unsatisfied_names == {
            "classifier_template",
            "model",
            "api_key_secret",
        }

    def test_classify_unsatisfied_excludes_resolver_filled_slots(self) -> None:
        """Resolver fills source_blob_id / output_path / label_field — these
        must NOT appear in unsatisfied_slots."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("label",))
        match = match_recipe(source, sink)
        assert match is not None
        for name in ("source_blob_id", "output_path", "label_field"):
            assert name not in match.unsatisfied_slots, f"resolver-filled slot {name!r} leaked into unsatisfied_slots"

    def test_classify_unsatisfied_excludes_optional_slots_with_defaults(self) -> None:
        """Optional slots (provider, required_input_fields) have declared
        defaults and must NOT appear in unsatisfied_slots."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("tag",))
        match = match_recipe(source, sink)
        assert match is not None
        for name in ("provider", "required_input_fields"):
            assert name not in match.unsatisfied_slots, f"optional slot {name!r} surfaced as unsatisfied"

    def test_classify_unsatisfied_carries_slot_specs(self) -> None:
        """Each unsatisfied entry is a SlotSpec with the recipe's declared
        metadata — slot_type, description, required=True."""
        source = _make_csv_source()
        sink = _make_single_json_sink(required_fields=("classification",))
        match = match_recipe(source, sink)
        assert match is not None
        spec = match.unsatisfied_slots["model"]
        assert spec.slot_type == "str"
        assert spec.required is True
        assert spec.description  # non-empty hint

    def test_split_threshold_unsatisfied_lists_field_and_threshold(self) -> None:
        """split-by-numeric-threshold: field, threshold are the only required
        slots the resolver does not pre-fill."""
        source = _make_csv_source()
        sink = _make_two_json_sink()
        match = match_recipe(source, sink)
        assert match is not None
        unsatisfied_names = set(match.unsatisfied_slots.keys())
        assert unsatisfied_names == {"field", "threshold"}

    def test_split_threshold_unsatisfied_threshold_slot_type_is_float(self) -> None:
        source = _make_csv_source()
        sink = _make_two_json_sink()
        match = match_recipe(source, sink)
        assert match is not None
        assert match.unsatisfied_slots["threshold"].slot_type == "float"

    # ------------------------------------------------------------------
    # I4: constructor-level invariants on ``unsatisfied_slots``
    # ------------------------------------------------------------------
    #
    # The match_recipe() construction site enforces the slots / unsatisfied_slots
    # contract by discipline (set-comprehension excludes resolved slots; filters
    # to ``spec.required``).  These tests pin the invariants at the constructor
    # itself so any *other* caller — apply-time reconstruction in routes.py,
    # test fixtures, future code paths — also crashes on contract violation.

    def test_rejects_slot_overlapping_with_unsatisfied(self) -> None:
        """A name present in both ``slots`` and ``unsatisfied_slots`` is a
        direct contradiction of the invariant: a slot cannot simultaneously
        be resolved and unsatisfied.  Constructor must raise InvariantError
        (server bug — only our own code constructs RecipeMatch)."""
        from elspeth.web.composer.recipes import SlotSpec

        with pytest.raises(InvariantError, match="overlap") as exc_info:
            RecipeMatch(
                recipe_name="test-recipe",
                slots={"foo": "resolved-value"},
                unsatisfied_slots={
                    "foo": SlotSpec(slot_type="str", description="conflicting", required=True),
                },
            )
        # Message must name the offending key so the audit trail / log
        # surfaces *which* slot violated the invariant, not just that one did.
        assert "foo" in str(exc_info.value)

    def test_rejects_optional_slot_in_unsatisfied(self) -> None:
        """``unsatisfied_slots`` is the schema for *required* slots the resolver
        could not pre-fill.  Optional slots have declared defaults and are
        auto-filled by ``validate_slots`` at apply time — surfacing them to
        the operator is a contract violation.  Constructor must raise
        InvariantError (server bug — only our own code constructs RecipeMatch)."""
        from elspeth.web.composer.recipes import SlotSpec

        with pytest.raises(InvariantError, match="optional") as exc_info:
            RecipeMatch(
                recipe_name="test-recipe",
                slots={},
                unsatisfied_slots={
                    "foo": SlotSpec(
                        slot_type="str",
                        description="should not be here",
                        required=False,
                    ),
                },
            )
        # Message must name the offending slot for debuggability.
        assert "foo" in str(exc_info.value)

    def test_apply_time_reconstruction_shape_is_trivially_valid(self) -> None:
        """routes.py:~2033 reconstructs a RecipeMatch at apply time with
        ``unsatisfied_slots={}``.  With the constructor invariants in place,
        an empty unsatisfied_slots mapping makes both checks trivially pass
        regardless of the contents of ``slots`` — verify here so a future
        refactor that changes the apply-site shape gets a regression signal."""
        match = RecipeMatch(
            recipe_name="classify-rows-llm-jsonl",
            slots={
                "source_blob_id": "blob-abc",
                "model": "gpt-4o-mini",
                "classifier_template": "Classify: {text}",
                "api_key_secret": "openai/key",
                "output_path": "out.jsonl",
                "label_field": "category",
            },
            unsatisfied_slots={},
        )
        # Constructor did not raise; both invariants vacuously satisfied.
        assert match.unsatisfied_slots == {}
        assert "source_blob_id" in match.slots
