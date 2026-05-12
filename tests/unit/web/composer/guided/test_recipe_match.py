# tests/unit/web/composer/guided/test_recipe_match.py
"""Tests for match_recipe() — deterministic recipe matcher (topology-only)."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.recipe_match import RecipeMatch, match_recipe
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
        )
        with pytest.raises((AttributeError, TypeError)):
            match.slots["source_blob_id"] = "mutated"  # type: ignore[index]

    def test_recipe_match_name_is_frozen(self) -> None:
        match = RecipeMatch(recipe_name="test-recipe", slots={})
        with pytest.raises(AttributeError):
            match.recipe_name = "other"  # type: ignore[misc]


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
# Source blob_ref missing from options (invariant violation — offensive crash)
# ---------------------------------------------------------------------------


class TestMissingBlobRef:
    def test_classify_raises_when_blob_ref_absent(self) -> None:
        """source.options without blob_ref → ValueError (state-machine invariant).

        The slot resolver must only be reached for blob-backed sources.
        When blob_ref is absent it means handle_step_1_source (steps.py)
        did not enrich the options, which is a dispatcher bug — crash loudly
        rather than silently producing an empty source_blob_id that fails
        recipe validation with an opaque error later.
        """
        source = SourceResolved(
            plugin="csv",
            options={},  # no blob_ref
            observed_columns=("text",),
            sample_rows=(),
        )
        sink = _make_single_json_sink(required_fields=("category",))
        with pytest.raises(ValueError, match="blob_ref"):
            match_recipe(source, sink)

    def test_threshold_raises_when_blob_ref_absent(self) -> None:
        """split-by-numeric-threshold resolver also crashes on missing blob_ref."""
        source = SourceResolved(
            plugin="csv",
            options={},  # no blob_ref
            observed_columns=("amount",),
            sample_rows=(),
        )
        sink = _make_two_json_sink()
        with pytest.raises(ValueError, match="blob_ref"):
            match_recipe(source, sink)
