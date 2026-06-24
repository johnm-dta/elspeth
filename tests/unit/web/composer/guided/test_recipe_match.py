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


# ---------------------------------------------------------------------------
# web-scrape-llm-rate-jsonl: canonical-source-plugin pin + predicate (P4.1)
# ---------------------------------------------------------------------------


class TestCanonicalSourcePluginIsResolved:
    """Pin the fact the predicate relies on: an inline_blob URL list
    materialises to a real registered source plugin, never the literal
    string 'inline_blob' and never 'web_scrape'.

    _MIME_TO_SOURCE is the single mapping that resolves a materialised
    inline_blob's MIME type to its concrete source plugin; the predicate
    must key on those resolved names, not on 'inline_blob'.
    """

    def test_mime_to_source_resolves_url_row_plugins(self) -> None:
        from elspeth.web.composer.tools.sources import _MIME_TO_SOURCE

        resolved_plugins = {plugin for plugin, _extra in _MIME_TO_SOURCE.values()}
        # The canonical URL list is JSON rows of {"url": ...}; CSV is the
        # other URL-row carrier. Both are real registered source plugins.
        assert "json" in resolved_plugins
        assert "csv" in resolved_plugins
        # web_scrape is a TRANSFORM, never a materialised source plugin.
        assert "web_scrape" not in resolved_plugins
        assert "inline_blob" not in resolved_plugins

    def test_url_row_source_plugins_constant_matches_resolved_names(self) -> None:
        from elspeth.web.composer.guided.recipe_match import _URL_ROW_SOURCE_PLUGINS

        assert frozenset({"json", "csv"}) == _URL_ROW_SOURCE_PLUGINS
        assert "web_scrape" not in _URL_ROW_SOURCE_PLUGINS


def _make_url_json_source(
    blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000099",
    *,
    with_blob: bool = True,
) -> SourceResolved:
    """A materialised inline_blob URL list: json plugin, url column, blob_ref."""
    options: dict[str, object] = {}
    if with_blob:
        options["blob_ref"] = blob_ref
    return SourceResolved(
        plugin="json",
        options=options,
        observed_columns=("url",),
        sample_rows=({"url": "https://dta.gov.au"},),
    )


def _make_url_csv_source(
    blob_ref: str = "a1b2c3d4-0000-0000-0000-000000000099",
) -> SourceResolved:
    """A materialised inline_blob URL list: csv plugin, url column, blob_ref."""
    return SourceResolved(
        plugin="csv",
        options={"blob_ref": blob_ref},
        observed_columns=("url",),
        sample_rows=({"url": "https://dta.gov.au"},),
    )


def _make_single_jsonl_sink(path: str = "outputs/ratings.jsonl") -> SinkResolved:
    return SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": path, "format": "jsonl"},
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )


def _make_single_json_sink_no_format(path: str = "outputs/ratings.jsonl") -> SinkResolved:
    """A single ``json`` output with NO ``format`` key — the REAL match-time shape.

    At ``match_recipe`` time the resolved json sink's options come from the
    operator's Step-2 SchemaForm submission, where the json sink's ``format``
    defaults to absent (jsonl is auto-detected from a ``.jsonl`` filename only at
    RUNTIME, not in the resolved options). Tests that hand-set ``format: jsonl``
    re-mask the bug this fixture exists to catch; route reachability/canonical
    assertions through THIS helper, not ``_make_single_jsonl_sink``.
    """
    return SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json",
                options={"path": path},  # NO format key — operator default
                required_fields=(),
                schema_mode="observed",
            ),
        )
    )


class TestWebScrapePredicate:
    def test_matches_blob_backed_json_url_source(self) -> None:
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        assert _web_scrape_predicate(_make_url_json_source(), _make_single_jsonl_sink()) is True

    def test_matches_blob_backed_csv_url_source(self) -> None:
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        assert _web_scrape_predicate(_make_url_csv_source(), _make_single_jsonl_sink()) is True

    def test_does_not_reference_web_scrape_as_source(self) -> None:
        """A source whose plugin is literally 'web_scrape' must NOT match.

        web_scrape is a transform, not a source.
        """
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        bad = SourceResolved(
            plugin="web_scrape",
            options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
            observed_columns=("url",),
            sample_rows=({"url": "https://dta.gov.au"},),
        )
        assert _web_scrape_predicate(bad, _make_single_jsonl_sink()) is False

    def test_no_match_without_blob_ref(self) -> None:
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        assert (
            _web_scrape_predicate(
                _make_url_json_source(with_blob=False),
                _make_single_jsonl_sink(),
            )
            is False
        )

    def test_no_match_without_url_column(self) -> None:
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        no_url = SourceResolved(
            plugin="json",
            options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099"},
            observed_columns=("name",),
            sample_rows=({"name": "x"},),
        )
        assert _web_scrape_predicate(no_url, _make_single_jsonl_sink()) is False

    def test_matches_json_output_without_explicit_jsonl_format(self) -> None:
        """A single ``json`` output with NO ``format: jsonl`` key MUST match.

        This is the real match-time shape (the Step-2 SchemaForm default leaves
        ``format`` absent; jsonl is auto-detected from the filename only at
        runtime). The predicate is format-blind — the builder force-sets jsonl
        on its output regardless — so this output matches. Before the E2E-review
        fix the predicate gated on ``format == "jsonl"`` and this returned False,
        which is why the canonical tutorial recipe never fired. Inverted to pin
        the corrected behaviour.
        """
        from elspeth.web.composer.guided.recipe_match import _web_scrape_predicate

        object_array = SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="json",
                    options={"path": "outputs/ratings.json"},  # no format key
                    required_fields=(),
                    schema_mode="observed",
                ),
            )
        )
        assert _web_scrape_predicate(_make_url_json_source(), object_array) is True


def test_web_scrape_predicate_registered_last() -> None:
    """The web-scrape predicate is registered, after the CSV recipes
    (most-specific-first ordering: the URL-row json source never collides
    with the CSV classify/split predicates, but order is asserted to keep
    registry edits intentional)."""
    from elspeth.web.composer.guided.recipe_match import _RECIPE_PREDICATES

    names = [name for _pred, name, _resolver in _RECIPE_PREDICATES]
    assert "web-scrape-llm-rate-jsonl" in names
    assert names[-1] == "web-scrape-llm-rate-jsonl"


# ---------------------------------------------------------------------------
# web-scrape-llm-rate-jsonl: end-to-end match_recipe (P4.2 — RecipeSpec registered)
# ---------------------------------------------------------------------------


def test_match_recipe_returns_web_scrape_match_for_url_source() -> None:
    """End-to-end: now that the RecipeSpec is registered, match_recipe returns
    a RecipeMatch (no InvariantError) for the canonical URL-row source."""
    from elspeth.web.composer.guided.recipe_match import match_recipe

    source = _make_url_json_source()
    sink = _make_single_jsonl_sink()
    result = match_recipe(source, sink)
    assert result is not None
    assert result.recipe_name == "web-scrape-llm-rate-jsonl"
    assert result.slots["source_blob_id"] == source.options["blob_ref"]
    assert result.slots["source_plugin"] == "json"
    # model/api_key_secret remain unsatisfied (operator fills them via recipe_offer).
    assert "model" in result.unsatisfied_slots
    assert "api_key_secret" in result.unsatisfied_slots
    assert "abuse_contact" in result.unsatisfied_slots
    assert "scraping_reason" in result.unsatisfied_slots


def test_match_recipe_returns_web_scrape_match_for_csv_url_source() -> None:
    from elspeth.web.composer.guided.recipe_match import match_recipe

    source = _make_url_csv_source()
    sink = _make_single_jsonl_sink()
    result = match_recipe(source, sink)
    assert result is not None
    assert result.recipe_name == "web-scrape-llm-rate-jsonl"
    assert result.slots["source_blob_id"] == source.options["blob_ref"]
    assert result.slots["source_plugin"] == "csv"


def test_match_recipe_fires_for_real_resolved_shape_without_explicit_format() -> None:
    """REACHABILITY CRUX (E2E review): match_recipe must fire for the (source,
    sink) shape the guided flow ACTUALLY produces at Step 2.5 — a json source
    with ``blob_ref`` + observed ``url`` column, and a single json sink whose
    options carry NO ``format`` key (the operator's Step-2 SchemaForm default).

    This is the test that would have caught the bug: the previous predicate
    gated on ``sink.options.format == "jsonl"``, but at match time the resolved
    json sink's ``format`` is absent (jsonl is auto-detected from the .jsonl
    filename only at RUNTIME), so the canonical tutorial pipeline never matched
    and silently degraded to LLM-driven compose. A ``format: jsonl`` fixture
    would NOT have caught it — that is why this asserts against the no-format
    shape, not ``_make_single_jsonl_sink``.
    """
    from elspeth.web.composer.guided.recipe_match import match_recipe

    source = SourceResolved(
        plugin="json",
        options={"blob_ref": "a1b2c3d4-0000-0000-0000-000000000099", "path": "composer_blobs/canonical-url-list.json"},
        observed_columns=("url",),
        sample_rows=({"url": "https://www.dta.gov.au"},),
    )
    sink = _make_single_json_sink_no_format(path="outputs/ratings.jsonl")
    result = match_recipe(source, sink)
    assert result is not None, "real no-explicit-format match-time shape must fire the web_scrape recipe (zero-LLM canonical)"
    assert result.recipe_name == "web-scrape-llm-rate-jsonl"
    assert result.slots["source_blob_id"] == source.options["blob_ref"]
    assert result.slots["source_plugin"] == "json"
    # The resolver carries the operator's path verbatim; the builder force-sets
    # format:jsonl on the output regardless of the matched sink's (absent) format.
    assert result.slots["output_path"] == "outputs/ratings.jsonl"


def test_canonical_seed_materialised_source_matches_web_scrape_recipe() -> None:
    """§4.1 zero-LLM lever: the REAL canonical tutorial seed, materialised by
    set_pipeline(source.inline_blob), matches the web_scrape recipe.

    Provenance pin — the materialised source shape this test encodes is what
    ``_execute_set_pipeline`` produces for the canonical ``inline_blob`` URL seed
    (``composer/tools/sessions.py``): an ``application/json`` inline blob binds
    the registered ``json`` source plugin via ``_MIME_TO_SOURCE``
    (``composer/tools/sources.py``) AND writes ``source.options["blob_ref"]`` =
    the persisted blob UUID UNCONDITIONALLY in the ``if inline_blob is not None``
    branch. So ``SourceResolved.plugin == "json"`` (never the ``"inline_blob"``
    authoring alias, never ``"web_scrape"``) and ``blob_ref`` IS present at
    ``match_recipe`` time — the predicate's blob-presence gate is satisfied and
    the recipe fires. If this assertion ever flips to None, the zero-LLM
    canonical compose is broken; do NOT relax the predicate — fix the
    materialisation or the fixture so the two agree.

    The sink uses the NO-explicit-format shape (``_make_single_json_sink_no_format``):
    the operator's Step-2 SchemaForm leaves ``format`` absent (jsonl is
    auto-detected from the ``.jsonl`` filename only at runtime), and the
    predicate is format-blind. A hand-set ``format: jsonl`` fixture would
    re-mask the E2E-review bug this test guards against.
    """
    from elspeth.web.composer.guided.recipe_match import match_recipe

    # The materialised canonical source: json plugin + path + blob_ref overlay
    # + observed url column.
    canonical_source = SourceResolved(
        plugin="json",  # _MIME_TO_SOURCE["application/json"] -> "json"
        options={
            "path": "composer_blobs/canonical-url-list.json",
            "blob_ref": "a1b2c3d4-0000-0000-0000-000000000099",
        },
        observed_columns=("url",),
        sample_rows=({"url": "https://www.dta.gov.au"},),
    )
    canonical_sink = _make_single_json_sink_no_format()

    result = match_recipe(canonical_source, canonical_sink)
    assert result is not None, "canonical seed must match the web_scrape recipe (zero-LLM §4.1)"
    assert result.recipe_name == "web-scrape-llm-rate-jsonl"
    # The slot resolver derives source_blob_id from the materialised blob_ref.
    assert result.slots["source_blob_id"] == canonical_source.options["blob_ref"]
    assert result.slots["source_plugin"] == "json"
