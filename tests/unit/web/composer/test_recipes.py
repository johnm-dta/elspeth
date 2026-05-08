"""Unit tests for ``src/elspeth/web/composer/recipes.py``.

Covers:

  * Slot validation: required vs optional slots, type coercion, error
    messaging that points the model toward the correct repair.
  * Specifically the URL-as-blob_id failure mode (recipe slot rejects a
    URL string with a hint to call create_blob first).
  * The two registered recipes (classify-rows-llm-jsonl,
    split-by-numeric-threshold) — generated set_pipeline args are
    structurally valid and reflect the supplied slots.
  * Registry surface (list_recipes, get_recipe).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from elspeth.web.composer.recipes import (
    RecipeSpec,
    RecipeValidationError,
    SlotSpec,
    apply_recipe,
    get_recipe,
    list_recipes,
)

# --------------------------------------------------------------------------
# Registry surface
# --------------------------------------------------------------------------


class TestRecipeRegistry:
    def test_two_recipes_registered(self) -> None:
        names = {r["name"] for r in list_recipes()}
        assert names == {"classify-rows-llm-jsonl", "split-by-numeric-threshold"}

    def test_get_recipe_by_name(self) -> None:
        spec = get_recipe("classify-rows-llm-jsonl")
        assert spec is not None
        assert spec.name == "classify-rows-llm-jsonl"

    def test_get_recipe_unknown_returns_none(self) -> None:
        assert get_recipe("nonexistent") is None

    def test_list_recipes_includes_slot_metadata(self) -> None:
        for entry in list_recipes():
            assert "slots" in entry
            for _slot_name, slot_meta in entry["slots"].items():
                assert "type" in slot_meta
                assert "required" in slot_meta
                assert "description" in slot_meta


# --------------------------------------------------------------------------
# SlotSpec construction: default value validated against slot_type
# --------------------------------------------------------------------------


class TestSlotSpecDefaultValidation:
    """Recipe authors must declare defaults that satisfy the slot_type contract.

    A typo like ``SlotSpec(slot_type="int", required=False, default="oops")``
    must crash at construction (i.e. at recipe module import) rather than at
    recipe-application time on a path the recipe's tests may not exercise.
    """

    def test_int_default_must_be_int(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'int'"):
            SlotSpec(slot_type="int", required=False, default="oops")

    def test_int_default_int_passes(self) -> None:
        SlotSpec(slot_type="int", required=False, default=42)

    def test_int_default_bool_rejected(self) -> None:
        # bool is a subclass of int in Python — _coerce_slot rejects it.
        with pytest.raises(ValueError, match="does not satisfy slot_type 'int'"):
            SlotSpec(slot_type="int", required=False, default=True)

    def test_float_default_must_be_numeric(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'float'"):
            SlotSpec(slot_type="float", required=False, default=[])

    def test_float_default_int_coerces(self) -> None:
        # _coerce_slot accepts int as a float — mirroring the runtime behaviour.
        SlotSpec(slot_type="float", required=False, default=1)

    def test_str_default_must_be_str(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'str'"):
            SlotSpec(slot_type="str", required=False, default=42)

    def test_blob_id_default_must_be_uuid(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'blob_id'"):
            SlotSpec(slot_type="blob_id", required=False, default="not-a-uuid")

    def test_str_list_default_must_be_iterable_of_str(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'str_list'"):
            SlotSpec(slot_type="str_list", required=False, default="single-string")

    def test_str_list_default_empty_tuple_passes(self) -> None:
        # The recipe registry uses ``default=()`` for required_input_fields —
        # this is the documented "explicit opt-out" path and must remain valid.
        SlotSpec(slot_type="str_list", required=False, default=())

    def test_str_list_default_with_strs_passes(self) -> None:
        SlotSpec(slot_type="str_list", required=False, default=("a", "b"))

    def test_str_list_default_with_non_str_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not satisfy slot_type 'str_list'"):
            SlotSpec(slot_type="str_list", required=False, default=("a", 42))

    def test_required_slot_default_not_validated(self) -> None:
        # Required slots never use ``default``; the validator raises on
        # missing operator input. The default is irrelevant — keeping ``None``
        # as the sentinel for "no default" should not trip the guard.
        SlotSpec(slot_type="int", required=True, default=None)

    def test_optional_slot_default_none_skipped(self) -> None:
        # ``default=None`` is the sentinel for "no operator-visible default";
        # it bypasses validation regardless of slot_type.
        SlotSpec(slot_type="int", required=False, default=None)


# --------------------------------------------------------------------------
# Slot validation: type coercion + error messages
# --------------------------------------------------------------------------


class TestBlobIdSlotValidation:
    """The recipe boundary must reject URL-as-blob_id with a clear repair hint."""

    def test_valid_uuid_passes(self) -> None:
        bid = str(uuid4())
        result = apply_recipe(
            "classify-rows-llm-jsonl",
            {
                "source_blob_id": bid,
                "classifier_template": "Classify: {{ row['text'] }}",
                "model": "anthropic/claude-3.5-sonnet",
                "api_key_secret": "OPENROUTER_API_KEY",
            },
        )
        # blob_id is the top-level set_pipeline source argument (sibling of
        # options), NOT a key within options. set_pipeline forwards it to
        # _resolve_source_blob, which materialises options["path"] and the
        # canonical options["blob_ref"]. Putting blob_id inside options
        # bypasses resolution and leaves the source unbound.
        assert result["source"]["blob_id"] == bid
        assert "blob_id" not in result["source"]["options"]

    def test_url_string_rejected_with_create_blob_hint(self) -> None:
        with pytest.raises(RecipeValidationError) as exc_info:
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": "https://example.com/data.csv",
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                },
            )
        msg = str(exc_info.value)
        assert "valid UUID" in msg
        assert "create_blob" in msg
        assert "mime_type='text/plain'" in msg

    def test_path_string_rejected(self) -> None:
        with pytest.raises(RecipeValidationError, match="valid UUID"):
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": "/tmp/data.csv",
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "model",
                    "api_key_secret": "OPENROUTER_API_KEY",
                },
            )

    def test_int_rejected(self) -> None:
        with pytest.raises(RecipeValidationError, match="UUID string"):
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": 42,
                    "classifier_template": "Classify: {{ row['text'] }}",
                    "model": "model",
                    "api_key_secret": "OPENROUTER_API_KEY",
                },
            )


class TestNumericSlotValidation:
    def test_int_threshold_coerced_to_float(self) -> None:
        result = apply_recipe(
            "split-by-numeric-threshold",
            {
                "source_blob_id": str(uuid4()),
                "field": "price",
                "threshold": 100,
            },
        )
        # threshold is float in the gate condition
        gate_node = next(n for n in result["nodes"] if n["node_type"] == "gate")
        assert "100.0" in gate_node["condition"]

    def test_string_threshold_coerced_to_float(self) -> None:
        result = apply_recipe(
            "split-by-numeric-threshold",
            {
                "source_blob_id": str(uuid4()),
                "field": "price",
                "threshold": "99.95",
            },
        )
        gate_node = next(n for n in result["nodes"] if n["node_type"] == "gate")
        assert "99.95" in gate_node["condition"]

    def test_bool_rejected(self) -> None:
        """bool is a subclass of int in Python — reject explicitly."""
        with pytest.raises(RecipeValidationError, match="bool"):
            apply_recipe(
                "split-by-numeric-threshold",
                {
                    "source_blob_id": str(uuid4()),
                    "field": "price",
                    "threshold": True,
                },
            )

    def test_unparseable_string_rejected(self) -> None:
        with pytest.raises(RecipeValidationError, match="could not coerce"):
            apply_recipe(
                "split-by-numeric-threshold",
                {
                    "source_blob_id": str(uuid4()),
                    "field": "price",
                    "threshold": "not-a-number",
                },
            )


class TestRequiredOptionalSlots:
    def test_missing_required_slot_rejected(self) -> None:
        with pytest.raises(RecipeValidationError, match="missing required slot 'classifier_template'"):
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": str(uuid4()),
                    "model": "model",
                },
            )

    def test_optional_slot_uses_default(self) -> None:
        result = apply_recipe(
            "classify-rows-llm-jsonl",
            {
                "source_blob_id": str(uuid4()),
                "classifier_template": "tmpl",
                "model": "m",
                "api_key_secret": "OPENROUTER_API_KEY",
            },
        )
        # provider defaults to 'openrouter'
        llm_node = next(n for n in result["nodes"] if n["plugin"] == "llm")
        assert llm_node["options"]["provider"] == "openrouter"
        # label_field defaults to 'classification'
        assert llm_node["options"]["response_field"] == "classification"

    def test_missing_required_api_key_secret_rejected(self) -> None:
        """api_key_secret is required — operator must explicitly choose a
        secret reference name (no sensible recipe-level default exists for
        a deployment-specific secret)."""
        with pytest.raises(RecipeValidationError, match="missing required slot 'api_key_secret'"):
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": str(uuid4()),
                    "classifier_template": "tmpl",
                    "model": "m",
                },
            )

    def test_unknown_slot_rejected(self) -> None:
        with pytest.raises(RecipeValidationError, match="does not accept slot"):
            apply_recipe(
                "classify-rows-llm-jsonl",
                {
                    "source_blob_id": str(uuid4()),
                    "classifier_template": "tmpl",
                    "model": "m",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    "fictitious_slot": "x",
                },
            )


# --------------------------------------------------------------------------
# Recipe outputs — structural shape
# --------------------------------------------------------------------------


class TestClassifyRecipe:
    def _apply(self, **slots):
        defaults = {
            "source_blob_id": str(uuid4()),
            "classifier_template": "Classify {{ row['subject'] }}",
            "model": "anthropic/claude-3.5-sonnet",
            "api_key_secret": "OPENROUTER_API_KEY",
        }
        defaults.update(slots)
        return apply_recipe("classify-rows-llm-jsonl", defaults)

    def test_source_uses_blob_id_with_observed_schema(self) -> None:
        result = self._apply()
        src = result["source"]
        assert src["plugin"] == "csv"
        # blob_id is the top-level set_pipeline source argument (sibling of
        # options); set_pipeline forwards it to _resolve_source_blob.
        assert "blob_id" in src
        assert "blob_id" not in src["options"]
        assert src["options"]["schema"] == {"mode": "observed"}
        # discard default per the precursor commit
        assert src["on_validation_failure"] == "discard"

    def test_llm_node_wires_template_and_response_field(self) -> None:
        result = self._apply(label_field="urgency")
        llm = next(n for n in result["nodes"] if n["plugin"] == "llm")
        assert llm["options"]["template"] == "Classify {{ row['subject'] }}"
        assert llm["options"]["response_field"] == "urgency"

    def test_output_is_jsonl(self) -> None:
        result = self._apply(output_path="outputs/tickets.jsonl")
        out = result["outputs"][0]
        assert out["plugin"] == "json"
        assert out["options"]["format"] == "jsonl"
        assert out["options"]["path"] == "outputs/tickets.jsonl"
        assert out["options"]["collision_policy"] == "auto_increment"

    def test_metadata_describes_recipe(self) -> None:
        result = self._apply()
        assert result["metadata"]["name"] == "classify-rows-llm-jsonl"


class TestThresholdRecipe:
    def _apply(self, **slots):
        defaults = {
            "source_blob_id": str(uuid4()),
            "field": "price",
            "threshold": 100.0,
        }
        defaults.update(slots)
        return apply_recipe("split-by-numeric-threshold", defaults)

    def test_type_coerce_node_targets_correct_field(self) -> None:
        result = self._apply(field="amount")
        coerce = next(n for n in result["nodes"] if n["plugin"] == "type_coerce")
        assert coerce["options"]["conversions"] == [{"field": "amount", "to": "float"}]

    def test_gate_condition_uses_field_and_threshold(self) -> None:
        result = self._apply(field="score", threshold=0.75)
        gate = next(n for n in result["nodes"] if n["node_type"] == "gate")
        assert gate["condition"] == "row['score'] >= 0.75"
        assert gate["routes"] == {"true": "above", "false": "below"}

    def test_two_jsonl_outputs(self) -> None:
        result = self._apply(
            above_output_path="outputs/hi.jsonl",
            below_output_path="outputs/lo.jsonl",
        )
        outputs = {o["sink_name"]: o for o in result["outputs"]}
        assert set(outputs) == {"above", "below"}
        assert outputs["above"]["options"]["path"] == "outputs/hi.jsonl"
        assert outputs["below"]["options"]["path"] == "outputs/lo.jsonl"

    def test_chain_order(self) -> None:
        """Pipeline must run type_coerce BEFORE the gate."""
        result = self._apply()
        ids = [n["id"] for n in result["nodes"]]
        assert ids.index("coerce_numeric") < ids.index("threshold_gate")


# --------------------------------------------------------------------------
# Unknown recipe + edge cases
# --------------------------------------------------------------------------


class TestUnknownRecipe:
    def test_apply_unknown_raises(self) -> None:
        with pytest.raises(RecipeValidationError, match="not registered"):
            apply_recipe("imaginary-recipe", {})

    def test_error_lists_available_recipes(self) -> None:
        with pytest.raises(RecipeValidationError) as exc_info:
            apply_recipe("imaginary-recipe", {})
        msg = str(exc_info.value)
        assert "classify-rows-llm-jsonl" in msg
        assert "split-by-numeric-threshold" in msg


# --------------------------------------------------------------------------
# End-to-end: recipe args must flow through set_pipeline so the source is
# blob-bound (options["blob_ref"] populated). compute_proof_diagnostics
# reads options["blob_ref"]; if recipes write blob_id in the wrong location
# it silently bypasses _resolve_source_blob and proof_diagnostics returns
# empty. This is the regression Fix 4 addresses.
# --------------------------------------------------------------------------


class TestRecipeIntegrationWithSetPipeline:
    """Apply a recipe through ``execute_tool('apply_pipeline_recipe', ...)``
    and confirm the resulting source carries the canonical ``blob_ref``
    key — without it, the proof step silently sees no source.
    """

    @pytest.fixture
    def _seeded_blob(self, tmp_path):
        from datetime import UTC, datetime
        from uuid import uuid4

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        session_id = str(uuid4())
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

        blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        storage_path = storage_dir / f"{blob_id}_orders.csv"
        body = b"order_id,customer,price\nO-1,Alice,49.95\n"
        storage_path.write_bytes(body)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=blob_id,
                    session_id=session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(body),
                    content_hash=_content_hash(body),
                    storage_path=str(storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )
        return engine, session_id, blob_id

    def _catalog(self):
        # Use the real PluginManager so set_pipeline's prevalidation can
        # see authentic schemas for csv/llm/type_coerce/json. Mocking
        # list_*/get_schema would force us to fabricate schemas — the
        # real catalog is cheap (builtin registration only) and avoids
        # masking schema mismatches.
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.web.catalog.service import CatalogServiceImpl

        pm = PluginManager()
        pm.register_builtin_plugins()
        return CatalogServiceImpl(pm)

    def test_classify_recipe_blob_id_at_top_level(self) -> None:
        """Recipe places blob_id at source top-level so set_pipeline's
        ``src_args.get('blob_id')`` finds it and invokes _resolve_source_blob.

        Regression test: previously blob_id was nested in source.options,
        which silently bypassed resolution and left the source with
        ``options['blob_id']`` (an unknown key) instead of the canonical
        ``options['blob_ref']``. compute_proof_diagnostics reads blob_ref,
        so the proof step silently saw zero source-bound diagnostics.
        """
        bid = str(uuid4())
        result = apply_recipe(
            "classify-rows-llm-jsonl",
            {
                "source_blob_id": bid,
                "classifier_template": "tmpl",
                "model": "model",
                "api_key_secret": "OPENROUTER_API_KEY",
            },
        )
        # Top-level — set_pipeline consumes this in src_args.get('blob_id').
        assert result["source"]["blob_id"] == bid
        # Must NOT be in options; if it were, set_pipeline would not see
        # it and _resolve_source_blob would not be invoked.
        assert "blob_id" not in result["source"]["options"]

    def test_threshold_recipe_blob_id_at_top_level(self) -> None:
        bid = str(uuid4())
        result = apply_recipe(
            "split-by-numeric-threshold",
            {
                "source_blob_id": bid,
                "field": "price",
                "threshold": 50.0,
            },
        )
        assert result["source"]["blob_id"] == bid
        assert "blob_id" not in result["source"]["options"]

    def test_threshold_source_resolves_to_blob_ref_via_resolve_source_blob(self, _seeded_blob, tmp_path) -> None:
        """End-to-end on the source segment only (the bug surface):
        feeding the recipe's ``source.blob_id`` + ``source.options`` into
        ``_resolve_source_blob`` (the function set_pipeline invokes for
        blob-bound sources) yields ``options['blob_ref']`` and the
        canonical storage path. compute_proof_diagnostics reads blob_ref,
        so this is the load-bearing invariant.

        This narrows scope to the bug surface (key mismatch). The full
        recipe DAG has separate, pre-existing prevalidation gaps for the
        llm/type_coerce transforms (unrelated to Fix 4) which would
        otherwise reject set_pipeline; testing the source segment in
        isolation avoids coupling Fix 4 to those.
        """
        from elspeth.web.composer.state import CompositionState, PipelineMetadata
        from elspeth.web.composer.tools import _resolve_source_blob

        engine, session_id, blob_id = _seeded_blob
        empty = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        # Build the recipe args, then exercise _resolve_source_blob with
        # the exact shape set_pipeline would feed it.
        args = apply_recipe(
            "split-by-numeric-threshold",
            {"source_blob_id": blob_id, "field": "price", "threshold": 50.0},
        )
        src_args = args["source"]

        resolved = _resolve_source_blob(
            blob_id=src_args["blob_id"],
            explicit_plugin=src_args["plugin"],
            caller_options=src_args["options"],
            on_validation_failure=src_args["on_validation_failure"],
            state=empty,
            catalog=self._catalog(),
            session_engine=engine,
            session_id=session_id,
        )

        # _resolve_source_blob returns a _ResolvedSourceBlob on success,
        # or a ToolResult on failure — type-discriminate.
        from elspeth.web.composer.tools import _ResolvedSourceBlob

        assert isinstance(resolved, _ResolvedSourceBlob), getattr(resolved, "data", resolved)
        # Canonical key set by _resolve_source_blob.
        assert resolved.options["blob_ref"] == blob_id
        # Storage path should resolve to the seeded file.
        assert "path" in resolved.options
        # The recipe's caller_options pass through (schema preserved).
        assert resolved.options["schema"] == {"mode": "observed"}


# --------------------------------------------------------------------------
# End-to-end: apply_pipeline_recipe MCP tool invocation drives the full
# chain through set_pipeline's prevalidation, then compute_proof_diagnostics
# reads the resulting state's source.options["blob_ref"]. This is the
# tier-up of Fix 4 (PARTIAL → VERIFIED): the previous coverage stopped at
# _resolve_source_blob in isolation because both recipes generated
# transform options that failed schema prevalidation downstream
# (filigree-obs-2344c737a6). With the recipes now emitting schema-valid
# options for llm and type_coerce, apply_pipeline_recipe can flow through
# set_pipeline successfully and the proof step can be exercised end-to-end.
# --------------------------------------------------------------------------


class TestApplyRecipeEndToEnd:
    """Drive each recipe through ``execute_tool('apply_pipeline_recipe', ...)``
    and confirm (a) the recipe's transform options now satisfy plugin
    schema prevalidation (no more "missing schema/api_key" failures) and
    (b) ``compute_proof_diagnostics`` reads the resulting state's
    ``source.options['blob_ref']`` end-to-end without error.
    """

    @pytest.fixture
    def _seeded(self, tmp_path):
        from datetime import UTC, datetime

        from sqlalchemy.pool import StaticPool

        from elspeth.web.blobs.service import content_hash as _content_hash
        from elspeth.web.sessions.engine import create_session_engine
        from elspeth.web.sessions.models import blobs_table, sessions_table
        from elspeth.web.sessions.schema import initialize_session_schema

        engine = create_session_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        initialize_session_schema(engine)
        session_id = str(uuid4())
        now = datetime.now(UTC)
        with engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=session_id,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Test",
                    created_at=now,
                    updated_at=now,
                )
            )

        blob_id = str(uuid4())
        storage_dir = tmp_path / "blobs" / session_id
        storage_dir.mkdir(parents=True)
        storage_path = storage_dir / f"{blob_id}_orders.csv"
        body = b"order_id,customer,price\nO-1,Alice,49.95\nO-2,Bob,150.00\n"
        storage_path.write_bytes(body)
        with engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=blob_id,
                    session_id=session_id,
                    filename="orders.csv",
                    mime_type="text/csv",
                    size_bytes=len(body),
                    content_hash=_content_hash(body),
                    storage_path=str(storage_path),
                    created_at=now,
                    created_by="user",
                    source_description=None,
                    status="ready",
                )
            )
        return engine, session_id, blob_id

    def _catalog(self):
        # Real PluginManager so set_pipeline's prevalidation sees authentic
        # schemas for csv/llm/type_coerce/json. Mocking the catalog would
        # mask the schema-validity contract under test here.
        from elspeth.plugins.infrastructure.manager import PluginManager
        from elspeth.web.catalog.service import CatalogServiceImpl

        pm = PluginManager()
        pm.register_builtin_plugins()
        return CatalogServiceImpl(pm)

    def test_classify_recipe_passes_set_pipeline_prevalidation_and_drives_proof_step(self, _seeded) -> None:
        """The classify recipe must:

        1. Pass ``_execute_set_pipeline`` prevalidation (the bug surface —
           previously failed with "schema required" / "api_key required"
           on the llm node). Verified by ``result.success``.
        2. Produce a state whose ``source.options['blob_ref']`` matches
           the seeded blob — proves ``_resolve_source_blob`` ran and the
           canonical key was written.
        3. Allow ``compute_proof_diagnostics`` to walk the resulting
           state without raising. The observed-mode blob has no missing
           columns so we expect zero diagnostics, but the call must
           successfully reach the blob read (proving the wiring closed
           the gap that previously made proof_diagnostics return [] by
           never finding ``blob_ref``).
        """
        from elspeth.web.composer.state import CompositionState, PipelineMetadata
        from elspeth.web.composer.tools import compute_proof_diagnostics, execute_tool

        engine, session_id, blob_id = _seeded
        empty = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = execute_tool(
            "apply_pipeline_recipe",
            {
                "recipe_name": "classify-rows-llm-jsonl",
                "slots": {
                    "source_blob_id": blob_id,
                    "classifier_template": "Classify: {{ row['customer'] }}",
                    "model": "anthropic/claude-3.5-sonnet",
                    "api_key_secret": "OPENROUTER_API_KEY",
                    # Declare the field referenced in classifier_template
                    # explicitly. The LLMConfig validator demands an
                    # explicit list when the template references row.*;
                    # supplying [customer] proves the slot drives the
                    # generated config end-to-end.
                    "required_input_fields": ["customer"],
                },
            },
            empty,
            self._catalog(),
            session_engine=engine,
            session_id=session_id,
        )
        # 1. set_pipeline prevalidation passed — this is the bug surface.
        assert result.success, getattr(result, "data", result)

        # 2. Source is now blob-bound.
        new_state = result.updated_state
        assert new_state.source is not None
        assert new_state.source.options["blob_ref"] == blob_id

        # 3. proof_diagnostics walks the state without raising and
        #    reaches the blob read (observable via the diagnostics list
        #    being a list — not None or an exception).
        diagnostics = compute_proof_diagnostics(
            new_state,
            session_engine=engine,
            session_id=session_id,
        )
        assert isinstance(diagnostics, list)
        # Observed schema + headers in the blob: no blocking diagnostics.
        # Any non-info diagnostic would be a real signal worth surfacing.
        assert all(d["severity"] != "blocking" for d in diagnostics), diagnostics

        # The wired secret_ref marker must reach the stored node options
        # — Pydantic prevalidation strips it temporarily, but the marker
        # is preserved in the durable state for runtime resolution.
        classifier = next(n for n in new_state.nodes if n.plugin == "llm")
        assert classifier.options["api_key"] == {"secret_ref": "OPENROUTER_API_KEY"}
        # Schema option flows through prevalidation and is durable.
        assert classifier.options["schema"] == {"mode": "observed"}

    def test_threshold_recipe_passes_set_pipeline_prevalidation_and_drives_proof_step(self, _seeded) -> None:
        """The threshold recipe must:

        1. Pass ``_execute_set_pipeline`` prevalidation (the bug surface —
           previously failed with "schema required" on the type_coerce node).
        2. Produce a state whose ``source.options['blob_ref']`` matches
           the seeded blob.
        3. Drive ``compute_proof_diagnostics`` end-to-end through the
           full apply_pipeline_recipe → set_pipeline → state mutation
           chain rather than the previous in-isolation
           ``_resolve_source_blob`` call.
        """
        from elspeth.web.composer.state import CompositionState, PipelineMetadata
        from elspeth.web.composer.tools import compute_proof_diagnostics, execute_tool

        engine, session_id, blob_id = _seeded
        empty = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = execute_tool(
            "apply_pipeline_recipe",
            {
                "recipe_name": "split-by-numeric-threshold",
                "slots": {
                    "source_blob_id": blob_id,
                    "field": "price",
                    "threshold": 100.0,
                },
            },
            empty,
            self._catalog(),
            session_engine=engine,
            session_id=session_id,
        )
        assert result.success, getattr(result, "data", result)

        new_state = result.updated_state
        assert new_state.source is not None
        assert new_state.source.options["blob_ref"] == blob_id

        diagnostics = compute_proof_diagnostics(
            new_state,
            session_engine=engine,
            session_id=session_id,
        )
        assert isinstance(diagnostics, list)
        assert all(d["severity"] != "blocking" for d in diagnostics), diagnostics

        # type_coerce now carries the required schema option through
        # prevalidation into the durable state.
        coerce = next(n for n in new_state.nodes if n.plugin == "type_coerce")
        assert coerce.options["schema"] == {"mode": "observed"}
        assert coerce.options["conversions"] == ({"field": "price", "to": "float"},) or coerce.options["conversions"] == [
            {"field": "price", "to": "float"}
        ]

    def test_apply_recipe_over_populated_state_emits_replacement_note(self, _seeded) -> None:
        """When the operator has hand-iterated state and the LLM applies a
        recipe, the destructive full-state replacement must be visible.

        Regression for the silent-overwrite issue: ``apply_pipeline_recipe``
        delegates to ``set_pipeline`` (full-state replacement). If the
        prior state had a source / nodes / outputs, the LLM (and the
        audit trail) should see a non-blocking note describing what was
        replaced. This preserves the recipe-on-existing-state ergonomics
        — we don't refuse the destructive call — but the action is no
        longer silent.
        """
        from elspeth.web.composer.state import (
            CompositionState,
            NodeSpec,
            OutputSpec,
            PipelineMetadata,
            SourceSpec,
        )
        from elspeth.web.composer.tools import execute_tool

        engine, session_id, blob_id = _seeded

        # Prior state: a hand-iterated pipeline the operator was building
        # (source set, two transforms, one output). Applying a recipe on
        # top of this is the silent-overwrite scenario.
        populated = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="hand_built_a",
                options={"path": "old-input.csv"},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="hand_built_a",
                    node_type="transform",
                    plugin="select_columns",
                    input="source",
                    on_success="hand_built_b",
                    on_error=None,
                    options={"fields": ["a", "b"]},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
                NodeSpec(
                    id="hand_built_b",
                    node_type="transform",
                    plugin="select_columns",
                    input="hand_built_a",
                    on_success="prior_output",
                    on_error=None,
                    options={"fields": ["a"]},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="prior_output",
                    plugin="json",
                    options={"path": "old-output.json"},
                    on_write_failure="quarantine",
                ),
            ),
            metadata=PipelineMetadata(),
            version=4,
        )

        result = execute_tool(
            "apply_pipeline_recipe",
            {
                "recipe_name": "split-by-numeric-threshold",
                "slots": {
                    "source_blob_id": blob_id,
                    "field": "price",
                    "threshold": 100.0,
                },
            },
            populated,
            self._catalog(),
            session_engine=engine,
            session_id=session_id,
        )

        assert result.success, getattr(result, "data", result)
        # The destructive replacement is now audible: data carries a
        # ``replaced_pipeline_note`` describing the prior counts. ToolResult
        # post-init deep-freezes the data field, so the type at this point
        # is a MappingProxyType wrapping the merged dict — accept any
        # ``Mapping`` rather than asserting ``dict``.
        from collections.abc import Mapping as _Mapping

        assert isinstance(result.data, _Mapping)
        note = result.data["replaced_pipeline_note"]
        assert "source=set" in note
        assert "2 node(s)" in note
        assert "1 output(s)" in note
        # Pre-existing data payload (from set_pipeline's source_blob
        # resolution) survives the merge — we did not clobber it.
        assert "source_blob" in result.data

    def test_apply_recipe_over_empty_state_emits_no_replacement_note(self, _seeded) -> None:
        """Recipe applied to a fresh session is not destructive — no note."""
        from elspeth.web.composer.state import CompositionState, PipelineMetadata
        from elspeth.web.composer.tools import execute_tool

        engine, session_id, blob_id = _seeded
        empty = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = execute_tool(
            "apply_pipeline_recipe",
            {
                "recipe_name": "split-by-numeric-threshold",
                "slots": {
                    "source_blob_id": blob_id,
                    "field": "price",
                    "threshold": 100.0,
                },
            },
            empty,
            self._catalog(),
            session_engine=engine,
            session_id=session_id,
        )
        assert result.success, getattr(result, "data", result)
        # No replacement note when there was nothing to replace.
        if result.data is not None:
            assert "replaced_pipeline_note" not in result.data


# --------------------------------------------------------------------------
# Deep-freeze invariants on RecipeSpec / SlotSpec
#
# RecipeSpec.__post_init__ calls ``freeze_fields(self, "slots")`` so the
# slot mapping cannot be mutated after construction. These tests pin that
# contract so a future refactor that swaps freeze_fields for a forbidden
# pattern (e.g. ``MappingProxyType(self.slots)`` — a CLAUDE.md-banned
# shallow wrap that leaves the original dict mutable through the wrapped
# reference) would fail loudly here rather than silently regress recipe
# integrity. Mirrors the pattern in
# tests/unit/web/composer/test_source_inspection.py::TestFrozenInvariants.
# --------------------------------------------------------------------------


class TestRecipeSpecFrozenInvariants:
    """RecipeSpec.slots must be deeply immutable post-construction.

    See ``deep_freeze`` (src/elspeth/contracts/freeze.py): ``Mapping``
    inputs are converted to a fresh ``MappingProxyType`` over a freshly
    materialised dict — both the wrapper and the underlying dict are
    detached from the caller's input, so external mutation of the input
    cannot leak through to ``recipe.slots``. SlotSpec values themselves
    are frozen dataclasses (``frozen=True, slots=True``).
    """

    def test_slots_item_assignment_new_key_raises_typeerror(self) -> None:
        """Item assignment on a registered recipe's slots is rejected.

        ``MappingProxyType`` raises ``TypeError`` on ``__setitem__``; this
        is the load-bearing barrier that prevents mutation of a recipe's
        slot table after construction.
        """
        spec = get_recipe("classify-rows-llm-jsonl")
        assert spec is not None
        with pytest.raises(TypeError):
            spec.slots["new_slot"] = SlotSpec(slot_type="str")  # type: ignore[index]

    def test_slots_item_assignment_existing_key_raises_typeerror(self) -> None:
        """Reassigning an existing slot is also rejected.

        Distinct from the new-key case: a hypothetical regression that
        wrapped ``self.slots`` in a ``MappingProxyType`` view (without
        the deep-freeze detach) would still raise ``TypeError`` on the
        proxy, but the underlying dict would remain mutable through the
        original reference. This test pins the proxy-level barrier.
        """
        spec = get_recipe("split-by-numeric-threshold")
        assert spec is not None
        # 'field' is a known slot of the threshold recipe.
        assert "field" in spec.slots
        with pytest.raises(TypeError):
            spec.slots["field"] = SlotSpec(slot_type="str")  # type: ignore[index]

    def test_external_dict_mutation_does_not_leak_into_slots(self) -> None:
        """Mutating the dict passed into ``RecipeSpec`` must not affect ``recipe.slots``.

        This is the discriminating regression test against a forbidden
        ``MappingProxyType(self.slots)`` shallow-wrap. ``deep_freeze``
        materialises a fresh dict from the input's items before wrapping,
        so post-construction mutations to the original dict are isolated.
        A shallow-wrap regression with a ``dict`` input would expose the
        leak — the proxy would be a *view* onto the still-mutable original.
        """
        external: dict[str, SlotSpec] = {"k": SlotSpec(slot_type="str")}
        recipe = RecipeSpec(
            name="leak-probe",
            description="test fixture for external-mutation isolation",
            slots=external,
            build=lambda _slots: {},
        )
        external["leaked"] = SlotSpec(slot_type="str")
        assert "leaked" not in recipe.slots
        # The original key survives.
        assert "k" in recipe.slots

    def test_slots_remain_read_only_when_input_is_already_frozen(self) -> None:
        """Even when the input is already a ``MappingProxyType``, the
        resulting ``recipe.slots`` is itself a read-only mapping that
        rejects item assignment.

        ``deep_freeze`` always returns a *fresh* ``MappingProxyType`` for
        Mapping inputs (it detaches the proxy from any view onto a
        possibly-mutable backing dict), so we don't assert identity here
        — just that the read-only contract holds regardless of input
        shape.
        """
        from types import MappingProxyType

        already_frozen = MappingProxyType({"k": SlotSpec(slot_type="str")})
        recipe = RecipeSpec(
            name="frozen-input-probe",
            description="test fixture for already-frozen-input idempotency",
            slots=already_frozen,
            build=lambda _slots: {},
        )
        assert isinstance(recipe.slots, MappingProxyType)
        with pytest.raises(TypeError):
            recipe.slots["new"] = SlotSpec(slot_type="str")  # type: ignore[index]

    def test_slot_spec_is_itself_frozen(self) -> None:
        """``SlotSpec`` is ``frozen=True`` so attribute reassignment must raise.

        If a future refactor drops ``frozen=True`` on SlotSpec, the deep
        freeze on ``RecipeSpec.slots`` would still block dict-level
        mutation but slot fields could be reassigned in place — undermining
        the recipe-table integrity the freeze guard is meant to protect.
        """
        from dataclasses import FrozenInstanceError

        slot = SlotSpec(slot_type="str", required=True, description="probe")
        with pytest.raises(FrozenInstanceError):
            slot.required = False  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            slot.slot_type = "int"  # type: ignore[misc]
