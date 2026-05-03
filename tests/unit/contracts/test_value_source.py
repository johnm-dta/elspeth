"""L0 unit tests for ``elspeth.contracts.value_source``.

Pins the discriminated-union shape and the catalog registry semantics.
"""

from __future__ import annotations

import dataclasses

import pytest

from elspeth.contracts.value_source import (
    CatalogValueSource,
    DerivedFromSiblingValueSource,
    UnknownCatalogIdError,
    ValueSource,
    get_catalog_values,
    list_registered_catalogs,
    register_catalog_reader,
)


class TestCatalogValueSource:
    def test_construct_with_valid_args(self) -> None:
        decl = CatalogValueSource(field_name="model", catalog_id="openrouter")
        assert decl.field_name == "model"
        assert decl.catalog_id == "openrouter"

    def test_empty_field_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="field_name must be non-empty"):
            CatalogValueSource(field_name="", catalog_id="openrouter")

    def test_empty_catalog_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="catalog_id must be non-empty"):
            CatalogValueSource(field_name="model", catalog_id="")

    def test_frozen(self) -> None:
        decl = CatalogValueSource(field_name="model", catalog_id="openrouter")
        with pytest.raises(dataclasses.FrozenInstanceError):
            decl.field_name = "x"  # type: ignore[misc]

    def test_applies_when_default_is_empty_tuple(self) -> None:
        decl = CatalogValueSource(field_name="model", catalog_id="openrouter")
        assert decl.applies_when == ()

    def test_applies_when_accepts_predicate_pairs(self) -> None:
        decl = CatalogValueSource(
            field_name="model",
            catalog_id="openrouter",
            applies_when=(("base_url", "https://openrouter.ai/api/v1"),),
        )
        assert decl.applies_when == (("base_url", "https://openrouter.ai/api/v1"),)

    def test_applies_when_rejects_malformed_entry(self) -> None:
        with pytest.raises(ValueError, match=r"must be .* tuples"):
            CatalogValueSource(
                field_name="model",
                catalog_id="openrouter",
                applies_when=(("base_url",),),  # type: ignore[arg-type]
            )

    def test_applies_when_rejects_empty_sibling_field(self) -> None:
        with pytest.raises(ValueError, match="sibling_field must be non-empty"):
            CatalogValueSource(
                field_name="model",
                catalog_id="openrouter",
                applies_when=(("", "value"),),
            )

    def test_applies_when_rejects_self_reference(self) -> None:
        with pytest.raises(ValueError, match="must differ from field_name"):
            CatalogValueSource(
                field_name="model",
                catalog_id="openrouter",
                applies_when=(("model", "value"),),
            )


class TestDerivedFromSiblingValueSource:
    def test_construct_with_valid_args(self) -> None:
        decl = DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        )
        assert decl.field_name == "model"
        assert decl.sibling_field == "deployment_name"
        assert decl.allow_empty_default is True

    def test_empty_field_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="field_name must be non-empty"):
            DerivedFromSiblingValueSource(
                field_name="",
                sibling_field="deployment_name",
                allow_empty_default=True,
            )

    def test_empty_sibling_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="sibling_field must be non-empty"):
            DerivedFromSiblingValueSource(
                field_name="model",
                sibling_field="",
                allow_empty_default=False,
            )

    def test_field_name_equals_sibling_field_rejected(self) -> None:
        with pytest.raises(ValueError, match="must differ"):
            DerivedFromSiblingValueSource(
                field_name="model",
                sibling_field="model",
                allow_empty_default=True,
            )

    def test_frozen(self) -> None:
        decl = DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            decl.allow_empty_default = False  # type: ignore[misc]


class TestValueSourceUnion:
    def test_union_accepts_both_variants(self) -> None:
        catalog: ValueSource = CatalogValueSource(field_name="model", catalog_id="openrouter")
        derived: ValueSource = DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        )
        # Compile-time check: both are valid ValueSource instances.
        assert isinstance(catalog, CatalogValueSource)
        assert isinstance(derived, DerivedFromSiblingValueSource)


class TestCatalogRegistry:
    """Registry tests use private namespacing to avoid colliding with
    real catalog ids registered by plugin pack imports."""

    def test_unknown_catalog_id_raises(self) -> None:
        with pytest.raises(UnknownCatalogIdError, match="not-a-real-catalog"):
            get_catalog_values("not-a-real-catalog")

    def test_register_and_resolve(self) -> None:
        catalog_id = "test_value_source_register_and_resolve"
        sentinel = frozenset({"alpha", "beta"})

        def reader() -> frozenset[str]:
            return sentinel

        register_catalog_reader(catalog_id, reader)
        try:
            assert get_catalog_values(catalog_id) is sentinel
            assert catalog_id in list_registered_catalogs()
        finally:
            # Best-effort cleanup so test reruns are deterministic.
            from elspeth.contracts.value_source import _CATALOG_READERS

            _CATALOG_READERS.pop(catalog_id, None)

    def test_register_idempotent_with_same_reader(self) -> None:
        catalog_id = "test_value_source_register_idempotent"

        def reader() -> frozenset[str]:
            return frozenset({"x"})

        register_catalog_reader(catalog_id, reader)
        try:
            register_catalog_reader(catalog_id, reader)  # same reader → ok
        finally:
            from elspeth.contracts.value_source import _CATALOG_READERS

            _CATALOG_READERS.pop(catalog_id, None)

    def test_register_different_reader_for_existing_id_rejected(self) -> None:
        catalog_id = "test_value_source_register_different_reader"

        def reader_a() -> frozenset[str]:
            return frozenset({"a"})

        def reader_b() -> frozenset[str]:
            return frozenset({"b"})

        register_catalog_reader(catalog_id, reader_a)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_catalog_reader(catalog_id, reader_b)
        finally:
            from elspeth.contracts.value_source import _CATALOG_READERS

            _CATALOG_READERS.pop(catalog_id, None)

    def test_register_empty_catalog_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="catalog_id must be non-empty"):
            register_catalog_reader("", lambda: frozenset())

    def test_unknown_catalog_id_inherits_keyerror(self) -> None:
        # Existing ``try/except KeyError`` paths must still catch the new exception.
        with pytest.raises(KeyError):
            get_catalog_values("absolutely-not-registered-anywhere")


class TestCatalogMissingDepHint:
    """L0 contract for the optional ``missing_dep_hint`` parameter.

    L3 plugin packs that depend on optional dependencies (e.g. litellm)
    register an actionable string alongside their reader. The walker
    quotes the string verbatim when the catalog is empty so the
    operator sees the specific install command instead of a generic
    "install the optional dependency". L0 stores the hint without
    interpreting it — preserves the contracts-leaf property.
    """

    def test_register_with_hint_stores_hint(self) -> None:
        from elspeth.contracts.value_source import (
            _CATALOG_DEP_HINTS,
            _CATALOG_READERS,
            get_catalog_missing_dep_hint,
        )

        catalog_id = "test_value_source_register_with_hint"

        def reader() -> frozenset[str]:
            return frozenset()

        register_catalog_reader(
            catalog_id,
            reader,
            missing_dep_hint="install elspeth[fakelib]",
        )
        try:
            assert get_catalog_missing_dep_hint(catalog_id) == "install elspeth[fakelib]"
        finally:
            _CATALOG_READERS.pop(catalog_id, None)
            _CATALOG_DEP_HINTS.pop(catalog_id, None)

    def test_register_without_hint_returns_none(self) -> None:
        from elspeth.contracts.value_source import (
            _CATALOG_READERS,
            get_catalog_missing_dep_hint,
        )

        catalog_id = "test_value_source_no_hint"

        def reader() -> frozenset[str]:
            return frozenset()

        register_catalog_reader(catalog_id, reader)
        try:
            assert get_catalog_missing_dep_hint(catalog_id) is None
        finally:
            _CATALOG_READERS.pop(catalog_id, None)

    def test_unknown_catalog_id_returns_none(self) -> None:
        from elspeth.contracts.value_source import get_catalog_missing_dep_hint

        # Hint accessor is non-raising — the walker only cares about the
        # presence/absence of a hint, never about whether the catalog id
        # is registered (that question is answered by ``get_catalog_values``).
        assert get_catalog_missing_dep_hint("never-registered-anywhere") is None

    def test_empty_string_hint_rejected(self) -> None:
        # An empty hint is a programmer bug — accepting it would surface
        # an empty parenthetical to operators (... cannot verify field
        # value () ...) which is worse than the generic fallback.
        with pytest.raises(ValueError, match="missing_dep_hint must be non-empty"):
            register_catalog_reader(
                "test_value_source_empty_hint",
                lambda: frozenset(),
                missing_dep_hint="",
            )
