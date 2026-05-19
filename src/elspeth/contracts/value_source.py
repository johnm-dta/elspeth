"""Plugin-config value-source declarations (L0).

Plugin authors can declare that a config field's *value* must conform to a
named protocol (catalog membership, sibling-derivation, etc.) beyond what
the type system can express. The L0 leaf carries the declaration shape;
L3 catalog readers and validators consume it.

Two variants:

* :class:`CatalogValueSource` — value must appear in a registered catalog
  (e.g. an LLM model identifier present in the published OpenRouter
  catalog). Catalog ID is opaque at L0; an L3 registry resolves it to a
  reader at validation time.
* :class:`DerivedFromSiblingValueSource` — value must equal a named
  sibling field (e.g. Azure's ``model`` field is derived from
  ``deployment_name``). When ``allow_empty_default`` is ``True``, an empty
  value is also accepted (a Pydantic ``model_validator`` is expected to
  fill it from the sibling at config-construction time).

Both variants are frozen, slot-bearing, scalar-only dataclasses — no
mutable container fields, so no freeze guard is required (per CLAUDE.md
"Scalar-Only Fields Need No Guard").

This module is L0 (contracts layer): no upward imports from core, engine,
or plugins. It is enforced as a leaf by the ``trust_tier.tier_model`` rule.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

__all__ = [
    "CatalogValueSource",
    "DerivedFromSiblingValueSource",
    "UnknownCatalogIdError",
    "ValueSource",
    "find_value_source_config",
    "get_catalog_missing_dep_hint",
    "get_catalog_values",
    "list_registered_catalogs",
    "register_catalog_reader",
    "register_value_source_plugin",
]


class UnknownCatalogIdError(KeyError):
    """Raised when a ``catalog_id`` is not registered.

    Distinct from :class:`KeyError` so callers can disambiguate
    "no catalog by that name" from incidental KeyErrors. Inherits from
    :class:`KeyError` so existing ``try/except KeyError`` paths still work.
    """


_CatalogReader = Callable[[], frozenset[str]]
_CATALOG_READERS: dict[str, _CatalogReader] = {}

# L3 plugin packs that depend on optional dependencies (e.g. ``litellm``)
# can register an actionable hint alongside their catalog reader. The
# walker (L2) reads the hint when the catalog is empty and includes it
# in the finding's ``reason`` so an operator sees ``install elspeth[llm]``
# instead of ``install the optional dependency``. Keeping the hint as
# an opaque string preserves L0's leaf property — L0 does not interpret
# the hint, only stores and surfaces it.
_CATALOG_DEP_HINTS: dict[str, str] = {}


def register_catalog_reader(
    catalog_id: str,
    reader: _CatalogReader,
    *,
    missing_dep_hint: str | None = None,
) -> None:
    """Register a catalog reader under ``catalog_id``.

    Idempotent re-registration with the same reader function is allowed
    (supports module-reimport during testing); registering a different
    reader for an existing id raises :class:`ValueError` so silent
    overrides cannot occur. L3 plugin packs call this at module-import
    time; the walker (L2) consumes the registry via
    :func:`get_catalog_values`.

    ``missing_dep_hint`` is an optional, operator-actionable string the
    walker quotes verbatim when the catalog is empty (e.g. the optional
    dependency is not installed). Providing a hint lets the registrar
    name the package and install command without leaking that knowledge
    into L2/L0. Callers that omit the hint get a generic message.
    """
    if not catalog_id:
        raise ValueError("catalog_id must be non-empty")
    if catalog_id in _CATALOG_READERS:
        existing = _CATALOG_READERS[catalog_id]
        if existing is not reader:
            raise ValueError(f"Catalog reader already registered for {catalog_id!r}")
    if missing_dep_hint is not None and not missing_dep_hint:
        raise ValueError("missing_dep_hint must be non-empty when provided")
    _CATALOG_READERS[catalog_id] = reader
    if missing_dep_hint is not None:
        _CATALOG_DEP_HINTS[catalog_id] = missing_dep_hint


def list_registered_catalogs() -> tuple[str, ...]:
    """Return the sorted tuple of registered catalog ids."""
    return tuple(sorted(_CATALOG_READERS))


def get_catalog_missing_dep_hint(catalog_id: str) -> str | None:
    """Return the registered missing-dep hint for ``catalog_id`` or ``None``.

    The walker calls this when the catalog is empty so the empty-catalog
    finding can quote the registrar's actionable string verbatim. Returns
    ``None`` when no hint was registered — callers fall back to a generic
    message.

    Uses ``in``/direct-access (not ``.get()``) to match the project's
    forbidden-defensive-programming idiom — see CLAUDE.md "Defensive
    Programming: Forbidden". Absence is a legitimate signal here ("no
    hint registered"), not a bug to hide.
    """
    if catalog_id not in _CATALOG_DEP_HINTS:
        return None
    return _CATALOG_DEP_HINTS[catalog_id]


# --- Plugin opt-in registry ─────────────────────────────────────────
#
# Plugins that participate in value-source compliance register their
# class with the attribute name that exposes the typed Pydantic config.
# The walker (L2) consults this registry instead of duck-typing —
# avoids defensive ``getattr``/``hasattr``/``isinstance`` patterns and
# makes opt-in explicit at the plugin pack's import site.

_VALUE_SOURCE_CONFIG_ATTRS: dict[type, str] = {}


def register_value_source_plugin(plugin_cls: type, *, config_attr: str) -> None:
    """Register that ``plugin_cls`` instances expose a typed config at ``config_attr``.

    L3 plugin packs call this at module-import time. The walker then
    uses :func:`find_value_source_config` to fetch the typed config for
    each transform — only opted-in plugins are inspected.

    Re-registering the same ``(class, attr)`` pair is idempotent;
    re-registering with a different attribute raises :class:`ValueError`
    so silent overrides cannot occur.
    """
    if not config_attr:
        raise ValueError("config_attr must be non-empty")
    if plugin_cls in _VALUE_SOURCE_CONFIG_ATTRS:
        existing = _VALUE_SOURCE_CONFIG_ATTRS[plugin_cls]
        if existing != config_attr:
            raise ValueError(
                f"Plugin {plugin_cls.__qualname__} already registered with config_attr "
                f"{existing!r}; cannot re-register with {config_attr!r}"
            )
    _VALUE_SOURCE_CONFIG_ATTRS[plugin_cls] = config_attr


def find_value_source_config(plugin: object) -> object | None:
    """Return the typed config for ``plugin`` if it opts into compliance.

    Walks ``type(plugin).__mro__`` so subclasses inherit registration.
    Returns ``None`` for plugins that did not register — the explicit
    "no value-source declarations on this plugin" signal.
    """
    for cls in type(plugin).__mro__:
        if cls in _VALUE_SOURCE_CONFIG_ATTRS:
            attr_name = _VALUE_SOURCE_CONFIG_ATTRS[cls]
            result: object = plugin.__getattribute__(attr_name)
            return result
    return None


def get_catalog_values(catalog_id: str) -> frozenset[str]:
    """Return the value set for ``catalog_id``.

    Raises :class:`UnknownCatalogIdError` when no reader is registered.
    Readers may return an empty frozenset (e.g. when the upstream
    dependency is not installed). Callers MUST translate empty-catalog
    into a structured error — never a silent pass.
    """
    try:
        reader = _CATALOG_READERS[catalog_id]
    except KeyError as exc:
        registered = list_registered_catalogs()
        raise UnknownCatalogIdError(f"No catalog reader registered for {catalog_id!r}; registered: {registered}") from exc
    return reader()


@dataclass(frozen=True, slots=True)
class CatalogValueSource:
    """Field value MUST appear in a registered catalog.

    The ``catalog_id`` is an opaque name; an L3 catalog registry resolves
    it to a reader at validation time. Keeping the resolver out of L0
    preserves the leaf property of the contracts layer.

    The optional ``applies_when`` predicate makes catalog membership
    *conditional* on sibling field values. Use it when the catalog's
    relevance depends on another config field — for example, OpenRouter's
    model catalog is only authoritative when ``base_url`` targets the
    canonical OpenRouter endpoint; if an operator overrides ``base_url``
    to a chaos test server (or any non-canonical endpoint), the model
    identifier semantics are owned by that endpoint, not by litellm's
    OpenRouter slug list. Encoding the conditional in the contract keeps
    the walker free of provider-specific knowledge.

    ``applies_when`` is a tuple of ``(sibling_field, expected_value)`` pairs.
    The check applies only when EVERY pair matches the config. An empty
    tuple (the default) means "always applies."
    """

    field_name: str
    catalog_id: str
    applies_when: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("CatalogValueSource.field_name must be non-empty")
        if not self.catalog_id:
            raise ValueError("CatalogValueSource.catalog_id must be non-empty")
        for entry in self.applies_when:
            # Length-only structural check (no isinstance — let Python's
            # natural unpacking failures handle non-iterable entries).
            # The type annotation already pins the expected shape; this
            # guard only catches the common typo of passing a 1-tuple or
            # 3-tuple by mistake.
            if len(entry) != 2:
                raise ValueError(f"CatalogValueSource.applies_when entries must be (sibling_field, expected_value) tuples; got {entry!r}")
            sibling_field, _expected = entry
            if not sibling_field:
                raise ValueError("CatalogValueSource.applies_when sibling_field must be non-empty")
            if sibling_field == self.field_name:
                raise ValueError(
                    f"CatalogValueSource.applies_when sibling_field {sibling_field!r} must differ from field_name {self.field_name!r}"
                )


@dataclass(frozen=True, slots=True)
class DerivedFromSiblingValueSource:
    """Field value MUST equal a named sibling field.

    ``allow_empty_default=True`` declares that an empty value (``""`` /
    ``None``) is also acceptable — the runtime config is expected to
    substitute the sibling at construction time (e.g. Azure's
    ``_set_model_from_deployment`` ``model_validator``).
    """

    field_name: str
    sibling_field: str
    allow_empty_default: bool

    def __post_init__(self) -> None:
        if not self.field_name:
            raise ValueError("DerivedFromSiblingValueSource.field_name must be non-empty")
        if not self.sibling_field:
            raise ValueError("DerivedFromSiblingValueSource.sibling_field must be non-empty")
        if self.field_name == self.sibling_field:
            raise ValueError("field_name and sibling_field must differ")


ValueSource = CatalogValueSource | DerivedFromSiblingValueSource
"""Discriminated union of value-source variants.

Walkers dispatch with ``isinstance`` on the concrete variant type. The
union deliberately does not include a generic "any value" branch — a
field without a declared protocol simply omits ``VALUE_SOURCES`` from
its config class.
"""
