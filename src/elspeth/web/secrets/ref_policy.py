"""Secret-ref field placement policy for web-authored pipelines."""

from __future__ import annotations

from collections.abc import Collection

from elspeth.core.secrets import SECRET_FIELD_NAMES, SECRET_FIELD_SUFFIXES, STRUCTURAL_FIELD_EXEMPTIONS

_PLUGIN_SPECIFIC_SECRET_REF_FIELDS: dict[tuple[str, str], frozenset[str]] = {
    ("sink", "database"): frozenset({"url"}),
}


def allowed_secret_ref_fields(component_type: str, plugin_name: str) -> frozenset[str]:
    """Return plugin-specific non-heuristic fields that may carry secret refs."""
    key = (component_type, plugin_name)
    if key in _PLUGIN_SPECIFIC_SECRET_REF_FIELDS:
        return _PLUGIN_SPECIFIC_SECRET_REF_FIELDS[key]
    return frozenset()


def allowed_secret_ref_fields_text(
    component_type: str,
    plugin_name: str,
    *,
    extra_fields: Collection[str] = frozenset(),
) -> str:
    """Human-readable credential field policy for validation errors."""
    exact_names = sorted(SECRET_FIELD_NAMES | allowed_secret_ref_fields(component_type, plugin_name) | frozenset(extra_fields))
    suffixes = ", ".join(SECRET_FIELD_SUFFIXES)
    exemptions = ", ".join(sorted(STRUCTURAL_FIELD_EXEMPTIONS))
    return f"{', '.join(exact_names)} or fields ending in {suffixes} (structural fields {exemptions} excepted)"
