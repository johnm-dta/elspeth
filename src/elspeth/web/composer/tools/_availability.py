"""Composer discovery availability gates."""

from __future__ import annotations

from typing import Any

from elspeth.core.secrets import is_secret_field
from elspeth.web.catalog.schemas import PluginKind, PluginSchemaInfo, PluginSecretRequirement, PluginSummary
from elspeth.web.composer.tools._common import ToolContext


def filter_secret_available_summaries(
    summaries: list[PluginSummary],
    context: ToolContext,
) -> list[PluginSummary]:
    """Hide plugins whose declared secret requirements are unavailable."""

    return [
        summary
        for summary in summaries
        if secret_unavailable_message(
            plugin_type=summary.plugin_type,
            plugin_name=summary.name,
            requirements=_requirements_from_summary(summary),
            context=context,
        )
        is None
    ]


def schema_secret_unavailable_message(
    schema: PluginSchemaInfo,
    context: ToolContext,
) -> str | None:
    return secret_unavailable_message(
        plugin_type=schema.plugin_type,
        plugin_name=schema.name,
        requirements=_requirements_from_schema(schema),
        context=context,
    )


def secret_unavailable_message(
    *,
    plugin_type: PluginKind,
    plugin_name: str,
    requirements: tuple[PluginSecretRequirement, ...],
    context: ToolContext,
) -> str | None:
    if not requirements:
        return None
    if context.secret_service is None or context.user_id is None:
        return None

    missing = tuple(req for req in requirements if not _requirement_satisfied(req, context))
    if not missing:
        return None

    details = "; ".join(_requirement_detail(req) for req in missing)
    return (
        f"{plugin_type} plugin {plugin_name!r} is unavailable because required "
        f"secret reference(s) are not configured: {details}. Call list_secret_refs "
        "and validate_secret_ref to inspect available refs; unavailable plugins "
        "are hidden from composer discovery."
    )


def _requirements_from_summary(summary: PluginSummary) -> tuple[PluginSecretRequirement, ...]:
    if summary.secret_requirements:
        return summary.secret_requirements
    return tuple(
        PluginSecretRequirement(field=field.name) for field in summary.config_fields if field.required and is_secret_field(field.name)
    )


def _requirements_from_schema(schema: PluginSchemaInfo) -> tuple[PluginSecretRequirement, ...]:
    if schema.secret_requirements:
        return schema.secret_requirements
    return _requirements_from_json_schema(schema.json_schema)


def _requirements_from_json_schema(schema: dict[str, Any]) -> tuple[PluginSecretRequirement, ...]:
    if "oneOf" in schema and "$defs" in schema:
        fields = _common_required_secret_fields_from_discriminated(schema)
    else:
        fields = tuple(
            field_name for field_name in schema.get("required", ()) if isinstance(field_name, str) and is_secret_field(field_name)
        )
    return tuple(PluginSecretRequirement(field=field_name) for field_name in fields)


def _common_required_secret_fields_from_discriminated(schema: dict[str, Any]) -> tuple[str, ...]:
    defs = schema["$defs"]
    required_by_variant: list[set[str]] = []
    ordered_first_variant: list[str] = []
    for entry in schema["oneOf"]:
        if not isinstance(entry, dict):
            continue
        raw_ref = entry.get("$ref")
        if not isinstance(raw_ref, str) or not raw_ref.startswith("#/$defs/"):
            continue
        variant = defs[raw_ref[len("#/$defs/") :]]
        raw_required = variant.get("required", ())
        if not isinstance(raw_required, list):
            continue
        fields = {field_name for field_name in raw_required if isinstance(field_name, str) and is_secret_field(field_name)}
        if not required_by_variant:
            ordered_first_variant = [field_name for field_name in raw_required if isinstance(field_name, str) and field_name in fields]
        required_by_variant.append(fields)

    if not required_by_variant:
        return ()
    common = set.intersection(*required_by_variant)
    return tuple(field_name for field_name in ordered_first_variant if field_name in common)


def _requirement_satisfied(
    requirement: PluginSecretRequirement,
    context: ToolContext,
) -> bool:
    if context.secret_service is None or context.user_id is None:
        return True
    if requirement.candidates:
        return any(context.secret_service.has_ref(context.user_id, name) for name in requirement.candidates)
    return _any_available_secret(context)


def _any_available_secret(context: ToolContext) -> bool:
    if context.secret_service is None or context.user_id is None:
        return True
    return any(context.secret_service.has_ref(context.user_id, item.name) for item in context.secret_service.list_refs(context.user_id))


def _requirement_detail(requirement: PluginSecretRequirement) -> str:
    if requirement.candidates:
        candidates = ", ".join(requirement.candidates)
        return f"{requirement.field} (expected one of: {candidates})"
    return requirement.field
