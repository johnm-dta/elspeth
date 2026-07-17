"""Public accessors for composer tool schema contracts.

The registered tool declarations are the wire authority.  Consumers must
select from :func:`get_tool_definitions` rather than reaching into the
dispatch registry's private lookup tables.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, cast

from elspeth.web.composer.tools._dispatch import get_tool_definitions


def _registered_set_pipeline_schema() -> Mapping[str, Any]:
    for definition in get_tool_definitions():
        if definition.get("name") == "set_pipeline":
            parameters = definition.get("parameters")
            if not isinstance(parameters, dict):  # pragma: no cover - registry integrity guard
                raise RuntimeError("registered set_pipeline parameters must be a JSON-schema object")
            return parameters
    raise RuntimeError("registered set_pipeline tool definition is missing")


def _schema_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{path}: schema node must be an object")
    return cast(Mapping[str, Any], value)


def _schema_sequence(value: object, *, path: str) -> Sequence[object]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise RuntimeError(f"{path}: schema union/required metadata must be an array")
    return cast(Sequence[object], value)


def _resolve_ref(schema: Mapping[str, Any], definitions: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    current = schema
    seen: set[str] = set()
    while "$ref" in current:
        ref = current["$ref"]
        if type(ref) is not str or not ref.startswith("#/$defs/"):
            raise RuntimeError(f"{path}: unsupported model schema reference")
        if ref in seen:
            raise RuntimeError(f"{path}: cyclic model schema reference")
        seen.add(ref)
        name = ref.removeprefix("#/$defs/")
        if name not in definitions:
            raise RuntimeError(f"{path}: unresolved model schema reference {name!r}")
        target = _schema_mapping(definitions[name], path=f"$defs.{name}")
        siblings = {key: value for key, value in current.items() if key != "$ref"}
        current = {**target, **siblings}
    return current


def _schema_branches(
    schema: Mapping[str, Any],
    definitions: Mapping[str, Any],
    *,
    path: str,
) -> tuple[Mapping[str, Any], ...]:
    resolved = _resolve_ref(schema, definitions, path=path)
    union_keywords = tuple(keyword for keyword in ("anyOf", "oneOf") if keyword in resolved)
    if len(union_keywords) > 1:
        raise RuntimeError(f"{path}: schema node cannot combine anyOf and oneOf")
    if union_keywords:
        keyword = union_keywords[0]
        raw_branches = _schema_sequence(resolved[keyword], path=f"{path}.{keyword}")
        branches: list[Mapping[str, Any]] = []
        for index, raw_branch in enumerate(raw_branches):
            branches.extend(
                _schema_branches(
                    _schema_mapping(raw_branch, path=f"{path}.{keyword}[{index}]"),
                    definitions,
                    path=path,
                )
            )
        return tuple(branches)

    raw_type = resolved.get("type")
    if isinstance(raw_type, list):
        if not raw_type or any(type(item) is not str for item in raw_type):
            raise RuntimeError(f"{path}: schema type union must contain exact strings")
        return tuple({**resolved, "type": item} for item in raw_type)
    if raw_type is not None and type(raw_type) is not str:
        raise RuntimeError(f"{path}: schema type must be an exact string or list")
    return (resolved,)


def _required_fields(schema: Mapping[str, Any], *, path: str) -> frozenset[str]:
    raw_required = schema.get("required", ())
    required = _schema_sequence(raw_required, path=f"{path}.required")
    if any(type(item) is not str for item in required):
        raise RuntimeError(f"{path}: required entries must be exact strings")
    return frozenset(cast(Sequence[str], required))


def _branch_type(schema: Mapping[str, Any], *, path: str) -> str | None:
    raw_type = schema.get("type")
    if raw_type is None:
        return None
    if type(raw_type) is not str:
        raise RuntimeError(f"{path}: normalized schema branch must have one exact type")
    return raw_type


def _types_compatible(runtime_type: str | None, advertised_type: str | None) -> bool:
    if advertised_type is None:
        return True
    if runtime_type == advertised_type:
        return True
    return runtime_type == "integer" and advertised_type == "number"


def _assert_branch_compatible(
    runtime: Mapping[str, Any],
    advertised: Mapping[str, Any],
    runtime_definitions: Mapping[str, Any],
    advertised_definitions: Mapping[str, Any],
    *,
    path: str,
) -> None:
    runtime_type = _branch_type(runtime, path=path)
    advertised_type = _branch_type(advertised, path=path)
    if not _types_compatible(runtime_type, advertised_type):
        raise RuntimeError(f"{path}: runtime type {runtime_type!r} is not advertised as {advertised_type!r}")

    runtime_enum = runtime.get("enum")
    advertised_enum = advertised.get("enum")
    if advertised_enum is not None:
        if runtime_enum is None:
            raise RuntimeError(f"{path}: advertised enum is narrower than the runtime model")
        runtime_values = frozenset(_schema_sequence(runtime_enum, path=f"{path}.enum"))
        advertised_values = frozenset(_schema_sequence(advertised_enum, path=f"{path}.enum"))
        if not runtime_values <= advertised_values:
            raise RuntimeError(f"{path}: runtime enum values are missing from the advertised schema")

    if runtime_type == "array":
        runtime_items = runtime.get("items")
        advertised_items = advertised.get("items")
        if runtime_items is not None and advertised_items is None:
            return
        if runtime_items is None and advertised_items is not None:
            raise RuntimeError(f"{path}[]: advertised item schema is narrower than the runtime model")
        if runtime_items is not None and advertised_items is not None:
            _assert_directional_compatibility(
                _schema_mapping(runtime_items, path=f"{path}[]"),
                _schema_mapping(advertised_items, path=f"{path}[]"),
                runtime_definitions,
                advertised_definitions,
                path=f"{path}[]",
            )

    if runtime_type != "object":
        return

    runtime_required = _required_fields(runtime, path=path)
    advertised_required = _required_fields(advertised, path=path)
    extra_advertised_required = advertised_required - runtime_required
    if extra_advertised_required:
        raise RuntimeError(f"{path}: advertised required fields are optional in the runtime model: {sorted(extra_advertised_required)!r}")

    runtime_properties = _schema_mapping(runtime.get("properties", {}), path=f"{path}.properties")
    advertised_properties = _schema_mapping(advertised.get("properties", {}), path=f"{path}.properties")
    for name, runtime_property in runtime_properties.items():
        property_path = f"{path}.{name}"
        if name not in advertised_properties:
            raise RuntimeError(f"{property_path}: typed runtime branch is not explicitly advertised")
        _assert_directional_compatibility(
            _schema_mapping(runtime_property, path=property_path),
            _schema_mapping(advertised_properties[name], path=property_path),
            runtime_definitions,
            advertised_definitions,
            path=property_path,
        )

    runtime_additional = runtime.get("additionalProperties", True)
    advertised_additional = advertised.get("additionalProperties", True)
    if runtime_additional is False:
        return
    if advertised_additional is False:
        raise RuntimeError(f"{path}.*: advertised schema rejects properties accepted by the runtime model")
    if runtime_additional is True:
        if advertised_additional is not True:
            raise RuntimeError(f"{path}.*: advertised value schema is narrower than the runtime model")
        return
    if advertised_additional is True:
        return
    _assert_directional_compatibility(
        _schema_mapping(runtime_additional, path=f"{path}.*"),
        _schema_mapping(advertised_additional, path=f"{path}.*"),
        runtime_definitions,
        advertised_definitions,
        path=f"{path}.*",
    )


def _assert_directional_compatibility(
    runtime_schema: Mapping[str, Any],
    advertised_schema: Mapping[str, Any],
    runtime_definitions: Mapping[str, Any],
    advertised_definitions: Mapping[str, Any],
    *,
    path: str,
) -> None:
    runtime_branches = _schema_branches(runtime_schema, runtime_definitions, path=path)
    advertised_branches = _schema_branches(advertised_schema, advertised_definitions, path=path)
    for runtime_branch in runtime_branches:
        failures: list[str] = []
        for advertised_branch in advertised_branches:
            try:
                _assert_branch_compatible(
                    runtime_branch,
                    advertised_branch,
                    runtime_definitions,
                    advertised_definitions,
                    path=path,
                )
            except RuntimeError as exc:
                failures.append(str(exc))
            else:
                break
        else:
            runtime_type = _branch_type(runtime_branch, path=path)
            detail = failures[0] if failures else "no advertised branches"
            raise RuntimeError(f"{path}: runtime type {runtime_type!r} is not compatibly advertised ({detail})")


def assert_set_pipeline_schema_compatible(*, advertised_schema: Mapping[str, Any] | None = None) -> None:
    """Fail if a runtime-valid typed branch is absent from the tool schema.

    The direction is deliberate: the advertised schema may be looser about
    unknown properties, while Pydantic remains the stricter runtime boundary.
    It must never be narrower in requiredness, nullability, enum membership,
    or any typed source/node/edge/output branch.
    """
    from elspeth.web.composer.redaction import SetPipelineArgumentsModel

    runtime_schema = SetPipelineArgumentsModel.model_json_schema()
    runtime_definitions = _schema_mapping(runtime_schema.get("$defs", {}), path="$defs")
    advertised = _registered_set_pipeline_schema() if advertised_schema is None else advertised_schema
    advertised_definitions = _schema_mapping(advertised.get("$defs", {}), path="$defs")
    _assert_directional_compatibility(
        runtime_schema,
        advertised,
        runtime_definitions,
        advertised_definitions,
        path="$",
    )


def canonical_set_pipeline_schema() -> dict[str, Any]:
    """Return an isolated, runtime-compatible registered schema copy."""
    registered = _registered_set_pipeline_schema()
    assert_set_pipeline_schema_compatible(advertised_schema=registered)
    return deepcopy(cast(dict[str, Any], registered))


__all__ = ["assert_set_pipeline_schema_compatible", "canonical_set_pipeline_schema"]
