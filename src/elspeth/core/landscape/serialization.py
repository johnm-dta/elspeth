"""Audit value serialization helpers for Landscape records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, overload

from elspeth.contracts.errors import AuditIntegrityError


@overload
def serialize_datetime(obj: dict[str, object]) -> dict[str, object]: ...


@overload
def serialize_datetime(obj: Mapping[str, object]) -> dict[str, object]: ...


@overload
def serialize_datetime(obj: list[object]) -> list[object]: ...


@overload
def serialize_datetime(obj: float) -> float: ...


@overload
def serialize_datetime(obj: datetime) -> str: ...


@overload
def serialize_datetime(obj: Any) -> Any: ...


def serialize_datetime(obj: Any) -> Any:
    """Convert audit values to JSON-serializable structures."""
    if isinstance(obj, float):
        if math.isnan(obj):
            raise AuditIntegrityError("NaN values are not allowed in audit data (violates audit integrity)")
        if math.isinf(obj):
            raise AuditIntegrityError("Infinity values are not allowed in audit data (violates audit integrity)")

    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_datetime(item) for item in obj]
    return obj


def dataclass_to_dict(obj: Any) -> Any:
    """Convert a dataclass or nested dataclass container to JSON-compatible data."""
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for dataclass_field in fields(obj):
            field_name = dataclass_field.name
            value = getattr(obj, field_name)
            result[field_name] = dataclass_to_dict(value)
        return result
    if isinstance(obj, Mapping):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    return serialize_datetime(obj)
