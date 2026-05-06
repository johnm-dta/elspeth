"""Type-aware bucket helpers for batch transform category values."""

from __future__ import annotations

from collections.abc import Iterable

type ScalarBucketKey = tuple[type[object], object]


def scalar_bucket_key(value: object) -> ScalarBucketKey:
    """Return a hashable key that keeps equal cross-type scalars separate."""
    return (type(value), value)


def same_scalar_bucket_value(left: object, right: object) -> bool:
    """Compare bucket values without merging ``True``/``1`` or ``False``/``0``."""
    return type(left) is type(right) and left == right


def scalar_bucket_contains(values: Iterable[object], candidate: object) -> bool:
    """Return whether ``candidate`` is already present under type-aware equality."""
    return any(same_scalar_bucket_value(value, candidate) for value in values)


def append_unique_bucket_value[T](values: list[T], candidate: T) -> None:
    """Append ``candidate`` only if no type-aware bucket match already exists."""
    if not scalar_bucket_contains(values, candidate):
        values.append(candidate)
