"""Credential-free durable attribution for effect-time sink diversions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from elspeth.contracts.hashing import stable_hash
from elspeth.engine._error_hash import compute_error_hash


@dataclass(frozen=True, slots=True)
class DiversionAttribution:
    """Exact hashes needed to reconstruct one diversion after restart."""

    ordinal: int
    reason_hash: str
    error_hash: str

    def as_mapping(self) -> dict[str, object]:
        return {
            "error_hash": self.error_hash,
            "ordinal": self.ordinal,
            "reason_hash": self.reason_hash,
        }


def build_diversion_attribution(*, ordinal: int, reason: str) -> DiversionAttribution:
    """Hash the same structured reason and error text used by audit routing."""
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("diversion attribution ordinal must be a non-negative exact integer")
    if type(reason) is not str or not reason:
        raise ValueError("diversion attribution reason must be a non-empty exact string")
    return DiversionAttribution(
        ordinal=ordinal,
        reason_hash=stable_hash({"diversion_reason": reason}),
        error_hash=compute_error_hash(reason),
    )


def parse_diversion_attribution(
    value: object,
    *,
    diverted_ordinals: Sequence[int],
) -> tuple[DiversionAttribution, ...]:
    """Validate an exact, ordered, one-for-one diverted-member attribution."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("diversion attribution must be an ordered sequence")
    result: list[DiversionAttribution] = []
    for item in value:
        if isinstance(item, DiversionAttribution):
            ordinal = item.ordinal
            reason_hash = item.reason_hash
            error_hash = item.error_hash
        else:
            if not isinstance(item, Mapping) or set(item) != {"error_hash", "ordinal", "reason_hash"}:
                raise ValueError("diversion attribution entries must have the exact closed field set")
            ordinal = item["ordinal"]
            reason_hash = item["reason_hash"]
            error_hash = item["error_hash"]
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("diversion attribution ordinal must be a non-negative exact integer")
        if not _is_lower_hex(reason_hash, length=64):
            raise ValueError("diversion attribution reason_hash must be lowercase 64-character hexadecimal")
        if not _is_lower_hex(error_hash, length=16):
            raise ValueError("diversion attribution error_hash must be lowercase 16-character hexadecimal")
        result.append(
            DiversionAttribution(
                ordinal=ordinal,
                reason_hash=reason_hash,
                error_hash=error_hash,
            )
        )
    expected = tuple(diverted_ordinals)
    if tuple(item.ordinal for item in result) != expected:
        raise ValueError("diversion attribution must cover diverted ordinals exactly and in order")
    return tuple(result)


def _is_lower_hex(value: object, *, length: int) -> bool:
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "DiversionAttribution",
    "build_diversion_attribution",
    "parse_diversion_attribution",
]
