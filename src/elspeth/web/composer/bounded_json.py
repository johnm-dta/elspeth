"""Shared resource bounds for untrusted Composer JSON and JSON-like values."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Final

JSON_MAX_DEPTH: Final[int] = 64
JSON_MAX_ITEMS: Final[int] = 10_000
JSON_MAX_STRING_CHARS: Final[int] = 65_536
JSON_MAX_TOTAL_TEXT_CHARS: Final[int] = 1_048_576
JSON_MAX_TOTAL_UTF8_BYTES: Final[int] = 1_048_576

PROVIDER_ARTIFACT_UNAVAILABLE: Final[str] = "<provider-artifact-unavailable>"


class JsonBoundaryError(ValueError):
    """Untrusted JSON exceeded a fixed structural or byte boundary."""


@dataclass(slots=True)
class JsonTraversalBudget:
    """One aggregate budget shared by a complete JSON-like traversal."""

    items: int = 0
    text_chars: int = 0
    utf8_bytes: int = 0

    def check_depth(self, depth: int, *, label: str) -> None:
        if depth > JSON_MAX_DEPTH:
            raise JsonBoundaryError(f"{label} exceeds the {JSON_MAX_DEPTH}-level JSON depth limit")

    def consume_items(self, count: int, *, label: str) -> None:
        if count < 0 or count > JSON_MAX_ITEMS - self.items:
            raise JsonBoundaryError(f"{label} exceeds the {JSON_MAX_ITEMS}-item JSON limit")
        self.items += count

    def consume_text(self, value: str, *, label: str, enforce_string_limit: bool = True) -> None:
        if enforce_string_limit and len(value) > JSON_MAX_STRING_CHARS:
            raise JsonBoundaryError(f"{label} contains a JSON string over {JSON_MAX_STRING_CHARS} characters")
        self.text_chars += len(value)
        if self.text_chars > JSON_MAX_TOTAL_TEXT_CHARS:
            raise JsonBoundaryError(f"{label} exceeds the aggregate JSON text limit")
        try:
            encoded_length = len(value.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise JsonBoundaryError(f"{label} contains text that is not valid UTF-8") from exc
        self.utf8_bytes += encoded_length
        if self.utf8_bytes > JSON_MAX_TOTAL_UTF8_BYTES:
            raise JsonBoundaryError(f"{label} exceeds the aggregate JSON UTF-8 byte limit")


def _preflight_raw_json(raw: str, *, label: str) -> None:
    try:
        raw_bytes = len(raw.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise JsonBoundaryError(f"{label} is not valid UTF-8 text") from exc
    if raw_bytes > JSON_MAX_TOTAL_UTF8_BYTES:
        raise JsonBoundaryError(f"{label} exceeds the {JSON_MAX_TOTAL_UTF8_BYTES}-byte JSON limit")

    # Reject excessive nesting before the recursive C decoder sees it. Braces
    # and brackets inside JSON strings are data, not containers.
    open_containers = 0
    in_string = False
    escaped = False
    for character in raw:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            if open_containers > JSON_MAX_DEPTH:
                raise JsonBoundaryError(f"{label} exceeds the {JSON_MAX_DEPTH}-level JSON depth limit")
            open_containers += 1
        elif character in "]}" and open_containers:
            open_containers -= 1


def _validate_decoded_json(value: Any, *, label: str) -> None:
    budget = JsonTraversalBudget()
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        budget.check_depth(depth, label=label)
        if type(current) is dict:
            budget.consume_items(len(current), label=label)
            for key, child in current.items():
                if type(key) is not str:
                    raise JsonBoundaryError(f"{label} contains a non-string object key")
                budget.consume_text(key, label=label, enforce_string_limit=False)
                stack.append((child, depth + 1))
            continue
        if type(current) is list:
            budget.consume_items(len(current), label=label)
            stack.extend((child, depth + 1) for child in current)
            continue
        if type(current) is str:
            budget.consume_text(current, label=label, enforce_string_limit=False)
            continue
        if current is None or type(current) in {bool, int}:
            continue
        if type(current) is float and math.isfinite(current):
            continue
        raise ValueError(f"{label} contains a value outside strict finite JSON")


def bounded_json_loads(raw: object, *, label: str) -> Any:
    """Decode exact JSON text under fixed byte, depth, item, and text bounds.

    Syntax errors remain ``JSONDecodeError`` so existing caller taxonomies keep
    their precise malformed-JSON class. Resource violations use the closed
    ``JsonBoundaryError`` class.
    """
    if type(raw) is not str:
        raise TypeError(f"{label} must be an exact JSON string")
    _preflight_raw_json(raw, label=label)

    def reject_constant(_value: str) -> Any:
        # Preserve the pre-existing ordinary-dispatch audit taxonomy:
        # canonical non-finite values are ValueError, while depth/item/byte
        # resource violations are JsonBoundaryError.
        raise ValueError(f"{label} contains a non-finite JSON number")

    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except RecursionError as exc:
        raise JsonBoundaryError(f"{label} exceeds the JSON decoder recursion limit") from exc
    _validate_decoded_json(value, label=label)
    return value


def require_bounded_text(value: object, *, label: str) -> str:
    """Validate one exact provider text field without recursive processing."""
    if type(value) is not str:
        raise JsonBoundaryError(f"{label} must be an exact string")
    budget = JsonTraversalBudget()
    budget.consume_text(value, label=label)
    return value
