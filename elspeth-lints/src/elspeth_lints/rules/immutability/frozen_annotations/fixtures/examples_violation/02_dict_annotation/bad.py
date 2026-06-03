from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    mapping: dict[str, int]
