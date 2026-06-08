from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    items: list
