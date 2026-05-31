from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    name: str
    count: int
