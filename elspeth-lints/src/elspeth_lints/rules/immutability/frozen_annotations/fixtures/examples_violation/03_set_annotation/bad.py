from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    unique: set[str]
