from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    data: dict[str, object]
