from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    items: Sequence[int]
