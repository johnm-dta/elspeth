from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Example:
    mapping: Mapping[str, int]
