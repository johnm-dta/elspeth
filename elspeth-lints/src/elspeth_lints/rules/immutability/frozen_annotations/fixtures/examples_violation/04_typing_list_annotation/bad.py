from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Example:
    items: List[int]
