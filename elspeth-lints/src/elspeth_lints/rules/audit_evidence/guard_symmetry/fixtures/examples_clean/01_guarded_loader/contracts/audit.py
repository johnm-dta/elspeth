from dataclasses import dataclass


@dataclass(frozen=True)
class Widget:
    size: int

    def __post_init__(self):
        if self.size < 0:
            raise ValueError("bad size")
