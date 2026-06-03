from dataclasses import dataclass

from elspeth.contracts.freeze import freeze_fields


@dataclass(frozen=True)
class Example:
    data: dict[str, object]

    def __post_init__(self):
        freeze_fields(self, "data")
