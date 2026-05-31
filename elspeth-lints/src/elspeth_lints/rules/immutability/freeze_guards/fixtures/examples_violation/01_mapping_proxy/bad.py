from types import MappingProxyType


class Example:
    def __post_init__(self):
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
