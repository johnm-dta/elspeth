"""Hook implementation for built-in transform plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinTransforms:
    """Hook implementer for built-in transform plugins."""

    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        """Return built-in transform plugin classes."""
        from elspeth.plugins.transforms.field_mapper import FieldMapper
        from elspeth.plugins.transforms.passthrough import PassThrough

        return [PassThrough, FieldMapper]


# Singleton instance for registration
builtin_transforms = ElspethBuiltinTransforms()
