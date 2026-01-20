"""Hook implementation for built-in transform plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinTransforms:
    """Hook implementer for built-in transform plugins."""

    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        """Return built-in transform plugin classes."""
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield
        from elspeth.plugins.transforms.batch_stats import BatchStats
        from elspeth.plugins.transforms.field_mapper import FieldMapper
        from elspeth.plugins.transforms.json_explode import JSONExplode
        from elspeth.plugins.transforms.keyword_filter import KeywordFilter
        from elspeth.plugins.transforms.passthrough import PassThrough

        return [
            PassThrough,
            FieldMapper,
            BatchStats,
            JSONExplode,
            KeywordFilter,
            AzureContentSafety,
            AzurePromptShield,
        ]


# Singleton instance for registration
builtin_transforms = ElspethBuiltinTransforms()
