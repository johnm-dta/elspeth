"""Hook implementation for built-in transform plugins."""

from typing import Any

from elspeth.plugins.hookspecs import hookimpl


class ElspethBuiltinTransforms:
    """Hook implementer for built-in transform plugins."""

    @hookimpl
    def elspeth_get_transforms(self) -> list[type[Any]]:
        """Return built-in transform plugin classes."""
        from elspeth.plugins.llm.azure import AzureLLMTransform
        from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
        from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
        from elspeth.plugins.transforms.batch_replicate import BatchReplicate
        from elspeth.plugins.transforms.batch_stats import BatchStats
        from elspeth.plugins.transforms.field_mapper import FieldMapper
        from elspeth.plugins.transforms.json_explode import JSONExplode
        from elspeth.plugins.transforms.passthrough import PassThrough

        return [
            PassThrough,
            FieldMapper,
            BatchStats,
            JSONExplode,
            BatchReplicate,
            OpenRouterLLMTransform,
            AzureLLMTransform,
            AzureBatchLLMTransform,
        ]


# Singleton instance for registration
builtin_transforms = ElspethBuiltinTransforms()
