from pydantic import BaseModel

from elspeth.contracts.discriminated import DiscriminatedPlugin


def test_plugin_with_discriminated_variants_is_recognised():
    class _AzureCfg(BaseModel):
        provider: str = "azure"

    class _OpenRouterCfg(BaseModel):
        provider: str = "openrouter"

    class MyPlugin:
        @classmethod
        def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
            return ("provider", {"azure": _AzureCfg, "openrouter": _OpenRouterCfg})

    assert isinstance(MyPlugin, DiscriminatedPlugin)


def test_plugin_without_method_is_not_discriminated():
    class MyPlugin:
        pass

    assert not isinstance(MyPlugin, DiscriminatedPlugin)
