from __future__ import annotations

from unittest.mock import MagicMock

from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.tools._common import ToolContext
from elspeth.web.composer.tools._shield_availability import azure_prompt_shield_available


class _FakeSecretService:
    def __init__(self, refs: set[str]) -> None:
        self._refs = refs

    def has_ref(self, user_id: str, name: str) -> bool:
        return name in self._refs


def _ctx(secret_service, user_id: str | None) -> ToolContext:
    # ToolContext.catalog (`_common.py:1700`) is REQUIRED with no default —
    # the FIRST frozen field. azure_prompt_shield_available never reads it, but
    # the dataclass constructor demands it, so inject a spec'd mock (CatalogService
    # Protocol lives at catalog/protocol.py:14, NOT catalog/service.py — matching
    # test_tools.py:21).
    return ToolContext(
        catalog=MagicMock(spec=CatalogService),
        secret_service=secret_service,
        user_id=user_id,
    )


def test_shield_available_when_key_configured() -> None:
    ctx = _ctx(_FakeSecretService({"AZURE_CONTENT_SAFETY_KEY"}), "alice")
    assert azure_prompt_shield_available(ctx) is True


def test_shield_unavailable_when_key_missing() -> None:
    ctx = _ctx(_FakeSecretService(set()), "alice")
    assert azure_prompt_shield_available(ctx) is False


def test_shield_undeterminable_defaults_to_false() -> None:
    # No secret service / no user => cannot determine => fail-safe State C.
    assert azure_prompt_shield_available(_ctx(None, "alice")) is False
    assert azure_prompt_shield_available(_ctx(_FakeSecretService({"AZURE_CONTENT_SAFETY_KEY"}), None)) is False
