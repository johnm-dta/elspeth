"""Auto-title uses caller-threaded composer sampling."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

import elspeth.web.sessions._auto_title as at


class _TitleService:
    def __init__(self) -> None:
        self.updates: list[tuple[object, str]] = []

    async def update_session_title(self, session_id: object, title: str) -> None:
        self.updates.append((session_id, title))


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_auto_title_omits_sampling_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion("My Title")

    monkeypatch.setattr(at, "_litellm_acompletion", fake_acompletion)
    service = _TitleService()

    await at.maybe_auto_title_session(
        service=service,
        session_id=uuid4(),
        user_message="Build a CSV pipeline",
        model="gpt-5",
        temperature=None,
        seed=None,
    )

    assert "temperature" not in captured
    assert "seed" not in captured
    assert service.updates[0][1] == "My Title"


@pytest.mark.asyncio
async def test_auto_title_sends_configured_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _completion("My Title")

    monkeypatch.setattr(at, "_litellm_acompletion", fake_acompletion)

    await at.maybe_auto_title_session(
        service=_TitleService(),
        session_id=uuid4(),
        user_message="Build a CSV pipeline",
        model="gpt-4o",
        temperature=0.0,
        seed=42,
    )

    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42
