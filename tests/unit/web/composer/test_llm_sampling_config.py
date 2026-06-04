"""Composer LLM sampling is operator-set and sent verbatim."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import elspeth.web.composer.service as svc
from elspeth.web.catalog.protocol import CatalogService
from elspeth.web.composer.service import ComposerServiceImpl
from elspeth.web.config import WebSettings


def _settings(data_dir: Path, **overrides: Any) -> WebSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    values.update(overrides)
    return WebSettings(**values)


def _service(tmp_path: Path, **settings_overrides: Any) -> ComposerServiceImpl:
    return ComposerServiceImpl(
        catalog=MagicMock(spec=CatalogService),
        settings=_settings(tmp_path, **settings_overrides),
    )


def _response(content: str = "reply") -> Any:
    message = type("Message", (), {"tool_calls": None, "content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


@pytest.mark.asyncio
async def test_call_llm_omits_sampling_when_settings_are_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response()

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)

    await _service(tmp_path)._call_llm([{"role": "user", "content": "hi"}], [])

    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_call_llm_sends_configured_sampling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response()

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)

    await _service(tmp_path, composer_temperature=0.0, composer_seed=42)._call_llm(
        [{"role": "user", "content": "hi"}],
        [],
    )

    assert captured["temperature"] == 0.0
    assert captured["seed"] == 42


@pytest.mark.asyncio
async def test_text_llm_omits_sampling_when_settings_are_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response("text")

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)

    await _service(tmp_path)._call_text_llm([{"role": "user", "content": "hi"}])

    assert "temperature" not in captured
    assert "seed" not in captured


@pytest.mark.asyncio
async def test_advisor_omits_sampling_when_settings_are_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _response("advice")

    monkeypatch.setattr(svc, "_litellm_acompletion", fake_acompletion)

    await _service(tmp_path)._call_advisor_with_audit(
        {
            "trigger": "reactive",
            "problem_summary": "stuck",
            "recent_errors": [],
            "attempted_actions": [],
        },
        recorder=None,
    )

    assert "temperature" not in captured
    assert "seed" not in captured
