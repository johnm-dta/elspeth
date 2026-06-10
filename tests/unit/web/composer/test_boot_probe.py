"""Boot probe for operator-set composer sampling config."""

from __future__ import annotations

import httpx
import pytest

import elspeth.web.composer.boot_probe as bp


@pytest.mark.asyncio
async def test_probe_raises_boot_config_error_on_bad_request(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.exceptions import BadRequestError

    async def fake_acompletion(**_kwargs: object) -> object:
        raise BadRequestError(
            message="Invalid value for 'temperature'.",
            model="gpt-5",
            llm_provider="openai",
        )

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(bp.ComposerBootConfigError, match="gpt-5"):
        await bp.probe_composer_config(model="gpt-5", temperature=0.0, seed=None)


@pytest.mark.asyncio
async def test_probe_fatal_on_seed_bad_request_without_phrase_matching(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.exceptions import BadRequestError

    async def fake_acompletion(**_kwargs: object) -> object:
        raise BadRequestError(
            message="Invalid value for 'seed'.",
            model="gpt-5",
            llm_provider="openai",
        )

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(bp.ComposerBootConfigError):
        await bp.probe_composer_config(model="gpt-5", temperature=None, seed=99999999999)


@pytest.mark.asyncio
async def test_probe_passes_through_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    assert await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42) is True


@pytest.mark.asyncio
async def test_probe_is_graceful_on_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_kwargs: object) -> object:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    assert await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42) is False


@pytest.mark.asyncio
async def test_probe_is_graceful_on_litellm_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.exceptions import InternalServerError

    async def fake_acompletion(**_kwargs: object) -> object:
        raise InternalServerError(
            message="Missing credentials.",
            model="gpt-4o",
            llm_provider="openai",
        )

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    assert await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42) is False


@pytest.mark.asyncio
async def test_probe_propagates_programmer_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**_kwargs: object) -> object:
        raise TypeError("signature drift")

    monkeypatch.setattr(bp, "_litellm_acompletion", fake_acompletion)

    with pytest.raises(TypeError, match="signature drift"):
        await bp.probe_composer_config(model="gpt-4o", temperature=0.0, seed=42)
