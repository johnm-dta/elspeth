"""Boot-time availability snapshot and computation for the composer service.

Extracted verbatim from ComposerServiceImpl._compute_availability (service.py)
to reduce the god-class surface. The logic is UNCHANGED; the enclosing
self reference is made explicit via the ``service`` parameter.

``ComposerAvailability`` is re-exported through ``service.py`` so all
existing ``from elspeth.web.composer.service import ComposerAvailability``
imports continue to resolve without change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl


@dataclass(frozen=True, slots=True)
class ComposerAvailability:
    """Boot-time availability snapshot for the composer service."""

    available: bool
    model: str
    provider: str | None
    reason: str | None = None
    missing_keys: tuple[str, ...] = ()


def compute_availability(service: ComposerServiceImpl) -> ComposerAvailability:
    """Infer whether the configured model has the required env at boot.

    This is a configuration/readiness signal, not a network health check.
    Keep it side-effect-free: LiteLLM provider probing has observable
    startup side effects in web lifespans, while the actual composer call
    path still validates provider requests through LiteLLM.
    """
    from elspeth.web.composer.service import (
        _PROVIDER_REQUIRED_ENV_KEYS,
        _infer_provider_from_model_name,
        _infer_provider_from_unprefixed_model_name,
    )

    provider = _infer_provider_from_model_name(service._model) or _infer_provider_from_unprefixed_model_name(service._model)
    if provider is None:
        return ComposerAvailability(
            available=False,
            model=service._model,
            provider=provider,
            reason=(
                f"Composer model {service._model} is unavailable: provider could not be inferred. "
                "Use a provider-prefixed model name or a recognized OpenAI/Anthropic model name."
            ),
        )

    if provider not in _PROVIDER_REQUIRED_ENV_KEYS:
        return ComposerAvailability(
            available=False,
            model=service._model,
            provider=provider,
            reason=f"Composer model {service._model} is unavailable: provider {provider!r} has no configured environment contract.",
        )
    required_keys = _PROVIDER_REQUIRED_ENV_KEYS[provider]

    missing_keys = tuple(key for key in required_keys if key not in os.environ or not os.environ[key])
    if not missing_keys:
        return ComposerAvailability(
            available=True,
            model=service._model,
            provider=provider,
        )

    missing = ", ".join(missing_keys)
    reason = f"Composer model {service._model} is unavailable: missing {missing}."

    return ComposerAvailability(
        available=False,
        model=service._model,
        provider=provider,
        reason=reason,
        missing_keys=missing_keys,
    )
