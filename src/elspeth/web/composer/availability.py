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

from elspeth.web.composer.provider_config import (
    PROVIDER_REQUIRED_ENV_KEYS,
    infer_provider_from_model_name,
    infer_provider_from_unprefixed_model_name,
)

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl


@dataclass(frozen=True, slots=True)
class ComposerAvailability:
    """Boot-time availability snapshot for the composer service.

    Correlation invariant (this is our own data — Tier 1): ``available`` is
    true *only* when there is no failure reason and nothing is missing, and
    ``missing_keys`` is non-empty *only* when unavailable. ``compute_availability``
    upholds this by construction; ``__post_init__`` makes any contradictory
    instance from a future construction site crash rather than record a
    self-inconsistent readiness snapshot.
    """

    available: bool
    model: str
    provider: str | None
    reason: str | None = None
    missing_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.available and (self.reason is not None or self.missing_keys):
            raise ValueError(
                "ComposerAvailability(available=True) is incompatible with a "
                f"failure reason or missing_keys (reason={self.reason!r}, "
                f"missing_keys={self.missing_keys!r})."
            )


def compute_availability(service: ComposerServiceImpl) -> ComposerAvailability:
    """Infer whether the configured model has the required env at boot.

    This is a configuration/readiness signal, not a network health check.
    Keep it side-effect-free: LiteLLM provider probing has observable
    startup side effects in web lifespans, while the actual composer call
    path still validates provider requests through LiteLLM.
    """
    provider = infer_provider_from_model_name(service._model) or infer_provider_from_unprefixed_model_name(service._model)
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

    if provider not in PROVIDER_REQUIRED_ENV_KEYS:
        return ComposerAvailability(
            available=False,
            model=service._model,
            provider=provider,
            reason=f"Composer model {service._model} is unavailable: provider {provider!r} has no configured environment contract.",
        )
    required_keys = PROVIDER_REQUIRED_ENV_KEYS[provider]

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
