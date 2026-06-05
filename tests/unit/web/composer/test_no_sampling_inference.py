"""Sampling must come from operator config, not per-model inference."""

from __future__ import annotations

import elspeth.web.composer.service as svc


def test_inference_helpers_and_hardcoded_sampling_constants_are_gone() -> None:
    assert not hasattr(svc, "_composer_llm_seed_for_model")
    assert not hasattr(svc, "_litellm_completion_supports_param")
    assert not hasattr(svc, "_COMPOSER_LLM_TEMPERATURE")
    assert not hasattr(svc, "_COMPOSER_LLM_SEED")
