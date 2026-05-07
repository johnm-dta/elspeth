"""LLM model catalog reader (L3).

Single source of truth for ``litellm.model_list``. Both the composer
``list_models`` tool and the value-source compliance walker resolve
catalog membership through this module so validate-time and
discovery-time agree on what "a known model" means.

Network-free by construction: ``litellm.model_list`` is the bundled
static catalog populated at import time. Live HTTP polling against
``https://openrouter.ai/api/v1/models`` is deliberately out of scope —
introducing a network dependency to the validate path would re-open
validate/runtime drift the moment the upstream is briefly unreachable.

If ``litellm`` is not installed, the readers return empty.

Registry primitives live in :mod:`elspeth.contracts.value_source` (L0).
This module registers OpenRouter at import time so the walker (L2) can
look up the catalog without importing anything from the plugin layer.
"""

from __future__ import annotations

from functools import lru_cache

from elspeth.contracts.value_source import register_catalog_reader

__all__ = [
    "MODEL_CATALOG_OPENROUTER",
    "OPENROUTER_LITELLM_PREFIX",
    "read_litellm_model_list",
]


MODEL_CATALOG_OPENROUTER = "openrouter"
"""Catalog id for OpenRouter models (``openrouter/...`` prefix in litellm)."""


@lru_cache(maxsize=1)
def read_litellm_model_list() -> tuple[str, ...]:
    """Return the full sorted list of ``litellm.model_list`` entries.

    Single point of truth for ``litellm.model_list`` access. Both the
    composer ``list_models`` tool and the openrouter catalog reader use
    this — they cannot drift on what counts as "what litellm knows."

    Returns an empty tuple on ``ImportError`` (litellm not installed) or
    when ``model_list`` is absent / not a list. The result is sorted so
    callers can rely on a stable order.
    """
    try:
        import litellm
    except ImportError:
        return ()
    raw = getattr(litellm, "model_list", None)
    if not isinstance(raw, list):
        return ()
    return tuple(sorted(str(m) for m in raw if isinstance(m, str)))


OPENROUTER_LITELLM_PREFIX = "openrouter/"


@lru_cache(maxsize=1)
def _read_openrouter_catalog() -> frozenset[str]:
    """Read the OpenRouter slice of ``litellm.model_list`` as raw OpenRouter slugs.

    ``litellm.model_list`` represents OpenRouter entries with an
    ``openrouter/`` routing prefix (e.g. ``openrouter/openai/gpt-4o``)
    so litellm can dispatch the HTTP call to the right provider. ELSPETH's
    :class:`OpenRouterLLMProvider` calls OpenRouter's ``/chat/completions``
    directly via httpx, and OpenRouter's API expects identifiers in
    their native un-prefixed form (e.g. ``openai/gpt-4o``, the same
    string visible on https://openrouter.ai/models).

    The catalog therefore strips the ``openrouter/`` prefix so the value
    set we validate against matches what the HTTP runtime actually
    sends. Filtering and stripping in one step keeps validate-time and
    runtime in lock-step — divergence here would either reject valid
    configs (false negatives) or accept invalid ones that fail at HTTP
    time (the original bug class this contract was meant to close).

    Empty frozenset when the upstream list is empty (e.g. litellm not
    installed); callers translate that into a structured error rather
    than a silent pass.
    """
    prefix = OPENROUTER_LITELLM_PREFIX
    return frozenset(m[len(prefix) :] for m in read_litellm_model_list() if m.startswith(prefix))


register_catalog_reader(
    MODEL_CATALOG_OPENROUTER,
    _read_openrouter_catalog,
    # The catalog is sourced from ``litellm.model_list``. When the
    # walker reports an empty catalog (the operator is running ELSPETH
    # in a deployment that did not install the optional dependency, or
    # litellm's bundled list is empty for this version), this hint is
    # the actionable remediation it surfaces verbatim.
    missing_dep_hint=(
        "install elspeth with the LLM extra (``uv pip install 'elspeth[llm]'``) "
        "to provide the litellm model catalog, or pin a static catalog snapshot"
    ),
)
