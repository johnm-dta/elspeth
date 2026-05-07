"""Token usage contract for LLM responses.

Replaces loose ``dict[str, int]`` with a frozen dataclass that encodes
``None = unknown`` at the type level.  This eliminates an entire class
of fabrication bugs where ``.get("prompt_tokens", 0)`` silently converts
"provider didn't report" into "zero tokens used."

Trust-tier notes
----------------
* ``known()`` / ``unknown()`` — used by our code (Tier 1/2).
* ``from_dict()`` — the **only** Tier 3 reconstruction path.
  Coerces non-int values to ``None`` so callers never need to
  ``isinstance``-check individual fields again.
* ``to_dict()`` — serialization boundary.  Omits ``None`` keys so
  downstream row storage (still plain dicts) is backward-compatible:
  ``{}`` for fully unknown, ``{"prompt_tokens": 10}`` for partial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from elspeth.contracts.freeze import require_int


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """LLM token usage with explicit unknown semantics.

    Attributes:
        prompt_tokens: Tokens consumed by the prompt, or ``None``
            if the provider did not report this.
        completion_tokens: Tokens generated in the response, or ``None``
            if the provider did not report this.
        reported_total: Provider-reported aggregate total, or ``None``
            if not reported. Preserved when the provider reports only an
            aggregate without a prompt/completion breakdown. Use the
            ``total_tokens`` property for the best available total.
        cached_prompt_tokens: Subset of ``prompt_tokens`` that the provider
            reports as served from its prompt cache (OpenAI / OpenRouter
            ``prompt_tokens_details.cached_tokens`` shape). ``None`` when
            unreported. Always ``<= prompt_tokens`` when both known.
        cache_creation_input_tokens: Anthropic write-side cache token count
            (number of input tokens written into the prompt cache on this
            call). ``None`` when unreported.
        cache_read_input_tokens: Anthropic read-side cache token count
            (number of input tokens served from the prompt cache on this
            call). ``None`` when unreported.
        reasoning_tokens: Provider-reported output tokens used for internal
            reasoning, or ``None`` when unreported. This is counted as output
            usage by providers that expose it, but remains separate audit
            metadata so callers do not infer it from completion totals.

    Note on provider differences: OpenAI/OpenRouter expose a single
    ``cached_tokens`` figure inside ``prompt_tokens_details``; Anthropic
    splits cache hits and misses across two sibling fields. The contract
    captures both shapes verbatim — no normalization that would flatten
    the audit record.
    """

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reported_total: int | None = None
    cached_prompt_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        """Validate token counts are non-negative when known.

        Negative token counts are physically impossible and indicate
        either an API bug or data corruption. Zero is acceptable
        (e.g., cached responses with 0 completion tokens).
        """
        require_int(self.prompt_tokens, "prompt_tokens", optional=True, min_value=0)
        require_int(self.completion_tokens, "completion_tokens", optional=True, min_value=0)
        require_int(self.reported_total, "reported_total", optional=True, min_value=0)
        require_int(self.cached_prompt_tokens, "cached_prompt_tokens", optional=True, min_value=0)
        require_int(
            self.cache_creation_input_tokens,
            "cache_creation_input_tokens",
            optional=True,
            min_value=0,
        )
        require_int(self.cache_read_input_tokens, "cache_read_input_tokens", optional=True, min_value=0)
        require_int(self.reasoning_tokens, "reasoning_tokens", optional=True, min_value=0)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_tokens(self) -> int | None:
        """Best available total: computed sum if both components are known,
        otherwise the provider-reported aggregate, otherwise ``None``.

        Prefers the computed sum because providers occasionally report
        inconsistent totals.
        """
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return self.reported_total

    @property
    def is_known(self) -> bool:
        """``True`` when both token counts were reported by the provider."""
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @property
    def has_data(self) -> bool:
        """``True`` when at least one token count was reported.

        Unlike ``is_known`` (which requires *both* counters), this returns
        ``True`` for partial provider responses that include only one counter
        or an aggregate total. Use this when deciding whether to emit usage
        to telemetry — partial data is still valuable operational signal.
        """
        return (
            self.prompt_tokens is not None
            or self.completion_tokens is not None
            or self.reported_total is not None
            or self.cached_prompt_tokens is not None
            or self.cache_creation_input_tokens is not None
            or self.cache_read_input_tokens is not None
            or self.reasoning_tokens is not None
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, int]:
        """Convert to a plain dict, omitting unknown (``None``) fields.

        Returns ``{}`` for fully unknown, ``{"prompt_tokens": 10}`` for
        partial, or all reported fields when known (``reported_total``
        serialized as ``total_tokens``). Cache fields are emitted under
        their canonical names so a future ``from_dict`` round-trips them
        without re-deriving from provider shape.
        """
        result: dict[str, int] = {}
        if self.prompt_tokens is not None:
            result["prompt_tokens"] = self.prompt_tokens
        if self.completion_tokens is not None:
            result["completion_tokens"] = self.completion_tokens
        if self.reported_total is not None:
            result["total_tokens"] = self.reported_total
        if self.cached_prompt_tokens is not None:
            result["cached_prompt_tokens"] = self.cached_prompt_tokens
        if self.cache_creation_input_tokens is not None:
            result["cache_creation_input_tokens"] = self.cache_creation_input_tokens
        if self.cache_read_input_tokens is not None:
            result["cache_read_input_tokens"] = self.cache_read_input_tokens
        if self.reasoning_tokens is not None:
            result["reasoning_tokens"] = self.reasoning_tokens
        return result

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def unknown(cls) -> TokenUsage:
        """Factory for fully-unknown usage (provider omitted data)."""
        return cls(prompt_tokens=None, completion_tokens=None)

    @classmethod
    def known(cls, prompt_tokens: int, completion_tokens: int) -> TokenUsage:
        """Factory for fully-known usage.

        Args:
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
        """
        return cls(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

    @classmethod
    def from_dict(cls, data: Any) -> TokenUsage:
        """Reconstruct from external (Tier 3) data.

        Coerces non-``int`` values to ``None`` so callers never need to
        validate individual fields.  Handles:
        - Missing keys  → ``None``
        - ``None`` values → ``None``
        - Non-int types (``float``, ``str``, …) → ``None``
        - Empty / non-dict input → fully unknown

        Cache token capture (Tier 3 boundary):
        - OpenAI / OpenRouter: ``cached_tokens`` lives nested at
          ``prompt_tokens_details.cached_tokens`` on the provider response,
          OR can be passed at top level under the canonical
          ``cached_prompt_tokens`` key (round-trip path).
        - Anthropic: ``cache_creation_input_tokens`` /
          ``cache_read_input_tokens`` are sibling fields on the usage object.
        - When neither provider exposes the relevant field, it stays
          ``None`` — a missing cache statistic must NOT be coerced to zero
          (per CLAUDE.md fabrication policy: absence is evidence, not zero).
        - Reasoning-capable providers may expose ``reasoning_tokens`` at top
          level or nested under ``completion_tokens_details`` /
          ``output_tokens_details``. The counter remains ``None`` when absent
          or malformed; it is never derived from completion totals.

        Args:
            data: Raw dict from an LLM API response, or ``None``/non-dict.
        """
        from collections.abc import Mapping

        if not isinstance(data, Mapping):
            return cls.unknown()

        raw_prompt = data.get("prompt_tokens")
        raw_completion = data.get("completion_tokens")
        raw_total = data.get("total_tokens")

        # bool is a subclass of int in Python, so isinstance(True, int) is True.
        # But True/False are not valid token counts — reject them as non-int.
        # Negative values are coerced to None (not passed through to crash
        # in __post_init__) — this is a Tier 3 boundary, so invalid external
        # data is coerced, not propagated.
        prompt = raw_prompt if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool) and raw_prompt >= 0 else None
        completion = (
            raw_completion if isinstance(raw_completion, int) and not isinstance(raw_completion, bool) and raw_completion >= 0 else None
        )
        total = raw_total if isinstance(raw_total, int) and not isinstance(raw_total, bool) and raw_total >= 0 else None

        # Cache field extraction. Try the canonical top-level form first
        # (round-trip path), then fall back to the OpenAI nested shape for
        # the from-provider-response path.
        cached_prompt = _coerce_nonneg_int(data.get("cached_prompt_tokens"))
        if cached_prompt is None:
            details = data.get("prompt_tokens_details")
            if isinstance(details, Mapping):
                cached_prompt = _coerce_nonneg_int(details.get("cached_tokens"))

        cache_creation = _coerce_nonneg_int(data.get("cache_creation_input_tokens"))
        cache_read = _coerce_nonneg_int(data.get("cache_read_input_tokens"))
        reasoning = _coerce_nonneg_int(data.get("reasoning_tokens"))
        if reasoning is None:
            for details_key in ("completion_tokens_details", "output_tokens_details"):
                details = data.get(details_key)
                if isinstance(details, Mapping):
                    reasoning = _coerce_nonneg_int(details.get("reasoning_tokens"))
                    if reasoning is not None:
                        break

        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            reported_total=total,
            cached_prompt_tokens=cached_prompt,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
            reasoning_tokens=reasoning,
        )


def _coerce_nonneg_int(value: Any) -> int | None:
    """Tier-3 helper: coerce arbitrary input to a non-negative int or ``None``.

    Mirrors the per-field coercion used for ``prompt_tokens`` /
    ``completion_tokens`` / ``total_tokens`` so cache fields obey the same
    "absence is evidence, not zero" semantics. ``bool`` is excluded because
    ``isinstance(True, int)`` is ``True`` in Python.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None
