"""Azure Content Safety transform for content moderation.

Uses Azure's Content Safety API to analyze text for harmful content:
- Hate speech
- Violence
- Sexual content
- Self-harm

Content is flagged when severity scores exceed configured thresholds.
Uses BaseAzureSafetyTransform for shared batch infrastructure.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms.azure.base import (
    BaseAzureSafetyConfig,
    BaseAzureSafetyTransform,
)
from elspeth.plugins.transforms.azure.errors import MalformedResponseError


class ContentSafetyThresholds(BaseModel):
    """Per-category severity thresholds for Azure Content Safety.

    Azure Content Safety returns severity scores from 0-6 for each category.
    Content is flagged when its severity exceeds the configured threshold.

    A threshold of 0 means all content of that type is blocked.
    A threshold of 6 means only the most severe content is blocked.
    """

    model_config = {"extra": "forbid", "frozen": True}

    hate: int = Field(..., ge=0, le=6, description="Hate content threshold (0-6)")
    violence: int = Field(..., ge=0, le=6, description="Violence content threshold (0-6)")
    sexual: int = Field(..., ge=0, le=6, description="Sexual content threshold (0-6)")
    self_harm: int = Field(..., ge=0, le=6, description="Self-harm content threshold (0-6)")


class AzureContentSafetyConfig(BaseAzureSafetyConfig):
    """Configuration for Azure Content Safety transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        thresholds: Per-category severity thresholds (0-6)
        schema: Schema configuration

    Optional:
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)
        batch_wait_timeout_seconds: Timeout for receiving a completed row result (default 3600)

    Example YAML:
        transforms:
          - plugin: azure_content_safety
            options:
              endpoint: https://my-resource.cognitiveservices.azure.com
              api_key: ${AZURE_CONTENT_SAFETY_KEY}
              fields: [content, title]
              thresholds:
                hate: 2
                violence: 2
                sexual: 2
                self_harm: 0
              on_error: quarantine_sink
              schema:
                mode: observed
    """

    thresholds: ContentSafetyThresholds = Field(
        ...,
        description="Per-category severity thresholds (0-6)",
    )


# Rebuild model to resolve nested model references
AzureContentSafetyConfig.model_rebuild()


# Explicit mapping from Azure Content Safety API category names to internal names.
# This is the ONLY place where Azure category names are translated.
# Unknown categories are REJECTED (fail closed) — see _analyze_content().
_AZURE_CATEGORY_MAP: dict[str, str] = {
    "Hate": "hate",
    "Violence": "violence",
    "Sexual": "sexual",
    "SelfHarm": "self_harm",
}

_EXPECTED_CATEGORIES: frozenset[str] = frozenset(_AZURE_CATEGORY_MAP.values())


class _ContentSafetyProbeRequestBody(TypedDict):
    text: str


class _ContentSafetyProbeCategory(TypedDict):
    category: str
    severity: int


class _ContentSafetyProbeResponseBody(TypedDict):
    categoriesAnalysis: list[_ContentSafetyProbeCategory]


class AzureContentSafety(BaseAzureSafetyTransform):
    """Analyze content using Azure Content Safety API.

    Checks text against Azure's moderation categories (hate, violence,
    sexual, self-harm) and blocks content exceeding configured thresholds.

    Uses BaseAzureSafetyTransform for row-level pipelining: multiple rows
    can be in flight concurrently with FIFO output ordering.
    """

    name = "azure_content_safety"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:6940d98599fd6dcb"
    config_model = AzureContentSafetyConfig
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 forward invariant."""
        return {
            "endpoint": "https://test.cognitiveservices.azure.com",
            "api_key": "test-key",
            "fields": ["content_safety_probe_text"],
            "thresholds": {
                "hate": 6,
                "violence": 6,
                "sexual": 6,
                "self_harm": 6,
            },
            "schema": {"mode": "observed"},
        }

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = AzureContentSafetyConfig.from_dict(config, plugin_name=self.name)
        super().__init__(config, cfg, "AzureContentSafetySchema")
        self._thresholds = cfg.thresholds

    def forward_invariant_probe_rows(self, probe: Any) -> list[Any]:
        """Inject a deterministic safe content field for invariant probing."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="content_safety_probe_text",
                value="safe content",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[Any],
        ctx: Any,
    ) -> TransformResult:
        """Exercise the real single-row validation path with a local fake client."""

        class _ProbeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> _ContentSafetyProbeResponseBody:
                return {
                    "categoriesAnalysis": [
                        {"category": "Hate", "severity": 0},
                        {"category": "Violence", "severity": 0},
                        {"category": "Sexual", "severity": 0},
                        {"category": "SelfHarm", "severity": 0},
                    ]
                }

        class _ProbeClient:
            def post(
                self,
                url: str,
                json: _ContentSafetyProbeRequestBody,
            ) -> _ProbeResponse:
                del url, json
                return _ProbeResponse()

            def close(self) -> None:
                return None

        return self._execute_forward_invariant_probe_with_client(
            probe_rows,
            ctx,
            client=_ProbeClient(),
        )

    def _analyze_field(
        self,
        value: str,
        field_name: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> TransformResult | None:
        """Analyze field via Content Safety API and check thresholds."""
        try:
            analysis = self._analyze_content(value, state_id, token_id=token_id)
        except ValueError as e:
            # Unknown category from Azure — fail CLOSED (security transform).
            # Not retryable: the API response is structurally valid but contains
            # categories we can't assess. Requires code update to handle.
            return TransformResult.error(
                {
                    "reason": "unknown_category",
                    "field": field_name,
                    "message": str(e),
                },
                retryable=False,
            )

        violation = self._check_thresholds(analysis)
        if violation:
            return TransformResult.error(
                {
                    "reason": "content_safety_violation",
                    "field": field_name,
                    "categories": violation,
                },
                retryable=False,
            )
        return None

    def _analyze_content(
        self,
        text: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> dict[str, int]:
        """Call Azure Content Safety API.

        Returns dict with category -> severity mapping.

        Uses AuditedHTTPClient for automatic audit recording and telemetry emission.
        """
        http_client = self._get_http_client(state_id, token_id=token_id)

        url = f"{self._endpoint}/contentsafety/text:analyze?api-version={self.API_VERSION}"

        # Make HTTP call - AuditedHTTPClient records to Landscape and emits telemetry
        response = http_client.post(url, json={"text": text})
        response.raise_for_status()

        # Parse response into category -> severity mapping
        # Azure API responses are external data (Tier 3: Zero Trust) — validate immediately
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            raise MalformedResponseError(f"Invalid JSON in Content Safety response: {e}") from e

        try:
            result: dict[str, int] = dict.fromkeys(_EXPECTED_CATEGORIES, 0)
            seen_categories: set[str] = set()

            for item in data["categoriesAnalysis"]:
                azure_category = item["category"]
                internal_name = _AZURE_CATEGORY_MAP.get(azure_category)
                if internal_name is None:
                    # Fail CLOSED: unknown category means Azure updated their taxonomy.
                    # We cannot assess content safety with unknown categories — reject.
                    raise ValueError(
                        f"Unknown Azure Content Safety category: {azure_category!r}. "
                        f"Known categories: {sorted(_AZURE_CATEGORY_MAP.keys())}. "
                        f"Update _AZURE_CATEGORY_MAP to handle this category."
                    )
                if internal_name in seen_categories:
                    raise MalformedResponseError(
                        f"Duplicate Azure Content Safety category: {azure_category!r} "
                        f"(internal: {internal_name!r}). A malformed response with duplicate "
                        f"categories could downgrade a previously flagged severity."
                    )
                seen_categories.add(internal_name)
                severity = item["severity"]
                if type(severity) is not int or not (0 <= severity <= 6):
                    raise MalformedResponseError(f"severity for {azure_category!r} must be int in [0, 6], got {severity!r}")
                result[internal_name] = severity

            # Fail CLOSED: verify all expected categories were returned by Azure.
            # If Azure changes to only returning flagged categories, absent ones
            # would silently default to 0 (safe) — that's a fail-open path.
            returned_categories = {
                _AZURE_CATEGORY_MAP[item["category"]] for item in data["categoriesAnalysis"] if item["category"] in _AZURE_CATEGORY_MAP
            }
            missing = _EXPECTED_CATEGORIES - returned_categories
            if missing:
                raise MalformedResponseError(
                    f"Azure Content Safety response missing expected categories: "
                    f"{sorted(missing)}. Returned: {sorted(returned_categories)}. "
                    f"Cannot assess content safety without all categories."
                )

            return result

        except (KeyError, TypeError) as e:
            # Malformed response structure — non-retryable
            raise MalformedResponseError(f"Malformed Content Safety response: {e}") from e

    def _check_thresholds(
        self,
        analysis: dict[str, int],
    ) -> dict[str, dict[str, Any]] | None:
        """Check if any category exceeds its threshold.

        Args:
            analysis: Category -> severity mapping from _analyze_content.
                      All 4 categories are guaranteed to be present (defaults applied at boundary).
        """
        t = self._thresholds
        categories: dict[str, dict[str, Any]] = {
            "hate": {"severity": analysis["hate"], "threshold": t.hate, "exceeded": analysis["hate"] > t.hate},
            "violence": {"severity": analysis["violence"], "threshold": t.violence, "exceeded": analysis["violence"] > t.violence},
            "sexual": {"severity": analysis["sexual"], "threshold": t.sexual, "exceeded": analysis["sexual"] > t.sexual},
            "self_harm": {"severity": analysis["self_harm"], "threshold": t.self_harm, "exceeded": analysis["self_harm"] > t.self_harm},
        }

        if any(info["exceeded"] for info in categories.values()):
            return categories
        return None
