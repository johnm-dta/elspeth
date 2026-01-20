"""Azure Content Safety transform for content moderation.

This module provides the AzureContentSafety transform which uses Azure's
Content Safety API to analyze text for harmful content categories:
- Hate speech
- Violence
- Sexual content
- Self-harm

Content is flagged when severity scores exceed configured thresholds.
"""

from typing import Any

import httpx
from pydantic import BaseModel, Field

from elspeth.contracts import Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.pooling import PoolConfig
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class ContentSafetyThresholds(BaseModel):
    """Per-category severity thresholds for Azure Content Safety.

    Azure Content Safety returns severity scores from 0-6 for each category.
    Content is flagged when its severity exceeds the configured threshold.

    A threshold of 0 means all content of that type is blocked.
    A threshold of 6 means only the most severe content is blocked.
    """

    hate: int = Field(..., ge=0, le=6, description="Hate content threshold (0-6)")
    violence: int = Field(..., ge=0, le=6, description="Violence content threshold (0-6)")
    sexual: int = Field(..., ge=0, le=6, description="Sexual content threshold (0-6)")
    self_harm: int = Field(..., ge=0, le=6, description="Self-harm content threshold (0-6)")


class AzureContentSafetyConfig(TransformDataConfig):
    """Configuration for Azure Content Safety transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        thresholds: Per-category severity thresholds (0-6)
        schema: Schema configuration

    Optional:
        pool_size: Number of concurrent API calls (1=sequential, >1=pooled)
        min_dispatch_delay_ms: Minimum AIMD dispatch delay (default 0)
        max_dispatch_delay_ms: Maximum AIMD backoff delay (default 5000)
        backoff_multiplier: Backoff multiplier on capacity error (default 2.0)
        recovery_step_ms: Recovery step in milliseconds (default 50)
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)

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
                fields: dynamic
    """

    endpoint: str = Field(..., description="Azure Content Safety endpoint URL")
    api_key: str = Field(..., description="Azure Content Safety API key")
    fields: str | list[str] = Field(
        ...,
        description="Field name(s) to analyze, or 'all' for all string fields",
    )
    thresholds: ContentSafetyThresholds = Field(
        ...,
        description="Per-category severity thresholds (0-6)",
    )

    # Pool configuration fields
    pool_size: int = Field(1, ge=1, description="Number of concurrent API calls (1=sequential)")
    min_dispatch_delay_ms: int = Field(0, ge=0, description="Minimum dispatch delay in milliseconds")
    max_dispatch_delay_ms: int = Field(5000, ge=0, description="Maximum dispatch delay in milliseconds")
    backoff_multiplier: float = Field(2.0, gt=1.0, description="Backoff multiplier on capacity error")
    recovery_step_ms: int = Field(50, ge=0, description="Recovery step in milliseconds")
    max_capacity_retry_seconds: int = Field(3600, gt=0, description="Max seconds to retry capacity errors")

    @property
    def pool_config(self) -> PoolConfig | None:
        """Get pool configuration if pooling is enabled.

        Returns None if pool_size <= 1 (sequential mode).
        """
        if self.pool_size <= 1:
            return None
        return PoolConfig(
            pool_size=self.pool_size,
            min_dispatch_delay_ms=self.min_dispatch_delay_ms,
            max_dispatch_delay_ms=self.max_dispatch_delay_ms,
            backoff_multiplier=self.backoff_multiplier,
            recovery_step_ms=self.recovery_step_ms,
            max_capacity_retry_seconds=self.max_capacity_retry_seconds,
        )


class AzureContentSafety(BaseTransform):
    """Analyze content using Azure Content Safety API.

    Checks text against Azure's moderation categories (hate, violence,
    sexual, self-harm) and blocks content exceeding configured thresholds.
    """

    name = "azure_content_safety"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    is_batch_aware = False
    creates_tokens = False

    API_VERSION = "2024-09-01"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = AzureContentSafetyConfig.from_dict(config)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._fields = cfg.fields
        self._thresholds = cfg.thresholds
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzureContentSafetySchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Create own HTTP client (following OpenRouter pattern)
        self._http_client: httpx.Client | None = None

    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Analyze row content against Azure Content Safety."""
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            if field_name not in row:
                continue  # Skip fields not present in this row

            value = row[field_name]
            if not isinstance(value, str):
                continue

            # Call Azure API
            try:
                analysis = self._analyze_content(value, ctx)
            except httpx.HTTPStatusError as e:
                is_rate_limit = e.response.status_code == 429
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "rate_limited" if is_rate_limit else "http_error",
                        "status_code": e.response.status_code,
                        "message": str(e),
                        "retryable": is_rate_limit,
                    },
                    retryable=is_rate_limit,
                )
            except httpx.RequestError as e:
                return TransformResult.error(
                    {
                        "reason": "api_error",
                        "error_type": "network_error",
                        "message": str(e),
                        "retryable": True,
                    },
                    retryable=True,
                )

            # Check thresholds
            violation = self._check_thresholds(analysis)
            if violation:
                return TransformResult.error(
                    {
                        "reason": "content_safety_violation",
                        "field": field_name,
                        "categories": violation,
                        "retryable": False,
                    }
                )

        return TransformResult.success(row)

    def _get_fields_to_scan(self, row: dict[str, Any]) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            return [k for k, v in row.items() if isinstance(v, str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def _get_http_client(self) -> httpx.Client:
        """Get or create HTTP client for API calls."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=30.0)
        return self._http_client

    def _analyze_content(
        self,
        text: str,
        ctx: PluginContext,
    ) -> dict[str, int]:
        """Call Azure Content Safety API."""
        client = self._get_http_client()

        url = f"{self._endpoint}/contentsafety/text:analyze?api-version={self.API_VERSION}"

        response = client.post(
            url,
            json={"text": text},
            headers={
                "Ocp-Apim-Subscription-Key": self._api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()

        # Parse response into category -> severity mapping
        # Azure API responses are external data (Tier 3: Zero Trust) - wrap parsing
        try:
            data = response.json()
            result: dict[str, int] = {}
            for item in data["categoriesAnalysis"]:
                category = item["category"].lower().replace("selfharm", "self_harm")
                result[category] = item["severity"]
            return result
        except (KeyError, TypeError, ValueError) as e:
            raise httpx.RequestError(f"Malformed API response: {e}") from e

    def _check_thresholds(
        self,
        analysis: dict[str, int],
    ) -> dict[str, dict[str, Any]] | None:
        """Check if any category exceeds its threshold.

        Missing categories in the analysis default to severity 0 (safe).
        This handles external API responses that may omit categories.
        """
        # External API data may not include all categories - default to 0 (safe)
        categories: dict[str, dict[str, Any]] = {
            "hate": {
                "severity": analysis.get("hate", 0),
                "threshold": self._thresholds.hate,
            },
            "violence": {
                "severity": analysis.get("violence", 0),
                "threshold": self._thresholds.violence,
            },
            "sexual": {
                "severity": analysis.get("sexual", 0),
                "threshold": self._thresholds.sexual,
            },
            "self_harm": {
                "severity": analysis.get("self_harm", 0),
                "threshold": self._thresholds.self_harm,
            },
        }

        for info in categories.values():
            info["exceeded"] = info["severity"] > info["threshold"]

        if any(info["exceeded"] for info in categories.values()):
            return categories
        return None

    def close(self) -> None:
        """Release resources."""
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None
