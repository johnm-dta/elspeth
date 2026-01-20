"""Azure Prompt Shield transform for jailbreak and prompt injection detection.

This module provides the AzurePromptShield transform which uses Azure's
Prompt Shield API to detect:
- User prompt attacks (jailbreak attempts in the user's message)
- Document attacks (prompt injection in documents/context)

Unlike Content Safety, Prompt Shield is binary detection - no thresholds.
Either an attack is detected or it isn't.
"""

from typing import Any

import httpx
from pydantic import Field

from elspeth.contracts import Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class AzurePromptShieldConfig(TransformDataConfig):
    """Configuration for Azure Prompt Shield transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        schema: Schema configuration

    Example YAML:
        transforms:
          - plugin: azure_prompt_shield
            options:
              endpoint: https://my-resource.cognitiveservices.azure.com
              api_key: ${AZURE_CONTENT_SAFETY_KEY}
              fields: [prompt, user_message]
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


class AzurePromptShield(BaseTransform):
    """Detect jailbreak attempts and prompt injection using Azure Prompt Shield.

    Analyzes text against Azure's Prompt Shield API which detects:
    - User prompt attacks: Direct jailbreak attempts in the user's message
    - Document attacks: Prompt injection hidden in documents or context

    Returns error result if any attack is detected (binary, no thresholds).
    """

    name = "azure_prompt_shield"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    is_batch_aware = False
    creates_tokens = False

    API_VERSION = "2024-09-01"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = AzurePromptShieldConfig.from_dict(config)
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key = cfg.api_key
        self._fields = cfg.fields
        self._on_error = cfg.on_error

        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzurePromptShieldSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Analyze row content for prompt injection attacks."""
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            if field_name not in row:
                continue  # Skip fields not present in this row

            value = row[field_name]
            if not isinstance(value, str):
                continue

            # Call Azure API
            try:
                analysis = self._analyze_prompt(value, ctx)
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

            # Check if any attack was detected
            if analysis["user_prompt_attack"] or analysis["document_attack"]:
                return TransformResult.error(
                    {
                        "reason": "prompt_injection_detected",
                        "field": field_name,
                        "attacks": analysis,
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

    def _analyze_prompt(
        self,
        text: str,
        ctx: PluginContext,
    ) -> dict[str, bool]:
        """Call Azure Prompt Shield API.

        Returns dict with:
            user_prompt_attack: True if jailbreak detected in user prompt
            document_attack: True if prompt injection detected in any document
        """
        if ctx.http_client is None:
            raise RuntimeError(
                "AzurePromptShield requires http_client in PluginContext"
            )

        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self.API_VERSION}"

        response = ctx.http_client.post(
            url,
            json={
                "userPrompt": text,
                "documents": [text],
            },
            headers={
                "Ocp-Apim-Subscription-Key": self._api_key,
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()

        # Parse response - Azure API responses are external data (Tier 3: Zero Trust)
        # Missing fields default to False (no attack) to avoid false positives on API changes
        data = response.json()

        # External API data - use .get() with safe defaults
        user_prompt_analysis = data.get("userPromptAnalysis", {})
        user_attack = user_prompt_analysis.get("attackDetected", False)

        documents_analysis = data.get("documentsAnalysis", [])
        doc_attack = any(doc.get("attackDetected", False) for doc in documents_analysis)

        return {
            "user_prompt_attack": user_attack,
            "document_attack": doc_attack,
        }

    def close(self) -> None:
        """Release resources."""
        pass
