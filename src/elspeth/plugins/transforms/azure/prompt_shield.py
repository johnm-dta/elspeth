"""Azure Prompt Shield transform for jailbreak and prompt injection detection.

Uses Azure's Prompt Shield API to detect:
- User prompt attacks (jailbreak attempts in the user's message)
- Document attacks (prompt injection in documents/context)

Unlike Content Safety, Prompt Shield is binary detection — no thresholds.
Either an attack is detected or it isn't.

Uses BaseAzureSafetyTransform for shared batch infrastructure.
"""

from __future__ import annotations

from typing import Any, TypedDict

from pydantic import Field

from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms.azure.base import (
    BaseAzureSafetyConfig,
    BaseAzureSafetyTransform,
)
from elspeth.plugins.transforms.azure.errors import MalformedResponseError


class AzurePromptShieldConfig(BaseAzureSafetyConfig):
    """Configuration for Azure Prompt Shield transform.

    Requires:
        endpoint: Azure Content Safety endpoint URL
        api_key: Azure Content Safety API key
        fields: Field name(s) to analyze, or 'all' for all string fields
        schema: Schema configuration

    Optional:
        analysis_type: Which analysis to run (default "both")
            - "both": Analyze as both user prompt and document (double cost)
            - "user_prompt": Only check for user prompt attacks (jailbreak)
            - "document": Only check for document attacks (prompt injection)
        max_capacity_retry_seconds: Timeout for capacity error retries (default 3600)
        batch_wait_timeout_seconds: Timeout for receiving a completed row result (default 3600)

    Example YAML:
        transforms:
          - plugin: azure_prompt_shield
            options:
              endpoint: https://my-resource.cognitiveservices.azure.com
              api_key: ${AZURE_CONTENT_SAFETY_KEY}
              fields: [prompt, user_message]
              analysis_type: user_prompt
              on_error: quarantine_sink
              schema:
                mode: observed
    """

    # Analysis type control — avoids double API cost when only one analysis is needed
    analysis_type: str = Field(
        "both",
        pattern=r"^(both|user_prompt|document)$",
        description="Which analysis to run: 'both', 'user_prompt', or 'document'",
    )


class _PromptShieldProbeRequestBody(TypedDict):
    userPrompt: str
    documents: list[str]


class _PromptShieldProbeUserAnalysis(TypedDict):
    attackDetected: bool


class _PromptShieldProbeDocumentAnalysis(TypedDict):
    attackDetected: bool


class _PromptShieldProbeResponseBody(TypedDict):
    userPromptAnalysis: _PromptShieldProbeUserAnalysis
    documentsAnalysis: list[_PromptShieldProbeDocumentAnalysis]


class AzurePromptShield(BaseAzureSafetyTransform):
    """Detect jailbreak attempts and prompt injection using Azure Prompt Shield.

    Analyzes text against Azure's Prompt Shield API which detects:
    - User prompt attacks: Direct jailbreak attempts in the user's message
    - Document attacks: Prompt injection hidden in documents or context

    Returns error result if any attack is detected (binary, no thresholds).

    Uses BaseAzureSafetyTransform for row-level pipelining: multiple rows
    can be in flight concurrently with FIFO output ordering.
    """

    name = "azure_prompt_shield"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:19f08887d6f0065d"
    config_model = AzurePromptShieldConfig
    passes_through_input = True

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 forward invariant."""
        return {
            "endpoint": "https://test.cognitiveservices.azure.com",
            "api_key": "test-key",
            "fields": ["prompt_shield_probe_text"],
            "schema": {"mode": "observed"},
        }

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = AzurePromptShieldConfig.from_dict(config, plugin_name=self.name)
        super().__init__(config, cfg, "AzurePromptShieldSchema")
        self._analysis_type = cfg.analysis_type

    def forward_invariant_probe_rows(self, probe: Any) -> list[Any]:
        """Inject a deterministic safe prompt field for invariant probing."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="prompt_shield_probe_text",
                value="safe prompt",
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

            def json(self) -> _PromptShieldProbeResponseBody:
                return {
                    "userPromptAnalysis": {"attackDetected": False},
                    "documentsAnalysis": [{"attackDetected": False}],
                }

        class _ProbeClient:
            def post(
                self,
                url: str,
                json: _PromptShieldProbeRequestBody,
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
        """Analyze field via Prompt Shield API for attack detection."""
        analysis = self._analyze_prompt(value, state_id, token_id=token_id)

        if analysis["user_prompt_attack"] or analysis["document_attack"]:
            return TransformResult.error(
                {
                    "reason": "prompt_injection_detected",
                    "field": field_name,
                    "attacks": analysis,
                },
                retryable=False,
            )
        return None

    def _analyze_prompt(
        self,
        text: str,
        state_id: str,
        *,
        token_id: str | None = None,
    ) -> dict[str, bool]:
        """Call Azure Prompt Shield API.

        Returns dict with:
            user_prompt_attack: True if jailbreak detected in user prompt
            document_attack: True if prompt injection detected in any document

        Uses AuditedHTTPClient for automatic audit recording and telemetry emission.
        Respects self._analysis_type to avoid double API cost when only one
        analysis path is needed.
        """
        http_client = self._get_http_client(state_id, token_id=token_id)

        url = f"{self._endpoint}/contentsafety/text:shieldPrompt?api-version={self.API_VERSION}"

        # Build request body based on analysis_type to avoid double cost.
        # "both" (default): text analyzed as both user prompt and document
        # "user_prompt": only user prompt analysis (empty documents list)
        # "document": only document analysis (empty user prompt)
        if self._analysis_type == "user_prompt":
            request_body = {"userPrompt": text, "documents": []}
        elif self._analysis_type == "document":
            request_body = {"userPrompt": "", "documents": [text]}
        else:
            request_body = {"userPrompt": text, "documents": [text]}

        # Make HTTP call - AuditedHTTPClient records to Landscape and emits telemetry
        response = http_client.post(url, json=request_body)
        response.raise_for_status()

        # Parse response - Azure API responses are external data (Tier 3: Zero Trust)
        # Security transform: fail CLOSED on malformed response
        #
        # We validate types strictly because:
        # - attackDetected=null would be falsy → fail OPEN (security vulnerability)
        # - attackDetected="true" would be truthy but for wrong reason
        # - Non-list documentsAnalysis would crash or misbehave
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            raise MalformedResponseError(f"Invalid JSON in response: {e}") from e

        # Tier 3 boundary: validate top-level response structure first
        if not isinstance(data, dict):
            raise MalformedResponseError(f"Response must be dict, got {type(data).__name__}")

        user_attack = False
        doc_attack = False

        # Validate and extract user prompt analysis (skip if analysis_type="document")
        if self._analysis_type != "document":
            try:
                user_prompt_analysis = data["userPromptAnalysis"]
            except KeyError as exc:
                raise MalformedResponseError("Response missing required key 'userPromptAnalysis'") from exc
            if not isinstance(user_prompt_analysis, dict):
                raise MalformedResponseError(f"userPromptAnalysis must be dict, got {type(user_prompt_analysis).__name__}")

            try:
                detected = user_prompt_analysis["attackDetected"]
            except KeyError as exc:
                raise MalformedResponseError("userPromptAnalysis missing required key 'attackDetected'") from exc
            if not isinstance(detected, bool):
                raise MalformedResponseError(f"userPromptAnalysis.attackDetected must be bool, got {type(detected).__name__}")
            user_attack = detected

        # Validate and extract document analysis (skip if analysis_type="user_prompt")
        if self._analysis_type != "user_prompt":
            try:
                documents_analysis = data["documentsAnalysis"]
            except KeyError as exc:
                raise MalformedResponseError("Response missing required key 'documentsAnalysis'") from exc
            if not isinstance(documents_analysis, list):
                raise MalformedResponseError(f"documentsAnalysis must be list, got {type(documents_analysis).__name__}")

            # We submitted exactly 1 document — Azure must return exactly 1 analysis.
            # Empty list = fail OPEN (no document was analyzed → no attack flagged).
            if len(documents_analysis) != 1:
                raise MalformedResponseError(
                    f"documentsAnalysis must have exactly 1 entry (matching submitted document count), got {len(documents_analysis)}"
                )

            doc = documents_analysis[0]
            if not isinstance(doc, dict):
                raise MalformedResponseError(f"documentsAnalysis[0] must be dict, got {type(doc).__name__}")
            try:
                attack_detected = doc["attackDetected"]
            except KeyError as exc:
                raise MalformedResponseError("documentsAnalysis[0] missing required key 'attackDetected'") from exc
            if not isinstance(attack_detected, bool):
                raise MalformedResponseError(f"documentsAnalysis[0].attackDetected must be bool, got {type(attack_detected).__name__}")
            doc_attack = attack_detected

        return {
            "user_prompt_attack": user_attack,
            "document_attack": doc_attack,
        }
