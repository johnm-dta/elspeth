"""Reusable, receipt-only live proof for operator-owned Bedrock Guardrails."""

from __future__ import annotations

import hashlib
import re
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.audit_protocols import CallRecorder
from elspeth.plugins.infrastructure.clients.base import TelemetryEmitCallback
from elspeth.plugins.transforms.aws.guardrail_profiles import BedrockGuardrailProfileSettings
from elspeth.plugins.transforms.aws.guardrails_client import (
    HARMFUL_CONTENT_FILTERS,
    PROMPT_ATTACK_FILTERS,
    BedrockGuardrailsClient,
    GuardrailSource,
)

_PROFILE_ALIAS = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*\Z")
_PLUGIN_POLICIES: dict[str, tuple[GuardrailSource, tuple[str, ...]]] = {
    "aws_bedrock_prompt_shield": ("INPUT", PROMPT_ATTACK_FILTERS),
    "aws_bedrock_content_safety": ("OUTPUT", HARMFUL_CONTENT_FILTERS),
}


class GuardrailLiveCheckError(RuntimeError):
    """Sanitized failure from the reusable live proof."""


@dataclass(frozen=True, slots=True)
class GuardrailLiveReceipt:
    """Bounded facts safe to retain from two approved live cases."""

    plugin_id: str
    profile_alias: str
    safe_case_passed: bool
    attack_case_blocked: bool
    request_ids_present: bool

    def __post_init__(self) -> None:
        if self.plugin_id not in _PLUGIN_POLICIES:
            raise ValueError("live receipt plugin id is invalid")
        if len(self.profile_alias) > 64 or _PROFILE_ALIAS.fullmatch(self.profile_alias) is None:
            raise ValueError("live receipt profile alias is invalid")
        for value in (self.safe_case_passed, self.attack_case_blocked, self.request_ids_present):
            if type(value) is not bool:
                raise TypeError("live receipt decision fields must be booleans")


def run_guardrail_live_check(
    *,
    profile: BedrockGuardrailProfileSettings,
    safe_text: str,
    blocked_text: str,
    execution: CallRecorder,
    state_id: str,
    run_id: str,
    telemetry_emit: TelemetryEmitCallback,
    sdk_client: Any | None = None,
) -> GuardrailLiveReceipt:
    """Run operator-approved safe/blocked cases and return no raw live data."""
    policy = _PLUGIN_POLICIES.get(profile.plugin)
    if policy is None or len(profile.alias) > 64:
        raise GuardrailLiveCheckError("Bedrock Guardrail live profile is invalid")
    source, required_filters = policy
    owns_sdk_client = sdk_client is None
    client: BedrockGuardrailsClient | None = None
    try:
        client = BedrockGuardrailsClient(
            execution=execution,
            state_id=state_id,
            run_id=run_id,
            telemetry_emit=telemetry_emit,
            guardrail_identifier=profile.guardrail_identifier,
            guardrail_version=profile.guardrail_version,
            region=profile.region,
            audit_salt=hashlib.sha256(f"elspeth-bedrock-guardrail-live:{run_id}".encode()).digest(),
            sdk_client=sdk_client,
        )
        safe = client.apply_guardrail(
            text=safe_text,
            source=source,
            required_filters=required_filters,
        )
        if safe.detected:
            raise GuardrailLiveCheckError("approved safe case did not pass")
        blocked = client.apply_guardrail(
            text=blocked_text,
            source=source,
            required_filters=required_filters,
        )
        if not blocked.detected:
            raise GuardrailLiveCheckError("approved blocked case was not blocked")
        return GuardrailLiveReceipt(
            plugin_id=profile.plugin,
            profile_alias=profile.alias,
            safe_case_passed=True,
            attack_case_blocked=True,
            request_ids_present=safe.request_id is not None and blocked.request_id is not None,
        )
    except GuardrailLiveCheckError:
        raise
    except Exception:
        raise GuardrailLiveCheckError("Bedrock Guardrail live check failed") from None
    finally:
        if owns_sdk_client and client is not None:
            close = getattr(client.sdk_client, "close", None)
            if callable(close):
                # The live proof must never let an SDK cleanup message bypass
                # its receipt-only error boundary.
                with suppress(Exception):
                    close()
