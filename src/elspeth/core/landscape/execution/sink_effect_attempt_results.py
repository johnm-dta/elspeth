"""Closed durable codecs for successful sink-effect provider results."""

from __future__ import annotations

import json
from collections.abc import Mapping

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SinkEffectAttemptAction,
    SinkEffectCommitResult,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectReconcileKind,
    SinkEffectReconcileResult,
)
from elspeth.core.landscape.errors import LandscapeRecordError

SinkEffectReturnedResult = SinkEffectInspection | SinkEffectCommitResult | SinkEffectReconcileResult


def _descriptor_payload(descriptor: ArtifactDescriptor | None) -> object:
    if descriptor is None:
        return None
    return {
        "artifact_type": descriptor.artifact_type,
        "content_hash": descriptor.content_hash,
        "metadata": None if descriptor.metadata is None else deep_thaw(descriptor.metadata),
        "path_or_uri": descriptor.path_or_uri,
        "size_bytes": descriptor.size_bytes,
    }


def _load_descriptor(value: object) -> ArtifactDescriptor | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise LandscapeRecordError("sink effect returned descriptor must be an object or null")
    try:
        return ArtifactDescriptor(
            artifact_type=value["artifact_type"],
            path_or_uri=value["path_or_uri"],
            content_hash=value["content_hash"],
            size_bytes=value["size_bytes"],
            metadata=value["metadata"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LandscapeRecordError("sink effect returned descriptor is incomplete or divergent") from exc


def encode_sink_effect_returned_result(result: SinkEffectReturnedResult) -> Mapping[str, object]:
    """Encode one exact provider return into its durable closed envelope."""
    if type(result) is SinkEffectInspection:
        return {
            "evidence": deep_thaw(result.evidence),
            "mode": result.mode.value,
            "reference": result.reference,
            "schema": "sink-effect-inspection-result-v1",
        }
    if type(result) is SinkEffectCommitResult:
        return {
            "accepted_ordinals": list(result.accepted_ordinals),
            "descriptor": _descriptor_payload(result.descriptor),
            "diverted_ordinals": list(result.diverted_ordinals),
            "evidence": deep_thaw(result.evidence),
            "schema": "sink-effect-commit-result-v1",
        }
    if type(result) is SinkEffectReconcileResult:
        return {
            "descriptor": _descriptor_payload(result.descriptor),
            "evidence": deep_thaw(result.evidence),
            "kind": result.kind.value,
            "schema": "sink-effect-reconcile-result-v1",
        }
    raise TypeError("result must be a closed sink-effect returned result")


def decode_sink_effect_returned_result(
    action: SinkEffectAttemptAction,
    evidence_json: str,
) -> SinkEffectReturnedResult:
    """Decode and validate one durable provider-return envelope."""
    try:
        payload = json.loads(evidence_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LandscapeRecordError("sink effect returned attempt has invalid result JSON") from exc
    if type(payload) is not dict or type(payload.get("evidence")) is not dict:
        raise LandscapeRecordError("sink effect returned attempt must contain exact evidence")
    evidence = payload["evidence"]
    try:
        if action is SinkEffectAttemptAction.INSPECT:
            if set(payload) != {"evidence", "mode", "reference", "schema"} or payload["schema"] != "sink-effect-inspection-result-v1":
                raise LandscapeRecordError("sink effect inspect result envelope is divergent")
            return SinkEffectInspection(
                mode=SinkEffectInspectionMode(payload["mode"]),
                reference=payload["reference"],
                evidence=evidence,
            )
        if action is SinkEffectAttemptAction.COMMIT:
            if (
                set(payload) != {"accepted_ordinals", "descriptor", "diverted_ordinals", "evidence", "schema"}
                or payload["schema"] != "sink-effect-commit-result-v1"
            ):
                raise LandscapeRecordError("sink effect commit result envelope is divergent")
            descriptor = _load_descriptor(payload["descriptor"])
            if descriptor is None:
                raise LandscapeRecordError("sink effect commit result requires a descriptor")
            return SinkEffectCommitResult(
                descriptor=descriptor,
                evidence=evidence,
                accepted_ordinals=payload["accepted_ordinals"],
                diverted_ordinals=payload["diverted_ordinals"],
            )
        if set(payload) != {"descriptor", "evidence", "kind", "schema"} or payload["schema"] != "sink-effect-reconcile-result-v1":
            raise LandscapeRecordError("sink effect reconcile result envelope is divergent")
        return SinkEffectReconcileResult(
            kind=SinkEffectReconcileKind(payload["kind"]),
            descriptor=_load_descriptor(payload["descriptor"]),
            evidence=evidence,
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, LandscapeRecordError):
            raise
        raise LandscapeRecordError("sink effect returned attempt result is incomplete or divergent") from exc


__all__ = [
    "SinkEffectReturnedResult",
    "decode_sink_effect_returned_result",
    "encode_sink_effect_returned_result",
]
