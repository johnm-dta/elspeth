"""Offline real-botocore proof for conditional S3 effect requests."""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from hashlib import sha256

import boto3
from botocore.stub import ANY, Stubber

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
)
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink

_CTX = RestrictedSinkEffectContext(
    run_id="run-botocore",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="operation-botocore",
    sink_node_id="sink-botocore",
)


def _member(ordinal: int, identity: int, row: dict[str, object]) -> SinkEffectMember:
    row_bytes = canonical_json(row).encode()
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{identity}",
        row_id=f"row-{identity}",
        ingest_sequence=identity,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(row_bytes).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{identity}".encode()).hexdigest(),
    )


def _prepare(
    sink: AWSS3Sink,
    *,
    effect_id: str,
    current: tuple[SinkEffectMember, ...],
    snapshot: tuple[SinkEffectMember, ...],
    predecessor=None,
):
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(
            effect_id=effect_id,
            target="{}",
            predecessor_descriptor=predecessor,
        ),
        _CTX,
    )
    return sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id=effect_id,
            effect_input=SinkEffectPipelineMembersInput(members=current, target_snapshot_members=snapshot),
            inspection=inspection,
        ),
        _CTX,
    )


def test_real_botocore_stubber_emits_conditional_create_then_etag_successor() -> None:
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    sink_config = {
        "bucket": "example-bucket",
        "key": "output.json",
        "format": "json",
        "overwrite": True,
        "schema": {"mode": "observed"},
    }
    first = _member(0, 1, {"id": 1})
    second_current = _member(0, 2, {"id": 2})
    second_snapshot = _member(1, 2, {"id": 2})

    with Stubber(client) as stubber:
        stubber.add_client_error(
            "head_object",
            service_error_code="NoSuchKey",
            service_message="missing",
            http_status_code=404,
            expected_params={"Bucket": "example-bucket", "Key": "output.json", "ChecksumMode": "ENABLED"},
        )
        first_sink = AWSS3Sink(sink_config)
        first_sink._s3_client = client
        first_plan = _prepare(
            first_sink,
            effect_id="a" * 64,
            current=(first,),
            snapshot=(first,),
        )
        stubber.add_response(
            "put_object",
            {"ETag": '"etag-1"'},
            {
                "Bucket": "example-bucket",
                "Key": "output.json",
                "Body": ANY,
                "ContentLength": first_plan.expected_descriptor.size_bytes,
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(first_plan.payload_hash)).decode("ascii"),
                "IfNoneMatch": "*",
                "Metadata": {
                    "elspeth-content-sha256": first_plan.payload_hash,
                    "elspeth-effect-id": first_plan.effect_id,
                    "elspeth-plan-hash": first_plan.plan_hash,
                    "elspeth-protocol-version": "sink-effect-v1",
                },
            },
        )
        first_result = first_sink.commit_effect(first_plan, _CTX)

        stubber.add_response(
            "head_object",
            {
                "ContentLength": first_plan.expected_descriptor.size_bytes,
                "ETag": '"etag-1"',
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(first_plan.payload_hash)).decode("ascii"),
                "Metadata": {
                    "elspeth-content-sha256": first_plan.payload_hash,
                    "elspeth-effect-id": first_plan.effect_id,
                    "elspeth-plan-hash": first_plan.plan_hash,
                    "elspeth-protocol-version": "sink-effect-v1",
                },
            },
            {"Bucket": "example-bucket", "Key": "output.json", "ChecksumMode": "ENABLED"},
        )
        second_sink = AWSS3Sink(sink_config)
        second_sink._s3_client = client
        second_plan = _prepare(
            second_sink,
            effect_id="b" * 64,
            current=(second_current,),
            snapshot=(first, second_snapshot),
            predecessor=first_result.descriptor,
        )
        stubber.add_response(
            "put_object",
            {"ETag": '"etag-2"'},
            {
                "Bucket": "example-bucket",
                "Key": "output.json",
                "Body": ANY,
                "ContentLength": second_plan.expected_descriptor.size_bytes,
                "ChecksumSHA256": base64.b64encode(bytes.fromhex(second_plan.payload_hash)).decode("ascii"),
                "IfMatch": '"etag-1"',
                "Metadata": {
                    "elspeth-content-sha256": second_plan.payload_hash,
                    "elspeth-effect-id": second_plan.effect_id,
                    "elspeth-plan-hash": second_plan.plan_hash,
                    "elspeth-protocol-version": "sink-effect-v1",
                },
            },
        )
        second_sink.commit_effect(second_plan, _CTX)

        stubber.assert_no_pending_responses()
