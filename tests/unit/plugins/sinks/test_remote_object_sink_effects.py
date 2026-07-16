"""Recovery proofs for conditional S3 and Azure object sink effects."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import md5, sha256
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.engine.orchestrator.preflight import validate_sink_effect_capability
from elspeth.plugins.sinks.aws_s3_sink import AWSS3Sink
from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSink
from tests.fixtures.base_classes import inject_write_failure

_SCHEMA = {"mode": "observed"}
_CTX = RestrictedSinkEffectContext(
    run_id="run-remote-1",
    run_started_at=datetime(2026, 7, 16, 3, 4, 5, 678901, tzinfo=UTC),
    operation_id="operation-remote-1",
    sink_node_id="sink-remote-1",
)


def _member(ordinal: int, row: dict[str, object], *, identity: int | None = None) -> SinkEffectMember:
    stable_identity = ordinal if identity is None else identity
    row_bytes = canonical_json(row).encode()
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{stable_identity}",
        row_id=f"row-{stable_identity}",
        ingest_sequence=stable_identity,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(row_bytes).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{stable_identity}-{row_bytes!r}".encode()).hexdigest(),
    )


def _prepare(
    sink: AWSS3Sink | AzureBlobSink,
    *,
    effect_id: str,
    current: tuple[SinkEffectMember, ...],
    target_snapshot: tuple[SinkEffectMember, ...],
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
            effect_input=SinkEffectPipelineMembersInput(
                members=current,
                target_snapshot_members=target_snapshot,
            ),
            inspection=inspection,
        ),
        _CTX,
    )


class _S3Missing(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "NoSuchKey"},
        "ResponseMetadata": {"HTTPStatusCode": 404},
    }


class _S3PreconditionFailed(Exception):
    response: ClassVar[dict[str, object]] = {
        "Error": {"Code": "PreconditionFailed"},
        "ResponseMetadata": {"HTTPStatusCode": 412},
    }


@dataclass
class _Object:
    body: bytes
    etag: str
    metadata: dict[str, str]


class _S3Store:
    def __init__(self) -> None:
        self.value: _Object | None = None
        self.requests: list[dict[str, object]] = []
        self.response_loss = False
        self.control: BaseException | None = None

    def head_object(self, **request: object) -> dict[str, object]:
        self.requests.append({"operation": "head", **request})
        if self.value is None:
            raise _S3Missing()
        return {
            "ContentLength": len(self.value.body),
            "ETag": self.value.etag,
            "Metadata": self.value.metadata,
            "ChecksumSHA256": base64.b64encode(sha256(self.value.body).digest()).decode("ascii"),
        }

    def put_object(self, **request: object) -> dict[str, object]:
        self.requests.append({"operation": "put", **request})
        if self.control is not None:
            raise self.control
        if request.get("IfNoneMatch") == "*" and self.value is not None:
            raise _S3PreconditionFailed()
        if "IfMatch" in request and (self.value is None or request["IfMatch"] != self.value.etag):
            raise _S3PreconditionFailed()
        body = request["Body"]
        assert hasattr(body, "read")
        payload = body.read()  # type: ignore[union-attr]
        assert type(payload) is bytes
        etag = f'"etag-{len(self.requests)}"'
        metadata = request["Metadata"]
        assert isinstance(metadata, dict)
        self.value = _Object(payload, etag, dict(metadata))
        if self.response_loss:
            self.response_loss = False
            raise ConnectionError("response lost after accepted write")
        return {"ETag": etag}


class ResourceNotFoundError(Exception):
    pass


class _AzurePreconditionFailed(Exception):
    status_code = 412


class _AzureBlob:
    def __init__(self, store: _AzureStore) -> None:
        self._store = store

    def get_blob_properties(self) -> SimpleNamespace:
        self._store.requests.append({"operation": "properties"})
        value = self._store.value
        if value is None:
            raise ResourceNotFoundError()
        return SimpleNamespace(
            size=len(value.body),
            etag=value.etag,
            metadata=value.metadata,
            content_settings=SimpleNamespace(content_md5=md5(value.body, usedforsecurity=False).digest()),
        )

    def upload_blob(self, data: object, **request: object) -> dict[str, object]:
        self._store.requests.append({"operation": "upload", **request})
        if request.get("if_none_match") == "*" and self._store.value is not None:
            raise _AzurePreconditionFailed()
        if "etag" in request and (self._store.value is None or request["etag"] != self._store.value.etag):
            raise _AzurePreconditionFailed()
        if hasattr(data, "read"):
            payload = data.read()  # type: ignore[union-attr]
        else:
            payload = data
        assert type(payload) is bytes
        metadata = request["metadata"]
        assert isinstance(metadata, dict)
        etag = f'"etag-{len(self._store.requests)}"'
        self._store.value = _Object(payload, etag, dict(metadata))
        if self._store.response_loss:
            self._store.response_loss = False
            raise ConnectionError("response lost after accepted upload")
        return {"etag": etag}


class _AzureStore:
    def __init__(self) -> None:
        self.value: _Object | None = None
        self.requests: list[dict[str, object]] = []
        self.response_loss = False
        self.blob = _AzureBlob(self)

    def get_blob_client(self, *_args: object, **_kwargs: object) -> _AzureBlob:
        return self.blob

    def close(self) -> None:
        return None


def _s3(store: _S3Store, **overrides: object) -> AWSS3Sink:
    config: dict[str, Any] = {
        "bucket": "bucket",
        "key": "runs/{{ run_id }}/{{ timestamp }}/out.json",
        "format": "json",
        "overwrite": True,
        "schema": _SCHEMA,
    }
    config.update(overrides)
    sink = inject_write_failure(AWSS3Sink(config))
    sink._s3_client = store
    return sink


def _azure(store: _AzureStore, **overrides: object) -> AzureBlobSink:
    config: dict[str, Any] = {
        "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake;EndpointSuffix=core.windows.net",
        "container": "container",
        "blob_path": "runs/{{ run_id }}/{{ timestamp }}/out.json",
        "format": "json",
        "overwrite": True,
        "schema": _SCHEMA,
    }
    config.update(overrides)
    sink = inject_write_failure(AzureBlobSink(config))
    sink._container_client = store  # type: ignore[assignment]
    return sink


@pytest.mark.parametrize("factory", [_s3, _azure])
def test_remote_sinks_declare_recoverable_pipeline_effects(factory: Any) -> None:
    store = _S3Store() if factory is _s3 else _AzureStore()
    sink = factory(store)
    validate_sink_effect_capability(sink, "write", SinkEffectInputKind.PIPELINE_MEMBERS)


def test_s3_fresh_process_successor_uses_snapshot_and_etag_condition() -> None:
    store = _S3Store()
    first_member = _member(0, {"id": 1})
    first_sink = _s3(store)
    first_plan = _prepare(
        first_sink,
        effect_id="a" * 64,
        current=(first_member,),
        target_snapshot=(first_member,),
    )
    first_result = first_sink.commit_effect(first_plan, _CTX)

    second_member = _member(1, {"id": 2}, identity=1)
    second_current = _member(0, {"id": 2}, identity=1)
    second_sink = _s3(store)
    second_plan = _prepare(
        second_sink,
        effect_id="b" * 64,
        current=(second_current,),
        target_snapshot=(first_member, second_member),
        predecessor=first_result.descriptor,
    )
    second_sink.commit_effect(second_plan, _CTX)

    assert store.value is not None
    assert json.loads(store.value.body) == [{"id": 1}, {"id": 2}]
    put_requests = [request for request in store.requests if request["operation"] == "put"]
    assert put_requests[0]["IfNoneMatch"] == "*"
    assert put_requests[1]["IfMatch"] == '"etag-2"'
    assert put_requests[1]["ChecksumSHA256"] == base64.b64encode(sha256(store.value.body).digest()).decode("ascii")
    assert put_requests[1]["Metadata"] == {
        "elspeth-content-sha256": second_plan.payload_hash,
        "elspeth-effect-id": second_plan.effect_id,
        "elspeth-plan-hash": second_plan.plan_hash,
        "elspeth-protocol-version": "sink-effect-v1",
    }


@pytest.mark.parametrize("control", [KeyboardInterrupt(), SystemExit()])
def test_s3_commit_does_not_convert_process_control_exceptions(control: BaseException) -> None:
    store = _S3Store()
    sink = _s3(store)
    member = _member(0, {"id": 1})
    plan = _prepare(sink, effect_id="f" * 64, current=(member,), target_snapshot=(member,))
    store.control = control

    with pytest.raises(type(control)):
        sink.commit_effect(plan, _CTX)


def test_s3_response_loss_reconciles_exact_effect_in_fresh_instance() -> None:
    store = _S3Store()
    sink = _s3(store)
    member = _member(0, {"id": 1})
    plan = _prepare(sink, effect_id="c" * 64, current=(member,), target_snapshot=(member,))
    store.response_loss = True

    with pytest.raises(RuntimeError, match="outcome is unknown"):
        sink.commit_effect(plan, _CTX)

    result = _s3(store).reconcile_effect(plan, _CTX)
    assert result.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert result.descriptor == plan.expected_descriptor


def test_diverted_successor_preserves_prior_snapshot_without_publication() -> None:
    store = _S3Store()
    first_member = _member(0, {"id": 1})
    first_sink = _s3(store, max_record_chars=10)
    first_plan = _prepare(
        first_sink,
        effect_id="2" * 64,
        current=(first_member,),
        target_snapshot=(first_member,),
    )
    first_result = first_sink.commit_effect(first_plan, _CTX)
    current = _member(0, {"id": "x" * 100}, identity=1)
    snapshot = _member(1, {"id": "x" * 100}, identity=1)
    successor = _s3(store, max_record_chars=10)

    plan = _prepare(
        successor,
        effect_id="3" * 64,
        current=(current,),
        target_snapshot=(first_member, snapshot),
        predecessor=first_result.descriptor,
    )

    assert plan.descriptor_mode is SinkEffectDescriptorMode.NO_PUBLICATION
    assert plan.expected_descriptor == first_result.descriptor
    assert plan.safe_evidence["accepted_ordinals"] == ()
    assert plan.safe_evidence["diverted_ordinals"] == (0,)


def test_azure_fresh_process_successor_and_response_loss_reconcile() -> None:
    store = _AzureStore()
    first_member = _member(0, {"id": 1})
    first_sink = _azure(store)
    first_plan = _prepare(
        first_sink,
        effect_id="d" * 64,
        current=(first_member,),
        target_snapshot=(first_member,),
    )
    first_result = first_sink.commit_effect(first_plan, _CTX)

    second_member = _member(1, {"id": 2}, identity=1)
    second_current = _member(0, {"id": 2}, identity=1)
    second_sink = _azure(store)
    second_plan = _prepare(
        second_sink,
        effect_id="e" * 64,
        current=(second_current,),
        target_snapshot=(first_member, second_member),
        predecessor=first_result.descriptor,
    )
    store.response_loss = True
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        second_sink.commit_effect(second_plan, _CTX)

    assert store.value is not None
    assert json.loads(store.value.body) == [{"id": 1}, {"id": 2}]
    upload_requests = [request for request in store.requests if request["operation"] == "upload"]
    assert upload_requests[0]["if_none_match"] == "*"
    assert upload_requests[1]["etag"] == '"etag-2"'
    assert upload_requests[1]["metadata"]["elspeth_protocol_version"] == "sink-effect-v1"  # type: ignore[index]
    assert (
        upload_requests[1]["content_settings"].content_md5
        == md5(  # type: ignore[union-attr]
            store.value.body,
            usedforsecurity=False,
        ).digest()
    )
    reconciled = _azure(store).reconcile_effect(second_plan, _CTX)
    assert reconciled.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert reconciled.descriptor == second_plan.expected_descriptor


def test_azure_effect_diverts_fixed_schema_extra_and_publishes_good_rows() -> None:
    store = _AzureStore()
    good = _member(0, {"id": 1, "name": "Ada"})
    bad = _member(1, {"id": 2, "name": "Grace", "extra": "bad"})
    sink = _azure(
        store,
        format="csv",
        blob_path="out.csv",
        schema={"mode": "fixed", "fields": ["id: int", "name: str"]},
    )

    plan = _prepare(
        sink,
        effect_id="4" * 64,
        current=(good, bad),
        target_snapshot=(good, bad),
    )
    result = sink.commit_effect(plan, _CTX)

    assert result.accepted_ordinals == (0,)
    assert result.diverted_ordinals == (1,)
    assert store.value is not None
    assert store.value.body.decode() == "id,name\r\n1,Ada\r\n"


@pytest.mark.parametrize("factory", [_s3, _azure])
def test_divergent_remote_metadata_is_unknown_and_never_credited(factory: Any) -> None:
    store = _S3Store() if factory is _s3 else _AzureStore()
    sink = factory(store)
    member = _member(0, {"id": 1})
    plan = _prepare(sink, effect_id="9" * 64, current=(member,), target_snapshot=(member,))
    sink.commit_effect(plan, _CTX)
    assert store.value is not None
    store.value.metadata.clear()

    result = factory(store).reconcile_effect(plan, _CTX)

    assert result.kind is SinkEffectReconcileKind.UNKNOWN


@pytest.mark.parametrize("factory", [_s3, _azure])
def test_remote_target_timestamp_is_bound_to_run_start(factory: Any) -> None:
    store = _S3Store() if factory is _s3 else _AzureStore()
    sink = factory(store)
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(effect_id="f" * 64, target="{}", predecessor_descriptor=None),
        _CTX,
    )
    assert "2026-07-16T03:04:05.678901+00:00" in inspection.reference


def test_effect_plan_body_is_durable_and_not_process_local(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELSPETH_EFFECT_SPOOL_DIR", str(tmp_path))
    store = _S3Store()
    member = _member(0, {"id": 1})
    plan = _prepare(_s3(store), effect_id="1" * 64, current=(member,), target_snapshot=(member,))
    stage = plan.safe_evidence["staging_path"]
    assert isinstance(stage, str)
    with open(stage, "rb") as stream:
        assert isinstance(stream, io.BufferedReader)
        assert json.load(stream) == [{"id": 1}]
    assert "payload" not in plan.safe_evidence


@pytest.mark.parametrize("factory", [_s3, _azure])
def test_remote_sinks_have_no_process_local_publication_authority(factory: Any) -> None:
    store = _S3Store() if factory is _s3 else _AzureStore()
    sink = factory(store)
    for obsolete in (
        "_buffered_rows",
        "_resolved_key",
        "_remote_etag",
        "_confirmed_artifact",
        "_resolved_blob_path",
        "_has_uploaded",
    ):
        assert not hasattr(sink, obsolete)
    with pytest.raises(RuntimeError, match="effect coordinator"):
        sink.write([{"id": 1}], SimpleNamespace(run_id="run", contract=None, landscape=None, operation_id="op"))
