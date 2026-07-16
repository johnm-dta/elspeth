"""Successor predecessor derivation must skip no-publication effects (elspeth-fac5260c6a).

A virtual (zero-accepted) predecessor never touched the remote target, so its
empty-hash artifact is not remote evidence. Declaring it as the stream
predecessor made ``inspect_remote_object`` fail its precondition on every
retry and permanently wedge the stream.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.sink_effects import SinkEffectMember, SinkEffectMemberCandidate
from elspeth.core.landscape.execution.sink_effect_identity import resolve_sink_effect_members
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink_effects import SinkEffectCoordinator
from elspeth.plugins.sinks._remote_object_effects import RemoteObjectPreconditionError
from tests.fixtures.landscape import make_factory, make_landscape_db, register_test_node
from tests.fixtures.stores import MockPayloadStore
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members
from tests.unit.engine.test_sink_effect_executor import _execution_request
from tests.unit.plugins.sinks.test_remote_object_sink_effects import _Object, _s3, _S3Store

_EMPTY_HASH = sha256(b"").hexdigest()


def _members_with_payloads(
    factory: RecorderFactory,
    payloads: list[dict[str, object]],
) -> tuple[str, str, tuple[SinkEffectMember, ...]]:
    """Mirror ``_pipeline_members`` but with caller-controlled row payloads."""
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="sink")
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal, payload in enumerate(payloads):
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_id,
            row_index=ordinal,
            data=payload,
            source_row_index=ordinal,
            ingest_sequence=ordinal,
        )
        token = factory.data_flow.create_token(row.row_id)
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink_id,
            run_id=run.run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    return run.run_id, sink_id, resolve_sink_effect_members(factory, candidates)


@pytest.fixture(autouse=True)
def _isolated_spool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ELSPETH_EFFECT_SPOOL_DIR", str(tmp_path / "spool"))


def test_successor_publishes_after_virtual_predecessor_on_absent_target() -> None:
    """Batch 1 all-diverted (virtual) -> batch 2 accepted must create the object."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 2)
        store = _S3Store()

        # Batch 1: every record exceeds the size cap and diverts, finalizing a
        # virtual NO_PUBLICATION effect whose artifact carries the empty hash.
        first = SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(
            _execution_request(run_id, sink_id, members[:1]),
            _s3(store, max_record_chars=10),
        )
        assert first.effect.publication_performed is False
        assert first.effect.publication_evidence_kind == "virtual"
        assert first.artifact.publication_performed is False
        assert first.artifact.content_hash == _EMPTY_HASH
        assert store.value is None

        # Batch 2: accepted members must publish cleanly instead of wedging on
        # "declared predecessor remote object is absent".
        successor_factory = make_factory(db, payload_store=payload_store)
        second = SinkEffectCoordinator(factory=successor_factory, worker_id="worker-b").execute(
            _execution_request(run_id, sink_id, members[1:]),
            _s3(store),
        )

        assert second.effect.publication_performed is True
        assert store.value is not None
        assert json.loads(store.value.body) == [{"ordinal": 1}]
        put_requests = [request for request in store.requests if request["operation"] == "put"]
        assert len(put_requests) == 1
        # No real publication precedes batch 2, so the write is a guarded create.
        assert put_requests[0]["IfNoneMatch"] == "*"
    finally:
        db.close()


def test_successor_replaces_existing_target_after_virtual_predecessor() -> None:
    """A pre-existing foreign object survives the virtual batch and is then
    replaced under its observed ETag by the first accepted batch."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 2)
        store = _S3Store()
        existing_body = b'[{"id": "existing"}]'
        store.value = _Object(existing_body, '"etag-existing"', {})

        first = SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(
            _execution_request(run_id, sink_id, members[:1]),
            _s3(store, max_record_chars=10),
        )
        assert first.effect.publication_evidence_kind == "virtual"
        assert store.value is not None and store.value.body == existing_body

        # Batch 2 must not wedge on "declared predecessor bytes do not match
        # remote metadata"; it replaces the object bound to its observed ETag.
        successor_factory = make_factory(db, payload_store=payload_store)
        second = SinkEffectCoordinator(factory=successor_factory, worker_id="worker-b").execute(
            _execution_request(run_id, sink_id, members[1:]),
            _s3(store),
        )

        assert second.effect.publication_performed is True
        assert json.loads(store.value.body) == [{"ordinal": 1}]
        put_requests = [request for request in store.requests if request["operation"] == "put"]
        assert len(put_requests) == 1
        # The replace stays fail-closed: it is fenced on the exact pre-image
        # ETag observed during inspection, not an unconditional overwrite.
        assert put_requests[0]["IfMatch"] == '"etag-existing"'
    finally:
        db.close()


def test_successor_publishes_after_chain_of_virtual_predecessors() -> None:
    """Consecutive all-diverted batches stay virtual; the first accepted batch
    walks past every virtual ancestor and publishes cleanly."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 3)
        store = _S3Store()
        coordinator = SinkEffectCoordinator(factory=factory, worker_id="worker-a")

        first = coordinator.execute(
            _execution_request(run_id, sink_id, members[:1]),
            _s3(store, max_record_chars=10),
        )
        assert first.effect.publication_evidence_kind == "virtual"

        # Batch 2 diverts everything as well: with no real publication behind
        # it, it must finalize virtual instead of wedging on batch 1.
        second = coordinator.execute(
            _execution_request(run_id, sink_id, members[1:2]),
            _s3(store, max_record_chars=10),
        )
        assert second.effect.publication_performed is False
        assert second.effect.publication_evidence_kind == "virtual"
        assert store.value is None

        third = coordinator.execute(
            _execution_request(run_id, sink_id, members[2:]),
            _s3(store),
        )

        assert third.effect.publication_performed is True
        assert store.value is not None
        assert json.loads(store.value.body) == [{"ordinal": 2}]
        put_requests = [request for request in store.requests if request["operation"] == "put"]
        assert len(put_requests) == 1
        assert put_requests[0]["IfNoneMatch"] == "*"
    finally:
        db.close()


def test_real_publication_survives_inherited_no_publication_gap() -> None:
    """Fail-closed predecessor evidence is preserved: a successor behind an
    inherited (no-publication) gap still fences on the last real bytes."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        # The middle record alone exceeds the successor's size cap, so only
        # batch 2 diverts while the snapshot rows stay serializable.
        run_id, sink_id, members = _members_with_payloads(
            factory,
            [{"ordinal": 0}, {"ordinal": 1, "pad": "x" * 100}, {"ordinal": 2}],
        )
        store = _S3Store()
        coordinator = SinkEffectCoordinator(factory=factory, worker_id="worker-a")

        first = coordinator.execute(
            _execution_request(run_id, sink_id, members[:1]),
            _s3(store),
        )
        assert first.effect.publication_performed is True
        real_etag = store.value.etag if store.value is not None else None
        assert real_etag is not None

        # Batch 2 diverts everything; the cumulative body equals the real
        # publication so the effect finalizes inherited without a write.
        second = coordinator.execute(
            _execution_request(run_id, sink_id, members[1:2]),
            _s3(store, max_record_chars=60),
        )
        assert second.effect.publication_performed is False
        assert second.effect.publication_evidence_kind == "inherited"
        assert store.value is not None and store.value.etag == real_etag

        third = coordinator.execute(
            _execution_request(run_id, sink_id, members[2:]),
            _s3(store),
        )

        assert third.effect.publication_performed is True
        assert json.loads(store.value.body) == [{"ordinal": 0}, {"ordinal": 2}]
        put_requests = [request for request in store.requests if request["operation"] == "put"]
        assert len(put_requests) == 2
        # The successor's replace is fenced on the real publication's ETag.
        assert put_requests[1]["IfMatch"] == real_etag
    finally:
        db.close()


def test_foreign_overwrite_after_inherited_gap_fails_predecessor_fence() -> None:
    """Tamper variant: the declared-predecessor byte fence must hold across an
    inherited no-publication gap. If a foreign actor overwrites the remote
    object after the gap, the successor must fail closed on the last real
    publication's bytes rather than replace the tampered object fenced only
    on its observed ETag."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        # The middle record alone exceeds the successor's size cap, so only
        # batch 2 diverts while the snapshot rows stay serializable.
        run_id, sink_id, members = _members_with_payloads(
            factory,
            [{"ordinal": 0}, {"ordinal": 1, "pad": "x" * 100}, {"ordinal": 2}],
        )
        store = _S3Store()
        coordinator = SinkEffectCoordinator(factory=factory, worker_id="worker-a")

        first = coordinator.execute(
            _execution_request(run_id, sink_id, members[:1]),
            _s3(store),
        )
        assert first.effect.publication_performed is True

        # Batch 2 diverts everything and finalizes inherited without a write.
        second = coordinator.execute(
            _execution_request(run_id, sink_id, members[1:2]),
            _s3(store, max_record_chars=60),
        )
        assert second.effect.publication_evidence_kind == "inherited"

        # A FOREIGN actor overwrites the object between batches.
        tampered_body = b'[{"id": "tampered"}]'
        store.value = _Object(tampered_body, '"etag-foreign"', {})

        with pytest.raises(
            RemoteObjectPreconditionError,
            match="declared predecessor bytes do not match remote metadata",
        ):
            coordinator.execute(
                _execution_request(run_id, sink_id, members[2:]),
                _s3(store),
            )

        # Fail-closed: the tampered object was never replaced.
        assert store.value is not None and store.value.body == tampered_body
        put_requests = [request for request in store.requests if request["operation"] == "put"]
        assert len(put_requests) == 1
    finally:
        db.close()
