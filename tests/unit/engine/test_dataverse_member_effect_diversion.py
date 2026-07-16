"""Durable row-error diversion for Dataverse member effects.

Regression coverage for elspeth-d88f8eee34: a member whose PATCH receives a
non-retryable row_data_error response must be durably classified and diverted
(with attribution), not re-PATCHed indefinitely by commit/reconcile cycles.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import replace
from datetime import timedelta
from typing import Any

from elspeth.contracts import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.sink_effects import (
    SinkEffectFinalizationMember,
    SinkEffectMemberCandidate,
    SinkEffectPipelineMembersInput,
    SinkEffectState,
)
from elspeth.core.landscape.execution.sink_effect_identity import (
    compute_pipeline_effect_identity,
    resolve_sink_effect_members,
)
from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionRequest,
    SinkEffectExecutionSeam,
    SinkEffectInjectedFault,
)
from elspeth.plugins.infrastructure.clients.dataverse import DataverseClientError, DataversePageResponse
from elspeth.plugins.sinks.dataverse import DataverseSink
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.landscape import make_factory, make_landscape_db, register_test_node
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_request

_CONFIG: dict[str, Any] = {
    "environment_url": "https://myorg.crm.dynamics.com",
    "auth": {
        "method": "service_principal",
        "tenant_id": "tenant-1",
        "client_id": "client-1",
        "client_secret": "secret-1",
    },
    "entity": "contacts",
    "alternate_key": "emailaddress1",
    "field_mapping": {"email": "emailaddress1", "name": "fullname"},
    "schema": {"mode": "observed"},
}

_BAD_KEY = "bad@example.com"


class _RowRejectingClient:
    """Fake Dataverse client: one key's PATCH is rejected as row data error."""

    def __init__(self) -> None:
        self.upserted: dict[str, int] = {}
        self.stored: set[str] = set()

    @staticmethod
    def _key_from_url(url: str) -> str:
        start = url.index("emailaddress1='") + len("emailaddress1='")
        return urllib.parse.unquote(url[start : url.index("'", start)])

    def upsert(self, url: str, payload: dict[str, Any]) -> None:
        del payload
        key = self._key_from_url(url)
        self.upserted[key] = self.upserted.get(key, 0) + 1
        if key == _BAD_KEY:
            raise DataverseClientError(
                "Bad request: a validation error occurred for this record",
                retryable=False,
                status_code=400,
                error_category="row_data_error",
            )
        self.stored.add(key)

    def get_page(self, url: str) -> DataversePageResponse:
        key = self._key_from_url(url)
        if key not in self.stored:
            raise DataverseClientError("Not found", retryable=False, status_code=404)
        return DataversePageResponse(
            status_code=200,
            rows=[{"emailaddress1": key}],
            latency_ms=1.0,
            headers={},
            request_headers={},
            request_url=url,
            next_link=None,
            paging_cookie=None,
            more_records=None,
        )


def _build_request(factory, run_id: str, source_id: str, sink_id: str) -> SinkEffectExecutionRequest:
    rows = [
        {"email": "a@example.com", "name": "Alice"},
        {"email": _BAD_KEY, "name": "Bad"},
        {"email": "c@example.com", "name": "Carol"},
    ]
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal, payload in enumerate(rows):
        row = factory.data_flow.create_row(
            run_id=run_id,
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
            run_id=run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    members = resolve_sink_effect_members(factory, candidates)
    reservation = _pipeline_request(run_id, sink_id, members)
    identity = compute_pipeline_effect_identity(
        run_id=run_id,
        sink_node_id=sink_id,
        role=reservation.role,
        sink_config={"name": "dataverse"},
        target_config={"entity": "contacts"},
        members=tuple(replace(member, member_effect_id=None) for member in members),
    )
    return SinkEffectExecutionRequest(
        reservation=reservation,
        effect_input=SinkEffectPipelineMembersInput(identity.members, identity.members),
        finalization_members=tuple(
            SinkEffectFinalizationMember(
                ordinal=member.ordinal,
                output_data=dict(member.row),
                duration_ms=0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="dataverse",
            )
            for member in identity.members
        ),
    )


def _make_sink(client: _RowRejectingClient) -> DataverseSink:
    sink = inject_write_failure(DataverseSink(dict(_CONFIG)))
    sink._client = client  # type: ignore[assignment]
    return sink


def test_row_data_error_member_diverts_durably_instead_of_retrying_forever() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="dataverse")
        request = _build_request(factory, run.run_id, source_id, sink_id)
        client = _RowRejectingClient()
        sink = _make_sink(client)

        result = SinkEffectCoordinator(
            factory=factory,
            worker_id="worker-a",
            lease_ttl=timedelta(minutes=5),
        ).execute(request, sink)

        # The group finalized: valid siblings landed, the rejected member did not.
        assert result.effect.state is SinkEffectState.FINALIZED
        assert client.stored == {"a@example.com", "c@example.com"}
        # The rejected PATCH was sent exactly once — never retried.
        assert client.upserted[_BAD_KEY] == 1

        # Durable partition: ordinal 1 diverted with attribution, siblings accepted.
        effect_id = result.effect.effect_id
        members = factory.execution.sink_effects.get_members(effect_id)
        assert [member.prepared_disposition for member in members] == ["accepted", "diverted", "accepted"]
        assert all(member.member_state is SinkEffectState.FINALIZED for member in members)
        diverted = members[1]
        assert diverted.reason_hash is not None

        # Live diversion log carries the real row-attributable reason.
        live = sink._get_diversions()
        assert [item.row_index for item in live] == [1]
        assert "400" in live[0].reason or "validation" in live[0].reason
    finally:
        db.close()


def test_diverted_member_recovers_from_durable_result_without_repatching() -> None:
    """Crash between member commits and finalization: recovery must reuse the
    durable diverted result instead of re-sending the rejected PATCH."""
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="dataverse")
        request = _build_request(factory, run.run_id, source_id, sink_id)
        client = _RowRejectingClient()

        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(observed)

        first_sink = _make_sink(client)
        try:
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                lease_ttl=timedelta(minutes=5),
                fault_hook=fail_once,
            ).execute(request, first_sink)
            raise AssertionError("expected the injected fault to interrupt the first execution")
        except SinkEffectInjectedFault:
            pass

        # Fresh process: new sink instance (empty live diversion log).
        recovered_sink = _make_sink(client)
        result = SinkEffectCoordinator(
            factory=make_factory(db),
            worker_id="worker-a",
            lease_ttl=timedelta(minutes=5),
        ).execute(request, recovered_sink)

        assert result.effect.state is SinkEffectState.FINALIZED
        # The rejected PATCH was never re-sent during recovery.
        assert client.upserted[_BAD_KEY] == 1
        members = factory.execution.sink_effects.get_members(result.effect.effect_id)
        assert [member.prepared_disposition for member in members] == ["accepted", "diverted", "accepted"]
    finally:
        db.close()
