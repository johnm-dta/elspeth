"""Durable coordinator behavior for per-record remote sink effects."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update

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
from elspeth.core.landscape.schema import sink_effects_table
from elspeth.engine.executors.sink_effects import SinkEffectCoordinator, SinkEffectExecutionRequest
from elspeth.plugins.sinks.chroma_sink import ChromaSink
from tests.fixtures.landscape import make_factory, make_landscape_db, register_test_node
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_request


class _ResponseLosingCollection:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[str, dict[str, object] | None]] = {}
        self.upsert_count_by_id: dict[str, int] = {}
        self.lose_first_response = True

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, object]] | None,
    ) -> None:
        assert len(ids) == len(documents) == 1
        document_id = ids[0]
        metadata = None if metadatas is None else dict(metadatas[0])
        self.documents[document_id] = (documents[0], metadata)
        self.upsert_count_by_id[document_id] = self.upsert_count_by_id.get(document_id, 0) + 1
        if self.lose_first_response:
            self.lose_first_response = False
            raise RuntimeError("response lost after Chroma upsert")

    def get(self, *, ids: list[str], include: list[str] | None = None) -> dict[str, object]:
        del include
        found = [document_id for document_id in ids if document_id in self.documents]
        return {
            "ids": found,
            "documents": [self.documents[document_id][0] for document_id in found],
            "metadatas": [self.documents[document_id][1] for document_id in found],
        }


def _chroma_config() -> dict[str, object]:
    return {
        "collection": "durable-members",
        "mode": "persistent",
        "persist_directory": "./unused",
        "field_mapping": {
            "document_field": "text",
            "id_field": "doc_id",
            "metadata_fields": ["topic"],
        },
        "on_duplicate": "overwrite",
        "schema": {
            "mode": "fixed",
            "fields": ["doc_id: str", "text: str", "topic: str"],
        },
    }


@pytest.mark.parametrize("takeover", [False, True])
def test_response_lost_member_is_reconciled_and_only_missing_members_are_committed(takeover: bool) -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="chroma_sink")
        candidates: list[SinkEffectMemberCandidate] = []
        for ordinal in range(3):
            payload = {"doc_id": f"d{ordinal}", "text": f"document {ordinal}", "topic": "test"}
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
        members = resolve_sink_effect_members(factory, candidates)
        reservation = _pipeline_request(run.run_id, sink_id, members)
        identity = compute_pipeline_effect_identity(
            run_id=run.run_id,
            sink_node_id=sink_id,
            role=reservation.role,
            sink_config={"name": "chroma_sink"},
            target_config={"collection": "durable-members"},
            members=tuple(replace(member, member_effect_id=None) for member in members),
        )
        request = SinkEffectExecutionRequest(
            reservation=reservation,
            effect_input=SinkEffectPipelineMembersInput(identity.members, identity.members),
            finalization_members=tuple(
                SinkEffectFinalizationMember(
                    ordinal=member.ordinal,
                    output_data=dict(member.row),
                    duration_ms=0,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.DEFAULT_FLOW,
                    sink_name="chroma_sink",
                )
                for member in identity.members
            ),
        )
        collection = _ResponseLosingCollection()
        first_sink = ChromaSink(_chroma_config())
        first_sink._collection = collection  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="response lost"):
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                lease_ttl=timedelta(minutes=5),
            ).execute(request, first_sink)

        durable_binding = factory.execution.sink_effects.get_members_for_tokens(
            run_id=run.run_id,
            sink_node_id=sink_id,
            role=reservation.role,
            token_ids=tuple(member.token_id for member in members),
        )
        effect_id = durable_binding[0].effect_id
        durable_after_loss = factory.execution.sink_effects.get_members(effect_id)
        assert [member.member_state for member in durable_after_loss] == [
            SinkEffectState.IN_FLIGHT,
            SinkEffectState.PREPARED,
            SinkEffectState.PREPARED,
        ]
        if takeover:
            expired_at = datetime.now(UTC) - timedelta(seconds=1)
            with db.engine.begin() as conn:
                conn.execute(
                    update(sink_effects_table)
                    .where(sink_effects_table.c.effect_id == effect_id)
                    .values(
                        lease_heartbeat_at=expired_at - timedelta(seconds=1),
                        lease_expires_at=expired_at,
                    )
                )

        recovered_sink = ChromaSink(_chroma_config())
        recovered_sink._collection = collection  # type: ignore[assignment]
        result = SinkEffectCoordinator(
            factory=make_factory(db),
            worker_id="worker-b" if takeover else "worker-a",
        ).execute(request, recovered_sink)

        assert result.effect.state is SinkEffectState.FINALIZED
        assert collection.upsert_count_by_id == {"d0": 1, "d1": 1, "d2": 1}
        assert all(member.member_state is SinkEffectState.FINALIZED for member in factory.execution.sink_effects.get_members(effect_id))
    finally:
        db.close()
