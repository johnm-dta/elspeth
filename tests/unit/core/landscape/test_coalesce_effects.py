"""Durable coalesce-effect identity and replay regressions (epoch 27)."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import NodeStateStatus, NodeType
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.engine import CoalesceParentCompletion
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.schema import (
    coalesce_effect_members_table,
    coalesce_effects_table,
    node_states_table,
    token_outcomes_table,
    token_parents_table,
    tokens_table,
)
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

_RUN_ID = "run-1"
_COALESCE_NODE_ID = "coalesce-0"
_CONTRACT = SchemaContract(mode="OBSERVED", fields=(), locked=True)


def _setup():
    setup = make_recorder_with_run(run_id=_RUN_ID)
    register_test_node(
        setup.data_flow,
        setup.run_id,
        _COALESCE_NODE_ID,
        node_type=NodeType.COALESCE,
        plugin_name="coalesce",
    )
    row = setup.data_flow.create_row(
        run_id=setup.run_id,
        source_node_id=setup.source_node_id,
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"source": True},
    )
    parents = [setup.data_flow.create_token(row.row_id) for _ in range(2)]
    refs = tuple(TokenRef(token_id=token.token_id, run_id=setup.run_id) for token in parents)
    completions: list[CoalesceParentCompletion] = []
    for ordinal, ref in enumerate(refs):
        state = setup.execution.begin_node_state(
            token_id=ref.token_id,
            node_id=_COALESCE_NODE_ID,
            run_id=setup.run_id,
            step_index=4,
            input_data={"ordinal": ordinal},
        )
        completions.append(
            CoalesceParentCompletion(
                parent_ref=ref,
                state_id=state.state_id,
                duration_ms=float(ordinal + 1),
                context_after=None,
            )
        )
    return setup, row, refs, tuple(completions)


def _materialize(setup, row, refs, completions=None):
    return setup.data_flow.coalesce_tokens(
        parent_refs=list(refs),
        row_id=row.row_id,
        coalesce_node_id=_COALESCE_NODE_ID,
        parent_state_ids=None if completions is None else [item.state_id for item in completions],
        merged_payload={"merged": True},
        merged_contract=_CONTRACT,
        step_in_pipeline=4,
    )


def test_materialization_is_idempotent_and_normalizes_parent_evidence() -> None:
    setup, row, refs, completions = _setup()

    first = _materialize(setup, row, refs, completions)
    second = _materialize(setup, row, refs, completions)

    assert second.token_id == first.token_id
    assert second.join_group_id == first.join_group_id
    with setup.db.connection() as conn:
        effect = conn.execute(select(coalesce_effects_table)).mappings().one()
        members = conn.execute(select(coalesce_effect_members_table).order_by(coalesce_effect_members_table.c.ordinal)).mappings().all()
        outcome_count = conn.execute(select(func.count()).select_from(token_outcomes_table)).scalar_one()
        states = conn.execute(
            select(node_states_table.c.state_id, node_states_table.c.status)
            .where(node_states_table.c.state_id.in_([item.state_id for item in completions]))
            .order_by(node_states_table.c.state_id)
        ).all()

    assert effect["status"] == "materialized"
    assert effect["result_token_id"] == first.token_id
    assert effect["result_join_group_id"] == first.join_group_id
    assert [(item["parent_token_id"], item["parent_state_id"]) for item in members] == [
        (completion.parent_ref.token_id, completion.state_id) for completion in completions
    ]
    assert outcome_count == 0
    assert {state.status for state in states} == {NodeStateStatus.OPEN.value}


def test_finalization_atomically_completes_states_outcomes_and_effect() -> None:
    setup, row, refs, completions = _setup()
    merged = _materialize(setup, row, refs, completions)

    setup.data_flow.finalize_coalesce_effect(merged=merged, parent_completions=completions)
    setup.data_flow.finalize_coalesce_effect(merged=merged, parent_completions=completions)

    with setup.db.connection() as conn:
        effect = conn.execute(select(coalesce_effects_table)).mappings().one()
        states = conn.execute(
            select(node_states_table.c.state_id, node_states_table.c.status).where(
                node_states_table.c.state_id.in_([item.state_id for item in completions])
            )
        ).all()
        outcomes = conn.execute(
            select(
                token_outcomes_table.c.token_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.join_group_id,
            ).order_by(token_outcomes_table.c.token_id)
        ).all()

    assert effect["status"] == "completed"
    assert effect["completed_at"] is not None
    assert {state.status for state in states} == {NodeStateStatus.COMPLETED.value}
    assert outcomes == sorted(
        [(ref.token_id, TerminalOutcome.SUCCESS.value, TerminalPath.COALESCED.value, merged.join_group_id) for ref in refs]
    )


def test_failed_finalization_rolls_back_all_terminal_evidence_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    setup, row, refs, completions = _setup()
    merged = _materialize(setup, row, refs, completions)
    original = setup.data_flow.outcomes.record_token_outcome
    calls = 0

    def fail_after_first_outcome(*args, **kwargs):
        nonlocal calls
        outcome_id = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            raise RuntimeError("injected coalesce finalization failure")
        return outcome_id

    monkeypatch.setattr(setup.data_flow.outcomes, "record_token_outcome", fail_after_first_outcome)

    with pytest.raises(AuditIntegrityError, match="finalization failure"):
        setup.data_flow.finalize_coalesce_effect(merged=merged, parent_completions=completions)

    with setup.db.connection() as conn:
        effect_status = conn.execute(select(coalesce_effects_table.c.status)).scalar_one()
        state_statuses = (
            conn.execute(
                select(node_states_table.c.status).where(node_states_table.c.state_id.in_([item.state_id for item in completions]))
            )
            .scalars()
            .all()
        )
        outcome_count = conn.execute(select(func.count()).select_from(token_outcomes_table)).scalar_one()
    assert effect_status == "materialized"
    assert set(state_statuses) == {NodeStateStatus.OPEN.value}
    assert outcome_count == 0

    monkeypatch.setattr(setup.data_flow.outcomes, "record_token_outcome", original)
    setup.data_flow.finalize_coalesce_effect(merged=merged, parent_completions=completions)

    with setup.db.connection() as conn:
        assert conn.execute(select(coalesce_effects_table.c.status)).scalar_one() == "completed"
        assert conn.execute(select(func.count()).select_from(token_outcomes_table)).scalar_one() == len(refs)


def test_same_parent_set_with_different_order_or_state_mapping_fails_closed() -> None:
    setup, row, refs, completions = _setup()
    _materialize(setup, row, refs, completions)

    with pytest.raises(AuditIntegrityError, match="ordered parent sequence"):
        _materialize(setup, row, tuple(reversed(refs)), tuple(reversed(completions)))

    forged = (
        replace(completions[0], state_id=completions[1].state_id),
        replace(completions[1], state_id=completions[0].state_id),
    )
    merged = _materialize(setup, row, refs, completions)
    with pytest.raises(AuditIntegrityError, match="parent/state membership"):
        setup.data_flow.finalize_coalesce_effect(merged=merged, parent_completions=forged)


def test_effect_result_and_member_evidence_are_mechanically_constrained() -> None:
    effect_constraint_names = {constraint.name for constraint in coalesce_effects_table.constraints}
    member_constraint_names = {constraint.name for constraint in coalesce_effect_members_table.constraints}
    token_index_names = {index.name for index in tokens_table.indexes}
    state_index_names = {index.name for index in node_states_table.indexes}

    assert "ck_coalesce_effects_lifecycle" in effect_constraint_names
    assert "fk_coalesce_effects_result_identity" in effect_constraint_names
    assert "uq_coalesce_effect_members_token" in member_constraint_names
    assert "uq_coalesce_effect_members_state" in member_constraint_names
    assert "fk_coalesce_effect_members_state_token" in member_constraint_names
    assert "uq_tokens_coalesce_result_identity" in token_index_names
    assert "uq_node_states_coalesce_member_identity" in state_index_names


def test_raw_parent_evidence_cannot_duplicate_token_or_state() -> None:
    setup, row, refs, _completions = _setup()
    merged = _materialize(setup, row, refs)
    with setup.db.connection() as conn:
        effect_id = conn.execute(select(coalesce_effects_table.c.effect_id)).scalar_one()
        assert conn.execute(
            select(func.count()).select_from(token_parents_table).where(token_parents_table.c.token_id == merged.token_id)
        ).scalar_one() == len(refs)

    with pytest.raises(IntegrityError), setup.db.write_connection() as conn:
        first = (
            conn.execute(select(coalesce_effect_members_table).where(coalesce_effect_members_table.c.effect_id == effect_id))
            .mappings()
            .first()
        )
        assert first is not None
        conn.execute(
            coalesce_effect_members_table.insert().values(
                effect_id=effect_id,
                run_id=_RUN_ID,
                ordinal=99,
                parent_token_id=first["parent_token_id"],
                parent_state_id=first["parent_state_id"],
            )
        )
