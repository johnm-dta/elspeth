# tests/unit/core/landscape/test_where_exactness_consolidated.py
"""Consolidated WHERE clause exactness tests for all landscape query methods.

Verifies that SQL queries use ``==`` (exact match) rather than ``>=`` / ``<=``
(range) operators.  The multi-run fixture creates three runs with
lexicographically ordered IDs (run-A < run-B < run-C) so that an inequality
operator would silently include data from adjacent runs.

One definitive test per query method — queries run-B and asserts only run-B
data is returned.  Both variants are kept for:

- ``find_call_by_request_hash`` — LLM cache safety requires both "returns
  target" and "same hash different run returns None" (cross-run leak).
- ``get_node`` — composite PK test needs both "returns target" and "mismatched
  run/node returns None".
- ``get_token_outcomes_for_row`` — composite PK test needs both "returns
  target" and "cross-run row ID returns empty".
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from elspeth.contracts import (
    CallStatus,
    CallType,
    ExportStatus,
    RunStatus,
    SecretResolutionInput,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema_contract import FieldContract, SchemaContract

if TYPE_CHECKING:
    from tests.fixtures.multi_run import MultiRunFixture

pytest_plugins = ["tests.fixtures.multi_run"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_secret_resolution(env_var: str, fingerprint_byte: str = "a") -> SecretResolutionInput:
    return SecretResolutionInput(
        env_var_name=env_var,
        source="keyvault",
        vault_url="https://test.vault.azure.net",
        secret_name=f"secret-{env_var}",
        timestamp=time.time(),
        resolution_latency_ms=1.0,
        fingerprint=fingerprint_byte * 64,
    )


def _make_schema_contract() -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name="val",
                original_name="val",
                python_type=str,
                required=True,
                source="inferred",
            ),
        ),
        locked=True,
    )


# ===========================================================================
# Run lifecycle methods
# ===========================================================================


class TestGetRunWhereExactness:
    """get_run must return exactly the target run, not adjacent ones."""

    def test_returns_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        run = fix.factory.run_lifecycle.get_run("run-B")

        assert run is not None
        assert run.run_id == "run-B"
        # Adjacent runs exist but are distinct — inequality would merge them
        run_a = fix.factory.run_lifecycle.get_run("run-A")
        assert run_a is not None
        assert run_a.run_id == "run-A"
        run_c = fix.factory.run_lifecycle.get_run("run-C")
        assert run_c is not None
        assert run_c.run_id == "run-C"


class TestCompleteRunWhereExactness:
    """complete_run must update only the target run's status."""

    def test_completes_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.complete_run("run-B", RunStatus.COMPLETED)

        run_b = fix.factory.run_lifecycle.get_run("run-B")
        assert run_b is not None
        assert run_b.status == RunStatus.COMPLETED
        # Adjacent runs must be unaffected
        run_a = fix.factory.run_lifecycle.get_run("run-A")
        assert run_a is not None
        assert run_a.status == RunStatus.RUNNING
        run_c = fix.factory.run_lifecycle.get_run("run-C")
        assert run_c is not None
        assert run_c.status == RunStatus.RUNNING


class TestUpdateRunStatusWhereExactness:
    """update_run_status must transition only the target run."""

    def test_updates_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.update_run_status("run-B", RunStatus.INTERRUPTED)

        run_b = fix.factory.run_lifecycle.get_run("run-B")
        assert run_b is not None
        assert run_b.status == RunStatus.INTERRUPTED
        run_a = fix.factory.run_lifecycle.get_run("run-A")
        assert run_a is not None
        assert run_a.status == RunStatus.RUNNING
        run_c = fix.factory.run_lifecycle.get_run("run-C")
        assert run_c is not None
        assert run_c.status == RunStatus.RUNNING


class TestUpdateRunContractWhereExactness:
    """update_run_contract must set contract only on the target run."""

    def test_updates_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        contract = _make_schema_contract()

        fix.factory.run_lifecycle.update_run_contract("run-B", contract)

        assert fix.factory.run_lifecycle.get_run_contract("run-B") is not None
        assert fix.factory.run_lifecycle.get_run_contract("run-A") is None
        assert fix.factory.run_lifecycle.get_run_contract("run-C") is None


class TestGetSourceFieldResolutionWhereExactness:
    """get_source_field_resolution must return resolution for the target run only."""

    def test_returns_only_target_run_resolution(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.record_source_field_resolution("run-B", {"Original Header": "original_header"}, "v1")

        resolution = fix.factory.run_lifecycle.get_source_field_resolution("run-B")
        assert resolution is not None
        assert resolution == {"Original Header": "original_header"}
        assert fix.factory.run_lifecycle.get_source_field_resolution("run-A") is None
        assert fix.factory.run_lifecycle.get_source_field_resolution("run-C") is None


class TestSetExportStatusWhereExactness:
    """set_export_status must update only the target run."""

    def test_updates_only_target_run(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.set_export_status("run-B", ExportStatus.COMPLETED)

        run_b = fix.factory.run_lifecycle.get_run("run-B")
        assert run_b is not None
        assert run_b.export_status == ExportStatus.COMPLETED
        run_a = fix.factory.run_lifecycle.get_run("run-A")
        assert run_a is not None
        assert run_a.export_status != ExportStatus.COMPLETED
        run_c = fix.factory.run_lifecycle.get_run("run-C")
        assert run_c is not None
        assert run_c.export_status != ExportStatus.COMPLETED


class TestGetSecretResolutionsForRunWhereExactness:
    """get_secret_resolutions_for_run must return resolutions for the target run only."""

    def test_returns_only_target_run_resolutions(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.record_secret_resolutions("run-A", [_make_secret_resolution("KEY_A", "a")])
        fix.factory.run_lifecycle.record_secret_resolutions("run-B", [_make_secret_resolution("KEY_B", "b")])
        fix.factory.run_lifecycle.record_secret_resolutions("run-C", [_make_secret_resolution("KEY_C", "c")])

        resolutions = fix.factory.run_lifecycle.get_secret_resolutions_for_run("run-B")

        assert len(resolutions) == 1
        assert resolutions[0].env_var_name == "KEY_B"
        assert resolutions[0].run_id == "run-B"


class TestListRunsWhereExactness:
    """list_runs with status filter must return only runs matching that status."""

    def test_filters_by_exact_status(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        fix.factory.run_lifecycle.complete_run("run-B", RunStatus.COMPLETED)

        completed_runs = fix.factory.run_lifecycle.list_runs(status=RunStatus.COMPLETED)
        assert len(completed_runs) == 1
        assert completed_runs[0].run_id == "run-B"

        running_runs = fix.factory.run_lifecycle.list_runs(status=RunStatus.RUNNING)
        assert len(running_runs) == 2
        assert {r.run_id for r in running_runs} == {"run-A", "run-C"}

    def test_unfiltered_returns_all(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        all_runs = fix.factory.run_lifecycle.list_runs()

        assert len(all_runs) == 3
        assert {r.run_id for r in all_runs} == {"run-A", "run-B", "run-C"}


# ===========================================================================
# Data flow methods — run-scoped collections
# ===========================================================================


class TestGetRowsWhereExactness:
    """get_rows must return only rows belonging to the target run."""

    def test_returns_only_target_run_rows(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        rows = fix.factory.query.get_rows(target.run_id)

        assert len(rows) == 2
        assert all(r.run_id == target.run_id for r in rows)
        assert {r.row_id for r in rows} == set(target.row_ids)


class TestGetAllTokensForRunWhereExactness:
    """get_all_tokens_for_run must scope tokens to the target run only."""

    def test_returns_only_target_run_tokens(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        tokens = fix.factory.query.get_all_tokens_for_run(target.run_id)

        assert len(tokens) == 2
        assert {t.token_id for t in tokens} == {t.token_id for t in target.tokens}


class TestGetAllNodeStatesForRunWhereExactness:
    """get_all_node_states_for_run must scope states to the target run only."""

    def test_returns_only_target_run_states(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        states = fix.factory.query.get_all_node_states_for_run(target.run_id)

        assert len(states) == 2
        assert {s.state_id for s in states} == {t.state_id for t in target.tokens}


class TestGetAllCallsForRunWhereExactness:
    """get_all_calls_for_run must scope calls to the target run only."""

    def test_returns_only_target_run_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        calls = fix.factory.query.get_all_calls_for_run(target.run_id)

        # Only the first token per run has a call
        expected_call_ids = {t.call_id for t in target.tokens if t.call_id is not None}
        assert len(calls) == 1
        assert {c.call_id for c in calls} == expected_call_ids


class TestGetAllRoutingEventsForRunWhereExactness:
    """get_all_routing_events_for_run must scope events to the target run only."""

    def test_returns_only_target_run_events(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        events = fix.factory.query.get_all_routing_events_for_run(target.run_id)

        expected_re_ids = {t.routing_event_id for t in target.tokens if t.routing_event_id is not None}
        assert len(events) == 1
        assert {e.event_id for e in events} == expected_re_ids


class TestGetAllTokenOutcomesForRunWhereExactness:
    """get_all_token_outcomes_for_run must scope outcomes to the target run only."""

    def test_returns_only_target_run_outcomes(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        outcomes = fix.factory.query.get_all_token_outcomes_for_run(target.run_id)

        assert len(outcomes) == 2
        assert {o.token_id for o in outcomes} == {t.token_id for t in target.tokens}


class TestGetBatchesWhereExactness:
    """get_batches must scope batches to the target run only."""

    def test_returns_only_target_run_batches(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        batches = fix.factory.execution.get_batches(target.run_id)

        assert len(batches) == 1
        assert batches[0].batch_id == target.batch_id


# ===========================================================================
# Data flow methods — graph structure
# ===========================================================================


class TestGetNodeWhereExactness:
    """get_node must return exactly the target node by composite PK (node_id, run_id)."""

    def test_returns_target_node(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        node = fix.factory.data_flow.get_node(target.source_node_id, target.run_id)

        assert node is not None
        assert node.node_id == target.source_node_id
        assert node.run_id == target.run_id

    def test_mismatched_run_id_returns_none(self, multi_run_landscape: MultiRunFixture) -> None:
        """Both dimensions of the composite PK must be exact.

        src-B exists in run-B but querying it with run-A's run_id must return
        None — verifies the run_id filter is ``==``, not ``>=``/``<=``.
        """
        fix = multi_run_landscape
        run_a = fix.run("A")
        run_b = fix.run("B")

        # run-B's node with run-A's run_id — must not find anything
        result = fix.factory.data_flow.get_node(run_b.source_node_id, run_a.run_id)
        assert result is None

        # run-A's node with run-B's run_id — must not find anything
        result2 = fix.factory.data_flow.get_node(run_a.source_node_id, run_b.run_id)
        assert result2 is None


class TestGetEdgesWhereExactness:
    """get_edges must return only edges for the target run."""

    def test_returns_only_target_run_edges(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        edges = fix.factory.data_flow.get_edges(target.run_id)

        assert len(edges) == 2
        assert {e.edge_id for e in edges} == {
            target.edge_id_source_to_transform,
            target.edge_id_transform_to_sink,
        }


# ===========================================================================
# Data flow methods — token-scoped lookups
# ===========================================================================


class TestGetTokenOutcomeWhereExactness:
    """get_token_outcome must return exactly the outcome for the target token."""

    def test_returns_target_token_outcome(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        tok = fix.run("B").tokens[0]

        outcome = fix.factory.data_flow.get_token_outcome(tok.token_id)

        assert outcome is not None
        assert outcome.token_id == tok.token_id


class TestGetTokenOutcomesForRowWhereExactness:
    """get_token_outcomes_for_row must scope by both run_id and row_id (composite PK)."""

    def test_returns_only_target_row_outcomes(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        outcomes = fix.factory.data_flow.get_token_outcomes_for_row(target.run_id, target.row_ids[0])

        assert len(outcomes) == 1
        assert outcomes[0].token_id == target.tokens[0].token_id

    def test_cross_run_row_id_returns_empty(self, multi_run_landscape: MultiRunFixture) -> None:
        """row-A-0 exists in run-A but must not appear when queried with run-B.

        Verifies the run_id WHERE clause is ``==``, not ``>=``/``<=``.
        """
        fix = multi_run_landscape

        outcomes = fix.factory.data_flow.get_token_outcomes_for_row("run-B", fix.run("A").row_ids[0])

        assert len(outcomes) == 0


# ===========================================================================
# Data flow methods — validation and transform errors
# ===========================================================================


class TestValidationErrorsWhereExactness:
    """record_validation_error + get_validation_errors_for_run must scope by run."""

    def test_returns_only_target_run_errors(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        for suffix in ("A", "B", "C"):
            run = fix.run(suffix)
            fix.factory.data_flow.record_validation_error(
                run_id=run.run_id,
                node_id=run.source_node_id,
                row_data={"bad_field": f"val-{suffix}"},
                error=f"schema violation in {suffix}",
                schema_mode="fixed",
                destination="discard",
            )

        errors = fix.factory.data_flow.get_validation_errors_for_run("run-B")

        assert len(errors) == 1
        assert errors[0].run_id == "run-B"
        assert errors[0].error == "schema violation in B"


class TestTransformErrorsWhereExactness:
    """record_transform_error + get_transform_errors_for_run must scope by run."""

    def test_returns_only_target_run_errors(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        for suffix in ("A", "B", "C"):
            run = fix.run(suffix)
            tok = run.tokens[0]
            fix.factory.data_flow.record_transform_error(
                ref=TokenRef(token_id=tok.token_id, run_id=run.run_id),
                transform_id=run.transform_node_id,
                row_data={"val": f"err-{suffix}"},
                error_details={"reason": "test_error"},
                destination="discard",
            )

        errors = fix.factory.data_flow.get_transform_errors_for_run("run-B")

        assert len(errors) == 1
        assert errors[0].run_id == "run-B"
        assert errors[0].token_id == fix.run("B").tokens[0].token_id


# ===========================================================================
# Execution methods — node states
# ===========================================================================


class TestGetNodeStateWhereExactness:
    """get_node_state must return only the targeted state, not neighbours."""

    def test_returns_only_target_state(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target_state_id = fix.run("B").tokens[0].state_id  # st-B-0

        state = fix.factory.execution.get_node_state(target_state_id)

        assert state is not None
        assert state.state_id == target_state_id


class TestGetNodeStatesForTokenWhereExactness:
    """get_node_states_for_token must return only states for the target token."""

    def test_returns_only_target_token_states(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target_token = fix.run("B").tokens[0]

        states = fix.factory.query.get_node_states_for_token(target_token.token_id)

        assert len(states) == 1
        assert states[0].state_id == target_token.state_id


# ===========================================================================
# Execution methods — calls
# ===========================================================================


class TestGetCallsWhereExactness:
    """get_calls (state-scoped) must return only calls for the target state."""

    def test_returns_only_target_state_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        target_state_id = target.tokens[0].state_id  # st-B-0 has a call

        calls = fix.factory.query.get_calls(target_state_id)

        assert len(calls) == 1
        assert calls[0].call_id == target.tokens[0].call_id

    def test_returns_empty_for_state_without_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        """Second token in each run has no call — verify empty, not leaking."""
        fix = multi_run_landscape

        calls = fix.factory.query.get_calls(fix.run("B").tokens[1].state_id)  # st-B-1

        assert calls == []


class TestFindCallByRequestHashWhereExactness:
    """find_call_by_request_hash is the MOST DANGEROUS mutation target.

    Used for LLM response caching.  If ``==`` becomes ``>=`` on run_id it
    could return a cached response from a *different* run, silently corrupting
    pipeline results.  Both variants are kept: "returns target" and "same hash
    different run returns None".
    """

    def test_returns_only_target_run_call(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        calls_b = fix.factory.query.get_calls(target.tokens[0].state_id)
        assert len(calls_b) == 1
        request_hash = calls_b[0].request_hash

        result = fix.factory.execution.find_call_by_request_hash(target.run_id, CallType.HTTP, request_hash)

        assert result is not None
        assert result.call_id == target.tokens[0].call_id

    def test_same_hash_different_run_returns_none(self, multi_run_landscape: MultiRunFixture) -> None:
        """Cross-run leak guard: run-B's hash must not resolve under run-A's run_id."""
        fix = multi_run_landscape
        run_b = fix.run("B")

        calls_b = fix.factory.query.get_calls(run_b.tokens[0].state_id)
        request_hash_b = calls_b[0].request_hash

        result = fix.factory.execution.find_call_by_request_hash(fix.run("A").run_id, CallType.HTTP, request_hash_b)

        assert result is None


# ===========================================================================
# Execution methods — routing events
# ===========================================================================


class TestGetRoutingEventsWhereExactness:
    """get_routing_events (state-scoped) must return only events for the target state."""

    def test_returns_only_target_state_events(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")
        target_state_id = target.tokens[0].state_id  # st-B-0 has a routing event

        events = fix.factory.query.get_routing_events(target_state_id)

        assert len(events) == 1
        assert events[0].event_id == target.tokens[0].routing_event_id

    def test_returns_empty_for_state_without_events(self, multi_run_landscape: MultiRunFixture) -> None:
        """Second token in each run has no routing event."""
        fix = multi_run_landscape

        events = fix.factory.query.get_routing_events(fix.run("B").tokens[1].state_id)

        assert events == []


# ===========================================================================
# Execution methods — batches
# ===========================================================================


class TestGetBatchWhereExactness:
    """get_batch must return only the targeted batch, not neighbours."""

    def test_returns_only_target_batch(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        batch = fix.factory.execution.get_batch(target.batch_id)

        assert batch is not None
        assert batch.batch_id == target.batch_id


class TestGetBatchMembersWhereExactness:
    """get_batch_members must return only members for the target batch."""

    def test_returns_only_target_batch_members(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape
        target = fix.run("B")

        members = fix.factory.execution.get_batch_members(target.batch_id)

        assert len(members) == 2
        assert {m.token_id for m in members} == {t.token_id for t in target.tokens}


# ===========================================================================
# Execution methods — operations
# ===========================================================================


class TestGetOperationWhereExactness:
    """get_operation must return only the targeted operation, not neighbours."""

    def test_returns_only_target_operation(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        op_b = fix.factory.execution.begin_operation(fix.run("B").run_id, fix.run("B").source_node_id, "source_load")
        op_c = fix.factory.execution.begin_operation(fix.run("C").run_id, fix.run("C").source_node_id, "source_load")

        result = fix.factory.execution.get_operation(op_b.operation_id)

        assert result is not None
        assert result.operation_id == op_b.operation_id
        assert result.operation_id != op_c.operation_id


class TestGetOperationCallsWhereExactness:
    """get_operation_calls must return only calls for the target operation."""

    def test_returns_only_target_operation_calls(self, multi_run_landscape: MultiRunFixture) -> None:
        fix = multi_run_landscape

        op_b = fix.factory.execution.begin_operation(fix.run("B").run_id, fix.run("B").source_node_id, "source_load")
        call_b = fix.factory.execution.record_operation_call(
            op_b.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "https://b.example.com"}),
            RawCallPayload({"ok": True}),
            latency_ms=10.0,
        )

        op_c = fix.factory.execution.begin_operation(fix.run("C").run_id, fix.run("C").source_node_id, "source_load")
        fix.factory.execution.record_operation_call(
            op_c.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            RawCallPayload({"url": "https://c.example.com"}),
            RawCallPayload({"ok": True}),
            latency_ms=10.0,
        )

        calls = fix.factory.execution.get_operation_calls(op_b.operation_id)

        assert len(calls) == 1
        assert calls[0].call_id == call_b.call_id
