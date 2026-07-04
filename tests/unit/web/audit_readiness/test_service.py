"""Tests for ReadinessService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from elspeth.contracts.composer_interpretation import (
    InterpretationChoice,
    InterpretationEventRecord,
    InterpretationKind,
    InterpretationSource,
)
from elspeth.contracts.secrets import SecretInventoryItem
from elspeth.web.audit_readiness.service import ReadinessService
from elspeth.web.composer.state import (
    CompositionState,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.execution.schemas import (
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationResult,
)

# ── Test factories ────────────────────────────────────────────────────────────
# Co-located here; if this conftest grows, extract to
# tests/integration/web/audit_readiness/conftest.py.
#
# NodeSpec has 13 required fields + 3 defaulted (trigger, output_mode,
# expected_output_count). OutputSpec has 4 required fields (name, plugin,
# options, on_write_failure — all required, no defaults).
# These factories cover ALL required kwargs so tests never TypeError at
# construction time (review B1, B2).
#
# Cross-reference: the identical factories must be used in test_explain.py
# and in any other test module that constructs NodeSpec/OutputSpec inline.


def make_node_spec(
    nid: str,
    plugin: str | None,
    *,
    input: str = "src_out",
    on_success: str | None = "out",
    node_type: str = "transform",
) -> NodeSpec:
    """Factory for NodeSpec covering all 13 required fields.

    Required-but-structural fields (on_error, condition, routes, fork_to,
    branches, policy, merge) are passed as None — they are required kwargs but
    are None for standard transform nodes.
    """
    return NodeSpec(
        id=nid,
        node_type=node_type,
        plugin=plugin,
        input=input,
        on_success=on_success,
        on_error=None,
        options={},
        condition=None,
        routes=None,
        fork_to=None,
        branches=None,
        policy=None,
        merge=None,
    )


def make_output_spec(name: str, plugin: str) -> OutputSpec:
    """Factory for OutputSpec covering all 4 required fields.

    on_write_failure defaults to "discard" — the canonical safe choice.
    """
    return OutputSpec(name=name, plugin=plugin, options={}, on_write_failure="discard")


def _state(*, source_plugin="csv", transforms=(), sinks=(("out", "csv"),)):
    src = (
        SourceSpec(
            plugin=source_plugin,
            on_success="src_out",
            options={},
            on_validation_failure="quarantine",
        )
        if source_plugin is not None
        else None
    )
    nodes = tuple(
        make_node_spec(
            nid,
            plg,
            input="src_out" if i == 0 else f"t{i - 1}_out",
            on_success=f"t{i}_out",
        )
        for i, (nid, plg) in enumerate(transforms)
    )
    outputs = tuple(make_output_spec(n, p) for n, p in sinks)
    return CompositionState(
        source=src,
        nodes=nodes,
        edges=(),
        outputs=outputs,
        metadata=PipelineMetadata(name="t", description=""),
        version=1,
    )


# Deterministic UUID for the test composition-state-id. The
# llm_interpretations row uses this to scope event lookups; tests that
# construct InterpretationEventRecord instances must pin the same value
# so the row builder sees them.
_TEST_COMPOSITION_STATE_ID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _blocked_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[])


def _interpretation_event_records_dispatch(events_by_source_and_state):
    """Build an AsyncMock that filters a fixed event list per call args.

    The ReadinessService issues two reads:
      (a) opt-out probe — sources=(AUTO_INTERPRETED_OPT_OUT,), composition_state_id=None
      (b) scoped events — no sources filter, composition_state_id=<current>

    ``events_by_source_and_state`` is a mapping ``{"opt_out": [...],
    "scoped": [...]}`` describing what each read should return. The
    dispatch routes by the ``sources`` kwarg (if AUTO_INTERPRETED_OPT_OUT
    is in the filter, route to the opt-out bucket; otherwise to scoped).
    """

    async def _list(
        _session_id,
        *,
        status="all",
        composition_state_id=None,
        sources=None,
    ):
        # `status` and `composition_state_id` accepted for signature
        # parity with the real method; the dispatch routes solely on
        # the `sources` argument (opt-out probe vs. scoped events).
        del status, composition_state_id
        if sources is not None and InterpretationSource.AUTO_INTERPRETED_OPT_OUT in sources:
            return list(events_by_source_and_state.get("opt_out", []))
        return list(events_by_source_and_state.get("scoped", []))

    return AsyncMock(side_effect=_list)


def _make_session_service(events_by_source_and_state=None):
    sess_svc = MagicMock()
    record = MagicMock()
    # record.id is read by ReadinessService.compute_snapshot as a UUID
    # (CompositionStateRecord.id — protocol.py:369). Pin a deterministic
    # value so test events bound to this id are correctly scoped.
    record.id = _TEST_COMPOSITION_STATE_ID
    sess_svc.get_current_state = AsyncMock(return_value=record)
    sess_svc.list_interpretation_events = _interpretation_event_records_dispatch(events_by_source_and_state or {})
    return sess_svc


def _make_service(state, validation_result, inventory=(), interpretation_events=None):
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(side_effect=AssertionError("ReadinessService must validate the already-read state"))
    exec_svc.validate_state = AsyncMock(return_value=validation_result)
    sess_svc = _make_session_service(interpretation_events)
    # Use scoped_secret_resolver mock (list_refs(user_id) only — no auth_provider_type).
    # Matches app.py:470 precedent and the _SecretServiceLike Protocol (fix C4).
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=list(inventory))
    settings = MagicMock()
    settings.payload_store_retention_days = 90
    return ReadinessService(
        execution_service=exec_svc,
        session_service=sess_svc,
        scoped_secret_resolver=scoped_resolver,
        settings=settings,
        state_from_record=lambda _record: state,
    )


def _make_service_with_execution_service(state, exec_svc, inventory=()):
    sess_svc = _make_session_service()
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=list(inventory))
    settings = MagicMock()
    settings.payload_store_retention_days = 90
    return ReadinessService(
        execution_service=exec_svc,
        session_service=sess_svc,
        scoped_secret_resolver=scoped_resolver,
        settings=settings,
        state_from_record=lambda _record: state,
    )


_UNSET: object = object()


def _make_event(
    *,
    choice: InterpretationChoice,
    interpretation_source: InterpretationSource = InterpretationSource.USER_APPROVED,
    affected_node_id: str | None = "llm_node",
    user_term: str | None = "cool",
    composition_state_id: UUID | None = _TEST_COMPOSITION_STATE_ID,
    event_id: UUID | None = None,
    runtime_model_identifier_at_resolve: str | None | object = _UNSET,
    runtime_model_version_at_resolve: str | None | object = _UNSET,
) -> InterpretationEventRecord:
    """Build a minimally-populated InterpretationEventRecord for tests.

    Field-population rules mirror the source-conditional CHECK
    constraints documented in ``contracts/composer_interpretation.py``:

      * USER_APPROVED + PENDING: surface fields populated; resolution
        fields (accepted_value, arguments_hash, resolved_at) are NULL.
      * USER_APPROVED + ACCEPTED_AS_DRAFTED/AMENDED: resolution fields
        populated.
      * AUTO_INTERPRETED_OPT_OUT: all surface and provenance fields
        NULL; choice is OPTED_OUT.
      * AUTO_INTERPRETED_NO_SURFACES: surface fields NULL; provenance
        fields populated; choice is OPTED_OUT (semantic, not session-
        wide opt-out).
    """
    now = datetime(2026, 5, 18, 12, 0, tzinfo=UTC)
    if interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT:
        return InterpretationEventRecord(
            id=event_id if event_id is not None else UUID("00000000-0000-0000-0000-000000000001"),
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            composition_state_id=None,
            affected_node_id=None,
            tool_call_id=None,
            user_term=None,
            kind=None,
            llm_draft=None,
            accepted_value=None,
            choice=InterpretationChoice.OPTED_OUT,
            created_at=now,
            resolved_at=now,
            actor="alice",
            model_identifier=None,
            model_version=None,
            provider=None,
            composer_skill_hash=None,
            arguments_hash=None,
            hash_domain_version=None,
            interpretation_source=interpretation_source,
            runtime_model_identifier_at_resolve=None,
            runtime_model_version_at_resolve=None,
            resolved_prompt_template_hash=None,
        )
    if interpretation_source is InterpretationSource.AUTO_INTERPRETED_NO_SURFACES:
        return InterpretationEventRecord(
            id=event_id if event_id is not None else UUID("00000000-0000-0000-0000-000000000002"),
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            composition_state_id=None,
            affected_node_id=None,
            tool_call_id=None,
            user_term=None,
            kind=InterpretationKind.VAGUE_TERM,
            llm_draft=None,
            accepted_value=None,
            choice=InterpretationChoice.OPTED_OUT,
            created_at=now,
            resolved_at=now,
            actor="alice",
            model_identifier="anthropic/claude-opus-4-7",
            model_version="2026-01-01",
            provider="anthropic",
            composer_skill_hash="0" * 64,
            arguments_hash=None,
            hash_domain_version=None,
            interpretation_source=interpretation_source,
            runtime_model_identifier_at_resolve=None,
            runtime_model_version_at_resolve=None,
            resolved_prompt_template_hash=None,
        )
    # USER_APPROVED row
    resolved = choice is not InterpretationChoice.PENDING
    return InterpretationEventRecord(
        id=event_id if event_id is not None else UUID("00000000-0000-0000-0000-000000000003"),
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
        composition_state_id=composition_state_id,
        affected_node_id=affected_node_id,
        tool_call_id="tc_1",
        user_term=user_term,
        kind=InterpretationKind.VAGUE_TERM,
        llm_draft="something cool",
        accepted_value="something cool" if resolved else None,
        choice=choice,
        created_at=now,
        resolved_at=now if resolved else None,
        actor="alice",
        model_identifier="anthropic/claude-opus-4-7",
        model_version="2026-01-01",
        provider="anthropic",
        composer_skill_hash="0" * 64,
        arguments_hash="a" * 64 if resolved else None,
        hash_domain_version="v1" if resolved else None,
        interpretation_source=interpretation_source,
        runtime_model_identifier_at_resolve=(
            ("anthropic/claude-opus-4-7" if resolved else None)
            if runtime_model_identifier_at_resolve is _UNSET
            else runtime_model_identifier_at_resolve  # type: ignore[return-value]
        ),
        runtime_model_version_at_resolve=(
            ("2026-01-01" if resolved else None) if runtime_model_version_at_resolve is _UNSET else runtime_model_version_at_resolve  # type: ignore[return-value]
        ),
        resolved_prompt_template_hash="b" * 64 if resolved else None,
    )


def _row(snap, row_id):
    matches = [r for r in snap.rows if r.id == row_id]
    if not matches:
        raise AssertionError(f"row {row_id!r} not in snapshot")
    return matches[0]


_OK = ValidationResult(is_valid=True, checks=[], errors=[], readiness=_ready_readiness(), semantic_contracts=[])


def test_validation_row_ok_when_no_errors():
    svc = _make_service(_state(transforms=(("t", "passthrough"),)), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "validation").status == "ok"


def test_compute_snapshot_populates_utc_checked_at():
    svc = _make_service(_state(transforms=(("t", "passthrough"),)), _OK)
    before = datetime.now(UTC)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    after = datetime.now(UTC)

    assert before <= snap.checked_at <= after
    assert snap.checked_at.tzinfo is UTC


def test_compute_snapshot_validates_already_read_state():
    state = _state(transforms=(("t", "passthrough"),))
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(side_effect=AssertionError("must not re-read session state"))
    exec_svc.validate_state = AsyncMock(return_value=_OK)
    svc = _make_service_with_execution_service(state, exec_svc)

    asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    exec_svc.validate.assert_not_awaited()
    exec_svc.validate_state.assert_awaited_once_with(
        state,
        user_id="alice",
        session_id=UUID("11111111-1111-1111-1111-111111111111"),
    )


def test_snapshot_preserves_raw_validation_result():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id="first",
                component_type="transform",
                message="first failed",
                suggestion="Fix first.",
                error_code=None,
            ),
            ValidationError(
                component_id="second",
                component_type="transform",
                message="second failed",
                suggestion="Fix second.",
                error_code=None,
            ),
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(transforms=(("first", "passthrough"), ("second", "passthrough"))), result)

    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    assert snap.validation_result == result


def test_validation_row_error_lists_component_ids():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id="out",
                component_type="sink",
                message="boom",
                suggestion=None,
                error_code=None,
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "validation")
    assert row.status == "error"
    assert row.component_ids == ("out",)


def test_validation_row_drops_engineer_prefix_and_uses_problem_wording():
    """elspeth-901a404926: the Validation row must not leak the
    "[component_type] component_id:" engineer prefix ("[unknown] unknown: …")
    on a novice surface, and the summary reads "problem to fix"."""
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message="Add an output step so your pipeline has somewhere to send its results.",
                suggestion=None,
                error_code="missing_sink",
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "validation")
    assert row.summary == "1 problem to fix — see details"
    assert row.detail == "Add an output step so your pipeline has somewhere to send its results."
    assert "unknown" not in (row.detail or "")


def test_validation_row_pluralizes_and_joins_messages_only():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message="Add a data source so your pipeline has data to read.",
                suggestion=None,
                error_code="missing_source",
            ),
            ValidationError(
                component_id=None,
                component_type=None,
                message="Add an output step so your pipeline has somewhere to send its results.",
                suggestion=None,
                error_code="missing_sink",
            ),
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "validation")
    assert row.summary == "2 problems to fix — see details"
    assert row.detail == (
        "Add a data source so your pipeline has data to read.\nAdd an output step so your pipeline has somewhere to send its results."
    )


def test_plugin_trust_row_ok_summary_when_boundary_plugins_present():
    # Default source_plugin="csv" (a Source — kind-derived boundary), plus
    # the default ("out", "csv") sink (Sink — also kind-derived boundary).
    # Exercises the boundary branch of _build_plugin_trust_row.
    svc = _make_service(
        _state(transforms=(("t", "passthrough"),), sinks=(("out", "csv"),)),
        _OK,
    )
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "plugin_trust")
    assert row.status == "ok"
    assert "external-boundary" in row.summary
    assert "source" in row.component_ids
    assert row.component_ids != ()


def test_plugin_trust_row_ok_summary_when_no_boundary_plugins():
    # Source=None and sinks=() means no source-or-sink kind in the
    # composition. With only an internal transform (passthrough,
    # determinism=DETERMINISTIC), the (kind, determinism) predicate has
    # nothing to classify as boundary — exercises the no-boundary branch.
    svc = _make_service(
        _state(
            source_plugin=None,
            transforms=(("t", "passthrough"),),
            sinks=(),
        ),
        _OK,
    )
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "plugin_trust")
    assert row.status == "ok"
    assert row.summary == "All plugins operate on pipeline data"
    assert row.component_ids == ()


def test_plugin_trust_row_error_on_unknown_plugin():
    # NodeSpec.plugin is str | None; a None plugin reaches the unknown
    # branch in _build_plugin_trust_row (service.py:153-162) and flips the
    # row to status="error".
    svc = _make_service(
        _state(transforms=(("bad", None),)),
        _OK,
    )
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "plugin_trust")
    assert row.status == "error"
    assert row.summary == "Unknown plugin in composition"
    assert "bad" in row.component_ids


def test_provenance_warning_on_identity_advisory():
    result = ValidationResult(
        is_valid=True,
        checks=[
            ValidationCheck(
                name="identity_node_advisory",
                passed=True,
                detail=("Node 'pass' is an identity-shaped passthrough between 'source' and sink 'out'."),
                affected_nodes=("pass",),  # structured field; no prose parse needed
                outcome_code=None,
            )
        ],
        errors=[],
        readiness=_ready_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(transforms=(("pass", "passthrough"),)), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "provenance")
    assert row.status == "warning"
    assert "pass" in (row.detail or "")
    assert "pass" in row.component_ids


def test_provenance_not_applicable_when_identity_advisory_check_was_skipped():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="identity_node_advisory",
                passed=False,
                detail="Skipped: path_allowlist failed",
                affected_nodes=(),
                outcome_code="validation.skipped_after_failure",
            )
        ],
        errors=[
            ValidationError(
                component_id="source",
                component_type="source",
                message="Path traversal blocked",
                suggestion="Use a file within the blobs directory.",
                error_code=None,
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "provenance")
    assert row.status == "not_applicable"
    assert row.summary == "Provenance check did not run"
    assert row.detail == "Skipped: path_allowlist failed"


def test_provenance_not_applicable_when_validation_failed_before_identity_advisory():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="path_allowlist",
                passed=False,
                detail="Source path is outside allowed source directories",
                affected_nodes=(),
                outcome_code=None,
            )
        ],
        errors=[
            ValidationError(
                component_id="source",
                component_type="source",
                message="Path traversal blocked",
                suggestion="Use a file within the blobs directory.",
                error_code=None,
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "provenance")
    assert row.status == "not_applicable"
    assert row.summary == "Provenance check did not run"
    assert row.detail == "Validation failed before provenance advisory analysis could run"


def test_retention_row_reports_system_value():
    svc = _make_service(_state(), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "retention")
    assert row.status == "not_applicable"
    assert "90" in row.summary


def test_llm_interpretations_not_applicable_when_no_llm_transforms():
    """No LLM transforms in the composition → not_applicable.

    Also asserts the service short-circuits: it does NOT query the
    session-service for interpretation events when there's nothing to
    interpret. This keeps the per-snapshot read cost zero for the
    common non-LLM composition.
    """
    svc = _make_service(_state(transforms=(("t", "passthrough"),)), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "not_applicable"
    assert row.summary == "No LLM transforms in this composition"
    assert row.component_ids == ()
    # No interpretation-events query was issued (short-circuit).
    sess_svc = svc._session_service  # type: ignore[attr-defined]
    sess_svc.list_interpretation_events.assert_not_called()


def test_llm_interpretations_not_applicable_when_llm_present_but_no_events():
    """LLM transforms present, no interpretation events yet → not_applicable.

    The surfacing hasn't been triggered yet — the panel reports the
    distinction between "this composition will never interpret" (no
    LLM) and "this composition could interpret but hasn't yet".
    """
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "not_applicable"
    assert row.summary == "No interpretation events yet for this composition"
    assert row.component_ids == ()


def test_llm_interpretations_warning_when_pending_events_present():
    """Any PENDING interpretation event → warning + node ids in component_ids."""
    events = {
        "opt_out": [],
        "scoped": [
            _make_event(
                choice=InterpretationChoice.PENDING,
                affected_node_id="j",
                user_term="cool",
            ),
        ],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "warning"
    assert row.summary == "1 pending interpretation review"
    assert row.component_ids == ("j",)
    assert row.detail is not None
    assert "cool" in row.detail


def test_llm_interpretations_ok_when_all_events_resolved():
    """All events resolved (USER_APPROVED + accepted) → ok."""
    events = {
        "opt_out": [],
        "scoped": [
            _make_event(
                choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
                affected_node_id="j",
                event_id=UUID("00000000-0000-0000-0000-00000000000a"),
            ),
            _make_event(
                choice=InterpretationChoice.AMENDED,
                affected_node_id="j",
                event_id=UUID("00000000-0000-0000-0000-00000000000b"),
            ),
        ],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "ok"
    assert row.summary == "2 interpretation(s) resolved"
    assert row.component_ids == ("j",)


def test_llm_interpretations_not_applicable_when_session_opted_out():
    """Session-wide opt-out → not_applicable with an explicit note.

    The opt-out signal is read via a SEPARATE query
    (sources=[AUTO_INTERPRETED_OPT_OUT]) because the opt-out row is
    written with composition_state_id=NULL and would not match a
    composition-state-scoped WHERE clause.
    """
    events = {
        "opt_out": [
            _make_event(
                choice=InterpretationChoice.OPTED_OUT,
                interpretation_source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
            )
        ],
        # Even if there were scoped events, opt-out wins. The dispatcher
        # would never read this bucket because the service short-circuits
        # after detecting opt-out, but the test pins that intent.
        "scoped": [],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "not_applicable"
    assert "opted out" in row.summary.lower()
    assert row.detail is not None
    assert "stop asking" in row.detail


def test_llm_interpretations_ok_when_auto_interpreted_no_surfaces_baked_in():
    """auto_interpreted_no_surfaces rows count as resolved, NOT as opt-out.

    Distinct from AUTO_INTERPRETED_OPT_OUT: AUTO_INTERPRETED_NO_SURFACES
    is the rate-cap-baked-in case where the LLM produced the
    interpretation silently because the surface budget was exhausted.
    The user did NOT opt out session-wide; this composition state
    simply has resolved interpretations bound to it. The row should
    surface as ok.
    """
    events = {
        "opt_out": [],
        "scoped": [
            _make_event(
                choice=InterpretationChoice.OPTED_OUT,
                interpretation_source=InterpretationSource.AUTO_INTERPRETED_NO_SURFACES,
                affected_node_id=None,  # NULL per the source-conditional CHECK
                user_term=None,
            ),
        ],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "ok"
    assert row.summary == "1 interpretation resolved"
    # affected_node_id is NULL on this row shape, so component_ids
    # excludes it (the dedup-and-sort projection skips None).
    assert row.component_ids == ()


def test_llm_interpretations_ok_when_composer_and_runtime_models_differ():
    """Composer drafter and runtime executor models are different roles.

    ``model_identifier`` records which composer LLM drafted the
    interpretation surface. ``runtime_model_identifier_at_resolve`` records
    which pipeline LLM will execute the resolved prompt. A mismatch between
    those two fields is normal and must not be reported as a rotated model.
    """
    events = {
        "opt_out": [],
        "scoped": [
            _make_event(
                choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
                affected_node_id="j",
                runtime_model_identifier_at_resolve="anthropic/claude-sonnet-4-5",
            ),
        ],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "ok"
    assert row.detail is None
    assert row.component_ids == ("j",)


def test_llm_interpretations_ok_when_runtime_model_matches_recorded():
    """F-19 parity case: runtime_model_identifier_at_resolve equals
    model_identifier (the common case where the operator did not rotate models)
    → row stays ``ok``; no drift warning is emitted.

    Pins the boundary so a regression that always-emits a drift warning
    surfaces immediately.
    """
    events = {
        "opt_out": [],
        "scoped": [
            _make_event(
                choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
                affected_node_id="j",
                # Both equal — factory default for both fields when resolved.
            ),
        ],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "ok"


def test_llm_interpretations_opt_out_overrides_runtime_model_drift():
    """Status precedence pin: session-wide opt-out is a stronger statement
    than runtime-model drift; opt-out keeps the row at ``not_applicable``
    even if scoped events would otherwise emit a drift warning.

    Without this pin a future drift implementation that blindly downgrades
    ``not_applicable`` to ``warning`` would silently leak opt-out as a
    warning state.
    """
    events = {
        "opt_out": [
            _make_event(
                choice=InterpretationChoice.OPTED_OUT,
                interpretation_source=InterpretationSource.AUTO_INTERPRETED_OPT_OUT,
            )
        ],
        "scoped": [],
    }
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK, interpretation_events=events)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    row = _row(snap, "llm_interpretations")
    assert row.status == "not_applicable"


def test_secrets_not_applicable_when_no_refs():
    svc = _make_service(_state(), _OK, inventory=())
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "secrets").status == "not_applicable"


def test_secrets_not_applicable_when_secret_refs_check_reports_no_refs():
    result = ValidationResult(
        is_valid=True,
        checks=[
            ValidationCheck(
                name="secret_refs",
                passed=True,
                detail="Secret scan completed without references",
                affected_nodes=(),
                outcome_code="secret_refs.no_refs",
            )
        ],
        errors=[],
        readiness=_ready_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result, inventory=())
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "secrets")
    assert row.status == "not_applicable"
    assert row.summary == "No secret references in this composition"


def test_secrets_not_applicable_when_no_ref_check_has_unrelated_inventory():
    result = ValidationResult(
        is_valid=True,
        checks=[
            ValidationCheck(
                name="secret_refs",
                passed=True,
                detail="Secret scan completed without references",
                affected_nodes=(),
                outcome_code="secret_refs.no_refs",
            )
        ],
        errors=[],
        readiness=_ready_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(
        _state(),
        result,
        inventory=(SecretInventoryItem(name="UNRELATED_API_KEY", scope="user", available=True),),
    )
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "secrets")
    assert row.status == "not_applicable"
    assert row.summary == "No secret references in this composition"


def test_secrets_error_on_missing_refs():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="secret_refs",
                passed=False,
                detail="Missing secret references: openai_key",
                affected_nodes=(),  # no node attribution for secret check
                outcome_code="secret_refs.unresolved",
            )
        ],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message="Cannot resolve secret references: openai_key",
                suggestion="Add via Secrets panel.",
                error_code="missing_secret_ref",  # structured discriminant
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "secrets").status == "error"


def test_secrets_error_when_secret_refs_check_failed_without_typed_error():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="secret_refs",
                passed=False,
                detail="Secret reference validation failed",
                affected_nodes=(),
                outcome_code="secret_refs.unresolved",
            )
        ],
        errors=[],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "secrets")
    assert row.status == "error"
    assert row.summary == "Secret reference check failed"


# The three values in service.py:262 _SECRET_ERROR_CODES are
# "missing_secret_ref", "fabricated_secret", and "disallowed_secret_ref" — the
# producer error codes from web/execution/validation.py:727/748/770. Only
# "missing_secret_ref" had explicit coverage; without per-code tests, a
# producer-side rename would silently demote the secrets row to ok/n_a and the
# audit panel would no longer surface the failure. One test per code keeps
# every membership entry load-bearing.


def test_secrets_error_on_fabricated_secret():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message="Fabricated secret reference: openai_key",
                suggestion="Define this secret before referencing it.",
                error_code="fabricated_secret",
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "secrets").status == "error"


def test_secrets_error_on_disallowed_secret_ref():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id=None,
                component_type=None,
                message="Disallowed secret reference: ELSPETH_FINGERPRINT_KEY",
                suggestion="Choose a non-reserved secret name.",
                error_code="disallowed_secret_ref",
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "secrets").status == "error"


def test_secrets_not_applicable_when_secret_check_was_skipped():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="path_allowlist",
                passed=False,
                detail="Source path is outside allowed source directories",
                affected_nodes=(),
                outcome_code=None,
            ),
            ValidationCheck(
                name="secret_refs",
                passed=False,
                detail="Skipped: path_allowlist failed",
                affected_nodes=(),
                outcome_code="validation.skipped_after_failure",
            ),
        ],
        errors=[
            ValidationError(
                component_id="source",
                component_type="source",
                message="Path traversal blocked",
                suggestion="Use a file within the blobs directory.",
                error_code=None,
            )
        ],
        readiness=_blocked_readiness(),
        semantic_contracts=[],
    )
    svc = _make_service(_state(), result)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "secrets")
    assert row.status == "not_applicable"
    assert row.summary == "Secret reference check did not run"
    assert row.detail == "Skipped: path_allowlist failed"


def test_plugin_trust_row_errors_on_non_catalog_plugin_name():
    svc = _make_service(
        _state(transforms=(("bad", "lmm"),)),
        _OK,
    )
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )

    row = _row(snap, "plugin_trust")
    assert row.status == "error"
    assert row.summary == "Unknown plugin in composition"
    assert row.component_ids == ("bad",)


def test_snapshot_raises_when_no_state():
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(return_value=_OK)
    sess_svc = MagicMock()
    sess_svc.get_current_state = AsyncMock(return_value=None)
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=[])
    settings = MagicMock(payload_store_retention_days=90)
    svc = ReadinessService(
        execution_service=exec_svc,
        session_service=sess_svc,
        scoped_secret_resolver=scoped_resolver,
        settings=settings,
    )
    with pytest.raises(LookupError, match="No composition state"):
        asyncio.run(
            svc.compute_snapshot(
                session_id=UUID("11111111-1111-1111-1111-111111111111"),
                user_id="alice",
            )
        )


# ── Module-private catalog helpers ────────────────────────────────────────────
#
# _plugin_catalog_snapshot / _is_registered_plugin / _get_plugin_class_for_kind
# are private but load-bearing: _record() (the entrypoint that classifies a
# composition's plugins as boundary-crossing) guards with _is_registered_plugin
# then resolves with _get_plugin_class_for_kind. The two helpers MUST share a
# single snapshot — otherwise a class registered between guard and resolve
# would slip past _record's safety net.


from typing import Any, cast  # noqa: E402  — co-located with the helper-tests block.

from elspeth.web.audit_readiness import service as _service_mod  # noqa: E402


class TestPluginCatalogHelpers:
    """Covers the offensive-programming guards and the shared-snapshot
    invariant on the three private catalog helpers.
    """

    def test_is_registered_plugin_raises_on_unknown_kind(self) -> None:
        """The PluginKind Literal forbids "bogus" at type-check time, but the
        runtime guard exists for non-typed callers (test/REPL/JSON-dispatch).
        """
        with pytest.raises(ValueError, match="unknown plugin kind"):
            _service_mod._is_registered_plugin(cast(Any, "bogus"), "anything")

    def test_get_plugin_class_for_kind_raises_on_unknown_kind(self) -> None:
        """Mirrors _is_registered_plugin's offensive guard so both helpers
        fail loudly on the same bad input rather than diverging.
        """
        with pytest.raises(ValueError, match="unknown plugin kind"):
            _service_mod._get_plugin_class_for_kind(cast(Any, "bogus"), "anything")

    def test_get_plugin_class_for_kind_raises_descriptive_runtime_error_when_guard_skipped(
        self,
    ) -> None:
        """The previous implementation used ``next(...)`` which raised an
        opaque ``StopIteration``. The contract is: when ``_is_registered_plugin``
        returns False (or the caller skipped that guard), resolution must
        raise ``RuntimeError`` with enough context to identify the bug.
        """
        ghost = "definitely_not_a_real_plugin_zzzzz"
        assert _service_mod._is_registered_plugin("source", ghost) is False

        with pytest.raises(RuntimeError) as exc_info:
            _service_mod._get_plugin_class_for_kind("source", ghost)

        msg = str(exc_info.value)
        assert ghost in msg, "RuntimeError must name the missing plugin"
        assert "source" in msg, "RuntimeError must name the plugin kind"
        assert "_is_registered_plugin" in msg, (
            "RuntimeError must point to the guard the caller skipped — the "
            "diagnostic is useless if it doesn't tell the reader what was missed."
        )

    def test_is_registered_and_get_class_share_single_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Structurally enforce the 'shared snapshot' contract: both helpers
        delegate to ``_plugin_catalog_snapshot()`` so monkeypatching that
        single function reroutes BOTH lookups together.
        """

        class _StubSource:
            name = "stub_src_for_shared_snapshot_test"

        fake_snapshot: dict[str, dict[str, type]] = {
            "source": {"stub_src_for_shared_snapshot_test": _StubSource},
            "transform": {},
            "sink": {},
        }
        monkeypatch.setattr(_service_mod, "_plugin_catalog_snapshot", lambda: fake_snapshot)

        # Membership query and class resolution both reroute through the
        # patched snapshot — they cannot disagree.
        assert _service_mod._is_registered_plugin("source", "stub_src_for_shared_snapshot_test") is True
        assert _service_mod._get_plugin_class_for_kind("source", "stub_src_for_shared_snapshot_test") is _StubSource

        # A name absent from the fake snapshot is rejected by BOTH paths.
        assert _service_mod._is_registered_plugin("source", "absent_from_fake") is False
        with pytest.raises(RuntimeError, match="not in catalog snapshot"):
            _service_mod._get_plugin_class_for_kind("source", "absent_from_fake")

    def test_plugin_manager_built_exactly_once_for_both_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The previous implementation instantiated ``PluginManager`` twice —
        once per helper — and relied on import-time stability to keep the
        two snapshots in sync. After the refactor, both helpers must share
        a single ``_plugin_catalog_snapshot()`` so ``PluginManager`` is built
        once per cache lifetime.
        """
        _service_mod._plugin_catalog_snapshot.cache_clear()

        import elspeth.plugins.infrastructure.manager as _manager_mod

        original = _manager_mod.PluginManager
        call_count = 0

        def _counting_manager(*args: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_manager_mod, "PluginManager", _counting_manager)

        # Two distinct entrypoints into the catalog. With separate
        # PluginManager instances per helper this would tick the counter
        # twice; with a shared snapshot the counter ticks once.
        _service_mod._is_registered_plugin("source", "no_such_plugin_zzz")
        with pytest.raises(RuntimeError):
            _service_mod._get_plugin_class_for_kind("source", "no_such_plugin_zzz")

        assert call_count == 1, (
            f"PluginManager instantiated {call_count} times; the shared-snapshot contract requires exactly 1 per cache lifetime."
        )
