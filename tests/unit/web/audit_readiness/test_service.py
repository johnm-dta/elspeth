"""Tests for ReadinessService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

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


def _make_service(state, validation_result, inventory=()):
    exec_svc = MagicMock()
    exec_svc.validate = AsyncMock(return_value=validation_result)
    sess_svc = MagicMock()
    record = MagicMock()
    sess_svc.get_current_state = AsyncMock(return_value=record)
    # Use scoped_secret_resolver mock (list_refs(user_id) only — no auth_provider_type).
    # Matches app.py:470 precedent and the _SecretServiceLike Protocol (fix C4).
    scoped_resolver = MagicMock()
    scoped_resolver.list_refs = MagicMock(return_value=list(inventory))
    settings = MagicMock()
    settings.payload_store_retention_days = 90
    return ReadinessService(
        execution_service=exec_svc,
        session_service=sess_svc,
        secret_service=scoped_resolver,
        settings=settings,
        state_from_record=lambda _record: state,
    )


def _row(snap, row_id):
    matches = [r for r in snap.rows if r.id == row_id]
    if not matches:
        raise AssertionError(f"row {row_id!r} not in snapshot")
    return matches[0]


_OK = ValidationResult(is_valid=True, checks=[], errors=[], semantic_contracts=[])


def test_validation_row_ok_when_no_errors():
    svc = _make_service(_state(transforms=(("t", "passthrough"),)), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "validation").status == "ok"


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


def test_plugin_trust_row_ok_summary_when_boundary_plugins_present():
    # Default source_plugin="csv" → classify_plugin("source", "csv") is BOUNDARY
    # (sources are uniformly BOUNDARY per trust.py:87-88). Exercises the
    # boundary branch of _build_plugin_trust_row (service.py:174-185).
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
    # source_plugin=None skips the source _record call (service.py:145-146),
    # all-internal transforms+sinks leave `boundary` empty, hitting the
    # no-boundary branch (service.py:164-172).
    svc = _make_service(
        _state(
            source_plugin=None,
            transforms=(("t", "passthrough"),),
            sinks=(("out", "csv"),),
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
            )
        ],
        errors=[],
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


def test_llm_interpretations_always_not_applicable_in_phase_2a():
    svc = _make_service(_state(transforms=(("j", "llm"),)), _OK)
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "llm_interpretations").status == "not_applicable"


def test_secrets_not_applicable_when_no_refs():
    svc = _make_service(_state(), _OK, inventory=())
    snap = asyncio.run(
        svc.compute_snapshot(
            session_id=UUID("11111111-1111-1111-1111-111111111111"),
            user_id="alice",
        )
    )
    assert _row(snap, "secrets").status == "not_applicable"


def test_secrets_error_on_missing_refs():
    result = ValidationResult(
        is_valid=False,
        checks=[
            ValidationCheck(
                name="secret_refs",
                passed=False,
                detail="Missing secret references: openai_key",
                affected_nodes=(),  # no node attribution for secret check
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
        secret_service=scoped_resolver,
        settings=settings,
    )
    with pytest.raises(LookupError, match="no composition state"):
        asyncio.run(
            svc.compute_snapshot(
                session_id=UUID("11111111-1111-1111-1111-111111111111"),
                user_id="alice",
            )
        )
