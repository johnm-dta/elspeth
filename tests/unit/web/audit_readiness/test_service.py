"""Tests for ReadinessService."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

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
    exec_svc.validate = AsyncMock(side_effect=AssertionError("ReadinessService must validate the already-read state"))
    exec_svc.validate_state = AsyncMock(return_value=validation_result)
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
        scoped_secret_resolver=scoped_resolver,
        settings=settings,
        state_from_record=lambda _record: state,
    )


def _make_service_with_execution_service(state, exec_svc, inventory=()):
    sess_svc = MagicMock()
    record = MagicMock()
    sess_svc.get_current_state = AsyncMock(return_value=record)
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
    exec_svc.validate_state.assert_awaited_once_with(state, user_id="alice")


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
                outcome_code=None,
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
