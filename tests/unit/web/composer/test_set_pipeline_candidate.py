"""Characterization for the ``set_pipeline`` candidate/settlement boundary.

These tests deliberately exercise the current executor before the candidate
builder is extracted.  They assert public state/result behavior and durable
inline-blob effects, not private control flow.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, insert, select, update
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.web.blobs.service import BlobServiceImpl, content_hash
from elspeth.web.catalog.policy_view import PolicyCatalogView
from elspeth.web.composer.audit import BufferingRecorder, begin_dispatch, dispatch_with_audit
from elspeth.web.composer.pipeline_proposal import reviewed_anchor_hash
from elspeth.web.composer.reviewed_source_authority import resolve_reviewed_source_authority
from elspeth.web.composer.state import CompositionState, PipelineMetadata, ValidationEntry, ValidationSummary
from elspeth.web.composer.tools import (
    SetPipelineCandidate,
    ToolContext,
    _execute_set_pipeline,
    build_set_pipeline_candidate,
    execute_tool,
)
from elspeth.web.composer.tools._common import normalize_tool_result_validation
from elspeth.web.dependencies import create_catalog_service
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY, SOURCE_AUTHORING_KEY
from elspeth.web.plugin_policy.models import (
    PluginAvailability,
    PluginAvailabilitySnapshot,
    PluginId,
    PluginUnavailableReason,
)
from elspeth.web.plugin_policy.profiles import OperatorProfileRegistry
from elspeth.web.plugin_policy.validation import (
    PluginPolicyFinding,
    ProfileAwareValidationResult,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, chat_messages_table, sessions_table
from elspeth.web.sessions.schema import initialize_session_schema


def _empty_state() -> CompositionState:
    return CompositionState(
        source=None,
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_set_pipeline_candidate_contract_is_frozen_and_slots_based() -> None:
    state = _empty_state()
    result = _execute_set_pipeline(
        _linear_args(Path("/tmp/candidate-contract")),
        state,
        _trained_context(),
    )
    candidate = SetPipelineCandidate(result=result, prepared_inline_blob=None)

    assert candidate.acceptable is True
    with pytest.raises(TypeError):
        vars(candidate)
    with pytest.raises(FrozenInstanceError):
        candidate.prepared_inline_blob = None  # type: ignore[misc]


def test_build_set_pipeline_candidate_constructs_without_publishing(tmp_path: Path) -> None:
    state = _empty_state()

    candidate = build_set_pipeline_candidate(
        _linear_args(tmp_path),
        state,
        _trained_context(data_dir=tmp_path),
    )

    assert candidate.acceptable is True
    assert candidate.prepared_inline_blob is None
    assert candidate.result.updated_state is not state
    assert state == _empty_state()


def _trained_context(*, data_dir: Path | None = None, **kwargs: Any) -> ToolContext:
    catalog = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    return ToolContext(
        catalog=PolicyCatalogView.for_trained_operator(catalog, snapshot),
        plugin_snapshot=snapshot,
        data_dir=None if data_dir is None else str(data_dir),
        **kwargs,
    )


def _blocked_context(plugin_id: PluginId, *, data_dir: Path) -> ToolContext:
    catalog = create_catalog_service()
    unrestricted = PluginAvailabilitySnapshot.for_trained_operator(catalog)
    snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="candidate-characterization-restricted",
        principal_scope="local:candidate-characterization",
        available=unrestricted.available - {plugin_id},
        unavailable=(PluginAvailability(plugin_id, PluginUnavailableReason.NOT_AUTHORIZED),),
        selected=unrestricted.selected,
        usable_profile_aliases=(),
        selected_profile_aliases=(),
        binding_generation_fingerprint="candidate-characterization-generation",
    )
    return ToolContext(
        catalog=PolicyCatalogView(catalog, snapshot, MagicMock(spec=OperatorProfileRegistry)),
        plugin_snapshot=snapshot,
        data_dir=str(data_dir),
    )


class _ProfileRejectingCatalog(PolicyCatalogView):
    """Real catalog projection with one deterministic profile finding."""

    def validate_composition_state(self, state: CompositionState) -> ProfileAwareValidationResult:
        finding = PluginPolicyFinding(
            stage="operator_profile_options",
            component_id="profile_prevalidation",
            component_type="transform",
            error_code="profile_unavailable",
            message="The requested operator profile is unavailable.",
        )
        return ProfileAwareValidationResult(
            authored_state=state,
            executable_state=state,
            policy_findings=(finding,),
            validation=state.validate(),
        )


def _profile_rejecting_context(*, data_dir: Path) -> ToolContext:
    full = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(full)
    catalog = _ProfileRejectingCatalog(full, snapshot, MagicMock(spec=OperatorProfileRegistry))
    return ToolContext(catalog=catalog, plugin_snapshot=snapshot, data_dir=str(data_dir))


class _FinalValidationRejectingCatalog(PolicyCatalogView):
    """Request catalog whose whole-state validation rejects one valid graph."""

    validated_states: list[CompositionState]

    def validate_composition_state(self, state: CompositionState) -> ProfileAwareValidationResult:
        self.validated_states.append(state)
        raw = state.validate()
        if raw.is_valid and state.metadata.name == "final-profile-reject":
            validation = ValidationSummary(
                is_valid=False,
                errors=(
                    ValidationEntry(
                        component="policy:final",
                        message="The request-scoped profile rejects this complete composition.",
                        severity="high",
                        error_code="profile_complete_state_rejected",
                    ),
                ),
                warnings=raw.warnings,
                suggestions=raw.suggestions,
                edge_contracts=raw.edge_contracts,
                semantic_contracts=raw.semantic_contracts,
            )
        else:
            validation = raw
        return ProfileAwareValidationResult(
            authored_state=state,
            executable_state=state,
            policy_findings=(),
            validation=validation,
        )


def _final_validation_rejecting_context(*, data_dir: Path) -> tuple[ToolContext, _FinalValidationRejectingCatalog]:
    full = create_catalog_service()
    snapshot = PluginAvailabilitySnapshot.for_trained_operator(full)
    catalog = _FinalValidationRejectingCatalog.for_trained_operator(full, snapshot)
    catalog.validated_states = []
    return ToolContext(catalog=catalog, plugin_snapshot=snapshot, data_dir=str(data_dir)), catalog


def _file_options(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "schema": {"mode": "observed"},
        "mode": "write",
        "collision_policy": "auto_increment",
    }


def _linear_args(tmp_path: Path) -> dict[str, Any]:
    return {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {
                "path": str(tmp_path / "blobs" / "input.csv"),
                "schema": {"mode": "observed"},
            },
            "on_validation_failure": "discard",
        },
        "nodes": [
            {
                "id": "copy",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "rows",
                "on_success": "main",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            }
        ],
        "edges": [
            {
                "id": "source_to_copy",
                "from_node": "source",
                "to_node": "copy",
                "edge_type": "on_success",
                "label": None,
            }
        ],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": _file_options(tmp_path / "outputs" / "result.jsonl") | {"format": "jsonl"},
                "on_write_failure": "discard",
            }
        ],
        "metadata": {"name": "linear"},
    }


def _reviewed_source_harness(tmp_path: Path) -> tuple[Any, str, str, Any]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    first_session = str(uuid4())
    second_session = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        for session_id in (first_session, second_session):
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="review-owner",
                    auth_provider_type="local",
                    title="reviewed source authority",
                    created_at=now,
                    updated_at=now,
                )
            )
    service = BlobServiceImpl(engine, tmp_path)
    first_blob = asyncio.run(
        service.create_blob(
            UUID(first_session),
            "first.csv",
            b"name,score\nAda,42\n",
            "text/csv",
        )
    )
    second_blob = asyncio.run(
        service.create_blob(
            UUID(second_session),
            "second.csv",
            b"name,score\nGrace,99\n",
            "text/csv",
        )
    )
    return engine, first_session, second_session, (first_blob, second_blob)


def _reviewed_source_facts(*, blob_id: str, source_name: str = "source", path: str | None = None) -> dict[str, Any]:
    stable_id = str(uuid4())
    return {
        "source_order": [stable_id],
        "reviewed_sources": {
            stable_id: {
                "name": source_name,
                "plugin": "csv",
                "options": {
                    "path": path if path is not None else f"blob:{blob_id}",
                    "blob_ref": blob_id,
                    "schema": {"mode": "observed"},
                    SOURCE_AUTHORING_KEY: {
                        "modality": "llm_generated",
                        "content_hash": "a" * 64,
                        "review_event_id": "review-event",
                        "resolved_kind": "invented_source",
                    },
                },
                "observed_columns": ["name", "score"],
                "sample_rows": [{"name": "Ada", "score": 42}],
                "on_validation_failure": "discard",
            }
        },
        "output_order": [],
        "reviewed_outputs": {},
    }


def _named_reviewed_pipeline(tmp_path: Path, facts: dict[str, Any]) -> dict[str, Any]:
    pipeline = _linear_args(tmp_path)
    source = pipeline.pop("source")
    reviewed = next(iter(facts["reviewed_sources"].values()))
    source.update(
        {
            "plugin": reviewed["plugin"],
            "options": deepcopy(reviewed["options"]),
            "on_validation_failure": reviewed["on_validation_failure"],
        }
    )
    pipeline["sources"] = {reviewed["name"]: source}
    return pipeline


def test_reviewed_source_authority_resolves_only_ready_owned_current_anchor(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    facts = _reviewed_source_facts(blob_id=str(blob.id))
    anchor = reviewed_anchor_hash(facts)

    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=facts,
        expected_reviewed_anchor_hash=anchor,
    )

    assert authority is not None
    assert authority.session_id == session_id
    assert authority.reviewed_anchor_hash == anchor
    assert authority.verified_blob_paths == {f"blob:{blob.id}": blob.storage_path}


@pytest.mark.parametrize(
    "failure",
    ["missing", "altered_uuid", "cross_session", "non_ready", "wrong_owner", "anchor_drift"],
)
def test_reviewed_source_authority_rejects_unverified_blob_custody(tmp_path: Path, failure: str) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    first_blob, second_blob = blobs
    selected_blob_id = str(first_blob.id)
    user_id = "review-owner"
    if failure in {"missing", "altered_uuid"}:
        selected_blob_id = str(uuid4())
    elif failure == "cross_session":
        selected_blob_id = str(second_blob.id)
    elif failure == "non_ready":
        with engine.begin() as conn:
            conn.execute(update(blobs_table).where(blobs_table.c.id == str(first_blob.id)).values(status="pending"))
    elif failure == "wrong_owner":
        user_id = "different-user"
    facts = _reviewed_source_facts(blob_id=selected_blob_id)
    expected_anchor = reviewed_anchor_hash(facts)
    if failure == "anchor_drift":
        expected_anchor = "f" * 64

    with pytest.raises(AuditIntegrityError):
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id=user_id,
            reviewed_facts=facts,
            expected_reviewed_anchor_hash=expected_anchor,
        )


def test_reviewed_source_authority_rechecks_ready_status_at_accept_time(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    facts = _reviewed_source_facts(blob_id=str(blob.id))
    anchor = reviewed_anchor_hash(facts)
    assert (
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_facts=facts,
            expected_reviewed_anchor_hash=anchor,
        )
        is not None
    )

    with engine.begin() as conn:
        conn.execute(update(blobs_table).where(blobs_table.c.id == str(blob.id)).values(status="error"))

    with pytest.raises(AuditIntegrityError, match="not ready"):
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_facts=facts,
            expected_reviewed_anchor_hash=anchor,
        )


@pytest.mark.parametrize("raw_path_kind", ("alternate", "other_session"))
def test_reviewed_source_authority_rejects_raw_path_not_bound_to_same_owned_blob(
    tmp_path: Path,
    raw_path_kind: str,
) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    first_blob, second_blob = blobs
    raw_path = "/etc/looks-reviewed.csv" if raw_path_kind == "alternate" else second_blob.storage_path
    facts = _reviewed_source_facts(blob_id=str(first_blob.id), path=raw_path)

    with pytest.raises(AuditIntegrityError, match="storage path"):
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_facts=facts,
            expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
        )


def test_reviewed_source_authority_checks_owner_without_recognized_blob(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    facts = _reviewed_source_facts(blob_id=str(blobs[0].id))
    options = next(iter(facts["reviewed_sources"].values()))["options"]
    options.pop("path")
    options.pop("blob_ref")

    with pytest.raises(AuditIntegrityError, match="owned by the current user session"):
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id="different-user",
            reviewed_facts=facts,
            expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
        )


def test_exact_reviewed_non_blob_path_remains_subject_to_candidate_path_policy(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    reviewed_path = str(tmp_path / "blobs" / "operator-reviewed.csv")
    facts = _reviewed_source_facts(blob_id=str(blobs[0].id), path=reviewed_path)
    options = next(iter(facts["reviewed_sources"].values()))["options"]
    options.pop("blob_ref")

    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
    )
    candidate = build_set_pipeline_candidate(
        _named_reviewed_pipeline(tmp_path, facts),
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=authority,
        ),
    )

    assert authority is not None
    assert authority.verified_blob_paths == {}
    assert candidate.acceptable is True, candidate.result.to_dict()
    assert candidate.result.updated_state.sources["source"].options["path"] == reviewed_path


def test_exact_owned_raw_storage_path_remains_private_executable_authority(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    facts = _reviewed_source_facts(blob_id=str(blob.id), path=blob.storage_path)
    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
    )

    candidate = build_set_pipeline_candidate(
        _named_reviewed_pipeline(tmp_path, facts),
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=authority,
        ),
    )

    assert authority is not None
    assert authority.verified_blob_paths == {}
    assert candidate.acceptable is True, candidate.result.to_dict()
    assert candidate.result.updated_state.sources["source"].options["path"] == blob.storage_path


def test_exact_reviewed_source_authority_allows_private_blob_resolution(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    facts = _reviewed_source_facts(blob_id=str(blob.id))
    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
    )

    candidate = build_set_pipeline_candidate(
        _named_reviewed_pipeline(tmp_path, facts),
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=authority,
        ),
    )

    assert candidate.acceptable is True, candidate.result.to_dict()
    assert candidate.result.updated_state.sources["source"].options["path"] == blob.storage_path


@pytest.mark.parametrize("mutation", ["name", "plugin", "options", "failure_policy"])
def test_reviewed_source_authority_requires_every_candidate_field_to_match(tmp_path: Path, mutation: str) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    facts = _reviewed_source_facts(blob_id=str(blob.id))
    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(facts),
    )
    pipeline = _named_reviewed_pipeline(tmp_path, facts)
    source = pipeline["sources"].pop("source")
    source_name = "source"
    if mutation == "name":
        source_name = "other"
    elif mutation == "plugin":
        source["plugin"] = "json"
    elif mutation == "options":
        source["options"][SOURCE_AUTHORING_KEY]["content_hash"] = "b" * 64
    else:
        source["on_validation_failure"] = "fail"
    pipeline["sources"][source_name] = source

    candidate = build_set_pipeline_candidate(
        pipeline,
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=authority,
        ),
    )

    assert candidate.acceptable is False


def test_generic_cross_session_and_filesystem_callers_cannot_reuse_reviewed_authority(tmp_path: Path) -> None:
    engine, session_id, other_session, blobs = _reviewed_source_harness(tmp_path)
    first_blob, second_blob = blobs
    facts = _reviewed_source_facts(blob_id=str(first_blob.id))
    pipeline = _named_reviewed_pipeline(tmp_path, facts)

    generic = build_set_pipeline_candidate(
        pipeline,
        _empty_state(),
        _trained_context(data_dir=tmp_path, session_engine=engine, session_id=session_id, user_id="review-owner"),
    )
    assert generic.acceptable is False

    other_facts = _reviewed_source_facts(blob_id=str(second_blob.id))
    other_authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=other_session,
        user_id="review-owner",
        reviewed_facts=other_facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(other_facts),
    )
    cross_session = build_set_pipeline_candidate(
        _named_reviewed_pipeline(tmp_path, other_facts),
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=other_authority,
        ),
    )
    assert cross_session.acceptable is False

    filesystem_facts = _reviewed_source_facts(blob_id=str(first_blob.id), path="/etc/looks-reviewed.csv")
    with pytest.raises(AuditIntegrityError, match="storage path"):
        resolve_reviewed_source_authority(
            engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_facts=filesystem_facts,
            expected_reviewed_anchor_hash=reviewed_anchor_hash(filesystem_facts),
        )


def test_reviewed_source_facts_for_another_source_cannot_authorize_candidate(tmp_path: Path) -> None:
    engine, session_id, _other_session, blobs = _reviewed_source_harness(tmp_path)
    blob = blobs[0]
    other_facts = _reviewed_source_facts(blob_id=str(blob.id), source_name="other")
    authority = resolve_reviewed_source_authority(
        engine=engine,
        session_id=session_id,
        user_id="review-owner",
        reviewed_facts=other_facts,
        expected_reviewed_anchor_hash=reviewed_anchor_hash(other_facts),
    )
    candidate_facts = deepcopy(other_facts)
    candidate_source = next(iter(candidate_facts["reviewed_sources"].values()))
    candidate_source["name"] = "source"

    candidate = build_set_pipeline_candidate(
        _named_reviewed_pipeline(tmp_path, candidate_facts),
        _empty_state(),
        _trained_context(
            data_dir=tmp_path,
            session_engine=engine,
            session_id=session_id,
            user_id="review-owner",
            reviewed_source_authority=authority,
        ),
    )

    assert candidate.acceptable is False


def test_candidate_uses_final_request_scoped_profile_validation(tmp_path: Path) -> None:
    args = _linear_args(tmp_path)
    args["source"]["on_success"] = "main"
    args["nodes"] = []
    args["edges"] = []
    args["metadata"] = {"name": "final-profile-reject"}
    state = _empty_state()
    context, catalog = _final_validation_rejecting_context(data_dir=tmp_path)

    candidate = build_set_pipeline_candidate(args, state, context)

    assert candidate.result.success is True
    assert candidate.result.updated_state.validate().is_valid is True
    assert (candidate.acceptable, len(catalog.validated_states)) == (False, 1)
    assert catalog.validated_states[0] is candidate.result.updated_state
    assert candidate.result.validation.errors[0].error_code == "profile_complete_state_rejected"

    normalized_again = normalize_tool_result_validation(candidate.result, catalog)

    assert normalized_again is candidate.result
    assert len(catalog.validated_states) == 1
    assert "_validation_snapshot_hash" not in candidate.result.to_dict()
    assert "_validation_snapshot_hash" not in repr(candidate.result)

    catalog.validated_states.clear()
    public_result = execute_tool(
        "set_pipeline",
        args,
        state,
        catalog,
        plugin_snapshot=context.plugin_snapshot,
        data_dir=str(tmp_path),
    )

    assert public_result.updated_state == candidate.result.updated_state
    assert public_result.validation == candidate.result.validation
    assert public_result.validation.is_valid is False
    assert len(catalog.validated_states) == 2
    assert catalog.validated_states[0] is state
    assert catalog.validated_states[1] is public_result.updated_state


@pytest.mark.parametrize("rejected", [False, True], ids=("success", "rejection"))
def test_public_set_pipeline_validates_current_and_candidate_exactly_once(
    tmp_path: Path,
    *,
    rejected: bool,
) -> None:
    args = _linear_args(tmp_path)
    args["source"]["on_success"] = "main"
    args["nodes"] = []
    args["edges"] = []
    if rejected:
        args["sources"] = {}
    state = _empty_state()
    context, catalog = _final_validation_rejecting_context(data_dir=tmp_path)

    result = execute_tool(
        "set_pipeline",
        args,
        state,
        catalog,
        plugin_snapshot=context.plugin_snapshot,
        data_dir=str(tmp_path),
    )

    assert result.success is not rejected
    assert len(catalog.validated_states) == 2
    assert catalog.validated_states[0] is state
    assert catalog.validated_states[1] is result.updated_state


def test_normalizer_revalidates_for_a_different_snapshot_and_preserves_rejection(tmp_path: Path) -> None:
    args = _linear_args(tmp_path)
    args["sources"] = {}
    state = _empty_state()
    context, _catalog = _final_validation_rejecting_context(data_dir=tmp_path)
    candidate = build_set_pipeline_candidate(args, state, context)
    rejection = candidate.result.validation.errors[0]

    full = create_catalog_service()
    original = context.plugin_snapshot
    other_snapshot = PluginAvailabilitySnapshot.create(
        policy_hash="different-request-policy",
        principal_scope=original.principal_scope,
        available=original.available,
        unavailable=original.unavailable,
        selected=original.selected,
        usable_profile_aliases=original.usable_profile_aliases,
        selected_profile_aliases=original.selected_profile_aliases,
        control_modes=original.control_modes,
        binding_generation_fingerprint=original.binding_generation_fingerprint,
    )
    other_catalog = _FinalValidationRejectingCatalog.for_trained_operator(full, other_snapshot)
    other_catalog.validated_states = []

    revalidated = normalize_tool_result_validation(candidate.result, other_catalog)

    assert other_snapshot.snapshot_hash != original.snapshot_hash
    assert revalidated is not candidate.result
    assert revalidated == candidate.result
    assert revalidated.updated_state is candidate.result.updated_state
    assert other_catalog.validated_states == [candidate.result.updated_state]
    assert tuple(entry for entry in revalidated.validation.errors if entry.component == "rejected_mutation") == (rejection,)
    assert revalidated._validation_snapshot_hash == other_snapshot.snapshot_hash


def _named_multi_source_queue_args(tmp_path: Path) -> dict[str, Any]:
    return {
        "sources": {
            "orders": {
                "plugin": "csv",
                "on_success": "inbound",
                "options": {
                    "path": str(tmp_path / "blobs" / "orders.csv"),
                    "schema": {"mode": "observed"},
                },
            },
            "refunds": {
                "plugin": "csv",
                "on_success": "inbound",
                "options": {
                    "path": str(tmp_path / "blobs" / "refunds.csv"),
                    "schema": {"mode": "observed"},
                },
            },
        },
        "nodes": [
            {
                "id": "inbound",
                "node_type": "queue",
                "input": "inbound",
                "options": {"description": "Orders and refunds interleave here"},
            },
            {
                "id": "consume_inbound",
                "node_type": "transform",
                "plugin": "passthrough",
                "input": "inbound",
                "on_success": "main",
                "on_error": "discard",
                "options": {"schema": {"mode": "observed"}},
            },
        ],
        "edges": [],
        "outputs": [
            {
                "sink_name": "main",
                "plugin": "json",
                "options": _file_options(tmp_path / "outputs" / "queued.jsonl") | {"format": "jsonl"},
                "on_write_failure": "discard",
            }
        ],
        "metadata": {"name": "named-multi-source-queue"},
    }


def _fork_coalesce_args(tmp_path: Path) -> dict[str, Any]:
    args = _linear_args(tmp_path)
    args["source"]["on_success"] = "rows"
    args["nodes"] = [
        {
            "id": "fork",
            "node_type": "gate",
            "input": "rows",
            "condition": "'all'",
            "routes": {"all": "fork"},
            "fork_to": ["original", "copy"],
        },
        {
            "id": "original_path",
            "node_type": "transform",
            "plugin": "passthrough",
            "input": "original",
            "on_success": "original_out",
            "on_error": "discard",
            "options": {"schema": {"mode": "observed"}},
        },
        {
            "id": "copy_path",
            "node_type": "transform",
            "plugin": "passthrough",
            "input": "copy",
            "on_success": "copy_out",
            "on_error": "discard",
            "options": {"schema": {"mode": "observed"}},
        },
        {
            "id": "merge",
            "node_type": "coalesce",
            "input": "branches",
            "branches": {"original": "original_out", "copy": "copy_out"},
            "policy": "require_all",
            "merge": "nested",
            "on_success": "main",
            "on_error": "discard",
            "options": {"schema": {"mode": "observed"}},
        },
    ]
    args["edges"] = []
    args["metadata"] = {"name": "fork-coalesce"}
    return args


def _gate_args(tmp_path: Path) -> dict[str, Any]:
    args = _linear_args(tmp_path)
    args["nodes"] = [
        {
            "id": "threshold",
            "node_type": "gate",
            "input": "rows",
            "condition": "row['score'] >= 0.5",
            "routes": {"true": "high", "false": "low"},
        }
    ]
    args["edges"] = []
    args["outputs"] = [
        {
            "sink_name": name,
            "plugin": "json",
            "options": _file_options(tmp_path / "outputs" / f"{name}.jsonl") | {"format": "jsonl"},
            "on_write_failure": "discard",
        }
        for name in ("high", "low")
    ]
    args["metadata"] = {"name": "gate"}
    return args


def _aggregation_args(tmp_path: Path) -> dict[str, Any]:
    args = _linear_args(tmp_path)
    args["nodes"] = [
        {
            "id": "stats",
            "node_type": "aggregation",
            "plugin": "batch_stats",
            "input": "rows",
            "on_success": "main",
            "on_error": "discard",
            "options": {"schema": {"mode": "observed"}, "value_field": "score"},
        }
    ]
    args["edges"] = []
    args["metadata"] = {"name": "aggregation"}
    return args


def _structured_llm_args(tmp_path: Path) -> dict[str, Any]:
    args = _linear_args(tmp_path)
    args["nodes"] = [
        {
            "id": "classify",
            "node_type": "transform",
            "plugin": "llm",
            "input": "rows",
            "on_success": "main",
            "on_error": "discard",
            "options": {
                "provider": "azure",
                "deployment_name": "candidate-test",
                "endpoint": "https://candidate-test.openai.azure.com",
                "api_key": {"secret_ref": "AZURE_OPENAI_API_KEY"},
                "prompt_template": "Classify {{ text }}",
                # Multi-query execution must use the pooled path so capacity
                # retries are bounded by the configured pool controller.
                "pool_size": 2,
                "queries": [
                    {
                        "name": "colour",
                        "input_fields": {"text": "text"},
                        "template": "Classify {{ text }}",
                        "response_format": "structured",
                        "output_fields": [
                            {"suffix": "label", "type": "string"},
                            {"suffix": "score", "type": "number"},
                        ],
                    }
                ],
                "schema": {"mode": "observed"},
            },
        }
    ]
    args["edges"] = []
    args["metadata"] = {"name": "structured-llm"}
    return args


def _secret_bearing_structured_fork_coalesce_args(tmp_path: Path) -> dict[str, Any]:
    """A complete fork whose enriched branch consumes structured LLM fields."""
    args = _fork_coalesce_args(tmp_path)
    args["source"]["options"]["schema"] = {
        "mode": "flexible",
        "fields": ["text: str"],
        "guaranteed_fields": ["text"],
    }
    args["nodes"][2] = {
        "id": "classify",
        "node_type": "transform",
        "plugin": "llm",
        "input": "copy",
        "on_success": "classified",
        "on_error": "discard",
        "options": {
            "provider": "azure",
            "deployment_name": "candidate-test",
            "endpoint": "https://candidate-test.openai.azure.com",
            "api_key": {"secret_ref": "AZURE_OPENAI_API_KEY"},
            "prompt_template": "Classify {{ text }}",
            "required_input_fields": ["text"],
            "pool_size": 2,
            "queries": [
                {
                    "name": "colour",
                    "input_fields": {"text": "text"},
                    "template": "Classify {{ text }}",
                    "response_format": "structured",
                    "output_fields": [
                        {"suffix": "label", "type": "string"},
                        {"suffix": "score", "type": "number"},
                    ],
                }
            ],
            "schema": {
                "mode": "flexible",
                "fields": ["text: str"],
                "guaranteed_fields": ["text"],
            },
        },
    }
    args["nodes"].insert(
        3,
        {
            "id": "select_classification",
            "node_type": "transform",
            "plugin": "field_mapper",
            "input": "classified",
            "on_success": "copy_out",
            "on_error": "discard",
            "options": {
                "schema": {
                    "mode": "flexible",
                    "fields": ["colour_label: str", "colour_score: float"],
                    "required_fields": ["colour_label", "colour_score"],
                },
                "mapping": {
                    "colour_label": "colour_label",
                    "colour_score": "colour_score",
                },
                "select_only": True,
                "strict": True,
            },
        },
    )
    args["metadata"] = {"name": "secret-bearing-structured-fork-coalesce"}
    return args


def _multi_output_args(tmp_path: Path) -> dict[str, Any]:
    args = _linear_args(tmp_path)
    args["outputs"] = [
        {
            "sink_name": "main",
            "plugin": "json",
            "options": _file_options(tmp_path / "outputs" / "main.jsonl") | {"format": "jsonl"},
            "on_write_failure": "discard",
        },
        {
            "sink_name": "quarantine",
            "plugin": "csv",
            "options": _file_options(tmp_path / "outputs" / "quarantine.csv"),
            "on_write_failure": "discard",
        },
    ]
    args["source"]["on_validation_failure"] = "quarantine"
    args["metadata"] = {"name": "multi-output"}
    return args


_EXPECTED_STATE_HASHES = {
    "linear": "7a9e54170c34e1e4672d7b2969860b35fedb35a73fc437d8e25f527712b322b9",
    "named_multi_source_queue": "bebdc72124a73cd5ebb5ddd1c7660345004c4276b289b4651980131d133ff61b",
    "fork_coalesce": "b1383d953ab65ab7339daf6223be072c8084ab7d59af4713bc9f8dfc85a52727",
    "gate": "4c8954595ad404c45adcf43cdaf7ce1a5c6e1154afa7e5f64b21e44d7fc19cb5",
    "aggregation": "ef2ecf5c7f1d7ff9f00709175102e4777563b43cc3d06edaac29b9d87d06073b",
    "structured_llm": "3815378a23c396501b91809644b330c423367ea26d20ec63bb0d64caed39a1ee",
    "multi_output": "50e5c9c2b421bdaa2db73585d569ae9b1f21d2f79c6a8d9adfb184420a8a0065",
}

_EXPECTED_WARNINGS = {
    "named_multi_source_queue": (
        {
            "component": "node:inbound",
            "message": "Node 'inbound' has no outgoing edges — its output is not connected to any downstream node or sink.",
            "severity": "medium",
        },
    ),
    "structured_llm": (
        {
            "component": "node:classify",
            "message": (
                "LLM node 'classify' has no authorized prompt-injection shield in front of it. Recommend inserting "
                "azure_prompt_shield (or the deployment equivalent prompt-injection shield) between the external-content "
                "fetch step and this LLM. The current draft routes internet-controlled text directly into the LLM without "
                "that shield, which is a prompt-injection exposure on untrusted remote content, but continuing without it "
                "is allowed. [user_term: prompt_injection_shield_recommendation]"
            ),
            "severity": "medium",
        },
    ),
}


def _portable_state_content(value: Any, *, data_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _portable_state_content(item, data_dir=data_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_state_content(item, data_dir=data_dir) for item in value]
    if isinstance(value, str):
        return value.replace(str(data_dir), "<DATA_DIR>")
    return value


@pytest.mark.parametrize(
    ("case", "factory", "expected_sources", "expected_node_kinds", "expected_outputs"),
    [
        ("linear", _linear_args, {"source"}, {"transform"}, {"main"}),
        (
            "named_multi_source_queue",
            _named_multi_source_queue_args,
            {"orders", "refunds"},
            {"queue", "transform"},
            {"main"},
        ),
        ("fork_coalesce", _fork_coalesce_args, {"source"}, {"gate", "transform", "coalesce"}, {"main"}),
        ("gate", _gate_args, {"source"}, {"gate"}, {"high", "low"}),
        ("aggregation", _aggregation_args, {"source"}, {"aggregation"}, {"main"}),
        ("structured_llm", _structured_llm_args, {"source"}, {"transform"}, {"main"}),
        ("multi_output", _multi_output_args, {"source"}, {"transform"}, {"main", "quarantine"}),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_current_executor_normalizes_supported_pipeline_shapes(
    tmp_path: Path,
    case: str,
    factory: Any,
    expected_sources: set[str],
    expected_node_kinds: set[str],
    expected_outputs: set[str],
) -> None:
    state = _empty_state()
    args = factory(tmp_path)

    result = _execute_set_pipeline(args, state, _trained_context(data_dir=tmp_path))

    assert result.success is True, (case, result.to_dict())
    assert result.updated_state.version == state.version + 1
    assert set(result.updated_state.sources) == expected_sources
    assert {node.node_type for node in result.updated_state.nodes} == expected_node_kinds
    assert {output.name for output in result.updated_state.outputs} == expected_outputs
    expected_affected_sources = {"source" if source_name == "source" else f"source:{source_name}" for source_name in expected_sources}
    assert set(result.affected_nodes) == expected_affected_sources | {node.id for node in result.updated_state.nodes} | expected_outputs
    assert result.validation.is_valid is True
    assert result.validation.errors == ()
    assert tuple(item.to_dict() for item in result.validation.warnings) == _EXPECTED_WARNINGS.get(case, ())
    assert result.validation.suggestions == ()
    assert result.validation.semantic_contracts == ()
    assert result.to_dict()["validation"]["is_valid"] is True
    assert result.to_dict()["validation"]["graph_repair_suggestions"] == []
    normalized = result.updated_state.to_dict()
    assert normalized["metadata"]["name"] == args["metadata"]["name"]
    assert set(normalized["sources"]) == expected_sources
    assert {node["id"] for node in normalized["nodes"]} == {node["id"] for node in args["nodes"]}
    assert normalized["edges"] == args["edges"]
    assert {output["name"] for output in normalized["outputs"]} == expected_outputs
    assert stable_hash(_portable_state_content(normalized, data_dir=tmp_path)) == _EXPECTED_STATE_HASHES[case]
    assert result.data is None


@pytest.mark.parametrize(
    "factory",
    [
        _linear_args,
        _named_multi_source_queue_args,
        _fork_coalesce_args,
        _gate_args,
        _aggregation_args,
        _structured_llm_args,
        _multi_output_args,
    ],
)
def test_candidate_matches_executor_for_custody_safe_arguments(tmp_path: Path, factory: Any) -> None:
    state = _empty_state()
    args = factory(tmp_path)
    context = _trained_context(data_dir=tmp_path)

    candidate = build_set_pipeline_candidate(args, state, context)
    executor_result = _execute_set_pipeline(args, state, context)

    assert candidate.acceptable is True
    assert candidate.prepared_inline_blob is None
    assert candidate.result == executor_result


def _semantic_failure_cases(tmp_path: Path) -> list[tuple[str, dict[str, Any], ToolContext, str, str | None]]:
    unknown = _linear_args(tmp_path)
    unknown["nodes"][0]["plugin"] = "not_installed"

    blocked = _linear_args(tmp_path)

    invalid_profile = _linear_args(tmp_path)

    invalid_options = _linear_args(tmp_path)
    invalid_options["nodes"][0]["options"] = {}

    escaping_path = _linear_args(tmp_path)
    escaping_path["outputs"][0]["options"]["path"] = "/etc/candidate-escape.json"

    invalid_gate = _gate_args(tmp_path)
    invalid_gate["nodes"][0]["condition"] = "row["

    manual_blob_ref = _linear_args(tmp_path)
    manual_blob_ref["source"]["options"]["blob_ref"] = str(uuid4())

    literal_credential = _structured_llm_args(tmp_path)
    literal_credential["nodes"][0]["options"]["api_key"] = "literal-secret-must-not-leak"

    stale_review = _structured_llm_args(tmp_path)
    stale_review["nodes"][0]["options"][INTERPRETATION_REQUIREMENTS_KEY] = [
        {
            "id": "prompt_template_review:classify",
            "kind": "llm_prompt_template",
            "user_term": "llm_prompt_template:classify",
            "status": "resolved",
            "draft": "Old prompt",
            "event_id": "stale-event",
            "accepted_value": "Old prompt",
            "accepted_artifact_hash": None,
            "resolved_prompt_template_hash": stable_hash("Old prompt"),
        }
    ]

    credential_error = (
        "Credential field(s) contain literal value(s): classify:api_key. Literal credential values were not stored. "
        "Set `<field>: {secret_ref: NAME}` directly in the node's options when calling set_pipeline / upsert_node. "
        "(The marker is stripped before option validation and resolved at execution time.) This rejection left pipeline "
        "state unchanged: repair by re-issuing only the rejected call with the marker substituted for the literal value "
        "— do not rebuild the pipeline from scratch. For a component already in state, patching just that component "
        "(patch_source_options / patch_node_options / patch_output_options) with the marker is the minimal correction. "
        "Alternatively, after the node already exists in state, call list_secret_refs -> validate_secret_ref -> "
        "wire_secret_ref to attach the marker post-hoc."
    )
    return [
        (
            "unknown_plugin",
            unknown,
            _trained_context(data_dir=tmp_path),
            "Node 'copy': transform plugin selection is unavailable (plugin_not_installed)",
            "plugin_not_installed",
        ),
        (
            "blocked_plugin",
            blocked,
            _blocked_context(PluginId("transform", "passthrough"), data_dir=tmp_path),
            "Node 'copy': transform plugin selection is unavailable (plugin_not_enabled)",
            "plugin_not_enabled",
        ),
        (
            "profile_validation",
            invalid_profile,
            _profile_rejecting_context(data_dir=tmp_path),
            "Node 'copy': Invalid options for transform 'passthrough': profile_unavailable",
            None,
        ),
        (
            "invalid_options",
            invalid_options,
            _trained_context(data_dir=tmp_path),
            "Node 'copy': Invalid options for transform 'passthrough': schema: Field required. Use 'schema: {mode: "
            "observed}' to infer types from data, or provide explicit field definitions with mode (fixed/flexible).",
            None,
        ),
        (
            "escaping_path",
            escaping_path,
            _trained_context(data_dir=tmp_path),
            "Output 'main': Path violation (S2): 'path' value '/etc/candidate-escape.json' is outside the allowed "
            f"directories. Sink output paths must be under {tmp_path / 'outputs'}/ or this session's own "
            f"{tmp_path / 'blobs'}/<session>/ subtree.",
            None,
        ),
        (
            "invalid_gate",
            invalid_gate,
            _trained_context(data_dir=tmp_path),
            "Node 'threshold': Invalid gate condition syntax: Invalid syntax: '[' was never closed",
            None,
        ),
        (
            "manual_blob_ref",
            manual_blob_ref,
            _trained_context(data_dir=tmp_path),
            "Use set_source_from_blob, source.blob_id, or source.inline_blob to bind a blob to the source. set_pipeline "
            "must not be called with 'blob_ref' in source.options because it cannot enforce that 'path' equals the "
            "blob's canonical storage_path.",
            None,
        ),
        (
            "credential_policy",
            literal_credential,
            _trained_context(data_dir=tmp_path),
            credential_error,
            None,
        ),
        (
            "resolver_owned_interpretation_review",
            stale_review,
            _trained_context(data_dir=tmp_path),
            "Node 'classify': set_pipeline options.interpretation_requirements[0] includes resolver-owned status "
            "'resolved'. Composer tool input may stage pending review requirements only; resolved review metadata may "
            "only be written by resolve_interpretation_event.",
            None,
        ),
    ]


@pytest.mark.parametrize("case_index", range(9))
def test_current_executor_semantic_failures_are_atomic(tmp_path: Path, case_index: int) -> None:
    case, args, context, expected_error, expected_error_code = _semantic_failure_cases(tmp_path)[case_index]
    state = _empty_state()
    before = state.to_dict()

    result = _execute_set_pipeline(args, state, context)

    assert result.success is False, (case, result.to_dict())
    assert result.updated_state is state
    assert result.updated_state.to_dict() == before
    assert result.affected_nodes == ()
    assert result.validation.is_valid is False
    assert result.validation.errors[0].component == "rejected_mutation"
    assert result.data["error"] == expected_error
    assert result.data.get("error_code") == expected_error_code
    assert result.validation.errors[0].message == expected_error
    assert result.validation.errors[0].error_code == expected_error_code
    assert result.to_dict()["version"] == state.version
    assert "literal-secret-must-not-leak" not in repr(result.to_dict())


@pytest.mark.parametrize("case_index", range(9))
def test_candidate_matches_executor_semantic_failures_without_side_effects(tmp_path: Path, case_index: int) -> None:
    case, args, context, _expected_error, _expected_error_code = _semantic_failure_cases(tmp_path)[case_index]
    state = _empty_state()
    before = state.to_dict()

    candidate = build_set_pipeline_candidate(args, state, context)
    executor_result = _execute_set_pipeline(args, state, context)

    assert candidate.acceptable is False, case
    assert candidate.prepared_inline_blob is None
    assert candidate.result == executor_result
    assert state.to_dict() == before


def test_current_executor_can_return_success_with_invalid_graph_candidate(tmp_path: Path) -> None:
    """The handler reports constructed state separately from acceptability.

    A missing output leaves a structurally constructed draft whose validation
    is invalid.  The extracted candidate boundary must retain both facts so
    explicit approval can refuse publication without rewriting the result.
    """
    args = _linear_args(tmp_path)
    args["outputs"] = []
    state = _empty_state()

    result = _execute_set_pipeline(args, state, _trained_context(data_dir=tmp_path))

    assert result.success is True
    assert result.updated_state.version == state.version + 1
    assert result.validation.is_valid is False
    assert result.validation.errors

    candidate = build_set_pipeline_candidate(args, state, _trained_context(data_dir=tmp_path))
    assert candidate.result == result
    assert candidate.result.success is True
    assert candidate.acceptable is False


def test_current_executor_reopens_stale_authoritative_review(tmp_path: Path) -> None:
    """Changed reviewed content must not inherit a stale approval event."""
    original_args = _structured_llm_args(tmp_path)
    first = _execute_set_pipeline(original_args, _empty_state(), _trained_context(data_dir=tmp_path))
    assert first.success and first.validation.is_valid
    original_node = first.updated_state.nodes[0]
    original_options = deep_thaw(original_node.options)
    requirements = original_options[INTERPRETATION_REQUIREMENTS_KEY]
    prompt_requirement = next(item for item in requirements if item["kind"] == "llm_prompt_template")
    prompt = original_options["prompt_template"]
    resolved = {
        **prompt_requirement,
        "status": "resolved",
        "event_id": "authoritative-prompt-review",
        "accepted_value": prompt,
        "accepted_artifact_hash": None,
        "resolved_prompt_template_hash": stable_hash(prompt),
    }
    previous = replace(
        first.updated_state,
        nodes=(replace(original_node, options={**original_options, INTERPRETATION_REQUIREMENTS_KEY: [resolved]}),),
    )
    changed_args = _structured_llm_args(tmp_path)
    changed_args["nodes"][0]["options"]["prompt_template"] = "Reclassify {{ text }}"

    result = _execute_set_pipeline(changed_args, previous, _trained_context(data_dir=tmp_path))

    assert result.success and result.validation.is_valid
    reconciled = result.updated_state.nodes[0].options[INTERPRETATION_REQUIREMENTS_KEY]
    current = next(item for item in reconciled if item["kind"] == "llm_prompt_template")
    assert current["draft"] == "Reclassify {{ text }}"
    assert current["status"] == "pending"
    assert current["event_id"] is None
    assert current["accepted_value"] is None
    assert current["resolved_prompt_template_hash"] is None


def test_secret_bearing_structured_llm_probe_preserves_contract_and_authored_marker(tmp_path: Path) -> None:
    """Resolver-free probes must expose LLM fields without rewriting secret wiring."""
    args = _secret_bearing_structured_fork_coalesce_args(tmp_path)
    original_marker = deepcopy(args["nodes"][2]["options"]["api_key"])

    result = _execute_set_pipeline(args, _empty_state(), _trained_context(data_dir=tmp_path))

    assert result.success is True, result.to_dict()
    assert result.validation.is_valid, result.validation.errors
    assert not any("computed contract probe" in warning.message.lower() for warning in result.validation.warnings)
    mapper_contract = next(contract for contract in result.validation.edge_contracts if contract.to_id == "select_classification")
    assert {"colour_label", "colour_score"} <= set(mapper_contract.producer_guarantees)
    assert {"colour_label", "colour_score"} <= set(mapper_contract.consumer_requires)
    assert mapper_contract.satisfied is True
    assert args["nodes"][2]["options"]["api_key"] == original_marker == {"secret_ref": "AZURE_OPENAI_API_KEY"}
    assert deep_thaw(result.updated_state.nodes[2].options["api_key"]) == original_marker


def test_structured_llm_probe_failure_abstains_and_blocks_downstream_field_mapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconstructed LLM output fields never become field-mapper authority."""
    from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
    from elspeth.plugins.infrastructure.templates import TemplateError

    args = _secret_bearing_structured_fork_coalesce_args(tmp_path)
    context = _trained_context(data_dir=tmp_path)
    original_manager = get_shared_plugin_manager()
    secret_canary = "RAW-PROBE-ERROR-SECRET-CANARY"

    class _FailingLlmProbeManager:
        def __getattr__(self, name: str):
            return getattr(original_manager, name)

        def get_transforms(self):
            return original_manager.get_transforms()

        def create_transform(self, plugin_name: str, options: dict[str, Any]):
            if plugin_name == "llm":
                raise TemplateError(secret_canary)
            return original_manager.create_transform(plugin_name, options)

    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: _FailingLlmProbeManager(),
    )

    result = _execute_set_pipeline(args, _empty_state(), context)

    assert result.success is True
    assert not result.validation.is_valid
    warning = next(
        warning
        for warning in result.validation.warnings
        if warning.component == "node:classify" and "computed contract probe" in warning.message.lower()
    )
    assert "TemplateError" in warning.message
    assert secret_canary not in repr(result.to_dict())
    mapper_contract = next(contract for contract in result.validation.edge_contracts if contract.to_id == "select_classification")
    assert "colour_label" not in mapper_contract.producer_guarantees
    assert "colour_score" not in mapper_contract.producer_guarantees
    assert {"colour_label", "colour_score"} <= set(mapper_contract.consumer_requires)
    assert {"colour_label", "colour_score"} <= set(mapper_contract.missing_fields)
    assert mapper_contract.satisfied is False


def _session_with_user_message() -> tuple[Any, str, str]:
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    session_id = str(uuid4())
    message_id = str(uuid4())
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="candidate-user",
                auth_provider_type="local",
                title="candidate characterization",
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            insert(chat_messages_table).values(
                id=message_id,
                session_id=session_id,
                role="user",
                content="Use this CSV: name,score\nada,42\n",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return engine, session_id, message_id


@pytest.mark.asyncio
async def test_current_executor_inline_blob_effects_are_single_settlement(tmp_path: Path) -> None:
    engine, session_id, message_id = _session_with_user_message()
    content = "name,score\nada,42\n"
    args = {
        "source": {
            "plugin": "csv",
            "on_success": "rows",
            "options": {"schema": {"mode": "observed"}},
            "inline_blob": {
                "filename": "ada.csv",
                "mime_type": "text/csv",
                "content": content,
            },
        },
        "nodes": [],
        "edges": [],
        "outputs": [],
        "metadata": {"name": "inline"},
    }
    state = _empty_state()
    before_state = deepcopy(state.to_dict())
    before_files = tuple((tmp_path / "blobs").rglob("*")) if (tmp_path / "blobs").exists() else ()
    with engine.begin() as conn:
        before_blob_rows = conn.execute(select(func.count()).select_from(blobs_table)).scalar_one()
        before_chat_rows = conn.execute(select(func.count()).select_from(chat_messages_table)).scalar_one()
        before_quota = conn.execute(select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0))).scalar_one()

    context = _trained_context(
        data_dir=tmp_path,
        session_engine=engine,
        session_id=session_id,
        user_message_id=message_id,
        user_message_content=f"Use this CSV: {content}",
    )
    recorder = BufferingRecorder()

    candidate = build_set_pipeline_candidate(args, state, context)

    with engine.begin() as conn:
        candidate_blob_rows = conn.execute(select(func.count()).select_from(blobs_table)).scalar_one()
        candidate_chat_rows = conn.execute(select(func.count()).select_from(chat_messages_table)).scalar_one()
        candidate_quota = conn.execute(select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0))).scalar_one()
    candidate_files = tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file())

    assert candidate.acceptable is False  # This intentionally incomplete graph still needs outputs.
    assert candidate.result.success is True
    assert candidate.prepared_inline_blob is not None
    assert candidate.result.data is None
    assert state.to_dict() == before_state
    assert candidate_blob_rows == before_blob_rows == 0
    assert candidate_chat_rows == before_chat_rows == 1
    assert candidate_quota == before_quota == 0
    assert candidate_files == before_files == ()
    assert recorder.invocations == ()

    audit = begin_dispatch(
        "candidate-inline-call",
        "set_pipeline",
        args,
        version_before=state.version,
        actor="candidate-characterization",
    )

    async def _dispatch() -> Any:
        return _execute_set_pipeline(args, state, context)

    outcome = await dispatch_with_audit(
        recorder=recorder,
        audit=audit,
        do_dispatch=_dispatch,
        version_after_provider=lambda value: value.updated_state.version,
        arg_error_payload_factory=lambda exc: {"error": exc.args[0]},
    )
    result = outcome.result

    with engine.begin() as conn:
        rows = conn.execute(select(blobs_table).where(blobs_table.c.session_id == session_id)).mappings().all()
        after_chat_rows = conn.execute(select(func.count()).select_from(chat_messages_table)).scalar_one()
        after_quota = conn.execute(select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0))).scalar_one()
    after_files = tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file())

    assert result.success is True, result.to_dict()
    assert state.to_dict() == before_state
    assert result.updated_state.version == state.version + 1
    assert before_blob_rows == 0
    assert len(rows) == 1
    assert rows[0]["size_bytes"] == len(content.encode("utf-8"))
    assert rows[0]["session_id"] == session_id
    assert rows[0]["filename"] == "ada.csv"
    assert rows[0]["mime_type"] == "text/csv"
    assert rows[0]["content_hash"] == content_hash(content.encode("utf-8"))
    assert rows[0]["storage_path"] == str(after_files[0])
    assert rows[0]["created_by"] == "assistant"
    assert rows[0]["source_description"] is None
    assert rows[0]["status"] == "ready"
    assert rows[0]["creation_modality"] == CreationModality.VERBATIM.value
    assert rows[0]["created_from_message_id"] == message_id
    assert rows[0]["creating_model_identifier"] is None
    assert rows[0]["creating_model_version"] is None
    assert rows[0]["creating_provider"] is None
    assert rows[0]["creating_composer_skill_hash"] is None
    assert rows[0]["creating_arguments_hash"] is None
    assert before_files == ()
    assert len(after_files) == 1
    assert after_files[0].read_text(encoding="utf-8") == content
    assert before_quota == 0
    assert after_quota == len(content.encode("utf-8"))
    assert after_chat_rows == before_chat_rows == 1
    assert len(recorder.invocations) == 1
    assert recorder.invocations[0].tool_name == "set_pipeline"
    assert recorder.invocations[0].status.value == "success"
    expected_inline_payload = {
        "blob_id": rows[0]["id"],
        "filename": "ada.csv",
        "mime_type": "text/csv",
        "size_bytes": len(content.encode("utf-8")),
        "content_hash": content_hash(content.encode("utf-8")),
    }
    assert deep_thaw(result.data) == {"inline_blob": expected_inline_payload}
    assert result.updated_state.to_dict()["sources"]["source"] == {
        "plugin": "csv",
        "on_success": "rows",
        "options": {
            "schema": {"mode": "observed"},
            "path": rows[0]["storage_path"],
            "blob_ref": rows[0]["id"],
        },
        "on_validation_failure": "discard",
    }
