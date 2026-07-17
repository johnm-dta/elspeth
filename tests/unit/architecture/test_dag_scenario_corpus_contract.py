from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError
from tests.fixtures.dag_scenario_corpus.schema import (
    EXPECTED_DIMENSIONS,
    EXPECTED_SCENARIOS,
    AuditEvidence,
    AuditRecordCount,
    CellStatus,
    ConfigEvidence,
    Dimension,
    EvidenceCell,
    EvidenceKind,
    EvidenceReference,
    GraphEvidence,
    HarnessCaseSpec,
    RecoveryEvidence,
    RunExpectation,
    RuntimeEvidence,
    ScenarioManifest,
    ScenarioRunEvidence,
    ScenarioSpec,
    Stage,
    Workflow,
)

EXPECTED_DIMENSION_VALUES = (
    "config",
    "build",
    "contracts",
    "runtime",
    "audit",
    "recovery",
    "concurrency",
    "freeform",
    "guided",
    "round_trip",
    "scale",
)

EXPECTED_SCENARIO_VALUES = (
    ("linear", "Linear source → transform → sink"),
    ("multiple-independent-sources", "Multiple independent sources"),
    ("multi-source-queue-fan-in", "Multi-source queue fan-in"),
    ("conditional-routing", "Conditional routing, including missing and error destinations"),
    ("fork-multiple-terminals-partial-failure", "Fork to multiple terminals with partial failure"),
    ("fork-coalesce-policies", "Fork and coalesce across every completion policy and merge strategy"),
    ("sequential-nested-fork-coalesce", "Sequential or nested forks and coalesces"),
    ("parallel-coalesces", "Parallel coalesces"),
    ("aggregation-immutable-batch", "Aggregation, batch closure, and immutable membership"),
    ("row-expansion-parent-child-recovery", "Row expansion with parent/child identity and recovery"),
    ("row-union-interleave", "Row union or interleave, whether supported or consistently rejected"),
    ("retry-quarantine-discard-routed-errors", "Retry, quarantine, discard, and routed error handling"),
    ("sink-write-pending-redrive", "Sink write and pending-sink redrive"),
    ("checkpoint-deterministic-resume", "Checkpoint and deterministic resume"),
    (
        "multi-worker-lease-reclaim-late-completion",
        "Multi-worker execution, lease expiry, reclaim, and late completion",
    ),
)


def _reference(*, kind: EvidenceKind = "harness") -> EvidenceReference:
    return EvidenceReference(
        id="evidence-1",
        kind=kind,
        locator="tests/path.py::test_case",
        claim="Exercises the production path",
        stages=("runtime",),
    )


def _expectation() -> RunExpectation:
    return RunExpectation(
        status="completed",
        output_rows=1,
        required_audit_record_types=("run_started",),
    )


def _case() -> HarnessCaseSpec:
    return HarnessCaseSpec(
        id="happy-path",
        workflow="run",
        fixture="linear.yaml",
        input_fixture="linear.jsonl",
        expected=_expectation(),
    )


def _scenario(cell: EvidenceCell) -> ScenarioSpec:
    return ScenarioSpec(
        id="linear",
        ordinal=1,
        title="Linear source → transform → sink",
        cases=(_case(),),
        dimensions={"config": cell},
    )


def _manifest(*cells: EvidenceCell) -> ScenarioManifest:
    return ScenarioManifest(
        schema_version=1,
        criteria_ref="docs/reference/dag-completeness.md",
        evidence=(_reference(),),
        scenarios=tuple(_scenario(cell) for cell in cells),
    )


def _valid_runtime() -> RuntimeEvidence:
    return RuntimeEvidence(
        attempted=True,
        run_id="run-1",
        status="completed",
        rows_processed=1,
        rows_succeeded=1,
        rows_failed=0,
        output_rows=1,
    )


def _valid_audit() -> AuditEvidence:
    return AuditEvidence(
        attempted=True,
        total_records=1,
        record_counts=(AuditRecordCount(record_type="run_started", count=1),),
        source_operation_count=1,
    )


def _valid_recovery() -> RecoveryEvidence:
    return RecoveryEvidence(
        attempted=True,
        database_reopened=True,
        checkpoint_id="checkpoint-1",
        checkpoint_sequence=1,
        can_resume=True,
        source_replayed=False,
        checkpoint_removed=True,
    )


def test_expected_dimension_and_scenario_constants_are_exact_and_ordered() -> None:
    assert EXPECTED_DIMENSIONS == EXPECTED_DIMENSION_VALUES
    assert EXPECTED_SCENARIOS == EXPECTED_SCENARIO_VALUES


def test_closed_vocabularies_are_exact() -> None:
    assert get_args(CellStatus) == ("pass", "partial", "fail", "unknown", "not_applicable")
    assert get_args(Dimension) == EXPECTED_DIMENSION_VALUES
    assert get_args(EvidenceKind) == ("harness", "pytest", "document", "decision")
    assert get_args(Stage) == ("config", "build", "runtime", "audit", "recovery")
    assert get_args(Workflow) == ("run", "recovery")


def test_non_empty_strings_are_strict_stripped_and_non_empty() -> None:
    reference = _reference()
    assert (
        EvidenceReference(
            id="  evidence-1  ",
            kind="harness",
            locator="  tests/path.py::test_case  ",
            claim="  claim  ",
        ).id
        == "evidence-1"
    )
    assert reference.executable is True

    with pytest.raises(ValidationError):
        EvidenceReference(id=" ", kind="harness", locator="test", claim="claim")
    with pytest.raises(ValidationError):
        EvidenceReference(id=1, kind="harness", locator="test", claim="claim")  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["harness", "pytest"])
def test_executable_evidence_kinds_are_executable(kind: EvidenceKind) -> None:
    assert _reference(kind=kind).executable is True


@pytest.mark.parametrize("kind", ["document", "decision"])
def test_non_executable_evidence_kinds_are_not_executable(kind: EvidenceKind) -> None:
    assert _reference(kind=kind).executable is False


def test_pass_without_evidence_is_rejected() -> None:
    with pytest.raises(ValidationError, match="pass"):
        EvidenceCell(status="pass")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "gap remains"),
        ("owner_issue", "elspeth-0123456789"),
        ("exit_gate", "close the gap"),
    ],
)
def test_pass_with_gap_metadata_is_rejected(field: str, value: str) -> None:
    values: dict[str, object] = {"status": "pass", "evidence": ("evidence-1",), field: value}
    with pytest.raises(ValidationError, match="pass"):
        EvidenceCell.model_validate(values)


@pytest.mark.parametrize("status", ["partial", "fail", "unknown"])
@pytest.mark.parametrize("missing", ["reason", "owner_issue", "exit_gate"])
def test_gap_status_without_owned_exit_gate_is_rejected(status: CellStatus, missing: str) -> None:
    values: dict[str, object] = {
        "status": status,
        "reason": "coverage gap",
        "owner_issue": "elspeth-0123456789",
        "exit_gate": "focused regression passes",
    }
    del values[missing]

    with pytest.raises(ValidationError, match=r"reason.*owner_issue.*exit_gate"):
        EvidenceCell.model_validate(values)


def test_owned_gap_status_is_accepted() -> None:
    cell = EvidenceCell(
        status="partial",
        reason="coverage gap",
        owner_issue="elspeth-0123456789",
        exit_gate="focused regression passes",
    )
    assert cell.status == "partial"


def test_invalid_issue_id_is_rejected() -> None:
    with pytest.raises(ValidationError, match="owner_issue"):
        EvidenceCell(
            status="fail",
            reason="gap",
            owner_issue="ELSPETH-123",
            exit_gate="fix lands",
        )


def test_not_applicable_without_reason_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not_applicable"):
        EvidenceCell(status="not_applicable")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence", ("evidence-1",)),
        ("owner_issue", "elspeth-0123456789"),
        ("exit_gate", "not relevant"),
    ],
)
def test_not_applicable_with_evidence_or_ownership_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {"status": "not_applicable", "reason": "not part of this scenario", field: value}
    with pytest.raises(ValidationError, match="not_applicable"):
        EvidenceCell.model_validate(values)


def test_not_applicable_with_reason_is_accepted() -> None:
    cell = EvidenceCell(status="not_applicable", reason="not part of this scenario")
    assert cell.reason == "not part of this scenario"


def test_unknown_fields_are_rejected_and_models_are_frozen() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ConfigEvidence.model_validate({"loaded": True, "settings_sha256": "abc", "unexpected": "field"})

    evidence = ConfigEvidence(loaded=True, settings_sha256="abc")
    with pytest.raises(ValidationError, match="frozen"):
        evidence.loaded = False


def test_strict_scalar_types_reject_coercion() -> None:
    with pytest.raises(ValidationError):
        ConfigEvidence(loaded=1, settings_sha256="abc")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RunExpectation(status="completed", output_rows="1", required_audit_record_types=())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        AuditRecordCount(record_type="run_started", count=True)


@pytest.mark.parametrize("missing", ["node_count", "edge_count", "topology_hash"])
def test_accepted_graph_requires_all_graph_facts(missing: str) -> None:
    values: dict[str, object] = {
        "accepted": True,
        "node_count": 3,
        "edge_count": 2,
        "topology_hash": "topology-sha",
    }
    del values[missing]
    with pytest.raises(ValidationError, match="accepted"):
        GraphEvidence.model_validate(values)


@pytest.mark.parametrize("field", ["rejection_type", "rejection_message"])
def test_accepted_graph_forbids_rejection_facts(field: str) -> None:
    values: dict[str, object] = {
        "accepted": True,
        "node_count": 3,
        "edge_count": 2,
        "topology_hash": "topology-sha",
        field: "rejected",
    }
    with pytest.raises(ValidationError, match="accepted"):
        GraphEvidence.model_validate(values)


@pytest.mark.parametrize("missing", ["rejection_type", "rejection_message"])
def test_rejected_graph_requires_both_rejection_facts(missing: str) -> None:
    values: dict[str, object] = {
        "accepted": False,
        "rejection_type": "ValueError",
        "rejection_message": "unsupported topology",
    }
    del values[missing]
    with pytest.raises(ValidationError, match="rejected"):
        GraphEvidence.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [("node_count", 3), ("edge_count", 2), ("topology_hash", "topology-sha")],
)
def test_rejected_graph_forbids_graph_facts(field: str, value: object) -> None:
    values: dict[str, object] = {
        "accepted": False,
        "rejection_type": "ValueError",
        "rejection_message": "unsupported topology",
        field: value,
    }
    with pytest.raises(ValidationError, match="rejected"):
        GraphEvidence.model_validate(values)


def test_accepted_and_rejected_graph_shapes_are_accepted() -> None:
    accepted = GraphEvidence(accepted=True, node_count=3, edge_count=2, topology_hash="topology-sha")
    rejected = GraphEvidence(accepted=False, rejection_type="ValueError", rejection_message="unsupported topology")
    assert accepted.node_count == 3
    assert rejected.rejection_type == "ValueError"


@pytest.mark.parametrize("missing", ["run_id", "status"])
def test_attempted_runtime_requires_run_identity_and_status(missing: str) -> None:
    values: dict[str, object] = {"attempted": True, "run_id": "run-1", "status": "completed"}
    del values[missing]
    with pytest.raises(ValidationError, match="attempted"):
        RuntimeEvidence.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "run-1"),
        ("status", "completed"),
        ("rows_processed", 1),
        ("rows_succeeded", 1),
        ("rows_failed", 1),
        ("output_rows", 1),
    ],
)
def test_unattempted_runtime_forbids_identity_status_and_nonzero_counters(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="unattempted"):
        RuntimeEvidence.model_validate({"attempted": False, field: value})


def test_attempted_and_unattempted_runtime_shapes_are_accepted() -> None:
    assert _valid_runtime().rows_processed == 1
    assert RuntimeEvidence(attempted=False).rows_processed == 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("total_records", 1),
        ("record_counts", ({"record_type": "run_started", "count": 1},)),
        ("source_operation_count", 1),
    ],
)
def test_unattempted_audit_forbids_records(field: str, value: object) -> None:
    values: dict[str, object] = {
        "attempted": False,
        "total_records": 0,
        "record_counts": (),
        "source_operation_count": 0,
        field: value,
    }
    with pytest.raises(ValidationError, match="unattempted"):
        AuditEvidence.model_validate(values)


def test_attempted_and_unattempted_audit_shapes_are_accepted() -> None:
    assert _valid_audit().record_counts[0].record_type == "run_started"
    unattempted = AuditEvidence(attempted=False, total_records=0, record_counts=(), source_operation_count=0)
    assert unattempted.total_records == 0


@pytest.mark.parametrize("missing", ["checkpoint_id", "checkpoint_sequence"])
def test_attempted_recovery_requires_checkpoint_identity(missing: str) -> None:
    values: dict[str, object] = {
        "attempted": True,
        "database_reopened": True,
        "checkpoint_id": "checkpoint-1",
        "checkpoint_sequence": 1,
        "can_resume": True,
        "source_replayed": False,
        "checkpoint_removed": True,
    }
    del values[missing]
    with pytest.raises(ValidationError, match="attempted"):
        RecoveryEvidence.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_id", "checkpoint-1"),
        ("checkpoint_sequence", 1),
        ("database_reopened", True),
        ("can_resume", True),
        ("source_replayed", True),
        ("checkpoint_removed", True),
    ],
)
def test_unattempted_recovery_forbids_checkpoint_identity_and_true_results(field: str, value: object) -> None:
    values: dict[str, object] = {
        "attempted": False,
        "database_reopened": False,
        "can_resume": False,
        "source_replayed": False,
        "checkpoint_removed": False,
        field: value,
    }
    with pytest.raises(ValidationError, match="unattempted"):
        RecoveryEvidence.model_validate(values)


def test_attempted_and_unattempted_recovery_shapes_are_accepted() -> None:
    assert _valid_recovery().checkpoint_id == "checkpoint-1"
    unattempted = RecoveryEvidence(
        attempted=False,
        database_reopened=False,
        can_resume=False,
        source_replayed=False,
        checkpoint_removed=False,
    )
    assert unattempted.checkpoint_id is None


def test_scenario_run_evidence_accepts_the_complete_observed_shape() -> None:
    evidence = ScenarioRunEvidence(
        schema_version=1,
        scenario_id="linear",
        case_id="happy-path",
        fixture_sha256="fixture-sha",
        config=ConfigEvidence(loaded=True, settings_sha256="settings-sha"),
        graph=GraphEvidence(accepted=True, node_count=3, edge_count=2, topology_hash="topology-sha"),
        runtime=_valid_runtime(),
        audit=_valid_audit(),
        recovery=_valid_recovery(),
        completed_stages=("config", "build", "runtime", "audit", "recovery"),
    )
    assert evidence.scenario_id == "linear"
    assert evidence.completed_stages[-1] == "recovery"


def test_manifest_verdict_is_complete_only_for_pass_or_not_applicable_cells() -> None:
    passing = EvidenceCell(status="pass", evidence=("evidence-1",))
    not_applicable = EvidenceCell(status="not_applicable", reason="not part of this scenario")
    assert _manifest(passing, not_applicable).verdict == "complete"

    for status in ("partial", "fail", "unknown"):
        gap = EvidenceCell(
            status=status,
            reason="coverage gap",
            owner_issue="elspeth-0123456789",
            exit_gate="focused regression passes",
        )
        assert _manifest(gap).verdict == "not_complete"
