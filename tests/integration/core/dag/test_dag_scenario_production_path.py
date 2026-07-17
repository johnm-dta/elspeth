"""Production-path integration evidence for maintained DAG scenarios."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from elspeth.engine.orchestrator import Orchestrator
from tests.fixtures.dag_scenario_corpus import harness as corpus_harness
from tests.fixtures.dag_scenario_corpus import loader as corpus_loader
from tests.fixtures.dag_scenario_corpus.harness import build_scenario, render_settings, run_scenario_case
from tests.fixtures.dag_scenario_corpus.loader import iter_harness_cases, load_manifest, resolve_fixture_path
from tests.fixtures.dag_scenario_corpus.plugins import install_corpus_plugin_manager
from tests.fixtures.dag_scenario_corpus.schema import HarnessCaseSpec, ScenarioRunEvidence, ScenarioSpec

MANIFEST = load_manifest()
RUN_CASES = [
    pytest.param(scenario, case, id=f"{scenario.id}:{case.id}") for scenario, case in iter_harness_cases(MANIFEST) if case.workflow == "run"
]
RECOVERY_CASES = [
    pytest.param(scenario, case, id=f"{scenario.id}:{case.id}")
    for scenario, case in iter_harness_cases(MANIFEST)
    if case.workflow == "recovery"
]


def _declared_case(scenario_id: str, case_id: str) -> tuple[ScenarioSpec, HarnessCaseSpec]:
    return next((scenario, case) for scenario, case in iter_harness_cases(MANIFEST) if (scenario.id, case.id) == (scenario_id, case_id))


def _assert_declared_run_evidence(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    evidence: ScenarioRunEvidence,
) -> None:
    fixture_path = resolve_fixture_path(case.fixture)
    input_path = resolve_fixture_path(case.input_fixture)
    expected_fixture_hash = hashlib.sha256(fixture_path.read_bytes() + b"\0" + input_path.read_bytes()).hexdigest()

    assert evidence.schema_version == 1
    assert (evidence.scenario_id, evidence.case_id) == (scenario.id, case.id)
    assert evidence.fixture_sha256 == expected_fixture_hash
    assert evidence.completed_stages == ("config", "build", "runtime", "audit")

    assert evidence.config.loaded is True
    assert len(evidence.config.settings_sha256) == 64

    assert evidence.graph.accepted is True
    assert evidence.graph.node_count is not None and evidence.graph.node_count > 0
    assert evidence.graph.edge_count is not None and evidence.graph.edge_count > 0
    assert evidence.graph.topology_hash is not None
    assert len(evidence.graph.topology_hash) == 64

    assert evidence.runtime.attempted is True
    assert evidence.runtime.status == case.expected.status
    assert evidence.runtime.output_rows == case.expected.output_rows

    assert evidence.audit.attempted is True
    assert evidence.audit.total_records > 0
    assert evidence.audit.total_records == sum(record.count for record in evidence.audit.record_counts)
    record_types = tuple(record.record_type for record in evidence.audit.record_counts)
    assert record_types == tuple(sorted(record_types))
    assert set(case.expected.required_audit_record_types) <= set(record_types)

    assert evidence.recovery.model_dump() == {
        "attempted": False,
        "database_reopened": False,
        "checkpoint_id": None,
        "checkpoint_sequence": None,
        "can_resume": False,
        "source_replayed": False,
        "checkpoint_removed": False,
    }


def _assert_declared_recovery_evidence(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    evidence: ScenarioRunEvidence,
) -> None:
    fixture_path = resolve_fixture_path(case.fixture)
    input_path = resolve_fixture_path(case.input_fixture)
    expected_fixture_hash = hashlib.sha256(fixture_path.read_bytes() + b"\0" + input_path.read_bytes()).hexdigest()

    assert evidence.schema_version == 1
    assert (evidence.scenario_id, evidence.case_id) == (scenario.id, case.id)
    assert evidence.fixture_sha256 == expected_fixture_hash
    assert evidence.completed_stages == ("config", "build", "runtime", "audit", "recovery")

    assert evidence.config.loaded is True
    assert len(evidence.config.settings_sha256) == 64
    assert evidence.graph.accepted is True
    assert evidence.graph.node_count is not None and evidence.graph.node_count > 0
    assert evidence.graph.edge_count is not None and evidence.graph.edge_count > 0
    assert evidence.graph.topology_hash is not None and len(evidence.graph.topology_hash) == 64

    assert evidence.runtime.attempted is True
    assert evidence.runtime.status == case.expected.status
    assert evidence.runtime.output_rows == case.expected.output_rows

    assert evidence.audit.attempted is True
    assert evidence.audit.total_records > 0
    assert evidence.audit.total_records == sum(record.count for record in evidence.audit.record_counts)
    record_types = tuple(record.record_type for record in evidence.audit.record_counts)
    assert record_types == tuple(sorted(record_types))
    assert set(case.expected.required_audit_record_types) <= set(record_types)

    assert evidence.recovery.attempted is True
    assert evidence.recovery.database_reopened is True
    assert evidence.recovery.checkpoint_id is not None
    assert evidence.recovery.checkpoint_sequence is not None
    assert evidence.recovery.can_resume is True
    assert evidence.recovery.source_replayed is False
    assert evidence.recovery.checkpoint_removed is True


def test_run_case_owns_production_preflight_without_pytest_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, case = _declared_case("linear", "happy-path")
    production_run = inspect.unwrap(Orchestrator.run)

    def run_without_autouse_defaults(self: Orchestrator, *args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("openrouter_catalog_sha256")
        assert kwargs.get("openrouter_catalog_source") in {"bundled", "live"}
        return production_run(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "run", run_without_autouse_defaults)
    install_corpus_plugin_manager(monkeypatch)

    evidence = run_scenario_case(scenario, case, tmp_path)

    assert evidence.runtime.status == "completed"


def test_render_settings_quotes_yaml_significant_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _scenario, case = _declared_case("linear", "happy-path")
    original_fixture = resolve_fixture_path(case.fixture)
    original_input = resolve_fixture_path(case.input_fixture)
    fixture_root = tmp_path / "fixture root : # corpus"
    copied_fixture = fixture_root / case.fixture
    copied_input = fixture_root / case.input_fixture
    copied_fixture.parent.mkdir(parents=True)
    copied_fixture.write_bytes(original_fixture.read_bytes())
    copied_input.write_bytes(original_input.read_bytes())
    monkeypatch.setattr(corpus_loader, "FIXTURE_ROOT", fixture_root)
    runtime_root = tmp_path / "runtime root : # output"
    runtime_root.mkdir()

    rendered = render_settings(case, runtime_root)
    install_corpus_plugin_manager(monkeypatch)
    built = build_scenario(rendered)

    assert rendered.settings.sources["primary"].options["path"] == str(copied_input)
    assert rendered.settings.sinks["output"].options["path"] == str(runtime_root / "output.jsonl")
    assert rendered.output_path == runtime_root / "output.jsonl"
    assert rendered.fault_marker == runtime_root / "fault-triggered.marker"
    assert built.graph_evidence.accepted is True


def test_generic_run_case_assertions_accept_future_case_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, case = _declared_case("linear", "happy-path")
    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)
    future_scenario = scenario.model_copy(update={"id": "future-run-scenario"})
    future_case = case.model_copy(update={"id": "future-run-case"})
    future_evidence = evidence.model_copy(
        update={
            "scenario_id": future_scenario.id,
            "case_id": future_case.id,
            "graph": evidence.graph.model_copy(update={"node_count": 7, "edge_count": 6}),
        }
    )

    _assert_declared_run_evidence(future_scenario, future_case, future_evidence)


@pytest.mark.parametrize(("scenario", "case"), RUN_CASES)
def test_declared_run_case_uses_complete_production_path(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)

    _assert_declared_run_evidence(scenario, case, evidence)


def test_linear_happy_path_has_exact_production_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, case = _declared_case("linear", "happy-path")
    assert (scenario.id, case.id, case.fixture, case.input_fixture) == (
        "linear",
        "happy-path",
        "linear/happy-path.yaml",
        "linear/input.csv",
    )

    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)
    _assert_declared_run_evidence(scenario, case, evidence)

    # The declared queue is a first-class runtime node between source and transform.
    assert evidence.graph.node_count == 4
    assert evidence.graph.edge_count == 3

    assert evidence.runtime.rows_processed == 3
    assert evidence.runtime.rows_succeeded == 3
    assert evidence.runtime.rows_failed == 0
    output_rows = [json.loads(line) for line in (tmp_path / "output.jsonl").read_text(encoding="utf-8").splitlines()]
    assert output_rows == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]

    audit_counts = {record.record_type: record.count for record in evidence.audit.record_counts}
    assert {"run", "node", "edge", "operation", "row"} <= set(audit_counts)
    assert audit_counts["run"] == 1
    assert audit_counts["node"] == 4
    assert audit_counts["edge"] == 3
    assert audit_counts["row"] == 3
    assert evidence.audit.source_operation_count == 1


@pytest.mark.parametrize(("scenario", "case"), RECOVERY_CASES)
def test_declared_recovery_case_reopens_and_resumes_publicly(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)

    _assert_declared_recovery_evidence(scenario, case, evidence)


def test_checkpoint_reopen_resume_has_exact_restart_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario, case = _declared_case("checkpoint-deterministic-resume", "reopen-resume")
    assert [(entry.values[0].id, entry.values[1].id) for entry in RECOVERY_CASES] == [("checkpoint-deterministic-resume", "reopen-resume")]

    production_run = inspect.unwrap(Orchestrator.run)
    production_resume = inspect.unwrap(Orchestrator.resume)
    monkeypatch.setattr(Orchestrator, "run", production_run)
    monkeypatch.setattr(Orchestrator, "resume", production_resume)
    install_corpus_plugin_manager(monkeypatch)

    built_objects: list[Any] = []
    production_build = corpus_harness.build_scenario

    def record_fresh_build(*args: Any, **kwargs: Any) -> Any:
        built = production_build(*args, **kwargs)
        built_objects.append(built)
        return built

    monkeypatch.setattr(corpus_harness, "build_scenario", record_fresh_build)

    evidence = corpus_harness.run_scenario_case(scenario, case, tmp_path)
    _assert_declared_recovery_evidence(scenario, case, evidence)

    assert len(built_objects) == 2
    initial, fresh = built_objects
    assert initial is not fresh
    assert initial.rendered.settings is not fresh.rendered.settings
    assert initial.bundle is not fresh.bundle
    assert initial.graph is not fresh.graph
    assert initial.config is not fresh.config
    assert evidence.runtime.rows_processed == 3
    # Three source rows are consumed into one terminal aggregation output.
    assert evidence.runtime.rows_succeeded == 1
    assert evidence.runtime.rows_failed == 0
    assert evidence.audit.source_operation_count == 1
    assert [json.loads(line) for line in (tmp_path / "output.jsonl").read_text(encoding="utf-8").splitlines()] == [
        {"value": 60, "count": 3}
    ]
    assert (tmp_path / "fault-triggered.marker").is_file()
