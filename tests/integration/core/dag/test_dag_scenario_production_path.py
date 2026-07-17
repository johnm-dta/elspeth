"""Production-path integration evidence for maintained DAG scenarios."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.fixtures.dag_scenario_corpus.harness import run_scenario_case
from tests.fixtures.dag_scenario_corpus.loader import iter_harness_cases, load_manifest, resolve_fixture_path
from tests.fixtures.dag_scenario_corpus.plugins import install_corpus_plugin_manager
from tests.fixtures.dag_scenario_corpus.schema import HarnessCaseSpec, ScenarioSpec

MANIFEST = load_manifest()
RUN_CASES = [
    pytest.param(scenario, case, id=f"{scenario.id}:{case.id}") for scenario, case in iter_harness_cases(MANIFEST) if case.workflow == "run"
]


@pytest.mark.parametrize(("scenario", "case"), RUN_CASES)
def test_declared_run_case_uses_complete_production_path(
    scenario: ScenarioSpec,
    case: HarnessCaseSpec,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert len(RUN_CASES) == 1
    assert (scenario.id, case.id, case.fixture, case.input_fixture) == (
        "linear",
        "happy-path",
        "linear/happy-path.yaml",
        "linear/input.csv",
    )

    fixture_path = resolve_fixture_path(case.fixture)
    input_path = resolve_fixture_path(case.input_fixture)
    expected_fixture_hash = hashlib.sha256(fixture_path.read_bytes() + b"\0" + input_path.read_bytes()).hexdigest()

    install_corpus_plugin_manager(monkeypatch)
    evidence = run_scenario_case(scenario, case, tmp_path)

    assert evidence.schema_version == 1
    assert (evidence.scenario_id, evidence.case_id) == (scenario.id, case.id)
    assert evidence.fixture_sha256 == expected_fixture_hash
    assert evidence.completed_stages == ("config", "build", "runtime", "audit")

    assert evidence.config.loaded is True
    assert len(evidence.config.settings_sha256) == 64

    assert evidence.graph.accepted is True
    # The declared queue is a first-class runtime node between source and transform.
    assert evidence.graph.node_count == 4
    assert evidence.graph.edge_count == 3
    assert evidence.graph.topology_hash is not None
    assert len(evidence.graph.topology_hash) == 64

    assert evidence.runtime.attempted is True
    assert evidence.runtime.status == case.expected.status
    assert evidence.runtime.rows_processed == 3
    assert evidence.runtime.rows_succeeded == 3
    assert evidence.runtime.rows_failed == 0
    assert evidence.runtime.output_rows == case.expected.output_rows
    output_rows = [json.loads(line) for line in (tmp_path / "output.jsonl").read_text(encoding="utf-8").splitlines()]
    assert output_rows == [
        {"id": 1, "value": 10},
        {"id": 2, "value": 20},
        {"id": 3, "value": 30},
    ]

    assert evidence.audit.attempted is True
    assert evidence.audit.total_records > 0
    audit_counts = {record.record_type: record.count for record in evidence.audit.record_counts}
    assert set(case.expected.required_audit_record_types) <= set(audit_counts)
    assert {"run", "node", "edge", "operation", "row"} <= set(audit_counts)
    assert audit_counts["run"] == 1
    assert audit_counts["node"] == 4
    assert audit_counts["edge"] == 3
    assert audit_counts["row"] == 3
    assert evidence.audit.source_operation_count == 1

    assert evidence.recovery.model_dump() == {
        "attempted": False,
        "database_reopened": False,
        "checkpoint_id": None,
        "checkpoint_sequence": None,
        "can_resume": False,
        "source_replayed": False,
        "checkpoint_removed": False,
    }
