"""Mechanical gate for audit integration test taxonomy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_AUDIT_ROOT = REPO_ROOT / "tests" / "integration" / "audit"
REPOSITORY_AUDIT_ROOT = REPO_ROOT / "tests" / "unit" / "core" / "landscape" / "repository_integration"

PRODUCTION_PATH_AUDIT_TESTS = frozenset(
    {
        "test_artifact_idempotency_contention.py",
        "test_audit_field_separation.py",
        "test_can_drop_rows_roundtrip.py",
        "test_export.py",
        "test_exporter_batch_queries.py",
        "test_recorder_explain.py",
        "test_recorder_routing_events.py",
        "test_source_boundary_orchestrator.py",
    }
)

REPOSITORY_AUDIT_TESTS = frozenset(
    {
        "test_call_index_process_contention.py",
        "test_contract_audit.py",
        "test_csv_sink_executor_audit.py",
        "test_declaration_contract_landscape_serialization_roundtrip.py",
        "test_declared_output_fields_serialization_roundtrip.py",
        "test_declared_required_fields_serialization_roundtrip.py",
        "test_error_persistence.py",
        "test_lineage_persisted_invariants.py",
        "test_not_null_constraints.py",
        "test_pass_through_violation_persists.py",
        "test_query_payload_corruption.py",
        "test_recorder_artifacts.py",
        "test_recorder_batches.py",
        "test_recorder_calls.py",
        "test_recorder_contracts.py",
        "test_recorder_errors.py",
        "test_recorder_grades.py",
        "test_recorder_node_states.py",
        "test_recorder_nodes.py",
        "test_recorder_queries.py",
        "test_recorder_row_data.py",
        "test_recorder_runs.py",
        "test_recorder_tokens.py",
        "test_routing_reason_atomicity.py",
        "test_schema_config_mode_serialization_roundtrip.py",
        "test_sink_required_fields_serialization_roundtrip.py",
        "test_source_guaranteed_fields_serialization_roundtrip.py",
        "test_sqlcipher_pipeline.py",
        "test_tier1_integrity.py",
    }
)


def _test_filenames(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.glob("test_*.py")}


def test_integration_audit_directory_contains_only_production_path_tests() -> None:
    """Keep repository-level audit persistence tests out of integration/audit."""
    assert _test_filenames(INTEGRATION_AUDIT_ROOT) == PRODUCTION_PATH_AUDIT_TESTS


def test_repository_audit_tests_are_truthfully_classified() -> None:
    """Preserve repository audit round-trip coverage under a non-integration label."""
    assert _test_filenames(REPOSITORY_AUDIT_ROOT) == REPOSITORY_AUDIT_TESTS


def test_audit_test_path_headers_match_file_locations() -> None:
    """Legacy moved tests may keep path headers; those headers must not lie."""
    mismatches: list[str] = []
    for root in (INTEGRATION_AUDIT_ROOT, REPOSITORY_AUDIT_ROOT):
        for path in sorted(root.glob("test_*.py")):
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            if not first_line.startswith("# tests/"):
                continue
            expected = f"# {path.relative_to(REPO_ROOT).as_posix()}"
            if first_line != expected:
                mismatches.append(f"{path.relative_to(REPO_ROOT)}: expected {expected!r}, got {first_line!r}")

    assert mismatches == []
