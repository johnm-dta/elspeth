"""Unit tests for scripts/eval/_backfill_lib.py — pure-logic helpers used by
the 2026-05-03 eval per-row output backfill (elspeth-77d2641032).
"""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta

import pytest
from scripts.eval._backfill_lib import (
    build_scenario_manifest,
    classify_correlation_confidence,
    enumerate_candidate_files,
    extract_sink_paths_from_final_yaml,
)


def test_extract_sink_paths_returns_absolute_path_per_sink_for_mapping_shape(tmp_path) -> None:
    """The captured final_yaml.json carries `sinks: {name: config}` (mapping)."""
    final_yaml_dict = {
        "yaml": (
            "sinks:\n"
            "  results:\n"
            "    plugin: jsonl\n"
            "    options:\n"
            "      path: outputs/results.jsonl\n"
            "  fraud_only:\n"
            "    plugin: csv\n"
            "    options:\n"
            "      path: outputs/q3_fraud_security_flags.csv\n"
        )
    }
    sinks = extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path))
    assert sorted(sinks) == sorted(
        [
            ("results", str((tmp_path / "outputs/results.jsonl").resolve())),
            ("fraud_only", str((tmp_path / "outputs/q3_fraud_security_flags.csv").resolve())),
        ]
    )


def test_extract_sink_paths_accepts_absolute_path_under_outputs_root(tmp_path) -> None:
    target = tmp_path / "outputs" / "already-absolute" / "results.jsonl"
    final_yaml_dict = {"yaml": (f"sinks:\n  results:\n    plugin: jsonl\n    options:\n      path: {target}\n")}
    sinks = extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path))
    assert sinks == [("results", str(target))]


def test_extract_sink_paths_rejects_absolute_paths_outside_allowed_data_roots(tmp_path) -> None:
    final_yaml_dict = {"yaml": ("sinks:\n  results:\n    plugin: jsonl\n    options:\n      path: /tmp/already-absolute/results.jsonl\n")}
    with pytest.raises(ValueError, match="outside allowed eval data roots"):
        extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path))


def test_extract_sink_paths_rejects_relative_traversal_outside_allowed_data_roots(tmp_path) -> None:
    final_yaml_dict = {"yaml": ("sinks:\n  results:\n    plugin: jsonl\n    options:\n      path: ../host-secret.txt\n")}
    with pytest.raises(ValueError, match="outside allowed eval data roots"):
        extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path))


def test_extract_sink_paths_rejects_relative_paths_outside_outputs_or_blobs(tmp_path) -> None:
    final_yaml_dict = {"yaml": ("sinks:\n  results:\n    plugin: jsonl\n    options:\n      path: logs/results.jsonl\n")}
    with pytest.raises(ValueError, match="outside allowed eval data roots"):
        extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path))


def test_extract_sink_paths_skips_sinks_without_path_options(tmp_path) -> None:
    final_yaml_dict = {"yaml": "sinks:\n  ignore:\n    plugin: noop\n    options: {}\n"}
    assert extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path)) == []


def test_extract_sink_paths_handles_missing_yaml_key(tmp_path) -> None:
    assert extract_sink_paths_from_final_yaml({}, data_dir=str(tmp_path)) == []


def test_extract_sink_paths_handles_yaml_without_sinks_section(tmp_path) -> None:
    final_yaml_dict = {"yaml": "source:\n  plugin: csv\n  options: {}\n"}
    assert extract_sink_paths_from_final_yaml(final_yaml_dict, data_dir=str(tmp_path)) == []


def test_enumerate_candidate_files_includes_base_and_auto_increment_siblings(tmp_path) -> None:
    base = tmp_path / "high_priority.jsonl"
    base.write_text("base\n")
    (tmp_path / "high_priority-1.jsonl").write_text("one\n")
    (tmp_path / "high_priority-2.jsonl").write_text("two\n")
    (tmp_path / "unrelated.jsonl").write_text("nope\n")

    candidates = enumerate_candidate_files(str(base))
    assert sorted(c.name for c in candidates) == [
        "high_priority-1.jsonl",
        "high_priority-2.jsonl",
        "high_priority.jsonl",
    ]


def test_enumerate_candidate_files_returns_empty_when_directory_missing(tmp_path) -> None:
    missing = tmp_path / "nope" / "results.jsonl"
    assert enumerate_candidate_files(str(missing)) == []


def test_enumerate_candidate_files_skips_non_matching_siblings(tmp_path) -> None:
    base = tmp_path / "results.jsonl"
    base.write_text("ok\n")
    (tmp_path / "results-1.jsonl").write_text("ok-1\n")
    (tmp_path / "resultsX.jsonl").write_text("not a sibling\n")
    (tmp_path / "results-1.csv").write_text("wrong suffix\n")

    candidates = enumerate_candidate_files(str(base))
    assert sorted(c.name for c in candidates) == ["results-1.jsonl", "results.jsonl"]


def test_enumerate_candidate_files_returns_empty_when_no_candidates_match(tmp_path) -> None:
    (tmp_path / "other.jsonl").write_text("nope\n")
    target = tmp_path / "results.jsonl"
    assert enumerate_candidate_files(str(target)) == []


def test_classify_correlation_confidence_high_when_mtime_inside_run_window() -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    file_mtime = datetime(2026, 5, 3, 13, 28, 14, tzinfo=UTC)
    assert classify_correlation_confidence(file_mtime, started, finished) == "high"


def test_classify_correlation_confidence_high_within_grace_after_finish() -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    file_mtime = finished + timedelta(seconds=45)
    assert classify_correlation_confidence(file_mtime, started, finished) == "high"


def test_classify_correlation_confidence_low_when_mtime_long_after_finish() -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    file_mtime = datetime(2026, 5, 4, 7, 55, 0, tzinfo=UTC)
    assert classify_correlation_confidence(file_mtime, started, finished) == "low"


def test_classify_correlation_confidence_low_when_mtime_before_start() -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    file_mtime = started - timedelta(seconds=10)
    assert classify_correlation_confidence(file_mtime, started, finished) == "low"


def test_classify_correlation_confidence_high_at_exact_start_boundary() -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    assert classify_correlation_confidence(started, started, finished) == "high"


def test_build_scenario_manifest_classifies_files_and_records_hashes(tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    inside = outputs_dir / "results.jsonl"
    inside.write_text('{"interaction_id":"INT-1001"}\n')
    outside = outputs_dir / "results-1.jsonl"
    outside.write_text('{"interaction_id":"OLD"}\n')

    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)

    inside_ts = (started + timedelta(seconds=14)).timestamp()
    outside_ts = (started - timedelta(days=5)).timestamp()
    os.utime(inside, (inside_ts, inside_ts))
    os.utime(outside, (outside_ts, outside_ts))

    manifest = build_scenario_manifest(
        scenario_id="p1_t1_happy",
        run_id="e9912276-8be5-4ccc-b74f-dd5f3c401946",
        run_started_at=started,
        run_finished_at=finished,
        sinks=[("results", str(inside))],
    )

    assert manifest["scenario_id"] == "p1_t1_happy"
    assert manifest["run_id"] == "e9912276-8be5-4ccc-b74f-dd5f3c401946"
    assert manifest["run_window"]["started_at"] == started.isoformat()
    assert manifest["run_window"]["finished_at"] == finished.isoformat()
    assert len(manifest["files"]) == 1
    file_record = manifest["files"][0]
    assert file_record["sink_name"] == "results"
    assert file_record["correlation_confidence"] == "high"
    assert file_record["sha256"] == hashlib.sha256(inside.read_bytes()).hexdigest()
    assert file_record["actual_path"] == str(inside)
    assert file_record["archived_as"] == "outputs/results.jsonl"
    assert file_record["size"] == inside.stat().st_size
    assert len(manifest["skipped_low_confidence"]) == 1
    assert manifest["skipped_low_confidence"][0]["actual_path"] == str(outside)


def test_build_scenario_manifest_handles_missing_files_gracefully(tmp_path) -> None:
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)

    manifest = build_scenario_manifest(
        scenario_id="empty",
        run_id="00000000-0000-0000-0000-000000000000",
        run_started_at=started,
        run_finished_at=finished,
        sinks=[("nope", str(tmp_path / "nonexistent.jsonl"))],
    )

    assert manifest["files"] == []
    assert manifest["skipped_low_confidence"] == []


def test_build_scenario_manifest_records_captured_by_metadata(tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    inside = outputs_dir / "results.jsonl"
    inside.write_text("ok\n")
    started = datetime(2026, 5, 3, 13, 28, 0, tzinfo=UTC)
    finished = datetime(2026, 5, 3, 13, 28, 30, tzinfo=UTC)
    os.utime(inside, ((started + timedelta(seconds=10)).timestamp(),) * 2)

    manifest = build_scenario_manifest(
        scenario_id="x",
        run_id="r",
        run_started_at=started,
        run_finished_at=finished,
        sinks=[("results", str(inside))],
        captured_by="custom-tool",
    )
    assert manifest["files"][0]["captured_by"] == "custom-tool"
    assert manifest["files"][0]["captured_at"]  # ISO timestamp populated
