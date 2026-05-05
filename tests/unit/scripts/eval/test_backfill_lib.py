"""Unit tests for scripts/eval/_backfill_lib.py — pure-logic helpers used by
the 2026-05-03 eval per-row output backfill (elspeth-77d2641032).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.eval._backfill_lib import (
    classify_correlation_confidence,
    enumerate_candidate_files,
    extract_sink_paths_from_final_yaml,
)


def test_extract_sink_paths_from_final_yaml_returns_name_and_path_for_each_sink() -> None:
    final_yaml_dict = {
        "yaml": (
            "outputs:\n"
            "  - plugin: jsonl\n"
            "    name: results\n"
            "    options:\n"
            "      path: /home/john/elspeth/data/outputs/results.jsonl\n"
            "  - plugin: csv\n"
            "    name: fraud_only\n"
            "    options:\n"
            "      path: /home/john/elspeth/data/outputs/q3_fraud_security_flags.csv\n"
        )
    }
    sinks = extract_sink_paths_from_final_yaml(final_yaml_dict)
    assert sinks == [
        ("results", "/home/john/elspeth/data/outputs/results.jsonl"),
        ("fraud_only", "/home/john/elspeth/data/outputs/q3_fraud_security_flags.csv"),
    ]


def test_extract_sink_paths_skips_outputs_without_path_options() -> None:
    final_yaml_dict = {"yaml": "outputs:\n  - plugin: noop\n    name: ignore\n    options: {}\n"}
    assert extract_sink_paths_from_final_yaml(final_yaml_dict) == []


def test_extract_sink_paths_handles_missing_yaml_key() -> None:
    assert extract_sink_paths_from_final_yaml({}) == []


def test_extract_sink_paths_handles_yaml_without_outputs_section() -> None:
    final_yaml_dict = {"yaml": "source:\n  plugin: csv\n  options: {}\n"}
    assert extract_sink_paths_from_final_yaml(final_yaml_dict) == []


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
