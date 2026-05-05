"""Unit tests for scripts/eval/_backfill_lib.py — pure-logic helpers used by
the 2026-05-03 eval per-row output backfill (elspeth-77d2641032).
"""

from __future__ import annotations

from scripts.eval._backfill_lib import extract_sink_paths_from_final_yaml


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
