"""Regression guard for the cross-check dispatcher overhead benchmark shape."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path


def _benchmark_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "tests/performance/benchmarks/test_cross_check_overhead.py"


def _benchmark_module() -> ast.Module:
    benchmark_path = _benchmark_path()
    return ast.parse(benchmark_path.read_text(encoding="utf-8"), filename=str(benchmark_path))


def _find_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in cross-check overhead benchmark")


def test_dispatcher_scaling_benchmark_varies_registered_and_applicable_counts() -> None:
    """The NFR gate must measure apply-cost, not only skip-cost.

    The review finding was that the benchmark varied only registered
    contracts while keeping applicable contracts fixed at one. Pin the
    benchmark API so future edits cannot silently collapse M back to the
    skip-only path.
    """
    module = _benchmark_module()
    benchmark = _find_function(module, "test_dispatcher_overhead_scales_with_registered_and_applicable_contracts")
    argument_names = {arg.arg for arg in benchmark.args.args}

    assert {"n_registered", "m_applicable"} <= argument_names

    parametrized_names: set[str] = set()
    for decorator in benchmark.decorator_list:
        if not (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "parametrize"
            and decorator.args
        ):
            continue
        first_arg = decorator.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            parametrized_names.update(name.strip() for name in first_arg.value.split(","))

    assert {"n_registered", "m_applicable"} <= parametrized_names


def test_dispatcher_scaling_tail_bound_ignores_single_scheduler_outlier() -> None:
    """The scaling benchmark should fail on sustained overhead, not one host spike."""
    namespace = runpy.run_path(str(_benchmark_path()))
    stable_tail_bound_sec = namespace["_stable_tail_bound_sec"]

    noisy_stats = {
        "q3": 12.0e-6,
        "iqr": 1.0e-6,
        "mean": 15.5e-6,
        "stddev": 119.6e-6,
    }

    assert noisy_stats["mean"] + 3 * noisy_stats["stddev"] > 54.0e-6
    assert stable_tail_bound_sec(noisy_stats) < 54.0e-6
