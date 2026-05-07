"""Tests for NaN/Infinity rejection on all audit-path json.dumps calls.

The Data Manifesto requires NaN/Infinity rejection at all Tier 1 boundaries.
Every json.dumps in data_flow_repository.py must pass allow_nan=False.
"""

import ast
from pathlib import Path

import elspeth.core.landscape.data_flow_repository as mod


class TestNoUnguardedJsonDumps:
    """Audit: every json.dumps in data_flow_repository.py must pass allow_nan=False."""

    def test_all_json_dumps_have_allow_nan_false(self) -> None:
        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "dumps"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
            ):
                kwarg_names = [kw.arg for kw in node.keywords]
                assert "allow_nan" in kwarg_names, f"json.dumps at line {node.lineno} is missing allow_nan=False"
