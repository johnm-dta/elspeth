"""Mechanical gate against weak hasattr-based branching in tests."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

_REVIEWED_HASATTR_SURFACE_CHECKS = frozenset(
    line.strip()
    for line in """
tests/integration/config/test_schema_validation_regression.py:158
tests/unit/contracts/test_config.py:83
tests/unit/contracts/test_config.py:91
tests/unit/contracts/test_engine_contracts.py:35
tests/unit/contracts/test_enums.py:21
tests/unit/contracts/test_enums.py:22
tests/unit/contracts/test_errors.py:167
tests/unit/contracts/test_pipeline_row.py:193
tests/unit/contracts/test_pipeline_row.py:194
tests/unit/contracts/test_secrets.py:75
tests/unit/contracts/test_tier_registry_migration.py:79
tests/unit/core/landscape/test_query_methods.py:68
tests/unit/core/landscape/test_query_methods.py:69
tests/unit/core/security/test_config_secrets.py:517
tests/unit/core/security/test_config_secrets.py:552
tests/unit/core/security/test_config_secrets.py:583
tests/unit/core/test_config_alignment.py:1012
tests/unit/core/test_payload_store.py:16
tests/unit/core/test_payload_store.py:17
tests/unit/core/test_payload_store.py:18
tests/unit/core/test_payload_store.py:19
tests/unit/core/test_token_outcomes.py:36
tests/unit/engine/test_plugin_detection.py:68
tests/unit/engine/test_plugin_detection.py:108
tests/unit/mcp/test_mcp_init.py:32
tests/unit/mcp/test_mcp_init.py:33
tests/unit/plugins/llm/test_pool_config.py:208
tests/unit/plugins/llm/test_pool_config.py:209
tests/unit/plugins/llm/test_pooled_executor.py:738
tests/unit/plugins/llm/test_pooled_executor.py:739
tests/unit/plugins/llm/test_pooled_executor.py:740
tests/unit/plugins/llm/test_pooled_executor.py:741
tests/unit/plugins/llm/test_pooled_executor.py:742
tests/unit/plugins/llm/test_pooled_executor.py:743
tests/unit/plugins/llm/test_provider_protocol.py:45
tests/unit/plugins/llm/test_transform.py:646
tests/unit/plugins/sinks/test_csv_sink.py:325
tests/unit/plugins/sources/test_csv_source.py:46
tests/unit/plugins/sources/test_csv_source.py:159
tests/unit/plugins/sources/test_json_source.py:41
tests/unit/plugins/sources/test_json_source.py:219
tests/unit/plugins/sources/test_null_source.py:65
tests/unit/plugins/test_base.py:126
tests/unit/plugins/test_base.py:127
tests/unit/plugins/test_base.py:161
tests/unit/plugins/test_builtin_plugin_metadata.py:21
tests/unit/plugins/test_builtin_plugin_metadata.py:29
tests/unit/plugins/test_builtin_plugin_metadata.py:37
tests/unit/plugins/test_builtin_plugin_metadata.py:45
tests/unit/plugins/test_builtin_plugin_metadata.py:57
tests/unit/plugins/test_builtin_plugin_metadata.py:65
tests/unit/plugins/test_builtin_plugin_metadata.py:73
tests/unit/plugins/test_builtin_plugin_metadata.py:85
tests/unit/plugins/test_builtin_plugin_metadata.py:93
tests/unit/plugins/test_builtin_plugin_metadata.py:101
tests/unit/plugins/test_builtin_plugin_metadata.py:109
tests/unit/plugins/test_builtin_plugin_metadata.py:117
tests/unit/plugins/test_builtin_plugin_metadata.py:125
tests/unit/plugins/test_builtin_plugin_metadata.py:133
tests/unit/plugins/test_discovery.py:59
tests/unit/plugins/test_discovery.py:519
tests/unit/plugins/test_integration.py:174
tests/unit/plugins/test_manager.py:385
tests/unit/plugins/test_node_id_protocol.py:94
tests/unit/plugins/test_node_id_protocol.py:100
tests/unit/plugins/test_node_id_protocol.py:106
tests/unit/plugins/test_protocols.py:21
tests/unit/plugins/test_protocols.py:81
tests/unit/plugins/test_protocols.py:82
tests/unit/plugins/test_protocols.py:396
tests/unit/plugins/test_protocols.py:402
tests/unit/plugins/test_protocols.py:419
tests/unit/plugins/test_protocols.py:425
tests/unit/plugins/test_protocols.py:431
tests/unit/plugins/test_protocols.py:626
tests/unit/plugins/test_results.py:115
tests/unit/plugins/test_results.py:116
tests/unit/plugins/test_results.py:117
tests/unit/plugins/test_results.py:154
tests/unit/plugins/test_results.py:155
tests/unit/plugins/test_results.py:156
tests/unit/plugins/test_results.py:415
tests/unit/plugins/test_results.py:416
tests/unit/plugins/test_results.py:417
tests/unit/plugins/test_results.py:434
tests/unit/plugins/test_validation_integration.py:48
tests/unit/plugins/transforms/azure/test_blob_sink.py:120
tests/unit/plugins/transforms/test_field_mapper.py:304
tests/unit/plugins/transforms/test_passthrough.py:34
tests/unit/plugins/transforms/test_passthrough.py:35
tests/unit/plugins/transforms/test_passthrough.py:110
tests/unit/telemetry/exporters/test_otlp.py:654
tests/unit/telemetry/exporters/test_otlp_integration.py:38
tests/unit/telemetry/test_contracts.py:291
tests/unit/telemetry/test_contracts.py:292
tests/unit/telemetry/test_contracts.py:404
tests/unit/telemetry/test_contracts.py:405
tests/unit/tui/test_lineage_tree.py:193
tests/unit/web/composer/test_no_sampling_inference.py:9
tests/unit/web/composer/test_no_sampling_inference.py:10
tests/unit/web/composer/test_no_sampling_inference.py:11
tests/unit/web/composer/test_no_sampling_inference.py:12
tests/unit/web/composer/test_state.py:93
tests/unit/web/composer/test_state.py:165
tests/unit/web/execution/test_routes.py:228
tests/unit/web/secrets/test_service.py:183
""".strip().splitlines()
)


def _is_direct_assert_surface_check(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> bool:
    parent = parents.get(node)
    if isinstance(parent, ast.Assert):
        return parent.test is node
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        grandparent = parents.get(parent)
        return isinstance(grandparent, ast.Assert) and grandparent.test is parent
    return False


def test_hasattr_in_tests_is_limited_to_reviewed_surface_assertions() -> None:
    """Forbid unreviewed hasattr() calls in tests, including direct asserts."""
    violations: list[str] = []
    seen_allowlist: set[str] = set()
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "hasattr":
                continue
            rel_path = path.relative_to(REPO_ROOT)
            location = f"{rel_path}:{node.lineno}"
            if location in _REVIEWED_HASATTR_SURFACE_CHECKS and _is_direct_assert_surface_check(node, parents):
                seen_allowlist.add(location)
                continue
            violations.append(f"{rel_path}:{node.lineno}:{node.col_offset}")

    assert violations == []
    assert sorted(_REVIEWED_HASATTR_SURFACE_CHECKS - seen_allowlist) == []
