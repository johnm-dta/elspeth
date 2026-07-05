"""Mechanical gate against weak hasattr-based branching in tests."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"

# Reviewed hasattr surface checks, keyed by content, not line number:
# "<repo-relative path>::<unparsed assert expression>". Moving a reviewed
# assert within its file (or shifting it by editing lines above it) does not
# invalidate the review. Adding, duplicating, removing, or rewording one does:
# the gate compares exact per-file multisets, so entry counts must match the
# tree. Repeated lines below are deliberate — one per reviewed occurrence.
# To allowlist a new reviewed assert, add the key the failure message prints.
_REVIEWED_HASATTR_SURFACE_CHECKS = tuple(
    line.strip()
    for line in """
tests/integration/config/test_schema_validation_regression.py::hasattr(ExecutionGraph, 'from_plugin_instances')
tests/unit/contracts/test_config.py::hasattr(core_config, name)
tests/unit/contracts/test_config.py::not hasattr(contract_config, name)
tests/unit/contracts/test_engine_contracts.py::not hasattr(entry, '__dict__')
tests/unit/contracts/test_enums.py::not hasattr(contracts, legacy_name)
tests/unit/contracts/test_enums.py::not hasattr(enums, legacy_name)
tests/unit/contracts/test_pipeline_row.py::hasattr(row, '__slots__')
tests/unit/contracts/test_pipeline_row.py::not hasattr(row, '__dict__')
tests/unit/contracts/test_secrets.py::not hasattr(item, 'value')
tests/unit/contracts/test_tier_registry_migration.py::not hasattr(registry_mod, 'TIER_1_ERRORS')
tests/unit/core/landscape/test_query_methods.py::not hasattr(factory.query._ops, 'execute_insert')
tests/unit/core/landscape/test_query_methods.py::not hasattr(factory.query._ops, 'execute_update')
tests/unit/core/security/test_config_secrets.py::hasattr(resolutions[0], 'fingerprint')
tests/unit/core/security/test_config_secrets.py::hasattr(resolutions[0], 'fingerprint')
tests/unit/core/security/test_config_secrets.py::hasattr(resolutions[0], 'fingerprint')
tests/unit/core/test_config_alignment.py::hasattr(runtime_cls, 'from_settings')
tests/unit/core/test_payload_store.py::hasattr(PayloadStore, 'delete')
tests/unit/core/test_payload_store.py::hasattr(PayloadStore, 'exists')
tests/unit/core/test_payload_store.py::hasattr(PayloadStore, 'retrieve')
tests/unit/core/test_payload_store.py::hasattr(PayloadStore, 'store')
tests/unit/core/test_token_outcomes.py::hasattr(TokenOutcome, '__dataclass_fields__')
tests/unit/engine/test_plugin_detection.py::hasattr(duck, 'process')
tests/unit/engine/test_plugin_detection.py::hasattr(duck, 'process')
tests/unit/mcp/test_mcp_init.py::hasattr(elspeth.mcp, 'create_server')
tests/unit/mcp/test_mcp_init.py::hasattr(elspeth.mcp, 'main')
tests/unit/plugins/llm/test_pool_config.py::not hasattr(throttle_config, 'max_capacity_retry_seconds')
tests/unit/plugins/llm/test_pool_config.py::not hasattr(throttle_config, 'pool_size')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'buffer_wait_ms')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'complete_index')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'complete_timestamp')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'result')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'submit_index')
tests/unit/plugins/llm/test_pooled_executor.py::hasattr(entry, 'submit_timestamp')
tests/unit/plugins/llm/test_provider_protocol.py::not hasattr(result, 'raw_response')
tests/unit/plugins/sinks/test_csv_sink.py::hasattr(sink, 'plugin_version')
tests/unit/plugins/sources/test_csv_source.py::hasattr(CSVSource, 'plugin_version')
tests/unit/plugins/sources/test_csv_source.py::hasattr(source, 'output_schema')
tests/unit/plugins/sources/test_json_source.py::hasattr(JSONSource, 'plugin_version')
tests/unit/plugins/sources/test_json_source.py::hasattr(source, 'output_schema')
tests/unit/plugins/sources/test_null_source.py::hasattr(source, 'plugin_version')
tests/unit/plugins/test_base.py::hasattr(BaseTransform, 'on_complete')
tests/unit/plugins/test_base.py::hasattr(BaseTransform, 'on_start')
tests/unit/plugins/test_base.py::not hasattr(base, 'BaseAggregation')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(BatchReplicate, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(BatchStats, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(CSVSink, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(CSVSource, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(DatabaseSink, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(FieldMapper, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(JSONExplode, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(JSONSink, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(JSONSource, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(KeywordFilter, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(NullSource, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(PassThrough, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(TextSource, 'plugin_version')
tests/unit/plugins/test_builtin_plugin_metadata.py::hasattr(Truncate, 'plugin_version')
tests/unit/plugins/test_discovery.py::hasattr(cls, 'name')
tests/unit/plugins/test_discovery.py::hasattr(hookimpl_obj, 'elspeth_get_source')
tests/unit/plugins/test_integration.py::not hasattr(base, 'BaseAggregation')
tests/unit/plugins/test_manager.py::hasattr(source, 'load')
tests/unit/plugins/test_node_id_protocol.py::not hasattr(base, 'BaseAggregation')
tests/unit/plugins/test_node_id_protocol.py::not hasattr(protocols, 'AggregationProtocol')
tests/unit/plugins/test_node_id_protocol.py::not hasattr(protocols, 'CoalesceProtocol')
tests/unit/plugins/test_protocols.py::hasattr(SinkProtocol, '__protocol_attrs__')
tests/unit/plugins/test_protocols.py::hasattr(SourceProtocol, '__protocol_attrs__')
tests/unit/plugins/test_protocols.py::hasattr(SourceProtocol, 'close')
tests/unit/plugins/test_protocols.py::hasattr(SourceProtocol, 'load')
tests/unit/plugins/test_protocols.py::not hasattr(base, 'BaseAggregation')
tests/unit/plugins/test_protocols.py::not hasattr(protocols, 'AggregationProtocol')
tests/unit/plugins/test_protocols.py::not hasattr(protocols, 'CoalescePolicy')
tests/unit/plugins/test_protocols.py::not hasattr(protocols, 'CoalesceProtocol')
tests/unit/plugins/test_protocols.py::not hasattr(protocols, 'PluginProtocol')
tests/unit/plugins/test_results.py::hasattr(result, 'duration_ms')
tests/unit/plugins/test_results.py::hasattr(result, 'duration_ms')
tests/unit/plugins/test_results.py::hasattr(result, 'input_hash')
tests/unit/plugins/test_results.py::hasattr(result, 'input_hash')
tests/unit/plugins/test_results.py::hasattr(result, 'output_hash')
tests/unit/plugins/test_results.py::hasattr(result, 'output_hash')
tests/unit/plugins/test_results.py::not hasattr(plugins, 'AggregationProtocol')
tests/unit/plugins/test_results.py::not hasattr(plugins, 'BaseAggregation')
tests/unit/plugins/test_results.py::not hasattr(plugins, 'CoalescePolicy')
tests/unit/plugins/test_results.py::not hasattr(plugins, 'CoalesceProtocol')
tests/unit/plugins/test_validation_integration.py::hasattr(source, 'load')
tests/unit/plugins/transforms/azure/test_blob_sink.py::hasattr(sink, 'input_schema')
tests/unit/plugins/transforms/test_passthrough.py::hasattr(transform, 'input_schema')
tests/unit/plugins/transforms/test_passthrough.py::hasattr(transform, 'output_schema')
tests/unit/plugins/transforms/test_passthrough.py::not hasattr(transform, 'validate_input')
tests/unit/telemetry/exporters/test_otlp.py::hasattr(result, 'resource_spans')
tests/unit/telemetry/exporters/test_otlp_integration.py::hasattr(proto, 'resource_spans')
tests/unit/telemetry/test_contracts.py::hasattr(config.backpressure_mode, 'value')
tests/unit/telemetry/test_contracts.py::hasattr(config.granularity, 'value')
tests/unit/telemetry/test_contracts.py::hasattr(event, 'run_id')
tests/unit/telemetry/test_contracts.py::hasattr(event, 'timestamp')
tests/unit/tui/test_lineage_tree.py::not hasattr(parent.children, 'append')
tests/unit/web/composer/test_no_sampling_inference.py::not hasattr(svc, '_COMPOSER_LLM_SEED')
tests/unit/web/composer/test_no_sampling_inference.py::not hasattr(svc, '_COMPOSER_LLM_TEMPERATURE')
tests/unit/web/composer/test_no_sampling_inference.py::not hasattr(svc, '_composer_llm_seed_for_model')
tests/unit/web/composer/test_no_sampling_inference.py::not hasattr(svc, '_litellm_completion_supports_param')
tests/unit/web/composer/test_state.py::not hasattr(state, 'source')
tests/unit/web/composer/test_state.py::not hasattr(state, 'source')
tests/unit/web/execution/test_routes.py::not hasattr(app.state, 'websocket_ticket_store')
tests/unit/web/secrets/test_service.py::not hasattr(result, 'available')
""".strip().splitlines()
)


def _enclosing_direct_assert(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> ast.Assert | None:
    """Return the Assert whose test is exactly this call (or `not` this call)."""
    parent = parents.get(node)
    if isinstance(parent, ast.Assert) and parent.test is node:
        return parent
    if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
        grandparent = parents.get(parent)
        if isinstance(grandparent, ast.Assert) and grandparent.test is parent:
            return grandparent
    return None


def test_hasattr_in_tests_is_limited_to_reviewed_surface_assertions() -> None:
    """Forbid unreviewed hasattr() calls in tests, including direct asserts."""
    allowed = Counter(_REVIEWED_HASATTR_SURFACE_CHECKS)
    seen: Counter[str] = Counter()
    locations: dict[str, list[str]] = {}
    violations: list[str] = []
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
            assert_stmt = _enclosing_direct_assert(node, parents)
            if assert_stmt is None:
                violations.append(f"{rel_path}:{node.lineno}:{node.col_offset} is not a direct assert surface check")
                continue
            key = f"{rel_path}::{ast.unparse(assert_stmt.test)}"
            seen[key] += 1
            locations.setdefault(key, []).append(str(node.lineno))

    unreviewed = seen - allowed
    for key in sorted(unreviewed):
        lines = ", ".join(locations[key])
        violations.append(f"unreviewed x{unreviewed[key]} (line {lines}): {key}")
    assert violations == []

    stale = allowed - seen
    assert sorted(f"stale allowlist entry x{count}: {key}" for key, count in stale.items()) == []
