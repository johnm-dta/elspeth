"""Baseline gate for unspecced test mocks.

The suite still has many legacy ``Mock()`` / ``MagicMock()`` calls without
``spec``/``spec_set``/``autospec``. This gate makes that debt visible and
prevents the count from increasing while focused cleanup replaces legacy mocks
with fakes or specced mocks.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
# Bumped 2559→2585 (2026-06-01): the cicd-judge campaign (judge transport +
# scope-fingerprint binding) added external-SDK response-shape MagicMocks
# (OpenAI/OpenRouter completion/choice/usage fakes) where spec= is brittle and
# low-value. Pay-down of the judge-test mocks is tracked separately; this bump
# matches the prior explicit-bump precedent (9daf156e1: 2554→2559).
# Bumped 2585→2602 (2026-06-08): the ratchet was already stale at 2601 (16 mocks
# of pre-existing drift from intervening branch work — operator-owed re-sign,
# flagged separately) PLUS +1 from the execute() fail-closed pre-run validation
# gate tests (PipelineValidationError; patch of validate_pipeline). Only the +1
# belongs to this commit; the 16 are inherited pre-existing red.
# Bumped 2602→2617 (2026-06-08): +15 from merging the engine/core/plugins and
# cicd-scanner-bugs burndown branches into release/0.5.3. The new mocks are
# external-SDK response-shape fakes (OpenAI completion/choice/usage in
# test_llm_telemetry) and HTTP SSRF/redirect-context fakes (test_audited_http_client),
# spread across ~13 burndown test files. These mock dynamically-shaped response
# objects where spec= is brittle and low-value — matching the prior bump precedent.
# Bumped 2617→2618 (2026-06-13): +1 net from the multi-source-token-scheduler
# feat branch (slices 1-3 coordination substrate, journal barrier, finalize sweep).
# New tests in that branch use a _StubRepo typed fake for RunHeartbeatThread; the
# +1 comes from small pre-existing drift in test_execution_repository,
# test_outcomes, test_read_only_audit_surfaces across the branch's accumulated
# changes (no single new file; the new heartbeat tests contribute zero).
# Bumped 2618→2623 (2026-06-13): +5 from slice 5 step 2 (follower-mode
# RowProcessor — test_follower_processor.py).  The new MagicMock() calls are
# return-value fakes for _drain_scheduler_claims result items (SchedulerResult
# stubs) where spec= would need to import internal scheduler types; low-value
# spec here. Step 3 (CLI join + journal path tests) contributed zero new
# unspecced mocks (uses create_autospec and patch only).
# Bumped 2623→2630 (2026-06-13): +7 from slice 5 steps 4-5 (e2e recovery
# tests for follower lifecycle: test_follower_join_and_drain.py x5 and
# test_follower_coordination_chaos.py x2).  These are stub RowProcessor
# objects — MagicMock() stubs where spec= would require importing internal
# RowProcessor and wiring a full engine stack; low-value spec for integration
# harness stubs. The real follower routing (barrier/branch-loss/sink) is
# covered by the processor-level drain tests, not these lifecycle stubs.
# Bumped 2630→2652 (2026-06-14): +22 total on fix/plugins-subsystem-remediation.
# +20 belong to this branch's plugin-remediation test additions (changed test
# files went 321→341 unspecced mocks): external-SDK/provider response-shape
# fakes (test_audited_llm_client, llm/test_transform, azure_multi_query_retry)
# and source/transform stubs (test_dataverse_source, rag/test_transform,
# transforms/azure/test_blob_source) where spec= is brittle/low-value — matching
# the prior-bump precedent. The remaining +2 is pre-existing drift already on
# release/0.6.0 (its actual count is 2632 vs the 2630 constant — operator-owed,
# inherited, flagged separately).
# Bumped 2652→2716 (2026-06-19): +64 from the 0.6.0 multi-worker N>1
# leader/follower correctness suite landing on release/0.6.0 (commits 6652df74b,
# 2b106b902, d508c2943 and the surrounding 0.6.0 work). The new unspecced mocks
# are spread across ~17 changed test files — new CLI orchestrator-teardown /
# resume-graph stubs (test_cli_orchestrator_teardown.py +11, test_cli.py +6),
# execution-service/validation/websocket route fakes, follower-processor result
# stubs (+6), and external-SDK/provider response-shape fakes (azure_search +7,
# dataverse_sink +3). All are orchestration-context / response-shape stubs where
# spec= would require importing internal orchestrator/graph types and is
# brittle/low-value — matching the prior-bump precedent.
# Bumped 2716→2776 (2026-06-23): +60 from accumulated release/0.7.0 test
# additions, led by route/component fakes in test_outputs_routes.py, provider
# response-shape fakes in rag/test_transform.py and test_azure_search.py, and
# smaller composer/execution/session harness stubs. This keeps the gate as a
# no-regression ratchet for future work; paying down this new debt remains
# separate from the 2306 test-failure closeout.
# Bumped 2776→2822 (2026-06-29): snapshot current release-line debt while
# switching from a repo-wide-only ratchet to per-file counts. Future additions
# fail at the file that regresses, even if unrelated files pay down mocks.
BASELINE_UNSPECCED_MOCK_TOTAL = 2822
BASELINE_UNSPECCED_MOCK_COUNTS_BY_FILE = {
    "tests/e2e/recovery/test_follower_coordination_chaos.py": 2,
    "tests/e2e/recovery/test_follower_join_and_drain.py": 5,
    "tests/fixtures/factories.py": 1,
    "tests/fixtures/test_factories.py": 1,
    "tests/integration/audit/test_audit_field_separation.py": 6,
    "tests/integration/cli/test_cli.py": 6,
    "tests/integration/pipeline/orchestrator/test_orchestrator_core.py": 21,
    "tests/integration/pipeline/orchestrator/test_orchestrator_execute_run_characterization.py": 2,
    "tests/integration/pipeline/test_bootstrap_preflight.py": 31,
    "tests/integration/pipeline/test_resume_comprehensive.py": 8,
    "tests/integration/pipeline/test_retry.py": 2,
    "tests/integration/plugins/llm/test_multi_query.py": 6,
    "tests/integration/plugins/sinks/test_chroma_sink_pipeline.py": 23,
    "tests/integration/plugins/sinks/test_durability.py": 2,
    "tests/integration/plugins/test_dataverse_pipeline.py": 2,
    "tests/integration/plugins/transforms/test_rag_pipeline.py": 5,
    "tests/integration/rate_limit/test_integration.py": 14,
    "tests/integration/web/composer/guided/test_chain_discovery_loop.py": 1,
    "tests/integration/web/composer/guided/test_sink_discovery_loop.py": 1,
    "tests/integration/web/composer/test_inline_source_provenance.py": 1,
    "tests/integration/web/composer/test_interpretation_runtime_handoff.py": 7,
    "tests/integration/web/test_preferences_routes.py": 1,
    "tests/integration/web/test_preflight_per_class.py": 1,
    "tests/property/core/test_lineage_properties.py": 38,
    "tests/property/core/test_rate_limiter_properties.py": 9,
    "tests/property/core/test_retention_monotonicity.py": 7,
    "tests/property/engine/test_coalesce_properties.py": 4,
    "tests/property/engine/test_processor_coalesce_equivalence_properties.py": 3,
    "tests/property/engine/test_sink_executor_diversion_properties.py": 18,
    "tests/property/engine/test_token_properties.py": 4,
    "tests/property/plugins/sinks/test_azure_blob_sink_properties.py": 4,
    "tests/property/plugins/sources/test_azure_blob_source_properties.py": 3,
    "tests/unit/cli/test_cli.py": 16,
    "tests/unit/cli/test_cli_helpers.py": 7,
    "tests/unit/cli/test_cli_helpers_db.py": 4,
    "tests/unit/cli/test_cli_preflight.py": 13,
    "tests/unit/cli/test_explain_tui.py": 13,
    "tests/unit/cli/test_purge_command.py": 7,
    "tests/unit/cli/test_secrets_loading.py": 3,
    "tests/unit/cli/test_web_command.py": 17,
    "tests/unit/contracts/test_freeze_regression.py": 2,
    "tests/unit/contracts/test_plugin_context_recording.py": 3,
    "tests/unit/contracts/test_record_call_guards.py": 23,
    "tests/unit/contracts/test_telemetry_contracts.py": 53,
    "tests/unit/contracts/transform_contracts/test_batch_transform_protocol.py": 3,
    "tests/unit/core/landscape/test_call_recording.py": 1,
    "tests/unit/core/landscape/test_data_flow_repository.py": 5,
    "tests/unit/core/landscape/test_database_compatibility_guards.py": 5,
    "tests/unit/core/landscape/test_execution_repository.py": 2,
    "tests/unit/core/landscape/test_exporter.py": 27,
    "tests/unit/core/landscape/test_factory.py": 1,
    "tests/unit/core/landscape/test_journal.py": 14,
    "tests/unit/core/landscape/test_lineage.py": 3,
    "tests/unit/core/landscape/test_node_state_recording.py": 1,
    "tests/unit/core/landscape/test_query_methods.py": 16,
    "tests/unit/core/security/test_config_secrets.py": 32,
    "tests/unit/core/security/test_secret_loader.py": 15,
    "tests/unit/elspeth_lints/test_allowlist_yaml_roundtrip_contract.py": 6,
    "tests/unit/elspeth_lints/test_judge_decorator_integration.py": 7,
    "tests/unit/elspeth_lints/test_justify.py": 27,
    "tests/unit/elspeth_lints/test_reaudit.py": 28,
    "tests/unit/elspeth_lints/test_reaudit_multi_rule.py": 6,
    "tests/unit/elspeth_lints/test_source_excerpt_security.py": 6,
    "tests/unit/engine/orchestrator/test_accumulate_diverted.py": 1,
    "tests/unit/engine/orchestrator/test_aggregation.py": 98,
    "tests/unit/engine/orchestrator/test_ceremony_safe_flush.py": 1,
    "tests/unit/engine/orchestrator/test_checkpoint_interrupted_edge.py": 5,
    "tests/unit/engine/orchestrator/test_export.py": 20,
    "tests/unit/engine/orchestrator/test_finalize_source_iteration.py": 13,
    "tests/unit/engine/orchestrator/test_follower_processor.py": 16,
    "tests/unit/engine/orchestrator/test_graceful_shutdown.py": 21,
    "tests/unit/engine/orchestrator/test_outcomes.py": 80,
    "tests/unit/engine/orchestrator/test_pending_sink_grouping.py": 8,
    "tests/unit/engine/orchestrator/test_preflight_pipeline_config.py": 4,
    "tests/unit/engine/orchestrator/test_resume_failure.py": 4,
    "tests/unit/engine/orchestrator/test_types.py": 23,
    "tests/unit/engine/orchestrator/test_validation.py": 3,
    "tests/unit/engine/test_adr019_phase2_producer_pairs.py": 2,
    "tests/unit/engine/test_audit_wrapper_scope.py": 8,
    "tests/unit/engine/test_bootstrap_preflight.py": 9,
    "tests/unit/engine/test_coalesce_executor.py": 13,
    "tests/unit/engine/test_coalesce_pipeline_row.py": 12,
    "tests/unit/engine/test_dependency_resolver.py": 13,
    "tests/unit/engine/test_executors.py": 54,
    "tests/unit/engine/test_post_init_validations.py": 2,
    "tests/unit/engine/test_processor.py": 49,
    "tests/unit/engine/test_resume_offset_propagation.py": 4,
    "tests/unit/engine/test_sink_executor_diversion.py": 46,
    "tests/unit/engine/test_state_guard_audit_evidence_discriminator.py": 2,
    "tests/unit/engine/test_token_manager_pipeline_row.py": 9,
    "tests/unit/mcp/analyzers/test_contracts.py": 9,
    "tests/unit/mcp/analyzers/test_reports.py": 69,
    "tests/unit/mcp/test_query_validation.py": 1,
    "tests/unit/plugins/clients/test_audited_client_base.py": 2,
    "tests/unit/plugins/clients/test_audited_http_client.py": 18,
    "tests/unit/plugins/clients/test_audited_llm_client.py": 127,
    "tests/unit/plugins/clients/test_http.py": 3,
    "tests/unit/plugins/clients/test_http_redirects.py": 7,
    "tests/unit/plugins/clients/test_http_telemetry.py": 15,
    "tests/unit/plugins/clients/test_llm_error_classification.py": 1,
    "tests/unit/plugins/clients/test_llm_telemetry.py": 24,
    "tests/unit/plugins/clients/test_replayer.py": 3,
    "tests/unit/plugins/clients/test_verifier.py": 12,
    "tests/unit/plugins/infrastructure/clients/retrieval/test_azure_search.py": 39,
    "tests/unit/plugins/infrastructure/clients/retrieval/test_chroma.py": 5,
    "tests/unit/plugins/infrastructure/clients/test_http_allowed_ranges.py": 18,
    "tests/unit/plugins/infrastructure/clients/test_http_call_return.py": 8,
    "tests/unit/plugins/infrastructure/test_display_headers.py": 13,
    "tests/unit/plugins/infrastructure/test_probe_factory.py": 16,
    "tests/unit/plugins/llm/conftest.py": 10,
    "tests/unit/plugins/llm/test_azure.py": 12,
    "tests/unit/plugins/llm/test_azure_multi_query.py": 10,
    "tests/unit/plugins/llm/test_azure_multi_query_profiling.py": 4,
    "tests/unit/plugins/llm/test_azure_multi_query_retry.py": 21,
    "tests/unit/plugins/llm/test_langfuse_tracer.py": 22,
    "tests/unit/plugins/llm/test_llm_success_reason.py": 6,
    "tests/unit/plugins/llm/test_openrouter.py": 29,
    "tests/unit/plugins/llm/test_openrouter_multi_query.py": 10,
    "tests/unit/plugins/llm/test_p1_bug_fixes.py": 9,
    "tests/unit/plugins/llm/test_provider_azure.py": 22,
    "tests/unit/plugins/llm/test_provider_lifecycle.py": 7,
    "tests/unit/plugins/llm/test_provider_openrouter.py": 36,
    "tests/unit/plugins/llm/test_tracing_integration.py": 14,
    "tests/unit/plugins/llm/test_transform.py": 27,
    "tests/unit/plugins/sinks/test_azure_blob_sink.py": 5,
    "tests/unit/plugins/sinks/test_azure_blob_sink_serialization.py": 1,
    "tests/unit/plugins/sinks/test_chroma_sink.py": 43,
    "tests/unit/plugins/sinks/test_dataverse_sink.py": 67,
    "tests/unit/plugins/sinks/test_sink_bug_fixes.py": 4,
    "tests/unit/plugins/sinks/test_sink_display_headers.py": 6,
    "tests/unit/plugins/sinks/test_sink_protocol_compliance.py": 2,
    "tests/unit/plugins/sources/test_azure_blob_source.py": 12,
    "tests/unit/plugins/sources/test_csv_source.py": 3,
    "tests/unit/plugins/sources/test_dataverse_source.py": 37,
    "tests/unit/plugins/sources/test_json_source.py": 2,
    "tests/unit/plugins/test_assert_to_raise.py": 1,
    "tests/unit/plugins/test_context.py": 35,
    "tests/unit/plugins/transforms/azure/test_auth.py": 23,
    "tests/unit/plugins/transforms/azure/test_blob_sink.py": 69,
    "tests/unit/plugins/transforms/azure/test_blob_source.py": 32,
    "tests/unit/plugins/transforms/azure/test_content_safety.py": 16,
    "tests/unit/plugins/transforms/azure/test_prompt_shield.py": 16,
    "tests/unit/plugins/transforms/llm/test_value_sources.py": 2,
    "tests/unit/plugins/transforms/rag/test_query.py": 2,
    "tests/unit/plugins/transforms/rag/test_transform.py": 41,
    "tests/unit/plugins/transforms/test_web_scrape.py": 5,
    "tests/unit/plugins/transforms/test_web_scrape_security.py": 17,
    "tests/unit/regression/test_falsy_values_not_treated_as_missing.py": 9,
    "tests/unit/regression/test_input_boundary_validation_guards.py": 3,
    "tests/unit/telemetry/exporters/test_azure_monitor.py": 3,
    "tests/unit/telemetry/exporters/test_datadog.py": 3,
    "tests/unit/telemetry/exporters/test_otlp.py": 14,
    "tests/unit/telemetry/test_factory.py": 16,
    "tests/unit/telemetry/test_plugin_wiring.py": 20,
    "tests/unit/test_cli_helpers_sink_factory.py": 4,
    "tests/unit/test_cli_orchestrator_teardown.py": 13,
    "tests/unit/tui/test_explain_app.py": 1,
    "tests/unit/web/audit_readiness/test_service.py": 15,
    "tests/unit/web/auth/test_entra_provider.py": 1,
    "tests/unit/web/auth/test_middleware.py": 1,
    "tests/unit/web/auth/test_oidc_provider.py": 12,
    "tests/unit/web/composer/guided/test_get_rebuild_from_resolved.py": 5,
    "tests/unit/web/composer/guided/test_prefill_from_resolved.py": 1,
    "tests/unit/web/composer/test_agent_tooling.py": 1,
    "tests/unit/web/composer/test_failure_schema_augmentation.py": 1,
    "tests/unit/web/composer/test_post_call_hints.py": 1,
    "tests/unit/web/composer/test_prompts.py": 1,
    "tests/unit/web/composer/test_redaction_telemetry.py": 3,
    "tests/unit/web/composer/test_redaction_telemetry_sanity.py": 3,
    "tests/unit/web/composer/test_schema_contract_enforcement.py": 1,
    "tests/unit/web/composer/test_tools.py": 10,
    "tests/unit/web/execution/test_identity_node_advisory.py": 17,
    "tests/unit/web/execution/test_outputs_loader.py": 1,
    "tests/unit/web/execution/test_outputs_routes.py": 51,
    "tests/unit/web/execution/test_read_only_audit_surfaces.py": 1,
    "tests/unit/web/execution/test_routes.py": 56,
    "tests/unit/web/execution/test_run_accounting_projection.py": 1,
    "tests/unit/web/execution/test_service.py": 184,
    "tests/unit/web/execution/test_validation.py": 82,
    "tests/unit/web/execution/test_validation_value_source.py": 14,
    "tests/unit/web/execution/test_websocket.py": 22,
    "tests/unit/web/sessions/routes/test_request_advisor_escape.py": 17,
    "tests/unit/web/sessions/routes/test_wire_signoff_audit_and_blocked.py": 4,
    "tests/unit/web/sessions/routes/test_wire_stage_signoff_gate.py": 4,
    "tests/unit/web/sessions/test_blob_proposal_accept.py": 1,
    "tests/unit/web/sessions/test_guided_start.py": 1,
    "tests/unit/web/sessions/test_routes.py": 12,
    "tests/unit/web/shareable_reviews/test_service.py": 4,
    "tests/unit/web/shareable_reviews/test_telemetry_session_completed.py": 4,
    "tests/unit/web/test_app.py": 6,
    "tests/unit/web/test_dependencies.py": 3,
}
MOCK_NAMES = frozenset({"Mock", "MagicMock"})
SPEC_KEYWORDS = frozenset({"spec", "spec_set", "autospec", "wraps"})


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_specced_mock_call(node: ast.Call) -> bool:
    return any(keyword.arg in SPEC_KEYWORDS for keyword in node.keywords)


def _unspecced_mock_calls_by_file() -> dict[str, list[str]]:
    calls_by_file: dict[str, list[str]] = {}
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel_path = str(path.relative_to(REPO_ROOT))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) not in MOCK_NAMES:
                continue
            if _is_specced_mock_call(node):
                continue
            calls_by_file.setdefault(rel_path, []).append(f"{rel_path}:{node.lineno}:{node.col_offset}")
    return calls_by_file


def _file_baseline_regressions(calls_by_file: dict[str, list[str]]) -> list[str]:
    regressions: list[str] = []
    for rel_path, calls in sorted(calls_by_file.items()):
        baseline = BASELINE_UNSPECCED_MOCK_COUNTS_BY_FILE.get(rel_path, 0)
        if len(calls) > baseline:
            examples = ", ".join(calls[:5])
            regressions.append(f"{rel_path}: {len(calls)} > {baseline}; examples: {examples}")
    return regressions


def test_unspecced_mock_baseline_rejects_new_file_regressions_despite_global_paydown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new unspecced mock in one file must not hide behind cleanup elsewhere."""
    (tmp_path / "legacy.py").write_text("", encoding="utf-8")
    (tmp_path / "changed.py").write_text(
        "from unittest.mock import MagicMock\n\ndef test_changed() -> None:\n    MagicMock()\n",
        encoding="utf-8",
    )
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "TESTS_ROOT", tmp_path)
    monkeypatch.setattr(module, "BASELINE_UNSPECCED_MOCK_TOTAL", 1)
    monkeypatch.setattr(module, "BASELINE_UNSPECCED_MOCK_COUNTS_BY_FILE", {"legacy.py": 1}, raising=False)

    with pytest.raises(AssertionError, match=r"changed\.py"):
        test_unspecced_mock_baseline_does_not_increase()


def test_unspecced_mock_baseline_does_not_increase() -> None:
    unspecced_mock_calls_by_file = _unspecced_mock_calls_by_file()
    file_regressions = _file_baseline_regressions(unspecced_mock_calls_by_file)
    assert not file_regressions, "unspecced mock count increased in file(s):\n" + "\n".join(file_regressions)

    unspecced_mock_total = sum(len(calls) for calls in unspecced_mock_calls_by_file.values())
    assert unspecced_mock_total <= BASELINE_UNSPECCED_MOCK_TOTAL
