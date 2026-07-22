# Executed DAG reassessment evidence

All commands ran from the frozen worktree at
`6e8a6bf5f2f8542bf5b95b1669ce3d3df68d93e3` unless an explicit `/tmp` browser
path is shown. Passing counts overlap the broad unit baseline and must not be
summed as unique tests.

## Baseline and repository gates

### Broad unit baseline

```bash
.venv/bin/pytest tests/unit -q -m 'not fingerprint_baseline'
```

Result: **25,022 passed, 33 skipped, 66 warnings in 158.30s**.

### Signed fingerprint baseline

```bash
.venv/bin/pytest tests/unit -q -m fingerprint_baseline
```

Result: **1 failed, 1 skipped, 12 warnings in 75.70s**. The failure reports
`trust_tier.tier_model` changing from 3,350 to 3,351 findings, with drift in
`plugins/sinks/chroma_sink.py`, `contracts/sink_effects.py`, and
`core/landscape/execution/sink_effect_finalization.py`. Owner:
`elspeth-18fe6e759e`.

### Contract-boundary gate

```bash
.venv/bin/python -m scripts.check_contracts
```

Result: exit 1 with four misplaced type definitions, 29 `dict[str, Any]`
violations, and one stale whitelist entry.

## Structural, configuration, authoring, identity, and scale evidence

### Core builder, schema, and plural sources

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q \
  tests/unit/core/dag/test_builder_validation.py \
  tests/unit/core/dag/test_graph_validation.py \
  tests/unit/core/test_dag_schema_propagation.py \
  tests/unit/core/test_multi_source_foundation.py::test_plural_sources_are_canonical_and_stable_named \
  tests/unit/core/test_multi_source_foundation.py::test_legacy_singular_source_yaml_is_rejected \
  tests/unit/core/test_multi_source_foundation.py::test_settings_round_trip_plural_only \
  tests/unit/core/test_multi_source_foundation.py::test_explicit_named_sources_keep_source_name_in_identity_and_audit_config \
  tests/unit/core/test_multi_source_foundation.py::test_plugin_bundle_instantiates_named_sources_via_production_path \
  tests/unit/core/test_multi_source_foundation.py::test_from_plugin_instances_builds_declared_queue_fan_in_via_production_path \
  tests/unit/core/test_multi_source_foundation.py::test_pipeline_config_assembly_preserves_named_sources \
  tests/unit/core/test_multi_source_foundation.py::test_graph_allows_multiple_source_roots_when_reachable \
  tests/unit/core/test_multi_source_foundation.py::test_graph_rejects_fan_in_without_queue
```

Result: **64 passed in 4.96s**.

### YAML importer and generator

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q \
  tests/unit/web/composer/test_yaml_importer.py \
  tests/unit/web/composer/test_yaml_generator.py
```

Result: **66 passed in 8.79s**.

### Composer/runtime agreement

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_both_reject_missing_required_field \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_both_reject_aggregation_nested_required_input_fields_without_upstream_guarantee \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_both_reject_direct_fork_to_sink_required_field_mismatch \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_both_accept_pass_through_downstream_of_coalesce \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_composer_warns_but_runtime_rejects_mixed_coalesce_branch_schemas \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeAgreement::test_both_accept_aggregation_with_input_fields_and_required_fields \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeGateRouteParityAgreement \
  tests/integration/pipeline/test_composer_runtime_agreement.py::TestComposerRuntimeQueueAgreement
```

Result: **12 passed in 5.65s**. The explicitly named mixed-coalesce case records
a deliberate Composer/runtime disagreement; a passing test is not evidence of
parity for that arm.

### Structural negatives

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -n 0 -q \
  tests/unit/core/test_dag.py::test_source_route_to_unconsumed_declared_queue_is_rejected \
  tests/unit/core/test_dag.py::TestDAGValidation::test_validate_raises_on_cycle \
  tests/unit/core/test_dag.py::TestDAGValidation::test_validate_rejects_duplicate_outgoing_edge_labels \
  tests/unit/core/test_dag.py::TestDAGValidation::test_validate_rejects_disconnected_graph \
  tests/unit/core/test_dag.py::TestSourceSinkValidation::test_validate_requires_at_least_one_source \
  tests/unit/core/test_dag.py::TestSourceSinkValidation::test_validate_requires_at_least_one_sink \
  tests/unit/core/test_dag.py::TestSourceSinkValidation::test_validate_catches_gate_sink_edge_without_route_label \
  tests/unit/core/test_dag.py::TestExecutionGraphFromConfig::test_terminal_transform_on_success_unknown_sink_raises \
  tests/unit/core/test_dag.py::TestExecutionGraphFromConfig::test_from_config_validates_route_targets \
  tests/unit/core/test_dag.py::test_from_plugin_instances_cycle_raises_graph_validation_error \
  tests/unit/core/test_dag.py::TestCoalesceNodes::test_duplicate_fork_branches_rejected_in_config_gate \
  tests/unit/core/test_dag.py::TestCoalesceNodes::test_duplicate_branch_names_across_coalesces_rejected \
  tests/unit/core/test_dag.py::TestAggregationOnSuccessValidation::test_terminal_aggregation_unknown_sink_raises \
  tests/unit/core/test_dag.py::TestCoalesceOnSuccessValidation::test_terminal_coalesce_unknown_sink_raises
```

Result: **14 passed in 0.95s**.

### Cardinality and identity

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q \
  tests/unit/engine/test_batch_token_identity.py \
  tests/unit/core/landscape/repository_integration/test_recorder_tokens.py::TestAtomicTokenOperations::test_expand_token_records_parent_expanded_outcome \
  tests/unit/core/landscape/repository_integration/test_recorder_tokens.py::TestAtomicTokenOperations::test_expand_token_stores_expected_count_contract \
  tests/unit/engine/test_processor.py::TestTransformModeOutcomeOrdering::test_cardinality_mismatch_does_not_record_parent_terminal_outcome \
  tests/unit/engine/test_processor.py::TestTransformModeOutcomeOrdering::test_expand_token_failure_does_not_record_parent_terminal_outcome \
  tests/unit/engine/test_processor.py::TestProcessRowMultiRowOutput \
  tests/property/audit/test_fork_join_balance.py::TestForkRecoveryInvariant::test_expand_token_persists_per_child_payload \
  tests/integration/core/test_batch_membership_contention.py
```

Result: **15 passed in 6.31s**.

### Canonical node and topology identity

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q \
  tests/unit/core/test_dag.py::TestDeterministicNodeIDs \
  tests/integration/checkpoint/test_topology_validation.py::TestTopologyHashSensitivity \
  tests/integration/checkpoint/test_topology_validation.py::TestTopologyHashDeterminism
```

Result: **8 passed in 3.67s**. The tests establish determinism, not secret-safe
identity or implementation-version completeness.

### Output-schema enforcement and runtime continuation

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -n 0 -q \
  tests/unit/core/dag/test_output_schema_enforcement.py \
  tests/integration/pipeline/orchestrator/test_orchestrator_core.py::TestOrchestrator::test_nonterminal_coalesce_continues_to_downstream_gate \
  tests/integration/pipeline/orchestrator/test_orchestrator_core.py::TestOrchestrator::test_traversal_context_keeps_nonterminal_coalesce_in_graph_step_order \
  tests/unit/engine/test_processor.py::TestResumeIncompleteToken::test_expanded_child_inside_coalesced_branch_resumes_after_expand_node
```

Result: **10 passed in 0.57s**.

### Full functional scalability directory

The first invocation inherited the default `not performance` marker and
selected no tests. The corrected evidence command was:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' \
  .venv/bin/python -m pytest -q -m performance tests/performance/scalability
```

Result: **12 passed in 97.51s**. This proves selected functional depth, sink
breadth, row volume, and wide-row behavior. The tests do not assert a supported
performance threshold and are excluded from default pytest.

## Runtime, recovery, concurrency, and atomicity evidence

The common prefix for these commands was:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_ADDOPTS='-p no:cacheprovider' \
  TMPDIR=/tmp/dag-reassessment-tests \
  .venv/bin/python -m pytest -q
```

### R1, R2, and TS-07 through TS-10

```bash
<prefix> \
  tests/unit/core/landscape/test_scheduler_pending_sink_claim.py \
  tests/unit/core/landscape/test_scheduler_events.py \
  tests/unit/core/landscape/test_scheduler_repository_coalesce_branch_losses.py \
  tests/unit/core/landscape/test_coordination_fence_constructs.py
```

Result: **127 passed, 4 warnings in 5.05s**.

### Production disposition drains

```bash
<prefix> \
  tests/unit/engine/test_scheduler_drain_characterization.py::test_sink_bound_result_parks_pending_sink_with_fenced_owner_and_tags_result \
  tests/unit/engine/test_scheduler_drain_characterization.py::test_claimed_token_failure_marks_failed_with_fence \
  tests/unit/engine/test_scheduler_drain_characterization.py::test_non_sink_terminal_marks_terminal_and_unregistered_build_is_unfenced \
  tests/unit/engine/test_processor.py::TestDurableSchedulerResumeDrain::test_aggregation_buffering_leaves_scheduler_work_blocked
```

Result: **4 passed in 4.60s**.

### Focused crash/restart and transaction faults

```bash
<prefix> \
  tests/unit/core/landscape/test_scheduler_lease_recovery_races.py \
  tests/unit/core/landscape/test_scheduler_repository_complete_barrier.py::test_complete_barrier_crash_atomicity \
  tests/integration/pipeline/test_aggregation_recovery.py::TestFlushOutputJournalDurability::test_timeout_flush_output_is_journal_durable_before_sink_write \
  tests/integration/pipeline/test_aggregation_recovery.py::TestFailedFlushReconcile::test_failed_flush_crash_between_terminal_write_and_release_resumes \
  tests/integration/pipeline/test_sink_effect_recovery.py::test_fresh_pipeline_executor_reuses_interrupted_open_state_and_publishes_once \
  tests/integration/pipeline/test_sink_effect_recovery.py::test_redrive_after_crash_before_reservation_recovers \
  tests/unit/engine/test_processor.py::TestDurableSchedulerResumeDrain::test_pending_sink_resume_repairs_already_outcomed_row_without_reemitting_sink \
  tests/unit/engine/test_processor.py::TestDurableSchedulerResumeDrain::test_recovers_expired_lease_then_drains_without_source_replay
```

Result: **16 passed in 4.32s**.

### Direct contention and fencing

```bash
<prefix> \
  tests/integration/engine/test_two_process_scheduler_contention.py \
  tests/integration/engine/test_multi_source_chaos.py::test_lease_expiry_mid_transform_peer_reclaim_bumps_attempt_and_fences_stale_owner \
  tests/e2e/recovery/test_suspended_winner_fences.py \
  tests/unit/engine/test_scheduler_drain_characterization.py::test_immediate_enqueue_routes_registered_worker_to_strict_and_unregistered_to_explicit_legacy \
  tests/unit/engine/test_scheduler_drain_characterization.py::test_immediate_enqueue_routing_ast_and_legacy_production_references_are_pinned
```

Result: **18 passed, 13 warnings in 8.67s**.

These four focused runtime groups total 165 passing cases with no failures or
skips. They do not execute the open registered-process, source-state,
parent/child, or long-plugin seams.

## Security evidence

```bash
.venv/bin/pytest -q \
  tests/unit/core/landscape/test_graph_audit_config_boundary.py \
  tests/unit/core/landscape/test_graph_recording.py::TestRegisterNodeDsnSanitization \
  tests/unit/core/landscape/test_data_flow_repository.py::TestRegisterNodeDirect::test_fingerprints_secret_fields_before_persisting_node_config \
  tests/unit/core/landscape/test_data_flow_repository.py::TestRegisterNodeDirect::test_from_plugin_instances_configs_are_fingerprinted_before_node_persistence \
  tests/unit/core/test_config.py::TestSecretFieldFingerprinting \
  tests/integration/checkpoint/test_topology_validation.py \
  tests/unit/core/checkpoint/test_compatibility.py \
  tests/property/core/test_checkpoint_properties.py::TestTopologyHashProperties \
  tests/integration/audit/test_export.py \
  tests/unit/web/execution/test_preflight_side_effects.py::test_profile_private_bindings_never_enter_run_or_node_audit_config \
  tests/unit/engine/orchestrator/test_preflight_pipeline_config.py
```

Result: **90 passed in 10.45s**.

Those tests prove real partial controls for known secret fields and database
DSNs. Exact production-path probes additionally reproduced raw values in node
identity/metadata, topology hashing, run settings, non-database credential
URLs, raw-secret development mode, and commencement-gate conditions. The probe
commands and outputs follow.

### Production builder and topology identity probe

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from dataclasses import fields
from elspeth.contracts.freeze import deep_thaw
from elspeth.core.canonical import compute_full_topology_hash
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import NodeInfo
from tests.fixtures.plugins import CollectSink, ListSource

def build(secret: str) -> ExecutionGraph:
    source = ListSource([], name="list_source", on_success="output")
    source.config = {"schema": {"mode": "observed"}, "api_key": secret}
    sink = CollectSink("output")
    sink.config = {
        "schema": {"mode": "observed"},
        "headers": {"authorization_token": secret},
    }
    return ExecutionGraph.from_plugin_instances(
        sources={"primary": source},
        source_settings_map={
            "primary": SourceSettings(
                plugin=source.name,
                on_success="output",
                options={},
            )
        },
        transforms=[],
        sinks={"output": sink},
        aggregations={},
        gates=[],
    )

secret_a = "dag-probe-secret-a"
secret_b = "dag-probe-secret-b"
graph_a = build(secret_a)
graph_b = build(secret_b)
configs_a = [deep_thaw(node.config) for node in graph_a.get_nodes()]
ids_a = sorted(str(node.node_id) for node in graph_a.get_nodes())
ids_b = sorted(str(node.node_id) for node in graph_b.get_nodes())
print({
    "production_builder_metadata_contains_raw_secret": secret_a in repr(configs_a),
    "node_identity_changes_when_only_secret_changes": ids_a != ids_b,
    "checkpoint_topology_hash_changes_when_only_secret_changes": (
        compute_full_topology_hash(graph_a) != compute_full_topology_hash(graph_b)
    ),
    "node_info_carries_implementation_version": (
        "plugin_version" in {field.name for field in fields(NodeInfo)}
    ),
})
PY
git status --short
```

Observed result:

```text
production_builder_metadata_contains_raw_secret: true
node_identity_changes_when_only_secret_changes: true
checkpoint_topology_hash_changes_when_only_secret_changes: true
node_info_carries_implementation_version: false
```

### Audit/export boundary probe

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json
import os
from elspeth.contracts import NodeType, RunStatus
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.engine.orchestrator.bootstrap import prepare_for_run
from tests.fixtures.landscape import make_factory, make_landscape_db

prepare_for_run()
schema = SchemaConfig.from_dict({"mode": "observed"})

def begin(factory, run_id, config):
    factory.run_lifecycle.begin_run(
        config=config,
        canonical_version="probe-v1",
        run_id=run_id,
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )

os.environ["ELSPETH_FINGERPRINT_KEY"] = "probe-fingerprint-key"
os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
db1 = make_landscape_db()
f1 = make_factory(db1)
run_secret = "run-settings-probe-value"
begin(f1, "run-settings-probe", {"api_key": run_secret})
f1.run_lifecycle.complete_run("run-settings-probe", RunStatus.COMPLETED)
run_export = json.dumps(list(LandscapeExporter(db1).export_run("run-settings-probe")))

db2 = make_landscape_db()
f2 = make_factory(db2)
begin(f2, "url-probe", {})
url_secret = "url-probe-password"
f2.data_flow.register_node(
    run_id="url-probe",
    plugin_name="http",
    node_type=NodeType.SOURCE,
    plugin_version="1",
    config={"url": f"https://user:{url_secret}@example.test/path"},
    node_id="source-url",
    schema_config=schema,
)
f2.run_lifecycle.complete_run("url-probe", RunStatus.COMPLETED)
url_export = json.dumps(list(LandscapeExporter(db2).export_run("url-probe")))

os.environ.pop("ELSPETH_FINGERPRINT_KEY", None)
os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"
db3 = make_landscape_db()
f3 = make_factory(db3)
begin(f3, "raw-mode-probe", {})
node_secret = "node-raw-mode-probe-value"
f3.data_flow.register_node(
    run_id="raw-mode-probe",
    plugin_name="http",
    node_type=NodeType.SOURCE,
    plugin_version="1",
    config={"api_key": node_secret},
    node_id="source-secret",
    schema_config=schema,
)
f3.run_lifecycle.complete_run("raw-mode-probe", RunStatus.COMPLETED)
raw_mode_export = json.dumps(
    list(LandscapeExporter(db3).export_run("raw-mode-probe"))
)

print({
    "raw_run_setting_reaches_audit_export": run_secret in run_export,
    "non_database_credential_url_reaches_audit_export": url_secret in url_export,
    "allow_raw_secrets_reaches_node_audit_export": node_secret in raw_mode_export,
})
PY
git status --short
```

Observed result: all three predicates were `true`.

### Commencement-gate audit probe

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import json
from elspeth.contracts.preflight import CommencementGateResult, PreflightResult
from elspeth.core.dependency_config import CommencementGateConfig
from elspeth.core.landscape.schema import preflight_results_table
from elspeth.engine.orchestrator.bootstrap import prepare_for_run
from tests.fixtures.landscape import make_factory, make_landscape_db

prepare_for_run()
secret = "gate-literal-probe-value"
condition = f"env['READY'] == '{secret}'"
validated = CommencementGateConfig(name="probe", condition=condition)
db = make_landscape_db()
factory = make_factory(db)
factory.run_lifecycle.begin_run(
    config={},
    canonical_version="probe-v1",
    run_id="gate-probe",
    openrouter_catalog_sha256="0" * 64,
    openrouter_catalog_source="bundled",
)
factory.run_lifecycle.record_preflight_results(
    "gate-probe",
    PreflightResult(
        dependency_runs=(),
        gate_results=(
            CommencementGateResult(
                name="probe",
                condition=validated.condition,
                result=True,
                context_snapshot={},
            ),
        ),
    ),
)
with db.connection() as conn:
    row = conn.execute(
        preflight_results_table.select().where(
            preflight_results_table.c.run_id == "gate-probe"
        )
    ).one()
print({
    "secret_literal_condition_accepted_by_config": validated.condition == condition,
    "secret_literal_condition_persisted_in_audit": (
        secret in json.loads(row.result_json)["condition"]
    ),
})
PY
git status --short
```

Observed result: both predicates were `true`. An initial probe with
`result=False` correctly failed fixture validation and is not used as behavior
evidence.

## Browser acceptance

The four source specs were copied byte-identically to `/tmp`; SHA-256 pairs
matched. Pinned `@playwright/test` 1.59.0 ran with an empty temporary config so
the frozen worktree received no auth/data/report artifacts:

```bash
/tmp/elspeth-pw-evidence/node_modules/.bin/playwright test \
  --config=/tmp/elspeth-playwright-readonly.config.js \
  topology.spec.ts mandatory-fields.spec.ts \
  schema-preview-parity.spec.ts yaml-export-roundtrip.spec.ts
```

Result: **6 skipped, exit 0**. All six are describe-level `test.skip` cases.
This is collection evidence only, not browser acceptance.

## Loomweave refresh

```bash
loomweave analyze --no-incremental --json .
```

Result: run `4f698166-9a0c-423a-aea9-8c7bf1c5a256` completed with 55,274
entities, 129 subsystems, and 118,568 edges. Filigree emission was disabled.
The analyzer reported one bounded warning: reference resolution was skipped for
`src/elspeth/web/aws_ecs_acceptance.py` after it exceeded the 2,000-site cap.
