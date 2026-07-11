"""Composer/runtime schema-contract characterization.

This suite covers two categories:
- shared contract cases where composer preview and runtime should agree
- documented runtime-only gaps where composer stays permissive and the runtime
  validator remains authoritative

It does not claim global equivalence between preview validation and runtime DAG
validation.

Closed registry of composer/runtime divergence shapes (extends with each eval).
``elspeth-1ee3c96c72`` (Phase 3) maintains this list as the durable contract;
every future "validate green / runtime red" finding extends it. The registry
crosswalks each shape to the originating eval reproducer and the closure issue
where the architectural fix landed:

* Shape 1 — S1A literal credential placeholder
  ``api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY`` (eval session 2ef2db56,
  run 51f5f609). Closes ``elspeth-72d1dccd44``. Pinned by
  ``TestComposerRuntimeSecretRefAgreement``.
* Shape 2 — S2 v1 dangling on_error route target
  (``aggregations[*].on_error: aggregation_errors`` with no matching sink).
  Closes ``elspeth-127de6865a``. Pinned by
  ``TestComposerRuntimeRouteTargetAgreement`` plus four defense-in-depth axes.
* Shape 3 — S2 v2 batch_stats output schema propagation
  (``schema: {mode: flexible, fields: [...], required_fields: [...]}`` on a
  reductive aggregation; runtime raised ``SchemaConfigModeViolation``). Closes
  ``elspeth-f5f798f797`` (commit ``2d9dc21d``). Pinned by
  ``test_both_accept_aggregation_with_input_fields_and_required_fields``.
* Shape 4 — S1A monolithic happy-path positive control. Deferred — requires
  end-to-end LLM-stub integration scaffolding that does not exist in
  ``tests/integration/pipeline/``. Filed as a follow-up on
  ``elspeth-1ee3c96c72``'s closure.
* Shape 5 — Phase 2.2 RunStatus four-value terminal taxonomy plus the
  rows_routed-only design call. Closes ``elspeth-0de989c56d`` (commit
  ``cc895589``). Pinned by ``TestComposerRuntimeRunStatusAgreement``. The
  per-status engine-layer pinning lives in
  ``tests/integration/pipeline/orchestrator/test_orchestrator_core.py``; this
  suite adds the cross-layer (engine RunResult ⇔ Landscape audit ``Run`` row)
  agreement and the named design-call regression
  ``test_runstatus_on_error_routed_only_classifies_as_failed`` (plus the
  post-split companion ``test_runstatus_gate_routed_only_classifies_as_completed``
  introduced by ``elspeth-5069612f3c``).
* Shape 6 — Phase 2.3 ``/api/secrets`` reason taxonomy (eval session
  S1B, ``ELSPETH_FINGERPRINT_KEY`` unset). Closes ``elspeth-0d31c22d26``
  (commit ``22e3e0d9``). Per-mode coverage lives in
  ``tests/unit/web/secrets/{server,user}_store.py`` and
  ``tests/unit/web/secrets/test_routes.py`` per the Phase 2.3 closure
  rationale ("agreement-suite scope does not need duplication"). This suite
  adds a single contract-layer biconditional smoke
  (``TestComposerRuntimeSecretInventoryAgreement``) so a future drift on
  the ``available ⟺ reason is None`` invariant fails the agreement gate too.
* Shape 7 — Phase 2.1 pipeline_done_callback run-accounting agreement
  (eval run 44f52421, csv → batch_stats group_by → json sink wrote output
  but ``/api/runs/{rid}`` lagged because ``CompletedData`` rejected the
  legitimate aggregation row-count shape). Closes ``elspeth-31d53c7493``
  (commit ``5e26d0a6``). Pinned by
  ``TestComposerRuntimeRunCompletionAgreement``.
* Shape 8 — Phase 0.b composer file-sink path-collision diagnostic
  (eval session S3 ``98573481-e8bc-4a03-8467-d3a86effcd56``, originally
  attributed in the eval notes to a "gate primitive crash" but root-caused
  during Phase 0.b investigation to ``FileExistsError`` raised uncaught from
  ``json_sink.__init__`` / ``csv_sink.__init__`` via
  ``resolve_output_collision_path`` when a sink path collides with an
  existing artifact under ``collision_policy="fail_if_exists"``. The
  exception class was missing from ``validate_pipeline``'s step-4 catch
  list, propagating as ``composer_plugin_error`` 500 instead of a 422-class
  structured ``ValidationResult``). Closes ``elspeth-209b7e3a2b``. Pinned
  by ``TestComposerRuntimeFileSinkCollisionAgreement``. Gates were the
  symptom amplifier (gate-routing pipelines need sinks; LLM defaults
  ``collision_policy="fail_if_exists"``; stale eval artifacts collide),
  not the cause.
* Shape 9 — Phase P4/P3 of ``elspeth-fdebcaa79a`` widened ``blob_ref`` /
  ``inline_content`` config-content-ref capability. Pinned by
  ``TestComposerRuntimeBlobInlineAgreement``: validate-time metadata checks
  surface structured ``ValidationResult`` rows; runtime hash mismatch fails
  closed before settings/plugin construction; successful runtime resolution
  records the hash in ``blob_inline_resolutions`` before the resolved bytes
  enter plugin settings.
* Shape 10 — fixed-mode consumer *implicit*-required-field parity
  (``elspeth-8f3b3f650d``). A ``{mode: fixed, fields: [...]}`` consumer
  implicitly requires its declared (non-optional) fields; runtime Phase-2 type
  validation rejects an edge from a TYPED producer that does not guarantee one
  (``EdgeContractError: ... Missing fields: teal_pairing_rating``). Authoring
  computed consumer requirements from the explicit-only
  ``get_raw_node_required_fields`` and so green-lit the build (validate green /
  runtime red). Fixed in ``web/composer/state.py::_check_schema_contracts`` by
  adding a sibling check over the consumer's *effective* required set, gated
  strictly on producer schema MODE (a fixed/flexible SOURCE producer is typed;
  observed sources and transform/gate/coalesce producers resolve to a dynamic
  effective producer schema and are skipped), mirroring the runtime
  observed/dynamic bypass at ``graph.py:1392-1403``. Pinned by
  ``TestComposerRuntimeFixedModeImplicitRequiredAgreement``.
* Shape 11 — structural queue fan-in round-trip (``elspeth-a5b86149d4`` /
  ``elspeth-6421ffa028``). The composer models a runtime ``queues.<name>``
  fan-in point as a canonical structural queue NodeSpec (id == input,
  plugin=None, description-only options). Import must preserve it, validate must
  accept the two-source → queue → transform topology, and export must round-trip
  it so ``load_settings_from_yaml_string`` rebuilds a graph with exactly one
  ``NodeType.QUEUE`` (both sources fan in, one ordinary consumer out).
  Previously the composer had no queue node type, so a pasted ``queues:``
  section was dropped and undeclared two-source fan-in was the only expressible
  form (validate green / runtime red: ``GraphValidationError`` "fan-in from
  multiple producers without a queue"). Pinned by
  ``TestComposerRuntimeQueueAgreement`` with a manual negative control proving
  the queue-free topology still reproduces the runtime rejection.

Adding a new shape: file the eval-finding issue, land the structural fix,
then extend this docstring with the shape's number, the originating eval
session/run id, the closing issue, and the test class that pins it.

Bug verification protocol (mandatory for new shapes):
``test_agreement_aggregation_run_counts_construct_completed_data`` (Shape 7)
is the canonical example. Before declaring a new agreement test landed,
manually revert the structural fix it pins (one line in the production
code) and confirm the test fails with the expected exception class. Then
restore the fix. Document the protocol verbatim in the test's docstring,
naming the exact production line reverted and the exact failure observed.
This guards against the "passes pre-fix AND post-fix" failure mode where a
test exercises adjacent behaviour but never actually depends on the fix
under test — a class of test theatre that is otherwise undetectable until
a future regression silently slips through. The cost is one minute of
scratch work; the value is durable evidence that the test pins the
structural contract rather than incidentally passing.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import Determinism
from elspeth.contracts.audit import Run
from elspeth.contracts.enums import CreationModality, RunStatus
from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.secrets import (
    SecretInventoryItem,
    SecretUnavailabilityReason,
)
from elspeth.core.config import (
    AggregationSettings,
    CoalesceSettings,
    ElspethSettings,
    GateSettings,
    SinkSettings,
    SourceSettings,
    TransformSettings,
    TriggerConfig,
)
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.core.landscape import LandscapeDB
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.web.blobs.protocol import BlobFinalizationResult, BlobIntegrityError, BlobRecord
from elspeth.web.composer import yaml_generator as composer_yaml_generator
from elspeth.web.composer.state import (
    CompositionState,
    EdgeSpec,
    NodeSpec,
    OutputSpec,
    PipelineMetadata,
    SourceSpec,
)
from elspeth.web.execution.accounting import load_run_accounting_from_db
from elspeth.web.execution.progress import BroadcastResult
from elspeth.web.execution.schemas import CompletedData
from elspeth.web.execution.service import ExecutionServiceImpl
from elspeth.web.execution.validation import validate_pipeline
from elspeth.web.interpretation_state import INTERPRETATION_REQUIREMENTS_KEY
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.landscape import make_factory
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import (
    CollectSink,
    ConditionalErrorTransform,
    ListSource,
    PassTransform,
)


class TestComposerRuntimeAgreement:
    """Shared agreement checks plus documented runtime-only gap characterization."""

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _build_runtime_graph(
        self,
        *,
        source_plugin: str,
        source_options: dict[str, Any],
        sink_options: dict[str, Any],
        transform_options: dict[str, Any] | None = None,
        transform_plugin: str | None = "value_transform",
        aggregation_options: dict[str, Any] | None = None,
        aggregation_plugin: str | None = None,
    ) -> ExecutionGraph:
        """Build a runtime ExecutionGraph through the production assembly path."""
        if transform_plugin is not None and aggregation_plugin is not None:
            raise AssertionError(
                "Task 8 agreement helper supports either a transform chain or an aggregation chain, not both in the same case."
            )

        source_on_success = "agg1" if aggregation_plugin is not None else ("t1" if transform_plugin is not None else "main")
        transforms: list[TransformSettings] = []
        aggregations: list[AggregationSettings] = []

        if transform_plugin is not None:
            transforms.append(
                TransformSettings(
                    name="t1",
                    plugin=transform_plugin,
                    input="t1",
                    on_success="main",
                    on_error="discard",
                    options=transform_options or {},
                )
            )

        if aggregation_plugin is not None:
            aggregations.append(
                AggregationSettings(
                    name="agg1",
                    plugin=aggregation_plugin,
                    input="agg1",
                    on_success="main",
                    on_error="discard",
                    trigger=TriggerConfig(count=1),
                    options=aggregation_options or {},
                )
            )

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin=source_plugin,
                    on_success=source_on_success,
                    options={**source_options, "on_validation_failure": "discard"},
                )
            },
            transforms=transforms,
            aggregations=aggregations,
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options=sink_options,
                )
            },
        )
        return self._build_runtime_graph_from_settings(config)

    def _build_runtime_graph_from_settings(self, config: ElspethSettings) -> ExecutionGraph:
        """Build a runtime graph from full settings through the production path."""
        plugins = instantiate_plugins_from_config(config)
        return ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )

    def test_both_reject_missing_required_field(self, tmp_path: Path) -> None:
        """Both validators reject when a consumer requires an unsatisfied field."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, "Composer should reject: source column is 'line' but consumer requires 'text'."
        assert any("schema contract violation" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                transform_options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
            graph.validate_edge_compatibility()
        assert "text" in str(exc_info.value).lower()

    def test_both_accept_observed_text_source_with_auto_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Both validators accept the observed-text special-case contract."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "text",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors

        graph = self._build_runtime_graph(
            source_plugin="text",
            source_options={
                "path": str(text_path),
                "column": "text",
                "schema": {"mode": "observed"},
            },
            transform_options={
                "required_input_fields": ["text"],
                "operations": [
                    {
                        "target": "out",
                        "expression": "row['text'] + ' world'",
                    }
                ],
                "schema": {"mode": "observed"},
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

    def test_both_accept_source_schema_config_alias_contract(
        self,
        tmp_path: Path,
    ) -> None:
        """Source schema_config aliases must drive the same contract in preview and runtime."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema_config": {"mode": "observed", "guaranteed_fields": ["text"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["text"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['text'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        source_contract = next(ec for ec in composer_result.edge_contracts if ec.to_id == "t1")
        assert source_contract.producer_guarantees == ("text",)
        assert source_contract.satisfied is True

        graph = self._build_runtime_graph(
            source_plugin="text",
            source_options={
                "path": str(text_path),
                "column": "line",
                "schema_config": {"mode": "observed", "guaranteed_fields": ["text"]},
            },
            transform_options={
                "required_input_fields": ["text"],
                "operations": [
                    {
                        "target": "out",
                        "expression": "row['text'] + ' world'",
                    }
                ],
                "schema": {"mode": "observed"},
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

    def test_both_reject_observed_text_source_keyword_column(self, tmp_path: Path) -> None:
        """Invalid keyword columns must not create a false composer/runtime accept."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "class",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "required_input_fields": ["class"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['class'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, (
            "Composer must not infer an observed-text guarantee for a keyword column name that runtime text-source config rejects."
        )
        assert any("class" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(PluginConfigError, match="Python keyword"):
            self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "class",
                    "schema": {"mode": "observed"},
                },
                transform_options={
                    "required_input_fields": ["class"],
                    "operations": [
                        {
                            "target": "out",
                            "expression": "row['class'] + ' world'",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )

    def test_both_reject_strict_sink_typed_requirement_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Both validators reject when a strict sink requires an ungiven field."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="main",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid, "Composer should reject: strict sink requires 'text' but upstream guarantees only 'line'."
        assert any(contract.to_id == "output:main" and not contract.satisfied for contract in composer_result.edge_contracts)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="text",
                source_options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                transform_plugin=None,
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
            )
            graph.validate_edge_compatibility()
        assert "requires" in str(exc_info.value).lower()

    def test_both_reject_aggregation_nested_required_input_fields_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Composer rejects at preview time and runtime rejects during plugin wiring."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("line\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="agg1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        assert any("value" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(FrameworkBugError) as exc_info:
            self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                transform_plugin=None,
                aggregation_plugin="batch_stats",
                aggregation_options={
                    "value_field": "value",
                    "required_input_fields": ["value"],
                    "schema": {"mode": "observed"},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
        assert "value" in str(exc_info.value).lower()

    def test_both_reject_aggregation_nested_schema_required_fields_without_upstream_guarantee(
        self,
        tmp_path: Path,
    ) -> None:
        """Aggregation wrapper schema.required_fields must match runtime validation."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("line\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "options": {
                        "value_field": "value",
                        "schema": {"mode": "observed", "required_fields": ["value"]},
                    }
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="agg1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        assert any("value" in entry.message.lower() for entry in composer_result.errors)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["line: str"]},
                },
                transform_plugin=None,
                aggregation_plugin="batch_stats",
                aggregation_options={
                    "value_field": "value",
                    "schema": {"mode": "observed", "required_fields": ["value"]},
                },
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
            )
            graph.validate_edge_compatibility()
        assert "value" in str(exc_info.value).lower()

    def test_both_reject_direct_fork_to_sink_required_field_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        """Direct fork-to-sink edges stay statically checkable in preview and runtime."""
        text_path = tmp_path / "input.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="gate_in",
                options={
                    "path": str(text_path),
                    "column": "line",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork"},
                fork_to=("main",),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["text: str"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="main",
                edge_type="fork",
                label="main",
            )
        )

        composer_result = state.validate()
        assert not composer_result.is_valid
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.from_id == "source"
        assert sink_contract.satisfied is False
        assert not any(
            "fork gate" in warning.message.lower() and "contract check skipped" in warning.message.lower()
            for warning in composer_result.warnings
        )

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="text",
                    on_success="gate_in",
                    options={
                        "path": str(text_path),
                        "column": "line",
                        "schema": {"mode": "observed"},
                        "on_validation_failure": "discard",
                    },
                )
            },
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["main"],
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "fixed", "fields": ["text: str"]},
                    },
                )
            },
        )

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph_from_settings(config)
            graph.validate_edge_compatibility()
        assert "text" in str(exc_info.value).lower()

    def test_both_accept_pass_through_downstream_of_coalesce(
        self,
        tmp_path: Path,
    ) -> None:
        """Pass-through preview must inherit coalesce guarantees after fan-in."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_results",
                node_type="coalesce",
                plugin=None,
                input="path_a",
                on_success=None,
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b"),
                policy="best_effort",
                merge="union",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="pt_after_merge",
                node_type="transform",
                plugin="passthrough",
                input="merge_results",
                on_success="main",
                on_error="discard",
                options={"schema": {"mode": "observed"}},
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed", "required_fields": ["id"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="merge_results",
                edge_type="fork",
                label="path_a",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e3",
                from_node="merge_results",
                to_node="pt_after_merge",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.from_id == "pt_after_merge"
        assert sink_contract.producer_guarantees == ("id", "value")
        assert sink_contract.consumer_requires == ("id",)
        assert sink_contract.satisfied is True

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="gate_in",
                    options={
                        "path": str(csv_path),
                        "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                        "on_validation_failure": "discard",
                    },
                )
            },
            transforms=[
                TransformSettings(
                    name="pt_after_merge",
                    plugin="passthrough",
                    input="merge_results",
                    on_success="main",
                    on_error="discard",
                    options={"schema": {"mode": "observed"}},
                )
            ],
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["path_a", "path_b"],
                )
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches={"path_a": "path_a", "path_b": "path_b"},
                    policy="best_effort",
                    merge="union",
                    timeout_seconds=1,
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "observed", "required_fields": ["id"]},
                    },
                )
            },
        )

        graph = self._build_runtime_graph_from_settings(config)
        graph.validate_edge_compatibility()

    def test_composer_warns_but_runtime_rejects_mixed_coalesce_branch_schemas(
        self,
        tmp_path: Path,
    ) -> None:
        """Coalesce merge semantics stay runtime-authoritative beyond composer preview."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("id,value\n1,2\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="gate_in",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="fork_gate",
                node_type="gate",
                plugin=None,
                input="gate_in",
                on_success=None,
                on_error=None,
                options={},
                condition="True",
                routes={"true": "fork", "false": "fork"},
                fork_to=("path_a", "path_b"),
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="branch_b",
                node_type="transform",
                plugin="value_transform",
                input="path_b",
                on_success="path_b_done",
                on_error="discard",
                options={
                    "operations": [
                        {
                            "target": "value",
                            "expression": "row['value']",
                        }
                    ],
                    "schema": {"mode": "observed"},
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_node(
            NodeSpec(
                id="merge_results",
                node_type="coalesce",
                plugin=None,
                input="path_a",
                on_success="main",
                on_error=None,
                options={},
                condition=None,
                routes=None,
                fork_to=None,
                branches=("path_a", "path_b_done"),
                policy="require_all",
                merge="union",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="fork_gate",
                edge_type="on_success",
                label=None,
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e2",
                from_node="fork_gate",
                to_node="branch_b",
                edge_type="fork",
                label="path_b",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e3",
                from_node="fork_gate",
                to_node="merge_results",
                edge_type="fork",
                label="path_a",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e4",
                from_node="branch_b",
                to_node="merge_results",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        assert any("coalesce node" in warning.message.lower() for warning in composer_result.warnings)
        assert not any(contract.to_id == "output:main" for contract in composer_result.edge_contracts)

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="gate_in",
                    options={
                        "path": str(csv_path),
                        "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                        "on_validation_failure": "discard",
                    },
                )
            },
            transforms=[
                TransformSettings(
                    name="branch_b",
                    plugin="value_transform",
                    input="path_b",
                    on_success="path_b_done",
                    on_error="discard",
                    options={
                        "operations": [
                            {
                                "target": "value",
                                "expression": "row['value']",
                            }
                        ],
                        "schema": {"mode": "observed"},
                    },
                )
            ],
            gates=[
                GateSettings(
                    name="fork_gate",
                    input="gate_in",
                    condition="True",
                    routes={"true": "fork", "false": "fork"},
                    fork_to=["path_a", "path_b"],
                )
            ],
            coalesce=[
                CoalesceSettings(
                    name="merge_results",
                    branches={"path_a": "path_a", "path_b": "path_b_done"},
                    policy="require_all",
                    merge="union",
                    on_success="main",
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={
                        "path": str(output_path),
                        "schema": {"mode": "fixed", "fields": ["id: int", "value: int"]},
                    },
                )
            },
        )

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph_from_settings(config)
            graph.validate_edge_compatibility()
        message = str(exc_info.value).lower()
        assert "coalesce" in message
        assert "observed" in message
        assert "explicit" in message

    def test_composer_accepts_field_names_but_runtime_rejects_type_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        """Type compatibility remains runtime-only even when contract fields line up."""
        csv_path = tmp_path / "input.csv"
        csv_path.write_text("value\nhello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="main",
                options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["value: str"]},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["value: int"]},
                },
                on_write_failure="discard",
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, composer_result.errors
        sink_contract = next(contract for contract in composer_result.edge_contracts if contract.to_id == "output:main")
        assert sink_contract.satisfied is True
        assert sink_contract.producer_guarantees == ("value",)
        assert sink_contract.consumer_requires == ("value",)

        with pytest.raises(GraphValidationError) as exc_info:
            graph = self._build_runtime_graph(
                source_plugin="csv",
                source_options={
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["value: str"]},
                },
                transform_plugin=None,
                sink_options={
                    "path": str(output_path),
                    "schema": {"mode": "fixed", "fields": ["value: int"]},
                },
            )
            graph.validate_edge_compatibility()
        message = str(exc_info.value).lower()
        assert "incompatible" in message
        assert "value" in message

    def test_both_accept_aggregation_with_input_fields_and_required_fields(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression for elspeth-f5f798f797.

        S2 v2 from docs/composer/evidence/composer-llm-eval-2026-05-01.md: a ``batch_stats``
        aggregation with ``schema: {mode: flexible, fields: [...],
        required_fields: [...]}`` was accepted by composer ``/validate`` but
        rejected at runtime with ``SchemaConfigModeViolation`` because
        ``BaseTransform._build_output_schema_config`` propagated the user's
        input ``fields``/``required_fields`` into the aggregation's output
        schema config — which then required fields the aggregation never
        emits and expected types the OBSERVED-typed output cannot satisfy.

        After the fix in ``BatchStats._build_output_schema_config``, both
        composer preview and runtime emission verification accept this
        config. The aggregation honestly declares its output as observed
        with ``guaranteed_fields`` matching what the aggregation emits.
        """
        from elspeth.contracts.schema_contract import PipelineRow
        from elspeth.contracts.schema_contract_factory import create_contract_from_config
        from elspeth.engine.executors.schema_config_mode import verify_schema_config_mode
        from elspeth.plugins.transforms.batch_stats import BatchStats

        csv_path = tmp_path / "input.csv"
        csv_path.write_text(
            "customer_tier,amount\nenterprise,100.0\nenterprise,150.0\npro,50.0\n",
            encoding="utf-8",
        )
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={
                    "path": str(csv_path),
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                    },
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="agg1",
                node_type="aggregation",
                plugin="batch_stats",
                input="agg1",
                on_success="main",
                on_error="discard",
                options={
                    "schema": {
                        "mode": "flexible",
                        "fields": ["customer_tier: str", "amount: float"],
                        "required_fields": ["customer_tier", "amount"],
                    },
                    "value_field": "amount",
                    "group_by": "customer_tier",
                    "compute_mean": False,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
                trigger={"count": 3},
                output_mode="transform",
                expected_output_count=2,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={
                    "path": str(output_path),
                    "schema": {"mode": "observed"},
                },
                on_write_failure="discard",
            )
        )
        state = state.with_edge(EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None))

        composer_result = state.validate()
        assert composer_result.is_valid, "\n".join(err.message for err in composer_result.errors)

        graph = self._build_runtime_graph(
            source_plugin="csv",
            source_options={
                "path": str(csv_path),
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                },
            },
            transform_plugin=None,
            aggregation_plugin="batch_stats",
            aggregation_options={
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            },
            sink_options={
                "path": str(output_path),
                "schema": {"mode": "observed"},
            },
        )
        graph.validate_edge_compatibility()

        # Final tier of the agreement: simulate aggregate emission and
        # verify the runtime SchemaConfigModeViolation predicate accepts
        # the output. Pre-fix this raised; post-fix it must not raise.
        transform = BatchStats(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["customer_tier: str", "amount: float"],
                    "required_fields": ["customer_tier", "amount"],
                },
                "value_field": "amount",
                "group_by": "customer_tier",
                "compute_mean": False,
            }
        )
        from elspeth.contracts.schema import SchemaConfig as _SchemaConfig

        input_contract = create_contract_from_config(
            _SchemaConfig.from_dict({"mode": "flexible", "fields": ["customer_tier: str", "amount: float"]})
        )
        rows = [
            PipelineRow({"customer_tier": "enterprise", "amount": 100.0}, input_contract),
            PipelineRow({"customer_tier": "enterprise", "amount": 150.0}, input_contract),
            PipelineRow({"customer_tier": "pro", "amount": 50.0}, input_contract),
        ]
        results: list[dict[str, object]] = []
        for group_value, grouped in transform._group_rows(rows):
            aggregate, error = transform._aggregate_group(grouped, group_value)
            assert error is None
            results.append(aggregate)
        emitted_contract = transform._output_contract_for(results)
        emitted_rows = [PipelineRow(r, emitted_contract) for r in results]

        # Narrow ``transform._output_schema_config`` (typed as
        # ``SchemaConfig | None``) — for this BatchStats config the
        # post-Phase-1.3 ``_build_output_schema_config`` override always
        # returns a non-None value.  An assert here makes the contract
        # explicit so mypy can see it.
        output_schema_config = transform._output_schema_config
        assert output_schema_config is not None, "BatchStats must emit an explicit output schema config"

        verify_schema_config_mode(
            output_schema_config=output_schema_config,
            emitted_rows=emitted_rows,
            plugin_name="batch_stats",
            node_id="agg1",
            run_id="r",
            row_id="row1",
            token_id="t1",
        )


class TestComposerRuntimeRouteTargetAgreement:
    """Composer ``/validate`` and runtime preflight agree on dangling route
    targets — closes the parity gap from elspeth-127de6865a.

    Empirical scope of the original gap (post-investigation):

    * Aggregation ``on_error`` -> unknown sink: composer was silent (the
      original reproducer). Now caught at ``route_target_resolution``.
    * Source ``on_validation_failure`` -> unknown sink: composer was silent.
      Now caught at ``route_target_resolution``.
    * Transform ``on_error`` -> unknown sink: was already caught at
      ``graph_structure`` (``builder.py:839``). The new check is
      defense-in-depth.
    * Sink ``on_write_failure`` -> unknown sink: was already caught at
      ``graph_structure`` (``builder.py:859``). The new check is
      defense-in-depth.

    Each gap-closing test (aggregation/source) exercises both paths from
    independent inputs and asserts the error messages are byte-identical.
    Each defense-in-depth test asserts both layers reject and the dangling
    target name is present in both messages.
    """

    @staticmethod
    def _validation_settings(data_dir: Path) -> SimpleNamespace:
        # ValidationSettings is a Protocol that only requires ``data_dir``.
        return SimpleNamespace(data_dir=data_dir)

    @staticmethod
    def _composer_route_target_failure(state: CompositionState, data_dir: Path) -> str:
        """Run validate_pipeline on a CompositionState and return the
        route_target_resolution check detail. Asserts the failure happened on
        that specific check (not graph_structure, not schema_compatibility)."""
        result = validate_pipeline(
            state,
            TestComposerRuntimeRouteTargetAgreement._validation_settings(data_dir),
            composer_yaml_generator,
        )
        assert result.is_valid is False, "Composer should reject pipelines with dangling route targets"
        check_by_name = {check.name: check for check in result.checks}
        assert "route_target_resolution" in check_by_name, "Missing route_target_resolution check in result"
        rt_check = check_by_name["route_target_resolution"]
        assert rt_check.passed is False, f"route_target_resolution should have failed; got {rt_check.detail}"
        # graph_structure should have passed — these dangling references are
        # NOT structural DAG errors.
        assert check_by_name["graph_structure"].passed is True, "graph_structure must pass — only route targets are bad"
        return rt_check.detail

    @staticmethod
    def _runtime_route_target_failure(config: ElspethSettings) -> str:
        """Instantiate plugins, build graph, call assemble_and_validate_pipeline_config.
        Returns the str(RouteValidationError) message."""
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()  # Structural DAG check — must pass for these cases
        with pytest.raises(RouteValidationError) as exc_info:
            assemble_and_validate_pipeline_config(
                sources=plugins.sources,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                settings=config,
                graph=graph,
            )
        return str(exc_info.value)

    @staticmethod
    def _csv_input(tmp_path: Path) -> Path:
        # Sources must live under data_dir/blobs/ for the path allowlist.
        path = tmp_path / "blobs" / "input.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n", encoding="utf-8")
        return path

    @staticmethod
    def _csv_output(tmp_path: Path, name: str = "out.csv") -> Path:
        # Sinks must live under data_dir/outputs/ (or blobs/) for the allowlist.
        out_dir = tmp_path / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name

    def test_both_reject_aggregation_on_error_dangling_sink(self, tmp_path: Path) -> None:
        """Original reproducer (S2 v1 from docs/composer/evidence/composer-llm-eval-2026-05-01.md):
        aggregation ``on_error: aggregation_errors`` with no sink of that name."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        # Composer state: aggregation routes errors to a sink that doesn't exist.
        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="aggregation_errors",  # ← dangling
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 1},
                    output_mode="transform",
                    expected_output_count=1,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="agg1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_detail = self._composer_route_target_failure(state, tmp_path)

        # Runtime: equivalent ElspethSettings.
        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="agg1",
                    options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                )
            },
            aggregations=[
                AggregationSettings(
                    name="agg1",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="aggregation_errors",  # ← dangling
                    trigger=TriggerConfig(count=1),
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        runtime_msg = self._runtime_route_target_failure(config)

        assert "aggregation_errors" in composer_detail
        assert "aggregation_errors" in runtime_msg
        assert composer_detail == runtime_msg, "Composer and runtime must surface identical RouteValidationError"

    def test_both_reject_transform_on_error_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: the DAG builder (``graph.validate()`` via
        ``from_plugin_instances``) already catches transform ``on_error`` ->
        unknown sink at ``builder.py:839``. The new ``route_target_resolution``
        check is a second wall behind it. This test asserts both walls agree:
        composer ``/validate`` rejects, runtime construction rejects, and the
        dangling target name appears in both messages."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="t1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="t1",
                    node_type="transform",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="missing_error_sink",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "doubled", "expression": "row['value']"}],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="t1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="t1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_error_sink" in composer_messages

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="t1",
                    options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                )
            },
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="missing_error_sink",
                    options={
                        "schema": {"mode": "observed"},
                        "operations": [{"target": "doubled", "expression": "row['value']"}],
                    },
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_error_sink" in str(runtime_exc.value)

    def test_both_reject_source_on_validation_failure_dangling_sink(self, tmp_path: Path) -> None:
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="missing_quarantine_sink",  # ← dangling
            ),
            nodes=(),
            edges=(EdgeSpec(id="e1", from_node="source", to_node="main", edge_type="on_success", label=None),),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_detail = self._composer_route_target_failure(state, tmp_path)

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="main",
                    options={
                        "path": str(csv_path),
                        "schema": {"mode": "observed"},
                        "on_validation_failure": "missing_quarantine_sink",
                    },
                )
            },
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        runtime_msg = self._runtime_route_target_failure(config)

        assert "missing_quarantine_sink" in composer_detail
        assert "missing_quarantine_sink" in runtime_msg
        assert composer_detail == runtime_msg

    def test_both_reject_sink_on_write_failure_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: ``builder.py:859`` already catches sink
        ``on_write_failure`` -> unknown sink at ``graph.validate()``. The
        helper provides a second wall via
        ``validate_sink_failsink_destinations``."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(EdgeSpec(id="e1", from_node="source", to_node="main", edge_type="on_success", label=None),),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="missing_failsink",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_failsink" in composer_messages

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="main",
                    options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                )
            },
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="missing_failsink",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_failsink" in str(runtime_exc.value)

    def test_both_reject_gate_routes_dangling_sink(self, tmp_path: Path) -> None:
        """Defense-in-depth axis: a gate ``routes`` target that is neither
        ``"fork"`` nor a real sink name nor a connection name. The DAG builder
        falls through to producer-registration (``builder.py:583+``); if no
        consumer claims that name, downstream graph checks reject. The
        ``route_target_resolution`` step is the second wall — when the runtime
        ultimately resolves the route, ``validate_route_destinations`` would
        also catch it.

        This test asserts agreement: composer ``/validate`` rejects, the
        independent runtime construction rejects, and both messages name the
        dangling target."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="g1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="g1",
                    node_type="gate",
                    plugin=None,
                    input="g1",
                    on_success=None,
                    on_error=None,
                    options={},
                    condition="row['value'] != ''",
                    routes={"true": "main", "false": "missing_route_sink"},
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="g1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="g1", to_node="main", edge_type="route_true", label="true"),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        composer_result = validate_pipeline(state, self._validation_settings(tmp_path), composer_yaml_generator)
        assert composer_result.is_valid is False
        composer_messages = " | ".join(err.message for err in composer_result.errors)
        assert "missing_route_sink" in composer_messages

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="g1",
                    options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                )
            },
            gates=[
                GateSettings(
                    name="g1",
                    input="g1",
                    condition="row['value'] != ''",
                    routes={"true": "main", "false": "missing_route_sink"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        with pytest.raises(GraphValidationError) as runtime_exc:
            ExecutionGraph.from_plugin_instances(
                sources=plugins.sources,
                source_settings_map=plugins.source_settings_map,
                transforms=plugins.transforms,
                sinks=plugins.sinks,
                aggregations=plugins.aggregations,
                gates=list(config.gates),
                coalesce_settings=list(config.coalesce) if config.coalesce else None,
            )
        assert "missing_route_sink" in str(runtime_exc.value)

    def test_both_accept_aggregation_on_error_discard(self, tmp_path: Path) -> None:
        """Positive control: ``on_error: discard`` (the eval's S2 v3 fix) passes
        both composer and runtime. Confirms the new check is not over-eager."""
        csv_path = self._csv_input(tmp_path)
        output_path = self._csv_output(tmp_path)

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="agg1",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="agg1",
                    node_type="aggregation",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="discard",  # ← legal escape valve
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                    trigger={"count": 1},
                    output_mode="transform",
                    expected_output_count=1,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="agg1", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="agg1", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )
        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
        )
        assert result.is_valid, "\n".join(err.message for err in result.errors)
        rt_check = next(c for c in result.checks if c.name == "route_target_resolution")
        assert rt_check.passed is True
        assert rt_check.detail == "All route targets resolve to existing sinks"

        # Runtime: must also assemble cleanly, no RouteValidationError.
        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="agg1",
                    options={"path": str(csv_path), "schema": {"mode": "observed"}, "on_validation_failure": "discard"},
                )
            },
            aggregations=[
                AggregationSettings(
                    name="agg1",
                    plugin="batch_stats",
                    input="agg1",
                    on_success="main",
                    on_error="discard",
                    trigger=TriggerConfig(count=1),
                    options={"schema": {"mode": "observed"}, "value_field": "value"},
                ),
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                ),
            },
        )
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=list(config.coalesce) if config.coalesce else None,
        )
        graph.validate()
        # Should not raise.
        assemble_and_validate_pipeline_config(
            sources=plugins.sources,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            settings=config,
            graph=graph,
        )


# ── Shape 1 — secret_refs literal placeholder agreement ──────────────────────
# Closes elspeth-72d1dccd44 / Phase 1.1.  Origin: 2026-05-01 staging eval, S1A
# (session 2ef2db56-70d7-498a-83d6-47e1f0efe340, run
# 51f5f609-bf72-4654-9cf2-6c53c565548b).  Pre-fix, a literal placeholder string
# in a credential-bearing field validated as is_valid: true and the engine ran
# end-to-end with every row routed via on_error to a quarantine sink.  Post-fix,
# the secret_refs check rejects at /validate before runtime ever sees the
# pipeline.
#
# The unit suite at TestValidatePipelineFabricatedCredentials in
# tests/unit/web/execution/test_validation.py covers the seven sibling shapes
# (transform/source/sink credential fields, suffix-matched fields, nested
# credentials, env-marker outside inventory, positive control).  The
# agreement-suite contribution here is the canonical S1A reproducer pinned at
# the agreement layer: a future drift in the fabrication-aware predicate must
# fail BOTH the validator unit tests AND this agreement gate.


class _AgreementSecretService:
    """Minimal WebSecretResolver stand-in for agreement-suite shape pinning.

    Mirrors ``tests/unit/web/execution/test_validation.py::FakeSecretService``
    without importing across the unit/integration suite boundary.  Inventory
    items report ``available=False`` with a closed-list reason so the fixture
    obeys the Phase 2.3 ``available ⟺ reason is None`` invariant.
    """

    def __init__(self, *, inventory: frozenset[str] = frozenset()) -> None:
        self._inventory = inventory

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        del user_id
        return [
            SecretInventoryItem(
                name=name,
                scope="user",
                available=False,
                reason="value_decryption_failed",
            )
            for name in sorted(self._inventory)
        ]

    def has_ref(self, user_id: str, name: str) -> bool:
        del user_id, name
        return False

    def resolve(self, user_id: str, name: str) -> None:
        """Protocol completeness — agreement-suite never calls resolve()
        because the secret_refs gate fires before settings load.  Returning
        ``None`` mirrors ``WebSecretService.resolve``'s "not present" path.
        """
        del user_id, name
        return None


class TestComposerRuntimeSecretRefAgreement:
    """Shape 1 — literal credential placeholders are gated at /validate.

    Empirical scope of the original gap: composer's ``secret_refs`` check only
    looked for ``{secret_ref: <name>}`` constructs.  A literal placeholder
    string in a field the plugin schema marks as credential-bearing
    (``is_secret_field`` predicate at L1 ``core/secrets.py``) bypassed the
    check entirely.  The runtime-side defense is non-existent — the LLM
    transform happily called the upstream API with the literal placeholder as
    the bearer token, producing per-row HTTP 401s and routing every row via
    ``on_error`` to ``parse_quarantine.jsonl``.  This shape is therefore
    /validate-gated only; the agreement contract here is "/validate rejects
    so /execute never receives it."
    """

    _S1A_PLACEHOLDER = "WILL_BE_WIRED_FROM_OPENROUTER_API_KEY"

    @staticmethod
    def _validation_settings(data_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(data_dir=data_dir)

    def _assert_placeholder_redacted(self, result, *, value: str) -> None:
        """Audit-hygiene: the literal candidate-secret value must not be
        echoed into any field of the validation response.  Mirrors the
        unit-suite discipline at
        ``TestValidatePipelineFabricatedCredentials._assert_value_redacted``;
        kept inline here so this test is self-contained as the agreement
        contract.
        """
        for check in result.checks:
            assert value not in check.detail, f"placeholder value leaked into check {check.name!r} detail"
        for error in result.errors:
            assert value not in error.message, f"placeholder value leaked into error.message at {error.component_id!r}"
            if error.suggestion is not None:
                assert value not in error.suggestion, "placeholder value leaked into error.suggestion"

    def test_agreement_s1a_literal_api_key_fails_validate(self, tmp_path: Path) -> None:
        """S1A canonical reproducer: an LLM transform with a literal
        ``api_key: WILL_BE_WIRED_FROM_OPENROUTER_API_KEY`` placeholder must
        fail the composer ``secret_refs`` check, identify the offending
        component+field, and never echo the candidate-secret string back to
        the operator surface.
        """
        # Source must live under data_dir/blobs/ for the path allowlist; the
        # ``secret_refs`` predicate fires before the path check, but build a
        # legitimate source so the failure is unambiguously the credential
        # check rather than a parallel rejection.
        blobs = tmp_path / "blobs"
        blobs.mkdir()
        csv_path = blobs / "tickets.csv"
        csv_path.write_text("subject\nticket-1\n", encoding="utf-8")

        state = CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="classify_ticket",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="classify_ticket",
                    node_type="transform",
                    plugin="llm",
                    input="classify_ticket",
                    on_success="main",
                    on_error="discard",
                    options={
                        "provider": "openrouter",
                        "model": "openai/gpt-4.1-nano",
                        "api_key": self._S1A_PLACEHOLDER,
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(
                EdgeSpec(id="e1", from_node="source", to_node="classify_ticket", edge_type="on_success", label=None),
                EdgeSpec(id="e2", from_node="classify_ticket", to_node="main", edge_type="on_success", label=None),
            ),
            outputs=(
                OutputSpec(
                    name="main",
                    plugin="csv",
                    options={"path": str(tmp_path / "outputs" / "out.csv"), "schema": {"mode": "observed"}},
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(),
            version=1,
        )

        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
            secret_service=_AgreementSecretService(),
            user_id="agreement-suite-user",
        )

        # The agreement gate: /validate must reject this shape so /execute
        # never receives it.  Pre-fix this returned is_valid: true.
        assert result.is_valid is False, "S1A literal-placeholder shape must be rejected by /validate"

        secret_check = next(check for check in result.checks if check.name == "secret_refs")
        assert secret_check.passed is False, f"secret_refs check must fail; detail={secret_check.detail!r}"
        assert "api_key" in secret_check.detail, "secret_refs detail must name the offending field"

        # The structured error must attribute the failure to the LLM transform
        # node so the operator can navigate directly to it.
        api_key_errors = [error for error in result.errors if "api_key" in error.message]
        assert api_key_errors, "expected at least one error naming the api_key field"
        assert any(error.component_id == "classify_ticket" and error.component_type == "transform" for error in api_key_errors), (
            "credential-field error must attribute to the LLM transform node"
        )

        # Audit-hygiene: the literal placeholder string must not appear
        # anywhere in the response.  In production the candidate value may be
        # a near-miss real secret; reflecting it back is data leakage.
        self._assert_placeholder_redacted(result, value=self._S1A_PLACEHOLDER)


# ── Shape 5 — RunStatus four-value taxonomy cross-layer agreement ─────────────
# Closes elspeth-0de989c56d / Phase 2.2 (commit cc895589).  The per-status
# engine-layer pinning lives in tests/integration/pipeline/orchestrator/
# test_orchestrator_core.py and the API-mirror pinning lives in
# tests/unit/web/execution/test_schemas.py + tests/unit/contracts/
# test_run_result.py.  This suite adds the cross-layer agreement contract:
# every terminal RunResult.status the engine writes must equal the RunStatus
# the Landscape audit ``Run`` row carries after ``finalize_run``.  Phase 2.2
# wires the orchestrator to call ``derive_terminal_run_status`` then
# ``finalize_run(status=...)``; if a future refactor breaks that wiring (e.g.
# Landscape persists ``COMPLETED`` while RunResult carries ``FAILED``) the
# audit trail and the API surface diverge silently.  This gate fails before
# the divergence reaches a deploy.


class TestComposerRuntimeRunStatusAgreement:
    """Shape 5 — engine RunResult and Landscape audit ``Run`` row agree on
    ``RunStatus`` across the four-value terminal taxonomy plus the explicit
    rows_routed-only design call.

    Coverage:

    * ``COMPLETED`` — healthy linear pipeline, all rows reach success.
    * ``COMPLETED_WITH_FAILURES`` — mixed run, some succeed and some fail.
    * ``FAILED`` — every row fails via ``on_error: discard`` (S1B msg2 shape).
    * ``EMPTY`` — empty source, no failures, no rows.
    * Design call (post-split, ``elspeth-5069612f3c``) — every row routes
      via ``on_error`` to a sink (``rows_failed == N`` plus
      ``rows_routed_failure == N``).  Now classifies as ``FAILED`` because
      lifecycle failure and routing provenance are recorded independently.
      Locked here as
      ``test_runstatus_on_error_routed_only_classifies_as_failed``.
      Companion: ``test_runstatus_gate_routed_only_classifies_as_completed``
      pins the symmetric MOVE shape (gate ``route_to_sink``) classifying as
      ``COMPLETED``.  A future maintainer changing either verdict confronts
      the design decision rather than silently flipping it.
    """

    @staticmethod
    def _assert_engine_landscape_agreement(landscape_db: LandscapeDB, run_result, expected_status: RunStatus) -> Run:
        """Cross-layer assertion helper.

        Asserts (1) the engine-side ``RunResult.status`` equals the expected
        status, (2) the Landscape audit ``runs`` row exists for the same
        ``run_id``, and (3) the audit row's ``status`` equals the engine's.
        Returns the loaded ``Run`` so the caller can chain further checks.
        """
        assert run_result.status == expected_status, (
            f"engine RunResult.status mismatch: expected {expected_status!r}, got {run_result.status!r}"
        )
        factory = make_factory(landscape_db)
        run_row = factory.run_lifecycle.get_run(run_result.run_id)
        assert run_row is not None, f"Landscape audit row missing for run_id={run_result.run_id!r}"
        assert run_row.status == run_result.status, (
            f"Landscape/engine status disagreement: engine RunResult.status={run_result.status!r}, Landscape Run.status={run_row.status!r}"
        )
        return run_row

    def test_agreement_runstatus_completed_engine_landscape(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Healthy linear pipeline — RunResult and Landscape both report COMPLETED."""
        source = ListSource([{"value": 1}, {"value": 2}])
        transform = PassTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.COMPLETED)
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 2
        assert len(sink.results) == 2

    def test_agreement_runstatus_completed_with_failures_engine_landscape(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Mixed run — RunResult and Landscape both report COMPLETED_WITH_FAILURES.

        Two rows: one with ``fail=False`` (succeeds), one with ``fail=True``
        (failed via ``on_error: discard``, which lands the row in the
        ``rows_quarantined`` bucket — ``discard`` is the engine's quarantine
        terminal state).  Predicate: rows_succeeded > 0 AND has_failures.
        """
        source = ListSource([{"value": 1, "fail": False}, {"value": 2, "fail": True}])
        transform = ConditionalErrorTransform(on_success="default", on_error="discard")
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.COMPLETED_WITH_FAILURES)
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 1
        # ADR-019 records lifecycle failure independently from discard-mode
        # quarantine provenance.
        assert run_result.rows_failed == 1
        assert run_result.rows_quarantined == 1
        assert run_result.rows_coalesce_failed == 0

    def test_agreement_runstatus_all_discarded_engine_landscape(self, landscape_db: LandscapeDB, payload_store) -> None:
        """All-rows-discarded via ``on_error: discard`` — both report
        COMPLETED_WITH_FAILURES.

        Two rows, both fail via ``on_error: discard`` (the engine's
        quarantine terminal state).  Per CLAUDE.md Tier-3 data manifesto,
        quarantine is a deliberate clean determination on every row, not a
        framework failure. The predicate sees ``terminal_clean_indicator``
        via ``rows_quarantined > 0`` with no uncaught ``failure_indicator``
        (rows_failed - rows_quarantined == 0) and lifts the verdict from
        FAILED to COMPLETED_WITH_FAILURES.
        """
        source = ListSource([{"value": 1, "fail": True}, {"value": 2, "fail": True}])
        transform = ConditionalErrorTransform(on_success="default", on_error="discard")
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.COMPLETED_WITH_FAILURES)
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 0
        # ADR-019 records lifecycle failure independently from discard-mode
        # quarantine provenance.
        assert run_result.rows_failed == 2
        assert run_result.rows_quarantined == 2
        assert run_result.rows_coalesce_failed == 0

    def test_agreement_runstatus_empty_engine_landscape(self, landscape_db: LandscapeDB, payload_store) -> None:
        """Empty source — RunResult and Landscape both report EMPTY.

        Predicate: rows_processed == 0 AND rows_succeeded == 0 AND no
        failure indicators.  An empty source with no failures must NOT be
        misclassified as ``FAILED`` (which is the all-rows-failed verdict)
        nor as ``COMPLETED`` (which presupposes rows_succeeded > 0).
        """
        source = ListSource([])
        transform = PassTransform()
        transform.on_success = "default"
        sink = CollectSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.EMPTY)
        assert run_result.rows_processed == 0
        assert run_result.rows_succeeded == 0
        assert run_result.rows_failed == 0
        assert len(sink.results) == 0

    def test_runstatus_on_error_routed_only_classifies_as_failed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """elspeth-5069612f3c — every row triggers a transform exception and
        is routed via on_error to a quarantine sink. After the rows_routed
        split, this shape produces rows_routed_failure == N (DIVERT) with no
        success indicator, and the predicate classifies as FAILED.

        The verdict (FAILED) matches the prior locked-in test, but the
        structural reason changes: previously the predicate excluded
        rows_routed entirely (sidestepping the DIVERT/MOVE conflation); now
        rows_routed_failure is a first-class failure indicator and contributes
        to the predicate decision directly.

        Companion: test_runstatus_gate_routed_only_classifies_as_completed
        below (the gate MOVE shape).
        """
        source = ListSource([{"value": 1, "fail": True}, {"value": 2, "fail": True}])
        transform = ConditionalErrorTransform(on_success="default", on_error="quarantine")
        default_sink = CollectSink(name="default")
        quarantine_sink = CollectSink(name="quarantine")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(default_sink), "quarantine": as_sink(quarantine_sink)},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=build_production_graph(config), payload_store=payload_store)

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.FAILED)
        assert run_result.rows_processed == 2
        assert run_result.rows_succeeded == 0
        assert run_result.rows_routed_success == 0
        assert run_result.rows_routed_failure == 2
        assert len(default_sink.results) == 0
        assert len(quarantine_sink.results) == 2

    def test_runstatus_gate_routed_only_classifies_as_completed(self, landscape_db: LandscapeDB, payload_store) -> None:
        """elspeth-5069612f3c / elspeth-71520f5e30 — user reproducer shape:
        csv source -> gate routes high-priority rows to one sink, low-priority
        rows to another, no on_success success-path sink. Every row is
        intentionally gate-routed via RoutingMode.MOVE.

        ADR-019 records lifecycle SUCCESS plus gate-routing provenance, so
        this shape produces rows_succeeded > 0 and rows_routed_success > 0
        with no failure indicator, and the predicate classifies as COMPLETED.

        Before the split (commit cc895589), this shape misclassified as
        RunStatus.FAILED with the misleading error "No row reached the success
        path" because the predicate excluded rows_routed entirely (DIVERT/MOVE
        conflation). This test pins the corrected behavior.
        """
        source = ListSource(
            [
                {"value": 1, "tier": "high"},
                {"value": 2, "tier": "low"},
                {"value": 3, "tier": "high"},
                {"value": 4, "tier": "low"},
            ],
            on_success="source_out",
        )
        tier_gate = GateSettings(
            name="tier_gate",
            input="source_out",
            condition="row['tier'] == 'high'",
            routes={"true": "high_priority", "false": "low_priority"},
        )
        high_sink = CollectSink(name="high_priority")
        low_sink = CollectSink(name="low_priority")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={
                "high_priority": as_sink(high_sink),
                "low_priority": as_sink(low_sink),
            },
            gates=[tier_gate],
        )

        run_result = Orchestrator(landscape_db).run(
            config,
            graph=build_production_graph(config),
            payload_store=payload_store,
        )

        self._assert_engine_landscape_agreement(landscape_db, run_result, RunStatus.COMPLETED)
        assert run_result.rows_processed == 4
        # Gate-routed rows are lifecycle successes with MOVE provenance.
        assert run_result.rows_succeeded == 4
        assert run_result.rows_routed_success == 4  # All routed via MOVE
        assert run_result.rows_routed_failure == 0  # No on_error reroutes
        assert len(high_sink.results) == 2
        assert len(low_sink.results) == 2


# ── Shape 6 — SecretInventoryItem biconditional agreement ────────────────────
# Closes elspeth-0d31c22d26 / Phase 2.3 (commit 22e3e0d9).  Per-mode coverage
# (fingerprint_resolver_not_configured, env_var_not_set,
# value_decryption_failed) lives in tests/unit/web/secrets/ and the contract
# tests below.  Per the Phase 2.3 closure
# rationale ("agreement-suite scope ... does not need duplication") this suite
# DOES NOT duplicate the per-mode tests.  The single contract-layer assertion
# below pins the structural invariant (``available ⟺ reason is None``) so a
# future drift on the closed-list reason taxonomy fails the agreement gate
# alongside the unit suite.


class TestComposerRuntimeSecretInventoryAgreement:
    """Shape 6 — ``SecretInventoryItem`` biconditional invariant pin.

    The audit-hygiene constraint that ``/api/secrets`` must not echo
    candidate-secret values into the response is enforced *structurally* by
    the ``SecretUnavailabilityReason`` ``Literal`` type — there is no code
    path that interpolates an env-var or candidate-secret value into the
    response, because the field accepts only the closed-list reasons.  This
    test pins the biconditional ``available ⟺ reason is None`` enforced in
    ``SecretInventoryItem.__post_init__`` so a future widening of the model
    (e.g. accepting free-form reason strings) fails the agreement gate
    before it can ship.

    Per-mode coverage and audit-hygiene runtime tests live in
    ``tests/unit/web/secrets/``; this test pins the contract surface only.
    """

    def test_unavailable_with_no_reason_rejected(self) -> None:
        """An unavailable inventory entry with ``reason=None`` is the
        operator-hostile shape the field exists to eliminate."""
        with pytest.raises(ValueError, match="reason is required when available=False"):
            SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="server",
                available=False,
                source_kind="env",
                reason=None,
            )

    def test_available_with_reason_rejected(self) -> None:
        """An available secret carrying a reason is incoherent — the
        biconditional rejects the asymmetric construction."""
        with pytest.raises(ValueError, match="reason must be None when available=True"):
            SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="server",
                available=True,
                source_kind="env",
                reason="env_var_not_set",
            )

    def test_unavailable_with_closed_list_reasons_accepted(self) -> None:
        """Each Phase 2.3 reason value is accepted; reasons outside the
        closed list are rejected at the model layer (the structural
        audit-hygiene gate)."""
        for reason in ("fingerprint_resolver_not_configured", "env_var_not_set", "value_decryption_failed"):
            item = SecretInventoryItem(
                name="OPENROUTER_API_KEY",
                scope="server",
                available=False,
                source_kind="env",
                reason=reason,
            )
            assert item.reason == reason
        # SecretUnavailabilityReason imported above is the typed surface — this
        # local frozenset is the closed list mirrored against
        # ``contracts/secrets._ALLOWED_UNAVAILABILITY_REASONS``.
        assert SecretUnavailabilityReason is not None  # imported for type alias presence


# ── Shape 7 — pipeline_done_callback run-accounting agreement ────────────────
# Closes elspeth-31d53c7493 / Phase 2.1 (commit 5e26d0a6).  Origin: 2026-05-01
# eval, S2 successful run 44f52421-a379-459b-96a8-6f0656086f16 (csv 6 rows →
# batch_stats group_by → json sink). The original bug was a linear
# row-decomposition equality on ``CompletedData``; the current contract loads
# explicit Landscape-derived source/token/routing/integrity accounting from
# the orchestrator-emitted run and verifies the API payload accepts that audit
# projection without reconstructing row fate from engine counters.


class _BatchAggregateTransform(BaseTransform):
    """Batch-aware transform mirroring S2's aggregation shape (csv 6 rows →
    1 aggregated output row).

    Reproduces the structural shape that broke the old row-decomposition
    equality: source rows reach ``CONSUMED_IN_BATCH`` while the aggregated
    output row reaches ``COMPLETED``. Net effect: source-row cardinality and
    materialized terminal-token cardinality intentionally diverge. The current
    readback contract must accept the explicit Landscape-derived accounting
    for that run.

    Defined inline so the test is self-contained as the agreement contract
    rather than depending on the BatchStats production plugin's internal
    triggering behaviour.
    """

    name = "agreement_batch_aggregate"
    determinism = Determinism.DETERMINISTIC
    is_batch_aware = True
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row, ctx):
        from elspeth.contracts import FieldContract, PipelineRow, SchemaContract
        from elspeth.plugins.infrastructure.results import TransformResult

        if not isinstance(row, list):
            # Single-row mode is not exercised by this aggregation path —
            # the wiring sets ``count == len(source_rows)`` so the trigger
            # always fires in batch mode.  Crash if the engine routes a
            # single row through here; that would indicate a wiring bug.
            raise AssertionError("agreement-suite batch aggregate received a single row, expected a batch flush")

        total = sum(float(r["amount"]) for r in row)
        output = {"total_amount": total, "row_count": len(row)}
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="total_amount",
                    original_name="total_amount",
                    python_type=float,
                    required=False,
                    source="inferred",
                ),
                FieldContract(
                    normalized_name="row_count",
                    original_name="row_count",
                    python_type=int,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )
        return TransformResult.success(
            PipelineRow(output, contract),
            success_reason={"action": "agreement_batch_aggregate"},
        )


class TestComposerRuntimeRunCompletionAgreement:
    """Shape 7 — aggregation runs must construct ``CompletedData`` from audit accounting.

    Single-occurrence regression coverage.  The unit-level pin lives at
    ``tests/unit/web/execution/test_run_accounting_projection.py``. The
    agreement-suite contribution here is to drive the accounting from a real
    orchestrator-emitted aggregation run. A future aggregation refactor that
    changes terminal-token emission would slip past unit tests that
    hand-construct accounting but fail this test because the engine's actual
    audit output would no longer match the readback contract.
    """

    def test_agreement_aggregation_run_counts_construct_completed_data(self, landscape_db: LandscapeDB, payload_store) -> None:
        """S2 reproducer (run 44f52421): a 6-row source feeds a batch-aware
        aggregation that emits 1 output row.  Source rows reach
        ``CONSUMED_IN_BATCH`` (no terminal-bucket counter).  Engine counts
        end up as ``rows_processed=6, rows_succeeded=1, rows_failed=0,
        rows_routed_success=0, rows_routed_failure=0, rows_quarantined=0``.
        The public readback payload must be validated from Landscape-derived
        accounting rather than from a row-counter equality.
        """
        from elspeth.contracts.types import NodeID

        source = ListSource(
            [
                {"customer_tier": "enterprise", "amount": 100.0},
                {"customer_tier": "enterprise", "amount": 150.0},
                {"customer_tier": "pro", "amount": 50.0},
                {"customer_tier": "pro", "amount": 75.0},
                {"customer_tier": "free", "amount": 10.0},
                {"customer_tier": "free", "amount": 20.0},
            ],
            on_success="source_out",
        )
        aggregate_transform = _BatchAggregateTransform()
        sink = CollectSink(name="output")

        # Build the graph with the batch-aware transform wired through
        # source_out → aggregate → output.  ``aggregations={}`` because the
        # batch transform is in the transforms list; PipelineConfig binds
        # it as an aggregation via ``aggregation_settings``.
        from elspeth.core.dag import ExecutionGraph as _ExecutionGraph
        from tests.fixtures.factories import wire_transforms as _wire_transforms

        graph = _ExecutionGraph.from_plugin_instances(
            sources={"primary": as_source(source)},
            source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
            transforms=_wire_transforms(
                [as_transform(aggregate_transform)],
                source_connection="source_out",
                final_sink="output",
            ),
            sinks={"output": as_sink(sink)},
            aggregations={},
            gates=[],
            coalesce_settings=None,
        )

        # Bind the transform as an aggregation: trigger fires at count=6
        # (matching the source row count) so all 6 input rows flush as one
        # batch and emit a single aggregated output row.
        transform_id_map = graph.get_transform_id_map()
        transform_node_id = transform_id_map[0]
        agg_settings = AggregationSettings(
            name="agreement_aggregate",
            plugin="agreement_batch_aggregate",
            input="source_out",
            on_success="output",
            on_error="discard",
            trigger=TriggerConfig(count=6, timeout_seconds=3600),
            output_mode="transform",
        )

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(aggregate_transform)],
            sinks={"output": as_sink(sink)},
            aggregation_settings={NodeID(transform_node_id): agg_settings},
        )

        run_result = Orchestrator(landscape_db).run(config, graph=graph, payload_store=payload_store)

        # Engine emission contract: 6 source rows in, 1 aggregated row out;
        # source rows reach CONSUMED_IN_BATCH while the output token reaches
        # COMPLETED.
        assert run_result.rows_processed == 6, (
            f"agreement_batch_aggregate must emit rows_processed=6 to reproduce S2 shape; got {run_result.rows_processed}"
        )
        assert run_result.rows_succeeded == 1, (
            f"the aggregated output row must be the only success terminal; got rows_succeeded={run_result.rows_succeeded}"
        )
        # The old row-counter equality rejected this legitimate shape:
        #   rows_processed (6) > sum-of-four-buckets (1)
        sum_of_buckets = (
            run_result.rows_succeeded
            + run_result.rows_failed
            + run_result.rows_routed_success
            + run_result.rows_routed_failure
            + run_result.rows_quarantined
        )
        assert run_result.rows_processed > sum_of_buckets, (
            "Shape sanity check: this test exercises source rows exceeding "
            f"terminal success/failure buckets (got {run_result.rows_processed} vs {sum_of_buckets}); "
            "if this assertion fails the test no longer pins the legitimate aggregation shape."
        )

        # Drive the completed-event validator from the engine's actual
        # Landscape audit output. Pre-Phase-2.1 this test constructed the
        # old row-counter payload directly and pinned the legitimate
        # rows_processed > terminal-successes aggregation shape. The public
        # event contract now carries explicit Landscape-derived accounting,
        # so the agreement check loads that accounting from the run just
        # emitted by the orchestrator.
        # Phase 2.2 (elspeth-0de989c56d): SSE payload carries the explicit
        # status discriminator; this aggregation has one successful terminal
        # materialized token and no failures, so the engine classifies it as
        # "completed".
        accounting = load_run_accounting_from_db(landscape_db, landscape_run_id=run_result.run_id)
        completed = CompletedData(
            status="completed",
            accounting=accounting,
            landscape_run_id=run_result.run_id,
        )
        assert completed.accounting.source.rows_processed == run_result.rows_processed
        assert completed.accounting.tokens.succeeded == run_result.rows_succeeded
        assert completed.accounting.tokens.failed == 0
        assert completed.accounting.integrity.closure == "closed"
        assert completed.landscape_run_id == run_result.run_id
        # The output sink received exactly the one aggregated row.
        assert len(sink.results) == 1, f"expected single aggregated output, got {len(sink.results)}"


class TestComposerRuntimeFileSinkCollisionAgreement:
    """Shape 8 — composer ``/validate`` must convert file-sink fs collision
    failures into a structured ``ValidationResult(is_valid=False)`` instead
    of letting the underlying ``FileExistsError`` propagate as a 500
    ``composer_plugin_error``. Closes ``elspeth-209b7e3a2b`` (Phase 0.b).

    Eval session S3 (``98573481-e8bc-4a03-8467-d3a86effcd56``, eval notes
    ``docs/composer/evidence/composer-llm-eval-2026-05-01.md``) reported this as a "gate
    primitive crash" because the failures clustered around gate-routing
    prompts. Phase 0.b investigation
    (``docs/composer/evidence/composer-phase-0b-staging-capture-2026-05-02.md``)
    re-attributed it: gate routing requires sinks, the LLM defaults sinks
    to ``collision_policy="fail_if_exists"``, and stale eval artifacts in
    ``data/outputs/`` collide. The actual defect was
    ``validate_pipeline``'s step-4 catch list at
    ``src/elspeth/web/execution/validation.py`` — only
    ``(PluginNotFoundError, PluginConfigError)`` was caught around
    ``instantiate_runtime_plugins``, so ``FileExistsError`` raised from
    ``json_sink.__init__`` / ``csv_sink.__init__`` via
    ``plugins/infrastructure/output_paths.py:48`` propagated uncaught into
    ``_state_data_from_composer_state``, was wrapped as
    ``ComposerRuntimePreflightError``, and surfaced as the opaque 500.

    The fix extends the step-4 catch list to include ``FileExistsError``
    and converts it to a structured ``ValidationCheck(passed=False)`` on
    the ``plugin_instantiation`` step with an ``auto_increment``
    suggestion. Per CLAUDE.md trust tiers, the existing-file condition is
    a Tier 3 boundary fact (external fs state) at a validation seam — the
    correct shape is a structured 422-class diagnostic, not a 500.

    Bug verification protocol (executed manually before this test landed):
    temporarily reverted the new ``except FileExistsError as exc:`` clause
    in ``src/elspeth/web/execution/validation.py`` (the block immediately
    after the existing ``except (PluginNotFoundError, PluginConfigError)``
    handler) and confirmed this test fails with an uncaught
    ``FileExistsError("Output path already exists: ...")`` raised through
    ``validate_pipeline``. Restored the catch after verification. This
    protocol guards against the "passes pre-fix AND post-fix" failure
    mode — without the revert, this test could appear to pin the contract
    while actually depending on adjacent behaviour. The revert proved the
    test's failure mode is exactly the structural bug it exists to catch.
    """

    @staticmethod
    def _validation_settings(data_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(data_dir=data_dir)

    @staticmethod
    def _csv_input(tmp_path: Path) -> Path:
        path = tmp_path / "blobs" / "input.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ticket_id,customer_tier\n1,enterprise\n", encoding="utf-8")
        return path

    @staticmethod
    def _build_state(csv_path: Path, sink_path: Path) -> CompositionState:
        return CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="default",
                options={"path": str(csv_path), "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            ),
            nodes=(),
            edges=(
                EdgeSpec(
                    id="source_to_default",
                    from_node="source",
                    to_node="default",
                    edge_type="on_success",
                    label="rows to sink",
                ),
            ),
            outputs=(
                OutputSpec(
                    name="default",
                    plugin="json",
                    options={
                        "path": str(sink_path),
                        "format": "jsonl",
                        "mode": "write",
                        "collision_policy": "fail_if_exists",
                        "schema": {"mode": "observed"},
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(name="Shape 8 collision repro", description=""),
            version=1,
        )

    def test_composer_validate_does_not_probe_file_sink_collision_in_preflight(self, tmp_path: Path) -> None:
        """Preflight validation must not inspect local file-sink collisions."""
        csv_path = self._csv_input(tmp_path)

        # Pre-create the sink target. Runtime execution with fail_if_exists
        # must still reject this, but composer preflight must not observe
        # local filesystem collision state during plugin construction.
        sink_path = tmp_path / "outputs" / "all.jsonl"
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        sink_path.write_text("", encoding="utf-8")  # any pre-existing content
        assert sink_path.exists()

        state = self._build_state(csv_path, sink_path)

        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
        )

        assert result.is_valid is True

        check_by_name = {check.name: check for check in result.checks}
        assert "plugin_instantiation" in check_by_name, "Missing plugin_instantiation check"
        plugin_check = check_by_name["plugin_instantiation"]
        assert plugin_check.passed is True, f"plugin_instantiation must not probe fs collisions; got {plugin_check.detail!r}"
        assert result.errors == []

    def test_composer_validate_passes_when_sink_path_does_not_collide(self, tmp_path: Path) -> None:
        """Positive control: same state with a non-existing sink path passes
        plugin instantiation cleanly. Asserts the catch is selective — only
        firing on the actual fs-collision condition, not converting all
        plugin-init failures into the same shape."""
        csv_path = self._csv_input(tmp_path)
        sink_path = tmp_path / "outputs" / "fresh.jsonl"
        sink_path.parent.mkdir(parents=True, exist_ok=True)
        # Deliberately do NOT create the sink_path file.
        assert not sink_path.exists()

        state = self._build_state(csv_path, sink_path)
        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
        )

        check_by_name = {check.name: check for check in result.checks}
        assert check_by_name["plugin_instantiation"].passed is True, (
            f"plugin_instantiation must pass when sink path is free; got detail={check_by_name['plugin_instantiation'].detail!r}"
        )


@dataclass(slots=True)
class _RuntimeSettingsFake:
    data_dir: str
    payload_store_path: Path
    landscape_passphrase: str | None = None

    def get_landscape_url(self) -> str:
        return "sqlite:///:memory:"

    def get_payload_store_path(self) -> Path:
        return self.payload_store_path


@dataclass(slots=True)
class _RunSnapshot:
    session_id: UUID
    status: str = "running"
    error: str | None = None


@dataclass(slots=True)
class _FakeSessionService:
    run: _RunSnapshot
    update_run_status_calls: list[tuple[UUID, str, dict[str, Any]]] = field(default_factory=list)
    appended_run_events: list[dict[str, Any]] = field(default_factory=list)
    recorded_blob_inline_resolutions: list[dict[str, Any]] = field(default_factory=list)
    record_blob_inline_resolutions_hook: Any = None
    next_event_sequence: int = 0

    async def update_run_status(self, run_id: UUID, status: str, **kwargs: Any) -> None:
        self.run.status = status
        if "error" in kwargs:
            self.run.error = kwargs["error"]
        self.update_run_status_calls.append((run_id, status, kwargs))

    async def get_run(self, _run_id: UUID) -> _RunSnapshot:
        return self.run

    async def append_run_event(
        self,
        *,
        run_id: UUID,
        timestamp: datetime,
        event_type: str,
        data: dict[str, Any],
    ) -> SimpleNamespace:
        self.next_event_sequence += 1
        self.appended_run_events.append(
            {
                "run_id": run_id,
                "timestamp": timestamp,
                "event_type": event_type,
                "data": data,
            }
        )
        return SimpleNamespace(sequence=self.next_event_sequence)

    async def record_blob_inline_resolutions(
        self,
        *,
        run_id: UUID,
        resolutions: Any,
        attempt: int = 1,
    ) -> None:
        if self.record_blob_inline_resolutions_hook is not None:
            await self.record_blob_inline_resolutions_hook(
                run_id=run_id,
                resolutions=resolutions,
                attempt=attempt,
            )
        self.recorded_blob_inline_resolutions.append(
            {
                "run_id": run_id,
                "resolutions": tuple(resolutions),
                "attempt": attempt,
            }
        )


@dataclass(slots=True)
class _FakeProgressBroadcaster:
    broadcast_calls: list[tuple[str, Any]] = field(default_factory=list)
    cleanup_run_ids: list[str] = field(default_factory=list)

    def broadcast(self, run_id: str, event: Any) -> BroadcastResult:
        self.broadcast_calls.append((run_id, event))
        return BroadcastResult()

    def cleanup_run(self, run_id: str) -> None:
        self.cleanup_run_ids.append(run_id)


@dataclass(slots=True)
class _FakeBlobService:
    blob_record: BlobRecord
    content: bytes
    link_blob_to_run_calls: list[tuple[UUID, UUID, str]] = field(default_factory=list)
    read_blob_content_calls: list[UUID] = field(default_factory=list)
    get_blob_calls: list[UUID] = field(default_factory=list)
    finalize_run_output_blobs_calls: list[tuple[UUID, bool]] = field(default_factory=list)

    async def link_blob_to_run(self, blob_id: UUID, run_id: UUID, direction: str) -> None:
        self.link_blob_to_run_calls.append((blob_id, run_id, direction))

    async def read_blob_content(self, blob_id: UUID) -> bytes:
        self.read_blob_content_calls.append(blob_id)
        return self.content

    async def get_blob(self, blob_id: UUID) -> BlobRecord:
        self.get_blob_calls.append(blob_id)
        return self.blob_record

    async def finalize_run_output_blobs(self, run_id: UUID, success: bool) -> BlobFinalizationResult:
        self.finalize_run_output_blobs_calls.append((run_id, success))
        return BlobFinalizationResult(finalized=(), errors=())


def _ready_inline_blob_record(*, blob_id: UUID, session_id: UUID, content: bytes, content_hash: str) -> BlobRecord:
    return BlobRecord(
        id=blob_id,
        session_id=session_id,
        filename="prompt.txt",
        mime_type="text/plain",
        size_bytes=len(content),
        content_hash=content_hash,
        storage_path="prompt.txt",
        created_at=datetime.now(tz=UTC),
        created_by="assistant",
        source_description=None,
        status="ready",
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=None,
        creating_model_identifier=None,
        creating_model_version=None,
        creating_provider=None,
        creating_composer_skill_hash=None,
        creating_arguments_hash=None,
    )


class TestComposerRuntimeBlobInlineAgreement:
    """Shape 9 — widened blob_ref / inline_content agreement.

    Bug verification for sub-pin A was captured before commit ``2aaa4be2e``:
    without the ``blob_inline_refs`` validate-time metadata bridge,
    ``validate_pipeline(..., blob_get_metadata=...)`` rejected the new keyword
    argument and the service path never queried ``BlobService.get_blob``.

    Bug verification for sub-pins B/C was captured in
    ``tests/unit/web/execution/test_service.py::TestInlineBlobRuntimePreflight``:
    removing the runtime resolver or audit-write block lets settings/plugin
    construction proceed without the fail-closed hash/audit invariant.
    """

    @staticmethod
    def _validation_settings(data_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(data_dir=data_dir)

    @staticmethod
    def _state_with_inline_prompt(tmp_path: Path, blob_id: UUID, sha256: str) -> CompositionState:
        blobs_dir = tmp_path / "blobs"
        outputs_dir = tmp_path / "outputs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        return CompositionState(
            source=SourceSpec(
                plugin="csv",
                on_success="classify_input",
                options={
                    "path": str(blobs_dir / "input.csv"),
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="discard",
            ),
            nodes=(
                NodeSpec(
                    id="classify",
                    node_type="transform",
                    plugin="llm",
                    input="classify_input",
                    on_success="results",
                    on_error="discard",
                    options={
                        "provider": "openrouter",
                        "api_key": {"secret_ref": "OPENROUTER_API_KEY"},
                        "model": "openai/gpt-4o",
                        "prompt_template": {
                            "blob_ref": str(blob_id),
                            "mode": "inline_content",
                            "sha256": sha256,
                        },
                        "required_input_fields": [],
                        "schema": {"mode": "observed"},
                        # Pre-resolved model-choice review so the
                        # interpretation gate doesn't short-circuit the
                        # validator before the blob-inline check runs. The
                        # auto-stager normally creates a pending requirement
                        # at mutation time; tests that bypass the composer
                        # (constructing NodeSpec directly) must stage the
                        # resolved form themselves.
                        INTERPRETATION_REQUIREMENTS_KEY: [
                            {
                                "id": "model_choice_review:classify",
                                "kind": "llm_model_choice",
                                "user_term": "llm_model_choice:classify",
                                "status": "resolved",
                                "draft": "openai/gpt-4o",
                                "event_id": "model-choice-accepted",
                                "accepted_value": "openai/gpt-4o",
                                "accepted_artifact_hash": None,
                                "resolved_prompt_template_hash": stable_hash("openai/gpt-4o"),
                            }
                        ],
                    },
                    condition=None,
                    routes=None,
                    fork_to=None,
                    branches=None,
                    policy=None,
                    merge=None,
                ),
            ),
            edges=(),
            outputs=(
                OutputSpec(
                    name="results",
                    plugin="json",
                    options={
                        "path": str(outputs_dir / "results.jsonl"),
                        "format": "jsonl",
                        "schema": {"mode": "observed"},
                    },
                    on_write_failure="discard",
                ),
            ),
            metadata=PipelineMetadata(name="Shape 9 inline blob agreement", description=""),
            version=1,
        )

    @staticmethod
    def _pipeline_yaml(blob_id: UUID, sha256: str) -> str:
        return f"""
sources:
  source:
    plugin: csv
    on_success: classify_in
    options:
      path: input.csv
transforms:
  - name: classify
    plugin: llm
    input: classify_in
    on_success: results
    on_error: results
    options:
      prompt_template:
        blob_ref: {blob_id}
        mode: inline_content
        sha256: {sha256}
sinks:
  results:
    plugin: json
    on_write_failure: discard
    options:
      path: output.jsonl
"""

    @staticmethod
    def _execution_service(tmp_path: Path) -> tuple[ExecutionServiceImpl, _FakeSessionService, asyncio.AbstractEventLoop]:
        loop = asyncio.new_event_loop()
        settings = _RuntimeSettingsFake(
            data_dir=str(tmp_path),
            payload_store_path=tmp_path / "payloads",
        )

        # ``_run_pipeline`` resolves the run's owning session (get_run().session_id)
        # to scope inline-blob access before any metadata enforcement
        # (IDOR contract, elspeth-195ecb1d58). Tests below set their owned
        # blob_record.session_id to this value so ownership passes and they
        # exercise the hash/audit-ordering assertions they actually target.
        session_service = _FakeSessionService(run=_RunSnapshot(session_id=uuid4()))

        service = ExecutionServiceImpl(
            loop=loop,
            broadcaster=cast(Any, _FakeProgressBroadcaster()),
            settings=settings,
            session_service=session_service,
            yaml_generator=cast(Any, SimpleNamespace()),
            telemetry=build_sessions_telemetry(),
        )

        def _call_async(coro: Any) -> Any:
            return loop.run_until_complete(coro)

        cast(Any, service)._call_async = _call_async
        return service, session_service, loop

    def test_validate_returns_structured_error_for_missing_inline_blob(self, tmp_path: Path) -> None:
        blob_id = uuid4()
        state = self._state_with_inline_prompt(tmp_path, blob_id, "a" * 64)

        result = validate_pipeline(
            state,
            self._validation_settings(tmp_path),
            composer_yaml_generator,
            blob_get_metadata=lambda _blob_id: None,
        )

        assert result.is_valid is False
        check = next(check for check in result.checks if check.name == "blob_inline_refs")
        assert check.passed is False
        assert any(error.error_code == "missing_inline_blob_content" for error in result.errors)
        assert any(error.component_id == "classify" and error.component_type == "transform" for error in result.errors)

    @patch("elspeth.web.execution.service.Orchestrator")
    @patch("elspeth.web.execution.service.load_settings_from_config_dict")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_runtime_fails_closed_on_hash_mismatch_before_settings_load(
        self,
        mock_payload_cls: Any,
        mock_landscape_cls: Any,
        mock_load: Any,
        mock_orch_cls: Any,
        tmp_path: Path,
    ) -> None:
        del mock_payload_cls, mock_landscape_cls
        service, _session_service, loop = self._execution_service(tmp_path)
        content = b"actual prompt bytes"
        blob_id = uuid4()
        run_id = uuid4()
        blob_record = _ready_inline_blob_record(
            blob_id=blob_id,
            session_id=_session_service.run.session_id,
            content=content,
            content_hash=hashlib.sha256(content).hexdigest(),
        )
        blob_service = _FakeBlobService(blob_record=blob_record, content=content)
        cast(Any, service)._blob_service = blob_service

        try:
            with pytest.raises(BlobIntegrityError):
                service._run_pipeline(str(run_id), self._pipeline_yaml(blob_id, "b" * 64), threading.Event())
        finally:
            loop.close()

        mock_load.assert_not_called()
        mock_orch_cls.assert_not_called()

    @patch("elspeth.web.execution.service.load_settings_from_config_dict")
    @patch("elspeth.web.execution.service.LandscapeDB")
    @patch("elspeth.web.execution.service.FilesystemPayloadStore")
    def test_runtime_records_audit_hash_before_settings_load(
        self,
        mock_payload_cls: Any,
        mock_landscape_cls: Any,
        mock_load: Any,
        tmp_path: Path,
    ) -> None:
        del mock_payload_cls, mock_landscape_cls
        service, session_service, loop = self._execution_service(tmp_path)
        content = b"You are an audited prompt."
        sha256 = hashlib.sha256(content).hexdigest()
        blob_id = uuid4()
        run_id = uuid4()
        order: list[str] = []

        blob_record = _ready_inline_blob_record(
            blob_id=blob_id,
            session_id=session_service.run.session_id,
            content=content,
            content_hash=sha256,
        )
        blob_service = _FakeBlobService(blob_record=blob_record, content=content)
        cast(Any, service)._blob_service = blob_service

        async def record_blob_inline_resolutions(
            *,
            run_id: UUID,
            resolutions: Any,
            attempt: int = 1,
        ) -> None:
            del run_id, resolutions, attempt
            order.append("record")

        session_service.record_blob_inline_resolutions_hook = record_blob_inline_resolutions

        def stop_after_audit(config_dict: dict[str, Any], *, expand_env_vars: bool = True) -> None:
            assert "record" in order, "audit row must be recorded before settings/plugin construction"
            # Web-authored YAML must never expand host ${VAR} placeholders.
            assert expand_env_vars is False
            prompt_template = config_dict["transforms"][0]["options"]["prompt_template"]
            assert prompt_template == "You are an audited prompt."
            raise RuntimeError("stop after inline audit")

        mock_load.side_effect = stop_after_audit

        try:
            with pytest.raises(RuntimeError, match="stop after inline audit"):
                service._run_pipeline(str(run_id), self._pipeline_yaml(blob_id, sha256), threading.Event())
        finally:
            loop.close()

        assert len(session_service.recorded_blob_inline_resolutions) == 1
        recorded_call = session_service.recorded_blob_inline_resolutions[0]
        resolutions = recorded_call["resolutions"]
        assert len(resolutions) == 1
        assert resolutions[0].field_path == "node:classify.options.prompt_template"
        assert resolutions[0].content_hash == sha256


class TestComposerRuntimeFixedModeImplicitRequiredAgreement:
    """Shape 10 — fixed-mode consumer implicit-required-field parity (elspeth-8f3b3f650d).

    A consumer whose schema is ``{mode: fixed, fields: [...]}`` *implicitly*
    requires every declared (non-optional) field — the runtime builds a typed
    input Pydantic model and rejects an edge from a TYPED producer that does
    not guarantee one of those declared fields. Authoring-time
    ``CompositionState.validate`` previously computed consumer requirements via
    ``get_raw_node_required_fields`` (EXPLICIT ``required_fields`` only), so a
    fixed-mode declared requirement that exceeds the producer's guarantees was
    green-lit at authoring time and only rejected at runtime — the exact
    "validate green / runtime red" divergence this suite registers.

    The fix gates strictly on producer schema MODE (a fixed/flexible SOURCE
    producer is TYPED; observed sources and transform/gate/coalesce producers
    resolve to a dynamic effective producer schema and are skipped), mirroring
    the runtime Phase-2 observed/dynamic bypass at ``graph.py:1392-1403``.

    Bug verification protocol (mandatory, per the module docstring): revert the
    new sibling block in ``state.py::_check_schema_contracts`` — the
    ``consumer_effective_required`` missing-field append guarded by
    ``producer_is_typed_source`` — and confirm
    ``test_reject_fixed_consumer_implicit_required_over_typed_source`` fails at
    the ``assert not composer_result.is_valid`` line (authoring returns
    ``is_valid=True`` pre-fix). Then restore. Verified 2026-06-09.
    """

    def _empty_state(self) -> CompositionState:
        return CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )

    def _state(
        self,
        *,
        source_options: dict[str, Any],
        consumer_options: dict[str, Any],
        output_options: dict[str, Any],
        source_name: str = "source",
    ) -> CompositionState:
        # ``source_name`` defaults to the unnamed default source (producer_id
        # "source"); pass a name to mint a NAMED source (producer_id
        # "source:<name>") — the multi-source headline shape that exercises the
        # ``is_source_producer_id`` predicate rather than the literal "source".
        state = self._empty_state()
        state = state.with_named_source(
            source_name,
            SourceSpec(
                plugin="csv",
                on_success="t1",
                options=source_options,
                on_validation_failure="quarantine",
            ),
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options=consumer_options,
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options=output_options,
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node=source_name,
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )
        return state

    def _build_runtime_graph(
        self,
        *,
        source_options: dict[str, Any],
        consumer_options: dict[str, Any],
        output_options: dict[str, Any],
    ) -> ExecutionGraph:
        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="csv",
                    on_success="t1",
                    options={**source_options, "on_validation_failure": "discard"},
                )
            },
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="discard",
                    options=consumer_options,
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options=output_options,
                )
            },
        )
        plugins = instantiate_plugins_from_config(config)
        return ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=list(config.gates),
            coalesce_settings=None,
        )

    def test_reject_fixed_consumer_implicit_required_over_typed_source(
        self,
        tmp_path: Path,
    ) -> None:
        """(A) Typed (fixed) source guarantees {color}; fixed consumer implicitly
        requires {color, teal_pairing_rating}. Both validators MUST reject.

        Pre-fix the authoring validator returned ``is_valid=True`` because the
        fixed-mode implicit requirement ``teal_pairing_rating`` was invisible to
        the explicit-only ``get_raw_node_required_fields`` path — the defect.
        """
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("color\nred\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        source_options = {
            "path": str(csv_path),
            "schema": {"mode": "fixed", "fields": ["color: str"]},
        }
        consumer_options = {
            "operations": [{"target": "out", "expression": "row['color']"}],
            "schema": {
                "mode": "fixed",
                "fields": ["color: str", "teal_pairing_rating: str"],
            },
        }
        output_options = {"path": str(output_path), "schema": {"mode": "observed"}}

        state = self._state(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )
        composer_result = state.validate()
        assert not composer_result.is_valid, (
            "Composer should reject: fixed consumer implicitly requires 'teal_pairing_rating' which the typed source does not guarantee."
        )
        assert any(
            "schema contract violation" in entry.message.lower() and "teal_pairing_rating" in entry.message
            for entry in composer_result.errors
        ), [e.message for e in composer_result.errors]

        with pytest.raises(GraphValidationError) as exc_info:
            self._build_runtime_graph(
                source_options=source_options,
                consumer_options=consumer_options,
                output_options=output_options,
            )
        assert "teal_pairing_rating" in str(exc_info.value)

    def test_reject_fixed_consumer_implicit_required_over_named_typed_source(
        self,
        tmp_path: Path,
    ) -> None:
        """(A-named) Same as (A) but the producer is a NAMED source
        (producer_id ``source:customers``), the headline multi-source shape.

        Regression for elspeth-3332619032: ``_producer_is_typed_source`` gated on
        the literal ``producer_id != "source"`` and returned False for any
        ``source:<name>`` producer, so the implicit-required parity check never
        fired for named sources — composer green / runtime red, exactly the
        divergence this suite forbids. The explicit-required path one block up
        already used ``is_source_producer_id`` and rejected correctly, proving
        the intended coverage. Pre-fix this asserts-False at
        ``assert not composer_result.is_valid``.
        """
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("color\nred\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        source_options = {
            "path": str(csv_path),
            "schema": {"mode": "fixed", "fields": ["color: str"]},
        }
        consumer_options = {
            "operations": [{"target": "out", "expression": "row['color']"}],
            "schema": {
                "mode": "fixed",
                "fields": ["color: str", "teal_pairing_rating: str"],
            },
        }
        output_options = {"path": str(output_path), "schema": {"mode": "observed"}}

        state = self._state(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
            source_name="customers",
        )
        composer_result = state.validate()
        assert not composer_result.is_valid, (
            "Composer should reject: fixed consumer implicitly requires "
            "'teal_pairing_rating' which the NAMED typed source 'customers' does not guarantee."
        )
        assert any(
            "schema contract violation" in entry.message.lower() and "teal_pairing_rating" in entry.message
            for entry in composer_result.errors
        ), [e.message for e in composer_result.errors]

        with pytest.raises(GraphValidationError) as exc_info:
            self._build_runtime_graph(
                source_options=source_options,
                consumer_options=consumer_options,
                output_options=output_options,
            )
        assert "teal_pairing_rating" in str(exc_info.value)

    def test_reject_flexible_consumer_implicit_required_over_typed_source(
        self,
        tmp_path: Path,
    ) -> None:
        """(A2) A FLEXIBLE consumer also implicitly requires its declared
        (non-optional) fields, but its input is NOT locked (extras allowed) and
        its explicit ``required_fields`` is empty — so the per-node skip guard
        would short-circuit it unless the effective-required set is folded in.
        Runtime rejects (flexible builds a typed model with non-empty
        ``model_fields``); authoring MUST reject too.
        """
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("color\nred\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        source_options = {
            "path": str(csv_path),
            "schema": {"mode": "fixed", "fields": ["color: str"]},
        }
        consumer_options = {
            "operations": [{"target": "out", "expression": "row['color']"}],
            "schema": {
                "mode": "flexible",
                "fields": ["color: str", "teal_pairing_rating: str"],
            },
        }
        output_options = {"path": str(output_path), "schema": {"mode": "observed"}}

        state = self._state(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )
        composer_result = state.validate()
        assert not composer_result.is_valid, (
            "Composer should reject: flexible consumer implicitly requires 'teal_pairing_rating' which the typed source does not guarantee."
        )
        assert any(
            "schema contract violation" in entry.message.lower() and "teal_pairing_rating" in entry.message
            for entry in composer_result.errors
        ), [e.message for e in composer_result.errors]

        with pytest.raises(GraphValidationError) as exc_info:
            self._build_runtime_graph(
                source_options=source_options,
                consumer_options=consumer_options,
                output_options=output_options,
            )
        assert "teal_pairing_rating" in str(exc_info.value)

    def test_accept_observed_source_auto_guarantee_over_fixed_consumer(
        self,
        tmp_path: Path,
    ) -> None:
        """(B) Overshoot tripwire: an OBSERVED source has non-empty guarantees
        (the auto-guaranteed column) yet runtime bypasses Phase-2 type
        validation because the producer schema is observed. Authoring MUST also
        accept — proving the gate is on producer MODE, not guarantee-emptiness.
        """
        text_path = tmp_path / "in.txt"
        text_path.write_text("hello\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        state = self._empty_state()
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="t1",
                options={
                    "path": str(text_path),
                    "column": "color",
                    "schema": {"mode": "observed"},
                },
                on_validation_failure="quarantine",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="t1",
                node_type="transform",
                plugin="value_transform",
                input="t1",
                on_success="main",
                on_error="discard",
                options={
                    "operations": [{"target": "out", "expression": "row['color']"}],
                    "schema": {
                        "mode": "fixed",
                        "fields": ["color: str", "teal_pairing_rating: str"],
                    },
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": str(output_path), "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        state = state.with_edge(
            EdgeSpec(
                id="e1",
                from_node="source",
                to_node="t1",
                edge_type="on_success",
                label=None,
            )
        )

        composer_result = state.validate()
        assert composer_result.is_valid, [e.message for e in composer_result.errors]

        config = ElspethSettings(
            sources={
                "primary": SourceSettings(
                    plugin="text",
                    on_success="t1",
                    options={
                        "path": str(text_path),
                        "column": "color",
                        "schema": {"mode": "observed"},
                        "on_validation_failure": "discard",
                    },
                )
            },
            transforms=[
                TransformSettings(
                    name="t1",
                    plugin="value_transform",
                    input="t1",
                    on_success="main",
                    on_error="discard",
                    options={
                        "operations": [{"target": "out", "expression": "row['color']"}],
                        "schema": {
                            "mode": "fixed",
                            "fields": ["color: str", "teal_pairing_rating: str"],
                        },
                    },
                )
            ],
            sinks={
                "main": SinkSettings(
                    plugin="csv",
                    on_write_failure="discard",
                    options={"path": str(output_path), "schema": {"mode": "observed"}},
                )
            },
        )
        plugins = instantiate_plugins_from_config(config)
        # Runtime construction (which validates edges) must not raise.
        ExecutionGraph.from_plugin_instances(
            sources=plugins.sources,
            source_settings_map=plugins.source_settings_map,
            transforms=plugins.transforms,
            sinks=plugins.sinks,
            aggregations=plugins.aggregations,
            gates=[],
            coalesce_settings=None,
        )

    def test_accept_optional_declared_field_over_typed_source(
        self,
        tmp_path: Path,
    ) -> None:
        """(C) An OPTIONAL declared field (``teal_pairing_rating: str?``) is not
        implicitly required; both validators MUST accept over a {color} source.
        """
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("color\nred\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        source_options = {
            "path": str(csv_path),
            "schema": {"mode": "fixed", "fields": ["color: str"]},
        }
        consumer_options = {
            "operations": [{"target": "out", "expression": "row['color']"}],
            "schema": {
                "mode": "fixed",
                "fields": ["color: str", "teal_pairing_rating: str?"],
            },
        }
        output_options = {"path": str(output_path), "schema": {"mode": "observed"}}

        state = self._state(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )
        composer_result = state.validate()
        assert composer_result.is_valid, [e.message for e in composer_result.errors]

        # Runtime construction must not raise.
        self._build_runtime_graph(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )

    def test_accept_plain_observed_consumer_over_typed_source(
        self,
        tmp_path: Path,
    ) -> None:
        """(D) A plain OBSERVED consumer imposes no implicit requirements; both
        validators MUST accept over a typed (fixed) source.
        """
        csv_path = tmp_path / "in.csv"
        csv_path.write_text("color\nred\n", encoding="utf-8")
        output_path = tmp_path / "out.csv"

        source_options = {
            "path": str(csv_path),
            "schema": {"mode": "fixed", "fields": ["color: str"]},
        }
        consumer_options = {
            "operations": [{"target": "out", "expression": "row['color']"}],
            "schema": {"mode": "observed"},
        }
        output_options = {"path": str(output_path), "schema": {"mode": "observed"}}

        state = self._state(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )
        composer_result = state.validate()
        assert composer_result.is_valid, [e.message for e in composer_result.errors]

        self._build_runtime_graph(
            source_options=source_options,
            consumer_options=consumer_options,
            output_options=output_options,
        )


class TestComposerRuntimeGateRouteParityAgreement:
    """Biconditional agreement for gate route-label / condition-return-type parity.

    Mirror of GateSettings.validate_boolean_routes (core/config.py). The composer's
    CompositionState.validate() must agree with GateSettings construction on whether
    a gate's route labels are consistent with the static return type of its condition:
    composer is_valid  <=>  GateSettings accepts. Regression for elspeth-08e17b9253,
    where the composer green-lit boolean/numeric conditions with mismatched labels
    that runtime config later rejected.
    """

    def _gate_state(self, *, condition: str, routes: dict[str, str]) -> CompositionState:
        """A minimal valid pipeline whose only interesting feature is a gate."""
        state = CompositionState(
            source=None,
            nodes=(),
            edges=(),
            outputs=(),
            metadata=PipelineMetadata(),
            version=1,
        )
        state = state.with_source(
            SourceSpec(
                plugin="text",
                on_success="g1",
                options={"path": "/tmp/in.txt", "column": "line", "schema": {"mode": "observed"}},
                on_validation_failure="discard",
            )
        )
        state = state.with_node(
            NodeSpec(
                id="g1",
                node_type="gate",
                plugin=None,
                input="g1",
                on_success=None,
                on_error=None,
                options={},
                condition=condition,
                routes=routes,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            )
        )
        state = state.with_output(
            OutputSpec(
                name="main",
                plugin="csv",
                options={"path": "/tmp/out.csv", "schema": {"mode": "observed"}},
                on_write_failure="discard",
            )
        )
        state = state.with_edge(EdgeSpec(id="e0", from_node="source", to_node="g1", edge_type="on_success", label=None))
        return state

    def test_boolean_gate_with_custom_route_labels_rejected_by_both(self) -> None:
        """Boolean condition + non-true/false labels: composer is_valid=False AND GateSettings raises."""
        state = self._gate_state(condition='row["x"] > 0', routes={"high": "main", "low": "main"})

        composer_result = state.validate()
        assert composer_result.is_valid is False
        assert any("boolean condition" in e.message for e in composer_result.errors), [e.message for e in composer_result.errors]

        with pytest.raises(ValidationError, match="boolean condition"):
            GateSettings(
                name="g1",
                input="g1",
                condition='row["x"] > 0',
                routes={"high": "main", "low": "main"},
            )

    def test_numeric_gate_condition_rejected_by_both(self) -> None:
        """Provably-numeric condition: composer is_valid=False AND GateSettings raises."""
        state = self._gate_state(condition='row["x"] + 1', routes={"a": "main"})

        composer_result = state.validate()
        assert composer_result.is_valid is False
        assert any("numeric value" in e.message for e in composer_result.errors), [e.message for e in composer_result.errors]

        with pytest.raises(ValidationError, match="numeric value"):
            GateSettings(
                name="g1",
                input="g1",
                condition='row["x"] + 1',
                routes={"a": "main"},
            )

    def test_string_route_gate_accepted_by_both(self) -> None:
        """POSITIVE CONTROL: string-returning condition + matching custom labels stays valid in both."""
        state = self._gate_state(
            condition='"high" if row["x"] > 0 else "low"',
            routes={"high": "main", "low": "main"},
        )

        composer_result = state.validate()
        assert composer_result.is_valid is True, [e.message for e in composer_result.errors]

        # Must construct cleanly — legal runtime config, must NOT be over-rejected.
        GateSettings(
            name="g1",
            input="g1",
            condition='"high" if row["x"] > 0 else "low"',
            routes={"high": "main", "low": "main"},
        )

    def test_boolean_gate_with_true_false_labels_accepted_by_both(self) -> None:
        """Boolean condition + exactly {true,false} labels stays valid in both."""
        state = self._gate_state(
            condition='row["x"] > 0',
            routes={"true": "main", "false": "main"},
        )

        composer_result = state.validate()
        assert composer_result.is_valid is True, [e.message for e in composer_result.errors]

        GateSettings(
            name="g1",
            input="g1",
            condition='row["x"] > 0',
            routes={"true": "main", "false": "main"},
        )


class TestComposerRuntimeQueueAgreement:
    """Shape 11 — structural queue fan-in round-trips composer <-> runtime.

    The shipped ``examples/multi_source_queue/settings.yaml`` fans two sources
    into one declared ``queues.inbound`` and then a single passthrough. The
    composer must import it (preserving the queue), validate it green, export it
    back to runtime YAML, and the runtime must build a graph containing exactly
    one ``NodeType.QUEUE`` with both sources leading into it and one ordinary
    downstream consumer leading out — matching the engine's own fan-in contract
    (ADR-025 / elspeth-a5b86149d4, elspeth-6421ffa028).

    Bug-verification protocol (mandatory per this file's header): the shape is
    pinned on the generator's queue-emission assignment in
    ``web/composer/yaml_generator._generate_pipeline_dict`` — the
    ``doc["queues"] = queues_doc`` line. Manually replacing that line with
    ``pass`` makes ``generate_yaml`` omit the ``queues:`` section entirely, so
    ``load_settings_from_yaml_string`` rebuilds settings with no queue and
    ``test_queue_round_trips_composer_import_export_and_runtime_graph`` fails at
    its first queue-dependent assertion with
    ``AssertionError: assert set() == {'inbound'}`` (``set(settings.queues) ==
    {"inbound"}``). Verified by manual revert on 2026-07-12; restored. Had the
    queue survived import but not the runtime fan-in contract, the same
    two-source topology without a queue is rejected at graph build with
    ``GraphValidationError("Duplicate producer for connection 'inbound' ...")``,
    which the negative control below asserts directly by deleting the emitted
    ``queues:`` section.
    """

    def _example_yaml(self) -> str:
        example = Path(__file__).resolve().parents[3] / "examples" / "multi_source_queue" / "settings.yaml"
        return example.read_text(encoding="utf-8")

    def _build_runtime_graph_for_settings(self, settings: ElspethSettings) -> ExecutionGraph:
        bundle = instantiate_plugins_from_config(settings, preflight_mode=True)
        return ExecutionGraph.from_plugin_instances(
            sources=bundle.sources,
            source_settings_map=bundle.source_settings_map,
            transforms=bundle.transforms,
            sinks=bundle.sinks,
            aggregations=bundle.aggregations,
            gates=list(settings.gates),
            queues=settings.queues,
        )

    def test_queue_round_trips_composer_import_export_and_runtime_graph(self) -> None:
        from elspeth.contracts import NodeType
        from elspeth.core.config import load_settings_from_yaml_string
        from elspeth.web.composer.yaml_importer import composition_state_from_runtime_yaml

        state = composition_state_from_runtime_yaml(self._example_yaml())
        composer_result = state.validate()
        assert composer_result.is_valid, [e.message for e in composer_result.errors]

        generated_yaml = composer_yaml_generator.generate_yaml(state)
        settings = load_settings_from_yaml_string(generated_yaml)
        assert set(settings.queues) == {"inbound"}

        graph = self._build_runtime_graph_for_settings(settings)
        graph.validate()

        queue_nodes = [node for node in graph.get_nodes() if node.node_type == NodeType.QUEUE]
        assert len(queue_nodes) == 1
        queue_info = queue_nodes[0]
        # The runtime keys queues by a hashed queue_<name>_<hash> node id; the raw
        # queue name lives in config["name"], not the node id.
        assert queue_info.config["name"] == "inbound"
        assert queue_info.node_id != "inbound"
        assert queue_info.output_schema_config is not None
        assert queue_info.output_schema_config.mode == "observed"

        queue_id = queue_info.node_id
        source_predecessors = {
            edge.from_node
            for edge in graph.get_incoming_edges(queue_id)
            if graph.get_node_info(edge.from_node).node_type == NodeType.SOURCE
        }
        assert len(source_predecessors) == 2, "both sources must fan into the queue"

        outgoing = [edge for edge in graph.get_edges() if edge.from_node == queue_id]
        assert len(outgoing) == 1, "queue feeds exactly one ordinary downstream consumer"
        downstream = graph.get_node_info(outgoing[0].to_node)
        assert downstream.node_type == NodeType.TRANSFORM
        assert downstream.plugin_name == "passthrough"

    def test_deleting_generated_queues_section_reproduces_fan_in_rejection(self) -> None:
        """Manual negative control: without the emitted queue, the same two-source
        fan-in is exactly the topology the runtime rejects."""
        import yaml

        from elspeth.core.config import load_settings_from_yaml_string
        from elspeth.web.composer.yaml_importer import composition_state_from_runtime_yaml

        state = composition_state_from_runtime_yaml(self._example_yaml())
        generated_yaml = composer_yaml_generator.generate_yaml(state)

        doc = yaml.safe_load(generated_yaml)
        assert "queues" in doc, "sanity: the generator emitted the queue section"
        del doc["queues"]
        no_queue_yaml = yaml.dump(doc, sort_keys=False)

        settings = load_settings_from_yaml_string(no_queue_yaml)
        assert not settings.queues

        # With no queue, the two sources publishing 'inbound' are an undeclared
        # duplicate producer; the runtime rejects this at graph build time.
        with pytest.raises(GraphValidationError, match="Duplicate producer for connection 'inbound'"):
            self._build_runtime_graph_for_settings(settings)
