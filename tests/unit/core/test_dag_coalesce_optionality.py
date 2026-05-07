# tests/unit/core/test_dag_coalesce_optionality.py
"""Tests for field optionality preservation in coalesce union merge schema.

Fix: builder.py union merge was stripping optionality markers from branch
field specs, making all merged fields appear required. Now uses SchemaConfig
and FieldDefinition directly — optionality is a first-class boolean field.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from elspeth.contracts import NodeType
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.core.config import (
    CoalesceSettings,
    GateSettings,
    SourceSettings,
    TransformSettings,
)
from elspeth.core.dag import ExecutionGraph, WiredTransform, merge_union_fields
from elspeth.core.dag.models import GraphValidationError
from tests.helpers.coalesce import _add_coalesce_with_computed_schema


class TestUnionMergeOptionalityPreservation:
    """Tests for optionality in union merge using SchemaConfig objects.

    This class tests the production merge_union_fields() function directly,
    providing both semantic verification AND regression coverage for the
    builder.py union merge logic.

    Changes to builder.py will be reflected here because _merge_schemas
    delegates to the production merge_union_fields() function.
    """

    def _merge_schemas(
        self,
        branch_schemas: dict[str, SchemaConfig],
        *,
        policy: str = "best_effort",
    ) -> SchemaConfig:
        """Delegate to production merge_union_fields() function.

        This thin wrapper calls the same production function that builder.py
        uses, ensuring tests exercise real logic rather than a reimplementation.

        - require_all: OR semantics (a field is required if ANY branch
          requires it; branch-exclusive fields keep their source flag)
        - all others (best_effort/quorum/first): AND semantics (a field is
          required only if every contributing branch requires it; branch-
          exclusive fields are forced optional)
        """
        return merge_union_fields(
            branch_schemas,
            require_all=(policy == "require_all"),
        )

    def _get_field(self, schema: SchemaConfig, name: str) -> FieldDefinition:
        """Get a FieldDefinition by name from a SchemaConfig."""
        assert schema.fields is not None
        for fd in schema.fields:
            if fd.name == name:
                return fd
        raise AssertionError(f"Field {name!r} not found in schema")

    def test_optional_field_preserved(self) -> None:
        """Optional field in one branch or exclusive to one branch both remain optional in merged output."""
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("score", "float", required=False)),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("label", "str")),
                ),
            }
        )
        assert self._get_field(result, "score").required is False  # explicit optional preserved
        assert self._get_field(result, "id").required is True  # present in all branches
        assert self._get_field(result, "label").required is False  # branch-exclusive → optional

    def test_mixed_required_and_optional_yields_optional(self) -> None:
        """Field required in branch A, optional in branch B → optional in output."""
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=True),),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=False),),
                ),
            }
        )
        assert self._get_field(result, "score").required is False

    def test_all_required_stays_required(self) -> None:
        """Field required in ALL branches → required in output."""
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"),),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"),),
                ),
            }
        )
        assert self._get_field(result, "id").required is True

    def test_all_optional_stays_optional(self) -> None:
        """Field optional in ALL branches → optional in output."""
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=False),),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=False),),
                ),
            }
        )
        assert self._get_field(result, "score").required is False

    # ── require_all OR-semantics tests ───────────────────────────────────────

    def test_require_all_keeps_branch_exclusive_required_field_required(self) -> None:
        """Under require_all, a branch-exclusive REQUIRED field stays required.

        Branch A always arrives and always produces 'label'. After merge,
        'label' is in every merged row. The merged spec must reflect this.

        Regression for the unconditional AND-downgrade bug that produced
        false positives in build-time validation for valid require_all DAGs.
        """
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("label", "str", required=True)),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"),),
                ),
            },
            policy="require_all",
        )
        # 'label' is exclusive to branch A but always present (require_all).
        assert self._get_field(result, "label").required is True
        assert self._get_field(result, "id").required is True

    def test_require_all_keeps_branch_exclusive_optional_field_optional(self) -> None:
        """Under require_all, a branch-exclusive OPTIONAL field stays optional.

        Branch A always arrives, but its 'score' field is optional —
        branch A's row may not have it. The merged row may not have it
        either. The merged spec correctly reflects this as optional.
        """
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"), FieldDefinition("score", "float", required=False)),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("id", "int"),),
                ),
            },
            policy="require_all",
        )
        # 'score' is optional in branch A → optional in merged.
        assert self._get_field(result, "score").required is False
        assert self._get_field(result, "id").required is True

    def test_require_all_shared_field_required_in_one_branch_becomes_required(self) -> None:
        """Under require_all, a shared field required in one branch is required in merged.

        Branch A requires 'score', branch B has it optional. Branch A always
        produces 'score'. After dict.update, 'score' is in every merged row
        (either A's value or B's overwrite if B has it). The merged spec
        is required=True under require_all.

        This is the OR semantic for shared fields. Under AND (the wrong
        semantic for require_all), this would incorrectly become optional.
        """
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=True),),
                ),
                "branch_b": SchemaConfig(
                    mode="flexible",
                    fields=(FieldDefinition("score", "float", required=False),),
                ),
            },
            policy="require_all",
        )
        assert self._get_field(result, "score").required is True

    def test_abstaining_branch_does_not_downgrade_others(self) -> None:
        """A non-observed branch with fields=None must not downgrade other branches' fields.

        Regression test for a latent bomb in the builder's union merge. Before
        the fix, a branch with ``fields=None`` would silently mark every OTHER
        branch's required fields as optional: the post-loop pass applied AND
        semantics by comparing each field's branch-presence set against the
        FULL set of branches in the coalesce, not just the branches that
        actually contributed a typed contract.

        With a ``fields=None`` branch in the mix, no field could ever be
        present in ALL branches, so every required field got silently
        downgraded to optional — a catastrophic false negative.

        The fix introduced ``contributing_branches`` to exclude abstaining
        branches from the denominator. This test locks that behaviour in:
        two branches both require ``id`` and ``score``, a third branch
        abstains with ``fields=None``. Both fields must remain required.

        If the ``contributing_branches`` guard is removed, this test fails
        because ``branch_b`` is missing from ``branches_with_field[id]``
        and ``branches_with_field[score]``, so both get downgraded to
        ``required=False``.
        """
        result = self._merge_schemas(
            {
                "branch_a": SchemaConfig(
                    mode="flexible",
                    fields=(
                        FieldDefinition("id", "str", required=True),
                        FieldDefinition("score", "float", required=True),
                    ),
                ),
                "branch_b": SchemaConfig(mode="flexible", fields=None),
                "branch_c": SchemaConfig(
                    mode="flexible",
                    fields=(
                        FieldDefinition("id", "str", required=True),
                        FieldDefinition("score", "float", required=True),
                    ),
                ),
            }
        )
        assert self._get_field(result, "id").required is True
        assert self._get_field(result, "score").required is True


class TestCoalesceSinkRequiredFieldValidation:
    """Build-time validation of coalesce output vs downstream sink required fields.

    When a sink's declared_required_fields references a field that the upstream
    coalesce marks as optional (branch-exclusive or AND-downgraded), the
    configuration is broken and should fail at build time — not crash at runtime
    with a generic 'upstream schema bug' error.
    """

    def test_coalesce_to_sink_required_field_mismatch_raises(self) -> None:
        """Build-time validation catches coalesce output missing a sink's required field.

        Branch A guarantees field 'x'; branch B does not. Under best_effort
        policy, branch A might not arrive (lost), so the merged row may be
        missing 'x'. A downstream sink declares 'x' required in its
        declared_required_fields. Validation must fail at build time.

        Note: this test uses best_effort policy because under require_all,
        branch A always arrives and the field IS guaranteed in the merged
        row — see test_builder_end_to_end_require_all_accepts_branch_exclusive_sink_requirement
        for the require_all variant.

        Branch transforms set guaranteed_fields explicitly to mirror what real
        plugin code (BaseTransform._build_output_schema_config) produces — the
        coalesce union path skips branches that don't declare guarantees.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        branch_a_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=True),
            ),
            guaranteed_fields=("id", "x"),
        )
        branch_b_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("id",),
        )
        # Coalesce output under best_effort: 'x' is optional because branch A
        # may not arrive. The coalesce node's own output_schema_config is unused
        # by the union path of get_effective_guaranteed_fields — the
        # intersection is computed from the branches' guaranteed_fields directly.
        # Set here for parity with real builder behavior.
        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=False),
            ),
            guaranteed_fields=("id",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            output_schema_config=branch_a_schema,
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            output_schema_config=branch_b_schema,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "best_effort", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id", "x"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(
            GraphValidationError,
            match=r"upstream.*does not guarantee",
        ):
            graph.validate_edge_compatibility()

    def test_coalesce_to_sink_validates_when_all_fields_guaranteed(self) -> None:
        """Build-time validation passes when coalesce guarantees all sink-required fields."""
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        # Both branches guarantee 'x' → merged output guarantees 'x'
        branch_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=True),
            ),
            guaranteed_fields=("id", "x"),
        )
        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=True),
            ),
            guaranteed_fields=("id", "x"),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            output_schema_config=branch_schema,
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            output_schema_config=branch_schema,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "require_all", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id", "x"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        # Should not raise — all sink-required fields are guaranteed by coalesce output
        graph.validate_edge_compatibility()

    def test_coalesce_to_sink_branch_exclusive_field_caught(self) -> None:
        """Build-time validation catches the branch-exclusive case under best_effort.

        Branch A has a required field that branch B lacks. Under best_effort,
        branch A may be lost — the merged schema marks 'exclusive_to_a' as
        optional. A sink requiring that field fails validation at build time.

        Note: this test uses best_effort policy because under require_all,
        branch A always arrives and the field IS guaranteed in the merged
        row — see test_builder_end_to_end_require_all_accepts_branch_exclusive_sink_requirement
        for the require_all variant where validation correctly accepts.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        branch_a_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("exclusive_to_a", "int", required=True),
            ),
            guaranteed_fields=("id", "exclusive_to_a"),
        )
        branch_b_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("id",),
        )

        # Coalesce output under best_effort: 'exclusive_to_a' is optional
        # (branch-exclusive AND the source branch may not arrive).
        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("exclusive_to_a", "int", required=False),
            ),
            guaranteed_fields=("id",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            output_schema_config=branch_a_schema,
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            output_schema_config=branch_b_schema,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "best_effort", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id", "exclusive_to_a"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(GraphValidationError, match=r"exclusive_to_a"):
            graph.validate_edge_compatibility()

    def test_transform_between_coalesce_and_sink_still_validated(self) -> None:
        """COALESCE → TRANSFORM → SINK topology is still validated.

        If a transform sits between the coalesce and sink, the sink is
        validated against the TRANSFORM's output_schema_config (not the
        coalesce's). If the transform's output schema doesn't guarantee a
        field the sink requires, validation fails at build time.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig
        from elspeth.core.dag.graph import ExecutionGraph
        from elspeth.core.dag.models import GraphValidationError

        # Coalesce produces 'id' (required) and 'x' (optional) via union merge
        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=False),
            ),
        )

        # Transform passes through the coalesce output (keeps 'x' optional).
        # Set guaranteed_fields explicitly to mirror what BaseTransform's
        # _build_output_schema_config produces in production.
        transform_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=False),
            ),
            guaranteed_fields=("id",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        branch_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
        )
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="a", output_schema_config=branch_schema)
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="b", output_schema_config=branch_schema)
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "require_all", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "post_transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="enrich",
            output_schema_config=transform_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id", "x"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "post_transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("post_transform", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(GraphValidationError, match=r"upstream 'post_transform'.*does not guarantee"):
            graph.validate_edge_compatibility()

    def test_transform_between_coalesce_and_sink_passes_when_transform_guarantees(self) -> None:
        """COALESCE → TRANSFORM → SINK passes when the transform guarantees required fields.

        Even if the coalesce marks 'x' optional, a transform that declares 'x'
        required in its output commits to producing it. The sink's requirement
        is satisfied by the transform's contract, not the coalesce's output.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig
        from elspeth.core.dag.graph import ExecutionGraph

        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=False),  # optional from coalesce
            ),
        )
        # Transform commits to always producing x — declared via guaranteed_fields
        # mirroring real BaseTransform._build_output_schema_config output.
        transform_output_schema = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("x", "int", required=True),
            ),
            guaranteed_fields=("id", "x"),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        branch_schema = SchemaConfig(mode="flexible", fields=(FieldDefinition("id", "str", required=True),))
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="a", output_schema_config=branch_schema)
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="b", output_schema_config=branch_schema)
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "require_all", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "post_transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="enrich",
            output_schema_config=transform_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id", "x"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "post_transform", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("post_transform", "sink", label="continue", mode=RoutingMode.MOVE)

        graph.validate_edge_compatibility()  # Should not raise

    def test_aggregation_predecessor_skipped(self) -> None:
        """AGGREGATION → SINK skips required-field validation.

        Aggregations have dynamic output by design (e.g., BatchStats produces
        count/sum/mean rather than the input fields). The builder stores their
        `options.schema` as input-validation, not output, so validating sink
        requirements against it produces false positives. Validation must skip.

        Regression test for the case where the new sink-required-fields walk
        was firing on aggregation→sink topologies and rejecting valid configs.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        # Aggregations have output_schema_config=None (dynamic output by design).
        # The validation skips them via the `predecessor_schema is None` path.
        # See elspeth-c3a98c358c for the fix that eliminated the explicit
        # AGGREGATION skip in favor of this semantic approach.
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "stats",
            node_type=NodeType.AGGREGATION,
            plugin_name="batch_stats",
            config={"options": {"value_field": "value"}},
            output_schema_config=None,  # Dynamic output — no static guarantees
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"total_records"}),
        )

        graph.add_edge("source", "stats", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("stats", "sink", label="continue", mode=RoutingMode.MOVE)

        graph.validate_edge_compatibility()  # Should not raise

    def test_shape_changing_transform_between_coalesce_and_sink_uses_guaranteed_fields(
        self,
    ) -> None:
        """Sink validator must consult get_effective_guaranteed_fields, not fields tuple.

        Regression test for the original critical bug. ``_validate_sink_required_fields``
        was originally written to read ``output_schema_config.fields`` directly.
        The base ``BaseTransform._build_output_schema_config()`` (see
        ``src/elspeth/plugins/infrastructure/base.py``) copies INPUT ``fields``
        into the output config, recomputing only ``guaranteed_fields``. For a
        shape-changing transform (e.g., FieldMapper renaming ``id`` →
        ``customer_id``), the output schema's ``fields`` tuple still contains
        the pre-rename input names.

        This test mirrors FieldMapper's ``_build_field_mapper_output_schema_config``
        pattern: ``fields`` holds the input ``id`` (required=True) and
        ``guaranteed_fields`` holds the post-rename ``customer_id``. The sink
        requires only ``customer_id``.

        If the validator is reverted to reading ``output_schema_config.fields``
        directly, it would see ``id`` but not ``customer_id`` and falsely
        reject this valid topology. Under the current implementation it calls
        ``get_effective_guaranteed_fields()``, which OR's in explicit
        ``guaranteed_fields`` and sees ``customer_id`` → passes.

        Topology: source → fork → [t_a, t_b] → coalesce(union) → field_mapper → sink.
        Both branches guarantee ``id``; the field_mapper renames ``id`` to
        ``customer_id``; the sink requires ``customer_id``.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        branch_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("id",),
        )
        coalesce_output_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("id",),
        )
        # Shape-changing transform schema mirroring FieldMapper's pattern:
        # fields holds INPUT names (pre-rename), guaranteed_fields holds
        # OUTPUT names (post-rename). A validator that reads fields directly
        # would see "id" and falsely reject; one that uses
        # get_effective_guaranteed_fields() sees "customer_id" via
        # guaranteed_fields and correctly accepts.
        field_mapper_output_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("customer_id",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="a",
            output_schema_config=branch_schema,
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="b",
            output_schema_config=branch_schema,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"a": "a", "b": "b"}, "policy": "require_all", "merge": "union"},
            output_schema_config=coalesce_output_schema,
        )
        graph.add_node(
            "field_mapper",
            node_type=NodeType.TRANSFORM,
            plugin_name="field_mapper",
            output_schema_config=field_mapper_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"customer_id"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "field_mapper", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("field_mapper", "sink", label="continue", mode=RoutingMode.MOVE)

        # Call the sink-required-fields validator directly. We skip the full
        # validate_edge_compatibility() walk because the mocked schemas are
        # not meant to exercise the cross-plugin compatibility path — this
        # test is narrowly scoped to the sink predecessor walk.
        graph._validate_sink_required_fields()  # Should not raise

    def test_multi_edge_sink_predecessor_deduped(self) -> None:
        """Sink validator must visit each unique predecessor once, even with parallel edges.

        Regression test for the multi-edge case. The original implementation
        used ``self._graph.in_edges(node_id)`` which yields one tuple per
        parallel edge in a ``MultiDiGraph``. A gate routing two labels (e.g.,
        ``true`` and ``false``) to the same sink would therefore cause the
        validator to run twice for that predecessor.

        The fix switched to ``self._graph.predecessors(node_id)`` which
        yields each unique predecessor node exactly once regardless of the
        number of parallel edges between them.

        This test builds a graph with a single gate routing to one sink via
        two different labels ("true" and "false"). It then patches
        ``get_effective_guaranteed_fields`` with a call counter and asserts
        it is invoked exactly once for the gate predecessor. If the
        implementation regresses to ``.in_edges()``, the counter reads 2
        and the assertion fails.
        """
        from unittest.mock import patch

        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        gate_output_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("id", "str", required=True),),
            guaranteed_fields=("id",),
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="fork",
            output_schema_config=gate_output_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"id"}),
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        # Two parallel edges from gate to sink — different labels, same source/target.
        graph.add_edge("gate", "sink", label="true", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink", label="false", mode=RoutingMode.MOVE)

        # Patch the participation-preserving walk helper with a counter. The
        # validator must call it exactly ONCE for the gate predecessor,
        # regardless of how many parallel edges exist between gate and sink.
        #
        # The sink validator now needs both effective fields and participation
        # state, so it goes through the vote helper rather than the older
        # fields-only helper. Patching the vote helper covers both the
        # deduplication test and the cache-sharing behavior.
        call_counts: dict[str, int] = {}
        original = ExecutionGraph._walk_effective_guarantee_vote

        def counting_walk(
            self_inner: ExecutionGraph, node_id: str, cache: dict[str, object], field_cache: dict[str, frozenset[str]] | None = None
        ) -> object:
            call_counts[node_id] = call_counts.get(node_id, 0) + 1
            return original(self_inner, node_id, cache, field_cache)

        with patch.object(ExecutionGraph, "_walk_effective_guarantee_vote", new=counting_walk):
            graph._validate_sink_required_fields()  # Should not raise

        assert call_counts.get("gate", 0) == 1, (
            f"Expected gate predecessor to be visited exactly once, "
            f"but was visited {call_counts.get('gate', 0)} times. "
            f"This indicates the validator is iterating in_edges (which yields "
            f"one tuple per parallel edge) instead of predecessors (which "
            f"yields each unique source node once)."
        )

    def test_predecessor_with_explicit_empty_guarantees_rejects_sink_requirement(self) -> None:
        """Build-time validation rejects when predecessor explicitly guarantees nothing.

        SchemaConfig distinguishes abstain (guaranteed_fields=None) from
        explicit-empty (guaranteed_fields=()) — see the docstring on
        SchemaConfig.declares_guaranteed_fields. The coalesce union logic
        already uses this distinction.

        The sink validator used to collapse both cases via
        `if not guaranteed: continue`, skipping rejection when the predecessor
        EXPLICITLY declared zero guarantees. A sink that requires a field in
        that configuration has a statically-provable contract mismatch that
        should fail at build time, not be pushed to runtime.

        This test constructs the exact scenario from the P2 bug report:
        predecessor declares fields=(x?) (x optional) AND
        guaranteed_fields=() (explicit empty). Sink requires x. Validation
        must reject.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        predecessor_schema = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition("x", "int", required=False),),
            guaranteed_fields=(),  # EXPLICIT empty — declares_guaranteed_fields=True
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "t",
            node_type=NodeType.TRANSFORM,
            plugin_name="t",
            output_schema_config=predecessor_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"x"}),
        )
        graph.add_edge("source", "t", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("t", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(GraphValidationError, match=r"does not guarantee"):
            graph._validate_sink_required_fields()

    def test_predecessor_with_abstaining_guarantees_skips_validation(self) -> None:
        """Abstaining predecessor (guaranteed_fields=None) skips validation.

        Regression protection for the fix to
        test_predecessor_with_explicit_empty_guarantees_rejects_sink_requirement:
        the fix must NOT reject abstaining predecessors. Only explicit-empty
        declarations should trigger rejection.

        A truly abstaining predecessor must have no typed fields and no
        explicit ``guaranteed_fields`` declaration. The static schema can't
        prove whether ``x`` will be produced, so the runtime check still
        applies.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        predecessor_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=None,  # ABSTAIN — declares_guaranteed_fields=False
        )

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "t",
            node_type=NodeType.TRANSFORM,
            plugin_name="t",
            output_schema_config=predecessor_schema,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"x"}),
        )
        graph.add_edge("source", "t", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("t", "sink", label="continue", mode=RoutingMode.MOVE)

        # Must NOT raise — predecessor abstains, can't statically validate.
        graph._validate_sink_required_fields()

    def test_pass_through_inheriting_explicit_empty_guarantees_rejects_sink_requirement(self) -> None:
        """Propagated empty participation must still fail sink validation.

        Regression test for the generated bug triage in
        ``docs/bugs/generated/core/dag/graph.py.md``. The immediate
        predecessor of the sink is an observed pass-through transform whose
        *local* schema abstains from guarantees, but its effective guarantee
        set is empty because an upstream participating schema contributed an
        explicit empty vote.

        The bug was that ``_validate_sink_required_fields()`` only looked at
        the pass-through node's local ``declares_guaranteed_fields`` flag
        after seeing an empty effective guarantee set, so it misclassified
        the node as abstaining and skipped validation entirely.
        """
        from elspeth.contracts import RoutingMode
        from elspeth.core.dag.graph import ExecutionGraph

        upstream_schema = SchemaConfig(
            mode="fixed",
            fields=(FieldDefinition("maybe_field", "str", required=False),),
        )

        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            output_schema_config=upstream_schema,
        )
        graph.add_node(
            "passthrough",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config={"schema": {"mode": "observed"}},
            output_schema_config=None,
            passes_through_input=True,
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="csv_sink",
            declared_required_fields=frozenset({"maybe_field"}),
        )

        graph.add_edge("source", "passthrough", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("passthrough", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(GraphValidationError, match=r"does not guarantee"):
            graph.validate_edge_compatibility()


class _BuilderMockSource:
    """Mock source plugin for builder end-to-end tests."""

    name = "mock_source"
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    _on_validation_failure = "discard"
    on_success = "output"


class _BuilderMockSink:
    """Mock sink plugin with no declared required fields."""

    name = "mock_sink"
    input_schema = None
    config: ClassVar[dict[str, Any]] = {}
    _on_write_failure: str = "discard"
    declared_required_fields: ClassVar[frozenset[str]] = frozenset()

    def _reset_diversion_log(self) -> None:
        pass


class _TransformWithTypedSchema:
    """Mock transform that exposes a typed _output_schema_config with fields."""

    input_schema = None
    output_schema = None
    config: ClassVar[dict[str, Any]] = {"schema": {"mode": "observed"}}
    on_error: str | None = None
    on_success: str | None = "output"
    declared_output_fields: frozenset[str] = frozenset()
    passes_through_input: bool = False

    def __init__(self, name: str, schema: SchemaConfig) -> None:
        self.name = name
        self._output_schema_config = schema


class TestBuilderBranchExclusiveFieldDowngrade:
    """End-to-end coverage for the builder's union-merge downgrade pass.

    The tests above this class (_merge_schemas and the manually-built graph
    with a pre-computed coalesce schema) all bypass the real builder. If the
    downgrade pass in ``src/elspeth/core/dag/builder.py`` were deleted, those
    tests would still pass.

    These tests drive ``ExecutionGraph.from_plugin_instances`` with asymmetric
    branch schemas and inspect the resulting coalesce node's
    ``output_schema_config`` directly, so a regression in the builder's
    downgrade pass will fail here loudly.
    """

    def _build_graph(
        self,
        *,
        branch_a_fields: tuple[FieldDefinition, ...],
        branch_b_fields: tuple[FieldDefinition, ...],
    ) -> ExecutionGraph:
        """Build: source → fork gate → [transform_a, transform_b] → coalesce → sink."""
        source = _BuilderMockSource()

        t_a = _TransformWithTypedSchema(
            "branch_transform_a",
            SchemaConfig(mode="flexible", fields=branch_a_fields),
        )
        t_b = _TransformWithTypedSchema(
            "branch_transform_b",
            SchemaConfig(mode="flexible", fields=branch_b_fields),
        )

        wired_a = WiredTransform(
            plugin=t_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_a",
                plugin=t_a.name,
                input="branch_a",
                on_success="t_a_out",
                on_error="discard",
                options={},
            ),
        )
        wired_b = WiredTransform(
            plugin=t_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_b",
                plugin=t_b.name,
                input="branch_b",
                on_success="t_b_out",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        coalesce = CoalesceSettings(
            name="merger",
            branches={"branch_a": "t_a_out", "branch_b": "t_b_out"},
            policy="require_all",
            merge="union",
            on_success="output",
        )

        return ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired_a, wired_b],
            sinks={"output": _BuilderMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

    def _get_field(self, schema: SchemaConfig, name: str) -> FieldDefinition:
        assert schema.fields is not None
        for fd in schema.fields:
            if fd.name == name:
                return fd
        raise AssertionError(f"Field {name!r} not found in schema")

    def test_builder_preserves_branch_exclusive_required_field_under_require_all(self) -> None:
        """End-to-end: builder's union merge KEEPS branch-exclusive fields required under require_all.

        Under require_all coalesce policy, every branch always arrives before
        the merge fires (CoalesceExecutor._should_merge enforces this). Then
        _merge_data unions every arrived branch's keys via dict.update, so a
        field produced by branch A is always in the merged row regardless of
        whether branch B has it.

        Therefore: a field required in branch A is GUARANTEED in the merged
        output under require_all, even if branch B doesn't declare it. The
        builder must reflect this by KEEPING the field required.

        Regresses a previous over-conservative AND-semantics fix that
        unconditionally downgraded branch-exclusive fields to optional,
        producing false positives in build-time validation for valid
        require_all DAGs that fed sinks requiring such fields.
        """
        graph = self._build_graph(
            branch_a_fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("exclusive_to_a", "int", required=True),
            ),
            branch_b_fields=(FieldDefinition("id", "str", required=True),),
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.fields is not None, "Expected typed schema, not observed"

        # Branch-exclusive REQUIRED field stays required under require_all:
        # branch A always arrives and always produces it.
        assert self._get_field(coal_schema, "exclusive_to_a").required is True
        # Field present in ALL branches stays required (trivially).
        assert self._get_field(coal_schema, "id").required is True

    def test_builder_keeps_shared_required_fields_required(self) -> None:
        """Control case: field required in BOTH branches stays required.

        Complements the branch-exclusive test — proves the downgrade is
        keyed on branch-presence, not applied blanket to every field.
        """
        graph = self._build_graph(
            branch_a_fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("shared", "int", required=True),
            ),
            branch_b_fields=(
                FieldDefinition("id", "str", required=True),
                FieldDefinition("shared", "int", required=True),
            ),
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.fields is not None

        assert self._get_field(coal_schema, "id").required is True
        assert self._get_field(coal_schema, "shared").required is True

    def test_builder_materializes_union_for_require_all_observed_branches(self) -> None:
        """Builder stores UNION (not intersection) on require_all coalesce output for observed branches.

        Under require_all, every branch always arrives and dict.update unions
        all branch keys into the merged row. A field guaranteed by ANY branch
        is therefore in every merged row, so the stored
        output_schema_config.guaranteed_fields should be the UNION of branch
        guarantees.

        Regression: builder.py used set.intersection(*guaranteed_sets)
        unconditionally at line 887. For OBSERVED-mode branches where
        guarantees are carried only via explicit guaranteed_fields (no typed
        fields to derive implicit guarantees from), the stale intersection
        was the only source of truth — and it was wrong under require_all.

        Downstream consumers reading the stored schema directly (nested
        coalesces via get_schema_config_from_node, deferred config gates via
        _best_schema_config) would see the stale intersection and
        under-report valid require_all guarantees.
        """
        source = _BuilderMockSource()
        # Observed-mode branches: guarantees ONLY via guaranteed_fields, no
        # typed fields to fall back to. This exposes the stale-tuple bug.
        t_a = _TransformWithTypedSchema(
            "t_a_plugin",
            SchemaConfig(mode="observed", fields=None, guaranteed_fields=("id", "a_only")),
        )
        t_b = _TransformWithTypedSchema(
            "t_b_plugin",
            SchemaConfig(mode="observed", fields=None, guaranteed_fields=("id", "b_only")),
        )
        wired_a = WiredTransform(
            plugin=t_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_a",
                plugin=t_a.name,
                input="branch_a",
                on_success="t_a_out",
                on_error="discard",
                options={},
            ),
        )
        wired_b = WiredTransform(
            plugin=t_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_b",
                plugin=t_b.name,
                input="branch_b",
                on_success="t_b_out",
                on_error="discard",
                options={},
            ),
        )
        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )
        coalesce = CoalesceSettings(
            name="merger",
            branches={"branch_a": "t_a_out", "branch_b": "t_b_out"},
            policy="require_all",
            merge="union",
            on_success="output",
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired_a, wired_b],
            sinks={"output": _BuilderMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        assert coal_schema.guaranteed_fields is not None, (
            "coalesce must materialize an explicit guaranteed_fields tuple because at least one branch declares guarantees"
        )

        # The materialized tuple must be the UNION of branch guarantees,
        # not the intersection. Previously stored as ('id',); must now include
        # 'a_only' and 'b_only' because under require_all both branches always
        # arrive.
        assert set(coal_schema.guaranteed_fields) == {"id", "a_only", "b_only"}, (
            f"Expected materialized union {{'id', 'a_only', 'b_only'}} for "
            f"require_all observed branches, got {set(coal_schema.guaranteed_fields)}. "
            f"The stale-intersection materialization would produce {{'id'}} only. "
            f"Downstream gates/nested-coalesces reading this schema would "
            f"under-report valid require_all guarantees."
        )
        # And the schema's effective guarantees (what downstream consumers see
        # when they call schema_config.get_effective_guaranteed_fields()) must
        # match — this is the path that gates and nested coalesces take.
        assert coal_schema.get_effective_guaranteed_fields() == frozenset({"id", "a_only", "b_only"})

    def test_builder_materializes_intersection_for_best_effort_observed_branches(self) -> None:
        """Control: best_effort materialization still uses INTERSECTION.

        Under best_effort, branches may be lost, so only fields guaranteed by
        EVERY branch survive. The materialized tuple must reflect that.
        Protects against an over-eager fix to
        test_builder_materializes_union_for_require_all_observed_branches
        that would union guarantees regardless of policy.
        """
        source = _BuilderMockSource()
        t_a = _TransformWithTypedSchema(
            "t_a_plugin",
            SchemaConfig(mode="observed", fields=None, guaranteed_fields=("id", "a_only")),
        )
        t_b = _TransformWithTypedSchema(
            "t_b_plugin",
            SchemaConfig(mode="observed", fields=None, guaranteed_fields=("id", "b_only")),
        )
        wired_a = WiredTransform(
            plugin=t_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_a",
                plugin=t_a.name,
                input="branch_a",
                on_success="t_a_out",
                on_error="discard",
                options={},
            ),
        )
        wired_b = WiredTransform(
            plugin=t_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_b",
                plugin=t_b.name,
                input="branch_b",
                on_success="t_b_out",
                on_error="discard",
                options={},
            ),
        )
        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )
        coalesce = CoalesceSettings(
            name="merger",
            branches={"branch_a": "t_a_out", "branch_b": "t_b_out"},
            policy="best_effort",
            merge="union",
            on_success="output",
            timeout_seconds=60.0,
        )

        graph = ExecutionGraph.from_plugin_instances(
            source=source,  # type: ignore[arg-type]
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired_a, wired_b],
            sinks={"output": _BuilderMockSink()},  # type: ignore[dict-item]
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        coal_schema = coalesce_nodes[0].output_schema_config
        assert coal_schema is not None
        # Intersection: only 'id' is in both branches' guarantees.
        assert set(coal_schema.guaranteed_fields or ()) == {"id"}

    def _build_end_to_end_branch_exclusive_setup(self, *, policy: str) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Build the source/transforms/gate/coalesce for an asymmetric-branch sink test.

        Returns (source, wired_a, wired_b, fork_gate, coalesce, sink_class).
        Branch A guarantees 'id' and 'exclusive_to_a'; branch B only guarantees 'id'.
        Sink declares both as required. The coalesce policy is parameterized.
        """

        class _SinkRequiringExclusive:
            name = "strict_sink"
            input_schema = None
            config: ClassVar[dict[str, Any]] = {}
            _on_write_failure: str = "discard"
            declared_required_fields: ClassVar[frozenset[str]] = frozenset({"id", "exclusive_to_a"})

            def _reset_diversion_log(self) -> None:
                pass

        source = _BuilderMockSource()
        t_a = _TransformWithTypedSchema(
            "branch_transform_a",
            SchemaConfig(
                mode="flexible",
                fields=(
                    FieldDefinition("id", "str", required=True),
                    FieldDefinition("exclusive_to_a", "int", required=True),
                ),
                guaranteed_fields=("id", "exclusive_to_a"),
            ),
        )
        t_b = _TransformWithTypedSchema(
            "branch_transform_b",
            SchemaConfig(
                mode="flexible",
                fields=(FieldDefinition("id", "str", required=True),),
                guaranteed_fields=("id",),
            ),
        )

        wired_a = WiredTransform(
            plugin=t_a,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_a",
                plugin=t_a.name,
                input="branch_a",
                on_success="t_a_out",
                on_error="discard",
                options={},
            ),
        )
        wired_b = WiredTransform(
            plugin=t_b,  # type: ignore[arg-type]
            settings=TransformSettings(
                name="t_b",
                plugin=t_b.name,
                input="branch_b",
                on_success="t_b_out",
                on_error="discard",
                options={},
            ),
        )

        fork_gate = GateSettings(
            name="splitter",
            input="source_out",
            condition="True",
            routes={"true": "fork", "false": "output"},
            fork_to=["branch_a", "branch_b"],
        )

        # best_effort requires a timeout; require_all does not. Pass timeout
        # unconditionally — it's harmless for require_all (the policy ignores it).
        coalesce = CoalesceSettings(
            name="merger",
            branches={"branch_a": "t_a_out", "branch_b": "t_b_out"},
            policy=policy,
            merge="union",
            on_success="output",
            timeout_seconds=60.0,
        )

        return source, wired_a, wired_b, fork_gate, coalesce, _SinkRequiringExclusive

    def test_builder_end_to_end_require_all_accepts_branch_exclusive_sink_requirement(self) -> None:
        """End-to-end: under require_all, the sink can require a branch-exclusive field.

        Branch A produces 'exclusive_to_a' as required, branch B does not.
        Sink declares 'exclusive_to_a' in declared_required_fields. Under
        require_all, every branch always arrives, so branch A's field IS in
        every merged row — the sink requirement is correctly satisfied.

        Build-time validation must NOT reject this configuration. This test
        regresses a previous over-conservative AND-semantics fix that
        unconditionally downgraded branch-exclusive fields to optional, causing
        validate_edge_compatibility() to raise on valid require_all DAGs.
        """
        source, wired_a, wired_b, fork_gate, coalesce, sink_class = self._build_end_to_end_branch_exclusive_setup(policy="require_all")

        # Build must succeed — under require_all, the sink requirement is
        # actually satisfied because branch A always arrives with the field.
        graph = ExecutionGraph.from_plugin_instances(
            source=source,
            source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
            transforms=[wired_a, wired_b],
            sinks={"output": sink_class()},
            aggregations={},
            gates=[fork_gate],
            coalesce_settings=[coalesce],
        )

        # Sanity check: the coalesce node's effective guarantees include
        # exclusive_to_a (via OR over branches under require_all).
        coalesce_nodes = [n for n in graph.get_nodes() if n.node_type == NodeType.COALESCE]
        assert len(coalesce_nodes) == 1
        guaranteed = graph.get_effective_guaranteed_fields(coalesce_nodes[0].node_id)
        assert "exclusive_to_a" in guaranteed
        assert "id" in guaranteed

    def test_builder_end_to_end_best_effort_rejects_branch_exclusive_sink_requirement(self) -> None:
        """End-to-end: under best_effort, a sink requiring a branch-exclusive field is rejected.

        Same setup as the require_all test, but with best_effort policy: the
        branch that produces 'exclusive_to_a' may NOT arrive, so the merged
        row may be missing the field. The sink's requirement cannot be
        statically satisfied — build-time validation must reject this.

        This is the AND-semantics case the previous over-conservative fix
        was trying to catch (correctly, for non-require_all policies).
        """
        source, wired_a, wired_b, fork_gate, coalesce, sink_class = self._build_end_to_end_branch_exclusive_setup(policy="best_effort")

        with pytest.raises(GraphValidationError, match=r"exclusive_to_a"):
            ExecutionGraph.from_plugin_instances(
                source=source,
                source_settings=SourceSettings(plugin=source.name, on_success="source_out", options={}),
                transforms=[wired_a, wired_b],
                sinks={"output": sink_class()},
                aggregations={},
                gates=[fork_gate],
                coalesce_settings=[coalesce],
            )

    def test_builder_materializes_typed_required_fields_as_guarantees(self) -> None:
        """Builder includes typed required fields in merged_guaranteed_tuple.

        Bug regression: The builder was only considering explicit guaranteed_fields
        when computing merged_guaranteed_tuple, ignoring typed required fields.
        A branch with mode="fixed", fields=(id, x) required but guaranteed_fields=None
        should still contribute {id, x} to the merged guarantees.

        This test verifies that the coalesce node's stored schema has the correct
        guaranteed_fields materialized from typed required fields. We use manual
        graph construction + get_schema_config_from_node to verify the stored schema.
        """
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: typed fields (id, x) required, NO explicit guaranteed_fields
        schema_a = SchemaConfig(
            mode="fixed",
            fields=(
                FieldDefinition("id", "int", required=True),
                FieldDefinition("x", "str", required=True),
            ),
            guaranteed_fields=None,  # No explicit declaration
        )
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_a,
        )

        # Branch B: typed field (id) required, NO explicit guaranteed_fields
        schema_b = SchemaConfig(
            mode="fixed",
            fields=(FieldDefinition("id", "int", required=True),),
            guaranteed_fields=None,  # No explicit declaration
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_b,
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # Add coalesce with require_all policy (union semantics)
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce",
            ["branch_a", "branch_b"],
            policy="require_all",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        # Verify effective guarantees via the graph API
        # Under require_all, merged guarantees = union of typed required fields
        # = {id, x} from branch_a | {id} from branch_b = {id, x}
        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"id", "x"})


class TestBestEffortBranchLossDocumentation:
    """Document the build-time vs runtime guarantee gap for best_effort.

    Systems review identified this as a "Shifting the Burden" archetype:
    build-time validation gives confidence, but cannot prove runtime branch
    arrival. These tests document the limitation explicitly.

    Build-time validation cannot prove a branch will arrive — it proves only
    that IF all declared branches arrive, the guaranteed fields will be present.
    Under best_effort, branches may timeout or be lost, producing merged rows
    that are missing branch-exclusive fields the sink requires.
    """

    def test_best_effort_build_passes_for_branch_exclusive_required_field(self) -> None:
        """Build-time validation PASSES for best_effort with branch-exclusive field.

        This documents expected behavior: best_effort uses intersection semantics,
        so a branch-exclusive field from branch_a is NOT guaranteed in the merged
        output (branch_a might not arrive). Therefore, a sink requiring that field
        would be REJECTED at build-time — this is correct.

        The inverse scenario (require_all) is covered by other tests where the
        sink requirement IS accepted because union semantics guarantee the field.
        """
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        # Branch A: has exclusive field 'x'
        schema_a = SchemaConfig(
            mode="fixed",
            fields=(
                FieldDefinition("id", "int", required=True),
                FieldDefinition("x", "str", required=True),  # exclusive to A
            ),
            guaranteed_fields=None,
        )
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_a,
        )

        # Branch B: only has 'id'
        schema_b = SchemaConfig(
            mode="fixed",
            fields=(FieldDefinition("id", "int", required=True),),
            guaranteed_fields=None,
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_b,
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # best_effort coalesce: intersection semantics
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce",
            ["branch_a", "branch_b"],
            policy="best_effort",
            extra_config={
                "branches": {"branch_a": {}, "branch_b": {}},
                "timeout_seconds": 30.0,
            },
        )

        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        # Under best_effort, intersection of guarantees = {id} only
        # Field 'x' is NOT guaranteed (branch_a might not arrive)
        result = graph.get_effective_guaranteed_fields("coalesce")
        assert result == frozenset({"id"}), (
            f"best_effort should use intersection semantics, got {result}. Branch-exclusive field 'x' should NOT be in guaranteed set."
        )

        # This documents the limitation: build-time validation correctly reflects
        # that 'x' is not guaranteed. A sink requiring 'x' would be rejected.
        # The runtime can still produce rows with 'x' (when branch_a arrives),
        # but build-time cannot promise it.


class TestDualComputationCrossPathAssertion:
    """Cross-path assertions: builder storage vs. graph query must agree.

    The builder computes merged guaranteed_fields at build time and stores them
    in the coalesce node's output_schema_config.guaranteed_fields. The graph's
    get_effective_guaranteed_fields() method recomputes the same value from
    predecessor schemas at query time.

    These two code paths MUST produce identical results. This test class exists
    because panel review identified the dual computation as a maintenance risk:
    changes to one path without updating the other would create silent validation
    divergence.

    Ref: builder.py:881 comment "must match ExecutionGraph.get_effective_guaranteed_fields()"
    """

    def test_require_all_stored_matches_computed(self) -> None:
        """Cross-path: builder storage == graph query under require_all policy."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="source")

        # Two branches with different guarantees
        schema_a = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_a"),
        )
        schema_b = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_b"),
        )
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_a,
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_b,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # require_all coalesce: union semantics
        # Simulate builder storing the pre-computed union result
        stored_guarantees = ("exclusive_a", "exclusive_b", "id")  # sorted union
        coalesce_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=stored_guarantees,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={
                "merge": "union",
                "branches": {"branch_a": {}, "branch_b": {}},
                "policy": "require_all",
            },
            output_schema_config=coalesce_schema,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        # Query the graph's computed value
        computed = graph.get_effective_guaranteed_fields("coalesce")

        # Cross-path assertion: stored == computed
        assert computed == frozenset(stored_guarantees), (
            f"Cross-path divergence detected! "
            f"Builder stored {stored_guarantees}, graph computed {computed}. "
            "Check builder.py and graph.py coalesce guarantee logic for consistency."
        )

    def test_best_effort_stored_matches_computed(self) -> None:
        """Cross-path: builder storage == graph query under best_effort policy."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="source")

        # Two branches with different guarantees
        schema_a = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_a"),
        )
        schema_b = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_b"),
        )
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_a,
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_b,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # best_effort coalesce: intersection semantics
        # Simulate builder storing the pre-computed intersection result
        stored_guarantees = ("id",)  # only common field survives
        coalesce_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=stored_guarantees,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={
                "merge": "union",
                "branches": {"branch_a": {}, "branch_b": {}},
                "policy": "best_effort",
                "timeout_seconds": 30.0,
            },
            output_schema_config=coalesce_schema,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        # Query the graph's computed value
        computed = graph.get_effective_guaranteed_fields("coalesce")

        # Cross-path assertion: stored == computed
        assert computed == frozenset(stored_guarantees), (
            f"Cross-path divergence detected! "
            f"Builder stored {stored_guarantees}, graph computed {computed}. "
            "Check builder.py and graph.py coalesce guarantee logic for consistency."
        )

    def test_quorum_equals_branches_uses_union_semantics(self) -> None:
        """Cross-path: quorum_count == len(branches) uses union semantics like require_all."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="source")

        # Two branches with different guarantees
        schema_a = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_a"),
        )
        schema_b = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=("id", "exclusive_b"),
        )
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_a,
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config={},
            output_schema_config=schema_b,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge("source", b, label=b)

        # quorum with quorum_count == 2 == len(branches): union semantics
        # This is runtime-equivalent to require_all (all branches must arrive)
        stored_guarantees = ("exclusive_a", "exclusive_b", "id")  # sorted union
        coalesce_schema = SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=stored_guarantees,
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={
                "merge": "union",
                "branches": {"branch_a": {}, "branch_b": {}},
                "policy": "quorum",
                "quorum_count": 2,  # == len(branches)
            },
            output_schema_config=coalesce_schema,
        )
        for b in ("branch_a", "branch_b"):
            graph.add_edge(b, "coalesce", label="continue")

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("coalesce", "sink", label="continue")

        # Query the graph's computed value
        computed = graph.get_effective_guaranteed_fields("coalesce")

        # Cross-path assertion: quorum=N should use union (same as require_all)
        assert computed == frozenset(stored_guarantees), (
            f"Cross-path divergence for quorum_count == len(branches)! "
            f"Builder stored {stored_guarantees}, graph computed {computed}. "
            "quorum_count == branch_count should use union semantics like require_all."
        )


class TestMergeUnionFieldsNullableSemantics:
    """Tests for nullable tracking in union field merging (P1 fix).

    The P1 bug: Under require_all with union_collision_policy='last_wins',
    if branch A has `x: int` (required, non-nullable) and branch B has
    `x: int?` (optional, nullable), the OR semantics marked `x` as required.
    But at runtime, B can win the collision and produce `x=None`, failing
    downstream consumers expecting `x: int`.

    The fix: Track nullable explicitly. A shared field is nullable if ANY
    branch has it nullable (because that branch can win via last_wins).
    """

    def test_require_all_mixed_nullable_shared_field_becomes_nullable(self) -> None:
        """P1 fix: When branch A has x:int and branch B has x:int?, merged x is nullable."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=False, nullable=True),),
        )
        merged = merge_union_fields({"a": branch_a, "b": branch_b}, require_all=True)

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        # Key assertion: even with OR semantics for required, nullable propagates
        assert x_field.nullable is True, "Shared field with nullable branch should be nullable"
        # OR semantics: required in merged output (required in ANY)
        assert x_field.required is True, "OR semantics: required if required in ANY branch"

    def test_require_all_all_required_non_nullable_stays_non_nullable(self) -> None:
        """When all branches require x:int (non-nullable), merged x is non-nullable."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields({"a": branch_a, "b": branch_b}, require_all=True)

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.required is True
        assert x_field.nullable is False, "All-required non-nullable should stay non-nullable"

    def test_best_effort_mixed_nullable_is_nullable_and_optional(self) -> None:
        """Under best_effort (AND semantics), mixed nullable becomes optional+nullable."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=False, nullable=True),),
        )
        merged = merge_union_fields({"a": branch_a, "b": branch_b}, require_all=False)

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        # AND semantics: optional if optional in ANY branch
        assert x_field.required is False, "AND semantics: optional if optional in ANY"
        assert x_field.nullable is True, "Nullable if nullable in ANY branch"

    def test_branch_exclusive_field_inherits_nullable_under_require_all(self) -> None:
        """Branch-exclusive fields under require_all preserve source nullable status."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition(name="shared", field_type="int", required=True, nullable=False),
                FieldDefinition(name="a_only", field_type="str", required=True, nullable=False),
            ),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="shared", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields({"a": branch_a, "b": branch_b}, require_all=True)

        assert merged.fields is not None
        a_only = next(f for f in merged.fields if f.name == "a_only")
        # Under require_all, branch-exclusive fields keep their required status
        assert a_only.required is True, "Branch-exclusive required under require_all"
        assert a_only.nullable is False, "Branch-exclusive non-nullable preserved"

    def test_branch_exclusive_field_forced_nullable_under_best_effort(self) -> None:
        """Branch-exclusive fields under best_effort become optional+nullable."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(
                FieldDefinition(name="shared", field_type="int", required=True, nullable=False),
                FieldDefinition(name="a_only", field_type="str", required=True, nullable=False),
            ),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="shared", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields({"a": branch_a, "b": branch_b}, require_all=False)

        assert merged.fields is not None
        a_only = next(f for f in merged.fields if f.name == "a_only")
        # Under best_effort, branch-exclusive fields become optional (branch may not arrive)
        assert a_only.required is False, "Branch-exclusive forced optional under best_effort"
        assert a_only.nullable is True, "Branch-exclusive forced nullable (may be absent)"


class TestCollisionPolicyNullableSemantics:
    """Tests for D5 fix: collision_policy-aware nullable for shared fields.

    The previous OR logic ("nullable if ANY branch is nullable") was designed
    for "any branch can win" semantics. But first_wins/last_wins are deterministic:
    - first_wins: first branch (in declaration order) always wins collisions
    - last_wins: last branch (in declaration order) always wins collisions

    The schema should reflect which branch's value actually appears at runtime.
    """

    def test_first_wins_uses_first_branch_nullable(self) -> None:
        """With first_wins, shared field nullable comes from FIRST branch only."""
        # Branch A (first): non-nullable
        # Branch B (second): nullable
        # With first_wins, A always wins → output is non-nullable
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="first_wins",
            branch_order=["a", "b"],
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is False, "first_wins: use first branch's nullable (False)"

    def test_first_wins_nullable_first_branch(self) -> None:
        """With first_wins, if first branch is nullable, merged is nullable."""
        # Branch A (first): nullable
        # Branch B (second): non-nullable
        # With first_wins, A always wins → output is nullable
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="first_wins",
            branch_order=["a", "b"],
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is True, "first_wins: use first branch's nullable (True)"

    def test_last_wins_uses_last_branch_nullable(self) -> None:
        """With last_wins, shared field nullable comes from LAST branch only."""
        # Branch A (first): nullable
        # Branch B (second): non-nullable
        # With last_wins, B always wins → output is non-nullable
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="last_wins",
            branch_order=["a", "b"],
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is False, "last_wins: use last branch's nullable (False)"

    def test_last_wins_nullable_last_branch(self) -> None:
        """With last_wins, if last branch is nullable, merged is nullable."""
        # Branch A (first): non-nullable
        # Branch B (second): nullable
        # With last_wins, B always wins → output is nullable
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="last_wins",
            branch_order=["a", "b"],
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is True, "last_wins: use last branch's nullable (True)"

    def test_fail_policy_uses_or_logic(self) -> None:
        """With fail policy, use OR logic (conservative; collisions fail at runtime)."""
        # Branch A: non-nullable
        # Branch B: nullable
        # With fail, collisions fail at runtime, so OR is conservative fallback
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="fail",
            branch_order=["a", "b"],
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is True, "fail policy: OR logic (conservative)"

    def test_branch_order_determines_first_last(self) -> None:
        """branch_order controls which branch is 'first' or 'last'."""
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )

        # With branch_order=["a", "b"], A is first
        merged_a_first = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="first_wins",
            branch_order=["a", "b"],
        )
        assert merged_a_first.fields is not None
        x_a_first = next(f for f in merged_a_first.fields if f.name == "x")
        assert x_a_first.nullable is True, "A is first, A is nullable"

        # With branch_order=["b", "a"], B is first
        merged_b_first = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="first_wins",
            branch_order=["b", "a"],
        )
        assert merged_b_first.fields is not None
        x_b_first = next(f for f in merged_b_first.fields if f.name == "x")
        assert x_b_first.nullable is False, "B is first, B is non-nullable"


class TestBuildCoalesceSchemaHandlesNullable:
    """Tests for _build_coalesce_schema() nullable flag handling.

    Fix: _build_coalesce_schema() was ignoring the nullable flag when building
    Pydantic field definitions. A field with required=True, nullable=True should
    produce (int | None, ...) not (int, ...), so that downstream type validation
    correctly rejects consumers expecting non-nullable fields.

    This completes the P1 fix — merge_union_fields() now tracks nullable, and
    _build_coalesce_schema() propagates it to the Pydantic schema used for DAG
    validation.
    """

    def test_required_nullable_produces_optional_pydantic_type(self) -> None:
        """A required+nullable field should produce `int | None` type (with required marker)."""
        from types import UnionType
        from typing import get_args, get_origin

        from elspeth.core.dag.graph import _build_coalesce_schema

        schema_config = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        pydantic_schema = _build_coalesce_schema(schema_config)

        # Check field annotation includes None
        field_info = pydantic_schema.model_fields["x"]
        annotation = field_info.annotation

        # Should be a Union type (int | None)
        origin = get_origin(annotation)
        assert origin is UnionType or annotation is type(int | None), f"Expected union type for nullable field, got {annotation}"

        # Should include both int and None
        args = get_args(annotation)
        assert int in args, f"Expected int in union, got {args}"
        assert type(None) in args, f"Expected NoneType in union, got {args}"

        # Should still be required (no default)
        assert field_info.is_required(), "Nullable but required field should have no default"

    def test_required_non_nullable_produces_strict_pydantic_type(self) -> None:
        """A required+non-nullable field should produce strict `int` type."""
        from elspeth.core.dag.graph import _build_coalesce_schema

        schema_config = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        pydantic_schema = _build_coalesce_schema(schema_config)

        field_info = pydantic_schema.model_fields["x"]

        # Should be plain int, not a union
        assert field_info.annotation is int, f"Expected int, got {field_info.annotation}"

        # Should be required
        assert field_info.is_required(), "Required non-nullable field should have no default"

    def test_optional_field_produces_optional_pydantic_type(self) -> None:
        """An optional field should produce `int | None` with default None."""
        from types import UnionType
        from typing import get_origin

        from elspeth.core.dag.graph import _build_coalesce_schema

        schema_config = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=False, nullable=False),),
        )
        pydantic_schema = _build_coalesce_schema(schema_config)

        field_info = pydantic_schema.model_fields["x"]
        annotation = field_info.annotation

        # Should be a Union type (int | None) since optional means may be absent
        origin = get_origin(annotation)
        assert origin is UnionType or annotation is type(int | None), f"Expected union type for optional field, got {annotation}"

        # Should have default None (not required)
        assert not field_info.is_required(), "Optional field should have default"
        assert field_info.default is None, "Optional field should default to None"


class TestMixedObservedExplicitBranches:
    """Tests for D1 fix: mixed observed/explicit branches must not silently discard fields.

    Bug: The loop in merge_union_fields() used `all_observed = True; break` on the
    FIRST observed branch, silently discarding typed fields from subsequent branches.

    Fix: Check `all(schema_cfg.is_observed for ...)` BEFORE the loop, then `continue`
    (not break) for observed branches during processing.
    """

    def test_mixed_observed_explicit_preserves_explicit_fields(self) -> None:
        """When one branch is observed and another has fields, fields must be preserved.

        Prior bug: breaks on first observed branch, returns observed schema, silently
        discarding branch B's typed fields.
        """
        # Branch A: observed (no fields)
        branch_a = SchemaConfig(mode="observed", fields=None)

        # Branch B: flexible with typed fields
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True),),
        )

        result = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
        )

        # Per docstring: "If all branches are observed-mode or no branches contribute
        # fields, returns an observed-mode schema."
        # Since NOT all branches are observed (branch_b is flexible), result should
        # preserve branch_b's fields.
        assert not result.is_observed, (
            f"Only branch_a is observed, branch_b has fields. "
            f"Result should preserve branch_b's fields, not return observed. "
            f"Got mode={result.mode}"
        )
        assert result.fields is not None, "Should have fields from branch_b"
        field_names = {fd.name for fd in result.fields}
        assert "x" in field_names, "Field 'x' from branch_b should be preserved"

    def test_all_observed_returns_observed(self) -> None:
        """When ALL branches are observed, result should be observed."""
        branch_a = SchemaConfig(mode="observed", fields=None)
        branch_b = SchemaConfig(mode="observed", fields=None)

        result = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
        )

        assert result.is_observed, "All branches observed -> result should be observed"

    def test_observed_first_explicit_second_preserves_fields(self) -> None:
        """Order independence: observed branch first, explicit branch second."""
        # Explicitly test dict ordering to ensure the fix works regardless of
        # iteration order (which caused the original bug when observed was first)
        from collections import OrderedDict

        branches: dict[str, SchemaConfig] = OrderedDict()
        branches["observed_first"] = SchemaConfig(mode="observed", fields=None)
        branches["explicit_second"] = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="y", field_type="str", required=False),),
        )

        result = merge_union_fields(branches, require_all=True)

        assert not result.is_observed, "Mixed branches should not return observed"
        assert result.fields is not None
        field_names = {fd.name for fd in result.fields}
        assert "y" in field_names, "Field 'y' from explicit branch should be preserved"

    def test_explicit_first_observed_second_preserves_fields(self) -> None:
        """Order independence: explicit branch first, observed branch second."""
        from collections import OrderedDict

        branches: dict[str, SchemaConfig] = OrderedDict()
        branches["explicit_first"] = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="z", field_type="float", required=True),),
        )
        branches["observed_second"] = SchemaConfig(mode="observed", fields=None)

        result = merge_union_fields(branches, require_all=True)

        assert not result.is_observed, "Mixed branches should not return observed"
        assert result.fields is not None
        field_names = {fd.name for fd in result.fields}
        assert "z" in field_names, "Field 'z' from explicit branch should be preserved"


class TestPartialArrivalNullableSoundness:
    """Tests for nullable soundness under partial-arrival scenarios.

    Under best_effort/quorum policies, branches may timeout. The schema must be
    sound for ALL possible arrival combinations, not just the collision_policy
    winner. If ANY branch that might arrive can produce None, the merged schema
    must be nullable.

    Bug scenario (P1):
    - Branch a: x: int? (nullable=True, processed first)
    - Branch b: x: int (nullable=False, processed last)
    - policy=best_effort, collision_policy=last_wins
    - Current code synthesizes x as non-nullable (uses b's nullable flag)
    - But if b times out, a delivers x=None — downstream schema is unsound!
    """

    def test_best_effort_last_wins_nullable_first_must_be_nullable(self) -> None:
        """Under best_effort, even if last branch is non-nullable, any branch might arrive.

        With last_wins, if the 'winning' last branch times out, the nullable
        first branch delivers the value. The schema must allow None.
        """
        # Branch a (first): nullable=True
        # Branch b (last): nullable=False
        # Under best_effort, either might arrive. Schema must be nullable.
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=False,  # best_effort: branches may not arrive
            collision_policy="last_wins",
            branch_order=["a", "b"],  # a is first, b is last
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        # Under partial-arrival, ANY branch might be the one that delivers.
        # If a delivers (b timed out), x can be None. Schema must reflect this.
        assert x_field.nullable is True, (
            "Under best_effort with last_wins, if nullable branch might arrive, "
            "merged schema must be nullable regardless of 'winner' declaration"
        )

    def test_best_effort_first_wins_nullable_last_must_be_nullable(self) -> None:
        """Under best_effort, even if first branch is non-nullable, any branch might arrive.

        With first_wins, if the 'winning' first branch times out, the nullable
        last branch delivers the value. The schema must allow None.
        """
        # Branch a (first): nullable=False
        # Branch b (last): nullable=True
        # Under best_effort, either might arrive. Schema must be nullable.
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=False,  # best_effort: branches may not arrive
            collision_policy="first_wins",
            branch_order=["a", "b"],  # a is first, b is last
        )

        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        # Under partial-arrival, ANY branch might be the one that delivers.
        # If b delivers (a timed out), x can be None. Schema must reflect this.
        assert x_field.nullable is True, (
            "Under best_effort with first_wins, if nullable branch might arrive, "
            "merged schema must be nullable regardless of 'winner' declaration"
        )

    def test_require_all_collision_policy_still_applies(self) -> None:
        """Under require_all, collision_policy correctly determines nullable.

        This is the existing correct behavior: all branches always arrive,
        so the configured winner is deterministic. first_wins/last_wins
        should use the winner's nullable flag.
        """
        branch_a = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=True),),
        )
        branch_b = SchemaConfig(
            mode="flexible",
            fields=(FieldDefinition(name="x", field_type="int", required=True, nullable=False),),
        )

        # With require_all + last_wins, b always wins, so non-nullable
        merged = merge_union_fields(
            {"a": branch_a, "b": branch_b},
            require_all=True,
            collision_policy="last_wins",
            branch_order=["a", "b"],
        )
        assert merged.fields is not None
        x_field = next(f for f in merged.fields if f.name == "x")
        assert x_field.nullable is False, "Under require_all with last_wins, last branch always wins"
