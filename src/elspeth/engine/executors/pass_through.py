"""Runtime verification of ``passes_through_input=True`` declarations.

ADR-008 introduced the runtime cross-check; ADR-009 §Clause 2 extracted the
verifier so single-token and batch-flush paths share one implementation;
ADR-010 §Decision 3 made the dispatcher the single call site for
declaration-contract dispatch; the ADR-010 §Semantics amendment (2026-04-20)
restructured dispatch into the 4-site audit-complete framework.

This contract registers for TWO dispatch sites:

    * ``post_emission_check`` — single-token path from ``TransformExecutor``.
    * ``batch_flush_check``   — batch-homogeneous path from
                                ``RowProcessor._cross_check_flush_output``.

Both share the ``verify_pass_through`` implementation. The site difference is
the SHAPE of the input bundle (``PostEmissionInputs`` carries a single
``input_row``; ``BatchFlushInputs`` carries a tuple of buffered tokens and a
pre-computed ``effective_input_fields`` intersection). Both sites'
decorated methods are thin adapters over ``verify_pass_through``.

Counter cardinality is bounded by the annotated transform set (short, known
at startup), so the ``transform=<name>`` tag is safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, TypedDict, cast

from opentelemetry import metrics

from elspeth.contracts.declaration_contracts import (
    BatchFlushInputs,
    BatchFlushOutputs,
    DeclarationContract,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    FrameworkBugError,
    OrchestrationInvariantError,
    PassThroughContractViolation,
)
from elspeth.contracts.schema_contract import PipelineRow

# Module-level counter — both call sites import this module and share the
# same instrument.
_VIOLATIONS_COUNTER = metrics.get_meter(__name__).create_counter(
    "pass_through_cross_check_violations_total",
    description="Count of passes_through_input=True transforms that dropped input fields at runtime",
)


def verify_pass_through(
    *,
    input_fields: frozenset[str],
    emitted_rows: Sequence[PipelineRow],
    can_drop_rows: bool,
    static_contract: frozenset[str],
    transform_name: str,
    transform_node_id: str,
    run_id: str,
    row_id: str,
    token_id: str,
) -> None:
    """Verify every emitted row preserves ``input_fields``.

    A field is "preserved" iff the emitted row's contract declares it AND its
    payload carries it. ``PipelineRow`` treats contract and payload as
    independent references; reading either alone leaves a one-sided blind
    spot, so runtime observation is the intersection of both.

    Raises ``PassThroughContractViolation`` on the first row that drops any
    input field (per-adopter semantics preserved from ADR-008 / ADR-009).
    Under audit-complete dispatch the violation is caught by the
    dispatcher's ``PluginContractViolation`` branch and aggregated with any
    other contract's raise on the same row.
    """
    if not emitted_rows:
        if can_drop_rows or not input_fields:
            return
        _VIOLATIONS_COUNTER.add(1, {"transform": transform_name})
        raise PassThroughContractViolation(
            transform=transform_name,
            transform_node_id=transform_node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            static_contract=static_contract,
            runtime_observed=frozenset(),
            divergence_set=input_fields,
            message=(
                f"Transform {transform_name!r} (node {transform_node_id!r}) "
                f"declared passes_through_input=True but emitted zero rows for "
                f"row {row_id!r}, dropping fields {sorted(input_fields)!r}."
            ),
        )

    for emitted in emitted_rows:
        if emitted.contract is None:
            raise FrameworkBugError(f"Transform {transform_name!r} emitted row with no contract. Framework invariant violated.")
        runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
        runtime_payload_fields = frozenset(emitted.keys())
        runtime_observed = runtime_contract_fields & runtime_payload_fields
        divergence = input_fields - runtime_observed
        if divergence:
            _VIOLATIONS_COUNTER.add(1, {"transform": transform_name})
            raise PassThroughContractViolation(
                transform=transform_name,
                transform_node_id=transform_node_id,
                run_id=run_id,
                row_id=row_id,
                token_id=token_id,
                static_contract=static_contract,
                runtime_observed=runtime_observed,
                divergence_set=frozenset(divergence),
                message=(
                    f"Transform {transform_name!r} (node {transform_node_id!r}) "
                    f"declared passes_through_input=True but dropped fields "
                    f"{sorted(divergence)!r} from row {row_id!r}."
                ),
            )


class PassThroughPayload(TypedDict):
    """Shape of PassThroughContractViolation.divergence_set projected into
    the DeclarationContractViolation payload.

    PassThroughContractViolation carries its own rich 9-key payload; this
    TypedDict is consulted when the contract is queried via
    ``payload_schema``. Under audit-complete aggregation, each child's
    ``to_audit_dict`` surfaces independently — this schema is for harness
    introspection only.
    """

    divergence_set: list[str]
    static_contract: list[str]
    runtime_observed: list[str]


class PassThroughDeclarationContract(DeclarationContract):
    """ADR-007/008/009 pass-through contract — ADR-010 framework adopter.

    Inherits the nominal ABC. Claims TWO dispatch sites via
    ``@implements_dispatch_site`` markers:
    ``post_emission_check`` (single-token path) and ``batch_flush_check``
    (batch-homogeneous intersection path). Both delegate to the shared
    ``verify_pass_through`` implementation.
    """

    name: ClassVar[str] = "passes_through_input"
    payload_schema: ClassVar[type] = PassThroughPayload

    def applies_to(self, plugin: Any) -> bool:
        # Direct attribute access, NOT getattr with default (CLAUDE.md
        # §Offensive Programming). A plugin missing passes_through_input
        # is a framework bug — let it crash.
        return cast(bool, plugin.passes_through_input)

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(
        self,
        inputs: PostEmissionInputs,
        outputs: PostEmissionOutputs,
    ) -> None:
        """Single-token path (TransformExecutor).

        ``inputs.effective_input_fields`` is caller-derived from
        ``input_row.contract.fields`` — contracts do NOT re-derive
        (panel F1 resolution).
        """
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {inputs.plugin.name!r} has no node_id set at cross-check time.")
        verify_pass_through(
            input_fields=inputs.effective_input_fields,
            emitted_rows=outputs.emitted_rows,
            can_drop_rows=inputs.plugin.can_drop_rows,
            static_contract=inputs.static_contract,
            transform_name=inputs.plugin.name,
            transform_node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    @implements_dispatch_site("batch_flush_check")
    def batch_flush_check(
        self,
        inputs: BatchFlushInputs,
        outputs: BatchFlushOutputs,
    ) -> None:
        """Batch-flush TRANSFORM mode (ADR-009 §Clause 2).

        ``inputs.effective_input_fields`` is the caller-computed INTERSECTION
        of every buffered token's contract — the weakest shared guarantee
        every emitted row must preserve.
        """
        transform_node_id = inputs.plugin.node_id
        if transform_node_id is None:
            raise OrchestrationInvariantError(f"Transform {inputs.plugin.name!r} has no node_id set at batch-flush cross-check time.")
        verify_pass_through(
            input_fields=inputs.effective_input_fields,
            emitted_rows=outputs.emitted_rows,
            can_drop_rows=inputs.plugin.can_drop_rows,
            static_contract=inputs.static_contract,
            transform_name=inputs.plugin.name,
            transform_node_id=transform_node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
        )

    # pre_emission_check and boundary_check fall through to the ABC's default
    # no-op bodies. The dispatcher never invokes them (no marker).

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        """Classmethod used by the invariant harness to prove the contract
        fires on a known-bad input.

        Returns a site-tagged ``ExampleBundle`` for POST_EMISSION — the
        single-token adopter site. The BATCH_FLUSH site delegates to the
        same verify_pass_through body so the harness's positive-case
        coverage is transitive.
        """
        from elspeth.contracts.schema_contract import (
            FieldContract,
            PipelineRow,
            SchemaContract,
        )

        def _row(fields: tuple[str, ...]) -> PipelineRow:
            c = SchemaContract(
                mode="OBSERVED",
                fields=tuple(
                    FieldContract(
                        normalized_name=n,
                        original_name=n,
                        python_type=str,
                        required=True,
                        source="inferred",
                        nullable=False,
                    )
                    for n in fields
                ),
                locked=True,
            )
            return PipelineRow(dict.fromkeys(fields, "v"), c)

        class _MinimalTransform:
            name = "NegativeExample"
            node_id = "neg-1"
            passes_through_input = True
            can_drop_rows = False
            _output_schema_config = None

        inputs = PostEmissionInputs(
            plugin=_MinimalTransform(),
            node_id="neg-1",
            run_id="neg-run",
            row_id="neg-row",
            token_id="neg-token",
            input_row=_row(("a", "b", "c")),
            static_contract=frozenset({"a", "b", "c"}),
            effective_input_fields=frozenset({"a", "b", "c"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("a", "c")),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        """N2 Layer A non-fire scenario — ``passes_through_input=False``
        plugin. ``applies_to`` must return False; even if the dispatcher
        were to invoke ``post_emission_check`` anyway (impossible under the
        registry's filter, but belt-and-suspenders), the rows preserve every
        input field so no violation raises.
        """
        from elspeth.contracts.schema_contract import (
            FieldContract,
            PipelineRow,
            SchemaContract,
        )

        def _row(fields: tuple[str, ...]) -> PipelineRow:
            c = SchemaContract(
                mode="OBSERVED",
                fields=tuple(
                    FieldContract(
                        normalized_name=n,
                        original_name=n,
                        python_type=str,
                        required=True,
                        source="inferred",
                        nullable=False,
                    )
                    for n in fields
                ),
                locked=True,
            )
            return PipelineRow(dict.fromkeys(fields, "v"), c)

        class _NonPassThroughTransform:
            name = "NonFireExample"
            node_id = "non-fire-1"
            passes_through_input = False  # the discriminator applies_to reads
            can_drop_rows = False
            _output_schema_config = None

        inputs = PostEmissionInputs(
            plugin=_NonPassThroughTransform(),
            node_id="non-fire-1",
            run_id="non-fire-run",
            row_id="non-fire-row",
            token_id="non-fire-token",
            input_row=_row(("a", "b", "c")),
            static_contract=frozenset({"a", "b", "c"}),
            effective_input_fields=frozenset({"a", "b", "c"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("a", "b", "c")),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def negative_example_batch_flush(cls) -> ExampleBundle:
        """Batch-flush negative-example companion for the shared invariant harness."""
        from elspeth.contracts.declaration_contracts import BatchFlushInputs, BatchFlushOutputs
        from elspeth.contracts.schema_contract import (
            FieldContract,
            PipelineRow,
            SchemaContract,
        )

        def _row(fields: tuple[str, ...]) -> PipelineRow:
            c = SchemaContract(
                mode="OBSERVED",
                fields=tuple(
                    FieldContract(
                        normalized_name=n,
                        original_name=n,
                        python_type=str,
                        required=True,
                        source="inferred",
                        nullable=False,
                    )
                    for n in fields
                ),
                locked=True,
            )
            return PipelineRow(dict.fromkeys(fields, "v"), c)

        class _MinimalTransform:
            name = "NegativeBatchFlushExample"
            node_id = "neg-batch-1"
            passes_through_input = True
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            _output_schema_config = None

        token_row = _row(("a", "b", "c"))
        inputs = BatchFlushInputs(
            plugin=_MinimalTransform(),
            node_id="neg-batch-1",
            run_id="neg-batch-run",
            row_id="neg-batch-row",
            token_id="neg-batch-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"a", "b", "c"}),
            effective_input_fields=frozenset({"a", "b", "c"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_row(("a", "c")),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply_batch_flush(cls) -> ExampleBundle:
        """Batch-flush non-fire companion for the shared invariant harness."""
        from elspeth.contracts.declaration_contracts import BatchFlushInputs, BatchFlushOutputs
        from elspeth.contracts.schema_contract import (
            FieldContract,
            PipelineRow,
            SchemaContract,
        )

        def _row(fields: tuple[str, ...]) -> PipelineRow:
            c = SchemaContract(
                mode="OBSERVED",
                fields=tuple(
                    FieldContract(
                        normalized_name=n,
                        original_name=n,
                        python_type=str,
                        required=True,
                        source="inferred",
                        nullable=False,
                    )
                    for n in fields
                ),
                locked=True,
            )
            return PipelineRow(dict.fromkeys(fields, "v"), c)

        class _NonPassThroughTransform:
            name = "NonFireBatchFlushExample"
            node_id = "non-fire-batch-1"
            passes_through_input = False
            can_drop_rows = False
            declared_output_fields: frozenset[str] = frozenset()
            _output_schema_config = None

        token_row = _row(("x", "y"))
        inputs = BatchFlushInputs(
            plugin=_NonPassThroughTransform(),
            node_id="non-fire-batch-1",
            run_id="non-fire-batch-run",
            row_id="non-fire-batch-row",
            token_id="non-fire-batch-token",
            buffered_tokens=(token_row,),
            static_contract=frozenset({"x", "y"}),
            effective_input_fields=frozenset({"x", "y"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_row(("x", "y")),))
        return ExampleBundle(site=DispatchSite.BATCH_FLUSH, args=(inputs, outputs))


# Module import side-effect: register with the framework. Bootstrap asserts
# the registration actually happened (orchestrator.core.prepare_for_run).
register_declaration_contract(PassThroughDeclarationContract())
