# tests/fixtures/base_classes.py
"""Protocol-compliant test base classes.

Single canonical definition of base classes for test sources, sinks,
and transforms. Migrated from tests/conftest.py with no behavioral changes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any, cast

from elspeth.contracts import (
    SINK_EFFECT_PROTOCOL_VERSION,
    CallType,
    Determinism,
    PluginSchema,
    ResolvedSinkEffectMode,
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectExecutionPurpose,
    SinkEffectInputKind,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
    SourceRow,
)
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import canonical_json, stable_hash
from elspeth.plugins.infrastructure.base import BaseTransform

if TYPE_CHECKING:
    from elspeth.contracts import (
        BatchTransformProtocol,
        SinkProtocol,
        SourceProtocol,
        TransformProtocol,
        TransformResult,
    )
    from elspeth.contracts.schema_contract import SchemaContract


class _TestSchema(PluginSchema):
    """Minimal schema for test fixtures."""

    pass


class _TestSourceBase:
    """Base class for test sources implementing SourceProtocol.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide: name, output_schema, load(ctx).
    """

    name: str
    output_schema: type[PluginSchema]
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    _on_validation_failure: str = "discard"
    on_success: str = "default"
    declared_guaranteed_fields: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        self._schema_contract: SchemaContract | None = None

    def wrap_rows(self, rows: list[dict[str, Any]]) -> Iterator[SourceRow]:
        """Wrap plain dicts in SourceRow.valid() as required by source protocol."""
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        for source_row_index, row in enumerate(rows):
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            if self._schema_contract is None:
                self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def close(self) -> None:
        pass

    def get_field_resolution(self) -> tuple[Mapping[str, str], str | None] | None:
        return None

    def get_schema_contract(self) -> SchemaContract | None:
        return self._schema_contract


class CallbackSource(_TestSourceBase):
    """Source with callbacks for deterministic MockClock testing.

    Enables tests to advance a MockClock between row yields.
    """

    name: str = "callback_source"
    output_schema: type[PluginSchema] = _TestSchema

    def __init__(
        self,
        rows: list[dict[str, Any]],
        output_schema: type[PluginSchema] | None = None,
        after_yield_callback: Callable[[int], None] | None = None,
        source_name: str = "callback_source",
        on_success: str = "default",
    ) -> None:
        super().__init__()
        self._rows = rows
        self._after_yield_callback = after_yield_callback
        self.name = source_name
        self.on_success = on_success
        if output_schema is not None:
            self.output_schema = output_schema

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        for i, row in enumerate(self._rows):
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            yield SourceRow.valid(row, contract=contract, source_row_index=i)
            if self._after_yield_callback is not None:
                self._after_yield_callback(i)


class _TestSinkBase:
    """Base class for test sinks implementing SinkProtocol."""

    name: str
    input_schema: type[PluginSchema] = _TestSchema
    idempotent: bool = True
    node_id: str | None = None
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = None
    declared_required_fields: frozenset[str] = frozenset()
    _on_write_failure: str | None = "discard"
    supports_resume: bool = False
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    effect_call_type = CallType.FILESYSTEM
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})

    def __init__(self) -> None:
        self.config: dict[str, Any] = {"schema": {"mode": "observed"}}
        self._diversion_log: list[Any] = []
        self._test_effect_plans: dict[str, SinkEffectPlan] = {}
        self._test_effect_rows: dict[str, tuple[dict[str, Any], ...]] = {}
        self._test_effect_ordinals: dict[str, tuple[int, ...]] = {}
        self._test_effect_commits: dict[str, SinkEffectCommitResult] = {}

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: Mapping[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode:
        del cls, config, purpose
        return ResolvedSinkEffectMode("write")

    def _reset_diversion_log(self) -> None:
        self._diversion_log = []

    def _get_diversions(self) -> tuple[RowDiversion, ...]:
        return tuple(self._diversion_log)

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del ctx
        committed = self._test_effect_commits.get(request.effect_id)
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference=request.target,
            evidence={"state": "committed" if committed is not None else "not_applied"},
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        if not isinstance(request.effect_input, SinkEffectPipelineMembersInput):
            raise TypeError("test sinks only support pipeline member effects")
        rows = tuple(deep_thaw(member.row) for member in request.effect_input.members)
        if any(type(row) is not dict for row in rows):
            raise TypeError("test sink effect rows must thaw to exact dictionaries")
        ordinals = tuple(member.ordinal for member in request.effect_input.members)
        payload_hash = sha256(canonical_json(rows).encode("utf-8")).hexdigest()
        plan = SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED,
            inspection_mode=request.inspection.mode,
            target=request.inspection.reference,
            plan_hash=stable_hash(
                {
                    "effect_id": request.effect_id,
                    "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
                    "payload_hash": payload_hash,
                    "target": request.inspection.reference,
                    "test_adapter": type(self).__qualname__,
                }
            ),
            payload_hash=payload_hash,
            expected_descriptor=None,
            safe_evidence={"sink_kind": "test_result_adapter"},
        )
        request.validate_plan(plan)
        existing = self._test_effect_plans.get(request.effect_id)
        if existing is not None and existing != plan:
            raise ValueError("test sink effect id was prepared with a different plan")
        self._test_effect_plans[request.effect_id] = plan
        self._test_effect_rows[request.effect_id] = rows
        self._test_effect_ordinals[request.effect_id] = ordinals
        return plan

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        prepared = self._test_effect_plans.get(plan.effect_id)
        if prepared is None or prepared != plan:
            raise ValueError("test sink can only commit the exact prepared plan")
        committed = self._test_effect_commits.get(plan.effect_id)
        if committed is not None:
            return committed
        result = self.write(list(self._test_effect_rows[plan.effect_id]), ctx)
        if not isinstance(result, SinkWriteResult):
            raise TypeError("test sink write must return SinkWriteResult")
        ordinals = self._test_effect_ordinals[plan.effect_id]
        diverted_indexes = {item.row_index for item in result.diversions}
        if any(index >= len(ordinals) for index in diverted_indexes):
            raise ValueError("test sink diversion index is outside the effect batch")
        diverted_ordinals = tuple(ordinals[index] for index in sorted(diverted_indexes))
        diverted_set = set(diverted_ordinals)
        accepted_ordinals = tuple(ordinal for ordinal in ordinals if ordinal not in diverted_set)
        self._diversion_log = [
            RowDiversion(
                row_index=ordinals[item.row_index],
                reason=item.reason,
                row_data=dict(item.row_data),
            )
            for item in result.diversions
        ]
        committed = SinkEffectCommitResult(
            descriptor=result.artifact,
            evidence={
                "accepted_ordinals": list(accepted_ordinals),
                "descriptor": {
                    "artifact_type": result.artifact.artifact_type,
                    "content_hash": result.artifact.content_hash,
                    "metadata": None if result.artifact.metadata is None else deep_thaw(result.artifact.metadata),
                    "path_or_uri": result.artifact.path_or_uri,
                    "size_bytes": result.artifact.size_bytes,
                },
                "diverted_ordinals": list(diverted_ordinals),
            },
            accepted_ordinals=accepted_ordinals,
            diverted_ordinals=diverted_ordinals,
        )
        self._test_effect_commits[plan.effect_id] = committed
        return committed

    def reconcile_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectReconcileResult:
        del ctx
        committed = self._test_effect_commits.get(plan.effect_id)
        if committed is None:
            return SinkEffectReconcileResult.not_applied(evidence={"state": "not_applied"})
        return SinkEffectReconcileResult.applied(
            committed.descriptor,
            evidence=committed.evidence,
            accepted_ordinals=committed.accepted_ordinals,
            diverted_ordinals=committed.diverted_ordinals,
        )

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def configure_for_resume(self) -> None:
        raise NotImplementedError("Test sinks do not support resume")

    def validate_output_target(self) -> Any:
        from elspeth.contracts.sink import OutputValidationResult

        return OutputValidationResult.success()

    @property
    def needs_resume_field_resolution(self) -> bool:
        return False

    def set_resume_field_resolution(self, resolution_mapping: dict[str, str]) -> None:
        pass

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> None:
        return None

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Minimal schema stub — production sinks return a JSON schema,
        test fixtures only need the attribute to satisfy SinkProtocol.
        """
        return {"type": "object", "additionalProperties": True}


class _TestTransformBase(BaseTransform):
    """Base class for test transforms inheriting production BaseTransform.

    Inherits lifecycle methods (on_start, on_complete, close) and the
    _on_start_called lifecycle guard from BaseTransform. This ensures test
    transforms automatically track any future BaseTransform attributes,
    preventing silent drift between test fixtures and production code.
    """

    name: str
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    plugin_version = "1.0.0"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})


# ─────────────────────────────────────────────────────────────────────────
# Type Cast Helpers
# ─────────────────────────────────────────────────────────────────────────


def as_source(source: Any) -> SourceProtocol:
    """Cast a test source to SourceProtocol."""
    return cast("SourceProtocol", source)


def as_transform(transform: Any) -> TransformProtocol:
    """Cast a test transform to TransformProtocol."""
    return cast("TransformProtocol", transform)


def as_batch_transform(transform: Any) -> BatchTransformProtocol:
    """Cast a test batch transform to BatchTransformProtocol."""
    return cast("BatchTransformProtocol", transform)


def inject_write_failure[S](sink: S, value: str = "discard") -> S:
    """Inject _on_write_failure on a production sink instance.

    Production code injects this via cli_helpers from SinkSettings.
    Tests that construct sinks directly bypass that path. Call this
    on any production sink (CSVSink, JSONSink, etc.) after construction.

    Returns the same sink for call-chaining.
    """
    # Access via Any — S is always a concrete sink with _on_write_failure,
    # but the generic type parameter can't express the SinkProtocol bound.
    s: Any = sink
    if s._on_write_failure is None:
        s._on_write_failure = value
    return sink


def as_sink(sink: Any) -> SinkProtocol:
    """Cast a test sink to SinkProtocol.

    Also ensures _on_write_failure is set if not already — production code
    injects this via cli_helpers, but tests that construct sinks directly
    bypass that path.
    """
    inject_write_failure(sink)
    return cast("SinkProtocol", sink)


def create_observed_contract(row: dict[str, Any]) -> SchemaContract:
    """Create an OBSERVED schema contract from a row."""
    from elspeth.contracts.schema_contract import FieldContract, SchemaContract

    fields = tuple(
        FieldContract(
            normalized_name=key,
            original_name=key,
            python_type=object,
            required=False,
            source="inferred",
        )
        for key in row
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def as_transform_result(result: Any) -> TransformResult:
    """Assert and cast a result to TransformResult."""
    from elspeth.engine.batch_adapter import ExceptionResult

    if isinstance(result, ExceptionResult):
        raise AssertionError(f"Expected TransformResult but got ExceptionResult: {result.exception}\n{result.traceback}")
    return cast("TransformResult", result)
