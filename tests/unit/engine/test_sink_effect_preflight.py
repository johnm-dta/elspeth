"""Local/declarative sink-effect capability preflight."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from elspeth.cli import _preflight_follower_sink_effects
from elspeth.contracts.sink_effects import SINK_EFFECT_PROTOCOL_VERSION, SinkEffectInputKind
from elspeth.engine.orchestrator.export import export_landscape
from elspeth.engine.orchestrator.preflight import (
    SinkEffectCapabilityError,
    assemble_and_validate_pipeline_config,
    validate_pipeline_sink_effect_capabilities,
    validate_sink_effect_capability,
)
from elspeth.engine.orchestrator.run_context_factory import RunContextFactory


class LegacyObservableSink:
    name = "legacy"

    def __init__(self) -> None:
        self.config = {"mode": "write"}
        self.on_start_calls = 0
        self.write_calls = 0

    def on_start(self, _ctx: object) -> None:
        self.on_start_calls += 1

    def write(self, _rows: object, _ctx: object) -> None:
        self.write_calls += 1


class EffectCapableSink(LegacyObservableSink):
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"write", "append", "overwrite", "conditional_put", "etag_guarded_upload"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS, SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})

    def __init__(self) -> None:
        super().__init__()
        self.effect_mode = "write"

    def inspect_effect(self, _request: object, _ctx: object) -> None:
        return None

    def prepare_effect(self, _request: object, _ctx: object) -> None:
        return None

    def commit_effect(self, _plan: object, _ctx: object) -> None:
        return None

    def reconcile_effect(self, _plan: object, _ctx: object) -> None:
        return None


def test_preflight_rejects_legacy_sink_before_lifecycle_or_io() -> None:
    sink = LegacyObservableSink()

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        validate_sink_effect_capability(
            sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_preflight_accepts_exact_protocol_declared_mode_and_callables() -> None:
    sink = EffectCapableSink()

    validate_sink_effect_capability(
        sink,
        mode="write",
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_mode_and_input_kind_capabilities_are_independent_and_inverse() -> None:
    pipeline_type = type(
        "PipelineOnlySink",
        (EffectCapableSink,),
        {"supported_effect_input_kinds": frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})},
    )
    export_type = type(
        "ExportOnlySink",
        (EffectCapableSink,),
        {"supported_effect_input_kinds": frozenset({SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})},
    )
    pipeline_sink = pipeline_type()
    export_sink = export_type()

    validate_sink_effect_capability(
        pipeline_sink,
        mode="write",
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )
    with pytest.raises(SinkEffectCapabilityError, match="audit_export_snapshot"):
        validate_sink_effect_capability(
            pipeline_sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        )

    validate_sink_effect_capability(
        export_sink,
        mode="write",
        required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
    )
    with pytest.raises(SinkEffectCapabilityError, match="pipeline_members"):
        validate_sink_effect_capability(
            export_sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )

    assert pipeline_sink.on_start_calls == export_sink.on_start_calls == 0
    assert pipeline_sink.write_calls == export_sink.write_calls == 0


def test_preflight_rejects_non_enum_required_input_kind() -> None:
    with pytest.raises(SinkEffectCapabilityError, match="required input kind"):
        validate_sink_effect_capability(
            EffectCapableSink(),
            mode="write",
            required_input_kind="pipeline_members",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("effect_protocol_version", "sink-effect-v0", "effect protocol"),
        ("supported_effect_modes", frozenset(), "supported effect mode"),
        ("supported_effect_modes", frozenset({"  "}), "supported effect mode"),
        ("supported_effect_modes", frozenset({"append"}), "mode 'write'"),
        ("supported_effect_modes", {"write"}, "frozenset"),
        ("supported_effect_input_kinds", frozenset(), "supported effect input kind"),
        ("supported_effect_input_kinds", frozenset({"pipeline_members"}), "SinkEffectInputKind"),
        ("supported_effect_input_kinds", {SinkEffectInputKind.PIPELINE_MEMBERS}, "frozenset"),
        ("commit_effect", None, "commit_effect"),
    ],
)
def test_preflight_fails_closed_on_inexact_declarations(
    attribute: str,
    value: object,
    message: str,
) -> None:
    sink_type = type("InvalidEffectCapableSink", (EffectCapableSink,), {attribute: value})
    sink = sink_type()

    with pytest.raises(SinkEffectCapabilityError, match=message):
        validate_sink_effect_capability(
            sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_preflight_requires_class_level_protocol_opt_in() -> None:
    sink = LegacyObservableSink()
    sink.effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    sink.supported_effect_modes = frozenset({"write"})
    sink.supported_effect_input_kinds = frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})
    sink.inspect_effect = lambda _request, _ctx: None
    sink.prepare_effect = lambda _request, _ctx: None
    sink.commit_effect = lambda _plan, _ctx: None
    sink.reconcile_effect = lambda _plan, _ctx: None

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        validate_sink_effect_capability(
            sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


@pytest.mark.parametrize(
    ("config", "effect_mode"),
    [
        ({"mode": "persistent", "on_duplicate": "overwrite"}, "overwrite"),
        ({"if_exists": "append"}, "append"),
        ({"mode": "append"}, "append"),
        ({"overwrite": False}, "conditional_put"),
        ({"overwrite": True}, "etag_guarded_upload"),
    ],
    ids=["chroma", "database", "local", "s3", "azure"],
)
def test_collection_preflight_uses_explicit_resolved_effect_mode(
    config: dict[str, object],
    effect_mode: str,
) -> None:
    sink = EffectCapableSink()
    sink.config = config
    sink.effect_mode = effect_mode

    validate_pipeline_sink_effect_capabilities(
        {"output": sink},  # type: ignore[dict-item]
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )


def test_collection_preflight_does_not_infer_chroma_connection_mode() -> None:
    sink = EffectCapableSink()
    sink.config = {"mode": "persistent", "on_duplicate": "overwrite"}
    del sink.effect_mode

    with pytest.raises(SinkEffectCapabilityError, match="resolved effect_mode"):
        validate_pipeline_sink_effect_capabilities(
            {"output": sink},  # type: ignore[dict-item]
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


@pytest.mark.parametrize("effect_mode", [None, ("write", "append"), "", "  "])
def test_collection_preflight_rejects_absent_or_ambiguous_effect_mode(effect_mode: object) -> None:
    sink = EffectCapableSink()
    sink.effect_mode = effect_mode  # type: ignore[assignment]

    with pytest.raises(SinkEffectCapabilityError, match="resolved effect_mode"):
        validate_pipeline_sink_effect_capabilities(
            {"output": sink},  # type: ignore[dict-item]
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


def test_non_run_pipeline_assembly_does_not_enforce_effect_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = LegacyObservableSink()
    resolve_calls = 0

    def resolve(_settings: object) -> dict[str, Any]:
        nonlocal resolve_calls
        resolve_calls += 1
        return {}

    monkeypatch.setattr("elspeth.engine.orchestrator.preflight.resolve_config", resolve)
    monkeypatch.setattr("elspeth.engine.orchestrator.preflight.validate_pipeline_route_targets", lambda **_kwargs: None)

    settings = SimpleNamespace(
        gates=(),
        coalesce=(),
    )
    graph = SimpleNamespace(
        get_aggregation_id_map=lambda: {},
        get_route_resolution_map=lambda: {},
        get_transform_id_map=lambda: {},
        get_config_gate_id_map=lambda: {},
    )

    pipeline_config = assemble_and_validate_pipeline_config(
        sources={"source": object()},  # type: ignore[dict-item]
        transforms=(),
        sinks={"output": sink},  # type: ignore[dict-item]
        aggregations={},
        settings=settings,  # type: ignore[arg-type]
        graph=graph,  # type: ignore[arg-type]
    )

    assert pipeline_config.sinks["output"] is sink
    assert resolve_calls == 1
    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


@pytest.mark.parametrize("include_source_on_start", [True, False], ids=["fresh", "resume"])
def test_run_context_preflights_effects_before_any_fresh_or_resume_setup(include_source_on_start: bool) -> None:
    sink = LegacyObservableSink()
    context_factory = object.__new__(RunContextFactory)
    config = SimpleNamespace(sinks={"output": sink})

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        context_factory.initialize_run_context(
            factory=None,  # type: ignore[arg-type]
            run_id="run-1",
            config=config,  # type: ignore[arg-type]
            graph=None,  # type: ignore[arg-type]
            settings=None,
            artifacts=None,  # type: ignore[arg-type]
            payload_store=None,  # type: ignore[arg-type]
            include_source_on_start=include_source_on_start,
        )

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_follower_preflight_rejects_before_plugin_lifecycle_or_io() -> None:
    sink = LegacyObservableSink()

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        _preflight_follower_sink_effects({"output": sink})  # type: ignore[dict-item]

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_follower_requires_pipeline_input_kind_and_rejects_export_only_sink() -> None:
    export_type = type(
        "ExportOnlySink",
        (EffectCapableSink,),
        {"supported_effect_input_kinds": frozenset({SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})},
    )
    sink = export_type()
    with pytest.raises(SinkEffectCapabilityError, match="pipeline_members"):
        _preflight_follower_sink_effects({"output": sink})  # type: ignore[dict-item]
    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_runtime_entry_points_construct_plugins_in_preflight_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.cli import _instantiate_plugins_for_runtime_preflight
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode_enabled

    observed: list[tuple[str, bool]] = []
    unsafe_side_effects: list[str] = []

    def record_constructor(component: str) -> None:
        safe = plugin_preflight_mode_enabled()
        observed.append((component, safe))
        if not safe:
            unsafe_side_effects.append(f"{component}:credential-client-initialized")

    class ConstructorProbeSource:
        def __init__(self, _config: dict[str, object]) -> None:
            record_constructor("source")

    class ConstructorProbeSink:
        def __init__(self, _config: dict[str, object]) -> None:
            record_constructor("sink")

    class PluginManager:
        def get_source_by_name(self, _name: str) -> type[ConstructorProbeSource]:
            return ConstructorProbeSource

        def get_sink_by_name(self, _name: str) -> type[ConstructorProbeSink]:
            return ConstructorProbeSink

    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: PluginManager(),
    )
    component = SimpleNamespace(plugin="probe", options={}, on_success="continue")
    settings = SimpleNamespace(
        sources={"source": component},
        transforms=(),
        aggregations=(),
        sinks={
            "output": SimpleNamespace(
                plugin="probe",
                options={},
                on_write_failure="fail",
            )
        },
    )

    bundle = _instantiate_plugins_for_runtime_preflight(settings)  # type: ignore[arg-type]

    assert bundle.sources
    assert bundle.sinks
    assert observed == [("source", True), ("sink", True)]
    assert unsafe_side_effects == []


def test_audit_export_factory_constructs_sink_in_preflight_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode_enabled
    from elspeth.plugins.infrastructure.runtime_factory import make_sink_factory

    observed: list[bool] = []
    unsafe_side_effects: list[str] = []

    class ConstructorProbeSink:
        def __init__(self, config: dict[str, object]) -> None:
            safe = plugin_preflight_mode_enabled()
            observed.append(safe)
            if not safe:
                unsafe_side_effects.append("credential-client-initialized")
            self.config = config

    class PluginManager:
        def get_sink_by_name(self, _name: str) -> type[ConstructorProbeSink]:
            return ConstructorProbeSink

    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: PluginManager(),
    )
    settings = SimpleNamespace(
        sinks={
            "audit-output": SimpleNamespace(
                plugin="probe",
                options={},
                on_write_failure="fail",
            )
        }
    )

    sink = make_sink_factory(settings)("audit-output")  # type: ignore[arg-type]

    assert sink.config == {}
    assert observed == [True]
    assert unsafe_side_effects == []


def test_audit_export_preflights_fresh_sink_before_node_or_lifecycle_or_io() -> None:
    sink = LegacyObservableSink()
    export_settings = SimpleNamespace(
        landscape=SimpleNamespace(
            export=SimpleNamespace(
                sign=False,
                include_raw_error_rows=False,
                sink="audit-output",
                format="json",
            )
        )
    )

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        export_landscape(
            db=None,  # type: ignore[arg-type]
            run_id="run-1",
            settings=export_settings,  # type: ignore[arg-type]
            sink_factory=lambda _name: sink,  # type: ignore[arg-type,return-value]
        )

    assert "node_id" not in vars(sink)
    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_audit_export_requires_export_input_kind_and_rejects_pipeline_only_sink() -> None:
    pipeline_type = type(
        "PipelineOnlySink",
        (EffectCapableSink,),
        {"supported_effect_input_kinds": frozenset({SinkEffectInputKind.PIPELINE_MEMBERS})},
    )
    sink = pipeline_type()
    export_settings = SimpleNamespace(
        landscape=SimpleNamespace(
            export=SimpleNamespace(
                sign=False,
                include_raw_error_rows=False,
                sink="audit-output",
                format="json",
            )
        )
    )
    with pytest.raises(SinkEffectCapabilityError, match="audit_export_snapshot"):
        export_landscape(
            db=None,  # type: ignore[arg-type]
            run_id="run-1",
            settings=export_settings,  # type: ignore[arg-type]
            sink_factory=lambda _name: sink,  # type: ignore[arg-type,return-value]
        )
    assert "node_id" not in vars(sink)
    assert sink.on_start_calls == 0
    assert sink.write_calls == 0
