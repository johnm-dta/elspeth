"""Local/declarative sink-effect capability preflight."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from elspeth.cli import _join_after_follower_sink_preflight, _preflight_follower_sink_effects
from elspeth.contracts.sink_effects import SINK_EFFECT_PROTOCOL_VERSION, SinkEffectInputKind
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.export import export_landscape
from elspeth.engine.orchestrator.preflight import (
    SinkEffectCapabilityError,
    assemble_and_validate_pipeline_config,
    execution_sinks_for_runtime,
    require_sink_effect_admission,
    validate_pipeline_sink_effect_capabilities,
    validate_sink_effect_capability,
)


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


@pytest.mark.parametrize("method_name", ["inspect_effect", "prepare_effect", "commit_effect", "reconcile_effect"])
def test_preflight_rejects_instance_shadowed_effect_method(method_name: str) -> None:
    sink = EffectCapableSink()
    setattr(sink, method_name, None)
    with pytest.raises(SinkEffectCapabilityError, match=method_name):
        validate_sink_effect_capability(
            sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


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

    validate_pipeline_sink_effect_capabilities(
        {"output": sink},  # type: ignore[dict-item]
        configured_modes={"output": effect_mode},
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )


def test_collection_preflight_does_not_infer_chroma_connection_mode() -> None:
    sink = EffectCapableSink()
    sink.config = {"mode": "persistent", "on_duplicate": "overwrite"}

    with pytest.raises(SinkEffectCapabilityError, match="configured effect mode"):
        validate_pipeline_sink_effect_capabilities(
            {"output": sink},  # type: ignore[dict-item]
            configured_modes={},
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


@pytest.mark.parametrize("effect_mode", [None, ("write", "append"), "", "  "])
def test_collection_preflight_rejects_absent_or_ambiguous_effect_mode(effect_mode: object) -> None:
    sink = EffectCapableSink()

    with pytest.raises(SinkEffectCapabilityError, match="configured effect mode"):
        validate_pipeline_sink_effect_capabilities(
            {"output": sink},  # type: ignore[dict-item]
            configured_modes={"output": effect_mode},  # type: ignore[dict-item]
            required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        )


def test_execution_sink_mapping_excludes_enabled_export_sink_once() -> None:
    pipeline_sink = EffectCapableSink()
    export_sink = EffectCapableSink()
    settings = SimpleNamespace(landscape=SimpleNamespace(export=SimpleNamespace(enabled=True, sink="audit-export")))
    selected = execution_sinks_for_runtime(
        settings,  # type: ignore[arg-type]
        {"pipeline": pipeline_sink, "audit-export": export_sink},  # type: ignore[dict-item]
    )
    assert selected == {"pipeline": pipeline_sink}


def test_exact_admission_receipt_skips_duplicate_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = EffectCapableSink()
    sinks = {"output": sink}
    modes = {"output": "write"}
    admission = validate_pipeline_sink_effect_capabilities(
        sinks,  # type: ignore[arg-type]
        configured_modes=modes,
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )
    monkeypatch.setattr(
        "elspeth.engine.orchestrator.preflight.validate_pipeline_sink_effect_capabilities",
        lambda *_args, **_kwargs: pytest.fail("receipt path must not revalidate"),
    )

    accepted = require_sink_effect_admission(
        sinks,  # type: ignore[arg-type]
        configured_modes=modes,
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        admission=admission,
    )

    assert accepted is admission


@pytest.mark.parametrize("mutation", ["instance", "mode", "name", "kind", "capability"])
def test_admission_receipt_rejects_changed_binding(mutation: str) -> None:
    sink = EffectCapableSink()
    sinks: dict[str, EffectCapableSink] = {"output": sink}
    modes = {"output": "write"}
    kind = SinkEffectInputKind.PIPELINE_MEMBERS
    admission = validate_pipeline_sink_effect_capabilities(
        sinks,  # type: ignore[arg-type]
        configured_modes=modes,
        required_input_kind=kind,
    )
    if mutation == "instance":
        sinks = {"output": EffectCapableSink()}
    elif mutation == "mode":
        modes = {"output": "append"}
    elif mutation == "name":
        sinks = {"renamed": sink}
        modes = {"renamed": "write"}
    elif mutation == "kind":
        kind = SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT
    else:
        sink.commit_effect = None  # type: ignore[assignment]

    with pytest.raises(SinkEffectCapabilityError, match="does not bind"):
        require_sink_effect_admission(
            sinks,  # type: ignore[arg-type]
            configured_modes=modes,
            required_input_kind=kind,
            admission=admission,
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
        landscape=SimpleNamespace(export=SimpleNamespace(enabled=False, sink=None)),
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


@pytest.mark.parametrize("operation", ["run", "resume"])
def test_direct_orchestrator_entry_rejects_before_fresh_or_resume_coordinator(operation: str) -> None:
    sink = LegacyObservableSink()
    orchestrator = object.__new__(Orchestrator)
    orchestrator._run_lifecycle = MagicMock()
    orchestrator._resume_coordinator = MagicMock()
    config = SimpleNamespace(sinks={"output": sink}, sink_effect_modes={}, sink_effect_admission=None)

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        if operation == "run":
            orchestrator.run(config, payload_store=object())  # type: ignore[arg-type]
        else:
            orchestrator.resume(object(), config, object(), payload_store=object())  # type: ignore[arg-type]

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0
    orchestrator._run_lifecycle.run.assert_not_called()
    orchestrator._resume_coordinator.resume.assert_not_called()


def test_follower_preflight_rejects_before_plugin_lifecycle_or_io() -> None:
    sink = LegacyObservableSink()

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        _preflight_follower_sink_effects({"output": sink}, {})  # type: ignore[dict-item]

    assert sink.on_start_calls == 0
    assert sink.write_calls == 0


def test_follower_rejection_never_calls_join_run(monkeypatch: pytest.MonkeyPatch) -> None:
    sink = LegacyObservableSink()
    plugins = SimpleNamespace(sinks={"output": sink}, sink_effect_modes={})
    orchestrator = SimpleNamespace(join_run=MagicMock())
    settings = SimpleNamespace(landscape=SimpleNamespace(export=SimpleNamespace(enabled=False, sink=None)))
    monkeypatch.setattr("elspeth.cli._instantiate_plugins_for_runtime_preflight", lambda _settings: plugins)

    with pytest.raises(SinkEffectCapabilityError, match="effect protocol"):
        _join_after_follower_sink_preflight(
            orchestrator,  # type: ignore[arg-type]
            "run-1",
            settings,  # type: ignore[arg-type]
        )

    orchestrator.join_run.assert_not_called()


def test_follower_requires_pipeline_input_kind_and_rejects_export_only_sink() -> None:
    export_type = type(
        "ExportOnlySink",
        (EffectCapableSink,),
        {"supported_effect_input_kinds": frozenset({SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})},
    )
    sink = export_type()
    with pytest.raises(SinkEffectCapabilityError, match="pipeline_members"):
        _preflight_follower_sink_effects({"output": sink}, {"output": "write"})  # type: ignore[dict-item]
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
        landscape=SimpleNamespace(export=SimpleNamespace(enabled=False, sink=None)),
    )

    bundle = _instantiate_plugins_for_runtime_preflight(settings)  # type: ignore[arg-type]

    assert bundle.sources
    assert bundle.sinks
    assert observed == [("source", True), ("sink", True)]
    assert unsafe_side_effects == []


def test_runtime_factory_does_not_construct_delayed_export_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.plugins.infrastructure.runtime_factory import instantiate_plugins_from_config

    constructed_sinks: list[str] = []

    class Source:
        def __init__(self, _config: dict[str, object]) -> None: ...

    class Sink:
        def __init__(self, config: dict[str, object]) -> None:
            constructed_sinks.append(str(config["marker"]))

    class PluginManager:
        def get_source_by_name(self, _name: str) -> type[Source]:
            return Source

        def get_sink_by_name(self, _name: str) -> type[Sink]:
            return Sink

    monkeypatch.setattr(
        "elspeth.plugins.infrastructure.manager.get_shared_plugin_manager",
        lambda: PluginManager(),
    )
    source_config = SimpleNamespace(plugin="source", options={}, on_success="continue")
    sink = lambda marker: SimpleNamespace(  # noqa: E731 - compact config fixture
        plugin="sink",
        options={"marker": marker},
        on_write_failure="fail",
    )
    settings = SimpleNamespace(
        sources={"source": source_config},
        transforms=(),
        aggregations=(),
        sinks={"pipeline": sink("pipeline"), "audit-export": sink("audit-export")},
        landscape=SimpleNamespace(export=SimpleNamespace(enabled=True, sink="audit-export")),
    )

    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)  # type: ignore[arg-type]

    assert set(bundle.sinks) == {"pipeline"}
    assert constructed_sinks == ["pipeline"]


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
    sink._resolved_effect_mode = "write"
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
