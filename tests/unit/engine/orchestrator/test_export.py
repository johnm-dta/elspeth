# tests/unit/engine/orchestrator/test_export.py
"""Tests for post-run export and schema reconstruction functions.

export.py has two responsibilities:
1. Export audit trail to JSON/CSV sinks after run completion
2. Reconstruct Pydantic schemas from JSON schema dicts (for pipeline resume)

The export functions mock the LandscapeExporter, RecorderFactory, and sinks.
The schema reconstruction functions are pure logic — no mocks needed.
"""

from __future__ import annotations

import ast
import json
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import ValidationError

from elspeth.contracts import CallType
from elspeth.contracts.audit_export import AuditExportContentStoreResolver
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.sink_effects import SINK_EFFECT_PROTOCOL_VERSION, AuditExportFormat, SinkEffectInputKind
from elspeth.core.landscape.factory import RecorderFactory as _RealRecorderFactory
from elspeth.engine.orchestrator.export import (
    export_landscape as _production_export_landscape,
)
from elspeth.engine.orchestrator.export import prepare_audit_export_binding
from elspeth.engine.orchestrator.preflight import (
    ResolvedSinkEffectMode,
    SinkEffectCapabilityError,
    SinkEffectExecutionPurpose,
    SinkEffectRuntimeBinding,
    validate_pipeline_sink_effect_capabilities,
)
from elspeth.engine.orchestrator.schema_reconstruction import _json_schema_to_python_type, reconstruct_schema_from_json


@contextmanager
def _noop_track_operation(*_args: Any, **_kwargs: Any) -> Any:
    """Stub for track_operation that wires ctx.operation_id without touching the DB."""
    yield SimpleNamespace(operation_id="op-export")


class _CallRecorder:
    def __init__(self, return_value: Any = None, side_effect: BaseException | None = None) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.side_effect is not None:
            raise self.side_effect
        return self.return_value

    @property
    def call_args(self) -> SimpleNamespace:
        assert self.calls
        args, kwargs = self.calls[-1]
        return SimpleNamespace(args=args, kwargs=kwargs)

    def assert_called_once(self) -> None:
        assert len(self.calls) == 1

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        assert self.calls == [(args, kwargs)]

    def assert_not_called(self) -> None:
        assert self.calls == []


class _SinkDouble:
    effect_call_type = CallType.FILESYSTEM
    name = "export_sink"
    plugin_version = "test"
    source_file_hash = "0" * 64
    effect_protocol_version = SINK_EFFECT_PROTOCOL_VERSION
    supported_effect_modes = frozenset({"write"})
    supported_effect_input_kinds = frozenset({SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT})
    supported_audit_export_formats = frozenset({AuditExportFormat.JSON, AuditExportFormat.CSV})

    @classmethod
    def _resolve_sink_effect_mode(
        cls,
        config: dict[str, object],
        *,
        purpose: SinkEffectExecutionPurpose,
    ) -> ResolvedSinkEffectMode:
        del cls, config, purpose
        return ResolvedSinkEffectMode("write")

    def __init__(self, *, config: dict[str, Any] | None = None, **overrides: Any) -> None:
        self.config = config or {}
        self._resolved_effect_mode = "write"
        self.node_id = None
        self.on_start = _CallRecorder()
        self.write = _CallRecorder()
        self.flush = _CallRecorder()
        self.on_complete = _CallRecorder()
        self.close = _CallRecorder()
        for k, v in overrides.items():
            setattr(self, k, v)

    def inspect_effect(self, _request: object, _ctx: object) -> None:
        return None

    def prepare_effect(self, _request: object, _ctx: object) -> None:
        return None

    def commit_effect(self, _plan: object, _ctx: object) -> None:
        return None

    def reconcile_effect(self, _plan: object, _ctx: object) -> None:
        return None


class _AuditContentStoreDouble:
    content_store_id = "archive-primary-v1"
    namespace = "audit-export"

    def is_durable(self) -> bool:
        return True

    def put_immutable(self, content: bytes, *, candidate_id: str, object_kind: str) -> str:
        del content, candidate_id, object_kind
        return f"sha256:{'a' * 64}"

    def open_registered(self, registration: object) -> object:
        return registration

    def mark_candidate_orphans(self, candidate_id: str, descriptors: tuple[object, ...]) -> None:
        del candidate_id, descriptors


_TEST_PAYLOAD_STORE = object()
_TEST_AUDIT_CONTENT_STORE = _AuditContentStoreDouble()
_TEST_AUDIT_CONTENT_STORE_RESOLVER = AuditExportContentStoreResolver()
_TEST_AUDIT_CONTENT_STORE_RESOLVER.register(_TEST_AUDIT_CONTENT_STORE)


def export_landscape(*args: Any, **kwargs: Any) -> None:
    """Keep existing unit scenarios explicit without repeating resource doubles."""
    kwargs.setdefault("payload_store", _TEST_PAYLOAD_STORE)
    kwargs.setdefault("audit_export_content_store", _TEST_AUDIT_CONTENT_STORE)
    kwargs.setdefault("audit_export_content_store_resolver", _TEST_AUDIT_CONTENT_STORE_RESOLVER)
    kwargs.setdefault("worker_id", "runtime-worker")
    _production_export_landscape(*args, **kwargs)


def _make_settings(*, fmt: str = "json", sign: bool = False, sink: str = "output", include_raw_error_rows: bool = False) -> Any:
    return SimpleNamespace(
        sinks={sink: SimpleNamespace(options={})},
        landscape=SimpleNamespace(
            export=SimpleNamespace(
                format=fmt,
                sign=sign,
                signing_secret_ref="ELSPETH_SIGNING_KEY" if sign else None,
                sink=sink,
                include_raw_error_rows=include_raw_error_rows,
                content_store=SimpleNamespace(content_store_id="archive-primary-v1", namespace="audit-export"),
            )
        ),
    )


def _make_sink_and_factory(*, config: dict[str, Any] | None = None, **overrides: Any) -> tuple[_SinkDouble, Any]:
    """Create a sink double and a factory that returns it."""
    sink = _SinkDouble(config=config, **overrides)
    for k, v in overrides.items():
        setattr(sink, k, v)
    binding = SinkEffectRuntimeBinding(
        sink_name="output",
        sink=sink,
        sink_type=type(sink),
        config_fingerprint=stable_hash({}),
        purpose=SinkEffectExecutionPurpose.AUDIT_EXPORT,
        effect_mode=ResolvedSinkEffectMode("write"),
    )
    return sink, lambda _name: binding


@pytest.fixture(autouse=True)
def _mock_recorder_factory():
    """Prevent export tests from hitting real DB via RecorderFactory.register_node()."""
    with patch("elspeth.core.landscape.factory.RecorderFactory"):
        yield


def test_export_landscape_materializes_snapshot_before_audit_rows_and_executes_effect() -> None:
    settings = _make_settings()
    sink, sink_factory = _make_sink_and_factory()
    binding, admission = prepare_audit_export_binding(settings, sink_factory)
    snapshot = object()
    events: list[str] = []
    register_node = _CallRecorder()

    def register(*args: Any, **kwargs: Any) -> None:
        events.append("register_node")
        register_node(*args, **kwargs)

    factory = SimpleNamespace(data_flow=SimpleNamespace(register_node=register, get_node=lambda *_a, **_k: None))

    def prepare(*args: Any, **kwargs: Any) -> object:
        events.append("snapshot")
        return snapshot

    def execute(*args: Any, **kwargs: Any) -> None:
        events.append("effect")
        assert kwargs["factory"] is factory
        assert kwargs["snapshot"] is snapshot
        assert kwargs["sink"] is sink
        assert kwargs["worker_id"] != "audit-export:run-1"

    with (
        patch("elspeth.core.landscape.factory.RecorderFactory", return_value=factory),
        patch("elspeth.engine.orchestrator.audit_export_effects.prepare_audit_export_snapshot", side_effect=prepare),
        patch("elspeth.engine.orchestrator.audit_export_effects.execute_audit_export_effect", side_effect=execute),
    ):
        export_landscape(
            object(),
            "run-1",
            settings,
            sink_factory,
            prepared_binding=binding,
            sink_effect_admission=admission,
        )

    assert events == ["snapshot", "register_node", "effect"]
    sink.write.assert_not_called()
    sink.flush.assert_not_called()
    sink.on_start.assert_not_called()
    sink.on_complete.assert_not_called()
    sink.close.assert_called_once()


def test_csv_export_runs_bundle_capability_probe_before_snapshot_reservation(tmp_path: Path) -> None:
    from elspeth.plugins.sinks.csv_sink import CSVSink

    target = tmp_path / "audit-bundle"
    options = {"path": str(target), "schema": {"mode": "observed"}}
    settings = _make_settings(fmt="csv")
    settings.sinks["output"].options = options
    sink = CSVSink(options)
    binding = SinkEffectRuntimeBinding(
        sink_name="output",
        sink=sink,
        sink_type=type(sink),
        config_fingerprint=stable_hash(options),
        purpose=SinkEffectExecutionPurpose.AUDIT_EXPORT,
        effect_mode=ResolvedSinkEffectMode("write"),
    )
    binding, admission = prepare_audit_export_binding(settings, lambda _name: binding)
    observed: list[str] = []
    factory = SimpleNamespace(data_flow=SimpleNamespace(register_node=_CallRecorder(), get_node=lambda *_a, **_k: None))

    with (
        patch("elspeth.core.landscape.factory.RecorderFactory", return_value=factory),
        patch(
            "elspeth.plugins.sinks._audit_export_bundle_effects.preflight_audit_export_bundle",
            side_effect=lambda path: observed.append(f"probe:{path}"),
        ),
        patch(
            "elspeth.engine.orchestrator.audit_export_effects.prepare_audit_export_snapshot",
            side_effect=lambda *_args, **_kwargs: observed.append("snapshot") or object(),
        ),
        patch("elspeth.engine.orchestrator.audit_export_effects.execute_audit_export_effect"),
    ):
        export_landscape(
            object(),
            "run-1",
            settings,
            lambda _name: binding,
            prepared_binding=binding,
            sink_effect_admission=admission,
        )

    assert observed == [f"probe:{target}", "snapshot"]


# =============================================================================
# export_landscape — JSON format
# =============================================================================


class _LegacyExportLandscapeJSON:
    """Tests for export_landscape with JSON format."""

    def _make_settings(self, *, fmt: str = "json", sign: bool = False, sink: str = "output", include_raw_error_rows: bool = False) -> Any:
        return _make_settings(fmt=fmt, sign=sign, sink=sink, include_raw_error_rows=include_raw_error_rows)

    def test_export_requires_explicit_payload_and_audit_content_stores(self) -> None:
        sink, factory = _make_sink_and_factory()
        payload_store = object()
        audit_content_store = _AuditContentStoreDouble()
        audit_content_store_resolver = AuditExportContentStoreResolver()
        audit_content_store_resolver.register(audit_content_store)

        class StopAfterExporterConstruction(Exception):
            pass

        with (
            patch("elspeth.core.landscape.factory.RecorderFactory") as recorder_factory,
            patch(
                "elspeth.core.landscape.exporter.LandscapeExporter",
                side_effect=StopAfterExporterConstruction,
            ) as exporter,
            pytest.raises(StopAfterExporterConstruction),
        ):
            _production_export_landscape(
                object(),
                "run-1",
                self._make_settings(),
                factory,
                payload_store=payload_store,
                audit_export_content_store=audit_content_store,
                audit_export_content_store_resolver=audit_content_store_resolver,
                worker_id="runtime-worker",
            )

        recorder_factory.assert_called_once_with(exporter.call_args.args[0], payload_store=payload_store)
        assert exporter.call_args.kwargs["read_model"] is not None
        assert sink.node_id is None

    def test_export_preflight_passes_explicit_audit_snapshot_kind(self) -> None:
        sink, factory = _make_sink_and_factory()

        class StopAfterPreflight(Exception):
            pass

        with (
            patch(
                "elspeth.engine.orchestrator.preflight.validate_pipeline_sink_effect_capabilities",
                side_effect=StopAfterPreflight,
            ) as validate,
            pytest.raises(StopAfterPreflight),
        ):
            export_landscape(object(), "run-1", self._make_settings(), factory)
        validate.assert_called_once_with(
            {"output": sink},
            configured_modes={"output": "write"},
            required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        )

    @pytest.mark.parametrize("tamper", ["name", "purpose", "config_fingerprint"])
    def test_prepared_export_binding_revalidates_settings_provenance_before_io(self, tamper: str) -> None:
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        binding, _original_admission = prepare_audit_export_binding(settings, factory)
        if tamper == "name":
            binding = replace(binding, sink_name="other")
        elif tamper == "purpose":
            binding = replace(binding, purpose=SinkEffectExecutionPurpose.FRESH)
        else:
            binding = replace(binding, config_fingerprint=stable_hash({"different": True}))
        admission = validate_pipeline_sink_effect_capabilities(
            {binding.sink_name: sink},
            configured_modes={binding.sink_name: "write"},
            required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        )

        with (
            patch(
                "elspeth.core.landscape.exporter.LandscapeExporter",
                side_effect=lambda *_args, **_kwargs: pytest.fail("export I/O must not begin"),
            ),
            pytest.raises(SinkEffectCapabilityError),
        ):
            export_landscape(
                object(),
                "run-1",
                settings,
                factory,
                prepared_binding=binding,
                sink_effect_admission=admission,
            )

        sink.on_start.assert_not_called()
        sink.write.assert_not_called()

    def test_prepared_export_rejects_receipt_for_separately_admitted_sink_before_io(self) -> None:
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        binding, _admission = prepare_audit_export_binding(settings, factory)
        other_sink = _SinkDouble()
        other_admission = validate_pipeline_sink_effect_capabilities(
            {"output": other_sink},
            configured_modes={"output": "write"},
            required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        )

        with (
            patch(
                "elspeth.core.landscape.exporter.LandscapeExporter",
                side_effect=lambda *_args, **_kwargs: pytest.fail("export I/O must not begin"),
            ),
            pytest.raises(SinkEffectCapabilityError, match="does not bind"),
        ):
            export_landscape(
                object(),
                "run-1",
                settings,
                factory,
                prepared_binding=binding,
                sink_effect_admission=other_admission,
            )

        sink.on_start.assert_not_called()
        sink.write.assert_not_called()

    def test_exact_prepared_export_binding_uses_receipt_without_capability_revalidation(self) -> None:
        class StopAfterBinding(Exception):
            pass

        settings = self._make_settings()
        _sink, factory = _make_sink_and_factory()
        binding, admission = prepare_audit_export_binding(settings, factory)
        with (
            patch(
                "elspeth.engine.orchestrator.preflight.validate_sink_effect_capability",
                side_effect=lambda *_args, **_kwargs: pytest.fail("prepared receipt consumption must be match-only"),
            ),
            patch("elspeth.core.landscape.exporter.LandscapeExporter", side_effect=StopAfterBinding),
            pytest.raises(StopAfterBinding),
        ):
            export_landscape(
                object(),
                "run-1",
                settings,
                factory,
                prepared_binding=binding,
                sink_effect_admission=admission,
            )

    def test_jsonl_export_helpers_do_not_probe_sink_shape_with_getattr(self) -> None:
        import inspect

        from elspeth.engine.orchestrator import export as export_module

        helper_sources = "\n".join(
            inspect.getsource(helper)
            for helper in (
                export_module._jsonl_export_staging_target,
                export_module._jsonl_filesystem_sink_path,
                export_module._close_sink_file_if_open,
                export_module._sink_supports_incremental_json_export_writes,
            )
        )

        assert "getattr(" not in helper_sources

    def test_json_export_writes_records_to_sink(self) -> None:
        """JSON format exports all records through sink.write()."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            export_landscape(db, "run-1", settings, factory)

        sink.on_start.assert_called_once()
        sink.write.assert_called_once()
        sink.flush.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_json_export_skips_write_when_no_records(self) -> None:
        """Empty export produces no sink.write() call."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        sink.write.assert_not_called()
        sink.flush.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_jsonl_export_writes_records_in_bounded_batches(self) -> None:
        """Large JSONL exports write bounded batches instead of one full-run list."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory(_format="jsonl")
        records = [{"record_type": "row", "index": i} for i in range(1001)]

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = records

            export_landscape(db, "run-1", settings, factory)

        assert [len(call[0][0]) for call in sink.write.calls] == [1000, 1]

    def test_json_array_export_writes_real_file_once_for_multi_batch_export(self, tmp_path: Path) -> None:
        """JSON array audit export publishes one final file, not one cumulative rewrite per batch."""
        from elspeth.plugins.sinks import json_sink as json_sink_module
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_path = tmp_path / "audit.json"
        sink = JSONSink({"path": str(output_path), "format": "json", "schema": {"mode": "observed"}})
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, payload_store=None, node_id="export:json")
        records = ({"record_type": "row", "index": i} for i in range(1001))
        replace_calls: list[tuple[Any, Any]] = []
        original_replace = json_sink_module.os.replace

        def counting_replace(src: Any, dst: Any) -> None:
            replace_calls.append((src, dst))
            original_replace(src, dst)

        with patch("elspeth.plugins.sinks.json_sink.os.replace", counting_replace):
            record_count, batches_written = _write_json_export_batches(  # noqa: F821 - removed legacy path
                sink=sink,
                ctx=ctx,
                records=records,
                batch_size=1000,
            )

        sink.close()

        assert record_count == 1001
        assert batches_written == 1
        assert len(replace_calls) == 1
        assert len(json.loads(output_path.read_text())) == 1001

    def test_jsonl_export_failure_removes_staged_partial_file(self, tmp_path: Path) -> None:
        """A failed multi-batch JSONL export must not leave a truncated final artifact."""
        from elspeth.plugins.sinks.json_sink import JSONSink

        output_path = tmp_path / "audit.jsonl"
        sink = JSONSink({"path": str(output_path), "format": "jsonl", "schema": {"mode": "observed"}})
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, payload_store=None, node_id="export:json")
        original_write = sink.write
        write_calls = 0

        def fail_on_second_batch(rows: list[dict[str, Any]], ctx_arg: PluginContext) -> Any:
            nonlocal write_calls
            write_calls += 1
            if write_calls == 2:
                raise OSError("disk full")
            return original_write(rows, ctx_arg)

        sink.write = fail_on_second_batch  # type: ignore[method-assign]

        try:
            with pytest.raises(OSError, match="disk full"):
                _write_json_export_batches(  # noqa: F821 - removed legacy path
                    sink=sink,
                    ctx=ctx,
                    records=({"record_type": "row", "index": i} for i in range(1001)),
                    batch_size=1000,
                )
        finally:
            sink.close()

        assert write_calls == 2
        assert not output_path.exists()
        assert not output_path.with_suffix(output_path.suffix + ".tmp").exists()

    def test_json_export_records_count_as_operation_output(self) -> None:
        """JSON record_count is captured after streaming, not pre-counted as input."""
        db = object()
        settings = self._make_settings()
        _sink, factory = _make_sink_and_factory()
        captured: list[dict[str, Any]] = []

        @contextmanager
        def capture_track_operation(*_args: Any, **kwargs: Any) -> Any:
            handle = SimpleNamespace(operation_id="op-export", output_data=None)
            captured.append({"kwargs": kwargs, "handle": handle})
            yield handle

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", capture_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"record_type": "run"}, {"record_type": "node"}]

            export_landscape(db, "run-1", settings, factory)

        assert captured[0]["kwargs"]["input_data"] == {"export_format": "json"}
        assert captured[0]["handle"].output_data == {"record_count": 2, "batches_written": 1}

    def test_missing_sink_raises_valueerror(self) -> None:
        """Referencing non-existent sink raises clear error."""
        db = object()
        settings = self._make_settings(sink="nonexistent")

        def bad_factory(name: str) -> Any:
            raise ValueError(f"Export sink '{name}' not found in sink configuration")

        # LandscapeExporter constructs its own (module-level imported, so
        # autouse-unpatched) RecorderFactory, which now eagerly probes the
        # engine. Patch it so the bad-factory ValueError is what surfaces.
        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter"),
            pytest.raises(ValueError, match=r"nonexistent.*not found"),
        ):
            export_landscape(db, "run-1", settings, bad_factory)

    def test_signing_reads_env_key(self) -> None:
        """Signing enabled reads ELSPETH_SIGNING_KEY from env."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
            patch.dict("os.environ", {"ELSPETH_SIGNING_KEY": "test-key-123"}),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        MockExporter.assert_called_once()
        assert MockExporter.call_args.args == (db,)
        assert MockExporter.call_args.kwargs["signing_key"] == b"test-key-123"
        assert MockExporter.call_args.kwargs["include_raw_error_rows"] is False
        assert MockExporter.call_args.kwargs["read_model"] is not None

    def test_raw_error_rows_opt_in_threads_to_exporter(self) -> None:
        """The export-config classification flag reaches the exporter
        (elspeth-384184c6ab): default False redacts raw failing rows from
        error records; the explicit opt-in restores them."""
        db: Any = object()
        settings = self._make_settings(include_raw_error_rows=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = []

            export_landscape(db, "run-1", settings, factory)

        MockExporter.assert_called_once()
        assert MockExporter.call_args.args == (db,)
        assert MockExporter.call_args.kwargs["signing_key"] is None
        assert MockExporter.call_args.kwargs["include_raw_error_rows"] is True
        assert MockExporter.call_args.kwargs["read_model"] is not None

    def test_signing_without_env_key_raises(self) -> None:
        """Signing enabled without ELSPETH_SIGNING_KEY raises ValueError."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            export_landscape(db, "run-1", settings, factory)

    @pytest.mark.parametrize("key_value", ["", "   "])
    def test_signing_with_empty_env_key_raises(self, key_value: str) -> None:
        """Signing enabled with an empty ELSPETH_SIGNING_KEY raises ValueError."""
        db = object()
        settings = self._make_settings(sign=True)
        _sink, factory = _make_sink_and_factory()

        with (
            patch.dict("os.environ", {"ELSPETH_SIGNING_KEY": key_value}, clear=True),
            pytest.raises(ValueError, match="ELSPETH_SIGNING_KEY"),
        ):
            export_landscape(db, "run-1", settings, factory)

    def test_sink_close_called_when_write_raises(self) -> None:
        """sink.close() must be called even when sink.write() raises."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        sink.write.side_effect = RuntimeError("write failed")

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="write failed"):
                export_landscape(db, "run-1", settings, factory)

        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()

    def test_sink_close_called_when_flush_raises(self) -> None:
        """sink.close() must be called even when sink.flush() raises."""
        db = object()
        settings = self._make_settings()
        sink, factory = _make_sink_and_factory()
        sink.flush.side_effect = RuntimeError("flush failed")

        with (
            patch("elspeth.core.landscape.exporter.LandscapeExporter") as MockExporter,
            patch("elspeth.engine.orchestrator.export.track_operation", _noop_track_operation),
        ):
            exporter = MockExporter.return_value
            exporter.export_run.return_value = [{"type": "run", "id": "r1"}]

            with pytest.raises(RuntimeError, match="flush failed"):
                export_landscape(db, "run-1", settings, factory)

        sink.write.assert_called_once()
        sink.on_complete.assert_called_once()
        sink.close.assert_called_once()


def test_export_module_has_no_direct_filesystem_csv_path() -> None:
    repo_root = Path(__file__).parents[4]
    path = repo_root / "src/elspeth/engine/orchestrator/export.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    obsolete_names = {"_FileSystemCsvAuditExportWriter", "_CsvRecordTypeSpool", "_export_csv_multifile"}

    defined_names = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}

    assert defined_names.isdisjoint(obsolete_names)


def test_export_module_does_not_define_resume_schema_reconstruction_helpers() -> None:
    """Resume schema reconstruction lives outside the post-run export module."""
    repo_root = Path(__file__).parents[4]
    path = repo_root / "src/elspeth/engine/orchestrator/export.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    schema_helper_names = {
        "reconstruct_schema_from_json",
        "_create_schema_model",
        "_model_name_for_field",
        "_json_schema_to_python_type",
    }

    offenders = [node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in schema_helper_names]

    assert offenders == []


# =============================================================================
# reconstruct_schema_from_json — Primitive types
# =============================================================================


class TestReconstructSchemaBasic:
    """Tests for reconstruct_schema_from_json with basic types."""

    def test_string_field(self) -> None:
        """String type maps correctly."""
        schema = {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Alice")
        assert instance.name == "Alice"

    def test_integer_field(self) -> None:
        """Integer type maps correctly."""
        schema = {
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(count=42)
        assert instance.count == 42

    def test_number_field(self) -> None:
        """Number type maps to float."""
        schema = {
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(score=3.14)
        assert instance.score == pytest.approx(3.14)

    def test_boolean_field(self) -> None:
        """Boolean type maps correctly."""
        schema = {
            "properties": {"active": {"type": "boolean"}},
            "required": ["active"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(active=True)
        assert instance.active is True

    def test_optional_field_defaults_to_none(self) -> None:
        """Fields not in 'required' default to None."""
        schema = {
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Bob")
        assert instance.name == "Bob"
        assert instance.age is None

    def test_all_fields_required(self) -> None:
        """All fields in required list are enforced."""
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(a="x", b=1)
        assert instance.a == "x"
        assert instance.b == 1

    def test_no_required_key_means_all_optional(self) -> None:
        """Missing 'required' key treats all fields as optional."""
        schema = {
            "properties": {"name": {"type": "string"}},
        }
        model = reconstruct_schema_from_json(schema)
        instance = model()
        assert instance.name is None

    def test_array_field(self) -> None:
        """Array type maps to list."""
        schema = {
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["tags"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_array_field_with_items_enforces_item_type(self) -> None:
        """Array with items schema enforces item type on resume."""
        schema = {
            "properties": {"scores": {"type": "array", "items": {"type": "integer"}}},
            "required": ["scores"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(scores=[1, 2, 3])
        assert instance.scores == [1, 2, 3]

        with pytest.raises(ValidationError, match="scores\\.0"):
            model(scores=["not-an-int"])

    def test_object_field(self) -> None:
        """Object type maps to dict."""
        schema = {
            "properties": {"metadata": {"type": "object"}},
            "required": ["metadata"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(metadata={"key": "val"})
        assert instance.metadata == {"key": "val"}

    def test_object_field_with_properties_enforces_nested_schema(self) -> None:
        """Nested object properties are reconstructed and validated."""
        schema = {
            "properties": {
                "profile": {
                    "type": "object",
                    "properties": {
                        "age": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                    "required": ["age", "name"],
                }
            },
            "required": ["profile"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(profile={"age": 42, "name": "Ada"})
        assert instance.profile.age == 42
        assert instance.profile.name == "Ada"

        with pytest.raises(ValidationError, match="profile\\.age"):
            model(profile={"age": "not-an-int", "name": "Ada"})

        with pytest.raises(ValidationError, match="profile\\.name"):
            model(profile={"age": 42})


# =============================================================================
# reconstruct_schema_from_json — Format specifiers
# =============================================================================


class TestReconstructSchemaFormats:
    """Tests for string format specifiers (datetime, date, UUID, etc.)."""

    def test_datetime_format(self) -> None:
        """date-time format maps to datetime."""
        schema = {
            "properties": {"ts": {"type": "string", "format": "date-time"}},
            "required": ["ts"],
        }
        model = reconstruct_schema_from_json(schema)
        now = datetime.now(tz=UTC)
        instance = model(ts=now)
        assert instance.ts == now

    def test_date_format(self) -> None:
        """date format maps to date."""
        schema = {
            "properties": {"d": {"type": "string", "format": "date"}},
            "required": ["d"],
        }
        model = reconstruct_schema_from_json(schema)
        today = datetime.now(tz=UTC).date()
        instance = model(d=today)
        assert instance.d == today

    def test_time_format(self) -> None:
        """time format maps to time."""
        schema = {
            "properties": {"t": {"type": "string", "format": "time"}},
            "required": ["t"],
        }
        model = reconstruct_schema_from_json(schema)
        now = time(12, 30, 0)
        instance = model(t=now)
        assert instance.t == now

    def test_duration_format(self) -> None:
        """duration format maps to timedelta."""
        schema = {
            "properties": {"dur": {"type": "string", "format": "duration"}},
            "required": ["dur"],
        }
        model = reconstruct_schema_from_json(schema)
        td = timedelta(hours=1, minutes=30)
        instance = model(dur=td)
        assert instance.dur == td

    def test_uuid_format(self) -> None:
        """uuid format maps to UUID."""
        schema = {
            "properties": {"id": {"type": "string", "format": "uuid"}},
            "required": ["id"],
        }
        model = reconstruct_schema_from_json(schema)
        uid = UUID("12345678-1234-5678-1234-567812345678")
        instance = model(id=uid)
        assert instance.id == uid

    def test_unknown_format_treated_as_string(self) -> None:
        """Unknown format falls back to str."""
        schema = {
            "properties": {"custom": {"type": "string", "format": "custom-fmt"}},
            "required": ["custom"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(custom="hello")
        assert instance.custom == "hello"


# =============================================================================
# reconstruct_schema_from_json — anyOf patterns
# =============================================================================


class TestReconstructSchemaAnyOf:
    """Tests for anyOf patterns (Decimal, nullable)."""

    def test_decimal_anyof_pattern(self) -> None:
        """anyOf with number+string maps to Decimal."""
        schema = {
            "properties": {
                "amount": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
            "required": ["amount"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(amount=Decimal("123.45"))
        assert instance.amount == Decimal("123.45")

    def test_nullable_type_pattern(self) -> None:
        """anyOf with type+null maps to Optional[type] — accepts both values and None."""
        schema = {
            "properties": {
                "name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        instance = model(name="Alice")
        assert instance.name == "Alice"
        # Nullable fields must also accept None
        instance_none = model(name=None)
        assert instance_none.name is None

    def test_nullable_datetime_pattern(self) -> None:
        """anyOf with datetime+null resolves to Optional[datetime]."""
        schema = {
            "properties": {
                "ts": {
                    "anyOf": [
                        {"type": "string", "format": "date-time"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["ts"],
        }
        model = reconstruct_schema_from_json(schema)
        now = datetime.now(tz=UTC)
        instance = model(ts=now)
        assert instance.ts == now
        # Nullable datetime must accept None
        instance_none = model(ts=None)
        assert instance_none.ts is None

    def test_nullable_decimal_anyof_pattern(self) -> None:
        """anyOf with number+string+null maps to Optional[Decimal].

        Regression: P1-2026-02-14 — Pydantic generates a 3-branch anyOf for
        Decimal | None: [{"type":"number"}, {"type":"string"}, {"type":"null"}].
        The old code only handled Decimal (2-branch, no null) and nullable
        (2-branch, one non-null), so this 3-branch pattern raised ValueError.
        """
        schema = {
            "properties": {
                "amount": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["amount"],
        }
        model = reconstruct_schema_from_json(schema)
        # Must accept Decimal values
        instance = model(amount=Decimal("99.99"))
        assert instance.amount == Decimal("99.99")
        # Must accept None (nullable)
        instance_none = model(amount=None)
        assert instance_none.amount is None

    def test_nullable_decimal_from_pydantic_model_json_schema(self) -> None:
        """Round-trip: Pydantic model with Decimal | None survives JSON schema reconstruction."""
        from pydantic import BaseModel

        class InvoiceRow(BaseModel):
            invoice_id: str
            amount: Decimal | None

        schema_dict = InvoiceRow.model_json_schema()
        model = reconstruct_schema_from_json(schema_dict)
        instance = model(invoice_id="INV-001", amount=Decimal("42.50"))
        assert instance.amount == Decimal("42.50")
        instance_none = model(invoice_id="INV-002", amount=None)
        assert instance_none.amount is None

    def test_unsupported_anyof_raises(self) -> None:
        """Unsupported anyOf pattern (e.g., Union[str, int]) raises."""
        schema = {
            "properties": {
                "weird": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["weird"],
        }
        with pytest.raises(AuditIntegrityError, match="unsupported anyOf"):
            reconstruct_schema_from_json(schema)


# =============================================================================
# reconstruct_schema_from_json — Error cases
# =============================================================================


class TestReconstructSchemaErrors:
    """Tests for error conditions in schema reconstruction."""

    def test_missing_properties_raises(self) -> None:
        """Schema without 'properties' key is malformed."""
        with pytest.raises(AuditIntegrityError, match="no 'properties'"):
            reconstruct_schema_from_json({"type": "object"})

    def test_empty_properties_without_additional_raises(self) -> None:
        """Empty properties without additionalProperties=true is invalid."""
        with pytest.raises(AuditIntegrityError, match="zero fields"):
            reconstruct_schema_from_json({"properties": {}})

    def test_empty_properties_with_additional_creates_dynamic(self) -> None:
        """Empty properties + additionalProperties=true creates dynamic schema."""
        schema = {"properties": {}, "additionalProperties": True}
        model = reconstruct_schema_from_json(schema)
        # Dynamic schema should accept arbitrary fields
        instance = model(any_field="value", another=42)
        assert instance.any_field == "value"

    def test_unsupported_type_raises(self) -> None:
        """Unknown JSON schema type (e.g., 'custom') raises."""
        schema = {
            "properties": {"x": {"type": "custom_type"}},
            "required": ["x"],
        }
        with pytest.raises(AuditIntegrityError, match=r"unsupported type.*custom_type"):
            reconstruct_schema_from_json(schema)

    def test_field_missing_type_raises(self) -> None:
        """Field without 'type' key (and no anyOf) raises."""
        schema = {
            "properties": {"x": {"description": "no type here"}},
            "required": ["x"],
        }
        with pytest.raises(AuditIntegrityError, match="no 'type'"):
            reconstruct_schema_from_json(schema)


# =============================================================================
# _json_schema_to_python_type — Direct tests
# =============================================================================


class TestJsonSchemaToPythonType:
    """Direct tests for the type mapping helper."""

    @pytest.mark.parametrize(
        ("field_info", "expected_type"),
        [
            ({"type": "string"}, str),
            ({"type": "integer"}, int),
            ({"type": "number"}, float),
            ({"type": "boolean"}, bool),
            ({"type": "array"}, list),
            ({"type": "object"}, dict),
            ({"type": "string", "format": "date-time"}, datetime),
            ({"type": "string", "format": "date"}, date),
            ({"type": "string", "format": "time"}, time),
            ({"type": "string", "format": "duration"}, timedelta),
            ({"type": "string", "format": "uuid"}, UUID),
        ],
        ids=[
            "string",
            "integer",
            "number",
            "boolean",
            "array",
            "object",
            "datetime",
            "date",
            "time",
            "duration",
            "uuid",
        ],
    )
    def test_type_mapping(self, field_info: dict[str, Any], expected_type: type) -> None:
        """Each JSON schema type maps to the correct Python type."""
        result = _json_schema_to_python_type("test_field", field_info)
        assert result is expected_type

    def test_decimal_anyof(self) -> None:
        """Decimal pattern recognized via anyOf."""
        field_info = {"anyOf": [{"type": "number"}, {"type": "string"}]}
        assert _json_schema_to_python_type("price", field_info) is Decimal

    def test_nullable_resolves_to_optional(self) -> None:
        """Nullable pattern resolves to Optional[inner_type]."""
        field_info = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
        result = _json_schema_to_python_type("count", field_info)
        # Should be int | None (UnionType), not bare int
        assert result == int | None

    def test_nullable_ref_resolves_through_defs(self) -> None:
        """Nullable $ref pattern (Optional[NestedModel]) resolves via $defs.

        Regression: Pydantic emits Optional[NestedModel] as:
          {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]}
        The $ref entry has NO "type" key, so filtering on item["type"]
        raised KeyError.
        """
        import types

        from pydantic import BaseModel, create_model

        field_info: dict[str, Any] = {
            "anyOf": [
                {"$ref": "#/$defs/Address"},
                {"type": "null"},
            ],
        }
        schema_defs: dict[str, Any] = {
            "Address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["city", "zip"],
            }
        }
        # Must not raise KeyError — the $ref item lacks a "type" key
        result = _json_schema_to_python_type(
            "address",
            field_info,
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=BaseModel,
        )
        # Should be Optional[AddressModel] — a UnionType containing the model and None
        assert isinstance(result, types.UnionType)
        type_args = result.__args__
        assert type(None) in type_args
        # The non-None arg should be a Pydantic model subclass
        model_type = next(t for t in type_args if t is not type(None))
        assert issubclass(model_type, BaseModel)
        instance = model_type(city="London", zip="SW1A 1AA")
        assert instance.city == "London"

    def test_nullable_ref_full_schema_roundtrip(self) -> None:
        """Full schema with Optional[NestedModel] field reconstructs correctly.

        Regression: reconstruct_schema_from_json crashed on schemas containing
        fields like Optional[Address] because the anyOf filter accessed
        item["type"] on $ref entries that have no "type" key.
        """
        schema: dict[str, Any] = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                }
            },
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "anyOf": [
                        {"$ref": "#/$defs/Address"},
                        {"type": "null"},
                    ],
                },
            },
            "required": ["name"],
        }
        model = reconstruct_schema_from_json(schema)
        # Non-null value
        instance = model(name="Alice", address={"city": "London"})
        assert instance.name == "Alice"
        assert instance.address.city == "London"
        # Null value (optional)
        instance2 = model(name="Bob", address=None)
        assert instance2.address is None


class TestExportNodeRegistrationIdempotence:
    """elspeth-08350558e6: the deterministic ``export:<sink>`` node must be
    re-registerable so an export retry (after a failed or lost publication
    response) reaches SinkEffectCoordinator reconciliation instead of
    crashing on the nodes composite primary key."""

    @staticmethod
    def _real_db_setup() -> tuple[Any, Any, str]:
        from elspeth.core.landscape.database import LandscapeDB
        from tests.fixtures.stores import MockPayloadStore

        db = LandscapeDB.in_memory()
        factory = _RealRecorderFactory(db, payload_store=MockPayloadStore())
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-export-retry")
        return db, factory, run.run_id

    def test_export_retry_after_lost_publication_response_reuses_registered_node(self) -> None:
        db, real_factory, run_id = self._real_db_setup()
        try:
            settings = _make_settings()
            _sink, sink_factory = _make_sink_and_factory(source_file_hash="sha256:" + "0" * 16)
            snapshot = object()
            executed: list[str] = []

            with (
                patch("elspeth.core.landscape.factory.RecorderFactory", _RealRecorderFactory),
                patch(
                    "elspeth.engine.orchestrator.audit_export_effects.prepare_audit_export_snapshot",
                    return_value=snapshot,
                ),
                patch(
                    "elspeth.engine.orchestrator.audit_export_effects.execute_audit_export_effect",
                    side_effect=RuntimeError("publication response lost"),
                ),
                pytest.raises(RuntimeError, match="publication response lost"),
            ):
                export_landscape(db, run_id, settings, sink_factory)

            assert real_factory.data_flow.get_node("export:output", run_id) is not None

            with (
                patch("elspeth.core.landscape.factory.RecorderFactory", _RealRecorderFactory),
                patch(
                    "elspeth.engine.orchestrator.audit_export_effects.prepare_audit_export_snapshot",
                    return_value=snapshot,
                ),
                patch(
                    "elspeth.engine.orchestrator.audit_export_effects.execute_audit_export_effect",
                    side_effect=lambda **_kwargs: executed.append("effect"),
                ),
            ):
                export_landscape(db, run_id, settings, sink_factory)

            assert executed == ["effect"], "retry must reach durable effect recovery"
            assert real_factory.data_flow.get_node("export:output", run_id) is not None
        finally:
            db.close()

    def test_export_refuses_to_reuse_node_registered_with_divergent_identity(self) -> None:
        from elspeth.contracts import NodeType
        from tests.fixtures.landscape import register_test_node

        db, real_factory, run_id = self._real_db_setup()
        try:
            register_test_node(
                real_factory.data_flow,
                run_id,
                "export:output",
                node_type=NodeType.TRANSFORM,
                plugin_name="imposter",
            )
            settings = _make_settings()
            _sink, sink_factory = _make_sink_and_factory(source_file_hash="sha256:" + "0" * 16)

            with (
                patch("elspeth.core.landscape.factory.RecorderFactory", _RealRecorderFactory),
                patch(
                    "elspeth.engine.orchestrator.audit_export_effects.prepare_audit_export_snapshot",
                    return_value=object(),
                ),
                patch("elspeth.engine.orchestrator.audit_export_effects.execute_audit_export_effect") as execute,
                pytest.raises(AuditIntegrityError, match="export:output"),
            ):
                export_landscape(db, run_id, settings, sink_factory)

            execute.assert_not_called()
        finally:
            db.close()
