"""Runtime preflight must not touch external systems during plugin setup."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest

from elspeth.core.config import load_settings_from_yaml_string
from elspeth.plugins.infrastructure.preflight import plugin_preflight_mode, plugin_preflight_mode_enabled
from elspeth.plugins.infrastructure.runtime_factory import instantiate_plugins_from_config
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sources.csv_source import CSVSource
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.composer import yaml_generator
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.config import WebSettings
from elspeth.web.execution.validation import validate_pipeline


def _forbid_socket_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("runtime preflight plugin instantiation must not open network sockets")

    monkeypatch.setattr(socket, "socket", fail)
    monkeypatch.setattr(socket, "create_connection", fail)
    monkeypatch.setattr(socket, "getaddrinfo", fail)


def _web_settings(tmp_path: Path) -> WebSettings:
    return WebSettings(
        data_dir=tmp_path,
        composer_max_composition_turns=10,
        composer_max_discovery_turns=5,
        composer_timeout_seconds=30.0,
        composer_rate_limit_per_minute=60,
        shareable_link_signing_key=b"\x00" * 32,
    )


def _csv_worker_probe_state(tmp_path: Path) -> CompositionState:
    blobs_dir = tmp_path / "blobs"
    outputs_dir = tmp_path / "outputs"
    blobs_dir.mkdir()
    outputs_dir.mkdir()
    input_path = blobs_dir / "input.csv"
    input_path.write_text("name\nAda\n", encoding="utf-8")
    return CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="primary",
            options={"path": str(input_path), "schema": {"mode": "observed"}},
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="primary",
                plugin="csv",
                options={"path": str(outputs_dir / "out.csv"), "schema": {"mode": "observed"}},
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(),
        version=1,
    )


def _minimal_csv_pipeline_yaml(tmp_path: Path) -> str:
    """Minimal CSV source → CSV sink pipeline YAML with absolute paths under tmp_path."""
    blobs_dir = tmp_path / "blobs"
    outputs_dir = tmp_path / "outputs"
    blobs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    input_path = blobs_dir / "probe_input.csv"
    input_path.write_text("name\nAda\n", encoding="utf-8")
    return f"""\
sources:
  primary:
    plugin: csv
    on_success: primary
    options:
      path: {input_path!s}
      on_validation_failure: discard
      schema:
        mode: observed
sinks:
  primary:
    plugin: csv
    on_write_failure: discard
    options:
      path: {outputs_dir / "probe_output.csv"!s}
      schema:
        mode: observed
"""


def _external_plugin_probe_pipeline_yaml(tmp_path: Path) -> str:
    """Pipeline YAML with representative external plugins for constructor-purity checks.

    Source: CSV (guaranteed importable, no network in constructor).
    Transforms:
      - llm (openrouter provider, probe_config): client deferred to on_start()
        via _create_provider(); constructor does schema/config parsing only.
      - web_scrape (probe_config): HTTP session deferred to on_start(); constructor
        computes IP allowlist and parses config.
    Sinks include all risky external-client families available in this checkout:
      - azure_blob: BlobServiceClient deferred via _get_blob_client() lazy property.
      - dataverse: DataverseClient/credential deferred to on_start().
      - chroma_sink: chromadb.HttpClient deferred to on_start().
      - csv: baseline, no external client.

    The RAG transform is omitted from the transform chain because it requires the
    on_start() provider (azure_search) for construction of a live RetrievalProvider,
    and adding it as a transform would require the full LifecycleContext during
    graph validation which is outside the scope of constructor-purity tests.
    RAG's on_start()-deferred pattern is structurally identical to the LLM transform
    already covered, and the sink representatives ensure coverage of all four
    external-client families.
    """
    blobs_dir = tmp_path / "blobs"
    outputs_dir = tmp_path / "outputs"
    blobs_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)
    input_path = blobs_dir / "external_probe_input.csv"
    input_path.write_text("llm_probe_text,web_scrape_probe_url\ntest,http://example.com\n", encoding="utf-8")
    return f"""\
sources:
  primary:
    plugin: csv
    on_success: llm_step
    options:
      path: {input_path!s}
      on_validation_failure: discard
      schema:
        mode: observed
transforms:
  - name: llm_step
    plugin: llm
    input: llm_step
    on_success: scrape_step
    on_error: discard
    options:
      provider: openrouter
      api_key: probe-key
      model: openai/gpt-4o
      prompt_template: "{{{{ row.llm_probe_text }}}}"
      schema:
        mode: observed
      required_input_fields: []
  - name: scrape_step
    plugin: web_scrape
    input: scrape_step
    on_success: csv_primary
    on_error: discard
    options:
      schema:
        mode: observed
      url_field: web_scrape_probe_url
      content_field: page_content
      fingerprint_field: page_fingerprint
      http:
        abuse_contact: invariants@example.com
        scraping_reason: ADR-009 invariant probe
        allowed_hosts:
          - 93.184.216.34/32
sinks:
  csv_primary:
    plugin: csv
    on_write_failure: discard
    options:
      path: {outputs_dir / "external_probe_output.csv"!s}
      schema:
        mode: observed
  azure_blob_probe:
    plugin: azure_blob
    on_write_failure: discard
    options:
      container: probe-container
      blob_path: probe/output.csv
      format: csv
      schema:
        mode: observed
      connection_string: DefaultEndpointsProtocol=https;AccountName=probe;AccountKey=cHJvYmUK;EndpointSuffix=core.windows.net
  dataverse_probe:
    plugin: dataverse
    on_write_failure: discard
    options:
      environment_url: https://invariant.example.crm.dynamics.com
      entity: probe_entity
      alternate_key: probe_field
      schema:
        mode: observed
      auth:
        method: managed_identity
      field_mapping:
        probe_field: probe_field
  chroma_probe:
    plugin: chroma_sink
    on_write_failure: discard
    options:
      collection: probe-collection
      mode: client
      host: invariant.example.com
      port: 8000
      ssl: true
      schema:
        mode: observed
      field_mapping:
        document_field: page_content
        id_field: probe_id
"""


def test_preflight_mode_instantiates_external_plugins_without_network(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Representative external constructors stay pure in preflight mode.

    Include plugins whose real runtime path creates Azure, Dataverse, OpenAI/
    OpenRouter, RAG, Chroma, or HTTP clients in lifecycle methods. Prefer each
    plugin's probe_config() where it exists; the test is about constructor
    purity, not live credentials.
    """
    pipeline_yaml = _external_plugin_probe_pipeline_yaml(tmp_path)
    settings = load_settings_from_yaml_string(pipeline_yaml)

    _forbid_socket_calls(monkeypatch)
    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)

    assert bundle.sources
    assert bundle.sinks


@pytest.mark.asyncio
async def test_run_sync_in_worker_preserves_preflight_mode_for_plugin_constructors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Constructors see preflight mode through the production worker path.

    This pins the runtime contract, not the ContextVar implementation detail:
    validate_pipeline() may run in a ThreadPoolExecutor via run_sync_in_worker(),
    and constructors must still observe preflight mode inside that worker.
    """
    observed: list[tuple[str, bool]] = []
    original_source_init = CSVSource.__init__
    original_sink_init = CSVSink.__init__

    def source_init(self: CSVSource, config: dict[str, Any]) -> None:
        observed.append(("source", plugin_preflight_mode_enabled()))
        original_source_init(self, config)

    def sink_init(self: CSVSink, config: dict[str, Any]) -> None:
        observed.append(("sink", plugin_preflight_mode_enabled()))
        original_sink_init(self, config)

    monkeypatch.setattr(CSVSource, "__init__", source_init)
    monkeypatch.setattr(CSVSink, "__init__", sink_init)

    result = await run_sync_in_worker(
        validate_pipeline,
        _csv_worker_probe_state(tmp_path),
        _web_settings(tmp_path),
        yaml_generator,
    )

    assert result.is_valid is True
    assert observed == [("source", True), ("sink", True)]


def test_runtime_mode_default_does_not_enable_preflight_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The normal execution path must remain real runtime mode by default.

    I5 strengthening: the previous version of this test only asserted
    ``bundle.sources``, which would pass even if the
    ContextVar were permanently stuck at True. The monkeypatched
    constructor probe directly observes ``plugin_preflight_mode_enabled()``
    at instantiation time and asserts the False default — that is the
    actual property the test name claims to pin.
    """
    observed: list[bool] = []
    original_source_init = CSVSource.__init__

    def source_init(self: CSVSource, config: dict[str, Any]) -> None:
        observed.append(plugin_preflight_mode_enabled())
        original_source_init(self, config)

    monkeypatch.setattr(CSVSource, "__init__", source_init)

    pipeline_yaml = _minimal_csv_pipeline_yaml(tmp_path)
    settings = load_settings_from_yaml_string(pipeline_yaml)

    bundle = instantiate_plugins_from_config(settings)

    assert bundle.sources
    assert observed == [False], (
        "Default runtime mode (no plugin_preflight_mode wrapper) MUST "
        "instantiate plugins with plugin_preflight_mode_enabled() == False. "
        f"Observed: {observed}"
    )


@pytest.mark.asyncio
async def test_preflight_context_isolated_between_concurrent_tasks() -> None:
    """I5 lock-in: ContextVar mutations in one asyncio task MUST NOT leak
    into a sibling task spawned via ``asyncio.create_task``.

    Python's ContextVar semantics give each task a copy of the parent
    context at task creation, so ``plugin_preflight_mode(True)`` inside
    Task A cannot be observed by Task B even when both run concurrently.
    Without this guarantee, two requests in flight at the same moment
    could see each other's preflight state — a tenant-isolation hazard
    in a multi-request server. The asyncio.Event barriers are the only
    way to assert ordering: without them, sequential ``await`` could
    mask an isolation bug by serialising the observations.
    """
    import asyncio

    a_set_mode = asyncio.Event()
    b_observed_state = asyncio.Event()
    observed_in_b: list[bool] = []
    observed_in_a_after_b: list[bool] = []

    async def task_a() -> None:
        with plugin_preflight_mode(True):
            assert plugin_preflight_mode_enabled() is True
            a_set_mode.set()
            await b_observed_state.wait()
            # Re-observe inside Task A — the state is still True here
            # because the context manager has not exited.
            observed_in_a_after_b.append(plugin_preflight_mode_enabled())

    async def task_b() -> None:
        await a_set_mode.wait()
        observed_in_b.append(plugin_preflight_mode_enabled())
        b_observed_state.set()

    await asyncio.gather(asyncio.create_task(task_a()), asyncio.create_task(task_b()))

    assert observed_in_b == [False], (
        "Sibling asyncio task observed Task A's preflight_mode=True — "
        "ContextVar isolation between create_task() siblings is broken. "
        f"Observed in Task B: {observed_in_b}"
    )
    assert observed_in_a_after_b == [True], (
        "Task A lost its own preflight_mode=True after Task B observed — "
        "context isolation is leaking the wrong direction. "
        f"Observed in Task A after B: {observed_in_a_after_b}"
    )
    # And the outer context is back to default — neither task's mutation
    # leaked out.
    assert plugin_preflight_mode_enabled() is False


def test_preflight_context_resets_when_body_raises() -> None:
    """I5 lock-in: ``plugin_preflight_mode`` uses try/finally + ContextVar.reset(token),
    so an exception inside the ``with`` block MUST still reset the
    ContextVar to its prior value.

    Without this guarantee, a plugin constructor that raises during
    runtime preflight would leave the ContextVar permanently True for
    the rest of the asyncio task, contaminating downstream plugin
    instantiations that should run in real-runtime mode.
    """

    class _ConstructorBoom(Exception):
        """Synthetic plugin-constructor failure to drive the test."""

    assert plugin_preflight_mode_enabled() is False  # baseline

    with pytest.raises(_ConstructorBoom), plugin_preflight_mode(True):
        assert plugin_preflight_mode_enabled() is True
        raise _ConstructorBoom("simulated plugin constructor failure")

    assert plugin_preflight_mode_enabled() is False, (
        "After an exception escaped plugin_preflight_mode(True), the "
        "ContextVar did not reset — try/finally with ContextVar.reset(token) "
        "should have restored the prior value regardless of the exception."
    )
