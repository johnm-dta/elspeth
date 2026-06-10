"""Tests for tutorial run service hardening."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import select

from elspeth.contracts import NodeType
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import run_attributions_table
from elspeth.web.composer import tutorial_telemetry as tutorial_telemetry_module
from elspeth.web.composer.skills import load_skill_with_hash
from elspeth.web.composer.tutorial_models import TutorialRunOutput
from elspeth.web.composer.tutorial_service import (
    TutorialRunIntegrityError,
    _cache_seed_skip_reason,
    _coalesce_run_source_hashes,
    _count_discarded_rows,
    _parse_rows_file,
    _plugin_nodes_from_composition_state,
    _plugin_nodes_from_pipeline_dict,
    _replay_cache_entry,
    _rows_from_artifacts,
    _state_matches_cached_topology,
    _store_successful_live_projection,
    tutorial_model_id,
)
from elspeth.web.config import WebSettings
from elspeth.web.preferences.tutorial_cache import CANONICAL_SEED_PROMPT, TutorialCache, TutorialCacheEntry, tutorial_cache_key
from elspeth.web.sessions.protocol import CompositionStateRecord, RunRecord
from tests.fixtures.landscape import make_factory, make_landscape_db


def _make_tutorial_settings(data_dir: Path, **overrides: Any) -> WebSettings:
    values: dict[str, Any] = {
        "data_dir": data_dir,
        "composer_max_composition_turns": 15,
        "composer_max_discovery_turns": 10,
        "composer_timeout_seconds": 85.0,
        "composer_rate_limit_per_minute": 10,
        "shareable_link_signing_key": b"\x00" * 32,
    }
    values.update(overrides)
    return WebSettings(**values)


def test_count_discarded_rows_counts_only_discard_destination() -> None:
    """_count_discarded_rows counts validation_errors whose destination is the
    'discard' sentinel, NOT quarantine-to-a-sink rows (which have a visible
    destination). This is what the tutorial UX surfaces so source-dropped rows
    are not silently invisible."""
    from elspeth.contracts.schema import SchemaConfig

    db = make_landscape_db()
    factory = make_factory(db)
    run_id = "run-discard-count"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=run_id)
    factory.data_flow.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )
    for i in range(2):
        factory.data_flow.record_validation_error(
            run_id=run_id,
            node_id="source-0",
            row_data={"i": i},
            error="comma split the row",
            schema_mode="parse",
            destination="discard",
        )
    # A quarantined-to-a-sink row must NOT be counted as discarded.
    factory.data_flow.record_validation_error(
        run_id=run_id,
        node_id="source-0",
        row_data={"i": 99},
        error="bad value",
        schema_mode="parse",
        destination="quarantine_sink",
    )

    with db.connection() as conn:
        assert _count_discarded_rows(conn, run_id) == 2

    other = "run-clean"
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id=other)
    with db.connection() as conn:
        assert _count_discarded_rows(conn, other) == 0


def test_coalesce_run_source_hashes_aggregates_row_hashes() -> None:
    hashes = ("a" * 64, "b" * 64)

    assert _coalesce_run_source_hashes(hashes, run_id="run-1") == stable_hash({"source_data_hashes": list(hashes)})


# I6 — silent-failure-hunter remediation. The cache-store early-return
# is the silent-failure surface; this helper returns the closed-list
# reason for it so the caller can emit a counter attribute. Each
# branch is tested separately so the conditional ordering (which
# resolves overlapping subset relationships — quarantined ⊂ failed,
# routed_failure ⊂ failed, routed_success ⊂ succeeded) is locked in.


def _run_record(
    *,
    status: str = "completed",
    rows_processed: int = 5,
    rows_succeeded: int = 5,
    rows_failed: int = 0,
    rows_routed_success: int = 0,
    rows_routed_failure: int = 0,
    rows_quarantined: int = 0,
) -> RunRecord:
    return RunRecord(
        id=uuid4(),
        session_id=uuid4(),
        state_id=uuid4(),
        status=status,  # type: ignore[arg-type]
        started_at=datetime(2026, 5, 19, tzinfo=UTC),
        finished_at=datetime(2026, 5, 19, tzinfo=UTC) if status != "pending" and status != "running" else None,
        rows_processed=rows_processed,
        rows_succeeded=rows_succeeded,
        rows_failed=rows_failed,
        rows_routed_success=rows_routed_success,
        rows_routed_failure=rows_routed_failure,
        rows_quarantined=rows_quarantined,
        error=None,
        landscape_run_id="landscape-run-1",
        pipeline_yaml="source:\n  plugin: 'null'\nsinks:\n  out:\n    plugin: json\n",
    )


@pytest.mark.asyncio
async def test_replay_cache_entry_attributes_synthesised_landscape_run_to_requester(tmp_path: Path) -> None:
    """Cached tutorial replays must have the same Landscape user attribution as live runs."""
    landscape_url = f"sqlite:///{tmp_path / 'landscape.db'}"
    settings = _make_tutorial_settings(tmp_path, landscape_url=landscape_url)
    current_state = _make_state_record(
        source={"plugin": "null"},
        transform_nodes=[],
        outputs=[{"name": "out", "plugin": "json"}],
    )
    started_at = datetime(2026, 5, 19, tzinfo=UTC)
    run_record = RunRecord(
        id=uuid4(),
        session_id=current_state.session_id,
        state_id=current_state.id,
        status="pending",
        started_at=started_at,
        finished_at=None,
        rows_processed=0,
        rows_succeeded=0,
        rows_failed=0,
        rows_routed_success=0,
        rows_routed_failure=0,
        rows_quarantined=0,
        error=None,
        landscape_run_id=None,
        pipeline_yaml=None,
    )
    cache_entry = TutorialCacheEntry(
        canonical_prompt=CANONICAL_SEED_PROMPT,
        model_id="tutorial-model",
        cached_at=started_at,
        rows=[{"url": "https://example.gov", "rating": 5}],
        source_data_hash="0" * 64,
        llm_call_count=2,
        pipeline_yaml="source:\n  plugin: 'null'\nsinks:\n  out:\n    plugin: json\n",
    )

    class _StubSessionService:
        def __init__(self) -> None:
            self.status_updates: list[tuple[Any, dict[str, Any]]] = []

        async def create_run(self, **_kwargs: Any) -> RunRecord:
            return run_record

        async def update_run_status(self, run_id: Any, **kwargs: Any) -> None:
            self.status_updates.append((run_id, dict(kwargs)))

    await _replay_cache_entry(
        session_service=cast(Any, _StubSessionService()),
        settings=settings,
        session_id=current_state.session_id,
        current_state=current_state,
        cache_entry=cache_entry,
        cache_key=tutorial_cache_key(CANONICAL_SEED_PROMPT, "tutorial-model"),
        user_id="alice",
        auth_provider_type="local",
    )

    with LandscapeDB.from_url(landscape_url, create_tables=False) as db, db.read_only_connection() as conn:
        attribution_row = conn.execute(select(run_attributions_table)).one()

    assert attribution_row.initiated_by_user_id == "alice"
    assert attribution_row.auth_provider_type == "local"


def test_cache_seed_skip_reason_returns_none_for_clean_completed_run() -> None:
    assert _cache_seed_skip_reason(_run_record()) is None


def test_cache_seed_skip_reason_status_not_completed_takes_precedence() -> None:
    # Even with healthy counters, a non-completed status disqualifies.
    # Use ``cancelled`` rather than ``failed`` — RunRecord's Tier-1
    # validator requires a ``failed`` row to carry a non-empty error
    # message; cancelled has no such constraint and exercises the same
    # branch of _cache_seed_skip_reason.
    rr = _run_record(status="cancelled", rows_succeeded=5)
    assert _cache_seed_skip_reason(rr) == "status_not_completed"


def test_cache_seed_skip_reason_zero_rows_processed() -> None:
    rr = _run_record(rows_processed=0, rows_succeeded=0)
    assert _cache_seed_skip_reason(rr) == "zero_rows_processed"


def test_cache_seed_skip_reason_quarantined_resolves_before_failed() -> None:
    # Quarantined ⊂ failed (Tier-1 constraint). The skip_reason must
    # report the MORE SPECIFIC value so operator dashboards can
    # distinguish "row was malformed" from "row raised in transform".
    rr = _run_record(rows_succeeded=4, rows_failed=1, rows_quarantined=1)
    assert _cache_seed_skip_reason(rr) == "rows_quarantined"


def test_cache_seed_skip_reason_routed_failure_resolves_before_failed() -> None:
    # Routed-failure ⊂ failed. Specificity ordering, same reason.
    rr = _run_record(rows_succeeded=4, rows_failed=1, rows_routed_failure=1)
    assert _cache_seed_skip_reason(rr) == "rows_routed_failure"


def test_cache_seed_skip_reason_rows_failed_when_not_subclassified() -> None:
    # A failed row that's neither quarantined nor routed — catch-all.
    rr = _run_record(rows_succeeded=4, rows_failed=1)
    assert _cache_seed_skip_reason(rr) == "rows_failed"


def test_cache_seed_skip_reason_routed_success_indicates_topology_mismatch() -> None:
    # Routed-success rows are still "successful" but their presence
    # means the run had routing — the linear tutorial pipeline
    # shouldn't. Cache cannot seed because the executed topology
    # doesn't match the cached pipeline_yaml.
    rr = _run_record(rows_routed_success=2)
    assert _cache_seed_skip_reason(rr) == "rows_routed_success"


def test_cache_seed_skip_reason_partial_success_catch_all() -> None:
    # rows_succeeded < rows_processed with no failures, no routing —
    # rows in non-terminal states like BUFFERED or other gaps.
    rr = _run_record(rows_processed=5, rows_succeeded=3)
    assert _cache_seed_skip_reason(rr) == "rows_partial_success"


class _RecordingCounter:
    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, object]]] = []

    def add(self, amount: int, *, attributes: dict[str, object]) -> None:
        self.calls.append((amount, dict(attributes)))


@pytest.mark.asyncio
async def test_store_successful_live_projection_emits_skip_counter_when_quarantined(tmp_path: Path, monkeypatch) -> None:
    # End-to-end: a quarantined-row RunRecord should NOT seed the cache
    # AND should emit the counter. The pin guards against a future
    # refactor that drops the record_tutorial_cache_skipped call from
    # the early-return branch (the silent-failure regression I6
    # closes).
    counter = _RecordingCounter()
    monkeypatch.setattr(tutorial_telemetry_module, "_TUTORIAL_CACHE_SKIPPED_COUNTER", counter)
    cache_dir = tmp_path / "tutorial-cache"
    cache_dir.mkdir()
    cache = TutorialCache(cache_dir=cache_dir)
    rr = _run_record(rows_succeeded=4, rows_failed=1, rows_quarantined=1)

    await _store_successful_live_projection(
        cache=cache,
        model_id="test-model-id",
        run_record=rr,
        output=TutorialRunOutput(rows=(), source_data_hash="0" * 64),
        llm_call_count=3,
    )

    assert counter.calls == [(1, {"skip_reason": "rows_quarantined"})]
    # Cache was not seeded — no entries written.
    assert list(cache_dir.iterdir()) == []


@pytest.mark.asyncio
async def test_store_successful_live_projection_seeds_cache_and_does_not_emit_counter_on_clean_run(tmp_path: Path, monkeypatch) -> None:
    # Inverse pin: a clean run MUST seed the cache and NOT emit the
    # skip counter. Together with the quarantined test this locks in
    # the two-branch contract of the helper.
    counter = _RecordingCounter()
    monkeypatch.setattr(tutorial_telemetry_module, "_TUTORIAL_CACHE_SKIPPED_COUNTER", counter)
    cache_dir = tmp_path / "tutorial-cache-clean"
    cache_dir.mkdir()
    cache = TutorialCache(cache_dir=cache_dir)
    rr = _run_record()

    await _store_successful_live_projection(
        cache=cache,
        model_id="test-model-id",
        run_record=rr,
        output=TutorialRunOutput(
            rows=({"url": "ato.gov.au", "score": 5, "rationale": "clear"},),
            source_data_hash="0" * 64,
        ),
        llm_call_count=3,
    )

    assert counter.calls == []
    # Cache was seeded — at least one entry written.
    assert list(cache_dir.iterdir())


def test_tutorial_model_id_includes_composer_model_core_skill_and_deployment_skill(tmp_path: Path) -> None:
    """Cache key must invalidate on any operator-controlled output-determining input.

    The tutorial cache stores output keyed by ``model_id``; promising "compound
    model identifier" in the cache docstring but using only ``composer_model``
    silently serves stale rows whenever the operator edits the composer skill
    markdown (which biases the transform model named in the generated YAML) or
    drops a deployment skill overlay into ``{data_dir}/skills``.
    """
    settings = _make_tutorial_settings(tmp_path, composer_model="anthropic/claude-sonnet-4.5")
    model_id = tutorial_model_id(settings)

    assert "composer=anthropic/claude-sonnet-4.5" in model_id
    _, expected_skill_hash = load_skill_with_hash("pipeline_composer")
    assert f"skill={expected_skill_hash}" in model_id
    empty_deployment_hash = hashlib.sha256(b"").hexdigest()
    assert f"deployment_skill={empty_deployment_hash}" in model_id


def test_tutorial_model_id_changes_when_composer_model_changes(tmp_path: Path) -> None:
    a = tutorial_model_id(_make_tutorial_settings(tmp_path, composer_model="openai/gpt-5"))
    b = tutorial_model_id(_make_tutorial_settings(tmp_path, composer_model="anthropic/claude-sonnet-4.5"))
    assert a != b, "composer_model is supposed to be part of the compound identifier"


def test_tutorial_model_id_changes_when_deployment_skill_overlay_is_added(tmp_path: Path) -> None:
    """Operator-supplied deployment overlay must shift the cache key.

    Drops a ``{data_dir}/skills/pipeline_composer.md`` with arbitrary content
    and asserts the resulting compound identifier differs from the
    no-overlay baseline.
    """
    settings = _make_tutorial_settings(tmp_path, composer_model="openai/gpt-5")
    baseline = tutorial_model_id(settings)

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "pipeline_composer.md").write_text(
        "# Deployment overlay\nUse anthropic/claude-haiku-5.0 for cheap rating tasks.\n",
        encoding="utf-8",
    )

    with_overlay = tutorial_model_id(settings)
    assert baseline != with_overlay, (
        "Adding a deployment skill overlay must invalidate the tutorial cache key — "
        "stale replays under a changed system prompt are a Tier-1 audit lie."
    )


# --- _parse_rows_file / _rows_from_artifacts --------------------------------
# Tier-1 audit invariant: row-parsing distinguishes three failure modes, none
# of which may be silently coalesced into ``[]``:
#   1. file format is not a recognised row format → ``None`` (caller skips)
#   2. file IS a row format but content is structurally corrupt → raise
#   3. file IS a row format and parses cleanly → return rows (possibly empty)


def test_parse_rows_file_returns_none_for_unrecognised_suffix(tmp_path: Path) -> None:
    """An auxiliary artifact (.txt, .parquet, .log) is not a row format.

    Returning ``None`` lets ``_rows_from_artifacts`` cleanly skip it and try
    the next artifact, instead of silently treating "I can't read this" as
    "this had zero rows" — which would shadow Tier-1 corruption behind the
    misleading "no readable file artifact with rows" message.
    """
    for name, content in [("notes.txt", "freeform text"), ("dump.parquet", b"\x00\x01")]:
        path = tmp_path / name
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        assert _parse_rows_file(path) is None, f"{name}: non-row format must return None, not []"


def test_parse_rows_file_raises_for_json_scalar_or_null(tmp_path: Path) -> None:
    """A .json artifact whose top-level is a scalar/null is structurally corrupt.

    The prior implementation fell through and returned [] — Tier-1 corruption
    presented to the operator as "the run finished with no rows".
    """
    for label, content in [
        ("null", "null"),
        ("integer", "42"),
        ("string", '"a string"'),
        ("boolean", "true"),
    ]:
        path = tmp_path / f"output_{label}.json"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(TutorialRunIntegrityError, match="JSON top-level"):
            _parse_rows_file(path)


def test_parse_rows_file_parses_csv_and_returns_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("url,rating\nato.gov.au,5\ndata.gov.au,4\n", encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == [{"url": "ato.gov.au", "rating": "5"}, {"url": "data.gov.au", "rating": "4"}]


def test_parse_rows_file_parses_json_list_of_objects(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text('[{"url": "ato.gov.au", "rating": 5}, {"url": "data.gov.au", "rating": 4}]', encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == [{"url": "ato.gov.au", "rating": 5}, {"url": "data.gov.au", "rating": 4}]


def test_parse_rows_file_parses_empty_csv_as_empty_rows(tmp_path: Path) -> None:
    """A header-only CSV legitimately yields zero rows — that is not corruption."""
    path = tmp_path / "empty.csv"
    path.write_text("url,rating\n", encoding="utf-8")
    rows = _parse_rows_file(path)
    assert rows == []  # legitimate empty result, distinct from "couldn't parse"


def _fake_artifact(artifact_id: str, path_or_uri: str, artifact_type: str = "file") -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(artifact_id=artifact_id, artifact_type=artifact_type, path_or_uri=path_or_uri)


def test_rows_from_artifacts_skips_auxiliary_and_returns_row_artifact_rows(tmp_path: Path) -> None:
    """A run that emits a debug .txt log AND a rows .csv must return the .csv rows.

    Auxiliary artifacts (.txt, .log) must be skipped without crashing the
    projection — only row-format artifacts contribute to the row sequence.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    aux = outputs / "debug.txt"
    aux.write_text("composer chain trace", encoding="utf-8")
    rows_file = outputs / "rows.csv"
    rows_file.write_text("url,rating\nato.gov.au,5\n", encoding="utf-8")

    artifacts = [
        _fake_artifact("aux-1", str(aux)),
        _fake_artifact("rows-1", str(rows_file)),
    ]

    rows = _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-aux")
    assert rows == [{"url": "ato.gov.au", "rating": "5"}]


def test_rows_from_artifacts_distinguishes_no_row_format_from_all_empty(tmp_path: Path) -> None:
    """The 'no row-bearing artifact found' message must name the recognised formats.

    Operator action differs between 'pipeline emitted .parquet but we can't
    read parquet' and 'pipeline emitted rows.csv but it was empty' — the
    prior implementation collapsed both into a single misleading message.
    """
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    only_aux = outputs / "summary.txt"
    only_aux.write_text("freeform", encoding="utf-8")
    artifacts = [_fake_artifact("aux-1", str(only_aux))]

    with pytest.raises(TutorialRunIntegrityError, match="no row-bearing artifact"):
        _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-only-aux")


def test_rows_from_artifacts_raises_when_all_row_artifacts_yield_zero_rows(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    empty = outputs / "empty.csv"
    empty.write_text("url,rating\n", encoding="utf-8")
    artifacts = [_fake_artifact("rows-1", str(empty))]

    with pytest.raises(TutorialRunIntegrityError, match="yielded zero rows"):
        _rows_from_artifacts(artifacts, data_dir=tmp_path, run_id="run-empty")


# --- _plugin_nodes_from_pipeline_dict --------------------------------------
# Production composer YAML (src/elspeth/web/composer/yaml_generator.py:86, 103)
# emits transforms as a LIST of dicts. The engine config loader at
# src/elspeth/core/config.py:1798 requires list-form. The cache-replay parser
# must accept that shape — dict-form is not a production-reachable contract.


def test_parser_handles_list_form_transforms_from_production_composer_yaml() -> None:
    """Production yaml_generator emits transforms as list[dict] — parse them.

    Pre-fix: parser only iterated dict-form transforms, silently dropping every
    transform from production composer YAML. The synthesised Landscape audit
    then recorded only source + sinks for any real tutorial cache replay —
    Tier-1 topology corruption in a different shape than C1.
    """
    doc = {
        "sources": {"primary": {"plugin": "inline_blob", "options": {"rows": [{"url": "ato.gov.au"}]}}},
        "transforms": [
            {"name": "scrape", "plugin": "web_scrape", "input": "source", "on_success": "rate", "on_error": "abort"},
            {"name": "rate", "plugin": "llm_rate", "input": "scrape", "on_success": "out", "on_error": "abort"},
        ],
        "sinks": {"out": {"plugin": "jsonl"}},
    }
    roles = _plugin_nodes_from_pipeline_dict(doc)
    assert roles == (
        (NodeType.SOURCE, "inline_blob"),
        (NodeType.TRANSFORM, "web_scrape"),
        (NodeType.TRANSFORM, "llm_rate"),
        (NodeType.SINK, "jsonl"),
    )


def test_parser_handles_list_form_aggregations() -> None:
    """Production aggregations are list-form (yaml_generator.py:126) — record them.

    Aggregations are plugin invocations like transforms; they have a plugin
    name and version and must appear in the synthesised audit topology.
    """
    doc = {
        "sources": {"primary": {"plugin": "csv"}},
        "transforms": [{"name": "norm", "plugin": "passthrough", "input": "source", "on_success": "batch", "on_error": "abort"}],
        "aggregations": [
            {"name": "batch", "plugin": "batch_stats", "input": "norm", "on_success": "out", "on_error": "abort"},
        ],
        "sinks": {"out": {"plugin": "jsonl"}},
    }
    roles = _plugin_nodes_from_pipeline_dict(doc)
    assert roles == (
        (NodeType.SOURCE, "csv"),
        (NodeType.TRANSFORM, "passthrough"),
        (NodeType.AGGREGATION, "batch_stats"),
        (NodeType.SINK, "jsonl"),
    )


def test_parser_rejects_gates_in_synthesised_replay_yaml() -> None:
    """Gates are inline routing primitives without plugin identity.

    The synthesised audit topology records (NodeType, plugin_name) pairs;
    gates have no plugin to attribute. The tutorial canonical pipeline does
    not generate them. If the cached YAML somehow contains a gate, refuse to
    synthesise a half-truth audit row rather than silently dropping it.
    """
    doc = {
        "sources": {"primary": {"plugin": "csv"}},
        "transforms": [],
        "gates": [{"name": "branch", "input": "source", "condition": "row.x > 0", "routes": []}],
        "sinks": {"out": {"plugin": "jsonl"}},
    }
    with pytest.raises(TutorialRunIntegrityError, match="gates"):
        _plugin_nodes_from_pipeline_dict(doc)


def test_parser_rejects_coalesce_in_synthesised_replay_yaml() -> None:
    doc = {
        "sources": {"primary": {"plugin": "csv"}},
        "coalesce": [{"name": "merge", "branches": ["a", "b"], "policy": "all", "merge": "concat"}],
        "sinks": {"out": {"plugin": "jsonl"}},
    }
    with pytest.raises(TutorialRunIntegrityError, match="coalesce"):
        _plugin_nodes_from_pipeline_dict(doc)


def test_parser_rejects_dict_form_transforms_no_legacy_compat() -> None:
    """Per No Legacy Code Policy, only the production list-form shape is supported.

    Dict-form transforms ``transforms: {name: {plugin: ...}}`` are not a
    contract the engine config loader accepts (core/config.py:1798
    requires list). The parser must reject this shape with a clear error
    rather than partially honouring a non-production format.
    """
    doc = {
        "sources": {"primary": {"plugin": "csv"}},
        "transforms": {"keep": {"plugin": "passthrough"}},
        "sinks": {"out": {"plugin": "jsonl"}},
    }
    with pytest.raises(TutorialRunIntegrityError, match=r"transforms.*must be a list"):
        _plugin_nodes_from_pipeline_dict(doc)


# --- _plugin_nodes_from_composition_state and _state_matches_cached_topology
# Tier-1 audit invariant: cache replay may attach cached pipeline_yaml + rows
# to the user's current_state.id ONLY when both describe the same plugin
# topology. Otherwise the synthesised run records state Y but pipeline Z,
# which is an audit lie. Mismatch → fall through to live compose so the
# audit faithfully reflects what was actually run.


def _make_state_record(
    *,
    source: dict[str, Any] | None,
    transform_nodes: list[dict[str, Any]],
    aggregation_nodes: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]],
) -> CompositionStateRecord:
    """Build a minimal CompositionStateRecord with the topology fields populated."""
    nodes: list[dict[str, Any]] = []
    for transform in transform_nodes:
        nodes.append({"node_type": "transform", **transform})
    if aggregation_nodes:
        for agg in aggregation_nodes:
            nodes.append({"node_type": "aggregation", **agg})
    return CompositionStateRecord(
        id=uuid4(),
        session_id=uuid4(),
        version=1,
        source=source,
        nodes=nodes,
        edges=[],
        outputs=outputs,
        metadata_={},
        is_valid=True,
        validation_errors=None,
        created_at=datetime(2026, 5, 19, tzinfo=UTC),
        derived_from_state_id=None,
        composer_meta=None,
    )


def test_plugin_nodes_from_composition_state_matches_yaml_emitter_order() -> None:
    """State record → topology must produce the same sequence the YAML emitter does.

    The composer YAML emitter (yaml_generator.py) writes source, then all
    transform-typed nodes in nodes-list order, then all aggregation-typed
    nodes, then outputs as sinks. The state-side topology helper must
    produce the same sequence so the cache-replay match check operates on
    apples-to-apples shapes.
    """
    record = _make_state_record(
        source={"plugin": "inline_blob"},
        transform_nodes=[
            {"id": "scrape", "plugin": "web_scrape"},
            {"id": "rate", "plugin": "llm_rate"},
        ],
        outputs=[{"name": "out", "plugin": "jsonl"}],
    )
    topology = _plugin_nodes_from_composition_state(record)
    assert topology == (
        (NodeType.SOURCE, "inline_blob"),
        (NodeType.TRANSFORM, "web_scrape"),
        (NodeType.TRANSFORM, "llm_rate"),
        (NodeType.SINK, "jsonl"),
    )


def test_state_matches_cached_topology_returns_true_when_sequences_match() -> None:
    record = _make_state_record(
        source={"plugin": "inline_blob"},
        transform_nodes=[
            {"id": "scrape", "plugin": "web_scrape"},
            {"id": "rate", "plugin": "llm_rate"},
        ],
        outputs=[{"name": "out", "plugin": "jsonl"}],
    )
    cached_yaml = (
        "source:\n  plugin: inline_blob\n"
        "transforms:\n"
        "  - name: scrape\n    plugin: web_scrape\n    input: source\n    on_success: rate\n    on_error: abort\n"
        "  - name: rate\n    plugin: llm_rate\n    input: scrape\n    on_success: out\n    on_error: abort\n"
        "sinks:\n  out:\n    plugin: jsonl\n"
    )
    assert _state_matches_cached_topology(record, cached_yaml) is True


def test_state_matches_cached_topology_returns_false_when_plugins_differ() -> None:
    """The reviewer's P2-2 scenario: client posts canonical prompt with
    unrelated session state. Cache must refuse replay."""
    record = _make_state_record(
        source={"plugin": "csv"},  # ← user has a csv source, not the canonical inline_blob
        transform_nodes=[{"id": "noop", "plugin": "passthrough"}],
        outputs=[{"name": "out", "plugin": "csv"}],
    )
    cached_yaml = (
        "source:\n  plugin: inline_blob\n"
        "transforms:\n  - name: scrape\n    plugin: web_scrape\n    input: source\n    on_success: out\n    on_error: abort\n"
        "sinks:\n  out:\n    plugin: jsonl\n"
    )
    assert _state_matches_cached_topology(record, cached_yaml) is False


def test_state_matches_cached_topology_returns_false_when_transform_count_differs() -> None:
    """A pipeline with the right plugins but missing/extra transforms is still a mismatch."""
    record = _make_state_record(
        source={"plugin": "inline_blob"},
        transform_nodes=[{"id": "rate", "plugin": "llm_rate"}],  # ← missing the web_scrape transform
        outputs=[{"name": "out", "plugin": "jsonl"}],
    )
    cached_yaml = (
        "source:\n  plugin: inline_blob\n"
        "transforms:\n"
        "  - name: scrape\n    plugin: web_scrape\n    input: source\n    on_success: rate\n    on_error: abort\n"
        "  - name: rate\n    plugin: llm_rate\n    input: scrape\n    on_success: out\n    on_error: abort\n"
        "sinks:\n  out:\n    plugin: jsonl\n"
    )
    assert _state_matches_cached_topology(record, cached_yaml) is False
