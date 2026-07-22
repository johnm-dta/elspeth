"""Tests for the Explain narrative builder."""

from __future__ import annotations

import pytest

from elspeth.web.audit_readiness.explain import build_narrative
from elspeth.web.composer.state import (
    CompositionState,
    PipelineMetadata,
    SourceSpec,
)

# Import shared factories (co-located in test_service.py or extracted to
# tests/unit/web/audit_readiness/conftest.py if this module grows).
# These factories cover ALL required NodeSpec/OutputSpec kwargs (review B1, B2).
from tests.unit.web.audit_readiness.test_service import _policy_readiness_snapshot, make_node_spec, make_output_spec


def _state(*, source_plugin="csv", transforms=(), sinks=(("out", "csv"),)):
    src = (
        SourceSpec(
            plugin=source_plugin,
            on_success="src_out",
            options={},
            on_validation_failure="quarantine",
        )
        if source_plugin is not None
        else None
    )
    nodes = tuple(
        make_node_spec(
            nid,
            plg,
            input="src_out" if i == 0 else f"t{i - 1}_out",
            on_success=f"t{i}_out",
        )
        for i, (nid, plg) in enumerate(transforms)
    )
    outputs = tuple(make_output_spec(n, p) for n, p in sinks)
    return CompositionState(
        source=src,
        nodes=nodes,
        edges=(),
        outputs=outputs,
        metadata=PipelineMetadata(name="t", description=""),
        version=1,
    )


def test_opens_with_recorded_promise():
    text = build_narrative(_state(), retention_days=90)
    assert text.startswith("When you run this pipeline, ELSPETH will record:")


def test_names_source_plugin():
    text = build_narrative(_state(source_plugin="csv"), retention_days=90)
    assert "csv" in text.lower()


def test_walks_transforms_in_order():
    text = build_narrative(
        _state(transforms=(("t1", "passthrough"), ("t2", "llm"))),
        retention_days=90,
    )
    assert text.index("t1") < text.index("t2")


def test_calls_out_llm_recording_details():
    text = build_narrative(_state(transforms=(("judge", "llm"),)), retention_days=90)
    assert "prompt" in text.lower()
    assert "response" in text.lower() or "model" in text.lower()


def test_includes_each_sink():
    text = build_narrative(
        _state(sinks=(("primary", "csv"), ("backup", "json"))),
        retention_days=90,
    )
    assert "primary" in text and "backup" in text


def test_mentions_retention():
    text = build_narrative(_state(), retention_days=42)
    assert "42" in text


def test_includes_sanitized_plugin_policy_readiness_rows():
    readiness = _policy_readiness_snapshot(tutorial_profile=None)

    text = build_narrative(
        _state(),
        retention_days=42,
        plugin_policy_readiness=readiness,
    )

    assert "Web plugin policy readiness:" in text
    assert "Tutorial LLM profile: error" in text


def test_is_deterministic():
    s = _state(transforms=(("t", "passthrough"),))
    assert build_narrative(s, retention_days=90) == build_narrative(s, retention_days=90)


def test_no_source_explains_incomplete():
    text = build_narrative(_state(source_plugin=None), retention_days=90)
    assert "no source" in text.lower() or "incomplete" in text.lower()


def test_closes_with_evidence_promise():
    text = build_narrative(_state(), retention_days=90)
    assert "evidence" in text.lower() or "answer" in text.lower()


def test_calls_out_dataverse_sink_as_boundary():
    """Pins that ``explain.py`` emits a boundary narrative when the plugin
    name is ``'dataverse'``. The production code dispatches by plugin name
    (``if plugin == "dataverse":`` in ``audit_readiness/explain.py``)
    because the narrative text is plugin-specific — each external-boundary
    sink has tailored prose naming the receiving system (Dataverse
    instance, Azure Blob container, etc.).

    Boundary classification at the panel level (``_build_plugin_trust_row``
    in ``audit_readiness/service.py``) is verified by
    ``tests/unit/web/audit_readiness/test_boundary_predicate_parity.py``;
    the (kind, determinism) predicate there independently confirms that
    every Sink is a boundary plugin.
    """
    text = build_narrative(_state(sinks=(("primary", "dataverse"),)), retention_days=90)
    assert "Dataverse" in text
    assert "external boundary" in text.lower()
    assert "Dataverse instance" in text


def test_transform_node_with_none_plugin_raises_runtime_error():
    state = _state(transforms=(("bad_transform", None),))

    with pytest.raises(RuntimeError, match=r"bad_transform.*plugin=None"):
        build_narrative(state, retention_days=90)
