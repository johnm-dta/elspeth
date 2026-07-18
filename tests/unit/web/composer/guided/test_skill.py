"""Tests for current per-step guided chat skills and sample redaction."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.prompts import _summarize_sample_row, load_step_chat_skill
from elspeth.web.composer.guided.protocol import GuidedStep


@pytest.mark.parametrize("step", list(GuidedStep))
def test_each_step_chat_skill_loads_with_base_preamble(step: GuidedStep) -> None:
    text = load_step_chat_skill(step)

    assert "# Guided Pipeline Composer" in text
    assert "invent" in text.lower() or "anti-fabrication" in text.lower()


def test_step_chat_skills_are_scoped() -> None:
    markers = {
        GuidedStep.STEP_1_SOURCE: "### Step 1 — Source",
        GuidedStep.STEP_2_SINK: "### This stage: the output",
        GuidedStep.STEP_3_TRANSFORMS: "### This stage: the transforms",
        GuidedStep.STEP_4_WIRE: "### Step 4 — Wiring constraints",
    }
    for step, marker in markers.items():
        text = load_step_chat_skill(step)
        assert marker in text
        assert all(other_marker not in text for other_step, other_marker in markers.items() if other_step is not step)


def test_sample_row_projection_redacts_values() -> None:
    projection = _summarize_sample_row(
        {
            "email": "person@example.test",
            "api_key": "sk-test-secret-row-value",
            "profile_url": "https://example.test/private?token=secret",
            "note": "customer asked for refunds",
        }
    )

    rendered = repr(projection)
    assert "person@example.test" not in rendered
    assert "sk-test-secret-row-value" not in rendered
    assert "https://example.test/private" not in rendered
    assert "customer asked for refunds" not in rendered
    assert set(projection.values()) == {
        "<sample:email-like>",
        "<sample:secret-like>",
        "<sample:url>",
        "<sample:string:26-chars>",
    }


def test_step_3_skill_keeps_fail_closed_field_mapper_rules() -> None:
    text = load_step_chat_skill(GuidedStep.STEP_3_TRANSFORMS)

    assert "mapping **keys**" in text
    assert "immediate upstream contract guarantees" in text
    assert "mapped output **targets**" in text
    assert "unproven source" in text


def test_step_4_skill_has_no_fixed_linear_topology_contract() -> None:
    text = load_step_chat_skill(GuidedStep.STEP_4_WIRE)

    assert "Exact reviewed graph" in text
    assert "chain_in" not in text
    assert "chain_{k}" not in text
    assert 'emits `"main"`' not in text
