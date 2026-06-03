"""Tests for the narrative-summary ``capability_tags`` opt-in on the bootstrap pair.

Phase 6A Task 8 (UX redesign 2026-05) — B6 simplification per multi-reviewer
adjudication. The opt-in rides the existing open-vocabulary
``capability_tags`` channel rather than adding a new ClassVar boolean,
Protocol modification, or catalog wire field. The frontend reads the tag
list it already receives via the catalog response.

Wire contract for the tag: an opted-in transform's output schema must
include a ``summary`` field that Phase 6B's narrative renderer surfaces.
"""

from __future__ import annotations

from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics
from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile


def test_batch_classifier_metrics_class_tags_narrative_summary() -> None:
    """Class-level declaration is the authoritative tag set."""
    assert "narrative-summary" in BatchClassifierMetrics.capability_tags


def test_batch_distribution_profile_class_tags_narrative_summary() -> None:
    assert "narrative-summary" in BatchDistributionProfile.capability_tags


def test_narrative_summary_tag_on_real_plugin_instance() -> None:
    """Q5 (quality-reviewer guard): the class attribute survives instantiation.

    CLAUDE.md mandates that integration tests use ``from_plugin_instances()``;
    the equivalent unit-level assertion is "instantiate the plugin, then read
    the tag off the instance." Catches the failure mode where the class
    attribute is correct but lost through some metaclass-style wrapping.
    """
    plugin = BatchClassifierMetrics(
        {
            "schema": {"mode": "observed"},
            "actual_field": "actual",
            "predicted_field": "predicted",
        }
    )
    assert "narrative-summary" in plugin.capability_tags

    profile = BatchDistributionProfile(
        {
            "schema": {"mode": "observed"},
            "value_field": "value",
        }
    )
    assert "narrative-summary" in profile.capability_tags
