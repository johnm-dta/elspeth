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

from typing import cast

from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.manager import PluginManager
from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics
from elspeth.plugins.transforms.batch_distribution_profile import BatchDistributionProfile

NARRATIVE_SUMMARY_TAG = "narrative-summary"


def test_batch_classifier_metrics_class_tags_narrative_summary() -> None:
    """Class-level declaration is the authoritative tag set."""
    assert NARRATIVE_SUMMARY_TAG in BatchClassifierMetrics.capability_tags


def test_batch_distribution_profile_class_tags_narrative_summary() -> None:
    assert NARRATIVE_SUMMARY_TAG in BatchDistributionProfile.capability_tags


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
    assert NARRATIVE_SUMMARY_TAG in plugin.capability_tags

    profile = BatchDistributionProfile(
        {
            "schema": {"mode": "observed"},
            "value_field": "value",
        }
    )
    assert NARRATIVE_SUMMARY_TAG in profile.capability_tags


def test_registered_narrative_summary_transforms_guarantee_summary_output_field() -> None:
    """Catalog-tagged transforms must satisfy the narrative renderer contract."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    tagged_transforms = [plugin_cls for plugin_cls in manager.get_transforms() if NARRATIVE_SUMMARY_TAG in plugin_cls.capability_tags]

    assert tagged_transforms, "Expected at least one builtin narrative-summary transform."

    failures: list[str] = []
    for plugin_cls in tagged_transforms:
        transform_cls = cast(type[BaseTransform], plugin_cls)
        transform = transform_cls(transform_cls.probe_config())
        output_schema_config = transform._output_schema_config
        assert output_schema_config is not None
        guaranteed_fields = frozenset(output_schema_config.guaranteed_fields or ())
        if "summary" not in guaranteed_fields:
            failures.append(f"{plugin_cls.name}: guaranteed_fields={sorted(guaranteed_fields)!r}")

    assert not failures, (
        "Transforms tagged with narrative-summary must guarantee a summary output field. "
        "The frontend renderer switches to NarrativeResults from this catalog tag and "
        "extracts narrative text from output rows named 'summary'. Offences:\n" + "\n".join(failures)
    )
