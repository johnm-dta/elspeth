"""Test that pre-validation and engine-validation paths agree on rejection.

Pre-validation calls config_cls.from_dict(config) — Pydantic validators only.
Engine calls plugin_cls(config) — __init__() guards + Pydantic validators.

If a guard lives only in __init__, pre-validation says "valid" but the engine
rejects at runtime — a confusing false positive. All rejection logic should
live in the Pydantic config model so from_dict() catches it.

This parametric test feeds configs that SHOULD be rejected to BOTH paths and
asserts both reject. A failure means a guard was added to __init__ without a
corresponding model_validator — a regression of the H2 divergence bug.
"""

import pytest

from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.plugins.infrastructure.validation import (
    validate_sink_config,
    validate_source_config,
    validate_transform_config,
)


def _make_observed_schema() -> dict[str, str]:
    return {"mode": "observed"}


# ── Invalid configs that both paths must reject ─────────────────────────

_TRANSFORM_REJECTION_CASES = [
    # ── batch_stats ───────────────────────────────────────────────────────
    pytest.param(
        "batch_stats",
        {
            "schema": _make_observed_schema(),
            "value_field": "amount",
            "group_by": "count",  # collides with aggregate output key
        },
        "group_by.*collides",
        id="batch_stats-group_by-collision",
    ),
    pytest.param(
        "batch_stats",
        {
            "schema": _make_observed_schema(),
            "value_field": "amount",
            "group_by": "mean",  # collides when compute_mean=True (default)
        },
        "group_by.*collides",
        id="batch_stats-group_by-mean-collision",
    ),
    pytest.param(
        "batch_stats",
        {
            "schema": _make_observed_schema(),
            "value_field": "amount",
            "group_by": "sum",  # collides with sum output key
        },
        "group_by.*collides",
        id="batch_stats-group_by-sum-collision",
    ),
    # ── field_mapper ──────────────────────────────────────────────────────
    pytest.param(
        "field_mapper",
        {
            "schema": _make_observed_schema(),
            "mapping": {"name": "full_name", "email": "full_name"},  # duplicate target
        },
        "duplicate target",
        id="field_mapper-duplicate-target",
    ),
    # ── json_explode ──────────────────────────────────────────────────────
    pytest.param(
        "json_explode",
        {
            "schema": _make_observed_schema(),
            "array_field": "items",
            "output_field": "items",  # same as array_field
        },
        "output_field and array_field must differ",
        id="json_explode-output-equals-array",
    ),
    pytest.param(
        "json_explode",
        {
            "schema": _make_observed_schema(),
            "array_field": "items",
            "output_field": "item_index",  # collides with auto-generated index
            "include_index": True,
        },
        "item_index.*conflicts",
        id="json_explode-output-index-collision",
    ),
    # ── line_explode ──────────────────────────────────────────────────────
    pytest.param(
        "line_explode",
        {
            "schema": _make_observed_schema(),
            "source_field": "content",
            "output_field": "content",  # same as source_field
        },
        "output_field and source_field must differ",
        id="line_explode-output-equals-source",
    ),
    # ── truncate ──────────────────────────────────────────────────────────
    pytest.param(
        "truncate",
        {
            "schema": _make_observed_schema(),
            "fields": {"title": 3},
            "suffix": "...",  # suffix_len (3) >= max_len (3)
        },
        "suffix length.*must be less than",
        id="truncate-suffix-exceeds-max-len",
    ),
    # ── value_transform ───────────────────────────────────────────────────
    pytest.param(
        "value_transform",
        {
            "schema": _make_observed_schema(),
            "operations": [],  # empty operations list
        },
        "operations must contain at least one",
        id="value_transform-empty-operations",
    ),
    # ── type_coerce ───────────────────────────────────────────────────────
    pytest.param(
        "type_coerce",
        {
            "schema": _make_observed_schema(),
            "conversions": [],  # empty conversions list
        },
        "conversions must contain at least one",
        id="type_coerce-empty-conversions",
    ),
    # ── batch_replicate ───────────────────────────────────────────────────
    pytest.param(
        "batch_replicate",
        {
            "schema": _make_observed_schema(),
            "copies_field": "n",
            "default_copies": 100,
            "max_copies": 10,  # default > max
        },
        "default_copies.*exceeds max_copies",
        id="batch_replicate-default-exceeds-max",
    ),
    # ── web_scrape ────────────────────────────────────────────────────────
    pytest.param(
        "web_scrape",
        {
            "schema": _make_observed_schema(),
            "url_field": "url",
            "content_field": "body_html",
            "fingerprint_field": "body_html",  # same as content_field
            "http": {
                "abuse_contact": "admin@example.com",
                "scraping_reason": "testing",
            },
        },
        "content_field and fingerprint_field must differ",
        id="web_scrape-field-collision",
    ),
]

_SOURCE_REJECTION_CASES = [
    # ── json source ───────────────────────────────────────────────────────
    pytest.param(
        "json",
        {
            "path": "/tmp/test.jsonl",
            "schema": _make_observed_schema(),
            "on_validation_failure": "quarantine",
            "data_key": "results",  # data_key + .jsonl extension = invalid
            # format is None (auto-detected from .jsonl extension)
        },
        "data_key.*not supported.*JSONL",
        id="json-data_key-auto-detected-jsonl",
    ),
    pytest.param(
        "json",
        {
            "path": "/tmp/test.json",
            "schema": _make_observed_schema(),
            "on_validation_failure": "quarantine",
            "format": "jsonl",
            "data_key": "results",  # explicit jsonl + data_key = invalid
        },
        "data_key.*not supported",
        id="json-data_key-explicit-jsonl",
    ),
    # ── csv source ────────────────────────────────────────────────────────
    pytest.param(
        "csv",
        {
            "path": "/tmp/test.csv",
            "schema": _make_observed_schema(),
            "on_validation_failure": "quarantine",
            "delimiter": ",,",  # must be single character
        },
        "delimiter must be a single character",
        id="csv-multi-char-delimiter",
    ),
    pytest.param(
        "csv",
        {
            "path": "/tmp/test.csv",
            "schema": _make_observed_schema(),
            "on_validation_failure": "quarantine",
            "encoding": "not-a-real-encoding",
        },
        "unknown encoding",
        id="csv-invalid-encoding",
    ),
]

_SINK_REJECTION_CASES = [
    # ── dataverse sink ────────────────────────────────────────────────────
    pytest.param(
        "dataverse",
        {
            "schema": _make_observed_schema(),
            "environment_url": "https://myorg.crm.dynamics.com",
            "auth": {
                "method": "managed_identity",
            },
            "entity": "contacts",
            "field_mapping": {"name": "fullname", "email": "emailaddress1"},
            "alternate_key": "contactid",  # not in field_mapping values
        },
        "alternate_key.*not found in field_mapping",
        id="dataverse-alternate_key-missing",
    ),
    # ── json sink ─────────────────────────────────────────────────────────
    pytest.param(
        "json",
        {
            "path": "/tmp/output.json",
            "schema": _make_observed_schema(),
            "format": "json",
            "mode": "append",  # json array format doesn't support append
        },
        "does not support.*append",
        id="json_sink-json-format-append",
    ),
    # ── csv sink ──────────────────────────────────────────────────────────
    pytest.param(
        "csv",
        {
            "path": "/tmp/output.csv",
            "schema": _make_observed_schema(),
            "delimiter": "TAB",  # must be single character
        },
        "delimiter must be a single character",
        id="csv_sink-multi-char-delimiter",
    ),
    pytest.param(
        "csv",
        {
            "path": "/tmp/output.csv",
            "schema": _make_observed_schema(),
            "encoding": "bogus-999",
        },
        "unknown encoding",
        id="csv_sink-invalid-encoding",
    ),
]


# ── Parametric test: pre-validation path ────────────────────────────────


@pytest.mark.parametrize("transform_type,config,error_pattern", _TRANSFORM_REJECTION_CASES)
def test_prevalidation_rejects_invalid_transform(transform_type, config, error_pattern):
    """Pre-validation (from_dict path) rejects known-invalid transform configs."""
    errors = validate_transform_config(transform_type, config)
    assert errors, f"Expected pre-validation to reject {transform_type} config, but it passed"
    error_text = " ".join(e.message for e in errors)
    assert pytest.importorskip("re").search(error_pattern, error_text, flags=2), (
        f"Expected error matching {error_pattern!r}, got: {error_text}"
    )


@pytest.mark.parametrize("source_type,config,error_pattern", _SOURCE_REJECTION_CASES)
def test_prevalidation_rejects_invalid_source(source_type, config, error_pattern):
    """Pre-validation (from_dict path) rejects known-invalid source configs."""
    errors = validate_source_config(source_type, config)
    assert errors, f"Expected pre-validation to reject {source_type} config, but it passed"
    error_text = " ".join(e.message for e in errors)
    assert pytest.importorskip("re").search(error_pattern, error_text, flags=2), (
        f"Expected error matching {error_pattern!r}, got: {error_text}"
    )


@pytest.mark.parametrize("sink_type,config,error_pattern", _SINK_REJECTION_CASES)
def test_prevalidation_rejects_invalid_sink(sink_type, config, error_pattern):
    """Pre-validation (from_dict path) rejects known-invalid sink configs."""
    errors = validate_sink_config(sink_type, config)
    assert errors, f"Expected pre-validation to reject {sink_type} config, but it passed"
    error_text = " ".join(e.message for e in errors)
    assert pytest.importorskip("re").search(error_pattern, error_text, flags=2), (
        f"Expected error matching {error_pattern!r}, got: {error_text}"
    )


# ── Parametric test: engine-instantiation path ──────────────────────────


@pytest.fixture(scope="module")
def plugin_manager():
    """Shared PluginManager for engine-path tests — avoids re-scanning plugins per parametrized case."""
    from elspeth.plugins.infrastructure.manager import PluginManager

    manager = PluginManager()
    manager.register_builtin_plugins()
    return manager


@pytest.mark.parametrize("transform_type,config,error_pattern", _TRANSFORM_REJECTION_CASES)
def test_engine_rejects_invalid_transform(transform_type, config, error_pattern, plugin_manager):
    """Engine path (plugin_cls(config)) rejects known-invalid transform configs."""
    plugin_cls = plugin_manager.get_transform_by_name(transform_type)

    with pytest.raises((ValueError, PluginConfigError)):
        plugin_cls(config)


@pytest.mark.parametrize("source_type,config,error_pattern", _SOURCE_REJECTION_CASES)
def test_engine_rejects_invalid_source(source_type, config, error_pattern, plugin_manager):
    """Engine path (plugin_cls(config)) rejects known-invalid source configs."""
    plugin_cls = plugin_manager.get_source_by_name(source_type)

    with pytest.raises((ValueError, PluginConfigError)):
        plugin_cls(config)


@pytest.mark.parametrize("sink_type,config,error_pattern", _SINK_REJECTION_CASES)
def test_engine_rejects_invalid_sink(sink_type, config, error_pattern, plugin_manager):
    """Engine path (plugin_cls(config)) rejects known-invalid sink configs."""
    plugin_cls = plugin_manager.get_sink_by_name(sink_type)

    with pytest.raises((ValueError, PluginConfigError)):
        plugin_cls(config)


# ── __init__ guard divergence tests ───────────────────────────────────────
#
# These test the single most dangerous divergence pattern: a rejection guard
# that lives ONLY in __init__() and has no corresponding model_validator.
# Pre-validation would say "valid" but the engine would reject at runtime.


def test_json_explode_on_success_rejected_by_prevalidation():
    """JSONExplode's __init__ guard for on_success must also be caught by pre-validation.

    JSONExplode.__init__() raises PluginConfigError when config contains
    'on_success'.  If this guard is ONLY in __init__ and not in the Pydantic
    model, pre-validation (validate_transform_config) would report zero errors
    for a config that crashes at engine startup.
    """
    config = {
        "schema": _make_observed_schema(),
        "array_field": "items",
        "on_success": "out",  # not allowed in plugin options
    }

    # Pre-validation must reject (not just engine)
    errors = validate_transform_config("json_explode", config)
    assert errors, (
        "Pre-validation accepted json_explode config with 'on_success' — "
        "this means the __init__ guard has no corresponding Pydantic validator, "
        "creating a pre-validation/engine divergence"
    )


# ── Coverage completeness: plugins tested ─────────────────────────────────

# Collect all plugin names that have at least one agreement test case.
_TESTED_TRANSFORM_NAMES = {p.values[0] for p in _TRANSFORM_REJECTION_CASES}
_TESTED_SOURCE_NAMES = {p.values[0] for p in _SOURCE_REJECTION_CASES}
_TESTED_SINK_NAMES = {p.values[0] for p in _SINK_REJECTION_CASES}


def test_all_plugins_with_model_validators_have_agreement_cases():
    """Every plugin whose config has a @model_validator must have at least
    one agreement test case.

    Without this, adding a model_validator to a new plugin creates a
    silent gap: no test proves pre-validation and engine agree on rejection.
    """
    import ast
    import inspect

    from elspeth.plugins.infrastructure.discovery import discover_all_plugins

    discovered = discover_all_plugins()

    # Plugins that legitimately have no rejecting model_validator.
    # Each needs a justification comment.
    EXEMPT = {
        "null",  # NullSource: no config_model
        "llm",  # LLMTransform: provider-dispatched, tested via provider-specific tests
        "rag_retrieval",  # RAG: provider-dispatched config validation
        "azure_content_safety",  # Azure safety: endpoint/api_key field validators only
        "azure_prompt_shield",  # Azure safety: endpoint/api_key field validators only
        "azure_blob",  # Azure: auth delegation, complex cloud config
        "dataverse",  # Dataverse source: query_mode validation, tested indirectly
        "chroma_sink",  # Chroma: connection/schema validators, requires chroma extras
        "azure_blob_sink",  # Azure: auth delegation + template compilation
        "dataverse_sink",  # Dataverse sink: tested via alternate_key case above
    }

    missing: list[str] = []

    type_to_tested = {
        "sources": _TESTED_SOURCE_NAMES,
        "transforms": _TESTED_TRANSFORM_NAMES,
        "sinks": _TESTED_SINK_NAMES,
    }

    for plugin_type, plugins in discovered.items():
        tested_names = type_to_tested[plugin_type]
        for cls in plugins:
            plugin_name: str = cls.name
            if plugin_name in EXEMPT or plugin_name in tested_names:
                continue

            # Check if the config model has any @model_validator
            config_model = cls.get_config_model()
            if config_model is None:
                continue

            try:
                source = inspect.getsource(config_model)
            except (OSError, TypeError):
                continue

            tree = ast.parse(source)
            has_model_validator = any(
                isinstance(node, ast.FunctionDef)
                and any(
                    isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "model_validator" for d in node.decorator_list
                )
                for node in ast.walk(tree)
            )

            if has_model_validator:
                missing.append(f"{plugin_type}/{plugin_name}")

    assert not missing, (
        "Plugins with @model_validator but no agreement test case. "
        "Add a rejection case to the appropriate _*_REJECTION_CASES list, "
        "or add to EXEMPT with justification:\n" + "\n".join(f"  - {m}" for m in missing)
    )
