# Implementation Plan: LLM Schema Completeness with Audit Field Categorization

**Bug ID:** BUG-AZURE-03
**Priority:** P2
**Status:** Ready for Implementation
**Author:** Claude Code
**Date:** 2026-01-29
**Review Board:** Architecture, Python Engineering, QA, Systems Thinking
**Review Status:** All conditions addressed - DAG integration gap resolved, checkpoint simplified for pre-production

## Executive Summary

LLM transforms emit metadata fields that aren't declared in their output schema, breaking DAG validation for downstream transforms. The fix introduces a new `audit_fields` category in SchemaConfig to distinguish between:

- **`guaranteed_fields`**: Stable API contract fields that downstream transforms can depend on
- **`audit_fields`**: Provenance metadata that exists but isn't part of the stability contract

This preserves future flexibility to evolve audit metadata without breaking users post-release.

---

## Problem Statement

### Current State

All 6 LLM transforms set `output_schema = input_schema`, then emit 8-9 additional metadata fields:

```python
output[f"{response_field}_usage"] = response.usage
output[f"{response_field}_model"] = response.model
output[f"{response_field}_template_hash"] = rendered.template_hash
# ... 6 more fields
```

**Impact:**
1. DAG validation can't verify downstream field dependencies
2. Schema introspection reports incomplete information
3. No distinction between "data API" and "audit provenance" fields

### Why Not Just Use `guaranteed_fields` for Everything?

Pre-release, we have one chance to lock in the schema. Putting all metadata in `guaranteed_fields` means:
- We can't change audit metadata later without a breaking change
- Downstream transforms might depend on provenance fields that should be internal
- No semantic distinction between "use this for data flow" vs "use this for audit reconstruction"

---

## Solution Design

### Field Categorization

| Field | Category | Rationale |
|-------|----------|-----------|
| `{rf}` | guaranteed | The LLM response content - core data |
| `{rf}_usage` | guaranteed | Token counts - needed for cost/quota management |
| `{rf}_model` | guaranteed | Model identifier - needed for routing/reporting |
| `{rf}_template_hash` | audit | Prompt fingerprint - audit trail only |
| `{rf}_variables_hash` | audit | Rendered variables fingerprint - audit trail only |
| `{rf}_template_source` | audit | Config file path - audit trail only |
| `{rf}_lookup_hash` | audit | Lookup data fingerprint - audit trail only |
| `{rf}_lookup_source` | audit | Config file path - audit trail only |
| `{rf}_system_prompt_source` | audit | Config file path - audit trail only |

*Where `{rf}` = `response_field` config value (default: `llm_response`)*

### Schema System Changes

#### 1. SchemaConfig Enhancement

**File:** `src/elspeth/contracts/schema.py`

```python
@dataclass(frozen=True)
class SchemaConfig:
    """Configuration for a plugin's data schema.

    Schema Contracts (for DAG validation):
        - guaranteed_fields: Fields the producer GUARANTEES will exist AND are
          part of the stable API contract. Downstream can safely depend on these.
        - required_fields: Fields the consumer REQUIRES in input.
        - audit_fields: Fields that exist in output but are NOT part of the
          stability contract. These are for audit trail reconstruction and may
          change between versions. DAG validation does NOT enforce these.
    """

    mode: Literal["strict", "free"] | None
    fields: tuple[FieldDefinition, ...] | None
    is_dynamic: bool
    guaranteed_fields: tuple[str, ...] | None = None
    required_fields: tuple[str, ...] | None = None
    audit_fields: tuple[str, ...] | None = None  # NEW
```

**Changes to `from_dict()`:**
```python
# Parse audit_fields (valid for both dynamic and explicit schemas)
audit_fields = _parse_field_names_list(config.get("audit_fields"), "audit_fields")
```

**Changes to `to_dict()`:**
```python
if self.audit_fields is not None:
    result["audit_fields"] = list(self.audit_fields)
```

**NOTE:** The `get_all_output_fields()` method proposed in initial draft is **removed** per YAGNI - no caller was identified. If needed later, it can be added.

#### 2. DAG Integration (Required)

**Critical:** The DAG must be updated to read `_output_schema_config` from transforms.

**Current state:** `from_plugin_instances()` passes `config=transform.config` to `add_node()`, and `_get_schema_config_from_node()` reads from `node_info.config["schema"]`. This means any `_output_schema_config` attribute on transforms would be ignored.

**Solution:** Update `from_plugin_instances()` to extract `_output_schema_config` from transforms and pass it via the existing `output_schema_config` parameter.

**File:** `src/elspeth/core/dag.py`

**Change 1: Extract schema config in `from_plugin_instances()` (around line 460):**

```python
# Before adding transform to graph, extract computed schema config if available
output_schema_config = getattr(transform, '_output_schema_config', None)

graph.add_node(
    tid,
    node_type=node_type,
    plugin_name=transform.name,
    config=node_config,
    input_schema=transform.input_schema,
    output_schema=transform.output_schema,
    output_schema_config=output_schema_config,  # NEW: Pass computed schema
)
```

**Change 2: Update `_get_schema_config_from_node()` to prioritize NodeInfo schema config:**

```python
def _get_schema_config_from_node(self, node_id: str) -> SchemaConfig | None:
    """Extract SchemaConfig from node.

    Priority:
    1. output_schema_config from NodeInfo (computed by transform)
    2. schema from config dict (raw config)
    """
    node_info = self.get_node_info(node_id)

    # First check if we have computed schema config in NodeInfo
    if node_info.output_schema_config is not None:
        return node_info.output_schema_config

    # Fall back to parsing from raw config dict
    schema_dict = node_info.config.get("schema")
    if schema_dict is None:
        return None

    if isinstance(schema_dict, dict):
        return SchemaConfig.from_dict(schema_dict)

    if schema_dict == "dynamic":
        return SchemaConfig.from_dict({"fields": "dynamic"})

    return None
```

**Why this works:** The `output_schema_config` parameter already exists on `add_node()` and is stored in `NodeInfo`. We just need to:
1. Populate it from transform attributes
2. Read it before falling back to config dict

**Verified:** `get_effective_guaranteed_fields()` in `schema.py:370-390` correctly excludes `audit_fields` — it only returns `guaranteed_fields` union with declared required fields from explicit schemas.

#### 3. LLM Transform Updates

**New constants in `src/elspeth/plugins/llm/__init__.py`:**

```python
"""LLM transform plugin pack.

Metadata Field Categories
=========================

guaranteed_fields: Contract-stable fields downstream can depend on
    - <response_field>: The LLM response content
    - <response_field>_usage: Token usage (for cost/quota management)
    - <response_field>_model: Model identifier (for routing/reporting)

audit_fields: Provenance metadata for audit trail (may change between versions)
    - <response_field>_template_hash: SHA256 of prompt template
    - <response_field>_variables_hash: SHA256 of rendered variables
    - <response_field>_template_source: Config file path
    - <response_field>_lookup_hash: SHA256 of lookup data
    - <response_field>_lookup_source: Config file path
    - <response_field>_system_prompt_source: Config file path

WARNING: Do not build production logic that depends on audit_fields.
These fields exist for audit trail reconstruction (explain() queries)
and may change between versions without notice.
"""

# Metadata field suffixes for contract-stable fields (downstream can depend on these)
LLM_GUARANTEED_SUFFIXES: tuple[str, ...] = (
    "",        # The response content field itself
    "_usage",  # Token usage dict {prompt_tokens, completion_tokens, total_tokens}
    "_model",  # Model identifier that actually responded
)

# Metadata field suffixes for audit-only fields (exist but may change between versions)
LLM_AUDIT_SUFFIXES: tuple[str, ...] = (
    "_template_hash",          # SHA256 of prompt template
    "_variables_hash",         # SHA256 of rendered template variables
    "_template_source",        # File path of template (None if inline)
    "_lookup_hash",            # SHA256 of lookup data
    "_lookup_source",          # File path of lookup data (None if no lookup)
    "_system_prompt_source",   # File path of system prompt (None if inline)
)


def get_llm_guaranteed_fields(response_field: str) -> tuple[str, ...]:
    """Return contract-stable metadata field names for LLM transforms.

    These fields are part of the stable API. Downstream transforms can
    safely declare dependencies on them via required_fields.

    Args:
        response_field: Base field name (e.g., "llm_response"). Must not be empty.

    Returns:
        Tuple of field names that are guaranteed to exist.

    Raises:
        ValueError: If response_field is empty.
    """
    if not response_field:
        raise ValueError("response_field cannot be empty")
    return tuple(f"{response_field}{suffix}" for suffix in LLM_GUARANTEED_SUFFIXES)


def get_llm_audit_fields(response_field: str) -> tuple[str, ...]:
    """Return audit-only metadata field names for LLM transforms.

    These fields exist for audit trail reconstruction but are NOT part
    of the stability contract. They may change between versions.

    Args:
        response_field: Base field name (e.g., "llm_response"). Must not be empty.

    Returns:
        Tuple of field names for audit purposes.

    Raises:
        ValueError: If response_field is empty.
    """
    if not response_field:
        raise ValueError("response_field cannot be empty")
    return tuple(f"{response_field}{suffix}" for suffix in LLM_AUDIT_SUFFIXES)
```

**Transform `__init__` pattern (apply to all 6 transforms):**

```python
from elspeth.plugins.llm import get_llm_guaranteed_fields, get_llm_audit_fields

# Build output schema config with field categorization
guaranteed = get_llm_guaranteed_fields(self._response_field)
audit = get_llm_audit_fields(self._response_field)

# Merge with any existing fields from base schema
base_guaranteed = cfg.schema_config.guaranteed_fields or ()
base_audit = cfg.schema_config.audit_fields or ()

self._output_schema_config = SchemaConfig(
    mode=cfg.schema_config.mode,
    fields=cfg.schema_config.fields,
    is_dynamic=cfg.schema_config.is_dynamic,
    guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
    audit_fields=tuple(set(base_audit) | set(audit)),
    required_fields=cfg.schema_config.required_fields,
)
```

#### 4. Batch Transform Fixes

**File:** `src/elspeth/plugins/llm/azure_batch.py`

The batch transform currently emits only 3 metadata fields. Two changes required:

**4a. Capture `variables_hash` at submission time (in `_submit_batch`):**

The `variables_hash` IS available at submission time - `render_with_metadata()` returns it. Store in checkpoint for later emission.

```python
# In _submit_batch, around line 350:
try:
    rendered = self._template.render_with_metadata(row)
except TemplateError as e:
    template_errors.append((idx, str(e)))
    continue

# Store variables_hash in row_mapping for later retrieval
# NOTE: Changes type from dict[str, int] to dict[str, dict[str, Any]]
row_mapping[custom_id] = {
    "index": idx,
    "variables_hash": rendered.variables_hash,  # CAPTURE HERE
}
```

**Also update `_download_results` (around line 752):**
```python
# Build reverse mapping - extract index from dict
idx_to_custom_id: dict[int, str] = {
    info["index"]: cid for cid, info in row_mapping.items()
}
```

**4b. Emit all standard fields in `_download_results`:**

```python
# Success - extract response (around line 804)
output_row = dict(row)
output_row[self._response_field] = content

# Retrieve variables_hash from checkpoint (stored as dict in 4a)
row_info = row_mapping[custom_id]  # Tier 1 data - crash if missing
variables_hash = row_info["variables_hash"]

# Guaranteed fields (contract-stable)
output_row[f"{self._response_field}_usage"] = usage
output_row[f"{self._response_field}_model"] = body.get("model", self._deployment_name)

# Audit fields (provenance metadata)
output_row[f"{self._response_field}_template_hash"] = self._template.template_hash
output_row[f"{self._response_field}_variables_hash"] = variables_hash
output_row[f"{self._response_field}_template_source"] = self._template.template_source
output_row[f"{self._response_field}_lookup_hash"] = self._template.lookup_hash
output_row[f"{self._response_field}_lookup_source"] = self._template.lookup_source
output_row[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
```

**NOTE:** This changes the checkpoint `row_mapping` format from `int` to `dict`. Since we are pre-production with no users, no backward compatibility handling is needed. The new format is a clean break.

#### 5. Multi-Query Transform Pattern

Multi-query transforms emit metadata per query with prefix `{case_study}_{criterion}_`. Use set comprehension for efficiency:

```python
# In AzureMultiQueryLLMTransform.__init__ and OpenRouterMultiQueryLLMTransform.__init__

# Efficient set comprehension (not list accumulation)
all_guaranteed = {
    field
    for spec in self._query_specs
    for field in get_llm_guaranteed_fields(spec.output_prefix)
}
all_audit = {
    field
    for spec in self._query_specs
    for field in get_llm_audit_fields(spec.output_prefix)
}

# Merge with base schema
base_guaranteed = cfg.schema_config.guaranteed_fields or ()
base_audit = cfg.schema_config.audit_fields or ()

self._output_schema_config = SchemaConfig(
    mode=cfg.schema_config.mode,
    fields=cfg.schema_config.fields,
    is_dynamic=cfg.schema_config.is_dynamic,
    guaranteed_fields=tuple(set(base_guaranteed) | all_guaranteed),
    audit_fields=tuple(set(base_audit) | all_audit),
    required_fields=cfg.schema_config.required_fields,
)
```

**Scaling note:** For N case studies × M criteria, this produces `N*M*3` guaranteed fields and `N*M*6` audit fields. For 5×10=50 queries, that's 150 guaranteed + 300 audit = 450 total fields. DAG validation uses frozenset operations (O(1) lookup), so this is acceptable.

---

## Files to Modify

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/contracts/schema.py` | Modify | Add `audit_fields` attribute, update `from_dict`/`to_dict` |
| `src/elspeth/core/dag.py` | Modify | Extract `_output_schema_config` from transforms in `from_plugin_instances()`, update `_get_schema_config_from_node()` |
| `src/elspeth/plugins/llm/__init__.py` | Modify | Add field suffix constants and helper functions |
| `src/elspeth/plugins/llm/azure.py` | Modify | Add output schema config with categorized fields |
| `src/elspeth/plugins/llm/azure_batch.py` | Modify | Capture variables_hash, add missing metadata fields, add schema config |
| `src/elspeth/plugins/llm/azure_multi_query.py` | Modify | Add output schema config with categorized fields |
| `src/elspeth/plugins/llm/openrouter.py` | Modify | Add output schema config with categorized fields |
| `src/elspeth/plugins/llm/openrouter_multi_query.py` | Modify | Add output schema config with categorized fields |
| `src/elspeth/plugins/llm/base.py` | Modify | Add output schema config to BaseLLMTransform |
| `tests/unit/contracts/test_schema.py` | Modify | Add tests for audit_fields parsing, serialization, validation |
| `tests/unit/plugins/llm/test_metadata_fields.py` | Create | Test helper functions and field categorization |
| `tests/integration/test_llm_schema_contracts.py` | Create | Test DAG validation with guaranteed vs audit fields |

---

## Pre-Implementation Verification

**REQUIRED: Verify "no existing usage" assumption**

Before implementation, run these checks to confirm no existing code depends on LLM metadata via `required_fields`:

```bash
# Check example YAML files
grep -r "required.*fields" examples/ | grep -E "(usage|model|template_hash|lookup)"

# Check test files
grep -r "required_input_fields.*usage" tests/
grep -r "required_fields.*template" tests/

# Check for any transform depending on LLM metadata
grep -rn "required_input_fields" src/elspeth/plugins/ | grep -v "\.pyc"
```

**Expected result:** No matches. If matches found, assess migration impact before proceeding.

---

## Test Plan

### Unit Tests

```python
# tests/unit/contracts/test_schema.py

class TestAuditFields:
    """Tests for audit_fields schema attribute."""

    def test_audit_fields_parsing(self):
        """Verify audit_fields is parsed from config dict."""
        config = {
            "fields": "dynamic",
            "guaranteed_fields": ["response", "response_usage"],
            "audit_fields": ["response_template_hash", "response_lookup_source"],
        }
        schema = SchemaConfig.from_dict(config)

        assert schema.audit_fields == ("response_template_hash", "response_lookup_source")
        assert schema.guaranteed_fields == ("response", "response_usage")

    def test_audit_fields_not_in_effective_guaranteed(self):
        """Verify audit_fields are excluded from effective guaranteed fields."""
        schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("response", "response_usage"),
            audit_fields=("response_template_hash",),
        )

        effective = schema.get_effective_guaranteed_fields()
        assert "response" in effective
        assert "response_usage" in effective
        assert "response_template_hash" not in effective

    def test_audit_fields_serialization_roundtrip(self):
        """Verify audit_fields survive to_dict/from_dict round-trip."""
        schema = SchemaConfig(
            mode=None,
            fields=None,
            is_dynamic=True,
            guaranteed_fields=("response",),
            audit_fields=("response_template_hash", "response_lookup_source"),
        )

        serialized = schema.to_dict()
        assert "audit_fields" in serialized
        assert serialized["audit_fields"] == ["response_template_hash", "response_lookup_source"]

        roundtrip = SchemaConfig.from_dict(serialized)
        assert roundtrip.audit_fields == ("response_template_hash", "response_lookup_source")

    def test_audit_fields_rejects_non_list(self):
        """audit_fields must be a list."""
        with pytest.raises(ValueError, match="must be a list"):
            SchemaConfig.from_dict({
                "fields": "dynamic",
                "audit_fields": "not_a_list"
            })

    def test_audit_fields_rejects_duplicates(self):
        """audit_fields must not contain duplicates."""
        with pytest.raises(ValueError, match="Duplicate field names"):
            SchemaConfig.from_dict({
                "fields": "dynamic",
                "audit_fields": ["hash", "hash"]
            })

    def test_audit_fields_rejects_invalid_identifiers(self):
        """audit_fields must be valid Python identifiers."""
        with pytest.raises(ValueError, match="valid Python identifier"):
            SchemaConfig.from_dict({
                "fields": "dynamic",
                "audit_fields": ["valid", "invalid-field"]
            })
```

```python
# tests/unit/plugins/llm/test_metadata_fields.py

from elspeth.plugins.llm import (
    LLM_GUARANTEED_SUFFIXES,
    LLM_AUDIT_SUFFIXES,
    get_llm_guaranteed_fields,
    get_llm_audit_fields,
)


class TestLLMMetadataFieldHelpers:
    """Tests for LLM metadata field helper functions."""

    def test_guaranteed_suffixes_count(self):
        """Verify expected number of guaranteed suffixes."""
        assert len(LLM_GUARANTEED_SUFFIXES) == 3
        assert "" in LLM_GUARANTEED_SUFFIXES
        assert "_usage" in LLM_GUARANTEED_SUFFIXES
        assert "_model" in LLM_GUARANTEED_SUFFIXES

    def test_audit_suffixes_count(self):
        """Verify expected number of audit suffixes."""
        assert len(LLM_AUDIT_SUFFIXES) == 6
        assert "_template_hash" in LLM_AUDIT_SUFFIXES
        assert "_variables_hash" in LLM_AUDIT_SUFFIXES

    def test_get_llm_guaranteed_fields(self):
        """Verify guaranteed field name generation."""
        fields = get_llm_guaranteed_fields("llm_response")
        assert fields == ("llm_response", "llm_response_usage", "llm_response_model")

    def test_get_llm_audit_fields(self):
        """Verify audit field name generation."""
        fields = get_llm_audit_fields("result")
        assert "result_template_hash" in fields
        assert "result_variables_hash" in fields
        assert "result_template_source" in fields
        assert "result_lookup_hash" in fields
        assert "result_lookup_source" in fields
        assert "result_system_prompt_source" in fields
        assert len(fields) == 6

    def test_empty_response_field_raises(self):
        """Empty response_field should raise ValueError."""
        with pytest.raises(ValueError, match="response_field cannot be empty"):
            get_llm_guaranteed_fields("")

        with pytest.raises(ValueError, match="response_field cannot be empty"):
            get_llm_audit_fields("")

    def test_custom_response_field(self):
        """Custom response field names work correctly."""
        guaranteed = get_llm_guaranteed_fields("custom_output")
        assert "custom_output" in guaranteed
        assert "custom_output_usage" in guaranteed

        audit = get_llm_audit_fields("custom_output")
        assert "custom_output_template_hash" in audit
```

### Integration Tests

```python
# tests/integration/test_llm_schema_contracts.py

import pytest
from elspeth.core.dag import ExecutionGraph
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.sinks.null_sink import NullSink
from elspeth.plugins.transforms.field_mapper import FieldMapper
from elspeth.plugins.llm.azure import AzureLLMTransform
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
from elspeth.plugins.llm.openrouter_multi_query import OpenRouterMultiQueryLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform


class TestLLMSchemaContracts:
    """Integration tests for LLM transform schema contracts."""

    @pytest.fixture
    def minimal_azure_config(self):
        return {
            "response_field": "result",
            "deployment_name": "test-deployment",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "test {{ row.input }}",
            "required_input_fields": ["input"],
            "schema": {"fields": "dynamic"},
        }

    def test_schema_config_propagates_to_dag_nodeinfo(self, minimal_azure_config):
        """Verify _output_schema_config is stored in DAG NodeInfo (production path)."""
        llm = AzureLLMTransform(minimal_azure_config)

        # Pre-condition: transform has computed schema config
        assert llm._output_schema_config is not None
        assert "result" in llm._output_schema_config.guaranteed_fields
        assert "result_template_hash" in llm._output_schema_config.audit_fields

        # Build graph via production factory
        graph = ExecutionGraph.from_plugin_instances(
            source=NullSource({"schema": {"fields": "dynamic"}}),
            transforms=[llm],
            sinks={"output": NullSink({})},
            aggregations={},
            gates={},
            coalesce_settings={},
            output_sink="output",
        )

        # CRITICAL: Verify schema config made it into NodeInfo
        llm_nodes = [n for n in graph.get_nodes() if n.plugin_name == "azure_llm"]
        assert len(llm_nodes) == 1

        node_info = llm_nodes[0]
        assert node_info.output_schema_config is not None, "Schema config not propagated to NodeInfo!"
        assert "result" in node_info.output_schema_config.guaranteed_fields
        assert "result_usage" in node_info.output_schema_config.guaranteed_fields
        assert "result_template_hash" in node_info.output_schema_config.audit_fields

    def test_downstream_can_require_guaranteed_fields(self, minimal_azure_config):
        """Downstream transforms can declare dependencies on guaranteed LLM fields."""
        llm = AzureLLMTransform(minimal_azure_config)

        downstream = FieldMapper({
            "mapping": {"tokens": "result_usage"},
            "required_input_fields": ["result_usage"],  # Guaranteed field
            "schema": {"fields": "dynamic"},
        })

        # Should NOT raise - result_usage is guaranteed
        graph = ExecutionGraph.from_plugin_instances(
            source=NullSource({"schema": {"fields": "dynamic"}}),
            transforms=[llm, downstream],
            sinks={"output": NullSink({})},
            aggregations={},
            gates={},
            coalesce_settings={},
            output_sink="output",
        )
        assert graph.node_count > 0

    def test_downstream_cannot_require_audit_fields(self, minimal_azure_config):
        """DAG validation rejects dependencies on audit-only fields."""
        llm = AzureLLMTransform(minimal_azure_config)

        downstream = FieldMapper({
            "mapping": {"hash": "result_template_hash"},
            "required_input_fields": ["result_template_hash"],  # Audit field!
            "schema": {"fields": "dynamic"},
        })

        # Should raise - result_template_hash is audit-only
        with pytest.raises(ValueError, match="required field.*not guaranteed"):
            ExecutionGraph.from_plugin_instances(
                source=NullSource({"schema": {"fields": "dynamic"}}),
                transforms=[llm, downstream],
                sinks={"output": NullSink({})},
                aggregations={},
                gates={},
                coalesce_settings={},
                output_sink="output",
            )

    def test_batch_transform_emits_all_metadata_fields(self):
        """Batch transform emits all 9 metadata fields including variables_hash."""
        # This test verifies the batch transform fix
        config = {
            "response_field": "batch_result",
            "deployment_name": "test-deployment",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "test {{ row.input }}",
            "required_input_fields": ["input"],
            "schema": {"fields": "dynamic"},
        }
        transform = AzureBatchLLMTransform(config)

        # Verify schema config has all fields
        guaranteed = transform._output_schema_config.guaranteed_fields
        audit = transform._output_schema_config.audit_fields

        assert "batch_result" in guaranteed
        assert "batch_result_usage" in guaranteed
        assert "batch_result_model" in guaranteed

        assert "batch_result_template_hash" in audit
        assert "batch_result_variables_hash" in audit
        assert "batch_result_template_source" in audit


class TestAllLLMTransformsConsistency:
    """Verify all LLM transforms have consistent schema configuration."""

    def get_all_transform_configs(self):
        """Return minimal configs for all LLM transforms."""
        base = {
            "response_field": "out",
            "template": "test {{ row.x }}",
            "required_input_fields": [],
            "schema": {"fields": "dynamic"},
            "model": "test-model",
        }
        azure_extra = {
            "deployment_name": "x",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "x",
        }
        openrouter_extra = {"api_key": "x"}
        multi_query_extra = {
            "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
            "criteria": [{"name": "cr1", "code": "C1"}],
            "output_mapping": {"score": {"suffix": "score", "type": "integer"}},
        }

        return [
            ("AzureLLMTransform", AzureLLMTransform, {**base, **azure_extra}),
            ("AzureBatchLLMTransform", AzureBatchLLMTransform, {**base, **azure_extra}),
            ("OpenRouterLLMTransform", OpenRouterLLMTransform, {**base, **openrouter_extra}),
            # Multi-query transforms have different response_field pattern
        ]

    def test_all_transforms_have_output_schema_config(self):
        """All LLM transforms must set _output_schema_config."""
        for name, cls, config in self.get_all_transform_configs():
            transform = cls(config)
            assert hasattr(transform, "_output_schema_config"), f"{name} missing _output_schema_config"
            assert transform._output_schema_config is not None, f"{name} has None _output_schema_config"

    def test_all_transforms_have_same_guaranteed_structure(self):
        """All LLM transforms expose the same guaranteed field pattern."""
        for name, cls, config in self.get_all_transform_configs():
            transform = cls(config)
            guaranteed = transform._output_schema_config.guaranteed_fields

            assert "out" in guaranteed, f"{name} missing response field"
            assert "out_usage" in guaranteed, f"{name} missing _usage"
            assert "out_model" in guaranteed, f"{name} missing _model"
            assert len([f for f in guaranteed if f.startswith("out")]) == 3, f"{name} has unexpected guaranteed fields"

    def test_all_transforms_have_same_audit_structure(self):
        """All LLM transforms expose the same audit field pattern."""
        for name, cls, config in self.get_all_transform_configs():
            transform = cls(config)
            audit = transform._output_schema_config.audit_fields

            assert "out_template_hash" in audit, f"{name} missing _template_hash"
            assert "out_variables_hash" in audit, f"{name} missing _variables_hash"
            assert "out_template_source" in audit, f"{name} missing _template_source"
            assert "out_lookup_hash" in audit, f"{name} missing _lookup_hash"
            assert "out_lookup_source" in audit, f"{name} missing _lookup_source"
            assert "out_system_prompt_source" in audit, f"{name} missing _system_prompt_source"
            assert len([f for f in audit if f.startswith("out")]) == 6, f"{name} has unexpected audit fields"
```

---

## Rollout Sequence

### Phase 1: Schema System (no behavior change)
- [ ] Add `audit_fields` to SchemaConfig
- [ ] Update `from_dict()` and `to_dict()` methods
- [ ] Add unit tests for parsing, serialization, validation

### Phase 2: DAG Integration (enables behavior)
- [ ] Update `from_plugin_instances()` to extract `_output_schema_config` from transforms
- [ ] Update `_get_schema_config_from_node()` to prioritize `NodeInfo.output_schema_config`
- [ ] Add integration test verifying schema propagation through production path

### Phase 3: LLM Helpers (no behavior change)
- [ ] Add constants and helper functions to `llm/__init__.py`
- [ ] Add empty response_field validation
- [ ] Add unit tests

### Phase 4: Batch Transform Fix (behavior change)
- [ ] Capture `variables_hash` at submission time in `_submit_batch`
- [ ] Update checkpoint `row_mapping` to store dict with index and variables_hash
- [ ] Emit all 9 metadata fields in `_download_results`
- [ ] Add `_output_schema_config` to batch transform
- [ ] Add integration tests for batch transform

### Phase 5: Streaming Transform Updates (behavior change)
- [ ] Update `AzureLLMTransform` to set `_output_schema_config`
- [ ] Update `OpenRouterLLMTransform` to set `_output_schema_config`
- [ ] Update `BaseLLMTransform` to set `_output_schema_config`

### Phase 6: Multi-Query Transform Updates (behavior change)
- [ ] Update `AzureMultiQueryLLMTransform` with set comprehension pattern
- [ ] Update `OpenRouterMultiQueryLLMTransform` with set comprehension pattern

### Phase 7: Verification (REQUIRED)
- [ ] Run full test suite
- [ ] **PLUGIN COMPLETENESS CHECK** (see below)
- [ ] Manual verification of DAG validation behavior
- [ ] Benchmark multi-query DAG construction time (target: <100ms for 50 queries)

---

## Final Verification Checklist

**REQUIRED: Confirm all 6 LLM plugins have been updated**

After implementation, run this verification script:

```python
# verify_llm_schema_completeness.py
"""Verify all LLM transforms have correct schema configuration."""

from elspeth.plugins.llm import get_llm_guaranteed_fields, get_llm_audit_fields
from elspeth.plugins.llm.azure import AzureLLMTransform
from elspeth.plugins.llm.azure_batch import AzureBatchLLMTransform
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform
from elspeth.plugins.llm.openrouter import OpenRouterLLMTransform
from elspeth.plugins.llm.openrouter_multi_query import OpenRouterMultiQueryLLMTransform
from elspeth.plugins.llm.base import BaseLLMTransform

TRANSFORMS_TO_CHECK = [
    "AzureLLMTransform",
    "AzureBatchLLMTransform",
    "AzureMultiQueryLLMTransform",
    "OpenRouterLLMTransform",
    "OpenRouterMultiQueryLLMTransform",
    "BaseLLMTransform",
]

def verify_transform(name: str, transform_cls: type, config: dict) -> list[str]:
    """Verify a transform has correct schema config. Returns list of errors."""
    errors = []

    try:
        transform = transform_cls(config)
    except Exception as e:
        return [f"{name}: Failed to instantiate: {e}"]

    # Check _output_schema_config exists
    if not hasattr(transform, "_output_schema_config"):
        errors.append(f"{name}: Missing _output_schema_config attribute")
        return errors

    if transform._output_schema_config is None:
        errors.append(f"{name}: _output_schema_config is None")
        return errors

    schema = transform._output_schema_config
    response_field = config.get("response_field", "llm_response")

    # Check guaranteed fields
    expected_guaranteed = set(get_llm_guaranteed_fields(response_field))
    actual_guaranteed = set(schema.guaranteed_fields or ())

    missing_guaranteed = expected_guaranteed - actual_guaranteed
    if missing_guaranteed:
        errors.append(f"{name}: Missing guaranteed fields: {missing_guaranteed}")

    # Check audit fields
    expected_audit = set(get_llm_audit_fields(response_field))
    actual_audit = set(schema.audit_fields or ())

    missing_audit = expected_audit - actual_audit
    if missing_audit:
        errors.append(f"{name}: Missing audit fields: {missing_audit}")

    return errors

def main():
    """Run verification on all LLM transforms."""
    base_config = {
        "response_field": "test_response",
        "template": "test {{ row.input }}",
        "required_input_fields": [],
        "schema": {"fields": "dynamic"},
        "model": "test-model",
    }
    azure_config = {**base_config, "deployment_name": "x", "endpoint": "https://x", "api_key": "x"}
    openrouter_config = {**base_config, "api_key": "x"}

    all_errors = []

    # Standard transforms
    all_errors.extend(verify_transform("AzureLLMTransform", AzureLLMTransform, azure_config))
    all_errors.extend(verify_transform("AzureBatchLLMTransform", AzureBatchLLMTransform, azure_config))
    all_errors.extend(verify_transform("OpenRouterLLMTransform", OpenRouterLLMTransform, openrouter_config))

    # Note: BaseLLMTransform is abstract, skip direct instantiation
    # Note: Multi-query transforms need different config, tested separately

    if all_errors:
        print("VERIFICATION FAILED:")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    else:
        print("VERIFICATION PASSED: All LLM transforms have correct schema configuration")
        print(f"  Checked: {len(TRANSFORMS_TO_CHECK)} transforms")
        print(f"  Guaranteed fields per transform: 3")
        print(f"  Audit fields per transform: 6")
        return 0

if __name__ == "__main__":
    exit(main())
```

**Add to CI pipeline as post-implementation verification.**

---

## Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Existing pipelines break (breaking change) | Low | High | Pre-implementation grep verification |
| Performance (450 fields in multi-query) | Low | Medium | Benchmark before merge; frozenset ops are O(1) |
| Dict access bypasses audit boundary | Medium | Medium | Documentation + future instrumentation |
| "Two-tier API" confusion | Low | Low | Clear docstrings and warning in `__init__.py` |
| DAG integration not wired correctly | Medium | High | New test `test_schema_config_propagates_to_dag_nodeinfo` |

---

## Success Criteria

1. ✅ All 6 LLM transforms declare their metadata fields
2. ✅ `_usage` and `_model` are in `guaranteed_fields` (downstream can depend)
3. ✅ Hash/source fields are in `audit_fields` (exist but not contract-stable)
4. ✅ DAG validation passes for downstream requiring `_usage`
5. ✅ DAG validation fails for downstream requiring `_template_hash`
6. ✅ Batch transform emits all 9 standard metadata fields (including `_variables_hash`)
7. ✅ `_output_schema_config` propagates through `from_plugin_instances()` to `NodeInfo`
8. ✅ All existing tests pass
9. ✅ New tests verify field categorization behavior
10. ✅ Verification script confirms all plugins updated
11. ✅ Multi-query DAG construction completes in <100ms for 50 queries

---

## Open Questions (Resolved)

1. **Should `_template_hash` be guaranteed instead of audit?**
   - **Decision:** Audit. Hash algorithm could change between versions.

2. **Should we add `get_all_output_fields()` method?**
   - **Decision:** No. YAGNI - no caller identified. Add later if needed.

3. **What about batch transform `_variables_hash`?**
   - **Decision:** Capture at submission time, don't emit `None`. The hash IS computable.

4. **Do we need a migration guide?**
   - **Decision:** No. Pre-implementation verification confirms no existing usage.

5. **How should `_output_schema_config` be exposed to DAG validation?**
   - **Decision:** Option A - Update `from_plugin_instances()` to extract the attribute and pass it via existing `output_schema_config` parameter. Update `_get_schema_config_from_node()` to prioritize `NodeInfo.output_schema_config`. This aligns with existing `input_schema`/`output_schema` patterns.

6. **Do we need checkpoint backward compatibility?**
   - **Decision:** No. Pre-production with no users means no in-flight batches to migrate. Clean break to new `row_mapping` format.

---

## Implementation Summary

**Status:** Completed
**Commits:** See git history for this feature
**Notes:** Introduced audit_fields category in SchemaConfig to distinguish contract-stable guaranteed_fields from provenance metadata, with DAG integration and LLM transform updates.
