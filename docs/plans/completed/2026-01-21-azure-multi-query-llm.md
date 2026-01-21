# Azure Multi-Query LLM Transform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create an LLM transform that executes multiple queries per row (case studies × criteria), running in parallel with all-or-nothing error handling.

**Architecture:** Cross-product query expansion where each (case_study, criterion) pair generates an LLM call. All queries for a row run in parallel using the existing `PooledExecutor`. Results are merged into the output row with `{case_study}_{criterion}_{output_field}` naming. JSON response parsing maps LLM output fields to row columns.

**Tech Stack:** Pydantic (config), PooledExecutor (parallel execution), PromptTemplate (Jinja2 rendering), AuditedLLMClient (Azure OpenAI)

---

## Task 1: Create QuerySpec Dataclass

**Files:**
- Create: `src/elspeth/plugins/llm/multi_query.py`
- Test: `tests/plugins/llm/test_multi_query.py`

**Step 1: Write the failing test**

```python
# tests/plugins/llm/test_multi_query.py
"""Tests for multi-query LLM support."""

import pytest

from elspeth.plugins.llm.multi_query import QuerySpec


class TestQuerySpec:
    """Tests for QuerySpec dataclass."""

    def test_query_spec_creation(self) -> None:
        """QuerySpec holds case study and criterion info."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "cs1_sym", "cs1_hist"],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG", "subcriteria": ["accuracy"]},
        )
        assert spec.case_study_name == "cs1"
        assert spec.criterion_name == "diagnosis"
        assert spec.output_prefix == "cs1_diagnosis"

    def test_query_spec_build_template_context(self) -> None:
        """QuerySpec builds template context with positional inputs and criterion."""
        spec = QuerySpec(
            case_study_name="cs1",
            criterion_name="diagnosis",
            input_fields=["cs1_bg", "cs1_sym", "cs1_hist"],
            output_prefix="cs1_diagnosis",
            criterion_data={"code": "DIAG", "subcriteria": ["accuracy"]},
        )
        row = {
            "cs1_bg": "45yo male",
            "cs1_sym": "chest pain",
            "cs1_hist": "family history",
            "other_field": "ignored",
        }
        context = spec.build_template_context(row)

        assert context["input_1"] == "45yo male"
        assert context["input_2"] == "chest pain"
        assert context["input_3"] == "family history"
        assert context["criterion"]["code"] == "DIAG"
        assert context["row"] == row  # Full row for row-based lookups
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_multi_query.py::TestQuerySpec -v`
Expected: FAIL with "No module named 'elspeth.plugins.llm.multi_query'"

**Step 3: Write minimal implementation**

```python
# src/elspeth/plugins/llm/multi_query.py
"""Multi-query LLM support for case study × criteria cross-product evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QuerySpec:
    """Specification for a single query in the cross-product.

    Represents one (case_study, criterion) pair to be evaluated.

    Attributes:
        case_study_name: Name of the case study (e.g., "cs1")
        criterion_name: Name of the criterion (e.g., "diagnosis")
        input_fields: List of row field names to map to input_1, input_2, etc.
        output_prefix: Prefix for output fields (e.g., "cs1_diagnosis")
        criterion_data: Full criterion object for template injection
    """

    case_study_name: str
    criterion_name: str
    input_fields: list[str]
    output_prefix: str
    criterion_data: dict[str, Any]

    def build_template_context(self, row: dict[str, Any]) -> dict[str, Any]:
        """Build template context for this query.

        Maps input_fields to input_1, input_2, etc. and injects criterion data.

        Args:
            row: Full row data

        Returns:
            Context dict with input_N, criterion, and row
        """
        context: dict[str, Any] = {}

        # Map input fields to positional variables
        for i, field_name in enumerate(self.input_fields, start=1):
            context[f"input_{i}"] = row.get(field_name, "")

        # Inject criterion data
        context["criterion"] = self.criterion_data

        # Include full row for row-based lookups
        context["row"] = row

        return context
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_multi_query.py::TestQuerySpec -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/multi_query.py tests/plugins/llm/test_multi_query.py
git commit -m "feat(llm): add QuerySpec dataclass for multi-query support"
```

---

## Task 2: Create MultiQueryConfig Pydantic Model

**Files:**
- Modify: `src/elspeth/plugins/llm/multi_query.py`
- Modify: `tests/plugins/llm/test_multi_query.py`

**Step 1: Write the failing tests**

```python
# Add to tests/plugins/llm/test_multi_query.py

from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.llm.multi_query import (
    CaseStudyConfig,
    CriterionConfig,
    MultiQueryConfig,
    QuerySpec,
)


class TestCaseStudyConfig:
    """Tests for CaseStudyConfig validation."""

    def test_case_study_requires_name(self) -> None:
        """CaseStudyConfig requires name."""
        with pytest.raises(PluginConfigError):
            CaseStudyConfig.from_dict({"input_fields": ["a", "b"]})

    def test_case_study_requires_input_fields(self) -> None:
        """CaseStudyConfig requires input_fields."""
        with pytest.raises(PluginConfigError):
            CaseStudyConfig.from_dict({"name": "cs1"})

    def test_valid_case_study(self) -> None:
        """Valid CaseStudyConfig passes validation."""
        config = CaseStudyConfig.from_dict({
            "name": "cs1",
            "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"],
        })
        assert config.name == "cs1"
        assert config.input_fields == ["cs1_bg", "cs1_sym", "cs1_hist"]


class TestCriterionConfig:
    """Tests for CriterionConfig validation."""

    def test_criterion_requires_name(self) -> None:
        """CriterionConfig requires name."""
        with pytest.raises(PluginConfigError):
            CriterionConfig.from_dict({"code": "DIAG"})

    def test_valid_criterion_minimal(self) -> None:
        """CriterionConfig works with just name."""
        config = CriterionConfig.from_dict({"name": "diagnosis"})
        assert config.name == "diagnosis"
        assert config.code is None
        assert config.subcriteria == []

    def test_valid_criterion_full(self) -> None:
        """CriterionConfig accepts all fields."""
        config = CriterionConfig.from_dict({
            "name": "diagnosis",
            "code": "DIAG",
            "description": "Assess diagnostic accuracy",
            "subcriteria": ["accuracy", "evidence"],
        })
        assert config.name == "diagnosis"
        assert config.code == "DIAG"
        assert config.description == "Assess diagnostic accuracy"
        assert config.subcriteria == ["accuracy", "evidence"]


class TestMultiQueryConfig:
    """Tests for MultiQueryConfig validation."""

    def test_config_requires_case_studies(self) -> None:
        """MultiQueryConfig requires case_studies."""
        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict({
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ input_1 }}",
                "criteria": [{"name": "diagnosis"}],
                "response_format": "json",
                "output_mapping": {"score": "score"},
                "schema": {"fields": "dynamic"},
            })

    def test_config_requires_criteria(self) -> None:
        """MultiQueryConfig requires criteria."""
        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict({
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "response_format": "json",
                "output_mapping": {"score": "score"},
                "schema": {"fields": "dynamic"},
            })

    def test_config_requires_output_mapping(self) -> None:
        """MultiQueryConfig requires output_mapping."""
        with pytest.raises(PluginConfigError):
            MultiQueryConfig.from_dict({
                "deployment_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com",
                "api_key": "key",
                "template": "{{ input_1 }}",
                "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
                "criteria": [{"name": "diagnosis"}],
                "response_format": "json",
                "schema": {"fields": "dynamic"},
            })

    def test_valid_config(self) -> None:
        """Valid MultiQueryConfig passes validation."""
        config = MultiQueryConfig.from_dict({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "key",
            "template": "{{ input_1 }} {{ criterion.name }}",
            "system_prompt": "You are an AI.",
            "case_studies": [
                {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym"]},
                {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym"]},
            ],
            "criteria": [
                {"name": "diagnosis", "code": "DIAG"},
                {"name": "treatment", "code": "TREAT"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score", "rationale": "rationale"},
            "schema": {"fields": "dynamic"},
        })
        assert len(config.case_studies) == 2
        assert len(config.criteria) == 2
        assert config.output_mapping == {"score": "score", "rationale": "rationale"}

    def test_expand_queries_creates_cross_product(self) -> None:
        """expand_queries creates QuerySpec for each (case_study, criterion) pair."""
        config = MultiQueryConfig.from_dict({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "key",
            "template": "{{ input_1 }}",
            "case_studies": [
                {"name": "cs1", "input_fields": ["cs1_a"]},
                {"name": "cs2", "input_fields": ["cs2_a"]},
            ],
            "criteria": [
                {"name": "diagnosis", "code": "DIAG"},
                {"name": "treatment", "code": "TREAT"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        specs = config.expand_queries()

        # 2 case studies × 2 criteria = 4 queries
        assert len(specs) == 4

        # Check cross-product
        prefixes = [s.output_prefix for s in specs]
        assert "cs1_diagnosis" in prefixes
        assert "cs1_treatment" in prefixes
        assert "cs2_diagnosis" in prefixes
        assert "cs2_treatment" in prefixes
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/llm/test_multi_query.py -v -k "Config"`
Expected: FAIL with ImportError for new classes

**Step 3: Write implementation**

```python
# Add to src/elspeth/plugins/llm/multi_query.py after QuerySpec

from pydantic import Field, field_validator

from elspeth.plugins.config_base import ConfigSchema
from elspeth.plugins.llm.azure import AzureOpenAIConfig


class CaseStudyConfig(ConfigSchema):
    """Configuration for a single case study.

    Attributes:
        name: Unique identifier for this case study (used in output field prefix)
        input_fields: Row field names to map to input_1, input_2, etc.
    """

    name: str = Field(..., description="Case study identifier")
    input_fields: list[str] = Field(..., description="Row fields to map to input_N")

    @field_validator("input_fields")
    @classmethod
    def validate_input_fields_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one input field."""
        if not v:
            raise ValueError("input_fields cannot be empty")
        return v


class CriterionConfig(ConfigSchema):
    """Configuration for a single evaluation criterion.

    All fields except 'name' are optional and available in templates
    via {{ criterion.field_name }}.

    Attributes:
        name: Unique identifier (used in output field prefix)
        code: Short code for lookups (e.g., "DIAG")
        description: Human-readable description
        subcriteria: List of subcriteria names/descriptions
        extra: Any additional fields for template use
    """

    name: str = Field(..., description="Criterion identifier")
    code: str | None = Field(None, description="Short code for lookups")
    description: str | None = Field(None, description="Human-readable description")
    subcriteria: list[str] = Field(default_factory=list, description="Subcriteria list")

    def to_template_data(self) -> dict[str, Any]:
        """Convert to dict for template injection."""
        return {
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "subcriteria": self.subcriteria,
        }


class MultiQueryConfig(AzureOpenAIConfig):
    """Configuration for multi-query LLM transform.

    Extends AzureOpenAIConfig with:
    - case_studies: List of case study definitions
    - criteria: List of criterion definitions
    - output_mapping: JSON field → row column suffix mapping
    - response_format: Expected LLM output format (json)

    The cross-product of case_studies × criteria defines all queries.
    """

    case_studies: list[CaseStudyConfig] = Field(
        ...,
        description="Case study definitions",
        min_length=1,
    )
    criteria: list[CriterionConfig] = Field(
        ...,
        description="Criterion definitions",
        min_length=1,
    )
    output_mapping: dict[str, str] = Field(
        ...,
        description="JSON field → row column suffix mapping",
    )
    response_format: str = Field(
        "json",
        description="Expected response format",
    )

    @field_validator("output_mapping")
    @classmethod
    def validate_output_mapping_not_empty(cls, v: dict[str, str]) -> dict[str, str]:
        """Ensure at least one output mapping."""
        if not v:
            raise ValueError("output_mapping cannot be empty")
        return v

    def expand_queries(self) -> list[QuerySpec]:
        """Expand config into QuerySpec list (case_studies × criteria).

        Returns:
            List of QuerySpec, one per (case_study, criterion) pair
        """
        specs: list[QuerySpec] = []

        for case_study in self.case_studies:
            for criterion in self.criteria:
                spec = QuerySpec(
                    case_study_name=case_study.name,
                    criterion_name=criterion.name,
                    input_fields=case_study.input_fields,
                    output_prefix=f"{case_study.name}_{criterion.name}",
                    criterion_data=criterion.to_template_data(),
                )
                specs.append(spec)

        return specs
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/llm/test_multi_query.py -v -k "Config"`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/multi_query.py tests/plugins/llm/test_multi_query.py
git commit -m "feat(llm): add MultiQueryConfig with case_studies and criteria"
```

---

## Task 3: Create AzureMultiQueryLLMTransform - Basic Structure

**Files:**
- Create: `src/elspeth/plugins/llm/azure_multi_query.py`
- Create: `tests/plugins/llm/test_azure_multi_query.py`

**Step 1: Write the failing test**

```python
# tests/plugins/llm/test_azure_multi_query.py
"""Tests for Azure Multi-Query LLM transform."""

from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts import Determinism
from elspeth.plugins.config_base import PluginConfigError
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform


# Common schema config
DYNAMIC_SCHEMA = {"fields": "dynamic"}


def make_config(**overrides: Any) -> dict[str, Any]:
    """Create valid config with optional overrides."""
    config = {
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "template": "Input: {{ input_1 }}\nCriterion: {{ criterion.name }}",
        "system_prompt": "You are an assessment AI. Respond in JSON.",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_bg", "cs1_sym", "cs1_hist"]},
            {"name": "cs2", "input_fields": ["cs2_bg", "cs2_sym", "cs2_hist"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG"},
            {"name": "treatment", "code": "TREAT"},
        ],
        "response_format": "json",
        "output_mapping": {"score": "score", "rationale": "rationale"},
        "schema": DYNAMIC_SCHEMA,
        "pool_size": 4,
    }
    config.update(overrides)
    return config


class TestAzureMultiQueryLLMTransformInit:
    """Tests for transform initialization."""

    def test_transform_has_correct_name(self) -> None:
        """Transform registers with correct plugin name."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.name == "azure_multi_query_llm"

    def test_transform_is_non_deterministic(self) -> None:
        """LLM transforms are non-deterministic."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.determinism == Determinism.NON_DETERMINISTIC

    def test_transform_is_batch_aware(self) -> None:
        """Transform supports batch aggregation."""
        transform = AzureMultiQueryLLMTransform(make_config())
        assert transform.is_batch_aware is True

    def test_transform_expands_queries_on_init(self) -> None:
        """Transform pre-computes query specs on initialization."""
        transform = AzureMultiQueryLLMTransform(make_config())
        # 2 case studies × 2 criteria = 4 queries
        assert len(transform._query_specs) == 4

    def test_transform_requires_case_studies(self) -> None:
        """Transform requires case_studies in config."""
        config = make_config()
        del config["case_studies"]
        with pytest.raises(PluginConfigError):
            AzureMultiQueryLLMTransform(config)

    def test_transform_requires_criteria(self) -> None:
        """Transform requires criteria in config."""
        config = make_config()
        del config["criteria"]
        with pytest.raises(PluginConfigError):
            AzureMultiQueryLLMTransform(config)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestAzureMultiQueryLLMTransformInit -v`
Expected: FAIL with "No module named 'elspeth.plugins.llm.azure_multi_query'"

**Step 3: Write implementation**

```python
# src/elspeth/plugins/llm/azure_multi_query.py
"""Azure Multi-Query LLM transform for case study × criteria evaluation.

Executes multiple LLM queries per row in parallel, merging all results
into a single output row with all-or-nothing error handling.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import AuditedLLMClient
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.multi_query import MultiQueryConfig, QuerySpec
from elspeth.plugins.llm.pooled_executor import PooledExecutor, RowContext
from elspeth.plugins.llm.templates import PromptTemplate
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from elspeth.core.landscape.recorder import LandscapeRecorder


class AzureMultiQueryLLMTransform(BaseTransform):
    """LLM transform that executes case_studies × criteria queries per row.

    For each row, expands the cross-product of case studies and criteria
    into individual LLM queries. All queries run in parallel (up to pool_size),
    with all-or-nothing error semantics (if any query fails, the row fails).

    Configuration example:
        transforms:
          - plugin: azure_multi_query_llm
            options:
              deployment_name: "gpt-4o"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Case: {{ input_1 }}, {{ input_2 }}
                Criterion: {{ criterion.name }}
              case_studies:
                - name: cs1
                  input_fields: [cs1_bg, cs1_sym]
                - name: cs2
                  input_fields: [cs2_bg, cs2_sym]
              criteria:
                - name: diagnosis
                  code: DIAG
                - name: treatment
                  code: TREAT
              response_format: json
              output_mapping:
                score: score
                rationale: rationale
              pool_size: 4
              schema:
                fields: dynamic

    Output fields per query:
        {case_study}_{criterion}_{json_field} for each output_mapping entry
        Plus metadata: _usage, _template_hash, _model
    """

    name = "azure_multi_query_llm"
    is_batch_aware = True
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize transform with multi-query configuration."""
        super().__init__(config)

        # Parse config
        cfg = MultiQueryConfig.from_dict(config)

        # Store Azure connection settings
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name
        self._model = cfg.model or cfg.deployment_name

        # Store template settings
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._on_error = cfg.on_error

        # Multi-query specific settings
        self._output_mapping = cfg.output_mapping
        self._response_format = cfg.response_format

        # Pre-expand query specs (case_studies × criteria)
        self._query_specs = cfg.expand_queries()

        # Schema from config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Pooled execution setup
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # Client caching (same pattern as AzureLLMTransform)
        self._recorder: LandscapeRecorder | None = None
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        self._underlying_client: AzureOpenAI | None = None

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution."""
        self._recorder = ctx.landscape

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process row(s) with all queries in parallel.

        For single row: executes all (case_study × criterion) queries,
        merges results into one output row.

        For batch: processes each row independently (batch of multi-query rows).
        """
        # Batch dispatch
        if isinstance(row, list):
            return self._process_batch(row, ctx)

        # Single row processing
        return self._process_single_row(row, ctx)

    def _process_single_row(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row with all queries."""
        # Placeholder - will be implemented in Task 4
        raise NotImplementedError("_process_single_row not yet implemented")

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows."""
        # Placeholder - will be implemented in Task 6
        raise NotImplementedError("_process_batch not yet implemented")

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._recorder = None
        with self._llm_clients_lock:
            self._llm_clients.clear()
        self._underlying_client = None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestAzureMultiQueryLLMTransformInit -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py tests/plugins/llm/test_azure_multi_query.py
git commit -m "feat(llm): add AzureMultiQueryLLMTransform skeleton"
```

---

## Task 4: Implement Single Query Processing

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_multi_query.py`
- Modify: `tests/plugins/llm/test_azure_multi_query.py`

**Step 1: Write the failing test**

```python
# Add to tests/plugins/llm/test_azure_multi_query.py

import json
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import Mock, patch


@contextmanager
def mock_azure_openai_responses(
    responses: list[dict[str, Any]],
) -> Generator[Mock, None, None]:
    """Mock Azure OpenAI to return sequence of JSON responses.

    Args:
        responses: List of dicts to return as JSON content
    """
    call_count = 0

    def make_response() -> Mock:
        nonlocal call_count
        content = json.dumps(responses[call_count % len(responses)])
        call_count += 1

        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5

        mock_message = Mock()
        mock_message.content = content

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = mock_usage
        mock_response.model_dump = Mock(return_value={"model": "gpt-4o"})

        return mock_response

    with patch("openai.AzureOpenAI") as mock_azure_class:
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = lambda **kwargs: make_response()
        mock_azure_class.return_value = mock_client
        yield mock_client


def make_plugin_context(state_id: str = "state-123") -> PluginContext:
    """Create a PluginContext with mocked landscape."""
    mock_landscape = Mock()
    mock_landscape.record_external_call = Mock()
    return PluginContext(
        run_id="run-123",
        landscape=mock_landscape,
        state_id=state_id,
    )


class TestSingleQueryProcessing:
    """Tests for _process_single_query method."""

    def test_process_single_query_renders_template(self) -> None:
        """Single query renders template with input fields and criterion."""
        responses = [{"score": 85, "rationale": "Good diagnosis"}]

        with mock_azure_openai_responses(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "45yo male",
                "cs1_sym": "chest pain",
                "cs1_hist": "family history",
            }
            spec = transform._query_specs[0]  # cs1_diagnosis

            result = transform._process_single_query(row, spec, ctx.state_id)

            # Check template was rendered with correct content
            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            user_message = messages[-1]["content"]

            assert "45yo male" in user_message
            assert "diagnosis" in user_message.lower()

    def test_process_single_query_parses_json_response(self) -> None:
        """Single query parses JSON and returns mapped fields."""
        responses = [{"score": 85, "rationale": "Excellent assessment"}]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]  # cs1_diagnosis

            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "success"
            assert result.row is not None
            # Output fields use prefix from spec
            assert result.row["cs1_diagnosis_score"] == 85
            assert result.row["cs1_diagnosis_rationale"] == "Excellent assessment"

    def test_process_single_query_handles_invalid_json(self) -> None:
        """Single query returns error on invalid JSON response."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content="not json"))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            mock_client.chat.completions.create.return_value = mock_response
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_config())
            ctx = make_plugin_context()

            row = {"cs1_bg": "data", "cs1_sym": "data", "cs1_hist": "data"}
            spec = transform._query_specs[0]

            result = transform._process_single_query(row, spec, ctx.state_id)

            assert result.status == "error"
            assert "json" in result.reason["reason"].lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestSingleQueryProcessing -v`
Expected: FAIL with AttributeError or NotImplementedError

**Step 3: Write implementation**

```python
# Add to AzureMultiQueryLLMTransform class in azure_multi_query.py

import json

from elspeth.plugins.clients.llm import LLMClientError, RateLimitError
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.templates import TemplateError


# Add these methods to the class:

def _get_underlying_client(self) -> AzureOpenAI:
    """Get or create the underlying Azure OpenAI client."""
    if self._underlying_client is None:
        from openai import AzureOpenAI

        self._underlying_client = AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_api_key,
            api_version=self._azure_api_version,
        )
    return self._underlying_client

def _get_llm_client(self, state_id: str) -> AuditedLLMClient:
    """Get or create LLM client for a state_id."""
    with self._llm_clients_lock:
        if state_id not in self._llm_clients:
            if self._recorder is None:
                raise RuntimeError("Transform requires recorder. Ensure on_start was called.")
            self._llm_clients[state_id] = AuditedLLMClient(
                recorder=self._recorder,
                state_id=state_id,
                underlying_client=self._get_underlying_client(),
                provider="azure",
            )
        return self._llm_clients[state_id]

def _process_single_query(
    self,
    row: dict[str, Any],
    spec: QuerySpec,
    state_id: str,
) -> TransformResult:
    """Process a single query (one case_study × criterion pair).

    Args:
        row: Full input row
        spec: Query specification with input field mapping
        state_id: State ID for audit trail

    Returns:
        TransformResult with mapped output fields

    Raises:
        CapacityError: On rate limit (for pooled retry)
    """
    # 1. Build template context with positional inputs and criterion
    template_context = spec.build_template_context(row)

    # 2. Create a temporary row dict for template rendering
    # The template expects {{ input_1 }}, {{ criterion }}, {{ row }}, {{ lookup }}
    render_row = {
        **{f"input_{i+1}": v for i, (k, v) in enumerate(template_context.items())
           if k.startswith("input_")},
    }
    # Add criterion and row to the context
    render_row["criterion"] = template_context["criterion"]
    render_row["row"] = template_context["row"]

    # 3. Render template (THEIR DATA - wrap)
    try:
        # Create a custom template that can access our context
        from jinja2 import StrictUndefined
        from jinja2.sandbox import SandboxedEnvironment

        env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
        template = env.from_string(self._template._template_string)

        # Render with full context
        rendered_prompt = template.render(
            input_1=template_context.get("input_1", ""),
            input_2=template_context.get("input_2", ""),
            input_3=template_context.get("input_3", ""),
            input_4=template_context.get("input_4", ""),
            input_5=template_context.get("input_5", ""),
            input_6=template_context.get("input_6", ""),
            criterion=template_context["criterion"],
            row=template_context["row"],
            lookup=self._template._lookup_data,
        )
    except Exception as e:
        return TransformResult.error({
            "reason": "template_rendering_failed",
            "error": str(e),
            "query": spec.output_prefix,
        })

    # 4. Build messages
    messages: list[dict[str, str]] = []
    if self._system_prompt:
        messages.append({"role": "system", "content": self._system_prompt})
    messages.append({"role": "user", "content": rendered_prompt})

    # 5. Get LLM client
    llm_client = self._get_llm_client(state_id)

    # 6. Call LLM (EXTERNAL - wrap, raise CapacityError for retry)
    try:
        response = llm_client.chat_completion(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
    except RateLimitError as e:
        raise CapacityError(429, str(e)) from e
    except LLMClientError as e:
        return TransformResult.error({
            "reason": "llm_call_failed",
            "error": str(e),
            "query": spec.output_prefix,
        })

    # 7. Parse JSON response (THEIR DATA - wrap)
    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError as e:
        return TransformResult.error({
            "reason": "json_parse_failed",
            "error": str(e),
            "query": spec.output_prefix,
            "raw_response": response.content[:500],  # Truncate for audit
        })

    # 8. Map output fields
    output: dict[str, Any] = {}
    for json_field, suffix in self._output_mapping.items():
        output_key = f"{spec.output_prefix}_{suffix}"
        if json_field not in parsed:
            return TransformResult.error({
                "reason": "missing_output_field",
                "field": json_field,
                "query": spec.output_prefix,
            })
        output[output_key] = parsed[json_field]

    # 9. Add metadata
    output[f"{spec.output_prefix}_usage"] = response.usage
    output[f"{spec.output_prefix}_model"] = response.model

    return TransformResult.success(output)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestSingleQueryProcessing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py tests/plugins/llm/test_azure_multi_query.py
git commit -m "feat(llm): implement single query processing with JSON parsing"
```

---

## Task 5: Implement Row Processing (All Queries in Parallel)

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_multi_query.py`
- Modify: `tests/plugins/llm/test_azure_multi_query.py`

**Step 1: Write the failing test**

```python
# Add to tests/plugins/llm/test_azure_multi_query.py

class TestRowProcessing:
    """Tests for full row processing (all queries)."""

    def test_process_row_executes_all_queries(self) -> None:
        """Process executes all (case_study × criterion) queries."""
        # 2 case studies × 2 criteria = 4 queries
        responses = [
            {"score": 85, "rationale": "CS1 diagnosis"},
            {"score": 90, "rationale": "CS1 treatment"},
            {"score": 75, "rationale": "CS2 diagnosis"},
            {"score": 80, "rationale": "CS2 treatment"},
        ]

        with mock_azure_openai_responses(responses) as mock_client:
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "case1 bg", "cs1_sym": "case1 sym", "cs1_hist": "case1 hist",
                "cs2_bg": "case2 bg", "cs2_sym": "case2 sym", "cs2_hist": "case2 hist",
            }

            result = transform.process(row, ctx)

            assert result.status == "success"
            assert mock_client.chat.completions.create.call_count == 4

    def test_process_row_merges_all_results(self) -> None:
        """All query results are merged into single output row."""
        responses = [
            {"score": 85, "rationale": "R1"},
            {"score": 90, "rationale": "R2"},
            {"score": 75, "rationale": "R3"},
            {"score": 80, "rationale": "R4"},
        ]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "bg1", "cs1_sym": "sym1", "cs1_hist": "hist1",
                "cs2_bg": "bg2", "cs2_sym": "sym2", "cs2_hist": "hist2",
                "original_field": "preserved",
            }

            result = transform.process(row, ctx)

            assert result.status == "success"
            output = result.row

            # Original fields preserved
            assert output["original_field"] == "preserved"

            # All 4 queries produced output (2 fields each = 8 assessment fields)
            assert "cs1_diagnosis_score" in output
            assert "cs1_diagnosis_rationale" in output
            assert "cs1_treatment_score" in output
            assert "cs2_diagnosis_score" in output
            assert "cs2_treatment_score" in output

    def test_process_row_fails_if_any_query_fails(self) -> None:
        """All-or-nothing: if any query fails, entire row fails."""
        # First 3 succeed, 4th returns invalid JSON
        call_count = [0]

        def make_response(**kwargs: Any) -> Mock:
            call_count[0] += 1
            if call_count[0] == 4:
                content = "not valid json"
            else:
                content = json.dumps({"score": 85, "rationale": "ok"})

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = make_response
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            row = {
                "cs1_bg": "bg", "cs1_sym": "sym", "cs1_hist": "hist",
                "cs2_bg": "bg", "cs2_sym": "sym", "cs2_hist": "hist",
            }

            result = transform.process(row, ctx)

            # Entire row fails
            assert result.status == "error"
            assert "query_failed" in result.reason["reason"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestRowProcessing -v`
Expected: FAIL with NotImplementedError

**Step 3: Write implementation**

```python
# Replace _process_single_row in azure_multi_query.py

def _process_single_row(
    self,
    row: dict[str, Any],
    ctx: PluginContext,
) -> TransformResult:
    """Process a single row with all queries in parallel.

    Executes all (case_study × criterion) queries for this row.
    All-or-nothing: if any query fails, the entire row fails.

    Args:
        row: Input row with all case study fields
        ctx: Plugin context with landscape and state_id

    Returns:
        TransformResult with all query results merged, or error
    """
    if ctx.landscape is None or ctx.state_id is None:
        raise RuntimeError(
            "Multi-query transform requires landscape recorder and state_id."
        )

    # Capture recorder for pooled execution
    if self._recorder is None:
        self._recorder = ctx.landscape

    # Build row contexts for each query (all share same state_id)
    query_contexts = [
        (spec, RowContext(row=row, state_id=ctx.state_id, row_index=i))
        for i, spec in enumerate(self._query_specs)
    ]

    # Execute all queries
    if self._executor is not None:
        # Parallel execution via PooledExecutor
        results = self._execute_queries_parallel(query_contexts, ctx.state_id)
    else:
        # Sequential fallback
        results = self._execute_queries_sequential(query_contexts, ctx.state_id)

    # Clean up cached clients
    with self._llm_clients_lock:
        self._llm_clients.pop(ctx.state_id, None)

    # Check for failures (all-or-nothing)
    failed = [(spec, r) for (spec, _), r in zip(query_contexts, results) if r.status != "success"]
    if failed:
        return TransformResult.error({
            "reason": "query_failed",
            "failed_queries": [
                {
                    "query": spec.output_prefix,
                    "error": r.reason,
                }
                for spec, r in failed
            ],
            "succeeded_count": len(results) - len(failed),
            "total_count": len(results),
        })

    # Merge all results into output row
    output = dict(row)
    for (spec, _), result in zip(query_contexts, results):
        if result.row is not None:
            output.update(result.row)

    return TransformResult.success(output)

def _execute_queries_parallel(
    self,
    query_contexts: list[tuple[QuerySpec, RowContext]],
    state_id: str,
) -> list[TransformResult]:
    """Execute queries in parallel via PooledExecutor."""
    # Create wrapper that captures the spec
    def process_with_spec(row: dict[str, Any], sid: str, spec: QuerySpec = None) -> TransformResult:
        return self._process_single_query(row, spec, sid)

    # Submit all queries to the executor
    # We need to pass spec info through, so we'll use a different approach
    results: list[TransformResult] = []

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=self._executor._pool_size) as executor:
        futures = {
            executor.submit(
                self._process_single_query,
                row_ctx.row,
                spec,
                state_id,
            ): i
            for i, (spec, row_ctx) in enumerate(query_contexts)
        }

        # Collect results in submission order
        results = [None] * len(query_contexts)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except CapacityError:
                # If capacity error escapes (shouldn't with proper retry), treat as error
                results[idx] = TransformResult.error({
                    "reason": "capacity_exhausted",
                    "query": query_contexts[idx][0].output_prefix,
                })

    return results

def _execute_queries_sequential(
    self,
    query_contexts: list[tuple[QuerySpec, RowContext]],
    state_id: str,
) -> list[TransformResult]:
    """Execute queries sequentially (fallback when no executor)."""
    results: list[TransformResult] = []

    for spec, row_ctx in query_contexts:
        try:
            result = self._process_single_query(row_ctx.row, spec, state_id)
        except CapacityError as e:
            result = TransformResult.error({
                "reason": "rate_limited",
                "error": str(e),
                "query": spec.output_prefix,
            })
        results.append(result)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestRowProcessing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py tests/plugins/llm/test_azure_multi_query.py
git commit -m "feat(llm): implement parallel row processing with all-or-nothing semantics"
```

---

## Task 6: Implement Batch Processing

**Files:**
- Modify: `src/elspeth/plugins/llm/azure_multi_query.py`
- Modify: `tests/plugins/llm/test_azure_multi_query.py`

**Step 1: Write the failing test**

```python
# Add to tests/plugins/llm/test_azure_multi_query.py

class TestBatchProcessing:
    """Tests for batch processing (aggregation mode)."""

    def test_process_batch_handles_list_input(self) -> None:
        """Process accepts list of rows for batch aggregation."""
        # 2 rows × 4 queries each = 8 total LLM calls
        responses = [{"score": i, "rationale": f"R{i}"} for i in range(8)]

        with mock_azure_openai_responses(responses):
            transform = AzureMultiQueryLLMTransform(make_config())
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            rows = [
                {"cs1_bg": "r1", "cs1_sym": "r1", "cs1_hist": "r1",
                 "cs2_bg": "r1", "cs2_sym": "r1", "cs2_hist": "r1"},
                {"cs1_bg": "r2", "cs1_sym": "r2", "cs1_hist": "r2",
                 "cs2_bg": "r2", "cs2_sym": "r2", "cs2_hist": "r2"},
            ]

            result = transform.process(rows, ctx)

            assert result.status == "success"
            assert result.is_multi_row
            assert len(result.rows) == 2

    def test_process_batch_preserves_row_independence(self) -> None:
        """Each row in batch is processed independently."""
        # First row succeeds, second row's 4th query fails
        responses = [
            # Row 1: all 4 succeed
            {"score": 1, "rationale": "R1-1"},
            {"score": 2, "rationale": "R1-2"},
            {"score": 3, "rationale": "R1-3"},
            {"score": 4, "rationale": "R1-4"},
            # Row 2: first 3 succeed, 4th fails
            {"score": 5, "rationale": "R2-1"},
            {"score": 6, "rationale": "R2-2"},
            {"score": 7, "rationale": "R2-3"},
        ]
        call_count = [0]

        def make_response(**kwargs: Any) -> Mock:
            call_count[0] += 1
            if call_count[0] == 8:  # Row 2, query 4
                content = "invalid json"
            else:
                idx = min(call_count[0] - 1, len(responses) - 1)
                content = json.dumps(responses[idx])

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
            mock_response.model_dump = Mock(return_value={})
            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = make_response
            mock_azure_class.return_value = mock_client

            # Use pool_size=1 to ensure sequential processing for predictable order
            config = make_config(pool_size=1)
            transform = AzureMultiQueryLLMTransform(config)
            transform.on_start(make_plugin_context())
            ctx = make_plugin_context()

            rows = [
                {"cs1_bg": "r1", "cs1_sym": "r1", "cs1_hist": "r1",
                 "cs2_bg": "r1", "cs2_sym": "r1", "cs2_hist": "r1"},
                {"cs1_bg": "r2", "cs1_sym": "r2", "cs1_hist": "r2",
                 "cs2_bg": "r2", "cs2_sym": "r2", "cs2_hist": "r2"},
            ]

            result = transform.process(rows, ctx)

            # Batch returns success_multi with per-row results
            assert result.status == "success"
            assert result.is_multi_row
            assert len(result.rows) == 2

            # Row 1 succeeded - has all output fields
            assert "cs1_diagnosis_score" in result.rows[0]

            # Row 2 failed - has error marker
            assert "cs1_diagnosis_score" not in result.rows[1] or result.rows[1].get("_error") is not None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestBatchProcessing -v`
Expected: FAIL with NotImplementedError

**Step 3: Write implementation**

```python
# Replace _process_batch in azure_multi_query.py

def _process_batch(
    self,
    rows: list[dict[str, Any]],
    ctx: PluginContext,
) -> TransformResult:
    """Process batch of rows (aggregation mode).

    Each row is processed independently with all-or-nothing semantics.
    Batch as a whole uses partial success (failed rows get error markers).

    Args:
        rows: List of input rows
        ctx: Plugin context

    Returns:
        TransformResult.success_multi with per-row results
    """
    if not rows:
        return TransformResult.success({"batch_empty": True, "row_count": 0})

    if ctx.landscape is None or ctx.state_id is None:
        raise RuntimeError(
            "Batch processing requires landscape recorder and state_id."
        )

    # Capture recorder
    if self._recorder is None:
        self._recorder = ctx.landscape

    # Process each row independently
    output_rows: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        # Create per-row state_id for audit trail uniqueness
        row_state_id = f"{ctx.state_id}_row{i}"

        try:
            result = self._process_single_row_internal(row, row_state_id)

            if result.status == "success" and result.row is not None:
                output_rows.append(result.row)
            else:
                # Row failed - include original with error marker
                error_row = dict(row)
                error_row["_error"] = result.reason
                output_rows.append(error_row)
        finally:
            # Clean up per-row client cache
            with self._llm_clients_lock:
                self._llm_clients.pop(row_state_id, None)

    return TransformResult.success_multi(output_rows)

def _process_single_row_internal(
    self,
    row: dict[str, Any],
    state_id: str,
) -> TransformResult:
    """Internal row processing with explicit state_id.

    Used by both single-row and batch processing paths.
    """
    # Build row contexts for each query
    query_contexts = [
        (spec, RowContext(row=row, state_id=state_id, row_index=i))
        for i, spec in enumerate(self._query_specs)
    ]

    # Execute all queries
    if self._executor is not None:
        results = self._execute_queries_parallel(query_contexts, state_id)
    else:
        results = self._execute_queries_sequential(query_contexts, state_id)

    # Check for failures (all-or-nothing for this row)
    failed = [(spec, r) for (spec, _), r in zip(query_contexts, results) if r.status != "success"]
    if failed:
        return TransformResult.error({
            "reason": "query_failed",
            "failed_queries": [
                {"query": spec.output_prefix, "error": r.reason}
                for spec, r in failed
            ],
            "succeeded_count": len(results) - len(failed),
            "total_count": len(results),
        })

    # Merge all results into output row
    output = dict(row)
    for (spec, _), result in zip(query_contexts, results):
        if result.row is not None:
            output.update(result.row)

    return TransformResult.success(output)
```

Also update `_process_single_row` to use the internal method:

```python
def _process_single_row(
    self,
    row: dict[str, Any],
    ctx: PluginContext,
) -> TransformResult:
    """Process a single row with all queries in parallel."""
    if ctx.landscape is None or ctx.state_id is None:
        raise RuntimeError(
            "Multi-query transform requires landscape recorder and state_id."
        )

    if self._recorder is None:
        self._recorder = ctx.landscape

    try:
        return self._process_single_row_internal(row, ctx.state_id)
    finally:
        with self._llm_clients_lock:
            self._llm_clients.pop(ctx.state_id, None)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/llm/test_azure_multi_query.py::TestBatchProcessing -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/azure_multi_query.py tests/plugins/llm/test_azure_multi_query.py
git commit -m "feat(llm): implement batch processing with per-row error handling"
```

---

## Task 7: Register Plugin and Add Integration Test

**Files:**
- Modify: `src/elspeth/plugins/llm/__init__.py`
- Create: `tests/integration/test_multi_query_integration.py`

**Step 1: Update plugin exports**

```python
# Add to src/elspeth/plugins/llm/__init__.py

from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

__all__ = [
    # ... existing exports ...
    "AzureMultiQueryLLMTransform",
]
```

**Step 2: Write integration test**

```python
# tests/integration/test_multi_query_integration.py
"""Integration tests for Azure Multi-Query LLM transform."""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform


def make_full_config() -> dict[str, Any]:
    """Create realistic config with 2 case studies × 5 criteria."""
    return {
        "deployment_name": "gpt-4o",
        "endpoint": "https://test.openai.azure.com",
        "api_key": "test-key",
        "system_prompt": "You are an assessment AI. Respond in JSON: {\"score\": <0-100>, \"rationale\": \"<text>\"}",
        "template": """
Case Study:
Background: {{ input_1 }}
Symptoms: {{ input_2 }}
History: {{ input_3 }}

Criterion: {{ criterion.name }}
Description: {{ criterion.description }}

Assess this case against the criterion.
""",
        "case_studies": [
            {"name": "cs1", "input_fields": ["cs1_background", "cs1_symptoms", "cs1_history"]},
            {"name": "cs2", "input_fields": ["cs2_background", "cs2_symptoms", "cs2_history"]},
        ],
        "criteria": [
            {"name": "diagnosis", "code": "DIAG", "description": "Assess diagnostic accuracy"},
            {"name": "treatment", "code": "TREAT", "description": "Assess treatment plan"},
            {"name": "prognosis", "code": "PROG", "description": "Assess prognosis accuracy"},
            {"name": "risk", "code": "RISK", "description": "Assess risk identification"},
            {"name": "followup", "code": "FOLLOW", "description": "Assess follow-up planning"},
        ],
        "response_format": "json",
        "output_mapping": {"score": "score", "rationale": "rationale"},
        "schema": {"fields": "dynamic"},
        "pool_size": 10,  # All 10 queries in parallel
        "temperature": 0.0,
    }


class TestMultiQueryIntegration:
    """Full integration tests for multi-query transform."""

    def test_full_assessment_matrix(self) -> None:
        """Test complete 2×5 assessment matrix."""
        # Generate responses for all 10 queries
        responses = []
        for cs in ["cs1", "cs2"]:
            for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                responses.append({
                    "score": len(responses) * 10,
                    "rationale": f"Assessment for {cs}_{crit}",
                })

        call_idx = [0]
        def make_response(**kwargs: Any) -> Mock:
            content = json.dumps(responses[call_idx[0] % len(responses)])
            call_idx[0] += 1

            mock_response = Mock()
            mock_response.choices = [Mock(message=Mock(content=content))]
            mock_response.model = "gpt-4o"
            mock_response.usage = Mock(prompt_tokens=50, completion_tokens=20)
            mock_response.model_dump = Mock(return_value={})
            return mock_response

        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = make_response
            mock_azure_class.return_value = mock_client

            transform = AzureMultiQueryLLMTransform(make_full_config())

            # Mock context
            mock_landscape = Mock()
            mock_landscape.record_external_call = Mock()
            transform.on_start(PluginContext(
                run_id="run-123",
                landscape=mock_landscape,
                state_id="init-state",
            ))

            ctx = PluginContext(
                run_id="run-123",
                landscape=mock_landscape,
                state_id="state-456",
            )

            row = {
                "user_id": "user-001",
                "cs1_background": "45yo male, office worker",
                "cs1_symptoms": "Chest pain, shortness of breath",
                "cs1_history": "Family history of heart disease",
                "cs2_background": "32yo female, athlete",
                "cs2_symptoms": "Knee pain after running",
                "cs2_history": "Previous ACL injury",
            }

            result = transform.process(row, ctx)

            # Should succeed
            assert result.status == "success"

            # Should have made 10 LLM calls
            assert mock_client.chat.completions.create.call_count == 10

            # Output should have all 20 assessment fields (10 scores + 10 rationales)
            output = result.row
            assert output["user_id"] == "user-001"  # Original preserved

            # Check all score fields exist
            for cs in ["cs1", "cs2"]:
                for crit in ["diagnosis", "treatment", "prognosis", "risk", "followup"]:
                    assert f"{cs}_{crit}_score" in output, f"Missing {cs}_{crit}_score"
                    assert f"{cs}_{crit}_rationale" in output, f"Missing {cs}_{crit}_rationale"

            transform.close()
```

**Step 3: Run integration test**

Run: `pytest tests/integration/test_multi_query_integration.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/plugins/llm/__init__.py tests/integration/test_multi_query_integration.py
git commit -m "feat(llm): register azure_multi_query_llm plugin and add integration test"
```

---

## Task 8: Add Contract Tests

**Files:**
- Create: `tests/contracts/transform_contracts/test_azure_multi_query_contract.py`

**Step 1: Write contract test class**

```python
# tests/contracts/transform_contracts/test_azure_multi_query_contract.py
"""Contract tests for AzureMultiQueryLLMTransform plugin."""

from typing import TYPE_CHECKING, Any
from unittest.mock import Mock, patch
import json

import pytest

from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

if TYPE_CHECKING:
    from elspeth.plugins.protocols import TransformProtocol

from .test_transform_protocol import TransformContractPropertyTestBase


def make_mock_response(content: str = '{"score": 85, "rationale": "test"}') -> Mock:
    """Create a mock Azure OpenAI response."""
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=content))]
    mock_response.model = "gpt-4o"
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    mock_response.model_dump = Mock(return_value={})
    return mock_response


class TestAzureMultiQueryLLMContract(TransformContractPropertyTestBase):
    """Contract tests for AzureMultiQueryLLMTransform.

    Inherits all standard transform contract tests plus adds
    multi-query-specific validation.
    """

    @pytest.fixture
    def transform(self) -> "TransformProtocol":
        """Return a configured transform instance."""
        return AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }} {{ criterion.name }}",
            "case_studies": [
                {"name": "cs1", "input_fields": ["cs1_a", "cs1_b"]},
            ],
            "criteria": [
                {"name": "test_criterion", "code": "TEST"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score", "rationale": "rationale"},
            "schema": {"fields": "dynamic"},
        })

    @pytest.fixture
    def valid_input(self) -> dict[str, Any]:
        """Return input that should process successfully (with mocked LLM)."""
        return {"cs1_a": "value_a", "cs1_b": "value_b"}

    @pytest.fixture
    def mock_azure_client(self) -> Mock:
        """Fixture to mock Azure OpenAI client for contract tests."""
        with patch("openai.AzureOpenAI") as mock_azure_class:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = make_mock_response()
            mock_azure_class.return_value = mock_client
            yield mock_client

    # Override process test to use mocked client
    def test_process_returns_transform_result(
        self,
        transform: "TransformProtocol",
        valid_input: dict[str, Any],
        mock_azure_client: Mock,
        mock_plugin_context: Mock,
    ) -> None:
        """Transform.process() returns TransformResult."""
        from elspeth.contracts import TransformResult

        # Setup context
        transform.on_start(mock_plugin_context)

        result = transform.process(valid_input, mock_plugin_context)

        assert isinstance(result, TransformResult)


class TestAzureMultiQueryLLMSpecific:
    """Multi-query-specific contract tests."""

    def test_query_expansion_matches_cross_product(self) -> None:
        """Query specs match case_studies × criteria cross-product."""
        transform = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }}",
            "case_studies": [
                {"name": "cs1", "input_fields": ["a"]},
                {"name": "cs2", "input_fields": ["b"]},
            ],
            "criteria": [
                {"name": "crit1"},
                {"name": "crit2"},
                {"name": "crit3"},
            ],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        # 2 case studies × 3 criteria = 6 queries
        assert len(transform._query_specs) == 6

        # Verify all combinations present
        prefixes = {s.output_prefix for s in transform._query_specs}
        assert prefixes == {
            "cs1_crit1", "cs1_crit2", "cs1_crit3",
            "cs2_crit1", "cs2_crit2", "cs2_crit3",
        }

    def test_is_batch_aware_true(self) -> None:
        """Transform declares batch awareness for aggregation support."""
        transform = AzureMultiQueryLLMTransform({
            "deployment_name": "gpt-4o",
            "endpoint": "https://test.openai.azure.com",
            "api_key": "test-key",
            "template": "{{ input_1 }}",
            "case_studies": [{"name": "cs1", "input_fields": ["a"]}],
            "criteria": [{"name": "crit1"}],
            "response_format": "json",
            "output_mapping": {"score": "score"},
            "schema": {"fields": "dynamic"},
        })

        assert transform.is_batch_aware is True
```

**Step 2: Run contract tests**

Run: `pytest tests/contracts/transform_contracts/test_azure_multi_query_contract.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/contracts/transform_contracts/test_azure_multi_query_contract.py
git commit -m "test: add contract tests for AzureMultiQueryLLMTransform"
```

---

## Task 9: Register Plugin in hookimpl and CLI

**Files:**
- Modify: `src/elspeth/plugins/llm/__init__.py`
- Modify: `src/elspeth/plugins/llm/hookimpl.py` (if exists, or create)
- Modify: `src/elspeth/cli.py`

**Step 1: Update LLM package exports**

```python
# src/elspeth/plugins/llm/__init__.py
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

__all__ = [
    # ... existing exports ...
    "AzureMultiQueryLLMTransform",
]
```

**Step 2: Register in CLI**

```python
# In src/elspeth/cli.py, find TRANSFORM_PLUGINS dict and add:
from elspeth.plugins.llm.azure_multi_query import AzureMultiQueryLLMTransform

TRANSFORM_PLUGINS: dict[str, type[BaseTransform]] = {
    # ... existing plugins ...
    "azure_multi_query_llm": AzureMultiQueryLLMTransform,
}
```

**Step 3: Verify registration**

Run: `elspeth plugins list | grep multi_query`
Expected: Shows `azure_multi_query_llm`

**Step 4: Commit**

```bash
git add src/elspeth/plugins/llm/__init__.py src/elspeth/cli.py
git commit -m "feat: register azure_multi_query_llm plugin"
```

---

## Task 10: Add Example Configuration

**Files:**
- Create: `examples/multi_query_assessment/suite.yaml`
- Create: `examples/multi_query_assessment/criteria_lookup.yaml`
- Create: `examples/multi_query_assessment/README.md`

**Step 1: Create example files**

```yaml
# examples/multi_query_assessment/suite.yaml
name: case_study_assessment
description: Assess multiple case studies against multiple criteria

source:
  plugin: csv_source
  options:
    path: input.csv
    schema:
      fields:
        - name: user_id
          type: string
        - name: cs1_background
          type: string
        - name: cs1_symptoms
          type: string
        - name: cs1_history
          type: string
        - name: cs2_background
          type: string
        - name: cs2_symptoms
          type: string
        - name: cs2_history
          type: string

transforms:
  - plugin: azure_multi_query_llm
    options:
      deployment_name: "${AZURE_OPENAI_DEPLOYMENT}"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_key: "${AZURE_OPENAI_KEY}"

      system_prompt: |
        You are a medical assessment AI. For each case study and criterion,
        provide a score (0-100) and rationale. Respond ONLY in JSON format:
        {"score": <number>, "rationale": "<explanation>"}

      template: |
        ## Case Study
        **Background:** {{ input_1 }}
        **Symptoms:** {{ input_2 }}
        **History:** {{ input_3 }}

        ## Evaluation Criterion: {{ criterion.name }}
        {{ criterion.description }}

        {% if criterion.subcriteria %}
        Consider these subcriteria:
        {% for sub in criterion.subcriteria %}
        - {{ sub }}
        {% endfor %}
        {% endif %}

        Provide your assessment.

      lookup_file: criteria_lookup.yaml

      case_studies:
        - name: cs1
          input_fields: [cs1_background, cs1_symptoms, cs1_history]
        - name: cs2
          input_fields: [cs2_background, cs2_symptoms, cs2_history]

      criteria:
        - name: diagnosis
          code: DIAG
          description: "Assess the accuracy and completeness of the diagnosis"
          subcriteria:
            - Correct primary diagnosis
            - Appropriate differential diagnoses considered
            - Evidence-based reasoning
        - name: treatment
          code: TREAT
          description: "Assess the appropriateness of the treatment plan"
          subcriteria:
            - Guideline-concordant treatment
            - Patient-specific considerations
            - Risk-benefit analysis
        - name: prognosis
          code: PROG
          description: "Assess the accuracy of prognostic assessment"
          subcriteria:
            - Realistic timeline
            - Key prognostic factors identified
            - Clear communication
        - name: risk
          code: RISK
          description: "Assess identification of risks and complications"
          subcriteria:
            - Comprehensive risk identification
            - Appropriate severity assessment
            - Mitigation strategies
        - name: followup
          code: FOLLOW
          description: "Assess the follow-up and monitoring plan"
          subcriteria:
            - Appropriate follow-up timeline
            - Clear monitoring parameters
            - Escalation criteria defined

      response_format: json
      output_mapping:
        score: score
        rationale: rationale

      pool_size: 10
      temperature: 0.0
      max_tokens: 500

      schema:
        fields: dynamic

sinks:
  - name: results
    plugin: csv_sink
    options:
      path: output.csv
```

```yaml
# examples/multi_query_assessment/criteria_lookup.yaml
DIAG:
  weight: 0.25
  guidance: "Focus on evidence-based diagnosis with clear reasoning chain"

TREAT:
  weight: 0.25
  guidance: "Ensure treatment aligns with current guidelines and patient factors"

PROG:
  weight: 0.20
  guidance: "Base prognosis on established prognostic factors"

RISK:
  weight: 0.15
  guidance: "Consider both common and serious complications"

FOLLOW:
  weight: 0.15
  guidance: "Define clear monitoring schedule and escalation triggers"
```

```markdown
# examples/multi_query_assessment/README.md
# Multi-Query Case Study Assessment Example

This example demonstrates the `azure_multi_query_llm` transform, which evaluates
multiple case studies against multiple criteria in a single pipeline step.

## Configuration

- **2 case studies** per row (cs1, cs2)
- **5 criteria** (diagnosis, treatment, prognosis, risk, followup)
- **10 LLM queries per row** (2 × 5 matrix)
- **All queries run in parallel** (pool_size: 10)
- **All-or-nothing error handling** per row

## Input Format

CSV with columns:
- `user_id` - Unique identifier
- `cs1_background`, `cs1_symptoms`, `cs1_history` - Case study 1 data
- `cs2_background`, `cs2_symptoms`, `cs2_history` - Case study 2 data

## Output Format

Original columns plus 20 assessment columns:
- `cs1_diagnosis_score`, `cs1_diagnosis_rationale`
- `cs1_treatment_score`, `cs1_treatment_rationale`
- ... (5 criteria × 2 case studies × 2 fields)

## Running

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

elspeth run --suite suite.yaml
```
```

**Step 2: Commit example**

```bash
git add examples/multi_query_assessment/
git commit -m "docs: add multi-query assessment example"
```

---

## Summary

**Total tasks:** 10

**Files created:**
- `src/elspeth/plugins/llm/multi_query.py` - QuerySpec, config models
- `src/elspeth/plugins/llm/azure_multi_query.py` - Main transform
- `tests/plugins/llm/test_multi_query.py` - Unit tests for config/spec
- `tests/plugins/llm/test_azure_multi_query.py` - Transform unit tests
- `tests/contracts/transform_contracts/test_azure_multi_query_contract.py` - Contract tests
- `tests/integration/test_multi_query_integration.py` - Integration tests
- `examples/multi_query_assessment/` - Example configuration

**Files modified:**
- `src/elspeth/plugins/llm/__init__.py` - Export new transform
- `src/elspeth/cli.py` - Register plugin in CLI

**Key patterns used:**
- Pydantic config validation (existing pattern from `AzureOpenAIConfig`)
- `PooledExecutor` for parallel execution (existing infrastructure)
- `PromptTemplate` for Jinja2 rendering with lookups (existing)
- `TransformResult.success_multi()` for batch output (existing)
- All-or-nothing error semantics per row
- JSON response parsing with field mapping

**Testing approach:**
- Unit tests for each component in isolation
- Integration test for full 2×5 assessment matrix
- Mocked Azure OpenAI client (existing pattern)
