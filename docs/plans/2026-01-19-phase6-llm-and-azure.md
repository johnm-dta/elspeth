# Phase 6: LLM Transforms & Azure Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM transform plugins (OpenRouter single, Azure single, Azure batch) with Jinja2 templating, Azure blob storage sources/sinks, and supporting infrastructure for external call recording.

**Architecture:** Three-part implementation:
- **Part A:** LLM transform plugins with Jinja2 prompt templating
- **Part B:** Azure Blob Storage source and sink plugins
- **Part C:** External call recording, replay, and verification infrastructure

**Tech Stack:**
- Jinja2 (prompt templating)
- httpx (HTTP client for OpenRouter)
- azure-storage-blob (Azure Blob Storage)
- azure-identity (Azure authentication)
- openai (Azure OpenAI SDK)

---

## Part A: LLM Transform Plugins

### Overview

Three LLM transform plugins, all using Jinja2 for prompt templating:

| Plugin | Provider | Pattern | Use Case |
|--------|----------|---------|----------|
| `OpenRouterLLMTransform` | OpenRouter | Single row → single call | General LLM access (Claude, Llama, Mistral, etc.) |
| `AzureLLMTransform` | Azure OpenAI | Single row → single call | Low-latency Azure workloads |
| `AzureBatchLLMTransform` | Azure OpenAI Batch API | Aggregation → batch submit → fan-out | High-volume workloads (50% cost savings) |

### Jinja2 Templating Design

**Why Jinja2:**
- Template is hashable separately from variables (audit trail)
- Battle-tested (Ansible, Flask, dbt, Airflow)
- Sandboxed mode available for safety
- No LLM framework lock-in

**Template Recording in Audit Trail:**
```python
# Recorded in node_states table:
template_hash = sha256(template_string)      # The Jinja2 template
variables_hash = sha256(canonical(row_data)) # The row data injected
rendered_hash = sha256(rendered_prompt)      # Final prompt sent to LLM
```

---

### Task A1: Create Jinja2 Template Engine Module

**Files:**
- Create: `src/elspeth/plugins/llm/__init__.py`
- Create: `src/elspeth/plugins/llm/templates.py`
- Create: `tests/plugins/llm/__init__.py`
- Create: `tests/plugins/llm/test_templates.py`

**Step 1: Write the failing test**

```python
# tests/plugins/llm/test_templates.py
"""Tests for Jinja2 prompt template engine."""

import pytest

from elspeth.plugins.llm.templates import PromptTemplate, TemplateError


class TestPromptTemplate:
    """Tests for PromptTemplate wrapper."""

    def test_simple_variable_substitution(self):
        """Basic variable substitution works."""
        template = PromptTemplate("Hello, {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello, World!"

    def test_template_with_loop(self):
        """Jinja2 loops work."""
        template = PromptTemplate("""
Analyze these items:
{% for item in items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
""".strip())
        result = template.render(items=[
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ])
        assert "- A: 1" in result
        assert "- B: 2" in result

    def test_template_with_default_filter(self):
        """Jinja2 default filter works."""
        template = PromptTemplate("Focus: {{ focus | default('general') }}")
        assert template.render() == "Focus: general"
        assert template.render(focus="quality") == "Focus: quality"

    def test_template_hash_is_stable(self):
        """Same template string produces same hash."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Hello, {{ name }}!")
        assert t1.template_hash == t2.template_hash

    def test_different_templates_have_different_hashes(self):
        """Different templates have different hashes."""
        t1 = PromptTemplate("Hello, {{ name }}!")
        t2 = PromptTemplate("Goodbye, {{ name }}!")
        assert t1.template_hash != t2.template_hash

    def test_render_returns_metadata(self):
        """render() returns prompt and audit metadata."""
        template = PromptTemplate("Analyze: {{ text }}")
        result = template.render_with_metadata(text="sample")

        assert result.prompt == "Analyze: sample"
        assert result.template_hash is not None
        assert result.variables_hash is not None
        assert result.rendered_hash is not None

    def test_undefined_variable_raises_error(self):
        """Missing required variable raises TemplateError."""
        template = PromptTemplate("Hello, {{ name }}!")
        with pytest.raises(TemplateError, match="name"):
            template.render()  # No 'name' provided

    def test_sandboxed_prevents_dangerous_operations(self):
        """Sandboxed environment blocks dangerous operations."""
        # Attempt to access file system
        dangerous = PromptTemplate("{{ ''.__class__.__mro__ }}")
        with pytest.raises(TemplateError):
            dangerous.render()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_templates.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'elspeth.plugins.llm'"

**Step 3: Implement PromptTemplate**

```python
# src/elspeth/plugins/llm/__init__.py
"""LLM transform plugins for ELSPETH."""

from elspeth.plugins.llm.templates import PromptTemplate, TemplateError

__all__ = ["PromptTemplate", "TemplateError"]


# src/elspeth/plugins/llm/templates.py
"""Jinja2-based prompt templating with audit support."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateSyntaxError, UndefinedError
from jinja2.sandbox import SandboxedEnvironment

from elspeth.core.canonical import canonical_json


class TemplateError(Exception):
    """Error in template rendering."""
    pass


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt with audit metadata."""
    prompt: str
    template_hash: str
    variables_hash: str
    rendered_hash: str


def _sha256(content: str) -> str:
    """Compute SHA-256 hash of string content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class PromptTemplate:
    """Jinja2 prompt template with audit trail support.

    Uses sandboxed environment to prevent dangerous operations.
    Tracks hashes of template, variables, and rendered output for audit.

    Example:
        template = PromptTemplate('''
            Analyze the following product:
            Name: {{ product.name }}
            Description: {{ product.description }}

            Provide a quality score from 1-10.
        ''')

        result = template.render_with_metadata(
            product={"name": "Widget", "description": "A useful widget"}
        )

        # result.prompt = rendered string
        # result.template_hash = hash of template
        # result.variables_hash = hash of input variables
        # result.rendered_hash = hash of final prompt
    """

    def __init__(self, template_string: str) -> None:
        """Initialize template.

        Args:
            template_string: Jinja2 template string

        Raises:
            TemplateError: If template syntax is invalid
        """
        self._template_string = template_string
        self._template_hash = _sha256(template_string)

        # Use sandboxed environment for security
        self._env = SandboxedEnvironment(
            undefined=StrictUndefined,  # Raise on undefined variables
            autoescape=False,           # No HTML escaping for prompts
        )

        try:
            self._template = self._env.from_string(template_string)
        except TemplateSyntaxError as e:
            raise TemplateError(f"Invalid template syntax: {e}") from e

    @property
    def template_hash(self) -> str:
        """SHA-256 hash of the template string."""
        return self._template_hash

    def render(self, **variables: Any) -> str:
        """Render template with variables.

        Args:
            **variables: Template variables

        Returns:
            Rendered prompt string

        Raises:
            TemplateError: If rendering fails (undefined variable, etc.)
        """
        try:
            return self._template.render(**variables)
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable: {e}") from e
        except Exception as e:
            raise TemplateError(f"Template rendering failed: {e}") from e

    def render_with_metadata(self, **variables: Any) -> RenderedPrompt:
        """Render template and return with audit metadata.

        Args:
            **variables: Template variables

        Returns:
            RenderedPrompt with prompt string and all hashes
        """
        prompt = self.render(**variables)

        # Compute variables hash using canonical JSON
        variables_canonical = canonical_json(variables)
        variables_hash = _sha256(variables_canonical)

        # Compute rendered prompt hash
        rendered_hash = _sha256(prompt)

        return RenderedPrompt(
            prompt=prompt,
            template_hash=self._template_hash,
            variables_hash=variables_hash,
            rendered_hash=rendered_hash,
        )
```

**Step 4: Run tests**

Run: `pytest tests/plugins/llm/test_templates.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/ tests/plugins/llm/
git commit -m "$(cat <<'EOF'
feat(plugins): add Jinja2 prompt template engine

Sandboxed Jinja2 templating with audit trail support:
- template_hash: hash of template string
- variables_hash: hash of input variables
- rendered_hash: hash of final prompt

Used by all LLM transform plugins.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task A2: Create Base LLM Transform

**Files:**
- Create: `src/elspeth/plugins/llm/base.py`
- Test: `tests/plugins/llm/test_base.py`

**Step 1: Write the failing test**

```python
# tests/plugins/llm/test_base.py
"""Tests for base LLM transform."""

import pytest
from unittest.mock import Mock, AsyncMock

from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, LLMResponse


class TestBaseLLMTransform:
    """Tests for BaseLLMTransform abstract base class."""

    def test_config_requires_template(self):
        """LLMConfig requires a prompt template."""
        with pytest.raises(ValueError, match="template"):
            LLMConfig(model="gpt-4")

    def test_config_accepts_template_string(self):
        """LLMConfig can take template as string."""
        config = LLMConfig(
            model="gpt-4",
            template="Analyze: {{ text }}",
        )
        assert config.template is not None

    def test_config_validates_model(self):
        """LLMConfig requires model name."""
        with pytest.raises(ValueError, match="model"):
            LLMConfig(template="Hello")

    def test_llm_response_has_required_fields(self):
        """LLMResponse has content and metadata."""
        response = LLMResponse(
            content="Analysis result",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            latency_ms=150.5,
        )
        assert response.content == "Analysis result"
        assert response.total_tokens == 30
```

**Step 2: Implement BaseLLMTransform**

```python
# src/elspeth/plugins/llm/base.py
"""Base class for LLM transforms."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator

from elspeth.contracts import TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.templates import PromptTemplate


class LLMConfig(BaseModel):
    """Configuration for LLM transforms."""

    model_config = {"frozen": True}

    model: str = Field(..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")

    # Response parsing
    response_field: str = Field("llm_response", description="Field name for LLM response in output")

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template is non-empty."""
        if not v or not v.strip():
            raise ValueError("template cannot be empty")
        return v


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (prompt + completion)."""
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)


class BaseLLMTransform(BaseTransform):
    """Abstract base class for LLM transforms.

    Subclasses implement _call_llm() for specific providers.
    Base class handles:
    - Template rendering
    - Audit trail recording
    - Response parsing
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._template = PromptTemplate(config.template)

    @property
    def name(self) -> str:
        return f"llm:{self._config.model}"

    @abstractmethod
    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None,
        ctx: PluginContext,
    ) -> LLMResponse:
        """Make the actual LLM call. Implemented by subclasses."""
        ...

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row through the LLM.

        1. Render prompt template with row data
        2. Call LLM provider
        3. Record call in audit trail
        4. Return transformed row
        """
        # Render template
        rendered = self._template.render_with_metadata(**row)

        # Call LLM (implemented by subclass)
        response = self._call_llm(
            prompt=rendered.prompt,
            system_prompt=self._config.system_prompt,
            ctx=ctx,
        )

        # Record in audit trail
        ctx.record_external_call(
            call_type="llm",
            request={
                "model": self._config.model,
                "template_hash": rendered.template_hash,
                "variables_hash": rendered.variables_hash,
                "rendered_hash": rendered.rendered_hash,
                "temperature": self._config.temperature,
            },
            response={
                "content_hash": _sha256(response.content),
                "model": response.model,
                "usage": response.usage,
            },
            latency_ms=response.latency_ms,
        )

        # Build output row
        output = dict(row)
        output[self._config.response_field] = response.content
        output[f"{self._config.response_field}_usage"] = response.usage

        return TransformResult.success(output)


def _sha256(content: str) -> str:
    """Compute SHA-256 hash."""
    import hashlib
    return hashlib.sha256(content.encode()).hexdigest()
```

**Step 3: Run tests and commit**

---

### Task A3: Implement OpenRouterLLMTransform

**Files:**
- Create: `src/elspeth/plugins/llm/openrouter.py`
- Test: `tests/plugins/llm/test_openrouter.py`

**Configuration:**
```yaml
transforms:
  - plugin: openrouter_llm
    options:
      model: "anthropic/claude-3-opus"
      template: |
        Analyze the following product review:

        Review: {{ review_text }}

        Provide sentiment (positive/negative/neutral) and key themes.
      api_key: "${OPENROUTER_API_KEY}"  # Env var interpolation
      temperature: 0.0
      response_field: "analysis"
```

**Implementation highlights:**
```python
# src/elspeth/plugins/llm/openrouter.py
"""OpenRouter LLM transform - access 100+ models via single API."""

import time
import httpx

from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig, LLMResponse


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration."""
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 60.0


class OpenRouterLLMTransform(BaseLLMTransform):
    """LLM transform using OpenRouter API."""

    def __init__(self, config: OpenRouterConfig) -> None:
        super().__init__(config)
        self._api_key = config.api_key
        self._base_url = config.base_url
        self._timeout = config.timeout_seconds

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str | None,
        ctx: PluginContext,
    ) -> LLMResponse:
        """Call OpenRouter API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()

        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._config.model,
                    "messages": messages,
                    "temperature": self._config.temperature,
                    "max_tokens": self._config.max_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            model=data.get("model", self._config.model),
            usage=data.get("usage", {}),
            latency_ms=latency_ms,
            raw_response=data,
        )
```

---

### Task A4: Implement AzureLLMTransform (Single)

**Files:**
- Create: `src/elspeth/plugins/llm/azure.py`
- Test: `tests/plugins/llm/test_azure.py`

**Configuration:**
```yaml
transforms:
  - plugin: azure_llm
    options:
      model: "gpt-4o"
      deployment_name: "my-gpt4o-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_key: "${AZURE_OPENAI_KEY}"
      api_version: "2024-10-21"
      template: |
        {{ instruction }}

        Input: {{ input_text }}
```

**Implementation uses Azure OpenAI SDK:**
```python
# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform - single call per row."""

from openai import AzureOpenAI

class AzureLLMTransform(BaseLLMTransform):
    """LLM transform using Azure OpenAI."""

    def __init__(self, config: AzureOpenAIConfig) -> None:
        super().__init__(config)
        self._client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )
        self._deployment = config.deployment_name

    def _call_llm(self, prompt: str, system_prompt: str | None, ctx: PluginContext) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=self._config.temperature,
            max_tokens=self._config.max_tokens,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            latency_ms=latency_ms,
        )
```

---

### Task A5: Implement AzureBatchLLMTransform (Aggregation Pattern)

**Files:**
- Create: `src/elspeth/plugins/llm/azure_batch.py`
- Test: `tests/plugins/llm/test_azure_batch.py`

**Configuration:**
```yaml
transforms:
  - plugin: azure_batch_llm
    options:
      model: "gpt-4o"
      deployment_name: "my-gpt4o-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_key: "${AZURE_OPENAI_KEY}"
      template: |
        Analyze: {{ text }}
      batch_size: 100              # Rows per batch
      poll_interval_seconds: 300   # How often to check batch status (5 min default)
      max_wait_hours: 24           # Maximum wait time
```

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    AzureBatchLLMTransform                       │
│                    (implements BaseAggregation)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  collect(row, ctx)                                               │
│  ├── Render template for this row                                │
│  ├── Store (token_id, rendered_prompt) in buffer                 │
│  └── Return CONTINUE (row added to batch)                       │
│                                                                  │
│  trigger(batch_info, ctx)                                        │
│  ├── Check if batch_size reached                                 │
│  └── Return FLUSH or CONTINUE                                    │
│                                                                  │
│  flush(ctx) -> AggregationResult                                 │
│  ├── Build JSONL from buffered prompts                           │
│  ├── Upload to Azure Blob Storage                                │
│  ├── Submit batch job to Azure OpenAI                           │
│  ├── Poll until complete (or timeout)                            │
│  ├── Download results                                            │
│  ├── Map results back to original token_ids                      │
│  └── Return AggregationResult with fan-out rows                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key implementation:**

```python
# src/elspeth/plugins/llm/azure_batch.py
"""Azure OpenAI Batch API transform - 50% cost savings for high volume."""

import json
import time
from dataclasses import dataclass
from typing import Any

from openai import AzureOpenAI

from elspeth.contracts import AggregationResult
from elspeth.plugins.base import BaseAggregation
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.templates import PromptTemplate


@dataclass
class BufferedRequest:
    """A request waiting in the batch buffer."""
    token_id: str
    custom_id: str  # Unique ID for matching response to request
    prompt: str
    row_data: dict[str, Any]


class AzureBatchLLMTransform(BaseAggregation):
    """Batch LLM transform using Azure OpenAI Batch API.

    Collects rows into batches, submits to Azure Batch API,
    waits for completion, and fans out results.

    Benefits:
    - 50% cost reduction vs real-time API
    - Separate, higher rate limits
    - Up to 24-hour turnaround
    """

    def __init__(self, config: AzureBatchConfig) -> None:
        self._config = config
        self._template = PromptTemplate(config.template)
        self._buffer: list[BufferedRequest] = []
        self._client = AzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )

    def collect(self, row: dict[str, Any], ctx: PluginContext) -> None:
        """Add row to batch buffer."""
        # Render prompt for this row
        rendered = self._template.render_with_metadata(**row)

        # Create unique ID for response matching
        custom_id = f"{ctx.token_id}"

        self._buffer.append(BufferedRequest(
            token_id=ctx.token_id,
            custom_id=custom_id,
            prompt=rendered.prompt,
            row_data=row,
        ))

    def should_flush(self, ctx: PluginContext) -> bool:
        """Check if batch should be submitted."""
        return len(self._buffer) >= self._config.batch_size

    def flush(self, ctx: PluginContext) -> AggregationResult:
        """Submit batch to Azure and wait for results."""
        if not self._buffer:
            return AggregationResult.empty()

        # 1. Build JSONL content
        jsonl_lines = []
        for req in self._buffer:
            jsonl_lines.append(json.dumps({
                "custom_id": req.custom_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._config.deployment_name,
                    "messages": [
                        {"role": "system", "content": self._config.system_prompt or ""},
                        {"role": "user", "content": req.prompt},
                    ],
                    "temperature": self._config.temperature,
                    "max_tokens": self._config.max_tokens,
                },
            }))

        jsonl_content = "\n".join(jsonl_lines)

        # 2. Upload file to Azure
        file_response = self._client.files.create(
            file=("batch_input.jsonl", jsonl_content.encode()),
            purpose="batch",
        )

        # 3. Submit batch job
        batch_response = self._client.batches.create(
            input_file_id=file_response.id,
            endpoint="/chat/completions",
            completion_window="24h",
        )

        batch_id = batch_response.id

        # Record batch submission in audit trail
        ctx.record_external_call(
            call_type="llm_batch",
            request={
                "batch_id": batch_id,
                "file_id": file_response.id,
                "row_count": len(self._buffer),
                "model": self._config.deployment_name,
            },
            response={"status": "submitted"},
            latency_ms=0,
        )

        # 4. Poll until complete
        results = self._poll_until_complete(batch_id, ctx)

        # 5. Map results back to tokens and build fan-out
        output_rows = []
        for req in self._buffer:
            if req.custom_id in results:
                result_content = results[req.custom_id]
                output_row = dict(req.row_data)
                output_row[self._config.response_field] = result_content
                output_rows.append({
                    "token_id": req.token_id,
                    "row": output_row,
                })
            else:
                # Missing result - record error
                output_rows.append({
                    "token_id": req.token_id,
                    "row": req.row_data,
                    "error": "No response in batch results",
                })

        # Clear buffer
        self._buffer = []

        return AggregationResult.fan_out(output_rows)

    def _poll_until_complete(
        self,
        batch_id: str,
        ctx: PluginContext,
    ) -> dict[str, str]:
        """Poll Azure until batch completes, return results by custom_id."""
        max_wait = self._config.max_wait_hours * 3600
        poll_interval = self._config.poll_interval_seconds
        waited = 0

        while waited < max_wait:
            batch = self._client.batches.retrieve(batch_id)

            if batch.status == "completed":
                # Download results
                output_file = self._client.files.content(batch.output_file_id)
                return self._parse_batch_results(output_file.text)

            if batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(f"Batch {batch_id} failed with status: {batch.status}")

            time.sleep(poll_interval)
            waited += poll_interval

        raise RuntimeError(f"Batch {batch_id} timed out after {max_wait}s")

    def _parse_batch_results(self, jsonl_content: str) -> dict[str, str]:
        """Parse JSONL results file into custom_id -> content map."""
        results = {}
        for line in jsonl_content.strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            custom_id = data["custom_id"]
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            results[custom_id] = content
        return results
```

---

### Task A6: Integration Tests for LLM Transforms

**Files:**
- Create: `tests/integration/test_llm_transforms.py`

Tests with mocked HTTP responses to verify:
1. Template rendering → API call → response parsing
2. Audit trail records template_hash, variables_hash, rendered_hash
3. Batch aggregation → fan-out works correctly
4. Error handling for API failures

---

## Part B: Azure Blob Storage Source & Sink

### Overview

| Plugin | Direction | Use Case |
|--------|-----------|----------|
| `AzureBlobSource` | Source | Read CSV/JSON/Parquet from Azure Blob Storage |
| `AzureBlobSink` | Sink | Write results to Azure Blob Storage |

---

### Task B1: Implement AzureBlobSource

**Files:**
- Create: `src/elspeth/plugins/azure/__init__.py`
- Create: `src/elspeth/plugins/azure/blob_source.py`
- Test: `tests/plugins/azure/test_blob_source.py`

**Configuration:**
```yaml
datasource:
  plugin: azure_blob
  options:
    connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
    container: "input-data"
    blob_path: "data/input.csv"
    format: csv  # csv, json, jsonl, parquet
    csv_options:
      delimiter: ","
      has_header: true
```

**Implementation:**
```python
# src/elspeth/plugins/azure/blob_source.py
"""Azure Blob Storage source plugin."""

from typing import Any, Iterator

from azure.storage.blob import BlobServiceClient

from elspeth.plugins.base import BaseSource
from elspeth.plugins.context import PluginContext


class AzureBlobSource(BaseSource):
    """Read data from Azure Blob Storage."""

    def __init__(self, config: AzureBlobSourceConfig) -> None:
        self._config = config
        self._client = BlobServiceClient.from_connection_string(
            config.connection_string
        )

    def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
        """Stream rows from blob."""
        container = self._client.get_container_client(self._config.container)
        blob = container.get_blob_client(self._config.blob_path)

        # Download blob content
        content = blob.download_blob().readall()

        # Parse based on format
        if self._config.format == "csv":
            yield from self._parse_csv(content)
        elif self._config.format == "json":
            yield from self._parse_json(content)
        elif self._config.format == "jsonl":
            yield from self._parse_jsonl(content)
        elif self._config.format == "parquet":
            yield from self._parse_parquet(content)

    def _parse_csv(self, content: bytes) -> Iterator[dict[str, Any]]:
        """Parse CSV content."""
        import csv
        import io

        text = content.decode("utf-8")
        reader = csv.DictReader(
            io.StringIO(text),
            delimiter=self._config.csv_options.get("delimiter", ","),
        )
        yield from reader
```

---

### Task B2: Implement AzureBlobSink

**Files:**
- Create: `src/elspeth/plugins/azure/blob_sink.py`
- Test: `tests/plugins/azure/test_blob_sink.py`

**Configuration:**
```yaml
sinks:
  output:
    plugin: azure_blob
    options:
      connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
      container: "output-data"
      blob_path: "results/{{ run_id }}/output.csv"  # Jinja2 for dynamic paths
      format: csv
      overwrite: true
```

---

### Task B3: Add Azure Authentication Options

Support multiple auth methods:
```yaml
# Option 1: Connection string
connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"

# Option 2: Managed Identity (for Azure-hosted workloads)
use_managed_identity: true
account_url: "https://mystorageaccount.blob.core.windows.net"

# Option 3: Service Principal
tenant_id: "${AZURE_TENANT_ID}"
client_id: "${AZURE_CLIENT_ID}"
client_secret: "${AZURE_CLIENT_SECRET}"
account_url: "https://mystorageaccount.blob.core.windows.net"
```

---

## Part C: External Call Infrastructure

### Overview

This part implements the foundational infrastructure from the original Phase 6 plan:

1. **CallRecorder** - Record external calls to landscape
2. **CallReplayer** - Replay recorded responses (for testing)
3. **CallVerifier** - Compare live vs recorded (for verification)
4. **Run Modes** - live / replay / verify

---

### Task C1: CallRecorder (from original Phase 6 Task 1)

Record external calls to `calls` table with request/response in PayloadStore.

*(Implementation as in original plan)*

---

### Task C2: CallReplayer (from original Phase 6 Task 3)

Replay recorded responses instead of making live calls.

---

### Task C3: CallVerifier (from original Phase 6 Task 5)

Compare live responses to recorded responses using DeepDiff.

---

### Task C4: Run Modes Integration

Add `run_mode` to ElspethSettings:
```yaml
run_mode: live    # live | replay | verify
```

Behavior:
- **live**: Make real API calls, record everything
- **replay**: Use recorded responses, skip API calls
- **verify**: Make real calls, compare to recorded, alert on differences

---

### Task C5: PluginContext.record_external_call()

Wire the `ctx.record_external_call()` method used by LLM transforms to CallRecorder.

---

## Summary

| Part | Tasks | Effort | Dependencies |
|------|-------|--------|--------------|
| **A: LLM Plugins** | A1-A6 | 3-4 days | Jinja2, httpx, openai SDK |
| **B: Azure Storage** | B1-B3 | 2 days | azure-storage-blob, azure-identity |
| **C: Call Infrastructure** | C1-C5 | 2-3 days | DeepDiff, PayloadStore |

**Recommended order:**
1. **C1** (CallRecorder) - foundation for audit
2. **A1** (Templates) - needed by all LLM plugins
3. **A2** (BaseLLMTransform) - shared base class
4. **A3** (OpenRouter) - quickest to test
5. **A4** (Azure Single) - validates Azure integration
6. **B1-B2** (Azure Storage) - enables Azure-native pipelines
7. **A5** (Azure Batch) - the big one for your workload
8. **C2-C5** (Replay/Verify) - nice-to-have for testing

---

## Appendix: Package Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
llm = [
    "jinja2>=3.1",
    "httpx>=0.27",
    "openai>=1.0",
]
azure = [
    "azure-storage-blob>=12.0",
    "azure-identity>=1.15",
]
```

Install with:
```bash
uv pip install -e ".[llm,azure]"
```
