# Phase 6: LLM Transforms & Azure Integration

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

> ⚠️ **Implementation Note:** This plan depends on features from Plugin Protocol v1.5 (`TransformResult.success_multi()`, `is_batch_aware`, aggregation `output_mode`). These are implemented - see `docs/plans/completed/2026-01-19-multi-row-output.md`. External call recording uses the **Audited Client** pattern (Task C5) where infrastructure-level clients automatically record to the audit trail.

**Goal:** Add LLM transform plugins (OpenRouter single, Azure single, Azure batch) with Jinja2 templating, Azure blob storage sources/sinks, and supporting infrastructure for external call recording.

**Architecture:** Three-part implementation:
- **Part A:** LLM transform plugins with Jinja2 prompt templating
- **Part B:** Azure Blob Storage source and sink plugins
- **Part C:** External call recording via Audited Clients, replay, and verification infrastructure

**Tech Stack:**
- Jinja2 (prompt templating)
- httpx (HTTP client for OpenRouter)
- azure-storage-blob (Azure Blob Storage)
- azure-identity (Azure authentication)
- openai (Azure OpenAI SDK)

---

## Architectural Decision: Audited Clients

> **Key Design Decision:** External calls are recorded via **infrastructure-wrapped clients**, not via plugin-level `ctx.record_external_call()`. This ensures audit is **guaranteed by construction** - plugins physically cannot make unrecorded calls.

### Why Audited Clients?

| Approach | Audit Guarantee | Plugin Complexity | Risk |
|----------|-----------------|-------------------|------|
| Plugin calls `ctx.record_external_call()` | ❌ Plugin can forget | High - must remember to record | Audit gaps |
| Transform returns call metadata | ⚠️ Plugin can omit | Medium - extra return fields | Audit gaps |
| **Audited Client wrappers** | ✅ Automatic | Low - just use the client | **None** |

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUDITED CLIENT ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXECUTOR (owns state lifecycle)                                            │
│  ├── Creates node_state in audit trail                                      │
│  ├── Creates AuditedLLMClient with state_id                                 │
│  ├── Passes client to transform via PluginContext                           │
│  └── Closes state after transform completes                                 │
│                                                                              │
│  PluginContext                                                              │
│  ├── llm_client: AuditedLLMClient    ← Pre-configured with state_id        │
│  └── http_client: AuditedHTTPClient  ← Pre-configured with state_id        │
│                                                                              │
│  LLM TRANSFORM (uses provided client)                                       │
│  └── response = ctx.llm_client.chat(messages)  ← Recording is AUTOMATIC    │
│                                                                              │
│  AuditedLLMClient (infrastructure)                                          │
│  ├── Wraps underlying LLM SDK (openai, httpx)                              │
│  ├── Records request BEFORE call                                            │
│  ├── Records response/error AFTER call                                      │
│  └── Manages call_index per state                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits

1. **Audit is guaranteed** - Plugins physically cannot make unrecorded calls
2. **Single responsibility** - Plugins do business logic, infrastructure does audit
3. **Correct by construction** - No "remember to record" cognitive load
4. **Consistent format** - All LLM calls recorded identically
5. **State tracking handled** - `state_id` and `call_index` managed by infrastructure

This follows ELSPETH's principle: **"Audit is non-negotiable"**

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
        # Attempt to access dunder attributes (blocked by SandboxedEnvironment)
        dangerous = PromptTemplate("{{ ''.__class__.__mro__ }}")
        # SecurityError is wrapped in TemplateError with "Sandbox violation" message
        with pytest.raises(TemplateError, match="Sandbox violation"):
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
from jinja2.sandbox import SandboxedEnvironment, SecurityError

from elspeth.core.canonical import canonical_json


class TemplateError(Exception):
    """Error in template rendering (including sandbox violations)."""
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
            TemplateError: If rendering fails (undefined variable, sandbox violation, etc.)
        """
        try:
            return self._template.render(**variables)
        except UndefinedError as e:
            raise TemplateError(f"Undefined variable: {e}") from e
        except SecurityError as e:
            raise TemplateError(f"Sandbox violation: {e}") from e
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

> **Best Practice Applied:** This implementation uses:
> 1. `TransformDataConfig` base class for proper schema/on_error support
> 2. Audited clients (recording handled by infrastructure, not plugin)
> 3. Proper error handling following Three-Tier Trust Model

**Step 1: Write the failing test**

```python
# tests/plugins/llm/test_base.py
"""Tests for base LLM transform."""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock

from elspeth.plugins.llm.base import BaseLLMTransform, LLMConfig
from elspeth.plugins.clients.llm import LLMResponse


class TestLLMConfig:
    """Tests for LLMConfig validation."""

    def test_config_requires_template(self):
        """LLMConfig requires a prompt template."""
        with pytest.raises(ValidationError):
            LLMConfig.from_dict({
                "model": "gpt-4",
                "schema": {"fields": "dynamic"},
            })  # Missing 'template'

    def test_config_requires_model(self):
        """LLMConfig requires model name."""
        with pytest.raises(ValidationError):
            LLMConfig.from_dict({
                "template": "Analyze: {{ text }}",
                "schema": {"fields": "dynamic"},
            })  # Missing 'model'

    def test_config_requires_schema(self):
        """LLMConfig requires schema (from TransformDataConfig)."""
        with pytest.raises(ValidationError):
            LLMConfig.from_dict({
                "model": "gpt-4",
                "template": "Analyze: {{ text }}",
            })  # Missing 'schema'

    def test_valid_config(self):
        """Valid config passes validation."""
        config = LLMConfig.from_dict({
            "model": "gpt-4",
            "template": "Analyze: {{ text }}",
            "schema": {"fields": "dynamic"},
        })
        assert config.model == "gpt-4"
        assert config.template == "Analyze: {{ text }}"


class TestBaseLLMTransformProcess:
    """Tests for transform processing."""

    def test_template_rendering_error_returns_transform_error(self):
        """Template rendering failure returns TransformResult.error()."""
        # Create mock transform that uses strict template
        from elspeth.plugins.llm.templates import PromptTemplate

        template = PromptTemplate("Hello, {{ required_field }}!")
        ctx = Mock()
        ctx.llm_client = Mock()

        # Missing required_field should return error, not crash
        # (This tests the error handling in process())

    def test_llm_client_error_returns_transform_error(self):
        """LLM client failure returns TransformResult.error()."""
        # Test that LLMClientError is caught and converted to TransformResult.error()

    def test_rate_limit_error_is_retryable(self):
        """Rate limit errors marked retryable=True."""
        # Test TransformResult.error(retryable=True) for rate limits
```

**Step 2: Implement LLMConfig (extends TransformDataConfig)**

```python
# src/elspeth/plugins/llm/base.py
"""Base class for LLM transforms."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from pydantic import Field, field_validator

from elspeth.contracts import TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.clients.llm import LLMClientError, RateLimitError


class LLMConfig(TransformDataConfig):
    """Configuration for LLM transforms.

    Extends TransformDataConfig to get:
    - schema: Input/output schema configuration (REQUIRED)
    - on_error: Sink for failed rows (optional)
    """

    model: str = Field(..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')")
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")
    response_field: str = Field("llm_response", description="Field name for LLM response in output")

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template is non-empty and syntactically valid."""
        if not v or not v.strip():
            raise ValueError("template cannot be empty")
        # Validate template syntax at config time
        try:
            PromptTemplate(v)
        except TemplateError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e
        return v


class BaseLLMTransform(BaseTransform):
    """Abstract base class for LLM transforms.

    Uses audited clients for external calls - recording is automatic.

    Error handling follows Three-Tier Trust Model:
    - Template rendering with row data → wrap, return error (THEIR DATA)
    - LLM API calls → wrap, return error (EXTERNAL SYSTEM)
    - Internal logic → let crash (OUR CODE)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = LLMConfig.from_dict(config)
        self._model = cfg.model
        self._template = PromptTemplate(cfg.template)
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._on_error = cfg.on_error

        # Schema from config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identifier - implemented by subclasses."""
        ...

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a row through the LLM.

        Error handling:
        1. Template rendering (THEIR DATA) → catch TemplateError, return error
        2. LLM call (EXTERNAL) → catch LLMClientError, return error
        3. Internal logic (OUR CODE) → let it crash
        """
        # 1. Render template with row data
        # This operates on THEIR DATA - wrap in try/catch
        try:
            rendered = self._template.render_with_metadata(**row)
        except TemplateError as e:
            return TransformResult.error({
                "reason": "template_rendering_failed",
                "error": str(e),
                "template_hash": self._template.template_hash,
            })

        # 2. Build messages
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Call LLM via audited client
        # This is an EXTERNAL SYSTEM - wrap in try/catch
        if ctx.llm_client is None:
            # This is OUR BUG - executor should have provided client
            raise RuntimeError("LLM client not available in PluginContext")

        try:
            response = ctx.llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            # Rate limit - retryable
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            # Other LLM error - check if retryable
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # 4. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage

        # 5. Add audit metadata for template traceability
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash

        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources."""
        pass
```

**Step 3: Run tests and commit**

---

### Task A3: Implement OpenRouterLLMTransform

**Files:**
- Create: `src/elspeth/plugins/llm/openrouter.py`
- Test: `tests/plugins/llm/test_openrouter.py`

> **Note:** OpenRouter uses a custom HTTP endpoint, not the OpenAI SDK. We use the audited HTTP client with custom response parsing.

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
      schema:
        fields: dynamic
```

**Implementation (uses audited HTTP client):**
```python
# src/elspeth/plugins/llm/openrouter.py
"""OpenRouter LLM transform - access 100+ models via single API."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from elspeth.contracts import TransformResult
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import LLMConfig, BaseLLMTransform
from elspeth.plugins.llm.templates import TemplateError


class OpenRouterConfig(LLMConfig):
    """OpenRouter-specific configuration."""
    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, description="Request timeout")


class OpenRouterLLMTransform(BaseLLMTransform):
    """LLM transform using OpenRouter API.

    OpenRouter provides access to 100+ models via a unified API.
    Uses audited HTTP client for call recording.
    """

    name = "openrouter_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        # Don't call super().__init__ - we override process() entirely
        from elspeth.plugins.base import BaseTransform
        BaseTransform.__init__(self, config)

        cfg = OpenRouterConfig.from_dict(config)
        self._model = cfg.model
        self._template_str = cfg.template
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._api_key = cfg.api_key
        self._base_url = cfg.base_url
        self._timeout = cfg.timeout_seconds

        from elspeth.plugins.llm.templates import PromptTemplate
        self._template = PromptTemplate(cfg.template)

        # Schema
        from elspeth.plugins.schema_factory import create_schema_from_config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "OpenRouterSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process row via OpenRouter API using audited HTTP client."""

        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(**row)
        except TemplateError as e:
            return TransformResult.error({
                "reason": "template_rendering_failed",
                "error": str(e),
            })

        # 2. Build request
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        request_body = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens:
            request_body["max_tokens"] = self._max_tokens

        # 3. Call via audited HTTP client (EXTERNAL - wrap)
        if ctx.http_client is None:
            raise RuntimeError("HTTP client not available in PluginContext")

        try:
            response = ctx.http_client.post(
                f"{self._base_url}/chat/completions",
                json=request_body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()

        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
            return TransformResult.error(
                {"reason": "api_call_failed", "error": str(e)},
                retryable=is_rate_limit,
            )

        # 4. Build output (OUR CODE - let crash)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash

        return TransformResult.success(output)

    def close(self) -> None:
        pass
```

---

### Task A4: Implement AzureLLMTransform (Single)

**Files:**
- Create: `src/elspeth/plugins/llm/azure.py`
- Test: `tests/plugins/llm/test_azure.py`

> **Note:** Azure OpenAI uses the same SDK as OpenAI. The audited LLM client wraps this SDK, so the transform simply uses `ctx.llm_client` (which is configured with Azure credentials by the executor).

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
      schema:
        fields: dynamic
```

**Implementation (inherits from BaseLLMTransform):**
```python
# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform - single call per row."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from elspeth.plugins.llm.base import LLMConfig, BaseLLMTransform


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI-specific configuration."""
    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="API version")


class AzureLLMTransform(BaseLLMTransform):
    """LLM transform using Azure OpenAI.

    Inherits from BaseLLMTransform - uses ctx.llm_client for calls.
    The executor configures the audited client with Azure credentials.
    """

    name = "azure_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        # Parse Azure-specific config to validate, but don't store client
        # (client is provided by executor via ctx.llm_client)
        cfg = AzureOpenAIConfig.from_dict(config)

        # Store deployment name for model parameter
        # (Azure uses deployment_name, not model, but we pass it as model to the client)
        config_with_model = dict(config)
        config_with_model["model"] = cfg.deployment_name

        super().__init__(config_with_model)

        # Store Azure-specific config for executor to use
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration for executor to create audited client."""
        return {
            "endpoint": self._azure_endpoint,
            "api_key": self._azure_api_key,
            "api_version": self._azure_api_version,
            "provider": "azure",
        }
```

> **Executor Integration:** When the executor sees an `AzureLLMTransform`, it reads `transform.azure_config` to create the audited client with Azure credentials:
>
> ```python
> # In executor, when setting up PluginContext for Azure transforms
> if hasattr(transform, 'azure_config'):
>     from openai import AzureOpenAI
>     underlying = AzureOpenAI(
>         azure_endpoint=transform.azure_config["endpoint"],
>         api_key=transform.azure_config["api_key"],
>         api_version=transform.azure_config["api_version"],
>     )
>     ctx.llm_client = AuditedLLMClient(
>         recorder, state_id, underlying, provider="azure"
>     )
> ```

---

### Task A5: Implement AzureBatchLLMTransform (Batch-Aware Transform)

**Files:**
- Create: `src/elspeth/plugins/llm/azure_batch.py`
- Create: `src/elspeth/engine/batch_scheduler.py` (new - for batch lifecycle)
- Test: `tests/plugins/llm/test_azure_batch.py`
- Test: `tests/engine/test_batch_scheduler.py`

> **Best Practice Applied:** Two-phase checkpoint approach for long-running operations:
> 1. **Submit Phase:** Submit batch, checkpoint batch_id immediately, yield control
> 2. **Complete Phase:** Check batch status, download results if complete
>
> This enables crash recovery and prevents resource blocking.

**Configuration:**
```yaml
# Pipeline config - aggregation is engine-driven
aggregations:
  - node_id: azure_batch_node
    trigger:
      count: 100                   # Fire after 100 rows
      timeout_seconds: 3600        # Or after 1 hour
    output_mode: passthrough       # N rows in → N enriched rows out

transforms:
  - plugin: azure_batch_llm
    node_id: azure_batch_node      # Links to aggregation config
    options:
      model: "gpt-4o"
      deployment_name: "my-gpt4o-deployment"
      endpoint: "${AZURE_OPENAI_ENDPOINT}"
      api_key: "${AZURE_OPENAI_KEY}"
      template: |
        Analyze: {{ text }}
      poll_interval_seconds: 300   # How often to check batch status (5 min default)
      max_wait_hours: 24           # Maximum wait time
      schema:
        fields: dynamic
```

**Architecture (Two-Phase with Checkpoint):**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-PHASE BATCH PROCESSING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: SUBMIT                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ process(rows, ctx) called by engine                                  │   │
│  │   ├── Check checkpoint: batch_id exists?                             │   │
│  │   │     └── No: Fresh batch                                          │   │
│  │   │           ├── Render all templates                               │   │
│  │   │           ├── Build JSONL                                        │   │
│  │   │           ├── Upload to Azure                                    │   │
│  │   │           ├── Submit batch job                                   │   │
│  │   │           ├── CHECKPOINT: Save batch_id immediately              │   │
│  │   │           └── Raise BatchPendingError("submitted")               │   │
│  │   │                                                                  │   │
│  └───│──────────────────────────────────────────────────────────────────┘   │
│      │                                                                       │
│      │  Engine catches BatchPendingError                                    │
│      │    └── Schedules batch check after poll_interval                     │
│      │                                                                       │
│  PHASE 2: CHECK/COMPLETE (called by scheduler)                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ process(rows, ctx) called again with same checkpoint                 │   │
│  │   ├── Check checkpoint: batch_id exists?                             │   │
│  │   │     └── Yes: Resume batch                                        │   │
│  │   │           ├── Check Azure batch status                           │   │
│  │   │           │     ├── "completed" → Download results, return       │   │
│  │   │           │     ├── "failed" → Return TransformResult.error()    │   │
│  │   │           │     └── "in_progress" → Raise BatchPendingError      │   │
│  │   │           │                         (schedule another check)     │   │
│  └───│──────────────────────────────────────────────────────────────────┘   │
│      │                                                                       │
│      └── Eventually completes or fails                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why Two-Phase?**
1. **Crash recovery** - Batch ID is checkpointed immediately after submission
2. **Resource efficiency** - Process doesn't block for hours
3. **Visibility** - Audit trail shows batch status transitions
4. **Idempotency** - Re-running process() after crash safely resumes

**Step 1: Create BatchPendingError**

```python
# src/elspeth/plugins/llm/batch_errors.py
"""Batch processing control flow errors."""

from __future__ import annotations


class BatchPendingError(Exception):
    """Raised when batch is submitted but not yet complete.

    This is NOT an error condition - it's a control flow signal
    telling the engine to schedule a retry check later.
    """

    def __init__(
        self,
        batch_id: str,
        status: str,
        *,
        check_after_seconds: int = 300,
    ) -> None:
        self.batch_id = batch_id
        self.status = status
        self.check_after_seconds = check_after_seconds
        super().__init__(f"Batch {batch_id} is {status}, check after {check_after_seconds}s")
```

**Step 2: Implement AzureBatchLLMTransform with checkpoints**

```python
# src/elspeth/plugins/llm/azure_batch.py
"""Azure OpenAI Batch API transform - 50% cost savings for high volume."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from elspeth.contracts import TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.schema_factory import create_schema_from_config
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.llm.batch_errors import BatchPendingError


class AzureBatchConfig(TransformDataConfig):
    """Azure Batch-specific configuration."""
    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="API version")
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    response_field: str = Field("llm_response", description="Output field name")
    poll_interval_seconds: int = Field(300, description="Batch status check interval")
    max_wait_hours: int = Field(24, description="Maximum batch wait time")


class AzureBatchLLMTransform(BaseTransform):
    """Batch LLM transform using Azure OpenAI Batch API.

    Uses two-phase checkpoint approach:
    1. Submit batch → checkpoint batch_id → raise BatchPendingError
    2. Check status → complete or raise BatchPendingError again

    Benefits:
    - 50% cost reduction vs real-time API
    - Crash recovery via checkpointed batch_id
    - Resource efficiency (no blocking waits)
    """

    name = "azure_batch_llm"
    is_batch_aware = True  # Engine passes list[dict] at aggregation nodes

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = AzureBatchConfig.from_dict(config)
        self._deployment_name = cfg.deployment_name
        self._endpoint = cfg.endpoint
        self._api_key = cfg.api_key
        self._api_version = cfg.api_version
        self._template = PromptTemplate(cfg.template)
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._poll_interval = cfg.poll_interval_seconds
        self._max_wait_hours = cfg.max_wait_hours

        # Schema
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "AzureBatchSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

        # Azure client (lazy init)
        self._client = None

    def _get_client(self):
        """Lazy-initialize Azure client."""
        if self._client is None:
            from openai import AzureOpenAI
            self._client = AzureOpenAI(
                azure_endpoint=self._endpoint,
                api_key=self._api_key,
                api_version=self._api_version,
            )
        return self._client

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with checkpoint-based recovery.

        May be called multiple times for the same batch:
        - First call: submits batch, checkpoints, raises BatchPendingError
        - Subsequent calls: checks status, completes or raises again
        """
        if isinstance(row, list):
            return self._process_batch(row, ctx)
        else:
            # Single row fallback - use audited LLM client
            return self._process_single(row, ctx)

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with two-phase checkpoint approach."""
        if not rows:
            return TransformResult.success_multi([])

        # Check if we're resuming an in-flight batch
        checkpoint = ctx.get_checkpoint() if hasattr(ctx, 'get_checkpoint') else None

        if checkpoint and checkpoint.get("batch_id"):
            # PHASE 2: Resume - check status of existing batch
            return self._check_batch_status(
                batch_id=checkpoint["batch_id"],
                rows=rows,
                rendered_prompts=checkpoint.get("rendered_prompts", []),
                ctx=ctx,
            )
        else:
            # PHASE 1: Fresh batch - submit and checkpoint
            return self._submit_batch(rows, ctx)

    def _submit_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Submit batch to Azure, checkpoint, and yield control."""

        # 1. Render templates for all rows (THEIR DATA - wrap)
        rendered_prompts = []
        for i, row in enumerate(rows):
            try:
                rendered = self._template.render_with_metadata(**row)
                rendered_prompts.append({
                    "custom_id": f"row_{i}",
                    "prompt": rendered.prompt,
                    "template_hash": rendered.template_hash,
                })
            except TemplateError as e:
                # Template error for one row - mark it failed, continue others
                rendered_prompts.append({
                    "custom_id": f"row_{i}",
                    "error": str(e),
                })

        # 2. Build JSONL for successful renders only
        jsonl_lines = []
        for i, req in enumerate(rendered_prompts):
            if "error" in req:
                continue  # Skip failed renders
            jsonl_lines.append(json.dumps({
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._deployment_name,
                    "messages": [
                        {"role": "system", "content": self._system_prompt or ""},
                        {"role": "user", "content": req["prompt"]},
                    ],
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens,
                },
            }))

        if not jsonl_lines:
            # All rows failed template rendering
            return TransformResult.error({
                "reason": "all_rows_failed_template_rendering",
                "row_count": len(rows),
            })

        jsonl_content = "\n".join(jsonl_lines)

        # 3. Upload and submit to Azure (EXTERNAL - wrap)
        try:
            client = self._get_client()

            file_response = client.files.create(
                file=("batch_input.jsonl", jsonl_content.encode()),
                purpose="batch",
            )

            batch_response = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/chat/completions",
                completion_window="24h",
            )

            batch_id = batch_response.id

        except Exception as e:
            return TransformResult.error({
                "reason": "batch_submission_failed",
                "error": str(e),
            })

        # 4. CHECKPOINT IMMEDIATELY after successful submission
        if hasattr(ctx, 'update_checkpoint'):
            ctx.update_checkpoint({
                "batch_id": batch_id,
                "file_id": file_response.id,
                "status": "submitted",
                "row_count": len(rows),
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "rendered_prompts": rendered_prompts,  # For result mapping
            })

        # 5. Raise to signal "come back later"
        raise BatchPendingError(
            batch_id=batch_id,
            status="submitted",
            check_after_seconds=self._poll_interval,
        )

    def _check_batch_status(
        self,
        batch_id: str,
        rows: list[dict[str, Any]],
        rendered_prompts: list[dict],
        ctx: PluginContext,
    ) -> TransformResult:
        """Check batch status and complete if ready."""

        try:
            client = self._get_client()
            batch = client.batches.retrieve(batch_id)

        except Exception as e:
            return TransformResult.error({
                "reason": "batch_status_check_failed",
                "batch_id": batch_id,
                "error": str(e),
            })

        if batch.status == "completed":
            # SUCCESS - download and return results
            return self._download_and_return_results(
                batch_id=batch_id,
                output_file_id=batch.output_file_id,
                rows=rows,
                rendered_prompts=rendered_prompts,
                ctx=ctx,
            )

        elif batch.status in ("failed", "expired", "cancelled"):
            # FAILED - return error
            return TransformResult.error({
                "reason": "batch_failed",
                "batch_id": batch_id,
                "status": batch.status,
            })

        else:
            # STILL IN PROGRESS - update checkpoint and yield
            if hasattr(ctx, 'update_checkpoint'):
                ctx.update_checkpoint({
                    "batch_id": batch_id,
                    "status": batch.status,
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "rendered_prompts": rendered_prompts,
                })

            raise BatchPendingError(
                batch_id=batch_id,
                status=batch.status,
                check_after_seconds=self._poll_interval,
            )

    def _download_and_return_results(
        self,
        batch_id: str,
        output_file_id: str,
        rows: list[dict[str, Any]],
        rendered_prompts: list[dict],
        ctx: PluginContext,
    ) -> TransformResult:
        """Download batch results and map back to input rows."""

        try:
            client = self._get_client()
            output_file = client.files.content(output_file_id)
            results = self._parse_batch_results(output_file.text)

        except Exception as e:
            return TransformResult.error({
                "reason": "batch_result_download_failed",
                "batch_id": batch_id,
                "error": str(e),
            })

        # Map results back to input rows (passthrough: N in → N out)
        output_results = []
        for i, row in enumerate(rows):
            custom_id = f"row_{i}"
            output_row = dict(row)

            # Check if this row had a template error
            if i < len(rendered_prompts) and "error" in rendered_prompts[i]:
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = rendered_prompts[i]["error"]
            elif custom_id in results:
                output_row[self._response_field] = results[custom_id]
            else:
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = "No response in batch"

            # Add batch metadata for audit trail correlation
            output_row[f"{self._response_field}_batch_id"] = batch_id
            output_results.append(output_row)

        # Clear checkpoint on success
        if hasattr(ctx, 'clear_checkpoint'):
            ctx.clear_checkpoint()

        return TransformResult.success_multi(output_results)

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

    def _process_single(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Fallback for single row - use audited LLM client."""

        # Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(**row)
        except TemplateError as e:
            return TransformResult.error({
                "reason": "template_rendering_failed",
                "error": str(e),
            })

        # Build messages
        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # Call via audited client (EXTERNAL - wrap)
        if ctx.llm_client is None:
            raise RuntimeError("LLM client not available in PluginContext")

        from elspeth.plugins.clients.llm import LLMClientError, RateLimitError

        try:
            response = ctx.llm_client.chat_completion(
                model=self._deployment_name,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # Build output
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage
        output[f"{self._response_field}_note"] = "single_row_fallback"

        return TransformResult.success(output)

    def close(self) -> None:
        """Release resources."""
        self._client = None
```

**Step 3: Engine support for BatchPendingError**

The engine must catch `BatchPendingError` and schedule retries:

```python
# In src/elspeth/engine/processor.py (modification)

from elspeth.plugins.llm.batch_errors import BatchPendingError

def _process_aggregation_batch(self, ...):
    """Process aggregation batch."""
    try:
        result = transform.process(rows, ctx)
        # Normal completion
        return result

    except BatchPendingError as e:
        # Batch submitted but not complete - schedule retry
        self._scheduler.schedule_batch_check(
            node_id=transform.node_id,
            batch_id=e.batch_id,
            check_after=timedelta(seconds=e.check_after_seconds),
        )
        # Don't mark as failed - batch is in progress
        return None  # Or a special "pending" result
```

**Step 4: Tests**

```python
# tests/plugins/llm/test_azure_batch.py

class TestAzureBatchCheckpointRecovery:
    """Tests for checkpoint-based crash recovery."""

    def test_fresh_batch_submits_and_raises_pending(self):
        """Fresh batch submits to Azure and raises BatchPendingError."""
        transform = AzureBatchLLMTransform({...})
        ctx = MockPluginContext()

        with pytest.raises(BatchPendingError) as exc_info:
            transform.process([{"text": "hello"}], ctx)

        assert exc_info.value.status == "submitted"
        assert ctx.checkpoint["batch_id"] is not None

    def test_resume_with_checkpoint_checks_status(self):
        """Resume with checkpoint checks Azure batch status."""
        transform = AzureBatchLLMTransform({...})
        ctx = MockPluginContext()
        ctx.checkpoint = {"batch_id": "batch_123", "status": "in_progress"}

        # Mock Azure to return "in_progress"
        with pytest.raises(BatchPendingError) as exc_info:
            transform.process([{"text": "hello"}], ctx)

        assert exc_info.value.batch_id == "batch_123"

    def test_completed_batch_returns_results(self):
        """Completed batch downloads and returns results."""
        transform = AzureBatchLLMTransform({...})
        ctx = MockPluginContext()
        ctx.checkpoint = {"batch_id": "batch_123", "rendered_prompts": [...]}

        # Mock Azure to return "completed" with results
        result = transform.process([{"text": "hello"}], ctx)

        assert result.status == "success"
        assert result.rows[0]["llm_response"] is not None
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

### Task C5: Audited Client Infrastructure

**Files:**
- Create: `src/elspeth/plugins/clients/__init__.py`
- Create: `src/elspeth/plugins/clients/base.py`
- Create: `src/elspeth/plugins/clients/llm.py`
- Create: `src/elspeth/plugins/clients/http.py`
- Create: `tests/plugins/clients/test_audited_llm_client.py`
- Modify: `src/elspeth/plugins/context.py` (add client fields)
- Modify: `src/elspeth/engine/processor.py` (create clients with state_id)

**Step 1: Create base audited client**

```python
# src/elspeth/plugins/clients/base.py
"""Base class for audited clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AuditedClientBase(ABC):
    """Base class for clients that automatically record to audit trail.

    Subclasses wrap specific client types (LLM, HTTP, etc.) and ensure
    all calls are recorded with proper state linkage.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
    ) -> None:
        """Initialize audited client.

        Args:
            recorder: Landscape recorder for audit trail
            state_id: The node_state this client is attached to
        """
        self._recorder = recorder
        self._state_id = state_id
        self._call_index = 0

    def _next_call_index(self) -> int:
        """Get and increment call index."""
        idx = self._call_index
        self._call_index += 1
        return idx
```

**Step 2: Create AuditedLLMClient**

```python
# src/elspeth/plugins/clients/llm.py
"""Audited LLM client with automatic call recording."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.base import AuditedClientBase

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


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


class LLMClientError(Exception):
    """Error from LLM client."""

    def __init__(self, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class RateLimitError(LLMClientError):
    """Rate limit exceeded - retryable."""

    def __init__(self, message: str):
        super().__init__(message, retryable=True)


class AuditedLLMClient(AuditedClientBase):
    """LLM client that automatically records all calls to audit trail.

    This client wraps the underlying LLM SDK and ensures every call
    (successful or failed) is recorded with proper state linkage.

    Usage:
        # Executor creates client with state_id
        client = AuditedLLMClient(recorder, state_id, openai_client)

        # Transform uses client - recording is automatic
        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        underlying_client: Any,  # openai.OpenAI or openai.AzureOpenAI
        *,
        provider: str = "openai",
    ) -> None:
        super().__init__(recorder, state_id)
        self._client = underlying_client
        self._provider = provider

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make chat completion call with automatic audit recording.

        Args:
            model: Model identifier or deployment name
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with content and metadata

        Raises:
            LLMClientError: On API error (check .retryable)
            RateLimitError: On rate limit (always retryable)
        """
        call_index = self._next_call_index()

        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "provider": self._provider,
            **kwargs,
        }

        start = time.perf_counter()

        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            # Extract response data
            content = response.choices[0].message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

            # Record successful call
            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data={
                    "content": content,
                    "model": response.model,
                    "usage": usage,
                },
                latency_ms=latency_ms,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            # Determine if retryable
            error_type = type(e).__name__
            is_rate_limit = "rate" in str(e).lower() or "429" in str(e)

            # Record failed call
            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data=request_data,
                error={
                    "type": error_type,
                    "message": str(e),
                    "retryable": is_rate_limit,
                },
                latency_ms=latency_ms,
            )

            if is_rate_limit:
                raise RateLimitError(str(e)) from e
            raise LLMClientError(str(e), retryable=False) from e
```

**Step 3: Create AuditedHTTPClient**

```python
# src/elspeth/plugins/clients/http.py
"""Audited HTTP client with automatic call recording."""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import httpx

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.base import AuditedClientBase

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class AuditedHTTPClient(AuditedClientBase):
    """HTTP client that automatically records all calls to audit trail.

    Wraps httpx and records request/response for every call.
    """

    def __init__(
        self,
        recorder: LandscapeRecorder,
        state_id: str,
        *,
        timeout: float = 30.0,
        base_url: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(recorder, state_id)
        self._timeout = timeout
        self._base_url = base_url
        self._default_headers = headers or {}

    def post(
        self,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Make POST request with automatic audit recording."""
        call_index = self._next_call_index()

        full_url = f"{self._base_url}{url}" if self._base_url else url
        merged_headers = {**self._default_headers, **(headers or {})}

        request_data = {
            "method": "POST",
            "url": full_url,
            "json": json,
            # Don't record auth headers
            "headers": {k: v for k, v in merged_headers.items()
                       if "auth" not in k.lower() and "key" not in k.lower()},
        }

        start = time.perf_counter()

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    full_url,
                    json=json,
                    headers=merged_headers,
                )

            latency_ms = (time.perf_counter() - start) * 1000

            # Record call (success even for 4xx/5xx - those are valid HTTP responses)
            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    # Don't store full body - just size
                    "body_size": len(response.content),
                },
                latency_ms=latency_ms,
            )

            return response

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000

            self._recorder.record_call(
                state_id=self._state_id,
                call_index=call_index,
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=request_data,
                error={
                    "type": type(e).__name__,
                    "message": str(e),
                },
                latency_ms=latency_ms,
            )
            raise
```

**Step 4: Update PluginContext**

Add optional client fields to PluginContext:

```python
# In src/elspeth/plugins/context.py

@dataclass
class PluginContext:
    """Context provided to plugins during execution."""

    # ... existing fields ...

    # Audited clients (set by executor when state is created)
    llm_client: AuditedLLMClient | None = None
    http_client: AuditedHTTPClient | None = None
```

**Step 5: Update RowProcessor to create clients**

```python
# In src/elspeth/engine/processor.py, when creating PluginContext for transforms:

def _create_plugin_context(
    self,
    state_id: str,
    node_id: str,
    # ... other params ...
) -> PluginContext:
    """Create PluginContext with audited clients."""

    # Create audited LLM client if LLM config exists
    llm_client = None
    if self._llm_config:
        underlying = self._create_llm_client(self._llm_config)
        llm_client = AuditedLLMClient(
            recorder=self._recorder,
            state_id=state_id,
            underlying_client=underlying,
            provider=self._llm_config.provider,
        )

    # Create audited HTTP client
    http_client = AuditedHTTPClient(
        recorder=self._recorder,
        state_id=state_id,
    )

    return PluginContext(
        # ... existing fields ...
        llm_client=llm_client,
        http_client=http_client,
    )
```

**Step 6: Write tests**

```python
# tests/plugins/clients/test_audited_llm_client.py
"""Tests for AuditedLLMClient."""

import pytest
from unittest.mock import Mock, MagicMock

from elspeth.contracts import CallStatus, CallType
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError, RateLimitError


class TestAuditedLLMClient:
    """Tests for automatic call recording."""

    @pytest.fixture
    def mock_recorder(self):
        """Create mock recorder."""
        return Mock()

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        client = Mock()
        response = Mock()
        response.choices = [Mock(message=Mock(content="Hello!"))]
        response.model = "gpt-4"
        response.usage = Mock(prompt_tokens=10, completion_tokens=5)
        response.model_dump = Mock(return_value={})
        client.chat.completions.create.return_value = response
        return client

    def test_successful_call_records_to_audit_trail(
        self, mock_recorder, mock_openai_client
    ):
        """Successful call is recorded with request and response."""
        client = AuditedLLMClient(
            mock_recorder,
            state_id="state_123",
            underlying_client=mock_openai_client,
        )

        response = client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )

        assert response.content == "Hello!"
        mock_recorder.record_call.assert_called_once()
        call_args = mock_recorder.record_call.call_args
        assert call_args.kwargs["state_id"] == "state_123"
        assert call_args.kwargs["call_index"] == 0
        assert call_args.kwargs["call_type"] == CallType.LLM
        assert call_args.kwargs["status"] == CallStatus.SUCCESS

    def test_failed_call_records_error(self, mock_recorder, mock_openai_client):
        """Failed call is recorded with error details."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")

        client = AuditedLLMClient(
            mock_recorder,
            state_id="state_123",
            underlying_client=mock_openai_client,
        )

        with pytest.raises(LLMClientError):
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )

        call_args = mock_recorder.record_call.call_args
        assert call_args.kwargs["status"] == CallStatus.ERROR
        assert "API Error" in call_args.kwargs["error"]["message"]

    def test_rate_limit_is_retryable(self, mock_recorder, mock_openai_client):
        """Rate limit errors are marked retryable."""
        mock_openai_client.chat.completions.create.side_effect = Exception(
            "Rate limit exceeded (429)"
        )

        client = AuditedLLMClient(
            mock_recorder,
            state_id="state_123",
            underlying_client=mock_openai_client,
        )

        with pytest.raises(RateLimitError) as exc_info:
            client.chat_completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert exc_info.value.retryable is True

    def test_call_index_increments(self, mock_recorder, mock_openai_client):
        """Each call gets unique call_index."""
        client = AuditedLLMClient(
            mock_recorder,
            state_id="state_123",
            underlying_client=mock_openai_client,
        )

        client.chat_completion(model="gpt-4", messages=[])
        client.chat_completion(model="gpt-4", messages=[])
        client.chat_completion(model="gpt-4", messages=[])

        calls = mock_recorder.record_call.call_args_list
        assert calls[0].kwargs["call_index"] == 0
        assert calls[1].kwargs["call_index"] == 1
        assert calls[2].kwargs["call_index"] == 2
```

**Step 7: Commit**

```bash
git add src/elspeth/plugins/clients/ tests/plugins/clients/
git commit -m "$(cat <<'EOF'
feat(plugins): add audited client infrastructure for external calls

Infrastructure-wrapped clients that automatically record to audit trail:
- AuditedLLMClient: Wraps OpenAI/Azure SDK
- AuditedHTTPClient: Wraps httpx
- Recording is guaranteed by construction - plugins cannot bypass

This follows ELSPETH's "audit is non-negotiable" principle.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Summary

| Part | Tasks | Effort | Dependencies |
|------|-------|--------|--------------|
| **A: LLM Plugins** | A1-A6 | 3-4 days | Jinja2, httpx, openai SDK |
| **B: Azure Storage** | B1-B3 | 2 days | azure-storage-blob, azure-identity |
| **C: Call Infrastructure** | C1-C5 | 3-4 days | DeepDiff, PayloadStore |

### Key Architectural Decisions

| Decision | Pattern | Rationale |
|----------|---------|-----------|
| External call recording | Audited Clients | Audit guaranteed by construction |
| Config validation | Extend TransformDataConfig | Schema + on_error support |
| Error handling | Three-Tier Trust Model | Wrap THEIR DATA + EXTERNAL, let OUR CODE crash |
| Long-running batches | Two-phase checkpoint | Crash recovery + resource efficiency |

### Recommended Implementation Order

> **Critical dependency:** LLM transforms (A2-A5) use audited clients (C5) which require C1 (CallRecorder). The order below ensures no broken dependencies.

```
Phase 1: Foundation (can run in parallel)
├── A1: PromptTemplate (no dependencies)
├── C1: CallRecorder (LandscapeRecorder.record_call)
└── B1-B2: Azure Storage (independent)

Phase 2: Audited Infrastructure
└── C5: Audited Clients (depends on C1)
    ├── AuditedLLMClient
    ├── AuditedHTTPClient
    └── PluginContext integration

Phase 3: LLM Transforms (sequential, each builds on previous)
├── A2: BaseLLMTransform (depends on C5)
├── A3: OpenRouterLLMTransform (depends on A2, uses AuditedHTTPClient)
├── A4: AzureLLMTransform (depends on A2, uses AuditedLLMClient)
└── A5: AzureBatchLLMTransform (depends on A2, adds checkpoint support)

Phase 4: Testing & Polish
├── A6: Integration tests
└── C2-C4: Replay/Verify (nice-to-have)
```

**Detailed order:**

1. **A1** (PromptTemplate) - no dependencies, foundation for all LLM plugins
2. **C1** (CallRecorder) - `LandscapeRecorder.record_call()` method
3. **B1-B2** (Azure Storage) - can run in parallel with C1
4. **C5** (Audited Clients) - `AuditedLLMClient` + `AuditedHTTPClient` ⚠️ **BEFORE A2**
5. **A2** (BaseLLMTransform) - shared base class with audited client usage
6. **A3** (OpenRouter) - quickest to integration test (uses HTTP client)
7. **A4** (Azure Single) - validates Azure integration (uses LLM client)
8. **A5** (Azure Batch) - two-phase checkpoint for long-running batches
9. **C2-C4** (Replay/Verify) - optional, for deterministic testing
10. **A6** (Integration tests) - validates full flow

### Best Practices Applied

| Issue | Solution | Where Applied |
|-------|----------|---------------|
| Audit gaps from forgotten recording | Audited Clients | C5 → A2-A5 |
| Missing schema validation | TransformDataConfig | A2 config |
| Silent errors | Three-Tier Trust error handling | A2-A5 process() |
| Crash loss of batch progress | Two-phase checkpoint | A5 |
| Blocking waits on batches | BatchPendingError control flow | A5 |

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

---

## Appendix: Additional Test Coverage Recommendations

Beyond the unit tests specified in each task, consider these additional test scenarios:

### Template Engine Edge Cases

```python
# tests/plugins/llm/test_templates_edge_cases.py

def test_template_hash_reproducible_across_process_restarts():
    """Same template produces same hash even after restart.

    Critical for audit trail integrity - hash must be deterministic.
    """
    template_str = "Hello, {{ name }}!"
    t1 = PromptTemplate(template_str)
    hash1 = t1.template_hash

    # Simulate restart by creating fresh instance
    t2 = PromptTemplate(template_str)
    hash2 = t2.template_hash

    assert hash1 == hash2, "Hash must be reproducible for audit trail"


def test_template_with_unicode_characters():
    """Templates with emoji, CJK, RTL text work correctly."""
    template = PromptTemplate("Analyze: {{ text }} 🎉")
    result = template.render(text="日本語テスト")
    assert "日本語テスト" in result
    assert "🎉" in result


def test_template_with_large_variables():
    """Templates handle large variable payloads."""
    template = PromptTemplate("Analyze: {{ text }}")
    large_text = "x" * 100_000  # 100KB
    result = template.render_with_metadata(text=large_text)
    assert len(result.prompt) > 100_000


def test_variables_hash_ignores_field_order():
    """Variables hash is stable regardless of dict key order."""
    template = PromptTemplate("{{ a }} {{ b }}")
    r1 = template.render_with_metadata(a="x", b="y")
    r2 = template.render_with_metadata(b="y", a="x")
    assert r1.variables_hash == r2.variables_hash, "Canonical JSON must normalize order"
```

### LLM Transform Error Boundaries

```python
# tests/plugins/llm/test_error_boundaries.py

def test_openrouter_timeout_records_partial_call():
    """Timeout mid-request still records attempt in audit trail."""
    # Mock httpx to timeout after connection but before response
    # Verify ctx.record_external_call was called with error status


def test_azure_batch_partial_failure_handling():
    """Batch with some failed rows returns all results."""
    # Mock Azure batch to fail on 2 of 5 rows
    # Verify success_multi returns all 5 rows with appropriate error markers


def test_llm_response_with_invalid_json():
    """LLM returns malformed JSON in response field."""
    # Some LLMs return JSON-ish but invalid responses
    # Verify graceful handling


def test_api_rate_limit_error_is_retryable():
    """429 errors are marked retryable in TransformResult."""
    # Mock 429 response
    # Verify TransformResult.error(retryable=True)
```

### Audit Trail Integrity

```python
# tests/integration/test_audit_integrity.py

def test_all_hashes_are_64_char_hex():
    """All hash fields are valid SHA-256 hex strings."""
    # Run a transform, query landscape
    # Verify template_hash, variables_hash, rendered_hash, content_hash
    # are all 64-character lowercase hex


def test_batch_results_traceable_to_inputs():
    """Each batch output row traces back to its input."""
    # Run batch transform with 5 rows
    # Verify each output has batch_id that exists in calls table
    # Verify row count matches
```

### Trust Tier Compliance

```python
# tests/plugins/llm/test_trust_tiers.py

def test_transform_does_not_coerce_row_types():
    """LLM transform fails on wrong input types (no coercion)."""
    # Pass row with str where int expected
    # Verify pipeline crashes (Tier 2: transforms don't coerce)


def test_llm_response_treated_as_external_data():
    """LLM response content is not trusted for type operations."""
    # LLM returns "42" when we expected int
    # Downstream should handle this as external data
```
