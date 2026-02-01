# Content Filtering Transforms Design

> **Status:** APPROVED
> **Date:** 2026-01-20
> **Author:** Claude + John

## Overview

Three new transform plugins for content filtering in ELSPETH pipelines:

| Plugin | Purpose | Determinism |
|--------|---------|-------------|
| `keyword_filter` | Block content matching regex patterns (PII, secrets) | `DETERMINISTIC` |
| `azure_content_safety` | Block content failing Azure moderation (hate, violence, sexual, self-harm) | `EXTERNAL_CALL` |
| `azure_prompt_shield` | Block jailbreak attempts and indirect prompt injection | `EXTERNAL_CALL` |

## Design Principles

### Pass/Fail Semantics

These are **filters**, not mutators:
- **Pass** → `TransformResult.success(row)` — row continues unchanged
- **Fail** → `TransformResult.error(reason)` — row routed to `on_error` sink

### Field Configuration

All three transforms share identical field configuration:
- `fields` parameter: list of field names OR `"all"`
- **Required** — no defaults allowed
- `"all"` scans all string-typed fields (per schema, or runtime string values if dynamic)

### Schema Enforcement

Per ELSPETH's three-tier trust model:
- Data reaching transforms has passed source validation
- Missing or wrong-type fields = **upstream bug** → crash
- Only **operations on values** (regex matching, API calls) are wrapped

---

## Plugin 1: Keyword Filter

### Configuration

```yaml
transforms:
  - plugin: keyword_filter
    options:
      fields: [message, subject]  # or "all" - REQUIRED
      blocked_patterns:           # REQUIRED, regex patterns
        - "\\bpassword\\b"
        - "\\bssn\\b"
        - "api[_-]?key"
        - "(?i)confidential"
      on_error: quarantine_sink
      schema:
        fields: dynamic
```

### Config Class

```python
class KeywordFilterConfig(TransformDataConfig):
    fields: str | list[str] = Field(
        ...,  # Required
        description="Field name(s) to scan, or 'all' for all string fields",
    )
    blocked_patterns: list[str] = Field(
        ...,  # Required
        description="Regex patterns that trigger blocking",
    )
```

### Error Result

```python
{
    "reason": "blocked_content",
    "field": "message",
    "matched_pattern": "\\bssn\\b",
    "match_context": "...please send your ssn to verify..."
}
```

Context: 40 characters before and after match, with `...` truncation.

### Attributes

```python
class KeywordFilter(BaseTransform):
    name = "keyword_filter"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    is_batch_aware = False
    creates_tokens = False
```

---

## Plugin 2: Azure Content Safety

### Configuration

```yaml
transforms:
  - plugin: azure_content_safety
    options:
      endpoint: ${AZURE_CONTENT_SAFETY_ENDPOINT}  # Env var REQUIRED
      api_key: ${AZURE_CONTENT_SAFETY_KEY}        # Env var REQUIRED
      fields: [user_input, prompt]                # or "all" - REQUIRED
      thresholds:                                 # REQUIRED
        hate: 2
        violence: 4
        sexual: 2
        self_harm: 0
      on_error: blocked_content_sink
      schema:
        fields: dynamic
```

### Config Class

```python
class ContentSafetyThresholds(BaseModel):
    hate: int = Field(..., ge=0, le=6)
    violence: int = Field(..., ge=0, le=6)
    sexual: int = Field(..., ge=0, le=6)
    self_harm: int = Field(..., ge=0, le=6)

class AzureContentSafetyConfig(TransformDataConfig):
    endpoint: str = Field(..., description="Azure Content Safety endpoint URL")
    api_key: str = Field(..., description="Azure Content Safety API key")
    fields: str | list[str] = Field(..., description="Field(s) to analyze")
    thresholds: ContentSafetyThresholds = Field(
        ...,
        description="Per-category severity thresholds (0-6)"
    )
```

### Error Result

```python
{
    "reason": "content_safety_violation",
    "field": "user_input",
    "categories": {
        "hate": {"severity": 4, "threshold": 2, "exceeded": True},
        "violence": {"severity": 0, "threshold": 4, "exceeded": False},
        "sexual": {"severity": 0, "threshold": 2, "exceeded": False},
        "self_harm": {"severity": 0, "threshold": 0, "exceeded": False}
    }
}
```

### API Details

- **Endpoint:** `POST {endpoint}/contentsafety/text:analyze?api-version=2024-09-01`
- **Auth:** `Ocp-Apim-Subscription-Key` header
- **Client:** `ctx.http_client` (audited)

### Attributes

```python
class AzureContentSafety(BaseTransform):
    name = "azure_content_safety"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    is_batch_aware = False
    creates_tokens = False
```

---

## Plugin 3: Azure Prompt Shield

### Configuration

```yaml
transforms:
  - plugin: azure_prompt_shield
    options:
      endpoint: ${AZURE_CONTENT_SAFETY_ENDPOINT}
      api_key: ${AZURE_CONTENT_SAFETY_KEY}
      fields: [user_input]  # or "all" - REQUIRED
      on_error: prompt_injection_sink
      schema:
        fields: dynamic
```

### Config Class

```python
class AzurePromptShieldConfig(TransformDataConfig):
    endpoint: str = Field(..., description="Azure Content Safety endpoint URL")
    api_key: str = Field(..., description="Azure Content Safety API key")
    fields: str | list[str] = Field(..., description="Field(s) to analyze")
    # No thresholds - binary detection
```

### Error Result

```python
{
    "reason": "prompt_injection_detected",
    "field": "user_input",
    "attacks": {
        "user_prompt_attack": True,
        "document_attack": False
    }
}
```

### API Details

- **Endpoint:** `POST {endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01`
- **Auth:** `Ocp-Apim-Subscription-Key` header
- **Detection:** Always checks both user prompt attacks AND document attacks

### Attributes

```python
class AzurePromptShield(BaseTransform):
    name = "azure_prompt_shield"
    determinism = Determinism.EXTERNAL_CALL
    plugin_version = "1.0.0"
    is_batch_aware = False
    creates_tokens = False
```

---

## Error Handling

### Three-Tier Trust Model Applied

| Scenario | Tier | Handling |
|----------|------|----------|
| Field missing from row | Our Code (upstream bug) | **Crash** |
| Field is not a string | Our Code (upstream bug) | **Crash** |
| Regex match fails content | Their Data (value) | `TransformResult.error()` |
| Azure API 4xx/5xx | External System | `TransformResult.error()` |
| Azure API 429 rate limit | External System | `TransformResult.error(retryable=True)` |
| Network timeout | External System | `TransformResult.error(retryable=True)` |

### API Error Result Structure

```python
{
    "reason": "api_error",
    "error_type": "rate_limited",  # or "http_error", "network_error"
    "status_code": 429,
    "message": "Rate limit exceeded",
    "retryable": True
}
```

---

## File Structure

```
src/elspeth/plugins/transforms/
├── keyword_filter.py
└── azure/
    ├── __init__.py
    ├── content_safety.py
    └── prompt_shield.py
```

---

## Testing Strategy

### Contract Tests

Each transform inherits from `TransformContractPropertyTestBase`:

```python
class TestKeywordFilterContract(TransformContractPropertyTestBase):
    @pytest.fixture
    def transform(self) -> TransformProtocol:
        return KeywordFilter({
            "fields": ["content"],
            "blocked_patterns": ["\\btest\\b"],
            "schema": {"fields": "dynamic"},
        })

    @pytest.fixture
    def valid_input(self) -> dict:
        return {"content": "safe message"}
```

### Plugin-Specific Tests

| Plugin | Test Cases |
|--------|------------|
| `keyword_filter` | Pattern match, no match passes, context extraction, multiple fields, `all` mode, regex edge cases |
| `azure_content_safety` | Threshold exceeded/not per category, API errors, rate limit retry |
| `azure_prompt_shield` | User attack, document attack, both clean, API errors |

### Mocked HTTP Client

Azure transforms use mocked `ctx.http_client`:

```python
@pytest.fixture
def mock_http_client():
    client = Mock(spec=AuditedHTTPClient)
    client.post.return_value = Mock(
        status_code=200,
        json=lambda: {"categoriesAnalysis": [...]}
    )
    return client
```

No integration tests against real Azure APIs in test suite.

---

## Registration

Add to `hookimpl.py`:

```python
from elspeth.plugins.transforms.keyword_filter import KeywordFilter
from elspeth.plugins.transforms.azure.content_safety import AzureContentSafety
from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShield

# In elspeth_get_transforms():
return [..., KeywordFilter, AzureContentSafety, AzurePromptShield]
```

Add to `cli.py` registries.
