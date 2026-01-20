# Template Files and Lookup Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable LLM templates to be loaded from external files with YAML-based lookup tables for two-dimensional data access.

**Architecture:** Config-time file expansion in `config.py` resolves and loads template/lookup files before plugin instantiation. Templates use explicit `row.*` and `lookup.*` namespaces. All audit metadata (hashes, source paths) flows through to output rows.

**Tech Stack:** Jinja2 (existing), PyYAML (existing), Pydantic validation

---

## Task 1: Update RenderedPrompt Dataclass

**Files:**
- Modify: `src/elspeth/plugins/llm/templates.py:21-29`
- Test: `tests/plugins/llm/test_templates.py`

**Step 1: Write the failing test**

Add to `tests/plugins/llm/test_templates.py`:

```python
def test_rendered_prompt_includes_source_metadata(self) -> None:
    """RenderedPrompt includes template and lookup source paths."""
    template = PromptTemplate(
        "Hello, {{ row.name }}!",
        template_source="prompts/greeting.j2",
        lookup_data={"greetings": ["Hi", "Hello"]},
        lookup_source="prompts/lookups.yaml",
    )
    result = template.render_with_metadata({"name": "World"})

    assert result.template_source == "prompts/greeting.j2"
    assert result.lookup_hash is not None
    assert result.lookup_source == "prompts/lookups.yaml"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_rendered_prompt_includes_source_metadata -v`

Expected: FAIL with `TypeError: PromptTemplate.__init__() got an unexpected keyword argument 'template_source'`

**Step 3: Update RenderedPrompt dataclass**

In `src/elspeth/plugins/llm/templates.py`, update the dataclass:

```python
@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered prompt with audit metadata."""

    prompt: str
    template_hash: str
    variables_hash: str
    rendered_hash: str
    # New fields for file-based templates
    template_source: str | None = None   # File path or None if inline
    lookup_hash: str | None = None       # Hash of lookup data or None
    lookup_source: str | None = None     # File path or None
```

**Step 4: Run test to verify it still fails (needs PromptTemplate changes)**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_rendered_prompt_includes_source_metadata -v`

Expected: Still FAIL (PromptTemplate doesn't accept new params yet)

**Step 5: Commit partial progress**

```bash
git add src/elspeth/plugins/llm/templates.py
git commit -m "feat(templates): add source metadata fields to RenderedPrompt"
```

---

## Task 2: Update PromptTemplate Constructor

**Files:**
- Modify: `src/elspeth/plugins/llm/templates.py:61-82`
- Test: `tests/plugins/llm/test_templates.py`

**Step 1: Update PromptTemplate.__init__ to accept new parameters**

```python
def __init__(
    self,
    template_string: str,
    *,
    template_source: str | None = None,
    lookup_data: dict[str, Any] | None = None,
    lookup_source: str | None = None,
) -> None:
    """Initialize template.

    Args:
        template_string: Jinja2 template string
        template_source: File path for audit (None if inline)
        lookup_data: Static lookup data from YAML file
        lookup_source: Lookup file path for audit (None if no lookup)

    Raises:
        TemplateError: If template syntax is invalid
    """
    self._template_string = template_string
    self._template_hash = _sha256(template_string)
    self._template_source = template_source

    # Lookup data for two-dimensional lookups
    self._lookup_data = lookup_data or {}
    self._lookup_source = lookup_source
    self._lookup_hash = (
        _sha256(canonical_json(lookup_data)) if lookup_data else None
    )

    # Use sandboxed environment for security
    self._env = SandboxedEnvironment(
        undefined=StrictUndefined,  # Raise on undefined variables
        autoescape=False,  # No HTML escaping for prompts
    )

    try:
        self._template = self._env.from_string(template_string)
    except TemplateSyntaxError as e:
        raise TemplateError(f"Invalid template syntax: {e}") from e
```

**Step 2: Add properties for new fields**

After the existing `template_hash` property:

```python
@property
def template_source(self) -> str | None:
    """File path if loaded from file, None if inline."""
    return self._template_source

@property
def lookup_hash(self) -> str | None:
    """SHA-256 hash of canonical JSON lookup data, or None."""
    return self._lookup_hash

@property
def lookup_source(self) -> str | None:
    """File path for lookup data, or None."""
    return self._lookup_source
```

**Step 3: Run test to verify it still fails (render method not updated)**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_rendered_prompt_includes_source_metadata -v`

Expected: Still FAIL (render_with_metadata signature/behavior not updated)

**Step 4: Commit partial progress**

```bash
git add src/elspeth/plugins/llm/templates.py
git commit -m "feat(templates): add lookup and source params to PromptTemplate"
```

---

## Task 3: Update render_with_metadata Method

**Files:**
- Modify: `src/elspeth/plugins/llm/templates.py:110-133`
- Test: `tests/plugins/llm/test_templates.py`

**Step 1: Update render method to use row namespace**

Replace the existing `render` method:

```python
def render(self, row: dict[str, Any] | None = None, **variables: Any) -> str:
    """Render template with variables.

    Args:
        row: Row data (accessed as row.* in template)
        **variables: Additional template variables (for backward compat)

    Returns:
        Rendered prompt string

    Raises:
        TemplateError: If rendering fails (undefined variable, sandbox violation, etc.)
    """
    # Build context with namespaced data
    context: dict[str, Any] = {}
    if row is not None:
        context["row"] = row
    context["lookup"] = self._lookup_data
    # Merge any extra variables for backward compatibility
    context.update(variables)

    try:
        return self._template.render(**context)
    except UndefinedError as e:
        raise TemplateError(f"Undefined variable: {e}") from e
    except SecurityError as e:
        raise TemplateError(f"Sandbox violation: {e}") from e
    except Exception as e:
        raise TemplateError(f"Template rendering failed: {e}") from e
```

**Step 2: Update render_with_metadata to use new signature**

```python
def render_with_metadata(self, row: dict[str, Any]) -> RenderedPrompt:
    """Render template and return with audit metadata.

    Args:
        row: Row data (accessed as row.* in template)

    Returns:
        RenderedPrompt with prompt string and all hashes
    """
    prompt = self.render(row)

    # Compute variables hash using canonical JSON (row data only)
    variables_hash = _sha256(canonical_json(row))

    # Compute rendered prompt hash
    rendered_hash = _sha256(prompt)

    return RenderedPrompt(
        prompt=prompt,
        template_hash=self._template_hash,
        variables_hash=variables_hash,
        rendered_hash=rendered_hash,
        template_source=self._template_source,
        lookup_hash=self._lookup_hash,
        lookup_source=self._lookup_source,
    )
```

**Step 3: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_rendered_prompt_includes_source_metadata -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/elspeth/plugins/llm/templates.py
git commit -m "feat(templates): update render methods to use row/lookup namespaces"
```

---

## Task 4: Fix Existing Tests for New Namespace

**Files:**
- Modify: `tests/plugins/llm/test_templates.py`

**Step 1: Run existing tests to see failures**

Run: `pytest tests/plugins/llm/test_templates.py -v`

Expected: Some tests fail because they use old `{{ name }}` syntax instead of `{{ row.name }}`

**Step 2: Update tests to use new namespace**

Update `test_simple_variable_substitution`:

```python
def test_simple_variable_substitution(self) -> None:
    """Basic variable substitution works with row namespace."""
    template = PromptTemplate("Hello, {{ row.name }}!")
    result = template.render({"name": "World"})
    assert result == "Hello, World!"
```

Update `test_template_with_loop`:

```python
def test_template_with_loop(self) -> None:
    """Jinja2 loops work with row namespace."""
    template = PromptTemplate(
        """
Analyze these items:
{% for item in row.items %}
- {{ item.name }}: {{ item.value }}
{% endfor %}
""".strip()
    )
    result = template.render({
        "items": [
            {"name": "A", "value": 1},
            {"name": "B", "value": 2},
        ]
    })
    assert "- A: 1" in result
    assert "- B: 2" in result
```

Update `test_template_with_default_filter`:

```python
def test_template_with_default_filter(self) -> None:
    """Jinja2 default filter works."""
    template = PromptTemplate("Focus: {{ row.focus | default('general') }}")
    assert template.render({}) == "Focus: general"
    assert template.render({"focus": "quality"}) == "Focus: quality"
```

Update `test_render_returns_metadata`:

```python
def test_render_returns_metadata(self) -> None:
    """render_with_metadata returns prompt and audit metadata."""
    template = PromptTemplate("Analyze: {{ row.text }}")
    result = template.render_with_metadata({"text": "sample"})

    assert result.prompt == "Analyze: sample"
    assert result.template_hash is not None
    assert result.variables_hash is not None
    assert result.rendered_hash is not None
```

Update `test_undefined_variable_raises_error`:

```python
def test_undefined_variable_raises_error(self) -> None:
    """Missing required variable raises TemplateError."""
    template = PromptTemplate("Hello, {{ row.name }}!")
    with pytest.raises(TemplateError, match="name"):
        template.render({})  # No 'name' in row
```

**Step 3: Run all template tests**

Run: `pytest tests/plugins/llm/test_templates.py -v`

Expected: All PASS

**Step 4: Commit**

```bash
git add tests/plugins/llm/test_templates.py
git commit -m "test(templates): update tests for row/lookup namespace"
```

---

## Task 5: Add Lookup Tests

**Files:**
- Modify: `tests/plugins/llm/test_templates.py`

**Step 1: Write test for simple lookup access**

```python
def test_lookup_simple_access(self) -> None:
    """Templates can access lookup data."""
    template = PromptTemplate(
        "Category: {{ lookup.categories[0] }}",
        lookup_data={"categories": ["Electronics", "Clothing", "Food"]},
    )
    result = template.render({})
    assert result == "Category: Electronics"
```

**Step 2: Run test**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_lookup_simple_access -v`

Expected: PASS

**Step 3: Write test for two-dimensional lookup**

```python
def test_lookup_two_dimensional(self) -> None:
    """Templates can do two-dimensional lookups: lookup.X[row.Y]."""
    template = PromptTemplate(
        "Tone: {{ lookup.tones[row.tone_id] }}",
        lookup_data={"tones": {0: "formal", 1: "casual", 2: "technical"}},
    )
    result = template.render({"tone_id": 1})
    assert result == "Tone: casual"
```

**Step 4: Run test**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_lookup_two_dimensional -v`

Expected: PASS

**Step 5: Write test for missing lookup key (strict mode)**

```python
def test_lookup_missing_key_raises_error(self) -> None:
    """Missing lookup key raises TemplateError (strict mode)."""
    template = PromptTemplate(
        "Category: {{ lookup.categories[row.cat_id] }}",
        lookup_data={"categories": {0: "A", 1: "B"}},
    )
    with pytest.raises(TemplateError):
        template.render({"cat_id": 99})  # No key 99
```

**Step 6: Run test**

Run: `pytest tests/plugins/llm/test_templates.py::TestPromptTemplate::test_lookup_missing_key_raises_error -v`

Expected: PASS

**Step 7: Write test for lookup iteration**

```python
def test_lookup_iteration(self) -> None:
    """Templates can iterate over lookup data."""
    template = PromptTemplate(
        """Categories:
{% for cat in lookup.categories %}
- {{ cat.name }}
{% endfor %}""",
        lookup_data={
            "categories": [
                {"name": "Electronics"},
                {"name": "Clothing"},
            ]
        },
    )
    result = template.render({})
    assert "- Electronics" in result
    assert "- Clothing" in result
```

**Step 8: Run all lookup tests**

Run: `pytest tests/plugins/llm/test_templates.py -v -k lookup`

Expected: All PASS

**Step 9: Commit**

```bash
git add tests/plugins/llm/test_templates.py
git commit -m "test(templates): add lookup functionality tests"
```

---

## Task 6: Add Lookup Hash Tests

**Files:**
- Modify: `tests/plugins/llm/test_templates.py`

**Step 1: Write test for lookup hash stability**

```python
def test_lookup_hash_is_stable(self) -> None:
    """Same lookup data produces same hash."""
    data = {"categories": ["A", "B", "C"]}
    t1 = PromptTemplate("{{ lookup.categories }}", lookup_data=data)
    t2 = PromptTemplate("{{ lookup.categories }}", lookup_data=data)
    assert t1.lookup_hash == t2.lookup_hash
```

**Step 2: Write test for no lookup = no hash**

```python
def test_no_lookup_has_none_hash(self) -> None:
    """Template without lookup data has None lookup_hash."""
    template = PromptTemplate("Hello, {{ row.name }}!")
    assert template.lookup_hash is None
    assert template.lookup_source is None
```

**Step 3: Run tests**

Run: `pytest tests/plugins/llm/test_templates.py -v -k "lookup_hash or no_lookup"`

Expected: All PASS

**Step 4: Commit**

```bash
git add tests/plugins/llm/test_templates.py
git commit -m "test(templates): add lookup hash tests"
```

---

## Task 7: Update LLMConfig with New Fields

**Files:**
- Modify: `src/elspeth/plugins/llm/base.py:26-64`
- Test: `tests/plugins/llm/test_openrouter.py` (or new test file)

**Step 1: Write failing test for LLMConfig with lookup**

Create or update test:

```python
def test_llm_config_accepts_lookup_fields(self) -> None:
    """LLMConfig accepts lookup and source metadata fields."""
    from elspeth.plugins.llm.base import LLMConfig

    config = LLMConfig.from_dict({
        "model": "test-model",
        "template": "Hello, {{ row.name }}!",
        "template_source": "prompts/test.j2",
        "lookup": {"key": "value"},
        "lookup_source": "prompts/lookups.yaml",
        "schema": {"fields": "dynamic"},
    })

    assert config.template_source == "prompts/test.j2"
    assert config.lookup == {"key": "value"}
    assert config.lookup_source == "prompts/lookups.yaml"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/llm/test_openrouter.py::test_llm_config_accepts_lookup_fields -v`

Expected: FAIL (fields don't exist)

**Step 3: Add new fields to LLMConfig**

In `src/elspeth/plugins/llm/base.py`:

```python
class LLMConfig(TransformDataConfig):
    """Configuration for LLM transforms."""

    model: str = Field(
        ..., description="Model identifier (e.g., 'gpt-4', 'claude-3-opus')"
    )
    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum tokens in response")
    response_field: str = Field(
        "llm_response", description="Field name for LLM response in output"
    )

    # New fields for file-based templates
    lookup: dict[str, Any] | None = Field(
        None, description="Lookup data loaded from YAML file"
    )
    template_source: str | None = Field(
        None, description="Template file path for audit (None if inline)"
    )
    lookup_source: str | None = Field(
        None, description="Lookup file path for audit (None if no lookup)"
    )

    @field_validator("template")
    @classmethod
    def validate_template(cls, v: str) -> str:
        """Validate template is non-empty and syntactically valid."""
        if not v or not v.strip():
            raise ValueError("template cannot be empty")
        try:
            PromptTemplate(v)
        except TemplateError as e:
            raise ValueError(f"Invalid Jinja2 template: {e}") from e
        return v
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/llm/test_openrouter.py::test_llm_config_accepts_lookup_fields -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/base.py tests/plugins/llm/test_openrouter.py
git commit -m "feat(llm): add lookup and source fields to LLMConfig"
```

---

## Task 8: Update BaseLLMTransform to Use New Fields

**Files:**
- Modify: `src/elspeth/plugins/llm/base.py:90-112`

**Step 1: Update __init__ to pass new params to PromptTemplate**

```python
def __init__(self, config: dict[str, Any]) -> None:
    super().__init__(config)

    cfg = LLMConfig.from_dict(config)
    self._model = cfg.model
    self._template = PromptTemplate(
        cfg.template,
        template_source=cfg.template_source,
        lookup_data=cfg.lookup,
        lookup_source=cfg.lookup_source,
    )
    self._system_prompt = cfg.system_prompt
    self._temperature = cfg.temperature
    self._max_tokens = cfg.max_tokens
    self._response_field = cfg.response_field
    self._on_error = cfg.on_error

    # Schema from config (unchanged)
    assert cfg.schema_config is not None
    schema = create_schema_from_config(
        cfg.schema_config,
        f"{self.name}Schema",
        allow_coercion=False,
    )
    self.input_schema = schema
    self.output_schema = schema
```

**Step 2: Run existing tests**

Run: `pytest tests/plugins/llm/ -v`

Expected: All PASS (no behavior change yet)

**Step 3: Commit**

```bash
git add src/elspeth/plugins/llm/base.py
git commit -m "feat(llm): pass lookup data to PromptTemplate in BaseLLMTransform"
```

---

## Task 9: Update BaseLLMTransform.process() for New Render Signature

**Files:**
- Modify: `src/elspeth/plugins/llm/base.py:116-185`

**Step 1: Update process() to use new render signature**

Change line 134 from:
```python
rendered = self._template.render_with_metadata(**row)
```

To:
```python
rendered = self._template.render_with_metadata(row)
```

**Step 2: Update audit metadata output**

Update lines 181-183 to include new fields:

```python
# Add audit metadata for template traceability
output[f"{self._response_field}_template_hash"] = rendered.template_hash
output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
output[f"{self._response_field}_template_source"] = rendered.template_source
output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
```

**Step 3: Update error result to include template_source**

Update the error block around line 136-142:

```python
except TemplateError as e:
    return TransformResult.error(
        {
            "reason": "template_rendering_failed",
            "error": str(e),
            "template_hash": self._template.template_hash,
            "template_source": self._template.template_source,
        }
    )
```

**Step 4: Run tests**

Run: `pytest tests/plugins/llm/ -v`

Expected: All PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/llm/base.py
git commit -m "feat(llm): update process() for new render signature and audit fields"
```

---

## Task 10: Add Config-Time File Expansion

**Files:**
- Modify: `src/elspeth/core/config.py`
- Test: `tests/core/test_config.py`

**Step 1: Write failing test for template file expansion**

Add to `tests/core/test_config.py`:

```python
import tempfile
from pathlib import Path

def test_expand_template_file(tmp_path: Path) -> None:
    """template_file is expanded to template content at config time."""
    from elspeth.core.config import _expand_template_files

    # Create template file
    template_file = tmp_path / "prompts" / "test.j2"
    template_file.parent.mkdir(parents=True)
    template_file.write_text("Hello, {{ row.name }}!")

    # Create settings file path (for relative resolution)
    settings_path = tmp_path / "settings.yaml"

    config = {
        "template_file": "prompts/test.j2",
    }

    expanded = _expand_template_files(config, settings_path)

    assert "template" in expanded
    assert expanded["template"] == "Hello, {{ row.name }}!"
    assert expanded["template_source"] == "prompts/test.j2"
    assert "template_file" not in expanded  # Original key removed
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config.py::test_expand_template_file -v`

Expected: FAIL (function doesn't exist)

**Step 3: Add _expand_template_files function**

Add to `src/elspeth/core/config.py` before `load_settings`:

```python
import yaml


class TemplateFileError(Exception):
    """Error loading template or lookup file."""


def _expand_template_files(
    options: dict[str, Any],
    settings_path: Path,
) -> dict[str, Any]:
    """Expand template_file and lookup_file to loaded content.

    Args:
        options: Plugin options dict
        settings_path: Path to settings file for resolving relative paths

    Returns:
        New dict with files loaded and paths recorded

    Raises:
        TemplateFileError: If files not found or invalid
    """
    result = dict(options)

    # Handle template_file
    if "template_file" in result:
        if "template" in result:
            raise TemplateFileError(
                "Cannot specify both 'template' and 'template_file'"
            )
        template_file = result.pop("template_file")
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = (settings_path.parent / template_path).resolve()

        if not template_path.exists():
            raise TemplateFileError(f"Template file not found: {template_path}")

        result["template"] = template_path.read_text(encoding="utf-8")
        result["template_source"] = template_file

    # Handle lookup_file
    if "lookup_file" in result:
        lookup_file = result.pop("lookup_file")
        lookup_path = Path(lookup_file)
        if not lookup_path.is_absolute():
            lookup_path = (settings_path.parent / lookup_path).resolve()

        if not lookup_path.exists():
            raise TemplateFileError(f"Lookup file not found: {lookup_path}")

        try:
            result["lookup"] = yaml.safe_load(
                lookup_path.read_text(encoding="utf-8")
            )
        except yaml.YAMLError as e:
            raise TemplateFileError(f"Invalid YAML in lookup file: {e}") from e

        result["lookup_source"] = lookup_file

    return result
```

**Step 4: Run test**

Run: `pytest tests/core/test_config.py::test_expand_template_file -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): add _expand_template_files for file loading"
```

---

## Task 11: Add Lookup File Expansion Test

**Files:**
- Modify: `tests/core/test_config.py`

**Step 1: Write test for lookup file expansion**

```python
def test_expand_lookup_file(tmp_path: Path) -> None:
    """lookup_file is expanded to parsed YAML at config time."""
    from elspeth.core.config import _expand_template_files

    # Create lookup file
    lookup_file = tmp_path / "prompts" / "lookups.yaml"
    lookup_file.parent.mkdir(parents=True, exist_ok=True)
    lookup_file.write_text("categories:\n  - Electronics\n  - Clothing\n")

    settings_path = tmp_path / "settings.yaml"

    config = {
        "template": "{{ lookup.categories }}",
        "lookup_file": "prompts/lookups.yaml",
    }

    expanded = _expand_template_files(config, settings_path)

    assert expanded["lookup"] == {"categories": ["Electronics", "Clothing"]}
    assert expanded["lookup_source"] == "prompts/lookups.yaml"
    assert "lookup_file" not in expanded
```

**Step 2: Run test**

Run: `pytest tests/core/test_config.py::test_expand_lookup_file -v`

Expected: PASS

**Step 3: Write test for both template and lookup files**

```python
def test_expand_template_and_lookup_files(tmp_path: Path) -> None:
    """Both template_file and lookup_file expand together."""
    from elspeth.core.config import _expand_template_files

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    (prompts_dir / "classify.j2").write_text("Category: {{ lookup.cats[row.id] }}")
    (prompts_dir / "lookups.yaml").write_text("cats:\n  0: A\n  1: B\n")

    settings_path = tmp_path / "settings.yaml"

    config = {
        "template_file": "prompts/classify.j2",
        "lookup_file": "prompts/lookups.yaml",
    }

    expanded = _expand_template_files(config, settings_path)

    assert expanded["template"] == "Category: {{ lookup.cats[row.id] }}"
    assert expanded["template_source"] == "prompts/classify.j2"
    assert expanded["lookup"] == {"cats": {0: "A", 1: "B"}}
    assert expanded["lookup_source"] == "prompts/lookups.yaml"
```

**Step 4: Run test**

Run: `pytest tests/core/test_config.py::test_expand_template_and_lookup_files -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/core/test_config.py
git commit -m "test(config): add lookup file expansion tests"
```

---

## Task 12: Add Error Handling Tests for File Expansion

**Files:**
- Modify: `tests/core/test_config.py`

**Step 1: Write test for missing template file**

```python
def test_expand_template_file_not_found(tmp_path: Path) -> None:
    """Missing template file raises TemplateFileError."""
    from elspeth.core.config import _expand_template_files, TemplateFileError

    settings_path = tmp_path / "settings.yaml"
    config = {"template_file": "prompts/missing.j2"}

    with pytest.raises(TemplateFileError, match="not found"):
        _expand_template_files(config, settings_path)
```

**Step 2: Write test for both template and template_file**

```python
def test_expand_template_file_with_inline_raises(tmp_path: Path) -> None:
    """Cannot specify both template and template_file."""
    from elspeth.core.config import _expand_template_files, TemplateFileError

    settings_path = tmp_path / "settings.yaml"
    config = {
        "template": "inline template",
        "template_file": "prompts/test.j2",
    }

    with pytest.raises(TemplateFileError, match="Cannot specify both"):
        _expand_template_files(config, settings_path)
```

**Step 3: Write test for invalid YAML**

```python
def test_expand_lookup_file_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML in lookup file raises TemplateFileError."""
    from elspeth.core.config import _expand_template_files, TemplateFileError

    lookup_file = tmp_path / "bad.yaml"
    lookup_file.write_text("invalid: yaml: content: [")

    settings_path = tmp_path / "settings.yaml"
    config = {
        "template": "test",
        "lookup_file": "bad.yaml",
    }

    with pytest.raises(TemplateFileError, match="Invalid YAML"):
        _expand_template_files(config, settings_path)
```

**Step 4: Run all error tests**

Run: `pytest tests/core/test_config.py -v -k "expand_template"`

Expected: All PASS

**Step 5: Commit**

```bash
git add tests/core/test_config.py
git commit -m "test(config): add error handling tests for template file expansion"
```

---

## Task 13: Integrate File Expansion into Config Loading

**Files:**
- Modify: `src/elspeth/core/config.py:1051-1063` (row_plugins processing)

**Step 1: Add integration test**

```python
def test_load_settings_expands_template_files(tmp_path: Path) -> None:
    """load_settings expands template_file in row_plugins."""
    from elspeth.core.config import load_settings

    # Create directory structure
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "test.j2").write_text("Hello {{ row.name }}")
    (prompts_dir / "lookups.yaml").write_text("greetings:\n  - Hello\n")

    # Create settings file
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text("""
datasource:
  plugin: csv_local
  options:
    path: test.csv

sinks:
  output:
    plugin: csv_local
    options:
      path: out.csv

output_sink: output

row_plugins:
  - plugin: openrouter_llm
    options:
      model: test
      template_file: prompts/test.j2
      lookup_file: prompts/lookups.yaml
      schema:
        fields: dynamic
""")

    settings = load_settings(settings_file)

    # Check that files were expanded
    plugin_opts = settings.row_plugins[0].options
    assert plugin_opts["template"] == "Hello {{ row.name }}"
    assert plugin_opts["template_source"] == "prompts/test.j2"
    assert plugin_opts["lookup"] == {"greetings": ["Hello"]}
    assert plugin_opts["lookup_source"] == "prompts/lookups.yaml"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/core/test_config.py::test_load_settings_expands_template_files -v`

Expected: FAIL (expansion not hooked into load_settings yet)

**Step 3: Hook expansion into load_settings**

Update `_fingerprint_config_options` in `config.py` to also expand template files. Add near the top of the function:

```python
def _fingerprint_config_options(raw_config: dict[str, Any], settings_path: Path | None = None) -> dict[str, Any]:
    """Walk config and fingerprint secrets in all plugin options.

    Also expands template_file and lookup_file references if settings_path provided.
    """
    import os

    allow_raw = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS", "").lower() == "true"
    fail_if_no_key = not allow_raw

    config = dict(raw_config)

    # ... existing landscape handling ...

    # === Row plugin options ===
    if "row_plugins" in config and isinstance(config["row_plugins"], list):
        plugins = []
        for plugin_config in config["row_plugins"]:
            if isinstance(plugin_config, dict):
                plugin = dict(plugin_config)
                if "options" in plugin and isinstance(plugin["options"], dict):
                    # Expand template files first (if settings_path available)
                    if settings_path is not None:
                        plugin["options"] = _expand_template_files(
                            plugin["options"], settings_path
                        )
                    # Then fingerprint secrets
                    plugin["options"] = _fingerprint_secrets(
                        plugin["options"], fail_if_no_key=fail_if_no_key
                    )
                plugins.append(plugin)
            else:
                plugins.append(plugin_config)
        config["row_plugins"] = plugins

    # ... rest of function unchanged ...
```

**Step 4: Update load_settings to pass settings_path**

In `load_settings`, change line 1131:

```python
raw_config = _fingerprint_config_options(raw_config, settings_path=config_path)
```

**Step 5: Update function signature**

```python
def _fingerprint_config_options(
    raw_config: dict[str, Any],
    settings_path: Path | None = None,
) -> dict[str, Any]:
```

**Step 6: Run test**

Run: `pytest tests/core/test_config.py::test_load_settings_expands_template_files -v`

Expected: PASS

**Step 7: Commit**

```bash
git add src/elspeth/core/config.py tests/core/test_config.py
git commit -m "feat(config): integrate template file expansion into load_settings"
```

---

## Task 14: Run Full Test Suite

**Files:** None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`

Expected: All PASS

**Step 2: Run type checking**

Run: `mypy src/elspeth/plugins/llm/templates.py src/elspeth/plugins/llm/base.py src/elspeth/core/config.py`

Expected: No errors

**Step 3: Run linting**

Run: `ruff check src/elspeth/plugins/llm/ src/elspeth/core/config.py`

Expected: No errors

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: fix any lint/type issues from template files feature"
```

---

## Summary

| Task | Description | Files Modified |
|------|-------------|----------------|
| 1 | Update RenderedPrompt dataclass | templates.py |
| 2 | Update PromptTemplate constructor | templates.py |
| 3 | Update render_with_metadata method | templates.py |
| 4 | Fix existing tests for new namespace | test_templates.py |
| 5 | Add lookup tests | test_templates.py |
| 6 | Add lookup hash tests | test_templates.py |
| 7 | Update LLMConfig with new fields | base.py |
| 8 | Update BaseLLMTransform init | base.py |
| 9 | Update BaseLLMTransform.process() | base.py |
| 10 | Add config-time file expansion | config.py |
| 11 | Add lookup file expansion test | test_config.py |
| 12 | Add error handling tests | test_config.py |
| 13 | Integrate into load_settings | config.py |
| 14 | Run full test suite | (verification) |
