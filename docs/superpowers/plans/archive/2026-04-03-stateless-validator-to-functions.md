# PluginConfigValidator → Module-Level Functions

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the stateless `PluginConfigValidator` class to module-level functions, eliminating a needless instantiation step in two production callers.

**Architecture:** Pure refactor — no behavioral changes. The class has no `__init__` and no instance state; every method only uses `self` to call sibling methods. Convert all methods to module-level functions, update callers (`manager.py`, `catalog/service.py`), update tests, and update the CI contracts whitelist. `validate_schema_config` has no production callers (only tests) — keep it as a module-level function for test use.

**Tech Stack:** Python, Pydantic (validation models), pluggy (plugin system)

**Filigree issue:** `elspeth-869a9614e3`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/elspeth/plugins/infrastructure/validation.py` | Convert class methods to module-level functions |
| Modify | `src/elspeth/plugins/infrastructure/manager.py` | Remove `PluginConfigValidator()` instantiation, call functions directly |
| Modify | `src/elspeth/web/catalog/service.py` | Remove `PluginConfigValidator()` instantiation, call functions directly |
| Modify | `config/cicd/contracts-whitelist.yaml` | Update fingerprints (class prefix removed from paths) |
| Modify | `tests/unit/plugins/test_validation.py` | Call functions directly instead of instantiating class |
| Modify | `tests/unit/plugins/test_validation_integration.py` | Update `_validator` attribute check, import functions |
| Modify | `tests/unit/plugins/llm/test_plugin_registration.py` | Call functions directly instead of instantiating class |

---

### Task 1: Baseline — run all affected tests

**Files:**
- Verify: `tests/unit/plugins/test_validation.py`
- Verify: `tests/unit/plugins/test_validation_integration.py`
- Verify: `tests/unit/plugins/llm/test_plugin_registration.py`

- [ ] **Step 1: Run all three test files to establish green baseline**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation.py tests/unit/plugins/test_validation_integration.py tests/unit/plugins/llm/test_plugin_registration.py -v`
Expected: All tests PASS

---

### Task 2: Convert `PluginConfigValidator` class to module-level functions

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/validation.py`

This is the core change. Every `self.method(...)` call becomes a direct function call. The two private helpers (`_extract_errors`, `_extract_wrapped_plugin_config_error`) become module-private functions.

- [ ] **Step 1: Rewrite `validation.py` — replace class with module-level functions**

Replace the entire `PluginConfigValidator` class (lines 59–387) with module-level functions. The module docstring, imports, `UnknownPluginTypeError`, and `ValidationError` dataclass (lines 1–57) stay unchanged.

Delete the class definition and dedent all methods to module level. Remove `self` from every parameter list and every internal call.

The new module-level functions (replacing lines 59–387):

```python
def validate_source_config(
    source_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate source plugin configuration.

    Args:
        source_type: Plugin type name (e.g., "csv", "json")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    config_model = get_source_config_model(source_type)

    if config_model is None:
        return []

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def get_source_config_model(source_type: str) -> type["PluginConfig"] | None:
    """Get Pydantic config model for source type.

    Returns:
        Config model class, or None for sources with no config (e.g., null_source)
    """
    if source_type == "csv":
        from elspeth.plugins.sources.csv_source import CSVSourceConfig

        return CSVSourceConfig
    elif source_type == "text":
        from elspeth.plugins.sources.text_source import TextSourceConfig

        return TextSourceConfig
    elif source_type == "json":
        from elspeth.plugins.sources.json_source import JSONSourceConfig

        return JSONSourceConfig
    elif source_type == "azure_blob":
        from elspeth.plugins.sources.azure_blob_source import AzureBlobSourceConfig

        return AzureBlobSourceConfig
    elif source_type == "dataverse":
        from elspeth.plugins.sources.dataverse import DataverseSourceConfig

        return DataverseSourceConfig
    elif source_type == "null":
        # NullSource has no config class (resume-only source)
        return None
    else:
        raise UnknownPluginTypeError(f"Unknown source type: {source_type}")


def validate_transform_config(
    transform_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate transform plugin configuration.

    Args:
        transform_type: Plugin type name (e.g., "passthrough", "field_mapper")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    config_model = get_transform_config_model(transform_type, config)

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def validate_sink_config(
    sink_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate sink plugin configuration.

    Args:
        sink_type: Plugin type name (e.g., "csv", "json")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    config_model = get_sink_config_model(sink_type)

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def _extract_wrapped_plugin_config_error(
    error: PluginConfigError,
    config: dict[str, object],
) -> list[ValidationError]:
    """Convert wrapped PluginConfigError causes into structured errors.

    PluginConfig.from_dict() wraps:
    - PydanticValidationError for model-level validation failures
    - ValueError for schema parsing failures before model validation
    """
    cause = error.__cause__

    if cause is None:
        return [ValidationError(field="config", message=str(error), value=config)]

    if type(cause) is PydanticValidationError:
        return _extract_errors(cause)

    if type(cause) is ValueError:
        if "schema" in config:
            return [ValidationError(field="schema", message=str(cause), value=config["schema"])]
        return [ValidationError(field="config", message=str(cause), value=config)]

    raise error


def validate_schema_config(
    schema_config: dict[str, Any],
) -> list[ValidationError]:
    """Validate schema configuration independently of plugin.

    Args:
        schema_config: Schema configuration dict (contents of 'schema' key)

    Returns:
        List of validation errors (empty if valid)
    """
    from elspeth.contracts.schema import SchemaConfig

    try:
        SchemaConfig.from_dict(schema_config)
        return []
    except ValueError as e:
        return [
            ValidationError(
                field="schema",
                message=str(e),
                value=schema_config,
            )
        ]


def get_transform_config_model(
    transform_type: str,
    config: dict[str, Any] | None = None,
) -> type["PluginConfig"]:
    """Get Pydantic config model for transform type.

    Args:
        transform_type: Plugin type name
        config: Plugin configuration dict (needed for provider dispatch on "llm")

    Returns:
        Config model class for the transform type
    """
    if transform_type == "passthrough":
        from elspeth.plugins.transforms.passthrough import PassThroughConfig

        return PassThroughConfig
    elif transform_type == "field_mapper":
        from elspeth.plugins.transforms.field_mapper import FieldMapperConfig

        return FieldMapperConfig
    elif transform_type == "json_explode":
        from elspeth.plugins.transforms.json_explode import JSONExplodeConfig

        return JSONExplodeConfig
    elif transform_type == "keyword_filter":
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig

        return KeywordFilterConfig
    elif transform_type == "truncate":
        from elspeth.plugins.transforms.truncate import TruncateConfig

        return TruncateConfig
    elif transform_type == "batch_replicate":
        from elspeth.plugins.transforms.batch_replicate import BatchReplicateConfig

        return BatchReplicateConfig
    elif transform_type == "batch_stats":
        from elspeth.plugins.transforms.batch_stats import BatchStatsConfig

        return BatchStatsConfig
    elif transform_type == "azure_content_safety":
        from elspeth.plugins.transforms.azure.content_safety import AzureContentSafetyConfig

        return AzureContentSafetyConfig
    elif transform_type == "azure_prompt_shield":
        from elspeth.plugins.transforms.azure.prompt_shield import AzurePromptShieldConfig

        return AzurePromptShieldConfig
    elif transform_type == "llm":
        from elspeth.plugins.transforms.llm.transform import _PROVIDERS

        provider = config["provider"] if config is not None and "provider" in config else None
        if provider in _PROVIDERS:
            config_cls, _ = _PROVIDERS[provider]
            return config_cls
        elif provider is not None:
            raise ValueError(f"Unknown LLM provider '{provider}'. Valid providers: {sorted(_PROVIDERS)}")
        else:
            from elspeth.plugins.transforms.llm.base import LLMConfig

            return LLMConfig
    elif transform_type == "azure_batch_llm":
        from elspeth.plugins.transforms.llm.azure_batch import AzureBatchConfig

        return AzureBatchConfig
    elif transform_type == "openrouter_batch_llm":
        from elspeth.plugins.transforms.llm.openrouter_batch import OpenRouterBatchConfig

        return OpenRouterBatchConfig
    elif transform_type == "web_scrape":
        from elspeth.plugins.transforms.web_scrape import WebScrapeConfig

        return WebScrapeConfig
    elif transform_type == "rag_retrieval":
        from elspeth.plugins.transforms.rag.config import RAGRetrievalConfig

        return RAGRetrievalConfig
    else:
        raise UnknownPluginTypeError(f"Unknown transform type: {transform_type}")


def get_sink_config_model(sink_type: str) -> type["PluginConfig"]:
    """Get Pydantic config model for sink type.

    Returns:
        Config model class for the sink type
    """
    if sink_type == "csv":
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig

        return CSVSinkConfig
    elif sink_type == "json":
        from elspeth.plugins.sinks.json_sink import JSONSinkConfig

        return JSONSinkConfig
    elif sink_type == "database":
        from elspeth.plugins.sinks.database_sink import DatabaseSinkConfig

        return DatabaseSinkConfig
    elif sink_type == "azure_blob":
        from elspeth.plugins.sinks.azure_blob_sink import AzureBlobSinkConfig

        return AzureBlobSinkConfig
    elif sink_type == "dataverse":
        from elspeth.plugins.sinks.dataverse import DataverseSinkConfig

        return DataverseSinkConfig
    elif sink_type == "chroma_sink":
        from elspeth.plugins.sinks.chroma_sink import ChromaSinkConfig

        return ChromaSinkConfig
    else:
        raise UnknownPluginTypeError(f"Unknown sink type: {sink_type}")


def _extract_errors(
    pydantic_error: PydanticValidationError,
) -> list[ValidationError]:
    """Convert Pydantic errors to structured ValidationError list."""
    errors: list[ValidationError] = []

    for err in pydantic_error.errors():
        field_path = ".".join(str(loc) for loc in err["loc"]) or "__model__"
        message = err["msg"]
        value = err["input"]

        errors.append(
            ValidationError(
                field=field_path,
                message=message,
                value=value,
            )
        )

    return errors
```

Also update the module docstring (lines 1–18) to remove the class-based usage example:

Replace lines 12–17:
```python
Usage:
    validator = PluginConfigValidator()
    errors = validator.validate_source_config("csv", config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    source = CSVSource(config)  # Assumes config is valid
```

With:
```python
Usage:
    from elspeth.plugins.infrastructure.validation import validate_source_config

    errors = validate_source_config("csv", config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    source = CSVSource(config)  # Assumes config is valid
```

- [ ] **Step 2: Run the validation unit tests (expect failures from tests still importing the class)**

> **Warning:** Between this task and completion of Tasks 3–5, any test that imports `manager.py` or `catalog/service.py` will also fail with `ImportError` (they still import the deleted class). Only run the validation-specific tests here, not the full suite.

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation.py -v --tb=short 2>&1 | head -30`
Expected: ImportError or AttributeError failures (tests still reference `PluginConfigValidator`)

---

### Task 3: Update `manager.py` — remove class instantiation

**Files:**
- Modify: `src/elspeth/plugins/infrastructure/manager.py`

> **Note:** The companion quick-win-dedup plan has already added `_raise_if_invalid` and `ValidationError` to this file. The current import is `from elspeth.plugins.infrastructure.validation import PluginConfigValidator, ValidationError`. We must preserve `ValidationError` in the new import.

- [ ] **Step 1: Update import in `manager.py`**

Replace:
```python
from elspeth.plugins.infrastructure.validation import PluginConfigValidator, ValidationError
```

With:
```python
from elspeth.plugins.infrastructure.validation import (
    ValidationError,
    validate_sink_config,
    validate_source_config,
    validate_transform_config,
)
```

- [ ] **Step 2: Remove `self._validator` instantiation**

Delete the line (content match — do not rely on line numbers):
```python
        self._validator = PluginConfigValidator()
```

- [ ] **Step 3: Update the three `create_*` method call sites**

In `create_source` (around line 221), replace:
```python
        errors = self._validator.validate_source_config(source_type, config)
```
With:
```python
        errors = validate_source_config(source_type, config)
```

In `create_transform` (around line 248), replace:
```python
        errors = self._validator.validate_transform_config(transform_type, config)
```
With:
```python
        errors = validate_transform_config(transform_type, config)
```

In `create_sink` (around line 275), replace:
```python
        errors = self._validator.validate_sink_config(sink_type, config)
```
With:
```python
        errors = validate_sink_config(sink_type, config)
```

- [ ] **Step 4: Run integration tests to verify manager still works**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation_integration.py -v --tb=short`
Expected: Most tests PASS. `test_plugin_manager_has_validator` will FAIL (it checks for `_validator` attribute — fixed in Task 5).

---

### Task 4: Update `catalog/service.py` — remove class instantiation

**Files:**
- Modify: `src/elspeth/web/catalog/service.py:10,34,142-146`

- [ ] **Step 1: Update imports in `service.py`**

Replace line 10:
```python
from elspeth.plugins.infrastructure.validation import PluginConfigValidator, UnknownPluginTypeError
```

With:
```python
from elspeth.plugins.infrastructure.validation import (
    UnknownPluginTypeError,
    get_sink_config_model,
    get_source_config_model,
    get_transform_config_model,
)
```

- [ ] **Step 2: Remove `self._validator` instantiation**

Delete line 34:
```python
        self._validator = PluginConfigValidator()
```

- [ ] **Step 3: Update `_resolve_config_model` method and stale comments**

Update the comment on line 66 from `# Get config model via PluginConfigValidator` to `# Get config model via validation module`. Update the docstring at line 137 to replace "wired into PluginConfigValidator's mapping" with "wired into validation module's dispatch functions".

Replace the method body (lines 140–150):
```python
        try:
            if plugin_type == "source":
                return self._validator.get_source_config_model(name)
            elif plugin_type == "transform":
                return self._validator.get_transform_config_model(name)
            elif plugin_type == "sink":
                return self._validator.get_sink_config_model(name)
            else:
                raise ValueError(f"Bug: _resolve_config_model called with invalid plugin_type: {plugin_type!r}")
        except UnknownPluginTypeError:
            return None
```

With:
```python
        try:
            if plugin_type == "source":
                return get_source_config_model(name)
            elif plugin_type == "transform":
                return get_transform_config_model(name)
            elif plugin_type == "sink":
                return get_sink_config_model(name)
            else:
                raise ValueError(f"Bug: _resolve_config_model called with invalid plugin_type: {plugin_type!r}")
        except UnknownPluginTypeError:
            return None
```

- [ ] **Step 4: Run catalog tests**

Run: `.venv/bin/python -m pytest tests/unit/web/ -v --tb=short -q`
Expected: All catalog tests PASS

---

### Task 5: Update test files — call functions directly

**Files:**
- Modify: `tests/unit/plugins/test_validation.py`
- Modify: `tests/unit/plugins/test_validation_integration.py`
- Modify: `tests/unit/plugins/llm/test_plugin_registration.py`

- [ ] **Step 1: Update `test_validation.py`**

Replace the import (line 5):
```python
from elspeth.plugins.infrastructure.validation import PluginConfigValidator
```

With:
```python
from elspeth.plugins.infrastructure.validation import (
    validate_schema_config,
    validate_sink_config,
    validate_source_config,
    validate_transform_config,
)
```

Then do a global find-and-replace throughout the file:
- Delete every line matching `validator = PluginConfigValidator()`
- Replace every `validator.validate_source_config(` with `validate_source_config(`
- Replace every `validator.validate_transform_config(` with `validate_transform_config(`
- Replace every `validator.validate_sink_config(` with `validate_sink_config(`
- Replace every `validator.validate_schema_config(` with `validate_schema_config(`

In the parametrized test `test_validator_returns_structured_error_for_invalid_plugin_schema` (around line 344), the `getattr(validator, validator_method)` pattern must change. Replace lines 348–353:
```python
    validator = PluginConfigValidator()
    validate = getattr(validator, validator_method)

    errors = validate(plugin_type, config)
```

With a direct dispatch:
```python
    dispatch = {
        "validate_source_config": validate_source_config,
        "validate_transform_config": validate_transform_config,
        "validate_sink_config": validate_sink_config,
    }
    validate = dispatch[validator_method]

    errors = validate(plugin_type, config)
```

- [ ] **Step 2: Update `test_validation_integration.py`**

The `test_plugin_manager_has_validator` test (line 14) checks for `manager._validator`. Since the validator attribute no longer exists, **delete this test entirely** (lines 14–20). The remaining tests verify the manager validates configs before creation — that behavior is unchanged.

In `TestValidatorRecognisesMissingPluginTypes` (lines 105–141), replace the pattern in each test method. Each method does:
```python
        from elspeth.plugins.infrastructure.validation import PluginConfigValidator

        validator = PluginConfigValidator()
        model = validator.get_source_config_model("dataverse")
```

Replace with direct function imports. For each of the 4 test methods:

`test_source_dataverse_recognised`:
```python
    def test_source_dataverse_recognised(self) -> None:
        from elspeth.plugins.infrastructure.validation import get_source_config_model

        model = get_source_config_model("dataverse")
        assert model is not None
        assert model.__name__ == "DataverseSourceConfig"
```

`test_transform_rag_retrieval_recognised`:
```python
    def test_transform_rag_retrieval_recognised(self) -> None:
        from elspeth.plugins.infrastructure.validation import get_transform_config_model

        model = get_transform_config_model("rag_retrieval")
        assert model.__name__ == "RAGRetrievalConfig"
```

`test_sink_dataverse_recognised`:
```python
    def test_sink_dataverse_recognised(self) -> None:
        from elspeth.plugins.infrastructure.validation import get_sink_config_model

        model = get_sink_config_model("dataverse")
        assert model.__name__ == "DataverseSinkConfig"
```

`test_sink_chroma_sink_recognised`:
```python
    def test_sink_chroma_sink_recognised(self) -> None:
        from elspeth.plugins.infrastructure.validation import get_sink_config_model

        model = get_sink_config_model("chroma_sink")
        assert model.__name__ == "ChromaSinkConfig"
```

- [ ] **Step 3: Update `test_plugin_registration.py`**

Replace the import (line 14):
```python
from elspeth.plugins.infrastructure.validation import PluginConfigValidator
```

With:
```python
from elspeth.plugins.infrastructure.validation import get_transform_config_model
```

Then throughout the file, replace every occurrence of:
```python
        validator = PluginConfigValidator()
        config_model = validator.get_transform_config_model(
```

With:
```python
        config_model = get_transform_config_model(
```

And replace every:
```python
        validator = PluginConfigValidator()
        with pytest.raises(...):
            validator.get_transform_config_model(
```

With:
```python
        with pytest.raises(...):
            get_transform_config_model(
```

- [ ] **Step 4: Run all three test files**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation.py tests/unit/plugins/test_validation_integration.py tests/unit/plugins/llm/test_plugin_registration.py -v`
Expected: All tests PASS

---

### Task 6: Update CI contracts whitelist

**Files:**
- Modify: `config/cicd/contracts-whitelist.yaml:125-128,265`

- [ ] **Step 1: Update the whitelist entries**

The contracts checker uses `file:qualified_name:param` fingerprints. With the class removed, the method prefix changes from `PluginConfigValidator.method_name` to just `method_name`.

Replace lines 125–128:
```yaml
  - "src/elspeth/plugins/infrastructure/validation.py:PluginConfigValidator.validate_source_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:PluginConfigValidator.validate_transform_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:PluginConfigValidator.validate_sink_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:PluginConfigValidator.validate_schema_config:schema_config"
```

With:
```yaml
  - "src/elspeth/plugins/infrastructure/validation.py:validate_source_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:validate_transform_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:validate_sink_config:config"
  - "src/elspeth/plugins/infrastructure/validation.py:validate_schema_config:schema_config"
```

Replace line 265:
```yaml
  - "src/elspeth/plugins/infrastructure/validation.py:PluginConfigValidator.get_transform_config_model:config"
```

With:
```yaml
  - "src/elspeth/plugins/infrastructure/validation.py:get_transform_config_model:config"
```

- [ ] **Step 2: Run the contracts checker**

Run: `.venv/bin/python -m scripts.check_contracts`
Expected: PASS (no new violations)

- [ ] **Step 3: Run the tier model enforcer**

Run: `.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model`
Expected: PASS (no new violations — layer dependencies unchanged)

---

### Task 7: Final verification and commit

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run full affected test suite**

Run: `.venv/bin/python -m pytest tests/unit/plugins/test_validation.py tests/unit/plugins/test_validation_integration.py tests/unit/plugins/llm/test_plugin_registration.py tests/unit/web/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run ruff lint check**

Run: `.venv/bin/python -m ruff check src/elspeth/plugins/infrastructure/validation.py src/elspeth/plugins/infrastructure/manager.py src/elspeth/web/catalog/service.py`
Expected: No lint errors

- [ ] **Step 3: Run mypy type check on affected files**

Run: `.venv/bin/python -m mypy src/elspeth/plugins/infrastructure/validation.py src/elspeth/plugins/infrastructure/manager.py src/elspeth/web/catalog/service.py`
Expected: No type errors

- [ ] **Step 4: Verify no remaining references to `PluginConfigValidator` in production code, tests, or config**

Run: `grep -r "PluginConfigValidator" src/ tests/ config/`
Expected: Zero matches (references in `docs/` are historical plans/specs and acceptable)

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/plugins/infrastructure/validation.py \
       src/elspeth/plugins/infrastructure/manager.py \
       src/elspeth/web/catalog/service.py \
       config/cicd/contracts-whitelist.yaml \
       tests/unit/plugins/test_validation.py \
       tests/unit/plugins/test_validation_integration.py \
       tests/unit/plugins/llm/test_plugin_registration.py
git commit -m "refactor: convert PluginConfigValidator to module-level functions

Stateless class with no __init__ or instance state — all methods only
used self to call sibling methods. Module-level functions are the
idiomatic Python equivalent. Eliminates needless instantiation in
manager.py and catalog/service.py.

Closes elspeth-869a9614e3"
```
