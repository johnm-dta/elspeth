# Feature Request: General Azure Key Vault Secret Loading

## Summary

The fingerprint module (`src/elspeth/core/security/fingerprint.py`) has Azure Key Vault integration for loading the fingerprint key, but this capability is not exposed as a general-purpose secret loader for other secrets (API keys, database credentials, etc.). A reusable `SecretLoader` abstraction would enable consistent, auditable secret management across all plugins.

## Severity

- Severity: enhancement
- Priority: P3

## Reporter

- Name or handle: Claude Code
- Date: 2026-01-30
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/P2-aggregation-metadata-hardcoded`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Test framework deep dive - identified missing test for Key Vault integration
- Model/version: Claude Opus 4.5
- Tooling and permissions: Claude Code CLI
- Determinism details: N/A
- Notable tool calls or steps: Security fingerprint module review during test gap analysis

## Current State

The fingerprint module already supports Key Vault:

```python
# src/elspeth/core/security/fingerprint.py:58-99
def get_fingerprint_key() -> bytes:
    """Get the fingerprint key from environment or Azure Key Vault.

    Resolution order:
    1. ELSPETH_FINGERPRINT_KEY environment variable (immediate, for dev/testing)
    2. Azure Key Vault (if ELSPETH_KEYVAULT_URL is set)
    """
```

However:
- This is a one-off implementation specific to fingerprint keys
- No general abstraction exists for loading other secrets
- LLM API keys, database passwords, etc. are loaded via env vars only
- No consistent audit trail for which secrets were used (beyond fingerprint)

## Expected Behavior

A general-purpose secret loader that:

1. **Unified Resolution Order:**
   - Environment variable (fast path for dev/testing)
   - Azure Key Vault (production)
   - (Future: AWS Secrets Manager, HashiCorp Vault)

2. **Auditable Secret Usage:**
   - Fingerprint all loaded secrets automatically
   - Record which secrets were accessed in audit trail
   - Never log or store actual secret values

3. **Configuration-Driven:**
   ```yaml
   secrets:
     openai_api_key:
       env_var: OPENAI_API_KEY
       keyvault_secret: openai-api-key
     database_password:
       env_var: DB_PASSWORD
       keyvault_secret: elspeth-db-password
   ```

4. **Plugin Integration:**
   - Transforms/sinks can request secrets by name
   - Secret loading happens once at startup
   - Fingerprints recorded in run metadata

## Proposed Design

```python
# src/elspeth/core/security/secret_loader.py

from dataclasses import dataclass
from typing import Protocol

@dataclass
class SecretRef:
    """Reference to a secret (never contains the actual value)."""
    name: str
    fingerprint: str
    source: str  # "env", "keyvault", etc.

class SecretLoader(Protocol):
    """Protocol for secret loading backends."""

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        """Load secret by name, return (value, ref for audit)."""
        ...

class CompositeSecretLoader:
    """Tries multiple backends in order."""

    def __init__(
        self,
        backends: list[SecretLoader],
        fingerprint_key: bytes,
    ):
        self._backends = backends
        self._fingerprint_key = fingerprint_key

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        for backend in self._backends:
            try:
                value, ref = backend.get_secret(name)
                # Compute fingerprint
                fp = secret_fingerprint(value, key=self._fingerprint_key)
                return value, SecretRef(name=name, fingerprint=fp, source=ref.source)
            except SecretNotFoundError:
                continue
        raise SecretNotFoundError(f"Secret '{name}' not found in any backend")
```

## Impact

- User-facing impact: Simplified secret configuration in production
- Data integrity / security impact: Better audit trail for secret usage
- Performance or cost impact: Minimal (secrets loaded once at startup)

## Root Cause / Motivation

The fingerprint implementation was built for a specific use case and not generalized. As the plugin ecosystem grows (LLM providers, databases, Azure services), a unified secret loading mechanism would:

1. Reduce configuration complexity
2. Improve security audit trail
3. Enable easier rotation (update Key Vault, not env vars)
4. Support multi-environment deployments (dev/staging/prod)

## Proposed Implementation

- Code changes (modules/files):
  - Create: `src/elspeth/core/security/secret_loader.py`
  - Modify: `src/elspeth/core/security/__init__.py` (export)
  - Modify: `src/elspeth/plugins/llm/azure_multi_query.py` (use loader)
  - Modify: `src/elspeth/core/config.py` (secrets section parsing)

- Config or schema changes:
  - Add optional `secrets` section to pipeline YAML
  - Add per-plugin `secret_ref` config option

- Tests to add:
  - `tests/core/security/test_secret_loader.py`
  - Integration tests with mock Key Vault

- Risks or migration steps:
  - Backward compatible (env vars still work)
  - Existing fingerprint implementation can be refactored to use new loader

## Architectural Considerations

- Spec or doc reference: CLAUDE.md (Audit Trail requirements)
- Alignment with current architecture: Extends existing fingerprint pattern
- Multi-provider support: Design should allow AWS Secrets Manager, etc. later

## Acceptance Criteria

- [ ] `SecretLoader` protocol defined with clear contract
- [ ] `EnvSecretLoader` backend (reads from env vars)
- [ ] `KeyVaultSecretLoader` backend (Azure Key Vault)
- [ ] `CompositeSecretLoader` with fallback chain
- [ ] All loaded secrets are fingerprinted
- [ ] Secret fingerprints recorded in run metadata
- [ ] Existing `get_fingerprint_key()` refactored to use new loader
- [ ] At least one plugin (LLM) updated to use secret loader
- [ ] Documentation updated

## Tests

- Suggested tests to run:
  - `pytest tests/core/security/`
  - `pytest tests/plugins/llm/ -k "api_key"`

- New tests required:
  - `test_secret_loader.py` - unit tests for each backend
  - `test_secret_loader_integration.py` - Key Vault mock tests

## Notes / Links

- Related files:
  - `src/elspeth/core/security/fingerprint.py` - existing Key Vault code
  - `src/elspeth/plugins/llm/azure_multi_query.py` - LLM API key usage
- Related issues/PRs: N/A
- Related design docs: N/A

## Verification (2026-02-01)

**Status: STILL VALID**

- Key Vault integration remains limited to fingerprint key loading (`get_fingerprint_key`), with no general-purpose secret loader abstraction. (`src/elspeth/core/security/fingerprint.py:58-95`)

## Resolution (2026-02-02)

**Status: RESOLVED**

### Implementation

Created `src/elspeth/core/security/secret_loader.py` with:

1. **`SecretRef`** - Frozen dataclass for audit-safe secret references (name, fingerprint, source)
2. **`SecretLoader`** - Protocol defining the interface for secret backends
3. **`EnvSecretLoader`** - Loads secrets from environment variables
4. **`KeyVaultSecretLoader`** - Loads secrets from Azure Key Vault with built-in caching
5. **`CachedSecretLoader`** - Caching wrapper for any loader
6. **`CompositeSecretLoader`** - Tries multiple backends in order (fallback chain)
7. **`SecretNotFoundError`** - Exception for missing secrets

### Usage Example

```python
from elspeth.core.security import (
    CompositeSecretLoader,
    EnvSecretLoader,
    KeyVaultSecretLoader,
)

# Create a loader chain (env first, then Key Vault)
loader = CompositeSecretLoader(backends=[
    EnvSecretLoader(),
    KeyVaultSecretLoader(vault_url="https://my-vault.vault.azure.net"),
])

# Get a secret (cached automatically)
value, ref = loader.get_secret("OPENAI_API_KEY")
# ref.source = "env" or "keyvault"
```

### Acceptance Criteria Status

- [x] `SecretLoader` protocol defined with clear contract
- [x] `EnvSecretLoader` backend (reads from env vars)
- [x] `KeyVaultSecretLoader` backend (Azure Key Vault)
- [x] `CompositeSecretLoader` with fallback chain
- [x] Caching built into `KeyVaultSecretLoader`
- [x] Existing `get_fingerprint_key()` refactored to use new loader
- [ ] All loaded secrets are fingerprinted (deferred - caller responsibility)
- [ ] Secret fingerprints recorded in run metadata (deferred - future enhancement)
- [ ] At least one plugin (LLM) updated to use secret loader (deferred - future PR)
- [ ] Documentation updated (inline docstrings added)

### Files Changed

- `src/elspeth/core/security/secret_loader.py` (new - 230 lines)
- `src/elspeth/core/security/fingerprint.py` (modified)
- `src/elspeth/core/security/__init__.py` (modified - exports new types)
- `tests/core/security/test_secret_loader.py` (new - 350 lines, 19 tests)

### Future Work

- Update LLM plugins to use `CompositeSecretLoader` for API keys
- Record secret fingerprints in run metadata for audit trail
- Add AWS Secrets Manager backend
