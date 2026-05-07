"""Unit tests for secret_loader backends and composition helpers."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Mapping
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from elspeth.core.security.secret_loader import (
    CachedSecretLoader,
    CompositeSecretLoader,
    EnvSecretLoader,
    KeyVaultSecretLoader,
    SecretNotFoundError,
    SecretRef,
    _get_keyvault_client,
)


def _install_fake_azure_modules(monkeypatch: pytest.MonkeyPatch) -> type[Exception]:
    """Install minimal fake azure modules needed by secret_loader imports."""
    azure_module = ModuleType("azure")
    identity_module = ModuleType("azure.identity")
    keyvault_module = ModuleType("azure.keyvault")
    secrets_module = ModuleType("azure.keyvault.secrets")
    core_module = ModuleType("azure.core")
    core_exceptions_module = ModuleType("azure.core.exceptions")

    class FakeDefaultAzureCredential:
        pass

    class FakeSecretClient:
        def __init__(self, *, vault_url: str, credential: object) -> None:
            self.vault_url = vault_url
            self.credential = credential

        def get_secret(self, name: str) -> object:  # pragma: no cover - patched in tests
            raise RuntimeError(name)

    class FakeResourceNotFoundError(Exception):
        pass

    identity_module.DefaultAzureCredential = FakeDefaultAzureCredential  # type: ignore[attr-defined]
    secrets_module.SecretClient = FakeSecretClient  # type: ignore[attr-defined]
    core_exceptions_module.ResourceNotFoundError = FakeResourceNotFoundError  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_module)
    monkeypatch.setitem(sys.modules, "azure.keyvault", keyvault_module)
    monkeypatch.setitem(sys.modules, "azure.keyvault.secrets", secrets_module)
    monkeypatch.setitem(sys.modules, "azure.core", core_module)
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", core_exceptions_module)
    return FakeResourceNotFoundError


def test_get_keyvault_client_creates_secret_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_keyvault_client should construct a usable SecretClient."""
    _install_fake_azure_modules(monkeypatch)
    client = _get_keyvault_client("https://unit-test-vault.vault.azure.net")

    assert client is not None
    assert client.vault_url == "https://unit-test-vault.vault.azure.net"


def test_get_keyvault_client_raises_helpful_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing Azure packages should raise the custom dependency message."""
    original_import = builtins.__import__

    def _patched_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "azure.identity":
            raise ImportError("azure.identity missing")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _patched_import)

    with pytest.raises(ImportError, match="azure-keyvault-secrets and azure-identity are required"):
        _get_keyvault_client("https://unit-test-vault.vault.azure.net")


class TestEnvSecretLoader:
    """Environment loader behavior."""

    def test_get_secret_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_SECRET", "secret-value")

        value, ref = EnvSecretLoader().get_secret("APP_SECRET")

        assert value == "secret-value"
        assert ref == SecretRef(name="APP_SECRET", fingerprint="", source="env")

    @pytest.mark.parametrize("env_value", [None, ""])
    def test_missing_or_empty_env_var_raises(self, monkeypatch: pytest.MonkeyPatch, env_value: str | None) -> None:
        if env_value is None:
            monkeypatch.delenv("APP_SECRET", raising=False)
        else:
            monkeypatch.setenv("APP_SECRET", env_value)

        with pytest.raises(SecretNotFoundError, match="APP_SECRET"):
            EnvSecretLoader().get_secret("APP_SECRET")


class TestKeyVaultSecretLoader:
    """Key Vault loader behavior."""

    def test_get_secret_caches_successful_lookup(self) -> None:
        loader = KeyVaultSecretLoader("https://cache-test.vault.azure.net")
        client = MagicMock()
        client.get_secret.return_value = MagicMock(value="from-vault")
        loader._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        first_value, first_ref = loader.get_secret("API_KEY")
        second_value, second_ref = loader.get_secret("API_KEY")

        assert first_value == "from-vault"
        assert second_value == "from-vault"
        assert first_ref.source == "keyvault"
        assert second_ref.source == "keyvault"
        assert client.get_secret.call_count == 1

    def test_get_secret_none_value_raises_secret_not_found(self) -> None:
        loader = KeyVaultSecretLoader("https://empty-secret.vault.azure.net")
        client = MagicMock()
        client.get_secret.return_value = MagicMock(value=None)
        loader._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        with pytest.raises(SecretNotFoundError, match="has no value"):
            loader.get_secret("EMPTY_SECRET")

    def test_get_secret_import_error_from_client_creation_propagates(self) -> None:
        loader = KeyVaultSecretLoader("https://imports.vault.azure.net")
        loader._get_client = MagicMock(side_effect=ImportError("azure unavailable"))  # type: ignore[method-assign]

        with pytest.raises(ImportError, match="azure unavailable"):
            loader.get_secret("API_KEY")

    def test_get_secret_translates_azure_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_not_found = _install_fake_azure_modules(monkeypatch)
        loader = KeyVaultSecretLoader("https://missing-secret.vault.azure.net")
        client = MagicMock()
        client.get_secret.side_effect = fake_not_found("404")
        loader._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        with pytest.raises(SecretNotFoundError, match="not found in Key Vault"):
            loader.get_secret("DOES_NOT_EXIST")

    def test_get_secret_works_when_azure_core_exceptions_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If azure.core.exceptions import fails, sentinel path should still allow success."""
        original_import = builtins.__import__

        def _patched_import(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "azure.core.exceptions":
                raise ImportError("azure.core missing")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        loader = KeyVaultSecretLoader("https://sentinel-path.vault.azure.net")
        client = MagicMock()
        client.get_secret.return_value = MagicMock(value="fallback-success")
        loader._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        value, ref = loader.get_secret("ANY_SECRET")

        assert value == "fallback-success"
        assert ref.source == "keyvault"

    def test_clear_cache_forces_refetch(self) -> None:
        loader = KeyVaultSecretLoader("https://clear-cache.vault.azure.net")
        client = MagicMock()
        client.get_secret.return_value = MagicMock(value="refetch-me")
        loader._get_client = MagicMock(return_value=client)  # type: ignore[method-assign]

        loader.get_secret("REFRESH")
        loader.clear_cache()
        loader.get_secret("REFRESH")

        assert client.get_secret.call_count == 2


class _CountingLoader:
    """Simple deterministic loader for cache/composition tests."""

    def __init__(self, value: str) -> None:
        self._value = value
        self.calls = 0

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        self.calls += 1
        return self._value, SecretRef(name=name, fingerprint="", source="stub")


class _MissingLoader:
    """Loader that always reports missing secrets."""

    def __init__(self) -> None:
        self.calls = 0

    def get_secret(self, name: str) -> tuple[str, SecretRef]:
        self.calls += 1
        raise SecretNotFoundError(f"{name} missing")


class TestCachedSecretLoader:
    """Generic cache wrapper behavior."""

    def test_get_secret_uses_cache_until_cleared(self) -> None:
        inner = _CountingLoader("cached-value")
        loader = CachedSecretLoader(inner=inner)

        first, _ = loader.get_secret("CACHE_ME")
        second, _ = loader.get_secret("CACHE_ME")
        loader.clear_cache()
        third, _ = loader.get_secret("CACHE_ME")

        assert first == "cached-value"
        assert second == "cached-value"
        assert third == "cached-value"
        assert inner.calls == 2

    def test_missing_secret_is_not_cached(self) -> None:
        inner = _MissingLoader()
        loader = CachedSecretLoader(inner=inner)

        with pytest.raises(SecretNotFoundError):
            loader.get_secret("MISSING")
        with pytest.raises(SecretNotFoundError):
            loader.get_secret("MISSING")

        assert inner.calls == 2


class TestCompositeSecretLoader:
    """Composition and fallback behavior."""

    def test_requires_at_least_one_backend(self) -> None:
        with pytest.raises(ValueError, match="at least one backend"):
            CompositeSecretLoader(backends=[])

    def test_uses_first_backend_that_succeeds(self) -> None:
        missing = _MissingLoader()
        fallback = _CountingLoader("resolved")
        loader = CompositeSecretLoader(backends=[missing, fallback])

        value, ref = loader.get_secret("CHAINED_SECRET")

        assert value == "resolved"
        assert ref.source == "stub"
        assert missing.calls == 1
        assert fallback.calls == 1

    def test_raises_when_all_backends_missing(self) -> None:
        first = _MissingLoader()
        second = _MissingLoader()
        loader = CompositeSecretLoader(backends=[first, second])

        with pytest.raises(SecretNotFoundError, match="not found in any backend"):
            loader.get_secret("NOPE")

        assert first.calls == 1
        assert second.calls == 1


class TestClearCacheThreadSafety:
    """Regression: clear_cache() must acquire the lock."""

    def test_keyvault_clear_cache_acquires_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_modules(monkeypatch)
        loader = KeyVaultSecretLoader("https://test.vault.azure.net")
        # Verify lock is acquired by checking that clear_cache
        # blocks when lock is held
        import threading

        lock_acquired = threading.Event()
        clear_started = threading.Event()

        def blocking_test() -> None:
            clear_started.set()
            with loader._lock:
                lock_acquired.set()
                # Hold lock briefly
                import time

                time.sleep(0.1)

        t = threading.Thread(target=blocking_test)
        t.start()
        clear_started.wait()
        lock_acquired.wait()
        # clear_cache should block until lock is released
        loader.clear_cache()
        t.join()

    def test_cached_loader_clear_cache_acquires_lock(self) -> None:
        inner = _MissingLoader()
        loader = CachedSecretLoader(inner)
        import threading

        lock_held = threading.Event()
        clear_done = threading.Event()

        def hold_lock() -> None:
            with loader._lock:
                lock_held.set()
                import time

                time.sleep(0.1)

        t = threading.Thread(target=hold_lock)
        t.start()
        lock_held.wait()
        loader.clear_cache()
        clear_done.set()
        t.join()
