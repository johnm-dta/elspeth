"""Tests for resolve_secret_refs() tree-walk helper."""

from __future__ import annotations

import pytest

from elspeth.contracts.secrets import ResolvedSecret, SecretInventoryItem
from elspeth.core.secrets import SecretResolutionError, resolve_secret_refs

_VALID_FINGERPRINT = "a" * 64


class FakeResolver:
    def __init__(self, secrets: dict[str, str]):
        self._secrets = secrets

    def list_refs(self, user_id: str) -> list[SecretInventoryItem]:
        return [SecretInventoryItem(name=k, scope="user", available=True) for k in self._secrets]

    def has_ref(self, user_id: str, name: str) -> bool:
        return name in self._secrets

    def resolve(self, user_id: str, name: str) -> ResolvedSecret | None:
        if name not in self._secrets:
            return None
        return ResolvedSecret(name=name, value=self._secrets[name], scope="user", fingerprint=_VALID_FINGERPRINT)


def test_replaces_secret_ref_in_flat_dict() -> None:
    resolver = FakeResolver({"MY_KEY": "sk-123"})
    config = {"api_key": {"secret_ref": "MY_KEY"}, "model": "gpt-4"}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"api_key": "sk-123", "model": "gpt-4"}
    assert len(resolutions) == 1
    assert resolutions[0].name == "MY_KEY"
    assert resolutions[0].value == "sk-123"


def test_replaces_nested_secret_ref() -> None:
    resolver = FakeResolver({"DB_PASS": "hunter2"})
    config = {"database": {"host": "localhost", "password": {"secret_ref": "DB_PASS"}}}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"database": {"host": "localhost", "password": "hunter2"}}
    assert len(resolutions) == 1
    assert resolutions[0].name == "DB_PASS"


def test_replaces_secret_ref_in_list() -> None:
    resolver = FakeResolver({"TOKEN_A": "abc", "TOKEN_B": "def"})
    config = {"tokens": [{"secret_ref": "TOKEN_A"}, "literal", {"secret_ref": "TOKEN_B"}]}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"tokens": ["abc", "literal", "def"]}
    assert len(resolutions) == 2
    resolved_names = {r.name for r in resolutions}
    assert resolved_names == {"TOKEN_A", "TOKEN_B"}


def test_replaces_exact_env_marker_when_name_is_declared_secret_ref() -> None:
    resolver = FakeResolver({"OPENROUTER_API_KEY": "sk-or-test"})
    config = {
        "api_key": "${OPENROUTER_API_KEY}",
        "label": "prefix-${OPENROUTER_API_KEY}",
    }

    result, resolutions = resolve_secret_refs(
        config,
        resolver,
        "user1",
        env_ref_names=frozenset({"OPENROUTER_API_KEY"}),
    )

    assert result == {
        "api_key": "sk-or-test",
        "label": "prefix-${OPENROUTER_API_KEY}",
    }
    assert len(resolutions) == 1
    assert resolutions[0].name == "OPENROUTER_API_KEY"


def test_exact_env_marker_without_declared_secret_name_is_left_for_config_loader() -> None:
    resolver = FakeResolver({"OPENROUTER_API_KEY": "sk-or-test"})
    config = {"api_key": "${OPENROUTER_API_KEY}"}

    result, resolutions = resolve_secret_refs(config, resolver, "user1")

    assert result == {"api_key": "${OPENROUTER_API_KEY}"}
    assert resolutions == []


def test_missing_exact_env_marker_secret_is_reported_with_other_missing_refs() -> None:
    resolver = FakeResolver({})
    config = {
        "api_key": "${OPENROUTER_API_KEY}",
        "token": {"secret_ref": "TOKEN"},
    }

    with pytest.raises(SecretResolutionError) as exc_info:
        resolve_secret_refs(
            config,
            resolver,
            "user1",
            env_ref_names=frozenset({"OPENROUTER_API_KEY"}),
        )

    assert exc_info.value.missing == ["OPENROUTER_API_KEY", "TOKEN"]


def test_raises_on_missing_secret() -> None:
    resolver = FakeResolver({})
    config = {"key": {"secret_ref": "MISSING_SECRET"}}
    with pytest.raises(SecretResolutionError) as exc_info:
        resolve_secret_refs(config, resolver, "user1")
    assert "MISSING_SECRET" in exc_info.value.missing
    assert "MISSING_SECRET" in str(exc_info.value)


def test_collects_all_missing_secrets() -> None:
    resolver = FakeResolver({})
    config = {
        "key_x": {"secret_ref": "SECRET_X"},
        "key_y": {"secret_ref": "SECRET_Y"},
    }
    with pytest.raises(SecretResolutionError) as exc_info:
        resolve_secret_refs(config, resolver, "user1")
    assert "SECRET_X" in exc_info.value.missing
    assert "SECRET_Y" in exc_info.value.missing
    error_str = str(exc_info.value)
    assert "SECRET_X" in error_str
    assert "SECRET_Y" in error_str


def test_leaves_non_ref_dicts_unchanged() -> None:
    resolver = FakeResolver({})
    config = {
        "options": {"timeout": 30, "retries": 3},
        "tags": ["a", "b"],
    }
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == config
    assert resolutions == []


def test_empty_config() -> None:
    resolver = FakeResolver({})
    result, resolutions = resolve_secret_refs({}, resolver, "user1")
    assert result == {}
    assert resolutions == []


def test_does_not_mutate_original() -> None:
    resolver = FakeResolver({"API_KEY": "secret-value"})
    original = {"key": {"secret_ref": "API_KEY"}, "nested": {"inner": {"secret_ref": "API_KEY"}}}
    import copy

    original_copy = copy.deepcopy(original)
    resolve_secret_refs(original, resolver, "user1")
    assert original == original_copy


def test_deeply_nested_list_in_dict_in_list() -> None:
    """Secret refs nested inside list > dict > dict structures are resolved."""
    resolver = FakeResolver({"X": "resolved-x"})
    config = {"items": [{"nested": {"secret_ref": "X"}}]}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"items": [{"nested": "resolved-x"}]}
    assert len(resolutions) == 1
    assert resolutions[0].name == "X"


def test_secret_ref_with_non_string_value() -> None:
    """A dict with secret_ref=42 (non-string) is NOT treated as a secret ref."""
    resolver = FakeResolver({})
    config = {"key": {"secret_ref": 42}}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"key": {"secret_ref": 42}}
    assert resolutions == []


def test_secret_ref_with_extra_keys() -> None:
    """A dict with secret_ref + extra keys is NOT treated as a secret ref."""
    resolver = FakeResolver({"X": "val"})
    config = {"key": {"secret_ref": "X", "extra": "y"}}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"key": {"secret_ref": "X", "extra": "y"}}
    assert resolutions == []


def test_mapping_is_secret_ref_detection() -> None:
    """_is_secret_ref detects Mapping types (not just dict) as potential secret refs.

    MappingProxyType can't be deepcopied, so it won't appear inside
    resolve_secret_refs input in practice. But _is_secret_ref uses
    Mapping check to handle OrderedDict and other Mapping subtypes.
    """
    from collections import OrderedDict

    resolver = FakeResolver({"KEY": "resolved"})
    config = {"items": [OrderedDict({"secret_ref": "KEY"})]}
    result, resolutions = resolve_secret_refs(config, resolver, "user1")
    assert result == {"items": ["resolved"]}
    assert len(resolutions) == 1
