"""Dynaconf key normalization for Elspeth settings loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STRUCTURAL_OPTION_KEYS = frozenset({"schema", "schema_config"})
USER_DATA_KEYS = frozenset({"options", "routes", "branches"})
SINK_NAME_COLLECTION_KEYS = frozenset({"sinks"})


def merge_dicts_preserving_env_override(base: dict[Any, Any], override: dict[Any, Any]) -> dict[Any, Any]:
    """Merge normalized duplicate dict keys, with override values winning."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_dicts_preserving_env_override(existing, value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True, slots=True)
class DynaconfKeyNormalizer:
    """Lowercase Dynaconf schema keys while preserving declared user-data maps."""

    structural_option_keys: frozenset[str] = STRUCTURAL_OPTION_KEYS
    user_data_keys: frozenset[str] = USER_DATA_KEYS
    sink_name_collection_keys: frozenset[str] = SINK_NAME_COLLECTION_KEYS

    def normalize(self, obj: Any) -> Any:
        return self._normalize(obj, preserve_nested=False, in_sink_names=False, in_options=False)

    def _normalize(
        self,
        obj: Any,
        *,
        preserve_nested: bool,
        in_sink_names: bool,
        in_options: bool,
    ) -> Any:
        if isinstance(obj, dict):
            return self._normalize_dict(obj, preserve_nested=preserve_nested, in_sink_names=in_sink_names, in_options=in_options)
        if isinstance(obj, list):
            return [
                self._normalize(item, preserve_nested=preserve_nested, in_sink_names=in_sink_names, in_options=in_options) for item in obj
            ]
        return obj

    def _normalize_dict(
        self,
        obj: dict[Any, Any],
        *,
        preserve_nested: bool,
        in_sink_names: bool,
        in_options: bool,
    ) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        env_structural_option_keys: set[str] = set()
        for key, value in obj.items():
            is_env_structural_option_key = (
                in_options and isinstance(key, str) and key.isupper() and key.lower() in self.structural_option_keys
            )

            if is_env_structural_option_key:
                new_key = key.lower()
            elif preserve_nested:
                new_key = key
            elif in_sink_names:
                new_key = key.lower() if key.isupper() else key
            else:
                new_key = key.lower()

            child = self._normalize_child(
                new_key,
                value,
                is_env_structural_option_key=is_env_structural_option_key,
                preserve_nested=preserve_nested,
                in_sink_names=in_sink_names,
            )
            self._merge_child(
                result,
                env_structural_option_keys,
                new_key,
                child,
                is_env_structural_option_key=is_env_structural_option_key,
            )
        return result

    def _normalize_child(
        self,
        new_key: Any,
        value: Any,
        *,
        is_env_structural_option_key: bool,
        preserve_nested: bool,
        in_sink_names: bool,
    ) -> Any:
        if is_env_structural_option_key:
            return self._normalize(value, preserve_nested=False, in_sink_names=False, in_options=False)
        if preserve_nested:
            return self._normalize(value, preserve_nested=True, in_sink_names=False, in_options=False)
        if new_key == "options":
            return self._normalize(value, preserve_nested=True, in_sink_names=False, in_options=True)
        if new_key in self.user_data_keys:
            return self._normalize(value, preserve_nested=True, in_sink_names=False, in_options=False)
        if new_key in self.sink_name_collection_keys:
            return self._normalize(value, preserve_nested=False, in_sink_names=True, in_options=False)
        if in_sink_names:
            return self._normalize(value, preserve_nested=False, in_sink_names=False, in_options=False)
        return self._normalize(value, preserve_nested=preserve_nested, in_sink_names=False, in_options=False)

    @staticmethod
    def _merge_child(
        result: dict[Any, Any],
        env_structural_option_keys: set[str],
        new_key: Any,
        child: Any,
        *,
        is_env_structural_option_key: bool,
    ) -> None:
        if new_key not in result:
            result[new_key] = child
            if is_env_structural_option_key:
                env_structural_option_keys.add(new_key)
            return

        if is_env_structural_option_key:
            if isinstance(result[new_key], dict) and isinstance(child, dict):
                result[new_key] = merge_dicts_preserving_env_override(result[new_key], child)
            else:
                result[new_key] = child
            env_structural_option_keys.add(new_key)
            return

        if isinstance(new_key, str) and new_key in env_structural_option_keys:
            if isinstance(child, dict) and isinstance(result[new_key], dict):
                result[new_key] = merge_dicts_preserving_env_override(child, result[new_key])
            return

        result[new_key] = child


def lowercase_schema_keys(obj: Any) -> Any:
    return DynaconfKeyNormalizer().normalize(obj)
