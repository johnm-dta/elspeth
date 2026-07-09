"""Shared helpers for declaration-contract runtime checks."""

from __future__ import annotations

from typing import Any

from elspeth.contracts.errors import FrameworkBugError
from elspeth.contracts.schema_contract import PipelineRow


def _require_bool_flag(plugin: Any, *, attr_name: str) -> bool:
    """Return a declaration flag only when it is an exact ``bool``."""

    value = getattr(plugin, attr_name)
    if type(value) is not bool:
        raise TypeError(f"{type(plugin).__name__}.{attr_name} must be bool, got {type(value).__name__!r}.")
    return value


def _runtime_observed_fields(emitted: PipelineRow, *, plugin_name: str) -> frozenset[str]:
    """Return fields present in both an emitted row's contract and payload."""

    if emitted.contract is None:
        raise FrameworkBugError(f"Transform {plugin_name!r} emitted row with no contract. Framework invariant violated.")
    runtime_contract_fields = frozenset(fc.normalized_name for fc in emitted.contract.fields)
    runtime_payload_fields = frozenset(emitted.keys())
    return runtime_contract_fields & runtime_payload_fields
