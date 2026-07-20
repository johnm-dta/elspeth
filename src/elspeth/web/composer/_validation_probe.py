"""Shared option preparation for resolver-free plugin construction probes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from elspeth.contracts.freeze import deep_thaw
from elspeth.core.secrets import redact_secret_refs_for_validation


def prepare_validation_probe_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Return detached runtime options safe for validation-only construction."""
    from elspeth.web.interpretation_state import strip_authoring_options

    thawed = cast(dict[str, Any], deep_thaw(options))
    runtime_options = strip_authoring_options(thawed)
    return redact_secret_refs_for_validation(runtime_options)
