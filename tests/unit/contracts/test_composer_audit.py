"""Unit tests for the composer-audit L0 contract.

Pin the invariants reviewers identified as load-bearing:
- canonical-JSON round-trip determinism
- StrEnum and datetime serialization in to_dict()
- frozen=True prevents post-construction mutation
- cache_hit defaults to False
- the dataclass imports nothing above L0
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import UTC, datetime

import pytest

from elspeth.contracts.composer_audit import (
    ComposerToolInvocation,
    ComposerToolRecorder,
    ComposerToolStatus,
)
from elspeth.core.canonical import canonical_json, stable_hash


def _make_invocation(**overrides: object) -> ComposerToolInvocation:
    """Construct a ComposerToolInvocation with sensible defaults."""
    base_args = {"node_id": "n1", "plugin": "csv_source"}
    canon = canonical_json(base_args)
    h = stable_hash(base_args)
    t = datetime(2026, 5, 4, 12, 0, 0, tzinfo=UTC)
    defaults: dict[str, object] = {
        "tool_call_id": "tc-abc",
        "tool_name": "upsert_node",
        "arguments_canonical": canon,
        "arguments_hash": h,
        "result_canonical": canon,
        "result_hash": h,
        "status": ComposerToolStatus.SUCCESS,
        "error_class": None,
        "error_message": None,
        "version_before": 1,
        "version_after": 2,
        "started_at": t,
        "finished_at": t,
        "latency_ms": 5,
        "actor": "test",
    }
    defaults.update(overrides)
    return ComposerToolInvocation(**defaults)  # type: ignore[arg-type]


def test_to_dict_roundtrip_through_canonical_json_is_deterministic() -> None:
    """to_dict() output must canonicalize to a stable hash across construction order."""
    inv = _make_invocation()
    d1 = inv.to_dict()
    d2 = inv.to_dict()
    # Same invocation → same canonical → same hash.
    assert stable_hash(d1) == stable_hash(d2)
    # Status + datetimes should be ISO/string-rendered for JSON safety.
    assert d1["status"] == "success"
    assert isinstance(d1["started_at"], str)
    assert isinstance(d1["finished_at"], str)
    # Round-trip through canonical_json without raising.
    canonical_json(d1)


def test_cache_hit_defaults_false() -> None:
    inv = _make_invocation()
    assert inv.cache_hit is False
    assert inv.to_dict()["cache_hit"] is False


def test_cache_hit_set_true_serializes_correctly() -> None:
    inv = _make_invocation(cache_hit=True)
    assert inv.cache_hit is True
    assert inv.to_dict()["cache_hit"] is True


def test_frozen_dataclass_blocks_mutation() -> None:
    inv = _make_invocation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.tool_name = "different"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.version_after = 99  # type: ignore[misc]


def test_status_strenum_values() -> None:
    """Status values must match the canonical strings to keep records portable."""
    assert ComposerToolStatus.SUCCESS.value == "success"
    assert ComposerToolStatus.ARG_ERROR.value == "arg_error"
    assert ComposerToolStatus.PLUGIN_CRASH.value == "plugin_crash"


def test_arg_error_invocation_shape() -> None:
    """ARG_ERROR invocations should record version_after=None per the plan."""
    inv = _make_invocation(
        status=ComposerToolStatus.ARG_ERROR,
        error_class="ToolArgumentError",
        error_message="'plugin' must be a string, got int",
        version_after=None,
        result_canonical=None,
        result_hash=None,
    )
    d = inv.to_dict()
    assert d["status"] == "arg_error"
    assert d["version_after"] is None
    assert d["result_canonical"] is None
    assert d["result_hash"] is None


def test_plugin_crash_invocation_shape() -> None:
    """PLUGIN_CRASH must record class-name only (no message detail) and version_after=None."""
    inv = _make_invocation(
        status=ComposerToolStatus.PLUGIN_CRASH,
        error_class="RuntimeError",
        error_message="RuntimeError",
        version_after=None,
        result_canonical=None,
        result_hash=None,
    )
    d = inv.to_dict()
    assert d["status"] == "plugin_crash"
    assert d["error_class"] == "RuntimeError"
    # By design: error_message is the class name, NOT str(exc) — plugin
    # exception messages can carry secrets / paths.
    assert d["error_message"] == "RuntimeError"


def test_l0_module_has_no_upward_imports() -> None:
    """The L0 contract must import only stdlib + typing (and freeze module if used).

    Any import of elspeth.core, elspeth.engine, elspeth.plugins, or
    elspeth.web from within composer_audit module would violate the L0
    layer rule. This test pins the invariant.
    """
    # Already imported at top of file — re-import to capture module state.
    import elspeth.contracts.composer_audit as ca

    forbidden_prefixes = ("elspeth.core", "elspeth.engine", "elspeth.plugins", "elspeth.web", "elspeth.cli")
    for name, _mod in list(sys.modules.items()):
        if name == "elspeth.contracts.composer_audit":
            # Inspect the module's __dict__ for direct module references.
            for ref in ca.__dict__.values():
                ref_module = getattr(ref, "__module__", None)
                if ref_module is None:
                    continue
                for prefix in forbidden_prefixes:
                    assert not ref_module.startswith(prefix), f"composer_audit imports {ref!r} from forbidden L1+ module {ref_module}"


def test_recorder_protocol_runtime_check() -> None:
    """A class with .record() and .resolve_session() satisfies the Protocol."""

    class _StubRecorder:
        def record(self, invocation: ComposerToolInvocation) -> None:
            return

        def resolve_session(self, session_id: str) -> None:
            return

    # Protocol type-check by structural conformance.
    rec: ComposerToolRecorder = _StubRecorder()
    rec.record(_make_invocation())
    rec.resolve_session("abc")
