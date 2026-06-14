"""Pin the ADR-030 follower teardown contract (H2 remediation).

Before the H2 fix, ``cli.py``'s ``join`` (the ``elspeth join`` follower entry)
tore plugins down in a bare ``try/except Exception: pass`` finally block. That
silently swallowed TIER-1 audit-integrity errors (``AuditIntegrityError`` /
``FrameworkBugError``) raised during ``on_complete`` / ``close`` — a permanent
hole hiding system-level corruption on the follower path (which had no
lifecycle at all before the multi-worker change).

The fix routes follower teardown through the canonical ``cleanup_plugins(...,
include_source=False)`` helper, whose contract is:

  - TIER-1 errors PROPAGATE (re-raised before the broad catch, cleanup.py:95-96);
  - non-Tier-1 errors keep best-effort continuation (every OTHER plugin still
    tears down) AND each is LOGGED with plugin name + hook phase;
  - ``include_source=False`` because the follower is drain-only — it never
    on_start'd the sources.

Coverage split (read before editing):
  - ``test_follower_teardown_*`` pin the HELPER contract directly — the exact
    call ``cli.join`` now makes (``cleanup_plugins(..., include_source=False)``).
    They assert WHAT the follower path delegates to. They do NOT, on their own,
    fail under the pre-fix bare ``except Exception: pass`` (they bypass cli.join).
  - ``test_cli_follower_join_wires_canonical_cleanup_not_bare_swallow`` is the
    REGRESSION pin for the cli.py defect: it inspects ``cli.join``'s source and
    asserts the canonical helper is CALLED in the teardown ``finally`` AND that no
    bare swallow remains. This test FAILS under the pre-fix code.

Together they pin both halves: the helper does the right thing, and cli.join
delegates to it (rather than swallowing).
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import structlog.testing

from elspeth.contracts import Determinism
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.engine.orchestrator import PipelineConfig
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform


class _Tier1OnCompleteTransform(PassTransform):
    """Transform whose on_complete raises a TIER-1 audit-integrity error."""

    determinism = Determinism.DETERMINISTIC

    def on_complete(self, ctx: Any) -> None:
        raise AuditIntegrityError("follower teardown found a corrupt journal row")


class _Tier1CloseTransform(PassTransform):
    """Transform whose close() raises a TIER-1 audit-integrity error."""

    determinism = Determinism.DETERMINISTIC

    def close(self) -> None:
        raise AuditIntegrityError("follower close found a corrupt journal row")


class _RecoverableOnCompleteTransform(PassTransform):
    """Transform whose on_complete raises a NON-Tier-1 (recoverable) error."""

    determinism = Determinism.DETERMINISTIC

    def on_complete(self, ctx: Any) -> None:
        raise RuntimeError("follower transform on_complete blew up (recoverable)")


class _CloseRecordingSink(CollectSink):
    """Sink that records whether close() ran (to prove best-effort continuation)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _follower_config(*transforms: Any, sink: Any) -> PipelineConfig:
    """Mirror the follower teardown call: transforms + sinks, NO source on_start."""
    return PipelineConfig(
        sources={"primary": as_source(ListSource([], name="src"))},
        transforms=[as_transform(t) for t in transforms],
        sinks={sink.name: as_sink(sink)},
    )


def test_follower_teardown_propagates_tier1_from_on_complete() -> None:
    """A TIER-1 error in a follower transform's on_complete PROPAGATES."""
    ctx = PluginContext(run_id="run-follower", config={}, landscape=None)
    config = _follower_config(_Tier1OnCompleteTransform(), sink=_CloseRecordingSink("output"))

    with pytest.raises(AuditIntegrityError, match="corrupt journal row"):
        cleanup_plugins(config, ctx, include_source=False)


def test_follower_teardown_propagates_tier1_from_close() -> None:
    """A TIER-1 error in a follower transform's close() PROPAGATES."""
    ctx = PluginContext(run_id="run-follower", config={}, landscape=None)
    config = _follower_config(_Tier1CloseTransform(), sink=_CloseRecordingSink("output"))

    with pytest.raises(AuditIntegrityError, match="corrupt journal row"):
        cleanup_plugins(config, ctx, include_source=False)


def test_follower_teardown_logs_and_continues_on_non_tier1() -> None:
    """A non-Tier-1 error is LOGGED with plugin name + phase AND other plugins
    still tear down (best-effort continuation).

    No in-flight exception is propagating, so cleanup_plugins raises a
    RuntimeError summarising the swallowed failure — but only AFTER every other
    hook ran. We assert: (1) the structured per-hook log names the plugin + hook;
    (2) the sink's close() still ran; (3) the aggregate RuntimeError is raised.
    """
    ctx = PluginContext(run_id="run-follower", config={}, landscape=None)
    sink = _CloseRecordingSink("output")
    bad = _RecoverableOnCompleteTransform(name="recoverable_transform")
    config = _follower_config(bad, sink=sink)

    with structlog.testing.capture_logs() as captured, pytest.raises(RuntimeError, match="Plugin cleanup failed"):
        cleanup_plugins(config, ctx, include_source=False)

    # Best-effort continuation: the sink still got torn down.
    assert sink.closed is True, "non-Tier-1 transform failure must not block the sink's close()"

    # The failure was LOGGED with plugin name + hook phase.
    hook_failures = [e for e in captured if e["event"] == "Plugin cleanup hook failed"]
    assert hook_failures, "the swallowed non-Tier-1 failure must be logged, not silently dropped"
    failure = hook_failures[0]
    assert failure["plugin"] == "recoverable_transform"
    assert failure["hook"] == "transform.on_complete"
    assert failure["error_type"] == "RuntimeError"


def test_follower_teardown_skips_source_lifecycle() -> None:
    """include_source=False: the follower never on_start'd the source, so its
    on_complete/close must NOT be called during teardown."""
    ctx = PluginContext(run_id="run-follower", config={}, landscape=None)

    class _SourceCloseSentinel(ListSource):
        def __init__(self) -> None:
            super().__init__([], name="src")
            self.closed = False

        def close(self) -> None:
            self.closed = True

    source = _SourceCloseSentinel()
    sink = _CloseRecordingSink("output")
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(PassTransform(name="pt"))],
        sinks={sink.name: as_sink(sink)},
    )

    cleanup_plugins(config, ctx, include_source=False)

    assert source.closed is False, "follower teardown must not close the source it never opened"
    assert sink.closed is True, "the sink IS torn down on the follower path"


def test_cli_follower_join_wires_canonical_cleanup_not_bare_swallow() -> None:
    """Source-level REGRESSION pin: the follower ``join`` path CALLS the canonical
    ``cleanup_plugins(...)`` IN ITS TEARDOWN ``finally`` BLOCK and no longer uses
    the bare ``except Exception: pass`` swallow that hid Tier-1 errors.

    This test FAILS under the pre-fix code (bare swallow, no cleanup_plugins call).
    We AST-walk the finally handlers so the call must genuinely live in a
    ``try/finally`` teardown (not a comment, not an unrelated call site), and
    assert no bare ``except Exception: pass`` remains anywhere in ``join``.
    """
    import ast
    import re
    import textwrap

    from elspeth import cli

    src = inspect.getsource(cli.join)
    # An actual call to cleanup_plugins (the canonical helper), include_source=False.
    assert re.search(r"cleanup_plugins\s*\(", src), "follower teardown must CALL the canonical cleanup_plugins(...)"
    assert "include_source=False" in src, "follower teardown must call cleanup_plugins with include_source=False (drain-only worker)"

    # AST: a cleanup_plugins(...) call must appear inside a try/finally's finalbody.
    tree = ast.parse(textwrap.dedent(src))

    def _calls_cleanup(nodes: list[ast.stmt]) -> bool:
        for stmt in nodes:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Call):
                    fn = sub.func
                    name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
                    if "cleanup_plugins" in name:
                        return True
        return False

    finally_has_cleanup = any(isinstance(node, ast.Try) and node.finalbody and _calls_cleanup(node.finalbody) for node in ast.walk(tree))
    assert finally_has_cleanup, "cleanup_plugins must be called inside a try/finally teardown block in cli.join"

    # The bare swallow this fix replaced must be gone from the follower finally
    # (any indentation of `except Exception:` immediately followed by a bare pass).
    assert not re.search(r"except Exception:\s*\n\s*pass\b", src), "the bare Tier-1-swallowing teardown must be removed"
