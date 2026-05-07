"""@tier_1_error(reason=...) decorator + registry tests (ADR-010 §Decision 2)."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.usefixtures("reset_tier_registry")


def test_decorator_registers_class() -> None:
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_error

    @tier_1_error(reason="test: registered via decorator", caller_module=__name__)
    class _TestViolation(Exception):
        pass

    assert _TestViolation in _TIER_1_ERRORS_VIEW


def test_decorator_requires_reason() -> None:
    from elspeth.contracts.tier_registry import tier_1_error

    malformed_decorator: Any = tier_1_error
    with pytest.raises(TypeError, match="reason"):
        malformed_decorator(type("_Bad", (Exception,), {}))


def test_reason_must_be_non_empty_string() -> None:
    from elspeth.contracts.tier_registry import tier_1_error

    with pytest.raises(ValueError, match="reason"):

        @tier_1_error(reason="", caller_module=__name__)
        class _Bad(Exception):
            pass


def test_reason_persists_and_is_queryable() -> None:
    from elspeth.contracts.tier_registry import tier_1_error, tier_1_reason

    @tier_1_error(reason="ADR-008: pass-through annotation lie", caller_module=__name__)
    class _Check(Exception):
        pass

    assert tier_1_reason(_Check) == "ADR-008: pass-through annotation lie"


def test_decorator_returns_the_class_unchanged() -> None:
    from elspeth.contracts.tier_registry import tier_1_error

    @tier_1_error(reason="test", caller_module=__name__)
    class _Foo(Exception):
        pass

    assert _Foo.__name__ == "_Foo"


def test_double_registration_idempotent_with_matching_reason() -> None:
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_error

    decorator = tier_1_error(reason="first", caller_module=__name__)

    @decorator
    class _Twice(Exception):
        pass

    decorator(_Twice)
    assert _TIER_1_ERRORS_VIEW.count(_Twice) == 1


def test_non_exception_class_raises_typeerror() -> None:
    from elspeth.contracts.tier_registry import tier_1_error

    with pytest.raises(TypeError, match="BaseException"):

        @tier_1_error(reason="test", caller_module=__name__)
        class _NotException:
            pass


def test_module_outside_allowlist_raises() -> None:
    """Decoration from a plugin module must fail."""
    from elspeth.contracts.tier_registry import _register_with_module_prefix

    # Simulate a plugin-module caller by passing an explicit module name.
    with pytest.raises(PermissionError, match="plugin"):
        _register_with_module_prefix(
            cls=type("X", (Exception,), {}),
            reason="plugin attempt",
            caller_module="elspeth.plugins.transforms.custom",
        )


def test_plugin_owned_class_cannot_spoof_allowed_caller_module() -> None:
    """The registry must bind Tier-1 ownership to the class's real module."""
    from elspeth.contracts.tier_registry import _register_with_module_prefix

    plugin_error = type(
        "PluginDefinedError",
        (Exception,),
        {"__module__": "elspeth.plugins.transforms.evil"},
    )

    with pytest.raises(PermissionError, match="Plugin modules cannot elevate"):
        _register_with_module_prefix(
            cls=plugin_error,
            reason="plugin spoof attempt",
            caller_module="elspeth.core.fake",
        )


def test_registration_after_freeze_raises() -> None:
    from elspeth.contracts.tier_registry import (
        FrameworkBugError,
        freeze_tier_registry,
        tier_1_error,
    )

    freeze_tier_registry()
    with pytest.raises(FrameworkBugError, match="frozen"):

        @tier_1_error(reason="too late", caller_module=__name__)
        class _Late(Exception):
            pass


def test_double_registration_with_conflicting_reason_raises() -> None:
    """Same class registered twice with different reasons must fail loudly."""
    from elspeth.contracts.tier_registry import _register_with_module_prefix

    class _Dup(Exception):
        pass

    _register_with_module_prefix(
        cls=_Dup,
        reason="first registration",
        caller_module="elspeth.contracts.test_fixture",
    )
    with pytest.raises(ValueError, match="already registered"):
        _register_with_module_prefix(
            cls=_Dup,
            reason="second registration with different reason",
            caller_module="elspeth.contracts.test_fixture",
        )


def test_tier_1_errors_view_supports_len() -> None:
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_error

    before = len(_TIER_1_ERRORS_VIEW)

    @tier_1_error(reason="len test", caller_module=__name__)
    class _Counted(Exception):
        pass

    assert len(_TIER_1_ERRORS_VIEW) == before + 1


def test_tier_1_errors_view_repr_shows_names() -> None:
    from elspeth.contracts.tier_registry import _TIER_1_ERRORS_VIEW, tier_1_error

    @tier_1_error(reason="repr test", caller_module=__name__)
    class _Repr(Exception):
        pass

    assert "_Repr" in repr(_TIER_1_ERRORS_VIEW)
    assert repr(_TIER_1_ERRORS_VIEW).startswith("TIER_1_ERRORS(")


def test_caller_module_kwarg_is_required() -> None:
    """M8 (issue elspeth-3af772b9e3): caller_module must be supplied explicitly.

    The prior inspect.stack()-based implementation read the caller's
    __name__ from frame 2 of the call stack, which made the allowlist
    check fragile to any intervening wrapper (decorator stacking,
    metaclass, functools.wraps) — a future contributor adding an
    innocent wrapper could let a plugin-module class elevate itself to
    Tier-1 silently. The fix: require caller_module=__name__ explicitly
    at every call site. Calling without caller_module must raise.
    """
    from elspeth.contracts.tier_registry import tier_1_error

    missing_caller_module: Any = tier_1_error
    with pytest.raises(TypeError, match="caller_module"):
        missing_caller_module(reason="missing caller_module")


def test_caller_module_empty_string_rejected() -> None:
    """Empty caller_module strings are rejected — the allowlist gate
    would otherwise pass the empty-startswith check and allow the registration.
    """
    from elspeth.contracts.tier_registry import tier_1_error

    with pytest.raises(ValueError, match="caller_module"):
        tier_1_error(reason="valid", caller_module="")


def test_tests_prefix_absent_when_pytest_not_loaded() -> None:
    """Smoke test: in a pytest-free subprocess, tests. is absent."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "assert 'pytest' not in sys.modules\n"
            "from elspeth.contracts.tier_registry import _ALLOWED_MODULE_PREFIXES\n"
            "assert 'tests.' not in _ALLOWED_MODULE_PREFIXES, f'tests. leaked into production: {_ALLOWED_MODULE_PREFIXES}'\n"
            "print('OK')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert "OK" in result.stdout


def test_freeze_tier_registry_uses_registry_lock() -> None:
    """freeze_tier_registry must serialize with concurrent registry mutation."""
    from threading import Event, Thread

    from elspeth.contracts import tier_registry

    started = Event()
    finished = Event()
    errors: list[BaseException] = []

    def freeze_registry() -> None:
        started.set()
        try:
            tier_registry.freeze_tier_registry()
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)
        finally:
            finished.set()

    tier_registry._REGISTRY_LOCK.acquire()
    try:
        thread = Thread(target=freeze_registry)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05), "freeze completed while the registry lock was held"
    finally:
        tier_registry._REGISTRY_LOCK.release()

    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    assert tier_registry.tier_registry_is_frozen()


def test_tier_1_registration_uses_registry_lock() -> None:
    """Tier-1 registration must not race a concurrent freeze/registration path."""
    from threading import Event, Thread

    from elspeth.contracts import tier_registry

    started = Event()
    finished = Event()
    errors: list[BaseException] = []

    class _ThreadRegistered(Exception):
        pass

    def register_error() -> None:
        started.set()
        try:
            tier_registry._register_with_module_prefix(
                cls=_ThreadRegistered,
                reason="registered while lock-protected",
                caller_module=__name__,
            )
        except BaseException as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)
        finally:
            finished.set()

    tier_registry._REGISTRY_LOCK.acquire()
    try:
        thread = Thread(target=register_error)
        thread.start()
        assert started.wait(timeout=1.0)
        assert not finished.wait(timeout=0.05), "registration completed while the registry lock was held"
    finally:
        tier_registry._REGISTRY_LOCK.release()

    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert errors == []
    assert _ThreadRegistered in tier_registry._TIER_1_ERRORS_VIEW
