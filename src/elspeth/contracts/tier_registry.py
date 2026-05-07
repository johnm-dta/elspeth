"""Tier-1 exception registry (ADR-010 §Decision 2).

Replaces the hand-maintained TIER_1_ERRORS tuple. Three safety mechanisms
beyond the v0 design (see plan Revision History — Review v1):

1. ``@tier_1_error(reason=...)`` is a factory — it requires a justification
   string. Auditors reviewing Tier-1 additions can grep for the reason.
2. Module-prefix allowlist: only callers in ``elspeth.contracts.*``,
   ``elspeth.engine.*``, or ``elspeth.core.*`` may register during normal
   runtime. Under pytest, ``tests.*`` is additionally allowed for test-only
   fixtures. Plugin modules cannot elevate their own exceptions
   (reviewer B6/F-6).
3. ``freeze_tier_registry()`` is called at end of bootstrap. Registration
   after freeze raises ``FrameworkBugError`` (reviewer B5/F-2).

``FrameworkBugError`` lives in this module so ``errors.py`` can expose the
same public exception without creating a circular import.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import TypeVar

from elspeth.contracts.registry_primitive import FrozenRegistry


# Intentional forward-declaration: FrameworkBugError is defined here first so
# ``errors.py`` can apply ``@tier_1_error`` to it without circular import.
class FrameworkBugError(Exception):
    """Raised when the framework encounters an internal inconsistency.

    This indicates a bug in ELSPETH itself, not user error or external failure.
    Unlike OrchestrationInvariantError (specific to orchestration flow), this
    is a general-purpose exception for any framework-level bug.

    Examples of conditions that trigger this:
    - Double-completing an operation (already completed, trying to complete again)
    - Missing required context (record_call with neither state_id nor operation_id)
    - Completing a non-existent operation

    Recovery: These errors indicate bugs in framework code that must be fixed.
    They should never occur in correct operation.

    Defined here so ``errors.py`` can expose the same public exception
    without introducing a circular import.
    """


_FROZEN: bool = False


def _get_frozen_flag() -> bool:
    return _FROZEN


def _set_frozen_flag(value: bool) -> None:
    global _FROZEN
    _FROZEN = value


_TIER_REGISTRY: FrozenRegistry[type[BaseException], dict[type[BaseException], str]] = FrozenRegistry(
    name="TIER_1_ERRORS",
    auxiliary={},
    frozen_getter=_get_frozen_flag,
    frozen_setter=_set_frozen_flag,
)
_REGISTRY = _TIER_REGISTRY.items
_REASONS = _TIER_REGISTRY.auxiliary
_REGISTRY_LOCK = _TIER_REGISTRY.lock

_ALLOWED_MODULE_PREFIXES: tuple[str, ...] = (
    "elspeth.contracts.",
    "elspeth.engine.",
    "elspeth.core.",
    *(("tests.",) if "pytest" in sys.modules else ()),
)

_ExcT = TypeVar("_ExcT", bound=BaseException)


def _module_is_allowed(module_name: str) -> bool:
    """Return True when ``module_name`` is allowed to own Tier-1 classes."""
    return any(module_name.startswith(p) or module_name == p.rstrip(".") for p in _ALLOWED_MODULE_PREFIXES)


def tier_1_error(_cls: type | None = None, *, reason: str, caller_module: str):  # type: ignore[no-untyped-def]
    """Factory returning a decorator that registers ``cls`` as Tier-1.

    Usage (the ``caller_module=__name__`` literal is load-bearing — see below):
        @tier_1_error(
            reason="ADR-008: annotation lie corrupts audit trail",
            caller_module=__name__,
        )
        class AuditIntegrityError(Exception): ...

    Args:
        reason: Non-empty justification string. Recorded in the registry
            and queryable via ``tier_1_reason(cls)``.
        caller_module: The calling module's ``__name__``. Every call site
            MUST pass the literal ``__name__`` here — the CI scanner
            ``scripts/cicd/enforce_tier_1_decoration.py`` (rule TDE2) rejects
            any non-literal value (variable, f-string, attribute, etc.)
            because the value is the primary input to the module-prefix
            allowlist check (``elspeth.contracts.*`` / ``elspeth.engine.*``
            / ``elspeth.core.*``). ADR-010 §Decision 2 M8 replaced the
            prior ``inspect.stack()`` scheme because
            frame-offset introspection breaks silently if any decorator
            or metaclass wrapping is added between ``@tier_1_error`` and
            the class — a future contributor's innocent-looking wrapper
            could thereby let a plugin-module class elevate itself to
            Tier-1. The explicit kwarg makes the allowlist check
            grep-auditable and wrapper-proof.

    Raises:
        TypeError: applied without ``reason`` (positional-class usage like
            ``@tier_1_error`` instead of ``@tier_1_error(reason=..., caller_module=...)``).
        ValueError: ``reason`` is empty.
        PermissionError: ``caller_module`` is outside the allowlist.
        FrameworkBugError: registry is frozen.
    """
    if _cls is not None:
        raise TypeError(
            "@tier_1_error requires reason and caller_module kwargs — use @tier_1_error(reason=..., caller_module=__name__) not @tier_1_error"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("@tier_1_error(reason=...) requires non-empty reason string")
    if not isinstance(caller_module, str) or not caller_module.strip():
        raise ValueError("@tier_1_error(caller_module=...) requires non-empty string; pass caller_module=__name__ literally")

    def _decorator(cls: type[_ExcT]) -> type[_ExcT]:
        return _register_with_module_prefix(cls=cls, reason=reason, caller_module=caller_module)

    return _decorator


def _register_with_module_prefix[ExcT: BaseException](*, cls: type[ExcT], reason: str, caller_module: str) -> type[ExcT]:
    def _frozen_error() -> FrameworkBugError:
        return FrameworkBugError(
            f"Cannot register {cls.__name__!r}: TIER_1_ERRORS registry is frozen. "
            f"Bootstrap is complete; new Tier-1 classes must be added before plugin discovery finishes."
        )

    with _TIER_REGISTRY.write_unfrozen(_frozen_error):
        if not (isinstance(cls, type) and issubclass(cls, BaseException)):
            raise TypeError(f"@tier_1_error applied to {cls!r} — must be a BaseException subclass")
        if not _module_is_allowed(caller_module):
            raise PermissionError(
                f"@tier_1_error used from {caller_module!r}; only allowed from "
                f"{_ALLOWED_MODULE_PREFIXES!r}. Plugin modules cannot elevate their "
                f"own exceptions to Tier-1 — request ADR review instead."
            )
        class_module = cls.__module__
        if not isinstance(class_module, str) or not class_module.strip():
            raise TypeError(
                f"@tier_1_error applied to {cls!r} with invalid __module__={class_module!r}; "
                "Tier-1 registration requires a non-empty module name."
            )
        if not _module_is_allowed(class_module):
            raise PermissionError(
                f"@tier_1_error cannot register {cls.__name__!r} from {class_module!r}; only "
                f"{_ALLOWED_MODULE_PREFIXES!r} may own Tier-1 exceptions. Plugin modules "
                "cannot elevate their own exceptions to Tier-1 — request ADR review instead."
            )
        if cls in _REGISTRY:
            if _REASONS[cls] != reason:
                raise ValueError(
                    f"{cls.__name__} already registered with reason {_REASONS[cls]!r}; double-registration with conflicting reason {reason!r}"
                )
            return cls
        _REGISTRY.append(cls)
        _REASONS[cls] = reason
    return cls


def tier_1_reason(cls: type[BaseException]) -> str:
    """Return the registered reason for ``cls``, or raise KeyError."""
    with _TIER_REGISTRY.read():
        return _REASONS[cls]


def freeze_tier_registry() -> None:
    """Seal the registry. Subsequent registrations raise ``FrameworkBugError``.

    Called at the end of orchestrator bootstrap after runtime-VAL registration
    has been validated.
    """
    _TIER_REGISTRY.freeze()


def tier_registry_is_frozen() -> bool:
    """Return whether the Tier-1 registry has been sealed by bootstrap."""
    return _TIER_REGISTRY.is_frozen()


class _Tier1ErrorsView:
    """Live view of the Tier-1 error registry.

    Live view over ``_REGISTRY``. Unlike a snapshot tuple, membership tests
    (``__contains__``), iteration, and ``count`` always see the current registry
    contents. Not a drop-in tuple replacement: ``len()`` is supported for test
    assertions, but indexing and use in ``except`` clauses remain unsupported.
    Use the ``errors.TIER_1_ERRORS`` module attribute when tuple semantics are
    required.
    """

    def __contains__(self, item: object) -> bool:
        with _TIER_REGISTRY.read():
            return item in _REGISTRY

    def __iter__(self) -> Iterator[type[BaseException]]:
        with _TIER_REGISTRY.read():
            return iter(list(_REGISTRY))

    def __len__(self) -> int:
        with _TIER_REGISTRY.read():
            return len(_REGISTRY)

    def __repr__(self) -> str:
        with _TIER_REGISTRY.read():
            names = [cls.__name__ for cls in _REGISTRY]
        return f"TIER_1_ERRORS({names})"

    def count(self, item: object) -> int:
        with _TIER_REGISTRY.read():
            return _REGISTRY.count(item)  # type: ignore[arg-type]  # list[type[BaseException]].count(object) is safe


_TIER_1_ERRORS_VIEW = _Tier1ErrorsView()
